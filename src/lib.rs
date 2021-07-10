#![feature(test)]
#![feature(destructuring_assignment)]
#![allow(non_snake_case)]
#![allow(clippy::many_single_char_names)]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate ndarray_stats;
extern crate statrs;

use levenberg_marquardt::LevenbergMarquardt;
use ndarray::array;
use ndarray::concatenate;
use ndarray::par_azip;
use ndarray::s;
use ndarray::Array;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::Slice;
use ndarray::Zip;
use ndarray::{Array1, ArrayView1};
use ndarray::{Ix1, Ix2};
use ndarray_linalg::cholesky::Cholesky;
use ndarray_linalg::eigh::*;
use ndarray_linalg::solveh::UPLO;
use ndarray_rand::rand_distr::num_traits::FloatConst;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use rand::distributions::Uniform;
use statrs::function::erf::erfc_inv;

mod faddeeva;
mod optimization;
mod util;

pub struct MultivariateNormal {
    loc: Array1<f64>,
    scale: Array2<f64>,
}

impl MultivariateNormal {
    pub fn new(loc: Array1<f64>, scale: Array2<f64>) -> Self {
        MultivariateNormal { loc, scale }
    }

    pub fn sample(&self) -> Array1<f64> {
        let sample_size = self.loc.len();
        let lower = self.scale.cholesky(UPLO::Lower).unwrap();
        let mut rng = rand::thread_rng();
        let ind_sample = Array::random_using(sample_size, StandardNormal, &mut rng);
        &self.loc + lower.dot(&ind_sample)
    }

    // Source: <http://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/>
    pub fn logp(&self, x: &Array1<f64>) -> f64 {
        let (eig_vals, eig_vecs) = self.scale.eigh(UPLO::Lower).unwrap();
        let logdet = eig_vals.mapv(|x| x.ln()).sum();
        let inv_vals = eig_vals.mapv(|x| 1. / x);
        let U = eig_vecs * inv_vals.mapv(|x| x.sqrt());
        let dev = x - &self.loc;
        let maha_dist = dev.dot(&U).mapv(|x| x.powi(2)).sum();
        -0.5 * ((eig_vals.len() as f64) * f64::TAU().ln() + maha_dist + logdet)
    }
}

pub struct MultivariateTruncNormal<D> {
    loc: Array1<f64>,
    scale: Array<f64, D>,
    low: Array1<f64>,
    high: Array1<f64>,
    log_normalizer: Array1<f64>,
}

impl MultivariateTruncNormal<Ix1> {
    pub fn new(loc: Array1<f64>, scale: Array1<f64>, low: Array1<f64>, high: Array1<f64>) -> Self {
        assert_eq!(loc.shape(), scale.shape());
        assert_eq!(loc.shape(), low.shape());
        assert_eq!(loc.shape(), high.shape());
        let log_normalizer = ln_normal_pr(&((&low - &loc) / &scale), &((&high - &loc) / &scale));
        MultivariateTruncNormal {
            loc,
            scale,
            low,
            high,
            log_normalizer,
        }
    }

    pub fn log_prob(&self, x: ArrayView1<f64>) -> f64 {
        let std_x = (&x - &self.loc) / &self.scale;
        let halfrtln2pi = 0.5 * (f64::TAU()).ln();
        Zip::from(&std_x)
            .and(&self.scale)
            .and(&self.log_normalizer)
            .par_map_collect(|&x, &s, &lnz| -(0.5 * x.powi(2) + halfrtln2pi + s.ln() - lnz))
            .sum()
    }

    pub fn sample(&self, max_iters: usize) -> (Array1<f64>, f64) {
        let X = trandn(
            &((&self.low - &self.loc) / &self.scale),
            &((&self.high - &self.loc) / &self.scale),
            max_iters,
        );
        let Z = &self.loc + &self.scale * X;
        let logp = self.log_prob(Z.view());
        (Z, logp)
    }
}

impl MultivariateTruncNormal<Ix2> {
    pub fn new(loc: Array1<f64>, scale: Array2<f64>, low: Array1<f64>, high: Array1<f64>) -> Self {
        //let log_normalizer = ln_normal_pr(&((&high - &loc) / &scale), &((&low - &loc) / &scale));
        MultivariateTruncNormal {
            loc,
            scale,
            low,
            high,
            log_normalizer: array![0.],
        }
    }

    pub fn sample(&self, n: usize, max_iters: usize) -> (Array2<f64>, Array1<f64>) {
        mv_truncnormal_rand(
            self.low.clone(),
            self.high.clone(),
            self.scale.clone(),
            n,
            max_iters,
        )
    }
}

fn ln_phi_f64(x: f64) -> f64 {
    // computes logarithm of tail of Z~N(0,1) mitigating numerical roundoff errors
    -0.5 * x.powi(2) - f64::LN_2() + faddeeva::erfcx(x * f64::FRAC_1_SQRT_2()).ln()
}

fn ln_normal_pr<D: ndarray::Dimension>(a: &Array<f64, D>, b: &Array<f64, D>) -> Array<f64, D> {
    // computes ln(Pr(a<Z<b)) where Z~N(0,1) very accurately for any 'a', 'b'
    Zip::from(a).and(b).par_map_collect(|&a, &b| {
        if a > 0.0 {
            let pa = ln_phi_f64(a);
            let pb = ln_phi_f64(b);
            pa + (-1. * (pb - pa).exp()).ln_1p()
        } else if b < 0.0 {
            let pa = ln_phi_f64(-a);
            let pb = ln_phi_f64(-b);
            pb + (-1. * (pa - pb).exp()).ln_1p()
        } else {
            let pa = faddeeva::erfc(-1. * a * f64::FRAC_1_SQRT_2()) / 2.;
            let pb = faddeeva::erfc(b * f64::FRAC_1_SQRT_2()) / 2.;
            (-pa - pb).ln_1p()
        }
    })
}

fn cholperm(
    sigma: &mut Array2<f64>,
    l: &mut Array1<f64>,
    u: &mut Array1<f64>,
) -> (Array2<f64>, Array1<usize>) {
    let d = l.shape()[0];
    let mut L: Array2<f64> = Array2::zeros((d, d));
    let mut z: Array1<f64> = Array1::zeros(d);
    let mut perm = Array::range(0., d as f64, 1.);

    for j in 0..d {
        let mut pr = Array1::from_elem(d, f64::INFINITY);
        let diag = sigma.diag();
        let L_chunk = L.slice(s![j..d, 0..j]);

        let subtr = L_chunk.dot(&z.slice(s![0..j]));
        let s = (&diag.slice_axis(Axis(0), Slice::from(j..d))
            - L_chunk.mapv(|x| x.powi(2)).sum_axis(Axis(1)))
        .map(|&x: &f64| if x > 0. { x.sqrt() } else { f64::EPSILON });
        let tl = (&l.slice_axis(Axis(0), Slice::from(j..d)) - &subtr) / &s;
        let tu = (&u.slice_axis(Axis(0), Slice::from(j..d)) - subtr) / s;

        pr.slice_mut(s![j..d]).assign(&ln_normal_pr(&tl, &tu));
        let k = pr.argmin().unwrap();
        // update rows and cols of sigma
        util::swap_rows(sigma, j, k);
        util::swap_cols(sigma, j, k);
        // update only rows of L
        util::swap_rows(&mut L, j, k);
        // update integration limits
        l.swap(j, k);
        u.swap(j, k);
        perm.swap(j, k); // keep track of permutation

        // construct L sequentially via Cholesky computation
        let mut s_scalar = sigma[[j, j]] - L.slice(s![j, 0..j]).mapv(|x| x.powi(2)).sum();
        // if s_scalar < -0.01, sigma isn't pos semi-def
        if s_scalar <= 0. {
            s_scalar = f64::EPSILON;
        }
        s_scalar = s_scalar.sqrt();
        L[[j, j]] = s_scalar;

        {
            let sub_term = L.slice(s![j + 1..d, 0..j]).dot(&L.slice(s![j, 0..j]).t());
            L.slice_mut(s![(j + 1)..d, j])
                .assign(&(&(&sigma.slice(s![(j + 1)..d, j]) - sub_term) / s_scalar));
        }

        // find mean value, z(j), of truncated normal
        let sub_term = L.slice(s![j, ..j]).dot(&z.slice(s![..j]));
        let tl_s = (l[[j]] - sub_term) / s_scalar;
        let tu_s = (u[[j]] - sub_term) / s_scalar;
        let w = ln_normal_pr(&array![tl_s], &array![tu_s])[[0]]; // aids in computing expected value of trunc. normal
        z[[j]] = ((-0.5 * tl_s.powi(2) - w).exp() - (-0.5 * tu_s.powi(2) - w).exp())
            / (f64::TAU()).sqrt();
    }
    (L, perm.mapv(|x| x as usize))
}

fn ntail(l: Array1<f64>, u: Array1<f64>, max_iters: usize) -> Array1<f64> {
    /*
    % samples a vector from the standard normal
    % distribution truncated over the region [l,u], where l>0 and
    % l and u are column vectors; uses acceptance-rejection from Rayleigh
    % distr. Similar to Marsaglia (1964)
    */
    let c = l.map(|x| x.powi(2) / 2.);
    let f = Zip::from(&c)
        .and(&u)
        .par_map_collect(|&c, &u| (c - u.powi(2) / 2.).exp_m1());
    // use rejection sample pattern
    let accept_condition = |x: &Array1<f64>, accepted: &mut Array1<bool>| {
        let test_sample: Array1<f64> = Array1::random(l.len(), Uniform::new(0., 1.));
        par_azip!((x in x, &s in &test_sample, &c in &c, acc in accepted) {
                if s.powi(2) * x < c {
                    *acc = true;
                }
        })
    };
    let proposal_sampler = || {
        let sample = Array1::random(l.len(), Uniform::new(0., 1.));
        &c - (1. + sample * &f).mapv(|x| x.ln())
    };
    let mut output_array = util::rejection_sample(&accept_condition, &proposal_sampler, max_iters);
    output_array.mapv_inplace(|x| (2. * x).sqrt());
    output_array
}

fn trnd(l: &Array1<f64>, u: &Array1<f64>, max_iters: usize) -> Array1<f64> {
    // use accept-reject pattern to sample from truncated N(0,1)
    let accept_condition = |x: &Array1<f64>, accepted: &mut Array1<bool>| {
        par_azip!((x in x, l in l, u in u, acc in accepted) {
                if x > l && x < u {
                    *acc = true;
                }
        })
    };
    let proposal_sampler = || Array1::random(l.len(), StandardNormal);
    util::rejection_sample(&accept_condition, &proposal_sampler, max_iters)
}

fn tn(l: &Array1<f64>, u: &Array1<f64>, max_iters: usize) -> Array1<f64> {
    /*
    % samples a column vector of length=length(l)=length(u)
    % from the standard multivariate normal distribution,
    % truncated over the region [l,u], where -a<l<u<a for some
    % 'a' and l and u are column vectors;
    % uses acceptance rejection and inverse-transform method;
    */
    // controls switch between methods
    let tol = 2.;
    // threshold can be tuned for maximum speed for each platform
    // case: abs(u-l)>tol, uses accept-reject from randn
    let mut coeff = Array1::ones(l.len());
    let gap = (u - l).map(|x| x.abs());
    let mut tl = l.clone();
    let mut tu = u.clone();
    par_azip!((gap in &gap, coeff in &mut coeff, tl in &mut tl, tu in &mut tu) if *gap < tol {*coeff = 0.;*tl=f64::NEG_INFINITY;*tu=f64::INFINITY;});
    let accept_reject = trnd(&tl, &tu, max_iters);
    // case: abs(u-l)<tol, uses inverse-transform
    let pl = (&tl * f64::FRAC_1_SQRT_2()).map(|x| faddeeva::erfc(*x) / 2.);
    let pu = (&tu * f64::FRAC_1_SQRT_2()).map(|x| faddeeva::erfc(*x) / 2.);
    let sample = Array1::random(l.len(), Uniform::new(0., 1.));

    let inverse_transform =
        f64::SQRT_2() * (2. * (&pl - (&pl - pu) * sample)).map(|x| erfc_inv(*x));
    &coeff * &accept_reject + (1. - &coeff) * &inverse_transform
}

/*
%% truncated normal generator
% * efficient generator of a vector of length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u];
% infinite values for 'u' and 'l' are accepted;
% * Remark:
% If you wish to simulate a random variable
% 'Z' from the non-standard Gaussian N(m,s^2)
% conditional on l<Z<u, then first simulate
% X=trandn((l-m)/s,(u-m)/s) and set Z=m+s*X;
%
% See also: norminvp
%
% For more help, see <a href="matlab:
% doc">Truncated Multivariate Student & Normal</a> documentation at the bottom.
% Reference: Z. I. Botev (2017), _The Normal Law Under Linear Restrictions:
% Simulation and Estimation via Minimax Tilting_, Journal of the Royal
% Statistical Society, Series B, Volume 79, Part 1, pp. 1-24
*/
fn trandn(l: &Array1<f64>, u: &Array1<f64>, max_iters: usize) -> Array1<f64> {
    let thresh = 0.66; // tunable threshold to choose method
    let mut tl = l.clone();
    let mut tu = u.clone();
    let mut coeff = Array1::zeros(l.len());
    par_azip!((tl in &mut tl, tu in &mut tu, coeff in &mut coeff) {
        if *tl > thresh {*coeff = 1.}
        else if *tu < -thresh {*tl = -*tu; *tu = -*tl; *coeff = -1.}
        else {*tl = -100.; *tu = 100.; *coeff=0.;} // sample from another method, set params to always accept
    });
    let acc_rej_sample = ntail(tl, tu, max_iters);
    let trunc_norm_sample = tn(l, u, max_iters);
    &coeff * acc_rej_sample + (1. - &coeff.mapv(|x: f64| x.abs())) * trunc_norm_sample
}

fn grad_psi(
    y: &Array1<f64>,
    L: &Array2<f64>,
    l: &Array1<f64>,
    u: &Array1<f64>,
) -> (Array1<f64>, Array2<f64>) {
    // implements gradient of psi(x) to find optimal exponential twisting;
    // assumes scaled 'L' with zero diagonal;
    let d = l.shape()[0];
    assert!(y.len() == 2 * (d - 1));
    let mut c = Array1::zeros(d);
    let mut x = Array1::zeros(d);
    x.slice_mut(s![..d - 1]).assign(&y.slice(s![..d - 1]));
    let mut mu = Array1::zeros(d);
    mu.slice_mut(s![..d - 1]).assign(&y.slice(s![d - 1..]));
    // compute now ~l and ~u
    c.slice_mut(s![1..d]).assign(&L.slice(s![1..d, ..]).dot(&x));
    let mut lt = l - &mu - &c;
    let mut ut = u - &mu - c;
    // compute gradients avoiding catastrophic cancellation
    let w = ln_normal_pr(&lt, &ut);
    let denom = (2. * f64::PI()).sqrt();
    let pl = (-0.5 * lt.map(|x| x.powi(2)) - &w).map(|x| x.exp()) / denom;
    let pu = (-0.5 * ut.map(|x| x.powi(2)) - w).map(|x| x.exp()) / denom;
    let P = &pl - &pu;
    // output the gradient
    let dfdx = P.t().dot(&L.slice(s![.., ..d - 1])) - mu.slice(s![..d - 1]);
    let dfdm = mu - x + &P;
    let grad = concatenate!(Axis(0), dfdx, dfdm.slice(s![..d - 1]));
    // compute jacobian
    lt = lt.map(|x| if x.is_infinite() { 0. } else { *x });
    ut = ut.map(|x| if x.is_infinite() { 0. } else { *x });
    let dP_vec = P.map(|x| -1. * x.powi(2)) + lt * pl - ut * pu;
    let dP = dP_vec
        .broadcast(L.shape())
        .unwrap()
        .reversed_axes()
        .slice(s![.., ..])
        .to_owned();
    // dPdm
    let DL = dP * L;
    let mx: Array2<f64> = -1. * Array2::eye(d) + &DL;
    let mx = mx.slice(s![..d - 1, ..d - 1]);
    let xx = L.t().dot(&DL);
    let xx = xx.slice(s![..d - 1, ..d - 1]);
    let a = concatenate!(Axis(1), xx, mx.t());
    let b = concatenate!(
        Axis(1),
        mx,
        Array2::eye(d - 1) * (dP_vec.slice(s![..d - 1]).map(|x| x + 1.))
    );
    let J = concatenate!(Axis(0), a, b);
    (grad, J)
}

fn psy(
    x: &Array1<f64>,
    L: &Array2<f64>,
    l: &Array1<f64>,
    u: &Array1<f64>,
    mu: &Array1<f64>,
) -> f64 {
    // implements psi(x,mu); assumes scaled 'L' without diagonal;
    let mut temp = Array1::zeros(x.len() + 1);
    temp.slice_mut(s![..x.len()]).assign(&x);
    let x = temp;
    let mut temp = Array1::zeros(mu.len() + 1);
    temp.slice_mut(s![..mu.len()]).assign(&mu);
    let mu = temp;
    // compute now ~l and ~u
    let c = L.dot(&x);
    let tl = l - &mu - &c;
    let tu = u - &mu - &c;
    (ln_normal_pr(&tl, &tu) + 0.5 * mu.mapv(|x| x.powi(2)) - x * mu).sum()
}

/*
% computes P(l<X<u), where X is normal with
% 'Cov(X)=L*L' and zero mean vector;
% exponential tilting uses parameter 'mu';
% Monte Carlo uses 'n' samples;
*/
fn mv_normal_pr(
    n: usize,
    L: &Array2<f64>,
    l: &Array1<f64>,
    u: &Array1<f64>,
    mu: &Array1<f64>,
    max_iters: usize,
) -> (f64, f64) {
    let d = l.shape()[0];
    let mut p = Array1::zeros(n);
    let mut temp = Array1::zeros(mu.shape()[0] + 1);
    temp.slice_mut(s![..d - 1]).assign(&mu);
    let mu = temp;
    let mut Z = Array2::zeros((d, n));
    let mut col;
    let mut tl;
    let mut tu;
    for k in 0..d - 1 {
        col = L.slice(s![k, ..k]).dot(&Z.slice(s![..k, ..]));
        tl = l[[k]] - mu[[k]] - &col;
        tu = u[[k]] - mu[[k]] - col;
        // simulate N(mu, 1) conditional on [tl,tu]
        Z.index_axis_mut(Axis(0), k)
            .assign(&(mu[[k]] + trandn(&tl, &tu, max_iters)));
        // update likelihood ratio
        p = p + ln_normal_pr(&tl, &tu) + 0.5 * mu[[k]].powi(2)
            - mu[[k]] * &Z.index_axis(Axis(0), k);
    }
    col = L.index_axis(Axis(0), d - 1).dot(&Z);
    tl = l[d - 1] - &col;
    tu = u[d - 1] - col;
    p = p + ln_normal_pr(&tl, &tu);
    p.mapv_inplace(|x: f64| x.exp());
    let prob = p.mean().unwrap();
    let rel_err = p.std(0.) / (n as f64).sqrt() / prob;
    (prob, rel_err)
}

#[allow(clippy::type_complexity)]
fn solve_optimial_tiling(
    mut l: Array1<f64>,
    mut u: Array1<f64>,
    mut sigma: Array2<f64>,
) -> (
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<usize>,
) {
    let d = l.shape()[0];
    let (mut L, perm) = cholperm(&mut sigma, &mut l, &mut u);
    let D = L.diag().to_owned();

    u /= &D;
    l /= &D;
    L = (L / (Array2::<f64>::zeros([D.len(), D.len()]) + &D).t()) - Array2::<f64>::eye(d);

    // find optimal tilting parameter via non-linear equation solver
    let problem = optimization::TilingProblem::new(L.clone(), l.clone(), u.clone());
    let (result, _report) = LevenbergMarquardt::new().minimize(problem);

    let x = result.get_x().slice(s![..d - 1]).to_owned();
    // assign saddlepoint x* and mu*
    let mu = result.get_x().slice(s![d - 1..(2 * (d - 1))]).to_owned();
    (L, l, u, x, mu, perm)
}

pub fn mv_truncnormal_cdf(
    l: Array1<f64>,
    u: Array1<f64>,
    sigma: Array2<f64>,
    n: usize,
    max_iters: usize,
) -> (f64, f64, f64) {
    let (L, l, u, x, mu, _) = solve_optimial_tiling(l, u, sigma);
    // compute psi star
    let (est, rel_err) = mv_normal_pr(n, &L, &l, &u, &mu, max_iters);
    // calculate an upper bound
    let log_upbnd = psy(&x, &L, &l, &u, &mu);
    /*
    if log_upbnd < -743. {
        panic!(
            "Natural log of upbnd probability is less than -743, yielding 0 after exponentiation!"
        )
    }
    */
    let upbnd = log_upbnd.exp();
    (est, rel_err, upbnd)
}

fn mv_truncnorm_proposal(
    L: &Array2<f64>,
    l: &Array1<f64>,
    u: &Array1<f64>,
    mu: &Array1<f64>,
    n: usize,
    max_iters: usize,
) -> (Array1<f64>, Array2<f64>) {
    /*
    % generates the proposals from the exponentially tilted
    % sequential importance sampling pdf;
    % output:    'p', log-likelihood of sample
    %             Z, random sample
    */
    let d = l.shape()[0];
    let mut logp = Array1::zeros(n);
    let mut temp = Array1::zeros(mu.shape()[0] + 1);
    temp.slice_mut(s![..d - 1]).assign(&mu);
    let mu = temp;
    let mut Z = Array2::zeros((d, n));
    let mut col;
    let mut tl;
    let mut tu;
    for k in 0..d {
        col = L.slice(s![k, ..k]).dot(&Z.slice(s![..k, ..]));
        tl = l[[k]] - mu[[k]] - &col;
        tu = u[[k]] - mu[[k]] - col;
        // simulate N(mu, 1) conditional on [tl,tu]
        Z.index_axis_mut(Axis(0), k)
            .assign(&(mu[[k]] + trandn(&tl, &tu, max_iters)));
        // update likelihood ratio
        logp = logp + ln_normal_pr(&tl, &tu) + 0.5 * mu[[k]].powi(2)
            - mu[[k]] * &Z.index_axis(Axis(0), k);
    }
    (logp, Z)
}

pub fn mv_truncnormal_rand(
    mut l: Array1<f64>,
    mut u: Array1<f64>,
    mut sigma: Array2<f64>,
    n: usize,
    max_iters: usize,
) -> (Array2<f64>, Array1<f64>) {
    let d = l.len();
    let (Lfull, perm) = cholperm(&mut sigma, &mut l, &mut u);
    let D = Lfull.diag().to_owned();

    u /= &D;
    l /= &D;
    let L = (&Lfull / &(Array2::<f64>::zeros([D.len(), D.len()]) + &D).t()) - Array2::<f64>::eye(d);

    // find optimal tilting parameter via non-linear equation solver
    let problem = optimization::TilingProblem::new(L.clone(), l.clone(), u.clone());
    let (result, _report) = LevenbergMarquardt::new().minimize(problem);

    let x = result.get_x().slice(s![..d - 1]).to_owned();
    // assign saddlepoint x* and mu*
    let mu = result.get_x().slice(s![d - 1..(2 * (d - 1))]).to_owned();
    let psi_star = psy(&x, &L, &l, &u, &mu); // compute psi star
    let (logp, mut Z) = mv_truncnorm_proposal(&L, &l, &u, &mu, n, max_iters);

    let accept_condition = |logp: &Array1<f64>, accepted: &mut Array1<bool>| {
        let test_sample: Array1<f64> = Array1::random(logp.len(), Uniform::new(0., 1.));
        par_azip!((&s in &test_sample, &logp in logp, acc in accepted) {
                if -1. * s.ln() > (psi_star - logp) {
                    *acc = true;
                }
        })
    };
    let mut accepted: Array1<bool> = Array1::from_elem(Z.ncols(), false);
    accept_condition(&logp, &mut accepted);
    let mut i = 0;
    while !accepted.fold(true, |a, b| a && *b) {
        let (logp, sample) = mv_truncnorm_proposal(&L, &l, &u, &mu, n, max_iters);
        Zip::from(Z.axis_iter_mut(Axis(1)))
            .and(sample.axis_iter(Axis(1)))
            .and(&accepted)
            .for_each(|mut z, s, &acc| {
                if !acc {
                    z.assign(&s);
                }
            });
        accept_condition(&logp, &mut accepted);
        i += 1;
        if i > max_iters {
            println!("Ran out of accept-reject rounds");
            break;
        }
    }
    // postprocess samples
    let mut unperm = perm.into_iter().zip(0..d).collect::<Vec<(usize, usize)>>();
    unperm.sort_by(|a, b| a.0.cmp(&b.0));
    let order: Vec<usize> = unperm.iter().map(|x| x.1).collect();

    // reverse scaling of L
    let mut rv = Lfull.dot(&Z);
    for (i, &ord) in order.iter().enumerate() {
        rv.row_mut(i).assign(&Z.row(ord));
    }
    (rv.reversed_axes(), logp)
}

#[cfg(test)]
mod tests {
    extern crate ndarray;
    extern crate test;
    use super::*;
    use ndarray::array;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::rand_distr::Uniform;
    use test::Bencher;

    #[bench]
    // with par e0::2.63e3, e1::1.22e4, e2::1.94e4, e3::4.85e4
    fn bench_ln_normal_pr(bench: &mut Bencher) {
        let normal = Normal::new(0., 1.).unwrap();
        let uniform = Uniform::new(0., 2.);
        let a = Array2::random((1000, 1), normal);
        let b = Array2::random((1000, 1), uniform);
        let c = a.clone() + b;
        bench.iter(|| test::black_box(ln_normal_pr(&a, &c)));
    }

    #[bench]
    // e1::1.57e5, e2::2.83e6, e3::4.20e8
    fn bench_cholperm(b: &mut Bencher) {
        let n = 10;
        let mut sigma = Array2::eye(n);
        sigma.row_mut(2).map_inplace(|x| *x += 0.01);
        sigma.column_mut(2).map_inplace(|x| *x += 0.01);
        let uniform = Uniform::new(0., 1.);
        let mut l = Array1::random(n, uniform);
        let mut u = 2. * &l;
        b.iter(|| test::black_box(cholperm(&mut sigma, &mut l, &mut u)));
    }

    #[test]
    fn test_grad_psi() {
        let d = 25;
        let mut l = Array1::ones(d) / 2.;
        let mut u = Array1::ones(d);
        let mut sigma: Array2<f64> =
            Array2::from_elem((25, 25), -0.07692307692307693) + Array2::<f64>::eye(25) * 2.;
        //let y = Array1::ones(d);
        let y = Array::range(0., 2. * (d - 1) as f64, 1.);

        let d = l.shape()[0];
        let (mut L, _perm) = cholperm(&mut sigma, &mut l, &mut u);
        let D = L.diag().to_owned();
        u = u / &D;
        l = l / &D;
        L = (L / (Array2::<f64>::zeros([D.len(), D.len()]) + &D).t()) - Array2::<f64>::eye(d);
        let (residuals, jacobian) = grad_psi(&y, &L, &l, &u);
    }

    #[test]
    fn test_mv_normal_cdf() {
        let d = 25;
        let l = Array1::ones(d) / 2.;
        let u = Array1::ones(d);
        let sigma: Array2<f64> =
            Array2::from_elem((25, 25), -0.07692307692307693) + Array2::<f64>::eye(25) * 2.;
        let (est, rel_err, upper_bound) = mv_truncnormal_cdf(l, u, sigma, 10000, 10);
        println!("{:?}", (est, rel_err, upper_bound));
        /* Should be close to:
        prob: 2.6853e-53
        relErr: 2.1390e-04
        upbnd: 2.8309e-53
        */
    }

    #[test]
    fn test_mv_truncnormal_rand() {
        let d = 3;
        let l = Array1::ones(d) / 2.;
        let u = Array1::ones(d);
        let sigma: Array2<f64> =
            Array2::from_elem((d, d), -0.07692307692307693) + Array2::<f64>::eye(d) * 2.;
        println!("l {}", l);
        println!("u {}", u);
        let (samples, logp) = mv_truncnormal_rand(l, u, sigma, 5, 10);
    }

    #[bench]
    fn bench_mv_normal_cdf(b: &mut Bencher) {
        let d = 25;
        let l = Array1::ones(d) / 2.;
        let u = Array1::ones(d);
        let sigma: Array2<f64> =
            Array2::from_elem((25, 25), -0.07692307692307693) + Array2::<f64>::eye(25) * 2.;
        b.iter(|| {
            test::black_box(mv_truncnormal_cdf(
                l.clone(),
                u.clone(),
                sigma.clone(),
                20000,
                10,
            ))
        });
    }
}
