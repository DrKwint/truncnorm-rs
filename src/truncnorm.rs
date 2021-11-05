#![allow(non_snake_case)]
#![allow(clippy::many_single_char_names)]
//! Rust re-write of [Truncated Normal and Student's t-distribution toolbox](https://www.mathworks.com/matlabcentral/fileexchange/53796-truncated-normal-and-student-s-t-distribution-toolbox)
//!
//! Reference: Z. I. Botev (2017), _The Normal Law Under Linear Restrictions:
//! Simulation and Estimation via Minimax Tilting_, Journal of the Royal
//! Statistical Society, Series B, Volume 79, Part 1, pp. 1-24
use crate::dist_util::cholperm;
use crate::dist_util::ln_normal_pr;
use crate::faddeeva::erfc;
use crate::tilting::TiltingProblem;
use crate::tilting::TiltingSolution;
use crate::util;
use ndarray::azip;
use ndarray::Zip;
use ndarray::{s, Axis};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use num::traits::FloatConst;
use rand::distributions::Uniform;
use rand::Rng;
use statrs::function::erf::erfc_inv;

fn ntail<R: Rng + ?Sized>(
    l: &Array1<f64>,
    u: &Array1<f64>,
    max_iters: usize,
    rng: &mut R,
) -> Array1<f64> {
    /*
    % samples a vector from the standard normal
    % distribution truncated over the region [l,u], where l>0 and
    % l and u are column vectors; uses acceptance-rejection from Rayleigh
    % distr. Similar to Marsaglia (1964)
    */
    let c = l.map(|x| x.powi(2) / 2.);
    let f = Zip::from(&c)
        .and(u)
        .map_collect(|&c, &u| (c - u * u / 2.).exp_m1());
    // use rejection sample pattern
    let mut accept_condition = |x: &Array1<f64>, accepted: &mut Array1<bool>, rng: &mut R| {
        let test_sample: Array1<f64> = Array1::random_using(l.len(), Uniform::new(0., 1.), rng);
        azip!((x in x, &s in &test_sample, &c in &c, acc in accepted) {
            if s * s * x < c {
            *acc = true;
            }
        })
    };
    let mut proposal_sampler = |rng: &mut R| {
        let sample = Array1::random_using(l.len(), Uniform::new(0., 1.), rng);
        &c - (1. + sample * &f).mapv(|x| x.ln())
    };
    let mut output_array =
        util::rejection_sample(&mut accept_condition, &mut proposal_sampler, max_iters, rng);
    output_array.mapv_inplace(|x| (2. * x).sqrt());
    output_array
}

fn trnd<R: Rng + ?Sized>(
    l: &Array1<f64>,
    u: &Array1<f64>,
    max_iters: usize,
    rng: &mut R,
) -> Array1<f64> {
    // use accept-reject pattern to sample from truncated N(0,1)
    let mut accept_condition = |x: &Array1<f64>, accepted: &mut Array1<bool>, _rng: &mut R| {
        azip!((x in x, l in l, u in u, acc in accepted) {
            if x > l && x < u {
            *acc = true;
            }
        })
    };
    let mut proposal_sampler = |rng: &mut R| Array1::random_using(l.len(), StandardNormal, rng);
    util::rejection_sample(&mut accept_condition, &mut proposal_sampler, max_iters, rng)
}

fn tn<R: Rng + ?Sized>(
    l: &Array1<f64>,
    u: &Array1<f64>,
    max_iters: usize,
    rng: &mut R,
) -> Array1<f64> {
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
    azip!((gap in &gap, coeff in &mut coeff, tl in &mut tl, tu in &mut tu) if *gap < tol {*coeff = 0.;*tl=f64::NEG_INFINITY;*tu=f64::INFINITY;});
    let accept_reject = trnd(&tl, &tu, max_iters, rng);
    // case: abs(u-l)<tol, uses inverse-transform
    let pl = (&tl * f64::FRAC_1_SQRT_2()).map(|x| erfc(*x) / 2.);
    let pu = (&tu * f64::FRAC_1_SQRT_2()).map(|x| erfc(*x) / 2.);
    let sample = Array1::random_using(l.len(), Uniform::new(0., 1.), rng);

    let inverse_transform =
        f64::SQRT_2() * (2. * (&pl - (&pl - &pu) * sample)).map(|x| erfc_inv(*x));
    let mut result = &coeff * &accept_reject + (1. - &coeff) * &inverse_transform;
    if result.iter().any(|x| x.is_nan()) {
        result = coeff
            .iter()
            .zip(inverse_transform.iter())
            .zip(accept_reject.iter())
            .map(|x| if *x.0 .0 == 0. { *x.0 .1 } else { *x.1 })
            .collect();
    }
    result
}

/// fast truncated normal generator
///
///  Infinite values for 'u' and 'l' are accepted;
///
///  If you wish to simulate a random variable
/// 'Z' from the non-standard Gaussian $N(\mu,\sigma^2)$
///  conditional on $l<Z<u$, first simulate
///  $X=trandn((l-m)/s,(u-m)/s)$ and set $Z=\mu+\sigma X$;
pub fn trandn<R: Rng + ?Sized>(
    l: &Array1<f64>,
    u: &Array1<f64>,
    max_iters: usize,
    rng: &mut R,
) -> Array1<f64> {
    let thresh = 0.66; // tunable threshold to choose method
    let mut tl = l.clone();
    let mut tu = u.clone();
    let mut coeff = Array1::zeros(l.len());
    azip!((tl in &mut tl, tu in &mut tu, coeff in &mut coeff) {
        if *tl > thresh {*coeff = 1.}
        else if *tu < -thresh {*tl = -*tu; *tu = -*tl; *coeff = -1.}
        else {*tl = -100.; *tu = 100.; *coeff=0.;} // sample from another method, set params to always accept
    });
    let acc_rej_sample = ntail(&tl, &tu, max_iters, rng);
    let trunc_norm_sample = tn(l, u, max_iters, rng);
    &coeff * acc_rej_sample + (1. - &coeff.mapv(|x: f64| x.abs())) * trunc_norm_sample
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
    temp.slice_mut(s![..x.len()]).assign(x);
    let x = temp;
    let mut temp = Array1::zeros(mu.len() + 1);
    temp.slice_mut(s![..mu.len()]).assign(mu);
    let mu = temp;
    // compute now ~l and ~u
    let c = L.dot(&x);
    let tl = l - &mu - &c;
    let tu = u - &mu - &c;
    (ln_normal_pr(&tl, &tu) + 0.5 * mu.mapv(|x| x * x) - x * mu).sum()
}

/*
% computes P(l<X<u), where X is normal with
% 'Cov(X)=L*L' and zero mean vector;
% exponential tilting uses parameter 'mu';
% Monte Carlo uses 'n' samples;
*/
fn mv_normal_pr<R: Rng + ?Sized>(
    n: usize,
    L: &Array2<f64>,
    l: &Array1<f64>,
    u: &Array1<f64>,
    mu: &Array1<f64>,
    max_iters: usize,
    rng: &mut R,
) -> (f64, f64) {
    let d = l.shape()[0];
    let mut p = Array1::zeros(n);
    let mut temp = Array1::zeros(mu.shape()[0] + 1);
    temp.slice_mut(s![..d - 1]).assign(mu);
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
            .assign(&(mu[[k]] + trandn(&tl, &tu, max_iters, rng)));
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
    debug_assert!(
        !prob.is_sign_negative(),
        "Returned invalid probability, {:?}",
        prob
    );
    let rel_err = p.std(0.) / (n as f64).sqrt() / prob;
    (prob, rel_err)
}

fn mv_truncnorm_proposal<R: Rng + ?Sized>(
    L: &Array2<f64>,
    l: &Array1<f64>,
    u: &Array1<f64>,
    mu: &Array1<f64>,
    n: usize,
    max_iters: usize,
    rng: &mut R,
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
    temp.slice_mut(s![..d - 1]).assign(mu);
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
            .assign(&(mu[[k]] + trandn(&tl, &tu, max_iters, rng)));
        // update likelihood ratio
        logp = logp + ln_normal_pr(&tl, &tu) + 0.5 * mu[[k]] * mu[[k]]
            - mu[[k]] * &Z.index_axis(Axis(0), k);
    }
    (logp, Z)
}

pub fn solved_mv_truncnormal_rand<R: Rng + ?Sized>(
    tilting_solution: &TiltingSolution,
    mut l: Array1<f64>,
    mut u: Array1<f64>,
    mut sigma: Array2<f64>,
    n: usize,
    max_iters: usize,
    rng: &mut R,
) -> Array2<f64> {
    let d = l.len();
    let (Lfull, perm) = cholperm(&mut sigma, &mut l, &mut u);
    let D = Lfull.diag().to_owned();

    u /= &D;
    l /= &D;
    let L = (&Lfull / &(Array2::<f64>::zeros([D.len(), D.len()]) + &D).t()) - Array2::<f64>::eye(d);

    let x = &tilting_solution.x;
    let mu = &tilting_solution.mu;
    let psi_star = psy(x, &L, &l, &u, mu); // compute psi star
    let (logp, mut Z) = mv_truncnorm_proposal(&L, &l, &u, mu, n, max_iters, rng);

    let accept_condition = |logp: &Array1<f64>, accepted: &mut Array1<bool>, rng: &mut R| {
        let test_sample: Array1<f64> = Array1::random_using(logp.len(), Uniform::new(0., 1.), rng);
        azip!((&s in &test_sample, &logp in logp, acc in accepted) {
            if -1. * s.ln() > (psi_star - logp) {
            *acc = true;
            }
        })
    };
    let mut accepted: Array1<bool> = Array1::from_elem(Z.ncols(), false);
    accept_condition(&logp, &mut accepted, rng);
    let mut i = 0;
    while !accepted.fold(true, |a, b| a && *b) {
        let (logp, sample) = mv_truncnorm_proposal(&L, &l, &u, mu, n, max_iters, rng);
        Zip::from(Z.axis_iter_mut(Axis(1)))
            .and(sample.axis_iter(Axis(1)))
            .and(&accepted)
            .for_each(|mut z, s, &acc| {
                if !acc {
                    z.assign(&s);
                }
            });
        accept_condition(&logp, &mut accepted, rng);
        i += 1;
        if i > max_iters {
            //println!("Ran out of accept-reject rounds");
            break;
        }
    }
    // postprocess samples
    let mut unperm = perm.into_iter().zip(0..d).collect::<Vec<(usize, usize)>>();
    unperm.sort_by(|a, b| a.0.cmp(&b.0));
    let order: Vec<usize> = unperm.iter().map(|x| x.1).collect();

    // reverse scaling of L
    let mut rv = Lfull.dot(&Z);
    let unperm_rv = rv.clone();
    for (i, &ord) in order.iter().enumerate() {
        rv.row_mut(i).assign(&unperm_rv.row(ord));
    }
    rv.reversed_axes()
}

/// truncated multivariate normal generator
pub fn mv_truncnormal_rand<R: Rng + ?Sized>(
    mut l: Array1<f64>,
    mut u: Array1<f64>,
    mut sigma: Array2<f64>,
    n: usize,
    max_iters: usize,
    rng: &mut R,
) -> Array2<f64> {
    let d = l.len();
    let (Lfull, perm) = cholperm(&mut sigma, &mut l, &mut u);
    let D = Lfull.diag().to_owned();

    u /= &D;
    l /= &D;
    let L = (&Lfull / &(Array2::<f64>::zeros([D.len(), D.len()]) + &D).t()) - Array2::<f64>::eye(d);

    // find optimal tilting parameter via non-linear equation solver
    let problem = TiltingProblem::new(l.clone(), u.clone(), sigma);
    let result = problem.solve_optimial_tilting();
    // assign saddlepoint x* and mu*
    let x = result.x.slice(s![..d - 1]).to_owned();
    let mu = result.x.slice(s![d - 1..(2 * (d - 1))]).to_owned();
    let psi_star = psy(&x, &L, &l, &u, &mu); // compute psi star
    let (logp, mut Z) = mv_truncnorm_proposal(&L, &l, &u, &mu, n, max_iters, rng);

    let accept_condition = |logp: &Array1<f64>, accepted: &mut Array1<bool>, rng: &mut R| {
        let test_sample: Array1<f64> = Array1::random_using(logp.len(), Uniform::new(0., 1.), rng);
        azip!((&s in &test_sample, &logp in logp, acc in accepted) {
            if -1. * s.ln() > (psi_star - logp) {
            *acc = true;
            }
        })
    };
    let mut accepted: Array1<bool> = Array1::from_elem(Z.ncols(), false);
    accept_condition(&logp, &mut accepted, rng);
    let mut i = 0;
    while !accepted.fold(true, |a, b| a && *b) {
        let (logp, sample) = mv_truncnorm_proposal(&L, &l, &u, &mu, n, max_iters, rng);
        Zip::from(Z.axis_iter_mut(Axis(1)))
            .and(sample.axis_iter(Axis(1)))
            .and(&accepted)
            .for_each(|mut z, s, &acc| {
                if !acc {
                    z.assign(&s);
                }
            });
        accept_condition(&logp, &mut accepted, rng);
        i += 1;
        if i > max_iters {
            break;
        }
    }
    // postprocess samples
    let mut unperm = perm.into_iter().zip(0..d).collect::<Vec<(usize, usize)>>();
    unperm.sort_by(|a, b| a.0.cmp(&b.0));
    let order: Vec<usize> = unperm.iter().map(|x| x.1).collect();

    // reverse scaling of L
    let mut rv = Lfull.dot(&Z);
    let unperm_rv = rv.clone();
    for (i, &ord) in order.iter().enumerate() {
        rv.row_mut(i).assign(&unperm_rv.row(ord));
    }
    rv.reversed_axes()
}

pub fn solved_mv_truncnormal_cdf<R: Rng + ?Sized>(
    tilting_solution: &TiltingSolution,
    n: usize,
    max_iters: usize,
    rng: &mut R,
) -> (f64, f64, f64) {
    // compute psi star
    let (est, rel_err) = mv_normal_pr(
        n,
        &tilting_solution.lower_tri,
        &tilting_solution.lower,
        &tilting_solution.upper,
        &tilting_solution.mu,
        max_iters,
        rng,
    );
    // calculate an upper bound
    let log_upbnd = psy(
        &tilting_solution.x,
        &tilting_solution.lower_tri,
        &tilting_solution.lower,
        &tilting_solution.upper,
        &tilting_solution.mu,
    );
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

/// multivariate normal cumulative distribution
pub fn mv_truncnormal_cdf<R: Rng + ?Sized>(
    l: Array1<f64>,
    u: Array1<f64>,
    sigma: Array2<f64>,
    n: usize,
    max_iters: usize,
    rng: &mut R,
) -> (f64, f64, f64) {
    let tilting_solution = TiltingProblem::new(l, u, sigma).solve_optimial_tilting();
    // compute psi star
    let (est, rel_err) = mv_normal_pr(
        n,
        &tilting_solution.lower_tri,
        &tilting_solution.lower,
        &tilting_solution.upper,
        &tilting_solution.mu,
        max_iters,
        rng,
    );
    // calculate an upper bound
    let log_upbnd = psy(
        &tilting_solution.x,
        &tilting_solution.lower_tri,
        &tilting_solution.lower,
        &tilting_solution.upper,
        &tilting_solution.mu,
    );
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

#[cfg(test)]
mod tests {
    extern crate ndarray;
    extern crate test;
    use super::*;
    use ndarray::{arr1, arr2};
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::rand_distr::Uniform;
    use test::Bencher;

    #[test]
    fn manual_rand_scale_test() {
        let l = arr1(&[f64::NEG_INFINITY, f64::NEG_INFINITY]);
        //let u: Array1<f64> = arr1(&[-7.33, 0.1]);
        let u = arr1(&[-7.75, 9.11]);
        let sigma = arr2(&[[10., -10.], [-10., 11.]]);
        let mut rng = rand::thread_rng();
        let n = 10;
        let max_iters = 10;
        let samples = mv_truncnormal_rand(l, u, sigma, n, max_iters, &mut rng);
        println!("samples: {:?}", samples);
    }

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

    /*
    #[test]
    fn test_grad_psi() {
        let d = 25;
        let mut l = Array1::ones(d) / 2.;
        let mut u = Array1::ones(d);
        let mut sigma: Array2<f64> =
            Array2::from_elem((25, 25), -0.07692307692307693) + Array2::<f64>::eye(25) * 2.;
        //let y = Array1::ones(d);
        let y = Array::range(0., 2. * (d - 1) as f64, 1.);

        let (mut L, _perm) = cholperm(&mut sigma, &mut l, &mut u);
        let D = L.diag().to_owned();
        u /= &D;
        l /= &D;
        L = (L / (Array2::<f64>::zeros([D.len(), D.len()]) + &D).t()) - Array2::<f64>::eye(d);
        let (residuals, jacobian) = grad_psi(&y, &L, &l, &u);
        println!("{:?}", (residuals, jacobian))
    }
    */

    #[test]
    fn test_mv_normal_cdf() {
        let d = 25;
        let l = Array1::ones(d) / 2.;
        let u = Array1::ones(d);
        let sigma: Array2<f64> =
            Array2::from_elem((25, 25), -0.07692307692307693) + Array2::<f64>::eye(25) * 2.;
        let mut rng = rand::thread_rng();
        let (est, rel_err, upper_bound) = mv_truncnormal_cdf(l, u, sigma, 10000, 10, &mut rng);
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
        let mut rng = rand::thread_rng();
        let (samples, logp) = mv_truncnormal_rand(l, u, sigma, 5, 10, &mut rng);
        println!("{:?}", (samples, logp));
    }

    #[bench]
    fn bench_mv_normal_cdf(b: &mut Bencher) {
        let d = 25;
        let l = Array1::ones(d) / 2.;
        let u = Array1::ones(d);
        let sigma: Array2<f64> =
            Array2::from_elem((25, 25), -0.07692307692307693) + Array2::<f64>::eye(25) * 2.;
        let mut rng = rand::thread_rng();
        b.iter(|| {
            test::black_box(mv_truncnormal_cdf(
                l.clone(),
                u.clone(),
                sigma.clone(),
                20000,
                10,
                &mut rng,
            ))
        });
    }
}
