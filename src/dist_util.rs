use crate::gauss::erfc;
use crate::gauss::erfcx;
use crate::util;
use ndarray::array;
use ndarray::s;
use ndarray::Array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::Slice;
use ndarray::Zip;
use ndarray_stats::QuantileExt;
use num::traits::FloatConst;

pub fn cholperm(
    sigma: &mut Array2<f64>,
    l: &mut Array1<f64>,
    u: &mut Array1<f64>,
) -> (Array2<f64>, Array1<usize>) {
    let d = l.shape()[0];
    let mut L: Array2<f64> = Array2::zeros((d, d));
    let mut z: Array1<f64> = Array1::zeros(d);
    let mut perm = Array1::range(0., d as f64, 1.);

    for j in 0..d {
        let mut pr = Array1::from_elem(d, f64::INFINITY);
        let diag = sigma.diag();
        let L_chunk = L.slice(s![j..d, 0..j]);

        let subtr = L_chunk.dot(&z.slice(s![0..j]));
        let s = (&diag.slice_axis(Axis(0), Slice::from(j..d))
            - L_chunk.mapv(|x| x * x).sum_axis(Axis(1)))
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
        let mut s_scalar = sigma[[j, j]] - L.slice(s![j, 0..j]).mapv(|x| x * x).sum();
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
        z[[j]] =
            ((-0.5 * tl_s * tl_s - w).exp() - (-0.5 * tu_s * tu_s - w).exp()) / (f64::TAU()).sqrt();
    }
    (L, perm.mapv(|x| x as usize))
}

fn ln_phi(x: f64) -> f64 {
    //! computes logarithm of tail of $Z\sim N(0,1)$ mitigating numerical roundoff errors
    -0.5 * x * x - f64::LN_2() + erfcx(x * f64::FRAC_1_SQRT_2()).ln()
}

pub fn ln_normal_pr<D: ndarray::Dimension>(a: &Array<f64, D>, b: &Array<f64, D>) -> Array<f64, D> {
    //! computes $\ln(\Pr(a<Z<b))$ where $Z\sim N(0,1)$ very accurately for any $a,b$
    Zip::from(a).and(b).map_collect(|&a, &b| {
        if a > 0. {
            let pa = ln_phi(a);
            let pb = ln_phi(b);
            pa + (-1. * (pb - pa).exp()).ln_1p()
        } else if b < 0. {
            let pa = ln_phi(-a);
            let pb = ln_phi(-b);
            pb + (-1. * (pa - pb).exp()).ln_1p()
        } else {
            let pa = erfc(-1. * a * f64::FRAC_1_SQRT_2()) / 2.;
            let pb = erfc(b * f64::FRAC_1_SQRT_2()) / 2.;
            (-pa - pb).ln_1p()
        }
    })
}
