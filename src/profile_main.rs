use ndarray::{Array1, Array2};
use truncnorm::mv_truncnormal_cdf;

fn main() {
    let d = 25;
    let l = Array1::ones(d) / 2.;
    let u = Array1::ones(d);
    let sigma: Array2<f64> =
        Array2::from_elem((25, 25), -0.07692307692307693) + Array2::<f64>::eye(25) * 2.;
    mv_truncnormal_cdf(l, u, sigma, 20000);
}
