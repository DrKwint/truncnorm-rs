//! High level distributions
use crate::dist_util::ln_normal_pr;
use crate::tilting::TiltingProblem;
use crate::tilting::TiltingSolution;
use crate::truncnorm;
use crate::truncnorm::solved_mv_truncnormal_rand;
use ndarray::array;
use ndarray::Array;
use ndarray::Array2;
use ndarray::Dimension;
use ndarray::Zip;
use ndarray::{Array1, ArrayView1};
use ndarray::{Ix1, Ix2};
use ndarray_linalg::cholesky::Cholesky;
use ndarray_linalg::eigh::Eigh;
use ndarray_linalg::solveh::UPLO;
use ndarray_rand::rand_distr::num_traits::FloatConst;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::prelude::Distribution;
use rand::Rng;

/// Multivariate normal distribution with diagonal covariance
pub type MultivariateTruncatedNormalDiag = MultivariateTruncatedNormal<Ix1>;

/// Multivariate normal distribution with full covariance
#[derive(Clone, Debug)]
pub struct MultivariateNormal {
    loc: Array1<f64>,
    scale: Array2<f64>,
}

impl MultivariateNormal {
    #[must_use]
    pub const fn new(loc: Array1<f64>, scale: Array2<f64>) -> Self {
        Self { loc, scale }
    }

    /// # Panics
    /// Blah
    ///
    /// Source: <http://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/>
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn logp(&self, x: &Array1<f64>) -> f64 {
        let (eig_vals, eig_vecs) = self.scale.eigh(UPLO::Lower).unwrap();
        let logdet = eig_vals.mapv(|x| x.ln()).sum();
        let inv_vals = eig_vals.mapv(|x| 1. / x);
        let U = eig_vecs * inv_vals.mapv(|x| x.sqrt());
        let dev = x - &self.loc;
        let maha_dist = dev.dot(&U).mapv(|x| x * x).sum();
        -0.5 * ((eig_vals.len() as f64).mul_add(f64::TAU().ln(), maha_dist) + logdet)
    }
}

impl Distribution<Array1<f64>> for MultivariateNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let sample_size = self.loc.len();
        let lower = self.scale.cholesky(UPLO::Lower).unwrap();
        let ind_sample = Array::random_using(sample_size, StandardNormal, rng);
        &self.loc + lower.dot(&ind_sample)
    }
}

/// Truncated multivariate normal distribution with full covariance
///
/// Note that the bounds are per dimension, so the distribution is defined in an axis-aligned box.
#[derive(Clone, Debug)]
pub struct MultivariateTruncatedNormal<D: Dimension> {
    loc: Array1<f64>,
    scale: Array<f64, D>,
    lbs: Array1<f64>,
    ubs: Array1<f64>,
    log_normalizer: Array1<f64>,
    max_iters: usize,
    tilting_solution: Option<TiltingSolution>,
}

impl MultivariateTruncatedNormal<Ix1> {
    /// # Panics
    #[must_use]
    pub fn new(
        loc: Array1<f64>,
        scale: Array1<f64>,
        lower_bounds: Array1<f64>,
        upper_bounds: Array1<f64>,
        max_accept_reject_iters: usize,
    ) -> Self {
        assert_eq!(loc.shape(), scale.shape());
        assert_eq!(loc.shape(), lower_bounds.shape());
        assert_eq!(loc.shape(), upper_bounds.shape());
        let log_normalizer = ln_normal_pr(
            &((&lower_bounds - &loc) / &scale),
            &((&upper_bounds - &loc) / &scale),
        );
        Self {
            loc,
            scale,
            lbs: lower_bounds,
            ubs: upper_bounds,
            log_normalizer,
            max_iters: max_accept_reject_iters,
            tilting_solution: None,
        }
    }

    #[must_use]
    pub fn log_prob(&self, x: ArrayView1<f64>) -> f64 {
        let std_x = (&x - &self.loc) / &self.scale;
        let halfrtln2pi = 0.5 * (f64::TAU()).ln();
        Zip::from(&std_x)
            .and(&self.scale)
            .and(&self.log_normalizer)
            .map_collect(|&x, &s, &lnz| -(0.5_f64.mul_add(x * x, halfrtln2pi) + s.ln() - lnz))
            .sum()
    }
}

impl Distribution<Array1<f64>> for MultivariateTruncatedNormal<Ix1> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let X = truncnorm::trandn(
            &((&self.lbs - &self.loc) / &self.scale),
            &((&self.ubs - &self.loc) / &self.scale),
            self.max_iters,
            rng,
        );
        &self.loc + &self.scale * X
    }
}

impl MultivariateTruncatedNormal<Ix2> {
    #[must_use]
    pub fn new(
        loc: Array1<f64>,
        scale: Array2<f64>,
        lbs: Array1<f64>,
        ubs: Array1<f64>,
        max_accept_reject_iters: usize,
    ) -> Self {
        Self {
            loc,
            scale,
            lbs,
            ubs,
            log_normalizer: array![0.], // unused field
            max_iters: max_accept_reject_iters,
            tilting_solution: None,
        }
    }

    pub fn cdf<R: Rng + ?Sized>(&mut self, n: usize, rng: &mut R) -> (f64, f64, f64) {
        let max_iters = self.max_iters;
        assert!(self.tilting_solution.is_some());
        truncnorm::solved_mv_truncnormal_cdf(self.get_tilting_solution(None), n, max_iters, rng)
    }

    pub fn try_get_tilting_solution(&self) -> Option<&TiltingSolution> {
        self.tilting_solution.as_ref()
    }

    /// # Panics
    pub fn get_tilting_solution(
        &mut self,
        old_solution: Option<&TiltingSolution>,
    ) -> &TiltingSolution {
        if self.tilting_solution.is_none() {
            self.tilting_solution = {
                let mut problem =
                    TiltingProblem::new(self.lbs.clone(), self.ubs.clone(), self.scale.clone());
                if let Some(old_soln) = old_solution {
                    problem.with_initialization(&old_soln.x, &old_soln.mu);
                }
                Some(problem.solve_optimial_tilting())
            };
        }
        self.tilting_solution.as_ref().unwrap()
    }

    pub fn sample_n<R: Rng + ?Sized>(
        &mut self,
        n: usize,
        rng: &mut R,
    ) -> (Array2<f64>, Array1<f64>) {
        let lbs = self.lbs.clone();
        let ubs = self.ubs.clone();
        let scale = self.scale.clone();
        let max_iters = self.max_iters;
        let tilting_solution = self.get_tilting_solution(None);
        solved_mv_truncnormal_rand(tilting_solution, lbs, ubs, scale, n, max_iters, rng)
    }
}

impl Distribution<Array2<f64>> for MultivariateTruncatedNormal<Ix2> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array2<f64> {
        let tilting_solution = self.tilting_solution.as_ref().unwrap();
        solved_mv_truncnormal_rand(
            tilting_solution,
            self.lbs.clone(),
            self.ubs.clone(),
            self.scale.clone(),
            1,
            self.max_iters,
            rng,
        )
        .0
    }
}
