//! High level distributions
use crate::tilting::TiltingProblem;
use crate::tilting::TiltingSolution;
use crate::truncnorm;
use crate::truncnorm::solved_mv_truncnormal_rand;
use ndarray::Array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::Dimension;
use ndarray::{Ix1, Ix2};
use rand::prelude::Distribution;
use rand::Rng;

/// Multivariate normal distribution with diagonal covariance
pub type MultivariateTruncatedNormalDiag = MultivariateTruncatedNormal<Ix1>;

/// Truncated multivariate normal distribution with full covariance
#[derive(Clone, Debug)]
pub struct MultivariateTruncatedNormal<D: Dimension> {
    loc: Array1<f64>,
    scale: Array<f64, D>,
    lbs: Array1<f64>,
    ubs: Array1<f64>,
    max_iters: usize,
    tilting_solution: Option<TiltingSolution>,
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
            max_iters: max_accept_reject_iters,
            tilting_solution: None,
        }
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

    pub fn cdf<R: Rng + ?Sized>(&mut self, n: usize, rng: &mut R) -> (f64, f64, f64) {
        let max_iters = self.max_iters;
        truncnorm::solved_mv_truncnormal_cdf(self.get_tilting_solution(None), n, max_iters, rng)
    }

    pub fn sample_n<R: Rng + ?Sized>(&mut self, n: usize, rng: &mut R) -> Array2<f64> {
        let lbs = self.lbs.clone();
        let ubs = self.ubs.clone();
        let scale = self.scale.clone();
        let max_iters = self.max_iters;
        let tilting_solution = self.get_tilting_solution(None);
        solved_mv_truncnormal_rand(tilting_solution, lbs, ubs, scale, n, max_iters, rng)
    }
}

impl Distribution<Array1<f64>> for MultivariateTruncatedNormal<Ix2> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
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
        .index_axis(Axis(0), 0)
        .to_owned()
    }
}
