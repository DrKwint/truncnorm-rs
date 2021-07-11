//! High level distributions
use crate::truncnorm;
use ndarray::array;
use ndarray::Array;
use ndarray::Array2;
use ndarray::Dimension;
use ndarray::Zip;
use ndarray::{Array1, ArrayView1};
use ndarray::{Ix1, Ix2};
use ndarray_linalg::cholesky::Cholesky;
use ndarray_linalg::eigh::*;
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
	pub fn new(loc: Array1<f64>, scale: Array2<f64>) -> Self {
		MultivariateNormal { loc, scale }
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
	low: Array1<f64>,
	high: Array1<f64>,
	log_normalizer: Array1<f64>,
	max_iters: usize,
}

impl MultivariateTruncatedNormal<Ix1> {
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
		let log_normalizer = truncnorm::ln_normal_pr(
			&((&lower_bounds - &loc) / &scale),
			&((&upper_bounds - &loc) / &scale),
		);
		Self {
			loc,
			scale,
			low: lower_bounds,
			high: upper_bounds,
			log_normalizer,
			max_iters: max_accept_reject_iters,
		}
	}

	pub fn log_prob(&self, x: ArrayView1<f64>) -> f64 {
		let std_x = (&x - &self.loc) / &self.scale;
		let halfrtln2pi = 0.5 * (f64::TAU()).ln();
		Zip::from(&std_x)
			.and(&self.scale)
			.and(&self.log_normalizer)
			.par_map_collect(|&x, &s, &lnz| {
				-(0.5 * x.powi(2) + halfrtln2pi + s.ln() - lnz)
			})
			.sum()
	}
}

impl Distribution<Array1<f64>> for MultivariateTruncatedNormal<Ix1> {
	fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> Array1<f64> {
		let X = truncnorm::trandn(
			&((&self.low - &self.loc) / &self.scale),
			&((&self.high - &self.loc) / &self.scale),
			self.max_iters,
		);
		&self.loc + &self.scale * X
	}
}

impl MultivariateTruncatedNormal<Ix2> {
	pub fn new(
		loc: Array1<f64>,
		scale: Array2<f64>,
		low: Array1<f64>,
		high: Array1<f64>,
		max_accept_reject_iters: usize,
	) -> Self {
		Self {
			loc,
			scale,
			low,
			high,
			log_normalizer: array![0.], // unused field
			max_iters: max_accept_reject_iters,
		}
	}
}

impl Distribution<Array2<f64>> for MultivariateTruncatedNormal<Ix2> {
	fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> Array2<f64> {
		truncnorm::mv_truncnormal_rand(
			self.low.clone(),
			self.high.clone(),
			self.scale.clone(),
			1,
			self.max_iters,
		)
		.0
	}
}
