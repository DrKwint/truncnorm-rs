#![feature(test)]
#![feature(destructuring_assignment)]
#![allow(non_snake_case)]
//! `truncnorm` provides (potentially) high-dimensional multivariate Normal
//! and TruncatedNormal distributions as well as low level binding to Gaussian
//! error functions.
//!
//! I've initially written all this code for my dissertation work. I've put
//! some effort into correctness and speed, but both could surely be improved.
//! Rely on this code at your own risk as no guarantees can be made about it.
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate ndarray_stats;
extern crate statrs;

pub mod distributions;
mod faddeeva;
pub mod truncnorm;
mod util;

/// `erf`/`erfc` family of error functions
///
/// Uses bindings to the [faddeeva](http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package)
/// C++ package and [statrs](https://crates.io/crates/statrs)
pub mod gauss {
    pub use crate::faddeeva::{erf, erfc, erfcx};
    pub use statrs::function::erf::erfc_inv;
}
