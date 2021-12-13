#![allow(non_snake_case)]
extern crate argmin;

use crate::dist_util::{cholperm, ln_normal_pr};
use argmin::prelude::ArgminOp;
use argmin::prelude::Error;
use argmin::prelude::*;
use argmin::solver::gaussnewton::GaussNewton;
use ndarray::{concatenate, s, Axis};
use ndarray::{Array1, Array2};
use num::traits::FloatConst;
use serde::{Deserialize, Serialize};

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
    let dfdx = P.clone().t().dot(&L.slice(s![.., ..d - 1])) - mu.slice(s![..d - 1]);
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

#[derive(Debug, Clone)]
pub struct TiltingProblem {
    d: usize,
    perm: Array1<usize>,
    x: Array1<f64>,
    L: Array2<f64>,
    l: Array1<f64>,
    u: Array1<f64>,
}

impl TiltingProblem {
    pub fn new(mut l: Array1<f64>, mut u: Array1<f64>, mut sigma: Array2<f64>) -> Self {
        // Calculate L, l, u
        let d = l.shape()[0];
        let (mut L, perm) = cholperm(&mut sigma, &mut l, &mut u);
        let D = L.diag().to_owned();
        u /= &D;
        l /= &D;
        L = (L / (Array2::<f64>::zeros([D.len(), D.len()]) + &D).t()) - Array2::<f64>::eye(d);
        Self {
            d,
            perm,
            x: Array1::zeros(2 * (l.len() - 1)),
            L,
            l,
            u,
        }
    }

    pub fn get_x(&self) -> Array1<f64> {
        self.x.clone() //Array1::from_vec(self.x.data.as_vec().clone())
    }

    pub fn with_initialization(&mut self, x: &Array1<f64>, mu: &Array1<f64>) {
        let mut vec = x.to_vec();
        vec.extend(mu.to_vec());
        if vec.len() < 2 * (self.d - 1) {
            vec.resize(2 * (self.d - 1), 0.);
        }
        self.x = Array1::from_vec(vec); //DVector::from_vec(vec);
    }

    pub fn solve_optimial_tilting(self) -> TiltingSolution {
        let solver = GaussNewton::new();
        //println!("Init: {:?}", self.x);
        let result = Executor::new(self.clone(), solver, self.x.clone())
            //.add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
            .max_iters(10)
            .run();
        //if result.is_err() {
        //    println!("TILTING RESULT ERR: {:?}", result.as_ref().err());
        //}
        let best_param = if result.is_ok() {
            result.ok().unwrap().state().get_best_param()
        } else {
            self.x
        };
        let x = best_param.slice(s![..self.d - 1]).to_owned();
        // assign saddlepoint x* and mu*
        let mu = best_param
            .slice(s![self.d - 1..(2 * (self.d - 1))])
            .to_owned();
        TiltingSolution {
            lower_tri: self.L,
            lower: self.l,
            upper: self.u,
            x,
            mu,
            permutation: self.perm,
        }
    }
}

impl ArgminOp for TiltingProblem {
    /// Type of the parameter vector
    type Param = Array1<f64>;
    /// Type of the return value computed by the cost function
    type Output = Array1<f64>;
    /// Type of the Hessian. Can be `()` if not needed.
    type Hessian = Array2<f64>;
    /// Type of the Jacobian. Can be `()` if not needed.
    type Jacobian = Array2<f64>;
    /// Floating point precision
    type Float = f64;

    /// Apply the cost function to a parameter `p`
    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let (residuals, _jacobian) = grad_psi(p, &self.L, &self.l, &self.u);
        Ok(residuals) //.mapv(|x| x * x).sum())
    }

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        let (residuals, _jacobian) = grad_psi(p, &self.L, &self.l, &self.u);
        Ok(residuals)
    }

    /// Compute the Hessian at parameter `p`.
    fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, Error> {
        let (_residuals, jacobian) = grad_psi(param, &self.L, &self.l, &self.u);
        let matrix_norm = jacobian.mapv(|x| x * x).sum().sqrt();
        let eye = Array2::<f64>::eye(jacobian.shape()[0]);
        // Trying to ensure that J.dot(J.T) is non-singular
        let stable_jac: Array2<f64> = jacobian + (eye * (matrix_norm * 1e-12));
        Ok(stable_jac)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiltingSolution {
    pub lower_tri: Array2<f64>,
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
    pub x: Array1<f64>,
    pub mu: Array1<f64>,
    pub permutation: Array1<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tilt() {
        let problem = TiltingProblem::new(Array1::zeros(10), Array1::ones(10), Array2::eye(10));
        let output = problem.solve_optimial_tilting();
        println!("output: {:?}", output);
    }
}
