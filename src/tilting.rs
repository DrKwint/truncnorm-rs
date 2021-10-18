#![allow(non_snake_case)]
extern crate levenberg_marquardt;
extern crate nalgebra;

use crate::dist_util::{cholperm, ln_normal_pr};
use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::storage::Owned;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dynamic;
use nalgebra::Matrix;
use nalgebra::VecStorage;
use nalgebra::U1;
use ndarray::{concatenate, s, Axis};
use ndarray::{Array1, Array2};
use num::traits::FloatConst;

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

#[derive(Debug, Clone)]
pub struct TiltingProblem {
    d: usize,
    sigma: Array2<f64>,
    perm: Array1<usize>,
    x: DVector<f64>,
    L: Array2<f64>,
    l: Array1<f64>,
    u: Array1<f64>,
    residuals: Option<DVector<f64>>,
    jacobian: Option<DMatrix<f64>>,
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

        let x = Array1::zeros(2 * (l.len() - 1));
        let (residuals, jacobian) = grad_psi(&x, &L, &l, &u);
        let jac_shape = jacobian.shape();

        Self {
            d,
            sigma,
            perm,
            x: DVector::zeros(2 * (l.len() - 1)),
            L,
            l,
            u,
            residuals: Some(DVector::from_vec(residuals.to_vec())),
            jacobian: Some(DMatrix::from_vec(
                jac_shape[0],
                jac_shape[1],
                jacobian.into_raw_vec(),
            )),
        }
    }

    pub fn get_x(&self) -> Array1<f64> {
        Array1::from_vec(self.x.data.as_vec().clone())
    }

    pub fn with_initialization(&mut self, x: &Array1<f64>, mu: &Array1<f64>) {
        let mut vec = x.to_vec();
        vec.extend(mu.to_vec());
        self.x = DVector::from_vec(vec);
    }

    pub fn solve_optimial_tilting(self) -> TiltingSolution {
        let (result, _report) = LevenbergMarquardt::new()
            .with_ftol(1e-2)
            .with_xtol(1e-2)
            .minimize(self.clone());
        //println!("{:?}", _report);

        let x = result.get_x().slice(s![..self.d - 1]).to_owned();
        // assign saddlepoint x* and mu*
        let mu = result
            .get_x()
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

impl<'a> LeastSquaresProblem<f64, Dynamic, Dynamic> for TiltingProblem {
    type ParameterStorage = VecStorage<f64, Dynamic, U1>;
    type JacobianStorage = Owned<f64, Dynamic, Dynamic>;
    type ResidualStorage = VecStorage<f64, Dynamic, U1>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.x = x.clone();
        let x = Array1::from_vec(x.data.as_vec().clone());
        let (residuals, jacobian) = grad_psi(&x, &self.L, &self.l, &self.u);
        let jac_shape = jacobian.shape();
        self.residuals = Some(DVector::from_vec(residuals.to_vec()));
        self.jacobian = Some(DMatrix::from_vec(
            jac_shape[0],
            jac_shape[1],
            jacobian.into_raw_vec(),
        ));
    }

    fn params(&self) -> DVector<f64> {
        self.x.clone()
    }

    fn residuals(&self) -> Option<Matrix<f64, Dynamic, U1, Self::ResidualStorage>> {
        self.residuals.clone()
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dynamic, Dynamic, Self::JacobianStorage>> {
        self.jacobian.clone()
    }
}

#[derive(Debug, Clone)]
pub struct TiltingSolution {
    pub lower_tri: Array2<f64>,
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
    pub x: Array1<f64>,
    pub mu: Array1<f64>,
    pub permutation: Array1<usize>,
}
