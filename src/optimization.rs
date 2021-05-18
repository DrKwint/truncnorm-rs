#[allow(non_snake_case)]
extern crate levenberg_marquardt;
extern crate nalgebra;

use crate::grad_psi;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::storage::Owned;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dynamic;
use nalgebra::Matrix;
use nalgebra::VecStorage;
use nalgebra::U1;
use ndarray::{Array1, Array2};

pub struct TilingProblem {
    x: DVector<f64>,
    L: Array2<f64>,
    l: Array1<f64>,
    u: Array1<f64>,
    residuals: Option<DVector<f64>>,
    jacobian: Option<DMatrix<f64>>,
}

impl TilingProblem {
    pub fn new(L: Array2<f64>, l: Array1<f64>, u: Array1<f64>) -> Self {
        let x = Array1::zeros(2 * (l.len() - 1));
        let (residuals, jacobian) = grad_psi(&x, &L, &l, &u);
        let jac_shape = jacobian.shape();
        TilingProblem {
            x: DVector::zeros(2 * (l.len() - 1)),
            L: L,
            l: l,
            u: u,
            residuals: Some(DVector::from_vec(residuals.to_vec())),
            jacobian: Some(DMatrix::from_vec(
                jac_shape[0],
                jac_shape[1],
                jacobian.into_raw_vec(),
            )),
        }
    }
    pub fn get_x(self: &Self) -> Array1<f64> {
        Array1::from_vec(self.x.data.as_vec().to_vec())
    }
}

impl<'a> LeastSquaresProblem<f64, Dynamic, Dynamic> for TilingProblem {
    type ParameterStorage = VecStorage<f64, Dynamic, U1>;
    type JacobianStorage = Owned<f64, Dynamic, Dynamic>;
    type ResidualStorage = VecStorage<f64, Dynamic, U1>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.x = x.clone();
        let x = Array1::from_vec(x.data.as_vec().to_vec());
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
