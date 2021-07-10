use ndarray::Array1;
use ndarray::Array2;
use ndarray::{azip, par_azip};

pub fn swap_rows<S: Clone>(matrix: &mut Array2<S>, a: usize, b: usize) {
    let a_temp = matrix.row(a).to_owned();
    let b_temp = matrix.row(b).to_owned();
    matrix.row_mut(a).assign(&b_temp);
    matrix.row_mut(b).assign(&a_temp);
}

pub fn swap_cols<S: Clone>(matrix: &mut Array2<S>, a: usize, b: usize) {
    let a_temp = matrix.column(a).to_owned();
    let b_temp = matrix.column(b).to_owned();
    matrix.column_mut(a).assign(&b_temp);
    matrix.column_mut(b).assign(&a_temp);
}

pub fn rejection_sample(
    accept_condition: &dyn Fn(&Array1<f64>, &mut Array1<bool>),
    proposal_sampler: &dyn Fn() -> Array1<f64>,
    max_iters: usize,
) -> Array1<f64> {
    let mut x: Array1<f64> = proposal_sampler();
    let mut accepted: Array1<bool> = Array1::from_elem(x.len(), false);
    accept_condition(&x, &mut accepted);
    let mut i = 0;
    let mut proposal = proposal_sampler();
    while !accepted.fold(true, |a, b| a && *b) && i < max_iters {
        i += 1;
        let mut is_accept = Array1::from_elem(proposal.len(), false);
        accept_condition(&proposal, &mut is_accept);
        let mut proposal_accepts = proposal
            .into_iter()
            .zip(is_accept)
            .filter(|x| x.1)
            .map(|x| x.0);
        azip!((x in &mut x, acc in &mut accepted) {
            if !*acc {
                if let Some(v) = proposal_accepts.next() {*x = v; *acc = true;}
            }
        });
        proposal = proposal_sampler()
    }
    x
}

#[allow(dead_code)]
pub fn par_rejection_sample(
    accept_condition: &dyn Fn(&Array1<f64>, &mut Array1<bool>),
    proposal_sampler: &dyn Fn() -> Array1<f64>,
    max_iters: usize,
) -> Array1<f64> {
    let mut x: Array1<f64> = proposal_sampler();
    let mut accepted: Array1<bool> = Array1::from_elem(x.len(), false);
    accept_condition(&x, &mut accepted);
    let mut i = 0;
    while !accepted.fold(true, |a, b| a && *b) && i < max_iters {
        i += 1;
        let sample = proposal_sampler();
        par_azip!((x in &mut x, &s in &sample, &acc in &accepted) {
            if !acc {
                *x = s;
            }
        });
        accept_condition(&x, &mut accepted);
    }
    x
}

#[cfg(test)]
mod tests {
    extern crate ndarray;
    extern crate test;
    use super::*;
    use ndarray::array;

    #[test]
    fn test_row_swap() {
        let mut matrix = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let new_matrix = array![[1, 2, 3], [7, 8, 9], [4, 5, 6]];
        swap_rows(&mut matrix, 1, 2);
        assert_eq!(matrix, new_matrix);
    }

    #[test]
    fn test_col_swap() {
        let mut matrix = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let new_matrix = array![[1, 3, 2], [4, 6, 5], [7, 9, 8]];
        swap_cols(&mut matrix, 1, 2);
        assert_eq!(matrix, new_matrix);
    }
}
