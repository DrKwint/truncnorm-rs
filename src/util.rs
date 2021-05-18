use ndarray::par_azip;
use ndarray::Array1;
use ndarray::Array2;

const MAX_ITERS: usize = 10;

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
) -> (Array1<f64>, Array1<bool>) {
    let mut x: Array1<f64> = proposal_sampler();
    let mut accepted: Array1<bool> = Array1::from_elem(x.len(), false);
    let mut i = 0;
    while !accepted.fold(true, |a, b| a && *b) && i < MAX_ITERS {
        i += 1;
        let sample = proposal_sampler();
        par_azip!((x in &mut x, &s in &sample, &acc in &accepted) {
            if !acc {
                *x = s;
            }
        });
        accept_condition(&x, &mut accepted);
    }
    (x, accepted)
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
