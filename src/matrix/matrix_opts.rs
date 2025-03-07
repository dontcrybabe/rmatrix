use nalgebra::SMatrix;
use rayon::prelude::*;
use std::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_mul_pd, _mm256_storeu_pd};

pub const BLOCK_SIZE: usize = 4;

pub fn transpose_matrix(matrix: &Vec<f64>, rows: usize, cols: usize) -> Vec<f64> {
    let mut transposed = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = matrix[row * cols + col];
        }
    }
    transposed
}

pub unsafe fn multiply_matrix_rayon(
    matrix_a: &Vec<f64>,
    matrix_b: &Vec<f64>,
    cols_a_rows_b: usize,
    rows_a: usize,
    cols_b: usize,
) -> Vec<f64> {
    let mut matrix_c = vec![0.0; rows_a * cols_b];
    matrix_c
        .par_chunks_mut(cols_b)
        .enumerate()
        .for_each(|(row_idx, row_c)| {
            for col_idx in 0..cols_b {
                let mut sum = 0.0;
                for k in 0..cols_a_rows_b {
                    sum += matrix_a[row_idx * cols_a_rows_b + k] * matrix_b[k * cols_b + col_idx];
                }
                row_c[col_idx] = sum;
            }
        });

    matrix_c
}

pub unsafe fn multiply_matrix_rayon_simd(
    matrix_a: &Vec<f64>,
    matrix_b: &Vec<f64>,
    rows_a: usize,
    cols_a_rows_b: usize,
    cols_b: usize,
) -> Vec<f64> {
    let mut matrix_c = vec![0.0; rows_a * cols_b];
    matrix_c
        .par_chunks_mut(cols_b)
        .enumerate()
        .for_each(|(row_idx, row_c)| {
            for col_idx in 0..cols_b {
                let mut sum = 0.0;
                for k in 0..cols_a_rows_b {
                    sum += matrix_a[row_idx * cols_a_rows_b + k] * matrix_b[k * cols_b + col_idx];
                }
                row_c[col_idx] = sum;
            }
        });

    matrix_c
}

pub unsafe fn multiply_matrix_simd(
    matrix_a: &Vec<f64>,
    matrix_b: &Vec<f64>,
    cols_a_rows_b: usize,
    rows_a: usize,
    cols_b: usize,
) -> Vec<f64> {
    let mut matrix_c = vec![0.0; rows_a * cols_b];
    let b_t = transpose_matrix(matrix_b, cols_a_rows_b, cols_b);
    if cols_a_rows_b % BLOCK_SIZE != 0 {
        for row_idx in 0..rows_a {
            for col_idx in 0..cols_b {
                let mut sum = 0.0;
                for block_start in (0..cols_a_rows_b).step_by(BLOCK_SIZE) {
                    let block_end = std::cmp::min(block_start + BLOCK_SIZE, cols_a_rows_b);

                    sum += unsafe {
                        compute_block_sum(
                            matrix_a,
                            &b_t,
                            row_idx,
                            col_idx,
                            cols_a_rows_b,
                            block_start,
                            block_end,
                        )
                    };
                }
                matrix_c[row_idx * cols_b + col_idx] = sum;
            }
        }
    } else {
        for row_idx in 0..rows_a {
            for col_idx in 0..cols_b {
                let mut sum = 0.0;
                for block_start in (0..cols_a_rows_b).step_by(BLOCK_SIZE) {
                    let block_end = block_start + BLOCK_SIZE;

                    sum += unsafe {
                        compute_block_sum(
                            matrix_a,
                            &b_t,
                            row_idx,
                            col_idx,
                            cols_a_rows_b,
                            block_start,
                            block_end,
                        )
                    };
                }
                matrix_c[row_idx * cols_b + col_idx] = sum;
            }
        }
    }
    matrix_c
}

#[inline(always)]
pub unsafe fn compute_block_sum(
    matrix_a: &Vec<f64>,
    matrix_b: &Vec<f64>,
    row_idx: usize,
    col_idx: usize,
    cols_a_rows_b: usize,
    block_start: usize,
    block_end: usize,
) -> f64 {
    let a_ptr = matrix_a.as_ptr().add(row_idx * cols_a_rows_b + block_start);
    let b_ptr = matrix_b.as_ptr().add(col_idx * cols_a_rows_b + block_start);

    let a_vec: __m256d = _mm256_loadu_pd(a_ptr);
    let b_vec: __m256d = _mm256_loadu_pd(b_ptr);

    let product = _mm256_mul_pd(a_vec, b_vec);

    let mut block_sum = [0.0; BLOCK_SIZE];
    _mm256_storeu_pd(block_sum.as_mut_ptr(), product);

    block_sum.iter().take(block_end - block_start).sum()
}

pub unsafe fn multiply_matrix_nalgebra() {
    type Matrix128x128 = SMatrix<f64, 128, 128>;

    let mut matrix1 = Matrix128x128::zeros();
    let mut matrix2 = Matrix128x128::zeros();
    matrix1.fill(1.1);
    matrix2.fill(2.2);
    let c: Matrix128x128 = matrix1 * matrix2;
    c;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matrix_simd() {
        let a: Vec<f64> = vec![1.0, 2.0];
        let b: Vec<f64> = vec![-3.0, 5.0, 4.0, -6.0];
        let cols_a_rows_b = 2;
        let rows_a = 1;
        let cols_b = 2;
        let c = unsafe { multiply_matrix_simd(&a, &b, cols_a_rows_b, rows_a, cols_b) };
        assert_eq!(c, vec![5.0, -7.0]);
        let c = unsafe { multiply_matrix_rayon(&a, &b, cols_a_rows_b, rows_a, cols_b) };
        assert_eq!(c, vec![5.0, -7.0]);
    }
}
