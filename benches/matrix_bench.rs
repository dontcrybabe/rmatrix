use criterion::{Criterion, black_box, criterion_group, criterion_main};

use rmatrix::matrix::*;
fn benchmark_multiply_matrix_rayon(c: &mut Criterion) {
    // Define test matrices
    let rows_a = 128;
    let cols_a_rows_b = 128;
    let cols_b = 128;

    let mut matrix_a: Vec<f64> = vec![1.0; rows_a * cols_a_rows_b];
    let mut matrix_b: Vec<f64> = vec![1.0; cols_a_rows_b * cols_b];
    matrix_a.fill(1.1);
    matrix_b.fill(2.2);
    // Benchmark the multiply_matrix function
    c.bench_function("multiply_matrix_rayon", |b| {
        b.iter(|| {
            // Run the multiply_matrix function; black_box prevents compiler optimizations
            unsafe {
                black_box(multiply_matrix_rayon(
                    &matrix_a,
                    &matrix_b,
                    rows_a,
                    cols_a_rows_b,
                    cols_b,
                ));
            }
        });
    });
}

fn benchmark_multiply_matrix_simd(c: &mut Criterion) {
    // Define test matrices
    let rows_a = 128;
    let cols_a_rows_b = 128;
    let cols_b = 128;

    let mut matrix_a: Vec<f64> = vec![1.0; rows_a * cols_a_rows_b];
    let mut matrix_b: Vec<f64> = vec![1.0; cols_a_rows_b * cols_b];
    matrix_a.fill(1.1);
    matrix_b.fill(2.2);
    // Benchmark the multiply_matrix function
    c.bench_function("multiply_matrix_simd", |b| {
        b.iter(|| {
            // Run the multiply_matrix function; black_box prevents compiler optimizations
            unsafe {
                black_box(multiply_matrix_simd(
                    &matrix_a,
                    &matrix_b,
                    rows_a,
                    cols_a_rows_b,
                    cols_b,
                ));
            }
        });
    });
}

fn benchmark_multiply_matrix_iter(c: &mut Criterion) {
    // Define test matrices
    let mut matrix1 = Matrix::new(128, 128);
    let mut matrix2 = Matrix::new(128, 128);
    matrix1.fill(1.1);
    matrix2.fill(2.2);

    // Benchmark the multiply_matrix function
    c.bench_function("multiply_matrix", |b| {
        b.iter(|| {
            // Run the multiply_matrix function; black_box prevents compiler optimizations
            unsafe { matrix1.multiply_by_matrix(&matrix2) }
        });
    });
}

fn benchmark_transpose(c: &mut Criterion) {
    // Define test matrices
    let rows_a = 128;
    let cols_a_rows_b = 128;
    let cols_b = 128;

    let mut matrix_a: Vec<f64> = vec![1.0; rows_a * cols_a_rows_b];
    let mut matrix_b: Vec<f64> = vec![1.0; cols_a_rows_b * cols_b];
    matrix_a.fill(1.1);
    matrix_b.fill(2.2);
    // Benchmark the multiply_matrix function
    c.bench_function("transpose_matrix", |b| {
        b.iter(|| {
            // Run the multiply_matrix function; black_box prevents compiler optimizations
            unsafe {
                black_box(transpose_matrix(&matrix_a, rows_a, cols_a_rows_b));
            }
        });
    });
}

// Criterion main group and entry point
criterion_group!(
    benches,
    benchmark_multiply_matrix_simd,
    benchmark_multiply_matrix_rayon,
    benchmark_multiply_matrix_iter,
    benchmark_transpose
);
criterion_main!(benches);
