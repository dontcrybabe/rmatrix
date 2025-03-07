extern crate nalgebra;
use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::SMatrix;

fn benchmark_nalgebra(c: &mut Criterion) {
    type Matrix128x128 = SMatrix<f64, 128, 128>;

    // Define test matrices
    let mut matrix1 = Matrix128x128::zeros();
    let mut matrix2 = Matrix128x128::zeros();
    matrix1.fill(1.1);
    matrix2.fill(2.2);
    // Benchmark the multiply_matrix function
    c.bench_function("nalgebra_mul", |b| {
        b.iter(|| {
            // Run the multiply_matrix function; black_box prevents compiler optimizations
            unsafe { matrix1 * matrix2 }
        });
    });
}

// Criterion main group and entry point
criterion_group!(nalgebra, benchmark_nalgebra);
criterion_main!(nalgebra);
