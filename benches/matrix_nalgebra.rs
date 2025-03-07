use criterion::{Criterion, criterion_group, criterion_main, black_box};
use rmatrix::matrix::*;

fn benchmark_nalgebra(c: &mut Criterion) {
    c.bench_function("nalgebra_mul", |b| {
        b.iter(|| {
            // Run the multiply_matrix function; black_box prevents compiler optimizations
            unsafe {
                black_box(multiply_matrix_nalgebra());
            }
        });
    });
}

// Criterion main group and entry point
criterion_group!(nalgebra, benchmark_nalgebra);
criterion_main!(nalgebra);
