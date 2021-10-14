use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use truncnorm::truncnorm::ln_normal_pr;

fn criterion_benchmark(c: &mut Criterion) {
    let lower: Array1<f64> = Array1::zeros(3);
    c.bench_function("ln_normal_pr", |b| {
        b.iter(|| ln_normal_pr(&lower, &Array1::ones(3)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
