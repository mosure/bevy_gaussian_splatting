use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use bevy_gaussian_splatting::{
    Gaussian3d, Gaussian4d, PlanarGaussian3d, PlanarGaussian4d, io::codec::CloudCodec,
    random_gaussians_3d, random_gaussians_4d,
};

const GAUSSIAN_COUNTS: [usize; 4] = [
    1000, 10000, 84_348, 1_244_819,
    // 6_131_954,
];

fn gaussian_cloud_3d_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode 3d gaussian clouds");
    for count in GAUSSIAN_COUNTS.iter() {
        group.throughput(Throughput::Bytes(
            *count as u64 * std::mem::size_of::<Gaussian3d>() as u64,
        ));
        group.bench_with_input(BenchmarkId::new("decode/3d", count), &count, |b, &count| {
            let gaussians = random_gaussians_3d(*count);
            let bytes = gaussians.encode();

            b.iter(|| PlanarGaussian3d::decode(bytes.as_slice()));
        });
    }
}

fn gaussian_cloud_4d_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode 4d gaussian clouds");
    for count in GAUSSIAN_COUNTS.iter() {
        group.throughput(Throughput::Bytes(
            *count as u64 * std::mem::size_of::<Gaussian4d>() as u64,
        ));
        group.bench_with_input(BenchmarkId::new("decode/4d", count), &count, |b, &count| {
            let gaussians = random_gaussians_4d(*count);
            let bytes = gaussians.encode();

            b.iter(|| PlanarGaussian4d::decode(bytes.as_slice()));
        });
    }
}

criterion_group! {
    name = io_benches;
    config = Criterion::default().sample_size(10);
    targets = gaussian_cloud_3d_decode_benchmark,
              gaussian_cloud_4d_decode_benchmark,
}
criterion_main!(io_benches);
