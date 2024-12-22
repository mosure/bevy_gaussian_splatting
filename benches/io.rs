use criterion::{
    BenchmarkId,
    criterion_group,
    criterion_main,
    Criterion,
    Throughput,
};

use bevy_gaussian_splatting::{
    Gaussian,
    Cloud,
    io::codec::CloudCodec,
    random_gaussians,
};


const GAUSSIAN_COUNTS: [usize; 4] = [
    1000,
    10000,
    84_348,
    1_244_819,
    // 6_131_954,
];

fn gaussian_cloud_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode gaussian clouds");
    for count in GAUSSIAN_COUNTS.iter() {
        group.throughput(Throughput::Bytes(*count as u64 * std::mem::size_of::<Gaussian>() as u64));
        group.bench_with_input(
            BenchmarkId::new("decode", count),
            &count,
            |b, &count| {
                let gaussians = random_gaussians(*count);
                let bytes = gaussians.encode();

                b.iter(|| Cloud::decode(bytes.as_slice()));
            },
        );
    }
}

criterion_group!{
    name = io_benches;
    config = Criterion::default().sample_size(10);
    targets = gaussian_cloud_decode_benchmark
}
criterion_main!(io_benches);
