use std::hint::black_box;

use bevy_gaussian_splatting::{
    gaussian::{
        formats::planar_3d_chunked::LodBounds,
        formats::planar_3d_lod::{GaussianLodBuildSettings, compare_gaussians},
        lod_build_gpu::{
            preprocess_lod_batch_cpu,
            sort::{GpuLodBatchSorter, GpuLodSortLimits},
        },
    },
    io::lod_build_external::{ExternalLodBatchPreprocessor, GpuExternalLodBatchPreprocessor},
    testing::LodTestScene,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rayon::prelude::*;

const BATCH_COUNTS: [usize; 3] = [1_024, 8_192, 65_536];
const SOURCE_INDEX_BASE: u64 = 1_000_000_000;
const SUPPORT_SIGMA: f32 = 3.0;

fn normalization_bounds() -> LodBounds {
    LodBounds::new([-1.0; 3], [1.0; 3]).expect("static benchmark bounds are valid")
}

fn records() -> Vec<bevy_gaussian_splatting::gaussian::formats::planar_3d::Gaussian3d> {
    let mut records = LodTestScene::workgroup_boundary(*BATCH_COUNTS.last().unwrap())
        .gaussians
        .into_iter()
        .map(|entry| entry.gaussian)
        .collect::<Vec<_>>();
    // Exercise the payload tiebreaker used by the real external merge instead
    // of benchmarking only the trivial distinct-Morton case. Each small group
    // shares a position/Morton code while retaining different Gaussian data.
    for group in records.chunks_mut(8) {
        let position = group[0].position_visibility.position;
        for gaussian in group {
            gaussian.position_visibility.position = position;
        }
    }
    records
}

fn cpu_oracle_benchmarks(c: &mut Criterion) {
    let records = records();
    let bounds = normalization_bounds();
    let mut group = c.benchmark_group("lod/preprocess_bounded_cpu_oracle");
    group.sample_size(10);
    for count in BATCH_COUNTS {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter(|| {
                let output = preprocess_lod_batch_cpu(
                    black_box(&records[..count]),
                    SOURCE_INDEX_BASE,
                    bounds,
                    SUPPORT_SIGMA,
                )
                .expect("bounded CPU preprocessing should succeed");
                black_box(output.records.len());
            });
        });
    }
    group.finish();

    // Match the external GPU batch contract: validate/support-bound every
    // record, compute the canonical Morton key, then sort by the exact
    // `(morton, Gaussian payload, source_index)` merge key. This is the
    // meaningful CPU baseline for `sort_morton_batch`, unlike preprocessing
    // alone above.
    let mut group = c.benchmark_group("lod/global_preprocess_sort_bounded_cpu");
    group.sample_size(10);
    for count in BATCH_COUNTS {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter(|| {
                let mut output = preprocess_lod_batch_cpu(
                    black_box(&records[..count]),
                    SOURCE_INDEX_BASE,
                    bounds,
                    SUPPORT_SIGMA,
                )
                .expect("bounded CPU preprocessing should succeed")
                .records;
                output.par_sort_unstable_by(|left, right| {
                    let left_index = usize::try_from(left.source_index - SOURCE_INDEX_BASE)
                        .expect("benchmark source index should fit usize");
                    let right_index = usize::try_from(right.source_index - SOURCE_INDEX_BASE)
                        .expect("benchmark source index should fit usize");
                    left.morton
                        .cmp(&right.morton)
                        .then_with(|| {
                            compare_gaussians(&records[left_index], &records[right_index])
                        })
                        .then_with(|| left.source_index.cmp(&right.source_index))
                });
                black_box(output.len());
            });
        });
    }
    group.finish();
}

/// Device creation and GPU work are strictly opt-in. In particular, normal
/// `cargo bench`, tests, and CI `cargo bench --no-run` never initialize wgpu.
fn gpu_stage_benchmarks(c: &mut Criterion) {
    if std::env::var("RUN_GPU_LOD_BENCHMARKS").as_deref() != Ok("1") {
        return;
    }

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .expect("RUN_GPU_LOD_BENCHMARKS=1 was set but no wgpu adapter was available");
    eprintln!("GPU LoD stage benchmark adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gaussian_lod_sort_benchmark_device"),
        ..Default::default()
    }))
    .expect("RUN_GPU_LOD_BENCHMARKS=1 was set but device creation failed");
    let records = records();
    let bounds = normalization_bounds();

    // Time the complete production external preprocessor, including the CPU
    // support-bound reconstruction performed after canonical GPU readback.
    // Stopping at `sort_morton_batch` would omit work included in the CPU
    // comparator and understate the GPU-assisted path's actual stage cost.
    let mut sorter = GpuLodBatchSorter::new(&device, GpuLodSortLimits::default())
        .expect("benchmark device must support the bounded Morton sorter");
    let settings = GaussianLodBuildSettings::default();
    let mut group = c.benchmark_group("lod/global_preprocess_sort_bounded_gpu");
    group.sample_size(10);
    for count in BATCH_COUNTS {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter(|| {
                let mut preprocessor = GpuExternalLodBatchPreprocessor {
                    device: &device,
                    queue: &queue,
                    sorter: &mut sorter,
                    settings,
                };
                let output = preprocessor
                    .preprocess(
                        black_box(&records[..count]),
                        SOURCE_INDEX_BASE,
                        bounds,
                        settings.support_sigma,
                    )
                    .expect("opt-in GPU external preprocessing should succeed");
                black_box(output.records.len());
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = lod_gpu_benches;
    config = Criterion::default().sample_size(10);
    targets = cpu_oracle_benchmarks, gpu_stage_benchmarks,
}
criterion_main!(lod_gpu_benches);
