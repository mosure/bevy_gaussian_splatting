use std::hint::black_box;

use bevy_gaussian_splatting::{
    gaussian::{
        formats::planar_3d_chunked::LodBounds,
        formats::planar_3d_lod::{
            GaussianLodBuildSettings, LodError, compare_gaussians, gaussian_support_bounds,
        },
        lod_build_gpu::{
            hierarchy::{
                GpuLodHierarchyBuilder, GpuLodHierarchyLimits, GpuLodHierarchyReductionGroup,
                GpuLodHierarchyReductionInput,
            },
            preprocess_lod_batch_cpu,
        },
    },
    io::lod_build_external::{
        ExternalLodBatchPreprocessor, GpuHierarchyExternalLodBatchPreprocessor,
    },
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
        label: Some("gaussian_lod_hierarchy_benchmark_device"),
        ..Default::default()
    }))
    .expect("RUN_GPU_LOD_BENCHMARKS=1 was set but device creation failed");
    let records = records();
    let bounds = normalization_bounds();

    // Time the complete production external preprocessor, including the CPU
    // support-bound reconstruction performed after canonical GPU readback.
    // Stopping at `sort_morton_batch` would omit work included in the CPU
    // comparator and understate the GPU-assisted path's actual stage cost.
    let mut hierarchy = GpuLodHierarchyBuilder::new(&device, GpuLodHierarchyLimits::default())
        .expect("benchmark device must support the bounded hierarchy builder");
    let settings = GaussianLodBuildSettings::default();
    let mut group = c.benchmark_group("lod/global_preprocess_sort_bounded_gpu");
    group.sample_size(10);
    for count in BATCH_COUNTS {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter(|| {
                let mut preprocessor = GpuHierarchyExternalLodBatchPreprocessor {
                    device: &device,
                    queue: &queue,
                    builder: &mut hierarchy,
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

    // These reducer microbenchmarks exercise the readable legacy ABI 6/v2 GPU
    // primitives only. The production ABI 15 builder deliberately uses the
    // CPU v3 hierarchy after GPU preprocessing/sort, so do not present these
    // numbers as an end-to-end builder speedup or quality-equivalent baseline.
    let leaf_count = *BATCH_COUNTS.last().unwrap();
    let leaf_groups = reduction_groups(leaf_count, settings.leaf_capacity as usize);
    let mut group =
        c.benchmark_group("lod/experimental_legacy_v2_leaf_reduce_gpu_dispatch_readback");
    group.sample_size(10);
    group.throughput(Throughput::Elements(leaf_count as u64));
    group.bench_function(BenchmarkId::from_parameter(leaf_count), |b| {
        b.iter(|| {
            let output = hierarchy
                .reduce_moment_merge_leaf_groups(
                    &device,
                    &queue,
                    black_box(&records[..leaf_count]),
                    &leaf_groups,
                    settings.support_sigma,
                )
                .expect("opt-in GPU global leaf reduction should succeed");
            black_box(output.len());
        });
    });
    group.finish();

    // Internal levels consume bounded summary records rather than the full
    // source. Use one representative per synthetic child, matching MomentMerge.
    let internal_count = 3_072;
    let internal_inputs = records[..internal_count]
        .iter()
        .copied()
        .map(|representative| GpuLodHierarchyReductionInput {
            representative,
            bounds: gaussian_support_bounds(&representative, settings.support_sigma)
                .expect("benchmark Gaussian support is valid"),
            inherited_error: LodError::ZERO,
        })
        .collect::<Vec<_>>();
    let internal_groups = reduction_groups(
        internal_inputs.len(),
        usize::from(settings.branching_factor),
    );
    let mut group =
        c.benchmark_group("lod/experimental_legacy_v2_internal_reduce_gpu_dispatch_readback");
    group.sample_size(10);
    group.throughput(Throughput::Elements(internal_count as u64));
    group.bench_function(BenchmarkId::from_parameter(internal_count), |b| {
        b.iter(|| {
            let output = hierarchy
                .reduce_moment_merge_summary_groups(
                    &device,
                    &queue,
                    black_box(&internal_inputs),
                    &internal_groups,
                    settings.support_sigma,
                )
                .expect("opt-in GPU global internal reduction should succeed");
            black_box(output.len());
        });
    });
    group.finish();
}

fn reduction_groups(input_count: usize, capacity: usize) -> Vec<GpuLodHierarchyReductionGroup> {
    let group_count = input_count.div_ceil(capacity);
    let base = input_count / group_count;
    let remainder = input_count % group_count;
    let mut start = 0_u32;
    (0..group_count)
        .map(|index| {
            let count = base + usize::from(index < remainder);
            let group = GpuLodHierarchyReductionGroup {
                start,
                count: count as u32,
            };
            start += count as u32;
            group
        })
        .collect()
}

criterion_group! {
    name = lod_gpu_benches;
    config = Criterion::default().sample_size(10);
    targets = cpu_oracle_benchmarks, gpu_stage_benchmarks,
}
criterion_main!(lod_gpu_benches);
