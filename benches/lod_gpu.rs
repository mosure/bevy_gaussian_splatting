use std::hint::black_box;

use bevy_gaussian_splatting::{
    gaussian::{
        formats::planar_3d_chunked::LodBounds,
        formats::planar_3d_lod::{GaussianLodBuildSettings, LodError, gaussian_support_bounds},
        lod_build_gpu::{
            hierarchy::{
                GpuLodHierarchyBuilder, GpuLodHierarchyLimits, GpuLodHierarchyReductionGroup,
                GpuLodHierarchyReductionInput,
            },
            preprocess_lod_batch_cpu,
        },
    },
    testing::LodTestScene,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const BATCH_COUNTS: [usize; 3] = [1_024, 8_192, 65_536];
const SOURCE_INDEX_BASE: u64 = 1_000_000_000;
const SUPPORT_SIGMA: f32 = 3.0;

fn normalization_bounds() -> LodBounds {
    LodBounds::new([-1.0; 3], [1.0; 3]).expect("static benchmark bounds are valid")
}

fn records() -> Vec<bevy_gaussian_splatting::gaussian::formats::planar_3d::Gaussian3d> {
    LodTestScene::workgroup_boundary(*BATCH_COUNTS.last().unwrap())
        .gaussians
        .into_iter()
        .map(|entry| entry.gaussian)
        .collect()
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
}

/// Device creation and GPU work are strictly opt-in. In particular, normal
/// `cargo bench`, tests, and CI `cargo bench --no-run` never initialize wgpu.
fn gpu_hierarchy_benchmarks(c: &mut Criterion) {
    if std::env::var("RUN_GPU_LOD_BENCHMARKS").as_deref() != Ok("1") {
        return;
    }

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .expect("RUN_GPU_LOD_BENCHMARKS=1 was set but no wgpu adapter was available");
    eprintln!(
        "GPU LoD hierarchy benchmark adapter: {:?}",
        adapter.get_info()
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gaussian_lod_hierarchy_benchmark_device"),
        ..Default::default()
    }))
    .expect("RUN_GPU_LOD_BENCHMARKS=1 was set but device creation failed");
    let records = records();
    let bounds = normalization_bounds();

    // Global/external builds use sort-only batches before the file merge.
    let mut hierarchy = GpuLodHierarchyBuilder::new(&device, GpuLodHierarchyLimits::default())
        .expect("benchmark device must support the bounded hierarchy builder");
    let settings = GaussianLodBuildSettings::default();
    let mut group = c.benchmark_group("lod/global_sort_bounded_gpu_dispatch_readback");
    group.sample_size(10);
    for count in BATCH_COUNTS {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter(|| {
                let output = hierarchy
                    .sort_morton_batch(
                        &device,
                        &queue,
                        black_box(&records[..count]),
                        SOURCE_INDEX_BASE,
                        bounds,
                        settings.support_sigma,
                    )
                    .expect("opt-in GPU global sort should succeed");
                black_box(output.len());
            });
        });
    }
    group.finish();

    // Time the actual post-merge leaf reducer with many explicit globally
    // planned groups in one bounded dispatch/readback.
    let leaf_count = *BATCH_COUNTS.last().unwrap();
    let leaf_groups = reduction_groups(leaf_count, settings.leaf_capacity as usize);
    let mut group = c.benchmark_group("lod/global_leaf_reduce_gpu_dispatch_readback");
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
    let mut group = c.benchmark_group("lod/global_internal_reduce_gpu_dispatch_readback");
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
    targets = cpu_oracle_benchmarks, gpu_hierarchy_benchmarks,
}
criterion_main!(lod_gpu_benches);
