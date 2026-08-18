use std::hint::black_box;

use bevy::math::Vec3;
use bevy_gaussian_splatting::{
    gaussian::{
        formats::planar_3d_lod::{
            CpuGaussianLodBuilder, GaussianLodBuildSettings, PlanarGaussian3dLod,
        },
        lod_settings::{GaussianLodSettings, GaussianStreamingSettings},
    },
    io::{
        lod::{LodCodecLimits, decode_manifest, decode_page, encode_manifest, encode_page},
        lod_build_external::{ExternalLodBuildConfig, ExternalLodBuildPlan},
    },
    random_gaussians_3d_seeded,
    stream::{
        atlas_upload::LodAtlasUploadQueue,
        cache::AtlasSlot,
        hierarchy::{AllResident, LodView, ManifestLodHierarchy, select_frontier},
        runtime::LodStreamingRuntime,
        transport::MemoryPageTransport,
    },
    testing::{LodTestScene, VirtualCityScene},
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const BUILD_COUNTS: [usize; 3] = [1_024, 8_192, 65_536];
const QUALITY_SWEEP: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const VIRTUAL_PAGE_COUNTS: [u32; 4] = [256, 1_024, 4_096, 16_384];
const EXTERNAL_PLAN_COUNTS: [u64; 3] = [10_000_000, 100_000_001, 250_000_000];
const RUNTIME_FIXTURE_LEVELS: u32 = 5;
const ATLAS_SLOTS_PER_UPLOAD: u32 = 64;
const ATLAS_CHURN_CASES: [(u32, u32); 2] = [(64, 4), (1_024, 8)];

#[derive(Clone)]
struct RuntimeBenchmarkFixture {
    manifest: bevy_gaussian_splatting::gaussian::formats::planar_3d_lod::GaussianLodManifest,
    transport: MemoryPageTransport,
    settings: GaussianLodSettings,
    streaming: GaussianStreamingSettings,
    view: LodView,
    source_count: usize,
}

fn build_settings() -> GaussianLodBuildSettings {
    GaussianLodBuildSettings {
        branching_factor: 8,
        leaf_capacity: 256,
        support_sigma: 3.0,
    }
}

/// A fully materialized, deterministic hierarchy used only by CPU runtime
/// benchmarks. Five nested octant levels keep both the resident page set and
/// the expanded candidate list substantial while preserving fixed bounds.
#[allow(clippy::field_reassign_with_default)]
fn runtime_benchmark_fixture() -> RuntimeBenchmarkFixture {
    let scene = LodTestScene::nested_octants(RUNTIME_FIXTURE_LEVELS);
    let source_count = scene.gaussians.len();
    let mut lod = CpuGaussianLodBuilder::new(GaussianLodBuildSettings {
        branching_factor: 8,
        leaf_capacity: 32,
        support_sigma: 3.0,
    })
    .build(&scene.cloud())
    .expect("runtime benchmark LoD build should succeed");

    // This fixture intentionally exercises thousands of bounded pages rather
    // than a source-sized virtual allocation. Keep that scale visible if a
    // builder change accidentally makes the benchmark trivial.
    assert!(lod.manifest.header.page_count >= 1_000);

    let mut transport = MemoryPageTransport::default();
    for page in &lod.pages {
        let encoded = encode_page(page).expect("runtime benchmark page should encode");
        let descriptor = lod
            .manifest
            .pages
            .iter_mut()
            .find(|descriptor| descriptor.id == page.id)
            .expect("each runtime benchmark page should have a descriptor");
        descriptor.storage = Some(
            bevy_gaussian_splatting::gaussian::formats::planar_3d_chunked::LodPageStorage {
                uri: format!("memory://runtime-benchmark-{}", page.id.0),
                byte_range: None,
                encoded_len: encoded.len() as u64,
            },
        );
        transport.insert(page.id, encoded);
    }
    lod.validate()
        .expect("runtime benchmark manifest should validate");

    let page_count = lod.manifest.header.page_count;
    let mut settings = GaussianLodSettings::default();
    settings.budgets.max_active_gaussians = source_count as u64;
    settings.budgets.max_resident_gaussians = lod.manifest.header.stored_gaussian_count;
    settings.budgets.max_resident_bytes = 256 * 1024 * 1024;
    settings.budgets.max_resident_pages = page_count;
    settings.budgets.max_pending_requests = page_count;
    settings.budgets.max_requests_per_frame = page_count;
    settings.budgets.max_upload_bytes_per_frame = 256 * 1024 * 1024;
    settings.budgets.max_traversal_nodes_per_view = lod.manifest.header.node_count;
    let streaming = GaussianStreamingSettings {
        max_concurrent_requests: page_count,
        ..Default::default()
    };

    RuntimeBenchmarkFixture {
        manifest: lod.manifest,
        transport,
        settings,
        streaming,
        view: LodView::perspective(
            Vec3::new(0.0, 0.0, 24.0),
            1_080.0,
            60.0_f32.to_radians(),
            0.01,
        ),
        source_count,
    }
}

fn fully_resident_runtime(
    fixture: &RuntimeBenchmarkFixture,
    settings: &GaussianLodSettings,
) -> LodStreamingRuntime<MemoryPageTransport> {
    let mut runtime = LodStreamingRuntime::new(
        fixture.manifest.clone(),
        fixture.transport.clone(),
        settings,
        &fixture.streaming,
    )
    .expect("runtime benchmark should construct");
    // The runtime deliberately loads a resident *cut*, not every page in the
    // manifest: at quality zero only the root is needed. Request completion is
    // therefore the correct steady-state condition for the selected quality.
    // Node count provides a manifest-bounded guard if a future scheduler needs
    // one admission round per hierarchy level or request batch.
    let maximum_admission_frames = usize::try_from(fixture.manifest.header.node_count)
        .expect("benchmark manifest node count should fit usize")
        .saturating_add(1);
    for _ in 0..maximum_admission_frames {
        let frame = runtime
            .update(fixture.view, settings, &fixture.streaming)
            .expect("runtime benchmark should make bounded progress");
        if frame.frontier().requested_nodes.is_empty() {
            return runtime;
        }
    }
    panic!("runtime benchmark fixture should admit the requested resident cut");
}

fn reference_build_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("lod/cpu_reference_build");
    group.sample_size(10);
    for count in BUILD_COUNTS {
        let cloud = random_gaussians_3d_seeded(count, 0x10d0_0000 ^ count as u64);
        let builder = CpuGaussianLodBuilder::new(build_settings());
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                let output = builder
                    .build(black_box(&cloud))
                    .expect("reference LoD build should succeed");
                black_box((
                    output.manifest.header.source_gaussian_count,
                    output.manifest.header.node_count,
                    output.manifest.header.stored_gaussian_count,
                ));
            });
        });
    }
    group.finish();
}

#[allow(clippy::field_reassign_with_default)]
fn traversal_quality_benchmarks(c: &mut Criterion) {
    let source_count = 65_536;
    let cloud = random_gaussians_3d_seeded(source_count, 0x10d0_7a6e);
    let lod = CpuGaussianLodBuilder::new(build_settings())
        .build(&cloud)
        .expect("traversal benchmark LoD build should succeed");
    let hierarchy = ManifestLodHierarchy::new(&lod.manifest)
        .expect("benchmark manifest should adapt to traversal");
    let view = LodView::perspective(
        Vec3::new(0.0, 0.0, 32.0),
        1080.0,
        60.0_f32.to_radians(),
        0.01,
    );

    let mut group = c.benchmark_group("lod/traversal_quality_sweep");
    for quality in QUALITY_SWEEP {
        let mut settings = GaussianLodSettings::default();
        settings.quality = quality;
        settings.budgets.max_active_gaussians = source_count as u64;
        settings.budgets.max_traversal_nodes_per_view = lod.manifest.header.node_count.max(1);
        group.throughput(Throughput::Elements(lod.manifest.header.node_count as u64));
        group.bench_with_input(
            BenchmarkId::new("quality_percent", (quality * 100.0) as u32),
            &quality,
            |b, _| {
                b.iter(|| {
                    let frontier = select_frontier(
                        black_box(&hierarchy),
                        &AllResident,
                        black_box(view),
                        black_box(&settings),
                    )
                    .expect("reference traversal should succeed");
                    black_box((
                        frontier.nodes.len(),
                        frontier.status.active_gaussians,
                        frontier.status.visited_nodes,
                    ));
                });
            },
        );
    }
    group.finish();
}

fn codec_benchmarks(c: &mut Criterion) {
    let source_count = 16_384;
    let cloud = random_gaussians_3d_seeded(source_count, 0x10d0_c0de);
    let lod: PlanarGaussian3dLod = CpuGaussianLodBuilder::new(build_settings())
        .build(&cloud)
        .expect("codec benchmark LoD build should succeed");
    let encoded_manifest =
        encode_manifest(&lod.manifest).expect("manifest encoding should succeed");
    let page = lod
        .pages
        .iter()
        .max_by_key(|page| page.gaussians.len())
        .expect("non-empty benchmark should have pages");
    let encoded_page = encode_page(page).expect("page encoding should succeed");
    let limits = LodCodecLimits {
        max_manifest_bytes: encoded_manifest.len() as u64,
        max_nodes: lod.manifest.header.node_count,
        max_pages: lod.manifest.header.page_count,
        max_page_bytes: encoded_page.len() as u64,
        max_page_gaussians: page.gaussians.len() as u32,
    };

    let mut group = c.benchmark_group("lod/codec");
    group.throughput(Throughput::Elements(source_count as u64));
    group.bench_function("manifest_encode", |b| {
        b.iter(|| {
            black_box(encode_manifest(black_box(&lod.manifest)).expect("manifest encode"));
        });
    });
    group.bench_function("manifest_decode", |b| {
        b.iter(|| {
            let manifest =
                decode_manifest(black_box(&encoded_manifest), limits).expect("manifest decode");
            black_box((manifest.header.node_count, manifest.header.page_count));
        });
    });

    group.throughput(Throughput::Elements(page.gaussians.len() as u64));
    group.bench_function("page_encode", |b| {
        b.iter(|| {
            black_box(encode_page(black_box(page)).expect("page encode"));
        });
    });
    group.bench_function("page_decode", |b| {
        b.iter(|| {
            let decoded = decode_page(black_box(&encoded_page), limits).expect("page decode");
            black_box(decoded.gaussians.len());
        });
    });
    group.finish();
}

fn virtual_page_generation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("lod/virtual_city_page_generation");
    group.sample_size(10);
    for gaussian_count in VIRTUAL_PAGE_COUNTS {
        let scene = VirtualCityScene {
            seed: 0x47a5_51a7_d15c_1a5e,
            page_count: 2,
            gaussians_per_page: gaussian_count,
            grid_width: 2,
        };
        group.throughput(Throughput::Elements(u64::from(gaussian_count)));
        group.bench_with_input(
            BenchmarkId::from_parameter(gaussian_count),
            &gaussian_count,
            |b, _| {
                b.iter(|| {
                    let page = black_box(scene)
                        .generate_page(black_box(1))
                        .expect("virtual benchmark page should exist");
                    black_box((page.len(), page.last().map(|entry| entry.stable_id)));
                });
            },
        );
    }
    group.finish();
}

/// Steady-state CPU orchestration after every deterministic in-memory page has
/// been admitted. This isolates per-frame hierarchy selection, pin updates,
/// and physical-range construction from codec, transport, and GPU work.
fn runtime_steady_state_benchmarks(c: &mut Criterion) {
    let fixture = runtime_benchmark_fixture();
    let mut group = c.benchmark_group("lod/runtime_steady_state_selection");
    group.sample_size(10);
    group.throughput(Throughput::Elements(fixture.source_count as u64));

    for quality in QUALITY_SWEEP {
        let mut settings = fixture.settings.clone();
        settings.quality = quality;
        let mut runtime = fully_resident_runtime(&fixture, &settings);
        let warm_frame = runtime
            .update(fixture.view, &settings, &fixture.streaming)
            .expect("resident runtime selection should succeed");
        assert!(warm_frame.has_complete_resident_cut());

        group.bench_with_input(
            BenchmarkId::new("quality_percent", (quality * 100.0) as u32),
            &quality,
            |b, _| {
                b.iter(|| {
                    let frame = runtime
                        .update(
                            black_box(fixture.view),
                            black_box(&settings),
                            black_box(&fixture.streaming),
                        )
                        .expect("resident runtime selection should succeed");
                    black_box((
                        frame.frontier().nodes.len(),
                        frame.candidate_count(),
                        frame.cache_stats().resident_pages,
                    ));
                });
            },
        );
    }
    group.finish();
}

/// Measures the bounded physical-range copy performed at the renderer-facing
/// stable-frontier boundary. Runtime selection and all page admission are
/// deliberately completed before Criterion starts timing; no candidate-sized
/// vector is constructed.
fn runtime_candidate_frontier_benchmarks(c: &mut Criterion) {
    let fixture = runtime_benchmark_fixture();
    let mut group = c.benchmark_group("lod/runtime_stable_frontier_range_copy");
    group.sample_size(10);

    for quality in QUALITY_SWEEP {
        let mut settings = fixture.settings.clone();
        settings.quality = quality;
        let mut runtime = fully_resident_runtime(&fixture, &settings);
        let frame = runtime
            .update(fixture.view, &settings, &fixture.streaming)
            .expect("resident candidate frontier should succeed");
        let candidate_limit = u32::try_from(frame.candidate_count())
            .expect("fixture candidates should fit the explicit API bound");
        assert!(frame.has_complete_resident_cut());
        group.throughput(Throughput::Elements(u64::from(candidate_limit)));

        group.bench_with_input(
            BenchmarkId::new("quality_percent", (quality * 100.0) as u32),
            &quality,
            |b, _| {
                b.iter(|| {
                    let frontier = frame
                        .candidate_frontier(black_box(candidate_limit))
                        .expect("resident candidate frontier should freeze ranges");
                    black_box((frontier.physical_ranges().len(), frontier.candidate_count()));
                });
            },
        );
    }
    group.finish();
}

/// Repeatedly dirties fixed-stride atlas slots without extracting them into a
/// render world. The queue remains populated between samples so every timed
/// write exercises last-write-wins coalescing instead of allocation or GPU IO.
fn atlas_upload_queue_churn_benchmarks(c: &mut Criterion) {
    let atlas = bevy::asset::AssetId::default();
    let mut group = c.benchmark_group("lod/atlas_upload_queue_coalescing");
    group.sample_size(10);

    for (slot_count, writes_per_slot) in ATLAS_CHURN_CASES {
        let dirty_slots: Vec<u32> = (0..slot_count).collect();
        let mut queue = LodAtlasUploadQueue::default();
        for &index in &dirty_slots {
            queue
                .enqueue_slot(
                    atlas,
                    AtlasSlot {
                        index,
                        generation: 1,
                    },
                    ATLAS_SLOTS_PER_UPLOAD,
                )
                .expect("benchmark dirty slot should be valid");
        }
        assert_eq!(queue.queued_slot_count(), dirty_slots.len());
        group.throughput(Throughput::Elements(
            u64::from(slot_count) * u64::from(writes_per_slot),
        ));

        group.bench_with_input(
            BenchmarkId::new("slots_x_writes", format!("{slot_count}x{writes_per_slot}")),
            &(slot_count, writes_per_slot),
            |b, _| {
                b.iter(|| {
                    for generation in 1..=writes_per_slot {
                        for &index in &dirty_slots {
                            queue
                                .enqueue_slot(
                                    atlas,
                                    AtlasSlot { index, generation },
                                    ATLAS_SLOTS_PER_UPLOAD,
                                )
                                .expect("benchmark dirty slot should coalesce");
                        }
                    }
                    black_box(queue.queued_slot_count());
                });
            },
        );
    }
    group.finish();
}

/// Capacity planning stays lazy even for virtual sources larger than 100M.
/// This measures only checked run/merge/hierarchy descriptor arithmetic and
/// never allocates a source-sized record vector.
fn external_build_plan_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("lod/external_build_plan_lazy");
    for source_count in EXTERNAL_PLAN_COUNTS {
        group.bench_with_input(
            BenchmarkId::from_parameter(source_count),
            &source_count,
            |b, &source_count| {
                b.iter(|| {
                    let plan = ExternalLodBuildPlan::new(
                        black_box(source_count),
                        black_box(ExternalLodBuildConfig::default()),
                    )
                    .expect("default external plan should admit the benchmark count");
                    black_box((
                        plan.initial_run_count,
                        plan.merge_pass_count,
                        plan.total_node_count,
                        plan.maximum_records_per_batch_buffer,
                        plan.maximum_spill_host_bytes,
                    ));
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = lod_benches;
    config = Criterion::default().sample_size(10);
    targets = reference_build_benchmarks,
              traversal_quality_benchmarks,
              codec_benchmarks,
              virtual_page_generation_benchmarks,
              runtime_steady_state_benchmarks,
              runtime_candidate_frontier_benchmarks,
              atlas_upload_queue_churn_benchmarks,
              external_build_plan_benchmarks,
}
criterion_main!(lod_benches);
