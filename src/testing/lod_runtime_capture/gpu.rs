use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU8, Ordering},
    mpsc::TrySendError,
};

use bevy::{
    app::SubApp,
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        ExtractSchedule, MainWorld, Render, RenderSystems,
        render_asset::RenderAssets,
        render_resource::{
            Buffer, BufferDescriptor, BufferId, BufferUsages, MapMode, PollType,
            TexelCopyBufferInfo, TexelCopyBufferLayout,
        },
        renderer::{RenderAdapterInfo, RenderContext, RenderDevice, RenderQueue, ViewQuery},
        texture::GpuImage,
        view::ExtractedView,
    },
};
use bevy_interleave::prelude::{GpuPlanar, GpuPlanarStorage, PlanarHandle};

use super::{
    CaptureRequest, CaptureStats, CompletedCapture, FrameRequest, outcomes::Outcome, rss_bytes,
};
use crate::{
    Gaussian3d, PlanarGaussian3dHandle,
    gaussian::formats::planar_3d::PlanarStorageGaussian3d,
    render::lod::{LodCompactionBuffers, LodCompactionLabel, LodIndirectArgs},
    render::ordered::{GlobalOrderRender, OrderedViews},
    render::point::{PointSplattingRender, PointViews},
    render::traversal::{GpuLodHierarchy, GpuLodTraversalOutputs, GpuLodTraversalRender},
    sort::radix::RadixSortLabel,
    stream::memory::LodMemoryLedger,
    stream::render_commit::LodRenderCandidates,
    testing::lod_capture::*,
    testing::lod_package_cpu::{LodPackageCpuTelemetry, PackageCpuFrame},
};

pub(super) const IMAGE_OFFSET: u64 = 512;
const INDIRECT_SIZE: u64 = std::mem::size_of::<LodIndirectArgs>() as u64;
const TIMESTAMP_OFFSET: u64 = 256;
const TIMESTAMP_COUNT: u32 = 4;

#[derive(Clone, Debug)]
struct DrawObservation {
    cloud: Entity,
    buffer: BufferId,
    generation: Option<u64>,
}

#[derive(Default)]
struct DrawProbeState {
    frame: Option<(u64, Entity)>,
    observations: Vec<DrawObservation>,
}

/// Installed only by the capture harness. The render command calls this after
/// issuing draw_indirect; indirect-buffer contents alone are not a draw proof.
#[derive(Resource, Default)]
pub struct LodRuntimeDrawProbe(Mutex<DrawProbeState>);

impl LodRuntimeDrawProbe {
    pub(crate) fn begin_capture(&self, frame: u64, camera: Entity) {
        let mut state = self.0.lock().unwrap();
        state.frame = Some((frame, camera));
        state.observations.clear();
    }

    pub(crate) fn attests_capture(
        &self,
        frame: u64,
        camera: Entity,
        cloud: Entity,
        buffer: BufferId,
        generation: Option<u64>,
    ) -> bool {
        let state = self.0.lock().unwrap();
        state.frame == Some((frame, camera))
            && state.observations.len() == 1
            && state.observations[0].cloud == cloud
            && state.observations[0].buffer == buffer
            && state.observations[0].generation == generation
    }

    pub(crate) fn observe(
        &self,
        camera: Entity,
        cloud: Entity,
        buffer: BufferId,
        generation: Option<u64>,
    ) {
        let mut state = self.0.lock().unwrap();
        if state.frame.is_some_and(|(_, expected)| camera == expected)
            && state.observations.len() < 2
        {
            state.observations.push(DrawObservation {
                cloud,
                buffer,
                generation,
            });
        }
    }
    fn reset(&self, request: Option<&FrameRequest>) {
        let mut state = self.0.lock().unwrap();
        state.frame = request.map(|request| (request.frame, request.camera));
        state.observations.clear();
    }
    fn attests(
        &self,
        request: &FrameRequest,
        cloud: Entity,
        buffer: BufferId,
        generation: Option<u64>,
    ) -> bool {
        self.attests_capture(request.frame, request.camera, cloud, buffer, generation)
    }
}

struct PendingFrame {
    request: FrameRequest,
    record: LodFrameCapture,
    indirect: bool,
    draw_attested: bool,
    timestamp_period_ns: f64,
    evidence: serde_json::Value,
}

struct CaptureSlot {
    staging: Buffer,
    queries: Option<wgpu::QuerySet>,
    query_resolve: Option<Buffer>,
    // 0 free; 1 encoded (not submitted yet); 2 mapping; 3 mapped; 4 failed.
    state: Arc<AtomicU8>,
    error: Arc<Mutex<Option<String>>>,
    pending: Option<PendingFrame>,
}

#[derive(Resource, Default)]
struct CaptureGpuState {
    slots: Vec<CaptureSlot>,
    current: Option<usize>,
    row_pitch: u32,
    staging_bytes: u64,
    cut_offset: u64,
}

#[derive(Resource, Default)]
struct ExtractedPackageCpuFrame {
    cpu: Option<PackageCpuFrame>,
    packages: Vec<serde_json::Value>,
    controllers: serde_json::Value,
}

/// This hook is installed only by the instrumented native/virtual capture.
/// It consumes the completed main-world update once at extraction, and stamps
/// it using the exact camera/frame request extracted by this same schedule.
/// First-frame absence and frames without requests intentionally emit no sample.
fn extract_package_cpu_frame(world: &mut World) {
    let (sample, packages, controllers) = {
        let mut main = world.resource_mut::<MainWorld>();
        let stamp = main
            .get_resource::<CaptureRequest>()
            .and_then(|request| request.current.as_ref())
            .map(|request| (request.frame, request.camera.to_bits()));
        main.init_resource::<LodPackageCpuTelemetry>();
        let probe = main.resource::<LodPackageCpuTelemetry>();
        let sample = match stamp {
            Some((frame, camera)) => probe.take_for_frame(frame, camera),
            None => {
                probe.take_completed();
                None
            }
        };
        let mut query = main.query::<(
            &crate::stream::package::GaussianLodPackageStatus,
            Option<&crate::stream::package::GaussianGpuLodPackageStatus>,
            Option<&crate::stream::package::GaussianLodPackageTestingSnapshot>,
        )>();
        let packages = query.iter(&main).take(2).map(|(status, gpu, work)| serde_json::json!({
            "phase":format!("{:?}",status.phase), "resident_pages":status.resident_pages,
            "terminal_failures":status.terminal_failures,
            "failure":status.failure.as_ref().map(|failure|format!("{failure:?}")),
            "gpu":gpu.map(|gpu|serde_json::json!({
                "residency_generation":gpu.residency_generation, "snapshot_pages":gpu.snapshot_pages,
                "pending_publication_pages":gpu.pending_publication_pages,
                "cutoff_unavailable":gpu.cutoff_unavailable,
                "spatial_mapping_bytes":gpu.spatial_mapping_bytes,
                "spatial_mapping_error":gpu.spatial_mapping_error,
                "visible_views":gpu.visible_views, "acknowledged_views":gpu.acknowledged_views,
                "queued_requests":gpu.queued_requests, "in_flight_requests":gpu.in_flight_requests,
                "capacity_blocked_requests":gpu.capacity_blocked_requests})),
            "work":work.map(|work|serde_json::json!({
                "available":work.runtime_work_available,
                "request_queue":work.runtime_request_queue_len,
                "in_flight":work.runtime_transport_in_flight_requests,
                "preprocess_waiting":work.preprocess_waiting_jobs,
                "preprocess_ready":work.preprocess_ready_pages,
                "capacity_blocked":work.runtime_capacity_blocked_requests})),
        })).collect();
        let controllers = stamp.map(|(_, camera)| {
            let camera = Entity::from_bits(camera);
            let point = main.get_resource::<crate::render::point::GaussianPointSplattingDiagnostics>()
                .and_then(|diagnostics|diagnostics.get(camera));
            let budget = main.get_resource::<crate::render::point::GaussianPointSplattingViewBudgetDiagnostics>()
                .and_then(|diagnostics|diagnostics.get(camera));
            serde_json::json!({
                "scope":"latest_completed_controller_observations_at_extraction;submission_ids_may_lag_captured_image",
                "point":point.map(|point|serde_json::json!({"submission":point.submission,
                    "gpu_ms":point.gpu_ms, "samples_per_pixel":point.samples_per_pixel,
                    "availability":format!("{:?}",point.availability), "error":point.error})),
                "view":budget.map(|budget|serde_json::json!({"submission":budget.submission,
                    "view_gpu_ms":budget.view_gpu_ms,"selected_gaussian_limit":budget.selected_gaussian_limit,
                    "action":format!("{:?}",budget.action), "target_unmet":budget.target_unmet})),
            })
        }).unwrap_or_default();
        (sample, packages, controllers)
    };
    world.insert_resource(ExtractedPackageCpuFrame {
        cpu: sample,
        packages,
        controllers,
    });
}

pub(super) fn install(app: &mut SubApp) {
    app.init_resource::<CaptureGpuState>()
        .init_resource::<ExtractedPackageCpuFrame>()
        .add_systems(ExtractSchedule, extract_package_cpu_frame)
        .init_resource::<LodRuntimeDrawProbe>()
        .add_systems(
            Core3d,
            begin_frame
                .before(LodCompactionLabel)
                .before(GpuLodTraversalRender)
                .before(PointSplattingRender)
                .before(GlobalOrderRender)
                .before(RadixSortLabel)
                .before(Core3dSystems::Prepass),
        )
        .add_systems(
            Core3d,
            timestamp_after_compaction
                .after(LodCompactionLabel)
                .before(RadixSortLabel),
        )
        .add_systems(
            Core3d,
            timestamp_after_sort
                .after(RadixSortLabel)
                .before(Core3dSystems::Prepass),
        )
        .add_systems(
            Core3d,
            timestamp_after_traversal
                .after(GpuLodTraversalRender)
                .before(PointSplattingRender)
                .before(GlobalOrderRender),
        )
        .add_systems(
            Core3d,
            timestamp_after_shared_renderer
                .after(PointSplattingRender)
                .after(GlobalOrderRender)
                .before(bevy::core_pipeline::upscaling::upscaling),
        )
        .add_systems(
            Core3d,
            // PostProcess produces the intermediate view texture. Upscaling
            // performs the final write into frame.target, even at 1:1 scale.
            // The copy and final timestamp must follow that write in the
            // RenderContext command-buffer dependency order.
            copy_frame.after(bevy::core_pipeline::upscaling::upscaling),
        )
        .add_systems(
            Render,
            collect_readbacks
                .in_set(RenderSystems::Cleanup)
                .after(RenderSystems::Render),
        );
}

fn begin_frame(
    mut context: RenderContext,
    view: ViewQuery<&'static ExtractedView>,
    request: Res<CaptureRequest>,
    device: Res<RenderDevice>,
    mut state: ResMut<CaptureGpuState>,
    probe: Res<LodRuntimeDrawProbe>,
) {
    request
        .stats
        .lock()
        .unwrap()
        .observe_renderer_loop(std::time::Instant::now());
    let view = view.into_inner();
    if request
        .current
        .as_ref()
        .is_some_and(|r| r.camera != view.retained_view_entity.main_entity.id())
    {
        return;
    }
    state.current = None;
    probe.reset(request.current.as_ref());
    let Some(frame) = &request.current else {
        return;
    };
    if request
        .stats
        .lock()
        .unwrap()
        .outcomes
        .as_ref()
        .is_some_and(|outcomes| !outcomes.can_encode(frame.frame))
    {
        return;
    }
    if state.slots.is_empty() {
        state.row_pitch =
            RenderDevice::align_copy_bytes_per_row(request.viewport[0] as usize * 4) as u32;
        state.cut_offset = IMAGE_OFFSET
            + if request.capture_images {
                u64::from(state.row_pitch) * u64::from(request.viewport[1])
            } else {
                0
            };
        state.staging_bytes = state.cut_offset
            + u64::from(request.cut_capacity) * super::hierarchy::SELECTED_RANGE_BYTES;
        let timestamps = device.features().contains(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        );
        for _ in 0..request.slots {
            let staging = device.create_buffer(&BufferDescriptor {
                label: Some("lod_capture_bounded_readback"),
                size: state.staging_bytes,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let queries = timestamps.then(|| {
                device
                    .wgpu_device()
                    .create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("lod_capture_stage_timestamps"),
                        ty: wgpu::QueryType::Timestamp,
                        count: TIMESTAMP_COUNT,
                    })
            });
            let query_resolve = timestamps.then(|| {
                device.create_buffer(&BufferDescriptor {
                    label: Some("lod_capture_query_resolve"),
                    size: 256,
                    usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            });
            state.slots.push(CaptureSlot {
                staging,
                queries,
                query_resolve,
                state: Default::default(),
                error: Default::default(),
                pending: None,
            });
        }
    }
    let Some(index) = state
        .slots
        .iter()
        .position(|slot| slot.state.load(Ordering::Acquire) == 0)
    else {
        let mut stats = request.stats.lock().unwrap();
        stats.dropped_ring_full += 1;
        stats.finish_request(frame.frame, Outcome::RingFull);
        return;
    };
    state.current = Some(index);
    if let Some(queries) = &state.slots[index].queries {
        context.command_encoder().write_timestamp(queries, 0);
    }
    let _ = frame;
}

fn timestamp_after_compaction(
    mut context: RenderContext,
    state: Res<CaptureGpuState>,
    request: Res<CaptureRequest>,
) {
    if request.pipeline.uses_gpu_hierarchy() {
        return;
    }
    if let Some(index) = state.current
        && let Some(queries) = &state.slots[index].queries
    {
        context.command_encoder().write_timestamp(queries, 1);
    }
}
fn timestamp_after_sort(
    mut context: RenderContext,
    state: Res<CaptureGpuState>,
    request: Res<CaptureRequest>,
) {
    if request.pipeline.uses_gpu_hierarchy() {
        return;
    }
    if let Some(index) = state.current
        && let Some(queries) = &state.slots[index].queries
    {
        context.command_encoder().write_timestamp(queries, 2);
    }
}

fn timestamp_after_traversal(
    mut context: RenderContext,
    state: Res<CaptureGpuState>,
    request: Res<CaptureRequest>,
) {
    if request.pipeline.uses_gpu_hierarchy()
        && let Some(index) = state.current
        && let Some(queries) = &state.slots[index].queries
    {
        context.command_encoder().write_timestamp(queries, 1);
    }
}

fn timestamp_after_shared_renderer(
    mut context: RenderContext,
    state: Res<CaptureGpuState>,
    request: Res<CaptureRequest>,
) {
    if request.pipeline.uses_gpu_hierarchy()
        && let Some(index) = state.current
        && let Some(queries) = &state.slots[index].queries
    {
        context.command_encoder().write_timestamp(queries, 2);
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn copy_frame(
    mut context: RenderContext,
    view: ViewQuery<&'static ExtractedView>,
    request: Res<CaptureRequest>,
    package_cpu: Res<ExtractedPackageCpuFrame>,
    mut state: ResMut<CaptureGpuState>,
    buffers: Res<LodCompactionBuffers<Gaussian3d>>,
    (points, ordered): (Res<PointViews>, Res<OrderedViews>),
    traversals: Res<GpuLodTraversalOutputs>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    gpu_clouds: Res<RenderAssets<PlanarStorageGaussian3d>>,
    ledger: Option<Res<LodMemoryLedger>>,
    atlas_generations: Res<crate::stream::atlas_upload::LodAtlasGpuGenerations>,
    clouds: Query<(
        Entity,
        &PlanarGaussian3dHandle,
        Option<&LodRenderCandidates>,
        Option<&GpuLodHierarchy>,
    )>,
    (adapter, device, ordered_diagnostics): (
        Res<RenderAdapterInfo>,
        Res<RenderDevice>,
        Res<crate::render::ordered::GaussianGlobalOrderDiagnostics>,
    ),
    queue: Res<RenderQueue>,
    probe: Res<LodRuntimeDrawProbe>,
) {
    let Some(index) = state.current.take() else {
        return;
    };
    let Some(frame) = request.current.clone() else {
        return;
    };
    let view = view.into_inner();
    let Some(image) = gpu_images.get(&frame.target) else {
        let mut stats = request.stats.lock().unwrap();
        stats.missing_drawable += 1;
        stats.finish_request(
            frame.frame,
            Outcome::MissingDrawable {
                reason: "capture target is not GPU-ready".into(),
            },
        );
        return;
    };
    let row_pitch = state.row_pitch;
    let cut_offset = state.cut_offset;
    let ring_bytes = state
        .slots
        .iter()
        .map(|slot| {
            slot.staging.size()
                + slot
                    .query_resolve
                    .as_ref()
                    .map_or(0, |buffer| buffer.size())
        })
        .sum::<u64>();
    let slot = &mut state.slots[index];
    let mut selected = 0;
    let mut candidates_count = 0;
    let mut output_capacity = 0;
    let mut generation = 0;
    let mut copied_indirect = false;
    let mut draw_attested = false;
    let mut renderer_gpu_bytes = 0;
    let mut evidence =
        serde_json::json!({"frame": frame.frame, "source": "no_drawable", "seam": "no_candidate"});
    // This focused harness owns one cloud and one view. Reject ambiguity rather
    // than adding indirect counts from unrelated consumers into one record.
    if request.pipeline.uses_gpu_hierarchy() {
        let inputs = clouds.iter().filter_map(|(entity, handle, _, hierarchy)| {
            let hierarchy = hierarchy?;
            (hierarchy.0.source == handle.handle().id()).then_some((
                entity,
                hierarchy.0.source,
                hierarchy.0.generation,
            ))
        });
        let capture = match request.pipeline {
            LodCapturePipeline::HierarchyPoint => {
                super::point::capture(view.retained_view_entity, &points, &traversals, inputs)
            }
            LodCapturePipeline::HierarchyOrdered => {
                super::ordered::capture(view.retained_view_entity, &ordered, &traversals, inputs)
            }
            _ => unreachable!(),
        };
        match capture {
            Ok(capture) => {
                output_capacity = u64::from(capture.capacity);
                generation = capture.generation;
                renderer_gpu_bytes = capture.gpu_bytes;
                evidence = capture.evidence;
                evidence["frame"] = serde_json::json!(frame.frame);
                context.command_encoder().copy_buffer_to_buffer(
                    &capture.renderer,
                    0,
                    &slot.staging,
                    0,
                    capture.renderer_header_bytes,
                );
                context.command_encoder().copy_buffer_to_buffer(
                    &capture.traversal,
                    0,
                    &slot.staging,
                    capture.renderer_header_bytes,
                    64,
                );
                if request.cut_capacity > 0 {
                    let (source_offset, capacity) = capture.selected_range;
                    if capacity > request.cut_capacity {
                        request.stats.lock().unwrap().fail_request(
                            frame.frame,
                            "hierarchy cut exceeds capture readback admission".to_owned(),
                        );
                        return;
                    }
                    context.command_encoder().copy_buffer_to_buffer(
                        &capture.traversal,
                        source_offset,
                        &slot.staging,
                        cut_offset,
                        u64::from(capacity) * super::hierarchy::SELECTED_RANGE_BYTES,
                    );
                    evidence["hierarchy_cut"] = serde_json::json!({
                        "capacity":capacity,
                        "index":"hierarchy_index.json",
                        "schema_version":2,
                        "stride_bytes":super::hierarchy::SELECTED_RANGE_BYTES,
                        "layout":super::hierarchy::SELECTED_RANGE_LAYOUT,
                        "omission_policy":super::hierarchy::OMISSION_POLICY,
                        "omission_enabled":evidence["omission_enabled"],
                        "scope":"same-submission complete resident logical cut; zero-count rows preserve source ownership and page residency without expanding records; physical omission does not alter logical selection or complete-cohort demand",
                        "logical_source_coverage":"requires independent pinned-manifest validation; physical readback alone does not attest source domains"
                    });
                }
                copied_indirect = true;
                // This receipt proves the draw consumed the current hierarchy.
                // The completed header still has to pass overflow validation.
                draw_attested = true;
            }
            Err(reason) => evidence["seam"] = serde_json::json!(reason),
        }
    }
    for (cloud, handle, candidates, _) in clouds
        .iter()
        .filter(|_| !request.pipeline.uses_gpu_hierarchy())
    {
        if request.pipeline == LodCapturePipeline::FlatSource {
            let Some(gpu_cloud) = gpu_clouds.get(handle.handle()) else {
                continue;
            };
            if candidates.is_some() {
                request.stats.lock().unwrap().fail_request(
                    frame.frame,
                    "flat capture unexpectedly has LoD candidates".to_owned(),
                );
                return;
            }
            if copied_indirect {
                request.stats.lock().unwrap().fail_request(
                    frame.frame,
                    "multiple flat clouds in one capture view".to_owned(),
                );
                return;
            }
            selected = gpu_cloud.len() as u64;
            candidates_count = selected;
            output_capacity = selected;
            let indirect = gpu_cloud.draw_indirect_buffer();
            draw_attested = probe.attests(&frame, cloud, indirect.id(), None);
            context
                .command_encoder()
                .copy_buffer_to_buffer(indirect, 0, &slot.staging, 0, 16);
            copied_indirect = true;
            evidence = serde_json::json!({"frame": frame.frame, "pipeline": "flat_source",
                "draw_command_attested": draw_attested, "source": "post_render_same_submission_copy",
                "generation_scope": "no_hierarchy_publication", "compaction": "not_applicable"});
            continue;
        }
        let Some(candidate) = candidates.and_then(|c| c.get(frame.camera)) else {
            continue;
        };
        selected = candidate.frontier().quality_status().active_gaussians;
        candidates_count = u64::from(candidate.frontier().candidate_count());
        evidence["candidate_phase"] = serde_json::json!(candidate.phase.load(Ordering::Acquire));
        evidence["hard_fallback_requested"] =
            serde_json::json!(candidate.render_hard_fallback_requested());
        evidence["atlas_current"] = serde_json::json!(atlas_generations.frontier_is_current(
            handle.handle().id().untyped(),
            candidate.required_atlas_ranges()
        ));
        evidence["candidate_count"] = serde_json::json!(candidate.frontier().candidate_count());
        let allocated = buffers.get(view.retained_view_entity, cloud, handle.handle().id());
        evidence["seam"] = serde_json::json!("no_ready_compaction");
        evidence["allocated_readiness"] =
            serde_json::json!(allocated.map(|state| format!("{:?}", state.readiness())));
        evidence["handshake_stage"] =
            serde_json::json!(allocated.map(|state| state.candidate_handshake_stage_for_testing()));
        evidence["allocated_capacity"] =
            serde_json::json!(allocated.map(|state| state.output_capacity()));
        evidence["allocated_last_radix"] = serde_json::json!(
            allocated
                .and_then(|state| state.last_radix_drawable_for_testing(candidate))
                .map(|state| format!("{state:?}"))
        );
        let Some(compaction) =
            buffers.get_ready(view.retained_view_entity, cloud, handle.handle().id())
        else {
            continue;
        };
        evidence["seam"] = serde_json::json!("no_last_radix_drawable");
        let Some(drawable) = compaction.last_radix_drawable_for_testing(candidate) else {
            continue;
        };
        evidence["seam"] = serde_json::json!("no_latched_selection");
        let Some(drawable_selected) = drawable.selected_gaussians else {
            continue;
        };
        if copied_indirect {
            request.stats.lock().unwrap().fail_request(
                frame.frame,
                "multiple drawables in one capture view".to_owned(),
            );
            return;
        }
        selected = drawable_selected;
        candidates_count = u64::from(drawable.rendered_candidate_count);
        output_capacity = u64::from(compaction.output_capacity());
        generation = drawable.compaction_generation;
        draw_attested = probe.attests(
            &frame,
            cloud,
            compaction.indirect_args_buffer.id(),
            Some(generation),
        );
        context.command_encoder().copy_buffer_to_buffer(
            &compaction.indirect_args_buffer,
            0,
            &slot.staging,
            0,
            INDIRECT_SIZE,
        );
        copied_indirect = true;
        evidence = serde_json::json!({"frame": frame.frame, "compaction_generation": generation,
            "compute_input_generation": drawable.compute_input_generation,
            "radix_publication_generation": drawable.radix_publication_generation,
            "candidate_fingerprint_primary": drawable.candidate_fingerprint_primary,
            "candidate_fingerprint_secondary": drawable.candidate_fingerprint_secondary,
            "atlas_allocation_epoch": drawable.candidate_atlas_allocation_epoch,
            "complete_package_frontier": candidates.is_some_and(|candidates| candidates.candidate_draw_required),
            "draw_command_attested": draw_attested, "source": "post_render_same_submission_copy"});
    }
    let limits = device.limits();
    evidence["device_limits"] = serde_json::json!({
        "max_storage_buffer_binding_size":limits.max_storage_buffer_binding_size,
        "max_buffer_size":limits.max_buffer_size,
        "max_compute_workgroups_per_dimension":limits.max_compute_workgroups_per_dimension,
    });
    evidence["ordered_admission"] = serde_json::json!(ordered_diagnostics.get(frame.camera).map(
        |status| serde_json::json!({
            "scope":"current_frame_admission;completed_counters_may_lag",
            "ready":status.ready,"error":status.error,"submission":status.submission,
            "gpu_bytes":status.gpu_bytes,"projected_gaussians":status.projected_gaussians,
        })
    ));
    evidence["path_frame"] = serde_json::json!(frame.path_frame);
    evidence["package_snapshot"] = serde_json::json!({
        "scope":"main_world_at_same_frame_extraction;GPU_feedback_acknowledgements_may_lag_this_submission",
        "packages":package_cpu.packages,
    });
    if !copied_indirect {
        let mut stats = request.stats.lock().unwrap();
        stats.missing_drawable += 1;
        stats.last_drawable_diagnostic = Some(evidence.clone());
        // Loading failures must not write thousands of empty screenshots. A
        // fixed initial sample preserves evidence; subsequent attempts update
        // the bounded diagnostic in capture_status.json without GPU readback.
        if stats.missing_drawable > 4 {
            stats.finish_request(
                frame.frame,
                Outcome::MissingDrawable {
                    reason: evidence["seam"]
                        .as_str()
                        .unwrap_or("no attested drawable")
                        .to_owned(),
                },
            );
            return;
        }
    }
    if copied_indirect && !draw_attested {
        request.stats.lock().unwrap().unattested_draws += 1;
    }
    if draw_attested {
        request.stats.lock().unwrap().attested_submissions += 1;
    }
    if request.pipeline.uses_gpu_hierarchy() {
        evidence["controllers"] = package_cpu.controllers.clone();
    }
    if let Some(ledger) = ledger {
        let memory = ledger.snapshot();
        evidence["owned_capacity_reservations"] = serde_json::json!({
            "scope": "shared_LoD_admitted_owned_capacities;not_RSS_or_device_memory;overlaps_private_buffer_snapshot",
            "cpu_bytes": memory.cpu_bytes, "gpu_bytes": memory.gpu_bytes,
            "total_bytes": memory.total_bytes, "allocations": memory.allocations,
            "limits": {"max_cpu_bytes": memory.limits.max_cpu_bytes, "max_gpu_bytes": memory.limits.max_gpu_bytes},
            "categories": memory.categories.iter().map(|category| serde_json::json!({
                "category": format!("{:?}", category.category), "bytes": category.bytes,
                "allocations": category.allocations})).collect::<Vec<_>>()});
    }
    let stamp = LodCaptureStamp {
        run_id: request.run_id.clone(),
        view_id: frame.camera.to_bits().to_string(),
        frame: frame.frame,
        generation,
    };
    let mut identity = request.identity.clone();
    identity.backend = match adapter.backend {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gl",
        _ => "unsupported",
    }
    .to_owned();
    identity.adapter = Some(adapter.name.clone());
    identity.driver = Some(format!("{} {}", adapter.driver, adapter.driver_info));
    let memory = buffers.memory_snapshot();
    let gpu_memory = std::collections::BTreeMap::from([
        (
            "other".to_owned(),
            LodCaptureAllocation {
                reserved: memory.live_buffer_bytes + ring_bytes + renderer_gpu_bytes,
                used: memory.live_buffer_bytes + ring_bytes + renderer_gpu_bytes,
            },
        ),
        (
            "retired".to_owned(),
            LodCaptureAllocation {
                reserved: memory.retired_buffer_bytes,
                used: memory.retired_buffer_bytes,
            },
        ),
    ]);
    // Missing categories are intentional: this partial observation does not
    // claim a complete atlas/CPU/backend memory audit.
    let mut cpu_ms = std::collections::BTreeMap::from([(
        "main_to_render_encode".to_owned(),
        frame.started.elapsed().as_secs_f64() * 1000.0,
    )]);
    if let Some(cpu) = package_cpu
        .cpu
        .as_ref()
        .filter(|cpu| cpu.frame == frame.frame && cpu.camera == frame.camera.to_bits())
    {
        cpu_ms.extend(cpu.sample.cpu_ms.clone());
        evidence["package_cpu_work"] = serde_json::json!({
            "stamp": &stamp,
            "scope_calls": &cpu.sample.calls,
            "destination_cache_hits": cpu.sample.destination_cache_hits,
            "destination_compilations": cpu.sample.destination_compilations,
            "canonical_visited_nodes": cpu.sample.canonical_visited_nodes,
        });
    }
    let record = LodFrameCapture {
        schema_version: LOD_CAPTURE_SCHEMA_VERSION,
        mode: LodCaptureMode::NativeGpu,
        identity,
        stamp: stamp.clone(),
        scenario: frame.scenario.clone(),
        camera: LodCaptureCamera {
            world_to_view: view
                .world_from_view
                .to_matrix()
                .inverse()
                .to_cols_array()
                .map(f64::from),
            projection: view.clip_from_view.to_cols_array().map(f64::from),
            viewport: [view.viewport.z, view.viewport.w],
            pixel_scale: 1.0,
        },
        counts: LodCaptureCounts {
            stamp: stamp.clone(),
            pipeline: request.pipeline,
            source: if copied_indirect {
                LodCountSource::GpuReadback
            } else {
                LodCountSource::CpuSelection
            },
            selected,
            transition_extra: candidates_count.saturating_sub(selected),
            candidates: candidates_count,
            output_capacity,
            compacted: None,
            drawn: None,
        },
        timings: Some(LodCaptureTimings {
            stamp: stamp.clone(),
            frame_wall_ms: None,
            cpu_ms,
            gpu_ms: Default::default(),
        }),
        memory: Some(LodCaptureMemory {
            stamp,
            cpu: Default::default(),
            gpu: gpu_memory,
            process_rss_bytes: rss_bytes(),
            device_used_bytes: None,
        }),
        image: None,
    };
    if let Some(queries) = &slot.queries {
        context.command_encoder().write_timestamp(queries, 3);
        let resolve = slot.query_resolve.as_ref().unwrap();
        context
            .command_encoder()
            .resolve_query_set(queries, 0..TIMESTAMP_COUNT, resolve, 0);
        context.command_encoder().copy_buffer_to_buffer(
            resolve,
            0,
            &slot.staging,
            TIMESTAMP_OFFSET,
            u64::from(TIMESTAMP_COUNT) * 8,
        );
    }
    if request.capture_images {
        context.command_encoder().copy_texture_to_buffer(
            image.texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &slot.staging,
                layout: TexelCopyBufferLayout {
                    offset: IMAGE_OFFSET,
                    bytes_per_row: Some(row_pitch),
                    rows_per_image: None,
                },
            },
            image.texture_descriptor.size,
        );
    }
    let frame_id = frame.frame;
    slot.pending = Some(PendingFrame {
        request: frame,
        record,
        indirect: copied_indirect,
        draw_attested,
        timestamp_period_ns: f64::from(queue.get_timestamp_period()),
        evidence,
    });
    slot.state.store(1, Ordering::Release);
    let mut stats = request.stats.lock().unwrap();
    stats.submitted += 1;
    if let Some(outcomes) = &mut stats.outcomes
        && let Err(error) = outcomes.encoded(frame_id)
    {
        stats.mapping_errors.push(error.to_owned());
    }
}

fn collect_readbacks(
    mut state: ResMut<CaptureGpuState>,
    device: Res<RenderDevice>,
    request: Res<CaptureRequest>,
) {
    // Non-blocking poll only. No wait for current-frame GPU completion occurs.
    if let Err(error) = device.poll(PollType::Poll) {
        let mut stats = request.stats.lock().unwrap();
        for slot in &state.slots {
            if let Some(pending) = &slot.pending {
                stats.finish_request(
                    pending.request.frame,
                    Outcome::MappingFailure {
                        error: error.to_string(),
                    },
                );
            }
        }
        stats.mapping_errors.push(error.to_string());
        return;
    }
    let row_pitch = state.row_pitch as usize;
    let cut_offset = state.cut_offset as usize;
    for slot in &mut state.slots {
        match slot.state.load(Ordering::Acquire) {
            1 => {
                let status = slot.state.clone();
                let error_slot = slot.error.clone();
                slot.state.store(2, Ordering::Release);
                slot.staging
                    .slice(..)
                    .map_async(MapMode::Read, move |result| {
                        if let Err(error) = result {
                            *error_slot.lock().unwrap() = Some(error.to_string());
                            status.store(4, Ordering::Release);
                        } else {
                            status.store(3, Ordering::Release);
                        }
                    });
            }
            3 => {
                let mut pending = slot.pending.take().unwrap();
                let data = slot.staging.slice(..).get_mapped_range();
                if pending.indirect && pending.record.counts.pipeline.uses_gpu_hierarchy() {
                    let (valid, traversal, projected, has_work) = match pending
                        .record
                        .counts
                        .pipeline
                    {
                        LodCapturePipeline::HierarchyPoint => {
                            let words = bytemuck::pod_read_unaligned::<[u32; 24]>(&data[..96]);
                            let config =
                                request.point_gpu.as_ref().expect("validated point capture");
                            let samples =
                                pending.evidence["samples_per_pixel"].as_u64().unwrap_or(0);
                            let samples_valid = if config.target_gpu_ms.is_some() {
                                (1..=u64::from(crate::render::point::GAUSSIAN_POINT_SPLATTING_MAX_SAMPLES_PER_PIXEL))
                                    .contains(&samples)
                            } else {
                                samples == u64::from(config.samples_per_pixel)
                            };
                            let valid = super::point::valid_counts(
                                &words,
                                config,
                                pending.record.counts.output_capacity,
                            ) && samples_valid;
                            pending.evidence["point_image_attested"] = serde_json::json!(valid);
                            pending.evidence["point"] = serde_json::json!({
                                "projected_gaussians":words[5], "requested_points":words[3],
                                "dispatched_points":if words[4] == 0 {words[3]} else {0}, "flags":words[4]});
                            (
                                valid,
                                <[u32; 16]>::try_from(&words[8..]).unwrap(),
                                words[5],
                                words[3] > 0,
                            )
                        }
                        LodCapturePipeline::HierarchyOrdered => {
                            let words = bytemuck::pod_read_unaligned::<[u32; 34]>(&data[..136]);
                            let config = request
                                .ordered_gpu
                                .as_ref()
                                .expect("validated ordered capture");
                            let valid = super::ordered::valid_counts(
                                &words,
                                config,
                                pending.record.counts.output_capacity,
                            );
                            pending.evidence["ordered_image_attested"] = serde_json::json!(valid);
                            pending.evidence["ordered"] = serde_json::json!({
                                "vertex_count":words[0], "projected_gaussians":words[1],
                                "flags":words[12], "traversal_failure":words[13],
                                "spatial_transition_edges":words[7],
                                "spatial_transition_records":words[14],
                                "spatial_transition_flags":words[15],
                                "spatial_required_edges":words[16],
                                "spatial_required_records":words[17],
                                "spatial_band_unavailable":words[15] & 16 != 0});
                            (
                                valid,
                                <[u32; 16]>::try_from(&words[18..]).unwrap(),
                                words[1],
                                words[0] > 0,
                            )
                        }
                        _ => unreachable!(),
                    };
                    pending.record.counts.selected = u64::from(traversal[8]);
                    pending.record.counts.candidates = u64::from(traversal[8]);
                    pending.record.counts.compacted = valid.then_some(u64::from(projected));
                    pending.record.counts.drawn = pending.record.counts.compacted;
                    pending.evidence["counts_valid"] = serde_json::json!(valid);
                    pending.evidence["traversal"] =
                        super::hierarchy::traversal_evidence(&traversal);
                    if request.cut_capacity > 0 {
                        let count = traversal[5];
                        let capacity = pending.evidence["hierarchy_cut"]["capacity"]
                            .as_u64()
                            .unwrap_or(0);
                        let rows = (valid && u64::from(count) <= capacity).then(|| {
                            let stride = super::hierarchy::SELECTED_RANGE_BYTES as usize;
                            data[cut_offset..cut_offset + count as usize * stride]
                                .chunks_exact(stride)
                                .map(bytemuck::pod_read_unaligned::<[u32; 4]>)
                                .collect::<Vec<_>>()
                        });
                        let cut_valid = rows.as_ref().is_some_and(|rows| {
                            super::hierarchy::valid_selected_ranges(
                                rows,
                                traversal[8],
                                pending.evidence["omission_enabled"] == true,
                            )
                        });
                        pending.evidence["hierarchy_cut"]["valid"] = serde_json::json!(cut_valid);
                        if cut_valid {
                            let rows = rows.unwrap();
                            pending.evidence["hierarchy_cut"]["omitted_nodes"] =
                                serde_json::json!(rows.iter().filter(|row| row[2] == 0).count());
                            pending.evidence["hierarchy_cut"]["physical_records"] =
                                serde_json::json!(traversal[8]);
                            pending.evidence["hierarchy_cut"]["selected"] = serde_json::json!(rows);
                        }
                    }
                    let mut stats = request.stats.lock().unwrap();
                    if !valid {
                        stats.attested_submissions = stats.attested_submissions.saturating_sub(1);
                        stats.unattested_draws += 1;
                    }
                    if stats.observe_draw_readback(valid && has_work, 1, projected) {
                        let observed = std::time::Instant::now();
                        let CaptureStats {
                            startup_timing,
                            renderer_loop_timing,
                            ..
                        } = &mut *stats;
                        for startup in startup_timing
                            .iter_mut()
                            .chain(renderer_loop_timing.iter_mut())
                        {
                            startup.observe(
                                &pending.record.counts,
                                &pending.request.scenario,
                                pending.request.started,
                                observed,
                                true,
                            );
                        }
                    }
                } else if pending.indirect {
                    let instance_count = u32::from_le_bytes(data[4..8].try_into().unwrap());
                    let vertex_count = u32::from_le_bytes(data[0..4].try_into().unwrap());
                    if pending.record.counts.pipeline == LodCapturePipeline::Hierarchy {
                        pending.record.counts.compacted = Some(u64::from(instance_count));
                    }
                    if pending.draw_attested {
                        pending.record.counts.drawn = Some(u64::from(instance_count));
                    }
                    let complete_package_frontier = pending.evidence["complete_package_frontier"]
                        == true
                        && pending.record.counts.pipeline == LodCapturePipeline::Hierarchy
                        && bytemuck::pod_read_unaligned::<LodIndirectArgs>(
                            &data[..INDIRECT_SIZE as usize],
                        )
                        .overflow_count
                            == 0;
                    {
                        let mut stats = request.stats.lock().unwrap();
                        if stats.observe_draw_readback(
                            pending.draw_attested,
                            vertex_count,
                            instance_count,
                        ) {
                            let observed = std::time::Instant::now();
                            let CaptureStats {
                                startup_timing,
                                renderer_loop_timing,
                                ..
                            } = &mut *stats;
                            for startup in startup_timing
                                .iter_mut()
                                .chain(renderer_loop_timing.iter_mut())
                            {
                                startup.observe(
                                    &pending.record.counts,
                                    &pending.request.scenario,
                                    pending.request.started,
                                    observed,
                                    complete_package_frontier,
                                );
                            }
                        }
                    }
                    pending.evidence["indirect"] = serde_json::json!({"vertex_count": vertex_count,
                        "instance_count": instance_count});
                    if pending.record.counts.pipeline == LodCapturePipeline::Hierarchy {
                        let args = bytemuck::pod_read_unaligned::<LodIndirectArgs>(
                            &data[..INDIRECT_SIZE as usize],
                        );
                        pending.evidence["indirect"]["candidate_hits"] =
                            serde_json::json!(args.candidate_hits);
                        pending.evidence["indirect"]["overflow_count"] =
                            serde_json::json!(args.overflow_count);
                    }
                }
                if slot.queries.is_some() {
                    let times: Vec<_> = data[TIMESTAMP_OFFSET as usize
                        ..TIMESTAMP_OFFSET as usize + TIMESTAMP_COUNT as usize * 8]
                        .chunks_exact(8)
                        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                        .collect();
                    if times.windows(2).all(|pair| pair[1] >= pair[0]) {
                        let timings = pending.record.timings.as_mut().unwrap();
                        let stages = if pending.record.counts.pipeline.uses_gpu_hierarchy() {
                            [
                                ("gpu_hierarchy", 0, 1),
                                (
                                    if pending.record.counts.pipeline
                                        == LodCapturePipeline::HierarchyPoint
                                    {
                                        "point_backend"
                                    } else {
                                        "ordered_backend"
                                    },
                                    1,
                                    2,
                                ),
                                ("postprocess", 2, 3),
                                ("view_total", 0, 3),
                            ]
                        } else {
                            [
                                ("compaction", 0, 1),
                                ("sort", 1, 2),
                                ("raster_and_postprocess", 2, 3),
                                ("view_total", 0, 3),
                            ]
                        };
                        for (name, start, end) in stages {
                            if name == "compaction"
                                && pending.record.counts.pipeline == LodCapturePipeline::FlatSource
                            {
                                continue;
                            }
                            timings.gpu_ms.insert(
                                name.to_owned(),
                                (times[end] - times[start]) as f64 * pending.timestamp_period_ns
                                    / 1_000_000.0,
                            );
                        }
                    } else {
                        pending.evidence["timestamp_error"] =
                            serde_json::json!("nonmonotonic timestamps");
                    }
                }
                let width = pending.record.camera.viewport[0] as usize * 4;
                let height = pending.record.camera.viewport[1] as usize;
                let mut rgba = Vec::with_capacity(if request.capture_images {
                    width * height
                } else {
                    0
                });
                for row in data[IMAGE_OFFSET as usize..cut_offset]
                    .chunks_exact(row_pitch)
                    .take(height)
                {
                    rgba.extend_from_slice(&row[..width]);
                }
                drop(data);
                slot.staging.unmap();
                slot.state.store(0, Ordering::Release);
                let completed = CompletedCapture {
                    record: pending.record,
                    rgba,
                    frame_wall_ms: pending.request.frame_wall_ms,
                    evidence: pending.evidence,
                };
                let frame_id = completed.record.stamp.frame;
                match request.sender.try_send(completed) {
                    Ok(()) => {}
                    Err(error) => {
                        let error = match error {
                            TrySendError::Full(_) => "bounded capture result channel full",
                            TrySendError::Disconnected(_) => "capture consumer disconnected",
                        }
                        .to_owned();
                        let mut stats = request.stats.lock().unwrap();
                        stats.finish_request(
                            frame_id,
                            Outcome::DeliveryFailure {
                                error: error.clone(),
                            },
                        );
                        stats.mapping_errors.push(error);
                    }
                }
            }
            4 => {
                let error = slot
                    .error
                    .lock()
                    .unwrap()
                    .take()
                    .unwrap_or_else(|| "unknown map error".to_owned());
                let mut stats = request.stats.lock().unwrap();
                if let Some(pending) = slot.pending.take() {
                    stats.finish_request(
                        pending.request.frame,
                        Outcome::MappingFailure {
                            error: error.clone(),
                        },
                    );
                }
                stats.mapping_errors.push(error);
                slot.staging.unmap();
                slot.state.store(0, Ordering::Release);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_copy_depends_on_the_actual_final_target_writer() {
        use bevy::ecs::schedule::{IntoSystemSet, NodeId};

        // Inspect the installed schedule rather than a duplicate test order.
        // No device or render resources are initialized by this graph check.
        let mut app = SubApp::new();
        install(&mut app);
        let schedules = app.world().resource::<bevy::ecs::schedule::Schedules>();
        let schedule = schedules.get(Core3d).unwrap();
        let graph = schedule.graph();
        // Bevy omits both system and set names without its `debug` feature.
        // Identify the real function sets by type, then inspect their graph
        // members and the installed dependency rather than display strings.
        let writer_set = graph
            .system_sets
            .get_key(
                bevy::core_pipeline::upscaling::upscaling
                    .into_system_set()
                    .intern(),
            )
            .expect("capture must reference the real final-target writer");
        let copy_set = graph
            .system_sets
            .get_key(copy_frame.into_system_set().intern())
            .expect("installed capture copy function set");
        let copy_nodes = graph
            .hierarchy()
            .graph()
            .edges(NodeId::Set(copy_set))
            .map(|(_, member)| member)
            .collect::<Vec<_>>();
        assert_eq!(
            copy_nodes.len(),
            1,
            "one actual copy system must be installed"
        );
        let edges = graph
            .dependency()
            .graph()
            .all_edges()
            .map(|(before, after)| {
                (
                    before,
                    graph.get_node_name(&before),
                    after,
                    graph.get_node_name(&after),
                )
            })
            .collect::<Vec<_>>();
        assert!(
            graph
                .dependency()
                .graph()
                .contains_edge(NodeId::Set(writer_set), copy_nodes[0]),
            "PostProcess alone does not order a capture after the final target write; writer={writer_set:?}, copy={copy_nodes:?}, actual dependency edges={edges:#?}"
        );
    }
}
