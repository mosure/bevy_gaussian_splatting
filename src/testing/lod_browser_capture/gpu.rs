use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU8, Ordering},
};

use bevy::{
    app::SubApp,
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        Render, RenderSystems,
        render_resource::{Buffer, BufferDescriptor, BufferId, BufferUsages, MapMode},
        renderer::{RenderAdapterInfo, RenderContext, RenderDevice, ViewQuery},
        view::ExtractedView,
    },
};
use bevy_interleave::prelude::PlanarHandle;
use serde_json::{Value, json};

use super::{BrowserCaptureRequest, BrowserFrame, BrowserPointGpuConfig, memory_json};
use crate::{
    Gaussian3d, PlanarGaussian3dHandle,
    render::{
        lod::{LodCompactionBuffers, LodCompactionLabel, LodIndirectArgs},
        point::{PointSplattingRender, PointViews},
        traversal::{GpuLodHierarchy, GpuLodTraversalOutputs, GpuLodTraversalRender},
    },
    stream::{memory::LodMemoryLedger, render_commit::LodRenderCandidates},
};

pub(super) const READBACK_BYTES: u64 = 96;

#[derive(Default)]
struct ProbeState {
    stamp: Option<(u64, Entity)>,
    draws: Vec<(Entity, BufferId, Option<u64>)>,
}

/// Installed only by the browser qualification runner. A command attestation
/// is recorded immediately after the renderer issues its indirect draw.
#[derive(Resource, Default)]
pub struct BrowserLodDrawProbe(Mutex<ProbeState>);

impl BrowserLodDrawProbe {
    pub(crate) fn observe(
        &self,
        camera: Entity,
        cloud: Entity,
        buffer: BufferId,
        generation: Option<u64>,
    ) {
        let mut state = self.0.lock().unwrap();
        if state.stamp.is_some_and(|(_, view)| view == camera) && state.draws.len() < 2 {
            state.draws.push((cloud, buffer, generation));
        }
    }
    fn reset(&self, frame: Option<&BrowserFrame>) {
        let mut state = self.0.lock().unwrap();
        state.stamp = frame.map(|frame| (frame.frame, frame.camera));
        state.draws.clear();
    }
    fn attests(
        &self,
        frame: &BrowserFrame,
        cloud: Entity,
        buffer: BufferId,
        generation: u64,
    ) -> bool {
        let state = self.0.lock().unwrap();
        state.stamp == Some((frame.frame, frame.camera))
            && state.draws == [(cloud, buffer, Some(generation))]
    }
}

struct Pending {
    record: Value,
    phase: &'static str,
    counts: PendingCounts,
}
enum PendingCounts {
    Quad {
        candidate_count: u64,
        output_capacity: u32,
        attested: bool,
    },
    Point {
        config: BrowserPointGpuConfig,
        output_capacity: u32,
    },
}
struct ReadbackSlot {
    buffer: Buffer,
    state: Arc<AtomicU8>,
    error: Arc<Mutex<Option<String>>>,
    pending: Option<Pending>,
}
#[derive(Resource, Default)]
struct BrowserReadbacks {
    slots: Vec<ReadbackSlot>,
}

pub(super) fn install(app: &mut SubApp) {
    app.init_resource::<BrowserLodDrawProbe>()
        .init_resource::<BrowserReadbacks>()
        .add_systems(
            Core3d,
            begin_view
                .before(LodCompactionLabel)
                .before(GpuLodTraversalRender)
                .before(PointSplattingRender)
                .before(Core3dSystems::Prepass),
        )
        .add_systems(
            Core3d,
            copy_view
                .after(PointSplattingRender)
                .after(bevy::core_pipeline::upscaling::upscaling),
        )
        .add_systems(
            Render,
            collect
                .in_set(RenderSystems::Cleanup)
                .after(RenderSystems::Render),
        );
}

fn begin_view(
    view: ViewQuery<&'static ExtractedView>,
    request: Res<BrowserCaptureRequest>,
    probe: Res<BrowserLodDrawProbe>,
) {
    let view = view.into_inner();
    if request
        .frame
        .as_ref()
        .is_none_or(|frame| frame.camera == view.retained_view_entity.main_entity.id())
    {
        probe.reset(request.frame.as_ref());
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_view(
    mut context: RenderContext,
    view: ViewQuery<&'static ExtractedView>,
    request: Res<BrowserCaptureRequest>,
    probe: Res<BrowserLodDrawProbe>,
    buffers: Res<LodCompactionBuffers<Gaussian3d>>,
    points: Res<PointViews>,
    traversals: Res<GpuLodTraversalOutputs>,
    device: Res<RenderDevice>,
    adapter: Res<RenderAdapterInfo>,
    ledger: Res<LodMemoryLedger>,
    mut readbacks: ResMut<BrowserReadbacks>,
    clouds: Query<(
        Entity,
        &PlanarGaussian3dHandle,
        &LodRenderCandidates,
        Option<&GpuLodHierarchy>,
    )>,
) {
    let view = view.into_inner();
    let Some(frame) = request
        .frame
        .as_ref()
        .filter(|frame| frame.camera == view.retained_view_entity.main_entity.id())
    else {
        return;
    };
    if readbacks.slots.is_empty() {
        for _ in 0..3 {
            readbacks.slots.push(ReadbackSlot {
                buffer: device.create_buffer(&BufferDescriptor {
                    label: Some("browser_lod_capture_indirect"),
                    size: READBACK_BYTES,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Default::default(),
                error: Default::default(),
                pending: None,
            });
        }
    }
    let Some(slot) = readbacks
        .slots
        .iter_mut()
        .find(|slot| slot.state.load(Ordering::Acquire) == 0)
    else {
        request.evidence.lock().unwrap().dropped_readbacks += 1;
        return;
    };
    if let Some(config) = &frame.point_gpu {
        let Some(point) = points.capture(view.retained_view_entity) else {
            pending_frame(
                &request,
                frame,
                &adapter,
                "no_current_point_image_submission",
            );
            return;
        };
        let Some(inputs) = points.capture_traversals(view.retained_view_entity, point.submission)
        else {
            pending_frame(
                &request,
                frame,
                &adapter,
                "no_current_point_traversal_receipt",
            );
            return;
        };
        if inputs.is_empty() {
            // Reload may compose an empty image before a package publishes
            // its first hierarchy snapshot. It provides no traversal proof.
            pending_frame(&request, frame, &adapter, "no_point_hierarchy_input");
            return;
        }
        if inputs.len() != 1 {
            request.evidence.lock().unwrap().invalid_counts += 1;
            return;
        }
        let ready = clouds.iter().find_map(|(entity, handle, _, hierarchy)| {
            let hierarchy = hierarchy?;
            let output = traversals.get(view.retained_view_entity, entity)?;
            (output.is_ready()
                && output.submission > 0
                && inputs[0] == output.into()
                && output.source == handle.handle().id()
                && output.source == hierarchy.0.source
                && output.residency_generation == hierarchy.0.generation)
                .then_some(output)
        });
        let Some(output) = ready else {
            pending_frame(
                &request,
                frame,
                &adapter,
                "point_input_does_not_match_current_resident_snapshot",
            );
            return;
        };
        let record = json!({"kind":"gpu_point_frame","frame":frame.frame,"phase":frame.phase,
            "view":frame.camera.to_bits().to_string(),"cloud":output.main_cloud.to_bits().to_string(),
            "generation":output.residency_generation.to_string(),
            "allocation_generation":output.generation.to_string(),
            "source_asset":format!("{:?}",output.source),
            "point_submission":point.submission.to_string(),"traversal_submission":output.submission.to_string(),
            "source":"post_render_same_submission_point_and_traversal_copy",
            "point_image_attested":true,"output_capacity":output.capacity,
            "samples_per_pixel":point.samples_per_pixel,
            "camera_world":view.world_from_view.to_matrix().to_cols_array(),
            "projection":view.clip_from_view.to_cols_array(),"viewport":[view.viewport.z,view.viewport.w],
            "adapter":{"name":adapter.name,"backend":format!("{:?}",adapter.backend),
                "driver":adapter.driver,"driver_info":adapter.driver_info},
            "memory":memory_json(&ledger)});
        context
            .command_encoder()
            .copy_buffer_to_buffer(&point.feedback, 0, &slot.buffer, 0, 32);
        context
            .command_encoder()
            .copy_buffer_to_buffer(&output.feedback, 0, &slot.buffer, 32, 64);
        slot.pending = Some(Pending {
            record,
            phase: frame.phase,
            counts: PendingCounts::Point {
                config: config.clone(),
                output_capacity: output.capacity,
            },
        });
        slot.state.store(1, Ordering::Release);
        return;
    }
    let mut ready = None;
    for (entity, handle, candidates, _) in &clouds {
        let Some(candidate) = candidates.get(frame.camera) else {
            continue;
        };
        let Some(state) =
            buffers.get_ready(view.retained_view_entity, entity, handle.handle().id())
        else {
            continue;
        };
        let Some(drawable) = state.last_radix_drawable_for_testing(candidate) else {
            continue;
        };
        if ready.is_some() {
            request.evidence.lock().unwrap().invalid_counts += 1;
            return;
        }
        ready = Some((entity, state, drawable));
    }
    let Some((cloud, state, drawable)) = ready else {
        let mut evidence = request.evidence.lock().unwrap();
        if evidence.records.len() < 256 {
            evidence.records.push_back(
                json!({"kind":"gpu_pending","frame":frame.frame,"phase":frame.phase,
                "adapter":{"name":adapter.name,"backend":format!("{:?}",adapter.backend),
                    "driver":adapter.driver,"driver_info":adapter.driver_info},
                "reason":"no_radix_published_drawable"}),
            );
        } else {
            evidence.dropped_readbacks += 1;
        }
        return;
    };
    let generation = drawable.compaction_generation;
    let attested = probe.attests(frame, cloud, state.indirect_args_buffer.id(), generation);
    let record = json!({"kind":"gpu_frame","frame":frame.frame,"phase":frame.phase,
        "view":frame.camera.to_bits().to_string(),"cloud":cloud.to_bits().to_string(),
        "generation":generation.to_string(),"compute_input_generation":drawable.compute_input_generation.to_string(),
        "radix_publication_generation":drawable.radix_publication_generation.to_string(),
        "candidate_fingerprint_primary":drawable.candidate_fingerprint_primary.map(|value|value.to_string()),
        "candidate_fingerprint_secondary":drawable.candidate_fingerprint_secondary.map(|value|value.to_string()),
        "atlas_allocation_epoch":drawable.candidate_atlas_allocation_epoch.map(|value|value.to_string()),
        "source":"post_render_same_submission_indirect_copy","draw_command_attested":attested,
        "selected":drawable.selected_gaussians,"candidates":drawable.rendered_candidate_count,
        "output_capacity":state.output_capacity(),
        "camera_world":view.world_from_view.to_matrix().to_cols_array(),
        "projection":view.clip_from_view.to_cols_array(),"viewport":[view.viewport.z,view.viewport.w],
        "adapter":{"name":adapter.name,"backend":format!("{:?}",adapter.backend),
            "driver":adapter.driver,"driver_info":adapter.driver_info},
        "memory":memory_json(&ledger)});
    context.command_encoder().copy_buffer_to_buffer(
        &state.indirect_args_buffer,
        0,
        &slot.buffer,
        0,
        std::mem::size_of::<LodIndirectArgs>() as u64,
    );
    slot.pending = Some(Pending {
        record,
        phase: frame.phase,
        counts: PendingCounts::Quad {
            candidate_count: u64::from(drawable.rendered_candidate_count),
            output_capacity: state.output_capacity(),
            attested,
        },
    });
    slot.state.store(1, Ordering::Release);
}

fn pending_frame(
    request: &BrowserCaptureRequest,
    frame: &BrowserFrame,
    adapter: &RenderAdapterInfo,
    reason: &str,
) {
    let mut evidence = request.evidence.lock().unwrap();
    if evidence.records.len() < 256 {
        evidence.records.push_back(
            json!({"kind":"gpu_pending","frame":frame.frame,"phase":frame.phase,
            "adapter":{"name":adapter.name,"backend":format!("{:?}",adapter.backend),
                "driver":adapter.driver,"driver_info":adapter.driver_info},"reason":reason}),
        );
    } else {
        evidence.dropped_readbacks += 1;
    }
}

fn collect(mut readbacks: ResMut<BrowserReadbacks>, request: Res<BrowserCaptureRequest>) {
    // Browser WebGPU delivers mapping callbacks asynchronously. There is no
    // blocking poll, GPU wait, JS spin, or promise-await inside a Bevy system.
    for slot in &mut readbacks.slots {
        match slot.state.load(Ordering::Acquire) {
            1 => {
                let state = Arc::clone(&slot.state);
                let error = Arc::clone(&slot.error);
                slot.state.store(2, Ordering::Release);
                slot.buffer
                    .slice(..)
                    .map_async(MapMode::Read, move |result| match result {
                        Ok(()) => state.store(3, Ordering::Release),
                        Err(failure) => {
                            *error.lock().unwrap() = Some(failure.to_string());
                            state.store(4, Ordering::Release);
                        }
                    });
            }
            3 => {
                let mut pending = slot.pending.take().expect("mapped submission stamp");
                let mapped = slot.buffer.slice(..).get_mapped_range();
                let (valid, attested) = match &pending.counts {
                    PendingCounts::Quad {
                        candidate_count,
                        output_capacity,
                        attested,
                    } => {
                        let args = bytemuck::pod_read_unaligned::<LodIndirectArgs>(
                            &mapped[..std::mem::size_of::<LodIndirectArgs>()],
                        );
                        let valid = args.vertex_count == 4
                            && args.first_vertex == 0
                            && args.first_instance == 0
                            && args.overflow_count == 0
                            && args.instance_count <= args.candidate_hits
                            && u64::from(args.candidate_hits) <= *candidate_count
                            && args.instance_count <= *output_capacity;
                        pending.record["indirect"] = json!({"vertex_count":args.vertex_count,"compacted":args.instance_count,
                            "candidate_hits":args.candidate_hits,"overflow_count":args.overflow_count,
                            "drawn":attested.then_some(args.instance_count)});
                        (valid, *attested && args.instance_count > 0)
                    }
                    PendingCounts::Point {
                        config,
                        output_capacity,
                    } => {
                        let words = bytemuck::pod_read_unaligned::<[u32; 24]>(&mapped[..96]);
                        let point_flags = words[4];
                        let projected = words[5];
                        let requested = words[3];
                        let selected = words[16];
                        let visits = words[17];
                        let requests = words[18];
                        let pages = words[19];
                        let traversal_flags = words[20];
                        let valid = point_flags == 0
                            && traversal_flags & !63 == 0
                            && traversal_flags & 1 == 0
                            && pending.record["samples_per_pixel"]
                                == json!(config.samples_per_pixel)
                            && projected <= selected
                            && selected <= *output_capacity
                            && selected <= config.max_projected_gaussians
                            && requested <= config.max_points_per_frame
                            && visits <= config.max_visited_nodes
                            && (requests <= config.max_page_requests || traversal_flags & 8 != 0)
                            && pages <= config.max_frontier_nodes;
                        pending.record["point"] = json!({"projected_gaussians":projected,"requested_points":requested,
                            "dispatched_points":if point_flags == 0 {requested} else {0},"flags":point_flags});
                        pending.record["traversal"] = json!({"selected_gaussians":selected,"visited_nodes":visits,
                            "requested_pages":requests,"selected_pages":pages,"flags":traversal_flags});
                        pending.record["point_image_attested"] = json!(valid);
                        (valid, projected > 0 && requested > 0)
                    }
                };
                drop(mapped);
                slot.buffer.unmap();
                slot.state.store(0, Ordering::Release);
                pending.record["counts_valid"] = json!(valid);
                let mut evidence = request.evidence.lock().unwrap();
                if !valid {
                    evidence.invalid_counts += 1;
                }
                if valid && attested {
                    evidence.attested_phases.insert(pending.phase.to_owned());
                }
                if evidence.records.len() < 256 {
                    evidence.records.push_back(pending.record);
                } else {
                    evidence.dropped_readbacks += 1;
                }
            }
            4 => {
                let mut evidence = request.evidence.lock().unwrap();
                evidence.mapping_errors += 1;
                if evidence.records.len() < 256 {
                    evidence.records.push_back(json!({"kind":"mapping_error",
                        "error":slot.error.lock().unwrap().take(),
                        "submission":slot.pending.take().map(|pending|pending.record)}));
                }
                slot.buffer.unmap();
                slot.state.store(0, Ordering::Release);
            }
            _ => {}
        }
    }
}
