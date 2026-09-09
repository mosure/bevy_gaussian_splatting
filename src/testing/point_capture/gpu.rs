//! Same-submission counters, timestamps and camera-target copies for the screen.

use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

use bevy::{
    app::SubApp,
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        Render, RenderSystems,
        render_asset::RenderAssets,
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, MapMode, PollType, TexelCopyBufferInfo,
            TexelCopyBufferLayout,
        },
        renderer::{RenderAdapterInfo, RenderContext, RenderDevice, RenderQueue, ViewQuery},
        texture::GpuImage,
        view::ExtractedView,
    },
};
use bevy_interleave::prelude::{GpuPlanarStorage, PlanarHandle};

use super::{Case, Completed, READBACK_SLOTS, Request, Shared, TIMED_FRAMES};
use crate::{
    PlanarGaussian3dHandle,
    gaussian::formats::planar_3d::PlanarStorageGaussian3d,
    render::{
        lod::LodCompactionLabel,
        point::{PointSplattingRender, PointViews},
    },
    sort::radix::RadixSortLabel,
    testing::lod_runtime_capture::LodRuntimeDrawProbe,
};

const TIMES: u64 = 256;
const IMAGE: u64 = 512;

struct Pending {
    case: Case,
    index: u32,
    row: serde_json::Value,
    image: bool,
    timestamp_period: f64,
}

struct Slot {
    staging: Buffer,
    queries: Option<wgpu::QuerySet>,
    resolve: Option<Buffer>,
    phase: Arc<AtomicU8>,
    pending: Option<Pending>,
}

#[derive(Resource, Default)]
struct State {
    slots: Vec<Slot>,
    current: Option<(usize, Request)>,
    case: Option<Case>,
    warmup: u32,
    encoded: u32,
    row_pitch: u32,
    last_point_submission: Option<u64>,
}

pub(super) fn install(app: &mut SubApp, shared: Shared) {
    app.insert_resource(shared)
        .init_resource::<State>()
        .init_resource::<LodRuntimeDrawProbe>()
        .add_systems(
            Core3d,
            begin
                .before(LodCompactionLabel)
                .before(RadixSortLabel)
                .before(Core3dSystems::Prepass)
                .before(PointSplattingRender),
        )
        .add_systems(
            Core3d,
            copy.after(PointSplattingRender)
                .after(bevy::core_pipeline::upscaling::upscaling),
        )
        .add_systems(
            Render,
            collect
                .in_set(RenderSystems::Cleanup)
                .after(RenderSystems::Render),
        );
}

fn begin(
    mut context: RenderContext,
    view: ViewQuery<&'static ExtractedView>,
    shared: Res<Shared>,
    device: Res<RenderDevice>,
    mut state: ResMut<State>,
    probe: Res<LodRuntimeDrawProbe>,
) {
    let Some(request) = shared.state.lock().unwrap().request.clone() else {
        return;
    };
    if request.camera != view.into_inner().retained_view_entity.main_entity.id() {
        return;
    }
    state.current = None;
    probe.begin_capture(request.frame, request.camera);
    if state.case != Some(request.case) {
        state.case = Some(request.case);
        state.warmup = 0;
        state.encoded = 0;
        state.last_point_submission = None;
    }
    if state.encoded >= TIMED_FRAMES {
        return;
    }
    if state.slots.is_empty() {
        state.row_pitch =
            RenderDevice::align_copy_bytes_per_row(shared.config.viewport[0] as usize * 4) as u32;
        let bytes = IMAGE
            + if shared.config.capture_images {
                u64::from(state.row_pitch) * u64::from(shared.config.viewport[1])
            } else {
                0
            };
        let timestamps = device.features().contains(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        );
        for _ in 0..READBACK_SLOTS {
            let staging = device.create_buffer(&BufferDescriptor {
                label: Some("point_compare_readback"),
                size: bytes,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let queries = timestamps.then(|| {
                device
                    .wgpu_device()
                    .create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("point_compare_whole_view_timestamps"),
                        ty: wgpu::QueryType::Timestamp,
                        count: 2,
                    })
            });
            let resolve = timestamps.then(|| {
                device.create_buffer(&BufferDescriptor {
                    label: Some("point_compare_query_resolve"),
                    size: 256,
                    usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            });
            state.slots.push(Slot {
                staging,
                queries,
                resolve,
                phase: Default::default(),
                pending: None,
            });
        }
    }
    let Some(index) = state
        .slots
        .iter()
        .position(|slot| slot.phase.load(Ordering::Acquire) == 0)
    else {
        shared.state.lock().unwrap().dropped_ring_full += 1;
        return;
    };
    if let Some(queries) = &state.slots[index].queries {
        context.command_encoder().write_timestamp(queries, 0);
    }
    state.current = Some((index, request));
}

#[allow(clippy::too_many_arguments)]
fn copy(
    mut context: RenderContext,
    view: ViewQuery<&'static ExtractedView>,
    shared: Res<Shared>,
    mut state: ResMut<State>,
    points: Res<PointViews>,
    images: Res<RenderAssets<GpuImage>>,
    gpu_clouds: Res<RenderAssets<PlanarStorageGaussian3d>>,
    clouds: Query<(Entity, &PlanarGaussian3dHandle)>,
    probe: Res<LodRuntimeDrawProbe>,
    adapter: Res<RenderAdapterInfo>,
    queue: Res<RenderQueue>,
) {
    let Some((slot_index, request)) = state.current.take() else {
        return;
    };
    let view = view.into_inner();
    if request.camera != view.retained_view_entity.main_entity.id() {
        return;
    }
    let Some(target) = images.get(&request.target) else {
        shared.state.lock().unwrap().unavailable_work += 1;
        return;
    };
    let mut point_frame = None;
    let mut indirect = None;
    if request.case == Case::Quad {
        let mut count = 0;
        for (entity, handle) in &clouds {
            let Some(cloud) = gpu_clouds.get(handle.handle()) else {
                continue;
            };
            let buffer = cloud.draw_indirect_buffer();
            if probe.attests_capture(request.frame, request.camera, entity, buffer.id(), None) {
                count += 1;
                indirect = Some(buffer.clone());
            }
        }
        if count != 1 {
            shared.state.lock().unwrap().unavailable_work += 1;
            return;
        }
    } else {
        let Some(frame) = points.capture(view.retained_view_entity) else {
            shared.state.lock().unwrap().unavailable_work += 1;
            return;
        };
        if request.case.samples() != Some(frame.samples_per_pixel)
            || state
                .last_point_submission
                .is_some_and(|old| frame.submission <= old)
        {
            shared.state.lock().unwrap().unavailable_work += 1;
            return;
        }
        state.last_point_submission = Some(frame.submission);
        point_frame = Some(frame);
    }
    if state.warmup < shared.config.warmup_frames {
        state.warmup += 1;
        return;
    }
    let sample_index = state.encoded;
    let capture_image = shared.config.capture_images && matches!(sample_index, 0 | 7 | 15);
    let row_pitch = state.row_pitch;
    let readback_bytes = state
        .slots
        .iter()
        .map(|s| s.staging.size() + s.resolve.as_ref().map_or(0, |b| b.size()))
        .sum::<u64>();
    let slot = &mut state.slots[slot_index];
    // End the measured scope before instrumentation copies, after the final
    // target write. Both timestamps belong to this RenderContext submission.
    if let Some(queries) = &slot.queries {
        context.command_encoder().write_timestamp(queries, 1);
        let resolve = slot.resolve.as_ref().unwrap();
        context
            .command_encoder()
            .resolve_query_set(queries, 0..2, resolve, 0);
        context
            .command_encoder()
            .copy_buffer_to_buffer(resolve, 0, &slot.staging, TIMES, 16);
    }
    if let Some(buffer) = &indirect {
        context
            .command_encoder()
            .copy_buffer_to_buffer(buffer, 0, &slot.staging, 0, 16);
    }
    if let Some(frame) = &point_frame {
        context
            .command_encoder()
            .copy_buffer_to_buffer(&frame.feedback, 0, &slot.staging, 0, 32);
    }
    if capture_image {
        context.command_encoder().copy_texture_to_buffer(
            target.texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &slot.staging,
                layout: TexelCopyBufferLayout {
                    offset: IMAGE,
                    bytes_per_row: Some(row_pitch),
                    rows_per_image: None,
                },
            },
            target.texture_descriptor.size,
        );
    }
    let row = serde_json::json!({
        "case":request.case,"sample_index":sample_index,"frame":request.frame,"view_id":request.camera.to_bits(),
        "camera":{"world_from_view":view.world_from_view.to_matrix().to_cols_array(),"clip_from_view":view.clip_from_view.to_cols_array(),
            "viewport":view.viewport.to_array()},
        "point_submission":point_frame.as_ref().map(|frame|frame.submission),
        "samples_per_pixel":request.case.samples(),
        "gps_gpu_bytes":point_frame.as_ref().map(|frame|frame.gpu_bytes),"capture_gpu_bytes":readback_bytes,
        "quad_draw_command_attested":indirect.is_some(),
        "point_new_work_encoded":point_frame.is_some(),
        "gpu_ms":null,"timestamp_available":slot.queries.is_some(),"image":null,
        "adapter":{"name":adapter.name,"vendor":adapter.vendor,"device":adapter.device,
            "backend":format!("{:?}",adapter.backend),"driver":adapter.driver,"driver_info":adapter.driver_info}
    });
    slot.pending = Some(Pending {
        case: request.case,
        index: sample_index,
        row,
        image: capture_image,
        timestamp_period: f64::from(queue.get_timestamp_period()),
    });
    slot.phase.store(1, Ordering::Release);
    state.encoded += 1;
}

fn collect(mut state: ResMut<State>, device: Res<RenderDevice>, shared: Res<Shared>) {
    if let Err(error) = device.poll(PollType::Poll) {
        shared.state.lock().unwrap().errors.push(error.to_string());
        return;
    }
    let row_pitch = state.row_pitch as usize;
    for slot in &mut state.slots {
        match slot.phase.load(Ordering::Acquire) {
            1 => {
                let phase = slot.phase.clone();
                let errors = shared.state.clone();
                slot.phase.store(2, Ordering::Release);
                slot.staging
                    .slice(..)
                    .map_async(MapMode::Read, move |result| {
                        if let Err(error) = result {
                            errors.lock().unwrap().errors.push(error.to_string());
                            phase.store(4, Ordering::Release);
                        } else {
                            phase.store(3, Ordering::Release);
                        }
                    });
            }
            3 => {
                let mut pending = slot.pending.take().unwrap();
                let data = slot.staging.slice(..).get_mapped_range();
                let word = |index: usize| {
                    u32::from_le_bytes(data[index * 4..index * 4 + 4].try_into().unwrap())
                };
                if pending.case == Case::Quad {
                    pending.row["quad_indirect"] =
                        serde_json::json!({"vertices":word(0),"instances":word(1)});
                    pending.row["complete_work"] = serde_json::json!(word(0) > 0 && word(1) > 0);
                } else {
                    let flags = word(4);
                    pending.row["point_feedback"] = serde_json::json!({"dispatch":[word(0),word(1),word(2)],
                        "requested_points":word(3),"dispatched_points":if flags==0{word(3)}else{0},
                        "projected_gaussians":word(5),"flags":flags,"overflow":flags&1!=0,"sampling_failed":flags&2!=0});
                    pending.row["complete_work"] =
                        serde_json::json!(flags == 0 && word(3) > 0 && word(5) > 0);
                }
                if slot.queries.is_some() {
                    let start = u64::from_le_bytes(
                        data[TIMES as usize..TIMES as usize + 8].try_into().unwrap(),
                    );
                    let end = u64::from_le_bytes(
                        data[TIMES as usize + 8..TIMES as usize + 16]
                            .try_into()
                            .unwrap(),
                    );
                    if end >= start {
                        pending.row["gpu_ms"] = serde_json::json!(
                            (end - start) as f64 * pending.timestamp_period / 1_000_000.0
                        );
                    } else {
                        pending.row["timestamp_error"] =
                            serde_json::json!("nonmonotonic timestamp");
                    }
                }
                let rgba = if pending.image {
                    let width = shared.config.viewport[0] as usize * 4;
                    let mut rgba = Vec::with_capacity(width * shared.config.viewport[1] as usize);
                    for row in data[IMAGE as usize..]
                        .chunks_exact(row_pitch)
                        .take(shared.config.viewport[1] as usize)
                    {
                        rgba.extend_from_slice(&row[..width]);
                    }
                    Some(rgba)
                } else {
                    None
                };
                drop(data);
                slot.staging.unmap();
                slot.phase.store(0, Ordering::Release);
                if let Err(error) = shared.sender.try_send(Completed {
                    case: pending.case,
                    index: pending.index,
                    row: pending.row,
                    rgba,
                }) {
                    shared
                        .state
                        .lock()
                        .unwrap()
                        .errors
                        .push(format!("bounded completion queue rejected sample: {error}"));
                }
            }
            _ => {}
        }
    }
}
