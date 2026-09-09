//! Bounded per-view resources and the portable point-splatting pass sequence.

use bevy::{
    asset::AssetId,
    core_pipeline::prepass::PreviousViewUniformOffset,
    prelude::*,
    render::{
        extract_component::DynamicUniformIndex,
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::{
            ExtractedView, RenderVisibleEntities, RetainedViewEntity, ViewDepthTexture, ViewTarget,
            ViewUniformOffset,
        },
    },
};
use bevy_interleave::interface::storage::PlanarStorageBindGroup;
use bytemuck::{Pod, Zeroable};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
};
use wgpu::{ComputePassTimestampWrites, RenderPassTimestampWrites};

#[cfg(feature = "testing")]
use crate::render::traversal::GpuLodTraversalCaptureInput;

use super::{
    pipeline::{PointPipeline, PointPipelines, STAGES},
    *,
};
use crate::{
    Gaussian3d, GaussianCamera, PlanarGaussian3d, PlanarGaussian3dHandle,
    gaussian::{
        cloud::CloudVisibilityClass, formats::planar_3d::PlanarStorageGaussian3d,
        settings::DrawMode,
    },
    render::{
        CloudPipeline, CloudUniform, GaussianComputeViewBindGroup, GaussianUniformBindGroups,
        lod::{LodCompactionBuffers, LodPointOutputProof},
        traversal::{
            GpuLodDrawAcknowledgement, GpuLodDrawAcknowledgements, GpuLodDrawRenderer,
            GpuLodHierarchy, GpuLodTraversalOutputs, GpuLodTraversalSettings, fixed_source,
        },
    },
    stream::{
        memory::{LodMemoryCategory, LodMemoryLease, LodMemoryLedger},
        render_commit::LodRenderCandidates,
    },
};

const RECORD_BYTES: u64 = 64;
const FEEDBACK_BYTES: u64 = 32;
const READBACK_BYTES: u64 = 272;
const MAX_SCAN_ITEMS: u32 = 65535 * 256;
/// Last asynchronously completed point pass for a main-world camera.
#[derive(Clone, Debug, Default)]
pub struct GaussianPointSplattingFrame {
    pub availability: GaussianPointSplattingAvailability,
    pub submission: u64,
    pub samples_per_pixel: u32,
    pub projected_gaussians: u32,
    /// Full requested process count, saturated to the hard limit plus one.
    pub requested_points: u32,
    /// Point attempts dispatched in each visibility pass, before support and
    /// viewport rejection. Zero when the complete frame exceeded a hard limit.
    pub dispatched_points: u32,
    pub overflow: bool,
    pub sampling_failed: bool,
    /// A requested hierarchy could not produce a complete resident cut.
    pub traversal_failed: bool,
    /// Point backend GPU time, including projection, visibility and composition.
    /// Absent when timestamp queries were not enabled on the render device.
    pub gpu_ms: Option<f32>,
    pub gpu_bytes: u64,
    /// Resident sample-layer capacity; automatic mode grows this on demand.
    pub allocated_samples_per_pixel: u32,
    /// The last layer request was reduced by a hard allocation limit.
    pub admission_limited: bool,
    /// Admission or asynchronous mapping failure, when no new image can be used.
    pub error: Option<String>,
}

/// Shared small diagnostics; never maps a current-frame GPU buffer synchronously.
#[derive(Resource, Clone, Default)]
pub struct GaussianPointSplattingDiagnostics(
    Arc<Mutex<HashMap<Entity, GaussianPointSplattingFrame>>>,
    Arc<AtomicU64>,
    #[cfg(feature = "testing")] Arc<AtomicU8>,
);

impl GaussianPointSplattingDiagnostics {
    /// Suspend receipt collection to exercise rendering under readback backpressure.
    #[cfg(feature = "testing")]
    #[doc(hidden)]
    pub fn set_feedback_paused_for_testing(&self, paused: bool) {
        self.2.store(u8::from(paused), Ordering::Release);
    }

    pub fn get(&self, camera: Entity) -> Option<GaussianPointSplattingFrame> {
        self.0.lock().unwrap().get(&camera).cloned()
    }
}

/// Shared diagnostics cannot survive replacement of the device.
pub(super) fn reset_shared_state(
    diagnostics: Res<GaussianPointSplattingDiagnostics>,
    mut readiness: ResMut<PointSplattingPipelineReadiness>,
) {
    diagnostics.0.lock().unwrap().clear();
    *readiness = default();
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Config {
    width: u32,
    height: u32,
    samples: u32,
    records: u32,
    items: u32,
    groups: u32,
    blocks: u32,
    max_points: u32,
    frame: u32,
    seed: u32,
    noise_frame: u32,
    padding: u32,
    origin: [u32; 2],
    padding2: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProjectConfig {
    output_start: u32,
    input_capacity: u32,
    source_seed: u32,
    frame: u32,
    identity_source: u32,
    draw_mode: u32,
    traversed_source: u32,
    padding: u32,
}

struct CloudInput {
    entity: Entity,
    asset: AssetId<PlanarGaussian3d>,
    capacity: u32,
    offset: u32,
    uniform: Buffer,
    dummy: Buffer,
    bind_group: BindGroup,
    kind: PointInputKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PointInputKind {
    Identity,
    Compacted(u64),
    Traversed(u64),
}

struct Pending {
    frame: GaussianPointSplattingFrame,
    sampling_policy: (u32, u32, Option<f32>),
    proofs: Vec<(Entity, AssetId<PlanarGaussian3d>, LodPointOutputProof)>,
    traversals: Vec<(Entity, u64, AssetId<PlanarGaussian3d>)>,
}

struct Readback {
    staging: Buffer,
    // free, encoded, mapping, mapped, failed
    phase: Arc<AtomicU8>,
    pending: Option<Pending>,
}

struct PointView {
    settings: GaussianPointSplattingSettings,
    controller: GaussianPointSplattingSampleController,
    extent: UVec2,
    capacity: u32,
    layer_capacity: u32,
    bytes: u64,
    frame: u32,
    has_complete_image: bool,
    encoded_this_frame: Option<(u64, u32)>,
    #[cfg(feature = "testing")]
    encoded_traversals: Vec<GpuLodTraversalCaptureInput>,
    admission_limited: bool,
    config: Buffer,
    viewport: Buffer,
    records: Buffer,
    work: Buffer,
    dispatch: Buffer,
    pixels: Buffer,
    _output: Texture,
    output: TextureView,
    composite: BindGroup,
    inputs: Vec<CloudInput>,
    readbacks: [Readback; 3],
    queries: Option<wgpu::QuerySet>,
    query_resolve: Option<Buffer>,
    lease: LodMemoryLease,
    input_capacity: usize,
}

#[derive(Resource, Default)]
pub(crate) struct PointViews(HashMap<RetainedViewEntity, PointView>);

/// Same-submission capture source; copy before a later frame rewrites feedback.
pub(crate) struct PointFrameCapture {
    pub submission: u64,
    pub samples_per_pixel: u32,
    pub feedback: Buffer,
    pub gpu_bytes: u64,
}

impl PointViews {
    /// Exact traversal inputs retained with this submission's image completion
    /// header. Testing retains the current input identities even when telemetry
    /// has no free slot; it does not publish a draw acknowledgement.
    #[cfg(feature = "testing")]
    pub(crate) fn capture_traversals(
        &self,
        view: RetainedViewEntity,
        submission: u64,
    ) -> Option<&[GpuLodTraversalCaptureInput]> {
        let state = self.0.get(&view)?;
        if state.encoded_this_frame?.0 != submission {
            return None;
        }
        Some(&state.encoded_traversals)
    }

    pub(crate) fn capture(&self, view: RetainedViewEntity) -> Option<PointFrameCapture> {
        let state = self.0.get(&view)?;
        let (submission, samples_per_pixel) = state.encoded_this_frame?;
        Some(PointFrameCapture {
            submission,
            samples_per_pixel,
            feedback: state.work.clone(),
            gpu_bytes: state.bytes,
        })
    }

    pub(super) fn traversed_inputs(
        &self,
        view: RetainedViewEntity,
    ) -> impl Iterator<Item = Entity> + '_ {
        self.0
            .get(&view)
            .into_iter()
            .flat_map(|state| state.inputs.iter())
            .filter_map(|input| {
                matches!(input.kind, PointInputKind::Traversed(_)).then_some(input.entity)
            })
    }
}

fn buffer(device: &RenderDevice, label: &'static str, size: u64, usage: BufferUsages) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage,
        mapped_at_creation: false,
    })
}

fn storage(device: &RenderDevice, label: &'static str, size: u64) -> Buffer {
    buffer(
        device,
        label,
        size,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    )
}

fn work_size(items: u32) -> u64 {
    let groups = items.div_ceil(256);
    FEEDBACK_BYTES
        + u64::from(items) * 4
        + u64::from(groups) * 8
        + u64::from(groups.div_ceil(256)) * 8
}

impl PointView {
    #[allow(clippy::too_many_arguments)]
    fn allocate(
        device: &RenderDevice,
        pipeline: &PointPipelines,
        ledger: &LodMemoryLedger,
        settings: &GaussianPointSplattingSettings,
        layer_capacity: u32,
        extent: UVec2,
        capacity: u32,
        input_count: usize,
        retained_output: Option<(Texture, TextureView)>,
    ) -> Result<Self, String> {
        settings.validate().map_err(|error| error.to_string())?;
        if settings.max_points_per_frame > 1_073_741_824 {
            return Err("point budget exceeds portable indirect dispatch range".into());
        }
        if capacity > settings.max_projected_gaussians {
            return Err("projected record budget exceeded".into());
        }
        let limits = device.limits();
        if extent.max_element() > limits.max_texture_dimension_2d {
            return Err("point workspace exceeds device texture limits".into());
        }
        let items = capacity
            .checked_mul(layer_capacity)
            .filter(|n| *n <= MAX_SCAN_ITEMS)
            .ok_or("point prefix exceeds portable scan capacity")?;
        let records_size = u64::from(capacity) * RECORD_BYTES;
        let scan_size = work_size(items);
        let fixed = records_size
            + scan_size
            + extent.as_u64vec2().element_product() * 8
            + READBACK_BYTES * 3
            + 64
            + 16
            + 256
            + 16
            + 16
            + input_count as u64 * 48;
        let allocation = settings
            .sample_layer_allocation(extent, layer_capacity, 8, fixed)
            .map_err(|error| error.to_string())?;
        let pixel_size = u64::from(allocation.sample_slots) * 8;
        if extent.max_element() > limits.max_texture_dimension_2d
            || [records_size, scan_size, pixel_size].iter().any(|size| {
                *size > limits.max_storage_buffer_binding_size || *size > limits.max_buffer_size
            })
        {
            return Err("point workspace exceeds device texture/storage limits".into());
        }
        let lease = ledger
            .try_reserve(LodMemoryCategory::CompactionGpu, allocation.total_gpu_bytes)
            .map_err(|error| error.to_string())?;
        let config = buffer(
            device,
            "point_config",
            64,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let viewport = buffer(
            device,
            "point_viewport",
            16,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        // Workspace replacement must not discard the last complete image.
        // Sharing its texture preserves ordered GPU writes without copying it.
        let (output_texture, output) = retained_output.unwrap_or_else(|| {
            let output_texture = device.create_texture(&TextureDescriptor {
                label: Some("point_complete_image"),
                size: Extent3d {
                    width: extent.x,
                    height: extent.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let output = output_texture.create_view(&default());
            (output_texture, output)
        });
        let composite = device.create_bind_group(
            "point_composite",
            &pipeline.composite_layout,
            &BindGroupEntries::sequential((&output, viewport.as_entire_binding())),
        );
        let timestamps = settings.target_gpu_ms.is_some()
            && device.features().contains(WgpuFeatures::TIMESTAMP_QUERY);
        let state = Self {
            settings: settings.clone(),
            controller: GaussianPointSplattingSampleController::new(settings).unwrap(),
            extent,
            capacity,
            layer_capacity,
            bytes: allocation.total_gpu_bytes,
            frame: 0,
            has_complete_image: false,
            encoded_this_frame: None,
            #[cfg(feature = "testing")]
            encoded_traversals: Vec::new(),
            admission_limited: false,
            config,
            viewport,
            records: storage(device, "point_projected_records", records_size),
            work: buffer(
                device,
                "point_prefix_and_indirect",
                scan_size,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            dispatch: buffer(
                device,
                "point_dispatch",
                16,
                BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            ),
            pixels: storage(device, "point_visibility_samples", pixel_size),
            _output: output_texture,
            output,
            composite,
            inputs: Vec::new(),
            readbacks: std::array::from_fn(|_| Readback {
                staging: buffer(
                    device,
                    "point_feedback",
                    READBACK_BYTES,
                    BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                ),
                phase: Arc::new(AtomicU8::new(0)),
                pending: None,
            }),
            queries: timestamps.then(|| {
                device
                    .wgpu_device()
                    .create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("point_timestamps"),
                        ty: wgpu::QueryType::Timestamp,
                        count: 2,
                    })
            }),
            query_resolve: timestamps.then(|| {
                buffer(
                    device,
                    "point_timestamp_resolve",
                    16,
                    BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                )
            }),
            lease,
            input_capacity: input_count,
        };
        state.lease.mark_gpu_materialized();
        Ok(state)
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn prepare(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    mut pipeline: ResMut<PointPipeline>,
    cloud_pipeline: Res<CloudPipeline<Gaussian3d>>,
    mut states: ResMut<PointViews>,
    mut readiness: ResMut<PointSplattingPipelineReadiness>,
    ledger: Res<LodMemoryLedger>,
    diagnostics: Res<GaussianPointSplattingDiagnostics>,
    views: Query<
        (
            &ExtractedView,
            &GaussianPointSplattingSettings,
            &RenderVisibleEntities,
            &Msaa,
            Option<&bevy::camera::MainPassResolutionOverride>,
            Option<&GpuLodTraversalSettings>,
        ),
        With<GaussianCamera>,
    >,
    clouds: Query<(
        &PlanarGaussian3dHandle,
        &CloudSettings,
        Option<&LodRenderCandidates>,
        Option<&GpuLodHierarchy>,
    )>,
    assets: Res<RenderAssets<PlanarStorageGaussian3d>>,
    compacted: Res<LodCompactionBuffers<Gaussian3d>>,
    traversed: Res<GpuLodTraversalOutputs>,
) {
    readiness.ready.clear();
    readiness.prepared.clear();
    readiness.claimed.clear();
    readiness.retained.clear();
    for state in states.0.values_mut() {
        state.encoded_this_frame = None;
    }
    let live: HashSet<_> = views
        .iter()
        .map(|(view, ..)| view.retained_view_entity)
        .collect();
    states.0.retain(|key, state| {
        if live.contains(key) {
            return true;
        }
        let lease = state.lease.clone();
        queue.on_submitted_work_done(move || drop(lease));
        false
    });
    diagnostics
        .0
        .lock()
        .unwrap()
        .retain(|camera, _| live.iter().any(|view| view.main_entity.id() == *camera));
    for (view, settings, visible, msaa, resolution_override, traversal_settings) in &views {
        let key = view.retained_view_entity;
        let extent = view.viewport.zw();
        readiness.claimed.insert(key);
        let retained = states.0.get(&key).is_some_and(|state| {
            state.has_complete_image
                && state.extent == extent
                && msaa.samples() == 1
                && resolution_override.is_none()
                && pipeline.get().is_some_and(|pipeline| {
                    pipeline
                        .composites
                        .get(&view.target_format)
                        .is_some_and(|id| cache.get_render_pipeline(*id).is_some())
                })
        });
        if retained {
            readiness.retained.insert(key);
            let state = states.0.get(&key).unwrap();
            queue.write_buffer(
                &state.viewport,
                0,
                bytemuck::cast_slice(&[view.viewport.x, view.viewport.y, extent.x, extent.y]),
            );
        }
        {
            let mut diagnostics = diagnostics.0.lock().unwrap();
            let frame = diagnostics.entry(key.main_entity.id()).or_default();
            frame.availability = if retained {
                GaussianPointSplattingAvailability::Retained
            } else {
                GaussianPointSplattingAvailability::Unavailable
            };
            frame.error = None;
        }
        let mut failure = None;
        if msaa.samples() != 1 {
            failure = Some("Gaussian point splatting requires Msaa::Off".into());
        }
        if let Err(error) = settings.validate() {
            failure = Some(error.to_string());
        }
        if resolution_override.is_some() {
            failure =
                Some("Gaussian point splatting does not support MainPassResolutionOverride".into());
        }
        if let Some(error) = failure.take() {
            diagnostics
                .0
                .lock()
                .unwrap()
                .entry(key.main_entity.id())
                .or_default()
                .error = Some(error);
            continue;
        }
        let pipeline = pipeline.initialize(&device, &cache, &cloud_pipeline);
        pipeline.composite(view.target_format, &cache);
        if !pipeline.loaded(view.target_format, &cache) {
            diagnostics
                .0
                .lock()
                .unwrap()
                .entry(key.main_entity.id())
                .or_default()
                .error = Some("Gaussian point splatting pipelines are compiling".into());
            for id in std::iter::once(&pipeline.project).chain(pipeline.work.iter()) {
                if let CachedPipelineState::Err(error) = cache.get_compute_pipeline_state(*id) {
                    diagnostics
                        .0
                        .lock()
                        .unwrap()
                        .entry(key.main_entity.id())
                        .or_default()
                        .error = Some(error.to_string());
                }
            }
            continue;
        }
        let mut requested = Vec::new();
        let mut cold_staging = false;
        if let Some(visible) = visible.get::<CloudVisibilityClass>() {
            for (entity, _) in &visible.entities_cpu_culling {
                let Ok((handle, cloud, candidates, hierarchy)) = clouds.get(*entity) else {
                    continue;
                };
                if !point_splatting_for_cloud(Some(settings), cloud) {
                    continue;
                }
                if hierarchy.is_some() && traversal_settings.is_some() {
                    let Some(output) = traversed
                        .get(key, *entity)
                        .filter(|output| output.is_ready())
                    else {
                        failure = Some("GPU hierarchy traversal is not ready".into());
                        continue;
                    };
                    if output.source != handle.0.id() || assets.get(output.source).is_none() {
                        failure =
                            Some("GPU hierarchy traversal is waiting for its source atlas".into());
                        continue;
                    }
                    requested.push((
                        *entity,
                        output.source,
                        output.capacity,
                        PointInputKind::Traversed(output.generation),
                    ));
                    continue;
                }
                if let Some(candidate) =
                    candidates.and_then(|set| set.by_camera.get(&key.main_entity.id()))
                    && !supports_candidate(candidate)
                {
                    continue;
                }
                let source =
                    match fixed_source(key, *entity, handle, candidates, &assets, &compacted) {
                        Ok(source) => source,
                        Err(error) => {
                            failure = Some(error.into());
                            continue;
                        }
                    };
                // Preflight the staged atlas before its source handle is published.
                cold_staging |= source.asset != handle.0.id();
                requested.push((
                    *entity,
                    source.asset,
                    source.capacity,
                    source
                        .compaction_generation
                        .map_or(PointInputKind::Identity, PointInputKind::Compacted),
                ));
            }
        }
        // Stable instance ordering also makes exact-depth ties independent of query order.
        requested.sort_by_key(|(entity, _, _, _)| entity.to_bits());

        let capacity = requested
            .iter()
            .try_fold(0u32, |sum, (_, _, count, _)| sum.checked_add(*count))
            .unwrap_or(u32::MAX)
            .max(1);
        if capacity > settings.max_projected_gaussians {
            failure = Some("projected record budget exceeded".into());
        }
        if let Some(error) = failure {
            diagnostics
                .0
                .lock()
                .unwrap()
                .entry(key.main_entity.id())
                .or_default()
                .error = Some(error);
            continue;
        }
        let mut controller = states.0.get(&key).map_or_else(
            || GaussianPointSplattingSampleController::new(settings).unwrap(),
            |state| {
                let mut controller = state.controller.clone();
                controller.reconfigure(&state.settings, settings);
                controller
            },
        );
        let layers = controller.samples_per_pixel();
        let replace = states.0.get(&key).is_none_or(|state| {
            state.extent != extent
                || state.capacity < capacity
                || state.layer_capacity < layers
                || layers.saturating_mul(2) <= state.layer_capacity
                || state.settings.target_gpu_ms.is_some() != settings.target_gpu_ms.is_some()
                || state.input_capacity < requested.len()
                || state.bytes > settings.max_gpu_bytes
        });
        let mut admitted_layers = layers;
        if replace {
            let retained_output = states
                .0
                .get(&key)
                .filter(|state| state.extent == extent)
                .map(|state| (state._output.clone(), state.output.clone()));
            // Each whole-workspace lease charges the shared image until the
            // predecessor fence completes. This conservative temporary charge
            // preserves the existing bounded retirement contract.
            let minimum_layers = if settings.target_gpu_ms.is_some() {
                settings.min_samples_per_pixel
            } else {
                layers
            };
            let mut failure = None;
            for candidate_layers in (minimum_layers..=layers).rev() {
                // A rejected growth request can keep the currently admitted
                // workspace instead of reserving an identical replacement.
                if candidate_layers < layers
                    && states.0.get(&key).is_some_and(|state| {
                        state.extent == extent
                            && state.capacity >= capacity
                            && state.layer_capacity >= candidate_layers
                            && state.input_capacity >= requested.len()
                            && state.bytes <= settings.max_gpu_bytes
                            && state.settings.target_gpu_ms.is_some()
                                == settings.target_gpu_ms.is_some()
                    })
                {
                    admitted_layers = candidate_layers;
                    failure = None;
                    break;
                }
                match PointView::allocate(
                    &device,
                    pipeline,
                    &ledger,
                    settings,
                    candidate_layers,
                    extent,
                    capacity,
                    requested.len(),
                    retained_output.clone(),
                ) {
                    Ok(mut state) => {
                        if let Some(old) = states.0.get(&key) {
                            state.frame = old.frame;
                            state.has_complete_image =
                                old.has_complete_image && old.extent == extent;
                        }
                        if let Some(old) = states.0.insert(key, state) {
                            queue.on_submitted_work_done(move || drop(old));
                        }
                        admitted_layers = candidate_layers;
                        failure = None;
                        break;
                    }
                    Err(error) => failure = Some(error),
                }
            }
            if let Some(error) = failure {
                diagnostics
                    .0
                    .lock()
                    .unwrap()
                    .entry(key.main_entity.id())
                    .or_default()
                    .error = Some(error);
                continue;
            }
        }
        let state = states.0.get_mut(&key).unwrap();
        controller.admit_layers(admitted_layers);
        state.controller = controller;
        state.settings = settings.clone();
        state.admission_limited = admitted_layers < layers;
        let signature: Vec<_> = state
            .inputs
            .iter()
            .map(|i| (i.entity, i.asset, i.capacity, i.kind))
            .collect();
        // Rebuild only when the admitted source ranges change, not on camera movement.
        if signature != requested {
            let mut offset = 0;
            let mut inputs = Vec::with_capacity(requested.len());
            let mut reuse = std::mem::take(&mut state.inputs);
            for (entity, asset, capacity, kind) in requested {
                let (config, dummy) = match reuse.pop() {
                    Some(input) => (input.uniform, input.dummy),
                    None => (
                        buffer(
                            &device,
                            "point_project_config",
                            32,
                            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                        ),
                        storage(&device, "point_identity_input", 16),
                    ),
                };
                let (entries, indirect) = match kind {
                    PointInputKind::Identity => (&dummy, &dummy),
                    PointInputKind::Compacted(_) => {
                        let state = compacted.get(key, entity, asset).unwrap();
                        (&state.active_entries_buffer, &state.indirect_args_buffer)
                    }
                    PointInputKind::Traversed(_) => {
                        let state = traversed.get(key, entity).unwrap();
                        (&state.entries, &state.indirect)
                    }
                };
                let bind_group = device.create_bind_group(
                    "point_project",
                    &pipeline.project_layout,
                    &BindGroupEntries::sequential((
                        config.as_entire_binding(),
                        entries.as_entire_binding(),
                        indirect.as_entire_binding(),
                        state.records.as_entire_binding(),
                    )),
                );
                inputs.push(CloudInput {
                    entity,
                    asset,
                    capacity,
                    offset,
                    uniform: config,
                    dummy,
                    bind_group,
                    kind,
                });
                offset += capacity;
            }
            state.inputs = inputs;
            // Existing buffers are reused across source changes. Unused slots
            // remain conservatively covered by the view's allocation lease.
            if !reuse.is_empty() {
                queue.on_submitted_work_done(move || drop(reuse));
            }
        }
        queue.write_buffer(
            &state.viewport,
            0,
            bytemuck::cast_slice(&[view.viewport.x, view.viewport.y, extent.x, extent.y]),
        );
        readiness.prepared.insert(key);
        if !cold_staging {
            readiness.ready.insert(key);
        }
        let mut diagnostics = diagnostics.0.lock().unwrap();
        let frame = diagnostics.entry(key.main_entity.id()).or_default();
        if cold_staging {
            frame.error =
                Some("Gaussian point splatting is waiting for the staged source handle".into());
        } else {
            frame.availability = GaussianPointSplattingAvailability::Ready;
        }
        frame.allocated_samples_per_pixel = state.layer_capacity;
        frame.admission_limited = state.admission_limited;
        frame.gpu_bytes = state.bytes;
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn render(
    mut context: RenderContext,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipeline: Res<PointPipeline>,
    mut states: ResMut<PointViews>,
    readiness: Res<PointSplattingPipelineReadiness>,
    uniforms: Res<GaussianUniformBindGroups>,
    diagnostics: Res<GaussianPointSplattingDiagnostics>,
    compacted: Res<LodCompactionBuffers<Gaussian3d>>,
    traversed: Res<GpuLodTraversalOutputs>,
    clouds: Query<(
        &PlanarStorageBindGroup<Gaussian3d>,
        &DynamicUniformIndex<CloudUniform>,
        &CloudSettings,
    )>,
    view: ViewQuery<(
        &ExtractedView,
        &ViewTarget,
        &ViewDepthTexture,
        &GaussianComputeViewBindGroup,
        &ViewUniformOffset,
        &PreviousViewUniformOffset,
    )>,
) {
    let (view, target, depth, view_bindings, view_offset, previous_offset) = view.into_inner();
    let key = view.retained_view_entity;
    let Some(pipeline) = pipeline.get() else {
        return;
    };
    let Some(state) = states.0.get_mut(&key) else {
        return;
    };
    if !readiness.is_ready(key) {
        if readiness.retained.contains(&key) {
            composite(&mut context, pipeline, &cache, state, view, target, false);
        }
        return;
    }
    let Some(uniforms) = uniforms.base_bind_group.as_ref() else {
        composite(&mut context, pipeline, &cache, state, view, target, false);
        return;
    };
    if state.inputs.iter().any(|input| {
        clouds.get(input.entity).is_err()
            || match input.kind {
                PointInputKind::Identity => false,
                PointInputKind::Compacted(generation) => compacted
                    .get(key, input.entity, input.asset)
                    .is_none_or(|s| !s.is_ready() || s.generation() != generation),
                PointInputKind::Traversed(generation) => {
                    traversed.get(key, input.entity).is_none_or(|s| {
                        !s.is_ready() || s.generation != generation || s.source != input.asset
                    })
                }
            }
    }) {
        composite(&mut context, pipeline, &cache, state, view, target, false);
        return;
    }
    let slot_index = state
        .readbacks
        .iter()
        .position(|slot| slot.phase.load(Ordering::Acquire) == 0);
    if !pipeline.loaded(view.target_format, &cache) {
        return;
    }
    let Ok(submission) = diagnostics
        .1
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
    else {
        composite(&mut context, pipeline, &cache, state, view, target, false);
        return;
    };
    let submission = submission + 1;
    state.frame = state.frame.wrapping_add(1).max(1);
    let samples = state
        .controller
        .samples_per_pixel()
        .min(state.layer_capacity);
    // Retained allocation capacity is not this frame's admitted work. In
    // particular, a lower hierarchy cap must reduce prefix work immediately.
    let records = state
        .inputs
        .iter()
        .map(|input| input.capacity)
        .sum::<u32>()
        .max(1);
    let items = records * samples;
    let config = Config {
        width: state.extent.x,
        height: state.extent.y,
        samples,
        records,
        items,
        groups: items.div_ceil(256),
        blocks: items.div_ceil(65536),
        max_points: state.settings.max_points_per_frame,
        frame: state.frame,
        seed: state.settings.seed,
        noise_frame: if state.settings.temporal_sampling {
            state.frame
        } else {
            0
        },
        padding: 0,
        origin: [view.viewport.x, view.viewport.y],
        padding2: [0; 2],
    };
    queue.write_buffer(&state.config, 0, bytemuck::bytes_of(&config));
    let work_binding = device.create_bind_group(
        "point_work",
        &pipeline.work_layout,
        &BindGroupEntries::sequential((
            state.config.as_entire_binding(),
            state.records.as_entire_binding(),
            state.work.as_entire_binding(),
            state.pixels.as_entire_binding(),
            &state.output,
            depth.view(),
        )),
    );
    let mut proofs = Vec::new();
    let mut traversal_proofs = Vec::new();
    #[cfg(feature = "testing")]
    state.encoded_traversals.clear();
    {
        let timestamp_writes = state
            .queries
            .as_ref()
            .map(|query_set| ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: None,
            });
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("point_projection"),
                timestamp_writes,
            });
        pass.set_pipeline(cache.get_compute_pipeline(pipeline.project).unwrap());
        pass.set_bind_group(
            0,
            &view_bindings.value,
            &[view_offset.offset, previous_offset.offset],
        );
        for input in &state.inputs {
            let Ok((storage, uniform, settings)) = clouds.get(input.entity) else {
                continue;
            };
            let compaction = match input.kind {
                PointInputKind::Compacted(_) => compacted.get(key, input.entity, input.asset),
                PointInputKind::Identity | PointInputKind::Traversed(_) => None,
            };
            let count = compaction.map_or(input.capacity, |s| s.candidate_count());
            let config = ProjectConfig {
                output_start: input.offset,
                input_capacity: input.capacity,
                source_seed: (input.entity.to_bits() as u32)
                    ^ ((input.entity.to_bits() >> 32) as u32).wrapping_mul(0x85ebca6b),
                frame: state.frame,
                identity_source: u32::from(input.kind == PointInputKind::Identity),
                draw_mode: match settings.draw_mode {
                    DrawMode::All => 0,
                    DrawMode::Selected => 1,
                    DrawMode::HighlightSelected => 2,
                },
                traversed_source: u32::from(matches!(input.kind, PointInputKind::Traversed(_))),
                padding: 0,
            };
            queue.write_buffer(&input.uniform, 0, bytemuck::bytes_of(&config));
            pass.set_bind_group(1, uniforms, &[uniform.index()]);
            pass.set_bind_group(2, &storage.bind_group, &[]);
            pass.set_bind_group(3, &input.bind_group, &[]);
            if count > 0 {
                pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
            }
            if let Some(proof) = compaction.and_then(|s| s.point_output_proof()) {
                proofs.push((input.entity, input.asset, proof));
            }
            if matches!(input.kind, PointInputKind::Traversed(_)) {
                let output = traversed.get(key, input.entity).unwrap();
                #[cfg(feature = "testing")]
                state.encoded_traversals.push(output.into());
                traversal_proofs.push((
                    output.main_cloud,
                    output.residency_generation,
                    output.source,
                ));
            }
        }
    }
    let sample_groups = (config.width * config.height * samples).div_ceil(256);
    let dispatches = [
        (sample_groups.min(65535), sample_groups.div_ceil(65535), 1),
        (config.groups, 1, 1),
        (config.blocks, 1, 1),
        (1, 1, 1),
        (config.groups, 1, 1),
        (0, 0, 0),
        (0, 0, 0),
        (config.width.div_ceil(8), config.height.div_ceil(8), 1),
    ];
    for (stage, (x, y, z)) in dispatches.into_iter().enumerate() {
        if stage == 5 {
            context
                .command_encoder()
                .copy_buffer_to_buffer(&state.work, 0, &state.dispatch, 0, 12);
        }
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some(STAGES[stage]),
                ..default()
            });
        pass.set_pipeline(cache.get_compute_pipeline(pipeline.work[stage]).unwrap());
        pass.set_bind_group(0, &work_binding, &[]);
        if stage == 5 || stage == 6 {
            pass.dispatch_workgroups_indirect(&state.dispatch, 0);
        } else {
            pass.dispatch_workgroups(x, y, z);
        }
    }
    composite(&mut context, pipeline, &cache, state, view, target, true);
    state.encoded_this_frame = Some((submission, samples));
    // Camera-driven rendering cannot wait for asynchronous telemetry. Only
    // completed captured submissions publish receipts or drive controllers.
    let Some(slot_index) = slot_index else {
        return;
    };
    let slot = &mut state.readbacks[slot_index];
    context.command_encoder().copy_buffer_to_buffer(
        &state.work,
        0,
        &slot.staging,
        0,
        FEEDBACK_BYTES,
    );
    if let (Some(queries), Some(resolve)) = (&state.queries, &state.query_resolve) {
        context
            .command_encoder()
            .resolve_query_set(queries, 0..2, resolve, 0);
        context
            .command_encoder()
            .copy_buffer_to_buffer(resolve, 0, &slot.staging, 256, 16);
    }
    slot.pending = Some(Pending {
        sampling_policy: (
            state.settings.samples_per_pixel,
            state.settings.min_samples_per_pixel,
            state.settings.target_gpu_ms,
        ),
        frame: GaussianPointSplattingFrame {
            availability: GaussianPointSplattingAvailability::Ready,
            submission,
            samples_per_pixel: samples,
            gpu_bytes: state.bytes,
            allocated_samples_per_pixel: state.layer_capacity,
            admission_limited: state.admission_limited,
            ..default()
        },
        proofs,
        traversals: traversal_proofs,
    });
    slot.phase.store(1, Ordering::Release);
}

fn composite(
    context: &mut RenderContext,
    pipeline: &PointPipelines,
    cache: &PipelineCache,
    state: &PointView,
    view: &ExtractedView,
    target: &ViewTarget,
    timestamp: bool,
) {
    let timestamp_writes = state
        .queries
        .as_ref()
        .filter(|_| timestamp)
        .map(|query_set| RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(1),
        });
    let mut pass = context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("point_composite"),
            color_attachments: &[Some(target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    pass.set_pipeline(
        cache
            .get_render_pipeline(pipeline.composites[&view.target_format])
            .unwrap(),
    );
    pass.set_bind_group(0, &state.composite, &[]);
    pass.set_viewport(
        view.viewport.x as f32,
        view.viewport.y as f32,
        state.extent.x as f32,
        state.extent.y as f32,
        0.0,
        1.0,
    );
    pass.draw(0..3, 0..1);
}

pub(super) fn collect(
    mut states: ResMut<PointViews>,
    mut compacted: ResMut<LodCompactionBuffers<Gaussian3d>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    diagnostics: Res<GaussianPointSplattingDiagnostics>,
    acknowledgements: Res<GpuLodDrawAcknowledgements>,
) {
    #[cfg(feature = "testing")]
    if diagnostics.2.load(Ordering::Acquire) != 0 {
        return;
    }
    if states.0.is_empty() {
        return;
    }
    let _ = device.poll(PollType::Poll);
    for (view, state) in &mut states.0 {
        for slot in &mut state.readbacks {
            match slot.phase.load(Ordering::Acquire) {
                1 => {
                    let phase = slot.phase.clone();
                    slot.phase.store(2, Ordering::Release);
                    slot.staging
                        .slice(..)
                        .map_async(MapMode::Read, move |result| {
                            phase.store(if result.is_ok() { 3 } else { 4 }, Ordering::Release);
                        });
                }
                3 => {
                    let mut pending = slot.pending.take().unwrap();
                    let bytes = slot.staging.slice(..).get_mapped_range();
                    let words: &[u32] = bytemuck::cast_slice(&bytes[..32]);
                    pending.frame.requested_points = words[3];
                    pending.frame.overflow = words[4] & 1 != 0;
                    pending.frame.sampling_failed = words[4] & 2 != 0;
                    pending.frame.traversal_failed = words[4] & 4 != 0;
                    let complete = words[4] == 0;
                    pending.frame.dispatched_points = if words[4] == 0 { words[3] } else { 0 };
                    pending.frame.projected_gaussians = words[5];
                    if state.queries.is_some() {
                        let timestamps: &[u64] = bytemuck::cast_slice(&bytes[256..272]);
                        if let Some(elapsed) = timestamps[1].checked_sub(timestamps[0]) {
                            let ms = elapsed as f64 * f64::from(queue.get_timestamp_period())
                                / 1_000_000.0;
                            pending.frame.gpu_ms = Some(ms as f32);
                        }
                    }
                    drop(bytes);
                    slot.staging.unmap();
                    let current_sampling_policy = pending.sampling_policy
                        == (
                            state.settings.samples_per_pixel,
                            state.settings.min_samples_per_pixel,
                            state.settings.target_gpu_ms,
                        );
                    if complete {
                        state.has_complete_image = true;
                        for (cloud, residency_generation, source) in pending.traversals {
                            acknowledgements.publish(
                                view.main_entity.id(),
                                cloud,
                                GpuLodDrawAcknowledgement {
                                    renderer: GpuLodDrawRenderer::GaussianPoints,
                                    submission: pending.frame.submission,
                                    residency_generation,
                                    source,
                                },
                            );
                        }
                        for (entity, asset, proof) in pending.proofs {
                            if let Some(output) = compacted.get_mut(*view, entity, asset) {
                                output.mark_point_output_ready(&proof);
                            }
                        }
                    }
                    // Image completion remains valid independently of whether
                    // a newer sampling decision superseded this measurement.
                    if current_sampling_policy && (complete || pending.frame.overflow) {
                        let _ = state.controller.observe_rendered_frame(
                            pending.frame.submission,
                            pending.frame.samples_per_pixel,
                            pending.frame.gpu_ms,
                            pending.frame.overflow,
                        );
                    }
                    let mut diagnostics = diagnostics.0.lock().unwrap();
                    let latest = diagnostics.entry(view.main_entity.id()).or_default();
                    if pending.frame.submission >= latest.submission {
                        // A completed older submission cannot erase this
                        // frame's resource-admission failure or availability.
                        pending.frame.availability = latest.availability;
                        pending.frame.error.clone_from(&latest.error);
                        *latest = pending.frame;
                    }
                    slot.phase.store(0, Ordering::Release);
                }
                4 => {
                    slot.pending = None;
                    diagnostics
                        .0
                        .lock()
                        .unwrap()
                        .entry(view.main_entity.id())
                        .or_default()
                        .error = Some("point feedback mapping failed".into());
                    slot.phase.store(0, Ordering::Release);
                }
                _ => {}
            }
        }
    }
}
