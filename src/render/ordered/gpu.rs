//! Bounded projection, stable gathering, shared radix, and one indirect draw.

use super::{
    pipeline::{Pipeline, Pipelines},
    *,
};
#[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
use crate::render::traversal::GpuLodTraversalCaptureInput;
use crate::{
    Gaussian3d, GaussianCamera, PlanarGaussian3d, PlanarGaussian3dHandle,
    gaussian::{
        cloud::CloudVisibilityClass, formats::planar_3d::PlanarStorageGaussian3d,
        settings::DrawMode,
    },
    render::{
        CloudPipeline, CloudUniform, GaussianComputeViewBindGroup, GaussianUniformBindGroups,
        lod::{LodCompactionBuffers, LodPointOutputProof},
        point::GaussianPointSplattingSettings,
        spatial_morph::GaussianLodSpatialTransitionSettings,
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
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
};

const HEADER_BYTES: u64 = 72;

/// Exercise padded multidimensional dispatch with tiny headless fixtures.
/// Device allocation and radix limits remain unchanged.
#[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
#[derive(Resource)]
#[doc(hidden)]
pub struct GlobalOrderDispatchTestLimit(pub u32);

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GlobalOrderPrepare;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GlobalOrderRender;

#[derive(Resource, Default)]
pub(crate) struct GlobalOrderReadiness {
    claimed: HashSet<RetainedViewEntity>,
    prepared: HashSet<RetainedViewEntity>,
    ready: HashSet<RetainedViewEntity>,
}

impl GlobalOrderReadiness {
    pub(crate) fn suppresses_per_cloud_pass(&self, view: RetainedViewEntity) -> bool {
        self.claimed.contains(&view)
    }
    pub(crate) fn is_prepared(&self, view: RetainedViewEntity) -> bool {
        self.prepared.contains(&view)
    }
    pub(crate) fn is_ready(&self, view: RetainedViewEntity) -> bool {
        self.ready.contains(&view)
    }
}

/// Counters from a completed draw submission; admission is current-frame state.
#[derive(Clone, Debug, Default)]
pub struct GaussianGlobalOrderFrame {
    pub ready: bool,
    pub submission: u64,
    pub projected_gaussians: u32,
    pub source_clouds: u32,
    pub gpu_bytes: u64,
    pub overflow: bool,
    pub traversal_failed: bool,
    pub error: Option<String>,
    /// Input records/edges carrying fractional current-camera representations.
    pub spatial_transition_records: u32,
    pub spatial_transition_edges: u32,
    /// Complete band requirement before optional transition capacity admission.
    pub spatial_required_records: u32,
    pub spatial_required_edges: u32,
    /// Bitmask: invalid pressure2, near bypass4, missing parent8,
    /// complete band unavailable16, exact selection unavailable32.
    pub spatial_transition_flags: u32,
    pub spatial_unavailable: bool,
    #[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
    pub projection_dispatch_rows: u32,
    #[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
    pub gather_dispatch: [u32; 3],
}

#[derive(Resource, Clone, Default)]
pub struct GaussianGlobalOrderDiagnostics(
    Arc<Mutex<HashMap<Entity, GaussianGlobalOrderFrame>>>,
    Arc<AtomicU64>,
);

impl GaussianGlobalOrderDiagnostics {
    pub fn get(&self, camera: Entity) -> Option<GaussianGlobalOrderFrame> {
        self.0.lock().unwrap().get(&camera).cloned()
    }
    fn error(&self, camera: Entity, error: impl Into<String>) {
        self.0.lock().unwrap().entry(camera).or_default().error = Some(error.into());
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceKind {
    Identity,
    Compacted(u64),
    Traversed(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Source {
    entity: Entity,
    asset: AssetId<PlanarGaussian3d>,
    capacity: u32,
    kind: SourceKind,
    spatial: bool,
}

struct Input {
    source: Source,
    offset: u32,
    config: Buffer,
    dummy: Buffer,
    bindings: BindGroup,
}

struct Pending {
    frame: GaussianGlobalOrderFrame,
    proofs: Vec<(Entity, AssetId<PlanarGaussian3d>, LodPointOutputProof)>,
    traversals: Vec<(Entity, u64, AssetId<PlanarGaussian3d>)>,
}

struct Readback {
    buffer: Buffer,
    phase: Arc<AtomicU8>,
    pending: Option<Pending>,
}

struct View {
    capacity: u32,
    dispatch_limit: u32,
    input_capacity: usize,
    bytes: u64,
    spatial: bool,
    spatial_requested: bool,
    config: Buffer,
    records: Buffer,
    header: Buffer,
    gather: BindGroup,
    draw: BindGroup,
    radix: [BindGroup; 4],
    inputs: Vec<Input>,
    readbacks: [Readback; 3],
    lease: LodMemoryLease,
    #[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
    encoded_this_frame: Option<OrderedFrameCapture>,
}

#[derive(Resource, Default)]
pub(crate) struct Views(HashMap<RetainedViewEntity, View>);

/// A draw command receipt, validated only once its copied header has completed.
#[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
pub(crate) struct OrderedFrameCapture {
    pub submission: u64,
    pub feedback: Buffer,
    pub gpu_bytes: u64,
    pub spatial: bool,
    pub spatial_requested: bool,
    pub traversals: Vec<GpuLodTraversalCaptureInput>,
}

#[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
impl Views {
    pub(crate) fn capture(&self, view: RetainedViewEntity) -> Option<&OrderedFrameCapture> {
        self.0.get(&view)?.encoded_this_frame.as_ref()
    }
}

pub(super) fn reset(
    mut readiness: ResMut<GlobalOrderReadiness>,
    diagnostics: Res<GaussianGlobalOrderDiagnostics>,
) {
    *readiness = default();
    diagnostics.0.lock().unwrap().clear();
}

fn buffer(device: &RenderDevice, label: &'static str, size: u64, usage: BufferUsages) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage,
        mapped_at_creation: false,
    })
}

#[derive(Clone, Copy, Debug)]
struct Allocation {
    records: u64,
    entries: u64,
    prefix: u64,
    status: u64,
    bytes: u64,
}

#[cfg(test)]
fn allocation(capacity: u32, inputs: usize) -> Result<Allocation, &'static str> {
    allocation_for(capacity, inputs, false)
}

fn allocation_for(capacity: u32, inputs: usize, spatial: bool) -> Result<Allocation, &'static str> {
    if capacity > MAX_ORDERED_GAUSSIANS {
        return Err("global quad order exceeds the radix tile limit");
    }
    let count = u64::from(capacity.max(1));
    let records = count
        .checked_mul(if spatial { 96 } else { 64 })
        .ok_or("global-order record allocation overflow")?;
    let entries = count
        .checked_mul(8)
        .ok_or("global-order entry allocation overflow")?
        .max(16);
    let prefix = count
        .checked_add(u64::from(capacity.max(1).div_ceil(256)) * 2)
        .and_then(|words| words.checked_mul(4))
        .ok_or("global-order prefix allocation overflow")?
        .max(16);
    let status = 1024 * u64::from(capacity.max(1).div_ceil(1024));
    let input_bytes = u64::try_from(inputs)
        .map_err(|_| "too many global-order sources")?
        .checked_mul(48)
        .ok_or("global-order input allocation overflow")?;
    let bytes = [
        records,
        entries,
        entries,
        prefix,
        status,
        4096,
        32,
        HEADER_BYTES * 4,
        64,
        input_bytes,
    ]
    .into_iter()
    .try_fold(0u64, u64::checked_add)
    .ok_or("global-order allocation overflow")?;
    Ok(Allocation {
        records,
        entries,
        prefix,
        status,
        bytes,
    })
}

/// Keep the ordinary one-dimensional dispatch; balance larger grids so the
/// final row adds fewer than `rows` padded groups rather than a whole row.
fn dispatch_grid(records: u32, max_dimension: u32) -> Option<[u32; 3]> {
    if max_dimension == 0 {
        return None;
    }
    let groups = records.div_ceil(256);
    if groups == 0 {
        return Some([0, 1, 1]);
    }
    let rows = groups.div_ceil(max_dimension);
    let columns = groups.div_ceil(rows);
    (rows <= max_dimension).then_some([columns, rows, 1])
}

fn validate_device_allocation(
    plan: &Allocation,
    capacity: u32,
    limits: &wgpu::Limits,
) -> Result<(), &'static str> {
    if [plan.records, plan.entries, plan.prefix, plan.status, 4096]
        .iter()
        .any(|bytes| {
            *bytes > limits.max_buffer_size || *bytes > limits.max_storage_buffer_binding_size
        })
    {
        return Err("global quad order exceeds device storage-buffer limits");
    }
    if limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroups_per_dimension < 256
        || capacity.div_ceil(1024) > limits.max_compute_workgroups_per_dimension
        || dispatch_grid(capacity, limits.max_compute_workgroups_per_dimension).is_none()
    {
        return Err("global quad order exceeds device compute-dispatch limits");
    }
    Ok(())
}

/// Pure admission for capture startup, before allocating a source atlas.
/// Uses the same limits and record layout as the eventual renderer allocation.
pub(crate) fn preflight_global_order(
    settings: &GaussianGlobalOrderSettings,
    capacity: u32,
    input_count: usize,
    spatial: bool,
    limits: &wgpu::Limits,
) -> Result<u64, String> {
    settings.validate()?;
    if capacity > settings.max_projected_gaussians {
        return Err("global quad order exceeds the shared projected-record budget".into());
    }
    let plan = allocation_for(capacity, input_count, spatial)?;
    if plan.bytes > settings.max_gpu_bytes {
        return Err("global quad order exceeds its camera GPU byte budget".into());
    }
    validate_device_allocation(&plan, capacity, limits)?;
    Ok(plan.bytes)
}

fn allocate(
    device: &RenderDevice,
    pipeline: &Pipeline,
    ledger: &LodMemoryLedger,
    settings: &GaussianGlobalOrderSettings,
    capacity: u32,
    inputs: usize,
    spatial: bool,
) -> Result<View, String> {
    preflight_global_order(settings, capacity, inputs, spatial, &device.limits())?;
    let plan = allocation_for(capacity, inputs, spatial)?;
    let limits = device.limits();
    let lease = ledger
        .try_reserve(LodMemoryCategory::CompactionGpu, plan.bytes)
        .map_err(|error| error.to_string())?;
    let config = buffer(
        device,
        "global_quad_config",
        32,
        BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    );
    let records = buffer(
        device,
        "global_quad_records",
        plan.records,
        BufferUsages::STORAGE,
    );
    let header = buffer(
        device,
        "global_quad_indirect",
        HEADER_BYTES,
        BufferUsages::STORAGE
            | BufferUsages::INDIRECT
            | BufferUsages::COPY_SRC
            | BufferUsages::COPY_DST,
    );
    let prefix = buffer(
        device,
        "global_quad_prefix",
        plan.prefix,
        BufferUsages::STORAGE,
    );
    let entries = std::array::from_fn::<_, 2, _>(|_| {
        buffer(
            device,
            "global_quad_entries",
            plan.entries,
            BufferUsages::STORAGE,
        )
    });
    let global = buffer(
        device,
        "global_quad_radix_global",
        4096,
        BufferUsages::STORAGE,
    );
    let status = buffer(
        device,
        "global_quad_radix_status",
        plan.status,
        BufferUsages::STORAGE,
    );
    let gather = device.create_bind_group(
        "global_quad_gather",
        &pipeline.gather_layout,
        &BindGroupEntries::sequential((
            config.as_entire_binding(),
            records.as_entire_binding(),
            prefix.as_entire_binding(),
            header.as_entire_binding(),
            entries[0].as_entire_binding(),
        )),
    );
    let draw = device.create_bind_group(
        "global_quad_draw",
        &pipeline.draw_layout,
        &BindGroupEntries::sequential((
            config.as_entire_binding(),
            records.as_entire_binding(),
            entries[0].as_entire_binding(),
        )),
    );
    let radix = std::array::from_fn(|pass| {
        let pass_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("global_quad_radix_pass"),
            contents: bytemuck::cast_slice(&[pass as u32, 0, 0, 0]),
            usage: BufferUsages::UNIFORM,
        });
        device.create_bind_group(
            "global_quad_radix",
            &pipeline.radix_layout,
            &BindGroupEntries::sequential((
                pass_buffer.as_entire_binding(),
                global.as_entire_binding(),
                status.as_entire_binding(),
                header.as_entire_binding(),
                entries[pass % 2].as_entire_binding(),
                entries[(pass + 1) % 2].as_entire_binding(),
            )),
        )
    });
    let view = View {
        capacity,
        dispatch_limit: limits.max_compute_workgroups_per_dimension,
        input_capacity: inputs,
        bytes: plan.bytes,
        spatial,
        spatial_requested: spatial,
        config,
        records,
        header,
        gather,
        draw,
        radix,
        inputs: Vec::new(),
        readbacks: std::array::from_fn(|_| Readback {
            buffer: buffer(
                device,
                "global_quad_feedback",
                HEADER_BYTES,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            ),
            phase: Arc::new(AtomicU8::new(0)),
            pending: None,
        }),
        lease,
        #[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
        encoded_this_frame: None,
    };
    view.lease.mark_gpu_materialized();
    Ok(view)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn prepare(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    cloud_pipeline: Res<CloudPipeline<Gaussian3d>>,
    mut pipelines: ResMut<Pipelines>,
    ledger: Res<LodMemoryLedger>,
    mut states: ResMut<Views>,
    mut readiness: ResMut<GlobalOrderReadiness>,
    diagnostics: Res<GaussianGlobalOrderDiagnostics>,
    #[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
    dispatch_test_limit: Option<Res<GlobalOrderDispatchTestLimit>>,
    views: Query<
        (
            &ExtractedView,
            &GaussianGlobalOrderSettings,
            &RenderVisibleEntities,
            &Msaa,
            Option<&bevy::camera::MainPassResolutionOverride>,
            Option<&GpuLodTraversalSettings>,
            Option<&GaussianLodSpatialTransitionSettings>,
            Option<&GaussianPointSplattingSettings>,
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
    *readiness = default();
    #[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
    for state in states.0.values_mut() {
        state.encoded_this_frame = None;
    }
    let live: HashSet<_> = views
        .iter()
        .filter(|(.., points)| points.is_none())
        .map(|(view, ..)| view.retained_view_entity)
        .collect();
    let expired: Vec<_> = states
        .0
        .keys()
        .filter(|key| !live.contains(key))
        .copied()
        .collect();
    for key in expired {
        if let Some(old) = states.0.remove(&key) {
            queue.on_submitted_work_done(move || drop(old));
        }
    }
    diagnostics
        .0
        .lock()
        .unwrap()
        .retain(|camera, _| live.iter().any(|view| view.main_entity.id() == *camera));
    'views: for (
        view,
        settings,
        visible,
        msaa,
        override_resolution,
        traversal_settings,
        spatial_settings,
        points,
    ) in &views
    {
        if points.is_some() {
            continue;
        }
        let key = view.retained_view_entity;
        readiness.claimed.insert(key);
        {
            let mut diagnostics = diagnostics.0.lock().unwrap();
            let frame = diagnostics.entry(key.main_entity.id()).or_default();
            frame.ready = false;
            frame.error = None;
        }
        if let Err(error) = settings.validate() {
            diagnostics.error(key.main_entity.id(), error);
            continue;
        }
        if msaa.samples() != 1
            || override_resolution.is_some()
            || view.viewport.z == 0
            || view.viewport.w == 0
        {
            diagnostics.error(key.main_entity.id(), "global quad order requires a nonempty viewport, Msaa::Off and no resolution override");
            continue;
        }
        let mut sources = Vec::new();
        let mut cold = false;
        let mut error = None;
        if let Some(visible) = visible.get::<CloudVisibilityClass>() {
            for (entity, _) in &visible.entities_cpu_culling {
                let Ok((handle, cloud, candidates, hierarchy)) = clouds.get(*entity) else {
                    continue;
                };
                if !global_order_for_cloud(Some(settings), cloud) {
                    continue;
                }
                if hierarchy.is_some() {
                    let Some(output) = traversed
                        .get(key, *entity)
                        .filter(|output| output.is_ready())
                    else {
                        error = Some("global quad order is waiting for GPU hierarchy traversal");
                        break;
                    };
                    if traversal_settings.is_none() || assets.get(output.source).is_none() {
                        error = Some(
                            "global quad order requires GPU traversal settings and its source atlas",
                        );
                        break;
                    }
                    cold |= output.source != handle.0.id();
                    sources.push(Source {
                        entity: *entity,
                        asset: output.source,
                        capacity: output.capacity,
                        kind: SourceKind::Traversed(output.generation),
                        spatial: spatial_settings
                            .is_some_and(|settings| settings.validate().is_ok())
                            // The authored node envelope bounds scales up to
                            // their stored value; enlarged dynamic splats keep
                            // the complete discrete cut.
                            && cloud.global_scale.abs() <= 1.0
                            && output.spatial_mapping.is_some(),
                    });
                    continue;
                }
                if candidates
                    .and_then(|set| set.by_camera.get(&key.main_entity.id()))
                    .is_some_and(|candidate| !supports_candidate(candidate))
                {
                    continue;
                }
                let source =
                    match fixed_source(key, *entity, handle, candidates, &assets, &compacted) {
                        Ok(source) => source,
                        Err(reason) => {
                            error = Some(reason);
                            break;
                        }
                    };
                cold |= source.asset != handle.0.id();
                sources.push(Source {
                    entity: *entity,
                    asset: source.asset,
                    capacity: source.capacity,
                    kind: source
                        .compaction_generation
                        .map_or(SourceKind::Identity, SourceKind::Compacted),
                    spatial: false,
                });
            }
        }

        sources.sort_unstable_by_key(|source| source.entity.to_bits());
        let active_capacity = sources
            .iter()
            .try_fold(0u32, |sum, source| sum.checked_add(source.capacity))
            .unwrap_or(u32::MAX);
        let capacity = active_capacity.max(1);
        if capacity > settings.max_projected_gaussians {
            error = Some("global quad order exceeds the shared projected-record budget");
        }
        if let Some(error) = error {
            diagnostics.error(key.main_entity.id(), error);
            continue;
        }
        let mut spatial = sources.iter().any(|source| source.spatial);
        // Optional representation work must not hide an otherwise admitted
        // complete discrete image while pipelines or extra bytes are unavailable.
        loop {
            let pipeline = pipelines.0[usize::from(spatial)]
                .get_or_insert_with(|| Pipeline::new(&device, &cache, &cloud_pipeline, spatial));
            pipeline.specialize(view.target_format, &cache);
            if !pipeline.loaded(view.target_format, &cache) {
                if spatial {
                    spatial = false;
                    continue;
                }
                diagnostics.error(
                    key.main_entity.id(),
                    pipeline
                        .error(&cache)
                        .unwrap_or_else(|| "global quad order pipelines are compiling".into()),
                );
                continue 'views;
            }
            if states.0.get(&key).is_none_or(|state| {
                state.spatial != spatial
                    || state.capacity < capacity
                    || state.input_capacity < sources.len()
                    || state.bytes > settings.max_gpu_bytes
            }) {
                match allocate(
                    &device,
                    pipeline,
                    &ledger,
                    settings,
                    capacity,
                    sources.len(),
                    spatial,
                ) {
                    Ok(state) => {
                        if let Some(old) = states.0.insert(key, state) {
                            queue.on_submitted_work_done(move || drop(old));
                        }
                    }
                    Err(_) if spatial => {
                        spatial = false;
                        continue;
                    }
                    Err(error) => {
                        diagnostics.error(key.main_entity.id(), error);
                        continue 'views;
                    }
                }
            }
            break;
        }
        for source in &mut sources {
            source.spatial &= spatial;
        }
        let pipeline = pipelines.0[usize::from(spatial)].as_ref().unwrap();
        let state = states.0.get_mut(&key).unwrap();
        state.spatial_requested = spatial_settings.is_some();
        #[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
        {
            state.dispatch_limit = dispatch_test_limit.as_ref().map_or(
                device.limits().max_compute_workgroups_per_dimension,
                |limit| {
                    limit
                        .0
                        .min(device.limits().max_compute_workgroups_per_dimension)
                },
            );
            if dispatch_grid(active_capacity, state.dispatch_limit).is_none() {
                diagnostics.error(
                    key.main_entity.id(),
                    "global quad order exceeds its test dispatch grid",
                );
                continue;
            }
        }
        if !state
            .inputs
            .iter()
            .map(|input| input.source)
            .eq(sources.iter().copied())
        {
            let mut previous = std::mem::take(&mut state.inputs).into_iter();
            let mut offset = 0;
            for source in sources {
                let (config, dummy) = match previous.next() {
                    Some(input) => (input.config, input.dummy),
                    None => (
                        buffer(
                            &device,
                            "global_quad_project_config",
                            32,
                            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                        ),
                        buffer(&device, "global_quad_identity", 16, BufferUsages::STORAGE),
                    ),
                };
                let (entries, indirect) = match source.kind {
                    SourceKind::Identity => (&dummy, &dummy),
                    SourceKind::Compacted(_) => {
                        let output = compacted.get(key, source.entity, source.asset).unwrap();
                        (&output.active_entries_buffer, &output.indirect_args_buffer)
                    }
                    SourceKind::Traversed(_) => {
                        let output = traversed.get(key, source.entity).unwrap();
                        (
                            &output.entries,
                            if source.spatial {
                                output.spatial_mapping.as_ref().unwrap()
                            } else {
                                &output.indirect
                            },
                        )
                    }
                };
                let bindings = device.create_bind_group(
                    "global_quad_project",
                    &pipeline.project_layout,
                    &BindGroupEntries::sequential((
                        config.as_entire_binding(),
                        entries.as_entire_binding(),
                        indirect.as_entire_binding(),
                        state.records.as_entire_binding(),
                        state.header.as_entire_binding(),
                    )),
                );
                state.inputs.push(Input {
                    source,
                    offset,
                    config,
                    dummy,
                    bindings,
                });
                offset += source.capacity;
            }
            let retired: Vec<_> = previous.collect();
            if !retired.is_empty() {
                queue.on_submitted_work_done(move || drop(retired));
            }
        }
        queue.write_buffer(
            &state.config,
            0,
            bytemuck::cast_slice(&[
                view.viewport.x,
                view.viewport.y,
                view.viewport.z,
                view.viewport.w,
                active_capacity,
                active_capacity.div_ceil(256),
                0,
                0,
            ]),
        );
        readiness.prepared.insert(key);
        if !cold {
            readiness.ready.insert(key);
        }
        let mut diagnostics = diagnostics.0.lock().unwrap();
        let frame = diagnostics.entry(key.main_entity.id()).or_default();
        frame.ready = !cold;
        frame.gpu_bytes = state.bytes;
        frame.source_clouds = state.inputs.len() as u32;
        if cold {
            frame.error = Some("global quad order is waiting for the staged source handle".into());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn radix(
    pass: &mut ComputePass<'_>,
    cache: &PipelineCache,
    pipeline: &Pipeline,
    state: &View,
    stage: usize,
    digit: usize,
    direct: Option<(u32, u32, u32)>,
    offset: u64,
) {
    pass.set_pipeline(cache.get_compute_pipeline(pipeline.radix[stage]).unwrap());
    pass.set_bind_group(3, &state.radix[digit], &[]);
    if let Some((x, y, z)) = direct {
        pass.dispatch_workgroups(x, y, z);
    } else {
        pass.dispatch_workgroups_indirect(&state.header, offset);
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn render(
    mut context: RenderContext,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipelines: Res<Pipelines>,
    mut states: ResMut<Views>,
    readiness: Res<GlobalOrderReadiness>,
    diagnostics: Res<GaussianGlobalOrderDiagnostics>,
    uniforms: Res<GaussianUniformBindGroups>,
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
    if !readiness.is_ready(key) {
        return;
    }
    let Some(state) = states.0.get_mut(&key) else {
        return;
    };
    let Some(pipeline) = &pipelines.0[usize::from(state.spatial)] else {
        return;
    };
    let Some(uniforms) = uniforms.base_bind_group.as_ref() else {
        return;
    };
    if state.inputs.iter().any(|input| {
        clouds.get(input.source.entity).is_err()
            || match input.source.kind {
                SourceKind::Identity => false,
                SourceKind::Compacted(generation) => compacted
                    .get(key, input.source.entity, input.source.asset)
                    .is_none_or(|state| !state.is_ready() || state.generation() != generation),
                SourceKind::Traversed(generation) => {
                    traversed.get(key, input.source.entity).is_none_or(|state| {
                        !state.is_ready()
                            || state.generation != generation
                            || state.source != input.source.asset
                    })
                }
            }
    }) {
        return;
    }
    let slot = state
        .readbacks
        .iter()
        .position(|slot| slot.phase.load(Ordering::Acquire) == 0);
    let Ok(submission) = diagnostics
        .1
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
    else {
        return;
    };
    let submission = submission + 1;
    let mut proofs = Vec::new();
    let mut traversal_proofs = Vec::new();
    #[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
    let mut captured_traversals = Vec::new();
    context
        .command_encoder()
        .clear_buffer(&state.header, 0, Some(HEADER_BYTES));
    {
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("global quad projection"),
                ..default()
            });
        pass.set_bind_group(
            0,
            &view_bindings.value,
            &[view_offset.offset, previous_offset.offset],
        );
        for input in &state.inputs {
            let (storage, uniform, cloud) = clouds.get(input.source.entity).unwrap();
            pass.set_pipeline(
                cache
                    .get_compute_pipeline(pipeline.project[usize::from(cloud.aabb)])
                    .unwrap(),
            );
            let draw_mode = match cloud.draw_mode {
                DrawMode::All => 0,
                DrawMode::Selected => 1,
                DrawMode::HighlightSelected => 2,
            };
            queue.write_buffer(
                &input.config,
                0,
                bytemuck::cast_slice(&[
                    input.offset,
                    input.source.capacity,
                    u32::from(input.source.kind == SourceKind::Identity),
                    draw_mode,
                    u32::from(cloud.opacity_adaptive_radius),
                    u32::from(input.source.kind != SourceKind::Identity),
                    u32::from(cloud.sort_mode == crate::sort::SortMode::None),
                    u32::from(matches!(input.source.kind, SourceKind::Traversed(_)))
                        | (u32::from(input.source.spatial) << 1),
                ]),
            );
            pass.set_bind_group(1, uniforms, &[uniform.index()]);
            pass.set_bind_group(2, &storage.bind_group, &[]);
            pass.set_bind_group(3, &input.bindings, &[]);
            if input.source.capacity > 0 {
                let [x, y, z] = dispatch_grid(input.source.capacity, state.dispatch_limit)
                    .expect("prepared source fits the admitted dispatch capacity");
                pass.dispatch_workgroups(x, y, z);
            }
            if matches!(input.source.kind, SourceKind::Compacted(_))
                && let Some(proof) = compacted
                    .get(key, input.source.entity, input.source.asset)
                    .and_then(|state| state.point_output_proof())
            {
                proofs.push((input.source.entity, input.source.asset, proof));
            }
            if matches!(input.source.kind, SourceKind::Traversed(_)) {
                let output = traversed.get(key, input.source.entity).unwrap();
                #[cfg(all(
                    feature = "testing",
                    feature = "headless",
                    not(target_arch = "wasm32")
                ))]
                captured_traversals.push(output.into());
                traversal_proofs.push((
                    output.main_cloud,
                    output.residency_generation,
                    output.source,
                ));
            }
        }
    }
    let capacity = state
        .inputs
        .iter()
        .map(|input| input.source.capacity)
        .sum::<u32>();
    // Dispatch boundaries preserve storage dependencies within each compute
    // pass. Gather produces the indirect header before radix reads it, so keep
    // those two phases separate without opening a pass for every kernel/digit.
    {
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("global quad gather"),
                ..default()
            });
        pass.set_bind_group(0, &state.gather, &[]);
        let dispatch = dispatch_grid(capacity, state.dispatch_limit)
            .expect("prepared view fits the admitted dispatch capacity");
        for (stage, [x, y, z]) in [(0, dispatch), (1, [1, 1, 1]), (2, dispatch)] {
            pass.set_pipeline(cache.get_compute_pipeline(pipeline.gather[stage]).unwrap());
            pass.dispatch_workgroups(x, y, z);
        }
    }
    {
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("global quad radix"),
                ..default()
            });
        for group in 0..3 {
            pass.set_bind_group(group, &pipeline.empty, &[]);
        }
        radix(&mut pass, &cache, pipeline, state, 0, 0, Some((1, 1, 1)), 0);
        radix(&mut pass, &cache, pipeline, state, 1, 0, None, 16);
        radix(&mut pass, &cache, pipeline, state, 2, 0, Some((1, 4, 1)), 0);
        for digit in 0..4 {
            radix(&mut pass, &cache, pipeline, state, 3, digit, None, 32);
            radix(
                &mut pass,
                &cache,
                pipeline,
                state,
                4,
                digit,
                Some((1, 256, 1)),
                0,
            );
            radix(&mut pass, &cache, pipeline, state, 5, digit, None, 32);
        }
    }
    {
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("globally ordered gaussian quads"),
                color_attachments: &[Some(target.get_color_attachment())],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: depth.view(),
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(
            cache
                .get_render_pipeline(pipeline.draws[&view.target_format])
                .unwrap(),
        );
        pass.set_bind_group(0, &state.draw, &[]);
        pass.set_viewport(
            view.viewport.x as f32,
            view.viewport.y as f32,
            view.viewport.z as f32,
            view.viewport.w as f32,
            0.0,
            1.0,
        );
        pass.draw_indirect(&state.header, 0);
    }
    #[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
    {
        state.encoded_this_frame = Some(OrderedFrameCapture {
            submission,
            feedback: state.header.clone(),
            gpu_bytes: state.bytes,
            spatial: state.spatial,
            spatial_requested: state.spatial_requested,
            traversals: captured_traversals,
        });
    }
    // A busy telemetry ring must not stop an admitted live draw. Only receipt
    // publication waits for a bounded readback slot.
    let Some(slot) = slot else {
        return;
    };
    let readback = &mut state.readbacks[slot];
    context.command_encoder().copy_buffer_to_buffer(
        &state.header,
        0,
        &readback.buffer,
        0,
        HEADER_BYTES,
    );
    readback.pending = Some(Pending {
        frame: GaussianGlobalOrderFrame {
            ready: true,
            submission,
            source_clouds: state.inputs.len() as u32,
            gpu_bytes: state.bytes,
            spatial_unavailable: state.spatial_requested && !state.spatial,
            #[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
            projection_dispatch_rows: state
                .inputs
                .iter()
                .map(|input| dispatch_grid(input.source.capacity, state.dispatch_limit).unwrap()[1])
                .max()
                .unwrap_or(0),
            #[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
            gather_dispatch: dispatch_grid(
                state.inputs.iter().map(|input| input.source.capacity).sum(),
                state.dispatch_limit,
            )
            .unwrap(),
            ..default()
        },
        proofs,
        traversals: traversal_proofs,
    });
    readback.phase.store(1, Ordering::Release);
}

pub(super) fn collect(
    mut states: ResMut<Views>,
    device: Res<RenderDevice>,
    diagnostics: Res<GaussianGlobalOrderDiagnostics>,
    mut compacted: ResMut<LodCompactionBuffers<Gaussian3d>>,
    acknowledgements: Res<GpuLodDrawAcknowledgements>,
) {
    if states.0.is_empty() {
        return;
    }
    let _ = device.poll(PollType::Poll);
    for (view, state) in &mut states.0 {
        for slot in &mut state.readbacks {
            match slot.phase.load(Ordering::Acquire) {
                1 => {
                    slot.phase.store(2, Ordering::Release);
                    let phase = slot.phase.clone();
                    slot.buffer
                        .slice(..)
                        .map_async(MapMode::Read, move |result| {
                            phase.store(if result.is_ok() { 3 } else { 4 }, Ordering::Release)
                        });
                }
                3 => {
                    let mut pending = slot.pending.take().unwrap();
                    let bytes = slot.buffer.slice(..).get_mapped_range();
                    let words: &[u32] = bytemuck::cast_slice(&bytes);
                    pending.frame.projected_gaussians = words[1];
                    pending.frame.overflow = words[12] & 1 != 0 || words[1] > state.capacity;
                    pending.frame.traversal_failed = words[12] & 2 != 0;
                    pending.frame.spatial_transition_edges = words[7];
                    pending.frame.spatial_transition_records = words[14];
                    pending.frame.spatial_transition_flags = words[15];
                    pending.frame.spatial_required_edges = words[16];
                    pending.frame.spatial_required_records = words[17];
                    pending.frame.spatial_unavailable |= words[15] & (16 | 32) != 0;
                    drop(bytes);
                    slot.buffer.unmap();
                    if !pending.frame.overflow && !pending.frame.traversal_failed {
                        for (cloud, residency_generation, source) in pending.traversals {
                            acknowledgements.publish(
                                view.main_entity.id(),
                                cloud,
                                GpuLodDrawAcknowledgement {
                                    renderer: GpuLodDrawRenderer::OrderedQuads,
                                    submission: pending.frame.submission,
                                    residency_generation,
                                    source,
                                },
                            );
                        }
                        for (entity, asset, proof) in pending.proofs {
                            if let Some(state) = compacted.get_mut(*view, entity, asset) {
                                state.mark_point_output_ready(&proof);
                            }
                        }
                    }
                    let mut frames = diagnostics.0.lock().unwrap();
                    let latest = frames.entry(view.main_entity.id()).or_default();
                    if pending.frame.submission >= latest.submission {
                        pending.frame.ready = latest.ready;
                        pending.frame.error.clone_from(&latest.error);
                        *latest = pending.frame;
                    }
                    slot.phase.store(0, Ordering::Release);
                }
                4 => {
                    slot.pending = None;
                    diagnostics.error(
                        view.main_entity.id(),
                        "global quad order feedback mapping failed",
                    );
                    slot.phase.store(0, Ordering::Release);
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn global_order_allocation_accounts_prefix_sort_and_input_storage() {
        let small = allocation(1, 1).unwrap();
        let larger = allocation(1025, 2).unwrap();
        assert_eq!(small.records, 64);
        assert_eq!(larger.status, 2048);
        assert_eq!(larger.prefix, 4 * (1025 + 2 * 5));
        assert_eq!(allocation(1025, 3).unwrap().bytes - larger.bytes, 48);
        let spatial = allocation_for(1025, 2, true).unwrap();
        assert_eq!(spatial.records, 1025 * 96);
        assert_eq!(spatial.bytes - larger.bytes, 1025 * 32);
        assert!(larger.bytes > small.bytes);

        let maximum = allocation(MAX_ORDERED_GAUSSIANS, 1).unwrap();
        assert_eq!(maximum.records, u64::from(MAX_ORDERED_GAUSSIANS) * 64);
        assert_eq!(maximum.status, 65_535 * 1024);
        assert!(allocation(MAX_ORDERED_GAUSSIANS + 1, 1).is_err());
        let mut settings = GaussianGlobalOrderSettings {
            max_projected_gaussians: MAX_ORDERED_GAUSSIANS,
            ..default()
        };
        assert!(settings.validate().is_ok());
        settings.max_projected_gaussians += 1;
        assert!(settings.validate().is_err());

        // A valid logical count does not authorize a device-sized allocation.
        assert!(
            validate_device_allocation(&maximum, MAX_ORDERED_GAUSSIANS, &wgpu::Limits::default())
                .is_err()
        );
        let large_device = wgpu::Limits {
            max_buffer_size: 8 * 1024 * 1024 * 1024,
            max_storage_buffer_binding_size: 4 * 1024 * 1024 * 1024,
            ..wgpu::Limits::default()
        };
        assert!(validate_device_allocation(&maximum, MAX_ORDERED_GAUSSIANS, &large_device).is_ok());
        let insufficient_dispatch = wgpu::Limits {
            max_compute_workgroups_per_dimension: 65_534,
            ..large_device
        };
        assert!(
            validate_device_allocation(&maximum, MAX_ORDERED_GAUSSIANS, &insufficient_dispatch)
                .is_err()
        );
    }

    #[test]
    fn global_order_dispatch_grid_preserves_padded_group_indices() {
        const LIMIT: u32 = 65_535;
        assert_eq!(dispatch_grid(0, LIMIT), Some([0, 1, 1]));
        assert_eq!(dispatch_grid(256, LIMIT), Some([1, 1, 1]));
        assert_eq!(dispatch_grid(LIMIT * 256, LIMIT), Some([LIMIT, 1, 1]));
        assert_eq!(dispatch_grid(LIMIT * 256 + 1, LIMIT), Some([32_768, 2, 1]));
        assert_eq!(
            dispatch_grid(LIMIT * 256 + 257, LIMIT),
            Some([32_769, 2, 1])
        );
        assert_eq!(
            dispatch_grid(MAX_ORDERED_GAUSSIANS, LIMIT),
            Some([LIMIT, 4, 1])
        );
        assert!(dispatch_grid(1, 0).is_none());
        assert!(dispatch_grid(10 * 256, 3).is_none());

        // Exercise the same group flattening and uniform padding rejection on
        // a tiny grid, including a partial final 256-record group.
        let records = 8u32 * 256 - 5;
        let [columns, rows, _] = dispatch_grid(records, 3).unwrap();
        let mut visited = Vec::new();
        for y in 0..rows {
            for x in 0..columns {
                let group = y * columns + x;
                if group >= records.div_ceil(256) {
                    continue;
                }
                for lane in 0..256 {
                    let record = group * 256 + lane;
                    if record < records {
                        visited.push(record);
                    }
                }
            }
        }
        assert_eq!(visited, (0..records).collect::<Vec<_>>());
    }
}
