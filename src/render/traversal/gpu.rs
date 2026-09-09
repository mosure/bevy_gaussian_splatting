use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU8, Ordering},
    },
};

use bevy::{
    asset::{AssetId, load_internal_asset, uuid_handle},
    camera::primitives::Frustum,
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::ExtractComponentPlugin,
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::{ExtractedView, RenderVisibleEntities, RetainedViewEntity},
    },
};
use bevy_interleave::prelude::*;
use bytemuck::{Pod, Zeroable};

#[cfg(test)]
use super::snapshot::GpuLodHierarchyTree;
use super::snapshot::{
    GpuLodHierarchy, GpuLodHierarchySnapshot, GpuLodPagePlacement, GpuLodTraversalSettings, Node,
};
use super::{
    GpuLodDrawAcknowledgements,
    admission::{admit_traversal_records, fixed_source},
};
use crate::render::ordered::{GaussianGlobalOrderSettings, global_order_for_cloud};
use crate::render::point::{GaussianPointSplattingSettings, point_splatting_for_cloud};
use crate::render::spatial_morph::{
    GaussianLodSpatialTransitionSettings, GpuLodSpatialMorph, SpatialMorphBuffers,
    SpatialMorphPipelines, SpatialMorphState,
};
use crate::stream::atlas_upload::LodAtlasGpuGenerations;
use crate::{
    CloudSettings, Gaussian3d, GaussianLodSettings, PlanarGaussian3d, PlanarGaussian3dHandle,
    gaussian::{
        cloud::CloudVisibilityClass,
        formats::{planar_3d::PlanarStorageGaussian3d, planar_3d_chunked::LodPageId},
        lod_settings::{LodQualityTarget, LodSelectionMode},
    },
    render::lod::{LodCompactionBuffers, LodCompactionPrepare},
    stream::{
        memory::{LodMemoryCategory, LodMemoryLease, LodMemoryLedger},
        render_commit::LodRenderCandidates,
    },
};

const SHADER: Handle<Shader> = uuid_handle!("f2db5f57-c35c-4084-ad35-0aafdd74d19b");
const HEADER_BYTES: u64 = 64;
const SCAN_WORKGROUP_BYTES: u32 = 256 * 9 * 4;
const STAGES: [&str; 32] = [
    "reset",
    "bootstrap",
    "traverse",
    "advance",
    "prepare_expand",
    "expand",
    "finish",
    "begin_resident",
    "scan_groups",
    "scatter",
    "bitmap_scan",
    "bitmap_groups",
    "bitmap_scatter",
    "next_bucket",
    "begin_demand",
    "begin_canonical",
    "rank_demand",
    "rank_demand_groups",
    "order_demand",
    "score_candidates",
    "score_next_depth",
    "cutoff_radix_histogram",
    "cutoff_radix_groups",
    "cutoff_radix_scatter",
    "cutoff_next_radix",
    "cutoff_costs",
    "cutoff_boundary",
    "cutoff_materialize",
    "classify_visibility",
    "physical_counts",
    "physical_groups",
    "physical_ranges",
];

fn scan_work_words(frontier_nodes: u32, topology_nodes: u32) -> u64 {
    let bitmap = u64::from(topology_nodes).div_ceil(32);
    6 * u64::from(frontier_nodes)
        + 8 * u64::from(frontier_nodes).div_ceil(256)
        + 13
        + 14 * bitmap
        + 2 * bitmap.div_ceil(256)
}

fn cutoff_work_words(topology_nodes: u32, candidates: u32) -> u64 {
    8 + u64::from(topology_nodes)
        + 8 * u64::from(candidates)
        + 260 * u64::from(candidates).div_ceil(256)
}

/// Dispatch boundaries are shared with the bounded raw-GPU oracle. Only a
/// copied indirect argument block requires a new pass; dependent storage work
/// is batched. Desired classification never reads the residency table.
fn traversal_schedule(
    config: &Config,
    max_depth: u32,
    mut emit: impl FnMut(bool, &'static str, &[(usize, Option<u32>)]),
) {
    let bitmap = config.counts[0].div_ceil(32);
    let groups = bitmap.div_ceil(256);
    let scan = [(10, Some(groups)), (11, Some(1)), (12, Some(groups))];
    emit(
        false,
        "lod_traversal_setup",
        &[
            (
                0,
                Some(bitmap.max(config.feedback_offsets[3]).div_ceil(256).max(1)),
            ),
            (1, Some(config.counts[2].div_ceil(256))),
        ],
    );
    if config.spatial[2] != 0 && config.quality[2] == 1.0 {
        let candidates = config.spatial[3].div_ceil(256);
        for _depth in 0..max_depth {
            emit(
                false,
                "lod_cutoff_scores",
                &[(19, Some(candidates)), (20, Some(1))],
            );
        }
        for _digit in 0..4 {
            emit(
                false,
                "lod_cutoff_radix",
                &[
                    (21, Some(candidates)),
                    (22, Some(1)),
                    (23, Some(candidates)),
                    (24, Some(1)),
                ],
            );
        }
        emit(
            false,
            "lod_cutoff_admission",
            &[
                (25, Some(candidates)),
                (26, Some(1)),
                (27, Some(candidates)),
            ],
        );
    } else {
        for _bucket in (1..8).rev() {
            // Leaves are never pending; depth D has exactly D internal rounds.
            for level in 0..max_depth {
                emit(false, "lod_desired_queue", &scan);
                let steps = [(2, None), (8, Some(1)), (9, None), (13, Some(1))];
                emit(
                    true,
                    "lod_desired_admission",
                    &steps[..if level + 1 == max_depth { 4 } else { 3 }],
                );
            }
        }
    }
    emit(false, "lod_resident_begin", &[(7, Some(1))]);
    for _ in 0..=max_depth {
        emit(
            true,
            "lod_resident_fallback",
            &[(2, None), (8, Some(1)), (9, None), (3, Some(1))],
        );
    }
    emit(
        false,
        "lod_canonical_ranges",
        &[
            (15, Some(1)),
            (10, Some(groups)),
            (11, Some(1)),
            (12, Some(groups)),
            (14, Some(1)),
        ],
    );
    emit(
        true,
        "lod_demand_admission",
        &[
            (16, None),
            (17, Some(1)),
            (18, None),
            (2, None),
            (8, Some(1)),
            (9, None),
            (4, Some(1)),
        ],
    );
    // Annotation reads logical ranges but does not require expanded Gaussians.
    emit(false, "lod_annotation_header", &[(6, Some(1))]);
    emit(false, "lod_spatial_annotation", &[]);
    emit(
        false,
        "lod_physical_ranges",
        &[
            (28, Some(config.counts[0].div_ceil(256))),
            (
                29,
                Some(
                    config.limits[1]
                        .div_ceil(256)
                        .max(config.feedback_offsets[3].div_ceil(256)),
                ),
            ),
            (30, Some(1)),
            (31, Some(config.limits[1].div_ceil(256))),
        ],
    );
    emit(true, "lod_traversal_expand", &[(5, None), (6, Some(1))]);
}

#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuLodTraversalPrepare;
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuLodTraversalRender;

/// One asynchronously completed traversal. Page identifiers are manifest IDs,
/// not atlas slots; publishers must reject feedback from replaced generations.
#[derive(Clone, Debug)]
pub struct GpuLodTraversalFeedback {
    pub generation: u64,
    pub submission: u64,
    pub source: AssetId<PlanarGaussian3d>,
    /// Required and pre-band complete child cohorts, including resident siblings.
    /// The raw GPU header counts references before this list is deduplicated.
    pub requested_pages: Vec<LodPageId>,
    /// Resident pages of the complete logical cut, including omitted navigation
    /// nodes, plus parent pages read by active spatial transitions.
    pub selected_pages: Vec<LodPageId>,
    pub selected_gaussians: u32,
    pub visited_nodes: u32,
    pub complete: bool,
    pub record_limited: bool,
    pub frontier_limited: bool,
    pub visit_limited: bool,
    pub request_overflow: bool,
    /// Continuous budget admission could not cover the full candidate graph.
    pub cutoff_unavailable: bool,
}

/// A bounded latest-result mailbox shared with main-world package publishers.
#[derive(Resource, Clone, Default)]
pub struct GpuLodTraversalFeedbacks(Arc<Mutex<HashMap<(Entity, Entity), GpuLodTraversalFeedback>>>);

impl GpuLodTraversalFeedbacks {
    pub fn take(&self, camera: Entity, cloud: Entity) -> Option<GpuLodTraversalFeedback> {
        self.0.lock().unwrap().remove(&(camera, cloud))
    }
}

/// Physical source indices and draw arguments [4, count, 0, 0, flags, 0, 0, 0].
/// The generation identifies these buffer allocations, while feedback reports
/// the publisher's residency generation independently.
pub struct GpuLodTraversalOutput {
    pub entries: Buffer,
    pub indirect: Buffer,
    pub capacity: u32,
    pub generation: u64,
    pub residency_generation: u64,
    pub main_cloud: Entity,
    pub source: AssetId<PlanarGaussian3d>,
    /// First 64 bytes are the same-submission counters documented in the shader.
    pub feedback: Buffer,
    pub submission: u64,
    /// The current-view support policy permits explicit zero-count logical ranges.
    pub omission_enabled: bool,
    /// Exact current-producer values: support inflation (zero disables), world
    /// Mip factor, perspective flag, and active spatial annotation flag.
    pub omission_parameters: [f32; 4],
    /// Exact production world-space frustum planes, including the full-camera
    /// frustum retained by Bevy for sub-camera projections.
    pub omission_frustum: [[f32; 4]; 6],
    /// Optional immutable correspondence for the annotated entry tail. Only
    /// exposed when the same-frame spatial annotation pipelines are ready.
    pub(crate) spatial_mapping: Option<Buffer>,
    /// Testing-only range of [node, physical start, physical count, omission flags] rows.
    #[cfg(feature = "testing")]
    pub(crate) capture_selected_range: (u64, u32),
    ready: bool,
}

impl GpuLodTraversalOutput {
    pub fn is_ready(&self) -> bool {
        self.ready
    }
}

#[derive(Resource, Default)]
pub struct GpuLodTraversalOutputs {
    states: HashMap<(RetainedViewEntity, Entity), ViewState>,
    trees: HashMap<usize, Weak<GpuTree>>,
    snapshots: HashMap<usize, Weak<GpuSnapshot>>,
    next_generation: u64,
}

impl GpuLodTraversalOutputs {
    pub fn get(&self, view: RetainedViewEntity, cloud: Entity) -> Option<&GpuLodTraversalOutput> {
        self.states.get(&(view, cloud)).map(|state| &state.output)
    }
}

pub struct GpuLodTraversalPlugin;

impl Plugin for GpuLodTraversalPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<GpuLodHierarchy>::default(),
            ExtractComponentPlugin::<GpuLodTraversalSettings>::default(),
            ExtractComponentPlugin::<GaussianLodSpatialTransitionSettings>::default(),
            ExtractComponentPlugin::<GpuLodSpatialMorph>::default(),
        ))
        .init_resource::<GpuLodTraversalFeedbacks>()
        .init_resource::<GpuLodDrawAcknowledgements>()
        .add_systems(PostUpdate, super::acknowledgement::prune);
        crate::render::spatial_morph::install(app);
        load_internal_asset!(app, SHADER, "traversal.wgsl", Shader::from_wgsl);
        let feedback = app.world().resource::<GpuLodTraversalFeedbacks>().clone();
        let acknowledgements = app.world().resource::<GpuLodDrawAcknowledgements>().clone();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .insert_resource(feedback)
                .insert_resource(acknowledgements)
                .init_gpu_resource::<GpuLodTraversalOutputs>()
                .init_gpu_resource::<Pipelines>()
                .init_gpu_resource::<SpatialMorphPipelines>()
                .add_systems(RenderStartup, clear_feedback)
                .add_systems(
                    Render,
                    prepare
                        .in_set(RenderSystems::PrepareResources)
                        .in_set(GpuLodTraversalPrepare)
                        .after(LodCompactionPrepare),
                )
                .add_systems(
                    Render,
                    collect
                        .in_set(RenderSystems::Cleanup)
                        .after(RenderSystems::Render),
                )
                .add_systems(
                    Core3d,
                    render
                        .in_set(Core3dSystems::MainPass)
                        .in_set(GpuLodTraversalRender),
                );
        }
    }
}

fn clear_feedback(
    feedbacks: Res<GpuLodTraversalFeedbacks>,
    acknowledgements: Res<GpuLodDrawAcknowledgements>,
) {
    // Main-world publishers share this mailbox across device recreation.
    feedbacks.0.lock().unwrap().clear();
    acknowledgements.clear();
}

#[derive(Resource, Default)]
struct Pipelines(Option<Pipeline>);

struct Pipeline {
    layout: BindGroupLayout,
    stages: [CachedComputePipelineId; STAGES.len()],
}

impl Pipeline {
    fn new(device: &RenderDevice, cache: &PipelineCache) -> Self {
        let entries: Vec<_> = (0..7)
            .map(|binding| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: if binding == 0 {
                        BufferBindingType::Uniform
                    } else {
                        BufferBindingType::Storage {
                            read_only: binding < 3 || binding == 6,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let descriptor = BindGroupLayoutDescriptor::new("lod_gpu_traversal", &entries);
        Self {
            layout: device.create_bind_group_layout(Some("lod_gpu_traversal"), &entries),
            stages: STAGES.map(|stage| {
                cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(format!("lod_gpu_{stage}").into()),
                    layout: vec![descriptor.clone()],
                    shader: SHADER,
                    entry_point: Some(stage.into()),
                    ..default()
                })
            }),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Config {
    world_from_local: [f32; 16],
    clip_from_world: [f32; 16],
    view: [f32; 4],
    quality: [f32; 4],
    counts: [u32; 4],
    limits: [u32; 4],
    offsets: [u32; 4],
    feedback_offsets: [u32; 4],
    // Selected page capacity, transition edge capacity, reserved.
    spatial: [u32; 4],
    // Support inflation (0 disables), Mip factor, perspective, active annotation.
    omission: [f32; 4],
    // The exact production ViewUniform frustum, including full-camera crop policy.
    frustum: [[f32; 4]; 6],
}

struct GpuTree {
    buffer: Buffer,
    source_order: Buffer,
    _lease: LodMemoryLease,
}
struct GpuSnapshot {
    tree: Arc<GpuTree>,
    pages: Buffer,
    data: Arc<GpuLodHierarchySnapshot>,
    _lease: LodMemoryLease,
}
struct Readback {
    buffer: Buffer,
    phase: Arc<AtomicU8>,
    submission: u64,
    // Decode and retain the exact immutable publication copied into this slot.
    snapshot: Option<Arc<GpuSnapshot>>,
}
struct ViewState {
    output: GpuLodTraversalOutput,
    snapshot: Arc<GpuSnapshot>,
    settings: GpuLodTraversalSettings,
    config: Buffer,
    config_data: Config,
    bind_group: BindGroup,
    dispatch: Buffer,
    readbacks: [Readback; 3],
    spatial_settings: Option<GaussianLodSpatialTransitionSettings>,
    spatial_source: Option<GpuLodSpatialMorph>,
    spatial: Option<SpatialMorphState>,
    _lease: LodMemoryLease,
}

impl ViewState {
    fn spatial_buffers(&self) -> SpatialMorphBuffers<'_> {
        let frontier = self.settings.max_frontier_nodes;
        let bitmap = self.config_data.counts[0].div_ceil(32);
        SpatialMorphBuffers {
            config: &self.config,
            nodes: &self.snapshot.tree.buffer,
            pages: &self.snapshot.pages,
            feedback: &self.output.feedback,
            entries: &self.output.entries,
            indirect: &self.output.indirect,
            selected_nodes_offset: self.config_data.feedback_offsets[2]
                + self.config_data.spatial[0]
                + 6 * frontier
                + 8 * frontier.div_ceil(256)
                + 13
                + 9 * bitmap,
            selected_page_capacity: self.config_data.spatial[0],
            frontier_nodes: frontier,
            entry_capacity: self.output.capacity,
        }
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

fn reserve(ledger: &LodMemoryLedger, bytes: u64) -> Result<LodMemoryLease, String> {
    ledger
        .try_reserve(LodMemoryCategory::CompactionGpu, bytes)
        .map_err(|error| error.to_string())
}

fn storage_fits(limits: &wgpu::Limits, bytes: u64) -> bool {
    bytes <= limits.max_storage_buffer_binding_size && bytes <= limits.max_buffer_size
}

fn ensure_snapshot(
    outputs: &mut GpuLodTraversalOutputs,
    device: &RenderDevice,
    queue: &RenderQueue,
    ledger: &LodMemoryLedger,
    snapshot: &Arc<GpuLodHierarchySnapshot>,
) -> Result<Arc<GpuSnapshot>, String> {
    let key = Arc::as_ptr(snapshot) as usize;
    if let Some(gpu) = outputs.snapshots.get(&key).and_then(Weak::upgrade) {
        return Ok(gpu);
    }
    let page_bytes = page_allocation_bytes(snapshot);
    let page_lease = reserve(ledger, page_bytes)?;
    let tree_key = Arc::as_ptr(&snapshot.tree) as usize;
    let tree = if let Some(tree) = outputs.trees.get(&tree_key).and_then(Weak::upgrade) {
        tree
    } else {
        let bytes = tree_allocation_bytes(snapshot);
        let lease = reserve(ledger, bytes)?;
        let data = buffer(
            device,
            "lod_hierarchy_nodes",
            tree_node_bytes(snapshot),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let source_order = buffer(
            device,
            "lod_hierarchy_source_order",
            tree_order_bytes(snapshot),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        queue.write_buffer(&data, 0, bytemuck::cast_slice(&snapshot.tree.nodes));
        queue.write_buffer(
            &source_order,
            0,
            bytemuck::cast_slice(&snapshot.tree.source_order),
        );
        if !snapshot.tree.candidates.is_empty() {
            queue.write_buffer(
                &source_order,
                snapshot.tree.source_order.len() as u64 * 4,
                bytemuck::cast_slice(&snapshot.tree.candidates),
            );
        }
        lease.mark_gpu_materialized();
        let gpu = Arc::new(GpuTree {
            buffer: data,
            source_order,
            _lease: lease,
        });
        outputs.trees.insert(tree_key, Arc::downgrade(&gpu));
        gpu
    };
    let pages = buffer(
        device,
        "lod_hierarchy_pages",
        page_bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    queue.write_buffer(&pages, 0, bytemuck::cast_slice(&snapshot.placements));
    page_lease.mark_gpu_materialized();
    let gpu = Arc::new(GpuSnapshot {
        tree,
        pages,
        data: snapshot.clone(),
        _lease: page_lease,
    });
    outputs.snapshots.insert(key, Arc::downgrade(&gpu));
    Ok(gpu)
}

struct AllocationPlan {
    work_bytes: u64,
    entry_bytes: u64,
    readback_bytes: u64,
    bytes: u64,
    spatial_bytes: u64,
    spatial_settings: Option<GaussianLodSpatialTransitionSettings>,
    spatial_source: Option<GpuLodSpatialMorph>,
    config: Config,
}

fn page_allocation_bytes(snapshot: &GpuLodHierarchySnapshot) -> u64 {
    (snapshot.placements.len() as u64 * size_of::<GpuLodPagePlacement>() as u64).max(16)
}

fn tree_node_bytes(snapshot: &GpuLodHierarchySnapshot) -> u64 {
    (snapshot.tree.nodes.len() as u64 * size_of::<Node>() as u64).max(16)
}

fn tree_order_bytes(snapshot: &GpuLodHierarchySnapshot) -> u64 {
    (snapshot.tree.source_order.len() as u64 * size_of::<u32>() as u64
        + snapshot.tree.candidates.len() as u64 * size_of::<[u32; 4]>() as u64)
        .max(16)
}

fn tree_allocation_bytes(snapshot: &GpuLodHierarchySnapshot) -> u64 {
    tree_node_bytes(snapshot) + tree_order_bytes(snapshot)
}

#[derive(Default)]
struct ViewFootprint {
    trees: HashSet<usize>,
    snapshots: HashSet<usize>,
    bytes: u64,
}

impl ViewFootprint {
    fn admit(
        &mut self,
        snapshot: &Arc<GpuLodHierarchySnapshot>,
        plan: &AllocationPlan,
        limit: u64,
    ) -> Result<(), &'static str> {
        let shared_bytes = if self.trees.insert(Arc::as_ptr(&snapshot.tree) as usize) {
            tree_allocation_bytes(snapshot)
        } else {
            0
        } + if self.snapshots.insert(Arc::as_ptr(snapshot) as usize) {
            page_allocation_bytes(snapshot)
        } else {
            0
        };
        self.bytes = self
            .bytes
            .checked_add(plan.bytes)
            .and_then(|bytes| bytes.checked_add(plan.spatial_bytes))
            .and_then(|bytes| bytes.checked_add(shared_bytes))
            .ok_or("GPU hierarchy view byte count overflow")?;
        if self.bytes > limit {
            return Err("GPU hierarchy view exceeds its aggregate byte budget");
        }
        Ok(())
    }
}

/// Admit the entire shared topology/page/workspace footprint before any GPU
/// allocation or upload. An undersized view must not retry a large tree upload.
fn plan_allocation(
    limits: &wgpu::Limits,
    snapshot: &GpuLodHierarchySnapshot,
    settings: &GpuLodTraversalSettings,
    spatial: Option<(&GaussianLodSpatialTransitionSettings, &GpuLodSpatialMorph)>,
) -> Result<AllocationPlan, String> {
    settings.validate()?;
    let tree = &snapshot.tree;
    if tree.root_count > settings.max_frontier_nodes
        || tree.root_count > settings.max_visited_nodes
        || tree.root_records > settings.max_selected_gaussians
    {
        return Err("GPU traversal budget cannot contain the complete root cut".into());
    }
    let nodes = settings.max_frontier_nodes;
    let (transition_nodes, spatial_bytes) = if let Some((settings, source)) = spatial {
        settings.validate()?;
        if !Arc::ptr_eq(&source.tree, tree) || source.mapping_bytes() > settings.max_mapping_bytes {
            return Err("spatial correspondence does not fit this tree or mapping budget".into());
        }
        (
            settings.max_transition_nodes.min(nodes),
            SpatialMorphState::extra_gpu_bytes(nodes, source)?,
        )
    } else {
        (0, 0)
    };
    let selected_pages = nodes + transition_nodes;
    let bitmap_words = (tree.page_ids.len() as u32).div_ceil(32);
    let offsets = [0, nodes, nodes * 2, nodes * 6];
    let feedback_offsets = [
        offsets[3] + bitmap_words,
        offsets[3] + bitmap_words * 2,
        offsets[3] + bitmap_words * 2 + settings.max_page_requests,
        bitmap_words,
    ];
    let base_work_bytes = HEADER_BYTES
        + (u64::from(feedback_offsets[2] + selected_pages)
            + scan_work_words(nodes, tree.nodes.len() as u32))
            * 4;
    let entry_bytes = (u64::from(settings.max_selected_gaussians) * 8
        + if transition_nodes != 0 {
            32 + 32 * u64::from(nodes)
        } else {
            0
        })
    .max(16);
    let readback_bytes = HEADER_BYTES + u64::from(settings.max_page_requests + selected_pages) * 4;
    let page_bytes = page_allocation_bytes(snapshot);
    let tree_bytes = tree_allocation_bytes(snapshot);
    let fixed_bytes = entry_bytes + readback_bytes * 3 + size_of::<Config>() as u64 + 48;
    let candidate_count = tree.candidates.len() as u32;
    let candidate_bytes = cutoff_work_words(tree.nodes.len() as u32, candidate_count) * 4;
    let cutoff = spatial.is_some()
        && candidate_count != 0
        && tree.node_count() as u64 <= u64::from(settings.max_visited_nodes)
        && candidate_count.div_ceil(256) <= limits.max_compute_workgroups_per_dimension
        && storage_fits(limits, base_work_bytes + candidate_bytes)
        && base_work_bytes
            + candidate_bytes
            + fixed_bytes
            + spatial_bytes
            + page_bytes
            + tree_bytes
            <= settings.max_gpu_bytes;
    let work_bytes = base_work_bytes + if cutoff { candidate_bytes } else { 0 };
    let bytes = work_bytes + fixed_bytes;
    if bytes + spatial_bytes + page_bytes + tree_bytes > settings.max_gpu_bytes
        || (spatial_bytes != 0 && !storage_fits(limits, spatial_bytes))
        || !storage_fits(limits, page_bytes)
        || !storage_fits(limits, tree_node_bytes(snapshot))
        || !storage_fits(limits, tree_order_bytes(snapshot))
        || work_bytes / 4 > u64::from(u32::MAX)
        || (tree.nodes.len() as u32).div_ceil(256) > limits.max_compute_workgroups_per_dimension
        || !storage_fits(limits, work_bytes)
        || !storage_fits(limits, entry_bytes)
        || readback_bytes > limits.max_buffer_size
        || size_of::<Config>() as u64 > limits.max_uniform_buffer_binding_size
        || nodes > limits.max_compute_workgroups_per_dimension
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroup_storage_size < SCAN_WORKGROUP_BYTES
        || limits.max_storage_buffers_per_shader_stage < 6
    {
        return Err("GPU traversal workspace exceeds its byte or device buffer budget".into());
    }
    let config_data = Config {
        world_from_local: Mat4::IDENTITY.to_cols_array(),
        clip_from_world: Mat4::IDENTITY.to_cols_array(),
        view: [1.0, 1.0, 1.0, 0.0],
        quality: [0.0; 4],
        counts: [
            tree.nodes.len() as u32,
            tree.page_ids.len() as u32,
            tree.root_count,
            tree.root_records,
        ],
        limits: [
            settings.max_selected_gaussians,
            nodes,
            settings.max_visited_nodes,
            settings.max_page_requests,
        ],
        offsets,
        feedback_offsets,
        spatial: [
            selected_pages,
            transition_nodes,
            if cutoff {
                ((base_work_bytes - HEADER_BYTES) / 4) as u32
            } else {
                0
            },
            candidate_count,
        ],
        omission: [0.0; 4],
        frustum: [[0.0; 4]; 6],
    };
    Ok(AllocationPlan {
        work_bytes,
        entry_bytes,
        readback_bytes,
        bytes,
        spatial_bytes,
        spatial_settings: spatial.map(|(settings, _)| settings.clone()),
        spatial_source: spatial.map(|(_, source)| source.clone()),
        config: config_data,
    })
}

fn allocate(
    device: &RenderDevice,
    pipeline: &Pipeline,
    snapshot: Arc<GpuSnapshot>,
    settings: &GpuLodTraversalSettings,
    generation: u64,
    plan: AllocationPlan,
    lease: LodMemoryLease,
) -> ViewState {
    let AllocationPlan {
        work_bytes,
        entry_bytes,
        readback_bytes,
        spatial_settings,
        spatial_source,
        config: config_data,
        ..
    } = plan;
    let config = buffer(
        device,
        "lod_traversal_config",
        size_of::<Config>() as u64,
        BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    );
    let feedback = buffer(
        device,
        "lod_traversal_state",
        work_bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    let entries = buffer(
        device,
        "lod_traversal_entries",
        entry_bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    let indirect = buffer(
        device,
        "lod_traversal_draw",
        32,
        BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
    );
    let bind_group = device.create_bind_group(
        "lod_traversal",
        &pipeline.layout,
        &BindGroupEntries::sequential((
            config.as_entire_binding(),
            snapshot.tree.buffer.as_entire_binding(),
            snapshot.pages.as_entire_binding(),
            feedback.as_entire_binding(),
            entries.as_entire_binding(),
            indirect.as_entire_binding(),
            snapshot.tree.source_order.as_entire_binding(),
        )),
    );
    lease.mark_gpu_materialized();
    ViewState {
        output: GpuLodTraversalOutput {
            entries,
            indirect,
            capacity: settings.max_selected_gaussians,
            generation,
            residency_generation: snapshot.data.generation,
            main_cloud: Entity::PLACEHOLDER,
            source: snapshot.data.source,
            feedback,
            submission: 0,
            omission_enabled: false,
            omission_parameters: [0.0; 4],
            omission_frustum: [[0.0; 4]; 6],
            spatial_mapping: None,
            #[cfg(feature = "testing")]
            capture_selected_range: (
                HEADER_BYTES + u64::from(config_data.offsets[2]) * 4,
                settings.max_frontier_nodes,
            ),
            ready: true,
        },
        snapshot,
        settings: settings.clone(),
        config,
        config_data,
        bind_group,
        dispatch: buffer(
            device,
            "lod_traversal_dispatch",
            16,
            BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        ),
        readbacks: std::array::from_fn(|_| Readback {
            buffer: buffer(
                device,
                "lod_traversal_readback",
                readback_bytes,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            ),
            phase: Arc::new(AtomicU8::new(0)),
            submission: 0,
            snapshot: None,
        }),
        spatial_settings,
        spatial_source,
        spatial: None,
        _lease: lease,
    }
}

/// The renderer tests a max-scale world sphere. Inflate the authored local
/// sphere-union box by an upper/lower singular-value ratio so its transformed
/// support still encloses that sphere under nonuniform scale. Gershgorin can
/// decline strongly sheared transforms; declining simply retains every range.
#[allow(clippy::too_many_arguments)]
fn omission_parameters(
    support_sigma: f32,
    global_scale: f32,
    world: Mat4,
    scale_bound: f32,
    projection: Mat4,
    viewport: Vec2,
    frustum_enabled: bool,
    spatial_active: bool,
) -> [f32; 4] {
    // Annotation ownership is independent of whether view omission is safe.
    // Active parent reads still need descriptor updates and page retention when
    // frustum culling or a conservative transform/support admission is disabled.
    let disabled_omission = [0.0, 0.0, 0.0, u32::from(spatial_active) as f32];
    if !(frustum_enabled
        && support_sigma >= 3.0
        && support_sigma.is_finite()
        && global_scale.is_finite()
        && global_scale.abs() <= 1.0)
    {
        return disabled_omission;
    }
    let linear = Mat3::from_mat4(world);
    let gram = linear.transpose() * linear;
    let lower = (gram.x_axis.x - gram.x_axis.y.abs() - gram.x_axis.z.abs())
        .min(gram.y_axis.y - gram.y_axis.x.abs() - gram.y_axis.z.abs())
        .min(gram.z_axis.z - gram.z_axis.x.abs() - gram.z_axis.y.abs());
    if !(lower > 0.0 && lower.is_finite()) {
        return disabled_omission;
    }
    let inflation = (scale_bound / lower.sqrt()).max(1.0);
    let focal =
        (projection.x_axis.x.abs() * viewport.x).min(projection.y_axis.y.abs() * viewport.y);
    let mip = 3.0 * (4.0 * crate::render::GAUSSIAN_MIP_FILTER_VARIANCE_2D).sqrt() / focal;
    if !(inflation >= 1.0 && inflation.is_finite() && mip >= 0.0 && mip.is_finite())
        || !(projection.w_axis.w == 0.0 || projection.w_axis.w == 1.0)
    {
        return disabled_omission;
    }
    [
        inflation * (1.0 + 64.0 * f32::EPSILON),
        mip * (1.0 + 64.0 * f32::EPSILON),
        u32::from(projection.w_axis.w == 0.0) as f32,
        u32::from(spatial_active) as f32,
    ]
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prepare(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    mut pipelines: ResMut<Pipelines>,
    mut spatial_pipelines: ResMut<SpatialMorphPipelines>,
    ledger: Res<LodMemoryLedger>,
    mut outputs: ResMut<GpuLodTraversalOutputs>,
    feedbacks: Res<GpuLodTraversalFeedbacks>,
    views: Query<(
        &ExtractedView,
        &GpuLodTraversalSettings,
        &RenderVisibleEntities,
        Option<&GaussianPointSplattingSettings>,
        Option<&GaussianGlobalOrderSettings>,
        Option<&GaussianLodSpatialTransitionSettings>,
        Option<&Frustum>,
    )>,
    clouds: Query<(
        Option<&GpuLodHierarchy>,
        Option<&PlanarGaussian3dHandle>,
        Option<&LodRenderCandidates>,
        &GlobalTransform,
        Option<&GaussianLodSettings>,
        &CloudSettings,
        Option<&GpuLodSpatialMorph>,
    )>,
    assets: Res<RenderAssets<PlanarStorageGaussian3d>>,
    compacted: Res<LodCompactionBuffers<Gaussian3d>>,
    atlas_generations: Res<LodAtlasGpuGenerations>,
) {
    let mut live = HashSet::new();
    for state in outputs.states.values_mut() {
        state.output.ready = false;
    }
    for (view, settings, visible, point_settings, ordered_settings, spatial_settings, frustum) in
        &views
    {
        let Some(visible) = visible.get::<CloudVisibilityClass>() else {
            continue;
        };
        let mut requested = Vec::new();
        let mut fixed_records = Some(0u32);
        for (entity, main_entity) in &visible.entities_cpu_culling {
            let Ok((hierarchy, handle, candidates, transform, quality, cloud, spatial_source)) =
                clouds.get(*entity)
            else {
                continue;
            };
            if (point_settings.is_some() && !point_splatting_for_cloud(point_settings, cloud))
                || (point_settings.is_none()
                    && ordered_settings.is_some()
                    && !global_order_for_cloud(ordered_settings, cloud))
            {
                continue;
            }
            let Some(hierarchy) = hierarchy else {
                // Reserve the same prepared source capacity consumed later by
                // either shared renderer. Unknown inputs defer the entire cut.
                if point_settings.is_none() && ordered_settings.is_none() {
                    continue;
                }
                if candidates
                    .and_then(|set| {
                        set.by_camera
                            .get(&view.retained_view_entity.main_entity.id())
                    })
                    .is_some_and(|candidate| {
                        if point_settings.is_some() {
                            !crate::render::point::supports_candidate(candidate)
                        } else {
                            !crate::render::ordered::supports_candidate(candidate)
                        }
                    })
                {
                    continue;
                }
                let reservation = handle.and_then(|handle| {
                    fixed_source(
                        view.retained_view_entity,
                        *entity,
                        handle,
                        candidates,
                        &assets,
                        &compacted,
                    )
                    .ok()
                });
                fixed_records = fixed_records.and_then(|total| {
                    reservation.and_then(|source| total.checked_add(source.capacity))
                });
                continue;
            };
            let key = (view.retained_view_entity, *entity);
            live.insert(key);
            requested.push((
                *entity,
                main_entity.id(),
                hierarchy.clone(),
                transform,
                quality,
                cloud.global_scale,
                if point_settings.is_none() && ordered_settings.is_some() {
                    spatial_settings.zip(spatial_source)
                } else {
                    None
                },
            ));
        }
        let Some(fixed_records) = fixed_records else {
            continue;
        };
        if requested.is_empty()
            || settings.validate().is_err()
            || point_settings.is_some_and(|settings| settings.validate().is_err())
            || (point_settings.is_none()
                && ordered_settings.is_some_and(|settings| settings.validate().is_err()))
        {
            continue;
        }
        let publication_ready = |snapshot: &Arc<GpuLodHierarchySnapshot>| {
            snapshot
                .required_atlas_slots
                .iter()
                .all(|slot| atlas_generations.is_current(snapshot.source.untyped(), *slot))
                && assets.get(snapshot.source).is_some_and(|source| {
                    snapshot.placements.iter().all(|page| {
                        u64::from(page.start) + u64::from(page.count) <= source.len() as u64
                    })
                })
        };
        let mut waiting_for_publication = false;
        for (entity, _, hierarchy, _, _, _, _) in &mut requested {
            if publication_ready(&hierarchy.0) {
                continue;
            }
            let previous = outputs.states.get(&(view.retained_view_entity, *entity));
            if let Some(previous) = previous.filter(|state| {
                state.snapshot.data.source == hierarchy.0.source
                    && Arc::ptr_eq(&state.snapshot.data.tree, &hierarchy.0.tree)
                    && publication_ready(&state.snapshot.data)
            }) {
                // A cold publication must not blank a still-authenticated cut.
                // Re-select its resident pages with this frame's camera below.
                *hierarchy = GpuLodHierarchy(previous.snapshot.data.clone());
            } else {
                waiting_for_publication = true;
            }
        }
        if waiting_for_publication {
            continue;
        }
        // Capacity is a view-wide contract, including the root cuts of every
        // visible cloud. Validate all buffers and the aggregate footprint before
        // compiling pipelines, allocating resources or queuing a topology upload.
        let roots = requested
            .iter()
            .map(|(entity, _, hierarchy, _, _, _, _)| {
                (*entity, hierarchy.0.tree.root_gaussian_count())
            })
            .collect::<Vec<_>>();
        let admission = admit_traversal_records(
            point_settings.map_or_else(
                || {
                    ordered_settings.map_or(settings.max_selected_gaussians, |settings| {
                        settings.max_projected_gaussians
                    })
                },
                |settings| settings.max_projected_gaussians,
            ),
            settings.max_selected_gaussians,
            fixed_records,
            &roots,
        );
        let plans = admission.and_then(|capacities| {
            let mut plans = HashMap::new();
            let mut footprint = ViewFootprint::default();
            let by_entity = requested
                .iter()
                .map(|request| (request.0, request))
                .collect::<HashMap<_, _>>();
            for (entity, capacity) in capacities {
                let (_, _, hierarchy, _, quality, _, spatial) = by_entity[&entity];
                if quality
                    .is_some_and(|settings| settings.selection_mode == LodSelectionMode::Frozen)
                {
                    return Err("GPU hierarchy traversal requires Dynamic selection");
                }
                let mut admitted_settings = settings.clone();
                admitted_settings.max_selected_gaussians = capacity;
                let plan =
                    plan_allocation(&device.limits(), &hierarchy.0, &admitted_settings, *spatial)
                        .map_err(|_| "GPU hierarchy buffer, root or device admission failed")?;
                footprint.admit(&hierarchy.0, &plan, settings.max_gpu_bytes)?;
                plans.insert(entity, (admitted_settings, plan));
            }
            Ok(plans)
        });
        let mut plans = match plans {
            Ok(plans) => plans,
            Err(error) => {
                debug!(view = ?view.retained_view_entity, %error, "GPU hierarchy admission deferred");
                continue;
            }
        };
        let pipeline = pipelines
            .0
            .get_or_insert_with(|| Pipeline::new(&device, &cache));
        if pipeline
            .stages
            .iter()
            .any(|id| cache.get_compute_pipeline(*id).is_none())
        {
            continue;
        }
        for (entity, main_entity, hierarchy, transform, quality, global_scale, spatial) in requested
        {
            let key = (view.retained_view_entity, entity);
            let (settings, plan) = plans.remove(&entity).unwrap();
            let replace = outputs.states.get(&key).is_none_or(|state| {
                !Arc::ptr_eq(&state.snapshot.data.tree, &hierarchy.0.tree)
                    || state.snapshot.data.source != hierarchy.0.source
                    || state.settings != settings
                    || state.spatial_settings.as_ref() != spatial.map(|(settings, _)| settings)
                    || match (state.spatial_source.as_ref(), spatial) {
                        (Some(old), Some((_, new))) => !Arc::ptr_eq(&old.words, &new.words),
                        (None, None) => false,
                        _ => true,
                    }
            });
            if replace {
                let result: Result<ViewState, String> = (|| {
                    // Reserve workspace before shared buffers. ensure_snapshot
                    // reserves page and topology leases before its first write;
                    // no fallible admission remains after an upload is queued.
                    let lease = reserve(&ledger, plan.bytes)?;
                    let snapshot =
                        ensure_snapshot(&mut outputs, &device, &queue, &ledger, &hierarchy.0)?;
                    outputs.next_generation = outputs.next_generation.wrapping_add(1).max(1);
                    Ok(allocate(
                        &device,
                        pipeline,
                        snapshot,
                        &settings,
                        outputs.next_generation,
                        plan,
                        lease,
                    ))
                })();
                match result {
                    Ok(mut state) => {
                        state.output.submission = outputs
                            .states
                            .get(&key)
                            .map_or(0, |old| old.output.submission);
                        if let Some(old) = outputs.states.insert(key, state) {
                            queue.on_submitted_work_done(move || drop(old));
                        }
                    }
                    Err(error) => {
                        debug!(?entity, %error, "GPU hierarchy admission deferred");
                        continue;
                    }
                }
            }
            if outputs
                .states
                .get(&key)
                .is_some_and(|state| !Arc::ptr_eq(&state.snapshot.data, &hierarchy.0))
            {
                // Page publication changes bindings, not the current-view
                // workspace. Keep the feedback ring alive during streaming.
                let snapshot =
                    match ensure_snapshot(&mut outputs, &device, &queue, &ledger, &hierarchy.0) {
                        Ok(snapshot) => snapshot,
                        Err(error) => {
                            debug!(?entity, %error, "GPU hierarchy publication deferred");
                            let previous = &outputs.states[&key].snapshot;
                            if !publication_ready(&previous.data) {
                                continue;
                            }
                            previous.clone()
                        }
                    };
                let state = outputs.states.get_mut(&key).unwrap();
                state.bind_group = device.create_bind_group(
                    "lod_traversal",
                    &pipeline.layout,
                    &BindGroupEntries::sequential((
                        state.config.as_entire_binding(),
                        snapshot.tree.buffer.as_entire_binding(),
                        snapshot.pages.as_entire_binding(),
                        state.output.feedback.as_entire_binding(),
                        state.output.entries.as_entire_binding(),
                        state.output.indirect.as_entire_binding(),
                        snapshot.tree.source_order.as_entire_binding(),
                    )),
                );
                state.output.residency_generation = snapshot.data.generation;
                let old = std::mem::replace(&mut state.snapshot, snapshot);
                queue.on_submitted_work_done(move || drop(old));
                if let Some(mut spatial) = state.spatial.take() {
                    spatial.rebind_snapshot(&device, &spatial_pipelines, state.spatial_buffers());
                    state.spatial = Some(spatial);
                }
            }
            let state = outputs.states.get_mut(&key).unwrap();
            state.output.main_cloud = main_entity;
            let world = transform.to_matrix();
            let linear = Mat3::from_mat4(world);
            let gram = linear.transpose() * linear;
            let scale = gram
                .x_axis
                .abs()
                .element_sum()
                .max(gram.y_axis.abs().element_sum())
                .max(gram.z_axis.abs().element_sum())
                .sqrt();
            let clip = view.clip_from_world.unwrap_or_else(|| {
                view.clip_from_view * view.world_from_view.to_matrix().inverse()
            });
            if !world.is_finite()
                || !clip.is_finite()
                || !clip.determinant().is_finite()
                || clip.determinant() == 0.0
                || !scale.is_finite()
                || view.viewport.z == 0
                || view.viewport.w == 0
            {
                continue;
            }
            state.config_data.world_from_local = world.to_cols_array();
            state.config_data.clip_from_world = clip.to_cols_array();
            state.config_data.view = [
                view.viewport.z as f32,
                view.viewport.w as f32,
                scale,
                quality.map_or(0.0, |settings| settings.frustum_margin),
            ];
            state.config_data.quality = match quality.map_or(
                LodQualityTarget::Original,
                GaussianLodSettings::quality_target,
            ) {
                LodQualityTarget::Coarsest => [0.0; 4],
                LodQualityTarget::Original => [1.0, 0.0, 2.0, 0.0],
                LodQualityTarget::Balanced {
                    detail_fraction,
                    max_error_px,
                } => [detail_fraction, max_error_px, 1.0, 0.0],
            };
            state.config_data.quality[3] =
                u32::from(quality.is_none_or(|settings| settings.frustum_culling)) as f32;

            // The view state already owns all queued traversal uploads before
            // optional annotation admission can fail. Keep the ordinary cut live.
            if state.spatial.is_none()
                && let (Some(settings), Some(source)) =
                    (&state.spatial_settings, &state.spatial_source)
            {
                match SpatialMorphState::new(
                    &device,
                    &queue,
                    &cache,
                    &mut spatial_pipelines,
                    &ledger,
                    source,
                    settings,
                    state.spatial_buffers(),
                ) {
                    Ok(spatial) => state.spatial = Some(spatial),
                    Err(error) => debug!(?entity, %error, "GPU spatial annotation deferred"),
                }
            }
            state.output.spatial_mapping = state
                .spatial
                .as_ref()
                .filter(|spatial| {
                    state.config_data.quality[2] == 1.0
                        && spatial.is_ready(&cache, &spatial_pipelines)
                })
                .map(|spatial| spatial.mapping_buffer().clone());
            state.config_data.omission = omission_parameters(
                state.snapshot.data.tree.support_sigma,
                global_scale,
                world,
                scale,
                view.clip_from_view,
                Vec2::new(view.viewport.z as f32, view.viewport.w as f32),
                state.config_data.quality[3] != 0.0 && frustum.is_some(),
                state.output.spatial_mapping.is_some(),
            );
            state.config_data.frustum = frustum.map_or([[0.0; 4]; 6], |frustum| {
                frustum.half_spaces.map(|plane| plane.normal_d().to_array())
            });
            state.output.omission_enabled = state.config_data.omission[0] > 0.0;
            state.output.omission_parameters = state.config_data.omission;
            state.output.omission_frustum = state.config_data.frustum;
            queue.write_buffer(&state.config, 0, bytemuck::bytes_of(&state.config_data));
            state.output.ready = true;
        }
    }
    let expired: Vec<_> = outputs
        .states
        .keys()
        .filter(|key| !live.contains(key))
        .copied()
        .collect();
    for key in expired {
        if let Some(old) = outputs.states.remove(&key) {
            queue.on_submitted_work_done(move || drop(old));
        }
    }
    outputs.trees.retain(|_, value| value.strong_count() > 0);
    outputs
        .snapshots
        .retain(|_, value| value.strong_count() > 0);
    feedbacks.0.lock().unwrap().retain(|(camera, cloud), _| {
        outputs.states.iter().any(|((view, _), state)| {
            view.main_entity.id() == *camera && state.output.main_cloud == *cloud
        })
    });
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

fn dispatch_pass(
    context: &mut RenderContext,
    pipelines: &[&ComputePipeline],
    bindings: &BindGroup,
    label: &'static str,
    indirect: &Buffer,
    dispatches: impl IntoIterator<Item = (usize, Option<u32>)>,
) {
    let mut pass = context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            ..default()
        });
    pass.set_bind_group(0, bindings, &[]);
    for (stage, direct) in dispatches {
        pass.set_pipeline(pipelines[stage]);
        if let Some(groups) = direct {
            pass.dispatch_workgroups(groups, 1, 1);
        } else {
            pass.dispatch_workgroups_indirect(indirect, 0);
        }
    }
}

fn render(
    mut context: RenderContext,
    cache: Res<PipelineCache>,
    pipelines: Res<Pipelines>,
    spatial_pipelines: Res<SpatialMorphPipelines>,
    mut outputs: ResMut<GpuLodTraversalOutputs>,
    view: ViewQuery<&ExtractedView>,
) {
    let Some(pipeline) = &pipelines.0 else {
        return;
    };
    let view = view.into_inner().retained_view_entity;
    for ((key, _), state) in &mut outputs.states {
        if *key != view || !state.output.ready {
            continue;
        }
        let stages = pipeline
            .stages
            .each_ref()
            .map(|id| cache.get_compute_pipeline(*id).unwrap());
        state.output.submission = state.output.submission.wrapping_add(1).max(1);
        let config = state.config_data;
        traversal_schedule(
            &config,
            state.snapshot.data.tree.max_depth,
            |copy_args, label, steps| {
                if steps.is_empty() {
                    if state.output.spatial_mapping.is_some()
                        && let Some(spatial) = &state.spatial
                    {
                        spatial.encode(&mut context, &cache, &spatial_pipelines);
                    }
                    return;
                }
                if copy_args {
                    context.command_encoder().copy_buffer_to_buffer(
                        &state.output.feedback,
                        0,
                        &state.dispatch,
                        0,
                        12,
                    );
                }
                dispatch_pass(
                    &mut context,
                    &stages,
                    &state.bind_group,
                    label,
                    &state.dispatch,
                    steps.iter().copied(),
                );
            },
        );
        // Rendering continues when feedback is busy. Snapshot ownership remains
        // pinned until a later complete result can inform the publisher.
        if let Some(slot) = state
            .readbacks
            .iter_mut()
            .find(|slot| slot.phase.load(Ordering::Acquire) == 0)
        {
            let encoder = context.command_encoder();
            encoder.copy_buffer_to_buffer(&state.output.feedback, 0, &slot.buffer, 0, HEADER_BYTES);
            let requests_size = u64::from(state.settings.max_page_requests) * 4;
            encoder.copy_buffer_to_buffer(
                &state.output.feedback,
                HEADER_BYTES + u64::from(config.feedback_offsets[1]) * 4,
                &slot.buffer,
                HEADER_BYTES,
                requests_size,
            );
            encoder.copy_buffer_to_buffer(
                &state.output.feedback,
                HEADER_BYTES + u64::from(config.feedback_offsets[2]) * 4,
                &slot.buffer,
                HEADER_BYTES + requests_size,
                u64::from(state.config_data.spatial[0]) * 4,
            );
            slot.submission = state.output.submission;
            slot.snapshot = Some(state.snapshot.clone());
            slot.phase.store(1, Ordering::Release);
        }
    }
}

fn deduplicate_page_requests(requests: &[u32], page_ids: &[LodPageId]) -> Option<Vec<LodPageId>> {
    let mut pages = Vec::with_capacity(requests.len());
    let mut seen = HashSet::with_capacity(requests.len());
    for &page in requests {
        let page = *page_ids.get(page as usize)?;
        if seen.insert(page) {
            pages.push(page);
        }
    }
    Some(pages)
}

fn collect(
    mut outputs: ResMut<GpuLodTraversalOutputs>,
    device: Res<RenderDevice>,
    feedbacks: Res<GpuLodTraversalFeedbacks>,
    ledger: Res<LodMemoryLedger>,
) {
    let _ = device.poll(PollType::Poll);
    for ((view, _), state) in &mut outputs.states {
        for slot in &mut state.readbacks {
            match slot.phase.load(Ordering::Acquire) {
                1 => {
                    let phase = slot.phase.clone();
                    slot.phase.store(2, Ordering::Release);
                    slot.buffer
                        .slice(..)
                        .map_async(MapMode::Read, move |result| {
                            phase.store(if result.is_ok() { 3 } else { 4 }, Ordering::Release)
                        });
                }
                3 => {
                    let snapshot = slot.snapshot.take().expect("submitted traversal snapshot");
                    let bytes = slot.buffer.slice(..).get_mapped_range();
                    let words: &[u32] = bytemuck::cast_slice(&bytes);
                    let request_count = words[10].min(state.settings.max_page_requests) as usize;
                    let selected_count = words[11].min(state.config_data.spatial[0]) as usize;
                    let page =
                        |index: &u32| snapshot.data.tree.page_ids.get(*index as usize).copied();
                    let selected_start = 16 + state.settings.max_page_requests as usize;
                    // Hash membership preserves the GPU's required-before-prefetch
                    // prefix. Bound and reserve its temporary buckets/control bytes
                    // before allocating; feedback admission never blocks display.
                    let Ok(_dedup_scratch) = ledger.try_reserve(
                        LodMemoryCategory::MetadataCpu,
                        request_count as u64 * 32 + 256,
                    ) else {
                        drop(bytes);
                        slot.buffer.unmap();
                        slot.phase.store(0, Ordering::Release);
                        continue;
                    };
                    let Some(requested_pages) = deduplicate_page_requests(
                        &words[16..16 + request_count],
                        &snapshot.data.tree.page_ids,
                    ) else {
                        drop(bytes);
                        slot.buffer.unmap();
                        slot.phase.store(0, Ordering::Release);
                        continue;
                    };
                    let result = GpuLodTraversalFeedback {
                        generation: snapshot.data.generation,
                        submission: slot.submission,
                        source: snapshot.data.source,
                        requested_pages,
                        selected_pages: words[selected_start..selected_start + selected_count]
                            .iter()
                            .filter_map(page)
                            .collect(),
                        selected_gaussians: words[8],
                        visited_nodes: words[9],
                        complete: words[12] & 1 == 0,
                        record_limited: words[12] & 2 != 0,
                        frontier_limited: words[12] & 4 != 0,
                        request_overflow: words[12] & 8 != 0,
                        visit_limited: words[12] & 16 != 0,
                        cutoff_unavailable: words[12] & 32 != 0,
                    };
                    drop(bytes);
                    slot.buffer.unmap();
                    slot.phase.store(0, Ordering::Release);
                    let mut mailbox = feedbacks.0.lock().unwrap();
                    let key = (view.main_entity.id(), state.output.main_cloud);
                    if mailbox.get(&key).is_none_or(|old| {
                        (result.generation, result.submission) >= (old.generation, old.submission)
                    }) {
                        mailbox.insert(key, result);
                    }
                }
                4 => {
                    slot.buffer.unmap();
                    slot.snapshot = None;
                    slot.phase.store(0, Ordering::Release);
                }
                _ => {}
            }
        }
    }
}
