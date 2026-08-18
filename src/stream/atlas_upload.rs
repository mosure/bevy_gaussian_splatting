//! Bounded GPU subrange uploads for streamed 3D Gaussian atlas slots.
//!
//! Ordinary [`PlanarGaussian3d`] asset changes continue through Bevy's
//! [`RenderAsset`](bevy::render::render_asset::RenderAsset) preparation path.
//! The streaming bridge owns fixed-size atlas assets, however, and can update a
//! single physical slot without cloning and recreating the complete GPU asset.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    num::{NonZeroU32, NonZeroU64},
};

use bevy::{
    asset::{AssetId, Assets, UntypedAssetId},
    ecs::system::SystemParam,
    prelude::*,
    render::{
        ExtractSchedule, GpuResourceAppExt, MainWorld, Render, RenderApp, RenderSystems,
        render_asset::{RenderAssets, prepare_assets},
        render_resource::{BufferInitDescriptor, BufferUsages, CommandEncoderDescriptor},
        renderer::{RenderDevice, RenderQueue},
    },
};
use bevy_interleave::prelude::Planar;

use bytemuck::Pod;
#[cfg(feature = "precompute_covariance_3d")]
use bytemuck::Zeroable;

use crate::{
    gaussian::formats::planar_3d::{PlanarGaussian3d, PlanarStorageGaussian3d},
    stream::cache::AtlasSlot,
};

#[cfg(any(test, lod_render_path))]
use crate::stream::runtime::LodPhysicalRange;

/// A main-world request to upload the final CPU contents of one physical
/// atlas slot.
///
/// Requests contain no Gaussian payload. Extraction snapshots the final atlas
/// contents after all main-world systems have run, so an ordinary asset edit
/// that happens earlier in the frame cannot be overwritten by an older page
/// payload. Repeated writes to the same physical slot are coalesced; the last
/// request (and therefore its generation) wins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodAtlasSlotUpload {
    pub atlas: AssetId<PlanarGaussian3d>,
    pub slot: AtlasSlot,
    pub gaussians_per_slot: u32,
}

impl LodAtlasSlotUpload {
    fn physical_start(self) -> Result<u32, LodAtlasUploadError> {
        self.slot
            .index
            .checked_mul(self.gaussians_per_slot)
            .ok_or(LodAtlasUploadError::AddressOverflow)
    }

    fn physical_end(self) -> Result<u32, LodAtlasUploadError> {
        self.physical_start()?
            .checked_add(self.gaussians_per_slot)
            .ok_or(LodAtlasUploadError::AddressOverflow)
    }

    fn validate_address(self) -> Result<(), LodAtlasUploadError> {
        if self.gaussians_per_slot == 0 {
            return Err(LodAtlasUploadError::ZeroSlotStride);
        }
        self.physical_end()?;
        Ok(())
    }

    fn validate_resident(self) -> Result<(), LodAtlasUploadError> {
        self.validate_address()?;
        if self.slot.generation == 0 {
            return Err(LodAtlasUploadError::ZeroGeneration);
        }
        Ok(())
    }
}

/// Coalescing main-world queue for bridge- or package-owned atlas writes.
#[derive(Resource, Default, Debug)]
pub struct LodAtlasUploadQueue {
    slots: HashMap<(AssetId<PlanarGaussian3d>, u32), LodAtlasSlotUpload>,
}

impl LodAtlasUploadQueue {
    /// Queues one complete fixed-stride physical slot. The atlas is sampled at
    /// extraction time; callers must mutate its CPU mirror before enqueueing.
    pub fn enqueue_slot(
        &mut self,
        atlas: AssetId<PlanarGaussian3d>,
        slot: AtlasSlot,
        gaussians_per_slot: u32,
    ) -> Result<(), LodAtlasUploadError> {
        let upload = LodAtlasSlotUpload {
            atlas,
            slot,
            gaussians_per_slot,
        };
        upload.validate_resident()?;
        self.slots.insert((atlas, slot.index), upload);
        Ok(())
    }

    /// Queues every physical slot without publishing allocator-generation
    /// proofs. This is the bounded fallback for an in-place source mutation:
    /// source-covered and padded slots may all have changed, including slots
    /// that were never resident in the current hierarchy cut.
    pub fn enqueue_complete_atlas(
        &mut self,
        atlas: AssetId<PlanarGaussian3d>,
        physical_gaussians: u32,
        gaussians_per_slot: u32,
    ) -> Result<(), LodAtlasUploadError> {
        if gaussians_per_slot == 0 {
            return Err(LodAtlasUploadError::ZeroSlotStride);
        }
        if physical_gaussians == 0 || !physical_gaussians.is_multiple_of(gaussians_per_slot) {
            return Err(LodAtlasUploadError::InvalidAtlasLength {
                physical_gaussians,
                gaussians_per_slot,
            });
        }
        let slot_count = physical_gaussians / gaussians_per_slot;
        let reserved_slots = usize::try_from(slot_count)
            .map_err(|_| LodAtlasUploadError::QueueAllocationFailed { slot_count })?;
        self.slots
            .try_reserve(reserved_slots)
            .map_err(|_| LodAtlasUploadError::QueueAllocationFailed { slot_count })?;
        for index in 0..slot_count {
            self.enqueue_cleared_slot(atlas, index, gaussians_per_slot)?;
        }
        Ok(())
    }

    /// Queues one physical slot while invalidating (rather than publishing) a
    /// residency generation. This is used when a cut clears a formerly active
    /// slot and has no allocator generation that may safely be rendered.
    pub fn enqueue_cleared_slot(
        &mut self,
        atlas: AssetId<PlanarGaussian3d>,
        slot_index: u32,
        gaussians_per_slot: u32,
    ) -> Result<(), LodAtlasUploadError> {
        let upload = LodAtlasSlotUpload {
            atlas,
            slot: AtlasSlot {
                index: slot_index,
                // Zero explicitly means "invalidate only". Future ACTIVE
                // frontiers still require an allocator-issued generation.
                generation: 0,
            },
            gaussians_per_slot,
        };
        upload.validate_address()?;
        self.slots.insert((atlas, slot_index), upload);
        Ok(())
    }

    pub fn queued_slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Read-only inspection for integration tests and package orchestration.
    pub fn queued_slots(&self) -> impl Iterator<Item = LodAtlasSlotUpload> + '_ {
        self.slots.values().copied()
    }
}

/// Global canonical-atlas work admitted to one render frame.
///
/// This bound is intentionally separate from the per-cloud atomic commit
/// bound. Multiple clouds may commit in the same application frame, while the
/// render bridge must keep their aggregate CPU snapshots and GPU staging work
/// finite. A physical slot remains atomic: if one slot is larger than the byte
/// limit it is deferred and reported through [`LodAtlasUploadBudgetStatus`].
#[derive(Resource, Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodAtlasUploadBudget {
    max_canonical_bytes_per_frame: NonZeroU64,
    max_slots_per_frame: NonZeroU32,
}

impl LodAtlasUploadBudget {
    pub const DEFAULT_MAX_CANONICAL_BYTES_PER_FRAME: u64 = 64 * 1024 * 1024;
    pub const DEFAULT_MAX_SLOTS_PER_FRAME: u32 = 256;

    pub const fn try_new(
        max_canonical_bytes_per_frame: u64,
        max_slots_per_frame: u32,
    ) -> Result<Self, LodAtlasUploadBudgetError> {
        let Some(max_canonical_bytes_per_frame) = NonZeroU64::new(max_canonical_bytes_per_frame)
        else {
            return Err(LodAtlasUploadBudgetError::ZeroCanonicalByteLimit);
        };
        let Some(max_slots_per_frame) = NonZeroU32::new(max_slots_per_frame) else {
            return Err(LodAtlasUploadBudgetError::ZeroSlotLimit);
        };
        Ok(Self {
            max_canonical_bytes_per_frame,
            max_slots_per_frame,
        })
    }

    pub const fn max_canonical_bytes_per_frame(self) -> u64 {
        self.max_canonical_bytes_per_frame.get()
    }

    pub const fn max_slots_per_frame(self) -> u32 {
        self.max_slots_per_frame.get()
    }

    pub fn set_max_canonical_bytes_per_frame(
        &mut self,
        value: u64,
    ) -> Result<(), LodAtlasUploadBudgetError> {
        self.max_canonical_bytes_per_frame =
            NonZeroU64::new(value).ok_or(LodAtlasUploadBudgetError::ZeroCanonicalByteLimit)?;
        Ok(())
    }

    pub fn set_max_slots_per_frame(&mut self, value: u32) -> Result<(), LodAtlasUploadBudgetError> {
        self.max_slots_per_frame =
            NonZeroU32::new(value).ok_or(LodAtlasUploadBudgetError::ZeroSlotLimit)?;
        Ok(())
    }
}

impl Default for LodAtlasUploadBudget {
    fn default() -> Self {
        Self::try_new(
            Self::DEFAULT_MAX_CANONICAL_BYTES_PER_FRAME,
            Self::DEFAULT_MAX_SLOTS_PER_FRAME,
        )
        .expect("default LoD atlas upload limits are non-zero")
    }
}

/// Typed budget fault observed by the deterministic render-world scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodAtlasUploadBudgetError {
    ZeroCanonicalByteLimit,
    ZeroSlotLimit,
    SlotCanonicalByteOverflow {
        atlas: AssetId<PlanarGaussian3d>,
        slot_index: u32,
    },
    SlotExceedsCanonicalByteLimit {
        atlas: AssetId<PlanarGaussian3d>,
        slot_index: u32,
        required: u64,
        limit: u64,
    },
}

impl std::fmt::Display for LodAtlasUploadBudgetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCanonicalByteLimit => {
                write!(
                    formatter,
                    "LoD atlas per-frame canonical byte limit is zero"
                )
            }
            Self::ZeroSlotLimit => write!(formatter, "LoD atlas per-frame slot limit is zero"),
            Self::SlotCanonicalByteOverflow { atlas, slot_index } => write!(
                formatter,
                "LoD atlas {atlas:?} slot {slot_index} canonical byte count overflowed"
            ),
            Self::SlotExceedsCanonicalByteLimit {
                atlas,
                slot_index,
                required,
                limit,
            } => write!(
                formatter,
                "LoD atlas {atlas:?} slot {slot_index} requires {required} canonical bytes, exceeding the per-frame limit {limit}"
            ),
        }
    }
}

impl std::error::Error for LodAtlasUploadBudgetError {}

/// Main-world status for configuration UIs and orchestration. Oversized
/// atomic slots remain queued and expose a typed error instead of disappearing
/// or bypassing the configured bound.
#[derive(Resource, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LodAtlasUploadBudgetStatus {
    last_error: Option<LodAtlasUploadBudgetError>,
}

impl LodAtlasUploadBudgetStatus {
    pub const fn last_error(self) -> Option<LodAtlasUploadBudgetError> {
        self.last_error
    }
}

#[derive(Clone, Debug)]
struct ExtractedLodAtlasSlotUpload {
    descriptor: LodAtlasSlotUpload,
    planes: PlanarGaussian3d,
}

#[derive(Clone, Debug)]
struct CoalescedLodAtlasUpload {
    descriptors: Vec<LodAtlasSlotUpload>,
    planes: PlanarGaussian3d,
}

impl CoalescedLodAtlasUpload {
    fn start(&self) -> Result<usize, LodAtlasUploadError> {
        self.descriptors
            .first()
            .ok_or(LodAtlasUploadError::EmptyCoalescedRange)?
            .physical_start()
            .map(|start| start as usize)
    }
}

#[cfg(feature = "precompute_covariance_3d")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LodCovarianceUploadRange {
    start: u32,
    count: u32,
}

#[cfg(feature = "precompute_covariance_3d")]
#[derive(Resource)]
struct LodCovariancePipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    max_workgroups_per_dimension: u32,
}

#[cfg(feature = "precompute_covariance_3d")]
impl FromWorld for LodCovariancePipeline {
    fn from_world(world: &mut World) -> Self {
        Self::new(world.resource::<RenderDevice>().wgpu_device())
    }
}

#[cfg(feature = "precompute_covariance_3d")]
impl LodCovariancePipeline {
    fn new(device: &wgpu::Device) -> Self {
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lod_atlas_covariance_layout"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                storage(3, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lod_atlas_covariance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lod_atlas_covariance_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("atlas_covariance.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lod_atlas_covariance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("derive_covariance"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            max_workgroups_per_dimension: device.limits().max_compute_workgroups_per_dimension,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_dispatch_buffers(
        &self,
        render_device: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        rotation: &wgpu::Buffer,
        scale_opacity: &wgpu::Buffer,
        covariance_3d_opacity: &wgpu::Buffer,
        ranges: &[LodCovarianceUploadRange],
    ) -> Result<u32, LodAtlasUploadError> {
        if ranges.is_empty() {
            return Ok(0);
        }
        let max_count = ranges.iter().map(|range| range.count).max().unwrap_or(0);
        let workgroups_x = max_count.div_ceil(64);
        if workgroups_x == 0 || workgroups_x > self.max_workgroups_per_dimension {
            return Err(LodAtlasUploadError::CovarianceDispatchLimit);
        }
        let chunk_size = usize::try_from(self.max_workgroups_per_dimension)
            .map_err(|_| LodAtlasUploadError::AddressOverflow)?;
        let mut prepared = Vec::new();
        prepared
            .try_reserve(ranges.len().div_ceil(chunk_size))
            .map_err(|_| LodAtlasUploadError::CoalescedAllocationFailed)?;
        for (chunk_index, chunk) in ranges.chunks(chunk_size).enumerate() {
            let range_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("lod_atlas_covariance_ranges"),
                contents: bytemuck::cast_slice(chunk),
                usage: BufferUsages::STORAGE,
            });
            let bind_group =
                render_device
                    .wgpu_device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("lod_atlas_covariance_bind_group"),
                        layout: &self.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: rotation.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: scale_opacity.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: covariance_3d_opacity.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: range_buffer.as_entire_binding(),
                            },
                        ],
                    });
            prepared.push((chunk_index, chunk.len() as u32, range_buffer, bind_group));
        }
        for (_, range_count, _, bind_group) in &prepared {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lod_atlas_covariance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, *range_count, 1);
        }
        u32::try_from(prepared.len()).map_err(|_| LodAtlasUploadError::AddressOverflow)
    }
}

#[derive(Clone, Debug)]
enum PendingLodAtlasSlotUpload {
    Ready(ExtractedLodAtlasSlotUpload),
    Invalid(LodAtlasSlotUpload),
}

type LodAtlasUploadKey = (AssetId<PlanarGaussian3d>, u32);

#[derive(Resource, Default, Debug)]
struct LodAtlasUploadScheduler {
    /// First cloud considered on the next admission pass. The cursor stores an
    /// asset id rather than a transient vector offset so insertions/removals do
    /// not make fairness depend on hash-map iteration order.
    next_atlas: Option<AssetId<PlanarGaussian3d>>,
}

#[derive(Default, Debug)]
struct PlannedLodAtlasUploads {
    admitted: Vec<LodAtlasSlotUpload>,
    deferred: Vec<LodAtlasSlotUpload>,
    deferred_canonical_bytes: u64,
    deferred_atlases: BTreeSet<AssetId<PlanarGaussian3d>>,
    oversized_slots: u64,
    first_error: Option<LodAtlasUploadBudgetError>,
}

fn canonical_slot_bytes(descriptor: LodAtlasSlotUpload) -> Result<u64, LodAtlasUploadBudgetError> {
    u64::from(descriptor.gaussians_per_slot)
        .checked_mul(std::mem::size_of::<crate::gaussian::formats::planar_3d::Gaussian3d>() as u64)
        .ok_or(LodAtlasUploadBudgetError::SlotCanonicalByteOverflow {
            atlas: descriptor.atlas,
            slot_index: descriptor.slot.index,
        })
}

fn record_deferred_descriptor(
    plan: &mut PlannedLodAtlasUploads,
    descriptor: LodAtlasSlotUpload,
    budget: LodAtlasUploadBudget,
) {
    plan.deferred_atlases.insert(descriptor.atlas);
    match canonical_slot_bytes(descriptor) {
        Ok(bytes) => {
            plan.deferred_canonical_bytes = plan.deferred_canonical_bytes.saturating_add(bytes);
            if bytes > budget.max_canonical_bytes_per_frame() {
                plan.oversized_slots = plan.oversized_slots.saturating_add(1);
                plan.first_error.get_or_insert(
                    LodAtlasUploadBudgetError::SlotExceedsCanonicalByteLimit {
                        atlas: descriptor.atlas,
                        slot_index: descriptor.slot.index,
                        required: bytes,
                        limit: budget.max_canonical_bytes_per_frame(),
                    },
                );
            }
        }
        Err(error) => {
            plan.oversized_slots = plan.oversized_slots.saturating_add(1);
            plan.first_error.get_or_insert(error);
            plan.deferred_canonical_bytes = u64::MAX;
        }
    }
    plan.deferred.push(descriptor);
}

fn plan_lod_atlas_uploads(
    scheduler: &mut LodAtlasUploadScheduler,
    descriptors: impl IntoIterator<Item = LodAtlasSlotUpload>,
    budget: LodAtlasUploadBudget,
) -> PlannedLodAtlasUploads {
    let mut groups = BTreeMap::<AssetId<PlanarGaussian3d>, VecDeque<LodAtlasSlotUpload>>::new();
    for descriptor in descriptors {
        groups
            .entry(descriptor.atlas)
            .or_default()
            .push_back(descriptor);
    }
    for group in groups.values_mut() {
        group
            .make_contiguous()
            .sort_unstable_by_key(|descriptor| descriptor.slot.index);
    }

    let order = groups.keys().copied().collect::<Vec<_>>();
    let mut plan = PlannedLodAtlasUploads::default();
    if order.is_empty() {
        return plan;
    }
    let mut cursor = scheduler.next_atlas.map_or(0, |next| {
        let offset = order.partition_point(|atlas| *atlas < next);
        if offset == order.len() { 0 } else { offset }
    });
    let mut remaining_bytes = budget.max_canonical_bytes_per_frame();
    let mut remaining_slots = u64::from(budget.max_slots_per_frame());
    let mut blocked = BTreeSet::new();
    let mut last_admitted_index = None;

    while remaining_slots != 0 {
        let mut admitted_in_cycle = false;
        for _ in 0..order.len() {
            let index = cursor;
            cursor = (cursor + 1) % order.len();
            let atlas = order[index];
            if blocked.contains(&atlas) {
                continue;
            }
            let Some(descriptor) = groups
                .get(&atlas)
                .and_then(|descriptors| descriptors.front())
                .copied()
            else {
                continue;
            };
            let bytes = match canonical_slot_bytes(descriptor) {
                Ok(bytes) => bytes,
                Err(error) => {
                    plan.first_error.get_or_insert(error);
                    blocked.insert(atlas);
                    continue;
                }
            };
            if bytes > budget.max_canonical_bytes_per_frame() {
                plan.first_error.get_or_insert(
                    LodAtlasUploadBudgetError::SlotExceedsCanonicalByteLimit {
                        atlas,
                        slot_index: descriptor.slot.index,
                        required: bytes,
                        limit: budget.max_canonical_bytes_per_frame(),
                    },
                );
                blocked.insert(atlas);
                continue;
            }
            if bytes > remaining_bytes {
                // Slot order is a generation/order proof within one atlas, so
                // do not bypass this head with a later, potentially newer slot.
                blocked.insert(atlas);
                continue;
            }
            groups
                .get_mut(&atlas)
                .expect("planned atlas group exists")
                .pop_front();
            plan.admitted.push(descriptor);
            remaining_bytes -= bytes;
            remaining_slots -= 1;
            last_admitted_index = Some(index);
            admitted_in_cycle = true;
            if remaining_slots == 0 {
                break;
            }
        }
        if !admitted_in_cycle {
            break;
        }
    }

    if let Some(index) = last_admitted_index {
        scheduler.next_atlas = Some(order[(index + 1) % order.len()]);
    }
    for (_, descriptors) in groups {
        for descriptor in descriptors {
            record_deferred_descriptor(&mut plan, descriptor, budget);
        }
    }
    plan
}

impl PendingLodAtlasSlotUpload {
    fn descriptor(&self) -> LodAtlasSlotUpload {
        match self {
            Self::Ready(upload) => upload.descriptor,
            Self::Invalid(upload) => *upload,
        }
    }
}

/// Uploads that have crossed into the render world for this preparation pass.
#[derive(Resource, Default)]
struct ExtractedLodAtlasUploads {
    slots: HashMap<LodAtlasUploadKey, PendingLodAtlasSlotUpload>,
    admitted: BTreeSet<LodAtlasUploadKey>,
    invalidations: BTreeSet<LodAtlasUploadKey>,
    frame_budget: LodAtlasUploadBudget,
    deferred_slots: u64,
    deferred_canonical_bytes: u64,
    deferred_atlases: BTreeSet<AssetId<PlanarGaussian3d>>,
    oversized_slots: u64,
}

/// Render-world proof that a physical atlas slot contains a particular
/// allocator generation.
///
/// LoD compaction consults this registry before accepting an ACTIVE bridge
/// frontier. The uploader invalidates a slot before every attempted write, so
/// a missing GPU asset or invalid CPU range cannot accidentally reuse an older
/// proof with the same allocator generation.
#[derive(Resource, Default, Debug)]
pub(crate) struct LodAtlasGpuGenerations {
    slots: HashMap<(UntypedAssetId, u32), u32>,
}

impl LodAtlasGpuGenerations {
    #[cfg(any(test, lod_render_path))]
    pub(crate) fn is_current(&self, atlas: UntypedAssetId, slot: AtlasSlot) -> bool {
        self.slots.get(&(atlas, slot.index)).copied() == Some(slot.generation)
    }

    #[cfg(any(test, lod_render_path))]
    pub(crate) fn frontier_is_current(
        &self,
        atlas: UntypedAssetId,
        ranges: &[LodPhysicalRange],
    ) -> bool {
        ranges
            .iter()
            .all(|range| self.is_current(atlas, range.slot))
    }

    fn invalidate(&mut self, atlas: AssetId<PlanarGaussian3d>, slot_index: u32) {
        self.slots.remove(&(atlas.untyped(), slot_index));
    }

    fn mark_current(&mut self, descriptor: LodAtlasSlotUpload) {
        if descriptor.slot.generation == 0 {
            return;
        }
        self.slots.insert(
            (descriptor.atlas.untyped(), descriptor.slot.index),
            descriptor.slot.generation,
        );
    }
}

/// Installs extraction and render preparation for bounded atlas subranges.
#[derive(Default)]
pub struct GaussianLodAtlasUploadPlugin;

impl Plugin for GaussianLodAtlasUploadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LodAtlasUploadQueue>()
            .init_resource::<LodAtlasUploadBudget>()
            .init_resource::<LodAtlasUploadBudgetStatus>()
            .init_resource::<LodAtlasUploadScheduler>();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ExtractedLodAtlasUploads>()
                .init_gpu_resource::<LodAtlasGpuGenerations>()
                .add_systems(ExtractSchedule, extract_lod_atlas_uploads)
                .add_systems(
                    Render,
                    apply_lod_atlas_uploads
                        .in_set(RenderSystems::PrepareAssets)
                        .after(prepare_assets::<PlanarStorageGaussian3d>),
                );
            #[cfg(feature = "precompute_covariance_3d")]
            render_app.init_gpu_resource::<LodCovariancePipeline>();
        }
    }
}

fn extract_lod_atlas_uploads(
    mut extracted: ResMut<ExtractedLodAtlasUploads>,
    mut main_world: ResMut<MainWorld>,
) {
    let queued = {
        let mut queue = main_world.resource_mut::<LodAtlasUploadQueue>();
        std::mem::take(&mut queue.slots)
    };
    let budget = *main_world.resource::<LodAtlasUploadBudget>();

    // A newer main-world request supersedes any render-world retry for the
    // same physical slot. Invalidate every queued key immediately, including
    // budget-deferred work, so an in-place source mutation can never retain a
    // stale same-generation GPU proof while waiting for admission.
    for key in queued.keys().copied() {
        extracted.invalidations.insert(key);
        extracted.slots.remove(&key);
    }

    let carried = extracted
        .slots
        .values()
        .map(PendingLodAtlasSlotUpload::descriptor)
        .collect::<Vec<_>>();
    let mut queued_descriptors = queued.into_values().collect::<Vec<_>>();
    queued_descriptors.sort_unstable_by_key(|descriptor| (descriptor.atlas, descriptor.slot.index));
    let queued_keys = queued_descriptors
        .iter()
        .map(|descriptor| (descriptor.atlas, descriptor.slot.index))
        .collect::<BTreeSet<_>>();

    let mut plan = {
        let mut scheduler = main_world.resource_mut::<LodAtlasUploadScheduler>();
        if carried.is_empty() {
            plan_lod_atlas_uploads(&mut scheduler, queued_descriptors.iter().copied(), budget)
        } else {
            // Render-world retries already own bounded CPU snapshots. Drain
            // that recovery backlog before cloning any newly queued planes,
            // preventing payload growth while a device/asset is unavailable.
            plan_lod_atlas_uploads(&mut scheduler, carried, budget)
        }
    };
    if !extracted.slots.is_empty() {
        for descriptor in queued_descriptors.iter().copied() {
            record_deferred_descriptor(&mut plan, descriptor, budget);
        }
    }

    extracted.frame_budget = budget;
    extracted.admitted.clear();
    extracted.admitted.extend(
        plan.admitted
            .iter()
            .map(|descriptor| (descriptor.atlas, descriptor.slot.index)),
    );
    extracted.deferred_slots = plan.deferred.len() as u64;
    extracted.deferred_canonical_bytes = plan.deferred_canonical_bytes;
    extracted.deferred_atlases = std::mem::take(&mut plan.deferred_atlases);
    extracted.oversized_slots = plan.oversized_slots;
    main_world
        .resource_mut::<LodAtlasUploadBudgetStatus>()
        .last_error = plan.first_error;

    // Only main-world descriptors are requeued. Carried render-world payloads
    // already remain in `extracted.slots` and are reconsidered next frame.
    {
        let mut queue = main_world.resource_mut::<LodAtlasUploadQueue>();
        for descriptor in plan.deferred {
            let key = (descriptor.atlas, descriptor.slot.index);
            if queued_keys.contains(&key) {
                queue.slots.insert(key, descriptor);
            }
        }
    }

    let assets = main_world.resource::<Assets<PlanarGaussian3d>>();
    for descriptor in plan.admitted {
        let key = (descriptor.atlas, descriptor.slot.index);
        if extracted.slots.contains_key(&key) {
            continue;
        }
        let upload = snapshot_slot(assets.get(descriptor.atlas), descriptor)
            .map(PendingLodAtlasSlotUpload::Ready)
            .unwrap_or(PendingLodAtlasSlotUpload::Invalid(descriptor));
        extracted.slots.insert(key, upload);
    }
}

fn snapshot_slot(
    atlas: Option<&PlanarGaussian3d>,
    descriptor: LodAtlasSlotUpload,
) -> Result<ExtractedLodAtlasSlotUpload, LodAtlasUploadError> {
    descriptor.validate_address()?;
    let atlas = atlas.ok_or(LodAtlasUploadError::MissingAtlasAsset)?;
    let start = descriptor.physical_start()? as usize;
    let end = descriptor.physical_end()? as usize;
    if end > atlas.len() {
        return Err(LodAtlasUploadError::SlotOutOfRange {
            start: start as u64,
            end: end as u64,
            atlas_len: atlas.len() as u64,
        });
    }
    if atlas.spherical_harmonic.len() != atlas.len()
        || atlas.rotation.len() != atlas.len()
        || atlas.scale_opacity.len() != atlas.len()
    {
        return Err(LodAtlasUploadError::InconsistentPlaneLengths);
    }

    Ok(ExtractedLodAtlasSlotUpload {
        descriptor,
        planes: PlanarGaussian3d {
            position_visibility: atlas.position_visibility[start..end].to_vec(),
            spherical_harmonic: atlas.spherical_harmonic[start..end].to_vec(),
            rotation: atlas.rotation[start..end].to_vec(),
            scale_opacity: atlas.scale_opacity[start..end].to_vec(),
        },
    })
}

fn encode_planar_copy<T: Pod>(
    render_device: &RenderDevice,
    encoder: &mut wgpu::CommandEncoder,
    destination: &wgpu::Buffer,
    start: usize,
    values: &[T],
    label: &'static str,
    staging: &mut Vec<bevy::render::render_resource::Buffer>,
) -> Result<(), LodAtlasUploadError> {
    let element_bytes = u64::try_from(std::mem::size_of::<T>())
        .map_err(|_| LodAtlasUploadError::AddressOverflow)?;
    let destination_offset = u64::try_from(start)
        .map_err(|_| LodAtlasUploadError::AddressOverflow)?
        .checked_mul(element_bytes)
        .ok_or(LodAtlasUploadError::AddressOverflow)?;
    let copy_bytes = u64::try_from(values.len())
        .map_err(|_| LodAtlasUploadError::AddressOverflow)?
        .checked_mul(element_bytes)
        .ok_or(LodAtlasUploadError::AddressOverflow)?;
    if copy_bytes == 0 {
        return Err(LodAtlasUploadError::EmptyCoalescedRange);
    }
    let source = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_buffer(&source, 0, destination, destination_offset, copy_bytes);
    staging.push(source);
    Ok(())
}

#[derive(Clone, Copy)]
struct LodAtlasCanonicalBuffers<'a> {
    position_visibility: &'a wgpu::Buffer,
    spherical_harmonic: &'a wgpu::Buffer,
    rotation: &'a wgpu::Buffer,
    scale_opacity: &'a wgpu::Buffer,
    count: usize,
}

fn encode_canonical_buffer_copies(
    render_device: &RenderDevice,
    encoder: &mut wgpu::CommandEncoder,
    atlas: LodAtlasCanonicalBuffers<'_>,
    upload: &CoalescedLodAtlasUpload,
    staging: &mut Vec<bevy::render::render_resource::Buffer>,
) -> Result<(), LodAtlasUploadError> {
    let start = upload.start()?;
    let count = upload.planes.len();
    if count == 0
        || upload.planes.spherical_harmonic.len() != count
        || upload.planes.rotation.len() != count
        || upload.planes.scale_opacity.len() != count
    {
        return Err(LodAtlasUploadError::InconsistentPlaneLengths);
    }
    let end = start
        .checked_add(count)
        .ok_or(LodAtlasUploadError::AddressOverflow)?;
    if end > atlas.count {
        return Err(LodAtlasUploadError::SlotOutOfRange {
            start: start as u64,
            end: end as u64,
            atlas_len: atlas.count as u64,
        });
    }
    staging
        .try_reserve(4)
        .map_err(|_| LodAtlasUploadError::CoalescedAllocationFailed)?;
    encode_planar_copy(
        render_device,
        encoder,
        atlas.position_visibility,
        start,
        &upload.planes.position_visibility,
        "lod_atlas_position_staging",
        staging,
    )?;
    encode_planar_copy(
        render_device,
        encoder,
        atlas.spherical_harmonic,
        start,
        &upload.planes.spherical_harmonic,
        "lod_atlas_sh_staging",
        staging,
    )?;
    encode_planar_copy(
        render_device,
        encoder,
        atlas.rotation,
        start,
        &upload.planes.rotation,
        "lod_atlas_rotation_staging",
        staging,
    )?;
    encode_planar_copy(
        render_device,
        encoder,
        atlas.scale_opacity,
        start,
        &upload.planes.scale_opacity,
        "lod_atlas_scale_opacity_staging",
        staging,
    )?;
    Ok(())
}

fn encode_canonical_atlas_copies(
    render_device: &RenderDevice,
    encoder: &mut wgpu::CommandEncoder,
    atlas: &PlanarStorageGaussian3d,
    upload: &CoalescedLodAtlasUpload,
    staging: &mut Vec<bevy::render::render_resource::Buffer>,
) -> Result<(), LodAtlasUploadError> {
    encode_canonical_buffer_copies(
        render_device,
        encoder,
        LodAtlasCanonicalBuffers {
            position_visibility: &atlas.position_visibility,
            spherical_harmonic: &atlas.spherical_harmonic,
            rotation: &atlas.rotation,
            scale_opacity: &atlas.scale_opacity,
            count: atlas.count,
        },
        upload,
        staging,
    )
}

fn submit_lod_atlas_batch(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    atlas: &PlanarStorageGaussian3d,
    uploads: &[CoalescedLodAtlasUpload],
    #[cfg(feature = "precompute_covariance_3d")] covariance_pipeline: Option<
        &LodCovariancePipeline,
    >,
) -> Result<(), LodAtlasUploadError> {
    if uploads.is_empty() {
        return Err(LodAtlasUploadError::EmptyCoalescedRange);
    }
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("lod_atlas_upload_encoder"),
    });
    let mut staging = Vec::new();
    let result: Result<u32, LodAtlasUploadError> = (|| {
        for upload in uploads {
            encode_canonical_atlas_copies(
                render_device,
                &mut encoder,
                atlas,
                upload,
                &mut staging,
            )?;
        }
        #[cfg(feature = "precompute_covariance_3d")]
        {
            let ranges = uploads
                .iter()
                .map(|upload| {
                    Ok(LodCovarianceUploadRange {
                        start: u32::try_from(upload.start()?)
                            .map_err(|_| LodAtlasUploadError::AddressOverflow)?,
                        count: u32::try_from(upload.planes.len())
                            .map_err(|_| LodAtlasUploadError::AddressOverflow)?,
                    })
                })
                .collect::<Result<Vec<_>, LodAtlasUploadError>>()?;
            covariance_pipeline
                .ok_or(LodAtlasUploadError::CovariancePipelineUnavailable)?
                .encode_dispatch_buffers(
                    render_device,
                    &mut encoder,
                    &atlas.rotation,
                    &atlas.scale_opacity,
                    &atlas.covariance_3d_opacity,
                    &ranges,
                )
        }
        #[cfg(not(feature = "precompute_covariance_3d"))]
        {
            Ok(0_u32)
        }
    })();
    result?;
    render_queue.submit([encoder.finish()]);
    Ok(())
}

#[derive(SystemParam)]
struct LodAtlasUploadGpuParams<'w> {
    gpu_assets: Res<'w, RenderAssets<PlanarStorageGaussian3d>>,
    render_queue: Res<'w, RenderQueue>,
    render_device: Res<'w, RenderDevice>,
    #[cfg(feature = "precompute_covariance_3d")]
    covariance_pipeline: Option<Res<'w, LodCovariancePipeline>>,
}

fn apply_lod_atlas_uploads(
    mut uploads: ResMut<ExtractedLodAtlasUploads>,
    mut generations: ResMut<LodAtlasGpuGenerations>,
    gpu: LodAtlasUploadGpuParams,
) {
    let mut deferred_atlases = std::mem::take(&mut uploads.deferred_atlases);
    uploads.deferred_slots = 0;
    uploads.deferred_canonical_bytes = 0;
    uploads.oversized_slots = 0;
    generations.slots.retain(|(atlas, _), _| {
        atlas
            .try_typed::<PlanarGaussian3d>()
            .is_ok_and(|atlas| gpu.gpu_assets.get(atlas).is_some())
    });
    for (atlas, slot_index) in std::mem::take(&mut uploads.invalidations) {
        generations.invalidate(atlas, slot_index);
    }

    let mut ready = Vec::new();
    for key in std::mem::take(&mut uploads.admitted) {
        let Some(pending) = uploads.slots.remove(&key) else {
            continue;
        };
        let descriptor = pending.descriptor();
        // Invalidate before every attempt. This matters when fallback and page
        // data reuse the same allocator generation in consecutive frames.
        generations.invalidate(descriptor.atlas, descriptor.slot.index);

        let PendingLodAtlasSlotUpload::Ready(upload) = pending else {
            continue;
        };
        ready.push(upload);
    }
    let mut gpu_ready = Vec::new();
    for upload in ready {
        let descriptor = upload.descriptor;
        if gpu.gpu_assets.get(descriptor.atlas).is_none() {
            deferred_atlases.insert(descriptor.atlas);
            uploads.slots.insert(
                (descriptor.atlas, descriptor.slot.index),
                PendingLodAtlasSlotUpload::Ready(upload),
            );
        } else {
            gpu_ready.push(upload);
        }
    }
    let coalesced = match coalesce_atlas_uploads(gpu_ready) {
        Ok(coalesced) => coalesced,
        Err(_) => return,
    };
    let mut batches = BTreeMap::<AssetId<PlanarGaussian3d>, Vec<CoalescedLodAtlasUpload>>::new();
    for upload in coalesced {
        let descriptor = upload.descriptors[0];
        batches.entry(descriptor.atlas).or_default().push(upload);
    }

    for (atlas_id, uploads) in batches {
        let Some(gpu_atlas) = gpu.gpu_assets.get(atlas_id) else {
            // This system runs after RenderAsset preparation. Keep every slot
            // invalid so the bridge retains/restores its complete fallback.
            continue;
        };
        let gpu_result = submit_lod_atlas_batch(
            &gpu.render_device,
            &gpu.render_queue,
            gpu_atlas,
            &uploads,
            #[cfg(feature = "precompute_covariance_3d")]
            gpu.covariance_pipeline.as_deref(),
        );
        if gpu_result.is_ok() {
            for upload in &uploads {
                for descriptor in &upload.descriptors {
                    generations.mark_current(*descriptor);
                }
            }
        } else {
            // Device recreation can temporarily leave the compute resource
            // unavailable. Queue writes remain ordered and recoverable.
            for upload in &uploads {
                let start = match upload.start() {
                    Ok(start) => start,
                    Err(_) => continue,
                };
                if gpu_atlas
                    .write_gaussian_3d_range(&gpu.render_queue, start, &upload.planes)
                    .is_err()
                {
                    continue;
                }
                #[cfg(feature = "precompute_covariance_3d")]
                if gpu_atlas
                    .write_gaussian_3d_covariance_range_cpu(
                        &gpu.render_queue,
                        start,
                        &upload.planes,
                    )
                    .is_err()
                {
                    continue;
                }
                for descriptor in &upload.descriptors {
                    generations.mark_current(*descriptor);
                }
            }
        }
    }
}

fn coalesce_atlas_uploads(
    uploads: Vec<ExtractedLodAtlasSlotUpload>,
) -> Result<Vec<CoalescedLodAtlasUpload>, LodAtlasUploadError> {
    let mut groups = BTreeMap::<(AssetId<PlanarGaussian3d>, u32), Vec<_>>::new();
    for upload in uploads {
        groups
            .entry((
                upload.descriptor.atlas,
                upload.descriptor.gaussians_per_slot,
            ))
            .or_default()
            .push(upload);
    }
    let mut coalesced = Vec::new();
    for (_, mut group) in groups {
        group.sort_unstable_by_key(|upload| upload.descriptor.slot.index);
        let mut current: Option<CoalescedLodAtlasUpload> = None;
        for mut upload in group {
            let contiguous = current.as_ref().is_some_and(|current| {
                current
                    .descriptors
                    .last()
                    .and_then(|descriptor| descriptor.slot.index.checked_add(1))
                    == Some(upload.descriptor.slot.index)
            });
            if !contiguous {
                if let Some(current) = current.take() {
                    coalesced.push(current);
                }
                current = Some(CoalescedLodAtlasUpload {
                    descriptors: Vec::new(),
                    planes: PlanarGaussian3d::default(),
                });
            }
            let current = current.as_mut().expect("coalesced range initialized");
            current.descriptors.push(upload.descriptor);
            current
                .planes
                .position_visibility
                .append(&mut upload.planes.position_visibility);
            current
                .planes
                .spherical_harmonic
                .append(&mut upload.planes.spherical_harmonic);
            current.planes.rotation.append(&mut upload.planes.rotation);
            current
                .planes
                .scale_opacity
                .append(&mut upload.planes.scale_opacity);
        }
        if let Some(current) = current {
            coalesced.push(current);
        }
    }
    Ok(coalesced)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodAtlasUploadError {
    ZeroSlotStride,
    ZeroGeneration,
    AddressOverflow,
    MissingAtlasAsset,
    InconsistentPlaneLengths,
    EmptyCoalescedRange,
    CoalescedAllocationFailed,
    CovariancePipelineUnavailable,
    CovarianceDispatchLimit,
    InvalidAtlasLength {
        physical_gaussians: u32,
        gaussians_per_slot: u32,
    },
    QueueAllocationFailed {
        slot_count: u32,
    },
    SlotOutOfRange {
        start: u64,
        end: u64,
        atlas_len: u64,
    },
}

impl std::fmt::Display for LodAtlasUploadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSlotStride => write!(formatter, "LoD atlas slot stride is zero"),
            Self::ZeroGeneration => write!(formatter, "LoD atlas slot generation is zero"),
            Self::AddressOverflow => write!(formatter, "LoD atlas slot address overflow"),
            Self::MissingAtlasAsset => write!(formatter, "LoD atlas asset is missing"),
            Self::InconsistentPlaneLengths => {
                write!(formatter, "LoD atlas planes have inconsistent lengths")
            }
            Self::EmptyCoalescedRange => {
                write!(formatter, "LoD atlas coalesced range is empty")
            }
            Self::CoalescedAllocationFailed => {
                write!(
                    formatter,
                    "failed to allocate bounded LoD atlas upload batch"
                )
            }
            Self::CovariancePipelineUnavailable => {
                write!(formatter, "LoD covariance compute pipeline is unavailable")
            }
            Self::CovarianceDispatchLimit => write!(
                formatter,
                "LoD covariance upload exceeds adapter dispatch dimensions"
            ),
            Self::InvalidAtlasLength {
                physical_gaussians,
                gaussians_per_slot,
            } => write!(
                formatter,
                "LoD atlas length {physical_gaussians} is not a positive multiple of slot stride {gaussians_per_slot}"
            ),
            Self::QueueAllocationFailed { slot_count } => write!(
                formatter,
                "failed to reserve LoD atlas upload queue for {slot_count} physical slots"
            ),
            Self::SlotOutOfRange {
                start,
                end,
                atlas_len,
            } => write!(
                formatter,
                "LoD atlas slot range {start}..{end} exceeds atlas length {atlas_len}"
            ),
        }
    }
}

impl std::error::Error for LodAtlasUploadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::formats::planar_3d::Gaussian3d;

    fn atlas_id(value: u128) -> AssetId<PlanarGaussian3d> {
        AssetId::Uuid {
            uuid: bevy::asset::uuid::Uuid::from_u128(value),
        }
    }

    fn descriptor(
        atlas: AssetId<PlanarGaussian3d>,
        slot_index: u32,
        gaussians_per_slot: u32,
    ) -> LodAtlasSlotUpload {
        LodAtlasSlotUpload {
            atlas,
            slot: AtlasSlot {
                index: slot_index,
                generation: slot_index + 1,
            },
            gaussians_per_slot,
        }
    }

    fn gaussian(x: f32) -> Gaussian3d {
        Gaussian3d {
            position_visibility: [x, x + 1.0, x + 2.0, 1.0].into(),
            rotation: [1.0, 0.0, 0.0, x].into(),
            scale_opacity: [x + 0.1, x + 0.2, x + 0.3, 0.5].into(),
            ..default()
        }
    }

    #[test]
    fn queue_coalesces_physical_slot_and_keeps_latest_generation() {
        let atlas = AssetId::<PlanarGaussian3d>::default();
        let mut queue = LodAtlasUploadQueue::default();
        queue
            .enqueue_slot(
                atlas,
                AtlasSlot {
                    index: 2,
                    generation: 7,
                },
                16,
            )
            .unwrap();
        queue
            .enqueue_slot(
                atlas,
                AtlasSlot {
                    index: 2,
                    generation: 8,
                },
                16,
            )
            .unwrap();
        assert_eq!(queue.queued_slot_count(), 1);
        assert_eq!(queue.slots.values().next().unwrap().slot.generation, 8);
    }

    #[test]
    fn aggregate_budget_is_nonzero_and_rejects_invalid_updates_without_mutation() {
        assert_eq!(
            LodAtlasUploadBudget::try_new(0, 1),
            Err(LodAtlasUploadBudgetError::ZeroCanonicalByteLimit)
        );
        assert_eq!(
            LodAtlasUploadBudget::try_new(1, 0),
            Err(LodAtlasUploadBudgetError::ZeroSlotLimit)
        );
        let mut budget = LodAtlasUploadBudget::try_new(4096, 4).unwrap();
        assert_eq!(
            budget.set_max_canonical_bytes_per_frame(0),
            Err(LodAtlasUploadBudgetError::ZeroCanonicalByteLimit)
        );
        assert_eq!(budget.max_canonical_bytes_per_frame(), 4096);
        assert_eq!(
            budget.set_max_slots_per_frame(0),
            Err(LodAtlasUploadBudgetError::ZeroSlotLimit)
        );
        assert_eq!(budget.max_slots_per_frame(), 4);
    }

    #[test]
    fn global_planner_is_deterministic_fair_and_preserves_slot_order() {
        let atlases = [atlas_id(1), atlas_id(2), atlas_id(3)];
        let record_bytes = std::mem::size_of::<Gaussian3d>() as u64;
        let budget = LodAtlasUploadBudget::try_new(record_bytes * 2, 2).unwrap();
        let input = vec![
            descriptor(atlases[2], 1, 1),
            descriptor(atlases[0], 1, 1),
            descriptor(atlases[1], 0, 1),
            descriptor(atlases[2], 0, 1),
            descriptor(atlases[0], 0, 1),
            descriptor(atlases[1], 1, 1),
        ];
        let mut scheduler = LodAtlasUploadScheduler::default();
        let first = plan_lod_atlas_uploads(&mut scheduler, input, budget);
        assert_eq!(
            first
                .admitted
                .iter()
                .map(|upload| (upload.atlas, upload.slot.index))
                .collect::<Vec<_>>(),
            vec![(atlases[0], 0), (atlases[1], 0)]
        );
        assert_eq!(first.deferred.len(), 4);
        assert_eq!(first.deferred_canonical_bytes, record_bytes * 4);
        assert_eq!(first.deferred_atlases.len(), 3);

        let second = plan_lod_atlas_uploads(&mut scheduler, first.deferred, budget);
        assert_eq!(
            second
                .admitted
                .iter()
                .map(|upload| (upload.atlas, upload.slot.index))
                .collect::<Vec<_>>(),
            vec![(atlases[2], 0), (atlases[0], 1)],
            "the next frame resumes after the last admitted cloud"
        );
        assert!(
            second
                .deferred
                .iter()
                .filter(|upload| upload.atlas == atlases[2])
                .all(|upload| upload.slot.index > 0),
            "later slots never bypass the per-cloud head"
        );
    }

    #[test]
    fn oversized_atomic_slot_is_deferred_with_typed_status() {
        let atlas = atlas_id(11);
        let record_bytes = std::mem::size_of::<Gaussian3d>() as u64;
        let budget = LodAtlasUploadBudget::try_new(record_bytes, 8).unwrap();
        let mut scheduler = LodAtlasUploadScheduler::default();
        let plan = plan_lod_atlas_uploads(&mut scheduler, [descriptor(atlas, 4, 2)], budget);
        assert!(plan.admitted.is_empty());
        assert_eq!(plan.deferred.len(), 1);
        assert_eq!(plan.deferred_canonical_bytes, record_bytes * 2);
        assert_eq!(plan.oversized_slots, 1);
        assert_eq!(
            plan.first_error,
            Some(LodAtlasUploadBudgetError::SlotExceedsCanonicalByteLimit {
                atlas,
                slot_index: 4,
                required: record_bytes * 2,
                limit: record_bytes,
            })
        );
    }

    #[test]
    fn complete_atlas_queue_is_bounded_and_publishes_no_residency_generation() {
        let atlas = AssetId::<PlanarGaussian3d>::default();
        let mut queue = LodAtlasUploadQueue::default();
        queue.enqueue_complete_atlas(atlas, 32, 8).unwrap();
        let mut slots = queue.queued_slots().collect::<Vec<_>>();
        slots.sort_by_key(|upload| upload.slot.index);
        assert_eq!(slots.len(), 4);
        assert_eq!(
            slots
                .iter()
                .map(|upload| upload.slot.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(slots.iter().all(|upload| upload.slot.generation == 0));

        let mut generations = LodAtlasGpuGenerations::default();
        generations.mark_current(slots[0]);
        assert!(!generations.is_current(
            atlas.untyped(),
            AtlasSlot {
                index: 0,
                generation: 1,
            }
        ));
    }

    #[test]
    fn snapshot_contains_exact_final_slot_planes() {
        let mut assets = Assets::<PlanarGaussian3d>::default();
        let handle = assets.add(PlanarGaussian3d::from(
            (0..12)
                .map(|index| gaussian(index as f32))
                .collect::<Vec<_>>(),
        ));
        let descriptor = LodAtlasSlotUpload {
            atlas: handle.id(),
            slot: AtlasSlot {
                index: 2,
                generation: 3,
            },
            gaussians_per_slot: 4,
        };
        let atlas = assets.get(&handle).unwrap();
        let upload = snapshot_slot(Some(atlas), descriptor).unwrap();
        assert_eq!(upload.planes.len(), 4);
        assert_eq!(
            upload
                .planes
                .position_visibility
                .iter()
                .map(|position| position.position[0])
                .collect::<Vec<_>>(),
            vec![8.0, 9.0, 10.0, 11.0]
        );
        assert_eq!(upload.planes.rotation, atlas.rotation[8..12]);
        assert_eq!(upload.planes.scale_opacity, atlas.scale_opacity[8..12]);
    }

    #[test]
    fn snapshot_rejects_overflow_and_out_of_bounds_without_allocating() {
        let atlas = PlanarGaussian3d::from(vec![Gaussian3d::default(); 4]);
        let mut queue = LodAtlasUploadQueue::default();
        assert_eq!(
            queue.enqueue_slot(
                AssetId::default(),
                AtlasSlot {
                    index: 0,
                    generation: 0,
                },
                4,
            ),
            Err(LodAtlasUploadError::ZeroGeneration)
        );
        let overflow = LodAtlasSlotUpload {
            atlas: AssetId::default(),
            slot: AtlasSlot {
                index: u32::MAX,
                generation: 1,
            },
            gaussians_per_slot: 2,
        };
        assert_eq!(
            snapshot_slot(Some(&atlas), overflow).unwrap_err(),
            LodAtlasUploadError::AddressOverflow
        );

        let outside = LodAtlasSlotUpload {
            atlas: AssetId::default(),
            slot: AtlasSlot {
                index: 1,
                generation: 1,
            },
            gaussians_per_slot: 4,
        };
        assert_eq!(
            snapshot_slot(Some(&atlas), outside).unwrap_err(),
            LodAtlasUploadError::SlotOutOfRange {
                start: 4,
                end: 8,
                atlas_len: 4,
            }
        );
    }

    #[test]
    fn generation_registry_rejects_reused_and_unuploaded_slots() {
        let atlas = AssetId::<PlanarGaussian3d>::default();
        let first = LodAtlasSlotUpload {
            atlas,
            slot: AtlasSlot {
                index: 0,
                generation: 1,
            },
            gaussians_per_slot: 4,
        };
        let mut generations = LodAtlasGpuGenerations::default();
        assert!(!generations.is_current(atlas.untyped(), first.slot));
        let ranges = [LodPhysicalRange {
            node: crate::LodNodeId(1),
            page: crate::LodPageId(1),
            slot: first.slot,
            physical_start: 0,
            count: 4,
        }];
        assert!(!generations.frontier_is_current(atlas.untyped(), &ranges));
        generations.mark_current(first);
        assert!(generations.is_current(atlas.untyped(), first.slot));
        assert!(generations.frontier_is_current(atlas.untyped(), &ranges));
        assert!(!generations.is_current(
            atlas.untyped(),
            AtlasSlot {
                index: 0,
                generation: 2,
            }
        ));
        generations.invalidate(atlas, 0);
        assert!(!generations.is_current(atlas.untyped(), first.slot));
    }

    #[test]
    fn adjacent_slots_coalesce_into_exact_planar_ranges_and_fewer_queue_writes() {
        let atlas = PlanarGaussian3d::from(
            (0..12)
                .map(|index| gaussian(index as f32))
                .collect::<Vec<_>>(),
        );
        let atlas_id = AssetId::<PlanarGaussian3d>::default();
        let uploads = [2_u32, 0, 4, 1]
            .into_iter()
            .map(|index| {
                snapshot_slot(
                    Some(&atlas),
                    LodAtlasSlotUpload {
                        atlas: atlas_id,
                        slot: AtlasSlot {
                            index,
                            generation: index + 1,
                        },
                        gaussians_per_slot: 2,
                    },
                )
                .unwrap()
            })
            .collect();
        let coalesced = coalesce_atlas_uploads(uploads).unwrap();
        assert_eq!(coalesced.len(), 2);
        assert_eq!(coalesced[0].start().unwrap(), 0);
        assert_eq!(coalesced[0].descriptors.len(), 3);
        assert_eq!(coalesced[0].planes.len(), 6);
        assert_eq!(coalesced[1].start().unwrap(), 8);
        assert_eq!(coalesced[1].planes.len(), 2);
        assert_eq!(
            coalesced[0]
                .planes
                .position_visibility
                .iter()
                .map(|position| position.position[0])
                .collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
        assert_eq!(coalesced.len() * 4, 8, "four planar writes per range");
        assert_eq!(4 * 4, 16, "uncoalesced baseline writes");
    }

    /// Opt in with:
    /// `RUN_GPU_LOD_ATLAS_TESTS=1 cargo test --no-default-features --features 'planar buffer_storage lod sh0 sort_std io_flexbuffers' gpu_atlas_copy_matches_cpu_oracle -- --ignored --nocapture`
    /// Add `precompute_covariance_3d` to the feature set to verify the ordered
    /// derived covariance plane in the same upload submission.
    #[test]
    #[ignore = "requires an explicitly requested wgpu adapter"]
    fn gpu_atlas_copy_matches_cpu_oracle() {
        use std::{
            sync::{Arc, mpsc},
            time::Duration,
        };

        #[cfg(feature = "precompute_covariance_3d")]
        use crate::gaussian::f32::Covariance3dOpacity;
        use bevy::render::{
            render_resource::{BufferDescriptor, BufferUsages},
            renderer::{RenderDevice, RenderQueue, WgpuWrapper},
        };

        if std::env::var("RUN_GPU_LOD_ATLAS_TESTS").as_deref() != Ok("1") {
            eprintln!("set RUN_GPU_LOD_ATLAS_TESTS=1 to execute the adapter test");
            return;
        }

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .expect("GPU atlas test requires an adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpu_covariance_dispatch_test"),
            ..Default::default()
        }))
        .expect("GPU atlas test could not create a device");
        let render_device = RenderDevice::from(device);
        let render_queue = RenderQueue(Arc::new(WgpuWrapper::new(queue)));

        let cpu_atlas = PlanarGaussian3d::from(
            (0..4)
                .map(|index| gaussian(index as f32 + 0.25))
                .collect::<Vec<_>>(),
        );
        let atlas_id = AssetId::<PlanarGaussian3d>::default();
        let adjacent = [1_u32, 0]
            .into_iter()
            .map(|index| {
                snapshot_slot(
                    Some(&cpu_atlas),
                    LodAtlasSlotUpload {
                        atlas: atlas_id,
                        slot: AtlasSlot {
                            index,
                            generation: index + 1,
                        },
                        gaussians_per_slot: 2,
                    },
                )
                .unwrap()
            })
            .collect();
        let coalesced = coalesce_atlas_uploads(adjacent).unwrap();
        assert_eq!(
            coalesced.len(),
            1,
            "adjacent slots must form one upload range"
        );
        let upload = &coalesced[0];
        assert_eq!(upload.start().unwrap(), 0);
        assert_eq!(upload.planes.len(), 4);

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let destination = |label, byte_len| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: byte_len,
                usage: storage | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let position_bytes = bytemuck::cast_slice::<_, u8>(&upload.planes.position_visibility);
        let sh_bytes = bytemuck::cast_slice::<_, u8>(&upload.planes.spherical_harmonic);
        let rotation_bytes = bytemuck::cast_slice::<_, u8>(&upload.planes.rotation);
        let scale_bytes = bytemuck::cast_slice::<_, u8>(&upload.planes.scale_opacity);
        let position_visibility = destination(
            "test_atlas_position_visibility",
            position_bytes.len() as u64,
        );
        let spherical_harmonic =
            destination("test_atlas_spherical_harmonic", sh_bytes.len() as u64);
        let rotation = destination("test_atlas_rotation", rotation_bytes.len() as u64);
        let scale_opacity = destination("test_atlas_scale_opacity", scale_bytes.len() as u64);
        #[cfg(feature = "precompute_covariance_3d")]
        let covariance_bytes = upload.planes.len() * std::mem::size_of::<Covariance3dOpacity>();
        #[cfg(feature = "precompute_covariance_3d")]
        let covariance = {
            let descriptor = BufferDescriptor {
                label: Some("test_atlas_covariance"),
                size: covariance_bytes as u64,
                usage: storage | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            };
            render_device.create_buffer(&descriptor)
        };
        let canonical_bytes =
            position_bytes.len() + sh_bytes.len() + rotation_bytes.len() + scale_bytes.len();
        #[cfg(feature = "precompute_covariance_3d")]
        let output_bytes = canonical_bytes + covariance_bytes;
        #[cfg(not(feature = "precompute_covariance_3d"))]
        let output_bytes = canonical_bytes;
        let readback = render_device.create_buffer(&BufferDescriptor {
            label: Some("test_atlas_readback"),
            size: output_bytes as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let error_scope = render_device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        #[cfg(feature = "precompute_covariance_3d")]
        let pipeline = LodCovariancePipeline::new(render_device.wgpu_device());
        let mut upload_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("test_atlas_upload_encoder"),
        });
        let mut staging = Vec::new();
        encode_canonical_buffer_copies(
            &render_device,
            &mut upload_encoder,
            LodAtlasCanonicalBuffers {
                position_visibility: &position_visibility,
                spherical_harmonic: &spherical_harmonic,
                rotation: &rotation,
                scale_opacity: &scale_opacity,
                count: upload.planes.len(),
            },
            upload,
            &mut staging,
        )
        .unwrap();
        #[cfg(feature = "precompute_covariance_3d")]
        assert_eq!(
            pipeline
                .encode_dispatch_buffers(
                    &render_device,
                    &mut upload_encoder,
                    &rotation,
                    &scale_opacity,
                    &covariance,
                    &[LodCovarianceUploadRange {
                        start: 0,
                        count: upload.planes.len() as u32,
                    }],
                )
                .unwrap(),
            1
        );
        assert_eq!(staging.len(), 4, "one staging buffer per canonical plane");
        let mut generations = LodAtlasGpuGenerations::default();
        assert!(
            upload
                .descriptors
                .iter()
                .all(|descriptor| !generations.is_current(atlas_id.untyped(), descriptor.slot))
        );
        render_queue.submit([upload_encoder.finish()]);
        for descriptor in &upload.descriptors {
            generations.mark_current(*descriptor);
        }
        assert!(
            upload
                .descriptors
                .iter()
                .all(|descriptor| generations.is_current(atlas_id.untyped(), descriptor.slot)),
            "generation proofs must publish only after the ordered upload is submitted"
        );

        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("test_covariance_readback_encoder"),
        });
        let mut readback_offset = 0_u64;
        for (buffer, len) in [
            (&position_visibility, position_bytes.len()),
            (&spherical_harmonic, sh_bytes.len()),
            (&rotation, rotation_bytes.len()),
            (&scale_opacity, scale_bytes.len()),
        ] {
            encoder.copy_buffer_to_buffer(buffer, 0, &readback, readback_offset, len as u64);
            readback_offset += len as u64;
        }
        #[cfg(feature = "precompute_covariance_3d")]
        encoder.copy_buffer_to_buffer(
            &covariance,
            0,
            &readback,
            readback_offset,
            covariance_bytes as u64,
        );
        let submission = render_queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (map_sender, map_receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = map_sender.send(result);
        });
        render_device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(30)),
            })
            .expect("GPU atlas test device poll failed");
        map_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("GPU atlas map callback did not run")
            .expect("GPU atlas output failed to map");
        let shader_error = pollster::block_on(error_scope.pop());
        assert!(
            shader_error.is_none(),
            "covariance dispatch validation failed: {shader_error:?}"
        );

        let mapped = slice.get_mapped_range();
        let mut expected_canonical = Vec::with_capacity(canonical_bytes);
        expected_canonical.extend_from_slice(position_bytes);
        expected_canonical.extend_from_slice(sh_bytes);
        expected_canonical.extend_from_slice(rotation_bytes);
        expected_canonical.extend_from_slice(scale_bytes);
        assert_eq!(&mapped[..canonical_bytes], expected_canonical);
        #[cfg(feature = "precompute_covariance_3d")]
        let actual = bytemuck::cast_slice::<u8, Covariance3dOpacity>(
            &mapped[canonical_bytes..canonical_bytes + covariance_bytes],
        );
        #[cfg(feature = "precompute_covariance_3d")]
        for ((actual, rotation), scale_opacity) in actual
            .iter()
            .zip(&upload.planes.rotation)
            .zip(&upload.planes.scale_opacity)
        {
            let expected = Covariance3dOpacity {
                cov3d: crate::gaussian::covariance::compute_covariance_3d(
                    Vec4::from_array(rotation.rotation),
                    Vec3::from_array(scale_opacity.scale),
                ),
                opacity: scale_opacity.opacity,
                pad: 0.0,
            };
            for (actual, expected) in actual.cov3d.iter().zip(expected.cov3d) {
                let tolerance = 2.0e-5 + expected.abs() * 5.0e-6;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{actual} != {expected} (tolerance {tolerance})"
                );
            }
            assert_eq!(actual.opacity, expected.opacity);
            assert_eq!(actual.pad, expected.pad);
        }
        drop(mapped);
        readback.unmap();
    }
}
