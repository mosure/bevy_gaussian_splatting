//! Per-view exact active-count compaction and indirect argument generation.
//!
//! This is the GPU boundary between hierarchy selection and sorting/rendering.
//! Today it can consume the identity source range (the legacy flat cloud) or a
//! [`LodCandidateFrontier`] validated by the bounded streaming runtime. A future
//! GPU hierarchy traversal can write the same bounded candidate buffer without
//! changing the exact-count compaction/sort boundary.

#[cfg(feature = "morph_interpolate")]
use std::any::TypeId;
use std::{
    collections::{HashMap, HashSet},
    fmt,
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

use bevy::{
    asset::{Asset, AssetId, load_internal_asset, uuid_handle},
    core_pipeline::{Core3d, Core3dSystems, prepass::PreviousViewUniformOffset},
    prelude::*,
    render::{
        Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup,
        RenderSystems,
        extract_component::DynamicUniformIndex,
        init_gpu_resource,
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding,
            BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferSize, BufferUsages,
            CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache, ShaderStages,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        sync_world::RenderEntity,
        view::{ExtractedView, RenderVisibleEntities, RetainedViewEntity, ViewUniformOffset},
    },
};
use bevy_interleave::{interface::storage::PlanarStorageBindGroup, prelude::*};
use bytemuck::{Pod, Zeroable};

#[cfg(feature = "morph_interpolate")]
use crate::{gaussian::formats::planar_3d::PlanarGaussian3d, morph::interpolate::InterpolateLabel};

use crate::{
    camera::GaussianCamera,
    gaussian::{
        cloud::CloudVisibilityClass,
        lod_settings::{GaussianLodSettings, LodQualityEndpoint, LodQualityTarget},
        settings::{CloudSettings, GaussianMode, RadixSortDepthBits},
    },
    render::{
        CloudPipeline, CloudPipelineKey, CloudPipelineReady, CloudUniform,
        GaussianComputeViewBindGroup, GaussianUniformBindGroups, ShaderDefines,
        shader_defs_with_defines,
    },
    sort::{
        SortEntry, SortMode,
        radix::{LodRadixBindGroups, RadixSortPipeline},
    },
    stream::{
        atlas_upload::LodAtlasGpuGenerations,
        render_commit::{
            LOD_RENDER_ACTIVE, LOD_RENDER_FAILED, LOD_RENDER_PREPARED, LOD_RENDER_WAITING,
            LodRenderCandidate, LodRenderCandidates,
        },
        runtime::{LodCandidateFrontier, LodPhysicalRange},
    },
};

const LOD_COMPACTION_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("7c3bfe93-e1f3-4cff-ba07-23c745621dac");

pub const LOD_COMPACTION_WORKGROUP_SIZE: u32 = 256;
/// Number of workgroup-count records scanned by one parallel scan workgroup.
pub const LOD_COMPACTION_SCAN_BLOCK_SIZE: u32 = LOD_COMPACTION_WORKGROUP_SIZE;
/// The second scan level is deliberately bounded to one parallel workgroup.
/// This still supports 65,536 candidate workgroups (16,777,216 candidates)
/// without a serial full-frontier scan.
pub const LOD_COMPACTION_MAX_SCAN_BLOCKS: u32 = LOD_COMPACTION_SCAN_BLOCK_SIZE;
pub const LOD_COMPACTION_MAX_CANDIDATE_WORKGROUPS: u32 =
    LOD_COMPACTION_SCAN_BLOCK_SIZE * LOD_COMPACTION_MAX_SCAN_BLOCKS;
pub const DRAW_INDIRECT_OFFSET: u64 = 0;
pub const DISPATCH_A_INDIRECT_OFFSET: u64 = 16;
pub const DISPATCH_C_INDIRECT_OFFSET: u64 = 28;
pub const LOD_INDIRECT_ARGS_SIZE: u64 = 48;
pub const DEFAULT_LOD_COMPACTION_AGGREGATE_BYTES: u64 = 512 * 1024 * 1024;

/// Render-world memory policy shared by all view/cloud compaction states of a
/// planar representation. Setting the limit to zero disables GPU compaction
/// and leaves every pair on the complete legacy path.
#[derive(Resource, Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodCompactionMemoryBudget {
    pub max_total_bytes: u64,
}

impl Default for LodCompactionMemoryBudget {
    fn default() -> Self {
        Self {
            max_total_bytes: DEFAULT_LOD_COMPACTION_AGGREGATE_BYTES,
        }
    }
}

/// Rejection reasons for candidate updates that would make the active frontier
/// incomplete or address outside its resident source allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LodCandidateConfigError {
    UnsupportedSortMode,
    CandidateCountExceedsCapacity {
        candidate_count: u32,
        output_capacity: u32,
    },
    IdentitySourceExceedsCapacity {
        source_count: u32,
        output_capacity: u32,
    },
    CandidateCountMismatch {
        declared: u32,
        actual: u32,
    },
    PhysicalRangeCountNotRepresentable {
        range_count: usize,
    },
    PhysicalRangeDescriptorCapacityExceeded {
        range_count: u32,
        descriptor_capacity: u32,
    },
    PhysicalRangeCountOverflow,
    PhysicalRangeOutOfRange {
        range_index: u32,
        physical_start: u32,
        count: u32,
        source_count: u32,
    },
}

impl fmt::Display for LodCandidateConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSortMode => {
                write!(formatter, "LoD bridge candidates require radix sorting")
            }
            Self::CandidateCountExceedsCapacity {
                candidate_count,
                output_capacity,
            } => write!(
                formatter,
                "candidate count {candidate_count} exceeds active capacity {output_capacity}"
            ),
            Self::IdentitySourceExceedsCapacity {
                source_count,
                output_capacity,
            } => write!(
                formatter,
                "identity source count {source_count} exceeds active capacity {output_capacity}"
            ),
            Self::CandidateCountMismatch { declared, actual } => write!(
                formatter,
                "declared candidate count {declared} does not match validated payload count {actual}"
            ),
            Self::PhysicalRangeCountNotRepresentable { range_count } => write!(
                formatter,
                "physical range count {range_count} is not representable as u32"
            ),
            Self::PhysicalRangeDescriptorCapacityExceeded {
                range_count,
                descriptor_capacity,
            } => write!(
                formatter,
                "physical range count {range_count} exceeds descriptor capacity {descriptor_capacity}"
            ),
            Self::PhysicalRangeCountOverflow => {
                write!(formatter, "physical range candidate count overflowed u32")
            }
            Self::PhysicalRangeOutOfRange {
                range_index,
                physical_start,
                count,
                source_count,
            } => write!(
                formatter,
                "physical range {range_index} [{physical_start}, {physical_start} + {count}) exceeds source count {source_count}"
            ),
        }
    }
}

impl std::error::Error for LodCandidateConfigError {}

fn validate_bridge_candidate_sort_mode(
    sort_mode: &SortMode,
) -> Result<(), LodCandidateConfigError> {
    if *sort_mode == SortMode::Radix {
        Ok(())
    } else {
        Err(LodCandidateConfigError::UnsupportedSortMode)
    }
}

/// Canonical key construction for state that must remain isolated per view,
/// render instance, and cloud asset. Multiple render-world entities may share
/// one cloud asset while carrying different transforms or LoD settings.
pub(crate) fn lod_view_cloud_key<A: Asset>(
    retained_view: RetainedViewEntity,
    entity: Entity,
    cloud: AssetId<A>,
) -> (RetainedViewEntity, Entity, AssetId<A>) {
    (retained_view, entity, cloud)
}

/// Returns the buffer containing the final output of the LSD radix passes.
/// Active entries start in buffer A (index 0), and each 8-bit pass swaps A/B.
pub const fn radix_sorted_output_buffer_index(radix_depth_bits: RadixSortDepthBits) -> usize {
    ((radix_depth_bits.bits() / 8) % 2) as usize
}

fn representable_source_count(source_len: usize) -> Option<u32> {
    u32::try_from(source_len).ok()
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub struct LodCompactionLabel;

#[derive(Default)]
struct LodCompactionPluginFlag;

impl Plugin for LodCompactionPluginFlag {
    fn build(&self, _app: &mut App) {}
}

/// Installs one compactor specialization for a planar Gaussian representation.
#[derive(Default)]
pub struct LodCompactionPlugin<R: PlanarSync> {
    marker: PhantomData<R>,
}

impl<R: PlanarSync> Plugin for LodCompactionPlugin<R>
where
    R::GpuPlanarType: GpuPlanarStorage,
{
    fn build(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_gpu_resource::<LodCompactionBuffers<R>>()
                .init_resource::<LodCompactionMemoryBudget>()
                // Keep the render/compaction plugin independently usable when
                // applications opt out of the automatic streaming bridge.
                .init_gpu_resource::<LodAtlasGpuGenerations>()
                .add_systems(ExtractSchedule, extract_lod_settings::<R>)
                .add_systems(
                    Render,
                    (
                        prepare_lod_compaction_buffers::<R>,
                        commit_lod_bridge_candidates::<R>
                            .after(prepare_lod_compaction_buffers::<R>),
                    )
                        .in_set(RenderSystems::PrepareResources),
                );

            #[cfg(feature = "morph_interpolate")]
            if TypeId::of::<R::PlanarType>() == TypeId::of::<PlanarGaussian3d>() {
                render_app.add_systems(
                    Core3d,
                    run_lod_compaction::<R>
                        .in_set(LodCompactionLabel)
                        .after(InterpolateLabel)
                        .before(Core3dSystems::Prepass),
                );
            } else {
                render_app.add_systems(
                    Core3d,
                    run_lod_compaction::<R>
                        .in_set(LodCompactionLabel)
                        .before(Core3dSystems::Prepass),
                );
            }

            #[cfg(not(feature = "morph_interpolate"))]
            render_app.add_systems(
                Core3d,
                run_lod_compaction::<R>
                    .in_set(LodCompactionLabel)
                    .before(Core3dSystems::Prepass),
            );
        }

        if app.is_plugin_added::<LodCompactionPluginFlag>() {
            return;
        }
        app.add_plugins(LodCompactionPluginFlag);
        load_internal_asset!(
            app,
            LOD_COMPACTION_SHADER_HANDLE,
            "lod_compaction.wgsl",
            Shader::from_wgsl
        );
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                RenderStartup,
                init_gpu_resource::<LodCompactionPipeline<R>>
                    .after(CloudPipelineReady)
                    .ambiguous_with_all(),
            );
        }
    }
}

#[allow(clippy::type_complexity)]
fn extract_lod_settings<R: PlanarSync>(
    mut commands: Commands,
    settings: Extract<
        Query<
            (
                RenderEntity,
                &ViewVisibility,
                Option<&GaussianLodSettings>,
                Option<&LodRenderCandidates>,
            ),
            With<R::PlanarTypeHandle>,
        >,
    >,
) {
    for (render_entity, visibility, lod_settings, bridge_candidates) in &settings {
        let mut entity = commands.entity(render_entity);
        match lod_settings.filter(|_| visibility.get()) {
            Some(settings) => {
                entity.insert(settings.clone());
            }
            None => {
                entity.remove::<GaussianLodSettings>();
            }
        }
        match bridge_candidates.filter(|_| visibility.get()) {
            Some(candidates) => {
                entity.insert(candidates.clone());
            }
            None => {
                entity.remove::<LodRenderCandidates>();
            }
        }
    }
}

/// Uniform shared by reset, filter, and finalize passes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LodCompactionUniform {
    pub source_count: u32,
    pub candidate_count: u32,
    pub output_capacity: u32,
    /// 0 = identity, 1 = explicit words, 2 = physical range descriptors.
    pub candidate_source_mode: u32,
    pub consumer_entries_a: u32,
    pub consumer_entries_c: u32,
    pub quality_endpoint: u32,
    pub frustum_culling: u32,
    pub frustum_margin: f32,
    pub candidate_range_count: u32,
    pub transform_scale_bound: f32,
    /// Word offset of the cached evaluation region in binding 1. This equals
    /// the actual range-descriptor prefix allocation.
    pub candidate_source_word_capacity: u32,
    pub _padding: [u32; 4],
}

const LOD_CANDIDATE_SOURCE_IDENTITY: u32 = 0;
const LOD_CANDIDATE_SOURCE_RANGES: u32 = 2;
const LOD_MIN_CANDIDATE_SOURCE_WORDS: u32 = 4;

/// Four-word GPU descriptor for one contiguous physical atlas range. The
/// cumulative candidate start permits logarithmic lookup without materializing
/// one index per Gaussian.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct LodGpuPhysicalRangeDescriptor {
    pub candidate_start: u32,
    pub physical_start: u32,
    pub count: u32,
    pub _padding: u32,
}

impl LodCompactionUniform {
    fn identity(
        source_count: u32,
        output_capacity: u32,
        endpoint: LodQualityEndpoint,
        frustum_culling: bool,
    ) -> Result<Self, LodCandidateConfigError> {
        if source_count > output_capacity {
            return Err(LodCandidateConfigError::IdentitySourceExceedsCapacity {
                source_count,
                output_capacity,
            });
        }
        Ok(Self {
            source_count,
            candidate_count: source_count,
            output_capacity,
            candidate_source_mode: LOD_CANDIDATE_SOURCE_IDENTITY,
            consumer_entries_a: LOD_COMPACTION_WORKGROUP_SIZE,
            consumer_entries_c: LOD_COMPACTION_WORKGROUP_SIZE,
            quality_endpoint: quality_endpoint_code(endpoint),
            frustum_culling: u32::from(frustum_culling),
            frustum_margin: 0.0,
            candidate_range_count: 0,
            transform_scale_bound: 1.0,
            candidate_source_word_capacity: LOD_MIN_CANDIDATE_SOURCE_WORDS,
            _padding: [0; 4],
        })
    }

    fn with_physical_ranges(
        mut self,
        candidate_count: u32,
        range_count: u32,
    ) -> Result<Self, LodCandidateConfigError> {
        if candidate_count > self.output_capacity {
            return Err(LodCandidateConfigError::CandidateCountExceedsCapacity {
                candidate_count,
                output_capacity: self.output_capacity,
            });
        }
        let descriptor_capacity = self.output_capacity
            / (std::mem::size_of::<LodGpuPhysicalRangeDescriptor>() as u32
                / std::mem::size_of::<u32>() as u32);
        if range_count > descriptor_capacity {
            return Err(
                LodCandidateConfigError::PhysicalRangeDescriptorCapacityExceeded {
                    range_count,
                    descriptor_capacity,
                },
            );
        }
        self.candidate_count = candidate_count;
        self.candidate_source_mode = LOD_CANDIDATE_SOURCE_RANGES;
        self.candidate_range_count = range_count;
        Ok(self)
    }

    fn initial(
        source_count: u32,
        output_capacity: u32,
        endpoint: LodQualityEndpoint,
        frustum_culling: bool,
    ) -> (Self, LodCompactionReadiness) {
        if source_count <= output_capacity {
            return (
                Self::identity(source_count, output_capacity, endpoint, frustum_culling)
                    .expect("source-sized identity allocation"),
                LodCompactionReadiness::PendingCandidates,
            );
        }

        (
            Self {
                source_count,
                candidate_count: 0,
                output_capacity,
                candidate_source_mode: LOD_CANDIDATE_SOURCE_RANGES,
                consumer_entries_a: LOD_COMPACTION_WORKGROUP_SIZE,
                consumer_entries_c: LOD_COMPACTION_WORKGROUP_SIZE,
                quality_endpoint: quality_endpoint_code(endpoint),
                frustum_culling: u32::from(frustum_culling),
                frustum_margin: 0.0,
                candidate_range_count: 0,
                transform_scale_bound: 1.0,
                candidate_source_word_capacity: LOD_MIN_CANDIDATE_SOURCE_WORDS,
                _padding: [0; 4],
            },
            LodCompactionReadiness::AwaitingCandidates,
        )
    }

    fn with_policy(mut self, settings: &GaussianLodSettings) -> Self {
        self.set_policy_fields(settings);
        self
    }

    fn set_policy_fields(&mut self, settings: &GaussianLodSettings) {
        self.quality_endpoint = quality_endpoint_code(settings.quality_endpoint());
        self.frustum_culling = u32::from(settings.frustum_culling);
        self.frustum_margin = finite_non_negative_or_zero(settings.frustum_margin);
    }
}

fn finite_non_negative_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Byte-compatible with a draw indirect record at offset 0 and dispatch
/// indirect records at offsets 16 (pass A) and 28 (pass C). The final two
/// words are GPU diagnostics.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct LodIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
    pub dispatch_x: u32,
    pub dispatch_y: u32,
    pub dispatch_z: u32,
    pub dispatch_c_x: u32,
    pub dispatch_c_y: u32,
    pub dispatch_c_z: u32,
    pub candidate_hits: u32,
    pub overflow_count: u32,
}

/// Failures from the bounded, opt-in GPU indirect-argument probe used by
/// headless render tests.
#[cfg(feature = "testing")]
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LodIndirectArgsReadbackError {
    StateNotReady,
    UnsupportedPlatform,
    DevicePoll(String),
    BufferMap(String),
    MappingChannelClosed,
    InvalidByteLength { expected: usize, actual: usize },
}

#[cfg(feature = "testing")]
impl fmt::Display for LodIndirectArgsReadbackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[cfg(feature = "testing")]
impl std::error::Error for LodIndirectArgsReadbackError {}

/// Count plus exclusive prefix used by the stable two-level GPU scan.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
struct LodScanRecord {
    count: u32,
    offset: u32,
}

/// CPU oracle for the shader's finalize pass.
pub fn finalized_indirect_args(
    candidate_hits: u32,
    output_capacity: u32,
    consumer_entries_a: u32,
    consumer_entries_c: u32,
) -> LodIndirectArgs {
    let instance_count = candidate_hits.min(output_capacity);
    let entries_a = consumer_entries_a.max(1);
    let entries_c = consumer_entries_c.max(1);
    LodIndirectArgs {
        vertex_count: 4,
        instance_count,
        first_vertex: 0,
        first_instance: 0,
        dispatch_x: instance_count.div_ceil(entries_a),
        dispatch_y: 1,
        dispatch_z: 1,
        dispatch_c_x: 1,
        dispatch_c_y: instance_count.div_ceil(entries_c),
        dispatch_c_z: 1,
        candidate_hits,
        overflow_count: candidate_hits.saturating_sub(output_capacity),
    }
}

/// Whether a per-view state may replace the complete legacy draw path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodCompactionReadiness {
    /// Buffers exist, but no complete bounded frontier has been committed.
    AwaitingCandidates,
    /// A complete identity/frontier is staged until a prepare-resources phase
    /// observes compiled compaction pipelines, before radix bind groups queue.
    PendingCandidates,
    /// Identity or candidate-list configuration is complete and may be drawn.
    Ready,
}

impl LodCompactionReadiness {
    fn after_commit(self) -> Self {
        match self {
            Self::Ready => Self::Ready,
            Self::AwaitingCandidates | Self::PendingCandidates => Self::PendingCandidates,
        }
    }

    fn after_prepare(self) -> Self {
        match self {
            Self::PendingCandidates => Self::Ready,
            state => state,
        }
    }

    fn synchronize_pipeline_readiness(self, pipelines_ready: bool) -> Self {
        if pipelines_ready {
            self.after_prepare()
        } else if self == Self::Ready {
            Self::PendingCandidates
        } else {
            self
        }
    }
}

/// Stable content identity for one complete render candidate. Both hashes are
/// deterministic and cover the view, physical ranges (including allocator
/// generations), and every explicit candidate index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LodCandidateFrontierFingerprint {
    primary: u64,
    secondary: u64,
    range_count: u32,
    candidate_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LodCandidateUploadPlan {
    ReuseVersion,
    ReuseFingerprint(LodCandidateFrontierFingerprint),
    Upload(LodCandidateFrontierFingerprint),
}

#[derive(Default)]
struct LodCandidateUploadTracker {
    /// The phase allocation is also the immutable candidate's cross-world
    /// version token. Keeping an Arc prevents address reuse while cached.
    version: Option<Arc<AtomicU8>>,
    fingerprint: Option<LodCandidateFrontierFingerprint>,
}

impl LodCandidateUploadTracker {
    fn plan(&self, candidate: &LodRenderCandidate) -> LodCandidateUploadPlan {
        let fingerprint = lod_bridge_candidate_fingerprint(candidate);
        if self.version.as_ref().is_some_and(|version| {
            Arc::ptr_eq(version, &candidate.phase) && self.fingerprint == Some(fingerprint)
        }) {
            return LodCandidateUploadPlan::ReuseVersion;
        }
        self.plan_fingerprint(&candidate.phase, fingerprint)
    }

    fn plan_fingerprint(
        &self,
        version: &Arc<AtomicU8>,
        fingerprint: LodCandidateFrontierFingerprint,
    ) -> LodCandidateUploadPlan {
        if self.version.as_ref().is_some_and(|current| {
            Arc::ptr_eq(current, version) && self.fingerprint == Some(fingerprint)
        }) {
            LodCandidateUploadPlan::ReuseVersion
        } else if self.fingerprint == Some(fingerprint) {
            LodCandidateUploadPlan::ReuseFingerprint(fingerprint)
        } else {
            LodCandidateUploadPlan::Upload(fingerprint)
        }
    }

    fn mark_synchronized(
        &mut self,
        version: &Arc<AtomicU8>,
        fingerprint: LodCandidateFrontierFingerprint,
    ) {
        self.version = Some(Arc::clone(version));
        self.fingerprint = Some(fingerprint);
    }

    fn mark_unversioned(&mut self, fingerprint: LodCandidateFrontierFingerprint) {
        self.version = None;
        self.fingerprint = Some(fingerprint);
    }

    #[cfg(feature = "testing")]
    fn revoke_for_testing_override(&mut self) {
        *self = Self::default();
    }
}

fn lod_candidate_frontier_fingerprint(
    frontier: &LodCandidateFrontier,
) -> LodCandidateFrontierFingerprint {
    lod_candidate_parts_fingerprint(
        frontier.view().0,
        frontier.physical_ranges(),
        frontier.candidate_count(),
    )
}

fn lod_bridge_candidate_fingerprint(
    candidate: &LodRenderCandidate,
) -> LodCandidateFrontierFingerprint {
    lod_candidate_parts_fingerprint(
        candidate.frontier().view().0,
        candidate.render_ranges(),
        candidate.rendered_candidate_count(),
    )
}

fn lod_candidate_parts_fingerprint(
    view: u64,
    ranges: &[LodPhysicalRange],
    candidate_count: u32,
) -> LodCandidateFrontierFingerprint {
    // Two independent fixed-width mixers make accidental equality negligible
    // without retaining another source-sized candidate Vec per camera.
    let mut primary = 0xcbf2_9ce4_8422_2325_u64;
    let mut secondary = 0x6eed_0e9d_a4d9_4a4f_u64;
    let mut write = |value: u64| {
        for byte in value.to_le_bytes() {
            primary ^= u64::from(byte);
            primary = primary.wrapping_mul(0x0000_0100_0000_01b3);
            secondary ^= u64::from(byte).wrapping_add(0x9e37_79b9_7f4a_7c15);
            secondary = secondary
                .rotate_left(27)
                .wrapping_mul(0x3c79_ac49_2ba7_b653)
                .wrapping_add(0x1c69_b3f7_4ac4_ae35);
        }
    };
    write(view);
    write(ranges.len() as u64);
    for range in ranges {
        write(range.node.0);
        write(range.page.0);
        write(u64::from(range.slot.index));
        write(u64::from(range.slot.generation));
        write(u64::from(range.physical_start));
        write(u64::from(range.count));
    }
    write(u64::from(candidate_count));
    LodCandidateFrontierFingerprint {
        primary,
        secondary,
        range_count: ranges.len().try_into().unwrap_or(u32::MAX),
        candidate_count,
    }
}

fn build_gpu_physical_range_descriptors(
    ranges: &[LodPhysicalRange],
    source_count: u32,
) -> Result<(Vec<LodGpuPhysicalRangeDescriptor>, u32), LodCandidateConfigError> {
    let range_count = u32::try_from(ranges.len()).map_err(|_| {
        LodCandidateConfigError::PhysicalRangeCountNotRepresentable {
            range_count: ranges.len(),
        }
    })?;
    let mut descriptors = Vec::with_capacity(range_count as usize);
    let mut candidate_start = 0u32;
    for (range_index, range) in ranges.iter().enumerate() {
        let end = range.physical_start.checked_add(range.count).ok_or(
            LodCandidateConfigError::PhysicalRangeOutOfRange {
                range_index: range_index as u32,
                physical_start: range.physical_start,
                count: range.count,
                source_count,
            },
        )?;
        if end > source_count {
            return Err(LodCandidateConfigError::PhysicalRangeOutOfRange {
                range_index: range_index as u32,
                physical_start: range.physical_start,
                count: range.count,
                source_count,
            });
        }
        if range.count == 0 {
            continue;
        }
        descriptors.push(LodGpuPhysicalRangeDescriptor {
            candidate_start,
            physical_start: range.physical_start,
            count: range.count,
            _padding: 0,
        });
        candidate_start = candidate_start
            .checked_add(range.count)
            .ok_or(LodCandidateConfigError::PhysicalRangeCountOverflow)?;
    }
    Ok((descriptors, candidate_start))
}

const LOD_SORTING_PASS_UNIFORM_SIZE: u64 = std::mem::size_of::<u32>() as u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodCompactionBufferRole {
    Aggregate,
    Config,
    CandidateIndices,
    CandidateEvaluations,
    ScanRecords,
    CandidateAndScanRecords,
    ActiveEntries,
    RadixScratch,
    SortingGlobal,
    SortingStatusCounters,
    SortingPass,
    IndirectArgs,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LodCompactionAllocationError {
    ZeroRequestedCapacity,
    ZeroComputeDispatchCapacity,
    SizeOverflow(LodCompactionBufferRole),
    BufferSizeLimit {
        buffer: LodCompactionBufferRole,
        required: u64,
        limit: u64,
    },
    StorageBindingSizeLimit {
        buffer: LodCompactionBufferRole,
        required: u64,
        limit: u64,
    },
    UniformBindingSizeLimit {
        buffer: LodCompactionBufferRole,
        required: u64,
        limit: u64,
    },
    NoUsableRecordCapacity {
        requested: u32,
        max_buffer_size: u64,
        max_storage_buffer_binding_size: u64,
    },
}

impl fmt::Display for LodCompactionAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroRequestedCapacity => {
                formatter.write_str("LoD compaction requested zero output capacity")
            }
            Self::ZeroComputeDispatchCapacity => formatter.write_str(
                "LoD compaction device limit permits zero compute workgroups per dimension",
            ),
            Self::SizeOverflow(buffer) => {
                write!(formatter, "LoD compaction {buffer:?} byte size overflowed")
            }
            Self::BufferSizeLimit {
                buffer,
                required,
                limit,
            } => write!(
                formatter,
                "LoD compaction {buffer:?} requires {required} bytes, exceeding max_buffer_size {limit}"
            ),
            Self::StorageBindingSizeLimit {
                buffer,
                required,
                limit,
            } => write!(
                formatter,
                "LoD compaction {buffer:?} requires {required} bytes, exceeding max_storage_buffer_binding_size {limit}"
            ),
            Self::UniformBindingSizeLimit {
                buffer,
                required,
                limit,
            } => write!(
                formatter,
                "LoD compaction {buffer:?} requires {required} bytes, exceeding max_uniform_buffer_binding_size {limit}"
            ),
            Self::NoUsableRecordCapacity {
                requested,
                max_buffer_size,
                max_storage_buffer_binding_size,
            } => write!(
                formatter,
                "LoD compaction requested {requested} records but device limits max_buffer_size={max_buffer_size} and max_storage_buffer_binding_size={max_storage_buffer_binding_size} cannot hold one complete record set"
            ),
        }
    }
}

impl std::error::Error for LodCompactionAllocationError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LodCompactionAllocationPlan {
    effective_capacity: u32,
    total_bytes: u64,
    config_bytes: u64,
    candidate_indices_bytes: u64,
    candidate_evaluations_bytes: u64,
    scan_records_bytes: u64,
    candidate_evaluations_and_scan_records_bytes: u64,
    candidate_and_scan_records_bytes: u64,
    /// Admission reserve for the initial minimum-prefix binding while its one
    /// grow-to-maximum replacement is allocated. Prefix capacity never shrinks
    /// during the state's lifetime, so no later retired generations exist.
    candidate_replacement_reserve_bytes: u64,
    scan_group_count: u32,
    scan_block_count: u32,
    active_entries_bytes: u64,
    radix_scratch_bytes: u64,
    sorting_global_bytes: u64,
    sorting_status_counter_bytes: u64,
    sorting_pass_bytes: u64,
    indirect_args_bytes: u64,
}

fn checked_lod_compaction_total_bytes(
    buffers: impl IntoIterator<Item = u64>,
) -> Result<u64, LodCompactionAllocationError> {
    buffers.into_iter().try_fold(0u64, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or(LodCompactionAllocationError::SizeOverflow(
                LodCompactionBufferRole::Aggregate,
            ))
    })
}

/// Aggregate budget after applying both the configured ceiling and a
/// conservative device-derived ceiling of two maximum-size buffers.
pub fn effective_lod_compaction_aggregate_budget(
    configured_max_total_bytes: u64,
    max_buffer_size: u64,
) -> u64 {
    configured_max_total_bytes.min(max_buffer_size.saturating_mul(2))
}

fn reserve_lod_compaction_bytes(used: &mut u64, requested: u64, limit: u64) -> bool {
    let Some(next) = used.checked_add(requested) else {
        return false;
    };
    if next > limit {
        return false;
    }
    *used = next;
    true
}

fn checked_record_buffer_bytes(
    buffer: LodCompactionBufferRole,
    capacity: u32,
    stride: u64,
) -> Result<u64, LodCompactionAllocationError> {
    u64::from(capacity)
        .checked_mul(stride)
        .ok_or(LodCompactionAllocationError::SizeOverflow(buffer))
}

fn validate_buffer_size(
    buffer: LodCompactionBufferRole,
    required: u64,
    max_buffer_size: u64,
) -> Result<(), LodCompactionAllocationError> {
    if required > max_buffer_size {
        Err(LodCompactionAllocationError::BufferSizeLimit {
            buffer,
            required,
            limit: max_buffer_size,
        })
    } else {
        Ok(())
    }
}

fn validate_storage_buffer_size(
    buffer: LodCompactionBufferRole,
    required: u64,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
) -> Result<(), LodCompactionAllocationError> {
    validate_buffer_size(buffer, required, max_buffer_size)?;
    if required > max_storage_buffer_binding_size {
        Err(LodCompactionAllocationError::StorageBindingSizeLimit {
            buffer,
            required,
            limit: max_storage_buffer_binding_size,
        })
    } else {
        Ok(())
    }
}

fn validate_uniform_buffer_size(
    buffer: LodCompactionBufferRole,
    required: u64,
    max_buffer_size: u64,
    max_uniform_buffer_binding_size: u64,
) -> Result<(), LodCompactionAllocationError> {
    validate_buffer_size(buffer, required, max_buffer_size)?;
    if required > max_uniform_buffer_binding_size {
        Err(LodCompactionAllocationError::UniformBindingSizeLimit {
            buffer,
            required,
            limit: max_uniform_buffer_binding_size,
        })
    } else {
        Ok(())
    }
}

fn candidate_evaluations_and_scan_record_bytes(candidate_capacity: u64) -> Option<u64> {
    let group_count = candidate_capacity.div_ceil(u64::from(LOD_COMPACTION_WORKGROUP_SIZE));
    let block_count = group_count.div_ceil(u64::from(LOD_COMPACTION_SCAN_BLOCK_SIZE));
    candidate_capacity
        .checked_mul(std::mem::size_of::<SortEntry>() as u64)?
        .checked_add(
            group_count
                .checked_add(block_count)?
                .checked_mul(std::mem::size_of::<LodScanRecord>() as u64)?,
        )
}

fn candidate_binding_bytes(candidate_capacity: u64, source_word_capacity: u64) -> Option<u64> {
    source_word_capacity
        .checked_mul(std::mem::size_of::<u32>() as u64)?
        .checked_add(candidate_evaluations_and_scan_record_bytes(
            candidate_capacity,
        )?)
}

/// Candidate prefixes are grow-only for a state's lifetime. The initial
/// four-word allocation covers a single physical range; the first larger
/// payload grows directly to the validated maximum so later range-descriptor
/// churn only rewrites bytes and cannot accumulate retired full-tail buffers.
fn candidate_source_capacity_after_upload(
    current_words: u32,
    required_words: u32,
    maximum_words: u32,
) -> u32 {
    let required_words = required_words.max(LOD_MIN_CANDIDATE_SOURCE_WORDS);
    if required_words <= current_words {
        current_words
    } else {
        maximum_words
            .max(required_words)
            .max(LOD_MIN_CANDIDATE_SOURCE_WORDS)
    }
}

fn candidate_and_scan_record_bytes(candidate_capacity: u64) -> Option<u64> {
    candidate_binding_bytes(candidate_capacity, candidate_capacity)
}

/// Largest prefix-plus-scan allocation that fits one storage binding. The
/// binary search applies the actual ceil-divided scan topology rather than
/// accepting independently fitting regions whose sum might exceed the limit.
fn max_candidate_capacity_for_combined_storage(storage_buffer_limit: u64) -> u64 {
    let topology_capacity = u64::from(LOD_COMPACTION_MAX_CANDIDATE_WORKGROUPS)
        * u64::from(LOD_COMPACTION_WORKGROUP_SIZE);
    let mut low = 0u64;
    let mut high = topology_capacity + 1;
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if candidate_and_scan_record_bytes(middle)
            .is_some_and(|required| required <= storage_buffer_limit)
        {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

fn plan_lod_compaction_allocation(
    requested_capacity: u32,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
    max_uniform_buffer_binding_size: u64,
    max_compute_workgroups_per_dimension: u32,
) -> Result<LodCompactionAllocationPlan, LodCompactionAllocationError> {
    if requested_capacity == 0 {
        return Err(LodCompactionAllocationError::ZeroRequestedCapacity);
    }
    if max_compute_workgroups_per_dimension == 0 {
        return Err(LodCompactionAllocationError::ZeroComputeDispatchCapacity);
    }

    let shader_defines = ShaderDefines::default();
    let config_bytes = std::mem::size_of::<LodCompactionUniform>() as u64;
    let indirect_args_bytes = std::mem::size_of::<LodIndirectArgs>() as u64;
    let sorting_global_bytes = u64::from(shader_defines.sorting_buffer_size);
    validate_uniform_buffer_size(
        LodCompactionBufferRole::Config,
        config_bytes,
        max_buffer_size,
        max_uniform_buffer_binding_size,
    )?;
    validate_uniform_buffer_size(
        LodCompactionBufferRole::SortingPass,
        LOD_SORTING_PASS_UNIFORM_SIZE,
        max_buffer_size,
        max_uniform_buffer_binding_size,
    )?;
    validate_storage_buffer_size(
        LodCompactionBufferRole::IndirectArgs,
        indirect_args_bytes,
        max_buffer_size,
        max_storage_buffer_binding_size,
    )?;
    validate_storage_buffer_size(
        LodCompactionBufferRole::SortingGlobal,
        sorting_global_bytes,
        max_buffer_size,
        max_storage_buffer_binding_size,
    )?;

    let storage_buffer_limit = max_buffer_size.min(max_storage_buffer_binding_size);
    let candidate_stride = std::mem::size_of::<u32>() as u64;
    let sort_entry_stride = std::mem::size_of::<SortEntry>() as u64;
    let status_bytes_per_tile = u64::from(shader_defines.radix_base)
        .checked_mul(std::mem::size_of::<u32>() as u64)
        .ok_or(LodCompactionAllocationError::SizeOverflow(
            LodCompactionBufferRole::SortingStatusCounters,
        ))?;
    let status_tile_capacity = storage_buffer_limit / status_bytes_per_tile;
    let status_record_capacity = status_tile_capacity
        .checked_mul(u64::from(shader_defines.workgroup_entries_c))
        .ok_or(LodCompactionAllocationError::SizeOverflow(
            LodCompactionBufferRole::SortingStatusCounters,
        ))?;
    let dispatch_record_capacity = u64::from(max_compute_workgroups_per_dimension)
        .checked_mul(u64::from(LOD_COMPACTION_WORKGROUP_SIZE))
        .ok_or(LodCompactionAllocationError::SizeOverflow(
            LodCompactionBufferRole::CandidateIndices,
        ))?;
    let combined_storage_capacity =
        max_candidate_capacity_for_combined_storage(storage_buffer_limit);
    let effective_capacity = u64::from(requested_capacity)
        .min(storage_buffer_limit / candidate_stride)
        .min(storage_buffer_limit / sort_entry_stride)
        .min(status_record_capacity)
        .min(dispatch_record_capacity)
        .min(combined_storage_capacity)
        .min(u64::from(u32::MAX));
    let effective_capacity = u32::try_from(effective_capacity).map_err(|_| {
        LodCompactionAllocationError::SizeOverflow(LodCompactionBufferRole::ActiveEntries)
    })?;
    if effective_capacity == 0 {
        return Err(LodCompactionAllocationError::NoUsableRecordCapacity {
            requested: requested_capacity,
            max_buffer_size,
            max_storage_buffer_binding_size,
        });
    }

    let candidate_indices_bytes = checked_record_buffer_bytes(
        LodCompactionBufferRole::CandidateIndices,
        effective_capacity,
        candidate_stride,
    )?;
    let candidate_evaluations_bytes = checked_record_buffer_bytes(
        LodCompactionBufferRole::CandidateEvaluations,
        effective_capacity,
        sort_entry_stride,
    )?;
    let scan_group_count = effective_capacity.div_ceil(LOD_COMPACTION_WORKGROUP_SIZE);
    let scan_block_count = scan_group_count.div_ceil(LOD_COMPACTION_SCAN_BLOCK_SIZE);
    debug_assert!(scan_block_count <= LOD_COMPACTION_MAX_SCAN_BLOCKS);
    let scan_record_count = scan_group_count.checked_add(scan_block_count).ok_or(
        LodCompactionAllocationError::SizeOverflow(LodCompactionBufferRole::ScanRecords),
    )?;
    let scan_records_bytes = checked_record_buffer_bytes(
        LodCompactionBufferRole::ScanRecords,
        scan_record_count,
        std::mem::size_of::<LodScanRecord>() as u64,
    )?;
    let candidate_evaluations_and_scan_records_bytes = candidate_evaluations_bytes
        .checked_add(scan_records_bytes)
        .ok_or(LodCompactionAllocationError::SizeOverflow(
            LodCompactionBufferRole::CandidateAndScanRecords,
        ))?;
    let candidate_and_scan_records_bytes = candidate_indices_bytes
        .checked_add(candidate_evaluations_and_scan_records_bytes)
        .ok_or(LodCompactionAllocationError::SizeOverflow(
            LodCompactionBufferRole::CandidateAndScanRecords,
        ))?;
    let active_entries_bytes = checked_record_buffer_bytes(
        LodCompactionBufferRole::ActiveEntries,
        effective_capacity,
        sort_entry_stride,
    )?;
    let radix_scratch_bytes = checked_record_buffer_bytes(
        LodCompactionBufferRole::RadixScratch,
        effective_capacity,
        sort_entry_stride,
    )?;
    let status_tile_count = effective_capacity.div_ceil(shader_defines.workgroup_entries_c);
    let sorting_status_counter_bytes = u64::from(status_tile_count)
        .checked_mul(status_bytes_per_tile)
        .ok_or(LodCompactionAllocationError::SizeOverflow(
            LodCompactionBufferRole::SortingStatusCounters,
        ))?;

    for (buffer, required) in [
        (
            LodCompactionBufferRole::CandidateAndScanRecords,
            candidate_and_scan_records_bytes,
        ),
        (LodCompactionBufferRole::ActiveEntries, active_entries_bytes),
        (LodCompactionBufferRole::RadixScratch, radix_scratch_bytes),
        (
            LodCompactionBufferRole::SortingStatusCounters,
            sorting_status_counter_bytes,
        ),
    ] {
        validate_storage_buffer_size(
            buffer,
            required,
            max_buffer_size,
            max_storage_buffer_binding_size,
        )?;
    }

    let sorting_pass_total_bytes = LOD_SORTING_PASS_UNIFORM_SIZE.checked_mul(4).ok_or(
        LodCompactionAllocationError::SizeOverflow(LodCompactionBufferRole::Aggregate),
    )?;
    // The first payload larger than the four-word minimum replaces the
    // combined candidate/evaluation binding directly at its maximum size.
    // wgpu may retain that initial binding until submitted work retires. Prefix
    // capacity is grow-only after that one replacement, so charging the exact
    // initial binding keeps aggregate admission a hard peak bound without a
    // recurring two-full-buffer penalty.
    let candidate_replacement_reserve_bytes = candidate_binding_bytes(
        u64::from(effective_capacity),
        u64::from(LOD_MIN_CANDIDATE_SOURCE_WORDS),
    )
    .ok_or(LodCompactionAllocationError::SizeOverflow(
        LodCompactionBufferRole::CandidateAndScanRecords,
    ))?;
    let total_bytes = checked_lod_compaction_total_bytes([
        config_bytes,
        candidate_and_scan_records_bytes,
        candidate_replacement_reserve_bytes,
        active_entries_bytes,
        radix_scratch_bytes,
        sorting_global_bytes,
        sorting_status_counter_bytes,
        sorting_pass_total_bytes,
        indirect_args_bytes,
    ])?;

    Ok(LodCompactionAllocationPlan {
        effective_capacity,
        total_bytes,
        config_bytes,
        candidate_indices_bytes,
        candidate_evaluations_bytes,
        scan_records_bytes,
        candidate_evaluations_and_scan_records_bytes,
        candidate_and_scan_records_bytes,
        candidate_replacement_reserve_bytes,
        scan_group_count,
        scan_block_count,
        active_entries_bytes,
        radix_scratch_bytes,
        sorting_global_bytes,
        sorting_status_counter_bytes,
        sorting_pass_bytes: LOD_SORTING_PASS_UNIFORM_SIZE,
        indirect_args_bytes,
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum LodCandidateOwnership {
    /// Production candidates are fail-closed when the extracted bridge payload
    /// disappears, even if a prior frontier happened to have the same shape.
    #[default]
    Bridge,
    /// The scale harness deliberately has no main-world streaming bridge. Its
    /// validated range upload remains authoritative until a real bridge payload
    /// or an explicit invalidation takes ownership again.
    #[cfg(feature = "testing")]
    TestingPhysicalRanges,
}

impl LodCandidateOwnership {
    const fn preserves_missing_bridge_candidate(self) -> bool {
        match self {
            Self::Bridge => false,
            #[cfg(feature = "testing")]
            Self::TestingPhysicalRanges => true,
        }
    }
}

const fn readiness_without_bridge_candidate(
    readiness: LodCompactionReadiness,
    ownership: LodCandidateOwnership,
) -> LodCompactionReadiness {
    if ownership.preserves_missing_bridge_candidate() {
        readiness
    } else {
        LodCompactionReadiness::AwaitingCandidates
    }
}

/// GPU buffers owned by one `(retained view, cloud asset)` pair.
pub struct GpuLodCompaction {
    /// A dynamically-sized range-descriptor prefix followed by fixed-capacity
    /// cached evaluations and stable-scan records. Keeping these roles in one
    /// binding preserves the WebGPU minimum storage-buffer binding budget.
    pub candidate_and_scan_buffer: Option<Buffer>,
    pub active_entries_buffer: Buffer,
    pub radix_scratch_buffer: Buffer,
    pub sorting_global_buffer: Buffer,
    pub sorting_status_counter_buffer: Buffer,
    pub sorting_pass_buffers: [Buffer; 4],
    pub indirect_args_buffer: Buffer,
    sorted_entry_bind_groups: [BindGroup; 2],
    config_buffer: Buffer,
    bind_group: Option<BindGroup>,
    compaction_layout: BindGroupLayout,
    candidate_evaluations_and_scan_records_bytes: u64,
    config: LodCompactionUniform,
    readiness: LodCompactionReadiness,
    candidate_upload: LodCandidateUploadTracker,
    candidate_ownership: LodCandidateOwnership,
    pipelines_ready: bool,
    generation: u64,
    compute_input_generation: u64,
    last_compaction_signature: Option<u64>,
    pending_sort_signature: Option<u64>,
    last_sorted_signature: Option<u64>,
}

impl GpuLodCompaction {
    fn new(
        render_device: &RenderDevice,
        pipeline: &LodCompactionPipeline<impl PlanarSync>,
        source_count: u32,
        allocation: LodCompactionAllocationPlan,
        lod_settings: &GaussianLodSettings,
        generation: u64,
    ) -> Self {
        let output_capacity = allocation.effective_capacity;
        debug_assert_eq!(
            allocation.scan_group_count,
            output_capacity.div_ceil(LOD_COMPACTION_WORKGROUP_SIZE)
        );
        debug_assert_eq!(
            allocation.scan_block_count,
            allocation
                .scan_group_count
                .div_ceil(LOD_COMPACTION_SCAN_BLOCK_SIZE)
        );
        debug_assert_eq!(
            allocation.candidate_and_scan_records_bytes,
            allocation.candidate_indices_bytes
                + allocation.candidate_evaluations_and_scan_records_bytes
        );
        let (config, readiness) = LodCompactionUniform::initial(
            source_count,
            output_capacity,
            lod_settings.quality_endpoint(),
            lod_settings.frustum_culling,
        );
        let config = config.with_policy(lod_settings);
        debug_assert_eq!(
            allocation.config_bytes,
            std::mem::size_of::<LodCompactionUniform>() as u64
        );
        let config_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_lod_compaction_config"),
            contents: bytemuck::bytes_of(&config),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let initial_candidate_and_scan_bytes = candidate_binding_bytes(
            u64::from(output_capacity),
            u64::from(LOD_MIN_CANDIDATE_SOURCE_WORDS),
        )
        .expect("validated candidate binding byte size");
        debug_assert_eq!(
            initial_candidate_and_scan_bytes,
            allocation.candidate_evaluations_and_scan_records_bytes
                + u64::from(LOD_MIN_CANDIDATE_SOURCE_WORDS) * std::mem::size_of::<u32>() as u64
        );
        let candidate_and_scan_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("gaussian_lod_candidate_and_scan_records"),
            size: initial_candidate_and_scan_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let active_entries_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("gaussian_lod_active_entries"),
            size: allocation.active_entries_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let radix_scratch_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("gaussian_lod_radix_scratch"),
            size: allocation.radix_scratch_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sorting_global_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("gaussian_lod_sorting_global"),
            size: allocation.sorting_global_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sorting_status_counter_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("gaussian_lod_sorting_status_counters"),
            size: allocation.sorting_status_counter_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sorting_pass_buffers = (0..4)
            .map(|index| {
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gaussian_lod_sorting_pass_index"),
                    contents: &[index, 0, 0, 0],
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("four radix pass buffers");
        debug_assert_eq!(allocation.sorting_pass_bytes, LOD_SORTING_PASS_UNIFORM_SIZE);
        let initial_args = finalized_indirect_args(
            0,
            output_capacity,
            LOD_COMPACTION_WORKGROUP_SIZE,
            LOD_COMPACTION_WORKGROUP_SIZE,
        );
        debug_assert_eq!(
            allocation.indirect_args_bytes,
            std::mem::size_of::<LodIndirectArgs>() as u64
        );
        let indirect_args_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_lod_indirect_args"),
            contents: bytemuck::bytes_of(&initial_args),
            usage: BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC,
        });
        let bind_group = create_compaction_bind_group(
            render_device,
            &pipeline.layout,
            &config_buffer,
            &candidate_and_scan_buffer,
            &active_entries_buffer,
            &indirect_args_buffer,
        );
        let sorted_entry_bind_groups = [
            create_sorted_entry_bind_group(
                render_device,
                &pipeline.sorted_layout,
                &active_entries_buffer,
            ),
            create_sorted_entry_bind_group(
                render_device,
                &pipeline.sorted_layout,
                &radix_scratch_buffer,
            ),
        ];
        Self {
            candidate_and_scan_buffer: Some(candidate_and_scan_buffer),
            active_entries_buffer,
            radix_scratch_buffer,
            sorting_global_buffer,
            sorting_status_counter_buffer,
            sorting_pass_buffers,
            indirect_args_buffer,
            sorted_entry_bind_groups,
            config_buffer,
            bind_group: Some(bind_group),
            compaction_layout: pipeline.layout.clone(),
            candidate_evaluations_and_scan_records_bytes: allocation
                .candidate_evaluations_and_scan_records_bytes,
            config,
            readiness,
            candidate_upload: LodCandidateUploadTracker::default(),
            candidate_ownership: LodCandidateOwnership::default(),
            pipelines_ready: false,
            generation,
            compute_input_generation: 1,
            last_compaction_signature: None,
            pending_sort_signature: None,
            last_sorted_signature: None,
        }
    }

    pub fn source_count(&self) -> u32 {
        self.config.source_count
    }

    pub fn output_capacity(&self) -> u32 {
        self.config.output_capacity
    }

    /// Device-safe capacity after applying buffer and storage-binding limits.
    pub fn effective_output_capacity(&self) -> u32 {
        self.config.output_capacity
    }

    pub fn candidate_count(&self) -> u32 {
        self.config.candidate_count
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn candidate_source_mode(&self) -> u32 {
        self.config.candidate_source_mode
    }

    pub fn candidate_range_count(&self) -> u32 {
        self.config.candidate_range_count
    }

    fn resize_candidate_source_prefix(
        &mut self,
        render_device: &RenderDevice,
        required_words: u32,
    ) {
        let source_words = candidate_source_capacity_after_upload(
            self.config.candidate_source_word_capacity,
            required_words,
            self.config.output_capacity,
        );
        if self.config.candidate_source_word_capacity == source_words {
            return;
        }
        debug_assert!(source_words > self.config.candidate_source_word_capacity);
        let size = u64::from(source_words) * std::mem::size_of::<u32>() as u64
            + self.candidate_evaluations_and_scan_records_bytes;

        // Drop the dependent bind group first, then the old buffer handle,
        // before allocating its one lifetime replacement. Capacity grows
        // directly to the validated maximum and never shrinks in place, so
        // later stable<->packed churn cannot form a chain of in-flight full
        // evaluation/scan generations.
        let old_bind_group = self.bind_group.take();
        drop(old_bind_group);
        let old_candidate_and_scan_buffer = self.candidate_and_scan_buffer.take();
        drop(old_candidate_and_scan_buffer);

        let candidate_and_scan_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("gaussian_lod_candidate_and_scan_records"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = create_compaction_bind_group(
            render_device,
            &self.compaction_layout,
            &self.config_buffer,
            &candidate_and_scan_buffer,
            &self.active_entries_buffer,
            &self.indirect_args_buffer,
        );
        self.candidate_and_scan_buffer = Some(candidate_and_scan_buffer);
        self.bind_group = Some(bind_group);
        self.config.candidate_source_word_capacity = source_words;
    }

    pub fn readiness(&self) -> LodCompactionReadiness {
        self.readiness
    }

    pub fn is_ready(&self) -> bool {
        self.readiness == LodCompactionReadiness::Ready
    }

    /// True after both compaction and active-radix variants have compiled.
    pub(crate) fn pipelines_ready(&self) -> bool {
        self.pipelines_ready
    }

    /// A staged bridge candidate must not execute until the main world has
    /// replaced the complete fallback atlas with generation-matched pages.
    pub(crate) fn hold_staged_candidates(&mut self) {
        if self.readiness != LodCompactionReadiness::AwaitingCandidates {
            self.readiness = LodCompactionReadiness::PendingCandidates;
        }
    }

    /// Pending candidates own a complete validated list and may have their
    /// dependent radix bind groups prepared before execution becomes Ready.
    pub(crate) fn has_staged_candidates(&self) -> bool {
        self.readiness != LodCompactionReadiness::AwaitingCandidates
    }

    /// Returns this state to the complete legacy draw path until a new
    /// identity or candidate frontier is explicitly committed.
    pub fn invalidate_candidates(&mut self, _render_device: &RenderDevice) {
        // Shrinking is synchronized with state destruction/recreation. Keeping
        // this capacity also preserves the cached payload when the exact same
        // frontier is later reactivated by fingerprint.
        self.candidate_ownership = LodCandidateOwnership::Bridge;
        self.readiness = LodCompactionReadiness::AwaitingCandidates;
        self.mark_compute_input_dirty();
    }

    fn synchronize_pipeline_readiness(&mut self, pipelines_ready: bool) {
        let was_ready = self.pipelines_ready;
        self.pipelines_ready = pipelines_ready;
        // Shader invalidation/hot reload must return to the complete legacy draw
        // until compaction and active radix can produce fresh sorted arguments.
        self.readiness = self
            .readiness
            .synchronize_pipeline_readiness(pipelines_ready);
        if was_ready != pipelines_ready {
            self.mark_compute_input_dirty();
        }
    }

    pub fn sorted_entry_bind_group(&self, radix_depth_bits: RadixSortDepthBits) -> &BindGroup {
        // Active entries are buffer A. An even number of LSD passes finishes in
        // A, while an odd number finishes in scratch buffer B.
        let index = radix_sorted_output_buffer_index(radix_depth_bits);
        &self.sorted_entry_bind_groups[index]
    }

    /// Uploads a complete hierarchy cut validated and frozen by
    /// [`LodStreamFrame::candidate_frontier`](crate::stream::runtime::LodStreamFrame::candidate_frontier),
    /// then commits candidate-list mode. Range validation completes before the
    /// GPU buffer or current configuration changes. Atlas upload/generation
    /// synchronization remains the caller's responsibility until the automatic
    /// page-atlas bridge is installed.
    pub fn upload_candidate_frontier(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        frontier: &LodCandidateFrontier,
    ) -> Result<(), LodCandidateConfigError> {
        let fingerprint = lod_candidate_frontier_fingerprint(frontier);
        if self.candidate_upload.fingerprint == Some(fingerprint) {
            self.candidate_upload.mark_unversioned(fingerprint);
            self.readiness = self.readiness.after_commit();
            self.candidate_ownership = LodCandidateOwnership::Bridge;
            return Ok(());
        }
        self.upload_candidate_frontier_data(render_device, render_queue, frontier)?;
        self.candidate_upload.mark_unversioned(fingerprint);
        self.candidate_ownership = LodCandidateOwnership::Bridge;
        Ok(())
    }

    fn synchronize_bridge_candidate_frontier(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        candidate: &LodRenderCandidate,
    ) -> Result<(), LodCandidateConfigError> {
        match self.candidate_upload.plan(candidate) {
            LodCandidateUploadPlan::ReuseVersion => {
                self.readiness = self.readiness.after_commit();
            }
            LodCandidateUploadPlan::ReuseFingerprint(fingerprint) => {
                self.candidate_upload
                    .mark_synchronized(&candidate.phase, fingerprint);
                self.readiness = self.readiness.after_commit();
            }
            LodCandidateUploadPlan::Upload(fingerprint) => {
                self.upload_candidate_data(
                    render_device,
                    render_queue,
                    candidate.rendered_candidate_count(),
                    candidate.render_ranges(),
                )?;
                self.candidate_upload
                    .mark_synchronized(&candidate.phase, fingerprint);
            }
        }
        self.candidate_ownership = LodCandidateOwnership::Bridge;
        Ok(())
    }

    fn upload_candidate_frontier_data(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        frontier: &LodCandidateFrontier,
    ) -> Result<(), LodCandidateConfigError> {
        self.upload_candidate_data(
            render_device,
            render_queue,
            frontier.candidate_count(),
            frontier.physical_ranges(),
        )
    }

    fn upload_candidate_data(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        candidate_count: u32,
        physical_ranges: &[LodPhysicalRange],
    ) -> Result<(), LodCandidateConfigError> {
        let (descriptors, range_candidate_count) =
            build_gpu_physical_range_descriptors(physical_ranges, self.config.source_count)?;
        if candidate_count != range_candidate_count {
            return Err(LodCandidateConfigError::CandidateCountMismatch {
                declared: candidate_count,
                actual: range_candidate_count,
            });
        }
        let descriptor_count = u32::try_from(descriptors.len()).map_err(|_| {
            LodCandidateConfigError::PhysicalRangeCountNotRepresentable {
                range_count: descriptors.len(),
            }
        })?;
        let required_source_words = descriptor_count
            .checked_mul(
                (std::mem::size_of::<LodGpuPhysicalRangeDescriptor>() / std::mem::size_of::<u32>())
                    as u32,
            )
            .ok_or(LodCandidateConfigError::PhysicalRangeCountOverflow)?;
        let mut next = self
            .config
            .with_physical_ranges(candidate_count, descriptor_count)?;
        let payload = bytemuck::cast_slice(&descriptors);

        self.resize_candidate_source_prefix(render_device, required_source_words);
        next.candidate_source_word_capacity = self.config.candidate_source_word_capacity;
        if !payload.is_empty() {
            render_queue.write_buffer(
                self.candidate_and_scan_buffer
                    .as_ref()
                    .expect("candidate binding is rebuilt synchronously"),
                0,
                payload,
            );
        }
        self.config = next;
        render_queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(&self.config));
        self.readiness = self.readiness.after_commit();
        self.mark_compute_input_dirty();
        Ok(())
    }

    /// Benchmark/testing injection that commits a complete candidate source
    /// directly from validated physical ranges without allocating an expanded
    /// source-sized index vector. Production bridge code uses the same
    /// descriptor builder and bounds.
    #[cfg(feature = "testing")]
    pub fn upload_physical_ranges_for_testing(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        physical_ranges: &[LodPhysicalRange],
    ) -> Result<u32, LodCandidateConfigError> {
        let (_, candidate_count) =
            build_gpu_physical_range_descriptors(physical_ranges, self.config.source_count)?;
        self.upload_candidate_data(
            render_device,
            render_queue,
            candidate_count,
            physical_ranges,
        )?;
        // A manual range payload supersedes any cached production bridge
        // version/fingerprint. The next real candidate must upload even when it
        // reuses the exact Arc/version that was current before this override.
        self.candidate_upload.revoke_for_testing_override();
        self.candidate_ownership = LodCandidateOwnership::TestingPhysicalRanges;
        Ok(candidate_count)
    }

    pub fn configure_sort_dispatch(
        &mut self,
        render_queue: &RenderQueue,
        entries_a: u32,
        entries_c: u32,
    ) {
        let entries_a = entries_a.max(1);
        let entries_c = entries_c.max(1);
        if self.config.consumer_entries_a != entries_a
            || self.config.consumer_entries_c != entries_c
        {
            self.config.consumer_entries_a = entries_a;
            self.config.consumer_entries_c = entries_c;
            render_queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(&self.config));
            self.mark_compute_input_dirty();
        }
    }

    fn set_policy(&mut self, render_queue: &RenderQueue, lod_settings: &GaussianLodSettings) {
        let mut next = self.config;
        next.set_policy_fields(lod_settings);
        if self.config != next {
            self.config = next;
            render_queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(&self.config));
            self.mark_compute_input_dirty();
        }
    }

    fn update_view_cloud_invariants(
        &mut self,
        render_queue: &RenderQueue,
        _view: &ExtractedView,
        transform: &GlobalTransform,
    ) {
        let matrix = transform.to_matrix();
        let transform_scale_bound = super::gaussian_transform_scale_bound(matrix);
        let mut next = self.config;
        next.transform_scale_bound = transform_scale_bound;
        if next != self.config {
            self.config = next;
            render_queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(&self.config));
            self.mark_compute_input_dirty();
        }
    }

    fn mark_compute_input_dirty(&mut self) {
        self.compute_input_generation = self.compute_input_generation.wrapping_add(1).max(1);
        self.last_compaction_signature = None;
        self.pending_sort_signature = None;
    }

    fn compute_signature(
        &self,
        view: &ExtractedView,
        transform: &GlobalTransform,
        settings: &CloudSettings,
        storage_generation: u32,
    ) -> u64 {
        compaction_signature(
            self.compute_input_generation,
            view,
            transform,
            settings,
            storage_generation,
        )
    }

    fn compaction_is_current(&self, signature: u64) -> bool {
        self.last_compaction_signature == Some(signature)
    }

    fn mark_compacted(&mut self, signature: u64) {
        self.last_compaction_signature = Some(signature);
        self.pending_sort_signature = Some(signature);
    }

    pub(crate) fn radix_sort_is_current(&self) -> bool {
        self.pending_sort_signature
            .is_some_and(|signature| self.last_sorted_signature == Some(signature))
    }

    pub(crate) fn sorted_signature(&self) -> Option<u64> {
        self.last_sorted_signature
    }

    pub(crate) fn sorted_output_buffer(&self, radix_depth_bits: RadixSortDepthBits) -> &Buffer {
        if radix_sorted_output_buffer_index(radix_depth_bits) == 0 {
            &self.active_entries_buffer
        } else {
            &self.radix_scratch_buffer
        }
    }

    pub(crate) fn mark_radix_sorted(&mut self) {
        if let Some(signature) = self.pending_sort_signature {
            self.last_sorted_signature = Some(signature);
        }
    }
}

fn compaction_signature(
    compute_input_generation: u64,
    view: &ExtractedView,
    transform: &GlobalTransform,
    settings: &CloudSettings,
    storage_generation: u32,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    compute_input_generation.hash(&mut hasher);
    for value in view.clip_from_view.to_cols_array() {
        value.to_bits().hash(&mut hasher);
    }
    for value in view.world_from_view.to_matrix().to_cols_array() {
        value.to_bits().hash(&mut hasher);
    }
    view.viewport.to_array().hash(&mut hasher);
    for value in transform.to_matrix().to_cols_array() {
        value.to_bits().hash(&mut hasher);
    }
    settings.global_opacity.to_bits().hash(&mut hasher);
    settings.global_scale.to_bits().hash(&mut hasher);
    settings.time.to_bits().hash(&mut hasher);
    settings.time_start.to_bits().hash(&mut hasher);
    settings.time_stop.to_bits().hash(&mut hasher);
    settings.radix_sort_depth_bits.hash(&mut hasher);
    storage_generation.hash(&mut hasher);
    hasher.finish()
}

/// Reads exactly one 48-byte indirect record from a ready compaction state.
///
/// This deliberately blocking helper is compiled only for testing and should
/// be called from an opt-in render-world probe after the frame's GPU work has
/// been submitted. Production rendering never maps or stalls on this buffer.
#[cfg(all(feature = "testing", not(target_arch = "wasm32")))]
pub fn read_lod_indirect_args_for_testing(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    state: &GpuLodCompaction,
) -> Result<LodIndirectArgs, LodIndirectArgsReadbackError> {
    if !state.is_ready() {
        return Err(LodIndirectArgsReadbackError::StateNotReady);
    }
    let staging = render_device.create_buffer(&BufferDescriptor {
        label: Some("gaussian_lod_indirect_args_test_readback"),
        size: LOD_INDIRECT_ARGS_SIZE,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = render_device.create_command_encoder(
        &bevy::render::render_resource::CommandEncoderDescriptor {
            label: Some("gaussian_lod_indirect_args_test_copy"),
        },
    );
    encoder.copy_buffer_to_buffer(
        &state.indirect_args_buffer,
        0,
        &staging,
        0,
        LOD_INDIRECT_ARGS_SIZE,
    );
    render_queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..LOD_INDIRECT_ARGS_SIZE);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    slice.map_async(
        bevy::render::render_resource::MapMode::Read,
        move |result| {
            let _ = sender.send(result.map_err(|error| error.to_string()));
        },
    );
    render_device
        .poll(bevy::render::render_resource::PollType::wait_indefinitely())
        .map_err(|error| LodIndirectArgsReadbackError::DevicePoll(error.to_string()))?;
    receiver
        .recv()
        .map_err(|_| LodIndirectArgsReadbackError::MappingChannelClosed)?
        .map_err(LodIndirectArgsReadbackError::BufferMap)?;

    let bytes = slice.get_mapped_range();
    if bytes.len() != LOD_INDIRECT_ARGS_SIZE as usize {
        let actual = bytes.len();
        drop(bytes);
        staging.unmap();
        return Err(LodIndirectArgsReadbackError::InvalidByteLength {
            expected: LOD_INDIRECT_ARGS_SIZE as usize,
            actual,
        });
    }
    let args = bytemuck::pod_read_unaligned::<LodIndirectArgs>(&bytes);
    drop(bytes);
    staging.unmap();
    Ok(args)
}

#[cfg(all(feature = "testing", target_arch = "wasm32"))]
pub fn read_lod_indirect_args_for_testing(
    _render_device: &RenderDevice,
    _render_queue: &RenderQueue,
    _state: &GpuLodCompaction,
) -> Result<LodIndirectArgs, LodIndirectArgsReadbackError> {
    Err(LodIndirectArgsReadbackError::UnsupportedPlatform)
}

fn create_sorted_entry_bind_group(
    render_device: &RenderDevice,
    layout: &BindGroupLayout,
    entries: &Buffer,
) -> BindGroup {
    render_device.create_bind_group(
        "gaussian_lod_sorted_entries",
        layout,
        &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: entries,
                offset: 0,
                size: BufferSize::new(entries.size()),
            }),
        }],
    )
}

fn create_compaction_bind_group(
    render_device: &RenderDevice,
    layout: &BindGroupLayout,
    config: &Buffer,
    candidate_indices: &Buffer,
    active_entries: &Buffer,
    indirect_args: &Buffer,
) -> BindGroup {
    render_device.create_bind_group(
        "gaussian_lod_compaction_bind_group",
        layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: config.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: candidate_indices.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: active_entries.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: indirect_args.as_entire_binding(),
            },
        ],
    )
}

/// Render-world map. A shared cloud asset can therefore have different exact
/// counts and indirect buffers for every render instance and camera.
#[derive(Resource)]
pub struct LodCompactionBuffers<R: PlanarSync> {
    entries: HashMap<(RetainedViewEntity, Entity, AssetId<R::PlanarType>), GpuLodCompaction>,
    next_generation: u64,
}

impl<R: PlanarSync> Default for LodCompactionBuffers<R> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            next_generation: 1,
        }
    }
}

impl<R: PlanarSync> LodCompactionBuffers<R> {
    /// Returns allocated state regardless of whether candidates are committed.
    /// Streaming code uses this to upload the first bounded frontier.
    pub fn get(
        &self,
        retained_view: RetainedViewEntity,
        entity: Entity,
        cloud: AssetId<R::PlanarType>,
    ) -> Option<&GpuLodCompaction> {
        self.entries
            .get(&lod_view_cloud_key(retained_view, entity, cloud))
    }

    /// Returns only state that may safely replace the complete legacy draw.
    pub fn get_ready(
        &self,
        retained_view: RetainedViewEntity,
        entity: Entity,
        cloud: AssetId<R::PlanarType>,
    ) -> Option<&GpuLodCompaction> {
        self.get(retained_view, entity, cloud)
            .filter(|state| state.is_ready())
    }

    /// Returns allocated state for uploads or invalidation, including states
    /// that are still awaiting their first complete candidate frontier.
    pub fn get_mut(
        &mut self,
        retained_view: RetainedViewEntity,
        entity: Entity,
        cloud: AssetId<R::PlanarType>,
    ) -> Option<&mut GpuLodCompaction> {
        self.entries
            .get_mut(&lod_view_cloud_key(retained_view, entity, cloud))
    }

    pub(crate) fn get_ready_mut(
        &mut self,
        retained_view: RetainedViewEntity,
        entity: Entity,
        cloud: AssetId<R::PlanarType>,
    ) -> Option<&mut GpuLodCompaction> {
        self.get_mut(retained_view, entity, cloud)
            .filter(|state| state.is_ready())
    }
}

#[derive(Clone, Copy)]
struct LodCompactionPipelines {
    reset: CachedComputePipelineId,
    count: CachedComputePipelineId,
    scan_groups: CachedComputePipelineId,
    scan_blocks: CachedComputePipelineId,
    add_block_offsets: CachedComputePipelineId,
    scatter: CachedComputePipelineId,
    finalize: CachedComputePipelineId,
}

impl LodCompactionPipelines {
    fn loaded(self, pipeline_cache: &PipelineCache) -> bool {
        [
            self.reset,
            self.count,
            self.scan_groups,
            self.scan_blocks,
            self.add_block_offsets,
            self.scatter,
            self.finalize,
        ]
        .into_iter()
        .all(|pipeline| {
            matches!(
                pipeline_cache.get_compute_pipeline_state(pipeline),
                CachedPipelineState::Ok(_)
            )
        })
    }
}

#[derive(Resource)]
struct LodCompactionPipeline<R: PlanarSync> {
    layout: BindGroupLayout,
    sorted_layout: BindGroupLayout,
    pipeline_layout: Vec<BindGroupLayoutDescriptor>,
    variants: HashMap<(GaussianMode, RadixSortDepthBits), LodCompactionPipelines>,
    marker: PhantomData<R>,
}

impl<R: PlanarSync> LodCompactionPipeline<R> {
    fn queue_variant(
        &mut self,
        pipeline_cache: &PipelineCache,
        mode: GaussianMode,
        radix_depth_bits: RadixSortDepthBits,
    ) {
        let variant_key = (mode, radix_depth_bits);
        if self.variants.contains_key(&variant_key) {
            return;
        }
        let shader_defs = shader_defs_with_defines(
            CloudPipelineKey {
                gaussian_mode: mode,
                ..default()
            },
            ShaderDefines::for_radix_depth_bits(radix_depth_bits),
        );
        let queue = |label: &'static str, entry_point: &'static str| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(label.into()),
                layout: self.pipeline_layout.clone(),
                immediate_size: 0,
                shader: LOD_COMPACTION_SHADER_HANDLE,
                shader_defs: shader_defs.clone(),
                entry_point: Some(entry_point.into()),
                zero_initialize_workgroup_memory: true,
            })
        };
        self.variants.insert(
            variant_key,
            LodCompactionPipelines {
                reset: queue("gaussian_lod_compaction_reset", "lod_reset"),
                count: queue("gaussian_lod_compaction_count", "lod_count"),
                scan_groups: queue("gaussian_lod_compaction_scan_groups", "lod_scan_groups"),
                scan_blocks: queue("gaussian_lod_compaction_scan_blocks", "lod_scan_blocks"),
                add_block_offsets: queue(
                    "gaussian_lod_compaction_add_block_offsets",
                    "lod_add_block_offsets",
                ),
                scatter: queue("gaussian_lod_compaction_scatter", "lod_scatter"),
                finalize: queue("gaussian_lod_compaction_finalize", "lod_finalize"),
            },
        );
    }
}

impl<R: PlanarSync> FromWorld for LodCompactionPipeline<R>
where
    R::GpuPlanarType: GpuPlanarStorage,
{
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let cloud_pipeline = world.resource::<CloudPipeline<R>>();
        let entries = [
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(
                        std::mem::size_of::<LodCompactionUniform>() as u64
                    ),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<u32>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<SortEntry>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(LOD_INDIRECT_ARGS_SIZE),
                },
                count: None,
            },
        ];
        let layout_descriptor =
            BindGroupLayoutDescriptor::new("gaussian_lod_compaction_layout", &entries);
        let layout = render_device
            .create_bind_group_layout(Some("gaussian_lod_compaction_layout"), &entries);
        Self {
            layout,
            sorted_layout: cloud_pipeline.sorted_layout.clone(),
            pipeline_layout: vec![
                cloud_pipeline.compute_view_layout_desc.clone(),
                cloud_pipeline.gaussian_uniform_layout_desc.clone(),
                cloud_pipeline.gaussian_cloud_layout_desc.clone(),
                layout_descriptor,
            ],
            variants: HashMap::new(),
            marker: PhantomData,
        }
    }
}

fn lod_compaction_request_is_eligible(
    requested_target: LodQualityTarget,
    candidate_target: Option<LodQualityTarget>,
    candidate_requires_compaction: bool,
) -> bool {
    // Waiting/prepared candidates need buffers to complete their handshake.
    // A package may publish an exact leaf frontier at quality one, while a
    // marginal high-detail ephemeral cut can render its retained flat source
    // directly.
    // Target equality also rejects the one-frame stale candidate possible when
    // UI settings change after the main-world bridge update.
    candidate_requires_compaction && candidate_target == Some(requested_target)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prepare_lod_compaction_buffers<R: PlanarSync>(
    mut buffers: ResMut<LodCompactionBuffers<R>>,
    mut radix_groups: ResMut<LodRadixBindGroups<R>>,
    mut pipeline: ResMut<LodCompactionPipeline<R>>,
    memory_budget: Res<LodCompactionMemoryBudget>,
    radix_pipeline: Res<RadixSortPipeline<R>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    gpu_clouds: Res<RenderAssets<R::GpuPlanarType>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities), With<GaussianCamera>>,
    clouds: Query<(
        Entity,
        &R::PlanarTypeHandle,
        &CloudSettings,
        &GaussianLodSettings,
        Option<&LodRenderCandidates>,
    )>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    let mut active = HashSet::new();
    let device_limits = render_device.limits();
    let aggregate_limit = effective_lod_compaction_aggregate_budget(
        memory_budget.max_total_bytes,
        device_limits.max_buffer_size,
    );
    let mut requests = Vec::new();
    for (view, visible_entities) in &views {
        let Some(visible_clouds) = visible_entities.get::<CloudVisibilityClass>() else {
            continue;
        };
        let camera = view.retained_view_entity.main_entity.id();
        for (render_entity, _) in &visible_clouds.entities_cpu_culling {
            let Ok((entity, handle, cloud_settings, lod_settings, candidates)) =
                clouds.get(*render_entity)
            else {
                continue;
            };
            let candidate = candidates.and_then(|set| set.by_camera.get(&camera));
            let candidate_target = candidate
                .filter(|candidate| !candidate.failed())
                .map(|candidate| candidate.frontier().quality_status().requested_target);
            let candidate_requires_compaction = candidate
                .is_some_and(|candidate| !candidate.failed() && candidate.requires_compaction());
            if !lod_compaction_request_is_eligible(
                lod_settings.quality_target(),
                candidate_target,
                candidate_requires_compaction,
            ) {
                continue;
            }
            if validate_bridge_candidate_sort_mode(&cloud_settings.sort_mode).is_err() {
                continue;
            }
            let Some(cloud) = gpu_clouds.get(handle.handle()) else {
                continue;
            };
            // u32 indices are a hard GPU ABI boundary. For representable
            // sources, allocate at most the active budget. Oversized flat
            // clouds start unready and therefore keep the complete legacy draw
            // until streaming commits an explicit bounded frontier.
            let Some(source_count) = representable_source_count(cloud.len()) else {
                continue;
            };
            let requested_capacity = source_count
                .min(lod_settings.max_active_gaussians_u32())
                .max(1);
            let Ok(allocation) = plan_lod_compaction_allocation(
                requested_capacity,
                device_limits.max_buffer_size,
                device_limits.max_storage_buffer_binding_size,
                device_limits.max_uniform_buffer_binding_size,
                device_limits.max_compute_workgroups_per_dimension,
            ) else {
                // No buffers are created. The key is intentionally not marked
                // active, so any prior state is removed and rendering stays on
                // the complete legacy path.
                continue;
            };
            requests.push((
                view.retained_view_entity,
                entity,
                handle.handle().id(),
                source_count,
                allocation,
                cloud_settings.gaussian_mode,
                cloud_settings.radix_sort_depth_bits,
                lod_settings,
            ));
        }
    }

    // Query/archetype order is not a memory-priority contract. Stable identity
    // order makes aggregate admission reproducible for the same view/cloud set.
    requests.sort_by(|left, right| {
        left.0
            .main_entity
            .cmp(&right.0.main_entity)
            .then_with(|| left.0.auxiliary_entity.cmp(&right.0.auxiliary_entity))
            .then_with(|| left.0.subview_index.cmp(&right.0.subview_index))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    requests.dedup_by(|right, left| right.0 == left.0 && right.1 == left.1 && right.2 == left.2);

    let mut aggregate_bytes = 0u64;
    let mut admitted = Vec::new();
    for request in requests {
        if reserve_lod_compaction_bytes(
            &mut aggregate_bytes,
            request.4.total_bytes,
            aggregate_limit,
        ) {
            admitted.push(request);
        }
    }

    // Drop states that are not part of the admitted set before creating any
    // replacements. This makes the configured aggregate limit a peak live
    // allocation bound during view/cloud churn, not only a steady-state bound.
    for (retained_view, entity, cloud_id, ..) in &admitted {
        active.insert(lod_view_cloud_key(*retained_view, *entity, *cloud_id));
    }
    // Bind groups own references to their bound buffers. Drop groups for removed
    // states before dropping the states themselves, otherwise churn can retain
    // the entire old allocation while the replacement is created.
    radix_groups.retain_keys(&active);
    buffers.entries.retain(|key, _| active.contains(key));

    // Determine the complete replacement set before allocating anything. If
    // two admitted keys cross sizes (old-large/new-small and
    // old-small/new-large), replacing them one at a time can exceed the
    // aggregate limit even though both the old and admitted totals fit.
    let recreate_keys = admitted
        .iter()
        .filter_map(|request| {
            let key = lod_view_cloud_key(request.0, request.1, request.2);
            buffers
                .entries
                .get(&key)
                .is_none_or(|entry| {
                    entry.source_count() != request.3
                        || entry.output_capacity() != request.4.effective_capacity
                })
                .then_some(key)
        })
        .collect::<HashSet<_>>();

    // Bind groups retain their buffers, so every dependent group in the full
    // replacement set must be dropped before any corresponding state. Only
    // after all old replacement allocations are gone may new allocation begin.
    for key in &recreate_keys {
        radix_groups.remove(key);
    }
    for key in &recreate_keys {
        buffers.entries.remove(key);
    }

    for (
        retained_view,
        entity,
        cloud_id,
        source_count,
        allocation,
        gaussian_mode,
        radix_sort_depth_bits,
        lod_settings,
    ) in admitted
    {
        pipeline.queue_variant(&pipeline_cache, gaussian_mode, radix_sort_depth_bits);
        let compaction_variant_key = (gaussian_mode, radix_sort_depth_bits);
        let compaction_pipelines_ready = pipeline
            .variants
            .get(&compaction_variant_key)
            .copied()
            .is_some_and(|pipelines| pipelines.loaded(&pipeline_cache));
        let pipelines_ready = compaction_pipelines_ready
            && radix_pipeline.variant_is_loaded(&pipeline_cache, radix_sort_depth_bits);
        let key = lod_view_cloud_key(retained_view, entity, cloud_id);
        if recreate_keys.contains(&key) {
            let generation = buffers.next_generation;
            buffers.next_generation = buffers.next_generation.wrapping_add(1).max(1);
            let mut entry = GpuLodCompaction::new(
                &render_device,
                &pipeline,
                source_count,
                allocation,
                lod_settings,
                generation,
            );
            let defines = ShaderDefines::for_radix_depth_bits(radix_sort_depth_bits);
            entry.configure_sort_dispatch(
                &render_queue,
                defines.radix_base * defines.entries_per_invocation_a,
                defines.workgroup_entries_c,
            );
            entry.synchronize_pipeline_readiness(pipelines_ready);
            buffers.entries.insert(key, entry);
        } else if let Some(entry) = buffers.entries.get_mut(&key) {
            // Identity and first candidate commits remain staged until the
            // compute variant is compiled; fallback rendering stays complete.
            entry.synchronize_pipeline_readiness(pipelines_ready);
            entry.set_policy(&render_queue, lod_settings);
            let defines = ShaderDefines::for_radix_depth_bits(radix_sort_depth_bits);
            entry.configure_sort_dispatch(
                &render_queue,
                defines.radix_base * defines.entries_per_invocation_a,
                defines.workgroup_entries_c,
            );
        }
    }
}

/// Automatically stages complete runtime frontiers and activates them only
/// after the main world has materialized the matching atlas generations.
#[allow(clippy::type_complexity)]
fn commit_lod_bridge_candidates<R: PlanarSync>(
    mut buffers: ResMut<LodCompactionBuffers<R>>,
    atlas_generations: Res<LodAtlasGpuGenerations>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    views: Query<(&ExtractedView, &RenderVisibleEntities), With<GaussianCamera>>,
    clouds: Query<(
        Entity,
        &R::PlanarTypeHandle,
        &CloudSettings,
        Option<&LodRenderCandidates>,
    )>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    for (view, visible_entities) in &views {
        let Some(visible_clouds) = visible_entities.get::<CloudVisibilityClass>() else {
            continue;
        };
        let camera = view.retained_view_entity.main_entity.id();
        for (render_entity, _) in &visible_clouds.entities_cpu_culling {
            let Ok((entity, handle, cloud_settings, candidates)) = clouds.get(*render_entity)
            else {
                continue;
            };
            let candidate = candidates.and_then(|set| set.by_camera.get(&camera));
            if validate_bridge_candidate_sort_mode(&cloud_settings.sort_mode).is_err() {
                if let Some(state) =
                    buffers.get_mut(view.retained_view_entity, entity, handle.handle().id())
                {
                    state.invalidate_candidates(&render_device);
                }
                if let Some(candidate) = candidate {
                    candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                }
                continue;
            }
            if let Some(candidate) = candidate {
                let requested_phase = candidate.phase.load(Ordering::Acquire);
                if requested_phase == LOD_RENDER_FAILED
                    || candidate.frontier.view().0 != camera.to_bits()
                {
                    if let Some(state) =
                        buffers.get_mut(view.retained_view_entity, entity, handle.handle().id())
                    {
                        state.invalidate_candidates(&render_device);
                    }
                    candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                    continue;
                }
                if !candidate.requires_compaction() {
                    // Marginal high-detail ephemeral cuts retain their logical
                    // candidate for quality/freeze/status provenance. The
                    // entity already references the exact flat source, so
                    // ordinary sorting and drawing need no atlas-generation
                    // handshake.
                    if let Some(state) =
                        buffers.get_mut(view.retained_view_entity, entity, handle.handle().id())
                    {
                        state.invalidate_candidates(&render_device);
                    }
                    candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
                    continue;
                }
            }
            let Some(state) =
                buffers.get_mut(view.retained_view_entity, entity, handle.handle().id())
            else {
                // Revoke an extracted capability if aggregate/device limits
                // removed its GPU state. The main world will restore the flat
                // atlas before publishing another active cut.
                if let Some(candidate) = candidate {
                    candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
                }
                continue;
            };
            let Some(candidate) = candidate else {
                if readiness_without_bridge_candidate(state.readiness, state.candidate_ownership)
                    != state.readiness
                {
                    state.invalidate_candidates(&render_device);
                }
                continue;
            };
            let requested_phase = candidate.phase.load(Ordering::Acquire);
            if requested_phase == LOD_RENDER_ACTIVE
                && !atlas_generations
                    .frontier_is_current(handle.handle().id().untyped(), candidate.render_ranges())
            {
                // ACTIVE is a capability, not merely a main-world intent. If
                // any physical slot upload is absent or has since been reused,
                // revoke the cut before compaction can read stale atlas data.
                state.invalidate_candidates(&render_device);
                candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
                continue;
            }
            if state
                .synchronize_bridge_candidate_frontier(&render_device, &render_queue, candidate)
                .is_err()
            {
                state.invalidate_candidates(&render_device);
                candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                continue;
            }

            let active_state_is_still_usable =
                requested_phase == LOD_RENDER_ACTIVE && state.pipelines_ready() && state.is_ready();
            if !active_state_is_still_usable {
                if state.pipelines_ready() {
                    candidate
                        .phase
                        .store(LOD_RENDER_PREPARED, Ordering::Release);
                } else {
                    candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
                }
                state.hold_staged_candidates();
            }
        }
    }
}

type LodViewQueryItem = (
    &'static ExtractedView,
    &'static GaussianComputeViewBindGroup,
    &'static ViewUniformOffset,
    &'static PreviousViewUniformOffset,
);

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_lod_compaction<R: PlanarSync>(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<LodCompactionPipeline<R>>,
    mut buffers: ResMut<LodCompactionBuffers<R>>,
    render_queue: Res<RenderQueue>,
    gaussian_uniforms: Res<GaussianUniformBindGroups>,
    view: ViewQuery<LodViewQueryItem>,
    clouds: Query<(
        Entity,
        &'static R::PlanarTypeHandle,
        Ref<'static, PlanarStorageBindGroup<R>>,
        &'static DynamicUniformIndex<CloudUniform>,
        &'static CloudSettings,
        &'static GaussianLodSettings,
        &'static GlobalTransform,
    )>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    let (extracted_view, view_bind_group, view_offset, previous_view_offset) = view.into_inner();
    let Some(uniform_bind_group) = gaussian_uniforms.base_bind_group.as_ref() else {
        return;
    };

    for (entity, handle, cloud_bind_group, cloud_uniform_index, cloud_settings, _, transform) in
        &clouds
    {
        let key = lod_view_cloud_key(
            extracted_view.retained_view_entity,
            entity,
            handle.handle().id(),
        );
        let Some(state) = buffers.entries.get_mut(&key) else {
            continue;
        };
        if !state.is_ready() {
            continue;
        }
        let Some(pipelines) = pipeline
            .variants
            .get(&(
                cloud_settings.gaussian_mode,
                cloud_settings.radix_sort_depth_bits,
            ))
            .copied()
        else {
            continue;
        };
        if !pipelines.loaded(&pipeline_cache) {
            continue;
        }

        state.update_view_cloud_invariants(&render_queue, extracted_view, transform);
        let signature = state.compute_signature(
            extracted_view,
            transform,
            cloud_settings,
            cloud_bind_group.last_changed().get(),
        );
        if state.compaction_is_current(signature) {
            continue;
        }

        macro_rules! dispatch_stage {
            ($label:literal, $pipeline_id:expr, $x:expr, $y:expr, $z:expr) => {{
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some($label),
                            ..default()
                        });
                pass.set_bind_group(
                    0,
                    &view_bind_group.value,
                    &[view_offset.offset, previous_view_offset.offset],
                );
                pass.set_bind_group(1, uniform_bind_group, &[cloud_uniform_index.index()]);
                pass.set_bind_group(2, &cloud_bind_group.bind_group, &[]);
                pass.set_bind_group(
                    3,
                    state
                        .bind_group
                        .as_ref()
                        .expect("ready compaction state has a candidate bind group"),
                    &[],
                );
                pass.set_pipeline(
                    pipeline_cache
                        .get_compute_pipeline($pipeline_id)
                        .expect("loaded LoD compaction pipeline"),
                );
                pass.dispatch_workgroups($x, $y, $z);
            }};
        }

        dispatch_stage!("lod_compaction_reset", pipelines.reset, 1, 1, 1);

        if state.candidate_count() > 0 {
            let candidate_workgroups = state
                .candidate_count()
                .div_ceil(LOD_COMPACTION_WORKGROUP_SIZE);
            let scan_blocks = candidate_workgroups.div_ceil(LOD_COMPACTION_SCAN_BLOCK_SIZE);
            debug_assert!(scan_blocks <= LOD_COMPACTION_MAX_SCAN_BLOCKS);

            dispatch_stage!(
                "lod_compaction_count",
                pipelines.count,
                candidate_workgroups,
                1,
                1
            );
            dispatch_stage!(
                "lod_compaction_scan_groups",
                pipelines.scan_groups,
                scan_blocks,
                1,
                1
            );
            dispatch_stage!("lod_compaction_scan_blocks", pipelines.scan_blocks, 1, 1, 1);
            dispatch_stage!(
                "lod_compaction_add_block_offsets",
                pipelines.add_block_offsets,
                scan_blocks,
                1,
                1
            );
            dispatch_stage!(
                "lod_compaction_scatter",
                pipelines.scatter,
                candidate_workgroups,
                1,
                1
            );
        }

        dispatch_stage!("lod_compaction_finalize", pipelines.finalize, 1, 1, 1);
        state.mark_compacted(signature);
    }
}

fn quality_endpoint_code(endpoint: LodQualityEndpoint) -> u32 {
    match endpoint {
        LodQualityEndpoint::Coarsest => 0,
        LodQualityEndpoint::Continuous => 1,
        LodQualityEndpoint::Original => 2,
    }
}

#[cfg(test)]
mod tests;
