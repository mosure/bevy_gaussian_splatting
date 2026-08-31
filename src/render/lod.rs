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
            ComputePipelineDescriptor, PipelineCache, ShaderStages, SpecializedRenderPipelines,
            TextureFormat, WgpuLimits,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        sync_world::RenderEntity,
        view::{ExtractedView, RenderVisibleEntities, RetainedViewEntity, ViewUniformOffset},
    },
};
use bevy_interleave::{interface::storage::PlanarStorageBindGroup, prelude::*};
use bytemuck::{Pod, Zeroable};

#[cfg(feature = "morph_interpolate")]
use crate::{
    gaussian::formats::planar_3d::PlanarGaussian3d,
    morph::interpolate::{GaussianInterpolate, InterpolateLabel},
};

#[cfg(feature = "morph_particles")]
use crate::morph::particle::{MorphLabel, ParticleBehaviorsHandle};

use crate::{
    camera::GaussianCamera,
    gaussian::{
        cloud::CloudVisibilityClass,
        lod_debug::{LodDebugMetadata, LodDebugResidency},
        lod_settings::{
            GaussianLodSettings, LodQualityEndpoint, LodQualityTarget, LodSelectionMode,
        },
        lodge_settings::GaussianLodgeSettings,
        settings::{CloudSettings, GaussianMode, RadixSortDepthBits},
    },
    render::{
        CloudPipeline, CloudPipelineKey, CloudPipelineReady, CloudUniform,
        GaussianComputeViewBindGroup, GaussianUniformBindGroups, LodDebugBindGroup,
        LodDebugCandidateEpoch, ShaderDefines, cloud_pipeline_key,
        gaussian_rasterization_is_supported, shader_defs_with_defines,
    },
    sort::{
        SortEntry, SortMode,
        radix::{LodRadixBindGroups, RadixSortPipeline},
    },
    stream::{
        atlas_upload::LodAtlasGpuGenerations,
        hierarchy::LodView,
        lodge::LodgeMembershipClass,
        render_commit::{
            LOD_RENDER_ACTIVE, LOD_RENDER_FAILED, LOD_RENDER_PREPARED, LOD_RENDER_TRANSITIONING,
            LOD_RENDER_WAITING, LodExternalActiveSetPresentation, LodRenderCandidate,
            LodRenderCandidates, LodRenderEnvironmentEpoch, LodViewBlendEndpoint,
        },
        runtime::{
            LodCandidateFrontier, LodPhysicalRange, LodTemporalTransitionMode, LodViewBlendBatch,
            LodViewBlendIdentity, LodViewBlendMetric, lod_view_blend_weight_checked,
        },
    },
};

#[cfg(any(test, feature = "testing"))]
use crate::stream::{render_commit::LodViewBlendWeightSnapshot, runtime::LodViewBlendEdge};

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
/// planar representation. Setting the limit to zero disables GPU compaction:
/// fallback-capable pairs stay on the complete legacy path, while package
/// transactions that require a candidate draw fail their render handshake.
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

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct LodExternalVisiblePipelineKey {
    retained_view: RetainedViewEntity,
    cloud: Entity,
    pipeline: CloudPipelineKey,
}

#[derive(Clone, Eq, PartialEq)]
struct LodCompactionEnvironmentFingerprint {
    max_total_bytes: u64,
    device_limits: WgpuLimits,
}

#[derive(Default)]
struct LodRenderEnvironmentSnapshot {
    external_visible_pipelines: HashSet<LodExternalVisiblePipelineKey>,
    compaction: Option<LodCompactionEnvironmentFingerprint>,
}

fn publish_lod_render_environment_change(
    epoch: &LodRenderEnvironmentEpoch,
    previous: &mut LodRenderEnvironmentSnapshot,
    external_visible_pipelines: HashSet<LodExternalVisiblePipelineKey>,
    compaction: LodCompactionEnvironmentFingerprint,
) -> bool {
    if previous.external_visible_pipelines == external_visible_pipelines
        && previous.compaction.as_ref() == Some(&compaction)
    {
        return false;
    }
    previous.external_visible_pipelines = external_visible_pipelines;
    previous.compaction = Some(compaction);
    epoch.advance();
    true
}

#[allow(clippy::type_complexity)]
fn update_lod_render_environment_epoch(
    epoch: Res<LodRenderEnvironmentEpoch>,
    memory_budget: Res<LodCompactionMemoryBudget>,
    render_device: Res<RenderDevice>,
    views: Query<(&ExtractedView, &RenderVisibleEntities, Option<&Msaa>), With<GaussianCamera>>,
    clouds: Query<(&CloudSettings, Option<&GaussianLodgeSettings>)>,
    mut previous: Local<LodRenderEnvironmentSnapshot>,
) {
    let mut external_visible_pipelines = HashSet::new();
    for (view, visible_entities, msaa) in &views {
        let Some(visible_clouds) = visible_entities.get::<CloudVisibilityClass>() else {
            continue;
        };
        for (render_entity, _) in &visible_clouds.entities_cpu_culling {
            let Ok((cloud_settings, Some(_))) = clouds.get(*render_entity) else {
                continue;
            };
            external_visible_pipelines.insert(LodExternalVisiblePipelineKey {
                retained_view: view.retained_view_entity,
                cloud: *render_entity,
                pipeline: cloud_pipeline_key(
                    cloud_settings,
                    false,
                    true,
                    msaa.cloned().unwrap_or_default().samples(),
                    view.target_format == TextureFormat::Rgba16Float,
                ),
            });
        }
    }
    publish_lod_render_environment_change(
        &epoch,
        &mut previous,
        external_visible_pipelines,
        LodCompactionEnvironmentFingerprint {
            max_total_bytes: memory_budget.max_total_bytes,
            device_limits: render_device.limits(),
        },
    );
}

#[cfg(test)]
mod render_environment_epoch_tests {
    use bevy::render::sync_world::MainEntity;

    use super::*;

    fn retained_view(entity: u32, subview: u32) -> RetainedViewEntity {
        RetainedViewEntity::new(
            MainEntity::from(Entity::from_raw_u32(entity).expect("valid test entity")),
            None,
            subview,
        )
    }

    fn pipeline_key(sample_count: u32, hdr: bool) -> LodExternalVisiblePipelineKey {
        LodExternalVisiblePipelineKey {
            retained_view: retained_view(1, 0),
            cloud: Entity::from_raw_u32(2).expect("valid test entity"),
            pipeline: cloud_pipeline_key(&CloudSettings::default(), false, true, sample_count, hdr),
        }
    }

    #[test]
    fn render_environment_epoch_changes_only_with_exact_pipeline_or_compaction_state() {
        let epoch = LodRenderEnvironmentEpoch::default();
        let mut previous = LodRenderEnvironmentSnapshot::default();
        let base = LodCompactionEnvironmentFingerprint {
            max_total_bytes: 32,
            device_limits: WgpuLimits::default(),
        };

        assert!(publish_lod_render_environment_change(
            &epoch,
            &mut previous,
            HashSet::new(),
            base.clone(),
        ));
        assert_eq!(epoch.current(), 1);
        assert!(!publish_lod_render_environment_change(
            &epoch,
            &mut previous,
            HashSet::new(),
            base.clone(),
        ));

        let visible = HashSet::from([pipeline_key(1, false)]);
        assert!(publish_lod_render_environment_change(
            &epoch,
            &mut previous,
            visible.clone(),
            base.clone(),
        ));
        assert_eq!(epoch.current(), 2);
        assert!(!publish_lod_render_environment_change(
            &epoch,
            &mut previous,
            visible,
            base.clone(),
        ));

        assert!(publish_lod_render_environment_change(
            &epoch,
            &mut previous,
            HashSet::from([pipeline_key(4, true)]),
            base.clone(),
        ));
        assert_eq!(epoch.current(), 3);

        let mut changed_budget = base.clone();
        changed_budget.max_total_bytes += 1;
        assert!(publish_lod_render_environment_change(
            &epoch,
            &mut previous,
            HashSet::from([pipeline_key(4, true)]),
            changed_budget.clone(),
        ));
        assert_eq!(epoch.current(), 4);

        let mut changed_limits = changed_budget;
        changed_limits.device_limits.max_buffer_size += 1;
        assert!(publish_lod_render_environment_change(
            &epoch,
            &mut previous,
            HashSet::from([pipeline_key(4, true)]),
            changed_limits,
        ));
        assert_eq!(epoch.current(), 5);
    }
}

/// Rejection reasons for candidate updates that would make the active frontier
/// incomplete or address outside its resident source allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LodCandidateConfigError {
    UnsupportedSortMode,
    SourceIndexExceedsEntryEncoding {
        source_count: u32,
        max_source_count: u32,
    },
    InvalidMorphWeight,
    InvalidExternalActiveSetWeight,
    MorphPayloadOverflow,
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
    ExternalActiveSetClassCountMismatch {
        range_count: usize,
        class_count: usize,
    },
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
            Self::SourceIndexExceedsEntryEncoding {
                source_count,
                max_source_count,
            } => write!(
                formatter,
                "LoD source count {source_count} exceeds the packed-entry limit {max_source_count}"
            ),
            Self::InvalidMorphWeight => {
                write!(
                    formatter,
                    "LoD view-blend weight must be finite and in 0..=1"
                )
            }
            Self::InvalidExternalActiveSetWeight => write!(
                formatter,
                "LoD external active-set weights must be finite, in 0..=1, and exactly complementary"
            ),
            Self::MorphPayloadOverflow => {
                write!(formatter, "LoD morph payload exceeds the portable u32 ABI")
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
            Self::ExternalActiveSetClassCountMismatch {
                range_count,
                class_count,
            } => write!(
                formatter,
                "external active-set class count {class_count} does not match physical range count {range_count}"
            ),
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

/// Applies the packed Entry source-index limit shared by hierarchy atlases and
/// resident external catalogs.
pub(crate) fn representable_source_count(source_len: usize) -> Option<u32> {
    let source_count = u32::try_from(source_len).ok()?;
    (source_count <= LOD_ENTRY_MAX_SOURCE_COUNT).then_some(source_count)
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub struct LodCompactionLabel;

/// Orders render-world observers after the radix-proven view-blend aggregate
/// has been published and any eligible Morphing candidate has activated.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub struct LodViewBlendPublicationLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
struct LodRenderEnvironmentUpdateLabel;

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
        let install_shared_system = !app.is_plugin_added::<LodCompactionPluginFlag>();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_gpu_resource::<LodCompactionBuffers<R>>()
                .init_resource::<LodCompactionMemoryBudget>()
                .init_resource::<LodRenderEnvironmentEpoch>()
                // Keep the render/compaction plugin independently usable when
                // applications opt out of the automatic streaming bridge.
                .init_gpu_resource::<LodAtlasGpuGenerations>()
                .add_systems(ExtractSchedule, extract_lod_settings::<R>)
                .add_systems(
                    Render,
                    (
                        prepare_lod_compaction_buffers::<R>.after(LodRenderEnvironmentUpdateLabel),
                        commit_lod_bridge_candidates::<R>
                            .after(prepare_lod_compaction_buffers::<R>),
                        update_lod_debug_candidate_epochs::<R>
                            .after(commit_lod_bridge_candidates::<R>),
                    )
                        .in_set(RenderSystems::PrepareResources),
                )
                .add_systems(
                    Render,
                    publish_lod_view_blend_after_radix::<R>
                        .in_set(RenderSystems::Cleanup)
                        .in_set(LodViewBlendPublicationLabel),
                );
            if install_shared_system {
                render_app.add_systems(
                    Render,
                    update_lod_render_environment_epoch
                        .in_set(RenderSystems::PrepareResources)
                        .in_set(LodRenderEnvironmentUpdateLabel),
                );
            }

            #[cfg(feature = "morph_particles")]
            render_app.configure_sets(Core3d, LodCompactionLabel.after(MorphLabel));

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

        if !install_shared_system {
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
                Option<&GaussianLodgeSettings>,
                Option<&LodRenderCandidates>,
            ),
            With<R::PlanarTypeHandle>,
        >,
    >,
) {
    for (render_entity, visibility, lod_settings, lodge_settings, bridge_candidates) in &settings {
        let mut entity = commands.entity(render_entity);
        match lod_settings.filter(|_| visibility.get()) {
            Some(settings) => {
                entity.insert(settings.clone());
            }
            None => {
                entity.remove::<GaussianLodSettings>();
            }
        }
        match lodge_settings.filter(|_| visibility.get()) {
            Some(settings) => {
                entity.insert(settings.clone());
            }
            None => {
                entity.remove::<GaussianLodgeSettings>();
            }
        }
        match bridge_candidates.filter(|_| visibility.get()) {
            Some(candidates) => {
                entity.insert((
                    candidates.clone(),
                    LodDebugCandidateEpoch {
                        candidates_are_current: candidates.candidates_are_current,
                        retained_current: candidates.retained_current,
                        debug_metadata_staged: candidates.debug_metadata_staged,
                        pending_candidate_active: candidates.by_camera.values().any(|candidate| {
                            candidate.render_is_active() || candidate.render_is_transitioning()
                        }),
                        pending_activation_armed: false,
                        required_slots: candidates
                            .by_camera
                            .values()
                            .filter(|candidate| !candidate.is_external_active_set())
                            .flat_map(LodRenderCandidate::required_atlas_ranges)
                            .map(|range| (range.page, range.slot.index, range.slot.generation))
                            .collect(),
                    },
                ));
            }
            None => {
                entity
                    .remove::<LodRenderCandidates>()
                    .remove::<LodDebugCandidateEpoch>();
            }
        }
    }
}

fn update_lod_debug_candidate_epochs<R: PlanarSync>(
    buffers: Res<LodCompactionBuffers<R>>,
    mut clouds: Query<(Entity, &LodRenderCandidates, &mut LodDebugCandidateEpoch)>,
) {
    for (entity, candidates, mut epoch) in &mut clouds {
        epoch.candidates_are_current = candidates.candidates_are_current;
        epoch.pending_candidate_active = candidates
            .by_camera
            .values()
            .any(|candidate| candidate.render_is_active() || candidate.render_is_transitioning());
        epoch.pending_activation_armed = buffers.has_pending_bridge_activation(entity);
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

/// Render-local policy shared by hierarchy and externally authored active-set
/// candidates. LODGE has no hierarchy quality endpoint, so its adapter uses
/// `Continuous` only as the otherwise-unused shader ABI sentinel.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LodCompactionPolicy {
    quality_endpoint: LodQualityEndpoint,
    selection_mode: LodSelectionMode,
    max_active_gaussians: u32,
    frustum_culling: bool,
    frustum_margin: f32,
}

impl LodCompactionPolicy {
    fn hierarchy(settings: &GaussianLodSettings) -> Self {
        Self {
            quality_endpoint: settings.quality_endpoint(),
            selection_mode: settings.selection_mode,
            max_active_gaussians: settings.max_active_gaussians_u32(),
            frustum_culling: settings.frustum_culling,
            frustum_margin: settings.frustum_margin,
        }
    }

    fn external_active_set(settings: &GaussianLodgeSettings) -> Self {
        Self {
            quality_endpoint: LodQualityEndpoint::Continuous,
            selection_mode: settings.selection_mode,
            max_active_gaussians: settings.max_active_gaussians_u32(),
            frustum_culling: settings.frustum_culling,
            frustum_margin: settings.frustum_margin,
        }
    }
}

const LOD_CANDIDATE_SOURCE_IDENTITY: u32 = 0;
const LOD_CANDIDATE_SOURCE_RANGES: u32 = 2;
const LOD_MIN_CANDIDATE_SOURCE_WORDS: u32 = 4;
// A physical Gaussian begins with a 16-byte position plane entry. WebGPU's
// u32-sized buffer limit therefore bounds every representable source index to
// 28 bits. Bits 28..29 carry a mode-qualified presentation class while the
// established two-bit Residency provenance remains in bits 30..31.
const LOD_ENTRY_SOURCE_INDEX_MASK: u32 = 0x0fff_ffff;
pub(crate) const LOD_ENTRY_MAX_SOURCE_COUNT: u32 = LOD_ENTRY_SOURCE_INDEX_MASK + 1;
pub const LOD_ENTRY_PRESENTATION_CLASS_SHIFT: u32 = 28;
pub const LOD_ENTRY_PRESENTATION_CLASS_MASK: u32 = 3 << LOD_ENTRY_PRESENTATION_CLASS_SHIFT;
#[cfg(test)]
const LOD_ENTRY_MORPH_FLAG: u32 =
    (LodExternalActiveSetClass::FirstOnly as u32) << LOD_ENTRY_PRESENTATION_CLASS_SHIFT;
pub const LOD_RANGE_PRESENTATION_CLASS_SHIFT: u32 = 2;
pub const LOD_RANGE_PRESENTATION_CLASS_MASK: u32 = 3 << LOD_RANGE_PRESENTATION_CLASS_SHIFT;
#[cfg(test)]
const LOD_RANGE_MORPH_FLAG: u32 =
    (LodExternalActiveSetClass::FirstOnly as u32) << LOD_RANGE_PRESENTATION_CLASS_SHIFT;
pub const LOD_PRESENTATION_HEADER_WORDS: u32 = 8;
const LOD_MORPH_HEADER_WORDS: u32 = LOD_PRESENTATION_HEADER_WORDS;
const LOD_MORPH_DESCRIPTOR_WORDS: u32 = 8;
const LOD_MORPH_MAPPING_WORDS: u32 = 2;

/// Runtime interpretation of the shared eight-word LoD presentation buffer.
/// A mode is part of the shader ABI: entry class `1` means a morph lookup only
/// in [`Self::Morph`], and means first-set-only opacity in
/// [`Self::ExternalActiveSet`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LodPresentationMode {
    #[default]
    None = 0,
    Morph = 1,
    ExternalActiveSet = 2,
}

/// Membership of one deduplicated external two-set union range.
///
/// The discriminant is stored in range metadata bits 2..3 and copied to sorted
/// Entry bits 28..29. `3` is deliberately unassigned and rejected by the
/// external shader path.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LodExternalActiveSetClass {
    #[default]
    Shared = 0,
    FirstOnly = 1,
    SecondOnly = 2,
}

impl From<LodgeMembershipClass> for LodExternalActiveSetClass {
    fn from(class: LodgeMembershipClass) -> Self {
        match class {
            LodgeMembershipClass::Shared => Self::Shared,
            LodgeMembershipClass::FirstOnly => Self::FirstOnly,
            LodgeMembershipClass::SecondOnly => Self::SecondOnly,
        }
    }
}

/// Header shared by the morph table and external two-active-set presentation.
/// Morph mode uses the first five words for its variable suffix; external mode
/// keeps them zero and consumes the two exact f32 weight bit patterns.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Pod, Zeroable)]
pub struct LodPresentationHeader {
    pub descriptor_count: u32,
    pub mapping_record_start: u32,
    pub mapping_record_count: u32,
    pub weight_start: u32,
    pub weight_count: u32,
    pub mode: u32,
    pub first_weight_bits: u32,
    pub second_weight_bits: u32,
}

impl LodPresentationHeader {
    const fn inactive() -> Self {
        Self {
            descriptor_count: 0,
            mapping_record_start: 0,
            mapping_record_count: 0,
            weight_start: 0,
            weight_count: 0,
            mode: LodPresentationMode::None as u32,
            first_weight_bits: 1.0_f32.to_bits(),
            second_weight_bits: 1.0_f32.to_bits(),
        }
    }

    const fn morph(
        descriptor_count: u32,
        mapping_record_start: u32,
        mapping_record_count: u32,
        weight_start: u32,
        weight_count: u32,
    ) -> Self {
        Self {
            descriptor_count,
            mapping_record_start,
            mapping_record_count,
            weight_start,
            weight_count,
            mode: LodPresentationMode::Morph as u32,
            first_weight_bits: 1.0_f32.to_bits(),
            second_weight_bits: 1.0_f32.to_bits(),
        }
    }

    /// Constructs an external-set header without recomputing or normalizing
    /// either host-derived weight. Shader consumers receive these exact bits.
    pub fn external_active_set(
        first_weight: f32,
        second_weight: f32,
    ) -> Result<Self, LodCandidateConfigError> {
        if !first_weight.is_finite()
            || !second_weight.is_finite()
            || !(0.0..=1.0).contains(&first_weight)
            || !(0.0..=1.0).contains(&second_weight)
            || first_weight.to_bits() != (1.0_f32 - second_weight).to_bits()
        {
            return Err(LodCandidateConfigError::InvalidExternalActiveSetWeight);
        }
        Ok(Self {
            descriptor_count: 0,
            mapping_record_start: 0,
            mapping_record_count: 0,
            weight_start: 0,
            weight_count: 0,
            mode: LodPresentationMode::ExternalActiveSet as u32,
            first_weight_bits: first_weight.to_bits(),
            second_weight_bits: second_weight.to_bits(),
        })
    }

    pub const fn words(self) -> [u32; LOD_PRESENTATION_HEADER_WORDS as usize] {
        [
            self.descriptor_count,
            self.mapping_record_start,
            self.mapping_record_count,
            self.weight_start,
            self.weight_count,
            self.mode,
            self.first_weight_bits,
            self.second_weight_bits,
        ]
    }

    pub fn external_active_set_coefficient(self, class: LodExternalActiveSetClass) -> f32 {
        match class {
            LodExternalActiveSetClass::Shared => 1.0,
            LodExternalActiveSetClass::FirstOnly => f32::from_bits(self.first_weight_bits),
            LodExternalActiveSetClass::SecondOnly => f32::from_bits(self.second_weight_bits),
        }
    }
}
/// Per-edge safety slew used only for late-residency endpoint admission and
/// Frozen-to-Dynamic catch-up. Fully resident Dynamic edges follow the current
/// camera statelessly, including abrupt pose changes.
const LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME: f32 = 1.0 / 12.0;

/// Bit-exact selector oracle used by end-to-end trajectory qualification.
/// Production render code calls the same runtime helper directly.
#[cfg(feature = "testing")]
pub fn lod_view_blend_weight_for_testing(
    view: LodView,
    target: LodQualityTarget,
    edge: &crate::stream::runtime::LodViewBlendEdge,
) -> f32 {
    crate::stream::runtime::lod_view_blend_weight(view, target, edge)
}

/// Exact parent/maximum-child selector pressures used by the weight oracle.
/// Same-side equality or reversed ordering is a valid categorical endpoint;
/// `None` is non-finite or threshold-contradictory and must fail qualification.
#[cfg(feature = "testing")]
pub fn lod_view_blend_pressures_for_testing(
    view: LodView,
    target: LodQualityTarget,
    edge: &crate::stream::runtime::LodViewBlendEdge,
) -> Option<(f32, f32)> {
    crate::stream::runtime::lod_view_blend_pressures_for_testing(view, target, edge)
}

const LOD_MORPH_MIN_BUFFER_BYTES: u64 =
    LOD_MORPH_HEADER_WORDS as u64 * std::mem::size_of::<u32>() as u64;
const LOD_PHYSICAL_RANGE_DESCRIPTOR_WORDS: u32 =
    (std::mem::size_of::<LodGpuPhysicalRangeDescriptor>() / std::mem::size_of::<u32>()) as u32;

/// Four-word GPU descriptor for one contiguous physical atlas range. The
/// cumulative candidate start permits logarithmic lookup without materializing
/// one index per Gaussian.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct LodGpuPhysicalRangeDescriptor {
    pub candidate_start: u32,
    pub physical_start: u32,
    pub count: u32,
    /// Packed range metadata: low two bits are [`LodDebugResidency`], bits
    /// 2..3 are [`LodExternalActiveSetClass`]. The class is interpreted only
    /// under the matching presentation-header mode. The existing fourth word
    /// makes both fields free in resident memory and upload size.
    pub metadata: u32,
}

pub(crate) enum LodCandidateMorphPlan<'a> {
    Disabled,
    Enabled {
        morph: &'a LodViewBlendBatch,
        required_words: u32,
    },
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LodCandidateMorphSynchronization {
    Disabled,
    Enabled,
    HardFallbackRequested,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LodViewBlendPressureEvaluation {
    Frozen,
    Valid { recovered_from_invalid: bool },
    Invalid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LodCandidateHardFallbackPolicy {
    /// No retained package cut depends on this token. The complete source stays
    /// available while render publishes the exact categorical target.
    RenderHardTarget,
    /// A retained package replacement was admitted as progressive Morphing.
    /// Main-world orchestration must cancel and re-admit it under hard-cut rules.
    RequestPackageReplan,
}

const fn lod_candidate_hard_fallback_policy(
    requires_package_replan: bool,
) -> LodCandidateHardFallbackPolicy {
    if requires_package_replan {
        LodCandidateHardFallbackPolicy::RequestPackageReplan
    } else {
        LodCandidateHardFallbackPolicy::RenderHardTarget
    }
}

fn publish_lod_candidate_hard_fallback(
    candidate: &LodRenderCandidate,
    policy: LodCandidateHardFallbackPolicy,
) -> LodCandidateMorphSynchronization {
    match policy {
        LodCandidateHardFallbackPolicy::RenderHardTarget => {
            candidate
                .publish_temporal_transition_mode(LodTemporalTransitionMode::BoundedHardCohort);
            LodCandidateMorphSynchronization::Disabled
        }
        LodCandidateHardFallbackPolicy::RequestPackageReplan => {
            candidate.request_hard_fallback();
            LodCandidateMorphSynchronization::HardFallbackRequested
        }
    }
}

fn lod_morph_word_capacity(required_words: u32) -> Result<u32, LodCandidateConfigError> {
    required_words
        .max(LOD_MORPH_HEADER_WORDS)
        .checked_next_power_of_two()
        .ok_or(LodCandidateConfigError::MorphPayloadOverflow)
}

const fn lod_morph_buffer_bytes(word_capacity: u32) -> Option<u64> {
    (word_capacity as u64).checked_mul(std::mem::size_of::<u32>() as u64)
}

/// Replaces the allocation plan's minimum 32-byte morph binding with the
/// exact resident grow-only capacity. A real growth temporarily owns both the
/// current and next power-of-two buffers because submitted bind groups may
/// retain the predecessor.
fn lod_compaction_admission_bytes_with_morph(
    allocation_total_bytes: u64,
    current_word_capacity: u32,
    required_words: u32,
) -> Option<u64> {
    let current_word_capacity = current_word_capacity.max(LOD_MORPH_HEADER_WORDS);
    let next_word_capacity = lod_morph_word_capacity(required_words).ok()?;
    let current_bytes = lod_morph_buffer_bytes(current_word_capacity)?;
    let next_bytes = lod_morph_buffer_bytes(next_word_capacity)?;
    let morph_peak_bytes = if next_word_capacity > current_word_capacity {
        current_bytes.checked_add(next_bytes)?
    } else {
        current_bytes
    };
    allocation_total_bytes
        .checked_sub(LOD_MORPH_MIN_BUFFER_BYTES)?
        .checked_add(morph_peak_bytes)
}

pub(crate) fn plan_lod_candidate_morph(
    candidate: &LodRenderCandidate,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
) -> Result<LodCandidateMorphPlan<'_>, LodCandidateConfigError> {
    // `BoundedHardCohort` is also a package-authored capacity veto. It is not
    // merely an adapter observation which render may upgrade again: the
    // package has already staged and leased only the exact target endpoint.
    if candidate.temporal_transition_mode() != Some(LodTemporalTransitionMode::Morphing) {
        return Ok(LodCandidateMorphPlan::Disabled);
    }
    let Some(morph) = candidate
        .temporal_transition()
        .and_then(|transition| transition.morph())
    else {
        return Ok(LodCandidateMorphPlan::Unsupported);
    };
    let identity = morph.identity();
    let required_words = LOD_MORPH_HEADER_WORDS
        .checked_add(
            identity
                .descriptor_count()
                .checked_mul(LOD_MORPH_DESCRIPTOR_WORDS)
                .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?,
        )
        .and_then(|words| {
            identity
                .mapping_record_count()
                .checked_mul(LOD_MORPH_MAPPING_WORDS)
                .and_then(|mapping| words.checked_add(mapping))
        })
        .and_then(|words| {
            u32::try_from(morph.edges().len())
                .ok()
                .and_then(|edge_count| words.checked_add(edge_count))
        })
        .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?;
    let allocation_words = lod_morph_word_capacity(required_words)?;
    let required_bytes = lod_morph_buffer_bytes(allocation_words)
        .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?;
    if required_bytes > max_buffer_size || required_bytes > max_storage_buffer_binding_size {
        return Ok(LodCandidateMorphPlan::Unsupported);
    }
    Ok(LodCandidateMorphPlan::Enabled {
        morph,
        required_words,
    })
}

fn enforce_lod_candidate_gaussian_morph_capability(
    candidate: &LodRenderCandidate,
    gaussian_mode: GaussianMode,
    fallback_policy: LodCandidateHardFallbackPolicy,
) -> bool {
    if gaussian_mode != GaussianMode::Gaussian3d
        && !candidate.render_is_active()
        && !candidate.render_is_transitioning()
        && candidate.temporal_transition_mode() == Some(LodTemporalTransitionMode::Morphing)
    {
        // ABI-16's direct parent map currently describes planar Gaussian3d
        // records. A 2D/4D candidate remains supported as an exact hard cut,
        // but must not upload morph-flagged descriptors which its raster
        // specialization cannot interpret.
        publish_lod_candidate_hard_fallback(candidate, fallback_policy);
        return true;
    }
    false
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LodViewBlendWeight {
    displayed: f32,
    desired: f32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct LodViewBlendEdgeKey {
    parent: crate::LodNodeId,
    children: Vec<crate::LodNodeId>,
    parent_metric: LodViewBlendMetricKey,
    child_metrics: Vec<LodViewBlendMetricKey>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct LodViewBlendMetricKey {
    center_bits: [u32; 3],
    radius_bits: u32,
    geometric_error_bits: u32,
    quality_min_bits: u32,
    quality_max_bits: u32,
    certificate_bits: u32,
    original_representation: bool,
}

impl LodViewBlendMetricKey {
    fn from_metric(metric: LodViewBlendMetric) -> Self {
        let node = metric.node_metrics();
        Self {
            center_bits: node.center.to_array().map(f32::to_bits),
            radius_bits: node.radius.to_bits(),
            geometric_error_bits: node.geometric_error.to_bits(),
            quality_min_bits: node.quality_min.to_bits(),
            quality_max_bits: node.quality_max.to_bits(),
            certificate_bits: node.high_fidelity_certificate.to_bits(),
            original_representation: metric.is_original_representation(),
        }
    }
}

impl LodViewBlendEdgeKey {
    fn from_edge(edge: &crate::stream::runtime::LodViewBlendEdge) -> Self {
        Self {
            parent: edge.parent(),
            children: edge.children().to_vec(),
            parent_metric: LodViewBlendMetricKey::from_metric(edge.parent_metric()),
            child_metrics: edge
                .child_metrics()
                .iter()
                .copied()
                .map(LodViewBlendMetricKey::from_metric)
                .collect(),
        }
    }

    fn matches_edge(&self, edge: &crate::stream::runtime::LodViewBlendEdge) -> bool {
        self.parent == edge.parent()
            && self.children.as_slice() == edge.children()
            && self.parent_metric == LodViewBlendMetricKey::from_metric(edge.parent_metric())
            && self.child_metrics.len() == edge.child_metrics().len()
            && self
                .child_metrics
                .iter()
                .zip(edge.child_metrics())
                .all(|(key, metric)| *key == LodViewBlendMetricKey::from_metric(*metric))
    }
}

#[derive(Clone, Debug, PartialEq)]
struct LodViewBlendEdgeState {
    key: LodViewBlendEdgeKey,
    weight: LodViewBlendWeight,
    record_count: u32,
    recovery_lag: bool,
    desired_initialized: bool,
    /// A newly admitted edge must produce and publish one exact retained
    /// endpoint image before current-camera evaluation may change its suffix.
    /// Common-key table replacement inherits this guard.
    initial_drawable_pending: bool,
}

#[derive(Clone, Debug)]
struct LodViewBlendEdgeAdmission {
    key: LodViewBlendEdgeKey,
    initial_weight: f32,
    record_count: u32,
    activation_requires_slew: bool,
}

/// One private retained view's last radix-proven blend presentation. This is
/// kept separate from the suffix currently being prepared: package retirement
/// may only observe bits which have already produced drawable sorted output.
#[derive(Clone, Debug, PartialEq)]
struct LodDrawableViewBlendSnapshot {
    displayed: Vec<f32>,
    desired: Vec<f32>,
    endpoints: Vec<LodViewBlendEndpoint>,
    lagging_edges: Vec<bool>,
    /// Per-edge exceptional recovery mode paired with the drawable weights.
    /// A newer CPU suffix may already have caught up while this older suffix is
    /// still the radix-proven image, so table replacement must not inherit the
    /// live flag independently of the displayed bits.
    recovery_edges: Vec<bool>,
    /// Invalid live pressure evaluations observed while this exact drawable
    /// suffix was held. This is presentation degradation, not ordinary slew
    /// lag. Multi-view publication reduces this mask by per-edge OR so the
    /// public count remains bounded by immutable edge count.
    invalid_pressure_edges: Vec<bool>,
    max_lag: f32,
    max_delta: f32,
    weighted_record_energy: f64,
}

impl LodDrawableViewBlendSnapshot {
    fn from_edge_states(
        edge_states: &[LodViewBlendEdgeState],
        max_delta: f32,
        weighted_record_energy: f64,
    ) -> Result<Self, LodCandidateConfigError> {
        if !max_delta.is_finite()
            || max_delta < 0.0
            || !weighted_record_energy.is_finite()
            || weighted_record_energy < 0.0
        {
            return Err(LodCandidateConfigError::InvalidMorphWeight);
        }
        let mut displayed = Vec::with_capacity(edge_states.len());
        let mut desired = Vec::with_capacity(edge_states.len());
        let mut endpoints = Vec::with_capacity(edge_states.len());
        let mut lagging_edges = Vec::with_capacity(edge_states.len());
        let mut recovery_edges = Vec::with_capacity(edge_states.len());
        let mut max_lag = 0.0_f32;
        for state in edge_states {
            let current = state.weight.displayed;
            let target = state.weight.desired;
            if !current.is_finite()
                || !(0.0..=1.0).contains(&current)
                || !target.is_finite()
                || !(0.0..=1.0).contains(&target)
            {
                return Err(LodCandidateConfigError::InvalidMorphWeight);
            }
            displayed.push(current);
            desired.push(target);
            endpoints.push(if current.to_bits() == 0.0_f32.to_bits() {
                LodViewBlendEndpoint::ParentExact
            } else if current.to_bits() == 1.0_f32.to_bits() {
                LodViewBlendEndpoint::ChildrenExact
            } else {
                LodViewBlendEndpoint::Fractional
            });
            let lag = (current - target).abs();
            lagging_edges.push(current.to_bits() != target.to_bits());
            recovery_edges.push(state.recovery_lag);
            max_lag = max_lag.max(lag);
        }
        Ok(Self {
            displayed,
            desired,
            endpoints,
            lagging_edges,
            recovery_edges,
            invalid_pressure_edges: vec![false; edge_states.len()],
            max_lag,
            max_delta,
            weighted_record_energy,
        })
    }

    fn merge_consumer(&mut self, consumer: &Self) -> Result<(), LodCandidateConfigError> {
        if self.displayed.len() != consumer.displayed.len()
            || self.desired.len() != consumer.desired.len()
            || self.endpoints.len() != consumer.endpoints.len()
            || self.lagging_edges.len() != consumer.lagging_edges.len()
            || self.recovery_edges.len() != consumer.recovery_edges.len()
            || self.invalid_pressure_edges.len() != consumer.invalid_pressure_edges.len()
        {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        for ((aggregate_endpoint, consumer_endpoint), (aggregate_lag, consumer_lag)) in self
            .endpoints
            .iter_mut()
            .zip(&consumer.endpoints)
            .zip(self.lagging_edges.iter_mut().zip(&consumer.lagging_edges))
        {
            if *aggregate_endpoint != *consumer_endpoint {
                *aggregate_endpoint = LodViewBlendEndpoint::Fractional;
            }
            *aggregate_lag |= *consumer_lag;
        }
        for (aggregate_invalid, consumer_invalid) in self
            .invalid_pressure_edges
            .iter_mut()
            .zip(&consumer.invalid_pressure_edges)
        {
            *aggregate_invalid |= *consumer_invalid;
        }
        for (aggregate_recovery, consumer_recovery) in
            self.recovery_edges.iter_mut().zip(&consumer.recovery_edges)
        {
            *aggregate_recovery |= *consumer_recovery;
        }
        self.max_lag = self.max_lag.max(consumer.max_lag);
        self.max_delta = self.max_delta.max(consumer.max_delta);
        self.weighted_record_energy = (self.weighted_record_energy
            + consumer.weighted_record_energy)
            .min(f64::from(f32::MAX));
        Ok(())
    }

    fn lagging_count(&self) -> u32 {
        self.lagging_edges
            .iter()
            .filter(|&&lagging| lagging)
            .count()
            .try_into()
            .unwrap_or(u32::MAX)
    }

    fn invalid_pressure_count(&self) -> u32 {
        self.invalid_pressure_edges
            .iter()
            .filter(|&&invalid| invalid)
            .count()
            .try_into()
            .unwrap_or(u32::MAX)
    }

    /// Retargets telemetry for a now-valid view while preserving the exact
    /// radix-proven displayed weights/endpoints. The next suffix advances
    /// separately; until then this snapshot truthfully reports recovery lag.
    fn retarget_pressure_targets(
        &mut self,
        edge_states: &[LodViewBlendEdgeState],
    ) -> Result<(), LodCandidateConfigError> {
        if self.displayed.len() != edge_states.len()
            || self.desired.len() != edge_states.len()
            || self.lagging_edges.len() != edge_states.len()
            || self.recovery_edges.len() != edge_states.len()
            || self.invalid_pressure_edges.len() != edge_states.len()
        {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        self.max_lag = 0.0;
        for ((((displayed, desired), lagging), recovery), state) in self
            .displayed
            .iter()
            .zip(&mut self.desired)
            .zip(&mut self.lagging_edges)
            .zip(&mut self.recovery_edges)
            .zip(edge_states)
        {
            *desired = state.weight.desired;
            *lagging = displayed.to_bits() != desired.to_bits();
            *recovery = state.recovery_lag;
            self.max_lag = self.max_lag.max((*displayed - *desired).abs());
        }
        Ok(())
    }

    fn recover_pressure_targets(
        &mut self,
        edge_states: &[LodViewBlendEdgeState],
    ) -> Result<(), LodCandidateConfigError> {
        self.retarget_pressure_targets(edge_states)?;
        self.invalid_pressure_edges.fill(false);
        Ok(())
    }
}

/// Compact CPU presentation state paired with one encoded morph suffix.
///
/// This deliberately excludes stable edge keys and record counts: the batch
/// identity proves their ordering, while replacement reconciliation already
/// owns those immutable values. Keeping only mutable presentation fields makes
/// the per-view double buffer bounded and allocation-stable during camera
/// motion.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LodRadixMorphEdgeState {
    weight: LodViewBlendWeight,
    recovery_lag: bool,
    desired_initialized: bool,
    initial_drawable_pending: bool,
}

impl From<&LodViewBlendEdgeState> for LodRadixMorphEdgeState {
    fn from(state: &LodViewBlendEdgeState) -> Self {
        Self {
            weight: state.weight,
            recovery_lag: state.recovery_lag,
            desired_initialized: state.desired_initialized,
            initial_drawable_pending: state.initial_drawable_pending,
        }
    }
}

/// Two-phase production proof of the morph state consumed by compaction and
/// its matching radix pass.
///
/// Prepare may already have staged a newer CPU suffix when a replacement
/// table arrives. Only `drawable_*`, promoted by the matching radix signature,
/// is therefore safe inheritance evidence. The two vectors swap on promotion,
/// reusing both allocations across ordinary camera frames.
#[derive(Default)]
struct LodRadixMorphStateTracker {
    pending_identity: Option<LodViewBlendIdentity>,
    pending_signature: Option<u64>,
    pending_edges: Vec<LodRadixMorphEdgeState>,
    pending_invalid_pressure: Vec<bool>,
    pending_evaluation_complete: bool,
    pending_max_delta: f32,
    pending_weighted_record_energy: f64,
    drawable_identity: Option<LodViewBlendIdentity>,
    drawable_signature: Option<u64>,
    drawable_edges: Vec<LodRadixMorphEdgeState>,
    drawable_invalid_pressure: Vec<bool>,
    drawable_evaluation_complete: bool,
    drawable_max_delta: f32,
    drawable_weighted_record_energy: f64,
}

impl LodRadixMorphStateTracker {
    /// Identity of the morph presentation which has actually crossed both
    /// compaction and radix. A live `GpuLodCompaction::morph_identity` may be
    /// newer: descriptor synchronization installs it before the render graph
    /// has produced the first drawable output for that table.
    fn drawable_identity(&self) -> Option<LodViewBlendIdentity> {
        self.drawable_identity
    }

    #[allow(clippy::too_many_arguments)]
    fn latch_compacted(
        &mut self,
        identity: LodViewBlendIdentity,
        signature: u64,
        edge_states: &[LodViewBlendEdgeState],
        invalid_pressure: &[bool],
        evaluation_complete: bool,
        max_delta: f32,
        weighted_record_energy: f64,
    ) -> bool {
        if edge_states.len() != invalid_pressure.len()
            || !max_delta.is_finite()
            || max_delta < 0.0
            || !weighted_record_energy.is_finite()
            || weighted_record_energy < 0.0
        {
            self.discard_pending();
            return false;
        }
        self.pending_edges.clear();
        self.pending_edges
            .extend(edge_states.iter().map(LodRadixMorphEdgeState::from));
        self.pending_invalid_pressure.clear();
        self.pending_invalid_pressure
            .extend_from_slice(invalid_pressure);
        self.pending_identity = Some(identity);
        self.pending_signature = Some(signature);
        self.pending_evaluation_complete = evaluation_complete;
        self.pending_max_delta = max_delta;
        self.pending_weighted_record_energy = weighted_record_energy;
        true
    }

    fn discard_pending(&mut self) {
        self.pending_identity = None;
        self.pending_signature = None;
        self.pending_evaluation_complete = false;
        self.pending_max_delta = 0.0;
        self.pending_weighted_record_energy = 0.0;
    }

    fn promote(&mut self, signature: u64) -> bool {
        if self.pending_signature != Some(signature) {
            self.discard_pending();
            return false;
        }
        let Some(identity) = self.pending_identity.take() else {
            self.discard_pending();
            return false;
        };
        std::mem::swap(&mut self.pending_edges, &mut self.drawable_edges);
        std::mem::swap(
            &mut self.pending_invalid_pressure,
            &mut self.drawable_invalid_pressure,
        );
        self.pending_signature = None;
        self.drawable_identity = Some(identity);
        self.drawable_signature = Some(signature);
        self.drawable_evaluation_complete = self.pending_evaluation_complete;
        self.drawable_max_delta = self.pending_max_delta;
        self.drawable_weighted_record_energy = self.pending_weighted_record_energy;
        self.pending_evaluation_complete = false;
        self.pending_max_delta = 0.0;
        self.pending_weighted_record_energy = 0.0;
        true
    }

    /// Attaches a checked desired oracle to an unchanged radix-proven suffix.
    /// Displayed bits remain the physical GPU proof; desired/recovery metadata
    /// may advance after the authored first draw without another buffer write.
    fn refresh_drawable_evaluation(
        &mut self,
        identity: LodViewBlendIdentity,
        edge_states: &[LodViewBlendEdgeState],
        invalid_pressure: &[bool],
        evaluation_complete: bool,
    ) -> bool {
        if self.drawable_identity != Some(identity)
            || self.drawable_edges.len() != edge_states.len()
            || self.drawable_invalid_pressure.len() != invalid_pressure.len()
        {
            return false;
        }
        if self
            .drawable_edges
            .iter()
            .zip(edge_states)
            .any(|(drawable, live)| {
                drawable.weight.displayed.to_bits() != live.weight.displayed.to_bits()
            })
        {
            return false;
        }
        for (drawable, live) in self.drawable_edges.iter_mut().zip(edge_states) {
            drawable.weight.desired = live.weight.desired;
            drawable.recovery_lag = live.recovery_lag;
            drawable.desired_initialized = live.desired_initialized;
            drawable.initial_drawable_pending = live.initial_drawable_pending;
        }
        self.drawable_invalid_pressure
            .clone_from_slice(invalid_pressure);
        self.drawable_evaluation_complete = evaluation_complete;
        true
    }

    fn drawable_evaluation_complete(&self, identity: LodViewBlendIdentity) -> bool {
        self.drawable_identity == Some(identity) && self.drawable_evaluation_complete
    }

    /// Reconstructs the exact presentation metadata consumed by the last
    /// matching compaction/radix pair. This is the production source for
    /// package publication; live Prepare state may already describe a newer
    /// suffix.
    fn drawable_snapshot(
        &self,
        identity: LodViewBlendIdentity,
    ) -> Result<Option<LodDrawableViewBlendSnapshot>, LodCandidateConfigError> {
        if self.drawable_identity != Some(identity) {
            return Ok(None);
        }
        if self.drawable_edges.len() != self.drawable_invalid_pressure.len() {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        let mut displayed = Vec::with_capacity(self.drawable_edges.len());
        let mut desired = Vec::with_capacity(self.drawable_edges.len());
        let mut endpoints = Vec::with_capacity(self.drawable_edges.len());
        let mut lagging_edges = Vec::with_capacity(self.drawable_edges.len());
        let mut recovery_edges = Vec::with_capacity(self.drawable_edges.len());
        let mut max_lag = 0.0_f32;
        for edge in &self.drawable_edges {
            let current = edge.weight.displayed;
            let target = edge.weight.desired;
            if !current.is_finite()
                || !(0.0..=1.0).contains(&current)
                || !target.is_finite()
                || !(0.0..=1.0).contains(&target)
            {
                return Err(LodCandidateConfigError::InvalidMorphWeight);
            }
            displayed.push(current);
            desired.push(target);
            endpoints.push(if current.to_bits() == 0.0_f32.to_bits() {
                LodViewBlendEndpoint::ParentExact
            } else if current.to_bits() == 1.0_f32.to_bits() {
                LodViewBlendEndpoint::ChildrenExact
            } else {
                LodViewBlendEndpoint::Fractional
            });
            lagging_edges.push(current.to_bits() != target.to_bits());
            recovery_edges.push(edge.recovery_lag);
            max_lag = max_lag.max((current - target).abs());
        }
        Ok(Some(LodDrawableViewBlendSnapshot {
            displayed,
            desired,
            endpoints,
            lagging_edges,
            recovery_edges,
            invalid_pressure_edges: self.drawable_invalid_pressure.clone(),
            max_lag,
            max_delta: self.drawable_max_delta,
            weighted_record_energy: self.drawable_weighted_record_energy,
        }))
    }

    fn reconciliation_seed(
        &self,
        identity: Option<LodViewBlendIdentity>,
        live: &[LodViewBlendEdgeState],
    ) -> Result<Option<Vec<LodViewBlendEdgeState>>, LodCandidateConfigError> {
        if self.drawable_identity != identity || identity.is_none() {
            return Ok(None);
        }
        if self.drawable_edges.len() != live.len()
            || self.drawable_invalid_pressure.len() != live.len()
        {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        let mut seed = live.to_vec();
        for (state, drawable) in seed.iter_mut().zip(&self.drawable_edges) {
            state.weight = drawable.weight;
            state.recovery_lag = drawable.recovery_lag;
            state.desired_initialized = drawable.desired_initialized;
            state.initial_drawable_pending = drawable.initial_drawable_pending;
        }
        Ok(Some(seed))
    }

    fn clear(&mut self) {
        self.pending_identity = None;
        self.pending_signature = None;
        self.pending_edges.clear();
        self.pending_invalid_pressure.clear();
        self.pending_evaluation_complete = false;
        self.pending_max_delta = 0.0;
        self.pending_weighted_record_energy = 0.0;
        self.drawable_identity = None;
        self.drawable_signature = None;
        self.drawable_edges.clear();
        self.drawable_invalid_pressure.clear();
        self.drawable_evaluation_complete = false;
        self.drawable_max_delta = 0.0;
        self.drawable_weighted_record_energy = 0.0;
    }
}

/// Reconstructs the edge state which produced the last radix-proven image.
///
/// `live` may already contain a newer CPU suffix. Pairing the drawable vectors
/// with those states first preserves their stable keys and non-weight lifecycle
/// fields, then edge admission reconciles the replacement table by key. New
/// edges still initialize from their authored endpoint.
#[cfg(test)]
fn lod_view_blend_drawable_reconciliation_seed(
    live: &[LodViewBlendEdgeState],
    drawable: &LodDrawableViewBlendSnapshot,
) -> Result<Vec<LodViewBlendEdgeState>, LodCandidateConfigError> {
    if drawable.displayed.len() != live.len()
        || drawable.desired.len() != live.len()
        || drawable.recovery_edges.len() != live.len()
    {
        return Err(LodCandidateConfigError::MorphPayloadOverflow);
    }
    let mut seed = live.to_vec();
    for (((state, displayed), desired), recovery_lag) in seed
        .iter_mut()
        .zip(&drawable.displayed)
        .zip(&drawable.desired)
        .zip(&drawable.recovery_edges)
    {
        if !displayed.is_finite()
            || !(0.0..=1.0).contains(displayed)
            || !desired.is_finite()
            || !(0.0..=1.0).contains(desired)
        {
            return Err(LodCandidateConfigError::InvalidMorphWeight);
        }
        state.weight = LodViewBlendWeight {
            displayed: *displayed,
            desired: *desired,
        };
        state.recovery_lag = *recovery_lag;
    }
    Ok(seed)
}

#[cfg(any(test, feature = "testing"))]
fn lod_view_blend_upload_stats_for_drawable_snapshot(
    mut live: LodViewBlendUploadStats,
    drawable: &LodDrawableViewBlendSnapshot,
) -> LodViewBlendUploadStats {
    live.edge_count = drawable.displayed.len().try_into().unwrap_or(u32::MAX);
    live.lagging_edge_count = drawable.lagging_count();
    live.last_max_delta = drawable.max_delta;
    live.last_weighted_record_energy = drawable.weighted_record_energy;
    live
}

fn reconcile_lod_view_blend_edge_admissions(
    previous: &[LodViewBlendEdgeState],
    admissions: &[LodViewBlendEdgeAdmission],
) -> Result<Vec<LodViewBlendEdgeState>, LodCandidateConfigError> {
    let mut previous_by_key = previous
        .iter()
        .cloned()
        .map(|state| (state.key.clone(), state))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::with_capacity(admissions.len());
    let next = admissions
        .iter()
        .map(|admission| {
            if !seen.insert(admission.key.clone()) {
                return Err(LodCandidateConfigError::MorphPayloadOverflow);
            }
            if let Some(mut state) = previous_by_key.remove(&admission.key) {
                state.record_count = admission.record_count;
                return Ok(state);
            }
            Ok(LodViewBlendEdgeState {
                key: admission.key.clone(),
                weight: LodViewBlendWeight::initial(admission.initial_weight)?,
                record_count: admission.record_count,
                // Capture admission provenance before any desired weight is
                // available. A common edge inherits this bit across batch
                // replacement and never consults the replacement's flag.
                recovery_lag: admission.activation_requires_slew,
                desired_initialized: false,
                initial_drawable_pending: true,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let removed_edge_is_exact = |state: &LodViewBlendEdgeState| {
        let displayed = state.weight.displayed.to_bits();
        displayed == 0.0_f32.to_bits() || displayed == 1.0_f32.to_bits()
    };
    if previous_by_key
        .values()
        .any(|state| !removed_edge_is_exact(state))
    {
        // Replacing the table would otherwise make a still-visible adjacency
        // disappear categorically. Runtime scheduling keeps lagging edges in
        // the overlap; this check is the render boundary's fail-closed proof.
        return Err(LodCandidateConfigError::MorphPayloadOverflow);
    }
    Ok(next)
}

fn reconcile_lod_view_blend_edges(
    previous: &[LodViewBlendEdgeState],
    morph: &LodViewBlendBatch,
) -> Result<Vec<LodViewBlendEdgeState>, LodCandidateConfigError> {
    let mut record_counts = vec![0_u32; morph.edges().len()];
    for descriptor in morph.descriptors() {
        let record_count = record_counts
            .get_mut(descriptor.edge_index as usize)
            .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?;
        *record_count = record_count
            .checked_add(descriptor.child_count)
            .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?;
    }
    let admissions = morph
        .edges()
        .iter()
        .zip(record_counts)
        .map(|(edge, record_count)| LodViewBlendEdgeAdmission {
            key: LodViewBlendEdgeKey::from_edge(edge),
            initial_weight: edge.initial_weight(),
            record_count,
            activation_requires_slew: edge.activation_requires_slew(),
        })
        .collect::<Vec<_>>();
    reconcile_lod_view_blend_edge_admissions(previous, &admissions)
}

impl LodViewBlendWeight {
    fn initial(weight: f32) -> Result<Self, LodCandidateConfigError> {
        if weight.to_bits() != 0.0_f32.to_bits() && weight.to_bits() != 1.0_f32.to_bits() {
            return Err(LodCandidateConfigError::InvalidMorphWeight);
        }
        Ok(Self {
            displayed: weight,
            desired: weight,
        })
    }

    fn advance_toward(&mut self, desired: f32) -> Result<bool, LodCandidateConfigError> {
        if !desired.is_finite() || !(0.0..=1.0).contains(&desired) {
            return Err(LodCandidateConfigError::InvalidMorphWeight);
        }
        self.desired = desired;
        let previous_bits = self.displayed.to_bits();
        let delta = (desired - self.displayed).clamp(
            -LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME,
            LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME,
        );
        self.displayed = (self.displayed + delta).clamp(0.0, 1.0);
        if (self.displayed - desired).abs() <= f32::EPSILON {
            self.displayed = desired;
        }
        Ok(self.displayed.to_bits() != previous_bits)
    }

    fn follow_exact(&mut self, desired: f32) -> Result<bool, LodCandidateConfigError> {
        if !desired.is_finite() || !(0.0..=1.0).contains(&desired) {
            return Err(LodCandidateConfigError::InvalidMorphWeight);
        }
        let changed = self.displayed.to_bits() != desired.to_bits();
        self.displayed = desired;
        self.desired = desired;
        Ok(changed)
    }
}

fn update_lod_view_blend_edge_weight(
    state: &mut LodViewBlendEdgeState,
    desired: f32,
    resumed_from_frozen: bool,
) -> Result<bool, LodCandidateConfigError> {
    let previous = state.weight.displayed;
    if !state.desired_initialized {
        state.desired_initialized = true;
        if resumed_from_frozen && previous.to_bits() != desired.to_bits() {
            state.recovery_lag = true;
        } else {
            state.recovery_lag &= previous.to_bits() != desired.to_bits();
        }
    } else if resumed_from_frozen && previous.to_bits() != desired.to_bits() {
        state.recovery_lag = true;
    }
    if state.recovery_lag {
        let changed = state.weight.advance_toward(desired)?;
        if state.weight.displayed.to_bits() == state.weight.desired.to_bits() {
            state.recovery_lag = false;
        }
        Ok(changed)
    } else {
        state.weight.follow_exact(desired)
    }
}

/// Publishes one fully checked selector target without moving the drawable
/// weight. A genuinely new, ordinarily prefetched edge keeps its authored
/// desired endpoint for the first exact draw. Common inherited edges and
/// late-residency edges instead expose the current target immediately so the
/// retained drawable reports truthful lag until its next suffix is sorted.
fn retarget_checked_lod_view_blend_edge_desired(
    state: &mut LodViewBlendEdgeState,
    desired: f32,
) -> Result<(), LodCandidateConfigError> {
    if !desired.is_finite() || !(0.0..=1.0).contains(&desired) {
        return Err(LodCandidateConfigError::InvalidMorphWeight);
    }
    let ordinary_authored_first_draw =
        state.initial_drawable_pending && !state.recovery_lag && !state.desired_initialized;
    if ordinary_authored_first_draw {
        return Ok(());
    }
    state.weight.desired = desired;
    state.desired_initialized = true;
    // A complete checked oracle which meets the retained displayed value has
    // finished recovery even when the edge is inherited by a replacement
    // table. Keeping the marker here would turn the next ordinary camera
    // retarget into an unnecessary bounded slew.
    state.recovery_lag &= state.weight.displayed.to_bits() != desired.to_bits();
    Ok(())
}

/// Marks the desired-only metadata overlay created by a Frozen-to-Dynamic
/// resume as recovery work before that retained suffix is published. The
/// displayed GPU suffix remains untouched; only edges whose authored first
/// draw has already been consumed may acquire resume provenance here.
fn mark_lod_view_blend_frozen_resume_recovery(states: &mut [LodViewBlendEdgeState]) {
    for state in states {
        if !state.initial_drawable_pending
            && state.desired_initialized
            && state.weight.displayed.to_bits() != state.weight.desired.to_bits()
        {
            state.recovery_lag = true;
        }
    }
}

/// Invalid pressure is a table-wide visual hold: no displayed or desired bits
/// move, but every edge must use bounded recovery when a later valid view can
/// again supply a target. This prevents a valid-after-invalid camera jump from
/// reintroducing the discontinuity which the hold avoided.
fn hold_lod_view_blend_weights_for_invalid_pressure(states: &mut [LodViewBlendEdgeState]) {
    for state in states {
        state.recovery_lag = true;
    }
}

fn lod_view_blend_lagging_edge_count(states: &[LodViewBlendEdgeState]) -> u32 {
    states
        .iter()
        .filter(|state| state.weight.displayed.to_bits() != state.weight.desired.to_bits())
        .count()
        .try_into()
        .unwrap_or(u32::MAX)
}

fn lod_view_blend_retirement_endpoint_is_current(
    displayed: f32,
    current_weight: Option<f32>,
    invalid_pressure: bool,
    endpoint: LodViewBlendEndpoint,
) -> bool {
    if invalid_pressure {
        return false;
    }
    let endpoint_weight = match endpoint {
        LodViewBlendEndpoint::Fractional => return false,
        LodViewBlendEndpoint::ParentExact => 0.0_f32,
        LodViewBlendEndpoint::ChildrenExact => 1.0_f32,
    };
    current_weight.is_some_and(|current| {
        displayed.to_bits() == endpoint_weight.to_bits()
            && current.to_bits() == endpoint_weight.to_bits()
    })
}

const fn missing_promoted_morph_predecessor_is_safe(
    has_drawable_bridge_output: bool,
    captured_morph_identity: Option<LodViewBlendIdentity>,
) -> bool {
    !has_drawable_bridge_output || captured_morph_identity.is_none()
}

fn update_lod_view_blend_edge_after_initial_draw(
    state: &mut LodViewBlendEdgeState,
    desired: Option<f32>,
    resumed_from_frozen: bool,
) -> Result<bool, LodCandidateConfigError> {
    if state.initial_drawable_pending {
        // The caller reaches this only after radix currentness captured the
        // installed suffix. Hold that exact endpoint through this visible
        // frame; the following drawable frame may track or recover.
        state.initial_drawable_pending = false;
        return Ok(false);
    }
    let Some(desired) = desired else {
        return Ok(false);
    };
    update_lod_view_blend_edge_weight(state, desired, resumed_from_frozen)
}

fn build_lod_morph_words(
    morph: &LodViewBlendBatch,
    edge_states: &[LodViewBlendEdgeState],
) -> Result<Vec<u32>, LodCandidateConfigError> {
    let descriptor_count = u32::try_from(morph.descriptors().len())
        .map_err(|_| LodCandidateConfigError::MorphPayloadOverflow)?;
    let mapping_record_count = u32::try_from(morph.records().len())
        .map_err(|_| LodCandidateConfigError::MorphPayloadOverflow)?;
    let weight_count = u32::try_from(morph.edges().len())
        .map_err(|_| LodCandidateConfigError::MorphPayloadOverflow)?;
    if edge_states.len() != morph.edges().len() {
        return Err(LodCandidateConfigError::MorphPayloadOverflow);
    }
    let mapping_record_start = LOD_MORPH_HEADER_WORDS
        .checked_add(
            descriptor_count
                .checked_mul(LOD_MORPH_DESCRIPTOR_WORDS)
                .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?,
        )
        .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?;
    let weight_start = mapping_record_start
        .checked_add(
            mapping_record_count
                .checked_mul(LOD_MORPH_MAPPING_WORDS)
                .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?,
        )
        .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?;
    let total_words = weight_start
        .checked_add(weight_count)
        .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?;
    let mut words = Vec::with_capacity(total_words as usize);
    words.extend_from_slice(
        &LodPresentationHeader::morph(
            descriptor_count,
            mapping_record_start,
            mapping_record_count,
            weight_start,
            weight_count,
        )
        .words(),
    );
    for descriptor in morph.descriptors() {
        if descriptor.edge_index >= weight_count {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        let mapping_end = descriptor
            .mapping_start
            .checked_add(descriptor.child_count)
            .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?;
        if mapping_end > mapping_record_count {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        words.extend_from_slice(&[
            descriptor.child_physical_start,
            descriptor.child_count,
            descriptor.mapping_start,
            descriptor.edge_index,
            0,
            0,
            0,
            0,
        ]);
    }
    for record in morph.records() {
        if record.parent_physical_index > LOD_ENTRY_SOURCE_INDEX_MASK || record.split_count == 0 {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        words.extend_from_slice(&[record.parent_physical_index, record.split_count]);
    }
    for (edge, state) in morph.edges().iter().zip(edge_states) {
        if !state.key.matches_edge(edge) {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        let weight = state.weight;
        if !weight.displayed.is_finite() || !(0.0..=1.0).contains(&weight.displayed) {
            return Err(LodCandidateConfigError::InvalidMorphWeight);
        }
        words.push(weight.displayed.to_bits());
    }
    debug_assert_eq!(words.len(), total_words as usize);
    Ok(words)
}

fn lod_morph_weight_start(identity: LodViewBlendIdentity) -> Result<u32, LodCandidateConfigError> {
    let mapping_record_start = LOD_MORPH_HEADER_WORDS
        .checked_add(
            identity
                .descriptor_count()
                .checked_mul(LOD_MORPH_DESCRIPTOR_WORDS)
                .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?,
        )
        .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?;
    mapping_record_start
        .checked_add(
            identity
                .mapping_record_count()
                .checked_mul(LOD_MORPH_MAPPING_WORDS)
                .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?,
        )
        .ok_or(LodCandidateConfigError::MorphPayloadOverflow)
}

impl LodCompactionUniform {
    fn identity(
        source_count: u32,
        output_capacity: u32,
        endpoint: LodQualityEndpoint,
        frustum_culling: bool,
    ) -> Result<Self, LodCandidateConfigError> {
        if source_count > LOD_ENTRY_MAX_SOURCE_COUNT {
            return Err(LodCandidateConfigError::SourceIndexExceedsEntryEncoding {
                source_count,
                max_source_count: LOD_ENTRY_MAX_SOURCE_COUNT,
            });
        }
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
        // Every emitted descriptor has a positive count, so a candidate with C
        // records can contain at most C physical ranges. The combined binding
        // is planned for that one-range-per-candidate worst case below.
        let descriptor_capacity = candidate_count;
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

    fn with_policy(mut self, policy: LodCompactionPolicy) -> Self {
        self.set_policy_fields(policy);
        self
    }

    fn set_policy_fields(&mut self, policy: LodCompactionPolicy) {
        self.quality_endpoint = quality_endpoint_code(policy.quality_endpoint);
        self.frustum_culling = u32::from(policy.frustum_culling);
        self.frustum_margin = finite_non_negative_or_zero(policy.frustum_margin);
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

    /// Commits a complete candidate after this frame has already observed the
    /// compaction/radix pipeline state. A cold candidate may run in the same
    /// frame when those pipelines are ready; drawability remains independently
    /// gated on radix publication.
    fn after_candidate_commit(self, pipelines_ready: bool) -> Self {
        self.after_commit()
            .synchronize_pipeline_readiness(pipelines_ready)
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

impl LodCandidateUploadPlan {
    const fn requires_recompute(self) -> bool {
        matches!(self, Self::ReuseFingerprint(_) | Self::Upload(_))
    }
}

const fn view_blend_predecessor_attestation_required(
    retained_package_replacement: bool,
    upload_plan: LodCandidateUploadPlan,
) -> bool {
    retained_package_replacement && !matches!(upload_plan, LodCandidateUploadPlan::ReuseVersion)
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

/// Candidate/table metadata paired with one encoded compaction generation.
///
/// The live descriptor and morph buffers may already contain a replacement by
/// Render Cleanup. This record is latched only after the compaction commands
/// which consume those bytes have been encoded, then becomes drawable only
/// when the matching radix commands are encoded in the ordered Core3d graph.
#[cfg(any(test, feature = "testing"))]
#[derive(Clone, Debug)]
struct LodRadixCandidateSnapshot {
    version: Option<Arc<AtomicU8>>,
    phase_at_compaction: Option<u8>,
    fingerprint: Option<LodCandidateFrontierFingerprint>,
    candidate_content_signature: Option<u64>,
    candidate_atlas_allocation_epoch: Option<u64>,
    rendered_candidate_count: u32,
    morph_identity: Option<LodViewBlendIdentity>,
    compute_input_generation: u64,
    compaction_signature: u64,
    view_blend: Option<LodLastRadixViewBlendForTesting>,
}

/// Two-phase publication for the physical candidate output. A staged
/// descriptor/count is never observable as drawable until radix promotes the
/// exact compaction generation which consumed it.
#[cfg(any(test, feature = "testing"))]
#[derive(Default)]
struct LodRadixDrawableTracker {
    pending: Option<LodRadixCandidateSnapshot>,
    drawable: Option<LodRadixCandidateSnapshot>,
    drawable_publication_generation: u64,
}

#[cfg(any(test, feature = "testing"))]
impl LodRadixDrawableTracker {
    fn latch_compacted(&mut self, snapshot: LodRadixCandidateSnapshot) {
        self.pending = Some(snapshot);
    }

    fn discard_pending(&mut self) {
        self.pending = None;
    }

    fn promote(&mut self, compaction_signature: u64) -> bool {
        let Some(pending) = self.pending.take() else {
            return false;
        };
        if pending.compaction_signature != compaction_signature {
            return false;
        }
        self.drawable = Some(pending);
        self.drawable_publication_generation =
            self.drawable_publication_generation.saturating_add(1);
        true
    }

    /// Attaches a newer checked selector oracle to an unchanged physical
    /// output. This is intentionally stricter than the ordinary testing
    /// publication refresh: the radix-latched displayed and desired bits must
    /// still match the complete live evaluation bit-for-bit. Candidate
    /// identity, generations, weights, upload counters, and indirect state are
    /// never changed here.
    fn refresh_complete_view_blend_evaluation(
        &mut self,
        identity: LodViewBlendIdentity,
        edge_states: &[LodViewBlendEdgeState],
        evaluated_weights: &[f32],
        evaluation_view: Option<LodView>,
        evaluation_target: Option<LodQualityTarget>,
    ) -> bool {
        let Some((evaluation_view, evaluation_target)) = evaluation_view.zip(evaluation_target)
        else {
            return false;
        };
        let Some(drawable) = self.drawable.as_mut() else {
            return false;
        };
        if drawable.morph_identity != Some(identity) {
            return false;
        }
        let Some(view_blend) = drawable.view_blend.as_mut() else {
            return false;
        };
        if view_blend.identity != identity
            || view_blend.edges.len() != edge_states.len()
            || view_blend.weights.len() != edge_states.len()
            || view_blend.invalid_pressure.len() != edge_states.len()
            || evaluated_weights.len() != edge_states.len()
            || view_blend.invalid_pressure.iter().any(|&invalid| invalid)
            || !view_blend
                .weights
                .iter()
                .zip(edge_states)
                .zip(evaluated_weights)
                .all(|((published, state), evaluated)| {
                    state.desired_initialized
                        && !state.initial_drawable_pending
                        && state.weight.displayed.to_bits() == evaluated.to_bits()
                        && state.weight.desired.to_bits() == evaluated.to_bits()
                        && published.displayed.to_bits() == state.weight.displayed.to_bits()
                        && published.desired.to_bits() == state.weight.desired.to_bits()
                })
        {
            return false;
        }

        view_blend.evaluation_view = Some(evaluation_view);
        view_blend.evaluation_target = Some(evaluation_target);
        view_blend.desired_evaluation_complete = true;
        true
    }

    /// Retargets desired/recovery/invalid metadata for an unchanged promoted
    /// suffix after a checked table-wide evaluation. Displayed bits and every
    /// physical generation remain immutable; lag is re-derived from the exact
    /// published weight vector.
    fn refresh_checked_view_blend_evaluation(
        &mut self,
        identity: LodViewBlendIdentity,
        edge_states: &[LodViewBlendEdgeState],
        invalid_pressure: &[bool],
        evaluation_view: Option<LodView>,
        evaluation_target: Option<LodQualityTarget>,
        evaluation_complete: bool,
    ) -> bool {
        let Some(drawable) = self.drawable.as_mut() else {
            return false;
        };
        let Some(view_blend) = drawable.view_blend.as_mut() else {
            return false;
        };
        if drawable.morph_identity != Some(identity)
            || view_blend.identity != identity
            || view_blend.weights.len() != edge_states.len()
            || view_blend.recovery_lag.len() != edge_states.len()
            || view_blend.invalid_pressure.len() != invalid_pressure.len()
            || view_blend
                .weights
                .iter()
                .zip(edge_states)
                .any(|(published, state)| {
                    published.displayed.to_bits() != state.weight.displayed.to_bits()
                })
        {
            return false;
        }
        for ((published, recovery), state) in view_blend
            .weights
            .iter_mut()
            .zip(&mut view_blend.recovery_lag)
            .zip(edge_states)
        {
            published.desired = state.weight.desired;
            *recovery = state.recovery_lag;
        }
        view_blend
            .invalid_pressure
            .clone_from_slice(invalid_pressure);
        view_blend.upload.lagging_edge_count = view_blend
            .weights
            .iter()
            .filter(|weight| weight.displayed.to_bits() != weight.desired.to_bits())
            .count()
            .try_into()
            .unwrap_or(u32::MAX);
        let evaluation = evaluation_view.zip(evaluation_target);
        view_blend.evaluation_view = evaluation.map(|(view, _)| view);
        view_blend.evaluation_target = evaluation.map(|(_, target)| target);
        view_blend.desired_evaluation_complete = evaluation_complete && evaluation.is_some();
        true
    }

    fn clear(&mut self) {
        self.pending = None;
        self.drawable = None;
    }
}

fn lod_candidate_frontier_fingerprint(
    frontier: &LodCandidateFrontier,
) -> LodCandidateFrontierFingerprint {
    lod_candidate_parts_fingerprint_with_residency(
        frontier.view().0,
        frontier.physical_ranges(),
        frontier.candidate_count(),
        |node| candidate_residency_code(frontier, node),
        frontier.temporal_transition(),
        frontier
            .temporal_transition()
            .map(|transition| transition.mode()),
    )
}

fn lod_bridge_candidate_fingerprint(
    candidate: &LodRenderCandidate,
) -> LodCandidateFrontierFingerprint {
    let fingerprint = lod_candidate_parts_fingerprint_with_residency(
        candidate.frontier().view().0,
        candidate.render_ranges(),
        candidate.rendered_candidate_count(),
        |node| candidate_residency_code(candidate.frontier(), node),
        candidate.temporal_transition(),
        candidate.temporal_transition_mode(),
    );
    let Some(presentation) = candidate.external_active_set() else {
        return fingerprint;
    };
    extend_lod_candidate_fingerprint(
        fingerprint,
        std::iter::once(0x4c4f_4447_455f_4558_u64)
            .chain([
                u64::from(presentation.pair().first.0),
                u64::from(presentation.pair().second.0),
            ])
            .chain(
                presentation
                    .first_center()
                    .into_iter()
                    .chain(presentation.second_center())
                    .map(|value| u64::from(value.to_bits())),
            )
            .chain(std::iter::once(presentation.range_classes().len() as u64))
            .chain(
                presentation
                    .range_classes()
                    .iter()
                    .map(|class| u64::from(*class as u8)),
            ),
    )
}

fn extend_lod_candidate_fingerprint(
    mut fingerprint: LodCandidateFrontierFingerprint,
    values: impl IntoIterator<Item = u64>,
) -> LodCandidateFrontierFingerprint {
    for value in values {
        for byte in value.to_le_bytes() {
            fingerprint.primary ^= u64::from(byte);
            fingerprint.primary = fingerprint.primary.wrapping_mul(0x0000_0100_0000_01b3);
            fingerprint.secondary ^= u64::from(byte).wrapping_add(0x9e37_79b9_7f4a_7c15);
            fingerprint.secondary = fingerprint
                .secondary
                .rotate_left(27)
                .wrapping_mul(0x3c79_ac49_2ba7_b653)
                .wrapping_add(0x1c69_b3f7_4ac4_ae35);
        }
    }
    fingerprint
}

fn lod_candidate_parts_fingerprint(
    view: u64,
    ranges: &[LodPhysicalRange],
    candidate_count: u32,
) -> LodCandidateFrontierFingerprint {
    lod_candidate_parts_fingerprint_with_residency(
        view,
        ranges,
        candidate_count,
        |_| LodDebugResidency::Unknown as u32,
        None,
        None,
    )
}

fn lod_candidate_parts_fingerprint_with_residency(
    view: u64,
    ranges: &[LodPhysicalRange],
    candidate_count: u32,
    mut residency_for_node: impl FnMut(crate::LodNodeId) -> u32,
    temporal_transition: Option<&crate::stream::runtime::LodTemporalTransition>,
    effective_transition_mode: Option<LodTemporalTransitionMode>,
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
        write(u64::from(residency_for_node(range.node)));
    }
    write(u64::from(candidate_count));
    if let Some(transition) = temporal_transition {
        write(1);
        write(
            match effective_transition_mode.unwrap_or_else(|| transition.mode()) {
                crate::stream::runtime::LodTemporalTransitionMode::Morphing => 1,
                crate::stream::runtime::LodTemporalTransitionMode::BoundedHardCohort => 0,
            },
        );
        write(transition.changed_gaussians());
        write(transition.atomic_budget_overshoot());
        write(transition.substitutions().len() as u64);
        for substitution in transition.substitutions() {
            write(substitution.key.parent.0);
            write(match substitution.key.direction {
                crate::stream::hierarchy::LodTemporalDirection::Coarsen => 0,
                crate::stream::hierarchy::LodTemporalDirection::Refine => 1,
            });
            write(substitution.previous_nodes.len() as u64);
            for node in &substitution.previous_nodes {
                write(node.0);
            }
            write(substitution.next_nodes.len() as u64);
            for node in &substitution.next_nodes {
                write(node.0);
            }
            write(substitution.previous_gaussians);
            write(substitution.next_gaussians);
        }
        if let Some(morph) = transition.morph() {
            let identity = morph.identity();
            write(identity.primary());
            write(identity.secondary());
            write(u64::from(identity.descriptor_count()));
            write(u64::from(identity.mapping_record_count()));
        }
    } else {
        write(0);
    }
    LodCandidateFrontierFingerprint {
        primary,
        secondary,
        range_count: ranges.len().try_into().unwrap_or(u32::MAX),
        candidate_count,
    }
}

/// Test-only signature for the live camera inputs that can change compaction,
/// frustum acceptance, and depth ordering even when a Frozen LoD cut keeps an
/// identical candidate fingerprint. Keeping these controls separate prevents
/// a sort/filter artifact from being misclassified as a hierarchy transition.
#[cfg(test)]
fn lod_live_camera_sort_signature(view: &ExtractedView) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut write = |value: u32| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for value in view.world_from_view.to_matrix().to_cols_array() {
        write(value.to_bits());
    }
    for value in view.clip_from_view.to_cols_array() {
        write(value.to_bits());
    }
    for value in view.viewport.to_array() {
        write(value);
    }
    hash
}

/// Reconstructs the selector's compact projection view from Bevy's extracted
/// matrices. This intentionally omits the frustum: view-blend pressure uses
/// projection, distance, and coverage, while compaction independently applies
/// the exact extracted frustum to the interpolated Gaussian support.
fn lod_view_blend_view(
    view: &ExtractedView,
    world_from_local: &GlobalTransform,
) -> Option<LodView> {
    let clip = view.clip_from_view;
    let viewport_height_px = view.viewport.w as f32;
    let y_scale = clip.y_axis.y.abs();
    if !clip.is_finite()
        || !viewport_height_px.is_finite()
        || viewport_height_px <= 0.0
        || !y_scale.is_finite()
        || y_scale <= f32::EPSILON
    {
        return None;
    }
    let camera_position = view.world_from_view.translation();
    let lod_view = if clip.w_axis.w == 0.0 {
        let vertical_fov_radians = 2.0 * (1.0 / y_scale).atan();
        let near_plane = clip.w_axis.z.abs().max(f32::EPSILON);
        LodView::perspective(
            camera_position,
            viewport_height_px,
            vertical_fov_radians,
            near_plane,
        )
    } else if clip.w_axis.w == 1.0 {
        let vertical_world_size = 2.0 / y_scale;
        // Orthographic pressure is translation-invariant and never consumes
        // near_plane; retain a finite positive sentinel for LodView validity.
        LodView::orthographic(
            camera_position,
            viewport_height_px,
            vertical_world_size,
            f32::EPSILON,
        )
    } else {
        return None;
    };
    Some(lod_view.with_world_from_local(world_from_local.to_matrix()))
}

/// Reconstructs the exact selector view used by render-owned blending. This
/// avoids duplicating Bevy projection-matrix interpretation in integration
/// qualification while keeping the production helper private.
#[cfg(feature = "testing")]
pub fn lod_view_blend_view_for_testing(
    view: &ExtractedView,
    world_from_local: &GlobalTransform,
) -> Option<LodView> {
    lod_view_blend_view(view, world_from_local)
}

fn build_gpu_physical_range_descriptors(
    ranges: &[LodPhysicalRange],
    source_count: u32,
) -> Result<(Vec<LodGpuPhysicalRangeDescriptor>, u32), LodCandidateConfigError> {
    build_gpu_physical_range_descriptors_with_residency(ranges, source_count, |_| {
        LodDebugResidency::Unknown as u32
    })
}

fn build_gpu_physical_range_descriptors_with_residency(
    ranges: &[LodPhysicalRange],
    source_count: u32,
    mut residency_for_node: impl FnMut(crate::LodNodeId) -> u32,
) -> Result<(Vec<LodGpuPhysicalRangeDescriptor>, u32), LodCandidateConfigError> {
    build_gpu_physical_range_descriptors_with_classes(
        ranges,
        source_count,
        &mut residency_for_node,
        |_, _| LodExternalActiveSetClass::Shared,
    )
}

fn build_gpu_physical_range_descriptors_with_classes(
    ranges: &[LodPhysicalRange],
    source_count: u32,
    mut residency_for_node: impl FnMut(crate::LodNodeId) -> u32,
    mut class_for_range: impl FnMut(usize, &LodPhysicalRange) -> LodExternalActiveSetClass,
) -> Result<(Vec<LodGpuPhysicalRangeDescriptor>, u32), LodCandidateConfigError> {
    if source_count > LOD_ENTRY_MAX_SOURCE_COUNT {
        return Err(LodCandidateConfigError::SourceIndexExceedsEntryEncoding {
            source_count,
            max_source_count: LOD_ENTRY_MAX_SOURCE_COUNT,
        });
    }
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
            metadata: residency_for_node(range.node) & 3
                | (((class_for_range(range_index, range) as u32)
                    << LOD_RANGE_PRESENTATION_CLASS_SHIFT)
                    & LOD_RANGE_PRESENTATION_CLASS_MASK),
        });
        candidate_start = candidate_start
            .checked_add(range.count)
            .ok_or(LodCandidateConfigError::PhysicalRangeCountOverflow)?;
    }
    Ok((descriptors, candidate_start))
}

/// Builds the range payload for one already-deduplicated external two-set
/// union. Membership remains range-level and is copied into every compacted
/// Entry without materializing a source-sized side buffer.
pub(crate) fn build_gpu_external_active_set_range_descriptors(
    ranges: &[LodPhysicalRange],
    classes: &[LodgeMembershipClass],
    source_count: u32,
) -> Result<(Vec<LodGpuPhysicalRangeDescriptor>, u32), LodCandidateConfigError> {
    if ranges.len() != classes.len() {
        return Err(
            LodCandidateConfigError::ExternalActiveSetClassCountMismatch {
                range_count: ranges.len(),
                class_count: classes.len(),
            },
        );
    }
    build_gpu_physical_range_descriptors_with_classes(
        ranges,
        source_count,
        |_| LodDebugResidency::Resident as u32,
        |range_index, _| classes[range_index].into(),
    )
}

fn candidate_range_is_morphed(candidate: &LodRenderCandidate, range: &LodPhysicalRange) -> bool {
    candidate
        .temporal_transition()
        .and_then(|transition| transition.morph())
        .is_some_and(|morph| {
            let descriptors = morph.descriptors();
            let index = descriptors.partition_point(|descriptor| {
                descriptor.child_physical_start < range.physical_start
            });
            descriptors.get(index).is_some_and(|descriptor| {
                descriptor.child_physical_start == range.physical_start
                    && descriptor.child_count == range.count
            })
        })
}

pub(crate) fn build_bridge_candidate_upload_descriptors(
    candidate: &LodRenderCandidate,
    source_count: u32,
    morph_enabled: bool,
) -> Result<(Vec<LodGpuPhysicalRangeDescriptor>, u32), LodCandidateConfigError> {
    let ranges = if morph_enabled {
        candidate.render_ranges()
    } else {
        candidate.target_render_ranges()
    };
    build_gpu_physical_range_descriptors_with_classes(
        ranges,
        source_count,
        |node| candidate_residency_code(candidate.frontier(), node),
        |_, range| {
            if morph_enabled && candidate_range_is_morphed(candidate, range) {
                LodExternalActiveSetClass::FirstOnly
            } else {
                LodExternalActiveSetClass::Shared
            }
        },
    )
}

#[inline]
fn candidate_residency_code(frontier: &LodCandidateFrontier, node: crate::LodNodeId) -> u32 {
    if frontier.is_ancestor_fallback(node) {
        LodDebugResidency::AncestorFallback as u32
    } else {
        LodDebugResidency::Resident as u32
    }
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
    morph_base_bytes: u64,
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

/// Aggregate admission favors package outputs that are already drawable, then
/// package requests that have no complete legacy fallback. Stable identity
/// order remains the tie-breaker within each class.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum LodCompactionAdmissionClass {
    RetainedRequiredOutput,
    RequiredOutput,
    FallbackCapable,
}

struct LodCompactionAdmissionRequest<'a, T> {
    payload: T,
    total_bytes: u64,
    class: LodCompactionAdmissionClass,
    required_phase: Option<&'a AtomicU8>,
    /// This allocation already exists and is the last drawable output while a
    /// render capability veto round-trips through package orchestration. A
    /// newly lowered ceiling may block all other work, but cannot retroactively
    /// destroy this state before the package cancels the held token.
    pinned_existing: bool,
}

fn admit_lod_compaction_requests<T>(
    mut requests: Vec<LodCompactionAdmissionRequest<'_, T>>,
    aggregate_limit: u64,
) -> Vec<T> {
    // This is a stable sort: callers first establish deterministic identity
    // order, which remains the tie-breaker within each admission class.
    requests.sort_by_key(|request| (!request.pinned_existing, request.class));

    let mut aggregate_bytes = 0u64;
    let mut admitted = Vec::new();
    for request in requests {
        if request.pinned_existing {
            aggregate_bytes = aggregate_bytes.saturating_add(request.total_bytes);
            admitted.push(request.payload);
        } else if reserve_lod_compaction_bytes(
            &mut aggregate_bytes,
            request.total_bytes,
            aggregate_limit,
        ) {
            admitted.push(request.payload);
        } else if let Some(phase) = request.required_phase {
            // Candidate-required package atlases have no safe raw-atlas draw.
            // Publish a terminal cross-world result instead of leaving their
            // main-world transaction WAITING forever behind a fixed prefix.
            phase.store(LOD_RENDER_FAILED, Ordering::Release);
        }
    }
    admitted
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

fn maximum_candidate_source_words(candidate_capacity: u64) -> Option<u64> {
    candidate_capacity.checked_mul(u64::from(LOD_PHYSICAL_RANGE_DESCRIPTOR_WORDS))
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
    candidate_binding_bytes(
        candidate_capacity,
        maximum_candidate_source_words(candidate_capacity)?,
    )
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
    let candidate_descriptor_stride = std::mem::size_of::<LodGpuPhysicalRangeDescriptor>() as u64;
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
        .min(storage_buffer_limit / candidate_descriptor_stride)
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
        candidate_descriptor_stride,
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
        LOD_MORPH_MIN_BUFFER_BYTES,
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
        morph_base_bytes: LOD_MORPH_MIN_BUFFER_BYTES,
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
#[cfg(any(test, feature = "testing"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodViewBlendUploadStats {
    pub immutable_table_upload_count: u64,
    pub weight_write_count: u64,
    pub buffer_allocation_count: u64,
    pub weight_bytes_written: u64,
    pub edge_count: u32,
    pub word_capacity: u32,
    pub lagging_edge_count: u32,
    pub last_max_delta: f32,
    pub last_weighted_record_energy: f64,
    /// Declared safety bound for late-readiness and Frozen-resume recovery.
    /// Ordinary fully resident Dynamic motion is exact and does not use it.
    pub max_weight_delta_per_frame: f32,
}

/// Exact immutable edge table and weight telemetry paired with one promoted
/// radix output. Unlike the live candidate snapshot, this remains available
/// while a replacement is only WAITING/PREPARED.
#[cfg(any(test, feature = "testing"))]
#[derive(Clone, Debug, PartialEq)]
pub struct LodLastRadixViewBlendForTesting {
    pub identity: LodViewBlendIdentity,
    pub edges: Vec<LodViewBlendEdge>,
    pub weights: Vec<LodViewBlendWeightSnapshot>,
    pub endpoints: Vec<LodViewBlendEndpoint>,
    pub recovery_lag: Vec<bool>,
    pub invalid_pressure: Vec<bool>,
    pub evaluation_view: Option<LodView>,
    pub evaluation_target: Option<LodQualityTarget>,
    pub desired_evaluation_complete: bool,
    pub upload: LodViewBlendUploadStats,
}

/// Testing-only proof of the exact candidate metadata consumed by the last
/// radix-promoted output.
///
/// `candidate_token_matches` distinguishes a retained old output from a
/// newly extracted replacement token. `candidate_content_matches` is the
/// direction-independent view/range/content comparison, so an Arc identity
/// change alone does not masquerade as a visual change. The indirect argument
/// readback from the same [`GpuLodCompaction`] can be paired with
/// `compaction_generation` and `radix_publication_generation` in Render
/// Cleanup.
#[cfg(any(test, feature = "testing"))]
#[derive(Clone, Debug, PartialEq)]
pub struct LodLastRadixDrawableForTesting {
    pub compaction_generation: u64,
    pub compute_input_generation: u64,
    pub radix_publication_generation: u64,
    pub rendered_candidate_count: u32,
    pub phase_at_compaction: Option<u8>,
    pub candidate_token_matches: bool,
    pub candidate_content_matches: bool,
    pub candidate_fingerprint_primary: Option<u64>,
    pub candidate_fingerprint_secondary: Option<u64>,
    pub candidate_range_count: Option<u32>,
    pub candidate_content_signature: Option<u64>,
    pub candidate_atlas_allocation_epoch: Option<u64>,
    pub morph_identity: Option<LodViewBlendIdentity>,
    pub view_blend: Option<LodLastRadixViewBlendForTesting>,
}

/// Testing-only metadata captured with one radix-proven view-blend suffix.
///
/// The live [`ExtractedView`] and upload counters may already describe the
/// next suffix by Render Cleanup. This record instead stays paired with the
/// exact drawable weights published to [`LodRenderCandidate`]. An absent
/// evaluation view/target is intentional for the first authored endpoint
/// publication, before any camera-conditioned desired value has been applied.
#[cfg(any(test, feature = "testing"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodViewBlendDrawablePublicationForTesting {
    /// Allocation identity of the private retained-view compaction state.
    pub compaction_generation: u64,
    /// Monotonic capture generation within this compaction allocation.
    pub publication_generation: u64,
    /// Selector view which produced captured desired bits, or which validated
    /// the still-authored endpoint during pending-activation preflight.
    pub evaluation_view: Option<LodView>,
    /// Quality target paired bit-for-bit with [`Self::evaluation_view`].
    pub evaluation_target: Option<LodQualityTarget>,
    /// True only when every captured desired edge was evaluated from the same
    /// view and target above. An authored endpoint remains false even after
    /// its view/target pair is attached as pressure-preflight evidence.
    pub desired_evaluation_complete: bool,
    /// Resource counters and aggregate status frozen at this drawable capture,
    /// never the suffix which may already be staged for the following frame.
    pub upload: LodViewBlendUploadStats,
}

/// Attaches the exact pending-activation pressure oracle to an already
/// radix-proven authored endpoint publication. This is metadata only: the
/// publication generation, drawable weights, and frozen upload counters keep
/// describing the same GPU output.
#[cfg(any(test, feature = "testing"))]
fn attach_view_blend_preflight_evaluation_for_testing(
    publication: &mut LodViewBlendDrawablePublicationForTesting,
    evaluation_view: Option<LodView>,
    evaluation_target: Option<LodQualityTarget>,
) {
    let evaluation = evaluation_view.zip(evaluation_target);
    publication.evaluation_view = evaluation.map(|(view, _)| view);
    publication.evaluation_target = evaluation.map(|(_, target)| target);
    publication.desired_evaluation_complete = false;
}

/// Refreshes the oracle paired with an unchanged drawable suffix after every
/// edge has been checked bit-exact against the current Dynamic evaluation.
/// No new GPU output exists, so both publication generations and the captured
/// upload counters deliberately remain unchanged.
#[cfg(any(test, feature = "testing"))]
fn refresh_complete_view_blend_evaluation_for_testing(
    publication: &mut LodViewBlendDrawablePublicationForTesting,
    evaluation_view: Option<LodView>,
    evaluation_target: Option<LodQualityTarget>,
) {
    let Some((view, target)) = evaluation_view.zip(evaluation_target) else {
        return;
    };
    publication.evaluation_view = Some(view);
    publication.evaluation_target = Some(target);
    publication.desired_evaluation_complete = true;
}

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
    /// Compact per-transition direct parent map shared by compaction and the
    /// LoD raster pipeline. The allocation grows only when a larger bounded
    /// cohort appears; ordinary camera frames update only the O(edge-count)
    /// weight suffix.
    morph_buffer: Buffer,
    presentation_header: LodPresentationHeader,
    morph_word_capacity: u32,
    morph_identity: Option<LodViewBlendIdentity>,
    morph_weight_word_start: u32,
    morph_edge_states: Vec<LodViewBlendEdgeState>,
    #[cfg(any(test, feature = "testing"))]
    morph_edges_for_testing: Vec<LodViewBlendEdge>,
    morph_displayed_scratch: Vec<f32>,
    morph_drawable_identity: Option<LodViewBlendIdentity>,
    morph_drawable_sort_signature: Option<u64>,
    morph_drawable_snapshot: Option<LodDrawableViewBlendSnapshot>,
    /// Physical morph state paired with the last matching compaction/radix
    /// output. Unlike `morph_drawable_snapshot`, this is never retargeted by a
    /// newer Prepare evaluation before the GPU suffix itself is promoted.
    morph_radix_state: LodRadixMorphStateTracker,
    #[cfg(any(test, feature = "testing"))]
    morph_pending_evaluation_view: Option<LodView>,
    #[cfg(any(test, feature = "testing"))]
    morph_pending_evaluation_target: Option<LodQualityTarget>,
    /// True only when every live desired bit belongs to one complete checked
    /// selector evaluation. This is production activation state: authored
    /// endpoint coincidences must not make a pending Morphing table ACTIVE.
    morph_pending_evaluation_complete: bool,
    #[cfg(any(test, feature = "testing"))]
    morph_drawable_publication_generation: u64,
    #[cfg(any(test, feature = "testing"))]
    morph_drawable_publication_for_testing: Option<LodViewBlendDrawablePublicationForTesting>,
    morph_immutable_upload_count: u64,
    morph_weight_write_count: u64,
    morph_buffer_allocation_count: u64,
    morph_weight_bytes_written: u64,
    morph_last_max_delta: f32,
    morph_last_weighted_record_energy: f64,
    morph_lagging_edge_count: u32,
    /// Current Dynamic-view pressure failures for this private table. Invalid
    /// evaluation holds the complete suffix; the drawable count remains set
    /// until a later valid evaluation has itself passed compaction and radix.
    morph_invalid_pressure_edges: Vec<bool>,
    morph_drawable_invalid_pressure_edges: Vec<bool>,
    morph_selection_frozen: bool,
    /// Prepare-time raster/debug/pressure attestation for the exact candidate
    /// synchronized in this private retained view. Cleanup combines it with
    /// radix-promoted evaluation state before publishing ACTIVE.
    morph_activation_preflight_valid: bool,
    sorted_entry_bind_groups: [BindGroup; 2],
    config_buffer: Buffer,
    bind_group: Option<BindGroup>,
    compaction_layout: BindGroupLayout,
    sorted_layout: BindGroupLayout,
    candidate_evaluations_and_scan_records_bytes: u64,
    /// Base allocation charge including the minimum morph header. The resident
    /// charge replaces that header with the grow-only current table capacity.
    allocation_total_bytes: u64,
    config: LodCompactionUniform,
    readiness: LodCompactionReadiness,
    /// True only after candidate compaction and radix have produced a complete
    /// output in this allocation. Readiness alone may describe the initial
    /// identity configuration and is not a package draw capability.
    has_drawable_bridge_output: bool,
    candidate_descriptor_committed: bool,
    candidate_upload: LodCandidateUploadTracker,
    #[cfg(any(test, feature = "testing"))]
    radix_drawable: LodRadixDrawableTracker,
    /// Content epoch for exactly the physical slots described by the current
    /// bridge candidate. Unlike the atlas-wide upload revision, this remains
    /// stable while unrelated replacement slots are staged.
    candidate_content_signature: Option<u64>,
    /// Atlas-wide fast path for the per-slot signature above. Stable frames
    /// avoid scanning the descriptor ranges; any direct write rechecks only
    /// the slots referenced by this candidate.
    candidate_atlas_content_revision: Option<u64>,
    /// Physical storage allocation which produced the currently drawable
    /// indirect output. A ticket/slot generation can repeat across an actual
    /// buffer recreation, so this render-local epoch is a separate proof.
    candidate_atlas_allocation_epoch: Option<u64>,
    /// Candidate capability published only after the newly synchronized
    /// descriptor has produced one complete compaction and radix output.
    pending_bridge_activation: Option<Arc<AtomicU8>>,
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
        policy: LodCompactionPolicy,
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
            policy.quality_endpoint,
            policy.frustum_culling,
        );
        let config = config.with_policy(policy);
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
        let inactive_presentation_header = LodPresentationHeader::inactive();
        let morph_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_lod_morph_table"),
            contents: bytemuck::bytes_of(&inactive_presentation_header),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let bind_group = create_compaction_bind_group(
            render_device,
            &pipeline.layout,
            &config_buffer,
            &candidate_and_scan_buffer,
            &active_entries_buffer,
            &indirect_args_buffer,
            &morph_buffer,
        );
        let sorted_entry_bind_groups = [
            create_sorted_entry_bind_group(
                render_device,
                &pipeline.sorted_layout,
                &active_entries_buffer,
                &morph_buffer,
            ),
            create_sorted_entry_bind_group(
                render_device,
                &pipeline.sorted_layout,
                &radix_scratch_buffer,
                &morph_buffer,
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
            morph_buffer,
            presentation_header: inactive_presentation_header,
            morph_word_capacity: LOD_MORPH_HEADER_WORDS,
            morph_identity: None,
            morph_weight_word_start: LOD_MORPH_HEADER_WORDS,
            morph_edge_states: Vec::new(),
            #[cfg(any(test, feature = "testing"))]
            morph_edges_for_testing: Vec::new(),
            morph_displayed_scratch: Vec::new(),
            morph_drawable_identity: None,
            morph_drawable_sort_signature: None,
            morph_drawable_snapshot: None,
            morph_radix_state: LodRadixMorphStateTracker::default(),
            #[cfg(any(test, feature = "testing"))]
            morph_pending_evaluation_view: None,
            #[cfg(any(test, feature = "testing"))]
            morph_pending_evaluation_target: None,
            morph_pending_evaluation_complete: false,
            #[cfg(any(test, feature = "testing"))]
            morph_drawable_publication_generation: 0,
            #[cfg(any(test, feature = "testing"))]
            morph_drawable_publication_for_testing: None,
            morph_immutable_upload_count: 0,
            morph_weight_write_count: 0,
            morph_buffer_allocation_count: 1,
            morph_weight_bytes_written: 0,
            morph_last_max_delta: 0.0,
            morph_last_weighted_record_energy: 0.0,
            morph_lagging_edge_count: 0,
            morph_invalid_pressure_edges: Vec::new(),
            morph_drawable_invalid_pressure_edges: Vec::new(),
            // This records a Frozen frame which actually held an installed
            // blend table, not merely the view's creation-time setting.
            morph_selection_frozen: false,
            morph_activation_preflight_valid: false,
            sorted_entry_bind_groups,
            config_buffer,
            bind_group: Some(bind_group),
            compaction_layout: pipeline.layout.clone(),
            sorted_layout: pipeline.sorted_layout.clone(),
            candidate_evaluations_and_scan_records_bytes: allocation
                .candidate_evaluations_and_scan_records_bytes,
            allocation_total_bytes: allocation.total_bytes,
            config,
            readiness,
            has_drawable_bridge_output: false,
            candidate_descriptor_committed: false,
            candidate_upload: LodCandidateUploadTracker::default(),
            #[cfg(any(test, feature = "testing"))]
            radix_drawable: LodRadixDrawableTracker::default(),
            candidate_content_signature: None,
            candidate_atlas_content_revision: None,
            candidate_atlas_allocation_epoch: None,
            pending_bridge_activation: None,
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

    fn resident_admission_bytes(&self) -> u64 {
        lod_compaction_admission_bytes_with_morph(
            self.allocation_total_bytes,
            self.morph_word_capacity,
            self.morph_word_capacity,
        )
        .unwrap_or(u64::MAX)
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

    #[cfg(any(test, feature = "testing"))]
    pub fn view_blend_upload_stats_for_testing(&self) -> LodViewBlendUploadStats {
        self.view_blend_upload_stats_for_drawable(None)
    }

    #[cfg(any(test, feature = "testing"))]
    fn view_blend_upload_stats_for_drawable(
        &self,
        drawable: Option<&LodDrawableViewBlendSnapshot>,
    ) -> LodViewBlendUploadStats {
        let live = LodViewBlendUploadStats {
            immutable_table_upload_count: self.morph_immutable_upload_count,
            weight_write_count: self.morph_weight_write_count,
            buffer_allocation_count: self.morph_buffer_allocation_count,
            weight_bytes_written: self.morph_weight_bytes_written,
            edge_count: self.morph_edge_states.len().try_into().unwrap_or(u32::MAX),
            word_capacity: self.morph_word_capacity,
            lagging_edge_count: self.morph_lagging_edge_count,
            last_max_delta: self.morph_last_max_delta,
            last_weighted_record_energy: self.morph_last_weighted_record_energy,
            max_weight_delta_per_frame: LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME,
        };
        match drawable {
            Some(drawable) => lod_view_blend_upload_stats_for_drawable_snapshot(live, drawable),
            None => live,
        }
    }

    /// Coherent metadata for the exact drawable suffix last published to the
    /// candidate. Unlike [`Self::view_blend_upload_stats_for_testing`], this
    /// never observes a newer CPU weight write or evaluation view.
    #[cfg(any(test, feature = "testing"))]
    pub fn view_blend_drawable_publication_for_testing(
        &self,
    ) -> Option<LodViewBlendDrawablePublicationForTesting> {
        self.morph_drawable_publication_for_testing
    }

    /// Exact candidate/table state consumed by the last radix-promoted draw.
    ///
    /// This deliberately does not read `self.config` or the live morph table:
    /// both may already describe a PREPARED replacement. The match booleans
    /// compare that physical output with the currently extracted candidate
    /// without exposing an address-derived token as content identity.
    #[cfg(any(test, feature = "testing"))]
    pub fn last_radix_drawable_for_testing(
        &self,
        candidate: &LodRenderCandidate,
    ) -> Option<LodLastRadixDrawableForTesting> {
        let drawable = self.radix_drawable.drawable.as_ref()?;
        let candidate_fingerprint = lod_bridge_candidate_fingerprint(candidate);
        let candidate_token_matches = drawable
            .version
            .as_ref()
            .is_some_and(|version| Arc::ptr_eq(version, &candidate.phase));
        let candidate_content_matches = drawable.fingerprint == Some(candidate_fingerprint);
        Some(LodLastRadixDrawableForTesting {
            compaction_generation: self.generation,
            compute_input_generation: drawable.compute_input_generation,
            radix_publication_generation: self.radix_drawable.drawable_publication_generation,
            rendered_candidate_count: drawable.rendered_candidate_count,
            phase_at_compaction: drawable.phase_at_compaction,
            candidate_token_matches,
            candidate_content_matches,
            candidate_fingerprint_primary: drawable.fingerprint.map(|value| value.primary),
            candidate_fingerprint_secondary: drawable.fingerprint.map(|value| value.secondary),
            candidate_range_count: drawable.fingerprint.map(|value| value.range_count),
            candidate_content_signature: drawable.candidate_content_signature,
            candidate_atlas_allocation_epoch: drawable.candidate_atlas_allocation_epoch,
            morph_identity: drawable.morph_identity,
            view_blend: drawable.view_blend.clone(),
        })
    }

    fn resize_candidate_source_prefix(
        &mut self,
        render_device: &RenderDevice,
        required_words: u32,
    ) {
        let source_words = candidate_source_capacity_after_upload(
            self.config.candidate_source_word_capacity,
            required_words,
            self.config
                .output_capacity
                .checked_mul(LOD_PHYSICAL_RANGE_DESCRIPTOR_WORDS)
                .expect("validated LoD capacity has a representable descriptor prefix"),
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
            &self.morph_buffer,
        );
        self.candidate_and_scan_buffer = Some(candidate_and_scan_buffer);
        self.bind_group = Some(bind_group);
        self.config.candidate_source_word_capacity = source_words;
    }

    fn resize_morph_buffer(&mut self, render_device: &RenderDevice, required_words: u32) {
        if required_words <= self.morph_word_capacity {
            return;
        }
        let word_capacity = lod_morph_word_capacity(required_words)
            .expect("validated LoD morph capacity remains representable");
        let morph_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("gaussian_lod_morph_table"),
            size: u64::from(word_capacity) * std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.bind_group = Some(create_compaction_bind_group(
            render_device,
            &self.compaction_layout,
            &self.config_buffer,
            self.candidate_and_scan_buffer
                .as_ref()
                .expect("candidate binding exists while compaction state is live"),
            &self.active_entries_buffer,
            &self.indirect_args_buffer,
            &morph_buffer,
        ));
        self.sorted_entry_bind_groups = [
            create_sorted_entry_bind_group(
                render_device,
                &self.sorted_layout,
                &self.active_entries_buffer,
                &morph_buffer,
            ),
            create_sorted_entry_bind_group(
                render_device,
                &self.sorted_layout,
                &self.radix_scratch_buffer,
                &morph_buffer,
            ),
        ];
        self.morph_buffer = morph_buffer;
        self.morph_word_capacity = word_capacity;
        self.morph_buffer_allocation_count = self.morph_buffer_allocation_count.saturating_add(1);
    }

    fn deactivate_morph(&mut self, render_queue: &RenderQueue) {
        if self.morph_identity.is_none() {
            return;
        }
        let inactive_header = LodPresentationHeader::inactive();
        render_queue.write_buffer(&self.morph_buffer, 0, bytemuck::bytes_of(&inactive_header));
        self.presentation_header = inactive_header;
        self.morph_identity = None;
        self.morph_weight_word_start = LOD_MORPH_HEADER_WORDS;
        self.morph_edge_states.clear();
        #[cfg(any(test, feature = "testing"))]
        self.morph_edges_for_testing.clear();
        self.clear_drawable_view_blend_snapshot();
        self.morph_radix_state.clear();
        self.morph_last_max_delta = 0.0;
        self.morph_last_weighted_record_energy = 0.0;
        self.morph_lagging_edge_count = 0;
        self.morph_invalid_pressure_edges.clear();
        self.morph_drawable_invalid_pressure_edges.clear();
        self.morph_selection_frozen = false;
        self.morph_activation_preflight_valid = false;
        self.mark_compute_input_dirty();
    }

    /// Installs a new immutable topology/map table and its exact retained
    /// endpoint weights. Ordinary camera frames update only the compact weight
    /// suffix through [`Self::update_view_blend_weights`].
    /// A render-only capability veto requests a package-authored hard replan
    /// without touching the retained morph table or drawable output.
    fn synchronize_candidate_morph(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        candidate: &LodRenderCandidate,
        fallback_policy: LodCandidateHardFallbackPolicy,
    ) -> Result<LodCandidateMorphSynchronization, LodCandidateConfigError> {
        let limits = render_device.limits();
        let (morph, required_words) = match plan_lod_candidate_morph(
            candidate,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
        )? {
            LodCandidateMorphPlan::Disabled => {
                self.deactivate_morph(render_queue);
                return Ok(LodCandidateMorphSynchronization::Disabled);
            }
            LodCandidateMorphPlan::Unsupported => {
                let fallback = publish_lod_candidate_hard_fallback(candidate, fallback_policy);
                if fallback == LodCandidateMorphSynchronization::Disabled {
                    self.deactivate_morph(render_queue);
                }
                return Ok(fallback);
            }
            LodCandidateMorphPlan::Enabled {
                morph,
                required_words,
            } => (morph, required_words),
        };

        let identity = morph.identity();
        if self.morph_identity != Some(identity) {
            // A replacement batch may describe the same direction-independent
            // authored edge while its displayed weight is still in flight.
            // Reconcile from the last radix-proven state rather than a newer
            // staged CPU suffix. `initial_weight` is used only when no stable-key
            // drawable predecessor exists.
            let reconciliation_seed = self
                .morph_radix_state
                .reconciliation_seed(self.morph_identity, &self.morph_edge_states)?;
            let previous = reconciliation_seed
                .as_deref()
                .unwrap_or(&self.morph_edge_states);
            let edge_states = reconcile_lod_view_blend_edges(previous, morph)?;
            let words = build_lod_morph_words(morph, &edge_states)?;
            let presentation_header = LodPresentationHeader {
                descriptor_count: words[0],
                mapping_record_start: words[1],
                mapping_record_count: words[2],
                weight_start: words[3],
                weight_count: words[4],
                mode: words[5],
                first_weight_bits: words[6],
                second_weight_bits: words[7],
            };
            self.resize_morph_buffer(render_device, required_words);
            render_queue.write_buffer(&self.morph_buffer, 0, bytemuck::cast_slice(&words));
            self.presentation_header = presentation_header;
            self.morph_identity = Some(identity);
            self.morph_weight_word_start = lod_morph_weight_start(identity)?;
            self.morph_edge_states = edge_states;
            #[cfg(any(test, feature = "testing"))]
            {
                self.morph_edges_for_testing = morph.edges().to_vec();
            }
            self.clear_drawable_view_blend_snapshot();
            self.morph_last_max_delta = 0.0;
            self.morph_last_weighted_record_energy = 0.0;
            self.morph_lagging_edge_count =
                lod_view_blend_lagging_edge_count(&self.morph_edge_states);
            self.morph_invalid_pressure_edges = vec![false; self.morph_edge_states.len()];
            self.morph_drawable_invalid_pressure_edges = vec![false; self.morph_edge_states.len()];
            self.morph_activation_preflight_valid = false;
            self.morph_immutable_upload_count = self.morph_immutable_upload_count.saturating_add(1);
            self.mark_compute_input_dirty();
        }
        Ok(LodCandidateMorphSynchronization::Enabled)
    }

    fn clear_drawable_view_blend_snapshot(&mut self) {
        self.morph_drawable_identity = None;
        self.morph_drawable_sort_signature = None;
        self.morph_drawable_snapshot = None;
        self.morph_pending_evaluation_complete = false;
        #[cfg(any(test, feature = "testing"))]
        {
            self.morph_pending_evaluation_view = None;
            self.morph_pending_evaluation_target = None;
            self.morph_drawable_publication_for_testing = None;
        }
    }

    /// Captures the suffix generation whose compaction and radix signature is
    /// already current. A later CPU suffix update must not overwrite this
    /// evidence while the next private output is still being prepared.
    fn capture_drawable_view_blend_snapshot(&mut self) -> Result<(), LodCandidateConfigError> {
        let Some(identity) = self.morph_identity else {
            return Ok(());
        };
        let Some(sort_signature) = self
            .last_sorted_signature
            .filter(|_| self.radix_sort_is_current())
        else {
            return Ok(());
        };
        if self.morph_drawable_identity == Some(identity)
            && self.morph_drawable_sort_signature == Some(sort_signature)
        {
            return Ok(());
        }
        let Some(mut snapshot) = self.morph_radix_state.drawable_snapshot(identity)? else {
            return Ok(());
        };
        if self.morph_drawable_invalid_pressure_edges.len() != snapshot.invalid_pressure_edges.len()
        {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        snapshot
            .invalid_pressure_edges
            .clone_from(&self.morph_drawable_invalid_pressure_edges);
        #[cfg(any(test, feature = "testing"))]
        {
            self.morph_drawable_publication_generation =
                self.morph_drawable_publication_generation.saturating_add(1);
            self.morph_drawable_publication_for_testing =
                Some(LodViewBlendDrawablePublicationForTesting {
                    compaction_generation: self.generation,
                    publication_generation: self.morph_drawable_publication_generation,
                    evaluation_view: self.morph_pending_evaluation_view,
                    evaluation_target: self.morph_pending_evaluation_target,
                    desired_evaluation_complete: self.morph_pending_evaluation_complete,
                    upload: self.view_blend_upload_stats_for_drawable(Some(&snapshot)),
                });
        }
        self.morph_drawable_snapshot = Some(snapshot);
        self.morph_drawable_identity = Some(identity);
        self.morph_drawable_sort_signature = Some(sort_signature);
        Ok(())
    }

    /// Evaluates the complete live selector table before an authored or cached
    /// drawable is published. This mutates desired telemetry only: the
    /// displayed suffix remains bit-identical to the endpoint already proven
    /// by compaction and radix. A new ordinary edge still gets one exact
    /// authored first draw; common inherited and late-residency edges expose
    /// their checked current target immediately.
    fn prime_initial_recovery_view_blend_desired(
        &mut self,
        view: &ExtractedView,
        transform: &GlobalTransform,
        lod_settings: &GaussianLodSettings,
        candidate: &LodRenderCandidate,
    ) -> Result<(), LodCandidateConfigError> {
        if lod_settings.selection_mode == LodSelectionMode::Frozen {
            return Ok(());
        }
        let Some(identity) = self.morph_identity else {
            return Ok(());
        };
        let Some(morph) = candidate
            .temporal_transition()
            .and_then(|transition| transition.morph())
            .filter(|morph| morph.identity() == identity)
        else {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        };
        if morph.edges().len() != self.morph_edge_states.len() {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        let pressure_view = lod_view_blend_view(view, transform)
            .ok_or(LodCandidateConfigError::InvalidMorphWeight)?;
        let target = lod_settings.quality_target();
        // Evaluate the complete table before mutating any desired bit. A
        // single invalid edge must hold the authored endpoint table-wide;
        // partially priming earlier edges would make that held publication
        // internally inconsistent.
        let mut desired_weights = Vec::with_capacity(morph.edges().len());
        for (edge, state) in morph.edges().iter().zip(&self.morph_edge_states) {
            if !state.key.matches_edge(edge) {
                return Err(LodCandidateConfigError::MorphPayloadOverflow);
            }
            let Some(desired) = lod_view_blend_weight_checked(pressure_view, target, edge) else {
                return Ok(());
            };
            desired_weights.push(desired);
        }
        for (state, desired) in self
            .morph_edge_states
            .iter_mut()
            .zip(desired_weights.iter().copied())
        {
            retarget_checked_lod_view_blend_edge_desired(state, desired)?;
        }
        if self.morph_selection_frozen {
            mark_lod_view_blend_frozen_resume_recovery(&mut self.morph_edge_states);
        }
        // Priming can change desired telemetry for common or recovery edges
        // without moving the displayed suffix. Keep the live aggregate paired
        // with those exact bits before a same-frame compaction snapshots them.
        self.morph_lagging_edge_count = lod_view_blend_lagging_edge_count(&self.morph_edge_states);
        let desired_evaluation_complete =
            self.morph_edge_states
                .iter()
                .zip(&desired_weights)
                .all(|(state, desired)| {
                    state.desired_initialized && state.weight.desired.to_bits() == desired.to_bits()
                });
        self.morph_pending_evaluation_complete = desired_evaluation_complete;
        #[cfg(any(test, feature = "testing"))]
        {
            self.morph_pending_evaluation_view = Some(pressure_view);
            self.morph_pending_evaluation_target = Some(target);
        }
        self.morph_radix_state.refresh_drawable_evaluation(
            identity,
            &self.morph_edge_states,
            &self.morph_invalid_pressure_edges,
            desired_evaluation_complete,
        );
        #[cfg(any(test, feature = "testing"))]
        self.radix_drawable.refresh_checked_view_blend_evaluation(
            identity,
            &self.morph_edge_states,
            &self.morph_invalid_pressure_edges,
            Some(pressure_view),
            Some(target),
            desired_evaluation_complete,
        );
        // The suffix may already have been captured under this identity and
        // sort signature. Refresh only desired/lag telemetry; displayed bits,
        // endpoints, invalid mask, GPU bytes, upload counters, and publication
        // generation continue to describe the exact radix-proven output.
        if let Some(snapshot) = self.morph_drawable_snapshot.as_mut() {
            snapshot.retarget_pressure_targets(&self.morph_edge_states)?;
            #[cfg(any(test, feature = "testing"))]
            if let Some(publication) = self.morph_drawable_publication_for_testing.as_mut() {
                publication.evaluation_view = Some(pressure_view);
                publication.evaluation_target = Some(target);
                publication.desired_evaluation_complete = desired_evaluation_complete;
                publication.upload.edge_count =
                    snapshot.displayed.len().try_into().unwrap_or(u32::MAX);
                publication.upload.lagging_edge_count = snapshot.lagging_count();
                publication.upload.last_max_delta = snapshot.max_delta;
                publication.upload.last_weighted_record_energy = snapshot.weighted_record_energy;
            }
        }
        Ok(())
    }

    fn drawable_view_blend_snapshot(
        &self,
        identity: LodViewBlendIdentity,
    ) -> Option<&LodDrawableViewBlendSnapshot> {
        if self.morph_drawable_identity != Some(identity) {
            return None;
        }
        self.morph_drawable_snapshot.as_ref()
    }

    /// Returns the exact morph presentation promoted by the current frame's
    /// radix pass. The live invalid-pressure mask is an intentional overlay:
    /// an invalid current view holds the same physical suffix but must degrade
    /// package evidence immediately instead of waiting for a nonexistent sort.
    fn promoted_view_blend_snapshot(
        &self,
        candidate: &LodRenderCandidate,
    ) -> Result<Option<LodDrawableViewBlendSnapshot>, LodCandidateConfigError> {
        let Some(identity) = candidate
            .temporal_transition()
            .and_then(|transition| transition.morph())
            .map(LodViewBlendBatch::identity)
        else {
            return Ok(None);
        };
        if !self.has_current_drawable_bridge_candidate(candidate) {
            return Ok(None);
        }
        let Some(mut snapshot) = self.morph_radix_state.drawable_snapshot(identity)? else {
            return Ok(None);
        };
        if snapshot.invalid_pressure_edges.len() != self.morph_drawable_invalid_pressure_edges.len()
        {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        snapshot
            .invalid_pressure_edges
            .clone_from(&self.morph_drawable_invalid_pressure_edges);
        Ok(Some(snapshot))
    }

    fn morph_activation_allowed(
        &self,
        candidate: &LodRenderCandidate,
        selection_mode: LodSelectionMode,
    ) -> bool {
        let Some(identity) = candidate
            .temporal_transition()
            .and_then(|transition| transition.morph())
            .map(LodViewBlendBatch::identity)
        else {
            return false;
        };
        self.morph_activation_preflight_valid
            && self.has_current_drawable_bridge_candidate(candidate)
            && !self
                .morph_drawable_invalid_pressure_edges
                .iter()
                .any(|&invalid| invalid)
            && (selection_mode == LodSelectionMode::Frozen
                || self
                    .morph_radix_state
                    .drawable_evaluation_complete(identity))
    }

    /// Re-attests package-approved edge retirement against the private output
    /// that is actually drawable and the current render view. MainWorld may
    /// have authored the replacement from an older pipelined observation; a
    /// changed endpoint or live selector result must replan before any shared
    /// descriptor/table bytes are overwritten.
    fn view_blend_predecessor_attestation_is_current(
        &self,
        view: &ExtractedView,
        transform: &GlobalTransform,
        lod_settings: &GaussianLodSettings,
        candidate: &LodRenderCandidate,
    ) -> Result<bool, LodCandidateConfigError> {
        let Some(drawable_identity) = self.morph_radix_state.drawable_identity() else {
            // A cold candidate can have a live morph table installed while
            // its first compaction/radix generation is still in flight. There
            // is no drawable morph predecessor to retire in that window, so
            // requiring an attestation would turn ordinary staging into a
            // sticky package replan loop.
            return if missing_promoted_morph_predecessor_is_safe(
                self.has_drawable_bridge_output,
                self.morph_drawable_identity,
            ) {
                Ok(true)
            } else {
                // A captured morph drawable can only have been reconstructed
                // from this tracker. Losing the promoted identity while that
                // output remains live is invariant loss, not a cold handoff.
                Err(LodCandidateConfigError::MorphPayloadOverflow)
            };
        };
        let next_keys = (candidate.view_blend_mode() == Some(LodTemporalTransitionMode::Morphing))
            .then(|| {
                candidate
                    .temporal_transition()
                    .and_then(|transition| transition.morph())
            })
            .flatten()
            .map(|morph| {
                morph
                    .edges()
                    .iter()
                    .map(LodViewBlendEdgeKey::from_edge)
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        let Some(drawable_states) = self
            .morph_radix_state
            .reconciliation_seed(Some(drawable_identity), &self.morph_edge_states)?
        else {
            return Ok(false);
        };
        let Some(drawable_snapshot) = self
            .morph_radix_state
            .drawable_snapshot(drawable_identity)?
        else {
            return Ok(false);
        };
        if drawable_states.len() != drawable_snapshot.displayed.len()
            || drawable_states.len() != drawable_snapshot.recovery_edges.len()
            || drawable_states.len() != drawable_snapshot.invalid_pressure_edges.len()
        {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }
        let removed = drawable_states
            .iter()
            .enumerate()
            .filter(|(_, state)| !next_keys.contains(&state.key))
            .collect::<Vec<_>>();
        if removed.is_empty() {
            return Ok(true);
        }

        let Some(attestation) = candidate.predecessor_view_blend_attestation() else {
            return Ok(false);
        };
        if !candidate.predecessor_view_blend_attestation_is_current(Some(drawable_identity))
            || attestation.predecessor_identity() != drawable_identity
            || attestation.requirements().len() != removed.len()
        {
            return Ok(false);
        }
        let Some(pressure_view) = lod_view_blend_view(view, transform) else {
            return Ok(false);
        };
        let target = lod_settings.quality_target();
        let mut matched = HashSet::with_capacity(removed.len());
        for requirement in attestation.requirements() {
            let key = LodViewBlendEdgeKey::from_edge(requirement.edge());
            let Some((index, state)) = removed.iter().copied().find(|(_, state)| state.key == key)
            else {
                return Ok(false);
            };
            if !matched.insert(index) {
                return Ok(false);
            }
            let current_weight =
                lod_view_blend_weight_checked(pressure_view, target, requirement.edge());
            if !lod_view_blend_retirement_endpoint_is_current(
                drawable_snapshot.displayed[index],
                current_weight,
                drawable_snapshot.invalid_pressure_edges[index],
                requirement.endpoint(),
            ) {
                return Ok(false);
            }
            debug_assert_eq!(
                state.recovery_lag, drawable_snapshot.recovery_edges[index],
                "retirement key and recovery provenance must come from one radix promotion",
            );
        }
        Ok(matched.len() == removed.len())
    }

    /// Computes a complete table of current-camera desired weights before any
    /// edge state is mutated. One invalid edge holds the entire private suffix
    /// and output, because compaction and raster must consume one coherent
    /// table generation. The later valid evaluation inherits recovery slew.
    fn stage_view_blend_pressure_evaluation(
        &mut self,
        view: &ExtractedView,
        transform: &GlobalTransform,
        lod_settings: &GaussianLodSettings,
        candidate: &LodRenderCandidate,
    ) -> Result<LodViewBlendPressureEvaluation, LodCandidateConfigError> {
        if lod_settings.selection_mode == LodSelectionMode::Frozen {
            return Ok(LodViewBlendPressureEvaluation::Frozen);
        }
        let Some(identity) = self.morph_identity else {
            return Ok(LodViewBlendPressureEvaluation::Valid {
                recovered_from_invalid: false,
            });
        };
        let Some(morph) = candidate
            .temporal_transition()
            .and_then(|transition| transition.morph())
            .filter(|morph| morph.identity() == identity)
        else {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        };
        if morph.edges().len() != self.morph_edge_states.len() {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }

        let pressure_view = lod_view_blend_view(view, transform);
        let target = lod_settings.quality_target();
        let recovered_from_invalid = self
            .morph_invalid_pressure_edges
            .iter()
            .any(|&invalid| invalid);
        self.morph_displayed_scratch.clear();
        self.morph_displayed_scratch.reserve(morph.edges().len());
        self.morph_invalid_pressure_edges.clear();
        self.morph_invalid_pressure_edges
            .reserve(morph.edges().len());
        for (edge, state) in morph.edges().iter().zip(&self.morph_edge_states) {
            if !state.key.matches_edge(edge) {
                return Err(LodCandidateConfigError::MorphPayloadOverflow);
            }
            match pressure_view.and_then(|pressure_view| {
                lod_view_blend_weight_checked(pressure_view, target, edge)
            }) {
                Some(desired) => {
                    self.morph_displayed_scratch.push(desired);
                    self.morph_invalid_pressure_edges.push(false);
                }
                None => {
                    // Keep the scratch aligned even though no element may be
                    // consumed until every edge has evaluated successfully.
                    self.morph_displayed_scratch.push(state.weight.desired);
                    self.morph_invalid_pressure_edges.push(true);
                }
            }
        }

        let invalid_pressure = self
            .morph_invalid_pressure_edges
            .iter()
            .any(|&invalid| invalid);

        self.morph_pending_evaluation_complete = !invalid_pressure
            && self
                .morph_edge_states
                .iter()
                .zip(&self.morph_displayed_scratch)
                .all(|(state, desired)| {
                    state.desired_initialized
                        && !state.initial_drawable_pending
                        && state.weight.desired.to_bits() == desired.to_bits()
                });
        #[cfg(any(test, feature = "testing"))]
        {
            self.morph_pending_evaluation_view = pressure_view;
            self.morph_pending_evaluation_target = Some(target);
        }

        if invalid_pressure {
            // The radix-proven output remains byte-identical, but its current
            // camera evaluation is explicitly degraded immediately.
            self.morph_drawable_invalid_pressure_edges
                .clone_from(&self.morph_invalid_pressure_edges);
            hold_lod_view_blend_weights_for_invalid_pressure(&mut self.morph_edge_states);
            self.morph_last_max_delta = 0.0;
            self.morph_last_weighted_record_energy = 0.0;
            #[cfg(any(test, feature = "testing"))]
            if let Some(identity) = self.morph_identity {
                self.radix_drawable.refresh_checked_view_blend_evaluation(
                    identity,
                    &self.morph_edge_states,
                    &self.morph_invalid_pressure_edges,
                    pressure_view,
                    Some(target),
                    false,
                );
            }
            return Ok(LodViewBlendPressureEvaluation::Invalid);
        }

        Ok(LodViewBlendPressureEvaluation::Valid {
            recovered_from_invalid,
        })
    }

    /// Validates a drawable-but-not-yet-active table without moving its exact
    /// authored endpoint. Invalid pressure is published while the retained cut
    /// remains visible; a recovered table clears the diagnosis but reruns
    /// compaction/radix once before it may activate under the recovered view.
    fn preflight_view_blend_activation(
        &mut self,
        view: &ExtractedView,
        transform: &GlobalTransform,
        lod_settings: &GaussianLodSettings,
        candidate: &LodRenderCandidate,
    ) -> Result<bool, LodCandidateConfigError> {
        let evaluation =
            self.stage_view_blend_pressure_evaluation(view, transform, lod_settings, candidate)?;
        #[cfg(any(test, feature = "testing"))]
        if evaluation != LodViewBlendPressureEvaluation::Frozen
            && let Some(publication) = self.morph_drawable_publication_for_testing.as_mut()
        {
            attach_view_blend_preflight_evaluation_for_testing(
                publication,
                self.morph_pending_evaluation_view,
                self.morph_pending_evaluation_target,
            );
        }
        match evaluation {
            LodViewBlendPressureEvaluation::Frozen => Ok(!self
                .morph_drawable_invalid_pressure_edges
                .iter()
                .any(|&invalid| invalid)),
            LodViewBlendPressureEvaluation::Invalid => Ok(false),
            LodViewBlendPressureEvaluation::Valid {
                recovered_from_invalid,
            } => {
                if recovered_from_invalid {
                    // Clear the pressure diagnosis immediately, but require a
                    // fresh radix-proven generation before this pending table
                    // can activate under the recovered view.
                    self.morph_drawable_invalid_pressure_edges.fill(false);
                    self.mark_compute_input_dirty();
                    Ok(false)
                } else {
                    Ok(!self
                        .morph_drawable_invalid_pressure_edges
                        .iter()
                        .any(|&invalid| invalid))
                }
            }
        }
    }

    /// Clears a recovered pressure failure immediately while retaining the
    /// exact drawable displayed bits. Desired bits may advance independently
    /// as telemetry, making the exceptional recovery lag explicit until the
    /// bounded suffix update completes compaction and radix.
    fn recover_drawable_view_blend_pressure_targets(
        &mut self,
    ) -> Result<(), LodCandidateConfigError> {
        let Some(snapshot) = self.morph_drawable_snapshot.as_mut() else {
            return Ok(());
        };
        snapshot.recover_pressure_targets(&self.morph_edge_states)?;
        self.morph_drawable_invalid_pressure_edges.fill(false);
        #[cfg(any(test, feature = "testing"))]
        {
            let snapshot = snapshot.clone();
            self.morph_drawable_publication_generation =
                self.morph_drawable_publication_generation.saturating_add(1);
            self.morph_drawable_publication_for_testing =
                Some(LodViewBlendDrawablePublicationForTesting {
                    compaction_generation: self.generation,
                    publication_generation: self.morph_drawable_publication_generation,
                    evaluation_view: self.morph_pending_evaluation_view,
                    evaluation_target: self.morph_pending_evaluation_target,
                    desired_evaluation_complete: self.morph_pending_evaluation_complete,
                    upload: self.view_blend_upload_stats_for_drawable(Some(&snapshot)),
                });
        }
        Ok(())
    }

    fn update_view_blend_weights(
        &mut self,
        render_queue: &RenderQueue,
        view: &ExtractedView,
        transform: &GlobalTransform,
        lod_settings: &GaussianLodSettings,
        candidate: &LodRenderCandidate,
    ) -> Result<(), LodCandidateConfigError> {
        let Some(identity) = self.morph_identity else {
            return Ok(());
        };
        let Some(transition) = candidate.temporal_transition() else {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        };
        let Some(morph) = transition
            .morph()
            .filter(|morph| morph.identity() == identity)
        else {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        };
        if morph.edges().len() != self.morph_edge_states.len() {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }

        let frozen = lod_settings.selection_mode == LodSelectionMode::Frozen;
        let resumed_from_frozen = self.morph_selection_frozen && !frozen;
        self.morph_selection_frozen = frozen;
        #[cfg(any(test, feature = "testing"))]
        let authored_publication_pending = self
            .morph_edge_states
            .iter()
            .any(|state| state.initial_drawable_pending);
        let pressure_evaluation =
            self.stage_view_blend_pressure_evaluation(view, transform, lod_settings, candidate)?;
        #[cfg(any(test, feature = "testing"))]
        if authored_publication_pending
            && pressure_evaluation != LodViewBlendPressureEvaluation::Frozen
            && let Some(publication) = self.morph_drawable_publication_for_testing.as_mut()
        {
            attach_view_blend_preflight_evaluation_for_testing(
                publication,
                self.morph_pending_evaluation_view,
                self.morph_pending_evaluation_target,
            );
        }
        if pressure_evaluation == LodViewBlendPressureEvaluation::Invalid {
            // Keep both weight fields and the GPU suffix/output bit-exact. The
            // drawable invalid mask publishes explicit degradation and every
            // edge has been latched into bounded recovery for a later valid
            // view; this candidate remains ACTIVE.
            return Ok(());
        }
        let recovered_from_invalid = matches!(
            pressure_evaluation,
            LodViewBlendPressureEvaluation::Valid {
                recovered_from_invalid: true
            }
        );
        let desired_weights = &self.morph_displayed_scratch;
        if !frozen && desired_weights.len() != self.morph_edge_states.len() {
            return Err(LodCandidateConfigError::MorphPayloadOverflow);
        }

        let mut changed = false;
        let mut max_delta = 0.0_f32;
        let mut weighted_record_energy = 0.0_f64;
        let mut lagging_edges = 0_u32;
        for (index, (edge, state)) in morph
            .edges()
            .iter()
            .zip(&mut self.morph_edge_states)
            .enumerate()
        {
            if !state.key.matches_edge(edge) {
                return Err(LodCandidateConfigError::MorphPayloadOverflow);
            }
            let desired = (!frozen).then(|| desired_weights[index]);
            let previous = state.weight.displayed;
            changed |= update_lod_view_blend_edge_after_initial_draw(
                state,
                desired,
                resumed_from_frozen || recovered_from_invalid,
            )?;
            let delta = (state.weight.displayed - previous).abs();
            max_delta = max_delta.max(delta);
            weighted_record_energy += f64::from(delta) * f64::from(state.record_count);
            let lag = (state.weight.displayed - state.weight.desired).abs();
            if lag > 0.0 {
                lagging_edges = lagging_edges.saturating_add(1);
            }
        }
        self.morph_last_max_delta = max_delta;
        self.morph_last_weighted_record_energy = weighted_record_energy;
        self.morph_lagging_edge_count = lagging_edges;
        if !frozen {
            self.morph_pending_evaluation_complete = self
                .morph_edge_states
                .iter()
                .zip(desired_weights)
                .all(|(state, desired)| {
                    state.desired_initialized
                        && !state.initial_drawable_pending
                        && state.weight.desired.to_bits() == desired.to_bits()
                });
        }
        let unchanged_evaluation_is_complete = !changed
            && !frozen
            && !recovered_from_invalid
            && self
                .morph_edge_states
                .iter()
                .zip(desired_weights)
                .all(|(state, desired)| {
                    state.weight.displayed.to_bits() == desired.to_bits()
                        && state.weight.desired.to_bits() == desired.to_bits()
                        && !state.initial_drawable_pending
                });
        if unchanged_evaluation_is_complete {
            self.morph_pending_evaluation_complete = true;
            if let Some(identity) = self.morph_identity {
                self.morph_radix_state.refresh_drawable_evaluation(
                    identity,
                    &self.morph_edge_states,
                    &self.morph_invalid_pressure_edges,
                    true,
                );
            }
            #[cfg(any(test, feature = "testing"))]
            {
                if let Some(publication) = self.morph_drawable_publication_for_testing.as_mut() {
                    refresh_complete_view_blend_evaluation_for_testing(
                        publication,
                        self.morph_pending_evaluation_view,
                        self.morph_pending_evaluation_target,
                    );
                }
                if let Some(identity) = self.morph_identity {
                    self.radix_drawable.refresh_complete_view_blend_evaluation(
                        identity,
                        &self.morph_edge_states,
                        desired_weights,
                        self.morph_pending_evaluation_view,
                        self.morph_pending_evaluation_target,
                    );
                }
            }
        }
        if recovered_from_invalid {
            self.recover_drawable_view_blend_pressure_targets()?;
        }
        if !changed {
            if recovered_from_invalid {
                // Clearing invalid degradation is also a coherent drawable
                // publication. Even an already-exact held endpoint gets a new
                // compaction/radix generation paired with the recovered view.
                self.mark_compute_input_dirty();
            }
            return Ok(());
        }

        self.morph_displayed_scratch.clear();
        self.morph_displayed_scratch.extend(
            self.morph_edge_states
                .iter()
                .map(|state| state.weight.displayed),
        );
        let byte_offset = u64::from(self.morph_weight_word_start)
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or(LodCandidateConfigError::MorphPayloadOverflow)?;
        let bytes = bytemuck::cast_slice(&self.morph_displayed_scratch);
        render_queue.write_buffer(&self.morph_buffer, byte_offset, bytes);
        self.morph_weight_write_count = self.morph_weight_write_count.saturating_add(1);
        self.morph_weight_bytes_written = self
            .morph_weight_bytes_written
            .saturating_add(bytes.len() as u64);
        self.mark_compute_input_dirty();
        Ok(())
    }

    pub fn readiness(&self) -> LodCompactionReadiness {
        self.readiness
    }

    pub fn is_ready(&self) -> bool {
        self.readiness == LodCompactionReadiness::Ready
    }

    pub(crate) fn has_drawable_bridge_output(&self) -> bool {
        self.has_drawable_bridge_output
    }

    /// Whether this private retained-view output is already the complete
    /// radix-published result for the exact shared candidate token. This is
    /// stronger than general drawability: an older retained cut remains
    /// drawable while a replacement is being compacted, but cannot satisfy a
    /// multi-subview atomic activation barrier.
    fn has_current_drawable_bridge_candidate(&self, candidate: &LodRenderCandidate) -> bool {
        self.candidate_descriptor_committed
            && self.has_drawable_bridge_output
            && self.radix_sort_is_current()
            && self.candidate_upload.plan(candidate) == LodCandidateUploadPlan::ReuseVersion
    }

    /// True after both compaction and active-radix variants have compiled.
    pub(crate) fn pipelines_ready(&self) -> bool {
        self.pipelines_ready
    }

    /// Pending candidates own a complete validated list and may have their
    /// dependent radix bind groups prepared before execution becomes Ready.
    pub(crate) fn has_staged_candidates(&self) -> bool {
        self.readiness != LodCompactionReadiness::AwaitingCandidates
    }

    /// Returns this state to the complete legacy draw path until a new
    /// identity or candidate frontier is explicitly committed.
    pub fn invalidate_candidates(&mut self, render_queue: &RenderQueue) {
        // Shrinking is synchronized with state destruction/recreation. Keeping
        // this capacity also preserves the cached payload when the exact same
        // frontier is later reactivated by fingerprint.
        self.candidate_ownership = LodCandidateOwnership::Bridge;
        self.readiness = LodCompactionReadiness::AwaitingCandidates;
        self.has_drawable_bridge_output = false;
        self.candidate_descriptor_committed = false;
        self.candidate_content_signature = None;
        self.candidate_atlas_content_revision = None;
        self.candidate_atlas_allocation_epoch = None;
        self.pending_bridge_activation = None;
        // Descriptor flags and the table header form one GPU capability. Clear
        // both sides before allowing an identical-looking hard-cut candidate
        // to reuse this state; otherwise refinement ranges can preserve bit 28
        // and address a stale parent table after an adapter/capacity downgrade.
        let inactive_header = LodPresentationHeader::inactive();
        render_queue.write_buffer(&self.morph_buffer, 0, bytemuck::bytes_of(&inactive_header));
        self.presentation_header = inactive_header;
        self.morph_identity = None;
        self.morph_weight_word_start = LOD_MORPH_HEADER_WORDS;
        self.morph_edge_states.clear();
        #[cfg(any(test, feature = "testing"))]
        self.morph_edges_for_testing.clear();
        self.clear_drawable_view_blend_snapshot();
        self.morph_radix_state.clear();
        self.morph_last_max_delta = 0.0;
        self.morph_last_weighted_record_energy = 0.0;
        self.morph_lagging_edge_count = 0;
        self.morph_invalid_pressure_edges.clear();
        self.morph_drawable_invalid_pressure_edges.clear();
        self.morph_selection_frozen = false;
        self.morph_activation_preflight_valid = false;
        self.candidate_upload = LodCandidateUploadTracker::default();
        #[cfg(any(test, feature = "testing"))]
        self.radix_drawable.clear();
        self.mark_compute_input_dirty();
    }

    fn retain_pending_activation_for(&mut self, candidate: &LodRenderCandidate) {
        if self
            .pending_bridge_activation
            .as_ref()
            .is_some_and(|phase| !Arc::ptr_eq(phase, &candidate.phase))
        {
            self.pending_bridge_activation = None;
        }
    }

    fn arm_bridge_activation(&mut self, candidate: &LodRenderCandidate) {
        debug_assert!(
            self.morph_identity.is_none(),
            "Morphing activation is published only after Cleanup aggregates every retained view"
        );
        if bridge_activation_can_publish_immediately(
            self.candidate_descriptor_committed,
            self.has_drawable_bridge_output,
            self.radix_sort_is_current(),
        ) {
            self.pending_bridge_activation = None;
            self.publish_candidate_phase_after_radix(&candidate.phase);
        } else {
            self.pending_bridge_activation = Some(Arc::clone(&candidate.phase));
        }
    }

    fn publish_candidate_phase_after_radix(&mut self, phase: &Arc<AtomicU8>) -> bool {
        debug_assert!(self.morph_identity.is_none());
        publish_bridge_activation_after_radix(phase)
    }

    fn defer_bridge_activation_for(&mut self, candidate: &LodRenderCandidate) {
        if self
            .pending_bridge_activation
            .as_ref()
            .is_some_and(|phase| Arc::ptr_eq(phase, &candidate.phase))
        {
            self.pending_bridge_activation = None;
        }
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
            self.candidate_content_signature = None;
            self.candidate_atlas_content_revision = None;
            self.candidate_atlas_allocation_epoch = None;
            self.readiness = self.readiness.after_candidate_commit(self.pipelines_ready);
            self.candidate_descriptor_committed = true;
            self.candidate_ownership = LodCandidateOwnership::Bridge;
            return Ok(());
        }
        self.upload_candidate_frontier_data(render_device, render_queue, frontier)?;
        self.candidate_upload.mark_unversioned(fingerprint);
        self.candidate_content_signature = None;
        self.candidate_atlas_content_revision = None;
        self.candidate_atlas_allocation_epoch = None;
        self.candidate_descriptor_committed = true;
        self.candidate_ownership = LodCandidateOwnership::Bridge;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn synchronize_bridge_candidate_frontier(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        candidate: &LodRenderCandidate,
        fallback_policy: LodCandidateHardFallbackPolicy,
        atlas_allocation_epoch: Option<u64>,
        atlas_content_revision: u64,
        content_signature: impl FnOnce() -> u64,
    ) -> Result<LodCandidateMorphSynchronization, LodCandidateConfigError> {
        let morph_synchronization = self.synchronize_candidate_morph(
            render_device,
            render_queue,
            candidate,
            fallback_policy,
        )?;
        if morph_synchronization == LodCandidateMorphSynchronization::HardFallbackRequested {
            return Ok(morph_synchronization);
        }
        let morph_enabled = morph_synchronization == LodCandidateMorphSynchronization::Enabled;
        let plan = self.candidate_upload.plan(candidate);
        let content_signature = if candidate_content_signature_must_refresh(
            plan,
            self.candidate_atlas_content_revision,
            atlas_content_revision,
            self.candidate_content_signature,
        ) {
            content_signature()
        } else {
            self.candidate_content_signature
                .expect("a stable atlas revision has a committed slot signature")
        };
        match plan {
            LodCandidateUploadPlan::ReuseVersion => {
                self.readiness = self.readiness.after_candidate_commit(self.pipelines_ready);
                if candidate_content_signature_changed(
                    self.candidate_content_signature,
                    content_signature,
                ) {
                    // The descriptor is unchanged, but one of its committed
                    // physical slots was rewritten in place.
                    self.mark_compute_input_dirty();
                }
            }
            LodCandidateUploadPlan::ReuseFingerprint(fingerprint) => {
                self.candidate_upload
                    .mark_synchronized(&candidate.phase, fingerprint);
                self.readiness = self.readiness.after_candidate_commit(self.pipelines_ready);
                // The descriptor bytes are identical, but this new Arc is a
                // distinct publication token. Force one compaction/radix
                // completion so it cannot remain PREPARED forever behind the
                // cached-signature early exits.
                self.mark_compute_input_dirty();
            }
            LodCandidateUploadPlan::Upload(fingerprint) => {
                self.upload_bridge_candidate_data(
                    render_device,
                    render_queue,
                    candidate,
                    morph_enabled,
                )?;
                self.candidate_upload
                    .mark_synchronized(&candidate.phase, fingerprint);
            }
        }
        debug_assert!(!plan.requires_recompute() || self.last_compaction_signature.is_none());
        self.candidate_content_signature = Some(content_signature);
        self.candidate_atlas_content_revision = Some(atlas_content_revision);
        self.candidate_atlas_allocation_epoch = atlas_allocation_epoch;
        self.candidate_descriptor_committed = true;
        self.candidate_ownership = LodCandidateOwnership::Bridge;
        Ok(morph_synchronization)
    }

    #[allow(clippy::too_many_arguments)]
    fn synchronize_bridge_external_active_set(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        candidate: &LodRenderCandidate,
        presentation: &LodExternalActiveSetPresentation,
        first_weight: f32,
        second_weight: f32,
        catalog_allocation_epoch: u64,
        catalog_content_revision: u64,
        content_signature: impl FnOnce() -> u64,
    ) -> Result<(), LodCandidateConfigError> {
        // Header and descriptor admission are independently fallible. Validate
        // both before mutating a retained presentation, including on the
        // same-content/new-token reuse path.
        LodPresentationHeader::external_active_set(first_weight, second_weight)?;
        self.validate_bridge_external_active_set(candidate, presentation)?;

        let plan = self.candidate_upload.plan(candidate);
        let content_signature = if candidate_content_signature_must_refresh(
            plan,
            self.candidate_atlas_content_revision,
            catalog_content_revision,
            self.candidate_content_signature,
        ) {
            content_signature()
        } else {
            self.candidate_content_signature
                .expect("a stable resident catalog revision has a committed signature")
        };

        match plan {
            LodCandidateUploadPlan::ReuseVersion => {
                // The presentation header is raster-only. Camera motion inside
                // one immutable pair therefore keeps the compacted/sorted union
                // current unless the resident catalog itself changed.
                self.update_external_active_set_weights(render_queue, first_weight, second_weight)?;
                self.readiness = self.readiness.after_candidate_commit(self.pipelines_ready);
                if candidate_content_signature_changed(
                    self.candidate_content_signature,
                    content_signature,
                ) {
                    self.mark_compute_input_dirty();
                }
            }
            LodCandidateUploadPlan::ReuseFingerprint(fingerprint) => {
                self.update_external_active_set_weights(render_queue, first_weight, second_weight)?;
                self.candidate_upload
                    .mark_synchronized(&candidate.phase, fingerprint);
                self.readiness = self.readiness.after_candidate_commit(self.pipelines_ready);
                // A new publication token needs a radix-latched ownership proof
                // even when its pair, classes, and physical ranges are equal.
                self.mark_compute_input_dirty();
            }
            LodCandidateUploadPlan::Upload(fingerprint) => {
                self.install_external_active_set_candidate(
                    render_device,
                    render_queue,
                    candidate.rendered_candidate_count(),
                    candidate.render_ranges(),
                    presentation.range_classes(),
                    first_weight,
                    second_weight,
                )?;
                self.candidate_upload
                    .mark_synchronized(&candidate.phase, fingerprint);
            }
        }
        debug_assert!(!plan.requires_recompute() || self.last_compaction_signature.is_none());
        self.candidate_content_signature = Some(content_signature);
        self.candidate_atlas_content_revision = Some(catalog_content_revision);
        self.candidate_atlas_allocation_epoch = Some(catalog_allocation_epoch);
        self.candidate_descriptor_committed = true;
        self.candidate_ownership = LodCandidateOwnership::Bridge;
        Ok(())
    }

    fn validate_bridge_external_active_set(
        &self,
        candidate: &LodRenderCandidate,
        presentation: &LodExternalActiveSetPresentation,
    ) -> Result<(), LodCandidateConfigError> {
        let (descriptors, candidate_count) = build_gpu_external_active_set_range_descriptors(
            candidate.render_ranges(),
            presentation.range_classes(),
            self.config.source_count,
        )?;
        if candidate.rendered_candidate_count() != candidate_count {
            return Err(LodCandidateConfigError::CandidateCountMismatch {
                declared: candidate.rendered_candidate_count(),
                actual: candidate_count,
            });
        }
        let descriptor_count = u32::try_from(descriptors.len()).map_err(|_| {
            LodCandidateConfigError::PhysicalRangeCountNotRepresentable {
                range_count: descriptors.len(),
            }
        })?;
        self.config
            .with_physical_ranges(candidate_count, descriptor_count)?;
        Ok(())
    }

    fn validate_bridge_candidate_presentation(
        &self,
        candidate: &LodRenderCandidate,
    ) -> Result<(), LodCandidateConfigError> {
        if let Some(presentation) = candidate.external_active_set() {
            self.validate_bridge_external_active_set(candidate, presentation)
        } else {
            self.validate_bridge_candidate_frontier(candidate)
        }
    }

    fn validate_bridge_candidate_frontier(
        &self,
        candidate: &LodRenderCandidate,
    ) -> Result<(), LodCandidateConfigError> {
        let (descriptors, candidate_count) = build_gpu_physical_range_descriptors_with_classes(
            candidate.render_ranges(),
            self.config.source_count,
            |node| candidate_residency_code(candidate.frontier(), node),
            |_, range| {
                if candidate_range_is_morphed(candidate, range) {
                    LodExternalActiveSetClass::FirstOnly
                } else {
                    LodExternalActiveSetClass::Shared
                }
            },
        )?;
        if candidate.rendered_candidate_count() != candidate_count {
            return Err(LodCandidateConfigError::CandidateCountMismatch {
                declared: candidate.rendered_candidate_count(),
                actual: candidate_count,
            });
        }
        let descriptor_count = u32::try_from(descriptors.len()).map_err(|_| {
            LodCandidateConfigError::PhysicalRangeCountNotRepresentable {
                range_count: descriptors.len(),
            }
        })?;
        self.config
            .with_physical_ranges(candidate_count, descriptor_count)?;
        Ok(())
    }

    fn upload_bridge_candidate_data(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        candidate: &LodRenderCandidate,
        morph_enabled: bool,
    ) -> Result<(), LodCandidateConfigError> {
        let (descriptors, range_candidate_count) = build_bridge_candidate_upload_descriptors(
            candidate,
            self.config.source_count,
            morph_enabled,
        )?;
        self.upload_candidate_descriptors(
            render_device,
            render_queue,
            if morph_enabled {
                candidate.rendered_candidate_count()
            } else {
                candidate.frontier().candidate_count()
            },
            range_candidate_count,
            descriptors,
        )
    }

    fn upload_candidate_frontier_data(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        frontier: &LodCandidateFrontier,
    ) -> Result<(), LodCandidateConfigError> {
        let (descriptors, range_candidate_count) =
            build_gpu_physical_range_descriptors_with_residency(
                frontier.physical_ranges(),
                self.config.source_count,
                |node| candidate_residency_code(frontier, node),
            )?;
        self.upload_candidate_descriptors(
            render_device,
            render_queue,
            frontier.candidate_count(),
            range_candidate_count,
            descriptors,
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
        self.upload_candidate_descriptors(
            render_device,
            render_queue,
            candidate_count,
            range_candidate_count,
            descriptors,
        )
    }

    fn upload_candidate_descriptors(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        candidate_count: u32,
        range_candidate_count: u32,
        descriptors: Vec<LodGpuPhysicalRangeDescriptor>,
    ) -> Result<(), LodCandidateConfigError> {
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
        self.readiness = self.readiness.after_candidate_commit(self.pipelines_ready);
        self.mark_compute_input_dirty();
        Ok(())
    }

    /// Updates only the shared presentation header for an already-installed
    /// external two-set union. Exact host f32 bits are written without
    /// normalizing either weight. The header is consumed only by rasterization:
    /// a camera-only weight change deliberately preserves the compacted and
    /// radix-sorted union. Pair, membership, or catalog-content changes still
    /// take the descriptor installation path and invalidate that proof.
    pub fn update_external_active_set_weights(
        &mut self,
        render_queue: &RenderQueue,
        first_weight: f32,
        second_weight: f32,
    ) -> Result<bool, LodCandidateConfigError> {
        let header = LodPresentationHeader::external_active_set(first_weight, second_weight)?;
        if self.morph_identity.is_none() && self.presentation_header == header {
            return Ok(false);
        }
        if self.morph_identity.is_some() {
            self.deactivate_morph(render_queue);
        }
        render_queue.write_buffer(&self.morph_buffer, 0, bytemuck::bytes_of(&header));
        self.presentation_header = header;
        Ok(true)
    }

    /// Installs one complete, already-deduplicated external active-set union
    /// through the same physical-range compaction and radix path as hierarchy
    /// candidates. This is the render primitive only: package ownership,
    /// residency leases, and candidate-token activation remain the caller's
    /// responsibility.
    #[allow(clippy::too_many_arguments)]
    pub fn install_external_active_set_candidate(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        declared_candidate_count: u32,
        physical_ranges: &[LodPhysicalRange],
        classes: &[LodgeMembershipClass],
        first_weight: f32,
        second_weight: f32,
    ) -> Result<(), LodCandidateConfigError> {
        let header = LodPresentationHeader::external_active_set(first_weight, second_weight)?;
        let (descriptors, range_candidate_count) = build_gpu_external_active_set_range_descriptors(
            physical_ranges,
            classes,
            self.config.source_count,
        )?;
        if declared_candidate_count != range_candidate_count {
            return Err(LodCandidateConfigError::CandidateCountMismatch {
                declared: declared_candidate_count,
                actual: range_candidate_count,
            });
        }
        let descriptor_count = u32::try_from(descriptors.len()).map_err(|_| {
            LodCandidateConfigError::PhysicalRangeCountNotRepresentable {
                range_count: descriptors.len(),
            }
        })?;
        descriptor_count
            .checked_mul(LOD_PHYSICAL_RANGE_DESCRIPTOR_WORDS)
            .ok_or(LodCandidateConfigError::PhysicalRangeCountOverflow)?;
        // Complete all fallible admission checks before changing either side
        // of the mode-qualified descriptor/header capability.
        self.config
            .with_physical_ranges(declared_candidate_count, descriptor_count)?;

        if self.morph_identity.is_some() {
            self.deactivate_morph(render_queue);
        }
        if self.presentation_header != header {
            render_queue.write_buffer(&self.morph_buffer, 0, bytemuck::bytes_of(&header));
            self.presentation_header = header;
        }
        self.upload_candidate_descriptors(
            render_device,
            render_queue,
            declared_candidate_count,
            range_candidate_count,
            descriptors,
        )?;
        // A raw external installation supersedes any hierarchy token and
        // content proof. Candidate extraction will attach its own identity
        // before activation in the integration patch.
        self.candidate_upload = LodCandidateUploadTracker::default();
        self.candidate_content_signature = None;
        self.candidate_atlas_content_revision = None;
        self.candidate_atlas_allocation_epoch = None;
        self.candidate_descriptor_committed = true;
        self.candidate_ownership = LodCandidateOwnership::Bridge;
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
        self.candidate_content_signature = None;
        self.candidate_atlas_content_revision = None;
        self.candidate_descriptor_committed = true;
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

    fn set_policy(&mut self, render_queue: &RenderQueue, policy: LodCompactionPolicy) {
        let mut next = self.config;
        next.set_policy_fields(policy);
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

    #[cfg(any(test, feature = "testing"))]
    fn view_blend_for_pending_radix_for_testing(&self) -> Option<LodLastRadixViewBlendForTesting> {
        let identity = self.morph_identity?;
        if self.morph_edges_for_testing.len() != self.morph_edge_states.len()
            || self.morph_invalid_pressure_edges.len() != self.morph_edge_states.len()
        {
            return None;
        }
        let weights = self
            .morph_edge_states
            .iter()
            .map(|state| LodViewBlendWeightSnapshot {
                displayed: state.weight.displayed,
                desired: state.weight.desired,
            })
            .collect::<Vec<_>>();
        let exact_lagging_edge_count = weights
            .iter()
            .filter(|weight| weight.displayed.to_bits() != weight.desired.to_bits())
            .count()
            .try_into()
            .unwrap_or(u32::MAX);
        let mut upload = self.view_blend_upload_stats_for_testing();
        debug_assert_eq!(
            upload.lagging_edge_count, exact_lagging_edge_count,
            "live view-blend lag count must match the weights latched for radix"
        );
        // Derive the published aggregate from the exact vector even in release
        // builds so a future live-cache regression cannot create torn testing
        // evidence.
        upload.lagging_edge_count = exact_lagging_edge_count;
        let endpoints = self
            .morph_edge_states
            .iter()
            .map(|state| {
                if state.weight.displayed.to_bits() == 0.0_f32.to_bits() {
                    LodViewBlendEndpoint::ParentExact
                } else if state.weight.displayed.to_bits() == 1.0_f32.to_bits() {
                    LodViewBlendEndpoint::ChildrenExact
                } else {
                    LodViewBlendEndpoint::Fractional
                }
            })
            .collect();
        Some(LodLastRadixViewBlendForTesting {
            identity,
            edges: self.morph_edges_for_testing.clone(),
            weights,
            endpoints,
            recovery_lag: self
                .morph_edge_states
                .iter()
                .map(|state| state.recovery_lag)
                .collect(),
            invalid_pressure: self.morph_invalid_pressure_edges.clone(),
            evaluation_view: self.morph_pending_evaluation_view,
            evaluation_target: self.morph_pending_evaluation_target,
            desired_evaluation_complete: self.morph_pending_evaluation_complete,
            upload,
        })
    }

    fn mark_compute_input_dirty(&mut self) {
        self.compute_input_generation = self.compute_input_generation.wrapping_add(1).max(1);
        self.last_compaction_signature = None;
        self.pending_sort_signature = None;
        self.morph_radix_state.discard_pending();
        #[cfg(any(test, feature = "testing"))]
        self.radix_drawable.discard_pending();
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
        let morph_latched = self.candidate_descriptor_committed
            && self.morph_identity.is_some_and(|identity| {
                self.morph_radix_state.latch_compacted(
                    identity,
                    signature,
                    &self.morph_edge_states,
                    &self.morph_invalid_pressure_edges,
                    self.morph_pending_evaluation_complete,
                    self.morph_last_max_delta,
                    self.morph_last_weighted_record_energy,
                )
            });
        if !morph_latched {
            self.morph_radix_state.discard_pending();
        }
        #[cfg(any(test, feature = "testing"))]
        if self.candidate_descriptor_committed {
            let version = self.candidate_upload.version.clone();
            let snapshot = LodRadixCandidateSnapshot {
                phase_at_compaction: version.as_ref().map(|phase| phase.load(Ordering::Acquire)),
                version,
                fingerprint: self.candidate_upload.fingerprint,
                candidate_content_signature: self.candidate_content_signature,
                candidate_atlas_allocation_epoch: self.candidate_atlas_allocation_epoch,
                rendered_candidate_count: self.config.candidate_count,
                morph_identity: self.morph_identity,
                compute_input_generation: self.compute_input_generation,
                compaction_signature: signature,
                view_blend: self.view_blend_for_pending_radix_for_testing(),
            };
            self.radix_drawable.latch_compacted(snapshot);
        } else {
            self.radix_drawable.discard_pending();
        }
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
            let morph_promoted = self.morph_radix_state.promote(signature);
            #[cfg(any(test, feature = "testing"))]
            let candidate_metadata_promoted = self.radix_drawable.promote(signature);
            clear_lod_view_blend_frame_energy_after_promotion(
                morph_promoted,
                &mut self.morph_last_max_delta,
                &mut self.morph_last_weighted_record_energy,
            );
            if self.candidate_descriptor_committed {
                debug_assert_eq!(
                    morph_promoted,
                    self.morph_identity.is_some(),
                    "a committed morph candidate radix output must promote its compacted state"
                );
                #[cfg(any(test, feature = "testing"))]
                debug_assert!(
                    candidate_metadata_promoted,
                    "a committed candidate radix output must promote its compacted metadata"
                );
                self.has_drawable_bridge_output = true;
                if let Some(phase) = self.pending_bridge_activation.take() {
                    // A Morphing candidate is not an ACTIVE capability until
                    // Cleanup has reduced and Release-published the exact
                    // promoted state from every private retained view. Hard
                    // and categorical candidates have no shared weight table
                    // and may retain direct radix activation.
                    if self.morph_identity.is_none() {
                        self.publish_candidate_phase_after_radix(&phase);
                    }
                }
            }
        }
    }
}

/// A promoted morph snapshot owns the just-finished drawable frame's event
/// energy. Clear the live accumulator only after radix promotion, so a later
/// sort-only compaction publishes zero instead of replaying that prior work.
/// A discarded/mismatched pending sort must leave the live values intact for
/// the suffix which has not yet become drawable.
fn clear_lod_view_blend_frame_energy_after_promotion(
    promoted: bool,
    max_delta: &mut f32,
    weighted_record_energy: &mut f64,
) {
    if promoted {
        *max_delta = 0.0;
        *weighted_record_energy = 0.0;
    }
}

pub(crate) fn publish_bridge_activation_after_radix(phase: &AtomicU8) -> bool {
    phase
        .compare_exchange(
            LOD_RENDER_PREPARED,
            LOD_RENDER_ACTIVE,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
}

const fn bridge_activation_can_publish_immediately(
    candidate_descriptor_committed: bool,
    has_drawable_bridge_output: bool,
    radix_sort_is_current: bool,
) -> bool {
    candidate_descriptor_committed && has_drawable_bridge_output && radix_sort_is_current
}

const fn lod_compaction_cache_allowed(has_interpolate: bool, has_particles: bool) -> bool {
    !has_interpolate && !has_particles
}

fn candidate_content_signature_changed(previous: Option<u64>, next: u64) -> bool {
    previous != Some(next)
}

fn candidate_content_signature_must_refresh(
    plan: LodCandidateUploadPlan,
    previous_atlas_revision: Option<u64>,
    atlas_revision: u64,
    previous_content_signature: Option<u64>,
) -> bool {
    !matches!(plan, LodCandidateUploadPlan::ReuseVersion)
        || previous_atlas_revision != Some(atlas_revision)
        || previous_content_signature.is_none()
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
    morph: &Buffer,
) -> BindGroup {
    render_device.create_bind_group(
        "gaussian_lod_sorted_entries",
        layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: entries,
                    offset: 0,
                    size: BufferSize::new(entries.size()),
                }),
            },
            BindGroupEntry {
                binding: 1,
                resource: morph.as_entire_binding(),
            },
        ],
    )
}

fn create_compaction_bind_group(
    render_device: &RenderDevice,
    layout: &BindGroupLayout,
    config: &Buffer,
    candidate_indices: &Buffer,
    active_entries: &Buffer,
    indirect_args: &Buffer,
    morph: &Buffer,
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
            BindGroupEntry {
                binding: 4,
                resource: morph.as_entire_binding(),
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
    /// Whether any view for `entity` will replace its retained output at the
    /// next completed compaction/radix pass.
    ///
    /// LoD debug annotations are committed in the main world after that render
    /// activation becomes observable. The draw path uses this signal to avoid
    /// binding the retained cut's annotation epoch to the replacement output in
    /// the activation frame.
    pub(crate) fn has_pending_bridge_activation(&self, entity: Entity) -> bool {
        self.entries.iter().any(|((_, cloud, _), state)| {
            *cloud == entity
                && (state.pending_bridge_activation.is_some()
                    || state.morph_activation_preflight_valid)
        })
    }

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
            .filter(|state| state.is_ready() && state.has_drawable_bridge_output())
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
    storage_buffers_per_shader_stage: u32,
    variants: HashMap<(GaussianMode, RadixSortDepthBits), LodCompactionPipelines>,
    marker: PhantomData<R>,
}

fn lod_compute_storage_buffer_count(layouts: &[BindGroupLayoutDescriptor]) -> u32 {
    layouts
        .iter()
        .flat_map(|layout| &layout.entries)
        .filter(|entry| {
            entry.visibility.contains(ShaderStages::COMPUTE)
                && matches!(
                    entry.ty,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { .. },
                        ..
                    }
                )
        })
        .count()
        .try_into()
        .unwrap_or(u32::MAX)
}

const fn lod_storage_buffer_count_is_supported(required: u32, available: u32) -> bool {
    required <= available
}

fn lod_compaction_layout_entries() -> [BindGroupLayoutEntry; 5] {
    [
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
        BindGroupLayoutEntry {
            binding: 4,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(LOD_MORPH_MIN_BUFFER_BYTES),
            },
            count: None,
        },
    ]
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
                lod_candidate: true,
                ..default()
            },
            ShaderDefines::for_radix_depth_bits(radix_depth_bits),
        );
        let mut shader_defs = shader_defs;
        shader_defs.push("LOD_MORPH_COMPACTION".into());
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
        let entries = lod_compaction_layout_entries();
        let layout_descriptor =
            BindGroupLayoutDescriptor::new("gaussian_lod_compaction_layout", &entries);
        let layout = render_device
            .create_bind_group_layout(Some("gaussian_lod_compaction_layout"), &entries);
        let pipeline_layout = vec![
            cloud_pipeline.compute_view_layout_desc.clone(),
            cloud_pipeline.gaussian_uniform_layout_desc.clone(),
            cloud_pipeline.gaussian_cloud_layout_desc.clone(),
            layout_descriptor,
        ];
        let storage_buffers_per_shader_stage = lod_compute_storage_buffer_count(&pipeline_layout);
        Self {
            layout,
            sorted_layout: cloud_pipeline.lod_sorted_layout.clone(),
            pipeline_layout,
            storage_buffers_per_shader_stage,
            variants: HashMap::new(),
            marker: PhantomData,
        }
    }
}

fn lod_compaction_request_is_eligible(
    candidate_present: bool,
    candidate_matches_policy: bool,
    retained_current: bool,
) -> bool {
    // Waiting/prepared candidates need buffers to complete their handshake.
    // A package may publish an exact leaf frontier at quality one. Every bridge
    // candidate remains an explicit bounded atlas cut.
    // Policy matching rejects the one-frame stale pending candidate possible
    // when settings change after the main-world bridge update. A retained
    // current package cut is different: it is the last complete draw capability
    // and must keep (or recover) its allocation until a matching replacement
    // can publish atomically. Commit still refuses to activate a stale pending
    // candidate; this predicate only keeps the retained allocation admitted.
    candidate_present && (candidate_matches_policy || retained_current)
}

fn lod_candidate_matches_extracted_policy(
    candidate: &LodRenderCandidate,
    lod_settings: Option<&GaussianLodSettings>,
    lodge_settings: Option<&GaussianLodgeSettings>,
) -> bool {
    if candidate.is_external_active_set() {
        return lodge_settings.is_some_and(|settings| {
            candidate.rendered_candidate_count() <= settings.max_active_gaussians_u32()
                && candidate.frontier().selection_view_frozen()
                    == (settings.selection_mode == LodSelectionMode::Frozen)
        });
    }
    let Some(settings) = lod_settings else {
        return false;
    };
    let frontier = candidate.frontier();
    lod_frontier_matches_extracted_policy(
        settings.quality_target(),
        settings.max_active_gaussians_u32(),
        settings.selection_mode == LodSelectionMode::Frozen,
        frontier.quality_status().requested_target,
        frontier.candidate_count(),
        frontier.selection_view_frozen(),
    )
}

fn lod_compaction_policy_for_candidate(
    candidate: &LodRenderCandidate,
    lod_settings: Option<&GaussianLodSettings>,
    lodge_settings: Option<&GaussianLodgeSettings>,
) -> Option<LodCompactionPolicy> {
    if candidate.is_external_active_set() {
        lodge_settings.map(LodCompactionPolicy::external_active_set)
    } else {
        lod_settings.map(LodCompactionPolicy::hierarchy)
    }
}

const fn lod_pending_candidate_policy_allows_synchronization(
    candidate_is_current: bool,
    transition_must_commit: bool,
    candidate_matches_policy: bool,
) -> bool {
    candidate_is_current || transition_must_commit || candidate_matches_policy
}

fn lod_frontier_matches_extracted_policy(
    requested_target: LodQualityTarget,
    max_active_gaussians: u32,
    selection_view_frozen: bool,
    candidate_target: LodQualityTarget,
    candidate_count: u32,
    candidate_selection_view_frozen: bool,
) -> bool {
    candidate_target == requested_target
        && candidate_count <= max_active_gaussians
        && candidate_selection_view_frozen == selection_view_frozen
}

fn lod_compaction_requested_capacity(
    source_count: u32,
    policy_capacity: u32,
    retained_output_capacity: Option<u32>,
    current_candidate_count: Option<u32>,
) -> u32 {
    let required = policy_capacity
        .max(retained_output_capacity.unwrap_or(0))
        .max(current_candidate_count.unwrap_or(0));
    source_count.min(required).max(1)
}

/// Resolves the storage that candidate descriptors address independently from
/// the storage currently bound to the entity's legacy draw. A cold transient
/// bridge deliberately leaves `source` on the entity while its bounded atlas
/// is uploaded and its pipelines are prepared.
fn lod_compaction_asset_id<A: Asset>(
    source: AssetId<A>,
    candidates: Option<&LodRenderCandidates>,
) -> Option<AssetId<A>> {
    candidates
        .and_then(|set| set.staging_atlas)
        .map(|atlas| atlas.untyped().try_typed::<A>().ok())
        .unwrap_or(Some(source))
}

pub(crate) const fn cold_staging_candidate_phase(
    atlas_current: bool,
    compaction_pipelines_ready: bool,
    raster_pipeline_ready: bool,
    debug_binding_ready: bool,
    frontier_valid: bool,
) -> u8 {
    if !atlas_current
        || !compaction_pipelines_ready
        || !raster_pipeline_ready
        || !debug_binding_ready
    {
        LOD_RENDER_WAITING
    } else if frontier_valid {
        LOD_RENDER_PREPARED
    } else {
        LOD_RENDER_FAILED
    }
}

/// Preparation capability for a package replacement which still has a
/// drawable retained cut. Pending debug annotations are main-world staging
/// work, so an incomplete debug binding may be bypassed only for that retained
/// package transaction: package staging begins only after RenderWorld
/// acknowledges the structurally valid candidate as PREPARED. Every cold or
/// bridge candidate still waits, and the debug binding remains an activation
/// prerequisite below.
pub(crate) const fn retained_candidate_preparation_phase(
    compaction_pipelines_ready: bool,
    raster_pipeline_ready: bool,
    debug_activation_ready: bool,
    retained_package_replacement: bool,
    frontier_valid: bool,
) -> u8 {
    if !compaction_pipelines_ready
        || !raster_pipeline_ready
        || (!debug_activation_ready && !retained_package_replacement)
    {
        LOD_RENDER_WAITING
    } else if frontier_valid {
        LOD_RENDER_PREPARED
    } else {
        LOD_RENDER_FAILED
    }
}

/// A retained package output owns the only live descriptor/table allocation.
/// Replacements therefore remain validation-only until every consumer needed
/// to recompute, radix-sort, and rasterize the new bytes is ready in the same
/// frame. Debug may still publish PREPARED for main-world sidecar staging, but
/// it is not permission to overwrite the retained GPU presentation.
const fn retained_replacement_synchronization_ready(
    atlas_current: bool,
    compaction_and_radix_ready: bool,
    raster_pipeline_ready: bool,
    debug_activation_ready: bool,
) -> bool {
    atlas_current && compaction_and_radix_ready && raster_pipeline_ready && debug_activation_ready
}

pub(crate) const fn debug_incomplete_candidate_phase(
    requested_phase: u8,
    candidate_is_current: bool,
    compaction_pipelines_ready: bool,
    raster_pipeline_ready: bool,
    retained_package_replacement: bool,
    frontier_valid: bool,
) -> u8 {
    if candidate_is_current {
        // A presentation-only debug toggle cannot revoke the already-drawable
        // current descriptor/output while its sidecar binding is rebuilt.
        requested_phase
    } else {
        retained_candidate_preparation_phase(
            compaction_pipelines_ready,
            raster_pipeline_ready,
            false,
            retained_package_replacement,
            frontier_valid,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LodCandidateRasterPipelineReadiness {
    Pending,
    Ready,
    Failed,
}

impl LodCandidateRasterPipelineReadiness {
    const fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Failed, _) | (_, Self::Failed) => Self::Failed,
            (Self::Pending, _) | (_, Self::Pending) => Self::Pending,
            (Self::Ready, Self::Ready) => Self::Ready,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LodCandidateRasterGate {
    readiness: LodCandidateRasterPipelineReadiness,
    debug_activation_ready: bool,
    /// Number of retained render subviews which will actually consume this
    /// camera/cloud candidate. The phase token is shared across them even
    /// though compaction/radix output is private to each retained view.
    consumer_count: u32,
}

impl LodCandidateRasterGate {
    const fn merge(self, other: Self) -> Self {
        Self {
            readiness: self.readiness.merge(other.readiness),
            debug_activation_ready: self.debug_activation_ready && other.debug_activation_ready,
            consumer_count: self.consumer_count.saturating_add(other.consumer_count),
        }
    }
}

fn record_multi_subview_drawable_output(
    updates: &mut HashMap<usize, (Arc<AtomicU8>, u32, u32)>,
    candidate: &LodRenderCandidate,
    expected_consumers: u32,
) {
    let identity = Arc::as_ptr(&candidate.phase) as usize;
    updates
        .entry(identity)
        .and_modify(|(_, expected, ready)| {
            *expected = (*expected).max(expected_consumers);
            *ready = ready.saturating_add(1);
        })
        .or_insert_with(|| (Arc::clone(&candidate.phase), expected_consumers, 1));
}

struct LodViewBlendPublication<'a> {
    candidate: &'a LodRenderCandidate,
    expected_consumers: u32,
    drawable_consumers: u32,
    activation_allowed_consumers: u32,
    snapshot: Option<LodDrawableViewBlendSnapshot>,
}

fn record_drawable_view_blend_publication<'a>(
    publications: &mut HashMap<usize, LodViewBlendPublication<'a>>,
    candidate: &'a LodRenderCandidate,
    state: Option<&GpuLodCompaction>,
    selection_mode: LodSelectionMode,
) -> Result<bool, LodCandidateConfigError> {
    if candidate.view_blend_mode() != Some(LodTemporalTransitionMode::Morphing) {
        return Ok(false);
    }
    let candidate_identity = Arc::as_ptr(&candidate.phase) as usize;
    let publication =
        publications
            .entry(candidate_identity)
            .or_insert_with(|| LodViewBlendPublication {
                candidate,
                expected_consumers: 0,
                drawable_consumers: 0,
                activation_allowed_consumers: 0,
                snapshot: None,
            });
    publication.expected_consumers = publication.expected_consumers.saturating_add(1);
    let Some(state) = state else {
        return Ok(false);
    };
    let Some(snapshot) = state.promoted_view_blend_snapshot(candidate)? else {
        return Ok(false);
    };
    publication.drawable_consumers = publication.drawable_consumers.saturating_add(1);
    if state.morph_activation_allowed(candidate, selection_mode) {
        publication.activation_allowed_consumers =
            publication.activation_allowed_consumers.saturating_add(1);
    }
    if let Some(aggregate) = publication.snapshot.as_mut() {
        aggregate.merge_consumer(&snapshot)?;
    } else {
        publication.snapshot = Some(snapshot);
    }
    Ok(true)
}

fn view_blend_publication_is_complete(publication: &LodViewBlendPublication<'_>) -> bool {
    publication.expected_consumers > 0
        && publication.drawable_consumers == publication.expected_consumers
        && publication.snapshot.is_some()
}

fn view_blend_publication_can_activate(publication: &LodViewBlendPublication<'_>) -> bool {
    view_blend_publication_is_complete(publication)
        && publication.activation_allowed_consumers == publication.expected_consumers
        && publication
            .snapshot
            .as_ref()
            .is_some_and(|snapshot| snapshot.invalid_pressure_count() == 0)
}

fn publish_complete_view_blend_publication(publication: &LodViewBlendPublication<'_>) -> bool {
    if publication.expected_consumers == 0
        || publication.drawable_consumers != publication.expected_consumers
    {
        return false;
    }
    let Some(snapshot) = publication.snapshot.as_ref() else {
        return false;
    };
    publication.candidate.publish_view_blend_aggregate_snapshot(
        &snapshot.displayed,
        &snapshot.desired,
        snapshot.lagging_count(),
        snapshot.invalid_pressure_count(),
        0,
        snapshot.max_lag,
        snapshot.max_delta,
        snapshot.weighted_record_energy.min(f64::from(f32::MAX)) as f32,
        &snapshot.endpoints,
    )
}

/// An ACTIVE candidate missing any formerly expected private drawable must not
/// leave an exact endpoint mask behind for package retirement. This publishes
/// a conservative fractional mask while preserving any available exact
/// representative weights; PREPARED candidates simply remain unactivated.
fn publish_incomplete_view_blend_hold(publication: &LodViewBlendPublication<'_>) -> bool {
    if !publication.candidate.render_is_active() && !publication.candidate.render_is_transitioning()
    {
        return true;
    }
    let (displayed, desired, mut endpoints, invalid_pressure_count, max_lag, max_delta, energy) =
        if let Some(snapshot) = publication.snapshot.as_ref() {
            (
                snapshot.displayed.clone(),
                snapshot.desired.clone(),
                snapshot.endpoints.clone(),
                snapshot.invalid_pressure_count(),
                snapshot.max_lag,
                snapshot.max_delta,
                snapshot.weighted_record_energy,
            )
        } else {
            let Some(morph) = publication
                .candidate
                .temporal_transition()
                .and_then(|transition| transition.morph())
            else {
                return false;
            };
            let displayed = morph
                .edges()
                .iter()
                .map(|edge| edge.initial_weight())
                .collect::<Vec<_>>();
            (
                displayed.clone(),
                displayed,
                vec![LodViewBlendEndpoint::Fractional; morph.edges().len()],
                0,
                0.0,
                0.0,
                0.0,
            )
        };
    endpoints.fill(LodViewBlendEndpoint::Fractional);
    let exact_lagging_count: u32 = displayed
        .iter()
        .zip(&desired)
        .filter(|(displayed, desired)| displayed.to_bits() != desired.to_bits())
        .count()
        .try_into()
        .unwrap_or(u32::MAX);
    let lagging_count = exact_lagging_count;
    let missing_consumer_count = publication
        .expected_consumers
        .saturating_sub(publication.drawable_consumers);
    publication.candidate.publish_view_blend_aggregate_snapshot(
        &displayed,
        &desired,
        lagging_count,
        invalid_pressure_count,
        missing_consumer_count,
        max_lag,
        max_delta,
        energy.min(f64::from(f32::MAX)) as f32,
        &endpoints,
    )
}

const fn multi_subview_activation_ready(expected_consumers: u32, ready_consumers: u32) -> bool {
    expected_consumers > 1 && ready_consumers == expected_consumers
}

fn lod_candidate_raster_pipeline_readiness(
    state: &CachedPipelineState,
) -> LodCandidateRasterPipelineReadiness {
    match state {
        CachedPipelineState::Queued | CachedPipelineState::Creating(_) => {
            LodCandidateRasterPipelineReadiness::Pending
        }
        CachedPipelineState::Ok(_) => LodCandidateRasterPipelineReadiness::Ready,
        CachedPipelineState::Err(_) => LodCandidateRasterPipelineReadiness::Failed,
    }
}

fn merge_cold_staging_phase(left: u8, right: u8) -> u8 {
    if left == LOD_RENDER_FAILED || right == LOD_RENDER_FAILED {
        LOD_RENDER_FAILED
    } else if left == LOD_RENDER_PREPARED && right == LOD_RENDER_PREPARED {
        LOD_RENDER_PREPARED
    } else {
        LOD_RENDER_WAITING
    }
}

fn record_cold_staging_phase(
    updates: &mut HashMap<usize, (Arc<AtomicU8>, u8)>,
    candidate: &LodRenderCandidate,
    phase: u8,
) {
    let identity = Arc::as_ptr(&candidate.phase) as usize;
    updates
        .entry(identity)
        .and_modify(|(_, current)| *current = merge_cold_staging_phase(*current, phase))
        .or_insert_with(|| (Arc::clone(&candidate.phase), phase));
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
        Option<&GaussianLodSettings>,
        Option<&GaussianLodgeSettings>,
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
    let storage_buffer_count_supported = lod_storage_buffer_count_is_supported(
        pipeline.storage_buffers_per_shader_stage,
        device_limits.max_storage_buffers_per_shader_stage,
    );
    let mut requests = Vec::new();
    for (view, visible_entities) in &views {
        let Some(visible_clouds) = visible_entities.get::<CloudVisibilityClass>() else {
            continue;
        };
        let camera = view.retained_view_entity.main_entity.id();
        for (render_entity, _) in &visible_clouds.entities_cpu_culling {
            let Ok((entity, handle, cloud_settings, lod_settings, lodge_settings, candidates)) =
                clouds.get(*render_entity)
            else {
                continue;
            };
            let candidate = candidates.and_then(|set| set.by_camera.get(&camera));
            let candidate_present = candidate.is_some_and(|candidate| !candidate.failed());
            let candidate_matches_policy = candidate
                .filter(|candidate| !candidate.failed())
                .is_some_and(|candidate| {
                    lod_candidate_matches_extracted_policy(candidate, lod_settings, lodge_settings)
                });
            let retained_current =
                candidates.is_some_and(|set| set.candidate_draw_required && set.retained_current);
            if !lod_compaction_request_is_eligible(
                candidate_present,
                candidate_matches_policy,
                retained_current,
            ) {
                continue;
            }
            let candidate = candidate.expect("eligible LoD request has a candidate");
            let Some(policy) =
                lod_compaction_policy_for_candidate(candidate, lod_settings, lodge_settings)
            else {
                continue;
            };
            let hard_fallback_policy = lod_candidate_hard_fallback_policy(
                candidates
                    .is_some_and(LodRenderCandidates::requires_package_hard_fallback_handshake),
            );
            if candidate.is_external_active_set()
                && cloud_settings.gaussian_mode != GaussianMode::Gaussian3d
            {
                candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                continue;
            }
            if !candidate.is_external_active_set() {
                enforce_lod_candidate_gaussian_morph_capability(
                    candidate,
                    cloud_settings.gaussian_mode,
                    hard_fallback_policy,
                );
            }
            let required_morph_words = if candidate.is_external_active_set() {
                Some(LOD_PRESENTATION_HEADER_WORDS)
            } else if candidate.render_hard_fallback_requested() {
                None
            } else {
                match plan_lod_candidate_morph(
                    candidate,
                    device_limits.max_buffer_size,
                    device_limits.max_storage_buffer_binding_size,
                ) {
                    Ok(LodCandidateMorphPlan::Enabled { required_words, .. }) => {
                        Some(required_words)
                    }
                    Ok(LodCandidateMorphPlan::Disabled) => Some(LOD_MORPH_HEADER_WORDS),
                    Ok(LodCandidateMorphPlan::Unsupported) => {
                        match publish_lod_candidate_hard_fallback(candidate, hard_fallback_policy) {
                            LodCandidateMorphSynchronization::HardFallbackRequested => None,
                            LodCandidateMorphSynchronization::Disabled => {
                                Some(LOD_MORPH_HEADER_WORDS)
                            }
                            LodCandidateMorphSynchronization::Enabled => unreachable!(
                                "a hard fallback decision cannot enable morph presentation"
                            ),
                        }
                    }
                    Err(error) => {
                        error!(
                            ?entity,
                            ?camera,
                            %error,
                            "invalid LoD morph payload; rejecting candidate before allocation"
                        );
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                        continue;
                    }
                }
            };
            let Some(compaction_id) = lod_compaction_asset_id(handle.handle().id(), candidates)
            else {
                continue;
            };
            if candidate.render_hard_fallback_requested() {
                if let Some(state) = buffers.get(view.retained_view_entity, entity, compaction_id) {
                    requests.push(LodCompactionAdmissionRequest {
                        payload: (
                            view.retained_view_entity,
                            entity,
                            compaction_id,
                            state.source_count(),
                            None,
                            cloud_settings.gaussian_mode,
                            cloud_settings.radix_sort_depth_bits,
                            policy,
                            true,
                        ),
                        total_bytes: state.resident_admission_bytes(),
                        class: LodCompactionAdmissionClass::RetainedRequiredOutput,
                        required_phase: Some(candidate.phase.as_ref()),
                        pinned_existing: true,
                    });
                }
                // A cold held package token has no state to preserve and may
                // not allocate one before the main world cancels/replans it.
                continue;
            }
            let required_morph_words = required_morph_words
                .expect("a non-held morph plan has an admission word requirement");
            if !storage_buffer_count_supported {
                error!(
                    ?entity,
                    ?camera,
                    required = pipeline.storage_buffers_per_shader_stage,
                    available = device_limits.max_storage_buffers_per_shader_stage,
                    "LoD compaction exceeds the adapter's compute-stage storage-buffer limit; rejecting candidate without allocating"
                );
                candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                continue;
            }
            if validate_bridge_candidate_sort_mode(&cloud_settings.sort_mode).is_err() {
                continue;
            }
            let Some(cloud) = gpu_clouds.get(compaction_id) else {
                continue;
            };
            // u32 indices are a hard GPU ABI boundary. A cold handoff sizes the
            // bounded staging atlas to current policy. Once a package has a
            // drawable current output, never shrink below that live allocation
            // or current candidate while its atomic replacement is pending.
            let Some(source_count) = representable_source_count(cloud.len()) else {
                continue;
            };
            let current_state = buffers.get(view.retained_view_entity, entity, compaction_id);
            let current_output_capacity = current_state.map(GpuLodCompaction::output_capacity);
            let current_morph_word_capacity =
                current_state.map_or(LOD_MORPH_HEADER_WORDS, |state| state.morph_word_capacity);
            let retained_output_capacity = if retained_current {
                current_output_capacity
            } else {
                None
            };
            let current_candidate_count = if candidates
                .is_some_and(|set| set.candidate_draw_required && set.candidates_are_current)
            {
                Some(candidate.rendered_candidate_count())
            } else {
                None
            };
            let requested_capacity = lod_compaction_requested_capacity(
                source_count,
                policy.max_active_gaussians,
                retained_output_capacity,
                current_candidate_count,
            );
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
            let Some(admission_total_bytes) = lod_compaction_admission_bytes_with_morph(
                allocation.total_bytes,
                current_morph_word_capacity,
                required_morph_words,
            ) else {
                candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                continue;
            };
            let required_phase = candidates
                .filter(|set| set.candidate_draw_required)
                .map(|_| candidate.phase.as_ref());
            let retains_required_output = required_phase.is_some()
                && buffers
                    .get(view.retained_view_entity, entity, compaction_id)
                    .is_some_and(|state| state.is_ready() && state.has_drawable_bridge_output());
            let class = if retains_required_output {
                LodCompactionAdmissionClass::RetainedRequiredOutput
            } else if required_phase.is_some() {
                LodCompactionAdmissionClass::RequiredOutput
            } else {
                LodCompactionAdmissionClass::FallbackCapable
            };
            requests.push(LodCompactionAdmissionRequest {
                payload: (
                    view.retained_view_entity,
                    entity,
                    compaction_id,
                    source_count,
                    Some(allocation),
                    cloud_settings.gaussian_mode,
                    cloud_settings.radix_sort_depth_bits,
                    policy,
                    false,
                ),
                total_bytes: admission_total_bytes,
                class,
                required_phase,
                pinned_existing: false,
            });
        }
    }

    // Query/archetype order is not a memory-priority contract. Stable identity
    // order is the reproducible tie-breaker within each admission class.
    requests.sort_by(|left, right| {
        left.payload
            .0
            .main_entity
            .cmp(&right.payload.0.main_entity)
            .then_with(|| {
                left.payload
                    .0
                    .auxiliary_entity
                    .cmp(&right.payload.0.auxiliary_entity)
            })
            .then_with(|| {
                left.payload
                    .0
                    .subview_index
                    .cmp(&right.payload.0.subview_index)
            })
            .then_with(|| left.payload.1.cmp(&right.payload.1))
            .then_with(|| left.payload.2.cmp(&right.payload.2))
    });
    requests.dedup_by(|right, left| {
        right.payload.0 == left.payload.0
            && right.payload.1 == left.payload.1
            && right.payload.2 == left.payload.2
    });

    let admitted = admit_lod_compaction_requests(requests, aggregate_limit);

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
            if request.8 {
                return None;
            }
            let allocation = request
                .4
                .expect("a non-held admission request has an allocation plan");
            buffers
                .entries
                .get(&key)
                .is_none_or(|entry| {
                    entry.source_count() != request.3
                        || entry.output_capacity() != allocation.effective_capacity
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
        policy,
        hard_fallback_requested,
    ) in admitted
    {
        if hard_fallback_requested {
            // Admission above charges and retains this key, but a render-only
            // veto may not reconfigure, resize, or replace the last drawable
            // state before package orchestration cancels the pending token.
            continue;
        }
        let allocation = allocation.expect("a non-held admitted request has an allocation plan");
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
                policy,
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
            entry.set_policy(&render_queue, policy);
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
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn commit_lod_bridge_candidates<R: PlanarSync>(
    mut buffers: ResMut<LodCompactionBuffers<R>>,
    atlas_generations: Res<LodAtlasGpuGenerations>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<CloudPipeline<R>>,
    mut raster_pipelines: ResMut<SpecializedRenderPipelines<CloudPipeline<R>>>,
    pipeline_cache: Res<PipelineCache>,
    gpu_clouds: Res<RenderAssets<R::GpuPlanarType>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities, Option<&Msaa>), With<GaussianCamera>>,
    clouds: Query<(
        Entity,
        &R::PlanarTypeHandle,
        Ref<PlanarStorageBindGroup<R>>,
        &CloudSettings,
        Option<&GaussianLodSettings>,
        Option<&GaussianLodgeSettings>,
        &GlobalTransform,
        Option<&LodRenderCandidates>,
        Option<&LodDebugMetadata>,
        Option<&LodDebugBindGroup<R>>,
    )>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    // One main-world camera may own several retained render subviews with
    // different target formats or MSAA. Its candidate phase is shared, so no
    // individual subview may publish PREPARED from its private raster key.
    // Queue every exact permutation first, then reduce them to one fail-closed
    // gate keyed by the immutable candidate identity.
    let mut raster_gates = HashMap::<(Entity, Entity, usize), LodCandidateRasterGate>::new();
    for (view, visible_entities, msaa) in &views {
        let Some(visible_clouds) = visible_entities.get::<CloudVisibilityClass>() else {
            continue;
        };
        let camera = view.retained_view_entity.main_entity.id();
        for (render_entity, _) in &visible_clouds.entities_cpu_culling {
            let Ok((
                entity,
                handle,
                _,
                cloud_settings,
                _,
                _,
                _,
                candidates,
                debug_metadata,
                debug_binding,
            )) = clouds.get(*render_entity)
            else {
                continue;
            };
            let Some(candidate) = candidates.and_then(|set| set.by_camera.get(&camera)) else {
                continue;
            };
            if !gaussian_rasterization_is_supported(
                cloud_settings.gaussian_mode,
                cloud_settings.rasterize_mode,
            ) {
                error!(
                    ?entity,
                    ?camera,
                    gaussian_mode = ?cloud_settings.gaussian_mode,
                    rasterize_mode = ?cloud_settings.rasterize_mode,
                    "unsupported LoD Gaussian/rasterization mode pair; rejecting render commit"
                );
                let gate = LodCandidateRasterGate {
                    readiness: LodCandidateRasterPipelineReadiness::Failed,
                    debug_activation_ready: false,
                    consumer_count: 1,
                };
                let identity = Arc::as_ptr(&candidate.phase) as usize;
                raster_gates
                    .entry((camera, entity, identity))
                    .and_modify(|aggregate| *aggregate = aggregate.merge(gate))
                    .or_insert(gate);
                continue;
            }
            let debug_required = cloud_settings.lod_debug.requires_metadata();
            if candidate.is_external_active_set() && debug_required {
                error!(
                    ?entity,
                    ?camera,
                    preset = ?cloud_settings.lod_debug,
                    "hierarchy LoD debug metadata is unsupported for an external active-set candidate"
                );
                let gate = LodCandidateRasterGate {
                    readiness: LodCandidateRasterPipelineReadiness::Failed,
                    debug_activation_ready: false,
                    consumer_count: 1,
                };
                let identity = Arc::as_ptr(&candidate.phase) as usize;
                raster_gates
                    .entry((camera, entity, identity))
                    .and_modify(|aggregate| *aggregate = aggregate.merge(gate))
                    .or_insert(gate);
                continue;
            }
            // Hierarchy Page/Level/Residency metadata has no meaning for a
            // resident LODGE catalog. Treat those presets as unsupported for
            // this candidate instead of waiting forever on atlas-slot/debug
            // invariants which external ranges deliberately do not publish.
            let debug_capability = if candidate.is_external_active_set() {
                LodDebugRenderCapability::Unsupported
            } else {
                classify_lod_debug_render_capability(
                    &pipeline,
                    &render_device,
                    &gpu_clouds,
                    handle,
                    debug_metadata,
                )
            };
            let debug_activation_ready = lod_debug_candidate_activation_ready(
                debug_required,
                debug_capability,
                candidates.is_some_and(|candidates| candidates.debug_metadata_staged),
                candidates
                    .zip(debug_binding)
                    .is_some_and(|(candidates, binding)| {
                        binding.candidate_invariants_ready(debug_metadata, candidates)
                    }),
            ) && (!debug_required
                || debug_capability == LodDebugRenderCapability::Unsupported
                || debug_binding.is_some_and(|debug| debug.ready));
            // Prewarm both stable debug permutations for this exact
            // mode/MSAA/HDR key. Enabling Page/Level/Residency later can then
            // switch bindings without sending an already-ACTIVE atlas through
            // SetItemPipeline's asynchronous `Skip` window. The initial cold
            // handoff waits for both variants while the immutable source is
            // still drawable.
            let mut readiness = LodCandidateRasterPipelineReadiness::Ready;
            let debug_variant_count = if !candidate.is_external_active_set()
                && pipeline.lod_debug_layout_desc.is_some()
            {
                2
            } else {
                1
            };
            for debug_pipeline_active in [false, true].into_iter().take(debug_variant_count) {
                let key = cloud_pipeline_key(
                    cloud_settings,
                    debug_pipeline_active,
                    true,
                    msaa.cloned().unwrap_or_default().samples(),
                    view.target_format == TextureFormat::Rgba16Float,
                );
                let raster_pipeline = raster_pipelines.specialize(&pipeline_cache, &pipeline, key);
                let raster_pipeline_state =
                    pipeline_cache.get_render_pipeline_state(raster_pipeline);
                let variant_readiness =
                    lod_candidate_raster_pipeline_readiness(raster_pipeline_state);
                readiness = readiness.merge(variant_readiness);
                if variant_readiness == LodCandidateRasterPipelineReadiness::Failed {
                    let CachedPipelineState::Err(error) = raster_pipeline_state else {
                        unreachable!("failed raster readiness requires a pipeline error")
                    };
                    error!(
                        ?entity,
                        ?camera,
                        lod_debug = debug_pipeline_active,
                        %error,
                        "LoD candidate raster pipeline failed; rejecting render commit"
                    );
                }
            }
            let gate = LodCandidateRasterGate {
                readiness,
                debug_activation_ready,
                consumer_count: 1,
            };
            let identity = Arc::as_ptr(&candidate.phase) as usize;
            raster_gates
                .entry((camera, entity, identity))
                .and_modify(|aggregate| *aggregate = aggregate.merge(gate))
                .or_insert(gate);
        }
    }

    let mut cold_staging_updates = HashMap::<usize, (Arc<AtomicU8>, u8)>::new();
    let mut multi_subview_drawable_outputs = HashMap::<usize, (Arc<AtomicU8>, u32, u32)>::new();
    for (view, visible_entities, _) in &views {
        let Some(visible_clouds) = visible_entities.get::<CloudVisibilityClass>() else {
            continue;
        };
        let camera = view.retained_view_entity.main_entity.id();
        for (render_entity, _) in &visible_clouds.entities_cpu_culling {
            let Ok((
                entity,
                handle,
                cloud_bind_group,
                cloud_settings,
                lod_settings,
                lodge_settings,
                transform,
                candidates,
                _,
                _,
            )) = clouds.get(*render_entity)
            else {
                continue;
            };
            let candidate = candidates.and_then(|set| set.by_camera.get(&camera));
            let retained_package_replacement = candidates.is_some_and(|set| {
                set.candidate_draw_required && set.retained_current && !set.candidates_are_current
            });
            let candidate_is_current = candidates
                .is_some_and(|set| set.candidate_draw_required && set.candidates_are_current);
            let hard_fallback_policy = lod_candidate_hard_fallback_policy(
                candidates
                    .is_some_and(LodRenderCandidates::requires_package_hard_fallback_handshake),
            );
            if let Some(candidate) = candidate {
                // This exact camera/cloud consumer crossed extraction and is
                // visible to a retained RenderView. Publish before buffer or
                // pipeline readiness so camera-only main-world churn cannot
                // cancel it in the N/N+1 gap before PREPARED.
                candidate.publish_render_claimed();
                if candidate.is_external_active_set()
                    && cloud_settings.gaussian_mode != GaussianMode::Gaussian3d
                {
                    candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                    continue;
                }
                if !candidate.is_external_active_set() {
                    enforce_lod_candidate_gaussian_morph_capability(
                        candidate,
                        cloud_settings.gaussian_mode,
                        hard_fallback_policy,
                    );
                }
                if candidate.render_hard_fallback_requested() {
                    if let Some(compaction_id) =
                        lod_compaction_asset_id(handle.handle().id(), candidates)
                        && let Some(state) =
                            buffers.get_mut(view.retained_view_entity, entity, compaction_id)
                    {
                        // A late radix completion may still hold this Arc. The
                        // WAITING phase already makes its CAS fail; removing the
                        // arm also prevents the stale token from lingering.
                        state.defer_bridge_activation_for(candidate);
                    }
                    continue;
                }
                if candidate.view_blend_replan_requested() {
                    if let Some(compaction_id) =
                        lod_compaction_asset_id(handle.handle().id(), candidates)
                        && let Some(state) =
                            buffers.get_mut(view.retained_view_entity, entity, compaction_id)
                    {
                        state.defer_bridge_activation_for(candidate);
                    }
                    continue;
                }
            }
            let raster_gate = candidate
                .and_then(|candidate| {
                    raster_gates.get(&(camera, entity, Arc::as_ptr(&candidate.phase) as usize))
                })
                .copied()
                .unwrap_or(LodCandidateRasterGate {
                    readiness: LodCandidateRasterPipelineReadiness::Pending,
                    debug_activation_ready: false,
                    consumer_count: 0,
                });
            let raster_pipeline_ready =
                raster_gate.readiness == LodCandidateRasterPipelineReadiness::Ready;
            let debug_activation_ready = raster_gate.debug_activation_ready;
            if raster_gate.readiness == LodCandidateRasterPipelineReadiness::Failed {
                if let Some(candidate) = candidate {
                    let cold_staging = candidates.is_some_and(|set| {
                        !set.candidate_draw_required
                            && set
                                .staging_atlas
                                .is_some_and(|atlas| atlas != handle.handle().id().untyped())
                    });
                    if cold_staging {
                        record_cold_staging_phase(
                            &mut cold_staging_updates,
                            candidate,
                            LOD_RENDER_FAILED,
                        );
                    } else {
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                    }
                }
                continue;
            }
            let cold_staging = candidates.is_some_and(|set| {
                !set.candidate_draw_required
                    && set
                        .staging_atlas
                        .is_some_and(|atlas| atlas != handle.handle().id().untyped())
            });
            let Some(compaction_id) = lod_compaction_asset_id(handle.handle().id(), candidates)
            else {
                if let Some(candidates) = candidates {
                    for candidate in candidates.by_camera.values() {
                        if cold_staging {
                            record_cold_staging_phase(
                                &mut cold_staging_updates,
                                candidate,
                                LOD_RENDER_FAILED,
                            );
                        } else {
                            candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                        }
                    }
                }
                continue;
            };
            let candidate_atlas = compaction_id.untyped();
            if validate_bridge_candidate_sort_mode(&cloud_settings.sort_mode).is_err() {
                if let Some(state) =
                    buffers.get_mut(view.retained_view_entity, entity, compaction_id)
                {
                    state.invalidate_candidates(&render_queue);
                }
                if let Some(candidate) = candidate {
                    if cold_staging {
                        record_cold_staging_phase(
                            &mut cold_staging_updates,
                            candidate,
                            LOD_RENDER_FAILED,
                        );
                    } else {
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                    }
                }
                continue;
            }
            if let Some(candidate) = candidate {
                let requested_phase = candidate.phase.load(Ordering::Acquire);
                if requested_phase == LOD_RENDER_FAILED
                    || candidate.frontier.view().0 != camera.to_bits()
                {
                    // A rejected replacement never owned the current GPU
                    // descriptor/output. Preserve that last-known-good state;
                    // the main world drops the failed token next update.
                    candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                    continue;
                }
                if !lod_pending_candidate_policy_allows_synchronization(
                    candidate_is_current,
                    candidates.is_some_and(|set| set.transition_must_commit),
                    lod_candidate_matches_extracted_policy(candidate, lod_settings, lodge_settings),
                ) {
                    // A settings edit can land after package orchestration but
                    // before extraction. Keep the previous drawable output and
                    // allocation intact; the main world will discard this stale
                    // pending token rather than letting it activate under a new
                    // quality, capacity, or selection-mode policy.
                    continue;
                }
            }
            let Some(state) = buffers.get_mut(view.retained_view_entity, entity, compaction_id)
            else {
                // Revoke an extracted capability if aggregate/device limits
                // removed its GPU state. The main world will restore the flat
                // atlas before publishing another active cut.
                if let Some(candidate) = candidate {
                    if cold_staging {
                        record_cold_staging_phase(
                            &mut cold_staging_updates,
                            candidate,
                            LOD_RENDER_WAITING,
                        );
                    } else {
                        candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
                    }
                }
                continue;
            };
            state.morph_activation_preflight_valid = false;
            let Some(candidate) = candidate else {
                if readiness_without_bridge_candidate(state.readiness, state.candidate_ownership)
                    != state.readiness
                {
                    state.invalidate_candidates(&render_queue);
                }
                continue;
            };
            let multi_subview_candidate = raster_gate.consumer_count > 1;
            if multi_subview_candidate {
                // No private radix node may publish a shared phase on its own.
                // The complete set of actual visible consumers is reduced
                // after every state has synchronized below.
                state.defer_bridge_activation_for(candidate);
            }
            let external_presentation = candidate.external_active_set();
            let resident_external_catalog = external_presentation.is_some()
                && compaction_id == handle.handle().id()
                && candidates.is_some_and(|set| set.staging_atlas.is_none());
            if external_presentation.is_some() && !resident_external_catalog {
                // The external ABI currently addresses one canonical resident
                // PlanarGaussian3d catalog. It must never reinterpret package
                // page-slot generations as stable catalog indices.
                candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                state.defer_bridge_activation_for(candidate);
                continue;
            }
            let resident_catalog_tick = cloud_bind_group.last_changed().get();
            let atlas_allocation_epoch = if resident_external_catalog {
                Some(lod_resident_catalog_epoch(resident_catalog_tick))
            } else {
                atlas_generations.allocation_epoch(candidate_atlas)
            };
            if !lod_drawable_atlas_allocation_is_current(
                state.has_drawable_bridge_output,
                state.candidate_atlas_allocation_epoch,
                atlas_allocation_epoch,
            ) {
                // Logical slot generations can repeat across real storage
                // recreation. Revoke this private indirect output before the
                // WAITING/RetainCurrent path can bind it to the new allocation.
                state.invalidate_candidates(&render_queue);
                if matches!(
                    candidate.phase.load(Ordering::Acquire),
                    LOD_RENDER_ACTIVE | LOD_RENDER_TRANSITIONING
                ) {
                    candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
                }
                continue;
            }
            if cold_staging {
                let atlas = candidate_atlas;
                let atlas_current = candidates.is_some_and(|set| {
                    set.by_camera.values().all(|candidate| {
                        atlas_generations
                            .frontier_is_current(atlas, candidate.required_atlas_ranges())
                    })
                });
                let phase = if atlas_current && state.pipelines_ready() {
                    cold_staging_candidate_phase(
                        true,
                        true,
                        raster_pipeline_ready,
                        debug_activation_ready,
                        state
                            .validate_bridge_candidate_presentation(candidate)
                            .is_ok(),
                    )
                } else {
                    cold_staging_candidate_phase(
                        atlas_current,
                        state.pipelines_ready(),
                        raster_pipeline_ready,
                        debug_activation_ready,
                        true,
                    )
                };
                record_cold_staging_phase(&mut cold_staging_updates, candidate, phase);
                continue;
            }
            state.retain_pending_activation_for(candidate);
            if !debug_activation_ready {
                state.defer_bridge_activation_for(candidate);
            }
            let requested_phase = candidate.phase.load(Ordering::Acquire);
            let atlas = candidate_atlas;
            let atlas_current = if resident_external_catalog {
                // The catalog bind group itself is the allocation proof. Range
                // bounds, declared counts, and one class per range remain
                // fail-closed host validation for every camera in the shared
                // candidate transaction.
                candidates.is_some_and(|set| {
                    set.by_camera.values().all(|candidate| {
                        candidate.external_active_set().is_some_and(|presentation| {
                            state
                                .validate_bridge_external_active_set(candidate, presentation)
                                .is_ok()
                        })
                    })
                })
            } else if candidates.is_some_and(|set| set.candidate_draw_required) {
                // A package stages the union of every camera frontier as one
                // transaction. Do not let an early camera publish merely
                // because its low-index subset landed before deferred slots
                // required by another view.
                candidates.is_some_and(|set| {
                    set.by_camera.values().all(|candidate| {
                        atlas_generations
                            .frontier_is_current(atlas, candidate.required_atlas_ranges())
                    })
                })
            } else {
                atlas_generations.frontier_is_current(atlas, candidate.required_atlas_ranges())
            };
            match lod_bridge_atlas_decision(requested_phase, atlas_current) {
                LodBridgeAtlasDecision::RejectActive => {
                    // ACTIVE is a capability, not merely a main-world intent. If
                    // any physical slot upload is absent or has since been reused,
                    // revoke the cut before compaction can read stale atlas data.
                    state.invalidate_candidates(&render_queue);
                    candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
                    continue;
                }
                LodBridgeAtlasDecision::RetainCurrent => {
                    // Validate the pending payload without replacing the one GPU
                    // descriptor buffer. The previous active descriptor/output can
                    // therefore be recomputed for a moving camera while arbitrarily
                    // many replacement slots upload under the per-frame budget.
                    let frontier_valid = state
                        .validate_bridge_candidate_presentation(candidate)
                        .is_ok();
                    candidate.phase.store(
                        retained_candidate_preparation_phase(
                            state.pipelines_ready(),
                            raster_pipeline_ready,
                            debug_activation_ready,
                            retained_package_replacement,
                            frontier_valid,
                        ),
                        Ordering::Release,
                    );
                    continue;
                }
                LodBridgeAtlasDecision::SynchronizePending => {}
            }
            if retained_package_replacement
                && !retained_replacement_synchronization_ready(
                    atlas_current,
                    state.pipelines_ready(),
                    raster_pipeline_ready,
                    debug_activation_ready,
                )
            {
                // This state still backs the retained package draw. Validate
                // the replacement without writing its descriptor/table bytes;
                // otherwise a skipped compute/radix/raster stage could pair
                // the retained sorted output with the replacement morph table.
                let frontier_valid = state
                    .validate_bridge_candidate_presentation(candidate)
                    .is_ok();
                candidate.phase.store(
                    retained_candidate_preparation_phase(
                        state.pipelines_ready(),
                        raster_pipeline_ready,
                        debug_activation_ready,
                        true,
                        frontier_valid,
                    ),
                    Ordering::Release,
                );
                state.defer_bridge_activation_for(candidate);
                continue;
            }
            if !debug_activation_ready {
                // A retained package replacement needs PREPARED to let the
                // main world finish its bounded debug-sidecar staging, but it
                // must not replace the descriptor/output which still draws the
                // previous cut. Cold/bridge candidates have no such retained
                // package transaction and remain WAITING.
                let frontier_valid = state
                    .validate_bridge_candidate_presentation(candidate)
                    .is_ok();
                candidate.phase.store(
                    debug_incomplete_candidate_phase(
                        requested_phase,
                        candidate_is_current,
                        state.pipelines_ready(),
                        raster_pipeline_ready,
                        retained_package_replacement,
                        frontier_valid,
                    ),
                    Ordering::Release,
                );
                continue;
            }
            if !candidate.is_external_active_set()
                && candidates
                    .is_some_and(LodRenderCandidates::requires_package_hard_fallback_handshake)
                && view_blend_predecessor_attestation_required(
                    retained_package_replacement,
                    state.candidate_upload.plan(candidate),
                )
            {
                let Some(lod_settings) = lod_settings else {
                    state.defer_bridge_activation_for(candidate);
                    continue;
                };
                let predecessor_current = state.view_blend_predecessor_attestation_is_current(
                    view,
                    transform,
                    lod_settings,
                    candidate,
                );
                if !matches!(predecessor_current, Ok(true)) {
                    // The pipelined main world approved retirement from older
                    // endpoint evidence, or this private view no longer agrees
                    // with the unanimous categorical side. Preserve the exact
                    // retained descriptor/table/output and ask package
                    // orchestration for a fresh non-hard selection.
                    candidate.request_view_blend_replan();
                    state.defer_bridge_activation_for(candidate);
                    continue;
                }
            }
            if let Some(presentation) = external_presentation {
                let Some((first_weight, second_weight)) =
                    lod_external_active_set_weights(view, transform, presentation)
                else {
                    // Preserve an ACTIVE presentation's last finite raster
                    // header. A pending candidate cannot become a capability
                    // until one current local camera produces valid weights.
                    if !matches!(
                        requested_phase,
                        LOD_RENDER_ACTIVE | LOD_RENDER_TRANSITIONING
                    ) {
                        candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
                    }
                    state.defer_bridge_activation_for(candidate);
                    continue;
                };
                let catalog_epoch = lod_resident_catalog_epoch(resident_catalog_tick);
                let catalog_signature = lod_resident_catalog_content_signature(
                    resident_catalog_tick,
                    state.source_count(),
                );
                if state
                    .synchronize_bridge_external_active_set(
                        &render_device,
                        &render_queue,
                        candidate,
                        presentation,
                        first_weight,
                        second_weight,
                        catalog_epoch,
                        catalog_epoch,
                        || catalog_signature,
                    )
                    .is_err()
                {
                    candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                    state.defer_bridge_activation_for(candidate);
                    continue;
                }
            } else {
                let atlas_content_revision = atlas_generations.content_revision(atlas);
                match state.synchronize_bridge_candidate_frontier(
                    &render_device,
                    &render_queue,
                    candidate,
                    hard_fallback_policy,
                    atlas_allocation_epoch,
                    atlas_content_revision,
                    || {
                        atlas_generations
                            .frontier_content_signature(atlas, candidate.required_atlas_ranges())
                    },
                ) {
                    Ok(LodCandidateMorphSynchronization::HardFallbackRequested) => {
                        // The package will cancel this unrendered token and re-run
                        // ordinary hard-cut admission. Preserve the complete prior
                        // descriptor, table, suffix, and sorted output meanwhile.
                        state.defer_bridge_activation_for(candidate);
                        continue;
                    }
                    Ok(
                        LodCandidateMorphSynchronization::Disabled
                        | LodCandidateMorphSynchronization::Enabled,
                    ) => {}
                    Err(_) => {
                        // Validation completes before candidate buffer/config mutation,
                        // so the previous active output remains a valid fallback.
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                        continue;
                    }
                }
            }

            if matches!(
                requested_phase,
                LOD_RENDER_ACTIVE | LOD_RENDER_TRANSITIONING
            ) {
                if state.morph_identity.is_some() {
                    let Some(lod_settings) = lod_settings else {
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                        state.defer_bridge_activation_for(candidate);
                        continue;
                    };
                    // Capture only the suffix generation already proven
                    // drawable by compaction+radix. The retained snapshot then
                    // remains valid while a private subview prepares its next
                    // camera-conditioned suffix, so all subviews can be
                    // reduced without requiring coincident sort completion.
                    if state.radix_sort_is_current()
                        && state
                            .prime_initial_recovery_view_blend_desired(
                                view,
                                transform,
                                lod_settings,
                                candidate,
                            )
                            .and_then(|()| state.capture_drawable_view_blend_snapshot())
                            .is_err()
                    {
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                        continue;
                    }
                    let drawable_before_update =
                        state.has_current_drawable_bridge_candidate(candidate);
                    if state.radix_sort_is_current()
                        && state
                            .update_view_blend_weights(
                                &render_queue,
                                view,
                                transform,
                                lod_settings,
                                candidate,
                            )
                            .is_err()
                    {
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                        continue;
                    }
                    state.morph_activation_preflight_valid = debug_activation_ready
                        && drawable_before_update
                        && !state
                            .morph_drawable_invalid_pressure_edges
                            .iter()
                            .any(|&invalid| invalid);
                }
                // Cleanup publishes the exact current radix generation. An
                // already-ACTIVE adjacency remains ACTIVE while camera weights
                // update; a legacy TRANSITIONING token waits at the aggregate
                // barrier instead of bypassing it here.
                state.defer_bridge_activation_for(candidate);
                continue;
            }
            if state.pipelines_ready() && raster_pipeline_ready {
                if state.morph_identity.is_some() {
                    let Some(lod_settings) = lod_settings else {
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                        state.defer_bridge_activation_for(candidate);
                        continue;
                    };
                    // A pending blend must never arm activation until its
                    // exact initial suffix is radix-current and every current
                    // retained-view pressure pair has evaluated successfully.
                    // This catches view-dependent invalidity which can arise
                    // after runtime constructed the immutable edge table.
                    if !state.radix_sort_is_current() {
                        candidate
                            .phase
                            .store(LOD_RENDER_PREPARED, Ordering::Release);
                        state.defer_bridge_activation_for(candidate);
                        continue;
                    }
                    // Validate every pressure pair before priming any late edge.
                    // A valid table then publishes the live desired weight while
                    // retaining the exact authored displayed suffix. Invalid or
                    // Frozen tables remain untouched and fail closed.
                    let pressure_valid = state.preflight_view_blend_activation(
                        view,
                        transform,
                        lod_settings,
                        candidate,
                    );
                    let Ok(pressure_valid) = pressure_valid else {
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                        continue;
                    };
                    state.morph_activation_preflight_valid = pressure_valid
                        && debug_activation_ready
                        && state.has_current_drawable_bridge_candidate(candidate);
                    if pressure_valid
                        && state
                            .prime_initial_recovery_view_blend_desired(
                                view,
                                transform,
                                lod_settings,
                                candidate,
                            )
                            .is_err()
                    {
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                        continue;
                    }
                    if state.capture_drawable_view_blend_snapshot().is_err() {
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                        continue;
                    }
                    if !pressure_valid {
                        candidate
                            .phase
                            .store(LOD_RENDER_PREPARED, Ordering::Release);
                        state.defer_bridge_activation_for(candidate);
                        continue;
                    }
                    // Consume the exact authored first draw only after it has
                    // been radix-proven. This may stage the next suffix, but
                    // Cleanup still publishes the just-promoted state and will
                    // not activate until its desired evaluation is complete.
                    if state
                        .update_view_blend_weights(
                            &render_queue,
                            view,
                            transform,
                            lod_settings,
                            candidate,
                        )
                        .is_err()
                    {
                        candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                        continue;
                    }
                    candidate
                        .phase
                        .store(LOD_RENDER_PREPARED, Ordering::Release);
                    state.defer_bridge_activation_for(candidate);
                    continue;
                }
                candidate
                    .phase
                    .store(LOD_RENDER_PREPARED, Ordering::Release);
                if multi_subview_candidate {
                    state.defer_bridge_activation_for(candidate);
                    if state.has_current_drawable_bridge_candidate(candidate) {
                        record_multi_subview_drawable_output(
                            &mut multi_subview_drawable_outputs,
                            candidate,
                            raster_gate.consumer_count,
                        );
                    }
                } else if debug_activation_ready {
                    state.arm_bridge_activation(candidate);
                }
            } else {
                candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
            }
        }
    }
    for (_, (candidate_phase, expected_consumers, ready_consumers)) in
        multi_subview_drawable_outputs
    {
        if multi_subview_activation_ready(expected_consumers, ready_consumers) {
            // Every private output is descriptor-current, drawable, and radix
            // current for this exact Arc identity. Each retained view owns its
            // private camera-conditioned weights and sorted output; only this
            // topology-capability publication is shared.
            publish_bridge_activation_after_radix(&candidate_phase);
        }
    }
    for (_, (candidate_phase, phase)) in cold_staging_updates {
        let current = candidate_phase.load(Ordering::Acquire);
        if current == LOD_RENDER_FAILED
            || matches!(current, LOD_RENDER_ACTIVE | LOD_RENDER_TRANSITIONING)
        {
            continue;
        }
        candidate_phase.store(phase, Ordering::Release);
    }
}

/// Publishes the exact state promoted by this frame's compaction/radix graph.
///
/// Prepare may stage a newer CPU suffix and cannot prove what the raster pass
/// will consume. Cleanup runs after the ordered Core3d graph, reduces every
/// private retained-view output for one shared candidate token, Release-
/// publishes the coherent package snapshot, and only then permits Morphing
/// PREPARED -> ACTIVE.
#[allow(clippy::type_complexity)]
fn publish_lod_view_blend_after_radix<R: PlanarSync>(
    buffers: Res<LodCompactionBuffers<R>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities), With<GaussianCamera>>,
    clouds: Query<(
        Entity,
        &R::PlanarTypeHandle,
        &GaussianLodSettings,
        Option<&LodRenderCandidates>,
    )>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    let mut publications = HashMap::<usize, LodViewBlendPublication<'_>>::new();
    for (view, visible_entities) in &views {
        let Some(visible_clouds) = visible_entities.get::<CloudVisibilityClass>() else {
            continue;
        };
        let camera = view.retained_view_entity.main_entity.id();
        for (render_entity, _) in &visible_clouds.entities_cpu_culling {
            let Ok((entity, handle, lod_settings, candidates)) = clouds.get(*render_entity) else {
                continue;
            };
            let Some((candidates, candidate)) = candidates
                .and_then(|candidates| candidates.by_camera.get(&camera).map(|c| (candidates, c)))
            else {
                continue;
            };
            if candidate.view_blend_mode() != Some(LodTemporalTransitionMode::Morphing) {
                continue;
            }
            let state = lod_compaction_asset_id(handle.handle().id(), Some(candidates)).and_then(
                |compaction_id| buffers.get(view.retained_view_entity, entity, compaction_id),
            );
            if record_drawable_view_blend_publication(
                &mut publications,
                candidate,
                state,
                lod_settings.selection_mode,
            )
            .is_err()
            {
                candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
            }
        }
    }

    for (_, publication) in publications {
        let candidate = publication.candidate;
        if candidate.phase.load(Ordering::Acquire) == LOD_RENDER_FAILED {
            continue;
        }
        if view_blend_publication_is_complete(&publication) {
            let activation_allowed = view_blend_publication_can_activate(&publication);
            if !publish_complete_view_blend_publication(&publication) {
                candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
                continue;
            }
            if activation_allowed {
                // Aggregate publication above completes its seqlock with a
                // Release store before this phase CAS. Main-world package
                // ownership can therefore never observe ACTIVE with older
                // endpoint/lag/invalid evidence.
                publish_bridge_activation_after_radix(&candidate.phase);
            }
        } else if !publish_incomplete_view_blend_hold(&publication) {
            candidate.phase.store(LOD_RENDER_FAILED, Ordering::Release);
        }
    }
}

/// Current render-world support for the optional annotation sidecar.
///
/// This is deliberately recomputed from the current extracted asset and
/// metadata rather than cached on the entity: a stale `Unsupported` result
/// from a replaced source must never bypass the new sidecar's activation gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LodDebugRenderCapability {
    Unknown,
    SupportedPending,
    Unsupported,
}

fn classify_lod_debug_render_capability<R: PlanarSync>(
    pipeline: &CloudPipeline<R>,
    render_device: &RenderDevice,
    gpu_clouds: &RenderAssets<R::GpuPlanarType>,
    handle: &R::PlanarTypeHandle,
    metadata: Option<&LodDebugMetadata>,
) -> LodDebugRenderCapability
where
    R::GpuPlanarType: GpuPlanarStorage,
{
    if pipeline.lod_debug_layout.is_none() {
        return LodDebugRenderCapability::Unsupported;
    }
    let Some(gpu_cloud) = gpu_clouds.get(handle.handle()) else {
        return LodDebugRenderCapability::Unknown;
    };
    let Some(metadata) = metadata else {
        return LodDebugRenderCapability::Unknown;
    };
    let record_count = metadata
        .sparse()
        .map(|sparse| sparse.record_count())
        .unwrap_or_else(|| metadata.records().len())
        .min(gpu_cloud.len());
    let byte_len = record_count
        .max(1)
        .checked_mul(std::mem::size_of::<
            crate::gaussian::lod_debug::LodDebugRecord,
        >())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(u64::MAX);
    let limits = render_device.limits();
    if byte_len > limits.max_storage_buffer_binding_size || byte_len > limits.max_buffer_size {
        LodDebugRenderCapability::Unsupported
    } else {
        LodDebugRenderCapability::SupportedPending
    }
}

const fn lod_debug_candidate_activation_ready(
    requires_metadata: bool,
    capability: LodDebugRenderCapability,
    debug_metadata_staged: bool,
    candidate_invariants_ready: bool,
) -> bool {
    if !requires_metadata {
        return true;
    }
    match capability {
        LodDebugRenderCapability::Unknown => false,
        LodDebugRenderCapability::SupportedPending => {
            debug_metadata_staged && candidate_invariants_ready
        }
        LodDebugRenderCapability::Unsupported => true,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LodBridgeAtlasDecision {
    RejectActive,
    RetainCurrent,
    SynchronizePending,
}

const fn lod_bridge_atlas_decision(
    requested_phase: u8,
    atlas_current: bool,
) -> LodBridgeAtlasDecision {
    if atlas_current {
        LodBridgeAtlasDecision::SynchronizePending
    } else if matches!(
        requested_phase,
        LOD_RENDER_ACTIVE | LOD_RENDER_TRANSITIONING
    ) {
        LodBridgeAtlasDecision::RejectActive
    } else {
        LodBridgeAtlasDecision::RetainCurrent
    }
}

fn lod_drawable_atlas_allocation_is_current(
    has_drawable_output: bool,
    drawable_allocation_epoch: Option<u64>,
    current_allocation_epoch: Option<u64>,
) -> bool {
    !has_drawable_output || drawable_allocation_epoch == current_allocation_epoch
}

fn lod_external_active_set_weights(
    view: &ExtractedView,
    transform: &GlobalTransform,
    presentation: &LodExternalActiveSetPresentation,
) -> Option<(f32, f32)> {
    let local_from_world = transform.to_matrix().inverse();
    if !local_from_world.is_finite() {
        return None;
    }
    let local_view = local_from_world.transform_point3(view.world_from_view.translation());
    if !local_view.is_finite() {
        return None;
    }
    presentation.opacity_weights(local_view.to_array())
}

/// Names the ordinary resident storage allocation independently from transient
/// package-atlas generations. The change tick is stable while the canonical
/// catalog bind group remains installed and changes when that GPU storage is
/// replaced.
fn lod_resident_catalog_epoch(storage_change_tick: u32) -> u64 {
    0x4c4f_4447_0000_0000_u64 | u64::from(storage_change_tick)
}

fn lod_resident_catalog_content_signature(storage_change_tick: u32, source_count: u32) -> u64 {
    let mut hasher = DefaultHasher::new();
    "lodge-resident-catalog".hash(&mut hasher);
    storage_change_tick.hash(&mut hasher);
    source_count.hash(&mut hasher);
    hasher.finish()
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
    #[cfg(feature = "morph_interpolate")] interpolate_writers: Query<
        (),
        With<GaussianInterpolate<R>>,
    >,
    #[cfg(feature = "morph_particles")] particle_writers: Query<(), With<ParticleBehaviorsHandle>>,
    clouds: Query<(
        Entity,
        &'static R::PlanarTypeHandle,
        Ref<'static, PlanarStorageBindGroup<R>>,
        &'static DynamicUniformIndex<CloudUniform>,
        &'static CloudSettings,
        &'static GlobalTransform,
    )>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    let (extracted_view, view_bind_group, view_offset, previous_view_offset) = view.into_inner();
    let Some(uniform_bind_group) = gaussian_uniforms.base_bind_group.as_ref() else {
        return;
    };

    for (entity, handle, cloud_bind_group, cloud_uniform_index, cloud_settings, transform) in
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

        #[cfg(feature = "morph_interpolate")]
        let has_interpolate = interpolate_writers.get(entity).is_ok();
        #[cfg(not(feature = "morph_interpolate"))]
        let has_interpolate = false;
        #[cfg(feature = "morph_particles")]
        let has_particles = particle_writers.get(entity).is_ok();
        #[cfg(not(feature = "morph_particles"))]
        let has_particles = false;
        if !lod_compaction_cache_allowed(has_interpolate, has_particles) {
            // These compute writers mutate positions/visibility in-place, so
            // neither the asset identity nor storage bind-group tick changes.
            state.mark_compute_input_dirty();
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
