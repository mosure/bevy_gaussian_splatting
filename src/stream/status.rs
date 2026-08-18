//! One cloud-facing LoD status assembled from source, selector, and commit state.
//!
//! Bridge, package, commit, and render resources retain their narrow
//! internal responsibilities. Applications and inspectors should normally read
//! [`GaussianLodStatus`] instead of inferring lifecycle or quality from those
//! implementation-specific components.

use bevy::prelude::*;

use crate::{
    CloudSettings,
    gaussian::{
        lod_debug::{LodDebugMetadata, LodDebugPreset},
        lod_settings::{
            GaussianLodSettings, LodDegradation, LodQualityEndpoint, LodQualityTarget,
            LodSelectionMode,
        },
    },
    stream::{
        bridge::{GaussianLodBridgePhase, GaussianLodBridgeStatus},
        package::{GaussianLodPackagePhase, GaussianLodPackageSource, GaussianLodPackageStatus},
        render_commit::{
            LodOrchestrationFailure, LodOrchestrationFailureCategory, LodOrchestrationFailureCode,
            LodRenderCandidates,
        },
    },
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Reflect)]
pub enum GaussianLodSourceKind {
    /// The ordinary flat/original cloud path is active.
    #[default]
    Original,
    /// A bounded hierarchy was built from an already resident flat cloud.
    Ephemeral,
    /// A prebuilt, independently addressable LoD package is active.
    Package,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Reflect)]
pub enum GaussianLodLifecycle {
    /// Exact original records are intentionally used; this is not a failure.
    #[default]
    Original,
    Building,
    Streaming,
    WaitingForRender,
    Active,
    Degraded,
    Fallback,
    Failed,
}

/// Main-world truth about the requested debug view.
///
/// Adapter-specific bind-group readiness remains a render-world diagnostic;
/// this enum never claims that an annotation was drawn when that cannot be
/// observed from the main world.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Reflect)]
pub enum GaussianLodDebugAvailability {
    #[default]
    Disabled,
    /// The exact flat endpoint owns no hierarchy metadata.
    UnavailableOriginalEndpoint,
    /// Hierarchy metadata has been requested but is not published yet.
    WaitingForMetadata,
    /// Metadata is present. Adapter-specific pipeline readiness remains a
    /// render-world diagnostic.
    MetadataReady,
}

/// Coherent per-cloud LoD snapshot for inspectors and application diagnostics.
#[derive(Component, Clone, Debug, PartialEq, Reflect)]
#[reflect(Component)]
pub struct GaussianLodStatus {
    /// Monotonic per-entity revision; unchanged observations retain it.
    pub revision: u64,
    pub source: GaussianLodSourceKind,
    pub lifecycle: GaussianLodLifecycle,
    /// Requested camera policy for hierarchy selection.
    pub selection_mode: LodSelectionMode,
    /// Number of committed views whose candidates use a captured selection
    /// camera. Zero means no frozen LoD view has reached render commit yet. A
    /// flat original source has no hierarchy view to capture; a prebuilt
    /// package can still publish its exact leaf frontier. Residency itself is
    /// never frozen.
    pub frozen_views: u32,
    pub requested_target: LodQualityTarget,
    /// Worst selected error over active views, when a selector result exists.
    pub achieved_max_error_px: Option<f32>,
    /// Worst effective guarded balanced-target pressure over active views,
    /// including projected error and the builder-authored fidelity certificate.
    pub achieved_max_target_ratio: Option<f32>,
    pub target_satisfied: Option<bool>,
    pub degradation: LodDegradation,
    pub active_views: u32,
    /// Maximum selector-frontier count over active views.
    pub selected_gaussians: u64,
    /// Maximum pending commit candidate count over active views.
    pub submitted_candidates: u32,
    pub resident_pages: u32,
    pub debug_preset: LodDebugPreset,
    pub debug_availability: GaussianLodDebugAvailability,
    pub failure: Option<LodOrchestrationFailure>,
}

impl GaussianLodStatus {
    fn same_observation(&self, other: &Self) -> bool {
        self.source == other.source
            && self.lifecycle == other.lifecycle
            && self.selection_mode == other.selection_mode
            && self.frozen_views == other.frozen_views
            && self.requested_target == other.requested_target
            && self.achieved_max_error_px == other.achieved_max_error_px
            && self.achieved_max_target_ratio == other.achieved_max_target_ratio
            && self.target_satisfied == other.target_satisfied
            && self.degradation == other.degradation
            && self.active_views == other.active_views
            && self.selected_gaussians == other.selected_gaussians
            && self.submitted_candidates == other.submitted_candidates
            && self.resident_pages == other.resident_pages
            && self.debug_preset == other.debug_preset
            && self.debug_availability == other.debug_availability
            && self.failure == other.failure
    }
}

#[derive(Default)]
pub struct GaussianLodStatusPlugin;

impl Plugin for GaussianLodStatusPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<GaussianLodSourceKind>()
            .register_type::<GaussianLodLifecycle>()
            .register_type::<GaussianLodDebugAvailability>()
            .register_type::<LodOrchestrationFailureCategory>()
            .register_type::<LodOrchestrationFailureCode>()
            .register_type::<LodOrchestrationFailure>()
            .register_type::<GaussianLodStatus>()
            .add_systems(Last, publish_gaussian_lod_status);
    }
}

#[allow(clippy::type_complexity)]
fn publish_gaussian_lod_status(
    mut commands: Commands,
    stale_statuses: Query<Entity, (With<GaussianLodStatus>, Without<GaussianLodSettings>)>,
    clouds: Query<(
        Entity,
        &GaussianLodSettings,
        Option<&CloudSettings>,
        Option<&GaussianLodPackageSource>,
        Option<&GaussianLodBridgeStatus>,
        Option<&GaussianLodPackageStatus>,
        Option<&LodRenderCandidates>,
        Option<&LodDebugMetadata>,
        Option<&GaussianLodStatus>,
    )>,
) {
    for entity in &stale_statuses {
        commands.entity(entity).remove::<GaussianLodStatus>();
    }

    for (
        entity,
        settings,
        cloud,
        package_source,
        bridge,
        package,
        candidates,
        debug_metadata,
        previous,
    ) in &clouds
    {
        let source = if package_source.is_some() {
            GaussianLodSourceKind::Package
        } else if bridge.is_some() {
            GaussianLodSourceKind::Ephemeral
        } else {
            GaussianLodSourceKind::Original
        };
        let (lifecycle, resident_pages, failure) = lifecycle(source, bridge, package);

        let mut achieved_max_error_px: Option<f32> = None;
        let mut achieved_max_target_ratio: Option<f32> = None;
        let mut degradation = LodDegradation::None;
        let mut selected_gaussians = 0_u64;
        let mut submitted_candidates = 0_u32;
        let mut active_views = 0_u32;
        let mut frozen_views = 0_u32;
        if let Some(candidates) = candidates {
            active_views = candidates.len().try_into().unwrap_or(u32::MAX);
            for candidate in candidates.by_camera.values() {
                if candidate.frontier().selection_view_frozen() {
                    frozen_views = frozen_views.saturating_add(1);
                }
                let quality = candidate.rendered_quality_status();
                achieved_max_error_px = Some(
                    achieved_max_error_px
                        .unwrap_or(0.0)
                        .max(quality.achieved_max_error_px),
                );
                achieved_max_target_ratio = Some(
                    achieved_max_target_ratio
                        .unwrap_or(0.0)
                        .max(quality.achieved_max_target_ratio),
                );
                degradation = degradation.merge(quality.degradation);
                selected_gaussians = selected_gaussians.max(quality.active_gaussians);
                if candidate.requires_compaction() {
                    submitted_candidates =
                        submitted_candidates.max(candidate.rendered_candidate_count());
                }
            }
        }
        if active_views == 0 {
            active_views = bridge.map_or(0, |status| status.active_views);
        }
        if selected_gaussians == 0 {
            selected_gaussians = bridge
                .map(|status| status.active_gaussians)
                .or_else(|| package.map(|status| status.active_gaussians))
                .unwrap_or(0);
        }

        let requested_target = settings.quality_target();
        let target_satisfied =
            lod_target_satisfied(achieved_max_target_ratio, degradation, selected_gaussians);
        let debug_preset = cloud
            .map(|settings| settings.lod_debug.preset)
            .unwrap_or_default();
        let debug_requested = cloud.is_some_and(|settings| settings.lod_debug.requires_metadata());
        let debug_availability = if !debug_requested {
            GaussianLodDebugAvailability::Disabled
        } else if source == GaussianLodSourceKind::Original
            && settings.quality_endpoint() == LodQualityEndpoint::Original
            && debug_metadata.is_none_or(LodDebugMetadata::is_empty)
        {
            GaussianLodDebugAvailability::UnavailableOriginalEndpoint
        } else if debug_metadata.is_none_or(LodDebugMetadata::is_empty) {
            GaussianLodDebugAvailability::WaitingForMetadata
        } else {
            GaussianLodDebugAvailability::MetadataReady
        };

        let mut next = GaussianLodStatus {
            revision: previous.map_or(1, |status| status.revision),
            source,
            lifecycle,
            selection_mode: settings.selection_mode,
            frozen_views,
            requested_target,
            achieved_max_error_px,
            achieved_max_target_ratio,
            target_satisfied,
            degradation,
            active_views,
            selected_gaussians,
            submitted_candidates,
            resident_pages,
            debug_preset,
            debug_availability,
            failure,
        };
        if previous.is_some_and(|status| status.same_observation(&next)) {
            continue;
        }
        if let Some(previous) = previous {
            next.revision = previous.revision.saturating_add(1);
        }
        commands.entity(entity).insert(next);
    }
}

fn lod_target_satisfied(
    achieved_max_target_ratio: Option<f32>,
    degradation: LodDegradation,
    selected_gaussians: u64,
) -> Option<bool> {
    achieved_max_target_ratio.map(|ratio| {
        ratio <= 1.0 && (selected_gaussians > 0 || degradation == LodDegradation::None)
    })
}

fn lifecycle(
    source: GaussianLodSourceKind,
    bridge: Option<&GaussianLodBridgeStatus>,
    package: Option<&GaussianLodPackageStatus>,
) -> (GaussianLodLifecycle, u32, Option<LodOrchestrationFailure>) {
    if let Some(status) = package {
        let lifecycle = match status.phase {
            GaussianLodPackagePhase::Loading => GaussianLodLifecycle::Streaming,
            GaussianLodPackagePhase::Active => GaussianLodLifecycle::Active,
            GaussianLodPackagePhase::Degraded => GaussianLodLifecycle::Degraded,
            GaussianLodPackagePhase::Failed => GaussianLodLifecycle::Failed,
        };
        return (lifecycle, status.resident_pages, status.failure.clone());
    }
    if let Some(status) = bridge {
        let lifecycle = match status.phase {
            GaussianLodBridgePhase::Building => GaussianLodLifecycle::Building,
            GaussianLodBridgePhase::StreamingFallback => GaussianLodLifecycle::Degraded,
            GaussianLodBridgePhase::WaitingForRender => GaussianLodLifecycle::WaitingForRender,
            GaussianLodBridgePhase::Active => GaussianLodLifecycle::Active,
            GaussianLodBridgePhase::CompleteFallback => GaussianLodLifecycle::Fallback,
        };
        return (lifecycle, status.resident_pages, status.failure.clone());
    }
    let lifecycle = match source {
        GaussianLodSourceKind::Original => GaussianLodLifecycle::Original,
        GaussianLodSourceKind::Ephemeral => GaussianLodLifecycle::Building,
        GaussianLodSourceKind::Package => GaussianLodLifecycle::Streaming,
    };
    (lifecycle, 0, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_degraded_cut_never_claims_to_meet_the_quality_target() {
        assert_eq!(
            lod_target_satisfied(Some(0.0), LodDegradation::Residency, 0),
            Some(false)
        );
        assert_eq!(
            lod_target_satisfied(Some(0.75), LodDegradation::None, 4),
            Some(true)
        );
        assert_eq!(
            lod_target_satisfied(Some(0.0), LodDegradation::None, 0),
            Some(true),
            "an empty but non-degraded frustum can satisfy the target"
        );
    }

    #[test]
    fn original_endpoint_debug_has_an_explicit_unavailable_state() {
        let mut app = App::new();
        app.add_plugins(GaussianLodStatusPlugin);
        let mut cloud = CloudSettings::default();
        cloud.lod_debug.apply_preset(crate::LodDebugPreset::Page);
        let entity = app
            .world_mut()
            .spawn((GaussianLodSettings::default(), cloud))
            .id();
        app.update();
        let status = app.world().get::<GaussianLodStatus>(entity).unwrap();
        assert_eq!(status.source, GaussianLodSourceKind::Original);
        assert_eq!(status.lifecycle, GaussianLodLifecycle::Original);
        assert_eq!(status.selection_mode, LodSelectionMode::Dynamic);
        assert_eq!(status.frozen_views, 0);
        assert_eq!(status.debug_preset, LodDebugPreset::Page);
        assert_eq!(
            status.debug_availability,
            GaussianLodDebugAvailability::UnavailableOriginalEndpoint
        );
        assert_eq!(status.requested_target, LodQualityTarget::Original);
    }

    #[test]
    fn frozen_selection_mode_is_published_and_reflected_while_capture_is_pending() {
        let mut app = App::new();
        app.add_plugins(GaussianLodStatusPlugin);
        let settings = GaussianLodSettings {
            quality: 0.5,
            selection_mode: LodSelectionMode::Frozen,
            ..Default::default()
        };
        let entity = app
            .world_mut()
            .spawn((settings, CloudSettings::default()))
            .id();
        app.update();

        let status = app.world().get::<GaussianLodStatus>(entity).unwrap();
        assert_eq!(status.source, GaussianLodSourceKind::Original);
        assert_eq!(status.lifecycle, GaussianLodLifecycle::Original);
        assert_eq!(status.debug_preset, LodDebugPreset::Off);
        assert_eq!(status.selection_mode, LodSelectionMode::Frozen);
        assert_eq!(status.frozen_views, 0);
        assert_eq!(
            status.requested_target,
            LodQualityTarget::Balanced {
                detail_fraction: 0.5,
                max_error_px: 2.0,
            }
        );
        assert_eq!(status.achieved_max_error_px, None);
        assert_eq!(status.achieved_max_target_ratio, None);
        assert_eq!(status.target_satisfied, None);
        let reflected = status.reflect_ref().as_struct().unwrap();
        assert_eq!(
            reflected
                .field("selection_mode")
                .and_then(|value| value.try_downcast_ref::<LodSelectionMode>()),
            Some(&LodSelectionMode::Frozen)
        );
        assert!(reflected.field("frozen_views").is_some());
    }

    #[test]
    fn unchanged_status_does_not_churn_revision() {
        let mut app = App::new();
        app.add_plugins(GaussianLodStatusPlugin);
        let entity = app
            .world_mut()
            .spawn((GaussianLodSettings::default(), CloudSettings::default()))
            .id();
        app.update();
        let first = app
            .world()
            .get::<GaussianLodStatus>(entity)
            .unwrap()
            .revision;
        app.update();
        assert_eq!(
            app.world()
                .get::<GaussianLodStatus>(entity)
                .unwrap()
                .revision,
            first
        );
    }

    #[test]
    fn removing_lod_settings_removes_the_unified_status() {
        let mut app = App::new();
        app.add_plugins(GaussianLodStatusPlugin);
        let entity = app
            .world_mut()
            .spawn((GaussianLodSettings::default(), CloudSettings::default()))
            .id();
        app.update();
        assert!(app.world().get::<GaussianLodStatus>(entity).is_some());

        app.world_mut()
            .entity_mut(entity)
            .remove::<GaussianLodSettings>();
        app.update();
        assert!(app.world().get::<GaussianLodStatus>(entity).is_none());
    }

    #[test]
    fn published_debug_metadata_reports_ready_without_claiming_a_draw() {
        let mut app = App::new();
        app.add_plugins(GaussianLodStatusPlugin);
        let settings = GaussianLodSettings {
            quality: 0.5,
            ..Default::default()
        };
        let mut cloud = CloudSettings::default();
        cloud.lod_debug.apply_preset(crate::LodDebugPreset::Level);
        let entity = app
            .world_mut()
            .spawn((
                settings,
                cloud,
                LodDebugMetadata::new(vec![crate::gaussian::lod_debug::LodDebugRecord::default()]),
            ))
            .id();
        app.update();
        let status = app.world().get::<GaussianLodStatus>(entity).unwrap();
        assert_eq!(status.source, GaussianLodSourceKind::Original);
        assert_eq!(
            status.debug_availability,
            GaussianLodDebugAvailability::MetadataReady
        );
    }
}
