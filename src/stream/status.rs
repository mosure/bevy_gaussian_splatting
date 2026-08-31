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
    io::lodge::GaussianLodgeHandle,
    stream::{
        bridge::{GaussianLodBridgePhase, GaussianLodBridgeStatus},
        package::{GaussianLodPackagePhase, GaussianLodPackageSource, GaussianLodPackageStatus},
        render_commit::{
            LodOrchestrationFailure, LodOrchestrationFailureCategory, LodOrchestrationFailureCode,
            LodRenderCandidate, LodRenderCandidates,
        },
        runtime::LodTemporalTransitionMode,
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
    /// Whether the currently drawable presentation satisfies the live target.
    /// Render-owned view-blend catch-up, missing-consumer barriers, and invalid
    /// pressure holds report `Some(false)` even when the selector's structural
    /// frontier is exact.
    pub target_satisfied: Option<bool>,
    pub degradation: LodDegradation,
    pub active_views: u32,
    /// Maximum complete, scene-wide selector-frontier count over active views.
    /// This includes off-frustum representatives retained for global coverage;
    /// it is not the number of splats emitted by the indirect draw.
    pub selected_gaussians: u64,
    /// Maximum candidate count passed to GPU compaction over active views,
    /// before per-splat frustum and visibility rejection. The exact indirect
    /// draw count remains render-world/GPU state and is not synchronously read
    /// back into this main-world status.
    pub submitted_candidates: u32,
    /// Total decoded page-cache occupancy, including the permanent coverage
    /// guard and warm unpinned pages. This is not the current draw-cut size.
    pub resident_pages: u32,
    /// Total camera-conditioned adjacent hierarchy edges currently presented
    /// by ACTIVE views. A nonzero count is a stable presentation capability,
    /// not evidence of a timed topology transition.
    pub view_blend_edges: u32,
    /// Edges whose displayed weight is catching up to the exact current-view
    /// pressure weight. This is normally zero; nonzero values are reserved for
    /// late-residency activation, Dynamic resumption after Frozen mode, and
    /// recovery after an invalid-pressure hold.
    pub view_blend_lagging_edges: u32,
    /// ACTIVE immutable edges whose pressure is invalid in at least one private
    /// view. Multi-view reduction counts an edge once. Render retains the last
    /// drawable weights while nonzero; this explicit degraded hold never
    /// satisfies the live target.
    pub view_blend_invalid_pressure_evaluations: u32,
    /// Expected private render consumers which do not yet have a coherent
    /// radix-proven snapshot for the ACTIVE blend table. Render Cleanup cannot
    /// publish the all-consumer aggregate barrier while this is nonzero.
    pub view_blend_missing_consumers: u32,
    /// Largest absolute displayed-versus-desired edge-weight difference.
    pub view_blend_max_lag: f32,
    /// Largest displayed edge-weight change in the most recently published
    /// drawable frame.
    pub view_blend_max_delta: f32,
    /// Sum of per-edge absolute weight change times mapped record count for the
    /// most recently published drawable frame.
    pub view_blend_weighted_record_energy: f32,
    /// Compatibility classification for a currently pending topology handoff.
    /// ACTIVE camera-conditioned blending is reported by the `view_blend_*`
    /// fields instead, so `None` still means no handoff is blocking the visible
    /// cut.
    pub temporal_transition_mode: Option<LodTemporalTransitionMode>,
    /// Legacy timed-transition progress. Camera-conditioned view blending has
    /// no global progress scalar and reports `None` here.
    pub temporal_transition_progress: Option<f32>,
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
            && self.view_blend_edges == other.view_blend_edges
            && self.view_blend_lagging_edges == other.view_blend_lagging_edges
            && self.view_blend_invalid_pressure_evaluations
                == other.view_blend_invalid_pressure_evaluations
            && self.view_blend_missing_consumers == other.view_blend_missing_consumers
            && self.view_blend_max_lag == other.view_blend_max_lag
            && self.view_blend_max_delta == other.view_blend_max_delta
            && self.view_blend_weighted_record_energy == other.view_blend_weighted_record_energy
            && self.temporal_transition_mode == other.temporal_transition_mode
            && self.temporal_transition_progress == other.temporal_transition_progress
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
            .register_type::<LodTemporalTransitionMode>()
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
    stale_statuses: Query<
        Entity,
        (
            With<GaussianLodStatus>,
            Or<(Without<GaussianLodSettings>, With<GaussianLodgeHandle>)>,
        ),
    >,
    clouds: Query<
        (
            Entity,
            &GaussianLodSettings,
            Option<&CloudSettings>,
            Option<&GaussianLodPackageSource>,
            Option<&GaussianLodBridgeStatus>,
            Option<&GaussianLodPackageStatus>,
            Option<&LodRenderCandidates>,
            Option<&LodDebugMetadata>,
            Option<&GaussianLodStatus>,
        ),
        Without<GaussianLodgeHandle>,
    >,
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
        let (mut lifecycle, resident_pages, mut failure) = lifecycle(source, bridge, package);

        let mut achieved_max_error_px: Option<f32> = None;
        let mut achieved_max_target_ratio: Option<f32> = None;
        let mut degradation = LodDegradation::None;
        let mut selected_gaussians = 0_u64;
        let mut submitted_candidates = 0_u32;
        let mut active_views = 0_u32;
        let mut frozen_views = 0_u32;
        let mut view_blend_edges = 0_u32;
        let mut view_blend_lagging_edges = 0_u32;
        let mut view_blend_invalid_pressure_evaluations = 0_u32;
        let mut view_blend_missing_consumers = 0_u32;
        let mut view_blend_max_lag = 0.0_f32;
        let mut view_blend_max_delta = 0.0_f32;
        let mut view_blend_weighted_record_energy = 0.0_f32;
        let mut temporal_transition_mode = None;
        let mut temporal_transition_progress: Option<f32> = None;
        if let Some(candidates) = candidates {
            for candidate in candidates.by_camera.values() {
                // This compatibility field describes only a pending topology
                // handoff. ACTIVE adjacent-edge blending is stable presentation
                // state and is exposed through the aggregate view-blend fields
                // below, never as an indefinitely pending transition.
                if !candidate.render_is_prepared() || candidate.render_is_active() {
                    continue;
                }
                let Some(mode) = candidate.temporal_transition_mode() else {
                    continue;
                };
                temporal_transition_mode = Some(match (temporal_transition_mode, mode) {
                    (Some(LodTemporalTransitionMode::Morphing), _)
                    | (_, LodTemporalTransitionMode::Morphing) => {
                        LodTemporalTransitionMode::Morphing
                    }
                    _ => LodTemporalTransitionMode::BoundedHardCohort,
                });
                if let Some(progress) = candidate.temporal_transition_progress() {
                    temporal_transition_progress = Some(
                        temporal_transition_progress
                            .map_or(progress, |current| current.min(progress)),
                    );
                }
            }
        }
        let requested_target = settings.quality_target();
        let mut visible_candidate_targets_match_request = true;
        let orchestration_retains_active_cut = bridge
            .is_some_and(|status| status.phase == GaussianLodBridgePhase::Active)
            || package.is_some_and(|status| {
                matches!(
                    status.phase,
                    GaussianLodPackagePhase::Active | GaussianLodPackagePhase::Degraded
                )
            });
        let candidate_set_is_visible = candidates.is_none_or(|candidates| {
            !orchestration_retains_active_cut
                || candidates
                    .by_camera
                    .values()
                    .all(LodRenderCandidate::render_is_active)
        });
        let package_retains_visible_metrics_during_pending_replacement = source
            == GaussianLodSourceKind::Package
            && package.is_some_and(|status| {
                matches!(
                    status.phase,
                    GaussianLodPackagePhase::Active | GaussianLodPackagePhase::Degraded
                )
            })
            && candidates.is_some_and(|candidates| {
                candidates.retained_current
                    && !candidates.candidates_are_current
                    && candidates
                        .by_camera
                        .values()
                        .any(|candidate| !candidate.render_is_active())
            });
        if let Some(candidates) = candidates.filter(|_| candidate_set_is_visible) {
            active_views = candidates.len().try_into().unwrap_or(u32::MAX);
            for candidate in candidates.by_camera.values() {
                if candidate.frontier().selection_view_frozen() {
                    frozen_views = frozen_views.saturating_add(1);
                }
                let quality = candidate.rendered_quality_status();
                visible_candidate_targets_match_request &= quality.requested_target
                    == requested_target
                    && !candidates.retained_current_is_stale;
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
                submitted_candidates =
                    submitted_candidates.max(candidate.rendered_candidate_count());
                if let Some(blend) = candidate.view_blend_status() {
                    view_blend_edges = view_blend_edges.saturating_add(blend.edge_count);
                    view_blend_lagging_edges =
                        view_blend_lagging_edges.saturating_add(blend.lagging_count);
                    view_blend_invalid_pressure_evaluations =
                        view_blend_invalid_pressure_evaluations
                            .saturating_add(blend.invalid_pressure_count);
                    view_blend_missing_consumers =
                        view_blend_missing_consumers.saturating_add(blend.missing_consumer_count);
                    view_blend_max_lag = view_blend_max_lag.max(blend.max_lag);
                    view_blend_max_delta = view_blend_max_delta.max(blend.max_delta);
                    view_blend_weighted_record_energy = (view_blend_weighted_record_energy
                        + blend.weighted_record_energy)
                        .min(f32::MAX);
                }
            }
        }
        if active_views == 0 {
            active_views = bridge.map_or(0, |status| status.active_views);
        }
        if package_retains_visible_metrics_during_pending_replacement {
            // `by_camera` describes the non-ACTIVE replacement in this exact
            // state, while the package still presents its previously committed
            // cut. Preserve only count-like facts from that coherent package
            // observation. In particular, do not attribute the pending cut's
            // achieved quality or degradation to the retained presentation.
            if let Some(retained) = previous.filter(|status| {
                status.source == GaussianLodSourceKind::Package
                    && matches!(
                        status.lifecycle,
                        GaussianLodLifecycle::Active | GaussianLodLifecycle::Degraded
                    )
            }) {
                active_views = retained.active_views;
                frozen_views = retained.frozen_views;
                submitted_candidates = retained.submitted_candidates;
                view_blend_edges = retained.view_blend_edges;
                view_blend_lagging_edges = retained.view_blend_lagging_edges;
                view_blend_invalid_pressure_evaluations =
                    retained.view_blend_invalid_pressure_evaluations;
                view_blend_missing_consumers = retained.view_blend_missing_consumers;
                view_blend_max_lag = retained.view_blend_max_lag;
                view_blend_max_delta = retained.view_blend_max_delta;
                view_blend_weighted_record_energy = retained.view_blend_weighted_record_energy;
            }
        }
        if selected_gaussians == 0 {
            selected_gaussians = bridge
                .map(|status| status.active_gaussians)
                .or_else(|| package.map(|status| status.active_gaussians))
                .unwrap_or(0);
        }
        if active_views > 0 && !visible_candidate_targets_match_request {
            // A retained cut remains the drawable truth while a replacement is
            // selected and committed. Its counts and degradation still
            // describe that cut, but its achieved quality cannot satisfy a
            // different request.
            achieved_max_error_px = None;
            achieved_max_target_ratio = None;
        }

        if view_blend_invalid_pressure_evaluations != 0 {
            lifecycle = GaussianLodLifecycle::Degraded;
            failure.get_or_insert_with(|| {
                LodOrchestrationFailure::with_detail(
                    LodOrchestrationFailureCode::UnsupportedConfiguration,
                    "camera-conditioned LoD pressure is invalid; retaining the last drawable blend weights until a valid evaluation recovers",
                )
            });
        }

        let target_satisfied = lod_target_satisfied(
            achieved_max_target_ratio,
            degradation,
            selected_gaussians,
            view_blend_lagging_edges,
            view_blend_invalid_pressure_evaluations,
            view_blend_missing_consumers,
        );
        let debug_preset = cloud
            .map(|settings| settings.lod_debug.preset)
            .unwrap_or_default();
        let debug_requested = cloud.is_some_and(|settings| settings.lod_debug.requires_metadata());
        let debug_metadata_missing_or_incomplete =
            debug_metadata.is_none_or(|metadata| metadata.is_empty() || !metadata.is_complete());
        let debug_availability = if !debug_requested {
            GaussianLodDebugAvailability::Disabled
        } else if source == GaussianLodSourceKind::Original
            && settings.quality_endpoint() == LodQualityEndpoint::Original
            && debug_metadata_missing_or_incomplete
        {
            GaussianLodDebugAvailability::UnavailableOriginalEndpoint
        } else if debug_metadata_missing_or_incomplete {
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
            view_blend_edges,
            view_blend_lagging_edges,
            view_blend_invalid_pressure_evaluations,
            view_blend_missing_consumers,
            view_blend_max_lag,
            view_blend_max_delta,
            view_blend_weighted_record_energy,
            temporal_transition_mode,
            temporal_transition_progress,
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
    lagging_view_blend_edges: u32,
    invalid_pressure_evaluations: u32,
    missing_view_blend_consumers: u32,
) -> Option<bool> {
    if lagging_view_blend_edges != 0
        || invalid_pressure_evaluations != 0
        || missing_view_blend_consumers != 0
    {
        return Some(false);
    }
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
    use std::sync::Arc;

    use crate::{
        gaussian::formats::planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
        io::lod::encode_page,
        stream::{
            hierarchy::LodView,
            runtime::{LodCandidateFrontier, LodStreamingRuntime},
            transport::MemoryPageTransport,
        },
        testing::LodTestScene,
    };

    #[test]
    fn empty_degraded_cut_never_claims_to_meet_the_quality_target() {
        assert_eq!(
            lod_target_satisfied(Some(0.0), LodDegradation::Residency, 0, 0, 0, 0),
            Some(false)
        );
        assert_eq!(
            lod_target_satisfied(Some(0.75), LodDegradation::None, 4, 0, 0, 0),
            Some(true)
        );
        assert_eq!(
            lod_target_satisfied(Some(0.0), LodDegradation::None, 0, 0, 0, 0),
            Some(true),
            "an empty but non-degraded frustum can satisfy the target"
        );
        assert_eq!(
            lod_target_satisfied(Some(0.5), LodDegradation::None, 4, 0, 1, 0),
            Some(false),
            "an invalid ACTIVE pressure evaluation cannot masquerade as target satisfaction"
        );
    }

    #[test]
    fn view_blend_lag_is_not_reported_as_visual_target_satisfaction() {
        assert_eq!(
            lod_target_satisfied(Some(0.5), LodDegradation::None, 4, 1, 0, 0),
            Some(false),
            "an exact selector frontier remains visually incomplete while a displayed weight catches up"
        );
        assert_eq!(
            lod_target_satisfied(None, LodDegradation::None, 0, 1, 0, 0),
            Some(false),
            "observable blend lag is a negative verdict even before current quality is available"
        );
        assert_eq!(
            lod_target_satisfied(None, LodDegradation::None, 0, 0, 0, 0),
            None,
            "without lag or current quality the status must remain unknown"
        );
        assert_eq!(
            lod_target_satisfied(Some(0.5), LodDegradation::None, 4, 0, 0, 0),
            Some(true),
            "converged camera-conditioned presentation preserves the structural quality verdict"
        );
    }

    #[test]
    fn missing_view_blend_consumers_block_visual_target_satisfaction() {
        assert_eq!(
            lod_target_satisfied(Some(0.5), LodDegradation::None, 4, 0, 0, 1),
            Some(false),
            "an ACTIVE table is incomplete until every expected private consumer is drawable"
        );
        assert_eq!(
            lod_target_satisfied(None, LodDegradation::None, 0, 0, 0, 2),
            Some(false),
            "observable missing consumers are a negative verdict even before quality is available"
        );
        assert_eq!(
            lod_target_satisfied(None, LodDegradation::None, 0, 0, 0, 0),
            None,
            "quality remains unknown when there is no lag, invalid pressure, or missing consumer"
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

    #[test]
    fn incomplete_sparse_debug_metadata_remains_preparing() {
        let mut app = App::new();
        app.add_plugins(GaussianLodStatusPlugin);
        let settings = GaussianLodSettings {
            quality: 0.5,
            ..Default::default()
        };
        let mut cloud = CloudSettings::default();
        cloud.lod_debug.apply_preset(crate::LodDebugPreset::Level);
        let mut atlas =
            crate::gaussian::lod_debug::LodDebugAnnotationAtlas::new_sparse(8, 16).unwrap();
        atlas.set_complete(false);
        let entity = app
            .world_mut()
            .spawn((settings, cloud, atlas.metadata()))
            .id();

        app.update();
        assert_eq!(
            app.world()
                .get::<GaussianLodStatus>(entity)
                .unwrap()
                .debug_availability,
            GaussianLodDebugAvailability::WaitingForMetadata
        );
    }

    #[test]
    fn retained_active_cut_hides_pending_candidate_quality_until_activation() {
        let (settings, frontier) = resident_status_test_frontier();
        let frontier =
            frontier.with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing);
        let camera = Entity::from_bits(7);
        let mut candidates = LodRenderCandidates::default();
        candidates.insert(camera, frontier);
        let phase = Arc::clone(&candidates.get(camera).unwrap().phase);
        phase.store(
            crate::stream::render_commit::LOD_RENDER_PREPARED,
            std::sync::atomic::Ordering::Release,
        );
        let retained_count = u64::from(candidates.get(camera).unwrap().rendered_candidate_count())
            .saturating_add(11);

        let mut app = App::new();
        app.add_plugins(GaussianLodStatusPlugin);
        let entity = app
            .world_mut()
            .spawn((
                settings,
                CloudSettings::default(),
                GaussianLodBridgeStatus {
                    phase: GaussianLodBridgePhase::Active,
                    active_views: 2,
                    resident_pages: 3,
                    active_gaussians: retained_count,
                    failure: None,
                },
                candidates,
            ))
            .id();

        app.update();
        let pending = app.world().get::<GaussianLodStatus>(entity).unwrap();
        assert_eq!(pending.lifecycle, GaussianLodLifecycle::Active);
        assert_eq!(pending.active_views, 2);
        assert_eq!(pending.selected_gaussians, retained_count);
        assert_eq!(pending.achieved_max_error_px, None);
        assert_eq!(pending.achieved_max_target_ratio, None);
        assert_eq!(pending.degradation, LodDegradation::None);
        assert_eq!(pending.submitted_candidates, 0);
        assert_eq!(
            pending.temporal_transition_mode,
            Some(LodTemporalTransitionMode::Morphing)
        );
        assert_eq!(pending.temporal_transition_progress, None);

        app.world()
            .get::<LodRenderCandidates>(entity)
            .unwrap()
            .get(camera)
            .unwrap()
            .publish_temporal_transition_mode(LodTemporalTransitionMode::BoundedHardCohort);
        app.update();
        let fallback = app.world().get::<GaussianLodStatus>(entity).unwrap();
        assert_eq!(
            fallback.temporal_transition_mode,
            Some(LodTemporalTransitionMode::BoundedHardCohort)
        );
        assert_eq!(fallback.temporal_transition_progress, None);

        phase.store(
            crate::stream::render_commit::LOD_RENDER_ACTIVE,
            std::sync::atomic::Ordering::Release,
        );
        app.update();
        let active = app.world().get::<GaussianLodStatus>(entity).unwrap();
        assert_eq!(active.active_views, 1);
        assert_ne!(active.selected_gaussians, retained_count);
        assert!(active.achieved_max_error_px.is_some());
        assert!(active.achieved_max_target_ratio.is_some());
        assert!(active.submitted_candidates > 0);
        assert_eq!(active.temporal_transition_mode, None);
        assert_eq!(active.temporal_transition_progress, None);

        let active_selected = active.selected_gaussians;
        let active_submitted = active.submitted_candidates;
        let active_degradation = active.degradation;
        let replacement_target = {
            let mut settings = app
                .world_mut()
                .get_mut::<GaussianLodSettings>(entity)
                .unwrap();
            settings.quality = 0.5;
            settings.quality_target()
        };
        app.update();
        let stale = app.world().get::<GaussianLodStatus>(entity).unwrap();
        assert_eq!(stale.lifecycle, GaussianLodLifecycle::Active);
        assert_eq!(stale.requested_target, replacement_target);
        assert_ne!(
            stale.requested_target,
            candidates_quality_target(app.world(), entity, camera)
        );
        assert_eq!(stale.selected_gaussians, active_selected);
        assert_eq!(stale.submitted_candidates, active_submitted);
        assert_eq!(stale.degradation, active_degradation);
        assert_eq!(stale.achieved_max_error_px, None);
        assert_eq!(stale.achieved_max_target_ratio, None);
        assert_eq!(stale.target_satisfied, None);

        // Camera motion is a different request even when the slider target is
        // unchanged. Package orchestration marks that stronger identity on the
        // retained candidate metadata so the unified status cannot report the
        // old camera's achieved metrics as current.
        app.world_mut()
            .get_mut::<GaussianLodSettings>(entity)
            .unwrap()
            .quality = 0.0;
        app.world_mut()
            .get_mut::<LodRenderCandidates>(entity)
            .unwrap()
            .retained_current_is_stale = true;
        app.update();
        let moved = app.world().get::<GaussianLodStatus>(entity).unwrap();
        assert_eq!(
            moved.requested_target,
            candidates_quality_target(app.world(), entity, camera)
        );
        assert_eq!(moved.lifecycle, GaussianLodLifecycle::Active);
        assert_eq!(moved.achieved_max_error_px, None);
        assert_eq!(moved.achieved_max_target_ratio, None);
        assert_eq!(moved.target_satisfied, None);
    }

    #[test]
    fn package_status_retains_committed_counts_until_pending_cut_activates() {
        let (settings, frontier) = resident_status_test_frontier();
        let first_camera = Entity::from_bits(17);
        let second_camera = Entity::from_bits(18);
        let replacement_camera = Entity::from_bits(19);

        let mut current = LodRenderCandidates::package_required();
        current.insert(first_camera, frontier.clone());
        current.insert(second_camera, frontier.clone());
        current.retained_current = true;
        current.candidates_are_current = true;
        for candidate in current.by_camera.values() {
            candidate.phase.store(
                crate::stream::render_commit::LOD_RENDER_ACTIVE,
                std::sync::atomic::Ordering::Release,
            );
        }

        let active_gaussians = frontier.quality_status().active_gaussians;
        let mut app = App::new();
        app.add_plugins(GaussianLodStatusPlugin);
        let entity = app
            .world_mut()
            .spawn((
                settings,
                CloudSettings::default(),
                GaussianLodPackageSource::native_directory("."),
                GaussianLodPackageStatus {
                    phase: GaussianLodPackagePhase::Active,
                    resident_pages: 3,
                    active_gaussians,
                    terminal_failures: 0,
                    failure: None,
                },
                current,
            ))
            .id();

        app.update();
        let committed = app
            .world()
            .get::<GaussianLodStatus>(entity)
            .unwrap()
            .clone();
        assert_eq!(committed.source, GaussianLodSourceKind::Package);
        assert_eq!(committed.lifecycle, GaussianLodLifecycle::Active);
        assert_eq!(committed.active_views, 2);
        assert!(committed.submitted_candidates > 0);
        assert!(committed.achieved_max_error_px.is_some());

        let mut pending = LodRenderCandidates::package_required();
        pending.insert(replacement_camera, frontier);
        pending.retained_current = true;
        pending.candidates_are_current = false;
        let pending_phase = Arc::clone(&pending.get(replacement_camera).unwrap().phase);
        pending_phase.store(
            crate::stream::render_commit::LOD_RENDER_PREPARED,
            std::sync::atomic::Ordering::Release,
        );
        let replacement_submitted = pending
            .get(replacement_camera)
            .unwrap()
            .rendered_candidate_count();
        app.world_mut().entity_mut(entity).insert(pending);
        app.world_mut()
            .get_mut::<GaussianLodPackageStatus>(entity)
            .unwrap()
            .phase = GaussianLodPackagePhase::Degraded;

        app.update();
        let prepared = app.world().get::<GaussianLodStatus>(entity).unwrap();
        assert_eq!(prepared.lifecycle, GaussianLodLifecycle::Degraded);
        assert_eq!(prepared.active_views, committed.active_views);
        assert_eq!(
            prepared.submitted_candidates,
            committed.submitted_candidates
        );
        assert_eq!(prepared.frozen_views, committed.frozen_views);
        assert_eq!(prepared.achieved_max_error_px, None);
        assert_eq!(prepared.achieved_max_target_ratio, None);
        assert_eq!(prepared.target_satisfied, None);

        app.world_mut()
            .get_mut::<GaussianLodPackageStatus>(entity)
            .unwrap()
            .phase = GaussianLodPackagePhase::Active;
        app.update();
        let active_package_pending = app.world().get::<GaussianLodStatus>(entity).unwrap();
        assert_eq!(
            active_package_pending.lifecycle,
            GaussianLodLifecycle::Active
        );
        assert_eq!(active_package_pending.active_views, committed.active_views);
        assert_eq!(
            active_package_pending.submitted_candidates,
            committed.submitted_candidates
        );

        pending_phase.store(
            crate::stream::render_commit::LOD_RENDER_ACTIVE,
            std::sync::atomic::Ordering::Release,
        );
        app.update();
        let replacement = app.world().get::<GaussianLodStatus>(entity).unwrap();
        assert_eq!(replacement.lifecycle, GaussianLodLifecycle::Active);
        assert_eq!(replacement.active_views, 1);
        assert_eq!(replacement.submitted_candidates, replacement_submitted);
        assert!(replacement.achieved_max_error_px.is_some());
        assert!(replacement.achieved_max_target_ratio.is_some());
    }

    fn resident_status_test_frontier() -> (GaussianLodSettings, LodCandidateFrontier) {
        let scene = LodTestScene::screen_space_ladder();
        let mut built = build_planar_3d_lod(
            &scene.cloud(),
            GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 16,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        let mut transport = MemoryPageTransport::default();
        for page in &built.pages {
            let payload = encode_page(page).unwrap();
            let descriptor = built
                .manifest
                .pages
                .iter_mut()
                .find(|descriptor| descriptor.id == page.id)
                .unwrap();
            descriptor.storage = Some(
                crate::gaussian::formats::planar_3d_chunked::LodPageStorage {
                    uri: format!("memory://{}", page.id.0),
                    byte_range: None,
                    encoded_len: payload.len() as u64,
                },
            );
            transport.insert(page.id, payload);
        }
        let mut settings = GaussianLodSettings {
            quality: 0.0,
            ..Default::default()
        };
        settings.budgets.max_requests_per_frame = 128;
        let streaming = crate::gaussian::lod_settings::GaussianStreamingSettings::default();
        let mut runtime =
            LodStreamingRuntime::new(built.manifest, transport, &settings, &streaming).unwrap();
        let frontier = (0..64)
            .find_map(|_| {
                runtime
                    .update(
                        LodView::perspective(Vec3::ZERO, 720.0, 1.0, 0.1),
                        &settings,
                        &streaming,
                    )
                    .unwrap()
                    .candidate_frontier(settings.max_active_gaussians_u32())
                    .ok()
            })
            .expect("root candidate should become resident");
        (settings, frontier)
    }

    fn candidates_quality_target(
        world: &World,
        entity: Entity,
        camera: Entity,
    ) -> LodQualityTarget {
        world
            .get::<LodRenderCandidates>(entity)
            .and_then(|candidates| candidates.get(camera))
            .map(LodRenderCandidate::rendered_quality_status)
            .expect("the retained candidate must still exist")
            .requested_target
    }
}
