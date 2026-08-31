//! Public lifecycle and presentation telemetry for a LODGE active-set cloud.

use bevy::prelude::*;

use crate::gaussian::lodge_settings::GaussianLodRepresentationKind;

use super::{
    lodge::{LodgePairCounts, LodgePairIdentity},
    render_commit::LodOrchestrationFailureCode,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Reflect)]
pub enum GaussianLodgeLifecycle {
    #[default]
    LoadingManifest,
    LoadingMembership,
    LoadingPages,
    WaitingForRender,
    Active,
    Degraded,
    Failed,
}

/// Cloud-facing truth for an externally authored active-set presentation.
///
/// Gaussian/class counts and cluster IDs describe the deterministic primary
/// (lowest-entity) visible camera's one deduplicated union, not two
/// independently drawn chunks. Page counts are deduplicated across all visible
/// private consumers. `target_satisfied` becomes true only after the complete
/// multi-view request has residency and radix-proven drawables.
#[derive(Component, Clone, Debug, PartialEq, Reflect)]
#[reflect(Component)]
pub struct GaussianLodgeStatus {
    pub revision: u64,
    pub representation: GaussianLodRepresentationKind,
    pub lifecycle: GaussianLodgeLifecycle,
    pub target_satisfied: bool,
    pub retained_stale_pair: bool,
    pub first_cluster: Option<u32>,
    pub second_cluster: Option<u32>,
    pub nearest_cluster: Option<u32>,
    /// Exact opacity coefficient of `second_cluster`; the first coefficient is
    /// `1 - second_weight` and shared records always use one.
    pub second_weight: f32,
    pub shared_gaussians: u64,
    pub first_only_gaussians: u64,
    pub second_only_gaussians: u64,
    pub submitted_candidates: u64,
    pub required_pages: u32,
    pub resident_required_pages: u32,
    /// Visible private consumers represented by this cloud status.
    pub visible_views: u32,
    /// Distinct pair identities across those private consumers.
    pub distinct_pairs: u32,
    /// Stable machine-readable classification paired with `failure`.
    pub failure_code: Option<LodOrchestrationFailureCode>,
    pub failure: Option<String>,
}

impl Default for GaussianLodgeStatus {
    fn default() -> Self {
        Self {
            revision: 0,
            representation: GaussianLodRepresentationKind::LodgeActiveSets,
            lifecycle: GaussianLodgeLifecycle::LoadingManifest,
            target_satisfied: false,
            retained_stale_pair: false,
            first_cluster: None,
            second_cluster: None,
            nearest_cluster: None,
            second_weight: 0.0,
            shared_gaussians: 0,
            first_only_gaussians: 0,
            second_only_gaussians: 0,
            submitted_candidates: 0,
            required_pages: 0,
            resident_required_pages: 0,
            visible_views: 0,
            distinct_pairs: 0,
            failure_code: None,
            failure: None,
        }
    }
}

impl GaussianLodgeStatus {
    /// Clears presentation-specific telemetry when no complete pair is being
    /// retained. The representation/lifecycle/failure fields are owned by the
    /// publisher so callers can fold this into one coherent revision update.
    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) fn clear_presentation(&mut self) -> bool {
        let changed = self.first_cluster.is_some()
            || self.second_cluster.is_some()
            || self.nearest_cluster.is_some()
            || self.second_weight.to_bits() != 0.0f32.to_bits()
            || self.shared_gaussians != 0
            || self.first_only_gaussians != 0
            || self.second_only_gaussians != 0
            || self.submitted_candidates != 0
            || self.required_pages != 0
            || self.resident_required_pages != 0
            || self.visible_views != 0
            || self.distinct_pairs != 0;
        self.first_cluster = None;
        self.second_cluster = None;
        self.nearest_cluster = None;
        self.second_weight = 0.0;
        self.shared_gaussians = 0;
        self.first_only_gaussians = 0;
        self.second_only_gaussians = 0;
        self.submitted_candidates = 0;
        self.required_pages = 0;
        self.resident_required_pages = 0;
        self.visible_views = 0;
        self.distinct_pairs = 0;
        changed
    }

    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) fn observe_pair(
        &mut self,
        identity: LodgePairIdentity,
        nearest_cluster: u32,
        second_weight: f32,
        counts: LodgePairCounts,
    ) {
        let changed = self.first_cluster != Some(identity.first.0)
            || self.second_cluster != Some(identity.second.0)
            || self.nearest_cluster != Some(nearest_cluster)
            || self.second_weight.to_bits() != second_weight.to_bits()
            || self.shared_gaussians != counts.shared
            || self.first_only_gaussians != counts.first_only
            || self.second_only_gaussians != counts.second_only
            || self.submitted_candidates != counts.union;
        self.first_cluster = Some(identity.first.0);
        self.second_cluster = Some(identity.second.0);
        self.nearest_cluster = Some(nearest_cluster);
        self.second_weight = second_weight;
        self.shared_gaussians = counts.shared;
        self.first_only_gaussians = counts.first_only;
        self.second_only_gaussians = counts.second_only;
        self.submitted_candidates = counts.union;
        // `required_pages` is a package-wide union across all visible private
        // consumers. The primary pair's local page count is deliberately not
        // written here; the resident publisher assigns the aggregate exactly
        // once after observing the representative pair.
        if changed {
            self.revision = self.revision.saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::formats::lodge::LodgeClusterId;

    #[test]
    fn lodge_status_reports_a_deduplicated_union_not_two_draws() {
        let mut status = GaussianLodgeStatus::default();
        status.observe_pair(
            LodgePairIdentity {
                first: LodgeClusterId(2),
                second: LodgeClusterId(7),
            },
            2,
            0.25,
            LodgePairCounts {
                shared: 10,
                first_only: 4,
                second_only: 6,
                union: 20,
                required_pages: 3,
            },
        );
        assert_eq!(status.submitted_candidates, 20);
        assert_eq!(status.shared_gaussians, 10);
        assert_eq!(status.revision, 1);
    }

    #[test]
    fn primary_pair_observation_does_not_churn_the_multi_view_page_union() {
        let mut status = GaussianLodgeStatus {
            required_pages: 7,
            ..Default::default()
        };
        let identity = LodgePairIdentity {
            first: LodgeClusterId(2),
            second: LodgeClusterId(7),
        };
        let counts = LodgePairCounts {
            shared: 10,
            first_only: 4,
            second_only: 6,
            union: 20,
            required_pages: 3,
        };
        status.observe_pair(identity, 2, 0.25, counts);
        let revision = status.revision;
        status.observe_pair(identity, 2, 0.25, counts);
        assert_eq!(status.required_pages, 7);
        assert_eq!(status.revision, revision);
    }

    #[test]
    fn clearing_an_unretained_presentation_resets_current_consumer_metrics() {
        let mut status = GaussianLodgeStatus {
            first_cluster: Some(1),
            second_cluster: Some(2),
            nearest_cluster: Some(1),
            second_weight: 0.25,
            shared_gaussians: 3,
            first_only_gaussians: 4,
            second_only_gaussians: 5,
            submitted_candidates: 12,
            required_pages: 6,
            resident_required_pages: 6,
            visible_views: 2,
            distinct_pairs: 2,
            ..Default::default()
        };
        assert!(status.clear_presentation());
        assert!(!status.clear_presentation());
        assert_eq!(status.first_cluster, None);
        assert_eq!(status.second_cluster, None);
        assert_eq!(status.nearest_cluster, None);
        assert_eq!(status.second_weight.to_bits(), 0.0f32.to_bits());
        assert_eq!(status.submitted_candidates, 0);
        assert_eq!(status.required_pages, 0);
        assert_eq!(status.visible_views, 0);
        assert_eq!(status.distinct_pairs, 0);
    }
}
