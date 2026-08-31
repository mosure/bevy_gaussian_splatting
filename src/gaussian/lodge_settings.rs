//! Runtime policy for externally authored LODGE active sets.
//!
//! LODGE does not expose the hierarchy selector used by
//! [`super::lod_settings::GaussianLodSettings`]:
//! each camera-cluster active set already contains the producer-authored mix of
//! trained LoD levels.  Keeping a separate component prevents `quality == 1`
//! from being misreported as the original source endpoint when an imported
//! active set makes no such promise.

use std::{error::Error, fmt};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use super::lod_settings::{LodBudgets, LodSelectionMode};

/// Presentation family selected for one LoD cloud.
///
/// This is primarily an observability value.  Existing [`super::lod_settings::GaussianLodSettings`]
/// entities remain hierarchy-backed without requiring a new component.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Reflect, Serialize, Deserialize)]
pub enum GaussianLodRepresentationKind {
    /// The crate's source-derived Morton/MomentMerge hierarchy.
    #[default]
    FinestHierarchy,
    /// Externally trained, camera-cluster active sets using LODGE blending.
    LodgeActiveSets,
}

/// Strategy name used by generic LoD-facing application code.
///
/// This alias keeps the existing hierarchy API source-compatible while making
/// the external representation an explicit, inspectable choice.
pub type GaussianLodStrategy = GaussianLodRepresentationKind;

/// Selection and residency policy for a LODGE active-set package.
///
/// There is intentionally no hierarchy quality slider.  The imported artifact
/// owns its trained level assignment; runtime selects the two nearest authored
/// camera clusters and blends their symmetric difference.
#[derive(Component, Clone, Debug, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Component)]
#[serde(default)]
pub struct GaussianLodgeSettings {
    /// Camera policy for selecting the authored cluster pair.  Frozen captures
    /// that view; page loading and publication continue.
    pub selection_mode: LodSelectionMode,
    #[reflect(ignore)]
    pub budgets: LodBudgets,
    /// Relative distance advantage required before replacing a retained
    /// secondary cluster.  This damps nearest-pair churn without changing the
    /// line-projection blend weight inside a stable pair. The default is zero
    /// and therefore follows LODGE's exact two-nearest rule; a positive value
    /// is an explicit application-level continuity extension.
    #[reflect(ignore)]
    pub pair_hysteresis: f32,
    /// Conservative per-record frustum filtering after the complete active-set
    /// union has become drawable.
    #[reflect(ignore)]
    pub frustum_culling: bool,
    #[reflect(ignore)]
    pub frustum_margin: f32,
}

impl Default for GaussianLodgeSettings {
    fn default() -> Self {
        Self {
            selection_mode: LodSelectionMode::default(),
            budgets: LodBudgets::default(),
            pair_hysteresis: 0.0,
            frustum_culling: true,
            frustum_margin: 0.0,
        }
    }
}

impl GaussianLodgeSettings {
    pub fn validate(&self) -> Result<(), GaussianLodgeSettingsError> {
        if !self.pair_hysteresis.is_finite() || !(0.0..=1.0).contains(&self.pair_hysteresis) {
            return Err(GaussianLodgeSettingsError::PairHysteresis(
                self.pair_hysteresis,
            ));
        }
        if !self.frustum_margin.is_finite() || self.frustum_margin < 0.0 {
            return Err(GaussianLodgeSettingsError::FrustumMargin(
                self.frustum_margin,
            ));
        }

        // Reuse the hierarchy budget validator without inheriting hierarchy
        // quality semantics.
        let probe = super::lod_settings::GaussianLodSettings {
            budgets: self.budgets,
            selection_mode: self.selection_mode,
            hysteresis: 0.0,
            frustum_culling: self.frustum_culling,
            frustum_margin: self.frustum_margin,
            ..Default::default()
        };
        probe
            .validate()
            .map_err(|error| GaussianLodgeSettingsError::Budgets(error.to_string()))
    }

    pub fn max_active_gaussians_u32(&self) -> u32 {
        self.budgets
            .max_active_gaussians
            .clamp(1, u64::from(u32::MAX)) as u32
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GaussianLodgeSettingsError {
    PairHysteresis(f32),
    FrustumMargin(f32),
    Budgets(String),
}

impl fmt::Display for GaussianLodgeSettingsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PairHysteresis(value) => write!(
                formatter,
                "LODGE pair_hysteresis {value} must be finite and in 0..=1"
            ),
            Self::FrustumMargin(value) => write!(
                formatter,
                "LODGE frustum_margin {value} must be finite and non-negative"
            ),
            Self::Budgets(error) => write!(formatter, "invalid LODGE budgets: {error}"),
        }
    }
}

impl Error for GaussianLodgeSettingsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lodge_settings_are_independent_from_hierarchy_quality_endpoints() {
        let settings = GaussianLodgeSettings::default();
        assert_eq!(settings.validate(), Ok(()));
        assert_eq!(settings.max_active_gaussians_u32(), 2_000_000);
    }

    #[test]
    fn lodge_settings_reject_nonfinite_or_out_of_range_policy() {
        for pair_hysteresis in [f32::NAN, -0.1, 1.1] {
            let settings = GaussianLodgeSettings {
                pair_hysteresis,
                ..Default::default()
            };
            assert!(matches!(
                settings.validate(),
                Err(GaussianLodgeSettingsError::PairHysteresis(_))
            ));
        }
        let settings = GaussianLodgeSettings {
            frustum_margin: f32::INFINITY,
            ..Default::default()
        };
        assert!(matches!(
            settings.validate(),
            Err(GaussianLodgeSettingsError::FrustumMargin(_))
        ));
    }
}
