//! Shared CPU-to-render-world commit contracts for streamed LoD clouds.
//!
//! Both ephemeral flat-cloud bridges and prebuilt packages use these types.
//! Keeping the two-phase handshake and atlas mirror here prevents either
//! orchestration frontend from owning renderer-facing state used by the other.

use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering},
    },
};

use bevy::{asset::AssetId, prelude::*, render::RenderApp};
use bevy_interleave::prelude::Planar;

use crate::{
    gaussian::formats::{
        planar_3d::{Gaussian3d, PlanarGaussian3d},
        planar_3d_chunked::{LodPageId, PlanarGaussian3dPage},
    },
    gaussian::lod_settings::LodEffectiveStatus,
    stream::{
        cache::AtlasSlot,
        lodge::{LodgeMembershipClass, LodgePairIdentity, projected_center_line_weight},
        runtime::{
            LodCandidateFrontier, LodPhysicalRange, LodTemporalTransitionMode, LodViewBlendEdge,
            LodViewBlendIdentity, PageAtlasLayout,
        },
    },
};

pub(crate) const LOD_RENDER_WAITING: u8 = 0;
pub(crate) const LOD_RENDER_PREPARED: u8 = 1;
pub(crate) const LOD_RENDER_ACTIVE: u8 = 2;
pub(crate) const LOD_RENDER_FAILED: u8 = 3;
/// The render world is drawing a complete child-cardinality morph cut while
/// main-world package orchestration retains both endpoint page leases.
pub(crate) const LOD_RENDER_TRANSITIONING: u8 = 4;

const LOD_TEMPORAL_MODE_NONE: u8 = 0;
const LOD_TEMPORAL_MODE_MORPHING: u8 = 1;
const LOD_TEMPORAL_MODE_BOUNDED_HARD_COHORT: u8 = 2;
const LOD_RENDER_FALLBACK_NONE: u8 = 0;
const LOD_RENDER_FALLBACK_HARD_REQUESTED: u8 = 1;

/// Stable high-level classification for an orchestration failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Reflect)]
#[non_exhaustive]
pub enum LodOrchestrationFailureCategory {
    Configuration,
    Source,
    Transport,
    Cache,
    Data,
    Runtime,
    Atlas,
    Render,
    Capacity,
    Internal,
}

/// Stable machine-readable code for bridge and package status failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Reflect)]
#[non_exhaustive]
pub enum LodOrchestrationFailureCode {
    InvalidConfiguration,
    UnsupportedConfiguration,
    SourceUnavailable,
    TransportRequestFailed,
    TransportRequestsExhausted,
    CacheFailed,
    DecodeValidationFailed,
    RuntimeFailed,
    AtlasCommitFailed,
    RenderCommitFailed,
    CapacityExceeded,
    InternalInvariant,
}

impl LodOrchestrationFailureCode {
    pub const fn category(self) -> LodOrchestrationFailureCategory {
        match self {
            Self::InvalidConfiguration | Self::UnsupportedConfiguration => {
                LodOrchestrationFailureCategory::Configuration
            }
            Self::SourceUnavailable => LodOrchestrationFailureCategory::Source,
            Self::TransportRequestFailed | Self::TransportRequestsExhausted => {
                LodOrchestrationFailureCategory::Transport
            }
            Self::CacheFailed => LodOrchestrationFailureCategory::Cache,
            Self::DecodeValidationFailed => LodOrchestrationFailureCategory::Data,
            Self::RuntimeFailed => LodOrchestrationFailureCategory::Runtime,
            Self::AtlasCommitFailed => LodOrchestrationFailureCategory::Atlas,
            Self::RenderCommitFailed => LodOrchestrationFailureCategory::Render,
            Self::CapacityExceeded => LodOrchestrationFailureCategory::Capacity,
            Self::InternalInvariant => LodOrchestrationFailureCategory::Internal,
        }
    }
}

/// Typed public failure with optional diagnostic detail for logs and UIs.
#[derive(Clone, Debug, Eq, PartialEq, Reflect)]
pub struct LodOrchestrationFailure {
    code: LodOrchestrationFailureCode,
    detail: Option<String>,
}

impl LodOrchestrationFailure {
    pub fn new(code: LodOrchestrationFailureCode) -> Self {
        Self { code, detail: None }
    }

    pub fn with_detail(code: LodOrchestrationFailureCode, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: Some(detail.into()),
        }
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    pub const fn category(&self) -> LodOrchestrationFailureCategory {
        self.code.category()
    }

    pub const fn code(&self) -> LodOrchestrationFailureCode {
        self.code
    }
}

impl fmt::Display for LodOrchestrationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(detail) = &self.detail {
            write!(formatter, "{:?}: {detail}", self.code)
        } else {
            write!(formatter, "{:?}", self.code)
        }
    }
}

impl std::error::Error for LodOrchestrationFailure {}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Reflect)]
#[non_exhaustive]
pub enum LodOrchestrationSource {
    EphemeralBridge,
    Package,
    /// Externally authored LODGE active sets backed by a resident canonical
    /// Gaussian catalog.
    ExternalActiveSets,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Reflect)]
#[non_exhaustive]
pub enum LodOrchestrationTransitionKind {
    Degraded,
    Failed,
    Recovered,
}

/// Emitted only when orchestration enters degradation/failure or recovers;
/// per-frame metric-only status updates do not emit messages.
#[derive(Clone, Debug, Message)]
pub struct LodOrchestrationTransition {
    pub entity: Entity,
    pub source: LodOrchestrationSource,
    pub kind: LodOrchestrationTransitionKind,
    pub failure: Option<LodOrchestrationFailure>,
}

/// Shared MainWorld/RenderWorld revision for renderer-owned facts which can
/// make an otherwise identical LoD render request admissible after failure.
///
/// RenderWorld advances this only when the complete visible external pipeline
/// key set or the compaction/device-limit fingerprint changes. MainWorld can
/// therefore durably latch an unchanged terminal request without suppressing a
/// retry after a real render-environment change.
#[derive(Clone, Debug, Resource)]
#[cfg_attr(not(lod_render_path), allow(dead_code))]
pub(crate) struct LodRenderEnvironmentEpoch(Arc<AtomicU64>);

impl Default for LodRenderEnvironmentEpoch {
    fn default() -> Self {
        Self(Arc::new(AtomicU64::new(0)))
    }
}

#[cfg_attr(not(lod_render_path), allow(dead_code))]
impl LodRenderEnvironmentEpoch {
    pub(crate) fn current(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }

    pub(crate) fn advance(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }
}

#[derive(Default)]
pub(crate) struct GaussianLodRenderCommitPlugin;

impl Plugin for GaussianLodRenderCommitPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<LodOrchestrationTransition>()
            .init_resource::<LodRenderEnvironmentEpoch>();
        let environment_epoch = app.world().resource::<LodRenderEnvironmentEpoch>().clone();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(environment_epoch);
        }
    }
}

/// One render-world commit guarded by a cross-world two-phase handshake.
#[derive(Clone, Debug)]
pub struct LodRenderCandidate {
    pub(crate) frontier: LodCandidateFrontier,
    pub(crate) phase: Arc<AtomicU8>,
    /// Mutually exclusive external active-set presentation. The physical
    /// ranges remain one deduplicated union; this sidecar supplies the stable
    /// pair centers and one class per range so RenderWorld can apply LODGE's
    /// exact opacity coefficients without weakening hierarchy-frontier types.
    external_active_set: Option<Arc<LodExternalActiveSetPresentation>>,
    /// Render-owned, independently published weights for the immutable adjacent
    /// hierarchy edges carried by this candidate. ACTIVE means the GPU has a
    /// drawable presentation for these edges; it does not imply that any edge
    /// has reached a categorical endpoint.
    view_blend_revision: Arc<AtomicU32>,
    /// Changes only when package-retirement evidence changes. Unlike the
    /// general seqlock revision, metric-only camera publications leave this
    /// epoch stable so a pipelined render frame does not spuriously invalidate
    /// an otherwise exact endpoint proof.
    view_blend_retirement_epoch: Arc<AtomicU32>,
    view_blend_edge_count: Arc<AtomicU32>,
    view_blend_lagging_count: Arc<AtomicU32>,
    /// Number of edge/view pressure evaluations which were invalid in the
    /// latest coherently published drawable snapshot. ACTIVE rendering retains
    /// the preceding displayed/desired weights while this is nonzero.
    view_blend_invalid_pressure_count: Arc<AtomicU32>,
    /// Expected private render consumers which do not yet have a radix-proven
    /// drawable snapshot for this immutable table.
    view_blend_missing_consumer_count: Arc<AtomicU32>,
    view_blend_max_lag_bits: Arc<AtomicU32>,
    view_blend_max_delta_bits: Arc<AtomicU32>,
    view_blend_weighted_record_energy_bits: Arc<AtomicU32>,
    view_blend_all_at_target: Arc<AtomicBool>,
    /// One byte per immutable edge: fractional, exact parent, or exact
    /// children. Package replacement reads this allocation-stable table only
    /// when edge content changes, so disjoint additions do not wait for
    /// unrelated fractional edges.
    view_blend_endpoints: Arc<RwLock<Vec<u8>>>,
    #[cfg(any(test, feature = "testing"))]
    view_blend_weights: Arc<RwLock<Vec<LodViewBlendWeightSnapshot>>>,
    /// Package-authored effective presentation mode, finalized before
    /// retirement and progressive-admission checks. Render-only capability
    /// rejection is published separately through `render_fallback`; it may not
    /// mutate this mode or expose a categorical frame under a blend token.
    temporal_mode: Arc<AtomicU8>,
    /// Render-discovered capability veto. A render backend publishes this
    /// request without changing `temporal_mode` or exposing the categorical
    /// target. Package orchestration cancels the unrendered transaction and
    /// rebuilds it under ordinary hard-cut admission rules.
    render_fallback: Arc<AtomicU8>,
    /// A camera/endpoint race invalidated package retirement evidence. Render
    /// retains the predecessor output while package orchestration cancels and
    /// reselects this token without entering a hard-fallback failure path.
    view_blend_replan_requested: Arc<AtomicBool>,
    /// Package proof for every predecessor edge removed by this replacement.
    /// Common edges need no attestation because render inherits their state.
    predecessor_view_blend_attestation: Option<LodViewBlendPredecessorAttestation>,
    /// Durable proof that a visible cloud candidate crossed into RenderWorld.
    /// Camera-only request churn may let this exact identity finish even while
    /// its private GPU state has not advanced to PREPARED yet.
    render_claimed: Arc<AtomicBool>,
}

/// Immutable content identity for one externally authored two-set union.
/// Camera position is deliberately absent: the render view evaluates the
/// projected center-line weight statelessly against these local-space centers.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LodExternalActiveSetPresentation {
    pair: LodgePairIdentity,
    first_center: [f32; 3],
    second_center: [f32; 3],
    /// Exact authored coefficient captured when LODGE selection is Frozen.
    /// Dynamic presentations leave this unset and evaluate the current private
    /// render view statelessly.
    frozen_second_weight_bits: Option<u32>,
    range_classes: Arc<[LodgeMembershipClass]>,
}

#[cfg_attr(not(lod_render_path), allow(dead_code))]
impl LodExternalActiveSetPresentation {
    pub(crate) fn new(
        pair: LodgePairIdentity,
        first_center: [f32; 3],
        second_center: [f32; 3],
        range_classes: Vec<LodgeMembershipClass>,
    ) -> Option<Self> {
        if pair.first == pair.second
            || first_center
                .iter()
                .chain(second_center.iter())
                .any(|value| !value.is_finite())
            || range_classes.is_empty()
        {
            return None;
        }
        let separation_squared = first_center
            .iter()
            .zip(second_center)
            .map(|(first, second)| {
                let delta = f64::from(*first) - f64::from(second);
                delta * delta
            })
            .sum::<f64>();
        // Centers are authored as f32 and evaluated in f64. Any distinct
        // finite tuple has a representable non-zero denominator here; using
        // f64::EPSILON would reject valid close centers accepted by the format
        // and the public pair-weight oracle.
        if !separation_squared.is_finite() || separation_squared == 0.0 {
            return None;
        }
        Some(Self {
            pair,
            first_center,
            second_center,
            frozen_second_weight_bits: None,
            range_classes: range_classes.into(),
        })
    }

    pub(crate) fn with_frozen_second_weight(mut self, second_weight: f32) -> Option<Self> {
        if !second_weight.is_finite() || !(0.0..=1.0).contains(&second_weight) {
            return None;
        }
        self.frozen_second_weight_bits = Some(second_weight.to_bits());
        Some(self)
    }

    pub(crate) const fn pair(&self) -> LodgePairIdentity {
        self.pair
    }

    pub(crate) const fn first_center(&self) -> [f32; 3] {
        self.first_center
    }

    pub(crate) const fn second_center(&self) -> [f32; 3] {
        self.second_center
    }

    pub(crate) fn range_classes(&self) -> &[LodgeMembershipClass] {
        &self.range_classes
    }

    /// Returns `(first, second)` opacity weights for a cloud-local camera
    /// position. Shared records remain unweighted by the caller.
    pub(crate) fn opacity_weights(&self, local_view: [f32; 3]) -> Option<(f32, f32)> {
        if let Some(bits) = self.frozen_second_weight_bits {
            let second = f32::from_bits(bits);
            return Some((1.0 - second, second));
        }
        let second =
            projected_center_line_weight(local_view, self.first_center, self.second_center).ok()?;
        Some((1.0 - second, second))
    }
}

/// One coherent render-owned observation of an immutable view-blend edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodViewBlendWeightSnapshot {
    pub displayed: f32,
    pub desired: f32,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodViewBlendEndpoint {
    Fractional = 0,
    ParentExact = 1,
    ChildrenExact = 2,
}

/// Presentation that RenderWorld has made drawable for one candidate token.
/// Reading phase first and capability state second pairs the ACTIVE release
/// with the final immutable presentation decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LodRenderActivePresentation {
    ViewBlend,
    HardTarget,
}

impl LodViewBlendEndpoint {
    fn from_weight(weight: f32) -> Self {
        if weight.to_bits() == 0.0_f32.to_bits() {
            Self::ParentExact
        } else if weight.to_bits() == 1.0_f32.to_bits() {
            Self::ChildrenExact
        } else {
            Self::Fractional
        }
    }

    fn from_byte(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Fractional),
            1 => Some(Self::ParentExact),
            2 => Some(Self::ChildrenExact),
            _ => None,
        }
    }
}

/// Constant-size cross-world status for one camera-conditioned blend table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodViewBlendStatusSnapshot {
    pub edge_count: u32,
    pub lagging_count: u32,
    /// Immutable edges with a non-finite or threshold-contradictory evaluation
    /// in any drawable private view. Multi-view reduction ORs by edge key, so
    /// this never exceeds `edge_count`. Such edges hold their prior exact
    /// state; they are neither converged nor ordinary slew lag.
    pub invalid_pressure_count: u32,
    /// Expected private consumers without a coherent radix-proven snapshot.
    /// This is incomplete presentation evidence even when every available
    /// consumer reports zero lag and valid pressure.
    pub missing_consumer_count: u32,
    pub max_lag: f32,
    pub max_delta: f32,
    pub weighted_record_energy: f32,
    pub all_at_target: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LodViewBlendRetirementSnapshot {
    pub status: LodViewBlendStatusSnapshot,
    pub endpoints: Vec<LodViewBlendEndpoint>,
    retirement_epoch: u32,
}

/// One removed immutable edge and the categorical predecessor side which the
/// package proved is represented by the replacement's first drawable cut.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LodViewBlendRetirementRequirement {
    edge: LodViewBlendEdge,
    endpoint: LodViewBlendEndpoint,
}

impl LodViewBlendRetirementRequirement {
    pub(crate) fn new(edge: LodViewBlendEdge, endpoint: LodViewBlendEndpoint) -> Self {
        Self { edge, endpoint }
    }

    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) fn edge(&self) -> &LodViewBlendEdge {
        &self.edge
    }

    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) const fn endpoint(&self) -> LodViewBlendEndpoint {
        self.endpoint
    }
}

/// Pipelined-render proof that a replacement was authored from the exact
/// predecessor endpoint evidence still bound in RenderWorld.
#[derive(Clone, Debug)]
pub(crate) struct LodViewBlendPredecessorAttestation {
    predecessor_identity: LodViewBlendIdentity,
    retirement_epoch: Arc<AtomicU32>,
    expected_retirement_epoch: u32,
    requirements: Arc<[LodViewBlendRetirementRequirement]>,
}

impl LodViewBlendPredecessorAttestation {
    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) const fn predecessor_identity(&self) -> LodViewBlendIdentity {
        self.predecessor_identity
    }

    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) fn requirements(&self) -> &[LodViewBlendRetirementRequirement] {
        &self.requirements
    }

    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) fn is_current(
        &self,
        drawable_predecessor_identity: Option<LodViewBlendIdentity>,
    ) -> bool {
        drawable_predecessor_identity == Some(self.predecessor_identity)
            && self.retirement_epoch.load(Ordering::Acquire) == self.expected_retirement_epoch
    }

    fn epoch_is_current(&self) -> bool {
        self.retirement_epoch.load(Ordering::Acquire) == self.expected_retirement_epoch
    }

    fn same_proof(&self, other: &Self) -> bool {
        self.predecessor_identity == other.predecessor_identity
            && Arc::ptr_eq(&self.retirement_epoch, &other.retirement_epoch)
            && self.expected_retirement_epoch == other.expected_retirement_epoch
            && self.requirements == other.requirements
    }
}

#[cfg(any(test, feature = "testing"))]
#[derive(Clone, Debug, PartialEq)]
pub struct LodViewBlendTestingSnapshot {
    pub status: LodViewBlendStatusSnapshot,
    pub endpoints: Vec<LodViewBlendEndpoint>,
    pub weights: Vec<LodViewBlendWeightSnapshot>,
}

/// Extracted on the cloud entity and consumed automatically by render LoD.
#[derive(Component, Clone, Debug, Default)]
pub struct LodRenderCandidates {
    pub(crate) by_camera: BTreeMap<Entity, LodRenderCandidate>,
    /// Atlas whose page generations back these candidates. During a cold
    /// transient handoff this differs from the entity's still-visible source
    /// handle, allowing render preparation to finish before the atomic swap.
    pub(crate) staging_atlas: Option<AssetId<PlanarGaussian3d>>,
    /// Package atlases are page caches and may never use an unfiltered draw.
    /// Keeping the flag on this already-public extracted component avoids a
    /// second marker in the public render-command query surface.
    pub(crate) candidate_draw_required: bool,
    /// Main-world orchestration still owns a last complete package cut while it
    /// selects, streams, or prepares the candidates extracted in `by_camera`.
    /// The render world must preserve that cut's drawable compaction allocation
    /// until the replacement activates, including while a pending candidate is
    /// present and after the requested active budget decreases.
    pub(crate) retained_current: bool,
    /// The candidates in `by_camera` are the retained current package cut, not
    /// a pending replacement. A stale pending candidate may keep the retained
    /// output allocated, but only a current candidate may be replayed when its
    /// recorded selection policy differs from the newly extracted request.
    pub(crate) candidates_are_current: bool,
    /// Package orchestration is retaining this drawable current cut for a
    /// different live camera/policy request. This is stronger than comparing
    /// quality targets: same-quality camera motion also invalidates the old
    /// cut's achieved-error and target-satisfaction diagnostics.
    pub(crate) retained_current_is_stale: bool,
    /// Every pending package range has a semantically matching debug record in
    /// its physical slot. The render world still waits for the target-scoped
    /// invariant revisions to reach the GPU before arming activation.
    pub(crate) debug_metadata_staged: bool,
    /// A pending morph already owns drawable GPU entries. Camera/settings churn
    /// may supersede it only after exact target publication, so every live view
    /// in this atomic transaction must continue past stale-policy filtering.
    pub(crate) transition_must_commit: bool,
}

impl LodRenderCandidate {
    /// Creates a two-phase commit for a runtime-validated complete frontier.
    pub fn new(frontier: LodCandidateFrontier) -> Self {
        Self::with_phase(frontier, Arc::new(AtomicU8::new(LOD_RENDER_WAITING)))
    }

    pub(crate) fn with_phase(frontier: LodCandidateFrontier, phase: Arc<AtomicU8>) -> Self {
        if complete_empty_candidate(frontier.candidate_count(), frontier.physical_ranges()) {
            // A complete empty cut performs no atlas reads, compaction, radix,
            // or draw. It may never be visited by a render pass, so publish its
            // capability at construction instead of waiting for GPU work.
            phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
        }
        let temporal_mode = Arc::new(AtomicU8::new(frontier.temporal_transition().map_or(
            LOD_TEMPORAL_MODE_NONE,
            |transition| match transition.mode() {
                LodTemporalTransitionMode::Morphing => LOD_TEMPORAL_MODE_MORPHING,
                LodTemporalTransitionMode::BoundedHardCohort => {
                    LOD_TEMPORAL_MODE_BOUNDED_HARD_COHORT
                }
            },
        )));
        let initial_weights: Arc<[LodViewBlendWeightSnapshot]> = frontier
            .temporal_transition()
            .and_then(|transition| transition.morph())
            .map(|morph| {
                morph
                    .edges()
                    .iter()
                    .map(|edge| LodViewBlendWeightSnapshot {
                        displayed: edge.initial_weight(),
                        desired: edge.initial_weight(),
                    })
                    .collect::<Vec<_>>()
                    .into()
            })
            .unwrap_or_else(|| Arc::from([]));
        let initial_all_at_target = frontier.temporal_transition().is_none_or(|transition| {
            initial_weights.len() == transition.substitutions().len()
                && initial_weights.iter().zip(transition.substitutions()).all(
                    |(weight, substitution)| {
                        let endpoint = match substitution.key.direction {
                            super::hierarchy::LodTemporalDirection::Coarsen => 0.0_f32,
                            super::hierarchy::LodTemporalDirection::Refine => 1.0_f32,
                        };
                        weight.displayed.to_bits() == endpoint.to_bits()
                            && weight.desired.to_bits() == endpoint.to_bits()
                    },
                )
        });
        let initial_endpoints = initial_weights
            .iter()
            .map(|weight| LodViewBlendEndpoint::from_weight(weight.displayed) as u8)
            .collect::<Vec<_>>();
        let initial_edge_count = u32::try_from(initial_weights.len()).unwrap_or(u32::MAX);
        let initial_missing_consumer_count = u32::from(
            initial_edge_count != 0
                && temporal_mode.load(Ordering::Relaxed) == LOD_TEMPORAL_MODE_MORPHING,
        );
        Self {
            frontier,
            phase,
            external_active_set: None,
            view_blend_revision: Arc::new(AtomicU32::new(0)),
            view_blend_retirement_epoch: Arc::new(AtomicU32::new(0)),
            view_blend_edge_count: Arc::new(AtomicU32::new(initial_edge_count)),
            view_blend_lagging_count: Arc::new(AtomicU32::new(0)),
            view_blend_invalid_pressure_count: Arc::new(AtomicU32::new(0)),
            view_blend_missing_consumer_count: Arc::new(AtomicU32::new(
                initial_missing_consumer_count,
            )),
            view_blend_max_lag_bits: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            view_blend_max_delta_bits: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            view_blend_weighted_record_energy_bits: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            view_blend_all_at_target: Arc::new(AtomicBool::new(
                initial_all_at_target && initial_missing_consumer_count == 0,
            )),
            view_blend_endpoints: Arc::new(RwLock::new(initial_endpoints)),
            #[cfg(any(test, feature = "testing"))]
            view_blend_weights: Arc::new(RwLock::new(initial_weights.to_vec())),
            temporal_mode,
            render_fallback: Arc::new(AtomicU8::new(LOD_RENDER_FALLBACK_NONE)),
            view_blend_replan_requested: Arc::new(AtomicBool::new(false)),
            predecessor_view_blend_attestation: None,
            render_claimed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a render commit for one complete, deduplicated external
    /// active-set union. The caller must construct `frontier` from the same
    /// ordered ranges represented by `range_classes`.
    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) fn new_external_active_set(
        frontier: LodCandidateFrontier,
        presentation: LodExternalActiveSetPresentation,
    ) -> Option<Self> {
        if presentation.range_classes().len() != frontier.physical_ranges().len()
            || frontier.temporal_transition().is_some()
        {
            return None;
        }
        let mut candidate = Self::new(frontier);
        candidate.external_active_set = Some(Arc::new(presentation));
        Some(candidate)
    }

    /// Logical selector provenance for the exact bounded atlas cut submitted
    /// by the renderer.
    pub fn frontier(&self) -> &LodCandidateFrontier {
        &self.frontier
    }

    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) fn external_active_set(&self) -> Option<&LodExternalActiveSetPresentation> {
        self.external_active_set.as_deref()
    }

    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) const fn is_external_active_set(&self) -> bool {
        self.external_active_set.is_some()
    }

    /// Generation-safe atlas ranges referenced by this rendered frame.
    pub fn render_ranges(&self) -> &[LodPhysicalRange] {
        if self.temporal_transition_mode() != Some(LodTemporalTransitionMode::Morphing) {
            self.frontier.physical_ranges()
        } else {
            self.temporal_transition()
                .and_then(|transition| transition.morph())
                .map_or_else(
                    || self.frontier.physical_ranges(),
                    |morph| morph.presentation_ranges(),
                )
        }
    }

    /// Exact settled target ranges, independent of an in-flight morph phase.
    pub(crate) fn target_render_ranges(&self) -> &[LodPhysicalRange] {
        self.frontier.physical_ranges()
    }

    /// Generation-safe union retained/materialized until the transition
    /// reaches its exact target endpoint.
    pub(crate) fn required_atlas_ranges(&self) -> &[LodPhysicalRange] {
        if self.temporal_transition_mode() != Some(LodTemporalTransitionMode::Morphing) {
            self.frontier.physical_ranges()
        } else {
            self.temporal_transition()
                .and_then(|transition| transition.morph())
                .map_or_else(
                    || self.frontier.physical_ranges(),
                    |morph| morph.required_ranges(),
                )
        }
    }

    /// Number of Gaussians actually submitted by the selected render source.
    pub fn rendered_candidate_count(&self) -> u32 {
        self.render_ranges()
            .iter()
            .fold(0_u32, |count, range| count.saturating_add(range.count))
    }

    /// Quality observation for the bounded cut that is actually rendered.
    pub fn rendered_quality_status(&self) -> LodEffectiveStatus {
        *self.frontier.quality_status()
    }

    /// Bounded parent/children transactions which produced this complete cut.
    /// Render backends with an authored morph map may consume this seam; older
    /// packages render the density-correct destination without interpolation.
    pub fn temporal_transition(&self) -> Option<&super::runtime::LodTemporalTransition> {
        self.frontier.temporal_transition()
    }

    /// Camera-continuous presentation payload. The historical transition name
    /// remains as a source-compatibility wrapper while disk ABI16 keeps its
    /// authored `morph_map` spelling.
    pub fn view_blend(&self) -> Option<&super::runtime::LodViewBlend> {
        self.temporal_transition()
    }

    /// Legacy scalar transition progress. Camera-conditioned adjacent-edge
    /// blending has no shared clock, so this is always `None`, including while
    /// a Morphing candidate remains ACTIVE with fractional edge weights.
    pub fn temporal_transition_progress(&self) -> Option<f32> {
        None
    }

    /// Effective presentation capability after render-adapter checks. Morphing
    /// may persist while the candidate is ACTIVE because a stationary
    /// fractional edge is stable presentation state, not a pending clock.
    pub fn temporal_transition_mode(&self) -> Option<LodTemporalTransitionMode> {
        self.temporal_transition()?;
        match self.temporal_mode.load(Ordering::Acquire) {
            LOD_TEMPORAL_MODE_MORPHING => Some(LodTemporalTransitionMode::Morphing),
            LOD_TEMPORAL_MODE_BOUNDED_HARD_COHORT => {
                Some(LodTemporalTransitionMode::BoundedHardCohort)
            }
            _ => None,
        }
    }

    pub fn view_blend_mode(&self) -> Option<super::runtime::LodViewBlendMode> {
        self.temporal_transition_mode()
    }

    /// Requests a package-authored categorical replan after a render-only
    /// capability check rejects the immutable view-blend table. The retained
    /// drawable output remains untouched and this candidate is held outside
    /// ACTIVE until the main world cancels it.
    #[cfg(any(test, lod_render_path))]
    pub(crate) fn request_hard_fallback(&self) {
        self.render_fallback
            .store(LOD_RENDER_FALLBACK_HARD_REQUESTED, Ordering::Release);
        self.phase.store(LOD_RENDER_WAITING, Ordering::Release);
    }

    pub(crate) fn render_hard_fallback_requested(&self) -> bool {
        self.render_fallback.load(Ordering::Acquire) == LOD_RENDER_FALLBACK_HARD_REQUESTED
    }

    /// Requests a non-hard package replan when pipelined render observation no
    /// longer matches the endpoint proof attached to this pending token.
    /// RenderWorld must call this before synchronizing any replacement table or
    /// descriptor bytes, so the retained predecessor remains drawable.
    #[cfg(any(test, lod_render_path))]
    pub(crate) fn request_view_blend_replan(&self) {
        self.view_blend_replan_requested
            .store(true, Ordering::Release);
        self.phase.store(LOD_RENDER_WAITING, Ordering::Release);
    }

    pub(crate) fn view_blend_replan_requested(&self) -> bool {
        self.view_blend_replan_requested.load(Ordering::Acquire)
    }

    /// Testing-only frozen observation of the non-hard pipelined retirement
    /// replan request carried by this exact candidate token.
    #[cfg(feature = "testing")]
    pub fn view_blend_replan_requested_for_testing(&self) -> bool {
        self.view_blend_replan_requested()
    }

    /// Publishes every independently evaluated adjacent-edge weight as one
    /// coherent, already-reduced snapshot. `endpoints` must be the unanimous drawable endpoint
    /// classification across every retained private render view; any disagreement
    /// is `Fractional`. Lag and energy fields are already reduced across those
    /// consumers. `displayed`/`desired` are one stable-key-ordered representative
    /// table retained only by testing builds, and therefore need not classify the
    /// unanimous endpoint mask by themselves.
    #[cfg(any(test, lod_render_path))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn publish_view_blend_aggregate_snapshot(
        &self,
        displayed: &[f32],
        desired: &[f32],
        lagging_count: u32,
        invalid_pressure_count: u32,
        missing_consumer_count: u32,
        max_lag: f32,
        max_delta: f32,
        weighted_record_energy: f32,
        endpoints: &[LodViewBlendEndpoint],
    ) -> bool {
        let expected = self
            .temporal_transition()
            .and_then(|transition| transition.morph())
            .map_or(0, |morph| morph.edges().len());
        if displayed.len() != expected
            || desired.len() != expected
            || endpoints.len() != expected
            || lagging_count > u32::try_from(expected).unwrap_or(u32::MAX)
            || invalid_pressure_count > u32::try_from(expected).unwrap_or(u32::MAX)
            || !max_lag.is_finite()
            || !(0.0..=1.0).contains(&max_lag)
            || !max_delta.is_finite()
            || max_delta < 0.0
            || !weighted_record_energy.is_finite()
            || weighted_record_energy < 0.0
            || displayed
                .iter()
                .chain(desired)
                .any(|weight| !weight.is_finite() || !(0.0..=1.0).contains(weight))
        {
            return false;
        }
        let all_at_target = invalid_pressure_count == 0
            && missing_consumer_count == 0
            && self.temporal_transition().is_none_or(|transition| {
                endpoints.len() == transition.substitutions().len()
                    && endpoints.iter().zip(transition.substitutions()).all(
                        |(endpoint, substitution)| {
                            *endpoint
                                == match substitution.key.direction {
                                    super::hierarchy::LodTemporalDirection::Coarsen => {
                                        LodViewBlendEndpoint::ParentExact
                                    }
                                    super::hierarchy::LodTemporalDirection::Refine => {
                                        LodViewBlendEndpoint::ChildrenExact
                                    }
                                }
                        },
                    )
            });

        let mut revision = self.view_blend_revision.load(Ordering::Acquire);
        loop {
            if revision & 1 != 0 {
                std::hint::spin_loop();
                revision = self.view_blend_revision.load(Ordering::Acquire);
                continue;
            }
            match self.view_blend_revision.compare_exchange_weak(
                revision,
                revision.wrapping_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => revision = observed,
            }
        }
        self.view_blend_edge_count
            .store(expected.try_into().unwrap_or(u32::MAX), Ordering::Relaxed);
        self.view_blend_lagging_count
            .store(lagging_count, Ordering::Relaxed);
        let previous_invalid_pressure_count = self
            .view_blend_invalid_pressure_count
            .load(Ordering::Relaxed);
        self.view_blend_invalid_pressure_count
            .store(invalid_pressure_count, Ordering::Relaxed);
        let previous_missing_consumer_count = self
            .view_blend_missing_consumer_count
            .load(Ordering::Relaxed);
        self.view_blend_missing_consumer_count
            .store(missing_consumer_count, Ordering::Relaxed);
        self.view_blend_max_lag_bits
            .store(max_lag.to_bits(), Ordering::Relaxed);
        self.view_blend_max_delta_bits
            .store(max_delta.to_bits(), Ordering::Relaxed);
        self.view_blend_weighted_record_energy_bits
            .store(weighted_record_energy.to_bits(), Ordering::Relaxed);
        self.view_blend_all_at_target
            .store(all_at_target, Ordering::Relaxed);

        let endpoints_changed = match self.view_blend_endpoints.write() {
            Ok(mut published) => {
                let changed = !published
                    .iter()
                    .copied()
                    .eq(endpoints.iter().map(|endpoint| *endpoint as u8));
                published.clear();
                published.extend(endpoints.iter().map(|endpoint| *endpoint as u8));
                changed
            }
            Err(poisoned) => {
                let mut published = poisoned.into_inner();
                let changed = !published
                    .iter()
                    .copied()
                    .eq(endpoints.iter().map(|endpoint| *endpoint as u8));
                published.clear();
                published.extend(endpoints.iter().map(|endpoint| *endpoint as u8));
                changed
            }
        };

        if endpoints_changed
            || previous_invalid_pressure_count != invalid_pressure_count
            || previous_missing_consumer_count != missing_consumer_count
        {
            self.view_blend_retirement_epoch
                .fetch_add(1, Ordering::Release);
        }

        #[cfg(any(test, feature = "testing"))]
        {
            match self.view_blend_weights.write() {
                Ok(mut published) => {
                    published.clear();
                    published.extend(displayed.iter().copied().zip(desired.iter().copied()).map(
                        |(displayed, desired)| LodViewBlendWeightSnapshot { displayed, desired },
                    ));
                }
                Err(poisoned) => {
                    let mut published = poisoned.into_inner();
                    published.clear();
                    published.extend(displayed.iter().copied().zip(desired.iter().copied()).map(
                        |(displayed, desired)| LodViewBlendWeightSnapshot { displayed, desired },
                    ));
                }
            }
        }
        self.view_blend_revision
            .store(revision.wrapping_add(2), Ordering::Release);
        true
    }

    pub fn view_blend_status(&self) -> Option<LodViewBlendStatusSnapshot> {
        if self.temporal_transition_mode() != Some(LodTemporalTransitionMode::Morphing) {
            return None;
        }
        loop {
            let before = self.view_blend_revision.load(Ordering::Acquire);
            if before & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let snapshot = self.load_view_blend_status();
            let after = self.view_blend_revision.load(Ordering::Acquire);
            if before == after {
                return Some(snapshot);
            }
        }
    }

    fn load_view_blend_status(&self) -> LodViewBlendStatusSnapshot {
        LodViewBlendStatusSnapshot {
            edge_count: self.view_blend_edge_count.load(Ordering::Relaxed),
            lagging_count: self.view_blend_lagging_count.load(Ordering::Relaxed),
            invalid_pressure_count: self
                .view_blend_invalid_pressure_count
                .load(Ordering::Relaxed),
            missing_consumer_count: self
                .view_blend_missing_consumer_count
                .load(Ordering::Relaxed),
            max_lag: f32::from_bits(self.view_blend_max_lag_bits.load(Ordering::Relaxed)),
            max_delta: f32::from_bits(self.view_blend_max_delta_bits.load(Ordering::Relaxed)),
            weighted_record_energy: f32::from_bits(
                self.view_blend_weighted_record_energy_bits
                    .load(Ordering::Relaxed),
            ),
            all_at_target: self.view_blend_all_at_target.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn view_blend_retirement_snapshot(&self) -> Option<LodViewBlendRetirementSnapshot> {
        if self.temporal_transition_mode() != Some(LodTemporalTransitionMode::Morphing) {
            return None;
        }
        loop {
            let before = self.view_blend_revision.load(Ordering::Acquire);
            if before & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let snapshot = self.load_view_blend_status();
            let retirement_epoch = self.view_blend_retirement_epoch.load(Ordering::Relaxed);
            let endpoint_bytes = match self.view_blend_endpoints.read() {
                Ok(published) => published.clone(),
                Err(poisoned) => poisoned.into_inner().clone(),
            };
            let endpoints = endpoint_bytes
                .into_iter()
                .map(LodViewBlendEndpoint::from_byte)
                .collect::<Option<Vec<_>>>()?;
            let after = self.view_blend_revision.load(Ordering::Acquire);
            if before == after {
                return Some(LodViewBlendRetirementSnapshot {
                    status: snapshot,
                    endpoints,
                    retirement_epoch,
                });
            }
        }
    }

    pub(crate) fn view_blend_predecessor_attestation(
        &self,
        snapshot: &LodViewBlendRetirementSnapshot,
        requirements: Vec<LodViewBlendRetirementRequirement>,
    ) -> Option<LodViewBlendPredecessorAttestation> {
        if requirements.is_empty()
            || self.view_blend_retirement_epoch.load(Ordering::Acquire) != snapshot.retirement_epoch
        {
            return None;
        }
        let predecessor_identity = self
            .temporal_transition()
            .and_then(|transition| transition.morph())?
            .identity();
        Some(LodViewBlendPredecessorAttestation {
            predecessor_identity,
            retirement_epoch: self.view_blend_retirement_epoch.clone(),
            expected_retirement_epoch: snapshot.retirement_epoch,
            requirements: requirements.into(),
        })
    }

    pub(crate) fn set_predecessor_view_blend_attestation(
        &mut self,
        attestation: LodViewBlendPredecessorAttestation,
    ) {
        self.predecessor_view_blend_attestation = Some(attestation);
    }

    pub(crate) fn predecessor_view_blend_attestation(
        &self,
    ) -> Option<&LodViewBlendPredecessorAttestation> {
        self.predecessor_view_blend_attestation.as_ref()
    }

    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) fn predecessor_view_blend_attestation_is_current(
        &self,
        drawable_predecessor_identity: Option<LodViewBlendIdentity>,
    ) -> bool {
        self.predecessor_view_blend_attestation
            .as_ref()
            .is_none_or(|attestation| attestation.is_current(drawable_predecessor_identity))
    }

    pub(crate) fn predecessor_view_blend_attestation_epoch_is_current(&self) -> bool {
        self.predecessor_view_blend_attestation
            .as_ref()
            .is_none_or(LodViewBlendPredecessorAttestation::epoch_is_current)
    }

    pub(crate) fn predecessor_view_blend_attestation_matches(
        &self,
        expected: &LodViewBlendPredecessorAttestation,
    ) -> bool {
        self.predecessor_view_blend_attestation
            .as_ref()
            .is_some_and(|attestation| attestation.same_proof(expected))
    }

    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn view_blend_weight_snapshots(&self) -> Vec<LodViewBlendWeightSnapshot> {
        match self.view_blend_weights.read() {
            Ok(published) => published.to_vec(),
            Err(poisoned) => poisoned.into_inner().to_vec(),
        }
    }

    /// True only when camera-conditioned convergence cannot yet own the live
    /// request: exceptional late-readiness/camera-jump slew, or an invalid
    /// pressure evaluation holding the prior drawable bits. A stationary valid
    /// fractional edge with displayed == desired is an exact fixed point.
    pub(crate) fn view_blend_is_lagging(&self) -> bool {
        self.view_blend_status().is_some_and(|status| {
            status.lagging_count != 0
                || status.invalid_pressure_count != 0
                || status.missing_consumer_count != 0
        })
    }

    #[cfg(feature = "testing")]
    pub fn view_blend_weight_snapshots_for_testing(&self) -> Vec<LodViewBlendWeightSnapshot> {
        self.view_blend_weight_snapshots()
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn view_blend_testing_snapshot(&self) -> Option<LodViewBlendTestingSnapshot> {
        loop {
            let before = self.view_blend_revision.load(Ordering::Acquire);
            if before & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let retirement = self.view_blend_retirement_snapshot()?;
            let weights = self.view_blend_weight_snapshots();
            let after = self.view_blend_revision.load(Ordering::Acquire);
            if before == after {
                return Some(LodViewBlendTestingSnapshot {
                    status: retirement.status,
                    endpoints: retirement.endpoints,
                    weights,
                });
            }
        }
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn view_blend_snapshot_for_testing(&self) -> Option<LodViewBlendTestingSnapshot> {
        self.view_blend_testing_snapshot()
    }

    pub(crate) fn publish_temporal_transition_mode(&self, mode: LodTemporalTransitionMode) {
        self.temporal_mode.store(
            match mode {
                LodTemporalTransitionMode::Morphing => LOD_TEMPORAL_MODE_MORPHING,
                LodTemporalTransitionMode::BoundedHardCohort => {
                    LOD_TEMPORAL_MODE_BOUNDED_HARD_COHORT
                }
            },
            Ordering::Release,
        );
    }

    /// Durably converts a completed transition into its exact target payload.
    ///
    /// The authored transition provenance remains on the immutable frontier,
    /// but recovery may move the render phase back through WAITING/PREPARED.
    /// Clearing the effective mode prevents those reset-like phases from
    /// resurrecting parent-union leases or bit-28 presentation entries after
    /// package ownership has already shrunk to the target.
    pub(crate) fn settle_temporal_transition(&self) {
        self.temporal_mode
            .store(LOD_TEMPORAL_MODE_NONE, Ordering::Release);
    }

    #[cfg(any(test, lod_render_path))]
    pub(crate) fn publish_render_claimed(&self) {
        self.render_claimed.store(true, Ordering::Release);
    }

    pub(crate) fn render_is_claimed(&self) -> bool {
        self.render_claimed.load(Ordering::Acquire)
    }

    /// Testing-only proof that this exact candidate identity crossed into a
    /// retained RenderView. A claimed-but-non-ACTIVE candidate distinguishes
    /// render admission from later pipeline/compaction activation stalls.
    #[cfg(feature = "testing")]
    pub fn render_is_claimed_for_testing(&self) -> bool {
        self.render_is_claimed()
    }

    /// Testing-only view of the full atlas union which must remain resident
    /// until this candidate reaches its exact target endpoint.
    #[cfg(feature = "testing")]
    pub fn required_atlas_ranges_for_testing(&self) -> &[LodPhysicalRange] {
        self.required_atlas_ranges()
    }

    /// Opaque identity of this candidate's two-phase render commit.
    ///
    /// Headless qualification uses this to prove that presentation-only
    /// settings do not replace an already-published package candidate. The
    /// value is process-local and has no semantic meaning beyond equality.
    #[cfg(feature = "testing")]
    pub fn render_commit_identity_for_testing(&self) -> usize {
        Arc::as_ptr(&self.phase) as usize
    }

    pub(crate) fn same_payload(&self, other: &Self) -> bool {
        if self.external_active_set != other.external_active_set {
            return false;
        }
        if self.external_active_set.is_some() {
            return self.frontier.same_render_payload(&other.frontier);
        }
        self.frontier.same_render_payload(&other.frontier)
            || ((self.render_is_active() || other.render_is_active())
                && self.temporal_transition_mode() != Some(LodTemporalTransitionMode::Morphing)
                && other.temporal_transition_mode() != Some(LodTemporalTransitionMode::Morphing)
                && self.frontier.same_settled_render_payload(&other.frontier))
    }

    /// Reuses an already-active physical payload without discarding semantic
    /// provenance carried by the retained cut (notably the package bootstrap
    /// fallback identity).
    pub(crate) fn inherit_active_payload_state(&mut self, previous: &Self) {
        debug_assert!(previous.render_is_active());
        debug_assert!(self.same_payload(previous));
        self.frontier
            .inherit_coverage_guard_identity(&previous.frontier);
        self.phase = previous.phase.clone();
        self.external_active_set = previous.external_active_set.clone();
        if self.frontier.same_render_payload(&previous.frontier) {
            self.view_blend_revision = previous.view_blend_revision.clone();
            self.view_blend_retirement_epoch = previous.view_blend_retirement_epoch.clone();
            self.view_blend_edge_count = previous.view_blend_edge_count.clone();
            self.view_blend_lagging_count = previous.view_blend_lagging_count.clone();
            self.view_blend_invalid_pressure_count =
                previous.view_blend_invalid_pressure_count.clone();
            self.view_blend_missing_consumer_count =
                previous.view_blend_missing_consumer_count.clone();
            self.view_blend_max_lag_bits = previous.view_blend_max_lag_bits.clone();
            self.view_blend_max_delta_bits = previous.view_blend_max_delta_bits.clone();
            self.view_blend_weighted_record_energy_bits =
                previous.view_blend_weighted_record_energy_bits.clone();
            self.view_blend_all_at_target = previous.view_blend_all_at_target.clone();
            self.view_blend_endpoints = previous.view_blend_endpoints.clone();
            #[cfg(any(test, feature = "testing"))]
            {
                self.view_blend_weights = previous.view_blend_weights.clone();
            }
            self.temporal_mode = previous.temporal_mode.clone();
            self.render_fallback = previous.render_fallback.clone();
            self.render_claimed = previous.render_claimed.clone();
        } else {
            // The compatibility path is categorical by construction; an ACTIVE
            // view blend may never masquerade as its exact selector endpoint.
            self.settle_temporal_transition();
        }
    }

    pub fn render_is_prepared(&self) -> bool {
        !self.render_hard_fallback_requested()
            && !self.view_blend_replan_requested()
            && matches!(
                self.phase.load(Ordering::Acquire),
                LOD_RENDER_PREPARED | LOD_RENDER_ACTIVE | LOD_RENDER_TRANSITIONING
            )
    }

    pub(crate) fn render_is_transitioning(&self) -> bool {
        !self.render_hard_fallback_requested()
            && !self.view_blend_replan_requested()
            && self.phase.load(Ordering::Acquire) == LOD_RENDER_TRANSITIONING
    }

    /// Testing-only phase proof for temporal render qualification. Transition
    /// provenance deliberately survives ACTIVE, so tests must not infer the
    /// live GPU phase from [`Self::temporal_transition_mode`].
    #[cfg(feature = "testing")]
    pub fn render_is_transitioning_for_testing(&self) -> bool {
        self.render_is_transitioning()
    }

    pub(crate) fn render_is_active(&self) -> bool {
        self.active_presentation().is_some()
    }

    pub(crate) fn active_presentation(&self) -> Option<LodRenderActivePresentation> {
        if self.phase.load(Ordering::Acquire) != LOD_RENDER_ACTIVE
            || self.render_hard_fallback_requested()
            || self.view_blend_replan_requested()
        {
            return None;
        }
        Some(
            if self.temporal_transition_mode() == Some(LodTemporalTransitionMode::Morphing) {
                LodRenderActivePresentation::ViewBlend
            } else {
                LodRenderActivePresentation::HardTarget
            },
        )
    }

    /// Testing-only proof that exact target descriptors and radix output have
    /// crossed the render-world activation boundary.
    #[cfg(feature = "testing")]
    pub fn render_is_active_for_testing(&self) -> bool {
        self.render_is_active()
    }

    pub fn failed(&self) -> bool {
        self.phase.load(Ordering::Acquire) == LOD_RENDER_FAILED
    }
}

fn complete_empty_candidate(candidate_count: u32, ranges: &[LodPhysicalRange]) -> bool {
    candidate_count == 0 && ranges.is_empty()
}

impl LodRenderCandidates {
    pub(crate) fn package_required() -> Self {
        Self {
            candidate_draw_required: true,
            debug_metadata_staged: true,
            ..Self::default()
        }
    }

    /// Whether a render-only morph veto must round-trip through package
    /// orchestration before any categorical target can activate. Package
    /// candidates use the entity's already-bound transient atlas and therefore
    /// have no bridge staging override; ephemeral bridge handshakes always
    /// carry their explicit bounded staging atlas.
    #[cfg(lod_render_path)]
    pub(crate) fn requires_package_hard_fallback_handshake(&self) -> bool {
        self.candidate_draw_required && self.staging_atlas.is_none()
    }

    pub fn insert(&mut self, camera: Entity, frontier: LodCandidateFrontier) {
        self.by_camera
            .insert(camera, LodRenderCandidate::new(frontier));
    }

    pub fn get(&self, camera: Entity) -> Option<&LodRenderCandidate> {
        self.by_camera.get(&camera)
    }

    pub fn len(&self) -> usize {
        self.by_camera.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_camera.is_empty()
    }

    /// Testing-only coherent package-retention flags for qualifying a
    /// retained drawable while an extracted replacement is still pending.
    ///
    /// Returns `(retained_current, candidates_are_current,
    /// retained_current_is_stale)` from this one extracted candidate set.
    #[cfg(any(test, feature = "testing"))]
    pub fn package_retention_for_testing(&self) -> (bool, bool, bool) {
        (
            self.retained_current,
            self.candidates_are_current,
            self.retained_current_is_stale,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AtlasSlotRecord {
    page: LodPageId,
    slot: AtlasSlot,
    materialized: bool,
}

/// Generation-safe CPU mirror of the bounded planar GPU atlas.
#[derive(Clone, Debug)]
pub struct LodPageAtlasMirror {
    layout: PageAtlasLayout,
    slot_count: u32,
    physical_gaussians: u32,
    slots: Vec<Option<AtlasSlotRecord>>,
}

impl LodPageAtlasMirror {
    pub fn new(layout: PageAtlasLayout, slot_count: u32) -> Result<Self, LodRenderCommitError> {
        if slot_count == 0 {
            return Err(LodRenderCommitError::ZeroAtlasSlots);
        }
        let physical_gaussians = slot_count
            .checked_mul(layout.gaussians_per_slot)
            .ok_or(LodRenderCommitError::AtlasSizeOverflow)?;
        let slots =
            usize::try_from(slot_count).map_err(|_| LodRenderCommitError::AtlasSizeOverflow)?;
        Ok(Self {
            layout,
            slot_count,
            physical_gaussians,
            slots: vec![None; slots],
        })
    }

    pub fn layout(&self) -> PageAtlasLayout {
        self.layout
    }

    pub fn slot_count(&self) -> u32 {
        self.slot_count
    }

    pub fn physical_gaussians(&self) -> u32 {
        self.physical_gaussians
    }

    /// Records a decoded page without mutating the visible fallback atlas.
    pub fn stage_page(
        &mut self,
        page: LodPageId,
        slot: AtlasSlot,
    ) -> Result<(), LodRenderCommitError> {
        let record = self
            .slots
            .get_mut(slot.index as usize)
            .ok_or(LodRenderCommitError::AtlasSlotOutOfRange(slot.index))?;
        *record = Some(AtlasSlotRecord {
            page,
            slot,
            materialized: false,
        });
        Ok(())
    }

    pub fn is_range_current(&self, range: LodPhysicalRange) -> bool {
        if !self.is_page_current(range.page, range.slot) {
            return false;
        }
        let Ok(expected_start) = self.layout.physical_index(range.slot, 0) else {
            return false;
        };
        range.physical_start >= expected_start
            && range
                .end()
                .is_some_and(|end| end <= expected_start + self.layout.gaussians_per_slot)
    }

    /// Returns whether the complete physical page currently occupies `slot`.
    /// Logical sibling ranges can use this to coalesce page-level work.
    pub fn is_page_current(&self, page: LodPageId, slot: AtlasSlot) -> bool {
        let Some(Some(record)) = self.slots.get(slot.index as usize) else {
            return false;
        };
        record.page == page && record.slot == slot && record.materialized
    }

    pub fn materialize_page(
        &mut self,
        atlas: &mut PlanarGaussian3d,
        page: &PlanarGaussian3dPage,
        slot: AtlasSlot,
    ) -> Result<(), LodRenderCommitError> {
        if atlas.len() != self.physical_gaussians as usize {
            return Err(LodRenderCommitError::AtlasLengthMismatch {
                expected: self.physical_gaussians,
                actual: atlas.len(),
            });
        }
        let payload = self.materialize_page_payload(page, slot)?;
        let start = slot
            .index
            .checked_mul(self.layout.gaussians_per_slot)
            .ok_or(LodRenderCommitError::AtlasSizeOverflow)? as usize;
        let end = start + self.layout.gaussians_per_slot as usize;
        for (offset, gaussian) in payload.iter().enumerate() {
            Planar::set(atlas, start + offset, gaussian);
        }
        debug_assert_eq!(start + payload.len(), end);
        Ok(())
    }

    /// Builds one fixed-stride physical payload without allocating or touching
    /// the rest of the atlas. Transient page-cache atlases keep this payload in
    /// sparse CPU staging storage until the corresponding GPU slot is retired.
    pub(crate) fn materialize_page_payload(
        &mut self,
        page: &PlanarGaussian3dPage,
        slot: AtlasSlot,
    ) -> Result<PlanarGaussian3d, LodRenderCommitError> {
        let Some(Some(record)) = self.slots.get(slot.index as usize) else {
            return Err(LodRenderCommitError::AtlasPageNotStaged(page.id));
        };
        if record.page != page.id || record.slot != slot {
            return Err(LodRenderCommitError::StaleAtlasSlot {
                page: page.id,
                slot,
            });
        }
        if page.gaussians.len() > self.layout.gaussians_per_slot as usize {
            return Err(LodRenderCommitError::PageExceedsAtlasStride {
                page: page.id,
                count: page.gaussians.len() as u32,
                stride: self.layout.gaussians_per_slot,
            });
        }
        let mut payload = PlanarGaussian3d::from(vec![
            Gaussian3d::default();
            self.layout.gaussians_per_slot as usize
        ]);
        for (offset, gaussian) in page.gaussians.iter().copied().enumerate() {
            Planar::set(&mut payload, offset, gaussian);
        }
        self.slots[slot.index as usize]
            .as_mut()
            .expect("validated staged slot")
            .materialized = true;
        Ok(payload)
    }

    pub(crate) fn materialized_slots(&self) -> Vec<AtlasSlot> {
        self.slots
            .iter()
            .flatten()
            .filter_map(|record| record.materialized.then_some(record.slot))
            .collect()
    }

    /// Invalidates one CPU mirror record after its retired GPU slot is cleared.
    pub(crate) fn clear_materialized_slot(
        &mut self,
        slot_index: u32,
    ) -> Result<(), LodRenderCommitError> {
        let record = self
            .slots
            .get_mut(slot_index as usize)
            .ok_or(LodRenderCommitError::AtlasSlotOutOfRange(slot_index))?;
        if let Some(record) = record {
            record.materialized = false;
        }
        Ok(())
    }

    pub fn validate_frontier(
        &self,
        frontier: &LodCandidateFrontier,
    ) -> Result<(), LodRenderCommitError> {
        self.validate_ranges(frontier.physical_ranges())
    }

    pub fn validate_ranges(&self, ranges: &[LodPhysicalRange]) -> Result<(), LodRenderCommitError> {
        for &range in ranges {
            if !self.is_range_current(range) {
                return Err(LodRenderCommitError::FrontierReferencesUnsynchronizedPage {
                    page: range.page,
                    slot: range.slot,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LodRenderCommitError {
    ZeroAtlasSlots,
    AtlasSizeOverflow,
    AtlasSlotOutOfRange(u32),
    AtlasPageNotStaged(LodPageId),
    StaleAtlasSlot {
        page: LodPageId,
        slot: AtlasSlot,
    },
    PageExceedsAtlasStride {
        page: LodPageId,
        count: u32,
        stride: u32,
    },
    AtlasLengthMismatch {
        expected: u32,
        actual: usize,
    },
    FrontierReferencesUnsynchronizedPage {
        page: LodPageId,
        slot: AtlasSlot,
    },
}

impl fmt::Display for LodRenderCommitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LodRenderCommitError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gaussian::formats::{lodge::LodgeClusterId, planar_3d_chunked::LodNodeId},
        stream::runtime::LodRuntimeViewId,
    };

    #[test]
    fn failure_codes_have_stable_categories_and_optional_detail() {
        let transport = LodOrchestrationFailure::with_detail(
            LodOrchestrationFailureCode::TransportRequestFailed,
            "request timed out",
        );
        assert_eq!(
            transport.category(),
            LodOrchestrationFailureCategory::Transport
        );
        assert_eq!(transport.detail(), Some("request timed out"));

        let render = LodOrchestrationFailure::new(LodOrchestrationFailureCode::RenderCommitFailed);
        assert_eq!(render.category(), LodOrchestrationFailureCategory::Render);
        assert_eq!(render.detail(), None);
    }

    #[test]
    fn failure_code_derives_its_category() {
        for code in [
            LodOrchestrationFailureCode::InvalidConfiguration,
            LodOrchestrationFailureCode::UnsupportedConfiguration,
            LodOrchestrationFailureCode::SourceUnavailable,
            LodOrchestrationFailureCode::TransportRequestFailed,
            LodOrchestrationFailureCode::TransportRequestsExhausted,
            LodOrchestrationFailureCode::CacheFailed,
            LodOrchestrationFailureCode::DecodeValidationFailed,
            LodOrchestrationFailureCode::RuntimeFailed,
            LodOrchestrationFailureCode::AtlasCommitFailed,
            LodOrchestrationFailureCode::RenderCommitFailed,
            LodOrchestrationFailureCode::CapacityExceeded,
            LodOrchestrationFailureCode::InternalInvariant,
        ] {
            let failure = LodOrchestrationFailure::new(code);
            assert_eq!(failure.category(), code.category());
        }
    }

    #[test]
    fn complete_empty_candidate_requires_no_render_publication() {
        assert!(complete_empty_candidate(0, &[]));
        assert!(!complete_empty_candidate(1, &[]));
        assert!(!complete_empty_candidate(
            0,
            &[LodPhysicalRange {
                node: LodNodeId(1),
                page: LodPageId(2),
                slot: AtlasSlot {
                    index: 3,
                    generation: 4,
                },
                physical_start: 24,
                count: 0,
            }]
        ));
    }

    #[test]
    fn external_active_set_candidate_keeps_pair_weights_separate_from_hierarchy_morphing() {
        let ranges = vec![
            LodPhysicalRange {
                node: LodNodeId(11),
                page: LodPageId(2),
                slot: AtlasSlot {
                    index: 0,
                    generation: 1,
                },
                physical_start: 0,
                count: 3,
            },
            LodPhysicalRange {
                node: LodNodeId(12),
                page: LodPageId(3),
                slot: AtlasSlot {
                    index: 1,
                    generation: 1,
                },
                physical_start: 3,
                count: 2,
            },
        ];
        let frontier =
            LodCandidateFrontier::complete_external_active_set(LodRuntimeViewId(7), ranges, false)
                .expect("validated resident ranges form a complete external union");
        let presentation = LodExternalActiveSetPresentation::new(
            LodgePairIdentity {
                first: LodgeClusterId(2),
                second: LodgeClusterId(9),
            },
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            vec![
                LodgeMembershipClass::Shared,
                LodgeMembershipClass::SecondOnly,
            ],
        )
        .expect("distinct finite cluster centers are valid");
        let candidate = LodRenderCandidate::new_external_active_set(frontier, presentation)
            .expect("one class per range constructs an external candidate");

        assert!(candidate.is_external_active_set());
        assert_eq!(candidate.temporal_transition(), None);
        assert_eq!(candidate.rendered_candidate_count(), 5);
        assert_eq!(
            candidate
                .external_active_set()
                .and_then(|presentation| presentation.opacity_weights([0.0, 0.0, 0.0])),
            Some((1.0, 0.0))
        );
        assert_eq!(
            candidate
                .external_active_set()
                .and_then(|presentation| presentation.opacity_weights([2.0, 0.0, 0.0])),
            Some((0.0, 1.0))
        );

        let frozen = LodExternalActiveSetPresentation::new(
            LodgePairIdentity {
                first: LodgeClusterId(2),
                second: LodgeClusterId(9),
            },
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            vec![LodgeMembershipClass::Shared],
        )
        .and_then(|presentation| presentation.with_frozen_second_weight(0.25))
        .expect("a finite authored Frozen coefficient is valid");
        assert_eq!(frozen.opacity_weights([2.0, 0.0, 0.0]), Some((0.75, 0.25)));
    }

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        feature = "sort_radix",
        not(feature = "buffer_texture")
    ))]
    fn render_hard_fallback_request_never_attests_active_or_mutates_authored_mode() {
        let settings = crate::GaussianLodSettings::default();
        let frontier = LodCandidateFrontier::complete_empty_for_test(
            crate::stream::runtime::LodRuntimeViewId::default(),
            &settings,
        )
        .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing);
        let candidate = LodRenderCandidate::new(frontier);
        assert_eq!(
            candidate.active_presentation(),
            Some(LodRenderActivePresentation::ViewBlend)
        );

        candidate.request_hard_fallback();
        assert!(candidate.render_hard_fallback_requested());
        assert_eq!(
            candidate.temporal_transition_mode(),
            Some(LodTemporalTransitionMode::Morphing),
            "render fallback requests must not rewrite package-authored mode"
        );
        assert_eq!(candidate.phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
        assert_eq!(candidate.active_presentation(), None);

        // Even a stale radix callback cannot make the package observe this
        // token as ACTIVE after the fallback request.
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
        assert_eq!(candidate.active_presentation(), None);
    }

    #[test]
    #[cfg(all(
        not(target_arch = "wasm32"),
        feature = "sort_radix",
        not(feature = "buffer_texture")
    ))]
    fn retirement_epoch_ignores_metrics_and_replan_preserves_the_authored_payload() {
        let settings = crate::GaussianLodSettings::default();
        let frontier = LodCandidateFrontier::complete_empty_for_test(
            crate::stream::runtime::LodRuntimeViewId(91),
            &settings,
        )
        .with_single_view_blend_edge_for_test();
        let mut candidate = LodRenderCandidate::new(frontier);
        let predecessor_identity = candidate
            .temporal_transition()
            .and_then(|transition| transition.morph())
            .expect("synthetic Morphing fixture has a batch identity")
            .identity();
        let retirement_epoch = candidate.view_blend_retirement_epoch.clone();
        let initial_epoch = retirement_epoch.load(Ordering::Acquire);
        candidate.set_predecessor_view_blend_attestation(LodViewBlendPredecessorAttestation {
            predecessor_identity,
            retirement_epoch: retirement_epoch.clone(),
            expected_retirement_epoch: initial_epoch,
            requirements: Arc::from([]),
        });
        assert!(
            candidate.predecessor_view_blend_attestation_is_current(Some(predecessor_identity))
        );

        assert!(candidate.publish_view_blend_aggregate_snapshot(
            &[0.0],
            &[0.0],
            0,
            0,
            1,
            0.0,
            0.25,
            3.0,
            &[LodViewBlendEndpoint::ParentExact],
        ));
        assert_eq!(retirement_epoch.load(Ordering::Acquire), initial_epoch);
        assert!(
            candidate.predecessor_view_blend_attestation_is_current(Some(predecessor_identity))
        );

        assert!(candidate.publish_view_blend_aggregate_snapshot(
            &[0.0],
            &[0.0],
            0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            &[LodViewBlendEndpoint::ParentExact],
        ));
        assert_ne!(retirement_epoch.load(Ordering::Acquire), initial_epoch);
        assert!(
            !candidate.predecessor_view_blend_attestation_is_current(Some(predecessor_identity))
        );

        let missing_epoch = retirement_epoch.load(Ordering::Acquire);
        candidate.set_predecessor_view_blend_attestation(LodViewBlendPredecessorAttestation {
            predecessor_identity,
            retirement_epoch: retirement_epoch.clone(),
            expected_retirement_epoch: missing_epoch,
            requirements: Arc::from([]),
        });
        assert!(candidate.publish_view_blend_aggregate_snapshot(
            &[0.0],
            &[0.0],
            0,
            1,
            0,
            0.0,
            0.0,
            0.0,
            &[LodViewBlendEndpoint::ParentExact],
        ));
        assert_ne!(retirement_epoch.load(Ordering::Acquire), missing_epoch);

        let invalid_epoch = retirement_epoch.load(Ordering::Acquire);
        candidate.set_predecessor_view_blend_attestation(LodViewBlendPredecessorAttestation {
            predecessor_identity,
            retirement_epoch: retirement_epoch.clone(),
            expected_retirement_epoch: invalid_epoch,
            requirements: Arc::from([]),
        });
        assert!(candidate.publish_view_blend_aggregate_snapshot(
            &[1.0],
            &[1.0],
            0,
            1,
            0,
            0.0,
            0.0,
            0.0,
            &[LodViewBlendEndpoint::ChildrenExact],
        ));
        assert_ne!(retirement_epoch.load(Ordering::Acquire), invalid_epoch);
        assert!(
            !candidate.predecessor_view_blend_attestation_is_current(Some(predecessor_identity))
        );

        let published = candidate
            .view_blend_testing_snapshot()
            .expect("coherent view-blend publication");
        candidate.request_view_blend_replan();
        assert!(candidate.view_blend_replan_requested());
        assert!(!candidate.render_hard_fallback_requested());
        assert_eq!(candidate.phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
        assert_eq!(
            candidate.temporal_transition_mode(),
            Some(LodTemporalTransitionMode::Morphing)
        );
        assert_eq!(candidate.view_blend_testing_snapshot(), Some(published));

        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
        assert_eq!(candidate.active_presentation(), None);
    }
}
