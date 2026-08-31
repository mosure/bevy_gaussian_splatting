//! Bounded runtime orchestration for hierarchy selection, page requests, cache
//! residency, decoded pages, and physical candidate ranges.
//!
//! The controller is deliberately renderer- and async-runtime-neutral. A game
//! supplies a [`LodPageTransport`], calls [`LodStreamingRuntime::update`] once
//! per view/update epoch, and uploads newly decoded pages outside the render
//! pass. No operation allocates in proportion to the manifest's virtual source
//! Gaussian count.

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fmt,
    num::NonZeroU32,
    sync::Arc,
};

use bevy::platform::collections::HashMap;
use bevy::prelude::Reflect;

use crate::{
    gaussian::{
        formats::{
            planar_3d_chunked::{
                LodBounds, LodNodeId, LodPageDescriptor, LodPageId, LodPageRange,
                PlanarGaussian3dPage,
            },
            planar_3d_lod::{
                GaussianLodManifest, LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES, MOMENT_MERGE_VERSION,
                gaussian_support_bounds,
            },
        },
        lod_settings::{
            GaussianLodSettings, GaussianStreamingSettings, LodDegradation, LodEffectiveStatus,
            LodQualityEndpoint, LodQualityTarget, LodSelectionMode,
        },
    },
    io::lod::LodCodecLimits,
    stream::{
        cache::{AtlasSlot, LodPageCache, PageCacheError, PageCacheLimits, PageCacheStats},
        hierarchy::{
            CompiledManifestLodHierarchy, LodFrontier, LodHierarchy, LodNodeMetrics,
            LodSelectionError, LodTemporalDirection, LodTemporalStepBudget,
            LodTemporalSubstitution, LodTemporalSubstitutionKey, LodView, ManifestHierarchyError,
            apply_temporal_substitution_step, select_frontier_with_visibility,
            temporal_frontier_with_visibility, temporal_substitution_candidates,
        },
        preprocess::{
            LodPagePreprocessAdmissionError, LodPagePreprocessError, LodPagePreprocessInput,
            LodPagePreprocessStats, LodPagePreprocessor,
        },
        transport::{
            LodPageTransport, LodPageTransportFailure, PagePoll, PageRequest, PageRequestClass,
            PageRequestPriority, PageRequestQueue, RequestEnqueue, RequestQueueError,
        },
    },
};

#[cfg(test)]
use crate::stream::hierarchy::{
    select_frontier_with_previous_and_visibility,
    select_frontier_with_previous_holds_and_visibility,
};

/// Fixed-stride physical addressing used by a bounded decoded GPU page atlas.
/// The stride is independent of the virtual source size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageAtlasLayout {
    pub gaussians_per_slot: u32,
}

impl PageAtlasLayout {
    pub fn new(gaussians_per_slot: u32) -> Result<Self, LodRuntimeError> {
        if gaussians_per_slot == 0 {
            Err(LodRuntimeError::ZeroAtlasStride)
        } else {
            Ok(Self { gaussians_per_slot })
        }
    }

    pub fn physical_index(self, slot: AtlasSlot, page_offset: u32) -> Result<u32, LodRuntimeError> {
        if page_offset >= self.gaussians_per_slot {
            return Err(LodRuntimeError::PageRangeExceedsAtlasStride {
                offset: page_offset,
                count: 1,
                stride: self.gaussians_per_slot,
            });
        }
        slot.index
            .checked_mul(self.gaussians_per_slot)
            .and_then(|start| start.checked_add(page_offset))
            .ok_or(LodRuntimeError::PhysicalIndexOverflow)
    }
}

/// A contiguous active representation in a generation-safe physical atlas.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodPhysicalRange {
    pub node: LodNodeId,
    pub page: LodPageId,
    pub slot: AtlasSlot,
    pub physical_start: u32,
    pub count: u32,
}

/// A complete, resident, generation-safe frontier represented by bounded
/// physical ranges. Construction is restricted to
/// [`LodStreamFrame::candidate_frontier`] so render code cannot accidentally
/// commit an arbitrary prefix or duplicate list as a complete scene
/// representation.
#[derive(Clone, Debug, PartialEq)]
pub struct LodCandidateFrontier {
    view: LodRuntimeViewId,
    physical_ranges: Vec<LodPhysicalRange>,
    /// Per-view selected ancestors that stand in for requested descendants
    /// whose pages are not resident. Keeping this provenance with the exact
    /// candidate lets the renderer attach Residency to the sorted output that
    /// owns it instead of racing a cloud-wide sidecar update.
    ancestor_fallback_nodes: BTreeSet<LodNodeId>,
    candidate_count: u32,
    quality_status: LodEffectiveStatus,
    selection_view_frozen: bool,
    /// This candidate is the runtime-owned globally covering coarse guard, not
    /// the ordinary camera-selected detail frontier.
    coverage_guard: bool,
    /// Density-correct topology transactions that produced this exact cut.
    /// ABI 16 packages attach their authored parent-record runs here so the
    /// renderer can morph optical depth without changing selection policy;
    /// older packages retain the bounded categorical endpoint.
    temporal_transition: Option<LodTemporalTransition>,
}

/// Truthful presentation capability for one bounded hierarchy transaction.
///
/// ABI 16 packages carry the canonical parent-record run map needed for a
/// density-preserving optical-depth morph. Older readable packages remain
/// correct and bounded, but their complete-cut cohort changes are categorical.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Reflect)]
pub enum LodTemporalTransitionMode {
    Morphing,
    BoundedHardCohort,
}

/// One direct child-to-parent lookup consumed by both LoD compaction and
/// rasterization. `split_count` is the number of child-cardinality records
/// which share the same parent representative at the parent endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodTemporalMorphRecord {
    pub parent_physical_index: u32,
    pub split_count: u32,
}

/// Bit-exact selector inputs for one endpoint of an adjacent hierarchy edge.
///
/// The immutable batch stores bits rather than live `f32` values so content
/// identity remains exact while every retained render view evaluates pressure
/// from its own current camera. Appearance and opacity error do not participate
/// in the current selector pressure and therefore are intentionally absent.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LodViewBlendMetric {
    center_bits: [u32; 3],
    radius_bits: u32,
    geometric_error_bits: u32,
    quality_min_bits: u32,
    quality_max_bits: u32,
    high_fidelity_certificate_bits: u32,
    original_representation: bool,
}

impl LodViewBlendMetric {
    fn from_node(metrics: LodNodeMetrics, original_representation: bool) -> Self {
        Self {
            center_bits: metrics.center.to_array().map(f32::to_bits),
            radius_bits: metrics.radius.to_bits(),
            geometric_error_bits: metrics.geometric_error.to_bits(),
            quality_min_bits: metrics.quality_min.to_bits(),
            quality_max_bits: metrics.quality_max.to_bits(),
            high_fidelity_certificate_bits: metrics.high_fidelity_certificate.to_bits(),
            original_representation,
        }
    }

    pub fn node_metrics(self) -> LodNodeMetrics {
        LodNodeMetrics {
            center: bevy::math::Vec3::from_array(self.center_bits.map(f32::from_bits)),
            radius: f32::from_bits(self.radius_bits),
            geometric_error: f32::from_bits(self.geometric_error_bits),
            appearance_error: 0.0,
            opacity_error: 0.0,
            quality_min: f32::from_bits(self.quality_min_bits),
            quality_max: f32::from_bits(self.quality_max_bits),
            high_fidelity_certificate: f32::from_bits(self.high_fidelity_certificate_bits),
            // Projection and pressure do not inspect representative count. A
            // positive sentinel keeps the reconstructed metric valid for debug
            // assertions and future callers which validate it first.
            representative_count: 1,
        }
    }

    pub const fn is_original_representation(self) -> bool {
        self.original_representation
    }
}

/// One direction-independent adjacent parent/children pressure boundary.
///
/// `initial_weight` is the exact endpoint already visible when this immutable
/// topology first becomes renderable: zero for a retained parent and one for
/// retained children. It participates in batch identity. Per-frame displayed
/// and desired weights are render-owned mutable state and never do.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LodViewBlendEdge {
    parent: LodNodeId,
    children: Vec<LodNodeId>,
    parent_metric: LodViewBlendMetric,
    child_metrics: Vec<LodViewBlendMetric>,
    initial_weight_bits: u32,
    activation_requires_slew: bool,
}

impl LodViewBlendEdge {
    pub const fn parent(&self) -> LodNodeId {
        self.parent
    }

    pub fn children(&self) -> &[LodNodeId] {
        &self.children
    }

    pub const fn parent_metric(&self) -> LodViewBlendMetric {
        self.parent_metric
    }

    pub fn child_metrics(&self) -> &[LodViewBlendMetric] {
        &self.child_metrics
    }

    pub fn initial_weight(&self) -> f32 {
        f32::from_bits(self.initial_weight_bits)
    }

    pub const fn initial_weight_bits(&self) -> u32 {
        self.initial_weight_bits
    }

    /// Exceptional continuity fallback for an edge whose child cohort became
    /// drawable only after the retained parent had already crossed the band.
    /// Fully resident Dynamic edges remain stateless and exact.
    pub const fn activation_requires_slew(&self) -> bool {
        self.activation_requires_slew
    }
}

/// Evaluates one immutable adjacent boundary against the current effective LoD
/// view. The endpoints meet successive selector decisions exactly:
///
/// - `P_parent <= 1` and `P_child_max <= 1` is the exact parent endpoint;
/// - `P_parent > 1` and `P_child_max >= 1` is the exact children endpoint;
/// - `P_parent > 1 > P_child_max` is the open interval, where the linear weight
///   keeps the interpolated endpoint pressure on the selector boundary.
///
/// Pressure ordering matters only inside the open blend interval. Equal or
/// reversed pressures on the same categorical side are still unambiguous:
/// both endpoints make the same selector decision. A threshold-contradictory
/// pair (`P_parent <= 1 < P_child_max`) fails closed to the retained endpoint.
pub(crate) fn lod_view_blend_weight(
    view: LodView,
    target: LodQualityTarget,
    edge: &LodViewBlendEdge,
) -> f32 {
    let retained = match edge.initial_weight_bits() {
        bits if bits == 1.0_f32.to_bits() => 1.0,
        _ => 0.0,
    };
    lod_view_blend_weight_checked(view, target, edge).unwrap_or(retained)
}

/// Checked camera-conditioned weight used by the render loop.
///
/// Unlike [`lod_view_blend_weight`], this does not substitute the immutable
/// admission endpoint for an invalid later view evaluation. RenderWorld must
/// retain its last drawable weight and publish explicit degradation instead of
/// snapping an ACTIVE fractional edge back to its historical endpoint.
pub(crate) fn lod_view_blend_weight_checked(
    view: LodView,
    target: LodQualityTarget,
    edge: &LodViewBlendEdge,
) -> Option<f32> {
    let (parent_pressure, child_pressure) = lod_view_blend_raw_pressures(view, target, edge)?;
    lod_view_blend_weight_from_pressures_checked(parent_pressure, child_pressure)
}

#[cfg(test)]
fn lod_view_blend_weight_from_pressures(
    parent_pressure: f32,
    child_pressure: f32,
    retained: f32,
) -> f32 {
    lod_view_blend_weight_from_pressures_checked(parent_pressure, child_pressure)
        .unwrap_or(retained)
}

fn lod_view_blend_weight_from_pressures_checked(
    parent_pressure: f32,
    child_pressure: f32,
) -> Option<f32> {
    if !parent_pressure.is_finite() || !child_pressure.is_finite() {
        return None;
    }
    if parent_pressure <= 1.0 {
        return (child_pressure <= 1.0).then_some(0.0);
    }
    if child_pressure >= 1.0 {
        return Some(1.0);
    }
    // This is the only open-interval case: parent > 1 > child. The span is
    // necessarily finite and positive, even when same-side hierarchy metrics
    // are equal or reversed at other views.
    let span = parent_pressure - child_pressure;
    if !span.is_finite() || span <= 0.0 {
        return None;
    }
    Some(((parent_pressure - 1.0) / span).clamp(0.0, 1.0))
}

fn lod_view_blend_pressures(
    view: LodView,
    target: LodQualityTarget,
    edge: &LodViewBlendEdge,
) -> Option<(f32, f32)> {
    let (parent_pressure, child_pressure) = lod_view_blend_raw_pressures(view, target, edge)?;
    lod_view_blend_weight_from_pressures_checked(parent_pressure, child_pressure)
        .map(|_| (parent_pressure, child_pressure))
}

fn lod_view_blend_raw_pressures(
    view: LodView,
    target: LodQualityTarget,
    edge: &LodViewBlendEdge,
) -> Option<(f32, f32)> {
    if edge.children.is_empty() || edge.children.len() != edge.child_metrics.len() {
        return None;
    }
    let parent = edge.parent_metric;
    let parent_metrics = parent.node_metrics();
    if !parent_metrics.validate() {
        return None;
    }
    let parent_pressure =
        view.selection_pressure(parent_metrics, target, parent.is_original_representation());
    if !parent_pressure.is_finite() {
        return None;
    }
    let mut child_pressure = 0.0_f32;
    for metric in edge.child_metrics.iter().copied() {
        let child_metrics = metric.node_metrics();
        if !child_metrics.validate() {
            return None;
        }
        let pressure =
            view.selection_pressure(child_metrics, target, metric.is_original_representation());
        if !pressure.is_finite() {
            return None;
        }
        child_pressure = child_pressure.max(pressure);
    }
    Some((parent_pressure, child_pressure))
}

/// Exact pressure pair consumed by the internal view-blend weight oracle.
///
/// Qualification uses this oracle to reject active edges whose projected
/// parent/child pressures are non-finite or threshold-contradictory. Same-side
/// equality or reversed ordering is a valid categorical endpoint. `None` is
/// therefore a validation failure, not an endpoint classification.
#[cfg(any(test, feature = "testing"))]
pub fn lod_view_blend_pressures_for_testing(
    view: LodView,
    target: LodQualityTarget,
    edge: &LodViewBlendEdge,
) -> Option<(f32, f32)> {
    lod_view_blend_pressures(view, target, edge)
}

fn lod_view_blend_batch_pressures_are_valid(
    view: LodView,
    target: LodQualityTarget,
    edges: &[LodViewBlendEdge],
) -> bool {
    edges
        .iter()
        .all(|edge| lod_view_blend_pressures(view, target, edge).is_some())
}

/// One contiguous child atlas range and its record-relative direct-map start.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodTemporalMorphDescriptor {
    pub child_physical_start: u32,
    pub child_count: u32,
    pub mapping_start: u32,
    /// Dense index into [`LodTemporalMorphBatch::edges`]. Direction never enters
    /// the shader payload; one parent-to-child weight applies in both directions.
    pub edge_index: u32,
}

fn view_blend_batch_structure_is_valid(
    edges: &[LodViewBlendEdge],
    descriptors: &[LodTemporalMorphDescriptor],
) -> bool {
    let mut endpoint_nodes = HashSet::new();
    edges.iter().all(|edge| {
        !edge.children().is_empty()
            && edge.children().len() == edge.child_metrics().len()
            && endpoint_nodes.insert(edge.parent())
            && edge
                .children()
                .iter()
                .all(|child| endpoint_nodes.insert(*child))
    }) && descriptors
        .iter()
        .all(|descriptor| (descriptor.edge_index as usize) < edges.len())
}

/// Canonical physical source order for an ABI16 presentation.
///
/// ABI16 direct-map runs concatenate immediate children in manifest child-range
/// order. Portable node IDs are opaque and may be sparse or numerically
/// scrambled, so ordering these ranges by `LodNodeId` can split a parent's
/// equal-depth proxy run around an unrelated cohort. Manifest indexes are
/// unique and preserve every validated contiguous immediate-child range.
fn manifest_ordered_presentation_nodes(
    hierarchy: &CompiledManifestLodHierarchy,
    nodes: &[LodNodeId],
) -> Result<Vec<LodNodeId>, LodNodeId> {
    let mut indexed = nodes
        .iter()
        .copied()
        .map(|node| {
            hierarchy
                .node_index(node)
                .map(|manifest_index| (manifest_index, node))
                .ok_or(node)
        })
        .collect::<Result<Vec<_>, _>>()?;
    indexed.sort_unstable_by_key(|(manifest_index, _)| *manifest_index);
    Ok(indexed.into_iter().map(|(_, node)| node).collect())
}

fn manifest_ordered_morph_presentation_ranges(
    hierarchy: &CompiledManifestLodHierarchy,
    mut presentation: BTreeMap<LodNodeId, LodPhysicalRange>,
) -> Option<Vec<LodPhysicalRange>> {
    let numeric_nodes = presentation.keys().copied().collect::<Vec<_>>();
    manifest_ordered_presentation_nodes(hierarchy, &numeric_nodes)
        .ok()?
        .into_iter()
        .map(|node| presentation.remove(&node))
        .collect()
}

/// Bounded destination-cardinality morph payload for one complete-cut cohort.
///
/// `presentation_ranges` is always a complete antichain. Refinement uses the
/// target children directly; coarsening temporarily retains the old children
/// until their parent-split endpoint is exact. `required_ranges` is the
/// generation-safe union of presentation, target, and parent lookup sources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LodTemporalMorphBatch {
    identity: LodTemporalMorphIdentity,
    presentation_ranges: Vec<LodPhysicalRange>,
    required_ranges: Vec<LodPhysicalRange>,
    edges: Vec<LodViewBlendEdge>,
    descriptors: Vec<LodTemporalMorphDescriptor>,
    records: Vec<LodTemporalMorphRecord>,
}

/// Compact immutable content identity for a morph batch. Runtime and render
/// hot paths compare this instead of scanning the expanded direct-record map.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LodTemporalMorphIdentity {
    primary: u64,
    secondary: u64,
    descriptor_count: u32,
    mapping_record_count: u32,
}

impl LodTemporalMorphIdentity {
    pub fn primary(self) -> u64 {
        self.primary
    }

    pub fn secondary(self) -> u64 {
        self.secondary
    }

    pub fn descriptor_count(self) -> u32 {
        self.descriptor_count
    }

    pub fn mapping_record_count(self) -> u32 {
        self.mapping_record_count
    }
}

impl LodTemporalMorphBatch {
    pub fn identity(&self) -> LodTemporalMorphIdentity {
        self.identity
    }

    pub fn presentation_ranges(&self) -> &[LodPhysicalRange] {
        &self.presentation_ranges
    }

    pub fn required_ranges(&self) -> &[LodPhysicalRange] {
        &self.required_ranges
    }

    pub fn edges(&self) -> &[LodViewBlendEdge] {
        &self.edges
    }

    pub fn descriptors(&self) -> &[LodTemporalMorphDescriptor] {
        &self.descriptors
    }

    pub fn records(&self) -> &[LodTemporalMorphRecord] {
        &self.records
    }
}

/// Camera-continuous presentation names. The on-disk ABI retains its historical
/// `morph_map` spelling; new runtime/render code should use these aliases.
pub type LodViewBlendRecord = LodTemporalMorphRecord;
pub type LodViewBlendDescriptor = LodTemporalMorphDescriptor;
pub type LodViewBlendBatch = LodTemporalMorphBatch;
pub type LodViewBlendIdentity = LodTemporalMorphIdentity;
pub type LodViewBlendMode = LodTemporalTransitionMode;
pub type LodViewBlend = LodTemporalTransition;

/// Observable bounded work used to advance one complete hierarchy cut.
#[derive(Clone, Debug, PartialEq)]
pub struct LodTemporalTransition {
    substitutions: Vec<LodTemporalSubstitution<LodNodeId>>,
    /// Retained drawable endpoint for each substitution when this immutable
    /// edge table first appears. Stable child-frontier boundaries start at one
    /// even though their direction-independent topology is represented by a
    /// refine-shaped parent/children substitution.
    initial_weight_bits: Vec<u32>,
    changed_gaussians: u64,
    atomic_budget_overshoot: u64,
    mode: LodTemporalTransitionMode,
    morph: Option<Arc<LodTemporalMorphBatch>>,
}

impl LodTemporalTransition {
    pub fn substitutions(&self) -> &[LodTemporalSubstitution<LodNodeId>] {
        &self.substitutions
    }

    pub fn changed_gaussians(&self) -> u64 {
        self.changed_gaussians
    }

    pub fn atomic_budget_overshoot(&self) -> u64 {
        self.atomic_budget_overshoot
    }

    fn initial_weight_bits(&self, edge_index: usize) -> Option<u32> {
        self.initial_weight_bits.get(edge_index).copied()
    }

    /// Compatibility accessor for the removed cohort clock. View-blend
    /// weights are independently evaluated from each current render view.
    pub fn progress(&self) -> f32 {
        0.0
    }

    pub fn mode(&self) -> LodTemporalTransitionMode {
        self.mode
    }

    pub fn morph(&self) -> Option<&LodTemporalMorphBatch> {
        self.morph.as_deref()
    }

    fn same_render_payload(&self, other: &Self) -> bool {
        self.mode == other.mode
            && self.morph.as_ref().map(|batch| batch.identity())
                == other.morph.as_ref().map(|batch| batch.identity())
    }
}

impl LodCandidateFrontier {
    /// Constructs the complete range union produced by the validated external
    /// active-set planner. This is intentionally crate-private: arbitrary
    /// callers still cannot manufacture a hierarchy frontier, while the LODGE
    /// adapter can reuse the established compaction/radix handshake without
    /// pretending its catalog is a Morton antichain.
    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) fn complete_external_active_set(
        view: LodRuntimeViewId,
        physical_ranges: Vec<LodPhysicalRange>,
        selection_view_frozen: bool,
    ) -> Option<Self> {
        if physical_ranges.iter().any(|range| {
            range.count == 0
                || !range.page.is_valid()
                || range.end().is_none()
                || range.slot.generation == 0
        }) {
            return None;
        }
        let candidate_count = physical_ranges
            .iter()
            .try_fold(0_u32, |total, range| total.checked_add(range.count))?;
        Some(Self {
            view,
            physical_ranges,
            ancestor_fallback_nodes: BTreeSet::new(),
            candidate_count,
            quality_status: LodEffectiveStatus {
                // External active-set selection has its own public status and
                // no hierarchy quality endpoint. `Coarsest` is an internal
                // render-adapter sentinel only; it is never surfaced as the
                // requested LODGE target.
                requested_target: LodQualityTarget::Coarsest,
                achieved_max_error_px: 0.0,
                achieved_max_target_ratio: 0.0,
                degradation: LodDegradation::None,
                active_gaussians: u64::from(candidate_count),
                visited_nodes: 0,
                requested_pages: 0,
            },
            selection_view_frozen,
            coverage_guard: false,
            temporal_transition: None,
        })
    }

    /// Defensive package-orchestration fixture for a complete cut that reads no
    /// atlas data. Production selectors currently retain a globally covering
    /// frontier and therefore cannot naturally emit this shape.
    #[cfg(all(
        test,
        not(target_arch = "wasm32"),
        feature = "sort_radix",
        not(feature = "buffer_texture")
    ))]
    pub(crate) fn complete_empty_for_test(
        view: LodRuntimeViewId,
        settings: &GaussianLodSettings,
    ) -> Self {
        Self {
            view,
            physical_ranges: Vec::new(),
            ancestor_fallback_nodes: BTreeSet::new(),
            candidate_count: 0,
            quality_status: LodEffectiveStatus {
                requested_target: settings.quality_target(),
                ..Default::default()
            },
            selection_view_frozen: settings.selection_mode == LodSelectionMode::Frozen,
            coverage_guard: false,
            temporal_transition: None,
        }
    }

    pub fn view(&self) -> LodRuntimeViewId {
        self.view
    }

    pub fn physical_ranges(&self) -> &[LodPhysicalRange] {
        &self.physical_ranges
    }

    /// Whether this physical range's logical node is serving as a resident
    /// ancestor fallback for this candidate's view.
    pub(crate) fn is_ancestor_fallback(&self, node: LodNodeId) -> bool {
        self.ancestor_fallback_nodes.contains(&node)
    }

    /// Equality of every field encoded into the per-view GPU candidate payload.
    /// Quality/status may change without recomputing the sorted entries, but a
    /// Residency provenance change may not inherit an already-active phase.
    pub(crate) fn same_render_payload(&self, other: &Self) -> bool {
        self.view == other.view
            && self.candidate_count == other.candidate_count
            && self.physical_ranges == other.physical_ranges
            && self.ancestor_fallback_nodes == other.ancestor_fallback_nodes
            && match (&self.temporal_transition, &other.temporal_transition) {
                (Some(left), Some(right)) => left.same_render_payload(right),
                (None, None) => true,
                _ => false,
            }
    }

    /// Equality of the exact settled GPU payload after a temporal transition
    /// has reached its target endpoint. Transition provenance is deliberately
    /// excluded: an ACTIVE morph candidate and the following stable selector
    /// frame both render `physical_ranges`, so restaging that identical cut
    /// would be redundant and could create a one-frame readiness dropout.
    pub(crate) fn same_settled_render_payload(&self, other: &Self) -> bool {
        self.view == other.view
            && self.candidate_count == other.candidate_count
            && self.physical_ranges == other.physical_ranges
            && self.ancestor_fallback_nodes == other.ancestor_fallback_nodes
    }

    pub fn candidate_count(&self) -> u32 {
        self.candidate_count
    }

    /// Quality target and achieved error of the complete resident cut used to
    /// construct this render candidate.
    pub fn quality_status(&self) -> &LodEffectiveStatus {
        &self.quality_status
    }

    /// True when this candidate was selected against the view snapshot captured
    /// on entry into [`LodSelectionMode::Frozen`].
    pub fn selection_view_frozen(&self) -> bool {
        self.selection_view_frozen
    }

    /// True for the bounded, permanently resident coarse cut that remains safe
    /// across arbitrary camera changes.
    pub fn is_coverage_guard(&self) -> bool {
        self.coverage_guard
    }

    /// Bounded topology work that produced this candidate, if any.
    pub fn temporal_transition(&self) -> Option<&LodTemporalTransition> {
        self.temporal_transition.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn with_temporal_transition_for_test(
        mut self,
        mode: LodTemporalTransitionMode,
    ) -> Self {
        let morph = (mode == LodTemporalTransitionMode::Morphing).then(|| {
            Arc::new(LodTemporalMorphBatch {
                identity: LodTemporalMorphIdentity {
                    primary: 1,
                    secondary: 2,
                    descriptor_count: 0,
                    mapping_record_count: 0,
                },
                presentation_ranges: self.physical_ranges.clone(),
                required_ranges: self.physical_ranges.clone(),
                edges: Vec::new(),
                descriptors: Vec::new(),
                records: Vec::new(),
            })
        });
        self.temporal_transition = Some(LodTemporalTransition {
            substitutions: Vec::new(),
            initial_weight_bits: Vec::new(),
            changed_gaussians: 0,
            atomic_budget_overshoot: 0,
            mode,
            morph,
        });
        self
    }

    /// One-edge Morphing fixture for status/publication tests whose counters
    /// must satisfy the same edge-cardinality invariants as production.
    #[cfg(test)]
    pub(crate) fn with_single_view_blend_edge_for_test(mut self) -> Self {
        let metric = LodViewBlendMetric::from_node(
            LodNodeMetrics {
                center: bevy::math::Vec3::ZERO,
                radius: 1.0,
                geometric_error: 1.0,
                appearance_error: 0.0,
                opacity_error: 0.0,
                quality_min: 0.0,
                quality_max: 1.0,
                high_fidelity_certificate: 1.0,
                representative_count: 1,
            },
            false,
        );
        let parent = LodNodeId(1);
        let child = LodNodeId(2);
        let initial_weight_bits = 0.0_f32.to_bits();
        self.temporal_transition = Some(LodTemporalTransition {
            substitutions: vec![LodTemporalSubstitution {
                key: LodTemporalSubstitutionKey {
                    parent,
                    direction: LodTemporalDirection::Refine,
                },
                previous_nodes: vec![parent],
                next_nodes: vec![child],
                previous_gaussians: 1,
                next_gaussians: 1,
            }],
            initial_weight_bits: vec![initial_weight_bits],
            changed_gaussians: 2,
            atomic_budget_overshoot: 0,
            mode: LodTemporalTransitionMode::Morphing,
            morph: Some(Arc::new(LodTemporalMorphBatch {
                identity: LodTemporalMorphIdentity {
                    primary: 3,
                    secondary: 4,
                    descriptor_count: 0,
                    mapping_record_count: 0,
                },
                presentation_ranges: self.physical_ranges.clone(),
                required_ranges: self.physical_ranges.clone(),
                edges: vec![LodViewBlendEdge {
                    parent,
                    children: vec![child],
                    parent_metric: metric,
                    child_metrics: vec![metric],
                    initial_weight_bits,
                    activation_requires_slew: false,
                }],
                descriptors: Vec::new(),
                records: Vec::new(),
            })),
        });
        self
    }

    /// Preserves the globally covering fallback identity when a newly selected
    /// frontier is byte-for-byte the same physical cut. The quality status may
    /// be rebound to a new request, but losing this identity would prevent the
    /// package orchestrator from planning the later bootstrap-to-target
    /// handoff.
    pub(crate) fn inherit_coverage_guard_identity(&mut self, previous: &Self) {
        debug_assert_eq!(self.view, previous.view);
        debug_assert_eq!(self.candidate_count, previous.candidate_count);
        debug_assert_eq!(self.physical_ranges, previous.physical_ranges);
        self.coverage_guard |= previous.coverage_guard;
    }
}

/// Stable application-provided identity for independent camera/subview state.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LodRuntimeViewId(pub u64);

/// Opaque token that groups every camera update belonging to one application
/// frame. Per-frame request and decoded-byte budgets are shared by all updates
/// made with the same token.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LodRuntimeFrameId(u64);

impl LodRuntimeFrameId {
    pub fn sequence(self) -> u64 {
        self.0
    }
}

impl LodPhysicalRange {
    pub fn end(self) -> Option<u32> {
        self.physical_start.checked_add(self.count)
    }
}

/// Observable result for one controller update.
#[derive(Clone, Debug, PartialEq)]
pub struct LodStreamFrame {
    view: LodRuntimeViewId,
    frontier: LodFrontier<LodNodeId>,
    physical_ranges: Vec<LodPhysicalRange>,
    ancestor_fallback_nodes: BTreeSet<LodNodeId>,
    selection_view_frozen: bool,
    /// True only after an unchanged selection reproduces the previous logical
    /// frontier for the exact view, policy, and residency revision. Transport
    /// quiescence alone is insufficient: temporal topology work deliberately
    /// releases independent complete-cut cohorts over several frames.
    selection_stable: bool,
    temporal_transition: Option<LodTemporalTransition>,
    /// True when every visible root is covered by a resident selected node.
    complete_resident_cut: bool,
    cache: PageCacheStats,
    queued_requests: u32,
    /// Transport requests plus admitted preprocessing work. Keeping both in a
    /// single pipeline count prevents completion checks from declaring idle
    /// while validated pages are still pending publication.
    in_flight_requests: u32,
    preprocess: LodPagePreprocessStats,
    /// Requests paused after a decoded page could not displace pinned fallbacks.
    capacity_blocked_requests: u32,
    /// Exact retained-cut admission failure for one atomic parent-to-children
    /// substitution. This is status, not an update error: `frontier` remains a
    /// complete resident ancestor cut while package orchestration decides
    /// whether publishing a slot-releasing cut can make progress.
    split_cohort_capacity_stall: Option<LodSplitCohortCapacityStall>,
    started_pages: Vec<LodPageId>,
    completed_pages: Vec<LodPageId>,
    /// Pages whose encoded payload reached preprocessing but failed checksum,
    /// codec, or support-bound validation during this update. Unlike
    /// `failed_pages`, these are emitted on the first rejection rather than
    /// only after retry exhaustion.
    preprocess_failed_pages: Vec<LodPageId>,
    failed_pages: Vec<LodPageId>,
}

impl LodStreamFrame {
    pub fn view(&self) -> LodRuntimeViewId {
        self.view
    }

    pub fn frontier(&self) -> &LodFrontier<LodNodeId> {
        &self.frontier
    }

    pub fn physical_ranges(&self) -> &[LodPhysicalRange] {
        &self.physical_ranges
    }

    /// Selected ancestors that substitute for missing requested descendants in
    /// this exact view update. Package staging unions these already-derived
    /// nodes across cameras instead of retraversing the hierarchy.
    pub(crate) fn ancestor_fallback_nodes(&self) -> impl Iterator<Item = LodNodeId> + '_ {
        self.ancestor_fallback_nodes.iter().copied()
    }

    /// True when selection and streaming demand used this view's frozen camera
    /// snapshot. Residency and physical ranges are still current for this frame.
    pub fn selection_view_frozen(&self) -> bool {
        self.selection_view_frozen
    }

    /// Whether the logical selector itself has reached a fixed point.
    pub fn selection_stable(&self) -> bool {
        self.selection_stable
    }

    /// True only when this frame changed the complete cut through the bounded
    /// temporal cohort planner. Package publication may use this as a narrow
    /// exception to the ordinary selector fixed-point gate.
    pub fn temporal_transition_applied(&self) -> bool {
        self.temporal_transition
            .as_ref()
            .is_some_and(|transition| transition.changed_gaussians != 0)
    }

    pub fn temporal_transition(&self) -> Option<&LodTemporalTransition> {
        self.temporal_transition.as_ref()
    }

    pub fn has_complete_resident_cut(&self) -> bool {
        self.complete_resident_cut
    }

    pub fn cache_stats(&self) -> PageCacheStats {
        self.cache
    }

    pub fn queued_requests(&self) -> u32 {
        self.queued_requests
    }

    pub fn in_flight_requests(&self) -> u32 {
        self.in_flight_requests
    }

    pub fn preprocess_stats(&self) -> LodPagePreprocessStats {
        self.preprocess
    }

    pub fn capacity_blocked_requests(&self) -> u32 {
        self.capacity_blocked_requests
    }

    pub fn split_cohort_capacity_stall(&self) -> Option<LodSplitCohortCapacityStall> {
        self.split_cohort_capacity_stall
    }

    pub fn started_pages(&self) -> &[LodPageId] {
        &self.started_pages
    }

    pub fn completed_pages(&self) -> &[LodPageId] {
        &self.completed_pages
    }

    pub fn preprocess_failed_pages(&self) -> &[LodPageId] {
        &self.preprocess_failed_pages
    }

    pub fn failed_pages(&self) -> &[LodPageId] {
        &self.failed_pages
    }

    /// Exact count represented by the physical ranges emitted this update.
    pub fn candidate_count(&self) -> u64 {
        self.physical_ranges
            .iter()
            .map(|range| u64::from(range.count))
            .sum()
    }

    /// Validates that this update contains a complete resident cut with exact,
    /// non-overlapping physical ranges and a representable bounded count, then
    /// freezes it without expanding a candidate-sized index vector.
    pub fn candidate_frontier(&self, limit: u32) -> Result<LodCandidateFrontier, LodRuntimeError> {
        if !self.complete_resident_cut {
            return Err(LodRuntimeError::NoResidentFrontier);
        }
        build_candidate_frontier(
            self.view,
            &self.physical_ranges,
            &self.ancestor_fallback_nodes,
            self.frontier.status,
            LodCandidateFrontierBuildOptions {
                selection_view_frozen: self.selection_view_frozen,
                coverage_guard: false,
                temporal_transition: self.temporal_transition.clone(),
                limit,
            },
        )
    }
}

/// Exact resident-capacity pressure for one selector-atomic child cohort.
///
/// The required footprint is the deduplicated union of every long-lived cache
/// lease and every physical page needed to replace `parent`. Unpinned resident
/// pages are deliberately excluded because the cache may evict them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodSplitCohortCapacityStall {
    pub view: LodRuntimeViewId,
    pub parent: LodNodeId,
    pub required_pages: u64,
    pub limit_pages: u64,
    pub required_decoded_bytes: u64,
    pub limit_decoded_bytes: u64,
    pub required_gaussians: u64,
    pub limit_gaussians: u64,
}

struct LodCandidateFrontierBuildOptions {
    selection_view_frozen: bool,
    coverage_guard: bool,
    temporal_transition: Option<LodTemporalTransition>,
    limit: u32,
}

fn build_candidate_frontier(
    view: LodRuntimeViewId,
    physical_ranges: &[LodPhysicalRange],
    ancestor_fallback_nodes: &BTreeSet<LodNodeId>,
    quality_status: LodEffectiveStatus,
    options: LodCandidateFrontierBuildOptions,
) -> Result<LodCandidateFrontier, LodRuntimeError> {
    let count = physical_ranges
        .iter()
        .map(|range| u64::from(range.count))
        .sum::<u64>();
    if count != quality_status.active_gaussians {
        return Err(LodRuntimeError::CandidateCountMismatch {
            frontier: quality_status.active_gaussians,
            physical: count,
        });
    }
    if count > u64::from(options.limit) {
        return Err(LodRuntimeError::CandidateExpansionLimit {
            count,
            limit: options.limit,
        });
    }
    let mut intervals = physical_ranges
        .iter()
        .map(|range| {
            range
                .end()
                .map(|end| (range.physical_start, end))
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    intervals.sort_unstable();
    for pair in intervals.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(LodRuntimeError::OverlappingPhysicalRanges {
                previous_end: pair[0].1,
                next_start: pair[1].0,
            });
        }
    }

    let represented_nodes = physical_ranges
        .iter()
        .map(|range| range.node)
        .collect::<BTreeSet<_>>();
    debug_assert!(ancestor_fallback_nodes.is_subset(&represented_nodes));

    Ok(LodCandidateFrontier {
        view,
        physical_ranges: physical_ranges.to_vec(),
        ancestor_fallback_nodes: ancestor_fallback_nodes.clone(),
        candidate_count: count as u32,
        quality_status,
        selection_view_frozen: options.selection_view_frozen,
        coverage_guard: options.coverage_guard,
        temporal_transition: options.temporal_transition,
    })
}

fn selected_ancestor_fallback_nodes(
    hierarchy: &CompiledManifestLodHierarchy,
    frontier: &LodFrontier<LodNodeId>,
) -> BTreeSet<LodNodeId> {
    let selected = frontier.nodes.iter().copied().collect::<BTreeSet<_>>();
    let mut fallback_nodes = BTreeSet::new();
    for &requested in &frontier.requested_nodes {
        let mut cursor = hierarchy.node(requested).and_then(|node| node.parent);
        while let Some(ancestor) = cursor {
            if selected.contains(&ancestor) {
                fallback_nodes.insert(ancestor);
                break;
            }
            cursor = hierarchy.node(ancestor).and_then(|node| node.parent);
        }
    }
    fallback_nodes
}

fn all_resident_coverage_guard_fallback_nodes(
    hierarchy: &CompiledManifestLodHierarchy,
    coverage_guard_nodes: &[LodNodeId],
    all_resident_nodes: &[LodNodeId],
) -> BTreeSet<LodNodeId> {
    let coverage_guard_nodes = coverage_guard_nodes
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut fallback_nodes = BTreeSet::new();
    for &selected in all_resident_nodes {
        if coverage_guard_nodes.contains(&selected) {
            continue;
        }
        let mut cursor = hierarchy.node(selected).and_then(|node| node.parent);
        while let Some(ancestor) = cursor {
            if coverage_guard_nodes.contains(&ancestor) {
                fallback_nodes.insert(ancestor);
                break;
            }
            cursor = hierarchy.node(ancestor).and_then(|node| node.parent);
        }
    }
    fallback_nodes
}

#[derive(Clone, Debug)]
struct InFlight<Ticket> {
    ticket: Ticket,
    request: PageRequest,
}

#[derive(Clone, Debug, Default)]
struct LodRuntimeViewState {
    previous_frontier: Vec<LodNodeId>,
    /// Policy that produced `previous_frontier`. Hysteresis history is valid
    /// only while this remains exactly unchanged.
    previous_lod_policy: Option<LodHysteresisPolicy>,
    /// Consecutive canonical demand for each independent hierarchy boundary.
    /// This is transient confirmation, not selector hysteresis: once a view is
    /// stationary the emitted cut always converges to the stateless target.
    temporal_demands: HashMap<LodTemporalSubstitutionKey<LodNodeId>, LodTemporalDemand>,
    /// One immutable expanded ABI16 batch retained while the same logical
    /// substitution and atlas generations remain pending. Frontier/candidate
    /// clones share this Arc; camera updates never recopy its direct records.
    temporal_morph_cache: Option<LodTemporalMorphCache>,
    /// Parent edges whose complete immediate-child cohort was requested while
    /// the parent remained the visible resident endpoint. The first immutable
    /// edge published after that cohort becomes resident carries the exceptional
    /// slew provenance; ordinary all-resident camera motion does not.
    late_view_blend_edges: BTreeSet<LodNodeId>,
    /// Immediate-child cohorts prepared before the selector crosses their
    /// parent boundary. This demand is deliberately absent from
    /// `requested_pages` and quality degradation; it uses Prefetch priority and
    /// its own cache pins/retention window.
    predictive_view_blend_nodes: BTreeMap<LodNodeId, Vec<LodNodeId>>,
    pinned_predictive_pages: BTreeSet<LodPageId>,
    /// Camera snapshot used only for selection and page-demand priority. It does
    /// not contain or freeze residency/physical availability.
    frozen_selection_view: Option<LodView>,
    selected_frontier: BTreeSet<LodPageId>,
    pinned_frontier: BTreeSet<LodPageId>,
    requested_pages: BTreeSet<LodPageId>,
    requested_pages_frame: LodRuntimeFrameId,
    /// Physical pages admitted to transport/preprocessing this frame. Logical
    /// `requested_pages` remains observable even while cohort admission gates
    /// work to a smaller atomic substitution.
    admitted_pages: BTreeSet<LodPageId>,
    admitted_pages_frame: LodRuntimeFrameId,
    /// Selector-atomic parent substitutions observed for this view. Admission
    /// is delayed until `finish_frame`, when every participating view has had a
    /// chance to publish its candidates and round-robin choice is deterministic.
    split_cohort_candidates: Vec<LodSplitCohortPlan>,
    split_cohort_candidates_frame: LodRuntimeFrameId,
    split_cohort_pressure: bool,
    split_cohort_pressure_frame: LodRuntimeFrameId,
    /// Exact selector fixed point for a stationary view. This is populated only
    /// after a full traversal reproduces `previous_frontier`, so transient
    /// boundary confirmation has already converged for the cached key.
    stable_selection: Option<StableSelectionCache>,
    /// Exact immutable-hierarchy selector result used to classify a resident
    /// coverage guard. Unlike `stable_selection`, this ignores physical
    /// residency but includes the live selection policy, view, and hysteresis
    /// input so ActiveBudget/TraversalBudget decisions remain authoritative.
    all_resident_selection: Option<AllResidentSelectionCache>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LodTemporalDemand {
    consecutive_frames: u8,
    last_seen_frame: LodRuntimeFrameId,
}

#[derive(Clone, Debug)]
struct LodTemporalMorphCache {
    identity: LodTemporalMorphIdentity,
    batch: Arc<LodTemporalMorphBatch>,
}

#[derive(Clone, Debug)]
struct LodRuntimeSelection {
    frontier: LodFrontier<LodNodeId>,
    temporal_transition: Option<LodTemporalTransition>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LodSplitCohortPlan {
    view: LodRuntimeViewId,
    parent: LodNodeId,
    pages: BTreeSet<LodPageId>,
}

impl LodSplitCohortPlan {
    fn key(&self) -> (LodRuntimeViewId, LodNodeId) {
        (self.view, self.parent)
    }
}

/// One bounded transaction which makes a selector-atomic child substitution
/// physically atomic under cache pressure. `pinned_pages` owns exactly one
/// cache pin per entry, independently of view, package, and frame leases.
#[derive(Clone, Debug)]
struct LodActiveSplitCohort {
    plan: LodSplitCohortPlan,
    pinned_pages: BTreeSet<LodPageId>,
    owner_updated_frame: LodRuntimeFrameId,
    /// Owner demand admitted independently of this cohort in the current
    /// frame. Restoring this exact set on preemption preserves overlapping
    /// root/direct work without retaining cohort-only pages.
    owner_base_admitted_pages: BTreeSet<LodPageId>,
    owner_base_admitted_pages_frame: LodRuntimeFrameId,
}

/// Runtime-owned coarse cut whose resource footprint is bounded by the same
/// resident and active limits as every camera cut. Its pins are deliberately
/// separate from `LodRuntimeViewState`: removing a camera must never make the
/// ordinary emergency coverage cut evictable. Package bootstraps are the sole
/// exception: their startup reserve is released after the first visible cut
/// acquires an independent package lease.
#[derive(Clone, Debug)]
struct LodRuntimeCoverageGuard {
    nodes: Vec<LodNodeId>,
    pages: BTreeSet<LodPageId>,
    pinned_pages: BTreeSet<LodPageId>,
    active_gaussians: u64,
    /// True only when the package-specific cold budget admitted this complete
    /// antichain. The ordinary emergency guard remains available independently
    /// and must not become a presentation signal for legacy coarse hierarchies.
    package_bootstrap: bool,
    /// Set exactly once after package orchestration publishes its first cut.
    /// Released bootstraps no longer own pins, requests, or demand priority.
    package_bootstrap_released: bool,
}

struct LodRuntimeCoverageGuardFootprint {
    pages: BTreeSet<LodPageId>,
    active_gaussians: u64,
    resident_bytes: u64,
    resident_gaussians: u64,
    encoded_bytes: Option<u64>,
}

/// Package-only budget for one globally complete cold-start antichain.
///
/// This is deliberately separate from the ordinary resident budgets: a page
/// can be valid and bounded for streaming without belonging in the ultra-fast
/// first presentation transaction.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LodPackageBootstrapBudget {
    pub max_pages: u32,
    pub max_active_gaussians: u64,
    pub max_encoded_bytes: u64,
    pub max_decoded_bytes: u64,
    pub max_gpu_bytes: u64,
    pub gpu_bytes_per_slot: u64,
}

/// Immutable package selection against an all-resident hierarchy. Physical
/// ranges are resolved only after every unique target page is decoded and
/// retained, so direct package publication never depends on navigation rungs.
#[derive(Clone, Debug)]
pub(crate) struct LodPackageTargetPlan {
    pages: BTreeSet<LodPageId>,
    views: Vec<LodPackageTargetView>,
}

#[derive(Clone, Debug)]
struct LodPackageTargetView {
    view: LodRuntimeViewId,
    frontier: LodFrontier<LodNodeId>,
    selection_view_frozen: bool,
}

impl LodPackageTargetPlan {
    pub(crate) fn pages(&self) -> &BTreeSet<LodPageId> {
        &self.pages
    }
}

impl LodRuntimeCoverageGuard {
    fn new(
        hierarchy: &CompiledManifestLodHierarchy,
        settings: &GaussianLodSettings,
    ) -> Result<Self, LodRuntimeError> {
        Self::new_with_package_bootstrap(hierarchy, settings, None)
    }

    fn new_with_package_bootstrap(
        hierarchy: &CompiledManifestLodHierarchy,
        settings: &GaussianLodSettings,
        package_budget: Option<LodPackageBootstrapBudget>,
    ) -> Result<Self, LodRuntimeError> {
        let mut nodes = hierarchy.roots().to_vec();
        nodes.sort_unstable();
        let mut footprint = Self::footprint(hierarchy, &nodes)?;
        let resident_pages = u64::try_from(footprint.pages.len()).unwrap_or(u64::MAX);
        if resident_pages > u64::from(settings.budgets.max_resident_pages) {
            return Err(LodRuntimeError::CoverageGuardPagesExceedLimit {
                actual: resident_pages,
                limit: u64::from(settings.budgets.max_resident_pages),
            });
        }
        if footprint.resident_bytes > settings.budgets.max_resident_bytes {
            return Err(LodRuntimeError::CoverageGuardBytesExceedLimit {
                actual: footprint.resident_bytes,
                limit: settings.budgets.max_resident_bytes,
            });
        }
        if footprint.resident_gaussians > settings.budgets.max_resident_gaussians {
            return Err(LodRuntimeError::CoverageGuardGaussiansExceedLimit {
                actual: footprint.resident_gaussians,
                limit: settings.budgets.max_resident_gaussians,
            });
        }
        let active_limit = settings
            .budgets
            .max_active_gaussians
            .min(u64::from(u32::MAX));
        if footprint.active_gaussians > active_limit {
            return Err(LodRuntimeError::CoverageGuardActiveGaussiansExceedLimit {
                actual: footprint.active_gaussians,
                limit: active_limit,
            });
        }

        let package_eligible = if let Some(budget) = package_budget {
            hierarchy
                .manifest()
                .build
                .has_bounded_refinement_amplification()
                && hierarchy.manifest().build.reducer_version == MOMENT_MERGE_VERSION
                && Self::package_bootstrap_footprint_fits(&footprint, settings, budget)
                && Self::has_transition_headroom(hierarchy, &nodes, &footprint, settings)?
        } else {
            false
        };
        if package_eligible {
            let budget = package_budget.expect("checked package bootstrap budget");
            if settings.quality_endpoint() != LodQualityEndpoint::Coarsest {
                while let Some(next_nodes) = Self::next_complete_level(hierarchy, &nodes) {
                    let next = Self::footprint(hierarchy, &next_nodes)?;
                    if !Self::package_bootstrap_footprint_fits(&next, settings, budget)
                        || !Self::has_transition_headroom(hierarchy, &next_nodes, &next, settings)?
                    {
                        break;
                    }
                    nodes = next_nodes;
                    footprint = next;
                }
            }

            return Ok(Self {
                nodes,
                pages: footprint.pages,
                pinned_pages: BTreeSet::new(),
                active_gaussians: footprint.active_gaussians,
                package_bootstrap: true,
                package_bootstrap_released: false,
            });
        }

        // A decoded atlas slot has fixed physical size even when its page uses
        // only a small prefix. Spend no more slots on a promoted guard than on
        // the root cut, but admit it only when the complete guard and root cut
        // fit together. The guard stays pinned while the roots stream, so an
        // individually valid guard that lacks this transition headroom would
        // permanently prevent a complete root forest from becoming resident.
        let root_page_count = footprint.pages.len();
        let root_pages = footprint.pages.clone();
        if u64::from(settings.budgets.max_resident_pages) > resident_pages {
            while let Some(next_nodes) = Self::next_complete_level(hierarchy, &nodes) {
                let next = Self::footprint(hierarchy, &next_nodes)?;
                let mut transition_pages = root_pages.clone();
                transition_pages.extend(next.pages.iter().copied());
                let (transition_bytes, transition_gaussians, _) =
                    Self::page_footprint(hierarchy, &transition_pages)?;
                if next.pages.len() > root_page_count
                    || transition_pages.len() > settings.budgets.max_resident_pages as usize
                    || transition_bytes > settings.budgets.max_resident_bytes
                    || transition_gaussians > settings.budgets.max_resident_gaussians
                    || next.active_gaussians > active_limit
                {
                    break;
                }

                // Replacing every expandable node at once preserves a complete,
                // non-overlapping global cut. Never accept a partial forest level.
                nodes = next_nodes;
                footprint = next;
            }
        }

        Ok(Self {
            nodes,
            pages: footprint.pages,
            pinned_pages: BTreeSet::new(),
            active_gaussians: footprint.active_gaussians,
            package_bootstrap: false,
            package_bootstrap_released: false,
        })
    }

    fn is_active(&self) -> bool {
        !self.package_bootstrap_released
    }

    fn contains_active_page(&self, page: LodPageId) -> bool {
        self.is_active() && self.pages.contains(&page)
    }

    fn next_complete_level(
        hierarchy: &CompiledManifestLodHierarchy,
        nodes: &[LodNodeId],
    ) -> Option<Vec<LodNodeId>> {
        let mut next = Vec::new();
        let mut expanded = false;
        for &node in nodes {
            let children = hierarchy.children(node);
            if children.is_empty() {
                next.push(node);
            } else {
                next.extend_from_slice(children);
                expanded = true;
            }
        }
        if !expanded {
            return None;
        }
        next.sort_unstable();
        Some(next)
    }

    fn package_bootstrap_footprint_fits(
        footprint: &LodRuntimeCoverageGuardFootprint,
        settings: &GaussianLodSettings,
        budget: LodPackageBootstrapBudget,
    ) -> bool {
        let page_count = u64::try_from(footprint.pages.len()).unwrap_or(u64::MAX);
        let gpu_bytes = page_count.saturating_mul(budget.gpu_bytes_per_slot);
        page_count <= u64::from(budget.max_pages)
            && page_count <= u64::from(settings.budgets.max_resident_pages)
            && footprint.active_gaussians <= budget.max_active_gaussians
            && footprint.active_gaussians <= settings.budgets.max_active_gaussians
            && footprint.resident_bytes <= budget.max_decoded_bytes
            && footprint.resident_bytes <= settings.budgets.max_resident_bytes
            && footprint.resident_gaussians <= settings.budgets.max_resident_gaussians
            && footprint
                .encoded_bytes
                .is_some_and(|bytes| bytes <= budget.max_encoded_bytes)
            && gpu_bytes <= budget.max_gpu_bytes
    }

    /// A bootstrap startup reserve must leave enough configured resident
    /// capacity for ordinary traversal's root footprint and every one-node
    /// atomic child substitution. This is the smallest useful transition
    /// proof: the selector can enter and refine guard regions one at a time
    /// before the first visible cut transfers protection to its package lease.
    fn has_transition_headroom(
        hierarchy: &CompiledManifestLodHierarchy,
        nodes: &[LodNodeId],
        footprint: &LodRuntimeCoverageGuardFootprint,
        settings: &GaussianLodSettings,
    ) -> Result<bool, LodRuntimeError> {
        let root = Self::footprint(hierarchy, hierarchy.roots())?;
        let mut base_pages = footprint.pages.clone();
        base_pages.extend(root.pages);
        let (base_bytes, base_gaussians, _) = Self::page_footprint(hierarchy, &base_pages)?;
        if base_pages.len() > settings.budgets.max_resident_pages as usize
            || base_bytes > settings.budgets.max_resident_bytes
            || base_gaussians > settings.budgets.max_resident_gaussians
        {
            return Ok(false);
        }
        for &node in nodes {
            let children = hierarchy.children(node);
            if children.is_empty() {
                continue;
            }
            let mut transition_pages = base_pages.clone();
            for &child in children {
                let representation = hierarchy
                    .representation(child)
                    .ok_or(LodRuntimeError::MissingNode(child))?;
                transition_pages.insert(representation.page);
            }
            let (resident_bytes, resident_gaussians, _) =
                Self::page_footprint(hierarchy, &transition_pages)?;
            if transition_pages.len() > settings.budgets.max_resident_pages as usize
                || resident_bytes > settings.budgets.max_resident_bytes
                || resident_gaussians > settings.budgets.max_resident_gaussians
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn footprint(
        hierarchy: &CompiledManifestLodHierarchy,
        nodes: &[LodNodeId],
    ) -> Result<LodRuntimeCoverageGuardFootprint, LodRuntimeError> {
        let mut pages = BTreeSet::new();
        let mut active_gaussians = 0_u64;
        for &node in nodes {
            let representation = hierarchy
                .representation(node)
                .ok_or(LodRuntimeError::MissingNode(node))?;
            pages.insert(representation.page);
            active_gaussians = active_gaussians
                .checked_add(u64::from(representation.count))
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
        }

        let (resident_bytes, resident_gaussians, encoded_bytes) =
            Self::page_footprint(hierarchy, &pages)?;

        Ok(LodRuntimeCoverageGuardFootprint {
            pages,
            active_gaussians,
            resident_bytes,
            resident_gaussians,
            encoded_bytes,
        })
    }

    fn page_footprint(
        hierarchy: &CompiledManifestLodHierarchy,
        pages: &BTreeSet<LodPageId>,
    ) -> Result<(u64, u64, Option<u64>), LodRuntimeError> {
        let mut resident_bytes = 0_u64;
        let mut resident_gaussians = 0_u64;
        let mut encoded_bytes = Some(0_u64);
        for &page in pages {
            let descriptor = hierarchy
                .page_descriptor(page)
                .ok_or(LodRuntimeError::MissingPageDescriptor(page))?;
            resident_bytes = resident_bytes
                .checked_add(descriptor.decoded_len)
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
            resident_gaussians = resident_gaussians
                .checked_add(u64::from(descriptor.gaussian_count))
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
            encoded_bytes = encoded_bytes.and_then(|bytes| {
                descriptor
                    .storage
                    .as_ref()
                    .and_then(|storage| bytes.checked_add(storage.encoded_len))
            });
        }
        Ok((resident_bytes, resident_gaussians, encoded_bytes))
    }
}

impl LodRuntimeViewState {
    fn selection_view(&mut self, current: LodView, mode: LodSelectionMode) -> LodView {
        match mode {
            LodSelectionMode::Dynamic => {
                self.frozen_selection_view = None;
                current
            }
            LodSelectionMode::Frozen => *self.frozen_selection_view.get_or_insert(current),
        }
    }

    #[cfg(test)]
    fn hysteresis_frontier(&self, lod_settings: &GaussianLodSettings) -> &[LodNodeId] {
        if self.previous_lod_policy == Some(LodHysteresisPolicy::from(lod_settings)) {
            &self.previous_frontier
        } else {
            &[]
        }
    }

    fn rendered_frontier(&self) -> &[LodNodeId] {
        &self.previous_frontier
    }

    fn clear_temporal_state(&mut self) {
        self.temporal_demands.clear();
    }

    fn cached_stable_selection(&self, key: StableSelectionKey) -> Option<&LodFrontier<LodNodeId>> {
        let cached = self.stable_selection.as_ref()?;
        (self.temporal_demands.is_empty()
            && self.previous_lod_policy == Some(key.policy)
            && self.previous_frontier.as_slice() == cached.frontier.nodes.as_slice()
            && cached.key == key)
            .then_some(&cached.frontier)
    }

    fn cached_stable_payload(&self, key: StableSelectionKey) -> Option<&StableFramePayload> {
        self.cached_stable_selection(key)?;
        self.stable_selection.as_ref()?.payload.as_ref()
    }

    fn promote_stable_payload(
        &mut self,
        key: StableSelectionKey,
        frontier: &LodFrontier<LodNodeId>,
        physical_ranges: &[LodPhysicalRange],
        complete_resident_cut: bool,
    ) {
        if !self.temporal_demands.is_empty()
            || self.previous_lod_policy != Some(key.policy)
            || self.previous_frontier.as_slice() != frontier.nodes.as_slice()
        {
            return;
        }
        let Some(cached) = self.stable_selection.as_mut() else {
            return;
        };
        if cached.frontier.nodes.as_slice() == frontier.nodes.as_slice()
            && cached.frontier.requested_nodes.as_slice() == frontier.requested_nodes.as_slice()
            && cached.key == key
        {
            // `record_frame_demand` canonicalizes requested_pages from logical
            // nodes to physical pages after selection. Keep that final status
            // in the cached output as well.
            cached.frontier = frontier.clone();
            cached.payload = Some(StableFramePayload {
                physical_ranges: physical_ranges.to_vec(),
                complete_resident_cut,
            });
        }
    }

    fn commit_frontier(&mut self, frontier: &[LodNodeId], lod_settings: &GaussianLodSettings) {
        // Selection advances before the cross-world render commit. Late-page
        // provenance therefore cannot be retired here: a pending candidate may
        // still be cancelled or fail before its inherited endpoint is ever
        // drawable. `acknowledge_rendered_frontier` owns that retirement after
        // ACTIVE publication.
        self.previous_frontier.clear();
        self.previous_frontier.extend_from_slice(frontier);
        self.previous_lod_policy = Some(LodHysteresisPolicy::from(lod_settings));
    }
}

/// A hierarchy boundary must remain on the same side of the canonical
/// threshold for this many distinct application frames before it may change.
/// This suppresses threshold chatter while preserving a history-independent
/// settled cut.
const LOD_TEMPORAL_CONFIRMATION_FRAMES: u8 = 2;
/// Candidate-local work remains bounded independently of virtual source size.
const LOD_TEMPORAL_MAX_SUBSTITUTIONS_PER_FRAME: usize = 256;
const LOD_TEMPORAL_MAX_CHANGED_GAUSSIANS_PER_FRAME: u64 = 256 * 1024;
const LOD_TEMPORAL_ACTIVE_FRACTION: u64 = 24;
const LOD_VIEW_BLEND_PREFETCH_PARENT_PRESSURE: f32 = 0.75;
const LOD_VIEW_BLEND_RELEASE_PARENT_PRESSURE: f32 = 2.0 / 3.0;
const LOD_VIEW_BLEND_RELEASE_CHILD_PRESSURE: f32 = 1.5;

fn temporal_changed_gaussian_budget(current_active_gaussians: u64) -> u64 {
    current_active_gaussians
        .div_ceil(LOD_TEMPORAL_ACTIVE_FRACTION)
        .clamp(1, LOD_TEMPORAL_MAX_CHANGED_GAUSSIANS_PER_FRAME)
}

fn view_blend_transition(
    substitutions: Vec<LodTemporalSubstitution<LodNodeId>>,
    initial_weight_bits: Vec<u32>,
    changed_gaussians: u64,
    atomic_budget_overshoot: u64,
) -> Option<LodTemporalTransition> {
    debug_assert_eq!(substitutions.len(), initial_weight_bits.len());
    (!substitutions.is_empty()).then_some(LodTemporalTransition {
        substitutions,
        initial_weight_bits,
        changed_gaussians,
        atomic_budget_overshoot,
        mode: LodTemporalTransitionMode::BoundedHardCohort,
        morph: None,
    })
}

fn merge_disjoint_view_blend_substitutions(
    stable: Vec<LodTemporalSubstitution<LodNodeId>>,
    applied: &[LodTemporalSubstitution<LodNodeId>],
) -> Vec<LodTemporalSubstitution<LodNodeId>> {
    let applied_endpoint_nodes = applied
        .iter()
        .flat_map(|substitution| {
            substitution
                .previous_nodes
                .iter()
                .chain(&substitution.next_nodes)
                .copied()
        })
        .collect::<BTreeSet<_>>();
    let mut by_parent = stable
        .into_iter()
        .filter(|substitution| {
            substitution
                .previous_nodes
                .iter()
                .chain(&substitution.next_nodes)
                .all(|node| !applied_endpoint_nodes.contains(node))
        })
        .map(|substitution| (substitution.key.parent, substitution))
        .collect::<BTreeMap<_, _>>();
    // A newly applied topology edge owns its retained endpoint. A stable edge
    // which touches that parent/child footprint must wait for the applied edge
    // to retire categorically: the render ABI carries one independent weight
    // per record and deliberately does not compose nested products.
    for substitution in applied {
        by_parent.insert(substitution.key.parent, substitution.clone());
    }
    by_parent.into_values().collect()
}

fn confirmed_temporal_keys(
    demands: &mut HashMap<LodTemporalSubstitutionKey<LodNodeId>, LodTemporalDemand>,
    candidate_keys: &BTreeSet<LodTemporalSubstitutionKey<LodNodeId>>,
    frame: LodRuntimeFrameId,
) -> BTreeSet<LodTemporalSubstitutionKey<LodNodeId>> {
    demands.retain(|key, _| candidate_keys.contains(key));
    for &key in candidate_keys {
        let demand = demands.entry(key).or_default();
        if demand.last_seen_frame != frame {
            demand.consecutive_frames =
                if demand.last_seen_frame.sequence().checked_add(1) == Some(frame.sequence()) {
                    demand.consecutive_frames.saturating_add(1)
                } else {
                    1
                };
            demand.last_seen_frame = frame;
        }
    }
    demands
        .iter()
        .filter_map(|(key, demand)| {
            (demand.consecutive_frames >= LOD_TEMPORAL_CONFIRMATION_FRAMES).then_some(*key)
        })
        .collect()
}

// Historical release-schedule oracle retained only for the benchmark below.
// Production selection uses `confirmed_temporal_keys` plus complete-cut
// cohorts; this code is excluded from every non-test build.
#[cfg(test)]
const LOD_COARSENING_MIN_HOLD_FRAMES: u64 = 2;
#[cfg(test)]
const LOD_COARSENING_STAGGER_BUCKETS: u64 = 3;

#[cfg(test)]
fn coarsening_stagger_bucket(node: LodNodeId) -> u64 {
    let mut value = node.0;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (value ^ (value >> 31)) % LOD_COARSENING_STAGGER_BUCKETS
}

#[cfg(test)]
fn coarsening_hold_frames(node: LodNodeId, depth_from_previous: u32) -> u64 {
    LOD_COARSENING_MIN_HOLD_FRAMES
        .saturating_add(coarsening_stagger_bucket(node))
        .saturating_add(
            u64::from(depth_from_previous.saturating_sub(1))
                .saturating_mul(LOD_COARSENING_STAGGER_BUCKETS),
        )
}

#[cfg(test)]
fn coarsening_refinement_depths(
    hierarchy: &CompiledManifestLodHierarchy,
    previous_frontier: &[LodNodeId],
    desired_frontier: &[LodNodeId],
) -> HashMap<LodNodeId, u32> {
    let desired = desired_frontier.iter().copied().collect::<HashSet<_>>();
    let mut refinements = HashMap::<LodNodeId, u32>::new();
    let mut path = Vec::new();
    for &previous in previous_frontier {
        if desired.contains(&previous) {
            continue;
        }
        path.clear();
        let mut cursor = previous;
        while let Some(parent) = hierarchy.parent(cursor) {
            path.push(parent);
            if desired.contains(&parent) {
                for (index, node) in path.iter().copied().enumerate() {
                    let depth = u32::try_from(index + 1).unwrap_or(u32::MAX);
                    refinements
                        .entry(node)
                        .and_modify(|current| *current = (*current).max(depth))
                        .or_insert(depth);
                }
                break;
            }
            cursor = parent;
        }
    }
    refinements
}

#[cfg(test)]
fn active_coarsening_holds(
    release_frames: &mut HashMap<LodNodeId, u64>,
    refinements: &HashMap<LodNodeId, u32>,
    frame_sequence: u64,
) -> BTreeSet<LodNodeId> {
    release_frames.retain(|node, _| refinements.contains_key(node));
    for (&node, &depth) in refinements {
        release_frames
            .entry(node)
            .or_insert_with(|| frame_sequence.saturating_add(coarsening_hold_frames(node, depth)));
    }
    release_frames
        .iter()
        .filter_map(|(&node, &release)| (release > frame_sequence).then_some(node))
        .collect()
}

/// Canonical subset of settings that can change hierarchy selection or the
/// interpretation of its previous cut.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LodHysteresisPolicy {
    target: LodQualityTarget,
    hysteresis: f32,
    frustum_culling: bool,
    frustum_margin: f32,
    max_active_gaussians: u64,
    max_traversal_nodes_per_view: u32,
}

#[derive(Clone, Debug)]
struct StableSelectionCache {
    key: StableSelectionKey,
    frontier: LodFrontier<LodNodeId>,
    /// Proven only after pins, physical ranges, and complete-cut validation
    /// succeed for the selector fixed point above.
    payload: Option<StableFramePayload>,
}

#[derive(Clone, Debug)]
struct AllResidentSelectionCache {
    key: AllResidentSelectionKey,
    frontier: LodFrontier<LodNodeId>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AllResidentSelectionKey {
    view: LodView,
    policy: LodHysteresisPolicy,
    selection_view_frozen: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct StableSelectionKey {
    view: LodView,
    policy: LodHysteresisPolicy,
    selection_view_frozen: bool,
    residency_revision: u64,
}

#[derive(Clone, Debug)]
struct StableFramePayload {
    physical_ranges: Vec<LodPhysicalRange>,
    complete_resident_cut: bool,
}

impl From<&GaussianLodSettings> for LodHysteresisPolicy {
    fn from(settings: &GaussianLodSettings) -> Self {
        Self {
            target: settings.quality_target(),
            hysteresis: settings.hysteresis,
            frustum_culling: settings.frustum_culling,
            frustum_margin: settings.frustum_margin,
            max_active_gaussians: settings.budgets.max_active_gaussians,
            max_traversal_nodes_per_view: settings.budgets.max_traversal_nodes_per_view,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LodRuntimeStructuralSettings {
    max_resident_gaussians: u64,
    max_resident_bytes: u64,
    max_resident_pages: u32,
    max_pending_requests: u32,
    max_encoded_page_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct SharedPageNodeRange {
    node: LodNodeId,
    range: LodPageRange,
    bounds: LodBounds,
}

impl LodRuntimeStructuralSettings {
    fn new(lod: &GaussianLodSettings, streaming: &GaussianStreamingSettings) -> Self {
        Self {
            max_resident_gaussians: lod.budgets.max_resident_gaussians,
            max_resident_bytes: lod.budgets.max_resident_bytes,
            max_resident_pages: lod.budgets.max_resident_pages,
            max_pending_requests: lod.budgets.max_pending_requests,
            max_encoded_page_bytes: streaming.effective_max_encoded_page_bytes(),
        }
    }

    fn validate_compatible(self, next: Self) -> Result<(), LodRuntimeError> {
        for (field, matches) in [
            (
                "budgets.max_resident_gaussians",
                self.max_resident_gaussians == next.max_resident_gaussians,
            ),
            (
                "budgets.max_resident_bytes",
                self.max_resident_bytes == next.max_resident_bytes,
            ),
            (
                "budgets.max_resident_pages",
                self.max_resident_pages == next.max_resident_pages,
            ),
            (
                "budgets.max_pending_requests",
                self.max_pending_requests == next.max_pending_requests,
            ),
            (
                "streaming.effective_max_encoded_page_bytes",
                self.max_encoded_page_bytes == next.max_encoded_page_bytes,
            ),
        ] {
            if !matches {
                return Err(LodRuntimeError::StructuralSettingsChanged(field));
            }
        }
        Ok(())
    }
}

/// Long-lived bounded state for one virtual Gaussian cloud.
pub struct LodStreamingRuntime<T: LodPageTransport> {
    hierarchy: CompiledManifestLodHierarchy,
    /// Slice-local validation plans for physical pages shared by logical nodes.
    /// Single-node pages retain the cheaper descriptor-wide preprocessing path.
    shared_page_node_ranges: BTreeMap<LodPageId, Vec<SharedPageNodeRange>>,
    cache: LodPageCache,
    decoded_pages: BTreeMap<LodPageId, PlanarGaussian3dPage>,
    queue: PageRequestQueue,
    transport: T,
    in_flight: BTreeMap<LodPageId, InFlight<T::Ticket>>,
    preprocessor: LodPagePreprocessor,
    preprocess_failures: BTreeMap<LodPageId, LodPagePreprocessError>,
    /// Frame in which a rejected decoded payload queued its bounded retry.
    /// The retry remains in the ordinary request queue, but cannot start until
    /// the next frame. Package transports use that boundary to invalidate a
    /// cached encoded payload before any retry can observe it again.
    preprocess_retry_deferred_frame: BTreeMap<LodPageId, LodRuntimeFrameId>,
    transport_failures: BTreeMap<LodPageId, LodPageTransportFailure>,
    attempts: BTreeMap<LodPageId, u32>,
    terminal_failures: BTreeSet<LodPageId>,
    terminal_requests: BTreeMap<LodPageId, PageRequest>,
    /// Retry-exhausted predictive requests are not visible quality failures.
    /// They remain tombstoned here so the same speculative cohort does not
    /// restart every frame, and are promoted atomically if the selector later
    /// requests that page for the actual target frontier.
    speculative_prefetch_terminal_requests: BTreeMap<LodPageId, PageRequest>,
    capacity_blocked: BTreeMap<LodPageId, PageRequest>,
    views: BTreeMap<LodRuntimeViewId, LodRuntimeViewState>,
    /// Caller-owned pins acquired through `retain_resident_page`. Tracking them
    /// by owner count lets split admission compute the exact long-lived union
    /// without scanning a source-sized manifest or confusing transient frame
    /// holds with retained-cut pressure.
    caller_page_leases: BTreeMap<LodPageId, u32>,
    active_split_cohort: Option<LodActiveSplitCohort>,
    /// Last admitted/attempted key. The next cohort starts strictly after this
    /// key and wraps, providing deterministic owner/view round-robin fairness.
    split_cohort_cursor: Option<(LodRuntimeViewId, LodNodeId)>,
    split_cohort_capacity_stall: Option<LodSplitCohortCapacityStall>,
    coverage_guard: LodRuntimeCoverageGuard,
    atlas_layout: PageAtlasLayout,
    pending_request_capacity: usize,
    structural_settings: LodRuntimeStructuralSettings,
    largest_decoded_page: (LodPageId, u64),
    /// Advances whenever the set of resident pages or any resident physical
    /// slot generation can change. Stable selector results are valid only for
    /// the exact revision at which they were proven.
    residency_revision: u64,
    #[cfg(test)]
    selection_traversals: u64,
    #[cfg(test)]
    all_resident_selection_traversals: u64,
    #[cfg(test)]
    frontier_pin_rebuilds: u64,
    #[cfg(test)]
    physical_range_rebuilds: u64,
    #[cfg(test)]
    stable_payload_hits: u64,
    #[cfg(test)]
    transport_request_starts: u64,
    #[cfg(test)]
    temporal_morph_batch_builds: u64,
    epoch: u64,
    frame_decoded_bytes: u64,
    frame_request_starts: u32,
    /// Short-lived eviction leases for pages reported as completed in the
    /// current application frame. A later camera can publish more pages before
    /// its own frontier pins are synchronized, so completion pages must remain
    /// resident until all cameras have observed the frame.
    frame_completion_holds: BTreeSet<LodPageId>,
    frame_finished: bool,
}

impl<T: LodPageTransport> Drop for LodStreamingRuntime<T> {
    fn drop(&mut self) {
        let in_flight = std::mem::take(&mut self.in_flight);
        for request in in_flight.into_values() {
            self.transport.cancel(&request.ticket);
        }
    }
}

impl<T: LodPageTransport> LodStreamingRuntime<T> {
    pub fn new(
        manifest: GaussianLodManifest,
        transport: T,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
    ) -> Result<Self, LodRuntimeError> {
        Self::validate_creation_settings(lod_settings, streaming_settings)?;
        let hierarchy = CompiledManifestLodHierarchy::new(manifest)
            .map_err(LodRuntimeError::InvalidManifest)?;
        Self::from_compiled_hierarchy(hierarchy, transport, lod_settings, streaming_settings, None)
    }

    /// Package construction variant which may promote one tightly bounded,
    /// globally complete progressive antichain as a cold-start presentation.
    pub(crate) fn from_validated_shared_manifest_with_package_bootstrap(
        manifest: Arc<GaussianLodManifest>,
        transport: T,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
        bootstrap: LodPackageBootstrapBudget,
    ) -> Result<Self, LodRuntimeError> {
        Self::validate_creation_settings(lod_settings, streaming_settings)?;
        let hierarchy = CompiledManifestLodHierarchy::from_validated_shared_manifest(manifest);
        Self::from_compiled_hierarchy(
            hierarchy,
            transport,
            lod_settings,
            streaming_settings,
            Some(bootstrap),
        )
    }

    fn validate_creation_settings(
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
    ) -> Result<(), LodRuntimeError> {
        lod_settings
            .validate()
            .map_err(|error| LodRuntimeError::InvalidSettings(error.to_string()))?;
        streaming_settings
            .validate()
            .map_err(|error| LodRuntimeError::InvalidSettings(error.to_string()))?;
        Ok(())
    }

    fn from_compiled_hierarchy(
        hierarchy: CompiledManifestLodHierarchy,
        transport: T,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
        package_bootstrap: Option<LodPackageBootstrapBudget>,
    ) -> Result<Self, LodRuntimeError> {
        let shared_page_node_ranges = if hierarchy.manifest().header.required_features
            & LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES
            != 0
        {
            let mut ranges_by_page = BTreeMap::<_, Vec<_>>::new();
            for node in &hierarchy.manifest().nodes {
                ranges_by_page
                    .entry(node.representation.page)
                    .or_default()
                    .push(SharedPageNodeRange {
                        node: node.id,
                        range: node.representation,
                        bounds: node.bounds,
                    });
            }
            ranges_by_page.retain(|_, ranges| {
                ranges.sort_unstable_by_key(|entry| entry.range.offset);
                ranges.len() > 1
            });
            ranges_by_page
        } else {
            BTreeMap::new()
        };
        let max_decoded_page_bytes = lod_settings
            .budgets
            .max_resident_bytes
            .min(lod_settings.budgets.max_upload_bytes_per_frame);
        let max_encoded_page_bytes = streaming_settings.effective_max_encoded_page_bytes();
        if max_encoded_page_bytes < 44 {
            return Err(LodRuntimeError::EncodedPageLimitTooSmall {
                limit: max_encoded_page_bytes,
                minimum: 44,
            });
        }
        for descriptor in &hierarchy.manifest().pages {
            if descriptor.decoded_len > max_decoded_page_bytes {
                return Err(LodRuntimeError::PageDecodedBytesExceedLimit {
                    page: descriptor.id,
                    actual: descriptor.decoded_len,
                    limit: max_decoded_page_bytes,
                });
            }
            if u64::from(descriptor.gaussian_count) > lod_settings.budgets.max_resident_gaussians {
                return Err(LodRuntimeError::PageGaussiansExceedLimit {
                    page: descriptor.id,
                    actual: u64::from(descriptor.gaussian_count),
                    limit: lod_settings.budgets.max_resident_gaussians,
                });
            }
            if let Some(storage) = &descriptor.storage
                && storage.encoded_len > max_encoded_page_bytes
            {
                return Err(LodRuntimeError::PageEncodedBytesExceedLimit {
                    page: descriptor.id,
                    actual: storage.encoded_len,
                    limit: max_encoded_page_bytes,
                });
            }
        }
        let coverage_guard = if let Some(bootstrap) = package_bootstrap {
            LodRuntimeCoverageGuard::new_with_package_bootstrap(
                &hierarchy,
                lod_settings,
                Some(bootstrap),
            )?
        } else {
            LodRuntimeCoverageGuard::new(&hierarchy, lod_settings)?
        };
        let maximum_page_gaussians = hierarchy
            .manifest()
            .pages
            .iter()
            .map(|descriptor| descriptor.gaussian_count)
            .max()
            .ok_or(LodRuntimeError::ManifestHasNoPages)?;
        let largest_decoded_page = hierarchy
            .manifest()
            .pages
            .iter()
            .map(|descriptor| (descriptor.id, descriptor.decoded_len))
            .max_by_key(|(_, decoded_len)| *decoded_len)
            .ok_or(LodRuntimeError::ManifestHasNoPages)?;
        let physical_address_count = u64::from(lod_settings.budgets.max_resident_pages)
            .checked_mul(u64::from(maximum_page_gaussians))
            .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
        if physical_address_count > u64::from(u32::MAX) + 1 {
            return Err(LodRuntimeError::AtlasAddressSpaceOverflow {
                slots: lod_settings.budgets.max_resident_pages,
                stride: maximum_page_gaussians,
            });
        }
        let cache = LodPageCache::new(PageCacheLimits::from(&lod_settings.budgets))
            .map_err(LodRuntimeError::Cache)?;
        let queue_capacity = usize::try_from(lod_settings.budgets.max_pending_requests)
            .map_err(|_| LodRuntimeError::RequestCapacityOverflow)?;
        let queue = PageRequestQueue::new(queue_capacity).map_err(LodRuntimeError::Queue)?;
        let preprocess_capacity = queue_capacity.min(
            usize::try_from(streaming_settings.max_concurrent_requests)
                .map_err(|_| LodRuntimeError::RequestCapacityOverflow)?,
        );
        let preprocess_byte_capacity = max_encoded_page_bytes
            .checked_add(lod_settings.budgets.max_upload_bytes_per_frame)
            .ok_or(LodRuntimeError::PreprocessAdmission(
                LodPagePreprocessAdmissionError::ByteLengthOverflow,
            ))?;
        let preprocessor =
            LodPagePreprocessor::with_byte_capacity(preprocess_capacity, preprocess_byte_capacity)
                .map_err(LodRuntimeError::PreprocessAdmission)?;
        for descriptor in &hierarchy.manifest().pages {
            let encoded_bytes = descriptor
                .storage
                .as_ref()
                .map_or(max_encoded_page_bytes, |storage| storage.encoded_len);
            preprocessor
                .validate_job_bytes(encoded_bytes, descriptor.decoded_len)
                .map_err(LodRuntimeError::PreprocessAdmission)?;
        }
        let mut runtime = Self {
            hierarchy,
            shared_page_node_ranges,
            cache,
            decoded_pages: BTreeMap::new(),
            queue,
            transport,
            in_flight: BTreeMap::new(),
            preprocessor,
            preprocess_failures: BTreeMap::new(),
            preprocess_retry_deferred_frame: BTreeMap::new(),
            transport_failures: BTreeMap::new(),
            attempts: BTreeMap::new(),
            terminal_failures: BTreeSet::new(),
            terminal_requests: BTreeMap::new(),
            speculative_prefetch_terminal_requests: BTreeMap::new(),
            capacity_blocked: BTreeMap::new(),
            views: BTreeMap::new(),
            caller_page_leases: BTreeMap::new(),
            active_split_cohort: None,
            split_cohort_cursor: None,
            split_cohort_capacity_stall: None,
            coverage_guard,
            atlas_layout: PageAtlasLayout::new(maximum_page_gaussians)?,
            pending_request_capacity: queue_capacity,
            structural_settings: LodRuntimeStructuralSettings::new(
                lod_settings,
                streaming_settings,
            ),
            largest_decoded_page,
            residency_revision: 0,
            #[cfg(test)]
            selection_traversals: 0,
            #[cfg(test)]
            all_resident_selection_traversals: 0,
            #[cfg(test)]
            frontier_pin_rebuilds: 0,
            #[cfg(test)]
            physical_range_rebuilds: 0,
            #[cfg(test)]
            stable_payload_hits: 0,
            #[cfg(test)]
            transport_request_starts: 0,
            #[cfg(test)]
            temporal_morph_batch_builds: 0,
            epoch: 0,
            frame_decoded_bytes: 0,
            frame_request_starts: 0,
            frame_completion_holds: BTreeSet::new(),
            frame_finished: true,
        };
        runtime.prime_coverage_guard_requests()?;
        Ok(runtime)
    }

    pub fn hierarchy(&self) -> &CompiledManifestLodHierarchy {
        &self.hierarchy
    }

    pub fn cache(&self) -> &LodPageCache {
        &self.cache
    }

    /// Capacity pressure is observable without failing the update that
    /// produced the retained ancestor cut. Multi-view callers should read this
    /// after `finish_frame`; individual frame values are point-in-time status.
    pub fn split_cohort_capacity_stall(&self) -> Option<LodSplitCohortCapacityStall> {
        self.split_cohort_capacity_stall
    }

    #[cfg(all(test, feature = "sort_radix", not(feature = "buffer_texture")))]
    pub(crate) fn package_bootstrap_pages_for_test(&self) -> Option<&BTreeSet<LodPageId>> {
        (self.coverage_guard.package_bootstrap && self.coverage_guard.is_active())
            .then_some(&self.coverage_guard.pages)
    }

    /// Releases the package-only cold-start reserve after the first visible cut
    /// has acquired its independent page leases. Ordinary runtime coverage
    /// guards are permanent and this method is a no-op for them.
    pub(crate) fn release_package_bootstrap_reserve(&mut self) -> Result<bool, LodRuntimeError> {
        if !self.coverage_guard.package_bootstrap || !self.coverage_guard.is_active() {
            return Ok(false);
        }

        // Remove each successfully released page from the ownership set. This
        // keeps a theoretically interrupted invariant failure retryable without
        // double-unpinning earlier pages.
        let pinned = self
            .coverage_guard
            .pinned_pages
            .iter()
            .copied()
            .collect::<Vec<_>>();
        for page in pinned {
            self.cache
                .unpin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
            self.coverage_guard.pinned_pages.remove(&page);
        }
        self.coverage_guard.package_bootstrap_released = true;
        self.split_cohort_capacity_stall = None;
        self.wake_capacity_blocked();
        Ok(true)
    }

    /// Selects the exact logical target for one view if every page were
    /// resident. The per-view cache is bounded to one immutable-hierarchy
    /// result and invalidates on view, policy, or frozen-state changes. This
    /// target is intentionally stateless so a stationary camera has one
    /// canonical settled cut regardless of its approach direction.
    fn all_resident_target_frontier(
        &mut self,
        view_id: LodRuntimeViewId,
        selection_view: LodView,
        selection_view_frozen: bool,
        lod_settings: &GaussianLodSettings,
    ) -> Result<LodFrontier<LodNodeId>, LodRuntimeError> {
        let key = AllResidentSelectionKey {
            view: selection_view,
            policy: LodHysteresisPolicy::from(lod_settings),
            selection_view_frozen,
        };
        if let Some(cached) = self
            .views
            .get(&view_id)
            .and_then(|state| state.all_resident_selection.as_ref())
            .filter(|cached| cached.key == key)
        {
            return Ok(cached.frontier.clone());
        }

        #[cfg(test)]
        {
            self.all_resident_selection_traversals =
                self.all_resident_selection_traversals.saturating_add(1);
        }
        let target = select_frontier_with_visibility(
            &self.hierarchy,
            &|_| true,
            selection_view,
            lod_settings,
            |_, metrics| {
                !lod_settings.frustum_culling
                    || selection_view.node_is_visible(metrics, lod_settings.frustum_margin)
            },
        )
        .map_err(LodRuntimeError::Selection)?;
        self.views
            .entry(view_id)
            .or_default()
            .all_resident_selection = Some(AllResidentSelectionCache {
            key,
            frontier: target.clone(),
        });
        Ok(target)
    }

    /// Selects the exact logical target each package camera would use if every
    /// page were resident, then returns the unique physical page union.
    /// Frozen cameras reuse the runtime-owned snapshot captured by their first
    /// update; a not-yet-seen frozen camera uses the supplied live view that its
    /// next update will capture.
    pub(crate) fn package_all_resident_target_plan(
        &mut self,
        views: &[(LodRuntimeViewId, LodView)],
        lod_settings: &GaussianLodSettings,
    ) -> Result<LodPackageTargetPlan, LodRuntimeError> {
        lod_settings
            .validate()
            .map_err(|error| LodRuntimeError::InvalidSettings(error.to_string()))?;
        let mut pages = BTreeSet::new();
        let mut target_views = Vec::with_capacity(views.len());
        for &(view_id, live_view) in views {
            let selection_view = self
                .views
                .entry(view_id)
                .or_default()
                .selection_view(live_view, lod_settings.selection_mode);
            selection_view.validate().map_err(|error| match error {
                LodSelectionError::InvalidView(field) => {
                    LodRuntimeError::Selection(LodSelectionError::InvalidView(field))
                }
                _ => unreachable!("LodView::validate only emits InvalidView"),
            })?;
            let target = self.all_resident_target_frontier(
                view_id,
                selection_view,
                lod_settings.selection_mode == LodSelectionMode::Frozen,
                lod_settings,
            )?;
            for &node in &target.nodes {
                pages.insert(
                    self.hierarchy
                        .page(node)
                        .ok_or(LodRuntimeError::MissingNode(node))?,
                );
            }
            target_views.push(LodPackageTargetView {
                view: view_id,
                frontier: target,
                selection_view_frozen: lod_settings.selection_mode == LodSelectionMode::Frozen,
            });
        }
        Ok(LodPackageTargetPlan {
            pages,
            views: target_views,
        })
    }

    /// Resolves a preplanned direct package cut once every target page is
    /// resident and decoded. No intermediate hierarchy representation is read.
    pub(crate) fn package_target_candidates(
        &mut self,
        plan: &LodPackageTargetPlan,
        max_active_gaussians: u32,
    ) -> Result<Option<Vec<(LodRuntimeViewId, LodCandidateFrontier)>>, LodRuntimeError> {
        if plan.pages.iter().any(|page| {
            !self.cache.contains(*page)
                || !self.decoded_pages.contains_key(page)
                || self.terminal_failures.contains(page)
        }) {
            return Ok(None);
        }
        let mut candidates = Vec::with_capacity(plan.views.len());
        for target in &plan.views {
            let ancestor_fallback_nodes =
                selected_ancestor_fallback_nodes(&self.hierarchy, &target.frontier);
            let physical_ranges = self.physical_ranges(&target.frontier)?;
            candidates.push((
                target.view,
                build_candidate_frontier(
                    target.view,
                    &physical_ranges,
                    &ancestor_fallback_nodes,
                    target.frontier.status,
                    LodCandidateFrontierBuildOptions {
                        selection_view_frozen: target.selection_view_frozen,
                        coverage_guard: false,
                        temporal_transition: None,
                        limit: max_active_gaussians,
                    },
                )?,
            ));
        }
        Ok(Some(candidates))
    }

    pub(crate) fn has_active_package_bootstrap(&self) -> bool {
        self.coverage_guard.package_bootstrap && self.coverage_guard.is_active()
    }

    pub(crate) fn active_coverage_guard_pages(&self) -> &BTreeSet<LodPageId> {
        &self.coverage_guard.pages
    }

    /// Adds a package-planned target page set to one view's current-frame
    /// demand and the bounded request queue. This does not bypass transport,
    /// preprocessing, cache, or per-frame start budgets; it only avoids making
    /// exact leaves wait behind every transient navigation rung.
    pub(crate) fn prime_package_pages_in_frame(
        &mut self,
        frame: LodRuntimeFrameId,
        view_id: LodRuntimeViewId,
        pages: &BTreeSet<LodPageId>,
    ) -> Result<(), LodRuntimeError> {
        let current = LodRuntimeFrameId(self.epoch);
        if frame != current || frame.0 == 0 {
            return Err(LodRuntimeError::InvalidFrameToken {
                expected: current,
                actual: frame,
            });
        }
        if self.frame_finished {
            return Err(LodRuntimeError::FrameAlreadyFinished(frame));
        }
        let admitted_pages = pages
            .iter()
            .copied()
            .filter(|page| self.page_reserves_streaming_capacity(*page))
            .collect::<BTreeSet<_>>();
        let view = self.views.entry(view_id).or_default();
        if view.requested_pages_frame != frame {
            view.requested_pages.clear();
            view.requested_pages_frame = frame;
        }
        view.requested_pages.extend(pages.iter().copied());
        if view.admitted_pages_frame != frame {
            view.admitted_pages.clear();
            view.admitted_pages_frame = frame;
        }
        view.admitted_pages.extend(admitted_pages);

        for &page_id in pages {
            if self.cache.contains(page_id)
                || self.in_flight.contains_key(&page_id)
                || self.preprocessor.contains(page_id)
                || self.queue.contains(page_id)
                || self.terminal_failures.contains(&page_id)
                || self.capacity_blocked.contains_key(&page_id)
            {
                continue;
            }
            let descriptor = self
                .hierarchy
                .page_descriptor(page_id)
                .ok_or(LodRuntimeError::MissingPageDescriptor(page_id))?;
            let mut request = PageRequest::new(page_id, PageRequestPriority::visible(u32::MAX));
            request.expected_bytes = descriptor
                .storage
                .as_ref()
                .map(|storage| storage.encoded_len);
            let _ = self.enqueue_pending_request(request);
        }
        Ok(())
    }

    /// Cancels all asynchronous demand owned by the supplied package views
    /// while preserving their live/frozen selection history. Package current-
    /// cut leases remain independent and continue protecting the visible cut.
    pub(crate) fn cancel_package_view_work(
        &mut self,
        view_ids: &[LodRuntimeViewId],
    ) -> Result<(), LodRuntimeError> {
        let mut released_capacity = false;
        if self
            .active_split_cohort
            .as_ref()
            .is_some_and(|cohort| view_ids.contains(&cohort.plan.view))
        {
            released_capacity |= self.release_active_split_cohort()?;
        }
        for &view_id in view_ids {
            if let Some(state) = self.views.get_mut(&view_id) {
                state.requested_pages.clear();
                state.requested_pages_frame = LodRuntimeFrameId::default();
                state.admitted_pages.clear();
                state.admitted_pages_frame = LodRuntimeFrameId::default();
                state.selected_frontier.clear();
                state.stable_selection = None;
            }
            released_capacity |= self.clear_predictive_view_blend_demand(view_id)?;
            released_capacity |= self.synchronize_view_pins(view_id)?;
        }
        if released_capacity {
            self.wake_capacity_blocked();
        }
        let frame = self.begin_frame();
        self.finish_frame(frame)
    }

    #[cfg(all(test, feature = "sort_radix", not(feature = "buffer_texture")))]
    pub(crate) fn pending_request_count_for_test(&self) -> usize {
        self.pending_request_count()
    }

    #[cfg(feature = "testing")]
    pub(crate) fn package_work_counts_for_testing(
        &self,
    ) -> (u32, u32, LodPagePreprocessStats, u32, u32, bool) {
        (
            self.queue.len().try_into().unwrap_or(u32::MAX),
            self.in_flight.len().try_into().unwrap_or(u32::MAX),
            self.preprocessor.stats(),
            self.capacity_blocked.len().try_into().unwrap_or(u32::MAX),
            self.views
                .values()
                .map(|view| view.requested_pages.len())
                .max()
                .unwrap_or(0)
                .try_into()
                .unwrap_or(u32::MAX),
            self.active_split_cohort.is_some(),
        )
    }

    #[cfg(test)]
    pub(crate) fn transport_request_starts_for_test(&self) -> u64 {
        self.transport_request_starts
    }

    #[cfg(all(test, feature = "sort_radix", not(feature = "buffer_texture")))]
    pub(crate) fn contains_view_for_test(&self, view_id: LodRuntimeViewId) -> bool {
        self.views.contains_key(&view_id)
    }

    #[cfg(all(test, feature = "sort_radix", not(feature = "buffer_texture")))]
    pub(crate) fn frozen_selection_view_for_test(
        &self,
        view_id: LodRuntimeViewId,
    ) -> Option<LodView> {
        self.views
            .get(&view_id)
            .and_then(|state| state.frozen_selection_view)
    }

    /// True while an optional adjacent-edge child cohort still needs bounded
    /// transport, preprocessing, or residency publication. Predictive pages do
    /// not count as target `requested_pages` or degradation, but a stationary
    /// package must continue driving the runtime until this work either becomes
    /// resident or reaches its isolated speculative terminal state.
    pub(crate) fn has_predictive_view_blend_work(&self) -> bool {
        self.views.values().any(|state| {
            state
                .predictive_view_blend_nodes
                .values()
                .flatten()
                .filter_map(|node| self.hierarchy.page(*node))
                .any(|page| {
                    !self.cache.contains(page)
                        && !self.terminal_failures.contains(&page)
                        && !self
                            .speculative_prefetch_terminal_requests
                            .contains_key(&page)
                })
        })
    }

    /// Adds one caller-owned eviction lease for a resident page.
    ///
    /// Package orchestration uses this narrow API to keep the last published
    /// render cut resident while a replacement is uploaded over multiple
    /// frames. Runtime view pins remain independently reference counted.
    pub(crate) fn retain_resident_page(
        &mut self,
        page: LodPageId,
    ) -> Result<AtlasSlot, LodRuntimeError> {
        let next_count = self
            .caller_page_leases
            .get(&page)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
        let slot = self
            .cache
            .pin_fallback(page)
            .map_err(LodRuntimeError::Cache)?;
        self.caller_page_leases.insert(page, next_count);
        Ok(slot)
    }

    /// Releases one caller-owned lease acquired by [`Self::retain_resident_page`].
    pub(crate) fn release_resident_page(&mut self, page: LodPageId) -> Result<(), LodRuntimeError> {
        let count = self
            .caller_page_leases
            .get(&page)
            .copied()
            .ok_or(LodRuntimeError::Cache(PageCacheError::NotPinned(page)))?;
        self.cache
            .unpin_fallback(page)
            .map_err(LodRuntimeError::Cache)?;
        if count == 1 {
            self.caller_page_leases.remove(&page);
        } else {
            self.caller_page_leases.insert(page, count - 1);
        }
        self.split_cohort_capacity_stall = None;
        // A caller lease can be the final hold preventing a decoded page from
        // entering the bounded atlas. Resume those requests immediately;
        // their view demand may be unchanged, so no later pin transition is
        // guaranteed to wake them.
        self.wake_capacity_blocked();
        Ok(())
    }

    pub fn decoded_page(&self, page: LodPageId) -> Option<&PlanarGaussian3dPage> {
        self.decoded_pages.get(&page)
    }

    pub fn atlas_layout(&self) -> PageAtlasLayout {
        self.atlas_layout
    }

    /// Produces a camera-bound render candidate for the package's cold-start
    /// global coarse cut. `None` means the reserve was released or at least one
    /// guard page has not completed decode/publication yet.
    ///
    /// The physical cut itself is camera-independent and globally covering,
    /// while its quality/error status is recomputed for `view` and rebased to
    /// the live requested target. This lets the render path admit it without
    /// pretending that coarse fallback satisfies the requested detail level.
    pub(crate) fn package_bootstrap_candidate(
        &mut self,
        view_id: LodRuntimeViewId,
        view: LodView,
        lod_settings: &GaussianLodSettings,
    ) -> Result<Option<LodCandidateFrontier>, LodRuntimeError> {
        if !self.coverage_guard.package_bootstrap
            || !self.coverage_guard.is_active()
            || (lod_settings.quality_endpoint() == LodQualityEndpoint::Coarsest
                && self
                    .coverage_guard
                    .nodes
                    .iter()
                    .any(|node| self.hierarchy.parent(*node).is_some()))
        {
            return Ok(None);
        }
        self.coverage_guard_candidate(view_id, view, lod_settings)
    }

    /// Rebinds the package's already-published bootstrap payload to a live
    /// camera without reacquiring the runtime cold-start reserve. Package cut
    /// leases, checked by the caller, keep these decoded pages resident after
    /// [`Self::release_package_bootstrap_reserve`].
    pub(crate) fn retained_package_bootstrap_candidate(
        &mut self,
        view_id: LodRuntimeViewId,
        view: LodView,
        lod_settings: &GaussianLodSettings,
    ) -> Result<Option<LodCandidateFrontier>, LodRuntimeError> {
        if !self.coverage_guard.package_bootstrap {
            return Ok(None);
        }
        self.build_coverage_guard_candidate(view_id, view, lod_settings, false)
    }

    pub(crate) fn coverage_guard_candidate(
        &mut self,
        view_id: LodRuntimeViewId,
        view: LodView,
        lod_settings: &GaussianLodSettings,
    ) -> Result<Option<LodCandidateFrontier>, LodRuntimeError> {
        if !self.coverage_guard.is_active() {
            return Ok(None);
        }
        self.build_coverage_guard_candidate(view_id, view, lod_settings, true)
    }

    fn build_coverage_guard_candidate(
        &mut self,
        view_id: LodRuntimeViewId,
        live_view: LodView,
        lod_settings: &GaussianLodSettings,
        maintain_runtime_reserve: bool,
    ) -> Result<Option<LodCandidateFrontier>, LodRuntimeError> {
        lod_settings
            .validate()
            .map_err(|error| LodRuntimeError::InvalidSettings(error.to_string()))?;
        let selection_view = self
            .views
            .entry(view_id)
            .or_default()
            .selection_view(live_view, lod_settings.selection_mode);
        selection_view.validate().map_err(|error| match error {
            LodSelectionError::InvalidView(field) => {
                LodRuntimeError::Selection(LodSelectionError::InvalidView(field))
            }
            _ => unreachable!("LodView::validate only emits InvalidView"),
        })?;
        if self.coverage_guard.active_gaussians > lod_settings.budgets.max_active_gaussians {
            return Err(LodRuntimeError::CoverageGuardActiveGaussiansExceedLimit {
                actual: self.coverage_guard.active_gaussians,
                limit: lod_settings.budgets.max_active_gaussians,
            });
        }
        if maintain_runtime_reserve {
            self.prime_coverage_guard_requests()?;
            self.synchronize_coverage_guard_pins()?;
        }
        if self
            .coverage_guard
            .pages
            .iter()
            .any(|page| !self.cache.contains(*page) || !self.decoded_pages.contains_key(page))
            || (maintain_runtime_reserve
                && self.coverage_guard.pinned_pages.len() != self.coverage_guard.pages.len())
        {
            return Ok(None);
        }

        let all_resident_target = self.all_resident_target_frontier(
            view_id,
            selection_view,
            lod_settings.selection_mode == LodSelectionMode::Frozen,
            lod_settings,
        )?;
        let ancestor_fallback_nodes = all_resident_coverage_guard_fallback_nodes(
            &self.hierarchy,
            &self.coverage_guard.nodes,
            &all_resident_target.nodes,
        );
        let requested_target = lod_settings.quality_target();
        let mut achieved_max_error_px = 0.0_f32;
        let mut achieved_max_target_ratio = 0.0_f32;
        for &node in &self.coverage_guard.nodes {
            let metrics = self
                .hierarchy
                .metrics(node)
                .ok_or(LodRuntimeError::MissingNode(node))?;
            if lod_settings.frustum_culling
                && !selection_view.node_is_visible(metrics, lod_settings.frustum_margin)
            {
                continue;
            }
            let error_px = selection_view.projected_error_px(metrics);
            let coverage = selection_view.projected_coverage(metrics);
            let is_leaf = self.hierarchy.children(node).is_empty();
            let pressure = requested_target.node_pressure(
                metrics.quality_threshold(),
                error_px,
                coverage,
                metrics.high_fidelity_certificate,
                is_leaf,
            );
            achieved_max_error_px = achieved_max_error_px.max(error_px);
            achieved_max_target_ratio = achieved_max_target_ratio.max(pressure);
        }
        let residency_degradation = if ancestor_fallback_nodes.is_empty() {
            LodDegradation::None
        } else {
            LodDegradation::Residency
        };
        let degradation = all_resident_target
            .status
            .degradation
            .merge(residency_degradation);
        let status = LodEffectiveStatus {
            requested_target,
            achieved_max_error_px,
            achieved_max_target_ratio,
            degradation,
            active_gaussians: self.coverage_guard.active_gaussians,
            visited_nodes: self
                .coverage_guard
                .nodes
                .len()
                .try_into()
                .unwrap_or(u32::MAX),
            requested_pages: 0,
        };
        let frontier = LodFrontier {
            nodes: self.coverage_guard.nodes.clone(),
            requested_nodes: Vec::new(),
            status,
        };
        let physical_ranges = self.physical_ranges(&frontier)?;
        build_candidate_frontier(
            view_id,
            &physical_ranges,
            &ancestor_fallback_nodes,
            status,
            LodCandidateFrontierBuildOptions {
                selection_view_frozen: lod_settings.selection_mode == LodSelectionMode::Frozen,
                coverage_guard: true,
                temporal_transition: None,
                limit: lod_settings.max_active_gaussians_u32(),
            },
        )
        .map(Some)
    }

    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Pages whose retry budget is exhausted. Terminal pages are not
    /// automatically enqueued by later updates, even if they remain visible.
    pub fn terminal_failures(&self) -> &BTreeSet<LodPageId> {
        &self.terminal_failures
    }

    pub fn is_terminal_failure(&self, page: LodPageId) -> bool {
        self.terminal_failures.contains(&page)
    }

    /// Last typed preprocessing failure observed for a page. The entry is
    /// retained across bounded retries and cleared after success or an explicit
    /// terminal-page retry.
    pub fn page_preprocess_error(&self, page: LodPageId) -> Option<&LodPagePreprocessError> {
        self.preprocess_failures.get(&page)
    }

    /// Last normalized transport failure observed for a page. The entry is
    /// retained across bounded retries and cleared after a payload succeeds or
    /// an explicit terminal-page retry.
    pub fn page_transport_failure(&self, page: LodPageId) -> Option<&LodPageTransportFailure> {
        self.transport_failures.get(&page)
    }

    /// Number of transport starts attempted since the last success or explicit
    /// terminal-page retry.
    pub fn page_attempts(&self, page: LodPageId) -> Option<u32> {
        self.attempts.get(&page).copied()
    }

    /// Explicitly requeues a terminal page using its last request priority.
    /// Attempt accounting restarts at zero; transport begin occurs on the next
    /// bounded update. Returns `false` when the page is not terminal.
    pub fn retry_terminal_failure(&mut self, page: LodPageId) -> Result<bool, LodRuntimeError> {
        if !self.terminal_failures.contains(&page) {
            return Ok(false);
        }
        let request = self
            .terminal_requests
            .remove(&page)
            .ok_or(LodRuntimeError::MissingTerminalRequest(page))?;
        let attempts = self.attempts.remove(&page);
        self.terminal_failures.remove(&page);
        match self.enqueue_pending_request(request) {
            RequestEnqueue::Rejected => {
                self.terminal_failures.insert(page);
                self.terminal_requests.insert(page, request);
                if let Some(attempts) = attempts {
                    self.attempts.insert(page, attempts);
                }
                Err(LodRuntimeError::RetryQueueRejected(page))
            }
            RequestEnqueue::Enqueued
            | RequestEnqueue::Promoted
            | RequestEnqueue::Duplicate
            | RequestEnqueue::Replaced(_) => {
                self.preprocess_failures.remove(&page);
                self.preprocess_retry_deferred_frame.remove(&page);
                self.transport_failures.remove(&page);
                Ok(true)
            }
        }
    }

    /// Polls bounded page work, selects a complete resident frontier, schedules
    /// missing pages, and emits generation-safe physical ranges for this view.
    pub fn update(
        &mut self,
        view: LodView,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
    ) -> Result<LodStreamFrame, LodRuntimeError> {
        let frame = self.begin_frame();
        let result = self.update_view_in_frame(
            frame,
            LodRuntimeViewId::default(),
            view,
            lod_settings,
            streaming_settings,
        );
        let finish = self.finish_frame(frame);
        match result {
            Err(error) => Err(error),
            Ok(mut frame) => {
                finish?;
                frame.split_cohort_capacity_stall = self.split_cohort_capacity_stall;
                Ok(frame)
            }
        }
    }

    /// Updates one camera as one complete orchestration frame. For multiple
    /// cameras in the same application frame, call [`Self::begin_frame`] once
    /// and use [`Self::update_view_in_frame`] so work budgets are shared.
    pub fn update_view(
        &mut self,
        view_id: LodRuntimeViewId,
        view: LodView,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
    ) -> Result<LodStreamFrame, LodRuntimeError> {
        let frame = self.begin_frame();
        let result =
            self.update_view_in_frame(frame, view_id, view, lod_settings, streaming_settings);
        let finish = self.finish_frame(frame);
        match result {
            Err(error) => Err(error),
            Ok(mut frame) => {
                finish?;
                frame.split_cohort_capacity_stall = self.split_cohort_capacity_stall;
                Ok(frame)
            }
        }
    }

    /// Starts one application frame and resets its shared decoded-byte and
    /// transport-start accounting. Pass the returned token to every camera via
    /// [`Self::update_view_in_frame`].
    pub fn begin_frame(&mut self) -> LodRuntimeFrameId {
        if self.epoch != 0 && !self.frame_finished {
            self.reconcile_frame_demand(LodRuntimeFrameId(self.epoch))
                .expect("implicit frame finish must preserve runtime pin invariants");
        }
        debug_assert!(self.frame_completion_holds.is_empty());
        self.epoch = self.epoch.wrapping_add(1).max(1);
        self.frame_decoded_bytes = 0;
        self.frame_request_starts = 0;
        self.frame_finished = false;
        LodRuntimeFrameId(self.epoch)
    }

    /// Cancels page work that no view requested in `frame`.
    ///
    /// Multi-view callers should invoke this once after their final
    /// [`Self::update_view_in_frame`] call. Starting the next frame also
    /// performs this reconciliation as a fail-safe, but explicit completion
    /// releases stale camera-cut work one frame earlier.
    pub fn finish_frame(&mut self, frame: LodRuntimeFrameId) -> Result<(), LodRuntimeError> {
        let current = LodRuntimeFrameId(self.epoch);
        if frame != current || frame.0 == 0 {
            return Err(LodRuntimeError::InvalidFrameToken {
                expected: current,
                actual: frame,
            });
        }
        if !self.frame_finished {
            self.reconcile_frame_demand(frame)?;
        }
        Ok(())
    }

    /// Multi-camera update that shares per-frame work budgets with every other
    /// update using `frame`. A stale token is rejected so callers cannot
    /// accidentally reset or reuse budget accounting.
    pub fn update_view_in_frame(
        &mut self,
        frame: LodRuntimeFrameId,
        view_id: LodRuntimeViewId,
        view: LodView,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
    ) -> Result<LodStreamFrame, LodRuntimeError> {
        let current = LodRuntimeFrameId(self.epoch);
        if frame != current || frame.0 == 0 {
            return Err(LodRuntimeError::InvalidFrameToken {
                expected: current,
                actual: frame,
            });
        }
        if self.frame_finished {
            return Err(LodRuntimeError::FrameAlreadyFinished(frame));
        }
        lod_settings
            .validate()
            .map_err(|error| LodRuntimeError::InvalidSettings(error.to_string()))?;
        streaming_settings
            .validate()
            .map_err(|error| LodRuntimeError::InvalidSettings(error.to_string()))?;
        self.structural_settings
            .validate_compatible(LodRuntimeStructuralSettings::new(
                lod_settings,
                streaming_settings,
            ))?;
        if self.coverage_guard.is_active()
            && self.coverage_guard.active_gaussians > lod_settings.budgets.max_active_gaussians
        {
            return Err(LodRuntimeError::CoverageGuardActiveGaussiansExceedLimit {
                actual: self.coverage_guard.active_gaussians,
                limit: lod_settings.budgets.max_active_gaussians,
            });
        }
        if self.largest_decoded_page.1 > lod_settings.budgets.max_upload_bytes_per_frame {
            return Err(LodRuntimeError::PageDecodedBytesExceedLimit {
                page: self.largest_decoded_page.0,
                actual: self.largest_decoded_page.1,
                limit: lod_settings.budgets.max_upload_bytes_per_frame,
            });
        }
        self.prime_coverage_guard_requests()?;

        let selection_view = self
            .views
            .entry(view_id)
            .or_default()
            .selection_view(view, lod_settings.selection_mode);
        let selection_view_frozen = lod_settings.selection_mode == LodSelectionMode::Frozen;

        let mut completed_pages = Vec::new();
        let mut preprocess_failed_pages = Vec::new();
        let mut failed_pages = Vec::new();
        self.poll_pages(frame, lod_settings, streaming_settings, &mut failed_pages)?;
        if self.drop_predictive_view_blend_cohorts_with_terminal_members()? {
            self.wake_capacity_blocked();
        }

        // Establish this view's current-frame demand before publishing any
        // completed preprocessing result. This prevents a camera cut from
        // committing a stale page merely because its worker won a race with
        // demand reconciliation.
        let policy = LodHysteresisPolicy::from(lod_settings);
        let selection = self.select_frontier(
            frame,
            view_id,
            selection_view,
            selection_view_frozen,
            lod_settings,
        )?;
        let mut frontier = selection.frontier;
        let mut temporal_transition = selection.temporal_transition;
        self.promote_target_requested_speculative_failures(
            &frontier.requested_nodes,
            &mut failed_pages,
        );
        let released_predictive_capacity =
            if lod_settings.quality_endpoint() == LodQualityEndpoint::Continuous {
                self.update_predictive_view_blend_demand(
                    view_id,
                    &frontier.nodes,
                    selection_view,
                    lod_settings.quality_target(),
                )?
            } else {
                self.clear_predictive_view_blend_demand(view_id)?
            };
        if released_predictive_capacity {
            self.wake_capacity_blocked();
        }
        self.observe_split_cohort_candidates(frame, view_id, &frontier)?;
        let mut stable_key = StableSelectionKey {
            view: selection_view,
            policy,
            selection_view_frozen,
            residency_revision: self.residency_revision,
        };
        let mut stable_payload = self
            .views
            .get(&view_id)
            .and_then(|state| state.cached_stable_payload(stable_key))
            .cloned();
        let requested_pages = if stable_payload.is_some() {
            self.refresh_frame_demand(frame, view_id, &frontier)
        } else {
            self.record_frame_demand(frame, view_id, &frontier)
        };
        frontier.status.requested_pages = requested_pages;
        self.commit_preprocessed_pages(
            frame,
            lod_settings,
            streaming_settings,
            &mut completed_pages,
            &mut preprocess_failed_pages,
            &mut failed_pages,
        )?;
        if self.drop_predictive_view_blend_cohorts_with_terminal_members()? {
            self.wake_capacity_blocked();
        }
        if !completed_pages.is_empty() {
            // Newly resident pages may refine the cut immediately, while all
            // expensive verification/decode work still occurred outside this
            // application-thread call on native builds.
            let selection = self.select_frontier(
                frame,
                view_id,
                selection_view,
                selection_view_frozen,
                lod_settings,
            )?;
            frontier = selection.frontier;
            temporal_transition = selection.temporal_transition;
            self.promote_target_requested_speculative_failures(
                &frontier.requested_nodes,
                &mut failed_pages,
            );
            let released_predictive_capacity =
                if lod_settings.quality_endpoint() == LodQualityEndpoint::Continuous {
                    self.update_predictive_view_blend_demand(
                        view_id,
                        &frontier.nodes,
                        selection_view,
                        lod_settings.quality_target(),
                    )?
                } else {
                    self.clear_predictive_view_blend_demand(view_id)?
                };
            if released_predictive_capacity {
                self.wake_capacity_blocked();
            }
            self.observe_split_cohort_candidates(frame, view_id, &frontier)?;
            stable_key.residency_revision = self.residency_revision;
            stable_payload = self
                .views
                .get(&view_id)
                .and_then(|state| state.cached_stable_payload(stable_key))
                .cloned();
            let requested_pages = if stable_payload.is_some() {
                self.refresh_frame_demand(frame, view_id, &frontier)
            } else {
                self.record_frame_demand(frame, view_id, &frontier)
            };
            frontier.status.requested_pages = requested_pages;
        }

        let (physical_ranges, complete_resident_cut) =
            if let Some(payload) = stable_payload.as_ref() {
                #[cfg(test)]
                {
                    self.stable_payload_hits = self.stable_payload_hits.saturating_add(1);
                }
                (
                    payload.physical_ranges.clone(),
                    payload.complete_resident_cut,
                )
            } else {
                if self.update_frontier_pins(view_id, &frontier)? {
                    self.wake_capacity_blocked();
                }
                let physical_ranges = self.physical_ranges(&frontier)?;
                let represented_count: u64 = physical_ranges
                    .iter()
                    .map(|range| u64::from(range.count))
                    .sum();
                if represented_count != frontier.status.active_gaussians {
                    return Err(LodRuntimeError::CandidateCountMismatch {
                        frontier: frontier.status.active_gaussians,
                        physical: represented_count,
                    });
                }
                // A missing child is covered by its selected resident ancestor,
                // but a missing visible root has no possible fallback. Do not let
                // a partial multi-root forest acquire the private GPU commit
                // capability.
                let complete_resident_cut = !frontier.requested_nodes.iter().any(|node| {
                    self.hierarchy.parent(*node).is_none()
                        && !frontier.nodes.iter().any(|selected| selected == node)
                });
                (physical_ranges, complete_resident_cut)
            };
        if let Some(transition) = temporal_transition.as_mut() {
            let morph = self
                .prepare_temporal_morph_batch(
                    view_id,
                    selection_view,
                    lod_settings.quality_target(),
                    &physical_ranges,
                    transition,
                )
                .filter(|morph| {
                    morph
                        .presentation_ranges()
                        .iter()
                        .map(|range| u64::from(range.count))
                        .sum::<u64>()
                        <= lod_settings.budgets.max_active_gaussians
                });
            transition.morph = morph;
            transition.mode = if transition.morph.is_some() {
                LodTemporalTransitionMode::Morphing
            } else {
                if let Some(state) = self.views.get_mut(&view_id) {
                    state.temporal_morph_cache = None;
                }
                LodTemporalTransitionMode::BoundedHardCohort
            };
        } else if let Some(state) = self.views.get_mut(&view_id) {
            state.temporal_morph_cache = None;
        }

        // Child demand is admitted as one selector-atomic cohort. Serializing
        // saturated substitutions prevents missing siblings from repeatedly
        // evicting one another while preserving the ordinary parallel path
        // inside the admitted cohort.
        self.reconcile_active_split_cohort_after_frontier(frame, view_id, &frontier)?;
        // A missing root has no ancestor fallback and must never wait behind a
        // pressured refinement transaction. Its fallback-critical request is
        // independent from non-root cohort admission.
        self.enqueue_missing_roots(&frontier, selection_view)?;
        if self.active_split_cohort.is_some() {
            self.enqueue_active_split_cohort()?;
        } else if !self.views.get(&view_id).is_some_and(|state| {
            state.split_cohort_pressure_frame == frame && state.split_cohort_pressure
        }) {
            self.enqueue_missing(&frontier, selection_view)?;
            self.enqueue_predictive_view_blend_demand(view_id, selection_view)?;
        }
        let started_pages =
            self.start_requests(lod_settings, streaming_settings, &mut failed_pages);
        if self.drop_predictive_view_blend_cohorts_with_terminal_members()? {
            self.wake_capacity_blocked();
        }
        if stable_payload.is_none() {
            let state = self.views.entry(view_id).or_default();
            state.commit_frontier(&frontier.nodes, lod_settings);
            state.promote_stable_payload(
                stable_key,
                &frontier,
                &physical_ranges,
                complete_resident_cut,
            );
        }
        let selection_stable = self
            .views
            .get(&view_id)
            .is_some_and(|state| state.cached_stable_selection(stable_key).is_some());
        let ancestor_fallback_nodes = selected_ancestor_fallback_nodes(&self.hierarchy, &frontier);

        Ok(LodStreamFrame {
            view: view_id,
            frontier,
            physical_ranges,
            ancestor_fallback_nodes,
            selection_view_frozen,
            selection_stable,
            temporal_transition,
            complete_resident_cut,
            cache: self.cache.stats(),
            queued_requests: self.queue.len().try_into().unwrap_or(u32::MAX),
            in_flight_requests: self
                .in_flight
                .len()
                .saturating_add(self.preprocessor.len())
                .try_into()
                .unwrap_or(u32::MAX),
            preprocess: self.preprocessor.stats(),
            capacity_blocked_requests: self.capacity_blocked.len().try_into().unwrap_or(u32::MAX),
            split_cohort_capacity_stall: self.split_cohort_capacity_stall,
            started_pages,
            completed_pages,
            preprocess_failed_pages,
            failed_pages,
        })
    }

    fn reconcile_frame_demand(&mut self, frame: LodRuntimeFrameId) -> Result<(), LodRuntimeError> {
        let owner_was_updated = self
            .active_split_cohort
            .as_ref()
            .is_none_or(|cohort| cohort.owner_updated_frame == frame);
        if !owner_was_updated {
            self.release_active_split_cohort()?;
        }
        // Transient completion holds are not retained-cut pressure. Drop them
        // before exact cohort admission, then reconcile any requests they wake.
        if self.release_frame_completion_holds() {
            self.wake_capacity_blocked();
        }
        self.schedule_next_split_cohort(frame)?;
        let mut demanded = self
            .views
            .values()
            .filter(|view| view.admitted_pages_frame == frame)
            .flat_map(|view| view.admitted_pages.iter().copied())
            .collect::<BTreeSet<_>>();
        if self.coverage_guard.is_active() {
            demanded.extend(self.coverage_guard.pages.iter().copied());
        }

        let cancelled_queued = self
            .queue
            .page_ids()
            .filter(|page| !demanded.contains(page))
            .collect::<Vec<_>>();
        for page in &cancelled_queued {
            self.queue.remove(*page);
            self.attempts.remove(page);
            self.preprocess_retry_deferred_frame.remove(page);
            self.transport_failures.remove(page);
        }

        let cancelled_in_flight = self
            .in_flight
            .keys()
            .filter(|page| !demanded.contains(page))
            .copied()
            .collect::<Vec<_>>();
        for page in &cancelled_in_flight {
            if let Some(request) = self.in_flight.remove(page) {
                self.transport.cancel(&request.ticket);
            }
            self.attempts.remove(page);
            self.preprocess_retry_deferred_frame.remove(page);
            self.transport_failures.remove(page);
        }

        let cancelled_preprocessing = self
            .preprocessor
            .page_ids()
            .into_iter()
            .filter(|page| !demanded.contains(page))
            .collect::<Vec<_>>();
        for page in &cancelled_preprocessing {
            self.preprocessor.cancel(*page);
            self.preprocess_failures.remove(page);
            self.preprocess_retry_deferred_frame.remove(page);
            self.transport_failures.remove(page);
            self.attempts.remove(page);
        }

        let cancelled_capacity_blocked = self
            .capacity_blocked
            .keys()
            .filter(|page| !demanded.contains(page))
            .copied()
            .collect::<Vec<_>>();
        for page in &cancelled_capacity_blocked {
            self.capacity_blocked.remove(page);
            self.attempts.remove(page);
            self.preprocess_retry_deferred_frame.remove(page);
            self.transport_failures.remove(page);
        }

        self.frame_finished = true;
        Ok(())
    }

    fn hold_frame_completion_page(&mut self, page: LodPageId) -> Result<(), LodRuntimeError> {
        if self.frame_completion_holds.contains(&page) {
            return Ok(());
        }
        self.cache
            .pin_fallback(page)
            .map_err(LodRuntimeError::Cache)?;
        self.frame_completion_holds.insert(page);
        Ok(())
    }

    /// Releases every lease acquired by `hold_frame_completion_page`. The set
    /// is taken first so explicit `finish_frame` is idempotent and an omitted
    /// finish cannot carry a stale lease beyond the next `begin_frame`.
    fn release_frame_completion_holds(&mut self) -> bool {
        let held = std::mem::take(&mut self.frame_completion_holds);
        let mut became_evictable = false;
        for page in held {
            became_evictable |= self
                .cache
                .get(page)
                .is_some_and(|resident| resident.pin_count == 1);
            self.cache
                .unpin_fallback(page)
                .expect("a frame completion hold must remain resident and pinned");
        }
        became_evictable
    }

    /// Releases a view's fallback holds without affecting other cameras.
    pub fn remove_view(&mut self, view_id: LodRuntimeViewId) -> Result<bool, LodRuntimeError> {
        if self
            .active_split_cohort
            .as_ref()
            .is_some_and(|cohort| cohort.plan.view == view_id)
        {
            self.release_active_split_cohort()?;
        }
        let Some(state) = self.views.remove(&view_id) else {
            return Ok(false);
        };
        for page in state.pinned_frontier {
            self.cache
                .unpin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
        }
        for page in state.pinned_predictive_pages {
            self.cache
                .unpin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
        }
        self.split_cohort_capacity_stall = None;
        self.wake_capacity_blocked();
        Ok(true)
    }

    /// Restores selector history and cache pins to the cut that the renderer
    /// actually retained after a pending package transaction was cancelled or
    /// failed. Runtime selection advances optimistically before the cross-world
    /// GPU commit; without this acknowledgement seam, the following temporal
    /// step would incorrectly start from an endpoint that was never visible.
    pub(crate) fn restore_rendered_frontier(
        &mut self,
        view_id: LodRuntimeViewId,
        nodes: &[LodNodeId],
    ) -> Result<(), LodRuntimeError> {
        if self
            .active_split_cohort
            .as_ref()
            .is_some_and(|cohort| cohort.plan.view == view_id)
        {
            self.release_active_split_cohort()?;
        }
        let released_predictive_capacity = self.clear_predictive_view_blend_demand(view_id)?;
        let restored = LodFrontier {
            nodes: nodes.to_vec(),
            requested_nodes: Vec::new(),
            status: LodEffectiveStatus::default(),
        };
        let released_capacity = self.update_frontier_pins(view_id, &restored)?;
        let state = self.views.entry(view_id).or_default();
        state.previous_frontier.clear();
        state.previous_frontier.extend_from_slice(nodes);
        // The request which produced this retained cut may differ from the
        // cancelled live request. Force one fresh canonical traversal instead
        // of attaching either policy's cached/hysteresis identity to it.
        state.previous_lod_policy = None;
        state.temporal_demands.clear();
        state.temporal_morph_cache = None;
        let restored_nodes = nodes.iter().copied().collect::<BTreeSet<_>>();
        state
            .late_view_blend_edges
            .retain(|parent| restored_nodes.contains(parent));
        state.stable_selection = None;
        state.all_resident_selection = None;
        state.requested_pages.clear();
        state.requested_pages_frame = LodRuntimeFrameId::default();
        state.admitted_pages.clear();
        state.admitted_pages_frame = LodRuntimeFrameId::default();
        state.split_cohort_candidates.clear();
        state.split_cohort_candidates_frame = LodRuntimeFrameId::default();
        state.split_cohort_pressure = false;
        state.split_cohort_pressure_frame = LodRuntimeFrameId::default();
        if released_capacity || released_predictive_capacity {
            self.wake_capacity_blocked();
        }
        Ok(())
    }

    /// Acknowledges the exact frontier for which RenderWorld has published a
    /// complete ACTIVE/drawable candidate. Late-residency provenance is kept
    /// through optimistic selection, staging, cancellation, and failure; only
    /// this cross-world acknowledgement may retire a parent whose replacement
    /// endpoint has actually become visible.
    pub(crate) fn acknowledge_rendered_frontier(
        &mut self,
        view_id: LodRuntimeViewId,
        nodes: &[LodNodeId],
    ) {
        let rendered_nodes = nodes.iter().copied().collect::<BTreeSet<_>>();
        self.views
            .entry(view_id)
            .or_default()
            .late_view_blend_edges
            .retain(|parent| rendered_nodes.contains(parent));
    }

    /// Rewinds only optimistic selector history to the still-visible cut while
    /// retaining the pending residency pins and confirmed temporal demand.
    /// Bridge orchestration uses this when it defers a valid resident wave: the
    /// same candidate remains protected and reproducible until publication,
    /// rather than advancing through endpoints which were never rendered.
    pub(crate) fn retry_from_rendered_frontier(
        &mut self,
        view_id: LodRuntimeViewId,
        nodes: &[LodNodeId],
    ) -> Result<(), LodRuntimeError> {
        let state = self.views.entry(view_id).or_default();
        state.previous_frontier.clear();
        state.previous_frontier.extend_from_slice(nodes);
        state.previous_lod_policy = None;
        state.stable_selection = None;
        state.all_resident_selection = None;
        Ok(())
    }

    fn record_late_view_blend_demand(
        &mut self,
        view_id: LodRuntimeViewId,
        rendered_frontier: &[LodNodeId],
        requested_nodes: &[LodNodeId],
    ) {
        if rendered_frontier.is_empty() || requested_nodes.is_empty() {
            return;
        }
        let rendered = rendered_frontier.iter().copied().collect::<BTreeSet<_>>();
        let mut late_edges = BTreeSet::new();
        for requested in requested_nodes.iter().copied() {
            let mut cursor = requested;
            while let Some(parent) = self.hierarchy.parent(cursor) {
                if rendered.contains(&parent) {
                    if !self.hierarchy.children(parent).is_empty() {
                        late_edges.insert(parent);
                    }
                    break;
                }
                cursor = parent;
            }
        }
        self.views
            .entry(view_id)
            .or_default()
            .late_view_blend_edges
            .extend(late_edges);
    }

    fn select_frontier(
        &mut self,
        frame: LodRuntimeFrameId,
        view_id: LodRuntimeViewId,
        view: LodView,
        selection_view_frozen: bool,
        lod_settings: &GaussianLodSettings,
    ) -> Result<LodRuntimeSelection, LodRuntimeError> {
        let endpoint = lod_settings.quality_endpoint();
        let policy = LodHysteresisPolicy::from(lod_settings);
        let residency_revision = self.residency_revision;
        let stable_key = StableSelectionKey {
            view,
            policy,
            selection_view_frozen,
            residency_revision,
        };
        if let Some(cached) = self
            .views
            .get(&view_id)
            .and_then(|state| state.cached_stable_selection(stable_key))
            .cloned()
        {
            if endpoint == LodQualityEndpoint::Continuous
                && self.hierarchy.manifest().morph_map.is_some()
            {
                self.record_late_view_blend_demand(view_id, &cached.nodes, &cached.requested_nodes);
            }
            let substitutions = self.stable_view_blend_substitutions(
                &cached.nodes,
                view,
                lod_settings.quality_target(),
            )?;
            let initial_weight_bits = vec![1.0_f32.to_bits(); substitutions.len()];
            return Ok(LodRuntimeSelection {
                frontier: cached,
                temporal_transition: view_blend_transition(
                    substitutions,
                    initial_weight_bits,
                    0,
                    0,
                ),
            });
        }

        #[cfg(test)]
        {
            self.selection_traversals = self.selection_traversals.saturating_add(1);
        }
        // The authoritative destination is stateless. Temporal confirmation
        // suppresses boundary chatter, but no prior approach direction is
        // allowed to change the final cut of a stationary camera.
        let desired = select_frontier_with_visibility(
            &self.hierarchy,
            &|node| {
                self.hierarchy
                    .page(node)
                    .is_some_and(|page| self.cache.contains(page))
            },
            view,
            lod_settings,
            |_, metrics| {
                !lod_settings.frustum_culling
                    || view.node_is_visible(metrics, lod_settings.frustum_margin)
            },
        )
        .map_err(LodRuntimeError::Selection)?;

        let rendered_frontier = self
            .views
            .get(&view_id)
            .map(|state| state.rendered_frontier().to_vec())
            .unwrap_or_default();

        if endpoint == LodQualityEndpoint::Continuous
            && self.hierarchy.manifest().morph_map.is_some()
            && !rendered_frontier.is_empty()
        {
            self.record_late_view_blend_demand(
                view_id,
                &rendered_frontier,
                &desired.requested_nodes,
            );
        }

        if endpoint != LodQualityEndpoint::Continuous {
            let state = self.views.entry(view_id).or_default();
            state.clear_temporal_state();
            state.stable_selection =
                (desired.nodes == rendered_frontier).then(|| StableSelectionCache {
                    key: stable_key,
                    frontier: desired.clone(),
                    payload: None,
                });
            return Ok(LodRuntimeSelection {
                frontier: desired,
                temporal_transition: None,
            });
        }

        // Stable cuts are the overwhelmingly common case. Avoid building a
        // full frontier set and walking parent chains every frame merely to
        // discover that there is no fine-to-coarse substitution to stagger.
        if desired.nodes.as_slice() == rendered_frontier.as_slice() {
            let substitutions = self.stable_view_blend_substitutions(
                &desired.nodes,
                view,
                lod_settings.quality_target(),
            )?;
            let state = self.views.entry(view_id).or_default();
            state.clear_temporal_state();
            state.stable_selection = Some(StableSelectionCache {
                key: stable_key,
                frontier: desired.clone(),
                payload: None,
            });
            let initial_weight_bits = vec![1.0_f32.to_bits(); substitutions.len()];
            return Ok(LodRuntimeSelection {
                frontier: desired,
                temporal_transition: view_blend_transition(
                    substitutions,
                    initial_weight_bits,
                    0,
                    0,
                ),
            });
        }

        self.views.entry(view_id).or_default().stable_selection = None;

        if rendered_frontier.is_empty() {
            self.views
                .entry(view_id)
                .or_default()
                .clear_temporal_state();
            return Ok(LodRuntimeSelection {
                frontier: desired,
                temporal_transition: None,
            });
        }

        let substitutions =
            temporal_substitution_candidates(&self.hierarchy, &rendered_frontier, &desired.nodes)
                .map_err(LodRuntimeError::Selection)?;
        let candidate_keys = substitutions
            .iter()
            .map(|substitution| substitution.key)
            .collect::<BTreeSet<_>>();
        let view_blend_capable = self.hierarchy.manifest().morph_map.is_some();
        let eligible = if view_blend_capable {
            self.views
                .entry(view_id)
                .or_default()
                .clear_temporal_state();
            candidate_keys.clone()
        } else {
            let state = self.views.entry(view_id).or_default();
            confirmed_temporal_keys(&mut state.temporal_demands, &candidate_keys, frame)
        };

        let current_active_gaussians =
            rendered_frontier.iter().try_fold(0_u64, |count, node| {
                let metrics = self
                    .hierarchy
                    .metrics(*node)
                    .ok_or(LodRuntimeError::Selection(LodSelectionError::MissingNode(
                        *node,
                    )))?;
                if !metrics.validate() {
                    return Err(LodRuntimeError::Selection(LodSelectionError::InvalidNode(
                        *node,
                    )));
                }
                count
                    .checked_add(u64::from(metrics.representative_count))
                    .ok_or(LodRuntimeError::Selection(LodSelectionError::CountOverflow))
            })?;
        let step = apply_temporal_substitution_step(
            &rendered_frontier,
            &desired.nodes,
            current_active_gaussians,
            &substitutions,
            &eligible,
            |node| {
                self.hierarchy.page(node).is_some_and(|page| {
                    self.cache.contains(page) && self.decoded_pages.contains_key(&page)
                })
            },
            LodTemporalStepBudget {
                max_active_gaussians: lod_settings.budgets.max_active_gaussians,
                max_changed_gaussians: if view_blend_capable {
                    u64::MAX
                } else {
                    temporal_changed_gaussian_budget(current_active_gaussians)
                },
                max_substitutions: if view_blend_capable {
                    usize::MAX
                } else {
                    LOD_TEMPORAL_MAX_SUBSTITUTIONS_PER_FRAME
                },
            },
        )
        .map_err(LodRuntimeError::Selection)?;
        let frontier = temporal_frontier_with_visibility(
            &self.hierarchy,
            &desired,
            &step,
            view,
            lod_settings,
            |_, metrics| {
                !lod_settings.frustum_culling
                    || view.node_is_visible(metrics, lod_settings.frustum_margin)
            },
        )
        .map_err(LodRuntimeError::Selection)?;
        let transition_substitutions = if view_blend_capable {
            self.merge_view_blend_substitutions(
                &step.substitutions,
                &frontier.nodes,
                view,
                lod_settings.quality_target(),
            )?
        } else {
            step.substitutions.clone()
        };
        let applied_keys = step
            .substitutions
            .iter()
            .map(|substitution| substitution.key)
            .collect::<BTreeSet<_>>();
        let initial_weight_bits = transition_substitutions
            .iter()
            .map(|substitution| {
                if !applied_keys.contains(&substitution.key) {
                    // A persistent edge rebuilt from an exact child-side
                    // frontier inherits the already-visible children.
                    1.0_f32.to_bits()
                } else {
                    match substitution.key.direction {
                        LodTemporalDirection::Refine => 0.0_f32.to_bits(),
                        LodTemporalDirection::Coarsen => 1.0_f32.to_bits(),
                    }
                }
            })
            .collect::<Vec<_>>();
        let temporal_transition = view_blend_transition(
            transition_substitutions,
            initial_weight_bits,
            step.changed_gaussians,
            step.atomic_budget_overshoot,
        );
        if step.reached_target && frontier.requested_nodes.is_empty() {
            let state = self.views.entry(view_id).or_default();
            state.clear_temporal_state();
            state.stable_selection = Some(StableSelectionCache {
                key: stable_key,
                frontier: frontier.clone(),
                payload: None,
            });
        }
        Ok(LodRuntimeSelection {
            frontier,
            temporal_transition,
        })
    }

    fn split_cohort_pressure(
        &self,
        frontier: &LodFrontier<LodNodeId>,
    ) -> Result<bool, LodRuntimeError> {
        if !self.capacity_blocked.is_empty() {
            return Ok(true);
        }
        let mut pending = frontier
            .requested_nodes
            .iter()
            .filter_map(|node| self.hierarchy.page(*node))
            .collect::<BTreeSet<_>>();
        pending.extend(self.queue.page_ids());
        pending.extend(self.in_flight.keys().copied());
        pending.extend(self.preprocessor.page_ids());
        pending.retain(|page| {
            !self.cache.contains(*page)
                && !self.terminal_failures.contains(page)
                && !self
                    .speculative_prefetch_terminal_requests
                    .contains_key(page)
        });

        let stats = self.cache.stats();
        let mut pages = u64::from(stats.resident_pages);
        let mut bytes = stats.resident_bytes;
        let mut gaussians = stats.resident_gaussians;
        for page in pending {
            let descriptor = self
                .hierarchy
                .page_descriptor(page)
                .ok_or(LodRuntimeError::MissingPageDescriptor(page))?;
            pages = pages
                .checked_add(1)
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
            bytes = bytes
                .checked_add(descriptor.decoded_len)
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
            gaussians = gaussians
                .checked_add(u64::from(descriptor.gaussian_count))
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
        }
        let limits = self.cache.limits();
        Ok(pages > u64::from(limits.max_pages)
            || bytes > limits.max_bytes
            || gaussians > limits.max_gaussians)
    }

    fn observe_split_cohort_candidates(
        &mut self,
        frame: LodRuntimeFrameId,
        view_id: LodRuntimeViewId,
        frontier: &LodFrontier<LodNodeId>,
    ) -> Result<(), LodRuntimeError> {
        let pressure = self.split_cohort_pressure(frontier)?;
        let selected = frontier.nodes.iter().copied().collect::<BTreeSet<_>>();
        let mut parents = BTreeSet::new();
        let mut missing_root = false;
        for &requested in &frontier.requested_nodes {
            if let Some(parent) = self.hierarchy.parent(requested)
                && selected.contains(&parent)
            {
                parents.insert(parent);
            } else if self.hierarchy.parent(requested).is_none() {
                missing_root = true;
            }
        }
        let mut plans = Vec::with_capacity(parents.len());
        for parent in parents {
            let mut pages = BTreeSet::new();
            let mut terminal = false;
            for &child in self.hierarchy.children(parent) {
                let page = self
                    .hierarchy
                    .page(child)
                    .ok_or(LodRuntimeError::MissingNode(child))?;
                terminal |= self.terminal_failures.contains(&page);
                pages.insert(page);
            }
            // A terminal child can never complete the atomic substitution.
            // Its siblings must not consume cache capacity until explicit retry.
            if !terminal && !pages.is_empty() {
                plans.push(LodSplitCohortPlan {
                    view: view_id,
                    parent,
                    pages,
                });
            }
        }
        plans.sort_unstable_by_key(LodSplitCohortPlan::key);
        // Root coverage has no ancestor transaction to cohort. Under pressure,
        // let fallback-critical roots complete before admitting any non-root
        // substitution for this view. Clearing the plans (rather than the
        // pressure bit) keeps child pages out of the ordinary fast path.
        if pressure && missing_root {
            plans.clear();
        }

        if let Some(active) = self.active_split_cohort.as_mut()
            && active.plan.view == view_id
        {
            active.owner_updated_frame = frame;
        }
        let state = self.views.entry(view_id).or_default();
        state.split_cohort_candidates = plans;
        state.split_cohort_candidates_frame = frame;
        state.split_cohort_pressure = pressure;
        state.split_cohort_pressure_frame = frame;
        Ok(())
    }

    fn reconcile_active_split_cohort_after_frontier(
        &mut self,
        frame: LodRuntimeFrameId,
        view_id: LodRuntimeViewId,
        frontier: &LodFrontier<LodNodeId>,
    ) -> Result<(), LodRuntimeError> {
        let Some(active) = self.active_split_cohort.as_ref() else {
            return Ok(());
        };
        if active.plan.view != view_id {
            return Ok(());
        }
        let parent_selected = frontier.nodes.contains(&active.plan.parent);
        let plan_still_requested = self.views.get(&view_id).is_some_and(|state| {
            state
                .split_cohort_candidates
                .iter()
                .any(|plan| plan.parent == active.plan.parent && plan.pages == active.plan.pages)
        });
        if !parent_selected || !plan_still_requested {
            self.release_active_split_cohort()?;
            // `record_frame_demand` ran while this cohort still owned the
            // view, so its pages were included in admitted work. Recompute
            // after releasing ownership; otherwise a camera/policy change can
            // keep obsolete siblings live through finish-frame cancellation.
            self.record_frame_demand(frame, view_id, frontier);
        }
        Ok(())
    }

    fn release_active_split_cohort(&mut self) -> Result<bool, LodRuntimeError> {
        let Some(cohort) = self.active_split_cohort.take() else {
            return Ok(false);
        };
        for page in cohort.pinned_pages {
            self.cache
                .unpin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
        }
        self.split_cohort_capacity_stall = None;
        self.wake_capacity_blocked();
        Ok(true)
    }

    /// Releases a cohort whose once-valid capacity proof no longer covers new
    /// current-frame demand, then restores only the owner's independently
    /// admitted work. This keeps overlapping root/direct pages live while
    /// making cohort-only requests eligible for finish-frame cancellation.
    fn preempt_active_split_cohort(
        &mut self,
        frame: LodRuntimeFrameId,
    ) -> Result<bool, LodRuntimeError> {
        let Some((owner, base_pages, base_frame)) =
            self.active_split_cohort.as_ref().map(|cohort| {
                (
                    cohort.plan.view,
                    cohort.owner_base_admitted_pages.clone(),
                    cohort.owner_base_admitted_pages_frame,
                )
            })
        else {
            return Ok(false);
        };
        self.release_active_split_cohort()?;
        if base_frame == frame
            && let Some(state) = self.views.get_mut(&owner)
            && state.admitted_pages_frame == frame
        {
            state.admitted_pages = base_pages;
        }
        Ok(true)
    }

    fn retained_split_pages(&self) -> BTreeSet<LodPageId> {
        // An active coverage guard owns its entire bounded transaction, even
        // while some pages are still queued or decoding. Guard-first commit
        // will eventually pin every page, so admitting against only the
        // already-resident subset can overbook the cache before that happens.
        // Released package bootstraps no longer own any part of this union.
        let mut pages = if self.coverage_guard.is_active() {
            self.coverage_guard.pages.clone()
        } else {
            BTreeSet::new()
        };
        pages.extend(self.caller_page_leases.keys().copied());
        for view in self.views.values() {
            pages.extend(view.pinned_frontier.iter().copied());
            pages.extend(view.pinned_predictive_pages.iter().copied());
            // Ordinary no-eviction waves and fallback-critical roots may be
            // admitted by another view in this same frame. They can become
            // resident before this cohort completes, so exact admission must
            // reserve their physical footprint as well.
            if view.admitted_pages_frame == LodRuntimeFrameId(self.epoch) {
                pages.extend(view.admitted_pages.iter().copied());
            }
        }
        if let Some(cohort) = &self.active_split_cohort {
            pages.extend(cohort.pinned_pages.iter().copied());
        }
        pages.retain(|page| self.page_reserves_streaming_capacity(*page));
        pages
    }

    /// A resident page already consumes physical capacity even if a later
    /// request failure was marked terminal. A nonresident terminal page has no
    /// queued future work until explicit retry and therefore must not reserve a
    /// slot in cohort admission proofs.
    fn page_reserves_streaming_capacity(&self, page: LodPageId) -> bool {
        self.cache.contains(page)
            || (!self.terminal_failures.contains(&page)
                && !self
                    .speculative_prefetch_terminal_requests
                    .contains_key(&page))
    }

    fn split_cohort_capacity(
        &self,
        plan: &LodSplitCohortPlan,
    ) -> Result<(bool, LodSplitCohortCapacityStall), LodRuntimeError> {
        let mut pages = self.retained_split_pages();
        pages.extend(
            plan.pages
                .iter()
                .copied()
                .filter(|page| self.page_reserves_streaming_capacity(*page)),
        );
        let (required_decoded_bytes, required_gaussians, _) =
            LodRuntimeCoverageGuard::page_footprint(&self.hierarchy, &pages)?;
        let required_pages = u64::try_from(pages.len()).unwrap_or(u64::MAX);
        let limits = self.cache.limits();
        let stall = LodSplitCohortCapacityStall {
            view: plan.view,
            parent: plan.parent,
            required_pages,
            limit_pages: u64::from(limits.max_pages),
            required_decoded_bytes,
            limit_decoded_bytes: limits.max_bytes,
            required_gaussians,
            limit_gaussians: limits.max_gaussians,
        };
        Ok((
            required_pages <= u64::from(limits.max_pages)
                && required_decoded_bytes <= limits.max_bytes
                && required_gaussians <= limits.max_gaussians,
            stall,
        ))
    }

    fn round_robin_split_plans(&self) -> Vec<LodSplitCohortPlan> {
        let mut by_view = BTreeMap::<LodRuntimeViewId, Vec<LodSplitCohortPlan>>::new();
        for state in self.views.values() {
            if state.split_cohort_candidates_frame == LodRuntimeFrameId(self.epoch)
                && state.split_cohort_pressure_frame == LodRuntimeFrameId(self.epoch)
                && state.split_cohort_pressure
            {
                for plan in &state.split_cohort_candidates {
                    by_view.entry(plan.view).or_default().push(plan.clone());
                }
            }
        }
        for plans in by_view.values_mut() {
            plans.sort_unstable_by_key(|plan| plan.parent);
        }
        let mut views = by_view.keys().copied().collect::<Vec<_>>();
        if let Some((cursor_view, _)) = self.split_cohort_cursor {
            let split = views.partition_point(|view| *view <= cursor_view);
            views.rotate_left(split);
        }

        let mut ordered = Vec::new();
        for view in views {
            let mut plans = by_view.remove(&view).unwrap_or_default();
            if let Some((cursor_view, cursor_parent)) = self.split_cohort_cursor
                && cursor_view == view
            {
                let split = plans.partition_point(|plan| plan.parent <= cursor_parent);
                plans.rotate_left(split);
            }
            ordered.extend(plans);
        }
        ordered
    }

    fn admit_split_cohort(
        &mut self,
        frame: LodRuntimeFrameId,
        plan: LodSplitCohortPlan,
    ) -> Result<(), LodRuntimeError> {
        debug_assert!(self.active_split_cohort.is_none());
        let owner_base_admitted_pages = self
            .views
            .get(&plan.view)
            .filter(|owner| owner.admitted_pages_frame == frame)
            .map(|owner| owner.admitted_pages.clone())
            .unwrap_or_default();
        let mut pinned_pages = BTreeSet::new();
        for &page in &plan.pages {
            if self.cache.contains(page) {
                if let Err(error) = self.cache.pin_fallback(page) {
                    for pinned in pinned_pages {
                        self.cache
                            .unpin_fallback(pinned)
                            .expect("split-cohort admission rollback owns its pins");
                    }
                    return Err(LodRuntimeError::Cache(error));
                }
                pinned_pages.insert(page);
            }
        }
        self.split_cohort_cursor = Some(plan.key());
        self.active_split_cohort = Some(LodActiveSplitCohort {
            plan: plan.clone(),
            pinned_pages,
            owner_updated_frame: frame,
            owner_base_admitted_pages,
            owner_base_admitted_pages_frame: frame,
        });
        let admitted_plan_pages = plan
            .pages
            .iter()
            .copied()
            .filter(|page| self.page_reserves_streaming_capacity(*page))
            .collect::<BTreeSet<_>>();
        let owner = self.views.entry(plan.view).or_default();
        if owner.admitted_pages_frame != frame {
            owner.admitted_pages.clear();
        }
        owner.admitted_pages.extend(admitted_plan_pages);
        owner.admitted_pages_frame = frame;
        self.split_cohort_capacity_stall = None;
        self.enqueue_active_split_cohort()
    }

    fn enqueue_active_split_cohort(&mut self) -> Result<(), LodRuntimeError> {
        let Some(plan) = self
            .active_split_cohort
            .as_ref()
            .map(|cohort| cohort.plan.clone())
        else {
            return Ok(());
        };
        for page_id in plan.pages {
            if self.cache.contains(page_id)
                || self.in_flight.contains_key(&page_id)
                || self.preprocessor.contains(page_id)
                || self.terminal_failures.contains(&page_id)
            {
                continue;
            }
            let descriptor = self
                .hierarchy
                .page_descriptor(page_id)
                .ok_or(LodRuntimeError::MissingPageDescriptor(page_id))?;
            let mut request = self.capacity_blocked.remove(&page_id).unwrap_or_else(|| {
                PageRequest::new(page_id, PageRequestPriority::visible(u32::MAX))
            });
            request.priority = PageRequestPriority::visible(u32::MAX);
            request.expected_bytes = descriptor
                .storage
                .as_ref()
                .map(|storage| storage.encoded_len);
            let _ = self.enqueue_pending_request(request);
        }
        Ok(())
    }

    fn schedule_next_split_cohort(
        &mut self,
        frame: LodRuntimeFrameId,
    ) -> Result<(), LodRuntimeError> {
        if let Some(plan) = self
            .active_split_cohort
            .as_ref()
            .map(|cohort| cohort.plan.clone())
        {
            let (fits, _) = self.split_cohort_capacity(&plan)?;
            if fits {
                self.split_cohort_capacity_stall = None;
                return Ok(());
            }
            self.preempt_active_split_cohort(frame)?;
        }
        let plans = self.round_robin_split_plans();
        let mut first_stall = None;
        for plan in plans {
            let (fits, stall) = self.split_cohort_capacity(&plan)?;
            if fits {
                return self.admit_split_cohort(frame, plan);
            }
            first_stall.get_or_insert(stall);
        }
        self.split_cohort_capacity_stall = first_stall;
        Ok(())
    }

    fn pin_active_split_cohort_page(&mut self, page: LodPageId) -> Result<(), LodRuntimeError> {
        let should_pin = self.active_split_cohort.as_ref().is_some_and(|cohort| {
            cohort.plan.pages.contains(&page) && !cohort.pinned_pages.contains(&page)
        });
        if !should_pin {
            return Ok(());
        }
        self.cache
            .pin_fallback(page)
            .map_err(LodRuntimeError::Cache)?;
        self.active_split_cohort
            .as_mut()
            .expect("checked active split cohort")
            .pinned_pages
            .insert(page);
        Ok(())
    }

    fn record_frame_demand(
        &mut self,
        frame: LodRuntimeFrameId,
        view_id: LodRuntimeViewId,
        frontier: &LodFrontier<LodNodeId>,
    ) -> u32 {
        let requested_page_count = frontier
            .requested_nodes
            .iter()
            .filter_map(|node| self.hierarchy.page(*node))
            .collect::<BTreeSet<_>>()
            .len()
            .try_into()
            .unwrap_or(u32::MAX);
        let requested_pages = frontier
            .requested_nodes
            .iter()
            .filter_map(|node| self.hierarchy.page(*node))
            .collect::<BTreeSet<_>>();
        let pressured = self.views.get(&view_id).is_some_and(|state| {
            state.split_cohort_pressure_frame == frame && state.split_cohort_pressure
        });
        let predictive_pages = if pressured {
            BTreeSet::new()
        } else {
            self.views
                .get(&view_id)
                .into_iter()
                .flat_map(|state| state.predictive_view_blend_nodes.values())
                .flatten()
                .filter_map(|node| self.hierarchy.page(*node))
                .filter(|page| self.page_reserves_streaming_capacity(*page))
                .collect::<BTreeSet<_>>()
        };
        let base_admitted_pages = if pressured {
            frontier
                .requested_nodes
                .iter()
                .filter(|node| self.hierarchy.parent(**node).is_none())
                .filter_map(|node| self.hierarchy.page(*node))
                .filter(|page| self.page_reserves_streaming_capacity(*page))
                .collect::<BTreeSet<_>>()
        } else {
            requested_pages
                .iter()
                .copied()
                .filter(|page| self.page_reserves_streaming_capacity(*page))
                .collect()
        };
        let active_pages = self
            .active_split_cohort
            .as_ref()
            .filter(|cohort| cohort.plan.view == view_id && cohort.owner_updated_frame == frame)
            .map(|cohort| {
                cohort
                    .plan
                    .pages
                    .iter()
                    .copied()
                    .filter(|page| self.page_reserves_streaming_capacity(*page))
                    .collect::<BTreeSet<_>>()
            });
        if let Some(cohort) = self
            .active_split_cohort
            .as_mut()
            .filter(|cohort| cohort.plan.view == view_id && cohort.owner_updated_frame == frame)
        {
            cohort
                .owner_base_admitted_pages
                .clone_from(&base_admitted_pages);
            cohort.owner_base_admitted_pages_frame = frame;
        }
        let mut admitted_pages = base_admitted_pages;
        admitted_pages.extend(active_pages.into_iter().flatten());
        admitted_pages.extend(predictive_pages);
        let view_state = self.views.entry(view_id).or_default();
        view_state.requested_pages = requested_pages;
        view_state.requested_pages_frame = frame;
        view_state.admitted_pages = admitted_pages;
        view_state.admitted_pages_frame = frame;
        requested_page_count
    }

    /// Advances the demand epoch for an exact cached selector result without
    /// rebuilding its immutable requested-page set. Frame reconciliation still
    /// observes every pending request as live.
    fn refresh_frame_demand(
        &mut self,
        frame: LodRuntimeFrameId,
        view_id: LodRuntimeViewId,
        frontier: &LodFrontier<LodNodeId>,
    ) -> u32 {
        self.record_frame_demand(frame, view_id, frontier);
        frontier
            .requested_nodes
            .iter()
            .filter_map(|node| self.hierarchy.page(*node))
            .collect::<BTreeSet<_>>()
            .len()
            .try_into()
            .unwrap_or(u32::MAX)
    }

    fn demanded_in_frame(&self, frame: LodRuntimeFrameId, page: LodPageId) -> bool {
        self.coverage_guard.contains_active_page(page)
            || self.active_split_cohort.as_ref().is_some_and(|cohort| {
                cohort.owner_updated_frame == frame && cohort.plan.pages.contains(&page)
            })
            || self.views.values().any(|view| {
                view.admitted_pages_frame == frame && view.admitted_pages.contains(&page)
            })
    }

    fn poll_pages(
        &mut self,
        frame: LodRuntimeFrameId,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
        failed_pages: &mut Vec<LodPageId>,
    ) -> Result<(), LodRuntimeError> {
        let pages = self.in_flight.keys().copied().collect::<Vec<_>>();
        for page_id in pages {
            let Some(in_flight) = self.in_flight.get(&page_id).cloned() else {
                continue;
            };
            let descriptor = self
                .hierarchy
                .page_descriptor(page_id)
                .cloned()
                .ok_or(LodRuntimeError::MissingPageDescriptor(page_id))?;
            let encoded_bytes = in_flight
                .request
                .expected_bytes
                .unwrap_or_else(|| streaming_settings.effective_max_encoded_page_bytes());
            if !self
                .preprocessor
                .has_capacity_for(encoded_bytes, descriptor.decoded_len)
            {
                // Leave the transport ticket untouched until exact count and
                // byte admission is available.
                continue;
            }
            match self.transport.poll(&in_flight.ticket) {
                // Pending also covers transport-local admission backpressure.
                // Keeping the existing ticket in flight makes that condition
                // retry-neutral; only an actual `Failed` poll consumes the
                // attempt already recorded when the ticket was created.
                PagePoll::Pending => {}
                PagePoll::Ready(payload) => {
                    self.in_flight.remove(&page_id);
                    self.preprocess_failures.remove(&page_id);
                    self.transport_failures.remove(&page_id);
                    let limits = page_codec_limits(
                        &descriptor,
                        streaming_settings.effective_max_encoded_page_bytes(),
                    );
                    self.preprocessor
                        .submit(LodPagePreprocessInput {
                            request: in_flight.request,
                            payload,
                            descriptor,
                            limits,
                            max_encoded_page_bytes: streaming_settings
                                .effective_max_encoded_page_bytes(),
                            support_sigma: self.hierarchy.manifest().build.settings.support_sigma,
                        })
                        .map_err(LodRuntimeError::PreprocessAdmission)?;
                }
                PagePoll::Failed(error) => {
                    self.in_flight.remove(&page_id);
                    self.transport_failures
                        .insert(page_id, T::classify_error(&error));
                    self.retry_or_fail(
                        in_flight.request,
                        streaming_settings.retry_limit,
                        failed_pages,
                    );
                }
            }
        }
        self.preprocessor.advance(
            frame.sequence(),
            NonZeroU32::new(
                lod_settings
                    .budgets
                    .max_cooperative_preprocess_gaussians_per_frame,
            )
            .expect("validated cooperative preprocessing budget is non-zero"),
        );
        Ok(())
    }

    fn commit_preprocessed_pages(
        &mut self,
        frame: LodRuntimeFrameId,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
        completed_pages: &mut Vec<LodPageId>,
        preprocess_failed_pages: &mut Vec<LodPageId>,
        failed_pages: &mut Vec<LodPageId>,
    ) -> Result<(), LodRuntimeError> {
        let mut ready_pages = self.preprocessor.ready_page_ids();
        ready_pages.sort_unstable_by_key(|page| {
            let admission_class = if self.coverage_guard.contains_active_page(*page) {
                0
            } else if self
                .active_split_cohort
                .as_ref()
                .is_some_and(|cohort| cohort.plan.pages.contains(page))
            {
                1
            } else {
                2
            };
            (admission_class, *page)
        });
        for page_id in ready_pages {
            // A camera page can decode faster than a larger presentation-guard
            // page. Publishing and pinning it first could consume the final
            // cache slot and capacity-block the active global guard. Keep
            // bounded ready work staged until every guard page owns its
            // fallback pin; guard pages themselves are always considered
            // first above. A terminal guard failure is already surfaced to the
            // caller and must not starve ordinary streaming forever.
            if !self.coverage_guard.contains_active_page(page_id)
                && self.coverage_guard.is_active()
                && self.coverage_guard.pinned_pages.len() != self.coverage_guard.pages.len()
                && !self
                    .coverage_guard
                    .pages
                    .iter()
                    .any(|guard| self.terminal_failures.contains(guard))
            {
                continue;
            }
            // A later view in this frame may still demand the page. Leave it
            // ready until demand is known instead of publishing or discarding
            // it speculatively.
            if !self.demanded_in_frame(frame, page_id) {
                continue;
            }
            let descriptor = self
                .hierarchy
                .page_descriptor(page_id)
                .cloned()
                .ok_or(LodRuntimeError::MissingPageDescriptor(page_id))?;
            if self
                .frame_decoded_bytes
                .checked_add(descriptor.decoded_len)
                .is_none_or(|bytes| bytes > lod_settings.budgets.max_upload_bytes_per_frame)
            {
                continue;
            }
            let Some(output) = self.preprocessor.take_ready(page_id) else {
                continue;
            };
            let page = match output.result {
                Ok(page) => page,
                Err(error) => {
                    self.preprocess_failures.insert(page_id, error);
                    preprocess_failed_pages.push(page_id);
                    self.preprocess_retry_deferred_frame.insert(page_id, frame);
                    self.retry_or_fail(
                        output.request,
                        streaming_settings.retry_limit,
                        failed_pages,
                    );
                    continue;
                }
            };
            if let Some(ranges) = self.shared_page_node_ranges.get(&page_id)
                && let Err(error) = validate_shared_page_node_ranges(
                    &page,
                    ranges,
                    self.hierarchy.manifest().build.settings.support_sigma,
                )
            {
                self.preprocess_failures.insert(page_id, error);
                preprocess_failed_pages.push(page_id);
                self.preprocess_retry_deferred_frame.insert(page_id, frame);
                self.retry_or_fail(output.request, streaming_settings.retry_limit, failed_pages);
                continue;
            }
            self.frame_decoded_bytes = self
                .frame_decoded_bytes
                .checked_add(descriptor.decoded_len)
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
            let insertion = match self.cache.insert(
                page_id,
                descriptor.decoded_len,
                u64::from(descriptor.gaussian_count),
                self.epoch,
            ) {
                Ok(insertion) => insertion,
                Err(PageCacheError::InsufficientEvictableCapacity) => {
                    // Keep only bounded request metadata. Retrying the same
                    // payload before pin/cache state changes would redownload
                    // and decode it forever.
                    self.capacity_blocked.insert(page_id, output.request);
                    continue;
                }
                Err(error) => return Err(LodRuntimeError::Cache(error)),
            };
            let residency_changed = !insertion.already_resident;
            for evicted in insertion.evicted {
                self.decoded_pages.remove(&evicted);
            }
            self.decoded_pages.insert(page_id, page);
            // Guard ownership is established first so fallback-critical pages
            // remain protected when the frame lease is released. Package-only
            // ownership is later transferred to the first visible cut's lease.
            self.pin_coverage_guard_page(page_id)?;
            // Cohort ownership is independent from any guard/view/package pin.
            // Acquire it before another ready page can invoke cache insertion.
            self.pin_active_split_cohort_page(page_id)?;
            // Acquire the transient lease before the loop considers another
            // ready page. Otherwise that later insertion can evict this page
            // while it is still present in `completed_pages`.
            self.hold_frame_completion_page(page_id)?;
            if residency_changed {
                self.residency_revision = self.residency_revision.wrapping_add(1).max(1);
            }
            self.clear_failure_state(page_id);
            self.capacity_blocked.remove(&page_id);
            // Insertion consumes capacity; even when it evicts an unpinned
            // page it does not create room for another decoded page. Waking
            // every blocked request here causes avoidable fetch/decode/reject
            // churn while a stationary cache is saturated. Lease and view-pin
            // release paths wake blocked work when capacity can actually have
            // become evictable.
            completed_pages.push(page_id);
        }
        Ok(())
    }

    fn retry_or_fail(
        &mut self,
        mut request: PageRequest,
        retry_limit: u32,
        failed_pages: &mut Vec<LodPageId>,
    ) {
        if self.terminal_failures.contains(&request.page_id)
            || self
                .speculative_prefetch_terminal_requests
                .contains_key(&request.page_id)
        {
            self.preprocess_retry_deferred_frame
                .remove(&request.page_id);
            return;
        }
        let attempts = self.attempts.get(&request.page_id).copied().unwrap_or(0);
        let maximum_attempts = retry_limit.saturating_add(1);
        if attempts < maximum_attempts {
            self.queue.enqueue(request);
        } else if request.priority.class == PageRequestClass::Prefetch
            && !self.page_is_target_requested_this_frame(request.page_id)
        {
            self.queue.remove(request.page_id);
            self.preprocess_retry_deferred_frame
                .remove(&request.page_id);
            self.speculative_prefetch_terminal_requests
                .entry(request.page_id)
                .or_insert(request);
            for view in self.views.values_mut() {
                view.admitted_pages.remove(&request.page_id);
            }
            if let Some(cohort) = self.active_split_cohort.as_mut() {
                cohort.owner_base_admitted_pages.remove(&request.page_id);
            }
        } else if self.terminal_failures.insert(request.page_id) {
            if request.priority.class == PageRequestClass::Prefetch {
                request.priority = PageRequestPriority::visible(request.priority.urgency);
            }
            self.queue.remove(request.page_id);
            self.preprocess_retry_deferred_frame
                .remove(&request.page_id);
            self.terminal_requests.insert(request.page_id, request);
            for view in self.views.values_mut() {
                view.admitted_pages.remove(&request.page_id);
            }
            if let Some(cohort) = self.active_split_cohort.as_mut() {
                cohort.owner_base_admitted_pages.remove(&request.page_id);
            }
            failed_pages.push(request.page_id);
        }
    }

    fn clear_failure_state(&mut self, page: LodPageId) {
        self.attempts.remove(&page);
        self.terminal_failures.remove(&page);
        self.terminal_requests.remove(&page);
        self.speculative_prefetch_terminal_requests.remove(&page);
        self.preprocess_failures.remove(&page);
        self.preprocess_retry_deferred_frame.remove(&page);
        self.transport_failures.remove(&page);
    }

    fn page_is_target_requested_this_frame(&self, page: LodPageId) -> bool {
        let frame = LodRuntimeFrameId(self.epoch);
        self.views
            .values()
            .any(|view| view.requested_pages_frame == frame && view.requested_pages.contains(&page))
    }

    /// Turns a retry-exhausted speculative miss into an ordinary visible
    /// terminal failure only when canonical selection actually requests it.
    /// Package degradation and `failed_pages` therefore describe target work,
    /// never an optional predictive sibling cohort.
    fn promote_target_requested_speculative_failures(
        &mut self,
        requested_nodes: &[LodNodeId],
        failed_pages: &mut Vec<LodPageId>,
    ) {
        let requested_pages = requested_nodes
            .iter()
            .filter_map(|node| self.hierarchy.page(*node))
            .collect::<BTreeSet<_>>();
        for page in requested_pages {
            let Some(mut request) = self.speculative_prefetch_terminal_requests.remove(&page)
            else {
                continue;
            };
            request.priority = PageRequestPriority::visible(request.priority.urgency);
            if self.terminal_failures.insert(page) {
                self.terminal_requests.insert(page, request);
                for view in self.views.values_mut() {
                    view.admitted_pages.remove(&page);
                }
                if let Some(cohort) = self.active_split_cohort.as_mut() {
                    cohort.owner_base_admitted_pages.remove(&page);
                }
                failed_pages.push(page);
            }
        }
    }

    fn wake_capacity_blocked(&mut self) {
        let requests = self.capacity_blocked.values().copied().collect::<Vec<_>>();
        for request in requests {
            if !matches!(self.queue.enqueue(request), RequestEnqueue::Rejected) {
                self.capacity_blocked.remove(&request.page_id);
            }
        }
    }

    fn pending_request_count(&self) -> usize {
        self.queue
            .len()
            .saturating_add(self.in_flight.len())
            .saturating_add(self.preprocessor.len())
            .saturating_add(self.capacity_blocked.len())
    }

    fn prime_coverage_guard_requests(&mut self) -> Result<(), LodRuntimeError> {
        if !self.coverage_guard.is_active() {
            return Ok(());
        }
        // Root/page cardinality is bounded and validated at construction; this
        // small copy avoids holding an immutable guard borrow while promoting
        // requests in the mutable bounded queue.
        let pages = self
            .coverage_guard
            .pages
            .iter()
            .copied()
            .collect::<Vec<_>>();
        for page_id in pages {
            if (self.cache.contains(page_id) && self.decoded_pages.contains_key(&page_id))
                || self.in_flight.contains_key(&page_id)
                || self.preprocessor.contains(page_id)
                || self.terminal_failures.contains(&page_id)
                || self.capacity_blocked.contains_key(&page_id)
            {
                continue;
            }
            let descriptor = self
                .hierarchy
                .page_descriptor(page_id)
                .ok_or(LodRuntimeError::MissingPageDescriptor(page_id))?;
            let mut request =
                PageRequest::new(page_id, PageRequestPriority::fallback_critical(u32::MAX));
            request.expected_bytes = descriptor
                .storage
                .as_ref()
                .map(|storage| storage.encoded_len);
            match self.enqueue_pending_request(request) {
                RequestEnqueue::Enqueued
                | RequestEnqueue::Promoted
                | RequestEnqueue::Duplicate
                | RequestEnqueue::Replaced(_)
                | RequestEnqueue::Rejected => {}
            }
        }
        Ok(())
    }

    fn pin_coverage_guard_page(&mut self, page: LodPageId) -> Result<bool, LodRuntimeError> {
        if !self.coverage_guard.contains_active_page(page)
            || self.coverage_guard.pinned_pages.contains(&page)
        {
            return Ok(false);
        }
        self.cache
            .pin_fallback(page)
            .map_err(LodRuntimeError::Cache)?;
        self.coverage_guard.pinned_pages.insert(page);
        Ok(true)
    }

    fn synchronize_coverage_guard_pins(&mut self) -> Result<bool, LodRuntimeError> {
        if !self.coverage_guard.is_active() {
            return Ok(false);
        }
        let resident = self
            .coverage_guard
            .pages
            .iter()
            .filter(|page| self.cache.contains(**page) && self.decoded_pages.contains_key(*page))
            .copied()
            .collect::<Vec<_>>();
        let mut changed = false;
        for page in resident {
            changed |= self.pin_coverage_guard_page(page)?;
        }
        Ok(changed)
    }

    /// Enqueues without allowing queued, transport/preprocessing in-flight,
    /// and capacity-blocked state to exceed the single configured
    /// pending-request budget. Existing queued pages may still be promoted
    /// without increasing aggregate state.
    fn enqueue_pending_request(&mut self, request: PageRequest) -> RequestEnqueue {
        if self.pending_request_count() < self.pending_request_capacity
            || self.queue.contains(request.page_id)
        {
            self.queue.enqueue(request)
        } else {
            RequestEnqueue::Rejected
        }
    }

    fn update_frontier_pins(
        &mut self,
        view_id: LodRuntimeViewId,
        frontier: &LodFrontier<LodNodeId>,
    ) -> Result<bool, LodRuntimeError> {
        #[cfg(test)]
        {
            self.frontier_pin_rebuilds = self.frontier_pin_rebuilds.saturating_add(1);
        }
        let selected = frontier
            .nodes
            .iter()
            .filter_map(|node| self.hierarchy.page(*node))
            .collect::<BTreeSet<_>>();
        self.views.entry(view_id).or_default().selected_frontier = selected;
        self.synchronize_view_pins(view_id)
    }

    fn synchronize_view_pins(
        &mut self,
        view_id: LodRuntimeViewId,
    ) -> Result<bool, LodRuntimeError> {
        let next = self
            .views
            .get(&view_id)
            .map(|state| state.selected_frontier.clone())
            .unwrap_or_default();
        let previous = self
            .views
            .get(&view_id)
            .map(|state| state.pinned_frontier.clone())
            .unwrap_or_default();
        let released_capacity = previous.difference(&next).next().is_some();
        for &page in next.difference(&previous) {
            self.cache
                .pin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
        }
        for &page in previous.difference(&next) {
            self.cache
                .unpin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
        }
        self.views.entry(view_id).or_default().pinned_frontier = next;
        if released_capacity {
            self.split_cohort_capacity_stall = None;
        }
        Ok(released_capacity)
    }

    fn physical_range_for_node(&self, node: LodNodeId) -> Option<LodPhysicalRange> {
        let representation = self.hierarchy.representation(node)?;
        let resident = self.cache.get(representation.page).copied()?;
        let decoded = self.decoded_pages.get(&representation.page)?;
        validate_page_range(representation, decoded).ok()?;
        let range_end = representation.offset.checked_add(representation.count)?;
        if range_end > self.atlas_layout.gaussians_per_slot {
            return None;
        }
        let physical_start = self
            .atlas_layout
            .physical_index(resident.slot, representation.offset)
            .ok()?;
        Some(LodPhysicalRange {
            node,
            page: representation.page,
            slot: resident.slot,
            physical_start,
            count: representation.count,
        })
    }

    fn view_blend_edge_for_substitution(
        &self,
        substitution: &LodTemporalSubstitution<LodNodeId>,
        initial_weight_bits: u32,
        activation_requires_slew: bool,
    ) -> Option<LodViewBlendEdge> {
        if !matches!(initial_weight_bits, bits if bits == 0.0_f32.to_bits() || bits == 1.0_f32.to_bits())
        {
            return None;
        }
        let parent = substitution.key.parent;
        let child_nodes = match substitution.key.direction {
            crate::stream::hierarchy::LodTemporalDirection::Refine => {
                substitution.next_nodes.as_slice()
            }
            crate::stream::hierarchy::LodTemporalDirection::Coarsen => {
                substitution.previous_nodes.as_slice()
            }
        };
        if child_nodes != self.hierarchy.children(parent) {
            return None;
        }
        if child_nodes.is_empty() {
            return None;
        }
        let parent_metrics = self.hierarchy.metrics(parent)?;
        let child_metrics = child_nodes
            .iter()
            .copied()
            .map(|child| {
                Some(LodViewBlendMetric::from_node(
                    self.hierarchy.metrics(child)?,
                    self.hierarchy.children(child).is_empty(),
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(LodViewBlendEdge {
            parent,
            children: child_nodes.to_vec(),
            parent_metric: LodViewBlendMetric::from_node(parent_metrics, false),
            child_metrics,
            initial_weight_bits,
            activation_requires_slew,
        })
    }

    fn direct_refine_substitution(
        &self,
        parent: LodNodeId,
    ) -> Result<LodTemporalSubstitution<LodNodeId>, LodRuntimeError> {
        let parent_metrics = self
            .hierarchy
            .metrics(parent)
            .ok_or(LodRuntimeError::Selection(LodSelectionError::MissingNode(
                parent,
            )))?;
        if !parent_metrics.validate() {
            return Err(LodRuntimeError::Selection(LodSelectionError::InvalidNode(
                parent,
            )));
        }
        let children = self.hierarchy.children(parent).to_vec();
        if children.is_empty() {
            return Err(LodRuntimeError::Selection(LodSelectionError::InvalidNode(
                parent,
            )));
        }
        let next_gaussians = children.iter().try_fold(0_u64, |count, child| {
            let metrics = self
                .hierarchy
                .metrics(*child)
                .ok_or(LodRuntimeError::Selection(LodSelectionError::MissingNode(
                    *child,
                )))?;
            if !metrics.validate() {
                return Err(LodRuntimeError::Selection(LodSelectionError::InvalidNode(
                    *child,
                )));
            }
            count
                .checked_add(u64::from(metrics.representative_count))
                .ok_or(LodRuntimeError::Selection(LodSelectionError::CountOverflow))
        })?;
        Ok(LodTemporalSubstitution {
            key: LodTemporalSubstitutionKey {
                parent,
                direction: LodTemporalDirection::Refine,
            },
            previous_nodes: vec![parent],
            next_nodes: children,
            previous_gaussians: u64::from(parent_metrics.representative_count),
            next_gaussians,
        })
    }

    /// Rebuilds every direct parent/complete-child boundary represented by an
    /// exact child-side frontier and currently inside its camera-conditioned
    /// pressure band. These persistent edges are presentation state, not a
    /// timed topology wave: all disjoint branches coexist in one table.
    fn stable_view_blend_substitutions(
        &self,
        frontier: &[LodNodeId],
        view: LodView,
        target: LodQualityTarget,
    ) -> Result<Vec<LodTemporalSubstitution<LodNodeId>>, LodRuntimeError> {
        if self.hierarchy.manifest().morph_map.is_none() {
            return Ok(Vec::new());
        }
        let selected = frontier.iter().copied().collect::<BTreeSet<_>>();
        let parents = frontier
            .iter()
            .filter_map(|node| self.hierarchy.parent(*node))
            .collect::<BTreeSet<_>>();
        let mut substitutions = Vec::new();
        for parent in parents {
            let children = self.hierarchy.children(parent);
            if children.is_empty() || !children.iter().all(|child| selected.contains(child)) {
                continue;
            }
            let substitution = self.direct_refine_substitution(parent)?;
            let edge = self
                .view_blend_edge_for_substitution(&substitution, 1.0_f32.to_bits(), false)
                .ok_or(LodRuntimeError::Selection(LodSelectionError::InvalidNode(
                    parent,
                )))?;
            let weight = lod_view_blend_weight(view, target, &edge);
            if weight > 0.0 && weight < 1.0 {
                substitutions.push(substitution);
            }
        }
        Ok(substitutions)
    }

    fn merge_view_blend_substitutions(
        &self,
        applied: &[LodTemporalSubstitution<LodNodeId>],
        frontier: &[LodNodeId],
        view: LodView,
        target: LodQualityTarget,
    ) -> Result<Vec<LodTemporalSubstitution<LodNodeId>>, LodRuntimeError> {
        let stable = self.stable_view_blend_substitutions(frontier, view, target)?;
        Ok(merge_disjoint_view_blend_substitutions(stable, applied))
    }

    fn update_predictive_view_blend_demand(
        &mut self,
        view_id: LodRuntimeViewId,
        frontier: &[LodNodeId],
        view: LodView,
        target: LodQualityTarget,
    ) -> Result<bool, LodRuntimeError> {
        if self.hierarchy.manifest().morph_map.is_none() {
            return self.clear_predictive_view_blend_demand(view_id);
        }

        let selected = frontier.iter().copied().collect::<BTreeSet<_>>();
        let previous = self
            .views
            .get(&view_id)
            .map(|state| state.predictive_view_blend_nodes.clone())
            .unwrap_or_default();
        let mut next = BTreeMap::new();

        // Keep an already prepared cohort through the complete blend band. It
        // is released only well outside on the parent side, or once the child
        // side is detailed enough that the next hierarchy edge owns demand.
        for (parent, children) in previous {
            if self.predictive_view_blend_cohort_has_terminal_member(&children) {
                continue;
            }
            let substitution = self.direct_refine_substitution(parent)?;
            let initial_weight_bits = if selected.contains(&parent) {
                0.0_f32.to_bits()
            } else {
                1.0_f32.to_bits()
            };
            let Some(edge) =
                self.view_blend_edge_for_substitution(&substitution, initial_weight_bits, false)
            else {
                continue;
            };
            let Some((parent_pressure, child_pressure)) =
                lod_view_blend_pressures(view, target, &edge)
            else {
                continue;
            };
            if parent_pressure > LOD_VIEW_BLEND_RELEASE_PARENT_PRESSURE
                && child_pressure < LOD_VIEW_BLEND_RELEASE_CHILD_PRESSURE
            {
                next.insert(parent, children);
            }
        }

        for parent in selected.iter().copied() {
            if self.hierarchy.children(parent).is_empty() {
                continue;
            }
            let substitution = self.direct_refine_substitution(parent)?;
            if self.predictive_view_blend_cohort_has_terminal_member(&substitution.next_nodes) {
                continue;
            }
            let Some(edge) =
                self.view_blend_edge_for_substitution(&substitution, 0.0_f32.to_bits(), false)
            else {
                continue;
            };
            let Some((parent_pressure, child_pressure)) =
                lod_view_blend_pressures(view, target, &edge)
            else {
                continue;
            };
            if parent_pressure >= LOD_VIEW_BLEND_PREFETCH_PARENT_PRESSURE
                && child_pressure < LOD_VIEW_BLEND_RELEASE_CHILD_PRESSURE
            {
                let mut pages = next
                    .values()
                    .flatten()
                    .chain(substitution.next_nodes.iter())
                    .filter_map(|node| self.hierarchy.page(*node))
                    .collect::<BTreeSet<_>>();
                pages.retain(|page| self.page_reserves_streaming_capacity(*page));
                let plan = LodSplitCohortPlan {
                    view: view_id,
                    parent,
                    pages,
                };
                if self.split_cohort_capacity(&plan)?.0 {
                    next.insert(parent, substitution.next_nodes);
                }
            }
        }

        let next_pages = next
            .values()
            .flatten()
            .filter_map(|node| self.hierarchy.page(*node))
            .filter(|page| self.cache.contains(*page))
            .collect::<BTreeSet<_>>();
        let previous_pins = self
            .views
            .get(&view_id)
            .map(|state| state.pinned_predictive_pages.clone())
            .unwrap_or_default();
        let released_capacity = previous_pins.difference(&next_pages).next().is_some();
        for &page in next_pages.difference(&previous_pins) {
            self.cache
                .pin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
        }
        for &page in previous_pins.difference(&next_pages) {
            self.cache
                .unpin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
        }
        let state = self.views.entry(view_id).or_default();
        state.predictive_view_blend_nodes = next;
        state.pinned_predictive_pages = next_pages;
        Ok(released_capacity)
    }

    fn predictive_view_blend_cohort_has_terminal_member(&self, children: &[LodNodeId]) -> bool {
        children.iter().any(|child| {
            self.hierarchy.page(*child).is_some_and(|page| {
                self.terminal_failures.contains(&page)
                    || self
                        .speculative_prefetch_terminal_requests
                        .contains_key(&page)
            })
        })
    }

    /// Removes a predictive parent and all of its siblings as one unit after
    /// any member exhausts retry. Resident siblings are unpinned immediately;
    /// queued/in-flight siblings disappear during ordinary end-of-frame demand
    /// reconciliation because the complete cohort is no longer admitted.
    fn drop_predictive_view_blend_cohorts_with_terminal_members(
        &mut self,
    ) -> Result<bool, LodRuntimeError> {
        let view_ids = self.views.keys().copied().collect::<Vec<_>>();
        let mut released_capacity = false;
        for view_id in view_ids {
            let (mut next, previous_pins) = self
                .views
                .get(&view_id)
                .map(|state| {
                    (
                        state.predictive_view_blend_nodes.clone(),
                        state.pinned_predictive_pages.clone(),
                    )
                })
                .unwrap_or_default();
            next.retain(|_, children| {
                !self.predictive_view_blend_cohort_has_terminal_member(children)
            });
            let retained_pages = next
                .values()
                .flatten()
                .filter_map(|node| self.hierarchy.page(*node))
                .collect::<BTreeSet<_>>();
            let next_pins = previous_pins
                .intersection(&retained_pages)
                .copied()
                .collect::<BTreeSet<_>>();
            for &page in previous_pins.difference(&next_pins) {
                self.cache
                    .unpin_fallback(page)
                    .map_err(LodRuntimeError::Cache)?;
                released_capacity = true;
            }
            let state = self.views.entry(view_id).or_default();
            state.predictive_view_blend_nodes = next;
            state.pinned_predictive_pages = next_pins;
        }
        Ok(released_capacity)
    }

    fn clear_predictive_view_blend_demand(
        &mut self,
        view_id: LodRuntimeViewId,
    ) -> Result<bool, LodRuntimeError> {
        let pinned = self
            .views
            .get_mut(&view_id)
            .map(|state| {
                state.predictive_view_blend_nodes.clear();
                std::mem::take(&mut state.pinned_predictive_pages)
            })
            .unwrap_or_default();
        let released_capacity = !pinned.is_empty();
        for page in pinned {
            self.cache
                .unpin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
        }
        Ok(released_capacity)
    }

    fn enqueue_predictive_view_blend_demand(
        &mut self,
        view_id: LodRuntimeViewId,
        view: LodView,
    ) -> Result<(), LodRuntimeError> {
        let predictive = self
            .views
            .get(&view_id)
            .map(|state| {
                state
                    .predictive_view_blend_nodes
                    .iter()
                    .flat_map(|(parent, children)| {
                        children.iter().copied().map(|child| (*parent, child))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for (parent, child) in predictive {
            self.enqueue_predictive_view_blend_node(parent, child, view)?;
        }
        Ok(())
    }

    fn enqueue_predictive_view_blend_node(
        &mut self,
        parent: LodNodeId,
        child: LodNodeId,
        view: LodView,
    ) -> Result<(), LodRuntimeError> {
        let page_id = self
            .hierarchy
            .page(child)
            .ok_or(LodRuntimeError::MissingNode(child))?;
        if self.cache.contains(page_id)
            || self.in_flight.contains_key(&page_id)
            || self.preprocessor.contains(page_id)
            || self.queue.contains(page_id)
            || self.terminal_failures.contains(&page_id)
            || self
                .speculative_prefetch_terminal_requests
                .contains_key(&page_id)
            || self.capacity_blocked.contains_key(&page_id)
        {
            return Ok(());
        }
        let descriptor = self
            .hierarchy
            .page_descriptor(page_id)
            .ok_or(LodRuntimeError::MissingPageDescriptor(page_id))?;
        let metrics = self
            .hierarchy
            .metrics(child)
            .ok_or(LodRuntimeError::MissingNode(child))?;
        let distance = view.distance_to_center(metrics);
        let urgency = if distance <= 0.0 {
            u32::MAX
        } else {
            (1_000_000.0 / distance).clamp(0.0, u32::MAX as f32) as u32
        };
        let mut request = PageRequest::new(page_id, PageRequestPriority::prefetch(urgency));
        request.expected_bytes = descriptor
            .storage
            .as_ref()
            .map(|storage| storage.encoded_len);
        request.fallback_page = self.hierarchy.page(parent);
        let _ = self.enqueue_pending_request(request);
        Ok(())
    }

    fn temporal_morph_identity(
        &self,
        view: LodRuntimeViewId,
        _target_ranges: &[LodPhysicalRange],
        transition: &LodTemporalTransition,
    ) -> Option<LodTemporalMorphIdentity> {
        self.hierarchy.manifest().morph_map.as_ref()?;
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
        macro_rules! write_range {
            ($range:expr) => {{
                let range = $range;
                write(range.node.0);
                write(range.page.0);
                write(u64::from(range.slot.index));
                write(u64::from(range.slot.generation));
                write(u64::from(range.physical_start));
                write(u64::from(range.count));
            }};
        }
        write(transition.substitutions().len() as u64);
        let mut descriptor_count = 0_u32;
        let mut mapping_record_count = 0_u32;
        if transition.initial_weight_bits.len() != transition.substitutions().len() {
            return None;
        }
        for (edge_index, substitution) in transition.substitutions().iter().enumerate() {
            let parent = substitution.key.parent;
            let activation_requires_slew = self
                .views
                .get(&view)
                .is_some_and(|state| state.late_view_blend_edges.contains(&parent));
            let edge = self.view_blend_edge_for_substitution(
                substitution,
                transition.initial_weight_bits(edge_index)?,
                activation_requires_slew,
            )?;
            let node_index = self.hierarchy.node_index(parent)?;
            let run_range = self
                .hierarchy
                .manifest()
                .morph_child_run_range_at(node_index)?;
            let parent_range = self.physical_range_for_node(parent)?;
            write(parent.0);
            write(u64::from(run_range.start));
            write(u64::from(run_range.count));
            write(u64::from(edge.initial_weight_bits()));
            write(u64::from(u8::from(edge.activation_requires_slew())));
            write(edge.children().len() as u64);
            for child in edge.children() {
                write(child.0);
            }
            macro_rules! write_metric {
                ($metric:expr) => {{
                    let metric = $metric;
                    for bits in metric.center_bits {
                        write(u64::from(bits));
                    }
                    write(u64::from(metric.radius_bits));
                    write(u64::from(metric.geometric_error_bits));
                    write(u64::from(metric.quality_min_bits));
                    write(u64::from(metric.quality_max_bits));
                    write(u64::from(metric.high_fidelity_certificate_bits));
                    write(u64::from(u8::from(metric.original_representation)));
                }};
            }
            write_metric!(edge.parent_metric());
            write(edge.child_metrics().len() as u64);
            for metric in edge.child_metrics().iter().copied() {
                write_metric!(metric);
            }
            write_range!(parent_range);
            let child_nodes = match substitution.key.direction {
                crate::stream::hierarchy::LodTemporalDirection::Refine => {
                    substitution.next_nodes.as_slice()
                }
                crate::stream::hierarchy::LodTemporalDirection::Coarsen => {
                    substitution.previous_nodes.as_slice()
                }
            };
            if child_nodes != self.hierarchy.children(parent) {
                return None;
            }
            descriptor_count =
                descriptor_count.checked_add(u32::try_from(child_nodes.len()).ok()?)?;
            for child in child_nodes.iter().copied() {
                let range = self.physical_range_for_node(child)?;
                mapping_record_count = mapping_record_count.checked_add(range.count)?;
                write_range!(range);
            }
        }
        Some(LodTemporalMorphIdentity {
            primary,
            secondary,
            descriptor_count,
            mapping_record_count,
        })
    }

    fn prepare_temporal_morph_batch(
        &mut self,
        view: LodRuntimeViewId,
        pressure_view: LodView,
        target: LodQualityTarget,
        target_ranges: &[LodPhysicalRange],
        transition: &LodTemporalTransition,
    ) -> Option<Arc<LodTemporalMorphBatch>> {
        let identity = self.temporal_morph_identity(view, target_ranges, transition)?;
        if let Some(batch) = self
            .views
            .get(&view)
            .and_then(|state| state.temporal_morph_cache.as_ref())
            .filter(|cached| cached.identity == identity)
            .map(|cached| Arc::clone(&cached.batch))
        {
            return lod_view_blend_batch_pressures_are_valid(pressure_view, target, batch.edges())
                .then_some(batch);
        }
        let batch =
            self.build_temporal_morph_batch_uncached(view, identity, target_ranges, transition)?;
        let batch = Arc::new(batch);
        if !lod_view_blend_batch_pressures_are_valid(pressure_view, target, batch.edges()) {
            // A retained endpoint remains the only safe visual fallback for a
            // malformed or threshold-contradictory edge. Do not advertise that
            // endpoint as a converged camera-conditioned blend: author this
            // transaction as the explicit bounded hard fallback instead.
            return None;
        }
        #[cfg(test)]
        {
            self.temporal_morph_batch_builds = self.temporal_morph_batch_builds.saturating_add(1);
        }
        self.views.entry(view).or_default().temporal_morph_cache = Some(LodTemporalMorphCache {
            identity,
            batch: Arc::clone(&batch),
        });
        Some(batch)
    }

    /// Builds the compact direct parent map only for an ABI-authored morph
    /// cohort. Any missing semantic proof fails closed to the bounded complete
    /// cut fallback; renderer code never guesses correspondence from geometry.
    fn build_temporal_morph_batch_uncached(
        &self,
        view: LodRuntimeViewId,
        identity: LodTemporalMorphIdentity,
        target_ranges: &[LodPhysicalRange],
        transition: &LodTemporalTransition,
    ) -> Option<LodTemporalMorphBatch> {
        self.hierarchy.manifest().morph_map.as_ref()?;

        let mut presentation = target_ranges
            .iter()
            .copied()
            .map(|range| (range.node, range))
            .collect::<BTreeMap<_, _>>();
        let mut required = target_ranges.to_vec();
        let mut edges = Vec::with_capacity(transition.substitutions().len());
        let mut descriptor_records =
            Vec::<(LodTemporalMorphDescriptor, Vec<LodTemporalMorphRecord>)>::new();

        if transition.initial_weight_bits.len() != transition.substitutions().len() {
            return None;
        }
        for (transition_edge_index, substitution) in transition.substitutions().iter().enumerate() {
            let parent = substitution.key.parent;
            let edge_index = u32::try_from(edges.len()).ok()?;
            let activation_requires_slew = self
                .views
                .get(&view)
                .is_some_and(|state| state.late_view_blend_edges.contains(&parent));
            let edge = self.view_blend_edge_for_substitution(
                substitution,
                transition.initial_weight_bits(transition_edge_index)?,
                activation_requires_slew,
            )?;
            let node_index = self.hierarchy.node_index(parent)?;
            let run_lengths = self
                .hierarchy
                .manifest()
                .morph_child_run_lengths_at(node_index)?;
            let parent_range = self.physical_range_for_node(parent)?;
            if parent_range.count as usize != run_lengths.len() {
                return None;
            }
            let child_nodes = match substitution.key.direction {
                crate::stream::hierarchy::LodTemporalDirection::Refine => {
                    substitution.next_nodes.as_slice()
                }
                crate::stream::hierarchy::LodTemporalDirection::Coarsen => {
                    substitution.previous_nodes.as_slice()
                }
            };
            if child_nodes != self.hierarchy.children(parent) {
                return None;
            }
            let child_ranges = child_nodes
                .iter()
                .copied()
                .map(|node| self.physical_range_for_node(node))
                .collect::<Option<Vec<_>>>()?;
            let child_count = child_ranges
                .iter()
                .try_fold(0_u32, |total, range| total.checked_add(range.count))?;
            let mapped_child_count = run_lengths
                .iter()
                .try_fold(0_u32, |total, run| total.checked_add(u32::from(*run)))?;
            if child_count != mapped_child_count {
                return None;
            }

            let mut expanded = Vec::with_capacity(child_count as usize);
            for (parent_local, run_length) in run_lengths.iter().copied().enumerate() {
                let parent_local = u32::try_from(parent_local).ok()?;
                let parent_physical_index =
                    parent_range.physical_start.checked_add(parent_local)?;
                expanded.extend(std::iter::repeat_n(
                    LodTemporalMorphRecord {
                        parent_physical_index,
                        split_count: u32::from(run_length),
                    },
                    usize::from(run_length),
                ));
            }
            debug_assert_eq!(expanded.len(), child_count as usize);
            let mut child_offset = 0_usize;
            for child_range in child_ranges.iter().copied() {
                let child_end = child_offset.checked_add(child_range.count as usize)?;
                descriptor_records.push((
                    LodTemporalMorphDescriptor {
                        child_physical_start: child_range.physical_start,
                        child_count: child_range.count,
                        mapping_start: 0,
                        edge_index,
                    },
                    expanded.get(child_offset..child_end)?.to_vec(),
                ));
                child_offset = child_end;
            }
            if child_offset != expanded.len() {
                return None;
            }

            if substitution.key.direction == crate::stream::hierarchy::LodTemporalDirection::Coarsen
            {
                presentation.remove(&parent)?;
                for range in child_ranges.iter().copied() {
                    if presentation.insert(range.node, range).is_some() {
                        return None;
                    }
                }
            }
            required.push(parent_range);
            required.extend(child_ranges);
            edges.push(edge);
        }

        // Physical-source lookup uses binary search, independent of logical
        // hierarchy order. Rebase each direct-record slice after sorting.
        descriptor_records.sort_unstable_by_key(|(descriptor, _)| {
            (descriptor.child_physical_start, descriptor.child_count)
        });
        let mut descriptors = Vec::with_capacity(descriptor_records.len());
        let mut records = Vec::new();
        for (mut descriptor, descriptor_mapping) in descriptor_records {
            descriptor.mapping_start = u32::try_from(records.len()).ok()?;
            records.extend(descriptor_mapping);
            descriptors.push(descriptor);
        }
        for pair in descriptors.windows(2) {
            if pair[0]
                .child_physical_start
                .checked_add(pair[0].child_count)?
                > pair[1].child_physical_start
            {
                return None;
            }
        }

        let presentation_ranges =
            manifest_ordered_morph_presentation_ranges(&self.hierarchy, presentation)?;
        required.extend(presentation_ranges.iter().copied());
        required.sort_unstable_by_key(|range| {
            (
                range.slot.index,
                range.slot.generation,
                range.physical_start,
                range.count,
                range.node,
            )
        });
        required.dedup();

        if !view_blend_batch_structure_is_valid(&edges, &descriptors) {
            return None;
        }

        Some(LodTemporalMorphBatch {
            identity,
            presentation_ranges,
            required_ranges: required,
            edges,
            descriptors,
            records,
        })
    }

    fn physical_ranges(
        &mut self,
        frontier: &LodFrontier<LodNodeId>,
    ) -> Result<Vec<LodPhysicalRange>, LodRuntimeError> {
        #[cfg(test)]
        {
            self.physical_range_rebuilds = self.physical_range_rebuilds.saturating_add(1);
        }
        // ABI16's exact child endpoint must consume the same stable equal-key
        // source order before and after a Morphing table retires. Legacy
        // categorical packages retain their historical node-ID ordering.
        let presentation_nodes = if self.hierarchy.manifest().morph_map.is_some() {
            manifest_ordered_presentation_nodes(&self.hierarchy, &frontier.nodes)
                .map_err(LodRuntimeError::MissingNode)?
        } else {
            frontier.nodes.clone()
        };
        let mut ranges = Vec::with_capacity(presentation_nodes.len());
        for node in presentation_nodes {
            let representation = self
                .hierarchy
                .representation(node)
                .ok_or(LodRuntimeError::MissingNode(node))?;
            let range = self.physical_range_for_node(node).ok_or_else(|| {
                if !self.cache.contains(representation.page) {
                    LodRuntimeError::SelectedPageNotResident(representation.page)
                } else {
                    LodRuntimeError::SelectedPageNotDecoded(representation.page)
                }
            })?;
            self.cache.touch(range.page, self.epoch);
            ranges.push(range);
        }
        Ok(ranges)
    }

    /// Fast path for a request wave proven to fit alongside current residency
    /// without eviction. Once the aggregate wave crosses any resident limit,
    /// `observe_split_cohort_candidates` gates it through one pinned cohort.
    fn enqueue_missing(
        &mut self,
        frontier: &LodFrontier<LodNodeId>,
        view: LodView,
    ) -> Result<(), LodRuntimeError> {
        for &node in &frontier.requested_nodes {
            self.enqueue_missing_node(node, view)?;
        }
        Ok(())
    }

    fn enqueue_missing_roots(
        &mut self,
        frontier: &LodFrontier<LodNodeId>,
        view: LodView,
    ) -> Result<(), LodRuntimeError> {
        for &node in &frontier.requested_nodes {
            if self.hierarchy.parent(node).is_none() {
                self.enqueue_missing_node(node, view)?;
            }
        }
        Ok(())
    }

    fn enqueue_missing_node(
        &mut self,
        node: LodNodeId,
        view: LodView,
    ) -> Result<(), LodRuntimeError> {
        let page_id = self
            .hierarchy
            .page(node)
            .ok_or(LodRuntimeError::MissingNode(node))?;
        if self.cache.contains(page_id)
            || self.in_flight.contains_key(&page_id)
            || self.preprocessor.contains(page_id)
            || self.queue.contains(page_id)
            || self.terminal_failures.contains(&page_id)
            || self.capacity_blocked.contains_key(&page_id)
        {
            return Ok(());
        }
        let descriptor = self
            .hierarchy
            .page_descriptor(page_id)
            .ok_or(LodRuntimeError::MissingPageDescriptor(page_id))?;
        let manifest_node = self
            .hierarchy
            .node(node)
            .ok_or(LodRuntimeError::MissingNode(node))?;
        let metrics = self
            .hierarchy
            .metrics(node)
            .ok_or(LodRuntimeError::MissingNode(node))?;
        let distance = view.distance_to_center(metrics);
        let urgency = if distance <= 0.0 {
            u32::MAX
        } else {
            (1_000_000.0 / distance).clamp(0.0, u32::MAX as f32) as u32
        };
        let mut request = PageRequest::new(
            page_id,
            if manifest_node.parent.is_none() {
                PageRequestPriority::fallback_critical(urgency)
            } else {
                PageRequestPriority::visible(urgency)
            },
        );
        request.expected_bytes = descriptor
            .storage
            .as_ref()
            .map(|storage| storage.encoded_len);
        let _ = self.enqueue_pending_request(request);
        Ok(())
    }

    fn start_requests(
        &mut self,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
        failed_pages: &mut Vec<LodPageId>,
    ) -> Vec<LodPageId> {
        let concurrency = streaming_settings.max_concurrent_requests as usize;
        let available = concurrency.saturating_sub(self.in_flight.len());
        let frame_limit = lod_settings
            .budgets
            .max_requests_per_frame
            .saturating_sub(self.frame_request_starts) as usize;
        let attempt_limit = available.min(frame_limit);
        let scan_limit = self.queue.len();
        let mut started = Vec::new();
        let mut attempted = 0;
        let mut deferred = Vec::new();
        for _ in 0..scan_limit {
            if attempted >= attempt_limit {
                break;
            }
            let Some(request) = self.queue.pop() else {
                break;
            };
            if self.terminal_failures.contains(&request.page_id) {
                continue;
            }
            if self
                .preprocess_retry_deferred_frame
                .get(&request.page_id)
                .is_some_and(|deferred_frame| deferred_frame.0 == self.epoch)
            {
                deferred.push(request);
                continue;
            }
            self.preprocess_retry_deferred_frame
                .remove(&request.page_id);
            attempted += 1;
            let attempts = self.attempts.entry(request.page_id).or_default();
            *attempts = attempts.saturating_add(1);
            self.frame_request_starts = self.frame_request_starts.saturating_add(1);
            #[cfg(test)]
            {
                self.transport_request_starts = self.transport_request_starts.saturating_add(1);
            }
            match self.transport.begin(request) {
                Ok(ticket) => {
                    self.in_flight
                        .insert(request.page_id, InFlight { ticket, request });
                    started.push(request.page_id);
                }
                Err(error) => {
                    self.transport_failures
                        .insert(request.page_id, T::classify_error(&error));
                    self.retry_or_fail(request, streaming_settings.retry_limit, failed_pages);
                }
            }
        }
        for request in deferred {
            let outcome = self.queue.enqueue(request);
            debug_assert!(
                !matches!(outcome, RequestEnqueue::Rejected),
                "a deferred preprocessing retry came from this bounded queue"
            );
        }
        started
    }
}

fn validate_page_range(
    range: LodPageRange,
    page: &PlanarGaussian3dPage,
) -> Result<(), LodRuntimeError> {
    let end = range.end().ok_or(LodRuntimeError::PhysicalIndexOverflow)? as usize;
    if end > page.gaussians.len() {
        Err(LodRuntimeError::PageRangeOutOfBounds {
            page: range.page,
            end: end as u64,
            count: page.gaussians.len() as u64,
        })
    } else {
        Ok(())
    }
}

fn validate_shared_page_node_ranges(
    page: &PlanarGaussian3dPage,
    ranges: &[SharedPageNodeRange],
    support_sigma: f32,
) -> Result<(), LodPagePreprocessError> {
    for entry in ranges {
        let end = entry
            .range
            .end()
            .ok_or(LodPagePreprocessError::PayloadOutsideNodeBounds {
                page: page.id,
                node: entry.node,
            })? as usize;
        let start = entry.range.offset as usize;
        let gaussians = page.gaussians.get(start..end).ok_or(
            LodPagePreprocessError::PayloadOutsideNodeBounds {
                page: page.id,
                node: entry.node,
            },
        )?;
        let mut actual_bounds: Option<LodBounds> = None;
        for gaussian in gaussians {
            let bounds = gaussian_support_bounds(gaussian, support_sigma)
                .map_err(|_| LodPagePreprocessError::InvalidSupportBounds(page.id))?;
            actual_bounds = Some(match actual_bounds {
                Some(current) => current.union(bounds),
                None => bounds,
            });
        }
        let actual_bounds =
            actual_bounds.ok_or(LodPagePreprocessError::PayloadOutsideNodeBounds {
                page: page.id,
                node: entry.node,
            })?;
        let epsilon = 1e-5 * entry.bounds.radius().max(actual_bounds.radius()).max(1.0);
        if !entry.bounds.contains_with_epsilon(&actual_bounds, epsilon) {
            return Err(LodPagePreprocessError::PayloadOutsideNodeBounds {
                page: page.id,
                node: entry.node,
            });
        }
    }
    Ok(())
}

fn page_codec_limits(
    descriptor: &LodPageDescriptor,
    max_encoded_page_bytes: u64,
) -> LodCodecLimits {
    LodCodecLimits {
        max_page_bytes: max_encoded_page_bytes,
        max_page_gaussians: descriptor.gaussian_count,
        ..Default::default()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LodRuntimeError {
    InvalidSettings(String),
    StructuralSettingsChanged(&'static str),
    InvalidManifest(ManifestHierarchyError),
    ManifestHasNoPages,
    ZeroAtlasStride,
    EncodedPageLimitTooSmall {
        limit: u64,
        minimum: u64,
    },
    PageEncodedBytesExceedLimit {
        page: LodPageId,
        actual: u64,
        limit: u64,
    },
    PageDecodedBytesExceedLimit {
        page: LodPageId,
        actual: u64,
        limit: u64,
    },
    PageGaussiansExceedLimit {
        page: LodPageId,
        actual: u64,
        limit: u64,
    },
    CoverageGuardPagesExceedLimit {
        actual: u64,
        limit: u64,
    },
    CoverageGuardBytesExceedLimit {
        actual: u64,
        limit: u64,
    },
    CoverageGuardGaussiansExceedLimit {
        actual: u64,
        limit: u64,
    },
    CoverageGuardActiveGaussiansExceedLimit {
        actual: u64,
        limit: u64,
    },
    InvalidPageSupportBounds {
        page: LodPageId,
    },
    PagePayloadOutsideDescriptor(LodPageId),
    RequestCapacityOverflow,
    PreprocessAdmission(LodPagePreprocessAdmissionError),
    Queue(RequestQueueError),
    Cache(PageCacheError),
    Selection(LodSelectionError<LodNodeId>),
    MissingNode(LodNodeId),
    MissingPageDescriptor(LodPageId),
    MissingTerminalRequest(LodPageId),
    RetryQueueRejected(LodPageId),
    InvalidFrameToken {
        expected: LodRuntimeFrameId,
        actual: LodRuntimeFrameId,
    },
    FrameAlreadyFinished(LodRuntimeFrameId),
    SelectedPageNotResident(LodPageId),
    SelectedPageNotDecoded(LodPageId),
    PageRangeOutOfBounds {
        page: LodPageId,
        end: u64,
        count: u64,
    },
    PageRangeExceedsAtlasStride {
        offset: u32,
        count: u32,
        stride: u32,
    },
    PhysicalIndexOverflow,
    AtlasAddressSpaceOverflow {
        slots: u32,
        stride: u32,
    },
    CandidateCountMismatch {
        frontier: u64,
        physical: u64,
    },
    NoResidentFrontier,
    OverlappingPhysicalRanges {
        previous_end: u32,
        next_start: u32,
    },
    CandidateExpansionLimit {
        count: u64,
        limit: u32,
    },
}

impl fmt::Display for LodRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LodRuntimeError {}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use std::sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    };
    use std::{hint::black_box, time::Instant};

    use super::*;
    use crate::{
        gaussian::formats::{
            planar_3d::{Gaussian3d, PlanarGaussian3d, gaussian_3d_gpu_bytes_per_record},
            planar_3d_chunked::{
                LOD_PAGE_SCHEMA_VERSION, LodBounds, LodIndexRange, LodPageEncoding, LodPageKind,
                LodPageStorage, LodSourceRange,
            },
            planar_3d_lod::{
                EXTERNAL_MOMENT_MERGE_VERSION, GaussianLodBuildMetadata, GaussianLodBuildSettings,
                GaussianLodManifestHeader, GaussianLodNode, GaussianLodQualityMetadata,
                LOD_CURRENT_REQUIRED_FEATURES, LOD_MANIFEST_MAGIC, LOD_MANIFEST_VERSION, LodError,
                LodMortonRange, LodQualityInterval, LodReducerKind, build_planar_3d_lod,
                lod_config_fingerprint_for_reducer,
            },
        },
        io::lod::{LodCodecError, decode_page, encode_page},
        material::spherical_harmonics::SphericalHarmonicCoefficients,
        stream::{
            hierarchy::{AllResident, LodTemporalDirection},
            transport::{MemoryPageTransport, MemoryTransportError, PagePayload},
        },
        testing::{
            LodTestScene, VirtualCityScene, upgrade_manifest_to_synthetic_abi16_lifecycle_fixture,
        },
    };

    #[cfg(not(target_arch = "wasm32"))]
    use crate::stream::preprocess::LodPagePreprocessBackend;

    const VIRTUAL_TREE_DEPTH: u16 = 3;
    const VIRTUAL_BRANCHING_FACTOR: u32 = 32;
    // The virtual fixture uses a wide, non-progressive topology like the
    // supported external CPU package builder.
    const VIRTUAL_BUILDER_ABI_VERSION: u32 = 5;
    const VIRTUAL_LEVEL_STARTS: [u32; 4] = [0, 1, 33, 1_057];
    const VIRTUAL_LEVEL_COUNTS: [u32; 4] = [1, 32, 1_024, 32_768];
    const VIRTUAL_NODE_COUNT: u32 = 33_825;

    fn view_blend_test_metric(quality_threshold: f32, geometric_error: f32) -> LodViewBlendMetric {
        LodViewBlendMetric::from_node(
            LodNodeMetrics {
                center: bevy::math::Vec3::ZERO,
                radius: 50.0,
                geometric_error,
                appearance_error: 0.0,
                opacity_error: 0.0,
                quality_min: quality_threshold,
                quality_max: quality_threshold,
                high_fidelity_certificate: 1.0,
                representative_count: 1,
            },
            false,
        )
    }

    fn view_blend_test_edge(
        parent_metric: LodViewBlendMetric,
        child_metrics: Vec<LodViewBlendMetric>,
        initial_weight: f32,
    ) -> LodViewBlendEdge {
        LodViewBlendEdge {
            parent: LodNodeId(1),
            children: (0..child_metrics.len())
                .map(|index| LodNodeId(index as u64 + 2))
                .collect(),
            parent_metric,
            child_metrics,
            initial_weight_bits: initial_weight.to_bits(),
            activation_requires_slew: false,
        }
    }

    fn view_blend_test_view() -> LodView {
        LodView::orthographic(bevy::math::Vec3::ZERO, 100.0, 100.0, 0.1)
    }

    fn view_blend_test_target(detail_fraction: f32) -> LodQualityTarget {
        LodQualityTarget::Balanced {
            detail_fraction,
            max_error_px: 100.0,
        }
    }

    fn view_blend_test_substitution(
        parent: u64,
        direction: LodTemporalDirection,
        previous_nodes: &[u64],
        next_nodes: &[u64],
    ) -> LodTemporalSubstitution<LodNodeId> {
        LodTemporalSubstitution {
            key: LodTemporalSubstitutionKey {
                parent: LodNodeId(parent),
                direction,
            },
            previous_nodes: previous_nodes.iter().copied().map(LodNodeId).collect(),
            next_nodes: next_nodes.iter().copied().map(LodNodeId).collect(),
            previous_gaussians: previous_nodes.len() as u64,
            next_gaussians: next_nodes.len() as u64,
        }
    }

    #[test]
    fn view_blend_weight_meets_selector_endpoints_and_interior_boundary() {
        let view = view_blend_test_view();
        let edge = view_blend_test_edge(
            view_blend_test_metric(0.25, 200.0),
            vec![view_blend_test_metric(0.75, 100.0)],
            0.0,
        );

        assert_eq!(
            lod_view_blend_weight(view, view_blend_test_target(0.2), &edge).to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(
            lod_view_blend_weight(view, view_blend_test_target(0.75), &edge).to_bits(),
            1.0_f32.to_bits()
        );

        let target = view_blend_test_target(0.5);
        let (parent_pressure, child_pressure) =
            lod_view_blend_pressures(view, target, &edge).expect("valid adjacent edge");
        let weight = lod_view_blend_weight(view, target, &edge);
        assert!(weight > 0.0 && weight < 1.0);
        let interpolated = (1.0 - weight) * parent_pressure + weight * child_pressure;
        assert!((interpolated - 1.0).abs() <= 4.0 * f32::EPSILON);
    }

    #[test]
    fn view_blend_weight_is_path_independent_and_reverses_exactly() {
        let view = view_blend_test_view();
        let target = view_blend_test_target(0.5);
        let edge = view_blend_test_edge(
            view_blend_test_metric(0.25, 200.0),
            vec![view_blend_test_metric(0.75, 100.0)],
            0.0,
        );
        let outward = lod_view_blend_weight(view, target, &edge);
        let _crossed = lod_view_blend_weight(view, view_blend_test_target(0.7), &edge);
        let reversed = lod_view_blend_weight(view, target, &edge);
        assert_eq!(outward.to_bits(), reversed.to_bits());

        let mut opposite_retained_endpoint = edge.clone();
        opposite_retained_endpoint.initial_weight_bits = 1.0_f32.to_bits();
        assert_eq!(
            outward.to_bits(),
            lod_view_blend_weight(view, target, &opposite_retained_endpoint).to_bits(),
            "valid view-conditioned weight must not retain approach direction"
        );
    }

    #[test]
    fn view_blend_weight_uses_the_maximum_immediate_child_pressure() {
        let view = view_blend_test_view();
        let target = view_blend_test_target(0.5);
        let lower = view_blend_test_metric(0.75, 100.0);
        let higher = view_blend_test_metric(0.625, 100.0);
        let edge = view_blend_test_edge(
            view_blend_test_metric(0.25, 200.0),
            vec![lower, higher],
            0.0,
        );
        let (_, child_pressure) =
            lod_view_blend_pressures(view, target, &edge).expect("valid adjacent edge");
        let expected = view.selection_pressure(higher.node_metrics(), target, false);
        assert_eq!(child_pressure.to_bits(), expected.to_bits());

        let parent_pressure =
            view.selection_pressure(edge.parent_metric().node_metrics(), target, false);
        let weight = lod_view_blend_weight(view, target, &edge);
        assert!(
            ((1.0 - weight) * parent_pressure + weight * child_pressure - 1.0).abs()
                <= 4.0 * f32::EPSILON
        );
    }

    #[test]
    fn view_blend_pressure_pairs_classify_same_side_before_open_interval_order() {
        assert_eq!(
            lod_view_blend_weight_from_pressures(1.0, 1.0, 1.0).to_bits(),
            0.0_f32.to_bits(),
            "the selector stops at the parent when both pressures equal its threshold"
        );
        assert_eq!(
            lod_view_blend_weight_from_pressures(0.0, 0.0, 1.0).to_bits(),
            0.0_f32.to_bits(),
            "Coarsest pressure is a valid zero-width parent endpoint"
        );
        assert_eq!(
            lod_view_blend_weight_from_pressures(4.418_560_5, 4.418_560_5, 0.0).to_bits(),
            1.0_f32.to_bits(),
            "equal pressures above the threshold are categorically children-exact"
        );
        assert_eq!(
            lod_view_blend_weight_from_pressures_checked(1.25, 1.5),
            Some(1.0),
            "reversed pressures above the threshold are categorically children-exact"
        );
        assert_eq!(
            lod_view_blend_weight_from_pressures(1.25, 1.5, 0.0).to_bits(),
            1.0_f32.to_bits(),
            "a valid same-side inversion must not fall back to its retained endpoint"
        );
        assert_eq!(
            lod_view_blend_weight_from_pressures(f32::MAX, f32::MAX, 0.0).to_bits(),
            1.0_f32.to_bits(),
            "Original pressure is a valid zero-width children endpoint"
        );
        assert_eq!(
            lod_view_blend_weight_from_pressures(1.25, 1.0, 0.0).to_bits(),
            1.0_f32.to_bits(),
            "a child exactly on its threshold is children-exact once its parent refines"
        );
        assert_eq!(
            lod_view_blend_weight_from_pressures(0.75, 0.9, 1.0).to_bits(),
            0.0_f32.to_bits(),
            "reversed pressures below the threshold are categorically parent-exact"
        );
        assert_eq!(
            lod_view_blend_weight_from_pressures_checked(0.75, 1.25),
            None,
            "a reversed pair which straddles the selector threshold is contradictory"
        );
        assert_eq!(
            lod_view_blend_weight_from_pressures(f32::NAN, 0.5, 1.0).to_bits(),
            1.0_f32.to_bits(),
            "non-finite pressure"
        );

        let view = view_blend_test_view();
        let target = view_blend_test_target(0.5);
        let empty = view_blend_test_edge(view_blend_test_metric(0.25, 200.0), Vec::new(), 1.0);
        assert_eq!(
            lod_view_blend_weight(view, target, &empty).to_bits(),
            1.0_f32.to_bits(),
            "empty child cohort"
        );

        let non_monotone = view_blend_test_edge(
            view_blend_test_metric(0.75, 100.0),
            vec![view_blend_test_metric(0.25, 200.0)],
            1.0,
        );
        let (parent_pressure, child_pressure) =
            lod_view_blend_raw_pressures(view, target, &non_monotone)
                .expect("valid metrics still produce a raw pressure pair");
        assert!(parent_pressure <= 1.0 && child_pressure > 1.0);
        assert_eq!(
            lod_view_blend_pressures_for_testing(view, target, &non_monotone),
            None,
            "the qualification oracle must expose threshold-contradictory edges as invalid"
        );
        assert_eq!(
            lod_view_blend_weight_checked(view, target, &non_monotone),
            None,
            "RenderWorld must distinguish invalid evaluation from an authored endpoint"
        );
        assert_eq!(
            lod_view_blend_weight(view, target, &non_monotone).to_bits(),
            1.0_f32.to_bits(),
            "a contradictory pair retains its authored endpoint"
        );

        let nan_metric = view_blend_test_edge(
            view_blend_test_metric(0.25, f32::NAN),
            vec![view_blend_test_metric(0.75, 100.0)],
            0.0,
        );
        assert_eq!(
            lod_view_blend_pressures_for_testing(view, target, &nan_metric),
            None,
            "the qualification oracle must expose non-finite edges as invalid"
        );
        assert_eq!(
            lod_view_blend_weight_checked(view, target, &nan_metric),
            None
        );
        assert_eq!(
            lod_view_blend_weight(view, target, &nan_metric).to_bits(),
            0.0_f32.to_bits(),
            "NaN metric"
        );

        let zero_width = view_blend_test_edge(
            view_blend_test_metric(0.5, 100.0),
            vec![view_blend_test_metric(0.5, 100.0)],
            1.0,
        );
        let (parent_pressure, child_pressure) =
            lod_view_blend_pressures_for_testing(view, target, &zero_width)
                .expect("same-side zero-width pressure is a valid categorical endpoint");
        assert_eq!(parent_pressure.to_bits(), child_pressure.to_bits());
        let expected: f32 = if parent_pressure <= 1.0 { 0.0 } else { 1.0 };
        assert_eq!(
            lod_view_blend_weight_checked(view, target, &zero_width)
                .expect("same-side zero-width edge has an exact selector endpoint")
                .to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    fn view_blend_batch_keeps_same_side_endpoints_and_rejects_only_crossed_thresholds() {
        let view = view_blend_test_view();
        let high_target = view_blend_test_target(0.9);
        let same_side = view_blend_test_edge(
            view_blend_test_metric(0.25, 200.0),
            vec![view_blend_test_metric(0.25, 200.0)],
            1.0,
        );
        let fractional = view_blend_test_edge(
            view_blend_test_metric(0.45, 200.0),
            vec![view_blend_test_metric(1.0, 100.0)],
            0.0,
        );
        let (same_side_parent, same_side_child) =
            lod_view_blend_raw_pressures(view, high_target, &same_side)
                .expect("same-side metrics produce finite raw pressures");
        assert!(
            same_side_child.to_bits() == same_side_parent.to_bits() && same_side_parent > 1.0,
            "expected the authenticated zero-width children side, got parent={same_side_parent}, child={same_side_child}"
        );
        let (fractional_parent, fractional_child) =
            lod_view_blend_raw_pressures(view, high_target, &fractional)
                .expect("fractional metrics produce finite raw pressures");
        assert!(
            fractional_parent > 1.0 && fractional_child < 1.0,
            "expected an open-band pair, got parent={fractional_parent}, child={fractional_child}"
        );
        assert!(lod_view_blend_batch_pressures_are_valid(
            view,
            high_target,
            &[same_side, fractional.clone()],
        ));

        let crossed_target = view_blend_test_target(0.5);
        let crossed = view_blend_test_edge(
            view_blend_test_metric(0.75, 100.0),
            vec![view_blend_test_metric(0.25, 200.0)],
            1.0,
        );
        let (crossed_parent, crossed_child) =
            lod_view_blend_raw_pressures(view, crossed_target, &crossed)
                .expect("crossed metrics produce finite raw pressures");
        assert!(crossed_parent <= 1.0 && crossed_child > 1.0);
        assert!(!lod_view_blend_batch_pressures_are_valid(
            view,
            crossed_target,
            &[crossed, fractional],
        ));
    }

    #[test]
    fn view_blend_batch_requires_dense_descriptor_edges_and_complete_child_metrics() {
        let edge = view_blend_test_edge(
            view_blend_test_metric(0.25, 200.0),
            vec![view_blend_test_metric(0.75, 100.0)],
            0.0,
        );
        let descriptor = LodTemporalMorphDescriptor {
            child_physical_start: 0,
            child_count: 1,
            mapping_start: 0,
            edge_index: 0,
        };
        assert!(view_blend_batch_structure_is_valid(
            std::slice::from_ref(&edge),
            std::slice::from_ref(&descriptor),
        ));

        let mut invalid_descriptor = descriptor;
        invalid_descriptor.edge_index = 1;
        assert!(!view_blend_batch_structure_is_valid(
            std::slice::from_ref(&edge),
            std::slice::from_ref(&invalid_descriptor),
        ));

        let mut nested_edge = edge.clone();
        nested_edge.parent = edge.children[0];
        nested_edge.children = vec![LodNodeId(3)];
        nested_edge.child_metrics = vec![view_blend_test_metric(0.9, 50.0)];
        let nested_descriptor = LodTemporalMorphDescriptor {
            child_physical_start: 1,
            edge_index: 1,
            ..descriptor
        };
        assert!(!view_blend_batch_structure_is_valid(
            &[edge.clone(), nested_edge],
            &[descriptor, nested_descriptor],
        ));

        let mut disjoint_edge = edge.clone();
        disjoint_edge.parent = LodNodeId(4);
        disjoint_edge.children = vec![LodNodeId(5)];
        assert!(view_blend_batch_structure_is_valid(
            &[edge.clone(), disjoint_edge],
            &[descriptor, nested_descriptor],
        ));

        let mut incomplete_edge = edge;
        incomplete_edge.child_metrics.clear();
        assert!(!view_blend_batch_structure_is_valid(
            std::slice::from_ref(&incomplete_edge),
            std::slice::from_ref(&descriptor),
        ));
    }

    #[test]
    fn stable_child_frontier_view_blend_inherits_the_children_endpoint() {
        let substitution = LodTemporalSubstitution {
            key: LodTemporalSubstitutionKey {
                parent: LodNodeId(1),
                direction: LodTemporalDirection::Refine,
            },
            previous_nodes: vec![LodNodeId(1)],
            next_nodes: vec![LodNodeId(2), LodNodeId(3)],
            previous_gaussians: 1,
            next_gaussians: 2,
        };
        let transition = view_blend_transition(vec![substitution], vec![1.0_f32.to_bits()], 0, 0)
            .expect("persistent edge");
        assert_eq!(transition.initial_weight_bits(0), Some(1.0_f32.to_bits()));
        assert_eq!(transition.changed_gaussians(), 0);
    }

    #[test]
    fn applied_child_edge_serializes_an_intersecting_stable_ancestor_edge() {
        let stable_ancestor =
            view_blend_test_substitution(1, LodTemporalDirection::Refine, &[1], &[2, 3]);
        let applied_child =
            view_blend_test_substitution(2, LodTemporalDirection::Coarsen, &[4, 5], &[2]);

        assert_eq!(
            merge_disjoint_view_blend_substitutions(
                vec![stable_ancestor],
                std::slice::from_ref(&applied_child),
            ),
            vec![applied_child],
            "the 1->[2,3] stable boundary must wait while 2->[4,5] owns node 2"
        );
    }

    #[test]
    fn applied_edge_merge_preserves_disjoint_stable_boundaries() {
        let stable_ancestor =
            view_blend_test_substitution(1, LodTemporalDirection::Refine, &[1], &[2, 3]);
        let stable_disjoint =
            view_blend_test_substitution(8, LodTemporalDirection::Refine, &[8], &[9, 10]);
        let applied_child =
            view_blend_test_substitution(2, LodTemporalDirection::Coarsen, &[4, 5], &[2]);

        assert_eq!(
            merge_disjoint_view_blend_substitutions(
                vec![stable_ancestor, stable_disjoint.clone()],
                std::slice::from_ref(&applied_child),
            ),
            vec![applied_child, stable_disjoint],
        );
    }

    #[test]
    fn late_view_blend_provenance_survives_selection_and_cancellation_until_drawable_ack() {
        let view_id = LodRuntimeViewId::default();
        let (manifest, transport, settings, streaming) = fixture();
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let parent = runtime
            .hierarchy
            .manifest()
            .nodes
            .iter()
            .find(|node| !node.children.is_empty())
            .map(|node| node.id)
            .expect("fixture has an interior node");
        let children = runtime.hierarchy.children(parent).to_vec();
        let parent_page = runtime.hierarchy.page(parent).unwrap();
        let parent_descriptor = runtime.hierarchy.page_descriptor(parent_page).unwrap();
        let parent_decoded_len = parent_descriptor.decoded_len;
        let parent_gaussian_count = parent_descriptor.gaussian_count;
        runtime
            .cache
            .insert(
                parent_page,
                parent_decoded_len,
                u64::from(parent_gaussian_count),
                0,
            )
            .unwrap();
        let settings = GaussianLodSettings::default();
        let state = runtime.views.entry(view_id).or_default();
        state.late_view_blend_edges.insert(parent);
        state.commit_frontier(&[parent], &settings);
        assert!(state.late_view_blend_edges.contains(&parent));
        state.commit_frontier(&children, &settings);
        assert!(state.late_view_blend_edges.contains(&parent));

        runtime
            .restore_rendered_frontier(view_id, &[parent])
            .unwrap();
        assert!(
            runtime.views[&view_id]
                .late_view_blend_edges
                .contains(&parent)
        );

        runtime.acknowledge_rendered_frontier(view_id, &children);
        assert!(
            !runtime.views[&view_id]
                .late_view_blend_edges
                .contains(&parent)
        );
    }

    struct VirtualRuntimeFixture {
        manifest: GaussianLodManifest,
        transport: MemoryPageTransport,
        lod_settings: GaussianLodSettings,
        streaming_settings: GaussianStreamingSettings,
        encoded_root_bytes: usize,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum ToggleTransportError {
        ForcedBeginFailure,
        Memory(MemoryTransportError),
    }

    struct ToggleMemoryTransport {
        inner: MemoryPageTransport,
        fail_begin: bool,
        begin_count: u32,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum AdmissionDeferringTransportError {
        Io,
    }

    struct AdmissionDeferringMemoryTransport {
        inner: MemoryPageTransport,
        admission_saturated: bool,
        fail_next_io: bool,
        begin_count: u32,
    }

    struct CancelCountingTransport {
        inner: MemoryPageTransport,
        cancellations: Arc<AtomicU32>,
    }

    impl LodPageTransport for CancelCountingTransport {
        type Ticket = u64;
        type Error = MemoryTransportError;

        fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
            self.inner.begin(request)
        }

        fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
            self.inner.poll(ticket)
        }

        fn cancel(&mut self, ticket: &Self::Ticket) {
            self.cancellations.fetch_add(1, Ordering::Relaxed);
            self.inner.cancel(ticket);
        }
    }

    impl ToggleMemoryTransport {
        fn failing(inner: MemoryPageTransport) -> Self {
            Self {
                inner,
                fail_begin: true,
                begin_count: 0,
            }
        }
    }

    impl LodPageTransport for ToggleMemoryTransport {
        type Ticket = u64;
        type Error = ToggleTransportError;

        fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
            self.begin_count = self.begin_count.saturating_add(1);
            if self.fail_begin {
                Err(ToggleTransportError::ForcedBeginFailure)
            } else {
                self.inner
                    .begin(request)
                    .map_err(ToggleTransportError::Memory)
            }
        }

        fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
            match self.inner.poll(ticket) {
                PagePoll::Pending => PagePoll::Pending,
                PagePoll::Ready(payload) => PagePoll::Ready(payload),
                PagePoll::Failed(error) => PagePoll::Failed(ToggleTransportError::Memory(error)),
            }
        }

        fn cancel(&mut self, ticket: &Self::Ticket) {
            self.inner.cancel(ticket);
        }

        fn classify_error(error: &Self::Error) -> LodPageTransportFailure {
            match error {
                ToggleTransportError::ForcedBeginFailure => {
                    LodPageTransportFailure::transport("forced begin failure")
                }
                ToggleTransportError::Memory(_) => {
                    LodPageTransportFailure::transport("memory transport failure")
                }
            }
        }
    }

    impl LodPageTransport for AdmissionDeferringMemoryTransport {
        type Ticket = u64;
        type Error = AdmissionDeferringTransportError;

        fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
            self.begin_count = self.begin_count.saturating_add(1);
            self.inner
                .begin(request)
                .map_err(|_| AdmissionDeferringTransportError::Io)
        }

        fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
            if self.admission_saturated {
                return PagePoll::Pending;
            }
            if self.fail_next_io {
                self.fail_next_io = false;
                self.inner.cancel(ticket);
                return PagePoll::Failed(AdmissionDeferringTransportError::Io);
            }
            match self.inner.poll(ticket) {
                PagePoll::Pending => PagePoll::Pending,
                PagePoll::Ready(payload) => PagePoll::Ready(payload),
                PagePoll::Failed(_) => PagePoll::Failed(AdmissionDeferringTransportError::Io),
            }
        }

        fn cancel(&mut self, ticket: &Self::Ticket) {
            self.inner.cancel(ticket);
        }

        fn classify_error(_error: &Self::Error) -> LodPageTransportFailure {
            LodPageTransportFailure::transport("forced I/O failure")
        }
    }

    fn fixture() -> (
        GaussianLodManifest,
        MemoryPageTransport,
        GaussianLodSettings,
        GaussianStreamingSettings,
    ) {
        let scene = LodTestScene::screen_space_ladder();
        let mut lod = build_planar_3d_lod(
            &scene.cloud(),
            GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 16,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        let mut transport = MemoryPageTransport::default();
        for page in &lod.pages {
            let encoded = encode_page(page).unwrap();
            let descriptor = lod
                .manifest
                .pages
                .iter_mut()
                .find(|descriptor| descriptor.id == page.id)
                .unwrap();
            descriptor.storage = Some(
                crate::gaussian::formats::planar_3d_chunked::LodPageStorage {
                    uri: format!("memory://{}", page.id.0),
                    byte_range: None,
                    encoded_len: encoded.len() as u64,
                },
            );
            transport.insert(page.id, encoded);
        }
        lod.manifest.validate().unwrap();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        settings.budgets.max_active_gaussians = 1_000_000;
        settings.budgets.max_resident_gaussians = 1_000_000;
        settings.budgets.max_resident_bytes = 256 * 1024 * 1024;
        settings.budgets.max_resident_pages = 1024;
        settings.budgets.max_requests_per_frame = 1024;
        let streaming = GaussianStreamingSettings {
            max_concurrent_requests:
                crate::gaussian::lod_settings::MAX_STREAMING_CONCURRENT_REQUESTS,
            ..Default::default()
        };
        (lod.manifest, transport, settings, streaming)
    }

    fn abi16_morph_fixture() -> (
        GaussianLodManifest,
        MemoryPageTransport,
        GaussianLodSettings,
        GaussianStreamingSettings,
    ) {
        let (manifest, transport, settings, streaming) = fixture();
        let manifest = upgrade_manifest_to_synthetic_abi16_lifecycle_fixture(manifest).unwrap();
        (manifest, transport, settings, streaming)
    }

    fn remap_manifest_to_reverse_sparse_node_ids(
        mut manifest: GaussianLodManifest,
    ) -> GaussianLodManifest {
        let node_count = manifest.nodes.len() as u64;
        let remap = manifest
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                // Reverse manifest order and leave gaps so neither dense lookup
                // nor numeric-ID ordering can accidentally satisfy the test.
                let reversed = node_count - index as u64;
                (node.id, LodNodeId(reversed * 2 + 1))
            })
            .collect::<BTreeMap<_, _>>();
        for root in &mut manifest.roots {
            *root = remap[root];
        }
        for node in &mut manifest.nodes {
            node.id = remap[&node.id];
            node.parent = node.parent.map(|parent| remap[&parent]);
        }
        manifest.validate().expect("remapped ABI16 manifest");
        manifest
    }

    #[test]
    fn morph_presentation_uses_manifest_child_order_for_sparse_equal_depth_ties() {
        let (manifest, transport, settings, streaming) = abi16_morph_fixture();
        let sparse_manifest = remap_manifest_to_reverse_sparse_node_ids(manifest);
        let hierarchy = CompiledManifestLodHierarchy::new(sparse_manifest.clone())
            .expect("sparse ABI16 hierarchy");
        assert_ne!(hierarchy.manifest().nodes[0].id, LodNodeId(1));
        assert!(
            hierarchy
                .manifest()
                .nodes
                .iter()
                .all(|node| hierarchy.node_index(node.id).is_some())
        );

        let parents = hierarchy
            .manifest()
            .nodes
            .iter()
            .filter(|node| !node.is_leaf())
            .fold(
                BTreeMap::<u16, Vec<LodNodeId>>::new(),
                |mut levels, node| {
                    levels.entry(node.depth).or_default().push(node.id);
                    levels
                },
            )
            .into_values()
            .find(|nodes| nodes.len() >= 2)
            .expect("fixture has two internal parents at one depth")
            .into_iter()
            .take(2)
            .collect::<Vec<_>>();
        assert_eq!(parents.len(), 2);

        let mut presentation = BTreeMap::new();
        let mut mapping_by_child = BTreeMap::<LodNodeId, Vec<(LodNodeId, u32, u16)>>::new();
        let mut expected_nodes = Vec::new();
        let mut expected_mapping = Vec::new();
        for parent in parents.iter().copied() {
            let parent_index = hierarchy.node_index(parent).expect("parent index");
            let run_lengths = hierarchy
                .manifest()
                .morph_child_run_lengths_at(parent_index)
                .expect("ABI16 parent runs");
            let expanded = run_lengths
                .iter()
                .copied()
                .enumerate()
                .flat_map(|(parent_local, split_count)| {
                    std::iter::repeat_n(
                        (parent, parent_local as u32, split_count),
                        usize::from(split_count),
                    )
                })
                .collect::<Vec<_>>();
            let mut offset = 0_usize;
            for child in hierarchy.children(parent).iter().copied() {
                let child_index = hierarchy.node_index(child).expect("child index");
                let child_node = &hierarchy.manifest().nodes[child_index];
                let end = offset + child_node.representation.count as usize;
                mapping_by_child.insert(child, expanded[offset..end].to_vec());
                offset = end;
                expected_nodes.push(child);
                presentation.insert(
                    child,
                    LodPhysicalRange {
                        node: child,
                        page: child_node.representation.page,
                        slot: AtlasSlot {
                            index: child_index as u32,
                            generation: 1,
                        },
                        physical_start: (child_index as u32) * 1_024,
                        count: child_node.representation.count,
                    },
                );
            }
            assert_eq!(offset, expanded.len());
            expected_mapping.extend(expanded);
        }

        let ordered = manifest_ordered_morph_presentation_ranges(&hierarchy, presentation.clone())
            .expect("every presentation node belongs to the manifest");
        assert_eq!(
            ordered.iter().map(|range| range.node).collect::<Vec<_>>(),
            expected_nodes,
            "opaque node IDs must not replace manifest child-range order"
        );
        let numeric_id_order = presentation.into_values().collect::<Vec<_>>();
        let numeric_id_nodes = numeric_id_order
            .iter()
            .map(|range| range.node)
            .collect::<Vec<_>>();
        assert_ne!(
            numeric_id_nodes, expected_nodes,
            "the adversarial fixture must distinguish numeric and manifest order"
        );
        let ordinary_target_order =
            manifest_ordered_presentation_nodes(&hierarchy, &numeric_id_nodes)
                .expect("ordinary ABI16 target nodes belong to the manifest");
        assert_eq!(
            ordinary_target_order, expected_nodes,
            "the exact t=1 Morphing endpoint and its ordinary target must have one stable tie order"
        );
        assert_eq!(
            ordinary_target_order
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            numeric_id_nodes.iter().copied().collect(),
            "physical ordering must not change logical frontier or lease membership"
        );

        // Exercise the real resident-runtime call site, not only its ordering
        // helper. Decode and admit exactly the pages backing this target cut;
        // the target remains numerically node-ID ordered at the logical layer.
        let mut decode_transport = transport.clone();
        let mut runtime =
            LodStreamingRuntime::new(sparse_manifest, transport, &settings, &streaming)
                .expect("sparse ABI16 runtime");
        let target_pages = numeric_id_nodes
            .iter()
            .filter_map(|node| runtime.hierarchy.page(*node))
            .collect::<BTreeSet<_>>();
        for page_id in target_pages {
            let ticket = decode_transport
                .begin(PageRequest::new(page_id, PageRequestPriority::visible(0)))
                .expect("fixture page request");
            let payload = match decode_transport.poll(&ticket) {
                PagePoll::Ready(payload) => payload,
                other => panic!("fixture page must be immediately ready: {other:?}"),
            };
            let decoded = decode_page(&payload.bytes, LodCodecLimits::default())
                .expect("fixture page decodes");
            let descriptor = runtime
                .hierarchy
                .page_descriptor(page_id)
                .expect("page descriptor");
            runtime
                .cache
                .insert(
                    page_id,
                    descriptor.decoded_len,
                    u64::from(descriptor.gaussian_count),
                    1,
                )
                .expect("target page admission");
            runtime.decoded_pages.insert(page_id, decoded);
        }
        let target_frontier = LodFrontier {
            nodes: numeric_id_nodes.clone(),
            requested_nodes: Vec::new(),
            status: LodEffectiveStatus::default(),
        };
        let actual_target_ranges = runtime
            .physical_ranges(&target_frontier)
            .expect("resident ordinary target ranges");
        assert_eq!(
            actual_target_ranges
                .iter()
                .map(|range| range.node)
                .collect::<Vec<_>>(),
            ordered.iter().map(|range| range.node).collect::<Vec<_>>(),
            "actual ordinary ABI16 target source order must equal the exact t=1 Morphing endpoint"
        );

        let ordered_mapping = ordered
            .iter()
            .flat_map(|range| mapping_by_child[&range.node].iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(ordered_mapping, expected_mapping);
        let child_depth = hierarchy
            .node(expected_nodes[0])
            .expect("first child")
            .depth;
        assert!(expected_nodes.iter().all(|child| {
            hierarchy
                .node(*child)
                .is_some_and(|node| node.depth == child_depth)
        }));
        let equal_depth_source = ordered_mapping
            .iter()
            .copied()
            .map(|mapping| (child_depth, mapping))
            .collect::<Vec<_>>();
        let mut stable_tie_order = equal_depth_source.clone();
        stable_tie_order.sort_by_key(|(depth, _)| *depth);
        assert_eq!(stable_tie_order, equal_depth_source);

        for parent in parents {
            let positions = ordered_mapping
                .iter()
                .enumerate()
                .filter_map(|(index, mapping)| (mapping.0 == parent).then_some(index))
                .collect::<Vec<_>>();
            assert_eq!(
                positions.last().unwrap() - positions[0] + 1,
                positions.len(),
                "one parent's equal-depth proxy cohort must stay contiguous"
            );

            let parent_index = hierarchy.node_index(parent).unwrap();
            let run_lengths = hierarchy
                .manifest()
                .morph_child_run_lengths_at(parent_index)
                .unwrap();
            let parent_mapping = positions
                .iter()
                .map(|index| ordered_mapping[*index])
                .collect::<Vec<_>>();
            let mut offset = 0_usize;
            for (parent_local, split_count) in run_lengths.iter().copied().enumerate() {
                let end = offset + usize::from(split_count);
                assert!(parent_mapping[offset..end].iter().all(|mapping| {
                    mapping.1 == parent_local as u32 && mapping.2 == split_count
                }));

                let parent_alpha = 0.73_f32;
                let parent_tau = -(1.0 - parent_alpha).ln();
                let proxy_alpha = 1.0 - (-parent_tau / f32::from(split_count)).exp();
                let recomposed_alpha = 1.0 - (1.0 - proxy_alpha).powi(i32::from(split_count));
                assert!((recomposed_alpha - parent_alpha).abs() <= 2.0e-6);
                offset = end;
            }
            assert_eq!(offset, parent_mapping.len());
        }
    }

    fn package_bootstrap_fixture() -> (
        CompiledManifestLodHierarchy,
        MemoryPageTransport,
        GaussianLodSettings,
        LodPackageBootstrapBudget,
    ) {
        let source = LodTestScene::nested_octants(3).cloud();
        let mut lod = build_planar_3d_lod(
            &source,
            GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 8,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        let mut transport = MemoryPageTransport::default();
        for page in &lod.pages {
            let encoded = encode_page(page).unwrap();
            let descriptor = lod
                .manifest
                .pages
                .iter_mut()
                .find(|descriptor| descriptor.id == page.id)
                .unwrap();
            descriptor.storage = Some(LodPageStorage {
                uri: format!("memory://{}", page.id.0),
                byte_range: None,
                encoded_len: encoded.len() as u64,
            });
            transport.insert(page.id, encoded);
        }
        lod.manifest.validate().unwrap();
        let stride = lod
            .manifest
            .pages
            .iter()
            .map(|descriptor| descriptor.gaussian_count)
            .max()
            .unwrap();
        let mut settings = GaussianLodSettings::default();
        settings.budgets.max_active_gaussians = 1_000_000;
        settings.budgets.max_resident_gaussians = 1_000_000;
        settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
        settings.budgets.max_resident_pages = 128;
        settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
        let budget = LodPackageBootstrapBudget {
            max_pages: 8,
            max_active_gaussians: 8_192,
            max_encoded_bytes: 2 * 1024 * 1024,
            max_decoded_bytes: 2 * 1024 * 1024,
            max_gpu_bytes: 2 * 1024 * 1024,
            gpu_bytes_per_slot: u64::from(stride) * gaussian_3d_gpu_bytes_per_record(),
        };
        (
            CompiledManifestLodHierarchy::new(lod.manifest).unwrap(),
            transport,
            settings,
            budget,
        )
    }

    #[test]
    fn temporal_boundary_confirmation_rejects_chatter_and_same_frame_reentry() {
        let refine = LodTemporalSubstitutionKey {
            parent: LodNodeId(7),
            direction: LodTemporalDirection::Refine,
        };
        let coarsen = LodTemporalSubstitutionKey {
            parent: LodNodeId(7),
            direction: LodTemporalDirection::Coarsen,
        };
        let mut demands = HashMap::new();

        let frame_one = LodRuntimeFrameId(1);
        assert!(
            confirmed_temporal_keys(&mut demands, &BTreeSet::from([refine]), frame_one).is_empty()
        );
        assert!(
            confirmed_temporal_keys(&mut demands, &BTreeSet::from([refine]), frame_one).is_empty(),
            "a second selector pass in one application frame must not satisfy confirmation"
        );
        assert!(
            confirmed_temporal_keys(
                &mut demands,
                &BTreeSet::from([coarsen]),
                LodRuntimeFrameId(2),
            )
            .is_empty(),
            "crossing back over the same boundary resets the opposite direction"
        );
        assert!(
            confirmed_temporal_keys(
                &mut demands,
                &BTreeSet::from([refine]),
                LodRuntimeFrameId(3),
            )
            .is_empty()
        );
        assert_eq!(
            confirmed_temporal_keys(
                &mut demands,
                &BTreeSet::from([refine]),
                LodRuntimeFrameId(4),
            ),
            BTreeSet::from([refine])
        );
    }

    #[test]
    fn temporal_energy_budget_scales_with_current_cut_and_has_a_native_cap() {
        assert_eq!(temporal_changed_gaussian_budget(1), 1);
        assert_eq!(temporal_changed_gaussian_budget(240), 10);
        assert_eq!(
            temporal_changed_gaussian_budget(u64::MAX),
            LOD_TEMPORAL_MAX_CHANGED_GAUSSIANS_PER_FRAME
        );
    }

    fn settle_temporal_fixture(
        runtime: &mut LodStreamingRuntime<MemoryPageTransport>,
        settings: &GaussianLodSettings,
        streaming: &GaussianStreamingSettings,
    ) -> LodStreamFrame {
        (0..256)
            .find_map(|_| {
                let frame = runtime.update(view(), settings, streaming).unwrap();
                (frame.selection_stable()
                    && frame.frontier().requested_nodes.is_empty()
                    && frame.queued_requests() == 0
                    && frame.in_flight_requests() == 0)
                    .then_some(frame)
            })
            .expect("temporal fixture reaches a stable resident cut")
    }

    fn ordered_coarsening_refinement_depths(
        hierarchy: &CompiledManifestLodHierarchy,
        previous_frontier: &[LodNodeId],
        desired_frontier: &[LodNodeId],
    ) -> BTreeMap<LodNodeId, u32> {
        let desired = desired_frontier.iter().copied().collect::<BTreeSet<_>>();
        let mut refinements = BTreeMap::<LodNodeId, u32>::new();
        let mut path = Vec::new();
        for &previous in previous_frontier {
            if desired.contains(&previous) {
                continue;
            }
            path.clear();
            let mut cursor = previous;
            while let Some(parent) = hierarchy.parent(cursor) {
                path.push(parent);
                if desired.contains(&parent) {
                    for (index, node) in path.iter().copied().enumerate() {
                        let depth = u32::try_from(index + 1).unwrap_or(u32::MAX);
                        refinements
                            .entry(node)
                            .and_modify(|current| *current = (*current).max(depth))
                            .or_insert(depth);
                    }
                    break;
                }
                cursor = parent;
            }
        }
        refinements
    }

    fn ordered_active_coarsening_holds(
        release_frames: &mut BTreeMap<LodNodeId, u64>,
        refinements: &BTreeMap<LodNodeId, u32>,
        frame_sequence: u64,
    ) -> BTreeSet<LodNodeId> {
        release_frames.retain(|node, _| refinements.contains_key(node));
        for (&node, &depth) in refinements {
            release_frames.entry(node).or_insert_with(|| {
                frame_sequence.saturating_add(coarsening_hold_frames(node, depth))
            });
        }
        release_frames
            .iter()
            .filter_map(|(&node, &release)| (release > frame_sequence).then_some(node))
            .collect()
    }

    fn hash_temporal_selection(
        hierarchy: &CompiledManifestLodHierarchy,
        view: LodView,
        settings: &GaussianLodSettings,
        previous: &[LodNodeId],
        releases: &mut HashMap<LodNodeId, u64>,
        frame: u64,
    ) -> LodFrontier<LodNodeId> {
        let desired = select_frontier_with_previous_and_visibility(
            hierarchy,
            &AllResident,
            view,
            settings,
            previous,
            |_, _| true,
        )
        .unwrap();
        if desired.nodes == previous || previous.is_empty() {
            releases.clear();
            return desired;
        }
        let refinements = coarsening_refinement_depths(hierarchy, previous, &desired.nodes);
        let held = active_coarsening_holds(releases, &refinements, frame);
        if held.is_empty() {
            desired
        } else {
            select_frontier_with_previous_holds_and_visibility(
                hierarchy,
                &AllResident,
                view,
                settings,
                previous,
                &held,
                |_, _| true,
            )
            .unwrap()
        }
    }

    fn ordered_temporal_selection(
        hierarchy: &CompiledManifestLodHierarchy,
        view: LodView,
        settings: &GaussianLodSettings,
        previous: &[LodNodeId],
        releases: &mut BTreeMap<LodNodeId, u64>,
        frame: u64,
    ) -> LodFrontier<LodNodeId> {
        let desired = select_frontier_with_previous_and_visibility(
            hierarchy,
            &AllResident,
            view,
            settings,
            previous,
            |_, _| true,
        )
        .unwrap();
        if desired.nodes == previous || previous.is_empty() {
            releases.clear();
            return desired;
        }
        let refinements = ordered_coarsening_refinement_depths(hierarchy, previous, &desired.nodes);
        let held = ordered_active_coarsening_holds(releases, &refinements, frame);
        if held.is_empty() {
            desired
        } else {
            select_frontier_with_previous_holds_and_visibility(
                hierarchy,
                &AllResident,
                view,
                settings,
                previous,
                &held,
                |_, _| true,
            )
            .unwrap()
        }
    }

    fn changing_cut_benchmark_fixture() -> (
        CompiledManifestLodHierarchy,
        GaussianLodSettings,
        LodView,
        LodView,
        Vec<LodNodeId>,
    ) {
        let fixture = virtual_runtime_fixture();
        let hierarchy = CompiledManifestLodHierarchy::new(fixture.manifest).unwrap();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 0.5;
        settings.budgets.max_active_gaussians = u64::MAX;
        settings.budgets.max_traversal_nodes_per_view = 1_000_000;
        let near = LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, 8.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        let far = LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, 1_000.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        let initial = select_frontier_with_previous_and_visibility(
            &hierarchy,
            &AllResident,
            near,
            &settings,
            &[],
            |_, _| true,
        )
        .unwrap()
        .nodes;
        let far_nodes = select_frontier_with_previous_and_visibility(
            &hierarchy,
            &AllResident,
            far,
            &settings,
            &initial,
            |_, _| true,
        )
        .unwrap()
        .nodes;
        assert_ne!(initial, far_nodes, "benchmark views must change the cut");
        (hierarchy, settings, near, far, initial)
    }

    #[test]
    fn hash_temporal_bookkeeping_matches_ordered_reference_on_large_changing_cuts() {
        let (hierarchy, settings, near, far, initial) = changing_cut_benchmark_fixture();
        let mut hash_previous = initial.clone();
        let mut ordered_previous = initial;
        let mut hash_releases = HashMap::new();
        let mut ordered_releases = BTreeMap::new();

        for frame in 1..=16 {
            let view = if frame % 2 == 0 { near } else { far };
            let hash = hash_temporal_selection(
                &hierarchy,
                view,
                &settings,
                &hash_previous,
                &mut hash_releases,
                frame,
            );
            let ordered = ordered_temporal_selection(
                &hierarchy,
                view,
                &settings,
                &ordered_previous,
                &mut ordered_releases,
                frame,
            );
            assert_eq!(hash, ordered, "frame={frame}");
            assert_eq!(hash_releases.len(), ordered_releases.len());
            assert!(
                hash_releases
                    .iter()
                    .all(|(node, release)| ordered_releases.get(node) == Some(release))
            );
            hash_previous = hash.nodes;
            ordered_previous = ordered.nodes;
        }
    }

    #[test]
    #[ignore = "manual large changing-cut temporal-selection benchmark"]
    fn benchmark_large_changing_cut_temporal_selection() {
        const FRAMES: u64 = 24;
        let (hierarchy, settings, near, far, initial) = changing_cut_benchmark_fixture();

        let mut ordered_previous = initial.clone();
        let mut ordered_releases = BTreeMap::new();
        let started = Instant::now();
        for frame in 1..=FRAMES {
            let view = if frame % 2 == 0 { near } else { far };
            let selected = black_box(ordered_temporal_selection(
                black_box(&hierarchy),
                black_box(view),
                black_box(&settings),
                black_box(&ordered_previous),
                &mut ordered_releases,
                frame,
            ));
            ordered_previous = selected.nodes;
        }
        let ordered_elapsed = started.elapsed();

        let mut hash_previous = initial;
        let mut hash_releases = HashMap::new();
        let started = Instant::now();
        for frame in 1..=FRAMES {
            let view = if frame % 2 == 0 { near } else { far };
            let selected = black_box(hash_temporal_selection(
                black_box(&hierarchy),
                black_box(view),
                black_box(&settings),
                black_box(&hash_previous),
                &mut hash_releases,
                frame,
            ));
            hash_previous = selected.nodes;
        }
        let hash_elapsed = started.elapsed();

        println!(
            "large changing-cut temporal selection: ordered={:?}/frame hash={:?}/frame speedup={:.3}x",
            ordered_elapsed / FRAMES as u32,
            hash_elapsed / FRAMES as u32,
            ordered_elapsed.as_secs_f64() / hash_elapsed.as_secs_f64(),
        );
    }

    #[test]
    fn exact_fixed_point_cache_hits_and_invalidates_for_view_and_policy_changes() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.quality = 0.0;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();

        for _ in 0..64 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            if frame.frontier.requested_nodes.is_empty()
                && frame.in_flight_requests == 0
                && frame.queued_requests == 0
            {
                break;
            }
        }

        // The first settled update proves that the selector reproduces its
        // previous cut. The following exact update must reuse the complete
        // frontier, including quality and request status.
        let proved = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(
            runtime.views[&LodRuntimeViewId::default()]
                .stable_selection
                .as_ref()
                .is_some_and(|cached| cached.payload.is_some())
        );
        let traversals = runtime.selection_traversals;
        let pin_rebuilds = runtime.frontier_pin_rebuilds;
        let range_rebuilds = runtime.physical_range_rebuilds;
        let payload_hits = runtime.stable_payload_hits;
        let cached = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(runtime.selection_traversals, traversals);
        assert_eq!(runtime.frontier_pin_rebuilds, pin_rebuilds);
        assert_eq!(runtime.physical_range_rebuilds, range_rebuilds);
        assert_eq!(runtime.stable_payload_hits, payload_hits + 1);
        assert_eq!(cached.frontier, proved.frontier);
        assert_eq!(cached.physical_ranges, proved.physical_ranges);
        assert_eq!(cached.complete_resident_cut, proved.complete_resident_cut);

        let moved = LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, 80.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        runtime.update(moved, &settings, &streaming).unwrap();
        assert_eq!(runtime.selection_traversals, traversals + 1);
        assert_eq!(runtime.frontier_pin_rebuilds, pin_rebuilds + 1);
        assert_eq!(runtime.physical_range_rebuilds, range_rebuilds + 1);
        assert_eq!(runtime.stable_payload_hits, payload_hits + 1);

        let mut changed_policy = settings.clone();
        changed_policy.frustum_margin = 0.25;
        runtime.update(moved, &changed_policy, &streaming).unwrap();
        assert_eq!(runtime.selection_traversals, traversals + 2);
        assert_eq!(runtime.frontier_pin_rebuilds, pin_rebuilds + 2);
        assert_eq!(runtime.physical_range_rebuilds, range_rebuilds + 2);
        assert_eq!(runtime.stable_payload_hits, payload_hits + 1);

        let mut frozen = changed_policy;
        frozen.selection_mode = LodSelectionMode::Frozen;
        runtime.update(moved, &frozen, &streaming).unwrap();
        assert_eq!(runtime.selection_traversals, traversals + 3);
        assert_eq!(runtime.frontier_pin_rebuilds, pin_rebuilds + 3);
        assert_eq!(runtime.physical_range_rebuilds, range_rebuilds + 3);
        assert_eq!(runtime.stable_payload_hits, payload_hits + 1);
    }

    #[test]
    fn same_view_residency_progress_invalidates_fixed_point_cache_immediately() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_resident_pages = 1;
        settings.budgets.max_active_gaussians = root_active_gaussians(&manifest);
        let root = root_page(&manifest);
        let victim = manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id != root)
            .expect("fixture needs a non-root eviction victim")
            .id;
        let victim_descriptor = manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == victim)
            .unwrap()
            .clone();
        let mut decode_transport = transport.clone();
        let victim_input = memory_preprocess_input(
            &manifest,
            &mut decode_transport,
            victim,
            streaming.effective_max_encoded_page_bytes(),
        );
        let victim_page = decode_page(&victim_input.payload.bytes, victim_input.limits).unwrap();
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        runtime
            .cache
            .insert(
                victim,
                victim_descriptor.decoded_len,
                u64::from(victim_descriptor.gaussian_count),
                0,
            )
            .unwrap();
        runtime.decoded_pages.insert(victim, victim_page);

        let first = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(first.frontier.nodes.is_empty());
        assert!(
            runtime.views[&LodRuntimeViewId::default()]
                .stable_selection
                .as_ref()
                .is_some_and(|cached| cached.payload.is_some()),
            "the empty resident fixed point should be cacheable while its root request is pending"
        );

        let mut observed_completion = false;
        for _ in 0..128 {
            let revision_before = runtime.residency_revision;
            let traversals_before = runtime.selection_traversals;
            let pin_rebuilds_before = runtime.frontier_pin_rebuilds;
            let range_rebuilds_before = runtime.physical_range_rebuilds;
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            if frame.completed_pages.is_empty() {
                continue;
            }
            observed_completion = true;
            assert!(runtime.residency_revision > revision_before);
            assert!(
                runtime.selection_traversals > traversals_before,
                "a page committed after the cached first selection must force the immediate second selection"
            );
            assert!(runtime.frontier_pin_rebuilds > pin_rebuilds_before);
            assert!(runtime.physical_range_rebuilds > range_rebuilds_before);
            assert!(runtime.cache.contains(root));
            assert!(
                !runtime.cache.contains(victim),
                "the completing root should evict the unpinned sacrificial page"
            );
            assert!(!frame.frontier.nodes.is_empty());
            break;
        }
        assert!(observed_completion, "fixture root should become resident");
    }

    #[test]
    fn stable_payload_refresh_keeps_in_flight_demand_alive() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.quality = 0.0;
        settings.budgets.max_active_gaussians = root_active_gaussians(&manifest);
        settings
            .budgets
            .max_cooperative_preprocess_gaussians_per_frame = 1;
        let root = root_page(&manifest);
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        runtime.preprocessor = LodPagePreprocessor::new_cooperative_for_tests(1).unwrap();

        let first = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(first.in_flight_requests, 1);
        assert!(!first.frontier.requested_nodes.is_empty());
        assert!(
            runtime.views[&LodRuntimeViewId::default()]
                .stable_selection
                .as_ref()
                .is_some_and(|cached| cached.payload.is_some())
        );

        let traversals = runtime.selection_traversals;
        let pin_rebuilds = runtime.frontier_pin_rebuilds;
        let range_rebuilds = runtime.physical_range_rebuilds;
        let payload_hits = runtime.stable_payload_hits;
        let cancellations = runtime.preprocessor.stats().cancellations;
        let frame = runtime.begin_frame();
        let second = runtime
            .update_view_in_frame(
                frame,
                LodRuntimeViewId::default(),
                view(),
                &settings,
                &streaming,
            )
            .unwrap();

        assert_eq!(runtime.selection_traversals, traversals);
        assert_eq!(runtime.frontier_pin_rebuilds, pin_rebuilds);
        assert_eq!(runtime.physical_range_rebuilds, range_rebuilds);
        assert_eq!(runtime.stable_payload_hits, payload_hits + 1);
        assert_eq!(second.in_flight_requests, 1);
        let state = &runtime.views[&LodRuntimeViewId::default()];
        assert_eq!(
            second.frontier.status.requested_pages,
            u32::try_from(state.requested_pages.len()).unwrap()
        );
        assert_eq!(state.requested_pages_frame, frame);
        assert!(state.requested_pages.contains(&root));
        assert!(runtime.preprocessor.contains(root));

        runtime.finish_frame(frame).unwrap();
        assert!(runtime.preprocessor.contains(root));
        assert_eq!(runtime.preprocessor.stats().cancellations, cancellations);
    }

    #[test]
    #[ignore = "manual compiled-manifest stable payload cache benchmark"]
    fn benchmark_large_compiled_manifest_stable_payload_cache() {
        const FULL_ITERATIONS: u32 = 24;
        const CACHED_ITERATIONS: u32 = 2_000;
        let (hierarchy, settings, view, _, initial) = changing_cut_benchmark_fixture();
        let fixed = select_frontier_with_previous_and_visibility(
            &hierarchy,
            &AllResident,
            view,
            &settings,
            &initial,
            |_, _| true,
        )
        .unwrap();
        assert_eq!(fixed.nodes, initial);

        let mut state = LodRuntimeViewState::default();
        state.commit_frontier(&fixed.nodes, &settings);
        let policy = LodHysteresisPolicy::from(&settings);
        let physical_ranges = fixed
            .nodes
            .iter()
            .copied()
            .enumerate()
            .map(|(index, node)| {
                let index = u32::try_from(index).unwrap();
                LodPhysicalRange {
                    node,
                    page: LodPageId(u64::from(index) + 1),
                    slot: AtlasSlot {
                        index,
                        generation: 1,
                    },
                    physical_start: index,
                    count: 1,
                }
            })
            .collect();
        let key = StableSelectionKey {
            view,
            policy,
            selection_view_frozen: false,
            residency_revision: 7,
        };
        state.stable_selection = Some(StableSelectionCache {
            key,
            frontier: fixed.clone(),
            payload: Some(StableFramePayload {
                physical_ranges,
                complete_resident_cut: true,
            }),
        });

        let started = Instant::now();
        for _ in 0..FULL_ITERATIONS {
            black_box(
                select_frontier_with_previous_and_visibility(
                    black_box(&hierarchy),
                    &AllResident,
                    black_box(view),
                    black_box(&settings),
                    black_box(&initial),
                    |_, _| true,
                )
                .unwrap(),
            );
        }
        let full_elapsed = started.elapsed();

        let started = Instant::now();
        for _ in 0..CACHED_ITERATIONS {
            let frontier = state
                .cached_stable_selection(black_box(key))
                .unwrap()
                .clone();
            let payload = state.cached_stable_payload(black_box(key)).unwrap().clone();
            black_box((frontier, payload));
        }
        let cached_elapsed = started.elapsed();
        let full_per_frame = full_elapsed / FULL_ITERATIONS;
        let cached_per_frame = cached_elapsed / CACHED_ITERATIONS;
        println!(
            "large compiled-manifest stable payload: full_selector={full_per_frame:?}/frame cached_frontier_and_ranges={cached_per_frame:?}/frame speedup={:.3}x",
            full_per_frame.as_secs_f64() / cached_per_frame.as_secs_f64(),
        );
    }

    #[test]
    fn releasing_caller_lease_wakes_capacity_blocked_request() {
        let (manifest, transport, settings, streaming) = fixture();
        let pages = manifest
            .pages
            .iter()
            .take(2)
            .map(|descriptor| descriptor.id)
            .collect::<Vec<_>>();
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let descriptor = runtime.hierarchy.page_descriptor(pages[0]).unwrap().clone();
        runtime
            .cache
            .insert(
                pages[0],
                descriptor.decoded_len,
                u64::from(descriptor.gaussian_count),
                1,
            )
            .unwrap();
        runtime.retain_resident_page(pages[0]).unwrap();
        runtime.capacity_blocked.insert(
            pages[1],
            PageRequest::new(pages[1], PageRequestPriority::visible(1)),
        );

        runtime.release_resident_page(pages[0]).unwrap();

        assert!(runtime.capacity_blocked.is_empty());
        assert!(runtime.queue.contains(pages[1]));
    }

    #[test]
    fn adding_view_pins_does_not_claim_to_release_blocked_capacity() {
        let (manifest, transport, settings, streaming) = fixture();
        let pages = manifest
            .pages
            .iter()
            .take(2)
            .map(|descriptor| descriptor.id)
            .collect::<Vec<_>>();
        assert_eq!(pages.len(), 2);
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        for &page in &pages {
            let descriptor = runtime.hierarchy.page_descriptor(page).unwrap().clone();
            runtime
                .cache
                .insert(
                    page,
                    descriptor.decoded_len,
                    u64::from(descriptor.gaussian_count),
                    1,
                )
                .unwrap();
        }

        let view_id = LodRuntimeViewId(91);
        runtime.views.entry(view_id).or_default().selected_frontier = BTreeSet::from([pages[0]]);
        assert!(!runtime.synchronize_view_pins(view_id).unwrap());

        runtime.views.get_mut(&view_id).unwrap().selected_frontier =
            BTreeSet::from([pages[0], pages[1]]);
        assert!(
            !runtime.synchronize_view_pins(view_id).unwrap(),
            "adding a pin consumes capacity and must not wake blocked work"
        );

        runtime.views.get_mut(&view_id).unwrap().selected_frontier = BTreeSet::from([pages[1]]);
        assert!(
            runtime.synchronize_view_pins(view_id).unwrap(),
            "dropping a pin can make a cache slot evictable"
        );
    }

    #[test]
    fn dropping_runtime_cancels_all_in_flight_transport_work() {
        let (manifest, transport, settings, streaming) = fixture();
        let cancellations = Arc::new(AtomicU32::new(0));
        let mut runtime = LodStreamingRuntime::new(
            manifest,
            CancelCountingTransport {
                inner: transport,
                cancellations: cancellations.clone(),
            },
            &settings,
            &streaming,
        )
        .unwrap();
        let frame = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(frame.in_flight_requests > 0);
        let in_flight = frame.in_flight_requests;
        drop(runtime);
        assert_eq!(cancellations.load(Ordering::Relaxed), in_flight);
    }

    #[test]
    fn finishing_frame_cancels_only_work_no_active_view_demands() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_pending_requests = 8;
        let cancellations = Arc::new(AtomicU32::new(0));
        let mut runtime = LodStreamingRuntime::new(
            manifest,
            CancelCountingTransport {
                inner: transport,
                cancellations: cancellations.clone(),
            },
            &settings,
            &streaming,
        )
        .unwrap();
        let pages = runtime
            .hierarchy
            .manifest()
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .filter(|page| !runtime.coverage_guard.pages.contains(page))
            .take(5)
            .collect::<Vec<_>>();
        assert_eq!(pages.len(), 5);
        let frame = runtime.begin_frame();
        let request = |page| PageRequest::new(page, PageRequestPriority::visible(1));
        runtime
            .views
            .entry(LodRuntimeViewId(7))
            .or_default()
            .requested_pages = BTreeSet::from([pages[0]]);
        runtime
            .views
            .get_mut(&LodRuntimeViewId(7))
            .unwrap()
            .requested_pages_frame = frame;
        runtime
            .views
            .get_mut(&LodRuntimeViewId(7))
            .unwrap()
            .admitted_pages = BTreeSet::from([pages[0]]);
        runtime
            .views
            .get_mut(&LodRuntimeViewId(7))
            .unwrap()
            .admitted_pages_frame = frame;
        assert_eq!(
            runtime.queue.enqueue(request(pages[0])),
            RequestEnqueue::Enqueued
        );
        assert_eq!(
            runtime.queue.enqueue(request(pages[1])),
            RequestEnqueue::Enqueued
        );
        runtime.in_flight.insert(
            pages[2],
            InFlight {
                ticket: 123,
                request: request(pages[2]),
            },
        );
        runtime.capacity_blocked.insert(pages[3], request(pages[3]));
        let preprocess_descriptor = runtime.hierarchy.page_descriptor(pages[4]).unwrap().clone();
        runtime
            .preprocessor
            .submit(LodPagePreprocessInput {
                request: request(pages[4]),
                payload: PagePayload::new(pages[4], Vec::new()),
                limits: page_codec_limits(
                    &preprocess_descriptor,
                    streaming.effective_max_encoded_page_bytes(),
                ),
                descriptor: preprocess_descriptor,
                max_encoded_page_bytes: streaming.effective_max_encoded_page_bytes(),
                support_sigma: runtime.hierarchy.manifest().build.settings.support_sigma,
            })
            .unwrap();
        for page in &pages {
            runtime.attempts.insert(*page, 1);
        }

        runtime.finish_frame(frame).unwrap();
        assert_eq!(cancellations.load(Ordering::Relaxed), 1);
        assert!(runtime.queue.contains(pages[0]));
        assert!(!runtime.queue.contains(pages[1]));
        assert!(!runtime.in_flight.contains_key(&pages[2]));
        assert!(!runtime.capacity_blocked.contains_key(&pages[3]));
        assert!(!runtime.preprocessor.contains(pages[4]));
        assert_eq!(runtime.attempts.get(&pages[0]), Some(&1));
        for page in &pages[1..] {
            assert!(!runtime.attempts.contains_key(page));
        }
        runtime.finish_frame(frame).unwrap();
        assert_eq!(cancellations.load(Ordering::Relaxed), 1);
    }

    fn memory_preprocess_input(
        manifest: &GaussianLodManifest,
        transport: &mut MemoryPageTransport,
        page: LodPageId,
        max_encoded_page_bytes: u64,
    ) -> LodPagePreprocessInput {
        let descriptor = manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == page)
            .unwrap()
            .clone();
        let mut request = PageRequest::new(page, PageRequestPriority::visible(1));
        request.expected_bytes = descriptor
            .storage
            .as_ref()
            .map(|storage| storage.encoded_len);
        let ticket = transport.begin(request).unwrap();
        let PagePoll::Ready(payload) = transport.poll(&ticket) else {
            panic!("memory transport must return an inserted page")
        };
        LodPagePreprocessInput {
            request,
            payload,
            limits: page_codec_limits(&descriptor, max_encoded_page_bytes),
            descriptor,
            max_encoded_page_bytes,
            support_sigma: manifest.build.settings.support_sigma,
        }
    }

    fn seed_resident_coverage_guard(
        runtime: &mut LodStreamingRuntime<MemoryPageTransport>,
        manifest: &GaussianLodManifest,
        transport: &mut MemoryPageTransport,
        streaming: &GaussianStreamingSettings,
    ) {
        let guard_pages = runtime
            .coverage_guard
            .pages
            .iter()
            .copied()
            .collect::<Vec<_>>();
        for page_id in guard_pages {
            let input = memory_preprocess_input(
                manifest,
                transport,
                page_id,
                streaming.effective_max_encoded_page_bytes(),
            );
            let page = decode_page(&input.payload.bytes, input.limits).unwrap();
            let descriptor = runtime.hierarchy.page_descriptor(page_id).unwrap().clone();
            runtime.queue.remove(page_id);
            runtime
                .cache
                .insert(
                    page_id,
                    descriptor.decoded_len,
                    u64::from(descriptor.gaussian_count),
                    0,
                )
                .unwrap();
            runtime.decoded_pages.insert(page_id, page);
            runtime.pin_coverage_guard_page(page_id).unwrap();
        }
    }

    fn stage_ready_pages(
        runtime: &mut LodStreamingRuntime<MemoryPageTransport>,
        manifest: &GaussianLodManifest,
        transport: &mut MemoryPageTransport,
        streaming: &GaussianStreamingSettings,
        pages: &[LodPageId],
    ) {
        runtime.preprocessor = LodPagePreprocessor::new_cooperative_for_tests(pages.len()).unwrap();
        for &page in pages {
            runtime
                .preprocessor
                .submit(memory_preprocess_input(
                    manifest,
                    transport,
                    page,
                    streaming.effective_max_encoded_page_bytes(),
                ))
                .unwrap();
        }
        for sequence in 1..=256 {
            runtime
                .preprocessor
                .advance(10_000 + sequence, NonZeroU32::MAX);
            if runtime.preprocessor.ready_page_ids().len() == pages.len() {
                return;
            }
        }
        panic!("cooperative preprocessing did not make every fixture page ready");
    }

    fn record_test_frame_demand(
        runtime: &mut LodStreamingRuntime<MemoryPageTransport>,
        frame: LodRuntimeFrameId,
        view: LodRuntimeViewId,
        pages: impl IntoIterator<Item = LodPageId>,
    ) {
        let pages = pages.into_iter().collect::<BTreeSet<_>>();
        let state = runtime.views.entry(view).or_default();
        state.requested_pages.clone_from(&pages);
        state.requested_pages_frame = frame;
        state.admitted_pages = pages;
        state.admitted_pages_frame = frame;
    }

    fn commit_test_ready_pages(
        runtime: &mut LodStreamingRuntime<MemoryPageTransport>,
        frame: LodRuntimeFrameId,
        settings: &GaussianLodSettings,
        streaming: &GaussianStreamingSettings,
    ) -> Vec<LodPageId> {
        let mut completed = Vec::new();
        let mut preprocess_failed = Vec::new();
        let mut failed = Vec::new();
        runtime
            .commit_preprocessed_pages(
                frame,
                settings,
                streaming,
                &mut completed,
                &mut preprocess_failed,
                &mut failed,
            )
            .unwrap();
        assert!(preprocess_failed.is_empty());
        assert!(failed.is_empty());
        completed
    }

    #[test]
    fn saturated_completion_batch_stays_resident_until_frame_finish() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_resident_pages = 3;
        settings.budgets.max_upload_bytes_per_frame = settings.budgets.max_resident_bytes;
        let mut payload_transport = transport.clone();
        let mut runtime =
            LodStreamingRuntime::new(manifest.clone(), transport, &settings, &streaming).unwrap();
        assert_eq!(runtime.coverage_guard.pages.len(), 1);
        seed_resident_coverage_guard(&mut runtime, &manifest, &mut payload_transport, &streaming);
        let pages = runtime
            .hierarchy
            .manifest()
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .filter(|page| !runtime.coverage_guard.pages.contains(page))
            .take(3)
            .collect::<Vec<_>>();
        assert_eq!(pages.len(), 3);
        stage_ready_pages(
            &mut runtime,
            &manifest,
            &mut payload_transport,
            &streaming,
            &pages,
        );

        let frame = runtime.begin_frame();
        record_test_frame_demand(
            &mut runtime,
            frame,
            LodRuntimeViewId(1),
            pages.iter().copied(),
        );
        let completed = commit_test_ready_pages(&mut runtime, frame, &settings, &streaming);

        assert_eq!(completed, pages[..2]);
        assert_eq!(
            runtime.frame_completion_holds,
            completed.iter().copied().collect()
        );
        assert!(runtime.capacity_blocked.contains_key(&pages[2]));
        for page in &completed {
            let resident = runtime.cache.get(*page).unwrap();
            assert_eq!(resident.pin_count, 1);
            assert!(runtime.decoded_pages.contains_key(page));
        }

        runtime.finish_frame(frame).unwrap();
        assert!(runtime.frame_completion_holds.is_empty());
        assert!(
            completed
                .iter()
                .all(|page| runtime.cache.get(*page).unwrap().pin_count == 0)
        );
        assert!(!runtime.capacity_blocked.contains_key(&pages[2]));
        assert!(runtime.queue.contains(pages[2]));
    }

    #[test]
    fn completion_hold_spans_every_view_in_an_application_frame() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_resident_pages = 2;
        settings.budgets.max_upload_bytes_per_frame = settings.budgets.max_resident_bytes;
        let mut payload_transport = transport.clone();
        let mut runtime =
            LodStreamingRuntime::new(manifest.clone(), transport, &settings, &streaming).unwrap();
        assert_eq!(runtime.coverage_guard.pages.len(), 1);
        seed_resident_coverage_guard(&mut runtime, &manifest, &mut payload_transport, &streaming);
        let pages = runtime
            .hierarchy
            .manifest()
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .filter(|page| !runtime.coverage_guard.pages.contains(page))
            .take(2)
            .collect::<Vec<_>>();
        assert_eq!(pages.len(), 2);
        stage_ready_pages(
            &mut runtime,
            &manifest,
            &mut payload_transport,
            &streaming,
            &pages,
        );

        let frame = runtime.begin_frame();
        record_test_frame_demand(&mut runtime, frame, LodRuntimeViewId(11), [pages[0]]);
        assert_eq!(
            commit_test_ready_pages(&mut runtime, frame, &settings, &streaming),
            vec![pages[0]]
        );
        record_test_frame_demand(&mut runtime, frame, LodRuntimeViewId(12), [pages[1]]);
        assert!(commit_test_ready_pages(&mut runtime, frame, &settings, &streaming).is_empty());

        assert!(runtime.cache.contains(pages[0]));
        assert!(runtime.decoded_pages.contains_key(&pages[0]));
        assert!(runtime.capacity_blocked.contains_key(&pages[1]));
        assert_eq!(runtime.frame_completion_holds, BTreeSet::from([pages[0]]));

        runtime.finish_frame(frame).unwrap();
        assert_eq!(runtime.cache.get(pages[0]).unwrap().pin_count, 0);
        assert!(runtime.queue.contains(pages[1]));
    }

    #[test]
    fn beginning_the_next_frame_releases_omitted_finish_completion_holds() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_resident_pages = 2;
        settings.budgets.max_upload_bytes_per_frame = settings.budgets.max_resident_bytes;
        let mut payload_transport = transport.clone();
        let mut runtime =
            LodStreamingRuntime::new(manifest.clone(), transport, &settings, &streaming).unwrap();
        assert_eq!(runtime.coverage_guard.pages.len(), 1);
        seed_resident_coverage_guard(&mut runtime, &manifest, &mut payload_transport, &streaming);
        let page = runtime
            .hierarchy
            .manifest()
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .find(|page| !runtime.coverage_guard.pages.contains(page))
            .unwrap();
        stage_ready_pages(
            &mut runtime,
            &manifest,
            &mut payload_transport,
            &streaming,
            &[page],
        );

        let omitted = runtime.begin_frame();
        record_test_frame_demand(&mut runtime, omitted, LodRuntimeViewId(21), [page]);
        assert_eq!(
            commit_test_ready_pages(&mut runtime, omitted, &settings, &streaming),
            vec![page]
        );
        assert_eq!(runtime.cache.get(page).unwrap().pin_count, 1);

        let next = runtime.begin_frame();
        assert_ne!(next, omitted);
        assert!(runtime.frame_completion_holds.is_empty());
        assert_eq!(runtime.cache.get(page).unwrap().pin_count, 0);
        runtime.finish_frame(next).unwrap();
    }

    #[test]
    fn preprocessing_admission_success_and_capacity_are_deterministic() {
        assert!(matches!(
            LodPagePreprocessor::with_byte_capacity(0, 1),
            Err(LodPagePreprocessAdmissionError::ZeroCapacity)
        ));
        let (manifest, mut transport, _, streaming) = fixture();
        let pages = manifest
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .take(2)
            .collect::<Vec<_>>();
        let mut preprocessor = LodPagePreprocessor::new_cooperative_for_tests(1).unwrap();
        preprocessor
            .submit(memory_preprocess_input(
                &manifest,
                &mut transport,
                pages[0],
                streaming.effective_max_encoded_page_bytes(),
            ))
            .unwrap();
        assert_eq!(preprocessor.stats().capacity, 1);
        assert_eq!(preprocessor.stats().waiting, 1);
        let pending_bytes = preprocessor.stats().pending_bytes;
        assert!(pending_bytes > 0);
        assert!(matches!(
            preprocessor.submit(memory_preprocess_input(
                &manifest,
                &mut transport,
                pages[0],
                streaming.effective_max_encoded_page_bytes(),
            )),
            Err(LodPagePreprocessAdmissionError::DuplicatePage(duplicate)) if duplicate == pages[0]
        ));
        assert!(matches!(
            preprocessor.submit(memory_preprocess_input(
                &manifest,
                &mut transport,
                pages[1],
                streaming.effective_max_encoded_page_bytes(),
            )),
            Err(LodPagePreprocessAdmissionError::CapacityExhausted { capacity: 1 })
        ));

        let full_page_budget = NonZeroU32::new(u32::MAX).unwrap();
        preprocessor.advance(1, full_page_budget);
        assert_eq!(preprocessor.stats().ready, 0);
        assert_eq!(preprocessor.stats().submitted, 1);
        preprocessor.advance(2, full_page_budget);
        assert_eq!(preprocessor.stats().ready, 1);
        assert_eq!(preprocessor.stats().pending_bytes, pending_bytes);
        let output = preprocessor.take_ready(pages[0]).unwrap();
        assert_eq!(output.request.page_id, pages[0]);
        assert_eq!(output.result.unwrap().id, pages[0]);
        assert_eq!(preprocessor.len(), 0);
        assert_eq!(preprocessor.stats().pending_bytes, 0);
    }

    #[test]
    fn preprocessing_byte_admission_and_cancellation_are_exact() {
        let (manifest, mut transport, _, streaming) = fixture();
        let pages = manifest
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .take(2)
            .collect::<Vec<_>>();
        let first = memory_preprocess_input(
            &manifest,
            &mut transport,
            pages[0],
            streaming.effective_max_encoded_page_bytes(),
        );
        let second = memory_preprocess_input(
            &manifest,
            &mut transport,
            pages[1],
            streaming.effective_max_encoded_page_bytes(),
        );
        let first_bytes = first.pending_bytes().unwrap();
        let second_bytes = second.pending_bytes().unwrap();
        assert!(matches!(
            LodPagePreprocessor::new_cooperative_with_byte_capacity_for_tests(2, 0),
            Err(LodPagePreprocessAdmissionError::ZeroByteCapacity)
        ));
        let mut preprocessor =
            LodPagePreprocessor::new_cooperative_with_byte_capacity_for_tests(2, first_bytes)
                .unwrap();
        preprocessor.submit(first).unwrap();
        assert_eq!(preprocessor.stats().byte_capacity, first_bytes);
        assert_eq!(preprocessor.stats().pending_bytes, first_bytes);
        assert_eq!(
            preprocessor.submit(second),
            Err(
                LodPagePreprocessAdmissionError::PendingByteCapacityExceeded {
                    requested: second_bytes,
                    pending: first_bytes,
                    capacity: first_bytes,
                }
            )
        );
        assert!(preprocessor.cancel(pages[0]));
        assert_eq!(preprocessor.stats().pending_bytes, 0);
        assert_eq!(preprocessor.len(), 0);
    }

    #[test]
    fn cooperative_preprocessing_runs_at_most_one_page_slice_per_application_frame() {
        let (manifest, mut transport, _, streaming) = fixture();
        let pages = manifest
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .take(2)
            .collect::<Vec<_>>();
        let first = memory_preprocess_input(
            &manifest,
            &mut transport,
            pages[0],
            streaming.effective_max_encoded_page_bytes(),
        );
        let second = memory_preprocess_input(
            &manifest,
            &mut transport,
            pages[1],
            streaming.effective_max_encoded_page_bytes(),
        );
        let byte_capacity = first
            .pending_bytes()
            .unwrap()
            .checked_add(second.pending_bytes().unwrap())
            .unwrap();
        let mut preprocessor =
            LodPagePreprocessor::new_cooperative_with_byte_capacity_for_tests(2, byte_capacity)
                .unwrap();
        preprocessor.submit(first).unwrap();
        preprocessor.submit(second).unwrap();
        let one_record_budget = NonZeroU32::MIN;
        preprocessor.advance(41, one_record_budget);
        let first_slice = preprocessor.stats();
        preprocessor.advance(41, one_record_budget);
        assert_eq!(preprocessor.stats(), first_slice);
        assert_eq!(preprocessor.stats().ready, 0);
        assert_eq!(preprocessor.stats().waiting, 1);
        assert_eq!(preprocessor.stats().submitted, 1);

        let mut frame = 42;
        while preprocessor.stats().ready == 0 {
            let before = preprocessor.stats();
            preprocessor.advance(frame, one_record_budget);
            let after = preprocessor.stats();
            if before.submitted == 1 && after.submitted == 1 {
                assert!(
                    after.cooperative_decoded_gaussians
                        <= before.cooperative_decoded_gaussians.saturating_add(1)
                );
            }
            frame += 1;
            assert!(frame < 1_000, "bounded first page did not complete");
        }
        assert_eq!(preprocessor.stats().ready, 1);
        assert_eq!(preprocessor.stats().waiting, 1);
        assert_eq!(preprocessor.stats().submitted, 0);

        let completed_first_frame = frame - 1;
        let after_first_completion = preprocessor.stats();
        preprocessor.advance(completed_first_frame, one_record_budget);
        assert_eq!(preprocessor.stats(), after_first_completion);
        preprocessor.advance(frame, one_record_budget);
        assert_eq!(preprocessor.stats().ready, 1);
        assert_eq!(preprocessor.stats().waiting, 0);
        assert_eq!(preprocessor.stats().submitted, 1);

        frame += 1;
        while preprocessor.stats().ready < 2 {
            preprocessor.advance(frame, one_record_budget);
            frame += 1;
            assert!(frame < 2_000, "bounded second page did not complete");
        }
        assert_eq!(preprocessor.stats().ready, 2);
        assert_eq!(preprocessor.stats().waiting, 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_preprocessing_worker_pool_executes_the_production_backend() {
        let (manifest, mut transport, _, streaming) = fixture();
        let page = manifest.pages[0].id;
        let input = memory_preprocess_input(
            &manifest,
            &mut transport,
            page,
            streaming.effective_max_encoded_page_bytes(),
        );
        let pending_bytes = input.pending_bytes().unwrap();
        let mut preprocessor = LodPagePreprocessor::new_native_for_tests(1, pending_bytes).unwrap();
        assert_eq!(
            preprocessor.stats().backend,
            LodPagePreprocessBackend::NativeWorkerPool
        );
        preprocessor.submit(input).unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while preprocessor.stats().ready == 0 && std::time::Instant::now() < deadline {
            preprocessor.advance(1, NonZeroU32::MIN);
            std::thread::yield_now();
        }

        let output = preprocessor
            .take_ready(page)
            .expect("native preprocessing worker should complete within the test deadline");
        assert_eq!(output.result.unwrap().id, page);
        assert_eq!(preprocessor.stats().pending_bytes, 0);
    }

    #[test]
    fn malformed_preprocessing_payload_is_typed_and_bounded() {
        let (manifest, mut transport, _, streaming) = fixture();
        let page = manifest.pages[0].id;
        let mut input = memory_preprocess_input(
            &manifest,
            &mut transport,
            page,
            streaming.effective_max_encoded_page_bytes(),
        );
        input.payload.bytes[0] ^= 0xff;
        let mut preprocessor = LodPagePreprocessor::new_cooperative_for_tests(1).unwrap();
        preprocessor.submit(input).unwrap();
        preprocessor.advance(1, NonZeroU32::new(u32::MAX).unwrap());
        assert_eq!(
            preprocessor.take_ready(page).unwrap().result,
            Err(LodPagePreprocessError::PayloadChecksumMismatch)
        );
        assert_eq!(preprocessor.len(), 0);
    }

    #[test]
    fn cooperative_codec_failure_precedes_an_earlier_support_failure() {
        let (manifest, mut transport, _, streaming) = fixture();
        let page_id = manifest.pages[0].id;
        let mut input = memory_preprocess_input(
            &manifest,
            &mut transport,
            page_id,
            streaming.effective_max_encoded_page_bytes(),
        );
        let mut page = decode_page(&input.payload.bytes, input.limits).unwrap();
        page.gaussians[0].scale_opacity.scale.fill(f32::MAX);
        let mut encoded = encode_page(&page).unwrap();
        encoded[36] ^= 1;
        input.payload = PagePayload::new(page_id, encoded);

        let mut preprocessor = LodPagePreprocessor::new_cooperative_for_tests(1).unwrap();
        preprocessor.submit(input).unwrap();
        let mut frame = 1;
        while preprocessor.stats().ready == 0 {
            preprocessor.advance(frame, NonZeroU32::MIN);
            frame += 1;
            assert!(frame < 1_000, "adversarial page did not terminate");
        }
        assert!(matches!(
            preprocessor.take_ready(page_id).unwrap().result,
            Err(LodPagePreprocessError::Codec(
                LodCodecError::ChecksumMismatch { .. }
            ))
        ));
    }

    #[test]
    fn cancelled_preprocessing_never_publishes_a_stale_result() {
        let (manifest, mut transport, _, streaming) = fixture();
        let page = manifest.pages[0].id;
        let input = memory_preprocess_input(
            &manifest,
            &mut transport,
            page,
            streaming.effective_max_encoded_page_bytes(),
        );
        let mut preprocessor = LodPagePreprocessor::new_cooperative_for_tests(1).unwrap();
        let pending_bytes = input.pending_bytes().unwrap();
        preprocessor.submit(input).unwrap();
        let budget = NonZeroU32::MIN;
        preprocessor.advance(1, budget);
        let active = preprocessor.stats();
        assert_eq!(active.waiting, 0);
        assert_eq!(active.submitted, 1);
        assert_eq!(active.pending_bytes, pending_bytes);
        assert_eq!(active.cooperative_budget_gaussians_per_frame, 1);
        assert!(preprocessor.contains(page));
        assert!(preprocessor.cancel(page));
        preprocessor.advance(2, budget);
        assert!(preprocessor.take_ready(page).is_none());
        assert_eq!(preprocessor.stats().cancellations, 1);
        assert_eq!(preprocessor.stats().pending_bytes, 0);
        assert_eq!(preprocessor.len(), 0);
    }

    #[test]
    fn runtime_retains_typed_preprocess_failure_through_terminal_retry_state() {
        let (manifest, mut transport, mut settings, mut streaming) = fixture();
        settings.quality = 0.0;
        settings.budgets.max_active_gaussians = root_active_gaussians(&manifest);
        settings.budgets.max_requests_per_frame = 1;
        streaming.retry_limit = 0;
        let page = root_page(&manifest);
        let encoded_len = manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == page)
            .and_then(|descriptor| descriptor.storage.as_ref())
            .unwrap()
            .encoded_len as usize;
        transport.insert(page, vec![0; encoded_len]);
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        runtime.update(view(), &settings, &streaming).unwrap();
        let failed = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(failed.preprocess_failed_pages(), &[page]);
        assert_eq!(failed.failed_pages(), &[page]);
        assert!(matches!(
            runtime.page_preprocess_error(page),
            Some(LodPagePreprocessError::Codec(_))
        ));
        assert!(runtime.is_terminal_failure(page));
    }

    #[test]
    fn terminal_guard_failure_is_surfaced_before_ordinary_streaming_resumes() {
        let (manifest, mut transport, settings, mut streaming) = fixture();
        streaming.retry_limit = 0;
        let hierarchy = CompiledManifestLodHierarchy::new(manifest.clone()).unwrap();
        let guard = LodRuntimeCoverageGuard::new(&hierarchy, &settings).unwrap();
        assert_eq!(guard.pages.len(), 1);
        let guard_page = *guard.pages.first().unwrap();
        let encoded_len = manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == guard_page)
            .and_then(|descriptor| descriptor.storage.as_ref())
            .unwrap()
            .encoded_len as usize;
        transport.insert(guard_page, vec![0; encoded_len]);

        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        runtime.preprocessor = LodPagePreprocessor::new_cooperative_for_tests(8).unwrap();
        let mut surfaced = false;
        let mut ordinary_completed = false;
        for _ in 0..256 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            surfaced |= frame.failed_pages().contains(&guard_page);
            ordinary_completed |= frame
                .completed_pages()
                .iter()
                .any(|page| *page != guard_page);
            if surfaced && ordinary_completed {
                break;
            }
        }

        assert!(surfaced, "the terminal guard failure must be observable");
        assert!(runtime.is_terminal_failure(guard_page));
        assert!(
            ordinary_completed,
            "a surfaced terminal guard failure must not starve ordinary streaming forever"
        );
    }

    #[test]
    fn updates_are_rejected_after_frame_demand_is_reconciled() {
        let (manifest, transport, settings, streaming) = fixture();
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let frame = runtime.begin_frame();
        runtime.finish_frame(frame).unwrap();
        assert_eq!(
            runtime
                .update_view_in_frame(frame, LodRuntimeViewId(5), view(), &settings, &streaming,),
            Err(LodRuntimeError::FrameAlreadyFinished(frame))
        );
    }

    #[test]
    fn shared_physical_page_validates_each_logical_node_slice() {
        let gaussian = |x| Gaussian3d {
            position_visibility: [x, 0.0, 0.0, 1.0].into(),
            spherical_harmonic: SphericalHarmonicCoefficients::default(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.1, 0.1, 0.1, 1.0].into(),
        };
        let page_id = LodPageId(1);
        let page = PlanarGaussian3dPage::new(page_id, vec![gaussian(-1.0), gaussian(1.0)]);
        let left = gaussian_support_bounds(&page.gaussians[0], 3.0).unwrap();
        let right = gaussian_support_bounds(&page.gaussians[1], 3.0).unwrap();
        let ranges = [
            SharedPageNodeRange {
                node: LodNodeId(1),
                range: LodPageRange {
                    page: page_id,
                    offset: 0,
                    count: 1,
                },
                bounds: left,
            },
            SharedPageNodeRange {
                node: LodNodeId(2),
                range: LodPageRange {
                    page: page_id,
                    offset: 1,
                    count: 1,
                },
                bounds: right,
            },
        ];
        assert_eq!(
            validate_shared_page_node_ranges(&page, &ranges, 3.0),
            Ok(())
        );

        let swapped = [
            SharedPageNodeRange {
                bounds: right,
                ..ranges[0]
            },
            SharedPageNodeRange {
                bounds: left,
                ..ranges[1]
            },
        ];
        assert_eq!(
            validate_shared_page_node_ranges(&page, &swapped, 3.0),
            Err(LodPagePreprocessError::PayloadOutsideNodeBounds {
                page: page_id,
                node: LodNodeId(1),
            })
        );
    }

    fn two_root_fixture() -> (
        GaussianLodManifest,
        MemoryPageTransport,
        GaussianLodSettings,
        GaussianStreamingSettings,
    ) {
        let gaussian = |x| Gaussian3d {
            position_visibility: [x, 0.0, 0.0, 1.0].into(),
            spherical_harmonic: SphericalHarmonicCoefficients::default(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.1, 0.1, 0.1, 1.0].into(),
        };
        let cloud: PlanarGaussian3d = vec![gaussian(-1.0), gaussian(1.0)].into();
        let mut lod = build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                branching_factor: 2,
                leaf_capacity: 1,
                support_sigma: 3.0,
            },
        )
        .unwrap();

        let original_root = lod.manifest.roots[0];
        let root_page = lod
            .manifest
            .nodes
            .iter()
            .find(|node| node.id == original_root)
            .unwrap()
            .representation
            .page;
        lod.manifest.nodes.retain(|node| node.id != original_root);
        lod.manifest.pages.retain(|page| page.id != root_page);
        lod.pages.retain(|page| page.id != root_page);
        for node in &mut lod.manifest.nodes {
            node.parent = None;
            node.depth = 0;
            node.quality.min = 0.0;
        }
        lod.manifest.roots = lod.manifest.nodes.iter().map(|node| node.id).collect();
        lod.manifest.header.node_count = lod.manifest.nodes.len() as u32;
        lod.manifest.header.page_count = lod.manifest.pages.len() as u32;
        lod.manifest.header.stored_gaussian_count = lod
            .manifest
            .pages
            .iter()
            .map(|page| u64::from(page.gaussian_count))
            .sum();
        lod.manifest.quality = GaussianLodQualityMetadata {
            max_depth: 0,
            coarsest_gaussian_count: 2,
            finest_gaussian_count: 2,
            max_error: lod
                .manifest
                .nodes
                .iter()
                .fold(LodError::ZERO, |error, node| error.max(node.error)),
        };

        let mut transport = MemoryPageTransport::default();
        for page in &lod.pages {
            let encoded = encode_page(page).unwrap();
            let descriptor = lod
                .manifest
                .pages
                .iter_mut()
                .find(|descriptor| descriptor.id == page.id)
                .unwrap();
            descriptor.storage = Some(LodPageStorage {
                uri: format!("memory://two-root-{}", page.id.0),
                byte_range: None,
                encoded_len: encoded.len() as u64,
            });
            transport.insert(page.id, encoded);
        }
        lod.validate().unwrap();

        let mut settings = GaussianLodSettings::default();
        settings.quality = 0.0;
        settings.budgets.max_requests_per_frame = 1;
        let streaming = GaussianStreamingSettings {
            max_concurrent_requests: 1,
            ..Default::default()
        };
        (lod.manifest, transport, settings, streaming)
    }

    fn view() -> LodView {
        LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, 8.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        )
    }

    fn root_page(manifest: &GaussianLodManifest) -> LodPageId {
        let root = manifest.roots[0];
        manifest
            .nodes
            .iter()
            .find(|node| node.id == root)
            .expect("fixture root must exist")
            .representation
            .page
    }

    fn root_active_gaussians(manifest: &GaussianLodManifest) -> u64 {
        manifest
            .roots
            .iter()
            .map(|root| {
                u64::from(
                    manifest
                        .nodes
                        .iter()
                        .find(|node| node.id == *root)
                        .expect("fixture root must exist")
                        .representation
                        .count,
                )
            })
            .sum()
    }

    fn first_level_promoted_forest(mut manifest: GaussianLodManifest) -> GaussianLodManifest {
        let original_root = manifest.roots[0];
        let root_index = manifest
            .nodes
            .iter()
            .position(|node| node.id == original_root)
            .unwrap();
        assert_eq!(root_index, 0, "promoted manifests are breadth-first");
        let root = manifest.nodes[root_index].clone();
        let child_start = root.children.start as usize;
        let child_end = root.children.end().unwrap() as usize;
        let roots = manifest.nodes[child_start..child_end]
            .iter()
            .map(|node| node.id)
            .collect::<Vec<_>>();
        assert!(roots.len() > 1);

        manifest.nodes.remove(root_index);
        for node in &mut manifest.nodes {
            if node.children.count > 0 {
                node.children.start -= 1;
            }
            node.depth -= 1;
            if node.parent == Some(original_root) {
                node.parent = None;
                node.quality.min = 0.0;
            }
        }
        let root_page_index = manifest
            .pages
            .iter()
            .position(|page| page.id == root.representation.page)
            .unwrap();
        let root_page = manifest.pages.remove(root_page_index);
        manifest.roots = roots;
        manifest.header.node_count -= 1;
        manifest.header.page_count -= 1;
        manifest.header.stored_gaussian_count -= u64::from(root_page.gaussian_count);
        manifest.quality.max_depth -= 1;
        manifest.quality.coarsest_gaussian_count = manifest
            .roots
            .iter()
            .map(|root| {
                u64::from(
                    manifest
                        .nodes
                        .iter()
                        .find(|node| node.id == *root)
                        .unwrap()
                        .representation
                        .count,
                )
            })
            .sum();
        manifest.quality.max_error = manifest.roots.iter().fold(LodError::ZERO, |error, root| {
            error.max(
                manifest
                    .nodes
                    .iter()
                    .find(|node| node.id == *root)
                    .unwrap()
                    .error,
            )
        });
        manifest.validate().unwrap();
        manifest
    }

    fn disjoint_two_root_guard_hierarchy() -> CompiledManifestLodHierarchy {
        let gaussian = |x| Gaussian3d {
            position_visibility: [x, 0.0, 0.0, 1.0].into(),
            spherical_harmonic: SphericalHarmonicCoefficients::default(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.1, 0.1, 0.1, 1.0].into(),
        };
        let cloud: PlanarGaussian3d =
            vec![gaussian(-3.0), gaussian(-1.0), gaussian(1.0), gaussian(3.0)].into();
        let lod = build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                branching_factor: 2,
                leaf_capacity: 1,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        let mut manifest = first_level_promoted_forest(lod.manifest);
        let roots = manifest.roots.clone();
        assert_eq!(roots.len(), 2);
        let children = roots
            .iter()
            .map(|root| {
                let node = manifest.nodes.iter().find(|node| node.id == *root).unwrap();
                let start = node.children.start as usize;
                let end = node.children.end().unwrap() as usize;
                manifest.nodes[start..end]
                    .iter()
                    .map(|child| child.id)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(children.iter().all(|children| children.len() == 2));

        let root_pages = [LodPageId(10_001), LodPageId(10_002)];
        let child_pages = [LodPageId(10_003), LodPageId(10_004)];
        let mut pages = Vec::with_capacity(4);
        for (ordinal, root) in roots.iter().enumerate() {
            let node = manifest
                .nodes
                .iter_mut()
                .find(|node| node.id == *root)
                .unwrap();
            node.representation.page = root_pages[ordinal];
            node.representation.offset = 0;
            let gaussian_count = node.representation.count;
            pages.push(LodPageDescriptor {
                id: root_pages[ordinal],
                kind: LodPageKind::Representatives,
                encoding: LodPageEncoding::F32Planar,
                gaussian_count,
                decoded_len: u64::from(gaussian_count) * std::mem::size_of::<Gaussian3d>() as u64,
                content_hash: 0x9e37_79b9_7f4a_7c15_u64 ^ root_pages[ordinal].0,
                bounds: node.bounds,
                storage: None,
            });
        }
        for (ordinal, child_ids) in children.iter().enumerate() {
            let mut offset = 0_u32;
            let mut bounds: Option<LodBounds> = None;
            for child_id in child_ids {
                let node = manifest
                    .nodes
                    .iter_mut()
                    .find(|node| node.id == *child_id)
                    .unwrap();
                node.representation.page = child_pages[ordinal];
                node.representation.offset = offset;
                offset += node.representation.count;
                bounds = Some(match bounds {
                    Some(bounds) => bounds.union(node.bounds),
                    None => node.bounds,
                });
            }
            pages.push(LodPageDescriptor {
                id: child_pages[ordinal],
                kind: LodPageKind::SourceLeaves,
                encoding: LodPageEncoding::F32Planar,
                gaussian_count: offset,
                decoded_len: u64::from(offset) * std::mem::size_of::<Gaussian3d>() as u64,
                content_hash: 0x9e37_79b9_7f4a_7c15_u64 ^ child_pages[ordinal].0,
                bounds: bounds.unwrap(),
                storage: None,
            });
        }
        manifest.pages = pages;
        manifest.header.required_features |= LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES;
        manifest.header.page_count = manifest.pages.len() as u32;
        manifest.header.stored_gaussian_count = manifest
            .pages
            .iter()
            .map(|page| u64::from(page.gaussian_count))
            .sum();
        manifest.validate().unwrap();
        CompiledManifestLodHierarchy::new(manifest).unwrap()
    }

    /// Builds only breadth-first topology and page descriptors for the lazy
    /// virtual city. The 32,768 leaf page counts remain logical: this helper
    /// materializes and encodes exactly one Gaussian, the root representative.
    fn virtual_runtime_fixture() -> VirtualRuntimeFixture {
        let city = VirtualCityScene::default();
        assert_eq!(city.page_count, VIRTUAL_LEVEL_COUNTS[3]);

        let root_id = LodNodeId(1);
        let root_page_id = LodPageId(1);
        let root_page = PlanarGaussian3dPage::new(
            root_page_id,
            vec![Gaussian3d {
                position_visibility: [0.0, 0.0, 0.0, 1.0].into(),
                spherical_harmonic: SphericalHarmonicCoefficients::default(),
                rotation: [1.0, 0.0, 0.0, 0.0].into(),
                scale_opacity: [0.25, 0.25, 0.25, 1.0].into(),
            }],
        );
        let encoded_root = encode_page(&root_page).unwrap();
        let bounds = LodBounds::new([-1.0; 3], [1.0; 3]).unwrap();
        let build_settings = GaussianLodBuildSettings {
            branching_factor: VIRTUAL_BRANCHING_FACTOR as u8,
            leaf_capacity: city.gaussians_per_page,
            support_sigma: 3.0,
        };
        let mut nodes = Vec::with_capacity(VIRTUAL_NODE_COUNT as usize);
        let mut pages = Vec::with_capacity(VIRTUAL_NODE_COUNT as usize);
        let mut stored_gaussian_count = 0_u64;

        for depth in 0..=VIRTUAL_TREE_DEPTH {
            let depth_index = usize::from(depth);
            let level_start = VIRTUAL_LEVEL_STARTS[depth_index];
            let level_count = VIRTUAL_LEVEL_COUNTS[depth_index];
            let descendant_leaf_count =
                VIRTUAL_BRANCHING_FACTOR.pow(u32::from(VIRTUAL_TREE_DEPTH - depth));
            for ordinal in 0..level_count {
                let flat_index = level_start + ordinal;
                let node_id = LodNodeId(u64::from(flat_index) + 1);
                let page_id = LodPageId(node_id.0);
                let first_leaf = u64::from(ordinal) * u64::from(descendant_leaf_count);
                let source = LodSourceRange {
                    start: first_leaf * u64::from(city.gaussians_per_page),
                    count: u64::from(descendant_leaf_count) * u64::from(city.gaussians_per_page),
                };
                let is_leaf = depth == VIRTUAL_TREE_DEPTH;
                let gaussian_count = if is_leaf { city.gaussians_per_page } else { 1 };
                let geometric_error = f32::from(VIRTUAL_TREE_DEPTH - depth);
                let error = LodError {
                    geometric: geometric_error,
                    appearance: 0.0,
                    opacity: 0.0,
                    combined: geometric_error,
                };
                let quality = LodQualityInterval {
                    min: f32::from(depth) / f32::from(VIRTUAL_TREE_DEPTH),
                    max: if is_leaf {
                        1.0
                    } else {
                        f32::from(depth + 1) / f32::from(VIRTUAL_TREE_DEPTH)
                    },
                };
                nodes.push(GaussianLodNode {
                    id: node_id,
                    parent: (depth > 0).then(|| {
                        let parent_index = VIRTUAL_LEVEL_STARTS[depth_index - 1]
                            + ordinal / VIRTUAL_BRANCHING_FACTOR;
                        LodNodeId(u64::from(parent_index) + 1)
                    }),
                    depth,
                    bounds,
                    children: if is_leaf {
                        LodIndexRange::empty()
                    } else {
                        LodIndexRange {
                            start: VIRTUAL_LEVEL_STARTS[depth_index + 1]
                                + ordinal * VIRTUAL_BRANCHING_FACTOR,
                            count: VIRTUAL_BRANCHING_FACTOR,
                        }
                    },
                    source,
                    morton: LodMortonRange {
                        min: first_leaf,
                        max: first_leaf + u64::from(descendant_leaf_count) - 1,
                    },
                    representation: LodPageRange {
                        page: page_id,
                        offset: 0,
                        count: gaussian_count,
                    },
                    error,
                    quality,
                    high_fidelity_certificate: if is_leaf { 1.0 } else { 0.0 },
                });
                pages.push(LodPageDescriptor {
                    id: page_id,
                    kind: if is_leaf {
                        LodPageKind::SourceLeaves
                    } else {
                        LodPageKind::Representatives
                    },
                    encoding: LodPageEncoding::F32Planar,
                    gaussian_count,
                    decoded_len: u64::from(gaussian_count)
                        * std::mem::size_of::<Gaussian3d>() as u64,
                    content_hash: if page_id == root_page_id {
                        root_page.content_hash()
                    } else {
                        0x9e37_79b9_7f4a_7c15_u64 ^ page_id.0
                    },
                    bounds,
                    storage: (page_id == root_page_id).then(|| LodPageStorage {
                        uri: "memory://virtual-city-root".to_owned(),
                        byte_range: None,
                        encoded_len: encoded_root.len() as u64,
                    }),
                });
                stored_gaussian_count += u64::from(gaussian_count);
            }
        }
        assert_eq!(nodes.len(), VIRTUAL_NODE_COUNT as usize);
        assert_eq!(pages.len(), VIRTUAL_NODE_COUNT as usize);

        let source_gaussian_count = city.source_gaussian_count();
        let root_error = nodes[0].error;
        let manifest = GaussianLodManifest {
            header: GaussianLodManifestHeader {
                magic: LOD_MANIFEST_MAGIC,
                manifest_version: LOD_MANIFEST_VERSION,
                page_schema_version: LOD_PAGE_SCHEMA_VERSION,
                required_features: LOD_CURRENT_REQUIRED_FEATURES,
                source_gaussian_count,
                stored_gaussian_count,
                node_count: VIRTUAL_NODE_COUNT,
                page_count: VIRTUAL_NODE_COUNT,
            },
            scene_bounds: Some(bounds),
            roots: vec![root_id],
            nodes,
            pages,
            build: GaussianLodBuildMetadata {
                settings: build_settings,
                reducer: LodReducerKind::MomentMerge,
                builder_abi_version: VIRTUAL_BUILDER_ABI_VERSION,
                reducer_version: EXTERNAL_MOMENT_MERGE_VERSION,
                source_fingerprint: 0x4f9a_2be3_a561_903d,
                config_fingerprint: lod_config_fingerprint_for_reducer(
                    build_settings,
                    None,
                    EXTERNAL_MOMENT_MERGE_VERSION,
                ),
            },
            quality: GaussianLodQualityMetadata {
                max_depth: VIRTUAL_TREE_DEPTH,
                coarsest_gaussian_count: 1,
                finest_gaussian_count: source_gaussian_count,
                max_error: root_error,
            },
            morph_map: None,
        };
        manifest.validate().unwrap();

        let encoded_root_bytes = encoded_root.len();
        let mut transport = MemoryPageTransport::default();
        assert!(transport.insert(root_page_id, encoded_root).is_none());

        let mut lod_settings = GaussianLodSettings::default();
        lod_settings.quality = 0.0;
        lod_settings.budgets.max_active_gaussians = 2;
        lod_settings.budgets.max_resident_gaussians = u64::from(city.gaussians_per_page);
        let max_page_decoded_bytes =
            u64::from(city.gaussians_per_page) * std::mem::size_of::<Gaussian3d>() as u64;
        lod_settings.budgets.max_resident_bytes = max_page_decoded_bytes;
        lod_settings.budgets.max_resident_pages = 2;
        lod_settings.budgets.max_pending_requests = 2;
        lod_settings.budgets.max_requests_per_frame = 1;
        lod_settings.budgets.max_upload_bytes_per_frame = max_page_decoded_bytes;
        lod_settings.budgets.max_traversal_nodes_per_view = 4;
        let streaming_settings = GaussianStreamingSettings {
            max_concurrent_requests: 1,
            max_compressed_cache_bytes: 4 * 1024,
            ..Default::default()
        };

        VirtualRuntimeFixture {
            manifest,
            transport,
            lod_settings,
            streaming_settings,
            encoded_root_bytes,
        }
    }

    fn virtual_forest_runtime() -> LodStreamingRuntime<MemoryPageTransport> {
        let fixture = virtual_runtime_fixture();
        let manifest = first_level_promoted_forest(fixture.manifest);
        let mut settings = fixture.lod_settings;
        settings.budgets.max_active_gaussians = 1_000_000;
        settings.budgets.max_resident_pages = 64;
        settings.budgets.max_pending_requests = 128;
        settings.budgets.max_requests_per_frame = 64;
        let mut runtime = LodStreamingRuntime::new(
            manifest,
            MemoryPageTransport::default(),
            &settings,
            &fixture.streaming_settings,
        )
        .unwrap();
        runtime.queue.clear();
        runtime
    }

    #[test]
    fn streams_from_roots_to_exact_frontier_without_holes() {
        let (manifest, transport, settings, streaming) = fixture();
        let source_count = manifest.header.source_gaussian_count;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();

        let first = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(first.candidate_count(), 0);
        assert!(!first.started_pages.is_empty());

        let mut final_frame = None;
        for _ in 0..64 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert_eq!(
                frame.candidate_count(),
                frame.frontier.status.active_gaussians
            );
            assert!(frame.failed_pages.is_empty());
            if frame.frontier.requested_nodes.is_empty()
                && frame.in_flight_requests == 0
                && frame.queued_requests == 0
            {
                final_frame = Some(frame);
                break;
            }
        }
        let final_frame = final_frame.expect("fixture should become fully resident");
        assert_eq!(final_frame.candidate_count(), source_count);
    }

    #[test]
    fn split_cohort_capacity_reserves_pending_active_guard_pages() {
        let (hierarchy, transport, settings, budget) = package_bootstrap_fixture();
        let streaming = GaussianStreamingSettings::default();
        let mut runtime = LodStreamingRuntime::from_compiled_hierarchy(
            hierarchy,
            transport,
            &settings,
            &streaming,
            Some(budget),
        )
        .unwrap();
        let guard_pages = runtime.coverage_guard.pages.clone();
        assert!(runtime.coverage_guard.is_active());
        assert!(guard_pages.len() >= 2);

        let (parent, cohort_pages) = runtime
            .hierarchy
            .manifest()
            .nodes
            .iter()
            .filter(|node| !node.children.is_empty())
            .find_map(|node| {
                let pages = runtime
                    .hierarchy
                    .children(node.id)
                    .iter()
                    .filter_map(|child| runtime.hierarchy.page(*child))
                    .collect::<BTreeSet<_>>();
                (guard_pages.difference(&pages).count() >= 2).then_some((node.id, pages))
            })
            .expect("fixture must expose a cohort distinct from two guard pages");
        let plan = LodSplitCohortPlan {
            view: LodRuntimeViewId::default(),
            parent,
            pages: cohort_pages,
        };
        let mut complete_union = guard_pages.clone();
        complete_union.extend(plan.pages.iter().copied());
        let limit_pages = u32::try_from(complete_union.len() - 1).unwrap();
        runtime.cache = LodPageCache::new(PageCacheLimits {
            max_pages: limit_pages,
            max_bytes: settings.budgets.max_resident_bytes,
            max_gaussians: settings.budgets.max_resident_gaussians,
        })
        .unwrap();

        let resident_guard = *guard_pages.first().unwrap();
        let descriptor = runtime.hierarchy.page_descriptor(resident_guard).unwrap();
        runtime
            .cache
            .insert(
                resident_guard,
                descriptor.decoded_len,
                u64::from(descriptor.gaussian_count),
                0,
            )
            .unwrap();
        runtime.pin_coverage_guard_page(resident_guard).unwrap();
        assert_eq!(runtime.coverage_guard.pinned_pages.len(), 1);

        let (fits, stall) = runtime.split_cohort_capacity(&plan).unwrap();
        assert!(!fits);
        assert_eq!(stall.required_pages, complete_union.len() as u64);
        assert_eq!(stall.limit_pages, u64::from(limit_pages));

        runtime.release_package_bootstrap_reserve().unwrap();
        let (fits_after_release, _) = runtime.split_cohort_capacity(&plan).unwrap();
        assert!(
            fits_after_release,
            "released package bootstraps must not retain pending guard capacity"
        );
    }

    #[test]
    fn pressured_forest_streams_missing_roots_before_non_root_cohorts() {
        let mut runtime = virtual_forest_runtime();

        let roots = runtime.hierarchy.roots();
        assert!(roots.len() >= 2);
        let missing_root = roots[0];
        let resident_parent = roots[1];
        let children = runtime.hierarchy.children(resident_parent).to_vec();
        assert!(!children.is_empty());
        let child_pages = children
            .iter()
            .filter_map(|child| runtime.hierarchy.page(*child))
            .collect::<BTreeSet<_>>();
        let missing_root_page = runtime.hierarchy.page(missing_root).unwrap();
        assert!(!child_pages.contains(&missing_root_page));
        let blocked_child = *child_pages.first().unwrap();
        runtime.capacity_blocked.insert(
            blocked_child,
            PageRequest::new(blocked_child, PageRequestPriority::visible(u32::MAX)),
        );

        let frame = runtime.begin_frame();
        let mut requested_nodes = vec![missing_root];
        requested_nodes.extend(children.iter().copied());
        let mixed = LodFrontier {
            nodes: vec![resident_parent],
            requested_nodes,
            status: LodEffectiveStatus::default(),
        };
        runtime
            .observe_split_cohort_candidates(frame, LodRuntimeViewId::default(), &mixed)
            .unwrap();
        runtime.record_frame_demand(frame, LodRuntimeViewId::default(), &mixed);
        let state = &runtime.views[&LodRuntimeViewId::default()];
        assert!(state.split_cohort_pressure);
        assert!(state.split_cohort_candidates.is_empty());
        assert_eq!(state.admitted_pages, BTreeSet::from([missing_root_page]));

        runtime.enqueue_missing_roots(&mixed, view()).unwrap();
        assert!(runtime.queue.contains(missing_root_page));
        assert!(
            child_pages
                .iter()
                .all(|page| !runtime.queue.contains(*page)),
            "pressured child pages must not enter the ordinary root path"
        );

        runtime.queue.remove(missing_root_page);
        let roots_resident = LodFrontier {
            nodes: vec![resident_parent],
            requested_nodes: children,
            status: LodEffectiveStatus::default(),
        };
        runtime
            .observe_split_cohort_candidates(frame, LodRuntimeViewId::default(), &roots_resident)
            .unwrap();
        runtime.record_frame_demand(frame, LodRuntimeViewId::default(), &roots_resident);
        assert_eq!(
            runtime.views[&LodRuntimeViewId::default()]
                .split_cohort_candidates
                .len(),
            1
        );
        runtime.schedule_next_split_cohort(frame).unwrap();
        let active = runtime.active_split_cohort.as_ref().unwrap();
        assert_eq!(active.plan.parent, resident_parent);
        assert_eq!(active.plan.pages, child_pages);
    }

    #[test]
    fn split_cohort_capacity_includes_other_views_current_admission() {
        let mut runtime = virtual_forest_runtime();
        runtime.coverage_guard.package_bootstrap = true;
        runtime.release_package_bootstrap_reserve().unwrap();
        assert!(!runtime.coverage_guard.is_active());

        let roots = runtime.hierarchy.roots();
        let direct_view = LodRuntimeViewId(1);
        let split_view = LodRuntimeViewId(2);
        let direct_root_page = runtime.hierarchy.page(roots[0]).unwrap();
        let parent = roots[1];
        let parent_page = runtime.hierarchy.page(parent).unwrap();
        let cohort_pages = runtime
            .hierarchy
            .children(parent)
            .iter()
            .filter_map(|child| runtime.hierarchy.page(*child))
            .collect::<BTreeSet<_>>();
        assert!(!cohort_pages.contains(&direct_root_page));
        let mut split_only_union = cohort_pages.clone();
        split_only_union.insert(parent_page);
        let limit_pages = u32::try_from(split_only_union.len()).unwrap();
        let old_limits = runtime.cache.limits();
        runtime.cache = LodPageCache::new(PageCacheLimits {
            max_pages: limit_pages,
            max_bytes: old_limits.max_bytes,
            max_gaussians: old_limits.max_gaussians,
        })
        .unwrap();
        let descriptor = runtime.hierarchy.page_descriptor(parent_page).unwrap();
        runtime
            .cache
            .insert(
                parent_page,
                descriptor.decoded_len,
                u64::from(descriptor.gaussian_count),
                0,
            )
            .unwrap();
        runtime.cache.pin_fallback(parent_page).unwrap();

        let frame = runtime.begin_frame();
        let direct = runtime.views.entry(direct_view).or_default();
        direct.admitted_pages = BTreeSet::from([direct_root_page]);
        direct.admitted_pages_frame = frame;
        let plan = LodSplitCohortPlan {
            view: split_view,
            parent,
            pages: cohort_pages.clone(),
        };
        let split = runtime.views.entry(split_view).or_default();
        split.pinned_frontier = BTreeSet::from([parent_page]);
        split.split_cohort_candidates = vec![plan.clone()];
        split.split_cohort_candidates_frame = frame;
        split.split_cohort_pressure = true;
        split.split_cohort_pressure_frame = frame;

        runtime.schedule_next_split_cohort(frame).unwrap();
        assert!(runtime.active_split_cohort.is_none());
        let stall = runtime.split_cohort_capacity_stall.unwrap();
        assert_eq!(
            stall.required_pages,
            u64::try_from(split_only_union.len() + 1).unwrap()
        );
        assert_eq!(stall.limit_pages, u64::from(limit_pages));
        assert_eq!(
            runtime.views[&direct_view].admitted_pages,
            BTreeSet::from([direct_root_page])
        );
        assert_eq!(
            runtime.queue.enqueue(PageRequest::new(
                direct_root_page,
                PageRequestPriority::fallback_critical(u32::MAX),
            )),
            RequestEnqueue::Enqueued
        );
        runtime.finish_frame(frame).unwrap();
        assert!(runtime.queue.contains(direct_root_page));
        assert!(
            cohort_pages
                .iter()
                .all(|page| !runtime.queue.contains(*page))
        );

        let next = runtime.begin_frame();
        let split = runtime.views.get_mut(&split_view).unwrap();
        split.split_cohort_candidates_frame = next;
        split.split_cohort_pressure_frame = next;
        runtime.schedule_next_split_cohort(next).unwrap();
        assert_eq!(runtime.active_split_cohort.as_ref().unwrap().plan, plan);
        runtime.finish_frame(next).unwrap();
        assert!(!runtime.queue.contains(direct_root_page));
        assert!(
            cohort_pages
                .iter()
                .all(|page| runtime.queue.contains(*page))
        );
    }

    #[test]
    fn active_cohort_preempts_when_new_root_admission_exceeds_its_proof() {
        let mut runtime = virtual_forest_runtime();
        runtime.coverage_guard.package_bootstrap = true;
        runtime.release_package_bootstrap_reserve().unwrap();

        let roots = runtime.hierarchy.roots();
        let split_view = LodRuntimeViewId(1);
        let root_view = LodRuntimeViewId(2);
        let root_page = runtime.hierarchy.page(roots[0]).unwrap();
        let parent = roots[1];
        let parent_page = runtime.hierarchy.page(parent).unwrap();
        let children = runtime.hierarchy.children(parent).to_vec();
        let cohort_pages = children
            .iter()
            .filter_map(|child| runtime.hierarchy.page(*child))
            .collect::<BTreeSet<_>>();
        let resident_child = *cohort_pages.first().unwrap();
        let mut admitted_union = cohort_pages.clone();
        admitted_union.insert(parent_page);
        let limit_pages = u32::try_from(admitted_union.len()).unwrap();
        let old_limits = runtime.cache.limits();
        runtime.cache = LodPageCache::new(PageCacheLimits {
            max_pages: limit_pages,
            max_bytes: old_limits.max_bytes,
            max_gaussians: old_limits.max_gaussians,
        })
        .unwrap();
        for page in [parent_page, resident_child] {
            let descriptor = runtime.hierarchy.page_descriptor(page).unwrap();
            runtime
                .cache
                .insert(
                    page,
                    descriptor.decoded_len,
                    u64::from(descriptor.gaussian_count),
                    0,
                )
                .unwrap();
        }
        runtime.cache.pin_fallback(parent_page).unwrap();
        let plan = LodSplitCohortPlan {
            view: split_view,
            parent,
            pages: cohort_pages.clone(),
        };

        let first = runtime.begin_frame();
        let split = runtime.views.entry(split_view).or_default();
        split.pinned_frontier = BTreeSet::from([parent_page]);
        split.split_cohort_candidates = vec![plan.clone()];
        split.split_cohort_candidates_frame = first;
        split.split_cohort_pressure = true;
        split.split_cohort_pressure_frame = first;
        runtime.schedule_next_split_cohort(first).unwrap();
        assert_eq!(runtime.active_split_cohort.as_ref().unwrap().plan, plan);
        assert_eq!(runtime.cache.get(resident_child).unwrap().pin_count, 1);
        runtime.finish_frame(first).unwrap();

        let second = runtime.begin_frame();
        let split = runtime.views.get_mut(&split_view).unwrap();
        split.split_cohort_candidates_frame = second;
        split.split_cohort_pressure_frame = second;
        runtime
            .active_split_cohort
            .as_mut()
            .unwrap()
            .owner_updated_frame = second;
        let split_frontier = LodFrontier {
            nodes: vec![parent],
            requested_nodes: children,
            status: LodEffectiveStatus::default(),
        };
        runtime.record_frame_demand(second, split_view, &split_frontier);
        let root = runtime.views.entry(root_view).or_default();
        root.admitted_pages = BTreeSet::from([root_page]);
        root.admitted_pages_frame = second;
        assert_eq!(
            runtime.queue.enqueue(PageRequest::new(
                root_page,
                PageRequestPriority::fallback_critical(u32::MAX),
            )),
            RequestEnqueue::Enqueued
        );

        runtime.schedule_next_split_cohort(second).unwrap();
        assert!(runtime.active_split_cohort.is_none());
        assert!(runtime.split_cohort_capacity_stall.is_some());
        assert!(runtime.views[&split_view].admitted_pages.is_empty());
        assert_eq!(
            runtime.views[&root_view].admitted_pages,
            BTreeSet::from([root_page])
        );
        assert_eq!(runtime.cache.get(resident_child).unwrap().pin_count, 0);

        runtime.finish_frame(second).unwrap();
        assert!(runtime.queue.contains(root_page));
        assert!(
            cohort_pages
                .iter()
                .all(|page| !runtime.queue.contains(*page))
        );
    }

    #[test]
    fn terminal_other_view_page_reenters_capacity_proof_only_after_explicit_retry() {
        let mut runtime = virtual_forest_runtime();
        runtime.coverage_guard.package_bootstrap = true;
        runtime.release_package_bootstrap_reserve().unwrap();

        let roots = runtime.hierarchy.roots();
        let direct_view = LodRuntimeViewId(1);
        let split_view = LodRuntimeViewId(2);
        let direct_root = roots[0];
        let direct_root_page = runtime.hierarchy.page(direct_root).unwrap();
        let parent = roots[1];
        let parent_page = runtime.hierarchy.page(parent).unwrap();
        let children = runtime.hierarchy.children(parent).to_vec();
        let cohort_pages = children
            .iter()
            .filter_map(|child| runtime.hierarchy.page(*child))
            .collect::<BTreeSet<_>>();
        let resident_child = *cohort_pages.first().unwrap();
        assert!(!cohort_pages.contains(&direct_root_page));
        let mut cohort_union = cohort_pages.clone();
        cohort_union.insert(parent_page);
        let limit_pages = u32::try_from(cohort_union.len()).unwrap();
        let old_limits = runtime.cache.limits();
        runtime.cache = LodPageCache::new(PageCacheLimits {
            max_pages: limit_pages,
            max_bytes: old_limits.max_bytes,
            max_gaussians: old_limits.max_gaussians,
        })
        .unwrap();
        for page in [parent_page, resident_child] {
            let descriptor = runtime.hierarchy.page_descriptor(page).unwrap();
            runtime
                .cache
                .insert(
                    page,
                    descriptor.decoded_len,
                    u64::from(descriptor.gaussian_count),
                    0,
                )
                .unwrap();
        }
        let excluded = cohort_union
            .iter()
            .copied()
            .chain([direct_root_page])
            .collect::<BTreeSet<_>>();
        let stale_pages = runtime
            .hierarchy
            .manifest()
            .pages
            .iter()
            .filter(|page| page.kind == LodPageKind::Representatives)
            .map(|page| page.id)
            .filter(|page| !excluded.contains(page))
            .take(limit_pages as usize - 2)
            .collect::<Vec<_>>();
        assert_eq!(stale_pages.len(), limit_pages as usize - 2);
        for page in stale_pages {
            let descriptor = runtime.hierarchy.page_descriptor(page).unwrap();
            runtime
                .cache
                .insert(
                    page,
                    descriptor.decoded_len,
                    u64::from(descriptor.gaussian_count),
                    0,
                )
                .unwrap();
        }
        assert_eq!(runtime.cache.stats().resident_pages, limit_pages);
        runtime.cache.pin_fallback(parent_page).unwrap();

        let terminal_request = PageRequest::new(
            direct_root_page,
            PageRequestPriority::fallback_critical(u32::MAX),
        );
        runtime.terminal_failures.insert(direct_root_page);
        runtime
            .terminal_requests
            .insert(direct_root_page, terminal_request);
        let direct_frontier = LodFrontier {
            nodes: Vec::new(),
            requested_nodes: vec![direct_root],
            status: LodEffectiveStatus::default(),
        };
        assert!(
            !runtime.split_cohort_pressure(&direct_frontier).unwrap(),
            "a terminal nonresident request must not create synthetic pressure"
        );

        let plan = LodSplitCohortPlan {
            view: split_view,
            parent,
            pages: cohort_pages.clone(),
        };
        let split_frontier = LodFrontier {
            nodes: vec![parent],
            requested_nodes: children,
            status: LodEffectiveStatus::default(),
        };
        let first = runtime.begin_frame();
        runtime.record_frame_demand(first, direct_view, &direct_frontier);
        assert_eq!(
            runtime.views[&direct_view].requested_pages,
            BTreeSet::from([direct_root_page])
        );
        assert!(runtime.views[&direct_view].admitted_pages.is_empty());
        let split = runtime.views.entry(split_view).or_default();
        split.pinned_frontier = BTreeSet::from([parent_page]);
        split.split_cohort_candidates = vec![plan.clone()];
        split.split_cohort_candidates_frame = first;
        split.split_cohort_pressure = true;
        split.split_cohort_pressure_frame = first;
        runtime.schedule_next_split_cohort(first).unwrap();
        assert_eq!(runtime.active_split_cohort.as_ref().unwrap().plan, plan);
        assert!(runtime.split_cohort_capacity_stall.is_none());
        assert_eq!(runtime.cache.get(resident_child).unwrap().pin_count, 1);
        runtime.finish_frame(first).unwrap();

        assert!(runtime.retry_terminal_failure(direct_root_page).unwrap());
        assert!(!runtime.is_terminal_failure(direct_root_page));
        let second = runtime.begin_frame();
        let split = runtime.views.get_mut(&split_view).unwrap();
        split.split_cohort_candidates_frame = second;
        split.split_cohort_pressure_frame = second;
        runtime
            .active_split_cohort
            .as_mut()
            .unwrap()
            .owner_updated_frame = second;
        runtime.record_frame_demand(second, split_view, &split_frontier);
        runtime.record_frame_demand(second, direct_view, &direct_frontier);
        assert_eq!(
            runtime.views[&direct_view].admitted_pages,
            BTreeSet::from([direct_root_page])
        );

        runtime.finish_frame(second).unwrap();
        assert!(runtime.active_split_cohort.is_none());
        let stall = runtime.split_cohort_capacity_stall.unwrap();
        assert_eq!(stall.required_pages, u64::from(limit_pages) + 1);
        assert_eq!(stall.limit_pages, u64::from(limit_pages));
        assert_eq!(runtime.cache.get(resident_child).unwrap().pin_count, 0);
        assert!(runtime.queue.contains(direct_root_page));
        assert!(cohort_pages.iter().all(|page| {
            !runtime.queue.contains(*page)
                && !runtime.in_flight.contains_key(page)
                && !runtime.preprocessor.contains(*page)
                && !runtime.capacity_blocked.contains_key(page)
        }));
    }

    #[test]
    fn invalidated_cohort_recomputes_owner_demand_before_frame_cancellation() {
        let mut runtime = virtual_forest_runtime();
        runtime.coverage_guard.package_bootstrap = true;
        runtime.release_package_bootstrap_reserve().unwrap();
        let view_id = LodRuntimeViewId::default();
        let roots = runtime.hierarchy.roots();
        let missing_root = roots[0];
        let missing_root_page = runtime.hierarchy.page(missing_root).unwrap();
        let parent = roots[1];
        let children = runtime.hierarchy.children(parent).to_vec();
        let cohort_pages = children
            .iter()
            .filter_map(|child| runtime.hierarchy.page(*child))
            .collect::<BTreeSet<_>>();
        let plan = LodSplitCohortPlan {
            view: view_id,
            parent,
            pages: cohort_pages.clone(),
        };
        let frame = runtime.begin_frame();
        runtime.active_split_cohort = Some(LodActiveSplitCohort {
            plan,
            pinned_pages: BTreeSet::new(),
            owner_updated_frame: frame,
            owner_base_admitted_pages: BTreeSet::new(),
            owner_base_admitted_pages_frame: frame,
        });
        let blocked = *cohort_pages.first().unwrap();
        runtime.capacity_blocked.insert(
            blocked,
            PageRequest::new(blocked, PageRequestPriority::visible(u32::MAX)),
        );

        let mut requested_nodes = vec![missing_root];
        requested_nodes.extend(children);
        let mixed = LodFrontier {
            nodes: vec![parent],
            requested_nodes,
            status: LodEffectiveStatus::default(),
        };
        runtime
            .observe_split_cohort_candidates(frame, view_id, &mixed)
            .unwrap();
        runtime.record_frame_demand(frame, view_id, &mixed);
        assert!(
            cohort_pages
                .iter()
                .all(|page| { runtime.views[&view_id].admitted_pages.contains(page) })
        );

        runtime
            .reconcile_active_split_cohort_after_frontier(frame, view_id, &mixed)
            .unwrap();
        assert!(runtime.active_split_cohort.is_none());
        assert_eq!(
            runtime.views[&view_id].admitted_pages,
            BTreeSet::from([missing_root_page])
        );
        runtime.enqueue_missing_roots(&mixed, view()).unwrap();
        runtime.finish_frame(frame).unwrap();
        assert!(runtime.queue.contains(missing_root_page));
        assert!(cohort_pages.iter().all(|page| {
            !runtime.queue.contains(*page)
                && !runtime.in_flight.contains_key(page)
                && !runtime.preprocessor.contains(*page)
                && !runtime.capacity_blocked.contains_key(page)
        }));
    }

    #[test]
    fn pressured_split_cohort_pins_partial_siblings_until_atomic_replacement() {
        let (manifest, transport, mut settings, mut streaming) = fixture();
        settings.quality = 1.0;
        settings.budgets.max_upload_bytes_per_frame = settings.budgets.max_resident_bytes;
        settings.budgets.max_requests_per_frame = 16;
        streaming.max_concurrent_requests = 16;

        let hierarchy = CompiledManifestLodHierarchy::new(manifest.clone()).unwrap();
        let guard = LodRuntimeCoverageGuard::new(&hierarchy, &settings).unwrap();
        let mut available_pages = guard.pages.clone();
        available_pages.extend(
            hierarchy
                .roots()
                .iter()
                .filter_map(|root| hierarchy.page(*root)),
        );
        let (waiting, parent, cohort_pages) = (0..manifest.nodes.len())
            .find_map(|_| {
                let waiting = select_frontier_with_previous_and_visibility(
                    &hierarchy,
                    &|node| {
                        hierarchy
                            .page(node)
                            .is_some_and(|page| available_pages.contains(&page))
                    },
                    view(),
                    &settings,
                    &[],
                    |_, _| true,
                )
                .unwrap();
                let selected = waiting.nodes.iter().copied().collect::<BTreeSet<_>>();
                let cohort = waiting.requested_nodes.iter().find_map(|requested| {
                    let parent = hierarchy.parent(*requested)?;
                    if !selected.contains(&parent) {
                        return None;
                    }
                    let pages = hierarchy
                        .children(parent)
                        .iter()
                        .filter_map(|child| hierarchy.page(*child))
                        .collect::<BTreeSet<_>>();
                    (pages.len() >= 2).then_some((parent, pages))
                });
                if let Some((parent, pages)) = cohort {
                    Some((waiting, parent, pages))
                } else {
                    let before = available_pages.len();
                    available_pages.extend(
                        waiting
                            .requested_nodes
                            .iter()
                            .filter_map(|node| hierarchy.page(*node)),
                    );
                    assert!(
                        available_pages.len() > before,
                        "fixture must progress toward a multi-page child cohort"
                    );
                    None
                }
            })
            .expect("fixture must expose a multi-page atomic child split");
        // Keep the already-streamed navigation path resident as it would be in
        // a warm cache. Only the selected cut is pinned by the view; these
        // ancestors remain ordinary LRU candidates.
        let mut base_pages = available_pages.clone();
        base_pages.extend(
            waiting
                .nodes
                .iter()
                .filter_map(|node| hierarchy.page(*node)),
        );
        let reproduced = select_frontier_with_previous_and_visibility(
            &hierarchy,
            &|node| {
                hierarchy
                    .page(node)
                    .is_some_and(|page| base_pages.contains(&page))
            },
            view(),
            &settings,
            &[],
            |_, _| true,
        )
        .unwrap();
        assert!(reproduced.nodes.contains(&parent));
        assert!(cohort_pages.difference(&base_pages).count() >= 2);
        let mut pressure_union = base_pages.clone();
        pressure_union.extend(cohort_pages.iter().copied());
        let stale = manifest
            .pages
            .iter()
            .map(|page| page.id)
            .find(|page| !pressure_union.contains(page))
            .expect("fixture must provide one evictable stale page");
        settings.budgets.max_resident_pages = u32::try_from(pressure_union.len()).unwrap();

        let mut decode_transport = transport.clone();
        let mut runtime =
            LodStreamingRuntime::new(manifest.clone(), transport, &settings, &streaming).unwrap();
        assert_eq!(runtime.coverage_guard.pages, guard.pages);
        let mut seeds = base_pages.clone();
        seeds.insert(stale);
        for page_id in seeds {
            let input = memory_preprocess_input(
                &manifest,
                &mut decode_transport,
                page_id,
                streaming.effective_max_encoded_page_bytes(),
            );
            let page = decode_page(&input.payload.bytes, input.limits).unwrap();
            let descriptor = runtime.hierarchy.page_descriptor(page_id).unwrap().clone();
            runtime.queue.remove(page_id);
            runtime
                .cache
                .insert(
                    page_id,
                    descriptor.decoded_len,
                    u64::from(descriptor.gaussian_count),
                    u64::from(page_id != stale),
                )
                .unwrap();
            runtime.decoded_pages.insert(page_id, page);
            runtime.pin_coverage_guard_page(page_id).unwrap();
        }

        let admitted = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(admitted.frontier().nodes.contains(&parent));
        let active = runtime
            .active_split_cohort
            .as_ref()
            .expect("the over-capacity wave must admit one bounded cohort");
        assert_eq!(active.plan.parent, parent);
        assert_eq!(active.plan.pages, cohort_pages);
        let mut observed_cohort_pages = active.pinned_pages.clone();
        let mut replaced = false;
        for _ in 0..512 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert_eq!(frame.capacity_blocked_requests(), 0);
            for &page in &observed_cohort_pages {
                assert!(
                    runtime.cache().contains(page),
                    "a committed cohort sibling must not be evicted before transfer"
                );
            }
            observed_cohort_pages.extend(
                cohort_pages
                    .iter()
                    .filter(|page| runtime.cache().contains(**page))
                    .copied(),
            );
            if !frame.frontier().nodes.contains(&parent) {
                replaced = true;
                break;
            }
        }
        assert!(replaced, "the pinned child cohort must replace its parent");
        assert!(runtime.active_split_cohort.as_ref().is_none_or(|cohort| {
            cohort.plan.view != LodRuntimeViewId::default() || cohort.plan.parent != parent
        }));
        assert!(!runtime.cache().contains(stale));
        assert!(
            cohort_pages
                .iter()
                .all(|page| runtime.cache().contains(*page))
        );
        assert!(runtime.cache().stats().resident_pages <= settings.budgets.max_resident_pages);
    }

    #[test]
    fn continuous_coarsening_is_bounded_and_original_endpoint_remains_categorical() {
        let (manifest, transport, mut settings, streaming) = fixture();
        let source_count = manifest.header.source_gaussian_count;
        settings.hysteresis = 0.0;
        settings.quality = 1.0;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();

        let mut exact = None;
        for _ in 0..64 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            if frame.frontier.requested_nodes.is_empty()
                && frame.in_flight_requests == 0
                && frame.queued_requests == 0
            {
                exact = Some(frame);
                break;
            }
        }
        let exact = exact.expect("fixture should become fully resident");
        assert_eq!(exact.candidate_count(), source_count);
        let exact_nodes = exact.frontier.nodes.clone();

        settings.quality = 0.10;
        let traversals = runtime.selection_traversals;
        let pin_rebuilds = runtime.frontier_pin_rebuilds;
        let range_rebuilds = runtime.physical_range_rebuilds;
        let payload_hits = runtime.stable_payload_hits;
        let first = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(
            !first.selection_stable(),
            "the first confirmation frame is not a selector fixed point"
        );
        assert_eq!(first.frontier.nodes, exact_nodes);
        assert_eq!(first.candidate_count(), source_count);
        assert_eq!(runtime.selection_traversals, traversals + 1);
        assert_eq!(runtime.frontier_pin_rebuilds, pin_rebuilds + 1);
        assert_eq!(runtime.physical_range_rebuilds, range_rebuilds + 1);
        assert_eq!(runtime.stable_payload_hits, payload_hits);
        assert!(
            runtime.views[&LodRuntimeViewId::default()]
                .stable_selection
                .is_none(),
            "an active temporal transition must never reuse a fixed-point cache"
        );

        let second = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(
            !second.selection_stable(),
            "a changing temporal frontier must remain unstable"
        );
        assert_eq!(runtime.selection_traversals, traversals + 2);
        assert_eq!(runtime.frontier_pin_rebuilds, pin_rebuilds + 2);
        assert_eq!(runtime.physical_range_rebuilds, range_rebuilds + 2);
        assert_eq!(runtime.stable_payload_hits, payload_hits);
        assert!(
            !runtime.views[&LodRuntimeViewId::default()]
                .temporal_demands
                .is_empty()
        );

        let mut cuts =
            BTreeSet::from([first.frontier.nodes.clone(), second.frontier.nodes.clone()]);
        let mut coarsest_seen = first.candidate_count().min(second.candidate_count());
        let mut stable = false;
        for _ in 0..20 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            frame
                .candidate_frontier(settings.max_active_gaussians_u32())
                .expect("every staggered state remains a complete resident cut");
            cuts.insert(frame.frontier.nodes.clone());
            coarsest_seen = coarsest_seen.min(frame.candidate_count());
            stable |= frame.selection_stable();
        }
        assert!(coarsest_seen < source_count);
        assert!(
            cuts.len() >= 3,
            "coarsening should not collapse in one swap"
        );
        assert!(
            stable,
            "the final repeated coarsened frontier must become stable"
        );

        settings.quality = 1.0;
        let refined = runtime.update(view(), &settings, &streaming).unwrap();
        // Explicit endpoint one remains categorical. Continuous refinement is
        // covered by the hierarchy cohort tests above.
        assert_eq!(refined.frontier.nodes, exact_nodes);
        assert_eq!(refined.candidate_count(), source_count);
    }

    #[test]
    fn cancelled_temporal_steps_restart_from_the_retained_refine_and_coarsen_cut() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.hysteresis = 0.0;
        settings.quality = 0.80;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let high = settle_temporal_fixture(&mut runtime, &settings, &streaming);
        let high_nodes = high.frontier().nodes.clone();

        settings.quality = 0.10;
        let first_coarsen = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(first_coarsen.frontier().nodes, high_nodes);
        assert!(first_coarsen.temporal_transition().is_none());
        let pending_coarsen = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(
            pending_coarsen
                .temporal_transition()
                .is_some_and(|transition| {
                    transition.substitutions().iter().all(|substitution| {
                        substitution.key.direction == LodTemporalDirection::Coarsen
                    })
                })
        );
        assert_ne!(pending_coarsen.frontier().nodes, high_nodes);

        runtime
            .restore_rendered_frontier(LodRuntimeViewId::default(), &high_nodes)
            .unwrap();
        let restored = &runtime.views[&LodRuntimeViewId::default()];
        assert_eq!(restored.previous_frontier, high_nodes);
        assert!(restored.temporal_demands.is_empty());
        assert!(restored.temporal_morph_cache.is_none());
        assert_eq!(restored.selected_frontier, restored.pinned_frontier);
        let retry_confirmation = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(retry_confirmation.frontier().nodes, high_nodes);
        assert!(retry_confirmation.temporal_transition().is_none());
        let retried_coarsen = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(
            retried_coarsen
                .temporal_transition()
                .is_some_and(|transition| {
                    transition.substitutions().iter().all(|substitution| {
                        substitution.key.direction == LodTemporalDirection::Coarsen
                    })
                })
        );

        let low = settle_temporal_fixture(&mut runtime, &settings, &streaming);
        let low_nodes = low.frontier().nodes.clone();
        assert_ne!(low_nodes, high_nodes);
        settings.quality = 0.80;
        let first_refine = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(first_refine.frontier().nodes, low_nodes);
        assert!(first_refine.temporal_transition().is_none());
        let pending_refine = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(
            pending_refine
                .temporal_transition()
                .is_some_and(|transition| {
                    transition.substitutions().iter().all(|substitution| {
                        substitution.key.direction == LodTemporalDirection::Refine
                    })
                })
        );
        assert_ne!(pending_refine.frontier().nodes, low_nodes);

        runtime
            .restore_rendered_frontier(LodRuntimeViewId::default(), &low_nodes)
            .unwrap();
        let retry_confirmation = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(retry_confirmation.frontier().nodes, low_nodes);
        assert!(retry_confirmation.temporal_transition().is_none());
        let retried_refine = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(
            retried_refine
                .temporal_transition()
                .is_some_and(|transition| {
                    transition.substitutions().iter().all(|substitution| {
                        substitution.key.direction == LodTemporalDirection::Refine
                    })
                })
        );
        assert_eq!(
            runtime.views[&LodRuntimeViewId::default()].selected_frontier,
            runtime.views[&LodRuntimeViewId::default()].pinned_frontier,
            "rollback/retry must not leak a stale endpoint pin"
        );
    }

    #[test]
    fn terminal_failure_does_not_restart_per_frame_and_explicit_retry_succeeds() {
        let (manifest, transport, mut settings, mut streaming) = fixture();
        settings.quality = 0.0;
        settings.budgets.max_active_gaussians = root_active_gaussians(&manifest);
        settings.budgets.max_requests_per_frame = 1;
        streaming.retry_limit = 1;
        let page = root_page(&manifest);
        let mut runtime = LodStreamingRuntime::new(
            manifest,
            ToggleMemoryTransport::failing(transport),
            &settings,
            &streaming,
        )
        .unwrap();

        // retry_limit counts retries after the initial start: one retry means
        // exactly two failed transport starts before the page is terminal.
        let first = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(first.failed_pages.is_empty());
        assert_eq!(first.queued_requests, 1);
        assert_eq!(runtime.page_attempts(page), Some(1));
        let terminal = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(terminal.failed_pages, vec![page]);
        assert_eq!(terminal.queued_requests, 0);
        assert_eq!(terminal.in_flight_requests, 0);
        assert_eq!(runtime.page_attempts(page), Some(2));
        assert!(runtime.is_terminal_failure(page));
        assert_eq!(runtime.terminal_failures(), &BTreeSet::from([page]));
        assert_eq!(
            runtime.page_transport_failure(page),
            Some(&LodPageTransportFailure::transport("forced begin failure"))
        );
        assert_eq!(runtime.transport_mut().begin_count, 2);

        // Visibility keeps requesting the page, but terminal state suppresses
        // enqueue/start and does not repeatedly report the same transition.
        for _ in 0..8 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert!(frame.started_pages.is_empty());
            assert!(frame.failed_pages.is_empty());
            assert_eq!(frame.queued_requests, 0);
            assert_eq!(frame.in_flight_requests, 0);
        }
        assert_eq!(runtime.transport_mut().begin_count, 2);

        runtime.transport_mut().fail_begin = false;
        assert!(runtime.retry_terminal_failure(page).unwrap());
        assert!(!runtime.is_terminal_failure(page));
        assert_eq!(runtime.page_attempts(page), None);
        assert_eq!(runtime.page_transport_failure(page), None);
        let restarted = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(restarted.started_pages, vec![page]);
        assert_eq!(runtime.page_attempts(page), Some(1));
        assert_eq!(runtime.transport_mut().begin_count, 3);

        let verifying = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(verifying.completed_pages.is_empty());
        assert_eq!(verifying.preprocess_stats().submitted, 1);
        assert_eq!(
            verifying.preprocess_stats().cooperative_decoded_gaussians,
            0
        );
        let completed = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(completed.completed_pages, vec![page]);
        assert_eq!(runtime.page_attempts(page), None);
        assert!(!runtime.is_terminal_failure(page));
        assert!(runtime.terminal_failures().is_empty());
        assert!(!runtime.retry_terminal_failure(page).unwrap());
    }

    #[test]
    fn predictive_terminal_failure_isolated_until_target_demand_and_drops_sibling_cohort() {
        let (manifest, transport, settings, streaming) = fixture();
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let (parent, children, child_pages) = runtime
            .hierarchy
            .manifest()
            .nodes
            .iter()
            .filter(|node| !node.children.is_empty())
            .find_map(|node| {
                let children = runtime.hierarchy.children(node.id).to_vec();
                let pages = children
                    .iter()
                    .filter_map(|child| runtime.hierarchy.page(*child))
                    .collect::<BTreeSet<_>>();
                (pages.len() >= 2).then_some((node.id, children, pages))
            })
            .expect("fixture has a sibling cohort spanning at least two pages");
        let failed_page = *child_pages.first().unwrap();
        let sibling_page = *child_pages.iter().nth(1).unwrap();
        let failed_child = children
            .iter()
            .copied()
            .find(|child| runtime.hierarchy.page(*child) == Some(failed_page))
            .unwrap();
        let sibling_descriptor = runtime.hierarchy.page_descriptor(sibling_page).unwrap();
        let sibling_decoded_len = sibling_descriptor.decoded_len;
        let sibling_gaussian_count = sibling_descriptor.gaussian_count;
        runtime
            .cache
            .insert(
                sibling_page,
                sibling_decoded_len,
                u64::from(sibling_gaussian_count),
                0,
            )
            .unwrap();
        runtime.cache.pin_fallback(sibling_page).unwrap();
        let state = runtime
            .views
            .entry(LodRuntimeViewId::default())
            .or_default();
        state
            .predictive_view_blend_nodes
            .insert(parent, children.clone());
        state.pinned_predictive_pages.insert(sibling_page);
        assert!(runtime.has_predictive_view_blend_work());

        runtime.attempts.insert(failed_page, 1);
        let mut failed_pages = Vec::new();
        runtime.retry_or_fail(
            PageRequest::new(failed_page, PageRequestPriority::prefetch(1)),
            0,
            &mut failed_pages,
        );
        assert!(failed_pages.is_empty());
        assert!(runtime.terminal_failures.is_empty());
        assert!(
            runtime
                .speculative_prefetch_terminal_requests
                .contains_key(&failed_page)
        );

        assert!(
            runtime
                .drop_predictive_view_blend_cohorts_with_terminal_members()
                .unwrap()
        );
        assert!(
            !runtime.views[&LodRuntimeViewId::default()]
                .predictive_view_blend_nodes
                .contains_key(&parent)
        );
        assert_eq!(runtime.cache.get(sibling_page).unwrap().pin_count, 0);
        assert!(!runtime.has_predictive_view_blend_work());

        runtime.promote_target_requested_speculative_failures(&[failed_child], &mut failed_pages);
        assert_eq!(failed_pages, vec![failed_page]);
        assert_eq!(runtime.terminal_failures, BTreeSet::from([failed_page]));
        assert!(
            !runtime
                .speculative_prefetch_terminal_requests
                .contains_key(&failed_page)
        );
        assert_eq!(
            runtime.terminal_requests[&failed_page].priority.class,
            PageRequestClass::Visible
        );
    }

    #[test]
    fn stationary_view_advances_predictive_transport_without_changing_the_target_cut() {
        let (manifest, transport, mut settings, mut streaming) = abi16_morph_fixture();
        settings.hysteresis = 0.0;
        settings.frustum_culling = false;
        settings.budgets.max_requests_per_frame = 1;
        streaming.max_concurrent_requests = 1;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        assert!(runtime.hierarchy.manifest().morph_map.is_some());

        let stationary_view = view();
        let (quality, expected_nodes, parent, predictive_pages) = (1..100)
            .find_map(|quality_step| {
                settings.quality = quality_step as f32 / 100.0;
                let target = select_frontier_with_visibility(
                    &runtime.hierarchy,
                    &|_| true,
                    stationary_view,
                    &settings,
                    |_, _| true,
                )
                .ok()?;
                let target_pages = target
                    .nodes
                    .iter()
                    .filter_map(|node| runtime.hierarchy.page(*node))
                    .collect::<BTreeSet<_>>();
                target.nodes.iter().copied().find_map(|parent| {
                    let substitution = runtime.direct_refine_substitution(parent).ok()?;
                    let edge = runtime.view_blend_edge_for_substitution(
                        &substitution,
                        0.0_f32.to_bits(),
                        false,
                    )?;
                    let (parent_pressure, child_pressure) = lod_view_blend_pressures(
                        stationary_view,
                        settings.quality_target(),
                        &edge,
                    )?;
                    if !(LOD_VIEW_BLEND_PREFETCH_PARENT_PRESSURE..=1.0).contains(&parent_pressure)
                        || child_pressure >= LOD_VIEW_BLEND_RELEASE_CHILD_PRESSURE
                    {
                        return None;
                    }
                    let predictive_pages = substitution
                        .next_nodes
                        .iter()
                        .filter_map(|node| runtime.hierarchy.page(*node))
                        .filter(|page| {
                            !target_pages.contains(page)
                                && !runtime.coverage_guard.pages.contains(page)
                        })
                        .collect::<BTreeSet<_>>();
                    (!predictive_pages.is_empty()).then_some((
                        settings.quality,
                        target.nodes.clone(),
                        parent,
                        predictive_pages,
                    ))
                })
            })
            .expect("fixture exposes a stationary selected parent inside the prefetch band");
        settings.quality = quality;

        let first_maintenance = (0..512)
            .find_map(|_| {
                let frame = runtime
                    .update(stationary_view, &settings, &streaming)
                    .unwrap();
                (frame.frontier().nodes == expected_nodes
                    && frame.frontier().requested_nodes.is_empty()
                    && runtime.has_predictive_view_blend_work())
                .then_some(frame)
            })
            .expect("stationary target cut begins predictive child-cohort maintenance");
        assert!(first_maintenance.frontier().nodes.contains(&parent));
        assert_eq!(first_maintenance.frontier().status.requested_pages, 0);
        assert_eq!(
            first_maintenance.frontier().status.degradation,
            LodDegradation::None
        );
        assert!(
            predictive_pages
                .iter()
                .any(|page| !runtime.cache.contains(*page)),
            "the observation point must precede completion of optional work"
        );

        let starts_before_maintenance = runtime.transport_request_starts_for_test();
        let mut completed_predictive_page = false;
        let mut reached_quiescence = false;
        for _ in 0..512 {
            let frame = runtime
                .update(stationary_view, &settings, &streaming)
                .unwrap();
            assert_eq!(
                frame.frontier().nodes,
                expected_nodes,
                "optional maintenance must not publish a regressive topology wave"
            );
            assert!(frame.frontier().requested_nodes.is_empty());
            assert_eq!(frame.frontier().status.requested_pages, 0);
            assert_eq!(frame.frontier().status.degradation, LodDegradation::None);
            completed_predictive_page |= frame
                .completed_pages()
                .iter()
                .any(|page| predictive_pages.contains(page));
            if !runtime.has_predictive_view_blend_work() {
                reached_quiescence = true;
                break;
            }
        }
        assert!(
            completed_predictive_page,
            "identical-view maintenance must advance transport and preprocessing to publication"
        );
        assert!(
            reached_quiescence,
            "the bounded predictive cohort must settle"
        );
        assert!(
            runtime.transport_request_starts_for_test() > starts_before_maintenance
                || predictive_pages
                    .iter()
                    .all(|page| runtime.cache.contains(*page)),
            "maintenance must either start remaining transport or finish the already-started request"
        );
        assert!(
            predictive_pages
                .iter()
                .all(|page| runtime.cache.contains(*page))
        );
    }

    #[test]
    fn transport_admission_backpressure_is_retry_neutral_but_io_failure_is_not() {
        let (manifest, transport, mut settings, mut streaming) = fixture();
        settings.quality = 0.0;
        settings.budgets.max_active_gaussians = root_active_gaussians(&manifest);
        settings.budgets.max_requests_per_frame = 1;
        streaming.retry_limit = 1;
        let page = root_page(&manifest);
        let transport = AdmissionDeferringMemoryTransport {
            inner: transport,
            admission_saturated: true,
            fail_next_io: true,
            begin_count: 0,
        };
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();

        let started = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(started.started_pages, vec![page]);
        assert_eq!(runtime.page_attempts(page), Some(1));
        assert_eq!(runtime.transport_mut().begin_count, 1);

        for _ in 0..8 {
            let deferred = runtime.update(view(), &settings, &streaming).unwrap();
            assert!(deferred.started_pages.is_empty());
            assert!(deferred.failed_pages.is_empty());
            assert_eq!(deferred.in_flight_requests, 1);
            assert_eq!(runtime.page_attempts(page), Some(1));
            assert_eq!(runtime.transport_mut().begin_count, 1);
            assert!(!runtime.is_terminal_failure(page));
            assert_eq!(runtime.page_transport_failure(page), None);
        }

        runtime.transport_mut().admission_saturated = false;
        let retried = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(retried.started_pages, vec![page]);
        assert!(retried.failed_pages.is_empty());
        assert_eq!(runtime.page_attempts(page), Some(2));
        assert_eq!(runtime.transport_mut().begin_count, 2);
        assert!(!runtime.is_terminal_failure(page));
        assert_eq!(
            runtime.page_transport_failure(page),
            Some(&LodPageTransportFailure::transport("forced I/O failure"))
        );

        let mut completed = false;
        for _ in 0..128 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            if frame.completed_pages == [page] {
                completed = true;
                break;
            }
        }
        assert!(completed, "the admitted retry should complete");
        assert_eq!(runtime.page_attempts(page), None);
        assert_eq!(runtime.page_transport_failure(page), None);
        assert!(!runtime.is_terminal_failure(page));
    }

    #[test]
    fn coarsest_streams_one_candidate_for_over_100m_logical_gaussians() {
        let mut fixture = virtual_runtime_fixture();
        fixture.lod_settings.selection_mode = LodSelectionMode::Frozen;
        let source_gaussian_count = fixture.manifest.header.source_gaussian_count;
        assert!(source_gaussian_count > 100_000_000);
        assert_eq!(fixture.manifest.nodes.len(), VIRTUAL_NODE_COUNT as usize);
        assert_eq!(fixture.manifest.pages.len(), VIRTUAL_NODE_COUNT as usize);
        assert!(u64::from(VIRTUAL_NODE_COUNT) * 1_000 < source_gaussian_count);
        assert_eq!(
            fixture
                .manifest
                .pages
                .iter()
                .filter(|page| page.storage.is_some())
                .count(),
            1
        );
        assert!(fixture.encoded_root_bytes < 4 * 1024);

        let root_node = fixture.manifest.roots[0];
        let root_page = fixture.manifest.nodes[0].representation.page;
        let mut runtime = LodStreamingRuntime::new(
            fixture.manifest,
            fixture.transport,
            &fixture.lod_settings,
            &fixture.streaming_settings,
        )
        .unwrap();
        assert_eq!(
            runtime.hierarchy().manifest().header.source_gaussian_count,
            source_gaussian_count
        );
        assert_eq!(
            runtime.hierarchy().manifest().pages.len(),
            VIRTUAL_NODE_COUNT as usize
        );
        assert!(
            runtime.shared_page_node_ranges.is_empty(),
            "one-page-per-node external packages must skip the shared-page index"
        );
        assert_eq!(
            runtime.atlas_layout().gaussians_per_slot,
            VirtualCityScene::default().gaussians_per_page
        );
        assert!(runtime.decoded_pages.is_empty());

        let captured_view = view();
        let moved_view = LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, 800.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        let first = runtime
            .update(
                captured_view,
                &fixture.lod_settings,
                &fixture.streaming_settings,
            )
            .unwrap();
        assert!(first.selection_view_frozen());
        assert_eq!(
            runtime.views[&LodRuntimeViewId::default()].frozen_selection_view,
            Some(captured_view)
        );
        assert_eq!(first.candidate_count(), 0);
        assert_eq!(first.started_pages, vec![root_page]);
        assert_eq!(first.in_flight_requests, 1);
        assert_eq!(first.cache.resident_gaussians, 0);

        let verifying = runtime
            .update(
                moved_view,
                &fixture.lod_settings,
                &fixture.streaming_settings,
            )
            .unwrap();
        assert!(verifying.selection_view_frozen());
        assert_eq!(
            runtime.views[&LodRuntimeViewId::default()].frozen_selection_view,
            Some(captured_view)
        );
        assert!(verifying.completed_pages.is_empty());
        assert_eq!(verifying.preprocess_stats().submitted, 1);
        assert_eq!(
            verifying.preprocess_stats().cooperative_decoded_gaussians,
            0
        );
        let resident = runtime
            .update(
                moved_view,
                &fixture.lod_settings,
                &fixture.streaming_settings,
            )
            .unwrap();
        assert!(resident.selection_view_frozen());
        assert_eq!(resident.completed_pages, vec![root_page]);
        assert_eq!(resident.frontier.nodes, vec![root_node]);
        assert!(resident.frontier.requested_nodes.is_empty());
        assert_eq!(
            resident.frontier.status.requested_target,
            crate::gaussian::lod_settings::LodQualityTarget::Coarsest
        );
        assert_eq!(resident.frontier.status.achieved_max_target_ratio, 0.0);
        assert_eq!(resident.frontier.status.active_gaussians, 1);
        assert_eq!(resident.candidate_count(), 1);
        assert!(
            resident
                .candidate_frontier(1)
                .unwrap()
                .selection_view_frozen()
        );
        assert_eq!(resident.cache.resident_pages, 1);
        assert_eq!(resident.cache.resident_gaussians, 1);
        assert_eq!(runtime.decoded_pages.len(), 1);
        assert_eq!(runtime.cache().limits().max_pages, 2);
        assert_eq!(
            runtime.cache().limits().max_gaussians,
            u64::from(VirtualCityScene::default().gaussians_per_page)
        );
        let materialized_gaussians = runtime
            .decoded_pages
            .values()
            .map(|page| page.gaussians.len())
            .sum::<usize>();
        assert_eq!(materialized_gaussians, 1);
        assert!(materialized_gaussians as u64 * 100_000_000 < source_gaussian_count);

        let mut dynamic_settings = fixture.lod_settings.clone();
        dynamic_settings.selection_mode = LodSelectionMode::Dynamic;
        let unfrozen = runtime
            .update(moved_view, &dynamic_settings, &fixture.streaming_settings)
            .unwrap();
        assert!(!unfrozen.selection_view_frozen());
        assert_eq!(
            runtime.views[&LodRuntimeViewId::default()].frozen_selection_view,
            None
        );
    }

    #[test]
    fn candidate_frontier_is_bounded_and_non_overlapping() {
        let frame = LodStreamFrame {
            view: LodRuntimeViewId::default(),
            frontier: LodFrontier {
                nodes: Vec::new(),
                requested_nodes: Vec::new(),
                status: crate::gaussian::lod_settings::LodEffectiveStatus {
                    active_gaussians: 10,
                    ..Default::default()
                },
            },
            physical_ranges: vec![LodPhysicalRange {
                node: LodNodeId(1),
                page: LodPageId(1),
                slot: AtlasSlot {
                    index: 0,
                    generation: 1,
                },
                physical_start: 0,
                count: 10,
            }],
            ancestor_fallback_nodes: BTreeSet::new(),
            selection_view_frozen: true,
            selection_stable: true,
            temporal_transition: None,
            complete_resident_cut: true,
            cache: Default::default(),
            queued_requests: 0,
            in_flight_requests: 0,
            preprocess: Default::default(),
            capacity_blocked_requests: 0,
            split_cohort_capacity_stall: None,
            started_pages: Vec::new(),
            completed_pages: Vec::new(),
            preprocess_failed_pages: Vec::new(),
            failed_pages: Vec::new(),
        };
        assert!(matches!(
            frame.candidate_frontier(9),
            Err(LodRuntimeError::CandidateExpansionLimit { .. })
        ));
        let frontier = frame.candidate_frontier(10).unwrap();
        assert_eq!(frontier.candidate_count(), 10);
        assert_eq!(frontier.quality_status(), &frame.frontier.status);
        assert!(frontier.selection_view_frozen());
        assert!(!frontier.is_coverage_guard());

        let mut overlapping = frame.clone();
        overlapping.physical_ranges.push(LodPhysicalRange {
            node: LodNodeId(2),
            page: LodPageId(2),
            slot: AtlasSlot {
                index: 1,
                generation: 1,
            },
            physical_start: 5,
            count: 5,
        });
        overlapping.frontier.status.active_gaussians = 15;
        assert!(matches!(
            overlapping.candidate_frontier(15),
            Err(LodRuntimeError::OverlappingPhysicalRanges { .. })
        ));
    }

    #[test]
    fn runtime_hysteresis_history_requires_an_unchanged_lod_policy() {
        let original = GaussianLodSettings {
            quality: 0.5,
            ..Default::default()
        };
        let mut state = LodRuntimeViewState::default();
        let frontier = [LodNodeId(3), LodNodeId(4)];
        state.commit_frontier(&frontier, &original);

        assert_eq!(state.hysteresis_frontier(&original), frontier);

        let mut changed_quality = original.clone();
        changed_quality.quality = 0.75;
        assert!(state.hysteresis_frontier(&changed_quality).is_empty());

        // Committing a successful selection resets history to the new policy.
        state.commit_frontier(&[LodNodeId(1)], &changed_quality);
        assert_eq!(state.hysteresis_frontier(&changed_quality), [LodNodeId(1)]);

        let mut ignored_residency_policy = changed_quality.clone();
        ignored_residency_policy.budgets.max_resident_bytes /= 2;
        ignored_residency_policy.budgets.max_pending_requests /= 2;
        assert_eq!(
            state.hysteresis_frontier(&ignored_residency_policy),
            [LodNodeId(1)],
            "non-selection residency budgets must not churn hysteresis history"
        );

        let mut changed_policy = changed_quality.clone();
        changed_policy.hysteresis *= 0.5;
        assert!(state.hysteresis_frontier(&changed_policy).is_empty());
    }

    #[test]
    fn frozen_selection_captures_moves_invariantly_and_recaptures_after_unfreeze() {
        let initial = view();
        let moved = LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, 80.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        let recapture =
            LodView::orthographic(bevy::math::Vec3::new(4.0, 2.0, 12.0), 720.0, 20.0, 0.01);
        let mut state = LodRuntimeViewState::default();

        assert_eq!(
            state.selection_view(initial, LodSelectionMode::Frozen),
            initial
        );
        assert_eq!(
            state.selection_view(moved, LodSelectionMode::Frozen),
            initial
        );
        assert_eq!(state.frozen_selection_view, Some(initial));

        assert_eq!(
            state.selection_view(moved, LodSelectionMode::Dynamic),
            moved
        );
        assert_eq!(state.frozen_selection_view, None);
        assert_eq!(
            state.selection_view(recapture, LodSelectionMode::Frozen),
            recapture
        );
        assert_eq!(state.frozen_selection_view, Some(recapture));
    }

    #[test]
    fn frozen_views_capture_independently_by_runtime_view_id() {
        let left = LodRuntimeViewId(21);
        let right = LodRuntimeViewId(22);
        let left_view = view();
        let right_view = LodView::perspective(
            bevy::math::Vec3::new(100.0, 0.0, 8.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        let moved = LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, -100.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        let mut views = BTreeMap::<LodRuntimeViewId, LodRuntimeViewState>::new();

        assert_eq!(
            views
                .entry(left)
                .or_default()
                .selection_view(left_view, LodSelectionMode::Frozen),
            left_view
        );
        assert_eq!(
            views
                .entry(right)
                .or_default()
                .selection_view(right_view, LodSelectionMode::Frozen),
            right_view
        );
        assert_eq!(
            views
                .get_mut(&left)
                .unwrap()
                .selection_view(moved, LodSelectionMode::Frozen),
            left_view
        );
        assert_eq!(
            views
                .get_mut(&right)
                .unwrap()
                .selection_view(moved, LodSelectionMode::Frozen),
            right_view
        );
    }

    #[test]
    fn frozen_selection_keeps_exact_candidate_payload_during_live_camera_motion() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.quality = 0.0;
        settings.selection_mode = LodSelectionMode::Frozen;
        let captured = view();
        let moved = LodView::perspective(
            bevy::math::Vec3::new(30.0, -10.0, 120.0),
            1080.0,
            50_f32.to_radians(),
            0.01,
        );
        assert_ne!(captured, moved);
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();

        let mut settled = None;
        for _ in 0..32 {
            let frame = runtime.update(captured, &settings, &streaming).unwrap();
            if frame.has_complete_resident_cut()
                && frame.selection_stable()
                && frame.queued_requests() == 0
                && frame.in_flight_requests() == 0
            {
                settled = Some(frame);
                break;
            }
        }
        let settled = settled.expect("coarsest frozen fixture must settle");
        let payload = settled
            .candidate_frontier(settings.max_active_gaussians_u32())
            .unwrap();

        let moved_frame = runtime.update(moved, &settings, &streaming).unwrap();
        let moved_payload = moved_frame
            .candidate_frontier(settings.max_active_gaussians_u32())
            .unwrap();
        assert!(moved_frame.selection_view_frozen());
        assert_eq!(moved_frame.frontier().nodes, settled.frontier().nodes);
        assert_eq!(moved_payload.physical_ranges(), payload.physical_ranges());
        assert!(moved_payload.same_render_payload(&payload));
        assert_eq!(
            runtime.views[&LodRuntimeViewId::default()].frozen_selection_view,
            Some(captured)
        );
    }

    #[test]
    fn frozen_view_keeps_frontier_and_residency_progress_mutable() {
        let mut settings = GaussianLodSettings {
            quality: 0.5,
            selection_mode: LodSelectionMode::Frozen,
            ..Default::default()
        };
        let captured = view();
        let mut state = LodRuntimeViewState::default();
        state.selection_view(captured, settings.selection_mode);

        state.commit_frontier(&[LodNodeId(1)], &settings);
        state.selected_frontier.insert(LodPageId(10));
        state.requested_pages.insert(LodPageId(11));

        // A later residency publication may refine the frontier without
        // changing the captured camera snapshot.
        state.commit_frontier(&[LodNodeId(2), LodNodeId(3)], &settings);
        state.selected_frontier.insert(LodPageId(11));
        state.requested_pages.clear();
        assert_eq!(state.previous_frontier, [LodNodeId(2), LodNodeId(3)]);
        assert_eq!(
            state.selected_frontier,
            BTreeSet::from([LodPageId(10), LodPageId(11)])
        );
        assert!(state.requested_pages.is_empty());
        assert_eq!(state.frozen_selection_view, Some(captured));

        // Quality remains independently editable and resets hysteresis history;
        // it does not discard the frozen camera until Dynamic is requested.
        settings.quality = 0.75;
        assert!(state.hysteresis_frontier(&settings).is_empty());
        assert_eq!(
            state.selection_view(captured, settings.selection_mode),
            captured
        );
    }

    #[test]
    fn fallback_provenance_tracks_the_selected_ancestor_per_frontier() {
        let (manifest, _, _, _) = fixture();
        let hierarchy = CompiledManifestLodHierarchy::new(manifest.clone()).unwrap();
        let (requested, parent, grandparent) = manifest
            .nodes
            .iter()
            .find_map(|node| {
                let parent = node.parent?;
                let grandparent = manifest
                    .nodes
                    .iter()
                    .find(|candidate| candidate.id == parent)?
                    .parent?;
                Some((node.id, parent, grandparent))
            })
            .expect("fixture has at least three hierarchy levels");

        let coarse = LodFrontier {
            nodes: vec![grandparent],
            requested_nodes: vec![requested],
            status: LodEffectiveStatus::default(),
        };
        assert_eq!(
            selected_ancestor_fallback_nodes(&hierarchy, &coarse),
            BTreeSet::from([grandparent])
        );

        let refined = LodFrontier {
            nodes: vec![parent],
            ..coarse
        };
        assert_eq!(
            selected_ancestor_fallback_nodes(&hierarchy, &refined),
            BTreeSet::from([parent]),
            "Residency provenance belongs to the exact per-view candidate cut"
        );
    }

    #[test]
    fn changed_fallback_provenance_cannot_inherit_an_active_render_phase() {
        use crate::stream::render_commit::{LOD_RENDER_ACTIVE, LodRenderCandidate};

        let range = LodPhysicalRange {
            node: LodNodeId(7),
            page: LodPageId(8),
            slot: AtlasSlot {
                index: 9,
                generation: 10,
            },
            physical_start: 11,
            count: 1,
        };
        let quality_status = LodEffectiveStatus {
            active_gaussians: 1,
            ..Default::default()
        };
        let resident = build_candidate_frontier(
            LodRuntimeViewId(12),
            &[range],
            &BTreeSet::new(),
            quality_status,
            LodCandidateFrontierBuildOptions {
                selection_view_frozen: false,
                coverage_guard: false,
                temporal_transition: None,
                limit: 1,
            },
        )
        .unwrap();
        let fallback = build_candidate_frontier(
            LodRuntimeViewId(12),
            &[range],
            &BTreeSet::from([range.node]),
            quality_status,
            LodCandidateFrontierBuildOptions {
                selection_view_frozen: false,
                coverage_guard: false,
                temporal_transition: None,
                limit: 1,
            },
        )
        .unwrap();
        let previous = LodRenderCandidate::new(resident);
        previous.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
        let replacement = LodRenderCandidate::new(fallback);

        assert!(!replacement.same_payload(&previous));
        assert!(!Arc::ptr_eq(&replacement.phase, &previous.phase));
        assert!(!replacement.render_is_active());
    }

    #[cfg(lod_render_path)]
    #[test]
    fn adapter_morph_fallback_switches_coarsening_to_the_exact_target_and_target_leases() {
        use crate::stream::render_commit::LodRenderCandidate;

        let target = LodPhysicalRange {
            node: LodNodeId(1),
            page: LodPageId(10),
            slot: AtlasSlot {
                index: 0,
                generation: 1,
            },
            physical_start: 0,
            count: 1,
        };
        let presented_child = LodPhysicalRange {
            node: LodNodeId(2),
            page: LodPageId(11),
            slot: AtlasSlot {
                index: 1,
                generation: 1,
            },
            physical_start: 8,
            count: 1,
        };
        let morph = Arc::new(LodTemporalMorphBatch {
            identity: LodTemporalMorphIdentity {
                primary: 1,
                secondary: 2,
                descriptor_count: 1,
                mapping_record_count: 1,
            },
            presentation_ranges: vec![presented_child],
            required_ranges: vec![target, presented_child],
            edges: vec![LodViewBlendEdge {
                parent: target.node,
                children: vec![presented_child.node],
                parent_metric: LodViewBlendMetric::from_node(
                    LodNodeMetrics {
                        center: bevy::math::Vec3::ZERO,
                        radius: 1.0,
                        geometric_error: 1.0,
                        appearance_error: 0.0,
                        opacity_error: 0.0,
                        quality_min: 0.0,
                        quality_max: 0.5,
                        high_fidelity_certificate: 0.0,
                        representative_count: 1,
                    },
                    false,
                ),
                child_metrics: vec![LodViewBlendMetric::from_node(
                    LodNodeMetrics {
                        center: bevy::math::Vec3::ZERO,
                        radius: 1.0,
                        geometric_error: 0.5,
                        appearance_error: 0.0,
                        opacity_error: 0.0,
                        quality_min: 0.5,
                        quality_max: 1.0,
                        high_fidelity_certificate: 1.0,
                        representative_count: 1,
                    },
                    true,
                )],
                initial_weight_bits: 1.0_f32.to_bits(),
                activation_requires_slew: true,
            }],
            descriptors: vec![LodTemporalMorphDescriptor {
                child_physical_start: presented_child.physical_start,
                child_count: presented_child.count,
                mapping_start: 0,
                edge_index: 0,
            }],
            records: vec![LodTemporalMorphRecord {
                parent_physical_index: target.physical_start,
                split_count: 1,
            }],
        });
        let transition = LodTemporalTransition {
            substitutions: vec![LodTemporalSubstitution {
                key: LodTemporalSubstitutionKey {
                    parent: target.node,
                    direction: LodTemporalDirection::Coarsen,
                },
                previous_nodes: vec![presented_child.node],
                next_nodes: vec![target.node],
                previous_gaussians: 1,
                next_gaussians: 1,
            }],
            initial_weight_bits: vec![1.0_f32.to_bits()],
            changed_gaussians: 1,
            atomic_budget_overshoot: 0,
            mode: LodTemporalTransitionMode::Morphing,
            morph: Some(morph),
        };
        let frontier = build_candidate_frontier(
            LodRuntimeViewId(7),
            &[target],
            &BTreeSet::new(),
            LodEffectiveStatus {
                active_gaussians: 1,
                ..Default::default()
            },
            LodCandidateFrontierBuildOptions {
                selection_view_frozen: false,
                coverage_guard: false,
                temporal_transition: Some(transition),
                limit: 1,
            },
        )
        .unwrap();
        let candidate = LodRenderCandidate::new(frontier);
        assert_eq!(candidate.render_ranges(), &[presented_child]);
        assert_eq!(
            candidate.required_atlas_ranges(),
            &[target, presented_child]
        );
        assert!(matches!(
            crate::render::lod::plan_lod_candidate_morph(&candidate, u64::MAX, u64::MAX).unwrap(),
            crate::render::lod::LodCandidateMorphPlan::Enabled { .. }
        ));

        let subview_candidate = LodRenderCandidate::new(candidate.frontier().clone());
        subview_candidate.phase.store(
            crate::stream::render_commit::LOD_RENDER_PREPARED,
            Ordering::Release,
        );
        assert_eq!(
            subview_candidate.temporal_transition_mode(),
            Some(LodTemporalTransitionMode::Morphing),
            "multiple private render views retain independent camera-conditioned weights"
        );
        assert!(matches!(
            crate::render::lod::plan_lod_candidate_morph(&subview_candidate, u64::MAX, u64::MAX,)
                .unwrap(),
            crate::render::lod::LodCandidateMorphPlan::Enabled { .. }
        ));
        assert_eq!(
            subview_candidate.required_atlas_ranges(),
            &[target, presented_child]
        );
        assert!(crate::render::lod::publish_bridge_activation_after_radix(
            &subview_candidate.phase,
        ));
        assert!(
            !crate::render::lod::publish_bridge_activation_after_radix(&subview_candidate.phase,),
            "the sibling subview observes shared exact-target activation without resetting it"
        );
        assert!(subview_candidate.render_is_active());
        assert_eq!(
            subview_candidate.required_atlas_ranges(),
            &[target, presented_child],
            "ACTIVE presentation retains its parent/child union"
        );

        candidate.publish_temporal_transition_mode(LodTemporalTransitionMode::BoundedHardCohort);
        assert_eq!(candidate.render_ranges(), &[target]);
        assert_eq!(candidate.required_atlas_ranges(), &[target]);
        assert_eq!(candidate.temporal_transition_progress(), None);
        assert!(matches!(
            crate::render::lod::plan_lod_candidate_morph(&candidate, u64::MAX, u64::MAX).unwrap(),
            crate::render::lod::LodCandidateMorphPlan::Disabled
        ));
        let (descriptors, count) =
            crate::render::lod::build_bridge_candidate_upload_descriptors(&candidate, 16, false)
                .unwrap();
        assert_eq!(count, target.count);
        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].physical_start, target.physical_start);
        assert_eq!(descriptors[0].count, target.count);
        assert_eq!(
            descriptors[0].metadata & !3,
            0,
            "target-only bounded upload must not emit the range bit which becomes entry bit 28"
        );

        let settled = LodRenderCandidate::new(candidate.frontier().clone());
        settled.phase.store(
            crate::stream::render_commit::LOD_RENDER_ACTIVE,
            Ordering::Release,
        );
        settled.settle_temporal_transition();
        assert!(
            settled.temporal_transition().is_some(),
            "settlement clears only the effective presentation mode; immutable authored provenance remains observable"
        );
        for reset_phase in [
            crate::stream::render_commit::LOD_RENDER_WAITING,
            crate::stream::render_commit::LOD_RENDER_PREPARED,
            crate::stream::render_commit::LOD_RENDER_FAILED,
            crate::stream::render_commit::LOD_RENDER_ACTIVE,
        ] {
            settled.phase.store(reset_phase, Ordering::Release);
            assert_eq!(settled.temporal_transition_mode(), None);
            assert_eq!(settled.render_ranges(), &[target]);
            assert_eq!(settled.required_atlas_ranges(), &[target]);
            assert!(matches!(
                crate::render::lod::plan_lod_candidate_morph(&settled, u64::MAX, u64::MAX,)
                    .unwrap(),
                crate::render::lod::LodCandidateMorphPlan::Disabled
            ));
            let (settled_descriptors, settled_count) =
                crate::render::lod::build_bridge_candidate_upload_descriptors(&settled, 16, false)
                    .unwrap();
            assert_eq!(settled_count, target.count);
            assert_eq!(settled_descriptors.len(), 1);
            assert_eq!(settled_descriptors[0].physical_start, target.physical_start);
            assert_eq!(settled_descriptors[0].metadata & !3, 0);
        }
    }

    #[test]
    fn persistent_active_view_blend_payload_inherits_without_a_second_commit() {
        use crate::stream::render_commit::{LOD_RENDER_ACTIVE, LodRenderCandidate};

        let range = LodPhysicalRange {
            node: LodNodeId(21),
            page: LodPageId(22),
            slot: AtlasSlot {
                index: 3,
                generation: 4,
            },
            physical_start: 12,
            count: 1,
        };
        let status = LodEffectiveStatus {
            active_gaussians: 1,
            ..Default::default()
        };
        let transitioned = build_candidate_frontier(
            LodRuntimeViewId(23),
            &[range],
            &BTreeSet::new(),
            status,
            LodCandidateFrontierBuildOptions {
                selection_view_frozen: false,
                coverage_guard: false,
                temporal_transition: None,
                limit: 1,
            },
        )
        .unwrap()
        .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing);
        let stable = build_candidate_frontier(
            LodRuntimeViewId(23),
            &[range],
            &BTreeSet::new(),
            status,
            LodCandidateFrontierBuildOptions {
                selection_view_frozen: false,
                coverage_guard: false,
                temporal_transition: None,
                limit: 1,
            },
        )
        .unwrap()
        .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing);
        let previous = LodRenderCandidate::new(transitioned);
        previous.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
        let mut next = LodRenderCandidate::new(stable);

        assert!(next.same_payload(&previous));
        next.inherit_active_payload_state(&previous);
        assert!(next.render_is_active());
        assert!(Arc::ptr_eq(&next.phase, &previous.phase));
        assert_eq!(
            next.temporal_transition_mode(),
            Some(LodTemporalTransitionMode::Morphing)
        );
        assert_eq!(next.temporal_transition_progress(), None);
        assert!(
            next.temporal_transition().is_some(),
            "stable same-payload inheritance retains immutable adjacent-edge provenance"
        );
        assert_eq!(
            next.required_atlas_ranges(),
            previous.required_atlas_ranges()
        );
    }

    #[test]
    fn two_million_candidate_frontier_stores_one_physical_range() {
        let mut frame = LodStreamFrame {
            view: LodRuntimeViewId(7),
            frontier: LodFrontier {
                nodes: vec![LodNodeId(1)],
                requested_nodes: Vec::new(),
                status: crate::gaussian::lod_settings::LodEffectiveStatus {
                    active_gaussians: 2_000_000,
                    ..Default::default()
                },
            },
            physical_ranges: vec![LodPhysicalRange {
                node: LodNodeId(1),
                page: LodPageId(1),
                slot: AtlasSlot {
                    index: 0,
                    generation: 1,
                },
                physical_start: 0,
                count: 2_000_000,
            }],
            ancestor_fallback_nodes: BTreeSet::new(),
            selection_view_frozen: false,
            selection_stable: true,
            temporal_transition: None,
            complete_resident_cut: true,
            cache: Default::default(),
            queued_requests: 0,
            in_flight_requests: 0,
            preprocess: Default::default(),
            capacity_blocked_requests: 0,
            split_cohort_capacity_stall: None,
            started_pages: Vec::new(),
            completed_pages: Vec::new(),
            preprocess_failed_pages: Vec::new(),
            failed_pages: Vec::new(),
        };
        frame.ancestor_fallback_nodes.insert(LodNodeId(1));
        assert_eq!(
            frame.ancestor_fallback_nodes().collect::<Vec<_>>(),
            vec![LodNodeId(1)]
        );
        let frontier = frame.candidate_frontier(2_000_000).unwrap();
        assert_eq!(frontier.candidate_count(), 2_000_000);
        assert_eq!(frontier.physical_ranges().len(), 1);
        assert!(frontier.is_ancestor_fallback(LodNodeId(1)));
        assert_eq!(
            std::mem::size_of_val(frontier.physical_ranges()),
            std::mem::size_of::<LodPhysicalRange>()
        );
        frame.complete_resident_cut = false;
        assert_eq!(
            frame.candidate_frontier(2_000_000),
            Err(LodRuntimeError::NoResidentFrontier)
        );
    }

    #[test]
    fn candidate_capability_rejects_a_partial_multi_root_forest() {
        let (manifest, transport, settings, streaming) = two_root_fixture();
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();

        let empty = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(!empty.has_complete_resident_cut());
        assert_eq!(
            empty.candidate_frontier(2),
            Err(LodRuntimeError::NoResidentFrontier)
        );

        let verifying = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(verifying.completed_pages().is_empty());
        assert_eq!(verifying.preprocess_stats().submitted, 1);
        assert_eq!(
            verifying.preprocess_stats().cooperative_decoded_gaussians,
            0
        );
        let partial = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(partial.completed_pages().len(), 1);
        assert_eq!(partial.physical_ranges().len(), 1);
        assert_eq!(partial.frontier().requested_nodes.len(), 1);
        assert!(!partial.has_complete_resident_cut());
        assert_eq!(
            partial.candidate_frontier(2),
            Err(LodRuntimeError::NoResidentFrontier)
        );
    }

    #[test]
    fn packed_promoted_guard_uses_one_slot_for_a_deeper_exact_global_partition() {
        let (manifest, _, settings, _) = fixture();
        let hierarchy = CompiledManifestLodHierarchy::new(manifest).unwrap();
        let root_footprint =
            LodRuntimeCoverageGuard::footprint(&hierarchy, hierarchy.roots()).unwrap();
        let guard = LodRuntimeCoverageGuard::new(&hierarchy, &settings).unwrap();

        assert!(guard.nodes.len() > hierarchy.roots().len());
        assert!(guard.active_gaussians > root_footprint.active_gaussians);
        assert_eq!(guard.pages.len(), root_footprint.pages.len());
        assert_ne!(guard.pages, root_footprint.pages);

        let guard_footprint = LodRuntimeCoverageGuard::footprint(&hierarchy, &guard.nodes).unwrap();
        assert_eq!(guard.pages, guard_footprint.pages);
        assert_eq!(guard.active_gaussians, guard_footprint.active_gaussians);
        assert!(guard_footprint.resident_bytes <= settings.budgets.max_resident_bytes);
        assert!(guard_footprint.resident_gaussians <= settings.budgets.max_resident_gaussians);
        assert!(guard.active_gaussians <= settings.budgets.max_active_gaussians);
        assert!(guard.active_gaussians <= u64::from(u32::MAX));

        let mut source_ranges = guard
            .nodes
            .iter()
            .map(|node| hierarchy.node(*node).unwrap().source)
            .collect::<Vec<_>>();
        source_ranges.sort_unstable_by_key(|range| range.start);
        let mut source_cursor = 0_u64;
        for range in source_ranges {
            assert_eq!(range.start, source_cursor);
            source_cursor = range.end().unwrap();
        }
        assert_eq!(
            source_cursor,
            hierarchy.manifest().header.source_gaussian_count,
            "the promoted guard must cover every source Gaussian exactly once"
        );

        let mut representation_ranges = BTreeMap::<LodPageId, Vec<LodPageRange>>::new();
        for &node in &guard.nodes {
            let representation = hierarchy.representation(node).unwrap();
            representation_ranges
                .entry(representation.page)
                .or_default()
                .push(representation);
        }
        for ranges in representation_ranges.values_mut() {
            ranges.sort_unstable_by_key(|range| range.offset);
            for pair in ranges.windows(2) {
                assert!(pair[0].end().unwrap() <= pair[1].offset);
            }
        }
    }

    #[test]
    fn package_bootstrap_planner_selects_the_deepest_complete_payload_bounded_cut() {
        let (hierarchy, _, settings, budget) = package_bootstrap_fixture();
        let guard = LodRuntimeCoverageGuard::new_with_package_bootstrap(
            &hierarchy,
            &settings,
            Some(budget),
        )
        .unwrap();
        assert!(guard.package_bootstrap);
        assert!(guard.nodes.len() > hierarchy.roots().len());

        let footprint = LodRuntimeCoverageGuard::footprint(&hierarchy, &guard.nodes).unwrap();
        assert_eq!(guard.pages, footprint.pages);
        assert!(guard.pages.len() <= budget.max_pages as usize);
        assert!(guard.active_gaussians <= budget.max_active_gaussians);
        assert!(footprint.resident_bytes <= budget.max_decoded_bytes);
        assert!(footprint.encoded_bytes.unwrap() <= budget.max_encoded_bytes);
        assert!(
            u64::try_from(guard.pages.len()).unwrap() * budget.gpu_bytes_per_slot
                <= budget.max_gpu_bytes
        );
        assert!(
            LodRuntimeCoverageGuard::has_transition_headroom(
                &hierarchy,
                &guard.nodes,
                &footprint,
                &settings,
            )
            .unwrap()
        );

        let next = LodRuntimeCoverageGuard::next_complete_level(&hierarchy, &guard.nodes)
            .expect("the bounded bootstrap must stop above the exact leaves");
        let next_footprint = LodRuntimeCoverageGuard::footprint(&hierarchy, &next).unwrap();
        assert!(
            !LodRuntimeCoverageGuard::package_bootstrap_footprint_fits(
                &next_footprint,
                &settings,
                budget,
            ) || !LodRuntimeCoverageGuard::has_transition_headroom(
                &hierarchy,
                &next,
                &next_footprint,
                &settings,
            )
            .unwrap(),
            "the planner stopped before the deepest admissible complete level"
        );

        let mut source_ranges = guard
            .nodes
            .iter()
            .map(|node| hierarchy.node(*node).unwrap().source)
            .collect::<Vec<_>>();
        source_ranges.sort_unstable_by_key(|range| range.start);
        let mut cursor = 0_u64;
        for range in source_ranges {
            assert_eq!(range.start, cursor);
            cursor = range.end().unwrap();
        }
        assert_eq!(cursor, hierarchy.manifest().header.source_gaussian_count);
    }

    #[test]
    fn released_package_bootstrap_still_streams_missing_navigation_roots() {
        let (hierarchy, transport, mut settings, budget) = package_bootstrap_fixture();
        settings.quality = 1.0;
        settings.budgets.max_requests_per_frame = 16;
        settings.budgets.max_upload_bytes_per_frame = settings.budgets.max_resident_bytes;
        let streaming = GaussianStreamingSettings {
            max_concurrent_requests: 16,
            ..Default::default()
        };
        let manifest = hierarchy.manifest().clone();
        let root_pages = hierarchy
            .roots()
            .iter()
            .filter_map(|root| hierarchy.page(*root))
            .collect::<BTreeSet<_>>();
        let mut decode_transport = transport.clone();
        let mut runtime = LodStreamingRuntime::from_compiled_hierarchy(
            hierarchy,
            transport,
            &settings,
            &streaming,
            Some(budget),
        )
        .unwrap();
        assert!(runtime.has_active_package_bootstrap());
        let bootstrap_pages = runtime.coverage_guard.pages.clone();
        assert!(
            root_pages.difference(&bootstrap_pages).next().is_some(),
            "the regression requires navigation roots distinct from the presentation bootstrap"
        );
        seed_resident_coverage_guard(&mut runtime, &manifest, &mut decode_transport, &streaming);
        for &page in &bootstrap_pages {
            runtime.retain_resident_page(page).unwrap();
        }
        runtime.release_package_bootstrap_reserve().unwrap();
        assert_eq!(
            runtime.cache().stats().resident_pages as usize,
            bootstrap_pages.len()
        );

        let starts_before = runtime.transport_request_starts_for_test();
        let mut navigation_started = false;
        let mut progressed = false;
        for _ in 0..256 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            navigation_started |= frame
                .started_pages()
                .iter()
                .any(|page| root_pages.contains(page));
            if frame.cache_stats().resident_pages as usize > bootstrap_pages.len() {
                progressed = true;
                break;
            }
        }
        assert!(
            navigation_started,
            "missing roots must retain ordinary demand"
        );
        assert!(
            progressed,
            "residency must advance beyond the bootstrap cut"
        );
        assert!(runtime.transport_request_starts_for_test() > starts_before);
    }

    #[test]
    fn package_bootstrap_rejects_legacy_reducers_and_missing_transition_capacity() {
        let (hierarchy, _, settings, budget) = package_bootstrap_fixture();
        let generous = LodRuntimeCoverageGuard::new_with_package_bootstrap(
            &hierarchy,
            &settings,
            Some(budget),
        )
        .unwrap();
        assert!(generous.package_bootstrap);
        let root_pages = LodRuntimeCoverageGuard::footprint(&hierarchy, hierarchy.roots())
            .unwrap()
            .pages;
        let mut generous_plus_root = generous.pages.clone();
        generous_plus_root.extend(root_pages.iter().copied());
        assert!(generous_plus_root.len() > generous.pages.len());

        let mut root_headroom_limited = settings.clone();
        root_headroom_limited.budgets.max_resident_pages =
            u32::try_from(generous.pages.len()).unwrap();
        let replanned = LodRuntimeCoverageGuard::new_with_package_bootstrap(
            &hierarchy,
            &root_headroom_limited,
            Some(budget),
        )
        .unwrap();
        assert_ne!(
            replanned.nodes, generous.nodes,
            "the deeper bootstrap alone fits, but its separately resident root does not"
        );
        let mut replanned_plus_root = replanned.pages.clone();
        replanned_plus_root.extend(root_pages.iter().copied());
        assert!(
            replanned_plus_root.len() <= root_headroom_limited.budgets.max_resident_pages as usize
        );

        let mut legacy = hierarchy.manifest().clone();
        legacy.build.builder_abi_version = VIRTUAL_BUILDER_ABI_VERSION;
        legacy.build.reducer_version = EXTERNAL_MOMENT_MERGE_VERSION;
        legacy.build.config_fingerprint = lod_config_fingerprint_for_reducer(
            legacy.build.settings,
            None,
            EXTERNAL_MOMENT_MERGE_VERSION,
        );
        for node in &mut legacy.nodes {
            if !node.is_leaf() {
                node.high_fidelity_certificate = 0.0;
            }
        }
        legacy.validate().unwrap();
        let legacy = CompiledManifestLodHierarchy::new(legacy).unwrap();
        let legacy_guard =
            LodRuntimeCoverageGuard::new_with_package_bootstrap(&legacy, &settings, Some(budget))
                .unwrap();
        assert!(!legacy_guard.package_bootstrap);

        let roots = hierarchy.roots().to_vec();
        let root = LodRuntimeCoverageGuard::footprint(&hierarchy, &roots).unwrap();
        let mut capacity_limited = settings;
        capacity_limited.budgets.max_resident_pages = u32::try_from(root.pages.len()).unwrap();
        capacity_limited.budgets.max_resident_bytes = root.resident_bytes;
        capacity_limited.budgets.max_resident_gaussians = root.resident_gaussians;
        let capacity_guard = LodRuntimeCoverageGuard::new_with_package_bootstrap(
            &hierarchy,
            &capacity_limited,
            Some(budget),
        )
        .unwrap();
        assert!(!capacity_guard.package_bootstrap);
        assert_eq!(capacity_guard.nodes, roots);
        assert_eq!(capacity_guard.pages, root.pages);
    }

    #[test]
    fn forest_guard_keeps_all_roots_when_the_whole_next_level_needs_more_slots() {
        let (manifest, _, settings, _) = fixture();
        let manifest = first_level_promoted_forest(manifest);
        let hierarchy = CompiledManifestLodHierarchy::new(manifest).unwrap();
        let roots = hierarchy.roots().to_vec();
        assert!(roots.len() > 1);
        let root_footprint = LodRuntimeCoverageGuard::footprint(&hierarchy, &roots).unwrap();

        let mut next_nodes = roots
            .iter()
            .flat_map(|root| hierarchy.children(*root).iter().copied())
            .collect::<Vec<_>>();
        next_nodes.sort_unstable();
        let next_footprint = LodRuntimeCoverageGuard::footprint(&hierarchy, &next_nodes).unwrap();
        assert!(next_footprint.pages.len() > root_footprint.pages.len());

        let guard = LodRuntimeCoverageGuard::new(&hierarchy, &settings).unwrap();
        assert_eq!(guard.nodes, roots);
        assert_eq!(guard.pages, root_footprint.pages);
        assert_eq!(guard.active_gaussians, root_footprint.active_gaussians);
    }

    #[test]
    fn promoted_forest_guard_requires_complete_root_transition_headroom() {
        let hierarchy = disjoint_two_root_guard_hierarchy();
        let roots = hierarchy.roots().to_vec();
        let root = LodRuntimeCoverageGuard::footprint(&hierarchy, &roots).unwrap();
        let next_nodes = LodRuntimeCoverageGuard::next_complete_level(&hierarchy, &roots).unwrap();
        let next = LodRuntimeCoverageGuard::footprint(&hierarchy, &next_nodes).unwrap();
        assert_eq!(root.pages.len(), 2);
        assert_eq!(next.pages.len(), 2);
        assert!(root.pages.is_disjoint(&next.pages));

        let mut transition_pages = root.pages.clone();
        transition_pages.extend(next.pages.iter().copied());
        let (transition_bytes, transition_gaussians, _) =
            LodRuntimeCoverageGuard::page_footprint(&hierarchy, &transition_pages).unwrap();
        assert_eq!(transition_pages.len(), 4);
        assert!(transition_bytes > root.resident_bytes.max(next.resident_bytes));
        assert!(transition_gaussians > root.resident_gaussians.max(next.resident_gaussians));

        let (_, _, mut settings, _) = fixture();
        settings.budgets.max_resident_pages = 4;
        settings.budgets.max_resident_bytes = transition_bytes;
        settings.budgets.max_resident_gaussians = transition_gaussians;
        let promoted = LodRuntimeCoverageGuard::new(&hierarchy, &settings).unwrap();
        assert_eq!(promoted.nodes, next_nodes);
        assert_eq!(promoted.pages, next.pages);

        let assert_roots = |settings: &GaussianLodSettings| {
            let guard = LodRuntimeCoverageGuard::new(&hierarchy, settings).unwrap();
            assert_eq!(guard.nodes, roots);
            assert_eq!(guard.pages, root.pages);
        };

        let mut page_limited = settings.clone();
        page_limited.budgets.max_resident_pages = 3;
        assert_roots(&page_limited);

        let mut byte_limited = settings.clone();
        byte_limited.budgets.max_resident_bytes = transition_bytes - 1;
        assert_roots(&byte_limited);

        let mut gaussian_limited = settings;
        gaussian_limited.budgets.max_resident_gaussians = transition_gaussians - 1;
        assert_roots(&gaussian_limited);
    }

    #[test]
    fn promoted_guard_falls_back_atomically_at_each_logical_resource_limit() {
        let (manifest, _, settings, _) = fixture();
        let hierarchy = CompiledManifestLodHierarchy::new(manifest).unwrap();
        let roots = hierarchy.roots().to_vec();
        let root_footprint = LodRuntimeCoverageGuard::footprint(&hierarchy, &roots).unwrap();
        let mut next_nodes = roots
            .iter()
            .flat_map(|root| hierarchy.children(*root).iter().copied())
            .collect::<Vec<_>>();
        next_nodes.sort_unstable();
        let next_footprint = LodRuntimeCoverageGuard::footprint(&hierarchy, &next_nodes).unwrap();
        assert_eq!(next_footprint.pages.len(), root_footprint.pages.len());
        assert!(next_footprint.resident_bytes > root_footprint.resident_bytes);
        assert!(next_footprint.resident_gaussians > root_footprint.resident_gaussians);
        assert!(next_footprint.active_gaussians > root_footprint.active_gaussians);

        let assert_roots = |settings: &GaussianLodSettings| {
            let guard = LodRuntimeCoverageGuard::new(&hierarchy, settings).unwrap();
            assert_eq!(guard.nodes, roots);
            assert_eq!(guard.pages, root_footprint.pages);
            assert_eq!(guard.active_gaussians, root_footprint.active_gaussians);
        };

        let mut byte_limited = settings.clone();
        byte_limited.budgets.max_resident_bytes = root_footprint.resident_bytes;
        assert_roots(&byte_limited);

        let mut gaussian_limited = settings.clone();
        gaussian_limited.budgets.max_resident_gaussians = root_footprint.resident_gaussians;
        assert_roots(&gaussian_limited);

        let mut active_limited = settings;
        active_limited.budgets.max_active_gaussians = root_footprint.active_gaussians;
        assert_roots(&active_limited);
    }

    #[test]
    fn one_slot_coarsest_guard_overlaps_its_target_and_reaches_a_complete_cut() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.quality = 0.0;
        settings.budgets.max_resident_pages = 1;
        settings.budgets.max_resident_gaussians = manifest
            .pages
            .iter()
            .map(|page| u64::from(page.gaussian_count))
            .max()
            .unwrap();
        let max_page_bytes = manifest
            .pages
            .iter()
            .map(|page| page.decoded_len)
            .max()
            .unwrap();
        settings.budgets.max_resident_bytes = max_page_bytes;
        settings.budgets.max_upload_bytes_per_frame = max_page_bytes;

        let roots = manifest.roots.clone();
        let root_pages = roots
            .iter()
            .map(|root| {
                manifest
                    .nodes
                    .iter()
                    .find(|node| node.id == *root)
                    .unwrap()
                    .representation
                    .page
            })
            .collect::<BTreeSet<_>>();
        let root_active = root_active_gaussians(&manifest);
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        runtime.preprocessor = LodPagePreprocessor::new_cooperative_for_tests(4).unwrap();

        assert_eq!(runtime.coverage_guard.nodes, roots);
        assert_eq!(runtime.coverage_guard.pages, root_pages);
        let mut complete = None;
        for _ in 0..128 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert_eq!(frame.capacity_blocked_requests(), 0);
            if frame.has_complete_resident_cut() && frame.candidate_count() == root_active {
                complete = Some(frame);
                break;
            }
        }
        let complete = complete.expect("the one-slot coarsest target must become resident");
        assert_eq!(complete.frontier().nodes, roots);
        complete
            .candidate_frontier(settings.max_active_gaussians_u32())
            .expect("the overlapping root guard must yield a complete render cut");
    }

    #[test]
    fn coverage_guard_is_camera_bound_and_survives_view_removal() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.quality = 0.0;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let guard_pages = runtime.coverage_guard.pages.clone();
        let guard_count = u32::try_from(runtime.coverage_guard.active_gaussians).unwrap();
        let guard_node_count = runtime.coverage_guard.nodes.len();
        let camera = LodRuntimeViewId(77);

        let mut guard = None;
        for _ in 0..128 {
            runtime
                .update_view(camera, view(), &settings, &streaming)
                .unwrap();
            guard = runtime
                .coverage_guard_candidate(camera, view(), &settings)
                .unwrap();
            if guard.is_some() {
                break;
            }
        }
        let guard = guard.expect("the fallback-critical root guard should become resident");
        assert!(guard.is_coverage_guard());
        assert_eq!(guard.view(), camera);
        assert_eq!(guard.candidate_count(), guard_count);
        assert_eq!(guard.physical_ranges().len(), guard_node_count);
        assert!(
            guard
                .physical_ranges()
                .iter()
                .all(|range| !guard.is_ancestor_fallback(range.node))
        );
        assert_eq!(
            guard.quality_status().requested_target,
            LodQualityTarget::Coarsest
        );
        assert_eq!(guard.quality_status().degradation, LodDegradation::None);
        assert!(!guard.selection_view_frozen());
        assert!(
            guard_pages
                .iter()
                .all(|page| runtime.cache().get(*page).unwrap().pin_count >= 1)
        );

        assert!(runtime.remove_view(camera).unwrap());
        assert!(
            guard_pages
                .iter()
                .all(|page| runtime.cache().get(*page).unwrap().pin_count == 1),
            "the runtime-owned guard pins must outlive all camera pins"
        );

        let moved = LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, 80.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        let mut detailed = settings.clone();
        detailed.quality = 1.0;
        let moved_guard = runtime
            .coverage_guard_candidate(LodRuntimeViewId(78), moved, &detailed)
            .unwrap()
            .expect("camera motion cannot invalidate the global guard cut");
        assert!(moved_guard.is_coverage_guard());
        assert_eq!(moved_guard.view(), LodRuntimeViewId(78));
        assert_eq!(moved_guard.physical_ranges(), guard.physical_ranges());
        assert!(
            moved_guard
                .physical_ranges()
                .iter()
                .any(|range| moved_guard.is_ancestor_fallback(range.node))
        );
        assert_eq!(
            moved_guard.quality_status().requested_target,
            LodQualityTarget::Original
        );
        assert_eq!(
            moved_guard.quality_status().degradation,
            LodDegradation::Residency,
            "a rebased coarse guard must not claim original-quality satisfaction"
        );
    }

    #[test]
    fn active_budget_blocked_coverage_guard_is_resident_and_reuses_ideal_selection() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.quality = 1.0;
        settings.budgets.max_active_gaussians = root_active_gaussians(&manifest);
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let camera = LodRuntimeViewId(79);

        let mut guard = None;
        for _ in 0..128 {
            runtime
                .update_view(camera, view(), &settings, &streaming)
                .unwrap();
            guard = runtime
                .coverage_guard_candidate(camera, view(), &settings)
                .unwrap();
            if guard.is_some() {
                break;
            }
        }
        let guard = guard.expect("the active-budget root guard should become resident");
        assert!(guard.quality_status().achieved_max_target_ratio > 1.0);
        assert_eq!(
            guard.quality_status().degradation,
            LodDegradation::ActiveBudget,
            "the all-resident selector, not quality pressure alone, owns the budget diagnosis"
        );
        assert!(
            guard
                .physical_ranges()
                .iter()
                .all(|range| !guard.is_ancestor_fallback(range.node)),
            "a guard node selected even with universal residency is Resident"
        );

        let traversals = runtime.all_resident_selection_traversals;
        let repeated = runtime
            .coverage_guard_candidate(camera, view(), &settings)
            .unwrap()
            .expect("the resident guard remains available");
        assert_eq!(repeated, guard);
        assert_eq!(
            runtime.all_resident_selection_traversals, traversals,
            "an unchanged view/policy/hysteresis key must reuse the bounded ideal-selection cache"
        );
    }

    #[test]
    fn coverage_guard_requests_are_fallback_critical_and_demand_independent() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.quality = 1.0;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let guard_pages = runtime.coverage_guard.pages.clone();

        let requests = (0..guard_pages.len())
            .map(|_| {
                runtime
                    .queue
                    .pop()
                    .expect("construction must prime the global guard before any view update")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            requests
                .iter()
                .map(|request| request.page_id)
                .collect::<BTreeSet<_>>(),
            guard_pages
        );
        for request in requests {
            assert_eq!(
                request.priority,
                PageRequestPriority::fallback_critical(u32::MAX)
            );
            assert!(matches!(
                runtime.queue.enqueue(request),
                RequestEnqueue::Enqueued
            ));
        }

        let frame = runtime.begin_frame();
        runtime.finish_frame(frame).unwrap();
        assert!(
            guard_pages.iter().all(|page| runtime.queue.contains(*page)),
            "frame reconciliation must retain guard demand without a camera update"
        );
    }

    #[test]
    fn coverage_guard_footprint_is_rejected_before_streaming() {
        let (manifest, transport, settings, streaming) = two_root_fixture();

        let mut page_limited = settings.clone();
        page_limited.budgets.max_resident_pages = 1;
        assert!(matches!(
            LodStreamingRuntime::new(
                manifest.clone(),
                transport.clone(),
                &page_limited,
                &streaming,
            ),
            Err(LodRuntimeError::CoverageGuardPagesExceedLimit {
                actual: 2,
                limit: 1,
            })
        ));

        let guard_bytes = manifest
            .pages
            .iter()
            .map(|page| page.decoded_len)
            .sum::<u64>();
        let mut byte_limited = settings.clone();
        byte_limited.budgets.max_resident_bytes = guard_bytes - 1;
        assert!(matches!(
            LodStreamingRuntime::new(
                manifest.clone(),
                transport.clone(),
                &byte_limited,
                &streaming,
            ),
            Err(LodRuntimeError::CoverageGuardBytesExceedLimit { .. })
        ));

        let mut gaussian_limited = settings.clone();
        gaussian_limited.budgets.max_resident_gaussians = 1;
        assert!(matches!(
            LodStreamingRuntime::new(
                manifest.clone(),
                transport.clone(),
                &gaussian_limited,
                &streaming,
            ),
            Err(LodRuntimeError::CoverageGuardGaussiansExceedLimit {
                actual: 2,
                limit: 1,
            })
        ));

        let mut active_limited = settings;
        active_limited.budgets.max_active_gaussians = 1;
        assert!(matches!(
            LodStreamingRuntime::new(manifest, transport, &active_limited, &streaming),
            Err(LodRuntimeError::CoverageGuardActiveGaussiansExceedLimit {
                actual: 2,
                limit: 1,
            })
        ));
    }

    #[test]
    fn cameras_share_residency_but_keep_independent_fallback_holds() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.quality = 0.0;
        let root_node = manifest.roots[0];
        let root = manifest
            .nodes
            .iter()
            .find(|node| node.id == root_node)
            .unwrap();
        let root_page = root.representation.page;
        let root_candidate_count = u64::from(root.representation.count);
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let left = LodRuntimeViewId(11);
        let right = LodRuntimeViewId(12);

        let first = runtime
            .update_view(left, view(), &settings, &streaming)
            .unwrap();
        assert_eq!(first.candidate_count(), 0);
        let mut left_resident = false;
        for _ in 0..128 {
            let frame = runtime
                .update_view(left, view(), &settings, &streaming)
                .unwrap();
            if frame.candidate_count() == root_candidate_count {
                left_resident = true;
                break;
            }
        }
        assert!(left_resident, "the camera root should become resident");
        let right_frame = runtime
            .update_view(right, view(), &settings, &streaming)
            .unwrap();
        assert_eq!(right_frame.candidate_count(), root_candidate_count);
        assert!(!runtime.coverage_guard.pages.contains(&root_page));
        assert_eq!(runtime.cache().get(root_page).unwrap().pin_count, 2);
        assert!(
            runtime.coverage_guard.pages.iter().all(|page| runtime
                .cache()
                .get(*page)
                .unwrap()
                .pin_count
                == 1)
        );

        assert!(runtime.remove_view(left).unwrap());
        assert_eq!(runtime.cache().get(root_page).unwrap().pin_count, 1);
        assert!(runtime.remove_view(right).unwrap());
        assert_eq!(runtime.cache().get(root_page).unwrap().pin_count, 0);
        assert!(!runtime.remove_view(right).unwrap());
    }

    #[test]
    fn cameras_in_one_frame_share_request_and_decoded_byte_budgets() {
        let (manifest, transport, mut settings, streaming) = fixture();
        let page_bytes = manifest
            .pages
            .iter()
            .map(|descriptor| (descriptor.id, descriptor.decoded_len))
            .collect::<BTreeMap<_, _>>();
        let decoded_budget = page_bytes.values().copied().max().unwrap();
        settings.budgets.max_requests_per_frame = 1;
        settings.budgets.max_upload_bytes_per_frame = decoded_budget;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let left = LodRuntimeViewId(21);
        let right = LodRuntimeViewId(22);
        let mut previous_frame = None;

        for _ in 0..16 {
            let frame = runtime.begin_frame();
            previous_frame = Some(frame);
            let left_frame = runtime
                .update_view_in_frame(frame, left, view(), &settings, &streaming)
                .unwrap();
            let right_frame = runtime
                .update_view_in_frame(frame, right, view(), &settings, &streaming)
                .unwrap();
            assert!(left_frame.started_pages.len() + right_frame.started_pages.len() <= 1);
            let completed_bytes = left_frame
                .completed_pages
                .iter()
                .chain(&right_frame.completed_pages)
                .map(|page| page_bytes[page])
                .sum::<u64>();
            assert!(completed_bytes <= decoded_budget);
        }

        let stale = previous_frame.unwrap();
        let current = runtime.begin_frame();
        assert!(matches!(
            runtime.update_view_in_frame(stale, left, view(), &settings, &streaming),
            Err(LodRuntimeError::InvalidFrameToken {
                expected,
                actual
            }) if expected == current && actual == stale
        ));
    }

    #[test]
    fn atlas_addressing_is_rejected_before_any_gpu_sized_allocation() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_resident_pages = u32::MAX;
        assert!(matches!(
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming),
            Err(LodRuntimeError::AtlasAddressSpaceOverflow { .. })
        ));
    }

    #[test]
    fn aggregate_pending_budget_includes_queue_in_flight_and_capacity_blocked() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_pending_requests = 4;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let pages = runtime
            .hierarchy
            .manifest()
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .filter(|page| !runtime.coverage_guard.pages.contains(page))
            .take(4)
            .collect::<Vec<_>>();
        assert_eq!(pages.len(), 4);
        let request = |page| PageRequest::new(page, PageRequestPriority::visible(1));

        runtime.in_flight.insert(
            pages[0],
            InFlight {
                ticket: 101,
                request: request(pages[0]),
            },
        );
        runtime.in_flight.insert(
            pages[1],
            InFlight {
                ticket: 102,
                request: request(pages[1]),
            },
        );
        runtime.capacity_blocked.insert(pages[2], request(pages[2]));
        assert_eq!(runtime.queue.len(), runtime.coverage_guard.pages.len());
        assert_eq!(runtime.pending_request_count(), 4);
        assert_eq!(
            runtime.enqueue_pending_request(request(pages[3])),
            RequestEnqueue::Rejected
        );
        assert_eq!(runtime.pending_request_count(), 4);
    }

    #[test]
    fn manifest_pages_are_rejected_before_transport_or_decode_when_over_budget() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_upload_bytes_per_frame = 1;
        assert!(matches!(
            LodStreamingRuntime::new(manifest.clone(), transport.clone(), &settings, &streaming),
            Err(LodRuntimeError::PageDecodedBytesExceedLimit { .. })
        ));

        let (_, _, settings, mut streaming) = fixture();
        streaming.max_encoded_page_bytes = 44;
        assert!(matches!(
            LodStreamingRuntime::new(manifest.clone(), transport.clone(), &settings, &streaming),
            Err(LodRuntimeError::PageEncodedBytesExceedLimit { .. })
        ));

        streaming.max_encoded_page_bytes = 43;
        assert!(matches!(
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming),
            Err(LodRuntimeError::EncodedPageLimitTooSmall {
                limit: 43,
                minimum: 44
            })
        ));
    }

    #[test]
    fn structural_reconfiguration_is_explicit_and_too_small_frame_budget_cannot_stall() {
        let (manifest, transport, mut settings, streaming) = fixture();
        let original_upload_budget = settings.budgets.max_upload_bytes_per_frame;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let started = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(!started.started_pages.is_empty());

        settings.budgets.max_upload_bytes_per_frame = 1;
        assert!(matches!(
            runtime.update(view(), &settings, &streaming),
            Err(LodRuntimeError::PageDecodedBytesExceedLimit { limit: 1, .. })
        ));
        settings.budgets.max_upload_bytes_per_frame = original_upload_budget;

        settings.budgets.max_resident_pages -= 1;
        assert_eq!(
            runtime.update(view(), &settings, &streaming),
            Err(LodRuntimeError::StructuralSettingsChanged(
                "budgets.max_resident_pages"
            ))
        );
    }

    #[test]
    fn ready_pages_respect_the_cumulative_decoded_byte_budget_per_update() {
        let (manifest, transport, mut settings, streaming) = fixture();
        let page_bytes = manifest
            .pages
            .iter()
            .map(|descriptor| (descriptor.id, descriptor.decoded_len))
            .collect::<BTreeMap<_, _>>();
        let frame_budget = page_bytes.values().copied().max().unwrap();
        settings.budgets.max_upload_bytes_per_frame = frame_budget;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();

        for _ in 0..16 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            let completed_bytes = frame
                .completed_pages
                .iter()
                .map(|page| page_bytes[page])
                .sum::<u64>();
            assert!(completed_bytes <= frame_budget);
        }
    }

    #[test]
    fn coverage_guard_survives_pinned_cache_pressure_and_view_removal() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.quality = 0.0;
        settings.budgets.max_resident_pages = 1;
        settings.budgets.max_pending_requests = 2;
        settings.budgets.max_resident_gaussians = manifest
            .pages
            .iter()
            .map(|page| u64::from(page.gaussian_count))
            .max()
            .unwrap();
        let max_page_bytes = manifest
            .pages
            .iter()
            .map(|page| page.decoded_len)
            .max()
            .unwrap();
        settings.budgets.max_resident_bytes = max_page_bytes;
        settings.budgets.max_upload_bytes_per_frame = max_page_bytes;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let guard_page = *runtime.coverage_guard.pages.first().unwrap();

        runtime.update(view(), &settings, &streaming).unwrap();
        runtime.update(view(), &settings, &streaming).unwrap();
        settings.quality = 1.0;

        let mut stalled = None;
        for _ in 0..64 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert!(
                frame.queued_requests + frame.in_flight_requests + frame.capacity_blocked_requests
                    <= settings.budgets.max_pending_requests
            );
            if frame.split_cohort_capacity_stall().is_some()
                && frame.in_flight_requests == 0
                && frame.queued_requests == 0
            {
                stalled = frame.split_cohort_capacity_stall();
                break;
            }
        }
        let stalled = stalled.expect("the impossible pinned substitution must report a stall");
        assert_eq!(stalled.limit_pages, 1);
        assert!(stalled.required_pages > stalled.limit_pages);
        assert!(runtime.capacity_blocked.is_empty());
        for _ in 0..8 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert!(frame.started_pages.is_empty());
            assert_eq!(frame.split_cohort_capacity_stall(), Some(stalled));
        }
        assert_eq!(runtime.coverage_guard.pinned_pages.len(), 1);
        assert_eq!(runtime.cache().get(guard_page).unwrap().pin_count, 2);

        assert!(runtime.remove_view(LodRuntimeViewId::default()).unwrap());
        let guard = runtime
            .coverage_guard_candidate(LodRuntimeViewId(99), view(), &settings)
            .unwrap()
            .expect("cache pressure and view removal must preserve the resident guard");
        assert!(guard.is_coverage_guard());
        assert_eq!(runtime.coverage_guard.pinned_pages.len(), 1);
        assert_eq!(runtime.cache().get(guard_page).unwrap().pin_count, 1);
        assert_eq!(
            guard.physical_ranges().len(),
            runtime.coverage_guard.nodes.len()
        );
        assert_eq!(runtime.pending_request_count(), 0);
        assert!(runtime.capacity_blocked.is_empty());
        assert!(runtime.queue.is_empty());
    }

    #[test]
    fn invalid_support_payload_exhausts_retry_budget_instead_of_redownloading_forever() {
        let scene = LodTestScene::screen_space_ladder();
        let mut lod = build_planar_3d_lod(
            &scene.cloud(),
            GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 16,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        let page_id = root_page(&lod.manifest);
        let page = lod
            .pages
            .iter_mut()
            .find(|page| page.id == page_id)
            .unwrap();
        page.gaussians[0].position_visibility.position[0] += 1_000_000.0;
        let encoded = encode_page(page).unwrap();
        let descriptor = lod
            .manifest
            .pages
            .iter_mut()
            .find(|descriptor| descriptor.id == page_id)
            .unwrap();
        descriptor.content_hash = page.content_hash();
        descriptor.storage = Some(LodPageStorage {
            uri: "memory://invalid-support".to_owned(),
            byte_range: None,
            encoded_len: encoded.len() as u64,
        });
        lod.manifest.validate().unwrap();

        let mut transport = MemoryPageTransport::default();
        transport.insert(page_id, encoded);
        let mut settings = GaussianLodSettings::default();
        settings.quality = 0.0;
        settings.budgets.max_active_gaussians = root_active_gaussians(&lod.manifest);
        let streaming = GaussianStreamingSettings {
            max_concurrent_requests: 1,
            retry_limit: 0,
            ..Default::default()
        };
        let mut runtime =
            LodStreamingRuntime::new(lod.manifest, transport, &settings, &streaming).unwrap();

        let first = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(first.started_pages, vec![page_id]);
        let verifying = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(verifying.failed_pages.is_empty());
        assert_eq!(verifying.preprocess_stats().submitted, 1);
        assert_eq!(
            verifying.preprocess_stats().cooperative_decoded_gaussians,
            0
        );
        let failed = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(failed.failed_pages, vec![page_id]);
        assert!(runtime.is_terminal_failure(page_id));
        for _ in 0..4 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert!(frame.started_pages.is_empty());
            assert!(frame.failed_pages.is_empty());
        }
    }

    #[test]
    fn decoded_support_must_stay_inside_advertised_page_bounds() {
        let scene = LodTestScene::screen_space_ladder();
        let mut lod = build_planar_3d_lod(
            &scene.cloud(),
            GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 16,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        let page = &mut lod.pages[0];
        let descriptor = lod
            .manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == page.id)
            .unwrap();
        page.gaussians[0].position_visibility.position[0] += 1_000_000.0;
        assert_eq!(
            crate::stream::preprocess::validate_decoded_page_bounds(page, descriptor, 3.0),
            Err(LodPagePreprocessError::PayloadOutsideDescriptor(page.id))
        );
    }
}
