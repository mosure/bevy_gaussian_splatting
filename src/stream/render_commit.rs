//! Shared CPU-to-render-world commit contracts for streamed LoD clouds.
//!
//! Both ephemeral flat-cloud bridges and prebuilt packages use these types.
//! Keeping the two-phase handshake and atlas mirror here prevents either
//! orchestration frontend from owning renderer-facing state used by the other.

use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

use bevy::prelude::*;
use bevy_interleave::prelude::Planar;

use crate::{
    gaussian::formats::{
        planar_3d::{Gaussian3d, PlanarGaussian3d},
        planar_3d_chunked::{LodPageId, PlanarGaussian3dPage},
    },
    gaussian::lod_settings::{LodDegradation, LodEffectiveStatus},
    stream::{
        cache::AtlasSlot,
        runtime::{LodCandidateFrontier, LodPhysicalRange, PageAtlasLayout},
    },
};

pub(crate) const LOD_RENDER_WAITING: u8 = 0;
pub(crate) const LOD_RENDER_PREPARED: u8 = 1;
pub(crate) const LOD_RENDER_ACTIVE: u8 = 2;
pub(crate) const LOD_RENDER_FAILED: u8 = 3;

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

#[derive(Default)]
pub(crate) struct GaussianLodRenderCommitPlugin;

impl Plugin for GaussianLodRenderCommitPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<LodOrchestrationTransition>();
    }
}

/// One render-world commit guarded by a cross-world two-phase handshake.
#[derive(Clone, Debug)]
pub struct LodRenderCandidate {
    pub(crate) frontier: LodCandidateFrontier,
    pub(crate) phase: Arc<AtomicU8>,
    /// Some high-detail ephemeral cuts render the retained flat source when
    /// compaction would save no more than the fixed bridge threshold. Package
    /// candidates and worthwhile ephemeral cuts keep this unset and use atlas
    /// compaction.
    retained_flat_source_count: Option<u32>,
}

/// Extracted on the cloud entity and consumed automatically by render LoD.
#[derive(Component, Clone, Debug, Default)]
pub struct LodRenderCandidates {
    pub(crate) by_camera: BTreeMap<Entity, LodRenderCandidate>,
}

impl LodRenderCandidate {
    /// Creates a two-phase commit for a runtime-validated complete frontier.
    pub fn new(frontier: LodCandidateFrontier) -> Self {
        Self {
            frontier,
            phase: Arc::new(AtomicU8::new(LOD_RENDER_WAITING)),
            retained_flat_source_count: None,
        }
    }

    pub(crate) fn with_phase(frontier: LodCandidateFrontier, phase: Arc<AtomicU8>) -> Self {
        Self {
            frontier,
            phase,
            retained_flat_source_count: None,
        }
    }

    /// Publishes selector provenance for status/freeze consumers while the
    /// renderer uses the exact retained flat asset directly.
    pub(crate) fn retained_flat_source(
        frontier: LodCandidateFrontier,
        source_gaussian_count: u32,
    ) -> Self {
        Self {
            frontier,
            phase: Arc::new(AtomicU8::new(LOD_RENDER_ACTIVE)),
            retained_flat_source_count: Some(source_gaussian_count),
        }
    }

    /// Logical selector provenance. The rendered count and quality can be more
    /// exact when a marginal high-detail cut uses the retained flat source;
    /// use [`Self::rendered_candidate_count`] and
    /// [`Self::rendered_quality_status`] for render-facing diagnostics.
    pub fn frontier(&self) -> &LodCandidateFrontier {
        &self.frontier
    }

    /// Generation-safe atlas ranges referenced by this rendered frame. A
    /// retained-flat-source candidate references no atlas ranges.
    pub fn render_ranges(&self) -> &[LodPhysicalRange] {
        if self.retained_flat_source_count.is_some() {
            &[]
        } else {
            self.frontier.physical_ranges()
        }
    }

    /// Number of Gaussians actually submitted by the selected render source.
    pub fn rendered_candidate_count(&self) -> u32 {
        self.retained_flat_source_count
            .unwrap_or_else(|| self.frontier.candidate_count())
    }

    /// Quality observation for what is actually rendered. Selector provenance
    /// remains available through [`Self::frontier`], but a retained flat source
    /// is exact even when the bypassed logical cut was only near-complete.
    pub fn rendered_quality_status(&self) -> LodEffectiveStatus {
        let status = *self.frontier.quality_status();
        self.retained_flat_source_count
            .map_or(status, |source_count| {
                retained_flat_source_quality_status(status, source_count)
            })
    }

    pub(crate) fn same_payload(&self, other: &Self) -> bool {
        self.retained_flat_source_count == other.retained_flat_source_count
            && self.frontier.view() == other.frontier.view()
            && self.frontier.candidate_count() == other.frontier.candidate_count()
            && self.frontier.physical_ranges() == other.frontier.physical_ranges()
    }

    /// Whether this candidate needs atlas compaction and active-entry radix.
    pub(crate) const fn requires_compaction(&self) -> bool {
        self.retained_flat_source_count.is_none()
    }

    pub fn render_is_prepared(&self) -> bool {
        matches!(
            self.phase.load(Ordering::Acquire),
            LOD_RENDER_PREPARED | LOD_RENDER_ACTIVE
        )
    }

    /// Activates a staged candidate after its referenced pages have been
    /// materialized into the atlas.
    pub fn activate(&self) -> bool {
        if !self.render_is_prepared() || self.failed() {
            return false;
        }
        self.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
        true
    }

    pub fn failed(&self) -> bool {
        self.phase.load(Ordering::Acquire) == LOD_RENDER_FAILED
    }
}

fn retained_flat_source_quality_status(
    mut status: LodEffectiveStatus,
    source_gaussian_count: u32,
) -> LodEffectiveStatus {
    status.achieved_max_error_px = 0.0;
    status.achieved_max_target_ratio = 0.0;
    status.degradation = LodDegradation::None;
    status.active_gaussians = u64::from(source_gaussian_count);
    status
}

impl LodRenderCandidates {
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
        if atlas.len() != self.physical_gaussians as usize {
            return Err(LodRenderCommitError::AtlasLengthMismatch {
                expected: self.physical_gaussians,
                actual: atlas.len(),
            });
        }
        let start = slot
            .index
            .checked_mul(self.layout.gaussians_per_slot)
            .ok_or(LodRenderCommitError::AtlasSizeOverflow)? as usize;
        let end = start + self.layout.gaussians_per_slot as usize;
        for index in start..end {
            Planar::set(atlas, index, Gaussian3d::default());
        }
        for (offset, gaussian) in page.gaussians.iter().copied().enumerate() {
            Planar::set(atlas, start + offset, gaussian);
        }
        self.slots[slot.index as usize]
            .as_mut()
            .expect("validated staged slot")
            .materialized = true;
        Ok(())
    }

    pub(crate) fn materialized_slots(&self) -> Vec<AtlasSlot> {
        self.slots
            .iter()
            .flatten()
            .filter_map(|record| record.materialized.then_some(record.slot))
            .collect()
    }

    pub(crate) fn mark_fallback_materialized(&mut self) {
        for record in self.slots.iter_mut().flatten() {
            record.materialized = false;
        }
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
    use crate::gaussian::lod_settings::LodQualityTarget;

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
    fn retained_flat_source_status_reports_the_actual_exact_draw() {
        let selected = LodEffectiveStatus {
            requested_target: LodQualityTarget::Balanced {
                detail_fraction: 0.9,
                max_error_px: 0.5,
            },
            achieved_max_error_px: 1.25,
            achieved_max_target_ratio: 1.1,
            degradation: LodDegradation::Residency,
            active_gaussians: 950,
            visited_nodes: 37,
            requested_pages: 2,
        };

        let rendered = retained_flat_source_quality_status(selected, 1_000);
        assert_eq!(rendered.requested_target, selected.requested_target);
        assert_eq!(rendered.achieved_max_error_px, 0.0);
        assert_eq!(rendered.achieved_max_target_ratio, 0.0);
        assert_eq!(rendered.degradation, LodDegradation::None);
        assert_eq!(rendered.active_gaussians, 1_000);
        assert_eq!(rendered.visited_nodes, selected.visited_nodes);
        assert_eq!(rendered.requested_pages, selected.requested_pages);
    }
}
