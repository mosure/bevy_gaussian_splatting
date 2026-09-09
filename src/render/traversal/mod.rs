//! Bounded resident-hierarchy traversal for point and globally ordered renderers.
//!
//! A publisher supplies immutable topology and a fenced page-placement snapshot.
//! Selection replaces a parent only after every child is resident and both the
//! record and node budgets admit the complete cohort. Incomplete child cohorts
//! request all their pages together, retaining resident siblings during loading.
//! Their prospective split also reserves record and node capacity for the frame;
//! unrelated pending cohorts cannot repeatedly spend the same available budget.
//! GPU output is independent of the CPU frontier handshake.
//! The initial GPU profile supports dynamic selection with discrete page cuts;
//! frozen selection is rejected until its full publication semantics are supported.

mod acknowledgement;
mod admission;
mod gpu;
mod snapshot;

pub(crate) use admission::fixed_source;

pub use acknowledgement::{
    GpuLodDrawAcknowledgement, GpuLodDrawAcknowledgements, GpuLodDrawRenderer,
};
pub use gpu::{
    GpuLodTraversalFeedback, GpuLodTraversalFeedbacks, GpuLodTraversalOutput,
    GpuLodTraversalOutputs, GpuLodTraversalPlugin, GpuLodTraversalPrepare, GpuLodTraversalRender,
};
pub use snapshot::{
    GpuLodHierarchy, GpuLodHierarchySnapshot, GpuLodHierarchyTree, GpuLodPagePlacement,
    GpuLodTraversalSettings,
};

/// A logical range whose entire current-view support was conservatively excluded.
pub const OMITTED_OUTSIDE_VIEW: u32 = 1;

/// Diagnostic identity captured when a renderer encodes this producer's input.
/// Residency can remain unchanged across many different camera submissions.
#[cfg(feature = "testing")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GpuLodTraversalCaptureInput {
    pub cloud: bevy::prelude::Entity,
    pub source: bevy::asset::AssetId<crate::PlanarGaussian3d>,
    pub residency_generation: u64,
    pub allocation_generation: u64,
    pub submission: u64,
}

#[cfg(feature = "testing")]
impl From<&GpuLodTraversalOutput> for GpuLodTraversalCaptureInput {
    fn from(output: &GpuLodTraversalOutput) -> Self {
        Self {
            cloud: output.main_cloud,
            source: output.source,
            residency_generation: output.residency_generation,
            allocation_generation: output.generation,
            submission: output.submission,
        }
    }
}
