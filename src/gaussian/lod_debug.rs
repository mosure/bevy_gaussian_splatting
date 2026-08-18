//! Configurable LoD debug annotations and their compact GPU metadata contract.
//!
//! [`LodDebugSettings`] is intentionally independent from streaming policy. It
//! controls presentation only, while [`LodDebugMetadata`] supplies optional
//! per-resident-record facts. Metadata can be constructed a page at a time so
//! an out-of-core runtime never has to allocate annotations for the full
//! virtual scene.

use std::{collections::HashMap, error::Error, fmt, ops::Range, sync::Arc};

use bevy::prelude::*;
use bevy::render::extract_component::ExtractComponent;
use bevy_args::ValueEnum;
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::gaussian::formats::{
    planar_3d::Gaussian3d,
    planar_3d_chunked::{
        LodBounds, LodNodeId, LodPageDescriptor, LodPageId, LodPageRange, PlanarGaussian3dPage,
    },
    planar_3d_lod::{
        GaussianLodManifest, GaussianLodNode, LodValidationError, PlanarGaussian3dLod,
        gaussian_support_bounds,
    },
};
use crate::stream::cache::AtlasSlot;

/// Coherent, self-configuring LoD debug views for viewers and inspectors.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize, ValueEnum,
)]
pub enum LodDebugPreset {
    /// Preserve authored appearance and disable hierarchy annotations.
    #[default]
    Off,
    /// Categorical hierarchy level colors.
    Level,
    /// Stable categorical page colors.
    Page,
    /// Resident/fallback provenance colors.
    Residency,
    /// Support-aware hierarchy boundaries over authored appearance.
    Boundaries,
    /// Current-view projected selector pressure relative to the quality target.
    SelectionPressure,
}

impl LodDebugPreset {
    /// Stable shader ABI code. These values are covered by contract tests.
    pub(crate) const fn shader_code(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Level => 1,
            Self::Page => 2,
            Self::Residency => 3,
            Self::Boundaries => 4,
            Self::SelectionPressure => 5,
        }
    }
}

/// Cloud-level LoD annotation configuration.
///
/// A single named preset is the complete public control. Palette, boundary,
/// and residency styling are deliberately fixed so screenshots remain
/// comparable across scenes and no scene-specific normalization is required.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize)]
#[serde(default)]
pub struct LodDebugSettings {
    pub preset: LodDebugPreset,
}

impl LodDebugSettings {
    /// Change the complete debug view in one operation.
    pub fn apply_preset(&mut self, preset: LodDebugPreset) {
        self.preset = preset;
    }

    pub const fn from_preset(preset: LodDebugPreset) -> Self {
        Self { preset }
    }

    /// Whether the render path needs per-Gaussian metadata.
    #[inline]
    pub const fn requires_metadata(&self) -> bool {
        !matches!(self.preset, LodDebugPreset::Off)
    }
}

/// Runtime provenance of a resident Gaussian representation.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize)]
#[repr(u32)]
pub enum LodDebugResidency {
    /// No runtime provenance was supplied (for example a static flattened LoD).
    #[default]
    Unknown = 0,
    /// The selected page itself was resident.
    Resident = 1,
    /// The runtime substituted a resident ancestor for a missing selected page.
    AncestorFallback = 2,
}

/// Compact storage-buffer record. The layout exactly matches `LodDebugRecord`
/// in `lod_debug.wgsl`. The final fields carry the owning node's exact
/// local-space bounding sphere so the view-dependent debug score can reproduce
/// selector geometry without a CPU rebuild or a per-Gaussian approximation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct LodDebugRecord {
    /// Stable 32-bit mix of the 64-bit page identifier.
    pub page_color_key: u32,
    pub hierarchy_depth: u32,
    /// Low 16 bits contain [`LodDebugResidency`]; high 16 bits contain the
    /// owning node's high-fidelity certificate as UNORM16.
    pub residency: u32,
    /// IEEE-754 bits of support-aware normalized distance to the nearest face.
    /// The sign bit is reserved as an exact-original representation flag;
    /// [`Self::boundary_distance`] masks it before decoding.
    pub boundary_distance_bits: u32,
    pub geometric_error: f32,
    /// Midpoint of the owning node's validated structural-quality interval.
    pub quality_threshold: f32,
    /// Center of the owning node's conservative local-space support sphere.
    pub node_center: [f32; 3],
    /// Radius of the owning node's conservative local-space support sphere.
    pub node_radius: f32,
}

impl LodDebugRecord {
    pub fn for_gaussian(
        node: &GaussianLodNode,
        gaussian: &Gaussian3d,
        support_sigma: f32,
        residency: LodDebugResidency,
    ) -> Result<Self, LodDebugMetadataError> {
        Self::for_node_fields(
            node.id,
            node.depth,
            node.bounds,
            node.representation,
            node.error.geometric,
            lod_debug_quality_threshold(node.quality.min, node.quality.max),
            node.high_fidelity_certificate,
            node.is_leaf(),
            gaussian,
            support_sigma,
            residency,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn for_node_fields(
        node_id: LodNodeId,
        depth: u16,
        bounds: LodBounds,
        representation: LodPageRange,
        geometric_error: f32,
        quality_threshold: f32,
        high_fidelity_certificate: f32,
        is_original_representation: bool,
        gaussian: &Gaussian3d,
        support_sigma: f32,
        residency: LodDebugResidency,
    ) -> Result<Self, LodDebugMetadataError> {
        let support = gaussian_support_bounds(gaussian, support_sigma)
            .map_err(|_| LodDebugMetadataError::InvalidGaussian(node_id))?;
        let boundary_distance = normalized_support_boundary_distance(bounds, support);
        let boundary_distance_bits = boundary_distance.to_bits()
            | if is_original_representation {
                LOD_DEBUG_ORIGINAL_REPRESENTATION_BIT
            } else {
                0
            };
        let node_center = bounds.center();
        let node_radius = bounds.radius();
        Ok(Self {
            page_color_key: stable_page_color_key(representation.page),
            hierarchy_depth: u32::from(depth),
            residency: pack_lod_debug_residency_certificate(residency, high_fidelity_certificate),
            boundary_distance_bits,
            geometric_error,
            quality_threshold,
            node_center,
            node_radius,
        })
    }

    #[inline]
    pub fn boundary_distance(self) -> f32 {
        f32::from_bits(self.boundary_distance_bits & !LOD_DEBUG_ORIGINAL_REPRESENTATION_BIT)
    }

    #[inline]
    pub const fn is_original_representation(self) -> bool {
        self.boundary_distance_bits & LOD_DEBUG_ORIGINAL_REPRESENTATION_BIT != 0
    }

    /// Runtime residency code stored in the low half of the packed word.
    #[inline]
    pub const fn residency_code(self) -> u32 {
        self.residency & LOD_DEBUG_RESIDENCY_MASK
    }

    /// Builder-authored high-fidelity certificate decoded from UNORM16.
    #[inline]
    pub fn high_fidelity_certificate(self) -> f32 {
        ((self.residency >> LOD_DEBUG_CERTIFICATE_SHIFT) & LOD_DEBUG_CERTIFICATE_MAX) as f32
            / LOD_DEBUG_CERTIFICATE_MAX as f32
    }
}

const LOD_DEBUG_ORIGINAL_REPRESENTATION_BIT: u32 = 1 << 31;
const LOD_DEBUG_RESIDENCY_MASK: u32 = 0x0000_ffff;
const LOD_DEBUG_CERTIFICATE_SHIFT: u32 = 16;
const LOD_DEBUG_CERTIFICATE_MAX: u32 = 0x0000_ffff;
const LOD_DEBUG_CERTIFICATE_MIN_USABLE_CODE: u32 = 2;

#[inline]
fn pack_lod_debug_residency_certificate(
    residency: LodDebugResidency,
    high_fidelity_certificate: f32,
) -> u32 {
    let certificate = if high_fidelity_certificate.is_finite() {
        high_fidelity_certificate.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut certificate_code = (certificate * LOD_DEBUG_CERTIFICATE_MAX as f32).round() as u32;
    // CPU selection treats values strictly above one UNORM16 quantum as
    // certified. Preserve that boundary after rounding: code 1 remains the
    // legacy/tiny sentinel, while the smallest usable value is promoted to 2.
    if certificate > 1.0 / LOD_DEBUG_CERTIFICATE_MAX as f32
        && certificate_code < LOD_DEBUG_CERTIFICATE_MIN_USABLE_CODE
    {
        certificate_code = LOD_DEBUG_CERTIFICATE_MIN_USABLE_CODE;
    }
    (certificate_code << LOD_DEBUG_CERTIFICATE_SHIFT) | residency as u32
}

#[inline]
fn lod_debug_quality_threshold(min: f32, max: f32) -> f32 {
    min + (max - min) * 0.5
}

/// Cheap-to-clone annotation payload aligned with the attached GPU cloud.
///
/// For streaming atlases, construct and upload one page with
/// [`Self::records_for_page`] at the same physical offset as its Gaussians.
#[derive(Clone, Component, Debug, Default, ExtractComponent)]
pub struct LodDebugMetadata {
    records: Arc<[LodDebugRecord]>,
}

/// Validated page-to-manifest lookup used by incremental debug annotation.
///
/// Construct this once for a manifest and reuse it for every decoded page.
/// This turns per-page descriptor and node discovery into constant-time page
/// lookup plus iteration over only the nodes represented by that page. The
/// index owns only payload-validation descriptors and annotation-relevant node
/// fields, so it can live beside runtime manifest state without borrowing or
/// cloning transport URIs and traversal-only data.
#[derive(Clone, Debug)]
pub struct LodDebugManifestIndex {
    support_sigma: f32,
    descriptors: Vec<LodPageDescriptor>,
    nodes: Vec<LodDebugIndexedNode>,
    descriptor_by_page: HashMap<LodPageId, usize>,
    node_indices_by_descriptor: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, Debug)]
struct LodDebugIndexedNode {
    id: LodNodeId,
    depth: u16,
    bounds: LodBounds,
    representation: LodPageRange,
    geometric_error: f32,
    quality_threshold: f32,
    high_fidelity_certificate: f32,
    is_original_representation: bool,
}

impl From<&GaussianLodNode> for LodDebugIndexedNode {
    fn from(node: &GaussianLodNode) -> Self {
        Self {
            id: node.id,
            depth: node.depth,
            bounds: node.bounds,
            representation: node.representation,
            geometric_error: node.error.geometric,
            quality_threshold: lod_debug_quality_threshold(node.quality.min, node.quality.max),
            high_fidelity_certificate: node.high_fidelity_certificate,
            is_original_representation: node.is_leaf(),
        }
    }
}

impl LodDebugManifestIndex {
    /// Validate `manifest` and build its reusable debug lookup.
    pub fn new(manifest: &GaussianLodManifest) -> Result<Self, LodDebugMetadataError> {
        manifest
            .validate()
            .map_err(LodDebugMetadataError::InvalidLod)?;

        let mut descriptors = Vec::new();
        descriptors
            .try_reserve_exact(manifest.pages.len())
            .map_err(|_| LodDebugMetadataError::AllocationFailed(manifest.pages.len()))?;
        let mut descriptor_by_page = HashMap::new();
        descriptor_by_page
            .try_reserve(manifest.pages.len())
            .map_err(|_| LodDebugMetadataError::AllocationFailed(manifest.pages.len()))?;
        for (descriptor_index, descriptor) in manifest.pages.iter().enumerate() {
            descriptor_by_page.insert(descriptor.id, descriptor_index);
            descriptors.push(LodPageDescriptor {
                id: descriptor.id,
                kind: descriptor.kind,
                encoding: descriptor.encoding,
                gaussian_count: descriptor.gaussian_count,
                decoded_len: descriptor.decoded_len,
                content_hash: descriptor.content_hash,
                bounds: descriptor.bounds,
                storage: None,
            });
        }

        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(manifest.nodes.len())
            .map_err(|_| LodDebugMetadataError::AllocationFailed(manifest.nodes.len()))?;
        nodes.extend(manifest.nodes.iter().map(LodDebugIndexedNode::from));

        let mut node_counts = Vec::new();
        node_counts
            .try_reserve_exact(manifest.pages.len())
            .map_err(|_| LodDebugMetadataError::AllocationFailed(manifest.pages.len()))?;
        node_counts.resize(manifest.pages.len(), 0_usize);
        for node in &manifest.nodes {
            let descriptor_index = descriptor_by_page
                .get(&node.representation.page)
                .copied()
                .ok_or(LodDebugMetadataError::UnknownPage(node.representation.page))?;
            node_counts[descriptor_index] = node_counts[descriptor_index]
                .checked_add(1)
                .ok_or(LodDebugMetadataError::CountOverflow)?;
        }

        let mut node_indices_by_descriptor = Vec::new();
        node_indices_by_descriptor
            .try_reserve_exact(manifest.pages.len())
            .map_err(|_| LodDebugMetadataError::AllocationFailed(manifest.pages.len()))?;
        for node_count in node_counts {
            let mut node_indices = Vec::new();
            node_indices
                .try_reserve_exact(node_count)
                .map_err(|_| LodDebugMetadataError::AllocationFailed(node_count))?;
            node_indices_by_descriptor.push(node_indices);
        }
        for (node_index, node) in manifest.nodes.iter().enumerate() {
            let descriptor_index = descriptor_by_page[&node.representation.page];
            node_indices_by_descriptor[descriptor_index].push(node_index);
        }

        Ok(Self {
            support_sigma: manifest.build.settings.support_sigma,
            descriptors,
            nodes,
            descriptor_by_page,
            node_indices_by_descriptor,
        })
    }

    /// Manifest descriptor index for `page`.
    #[inline]
    pub fn descriptor_index(&self, page: LodPageId) -> Option<usize> {
        self.descriptor_by_page.get(&page).copied()
    }

    /// Validated manifest descriptor for `page`.
    #[inline]
    pub fn descriptor(&self, page: LodPageId) -> Option<&LodPageDescriptor> {
        self.descriptor_index(page)
            .map(|descriptor_index| &self.descriptors[descriptor_index])
    }

    /// Manifest node indices whose representation is stored in `page`.
    #[inline]
    pub fn node_indices(&self, page: LodPageId) -> Option<&[usize]> {
        self.descriptor_index(page)
            .map(|descriptor_index| self.node_indices_by_descriptor[descriptor_index].as_slice())
    }

    /// Stable logical node identities whose ranges are stored in `page`.
    pub fn node_ids(
        &self,
        page: LodPageId,
    ) -> Option<impl ExactSizeIterator<Item = LodNodeId> + '_> {
        let indices = self.node_indices(page)?;
        Some(indices.iter().map(|&node_index| self.nodes[node_index].id))
    }

    /// Build debug records in decoded page-local order without revalidating or
    /// linearly scanning the manifest.
    pub fn records_for_page(
        &self,
        page: &PlanarGaussian3dPage,
        residency: LodDebugResidency,
    ) -> Result<Vec<LodDebugRecord>, LodDebugMetadataError> {
        self.records_for_page_with_node_residency(page, |_| residency)
    }

    /// Build debug records while resolving runtime provenance independently
    /// for every logical node represented by `page`.
    ///
    /// Physical pages may contain several sibling node ranges. Callers that
    /// track ancestor fallback per selected node must use this variant rather
    /// than assigning one provenance value to the whole page.
    pub fn records_for_page_with_node_residency(
        &self,
        page: &PlanarGaussian3dPage,
        mut residency_for_node: impl FnMut(LodNodeId) -> LodDebugResidency,
    ) -> Result<Vec<LodDebugRecord>, LodDebugMetadataError> {
        let descriptor_index = self
            .descriptor_index(page.id)
            .ok_or(LodDebugMetadataError::UnknownPage(page.id))?;
        let descriptor = &self.descriptors[descriptor_index];
        page.validate(descriptor)
            .map_err(|_| LodDebugMetadataError::InvalidPage(page.id))?;

        let mut records = Vec::new();
        records
            .try_reserve_exact(page.gaussians.len())
            .map_err(|_| LodDebugMetadataError::AllocationFailed(page.gaussians.len()))?;
        records.resize(page.gaussians.len(), None);

        for &node_index in &self.node_indices_by_descriptor[descriptor_index] {
            let node = &self.nodes[node_index];
            let residency = residency_for_node(node.id);
            let start = node.representation.offset as usize;
            let end = start
                .checked_add(node.representation.count as usize)
                .ok_or(LodDebugMetadataError::CountOverflow)?;
            let gaussians = page
                .gaussians
                .get(start..end)
                .ok_or(LodDebugMetadataError::InvalidNodeRange(node.id))?;
            let record_slots = records
                .get_mut(start..end)
                .ok_or(LodDebugMetadataError::InvalidNodeRange(node.id))?;
            for (slot, gaussian) in record_slots.iter_mut().zip(gaussians) {
                if slot.is_some() {
                    return Err(LodDebugMetadataError::OverlappingNodeRange(node.id));
                }
                *slot = Some(LodDebugRecord::for_node_fields(
                    node.id,
                    node.depth,
                    node.bounds,
                    node.representation,
                    node.geometric_error,
                    node.quality_threshold,
                    node.high_fidelity_certificate,
                    node.is_original_representation,
                    gaussian,
                    self.support_sigma,
                    residency,
                )?);
            }
        }

        records
            .into_iter()
            .map(|record| record.ok_or(LodDebugMetadataError::UncoveredPage(page.id)))
            .collect()
    }
}

/// Generation-safe mutable mirror of a bounded physical Gaussian atlas.
///
/// The component owns exactly `slot_count * records_per_slot` records. Cloning
/// [`Self::metadata`] is cheap; the first subsequent mutation uses
/// copy-on-write so readers always observe an immutable, frame-consistent
/// snapshot.
#[derive(Clone, Component, Debug)]
pub struct LodDebugAnnotationAtlas {
    slot_count: u32,
    records_per_slot: u32,
    slots: Vec<LodDebugAnnotationSlot>,
    metadata: LodDebugMetadata,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LodDebugAnnotationSlot {
    generation: u32,
    page: Option<LodPageId>,
}

impl LodDebugAnnotationAtlas {
    pub fn new(
        slot_count: u32,
        records_per_slot: u32,
    ) -> Result<Self, LodDebugAnnotationAtlasError> {
        if slot_count == 0 {
            return Err(LodDebugAnnotationAtlasError::ZeroSlotCount);
        }
        if records_per_slot == 0 {
            return Err(LodDebugAnnotationAtlasError::ZeroSlotStride);
        }
        let physical_count = slot_count
            .checked_mul(records_per_slot)
            .ok_or(LodDebugAnnotationAtlasError::CapacityOverflow)?;
        let physical_count = usize::try_from(physical_count)
            .map_err(|_| LodDebugAnnotationAtlasError::CapacityOverflow)?;
        let slot_len = usize::try_from(slot_count)
            .map_err(|_| LodDebugAnnotationAtlasError::CapacityOverflow)?;

        let mut records = Vec::new();
        records
            .try_reserve_exact(physical_count)
            .map_err(|_| LodDebugAnnotationAtlasError::AllocationFailed(physical_count))?;
        records.resize(physical_count, LodDebugRecord::default());
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(slot_len)
            .map_err(|_| LodDebugAnnotationAtlasError::AllocationFailed(slot_len))?;
        slots.resize(slot_len, LodDebugAnnotationSlot::default());

        Ok(Self {
            slot_count,
            records_per_slot,
            slots,
            metadata: LodDebugMetadata::new(records),
        })
    }

    #[inline]
    pub const fn slot_count(&self) -> u32 {
        self.slot_count
    }

    #[inline]
    pub const fn records_per_slot(&self) -> u32 {
        self.records_per_slot
    }

    #[inline]
    pub fn physical_record_count(&self) -> usize {
        self.metadata.len()
    }

    /// Immutable snapshot suitable for attaching to the same cloud entity.
    #[inline]
    pub fn metadata(&self) -> LodDebugMetadata {
        self.metadata.clone()
    }

    #[inline]
    pub fn page(&self, slot: AtlasSlot) -> Option<LodPageId> {
        self.slots.get(slot.index as usize).and_then(|state| {
            (state.generation == slot.generation)
                .then_some(state.page)
                .flatten()
        })
    }

    /// Clears a slot only if its generation is current. The generation is
    /// retained after clearing so a late write from an older generation cannot
    /// resurrect stale metadata.
    pub fn clear_slot(
        &mut self,
        slot: AtlasSlot,
    ) -> Result<LodDebugAtlasUpdate, LodDebugAnnotationAtlasError> {
        let range = self.slot_range(slot.index)?;
        let state = &mut self.slots[slot.index as usize];
        if slot.generation == 0 {
            return Err(LodDebugAnnotationAtlasError::InvalidGeneration);
        }
        if state.generation != slot.generation {
            return Err(LodDebugAnnotationAtlasError::StaleGeneration {
                slot,
                current: state.generation,
            });
        }
        state.page = None;
        Arc::make_mut(&mut self.metadata.records)[range.clone()].fill(LodDebugRecord::default());
        Ok(LodDebugAtlasUpdate {
            range,
            slot,
            page: None,
        })
    }

    /// Builds and writes one page at the exact physical slot used by the
    /// Gaussian atlas. A newer wrapping generation replaces an old occupant;
    /// equal generations may be refreshed; older generations are rejected.
    /// The unused tail of the fixed-stride slot is always zeroed.
    pub fn write_page(
        &mut self,
        manifest: &GaussianLodManifest,
        page: &PlanarGaussian3dPage,
        slot: AtlasSlot,
        residency: LodDebugResidency,
    ) -> Result<LodDebugAtlasUpdate, LodDebugAnnotationAtlasError> {
        let index =
            LodDebugManifestIndex::new(manifest).map_err(LodDebugAnnotationAtlasError::Metadata)?;
        self.write_page_indexed(&index, page, slot, residency)
    }

    /// Indexed variant of [`Self::write_page`] for streaming callers that
    /// write many pages from the same validated manifest.
    pub fn write_page_indexed(
        &mut self,
        index: &LodDebugManifestIndex,
        page: &PlanarGaussian3dPage,
        slot: AtlasSlot,
        residency: LodDebugResidency,
    ) -> Result<LodDebugAtlasUpdate, LodDebugAnnotationAtlasError> {
        self.write_page_indexed_with_node_residency(index, page, slot, |_| residency)
    }

    /// Indexed page write with independently resolved provenance for each
    /// logical node range stored in the physical page.
    pub fn write_page_indexed_with_node_residency(
        &mut self,
        index: &LodDebugManifestIndex,
        page: &PlanarGaussian3dPage,
        slot: AtlasSlot,
        residency_for_node: impl FnMut(LodNodeId) -> LodDebugResidency,
    ) -> Result<LodDebugAtlasUpdate, LodDebugAnnotationAtlasError> {
        let range = self.slot_range(slot.index)?;
        if page.gaussians.len() > self.records_per_slot as usize {
            return Err(LodDebugAnnotationAtlasError::PageExceedsSlotStride {
                page: page.id,
                count: page.gaussians.len(),
                stride: self.records_per_slot,
            });
        }
        if slot.generation == 0 {
            return Err(LodDebugAnnotationAtlasError::InvalidGeneration);
        }
        let state = self.slots[slot.index as usize];
        if state.generation != 0
            && state.generation != slot.generation
            && !generation_is_newer(slot.generation, state.generation)
        {
            return Err(LodDebugAnnotationAtlasError::StaleGeneration {
                slot,
                current: state.generation,
            });
        }

        let page_records = index
            .records_for_page_with_node_residency(page, residency_for_node)
            .map_err(LodDebugAnnotationAtlasError::Metadata)?;
        let records = Arc::make_mut(&mut self.metadata.records);
        records[range.clone()].fill(LodDebugRecord::default());
        let page_end = range
            .start
            .checked_add(page_records.len())
            .ok_or(LodDebugAnnotationAtlasError::CapacityOverflow)?;
        records[range.start..page_end].copy_from_slice(&page_records);
        self.slots[slot.index as usize] = LodDebugAnnotationSlot {
            generation: slot.generation,
            page: Some(page.id),
        };

        Ok(LodDebugAtlasUpdate {
            range,
            slot,
            page: Some(page.id),
        })
    }

    fn slot_range(&self, index: u32) -> Result<Range<usize>, LodDebugAnnotationAtlasError> {
        if index >= self.slot_count {
            return Err(LodDebugAnnotationAtlasError::SlotOutOfRange {
                index,
                slot_count: self.slot_count,
            });
        }
        let start = index
            .checked_mul(self.records_per_slot)
            .ok_or(LodDebugAnnotationAtlasError::CapacityOverflow)? as usize;
        let end = start
            .checked_add(self.records_per_slot as usize)
            .ok_or(LodDebugAnnotationAtlasError::CapacityOverflow)?;
        Ok(start..end)
    }
}

/// Exact physical range dirtied by an annotation-atlas operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LodDebugAtlasUpdate {
    pub range: Range<usize>,
    pub slot: AtlasSlot,
    pub page: Option<LodPageId>,
}

#[derive(Debug)]
pub enum LodDebugAnnotationAtlasError {
    ZeroSlotCount,
    ZeroSlotStride,
    CapacityOverflow,
    AllocationFailed(usize),
    SlotOutOfRange {
        index: u32,
        slot_count: u32,
    },
    InvalidGeneration,
    StaleGeneration {
        slot: AtlasSlot,
        current: u32,
    },
    PageExceedsSlotStride {
        page: LodPageId,
        count: usize,
        stride: u32,
    },
    Metadata(LodDebugMetadataError),
}

impl fmt::Display for LodDebugAnnotationAtlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSlotCount => write!(f, "LoD debug annotation atlas needs at least one slot"),
            Self::ZeroSlotStride => write!(f, "LoD debug annotation atlas slot stride is zero"),
            Self::CapacityOverflow => write!(f, "LoD debug annotation atlas capacity overflow"),
            Self::AllocationFailed(count) => {
                write!(f, "failed to allocate {count} LoD debug atlas records")
            }
            Self::SlotOutOfRange { index, slot_count } => write!(
                f,
                "LoD debug annotation slot {index} is outside atlas slot count {slot_count}"
            ),
            Self::InvalidGeneration => write!(f, "LoD debug annotation generation zero is invalid"),
            Self::StaleGeneration { slot, current } => write!(
                f,
                "LoD debug annotation slot {} generation {} is stale; current generation is {current}",
                slot.index, slot.generation
            ),
            Self::PageExceedsSlotStride {
                page,
                count,
                stride,
            } => write!(
                f,
                "LoD debug page {} has {count} records, exceeding slot stride {stride}",
                page.0
            ),
            Self::Metadata(error) => error.fmt(f),
        }
    }
}

impl Error for LodDebugAnnotationAtlasError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Metadata(error) => Some(error),
            _ => None,
        }
    }
}

#[inline]
const fn generation_is_newer(candidate: u32, current: u32) -> bool {
    let distance = candidate.wrapping_sub(current);
    distance != 0 && distance < (1_u32 << 31)
}

impl LodDebugMetadata {
    pub fn new(records: impl Into<Arc<[LodDebugRecord]>>) -> Self {
        Self {
            records: records.into(),
        }
    }

    #[inline]
    pub fn records(&self) -> &[LodDebugRecord] {
        &self.records
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Flatten all pages in `lod.pages` order. The returned page layouts make
    /// the alignment explicit so callers can flatten Gaussian payloads in the
    /// same order without relying on page identifiers being dense.
    pub fn from_lod_page_order(
        lod: &PlanarGaussian3dLod,
        residency: LodDebugResidency,
    ) -> Result<(Self, Vec<LodDebugPageLayout>), LodDebugMetadataError> {
        lod.validate().map_err(LodDebugMetadataError::InvalidLod)?;
        let index = LodDebugManifestIndex::new(&lod.manifest)?;

        let total = lod.pages.iter().try_fold(0_usize, |total, page| {
            total
                .checked_add(page.gaussians.len())
                .ok_or(LodDebugMetadataError::CountOverflow)
        })?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(total)
            .map_err(|_| LodDebugMetadataError::AllocationFailed(total))?;
        let mut layouts = Vec::new();
        layouts
            .try_reserve_exact(lod.pages.len())
            .map_err(|_| LodDebugMetadataError::AllocationFailed(lod.pages.len()))?;

        for page in &lod.pages {
            let offset = records.len();
            let page_records = index.records_for_page(page, residency)?;
            records.extend_from_slice(&page_records);
            layouts.push(LodDebugPageLayout {
                page: page.id,
                offset,
                count: page_records.len(),
            });
        }

        Ok((Self::new(records), layouts))
    }

    /// Build records for one decoded page in page-local Gaussian order.
    pub fn records_for_page(
        manifest: &GaussianLodManifest,
        page: &PlanarGaussian3dPage,
        residency: LodDebugResidency,
    ) -> Result<Vec<LodDebugRecord>, LodDebugMetadataError> {
        LodDebugManifestIndex::new(manifest)?.records_for_page(page, residency)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodDebugPageLayout {
    pub page: LodPageId,
    pub offset: usize,
    pub count: usize,
}

#[derive(Debug)]
pub enum LodDebugMetadataError {
    InvalidLod(LodValidationError),
    UnknownPage(LodPageId),
    InvalidPage(LodPageId),
    InvalidGaussian(LodNodeId),
    InvalidNodeRange(LodNodeId),
    OverlappingNodeRange(LodNodeId),
    UncoveredPage(LodPageId),
    CountOverflow,
    AllocationFailed(usize),
}

impl fmt::Display for LodDebugMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLod(error) => write!(f, "invalid LoD input: {error}"),
            Self::UnknownPage(page) => {
                write!(f, "LoD debug page {} is not in the manifest", page.0)
            }
            Self::InvalidPage(page) => {
                write!(f, "LoD debug page {} does not match its descriptor", page.0)
            }
            Self::InvalidGaussian(node) => {
                write!(f, "LoD debug node {} contains an invalid Gaussian", node.0)
            }
            Self::InvalidNodeRange(node) => {
                write!(f, "LoD debug node {} has an invalid page range", node.0)
            }
            Self::OverlappingNodeRange(node) => {
                write!(f, "LoD debug node {} overlaps another page range", node.0)
            }
            Self::UncoveredPage(page) => write!(
                f,
                "LoD debug page {} is not completely covered by nodes",
                page.0
            ),
            Self::CountOverflow => write!(f, "LoD debug metadata count overflow"),
            Self::AllocationFailed(count) => {
                write!(f, "failed to allocate {count} LoD debug records")
            }
        }
    }
}

impl Error for LodDebugMetadataError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidLod(error) => Some(error),
            _ => None,
        }
    }
}

/// Distance from a Gaussian's support bound to the nearest owning-node face,
/// normalized independently on each non-degenerate axis.
pub fn normalized_support_boundary_distance(node: LodBounds, support: LodBounds) -> f32 {
    let mut distance = 0.5_f32;
    let mut saw_non_degenerate_axis = false;
    for axis in 0..3 {
        let extent = node.max[axis] - node.min[axis];
        if !extent.is_finite() || extent <= f32::EPSILON {
            continue;
        }
        saw_non_degenerate_axis = true;
        let lower = (support.min[axis] - node.min[axis]) / extent;
        let upper = (node.max[axis] - support.max[axis]) / extent;
        distance = distance.min(lower).min(upper);
    }
    if saw_non_degenerate_axis {
        distance.clamp(0.0, 0.5)
    } else {
        0.0
    }
}

/// Stable, inexpensive avalanche hash for categorical page coloring.
pub const fn stable_page_color_key(page: LodPageId) -> u32 {
    let folded = (page.0 as u32) ^ ((page.0 >> 32) as u32);
    let mut value = folded ^ 0x9e37_79b9;
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

/// CPU oracle for the shader's deterministic categorical page palette.
pub fn lod_debug_page_color(page_color_key: u32) -> [f32; 3] {
    let hash = stable_page_color_key(LodPageId(u64::from(page_color_key)));
    let hue = (hash & 0x00ff_ffff) as f32 / 16_777_215.0;
    hsv_to_rgb(hue, 0.72, 0.95)
}

/// CPU oracle for the shader's fixed hierarchy-level palette.
///
/// The documented order is purple, cyan, green, yellow, orange, red, blue,
/// and pink for levels 0 through 7, then repeats for deeper trees. Keeping
/// level colors categorical and seed-independent makes adjacent hierarchy
/// levels immediately distinguishable and screenshots comparable over time.
pub fn lod_debug_level_color(hierarchy_level: u32) -> [f32; 3] {
    const COLORS: [[f32; 3]; 8] = [
        [0.72, 0.32, 0.95],
        [0.05, 0.78, 0.95],
        [0.15, 0.85, 0.35],
        [0.95, 0.85, 0.10],
        [1.00, 0.48, 0.08],
        [0.95, 0.12, 0.16],
        [0.18, 0.35, 1.00],
        [1.00, 0.22, 0.65],
    ];
    COLORS[hierarchy_level as usize % COLORS.len()]
}

/// CPU oracle for the current-view selection-pressure palette.
///
/// For a balanced target, `pressure` is the lesser of the guarded structural
/// detail ratio and the projected-error ratio. The exact target (1.0) is green;
/// values above it progress through yellow and orange to red, while comfortably
/// safe values are blue/cyan.
pub fn lod_debug_selection_pressure_color(pressure: f32) -> [f32; 3] {
    const BLUE: [f32; 3] = [0.05, 0.20, 0.90];
    const CYAN: [f32; 3] = [0.00, 0.82, 1.00];
    const GREEN: [f32; 3] = [0.10, 0.90, 0.20];
    const YELLOW: [f32; 3] = [1.00, 0.90, 0.00];
    const ORANGE: [f32; 3] = [1.00, 0.45, 0.00];
    const RED: [f32; 3] = [0.95, 0.05, 0.05];

    let pressure = if pressure.is_nan() {
        0.0
    } else if pressure.is_finite() {
        pressure.max(0.0)
    } else if pressure.is_sign_positive() {
        f32::INFINITY
    } else {
        0.0
    };
    if pressure <= 0.5 {
        mix_rgb(BLUE, CYAN, pressure / 0.5)
    } else if pressure <= 1.0 {
        mix_rgb(CYAN, GREEN, (pressure - 0.5) / 0.5)
    } else if pressure <= 1.5 {
        mix_rgb(GREEN, YELLOW, (pressure - 1.0) / 0.5)
    } else if pressure <= 2.0 {
        mix_rgb(YELLOW, ORANGE, (pressure - 1.5) / 0.5)
    } else if pressure <= 4.0 {
        mix_rgb(ORANGE, RED, (pressure - 2.0) / 2.0)
    } else {
        RED
    }
}

#[inline]
fn mix_rgb(from: [f32; 3], to: [f32; 3], amount: f32) -> [f32; 3] {
    let amount = amount.clamp(0.0, 1.0);
    if amount <= 0.0 {
        return from;
    }
    if amount >= 1.0 {
        return to;
    }
    std::array::from_fn(|channel| from[channel] + (to[channel] - from[channel]) * amount)
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> [f32; 3] {
    let h = (hue.fract() * 6.0).clamp(0.0, 6.0);
    let sector = h.floor() as u32;
    let fraction = h - sector as f32;
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * fraction);
    let t = value * (1.0 - saturation * (1.0 - fraction));
    match sector % 6 {
        0 => [value, t, p],
        1 => [q, value, p],
        2 => [p, value, t],
        3 => [p, q, value],
        4 => [t, p, value],
        _ => [value, p, q],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gaussian::formats::planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
        testing::LodTestScene,
    };

    #[test]
    fn defaults_are_disabled_and_shader_codes_are_stable() {
        let settings = LodDebugSettings::default();
        assert!(!settings.requires_metadata());
        assert_eq!(LodDebugPreset::Off.shader_code(), 0);
        assert_eq!(LodDebugPreset::Level.shader_code(), 1);
        assert_eq!(LodDebugPreset::Page.shader_code(), 2);
        assert_eq!(LodDebugPreset::Residency.shader_code(), 3);
        assert_eq!(LodDebugPreset::Boundaries.shader_code(), 4);
        assert_eq!(LodDebugPreset::SelectionPressure.shader_code(), 5);
    }

    #[test]
    fn every_promoted_preset_is_complete_and_requires_metadata_except_off() {
        for preset in [
            LodDebugPreset::Off,
            LodDebugPreset::Level,
            LodDebugPreset::Page,
            LodDebugPreset::Residency,
            LodDebugPreset::Boundaries,
            LodDebugPreset::SelectionPressure,
        ] {
            let settings = LodDebugSettings::from_preset(preset);
            assert_eq!(settings.preset, preset);
            assert_eq!(settings.requires_metadata(), preset != LodDebugPreset::Off);
        }
    }

    #[test]
    fn settings_serialize_with_clouds_and_round_trip() {
        let debug = LodDebugSettings::from_preset(LodDebugPreset::SelectionPressure);
        let cloud = crate::CloudSettings {
            lod_debug: debug,
            ..default()
        };

        let json = serde_json::to_string(&cloud).unwrap();
        assert!(json.contains("lod_debug"));
        let decoded: crate::CloudSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.lod_debug, debug);
    }

    #[test]
    fn gpu_record_layout_is_compact_and_decodes_boundary_distance() {
        assert_eq!(std::mem::size_of::<LodDebugRecord>(), 40);
        assert_eq!(std::mem::align_of::<LodDebugRecord>(), 4);
        assert_eq!(std::mem::offset_of!(LodDebugRecord, page_color_key), 0);
        assert_eq!(std::mem::offset_of!(LodDebugRecord, residency), 8);
        assert_eq!(
            std::mem::offset_of!(LodDebugRecord, boundary_distance_bits),
            12
        );
        assert_eq!(std::mem::offset_of!(LodDebugRecord, geometric_error), 16);
        assert_eq!(std::mem::offset_of!(LodDebugRecord, quality_threshold), 20);
        assert_eq!(std::mem::offset_of!(LodDebugRecord, node_center), 24);
        assert_eq!(std::mem::offset_of!(LodDebugRecord, node_radius), 36);

        let record = LodDebugRecord {
            boundary_distance_bits: 0.125_f32.to_bits(),
            ..default()
        };
        assert_eq!(record.boundary_distance(), 0.125);
        assert!(!record.is_original_representation());

        let original = LodDebugRecord {
            boundary_distance_bits: 0.125_f32.to_bits() | LOD_DEBUG_ORIGINAL_REPRESENTATION_BIT,
            ..default()
        };
        assert_eq!(original.boundary_distance(), 0.125);
        assert!(original.is_original_representation());

        let certificate = 0.582_990_4;
        let packed = LodDebugRecord {
            residency: pack_lod_debug_residency_certificate(
                LodDebugResidency::AncestorFallback,
                certificate,
            ),
            ..default()
        };
        assert_eq!(
            packed.residency_code(),
            LodDebugResidency::AncestorFallback as u32
        );
        assert!(
            (packed.high_fidelity_certificate() - certificate).abs()
                <= 0.5 / LOD_DEBUG_CERTIFICATE_MAX as f32 + f32::EPSILON
        );
        let zero = LodDebugRecord {
            residency: pack_lod_debug_residency_certificate(LodDebugResidency::Resident, f32::NAN),
            ..default()
        };
        assert_eq!(zero.residency_code(), LodDebugResidency::Resident as u32);
        assert_eq!(zero.high_fidelity_certificate(), 0.0);

        let quantum = 1.0 / LOD_DEBUG_CERTIFICATE_MAX as f32;
        let legacy_threshold = LodDebugRecord {
            residency: pack_lod_debug_residency_certificate(LodDebugResidency::Resident, quantum),
            ..default()
        };
        assert_eq!(legacy_threshold.high_fidelity_certificate(), quantum);
        let smallest_usable = LodDebugRecord {
            residency: pack_lod_debug_residency_certificate(
                LodDebugResidency::Resident,
                quantum * 1.25,
            ),
            ..default()
        };
        assert_eq!(smallest_usable.high_fidelity_certificate(), quantum * 2.0);
    }

    #[test]
    fn support_aware_boundary_distance_detects_face_overlap() {
        let node = LodBounds::new([-1.0; 3], [1.0; 3]).unwrap();
        let centered = LodBounds::new([-0.25; 3], [0.25; 3]).unwrap();
        let touching = LodBounds::new([-1.0, -0.1, -0.1], [-0.8, 0.1, 0.1]).unwrap();
        assert!((normalized_support_boundary_distance(node, centered) - 0.375).abs() < 1e-6);
        assert_eq!(normalized_support_boundary_distance(node, touching), 0.0);
    }

    #[test]
    fn manifest_pages_generate_complete_deterministic_metadata() {
        let cloud: crate::PlanarGaussian3d = LodTestScene::nested_octants(1)
            .gaussians
            .iter()
            .map(|gaussian| gaussian.gaussian)
            .collect();
        let lod = build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                leaf_capacity: 2,
                ..default()
            },
        )
        .unwrap();

        let (first, layouts) =
            LodDebugMetadata::from_lod_page_order(&lod, LodDebugResidency::AncestorFallback)
                .unwrap();
        let (second, second_layouts) =
            LodDebugMetadata::from_lod_page_order(&lod, LodDebugResidency::AncestorFallback)
                .unwrap();

        assert_eq!(layouts, second_layouts);
        assert_eq!(first.records(), second.records());
        assert_eq!(layouts.len(), lod.pages.len());
        assert_eq!(
            first.len(),
            lod.pages
                .iter()
                .map(|page| page.gaussians.len())
                .sum::<usize>()
        );
        assert!(first.records().iter().all(|record| {
            record.residency_code() == LodDebugResidency::AncestorFallback as u32
                && record.boundary_distance().is_finite()
                && (0.0..=0.5).contains(&record.boundary_distance())
        }));
        for layout in &layouts {
            for node in lod
                .manifest
                .nodes
                .iter()
                .filter(|node| node.representation.page == layout.page)
            {
                let start = layout.offset + node.representation.offset as usize;
                let end = start + node.representation.count as usize;
                assert!(first.records()[start..end].iter().all(|record| {
                    record.node_center == node.bounds.center()
                        && record.node_radius == node.bounds.radius()
                        && record.quality_threshold
                            == lod_debug_quality_threshold(node.quality.min, node.quality.max)
                        && (record.high_fidelity_certificate() - node.high_fidelity_certificate)
                            .abs()
                            <= 0.5 / LOD_DEBUG_CERTIFICATE_MAX as f32 + f32::EPSILON
                        && record.is_original_representation() == node.is_leaf()
                }));
            }
        }
    }

    #[test]
    fn manifest_index_matches_convenience_records_and_atlas_writes() {
        let cloud: crate::PlanarGaussian3d = LodTestScene::nested_octants(2)
            .gaussians
            .iter()
            .map(|gaussian| gaussian.gaussian)
            .collect();
        let lod = build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                leaf_capacity: 2,
                ..default()
            },
        )
        .unwrap();
        let index = {
            let manifest_snapshot = lod.manifest.clone();
            LodDebugManifestIndex::new(&manifest_snapshot).unwrap()
        };

        assert_eq!(index.descriptor(LodPageId::INVALID), None);
        assert_eq!(index.node_indices(LodPageId::INVALID), None);
        for page in &lod.pages {
            let descriptor_index = index.descriptor_index(page.id).unwrap();
            assert_eq!(
                index.descriptor(page.id),
                Some(&lod.manifest.pages[descriptor_index])
            );
            let expected_node_indices = lod
                .manifest
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(node_index, node)| {
                    (node.representation.page == page.id).then_some(node_index)
                })
                .collect::<Vec<_>>();
            assert_eq!(
                index.node_indices(page.id).unwrap(),
                expected_node_indices.as_slice()
            );

            let convenience = LodDebugMetadata::records_for_page(
                &lod.manifest,
                page,
                LodDebugResidency::AncestorFallback,
            )
            .unwrap();
            let indexed = index
                .records_for_page(page, LodDebugResidency::AncestorFallback)
                .unwrap();
            assert_eq!(indexed, convenience);
        }

        let slot_count = u32::try_from(lod.pages.len()).unwrap();
        let records_per_slot = lod
            .pages
            .iter()
            .map(|page| u32::try_from(page.gaussians.len()).unwrap())
            .max()
            .unwrap();
        let mut convenience_atlas =
            LodDebugAnnotationAtlas::new(slot_count, records_per_slot).unwrap();
        let mut indexed_atlas = LodDebugAnnotationAtlas::new(slot_count, records_per_slot).unwrap();
        for (slot_index, page) in lod.pages.iter().enumerate() {
            let slot = AtlasSlot {
                index: u32::try_from(slot_index).unwrap(),
                generation: 1,
            };
            let convenience_update = convenience_atlas
                .write_page(&lod.manifest, page, slot, LodDebugResidency::Resident)
                .unwrap();
            let indexed_update = indexed_atlas
                .write_page_indexed(&index, page, slot, LodDebugResidency::Resident)
                .unwrap();
            assert_eq!(indexed_update, convenience_update);
        }
        assert_eq!(
            indexed_atlas.metadata().records(),
            convenience_atlas.metadata().records()
        );
    }

    #[test]
    fn shared_page_residency_is_node_granular_and_swaps_at_equal_generation() {
        let cloud = LodTestScene::nested_octants(2).cloud();
        let lod = build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                leaf_capacity: 2,
                ..default()
            },
        )
        .unwrap();
        let index = LodDebugManifestIndex::new(&lod.manifest).unwrap();
        let page = lod
            .pages
            .iter()
            .find(|page| {
                index
                    .node_indices(page.id)
                    .is_some_and(|nodes| nodes.len() >= 2)
            })
            .expect("fixture must contain sibling node ranges in one physical page");
        let node_ids = index.node_ids(page.id).unwrap().take(2).collect::<Vec<_>>();
        let first_node = lod
            .manifest
            .nodes
            .iter()
            .find(|node| node.id == node_ids[0])
            .unwrap();
        let second_node = lod
            .manifest
            .nodes
            .iter()
            .find(|node| node.id == node_ids[1])
            .unwrap();
        let slot = AtlasSlot {
            index: 0,
            generation: 1,
        };
        let mut atlas = LodDebugAnnotationAtlas::new(1, page.gaussians.len() as u32).unwrap();

        atlas
            .write_page_indexed_with_node_residency(&index, page, slot, |node| {
                if node == first_node.id {
                    LodDebugResidency::AncestorFallback
                } else {
                    LodDebugResidency::Resident
                }
            })
            .unwrap();
        let first_snapshot = atlas.metadata();
        let first_range = first_node.representation.offset as usize
            ..first_node.representation.end().unwrap() as usize;
        let second_range = second_node.representation.offset as usize
            ..second_node.representation.end().unwrap() as usize;
        assert!(
            first_snapshot.records()[first_range.clone()]
                .iter()
                .all(|record| {
                    record.residency_code() == LodDebugResidency::AncestorFallback as u32
                })
        );
        assert!(
            first_snapshot.records()[second_range.clone()]
                .iter()
                .all(|record| { record.residency_code() == LodDebugResidency::Resident as u32 })
        );

        atlas
            .write_page_indexed_with_node_residency(&index, page, slot, |node| {
                if node == second_node.id {
                    LodDebugResidency::AncestorFallback
                } else {
                    LodDebugResidency::Resident
                }
            })
            .unwrap();
        let second_snapshot = atlas.metadata();
        assert_ne!(first_snapshot.records(), second_snapshot.records());
        assert!(
            second_snapshot.records()[first_range]
                .iter()
                .all(|record| { record.residency_code() == LodDebugResidency::Resident as u32 })
        );
        assert!(
            second_snapshot.records()[second_range]
                .iter()
                .all(|record| {
                    record.residency_code() == LodDebugResidency::AncestorFallback as u32
                })
        );
    }

    #[test]
    fn manifest_index_rejects_invalid_manifest_during_construction() {
        let cloud: crate::PlanarGaussian3d = LodTestScene::nested_octants(1)
            .gaussians
            .iter()
            .map(|gaussian| gaussian.gaussian)
            .collect();
        let lod = build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                leaf_capacity: 2,
                ..default()
            },
        )
        .unwrap();
        let mut invalid_manifest = lod.manifest;
        invalid_manifest.header.node_count = invalid_manifest.header.node_count.saturating_add(1);

        assert!(matches!(
            LodDebugManifestIndex::new(&invalid_manifest),
            Err(LodDebugMetadataError::InvalidLod(_))
        ));
    }

    #[test]
    fn palettes_are_bounded_and_categorical_colors_are_stable() {
        let page = stable_page_color_key(LodPageId(42));
        let first = lod_debug_page_color(page);
        assert_eq!(first, lod_debug_page_color(page));
        assert!(
            first
                .into_iter()
                .all(|channel| (0.0..=1.0).contains(&channel))
        );

        let level_colors = (0..8).map(lod_debug_level_color).collect::<Vec<_>>();
        assert_eq!(level_colors[0], lod_debug_level_color(8));
        assert_eq!(level_colors[1], lod_debug_level_color(9));
        for (index, color) in level_colors.iter().enumerate() {
            assert!(color.iter().all(|channel| (0.0..=1.0).contains(channel)));
            assert!(level_colors[index + 1..].iter().all(|other| other != color));
        }
    }

    #[test]
    fn selection_pressure_palette_has_semantic_anchor_colors() {
        assert_eq!(lod_debug_selection_pressure_color(0.0), [0.05, 0.20, 0.90]);
        assert_eq!(lod_debug_selection_pressure_color(0.5), [0.00, 0.82, 1.00]);
        assert_eq!(lod_debug_selection_pressure_color(1.0), [0.10, 0.90, 0.20]);
        assert_eq!(lod_debug_selection_pressure_color(1.5), [1.00, 0.90, 0.00]);
        assert_eq!(lod_debug_selection_pressure_color(2.0), [1.00, 0.45, 0.00]);
        assert_eq!(lod_debug_selection_pressure_color(4.0), [0.95, 0.05, 0.05]);
        assert_eq!(
            lod_debug_selection_pressure_color(f32::INFINITY),
            [0.95, 0.05, 0.05]
        );
        assert_eq!(
            lod_debug_selection_pressure_color(f32::NAN),
            [0.05, 0.20, 0.90]
        );

        for index in 0..=400 {
            let color = lod_debug_selection_pressure_color(index as f32 / 100.0);
            assert!(
                color
                    .into_iter()
                    .all(|channel| (0.0..=1.0).contains(&channel))
            );
        }
    }

    #[test]
    fn mutable_atlas_is_bounded_generation_safe_and_snapshot_consistent() {
        assert!(matches!(
            LodDebugAnnotationAtlas::new(0, 4),
            Err(LodDebugAnnotationAtlasError::ZeroSlotCount)
        ));
        assert!(matches!(
            LodDebugAnnotationAtlas::new(4, 0),
            Err(LodDebugAnnotationAtlasError::ZeroSlotStride)
        ));
        assert!(matches!(
            LodDebugAnnotationAtlas::new(u32::MAX, 2),
            Err(LodDebugAnnotationAtlasError::CapacityOverflow)
        ));

        let cloud: crate::PlanarGaussian3d = LodTestScene::nested_octants(1)
            .gaussians
            .iter()
            .map(|gaussian| gaussian.gaussian)
            .collect();
        let lod = build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                leaf_capacity: 2,
                ..default()
            },
        )
        .unwrap();
        let page = lod.pages.first().unwrap();
        let stride = page.gaussians.len() as u32 + 3;
        let mut atlas = LodDebugAnnotationAtlas::new(2, stride).unwrap();
        assert_eq!(atlas.physical_record_count(), (stride * 2) as usize);

        assert!(matches!(
            atlas.write_page(
                &lod.manifest,
                page,
                AtlasSlot {
                    index: 0,
                    generation: 0,
                },
                LodDebugResidency::Resident,
            ),
            Err(LodDebugAnnotationAtlasError::InvalidGeneration)
        ));

        let largest_page = lod
            .pages
            .iter()
            .max_by_key(|page| page.gaussians.len())
            .unwrap();
        assert!(largest_page.gaussians.len() > 1);
        let mut undersized =
            LodDebugAnnotationAtlas::new(1, largest_page.gaussians.len().saturating_sub(1) as u32)
                .unwrap();
        assert!(matches!(
            undersized.write_page(
                &lod.manifest,
                largest_page,
                AtlasSlot {
                    index: 0,
                    generation: 1,
                },
                LodDebugResidency::Resident,
            ),
            Err(LodDebugAnnotationAtlasError::PageExceedsSlotStride { .. })
        ));

        let before_write = atlas.metadata();
        let first_slot = AtlasSlot {
            index: 1,
            generation: 1,
        };
        let update = atlas
            .write_page(&lod.manifest, page, first_slot, LodDebugResidency::Resident)
            .unwrap();
        assert_eq!(update.range, stride as usize..(stride * 2) as usize);
        assert_eq!(update.page, Some(page.id));
        assert_eq!(atlas.page(first_slot), Some(page.id));
        assert!(
            before_write
                .records()
                .iter()
                .all(|record| *record == default())
        );
        let after_write = atlas.metadata();
        let page_start = stride as usize;
        assert!(
            after_write.records()[page_start..page_start + page.gaussians.len()]
                .iter()
                .all(|record| record.residency_code() == LodDebugResidency::Resident as u32)
        );
        assert!(
            after_write.records()[page_start + page.gaussians.len()..]
                .iter()
                .all(|record| *record == default())
        );

        let reused_slot = AtlasSlot {
            index: 1,
            generation: 2,
        };
        atlas
            .write_page(
                &lod.manifest,
                page,
                reused_slot,
                LodDebugResidency::AncestorFallback,
            )
            .unwrap();
        assert!(matches!(
            atlas.write_page(&lod.manifest, page, first_slot, LodDebugResidency::Resident,),
            Err(LodDebugAnnotationAtlasError::StaleGeneration { .. })
        ));
        assert!(matches!(
            atlas.clear_slot(first_slot),
            Err(LodDebugAnnotationAtlasError::StaleGeneration { .. })
        ));
        let cleared = atlas.clear_slot(reused_slot).unwrap();
        assert_eq!(cleared.page, None);
        assert_eq!(atlas.page(reused_slot), None);
        assert!(
            atlas.metadata().records()[cleared.range]
                .iter()
                .all(|record| *record == default())
        );
    }

    #[test]
    fn mutable_atlas_rejects_out_of_range_and_zero_generations() {
        let mut atlas = LodDebugAnnotationAtlas::new(1, 1).unwrap();
        assert!(matches!(
            atlas.clear_slot(AtlasSlot {
                index: 1,
                generation: 1,
            }),
            Err(LodDebugAnnotationAtlasError::SlotOutOfRange { .. })
        ));
        assert!(matches!(
            atlas.clear_slot(AtlasSlot {
                index: 0,
                generation: 0,
            }),
            Err(LodDebugAnnotationAtlasError::InvalidGeneration)
        ));
    }
}
