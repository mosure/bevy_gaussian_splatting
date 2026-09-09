use std::sync::Arc;

use bevy::{asset::AssetId, prelude::*, render::extract_component::ExtractComponent};
use bytemuck::{Pod, Zeroable};

use crate::stream::cache::AtlasSlot;
use crate::{
    GaussianLodManifest, PlanarGaussian3d,
    gaussian::formats::planar_3d_chunked::{LodNodeId, LodPageId},
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct Node {
    pub center_radius: [f32; 4],
    pub half_extents: [f32; 4],
    pub error_quality: [f32; 4],
    // Child start, child count, page index, page-local representation offset.
    pub topology: [u32; 4],
    // Representation count, complete child count, original flag, source-domain rank.
    pub counts: [u32; 4],
}

/// Immutable, manifest-order topology compiled once and shared by all views.
pub struct GpuLodHierarchyTree {
    pub(super) nodes: Vec<Node>,
    /// Rank to compiled-node index. A covering antichain is ordered by its
    /// exact canonical source domains, independent of refinement depth.
    pub(super) source_order: Vec<u32>,
    /// Source-ordered internal nodes: compiled index, parent, depth, root alias.
    pub(super) candidates: Vec<[u32; 4]>,
    pub(super) root_count: u32,
    pub(super) support_sigma: f32,
    pub(super) root_records: u32,
    pub(super) max_depth: u32,
    pub(super) page_lengths: Vec<u32>,
    pub(super) page_ids: Vec<LodPageId>,
}

impl GpuLodHierarchyTree {
    pub fn from_manifest(manifest: &GaussianLodManifest) -> Result<Self, String> {
        manifest.validate().map_err(|error| error.to_string())?;
        Self::from_validated_manifest(manifest)
    }

    /// Compiles a manifest already validated by the package runtime. Keeping
    /// validation outside this path avoids unaccounted duplicate index scratch.
    pub(crate) fn from_validated_manifest(manifest: &GaussianLodManifest) -> Result<Self, String> {
        let root_count =
            u32::try_from(manifest.roots.len()).map_err(|_| "too many hierarchy roots")?;
        let node_count =
            u32::try_from(manifest.nodes.len()).map_err(|_| "too many hierarchy nodes")?;
        node_count
            .checked_add(root_count)
            .ok_or("hierarchy index overflow")?;
        u32::try_from(manifest.pages.len()).map_err(|_| "too many hierarchy pages")?;
        if root_count == 0 || manifest.quality.max_depth > 64 {
            return Err("GPU traversal requires roots and at most 64 hierarchy levels".into());
        }
        let mut pages: Vec<(LodPageId, u32)> = manifest
            .pages
            .iter()
            .enumerate()
            .map(|(index, page)| (page.id, index as u32))
            .collect();
        pages.sort_unstable_by_key(|&(id, _)| id);
        let mut roots: Vec<(LodNodeId, u32)> = manifest
            .roots
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index as u32))
            .collect();
        roots.sort_unstable_by_key(|&(id, _)| id);
        let mut root_indices = vec![0u32; manifest.roots.len()];
        // O(N log R), with scratch proportional to roots rather than all nodes.
        for (index, node) in manifest.nodes.iter().enumerate() {
            if let Ok(root) = roots.binary_search_by_key(&node.id, |&(id, _)| id) {
                root_indices[roots[root].1 as usize] = index as u32;
            }
        }
        let mut nodes = Vec::with_capacity(manifest.nodes.len() + root_count as usize);
        for index in root_indices
            .iter()
            .map(|&index| index as usize)
            .chain(0..manifest.nodes.len())
        {
            let node = &manifest.nodes[index];
            let child_end = node.children.end().ok_or("child range overflow")? as usize;
            let children = &manifest.nodes[node.children.start as usize..child_end];
            let child_records = children
                .iter()
                .try_fold(0u32, |sum, child| {
                    sum.checked_add(child.representation.count)
                })
                .ok_or("child representation count overflow")?;
            if !children.is_empty() && child_records < node.representation.count {
                return Err(
                    "GPU cohort admission requires nondecreasing refinement record counts".into(),
                );
            }
            let center = node.bounds.center();
            nodes.push(Node {
                center_radius: [center[0], center[1], center[2], node.bounds.radius()],
                half_extents: [
                    (node.bounds.max[0] - center[0]).max(center[0] - node.bounds.min[0]),
                    (node.bounds.max[1] - center[1]).max(center[1] - node.bounds.min[1]),
                    (node.bounds.max[2] - center[2]).max(center[2] - node.bounds.min[2]),
                    0.0,
                ],
                error_quality: [
                    node.error.geometric,
                    node.quality.min + (node.quality.max - node.quality.min) * 0.5,
                    node.high_fidelity_certificate,
                    0.0,
                ],
                topology: [
                    root_count + node.children.start,
                    node.children.count,
                    pages[pages
                        .binary_search_by_key(&node.representation.page, |&(id, _)| id)
                        .expect("validated node page")]
                    .1,
                    node.representation.offset,
                ],
                counts: [
                    node.representation.count,
                    child_records,
                    u32::from(node.is_leaf()),
                    0,
                ],
            });
        }
        let mut source_order = (0..nodes.len() as u32).collect::<Vec<_>>();
        source_order.sort_unstable_by_key(|&compiled| {
            let authored = if compiled < root_count {
                root_indices[compiled as usize] as usize
            } else {
                (compiled - root_count) as usize
            };
            (manifest.nodes[authored].source.start, compiled)
        });
        for (rank, &compiled) in source_order.iter().enumerate() {
            nodes[compiled as usize].counts[3] = rank as u32;
        }
        // Parent links use existing padding; rendering metrics ignore this word.
        // Root aliases are the actual frontier roots, so redirect their children.
        for parent in (root_count as usize..nodes.len()).chain(0..root_count as usize) {
            let node = nodes[parent];
            for child in node.topology[0]..node.topology[0] + node.topology[1] {
                nodes[child as usize].error_quality[3] = f32::from_bits(parent as u32);
            }
        }
        let candidate_count = nodes
            .iter()
            .enumerate()
            .filter(|&(index, node)| {
                node.topology[1] != 0
                    && (index < root_count as usize
                        || manifest.nodes[index - root_count as usize].depth != 0)
            })
            .count();
        let mut candidates = Vec::with_capacity(candidate_count);
        for &index in &source_order {
            let authored = if index < root_count {
                root_indices[index as usize]
            } else {
                index - root_count
            };
            if nodes[index as usize].topology[1] == 0
                || (index >= root_count && manifest.nodes[authored as usize].depth == 0)
            {
                continue;
            }
            candidates.push([
                index,
                if index < root_count {
                    u32::MAX
                } else {
                    nodes[index as usize].error_quality[3].to_bits()
                },
                u32::from(manifest.nodes[authored as usize].depth),
                if index < root_count {
                    authored + root_count
                } else {
                    u32::MAX
                },
            ]);
        }
        let root_records = nodes[..root_count as usize]
            .iter()
            .try_fold(0u32, |sum, node| sum.checked_add(node.counts[0]))
            .ok_or("root representation count overflow")?;
        Ok(Self {
            nodes,
            source_order,
            candidates,
            root_count,
            support_sigma: manifest.build.settings.support_sigma,
            root_records,
            max_depth: u32::from(manifest.quality.max_depth),
            page_lengths: manifest
                .pages
                .iter()
                .map(|page| page.gaussian_count)
                .collect(),
            page_ids: manifest.pages.iter().map(|page| page.id).collect(),
        })
    }

    /// Owned vector capacities simultaneously live during validated compilation.
    /// Package maps and manifest ownership are reserved separately by the caller.
    pub(crate) fn compilation_bytes(
        nodes: usize,
        roots: usize,
        pages: usize,
    ) -> Result<u64, String> {
        let nodes = u32::try_from(nodes).map_err(|_| "node capacity overflow")?;
        let roots = u32::try_from(roots).map_err(|_| "root capacity overflow")?;
        nodes.checked_add(roots).ok_or("hierarchy index overflow")?;
        let nodes = u64::from(nodes);
        let roots = u64::from(roots);
        let pages = u64::from(u32::try_from(pages).map_err(|_| "page capacity overflow")?);
        nodes
            .checked_add(roots)
            .and_then(|count| {
                count.checked_mul(
                    (size_of::<Node>() + size_of::<u32>() + size_of::<[u32; 4]>()) as u64,
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    roots.checked_mul((size_of::<(LodNodeId, u32)>() + size_of::<u32>()) as u64)?,
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(pages.checked_mul(
                    (size_of::<(LodPageId, u32)>() + size_of::<LodPageId>() + size_of::<u32>())
                        as u64,
                )?)
            })
            .ok_or_else(|| "hierarchy host compilation capacity overflow".into())
    }

    pub fn page_ids(&self) -> &[LodPageId] {
        &self.page_ids
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len() - self.root_count as usize
    }

    /// Irreducible record admission required for the complete authored root cut.
    pub fn root_gaussian_count(&self) -> u32 {
        self.root_records
    }

    pub fn root_node_count(&self) -> u32 {
        self.root_count
    }
}

/// Physical source range for one complete, authenticated resident page.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuLodPagePlacement {
    pub start: u32,
    pub count: u32,
}

/// Snapshot ownership must pin every resident placement until the final GPU
/// fence releases this value. Feedback identifies selected pages and complete
/// incomplete-cohort demand, including resident siblings, so publishers can
/// replace the snapshot and retire unrelated pins under pressure.
#[derive(Clone)]
pub struct GpuLodHierarchySnapshot {
    pub tree: Arc<GpuLodHierarchyTree>,
    pub generation: u64,
    pub source: AssetId<PlanarGaussian3d>,
    pub(super) placements: Vec<GpuLodPagePlacement>,
    pub(super) required_atlas_slots: Vec<AtlasSlot>,
    pub(super) _ownership: Arc<dyn Send + Sync>,
}

#[derive(Component, Clone, ExtractComponent)]
pub struct GpuLodHierarchy(pub Arc<GpuLodHierarchySnapshot>);

impl GpuLodHierarchy {
    /// Reads residency from the authenticated immutable page table without
    /// another ordered page-ID lookup in package ancestry walks.
    pub(crate) fn page_is_resident(&self, page_index: usize) -> bool {
        self.0
            .placements
            .get(page_index)
            .is_some_and(|placement| placement.count != 0)
    }

    /// `generation` must identify this immutable snapshot uniquely across
    /// publisher restarts for the same entity/source; delayed GPU acknowledgements
    /// can outlive a previous publisher instance.
    pub fn new(
        tree: Arc<GpuLodHierarchyTree>,
        generation: u64,
        source: AssetId<PlanarGaussian3d>,
        placements: Vec<Option<GpuLodPagePlacement>>,
        ownership: Arc<dyn Send + Sync>,
    ) -> Result<Self, String> {
        if generation == 0 || placements.len() != tree.page_ids.len() {
            return Err(
                "GPU hierarchy requires a nonzero generation and one placement per manifest page"
                    .into(),
            );
        }
        let mut ranges = Vec::new();
        for (index, placement) in placements.iter().enumerate() {
            if let Some(placement) = placement {
                let end = placement
                    .start
                    .checked_add(placement.count)
                    .ok_or("page placement overflow")?;
                if placement.count != tree.page_lengths[index] || end > 0x1000_0000 {
                    return Err("GPU page placement length or physical index is invalid".into());
                }
                ranges.push((placement.start, end));
            }
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err("resident GPU page placements overlap".into());
        }
        Ok(Self(Arc::new(GpuLodHierarchySnapshot {
            tree,
            generation,
            source,
            placements: placements
                .into_iter()
                .map(Option::unwrap_or_default)
                .collect(),
            required_atlas_slots: Vec::new(),
            _ownership: ownership,
        })))
    }

    /// Package atlas pages occupy distinct, aligned fixed-size slots. Deriving
    /// counts from this authenticated tree avoids a second descriptor lookup
    /// and validates overlap with one occupancy bit per admitted atlas slot.
    /// The caller reserves the dense table, required slots and
    /// `8 * ceil(slot_count / 64)` temporary occupancy bytes before this call.
    pub(crate) fn from_fixed_slots(
        tree: Arc<GpuLodHierarchyTree>,
        generation: u64,
        source: AssetId<PlanarGaussian3d>,
        slot_count: u32,
        gaussians_per_slot: u32,
        resident_pages: impl ExactSizeIterator<Item = (usize, AtlasSlot)>,
        ownership: Arc<dyn Send + Sync>,
    ) -> Result<Self, String> {
        let physical_capacity = u64::from(slot_count) * u64::from(gaussians_per_slot);
        if generation == 0
            || slot_count == 0
            || gaussians_per_slot == 0
            || physical_capacity > 0x1000_0000
            || resident_pages.len() > slot_count as usize
            || resident_pages.len() > tree.page_ids.len()
        {
            return Err("invalid fixed-slot hierarchy capacity or generation".into());
        }
        let mut placements = vec![GpuLodPagePlacement::default(); tree.page_ids.len()];
        let mut occupied = vec![0_u64; (u64::from(slot_count).div_ceil(64)) as usize];
        let mut required_atlas_slots = Vec::with_capacity(resident_pages.len());
        for (page_index, slot) in resident_pages {
            let count = *tree
                .page_lengths
                .get(page_index)
                .ok_or("fixed-slot page index is outside the manifest")?;
            if count == 0
                || count > gaussians_per_slot
                || slot.index >= slot_count
                || slot.generation == 0
            {
                return Err("fixed-slot page length or atlas slot is invalid".into());
            }
            let word = &mut occupied[slot.index as usize / 64];
            let bit = 1_u64 << (slot.index % 64);
            if *word & bit != 0 || placements[page_index].count != 0 {
                return Err("fixed-slot hierarchy repeats a page or overlaps an atlas slot".into());
            }
            *word |= bit;
            // The checked full capacity bounds this multiplication and makes
            // alignment/nonoverlap independent of individual page lengths.
            placements[page_index] = GpuLodPagePlacement {
                start: slot.index * gaussians_per_slot,
                count,
            };
            required_atlas_slots.push(slot);
        }
        Ok(Self(Arc::new(GpuLodHierarchySnapshot {
            tree,
            generation,
            source,
            placements,
            required_atlas_slots,
            _ownership: ownership,
        })))
    }

    /// Requires the render-world upload generation of every leased atlas slot.
    /// Manual immutable GPU sources may leave this list empty.
    pub fn with_required_atlas_slots(mut self, slots: Vec<AtlasSlot>) -> Self {
        Arc::make_mut(&mut self.0).required_atlas_slots = slots;
        self
    }
}

/// Explicit camera opt-in. Queue and record admission always applies to a
/// complete child cohort; hitting a limit keeps the resident parent visible.
#[derive(Component, Clone, Debug, PartialEq, ExtractComponent)]
pub struct GpuLodTraversalSettings {
    /// Complete selected record capacity shared by all visible hierarchy clouds.
    /// Aggregate hierarchy output cap. Fixed flat and CPU-compacted sources
    /// reserve their own part of the renderer's shared projected-record budget.
    pub max_selected_gaussians: u32,
    pub max_frontier_nodes: u32,
    pub max_visited_nodes: u32,
    /// Atomic child-cohort demand capacity in page references. Shared pages may
    /// occupy multiple slots; completed host feedback deduplicates their IDs.
    pub max_page_requests: u32,
    /// View-wide topology, page-table and traversal workspace allocation ceiling.
    /// Shared topology/page buffers count once within the view; the global ledger
    /// also charges overlap with retired generations until their GPU fences finish.
    pub max_gpu_bytes: u64,
}

impl Default for GpuLodTraversalSettings {
    fn default() -> Self {
        Self {
            max_selected_gaussians: 1_048_576,
            max_frontier_nodes: 16_384,
            max_visited_nodes: 262_144,
            max_page_requests: 1_024,
            max_gpu_bytes: 256 * 1024 * 1024,
        }
    }
}

impl GpuLodTraversalSettings {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_selected_gaussians == 0
            || self.max_selected_gaussians > 0x0fff_ffff
            || self.max_frontier_nodes == 0
            || self.max_frontier_nodes > 65_535
            || self.max_visited_nodes == 0
            || self.max_page_requests == 0
            || self.max_page_requests > 1_048_576
            || self.max_gpu_bytes == 0
        {
            return Err(
                "invalid GPU traversal record, frontier, visit, request or byte budget".into(),
            );
        }
        Ok(())
    }
}
