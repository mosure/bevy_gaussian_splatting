//! Bounded, camera-conditioned adjacent hierarchy representation transitions.
//!
//! The immutable correspondence is independent of atlas placement. Render
//! consumers evaluate weights from the current view; no elapsed-time state or
//! delayed renderer acknowledgement participates in a transition weight.

use std::sync::Arc;

use bevy::{prelude::*, render::extract_component::ExtractComponent};
use serde::{Deserialize, Serialize};

use crate::{
    GaussianLodManifest,
    render::traversal::GpuLodHierarchyTree,
    stream::memory::{LodMemoryCategory, LodMemoryLease, LodMemoryLedger},
};

mod gpu;
pub(crate) use gpu::{SpatialMorphBuffers, SpatialMorphPipelines, SpatialMorphState};

pub(crate) fn install(app: &mut App) {
    use bevy::asset::{load_internal_asset, uuid_handle};
    const ANNOTATE: Handle<Shader> = uuid_handle!("4b1db7a8-b4b1-480b-aaba-9e6a749fd3df");
    const PROJECT: Handle<Shader> = uuid_handle!("0cfa7935-13b6-44aa-a5a4-ecb37abf8ec2");
    const METRICS: Handle<Shader> = uuid_handle!("18aef454-5cc8-4b6b-9521-a3c213dad010");
    load_internal_asset!(
        app,
        ANNOTATE,
        "spatial_morph/annotate.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        PROJECT,
        "spatial_morph/project.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        METRICS,
        "spatial_morph/metrics.wgsl",
        Shader::from_wgsl
    );
    app.register_type::<GaussianLodSpatialTransitionSettings>();
}

/// Camera opt-in for spatial transitions of authored GPU hierarchy edges.
/// Limits supplement the traversal and renderer's existing byte/record ceilings.
#[derive(Component, Clone, Debug, PartialEq, ExtractComponent, Reflect, Serialize, Deserialize)]
#[reflect(Component)]
#[serde(default, deny_unknown_fields)]
pub struct GaussianLodSpatialTransitionSettings {
    /// Maximum fractional adjacent edges admitted in this view.
    pub max_transition_nodes: u32,
    /// Maximum child-cardinality records participating in fractional edges.
    pub max_transition_records: u32,
    /// Immutable node descriptors and cumulative run ends, including the header.
    pub max_mapping_bytes: u64,
}

impl Default for GaussianLodSpatialTransitionSettings {
    fn default() -> Self {
        Self {
            max_transition_nodes: 256,
            max_transition_records: 65_536,
            max_mapping_bytes: 32 * 1024 * 1024,
        }
    }
}

impl GaussianLodSpatialTransitionSettings {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_transition_nodes == 0
            || self.max_transition_nodes > 65_535
            || self.max_transition_records == 0
            || self.max_transition_records > 0x0fff_ffff
            || self.max_mapping_bytes < 32
        {
            return Err(
                "spatial transition limits require 1..=65535 edges, positive portable records and at least 32 mapping bytes",
            );
        }
        Ok(())
    }
}

const HEADER_WORDS: usize = 8;
const NODE_WORDS: usize = 4;
const NO_PARENT: u32 = u32::MAX;

fn covariance_envelope_scale(support_sigma: f32) -> Result<f32, String> {
    // Both endpoint means lie in the node box. Their sigma-support radii in
    // any direction are bounded by that box's directional half extent; convex
    // covariance preserves this radius bound. Add the mean and radius bounds.
    let scale = 1.0 + 3.0 / support_sigma;
    if support_sigma > 0.0 && support_sigma.is_finite() && scale.is_finite() {
        Ok(scale)
    } else {
        Err("spatial transitions require finite positive authored support sigma".into())
    }
}

/// Authenticated logical correspondence for exactly one compiled hierarchy.
/// Clone/extraction shares both metadata and its memory reservation.
#[derive(Component, Clone, ExtractComponent)]
pub struct GpuLodSpatialMorph {
    pub(crate) tree: Arc<GpuLodHierarchyTree>,
    pub(crate) words: Arc<Vec<u32>>,
    _lease: LodMemoryLease,
}

impl GpuLodSpatialMorph {
    /// Constant-time owned mapping size before compilation or admission.
    pub fn required_mapping_bytes(manifest: &GaussianLodManifest) -> Result<u64, String> {
        covariance_envelope_scale(manifest.build.settings.support_sigma)?;
        let map = manifest
            .morph_map
            .as_ref()
            .ok_or("package has no authored parent-record correspondence")?;
        let words = manifest
            .nodes
            .len()
            .checked_add(manifest.roots.len())
            .and_then(|nodes| nodes.checked_mul(NODE_WORDS))
            .and_then(|words| words.checked_add(HEADER_WORDS))
            .and_then(|words| words.checked_add(map.child_run_lengths.len()))
            .ok_or("spatial mapping byte overflow")?;
        Ok(u64::from(
            u32::try_from(words).map_err(|_| "spatial mapping exceeds portable shader indexing")?,
        ) * 4)
    }

    /// Builds the hierarchy and its optional transition metadata together, so
    /// mappings cannot accidentally refer to a different compiled node order.
    pub fn from_manifest(
        manifest: &GaussianLodManifest,
        max_mapping_bytes: u64,
        ledger: &LodMemoryLedger,
    ) -> Result<(Arc<GpuLodHierarchyTree>, Self), String> {
        manifest.validate().map_err(|error| error.to_string())?;
        Self::from_validated_manifest(manifest, max_mapping_bytes, ledger)
    }

    pub(crate) fn from_validated_manifest(
        manifest: &GaussianLodManifest,
        max_mapping_bytes: u64,
        ledger: &LodMemoryLedger,
    ) -> Result<(Arc<GpuLodHierarchyTree>, Self), String> {
        let tree = Arc::new(GpuLodHierarchyTree::from_validated_manifest(manifest)?);
        let map = Self::for_validated_tree(tree.clone(), manifest, max_mapping_bytes, ledger)?;
        Ok((tree, map))
    }

    /// Package-internal lazy attachment; `manifest` must be the immutable
    /// validated manifest from which this exact tree was compiled.
    pub(crate) fn for_validated_tree(
        tree: Arc<GpuLodHierarchyTree>,
        manifest: &GaussianLodManifest,
        max_mapping_bytes: u64,
        ledger: &LodMemoryLedger,
    ) -> Result<Self, String> {
        let map = manifest
            .morph_map
            .as_ref()
            .ok_or("package has no authored parent-record correspondence")?;
        let root_count = manifest.roots.len();
        let node_count = manifest
            .nodes
            .len()
            .checked_add(root_count)
            .ok_or("spatial node count overflow")?;
        let run_start = node_count
            .checked_mul(NODE_WORDS)
            .and_then(|words| words.checked_add(HEADER_WORDS))
            .ok_or("spatial descriptor byte overflow")?;
        let word_count = run_start
            .checked_add(map.child_run_lengths.len())
            .ok_or("spatial mapping byte overflow")?;
        let word_count_u32 = u32::try_from(word_count)
            .map_err(|_| "spatial mapping exceeds portable shader indexing")?;
        let bytes = Self::required_mapping_bytes(manifest)?;
        debug_assert_eq!(bytes, u64::from(word_count_u32) * 4);
        if bytes > max_mapping_bytes {
            return Err(format!(
                "spatial mapping requires {bytes} bytes, limit {max_mapping_bytes}"
            ));
        }
        let lease = ledger
            .try_reserve(LodMemoryCategory::MetadataCpu, bytes)
            .map_err(|error| error.to_string())?;
        let mut words = vec![0; word_count];
        words[..5].copy_from_slice(&[
            node_count as u32,
            HEADER_WORDS as u32,
            run_start as u32,
            map.child_run_lengths.len() as u32,
            root_count as u32,
        ]);
        words[5] = covariance_envelope_scale(manifest.build.settings.support_sigma)?.to_bits();
        for node in words[HEADER_WORDS..run_start].chunks_exact_mut(NODE_WORDS) {
            node[0] = NO_PARENT;
        }
        for (index, node) in manifest.nodes.iter().enumerate() {
            let compiled = root_count + index;
            let descriptor = HEADER_WORDS + compiled * NODE_WORDS;
            let runs = map.node_runs[index];
            words[descriptor + 2] = runs.start;
            words[descriptor + 3] = runs.count;
            let mut end = 0u32;
            for (local, &length) in manifest
                .morph_child_run_lengths_at(index)
                .ok_or("invalid spatial run range")?
                .iter()
                .enumerate()
            {
                end = end
                    .checked_add(u32::from(length))
                    .ok_or("spatial run sum overflow")?;
                words[run_start + runs.start as usize + local] = end;
            }
            let mut child_offset = 0u32;
            for child in node.children.start..node.children.end().ok_or("child range overflow")? {
                let child_descriptor = HEADER_WORDS + (root_count + child as usize) * NODE_WORDS;
                words[child_descriptor] = compiled as u32;
                words[child_descriptor + 1] = child_offset;
                child_offset = child_offset
                    .checked_add(manifest.nodes[child as usize].representation.count)
                    .ok_or("spatial child count overflow")?;
            }
        }
        // The traversal has a short alias prefix for roots, followed by the
        // complete manifest order. Alias descriptors retain the same run map.
        let scratch_bytes = root_count
            .checked_mul(size_of::<(crate::LodNodeId, usize)>())
            .ok_or("spatial root scratch overflow")?;
        let _scratch = ledger
            .try_reserve(LodMemoryCategory::MetadataCpu, scratch_bytes as u64)
            .map_err(|error| error.to_string())?;
        let mut roots: Vec<_> = manifest
            .roots
            .iter()
            .copied()
            .enumerate()
            .map(|(alias, id)| (id, alias))
            .collect();
        roots.sort_unstable_by_key(|&(id, _)| id);
        for (index, node) in manifest.nodes.iter().enumerate() {
            let Ok(root) = roots.binary_search_by_key(&node.id, |&(id, _)| id) else {
                continue;
            };
            let alias = roots[root].1;
            let source = HEADER_WORDS + (root_count + index) * NODE_WORDS;
            let destination = HEADER_WORDS + alias * NODE_WORDS;
            words.copy_within(source..source + NODE_WORDS, destination);
        }
        Ok(Self {
            tree,
            words: Arc::new(words),
            _lease: lease,
        })
    }

    pub fn mapping_bytes(&self) -> u64 {
        self.words.len() as u64 * 4
    }
}

#[cfg(test)]
mod tests {
    use super::covariance_envelope_scale;

    #[test]
    fn covariance_envelope_covers_near_crossing_between_safe_endpoints() {
        let (parent_mean, parent_radius, child_mean, child_radius, near) =
            (-3.2_f32, 3.0_f32, -0.2_f32, 0.0_f32, -0.1_f32);
        assert!(parent_mean + parent_radius < near);
        assert!(child_mean + child_radius < near);
        let mean = (parent_mean + child_mean) * 0.5;
        let radius = ((parent_radius.powi(2) + child_radius.powi(2)) * 0.5).sqrt();
        assert!(mean + radius > near);
        // The authored parent box also contains the child's point support.
        let envelope_radius = parent_radius * covariance_envelope_scale(3.0).unwrap();
        assert!(parent_mean + envelope_radius > near);
        assert!(mean - radius >= parent_mean - envelope_radius);
        assert!(mean + radius <= parent_mean + envelope_radius);
        assert_eq!(covariance_envelope_scale(1.5).unwrap(), 3.0);
        assert!(covariance_envelope_scale(0.0).is_err());
    }
}
