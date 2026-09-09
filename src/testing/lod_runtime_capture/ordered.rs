//! Native ordered-quad capture uses the draw's exact hierarchy input receipt.

use super::hierarchy::{
    HierarchyCapture, current_traversal, traversal_input_evidence, valid_traversal,
};
use crate::{
    PlanarGaussian3d,
    render::{
        ordered::{GaussianGlobalOrderSettings, OrderedViews},
        traversal::{GpuLodTraversalOutputs, GpuLodTraversalSettings},
    },
};
use bevy::{asset::AssetId, prelude::*, render::view::RetainedViewEntity};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeOrderedGpuConfig {
    pub max_projected_gaussians: u32,
    pub max_gpu_bytes: u64,
    pub max_traversal_gpu_bytes: u64,
    pub max_frontier_nodes: u32,
    pub max_visited_nodes: u32,
    pub max_page_requests: u32,
}

impl RuntimeOrderedGpuConfig {
    pub(super) fn ordered(&self) -> GaussianGlobalOrderSettings {
        GaussianGlobalOrderSettings {
            max_projected_gaussians: self.max_projected_gaussians,
            max_gpu_bytes: self.max_gpu_bytes,
        }
    }

    pub(super) fn traversal(&self, max_active: u64) -> GpuLodTraversalSettings {
        GpuLodTraversalSettings {
            max_selected_gaussians: max_active.min(u64::from(self.max_projected_gaussians)) as u32,
            max_frontier_nodes: self.max_frontier_nodes,
            max_visited_nodes: self.max_visited_nodes,
            max_page_requests: self.max_page_requests,
            max_gpu_bytes: self.max_traversal_gpu_bytes,
        }
    }
}

pub(super) fn capture(
    view: RetainedViewEntity,
    ordered: &OrderedViews,
    traversals: &GpuLodTraversalOutputs,
    clouds: impl Iterator<Item = (Entity, AssetId<PlanarGaussian3d>, u64)>,
) -> Result<HierarchyCapture, &'static str> {
    let draw = ordered
        .capture(view)
        .ok_or("no_current_ordered_draw_submission")?;
    let output = current_traversal(view, traversals, clouds, &draw.traversals)?;
    Ok(HierarchyCapture {
        renderer: draw.feedback.clone(),
        renderer_header_bytes: 72,
        traversal: output.feedback.clone(),
        selected_range: output.capture_selected_range,
        generation: output.residency_generation,
        capacity: output.capacity,
        gpu_bytes: draw.gpu_bytes,
        evidence: serde_json::json!({
            "pipeline":"hierarchy_ordered",
            "source":"post_render_same_submission_ordered_draw_and_traversal_copy",
            "ordered_draw_receipt":true, "ordered_submission":draw.submission,
            "spatial_pipeline":draw.spatial,
            "spatial_requested":draw.spatial_requested,
            "traversal_submission":output.submission,
            "renderer_traversal_mapping":traversal_input_evidence(draw.submission, &draw.traversals, output),
            "residency_generation":output.residency_generation,
            "allocation_generation":output.generation,
            "omission_enabled":output.omission_enabled,
            "omission_parameters":output.omission_parameters,
            "omission_parameter_layout":["support_inflation_or_zero","mip_world_factor","perspective","spatial_annotation_active"],
            "omission_frustum":output.omission_frustum,
            "omission_frustum_scope":"six actual ViewUniform world-space frustum planes used by this producer submission; camera cropping can preserve the full-camera frustum",
            "omission_evidence_scope":"producer uniform inputs; readback validates physical expansion and pinned-manifest attribution validates logical coverage; neither independently reproves each omission predicate",
            "cloud":output.main_cloud.to_bits().to_string(),
            "source_asset":format!("{:?}",output.source),
            "count_scope":"physically expanded GPU hierarchy records; logical cut nodes are reported separately; compacted/drawn admitted globally ordered quads; no pixel coverage assertion",
        }),
    })
}

pub(super) fn valid_counts(
    words: &[u32; 34],
    config: &RuntimeOrderedGpuConfig,
    capacity: u64,
) -> bool {
    let traversal: &[u32; 16] = words[18..].try_into().unwrap();
    valid_traversal(traversal, &config.traversal(capacity), capacity, words[7])
        && words[14] <= traversal[8]
        && words[0] == 4
        && words[2] == 0
        && words[3] == 0
        && words[12] == 0
        && words[13] == 0
        && words[1] <= traversal[8]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_capture_requires_completed_bounded_draw_and_complete_root_cover() {
        let config = RuntimeOrderedGpuConfig {
            max_projected_gaussians: 64,
            max_gpu_bytes: 1 << 20,
            max_traversal_gpu_bytes: 1 << 20,
            max_frontier_nodes: 16,
            max_visited_nodes: 32,
            max_page_requests: 4,
        };
        let mut words = [0; 34];
        words[0] = 4;
        words[1] = 24;
        words[26] = 32;
        words[27] = 8;
        words[28] = 4;
        words[29] = 2;
        // A bounded, coarser complete cut remains a valid image.
        words[30] = 2 | 8 | 16;
        assert!(valid_counts(&words, &config, 64));
        // Transition parents add pinned pages, while the visible cut remains
        // independently bounded by the ordinary frontier/record limits.
        let mut spatial = words;
        spatial[7] = 4;
        spatial[14] = 24;
        spatial[29] = 20;
        assert!(valid_counts(&spatial, &config, 64));
        spatial[29] = 21;
        assert!(!valid_counts(&spatial, &config, 64));
        spatial[29] = 20;
        spatial[14] = 33;
        assert!(!valid_counts(&spatial, &config, 64));
        for (index, value) in [
            (0, 0),
            (1, 33),
            (12, 1),
            (12, 2),
            (13, 2),
            (26, 65),
            (27, 33),
            (30, 1),
            (30, 64),
        ] {
            let mut invalid = words;
            invalid[index] = value;
            assert!(!valid_counts(&invalid, &config, 64), "index={index}");
        }
    }
}
