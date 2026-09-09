//! Native GPS capture uses the renderer's exact image/traversal receipt. Point
//! attempts remain separate from projected Gaussian counts and pixel coverage.

use super::hierarchy::{
    HierarchyCapture, current_traversal, traversal_input_evidence, valid_traversal,
};
use bevy::{asset::AssetId, prelude::*, render::view::RetainedViewEntity};
use serde::{Deserialize, Serialize};

use crate::{
    PlanarGaussian3d,
    render::{
        point::{GaussianPointSplattingSettings, GaussianPointSplattingViewBudget, PointViews},
        traversal::{GpuLodTraversalOutputs, GpuLodTraversalSettings},
    },
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePointGpuConfig {
    pub samples_per_pixel: u32,
    #[serde(default = "minimum_samples")]
    pub min_samples_per_pixel: u32,
    pub max_projected_gaussians: u32,
    pub max_points_per_frame: u32,
    pub max_gpu_bytes: u64,
    pub max_traversal_gpu_bytes: u64,
    pub max_frontier_nodes: u32,
    pub max_visited_nodes: u32,
    pub max_page_requests: u32,
    #[serde(default)]
    pub target_gpu_ms: Option<f32>,
    #[serde(default)]
    pub target_view_gpu_ms: Option<f32>,
}

fn minimum_samples() -> u32 {
    1
}

impl RuntimePointGpuConfig {
    pub(super) fn point(&self) -> GaussianPointSplattingSettings {
        GaussianPointSplattingSettings {
            samples_per_pixel: self.samples_per_pixel,
            min_samples_per_pixel: self.min_samples_per_pixel,
            max_projected_gaussians: self.max_projected_gaussians,
            max_points_per_frame: self.max_points_per_frame,
            max_gpu_bytes: self.max_gpu_bytes,
            temporal_sampling: false,
            target_gpu_ms: self.target_gpu_ms,
            ..Default::default()
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

    pub(super) fn view_budget(&self, max_active: u64) -> Option<GaussianPointSplattingViewBudget> {
        let max_selected_gaussians = self.traversal(max_active).max_selected_gaussians;
        self.target_view_gpu_ms
            .map(|target_view_gpu_ms| GaussianPointSplattingViewBudget {
                target_view_gpu_ms,
                max_selected_gaussians,
                min_selected_gaussians: GaussianPointSplattingViewBudget::default()
                    .min_selected_gaussians
                    .min(max_selected_gaussians),
            })
    }
}

pub(super) fn capture(
    view: RetainedViewEntity,
    points: &PointViews,
    traversals: &GpuLodTraversalOutputs,
    clouds: impl Iterator<Item = (Entity, AssetId<PlanarGaussian3d>, u64)>,
) -> Result<HierarchyCapture, &'static str> {
    let point = points
        .capture(view)
        .ok_or("no_current_point_image_submission")?;
    let inputs = points
        .capture_traversals(view, point.submission)
        .ok_or("no_current_point_traversal_receipt")?;
    let output = current_traversal(view, traversals, clouds, inputs)?;
    Ok(HierarchyCapture {
        renderer: point.feedback,
        renderer_header_bytes: 32,
        traversal: output.feedback.clone(),
        selected_range: output.capture_selected_range,
        generation: output.residency_generation,
        capacity: output.capacity,
        gpu_bytes: point.gpu_bytes,
        evidence: serde_json::json!({
            "pipeline":"hierarchy_point",
            "source":"post_render_same_submission_point_and_traversal_copy",
            "point_image_receipt":true,
            "point_submission":point.submission,
            "traversal_submission":output.submission,
            "renderer_traversal_mapping":traversal_input_evidence(point.submission, inputs, output),
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
            "samples_per_pixel":point.samples_per_pixel,
            "count_scope":"physically expanded GPU hierarchy records; logical cut nodes are reported separately; compacted/drawn admitted projected Gaussians; stochastic point attempts reported separately; no pixel coverage assertion",
        }),
    })
}

pub(super) fn valid_counts(
    words: &[u32; 24],
    config: &RuntimePointGpuConfig,
    capacity: u64,
) -> bool {
    let traversal: &[u32; 16] = words[8..].try_into().unwrap();
    valid_traversal(traversal, &config.traversal(capacity), capacity, 0)
        && words[4] == 0
        && words[5] <= words[16]
        && words[3] <= config.max_points_per_frame
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_capture_control_defaults_are_fixed_and_targets_validate() {
        let mut config: RuntimePointGpuConfig = serde_json::from_value(serde_json::json!({
            "samples_per_pixel":4, "max_projected_gaussians":1024,
            "max_points_per_frame":65536, "max_gpu_bytes":1048576,
            "max_traversal_gpu_bytes":1048576, "max_frontier_nodes":64,
            "max_visited_nodes":128, "max_page_requests":16,
        }))
        .unwrap();
        assert!(config.point().target_gpu_ms.is_none());
        assert!(config.view_budget(1024).is_none());
        config.target_gpu_ms = Some(2.0);
        config.target_view_gpu_ms = Some(4.0);
        config.point().validate().unwrap();
        let policy = config.view_budget(512).unwrap();
        policy.validate().unwrap();
        assert_eq!(policy.max_selected_gaussians, 512);
        config.target_gpu_ms = Some(f32::NAN);
        config.target_view_gpu_ms = Some(0.0);
        assert!(config.point().validate().is_err());
        assert!(config.view_budget(512).unwrap().validate().is_err());
    }

    #[test]
    fn point_counts_reject_overflow_and_keep_safe_traversal_limit_flags() {
        let config = RuntimePointGpuConfig {
            samples_per_pixel: 1,
            min_samples_per_pixel: 1,
            max_projected_gaussians: 64,
            max_points_per_frame: 1024,
            max_gpu_bytes: 1024 * 1024,
            max_traversal_gpu_bytes: 1024 * 1024,
            max_frontier_nodes: 16,
            max_visited_nodes: 32,
            max_page_requests: 4,
            target_gpu_ms: None,
            target_view_gpu_ms: None,
        };
        let mut words = [0; 24];
        words[3] = 128;
        words[5] = 32;
        words[16] = 64;
        words[17] = 8;
        words[18] = 5;
        words[19] = 2;
        words[20] = 8;
        assert!(valid_counts(&words, &config, 64));
        for (index, value) in [
            (4, 1),
            (5, 65),
            (16, 65),
            (17, 33),
            (20, 1),
            (20, 64),
            (20, 0),
        ] {
            let mut invalid = words;
            invalid[index] = value;
            assert!(!valid_counts(&invalid, &config, 64));
        }
    }
}
