//! Shared receipt validation for captures consuming a GPU-selected hierarchy.

use crate::{
    PlanarGaussian3d,
    render::traversal::{
        GpuLodTraversalCaptureInput, GpuLodTraversalOutput, GpuLodTraversalOutputs,
        GpuLodTraversalSettings, OMITTED_OUTSIDE_VIEW,
    },
};
use bevy::{
    asset::AssetId,
    prelude::*,
    render::{render_resource::Buffer, view::RetainedViewEntity},
};

pub(super) struct HierarchyCapture {
    pub renderer: Buffer,
    pub renderer_header_bytes: u64,
    pub traversal: Buffer,
    pub selected_range: (u64, u32),
    pub generation: u64,
    pub capacity: u32,
    pub gpu_bytes: u64,
    pub evidence: serde_json::Value,
}

pub(super) const SELECTED_RANGE_BYTES: u64 = 16;
pub(super) const SELECTED_RANGE_LAYOUT: &str =
    "GPU node index, physical output start, physical record count, omission flags";
pub(super) const OMISSION_POLICY: &str = "conservative_current_view_world_support";

/// Checks physical expansion only. Manifest-backed consumers independently
/// verify representation counts and the complete logical source antichain.
pub(super) fn valid_selected_ranges(
    rows: &[[u32; 4]],
    selected: u32,
    omission_enabled: bool,
) -> bool {
    let mut end = 0_u32;
    for &[_, start, count, flags] in rows {
        if start != end
            || !matches!((flags, count), (0, 1..) | (OMITTED_OUTSIDE_VIEW, 0))
            || (flags != 0 && !omission_enabled)
        {
            return false;
        }
        let Some(next) = end.checked_add(count) else {
            return false;
        };
        end = next;
    }
    end == selected
}

pub(super) fn current_traversal<'a>(
    view: RetainedViewEntity,
    traversals: &'a GpuLodTraversalOutputs,
    clouds: impl Iterator<Item = (Entity, AssetId<PlanarGaussian3d>, u64)>,
    inputs: &[GpuLodTraversalCaptureInput],
) -> Result<&'a GpuLodTraversalOutput, &'static str> {
    if inputs.len() != 1 {
        return Err("capture_requires_exactly_one_renderer_hierarchy_input");
    }
    clouds
        .filter_map(|(entity, source, generation)| {
            let output = traversals.get(view, entity)?;
            (output.is_ready()
                && output.submission > 0
                && output.source == source
                && output.residency_generation == generation
                && inputs[0] == output.into())
            .then_some(output)
        })
        .next()
        .ok_or("renderer_input_does_not_match_current_resident_snapshot")
}

/// Retains the renderer's observed input identities separately from the current
/// traversal output matched above. Their submission counters are independent.
/// Both headers are copied after the renderer in this same render invocation;
/// the completed readback still has to validate image/count success.
pub(super) fn traversal_input_evidence(
    renderer_submission: u64,
    inputs: &[GpuLodTraversalCaptureInput],
    output: &GpuLodTraversalOutput,
) -> serde_json::Value {
    serde_json::json!({
        "schema_version":1,
        "association":"current_render_encoded_inputs",
        "renderer_submission":renderer_submission,
        "renderer_inputs":inputs.iter().map(|input| serde_json::json!({
            "cloud":input.cloud.to_bits().to_string(),
            "residency_generation":input.residency_generation,
            "source_asset":format!("{:?}",input.source),
            "allocation_generation":input.allocation_generation,
            "submission":input.submission,
        })).collect::<Vec<_>>(),
        "traversal_output":{
            "cloud":output.main_cloud.to_bits().to_string(),
            "source_asset":format!("{:?}",output.source),
            "residency_generation":output.residency_generation,
            "allocation_generation":output.generation,
            "submission":output.submission,
        },
    })
}

pub(super) fn valid_traversal(
    words: &[u32; 16],
    settings: &GpuLodTraversalSettings,
    capacity: u64,
    parent_pages: u32,
) -> bool {
    let flags = words[12];
    flags & !63 == 0
        && flags & 1 == 0
        && u64::from(words[8]) <= capacity
        && words[8] <= settings.max_selected_gaussians
        && words[9] <= settings.max_visited_nodes
        && (words[10] <= settings.max_page_requests || flags & 8 != 0)
        && words[5] <= settings.max_frontier_nodes
        && parent_pages <= settings.max_frontier_nodes
        && words[11] <= settings.max_frontier_nodes.saturating_add(parent_pages)
}

pub(super) fn traversal_evidence(words: &[u32; 16]) -> serde_json::Value {
    serde_json::json!({
        "request_count_scope":"bounded_raw_page_references;shared_page_references_may_repeat;host_feedback_deduplicates",
        "selected_gaussians":words[8], "visited_nodes":words[9],
        "selected_nodes":words[5],
        "selected_count_scope":"physical expanded records; selected_nodes separately counts the complete logical cut including explicitly omitted ranges",
        "requested_pages":words[10], "selected_pages":words[11], "flags":words[12],
        "request_order_scope":"complete required child cohorts before pre-band cohorts; stable dedup preserves per-view priority; physical omission does not alter logical residency or cohort demand",
        "complete":words[12] & 1 == 0, "record_limited":words[12] & 2 != 0,
        "frontier_limited":words[12] & 4 != 0, "request_overflow":words[12] & 8 != 0,
        "visit_limited":words[12] & 16 != 0,
        "cutoff_unavailable":words[12] & 32 != 0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_page_requests_do_not_require_child_visits() {
        let settings = GpuLodTraversalSettings {
            max_visited_nodes: 1,
            max_page_requests: 8,
            ..Default::default()
        };
        let mut words = [0; 16];
        words[9] = 1;
        words[10] = 8;
        assert!(valid_traversal(&words, &settings, 1, 0));
        words[10] = 9;
        assert!(!valid_traversal(&words, &settings, 1, 0));
        words[12] = 8;
        assert!(valid_traversal(&words, &settings, 1, 0));
    }

    #[test]
    fn physical_range_receipt_requires_explicit_omissions_and_exact_prefix() {
        let mut rows = [
            [0, 0, 0, OMITTED_OUTSIDE_VIEW],
            [1, 0, 3, 0],
            [2, 3, 0, OMITTED_OUTSIDE_VIEW],
        ];
        assert!(valid_selected_ranges(&rows, 3, true));
        assert!(!valid_selected_ranges(&rows, 3, false));
        rows[1][1] = 1;
        assert!(!valid_selected_ranges(&rows, 3, true));
        rows[1][1] = 0;
        rows[0][3] = 0;
        assert!(!valid_selected_ranges(&rows, 3, true));
        rows[0][3] = OMITTED_OUTSIDE_VIEW;
        rows[1][3] = OMITTED_OUTSIDE_VIEW;
        assert!(!valid_selected_ranges(&rows, 3, true));
        assert!(valid_selected_ranges(
            &[[0, 0, 0, OMITTED_OUTSIDE_VIEW]],
            0,
            true
        ));
    }
}
