//! Complete fixed-region context selection from an authenticated captured cut.
//! Every potential OBB intersection is retained, regardless of opacity/occlusion.

use super::super::attribute_cut::{cut_from_receipt, frame_row};
use super::*;
use crate::{
    camera::path::GaussianCameraPath,
    testing::{
        lod_capture::{LodCapturePipeline, LodCountSource, LodFrameCapture},
        lod_runtime_capture::RuntimeCaptureConfig,
        lod_scenes::{LodPixelCrop, LodTestCamera},
        render_oracle::{ProductionPixelProjector, fit::FitView},
    },
};

const MAX_SCAN_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ContextView {
    pub frame_index: usize,
    pub calibration_viewport: [u32; 2],
    pub crop: LodPixelCrop,
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapturedContext {
    pub capture: PinnedFile,
    pub receipt: PinnedFile,
    pub frame: u64,
    pub camera_path: PinnedFile,
    /// Union of every train/evaluation physical crop. The same frozen cut is
    /// reprojected at each pose; no per-pose runtime cut equivalence is implied.
    pub views: Vec<ContextView>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ContextRecordRun {
    pub node: LodNodeId,
    pub page: LodPageId,
    pub decoded_page_offset: u32,
    pub count: u32,
    pub captured_node_output_start: u32,
    pub node_record_offset: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ContextEvidence {
    pub schema_version: u32,
    pub complete_region_support_union: bool,
    pub selected_cut_records: u64,
    /// Present only for physically omitted cuts: context is restricted to
    /// contained crops of the identical captured full-camera frustum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub omitted_logical_nodes_captured_view_only: Option<u32>,
    pub scanned_encoded_bytes: u64,
    pub scanned_decoded_bytes: u64,
    pub context_records: u64,
    /// All cut owners retain their captured order, including owned substitutions.
    pub owned_output_starts: BTreeMap<LodNodeId, u32>,
    pub runs: Vec<ContextRecordRun>,
}

fn cameras(profile: &CapturedContext) -> CaptureResult<Vec<LodTestCamera>> {
    if profile.views.is_empty() || profile.views.len() > 8 {
        return Err("captured context admits one to eight frozen camera/crop regions".into());
    }
    let path = GaussianCameraPath::from_json(&read_pinned(&profile.camera_path, 4 * 1024 * 1024)?)?;
    let mut cameras = Vec::new();
    for view in &profile.views {
        if view.calibration_viewport.contains(&0)
            || view.calibration_viewport.iter().any(|&n| n > 8192)
        {
            return Err("context calibration viewport exceeds 8192 pixels per axis".into());
        }
        view.crop.validate(view.calibration_viewport)?;
        let frame = path
            .frames()
            .get(view.frame_index)
            .ok_or("context camera frame is absent")?;
        let camera = LodTestCamera::from_camera_path_frame(
            frame,
            view.calibration_viewport,
            view.near,
            view.far,
        )?;
        cameras.push(camera);
    }
    Ok(cameras)
}

pub(super) fn validate_views(profile: &CapturedContext, actual: &[FitView]) -> CaptureResult<()> {
    let expected = cameras(profile)?
        .into_iter()
        .zip(&profile.views)
        .map(|(camera, view)| camera.with_crop(view.crop))
        .collect::<Result<Vec<_>, _>>()?;
    for view in actual {
        if !expected.iter().any(|camera| {
            let mut camera = *camera;
            camera.viewport = view.camera.viewport; // Sparse grid never changes physical crop.
            camera == view.camera
        }) {
            return Err(
                "fit camera/crop is outside the authenticated context support union".into(),
            );
        }
    }
    Ok(())
}

fn append_record(
    runs: &mut Vec<ContextRecordRun>,
    node: &GaussianLodNode,
    output_start: u32,
    offset: u32,
) {
    if let Some(last) = runs.last_mut()
        && last.node == node.id
        && last.node_record_offset + last.count == offset
    {
        last.count += 1;
        return;
    }
    runs.push(ContextRecordRun {
        node: node.id,
        page: node.representation.page,
        decoded_page_offset: node.representation.offset + offset,
        count: 1,
        captured_node_output_start: output_start,
        node_record_offset: offset,
    });
}

pub(super) fn scan(
    manifest: &GaussianLodManifest,
    config: &CohortExportConfig,
    owned: &[&GaussianLodNode],
    max_context: u64,
) -> CaptureResult<ContextEvidence> {
    let started = std::time::Instant::now();
    let profile = config
        .captured_context
        .as_ref()
        .ok_or("missing captured context profile")?;
    if !config.context_nodes.is_empty() || config.include_original_context {
        return Err(
            "captured record context is exclusive with manual nodes/original context".into(),
        );
    }
    let record: LodFrameCapture =
        serde_json::from_value(frame_row(&profile.capture, profile.frame, true)?)?;
    record.validate()?;
    let receipt = frame_row(&profile.receipt, profile.frame, false)?;
    let capture_root = fs::canonicalize(
        profile
            .capture
            .path
            .parent()
            .ok_or("capture has no directory")?,
    )?;
    if fs::canonicalize(
        profile
            .receipt
            .path
            .parent()
            .ok_or("receipt has no directory")?,
    )? != capture_root
    {
        return Err("context capture and cut receipt must share one run directory".into());
    }
    if record.identity.manifest_sha256 != config.manifest.sha256.to_lowercase()
        || record.counts.pipeline != LodCapturePipeline::HierarchyOrdered
        || record.counts.source != LodCountSource::GpuReadback
        || record.counts.drawn.is_none()
        || record.counts.transition_extra != 0
        || record.counts.candidates != record.counts.selected
        || receipt["pipeline"] != "hierarchy_ordered"
        || receipt["counts_valid"] != true
        || receipt["ordered_draw_receipt"] != true
        || receipt["ordered_image_attested"] != true
        || receipt["source"] != "post_render_same_submission_ordered_draw_and_traversal_copy"
        || receipt["hierarchy_cut"]["valid"] != true
        || receipt["residency_generation"].as_u64() != Some(record.stamp.generation)
        || receipt["traversal"]["selected_gaussians"].as_u64() != Some(record.counts.selected)
        || receipt["traversal"]["complete"] != true
        || receipt["traversal"]["flags"]
            .as_u64()
            .is_none_or(|f| f & !63 != 0 || f & 1 != 0)
        || receipt["ordered"]["flags"] != 0
        || receipt["ordered"]["traversal_failure"] != 0
        || receipt["ordered"]["vertex_count"] != 4
        || receipt["ordered_submission"]
            .as_u64()
            .is_none_or(|n| n == 0)
        || receipt["traversal_submission"]
            .as_u64()
            .is_none_or(|n| n == 0)
        || receipt["ordered"]["projected_gaussians"].as_u64() != record.counts.drawn
    {
        return Err("captured context requires a complete attested ordered cut".into());
    }
    let settings_path = capture_root.join("settings.json");
    let settings_bytes = read_pinned(
        &PinnedFile {
            path: settings_path,
            sha256: record.identity.settings_sha256.clone(),
        },
        1024 * 1024,
    )?;
    let settings: RuntimeCaptureConfig = serde_json::from_slice(&settings_bytes)?;
    if settings.source_metadata.is_some()
        || !settings.capture_hierarchy_cut
        || settings.render_mode != LodCapturePipeline::HierarchyOrdered
    {
        return Err(
            "captured context requires native identity-cloud fixed three-sigma settings".into(),
        );
    }
    let cut = cut_from_receipt(manifest, &receipt, record.counts.selected)?;
    let omitted_nodes = cut.iter().filter(|range| range.physical_count == 0).count() as u32;
    let starts = cut
        .iter()
        .filter(|range| range.physical_count > 0)
        .map(|range| (range.node.id, range.output_start))
        .collect::<BTreeMap<_, _>>();
    let mut owned_output_starts = BTreeMap::new();
    for node in owned {
        owned_output_starts.insert(
            node.id,
            *starts
                .get(&node.id)
                .ok_or("owned node must belong to the physically expanded captured cut")?,
        );
    }
    let camera_values = cameras(profile)?;
    let path = GaussianCameraPath::from_json(&read_pinned(&profile.camera_path, 4 * 1024 * 1024)?)?;
    if omitted_nodes > 0 {
        // The captured omission proof belongs to one frustum. Reprojecting its
        // remaining records elsewhere would silently discard possible context.
        // Cropped capture frusta require a separate containment proof; admit
        // only full-camera captures and already validated contained crops here.
        if settings.camera_crop.is_some() {
            return Err("omitted-cut context requires an uncropped captured camera frustum".into());
        }
        for view in &profile.views {
            let frame = &path.frames()[view.frame_index];
            if view.calibration_viewport != record.camera.viewport
                || view.near != settings.near
                || view.far != 100_000.0
                || record.camera.projection
                    != frame
                        .projection(view.near, view.far)?
                        .get_clip_from_view()
                        .to_cols_array()
                        .map(f64::from)
                || record.camera.world_to_view
                    != frame
                        .transform()
                        .to_matrix()
                        .inverse()
                        .to_cols_array()
                        .map(f64::from)
            {
                return Err("physically omitted cut cannot provide complete context at another camera/frustum".into());
            }
        }
    }
    let projectors = camera_values
        .into_iter()
        .zip(&profile.views)
        .map(|(camera, view)| {
            let frame = &path.frames()[view.frame_index];
            ProductionPixelProjector::new(
                camera,
                frame.projection(view.near, view.far)?.get_clip_from_view(),
                frame.transform().to_matrix().inverse(),
            )
            .map_err(Into::into)
        })
        .collect::<CaptureResult<Vec<_>>>()?;
    let mut pages = BTreeMap::<LodPageId, Vec<usize>>::new();
    for (index, range) in cut
        .iter()
        .enumerate()
        .filter(|(_, range)| range.physical_count > 0)
    {
        pages
            .entry(range.node.representation.page)
            .or_default()
            .push(index);
    }
    let descriptors = manifest
        .pages
        .iter()
        .map(|page| (page.id, page))
        .collect::<BTreeMap<_, _>>();
    let mut encoded_bytes = 0_u64;
    let mut decoded_bytes = 0_u64;
    for id in pages.keys() {
        let descriptor = descriptors[id];
        let storage = descriptor
            .storage
            .as_ref()
            .ok_or("context cut page has no storage")?;
        validate_native_page_location(
            *id,
            &ManifestPageLocation {
                uri: storage.uri.clone().into(),
                byte_range: storage.byte_range,
                encoded_len: storage.encoded_len,
            },
            MAX_PAGE_BYTES,
        )?;
        encoded_bytes = encoded_bytes
            .checked_add(storage.encoded_len)
            .ok_or("context scan encoded overflow")?;
        decoded_bytes = decoded_bytes
            .checked_add(descriptor.decoded_len)
            .ok_or("context scan decoded overflow")?;
        if encoded_bytes > MAX_SCAN_BYTES
            || decoded_bytes > MAX_SCAN_BYTES
            || descriptor.decoded_len > MAX_PAGE_BYTES
        {
            return Err("context cut scan exceeds 2 GiB encoded/decoded work".into());
        }
    }
    let root = config
        .manifest
        .path
        .parent()
        .ok_or("manifest has no parent")?;
    let mut runs = Vec::new();
    let mut count = 0_u64;
    for (id, ranges) in pages {
        if started.elapsed().as_secs() >= 180 {
            return Err("context selection exceeds 180 seconds; no partial union".into());
        }
        let descriptor = descriptors[&id];
        let storage = descriptor
            .storage
            .as_ref()
            .ok_or("context storage disappeared")?;
        let path = fs::canonicalize(root.join(&storage.uri))?;
        if !path.starts_with(root) {
            return Err("context page escapes package".into());
        }
        let (start, length) = storage.byte_range.unwrap_or((0, storage.encoded_len));
        let encoded = read_exact_range(&path, start, length, storage.byte_range.is_none())?;
        let page = decode_page_with_descriptor(&encoded, descriptor, config.admission.limits()?)?;
        drop(encoded);
        for index in ranges {
            let range = &cut[index];
            if owned_output_starts.contains_key(&range.node.id) {
                continue;
            }
            let representation = range.node.representation;
            let records = page
                .gaussians
                .get(
                    representation.offset as usize
                        ..(representation.offset + representation.count) as usize,
                )
                .ok_or("context range exceeds decoded page")?;
            for (offset, gaussian) in records.iter().enumerate() {
                if projectors
                    .iter()
                    .zip(&profile.views)
                    .any(|(projector, view)| projector.intersects_region(gaussian, view.crop))
                {
                    count += 1;
                    if count > max_context {
                        return Err("complete region context exceeds source/candidate record cap; no tail omitted".into());
                    }
                    append_record(&mut runs, range.node, range.output_start, offset as u32);
                    if runs.len() > MAX_RANGES {
                        return Err("context record runs exceed 32768".into());
                    }
                }
            }
        }
    }
    runs.sort_unstable_by_key(|run| (run.captured_node_output_start, run.node_record_offset));
    Ok(ContextEvidence {
        schema_version: 1,
        complete_region_support_union: true,
        selected_cut_records: record.counts.selected,
        omitted_logical_nodes_captured_view_only: (omitted_nodes > 0).then_some(omitted_nodes),
        scanned_encoded_bytes: encoded_bytes,
        scanned_decoded_bytes: decoded_bytes,
        context_records: count,
        owned_output_starts,
        runs,
    })
}

pub(super) fn verify_selection(
    expected: &Option<ContextEvidence>,
    actual: &Option<ContextEvidence>,
) -> CaptureResult<()> {
    if expected != actual {
        return Err("context runs differ from the recomputed complete support union".into());
    }
    Ok(())
}

pub(super) fn payload(
    manifest: &GaussianLodManifest,
    evidence: &ContextEvidence,
) -> CaptureResult<PayloadPlan> {
    let nodes = manifest
        .nodes
        .iter()
        .map(|node| (node.id, node))
        .collect::<BTreeMap<_, _>>();
    let mut output_start = 0;
    let mut ranges = Vec::new();
    for run in &evidence.runs {
        let node = nodes.get(&run.node).ok_or("context run node missing")?;
        ranges.push(CohortRange {
            owner: node.id,
            node: node.id,
            source: node.source,
            conservative_node_bounds: node.bounds,
            page: run.page,
            decoded_page_offset: run.decoded_page_offset,
            output_start,
            output_count: run.count,
            decoded_records_sha256: String::new(),
        });
        output_start += u64::from(run.count);
    }
    let mut header = Vec::new();
    write_ply_header(&mut header, output_start)?;
    let bytes = (header.len() as u64)
        .checked_add(
            output_start
                .checked_mul(PLY_RECORD_BYTES)
                .ok_or("context output size overflow")?,
        )
        .ok_or("context size overflow")?;
    Ok(PayloadPlan {
        role: CohortRole::Context,
        count: output_start,
        ranges,
        header,
        bytes,
    })
}

/// Captured stable order for original source, seed representatives and context.
type CohortOrdering = (Option<Vec<u64>>, Option<Vec<u64>>, Option<Vec<u64>>);

pub(super) fn ordering(sidecar: &CohortSidecar) -> CaptureResult<CohortOrdering> {
    let Some(evidence) = &sidecar.context_selection else {
        return Ok((None, None, None));
    };
    let owned_starts = sidecar.payloads[1]
        .ranges
        .iter()
        .map(|r| (r.owner, r.source.start))
        .collect::<BTreeMap<_, _>>();
    let mut result = Vec::new();
    for payload in &sidecar.payloads[..2] {
        let mut keys = Vec::with_capacity(payload.gaussian_count as usize);
        for range in &payload.ranges {
            let prefix = u64::from(
                *evidence
                    .owned_output_starts
                    .get(&range.owner)
                    .ok_or("owned order missing")?,
            ) << 32;
            let start = if payload.role == CohortRole::Source {
                range.source.start - owned_starts[&range.owner]
            } else {
                0
            };
            for index in 0..range.output_count {
                let low = u32::try_from(start + u64::from(index))?;
                keys.push(prefix | u64::from(low));
            }
        }
        result.push(keys);
    }
    let context = evidence
        .runs
        .iter()
        .flat_map(|run| {
            (0..run.count).map(move |offset| {
                (u64::from(run.captured_node_output_start) << 32)
                    | u64::from(run.node_record_offset + offset)
            })
        })
        .collect();
    Ok((
        Some(result.remove(0)),
        Some(result.remove(0)),
        Some(context),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::lod_scenes::LodProjection;
    use bevy::prelude::{Mat4, Vec2, Vec3};

    #[test]
    fn region_context_retains_boundary_and_hidden_records_and_rejects_forged_omission() {
        let camera = LodTestCamera {
            position: Vec3::ZERO,
            target: -Vec3::Z,
            projection: LodProjection::Calibrated {
                focal_length_px: Vec2::splat(64.0),
                principal_point_px: Vec2::splat(64.0),
                image_size: [128, 128],
                crop: None,
            },
            viewport: [128, 128],
            near: 0.1,
            ..Default::default()
        };
        let projector = ProductionPixelProjector::new(
            camera,
            Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1),
            Mat4::IDENTITY,
        )
        .unwrap();
        let crop = LodPixelCrop {
            origin: [60, 60],
            size: [8, 8],
        };
        let mut near = Gaussian3d::default();
        near.position_visibility.position = [0.0, 0.0, -3.0];
        near.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
        near.scale_opacity.scale = [0.1; 3];
        near.scale_opacity.opacity = 1.0;
        let mut boundary = near;
        boundary.position_visibility.position[0] = 0.375; // Center 72, support reaches crop.
        let mut hidden = near;
        hidden.position_visibility.position[2] = -6.0;
        hidden.scale_opacity.opacity = 0.0; // Deliberately retained without alpha pruning.
        let mut outside = near;
        outside.position_visibility.position[0] = 2.0;
        let retained = [near, boundary, hidden, outside]
            .iter()
            .enumerate()
            .filter_map(|(i, g)| projector.intersects_region(g, crop).then_some(i))
            .collect::<Vec<_>>();
        assert_eq!(retained, vec![0, 1, 2]);
        let evidence = ContextEvidence {
            schema_version: 1,
            complete_region_support_union: true,
            selected_cut_records: 4,
            omitted_logical_nodes_captured_view_only: None,
            scanned_encoded_bytes: 64,
            scanned_decoded_bytes: 64,
            context_records: 3,
            owned_output_starts: BTreeMap::new(),
            runs: vec![ContextRecordRun {
                node: LodNodeId(1),
                page: LodPageId(1),
                decoded_page_offset: 0,
                count: 3,
                captured_node_output_start: 0,
                node_record_offset: 0,
            }],
        };
        let mut forged = evidence.clone();
        forged.runs[0].count = 2;
        forged.context_records = 2;
        assert!(verify_selection(&Some(evidence.clone()), &Some(evidence.clone())).is_ok());
        assert!(verify_selection(&Some(evidence), &Some(forged)).is_err());
    }
}
