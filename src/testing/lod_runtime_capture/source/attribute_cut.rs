//! Bounded CPU attribution of an authenticated, actually rendered GPU cut.
//! This is a diagnostic owner-selection tool, not GPU image parity evidence.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use bevy::prelude::Mat4;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::super::hierarchy::{OMISSION_POLICY, SELECTED_RANGE_BYTES, SELECTED_RANGE_LAYOUT};
use super::{CaptureResult, cohort::PinnedFile, export_rung::read_exact_range, hash_file};
use crate::render::traversal::OMITTED_OUTSIDE_VIEW;
use crate::{
    camera::path::GaussianCameraPath,
    gaussian::{
        formats::{
            planar_3d_chunked::LodPageId,
            planar_3d_lod::{GaussianLodManifest, GaussianLodNode},
        },
        settings::GaussianColorSpace,
    },
    io::lod::{LodCodecLimits, decode_manifest, decode_page_with_descriptor},
    stream::transport::{ManifestPageLocation, validate_native_page_location},
    testing::{
        lod_capture::{LodCaptureMode, LodCapturePipeline, LodCountSource, LodFrameCapture},
        lod_runtime_capture::RuntimeCaptureConfig,
        lod_scenes::LodTestCamera,
        render_oracle::{ProductionPixelContribution, ProductionPixelProjector},
    },
};

const MAX_RECORDS: u64 = 30_000_000;
const MAX_PAGE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOTAL_PAGE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_HITS: usize = 100_000;
const MAX_CUT_NODES: usize = 32_768;
const MAX_RECEIPT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_LINE_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    manifest: PinnedFile,
    capture: PinnedFile,
    receipt: PinnedFile,
    frame: u64,
    camera_path: PinnedFile,
    camera_frame: usize,
    pixels: Vec<[u32; 2]>,
    output: PathBuf,
}

fn limits() -> LodCodecLimits {
    LodCodecLimits {
        max_manifest_bytes: 256 * 1024 * 1024,
        max_nodes: 262_144,
        max_pages: 262_144,
        max_page_bytes: MAX_PAGE_BYTES,
        max_page_gaussians: 65_535,
    }
}

fn verify(file: &PinnedFile, limit: u64) -> CaptureResult<()> {
    if file.sha256.len() != 64
        || !file.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        || fs::metadata(&file.path)?.len() > limit
        || !hash_file(&file.path)?.eq_ignore_ascii_case(&file.sha256)
    {
        return Err(format!(
            "attribution input exceeds admission or SHA256 differs: {}",
            file.path.display()
        )
        .into());
    }
    Ok(())
}

fn pinned_bytes(file: &PinnedFile, limit: u64) -> CaptureResult<Vec<u8>> {
    let size = fs::metadata(&file.path)?.len();
    if size > limit {
        return Err("attribution input exceeds byte admission".into());
    }
    let bytes = read_exact_range(&file.path, 0, size, true)?;
    if file.sha256.len() != 64
        || !format!("{:x}", Sha256::digest(&bytes)).eq_ignore_ascii_case(&file.sha256)
    {
        return Err("attribution input SHA256 mismatch".into());
    }
    Ok(bytes)
}

fn bounded_bytes(path: &Path, limit: u64) -> CaptureResult<Vec<u8>> {
    let size = fs::metadata(path)?.len();
    if size > limit {
        return Err("attribution metadata exceeds byte admission".into());
    }
    read_exact_range(path, 0, size, true)
}

/// Read one matching row while bounding both the file and any single JSON line.
pub(super) fn frame_row(file: &PinnedFile, frame: u64, capture: bool) -> CaptureResult<Value> {
    verify(file, MAX_RECEIPT_BYTES)?;
    let mut reader = BufReader::new(File::open(&file.path)?);
    let mut found = None;
    let mut line = Vec::new();
    loop {
        line.clear();
        if reader
            .by_ref()
            .take(MAX_LINE_BYTES + 1)
            .read_until(b'\n', &mut line)?
            == 0
        {
            break;
        }
        if line.len() as u64 > MAX_LINE_BYTES {
            return Err("capture JSONL row exceeds 4 MiB".into());
        }
        let value: Value = serde_json::from_slice(&line)?;
        let row_frame = if capture {
            &value["stamp"]["frame"]
        } else {
            &value["frame"]
        };
        if row_frame.as_u64() == Some(frame) && found.replace(value).is_some() {
            return Err("duplicate capture frame; view identity is ambiguous".into());
        }
    }
    verify(file, MAX_RECEIPT_BYTES)?;
    found.ok_or_else(|| "requested capture frame is absent".into())
}

pub(super) struct CutNode<'a> {
    pub node: &'a GaussianLodNode,
    pub output_start: u32,
    pub physical_count: u32,
}

fn compiled_nodes(manifest: &GaussianLodManifest) -> CaptureResult<Vec<&GaussianLodNode>> {
    let nodes = manifest
        .nodes
        .iter()
        .map(|node| (node.id, node))
        .collect::<BTreeMap<_, _>>();
    let mut compiled = Vec::with_capacity(manifest.roots.len() + manifest.nodes.len());
    for root in &manifest.roots {
        compiled.push(
            *nodes
                .get(root)
                .ok_or("compiled root is absent from manifest")?,
        );
    }
    compiled.extend(manifest.nodes.iter());
    Ok(compiled)
}

/// Read both documented range layouts. Two-word captures expand each complete
/// representation; four-word captures must state their current-view policy.
pub(super) fn cut_from_receipt<'a>(
    manifest: &'a GaussianLodManifest,
    receipt: &Value,
    selected: u64,
) -> CaptureResult<Vec<CutNode<'a>>> {
    let evidence = &receipt["hierarchy_cut"];
    let (rows, omission_enabled): (Vec<[u32; 4]>, bool) =
        if evidence["layout"] == "GPU node index, expanded record output start" {
            let pairs: Vec<[u32; 2]> = serde_json::from_value(evidence["selected"].clone())?;
            let compiled = compiled_nodes(manifest)?;
            let rows = pairs
                .into_iter()
                .map(|[index, start]| {
                    let node = compiled
                        .get(index as usize)
                        .ok_or("captured GPU node index is outside the pinned manifest")?;
                    Ok([index, start, node.representation.count, 0])
                })
                .collect::<CaptureResult<Vec<_>>>()?;
            (rows, false)
        } else if evidence["schema_version"] == 2
            && evidence["layout"] == SELECTED_RANGE_LAYOUT
            && evidence["stride_bytes"] == SELECTED_RANGE_BYTES
            && evidence["omission_policy"] == OMISSION_POLICY
        {
            let enabled = evidence["omission_enabled"]
                .as_bool()
                .ok_or("cut omission policy is absent")?;
            let rows: Vec<[u32; 4]> = serde_json::from_value(evidence["selected"].clone())?;
            if receipt["traversal"]["selected_nodes"].as_u64() != Some(rows.len() as u64)
                || evidence["physical_records"].as_u64() != Some(selected)
                || evidence["omitted_nodes"].as_u64()
                    != Some(rows.iter().filter(|row| row[2] == 0).count() as u64)
            {
                return Err("cut logical or physical summary differs from its rows".into());
            }
            (rows, enabled)
        } else {
            return Err("unsupported captured cut layout or omission policy".into());
        };
    if evidence["valid"] != true || rows.len() as u64 > evidence["capacity"].as_u64().unwrap_or(0) {
        return Err("selected cut is invalid or exceeds captured capacity".into());
    }
    cut_plan(manifest, &rows, selected, omission_enabled)
}

fn cut_plan<'a>(
    manifest: &'a GaussianLodManifest,
    rows: &[[u32; 4]],
    selected: u64,
    omission_enabled: bool,
) -> CaptureResult<Vec<CutNode<'a>>> {
    if rows.is_empty() || rows.len() > MAX_CUT_NODES || selected > MAX_RECORDS {
        return Err("cut exceeds 32768 nodes or 30M physically selected records".into());
    }
    let compiled = compiled_nodes(manifest)?;
    let mut seen = BTreeSet::new();
    let mut cut = Vec::with_capacity(rows.len());
    for &[index, output_start, physical_count, flags] in rows {
        let node = *compiled
            .get(index as usize)
            .ok_or("captured GPU node index is outside the pinned manifest")?;
        if !seen.insert(node.id) {
            return Err("cut repeats a logical node/root alias".into());
        }
        match flags {
            0 if physical_count == node.representation.count && physical_count > 0 => {}
            OMITTED_OUTSIDE_VIEW if omission_enabled && physical_count == 0 => {}
            _ => {
                return Err(
                    "cut physical count or omission flags differ from its representation/policy"
                        .into(),
                );
            }
        }
        cut.push(CutNode {
            node,
            output_start,
            physical_count,
        });
    }
    // Zero-count logical ranges may share the following physical start. They
    // must precede that physical range when checking a gap-free expansion.
    cut.sort_unstable_by_key(|range| (range.output_start, range.physical_count != 0));
    let mut end = 0_u64;
    for range in &cut {
        if u64::from(range.output_start) != end {
            return Err("cut expanded output has a gap or overlap".into());
        }
        end += u64::from(range.physical_count);
    }
    if end != selected {
        return Err("cut expanded record sum differs from GPU receipt".into());
    }
    let mut source = cut
        .iter()
        .map(|range| range.node.source)
        .collect::<Vec<_>>();
    source.sort_unstable_by_key(|range| range.start);
    let mut end = 0_u64;
    for range in source {
        if range.start != end || range.count == 0 {
            return Err("cut does not cover the whole canonical source exactly once".into());
        }
        end = range.end().ok_or("source range overflow")?;
    }
    if end != manifest.header.source_gaussian_count {
        return Err("cut omits canonical source domains".into());
    }
    Ok(cut)
}

fn matching_matrix(actual: [f64; 16], expected: Mat4) -> bool {
    actual
        .into_iter()
        .zip(expected.to_cols_array())
        .all(|(a, b)| a.is_finite() && (a - f64::from(b)).abs() <= 1e-5 + 2e-7 * f64::from(b.abs()))
}

#[derive(Clone, Copy)]
struct Hit {
    owner: usize,
    output_index: u32,
    sample: ProductionPixelContribution,
}

fn summarize(pixel: [u32; 2], hits: &mut [Hit], cut: &[CutNode<'_>]) -> Value {
    hits.sort_unstable_by(|a, b| {
        b.sample
            .view_depth
            .total_cmp(&a.sample.view_depth)
            .then_with(|| a.output_index.cmp(&b.output_index))
    });
    let mut rgba = [0.0_f32; 4];
    for hit in hits.iter() {
        let alpha = hit.sample.alpha;
        for (channel, color) in hit.sample.color.into_iter().enumerate() {
            rgba[channel] = color * alpha + rgba[channel] * (1.0 - alpha);
        }
        rgba[3] = alpha + rgba[3] * (1.0 - alpha);
    }
    let mut weights = BTreeMap::<usize, (f64, usize)>::new();
    let mut transmittance = 1.0_f64;
    for hit in hits.iter().rev() {
        let entry = weights.entry(hit.owner).or_default();
        entry.0 += transmittance * f64::from(hit.sample.alpha);
        entry.1 += 1;
        transmittance *= 1.0 - f64::from(hit.sample.alpha);
    }
    let mut weights = weights.into_iter().collect::<Vec<_>>();
    weights.sort_unstable_by(|(a, (wa, _)), (b, (wb, _))| {
        wb.total_cmp(wa)
            .then_with(|| cut[*a].node.id.cmp(&cut[*b].node.id))
    });
    let owners = weights.into_iter().map(|(index, (weight, hits))| {
        let node = cut[index].node;
        json!({"node":node.id,"final_alpha_weight":weight,"retained_hits":hits,
            "source":node.source,"source_count":node.source.count,
            "representation_count":node.representation.count,"original_leaf":node.is_leaf(),"depth":node.depth})
    }).collect::<Vec<_>>();
    json!({"pixel":pixel,"linear_premultiplied_rgba":rgba,"transmittance":transmittance,
        "retained_hits":hits.len(),"owners_by_final_alpha_weight":owners})
}

/// Stream one authenticated selected page at a time and rank actual cut owners
/// at up to eight physical pixel centers. No partial report is published.
pub fn attribute_cut(config_path: &Path) -> CaptureResult<()> {
    let started = Instant::now();
    let config_size = fs::metadata(config_path)?.len();
    if config_size > 1024 * 1024 {
        return Err("attribution config exceeds 1 MiB".into());
    }
    let bytes = read_exact_range(config_path, 0, config_size, true)?;
    let mut config: Config = serde_json::from_slice(&bytes)?;
    let base = config_path.parent().unwrap_or(Path::new("."));
    for file in [
        &mut config.manifest,
        &mut config.capture,
        &mut config.receipt,
        &mut config.camera_path,
    ] {
        file.path = fs::canonicalize(base.join(&file.path))?;
    }
    config.output = base.join(&config.output);
    if config.output.exists() {
        return Err("attribution output exists; refusing overwrite".into());
    }
    if config.pixels.is_empty()
        || config.pixels.len() > 8
        || config.pixels.iter().collect::<BTreeSet<_>>().len() != config.pixels.len()
    {
        return Err("attribution requires 1..=8 distinct physical pixel probes".into());
    }
    let record: LodFrameCapture =
        serde_json::from_value(frame_row(&config.capture, config.frame, true)?)?;
    record.validate()?;
    let evidence = frame_row(&config.receipt, config.frame, false)?;
    if record.mode != LodCaptureMode::NativeGpu
        || record.counts.source != LodCountSource::GpuReadback
        || record.counts.pipeline != LodCapturePipeline::HierarchyOrdered
        || record.counts.drawn.is_none()
        || record.counts.transition_extra != 0
        || record.counts.candidates != record.counts.selected
        || record.identity.manifest_sha256 != config.manifest.sha256.to_lowercase()
        || evidence["pipeline"] != "hierarchy_ordered"
        || evidence["source"] != "post_render_same_submission_ordered_draw_and_traversal_copy"
        || evidence["ordered_image_attested"] != true
        || evidence["ordered_draw_receipt"] != true
        || evidence["counts_valid"] != true
        || evidence["traversal"]["complete"] != true
        || evidence["traversal"]["flags"]
            .as_u64()
            .is_none_or(|flags| flags & !63 != 0 || flags & 1 != 0)
        || evidence["ordered"]["flags"] != 0
        || evidence["ordered"]["traversal_failure"] != 0
        || evidence["ordered"]["vertex_count"] != 4
        || evidence["hierarchy_cut"]["valid"] != true
        || evidence["residency_generation"].as_u64() != Some(record.stamp.generation)
        || evidence["traversal"]["selected_gaussians"].as_u64() != Some(record.counts.selected)
        || evidence["ordered"]["projected_gaussians"].as_u64() != record.counts.drawn
        || evidence["ordered_submission"]
            .as_u64()
            .is_none_or(|n| n == 0)
        || evidence["traversal_submission"]
            .as_u64()
            .is_none_or(|n| n == 0)
    {
        return Err(
            "capture is not an attested complete ordered hierarchy image and selected cut".into(),
        );
    }
    let capture_root = config
        .capture
        .path
        .parent()
        .ok_or("capture has no parent")?;
    if config.receipt.path.parent() != Some(capture_root) {
        return Err("capture and receipt must share a run directory".into());
    }
    let settings_bytes = bounded_bytes(&capture_root.join("settings.json"), 1024 * 1024)?;
    if format!("{:x}", Sha256::digest(&settings_bytes)) != record.identity.settings_sha256 {
        return Err("capture settings mismatch or external source metadata is unsupported".into());
    }
    let settings: RuntimeCaptureConfig = serde_json::from_slice(&settings_bytes)?;
    if settings.source_metadata.is_some()
        || settings.render_mode != LodCapturePipeline::HierarchyOrdered
        || !settings.capture_hierarchy_cut
    {
        return Err(
            "attribution requires identity cloud transform and captured ordered cut".into(),
        );
    }
    let image = record
        .image
        .as_ref()
        .ok_or("capture must include its attested image")?;
    let image_path = fs::canonicalize(capture_root.join(&image.path))?;
    if !image_path.starts_with(capture_root) {
        return Err("image escapes capture directory".into());
    }
    verify(
        &PinnedFile {
            path: image_path,
            sha256: image.sha256.clone(),
        },
        MAX_RECEIPT_BYTES,
    )?;
    let path_bytes = pinned_bytes(&config.camera_path, 4 * 1024 * 1024)?;
    let segments = bounded_bytes(&capture_root.join("camera_path.json"), 1024 * 1024)?;
    let mut path_hash = Sha256::new();
    path_hash.update(&segments);
    path_hash.update(&path_bytes);
    if format!("{:x}", path_hash.finalize()) != record.identity.camera_path_sha256 {
        return Err("camera path does not match captured path identity".into());
    }
    let mut path_frame = evidence["path_frame"]
        .as_u64()
        .ok_or("missing captured path frame")?;
    let segment = settings
        .segments
        .iter()
        .find(|segment| {
            if path_frame < u64::from(segment.frames) {
                true
            } else {
                path_frame -= u64::from(segment.frames);
                false
            }
        })
        .ok_or("captured path frame is outside the schedule")?;
    if segment.camera_frame != Some(config.camera_frame) {
        return Err("camera frame differs from captured schedule".into());
    }
    let path = GaussianCameraPath::from_json(&path_bytes)?;
    let frame = path
        .frames()
        .get(config.camera_frame)
        .ok_or("camera frame is outside pinned path")?;
    let camera = LodTestCamera::from_camera_path_frame(
        frame,
        record.camera.viewport,
        settings.near,
        100_000.0,
    )?;
    let projection = frame
        .projection(settings.near, 100_000.0)?
        .get_clip_from_view();
    let view = frame.transform().to_matrix().inverse();
    if !matching_matrix(record.camera.projection, projection)
        || !matching_matrix(record.camera.world_to_view, view)
        || settings.viewport != record.camera.viewport
        || config
            .pixels
            .iter()
            .any(|p| p[0] >= camera.viewport[0] || p[1] >= camera.viewport[1])
    {
        return Err("calibrated camera matrices, viewport or probes differ from capture".into());
    }
    let projector = ProductionPixelProjector::new(
        camera,
        projection,
        Mat4::from_cols_array(&record.camera.world_to_view.map(|v| v as f32)),
    )?;
    let manifest = decode_manifest(
        &pinned_bytes(&config.manifest, limits().max_manifest_bytes)?,
        limits(),
    )?;
    let cut = cut_from_receipt(&manifest, &evidence, record.counts.selected)?;
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
    for (id, ranges) in &pages {
        let descriptor = descriptors
            .get(id)
            .ok_or("cut page is absent from manifest")?;
        let storage = descriptor
            .storage
            .as_ref()
            .ok_or("cut page has no storage")?;
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
            .ok_or("encoded byte overflow")?;
        decoded_bytes = decoded_bytes
            .checked_add(descriptor.decoded_len)
            .ok_or("decoded byte overflow")?;
        if encoded_bytes > MAX_TOTAL_PAGE_BYTES
            || decoded_bytes > MAX_TOTAL_PAGE_BYTES
            || descriptor.decoded_len > MAX_PAGE_BYTES
        {
            return Err(
                "cut exceeds 2 GiB encoded/decoded page work; no prefix attribution".into(),
            );
        }
        let mut physical = ranges
            .iter()
            .map(|&index| {
                let range = cut[index].node.representation;
                let end = range
                    .offset
                    .checked_add(range.count)
                    .ok_or("physical range overflow")?;
                if end > descriptor.gaussian_count {
                    return Err("cut range exceeds decoded page");
                }
                Ok((range.offset, end))
            })
            .collect::<Result<Vec<_>, &str>>()?;
        physical.sort_unstable();
        if physical.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err("cut repeats physical page records".into());
        }
    }
    let root = config
        .manifest
        .path
        .parent()
        .ok_or("manifest has no parent")?;
    let mut hits = (0..config.pixels.len())
        .map(|_| Vec::<Hit>::new())
        .collect::<Vec<_>>();
    let mut samples = [None; 8];
    let mut total_hits = 0;
    let mut visited_records = 0_u64;
    let mut page_hash = Sha256::new();
    for (id, owners) in &pages {
        if started.elapsed().as_secs() >= 180 {
            return Err("attribution exceeds 180 seconds; no partial report".into());
        }
        let descriptor = descriptors[id];
        let storage = descriptor
            .storage
            .as_ref()
            .ok_or("page storage disappeared")?;
        let page_path = fs::canonicalize(root.join(&storage.uri))?;
        if !page_path.starts_with(root) {
            return Err("page symlink escapes package".into());
        }
        let (start, len) = storage.byte_range.unwrap_or((0, storage.encoded_len));
        let encoded = read_exact_range(&page_path, start, len, storage.byte_range.is_none())?;
        page_hash.update(serde_json::to_vec(id)?);
        page_hash.update(Sha256::digest(&encoded));
        let page = decode_page_with_descriptor(&encoded, descriptor, limits())?;
        drop(encoded);
        for &owner in owners {
            let range = cut[owner].node.representation;
            let gaussians =
                &page.gaussians[range.offset as usize..(range.offset + range.count) as usize];
            for (offset, gaussian) in gaussians.iter().enumerate() {
                projector.sample(
                    gaussian,
                    &config.pixels,
                    GaussianColorSpace::SrgbRec709Display,
                    &mut samples[..config.pixels.len()],
                );
                for (pixel, sample) in samples[..config.pixels.len()].iter().enumerate() {
                    if let Some(sample) = sample {
                        if total_hits == MAX_HITS {
                            return Err(
                                "attribution exceeds 100k retained pixel hits; no omitted tail"
                                    .into(),
                            );
                        }
                        hits[pixel].push(Hit {
                            owner,
                            output_index: cut[owner].output_start + offset as u32,
                            sample: *sample,
                        });
                        total_hits += 1;
                    }
                }
            }
            visited_records += u64::from(range.count);
        }
    }
    if visited_records != record.counts.selected {
        return Err("attribution did not visit every selected record".into());
    }
    let probes = config
        .pixels
        .iter()
        .zip(&mut hits)
        .map(|(&pixel, hits)| summarize(pixel, hits, &cut))
        .collect::<Vec<_>>();
    for (file, limit) in [
        (&config.manifest, limits().max_manifest_bytes),
        (&config.capture, MAX_RECEIPT_BYTES),
        (&config.receipt, MAX_RECEIPT_BYTES),
        (&config.camera_path, 4 * 1024 * 1024),
    ] {
        verify(file, limit)?;
    }
    let report = json!({
        "schema_version":1,"kind":"diagnostic_cpu_captured_cut_attribution","config":config,
        "capture_stamp":record.stamp,"capture_identity":record.identity,"camera":record.camera,
        "gpu_image_parity_qualified":false,"quality_certificate":false,"complete_source_antichain_validated":true,
        "ordering":"32-bit forward camera depth (-view-space Z), far-to-near, ties by captured expanded output start plus record offset",
        "projection":"shared CPU production covariance/SH/Mip and fixed three-sigma OBB; identity cloud, default opacity/scale, sRGB SH; six-plane authored world support plus Mip margin",
        "scope":"all selected records visited; nonzero probe support retained without alpha threshold; node source domains are canonical ownership, not exact reducer contributor lineage",
        "selected_nodes":cut.len(),"selected_records":visited_records,
        "omitted_logical_nodes":cut.iter().filter(|range| range.physical_count == 0).count(),
        "physical_omission_scope":"omitted logical nodes preserve complete source ownership and are excluded from pixel attribution; no absent page records are reconstructed","source_records":manifest.header.source_gaussian_count,
        "unique_pages":pages.len(),"encoded_page_bytes_read":encoded_bytes,"decoded_page_bytes":decoded_bytes,
        "page_authentication_digest":format!("{:x}",page_hash.finalize()),
        "page_authentication_digest_layout":"page ID JSON bytes then encoded SHA256 bytes, in ascending page-ID order; each page checked against manifest descriptor",
        "retained_hits":total_hits,"elapsed_seconds":started.elapsed().as_secs_f64(),"probes":probes
    });
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&config.output)?;
    let write = (|| -> CaptureResult<()> {
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &report)?;
        writer.flush()?;
        Ok(())
    })();
    if write.is_err() {
        let _ = fs::remove_file(&config.output);
    }
    write
}

#[cfg(test)]
mod tests {
    use super::super::export_rung::tests::Fixture;
    use super::*;

    #[test]
    fn cut_admission_resolves_root_aliases_and_rejects_missing_or_overlapping_work() {
        let fixture = Fixture::new();
        let manifest = &fixture.manifest;
        assert_eq!(manifest.roots.len(), 1);
        let root = manifest
            .nodes
            .iter()
            .position(|node| node.id == manifest.roots[0])
            .unwrap();
        let count = u64::from(manifest.nodes[root].representation.count);
        let alias = cut_plan(manifest, &[[0, 0, count as u32, 0]], count, false).unwrap();
        let authored = cut_plan(
            manifest,
            &[[1 + root as u32, 0, count as u32, 0]],
            count,
            false,
        )
        .unwrap();
        assert_eq!(alias[0].node.id, authored[0].node.id);
        assert!(
            cut_plan(
                manifest,
                &[
                    [0, 0, count as u32, 0],
                    [1 + root as u32, count as u32, count as u32, 0]
                ],
                count * 2,
                false
            )
            .is_err()
        );
        let mut end = 0;
        let mut leaves = manifest
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.is_leaf())
            .map(|(index, node)| {
                let pair = [index as u32 + 1, end, node.representation.count, 0];
                end += node.representation.count;
                pair
            })
            .collect::<Vec<_>>();
        leaves.reverse(); // GPU acceptance slot order need not equal record order.
        assert!(cut_plan(manifest, &leaves, u64::from(end), false).is_ok());
        assert!(cut_plan(manifest, &leaves[1..], u64::from(end), false).is_err());
        leaves[0][1] += 1;
        assert!(cut_plan(manifest, &leaves, u64::from(end), false).is_err());
        leaves[0][1] -= 1;
        leaves.sort_unstable_by_key(|row| row[1]);
        let omitted = leaves[0][2];
        leaves[0][2] = 0;
        leaves[0][3] = OMITTED_OUTSIDE_VIEW;
        for row in &mut leaves[1..] {
            row[1] -= omitted;
        }
        let physical = u64::from(end - omitted);
        assert!(cut_plan(manifest, &leaves, physical, true).is_ok());
        assert!(cut_plan(manifest, &leaves, physical, false).is_err());
        assert!(
            cut_plan(manifest, &leaves[1..], physical, true).is_err(),
            "physical coverage must not substitute for complete logical source coverage"
        );
        leaves[0][3] = 0;
        assert!(cut_plan(manifest, &leaves, physical, true).is_err());

        let historical = json!({"hierarchy_cut": {"valid":true,"capacity":1,
            "layout":"GPU node index, expanded record output start", "selected":[[0,0]]}});
        assert_eq!(
            cut_from_receipt(manifest, &historical, count).unwrap()[0].physical_count,
            count as u32
        );
        let omitted_receipt = json!({"traversal":{"selected_nodes":1}, "hierarchy_cut":{
            "schema_version":2,"stride_bytes":SELECTED_RANGE_BYTES,"layout":SELECTED_RANGE_LAYOUT,
            "omission_policy":OMISSION_POLICY,"omission_enabled":true,"valid":true,"capacity":1,
            "physical_records":0,"omitted_nodes":1,"selected":[[0,0,0,OMITTED_OUTSIDE_VIEW]]}});
        assert_eq!(
            cut_from_receipt(manifest, &omitted_receipt, 0).unwrap()[0].physical_count,
            0
        );
    }

    #[test]
    fn attribution_uses_captured_record_ties_and_final_composited_alpha() {
        let fixture = Fixture::new();
        let cut = fixture
            .manifest
            .nodes
            .iter()
            .take(2)
            .enumerate()
            .map(|(index, node)| CutNode {
                node,
                output_start: index as u32,
                physical_count: node.representation.count,
            })
            .collect::<Vec<_>>();
        let mut hits = [
            Hit {
                owner: 0,
                output_index: 1,
                sample: ProductionPixelContribution {
                    view_depth: 4.0,
                    color: [0.0, 1.0, 0.0],
                    alpha: 0.5,
                },
            },
            Hit {
                owner: 1,
                output_index: 0,
                sample: ProductionPixelContribution {
                    view_depth: 4.0,
                    color: [1.0, 0.0, 0.0],
                    alpha: 0.5,
                },
            },
        ];
        let summary = summarize([0, 0], &mut hits, &cut);
        assert_eq!(
            summary["linear_premultiplied_rgba"],
            json!([0.25, 0.5, 0.0, 0.75])
        );
        assert_eq!(
            summary["owners_by_final_alpha_weight"][0]["node"],
            json!(cut[0].node.id)
        );
        assert_eq!(
            summary["owners_by_final_alpha_weight"][0]["final_alpha_weight"],
            0.5
        );
        assert_eq!(
            summary["owners_by_final_alpha_weight"][1]["final_alpha_weight"],
            0.25
        );
    }
}
