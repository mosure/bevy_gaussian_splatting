//! CPU-only, policy-independent export of a validated complete authored cut.
//! One encoded/decoded page is resident at a time. The PLY keeps canonical
//! source-node order even when several selected ranges share a packed page.

use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    gaussian::formats::{
        planar_3d::Gaussian3d,
        planar_3d_chunked::{LodBounds, LodNodeId, LodPageId, LodSourceRange},
        planar_3d_lod::GaussianLodManifest,
    },
    io::lod::{LodCodecLimits, decode_manifest, decode_page_with_descriptor},
    stream::transport::{ManifestPageLocation, validate_native_page_location},
};

use super::{CaptureResult, PLY_RECORD_BYTES, hash_file, write_ply_gaussian, write_ply_header};

const MAX_OUTPUT_GAUSSIANS: u64 = 8_000_000;
const MAX_TOTAL_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_PAGE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PAGE_GAUSSIANS: u32 = 65_535;
const MAX_NODES: u32 = 262_144;
const MAX_PAGES: u32 = 65_536;

#[derive(Debug, Serialize)]
struct RungOwner {
    node: LodNodeId,
    depth: u16,
    original_leaf: bool,
    source: LodSourceRange,
    /// Includes all source support and the emitted representative support;
    /// this is conservative geometry, never a measured radiance residual.
    conservative_node_bounds: LodBounds,
    page: LodPageId,
    decoded_page_offset: u32,
    output_start: u64,
    output_count: u32,
    /// Unencoded f32 records, position/visibility, interleaved SH,
    /// rotation, scales/opacity, little-endian; before PLY log conversion.
    decoded_records_sha256: String,
}

struct RungPlan {
    owners: Vec<RungOwner>,
    /// Each unique page maps to owner indices in the output order.
    pages: BTreeMap<LodPageId, Vec<usize>>,
    gaussian_count: u64,
    encoded_bytes: u64,
    decoded_bytes: u64,
}

fn limits() -> LodCodecLimits {
    LodCodecLimits {
        max_manifest_bytes: 64 * 1024 * 1024,
        max_nodes: MAX_NODES,
        max_pages: MAX_PAGES,
        max_page_bytes: MAX_PAGE_BYTES,
        max_page_gaussians: MAX_PAGE_GAUSSIANS,
    }
}

/// Select at the requested absolute depth, retaining any original leaf which
/// ends earlier. A validated manifest proves the tree; contiguous source-range
/// coverage proves this cut has neither holes nor ancestor/descendant overlap.
fn plan_rung(
    manifest: &GaussianLodManifest,
    depth: u16,
    max_gaussians: u64,
) -> CaptureResult<RungPlan> {
    if max_gaussians == 0 || max_gaussians > MAX_OUTPUT_GAUSSIANS {
        return Err("max_gaussians must be in 1..=8000000".into());
    }
    if manifest.header.source_gaussian_count == 0 {
        return Err("cannot export an empty scene as a quality endpoint".into());
    }
    let mut nodes = manifest
        .nodes
        .iter()
        .filter(|node| node.depth == depth || (node.depth < depth && node.is_leaf()))
        .collect::<Vec<_>>();
    nodes.sort_unstable_by_key(|node| (node.source.start, node.id));
    let mut source_end = 0_u64;
    let mut gaussian_count = 0_u64;
    let mut pages = BTreeMap::<LodPageId, Vec<usize>>::new();
    let mut owners = Vec::with_capacity(nodes.len());
    for node in nodes {
        if node.source.start != source_end || node.source.count == 0 {
            return Err("rung has overlapping source ownership or a coverage gap".into());
        }
        source_end = node.source.end().ok_or("source range overflow")?;
        let output_start = gaussian_count;
        gaussian_count = gaussian_count
            .checked_add(u64::from(node.representation.count))
            .ok_or("rung Gaussian count overflow")?;
        if gaussian_count > max_gaussians {
            return Err("authored rung exceeds max_gaussians; no prefix was exported".into());
        }
        pages
            .entry(node.representation.page)
            .or_default()
            .push(owners.len());
        owners.push(RungOwner {
            node: node.id,
            depth: node.depth,
            original_leaf: node.is_leaf(),
            source: node.source,
            conservative_node_bounds: node.bounds,
            page: node.representation.page,
            decoded_page_offset: node.representation.offset,
            output_start,
            output_count: node.representation.count,
            decoded_records_sha256: String::new(),
        });
    }
    if source_end != manifest.header.source_gaussian_count || owners.is_empty() {
        return Err("rung does not cover the complete canonical source".into());
    }
    let descriptors = manifest
        .pages
        .iter()
        .map(|page| (page.id, page))
        .collect::<BTreeMap<_, _>>();
    let mut encoded_bytes = 0_u64;
    let mut decoded_bytes = 0_u64;
    for (id, owner_indices) in &pages {
        let descriptor = descriptors.get(id).ok_or("rung refers to missing page")?;
        let storage = descriptor
            .storage
            .as_ref()
            .ok_or("rung page has no storage location")?;
        let location = ManifestPageLocation {
            uri: storage.uri.clone().into(),
            byte_range: storage.byte_range,
            encoded_len: storage.encoded_len,
        };
        validate_native_page_location(*id, &location, MAX_PAGE_BYTES)?;
        if descriptor.gaussian_count > MAX_PAGE_GAUSSIANS || descriptor.decoded_len > MAX_PAGE_BYTES
        {
            return Err("rung page exceeds decoded working-set admission".into());
        }
        encoded_bytes = encoded_bytes
            .checked_add(storage.encoded_len)
            .ok_or("encoded byte overflow")?;
        decoded_bytes = decoded_bytes
            .checked_add(descriptor.decoded_len)
            .ok_or("decoded byte overflow")?;
        if encoded_bytes > MAX_TOTAL_BYTES || decoded_bytes > MAX_TOTAL_BYTES {
            return Err("rung exceeds 2 GiB total encoded/decoded work admission".into());
        }
        let mut ranges = owner_indices
            .iter()
            .map(|&index| {
                let owner = &owners[index];
                let end = owner
                    .decoded_page_offset
                    .checked_add(owner.output_count)
                    .ok_or("page range overflow")?;
                if end > descriptor.gaussian_count {
                    return Err("rung range exceeds decoded page");
                }
                Ok((owner.decoded_page_offset, end))
            })
            .collect::<Result<Vec<_>, &str>>()?;
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err("rung repeats physical page records".into());
        }
    }
    Ok(RungPlan {
        owners,
        pages,
        gaussian_count,
        encoded_bytes,
        decoded_bytes,
    })
}

pub(super) fn read_exact_range(
    path: &Path,
    start: u64,
    length: u64,
    whole_file: bool,
) -> CaptureResult<Vec<u8>> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    let end = start.checked_add(length).ok_or("file range overflow")?;
    if (whole_file && size != length) || end > size {
        return Err("page/manifest size disagrees with admitted bytes".into());
    }
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(usize::try_from(length)?)?;
    bytes.resize(usize::try_from(length)?, 0);
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

pub(super) fn hash_gaussian(hash: &mut Sha256, gaussian: &Gaussian3d) {
    for value in gaussian
        .position_visibility
        .position
        .into_iter()
        .chain([gaussian.position_visibility.visibility])
        .chain(gaussian.spherical_harmonic.coefficients)
        .chain(gaussian.rotation.rotation)
        .chain(gaussian.scale_opacity.scale)
        .chain([gaussian.scale_opacity.opacity])
    {
        hash.update(value.to_le_bytes());
    }
}

/// Own only files successfully created by this invocation. Failure removes
/// partial results; create_new never permits overwriting an existing artifact.
struct PartialExport {
    paths: Vec<PathBuf>,
    complete: bool,
}

impl Drop for PartialExport {
    fn drop(&mut self) {
        if !self.complete {
            for path in &self.paths {
                let _ = fs::remove_file(path);
            }
        }
    }
}

/// Export a complete authored rung without opening a GPU or running selection.
/// The sibling `.rung.json` records source ownership and authenticated payload
/// lineage. It deliberately does not claim the original input's SHA-256 or
/// radiance quality: neither is contained in the manifest's source fingerprint.
pub fn export_rung(
    manifest_path: &Path,
    output: &Path,
    depth: u16,
    max_gaussians: u64,
) -> CaptureResult<()> {
    let sidecar = output.with_extension("rung.json");
    if output == sidecar || output.exists() || sidecar.exists() {
        return Err("rung output and sidecar must be new distinct paths".into());
    }
    if max_gaussians == 0 || max_gaussians > MAX_OUTPUT_GAUSSIANS {
        return Err("max_gaussians must be in 1..=8000000".into());
    }
    let manifest_path = fs::canonicalize(manifest_path)?;
    let root = manifest_path
        .parent()
        .ok_or("manifest has no parent directory")?;
    let manifest_size = fs::metadata(&manifest_path)?.len();
    if manifest_size > limits().max_manifest_bytes {
        return Err("manifest exceeds 64 MiB admission before allocation".into());
    }
    let manifest_bytes = read_exact_range(&manifest_path, 0, manifest_size, true)?;
    let manifest_sha256 = format!("{:x}", Sha256::digest(&manifest_bytes));
    let manifest = decode_manifest(&manifest_bytes, limits())?;
    drop(manifest_bytes);
    let mut plan = plan_rung(&manifest, depth, max_gaussians)?;
    let descriptors = manifest
        .pages
        .iter()
        .map(|page| (page.id, page))
        .collect::<BTreeMap<_, _>>();
    let mut header = Vec::new();
    write_ply_header(&mut header, plan.gaussian_count)?;
    let output_bytes = plan
        .gaussian_count
        .checked_mul(PLY_RECORD_BYTES)
        .and_then(|bytes| bytes.checked_add(header.len() as u64))
        .ok_or("PLY byte overflow")?;
    if output_bytes > MAX_TOTAL_BYTES {
        return Err("rung PLY exceeds 2 GiB output admission".into());
    }
    // All count/byte/topology checks happen before output reservation or page
    // allocation. Input path containment is checked for each opened page.
    let mut partial = PartialExport {
        paths: Vec::new(),
        complete: false,
    };
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    partial.paths.push(output.to_path_buf());
    let mut metadata_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&sidecar)?;
    partial.paths.push(sidecar.clone());
    let mut writer = BufWriter::new(file);
    writer.write_all(&header)?;
    let mut page_evidence = Vec::with_capacity(plan.pages.len());
    for (page_id, owner_indices) in &plan.pages {
        let descriptor = descriptors[page_id];
        let storage = descriptor
            .storage
            .as_ref()
            .ok_or("page storage disappeared")?;
        let path = fs::canonicalize(root.join(&storage.uri))?;
        if !path.starts_with(root) {
            return Err("page symlink escapes the package directory".into());
        }
        let (start, length) = storage.byte_range.unwrap_or((0, storage.encoded_len));
        let encoded = read_exact_range(&path, start, length, storage.byte_range.is_none())?;
        let encoded_sha256 = format!("{:x}", Sha256::digest(&encoded));
        let page = decode_page_with_descriptor(&encoded, descriptor, limits())?;
        drop(encoded);
        for &index in owner_indices {
            let owner = &mut plan.owners[index];
            let start = owner.decoded_page_offset as usize;
            let end = start + owner.output_count as usize;
            let records = page
                .gaussians
                .get(start..end)
                .ok_or("decoded page range disappeared")?;
            let offset = (header.len() as u64)
                .checked_add(
                    owner
                        .output_start
                        .checked_mul(PLY_RECORD_BYTES)
                        .ok_or("PLY offset overflow")?,
                )
                .ok_or("PLY offset overflow")?;
            writer.seek(SeekFrom::Start(offset))?;
            let mut hash = Sha256::new();
            for gaussian in records {
                hash_gaussian(&mut hash, gaussian);
                write_ply_gaussian(&mut writer, gaussian)?;
            }
            owner.decoded_records_sha256 = format!("{:x}", hash.finalize());
        }
        page_evidence.push(serde_json::json!({
            "page": page_id,
            "storage": storage,
            "encoding": descriptor.encoding,
            "gaussian_count": descriptor.gaussian_count,
            "decoded_len": descriptor.decoded_len,
            "conservative_page_bounds": descriptor.bounds,
            "decoded_versioned_content_hash": format!("{:016x}", descriptor.content_hash),
            "encoded_sha256": encoded_sha256,
            "descriptor_and_container_validated": true,
        }));
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    if writer.get_ref().metadata()?.len() != output_bytes {
        return Err("written PLY length disagrees with the complete rung plan".into());
    }
    let output_sha256 = hash_file(output)?;
    let evidence = serde_json::json!({
        "schema_version": 1,
        "kind": "authored_rung_export",
        "requested_depth": depth,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "manifest_header": manifest.header,
        "builder": manifest.build,
        "original_input_sha256": null,
        "canonical_decoded_source_fingerprint": format!("{:016x}", manifest.build.source_fingerprint),
        "source_gaussian_count": manifest.header.source_gaussian_count,
        "conservative_scene_bounds": manifest.scene_bounds,
        "output_path": output,
        "output_sha256": output_sha256,
        "output_gaussian_count": plan.gaussian_count,
        "output_bytes": output_bytes,
        "output_order": "selected nodes by canonical source range, then page-local record order",
        "complete_source_antichain_validated": true,
        "bounds_are_radiance_certificates": false,
        "originals_retained_for_shallower_leaves": true,
        "decoded_record_hash_layout": "little-endian f32: position[3], visibility, interleaved RGB SH, rotation[4], scale[3], opacity",
        "ply_encoding": "binary little-endian f32, channel-major SH, opacity logit, log scales",
        "ply_scalar_roundtrip_measured": false,
        "required_endpoint_raster_support": "authored 3-sigma; native capture_lod fixes opacity_adaptive_radius=false for both flat and package clouds; ordinary flat adaptive cutoff differs",
        "cloud_transform_and_color_space": "external scene metadata; manifest and PLY do not encode these settings",
        "admission": {
            "max_output_gaussians": max_gaussians,
            "max_manifest_bytes": limits().max_manifest_bytes,
            "max_manifest_nodes": MAX_NODES,
            "max_manifest_pages": MAX_PAGES,
            "max_page_encoded_and_decoded_bytes": MAX_PAGE_BYTES,
            "max_page_gaussians": MAX_PAGE_GAUSSIANS,
            "max_total_encoded_decoded_or_output_bytes": MAX_TOTAL_BYTES,
            "unique_pages": plan.pages.len(),
            "total_encoded_bytes_read": plan.encoded_bytes,
            "total_decoded_page_bytes": plan.decoded_bytes,
            "simultaneously_decoded_pages": 1,
        },
        "owners": plan.owners,
        "pages": page_evidence,
    });
    serde_json::to_writer_pretty(&mut metadata_file, &evidence)?;
    metadata_file.write_all(b"\n")?;
    metadata_file.sync_all()?;
    partial.complete = true;
    println!(
        "exported {} Gaussians from {} complete-cut nodes to {}",
        plan.gaussian_count,
        evidence["owners"].as_array().unwrap().len(),
        output.display()
    );
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        GaussianLodBuildSettings, PlanarGaussian3d, build_planar_3d_lod,
        gaussian::formats::planar_3d_chunked::LodPageStorage,
        io::{
            lod::{encode_manifest, encode_page},
            ply::{PlyShCompatibility, stream_ply_3d_with_sh_compatibility},
        },
        testing::LodTestScene,
    };
    use bevy_interleave::prelude::Planar;
    use std::{
        io::BufReader,
        sync::atomic::{AtomicU64, Ordering},
    };

    pub(crate) struct Fixture {
        pub(crate) root: PathBuf,
        pub(crate) manifest: GaussianLodManifest,
    }
    impl Fixture {
        pub(crate) fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "bgs-rung-export-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            let cloud = PlanarGaussian3d::from_interleaved(
                LodTestScene::nested_octants(2)
                    .cloud()
                    .iter()
                    .take(17)
                    .collect(),
            );
            let mut lod = build_planar_3d_lod(
                &cloud,
                GaussianLodBuildSettings {
                    branching_factor: 4,
                    leaf_capacity: 4,
                    support_sigma: 3.0,
                },
            )
            .unwrap();
            let mut packed = vec![0_u8; 16];
            for page in &lod.pages {
                let encoded = encode_page(page).unwrap();
                let start = packed.len() as u64;
                let len = encoded.len() as u64;
                packed.extend(encoded);
                lod.manifest
                    .pages
                    .iter_mut()
                    .find(|descriptor| descriptor.id == page.id)
                    .unwrap()
                    .storage = Some(LodPageStorage {
                    uri: "packed-pages.bin".to_owned(),
                    byte_range: Some((start, len)),
                    encoded_len: len,
                });
            }
            fs::write(root.join("packed-pages.bin"), packed).unwrap();
            fs::write(
                root.join("scene.gsplatlod"),
                encode_manifest(&lod.manifest).unwrap(),
            )
            .unwrap();
            Self {
                root,
                manifest: lod.manifest,
            }
        }
        fn export(&self, name: &str, depth: u16, max: u64) -> CaptureResult<()> {
            export_rung(
                &self.root.join("scene.gsplatlod"),
                &self.root.join(name),
                depth,
                max,
            )
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn rung_export_preserves_complete_source_coverage_and_shallower_original_leaves() {
        let fixture = Fixture::new();
        let depth = fixture.manifest.quality.max_depth;
        assert!(
            fixture
                .manifest
                .nodes
                .iter()
                .any(|node| node.is_leaf() && node.depth < depth),
            "uneven fixture must contain a shallow original leaf"
        );
        fixture.export("leaves.ply", depth, 17).unwrap();
        let sidecar: serde_json::Value =
            serde_json::from_slice(&fs::read(fixture.root.join("leaves.rung.json")).unwrap())
                .unwrap();
        assert_eq!(sidecar["output_gaussian_count"], 17);
        assert_eq!(sidecar["source_gaussian_count"], 17);
        assert_eq!(sidecar["complete_source_antichain_validated"], true);
        assert!(sidecar["original_input_sha256"].is_null());
        assert_eq!(
            sidecar["output_sha256"],
            hash_file(&fixture.root.join("leaves.ply")).unwrap()
        );
        let mut source_end = 0;
        let mut output_end = 0;
        for owner in sidecar["owners"].as_array().unwrap() {
            assert_eq!(owner["source"]["start"].as_u64().unwrap(), source_end);
            assert_eq!(owner["output_start"].as_u64().unwrap(), output_end);
            assert_eq!(owner["original_leaf"], true);
            assert_eq!(owner["decoded_records_sha256"].as_str().unwrap().len(), 64);
            source_end += owner["source"]["count"].as_u64().unwrap();
            output_end += owner["output_count"].as_u64().unwrap();
        }
        assert_eq!((source_end, output_end), (17, 17));
        let mut count = 0;
        stream_ply_3d_with_sh_compatibility(
            &mut BufReader::new(File::open(fixture.root.join("leaves.ply")).unwrap()),
            4,
            PlyShCompatibility::RequireRepresentable,
            |batch| {
                count += batch.len();
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(count, 17);
        // Asking beyond maximum depth must still export the same complete
        // original cut, rather than silently dropping shallower leaves.
        fixture.export("beyond.ply", depth + 1, 17).unwrap();
        assert_eq!(
            fs::read(fixture.root.join("leaves.ply")).unwrap(),
            fs::read(fixture.root.join("beyond.ply")).unwrap()
        );
    }

    #[test]
    fn rung_export_rejects_budget_overwrite_and_corrupt_pages_without_partial_artifacts() {
        let fixture = Fixture::new();
        assert!(fixture.export("limited.ply", u16::MAX, 16).is_err());
        assert!(!fixture.root.join("limited.ply").exists());
        assert!(!fixture.root.join("limited.rung.json").exists());
        fs::write(fixture.root.join("existing.ply"), b"keep").unwrap();
        assert!(fixture.export("existing.ply", 0, 17).is_err());
        assert_eq!(
            fs::read(fixture.root.join("existing.ply")).unwrap(),
            b"keep"
        );
        fs::write(fixture.root.join("existing-sidecar.rung.json"), b"keep").unwrap();
        assert!(fixture.export("existing-sidecar.ply", 0, 17).is_err());
        assert!(!fixture.root.join("existing-sidecar.ply").exists());
        let plan = plan_rung(&fixture.manifest, 0, 17).unwrap();
        let page = *plan.pages.keys().next().unwrap();
        let storage = fixture
            .manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == page)
            .unwrap()
            .storage
            .as_ref()
            .unwrap();
        let (start, length) = storage.byte_range.unwrap();
        let packed_path = fixture.root.join("packed-pages.bin");
        let mut bytes = fs::read(&packed_path).unwrap();
        bytes[(start + length - 1) as usize] ^= 1;
        fs::write(packed_path, bytes).unwrap();
        assert!(fixture.export("corrupt.ply", 0, 17).is_err());
        assert!(!fixture.root.join("corrupt.ply").exists());
        assert!(!fixture.root.join("corrupt.rung.json").exists());
    }

    #[test]
    fn rung_export_preflights_source_and_page_ranges_before_allocating_payloads() {
        let fixture = Fixture::new();
        let mut manifest = fixture.manifest.clone();
        let root = manifest.roots[0];
        manifest
            .nodes
            .iter_mut()
            .find(|node| node.id == root)
            .unwrap()
            .source
            .start = 1;
        assert!(plan_rung(&manifest, 0, 17).is_err());
        let mut manifest = fixture.manifest.clone();
        let page = plan_rung(&manifest, 0, 17).unwrap().owners[0].page;
        let descriptor = manifest
            .pages
            .iter_mut()
            .find(|descriptor| descriptor.id == page)
            .unwrap();
        descriptor.storage.as_mut().unwrap().encoded_len = MAX_PAGE_BYTES + 1;
        descriptor.storage.as_mut().unwrap().byte_range = None;
        assert!(plan_rung(&manifest, 0, 17).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn rung_export_rejects_symlinks_outside_the_package_root() {
        let fixture = Fixture::new();
        let other = Fixture::new();
        let packed = fixture.root.join("packed-pages.bin");
        fs::remove_file(&packed).unwrap();
        std::os::unix::fs::symlink(other.root.join("packed-pages.bin"), &packed).unwrap();
        assert!(fixture.export("escaped.ply", 0, 17).is_err());
        assert!(!fixture.root.join("escaped.ply").exists());
        assert!(!fixture.root.join("escaped.rung.json").exists());
    }
}
