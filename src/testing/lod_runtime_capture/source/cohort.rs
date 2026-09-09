//! Authenticated partial-domain exports for bounded local representation work.
//!
//! Node source intervals address canonical decoded records, never raw PLY rows.
//! Seed and immutable context domains must be disjoint. Original records come
//! from a complete original-leaf cover of each requested domain; no prefix or
//! projected-center filter can silently change that ownership.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    gaussian::{
        formats::{
            planar_3d::Gaussian3d,
            planar_3d_chunked::{LodBounds, LodNodeId, LodPageId, LodSourceRange},
            planar_3d_lod::{GaussianLodManifest, GaussianLodNode},
        },
        settings::GaussianColorSpace,
    },
    io::lod::{LodCodecLimits, decode_manifest, decode_page_with_descriptor},
    stream::transport::{ManifestPageLocation, validate_native_page_location},
};

use super::{
    CaptureResult, PLY_RECORD_BYTES,
    export_rung::{hash_gaussian, read_exact_range},
    hash_file, write_ply_gaussian, write_ply_header,
};

mod context;

const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_SIDECAR_BYTES: u64 = 32 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 256 * 1024 * 1024;
const MAX_MANIFEST_ITEMS: u32 = 262_144;
const MAX_PAGE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PAGE_GAUSSIANS: u32 = 65_535;
const MAX_TOTAL_PAGE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_RANGES: usize = 32_768;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PinnedFile {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct CohortAdmission {
    pub max_manifest_bytes: u64,
    pub max_manifest_nodes: u32,
    pub max_manifest_pages: u32,
    pub max_source_gaussians: u64,
    pub max_candidate_gaussians: u64,
}

impl Default for CohortAdmission {
    fn default() -> Self {
        Self {
            max_manifest_bytes: MAX_MANIFEST_BYTES,
            max_manifest_nodes: MAX_MANIFEST_ITEMS,
            max_manifest_pages: MAX_MANIFEST_ITEMS,
            max_source_gaussians: 100_000,
            max_candidate_gaussians: 30_000,
        }
    }
}

impl CohortAdmission {
    fn limits(self) -> CaptureResult<LodCodecLimits> {
        if !(1..=MAX_MANIFEST_BYTES).contains(&self.max_manifest_bytes)
            || !(1..=MAX_MANIFEST_ITEMS).contains(&self.max_manifest_nodes)
            || !(1..=MAX_MANIFEST_ITEMS).contains(&self.max_manifest_pages)
            || !(1..=100_000).contains(&self.max_source_gaussians)
            || !(1..=30_000).contains(&self.max_candidate_gaussians)
        {
            return Err("cohort admission exceeds bounded metadata/source/candidate limits".into());
        }
        Ok(LodCodecLimits {
            max_manifest_bytes: self.max_manifest_bytes,
            max_nodes: self.max_manifest_nodes,
            max_pages: self.max_manifest_pages,
            max_page_bytes: MAX_PAGE_BYTES,
            max_page_gaussians: MAX_PAGE_GAUSSIANS,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CohortExportConfig {
    pub manifest: PinnedFile,
    pub output_directory: PathBuf,
    pub owned_nodes: Vec<LodNodeId>,
    #[serde(default)]
    pub context_nodes: Vec<LodNodeId>,
    #[serde(default)]
    pub captured_context: Option<context::CapturedContext>,
    #[serde(default)]
    pub include_original_context: bool,
    #[serde(default)]
    pub admission: CohortAdmission,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CohortRole {
    Source,
    Seed,
    Context,
    OriginalContext,
}

impl CohortRole {
    fn filename(self) -> &'static str {
        match self {
            Self::Source => "source.ply",
            Self::Seed => "seed.ply",
            Self::Context => "context.ply",
            Self::OriginalContext => "original-context.ply",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CohortRange {
    /// Requested owner; for source payloads, `node` is an original leaf below it.
    pub owner: LodNodeId,
    pub node: LodNodeId,
    pub source: LodSourceRange,
    pub conservative_node_bounds: LodBounds,
    pub page: LodPageId,
    pub decoded_page_offset: u32,
    pub output_start: u64,
    pub output_count: u32,
    /// Before the PLY log/logit roundtrip. This is not a raw-source byte hash.
    pub decoded_records_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CohortPayload {
    pub role: CohortRole,
    pub file: PinnedFile,
    pub gaussian_count: u64,
    pub bytes: u64,
    pub ranges: Vec<CohortRange>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PageEvidence {
    page: LodPageId,
    encoded_sha256: String,
    decoded_versioned_content_hash: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CohortSidecar {
    pub schema_version: u32,
    pub kind: String,
    pub config: CohortExportConfig,
    pub full_source_gaussian_count: u64,
    pub canonical_decoded_source_fingerprint: String,
    pub partial_domains_validated: bool,
    pub complete_source_antichain_validated: bool,
    pub original_input_sha256: Option<String>,
    pub raw_source_ordinal_mapping_verified: bool,
    pub decoded_record_hash_layout: String,
    pub output_order: String,
    #[serde(default)]
    pub context_selection: Option<context::ContextEvidence>,
    pub payloads: Vec<CohortPayload>,
    pages: Vec<PageEvidence>,
    pub encoded_page_bytes_read: u64,
    pub decoded_page_bytes: u64,
    pub simultaneously_decoded_pages: u32,
}

pub(crate) struct LoadedCohort {
    pub source: Vec<Gaussian3d>,
    pub initial: Vec<Gaussian3d>,
    pub context: Vec<Gaussian3d>,
    pub domains: Vec<LodBounds>,
    pub source_order: Option<Vec<u64>>,
    pub initial_order: Option<Vec<u64>>,
    pub context_order: Option<Vec<u64>>,
    /// Authored page color convention, matching the exported PLY. External scene
    /// transforms and alternative color encodings require a separate profile.
    pub color_space: GaussianColorSpace,
    pub sidecar: CohortSidecar,
    sidecar_pin: PinnedFile,
}

impl LoadedCohort {
    pub fn validate_views(
        &self,
        views: &[crate::testing::render_oracle::fit::FitView],
    ) -> CaptureResult<()> {
        if let Some(profile) = &self.sidecar.config.captured_context {
            context::validate_views(profile, views)?;
        }
        Ok(())
    }
    /// Recheck the inputs whose decoded values were used during optimization.
    /// Page lineage was verified on import; the pinned manifest and receipt
    /// retain those exact page hashes even if the source package is later moved.
    pub fn verify_inputs(&self) -> CaptureResult<()> {
        for (file, limit) in [
            (&self.sidecar_pin, MAX_SIDECAR_BYTES),
            (
                &self.sidecar.config.manifest,
                self.sidecar.config.admission.max_manifest_bytes,
            ),
        ] {
            if fs::metadata(&file.path)?.len() > limit
                || !hash_matches(&hash_file(&file.path)?, &file.sha256)
            {
                return Err("cohort metadata changed during fitting".into());
            }
        }
        for payload in &self.sidecar.payloads {
            if fs::metadata(&payload.file.path)?.len() != payload.bytes
                || !hash_matches(&hash_file(&payload.file.path)?, &payload.file.sha256)
            {
                return Err("cohort payload changed during fitting".into());
            }
        }
        Ok(())
    }
}

struct PayloadPlan {
    role: CohortRole,
    count: u64,
    ranges: Vec<CohortRange>,
    header: Vec<u8>,
    bytes: u64,
}

struct Plan {
    payloads: Vec<PayloadPlan>,
    /// All output roles share a single physical-page read/decode.
    pages: BTreeMap<LodPageId, Vec<(usize, usize)>>,
    encoded_bytes: u64,
    decoded_bytes: u64,
    context_selection: Option<context::ContextEvidence>,
}

struct SelectedNodes<'a> {
    owned: Vec<&'a GaussianLodNode>,
    context: Vec<&'a GaussianLodNode>,
}

fn hash_matches(actual: &str, expected: &str) -> bool {
    expected.len() == 64
        && expected.bytes().all(|v| v.is_ascii_hexdigit())
        && actual.eq_ignore_ascii_case(expected)
}

fn read_pinned(file: &PinnedFile, max_bytes: u64) -> CaptureResult<Vec<u8>> {
    if file.sha256.len() != 64 || !file.sha256.bytes().all(|value| value.is_ascii_hexdigit()) {
        return Err("cohort input requires a SHA256 identity".into());
    }
    let bytes = fs::metadata(&file.path)?.len();
    if bytes > max_bytes {
        return Err(format!(
            "cohort input exceeds {max_bytes} bytes: {}",
            file.path.display()
        )
        .into());
    }
    let bytes = read_exact_range(&file.path, 0, bytes, true)?;
    if !hash_matches(&format!("{:x}", Sha256::digest(&bytes)), &file.sha256) {
        return Err(format!("cohort SHA256 mismatch: {}", file.path.display()).into());
    }
    Ok(bytes)
}

fn manifest(config: &CohortExportConfig) -> CaptureResult<GaussianLodManifest> {
    let limits = config.admission.limits()?;
    decode_manifest(
        &read_pinned(&config.manifest, limits.max_manifest_bytes)?,
        limits,
    )
    .map_err(Into::into)
}

fn selected_nodes<'a>(
    manifest: &'a GaussianLodManifest,
    config: &CohortExportConfig,
) -> CaptureResult<SelectedNodes<'a>> {
    if config.owned_nodes.is_empty()
        || config
            .owned_nodes
            .len()
            .saturating_add(config.context_nodes.len())
            > MAX_RANGES
    {
        return Err("cohort requires owned nodes and at most 32768 selected nodes".into());
    }
    let nodes = manifest
        .nodes
        .iter()
        .map(|node| (node.id, node))
        .collect::<BTreeMap<_, _>>();
    let mut ids = BTreeSet::new();
    let mut select = |requested: &[LodNodeId]| -> CaptureResult<Vec<&'a GaussianLodNode>> {
        let mut result = Vec::with_capacity(requested.len());
        for id in requested {
            if !ids.insert(*id) {
                return Err("cohort owned/context node IDs are repeated".into());
            }
            result.push(*nodes.get(id).ok_or("cohort node is absent from manifest")?);
        }
        result.sort_unstable_by_key(|node| (node.source.start, node.id));
        Ok(result)
    };
    let owned = select(&config.owned_nodes)?;
    let context = select(&config.context_nodes)?;
    let mut combined = owned.iter().chain(&context).copied().collect::<Vec<_>>();
    combined.sort_unstable_by_key(|node| node.source.start);
    let mut previous_end = 0;
    for node in combined {
        let end = node.source.end().ok_or("cohort source interval overflow")?;
        if node.source.count == 0 || node.source.start < previous_end {
            return Err(
                "cohort owned/context domains overlap (including ancestor/descendant overlap)"
                    .into(),
            );
        }
        previous_end = end;
    }
    Ok(SelectedNodes { owned, context })
}

fn payload_plan(
    role: CohortRole,
    owners: &[&GaussianLodNode],
    leaves: &[&GaussianLodNode],
) -> CaptureResult<PayloadPlan> {
    let originals = matches!(role, CohortRole::Source | CohortRole::OriginalContext);
    let mut ranges = Vec::new();
    let mut count = 0_u64;
    for owner in owners {
        let owned_end = owner.source.end().ok_or("owner source interval overflow")?;
        let mut add = |node: &GaussianLodNode| -> CaptureResult<()> {
            if ranges.len() == MAX_RANGES {
                return Err("cohort exceeds bounded payload range count".into());
            }
            ranges.push(CohortRange {
                owner: owner.id,
                node: node.id,
                source: node.source,
                conservative_node_bounds: node.bounds,
                page: node.representation.page,
                decoded_page_offset: node.representation.offset,
                output_start: count,
                output_count: node.representation.count,
                decoded_records_sha256: String::new(),
            });
            count = count
                .checked_add(u64::from(node.representation.count))
                .ok_or("cohort count overflow")?;
            Ok(())
        };
        if originals {
            let first = leaves.partition_point(|node| node.source.start < owner.source.start);
            let mut end = owner.source.start;
            for node in &leaves[first..] {
                if node.source.start >= owned_end {
                    break;
                }
                if node.source.start != end
                    || node.source.count != u64::from(node.representation.count)
                {
                    return Err("original leaves do not exactly cover cohort domain".into());
                }
                end = node.source.end().ok_or("leaf source interval overflow")?;
                if end > owned_end {
                    return Err("original leaf crosses cohort ownership boundary".into());
                }
                add(node)?;
            }
            if end != owned_end {
                return Err("original leaf cover is incomplete; no prefix was exported".into());
            }
        } else {
            add(owner)?;
        }
    }
    let mut header = Vec::new();
    write_ply_header(&mut header, count)?;
    let bytes = count
        .checked_mul(PLY_RECORD_BYTES)
        .and_then(|bytes| bytes.checked_add(header.len() as u64))
        .ok_or("cohort output byte overflow")?;
    Ok(PayloadPlan {
        role,
        count,
        ranges,
        header,
        bytes,
    })
}

fn plan(manifest: &GaussianLodManifest, config: &CohortExportConfig) -> CaptureResult<Plan> {
    config.admission.limits()?;
    let SelectedNodes { owned, context } = selected_nodes(manifest, config)?;
    let counts = |nodes: &[&GaussianLodNode]| -> CaptureResult<(u64, u64)> {
        nodes
            .iter()
            .try_fold((0_u64, 0_u64), |(source, seed), node| {
                Ok((
                    source
                        .checked_add(node.source.count)
                        .ok_or("source count overflow")?,
                    seed.checked_add(u64::from(node.representation.count))
                        .ok_or("seed count overflow")?,
                ))
            })
    };
    let (owned_source, owned_seed) = counts(&owned)?;
    let context_selection = if config.captured_context.is_some() {
        let capacity = config
            .admission
            .max_source_gaussians
            .checked_sub(owned_source)
            .zip(
                config
                    .admission
                    .max_candidate_gaussians
                    .checked_sub(owned_seed),
            )
            .map(|(source, seed)| source.min(seed))
            .ok_or("complete owned domains exceed fit admission")?;
        Some(context::scan(manifest, config, &owned, capacity)?)
    } else {
        None
    };
    let (context_source, context_seed) = counts(&context)?;
    let maximum_context = if config.include_original_context {
        context_source.max(context_seed)
    } else {
        context_seed
    };
    if owned_source
        .checked_add(maximum_context)
        .ok_or("teacher count overflow")?
        > config.admission.max_source_gaussians
        || owned_seed
            .checked_add(maximum_context)
            .ok_or("candidate count overflow")?
            > config.admission.max_candidate_gaussians
    {
        return Err(
            "complete cohort plus context exceeds source/candidate cap; no prefix was exported"
                .into(),
        );
    }
    let mut leaves = manifest
        .nodes
        .iter()
        .filter(|node| node.is_leaf())
        .collect::<Vec<_>>();
    leaves.sort_unstable_by_key(|node| node.source.start);
    let mut payloads = vec![
        payload_plan(CohortRole::Source, &owned, &leaves)?,
        payload_plan(CohortRole::Seed, &owned, &leaves)?,
        match &context_selection {
            Some(evidence) => context::payload(manifest, evidence)?,
            None => payload_plan(CohortRole::Context, &context, &leaves)?,
        },
    ];
    if config.include_original_context {
        payloads.push(payload_plan(
            CohortRole::OriginalContext,
            &context,
            &leaves,
        )?);
    }
    if payloads
        .iter()
        .map(|payload| payload.ranges.len())
        .sum::<usize>()
        > MAX_RANGES
    {
        return Err("cohort exceeds 32768 total payload ranges".into());
    }
    // Both frozen contexts, when requested, must independently fit the same
    // complete teacher/candidate contract. The optional context is never free.
    for context in &payloads[2..] {
        if payloads[0]
            .count
            .checked_add(context.count)
            .ok_or("teacher count overflow")?
            > config.admission.max_source_gaussians
            || payloads[1]
                .count
                .checked_add(context.count)
                .ok_or("candidate count overflow")?
                > config.admission.max_candidate_gaussians
        {
            return Err(
                "complete cohort plus context exceeds source/candidate cap; no prefix was exported"
                    .into(),
            );
        }
    }
    let descriptors = manifest
        .pages
        .iter()
        .map(|page| (page.id, page))
        .collect::<BTreeMap<_, _>>();
    let mut pages = BTreeMap::<_, Vec<_>>::new();
    for (payload_index, payload) in payloads.iter().enumerate() {
        for (range_index, range) in payload.ranges.iter().enumerate() {
            pages
                .entry(range.page)
                .or_default()
                .push((payload_index, range_index));
        }
    }
    let mut encoded_bytes = 0_u64;
    let mut decoded_bytes = 0_u64;
    for (id, uses) in &pages {
        let descriptor = descriptors
            .get(id)
            .ok_or("cohort page missing from manifest")?;
        let storage = descriptor
            .storage
            .as_ref()
            .ok_or("cohort page has no location")?;
        validate_native_page_location(
            *id,
            &ManifestPageLocation {
                uri: storage.uri.clone().into(),
                byte_range: storage.byte_range,
                encoded_len: storage.encoded_len,
            },
            MAX_PAGE_BYTES,
        )?;
        if descriptor.gaussian_count > MAX_PAGE_GAUSSIANS || descriptor.decoded_len > MAX_PAGE_BYTES
        {
            return Err("cohort page exceeds decoded working-set admission".into());
        }
        encoded_bytes = encoded_bytes
            .checked_add(storage.encoded_len)
            .ok_or("encoded byte overflow")?;
        decoded_bytes = decoded_bytes
            .checked_add(descriptor.decoded_len)
            .ok_or("decoded byte overflow")?;
        if encoded_bytes > MAX_TOTAL_PAGE_BYTES || decoded_bytes > MAX_TOTAL_PAGE_BYTES {
            return Err("cohort exceeds 512 MiB aggregate page work admission".into());
        }
        let mut physical = Vec::with_capacity(uses.len());
        for &(payload, range) in uses {
            let range = &payloads[payload].ranges[range];
            let end = range
                .decoded_page_offset
                .checked_add(range.output_count)
                .ok_or("page range overflow")?;
            if end > descriptor.gaussian_count {
                return Err("cohort range exceeds decoded page".into());
            }
            physical.push((payload, range.decoded_page_offset, end));
        }
        physical.sort_unstable();
        if physical
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0 && pair[0].2 > pair[1].1)
        {
            return Err("cohort repeats physical records within an output payload".into());
        }
    }
    Ok(Plan {
        payloads,
        pages,
        encoded_bytes,
        decoded_bytes,
        context_selection,
    })
}

fn visit_pages(
    manifest: &GaussianLodManifest,
    config: &CohortExportConfig,
    plan: &mut Plan,
    mut visit: impl FnMut(usize, &CohortRange, &[Gaussian3d]) -> CaptureResult<()>,
) -> CaptureResult<Vec<PageEvidence>> {
    let root = config
        .manifest
        .path
        .parent()
        .ok_or("manifest has no parent directory")?;
    let descriptors = manifest
        .pages
        .iter()
        .map(|page| (page.id, page))
        .collect::<BTreeMap<_, _>>();
    let mut evidence = Vec::with_capacity(plan.pages.len());
    for (id, uses) in &plan.pages {
        let descriptor = descriptors[id];
        let storage = descriptor
            .storage
            .as_ref()
            .ok_or("page location disappeared")?;
        let path = fs::canonicalize(root.join(&storage.uri))?;
        if !path.starts_with(root) {
            return Err("cohort page symlink escapes package directory".into());
        }
        let (start, len) = storage.byte_range.unwrap_or((0, storage.encoded_len));
        let encoded = read_exact_range(&path, start, len, storage.byte_range.is_none())?;
        let encoded_sha256 = format!("{:x}", Sha256::digest(&encoded));
        let page = decode_page_with_descriptor(&encoded, descriptor, config.admission.limits()?)?;
        drop(encoded);
        for &(payload, range_index) in uses {
            let range = &mut plan.payloads[payload].ranges[range_index];
            let start = range.decoded_page_offset as usize;
            let records = page
                .gaussians
                .get(start..start + range.output_count as usize)
                .ok_or("decoded cohort range disappeared")?;
            let mut hash = Sha256::new();
            for gaussian in records {
                hash_gaussian(&mut hash, gaussian);
            }
            range.decoded_records_sha256 = format!("{:x}", hash.finalize());
            visit(payload, range, records)?;
        }
        evidence.push(PageEvidence {
            page: *id,
            encoded_sha256,
            decoded_versioned_content_hash: descriptor.content_hash,
        });
    }
    Ok(evidence)
}

struct PartialDirectory {
    root: PathBuf,
    files: Vec<PathBuf>,
    complete: bool,
}

impl PartialDirectory {
    fn create_file(&mut self, name: &str) -> CaptureResult<File> {
        let path = self.root.join(name);
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        self.files.push(path);
        Ok(file)
    }
}

impl Drop for PartialDirectory {
    fn drop(&mut self) {
        if !self.complete {
            for path in &self.files {
                let _ = fs::remove_file(path);
            }
            // Do not delete unrelated files if another process used this directory.
            let _ = fs::remove_dir(&self.root);
        }
    }
}

fn offset(header: usize, range: &CohortRange) -> CaptureResult<u64> {
    range
        .output_start
        .checked_mul(PLY_RECORD_BYTES)
        .and_then(|value| value.checked_add(header as u64))
        .ok_or_else(|| "cohort PLY offset overflow".into())
}

/// Export the explicitly selected partial domains; all count/page admission
/// precedes output creation. `cohort.json` is the typed, hash-pinnable receipt.
pub fn export_cohort(config_path: &Path) -> CaptureResult<()> {
    let size = fs::metadata(config_path)?.len();
    if size > MAX_CONFIG_BYTES {
        return Err("cohort config exceeds 1 MiB".into());
    }
    let mut config: CohortExportConfig =
        serde_json::from_slice(&read_exact_range(config_path, 0, size, true)?)?;
    config.manifest.path = fs::canonicalize(&config.manifest.path)?;
    let manifest = manifest(&config)?;
    let mut plan = plan(&manifest, &config)?;
    fs::create_dir(&config.output_directory)?;
    let mut partial = PartialDirectory {
        root: config.output_directory.clone(),
        files: Vec::new(),
        complete: false,
    };
    config.output_directory = fs::canonicalize(&config.output_directory)?;
    partial.root = config.output_directory.clone();
    let mut writers = Vec::with_capacity(plan.payloads.len());
    for payload in &plan.payloads {
        let mut writer = BufWriter::new(partial.create_file(payload.role.filename())?);
        writer.write_all(&payload.header)?;
        writers.push(writer);
    }
    let header_lengths = plan
        .payloads
        .iter()
        .map(|payload| payload.header.len())
        .collect::<Vec<_>>();
    let pages = visit_pages(&manifest, &config, &mut plan, |payload, range, records| {
        let writer = &mut writers[payload];
        writer.seek(SeekFrom::Start(offset(header_lengths[payload], range)?))?;
        for gaussian in records {
            write_ply_gaussian(writer, gaussian)?;
        }
        Ok(())
    })?;
    let mut payloads = Vec::with_capacity(plan.payloads.len());
    for (payload, writer) in plan.payloads.iter_mut().zip(&mut writers) {
        writer.flush()?;
        writer.get_ref().sync_all()?;
        if writer.get_ref().metadata()?.len() != payload.bytes {
            return Err("cohort PLY length disagrees with complete export plan".into());
        }
        let path = config.output_directory.join(payload.role.filename());
        payloads.push(CohortPayload {
            role: payload.role,
            file: PinnedFile {
                sha256: hash_file(&path)?,
                path,
            },
            gaussian_count: payload.count,
            bytes: payload.bytes,
            ranges: std::mem::take(&mut payload.ranges),
        });
    }
    let sidecar = CohortSidecar {
        schema_version: 1,
        kind: "authored_cohort_export".into(),
        config,
        full_source_gaussian_count: manifest.header.source_gaussian_count,
        canonical_decoded_source_fingerprint: format!("{:016x}", manifest.build.source_fingerprint),
        partial_domains_validated: true,
        complete_source_antichain_validated: false,
        original_input_sha256: None,
        raw_source_ordinal_mapping_verified: false,
        decoded_record_hash_layout: "little-endian f32: position[3], visibility, interleaved RGB SH, rotation[4], scale[3], opacity; before PLY log/logit conversion".into(),
        output_order: "owned and manual context roles: canonical source order then page-local records; captured context: captured output order; explicit renderer tie keys retained when present".into(),
        context_selection: plan.context_selection.take(),
        payloads,
        pages,
        encoded_page_bytes_read: plan.encoded_bytes,
        decoded_page_bytes: plan.decoded_bytes,
        simultaneously_decoded_pages: 1,
    };
    let bytes = serde_json::to_vec_pretty(&sidecar)?;
    if bytes.len() as u64 >= MAX_SIDECAR_BYTES {
        return Err("cohort sidecar exceeds 32 MiB".into());
    }
    let mut file = partial.create_file("cohort.json")?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    partial.complete = true;
    println!(
        "exported {} owned originals, {} seed records, {} frozen context records to {}",
        sidecar.payloads[0].gaussian_count,
        sidecar.payloads[1].gaussian_count,
        sidecar.payloads[2].gaussian_count,
        sidecar.config.output_directory.display()
    );
    Ok(())
}

/// Import authenticates the manifest and pages again, checks exact PLY bytes
/// against each decoded source range, then retains the authenticated native
/// records. PLY is an export artifact: its log-scale clamp and quaternion
/// normalization must not change the package geometry used for supervision.
/// A pinned sidecar alone is never treated as proof of canonical source lineage.
pub(crate) fn load_cohort(
    sidecar_path: &Path,
    expected_sha256: &str,
    original_context: bool,
) -> CaptureResult<LoadedCohort> {
    let sidecar_pin = PinnedFile {
        path: sidecar_path.into(),
        sha256: expected_sha256.into(),
    };
    let sidecar: CohortSidecar =
        serde_json::from_slice(&read_pinned(&sidecar_pin, MAX_SIDECAR_BYTES)?)?;
    if sidecar.schema_version != 1
        || sidecar.kind != "authored_cohort_export"
        || !sidecar.partial_domains_validated
        || sidecar.complete_source_antichain_validated
        || sidecar.original_input_sha256.is_some()
        || sidecar.raw_source_ordinal_mapping_verified
        || sidecar.simultaneously_decoded_pages != 1
        || (original_context && !sidecar.config.include_original_context)
        || fs::canonicalize(&sidecar.config.manifest.path)? != sidecar.config.manifest.path
    {
        return Err("invalid or unsupported cohort partial-domain contract".into());
    }
    let manifest = manifest(&sidecar.config)?;
    if sidecar.full_source_gaussian_count != manifest.header.source_gaussian_count
        || sidecar.canonical_decoded_source_fingerprint
            != format!("{:016x}", manifest.build.source_fingerprint)
    {
        return Err("cohort canonical source identity disagrees with manifest".into());
    }
    let mut plan = plan(&manifest, &sidecar.config)?;
    if plan.payloads.len() != sidecar.payloads.len()
        || plan.encoded_bytes != sidecar.encoded_page_bytes_read
        || plan.decoded_bytes != sidecar.decoded_page_bytes
    {
        return Err("cohort payload/work plan disagrees with sidecar".into());
    }
    context::verify_selection(&plan.context_selection, &sidecar.context_selection)?;
    let mut readers = Vec::with_capacity(plan.payloads.len());
    for (planned, payload) in plan.payloads.iter().zip(&sidecar.payloads) {
        if planned.role != payload.role
            || planned.count != payload.gaussian_count
            || planned.bytes != payload.bytes
            || planned.ranges.len() != payload.ranges.len()
            || payload.file.path
                != sidecar
                    .config
                    .output_directory
                    .join(payload.role.filename())
            || fs::metadata(&payload.file.path)?.len() != planned.bytes
            || !hash_matches(&hash_file(&payload.file.path)?, &payload.file.sha256)
        {
            return Err("cohort payload identity/count/size disagrees with complete plan".into());
        }
        for (expected, actual) in planned.ranges.iter().zip(&payload.ranges) {
            let mut actual = actual.clone();
            actual.decoded_records_sha256.clear();
            if *expected != actual {
                return Err("cohort owner/source/page range disagrees with manifest".into());
            }
        }
        let mut reader = BufReader::new(File::open(&payload.file.path)?);
        let mut header = vec![0; planned.header.len()];
        reader.read_exact(&mut header)?;
        if header != planned.header {
            return Err("cohort PLY header disagrees with export profile".into());
        }
        readers.push(reader);
    }
    let header_lengths = plan
        .payloads
        .iter()
        .map(|payload| payload.header.len())
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(PLY_RECORD_BYTES as usize);
    let mut actual = vec![0; PLY_RECORD_BYTES as usize];
    let context_index = if original_context { 3 } else { 2 };
    let mut native = plan
        .payloads
        .iter()
        .enumerate()
        .map(|(index, payload)| {
            if index > 1 && index != context_index {
                return Ok(None);
            }
            let count = usize::try_from(payload.count)?;
            let mut records = Vec::new();
            records.try_reserve_exact(count)?;
            records.resize(count, Gaussian3d::default());
            Ok(Some(records))
        })
        .collect::<CaptureResult<Vec<Option<Vec<Gaussian3d>>>>>()?;
    let mut copied = vec![0_u64; plan.payloads.len()];
    let pages = visit_pages(
        &manifest,
        &sidecar.config,
        &mut plan,
        |payload, range, records| {
            let reader = &mut readers[payload];
            reader.seek(SeekFrom::Start(offset(header_lengths[payload], range)?))?;
            for gaussian in records {
                expected.clear();
                write_ply_gaussian(&mut expected, gaussian)?;
                reader.read_exact(&mut actual)?;
                if expected != actual {
                    return Err("cohort PLY does not encode its authenticated page records".into());
                }
            }
            if let Some(destination) = &mut native[payload] {
                let start = usize::try_from(range.output_start)?;
                let end = start
                    .checked_add(records.len())
                    .ok_or("native cohort range overflow")?;
                destination
                    .get_mut(start..end)
                    .ok_or("native cohort range exceeds admitted payload")?
                    .copy_from_slice(records);
                copied[payload] += records.len() as u64;
            }
            Ok(())
        },
    )?;
    if pages != sidecar.pages
        || plan
            .payloads
            .iter()
            .zip(&sidecar.payloads)
            .any(|(planned, payload)| planned.ranges != payload.ranges)
    {
        return Err("cohort authenticated page/decoded range hash mismatch".into());
    }
    for (index, records) in native.iter().enumerate() {
        if records.is_some() && copied[index] != plan.payloads[index].count {
            return Err("native cohort import did not cover the complete payload".into());
        }
    }
    let source = native[0].take().ok_or("missing native source payload")?;
    let initial = native[1].take().ok_or("missing native seed payload")?;
    let context = native[context_index]
        .take()
        .ok_or("missing native context payload")?;
    let domains = sidecar.payloads[1]
        .ranges
        .iter()
        .flat_map(|range| {
            std::iter::repeat_n(range.conservative_node_bounds, range.output_count as usize)
        })
        .collect();
    let (source_order, initial_order, context_order) = context::ordering(&sidecar)?;
    let result = LoadedCohort {
        source,
        initial,
        context,
        domains,
        source_order,
        initial_order,
        context_order,
        color_space: GaussianColorSpace::SrgbRec709Display,
        sidecar,
        sidecar_pin,
    };
    result.verify_inputs()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::lod_runtime_capture::source::export_rung::tests::Fixture;

    fn config(fixture: &Fixture) -> CohortExportConfig {
        let mut children = fixture
            .manifest
            .nodes
            .iter()
            .filter(|node| node.parent == Some(fixture.manifest.roots[0]))
            .collect::<Vec<_>>();
        children.sort_unstable_by_key(|node| node.source.start);
        assert!(children.len() >= 2);
        let path = fixture.root.join("scene.gsplatlod");
        CohortExportConfig {
            manifest: PinnedFile {
                sha256: hash_file(&path).unwrap(),
                path,
            },
            output_directory: fixture.root.join("cohort"),
            owned_nodes: vec![children[0].id],
            context_nodes: vec![children[1].id],
            captured_context: None,
            include_original_context: true,
            admission: CohortAdmission::default(),
        }
    }

    fn export(fixture: &Fixture, config: &CohortExportConfig) -> CaptureResult<()> {
        let path = fixture.root.join("cohort-config.json");
        fs::write(&path, serde_json::to_vec(config)?)?;
        export_cohort(&path)
    }

    #[test]
    fn cohort_export_authenticates_partial_domains_and_original_leaf_cover() {
        let fixture = Fixture::new();
        let config = config(&fixture);
        let planned = plan(&fixture.manifest, &config).unwrap();
        assert!(planned.payloads[0].count > planned.payloads[1].count);
        export(&fixture, &config).unwrap();
        let path = config.output_directory.join("cohort.json");
        let sha = hash_file(&path).unwrap();
        let loaded = load_cohort(&path, &sha, false).unwrap();
        assert_eq!(loaded.source.len() as u64, planned.payloads[0].count);
        assert_eq!(loaded.initial.len() as u64, planned.payloads[1].count);
        assert_eq!(loaded.context.len() as u64, planned.payloads[2].count);
        // Fitting must retain the authenticated f32 values bit-for-bit. PLY
        // log/exp, anisotropy clamping and quaternion normalization are export
        // reload effects, not transformations of native package supervision.
        for (records, payload) in [&loaded.source, &loaded.initial, &loaded.context]
            .into_iter()
            .zip(&loaded.sidecar.payloads)
        {
            for range in &payload.ranges {
                let start = range.output_start as usize;
                let mut hash = Sha256::new();
                for record in &records[start..start + range.output_count as usize] {
                    hash_gaussian(&mut hash, record);
                }
                assert_eq!(
                    format!("{:x}", hash.finalize()),
                    range.decoded_records_sha256
                );
            }
        }
        assert_eq!(loaded.domains.len(), loaded.initial.len());
        assert!(loaded.sidecar.partial_domains_validated);
        assert!(!loaded.sidecar.complete_source_antichain_validated);
        assert!(loaded.sidecar.original_input_sha256.is_none());
        let original_context = load_cohort(&path, &sha, true).unwrap();
        assert_eq!(
            original_context.context.len() as u64,
            planned.payloads[3].count
        );
        loaded.verify_inputs().unwrap();

        // The receipt cannot invent source coverage even if a caller pins the
        // modified receipt's new hash. Import re-derives it from the manifest.
        let mut forged: CohortSidecar = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        forged.payloads[0].ranges[0].source.start += 1;
        fs::write(&path, serde_json::to_vec(&forged).unwrap()).unwrap();
        assert!(load_cohort(&path, &hash_file(&path).unwrap(), false).is_err());
        assert!(loaded.verify_inputs().is_err());
    }

    #[test]
    fn cohort_export_rejects_overlap_incomplete_cover_and_aggregate_limits() {
        let fixture = Fixture::new();
        let config = config(&fixture);
        let mut repeated = config.clone();
        repeated.context_nodes = repeated.owned_nodes.clone();
        assert!(plan(&fixture.manifest, &repeated).is_err());
        let mut ancestor = config.clone();
        ancestor.context_nodes = vec![fixture.manifest.roots[0]];
        assert!(plan(&fixture.manifest, &ancestor).is_err());

        let planned = plan(&fixture.manifest, &config).unwrap();
        let mut limited = config.clone();
        limited.admission.max_source_gaussians = planned.payloads[0].count;
        assert!(
            export(&fixture, &limited).is_err(),
            "immutable context must consume teacher admission"
        );
        assert!(!limited.output_directory.exists());
        let mut incomplete = fixture.manifest.clone();
        let leaf = planned.payloads[0].ranges[0].node;
        incomplete
            .nodes
            .iter_mut()
            .find(|node| node.id == leaf)
            .unwrap()
            .source
            .start += 1;
        assert!(plan(&incomplete, &config).is_err());

        fs::create_dir(&config.output_directory).unwrap();
        fs::write(config.output_directory.join("keep"), b"untouched").unwrap();
        assert!(export(&fixture, &config).is_err());
        assert_eq!(
            fs::read(config.output_directory.join("keep")).unwrap(),
            b"untouched"
        );
    }

    #[test]
    fn cohort_export_and_import_reject_corrupt_pages_without_partial_publication() {
        let fixture = Fixture::new();
        let mut config = config(&fixture);
        export(&fixture, &config).unwrap();
        let path = config.output_directory.join("cohort.json");
        let sha = hash_file(&path).unwrap();
        let planned = plan(&fixture.manifest, &config).unwrap();
        let page = planned.pages.keys().next().unwrap();
        let storage = fixture
            .manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == *page)
            .unwrap()
            .storage
            .as_ref()
            .unwrap();
        let (start, length) = storage.byte_range.unwrap();
        let packed_path = fixture.root.join(&storage.uri);
        let mut bytes = fs::read(&packed_path).unwrap();
        bytes[(start + length - 1) as usize] ^= 1;
        fs::write(packed_path, bytes).unwrap();
        assert!(load_cohort(&path, &sha, false).is_err());
        config.output_directory = fixture.root.join("corrupt-export");
        assert!(export(&fixture, &config).is_err());
        assert!(!config.output_directory.exists());
    }
}
