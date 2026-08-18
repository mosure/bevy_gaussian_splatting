//! Deterministic CPU reference construction and validation for 3D Gaussian LoD.
//!
//! This module is intentionally renderer-independent.  It establishes the
//! versioned hierarchy/page contract and provides a bounded-complexity CPU
//! builder against which the bounded GPU hierarchy primitives are tested.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    error::Error,
    fmt,
    mem::size_of,
};

use bevy::math::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::{
    gaussian::{
        f32::{PositionVisibility, Rotation, ScaleOpacity},
        formats::{
            planar_3d::{Gaussian3d, PlanarGaussian3d},
            planar_3d_chunked::{
                GaussianField, LOD_PAGE_SCHEMA_VERSION, LodBounds, LodBoundsError, LodIndexRange,
                LodNodeId, LodPageDescriptor, LodPageEncoding, LodPageId, LodPageKind,
                LodPageRange, LodPageValidationError, LodSourceRange, PlanarGaussian3dPage,
                StableHasher, stable_gaussian_hash, validate_gaussian,
            },
        },
    },
    material::spherical_harmonics::{SH_COEFF_COUNT, SH_DEGREE, SphericalHarmonicCoefficients},
};

pub const LOD_MANIFEST_MAGIC: [u8; 8] = *b"BGSLOD3\0";
pub const LOD_MANIFEST_VERSION: u16 = 3;
/// CPU builder ABI which guarantees that one MomentMerge refinement cannot
/// increase its parent's representation count by more than the configured
/// [`GaussianLodBuildSettings::branching_factor`]. ABI 13 also guarantees that
/// representative covariance uses the renderer's `Q D Q^T` convention, emits
/// a conservative high-fidelity certificate (including anisotropy growth) for
/// every hierarchy node, uses risk-ranked adjacent agglomeration for ordinary
/// reductions averaging at most 16 source records per representative while preserving
/// a conservative balanced-partition selection-metadata envelope,
/// preserves the risk-ranked adjacent-pair bridge directly above 64-record
/// logical leaves, and packs logical node payloads into independently bounded
/// physical pages.
pub const PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION: u32 = 13;
// External CPU/GPU package builders use distinct, wide topologies. Their ABI
// values remain readable even though only the progressive CPU builder lives in
// this module.
const EXTERNAL_CPU_MOMENT_MERGE_BUILDER_ABI_VERSION: u32 = 5;
const EXTERNAL_GPU_MOMENT_MERGE_BUILDER_ABI_VERSION: u32 = 6;
/// MomentMerge version 2 fixes the covariance convention used for both source
/// accumulation and representative eigensolve output. Version 1 pages can
/// contain representatives whose anisotropic axes are transposed at render time.
pub const MOMENT_MERGE_VERSION: u32 = 2;
pub const LOD_REQUIRED_FEATURE_SH0: u64 = 1 << 0;
pub const LOD_REQUIRED_FEATURE_SH1: u64 = 1 << 1;
pub const LOD_REQUIRED_FEATURE_SH2: u64 = 1 << 2;
pub const LOD_REQUIRED_FEATURE_SH3: u64 = 1 << 3;
pub const LOD_REQUIRED_FEATURE_SH4: u64 = 1 << 4;
/// Every node carries a validated, monotonic high-fidelity certificate.
pub const LOD_REQUIRED_FEATURE_HIGH_FIDELITY_CERTIFICATE: u64 = 1 << 5;
/// Multiple same-depth, same-kind node ranges may share one physical page. The
/// page descriptor bound covers their union; payload validation checks each
/// node's referenced slice against that node's own bound.
pub const LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES: u64 = 1 << 6;
pub const LOD_REQUIRED_FEATURE_SH_MASK: u64 = LOD_REQUIRED_FEATURE_SH0
    | LOD_REQUIRED_FEATURE_SH1
    | LOD_REQUIRED_FEATURE_SH2
    | LOD_REQUIRED_FEATURE_SH3
    | LOD_REQUIRED_FEATURE_SH4;
pub const LOD_CURRENT_SH_FEATURE: u64 = 1 << SH_DEGREE;
pub const LOD_CURRENT_REQUIRED_FEATURES: u64 =
    LOD_CURRENT_SH_FEATURE | LOD_REQUIRED_FEATURE_HIGH_FIDELITY_CERTIFICATE;
pub const LOD_SUPPORTED_REQUIRED_FEATURES: u64 =
    LOD_CURRENT_REQUIRED_FEATURES | LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES;

const PROGRESSIVE_LOGICAL_LEAF_CAPACITY: u32 = 64;
const PROGRESSIVE_PHYSICAL_PAGE_CAPACITY: u32 = 1024;
const HIGH_FIDELITY_PAIR_CERTIFICATE: f32 = 0.95;
const HIGH_FIDELITY_MAX_REPRESENTATIVE_NUMERATOR: usize = 7;
const HIGH_FIDELITY_MAX_REPRESENTATIVE_DENOMINATOR: usize = 8;
/// Bound the more expensive risk-aware ordinary rung to the near-leaf regime.
/// Larger reductions retain the deterministic balanced reducer.
const PROGRESSIVE_RISK_AWARE_MAX_SOURCES_PER_REPRESENTATIVE: usize = 16;

pub const LOD_MORTON_BITS_PER_AXIS: u32 = 21;
pub const LOD_MORTON_AXIS_MAX: u32 = (1 << LOD_MORTON_BITS_PER_AXIS) - 1;

/// Immutable choices which affect topology, representatives, or page identity.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GaussianLodBuildSettings {
    /// Maximum number of children in an internal node and maximum permitted
    /// parent-to-children representation-count amplification for progressive
    /// MomentMerge hierarchies.
    ///
    /// The CPU builder uses binary topology for progressive MomentMerge, so
    /// this value normally controls the count amplification rather than the
    /// child count. Other versioned builders may use the full child bound.
    pub branching_factor: u8,
    /// Maximum number of decoded Gaussians stored in a physical page. The
    /// progressive CPU builder additionally caps ordinary physical pages at
    /// 1024 records and uses smaller logical leaves so multiple node slices can
    /// share one page without increasing per-request decode work.
    pub leaf_capacity: u32,
    /// Truncated Gaussian support radius, measured in standard deviations.
    pub support_sigma: f32,
}

impl Default for GaussianLodBuildSettings {
    fn default() -> Self {
        Self {
            branching_factor: 8,
            leaf_capacity: 1024,
            support_sigma: 3.0,
        }
    }
}

impl GaussianLodBuildSettings {
    pub fn validate(self) -> Result<(), LodBuildSettingsError> {
        if !(2..=32).contains(&self.branching_factor) {
            return Err(LodBuildSettingsError::BranchingFactor(
                self.branching_factor,
            ));
        }
        if self.leaf_capacity == 0 {
            return Err(LodBuildSettingsError::LeafCapacity);
        }
        if !self.support_sigma.is_finite() || self.support_sigma <= 0.0 {
            return Err(LodBuildSettingsError::SupportSigma(self.support_sigma));
        }
        Ok(())
    }

    pub fn stable_hash(self) -> u64 {
        let mut hash = StableHasher::new();
        hash.write(&[self.branching_factor]);
        hash.write(&self.leaf_capacity.to_le_bytes());
        hash.write(&canonical_f32_bits(self.support_sigma).to_le_bytes());
        hash.finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LodBuildSettingsError {
    BranchingFactor(u8),
    LeafCapacity,
    SupportSigma(f32),
}

impl fmt::Display for LodBuildSettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BranchingFactor(value) => write!(
                f,
                "LoD branching factor {value} is outside the supported range 2..=32"
            ),
            Self::LeafCapacity => write!(f, "LoD leaf capacity must be non-zero"),
            Self::SupportSigma(value) => write!(
                f,
                "LoD support sigma {value} must be finite and greater than zero"
            ),
        }
    }
}

impl Error for LodBuildSettingsError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LodReducerKind {
    MomentMerge,
}

fn moment_merge_config_fingerprint(build: GaussianLodBuildSettings) -> u64 {
    let mut hash = StableHasher::new();
    // Preserve the promoted MomentMerge fingerprint byte-for-byte so valid
    // manifests do not change when the unused reducer configuration is removed.
    hash.write(b"BGSLOD MomentMerge config");
    hash.write(&build.stable_hash().to_le_bytes());
    hash.write(&MOMENT_MERGE_VERSION.to_le_bytes());
    hash.finish()
}

/// Fingerprint for every immutable hierarchy and page-encoding choice.
///
/// `None` returns the MomentMerge-and-build fingerprint directly. A compressed
/// representative degree uses a domain-separated extension; source-leaf pages
/// are always full-degree f32 and therefore need no additional flag.
pub fn lod_config_fingerprint(
    build: GaussianLodBuildSettings,
    compressed_representative_sh_degree: Option<u8>,
) -> u64 {
    let base = moment_merge_config_fingerprint(build);
    let Some(degree) = compressed_representative_sh_degree else {
        return base;
    };
    let mut hash = StableHasher::new();
    hash.write(b"BGSLOD representative F16 SH v1");
    hash.write(&base.to_le_bytes());
    hash.write(&[degree]);
    hash.finish()
}

/// Conservative selection-error metadata. All fields are monotonic toward a
/// root in manifests produced by [`CpuGaussianLodBuilder`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LodError {
    /// Cloud-local spatial extent not represented as distinct source means.
    pub geometric: f32,
    /// RMS spherical-harmonic residual.
    pub appearance: f32,
    /// Maximum source-versus-representative opacity difference.
    pub opacity: f32,
    /// Scalar scheduling error, currently the maximum of the three channels.
    pub combined: f32,
}

impl LodError {
    pub const ZERO: Self = Self {
        geometric: 0.0,
        appearance: 0.0,
        opacity: 0.0,
        combined: 0.0,
    };

    #[inline]
    pub fn max(self, other: Self) -> Self {
        let geometric = self.geometric.max(other.geometric);
        let appearance = self.appearance.max(other.appearance);
        let opacity = self.opacity.max(other.opacity);
        Self {
            geometric,
            appearance,
            opacity,
            combined: self
                .combined
                .max(other.combined)
                .max(geometric)
                .max(appearance)
                .max(opacity),
        }
    }

    fn validate(self) -> bool {
        let components = [self.geometric, self.appearance, self.opacity, self.combined];
        components
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && self.combined + f32::EPSILON >= self.geometric
            && self.combined + f32::EPSILON >= self.appearance
            && self.combined + f32::EPSILON >= self.opacity
    }

    fn contains(self, child: Self) -> bool {
        let epsilon = 1e-5 * self.combined.max(child.combined).max(1.0);
        self.geometric + epsilon >= child.geometric
            && self.appearance + epsilon >= child.appearance
            && self.opacity + epsilon >= child.opacity
            && self.combined + epsilon >= child.combined
    }
}

/// Quality interval at which a node is eligible to represent its region.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LodQualityInterval {
    pub min: f32,
    pub max: f32,
}

impl LodQualityInterval {
    fn validate(self) -> bool {
        self.min.is_finite()
            && self.max.is_finite()
            && self.min >= 0.0
            && self.max <= 1.0
            && self.min <= self.max
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodMortonRange {
    pub min: u64,
    pub max: u64,
}

/// One node in the breadth-first Morton hierarchy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GaussianLodNode {
    pub id: LodNodeId,
    pub parent: Option<LodNodeId>,
    pub depth: u16,
    pub bounds: LodBounds,
    /// Immediate children in [`GaussianLodManifest::nodes`].
    pub children: LodIndexRange,
    /// Descendants in the canonical Morton-sorted source sequence.
    pub source: LodSourceRange,
    pub morton: LodMortonRange,
    /// Replacement records used when this node is accepted by traversal.
    pub representation: LodPageRange,
    pub error: LodError,
    pub quality: LodQualityInterval,
    /// Continuous, view-independent confidence that this node's replacement
    /// records preserve projected alpha mass and source support. Exact-source
    /// leaves are 1; uncertified or high-risk representatives approach 0.
    ///
    /// Missing data decodes conservatively as zero. Manifest v3 requires the
    /// corresponding feature bit and validates this value monotonically toward
    /// the root, so old manifests cannot silently acquire certification.
    #[serde(default)]
    pub high_fidelity_certificate: f32,
}

impl GaussianLodNode {
    #[inline]
    pub const fn is_leaf(&self) -> bool {
        self.children.count == 0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GaussianLodQualityMetadata {
    pub max_depth: u16,
    pub coarsest_gaussian_count: u64,
    pub finest_gaussian_count: u64,
    pub max_error: LodError,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GaussianLodBuildMetadata {
    pub settings: GaussianLodBuildSettings,
    pub reducer: LodReducerKind,
    pub builder_abi_version: u32,
    pub reducer_version: u32,
    /// Hash of the canonical Morton-sorted decoded source.
    pub source_fingerprint: u64,
    pub config_fingerprint: u64,
}

impl GaussianLodBuildMetadata {
    /// Whether the manifest guarantees binary topology with monotonic,
    /// configured parent-to-children representation-count amplification.
    pub const fn has_bounded_refinement_amplification(&self) -> bool {
        is_progressive_moment_merge_builder_abi(self.builder_abi_version)
            && matches!(self.reducer, LodReducerKind::MomentMerge)
    }
}

const fn is_progressive_moment_merge_builder_abi(builder_abi_version: u32) -> bool {
    builder_abi_version == PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION
}

const fn is_supported_moment_merge_builder_abi(builder_abi_version: u32) -> bool {
    matches!(
        builder_abi_version,
        EXTERNAL_CPU_MOMENT_MERGE_BUILDER_ABI_VERSION
            | EXTERNAL_GPU_MOMENT_MERGE_BUILDER_ABI_VERSION
            | PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION
    )
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GaussianLodManifestHeader {
    pub magic: [u8; 8],
    pub manifest_version: u16,
    pub page_schema_version: u16,
    /// Required decoded-data ABI features. Manifest version 3 requires the
    /// compiled SH layout and the node high-fidelity certificate contract.
    pub required_features: u64,
    pub source_gaussian_count: u64,
    pub stored_gaussian_count: u64,
    pub node_count: u32,
    pub page_count: u32,
}

/// Portable hierarchy manifest. It is deliberately separate from page bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GaussianLodManifest {
    pub header: GaussianLodManifestHeader,
    pub scene_bounds: Option<LodBounds>,
    pub roots: Vec<LodNodeId>,
    /// Breadth-first order makes every immediate-child range contiguous.
    pub nodes: Vec<GaussianLodNode>,
    pub pages: Vec<LodPageDescriptor>,
    pub build: GaussianLodBuildMetadata,
    pub quality: GaussianLodQualityMetadata,
}

impl GaussianLodManifest {
    pub fn validate(&self) -> Result<(), LodValidationError> {
        if self.header.magic != LOD_MANIFEST_MAGIC {
            return Err(LodValidationError::InvalidMagic(self.header.magic));
        }
        if self.header.manifest_version != LOD_MANIFEST_VERSION {
            return Err(LodValidationError::UnsupportedManifestVersion(
                self.header.manifest_version,
            ));
        }
        if self.header.page_schema_version != LOD_PAGE_SCHEMA_VERSION {
            return Err(LodValidationError::UnsupportedPageVersion(
                self.header.page_schema_version,
            ));
        }
        let required_sh = self.header.required_features & LOD_REQUIRED_FEATURE_SH_MASK;
        if required_sh != LOD_CURRENT_SH_FEATURE {
            return Err(LodValidationError::IncompatibleSphericalHarmonics {
                required: required_sh,
                supported: LOD_CURRENT_SH_FEATURE,
            });
        }
        let unsupported_features = self.header.required_features & !LOD_SUPPORTED_REQUIRED_FEATURES;
        if unsupported_features != 0 {
            return Err(LodValidationError::UnsupportedRequiredFeatures(
                unsupported_features,
            ));
        }
        if self.header.required_features & LOD_REQUIRED_FEATURE_HIGH_FIDELITY_CERTIFICATE == 0 {
            return Err(LodValidationError::MissingHighFidelityCertificateFeature);
        }
        let shared_node_pages =
            self.header.required_features & LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES != 0;
        self.build
            .settings
            .validate()
            .map_err(LodValidationError::InvalidBuildSettings)?;
        if !is_supported_moment_merge_builder_abi(self.build.builder_abi_version)
            || self.build.reducer != LodReducerKind::MomentMerge
            || self.build.reducer_version != MOMENT_MERGE_VERSION
        {
            return Err(LodValidationError::InvalidBuildVersion);
        }
        let actual_node_count = u32::try_from(self.nodes.len())
            .map_err(|_| LodValidationError::CountOverflow("nodes"))?;
        let actual_page_count = u32::try_from(self.pages.len())
            .map_err(|_| LodValidationError::CountOverflow("pages"))?;
        if actual_node_count != self.header.node_count {
            return Err(LodValidationError::CountMismatch("nodes"));
        }
        if actual_page_count != self.header.page_count {
            return Err(LodValidationError::CountMismatch("pages"));
        }

        if self.header.source_gaussian_count == 0 {
            if self.scene_bounds.is_some()
                || !self.roots.is_empty()
                || !self.nodes.is_empty()
                || !self.pages.is_empty()
                || self.header.stored_gaussian_count != 0
                || self.quality != GaussianLodQualityMetadata::default()
            {
                return Err(LodValidationError::InvalidEmptyManifest);
            }
            return Ok(());
        }

        let scene_bounds = self
            .scene_bounds
            .ok_or(LodValidationError::MissingSceneBounds)?;
        scene_bounds
            .validate()
            .map_err(LodValidationError::InvalidSceneBounds)?;
        if self.roots.is_empty() || self.nodes.is_empty() || self.pages.is_empty() {
            return Err(LodValidationError::IncompleteManifest);
        }

        let mut page_by_id = HashMap::with_capacity(self.pages.len());
        let mut compressed_representative_sh_degree = None;
        let mut saw_f32_representative = false;
        let mut stored_gaussian_count = 0_u64;
        for (index, page) in self.pages.iter().enumerate() {
            page.validate()
                .map_err(|source| LodValidationError::InvalidPage { index, source })?;
            if page_by_id.insert(page.id, index).is_some() {
                return Err(LodValidationError::DuplicatePageId(page.id));
            }
            match (page.kind, page.encoding) {
                (LodPageKind::SourceLeaves | LodPageKind::Mixed, LodPageEncoding::F16Sh { .. }) => {
                    return Err(LodValidationError::CompressedSourceLeaf(page.id));
                }
                (LodPageKind::Representatives, LodPageEncoding::F16Sh { degree }) => {
                    if saw_f32_representative
                        || compressed_representative_sh_degree
                            .is_some_and(|current| current != degree)
                    {
                        return Err(LodValidationError::InconsistentRepresentativeEncoding);
                    }
                    compressed_representative_sh_degree = Some(degree);
                }
                (LodPageKind::Representatives, LodPageEncoding::F32Planar) => {
                    if compressed_representative_sh_degree.is_some() {
                        return Err(LodValidationError::InconsistentRepresentativeEncoding);
                    }
                    saw_f32_representative = true;
                }
                _ => {}
            }
            stored_gaussian_count = stored_gaussian_count
                .checked_add(u64::from(page.gaussian_count))
                .ok_or(LodValidationError::CountOverflow("stored Gaussians"))?;
        }
        if stored_gaussian_count != self.header.stored_gaussian_count {
            return Err(LodValidationError::CountMismatch("stored Gaussians"));
        }
        if self.build.config_fingerprint
            != lod_config_fingerprint(self.build.settings, compressed_representative_sh_degree)
        {
            return Err(LodValidationError::ConfigFingerprintMismatch);
        }

        let mut node_by_id = HashMap::with_capacity(self.nodes.len());
        let mut page_ranges: HashMap<LodPageId, Vec<(u32, u32, u16, bool)>> = HashMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if !node.id.is_valid() {
                return Err(LodValidationError::InvalidNodeId(index));
            }
            if node_by_id.insert(node.id, index).is_some() {
                return Err(LodValidationError::DuplicateNodeId(node.id));
            }
            node.bounds
                .validate()
                .map_err(|source| LodValidationError::InvalidNodeBounds { index, source })?;
            if node.source.count == 0
                || node
                    .source
                    .end()
                    .is_none_or(|end| end > self.header.source_gaussian_count)
            {
                return Err(LodValidationError::InvalidSourceRange(node.id));
            }
            if node.morton.min > node.morton.max {
                return Err(LodValidationError::InvalidMortonRange(node.id));
            }
            if node.representation.count == 0 {
                return Err(LodValidationError::EmptyRepresentation(node.id));
            }
            let page_index = *page_by_id
                .get(&node.representation.page)
                .ok_or(LodValidationError::UnknownPage(node.representation.page))?;
            let representation_end = node
                .representation
                .end()
                .ok_or(LodValidationError::InvalidPageRange(node.id))?;
            if representation_end > self.pages[page_index].gaussian_count {
                return Err(LodValidationError::InvalidPageRange(node.id));
            }
            let owns_entire_page = node.representation.offset == 0
                && representation_end == self.pages[page_index].gaussian_count;
            if !shared_node_pages || owns_entire_page {
                let page_bounds = &self.pages[page_index].bounds;
                let page_bounds_epsilon = bounds_epsilon(&node.bounds, page_bounds);
                if !node
                    .bounds
                    .contains_with_epsilon(page_bounds, page_bounds_epsilon)
                {
                    return Err(LodValidationError::RepresentationOutsideNode(node.id));
                }
            }
            page_ranges
                .entry(node.representation.page)
                .or_default()
                .push((
                    node.representation.offset,
                    representation_end,
                    node.depth,
                    node.is_leaf(),
                ));
            if !node.error.validate() {
                return Err(LodValidationError::InvalidError(node.id));
            }
            if !node.quality.validate() {
                return Err(LodValidationError::InvalidQualityInterval(node.id));
            }
            if !node.high_fidelity_certificate.is_finite()
                || !(0.0..=1.0).contains(&node.high_fidelity_certificate)
            {
                return Err(LodValidationError::InvalidHighFidelityCertificate(node.id));
            }
            let child_end = node
                .children
                .end()
                .ok_or(LodValidationError::InvalidChildRange(node.id))?;
            if child_end as usize > self.nodes.len() {
                return Err(LodValidationError::InvalidChildRange(node.id));
            }
        }

        for node in &self.nodes {
            if let Some(parent) = node.parent
                && !node_by_id.contains_key(&parent)
            {
                return Err(LodValidationError::UnknownParent {
                    node: node.id,
                    parent,
                });
            }
        }

        let mut root_indices = Vec::with_capacity(self.roots.len());
        let mut unique_roots = HashSet::with_capacity(self.roots.len());
        for root in &self.roots {
            if !unique_roots.insert(*root) {
                return Err(LodValidationError::DuplicateRoot(*root));
            }
            let index = *node_by_id
                .get(root)
                .ok_or(LodValidationError::UnknownRoot(*root))?;
            if self.nodes[index].parent.is_some() || self.nodes[index].depth != 0 {
                return Err(LodValidationError::InvalidRoot(*root));
            }
            if self.nodes[index].quality.min.abs() > f32::EPSILON {
                return Err(LodValidationError::InvalidQualityInterval(*root));
            }
            root_indices.push(index);
        }

        root_indices.sort_unstable_by_key(|index| self.nodes[*index].source.start);
        validate_source_partition(
            self.header.source_gaussian_count,
            root_indices.iter().map(|index| self.nodes[*index].source),
        )
        .map_err(|_| LodValidationError::RootSourcePartition)?;

        let mut visited = vec![false; self.nodes.len()];
        let mut queue = VecDeque::from(root_indices.clone());
        while let Some(index) = queue.pop_front() {
            if std::mem::replace(&mut visited[index], true) {
                return Err(LodValidationError::CycleOrSharedChild(self.nodes[index].id));
            }
            let node = &self.nodes[index];
            let child_start = node.children.start as usize;
            let child_end = node.children.end().unwrap() as usize;
            if child_start == child_end {
                if self.pages[*page_by_id.get(&node.representation.page).unwrap()].kind
                    == LodPageKind::Representatives
                {
                    return Err(LodValidationError::PageKindMismatch(node.id));
                }
                if u64::from(node.representation.count) != node.source.count
                    || node.source.count > u64::from(self.build.settings.leaf_capacity)
                {
                    return Err(LodValidationError::InvalidLeafRepresentation(node.id));
                }
                if (node.quality.max - 1.0).abs() > f32::EPSILON {
                    return Err(LodValidationError::InvalidQualityInterval(node.id));
                }
                if (node.high_fidelity_certificate - 1.0).abs() > f32::EPSILON {
                    return Err(LodValidationError::InvalidHighFidelityCertificate(node.id));
                }
                continue;
            }
            let progressive_moment_merge = self.build.has_bounded_refinement_amplification();
            if (progressive_moment_merge && child_end - child_start != 2)
                || child_end - child_start < 2
                || child_end - child_start > usize::from(self.build.settings.branching_factor)
            {
                return Err(LodValidationError::InvalidBranching(node.id));
            }
            if self.pages[*page_by_id.get(&node.representation.page).unwrap()].kind
                == LodPageKind::SourceLeaves
            {
                return Err(LodValidationError::PageKindMismatch(node.id));
            }

            let child_indices = child_start..child_end;
            validate_absolute_source_partition(
                node.source,
                child_indices.clone().map(|child| self.nodes[child].source),
            )
            .map_err(|_| LodValidationError::ChildSourcePartition(node.id))?;

            let mut previous_morton_max = None;
            let mut first_morton_min = None;
            let mut child_representation_count = 0_u64;
            for child_index in child_indices {
                let child = &self.nodes[child_index];
                child_representation_count = child_representation_count
                    .checked_add(u64::from(child.representation.count))
                    .ok_or(LodValidationError::CountOverflow("child representations"))?;
                if child.parent != Some(node.id) {
                    return Err(LodValidationError::ParentChildMismatch {
                        parent: node.id,
                        child: child.id,
                    });
                }
                if u32::from(child.depth) != u32::from(node.depth) + 1 {
                    return Err(LodValidationError::DepthMismatch(child.id));
                }
                if child.quality.min + f32::EPSILON < node.quality.min
                    || child.quality.max + f32::EPSILON < node.quality.max
                {
                    return Err(LodValidationError::InvalidQualityInterval(child.id));
                }
                if let Some(previous) = previous_morton_max
                    && child.morton.min < previous
                {
                    return Err(LodValidationError::MortonOrder(node.id));
                }
                first_morton_min.get_or_insert(child.morton.min);
                previous_morton_max = Some(child.morton.max);
                if child.morton.min < node.morton.min || child.morton.max > node.morton.max {
                    return Err(LodValidationError::InvalidMortonRange(child.id));
                }
                let epsilon = bounds_epsilon(&node.bounds, &child.bounds);
                if !node.bounds.contains_with_epsilon(&child.bounds, epsilon) {
                    return Err(LodValidationError::BoundsDoNotContainChild {
                        parent: node.id,
                        child: child.id,
                    });
                }
                if !node.error.contains(child.error) {
                    return Err(LodValidationError::NonMonotonicError {
                        parent: node.id,
                        child: child.id,
                    });
                }
                if node.high_fidelity_certificate > child.high_fidelity_certificate + f32::EPSILON {
                    return Err(LodValidationError::NonMonotonicHighFidelityCertificate {
                        parent: node.id,
                        child: child.id,
                    });
                }
                queue.push_back(child_index);
            }
            if progressive_moment_merge {
                let parent_count = u64::from(node.representation.count);
                let maximum_child_count = parent_count
                    .checked_mul(u64::from(self.build.settings.branching_factor))
                    .ok_or(LodValidationError::CountOverflow(
                        "refinement amplification",
                    ))?;
                if child_representation_count < parent_count
                    || child_representation_count > maximum_child_count
                {
                    return Err(LodValidationError::InvalidRefinementAmplification {
                        node: node.id,
                        parent_count,
                        child_count: child_representation_count,
                        maximum: self.build.settings.branching_factor,
                    });
                }
            }
            if first_morton_min != Some(node.morton.min)
                || previous_morton_max != Some(node.morton.max)
            {
                return Err(LodValidationError::InvalidMortonRange(node.id));
            }
        }
        if let Some(index) = visited.iter().position(|visited| !visited) {
            return Err(LodValidationError::UnreachableNode(self.nodes[index].id));
        }

        for (page_id, ranges) in &mut page_ranges {
            ranges.sort_unstable_by_key(|&(start, end, _, _)| (start, end));
            if ranges.len() > 1 && !shared_node_pages {
                return Err(LodValidationError::MissingSharedNodePageFeature(*page_id));
            }
            let page = &self.pages[*page_by_id.get(page_id).unwrap()];
            if ranges.len() > 1 {
                let expected_depth = ranges[0].2;
                let expected_leaf_kind = ranges[0].3;
                if page.kind == LodPageKind::Mixed
                    || ranges.iter().any(|&(_, _, depth, leaf_kind)| {
                        depth != expected_depth || leaf_kind != expected_leaf_kind
                    })
                {
                    return Err(LodValidationError::InhomogeneousSharedNodePage(*page_id));
                }
            }
            let mut expected_offset = 0;
            for &(start, end, _, _) in ranges.iter() {
                if start != expected_offset || end <= start {
                    return Err(LodValidationError::PageCoverage(*page_id));
                }
                expected_offset = end;
            }
            if expected_offset != page.gaussian_count {
                return Err(LodValidationError::PageCoverage(*page_id));
            }
        }
        if page_ranges.len() != self.pages.len() {
            return Err(LodValidationError::UnreferencedPage);
        }

        for index in &root_indices {
            let root = &self.nodes[*index];
            let epsilon = bounds_epsilon(&scene_bounds, &root.bounds);
            if !scene_bounds.contains_with_epsilon(&root.bounds, epsilon) {
                return Err(LodValidationError::SceneBoundsDoNotContainRoot(root.id));
            }
        }

        let actual_max_depth = self.nodes.iter().map(|node| node.depth).max().unwrap();
        let actual_coarsest = root_indices.iter().try_fold(0_u64, |count, index| {
            count.checked_add(u64::from(self.nodes[*index].representation.count))
        });
        let actual_max_error = root_indices.iter().fold(LodError::ZERO, |error, index| {
            error.max(self.nodes[*index].error)
        });
        if self.quality.max_depth != actual_max_depth
            || self.quality.coarsest_gaussian_count
                != actual_coarsest.ok_or(LodValidationError::CountOverflow("coarsest"))?
            || self.quality.finest_gaussian_count != self.header.source_gaussian_count
            || self.quality.max_error != actual_max_error
        {
            return Err(LodValidationError::QualityMetadataMismatch);
        }

        Ok(())
    }
}

/// Complete CPU/reference LoD product.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanarGaussian3dLod {
    pub manifest: GaussianLodManifest,
    pub pages: Vec<PlanarGaussian3dPage>,
}

impl PlanarGaussian3dLod {
    pub fn validate(&self) -> Result<(), LodValidationError> {
        self.manifest.validate()?;
        if self.pages.len() != self.manifest.pages.len() {
            return Err(LodValidationError::PayloadCountMismatch);
        }
        let descriptors: HashMap<_, _> = self
            .manifest
            .pages
            .iter()
            .map(|descriptor| (descriptor.id, descriptor))
            .collect();
        let mut seen = HashSet::with_capacity(self.pages.len());
        for page in &self.pages {
            if !seen.insert(page.id) {
                return Err(LodValidationError::DuplicatePayload(page.id));
            }
            let descriptor = descriptors
                .get(&page.id)
                .ok_or(LodValidationError::UnknownPayload(page.id))?;
            page.validate(descriptor)
                .map_err(|source| LodValidationError::InvalidPayload {
                    page: page.id,
                    source,
                })?;

            let mut actual_bounds: Option<LodBounds> = None;
            for gaussian in &page.gaussians {
                let bounds =
                    gaussian_support_bounds(gaussian, self.manifest.build.settings.support_sigma)
                        .map_err(|source| LodValidationError::InvalidPayloadBounds {
                        page: page.id,
                        source: Box::new(source),
                    })?;
                actual_bounds = Some(match actual_bounds {
                    Some(current) => current.union(bounds),
                    None => bounds,
                });
            }
            let actual_bounds = actual_bounds.expect("validated pages are non-empty");
            let epsilon = bounds_epsilon(&descriptor.bounds, &actual_bounds);
            if !descriptor
                .bounds
                .contains_with_epsilon(&actual_bounds, epsilon)
            {
                return Err(LodValidationError::PayloadOutsideDescriptor(page.id));
            }
        }
        let payloads: HashMap<_, _> = self.pages.iter().map(|page| (page.id, page)).collect();
        for node in &self.manifest.nodes {
            let page = payloads
                .get(&node.representation.page)
                .expect("manifest and payload identities were validated");
            let start = node.representation.offset as usize;
            let end = node.representation.end().unwrap() as usize;
            let mut actual_bounds: Option<LodBounds> = None;
            for gaussian in &page.gaussians[start..end] {
                let bounds =
                    gaussian_support_bounds(gaussian, self.manifest.build.settings.support_sigma)
                        .map_err(|source| LodValidationError::InvalidPayloadBounds {
                        page: page.id,
                        source: Box::new(source),
                    })?;
                actual_bounds = Some(match actual_bounds {
                    Some(current) => current.union(bounds),
                    None => bounds,
                });
            }
            let actual_bounds = actual_bounds.expect("validated node ranges are non-empty");
            let epsilon = bounds_epsilon(&node.bounds, &actual_bounds);
            if !node.bounds.contains_with_epsilon(&actual_bounds, epsilon) {
                return Err(LodValidationError::RepresentationOutsideNode(node.id));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct MomentMergeReducer {
    pub support_sigma: f32,
}

impl Default for MomentMergeReducer {
    fn default() -> Self {
        Self { support_sigma: 3.0 }
    }
}

impl MomentMergeReducer {
    pub fn new(support_sigma: f32) -> Result<Self, LodBuildSettingsError> {
        let settings = GaussianLodBuildSettings {
            support_sigma,
            ..GaussianLodBuildSettings::default()
        };
        settings.validate()?;
        Ok(Self { support_sigma })
    }

    pub fn reduce(&self, gaussians: &[Gaussian3d]) -> Result<MomentMergeResult, LodBuildError> {
        if gaussians.is_empty() {
            return Err(LodBuildError::EmptyReduction);
        }
        let mut accumulator = MomentAccumulator::new();
        for (index, gaussian) in gaussians.iter().enumerate() {
            validate_gaussian(gaussian)
                .map_err(|field| LodBuildError::InvalidGaussian { index, field })?;
            accumulator.add(gaussian, self.support_sigma)?;
        }
        accumulator.finish(self.support_sigma)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MomentMergeResult {
    pub gaussian: Gaussian3d,
    pub support_bounds: LodBounds,
    pub error: LodError,
    pub source_count: u64,
    pub total_weight: f64,
    raster_risk: MomentMergeRasterRisk,
}

impl MomentMergeResult {
    /// Internal preprocessing diagnostic. It is deliberately not serialized or
    /// exposed as runtime policy until real-scene calibration establishes a
    /// stable selector contract.
    #[cfg(test)]
    pub(crate) const fn raster_risk(&self) -> MomentMergeRasterRisk {
        self.raster_risk
    }

    /// Conservative all-view certificate for both raster geometry and SH
    /// appearance. `LodError::appearance` is RMS over all stored coefficients.
    /// Cauchy--Schwarz bounds one RGB channel's worst-direction SH delta by
    /// `sqrt(total_coefficients) * sqrt(coefficients_per_channel / 4pi) * RMS`.
    /// Mapping that non-negative bound through `1 / (1 + bound)` keeps the
    /// result continuous and makes 0.95 require at most about 5.26% color
    /// deviation. Opacity error is intentionally excluded: projected alpha
    /// mass is already bounded by the raster certificate, while compositing
    /// coincident source alpha makes raw per-record opacity differences noisy.
    fn high_fidelity_certificate(&self) -> f32 {
        let coefficients_per_channel = (SH_COEFF_COUNT / 3).max(1) as f32;
        let worst_direction_factor = (SH_COEFF_COUNT as f32).sqrt()
            * (coefficients_per_channel / (4.0 * std::f32::consts::PI)).sqrt();
        let appearance_bound = worst_direction_factor * self.error.appearance;
        let appearance_certificate = (1.0 + appearance_bound).recip().clamp(0.0, 1.0);
        self.raster_risk
            .high_fidelity_certificate()
            .min(appearance_certificate)
    }
}

/// View-independent analytic warning signals for a MomentMerge representative.
///
/// These are preprocessing diagnostics, not an advertised quality estimate.
/// The sampled projection term evaluates a fixed, rotation-symmetric direction
/// set. The Minkowski upper bound is conservative for every orthographic view,
/// exact for identical source covariance, and may overestimate risk when source
/// covariance frames differ substantially.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MomentMergeRasterRisk {
    pub(crate) sampled_projected_alpha_mass_inflation: f32,
    pub(crate) projected_alpha_mass_inflation_upper_bound: f32,
    pub(crate) support_leakage_fraction: f32,
    pub(crate) support_growth: f32,
    pub(crate) major_scale_growth: f32,
    pub(crate) anisotropy_growth: f32,
}

impl MomentMergeRasterRisk {
    #[cfg(test)]
    fn score(self) -> f32 {
        (self.sampled_projected_alpha_mass_inflation - 1.0)
            .max(0.0)
            .max(self.support_leakage_fraction)
            .max((self.support_growth - 1.0).max(0.0))
            .max((self.major_scale_growth - 1.0).max(0.0))
            .max((self.anisotropy_growth - 1.0).max(0.0))
    }

    /// Conservative confidence used to certify new hierarchy nodes. The first
    /// term bounds projected alpha-mass inflation for every orthographic view.
    /// The second rejects representatives whose spatial support or anisotropy
    /// grows beyond the source support envelope and source shape extrema.
    pub(crate) fn high_fidelity_certificate(self) -> f32 {
        let alpha_mass = self.projected_alpha_mass_inflation_upper_bound.max(1.0);
        let support = self
            .support_growth
            .max(self.major_scale_growth)
            .max(self.anisotropy_growth)
            .max(1.0);
        (alpha_mass.recip()).min(support.recip()).clamp(0.0, 1.0)
    }
}

const PROJECTED_ALPHA_MASS_DIRECTIONS: [[f64; 3]; 13] = {
    const S: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const T: f64 = 0.577_350_269_189_625_8;
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [S, S, 0.0],
        [S, -S, 0.0],
        [S, 0.0, S],
        [S, 0.0, -S],
        [0.0, S, S],
        [0.0, S, -S],
        [T, T, T],
        [T, T, -T],
        [T, -T, T],
        [-T, T, T],
    ]
};

#[derive(Clone, Debug, Default)]
pub struct CpuGaussianLodBuilder {
    pub settings: GaussianLodBuildSettings,
}

impl CpuGaussianLodBuilder {
    pub const fn new(settings: GaussianLodBuildSettings) -> Self {
        Self { settings }
    }

    pub fn build(&self, cloud: &PlanarGaussian3d) -> Result<PlanarGaussian3dLod, LodBuildError> {
        self.settings
            .validate()
            .map_err(LodBuildError::InvalidSettings)?;
        validate_plane_lengths(cloud)?;

        let mut source = Vec::with_capacity(cloud.position_visibility.len());
        for (index, gaussian) in cloud.iter().enumerate() {
            validate_gaussian(&gaussian)
                .map_err(|field| LodBuildError::InvalidGaussian { index, field })?;
            source.push(canonicalize_gaussian_zeros(gaussian));
        }

        if source.is_empty() {
            let output = empty_lod(self.settings);
            output.validate().map_err(LodBuildError::Validation)?;
            return Ok(output);
        }

        let center_bounds = source_center_bounds(&source)?;
        let mut keyed = Vec::with_capacity(source.len());
        for gaussian in source {
            keyed.push(KeyedGaussian {
                morton: canonical_lod_morton_code(
                    gaussian.position_visibility.position,
                    center_bounds,
                ),
                gaussian,
            });
        }
        keyed.sort_unstable_by(|left, right| {
            left.morton
                .cmp(&right.morton)
                .then_with(|| compare_gaussians(&left.gaussian, &right.gaussian))
        });

        let source_fingerprint = source_fingerprint(&keyed);
        let canonical_morton: Vec<_> = keyed.iter().map(|entry| entry.morton).collect();
        let canonical_source: Vec<_> = keyed.iter().map(|entry| entry.gaussian).collect();
        // Logical leaves are deliberately independent of transport page size.
        // With capacity >= 2, two adjacent logical leaves and every adjacent-pair
        // bridge fit in one physical page. Capacity 1 retains a forced 2:1
        // parent; its low certificate makes
        // high-quality traversal refine to the exact one-record leaves.
        let logical_leaf_capacity = if self.settings.leaf_capacity < 2 {
            1
        } else {
            PROGRESSIVE_LOGICAL_LEAF_CAPACITY.min(self.settings.leaf_capacity / 2)
        };
        let leaf_ranges = balanced_ranges(
            canonical_source.len(),
            logical_leaf_capacity as usize,
            canonical_source.len() > 1,
        );
        let mut temporary = Vec::new();
        let mut current_level = Vec::with_capacity(leaf_ranges.len());

        for range in leaf_ranges {
            let source_range = LodSourceRange {
                start: range.start as u64,
                count: (range.end - range.start) as u64,
            };
            let mut accumulator = MomentAccumulator::new();
            for gaussian in &canonical_source[range.clone()] {
                accumulator.add(gaussian, self.settings.support_sigma)?;
            }
            let bounds = accumulator
                .bounds
                .ok_or(LodBuildError::DerivedNonFinite("leaf bounds"))?;
            let index = temporary.len();
            temporary.push(TempNode {
                children: Vec::new(),
                source: source_range,
                morton: LodMortonRange {
                    min: canonical_morton[range.start],
                    max: canonical_morton[range.end - 1],
                },
                bounds,
                accumulator,
                representatives: Vec::new(),
                representation_count: range.end - range.start,
                error: LodError::ZERO,
                high_fidelity_certificate: 1.0,
            });
            current_level.push(index);
        }

        let mut deepest_choices = if self.settings.leaf_capacity >= 2 {
            plan_high_fidelity_deepest_choices(
                &canonical_source,
                &temporary,
                &current_level,
                self.settings.support_sigma,
                usize::from(self.settings.branching_factor),
            )?
        } else {
            HashMap::new()
        };

        while current_level.len() > 1 {
            // Pairing produces the deepest hierarchy available for the fixed
            // leaf-page capacity. An odd final node is carried to the next
            // level instead of creating an invalid unary parent. This keeps
            // default page granularity efficient while providing enough
            // refinement stages for bounded progressive representations.
            let paired_len = current_level.len() / 2 * 2;
            let child_groups = current_level[..paired_len]
                .chunks_exact(2)
                .map(|pair| pair.to_vec())
                .collect::<Vec<_>>();
            let carried_node = current_level.get(paired_len).copied();
            let mut next_level =
                Vec::with_capacity(child_groups.len() + carried_node.is_some() as usize);
            for children in child_groups {
                let first = &temporary[children[0]];
                let last = &temporary[*children.last().unwrap()];
                let source = LodSourceRange {
                    start: first.source.start,
                    count: last.source.end().unwrap() - first.source.start,
                };
                let morton = LodMortonRange {
                    min: first.morton.min,
                    max: last.morton.max,
                };
                let mut accumulator = MomentAccumulator::new();
                let mut bounds = first.bounds;
                let mut error = LodError::ZERO;
                let mut high_fidelity_certificate = 1.0_f32;
                for child in &children {
                    let child = &temporary[*child];
                    accumulator.combine(&child.accumulator)?;
                    bounds = bounds.union(child.bounds);
                    error = error.max(child.error);
                    high_fidelity_certificate =
                        high_fidelity_certificate.min(child.high_fidelity_certificate);
                }
                let source_start = usize::try_from(source.start)
                    .map_err(|_| LodBuildError::CountOverflow("internal source start"))?;
                let source_end = usize::try_from(source.end().unwrap())
                    .map_err(|_| LodBuildError::CountOverflow("internal source end"))?;
                let child_representatives = children.iter().try_fold(0_usize, |count, child| {
                    count
                        .checked_add(temporary[*child].representation_count)
                        .ok_or(LodBuildError::CountOverflow("child representations"))
                })?;
                let children_are_exact_leaves = children
                    .iter()
                    .all(|child| temporary[*child].children.is_empty());
                let source_records = &canonical_source[source_start..source_end];
                let (representatives, policy_envelope) = if children_are_exact_leaves
                    && let Some(choice) = deepest_choices.remove(&children[0])
                {
                    (
                        choice.into_representatives(),
                        ProgressiveSelectionEnvelope::IDENTITY,
                    )
                } else {
                    let rung = progressive_moment_merge_representatives(
                        source_records,
                        child_representatives
                            .div_ceil(usize::from(self.settings.branching_factor))
                            .max(1),
                        self.settings.support_sigma,
                    )?;
                    (rung.representatives, rung.policy_envelope)
                };
                for representative in &representatives {
                    bounds = bounds.union(representative.support_bounds);
                    error = error.max(representative.error);
                    high_fidelity_certificate =
                        high_fidelity_certificate.min(representative.high_fidelity_certificate());
                }
                if let Some(policy_bounds) = policy_envelope.support_bounds {
                    bounds = bounds.union(policy_bounds);
                }
                error = error.max(policy_envelope.error);
                high_fidelity_certificate =
                    high_fidelity_certificate.min(policy_envelope.high_fidelity_certificate_cap);
                let index = temporary.len();
                temporary.push(TempNode {
                    children,
                    source,
                    morton,
                    bounds,
                    accumulator,
                    representation_count: representatives.len(),
                    representatives,
                    error,
                    high_fidelity_certificate,
                });
                next_level.push(index);
            }
            if let Some(carried_node) = carried_node {
                next_level.push(carried_node);
            }
            current_level = next_level;
        }

        let root = current_level[0];
        let (order, parents, depths) = breadth_first_order(&temporary, root)?;
        let max_depth = *depths.iter().max().unwrap();
        let node_count =
            u32::try_from(order.len()).map_err(|_| LodBuildError::CountOverflow("nodes"))?;

        let mut old_to_new = vec![usize::MAX; temporary.len()];
        for (new_index, old_index) in order.iter().copied().enumerate() {
            old_to_new[old_index] = new_index;
        }

        let mut nodes = Vec::with_capacity(order.len());
        let mut node_page_payloads = Vec::with_capacity(order.len());

        for (new_index, old_index) in order.iter().copied().enumerate() {
            let temporary_node = &temporary[old_index];
            let node_id = LodNodeId((new_index as u64) + 1);
            let is_leaf = temporary_node.children.is_empty();
            let page_gaussians = if is_leaf {
                let start = usize::try_from(temporary_node.source.start)
                    .map_err(|_| LodBuildError::CountOverflow("leaf start"))?;
                let end = usize::try_from(temporary_node.source.end().unwrap())
                    .map_err(|_| LodBuildError::CountOverflow("leaf end"))?;
                canonical_source[start..end].to_vec()
            } else {
                temporary_node
                    .representatives
                    .iter()
                    .map(|representative| representative.gaussian)
                    .collect()
            };
            let gaussian_count = u32::try_from(page_gaussians.len())
                .map_err(|_| LodBuildError::CountOverflow("page Gaussians"))?;
            let page_kind = if is_leaf {
                LodPageKind::SourceLeaves
            } else {
                LodPageKind::Representatives
            };

            let children = if temporary_node.children.is_empty() {
                LodIndexRange::empty()
            } else {
                let first = old_to_new[temporary_node.children[0]];
                for (offset, child) in temporary_node.children.iter().copied().enumerate() {
                    if old_to_new[child] != first + offset {
                        return Err(LodBuildError::NonContiguousChildren);
                    }
                }
                LodIndexRange {
                    start: u32::try_from(first)
                        .map_err(|_| LodBuildError::CountOverflow("child index"))?,
                    count: u32::try_from(temporary_node.children.len())
                        .map_err(|_| LodBuildError::CountOverflow("child count"))?,
                }
            };
            let depth = depths[new_index];
            let quality = if max_depth == 0 {
                LodQualityInterval { min: 0.0, max: 1.0 }
            } else {
                let min = f32::from(depth) / f32::from(max_depth);
                let max = if is_leaf {
                    1.0
                } else {
                    f32::from(depth + 1) / f32::from(max_depth)
                };
                LodQualityInterval { min, max }
            };
            nodes.push(GaussianLodNode {
                id: node_id,
                parent: parents[new_index].map(|parent| LodNodeId((parent as u64) + 1)),
                depth,
                bounds: temporary_node.bounds,
                children,
                source: temporary_node.source,
                morton: temporary_node.morton,
                representation: LodPageRange {
                    page: LodPageId::INVALID,
                    offset: 0,
                    count: gaussian_count,
                },
                error: temporary_node.error,
                quality,
                high_fidelity_certificate: temporary_node.high_fidelity_certificate,
            });
            node_page_payloads.push(NodePagePayload {
                depth,
                kind: page_kind,
                gaussians: page_gaussians,
            });
        }

        let physical_page_capacity = self
            .settings
            .leaf_capacity
            .min(PROGRESSIVE_PHYSICAL_PAGE_CAPACITY) as usize;
        let packed = pack_node_pages(
            node_page_payloads,
            physical_page_capacity,
            self.settings.support_sigma,
        )?;
        for (node, representation) in nodes.iter_mut().zip(packed.node_ranges.iter().copied()) {
            node.representation = representation;
        }
        let page_count = u32::try_from(packed.descriptors.len())
            .map_err(|_| LodBuildError::CountOverflow("pages"))?;
        let required_features = LOD_CURRENT_REQUIRED_FEATURES
            | if packed.has_shared_page {
                LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES
            } else {
                0
            };

        let scene_bounds = Some(nodes[0].bounds);
        let max_error = nodes[0].error;
        let coarsest_gaussian_count = u64::from(nodes[0].representation.count);
        let output = PlanarGaussian3dLod {
            manifest: GaussianLodManifest {
                header: GaussianLodManifestHeader {
                    magic: LOD_MANIFEST_MAGIC,
                    manifest_version: LOD_MANIFEST_VERSION,
                    page_schema_version: LOD_PAGE_SCHEMA_VERSION,
                    required_features,
                    source_gaussian_count: canonical_source.len() as u64,
                    stored_gaussian_count: packed.stored_gaussian_count,
                    node_count,
                    page_count,
                },
                scene_bounds,
                roots: vec![LodNodeId(1)],
                nodes,
                pages: packed.descriptors,
                build: GaussianLodBuildMetadata {
                    settings: self.settings,
                    reducer: LodReducerKind::MomentMerge,
                    builder_abi_version: PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION,
                    reducer_version: MOMENT_MERGE_VERSION,
                    source_fingerprint,
                    config_fingerprint: moment_merge_config_fingerprint(self.settings),
                },
                quality: GaussianLodQualityMetadata {
                    max_depth,
                    coarsest_gaussian_count,
                    finest_gaussian_count: canonical_source.len() as u64,
                    max_error,
                },
            },
            pages: packed.pages,
        };
        output.validate().map_err(LodBuildError::Validation)?;
        Ok(output)
    }
}

struct NodePagePayload {
    depth: u16,
    kind: LodPageKind,
    gaussians: Vec<Gaussian3d>,
}

struct PendingNodePage {
    depth: u16,
    kind: LodPageKind,
    gaussians: Vec<Gaussian3d>,
    node_ranges: Vec<(usize, u32, u32)>,
}

struct PackedNodePages {
    descriptors: Vec<LodPageDescriptor>,
    pages: Vec<PlanarGaussian3dPage>,
    node_ranges: Vec<LodPageRange>,
    stored_gaussian_count: u64,
    has_shared_page: bool,
}

/// Packs only compatible logical payloads together. Keeping depth and kind
/// homogeneous lets streaming policy reason about a page without inspecting
/// every node, while `LodPageRange` preserves exact per-node ownership.
fn pack_node_pages(
    payloads: Vec<NodePagePayload>,
    physical_capacity: usize,
    support_sigma: f32,
) -> Result<PackedNodePages, LodBuildError> {
    let payload_count = payloads.len();
    let mut packed = PackedNodePages {
        descriptors: Vec::new(),
        pages: Vec::new(),
        node_ranges: vec![
            LodPageRange {
                page: LodPageId::INVALID,
                offset: 0,
                count: 0,
            };
            payload_count
        ],
        stored_gaussian_count: 0,
        has_shared_page: false,
    };
    let mut pending: Option<PendingNodePage> = None;

    for (node_index, payload) in payloads.into_iter().enumerate() {
        if payload.gaussians.is_empty() {
            return Err(LodBuildError::EmptyReduction);
        }
        if payload.gaussians.len() > physical_capacity {
            return Err(LodBuildError::CountOverflow("physical page capacity"));
        }
        let can_append = pending.as_ref().is_some_and(|page| {
            page.depth == payload.depth
                && page.kind == payload.kind
                && page.gaussians.len() + payload.gaussians.len() <= physical_capacity
        });
        if !can_append {
            if let Some(page) = pending.take() {
                finish_node_page(page, support_sigma, &mut packed)?;
            }
            pending = Some(PendingNodePage {
                depth: payload.depth,
                kind: payload.kind,
                gaussians: Vec::with_capacity(physical_capacity),
                node_ranges: Vec::new(),
            });
        }

        let page = pending.as_mut().unwrap();
        let offset = u32::try_from(page.gaussians.len())
            .map_err(|_| LodBuildError::CountOverflow("page offset"))?;
        let count = u32::try_from(payload.gaussians.len())
            .map_err(|_| LodBuildError::CountOverflow("page Gaussians"))?;
        page.gaussians.extend(payload.gaussians);
        page.node_ranges.push((node_index, offset, count));
    }
    if let Some(page) = pending {
        finish_node_page(page, support_sigma, &mut packed)?;
    }
    debug_assert!(
        packed
            .node_ranges
            .iter()
            .all(|range| range.page.is_valid() && range.count > 0)
    );
    Ok(packed)
}

fn finish_node_page(
    pending: PendingNodePage,
    support_sigma: f32,
    packed: &mut PackedNodePages,
) -> Result<(), LodBuildError> {
    let page_number = u64::try_from(packed.pages.len())
        .map_err(|_| LodBuildError::CountOverflow("page id"))?
        .checked_add(1)
        .ok_or(LodBuildError::CountOverflow("page id"))?;
    let page_id = LodPageId(page_number);
    let gaussian_count = u32::try_from(pending.gaussians.len())
        .map_err(|_| LodBuildError::CountOverflow("page Gaussians"))?;
    for &(node_index, offset, count) in &pending.node_ranges {
        packed.node_ranges[node_index] = LodPageRange {
            page: page_id,
            offset,
            count,
        };
    }
    packed.has_shared_page |= pending.node_ranges.len() > 1;
    packed.stored_gaussian_count = packed
        .stored_gaussian_count
        .checked_add(u64::from(gaussian_count))
        .ok_or(LodBuildError::CountOverflow("stored Gaussians"))?;

    let page = PlanarGaussian3dPage::new(page_id, pending.gaussians);
    let mut page_bounds: Option<LodBounds> = None;
    for gaussian in &page.gaussians {
        let gaussian_bounds = gaussian_support_bounds(gaussian, support_sigma)?;
        page_bounds = Some(match page_bounds {
            Some(current) => current.union(gaussian_bounds),
            None => gaussian_bounds,
        });
    }
    let descriptor = LodPageDescriptor {
        id: page_id,
        kind: pending.kind,
        encoding: LodPageEncoding::F32Planar,
        gaussian_count,
        decoded_len: u64::from(gaussian_count)
            .checked_mul(size_of::<Gaussian3d>() as u64)
            .ok_or(LodBuildError::CountOverflow("decoded page bytes"))?,
        content_hash: page.content_hash(),
        bounds: page_bounds.expect("non-empty logical payloads produce non-empty pages"),
        storage: None,
    };
    packed.descriptors.push(descriptor);
    packed.pages.push(page);
    Ok(())
}

/// Convenience entry point for the deterministic CPU reference builder.
pub fn build_planar_3d_lod(
    cloud: &PlanarGaussian3d,
    settings: GaussianLodBuildSettings,
) -> Result<PlanarGaussian3dLod, LodBuildError> {
    CpuGaussianLodBuilder::new(settings).build(cloud)
}

/// Canonical CPU/GPU support contract for offline LoD construction.
///
/// The sphere uses the largest local scale and therefore conservatively
/// contains the rotated anisotropic Gaussian. Every arithmetic operation is
/// f32 and the radius/endpoints are expanded outward by one ULP, matching the
/// bounded WGSL preprocessor bit-for-bit.
pub fn gaussian_support_bounds(
    gaussian: &Gaussian3d,
    support_sigma: f32,
) -> Result<LodBounds, LodBuildError> {
    validate_gaussian(gaussian)
        .map_err(|field| LodBuildError::InvalidGaussian { index: 0, field })?;
    if !support_sigma.is_finite() || support_sigma <= 0.0 {
        return Err(LodBuildError::InvalidSettings(
            LodBuildSettingsError::SupportSigma(support_sigma),
        ));
    }
    let max_scale = gaussian
        .scale_opacity
        .scale
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    let radius = next_up(support_sigma * max_scale);
    let position = gaussian.position_visibility.position;
    let min = std::array::from_fn(|axis| next_down(position[axis] - radius));
    let max = std::array::from_fn(|axis| next_up(position[axis] + radius));
    if !radius.is_finite() || !min.iter().chain(&max).all(|value| value.is_finite()) {
        return Err(LodBuildError::DerivedNonFinite("Gaussian support"));
    }
    LodBounds::new(min, max).map_err(LodBuildError::InvalidBounds)
}

#[derive(Clone)]
struct MomentAccumulator {
    count: u64,
    weight: f64,
    weighted_position: [f64; 3],
    weighted_second_moment: [[f64; 3]; 3],
    weighted_sh: [f64; SH_COEFF_COUNT],
    weighted_sh_squared: [f64; SH_COEFF_COUNT],
    optical_depth: f64,
    min_opacity: f32,
    max_opacity: f32,
    max_visibility: f32,
    bounds: Option<LodBounds>,
    projected_alpha_mass_sqrt_sum: [[f64; 3]; 3],
    sampled_projected_alpha_mass: [f64; PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
    sampled_support_min: [f64; PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
    sampled_support_max: [f64; PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
    max_source_major_scale: f64,
    max_source_anisotropy: f64,
}

impl MomentAccumulator {
    fn new() -> Self {
        Self {
            count: 0,
            weight: 0.0,
            weighted_position: [0.0; 3],
            weighted_second_moment: [[0.0; 3]; 3],
            weighted_sh: [0.0; SH_COEFF_COUNT],
            weighted_sh_squared: [0.0; SH_COEFF_COUNT],
            optical_depth: 0.0,
            min_opacity: f32::INFINITY,
            max_opacity: f32::NEG_INFINITY,
            max_visibility: f32::NEG_INFINITY,
            bounds: None,
            projected_alpha_mass_sqrt_sum: [[0.0; 3]; 3],
            sampled_projected_alpha_mass: [0.0; PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
            sampled_support_min: [f64::INFINITY; PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
            sampled_support_max: [f64::NEG_INFINITY; PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
            max_source_major_scale: 0.0,
            max_source_anisotropy: 0.0,
        }
    }

    fn add(&mut self, gaussian: &Gaussian3d, support_sigma: f32) -> Result<(), LodBuildError> {
        let bounds = gaussian_support_bounds(gaussian, support_sigma)?;
        self.bounds = Some(match self.bounds {
            Some(current) => current.union(bounds),
            None => bounds,
        });
        self.count = self
            .count
            .checked_add(1)
            .ok_or(LodBuildError::CountOverflow("moment samples"))?;

        let opacity = gaussian.scale_opacity.opacity.clamp(0.0, 1.0);
        let visibility = gaussian.position_visibility.visibility.clamp(0.0, 1.0);
        // The epsilon gives completely transparent records deterministic finite
        // spatial moments while having negligible visual influence.
        let weight = f64::from((opacity * visibility).max(1e-12));
        self.weight += weight;
        let position = gaussian.position_visibility.position.map(f64::from);
        let covariance = gaussian_covariance(gaussian)?;
        let effective_alpha = f64::from(opacity * visibility);
        let projected_area_matrix = symmetric_adjugate(covariance);
        let projected_area_sqrt = symmetric_psd_sqrt(projected_area_matrix)?;
        for (accumulated_row, projected_row) in self
            .projected_alpha_mass_sqrt_sum
            .iter_mut()
            .zip(projected_area_sqrt)
        {
            for (accumulated, projected) in accumulated_row.iter_mut().zip(projected_row) {
                *accumulated += effective_alpha * projected;
            }
        }
        for (index, direction) in PROJECTED_ALPHA_MASS_DIRECTIONS.iter().copied().enumerate() {
            let projected_area = quadratic_form_f64(projected_area_matrix, direction)
                .max(0.0)
                .sqrt();
            self.sampled_projected_alpha_mass[index] += effective_alpha * projected_area;

            let center = dot_f64(position, direction);
            let radius = f64::from(support_sigma)
                * quadratic_form_f64(covariance, direction).max(0.0).sqrt();
            self.sampled_support_min[index] = self.sampled_support_min[index].min(center - radius);
            self.sampled_support_max[index] = self.sampled_support_max[index].max(center + radius);
        }
        let (major_scale, anisotropy) = scale_shape(gaussian.scale_opacity.scale);
        self.max_source_major_scale = self.max_source_major_scale.max(major_scale);
        self.max_source_anisotropy = self.max_source_anisotropy.max(anisotropy);
        for row in 0..3 {
            self.weighted_position[row] += weight * position[row];
            for column in 0..3 {
                self.weighted_second_moment[row][column] +=
                    weight * (covariance[row][column] + position[row] * position[column]);
            }
        }
        for (index, coefficient) in gaussian
            .spherical_harmonic
            .coefficients
            .iter()
            .copied()
            .enumerate()
        {
            let coefficient = f64::from(coefficient);
            self.weighted_sh[index] += weight * coefficient;
            self.weighted_sh_squared[index] += weight * coefficient * coefficient;
        }
        let opacity_for_log = f64::from((opacity * visibility).min(1.0 - f32::EPSILON));
        self.optical_depth += -(-opacity_for_log).ln_1p();
        self.min_opacity = self.min_opacity.min(opacity);
        self.max_opacity = self.max_opacity.max(opacity);
        self.max_visibility = self
            .max_visibility
            .max(gaussian.position_visibility.visibility);
        self.ensure_finite()
    }

    fn combine(&mut self, other: &Self) -> Result<(), LodBuildError> {
        if other.count == 0 {
            return Ok(());
        }
        self.count = self
            .count
            .checked_add(other.count)
            .ok_or(LodBuildError::CountOverflow("moment samples"))?;
        self.weight += other.weight;
        for row in 0..3 {
            self.weighted_position[row] += other.weighted_position[row];
            for column in 0..3 {
                self.weighted_second_moment[row][column] +=
                    other.weighted_second_moment[row][column];
            }
        }
        for index in 0..SH_COEFF_COUNT {
            self.weighted_sh[index] += other.weighted_sh[index];
            self.weighted_sh_squared[index] += other.weighted_sh_squared[index];
        }
        self.optical_depth += other.optical_depth;
        self.min_opacity = self.min_opacity.min(other.min_opacity);
        self.max_opacity = self.max_opacity.max(other.max_opacity);
        self.max_visibility = self.max_visibility.max(other.max_visibility);
        self.bounds = match (self.bounds, other.bounds) {
            (Some(left), Some(right)) => Some(left.union(right)),
            (None, Some(right)) => Some(right),
            (left, None) => left,
        };
        for row in 0..3 {
            for column in 0..3 {
                self.projected_alpha_mass_sqrt_sum[row][column] +=
                    other.projected_alpha_mass_sqrt_sum[row][column];
            }
        }
        for index in 0..PROJECTED_ALPHA_MASS_DIRECTIONS.len() {
            self.sampled_projected_alpha_mass[index] += other.sampled_projected_alpha_mass[index];
            self.sampled_support_min[index] =
                self.sampled_support_min[index].min(other.sampled_support_min[index]);
            self.sampled_support_max[index] =
                self.sampled_support_max[index].max(other.sampled_support_max[index]);
        }
        self.max_source_major_scale = self
            .max_source_major_scale
            .max(other.max_source_major_scale);
        self.max_source_anisotropy = self.max_source_anisotropy.max(other.max_source_anisotropy);
        self.ensure_finite()
    }

    fn ensure_finite(&self) -> Result<(), LodBuildError> {
        if !self.weight.is_finite()
            || !self.optical_depth.is_finite()
            || !self
                .weighted_position
                .iter()
                .chain(self.weighted_second_moment.iter().flatten())
                .chain(self.weighted_sh.iter())
                .chain(self.weighted_sh_squared.iter())
                .chain(self.projected_alpha_mass_sqrt_sum.iter().flatten())
                .chain(self.sampled_projected_alpha_mass.iter())
                .chain(self.sampled_support_min.iter())
                .chain(self.sampled_support_max.iter())
                .chain([&self.max_source_major_scale, &self.max_source_anisotropy])
                .all(|value| value.is_finite())
        {
            return Err(LodBuildError::DerivedNonFinite("moment accumulation"));
        }
        Ok(())
    }

    fn finish(&self, support_sigma: f32) -> Result<MomentMergeResult, LodBuildError> {
        if self.count == 0 || self.weight <= 0.0 {
            return Err(LodBuildError::EmptyReduction);
        }
        let mean = self
            .weighted_position
            .map(|weighted_position| weighted_position / self.weight);
        let mut covariance = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                self.weighted_second_moment[row][column] / self.weight - mean[row] * mean[column]
            })
        });
        // Remove accumulation asymmetry before the symmetric eigensolve.
        for (row, column) in [(0, 1), (0, 2), (1, 2)] {
            let value = 0.5 * (covariance[row][column] + covariance[column][row]);
            covariance[row][column] = value;
            covariance[column][row] = value;
        }
        let (rotation, scale) = covariance_to_rotation_scale(covariance)?;
        let mut coefficients = [0.0; SH_COEFF_COUNT];
        let mut appearance_variance = 0.0;
        for (index, coefficient) in coefficients.iter_mut().enumerate() {
            let mean_coefficient = self.weighted_sh[index] / self.weight;
            *coefficient = checked_f32(mean_coefficient, "merged SH coefficient")?;
            appearance_variance += (self.weighted_sh_squared[index] / self.weight
                - mean_coefficient * mean_coefficient)
                .max(0.0);
        }
        let opacity =
            checked_f32(1.0 - (-self.optical_depth).exp(), "merged opacity")?.clamp(0.0, 1.0);
        let position = [
            checked_f32(mean[0], "merged position")?,
            checked_f32(mean[1], "merged position")?,
            checked_f32(mean[2], "merged position")?,
        ];
        let gaussian = Gaussian3d {
            position_visibility: PositionVisibility {
                position,
                visibility: self.max_visibility,
            },
            spherical_harmonic: SphericalHarmonicCoefficients { coefficients },
            rotation: Rotation { rotation },
            scale_opacity: ScaleOpacity { scale, opacity },
        };
        let representative_covariance = gaussian_covariance(&gaussian)?;
        let raster_risk = moment_merge_raster_risk(
            &gaussian,
            representative_covariance,
            support_sigma,
            self.projected_alpha_mass_sqrt_sum,
            self.sampled_projected_alpha_mass,
            self.sampled_support_min,
            self.sampled_support_max,
            self.max_source_major_scale,
            self.max_source_anisotropy,
        )?;
        let support_bounds = gaussian_support_bounds(&gaussian, support_sigma)?;
        let source_bounds = self
            .bounds
            .ok_or(LodBuildError::DerivedNonFinite("source bounds"))?;
        let geometric = farthest_corner_distance(source_bounds, position)?;
        let appearance = checked_f32(
            (appearance_variance / SH_COEFF_COUNT.max(1) as f64).sqrt(),
            "appearance error",
        )?;
        let opacity_error = (opacity - self.min_opacity)
            .abs()
            .max((opacity - self.max_opacity).abs());
        let combined = geometric.max(appearance).max(opacity_error);
        Ok(MomentMergeResult {
            gaussian,
            support_bounds,
            error: LodError {
                geometric,
                appearance,
                opacity: opacity_error,
                combined,
            },
            source_count: self.count,
            total_weight: self.weight,
            raster_risk,
        })
    }
}

#[derive(Clone)]
struct TempNode {
    children: Vec<usize>,
    source: LodSourceRange,
    morton: LodMortonRange,
    bounds: LodBounds,
    accumulator: MomentAccumulator,
    representatives: Vec<MomentMergeResult>,
    /// Exact number of records submitted when this temporary node is selected.
    /// Leaves use their source count; internal nodes use `representatives.len()`.
    representation_count: usize,
    error: LodError,
    high_fidelity_certificate: f32,
}

#[derive(Clone, Copy)]
struct KeyedGaussian {
    morton: u64,
    gaussian: Gaussian3d,
}

fn empty_lod(settings: GaussianLodBuildSettings) -> PlanarGaussian3dLod {
    PlanarGaussian3dLod {
        manifest: GaussianLodManifest {
            header: GaussianLodManifestHeader {
                magic: LOD_MANIFEST_MAGIC,
                manifest_version: LOD_MANIFEST_VERSION,
                page_schema_version: LOD_PAGE_SCHEMA_VERSION,
                required_features: LOD_CURRENT_REQUIRED_FEATURES,
                source_gaussian_count: 0,
                stored_gaussian_count: 0,
                node_count: 0,
                page_count: 0,
            },
            scene_bounds: None,
            roots: Vec::new(),
            nodes: Vec::new(),
            pages: Vec::new(),
            build: GaussianLodBuildMetadata {
                settings,
                reducer: LodReducerKind::MomentMerge,
                builder_abi_version: PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION,
                reducer_version: MOMENT_MERGE_VERSION,
                source_fingerprint: StableHasher::new().finish(),
                config_fingerprint: moment_merge_config_fingerprint(settings),
            },
            quality: GaussianLodQualityMetadata::default(),
        },
        pages: Vec::new(),
    }
}

fn progressive_moment_merge_representatives(
    source: &[Gaussian3d],
    representative_count: usize,
    support_sigma: f32,
) -> Result<ProgressiveMomentMergeRung, LodBuildError> {
    if source.is_empty() || representative_count == 0 || representative_count > source.len() {
        return Err(LodBuildError::EmptyReduction);
    }

    // The ordinary rung immediately above the leaf bridge still operates in a
    // small reduction-ratio regime. Greedily agglomerating adjacent Morton
    // intervals there avoids forcing a balanced interval across a scene gap or
    // fold while preserving exact cardinality and contiguous source coverage.
    // The payload follows that safer partition, but its selection metadata is
    // conservatively enveloped by a balanced partition over the same
    // source/count. The raster certificate is intentionally one-sided:
    // optimizing it can remove elongation without proving compositing PSNR.
    // Keeping that bounds/error/certificate envelope prevents a payload
    // improvement from silently loosening runtime selection. Coarser rungs keep
    // the linear balanced reducer so preprocessing remains bounded when one
    // representative summarizes a large source interval.
    if source.len().div_ceil(representative_count)
        <= PROGRESSIVE_RISK_AWARE_MAX_SOURCES_PER_REPRESENTATIVE
    {
        let representatives = risk_aware_progressive_moment_merge_representatives(
            source,
            representative_count,
            support_sigma,
        )?;
        let balanced_oracle = balanced_progressive_moment_merge_representatives(
            source,
            representative_count,
            support_sigma,
        )?;
        let policy_envelope = progressive_selection_envelope(&balanced_oracle);
        return Ok(ProgressiveMomentMergeRung {
            representatives,
            policy_envelope,
        });
    }

    Ok(ProgressiveMomentMergeRung {
        representatives: balanced_progressive_moment_merge_representatives(
            source,
            representative_count,
            support_sigma,
        )?,
        policy_envelope: ProgressiveSelectionEnvelope::IDENTITY,
    })
}

struct ProgressiveMomentMergeRung {
    representatives: Vec<MomentMergeResult>,
    /// Independent conservative selection-policy envelope. This is separate
    /// from the metadata of the emitted payload so clustering cannot grade its
    /// own optimization as a runtime fidelity improvement.
    policy_envelope: ProgressiveSelectionEnvelope,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ProgressiveSelectionEnvelope {
    support_bounds: Option<LodBounds>,
    error: LodError,
    high_fidelity_certificate_cap: f32,
}

impl ProgressiveSelectionEnvelope {
    const IDENTITY: Self = Self {
        support_bounds: None,
        error: LodError::ZERO,
        high_fidelity_certificate_cap: 1.0,
    };
}

fn progressive_selection_envelope(
    representatives: &[MomentMergeResult],
) -> ProgressiveSelectionEnvelope {
    representatives.iter().fold(
        ProgressiveSelectionEnvelope::IDENTITY,
        |mut envelope, representative| {
            envelope.support_bounds = Some(
                envelope
                    .support_bounds
                    .map_or(representative.support_bounds, |bounds| {
                        bounds.union(representative.support_bounds)
                    }),
            );
            envelope.error = envelope.error.max(representative.error);
            envelope.high_fidelity_certificate_cap = envelope
                .high_fidelity_certificate_cap
                .min(representative.high_fidelity_certificate());
            envelope
        },
    )
}

fn balanced_progressive_moment_merge_representatives(
    source: &[Gaussian3d],
    representative_count: usize,
    support_sigma: f32,
) -> Result<Vec<MomentMergeResult>, LodBuildError> {
    let ranges = balanced_ranges_for_group_count(source.len(), representative_count);
    let mut representatives = Vec::with_capacity(representative_count);
    for range in ranges {
        let mut accumulator = MomentAccumulator::new();
        for gaussian in &source[range] {
            accumulator.add(gaussian, support_sigma)?;
        }
        representatives.push(accumulator.finish(support_sigma)?);
    }
    Ok(representatives)
}

#[derive(Clone)]
struct ProgressiveAgglomerationCluster {
    source_start: usize,
    source_end: usize,
    accumulator: MomentAccumulator,
    previous: Option<usize>,
    next: Option<usize>,
    generation: usize,
    active: bool,
}

#[derive(Clone, Copy, Debug)]
struct ProgressiveAgglomerationCandidate {
    certificate: f32,
    merged_source_count: usize,
    source_start: usize,
    left: usize,
    right: usize,
    left_generation: usize,
    right_generation: usize,
}

impl ProgressiveAgglomerationCandidate {
    fn is_current(self, clusters: &[ProgressiveAgglomerationCluster]) -> bool {
        let Some(left) = clusters.get(self.left) else {
            return false;
        };
        let Some(right) = clusters.get(self.right) else {
            return false;
        };
        left.active
            && right.active
            && left.generation == self.left_generation
            && right.generation == self.right_generation
            && left.next == Some(self.right)
            && right.previous == Some(self.left)
    }
}

impl PartialEq for ProgressiveAgglomerationCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.certificate.total_cmp(&other.certificate) == Ordering::Equal
            && self.merged_source_count == other.merged_source_count
            && self.source_start == other.source_start
            && self.left == other.left
            && self.right == other.right
            && self.left_generation == other.left_generation
            && self.right_generation == other.right_generation
    }
}

impl Eq for ProgressiveAgglomerationCandidate {}

impl PartialOrd for ProgressiveAgglomerationCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProgressiveAgglomerationCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.certificate
            .total_cmp(&other.certificate)
            // BinaryHeap is a max-heap: reverse the integer comparisons so a
            // smaller merged interval, then an earlier Morton/source start,
            // wins when certificates tie.
            .then_with(|| other.merged_source_count.cmp(&self.merged_source_count))
            .then_with(|| other.source_start.cmp(&self.source_start))
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
            .then_with(|| other.left_generation.cmp(&self.left_generation))
            .then_with(|| other.right_generation.cmp(&self.right_generation))
    }
}

fn progressive_agglomeration_candidate(
    clusters: &[ProgressiveAgglomerationCluster],
    left: usize,
    right: usize,
    support_sigma: f32,
) -> Result<ProgressiveAgglomerationCandidate, LodBuildError> {
    debug_assert!(clusters[left].active && clusters[right].active);
    debug_assert_eq!(clusters[left].next, Some(right));
    debug_assert_eq!(clusters[right].previous, Some(left));
    debug_assert_eq!(clusters[left].source_end, clusters[right].source_start);
    let mut accumulator = clusters[left].accumulator.clone();
    accumulator.combine(&clusters[right].accumulator)?;
    let certificate = accumulator
        .finish(support_sigma)?
        .high_fidelity_certificate();
    Ok(ProgressiveAgglomerationCandidate {
        certificate,
        merged_source_count: clusters[right].source_end - clusters[left].source_start,
        source_start: clusters[left].source_start,
        left,
        right,
        left_generation: clusters[left].generation,
        right_generation: clusters[right].generation,
    })
}

fn push_progressive_agglomeration_candidate(
    heap: &mut BinaryHeap<ProgressiveAgglomerationCandidate>,
    clusters: &[ProgressiveAgglomerationCluster],
    left: usize,
    support_sigma: f32,
) -> Result<(), LodBuildError> {
    if let Some(right) = clusters[left].next {
        heap.push(progressive_agglomeration_candidate(
            clusters,
            left,
            right,
            support_sigma,
        )?);
    }
    Ok(())
}

fn risk_aware_progressive_moment_merge_representatives(
    source: &[Gaussian3d],
    representative_count: usize,
    support_sigma: f32,
) -> Result<Vec<MomentMergeResult>, LodBuildError> {
    let mut clusters = Vec::with_capacity(source.len());
    for (index, gaussian) in source.iter().enumerate() {
        let mut accumulator = MomentAccumulator::new();
        accumulator.add(gaussian, support_sigma)?;
        clusters.push(ProgressiveAgglomerationCluster {
            source_start: index,
            source_end: index + 1,
            accumulator,
            previous: index.checked_sub(1),
            next: (index + 1 < source.len()).then_some(index + 1),
            generation: 0,
            active: true,
        });
    }

    let mut heap = BinaryHeap::with_capacity(source.len().saturating_sub(1));
    for left in 0..source.len().saturating_sub(1) {
        push_progressive_agglomeration_candidate(&mut heap, &clusters, left, support_sigma)?;
    }

    let mut active_count = source.len();
    while active_count > representative_count {
        let candidate = loop {
            let candidate = heap.pop().ok_or(LodBuildError::CountOverflow(
                "risk-aware progressive agglomeration",
            ))?;
            if candidate.is_current(&clusters) {
                break candidate;
            }
        };
        let left = candidate.left;
        let right = candidate.right;
        let previous = clusters[left].previous;
        let next = clusters[right].next;

        let mut accumulator = clusters[left].accumulator.clone();
        accumulator.combine(&clusters[right].accumulator)?;
        clusters[left].accumulator = accumulator;
        clusters[left].source_end = clusters[right].source_end;
        clusters[left].next = next;
        clusters[left].generation =
            clusters[left]
                .generation
                .checked_add(1)
                .ok_or(LodBuildError::CountOverflow(
                    "risk-aware agglomeration generation",
                ))?;
        clusters[right].active = false;
        clusters[right].previous = None;
        clusters[right].next = None;
        if let Some(next) = next {
            clusters[next].previous = Some(left);
        }
        active_count -= 1;

        if let Some(previous) = previous {
            push_progressive_agglomeration_candidate(
                &mut heap,
                &clusters,
                previous,
                support_sigma,
            )?;
        }
        push_progressive_agglomeration_candidate(&mut heap, &clusters, left, support_sigma)?;
    }

    let mut representatives = Vec::with_capacity(representative_count);
    let mut cursor = Some(0_usize);
    let mut expected_source_start = 0_usize;
    for _ in 0..representative_count {
        let index = cursor.ok_or(LodBuildError::CountOverflow(
            "risk-aware progressive representative partition",
        ))?;
        let cluster = &clusters[index];
        if !cluster.active || cluster.source_start != expected_source_start {
            return Err(LodBuildError::CountOverflow(
                "risk-aware progressive representative partition",
            ));
        }
        representatives.push(cluster.accumulator.finish(support_sigma)?);
        expected_source_start = cluster.source_end;
        cursor = cluster.next;
    }
    if cursor.is_some() || expected_source_start != source.len() {
        return Err(LodBuildError::CountOverflow(
            "risk-aware progressive representative partition",
        ));
    }
    Ok(representatives)
}

struct DeepestRepresentationChoice {
    representatives: Vec<MomentMergeResult>,
}

impl DeepestRepresentationChoice {
    fn into_representatives(self) -> Vec<MomentMergeResult> {
        self.representatives
    }
}

struct DeepestPairingPlan {
    node_key: usize,
    source_start: usize,
    source_end: usize,
    base_pair_count: usize,
    pair_count: usize,
    base_quality: PairingQualityScore,
    adjusted_quality: Option<PairingQualityScore>,
}

#[derive(Clone, Copy, Debug)]
struct PairingQualityScore {
    minimum_certificate: f32,
    certificate_sum: f64,
}

impl PartialEq for PairingQualityScore {
    fn eq(&self, other: &Self) -> bool {
        self.minimum_certificate
            .total_cmp(&other.minimum_certificate)
            == Ordering::Equal
            && self.certificate_sum.total_cmp(&other.certificate_sum) == Ordering::Equal
    }
}

impl Eq for PairingQualityScore {}

impl PartialOrd for PairingQualityScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PairingQualityScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.minimum_certificate
            .total_cmp(&other.minimum_certificate)
            .then_with(|| self.certificate_sum.total_cmp(&other.certificate_sum))
    }
}

impl DeepestPairingPlan {
    fn representation_count(&self) -> usize {
        (self.source_end - self.source_start).saturating_sub(self.pair_count)
    }

    fn quality(&self) -> PairingQualityScore {
        if self.pair_count == self.base_pair_count {
            self.base_quality
        } else {
            self.adjusted_quality
                .expect("a globally adjusted bridge has a precomputed quality score")
        }
    }

    fn next_adjustment(&self, choice_index: usize) -> Option<PairingAdjustment> {
        if self.pair_count != self.base_pair_count {
            return None;
        }
        Some(PairingAdjustment {
            resulting_quality: self.adjusted_quality?,
            choice_index,
            next_pair_count: self.base_pair_count + 1,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct PairingAdjustment {
    resulting_quality: PairingQualityScore,
    choice_index: usize,
    next_pair_count: usize,
}

impl PartialEq for PairingAdjustment {
    fn eq(&self, other: &Self) -> bool {
        self.resulting_quality == other.resulting_quality
            && self.choice_index == other.choice_index
            && self.next_pair_count == other.next_pair_count
    }
}

impl Eq for PairingAdjustment {}

impl PartialOrd for PairingAdjustment {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PairingAdjustment {
    fn cmp(&self, other: &Self) -> Ordering {
        self.resulting_quality
            .cmp(&other.resulting_quality)
            // Prefer the earlier Morton domain when risk scores tie.
            .then_with(|| other.choice_index.cmp(&self.choice_index))
            .then_with(|| other.next_pair_count.cmp(&self.next_pair_count))
    }
}

/// Precompute a variable-rate bridge immediately above exact leaves. Every
/// bridge representative is either an exact source record or a two-record
/// adjacent-Morton MomentMerge. Certified pairs are retained whenever they
/// already save at least 1/8 of the source records. Otherwise an exact-cardinality
/// path DP selects the least-risk 1/8 pairing instead of collapsing the domain
/// through the ordinary 8:1 reducer.
///
/// The analytic 7/8 bridge rate can exceed the strict 2N stored-record budget
/// by a small amount after integer-rounded coarser rungs. A global deterministic
/// heap adds the least-risk next adjacent pair until the exact projected
/// hierarchy fits, then removes unnecessary tail adjustments without ever
/// weakening the certificate attached to the resulting payload.
fn plan_high_fidelity_deepest_choices(
    canonical_source: &[Gaussian3d],
    temporary: &[TempNode],
    leaf_level: &[usize],
    support_sigma: f32,
    branching_factor: usize,
) -> Result<HashMap<usize, DeepestRepresentationChoice>, LodBuildError> {
    let paired_len = leaf_level.len() / 2 * 2;
    let mut plans = Vec::with_capacity(paired_len / 2);
    for children in leaf_level[..paired_len].chunks_exact(2) {
        let first = &temporary[children[0]];
        let last = &temporary[children[1]];
        let source_start = usize::try_from(first.source.start)
            .map_err(|_| LodBuildError::CountOverflow("internal source start"))?;
        let source_end = usize::try_from(last.source.end().unwrap())
            .map_err(|_| LodBuildError::CountOverflow("internal source end"))?;
        let source = &canonical_source[source_start..source_end];
        let pair_certificates = adjacent_pair_certificates(source, support_sigma)?;
        let certified = maximum_certified_pairing_score(&pair_certificates);
        let maximum_representative_count = source
            .len()
            .saturating_mul(HIGH_FIDELITY_MAX_REPRESENTATIVE_NUMERATOR)
            / HIGH_FIDELITY_MAX_REPRESENTATIVE_DENOMINATOR;
        let minimum_pair_count = source
            .len()
            .saturating_sub(maximum_representative_count)
            .max(source.len().div_ceil(branching_factor))
            .min(source.len() / 2);
        let base_pair_count = certified.merge_count.max(minimum_pair_count);
        let certified_only = certified.merge_count >= minimum_pair_count;
        let base_quality = optimal_pairing_quality_score(
            &pair_certificates,
            base_pair_count,
            certified_only.then_some(HIGH_FIDELITY_PAIR_CERTIFICATE),
        )
        .ok_or(LodBuildError::CountOverflow("deepest bridge pairing"))?;
        let adjusted_quality = if base_pair_count < source.len() / 2 {
            optimal_pairing_quality_score(&pair_certificates, base_pair_count + 1, None)
        } else {
            None
        };
        plans.push(DeepestPairingPlan {
            node_key: children[0],
            source_start,
            source_end,
            base_pair_count,
            pair_count: base_pair_count,
            base_quality,
            adjusted_quality,
        });
    }

    let projected_storage = |plans: &[DeepestPairingPlan]| {
        let mut stored = canonical_source.len();
        let mut level =
            Vec::with_capacity(plans.len() + usize::from(paired_len < leaf_level.len()));
        for plan in plans {
            let count = plan.representation_count();
            stored = stored.checked_add(count)?;
            level.push(count);
        }
        if let Some(carried) = leaf_level.get(paired_len) {
            level.push(temporary[*carried].representation_count);
        }
        while level.len() > 1 {
            let paired = level.len() / 2 * 2;
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            for pair in level[..paired].chunks_exact(2) {
                let count = pair[0]
                    .checked_add(pair[1])?
                    .div_ceil(branching_factor)
                    .max(1);
                stored = stored.checked_add(count)?;
                next.push(count);
            }
            if let Some(carried) = level.get(paired) {
                next.push(*carried);
            }
            level = next;
        }
        Some(stored)
    };

    let maximum_storage = canonical_source
        .len()
        .checked_mul(2)
        .ok_or(LodBuildError::CountOverflow("progressive storage budget"))?;
    let proposed_storage = projected_storage(&plans)
        .ok_or(LodBuildError::CountOverflow("progressive stored Gaussians"))?;
    let required_adjustments = proposed_storage.saturating_sub(maximum_storage);
    let mut adjustments = BinaryHeap::new();
    for (choice_index, plan) in plans.iter().enumerate() {
        if let Some(adjustment) = plan.next_adjustment(choice_index) {
            adjustments.push(adjustment);
        }
    }
    let mut applied_adjustments = Vec::with_capacity(required_adjustments);
    for _ in 0..required_adjustments {
        let adjustment = adjustments
            .pop()
            .ok_or(LodBuildError::CountOverflow("progressive storage budget"))?;
        let plan = &mut plans[adjustment.choice_index];
        if adjustment.next_pair_count != plan.pair_count + 1 {
            return Err(LodBuildError::CountOverflow(
                "progressive pairing adjustment",
            ));
        }
        plan.pair_count = adjustment.next_pair_count;
        applied_adjustments.push(adjustment.choice_index);
    }

    let adjusted_storage = projected_storage(&plans)
        .ok_or(LodBuildError::CountOverflow("progressive stored Gaussians"))?;
    if adjusted_storage > maximum_storage {
        return Err(LodBuildError::CountOverflow("progressive storage budget"));
    }

    // One deepest-record reduction can also cross an integer boundary at one
    // or more coarser rungs. Binary-search the least-risk prefix which keeps
    // the exact projected hierarchy inside the cap.
    let mut lower = 0;
    let mut upper = applied_adjustments.len();
    while lower < upper {
        let middle = lower + (upper - lower) / 2;
        for plan in &mut plans {
            plan.pair_count = plan.base_pair_count;
        }
        for &choice_index in &applied_adjustments[..middle] {
            plans[choice_index].pair_count += 1;
        }
        let candidate_storage = projected_storage(&plans)
            .ok_or(LodBuildError::CountOverflow("progressive stored Gaussians"))?;
        if candidate_storage <= maximum_storage {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    for plan in &mut plans {
        plan.pair_count = plan.base_pair_count;
    }
    for &choice_index in &applied_adjustments[..lower] {
        plans[choice_index].pair_count += 1;
    }
    let final_storage = projected_storage(&plans)
        .ok_or(LodBuildError::CountOverflow("progressive stored Gaussians"))?;
    if final_storage > maximum_storage {
        return Err(LodBuildError::CountOverflow("progressive storage budget"));
    }

    let mut choices = HashMap::with_capacity(plans.len());
    for plan in plans {
        let source = &canonical_source[plan.source_start..plan.source_end];
        let representatives = paired_leaf_representatives(
            source,
            support_sigma,
            plan.pair_count,
            Some(plan.quality().minimum_certificate),
        )?;
        debug_assert_eq!(representatives.len(), plan.representation_count());
        debug_assert!(
            representatives
                .iter()
                .all(|representative| representative.source_count <= 2)
        );
        choices.insert(
            plan.node_key,
            DeepestRepresentationChoice { representatives },
        );
    }
    Ok(choices)
}

#[derive(Clone, Copy, Default)]
struct LeafPairingScore {
    merge_count: usize,
    certificate_sum: f64,
}

impl LeafPairingScore {
    fn with_pair(self, certificate: f32) -> Self {
        Self {
            merge_count: self.merge_count + 1,
            certificate_sum: self.certificate_sum + f64::from(certificate),
        }
    }

    fn cmp(self, other: Self) -> Ordering {
        self.merge_count
            .cmp(&other.merge_count)
            .then_with(|| self.certificate_sum.total_cmp(&other.certificate_sum))
    }
}

fn adjacent_pair_candidates(
    source: &[Gaussian3d],
    support_sigma: f32,
) -> Result<Vec<MomentMergeResult>, LodBuildError> {
    if source.len() < 2 {
        return Err(LodBuildError::EmptyReduction);
    }

    let mut candidates = Vec::with_capacity(source.len() - 1);
    for pair in source.windows(2) {
        let mut accumulator = MomentAccumulator::new();
        accumulator.add(&pair[0], support_sigma)?;
        accumulator.add(&pair[1], support_sigma)?;
        candidates.push(accumulator.finish(support_sigma)?);
    }
    Ok(candidates)
}

fn adjacent_pair_certificates(
    source: &[Gaussian3d],
    support_sigma: f32,
) -> Result<Vec<f32>, LodBuildError> {
    Ok(adjacent_pair_candidates(source, support_sigma)?
        .iter()
        .map(MomentMergeResult::high_fidelity_certificate)
        .collect())
}

/// Maximum-cardinality matching over the certified edges of an adjacent-pair
/// path. Total certificate breaks cardinality ties deterministically.
fn maximum_certified_pairing_score(certificates: &[f32]) -> LeafPairingScore {
    let source_count = certificates.len() + 1;
    let mut suffix_scores = vec![LeafPairingScore::default(); source_count + 1];
    for index in (0..source_count).rev() {
        let skip = suffix_scores[index + 1];
        let pair = certificates.get(index).and_then(|certificate| {
            (*certificate >= HIGH_FIDELITY_PAIR_CERTIFICATE)
                .then_some(suffix_scores[(index + 2).min(source_count)].with_pair(*certificate))
        });
        suffix_scores[index] = if pair.is_some_and(|pair| pair.cmp(skip) != Ordering::Less) {
            pair.unwrap()
        } else {
            skip
        };
    }
    suffix_scores[0]
}

/// Lexicographically optimize an exact-cardinality path matching: maximize its
/// worst certificate first, then maximize total certificate at that bottleneck.
/// The first DP obtains the exact bottleneck for every suffix/count state; the
/// second is an ordinary maximum-weight matching restricted to edges at or
/// above that bottleneck. Exact carries are neutral and do not enter the score.
fn optimal_pairing_quality_score(
    certificates: &[f32],
    pair_count: usize,
    minimum_certificate: Option<f32>,
) -> Option<PairingQualityScore> {
    let bottleneck =
        maximum_pairing_bottleneck(certificates, pair_count, minimum_certificate.unwrap_or(0.0))?;
    let certificate_sum = pairing_score_table(certificates, Some(bottleneck))[0]
        .get(pair_count)
        .copied()?;
    certificate_sum.is_finite().then_some(PairingQualityScore {
        minimum_certificate: bottleneck,
        certificate_sum,
    })
}

fn maximum_pairing_bottleneck(
    certificates: &[f32],
    pair_count: usize,
    minimum_certificate: f32,
) -> Option<f32> {
    let source_count = certificates.len() + 1;
    let maximum_pairs = source_count / 2;
    if pair_count > maximum_pairs {
        return None;
    }
    let mut scores = vec![vec![-1.0_f32; maximum_pairs + 1]; source_count + 1];
    scores[source_count][0] = 1.0;
    for index in (0..source_count).rev() {
        for count in 0..=maximum_pairs {
            let skip = scores[index + 1][count];
            let take = if count > 0
                && let Some(certificate) = certificates.get(index)
                && *certificate >= minimum_certificate
            {
                let suffix = scores[(index + 2).min(source_count)][count - 1];
                if suffix >= 0.0 {
                    suffix.min(*certificate)
                } else {
                    -1.0
                }
            } else {
                -1.0
            };
            scores[index][count] = if take.total_cmp(&skip) != Ordering::Less {
                take
            } else {
                skip
            };
        }
    }
    (scores[0][pair_count] >= 0.0).then_some(scores[0][pair_count])
}

fn pairing_score_table(certificates: &[f32], minimum_certificate: Option<f32>) -> Vec<Vec<f64>> {
    let source_count = certificates.len() + 1;
    let maximum_pairs = source_count / 2;
    let mut scores = vec![vec![f64::NEG_INFINITY; maximum_pairs + 1]; source_count + 1];
    scores[source_count][0] = 0.0;
    for index in (0..source_count).rev() {
        for pair_count in 0..=maximum_pairs {
            let skip = scores[index + 1][pair_count];
            let take = if pair_count > 0
                && let Some(certificate) = certificates.get(index)
                && minimum_certificate.is_none_or(|minimum| *certificate >= minimum)
            {
                let suffix = scores[(index + 2).min(source_count)][pair_count - 1];
                if suffix.is_finite() {
                    suffix + f64::from(*certificate)
                } else {
                    f64::NEG_INFINITY
                }
            } else {
                f64::NEG_INFINITY
            };
            scores[index][pair_count] = if take.total_cmp(&skip) != Ordering::Less {
                take
            } else {
                skip
            };
        }
    }
    scores
}

fn optimal_pairing_indices(
    certificates: &[f32],
    pair_count: usize,
    minimum_certificate: Option<f32>,
) -> Option<Vec<usize>> {
    let scores = pairing_score_table(certificates, minimum_certificate);
    if !scores[0].get(pair_count).copied()?.is_finite() {
        return None;
    }
    let source_count = certificates.len() + 1;
    let mut selected = Vec::with_capacity(pair_count);
    let mut index = 0;
    let mut remaining = pair_count;
    while remaining > 0 && index < certificates.len() {
        let skip = scores[index + 1][remaining];
        let certificate = certificates[index];
        let take = if minimum_certificate.is_none_or(|minimum| certificate >= minimum) {
            let suffix = scores[(index + 2).min(source_count)][remaining - 1];
            if suffix.is_finite() {
                suffix + f64::from(certificate)
            } else {
                f64::NEG_INFINITY
            }
        } else {
            f64::NEG_INFINITY
        };
        if take.is_finite() && take.total_cmp(&skip) != Ordering::Less {
            selected.push(index);
            remaining -= 1;
            index += 2;
        } else {
            index += 1;
        }
    }
    (remaining == 0).then_some(selected)
}

/// Materialize one exact-cardinality adjacent-pair plan. Unmatched source
/// records pass through byte-exact; no bridge representative ever summarizes
/// more than two source records.
fn paired_leaf_representatives(
    source: &[Gaussian3d],
    support_sigma: f32,
    pair_count: usize,
    minimum_certificate: Option<f32>,
) -> Result<Vec<MomentMergeResult>, LodBuildError> {
    let mut candidates = adjacent_pair_candidates(source, support_sigma)?
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();
    let certificates = candidates
        .iter()
        .map(|candidate| candidate.as_ref().unwrap().high_fidelity_certificate())
        .collect::<Vec<_>>();
    let selected = optimal_pairing_indices(&certificates, pair_count, minimum_certificate)
        .ok_or(LodBuildError::CountOverflow("deepest bridge pairing"))?;
    let mut take_pair = vec![false; source.len()];
    for index in selected {
        take_pair[index] = true;
    }

    let mut representatives = Vec::with_capacity(source.len().saturating_sub(pair_count));
    let mut index = 0;
    while index < source.len() {
        if take_pair[index] {
            representatives.push(
                candidates[index]
                    .take()
                    .expect("the pairing plan only selects certified candidates"),
            );
            index += 2;
        } else {
            representatives.push(exact_source_representative(source[index], support_sigma)?);
            index += 1;
        }
    }
    debug_assert_eq!(representatives.len(), source.len() - pair_count);
    if let Some(minimum_certificate) = minimum_certificate {
        debug_assert!(representatives.iter().all(|representative| {
            representative.high_fidelity_certificate() >= minimum_certificate
        }));
    }
    Ok(representatives)
}

#[cfg(test)]
fn high_fidelity_leaf_representatives(
    source: &[Gaussian3d],
    support_sigma: f32,
) -> Result<Vec<MomentMergeResult>, LodBuildError> {
    let certificates = adjacent_pair_certificates(source, support_sigma)?;
    let certified = maximum_certified_pairing_score(&certificates);
    paired_leaf_representatives(
        source,
        support_sigma,
        certified.merge_count,
        Some(HIGH_FIDELITY_PAIR_CERTIFICATE),
    )
}

fn exact_source_representative(
    gaussian: Gaussian3d,
    support_sigma: f32,
) -> Result<MomentMergeResult, LodBuildError> {
    let opacity = gaussian.scale_opacity.opacity.clamp(0.0, 1.0);
    let visibility = gaussian.position_visibility.visibility.clamp(0.0, 1.0);
    Ok(MomentMergeResult {
        gaussian,
        support_bounds: gaussian_support_bounds(&gaussian, support_sigma)?,
        error: LodError::ZERO,
        source_count: 1,
        total_weight: f64::from((opacity * visibility).max(1e-12)),
        raster_risk: MomentMergeRasterRisk {
            sampled_projected_alpha_mass_inflation: 1.0,
            projected_alpha_mass_inflation_upper_bound: 1.0,
            support_leakage_fraction: 0.0,
            support_growth: 1.0,
            major_scale_growth: 1.0,
            anisotropy_growth: 1.0,
        },
    })
}

fn validate_plane_lengths(cloud: &PlanarGaussian3d) -> Result<(), LodBuildError> {
    let expected = cloud.position_visibility.len();
    let lengths = [
        ("spherical_harmonic", cloud.spherical_harmonic.len()),
        ("rotation", cloud.rotation.len()),
        ("scale_opacity", cloud.scale_opacity.len()),
    ];
    for (plane, actual) in lengths {
        if actual != expected {
            return Err(LodBuildError::PlaneLengthMismatch {
                plane,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

type BreadthFirstLayout = (Vec<usize>, Vec<Option<usize>>, Vec<u16>);

fn breadth_first_order(
    nodes: &[TempNode],
    root: usize,
) -> Result<BreadthFirstLayout, LodBuildError> {
    let mut order = vec![root];
    let mut parents = vec![None];
    let mut depths = vec![0_u16];
    let mut cursor = 0;
    while cursor < order.len() {
        let depth = depths[cursor];
        for child in &nodes[order[cursor]].children {
            order.push(*child);
            parents.push(Some(cursor));
            depths.push(
                depth
                    .checked_add(1)
                    .ok_or(LodBuildError::CountOverflow("hierarchy depth"))?,
            );
        }
        cursor += 1;
    }
    Ok((order, parents, depths))
}

fn balanced_ranges(
    len: usize,
    maximum_group_size: usize,
    force_at_least_two_groups: bool,
) -> Vec<std::ops::Range<usize>> {
    debug_assert!(len > 0);
    debug_assert!(maximum_group_size > 0);
    let mut group_count = len.div_ceil(maximum_group_size);
    if force_at_least_two_groups {
        group_count = group_count.max(2).min(len);
    }
    balanced_ranges_for_group_count(len, group_count)
}

fn balanced_ranges_for_group_count(len: usize, group_count: usize) -> Vec<std::ops::Range<usize>> {
    debug_assert!(len > 0);
    debug_assert!((1..=len).contains(&group_count));
    let base = len / group_count;
    let remainder = len % group_count;
    let mut ranges = Vec::with_capacity(group_count);
    let mut start = 0;
    for group in 0..group_count {
        let count = base + usize::from(group < remainder);
        ranges.push(start..start + count);
        start += count;
    }
    ranges
}

fn source_center_bounds(source: &[Gaussian3d]) -> Result<LodBounds, LodBuildError> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for gaussian in source {
        for axis in 0..3 {
            min[axis] = min[axis].min(gaussian.position_visibility.position[axis]);
            max[axis] = max[axis].max(gaussian.position_visibility.position[axis]);
        }
    }
    let bounds = LodBounds::new(min, max).map_err(LodBuildError::InvalidBounds)?;
    if bounds
        .extent()
        .into_iter()
        .any(|extent| !extent.is_finite())
    {
        return Err(LodBuildError::DerivedNonFinite(
            "Morton normalization extent",
        ));
    }
    Ok(bounds)
}

/// Canonical 63-bit Morton key shared by CPU and GPU offline preprocessing.
/// Quantization deliberately uses only f32 operations to match WGSL.
pub fn canonical_lod_morton_code(position: [f32; 3], bounds: LodBounds) -> u64 {
    let mut quantized = [0_u32; 3];
    for axis in 0..3 {
        let extent = bounds.max[axis] - bounds.min[axis];
        if extent > 0.0 {
            let normalized = ((position[axis] - bounds.min[axis]) / extent).clamp(0.0, 1.0);
            quantized[axis] = (normalized * LOD_MORTON_AXIS_MAX as f32).floor() as u32;
        }
    }
    interleave_morton(quantized[0], quantized[1], quantized[2])
}

fn interleave_morton(x: u32, y: u32, z: u32) -> u64 {
    let mut code = 0_u64;
    for bit in 0..LOD_MORTON_BITS_PER_AXIS {
        code |= u64::from((x >> bit) & 1) << (3 * bit);
        code |= u64::from((y >> bit) & 1) << (3 * bit + 1);
        code |= u64::from((z >> bit) & 1) << (3 * bit + 2);
    }
    code
}

fn source_fingerprint(source: &[KeyedGaussian]) -> u64 {
    let mut hash = StableHasher::new();
    hash.write(&(source.len() as u64).to_le_bytes());
    for entry in source {
        hash.write(&entry.morton.to_le_bytes());
        hash.write(&stable_gaussian_hash(&entry.gaussian).to_le_bytes());
    }
    hash.finish()
}

pub(crate) fn compare_gaussians(left: &Gaussian3d, right: &Gaussian3d) -> Ordering {
    compare_f32_slices(
        &left.position_visibility.position,
        &right.position_visibility.position,
    )
    .then_with(|| {
        canonical_f32(left.position_visibility.visibility)
            .total_cmp(&canonical_f32(right.position_visibility.visibility))
    })
    .then_with(|| {
        compare_f32_slices(
            &left.spherical_harmonic.coefficients,
            &right.spherical_harmonic.coefficients,
        )
    })
    .then_with(|| compare_f32_slices(&left.rotation.rotation, &right.rotation.rotation))
    .then_with(|| compare_f32_slices(&left.scale_opacity.scale, &right.scale_opacity.scale))
    .then_with(|| {
        canonical_f32(left.scale_opacity.opacity)
            .total_cmp(&canonical_f32(right.scale_opacity.opacity))
    })
}

fn compare_f32_slices(left: &[f32], right: &[f32]) -> Ordering {
    left.iter()
        .zip(right)
        .map(|(left, right)| canonical_f32(*left).total_cmp(&canonical_f32(*right)))
        .find(|ordering| !ordering.is_eq())
        .unwrap_or_else(|| left.len().cmp(&right.len()))
}

#[inline]
fn canonical_f32(value: f32) -> f32 {
    if value == 0.0 { 0.0 } else { value }
}

pub(crate) fn canonicalize_gaussian_zeros(mut gaussian: Gaussian3d) -> Gaussian3d {
    gaussian.position_visibility.position =
        gaussian.position_visibility.position.map(canonical_f32);
    gaussian.position_visibility.visibility =
        canonical_f32(gaussian.position_visibility.visibility);
    gaussian.spherical_harmonic.coefficients =
        gaussian.spherical_harmonic.coefficients.map(canonical_f32);
    gaussian.rotation.rotation = gaussian.rotation.rotation.map(canonical_f32);
    gaussian.scale_opacity.scale = gaussian.scale_opacity.scale.map(canonical_f32);
    gaussian.scale_opacity.opacity = canonical_f32(gaussian.scale_opacity.opacity);
    gaussian
}

#[inline]
fn canonical_f32_bits(value: f32) -> u32 {
    canonical_f32(value).to_bits()
}

fn gaussian_covariance(gaussian: &Gaussian3d) -> Result<[[f64; 3]; 3], LodBuildError> {
    let [w, x, y, z] = gaussian.rotation.rotation.map(f64::from);
    let norm = (w * w + x * x + y * y + z * z).sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(LodBuildError::DerivedNonFinite("rotation normalization"));
    }
    let (w, x, y, z) = (w / norm, x / norm, y / norm, z / norm);
    let rotation = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let scale_squared = gaussian
        .scale_opacity
        .scale
        .map(|scale| f64::from(scale) * f64::from(scale));
    let mut covariance = [[0.0; 3]; 3];
    // These literals are the rows of the conventional quaternion matrix Q.
    // The renderer stores them as columns in R=Q^T and evaluates R^T D R,
    // therefore its effective covariance is Q D Q^T.
    for row in 0..3 {
        for column in 0..3 {
            for axis in 0..3 {
                covariance[row][column] +=
                    rotation[row][axis] * scale_squared[axis] * rotation[column][axis];
            }
            if !covariance[row][column].is_finite() {
                return Err(LodBuildError::DerivedNonFinite("Gaussian covariance"));
            }
        }
    }
    Ok(covariance)
}

#[inline]
fn dot_f64(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

#[inline]
fn quadratic_form_f64(matrix: [[f64; 3]; 3], direction: [f64; 3]) -> f64 {
    dot_f64(
        direction,
        [
            dot_f64(matrix[0], direction),
            dot_f64(matrix[1], direction),
            dot_f64(matrix[2], direction),
        ],
    )
}

/// For an orthographic view direction `n`, `sqrt(n^T adj(Sigma) n)` is the
/// area factor of the projected 2D Gaussian covariance. The common `2*pi`
/// factor cancels when representative and source alpha masses are compared.
fn symmetric_adjugate(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[0][2];
    let d = matrix[1][1];
    let e = matrix[1][2];
    let f = matrix[2][2];
    [
        [d * f - e * e, c * e - b * f, b * e - c * d],
        [c * e - b * f, a * f - c * c, b * c - a * e],
        [b * e - c * d, b * c - a * e, a * d - b * b],
    ]
}

/// Deterministic principal square root of a symmetric positive-semidefinite
/// 3x3 matrix. Negative roundoff in analytically PSD inputs is clamped to zero,
/// matching the covariance reconstruction path.
fn symmetric_psd_sqrt(matrix: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], LodBuildError> {
    let scale = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() {
        return Err(LodBuildError::DerivedNonFinite(
            "projected-area PSD square root",
        ));
    }
    if scale == 0.0 {
        return Ok([[0.0; 3]; 3]);
    }
    // The shared Jacobi solve uses an absolute floor in its stopping rule.
    // Normalizing first preserves rotated eigenspaces for very small splats.
    let normalized = matrix.map(|row| row.map(|value| value / scale));
    let (eigenvalues, eigenvectors) = symmetric_eigendecomposition(normalized)?;
    let roots = eigenvalues.map(|value| (value.max(0.0) * scale).sqrt());
    let mut square_root = std::array::from_fn::<_, 3, _>(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|axis| eigenvectors[row][axis] * roots[axis] * eigenvectors[column][axis])
                .sum::<f64>()
        })
    });
    for (row, column) in [(0, 1), (0, 2), (1, 2)] {
        let value = 0.5 * (square_root[row][column] + square_root[column][row]);
        square_root[row][column] = value;
        square_root[column][row] = value;
    }
    for value in square_root.iter_mut().flatten() {
        if !value.is_finite() {
            return Err(LodBuildError::DerivedNonFinite(
                "projected-area PSD square root",
            ));
        }
        if *value == 0.0 {
            *value = 0.0;
        }
    }
    Ok(square_root)
}

fn scale_shape(scale: [f32; 3]) -> (f64, f64) {
    let mut sorted = scale.map(f64::from);
    sorted.sort_unstable_by(f64::total_cmp);
    let minor = sorted[0];
    let major = sorted[2];
    let anisotropy = if major == 0.0 {
        1.0
    } else if minor <= f64::EPSILON {
        f64::from(f32::MAX)
    } else {
        (major / minor).min(f64::from(f32::MAX))
    };
    (major, anisotropy)
}

#[allow(clippy::too_many_arguments)]
fn moment_merge_raster_risk(
    representative: &Gaussian3d,
    representative_covariance: [[f64; 3]; 3],
    support_sigma: f32,
    projected_alpha_mass_sqrt_sum: [[f64; 3]; 3],
    sampled_source_alpha_mass: [f64; PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
    sampled_source_support_min: [f64; PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
    sampled_source_support_max: [f64; PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
    max_source_major_scale: f64,
    max_source_anisotropy: f64,
) -> Result<MomentMergeRasterRisk, LodBuildError> {
    let representative_alpha = f64::from(
        representative.scale_opacity.opacity.clamp(0.0, 1.0)
            * representative
                .position_visibility
                .visibility
                .clamp(0.0, 1.0),
    );
    if representative_alpha == 0.0 && sampled_source_alpha_mass.iter().all(|mass| *mass == 0.0) {
        return Ok(MomentMergeRasterRisk::default());
    }

    let representative_projected_area = symmetric_adjugate(representative_covariance);
    let position = representative.position_visibility.position.map(f64::from);
    let mut sampled_projected_alpha_mass_inflation = 0.0_f64;
    let mut support_leakage_fraction = 0.0_f64;
    let mut support_growth = 1.0_f64;
    for (index, direction) in PROJECTED_ALPHA_MASS_DIRECTIONS.iter().copied().enumerate() {
        let representative_mass = representative_alpha
            * quadratic_form_f64(representative_projected_area, direction)
                .max(0.0)
                .sqrt();
        let source_mass = sampled_source_alpha_mass[index];
        let inflation = if source_mass > f64::EPSILON {
            representative_mass / source_mass
        } else if representative_mass <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        };
        sampled_projected_alpha_mass_inflation =
            sampled_projected_alpha_mass_inflation.max(inflation);

        let center = dot_f64(position, direction);
        let radius = f64::from(support_sigma)
            * quadratic_form_f64(representative_covariance, direction)
                .max(0.0)
                .sqrt();
        let representative_min = center - radius;
        let representative_max = center + radius;
        let representative_span = (representative_max - representative_min).max(0.0);
        let source_min = sampled_source_support_min[index];
        let source_max = sampled_source_support_max[index];
        let source_span = (source_max - source_min).max(0.0);
        let overlap_span =
            (representative_max.min(source_max) - representative_min.max(source_min)).max(0.0);
        let leakage = if representative_span > f64::EPSILON {
            (1.0 - overlap_span / representative_span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        support_leakage_fraction = support_leakage_fraction.max(leakage);
        let span_growth = if source_span > f64::EPSILON {
            representative_span / source_span
        } else if representative_span <= f64::EPSILON {
            1.0
        } else {
            f64::INFINITY
        };
        let retained_support_growth = if representative_span <= f64::EPSILON {
            1.0
        } else if overlap_span > f64::EPSILON {
            representative_span / overlap_span
        } else {
            f64::INFINITY
        };
        support_growth = support_growth.max(span_growth).max(retained_support_growth);
    }

    let (representative_major_scale, representative_anisotropy) =
        scale_shape(representative.scale_opacity.scale);
    let major_scale_growth = ratio_or_infinity(
        representative_major_scale,
        max_source_major_scale,
        f64::EPSILON,
    );
    let anisotropy_growth = ratio_or_infinity(
        representative_anisotropy,
        max_source_anisotropy,
        f64::EPSILON,
    );
    let projected_alpha_mass_inflation_upper_bound = projected_alpha_mass_inflation_upper_bound(
        representative_alpha,
        representative_projected_area,
        projected_alpha_mass_sqrt_sum,
    )?;

    Ok(MomentMergeRasterRisk {
        sampled_projected_alpha_mass_inflation: bounded_risk_f32(
            sampled_projected_alpha_mass_inflation,
            "sampled projected alpha-mass inflation",
        )?,
        projected_alpha_mass_inflation_upper_bound: bounded_upper_risk_f32(
            projected_alpha_mass_inflation_upper_bound,
            "projected alpha-mass inflation upper bound",
        )?,
        support_leakage_fraction: bounded_risk_f32(
            support_leakage_fraction,
            "representative support leakage",
        )?,
        support_growth: bounded_upper_risk_f32(support_growth, "representative support growth")?,
        major_scale_growth: bounded_upper_risk_f32(
            major_scale_growth,
            "representative major-scale growth",
        )?,
        anisotropy_growth: bounded_upper_risk_f32(
            anisotropy_growth,
            "representative anisotropy growth",
        )?,
    })
}

fn ratio_or_infinity(numerator: f64, denominator: f64, epsilon: f64) -> f64 {
    if denominator > epsilon {
        numerator / denominator
    } else if numerator <= epsilon {
        1.0
    } else {
        f64::INFINITY
    }
}

/// Let `A_i = adj(Sigma_i)` and `S = sum(alpha_i sqrt(A_i))`. Minkowski's
/// inequality gives `sum alpha_i ||sqrt(A_i)n|| >= ||S n||` for every unit view
/// direction `n`, so `B = S^T S` is a safe quadratic lower bound on source
/// projected alpha mass. It is exact when all source covariances are identical.
/// The largest generalized eigenvalue of `(adj(Sigma_rep), B)` therefore gives
/// a conservative all-view upper bound on projected alpha-mass inflation.
fn projected_alpha_mass_inflation_upper_bound(
    representative_alpha: f64,
    representative_projected_area: [[f64; 3]; 3],
    source_projected_alpha_mass_sqrt_sum: [[f64; 3]; 3],
) -> Result<f64, LodBuildError> {
    if representative_alpha <= f64::EPSILON {
        return Ok(0.0);
    }
    let source_projected_alpha_mass_quadratic = multiply_3x3(
        transpose_3x3(source_projected_alpha_mass_sqrt_sum),
        source_projected_alpha_mass_sqrt_sum,
    );
    let Some(cholesky) = cholesky_3x3(source_projected_alpha_mass_quadratic) else {
        return Ok(f64::INFINITY);
    };
    let inverse = invert_lower_triangular_3x3(cholesky);
    let mut normalized = multiply_3x3(
        multiply_3x3(inverse, representative_projected_area),
        transpose_3x3(inverse),
    );
    for (row, column) in [(0, 1), (0, 2), (1, 2)] {
        let symmetric = 0.5 * (normalized[row][column] + normalized[column][row]);
        normalized[row][column] = symmetric;
        normalized[column][row] = symmetric;
    }
    let (eigenvalues, _) = symmetric_eigendecomposition(normalized)?;
    let maximum = eigenvalues.into_iter().fold(0.0_f64, f64::max).max(0.0);
    Ok(representative_alpha * maximum.sqrt())
}

fn cholesky_3x3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let scale = matrix
        .iter()
        .enumerate()
        .map(|(axis, row)| row[axis].abs())
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let minimum_pivot = scale * 1e-12;
    let mut lower = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..=row {
            let previous = (0..column)
                .map(|index| lower[row][index] * lower[column][index])
                .sum::<f64>();
            if row == column {
                let pivot = matrix[row][row] - previous;
                if !pivot.is_finite() || pivot <= minimum_pivot {
                    return None;
                }
                lower[row][column] = pivot.sqrt();
            } else {
                lower[row][column] = (matrix[row][column] - previous) / lower[column][column];
                if !lower[row][column].is_finite() {
                    return None;
                }
            }
        }
    }
    Some(lower)
}

fn invert_lower_triangular_3x3(lower: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut inverse = [[0.0; 3]; 3];
    for column in [0, 1, 2] {
        for row in column..3 {
            if row == column {
                inverse[row][column] = lower[row][row].recip();
            } else {
                let previous = (column..row)
                    .map(|index| lower[row][index] * inverse[index][column])
                    .sum::<f64>();
                inverse[row][column] = -previous / lower[row][row];
            }
        }
    }
    inverse
}

fn multiply_3x3(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|index| left[row][index] * right[index][column])
                .sum()
        })
    })
}

fn transpose_3x3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

fn bounded_risk_f32(value: f64, name: &'static str) -> Result<f32, LodBuildError> {
    if value.is_nan() || value < 0.0 {
        return Err(LodBuildError::DerivedNonFinite(name));
    }
    Ok(value.min(f64::from(f32::MAX)) as f32)
}

/// Convert a conservative f64 upper bound without rounding it downward.
fn bounded_upper_risk_f32(value: f64, name: &'static str) -> Result<f32, LodBuildError> {
    if value.is_nan() || value < 0.0 {
        return Err(LodBuildError::DerivedNonFinite(name));
    }
    if value >= f64::from(f32::MAX) {
        return Ok(f32::MAX);
    }
    let rounded = value as f32;
    Ok(if f64::from(rounded) < value {
        next_up(rounded)
    } else {
        rounded
    })
}

fn covariance_to_rotation_scale(
    covariance: [[f64; 3]; 3],
) -> Result<([f32; 4], [f32; 3]), LodBuildError> {
    let (mut eigenvalues, mut eigenvectors) = symmetric_eigendecomposition(covariance)?;
    let mut order = [0_usize, 1, 2];
    order.sort_unstable_by(|left, right| {
        eigenvalues[*right]
            .total_cmp(&eigenvalues[*left])
            .then_with(|| left.cmp(right))
    });
    eigenvalues = order.map(|index| eigenvalues[index].max(0.0));
    eigenvectors =
        std::array::from_fn(|row| std::array::from_fn(|column| eigenvectors[row][order[column]]));
    if determinant(eigenvectors) < 0.0 {
        for row in &mut eigenvectors {
            row[2] = -row[2];
        }
    }

    // Eigenvector columns form V in Sigma = V D V^T. A stored quaternion for
    // V makes the renderer construct R=V^T and evaluate R^T D R = V D V^T.
    let matrix = Mat3::from_cols(
        Vec3::new(
            eigenvectors[0][0] as f32,
            eigenvectors[1][0] as f32,
            eigenvectors[2][0] as f32,
        ),
        Vec3::new(
            eigenvectors[0][1] as f32,
            eigenvectors[1][1] as f32,
            eigenvectors[2][1] as f32,
        ),
        Vec3::new(
            eigenvectors[0][2] as f32,
            eigenvectors[1][2] as f32,
            eigenvectors[2][2] as f32,
        ),
    );
    let mut quaternion = Quat::from_mat3(&matrix).normalize();
    if quaternion.w < 0.0
        || (quaternion.w == 0.0
            && [quaternion.x, quaternion.y, quaternion.z]
                .into_iter()
                .find(|value| *value != 0.0)
                .is_some_and(|value| value < 0.0))
    {
        quaternion = -quaternion;
    }
    if !quaternion.is_finite() {
        return Err(LodBuildError::DerivedNonFinite("merged rotation"));
    }
    let scale = [
        checked_f32(eigenvalues[0].sqrt(), "merged scale")?,
        checked_f32(eigenvalues[1].sqrt(), "merged scale")?,
        checked_f32(eigenvalues[2].sqrt(), "merged scale")?,
    ];
    Ok((
        [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
        scale,
    ))
}

fn symmetric_eigendecomposition(
    mut matrix: [[f64; 3]; 3],
) -> Result<([f64; 3], [[f64; 3]; 3]), LodBuildError> {
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let candidates = [
            (matrix[0][1].abs(), 0, 1),
            (matrix[0][2].abs(), 0, 2),
            (matrix[1][2].abs(), 1, 2),
        ];
        let &(_, p, q) = candidates
            .iter()
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .unwrap();
        let off_diagonal = matrix[p][q];
        let scale = matrix[p][p].abs().max(matrix[q][q].abs()).max(1.0);
        if off_diagonal.abs() <= 1e-14 * scale {
            break;
        }
        let tau = (matrix[q][q] - matrix[p][p]) / (2.0 * off_diagonal);
        let t = if tau == 0.0 {
            1.0
        } else {
            tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt())
        };
        let cosine = 1.0 / (1.0 + t * t).sqrt();
        let sine = t * cosine;

        let app = matrix[p][p];
        let aqq = matrix[q][q];
        matrix[p][p] =
            cosine * cosine * app - 2.0 * sine * cosine * off_diagonal + sine * sine * aqq;
        matrix[q][q] =
            sine * sine * app + 2.0 * sine * cosine * off_diagonal + cosine * cosine * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for index in 0..3 {
            if index != p && index != q {
                let aip = matrix[index][p];
                let aiq = matrix[index][q];
                matrix[index][p] = cosine * aip - sine * aiq;
                matrix[p][index] = matrix[index][p];
                matrix[index][q] = sine * aip + cosine * aiq;
                matrix[q][index] = matrix[index][q];
            }
            let vip = vectors[index][p];
            let viq = vectors[index][q];
            vectors[index][p] = cosine * vip - sine * viq;
            vectors[index][q] = sine * vip + cosine * viq;
        }
    }
    let values = [matrix[0][0], matrix[1][1], matrix[2][2]];
    if !values
        .iter()
        .chain(vectors.iter().flatten())
        .all(|value| value.is_finite())
    {
        return Err(LodBuildError::DerivedNonFinite(
            "covariance eigendecomposition",
        ));
    }
    Ok((values, vectors))
}

fn determinant(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn checked_f32(value: f64, name: &'static str) -> Result<f32, LodBuildError> {
    if !value.is_finite() || value < -(f32::MAX as f64) || value > f32::MAX as f64 {
        return Err(LodBuildError::DerivedNonFinite(name));
    }
    Ok(value as f32)
}

fn farthest_corner_distance(bounds: LodBounds, point: [f32; 3]) -> Result<f32, LodBuildError> {
    let delta = std::array::from_fn::<_, 3, _>(|axis| {
        (f64::from(point[axis]) - f64::from(bounds.min[axis]))
            .abs()
            .max((f64::from(point[axis]) - f64::from(bounds.max[axis])).abs())
    });
    checked_f32(
        (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt(),
        "geometric error",
    )
}

fn bounds_epsilon(left: &LodBounds, right: &LodBounds) -> f32 {
    1e-5 * left.radius().max(right.radius()).max(1.0)
}

fn validate_source_partition(
    expected_count: u64,
    ranges: impl IntoIterator<Item = LodSourceRange>,
) -> Result<(), ()> {
    let mut expected_start = 0_u64;
    for range in ranges {
        if range.start != expected_start || range.count == 0 {
            return Err(());
        }
        expected_start = range.end().ok_or(())?;
    }
    (expected_start == expected_count).then_some(()).ok_or(())
}

fn validate_absolute_source_partition(
    parent: LodSourceRange,
    ranges: impl IntoIterator<Item = LodSourceRange>,
) -> Result<(), ()> {
    let parent_end = parent.end().ok_or(())?;
    let mut expected_start = parent.start;
    for range in ranges {
        if range.start != expected_start || range.count == 0 {
            return Err(());
        }
        expected_start = range.end().ok_or(())?;
        if expected_start > parent_end {
            return Err(());
        }
    }
    (expected_start == parent_end).then_some(()).ok_or(())
}

fn next_down(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f32::from_bits(1);
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

fn next_up(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

#[derive(Debug)]
pub enum LodBuildError {
    InvalidSettings(LodBuildSettingsError),
    PlaneLengthMismatch {
        plane: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidGaussian {
        index: usize,
        field: GaussianField,
    },
    InvalidBounds(LodBoundsError),
    DerivedNonFinite(&'static str),
    EmptyReduction,
    CountOverflow(&'static str),
    NonContiguousChildren,
    Validation(LodValidationError),
}

impl fmt::Display for LodBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSettings(error) => write!(f, "invalid LoD build settings: {error}"),
            Self::PlaneLengthMismatch {
                plane,
                expected,
                actual,
            } => write!(
                f,
                "planar field {plane} has length {actual}, expected {expected}"
            ),
            Self::InvalidGaussian { index, field } => {
                write!(f, "source Gaussian {index} has invalid {field:?}")
            }
            Self::InvalidBounds(error) => write!(f, "invalid LoD bounds: {error}"),
            Self::DerivedNonFinite(stage) => {
                write!(f, "LoD build produced a non-finite value during {stage}")
            }
            Self::EmptyReduction => write!(f, "cannot reduce an empty Gaussian sequence"),
            Self::CountOverflow(name) => write!(f, "LoD {name} exceeds the format limit"),
            Self::NonContiguousChildren => {
                write!(f, "breadth-first hierarchy children are not contiguous")
            }
            Self::Validation(error) => write!(f, "built LoD failed validation: {error}"),
        }
    }
}

impl Error for LodBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidSettings(error) => Some(error),
            Self::InvalidBounds(error) => Some(error),
            Self::Validation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LodBoundsError> for LodBuildError {
    fn from(value: LodBoundsError) -> Self {
        Self::InvalidBounds(value)
    }
}

#[derive(Debug)]
pub enum LodValidationError {
    InvalidMagic([u8; 8]),
    UnsupportedManifestVersion(u16),
    UnsupportedPageVersion(u16),
    IncompatibleSphericalHarmonics {
        required: u64,
        supported: u64,
    },
    UnsupportedRequiredFeatures(u64),
    MissingHighFidelityCertificateFeature,
    MissingSharedNodePageFeature(LodPageId),
    InhomogeneousSharedNodePage(LodPageId),
    InvalidBuildSettings(LodBuildSettingsError),
    InvalidBuildVersion,
    ConfigFingerprintMismatch,
    CompressedSourceLeaf(LodPageId),
    InconsistentRepresentativeEncoding,
    CountOverflow(&'static str),
    CountMismatch(&'static str),
    InvalidEmptyManifest,
    MissingSceneBounds,
    InvalidSceneBounds(LodBoundsError),
    IncompleteManifest,
    InvalidPage {
        index: usize,
        source: LodPageValidationError,
    },
    DuplicatePageId(LodPageId),
    InvalidNodeId(usize),
    DuplicateNodeId(LodNodeId),
    InvalidNodeBounds {
        index: usize,
        source: LodBoundsError,
    },
    InvalidSourceRange(LodNodeId),
    InvalidMortonRange(LodNodeId),
    EmptyRepresentation(LodNodeId),
    UnknownPage(LodPageId),
    InvalidPageRange(LodNodeId),
    RepresentationOutsideNode(LodNodeId),
    InvalidError(LodNodeId),
    InvalidQualityInterval(LodNodeId),
    InvalidHighFidelityCertificate(LodNodeId),
    InvalidChildRange(LodNodeId),
    UnknownParent {
        node: LodNodeId,
        parent: LodNodeId,
    },
    DuplicateRoot(LodNodeId),
    UnknownRoot(LodNodeId),
    InvalidRoot(LodNodeId),
    RootSourcePartition,
    CycleOrSharedChild(LodNodeId),
    PageKindMismatch(LodNodeId),
    InvalidLeafRepresentation(LodNodeId),
    InvalidBranching(LodNodeId),
    InvalidRefinementAmplification {
        node: LodNodeId,
        parent_count: u64,
        child_count: u64,
        maximum: u8,
    },
    ChildSourcePartition(LodNodeId),
    ParentChildMismatch {
        parent: LodNodeId,
        child: LodNodeId,
    },
    DepthMismatch(LodNodeId),
    MortonOrder(LodNodeId),
    BoundsDoNotContainChild {
        parent: LodNodeId,
        child: LodNodeId,
    },
    NonMonotonicError {
        parent: LodNodeId,
        child: LodNodeId,
    },
    NonMonotonicHighFidelityCertificate {
        parent: LodNodeId,
        child: LodNodeId,
    },
    UnreachableNode(LodNodeId),
    PageCoverage(LodPageId),
    UnreferencedPage,
    SceneBoundsDoNotContainRoot(LodNodeId),
    QualityMetadataMismatch,
    PayloadCountMismatch,
    DuplicatePayload(LodPageId),
    UnknownPayload(LodPageId),
    InvalidPayload {
        page: LodPageId,
        source: LodPageValidationError,
    },
    InvalidPayloadBounds {
        page: LodPageId,
        source: Box<LodBuildError>,
    },
    PayloadOutsideDescriptor(LodPageId),
}

impl fmt::Display for LodValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Error for LodValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian(position: [f32; 3], scale: [f32; 3], opacity: f32, dc: f32) -> Gaussian3d {
        let mut coefficients = [0.0; SH_COEFF_COUNT];
        coefficients[0] = dc;
        Gaussian3d {
            position_visibility: [position[0], position[1], position[2], 1.0].into(),
            spherical_harmonic: SphericalHarmonicCoefficients { coefficients },
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [scale[0], scale[1], scale[2], opacity].into(),
        }
    }

    fn fixture(count: usize) -> PlanarGaussian3d {
        (0..count)
            .map(|index| {
                let x = (index % 7) as f32 - 3.0;
                let y = ((index / 7) % 5) as f32 - 2.0;
                let z = (index / 35) as f32;
                gaussian(
                    [x, y, z],
                    [0.05 + index as f32 * 0.001, 0.1, 0.2],
                    0.2 + (index % 3) as f32 * 0.1,
                    index as f32 * 0.01,
                )
            })
            .collect()
    }

    fn assert_matrix_close(left: [[f64; 3]; 3], right: [[f64; 3]; 3], epsilon: f64) {
        for row in 0..3 {
            for column in 0..3 {
                assert!(
                    (left[row][column] - right[row][column]).abs() <= epsilon,
                    "matrix mismatch [{row}][{column}]: {} vs {}",
                    left[row][column],
                    right[row][column]
                );
            }
        }
    }

    fn renderer_covariance(gaussian: &Gaussian3d) -> [[f64; 3]; 3] {
        let packed = crate::gaussian::covariance::compute_covariance_3d(
            bevy::math::Vec4::from_array(gaussian.rotation.rotation),
            Vec3::from_array(gaussian.scale_opacity.scale),
        );
        [
            [packed[0] as f64, packed[1] as f64, packed[2] as f64],
            [packed[1] as f64, packed[3] as f64, packed[4] as f64],
            [packed[2] as f64, packed[4] as f64, packed[5] as f64],
        ]
    }

    fn quadratic_form(matrix: [[f64; 3]; 3], direction: [f64; 3]) -> f64 {
        (0..3)
            .flat_map(|row| (0..3).map(move |column| (row, column)))
            .map(|(row, column)| direction[row] * matrix[row][column] * direction[column])
            .sum()
    }

    #[test]
    fn support_bounds_follow_rotated_covariance() {
        let angle = std::f32::consts::FRAC_PI_4;
        let mut gaussian = gaussian([2.0, -1.0, 4.0], [2.0, 1.0, 0.5], 0.5, 0.0);
        gaussian.rotation.rotation = [angle.cos(), 0.0, 0.0, angle.sin()];
        let covariance = gaussian_covariance(&gaussian).unwrap();
        let bounds = gaussian_support_bounds(&gaussian, 3.0).unwrap();
        for (axis, row) in covariance.iter().enumerate() {
            let expected = 3.0 * row[axis].sqrt() as f32;
            assert!(bounds.min[axis] <= gaussian.position_visibility.position[axis] - expected);
            assert!(bounds.max[axis] >= gaussian.position_visibility.position[axis] + expected);
        }
    }

    #[test]
    fn builder_rejects_a_non_finite_f32_morton_normalization_extent() {
        let source: PlanarGaussian3d = vec![
            gaussian([-f32::MAX, 0.0, 0.0], [0.0; 3], 0.5, 0.0),
            gaussian([f32::MAX, 0.0, 0.0], [0.0; 3], 0.5, 0.0),
        ]
        .into();
        let error = CpuGaussianLodBuilder::default().build(&source).unwrap_err();
        assert!(matches!(
            error,
            LodBuildError::DerivedNonFinite("Morton normalization extent")
        ));
    }

    #[test]
    fn moment_merge_preserves_mixture_moments_and_transmittance() {
        let source = [
            gaussian([-1.0, 0.0, 0.0], [0.1; 3], 0.5, -1.0),
            gaussian([1.0, 0.0, 0.0], [0.1; 3], 0.5, 1.0),
        ];
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();
        assert!(merged.gaussian.position_visibility.position[0].abs() < 1e-6);
        assert!((merged.gaussian.scale_opacity.opacity - 0.75).abs() < 1e-6);
        assert!(merged.gaussian.spherical_harmonic.coefficients[0].abs() < 1e-6);

        let actual = gaussian_covariance(&merged.gaussian).unwrap();
        let expected = [[1.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]];
        assert_matrix_close(actual, expected, 1e-5);

        let risk = merged.raster_risk();
        assert!(risk.sampled_projected_alpha_mass_inflation > 7.0);
        assert!(
            risk.projected_alpha_mass_inflation_upper_bound
                >= risk.sampled_projected_alpha_mass_inflation
        );
        assert!(risk.support_leakage_fraction > 0.5);
        assert!(risk.support_growth > 2.0);
        assert!(risk.major_scale_growth > 10.0);
        assert!(risk.score() > 9.0);
        assert!(risk.high_fidelity_certificate() < 0.11);
    }

    #[test]
    fn minkowski_bound_is_exact_for_coincident_identical_covariance() {
        let source = [gaussian([0.0; 3], [0.1; 3], 0.1, 0.0); 8];
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();
        let risk = merged.raster_risk();

        // The sampled ratio is exact for this isotropic fixture: aggregation
        // loses projected alpha mass, and the representative support is exact.
        assert!((0.70..0.73).contains(&risk.sampled_projected_alpha_mass_inflation));
        assert!(risk.support_leakage_fraction < 1e-5);
        assert!((risk.support_growth - 1.0).abs() < 1e-5);
        assert!((risk.major_scale_growth - 1.0).abs() < 1e-5);
        assert!(risk.score() < 1e-5);

        // Minkowski retains the cross terms for identical covariance, making
        // the all-view proof exact instead of falsely refining this safe
        // low-alpha overlap.
        assert!(
            (risk.projected_alpha_mass_inflation_upper_bound
                - risk.sampled_projected_alpha_mass_inflation)
                .abs()
                < 1e-5,
            "Minkowski bound should be exact: {risk:?}"
        );
        assert!((risk.high_fidelity_certificate() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn anisotropy_growth_limits_the_high_fidelity_certificate() {
        let risk = MomentMergeRasterRisk {
            sampled_projected_alpha_mass_inflation: 1.0,
            projected_alpha_mass_inflation_upper_bound: 1.0,
            support_leakage_fraction: 0.0,
            support_growth: 1.0,
            major_scale_growth: 1.0,
            anisotropy_growth: 4.0,
        };

        assert_eq!(risk.score(), 3.0);
        assert_eq!(risk.high_fidelity_certificate(), 0.25);
    }

    #[test]
    fn moment_merge_source_covariance_matches_renderer_convention() {
        let mut source = gaussian([0.0; 3], [2.0, 0.75, 0.2], 0.5, 0.0);
        let rotation = Quat::from_euler(bevy::math::EulerRot::XYZ, 0.31, -0.77, 1.12).normalize();
        source.rotation.rotation = [rotation.w, rotation.x, rotation.y, rotation.z];

        assert_matrix_close(
            gaussian_covariance(&source).unwrap(),
            renderer_covariance(&source),
            1e-5,
        );
    }

    #[test]
    fn projected_area_psd_square_root_reconstructs_rotated_adjugate() {
        let mut source = gaussian([0.0; 3], [2e-4, 7.5e-5, 2e-5], 0.5, 0.0);
        let rotation = Quat::from_euler(bevy::math::EulerRot::XYZ, -0.41, 0.83, 1.37).normalize();
        source.rotation.rotation = [rotation.w, rotation.x, rotation.y, rotation.z];
        let projected_area = symmetric_adjugate(gaussian_covariance(&source).unwrap());
        let square_root = symmetric_psd_sqrt(projected_area).unwrap();
        let reconstructed = multiply_3x3(transpose_3x3(square_root), square_root);

        assert_matrix_close(reconstructed, projected_area, 1e-24);
    }

    #[test]
    fn minkowski_bound_dominates_dense_all_view_projection_sweep() {
        let mut source = [
            gaussian([-0.4, 0.1, 0.2], [0.9, 0.25, 0.08], 0.15, 0.0),
            gaussian([0.3, -0.2, 0.1], [0.6, 0.18, 0.05], 0.4, 0.0),
            gaussian([0.1, 0.4, -0.3], [0.45, 0.12, 0.03], 0.7, 0.0),
        ];
        for (sample, angles) in
            source
                .iter_mut()
                .zip([[0.2, -0.7, 1.1], [-0.8, 0.35, 0.4], [1.0, 0.6, -0.5]])
        {
            let rotation =
                Quat::from_euler(bevy::math::EulerRot::XYZ, angles[0], angles[1], angles[2])
                    .normalize();
            sample.rotation.rotation = [rotation.w, rotation.x, rotation.y, rotation.z];
        }

        let merged = MomentMergeReducer::default().reduce(&source).unwrap();
        let risk = merged.raster_risk();
        let representative_area =
            symmetric_adjugate(gaussian_covariance(&merged.gaussian).unwrap());
        let representative_alpha = f64::from(
            merged.gaussian.scale_opacity.opacity * merged.gaussian.position_visibility.visibility,
        );
        let source_area = source.map(|sample| {
            (
                f64::from(sample.scale_opacity.opacity * sample.position_visibility.visibility),
                symmetric_adjugate(gaussian_covariance(&sample).unwrap()),
            )
        });
        let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        let mut sampled_max = 0.0_f64;
        for index in 0..4096 {
            let z = 1.0 - 2.0 * (index as f64 + 0.5) / 4096.0;
            let radius = (1.0 - z * z).sqrt();
            let azimuth = golden_angle * index as f64;
            let direction = [radius * azimuth.cos(), radius * azimuth.sin(), z];
            let numerator = representative_alpha
                * quadratic_form_f64(representative_area, direction)
                    .max(0.0)
                    .sqrt();
            let denominator = source_area
                .iter()
                .map(|(alpha, area)| alpha * quadratic_form_f64(*area, direction).max(0.0).sqrt())
                .sum::<f64>();
            sampled_max = sampled_max.max(numerator / denominator);
        }

        assert!(risk.projected_alpha_mass_inflation_upper_bound.is_finite());
        assert!(
            f64::from(risk.projected_alpha_mass_inflation_upper_bound) >= sampled_max,
            "Minkowski bound {} fell below dense sampled ratio {sampled_max}",
            risk.projected_alpha_mass_inflation_upper_bound
        );
    }

    #[test]
    fn diagonal_cluster_renders_its_merged_principal_axis_without_transposition() {
        let source = [
            gaussian([-1.0, -1.0, 0.0], [0.01; 3], 0.25, 0.0),
            gaussian([1.0, 1.0, 0.0], [0.01; 3], 0.25, 0.0),
        ];
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();
        let covariance = renderer_covariance(&merged.gaussian);
        let inverse_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;
        let along = quadratic_form(covariance, [inverse_sqrt_two, inverse_sqrt_two, 0.0]);
        let across = quadratic_form(covariance, [inverse_sqrt_two, -inverse_sqrt_two, 0.0]);

        assert!(
            covariance[0][1] > 0.0,
            "merged covariance points across the source diagonal: {covariance:?}"
        );
        assert!(
            along > across * 1_000.0,
            "merged major axis is not aligned with the source diagonal: along={along}, across={across}"
        );
        let risk = merged.raster_risk();
        assert!(risk.sampled_projected_alpha_mass_inflation > 20.0);
        assert!(risk.major_scale_growth > 100.0);
        assert!(risk.high_fidelity_certificate() < 0.01);
    }

    #[test]
    fn curved_thin_surface_cannot_hide_inside_one_moment_ellipse() {
        let source: Vec<_> = (0..16)
            .map(|index| {
                let angle = std::f32::consts::TAU * index as f32 / 16.0;
                let mut sample = gaussian(
                    [angle.cos(), angle.sin(), 0.0],
                    [0.08, 0.01, 0.005],
                    0.2,
                    0.0,
                );
                let tangent = Quat::from_rotation_z(angle + std::f32::consts::FRAC_PI_2);
                sample.rotation.rotation = [tangent.w, tangent.x, tangent.y, tangent.z];
                sample
            })
            .collect();
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();
        let risk = merged.raster_risk();

        assert!(
            risk.sampled_projected_alpha_mass_inflation > 50.0,
            "{risk:?}"
        );
        assert!(risk.projected_alpha_mass_inflation_upper_bound > 50.0);
        assert!(risk.support_leakage_fraction > 0.4);
        assert!(risk.support_growth > 1.5);
        assert!(risk.major_scale_growth > 8.0);
        assert!(risk.high_fidelity_certificate() < 0.02);
    }

    #[test]
    fn covariance_eigendecomposition_round_trips_rotation_convention() {
        let mut source = gaussian([0.0; 3], [2.0, 0.75, 0.2], 0.5, 0.0);
        let rotation = Quat::from_euler(bevy::math::EulerRot::XYZ, 0.31, -0.77, 1.12).normalize();
        source.rotation.rotation = [rotation.w, rotation.x, rotation.y, rotation.z];
        let expected = renderer_covariance(&source);
        let (rotation, scale) = covariance_to_rotation_scale(expected).unwrap();
        let reconstructed = Gaussian3d {
            rotation: Rotation { rotation },
            scale_opacity: ScaleOpacity {
                scale,
                opacity: 0.5,
            },
            ..source
        };
        let actual = renderer_covariance(&reconstructed);
        assert_matrix_close(actual, expected, 1e-5);
    }

    #[test]
    fn builder_is_deterministic_across_input_order() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 3,
            support_sigma: 3.0,
        };
        let source = fixture(37);
        let mut reversed: Vec<_> = source.iter().collect();
        reversed.reverse();
        let reversed: PlanarGaussian3d = reversed.into();

        let first = build_planar_3d_lod(&source, settings).unwrap();
        let second = build_planar_3d_lod(&reversed, settings).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.manifest.roots, vec![LodNodeId(1)]);
    }

    #[test]
    fn signed_zero_inputs_produce_byte_identical_pages_across_reversal() {
        let mut first = gaussian([-0.0, 1.0, 2.0], [0.1; 3], 0.5, 0.0);
        first.spherical_harmonic.coefficients[1] = 0.0;
        let mut second = gaussian([0.0, 1.0, 2.0], [0.1; 3], 0.5, -0.0);
        second.spherical_harmonic.coefficients[1] = -0.0;
        let settings = GaussianLodBuildSettings {
            branching_factor: 2,
            leaf_capacity: 2,
            support_sigma: 3.0,
        };

        let forward = build_planar_3d_lod(&vec![first, second].into(), settings).unwrap();
        let reverse = build_planar_3d_lod(&vec![second, first].into(), settings).unwrap();
        assert_eq!(forward.manifest, reverse.manifest);
        assert_eq!(forward.pages.len(), reverse.pages.len());
        for (left, right) in forward.pages.iter().zip(&reverse.pages) {
            assert_eq!(
                crate::io::lod::encode_page(left).unwrap(),
                crate::io::lod::encode_page(right).unwrap()
            );
        }
    }

    #[test]
    fn manifest_requires_exact_compiled_sh_layout() {
        let mut output =
            build_planar_3d_lod(&fixture(4), GaussianLodBuildSettings::default()).unwrap();
        assert_eq!(
            output.manifest.header.required_features,
            LOD_CURRENT_REQUIRED_FEATURES | LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES
        );

        let incompatible = if LOD_CURRENT_SH_FEATURE == LOD_REQUIRED_FEATURE_SH0 {
            LOD_REQUIRED_FEATURE_SH1
        } else {
            LOD_REQUIRED_FEATURE_SH0
        };
        output.manifest.header.required_features = incompatible
            | LOD_REQUIRED_FEATURE_HIGH_FIDELITY_CERTIFICATE
            | LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES;
        assert!(matches!(
            output.manifest.validate(),
            Err(LodValidationError::IncompatibleSphericalHarmonics {
                required,
                supported,
            }) if required == incompatible && supported == LOD_CURRENT_SH_FEATURE
        ));

        output.manifest.header.required_features = 0;
        assert!(matches!(
            output.manifest.validate(),
            Err(LodValidationError::IncompatibleSphericalHarmonics {
                required: 0,
                supported,
            }) if supported == LOD_CURRENT_SH_FEATURE
        ));
    }

    #[test]
    fn manifest_v3_requires_and_validates_high_fidelity_certificates() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 8,
            leaf_capacity: 2,
            support_sigma: 3.0,
        };
        let source: PlanarGaussian3d = (0..16)
            .map(|index| {
                let angle = std::f32::consts::TAU * index as f32 / 16.0;
                gaussian(
                    [angle.cos(), angle.sin(), 0.0],
                    [0.08, 0.01, 0.005],
                    0.2,
                    0.0,
                )
            })
            .collect();
        let output = build_planar_3d_lod(&source, settings).unwrap();
        assert_eq!(output.manifest.header.manifest_version, 3);
        assert_ne!(
            output.manifest.header.required_features
                & LOD_REQUIRED_FEATURE_HIGH_FIDELITY_CERTIFICATE,
            0
        );
        assert!(output.manifest.nodes[0].high_fidelity_certificate < 1.0 / 1.715);
        for node in &output.manifest.nodes {
            assert!((0.0..=1.0).contains(&node.high_fidelity_certificate));
            if node.is_leaf() {
                assert_eq!(node.high_fidelity_certificate, 1.0);
            } else {
                let start = node.children.start as usize;
                let end = node.children.end().unwrap() as usize;
                assert!(
                    output.manifest.nodes[start..end]
                        .iter()
                        .all(|child| node.high_fidelity_certificate
                            <= child.high_fidelity_certificate)
                );
            }
        }

        let mut missing_feature = output.manifest.clone();
        missing_feature.header.required_features &= !LOD_REQUIRED_FEATURE_HIGH_FIDELITY_CERTIFICATE;
        assert!(matches!(
            missing_feature.validate(),
            Err(LodValidationError::MissingHighFidelityCertificateFeature)
        ));

        let mut stale_schema = output.manifest.clone();
        stale_schema.header.manifest_version = 2;
        assert!(matches!(
            stale_schema.validate(),
            Err(LodValidationError::UnsupportedManifestVersion(2))
        ));

        let mut invalid_value = output.manifest.clone();
        invalid_value.nodes[0].high_fidelity_certificate = f32::NAN;
        assert!(matches!(
            invalid_value.validate(),
            Err(LodValidationError::InvalidHighFidelityCertificate(_))
        ));

        let parent_index = output
            .manifest
            .nodes
            .iter()
            .enumerate()
            .find_map(|(parent_index, node)| {
                let start = node.children.start as usize;
                let end = node.children.end()? as usize;
                output.manifest.nodes[start..end]
                    .iter()
                    .any(|child| child.high_fidelity_certificate < 1.0)
                    .then_some(parent_index)
            })
            .expect("curved fixture must contain a nontrivially certified edge");
        let mut non_monotonic = output.manifest.clone();
        non_monotonic.nodes[parent_index].high_fidelity_certificate = 1.0;
        assert!(matches!(
            non_monotonic.validate(),
            Err(LodValidationError::NonMonotonicHighFidelityCertificate { .. })
        ));
    }

    #[test]
    fn two_to_one_leaf_rung_certifies_overlap_and_rejects_disconnected_geometry() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 8,
            leaf_capacity: 2,
            support_sigma: 3.0,
        };
        let coincident: PlanarGaussian3d = vec![gaussian([0.0; 3], [0.1; 3], 0.1, 0.0); 2].into();
        let safe = build_planar_3d_lod(&coincident, settings).unwrap();
        assert_eq!(safe.manifest.nodes[0].representation.count, 1);
        assert_eq!(safe.manifest.nodes[0].children.count, 2);
        assert!((safe.manifest.nodes[0].high_fidelity_certificate - 1.0).abs() < 1e-5);

        let disconnected: PlanarGaussian3d = vec![
            gaussian([-1.0, 0.0, 0.0], [0.1; 3], 0.5, 0.0),
            gaussian([1.0, 0.0, 0.0], [0.1; 3], 0.5, 0.0),
        ]
        .into();
        let unsafe_rung = build_planar_3d_lod(&disconnected, settings).unwrap();
        assert_eq!(unsafe_rung.manifest.nodes[0].representation.count, 1);
        assert!(unsafe_rung.manifest.nodes[0].high_fidelity_certificate < 0.11);
    }

    #[test]
    fn capacity_one_keeps_forced_merge_and_fails_closed() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 8,
            leaf_capacity: 1,
            support_sigma: 3.0,
        };
        let disconnected: PlanarGaussian3d = vec![
            gaussian([-1.0, 0.0, 0.0], [0.1; 3], 0.5, 0.0),
            gaussian([1.0, 0.0, 0.0], [0.1; 3], 0.5, 0.0),
        ]
        .into();
        let output = build_planar_3d_lod(&disconnected, settings).unwrap();
        assert_eq!(output.manifest.nodes[0].representation.count, 1);
        assert!(output.manifest.nodes[0].high_fidelity_certificate < 0.11);
        assert_eq!(
            output.manifest.header.node_count,
            output.manifest.header.page_count
        );
        assert_eq!(
            output.manifest.header.required_features,
            LOD_CURRENT_REQUIRED_FEATURES
        );
        assert!(
            output
                .manifest
                .pages
                .iter()
                .all(|page| page.gaussian_count == 1)
        );
    }

    #[test]
    fn risk_aware_progressive_rung_avoids_balanced_scene_gap_merges() {
        let source = [(0.0_f32, 3_usize), (10.0_f32, 6_usize), (20.0_f32, 3_usize)]
            .into_iter()
            .flat_map(|(x, count)| {
                std::iter::repeat_n(gaussian([x, 0.0, 0.0], [0.01; 3], 0.25, 0.0), count)
            })
            .collect::<Vec<_>>();
        let representative_count = 3;
        let support_sigma = 3.0;

        // Equal-count ranges are [0..4, 4..8, 8..12], so two of the three
        // representatives bridge a ten-unit empty interval. The risk-aware
        // exact-cardinality partition instead retains the natural [3, 6, 3]
        // contiguous Morton runs.
        let balanced = balanced_progressive_moment_merge_representatives(
            &source,
            representative_count,
            support_sigma,
        )
        .unwrap();
        let first =
            progressive_moment_merge_representatives(&source, representative_count, support_sigma)
                .unwrap();
        let second =
            progressive_moment_merge_representatives(&source, representative_count, support_sigma)
                .unwrap();

        let balanced_certificate = balanced
            .iter()
            .map(MomentMergeResult::high_fidelity_certificate)
            .fold(1.0_f32, f32::min);
        let risk_aware_certificate = first
            .representatives
            .iter()
            .map(MomentMergeResult::high_fidelity_certificate)
            .fold(1.0_f32, f32::min);
        assert!(balanced_certificate < 0.01, "{balanced_certificate}");
        assert!(risk_aware_certificate > 0.999, "{risk_aware_certificate}");
        assert_eq!(first.representatives, second.representatives);
        assert_eq!(first.policy_envelope, second.policy_envelope);
        assert_eq!(first.representatives.len(), representative_count);
        assert_eq!(
            first
                .representatives
                .iter()
                .map(|representative| representative.source_count)
                .collect::<Vec<_>>(),
            [3, 6, 3]
        );
        assert_eq!(
            first
                .representatives
                .iter()
                .map(|representative| representative.source_count)
                .sum::<u64>(),
            source.len() as u64
        );

        // The emitted geometry retains the safe natural runs, but selection is
        // no more permissive than the balanced partition: policy metadata is
        // enveloped by the same-count oracle rather than graded by the
        // optimized payload itself.
        let balanced_envelope = progressive_selection_envelope(&balanced);
        let emitted_envelope = progressive_selection_envelope(&first.representatives);
        assert_eq!(first.policy_envelope, balanced_envelope);
        assert_eq!(
            first.policy_envelope.high_fidelity_certificate_cap,
            balanced_certificate
        );
        assert_ne!(
            first.policy_envelope.support_bounds,
            emitted_envelope.support_bounds
        );
        assert!(first.policy_envelope.error.geometric > emitted_envelope.error.geometric);
        assert!(
            first.policy_envelope.high_fidelity_certificate_cap
                < emitted_envelope.high_fidelity_certificate_cap
        );
    }

    #[test]
    fn progressive_rung_retains_balanced_fallback_above_risk_aware_bound() {
        let source = (0..34)
            .map(|index| gaussian([index as f32, 0.0, 0.0], [0.01; 3], 0.25, 0.0))
            .collect::<Vec<_>>();
        let expected = balanced_progressive_moment_merge_representatives(&source, 2, 3.0).unwrap();
        let actual = progressive_moment_merge_representatives(&source, 2, 3.0).unwrap();

        assert_eq!(source.len().div_ceil(actual.representatives.len()), 17);
        assert_eq!(actual.representatives, expected);
        assert_eq!(
            actual.policy_envelope,
            ProgressiveSelectionEnvelope::IDENTITY
        );
    }

    #[test]
    fn adaptive_pairing_shifts_around_unsafe_morton_neighbor() {
        let source = [
            gaussian([-1.0, 0.0, 0.0], [0.05; 3], 0.5, 0.0),
            gaussian([0.0, 0.0, 0.0], [0.05; 3], 0.5, 0.0),
            gaussian([0.0, 0.0, 0.0], [0.05; 3], 0.5, 0.0),
        ];
        let representatives = high_fidelity_leaf_representatives(&source, 3.0).unwrap();
        assert_eq!(representatives.len(), 2);
        assert_eq!(representatives[0].source_count, 1);
        assert_eq!(representatives[1].source_count, 2);
        assert!(representatives.iter().all(|representative| {
            representative.high_fidelity_certificate() >= HIGH_FIDELITY_PAIR_CERTIFICATE
        }));
    }

    #[test]
    fn high_fidelity_pairing_certifies_sh_appearance() {
        let first = gaussian([0.0; 3], [0.1; 3], 0.1, -1.0);
        let opposite = gaussian([0.0; 3], [0.1; 3], 0.1, 1.0);
        let unsafe_merge = MomentMergeReducer::new(3.0)
            .unwrap()
            .reduce(&[first, opposite])
            .unwrap();
        assert!((unsafe_merge.raster_risk().high_fidelity_certificate() - 1.0).abs() < 1e-5);
        assert!(unsafe_merge.high_fidelity_certificate() < HIGH_FIDELITY_PAIR_CERTIFICATE);
        assert_eq!(
            high_fidelity_leaf_representatives(&[first, opposite], 3.0)
                .unwrap()
                .len(),
            2
        );

        let safe = high_fidelity_leaf_representatives(&[first, first], 3.0).unwrap();
        assert_eq!(safe.len(), 1);
        assert!(safe[0].high_fidelity_certificate() >= HIGH_FIDELITY_PAIR_CERTIFICATE);
    }

    #[test]
    fn unsafe_deepest_bridge_uses_only_exact_records_and_adjacent_pairs() {
        let source = (0..32)
            .map(|index| {
                gaussian(
                    [index as f32 * 10.0, 0.0, 0.0],
                    [0.01; 3],
                    0.5,
                    index as f32 * 0.25,
                )
            })
            .collect::<Vec<_>>();
        let certificates = adjacent_pair_certificates(&source, 3.0).unwrap();
        assert_eq!(
            maximum_certified_pairing_score(&certificates).merge_count,
            0
        );

        let pair_count = source.len() - source.len() * 7 / 8;
        let representatives = paired_leaf_representatives(&source, 3.0, pair_count, None).unwrap();
        assert_eq!(representatives.len(), 28);
        assert!(
            representatives
                .iter()
                .all(|representative| representative.source_count <= 2)
        );
        assert!(representatives.iter().any(|representative| {
            representative.high_fidelity_certificate() < HIGH_FIDELITY_PAIR_CERTIFICATE
        }));
    }

    #[test]
    fn exact_cardinality_pairing_is_deterministic_and_non_overlapping() {
        let certificates = [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
        let quality = optimal_pairing_quality_score(&certificates, 3, None).unwrap();
        let first =
            optimal_pairing_indices(&certificates, 3, Some(quality.minimum_certificate)).unwrap();
        let second =
            optimal_pairing_indices(&certificates, 3, Some(quality.minimum_certificate)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        assert!(first.windows(2).all(|pair| pair[0] + 1 < pair[1]));
        let selected_sum = first
            .iter()
            .map(|index| f64::from(certificates[*index]))
            .sum::<f64>();
        assert!((selected_sum - quality.certificate_sum).abs() <= f64::EPSILON * 4.0);

        // A sum-only matcher would choose edges 0 and 2 (0.9 total, 0.1
        // bottleneck). The bridge must instead choose the uniformly safer
        // edges 1 and 3 before using certificate sum as a tie-break.
        let bottleneck_fixture = [0.1, 0.4, 0.8, 0.4];
        let quality = optimal_pairing_quality_score(&bottleneck_fixture, 2, None).unwrap();
        assert_eq!(quality.minimum_certificate, 0.4);
        assert_eq!(quality.certificate_sum, 2.0 * f64::from(0.4_f32));
        assert_eq!(
            optimal_pairing_indices(&bottleneck_fixture, 2, Some(quality.minimum_certificate),)
                .unwrap(),
            [1, 3]
        );
    }

    #[test]
    fn shared_pages_are_homogeneous_and_validate_each_payload_slice() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 8,
            leaf_capacity: 128,
            support_sigma: 3.0,
        };
        let source: PlanarGaussian3d = (0..128)
            .map(|index| gaussian([index as f32 * 10.0, 0.0, 0.0], [0.01; 3], 0.5, 0.0))
            .collect();
        let output = build_planar_3d_lod(&source, settings).unwrap();
        let leaf_nodes: Vec<_> = output
            .manifest
            .nodes
            .iter()
            .filter(|node| node.is_leaf())
            .collect();
        assert_eq!(leaf_nodes.len(), 2);
        assert_eq!(
            leaf_nodes[0].representation.page,
            leaf_nodes[1].representation.page
        );
        assert_eq!(leaf_nodes[0].depth, leaf_nodes[1].depth);
        let shared_page_id = leaf_nodes[0].representation.page;

        let mut mixed = output.manifest.clone();
        mixed
            .pages
            .iter_mut()
            .find(|page| page.id == shared_page_id)
            .unwrap()
            .kind = LodPageKind::Mixed;
        assert!(matches!(
            mixed.validate(),
            Err(LodValidationError::InhomogeneousSharedNodePage(page))
                if page == shared_page_id
        ));

        let mut invalid_single_page_bound = output.manifest.clone();
        let root = &invalid_single_page_bound.nodes[0];
        let root_page = root.representation.page;
        invalid_single_page_bound
            .pages
            .iter_mut()
            .find(|page| page.id == root_page)
            .unwrap()
            .bounds = LodBounds::new([-10_000.0; 3], [10_000.0; 3]).unwrap();
        assert!(matches!(
            invalid_single_page_bound.validate(),
            Err(LodValidationError::RepresentationOutsideNode(node))
                if node == LodNodeId(1)
        ));

        let mut swapped = output.clone();
        let page_index = swapped
            .pages
            .iter()
            .position(|page| page.id == shared_page_id)
            .unwrap();
        let first = leaf_nodes[0].representation.offset as usize;
        let second = leaf_nodes[1].representation.offset as usize;
        swapped.pages[page_index].gaussians.swap(first, second);
        let content_hash = swapped.pages[page_index].content_hash();
        swapped
            .manifest
            .pages
            .iter_mut()
            .find(|page| page.id == shared_page_id)
            .unwrap()
            .content_hash = content_hash;
        swapped.manifest.validate().unwrap();
        assert!(matches!(
            swapped.validate(),
            Err(LodValidationError::RepresentationOutsideNode(_))
        ));
    }

    #[test]
    fn progressive_physical_pages_are_capped_independently_of_setting() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 8,
            leaf_capacity: 4_096,
            support_sigma: 3.0,
        };
        let source: PlanarGaussian3d = vec![gaussian([0.0; 3], [0.1; 3], 0.1, 0.0); 2_048].into();
        let output = build_planar_3d_lod(&source, settings).unwrap();
        assert!(
            output
                .manifest
                .nodes
                .iter()
                .filter(|node| node.is_leaf())
                .all(|node| node.source.count <= u64::from(PROGRESSIVE_LOGICAL_LEAF_CAPACITY))
        );
        assert!(
            output
                .manifest
                .pages
                .iter()
                .all(|page| page.gaussian_count <= PROGRESSIVE_PHYSICAL_PAGE_CAPACITY)
        );
        assert!(
            output
                .manifest
                .pages
                .iter()
                .any(|page| page.gaussian_count == PROGRESSIVE_PHYSICAL_PAGE_CAPACITY)
        );
    }

    #[test]
    fn missing_serialized_certificate_defaults_to_zero_and_fails_closed() {
        let output = build_planar_3d_lod(&fixture(4), GaussianLodBuildSettings::default()).unwrap();
        let mut encoded = serde_json::to_value(&output.manifest).unwrap();
        let nodes = encoded
            .get_mut("nodes")
            .and_then(serde_json::Value::as_array_mut)
            .unwrap();
        for node in nodes {
            node.as_object_mut()
                .unwrap()
                .remove("high_fidelity_certificate");
        }
        let decoded: GaussianLodManifest = serde_json::from_value(encoded).unwrap();
        assert!(
            decoded
                .nodes
                .iter()
                .all(|node| node.high_fidelity_certificate == 0.0)
        );
        assert!(matches!(
            decoded.validate(),
            Err(LodValidationError::InvalidHighFidelityCertificate(_))
                | Err(LodValidationError::NonMonotonicHighFidelityCertificate { .. })
        ));
    }

    #[test]
    fn default_builder_records_the_promoted_moment_merge_contract() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 3,
            support_sigma: 3.0,
        };
        let source = fixture(37);
        let output = build_planar_3d_lod(&source, settings).unwrap();
        assert_eq!(output.manifest.quality.coarsest_gaussian_count, 1);
        assert_eq!(output.manifest.build.reducer, LodReducerKind::MomentMerge);
        assert_eq!(
            output.manifest.build.builder_abi_version,
            PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION
        );
        assert_eq!(
            output.manifest.build.config_fingerprint,
            moment_merge_config_fingerprint(settings)
        );
        assert_ne!(
            output.manifest.build.config_fingerprint,
            settings.stable_hash()
        );
        assert_eq!(output.manifest.build.reducer_version, MOMENT_MERGE_VERSION);

        let mut stale = output.manifest.clone();
        stale.build.reducer_version = MOMENT_MERGE_VERSION - 1;
        assert!(matches!(
            stale.validate(),
            Err(LodValidationError::InvalidBuildVersion)
        ));

        let mut unknown_builder = output.manifest.clone();
        unknown_builder.build.builder_abi_version = u32::MAX;
        assert!(matches!(
            unknown_builder.validate(),
            Err(LodValidationError::InvalidBuildVersion)
        ));
    }

    #[test]
    fn adaptive_rung_reserves_integer_rounded_storage_budget() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 8,
            leaf_capacity: 16,
            support_sigma: 3.0,
        };
        let mut source = Vec::new();
        for block in 0..4 {
            let base = block as f32 * 10_000.0;
            for local in 0..16 {
                let x = match local {
                    0 | 1 => base,
                    2 | 3 => base + 100.0,
                    _ => base + local as f32 * 100.0,
                };
                source.push(gaussian([x, 0.0, 0.0], [0.01; 3], 0.5, 0.0));
            }
        }
        let cloud: PlanarGaussian3d = source.clone().into();
        let output = build_planar_3d_lod(&cloud, settings).unwrap();
        assert!(
            output.manifest.header.stored_gaussian_count
                <= 2 * output.manifest.header.source_gaussian_count
        );

        let deepest_parents: Vec<_> = output
            .manifest
            .nodes
            .iter()
            .filter(|node| {
                if node.is_leaf() {
                    return false;
                }
                let start = node.children.start as usize;
                let end = node.children.end().unwrap() as usize;
                output.manifest.nodes[start..end]
                    .iter()
                    .all(GaussianLodNode::is_leaf)
            })
            .collect();
        assert_eq!(deepest_parents.len(), 4);
        let admitted: Vec<_> = deepest_parents
            .iter()
            .copied()
            .filter(|node| node.high_fidelity_certificate >= HIGH_FIDELITY_PAIR_CERTIFICATE)
            .collect();
        assert_eq!(admitted.len(), 3);
        assert!(
            admitted
                .iter()
                .all(|node| { u64::from(node.representation.count) * 8 <= node.source.count * 7 })
        );
        assert!(deepest_parents.iter().all(|node| {
            let count = u64::from(node.representation.count);
            count * 2 >= node.source.count && count * 8 <= node.source.count * 7
        }));
        let adjusted = deepest_parents
            .iter()
            .copied()
            .find(|node| node.high_fidelity_certificate < HIGH_FIDELITY_PAIR_CERTIFICATE)
            .expect("integer storage adjustment must remain fail-closed");
        assert_eq!(adjusted.source.count, 16);
        assert_eq!(adjusted.representation.count, 13);

        source.reverse();
        let reversed: PlanarGaussian3d = source.into();
        assert_eq!(output, build_planar_3d_lod(&reversed, settings).unwrap());
    }

    #[test]
    fn bridge_storage_is_deterministic_and_bounded_for_awkward_sizes() {
        for branching_factor in [2, 3, 8, 32] {
            for source_count in [3, 17, 65, 129, 257] {
                let settings = GaussianLodBuildSettings {
                    branching_factor,
                    leaf_capacity: 128,
                    support_sigma: 3.0,
                };
                let source = (0..source_count)
                    .map(|index| {
                        gaussian(
                            [index as f32 * 10.0, 0.0, 0.0],
                            [0.01; 3],
                            0.5,
                            index as f32 * 0.25,
                        )
                    })
                    .collect::<Vec<_>>();
                let cloud: PlanarGaussian3d = source.clone().into();
                let output = build_planar_3d_lod(&cloud, settings).unwrap();
                output.validate().unwrap();
                assert!(
                    output.manifest.header.stored_gaussian_count
                        <= 2 * output.manifest.header.source_gaussian_count,
                    "branching={branching_factor}, source={source_count}"
                );

                for node in output.manifest.nodes.iter().filter(|node| {
                    if node.is_leaf() {
                        return false;
                    }
                    let start = node.children.start as usize;
                    let end = node.children.end().unwrap() as usize;
                    output.manifest.nodes[start..end]
                        .iter()
                        .all(GaussianLodNode::is_leaf)
                }) {
                    let representation_count = u64::from(node.representation.count);
                    assert!(representation_count * 2 >= node.source.count);
                    assert!(representation_count * 8 <= node.source.count * 7);
                }

                let mut reversed = source;
                reversed.reverse();
                let reversed: PlanarGaussian3d = reversed.into();
                assert_eq!(output, build_planar_3d_lod(&reversed, settings).unwrap());
            }
        }
    }

    #[test]
    fn default_progressive_hierarchy_bounds_every_refinement_and_storage_overhead() {
        const SOURCE_COUNT: usize = 65_536;
        let settings = GaussianLodBuildSettings::default();
        let source: PlanarGaussian3d =
            vec![gaussian([0.0; 3], [0.1; 3], 0.1, 0.0); SOURCE_COUNT].into();
        let output = build_planar_3d_lod(&source, settings).unwrap();
        output.validate().unwrap();

        assert_eq!(
            output.manifest.build.builder_abi_version,
            PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION
        );
        assert_eq!(output.manifest.header.node_count, 2_047);
        assert_eq!(output.manifest.header.page_count, 108);
        assert_ne!(
            output.manifest.header.required_features & LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES,
            0
        );
        assert!(
            output
                .manifest
                .pages
                .iter()
                .all(|page| page.gaussian_count <= PROGRESSIVE_PHYSICAL_PAGE_CAPACITY)
        );
        assert!(
            output
                .manifest
                .nodes
                .iter()
                .any(|node| node.representation.offset > 0)
        );

        for descriptor in &output.manifest.pages {
            let referencing: Vec<_> = output
                .manifest
                .nodes
                .iter()
                .filter(|node| node.representation.page == descriptor.id)
                .collect();
            let depth = referencing[0].depth;
            assert!(referencing.iter().all(|node| node.depth == depth));
            assert!(
                referencing.iter().all(|node| {
                    node.is_leaf() == (descriptor.kind == LodPageKind::SourceLeaves)
                })
            );
        }

        for node in &output.manifest.nodes {
            if node.is_leaf() {
                assert!(node.source.count <= u64::from(PROGRESSIVE_LOGICAL_LEAF_CAPACITY));
                continue;
            }
            let child_start = node.children.start as usize;
            let child_end = node.children.end().unwrap() as usize;
            assert_eq!(child_end - child_start, 2);
            let child_count = output.manifest.nodes[child_start..child_end]
                .iter()
                .map(|child| u64::from(child.representation.count))
                .sum::<u64>();
            let parent_count = u64::from(node.representation.count);
            assert!(child_count >= parent_count);
            assert!(
                child_count <= parent_count * u64::from(settings.branching_factor),
                "node {:?} refines from {parent_count} to {child_count} records",
                node.id
            );
            if output.manifest.nodes[child_start..child_end]
                .iter()
                .all(GaussianLodNode::is_leaf)
            {
                assert_eq!(parent_count, node.source.count.div_ceil(2));
                assert!(parent_count <= u64::from(PROGRESSIVE_LOGICAL_LEAF_CAPACITY));
                assert!(node.high_fidelity_certificate >= HIGH_FIDELITY_PAIR_CERTIFICATE);
                assert!(parent_count * 8 <= node.source.count * 7);
            }
        }

        let mut cut = output.manifest.roots.clone();
        let mut cut_counts = Vec::new();
        loop {
            let count = cut
                .iter()
                .map(|id| {
                    let index = usize::try_from(id.0 - 1).unwrap();
                    u64::from(output.manifest.nodes[index].representation.count)
                })
                .sum::<u64>();
            assert!(count > 0);
            if let Some(previous) = cut_counts.last() {
                assert!(count >= *previous);
                assert!(count <= *previous * u64::from(settings.branching_factor));
            }
            cut_counts.push(count);

            let mut refined = Vec::new();
            let mut changed = false;
            for id in cut {
                let node = &output.manifest.nodes[usize::try_from(id.0 - 1).unwrap()];
                if node.is_leaf() {
                    refined.push(id);
                } else {
                    changed = true;
                    let start = node.children.start as usize;
                    let end = node.children.end().unwrap() as usize;
                    refined.extend(output.manifest.nodes[start..end].iter().map(|node| node.id));
                }
            }
            if !changed {
                break;
            }
            cut = refined;
        }
        assert_eq!(
            cut_counts,
            [
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                512,
                4_096,
                32_768,
                SOURCE_COUNT as u64,
            ]
        );

        // The 2:1 rung contributes N/2 derived records. Every coarser default
        // rung is 8x smaller, so this exact power-of-two fixture stores
        // N + N/2 + N/16 + ... and remains comfortably below the 2N cap.
        let expected_stored =
            SOURCE_COUNT as u64 + 32_768 + 4_096 + 512 + 64 + 32 + 16 + 8 + 4 + 2 + 1;
        assert_eq!(
            output.manifest.header.stored_gaussian_count,
            expected_stored
        );
        assert!(output.manifest.header.stored_gaussian_count <= 2 * SOURCE_COUNT as u64);
    }

    #[test]
    fn progressive_manifest_validates_its_configured_amplification_bound() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 64,
            support_sigma: 3.0,
        };
        // A sufficiently wide deepest rung exercises the configured 4:1
        // parent/child representation bound. Tiny binary fixtures naturally
        // bottom out at 2:1 and cannot distinguish a forged factor of two.
        let mut output = build_planar_3d_lod(&fixture(256), settings).unwrap();
        output.manifest.build.settings.branching_factor = 2;
        output.manifest.build.config_fingerprint =
            moment_merge_config_fingerprint(output.manifest.build.settings);
        let error = output.manifest.validate().unwrap_err();
        assert!(
            matches!(
                error,
                LodValidationError::InvalidRefinementAmplification { .. }
            ),
            "unexpected validation error: {error:?}"
        );
    }

    #[test]
    fn builder_emits_valid_partitions_and_endpoint_counts() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 3,
            support_sigma: 2.5,
        };
        let output = build_planar_3d_lod(&fixture(37), settings).unwrap();
        output.validate().unwrap();
        assert_eq!(output.manifest.quality.coarsest_gaussian_count, 1);
        assert_eq!(output.manifest.quality.finest_gaussian_count, 37);
        assert!(output.manifest.quality.max_depth >= 2);

        for node in &output.manifest.nodes {
            if node.is_leaf() {
                assert!(node.source.count <= 3);
                assert_eq!(node.error, LodError::ZERO);
                assert_eq!(node.quality.max, 1.0);
            } else {
                assert!((2..=4).contains(&node.children.count));
            }
        }
    }

    #[test]
    fn manifest_rejects_broken_parent_bounds_and_error() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 2,
            support_sigma: 3.0,
        };
        let output = build_planar_3d_lod(&fixture(16), settings).unwrap();

        let mut broken_bounds = output.manifest.clone();
        broken_bounds.nodes[0].bounds.max = broken_bounds.nodes[0].bounds.min;
        assert!(matches!(
            broken_bounds.validate(),
            Err(LodValidationError::BoundsDoNotContainChild { .. })
                | Err(LodValidationError::SceneBoundsDoNotContainRoot(_))
                | Err(LodValidationError::RepresentationOutsideNode(_))
        ));

        let mut broken_error = output.manifest;
        broken_error.nodes[0].error = LodError::ZERO;
        assert!(matches!(
            broken_error.validate(),
            Err(LodValidationError::NonMonotonicError { .. })
                | Err(LodValidationError::QualityMetadataMismatch)
        ));
    }

    #[test]
    fn empty_and_singleton_clouds_are_valid() {
        let empty = build_planar_3d_lod(
            &PlanarGaussian3d::default(),
            GaussianLodBuildSettings::default(),
        )
        .unwrap();
        empty.validate().unwrap();
        assert!(empty.manifest.nodes.is_empty());

        let singleton: PlanarGaussian3d = vec![gaussian([0.0; 3], [0.1; 3], 0.5, 0.0)].into();
        let singleton =
            build_planar_3d_lod(&singleton, GaussianLodBuildSettings::default()).unwrap();
        singleton.validate().unwrap();
        assert_eq!(singleton.manifest.nodes.len(), 1);
        assert_eq!(singleton.manifest.quality.coarsest_gaussian_count, 1);
        assert_eq!(singleton.manifest.quality.finest_gaussian_count, 1);
    }
}
