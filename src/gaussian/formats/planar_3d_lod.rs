//! Deterministic CPU reference construction and validation for 3D Gaussian LoD.
//!
//! This module is intentionally renderer-independent.  It establishes the
//! versioned hierarchy/page contract and provides a bounded-complexity CPU
//! builder against which the bounded GPU hierarchy primitives are tested.

#[cfg(test)]
use std::cell::Cell;
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    error::Error,
    fmt,
    mem::size_of,
};

use bevy::math::{Mat3, Quat, Vec3};
#[cfg(feature = "sort_rayon")]
use rayon::prelude::*;
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
/// [`GaussianLodBuildSettings::branching_factor`]. ABI 14 also guarantees
/// renderer-compatible covariance, all-view projected-alpha conservation, a
/// conservative high-fidelity certificate (including anisotropy growth), and
/// risk-ranked adjacent agglomeration for ordinary
/// reductions averaging at most 16 source records per representative while preserving
/// a conservative balanced-partition selection-metadata envelope,
/// preserves the risk-ranked adjacent-pair bridge directly above 64-record
/// logical leaves, and packs logical node payloads into independently bounded
/// physical pages.
pub const PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION: u32 = 14;
#[cfg(test)]
thread_local! {
    static GAUSSIAN_SUPPORT_FULL_VALIDATIONS: Cell<u64> = const { Cell::new(0) };
}
// External CPU/GPU package builders use distinct, wide topologies. Their ABI
// values remain readable even though only the progressive CPU builder lives in
// this module.
const EXTERNAL_CPU_MOMENT_MERGE_BUILDER_ABI_VERSION: u32 = 5;
const EXTERNAL_GPU_MOMENT_MERGE_BUILDER_ABI_VERSION: u32 = 6;
/// Read-only external-memory progressive ABI. Each internal rung contains a
/// bounded number of v3 representatives accumulated directly from disjoint
/// intervals of the canonical source stream.
const EXTERNAL_PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION: u32 = 15;
/// External-memory progressive ABI with renderer-consistent spatial fitting
/// and a required monotone child-record to parent-record morph map.
pub(crate) const EXTERNAL_SPATIAL_MOMENT_MERGE_BUILDER_ABI_VERSION: u32 = 16;
/// External CPU/GPU builders retain their shared optical-depth-union reducer
/// until the GPU implementation can provide the same all-view proof as ABI 14.
pub(crate) const EXTERNAL_MOMENT_MERGE_VERSION: u32 = 2;
/// MomentMerge version 3 conservatively calibrates representative opacity so
/// projected alpha mass cannot inflate in any view. Version 2 fixed the
/// covariance convention, but widely separated surface splats could still
/// become a large, nearly opaque ellipsoid.
pub const MOMENT_MERGE_VERSION: u32 = 3;
/// MomentMerge version 4 retains v3's all-view projected-alpha ceiling and
/// adds bounded spatial fitting plus a monotone morph correspondence proof.
pub const SPATIAL_MOMENT_MERGE_VERSION: u32 = 4;
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
/// Every internal node has a compact, monotone mapping from its concatenated
/// immediate-child records to its own representation records.
pub const LOD_REQUIRED_FEATURE_MONOTONE_MORPH_MAP: u64 = 1 << 7;
pub const LOD_MORPH_MAP_SCHEMA_VERSION: u16 = 1;
pub const LOD_REQUIRED_FEATURE_SH_MASK: u64 = LOD_REQUIRED_FEATURE_SH0
    | LOD_REQUIRED_FEATURE_SH1
    | LOD_REQUIRED_FEATURE_SH2
    | LOD_REQUIRED_FEATURE_SH3
    | LOD_REQUIRED_FEATURE_SH4;
pub const LOD_CURRENT_SH_FEATURE: u64 = 1 << SH_DEGREE;
pub const LOD_CURRENT_REQUIRED_FEATURES: u64 =
    LOD_CURRENT_SH_FEATURE | LOD_REQUIRED_FEATURE_HIGH_FIDELITY_CERTIFICATE;
pub const LOD_SUPPORTED_REQUIRED_FEATURES: u64 = LOD_CURRENT_REQUIRED_FEATURES
    | LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES
    | LOD_REQUIRED_FEATURE_MONOTONE_MORPH_MAP;

const PROGRESSIVE_LOGICAL_LEAF_CAPACITY: u32 = 64;
const PROGRESSIVE_PHYSICAL_PAGE_CAPACITY: u32 = 1024;
const HIGH_FIDELITY_PAIR_CERTIFICATE: f32 = 0.95;
const HIGH_FIDELITY_MAX_REPRESENTATIVE_NUMERATOR: usize = 7;
const HIGH_FIDELITY_MAX_REPRESENTATIVE_DENOMINATOR: usize = 8;
/// Bound the more expensive risk-aware ordinary rung to the near-leaf regime.
/// Larger reductions retain the deterministic balanced reducer.
const PROGRESSIVE_RISK_AWARE_MAX_SOURCES_PER_REPRESENTATIVE: usize = 16;
/// Expensive linear builder passes poll cancellation at this record cadence.
/// Leaf and deepest-bridge work items are already capped at 64 and 128 source
/// records respectively, so one poll per such item is also bounded by this
/// limit.
const LOD_BUILD_CANCEL_CHECK_INTERVAL: usize = 256;
/// Fixed sorted runs bound the longest non-cooperative Morton-sort operation.
const LOD_MORTON_SORT_RUN_LEN: usize = 64 * 1024;

#[derive(Debug)]
enum CancelableLodBuildError {
    Canceled,
    Build(LodBuildError),
}

type CancelableLodBuildResult<T> = Result<T, CancelableLodBuildError>;

impl From<LodBuildError> for CancelableLodBuildError {
    fn from(error: LodBuildError) -> Self {
        Self::Build(error)
    }
}

#[derive(Clone, Copy)]
struct LodBuildCancellation<'a> {
    is_canceled: &'a (dyn Fn() -> bool + Sync),
}

impl LodBuildCancellation<'_> {
    #[inline]
    fn check(self) -> CancelableLodBuildResult<()> {
        if (self.is_canceled)() {
            Err(CancelableLodBuildError::Canceled)
        } else {
            Ok(())
        }
    }

    #[inline]
    fn poll(self, index: usize) -> CancelableLodBuildResult<()> {
        if index.is_multiple_of(LOD_BUILD_CANCEL_CHECK_INTERVAL) {
            self.check()?;
        }
        Ok(())
    }
}

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

fn moment_merge_config_fingerprint_for_reducer(
    build: GaussianLodBuildSettings,
    reducer_version: u32,
) -> u64 {
    let mut hash = StableHasher::new();
    // Preserve the promoted MomentMerge fingerprint byte-for-byte so valid
    // manifests do not change when the unused reducer configuration is removed.
    hash.write(b"BGSLOD MomentMerge config");
    hash.write(&build.stable_hash().to_le_bytes());
    hash.write(&reducer_version.to_le_bytes());
    hash.finish()
}

fn moment_merge_config_fingerprint(build: GaussianLodBuildSettings) -> u64 {
    moment_merge_config_fingerprint_for_reducer(build, MOMENT_MERGE_VERSION)
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
    lod_config_fingerprint_for_reducer(
        build,
        compressed_representative_sh_degree,
        MOMENT_MERGE_VERSION,
    )
}

pub(crate) fn lod_config_fingerprint_for_reducer(
    build: GaussianLodBuildSettings,
    compressed_representative_sh_degree: Option<u8>,
    reducer_version: u32,
) -> u64 {
    let base = moment_merge_config_fingerprint_for_reducer(build, reducer_version);
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
    /// Whether the manifest guarantees monotonic, configured
    /// parent-to-children representation-count amplification.
    pub const fn has_bounded_refinement_amplification(&self) -> bool {
        is_bounded_refinement_moment_merge_builder_abi(self.builder_abi_version)
            && matches!(self.reducer, LodReducerKind::MomentMerge)
    }

    /// ABI 14 additionally fixes the progressive in-memory topology to binary
    /// parent/child replacement. ABI 15 retains the configured external
    /// branching factor while providing the same amplification bound.
    const fn has_binary_progressive_topology(&self) -> bool {
        is_progressive_moment_merge_builder_abi(self.builder_abi_version)
            && matches!(self.reducer, LodReducerKind::MomentMerge)
    }
}

const fn is_progressive_moment_merge_builder_abi(builder_abi_version: u32) -> bool {
    builder_abi_version == PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION
}

const fn is_bounded_refinement_moment_merge_builder_abi(builder_abi_version: u32) -> bool {
    matches!(
        builder_abi_version,
        PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION
            | EXTERNAL_PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION
            | EXTERNAL_SPATIAL_MOMENT_MERGE_BUILDER_ABI_VERSION
    )
}

const fn moment_merge_reducer_version_for_builder_abi(builder_abi_version: u32) -> Option<u32> {
    match builder_abi_version {
        EXTERNAL_CPU_MOMENT_MERGE_BUILDER_ABI_VERSION
        | EXTERNAL_GPU_MOMENT_MERGE_BUILDER_ABI_VERSION => Some(EXTERNAL_MOMENT_MERGE_VERSION),
        PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION
        | EXTERNAL_PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION => Some(MOMENT_MERGE_VERSION),
        EXTERNAL_SPATIAL_MOMENT_MERGE_BUILDER_ABI_VERSION => Some(SPATIAL_MOMENT_MERGE_VERSION),
        _ => None,
    }
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

/// Compact morph correspondence for ABI 16 hierarchies.
///
/// `node_runs` is index-aligned with [`GaussianLodManifest::nodes`]. For one
/// internal node, the referenced u16 values are ordered by parent-local record
/// index. Run `p` maps the next `child_run_lengths[p]` records from the
/// concatenation of the node's immediate children to parent-local record `p`.
/// Children are concatenated in manifest child-range order and each child's
/// records retain their page-local representation order. Positive runs make
/// the mapping monotone and surjective without storing a parent index per run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GaussianLodMorphMap {
    pub schema_version: u16,
    pub node_runs: Vec<LodIndexRange>,
    pub child_run_lengths: Vec<u16>,
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
    /// Required for ABI 16 and absent from every older readable ABI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub morph_map: Option<GaussianLodMorphMap>,
}

impl GaussianLodManifest {
    /// Stable manifest-global run range for one node index.
    #[inline]
    pub fn morph_child_run_range_at(&self, node_index: usize) -> Option<LodIndexRange> {
        self.morph_map.as_ref()?.node_runs.get(node_index).copied()
    }

    /// Zero-allocation run slice for one node index.
    pub fn morph_child_run_lengths_at(&self, node_index: usize) -> Option<&[u16]> {
        let morph_map = self.morph_map.as_ref()?;
        let range = *morph_map.node_runs.get(node_index)?;
        let end = range.end()? as usize;
        morph_map.child_run_lengths.get(range.start as usize..end)
    }

    /// Parent-local record corresponding to a concatenated child-local record.
    pub fn morph_parent_record_at(
        &self,
        node_index: usize,
        child_record_index: u32,
    ) -> Option<u16> {
        let mut child_start = 0_u32;
        for (parent_record, run) in self
            .morph_child_run_lengths_at(node_index)?
            .iter()
            .copied()
            .enumerate()
        {
            let child_end = child_start.checked_add(u32::from(run))?;
            if child_record_index < child_end {
                return u16::try_from(parent_record).ok();
            }
            child_start = child_end;
        }
        None
    }

    /// Node-id convenience wrapper. Runtime upload paths should retain the
    /// validated manifest index and use [`Self::morph_child_run_range_at`].
    pub fn morph_child_run_range(&self, node: LodNodeId) -> Option<LodIndexRange> {
        let node_index = self
            .nodes
            .iter()
            .position(|candidate| candidate.id == node)?;
        self.morph_child_run_range_at(node_index)
    }

    /// Node-id convenience oracle for tests and CPU transition planning.
    pub fn morph_parent_record(&self, node: LodNodeId, child_record_index: u32) -> Option<u16> {
        let node_index = self
            .nodes
            .iter()
            .position(|candidate| candidate.id == node)?;
        self.morph_parent_record_at(node_index, child_record_index)
    }

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
        if self.build.reducer != LodReducerKind::MomentMerge
            || moment_merge_reducer_version_for_builder_abi(self.build.builder_abi_version)
                != Some(self.build.reducer_version)
        {
            return Err(LodValidationError::InvalidBuildVersion);
        }
        let morph_feature =
            self.header.required_features & LOD_REQUIRED_FEATURE_MONOTONE_MORPH_MAP != 0;
        let spatial_builder =
            self.build.builder_abi_version == EXTERNAL_SPATIAL_MOMENT_MERGE_BUILDER_ABI_VERSION;
        match (spatial_builder, morph_feature, self.morph_map.as_ref()) {
            (true, true, Some(morph_map)) => {
                if morph_map.schema_version != LOD_MORPH_MAP_SCHEMA_VERSION {
                    return Err(LodValidationError::UnsupportedMorphMapVersion(
                        morph_map.schema_version,
                    ));
                }
                if self.build.settings.leaf_capacity > u32::from(u16::MAX) {
                    return Err(LodValidationError::MorphRecordCapacityExceeded(
                        self.build.settings.leaf_capacity,
                    ));
                }
            }
            (true, false, _) => {
                return Err(LodValidationError::MissingMonotoneMorphMapFeature);
            }
            (true, true, None) => return Err(LodValidationError::MissingMorphMap),
            (false, true, _) | (false, false, Some(_)) => {
                return Err(LodValidationError::UnexpectedMorphMap);
            }
            (false, false, None) => {}
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
        if let Some(morph_map) = &self.morph_map {
            if morph_map.node_runs.len() != self.nodes.len() {
                return Err(LodValidationError::MorphNodeCountMismatch);
            }
            let mut expected_start = 0_u32;
            for (node_index, range) in morph_map.node_runs.iter().copied().enumerate() {
                let end = range
                    .end()
                    .ok_or(LodValidationError::InvalidMorphRunRange(node_index))?;
                if range.start != expected_start || end as usize > morph_map.child_run_lengths.len()
                {
                    return Err(LodValidationError::InvalidMorphRunRange(node_index));
                }
                expected_start = end;
            }
            if expected_start as usize != morph_map.child_run_lengths.len() {
                return Err(LodValidationError::MorphRunCoverageMismatch);
            }
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
            != lod_config_fingerprint_for_reducer(
                self.build.settings,
                compressed_representative_sh_degree,
                self.build.reducer_version,
            )
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
                if self
                    .morph_child_run_range_at(index)
                    .is_some_and(|range| !range.is_empty())
                {
                    return Err(LodValidationError::InvalidLeafMorphRuns(node.id));
                }
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
            let bounded_refinement = self.build.has_bounded_refinement_amplification();
            if (self.build.has_binary_progressive_topology() && child_end - child_start != 2)
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
            if bounded_refinement {
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
            if let Some(run_lengths) = self.morph_child_run_lengths_at(index) {
                if run_lengths.len() != node.representation.count as usize {
                    return Err(LodValidationError::InvalidMorphRunCount {
                        node: node.id,
                        expected: node.representation.count,
                        actual: run_lengths.len(),
                    });
                }
                let mut mapped_child_count = 0_u64;
                for &run_length in run_lengths {
                    if run_length == 0 {
                        return Err(LodValidationError::ZeroMorphRun(node.id));
                    }
                    mapped_child_count = mapped_child_count
                        .checked_add(u64::from(run_length))
                        .ok_or(LodValidationError::CountOverflow("morph child records"))?;
                }
                if mapped_child_count != child_representation_count {
                    return Err(LodValidationError::MorphChildCoverageMismatch {
                        node: node.id,
                        expected: child_representation_count,
                        actual: mapped_child_count,
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
        self.accumulate_validated(gaussians)?
            .finish(self.support_sigma)
    }

    /// Reducer used by external CPU builds and as the CPU oracle for the
    /// external GPU builder. Their ABI v2 contract retains raw optical-depth
    /// union opacity until both implementations can promote together.
    #[cfg(test)]
    pub(crate) fn reduce_external_v2(
        &self,
        gaussians: &[Gaussian3d],
    ) -> Result<MomentMergeResult, LodBuildError> {
        self.accumulate_validated(gaussians)?
            .finish_external_v2(self.support_sigma)
    }

    fn accumulate_validated(
        &self,
        gaussians: &[Gaussian3d],
    ) -> Result<MomentAccumulator, LodBuildError> {
        if gaussians.is_empty() {
            return Err(LodBuildError::EmptyReduction);
        }
        let mut accumulator = MomentAccumulator::new();
        for (index, gaussian) in gaussians.iter().enumerate() {
            validate_gaussian(gaussian)
                .map_err(|field| LodBuildError::InvalidGaussian { index, field })?;
            accumulator.add(gaussian, self.support_sigma)?;
        }
        Ok(accumulator)
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
    pub(crate) fn high_fidelity_certificate(&self) -> f32 {
        let appearance_certificate = appearance_error_certificate(self.error.appearance);
        self.raster_risk
            .high_fidelity_certificate()
            .min(appearance_certificate)
    }
}

#[cfg(any(feature = "lod_build", test))]
const SPATIAL_FIT_MAX_RELATIVE_BOUNDARY_ERROR: f64 = 0.10;
#[cfg(any(feature = "lod_build", test))]
const SPATIAL_FIT_MIN_REFERENCE_ALPHA: f64 = 1e-4;
#[cfg(any(feature = "lod_build", test))]
const SPATIAL_FIT_TANGENT_FACTORS: [f32; 5] = [1.031_25, 1.062_5, 1.125, 1.25, 1.5];
#[cfg(any(feature = "lod_build", test))]
const SPATIAL_FIT_MAX_SIBLING_NODES: usize = 32;
#[cfg(any(feature = "lod_build", test))]
const SPATIAL_FIT_TANGENT_GRID: [f32; 3] = [0.0, 0.5, 1.0];
#[cfg(any(feature = "lod_build", test))]
const SPATIAL_FIT_PROJECTED_SCALE_FACTORS: [f64; 7] = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0];
#[cfg(any(feature = "lod_build", test))]
const SPATIAL_FIT_MAX_PROBES_PER_NODE_PAIR: usize =
    SPATIAL_FIT_TANGENT_GRID.len() * SPATIAL_FIT_TANGENT_GRID.len();
#[cfg(any(feature = "lod_build", test))]
const SPATIAL_FIT_SAMPLE_POINTS_PER_DIRECTION: usize = 3;

/// One bounded ABI 16 node participating in a sibling-cohort spatial fit.
///
/// Risk-aware rungs retain their original records and exact representative
/// source partitions until their at-most-32-node sibling cohort is complete.
/// Coarser streamed rungs use `None` and therefore take the conservative
/// selection-error path instead of buffering an unbounded source interval.
#[cfg(any(feature = "lod_build", test))]
#[derive(Clone, Debug)]
pub(crate) struct SpatialMomentMergeNode {
    pub(crate) representatives: Vec<MomentMergeResult>,
    pub(crate) source_records: Option<Vec<Gaussian3d>>,
    pub(crate) source_ranges: Vec<std::ops::Range<usize>>,
    pub(crate) authored_support_bounds: LodBounds,
    pub(crate) spatial_certificate_cap: f32,
    pub(crate) spatial_geometric_error_floor: f32,
}

#[cfg(any(feature = "lod_build", test))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SpatialMomentMergeFitReport {
    /// All authored-support touching pairs inside this future-parent cohort.
    pub(crate) touching_node_pairs: u32,
    /// Touching pairs with retained source partitions and therefore measured.
    pub(crate) overlapping_node_pairs: u32,
    /// Touching coarse pairs lacking retained source payload. These are left
    /// unchanged instead of being assigned an artificial infinite error.
    pub(crate) unmeasured_touching_node_pairs: u32,
    pub(crate) accepted_edits: u32,
    pub(crate) unsafe_node_pairs: u32,
    pub(crate) maximum_relative_boundary_error_before: f32,
    pub(crate) maximum_relative_boundary_error_after: f32,
    pub(crate) cohort_composited_error_before: f64,
    pub(crate) cohort_composited_error_after: f64,
}

#[cfg(any(feature = "lod_build", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SpatialBoundaryProbe {
    left_node: usize,
    left_representative: usize,
    right_node: usize,
    right_representative: usize,
}

#[cfg(any(feature = "lod_build", test))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SpatialBoundaryMetrics {
    relative_boundary_error: f64,
    composited_error: f64,
    relative_boundary_error_by_scale: [f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()],
    composited_error_by_scale: [f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()],
}

/// Probe geometry and flat-source alpha samples which cannot change during a
/// spatial fit. Candidate edits retain both representative centers and source
/// partitions, so computing this table once is byte-equivalent to replaying
/// the source renderer for every widening candidate.
#[cfg(any(feature = "lod_build", test))]
#[derive(Clone, Debug, PartialEq)]
struct SpatialBoundaryReference {
    characteristic_pixels_per_world: [f64; PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
    sample_points: [[[f64; 2]; SPATIAL_FIT_SAMPLE_POINTS_PER_DIRECTION];
        PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
    source_alpha: [[[f64; SPATIAL_FIT_SAMPLE_POINTS_PER_DIRECTION];
        SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()];
        PROJECTED_ALPHA_MASS_DIRECTIONS.len()],
}

#[cfg(any(feature = "lod_build", test))]
#[derive(Clone, Copy)]
enum SpatialBoundaryMetricMode {
    Cached,
    #[cfg(test)]
    BruteForce,
}

#[cfg(any(feature = "lod_build", test))]
#[derive(Clone, Debug)]
struct SpatialRepresentativeOverride {
    node: usize,
    representative: usize,
    value: MomentMergeResult,
}

#[cfg(any(feature = "lod_build", test))]
type SpatialFitCandidate = (
    f64,
    f64,
    usize,
    Vec<SpatialRepresentativeOverride>,
    Vec<(usize, SpatialBoundaryMetrics)>,
);

#[cfg(any(feature = "lod_build", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SpatialMomentMergeFitBounds {
    pub(crate) node_pair_checks: u64,
    pub(crate) boundary_probes: u64,
    pub(crate) scratch_host_bytes: u64,
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_fit_explicit_vec_payload_bytes(
    node_pair_capacity: usize,
    probe_capacity: usize,
) -> Option<usize> {
    probe_capacity
        // Frozen probe keys plus one at-most-nine-key pair-probe temporary.
        .checked_mul(size_of::<SpatialBoundaryProbe>())?
        .checked_add(
            SPATIAL_FIT_MAX_PROBES_PER_NODE_PAIR.checked_mul(size_of::<SpatialBoundaryProbe>())?,
        )?
        // One immutable flat-source reference table per frozen probe.
        .checked_add(probe_capacity.checked_mul(size_of::<SpatialBoundaryReference>())?)?
        // Current and initial metrics remain live throughout candidate search.
        .checked_add(
            probe_capacity.checked_mul(size_of::<SpatialBoundaryMetrics>().checked_mul(2)?)?,
        )?
        // Two sorted incidence entries per probe.
        .checked_add(
            probe_capacity.checked_mul(size_of::<(usize, usize, usize)>().checked_mul(2)?)?,
        )?
        // Before sort/dedup, two edited representatives can each contribute
        // every probe index.
        .checked_add(probe_capacity.checked_mul(size_of::<usize>().checked_mul(2)?)?)?
        // Current and retained-best affected metrics may coexist.
        .checked_add(
            probe_capacity
                .checked_mul(size_of::<(usize, SpatialBoundaryMetrics)>().checked_mul(2)?)?,
        )?
        // Unsafe-pair aggregation is preallocated to the exact node-pair cap.
        .checked_add(node_pair_capacity.checked_mul(size_of::<(usize, usize, f64)>())?)?
        // Current and retained-best candidates each own at most two edits.
        .checked_add(size_of::<SpatialRepresentativeOverride>().checked_mul(4)?)
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_fit_scratch_host_bytes(
    node_count: usize,
    node_pair_capacity: usize,
    probe_capacity: usize,
) -> Option<usize> {
    spatial_fit_explicit_vec_payload_bytes(node_pair_capacity, probe_capacity)?
        // The caller's bounded cohort-node payload participates in the fit.
        .checked_add(node_count.checked_mul(size_of::<SpatialMomentMergeNode>())?)?
        // One precomputed widening per side is held outside the candidate Vecs.
        .checked_add(size_of::<MomentMergeResult>().checked_mul(2)?)
}

/// Allocation bound for one ABI 16 sibling-cohort fit. The validated
/// branching limit keeps the outer pair count at 496 and the fixed tangential
/// grid keeps the probe count at 4,464 without a representative cross product.
#[cfg(any(feature = "lod_build", test))]
pub(crate) fn spatial_moment_merge_fit_bounds(
    node_count: usize,
) -> Option<SpatialMomentMergeFitBounds> {
    if node_count > SPATIAL_FIT_MAX_SIBLING_NODES {
        return None;
    }
    let node_pair_checks = node_count.checked_mul(node_count.saturating_sub(1))? / 2;
    let boundary_probes = node_pair_checks.checked_mul(SPATIAL_FIT_MAX_PROBES_PER_NODE_PAIR)?;
    let scratch_host_bytes =
        spatial_fit_scratch_host_bytes(node_count, node_pair_checks, boundary_probes)?;
    Some(SpatialMomentMergeFitBounds {
        node_pair_checks: node_pair_checks.try_into().ok()?,
        boundary_probes: boundary_probes.try_into().ok()?,
        scratch_host_bytes: scratch_host_bytes.try_into().ok()?,
    })
}

/// Fit all geometrically adjacent siblings, not merely consecutive Morton
/// intervals. The pair bound is `B*(B-1)/2 <= 496` for the validated maximum
/// branching factor. Each pair samples a fixed 3x3 grid over its two tangential
/// overlap axes and deterministically selects the nearest contributor from each
/// node, deduplicating the resulting at-most-nine representative pairs. This
/// covers elongated seams without a representative cross product. Candidate
/// edits keep centers fixed, widen only the two tangent axes, remain inside
/// that target pair's authored support AABB (never a looser
/// disconnected-sibling cohort box), and rerun the existing all-view opacity
/// ceiling. Source references use the flat renderer's adaptive support cutoff;
/// emitted representatives use the LoD candidate's at-least-3-sigma cutoff.
/// Both are compared across a deterministic 0.0625x..4x pixel-scale ladder
/// normalized by projected source sigma, so the fixed 0.3-pixel mip variance
/// cannot make an arbitrary one-pixel/world fit look universally safe.
///
/// An edit is committed only when its target boundary strictly improves, every
/// other boundary touched by either edited representative does not regress,
/// and the sum of renderer-consistent composited probe error over the cohort
/// does not increase. A candidate must also preserve every contributor-order
/// key: centers, authored node bounds, probe targets, and representative index
/// are immutable, while the only support-dependent key is whether a
/// representative overlaps the other authored node. Requiring that overlap
/// bit to remain unchanged against every node proves that the frozen 3x3 probe
/// topology and its incidence index remain valid after each sequential edit.
/// If no admissible edit repairs an overlapping boundary, both nodes receive a
/// selection-visible geometric-error floor spanning that sibling pair; the
/// high-fidelity cap is lowered as an additional q>=.90 signal. This makes
/// ordinary q=.65 traversal refine past a known-bad spatial representation
/// instead of displaying a page/node grid.
#[cfg(any(feature = "lod_build", test))]
pub(crate) fn fit_spatial_moment_merge_sibling_cohort(
    nodes: &mut [SpatialMomentMergeNode],
    support_sigma: f32,
) -> Result<SpatialMomentMergeFitReport, LodBuildError> {
    fit_spatial_moment_merge_sibling_cohort_with_mode(
        nodes,
        support_sigma,
        SpatialBoundaryMetricMode::Cached,
    )
}

#[cfg(test)]
fn fit_spatial_moment_merge_sibling_cohort_brute_force(
    nodes: &mut [SpatialMomentMergeNode],
    support_sigma: f32,
) -> Result<SpatialMomentMergeFitReport, LodBuildError> {
    fit_spatial_moment_merge_sibling_cohort_with_mode(
        nodes,
        support_sigma,
        SpatialBoundaryMetricMode::BruteForce,
    )
}

#[cfg(any(feature = "lod_build", test))]
fn fit_spatial_moment_merge_sibling_cohort_with_mode(
    nodes: &mut [SpatialMomentMergeNode],
    support_sigma: f32,
    metric_mode: SpatialBoundaryMetricMode,
) -> Result<SpatialMomentMergeFitReport, LodBuildError> {
    if nodes.len() < 2 {
        return Ok(SpatialMomentMergeFitReport::default());
    }
    if nodes.len() > SPATIAL_FIT_MAX_SIBLING_NODES {
        return Err(LodBuildError::CountOverflow("spatial sibling cohort"));
    }
    for node in nodes.iter() {
        if node.representatives.is_empty() {
            return Err(LodBuildError::EmptyReduction);
        }
        if let Some(source) = &node.source_records {
            validate_spatial_source_partition(source.len(), &node.source_ranges)?;
            if node.source_ranges.len() != node.representatives.len() {
                return Err(LodBuildError::CountOverflow(
                    "spatial representative source partition",
                ));
            }
        } else if !node.source_ranges.is_empty() {
            return Err(LodBuildError::CountOverflow(
                "streamed spatial source partition",
            ));
        }
    }

    let maximum_node_pairs = nodes.len() * nodes.len().saturating_sub(1) / 2;
    debug_assert!(maximum_node_pairs <= 496);
    let mut probes = Vec::with_capacity(maximum_node_pairs * SPATIAL_FIT_MAX_PROBES_PER_NODE_PAIR);
    debug_assert_eq!(
        probes.capacity(),
        maximum_node_pairs * SPATIAL_FIT_MAX_PROBES_PER_NODE_PAIR
    );
    let mut touching_node_pairs = 0_u32;
    let mut overlapping_node_pairs = 0_u32;
    let mut unmeasured_touching_node_pairs = 0_u32;
    for left_node in 0..nodes.len() {
        for right_node in left_node + 1..nodes.len() {
            if !lod_bounds_touch_or_overlap(
                nodes[left_node].authored_support_bounds,
                nodes[right_node].authored_support_bounds,
            ) {
                continue;
            }
            touching_node_pairs = touching_node_pairs
                .checked_add(1)
                .ok_or(LodBuildError::CountOverflow("touching spatial node pairs"))?;
            // Only measured risk-aware cohorts authorize a spatial edit or a
            // selection-error floor. Coarse streamed rungs deliberately carry
            // no source payload; treating that absence as infinite error would
            // conservatively refine every coarse node and destroy useful LoD.
            if nodes[left_node].source_records.is_none()
                || nodes[right_node].source_records.is_none()
            {
                unmeasured_touching_node_pairs =
                    unmeasured_touching_node_pairs.checked_add(1).ok_or(
                        LodBuildError::CountOverflow("unmeasured touching spatial node pairs"),
                    )?;
                continue;
            }
            overlapping_node_pairs =
                overlapping_node_pairs
                    .checked_add(1)
                    .ok_or(LodBuildError::CountOverflow(
                        "overlapping spatial node pairs",
                    ))?;
            probes.extend(spatial_boundary_probes_for_pair(
                nodes, left_node, right_node,
            ));
        }
    }
    if probes.is_empty() {
        return Ok(SpatialMomentMergeFitReport {
            touching_node_pairs,
            overlapping_node_pairs,
            unmeasured_touching_node_pairs,
            ..Default::default()
        });
    }
    let mut references = Vec::with_capacity(probes.len());
    debug_assert_eq!(references.capacity(), probes.len());
    for probe in probes.iter().copied() {
        references.push(spatial_boundary_reference(nodes, probe)?);
    }
    let mut metrics = Vec::with_capacity(probes.len());
    debug_assert_eq!(metrics.capacity(), probes.len());
    for (probe, reference) in probes.iter().copied().zip(&references) {
        metrics.push(spatial_boundary_metrics_with_mode(
            nodes,
            probe,
            reference,
            &[],
            metric_mode,
        )?);
    }
    let mut initial_metrics = Vec::with_capacity(metrics.len());
    debug_assert_eq!(initial_metrics.capacity(), metrics.len());
    initial_metrics.extend_from_slice(&metrics);
    let probe_incidence = spatial_probe_incidence(&probes);
    let mut cohort_error_by_scale = [0.0_f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()];
    for metric in &metrics {
        for (cohort, error) in cohort_error_by_scale
            .iter_mut()
            .zip(metric.composited_error_by_scale)
        {
            *cohort += error;
        }
    }
    let mut accepted_edits = 0_u32;

    for probe_index in 0..probes.len() {
        let baseline = metrics[probe_index];
        if baseline.relative_boundary_error <= SPATIAL_FIT_MAX_RELATIVE_BOUNDARY_ERROR {
            continue;
        }
        let probe = probes[probe_index];
        let pair_support_envelope = nodes[probe.left_node]
            .authored_support_bounds
            .union(nodes[probe.right_node].authored_support_bounds);
        let Some(left_source) =
            spatial_probe_source(nodes, probe.left_node, probe.left_representative)
        else {
            continue;
        };
        let Some(right_source) =
            spatial_probe_source(nodes, probe.right_node, probe.right_representative)
        else {
            continue;
        };

        let mut best: Option<SpatialFitCandidate> = None;
        let factor_modes = [(true, true), (true, false), (false, true)];
        for (factor_index, factor) in SPATIAL_FIT_TANGENT_FACTORS.iter().copied().enumerate() {
            // The three modes share the same two source partitions and factor.
            // Materialize each widening once, then clone its immutable result
            // into the candidate modes which need it.
            let widened_left = spatial_widened_representative(
                left_source,
                &nodes[probe.left_node].representatives[probe.left_representative],
                factor,
                support_sigma,
            )?;
            let widened_left = match widened_left {
                Some(value)
                    if oriented_support_inside(
                        &value.gaussian,
                        support_sigma,
                        pair_support_envelope,
                    )? =>
                {
                    Some(value)
                }
                _ => None,
            };
            let widened_right = spatial_widened_representative(
                right_source,
                &nodes[probe.right_node].representatives[probe.right_representative],
                factor,
                support_sigma,
            )?;
            let widened_right = match widened_right {
                Some(value)
                    if oriented_support_inside(
                        &value.gaussian,
                        support_sigma,
                        pair_support_envelope,
                    )? =>
                {
                    Some(value)
                }
                _ => None,
            };
            for (mode_index, (widen_left, widen_right)) in factor_modes.iter().copied().enumerate()
            {
                let mut overrides = Vec::with_capacity(2);
                if widen_left {
                    let Some(value) = widened_left.clone() else {
                        continue;
                    };
                    overrides.push(SpatialRepresentativeOverride {
                        node: probe.left_node,
                        representative: probe.left_representative,
                        value,
                    });
                }
                if widen_right {
                    let Some(value) = widened_right.clone() else {
                        continue;
                    };
                    overrides.push(SpatialRepresentativeOverride {
                        node: probe.right_node,
                        representative: probe.right_representative,
                        value,
                    });
                }
                if !spatial_candidate_preserves_probe_topology(nodes, &overrides) {
                    continue;
                }

                let affected_indices = spatial_affected_probe_indices(
                    &probe_incidence,
                    overrides
                        .iter()
                        .map(|candidate| (candidate.node, candidate.representative)),
                );
                let mut affected = Vec::with_capacity(affected_indices.len());
                let mut candidate_cohort_error_by_scale = cohort_error_by_scale;
                let mut all_boundaries_no_worse = true;
                for affected_index in affected_indices {
                    let candidate_metrics = spatial_boundary_metrics_with_mode(
                        nodes,
                        probes[affected_index],
                        &references[affected_index],
                        &overrides,
                        metric_mode,
                    )?;
                    for (cohort_error, (current_error, candidate_error)) in
                        candidate_cohort_error_by_scale.iter_mut().zip(
                            metrics[affected_index]
                                .composited_error_by_scale
                                .into_iter()
                                .zip(candidate_metrics.composited_error_by_scale),
                        )
                    {
                        *cohort_error = *cohort_error - current_error + candidate_error;
                    }
                    all_boundaries_no_worse &= spatial_boundary_metrics_no_worse(
                        candidate_metrics,
                        metrics[affected_index],
                    );
                    affected.push((affected_index, candidate_metrics));
                }
                let target = affected
                    .iter()
                    .find_map(|(index, metrics)| (*index == probe_index).then_some(*metrics))
                    .expect("target boundary is affected by its own candidate");
                let candidate_cohort_error = candidate_cohort_error_by_scale.iter().sum::<f64>();
                if !spatial_boundary_metrics_strictly_better(target, baseline)
                    || !all_boundaries_no_worse
                    || !candidate_cohort_error_by_scale
                        .iter()
                        .copied()
                        .zip(cohort_error_by_scale)
                        .all(|(candidate, current)| float_no_worse(candidate, current))
                {
                    continue;
                }
                let order = factor_index * factor_modes.len() + mode_index;
                let candidate_key = (
                    target.relative_boundary_error,
                    candidate_cohort_error,
                    order,
                );
                if best.as_ref().is_none_or(|current| {
                    candidate_key.0.total_cmp(&current.0).is_lt()
                        || (candidate_key.0.total_cmp(&current.0).is_eq()
                            && (candidate_key.1.total_cmp(&current.1).is_lt()
                                || (candidate_key.1.total_cmp(&current.1).is_eq()
                                    && candidate_key.2 < current.2)))
                }) {
                    best = Some((
                        candidate_key.0,
                        candidate_key.1,
                        candidate_key.2,
                        overrides,
                        affected,
                    ));
                }
            }
        }
        if let Some((_, _, _, overrides, affected)) = best {
            for candidate in overrides {
                nodes[candidate.node].representatives[candidate.representative] = candidate.value;
            }
            for (index, value) in affected {
                for (cohort_error, (current_error, candidate_error)) in
                    cohort_error_by_scale.iter_mut().zip(
                        metrics[index]
                            .composited_error_by_scale
                            .into_iter()
                            .zip(value.composited_error_by_scale),
                    )
                {
                    *cohort_error = *cohort_error - current_error + candidate_error;
                }
                metrics[index] = value;
            }
            accepted_edits = accepted_edits
                .checked_add(1)
                .ok_or(LodBuildError::CountOverflow("spatial fit edits"))?;
        }
    }

    let mut unsafe_pairs = Vec::<(usize, usize, f64)>::with_capacity(maximum_node_pairs);
    debug_assert_eq!(unsafe_pairs.capacity(), maximum_node_pairs);
    for (probe, metric) in probes.iter().copied().zip(metrics.iter().copied()) {
        if metric.relative_boundary_error <= SPATIAL_FIT_MAX_RELATIVE_BOUNDARY_ERROR {
            continue;
        }
        if let Some((_, _, maximum_error)) = unsafe_pairs
            .iter_mut()
            .find(|(left, right, _)| *left == probe.left_node && *right == probe.right_node)
        {
            *maximum_error = maximum_error.max(metric.relative_boundary_error);
        } else {
            unsafe_pairs.push((
                probe.left_node,
                probe.right_node,
                metric.relative_boundary_error,
            ));
        }
    }
    for (left_node, right_node, maximum_error) in unsafe_pairs.iter().copied() {
        let pair_bounds = nodes[left_node]
            .authored_support_bounds
            .union(nodes[right_node].authored_support_bounds);
        let geometric_floor = pair_bounds.radius();
        let certificate_cap = (1.0 + maximum_error as f32).recip().min(0.5);
        for node_index in [left_node, right_node] {
            nodes[node_index].spatial_geometric_error_floor = nodes[node_index]
                .spatial_geometric_error_floor
                .max(geometric_floor);
            nodes[node_index].spatial_certificate_cap = nodes[node_index]
                .spatial_certificate_cap
                .min(certificate_cap);
        }
    }

    Ok(SpatialMomentMergeFitReport {
        touching_node_pairs,
        overlapping_node_pairs,
        unmeasured_touching_node_pairs,
        accepted_edits,
        unsafe_node_pairs: unsafe_pairs.len().try_into().unwrap_or(u32::MAX),
        maximum_relative_boundary_error_before: initial_metrics
            .iter()
            .map(|metric| metric.relative_boundary_error)
            .fold(0.0_f64, f64::max) as f32,
        maximum_relative_boundary_error_after: metrics
            .iter()
            .map(|metric| metric.relative_boundary_error)
            .fold(0.0_f64, f64::max) as f32,
        cohort_composited_error_before: initial_metrics
            .iter()
            .map(|metric| metric.composited_error)
            .sum(),
        cohort_composited_error_after: metrics.iter().map(|metric| metric.composited_error).sum(),
    })
}

/// Conservative SH appearance certificate shared by builders that add a page
/// encoding error after MomentMerge has produced its raster certificate.
pub(crate) fn appearance_error_certificate(appearance_error: f32) -> f32 {
    let coefficients_per_channel = (SH_COEFF_COUNT / 3).max(1) as f32;
    let worst_direction_factor = (SH_COEFF_COUNT as f32).sqrt()
        * (coefficients_per_channel / (4.0 * std::f32::consts::PI)).sqrt();
    let appearance_bound = worst_direction_factor * appearance_error.max(0.0);
    (1.0 + appearance_bound).recip().clamp(0.0, 1.0)
}

/// View-independent analytic warning signals for a MomentMerge representative.
///
/// These are preprocessing diagnostics, not an advertised quality estimate.
/// The sampled projection term evaluates a fixed, rotation-symmetric direction
/// set. The Minkowski upper bound is conservative for every orthographic view,
/// exact for identical source covariance, and may overestimate risk when source
/// covariance frames differ substantially. Both alpha-mass terms deliberately
/// describe the raw optical-depth-union representative before ABI 14 opacity
/// calibration. The emitted representative is all-view-safe, while retaining
/// the pre-calibration correction magnitude keeps hierarchy pairing and the
/// serialized fidelity certificate conservative.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MomentMergeRasterRisk {
    pub(crate) raw_sampled_projected_alpha_mass_inflation: f32,
    pub(crate) raw_projected_alpha_mass_inflation_upper_bound: f32,
    pub(crate) support_leakage_fraction: f32,
    pub(crate) support_growth: f32,
    pub(crate) major_scale_growth: f32,
    pub(crate) anisotropy_growth: f32,
}

impl MomentMergeRasterRisk {
    #[cfg(test)]
    fn score(self) -> f32 {
        (self.raw_sampled_projected_alpha_mass_inflation - 1.0)
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
        let alpha_mass = self.raw_projected_alpha_mass_inflation_upper_bound.max(1.0);
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

#[cfg(any(feature = "lod_build", test))]
fn validate_spatial_source_partition(
    source_count: usize,
    ranges: &[std::ops::Range<usize>],
) -> Result<(), LodBuildError> {
    let mut expected_start = 0_usize;
    for range in ranges {
        if range.start != expected_start || range.end <= range.start || range.end > source_count {
            return Err(LodBuildError::CountOverflow(
                "spatial representative source partition",
            ));
        }
        expected_start = range.end;
    }
    if expected_start != source_count {
        return Err(LodBuildError::CountOverflow(
            "spatial representative source coverage",
        ));
    }
    Ok(())
}

#[cfg(any(feature = "lod_build", test))]
fn lod_bounds_touch_or_overlap(left: LodBounds, right: LodBounds) -> bool {
    (0..3).all(|axis| left.max[axis] >= right.min[axis] && right.max[axis] >= left.min[axis])
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_boundary_probes_for_pair(
    nodes: &[SpatialMomentMergeNode],
    left_node: usize,
    right_node: usize,
) -> Vec<SpatialBoundaryProbe> {
    let left_bounds = nodes[left_node].authored_support_bounds;
    let right_bounds = nodes[right_node].authored_support_bounds;
    let left_center = left_bounds.center();
    let right_center = right_bounds.center();
    let separation_axis = (0..3)
        .max_by(|left_axis, right_axis| {
            (left_center[*left_axis] - right_center[*left_axis])
                .abs()
                .total_cmp(&(left_center[*right_axis] - right_center[*right_axis]).abs())
                .then_with(|| right_axis.cmp(left_axis))
        })
        .expect("three-dimensional bounds contain an axis");
    let tangent_axes = match separation_axis {
        0 => [1, 2],
        1 => [0, 2],
        2 => [0, 1],
        _ => unreachable!("three-dimensional bounds contain only three axes"),
    };
    let overlap_min = [0, 1, 2].map(|axis| left_bounds.min[axis].max(right_bounds.min[axis]));
    let overlap_max = [0, 1, 2].map(|axis| left_bounds.max[axis].min(right_bounds.max[axis]));
    let mut probes = Vec::with_capacity(SPATIAL_FIT_MAX_PROBES_PER_NODE_PAIR);
    debug_assert_eq!(probes.capacity(), SPATIAL_FIT_MAX_PROBES_PER_NODE_PAIR);
    for first_tangent in SPATIAL_FIT_TANGENT_GRID {
        for second_tangent in SPATIAL_FIT_TANGENT_GRID {
            let mut target = [0.0_f32; 3];
            target[separation_axis] =
                0.5 * (overlap_min[separation_axis] + overlap_max[separation_axis]);
            for (axis, factor) in tangent_axes
                .into_iter()
                .zip([first_tangent, second_tangent])
            {
                target[axis] = overlap_min[axis] + factor * (overlap_max[axis] - overlap_min[axis]);
            }
            let probe = SpatialBoundaryProbe {
                left_node,
                left_representative: spatial_nearest_boundary_contributor(
                    &nodes[left_node],
                    right_bounds,
                    target,
                ),
                right_node,
                right_representative: spatial_nearest_boundary_contributor(
                    &nodes[right_node],
                    left_bounds,
                    target,
                ),
            };
            if !probes.contains(&probe) {
                probes.push(probe);
            }
        }
    }
    debug_assert!(!probes.is_empty());
    debug_assert!(probes.len() <= SPATIAL_FIT_MAX_PROBES_PER_NODE_PAIR);
    probes
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_probe_incidence(probes: &[SpatialBoundaryProbe]) -> Vec<(usize, usize, usize)> {
    let mut incidence = Vec::with_capacity(probes.len().saturating_mul(2));
    debug_assert_eq!(incidence.capacity(), probes.len().saturating_mul(2));
    for (probe_index, probe) in probes.iter().copied().enumerate() {
        incidence.push((probe.left_node, probe.left_representative, probe_index));
        incidence.push((probe.right_node, probe.right_representative, probe_index));
    }
    incidence.sort_unstable();
    incidence
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_affected_probe_indices(
    incidence: &[(usize, usize, usize)],
    edited_representatives: impl IntoIterator<Item = (usize, usize)>,
) -> Vec<usize> {
    let mut affected = Vec::with_capacity(incidence.len());
    debug_assert_eq!(affected.capacity(), incidence.len());
    for key in edited_representatives {
        let start = incidence.partition_point(|entry| (entry.0, entry.1) < key);
        let end = incidence.partition_point(|entry| (entry.0, entry.1) <= key);
        affected.extend(incidence[start..end].iter().map(|entry| entry.2));
    }
    affected.sort_unstable();
    affected.dedup();
    affected
}

/// Prove that a candidate cannot change any fixed-grid contributor key.
///
/// `spatial_nearest_boundary_contributor` orders representatives first by
/// support overlap with the other node, then by immutable center distances and
/// finally by index. Spatial widening keeps centers fixed. Checking the sole
/// mutable comparison bit against every authored node is therefore equivalent
/// to recomputing and comparing all sorted/deduplicated 3x3 probe keys, without
/// rescanning every representative for every candidate.
#[cfg(any(feature = "lod_build", test))]
fn spatial_candidate_preserves_probe_topology(
    nodes: &[SpatialMomentMergeNode],
    overrides: &[SpatialRepresentativeOverride],
) -> bool {
    overrides.iter().all(|candidate| {
        let original = &nodes[candidate.node].representatives[candidate.representative];
        original
            .gaussian
            .position_visibility
            .position
            .iter()
            .zip(candidate.value.gaussian.position_visibility.position)
            .all(|(original, candidate)| original.to_bits() == candidate.to_bits())
            && nodes.iter().enumerate().all(|(other_node, other)| {
                other_node == candidate.node
                    || lod_bounds_touch_or_overlap(
                        original.support_bounds,
                        other.authored_support_bounds,
                    ) == lod_bounds_touch_or_overlap(
                        candidate.value.support_bounds,
                        other.authored_support_bounds,
                    )
            })
    })
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_nearest_boundary_contributor(
    node: &SpatialMomentMergeNode,
    other_bounds: LodBounds,
    target: [f32; 3],
) -> usize {
    node.representatives
        .iter()
        .enumerate()
        .min_by(|(left_index, left), (right_index, right)| {
            let left_overlap = lod_bounds_touch_or_overlap(left.support_bounds, other_bounds);
            let right_overlap = lod_bounds_touch_or_overlap(right.support_bounds, other_bounds);
            (!left_overlap)
                .cmp(&(!right_overlap))
                .then_with(|| {
                    point_to_point_squared_distance(
                        left.gaussian.position_visibility.position,
                        target,
                    )
                    .total_cmp(&point_to_point_squared_distance(
                        right.gaussian.position_visibility.position,
                        target,
                    ))
                })
                .then_with(|| {
                    point_to_bounds_squared_distance(
                        left.gaussian.position_visibility.position,
                        other_bounds,
                    )
                    .total_cmp(&point_to_bounds_squared_distance(
                        right.gaussian.position_visibility.position,
                        other_bounds,
                    ))
                })
                .then_with(|| left_index.cmp(right_index))
        })
        .map(|(index, _)| index)
        .expect("validated spatial nodes contain representatives")
}

#[cfg(any(feature = "lod_build", test))]
fn point_to_bounds_squared_distance(point: [f32; 3], bounds: LodBounds) -> f64 {
    (0..3)
        .map(|axis| {
            let delta = if point[axis] < bounds.min[axis] {
                f64::from(bounds.min[axis] - point[axis])
            } else if point[axis] > bounds.max[axis] {
                f64::from(point[axis] - bounds.max[axis])
            } else {
                0.0
            };
            delta * delta
        })
        .sum()
}

#[cfg(any(feature = "lod_build", test))]
fn point_to_point_squared_distance(left: [f32; 3], right: [f32; 3]) -> f64 {
    (0..3)
        .map(|axis| {
            let delta = f64::from(left[axis]) - f64::from(right[axis]);
            delta * delta
        })
        .sum()
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_probe_source(
    nodes: &[SpatialMomentMergeNode],
    node_index: usize,
    representative_index: usize,
) -> Option<&[Gaussian3d]> {
    let node = nodes.get(node_index)?;
    let range = node.source_ranges.get(representative_index)?.clone();
    node.source_records.as_ref()?.get(range)
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_boundary_reference(
    nodes: &[SpatialMomentMergeNode],
    probe: SpatialBoundaryProbe,
) -> Result<SpatialBoundaryReference, LodBuildError> {
    let left_source = spatial_probe_source(nodes, probe.left_node, probe.left_representative)
        .ok_or(LodBuildError::EmptyReduction)?;
    let right_source = spatial_probe_source(nodes, probe.right_node, probe.right_representative)
        .ok_or(LodBuildError::EmptyReduction)?;
    let left_representative = &nodes[probe.left_node].representatives[probe.left_representative];
    let right_representative = &nodes[probe.right_node].representatives[probe.right_representative];

    let mut characteristic_pixels_per_world = [0.0_f64; PROJECTED_ALPHA_MASS_DIRECTIONS.len()];
    let mut sample_points = [[[0.0_f64; 2]; SPATIAL_FIT_SAMPLE_POINTS_PER_DIRECTION];
        PROJECTED_ALPHA_MASS_DIRECTIONS.len()];
    let mut source_alpha = [[[0.0_f64; SPATIAL_FIT_SAMPLE_POINTS_PER_DIRECTION];
        SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()];
        PROJECTED_ALPHA_MASS_DIRECTIONS.len()];

    for (direction_index, direction) in PROJECTED_ALPHA_MASS_DIRECTIONS.iter().copied().enumerate()
    {
        let (horizontal, vertical) = spatial_projection_basis(direction);
        let characteristic = spatial_characteristic_pixels_per_world(
            left_source.iter().chain(right_source.iter()),
            horizontal,
            vertical,
        )?;
        characteristic_pixels_per_world[direction_index] = characteristic;
        let left_center = spatial_project_point(
            left_representative.gaussian.position_visibility.position,
            horizontal,
            vertical,
        );
        let right_center = spatial_project_point(
            right_representative.gaussian.position_visibility.position,
            horizontal,
            vertical,
        );
        let points = [
            left_center,
            [
                0.5 * (left_center[0] + right_center[0]),
                0.5 * (left_center[1] + right_center[1]),
            ],
            right_center,
        ];
        sample_points[direction_index] = points;
        for (scale_index, scale_factor) in SPATIAL_FIT_PROJECTED_SCALE_FACTORS
            .iter()
            .copied()
            .enumerate()
        {
            let pixels_per_world = characteristic * scale_factor;
            for (point_index, point) in points.into_iter().enumerate() {
                source_alpha[direction_index][scale_index][point_index] =
                    spatial_renderer_alpha_at(
                        left_source.iter().chain(right_source.iter()),
                        point,
                        horizontal,
                        vertical,
                        pixels_per_world,
                        false,
                    )?;
            }
        }
    }

    Ok(SpatialBoundaryReference {
        characteristic_pixels_per_world,
        sample_points,
        source_alpha,
    })
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_boundary_metrics_from_reference(
    nodes: &[SpatialMomentMergeNode],
    probe: SpatialBoundaryProbe,
    reference: &SpatialBoundaryReference,
    overrides: &[SpatialRepresentativeOverride],
) -> Result<SpatialBoundaryMetrics, LodBuildError> {
    let left_representative =
        spatial_probe_representative(nodes, probe.left_node, probe.left_representative, overrides);
    let right_representative = spatial_probe_representative(
        nodes,
        probe.right_node,
        probe.right_representative,
        overrides,
    );

    let mut boundary_reference_by_scale = [0.0_f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()];
    let mut boundary_error_by_scale = [0.0_f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()];
    let mut composited_error_by_scale = [0.0_f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()];
    for (direction_index, direction) in PROJECTED_ALPHA_MASS_DIRECTIONS.iter().copied().enumerate()
    {
        let (horizontal, vertical) = spatial_projection_basis(direction);
        let characteristic = reference.characteristic_pixels_per_world[direction_index];
        for (scale_index, scale_factor) in SPATIAL_FIT_PROJECTED_SCALE_FACTORS
            .iter()
            .copied()
            .enumerate()
        {
            let pixels_per_world = characteristic * scale_factor;
            for (point_index, point) in reference.sample_points[direction_index]
                .into_iter()
                .enumerate()
            {
                let source_alpha =
                    reference.source_alpha[direction_index][scale_index][point_index];
                let emitted = spatial_renderer_alpha_at(
                    [
                        &left_representative.gaussian,
                        &right_representative.gaussian,
                    ],
                    point,
                    horizontal,
                    vertical,
                    pixels_per_world,
                    true,
                )?;
                let error = (emitted - source_alpha).abs();
                composited_error_by_scale[scale_index] += error;
                if point_index == 1 {
                    boundary_reference_by_scale[scale_index] += source_alpha;
                    boundary_error_by_scale[scale_index] += error;
                }
            }
        }
    }
    Ok(spatial_boundary_metrics_from_accumulators(
        boundary_reference_by_scale,
        boundary_error_by_scale,
        composited_error_by_scale,
    ))
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_boundary_metrics_with_mode(
    nodes: &[SpatialMomentMergeNode],
    probe: SpatialBoundaryProbe,
    reference: &SpatialBoundaryReference,
    overrides: &[SpatialRepresentativeOverride],
    mode: SpatialBoundaryMetricMode,
) -> Result<SpatialBoundaryMetrics, LodBuildError> {
    match mode {
        SpatialBoundaryMetricMode::Cached => {
            spatial_boundary_metrics_from_reference(nodes, probe, reference, overrides)
        }
        #[cfg(test)]
        SpatialBoundaryMetricMode::BruteForce => {
            spatial_boundary_metrics_brute_force(nodes, probe, overrides)
        }
    }
}

#[cfg(test)]
fn spatial_boundary_metrics(
    nodes: &[SpatialMomentMergeNode],
    probe: SpatialBoundaryProbe,
    overrides: &[SpatialRepresentativeOverride],
) -> Result<SpatialBoundaryMetrics, LodBuildError> {
    let reference = spatial_boundary_reference(nodes, probe)?;
    spatial_boundary_metrics_from_reference(nodes, probe, &reference, overrides)
}

#[cfg(test)]
fn spatial_boundary_metrics_brute_force(
    nodes: &[SpatialMomentMergeNode],
    probe: SpatialBoundaryProbe,
    overrides: &[SpatialRepresentativeOverride],
) -> Result<SpatialBoundaryMetrics, LodBuildError> {
    let Some(left_source) = spatial_probe_source(nodes, probe.left_node, probe.left_representative)
    else {
        return Ok(SpatialBoundaryMetrics {
            relative_boundary_error: f64::INFINITY,
            composited_error: 0.0,
            relative_boundary_error_by_scale: [f64::INFINITY;
                SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()],
            composited_error_by_scale: [0.0; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()],
        });
    };
    let Some(right_source) =
        spatial_probe_source(nodes, probe.right_node, probe.right_representative)
    else {
        return Ok(SpatialBoundaryMetrics {
            relative_boundary_error: f64::INFINITY,
            composited_error: 0.0,
            relative_boundary_error_by_scale: [f64::INFINITY;
                SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()],
            composited_error_by_scale: [0.0; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()],
        });
    };
    let left_representative =
        spatial_probe_representative(nodes, probe.left_node, probe.left_representative, overrides);
    let right_representative = spatial_probe_representative(
        nodes,
        probe.right_node,
        probe.right_representative,
        overrides,
    );

    let mut boundary_reference_by_scale = [0.0_f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()];
    let mut boundary_error_by_scale = [0.0_f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()];
    let mut composited_error_by_scale = [0.0_f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()];
    for direction in PROJECTED_ALPHA_MASS_DIRECTIONS {
        let (horizontal, vertical) = spatial_projection_basis(direction);
        let characteristic_pixels_per_world = spatial_characteristic_pixels_per_world(
            left_source.iter().chain(right_source.iter()),
            horizontal,
            vertical,
        )?;
        let left_center = spatial_project_point(
            left_representative.gaussian.position_visibility.position,
            horizontal,
            vertical,
        );
        let right_center = spatial_project_point(
            right_representative.gaussian.position_visibility.position,
            horizontal,
            vertical,
        );
        let midpoint = [
            0.5 * (left_center[0] + right_center[0]),
            0.5 * (left_center[1] + right_center[1]),
        ];
        for (scale_index, scale_factor) in SPATIAL_FIT_PROJECTED_SCALE_FACTORS
            .iter()
            .copied()
            .enumerate()
        {
            let pixels_per_world = characteristic_pixels_per_world * scale_factor;
            for (probe_index, point) in [left_center, midpoint, right_center]
                .into_iter()
                .enumerate()
            {
                let reference = spatial_renderer_alpha_at(
                    left_source.iter().chain(right_source.iter()),
                    point,
                    horizontal,
                    vertical,
                    pixels_per_world,
                    false,
                )?;
                let emitted = spatial_renderer_alpha_at(
                    [
                        &left_representative.gaussian,
                        &right_representative.gaussian,
                    ],
                    point,
                    horizontal,
                    vertical,
                    pixels_per_world,
                    true,
                )?;
                let error = (emitted - reference).abs();
                composited_error_by_scale[scale_index] += error;
                if probe_index == 1 {
                    boundary_reference_by_scale[scale_index] += reference;
                    boundary_error_by_scale[scale_index] += error;
                }
            }
        }
    }
    Ok(spatial_boundary_metrics_from_accumulators(
        boundary_reference_by_scale,
        boundary_error_by_scale,
        composited_error_by_scale,
    ))
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_boundary_metrics_from_accumulators(
    boundary_reference_by_scale: [f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()],
    boundary_error_by_scale: [f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()],
    composited_error_by_scale: [f64; SPATIAL_FIT_PROJECTED_SCALE_FACTORS.len()],
) -> SpatialBoundaryMetrics {
    let relative_boundary_error_by_scale = std::array::from_fn(|scale_index| {
        let reference = boundary_reference_by_scale[scale_index];
        if reference <= SPATIAL_FIT_MIN_REFERENCE_ALPHA {
            0.0
        } else {
            boundary_error_by_scale[scale_index] / reference
        }
    });
    SpatialBoundaryMetrics {
        relative_boundary_error: relative_boundary_error_by_scale
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
        composited_error: composited_error_by_scale.iter().sum(),
        relative_boundary_error_by_scale,
        composited_error_by_scale,
    }
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_probe_representative<'a>(
    nodes: &'a [SpatialMomentMergeNode],
    node: usize,
    representative: usize,
    overrides: &'a [SpatialRepresentativeOverride],
) -> &'a MomentMergeResult {
    overrides
        .iter()
        .find(|candidate| candidate.node == node && candidate.representative == representative)
        .map(|candidate| &candidate.value)
        .unwrap_or(&nodes[node].representatives[representative])
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_projection_basis(direction: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let helper = if direction[2].abs() < 0.875 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let mut horizontal = cross_f64(helper, direction);
    let horizontal_length = dot_f64(horizontal, horizontal).sqrt();
    horizontal = horizontal.map(|value| value / horizontal_length);
    let vertical = cross_f64(direction, horizontal);
    (horizontal, vertical)
}

#[cfg(any(feature = "lod_build", test))]
fn cross_f64(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_project_point(point: [f32; 3], horizontal: [f64; 3], vertical: [f64; 3]) -> [f64; 2] {
    let point = point.map(f64::from);
    [dot_f64(point, horizontal), dot_f64(point, vertical)]
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_characteristic_pixels_per_world<'a>(
    samples: impl IntoIterator<Item = &'a Gaussian3d>,
    horizontal: [f64; 3],
    vertical: [f64; 3],
) -> Result<f64, LodBuildError> {
    let mut variance_sum = 0.0_f64;
    let mut count = 0_u64;
    for sample in samples {
        let covariance = gaussian_covariance(sample)?;
        let xx = quadratic_form_f64(covariance, horizontal).max(0.0);
        let yy = quadratic_form_f64(covariance, vertical).max(0.0);
        variance_sum += 0.5 * (xx + yy);
        count = count.checked_add(1).ok_or(LodBuildError::CountOverflow(
            "spatial characteristic samples",
        ))?;
    }
    if count == 0 || !variance_sum.is_finite() || variance_sum < 0.0 {
        return Err(LodBuildError::DerivedNonFinite(
            "spatial characteristic projected variance",
        ));
    }
    let characteristic_sigma = (variance_sum / count as f64).sqrt().max(1e-12);
    let pixels_per_world = characteristic_sigma.recip().clamp(1e-6, 1e6);
    if pixels_per_world.is_finite() {
        Ok(pixels_per_world)
    } else {
        Err(LodBuildError::DerivedNonFinite(
            "spatial characteristic pixels per world",
        ))
    }
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_renderer_alpha_at<'a>(
    samples: impl IntoIterator<Item = &'a Gaussian3d>,
    point: [f64; 2],
    horizontal: [f64; 3],
    vertical: [f64; 3],
    pixels_per_world: f64,
    lod_candidate: bool,
) -> Result<f64, LodBuildError> {
    if !pixels_per_world.is_finite() || pixels_per_world <= 0.0 {
        return Err(LodBuildError::DerivedNonFinite("spatial pixels per world"));
    }
    let pixel_variance_scale = pixels_per_world * pixels_per_world;
    let point = point.map(|coordinate| coordinate * pixels_per_world);
    let mut remaining = 1.0_f64;
    for sample in samples {
        let authored_opacity = sample.scale_opacity.opacity.clamp(0.0, 1.0);
        let visible_opacity =
            authored_opacity * sample.position_visibility.visibility.clamp(0.0, 1.0);
        if visible_opacity <= 0.0 {
            continue;
        }
        let covariance = gaussian_covariance(sample)?;
        let unfiltered_xx = quadratic_form_f64(covariance, horizontal);
        let unfiltered_xy = dot_f64(horizontal, matrix_vector_product_3x3(covariance, vertical));
        let unfiltered_yy = quadratic_form_f64(covariance, vertical);
        let filtered = crate::render::gaussian_mip_filter_covariance_2d([
            checked_f32(
                unfiltered_xx * pixel_variance_scale,
                "spatial projected covariance",
            )?,
            checked_f32(
                unfiltered_xy * pixel_variance_scale,
                "spatial projected covariance",
            )?,
            checked_f32(
                unfiltered_yy * pixel_variance_scale,
                "spatial projected covariance",
            )?,
        ]);
        let [xx, xy, yy] = filtered.covariance.map(f64::from);
        let opacity = f64::from(visible_opacity * filtered.opacity_scale);
        if opacity <= 0.0 {
            continue;
        }
        let mid = 0.5 * (xx + yy);
        let radius = (0.25 * (xx - yy) * (xx - yy) + xy * xy).sqrt();
        let major_variance = mid + radius;
        let minor_variance = (mid - radius).max(f64::MIN_POSITIVE);
        if !major_variance.is_finite() || major_variance <= 0.0 || !minor_variance.is_finite() {
            continue;
        }
        let major_axis = if xy.abs() + (major_variance - xx).abs() > 1e-15 {
            let length = (xy * xy + (major_variance - xx).powi(2)).sqrt();
            [-xy / length, (major_variance - xx) / length]
        } else {
            [1.0, 0.0]
        };
        let minor_axis = [major_axis[1], -major_axis[0]];
        let center =
            spatial_project_point(sample.position_visibility.position, horizontal, vertical)
                .map(|coordinate| coordinate * pixels_per_world);
        let delta = [point[0] - center[0], point[1] - center[1]];
        let major = delta[0] * major_axis[0] + delta[1] * major_axis[1];
        let minor = delta[0] * minor_axis[0] + delta[1] * minor_axis[1];
        let cutoff = f64::from(crate::render::gaussian_support_cutoff(
            authored_opacity,
            true,
            lod_candidate,
        ));
        if major.abs() > cutoff * major_variance.sqrt()
            || minor.abs() > cutoff * minor_variance.sqrt()
        {
            continue;
        }
        let power = -0.5 * (major * major / major_variance + minor * minor / minor_variance);
        let alpha = (power.exp() * opacity).min(0.999);
        remaining *= 1.0 - alpha;
    }
    Ok(1.0 - remaining)
}

#[cfg(any(feature = "lod_build", test))]
fn matrix_vector_product_3x3(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| dot_f64(row, vector))
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_widened_representative(
    source: &[Gaussian3d],
    original: &MomentMergeResult,
    tangent_factor: f32,
    support_sigma: f32,
) -> Result<Option<MomentMergeResult>, LodBuildError> {
    if !tangent_factor.is_finite() || tangent_factor <= 1.0 || source.is_empty() {
        return Ok(None);
    }
    let mut accumulator = MomentAccumulator::new();
    for gaussian in source {
        accumulator.add(gaussian, support_sigma)?;
    }
    let mut gaussian = original.gaussian;
    let normal_axis = gaussian
        .scale_opacity
        .scale
        .iter()
        .enumerate()
        .min_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| left_index.cmp(right_index))
        })
        .map(|(index, _)| index)
        .unwrap();
    for axis in 0..3 {
        if axis != normal_axis {
            gaussian.scale_opacity.scale[axis] *= tangent_factor;
        }
    }
    if !gaussian
        .scale_opacity
        .scale
        .iter()
        .all(|scale| scale.is_finite() && *scale >= 0.0)
    {
        return Ok(None);
    }
    gaussian.scale_opacity.opacity = checked_f32(
        1.0 - (-accumulator.optical_depth).exp(),
        "spatial fitted opacity",
    )?
    .clamp(0.0, 1.0);
    let raw_union_gaussian = gaussian;
    let covariance = gaussian_covariance(&gaussian)?;
    let projected_area = symmetric_adjugate(covariance);
    let raw_projected_alpha_mass_inflation_upper_bound = calibrate_projected_alpha_mass(
        &mut gaussian,
        projected_area,
        accumulator.projected_alpha_mass_sqrt_sum,
    )?;
    let raster_risk = moment_merge_raster_risk(
        &raw_union_gaussian,
        covariance,
        projected_area,
        support_sigma,
        raw_projected_alpha_mass_inflation_upper_bound,
        accumulator.sampled_projected_alpha_mass,
        accumulator.sampled_support_min,
        accumulator.sampled_support_max,
        accumulator.max_source_major_scale,
        accumulator.max_source_anisotropy,
    )?;
    let source_bounds = accumulator
        .bounds
        .ok_or(LodBuildError::DerivedNonFinite("spatial source bounds"))?;
    let geometric = farthest_corner_distance(source_bounds, gaussian.position_visibility.position)?;
    let opacity = gaussian.scale_opacity.opacity;
    let opacity_error = (opacity - accumulator.min_opacity)
        .abs()
        .max((opacity - accumulator.max_opacity).abs());
    let appearance = original.error.appearance;
    let combined = geometric.max(appearance).max(opacity_error);
    Ok(Some(MomentMergeResult {
        gaussian,
        support_bounds: gaussian_support_bounds(&gaussian, support_sigma)?,
        error: LodError {
            geometric,
            appearance,
            opacity: opacity_error,
            combined,
        },
        source_count: accumulator.count,
        total_weight: accumulator.weight,
        raster_risk,
    }))
}

/// Exact oriented support AABB used only by ABI 16's spatial fitter. The
/// portable manifest continues to retain its older conservative sphere bounds;
/// this narrower envelope prevents the fitter from widening past authored
/// source support while preserving format compatibility.
#[cfg(any(feature = "lod_build", test))]
pub(crate) fn gaussian_oriented_support_bounds(
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
    let covariance = gaussian_covariance(gaussian)?;
    let position = gaussian.position_visibility.position;
    let mut min = [0.0_f32; 3];
    let mut max = [0.0_f32; 3];
    for axis in 0..3 {
        let radius = checked_f32(
            f64::from(support_sigma) * covariance[axis][axis].max(0.0).sqrt(),
            "oriented Gaussian support",
        )?;
        let radius = next_up(radius);
        min[axis] = next_down(position[axis] - radius);
        max[axis] = next_up(position[axis] + radius);
    }
    LodBounds::new(min, max).map_err(LodBuildError::InvalidBounds)
}

#[cfg(any(feature = "lod_build", test))]
fn oriented_support_inside(
    gaussian: &Gaussian3d,
    support_sigma: f32,
    envelope: LodBounds,
) -> Result<bool, LodBuildError> {
    let support = gaussian_oriented_support_bounds(gaussian, support_sigma)?;
    let epsilon = bounds_epsilon(&envelope, &support);
    Ok(envelope.contains_with_epsilon(&support, epsilon))
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_boundary_metrics_no_worse(
    candidate: SpatialBoundaryMetrics,
    current: SpatialBoundaryMetrics,
) -> bool {
    candidate
        .relative_boundary_error_by_scale
        .into_iter()
        .zip(current.relative_boundary_error_by_scale)
        .all(|(candidate, current)| float_no_worse(candidate, current))
        && candidate
            .composited_error_by_scale
            .into_iter()
            .zip(current.composited_error_by_scale)
            .all(|(candidate, current)| float_no_worse(candidate, current))
}

#[cfg(any(feature = "lod_build", test))]
fn spatial_boundary_metrics_strictly_better(
    candidate: SpatialBoundaryMetrics,
    current: SpatialBoundaryMetrics,
) -> bool {
    spatial_boundary_metrics_no_worse(candidate, current)
        && float_strictly_better(
            candidate.relative_boundary_error,
            current.relative_boundary_error,
        )
}

#[cfg(any(feature = "lod_build", test))]
fn float_no_worse(candidate: f64, current: f64) -> bool {
    if candidate == current {
        return true;
    }
    if !candidate.is_finite() || !current.is_finite() {
        return candidate < current;
    }
    let tolerance = 64.0 * f64::EPSILON * candidate.abs().max(current.abs()).max(1.0);
    candidate <= current + tolerance
}

#[cfg(any(feature = "lod_build", test))]
fn float_strictly_better(candidate: f64, current: f64) -> bool {
    if !candidate.is_finite() {
        return false;
    }
    if !current.is_finite() {
        return true;
    }
    let tolerance = 64.0 * f64::EPSILON * candidate.abs().max(current.abs()).max(1.0);
    candidate < current - tolerance
}

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

        build_planar_3d_lod_owned(cloud.iter().collect(), self.settings).map(|(lod, _)| lod)
    }

    fn build_owned_cancelable(
        &self,
        mut source: Vec<Gaussian3d>,
        cancellation: LodBuildCancellation<'_>,
    ) -> CancelableLodBuildResult<(PlanarGaussian3dLod, Vec<Gaussian3d>)> {
        cancellation.check()?;
        self.settings
            .validate()
            .map_err(LodBuildError::InvalidSettings)?;

        for (index, gaussian) in source.iter_mut().enumerate() {
            cancellation.poll(index)?;
            validate_gaussian(gaussian)
                .map_err(|field| LodBuildError::InvalidGaussian { index, field })?;
            *gaussian = canonicalize_gaussian_zeros(*gaussian);
        }
        cancellation.check()?;

        if source.is_empty() {
            let output = empty_lod(self.settings);
            return Ok((output, source));
        }

        let center_bounds = source_center_bounds(&source, cancellation)?;
        let mut keyed = Vec::with_capacity(source.len());
        for (source_index, gaussian) in source.iter().enumerate() {
            cancellation.poll(source_index)?;
            keyed.push(MortonSourceIndex {
                morton: canonical_lod_morton_code(
                    gaussian.position_visibility.position,
                    center_bounds,
                ),
                source_index,
            });
        }
        cancellation.check()?;
        sort_morton_source_indices(&mut keyed, &source, cancellation)?;
        cancellation.check()?;

        let source_fingerprint = source_fingerprint(&keyed, &source, cancellation)?;
        let mut canonical_morton = Vec::with_capacity(keyed.len());
        let mut canonical_source = Vec::with_capacity(keyed.len());
        for (index, entry) in keyed.iter().enumerate() {
            cancellation.poll(index)?;
            canonical_morton.push(entry.morton);
            canonical_source.push(source[entry.source_index]);
        }
        cancellation.check()?;
        drop(keyed);
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
        // Each leaf owns a disjoint canonical source range. Rayon preserves the
        // indexed iterator order during collection, and errors are flattened
        // afterwards in Morton order, so worker scheduling cannot affect either
        // node identity or which invalid derived value is reported first.
        #[cfg(feature = "sort_rayon")]
        let leaf_results: Vec<_> = leaf_ranges
            .par_iter()
            .map(|range| {
                cancellation.check()?;
                let node = build_leaf_temp_node(
                    range.clone(),
                    &canonical_source,
                    &canonical_morton,
                    self.settings.support_sigma,
                )?;
                cancellation.check()?;
                Ok::<_, CancelableLodBuildError>(node)
            })
            .collect();
        #[cfg(not(feature = "sort_rayon"))]
        let leaf_results: Vec<_> = leaf_ranges
            .iter()
            .map(|range| {
                cancellation.check()?;
                let node = build_leaf_temp_node(
                    range.clone(),
                    &canonical_source,
                    &canonical_morton,
                    self.settings.support_sigma,
                )?;
                cancellation.check()?;
                Ok::<_, CancelableLodBuildError>(node)
            })
            .collect();
        let mut temporary: Vec<_> = leaf_results.into_iter().collect::<Result<_, _>>()?;
        let mut current_level = (0..temporary.len()).collect::<Vec<_>>();
        cancellation.check()?;

        let mut deepest_choices = if self.settings.leaf_capacity >= 2 {
            plan_high_fidelity_deepest_choices(
                &canonical_source,
                &temporary,
                &current_level,
                self.settings.support_sigma,
                usize::from(self.settings.branching_factor),
                cancellation,
            )?
        } else {
            HashMap::new()
        };
        cancellation.check()?;

        while current_level.len() > 1 {
            cancellation.check()?;
            // Pairing produces the deepest hierarchy available for the fixed
            // leaf-page capacity. An odd final node is carried to the next
            // level instead of creating an invalid unary parent. This keeps
            // default page granularity efficient while providing enough
            // refinement stages for bounded progressive representations.
            let paired_len = current_level.len() / 2 * 2;
            let child_groups = current_level[..paired_len]
                .chunks_exact(2)
                .map(|pair| [pair[0], pair[1]])
                .collect::<Vec<_>>();
            let carried_node = current_level.get(paired_len).copied();
            // Move each precomputed deepest choice into its one owning pair
            // before borrowing the temporary hierarchy across worker threads.
            // Parent nodes are then appended in canonical pair order.
            let parent_work = child_groups
                .into_iter()
                .map(|children| {
                    let deepest_choice = deepest_choices.remove(&children[0]);
                    (children, deepest_choice)
                })
                .collect::<Vec<_>>();
            #[cfg(feature = "sort_rayon")]
            let parent_results: Vec<_> = parent_work
                .into_par_iter()
                .map(|(children, deepest_choice)| {
                    build_parent_temp_node(
                        children,
                        deepest_choice,
                        &temporary,
                        &canonical_source,
                        self.settings.support_sigma,
                        usize::from(self.settings.branching_factor),
                        cancellation,
                    )
                })
                .collect();
            #[cfg(not(feature = "sort_rayon"))]
            let parent_results: Vec<_> = parent_work
                .into_iter()
                .map(|(children, deepest_choice)| {
                    build_parent_temp_node(
                        children,
                        deepest_choice,
                        &temporary,
                        &canonical_source,
                        self.settings.support_sigma,
                        usize::from(self.settings.branching_factor),
                        cancellation,
                    )
                })
                .collect();
            let parent_nodes = parent_results.into_iter().collect::<Result<Vec<_>, _>>()?;
            let first_parent = temporary.len();
            let mut next_level =
                Vec::with_capacity(parent_nodes.len() + carried_node.is_some() as usize);
            next_level.extend(first_parent..first_parent + parent_nodes.len());
            temporary.extend(parent_nodes);
            if let Some(carried_node) = carried_node {
                next_level.push(carried_node);
            }
            current_level = next_level;
            cancellation.check()?;
        }

        let root = current_level[0];
        let (order, parents, depths) = breadth_first_order(&temporary, root, cancellation)?;
        let mut max_depth = 0_u16;
        for (index, depth) in depths.iter().copied().enumerate() {
            cancellation.poll(index)?;
            max_depth = max_depth.max(depth);
        }
        let node_count =
            u32::try_from(order.len()).map_err(|_| LodBuildError::CountOverflow("nodes"))?;

        let mut old_to_new = vec![usize::MAX; temporary.len()];
        for (new_index, old_index) in order.iter().copied().enumerate() {
            cancellation.poll(new_index)?;
            old_to_new[old_index] = new_index;
        }

        let mut nodes = Vec::with_capacity(order.len());
        let mut node_page_payloads = Vec::with_capacity(order.len());

        for (new_index, old_index) in order.iter().copied().enumerate() {
            cancellation.poll(new_index)?;
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
                        return Err(LodBuildError::NonContiguousChildren.into());
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
            cancellation,
        )?;
        for (index, (node, representation)) in nodes
            .iter_mut()
            .zip(packed.node_ranges.iter().copied())
            .enumerate()
        {
            cancellation.poll(index)?;
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
                morph_map: None,
            },
            pages: packed.pages,
        };
        cancellation.check()?;
        Ok((output, source))
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
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<PackedNodePages> {
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
        cancellation.poll(node_index)?;
        if payload.gaussians.is_empty() {
            return Err(LodBuildError::EmptyReduction.into());
        }
        if payload.gaussians.len() > physical_capacity {
            return Err(LodBuildError::CountOverflow("physical page capacity").into());
        }
        let can_append = pending.as_ref().is_some_and(|page| {
            page.depth == payload.depth
                && page.kind == payload.kind
                && page.gaussians.len() + payload.gaussians.len() <= physical_capacity
        });
        if !can_append {
            if let Some(page) = pending.take() {
                finish_node_page(page, support_sigma, &mut packed, cancellation)?;
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
        finish_node_page(page, support_sigma, &mut packed, cancellation)?;
    }
    debug_assert!(
        packed
            .node_ranges
            .iter()
            .all(|range| range.page.is_valid() && range.count > 0)
    );
    cancellation.check()?;
    Ok(packed)
}

fn finish_node_page(
    pending: PendingNodePage,
    support_sigma: f32,
    packed: &mut PackedNodePages,
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<()> {
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
    for (index, gaussian) in page.gaussians.iter().enumerate() {
        cancellation.poll(index)?;
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

/// Builds from an owned interleaved source while retaining its original order
/// for callers that also need an exact flat fallback. Signed zeroes are
/// canonicalized in both the returned source and the deterministic hierarchy.
pub(crate) fn build_planar_3d_lod_owned(
    source: Vec<Gaussian3d>,
    settings: GaussianLodBuildSettings,
) -> Result<(PlanarGaussian3dLod, Vec<Gaussian3d>), LodBuildError> {
    let (output, source) = build_planar_3d_lod_owned_cancelable(source, settings, &|| false)?
        .expect("a false predicate cannot cancel an owned LoD build");
    output.validate().map_err(LodBuildError::Validation)?;
    Ok((output, source))
}

/// Cooperatively builds an owned transient hierarchy. `Ok(None)` means the
/// caller canceled the job; ordinary construction failures retain the stable
/// public [`LodBuildError`] surface used by the non-cancelable entry point. The
/// transient consumer validates the manifest and encoded pages before runtime
/// activation; the non-cancelable wrapper additionally performs the complete
/// reference payload validation here.
pub(crate) fn build_planar_3d_lod_owned_cancelable<F>(
    source: Vec<Gaussian3d>,
    settings: GaussianLodBuildSettings,
    is_canceled: &F,
) -> Result<Option<(PlanarGaussian3dLod, Vec<Gaussian3d>)>, LodBuildError>
where
    F: Fn() -> bool + Sync,
{
    let cancellation = LodBuildCancellation { is_canceled };
    match CpuGaussianLodBuilder::new(settings).build_owned_cancelable(source, cancellation) {
        Ok(output) => Ok(Some(output)),
        Err(CancelableLodBuildError::Canceled) => Ok(None),
        Err(CancelableLodBuildError::Build(error)) => Err(error),
    }
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
    #[cfg(test)]
    GAUSSIAN_SUPPORT_FULL_VALIDATIONS
        .set(GAUSSIAN_SUPPORT_FULL_VALIDATIONS.get().saturating_add(1));
    validate_gaussian(gaussian)
        .map_err(|field| LodBuildError::InvalidGaussian { index: 0, field })?;
    if !support_sigma.is_finite() || support_sigma <= 0.0 {
        return Err(LodBuildError::InvalidSettings(
            LodBuildSettingsError::SupportSigma(support_sigma),
        ));
    }
    gaussian_support_bounds_trusted_decoded(gaussian, support_sigma)
}

/// Computes support for a runtime page already authenticated and semantically
/// validated by the decoder. This deliberately reads only position and scale;
/// rescanning opacity, rotation, and every SH coefficient on the main thread
/// would defeat bounded LoD debug preparation.
pub(crate) fn gaussian_support_bounds_trusted_decoded(
    gaussian: &Gaussian3d,
    support_sigma: f32,
) -> Result<LodBounds, LodBuildError> {
    debug_assert!(support_sigma.is_finite() && support_sigma > 0.0);
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

#[cfg(test)]
pub(crate) fn gaussian_support_full_validation_count_for_test() -> u64 {
    GAUSSIAN_SUPPORT_FULL_VALIDATIONS.get()
}

#[derive(Clone)]
pub(crate) struct MomentAccumulator {
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
    pub(crate) fn new() -> Self {
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

    pub(crate) fn add(
        &mut self,
        gaussian: &Gaussian3d,
        support_sigma: f32,
    ) -> Result<(), LodBuildError> {
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
        let covariance_frame = gaussian_covariance_frame(gaussian)?;
        let covariance = covariance_frame.covariance;
        let projected_area_sqrt = covariance_frame.projected_area_sqrt;
        let effective_alpha = f64::from(opacity * visibility);
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
            let projected = projected_area_sqrt.map(|row| dot_f64(row, direction));
            let projected_area = dot_f64(projected, projected).max(0.0).sqrt();
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

    pub(crate) fn finish(&self, support_sigma: f32) -> Result<MomentMergeResult, LodBuildError> {
        self.finish_with_projected_alpha_calibration(support_sigma, true)
    }

    #[cfg(test)]
    fn finish_external_v2(&self, support_sigma: f32) -> Result<MomentMergeResult, LodBuildError> {
        self.finish_with_projected_alpha_calibration(support_sigma, false)
    }

    fn finish_with_projected_alpha_calibration(
        &self,
        support_sigma: f32,
        calibrate_projected_alpha: bool,
    ) -> Result<MomentMergeResult, LodBuildError> {
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
        let union_opacity =
            checked_f32(1.0 - (-self.optical_depth).exp(), "merged opacity")?.clamp(0.0, 1.0);
        let position = [
            checked_f32(mean[0], "merged position")?,
            checked_f32(mean[1], "merged position")?,
            checked_f32(mean[2], "merged position")?,
        ];
        let mut gaussian = Gaussian3d {
            position_visibility: PositionVisibility {
                position,
                visibility: self.max_visibility,
            },
            spherical_harmonic: SphericalHarmonicCoefficients { coefficients },
            rotation: Rotation { rotation },
            scale_opacity: ScaleOpacity {
                scale,
                opacity: union_opacity,
            },
        };
        // Pairing and fidelity metadata must retain how unsafe the raw optical-
        // depth-union representative was. ABI 14 only changes the emitted
        // opacity; evaluating risk after calibration would erase the magnitude
        // of that correction and can silently promote a structurally poor pair.
        let raw_union_gaussian = gaussian;
        let representative_covariance = gaussian_covariance(&gaussian)?;
        let representative_projected_area = symmetric_adjugate(representative_covariance);
        let raw_projected_alpha_mass_inflation_upper_bound = if calibrate_projected_alpha {
            calibrate_projected_alpha_mass(
                &mut gaussian,
                representative_projected_area,
                self.projected_alpha_mass_sqrt_sum,
            )?
        } else {
            let representative_alpha = f64::from(
                gaussian.scale_opacity.opacity.clamp(0.0, 1.0)
                    * gaussian.position_visibility.visibility.clamp(0.0, 1.0),
            );
            projected_alpha_mass_inflation_upper_bound(
                representative_alpha,
                representative_projected_area,
                self.projected_alpha_mass_sqrt_sum,
            )?
        };
        let raster_risk = moment_merge_raster_risk(
            &raw_union_gaussian,
            representative_covariance,
            representative_projected_area,
            support_sigma,
            raw_projected_alpha_mass_inflation_upper_bound,
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
        let opacity = gaussian.scale_opacity.opacity;
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

fn build_leaf_temp_node(
    range: std::ops::Range<usize>,
    canonical_source: &[Gaussian3d],
    canonical_morton: &[u64],
    support_sigma: f32,
) -> Result<TempNode, LodBuildError> {
    let source = LodSourceRange {
        start: range.start as u64,
        count: (range.end - range.start) as u64,
    };
    let mut accumulator = MomentAccumulator::new();
    for gaussian in &canonical_source[range.clone()] {
        accumulator.add(gaussian, support_sigma)?;
    }
    let bounds = accumulator
        .bounds
        .ok_or(LodBuildError::DerivedNonFinite("leaf bounds"))?;
    Ok(TempNode {
        children: Vec::new(),
        source,
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
    })
}

fn build_parent_temp_node(
    children: [usize; 2],
    deepest_choice: Option<DeepestRepresentationChoice>,
    temporary: &[TempNode],
    canonical_source: &[Gaussian3d],
    support_sigma: f32,
    branching_factor: usize,
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<TempNode> {
    cancellation.check()?;
    let first = &temporary[children[0]];
    let last = &temporary[children[1]];
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
    for child in children {
        let child = &temporary[child];
        accumulator.combine(&child.accumulator)?;
        bounds = bounds.union(child.bounds);
        error = error.max(child.error);
        high_fidelity_certificate = high_fidelity_certificate.min(child.high_fidelity_certificate);
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
    let (representatives, policy_envelope) =
        if children_are_exact_leaves && let Some(choice) = deepest_choice {
            (
                choice.into_representatives(),
                ProgressiveSelectionEnvelope::IDENTITY,
            )
        } else {
            let rung = progressive_moment_merge_representatives(
                source_records,
                child_representatives.div_ceil(branching_factor).max(1),
                support_sigma,
                cancellation,
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
    Ok(TempNode {
        children: children.to_vec(),
        source,
        morton,
        bounds,
        accumulator,
        representation_count: representatives.len(),
        representatives,
        error,
        high_fidelity_certificate,
    })
}

#[derive(Clone, Copy)]
struct MortonSourceIndex {
    morton: u64,
    source_index: usize,
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
            morph_map: None,
        },
        pages: Vec::new(),
    }
}

/// Bounded crate-internal entry point for external builders that can buffer
/// one explicitly capped source domain. It reuses ABI 14's risk-aware adjacent
/// agglomeration and conservative balanced selection envelope without exposing
/// the transient builder's cancellation machinery.
#[cfg(feature = "lod_build")]
pub(crate) fn build_progressive_moment_merge_rung(
    source: &[Gaussian3d],
    representative_count: usize,
    support_sigma: f32,
) -> Result<ProgressiveMomentMergeRung, LodBuildError> {
    let never_cancel = || false;
    match progressive_moment_merge_representatives(
        source,
        representative_count,
        support_sigma,
        LodBuildCancellation {
            is_canceled: &never_cancel,
        },
    ) {
        Ok(rung) => Ok(rung),
        Err(CancelableLodBuildError::Build(error)) => Err(error),
        Err(CancelableLodBuildError::Canceled) => {
            unreachable!("the external rung wrapper never requests cancellation")
        }
    }
}

fn progressive_moment_merge_representatives(
    source: &[Gaussian3d],
    representative_count: usize,
    support_sigma: f32,
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<ProgressiveMomentMergeRung> {
    cancellation.check()?;
    if source.is_empty() || representative_count == 0 || representative_count > source.len() {
        return Err(LodBuildError::EmptyReduction.into());
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
        let (representatives, source_ranges) = risk_aware_progressive_moment_merge_representatives(
            source,
            representative_count,
            support_sigma,
            cancellation,
        )?;
        let balanced_oracle = balanced_progressive_moment_merge_representatives(
            source,
            representative_count,
            support_sigma,
            cancellation,
        )?;
        let policy_envelope = progressive_selection_envelope(&balanced_oracle, cancellation)?;
        return Ok(ProgressiveMomentMergeRung {
            representatives,
            source_ranges,
            policy_envelope,
        });
    }

    let source_ranges = balanced_ranges_for_group_count(source.len(), representative_count);
    Ok(ProgressiveMomentMergeRung {
        representatives: balanced_progressive_moment_merge_representatives(
            source,
            representative_count,
            support_sigma,
            cancellation,
        )?,
        source_ranges,
        policy_envelope: ProgressiveSelectionEnvelope::IDENTITY,
    })
}

pub(crate) struct ProgressiveMomentMergeRung {
    pub(crate) representatives: Vec<MomentMergeResult>,
    /// Exact, contiguous source intervals aligned with `representatives`.
    #[cfg_attr(not(feature = "lod_build"), allow(dead_code))]
    pub(crate) source_ranges: Vec<std::ops::Range<usize>>,
    /// Independent conservative selection-policy envelope. This is separate
    /// from the metadata of the emitted payload so clustering cannot grade its
    /// own optimization as a runtime fidelity improvement.
    pub(crate) policy_envelope: ProgressiveSelectionEnvelope,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ProgressiveSelectionEnvelope {
    pub(crate) support_bounds: Option<LodBounds>,
    pub(crate) error: LodError,
    pub(crate) high_fidelity_certificate_cap: f32,
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
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<ProgressiveSelectionEnvelope> {
    let mut envelope = ProgressiveSelectionEnvelope::IDENTITY;
    for (index, representative) in representatives.iter().enumerate() {
        cancellation.poll(index)?;
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
    }
    Ok(envelope)
}

fn balanced_progressive_moment_merge_representatives(
    source: &[Gaussian3d],
    representative_count: usize,
    support_sigma: f32,
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<Vec<MomentMergeResult>> {
    let ranges = balanced_ranges_for_group_count(source.len(), representative_count);
    let mut representatives = Vec::with_capacity(representative_count);
    for (range_index, range) in ranges.into_iter().enumerate() {
        cancellation.poll(range_index)?;
        let mut accumulator = MomentAccumulator::new();
        for (index, gaussian) in source[range].iter().enumerate() {
            cancellation.poll(index)?;
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

/// Conservative peak allocation for the crate-internal risk-aware rung,
/// including its externally owned source buffer, cluster state, candidate
/// heap, and an output vector pessimistically sized to the full source. Vec
/// headers and allocator bookkeeping are covered by one extra record of each
/// payload type. The heap allowance is four times the source count: live stale
/// candidates can approach `2N`, and another factor of two covers geometric
/// Vec capacity growth.
#[cfg(any(feature = "lod_build", test))]
pub(crate) fn progressive_risk_aware_host_bytes_upper_bound(source_count: usize) -> Option<u64> {
    let capacity = source_count.checked_add(1)?;
    let candidate_capacity = source_count.checked_mul(4)?.checked_add(1)?;
    let bytes = capacity
        .checked_mul(size_of::<Gaussian3d>())?
        .checked_add(capacity.checked_mul(size_of::<ProgressiveAgglomerationCluster>())?)?
        .checked_add(
            candidate_capacity.checked_mul(size_of::<ProgressiveAgglomerationCandidate>())?,
        )?
        .checked_add(capacity.checked_mul(size_of::<MomentMergeResult>())?)?;
    u64::try_from(bytes).ok()
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
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<(Vec<MomentMergeResult>, Vec<std::ops::Range<usize>>)> {
    let mut clusters = Vec::with_capacity(source.len());
    for (index, gaussian) in source.iter().enumerate() {
        cancellation.poll(index)?;
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
        cancellation.poll(left)?;
        push_progressive_agglomeration_candidate(&mut heap, &clusters, left, support_sigma)?;
    }

    let mut active_count = source.len();
    let mut merge_count = 0_usize;
    while active_count > representative_count {
        cancellation.poll(merge_count)?;
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
        merge_count += 1;
    }

    let mut representatives = Vec::with_capacity(representative_count);
    let mut source_ranges = Vec::with_capacity(representative_count);
    let mut cursor = Some(0_usize);
    let mut expected_source_start = 0_usize;
    for output_index in 0..representative_count {
        cancellation.poll(output_index)?;
        let index = cursor.ok_or(LodBuildError::CountOverflow(
            "risk-aware progressive representative partition",
        ))?;
        let cluster = &clusters[index];
        if !cluster.active || cluster.source_start != expected_source_start {
            return Err(LodBuildError::CountOverflow(
                "risk-aware progressive representative partition",
            )
            .into());
        }
        representatives.push(cluster.accumulator.finish(support_sigma)?);
        source_ranges.push(cluster.source_start..cluster.source_end);
        expected_source_start = cluster.source_end;
        cursor = cluster.next;
    }
    if cursor.is_some() || expected_source_start != source.len() {
        return Err(LodBuildError::CountOverflow(
            "risk-aware progressive representative partition",
        )
        .into());
    }
    Ok((representatives, source_ranges))
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

fn projected_progressive_storage(
    plans: &[DeepestPairingPlan],
    source_count: usize,
    carried_count: Option<usize>,
    branching_factor: usize,
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<Option<usize>> {
    let mut stored = source_count;
    let mut level = Vec::with_capacity(plans.len() + usize::from(carried_count.is_some()));
    for (index, plan) in plans.iter().enumerate() {
        cancellation.poll(index)?;
        let count = plan.representation_count();
        let Some(next_stored) = stored.checked_add(count) else {
            return Ok(None);
        };
        stored = next_stored;
        level.push(count);
    }
    if let Some(carried_count) = carried_count {
        level.push(carried_count);
    }
    while level.len() > 1 {
        cancellation.check()?;
        let paired = level.len() / 2 * 2;
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for (index, pair) in level[..paired].chunks_exact(2).enumerate() {
            cancellation.poll(index)?;
            let Some(count) = pair[0].checked_add(pair[1]) else {
                return Ok(None);
            };
            let count = count.div_ceil(branching_factor).max(1);
            let Some(next_stored) = stored.checked_add(count) else {
                return Ok(None);
            };
            stored = next_stored;
            next.push(count);
        }
        if let Some(carried) = level.get(paired) {
            next.push(*carried);
        }
        level = next;
    }
    Ok(Some(stored))
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
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<HashMap<usize, DeepestRepresentationChoice>> {
    cancellation.check()?;
    let paired_len = leaf_level.len() / 2 * 2;
    let child_pairs = leaf_level[..paired_len]
        .chunks_exact(2)
        .map(|children| [children[0], children[1]])
        .collect::<Vec<_>>();
    #[cfg(feature = "sort_rayon")]
    let plan_results: Vec<_> = child_pairs
        .par_iter()
        .map(|children| {
            cancellation.check()?;
            let plan = build_deepest_pairing_plan(
                *children,
                canonical_source,
                temporary,
                support_sigma,
                branching_factor,
            )?;
            cancellation.check()?;
            Ok::<_, CancelableLodBuildError>(plan)
        })
        .collect();
    #[cfg(not(feature = "sort_rayon"))]
    let plan_results: Vec<_> = child_pairs
        .iter()
        .map(|children| {
            cancellation.check()?;
            let plan = build_deepest_pairing_plan(
                *children,
                canonical_source,
                temporary,
                support_sigma,
                branching_factor,
            )?;
            cancellation.check()?;
            Ok::<_, CancelableLodBuildError>(plan)
        })
        .collect();
    let mut plans = plan_results.into_iter().collect::<Result<Vec<_>, _>>()?;

    let carried_count = leaf_level
        .get(paired_len)
        .map(|carried| temporary[*carried].representation_count);

    let maximum_storage = canonical_source
        .len()
        .checked_mul(2)
        .ok_or(LodBuildError::CountOverflow("progressive storage budget"))?;
    let proposed_storage = projected_progressive_storage(
        &plans,
        canonical_source.len(),
        carried_count,
        branching_factor,
        cancellation,
    )?
    .ok_or(LodBuildError::CountOverflow("progressive stored Gaussians"))?;
    let required_adjustments = proposed_storage.saturating_sub(maximum_storage);
    let mut adjustments = BinaryHeap::new();
    for (choice_index, plan) in plans.iter().enumerate() {
        cancellation.poll(choice_index)?;
        if let Some(adjustment) = plan.next_adjustment(choice_index) {
            adjustments.push(adjustment);
        }
    }
    let mut applied_adjustments = Vec::with_capacity(required_adjustments);
    for adjustment_index in 0..required_adjustments {
        cancellation.poll(adjustment_index)?;
        let adjustment = adjustments
            .pop()
            .ok_or(LodBuildError::CountOverflow("progressive storage budget"))?;
        let plan = &mut plans[adjustment.choice_index];
        if adjustment.next_pair_count != plan.pair_count + 1 {
            return Err(LodBuildError::CountOverflow("progressive pairing adjustment").into());
        }
        plan.pair_count = adjustment.next_pair_count;
        applied_adjustments.push(adjustment.choice_index);
    }

    let adjusted_storage = projected_progressive_storage(
        &plans,
        canonical_source.len(),
        carried_count,
        branching_factor,
        cancellation,
    )?
    .ok_or(LodBuildError::CountOverflow("progressive stored Gaussians"))?;
    if adjusted_storage > maximum_storage {
        return Err(LodBuildError::CountOverflow("progressive storage budget").into());
    }

    // One deepest-record reduction can also cross an integer boundary at one
    // or more coarser rungs. Binary-search the least-risk prefix which keeps
    // the exact projected hierarchy inside the cap.
    let mut lower = 0;
    let mut upper = applied_adjustments.len();
    while lower < upper {
        cancellation.check()?;
        let middle = lower + (upper - lower) / 2;
        for (index, plan) in plans.iter_mut().enumerate() {
            cancellation.poll(index)?;
            plan.pair_count = plan.base_pair_count;
        }
        for (index, &choice_index) in applied_adjustments[..middle].iter().enumerate() {
            cancellation.poll(index)?;
            plans[choice_index].pair_count += 1;
        }
        let candidate_storage = projected_progressive_storage(
            &plans,
            canonical_source.len(),
            carried_count,
            branching_factor,
            cancellation,
        )?
        .ok_or(LodBuildError::CountOverflow("progressive stored Gaussians"))?;
        if candidate_storage <= maximum_storage {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    for (index, plan) in plans.iter_mut().enumerate() {
        cancellation.poll(index)?;
        plan.pair_count = plan.base_pair_count;
    }
    for (index, &choice_index) in applied_adjustments[..lower].iter().enumerate() {
        cancellation.poll(index)?;
        plans[choice_index].pair_count += 1;
    }
    let final_storage = projected_progressive_storage(
        &plans,
        canonical_source.len(),
        carried_count,
        branching_factor,
        cancellation,
    )?
    .ok_or(LodBuildError::CountOverflow("progressive stored Gaussians"))?;
    if final_storage > maximum_storage {
        return Err(LodBuildError::CountOverflow("progressive storage budget").into());
    }

    let plan_count = plans.len();
    #[cfg(feature = "sort_rayon")]
    let choice_results: Vec<_> = plans
        .into_par_iter()
        .map(|plan| {
            cancellation.check()?;
            let choice = materialize_deepest_pairing_plan(plan, canonical_source, support_sigma)?;
            cancellation.check()?;
            Ok::<_, CancelableLodBuildError>(choice)
        })
        .collect();
    #[cfg(not(feature = "sort_rayon"))]
    let choice_results: Vec<_> = plans
        .into_iter()
        .map(|plan| {
            cancellation.check()?;
            let choice = materialize_deepest_pairing_plan(plan, canonical_source, support_sigma)?;
            cancellation.check()?;
            Ok::<_, CancelableLodBuildError>(choice)
        })
        .collect();
    let mut choices = HashMap::with_capacity(plan_count);
    for (node_key, choice) in choice_results.into_iter().collect::<Result<Vec<_>, _>>()? {
        choices.insert(node_key, choice);
    }
    Ok(choices)
}

fn build_deepest_pairing_plan(
    children: [usize; 2],
    canonical_source: &[Gaussian3d],
    temporary: &[TempNode],
    support_sigma: f32,
    branching_factor: usize,
) -> Result<DeepestPairingPlan, LodBuildError> {
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
    Ok(DeepestPairingPlan {
        node_key: children[0],
        source_start,
        source_end,
        base_pair_count,
        pair_count: base_pair_count,
        base_quality,
        adjusted_quality,
    })
}

fn materialize_deepest_pairing_plan(
    plan: DeepestPairingPlan,
    canonical_source: &[Gaussian3d],
    support_sigma: f32,
) -> Result<(usize, DeepestRepresentationChoice), LodBuildError> {
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
    Ok((
        plan.node_key,
        DeepestRepresentationChoice { representatives },
    ))
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
            raw_sampled_projected_alpha_mass_inflation: 1.0,
            raw_projected_alpha_mass_inflation_upper_bound: 1.0,
            support_leakage_fraction: 0.0,
            support_growth: 1.0,
            major_scale_growth: 1.0,
            anisotropy_growth: 1.0,
        },
    })
}

pub(crate) fn validate_plane_lengths(cloud: &PlanarGaussian3d) -> Result<(), LodBuildError> {
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
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<BreadthFirstLayout> {
    let mut order = vec![root];
    let mut parents = vec![None];
    let mut depths = vec![0_u16];
    let mut cursor = 0;
    while cursor < order.len() {
        cancellation.poll(cursor)?;
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

fn source_center_bounds(
    source: &[Gaussian3d],
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<LodBounds> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for (index, gaussian) in source.iter().enumerate() {
        cancellation.poll(index)?;
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
        return Err(LodBuildError::DerivedNonFinite("Morton normalization extent").into());
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

fn compare_morton_source_indices(
    left: &MortonSourceIndex,
    right: &MortonSourceIndex,
    source: &[Gaussian3d],
) -> Ordering {
    left.morton
        .cmp(&right.morton)
        .then_with(|| compare_gaussians(&source[left.source_index], &source[right.source_index]))
}

fn merge_morton_runs(
    input: &[MortonSourceIndex],
    output: &mut [MortonSourceIndex],
    run_width: usize,
    source: &[Gaussian3d],
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<()> {
    let span = run_width.saturating_mul(2);
    #[cfg(feature = "sort_rayon")]
    let results: Vec<_> = output
        .par_chunks_mut(span)
        .enumerate()
        .map(|(run_index, destination)| {
            let start = run_index * span;
            let middle = (start + run_width).min(input.len());
            merge_morton_run_pair(
                &input[start..middle],
                &input[middle..start + destination.len()],
                destination,
                source,
                cancellation,
            )
        })
        .collect();
    #[cfg(not(feature = "sort_rayon"))]
    let results: Vec<_> = output
        .chunks_mut(span)
        .enumerate()
        .map(|(run_index, destination)| {
            let start = run_index * span;
            let middle = (start + run_width).min(input.len());
            merge_morton_run_pair(
                &input[start..middle],
                &input[middle..start + destination.len()],
                destination,
                source,
                cancellation,
            )
        })
        .collect();
    results.into_iter().collect()
}

fn merge_morton_run_pair(
    left: &[MortonSourceIndex],
    right: &[MortonSourceIndex],
    output: &mut [MortonSourceIndex],
    source: &[Gaussian3d],
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<()> {
    cancellation.check()?;
    let mut left_index = 0;
    let mut right_index = 0;
    for (output_index, destination) in output.iter_mut().enumerate() {
        cancellation.poll(output_index)?;
        let take_left = right_index == right.len()
            || (left_index < left.len()
                && compare_morton_source_indices(&left[left_index], &right[right_index], source)
                    != Ordering::Greater);
        if take_left {
            *destination = left[left_index];
            left_index += 1;
        } else {
            *destination = right[right_index];
            right_index += 1;
        }
    }
    cancellation.check()
}

fn sort_morton_source_indices(
    entries: &mut [MortonSourceIndex],
    source: &[Gaussian3d],
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<()> {
    cancellation.check()?;
    #[cfg(feature = "sort_rayon")]
    let results: Vec<_> = entries
        .par_chunks_mut(LOD_MORTON_SORT_RUN_LEN)
        .map(|run| {
            cancellation.check()?;
            run.sort_unstable_by(|left, right| compare_morton_source_indices(left, right, source));
            cancellation.check()
        })
        .collect();
    #[cfg(not(feature = "sort_rayon"))]
    let results: Vec<_> = entries
        .chunks_mut(LOD_MORTON_SORT_RUN_LEN)
        .map(|run| {
            cancellation.check()?;
            run.sort_unstable_by(|left, right| compare_morton_source_indices(left, right, source));
            cancellation.check()
        })
        .collect();
    results
        .into_iter()
        .collect::<CancelableLodBuildResult<()>>()?;
    if entries.len() <= LOD_MORTON_SORT_RUN_LEN {
        return Ok(());
    }

    let mut scratch = Vec::with_capacity(entries.len());
    for chunk in entries.chunks(LOD_BUILD_CANCEL_CHECK_INTERVAL) {
        cancellation.check()?;
        scratch.extend_from_slice(chunk);
    }

    let mut run_width = LOD_MORTON_SORT_RUN_LEN;
    let mut input_is_entries = true;
    while run_width < entries.len() {
        cancellation.check()?;
        if input_is_entries {
            merge_morton_runs(entries, &mut scratch, run_width, source, cancellation)?;
        } else {
            merge_morton_runs(&scratch, entries, run_width, source, cancellation)?;
        }
        input_is_entries = !input_is_entries;
        run_width = run_width.saturating_mul(2);
    }
    if !input_is_entries {
        for (index, (destination, sorted)) in entries.iter_mut().zip(&scratch).enumerate() {
            cancellation.poll(index)?;
            *destination = *sorted;
        }
    }
    cancellation.check()
}

fn source_fingerprint(
    source_order: &[MortonSourceIndex],
    source: &[Gaussian3d],
    cancellation: LodBuildCancellation<'_>,
) -> CancelableLodBuildResult<u64> {
    let mut hash = StableHasher::new();
    hash.write(&(source_order.len() as u64).to_le_bytes());
    for (index, entry) in source_order.iter().enumerate() {
        cancellation.poll(index)?;
        hash.write(&entry.morton.to_le_bytes());
        hash.write(&stable_gaussian_hash(&source[entry.source_index]).to_le_bytes());
    }
    Ok(hash.finish())
}

/// Canonical total order for Gaussian payloads inside an equal Morton key.
///
/// External CPU runs, GPU readback fixups, and performance oracles must use
/// this exact comparison before the source-index tiebreaker. In particular,
/// sorting only by `(morton, source_index)` is not the package builder's merge
/// contract when spatial quantization produces collisions.
pub fn compare_gaussians(left: &Gaussian3d, right: &Gaussian3d) -> Ordering {
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

fn normalized_gaussian_rotation(gaussian: &Gaussian3d) -> Result<[[f64; 3]; 3], LodBuildError> {
    let [w, x, y, z] = gaussian.rotation.rotation.map(f64::from);
    let norm = (w * w + x * x + y * y + z * z).sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(LodBuildError::DerivedNonFinite("rotation normalization"));
    }
    let (w, x, y, z) = (w / norm, x / norm, y / norm, z / norm);
    Ok([
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
    ])
}

fn rotate_diagonal_symmetric(
    rotation: [[f64; 3]; 3],
    diagonal: [f64; 3],
    derived_name: &'static str,
) -> Result<[[f64; 3]; 3], LodBuildError> {
    let mut matrix = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..=row {
            let value = (0..3)
                .map(|axis| rotation[row][axis] * diagonal[axis] * rotation[column][axis])
                .sum::<f64>();
            if !value.is_finite() {
                return Err(LodBuildError::DerivedNonFinite(derived_name));
            }
            matrix[row][column] = value;
            matrix[column][row] = value;
        }
    }
    Ok(matrix)
}

fn gaussian_covariance(gaussian: &Gaussian3d) -> Result<[[f64; 3]; 3], LodBuildError> {
    let rotation = normalized_gaussian_rotation(gaussian)?;
    let scale_squared = gaussian
        .scale_opacity
        .scale
        .map(|scale| f64::from(scale) * f64::from(scale));
    // Q is the conventional quaternion matrix. The renderer stores Q^T and
    // evaluates R^T D R, so its effective covariance is also Q D Q^T.
    rotate_diagonal_symmetric(rotation, scale_squared, "Gaussian covariance")
}

#[derive(Clone, Copy)]
struct GaussianCovarianceFrame {
    covariance: [[f64; 3]; 3],
    projected_area_sqrt: [[f64; 3]; 3],
}

fn gaussian_covariance_frame(
    gaussian: &Gaussian3d,
) -> Result<GaussianCovarianceFrame, LodBuildError> {
    let rotation = normalized_gaussian_rotation(gaussian)?;
    let [scale_x, scale_y, scale_z] = gaussian.scale_opacity.scale.map(f64::from);
    let covariance = rotate_diagonal_symmetric(
        rotation,
        [scale_x * scale_x, scale_y * scale_y, scale_z * scale_z],
        "Gaussian covariance",
    )?;
    // For Sigma = Q diag(sx^2, sy^2, sz^2) Q^T, the principal square root of
    // adj(Sigma) is Q diag(|sy sz|, |sx sz|, |sx sy|) Q^T. Deriving it from the
    // authored frame avoids a Jacobi eigensolve for every source insertion.
    let projected_area_sqrt = rotate_diagonal_symmetric(
        rotation,
        [
            (scale_y * scale_z).abs(),
            (scale_x * scale_z).abs(),
            (scale_x * scale_y).abs(),
        ],
        "projected-area PSD square root",
    )?;
    Ok(GaussianCovarianceFrame {
        covariance,
        projected_area_sqrt,
    })
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
/// matching the covariance reconstruction path. Retained as a test oracle for
/// the authored-frame fast path.
#[cfg(test)]
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
    representative_projected_area: [[f64; 3]; 3],
    support_sigma: f32,
    raw_projected_alpha_mass_inflation_upper_bound: f64,
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

    let position = representative.position_visibility.position.map(f64::from);
    let mut raw_sampled_projected_alpha_mass_inflation = 0.0_f64;
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
        raw_sampled_projected_alpha_mass_inflation =
            raw_sampled_projected_alpha_mass_inflation.max(inflation);

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
    Ok(MomentMergeRasterRisk {
        raw_sampled_projected_alpha_mass_inflation: bounded_risk_f32(
            raw_sampled_projected_alpha_mass_inflation,
            "raw sampled projected alpha-mass inflation",
        )?,
        raw_projected_alpha_mass_inflation_upper_bound: bounded_upper_risk_f32(
            raw_projected_alpha_mass_inflation_upper_bound,
            "raw projected alpha-mass inflation upper bound",
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
    let maximum = if let Some(cholesky) = cholesky_3x3(source_projected_alpha_mass_quadratic) {
        let inverse = invert_lower_triangular_3x3(cholesky);
        largest_symmetric_eigenvalue(multiply_3x3(
            multiply_3x3(inverse, representative_projected_area),
            transpose_3x3(inverse),
        ))?
    } else {
        support_restricted_projected_alpha_quotient(
            representative_projected_area,
            source_projected_alpha_mass_sqrt_sum,
        )?
    };
    Ok(representative_alpha * maximum.sqrt())
}

fn largest_symmetric_eigenvalue(mut matrix: [[f64; 3]; 3]) -> Result<f64, LodBuildError> {
    for (row, column) in [(0, 1), (0, 2), (1, 2)] {
        let symmetric = 0.5 * (matrix[row][column] + matrix[column][row]);
        matrix[row][column] = symmetric;
        matrix[column][row] = symmetric;
    }
    let (eigenvalues, _) = symmetric_eigendecomposition(matrix)?;
    Ok(eigenvalues.into_iter().fold(0.0_f64, f64::max).max(0.0))
}

/// Generalized PSD quotient used when the source projected-area quadratic is
/// singular or too ill-conditioned for Cholesky. On the range of
/// `source_projected_alpha_mass_sqrt_sum`, this evaluates the same quotient as
/// the positive-definite path. Directions in its nullspace are admissible only
/// when the representative projected area is null there too; otherwise no
/// positive representative alpha can satisfy the all-view bound.
///
/// The eigensolve is performed on the square-root sum rather than its square,
/// retaining twice as many conditioning bits for nearly planar splats. A small
/// reconstruction guard separates authored zero modes from Jacobi roundoff.
/// Supported singular values are reduced by that guard and the numerator is
/// enlarged by its corresponding roundoff allowance, so the finite quotient
/// remains conservative on the retained support. A rank-deficient
/// representative whose f32 frame re-encoding materially leaks outside that
/// support still returns infinity and is deliberately calibrated to zero.
fn support_restricted_projected_alpha_quotient(
    representative_projected_area: [[f64; 3]; 3],
    source_projected_alpha_mass_sqrt_sum: [[f64; 3]; 3],
) -> Result<f64, LodBuildError> {
    let source_scale = source_projected_alpha_mass_sqrt_sum
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let representative_scale = representative_projected_area
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !source_scale.is_finite() || !representative_scale.is_finite() {
        return Err(LodBuildError::DerivedNonFinite(
            "projected alpha-mass calibration",
        ));
    }
    if source_scale == 0.0 {
        return Ok(if representative_scale == 0.0 {
            0.0
        } else {
            f64::INFINITY
        });
    }

    let normalized_source =
        source_projected_alpha_mass_sqrt_sum.map(|row| row.map(|value| value / source_scale));
    let (source_eigenvalues, source_eigenvectors) =
        symmetric_eigendecomposition(normalized_source)?;
    let reconstructed_source = rotate_diagonal_symmetric(
        source_eigenvectors,
        source_eigenvalues,
        "projected alpha-mass support reconstruction",
    )?;
    let reconstruction_error = normalized_source
        .iter()
        .flatten()
        .zip(reconstructed_source.iter().flatten())
        .map(|(source, reconstructed)| (source - reconstructed).powi(2))
        .sum::<f64>()
        .sqrt();
    // The Jacobi solver's stopping rule and the reconstruction itself each
    // contribute a few ulps. This is an absolute guard because the source was
    // normalized above.
    let support_guard = reconstruction_error + 64.0 * f64::EPSILON;
    let mut inverse_supported_singular_value = [0.0; 3];
    let mut supported = [false; 3];
    for axis in 0..3 {
        let singular_value = source_eigenvalues[axis].abs();
        if singular_value > support_guard {
            supported[axis] = true;
            inverse_supported_singular_value[axis] =
                (source_scale * (singular_value - support_guard)).recip();
        }
    }

    let representative_in_source_frame = multiply_3x3(
        multiply_3x3(
            transpose_3x3(source_eigenvectors),
            representative_projected_area,
        ),
        source_eigenvectors,
    );
    let numerator_guard = representative_scale * (support_guard + 64.0 * f64::EPSILON);
    for row in 0..3 {
        for column in 0..3 {
            if (!supported[row] || !supported[column])
                && representative_in_source_frame[row][column].abs() > numerator_guard
            {
                return Ok(f64::INFINITY);
            }
        }
    }
    if !supported.into_iter().any(|is_supported| is_supported) {
        return Ok(0.0);
    }

    let mut normalized_representative = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            representative_in_source_frame[row][column]
                * inverse_supported_singular_value[row]
                * inverse_supported_singular_value[column]
        })
    });
    for axis in 0..3 {
        if supported[axis] {
            // Enlarge the support-restricted numerator to cover eigenspace and
            // adjugate roundoff instead of letting it reduce the proven bound.
            normalized_representative[axis][axis] += numerator_guard
                * inverse_supported_singular_value[axis]
                * inverse_supported_singular_value[axis];
        }
    }
    largest_symmetric_eigenvalue(normalized_representative)
}

/// Keep a MomentMerge representative inside the source mixture's projected
/// alpha-mass envelope for every view direction. Mixture covariance includes
/// between-center spread, which is correct as a 3D moment but can turn thin,
/// separated surface splats into a much larger raster primitive. Raw optical
/// depth union then overdraws that enlarged footprint. The existing Minkowski
/// bound is linear in representative alpha, so reducing opacity by that bound
/// is the narrow correction that preserves covariance without permitting a
/// bright, opaque blob.
///
/// The return value is the conservative all-view inflation bound of the raw
/// optical-depth-union representative, not the calibrated output. Callers use
/// that pre-calibration risk for pairing and fidelity certificates so the
/// required correction cannot disappear from hierarchy metadata.
fn calibrate_projected_alpha_mass(
    representative: &mut Gaussian3d,
    representative_projected_area: [[f64; 3]; 3],
    source_projected_alpha_mass_sqrt_sum: [[f64; 3]; 3],
) -> Result<f64, LodBuildError> {
    let visibility = representative
        .position_visibility
        .visibility
        .clamp(0.0, 1.0);
    let raw_opacity = representative.scale_opacity.opacity.clamp(0.0, 1.0);
    if visibility == 0.0 || raw_opacity <= 0.0 {
        representative.scale_opacity.opacity = raw_opacity;
        return Ok(0.0);
    }

    let unit_alpha_bound = projected_alpha_mass_inflation_upper_bound(
        1.0,
        representative_projected_area,
        source_projected_alpha_mass_sqrt_sum,
    )?;
    let raw_inflation_upper_bound = f64::from(raw_opacity * visibility) * unit_alpha_bound;
    if unit_alpha_bound.is_infinite() {
        representative.scale_opacity.opacity = 0.0;
        return Ok(raw_inflation_upper_bound);
    }
    if !unit_alpha_bound.is_finite() || unit_alpha_bound < 0.0 {
        return Err(LodBuildError::DerivedNonFinite(
            "projected alpha-mass calibration",
        ));
    }

    let maximum_opacity = if unit_alpha_bound <= f64::EPSILON {
        1.0
    } else {
        (f64::from(visibility) * unit_alpha_bound)
            .recip()
            .clamp(0.0, 1.0)
    };
    representative.scale_opacity.opacity = if f64::from(raw_opacity) <= maximum_opacity {
        raw_opacity
    } else {
        // Convert downward so f32 storage cannot round a proven upper bound
        // into a small violation. This also keeps every emitted alpha finite
        // and non-negative.
        let rounded = maximum_opacity as f32;
        if f64::from(rounded) > maximum_opacity {
            next_down(rounded).max(0.0)
        } else {
            rounded.max(0.0)
        }
    };
    let mut calibrated_bound =
        f64::from(representative.scale_opacity.opacity * visibility) * unit_alpha_bound;
    if calibrated_bound > 1.0 && representative.scale_opacity.opacity > 0.0 {
        representative.scale_opacity.opacity =
            next_down(representative.scale_opacity.opacity).max(0.0);
        calibrated_bound =
            f64::from(representative.scale_opacity.opacity * visibility) * unit_alpha_bound;
    }
    // A subnormal visibility can quantize many adjacent opacities to the same
    // effective alpha. Fail closed instead of iterating over that plateau.
    if calibrated_bound > 1.0 {
        representative.scale_opacity.opacity = 0.0;
        calibrated_bound = 0.0;
    }
    debug_assert!(calibrated_bound <= 1.0);
    Ok(raw_inflation_upper_bound)
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
    MissingMonotoneMorphMapFeature,
    MissingMorphMap,
    UnexpectedMorphMap,
    UnsupportedMorphMapVersion(u16),
    MorphRecordCapacityExceeded(u32),
    MorphNodeCountMismatch,
    InvalidMorphRunRange(usize),
    MorphRunCoverageMismatch,
    InvalidLeafMorphRuns(LodNodeId),
    InvalidMorphRunCount {
        node: LodNodeId,
        expected: u32,
        actual: usize,
    },
    ZeroMorphRun(LodNodeId),
    MorphChildCoverageMismatch {
        node: LodNodeId,
        expected: u64,
        actual: u64,
    },
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

    fn no_cancellation() -> LodBuildCancellation<'static> {
        fn never() -> bool {
            false
        }
        LodBuildCancellation {
            is_canceled: &never,
        }
    }

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

    fn spatial_test_node(source_records: Vec<Gaussian3d>) -> SpatialMomentMergeNode {
        let representative = MomentMergeReducer::default()
            .reduce(&source_records)
            .unwrap();
        let authored_support_bounds = source_records
            .iter()
            .map(|sample| gaussian_oriented_support_bounds(sample, 3.0).unwrap())
            .reduce(LodBounds::union)
            .unwrap();
        let source_count = source_records.len();
        SpatialMomentMergeNode {
            representatives: vec![representative],
            source_records: Some(source_records),
            source_ranges: std::iter::once(0..source_count).collect(),
            authored_support_bounds,
            spatial_certificate_cap: 1.0,
            spatial_geometric_error_floor: 0.0,
        }
    }

    fn spatial_test_node_with_partitions(
        partitions: Vec<Vec<Gaussian3d>>,
    ) -> SpatialMomentMergeNode {
        let mut source_records = Vec::new();
        let mut source_ranges = Vec::new();
        let mut representatives = Vec::new();
        for partition in partitions {
            assert!(!partition.is_empty());
            let start = source_records.len();
            representatives.push(MomentMergeReducer::default().reduce(&partition).unwrap());
            source_records.extend(partition);
            source_ranges.push(start..source_records.len());
        }
        let authored_support_bounds = source_records
            .iter()
            .map(|sample| gaussian_oriented_support_bounds(sample, 3.0).unwrap())
            .reduce(LodBounds::union)
            .unwrap();
        SpatialMomentMergeNode {
            representatives,
            source_records: Some(source_records),
            source_ranges,
            authored_support_bounds,
            spatial_certificate_cap: 1.0,
            spatial_geometric_error_floor: 0.0,
        }
    }

    fn feasible_spatial_strip(x: f32) -> Vec<Gaussian3d> {
        let mut strip = [-1.5_f32, -0.5, 0.5, 1.5]
            .into_iter()
            .map(|y| gaussian([x, y, 1.0], [0.2, 0.2, 0.01], 0.2, 0.0))
            .collect::<Vec<_>>();
        // Tiny authored contributors make the strict source-support envelope
        // wider than the dominant surface without materially changing its
        // composited reference. Tangent widening therefore has real source
        // support available on both axes.
        strip.push(gaussian(
            [x - 1.5, -3.0, 1.0],
            [0.2, 0.2, 0.01],
            0.000_1,
            0.0,
        ));
        strip.push(gaussian(
            [x + 1.5, 3.0, 1.0],
            [0.2, 0.2, 0.01],
            0.000_1,
            0.0,
        ));
        strip
    }

    fn spatial_fit_benchmark_node(x: f32) -> SpatialMomentMergeNode {
        let partitions = [-12.0_f32, 0.0, 12.0]
            .into_iter()
            .flat_map(|z| {
                [-12.0_f32, 0.0, 12.0].into_iter().map(move |y| {
                    let mut partition = feasible_spatial_strip(x);
                    for sample in &mut partition {
                        sample.position_visibility.position[1] += y;
                        sample.position_visibility.position[2] += z;
                    }
                    partition
                })
            })
            .collect();
        spatial_test_node_with_partitions(partitions)
    }

    fn spatial_fit_exactness_node(x: f32) -> SpatialMomentMergeNode {
        let partitions = [-12.0_f32, 0.0, 12.0]
            .into_iter()
            .map(|y| {
                let mut partition = feasible_spatial_strip(x);
                for sample in &mut partition {
                    sample.position_visibility.position[1] += y;
                }
                partition
            })
            .collect();
        spatial_test_node_with_partitions(partitions)
    }

    fn spatial_nearest_boundary_contributor_with_overrides_for_test(
        nodes: &[SpatialMomentMergeNode],
        node_index: usize,
        other_bounds: LodBounds,
        target: [f32; 3],
        overrides: &[SpatialRepresentativeOverride],
    ) -> usize {
        nodes[node_index]
            .representatives
            .iter()
            .enumerate()
            .min_by(|(left_index, _), (right_index, _)| {
                let left = spatial_probe_representative(nodes, node_index, *left_index, overrides);
                let right =
                    spatial_probe_representative(nodes, node_index, *right_index, overrides);
                let left_overlap = lod_bounds_touch_or_overlap(left.support_bounds, other_bounds);
                let right_overlap = lod_bounds_touch_or_overlap(right.support_bounds, other_bounds);
                (!left_overlap)
                    .cmp(&(!right_overlap))
                    .then_with(|| {
                        point_to_point_squared_distance(
                            left.gaussian.position_visibility.position,
                            target,
                        )
                        .total_cmp(&point_to_point_squared_distance(
                            right.gaussian.position_visibility.position,
                            target,
                        ))
                    })
                    .then_with(|| {
                        point_to_bounds_squared_distance(
                            left.gaussian.position_visibility.position,
                            other_bounds,
                        )
                        .total_cmp(&point_to_bounds_squared_distance(
                            right.gaussian.position_visibility.position,
                            other_bounds,
                        ))
                    })
                    .then_with(|| left_index.cmp(right_index))
            })
            .map(|(index, _)| index)
            .unwrap()
    }

    fn spatial_fixed_scale_metrics(
        nodes: &[SpatialMomentMergeNode],
        probe: SpatialBoundaryProbe,
        overrides: &[SpatialRepresentativeOverride],
        pixels_per_world: f64,
    ) -> (f64, f64) {
        let left_source =
            spatial_probe_source(nodes, probe.left_node, probe.left_representative).unwrap();
        let right_source =
            spatial_probe_source(nodes, probe.right_node, probe.right_representative).unwrap();
        let left_representative = spatial_probe_representative(
            nodes,
            probe.left_node,
            probe.left_representative,
            overrides,
        );
        let right_representative = spatial_probe_representative(
            nodes,
            probe.right_node,
            probe.right_representative,
            overrides,
        );
        let mut boundary_reference = 0.0_f64;
        let mut boundary_error = 0.0_f64;
        let mut composited_error = 0.0_f64;
        for direction in PROJECTED_ALPHA_MASS_DIRECTIONS {
            let (horizontal, vertical) = spatial_projection_basis(direction);
            let left_center = spatial_project_point(
                left_representative.gaussian.position_visibility.position,
                horizontal,
                vertical,
            );
            let right_center = spatial_project_point(
                right_representative.gaussian.position_visibility.position,
                horizontal,
                vertical,
            );
            let midpoint = [
                0.5 * (left_center[0] + right_center[0]),
                0.5 * (left_center[1] + right_center[1]),
            ];
            for (point_index, point) in [left_center, midpoint, right_center]
                .into_iter()
                .enumerate()
            {
                let reference = spatial_renderer_alpha_at(
                    left_source.iter().chain(right_source.iter()),
                    point,
                    horizontal,
                    vertical,
                    pixels_per_world,
                    false,
                )
                .unwrap();
                let emitted = spatial_renderer_alpha_at(
                    [
                        &left_representative.gaussian,
                        &right_representative.gaussian,
                    ],
                    point,
                    horizontal,
                    vertical,
                    pixels_per_world,
                    true,
                )
                .unwrap();
                let error = (emitted - reference).abs();
                composited_error += error;
                if point_index == 1 {
                    boundary_reference += reference;
                    boundary_error += error;
                }
            }
        }
        (
            if boundary_reference <= SPATIAL_FIT_MIN_REFERENCE_ALPHA {
                0.0
            } else {
                boundary_error / boundary_reference
            },
            composited_error,
        )
    }

    struct SpatialSelectionHierarchy {
        spatial_error_floor: f32,
    }

    impl crate::stream::hierarchy::LodHierarchy for SpatialSelectionHierarchy {
        type NodeId = u32;

        fn roots(&self) -> &[Self::NodeId] {
            const ROOTS: [u32; 1] = [0];
            &ROOTS
        }

        fn parent(&self, node: Self::NodeId) -> Option<Self::NodeId> {
            match node {
                1 | 2 => Some(0),
                3 | 4 => Some(1),
                5 | 6 => Some(2),
                _ => None,
            }
        }

        fn children(&self, node: Self::NodeId) -> &[Self::NodeId] {
            const ROOT_CHILDREN: [u32; 2] = [1, 2];
            const LEFT_CHILDREN: [u32; 2] = [3, 4];
            const RIGHT_CHILDREN: [u32; 2] = [5, 6];
            match node {
                0 => &ROOT_CHILDREN,
                1 => &LEFT_CHILDREN,
                2 => &RIGHT_CHILDREN,
                _ => &[],
            }
        }

        fn metrics(&self, node: Self::NodeId) -> Option<crate::stream::hierarchy::LodNodeMetrics> {
            (node <= 6).then_some(crate::stream::hierarchy::LodNodeMetrics {
                center: bevy::math::Vec3::new(0.0, 0.0, 1.0),
                radius: 24.0,
                // The root models normal propagation of the measured child
                // error. Only the two internal-node values vary, proving that
                // their fitter-authored floor is what removes them at q=.65.
                geometric_error: if node == 0 {
                    1_000.0
                } else if node <= 2 {
                    self.spatial_error_floor
                } else {
                    0.0
                },
                appearance_error: 0.0,
                opacity_error: 0.0,
                quality_min: 1.0,
                quality_max: 1.0,
                high_fidelity_certificate: 1.0,
                representative_count: 1,
            })
        }
    }

    fn emitted_projected_alpha_mass_inflation_upper_bound(
        source: &[Gaussian3d],
        representative: &Gaussian3d,
    ) -> f64 {
        let reducer = MomentMergeReducer::default();
        let accumulator = reducer.accumulate_validated(source).unwrap();
        let representative_projected_area =
            symmetric_adjugate(gaussian_covariance(representative).unwrap());
        let representative_alpha = f64::from(
            representative.scale_opacity.opacity.clamp(0.0, 1.0)
                * representative
                    .position_visibility
                    .visibility
                    .clamp(0.0, 1.0),
        );
        projected_alpha_mass_inflation_upper_bound(
            representative_alpha,
            representative_projected_area,
            accumulator.projected_alpha_mass_sqrt_sum,
        )
        .unwrap()
    }

    fn assert_finite_nonzero_projected_alpha_safe(
        source: &[Gaussian3d],
        representative: &Gaussian3d,
    ) {
        assert!(representative.scale_opacity.opacity.is_finite());
        assert!(representative.scale_opacity.opacity > 0.0);
        let emitted_bound =
            emitted_projected_alpha_mass_inflation_upper_bound(source, representative);
        assert!(emitted_bound.is_finite());
        assert!(
            emitted_bound <= 1.0,
            "emitted all-view bound exceeds one: {emitted_bound}"
        );

        let representative_area = symmetric_adjugate(gaussian_covariance(representative).unwrap());
        let representative_alpha = f64::from(
            representative.scale_opacity.opacity
                * representative
                    .position_visibility
                    .visibility
                    .clamp(0.0, 1.0),
        );
        let source_area = source
            .iter()
            .map(|sample| {
                (
                    f64::from(
                        sample.scale_opacity.opacity.clamp(0.0, 1.0)
                            * sample.position_visibility.visibility.clamp(0.0, 1.0),
                    ),
                    symmetric_adjugate(gaussian_covariance(sample).unwrap()),
                )
            })
            .collect::<Vec<_>>();
        let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        for index in 0..2048 {
            let z = 1.0 - 2.0 * (index as f64 + 0.5) / 2048.0;
            let radius = (1.0 - z * z).sqrt();
            let azimuth = golden_angle * index as f64;
            let direction = [radius * azimuth.cos(), radius * azimuth.sin(), z];
            let representative_mass = representative_alpha
                * quadratic_form_f64(representative_area, direction)
                    .max(0.0)
                    .sqrt();
            let source_mass = source_area
                .iter()
                .map(|(alpha, area)| alpha * quadratic_form_f64(*area, direction).max(0.0).sqrt())
                .sum::<f64>();
            assert!(
                representative_mass <= source_mass * (1.0 + 2e-6) + 1e-30,
                "projected alpha mass inflated in direction {direction:?}: representative={representative_mass}, source={source_mass}, bound={emitted_bound}"
            );
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

    fn assert_matrix_relative_close(
        left: [[f64; 3]; 3],
        right: [[f64; 3]; 3],
        relative_epsilon: f64,
        absolute_epsilon: f64,
    ) {
        for row in 0..3 {
            for column in 0..3 {
                let tolerance = absolute_epsilon
                    + relative_epsilon * left[row][column].abs().max(right[row][column].abs());
                assert!(
                    (left[row][column] - right[row][column]).abs() <= tolerance,
                    "matrix mismatch [{row}][{column}]: {} vs {}, tolerance {tolerance}",
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

    /// Front-to-back alpha at one pixel for the default 3D Gaussian OBB path.
    ///
    /// The fixture uses a front-on unit projection, so the shader's projected
    /// covariance is the world-space xy block plus its 0.3 pixel low-pass term.
    /// Peak opacity is determinant-normalized for that dilation. Flat sources
    /// retain the authored-opacity adaptive cutoff, while LoD candidates keep
    /// at least the authored three-sigma OBB before fragment evaluation.
    fn renderer_mip_obb_alpha_at(
        samples: &[Gaussian3d],
        point: [f64; 2],
        lod_candidate: bool,
    ) -> f64 {
        let remaining_transmittance = samples.iter().fold(1.0_f64, |remaining, sample| {
            let authored_opacity = f64::from(sample.scale_opacity.opacity.clamp(0.0, 1.0));
            if authored_opacity == 0.0 {
                return remaining;
            }

            let covariance = gaussian_covariance(sample).unwrap();
            let unfiltered_xx = covariance[0][0];
            let xy = covariance[0][1];
            let unfiltered_yy = covariance[1][1];
            let filtered = crate::render::gaussian_mip_filter_covariance_2d([
                unfiltered_xx as f32,
                xy as f32,
                unfiltered_yy as f32,
            ]);
            let [xx, xy, yy] = filtered.covariance.map(f64::from);
            let opacity_scale = f64::from(filtered.opacity_scale);
            let opacity = authored_opacity * opacity_scale;
            if opacity == 0.0 {
                return remaining;
            }
            let mid = 0.5 * (xx + yy);
            let radius = (0.25 * (xx - yy) * (xx - yy) + xy * xy).sqrt();
            let major_variance = mid + radius;
            let minor_variance = (mid - radius).max(f64::MIN_POSITIVE);
            let major_axis = if xy.abs() + (major_variance - xx).abs() > 1e-15 {
                let length = (xy * xy + (major_variance - xx).powi(2)).sqrt();
                [-xy / length, (major_variance - xx) / length]
            } else {
                [1.0, 0.0]
            };
            let minor_axis = [major_axis[1], -major_axis[0]];
            let delta = [
                point[0] - f64::from(sample.position_visibility.position[0]),
                point[1] - f64::from(sample.position_visibility.position[1]),
            ];
            let major = delta[0] * major_axis[0] + delta[1] * major_axis[1];
            let minor = delta[0] * minor_axis[0] + delta[1] * minor_axis[1];
            let cutoff = f64::from(crate::render::gaussian_support_cutoff(
                authored_opacity as f32,
                true,
                lod_candidate,
            ));
            if major.abs() > cutoff * major_variance.sqrt()
                || minor.abs() > cutoff * minor_variance.sqrt()
            {
                return remaining;
            }

            let power = -0.5 * (major * major / major_variance + minor * minor / minor_variance);
            let alpha = (power.exp() * opacity).min(0.999);
            remaining * (1.0 - alpha)
        });
        1.0 - remaining_transmittance
    }

    fn renderer_mip_obb_alpha_mass(
        samples: &[Gaussian3d],
        center: [f64; 2],
        half_extent: f64,
        lod_candidate: bool,
    ) -> f64 {
        const SAMPLE_COUNT: usize = 33;
        (0..SAMPLE_COUNT)
            .flat_map(|y| (0..SAMPLE_COUNT).map(move |x| (x, y)))
            .map(|(x, y)| {
                let denominator = (SAMPLE_COUNT - 1) as f64;
                let point = [
                    center[0] - half_extent + 2.0 * half_extent * x as f64 / denominator,
                    center[1] - half_extent + 2.0 * half_extent * y as f64 / denominator,
                ];
                renderer_mip_obb_alpha_at(samples, point, lod_candidate)
            })
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
    fn moment_merge_preserves_mixture_moments_without_projected_alpha_inflation() {
        let source = [
            gaussian([-1.0, 0.0, 0.0], [0.1; 3], 0.5, -1.0),
            gaussian([1.0, 0.0, 0.0], [0.1; 3], 0.5, 1.0),
        ];
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();
        assert!(merged.gaussian.position_visibility.position[0].abs() < 1e-6);
        assert!(merged.gaussian.scale_opacity.opacity < 0.1);
        assert!(merged.gaussian.scale_opacity.opacity >= 0.0);
        assert!(merged.gaussian.scale_opacity.opacity.is_finite());
        assert!(merged.gaussian.spherical_harmonic.coefficients[0].abs() < 1e-6);

        let actual = gaussian_covariance(&merged.gaussian).unwrap();
        let expected = [[1.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]];
        assert_matrix_close(actual, expected, 1e-5);

        let risk = merged.raster_risk();
        let emitted_bound =
            emitted_projected_alpha_mass_inflation_upper_bound(&source, &merged.gaussian);
        assert!(
            risk.raw_projected_alpha_mass_inflation_upper_bound
                >= risk.raw_sampled_projected_alpha_mass_inflation
        );
        assert!(risk.raw_sampled_projected_alpha_mass_inflation > 1.0);
        assert!(risk.raw_projected_alpha_mass_inflation_upper_bound > 7.0);
        assert!(emitted_bound <= 1.0);
        assert!(risk.support_leakage_fraction > 0.5);
        assert!(risk.support_growth > 2.0);
        assert!(risk.major_scale_growth > 10.0);
        assert!(risk.score() > 9.0);
        assert!(risk.high_fidelity_certificate() < 0.11);
    }

    #[test]
    fn external_v2_reducer_retains_raw_union_opacity() {
        let source = [
            gaussian([-1.0, 0.0, 0.0], [0.1; 3], 0.5, 0.0),
            gaussian([1.0, 0.0, 0.0], [0.1; 3], 0.5, 0.0),
        ];
        let reducer = MomentMergeReducer::default();
        let calibrated = reducer.reduce(&source).unwrap();
        let external_v2 = reducer.reduce_external_v2(&source).unwrap();

        assert!(calibrated.gaussian.scale_opacity.opacity < 0.1);
        assert!((external_v2.gaussian.scale_opacity.opacity - 0.75).abs() < 1e-6);
        assert!(
            external_v2
                .raster_risk()
                .raw_projected_alpha_mass_inflation_upper_bound
                > 7.0
        );
        assert_eq!(
            calibrated
                .raster_risk()
                .raw_projected_alpha_mass_inflation_upper_bound,
            external_v2
                .raster_risk()
                .raw_projected_alpha_mass_inflation_upper_bound
        );
    }

    #[test]
    fn minkowski_bound_is_exact_for_coincident_identical_covariance() {
        let source = [gaussian([0.0; 3], [0.1; 3], 0.1, 0.0); 8];
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();
        let risk = merged.raster_risk();

        let expected_union_opacity = 1.0 - 0.9_f32.powi(8);
        assert!(
            (merged.gaussian.scale_opacity.opacity - expected_union_opacity).abs() < 1e-6,
            "safe coincident sources should retain optical-depth union"
        );

        // The sampled ratio is exact for this isotropic fixture: aggregation
        // loses projected alpha mass, and the representative support is exact.
        assert!((0.70..0.73).contains(&risk.raw_sampled_projected_alpha_mass_inflation));
        assert!(risk.support_leakage_fraction < 1e-5);
        assert!((risk.support_growth - 1.0).abs() < 1e-5);
        assert!((risk.major_scale_growth - 1.0).abs() < 1e-5);
        assert!(risk.score() < 1e-5);

        // Minkowski retains the cross terms for identical covariance, making
        // the all-view proof exact instead of falsely refining this safe
        // low-alpha overlap.
        assert!(
            (risk.raw_projected_alpha_mass_inflation_upper_bound
                - risk.raw_sampled_projected_alpha_mass_inflation)
                .abs()
                < 1e-5,
            "Minkowski bound should be exact: {risk:?}"
        );
        assert!((risk.high_fidelity_certificate() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn projected_alpha_calibration_preserves_coincident_zero_scale_sources() {
        let source = [
            gaussian([0.25, -0.5, 1.0], [0.0; 3], 0.25, 0.0),
            gaussian([0.25, -0.5, 1.0], [0.0; 3], 0.5, 0.0),
        ];
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();

        assert!((merged.gaussian.scale_opacity.opacity - 0.625).abs() < 1e-6);
        assert_eq!(merged.gaussian.scale_opacity.scale, [0.0; 3]);
        assert_finite_nonzero_projected_alpha_safe(&source, &merged.gaussian);
    }

    #[test]
    fn projected_alpha_calibration_preserves_coplanar_sources() {
        let source = [
            gaussian([-0.5, -0.25, 0.0], [0.25, 0.12, 0.0], 0.35, 0.0),
            gaussian([0.3, -0.1, 0.0], [0.18, 0.22, 0.0], 0.55, 0.0),
            gaussian([0.1, 0.4, 0.0], [0.3, 0.1, 0.0], 0.25, 0.0),
        ];
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();

        assert_eq!(merged.gaussian.scale_opacity.scale[2], 0.0);
        assert!(
            merged
                .raster_risk()
                .raw_projected_alpha_mass_inflation_upper_bound
                .is_finite()
        );
        assert_finite_nonzero_projected_alpha_safe(&source, &merged.gaussian);
    }

    #[test]
    fn projected_alpha_calibration_rejects_area_outside_zero_scale_support() {
        let source = [
            gaussian([-1.0, -1.0, 0.0], [0.0; 3], 0.5, 0.0),
            gaussian([1.0, -1.0, 0.0], [0.0; 3], 0.5, 0.0),
            gaussian([0.0, 1.0, 0.0], [0.0; 3], 0.5, 0.0),
        ];
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();

        assert_eq!(merged.gaussian.scale_opacity.opacity, 0.0);
        assert_eq!(
            merged
                .raster_risk()
                .raw_projected_alpha_mass_inflation_upper_bound,
            f32::MAX
        );
    }

    #[test]
    fn projected_alpha_calibration_preserves_near_singular_sources() {
        let source = [
            gaussian([0.0; 3], [1.0, 0.4, 1e-8], 0.1, 0.0),
            gaussian([0.0; 3], [1.0, 0.4, 1e-8], 0.2, 0.0),
            gaussian([0.0; 3], [1.0, 0.4, 1e-8], 0.4, 0.0),
        ];
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();

        assert!(
            merged
                .raster_risk()
                .raw_projected_alpha_mass_inflation_upper_bound
                .is_finite()
        );
        assert_finite_nonzero_projected_alpha_safe(&source, &merged.gaussian);
    }

    #[test]
    fn projected_alpha_calibration_retains_the_spd_bound() {
        let mut source = [gaussian([0.0; 3], [0.8, 0.3, 0.1], 0.2, 0.0); 3];
        let rotation = Quat::from_euler(bevy::math::EulerRot::XYZ, 0.37, -0.61, 1.19).normalize();
        for sample in &mut source {
            sample.rotation.rotation = [rotation.w, rotation.x, rotation.y, rotation.z];
        }
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();

        assert_finite_nonzero_projected_alpha_safe(&source, &merged.gaussian);
    }

    #[test]
    fn anisotropy_growth_limits_the_high_fidelity_certificate() {
        let risk = MomentMergeRasterRisk {
            raw_sampled_projected_alpha_mass_inflation: 1.0,
            raw_projected_alpha_mass_inflation_upper_bound: 1.0,
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
    fn analytic_projected_area_sqrt_reconstructs_rotated_adjugate() {
        let mut source = gaussian([0.0; 3], [2e-4, 7.5e-5, 2e-5], 0.5, 0.0);
        let rotation = Quat::from_euler(bevy::math::EulerRot::XYZ, -0.41, 0.83, 1.37).normalize();
        source.rotation.rotation = [rotation.w, rotation.x, rotation.y, rotation.z];
        let frame = gaussian_covariance_frame(&source).unwrap();
        let projected_area = symmetric_adjugate(frame.covariance);
        let generic = symmetric_psd_sqrt(projected_area).unwrap();
        let reconstructed = multiply_3x3(
            transpose_3x3(frame.projected_area_sqrt),
            frame.projected_area_sqrt,
        );

        assert_matrix_close(reconstructed, projected_area, 1e-24);
        assert_matrix_relative_close(frame.projected_area_sqrt, generic, 1e-8, 1e-24);
    }

    #[test]
    fn analytic_projected_area_sqrt_matches_generic_for_random_rotated_frames() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(0xa11f_1e57_5eed);
        for _ in 0..256 {
            let mut source = gaussian(
                [0.0; 3],
                std::array::from_fn(|_| 10.0_f32.powf(rng.random_range(-3.0..1.0))),
                0.5,
                0.0,
            );
            let rotation = Quat::from_euler(
                bevy::math::EulerRot::XYZ,
                rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
                rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
                rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
            )
            .normalize();
            let quaternion_scale = rng.random_range(0.1..10.0);
            source.rotation.rotation = [
                rotation.w * quaternion_scale,
                rotation.x * quaternion_scale,
                rotation.y * quaternion_scale,
                rotation.z * quaternion_scale,
            ];

            let frame = gaussian_covariance_frame(&source).unwrap();
            let generic = symmetric_psd_sqrt(symmetric_adjugate(frame.covariance)).unwrap();
            assert_matrix_relative_close(frame.projected_area_sqrt, generic, 2e-7, 1e-12);
        }
    }

    #[test]
    fn analytic_projected_area_sqrt_matches_generic_for_degenerate_scales() {
        let cases = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.5, 2.0],
            [1e-20, 0.25, 3.0],
        ];
        for (index, scale) in cases.into_iter().enumerate() {
            let mut source = gaussian([0.0; 3], scale, 0.5, 0.0);
            let rotation = Quat::from_euler(
                bevy::math::EulerRot::XYZ,
                0.31 + index as f32 * 0.17,
                -0.83 + index as f32 * 0.11,
                1.37 - index as f32 * 0.13,
            )
            .normalize();
            source.rotation.rotation = [rotation.w, rotation.x, rotation.y, rotation.z];

            let frame = gaussian_covariance_frame(&source).unwrap();
            let projected_area = symmetric_adjugate(frame.covariance);
            let generic = symmetric_psd_sqrt(projected_area).unwrap();
            let reconstructed = multiply_3x3(
                transpose_3x3(frame.projected_area_sqrt),
                frame.projected_area_sqrt,
            );
            let covariance_scale = frame
                .covariance
                .iter()
                .flatten()
                .map(|value| value.abs())
                .fold(1.0_f64, f64::max);

            assert_matrix_relative_close(frame.projected_area_sqrt, generic, 2e-7, 1e-7);
            assert_matrix_relative_close(
                reconstructed,
                projected_area,
                1e-10,
                1e-14 * covariance_scale * covariance_scale,
            );
        }
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

        let emitted_bound =
            emitted_projected_alpha_mass_inflation_upper_bound(&source, &merged.gaussian);
        assert!(
            risk.raw_projected_alpha_mass_inflation_upper_bound
                .is_finite()
        );
        assert!(
            sampled_max <= 1.0,
            "calibrated representative inflated projected alpha mass by {sampled_max}"
        );
        assert!(emitted_bound <= 1.0);
        assert!(
            emitted_bound >= sampled_max,
            "Minkowski bound {} fell below dense sampled ratio {sampled_max}",
            emitted_bound
        );
        assert!(f64::from(risk.raw_projected_alpha_mass_inflation_upper_bound) >= emitted_bound);
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
        let emitted_bound =
            emitted_projected_alpha_mass_inflation_upper_bound(&source, &merged.gaussian);
        assert!(risk.raw_sampled_projected_alpha_mass_inflation > 1.0);
        assert!(risk.raw_projected_alpha_mass_inflation_upper_bound > 1.0);
        assert!(emitted_bound <= 1.0);
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
        let emitted_bound =
            emitted_projected_alpha_mass_inflation_upper_bound(&source, &merged.gaussian);

        assert!(risk.raw_sampled_projected_alpha_mass_inflation > 1.0);
        assert!(risk.raw_projected_alpha_mass_inflation_upper_bound > 1.0);
        assert!(emitted_bound <= 1.0);
        assert!(risk.support_leakage_fraction > 0.4);
        assert!(risk.support_growth > 1.5);
        assert!(risk.major_scale_growth > 8.0);
        assert!(
            risk.high_fidelity_certificate() <= risk.major_scale_growth.recip(),
            "geometric growth must remain represented after opacity calibration: {risk:?}"
        );
        assert!(risk.high_fidelity_certificate() < 0.125);
    }

    #[test]
    fn garden_style_separated_thin_disks_are_faint_and_force_q65_refinement() {
        let source = [
            gaussian([-0.5, 0.0, 0.0], [0.05, 0.05, 0.000_05], 0.95, 0.0),
            gaussian([0.5, 0.0, 0.0], [0.05, 0.05, 0.000_05], 0.95, 0.0),
        ];
        let merged = MomentMergeReducer::default().reduce(&source).unwrap();
        let risk = merged.raster_risk();
        let emitted_bound =
            emitted_projected_alpha_mass_inflation_upper_bound(&source, &merged.gaussian);

        assert!(merged.gaussian.scale_opacity.opacity < 0.2);
        assert!(merged.gaussian.scale_opacity.opacity >= 0.0);
        assert!(merged.gaussian.scale_opacity.opacity.is_finite());
        assert!(risk.raw_sampled_projected_alpha_mass_inflation > 1.0);
        assert!(risk.raw_projected_alpha_mass_inflation_upper_bound > 1.0);
        assert!(emitted_bound <= 1.0);
        assert!(merged.high_fidelity_certificate() < 0.125);

        let settings = crate::gaussian::lod_settings::GaussianLodSettings {
            quality: 0.65,
            ..Default::default()
        };
        let target = settings.quality_target();
        let zero_projection_pressure =
            target.node_pressure(1.0, 0.0, 0.0, merged.high_fidelity_certificate(), false);
        assert_eq!(
            zero_projection_pressure, 0.0,
            "q=.65 must not turn a source-risk certificate into distance-independent refinement"
        );

        // ABI 16 carries the merge's source-support geometric error into the
        // selection-visible node metadata. At q=.65 that projected error, not
        // the high-quality-only certificate gate, rejects the representative
        // when it is large on screen while still allowing useful far-field
        // coarsening.
        assert!(merged.error.geometric > 0.0);
        let metrics = crate::stream::hierarchy::LodNodeMetrics {
            center: bevy::math::Vec3::new(0.0, 0.0, 0.0),
            radius: merged.support_bounds.radius(),
            geometric_error: merged.error.geometric,
            appearance_error: 0.0,
            opacity_error: 0.0,
            quality_min: 0.0,
            quality_max: 1.0,
            high_fidelity_certificate: merged.high_fidelity_certificate(),
            representative_count: 1,
        };
        let effective_limit = target.effective_max_screen_space_error_px().unwrap();
        let view_for_error = |error_px: f32| {
            crate::stream::hierarchy::LodView::orthographic(
                bevy::math::Vec3::new(0.0, 0.0, 4.0),
                720.0,
                merged.error.geometric * 720.0 / error_px,
                0.1,
            )
        };
        let near_view = view_for_error(1.25 * effective_limit);
        let far_view = view_for_error(0.5 * effective_limit);
        let pressure_at = |view: crate::stream::hierarchy::LodView| {
            target.node_pressure(
                1.0,
                view.projected_error_px(metrics),
                view.projected_coverage(metrics),
                metrics.high_fidelity_certificate,
                false,
            )
        };
        assert!(
            pressure_at(near_view) > 1.0,
            "an extreme surface merge must refine when its geometric error is screen-visible"
        );
        assert!(
            pressure_at(far_view) <= 1.0,
            "q=.65 must retain a useful distance response for the same representative"
        );

        let high_quality = crate::gaussian::lod_settings::GaussianLodSettings {
            quality: 0.95,
            ..Default::default()
        };
        assert!(
            high_quality.quality_target().node_pressure(
                1.0,
                0.0,
                0.0,
                merged.high_fidelity_certificate(),
                false,
            ) > 1.0,
            "the raw raster-risk certificate must remain authoritative at high quality"
        );
    }

    #[test]
    fn independently_reduced_siblings_preserve_boundary_alpha_mass() {
        // A uniform 8x8 surface is split into four adjacent hierarchy nodes.
        // Every source support remains exclusively owned by one node, matching
        // the external builder. Each node is then reduced independently to one
        // representative, which is the seam-producing operation under test.
        let mut source = Vec::new();
        let mut siblings = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for y in 0..8 {
            for x in 0..8 {
                let position = [-14.0 + 4.0 * x as f32, -14.0 + 4.0 * y as f32, 1.0];
                let sample = gaussian(position, [1.6, 1.6, 0.01], 0.2, 0.0);
                source.push(sample);
                siblings[usize::from(y >= 4) * 2 + usize::from(x >= 4)].push(sample);
            }
        }
        let reduced = siblings
            .iter()
            .map(|sibling| MomentMergeReducer::default().reduce(sibling).unwrap())
            .collect::<Vec<_>>();
        let representatives = reduced
            .iter()
            .map(|representative| representative.gaussian)
            .collect::<Vec<_>>();

        // These equal-area patches differ only by an integer multiple of the
        // uniform source spacing: the source raster therefore provides the
        // same alpha mass at a node interior and across a sibling boundary.
        let source_interior = renderer_mip_obb_alpha_mass(&source, [-8.0, -8.0], 2.0, false);
        let source_boundary = renderer_mip_obb_alpha_mass(&source, [0.0, -8.0], 2.0, false);
        assert!(
            (source_boundary / source_interior - 1.0).abs() < 0.01,
            "synthetic source is not spatially uniform: interior={source_interior}, boundary={source_boundary}"
        );

        let emitted_interior =
            renderer_mip_obb_alpha_mass(&representatives, [-8.0, -8.0], 2.0, true);
        let emitted_boundary =
            renderer_mip_obb_alpha_mass(&representatives, [0.0, -8.0], 2.0, true);
        let interior_retention = emitted_interior / source_interior;
        let boundary_retention = emitted_boundary / source_boundary;

        assert!((source_interior - 202.197_838_137_588_04).abs() < 1e-9);
        assert!((source_boundary - 202.197_838_137_588_04).abs() < 1e-9);
        assert!((emitted_interior - 368.616_285_573_197_3).abs() < 1e-9);
        assert!((emitted_boundary - 189.281_802_650_101_47).abs() < 1e-9);
        assert!(boundary_retention < 0.52 * interior_retention);

        let mut spatial_nodes = siblings
            .iter()
            .cloned()
            .map(spatial_test_node)
            .collect::<Vec<_>>();
        let before = spatial_nodes
            .iter()
            .flat_map(|node| node.representatives.iter().map(|result| result.gaussian))
            .collect::<Vec<_>>();
        let report = fit_spatial_moment_merge_sibling_cohort(&mut spatial_nodes, 3.0).unwrap();
        assert_eq!(report.touching_node_pairs, 6);
        assert_eq!(report.overlapping_node_pairs, 6);
        assert_eq!(report.unmeasured_touching_node_pairs, 0);
        assert_eq!(
            report.touching_node_pairs,
            report.overlapping_node_pairs + report.unmeasured_touching_node_pairs
        );
        let after = spatial_nodes
            .iter()
            .flat_map(|node| node.representatives.iter().map(|result| result.gaussian))
            .collect::<Vec<_>>();
        assert_eq!(report.accepted_edits, 0);
        assert!(report.unsafe_node_pairs > 0);
        assert_eq!(after, before, "infeasible fit changed representative bytes");
        assert!(
            spatial_nodes
                .iter()
                .any(|node| node.spatial_geometric_error_floor > 0.0)
        );
        let spatial_error_floor = spatial_nodes
            .iter()
            .map(|node| node.spatial_geometric_error_floor)
            .fold(0.0_f32, f32::max);
        let settings = crate::gaussian::lod_settings::GaussianLodSettings {
            quality: 0.65,
            ..Default::default()
        };
        let view = crate::stream::hierarchy::LodView::orthographic(
            bevy::math::Vec3::new(0.0, 0.0, 64.0),
            720.0,
            56.0,
            0.1,
        );
        let baseline = crate::stream::hierarchy::select_frontier(
            &SpatialSelectionHierarchy {
                spatial_error_floor: 0.0,
            },
            &crate::stream::hierarchy::AllResident,
            view,
            &settings,
        )
        .unwrap();
        assert_eq!(baseline.nodes, [1, 2]);
        let guarded = crate::stream::hierarchy::select_frontier(
            &SpatialSelectionHierarchy {
                spatial_error_floor,
            },
            &crate::stream::hierarchy::AllResident,
            view,
            &settings,
        )
        .unwrap();
        assert_eq!(guarded.nodes, [3, 4, 5, 6]);
    }

    #[test]
    fn feasible_spatial_sibling_fit_improves_boundary_and_cohort_error() {
        let mut nodes = vec![
            spatial_test_node(feasible_spatial_strip(-0.35)),
            spatial_test_node(feasible_spatial_strip(0.35)),
        ];
        let before = nodes
            .iter()
            .map(|node| node.representatives[0].gaussian)
            .collect::<Vec<_>>();
        let report = fit_spatial_moment_merge_sibling_cohort(&mut nodes, 3.0).unwrap();
        assert_eq!(report.touching_node_pairs, 1);
        assert_eq!(report.overlapping_node_pairs, 1);
        assert_eq!(report.unmeasured_touching_node_pairs, 0);
        assert!(report.maximum_relative_boundary_error_before > 0.1);
        assert!(report.accepted_edits > 0);
        assert!(
            report.maximum_relative_boundary_error_after
                < report.maximum_relative_boundary_error_before
        );
        assert!(report.cohort_composited_error_after <= report.cohort_composited_error_before);
        assert_ne!(
            nodes
                .iter()
                .map(|node| node.representatives[0].gaussian)
                .collect::<Vec<_>>(),
            before
        );
    }

    #[test]
    fn source_less_touching_coarse_pair_is_reported_without_blanket_refinement() {
        let mut nodes = vec![
            spatial_test_node(feasible_spatial_strip(-0.35)),
            spatial_test_node(feasible_spatial_strip(0.35)),
        ];
        for node in &mut nodes {
            node.source_records = None;
            node.source_ranges.clear();
        }
        let before = nodes
            .iter()
            .map(|node| node.representatives[0].gaussian)
            .collect::<Vec<_>>();
        let report = fit_spatial_moment_merge_sibling_cohort(&mut nodes, 3.0).unwrap();
        assert_eq!(report.touching_node_pairs, 1);
        assert_eq!(report.overlapping_node_pairs, 0);
        assert_eq!(report.unmeasured_touching_node_pairs, 1);
        assert_eq!(report.accepted_edits, 0);
        assert_eq!(report.unsafe_node_pairs, 0);
        assert!(nodes.iter().all(|node| {
            node.spatial_geometric_error_floor == 0.0 && node.spatial_certificate_cap == 1.0
        }));
        assert_eq!(
            nodes
                .iter()
                .map(|node| node.representatives[0].gaussian)
                .collect::<Vec<_>>(),
            before
        );
    }

    #[test]
    fn spatial_fit_branching_bound_is_explicit_and_allocation_bounded() {
        let maximum = spatial_moment_merge_fit_bounds(32).unwrap();
        assert_eq!(maximum.node_pair_checks, 496);
        assert_eq!(maximum.boundary_probes, 4_464);
        assert_eq!(size_of::<SpatialBoundaryReference>(), 2_912);
        let maximum_reference_cache_bytes = maximum
            .boundary_probes
            .checked_mul(size_of::<SpatialBoundaryReference>() as u64)
            .unwrap();
        assert!(maximum.scratch_host_bytes >= maximum_reference_cache_bytes);
        let node_pair_capacity = maximum.node_pair_checks as usize;
        let probe_capacity = maximum.boundary_probes as usize;
        let probes = Vec::<SpatialBoundaryProbe>::with_capacity(probe_capacity);
        let pair_probe_temporary =
            Vec::<SpatialBoundaryProbe>::with_capacity(SPATIAL_FIT_MAX_PROBES_PER_NODE_PAIR);
        let references = Vec::<SpatialBoundaryReference>::with_capacity(probe_capacity);
        let metrics = Vec::<SpatialBoundaryMetrics>::with_capacity(probe_capacity);
        let initial_metrics = Vec::<SpatialBoundaryMetrics>::with_capacity(probe_capacity);
        let incidence = Vec::<(usize, usize, usize)>::with_capacity(probe_capacity * 2);
        let affected_indices = Vec::<usize>::with_capacity(probe_capacity * 2);
        let current_affected =
            Vec::<(usize, SpatialBoundaryMetrics)>::with_capacity(probe_capacity);
        let best_affected = Vec::<(usize, SpatialBoundaryMetrics)>::with_capacity(probe_capacity);
        let unsafe_pairs = Vec::<(usize, usize, f64)>::with_capacity(node_pair_capacity);
        let current_overrides = Vec::<SpatialRepresentativeOverride>::with_capacity(2);
        let best_overrides = Vec::<SpatialRepresentativeOverride>::with_capacity(2);
        let actual_vec_payload_bytes = probes.capacity() * size_of::<SpatialBoundaryProbe>()
            + pair_probe_temporary.capacity() * size_of::<SpatialBoundaryProbe>()
            + references.capacity() * size_of::<SpatialBoundaryReference>()
            + metrics.capacity() * size_of::<SpatialBoundaryMetrics>()
            + initial_metrics.capacity() * size_of::<SpatialBoundaryMetrics>()
            + incidence.capacity() * size_of::<(usize, usize, usize)>()
            + affected_indices.capacity() * size_of::<usize>()
            + current_affected.capacity() * size_of::<(usize, SpatialBoundaryMetrics)>()
            + best_affected.capacity() * size_of::<(usize, SpatialBoundaryMetrics)>()
            + unsafe_pairs.capacity() * size_of::<(usize, usize, f64)>()
            + current_overrides.capacity() * size_of::<SpatialRepresentativeOverride>()
            + best_overrides.capacity() * size_of::<SpatialRepresentativeOverride>();
        assert_eq!(
            actual_vec_payload_bytes,
            spatial_fit_explicit_vec_payload_bytes(node_pair_capacity, probe_capacity).unwrap()
        );
        let non_vec_payload_bytes =
            32 * size_of::<SpatialMomentMergeNode>() + 2 * size_of::<MomentMergeResult>();
        assert_eq!(
            maximum.scratch_host_bytes as usize,
            actual_vec_payload_bytes + non_vec_payload_bytes
        );
        assert!(spatial_moment_merge_fit_bounds(33).is_none());

        let probes = vec![
            SpatialBoundaryProbe {
                left_node: 0,
                left_representative: 2,
                right_node: 1,
                right_representative: 4,
            },
            SpatialBoundaryProbe {
                left_node: 0,
                left_representative: 3,
                right_node: 1,
                right_representative: 4,
            },
            SpatialBoundaryProbe {
                left_node: 0,
                left_representative: 2,
                right_node: 2,
                right_representative: 1,
            },
        ];
        let incidence = spatial_probe_incidence(&probes);
        assert_eq!(incidence.len(), probes.len() * 2);
        assert!(incidence.windows(2).all(|pair| pair[0] <= pair[1]));
        let edited = [(0_usize, 2_usize), (1, 4)];
        let indexed = spatial_affected_probe_indices(&incidence, edited);
        let brute_force = probes
            .iter()
            .enumerate()
            .filter_map(|(index, probe)| {
                edited
                    .iter()
                    .any(|(node, representative)| {
                        (*node == probe.left_node && *representative == probe.left_representative)
                            || (*node == probe.right_node
                                && *representative == probe.right_representative)
                    })
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(indexed, brute_force);
    }

    #[test]
    fn spatial_fit_rejects_a_candidate_that_changes_contributor_topology() {
        let nodes = vec![
            spatial_test_node_with_partitions(vec![
                vec![gaussian([-0.1, 0.0, 0.0], [0.001; 3], 0.2, 0.0)],
                vec![gaussian([-1.0, 0.0, 0.0], [0.5; 3], 0.2, 0.0)],
            ]),
            spatial_test_node(vec![gaussian([0.1, 0.0, 0.0], [0.016; 3], 0.2, 0.0)]),
        ];
        let other_bounds = nodes[1].authored_support_bounds;
        let target = [0.1, 0.0, 0.0];
        assert_eq!(
            spatial_nearest_boundary_contributor(&nodes[0], other_bounds, target),
            1,
            "the original overlap-first key must select the broad representative"
        );

        let mut value = nodes[0].representatives[0].clone();
        value.gaussian.scale_opacity.scale = [0.1; 3];
        value.support_bounds = gaussian_support_bounds(&value.gaussian, 3.0).unwrap();
        let overrides = [SpatialRepresentativeOverride {
            node: 0,
            representative: 0,
            value,
        }];
        assert_eq!(
            spatial_nearest_boundary_contributor_with_overrides_for_test(
                &nodes,
                0,
                other_bounds,
                target,
                &overrides,
            ),
            0,
            "changing the support-overlap bit must change this contributor key"
        );
        assert!(
            !spatial_candidate_preserves_probe_topology(&nodes, &overrides),
            "the candidate guard must reject a changed fixed-grid contributor topology"
        );
    }

    #[test]
    fn spatial_reference_cache_is_bit_exact_with_brute_force_and_fit_decisions() {
        let nodes = vec![
            spatial_fit_exactness_node(-0.35),
            spatial_fit_exactness_node(0.35),
        ];
        assert!(nodes.iter().all(|node| node.representatives.len() == 3));
        let probes = spatial_boundary_probes_for_pair(&nodes, 0, 1);
        assert!(probes.len() >= 3);
        for probe in probes.iter().copied() {
            let reference = spatial_boundary_reference(&nodes, probe).unwrap();
            assert_eq!(
                spatial_boundary_metrics_from_reference(&nodes, probe, &reference, &[]).unwrap(),
                spatial_boundary_metrics_brute_force(&nodes, probe, &[]).unwrap()
            );

            let overrides = [
                SpatialRepresentativeOverride {
                    node: probe.left_node,
                    representative: probe.left_representative,
                    value: spatial_widened_representative(
                        spatial_probe_source(&nodes, probe.left_node, probe.left_representative)
                            .unwrap(),
                        &nodes[probe.left_node].representatives[probe.left_representative],
                        SPATIAL_FIT_TANGENT_FACTORS[2],
                        3.0,
                    )
                    .unwrap()
                    .unwrap(),
                },
                SpatialRepresentativeOverride {
                    node: probe.right_node,
                    representative: probe.right_representative,
                    value: spatial_widened_representative(
                        spatial_probe_source(&nodes, probe.right_node, probe.right_representative)
                            .unwrap(),
                        &nodes[probe.right_node].representatives[probe.right_representative],
                        SPATIAL_FIT_TANGENT_FACTORS[2],
                        3.0,
                    )
                    .unwrap()
                    .unwrap(),
                },
            ];
            assert_eq!(
                spatial_boundary_metrics_from_reference(&nodes, probe, &reference, &overrides,)
                    .unwrap(),
                spatial_boundary_metrics_brute_force(&nodes, probe, &overrides).unwrap()
            );
        }

        let mut cached_nodes = nodes.clone();
        let mut brute_force_nodes = nodes;
        let cached_report =
            fit_spatial_moment_merge_sibling_cohort(&mut cached_nodes, 3.0).unwrap();
        let brute_force_report =
            fit_spatial_moment_merge_sibling_cohort_brute_force(&mut brute_force_nodes, 3.0)
                .unwrap();
        assert_eq!(cached_report, brute_force_report);
        assert!(
            cached_report.accepted_edits >= 2,
            "fixture must exercise at least two sequential accepted edits: {cached_report:?}"
        );
        for (cached, brute_force) in cached_nodes.iter().zip(&brute_force_nodes) {
            assert_eq!(
                cached.spatial_certificate_cap.to_bits(),
                brute_force.spatial_certificate_cap.to_bits()
            );
            assert_eq!(
                cached.spatial_geometric_error_floor.to_bits(),
                brute_force.spatial_geometric_error_floor.to_bits()
            );
            assert_eq!(cached.representatives, brute_force.representatives);
            for (cached, brute_force) in cached
                .representatives
                .iter()
                .zip(&brute_force.representatives)
            {
                assert_eq!(
                    bytemuck::bytes_of(&cached.gaussian),
                    bytemuck::bytes_of(&brute_force.gaussian)
                );
            }
        }
    }

    #[test]
    #[ignore = "manual release-mode spatial fitter microbenchmark"]
    fn spatial_reference_cache_max_eight_node_cohort_microbenchmark() {
        let nodes = (0..8)
            .map(|index| spatial_fit_benchmark_node(-0.35 + index as f32 * 0.1))
            .collect::<Vec<_>>();
        let mut probes = Vec::new();
        for left in 0..nodes.len() {
            for right in left + 1..nodes.len() {
                probes.extend(spatial_boundary_probes_for_pair(&nodes, left, right));
            }
        }
        assert_eq!(probes.len(), 252);
        let failing_probes = probes
            .iter()
            .copied()
            .filter(|probe| {
                spatial_boundary_metrics_brute_force(&nodes, *probe, &[])
                    .unwrap()
                    .relative_boundary_error
                    > SPATIAL_FIT_MAX_RELATIVE_BOUNDARY_ERROR
            })
            .count();
        assert!(failing_probes >= probes.len() / 2);

        let mut brute_force_nodes = nodes.clone();
        let brute_force_started = std::time::Instant::now();
        let brute_force_report =
            fit_spatial_moment_merge_sibling_cohort_brute_force(&mut brute_force_nodes, 3.0)
                .unwrap();
        let brute_force_elapsed = brute_force_started.elapsed();

        let mut cached_nodes = nodes;
        let cached_started = std::time::Instant::now();
        let cached_report =
            fit_spatial_moment_merge_sibling_cohort(&mut cached_nodes, 3.0).unwrap();
        let cached_elapsed = cached_started.elapsed();
        assert_eq!(cached_report, brute_force_report);
        assert_eq!(cached_nodes.len(), brute_force_nodes.len());
        for (cached, brute_force) in cached_nodes.iter().zip(&brute_force_nodes) {
            assert_eq!(cached.representatives, brute_force.representatives);
        }
        eprintln!(
            "spatial cache benchmark: probes={}; failing={}; brute_force={:?}; cached={:?}; speedup={:.2}x",
            probes.len(),
            failing_probes,
            brute_force_elapsed,
            cached_elapsed,
            brute_force_elapsed.as_secs_f64() / cached_elapsed.as_secs_f64()
        );
    }

    #[test]
    fn single_projected_scale_improvement_cannot_mask_another_zoom_regression() {
        let guarded_strip = |x: f32| {
            let mut source = feasible_spatial_strip(x);
            for (offset_x, y) in [(-4.0_f32, -6.0_f32), (4.0, 6.0)] {
                source.push(gaussian(
                    [x + offset_x, y, 1.0],
                    [0.2, 0.2, 0.01],
                    0.000_001,
                    0.0,
                ));
            }
            source
        };
        let nodes = vec![
            spatial_test_node(guarded_strip(-0.35)),
            spatial_test_node(guarded_strip(0.35)),
        ];
        let probe = spatial_boundary_probes_for_pair(&nodes, 0, 1)[0];
        let overrides = [
            SpatialRepresentativeOverride {
                node: 0,
                representative: 0,
                value: spatial_widened_representative(
                    spatial_probe_source(&nodes, 0, 0).unwrap(),
                    &nodes[0].representatives[0],
                    1.5,
                    3.0,
                )
                .unwrap()
                .unwrap(),
            },
            SpatialRepresentativeOverride {
                node: 1,
                representative: 0,
                value: spatial_widened_representative(
                    spatial_probe_source(&nodes, 1, 0).unwrap(),
                    &nodes[1].representatives[0],
                    1.5,
                    3.0,
                )
                .unwrap()
                .unwrap(),
            },
        ];
        let pair_envelope = nodes[0]
            .authored_support_bounds
            .union(nodes[1].authored_support_bounds);
        assert!(overrides.iter().all(|candidate| {
            oriented_support_inside(&candidate.value.gaussian, 3.0, pair_envelope).unwrap()
        }));

        let baseline_at_one = spatial_fixed_scale_metrics(&nodes, probe, &[], 1.0);
        let candidate_at_one = spatial_fixed_scale_metrics(&nodes, probe, &overrides, 1.0);
        assert!(candidate_at_one.0 < baseline_at_one.0);
        assert!(candidate_at_one.1 < baseline_at_one.1);
        let regresses_elsewhere = [0.25_f64, 0.5, 2.0, 4.0].into_iter().any(|scale| {
            let baseline = spatial_fixed_scale_metrics(&nodes, probe, &[], scale);
            let candidate = spatial_fixed_scale_metrics(&nodes, probe, &overrides, scale);
            !float_no_worse(candidate.0, baseline.0) || !float_no_worse(candidate.1, baseline.1)
        });
        assert!(regresses_elsewhere);

        let baseline_ladder = spatial_boundary_metrics(&nodes, probe, &[]).unwrap();
        let candidate_ladder = spatial_boundary_metrics(&nodes, probe, &overrides).unwrap();
        assert!(!spatial_boundary_metrics_no_worse(
            candidate_ladder,
            baseline_ladder
        ));
    }

    #[test]
    fn elongated_multi_representative_seam_checks_its_worst_segment() {
        let end_partition = |x: f32, center_y: f32| {
            [-1.5_f32, -0.5, 0.5, 1.5]
                .into_iter()
                .map(|offset| gaussian([x, center_y + offset, 1.0], [0.2, 0.2, 0.01], 0.2, 0.0))
                .collect::<Vec<_>>()
        };
        let node = |x: f32| {
            spatial_test_node_with_partitions(vec![
                end_partition(x, -6.0),
                vec![gaussian([x, 0.0, 1.0], [0.2, 0.2, 0.01], 0.2, 0.0)],
                end_partition(x, 6.0),
            ])
        };
        let mut nodes = vec![node(-0.35), node(0.35)];
        let probes = spatial_boundary_probes_for_pair(&nodes, 0, 1);
        assert!(probes.len() >= 3);
        for representative in 0..3 {
            assert!(probes.iter().any(|probe| {
                probe.left_representative == representative
                    && probe.right_representative == representative
            }));
        }
        let before = probes
            .iter()
            .copied()
            .map(|probe| spatial_boundary_metrics(&nodes, probe, &[]).unwrap())
            .collect::<Vec<_>>();
        let central = probes
            .iter()
            .position(|probe| probe.left_representative == 1 && probe.right_representative == 1)
            .unwrap();
        assert!(before[central].relative_boundary_error <= 1e-6);
        let worst = before
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.relative_boundary_error
                    .total_cmp(&right.relative_boundary_error)
            })
            .map(|(index, _)| index)
            .unwrap();
        assert_ne!(worst, central);
        assert!(before[worst].relative_boundary_error > 0.1);

        fit_spatial_moment_merge_sibling_cohort(&mut nodes, 3.0).unwrap();
        let after = spatial_boundary_metrics(&nodes, probes[worst], &[]).unwrap();
        let worst_improved = after.relative_boundary_error < before[worst].relative_boundary_error;
        let forced_refinement = nodes
            .iter()
            .all(|node| node.spatial_geometric_error_floor > 0.0);
        assert!(worst_improved || forced_refinement);
    }

    #[test]
    fn disconnected_sibling_cannot_expand_a_touching_pair_envelope() {
        let pair = || {
            vec![
                spatial_test_node(feasible_spatial_strip(-0.35)),
                spatial_test_node(feasible_spatial_strip(0.35)),
            ]
        };
        let mut isolated_pair = pair();
        let isolated_report =
            fit_spatial_moment_merge_sibling_cohort(&mut isolated_pair, 3.0).unwrap();
        assert!(isolated_report.accepted_edits > 0);

        let mut cohort_with_gap = pair();
        cohort_with_gap.push(spatial_test_node(vec![gaussian(
            [20.0, 0.0, 1.0],
            [0.2, 0.2, 0.01],
            0.2,
            0.0,
        )]));
        let third_before = cohort_with_gap[2].representatives[0].gaussian;
        let report = fit_spatial_moment_merge_sibling_cohort(&mut cohort_with_gap, 3.0).unwrap();
        assert_eq!(report, isolated_report);
        for node_index in 0..2 {
            assert_eq!(
                cohort_with_gap[node_index].representatives[0].gaussian,
                isolated_pair[node_index].representatives[0].gaussian
            );
        }
        assert_eq!(cohort_with_gap[2].representatives[0].gaussian, third_before);

        let pair_envelope = cohort_with_gap[0]
            .authored_support_bounds
            .union(cohort_with_gap[1].authored_support_bounds);
        assert!(pair_envelope.max[0] < cohort_with_gap[2].authored_support_bounds.min[0]);
        for node in &cohort_with_gap[..2] {
            for representative in &node.representatives {
                assert!(
                    oriented_support_inside(&representative.gaussian, 3.0, pair_envelope).unwrap()
                );
            }
        }
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

    #[cfg(feature = "sort_rayon")]
    #[test]
    fn parallel_builder_is_byte_identical_across_worker_counts() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 8,
            leaf_capacity: 128,
            support_sigma: 3.0,
        };
        // Exercise many independent leaves, deepest pairing plans, and parent
        // rungs so this covers every parallel builder phase rather than only
        // the Morton sort.
        let source = fixture(4_097);
        let build_with_threads = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| build_planar_3d_lod(&source, settings).unwrap())
        };

        let single_threaded = build_with_threads(1);
        let parallel = build_with_threads(4);

        assert_eq!(single_threaded.manifest, parallel.manifest);
        assert_eq!(single_threaded.pages.len(), parallel.pages.len());
        for (single_page, parallel_page) in single_threaded.pages.iter().zip(&parallel.pages) {
            assert_eq!(
                crate::io::lod::encode_page(single_page).unwrap(),
                crate::io::lod::encode_page(parallel_page).unwrap()
            );
        }
    }

    #[test]
    fn owned_builder_honors_cancellation_before_morton_sort() {
        let source = fixture(257).iter().collect();
        let result = build_planar_3d_lod_owned_cancelable(
            source,
            GaussianLodBuildSettings::default(),
            &|| true,
        )
        .unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn owned_builder_honors_cancellation_after_parallel_work_begins() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        let source = fixture(4_097).iter().collect();
        let polls = AtomicUsize::new(0);
        let cancel_after = 384;
        let result = build_planar_3d_lod_owned_cancelable(
            source,
            GaussianLodBuildSettings {
                branching_factor: 8,
                leaf_capacity: 128,
                support_sigma: 3.0,
            },
            &|| polls.fetch_add(1, AtomicOrdering::Relaxed) >= cancel_after,
        )
        .unwrap();

        assert!(result.is_none());
        assert!(polls.load(AtomicOrdering::Relaxed) > cancel_after);
    }

    #[test]
    fn large_balanced_reduction_polls_cancellation_in_bounded_chunks() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        let source = fixture(4_097).iter().collect::<Vec<_>>();
        let polls = AtomicUsize::new(0);
        let cancellation = LodBuildCancellation {
            is_canceled: &|| polls.fetch_add(1, AtomicOrdering::Relaxed) >= 2,
        };
        let result =
            balanced_progressive_moment_merge_representatives(&source, 1, 3.0, cancellation);

        assert!(matches!(result, Err(CancelableLodBuildError::Canceled)));
        assert!(polls.load(AtomicOrdering::Relaxed) <= 4);
    }

    #[test]
    fn owned_builder_matches_planar_output_and_preserves_canonical_original_order() {
        let settings = GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 16,
            support_sigma: 3.0,
        };
        let mut source = fixture(37).iter().collect::<Vec<_>>();
        source.rotate_left(11);
        source[0].position_visibility.position[0] = -0.0;
        *source[3]
            .spherical_harmonic
            .coefficients
            .last_mut()
            .expect("every SH profile has a DC coefficient") = -0.0;
        source[9].rotation.rotation[2] = -0.0;
        source[17].scale_opacity.opacity = -0.0;
        let expected_fallback = source
            .iter()
            .copied()
            .map(canonicalize_gaussian_zeros)
            .collect::<Vec<_>>();
        let planar = PlanarGaussian3d::from(source.clone());

        let expected_lod = build_planar_3d_lod(&planar, settings).unwrap();
        let (owned_lod, fallback) = build_planar_3d_lod_owned(source, settings).unwrap();

        assert_eq!(owned_lod, expected_lod);
        assert_eq!(fallback, expected_fallback);
        assert_eq!(fallback[0].position_visibility.position[0].to_bits(), 0);
        assert_eq!(
            fallback[3]
                .spherical_harmonic
                .coefficients
                .last()
                .unwrap()
                .to_bits(),
            0
        );
        assert_eq!(fallback[9].rotation.rotation[2].to_bits(), 0);
        assert_eq!(fallback[17].scale_opacity.opacity.to_bits(), 0);
        assert_eq!(
            owned_lod.manifest.build.builder_abi_version,
            PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION
        );
    }

    #[test]
    fn compact_morton_index_sort_matches_full_record_reference_and_fingerprint() {
        let mut source = fixture(LOD_MORTON_SORT_RUN_LEN + 257)
            .iter()
            .collect::<Vec<_>>();
        // Force Morton collisions while leaving the full Gaussian comparator
        // responsible for a deterministic order inside each cell.
        for (index, gaussian) in source.iter_mut().enumerate() {
            gaussian.position_visibility.position =
                [(index % 3) as f32, ((index / 3) % 2) as f32, 0.0];
            *gaussian = canonicalize_gaussian_zeros(*gaussian);
        }
        let bounds = source_center_bounds(&source, no_cancellation()).unwrap();
        let mut compact = source
            .iter()
            .enumerate()
            .map(|(source_index, gaussian)| MortonSourceIndex {
                morton: canonical_lod_morton_code(gaussian.position_visibility.position, bounds),
                source_index,
            })
            .collect::<Vec<_>>();
        sort_morton_source_indices(&mut compact, &source, no_cancellation()).unwrap();

        let mut full_record_reference = source
            .iter()
            .copied()
            .map(|gaussian| {
                (
                    canonical_lod_morton_code(gaussian.position_visibility.position, bounds),
                    gaussian,
                )
            })
            .collect::<Vec<_>>();
        full_record_reference.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| compare_gaussians(&left.1, &right.1))
        });

        let compact_records = compact
            .iter()
            .map(|entry| (entry.morton, source[entry.source_index]))
            .collect::<Vec<_>>();
        assert_eq!(compact_records, full_record_reference);

        let mut reference_hash = StableHasher::new();
        reference_hash.write(&(full_record_reference.len() as u64).to_le_bytes());
        for (morton, gaussian) in &full_record_reference {
            reference_hash.write(&morton.to_le_bytes());
            reference_hash.write(&stable_gaussian_hash(gaussian).to_le_bytes());
        }
        assert_eq!(
            source_fingerprint(&compact, &source, no_cancellation()).unwrap(),
            reference_hash.finish()
        );
        assert!(size_of::<MortonSourceIndex>() < size_of::<Gaussian3d>());

        let host = include_str!("planar_3d_lod.rs");
        assert!(host.contains(".par_chunks_mut(LOD_MORTON_SORT_RUN_LEN)"));
        assert!(host.contains(".chunks_mut(LOD_MORTON_SORT_RUN_LEN)"));
        assert!(!host.contains(concat!("struct Keyed", "Gaussian")));
    }

    #[test]
    fn morton_merge_polls_cancellation_inside_a_large_sort() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        let source = fixture(LOD_MORTON_SORT_RUN_LEN * 2 + 1)
            .iter()
            .collect::<Vec<_>>();
        let bounds = source_center_bounds(&source, no_cancellation()).unwrap();
        let mut entries = source
            .iter()
            .enumerate()
            .rev()
            .map(|(source_index, gaussian)| MortonSourceIndex {
                morton: canonical_lod_morton_code(gaussian.position_visibility.position, bounds),
                source_index,
            })
            .collect::<Vec<_>>();
        // Initial/run checks and scratch initialization consume about 520
        // polls. Cancel just inside the first merge pass, whose inner loop
        // checks at the tighter 256-record cadence.
        let polls = AtomicUsize::new(0);
        let cancel_after = 530;
        let cancellation = LodBuildCancellation {
            is_canceled: &|| polls.fetch_add(1, AtomicOrdering::Relaxed) >= cancel_after,
        };

        let result = sort_morton_source_indices(&mut entries, &source, cancellation);

        assert!(matches!(result, Err(CancelableLodBuildError::Canceled)));
        assert!(polls.load(AtomicOrdering::Relaxed) <= cancel_after + 16);
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
            no_cancellation(),
        )
        .unwrap();
        let first = progressive_moment_merge_representatives(
            &source,
            representative_count,
            support_sigma,
            no_cancellation(),
        )
        .unwrap();
        let second = progressive_moment_merge_representatives(
            &source,
            representative_count,
            support_sigma,
            no_cancellation(),
        )
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
        let balanced_envelope =
            progressive_selection_envelope(&balanced, no_cancellation()).unwrap();
        let emitted_envelope =
            progressive_selection_envelope(&first.representatives, no_cancellation()).unwrap();
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
    fn risk_aware_host_bound_covers_stale_candidate_heap_growth() {
        let source_count = 8 * 1024_usize;
        let bound = progressive_risk_aware_host_bytes_upper_bound(source_count).unwrap();
        let candidate_floor = u64::try_from(source_count * 4 + 1).unwrap()
            * size_of::<ProgressiveAgglomerationCandidate>() as u64;
        let other_floor = u64::try_from(source_count + 1).unwrap()
            * (size_of::<Gaussian3d>()
                + size_of::<ProgressiveAgglomerationCluster>()
                + size_of::<MomentMergeResult>()) as u64;
        assert!(bound >= candidate_floor + other_floor);
    }

    #[test]
    fn progressive_rung_retains_balanced_fallback_above_risk_aware_bound() {
        let source = (0..34)
            .map(|index| gaussian([index as f32, 0.0, 0.0], [0.01; 3], 0.25, 0.0))
            .collect::<Vec<_>>();
        let expected =
            balanced_progressive_moment_merge_representatives(&source, 2, 3.0, no_cancellation())
                .unwrap();
        let actual =
            progressive_moment_merge_representatives(&source, 2, 3.0, no_cancellation()).unwrap();

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

        let mut stale_progressive_builder = output.manifest.clone();
        stale_progressive_builder.build.builder_abi_version =
            PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION - 1;
        assert!(matches!(
            stale_progressive_builder.validate(),
            Err(LodValidationError::InvalidBuildVersion)
        ));

        let mut external_v2 = output.manifest.clone();
        external_v2.build.builder_abi_version = EXTERNAL_CPU_MOMENT_MERGE_BUILDER_ABI_VERSION;
        external_v2.build.reducer_version = EXTERNAL_MOMENT_MERGE_VERSION;
        external_v2.build.config_fingerprint =
            lod_config_fingerprint_for_reducer(settings, None, EXTERNAL_MOMENT_MERGE_VERSION);
        assert!(external_v2.validate().is_ok());

        let mut legacy_gpu_external_v2 = external_v2.clone();
        legacy_gpu_external_v2.build.builder_abi_version =
            EXTERNAL_GPU_MOMENT_MERGE_BUILDER_ABI_VERSION;
        assert!(legacy_gpu_external_v2.validate().is_ok());

        let mut external_with_progressive_reducer = external_v2;
        external_with_progressive_reducer.build.reducer_version = MOMENT_MERGE_VERSION;
        external_with_progressive_reducer.build.config_fingerprint =
            lod_config_fingerprint(settings, None);
        assert!(matches!(
            external_with_progressive_reducer.validate(),
            Err(LodValidationError::InvalidBuildVersion)
        ));

        let mut external_progressive = output.manifest.clone();
        external_progressive.build.builder_abi_version =
            EXTERNAL_PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION;
        external_progressive.build.reducer_version = MOMENT_MERGE_VERSION;
        external_progressive.build.config_fingerprint = lod_config_fingerprint(settings, None);
        assert!(external_progressive.validate().is_ok());
        assert!(
            external_progressive
                .build
                .has_bounded_refinement_amplification()
        );

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
