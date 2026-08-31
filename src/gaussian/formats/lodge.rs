//! Portable semantic contract for an external LODGE hierarchy.
//!
//! A `.gslodge` file is a companion to, rather than a replacement for, the
//! ordinary `.gsplatlod` manifest and Gaussian pages. Level zero maps the
//! original leaf records from that base manifest; coarser LODGE levels use the
//! same independently decodable page format. Every referenced object also has
//! a SHA-256 digest, allowing a trusted sidecar to authenticate its dependency
//! closure without changing the existing LoD package ABI.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use serde::{Deserialize, Serialize};

use super::{
    planar_3d_chunked::{
        LOD_PAGE_SCHEMA_VERSION, LodIndexRange, LodPageDescriptor, LodPageId, LodPageKind,
    },
    planar_3d_lod::GaussianLodManifest,
};

pub const LODGE_MANIFEST_MAGIC: [u8; 8] = *b"BGSLOG1\0";
pub const LODGE_MANIFEST_VERSION: u16 = 1;
pub const LODGE_MEMBERSHIP_SCHEMA_VERSION: u16 = 1;

pub const LODGE_FEATURE_STABLE_GAUSSIAN_IDS: u64 = 1 << 0;
pub const LODGE_FEATURE_DEPTH_FILTER_METADATA: u64 = 1 << 1;
pub const LODGE_FEATURE_CAMERA_CLUSTERS: u64 = 1 << 2;
pub const LODGE_FEATURE_AUTHENTICATED_DEPENDENCIES: u64 = 1 << 3;
pub const LODGE_FEATURE_DELTA_ULEB128_MEMBERSHIPS: u64 = 1 << 4;
pub const LODGE_REQUIRED_FEATURES: u64 = LODGE_FEATURE_STABLE_GAUSSIAN_IDS
    | LODGE_FEATURE_DEPTH_FILTER_METADATA
    | LODGE_FEATURE_CAMERA_CLUSTERS
    | LODGE_FEATURE_AUTHENTICATED_DEPENDENCIES
    | LODGE_FEATURE_DELTA_ULEB128_MEMBERSHIPS;

/// Stable identifier of one Gaussian record across all LODGE levels.
///
/// IDs are dense, start at one, and never depend on camera-cluster membership
/// order. Zero is reserved as an invalid/sentinel value.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct LodgeGaussianId(pub u64);

impl LodgeGaussianId {
    pub const INVALID: Self = Self(0);

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }
}

/// Discrete LODGE level. Level zero is always the unfiltered original level.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct LodgeLevelId(pub u16);

/// Stable camera-cluster identifier. Zero is reserved.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct LodgeClusterId(pub u32);

impl LodgeClusterId {
    pub const INVALID: Self = Self(0);

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }
}

/// Page-local address obtained from a stable Gaussian ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodgePageLocator {
    pub page: LodPageId,
    pub offset: u32,
}

/// A dense stable-ID run backed by a contiguous range of an ordinary page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodgeRecordRun {
    pub first_id: LodgeGaussianId,
    pub count: u32,
    pub page: LodPageId,
    pub page_offset: u32,
}

impl LodgeRecordRun {
    #[inline]
    pub fn stable_end(self) -> Option<u64> {
        self.first_id.0.checked_add(u64::from(self.count))
    }

    #[inline]
    pub fn page_end(self) -> Option<u32> {
        self.page_offset.checked_add(self.count)
    }

    #[inline]
    pub fn locate(self, id: LodgeGaussianId) -> Option<LodgePageLocator> {
        let delta = id.0.checked_sub(self.first_id.0)?;
        if delta >= u64::from(self.count) {
            return None;
        }
        Some(LodgePageLocator {
            page: self.page,
            offset: self.page_offset.checked_add(u32::try_from(delta).ok()?)?,
        })
    }
}

/// Reproducible metadata for the offline filter which produced one level.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LodgeLevelFilter {
    /// Exact original Gaussians from the base `.gsplatlod` source leaves.
    Original,
    /// Depth-aware 3D smoothing, importance pruning, and optional fine-tuning.
    DepthAware3dV1 {
        reference_depth: f32,
        reference_focal_length_px: f32,
        smoothing_scale: f32,
        importance_threshold: f32,
        fine_tune_steps: u32,
    },
}

/// One discrete distance level. Its upper bound is the next level's
/// `distance_min`; the final level has an implicit infinite upper bound.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LodgeLevelDescriptor {
    pub id: LodgeLevelId,
    pub distance_min: f32,
    /// Range into [`GaussianLodgeManifest::record_runs`].
    pub records: LodIndexRange,
    pub filter: LodgeLevelFilter,
}

/// Immutable object identity. SHA-256 authenticates exact encoded bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodgeAuthenticatedObject {
    pub uri: String,
    pub encoded_len: u64,
    pub sha256: [u8; 32],
}

/// SHA-256 for the exact encoded page bytes named by a page descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodgePageAuthentication {
    pub page: LodPageId,
    pub encoded_sha256: [u8; 32],
}

/// Encoding of a cluster's sorted stable-ID membership stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LodgeMembershipEncoding {
    /// Strictly increasing IDs encoded as unsigned LEB128 deltas from zero.
    DeltaUleb128StableIdsV1,
}

/// Independently authenticated range for one compressed cluster membership.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodgeMembershipEntry {
    pub cluster: LodgeClusterId,
    /// `(start, length)` in [`LodgeMembershipIndexDescriptor::object`].
    pub byte_range: (u64, u64),
    pub member_count: u64,
    pub first_id: LodgeGaussianId,
    pub last_id: LodgeGaussianId,
    pub encoded_sha256: [u8; 32],
}

/// External compressed membership object and its authenticated range index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodgeMembershipIndexDescriptor {
    pub schema_version: u16,
    pub encoding: LodgeMembershipEncoding,
    pub object: LodgeAuthenticatedObject,
    /// `(start, length)` of the immutable range index. Membership streams
    /// follow it contiguously in cluster order.
    pub index_byte_range: (u64, u64),
    pub index_sha256: [u8; 32],
    pub entries: Vec<LodgeMembershipEntry>,
}

/// Camera-position cluster and its flattened neighbor/membership references.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LodgeCameraCluster {
    pub id: LodgeClusterId,
    pub center: [f32; 3],
    pub radius: f32,
    /// Range into [`GaussianLodgeManifest::neighbors`].
    pub neighbors: LodIndexRange,
    /// Index into [`LodgeMembershipIndexDescriptor::entries`].
    pub membership_entry: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GaussianLodgeManifestHeader {
    pub magic: [u8; 8],
    pub manifest_version: u16,
    pub page_schema_version: u16,
    pub required_features: u64,
    pub base_page_count: u32,
    pub extra_page_count: u32,
    pub level_count: u32,
    pub cluster_count: u32,
    pub record_run_count: u32,
    pub neighbor_count: u32,
    pub stable_gaussian_count: u64,
    pub total_membership_ids: u64,
}

/// Portable LODGE sidecar manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GaussianLodgeManifest {
    pub header: GaussianLodgeManifestHeader,
    /// Exact encoded identity of the companion `.gsplatlod` manifest.
    pub base_manifest: LodgeAuthenticatedObject,
    /// Coarser levels retain the ordinary Gaussian page representation.
    pub extra_pages: Vec<LodPageDescriptor>,
    /// Sorted, unique authentication records covering base and extra pages.
    pub page_authentication: Vec<LodgePageAuthentication>,
    pub levels: Vec<LodgeLevelDescriptor>,
    /// Globally sorted dense-ID runs, partitioned by `levels[*].records`.
    pub record_runs: Vec<LodgeRecordRun>,
    pub clusters: Vec<LodgeCameraCluster>,
    /// Flattened, per-cluster sorted neighbor IDs.
    pub neighbors: Vec<LodgeClusterId>,
    pub membership_index: LodgeMembershipIndexDescriptor,
}

impl GaussianLodgeManifest {
    /// Zero-allocation run slice for one level.
    pub fn record_runs_for_level(&self, level: LodgeLevelId) -> Option<&[LodgeRecordRun]> {
        let descriptor = self.levels.get(usize::from(level.0))?;
        if descriptor.id != level {
            return None;
        }
        index_slice(&self.record_runs, descriptor.records)
    }

    /// Zero-allocation neighbor slice for one camera cluster.
    pub fn neighbors_for_cluster(&self, cluster: LodgeClusterId) -> Option<&[LodgeClusterId]> {
        let index = self
            .clusters
            .binary_search_by_key(&cluster, |entry| entry.id)
            .ok()?;
        let descriptor = &self.clusters[index];
        index_slice(&self.neighbors, descriptor.neighbors)
    }

    pub fn membership_for_cluster(&self, cluster: LodgeClusterId) -> Option<&LodgeMembershipEntry> {
        let index = self
            .clusters
            .binary_search_by_key(&cluster, |entry| entry.id)
            .ok()?;
        let descriptor = &self.clusters[index];
        let entry = self
            .membership_index
            .entries
            .get(descriptor.membership_entry as usize)?;
        (entry.cluster == cluster).then_some(entry)
    }

    pub fn authentication_for_page(&self, page: LodPageId) -> Option<&LodgePageAuthentication> {
        self.page_authentication
            .binary_search_by_key(&page, |entry| entry.page)
            .ok()
            .map(|index| &self.page_authentication[index])
    }

    /// Binary-searches the global run table without loading a Gaussian page.
    pub fn locate_gaussian(&self, id: LodgeGaussianId) -> Option<LodgePageLocator> {
        let index = self
            .record_runs
            .partition_point(|run| run.first_id.0 <= id.0)
            .checked_sub(1)?;
        self.record_runs[index].locate(id)
    }

    /// Validates the self-contained sidecar contract. Dependency hashes are
    /// checked by the sidecar codec and object/page loaders when bytes arrive.
    pub fn validate(&self) -> Result<(), LodgeValidationError> {
        if self.header.magic != LODGE_MANIFEST_MAGIC {
            return Err(LodgeValidationError::InvalidMagic(self.header.magic));
        }
        if self.header.manifest_version != LODGE_MANIFEST_VERSION {
            return Err(LodgeValidationError::UnsupportedManifestVersion(
                self.header.manifest_version,
            ));
        }
        if self.header.page_schema_version != LOD_PAGE_SCHEMA_VERSION {
            return Err(LodgeValidationError::UnsupportedPageSchema(
                self.header.page_schema_version,
            ));
        }
        let unsupported = self.header.required_features & !LODGE_REQUIRED_FEATURES;
        if unsupported != 0 {
            return Err(LodgeValidationError::UnsupportedRequiredFeatures(
                unsupported,
            ));
        }
        let missing = LODGE_REQUIRED_FEATURES & !self.header.required_features;
        if missing != 0 {
            return Err(LodgeValidationError::MissingRequiredFeatures(missing));
        }

        check_count(
            "extra pages",
            self.header.extra_page_count,
            self.extra_pages.len(),
        )?;
        check_count("levels", self.header.level_count, self.levels.len())?;
        check_count("clusters", self.header.cluster_count, self.clusters.len())?;
        check_count(
            "record runs",
            self.header.record_run_count,
            self.record_runs.len(),
        )?;
        check_count(
            "neighbors",
            self.header.neighbor_count,
            self.neighbors.len(),
        )?;
        let expected_auth = self
            .header
            .base_page_count
            .checked_add(self.header.extra_page_count)
            .ok_or(LodgeValidationError::CountOverflow("authenticated pages"))?;
        check_count(
            "authenticated pages",
            expected_auth,
            self.page_authentication.len(),
        )?;
        if self.header.base_page_count == 0 {
            return Err(LodgeValidationError::Invalid("base page count is zero"));
        }

        validate_object(&self.base_manifest, "base manifest")?;
        if !self.base_manifest.uri.ends_with(".gsplatlod") {
            return Err(LodgeValidationError::Invalid(
                "base manifest URI must end in .gsplatlod",
            ));
        }

        let mut previous_page = LodPageId::INVALID;
        for (index, auth) in self.page_authentication.iter().enumerate() {
            if !auth.page.is_valid() || auth.page <= previous_page {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "page authentication order",
                    index,
                });
            }
            require_hash(auth.encoded_sha256, "page SHA-256")?;
            previous_page = auth.page;
        }

        let mut previous_extra = LodPageId::INVALID;
        for (index, page) in self.extra_pages.iter().enumerate() {
            page.validate().map_err(|error| {
                LodgeValidationError::InvalidOwned(format!("extra page {index}: {error}"))
            })?;
            if page.id <= previous_extra {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "extra page order",
                    index,
                });
            }
            if page.kind != LodPageKind::Representatives {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "extra page kind",
                    index,
                });
            }
            let storage = page
                .storage
                .as_ref()
                .ok_or(LodgeValidationError::InvalidIndexed {
                    field: "extra page storage",
                    index,
                })?;
            validate_relative_uri(&storage.uri)?;
            if self
                .page_authentication
                .binary_search_by_key(&page.id, |auth| auth.page)
                .is_err()
            {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "extra page authentication",
                    index,
                });
            }
            previous_extra = page.id;
        }

        self.validate_levels_and_runs()?;
        self.validate_clusters()?;
        self.validate_memberships()?;
        Ok(())
    }

    /// Adds the semantic checks which require the decoded base hierarchy. The
    /// caller must independently compare the base manifest's encoded length
    /// and SHA-256 with [`Self::base_manifest`].
    pub fn validate_against_base(
        &self,
        base: &GaussianLodManifest,
    ) -> Result<(), LodgeValidationError> {
        self.validate()?;
        base.validate().map_err(|error| {
            LodgeValidationError::InvalidOwned(format!("invalid base manifest: {error}"))
        })?;
        check_count("base pages", self.header.base_page_count, base.pages.len())?;
        let finest = self
            .levels
            .first()
            .ok_or(LodgeValidationError::Invalid("missing finest level"))?;
        let finest_runs = index_slice(&self.record_runs, finest.records)
            .ok_or(LodgeValidationError::Invalid("invalid finest run range"))?;
        let finest_count = finest_runs
            .iter()
            .try_fold(0_u64, |total, run| total.checked_add(u64::from(run.count)));
        if finest_count != Some(base.header.source_gaussian_count) {
            return Err(LodgeValidationError::Invalid(
                "finest level does not contain every original Gaussian",
            ));
        }

        let mut page_counts = BTreeMap::new();
        for page in &base.pages {
            if page_counts.insert(page.id, page.gaussian_count).is_some() {
                return Err(LodgeValidationError::Invalid("duplicate base page ID"));
            }
        }
        for page in &self.extra_pages {
            if page_counts.insert(page.id, page.gaussian_count).is_some() {
                return Err(LodgeValidationError::Invalid(
                    "extra page ID collides with a base page",
                ));
            }
        }
        let authenticated = self
            .page_authentication
            .iter()
            .map(|entry| entry.page)
            .collect::<Vec<_>>();
        if !page_counts
            .keys()
            .copied()
            .eq(authenticated.iter().copied())
        {
            return Err(LodgeValidationError::Invalid(
                "page authentication does not exactly cover base and extra pages",
            ));
        }

        for (level_index, level) in self.levels.iter().enumerate() {
            let runs = index_slice(&self.record_runs, level.records)
                .ok_or(LodgeValidationError::Invalid("invalid level run range"))?;
            for run in runs {
                let page_count =
                    page_counts
                        .get(&run.page)
                        .ok_or(LodgeValidationError::Invalid(
                            "record run references an unknown page",
                        ))?;
                if run.page_end().is_none_or(|end| end > *page_count) {
                    return Err(LodgeValidationError::Invalid("record run exceeds its page"));
                }
                let is_extra = self
                    .extra_pages
                    .binary_search_by_key(&run.page, |page| page.id)
                    .is_ok();
                if (level_index == 0 && is_extra) || (level_index != 0 && !is_extra) {
                    return Err(LodgeValidationError::Invalid(
                        "finest/coarse level references the wrong page set",
                    ));
                }
            }
        }

        self.validate_finest_mapping(base, finest_runs)?;
        self.validate_extra_page_coverage()?;
        Ok(())
    }

    fn validate_levels_and_runs(&self) -> Result<(), LodgeValidationError> {
        if self.levels.len() < 2 {
            return Err(LodgeValidationError::Invalid(
                "LODGE requires an original and at least one coarser level",
            ));
        }
        let mut expected_run_start = 0_u32;
        let mut expected_id = 1_u64;
        let mut previous_distance = None;
        for (index, level) in self.levels.iter().enumerate() {
            if usize::from(level.id.0) != index {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "level ID",
                    index,
                });
            }
            if !level.distance_min.is_finite() || level.distance_min < 0.0 {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "level distance",
                    index,
                });
            }
            if index == 0 {
                if level.distance_min.to_bits() != 0.0_f32.to_bits()
                    || level.filter != LodgeLevelFilter::Original
                {
                    return Err(LodgeValidationError::Invalid(
                        "level zero must be the original level at distance zero",
                    ));
                }
            } else {
                if previous_distance.is_none_or(|previous| level.distance_min <= previous) {
                    return Err(LodgeValidationError::InvalidIndexed {
                        field: "level distance order",
                        index,
                    });
                }
                match level.filter {
                    LodgeLevelFilter::DepthAware3dV1 {
                        reference_depth,
                        reference_focal_length_px,
                        smoothing_scale,
                        importance_threshold,
                        ..
                    } if reference_depth.to_bits() == level.distance_min.to_bits()
                        && reference_focal_length_px.is_finite()
                        && reference_focal_length_px > 0.0
                        && smoothing_scale.is_finite()
                        && smoothing_scale >= 0.0
                        && importance_threshold.is_finite()
                        && (0.0..=1.0).contains(&importance_threshold) => {}
                    _ => {
                        return Err(LodgeValidationError::InvalidIndexed {
                            field: "level filter metadata",
                            index,
                        });
                    }
                }
            }
            previous_distance = Some(level.distance_min);
            let end = level
                .records
                .end()
                .ok_or(LodgeValidationError::CountOverflow("level record range"))?;
            if level.records.start != expected_run_start
                || level.records.count == 0
                || end as usize > self.record_runs.len()
            {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "level record range",
                    index,
                });
            }
            for (run_offset, run) in self.record_runs[level.records.start as usize..end as usize]
                .iter()
                .enumerate()
            {
                let run_index = level.records.start as usize + run_offset;
                if run.first_id.0 != expected_id
                    || run.count == 0
                    || !run.page.is_valid()
                    || run.page_end().is_none()
                {
                    return Err(LodgeValidationError::InvalidIndexed {
                        field: "record run",
                        index: run_index,
                    });
                }
                expected_id = run
                    .stable_end()
                    .ok_or(LodgeValidationError::CountOverflow("stable Gaussian IDs"))?;
            }
            expected_run_start = end;
        }
        if expected_run_start as usize != self.record_runs.len()
            || expected_id.checked_sub(1) != Some(self.header.stable_gaussian_count)
        {
            return Err(LodgeValidationError::Invalid(
                "levels do not exactly cover the stable Gaussian catalog",
            ));
        }
        Ok(())
    }

    fn validate_clusters(&self) -> Result<(), LodgeValidationError> {
        if self.clusters.len() < 2 {
            return Err(LodgeValidationError::Invalid(
                "LODGE requires at least two camera clusters for pair blending",
            ));
        }
        let mut expected_neighbor_start = 0_u32;
        let mut previous_cluster = LodgeClusterId::INVALID;
        let mut centers = BTreeSet::new();
        for (index, cluster) in self.clusters.iter().enumerate() {
            if !cluster.id.is_valid() || cluster.id <= previous_cluster {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "cluster ID order",
                    index,
                });
            }
            if cluster.center.iter().any(|value| !value.is_finite())
                || !cluster.radius.is_finite()
                || cluster.radius < 0.0
            {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "cluster sphere",
                    index,
                });
            }
            let center_key = cluster.center.map(|value| {
                if value == 0.0 {
                    0.0_f32.to_bits()
                } else {
                    value.to_bits()
                }
            });
            if !centers.insert(center_key) {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "cluster center collision",
                    index,
                });
            }
            let end = cluster
                .neighbors
                .end()
                .ok_or(LodgeValidationError::CountOverflow("cluster neighbors"))?;
            if cluster.neighbors.start != expected_neighbor_start
                || end as usize > self.neighbors.len()
                || (self.clusters.len() > 1 && cluster.neighbors.count == 0)
            {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "cluster neighbor range",
                    index,
                });
            }
            if cluster.membership_entry as usize != index {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "cluster membership entry",
                    index,
                });
            }
            let mut previous_neighbor = LodgeClusterId::INVALID;
            for neighbor in &self.neighbors[cluster.neighbors.start as usize..end as usize] {
                if *neighbor == cluster.id
                    || *neighbor <= previous_neighbor
                    || self
                        .clusters
                        .binary_search_by_key(neighbor, |candidate| candidate.id)
                        .is_err()
                {
                    return Err(LodgeValidationError::InvalidIndexed {
                        field: "cluster neighbor",
                        index,
                    });
                }
                previous_neighbor = *neighbor;
            }
            expected_neighbor_start = end;
            previous_cluster = cluster.id;
        }
        if expected_neighbor_start as usize != self.neighbors.len() {
            return Err(LodgeValidationError::Invalid(
                "cluster ranges do not cover the neighbor table",
            ));
        }
        Ok(())
    }

    fn validate_memberships(&self) -> Result<(), LodgeValidationError> {
        let descriptor = &self.membership_index;
        if descriptor.schema_version != LODGE_MEMBERSHIP_SCHEMA_VERSION {
            return Err(LodgeValidationError::UnsupportedMembershipSchema(
                descriptor.schema_version,
            ));
        }
        validate_object(&descriptor.object, "membership object")?;
        require_hash(descriptor.index_sha256, "membership index SHA-256")?;
        check_count(
            "membership entries",
            self.header.cluster_count,
            descriptor.entries.len(),
        )?;
        let (index_start, index_len) = descriptor.index_byte_range;
        if index_start != 0 || index_len == 0 {
            return Err(LodgeValidationError::Invalid(
                "membership index must be a nonempty prefix",
            ));
        }
        let mut expected_start = index_len;
        let mut total_members = 0_u64;
        for (index, entry) in descriptor.entries.iter().enumerate() {
            if entry.cluster != self.clusters[index].id
                || entry.member_count == 0
                || !entry.first_id.is_valid()
                || entry.first_id > entry.last_id
                || entry.last_id.0 > self.header.stable_gaussian_count
                || entry.last_id.0 - entry.first_id.0 + 1 < entry.member_count
            {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "membership entry",
                    index,
                });
            }
            let (start, len) = entry.byte_range;
            if start != expected_start || len == 0 {
                return Err(LodgeValidationError::InvalidIndexed {
                    field: "membership byte range",
                    index,
                });
            }
            expected_start = start
                .checked_add(len)
                .ok_or(LodgeValidationError::CountOverflow(
                    "membership object bytes",
                ))?;
            require_hash(entry.encoded_sha256, "membership stream SHA-256")?;
            total_members = total_members
                .checked_add(entry.member_count)
                .ok_or(LodgeValidationError::CountOverflow("membership IDs"))?;
        }
        if expected_start != descriptor.object.encoded_len {
            return Err(LodgeValidationError::Invalid(
                "membership ranges do not cover the membership object",
            ));
        }
        if total_members != self.header.total_membership_ids {
            return Err(LodgeValidationError::CountMismatch("membership IDs"));
        }
        Ok(())
    }

    fn validate_finest_mapping(
        &self,
        base: &GaussianLodManifest,
        actual: &[LodgeRecordRun],
    ) -> Result<(), LodgeValidationError> {
        let mut leaves = base
            .nodes
            .iter()
            .filter(|node| node.is_leaf())
            .collect::<Vec<_>>();
        leaves.sort_unstable_by_key(|node| node.source.start);
        let mut run_index = 0_usize;
        let mut consumed_in_run = 0_u32;
        for leaf in leaves {
            let mut remaining = leaf.representation.count;
            let mut expected_page_offset = leaf.representation.offset;
            while remaining != 0 {
                let run = actual.get(run_index).ok_or(LodgeValidationError::Invalid(
                    "finest level ends before the base leaf sequence",
                ))?;
                if run.page != leaf.representation.page
                    || run.page_offset.checked_add(consumed_in_run) != Some(expected_page_offset)
                {
                    return Err(LodgeValidationError::Invalid(
                        "finest stable-ID order differs from base source-leaf order",
                    ));
                }
                let available = run.count - consumed_in_run;
                let take = remaining.min(available);
                remaining -= take;
                consumed_in_run += take;
                expected_page_offset += take;
                if consumed_in_run == run.count {
                    run_index += 1;
                    consumed_in_run = 0;
                }
            }
        }
        if run_index != actual.len() || consumed_in_run != 0 {
            return Err(LodgeValidationError::Invalid(
                "finest level has records beyond the base source-leaf sequence",
            ));
        }
        Ok(())
    }

    fn validate_extra_page_coverage(&self) -> Result<(), LodgeValidationError> {
        let mut ranges: BTreeMap<LodPageId, Vec<(u32, u32)>> = self
            .extra_pages
            .iter()
            .map(|page| (page.id, Vec::new()))
            .collect();
        for level in self.levels.iter().skip(1) {
            for run in index_slice(&self.record_runs, level.records)
                .ok_or(LodgeValidationError::Invalid("invalid coarse run range"))?
            {
                ranges
                    .get_mut(&run.page)
                    .ok_or(LodgeValidationError::Invalid(
                        "coarse record run references a non-extra page",
                    ))?
                    .push((run.page_offset, run.page_end().unwrap()));
            }
        }
        for page in &self.extra_pages {
            let page_ranges = ranges.get_mut(&page.id).unwrap();
            page_ranges.sort_unstable();
            let mut expected = 0_u32;
            for &(start, end) in page_ranges.iter() {
                if start != expected || end <= start {
                    return Err(LodgeValidationError::Invalid(
                        "extra page ranges overlap or leave gaps",
                    ));
                }
                expected = end;
            }
            if expected != page.gaussian_count {
                return Err(LodgeValidationError::Invalid(
                    "extra page ranges do not cover the complete page",
                ));
            }
        }
        Ok(())
    }
}

fn index_slice<T>(values: &[T], range: LodIndexRange) -> Option<&[T]> {
    let end = range.end()? as usize;
    values.get(range.start as usize..end)
}

fn check_count(
    field: &'static str,
    expected: u32,
    actual: usize,
) -> Result<(), LodgeValidationError> {
    if usize::try_from(expected).ok() != Some(actual) {
        Err(LodgeValidationError::CountMismatch(field))
    } else {
        Ok(())
    }
}

fn validate_object(
    object: &LodgeAuthenticatedObject,
    field: &'static str,
) -> Result<(), LodgeValidationError> {
    validate_relative_uri(&object.uri)?;
    if object.encoded_len == 0 {
        return Err(LodgeValidationError::InvalidOwned(format!(
            "{field} encoded length is zero"
        )));
    }
    require_hash(object.sha256, field)
}

fn validate_relative_uri(uri: &str) -> Result<(), LodgeValidationError> {
    if uri.is_empty()
        || uri.starts_with('/')
        || uri.contains('\\')
        || uri.contains('?')
        || uri.contains('#')
        || uri.contains('%')
        || uri.contains(':')
        || uri
            .chars()
            .any(|character| character.is_whitespace() || character.is_control())
        || uri
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        Err(LodgeValidationError::Invalid(
            "dependency URI is not a canonical safe relative path",
        ))
    } else {
        Ok(())
    }
}

fn require_hash(hash: [u8; 32], field: &'static str) -> Result<(), LodgeValidationError> {
    if hash.iter().all(|byte| *byte == 0) {
        Err(LodgeValidationError::InvalidOwned(format!(
            "{field} is unset"
        )))
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LodgeValidationError {
    InvalidMagic([u8; 8]),
    UnsupportedManifestVersion(u16),
    UnsupportedPageSchema(u16),
    UnsupportedMembershipSchema(u16),
    UnsupportedRequiredFeatures(u64),
    MissingRequiredFeatures(u64),
    CountOverflow(&'static str),
    CountMismatch(&'static str),
    Invalid(&'static str),
    InvalidIndexed { field: &'static str, index: usize },
    InvalidOwned(String),
}

impl fmt::Display for LodgeValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic(magic) => write!(f, "invalid LODGE manifest magic {magic:?}"),
            Self::UnsupportedManifestVersion(version) => {
                write!(f, "unsupported LODGE manifest version {version}")
            }
            Self::UnsupportedPageSchema(version) => {
                write!(f, "unsupported LODGE page schema {version}")
            }
            Self::UnsupportedMembershipSchema(version) => {
                write!(f, "unsupported LODGE membership schema {version}")
            }
            Self::UnsupportedRequiredFeatures(features) => {
                write!(f, "unsupported LODGE required features {features:#x}")
            }
            Self::MissingRequiredFeatures(features) => {
                write!(f, "missing LODGE required features {features:#x}")
            }
            Self::CountOverflow(field) => write!(f, "LODGE {field} count overflowed"),
            Self::CountMismatch(field) => write!(f, "LODGE {field} count does not match"),
            Self::Invalid(message) => write!(f, "invalid LODGE manifest: {message}"),
            Self::InvalidIndexed { field, index } => {
                write!(f, "invalid LODGE {field} at index {index}")
            }
            Self::InvalidOwned(message) => write!(f, "invalid LODGE manifest: {message}"),
        }
    }
}

impl Error for LodgeValidationError {}

#[cfg(test)]
pub(crate) mod tests {
    use std::mem::size_of;

    use super::*;
    use crate::{
        gaussian::formats::planar_3d_chunked::{LodBounds, LodPageEncoding, LodPageStorage},
        material::spherical_harmonics::SH_DEGREE,
    };

    fn hash(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    pub(crate) fn fixture() -> GaussianLodgeManifest {
        let extra_page = LodPageDescriptor {
            id: LodPageId(2),
            kind: LodPageKind::Representatives,
            encoding: LodPageEncoding::F16Sh {
                degree: (SH_DEGREE as u8).min(1),
            },
            gaussian_count: 2,
            decoded_len: 2 * size_of::<crate::Gaussian3d>() as u64,
            content_hash: 9,
            bounds: LodBounds {
                min: [-1.0; 3],
                max: [1.0; 3],
            },
            storage: Some(LodPageStorage {
                uri: "lodge/level-1.gspage".into(),
                byte_range: None,
                encoded_len: 128,
            }),
        };
        GaussianLodgeManifest {
            header: GaussianLodgeManifestHeader {
                magic: LODGE_MANIFEST_MAGIC,
                manifest_version: LODGE_MANIFEST_VERSION,
                page_schema_version: LOD_PAGE_SCHEMA_VERSION,
                required_features: LODGE_REQUIRED_FEATURES,
                base_page_count: 1,
                extra_page_count: 1,
                level_count: 2,
                cluster_count: 2,
                record_run_count: 2,
                neighbor_count: 2,
                stable_gaussian_count: 4,
                total_membership_ids: 4,
            },
            base_manifest: LodgeAuthenticatedObject {
                uri: "scene.gsplatlod".into(),
                encoded_len: 512,
                sha256: hash(1),
            },
            extra_pages: vec![extra_page],
            page_authentication: vec![
                LodgePageAuthentication {
                    page: LodPageId(1),
                    encoded_sha256: hash(2),
                },
                LodgePageAuthentication {
                    page: LodPageId(2),
                    encoded_sha256: hash(3),
                },
            ],
            levels: vec![
                LodgeLevelDescriptor {
                    id: LodgeLevelId(0),
                    distance_min: 0.0,
                    records: LodIndexRange { start: 0, count: 1 },
                    filter: LodgeLevelFilter::Original,
                },
                LodgeLevelDescriptor {
                    id: LodgeLevelId(1),
                    distance_min: 4.0,
                    records: LodIndexRange { start: 1, count: 1 },
                    filter: LodgeLevelFilter::DepthAware3dV1 {
                        reference_depth: 4.0,
                        reference_focal_length_px: 1200.0,
                        smoothing_scale: 1.0,
                        importance_threshold: 0.25,
                        fine_tune_steps: 100,
                    },
                },
            ],
            record_runs: vec![
                LodgeRecordRun {
                    first_id: LodgeGaussianId(1),
                    count: 2,
                    page: LodPageId(1),
                    page_offset: 0,
                },
                LodgeRecordRun {
                    first_id: LodgeGaussianId(3),
                    count: 2,
                    page: LodPageId(2),
                    page_offset: 0,
                },
            ],
            clusters: vec![
                LodgeCameraCluster {
                    id: LodgeClusterId(1),
                    center: [0.0, 0.0, 2.0],
                    radius: 1.0,
                    neighbors: LodIndexRange { start: 0, count: 1 },
                    membership_entry: 0,
                },
                LodgeCameraCluster {
                    id: LodgeClusterId(2),
                    center: [0.0, 0.0, -2.0],
                    radius: 1.0,
                    neighbors: LodIndexRange { start: 1, count: 1 },
                    membership_entry: 1,
                },
            ],
            neighbors: vec![LodgeClusterId(2), LodgeClusterId(1)],
            membership_index: LodgeMembershipIndexDescriptor {
                schema_version: LODGE_MEMBERSHIP_SCHEMA_VERSION,
                encoding: LodgeMembershipEncoding::DeltaUleb128StableIdsV1,
                object: LodgeAuthenticatedObject {
                    uri: "lodge/members.bgslmem".into(),
                    encoded_len: 10,
                    sha256: hash(4),
                },
                index_byte_range: (0, 4),
                index_sha256: hash(5),
                entries: vec![
                    LodgeMembershipEntry {
                        cluster: LodgeClusterId(1),
                        byte_range: (4, 3),
                        member_count: 2,
                        first_id: LodgeGaussianId(1),
                        last_id: LodgeGaussianId(3),
                        encoded_sha256: hash(6),
                    },
                    LodgeMembershipEntry {
                        cluster: LodgeClusterId(2),
                        byte_range: (7, 3),
                        member_count: 2,
                        first_id: LodgeGaussianId(2),
                        last_id: LodgeGaussianId(4),
                        encoded_sha256: hash(7),
                    },
                ],
            },
        }
    }

    #[test]
    fn valid_sidecar_maps_stable_ids_without_pages() {
        let manifest = fixture();
        manifest.validate().unwrap();
        assert_eq!(
            manifest.locate_gaussian(LodgeGaussianId(4)),
            Some(LodgePageLocator {
                page: LodPageId(2),
                offset: 1,
            })
        );
        assert_eq!(
            manifest.neighbors_for_cluster(LodgeClusterId(1)),
            Some(&[LodgeClusterId(2)][..])
        );
        assert_eq!(
            manifest
                .membership_for_cluster(LodgeClusterId(2))
                .unwrap()
                .first_id,
            LodgeGaussianId(2)
        );
    }

    #[test]
    fn stable_id_gaps_and_dependency_path_traversal_are_rejected() {
        let mut manifest = fixture();
        manifest.record_runs[1].first_id = LodgeGaussianId(4);
        assert!(manifest.validate().is_err());

        let mut manifest = fixture();
        manifest.base_manifest.uri = "../scene.gsplatlod".into();
        assert!(manifest.validate().is_err());

        for uri in [
            "https:scene.gsplatlod",
            "file:scene.gsplatlod",
            "C:scene.gsplatlod",
        ] {
            let mut manifest = fixture();
            manifest.base_manifest.uri = uri.into();
            assert!(
                manifest.validate().is_err(),
                "accepted scheme-like URI {uri}"
            );
        }
    }

    #[test]
    fn sidecar_rejects_a_single_cluster_that_cannot_form_a_blend_pair() {
        let mut manifest = fixture();
        manifest.header.cluster_count = 1;
        manifest.clusters.truncate(1);
        assert!(matches!(
            manifest.validate(),
            Err(LodgeValidationError::Invalid(
                "LODGE requires at least two camera clusters for pair blending"
            ))
        ));
    }

    #[test]
    fn sidecar_rejects_coincident_cluster_centers() {
        let mut manifest = fixture();
        manifest.clusters[1].center = manifest.clusters[0].center;
        assert!(matches!(
            manifest.validate(),
            Err(LodgeValidationError::InvalidIndexed {
                field: "cluster center collision",
                index: 1,
            })
        ));
    }

    #[test]
    fn membership_ranges_must_be_contiguous_and_authenticated() {
        let mut manifest = fixture();
        manifest.membership_index.entries[1].byte_range.0 += 1;
        assert!(manifest.validate().is_err());

        let mut manifest = fixture();
        manifest.membership_index.entries[0].encoded_sha256 = [0; 32];
        assert!(manifest.validate().is_err());
    }
}
