//! Bounded canonical membership-object authoring for external LODGE producers.
//!
//! Upstream LODGE does not define a portable artifact format. This module does
//! not parse an upstream checkpoint; it converts already ordered camera-cluster
//! memberships into the crate-owned `BGSLMEM` v1 object consumed by a
//! [`GaussianLodgeManifest`](crate::gaussian::formats::lodge::GaussianLodgeManifest).

use std::{error::Error, fmt};

use sha2::{Digest, Sha256};

use crate::{
    gaussian::formats::lodge::{
        LODGE_MEMBERSHIP_SCHEMA_VERSION, LodgeAuthenticatedObject, LodgeClusterId, LodgeGaussianId,
        LodgeMembershipEncoding, LodgeMembershipEntry, LodgeMembershipIndexDescriptor,
    },
    io::lodge::{
        LodgeCodecError, LodgeCodecLimits, sha256_bytes, verify_lodge_authenticated_object,
    },
};

/// Magic of the crate-owned canonical membership object.
pub const LODGE_MEMBERSHIP_OBJECT_MAGIC: [u8; 8] = *b"BGSLMEM\0";
pub const LODGE_MEMBERSHIP_OBJECT_VERSION: u16 = 1;
pub const LODGE_MEMBERSHIP_OBJECT_HEADER_LEN: usize = 40;
pub const LODGE_MEMBERSHIP_DIRECTORY_ENTRY_LEN: usize = 80;

/// Bounded replay configuration for one canonical membership-object build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodgeMembershipArtifactConfig {
    /// Maximum IDs a replay source may expose in one callback.
    pub replay_batch_ids: usize,
    pub limits: LodgeCodecLimits,
}

impl Default for LodgeMembershipArtifactConfig {
    fn default() -> Self {
        Self {
            replay_batch_ids: 65_536,
            limits: LodgeCodecLimits::default(),
        }
    }
}

impl LodgeMembershipArtifactConfig {
    pub fn validate(self) -> Result<Self, LodgeMembershipBuildError> {
        self.limits
            .validate()
            .map_err(LodgeMembershipBuildError::Codec)?;
        if self.replay_batch_ids == 0 {
            return Err(LodgeMembershipBuildError::InvalidConfig(
                "replay_batch_ids must be greater than zero",
            ));
        }
        Ok(self)
    }
}

/// Replayable, bounded source for one sorted cluster membership.
///
/// Every replay MUST emit the same strictly increasing stable IDs and return
/// the exact number emitted. A callback batch MUST contain between one and
/// `batch_ids` entries. The builder replays every source twice and rejects
/// semantic drift before returning an artifact.
pub trait ReplayableLodgeMembershipSource {
    fn replay(
        &self,
        batch_ids: usize,
        consume: &mut dyn FnMut(&[LodgeGaussianId]) -> Result<(), LodgeMembershipBuildError>,
    ) -> Result<u64, LodgeMembershipBuildError>;
}

/// Zero-copy replay adapter for an in-memory sorted stable-ID slice.
#[derive(Clone, Copy, Debug)]
pub struct LodgeMembershipSliceSource<'a> {
    ids: &'a [LodgeGaussianId],
}

impl<'a> LodgeMembershipSliceSource<'a> {
    pub const fn new(ids: &'a [LodgeGaussianId]) -> Self {
        Self { ids }
    }

    pub const fn ids(self) -> &'a [LodgeGaussianId] {
        self.ids
    }
}

impl ReplayableLodgeMembershipSource for LodgeMembershipSliceSource<'_> {
    fn replay(
        &self,
        batch_ids: usize,
        consume: &mut dyn FnMut(&[LodgeGaussianId]) -> Result<(), LodgeMembershipBuildError>,
    ) -> Result<u64, LodgeMembershipBuildError> {
        if batch_ids == 0 {
            return Err(LodgeMembershipBuildError::InvalidConfig(
                "batch_ids must be greater than zero",
            ));
        }
        for batch in self.ids.chunks(batch_ids) {
            consume(batch)?;
        }
        u64::try_from(self.ids.len()).map_err(|_| LodgeMembershipBuildError::LengthOverflow)
    }
}

/// Ordered cluster ID and its replayable membership source.
#[derive(Clone, Copy)]
pub struct LodgeClusterMembershipInput<'a> {
    pub cluster: LodgeClusterId,
    pub source: &'a dyn ReplayableLodgeMembershipSource,
}

impl<'a> LodgeClusterMembershipInput<'a> {
    pub const fn new(
        cluster: LodgeClusterId,
        source: &'a dyn ReplayableLodgeMembershipSource,
    ) -> Self {
        Self { cluster, source }
    }
}

impl fmt::Debug for LodgeClusterMembershipInput<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LodgeClusterMembershipInput")
            .field("cluster", &self.cluster)
            .field("source", &"dyn ReplayableLodgeMembershipSource")
            .finish()
    }
}

/// Complete canonical membership object and its manifest-ready descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalLodgeMembershipArtifact {
    encoded: Vec<u8>,
    descriptor: LodgeMembershipIndexDescriptor,
}

impl CanonicalLodgeMembershipArtifact {
    pub fn encoded(&self) -> &[u8] {
        &self.encoded
    }

    pub const fn descriptor(&self) -> &LodgeMembershipIndexDescriptor {
        &self.descriptor
    }

    pub fn into_parts(self) -> (Vec<u8>, LodgeMembershipIndexDescriptor) {
        (self.encoded, self.descriptor)
    }

    pub fn validate(
        &self,
        stable_gaussian_count: u64,
        limits: LodgeCodecLimits,
    ) -> Result<(), LodgeMembershipBuildError> {
        validate_canonical_lodge_membership_artifact(
            &self.encoded,
            &self.descriptor,
            stable_gaussian_count,
            limits,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MembershipPlan {
    cluster: LodgeClusterId,
    member_count: u64,
    first_id: LodgeGaussianId,
    last_id: LodgeGaussianId,
    encoded_len: u64,
    encoded_sha256: [u8; 32],
}

/// Builds a deterministic crate-owned `BGSLMEM` v1 membership object.
///
/// `inputs` MUST contain at least two entries strictly ordered by nonzero
/// cluster ID. Each
/// source is scanned once to establish bounded sizes and identities, then
/// replayed into one exactly reserved object. A source which changes between
/// those passes is rejected. The returned descriptor can be assigned directly
/// to [`GaussianLodgeManifest::membership_index`](crate::gaussian::formats::lodge::GaussianLodgeManifest::membership_index).
pub fn build_canonical_lodge_membership_artifact(
    uri: &str,
    stable_gaussian_count: u64,
    inputs: &[LodgeClusterMembershipInput<'_>],
    config: LodgeMembershipArtifactConfig,
) -> Result<CanonicalLodgeMembershipArtifact, LodgeMembershipBuildError> {
    let config = config.validate()?;
    validate_relative_uri(uri)?;
    enforce_stable_count(stable_gaussian_count, config.limits)?;
    validate_cluster_inputs(inputs, config.limits)?;

    let mut plans = Vec::new();
    plans.try_reserve_exact(inputs.len()).map_err(|error| {
        LodgeMembershipBuildError::Allocation {
            field: "membership plans",
            detail: error.to_string(),
        }
    })?;
    let mut total_membership_ids = 0_u64;
    for input in inputs {
        let plan = scan_membership(
            input.cluster,
            input.source,
            stable_gaussian_count,
            config,
            |_| Ok(()),
        )?;
        total_membership_ids = total_membership_ids
            .checked_add(plan.member_count)
            .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
        enforce_limit(
            "total membership IDs",
            total_membership_ids,
            config.limits.max_total_membership_ids,
        )?;
        plans.push(plan);
    }

    let prefix_len = canonical_prefix_len(plans.len())?;
    let mut object_len = prefix_len;
    for plan in &plans {
        object_len = object_len
            .checked_add(plan.encoded_len)
            .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
    }
    enforce_limit(
        "membership object bytes",
        object_len,
        config.limits.max_dependency_bytes,
    )?;
    let object_capacity =
        usize::try_from(object_len).map_err(|_| LodgeMembershipBuildError::LengthOverflow)?;

    let mut entries = Vec::new();
    entries.try_reserve_exact(plans.len()).map_err(|error| {
        LodgeMembershipBuildError::Allocation {
            field: "membership entries",
            detail: error.to_string(),
        }
    })?;
    let mut stream_start = prefix_len;
    for plan in &plans {
        entries.push(LodgeMembershipEntry {
            cluster: plan.cluster,
            byte_range: (stream_start, plan.encoded_len),
            member_count: plan.member_count,
            first_id: plan.first_id,
            last_id: plan.last_id,
            encoded_sha256: plan.encoded_sha256,
        });
        stream_start = stream_start
            .checked_add(plan.encoded_len)
            .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
    }
    debug_assert_eq!(stream_start, object_len);

    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(object_capacity)
        .map_err(|error| LodgeMembershipBuildError::Allocation {
            field: "membership object",
            detail: error.to_string(),
        })?;
    write_header(&mut encoded, plans.len(), prefix_len, object_len)?;
    for entry in &entries {
        write_directory_entry(&mut encoded, entry);
    }
    debug_assert_eq!(encoded.len() as u64, prefix_len);
    let index_sha256 = sha256_bytes(&encoded);

    for ((input, expected), entry) in inputs.iter().zip(&plans).zip(&entries) {
        if encoded.len() as u64 != entry.byte_range.0 {
            return Err(LodgeMembershipBuildError::LengthOverflow);
        }
        let expected_end = entry
            .byte_range
            .0
            .checked_add(entry.byte_range.1)
            .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
        let expected_end_usize =
            usize::try_from(expected_end).map_err(|_| LodgeMembershipBuildError::LengthOverflow)?;
        let actual = scan_membership(
            input.cluster,
            input.source,
            stable_gaussian_count,
            config,
            |bytes| {
                let next_len = encoded
                    .len()
                    .checked_add(bytes.len())
                    .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
                if next_len > expected_end_usize {
                    return Err(LodgeMembershipBuildError::ReplayDrift {
                        cluster: input.cluster,
                    });
                }
                encoded.extend_from_slice(bytes);
                Ok(())
            },
        )?;
        if &actual != expected {
            return Err(LodgeMembershipBuildError::ReplayDrift {
                cluster: input.cluster,
            });
        }
        if encoded.len() as u64 != expected_end {
            return Err(LodgeMembershipBuildError::ReplayDrift {
                cluster: input.cluster,
            });
        }
    }
    debug_assert_eq!(encoded.len() as u64, object_len);

    let descriptor = LodgeMembershipIndexDescriptor {
        schema_version: LODGE_MEMBERSHIP_SCHEMA_VERSION,
        encoding: LodgeMembershipEncoding::DeltaUleb128StableIdsV1,
        object: LodgeAuthenticatedObject {
            uri: uri.to_owned(),
            encoded_len: object_len,
            sha256: sha256_bytes(&encoded),
        },
        index_byte_range: (0, prefix_len),
        index_sha256,
        entries,
    };
    validate_canonical_lodge_membership_artifact(
        &encoded,
        &descriptor,
        stable_gaussian_count,
        config.limits,
    )?;
    Ok(CanonicalLodgeMembershipArtifact {
        encoded,
        descriptor,
    })
}

/// Validates both authentication and the fixed canonical `BGSLMEM` directory.
///
/// This is stricter than the runtime's generic membership-object proof, which
/// deliberately accepts authenticated producer-owned opaque prefixes.
pub fn validate_canonical_lodge_membership_artifact(
    encoded: &[u8],
    descriptor: &LodgeMembershipIndexDescriptor,
    stable_gaussian_count: u64,
    limits: LodgeCodecLimits,
) -> Result<(), LodgeMembershipBuildError> {
    let limits = limits
        .validate()
        .map_err(LodgeMembershipBuildError::Codec)?;
    enforce_stable_count(stable_gaussian_count, limits)?;
    validate_relative_uri(&descriptor.object.uri)?;
    if descriptor.schema_version != LODGE_MEMBERSHIP_SCHEMA_VERSION {
        return Err(LodgeMembershipBuildError::UnsupportedDirectoryVersion(
            descriptor.schema_version,
        ));
    }
    if descriptor.encoding != LodgeMembershipEncoding::DeltaUleb128StableIdsV1 {
        return Err(LodgeMembershipBuildError::HeaderMismatch(
            "membership encoding",
        ));
    }
    if descriptor.entries.len() < 2 {
        return Err(LodgeMembershipBuildError::InsufficientClusters {
            actual: descriptor.entries.len(),
        });
    }
    enforce_limit(
        "clusters",
        descriptor.entries.len() as u64,
        u64::from(limits.max_clusters),
    )?;
    enforce_limit(
        "membership object bytes",
        encoded.len() as u64,
        limits.max_dependency_bytes,
    )?;
    verify_lodge_authenticated_object(encoded, &descriptor.object, limits.max_dependency_bytes)
        .map_err(LodgeMembershipBuildError::Codec)?;

    let expected_prefix_len = canonical_prefix_len(descriptor.entries.len())?;
    if descriptor.index_byte_range != (0, expected_prefix_len) {
        return Err(LodgeMembershipBuildError::HeaderMismatch(
            "index byte range",
        ));
    }
    let prefix_end = usize::try_from(expected_prefix_len)
        .map_err(|_| LodgeMembershipBuildError::LengthOverflow)?;
    let prefix = encoded
        .get(..prefix_end)
        .ok_or(LodgeMembershipBuildError::Truncated("membership prefix"))?;
    if sha256_bytes(prefix) != descriptor.index_sha256 {
        return Err(LodgeMembershipBuildError::Codec(
            LodgeCodecError::Sha256Mismatch("membership index"),
        ));
    }

    validate_header(
        prefix,
        descriptor.entries.len(),
        expected_prefix_len,
        encoded.len() as u64,
    )?;

    let mut expected_cluster = LodgeClusterId::INVALID;
    let mut expected_stream_start = expected_prefix_len;
    let mut total_membership_ids = 0_u64;
    for (index, entry) in descriptor.entries.iter().enumerate() {
        if !entry.cluster.is_valid() || entry.cluster <= expected_cluster {
            return Err(LodgeMembershipBuildError::InvalidClusterOrder {
                index,
                previous: expected_cluster,
                actual: entry.cluster,
            });
        }
        if entry.byte_range.0 != expected_stream_start || entry.byte_range.1 == 0 {
            return Err(LodgeMembershipBuildError::DirectoryMismatch {
                index,
                field: "stream byte range",
            });
        }
        enforce_limit(
            "members per cluster",
            entry.member_count,
            limits.max_members_per_cluster,
        )?;
        enforce_limit(
            "membership stream bytes",
            entry.byte_range.1,
            limits.max_membership_stream_bytes,
        )?;
        total_membership_ids = total_membership_ids
            .checked_add(entry.member_count)
            .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
        enforce_limit(
            "total membership IDs",
            total_membership_ids,
            limits.max_total_membership_ids,
        )?;
        validate_directory_entry(prefix, index, entry)?;

        let stream_end = entry
            .byte_range
            .0
            .checked_add(entry.byte_range.1)
            .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
        let start = usize::try_from(entry.byte_range.0)
            .map_err(|_| LodgeMembershipBuildError::LengthOverflow)?;
        let end =
            usize::try_from(stream_end).map_err(|_| LodgeMembershipBuildError::LengthOverflow)?;
        let stream = encoded
            .get(start..end)
            .ok_or(LodgeMembershipBuildError::Truncated("membership stream"))?;
        if sha256_bytes(stream) != entry.encoded_sha256 {
            return Err(LodgeMembershipBuildError::Codec(
                LodgeCodecError::Sha256Mismatch("membership stream"),
            ));
        }
        validate_encoded_membership_stream(stream, entry, stable_gaussian_count)?;
        expected_stream_start = stream_end;
        expected_cluster = entry.cluster;
    }
    if expected_stream_start != encoded.len() as u64 {
        return Err(LodgeMembershipBuildError::HeaderMismatch(
            "membership object coverage",
        ));
    }
    Ok(())
}

fn validate_cluster_inputs(
    inputs: &[LodgeClusterMembershipInput<'_>],
    limits: LodgeCodecLimits,
) -> Result<(), LodgeMembershipBuildError> {
    if inputs.len() < 2 {
        return Err(LodgeMembershipBuildError::InsufficientClusters {
            actual: inputs.len(),
        });
    }
    enforce_limit(
        "clusters",
        inputs.len() as u64,
        u64::from(limits.max_clusters),
    )?;
    let _ = u32::try_from(inputs.len()).map_err(|_| LodgeMembershipBuildError::LengthOverflow)?;
    let mut previous = LodgeClusterId::INVALID;
    for (index, input) in inputs.iter().enumerate() {
        if !input.cluster.is_valid() || input.cluster <= previous {
            return Err(LodgeMembershipBuildError::InvalidClusterOrder {
                index,
                previous,
                actual: input.cluster,
            });
        }
        previous = input.cluster;
    }
    Ok(())
}

fn scan_membership(
    cluster: LodgeClusterId,
    source: &dyn ReplayableLodgeMembershipSource,
    stable_gaussian_count: u64,
    config: LodgeMembershipArtifactConfig,
    mut emit: impl FnMut(&[u8]) -> Result<(), LodgeMembershipBuildError>,
) -> Result<MembershipPlan, LodgeMembershipBuildError> {
    let mut member_count = 0_u64;
    let mut encoded_len = 0_u64;
    let mut first_id = None;
    let mut previous = LodgeGaussianId::INVALID;
    let mut encoded_sha256 = Sha256::new();
    let reported = source.replay(config.replay_batch_ids, &mut |batch| {
        if batch.is_empty() {
            return Err(LodgeMembershipBuildError::EmptyReplayBatch { cluster });
        }
        if batch.len() > config.replay_batch_ids {
            return Err(LodgeMembershipBuildError::ReplayBatchTooLarge {
                cluster,
                actual: batch.len(),
                limit: config.replay_batch_ids,
            });
        }
        for id in batch.iter().copied() {
            if !id.is_valid() || id <= previous {
                return Err(LodgeMembershipBuildError::InvalidMembershipOrder {
                    cluster,
                    index: member_count,
                    previous,
                    actual: id,
                });
            }
            if id.0 > stable_gaussian_count {
                return Err(LodgeMembershipBuildError::MembershipIdOutOfRange {
                    cluster,
                    id,
                    stable_gaussian_count,
                });
            }
            let delta =
                id.0.checked_sub(previous.0)
                    .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
            let mut bytes = [0_u8; 10];
            let byte_count = encode_uleb128(delta, &mut bytes);
            encoded_len = encoded_len
                .checked_add(byte_count as u64)
                .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
            enforce_limit(
                "membership stream bytes",
                encoded_len,
                config.limits.max_membership_stream_bytes,
            )?;
            member_count = member_count
                .checked_add(1)
                .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
            enforce_limit(
                "members per cluster",
                member_count,
                config.limits.max_members_per_cluster,
            )?;
            let encoded = &bytes[..byte_count];
            encoded_sha256.update(encoded);
            emit(encoded)?;
            first_id.get_or_insert(id);
            previous = id;
        }
        Ok(())
    })?;
    if reported != member_count {
        return Err(LodgeMembershipBuildError::SourceCountMismatch {
            cluster,
            reported,
            observed: member_count,
        });
    }
    let first_id = first_id.ok_or(LodgeMembershipBuildError::EmptyMembership { cluster })?;
    Ok(MembershipPlan {
        cluster,
        member_count,
        first_id,
        last_id: previous,
        encoded_len,
        encoded_sha256: encoded_sha256.finalize().into(),
    })
}

fn canonical_prefix_len(entry_count: usize) -> Result<u64, LodgeMembershipBuildError> {
    let directory_len = entry_count
        .checked_mul(LODGE_MEMBERSHIP_DIRECTORY_ENTRY_LEN)
        .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
    LODGE_MEMBERSHIP_OBJECT_HEADER_LEN
        .checked_add(directory_len)
        .and_then(|length| u64::try_from(length).ok())
        .ok_or(LodgeMembershipBuildError::LengthOverflow)
}

fn write_header(
    encoded: &mut Vec<u8>,
    entry_count: usize,
    prefix_len: u64,
    object_len: u64,
) -> Result<(), LodgeMembershipBuildError> {
    let entry_count =
        u32::try_from(entry_count).map_err(|_| LodgeMembershipBuildError::LengthOverflow)?;
    encoded.extend_from_slice(&LODGE_MEMBERSHIP_OBJECT_MAGIC);
    encoded.extend_from_slice(&LODGE_MEMBERSHIP_OBJECT_VERSION.to_le_bytes());
    encoded.extend_from_slice(&LODGE_MEMBERSHIP_SCHEMA_VERSION.to_le_bytes());
    encoded.extend_from_slice(&0_u16.to_le_bytes()); // flags
    encoded.extend_from_slice(&0_u16.to_le_bytes()); // reserved
    encoded.extend_from_slice(&entry_count.to_le_bytes());
    encoded.extend_from_slice(&(LODGE_MEMBERSHIP_DIRECTORY_ENTRY_LEN as u32).to_le_bytes());
    encoded.extend_from_slice(&prefix_len.to_le_bytes());
    encoded.extend_from_slice(&object_len.to_le_bytes());
    debug_assert_eq!(encoded.len(), LODGE_MEMBERSHIP_OBJECT_HEADER_LEN);
    Ok(())
}

fn write_directory_entry(encoded: &mut Vec<u8>, entry: &LodgeMembershipEntry) {
    encoded.extend_from_slice(&entry.cluster.0.to_le_bytes());
    encoded.extend_from_slice(&0_u32.to_le_bytes());
    encoded.extend_from_slice(&entry.byte_range.0.to_le_bytes());
    encoded.extend_from_slice(&entry.byte_range.1.to_le_bytes());
    encoded.extend_from_slice(&entry.member_count.to_le_bytes());
    encoded.extend_from_slice(&entry.first_id.0.to_le_bytes());
    encoded.extend_from_slice(&entry.last_id.0.to_le_bytes());
    encoded.extend_from_slice(&entry.encoded_sha256);
}

fn validate_header(
    prefix: &[u8],
    entry_count: usize,
    prefix_len: u64,
    object_len: u64,
) -> Result<(), LodgeMembershipBuildError> {
    if prefix.len() < LODGE_MEMBERSHIP_OBJECT_HEADER_LEN {
        return Err(LodgeMembershipBuildError::Truncated(
            "membership object header",
        ));
    }
    if prefix[..8] != LODGE_MEMBERSHIP_OBJECT_MAGIC {
        return Err(LodgeMembershipBuildError::InvalidObjectMagic);
    }
    let version = read_u16(prefix, 8)?;
    if version != LODGE_MEMBERSHIP_OBJECT_VERSION {
        return Err(LodgeMembershipBuildError::UnsupportedObjectVersion(version));
    }
    let schema = read_u16(prefix, 10)?;
    if schema != LODGE_MEMBERSHIP_SCHEMA_VERSION {
        return Err(LodgeMembershipBuildError::UnsupportedDirectoryVersion(
            schema,
        ));
    }
    if prefix[12..16].iter().any(|byte| *byte != 0) {
        return Err(LodgeMembershipBuildError::NonZeroReservedBytes);
    }
    let expected_count =
        u32::try_from(entry_count).map_err(|_| LodgeMembershipBuildError::LengthOverflow)?;
    if read_u32(prefix, 16)? != expected_count {
        return Err(LodgeMembershipBuildError::HeaderMismatch(
            "directory entry count",
        ));
    }
    if read_u32(prefix, 20)? != LODGE_MEMBERSHIP_DIRECTORY_ENTRY_LEN as u32 {
        return Err(LodgeMembershipBuildError::HeaderMismatch(
            "directory entry length",
        ));
    }
    if read_u64(prefix, 24)? != prefix_len {
        return Err(LodgeMembershipBuildError::HeaderMismatch("prefix length"));
    }
    if read_u64(prefix, 32)? != object_len {
        return Err(LodgeMembershipBuildError::HeaderMismatch("object length"));
    }
    Ok(())
}

fn validate_directory_entry(
    prefix: &[u8],
    index: usize,
    expected: &LodgeMembershipEntry,
) -> Result<(), LodgeMembershipBuildError> {
    let offset = LODGE_MEMBERSHIP_OBJECT_HEADER_LEN
        .checked_add(
            index
                .checked_mul(LODGE_MEMBERSHIP_DIRECTORY_ENTRY_LEN)
                .ok_or(LodgeMembershipBuildError::LengthOverflow)?,
        )
        .ok_or(LodgeMembershipBuildError::LengthOverflow)?;
    let entry = prefix
        .get(offset..offset + LODGE_MEMBERSHIP_DIRECTORY_ENTRY_LEN)
        .ok_or(LodgeMembershipBuildError::Truncated("membership directory"))?;
    if read_u32(entry, 0)? != expected.cluster.0 {
        return Err(LodgeMembershipBuildError::DirectoryMismatch {
            index,
            field: "cluster ID",
        });
    }
    if read_u32(entry, 4)? != 0 {
        return Err(LodgeMembershipBuildError::NonZeroReservedBytes);
    }
    let fields = [
        ("stream offset", read_u64(entry, 8)?, expected.byte_range.0),
        ("stream length", read_u64(entry, 16)?, expected.byte_range.1),
        ("member count", read_u64(entry, 24)?, expected.member_count),
        ("first ID", read_u64(entry, 32)?, expected.first_id.0),
        ("last ID", read_u64(entry, 40)?, expected.last_id.0),
    ];
    for (field, actual, expected) in fields {
        if actual != expected {
            return Err(LodgeMembershipBuildError::DirectoryMismatch { index, field });
        }
    }
    if entry[48..80] != expected.encoded_sha256 {
        return Err(LodgeMembershipBuildError::DirectoryMismatch {
            index,
            field: "stream SHA-256",
        });
    }
    Ok(())
}

fn validate_encoded_membership_stream(
    encoded: &[u8],
    entry: &LodgeMembershipEntry,
    stable_gaussian_count: u64,
) -> Result<(), LodgeMembershipBuildError> {
    if entry.member_count == 0 {
        return Err(LodgeMembershipBuildError::EmptyMembership {
            cluster: entry.cluster,
        });
    }
    let mut cursor = 0_usize;
    let mut count = 0_u64;
    let mut previous = LodgeGaussianId::INVALID;
    let mut first = None;
    while cursor < encoded.len() {
        if count == entry.member_count {
            return Err(LodgeMembershipBuildError::InvalidEncodedMembership {
                cluster: entry.cluster,
                detail: "stream contains more IDs than declared",
            });
        }
        let delta = decode_uleb128(encoded, &mut cursor).map_err(|detail| {
            LodgeMembershipBuildError::InvalidEncodedMembership {
                cluster: entry.cluster,
                detail,
            }
        })?;
        if delta == 0 {
            return Err(LodgeMembershipBuildError::InvalidEncodedMembership {
                cluster: entry.cluster,
                detail: "stream contains a zero delta",
            });
        }
        let id = previous.0.checked_add(delta).ok_or(
            LodgeMembershipBuildError::InvalidEncodedMembership {
                cluster: entry.cluster,
                detail: "stable ID overflow",
            },
        )?;
        if id > stable_gaussian_count {
            return Err(LodgeMembershipBuildError::MembershipIdOutOfRange {
                cluster: entry.cluster,
                id: LodgeGaussianId(id),
                stable_gaussian_count,
            });
        }
        let id = LodgeGaussianId(id);
        first.get_or_insert(id);
        previous = id;
        count += 1;
    }
    if count != entry.member_count || first != Some(entry.first_id) || previous != entry.last_id {
        return Err(LodgeMembershipBuildError::InvalidEncodedMembership {
            cluster: entry.cluster,
            detail: "stream count or endpoints differ from the directory",
        });
    }
    Ok(())
}

fn encode_uleb128(mut value: u64, bytes: &mut [u8; 10]) -> usize {
    let mut length = 0;
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        bytes[length] = byte;
        length += 1;
        if value == 0 {
            return length;
        }
    }
}

fn decode_uleb128(bytes: &[u8], cursor: &mut usize) -> Result<u64, &'static str> {
    let start = *cursor;
    let mut value = 0_u64;
    for byte_index in 0..10_u32 {
        let byte = *bytes.get(*cursor).ok_or("truncated uLEB128 value")?;
        *cursor += 1;
        let payload = u64::from(byte & 0x7f);
        if byte_index == 9 && payload > 1 {
            return Err("uLEB128 value overflows u64");
        }
        value |= payload << (byte_index * 7);
        if byte & 0x80 == 0 {
            if *cursor - start > 1 && payload == 0 {
                return Err("uLEB128 value is not shortest-form canonical");
            }
            return Ok(value);
        }
    }
    Err("uLEB128 value exceeds ten bytes")
}

fn enforce_stable_count(
    stable_gaussian_count: u64,
    limits: LodgeCodecLimits,
) -> Result<(), LodgeMembershipBuildError> {
    if stable_gaussian_count == 0 {
        return Err(LodgeMembershipBuildError::InvalidConfig(
            "stable_gaussian_count must be greater than zero",
        ));
    }
    enforce_limit(
        "stable Gaussians",
        stable_gaussian_count,
        limits.max_stable_gaussians,
    )
}

fn enforce_limit(
    field: &'static str,
    actual: u64,
    limit: u64,
) -> Result<(), LodgeMembershipBuildError> {
    if actual > limit {
        Err(LodgeMembershipBuildError::LimitExceeded {
            field,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn validate_relative_uri(uri: &str) -> Result<(), LodgeMembershipBuildError> {
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
        Err(LodgeMembershipBuildError::InvalidUri)
    } else {
        Ok(())
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, LodgeMembershipBuildError> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or(LodgeMembershipBuildError::Truncated("u16 field"))?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, LodgeMembershipBuildError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(LodgeMembershipBuildError::Truncated("u32 field"))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, LodgeMembershipBuildError> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(LodgeMembershipBuildError::Truncated("u64 field"))?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LodgeMembershipBuildError {
    Codec(LodgeCodecError),
    InvalidConfig(&'static str),
    InvalidUri,
    InsufficientClusters {
        actual: usize,
    },
    InvalidClusterOrder {
        index: usize,
        previous: LodgeClusterId,
        actual: LodgeClusterId,
    },
    EmptyReplayBatch {
        cluster: LodgeClusterId,
    },
    ReplayBatchTooLarge {
        cluster: LodgeClusterId,
        actual: usize,
        limit: usize,
    },
    EmptyMembership {
        cluster: LodgeClusterId,
    },
    InvalidMembershipOrder {
        cluster: LodgeClusterId,
        index: u64,
        previous: LodgeGaussianId,
        actual: LodgeGaussianId,
    },
    MembershipIdOutOfRange {
        cluster: LodgeClusterId,
        id: LodgeGaussianId,
        stable_gaussian_count: u64,
    },
    SourceCountMismatch {
        cluster: LodgeClusterId,
        reported: u64,
        observed: u64,
    },
    ReplayDrift {
        cluster: LodgeClusterId,
    },
    LimitExceeded {
        field: &'static str,
        actual: u64,
        limit: u64,
    },
    LengthOverflow,
    Allocation {
        field: &'static str,
        detail: String,
    },
    InvalidObjectMagic,
    UnsupportedObjectVersion(u16),
    UnsupportedDirectoryVersion(u16),
    NonZeroReservedBytes,
    Truncated(&'static str),
    HeaderMismatch(&'static str),
    DirectoryMismatch {
        index: usize,
        field: &'static str,
    },
    InvalidEncodedMembership {
        cluster: LodgeClusterId,
        detail: &'static str,
    },
    Source(String),
}

impl fmt::Display for LodgeMembershipBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(error) => write!(formatter, "LODGE codec error: {error}"),
            Self::InvalidConfig(detail) => {
                write!(formatter, "invalid membership build config: {detail}")
            }
            Self::InvalidUri => write!(
                formatter,
                "membership object URI is not a canonical safe relative path"
            ),
            Self::InsufficientClusters { actual } => write!(
                formatter,
                "canonical membership object requires at least two clusters, found {actual}"
            ),
            Self::InvalidClusterOrder {
                index,
                previous,
                actual,
            } => write!(
                formatter,
                "cluster {actual:?} at index {index} is zero, duplicate, or not ordered after {previous:?}"
            ),
            Self::EmptyReplayBatch { cluster } => write!(
                formatter,
                "membership source for cluster {cluster:?} emitted an empty batch"
            ),
            Self::ReplayBatchTooLarge {
                cluster,
                actual,
                limit,
            } => write!(
                formatter,
                "membership source for cluster {cluster:?} emitted {actual} IDs in one batch, limit {limit}"
            ),
            Self::EmptyMembership { cluster } => {
                write!(formatter, "membership for cluster {cluster:?} is empty")
            }
            Self::InvalidMembershipOrder {
                cluster,
                index,
                previous,
                actual,
            } => write!(
                formatter,
                "membership for cluster {cluster:?} has invalid ID {actual:?} at index {index} after {previous:?}"
            ),
            Self::MembershipIdOutOfRange {
                cluster,
                id,
                stable_gaussian_count,
            } => write!(
                formatter,
                "membership for cluster {cluster:?} uses ID {id:?} above catalog size {stable_gaussian_count}"
            ),
            Self::SourceCountMismatch {
                cluster,
                reported,
                observed,
            } => write!(
                formatter,
                "membership source for cluster {cluster:?} reported {reported} IDs after emitting {observed}"
            ),
            Self::ReplayDrift { cluster } => write!(
                formatter,
                "membership source for cluster {cluster:?} changed between bounded replays"
            ),
            Self::LimitExceeded {
                field,
                actual,
                limit,
            } => write!(
                formatter,
                "membership {field} {actual} exceeds limit {limit}"
            ),
            Self::LengthOverflow => {
                write!(formatter, "canonical membership artifact length overflow")
            }
            Self::Allocation { field, detail } => {
                write!(formatter, "failed to reserve bounded {field}: {detail}")
            }
            Self::InvalidObjectMagic => write!(formatter, "invalid BGSLMEM object magic"),
            Self::UnsupportedObjectVersion(version) => {
                write!(formatter, "unsupported BGSLMEM object version {version}")
            }
            Self::UnsupportedDirectoryVersion(version) => {
                write!(formatter, "unsupported BGSLMEM directory version {version}")
            }
            Self::NonZeroReservedBytes => write!(formatter, "BGSLMEM reserved bytes are nonzero"),
            Self::Truncated(field) => write!(formatter, "truncated BGSLMEM {field}"),
            Self::HeaderMismatch(field) => write!(formatter, "BGSLMEM header {field} mismatch"),
            Self::DirectoryMismatch { index, field } => write!(
                formatter,
                "BGSLMEM directory entry {index} {field} mismatch"
            ),
            Self::InvalidEncodedMembership { cluster, detail } => write!(
                formatter,
                "BGSLMEM stream for cluster {cluster:?} is invalid: {detail}"
            ),
            Self::Source(detail) => write!(formatter, "membership source failed: {detail}"),
        }
    }
}

impl Error for LodgeMembershipBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Codec(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LodgeCodecError> for LodgeMembershipBuildError {
    fn from(error: LodgeCodecError) -> Self {
        Self::Codec(error)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::{
        gaussian::formats::lodge::tests::fixture as manifest_fixture,
        io::lodge::decode_lodge_membership_entry,
    };

    fn ids(values: &[u64]) -> Vec<LodgeGaussianId> {
        values.iter().copied().map(LodgeGaussianId).collect()
    }

    fn fixture() -> CanonicalLodgeMembershipArtifact {
        let first_ids = ids(&[1, 3, 128, 1024]);
        let second_ids = ids(&[2, 4, 129, 2048]);
        let first = LodgeMembershipSliceSource::new(&first_ids);
        let second = LodgeMembershipSliceSource::new(&second_ids);
        build_canonical_lodge_membership_artifact(
            "memberships/scene.bgslmem",
            2048,
            &[
                LodgeClusterMembershipInput::new(LodgeClusterId(1), &first),
                LodgeClusterMembershipInput::new(LodgeClusterId(2), &second),
            ],
            LodgeMembershipArtifactConfig {
                replay_batch_ids: 2,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn canonical_artifact_is_deterministic_and_round_trips_streams() {
        let first = fixture();
        let second = fixture();
        assert_eq!(first, second);
        assert_eq!(&first.encoded()[..8], &LODGE_MEMBERSHIP_OBJECT_MAGIC);
        assert_eq!(
            first.descriptor().index_byte_range.1,
            (LODGE_MEMBERSHIP_OBJECT_HEADER_LEN + 2 * LODGE_MEMBERSHIP_DIRECTORY_ENTRY_LEN) as u64
        );
        first.validate(2048, LodgeCodecLimits::default()).unwrap();
        for entry in &first.descriptor().entries {
            let start = entry.byte_range.0 as usize;
            let end = (entry.byte_range.0 + entry.byte_range.1) as usize;
            let decoded = decode_lodge_membership_entry(
                &first.encoded()[start..end],
                entry,
                2048,
                LodgeCodecLimits::default(),
            )
            .unwrap();
            assert_eq!(decoded.len() as u64, entry.member_count);
            assert_eq!(decoded.first().copied(), Some(entry.first_id));
            assert_eq!(decoded.last().copied(), Some(entry.last_id));
        }
    }

    #[test]
    fn descriptor_is_ready_for_the_semantic_manifest() {
        let first_ids = ids(&[1, 3]);
        let second_ids = ids(&[2, 4]);
        let first = LodgeMembershipSliceSource::new(&first_ids);
        let second = LodgeMembershipSliceSource::new(&second_ids);
        let artifact = build_canonical_lodge_membership_artifact(
            "memberships/scene.bgslmem",
            4,
            &[
                LodgeClusterMembershipInput::new(LodgeClusterId(1), &first),
                LodgeClusterMembershipInput::new(LodgeClusterId(2), &second),
            ],
            LodgeMembershipArtifactConfig::default(),
        )
        .unwrap();
        let mut manifest = manifest_fixture();
        manifest.membership_index = artifact.descriptor().clone();
        manifest.validate().unwrap();
    }

    #[test]
    fn authentication_and_directory_validation_reject_tampering() {
        let artifact = fixture();
        let (mut encoded, descriptor) = artifact.clone().into_parts();
        encoded[0] ^= 0xff;
        assert!(matches!(
            validate_canonical_lodge_membership_artifact(
                &encoded,
                &descriptor,
                2048,
                LodgeCodecLimits::default()
            ),
            Err(LodgeMembershipBuildError::Codec(
                LodgeCodecError::Sha256Mismatch("authenticated object")
            ))
        ));

        let (mut encoded, mut descriptor) = artifact.into_parts();
        encoded[0] ^= 0xff;
        descriptor.index_sha256 = sha256_bytes(&encoded[..descriptor.index_byte_range.1 as usize]);
        descriptor.object.sha256 = sha256_bytes(&encoded);
        assert_eq!(
            validate_canonical_lodge_membership_artifact(
                &encoded,
                &descriptor,
                2048,
                LodgeCodecLimits::default()
            ),
            Err(LodgeMembershipBuildError::InvalidObjectMagic)
        );

        let (mut encoded, mut descriptor) = fixture().into_parts();
        let first_stream_offset = LODGE_MEMBERSHIP_OBJECT_HEADER_LEN + 8;
        encoded[first_stream_offset] ^= 1;
        descriptor.index_sha256 = sha256_bytes(&encoded[..descriptor.index_byte_range.1 as usize]);
        descriptor.object.sha256 = sha256_bytes(&encoded);
        assert_eq!(
            validate_canonical_lodge_membership_artifact(
                &encoded,
                &descriptor,
                2048,
                LodgeCodecLimits::default()
            ),
            Err(LodgeMembershipBuildError::DirectoryMismatch {
                index: 0,
                field: "stream offset",
            })
        );
    }

    struct DriftingSource {
        replay: Cell<u32>,
        first: Vec<LodgeGaussianId>,
        second: Vec<LodgeGaussianId>,
    }

    impl ReplayableLodgeMembershipSource for DriftingSource {
        fn replay(
            &self,
            _batch_ids: usize,
            consume: &mut dyn FnMut(&[LodgeGaussianId]) -> Result<(), LodgeMembershipBuildError>,
        ) -> Result<u64, LodgeMembershipBuildError> {
            let replay = self.replay.get();
            self.replay.set(replay + 1);
            let values = if replay == 0 {
                &self.first
            } else {
                &self.second
            };
            consume(values)?;
            Ok(values.len() as u64)
        }
    }

    #[test]
    fn replay_drift_is_rejected() {
        let source = DriftingSource {
            replay: Cell::new(0),
            first: ids(&[1, 3, 5]),
            second: ids(&[1, 4, 5]),
        };
        let stable_ids = ids(&[2]);
        let stable = LodgeMembershipSliceSource::new(&stable_ids);
        assert!(matches!(
            build_canonical_lodge_membership_artifact(
                "members.bgslmem",
                5,
                &[
                    LodgeClusterMembershipInput::new(LodgeClusterId(1), &source),
                    LodgeClusterMembershipInput::new(LodgeClusterId(2), &stable),
                ],
                LodgeMembershipArtifactConfig::default()
            ),
            Err(LodgeMembershipBuildError::ReplayDrift {
                cluster: LodgeClusterId(1)
            })
        ));
    }

    #[test]
    fn cluster_and_membership_order_are_rejected() {
        let duplicate_ids = ids(&[1, 1]);
        let duplicate = LodgeMembershipSliceSource::new(&duplicate_ids);
        let valid_ids = ids(&[2]);
        let valid = LodgeMembershipSliceSource::new(&valid_ids);
        assert!(matches!(
            build_canonical_lodge_membership_artifact(
                "members.bgslmem",
                2,
                &[
                    LodgeClusterMembershipInput::new(LodgeClusterId(1), &duplicate),
                    LodgeClusterMembershipInput::new(LodgeClusterId(2), &valid),
                ],
                LodgeMembershipArtifactConfig::default()
            ),
            Err(LodgeMembershipBuildError::InvalidMembershipOrder { .. })
        ));

        let values = ids(&[1]);
        let source = LodgeMembershipSliceSource::new(&values);
        assert_eq!(
            build_canonical_lodge_membership_artifact(
                "members.bgslmem",
                2,
                &[LodgeClusterMembershipInput::new(LodgeClusterId(1), &source,)],
                LodgeMembershipArtifactConfig::default(),
            ),
            Err(LodgeMembershipBuildError::InsufficientClusters { actual: 1 })
        );
        assert!(matches!(
            build_canonical_lodge_membership_artifact(
                "members.bgslmem",
                2,
                &[
                    LodgeClusterMembershipInput::new(LodgeClusterId(2), &source),
                    LodgeClusterMembershipInput::new(LodgeClusterId(1), &source),
                ],
                LodgeMembershipArtifactConfig::default()
            ),
            Err(LodgeMembershipBuildError::InvalidClusterOrder { index: 1, .. })
        ));

        for uri in [
            "https:members.bgslmem",
            "file:members.bgslmem",
            "C:members.bgslmem",
        ] {
            assert_eq!(
                build_canonical_lodge_membership_artifact(
                    uri,
                    2,
                    &[
                        LodgeClusterMembershipInput::new(LodgeClusterId(1), &source),
                        LodgeClusterMembershipInput::new(LodgeClusterId(2), &source),
                    ],
                    LodgeMembershipArtifactConfig::default(),
                ),
                Err(LodgeMembershipBuildError::InvalidUri),
                "accepted scheme-like URI {uri}",
            );
        }
    }

    #[test]
    fn configured_limits_apply_before_object_allocation() {
        let values = ids(&[1, 2, 3]);
        let source = LodgeMembershipSliceSource::new(&values);
        let mut limits = LodgeCodecLimits::default();
        limits.max_members_per_cluster = 2;
        assert_eq!(
            build_canonical_lodge_membership_artifact(
                "members.bgslmem",
                3,
                &[
                    LodgeClusterMembershipInput::new(LodgeClusterId(1), &source),
                    LodgeClusterMembershipInput::new(LodgeClusterId(2), &source),
                ],
                LodgeMembershipArtifactConfig {
                    replay_batch_ids: 2,
                    limits,
                }
            ),
            Err(LodgeMembershipBuildError::LimitExceeded {
                field: "members per cluster",
                actual: 3,
                limit: 2,
            })
        );

        let mut limits = LodgeCodecLimits::default();
        limits.max_dependency_bytes = 64;
        assert!(matches!(
            build_canonical_lodge_membership_artifact(
                "members.bgslmem",
                3,
                &[
                    LodgeClusterMembershipInput::new(LodgeClusterId(1), &source),
                    LodgeClusterMembershipInput::new(LodgeClusterId(2), &source),
                ],
                LodgeMembershipArtifactConfig {
                    replay_batch_ids: 2,
                    limits,
                }
            ),
            Err(LodgeMembershipBuildError::LimitExceeded {
                field: "membership object bytes",
                ..
            })
        ));
    }
}
