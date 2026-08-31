//! Authenticated, bounded `.gslodge` sidecar codec and Bevy asset loader.

use std::{error::Error, fmt, sync::Arc};

use bevy::{
    asset::{AssetLoader, AsyncReadExt, LoadContext, io::Reader},
    prelude::*,
    reflect::TypePath,
};
use serde::{
    Deserialize, Serialize,
    de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor},
};
use sha2::{Digest, Sha256};

use crate::gaussian::formats::lodge::{
    GaussianLodgeManifest, LODGE_MANIFEST_VERSION, LodgeAuthenticatedObject, LodgeGaussianId,
    LodgeMembershipEntry, LodgePageAuthentication, LodgeValidationError,
};
use crate::gaussian::formats::planar_3d_chunked::LodPageDescriptor;

pub const LODGE_CONTAINER_MAGIC: [u8; 8] = *b"BGSLODGE";
pub const LODGE_CONTAINER_VERSION: u16 = 1;
/// Fixed v1 prefix. Counts in this prefix gate every allocation-driving
/// collection before the authenticated payload reaches serde.
pub const LODGE_HEADER_LEN: usize = 120;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum LodgeManifestEncoding {
    Flexbuffers = 1,
    Json = 2,
}

impl TryFrom<u8> for LodgeManifestEncoding {
    type Error = LodgeCodecError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Flexbuffers),
            2 => Ok(Self::Json),
            other => Err(LodgeCodecError::UnsupportedEncoding(other)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodgeCodecLimits {
    pub max_manifest_bytes: u64,
    pub max_levels: u32,
    pub max_clusters: u32,
    pub max_record_runs: u32,
    pub max_extra_pages: u32,
    pub max_page_authentications: u32,
    pub max_neighbors: u32,
    pub max_stable_gaussians: u64,
    pub max_total_membership_ids: u64,
    pub max_members_per_cluster: u64,
    /// Maximum for one authenticated dependency and for the checked aggregate
    /// of sidecar-declared base-manifest, membership-object, and extra-page
    /// encoded lengths. Base-package page lengths are checked after the base
    /// manifest is authenticated.
    pub max_dependency_bytes: u64,
    pub max_membership_stream_bytes: u64,
}

impl Default for LodgeCodecLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: Self::DEFAULT_MAX_MANIFEST_BYTES,
            max_levels: Self::DEFAULT_MAX_LEVELS,
            max_clusters: Self::DEFAULT_MAX_CLUSTERS,
            max_record_runs: Self::DEFAULT_MAX_RECORD_RUNS,
            max_extra_pages: Self::DEFAULT_MAX_EXTRA_PAGES,
            max_page_authentications: Self::DEFAULT_MAX_PAGE_AUTHENTICATIONS,
            max_neighbors: Self::DEFAULT_MAX_NEIGHBORS,
            max_stable_gaussians: Self::DEFAULT_MAX_STABLE_GAUSSIANS,
            max_total_membership_ids: Self::DEFAULT_MAX_TOTAL_MEMBERSHIP_IDS,
            max_members_per_cluster: Self::DEFAULT_MAX_MEMBERS_PER_CLUSTER,
            max_dependency_bytes: Self::DEFAULT_MAX_DEPENDENCY_BYTES,
            max_membership_stream_bytes: Self::DEFAULT_MAX_MEMBERSHIP_STREAM_BYTES,
        }
    }
}

impl LodgeCodecLimits {
    pub const DEFAULT_MAX_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;
    pub const DEFAULT_MAX_LEVELS: u32 = 64;
    pub const DEFAULT_MAX_CLUSTERS: u32 = 65_536;
    pub const DEFAULT_MAX_RECORD_RUNS: u32 = 1_048_576;
    pub const DEFAULT_MAX_EXTRA_PAGES: u32 = 262_144;
    pub const DEFAULT_MAX_PAGE_AUTHENTICATIONS: u32 = 524_288;
    pub const DEFAULT_MAX_NEIGHBORS: u32 = 4_194_304;
    pub const DEFAULT_MAX_STABLE_GAUSSIANS: u64 = 4_000_000_000;
    pub const DEFAULT_MAX_TOTAL_MEMBERSHIP_IDS: u64 = 16_000_000_000;
    pub const DEFAULT_MAX_MEMBERS_PER_CLUSTER: u64 = 1_000_000_000;
    pub const DEFAULT_MAX_DEPENDENCY_BYTES: u64 = 1 << 40;
    pub const DEFAULT_MAX_MEMBERSHIP_STREAM_BYTES: u64 = 512 * 1024 * 1024;

    pub fn validate(self) -> Result<Self, LodgeCodecError> {
        if self.max_manifest_bytes < LODGE_HEADER_LEN as u64
            || self.max_levels < 2
            || self.max_clusters == 0
            || self.max_record_runs == 0
            || self.max_extra_pages == 0
            || self.max_page_authentications == 0
            || self.max_neighbors == 0
            || self.max_stable_gaussians == 0
            || self.max_total_membership_ids == 0
            || self.max_members_per_cluster == 0
            || self.max_dependency_bytes == 0
            || self.max_membership_stream_bytes == 0
        {
            Err(LodgeCodecError::InvalidLimits)
        } else {
            Ok(self)
        }
    }
}

#[derive(Asset, Clone, Debug, TypePath)]
pub struct GaussianLodgeAsset {
    manifest: Arc<GaussianLodgeManifest>,
}

impl GaussianLodgeAsset {
    pub fn new(manifest: GaussianLodgeManifest) -> Result<Self, LodgeValidationError> {
        manifest.validate()?;
        Ok(Self::from_validated_manifest(manifest))
    }

    pub fn manifest(&self) -> &GaussianLodgeManifest {
        self.manifest.as_ref()
    }

    pub fn shared_manifest(&self) -> Arc<GaussianLodgeManifest> {
        Arc::clone(&self.manifest)
    }

    /// Authenticates and semantically binds the companion `.gsplatlod`
    /// manifest before any LODGE record run is resolved against it.
    ///
    /// SHA-256 proves byte identity relative to this sidecar; applications are
    /// still responsible for deciding which sidecar roots they trust.
    pub fn validate_base_dependency(
        &self,
        encoded_base_manifest: &[u8],
        base_manifest: &crate::gaussian::formats::planar_3d_lod::GaussianLodManifest,
        max_encoded_bytes: u64,
    ) -> Result<(), LodgeCodecError> {
        verify_lodge_authenticated_object(
            encoded_base_manifest,
            &self.manifest.base_manifest,
            max_encoded_bytes,
        )?;
        self.manifest
            .validate_against_base(base_manifest)
            .map_err(LodgeCodecError::ManifestValidation)
    }

    fn from_validated_manifest(manifest: GaussianLodgeManifest) -> Self {
        Self {
            manifest: Arc::new(manifest),
        }
    }
}

#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct GaussianLodgeHandle(pub Handle<GaussianLodgeAsset>);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GaussianLodgeManifestLoaderSettings {
    pub max_encoded_bytes: u64,
    pub max_levels: u32,
    pub max_clusters: u32,
    pub max_record_runs: u32,
    pub max_extra_pages: u32,
    pub max_page_authentications: u32,
    pub max_neighbors: u32,
    pub max_stable_gaussians: u64,
    pub max_total_membership_ids: u64,
    pub max_members_per_cluster: u64,
    pub max_dependency_bytes: u64,
    pub max_membership_stream_bytes: u64,
}

impl Default for GaussianLodgeManifestLoaderSettings {
    fn default() -> Self {
        let limits = LodgeCodecLimits::default();
        Self {
            max_encoded_bytes: limits.max_manifest_bytes,
            max_levels: limits.max_levels,
            max_clusters: limits.max_clusters,
            max_record_runs: limits.max_record_runs,
            max_extra_pages: limits.max_extra_pages,
            max_page_authentications: limits.max_page_authentications,
            max_neighbors: limits.max_neighbors,
            max_stable_gaussians: limits.max_stable_gaussians,
            max_total_membership_ids: limits.max_total_membership_ids,
            max_members_per_cluster: limits.max_members_per_cluster,
            max_dependency_bytes: limits.max_dependency_bytes,
            max_membership_stream_bytes: limits.max_membership_stream_bytes,
        }
    }
}

impl From<&GaussianLodgeManifestLoaderSettings> for LodgeCodecLimits {
    fn from(settings: &GaussianLodgeManifestLoaderSettings) -> Self {
        Self {
            max_manifest_bytes: settings.max_encoded_bytes,
            max_levels: settings.max_levels,
            max_clusters: settings.max_clusters,
            max_record_runs: settings.max_record_runs,
            max_extra_pages: settings.max_extra_pages,
            max_page_authentications: settings.max_page_authentications,
            max_neighbors: settings.max_neighbors,
            max_stable_gaussians: settings.max_stable_gaussians,
            max_total_membership_ids: settings.max_total_membership_ids,
            max_members_per_cluster: settings.max_members_per_cluster,
            max_dependency_bytes: settings.max_dependency_bytes,
            max_membership_stream_bytes: settings.max_membership_stream_bytes,
        }
    }
}

#[derive(Default, TypePath)]
pub struct GaussianLodgeManifestLoader;

impl AssetLoader for GaussianLodgeManifestLoader {
    type Asset = GaussianLodgeAsset;
    type Settings = GaussianLodgeManifestLoaderSettings;
    type Error = LodgeAssetLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &Self::Settings,
        _: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let bytes = read_bounded(reader, settings.max_encoded_bytes).await?;
        let manifest = decode_lodge_manifest(&bytes, settings.into())?;
        Ok(GaussianLodgeAsset::from_validated_manifest(manifest))
    }

    fn extensions(&self) -> &[&str] {
        &["gslodge"]
    }
}

async fn read_bounded(
    reader: &mut dyn Reader,
    max_encoded_bytes: u64,
) -> Result<Vec<u8>, LodgeAssetLoaderError> {
    let max = usize::try_from(max_encoded_bytes).map_err(|_| LodgeCodecError::InvalidLimits)?;
    let probe_limit = max_encoded_bytes
        .checked_add(1)
        .ok_or(LodgeCodecError::InvalidLimits)?;
    let mut bytes = Vec::with_capacity(max.min(1024 * 1024));
    let mut bounded = reader.take(probe_limit);
    bounded.read_to_end(&mut bytes).await?;
    if bytes.len() > max {
        return Err(LodgeCodecError::LimitExceeded {
            field: "encoded bytes",
            actual: bytes.len() as u64,
            limit: max_encoded_bytes,
        }
        .into());
    }
    Ok(bytes)
}

pub fn encode_lodge_manifest(manifest: &GaussianLodgeManifest) -> Result<Vec<u8>, LodgeCodecError> {
    #[cfg(feature = "io_flexbuffers")]
    let encoding = LodgeManifestEncoding::Flexbuffers;
    #[cfg(not(feature = "io_flexbuffers"))]
    let encoding = LodgeManifestEncoding::Json;
    encode_lodge_manifest_with_encoding(manifest, encoding)
}

pub fn encode_lodge_manifest_with_encoding(
    manifest: &GaussianLodgeManifest,
    encoding: LodgeManifestEncoding,
) -> Result<Vec<u8>, LodgeCodecError> {
    manifest
        .validate()
        .map_err(LodgeCodecError::ManifestValidation)?;
    let payload = match encoding {
        LodgeManifestEncoding::Flexbuffers => {
            #[cfg(feature = "io_flexbuffers")]
            {
                let mut serializer = flexbuffers::FlexbufferSerializer::new();
                manifest
                    .serialize(&mut serializer)
                    .map_err(|error| LodgeCodecError::Serialize(error.to_string()))?;
                serializer.view().to_vec()
            }
            #[cfg(not(feature = "io_flexbuffers"))]
            {
                return Err(LodgeCodecError::EncodingUnavailable(encoding));
            }
        }
        LodgeManifestEncoding::Json => serde_json::to_vec(manifest)
            .map_err(|error| LodgeCodecError::Serialize(error.to_string()))?,
    };

    let payload_len = u64::try_from(payload.len()).map_err(|_| LodgeCodecError::LengthOverflow)?;
    let payload_sha256 = sha256_bytes(&payload);
    let mut encoded = Vec::with_capacity(LODGE_HEADER_LEN + payload.len());
    encoded.extend_from_slice(&LODGE_CONTAINER_MAGIC);
    encoded.extend_from_slice(&LODGE_CONTAINER_VERSION.to_le_bytes());
    encoded.extend_from_slice(&LODGE_MANIFEST_VERSION.to_le_bytes());
    encoded.push(encoding as u8);
    encoded.push(0); // flags
    encoded.extend_from_slice(&0_u16.to_le_bytes());
    encoded.extend_from_slice(&payload_len.to_le_bytes());
    encoded.extend_from_slice(&payload_sha256);
    encoded.extend_from_slice(&manifest.base_manifest.sha256);
    encoded.extend_from_slice(&manifest.header.level_count.to_le_bytes());
    encoded.extend_from_slice(&manifest.header.cluster_count.to_le_bytes());
    encoded.extend_from_slice(&manifest.header.record_run_count.to_le_bytes());
    encoded.extend_from_slice(&manifest.header.extra_page_count.to_le_bytes());
    let authentication_count = u32::try_from(manifest.page_authentication.len())
        .map_err(|_| LodgeCodecError::LengthOverflow)?;
    encoded.extend_from_slice(&authentication_count.to_le_bytes());
    encoded.extend_from_slice(&manifest.header.neighbor_count.to_le_bytes());
    encoded.extend_from_slice(&manifest.header.cluster_count.to_le_bytes());
    encoded.extend_from_slice(&0_u32.to_le_bytes());
    debug_assert_eq!(encoded.len(), LODGE_HEADER_LEN);
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

pub fn decode_lodge_manifest(
    encoded: &[u8],
    limits: LodgeCodecLimits,
) -> Result<GaussianLodgeManifest, LodgeCodecError> {
    let limits = limits.validate()?;
    enforce_limit(
        "manifest bytes",
        encoded.len() as u64,
        limits.max_manifest_bytes,
    )?;
    if encoded.len() < 12 {
        return Err(LodgeCodecError::Truncated("manifest header prefix"));
    }
    if encoded[..8] != LODGE_CONTAINER_MAGIC {
        return Err(LodgeCodecError::InvalidMagic);
    }
    let container_version = read_u16(encoded, 8)?;
    if container_version != LODGE_CONTAINER_VERSION {
        return Err(LodgeCodecError::UnsupportedContainerVersion(
            container_version,
        ));
    }
    let semantic_version = read_u16(encoded, 10)?;
    if semantic_version != LODGE_MANIFEST_VERSION {
        return Err(LodgeCodecError::UnsupportedSemanticVersion(
            semantic_version,
        ));
    }
    if encoded.len() < LODGE_HEADER_LEN {
        return Err(LodgeCodecError::Truncated("manifest header"));
    }
    if encoded[13..16].iter().any(|byte| *byte != 0)
        || encoded[116..120].iter().any(|byte| *byte != 0)
    {
        return Err(LodgeCodecError::NonZeroReservedBytes);
    }
    let encoding = LodgeManifestEncoding::try_from(encoded[12])?;
    let payload_len = read_u64(encoded, 16)?;
    let expected_payload_sha256 = read_hash(encoded, 24)?;
    let expected_base_sha256 = read_hash(encoded, 56)?;
    let envelope = LodgeEnvelopeCounts {
        levels: read_u32(encoded, 88)?,
        clusters: read_u32(encoded, 92)?,
        record_runs: read_u32(encoded, 96)?,
        extra_pages: read_u32(encoded, 100)?,
        page_authentications: read_u32(encoded, 104)?,
        neighbors: read_u32(encoded, 108)?,
        membership_entries: read_u32(encoded, 112)?,
    };
    envelope.enforce(limits)?;

    let payload_len = usize::try_from(payload_len).map_err(|_| LodgeCodecError::LengthOverflow)?;
    let expected_len = LODGE_HEADER_LEN
        .checked_add(payload_len)
        .ok_or(LodgeCodecError::LengthOverflow)?;
    if encoded.len() != expected_len {
        return Err(LodgeCodecError::LengthMismatch {
            expected: expected_len as u64,
            actual: encoded.len() as u64,
        });
    }
    let payload = &encoded[LODGE_HEADER_LEN..];
    let actual_payload_sha256 = sha256_bytes(payload);
    if actual_payload_sha256 != expected_payload_sha256 {
        return Err(LodgeCodecError::Sha256Mismatch("manifest payload"));
    }

    let manifest = match encoding {
        LodgeManifestEncoding::Flexbuffers => {
            #[cfg(feature = "io_flexbuffers")]
            {
                let reader = flexbuffers::Reader::get_root(payload)
                    .map_err(|error| LodgeCodecError::Deserialize(error.to_string()))?;
                let map = reader
                    .get_map()
                    .map_err(|error| LodgeCodecError::Deserialize(error.to_string()))?;
                validate_flexbuffer_collection_limits(&map, limits)?;
                GaussianLodgeManifest::deserialize(reader)
                    .map_err(|error| LodgeCodecError::Deserialize(error.to_string()))?
            }
            #[cfg(not(feature = "io_flexbuffers"))]
            {
                return Err(LodgeCodecError::EncodingUnavailable(encoding));
            }
        }
        LodgeManifestEncoding::Json => {
            validate_json_collection_limits(payload, limits)?;
            serde_json::from_slice(payload)
                .map_err(|error| LodgeCodecError::Deserialize(error.to_string()))?
        }
    };

    validate_decoded_limits(&manifest, limits)?;
    envelope.check_manifest(&manifest)?;
    if manifest.base_manifest.sha256 != expected_base_sha256 {
        return Err(LodgeCodecError::Sha256Mismatch("base manifest identity"));
    }
    manifest
        .validate()
        .map_err(LodgeCodecError::ManifestValidation)?;
    Ok(manifest)
}

/// SHA-256 digest used by sidecar, dependency, page, index, and stream checks.
pub fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Verifies an exact external object against its bounded authenticated
/// descriptor before any object-specific parser observes the bytes.
pub fn verify_lodge_authenticated_object(
    encoded: &[u8],
    descriptor: &LodgeAuthenticatedObject,
    max_encoded_bytes: u64,
) -> Result<(), LodgeCodecError> {
    enforce_limit(
        "authenticated object bytes",
        encoded.len() as u64,
        max_encoded_bytes,
    )?;
    if descriptor.encoded_len != encoded.len() as u64 {
        return Err(LodgeCodecError::LengthMismatch {
            expected: descriptor.encoded_len,
            actual: encoded.len() as u64,
        });
    }
    if sha256_bytes(encoded) != descriptor.sha256 {
        return Err(LodgeCodecError::Sha256Mismatch("authenticated object"));
    }
    Ok(())
}

/// Verifies exact encoded page bytes before handing them to the existing
/// ordinary page decoder.
pub fn verify_lodge_page_bytes(
    encoded: &[u8],
    descriptor: &LodPageDescriptor,
    authentication: &LodgePageAuthentication,
    max_encoded_bytes: u64,
) -> Result<(), LodgeCodecError> {
    if descriptor.id != authentication.page {
        return Err(LodgeCodecError::PageAuthenticationMismatch);
    }
    enforce_limit(
        "authenticated page bytes",
        encoded.len() as u64,
        max_encoded_bytes,
    )?;
    let expected_len = descriptor
        .storage
        .as_ref()
        .map(|storage| storage.encoded_len)
        .ok_or(LodgeCodecError::MissingPageStorage)?;
    if expected_len != encoded.len() as u64 {
        return Err(LodgeCodecError::LengthMismatch {
            expected: expected_len,
            actual: encoded.len() as u64,
        });
    }
    if sha256_bytes(encoded) != authentication.encoded_sha256 {
        return Err(LodgeCodecError::Sha256Mismatch("page"));
    }
    Ok(())
}

/// Encodes a nonempty, strictly increasing stable-ID set as canonical unsigned
/// LEB128 deltas. Gaussian pages are neither needed nor touched.
pub fn encode_lodge_membership_ids(ids: &[LodgeGaussianId]) -> Result<Vec<u8>, LodgeCodecError> {
    if ids.is_empty() {
        return Err(LodgeCodecError::EmptyMembership);
    }
    let mut previous = 0_u64;
    let mut encoded = Vec::with_capacity(ids.len());
    for (index, id) in ids.iter().copied().enumerate() {
        let delta =
            id.0.checked_sub(previous)
                .filter(|delta| *delta != 0)
                .ok_or(LodgeCodecError::InvalidMembershipId { index })?;
        encode_uleb128(delta, &mut encoded);
        previous = id.0;
    }
    Ok(encoded)
}

/// Decodes one bounded membership stream into sorted stable IDs without
/// loading or decoding any Gaussian page.
pub fn decode_lodge_membership_ids(
    encoded: &[u8],
    expected_count: u64,
    stable_gaussian_count: u64,
    limits: LodgeCodecLimits,
) -> Result<Vec<LodgeGaussianId>, LodgeCodecError> {
    let limits = limits.validate()?;
    if expected_count == 0 {
        return Err(LodgeCodecError::EmptyMembership);
    }
    enforce_limit(
        "membership IDs",
        expected_count,
        limits.max_members_per_cluster,
    )?;
    enforce_limit(
        "membership stream bytes",
        encoded.len() as u64,
        limits.max_membership_stream_bytes,
    )?;
    if stable_gaussian_count == 0 || stable_gaussian_count > limits.max_stable_gaussians {
        return Err(LodgeCodecError::LimitExceeded {
            field: "stable Gaussians",
            actual: stable_gaussian_count,
            limit: limits.max_stable_gaussians,
        });
    }
    let capacity = usize::try_from(expected_count).map_err(|_| LodgeCodecError::LengthOverflow)?;
    let mut ids = Vec::with_capacity(capacity);
    let mut cursor = 0_usize;
    let mut previous = 0_u64;
    while cursor < encoded.len() {
        if ids.len() == capacity {
            return Err(LodgeCodecError::MembershipCountMismatch {
                expected: expected_count,
                actual: ids.len() as u64 + 1,
            });
        }
        let delta = decode_uleb128(encoded, &mut cursor)?;
        if delta == 0 {
            return Err(LodgeCodecError::NonCanonicalMembership);
        }
        let id = previous
            .checked_add(delta)
            .ok_or(LodgeCodecError::MembershipIdOverflow)?;
        if id > stable_gaussian_count {
            return Err(LodgeCodecError::MembershipIdOutOfRange {
                id,
                stable_gaussian_count,
            });
        }
        ids.push(LodgeGaussianId(id));
        previous = id;
    }
    if ids.len() as u64 != expected_count {
        return Err(LodgeCodecError::MembershipCountMismatch {
            expected: expected_count,
            actual: ids.len() as u64,
        });
    }
    Ok(ids)
}

/// Verifies an entry's range length/digest and semantic endpoints while
/// decoding its stable-ID stream.
pub fn decode_lodge_membership_entry(
    encoded_range: &[u8],
    entry: &LodgeMembershipEntry,
    stable_gaussian_count: u64,
    limits: LodgeCodecLimits,
) -> Result<Vec<LodgeGaussianId>, LodgeCodecError> {
    if entry.byte_range.1 != encoded_range.len() as u64 {
        return Err(LodgeCodecError::LengthMismatch {
            expected: entry.byte_range.1,
            actual: encoded_range.len() as u64,
        });
    }
    if sha256_bytes(encoded_range) != entry.encoded_sha256 {
        return Err(LodgeCodecError::Sha256Mismatch("membership stream"));
    }
    let ids = decode_lodge_membership_ids(
        encoded_range,
        entry.member_count,
        stable_gaussian_count,
        limits,
    )?;
    if ids.first().copied() != Some(entry.first_id) || ids.last().copied() != Some(entry.last_id) {
        return Err(LodgeCodecError::MembershipEndpointMismatch);
    }
    Ok(ids)
}

fn encode_uleb128(mut value: u64, output: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn decode_uleb128(bytes: &[u8], cursor: &mut usize) -> Result<u64, LodgeCodecError> {
    let start = *cursor;
    let mut value = 0_u64;
    for byte_index in 0..10_u32 {
        let byte = *bytes
            .get(*cursor)
            .ok_or(LodgeCodecError::Truncated("membership ULEB128"))?;
        *cursor += 1;
        let payload = u64::from(byte & 0x7f);
        if byte_index == 9 && payload > 1 {
            return Err(LodgeCodecError::MembershipIdOverflow);
        }
        value |= payload << (byte_index * 7);
        if byte & 0x80 == 0 {
            if *cursor - start > 1 && payload == 0 {
                return Err(LodgeCodecError::NonCanonicalMembership);
            }
            return Ok(value);
        }
    }
    Err(LodgeCodecError::MembershipIdOverflow)
}

#[derive(Clone, Copy, Debug)]
struct LodgeEnvelopeCounts {
    levels: u32,
    clusters: u32,
    record_runs: u32,
    extra_pages: u32,
    page_authentications: u32,
    neighbors: u32,
    membership_entries: u32,
}

impl LodgeEnvelopeCounts {
    fn enforce(self, limits: LodgeCodecLimits) -> Result<(), LodgeCodecError> {
        enforce_limit("levels", self.levels.into(), limits.max_levels.into())?;
        enforce_limit("clusters", self.clusters.into(), limits.max_clusters.into())?;
        enforce_limit(
            "record runs",
            self.record_runs.into(),
            limits.max_record_runs.into(),
        )?;
        enforce_limit(
            "extra pages",
            self.extra_pages.into(),
            limits.max_extra_pages.into(),
        )?;
        enforce_limit(
            "page authentications",
            self.page_authentications.into(),
            limits.max_page_authentications.into(),
        )?;
        enforce_limit(
            "neighbors",
            self.neighbors.into(),
            limits.max_neighbors.into(),
        )?;
        enforce_limit(
            "membership entries",
            self.membership_entries.into(),
            limits.max_clusters.into(),
        )
    }

    fn check_manifest(self, manifest: &GaussianLodgeManifest) -> Result<(), LodgeCodecError> {
        check_envelope_count(
            "levels",
            self.levels,
            manifest.header.level_count,
            manifest.levels.len(),
        )?;
        check_envelope_count(
            "clusters",
            self.clusters,
            manifest.header.cluster_count,
            manifest.clusters.len(),
        )?;
        check_envelope_count(
            "record runs",
            self.record_runs,
            manifest.header.record_run_count,
            manifest.record_runs.len(),
        )?;
        check_envelope_count(
            "extra pages",
            self.extra_pages,
            manifest.header.extra_page_count,
            manifest.extra_pages.len(),
        )?;
        check_envelope_count(
            "page authentications",
            self.page_authentications,
            manifest
                .header
                .base_page_count
                .checked_add(manifest.header.extra_page_count)
                .ok_or(LodgeCodecError::LengthOverflow)?,
            manifest.page_authentication.len(),
        )?;
        check_envelope_count(
            "neighbors",
            self.neighbors,
            manifest.header.neighbor_count,
            manifest.neighbors.len(),
        )?;
        check_envelope_count(
            "membership entries",
            self.membership_entries,
            manifest.header.cluster_count,
            manifest.membership_index.entries.len(),
        )
    }
}

fn validate_decoded_limits(
    manifest: &GaussianLodgeManifest,
    limits: LodgeCodecLimits,
) -> Result<(), LodgeCodecError> {
    let counts = LodgeEnvelopeCounts {
        levels: usize_to_u32(manifest.levels.len())?,
        clusters: usize_to_u32(manifest.clusters.len())?,
        record_runs: usize_to_u32(manifest.record_runs.len())?,
        extra_pages: usize_to_u32(manifest.extra_pages.len())?,
        page_authentications: usize_to_u32(manifest.page_authentication.len())?,
        neighbors: usize_to_u32(manifest.neighbors.len())?,
        membership_entries: usize_to_u32(manifest.membership_index.entries.len())?,
    };
    counts.enforce(limits)?;
    enforce_limit(
        "stable Gaussians",
        manifest.header.stable_gaussian_count,
        limits.max_stable_gaussians,
    )?;
    enforce_limit(
        "total membership IDs",
        manifest.header.total_membership_ids,
        limits.max_total_membership_ids,
    )?;
    enforce_limit(
        "base manifest bytes",
        manifest.base_manifest.encoded_len,
        limits.max_dependency_bytes,
    )?;
    enforce_limit(
        "membership object bytes",
        manifest.membership_index.object.encoded_len,
        limits.max_dependency_bytes,
    )?;
    let mut declared_dependency_bytes = manifest
        .base_manifest
        .encoded_len
        .checked_add(manifest.membership_index.object.encoded_len)
        .ok_or(LodgeCodecError::LengthOverflow)?;
    for page in &manifest.extra_pages {
        let encoded_len = page
            .storage
            .as_ref()
            .ok_or(LodgeCodecError::MissingPageStorage)?
            .encoded_len;
        enforce_limit("extra page bytes", encoded_len, limits.max_dependency_bytes)?;
        declared_dependency_bytes = declared_dependency_bytes
            .checked_add(encoded_len)
            .ok_or(LodgeCodecError::LengthOverflow)?;
    }
    enforce_limit(
        "declared dependency bytes",
        declared_dependency_bytes,
        limits.max_dependency_bytes,
    )?;
    for entry in &manifest.membership_index.entries {
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
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum LodgeCollection {
    Levels,
    Clusters,
    RecordRuns,
    ExtraPages,
    PageAuthentications,
    Neighbors,
    MembershipEntries,
}

impl LodgeCollection {
    const fn field(self) -> &'static str {
        match self {
            Self::Levels => "levels",
            Self::Clusters => "clusters",
            Self::RecordRuns => "record runs",
            Self::ExtraPages => "extra pages",
            Self::PageAuthentications => "page authentications",
            Self::Neighbors => "neighbors",
            Self::MembershipEntries => "membership entries",
        }
    }

    const fn limit(self, limits: LodgeCodecLimits) -> u64 {
        match self {
            Self::Levels => limits.max_levels as u64,
            Self::Clusters | Self::MembershipEntries => limits.max_clusters as u64,
            Self::RecordRuns => limits.max_record_runs as u64,
            Self::ExtraPages => limits.max_extra_pages as u64,
            Self::PageAuthentications => limits.max_page_authentications as u64,
            Self::Neighbors => limits.max_neighbors as u64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct LodgeCollectionOverflow {
    collection: LodgeCollection,
    actual: u64,
    limit: u64,
}

#[cfg(feature = "io_flexbuffers")]
fn validate_flexbuffer_collection_limits(
    map: &flexbuffers::MapReader<&[u8]>,
    limits: LodgeCodecLimits,
) -> Result<(), LodgeCodecError> {
    validate_flexbuffer_map_keys(map)?;
    let keys = map.keys_vector();
    for index in 0..map.len() {
        let key = keys
            .index(index)
            .and_then(|reader| reader.get_key())
            .map_err(|error| LodgeCodecError::Deserialize(error.to_string()))?;
        let collection = match key {
            "levels" => Some(LodgeCollection::Levels),
            "clusters" => Some(LodgeCollection::Clusters),
            "record_runs" => Some(LodgeCollection::RecordRuns),
            "extra_pages" => Some(LodgeCollection::ExtraPages),
            "page_authentication" => Some(LodgeCollection::PageAuthentications),
            "neighbors" => Some(LodgeCollection::Neighbors),
            "membership_index" => {
                let membership = map
                    .index(index)
                    .and_then(|reader| reader.get_map())
                    .map_err(|error| LodgeCodecError::Deserialize(error.to_string()))?;
                validate_flexbuffer_membership_entries(&membership, limits)?;
                None
            }
            _ => None,
        };
        if let Some(collection) = collection {
            let values = map
                .index(index)
                .and_then(|reader| reader.get_vector())
                .map_err(|error| LodgeCodecError::Deserialize(error.to_string()))?;
            enforce_limit(
                collection.field(),
                values.len() as u64,
                collection.limit(limits),
            )?;
        }
    }
    Ok(())
}

#[cfg(feature = "io_flexbuffers")]
fn validate_flexbuffer_membership_entries(
    map: &flexbuffers::MapReader<&[u8]>,
    limits: LodgeCodecLimits,
) -> Result<(), LodgeCodecError> {
    validate_flexbuffer_map_keys(map)?;
    let keys = map.keys_vector();
    for index in 0..map.len() {
        let key = keys
            .index(index)
            .and_then(|reader| reader.get_key())
            .map_err(|error| LodgeCodecError::Deserialize(error.to_string()))?;
        if key == "entries" {
            let entries = map
                .index(index)
                .and_then(|reader| reader.get_vector())
                .map_err(|error| LodgeCodecError::Deserialize(error.to_string()))?;
            enforce_limit(
                LodgeCollection::MembershipEntries.field(),
                entries.len() as u64,
                LodgeCollection::MembershipEntries.limit(limits),
            )?;
        }
    }
    Ok(())
}

#[cfg(feature = "io_flexbuffers")]
fn validate_flexbuffer_map_keys(
    map: &flexbuffers::MapReader<&[u8]>,
) -> Result<(), LodgeCodecError> {
    let keys = map.keys_vector();
    let mut previous_key = None;
    for index in 0..map.len() {
        let key = keys
            .index(index)
            .and_then(|reader| reader.get_key())
            .map_err(|error| LodgeCodecError::Deserialize(error.to_string()))?;
        if previous_key.is_some_and(|previous| previous >= key) {
            return Err(LodgeCodecError::InvalidManifestMapKeys);
        }
        previous_key = Some(key);
    }
    Ok(())
}

struct LodgeShapeSeed<'a> {
    limits: LodgeCodecLimits,
    overflow: &'a std::cell::Cell<Option<LodgeCollectionOverflow>>,
}

impl<'de> DeserializeSeed<'de> for LodgeShapeSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(LodgeShapeVisitor {
            limits: self.limits,
            overflow: self.overflow,
        })
    }
}

struct LodgeShapeVisitor<'a> {
    limits: LodgeCodecLimits,
    overflow: &'a std::cell::Cell<Option<LodgeCollectionOverflow>>,
}

impl<'de> Visitor<'de> for LodgeShapeVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a LODGE manifest map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while let Some(field) = map.next_key::<String>()? {
            let collection = match field.as_str() {
                "levels" => Some(LodgeCollection::Levels),
                "clusters" => Some(LodgeCollection::Clusters),
                "record_runs" => Some(LodgeCollection::RecordRuns),
                "extra_pages" => Some(LodgeCollection::ExtraPages),
                "page_authentication" => Some(LodgeCollection::PageAuthentications),
                "neighbors" => Some(LodgeCollection::Neighbors),
                "membership_index" => {
                    map.next_value_seed(MembershipShapeSeed {
                        limits: self.limits,
                        overflow: self.overflow,
                    })?;
                    None
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                    None
                }
            };
            if let Some(collection) = collection {
                map.next_value_seed(BoundedSequenceSeed {
                    collection,
                    limit: collection.limit(self.limits),
                    overflow: self.overflow,
                })?;
            }
        }
        Ok(())
    }
}

struct MembershipShapeSeed<'a> {
    limits: LodgeCodecLimits,
    overflow: &'a std::cell::Cell<Option<LodgeCollectionOverflow>>,
}

impl<'de> DeserializeSeed<'de> for MembershipShapeSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(MembershipShapeVisitor {
            limits: self.limits,
            overflow: self.overflow,
        })
    }
}

struct MembershipShapeVisitor<'a> {
    limits: LodgeCodecLimits,
    overflow: &'a std::cell::Cell<Option<LodgeCollectionOverflow>>,
}

impl<'de> Visitor<'de> for MembershipShapeVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a LODGE membership-index map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while let Some(field) = map.next_key::<String>()? {
            if field == "entries" {
                let collection = LodgeCollection::MembershipEntries;
                map.next_value_seed(BoundedSequenceSeed {
                    collection,
                    limit: collection.limit(self.limits),
                    overflow: self.overflow,
                })?;
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(())
    }
}

struct BoundedSequenceSeed<'a> {
    collection: LodgeCollection,
    limit: u64,
    overflow: &'a std::cell::Cell<Option<LodgeCollectionOverflow>>,
}

impl<'de> DeserializeSeed<'de> for BoundedSequenceSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(BoundedSequenceVisitor {
            collection: self.collection,
            limit: self.limit,
            overflow: self.overflow,
        })
    }
}

struct BoundedSequenceVisitor<'a> {
    collection: LodgeCollection,
    limit: u64,
    overflow: &'a std::cell::Cell<Option<LodgeCollectionOverflow>>,
}

impl<'de> Visitor<'de> for BoundedSequenceVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded LODGE collection")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut count = 0_u64;
        while sequence.next_element::<IgnoredAny>()?.is_some() {
            count = count.checked_add(1).ok_or_else(|| {
                <A::Error as de::Error>::custom("LODGE collection length overflow")
            })?;
            if count > self.limit {
                self.overflow.set(Some(LodgeCollectionOverflow {
                    collection: self.collection,
                    actual: count,
                    limit: self.limit,
                }));
                return Err(<A::Error as de::Error>::custom(
                    "LODGE collection exceeds its limit",
                ));
            }
        }
        Ok(())
    }
}

fn validate_json_collection_limits(
    payload: &[u8],
    limits: LodgeCodecLimits,
) -> Result<(), LodgeCodecError> {
    let overflow = std::cell::Cell::new(None);
    let mut deserializer = serde_json::Deserializer::from_slice(payload);
    let result = LodgeShapeSeed {
        limits,
        overflow: &overflow,
    }
    .deserialize(&mut deserializer);
    if let Some(overflow) = overflow.get() {
        return Err(LodgeCodecError::LimitExceeded {
            field: overflow.collection.field(),
            actual: overflow.actual,
            limit: overflow.limit,
        });
    }
    result.map_err(|error| LodgeCodecError::Deserialize(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| LodgeCodecError::Deserialize(error.to_string()))
}

fn enforce_limit(field: &'static str, actual: u64, limit: u64) -> Result<(), LodgeCodecError> {
    if actual > limit {
        Err(LodgeCodecError::LimitExceeded {
            field,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn check_envelope_count(
    field: &'static str,
    encoded: u32,
    payload_header: u32,
    decoded: usize,
) -> Result<(), LodgeCodecError> {
    if encoded != payload_header || usize::try_from(encoded).ok() != Some(decoded) {
        Err(LodgeCodecError::CountMismatch {
            field,
            encoded,
            payload_header,
            decoded: decoded as u64,
        })
    } else {
        Ok(())
    }
}

fn usize_to_u32(value: usize) -> Result<u32, LodgeCodecError> {
    u32::try_from(value).map_err(|_| LodgeCodecError::LengthOverflow)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, LodgeCodecError> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or(LodgeCodecError::Truncated("u16 field"))?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, LodgeCodecError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(LodgeCodecError::Truncated("u32 field"))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, LodgeCodecError> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(LodgeCodecError::Truncated("u64 field"))?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

fn read_hash(bytes: &[u8], offset: usize) -> Result<[u8; 32], LodgeCodecError> {
    bytes
        .get(offset..offset + 32)
        .ok_or(LodgeCodecError::Truncated("SHA-256 field"))?
        .try_into()
        .map_err(|_| LodgeCodecError::Truncated("SHA-256 field"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LodgeCodecError {
    InvalidLimits,
    InvalidMagic,
    UnsupportedContainerVersion(u16),
    UnsupportedSemanticVersion(u16),
    UnsupportedEncoding(u8),
    EncodingUnavailable(LodgeManifestEncoding),
    NonZeroReservedBytes,
    Truncated(&'static str),
    LengthOverflow,
    LengthMismatch {
        expected: u64,
        actual: u64,
    },
    LimitExceeded {
        field: &'static str,
        actual: u64,
        limit: u64,
    },
    CountMismatch {
        field: &'static str,
        encoded: u32,
        payload_header: u32,
        decoded: u64,
    },
    InvalidManifestMapKeys,
    Sha256Mismatch(&'static str),
    Serialize(String),
    Deserialize(String),
    ManifestValidation(LodgeValidationError),
    EmptyMembership,
    InvalidMembershipId {
        index: usize,
    },
    NonCanonicalMembership,
    MembershipIdOverflow,
    MembershipIdOutOfRange {
        id: u64,
        stable_gaussian_count: u64,
    },
    MembershipCountMismatch {
        expected: u64,
        actual: u64,
    },
    MembershipEndpointMismatch,
    MissingPageStorage,
    PageAuthenticationMismatch,
}

impl fmt::Display for LodgeCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => write!(f, "LODGE codec limits are invalid"),
            Self::InvalidMagic => write!(f, "invalid LODGE container magic"),
            Self::UnsupportedContainerVersion(version) => {
                write!(f, "unsupported LODGE container version {version}")
            }
            Self::UnsupportedSemanticVersion(version) => {
                write!(f, "unsupported LODGE semantic version {version}")
            }
            Self::UnsupportedEncoding(encoding) => {
                write!(f, "unsupported LODGE manifest encoding {encoding}")
            }
            Self::EncodingUnavailable(encoding) => {
                write!(f, "LODGE encoding {encoding:?} is not compiled in")
            }
            Self::NonZeroReservedBytes => write!(f, "LODGE reserved header bytes must be zero"),
            Self::Truncated(field) => write!(f, "truncated LODGE {field}"),
            Self::LengthOverflow => write!(f, "LODGE encoded length overflowed"),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "LODGE length is {actual}, expected {expected}")
            }
            Self::LimitExceeded {
                field,
                actual,
                limit,
            } => write!(f, "LODGE {field} {actual} exceeds configured limit {limit}"),
            Self::CountMismatch {
                field,
                encoded,
                payload_header,
                decoded,
            } => write!(
                f,
                "LODGE {field} count is {encoded} in the container, {payload_header} in the payload header, and {decoded} after decoding"
            ),
            Self::InvalidManifestMapKeys => write!(
                f,
                "Flexbuffers LODGE map keys must be strictly sorted and unique"
            ),
            Self::Sha256Mismatch(field) => write!(f, "LODGE {field} SHA-256 mismatch"),
            Self::Serialize(error) => write!(f, "failed to serialize LODGE data: {error}"),
            Self::Deserialize(error) => write!(f, "failed to deserialize LODGE data: {error}"),
            Self::ManifestValidation(error) => write!(f, "invalid LODGE manifest: {error}"),
            Self::EmptyMembership => write!(f, "LODGE membership stream is empty"),
            Self::InvalidMembershipId { index } => {
                write!(
                    f,
                    "LODGE membership ID at index {index} is zero or not increasing"
                )
            }
            Self::NonCanonicalMembership => {
                write!(f, "LODGE membership ULEB128 is not canonical")
            }
            Self::MembershipIdOverflow => write!(f, "LODGE membership ID overflowed"),
            Self::MembershipIdOutOfRange {
                id,
                stable_gaussian_count,
            } => write!(
                f,
                "LODGE membership ID {id} exceeds catalog size {stable_gaussian_count}"
            ),
            Self::MembershipCountMismatch { expected, actual } => write!(
                f,
                "LODGE membership contains {actual} IDs, expected {expected}"
            ),
            Self::MembershipEndpointMismatch => {
                write!(f, "LODGE membership endpoints do not match its descriptor")
            }
            Self::MissingPageStorage => write!(f, "LODGE page has no storage descriptor"),
            Self::PageAuthenticationMismatch => {
                write!(f, "LODGE page descriptor and authentication IDs differ")
            }
        }
    }
}

impl Error for LodgeCodecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ManifestValidation(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum LodgeAssetLoaderError {
    Io(std::io::Error),
    Codec(LodgeCodecError),
}

impl fmt::Display for LodgeAssetLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "LODGE asset IO failed: {error}"),
            Self::Codec(error) => write!(f, "LODGE asset decode failed: {error}"),
        }
    }
}

impl Error for LodgeAssetLoaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Codec(error) => Some(error),
        }
    }
}

impl From<std::io::Error> for LodgeAssetLoaderError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<LodgeCodecError> for LodgeAssetLoaderError {
    fn from(value: LodgeCodecError) -> Self {
        Self::Codec(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::formats::lodge::tests::fixture;

    #[test]
    fn sidecar_round_trips_with_outer_counts_and_sha256() {
        let manifest = fixture();
        for encoding in [
            LodgeManifestEncoding::Json,
            LodgeManifestEncoding::Flexbuffers,
        ] {
            if encoding == LodgeManifestEncoding::Flexbuffers && !cfg!(feature = "io_flexbuffers") {
                continue;
            }
            let encoded = encode_lodge_manifest_with_encoding(&manifest, encoding).unwrap();
            assert_eq!(&encoded[..8], &LODGE_CONTAINER_MAGIC);
            assert_eq!(read_u16(&encoded, 8).unwrap(), LODGE_CONTAINER_VERSION);
            assert_eq!(read_u32(&encoded, 88).unwrap(), manifest.header.level_count);
            assert_eq!(
                read_u32(&encoded, 92).unwrap(),
                manifest.header.cluster_count
            );
            assert_eq!(
                read_hash(&encoded, 56).unwrap(),
                manifest.base_manifest.sha256
            );
            assert_eq!(
                decode_lodge_manifest(&encoded, LodgeCodecLimits::default()).unwrap(),
                manifest
            );
        }
    }

    #[test]
    fn payload_and_base_identity_tampering_fail_closed() {
        let manifest = fixture();
        let mut encoded =
            encode_lodge_manifest_with_encoding(&manifest, LodgeManifestEncoding::Json).unwrap();
        *encoded.last_mut().unwrap() ^= 1;
        assert_eq!(
            decode_lodge_manifest(&encoded, LodgeCodecLimits::default()),
            Err(LodgeCodecError::Sha256Mismatch("manifest payload"))
        );

        let mut encoded =
            encode_lodge_manifest_with_encoding(&manifest, LodgeManifestEncoding::Json).unwrap();
        encoded[56] ^= 1;
        assert_eq!(
            decode_lodge_manifest(&encoded, LodgeCodecLimits::default()),
            Err(LodgeCodecError::Sha256Mismatch("base manifest identity"))
        );
    }

    #[test]
    fn dependency_and_page_bytes_are_verified_before_parsing() {
        let object_bytes = b"authenticated base manifest";
        let object = LodgeAuthenticatedObject {
            uri: "scene.gsplatlod".into(),
            encoded_len: object_bytes.len() as u64,
            sha256: sha256_bytes(object_bytes),
        };
        verify_lodge_authenticated_object(object_bytes, &object, 1024).unwrap();
        assert_eq!(
            verify_lodge_authenticated_object(b"wrong", &object, 1024),
            Err(LodgeCodecError::LengthMismatch {
                expected: object_bytes.len() as u64,
                actual: 5,
            })
        );

        let manifest = fixture();
        let mut descriptor = manifest.extra_pages[0].clone();
        let page_bytes = b"ordinary encoded Gaussian page";
        descriptor.storage.as_mut().unwrap().encoded_len = page_bytes.len() as u64;
        let authentication = LodgePageAuthentication {
            page: descriptor.id,
            encoded_sha256: sha256_bytes(page_bytes),
        };
        verify_lodge_page_bytes(page_bytes, &descriptor, &authentication, 1024).unwrap();
        let mut wrong_id = authentication;
        wrong_id.page = crate::LodPageId(999);
        assert_eq!(
            verify_lodge_page_bytes(page_bytes, &descriptor, &wrong_id, 1024),
            Err(LodgeCodecError::PageAuthenticationMismatch)
        );
    }

    #[test]
    fn outer_limits_run_before_hash_or_serde() {
        let manifest = fixture();
        let mut encoded =
            encode_lodge_manifest_with_encoding(&manifest, LodgeManifestEncoding::Json).unwrap();
        let mut limits = LodgeCodecLimits::default();
        limits.max_levels = manifest.header.level_count;
        let advertised = limits.max_levels + 1;
        encoded[88..92].copy_from_slice(&advertised.to_le_bytes());
        encoded[LODGE_HEADER_LEN] ^= 0xff;
        assert_eq!(
            decode_lodge_manifest(&encoded, limits),
            Err(LodgeCodecError::LimitExceeded {
                field: "levels",
                actual: advertised.into(),
                limit: limits.max_levels.into(),
            })
        );
    }

    #[test]
    fn declared_dependency_closure_is_bounded_before_fetch() {
        let manifest = fixture();
        let encoded =
            encode_lodge_manifest_with_encoding(&manifest, LodgeManifestEncoding::Json).unwrap();
        let limits = LodgeCodecLimits {
            // Each object is individually below this limit (512, 128, and 10
            // bytes), but their sidecar-declared closure is not.
            max_dependency_bytes: 600,
            ..Default::default()
        };
        assert_eq!(
            decode_lodge_manifest(&encoded, limits),
            Err(LodgeCodecError::LimitExceeded {
                field: "declared dependency bytes",
                actual: 650,
                limit: 600,
            })
        );
    }

    #[test]
    fn json_nested_membership_limit_runs_before_manifest_allocation() {
        let manifest = fixture();
        let encoded =
            encode_lodge_manifest_with_encoding(&manifest, LodgeManifestEncoding::Json).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&encoded[LODGE_HEADER_LEN..]).unwrap();
        let entries = value["membership_index"]["entries"].as_array_mut().unwrap();
        entries.push(entries[0].clone());
        let payload = serde_json::to_vec(&value).unwrap();
        let mut oversized = encoded[..LODGE_HEADER_LEN].to_vec();
        oversized[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        oversized[24..56].copy_from_slice(&sha256_bytes(&payload));
        oversized.extend_from_slice(&payload);

        let mut limits = LodgeCodecLimits::default();
        limits.max_clusters = manifest.header.cluster_count;
        assert_eq!(
            decode_lodge_manifest(&oversized, limits),
            Err(LodgeCodecError::LimitExceeded {
                field: "membership entries",
                actual: 3,
                limit: 2,
            })
        );
    }

    #[test]
    fn membership_codec_round_trips_without_gaussian_pages() {
        let ids = [
            LodgeGaussianId(1),
            LodgeGaussianId(127),
            LodgeGaussianId(128),
            LodgeGaussianId(16_384),
            LodgeGaussianId(10_000_000),
        ];
        let encoded = encode_lodge_membership_ids(&ids).unwrap();
        let decoded = decode_lodge_membership_ids(
            &encoded,
            ids.len() as u64,
            ids.last().unwrap().0,
            LodgeCodecLimits::default(),
        )
        .unwrap();
        assert_eq!(decoded, ids);

        let entry = LodgeMembershipEntry {
            cluster: crate::LodgeClusterId(1),
            byte_range: (100, encoded.len() as u64),
            member_count: ids.len() as u64,
            first_id: ids[0],
            last_id: *ids.last().unwrap(),
            encoded_sha256: sha256_bytes(&encoded),
        };
        assert_eq!(
            decode_lodge_membership_entry(
                &encoded,
                &entry,
                ids.last().unwrap().0,
                LodgeCodecLimits::default(),
            )
            .unwrap(),
            ids
        );
    }

    #[test]
    fn membership_codec_rejects_noncanonical_truncated_and_unbounded_data() {
        let limits = LodgeCodecLimits::default();
        assert_eq!(
            decode_lodge_membership_ids(&[0], 1, 10, limits),
            Err(LodgeCodecError::NonCanonicalMembership)
        );
        assert_eq!(
            decode_lodge_membership_ids(&[0x81, 0], 1, 10, limits),
            Err(LodgeCodecError::NonCanonicalMembership)
        );
        assert_eq!(
            decode_lodge_membership_ids(&[0x80], 1, 10, limits),
            Err(LodgeCodecError::Truncated("membership ULEB128"))
        );
        assert_eq!(
            decode_lodge_membership_ids(&[11], 1, 10, limits),
            Err(LodgeCodecError::MembershipIdOutOfRange {
                id: 11,
                stable_gaussian_count: 10,
            })
        );
    }
}
