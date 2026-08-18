//! Fallible, versioned codecs and Bevy asset loaders for virtual Gaussian scenes.

use std::{error::Error, fmt, mem::size_of};

use bevy::{
    asset::{AssetLoader, AsyncReadExt, LoadContext, io::Reader},
    prelude::*,
    reflect::TypePath,
};
use serde::{Deserialize, Serialize};

use crate::{
    gaussian::formats::{
        planar_3d::Gaussian3d,
        planar_3d_chunked::{
            LOD_PAGE_SCHEMA_VERSION, LodPageDescriptor, LodPageEncoding, LodPageId,
            PlanarGaussian3dPage, validate_gaussian,
        },
        planar_3d_lod::GaussianLodManifest,
    },
    material::spherical_harmonics::{SH_CHANNELS, SH_COEFF_COUNT, SH_DEGREE},
};

#[cfg_attr(not(feature = "lod"), allow(dead_code))]
mod page_decoder;

#[cfg(any(test, all(feature = "lod", target_arch = "wasm32")))]
pub(crate) use page_decoder::{IncrementalLodPageDecoder, LodPageDecodeProgress};

const MANIFEST_CONTAINER_MAGIC: [u8; 8] = *b"BGSLODC\0";
const PAGE_CONTAINER_MAGIC: [u8; 8] = *b"BGSPAGE\0";
// Manifest container 2 adds pre-deserialization node/page count gates. Full
// f32 pages retain container version 1 byte-for-byte; container version 2 is
// the explicit reduced-degree binary16 SH representation.
const MANIFEST_CONTAINER_VERSION: u16 = 2;
const PAGE_CONTAINER_VERSION: u16 = 1;
const F16_SH_PAGE_CONTAINER_VERSION: u16 = 2;
const MANIFEST_HEADER_LEN: usize = 40;
const PAGE_HEADER_LEN: usize = 44;
const FLOATS_PER_GAUSSIAN: usize = 4 + SH_COEFF_COUNT + 4 + 4;
const PAGE_SH_COEFFICIENT_COUNT: u32 = SH_COEFF_COUNT as u32;
/// Conservative upper bound used to translate a record budget into raw page
/// bytes before a container header has been decoded.
#[cfg(any(test, all(feature = "lod", target_arch = "wasm32")))]
pub(crate) const MAX_ENCODED_PAGE_GAUSSIAN_BYTES: usize = FLOATS_PER_GAUSSIAN * size_of::<f32>();

const LOD_SHARD_MAGIC: [u8; 8] = *b"BGSSHARD";
pub const LOD_SHARD_CONTAINER_VERSION: u16 = 1;
pub const LOD_SHARD_HEADER_LEN: usize = 40;
pub const LOD_SHARD_ENTRY_LEN: usize = 32;

/// One immutable range-table entry in a `.bgslodpack` shard. Offsets are
/// absolute so ordinary file and HTTP transports can request a page without
/// first downloading the table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodShardEntry {
    pub page_id: LodPageId,
    pub byte_offset: u64,
    pub encoded_len: u64,
    pub content_hash: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LodShardIndex {
    pub file_len: u64,
    pub entries: Vec<LodShardEntry>,
}

pub fn lod_shard_prefix_len(entry_count: u32) -> Result<u64, LodCodecError> {
    u64::from(entry_count)
        .checked_mul(LOD_SHARD_ENTRY_LEN as u64)
        .and_then(|table| table.checked_add(LOD_SHARD_HEADER_LEN as u64))
        .ok_or(LodCodecError::LengthOverflow)
}

/// Encodes the fixed header and complete range table. Payload bytes follow
/// this prefix and are independently checksummed by the existing page codec.
pub fn encode_lod_shard_index(index: &LodShardIndex) -> Result<Vec<u8>, LodCodecError> {
    let entry_count =
        u32::try_from(index.entries.len()).map_err(|_| LodCodecError::LengthOverflow)?;
    let prefix_len = lod_shard_prefix_len(entry_count)?;
    validate_lod_shard_entries(&index.entries, prefix_len, index.file_len)?;
    let table_len = u64::from(entry_count)
        .checked_mul(LOD_SHARD_ENTRY_LEN as u64)
        .ok_or(LodCodecError::LengthOverflow)?;
    let mut table = Vec::with_capacity(table_len as usize);
    for entry in &index.entries {
        table.extend_from_slice(&entry.page_id.0.to_le_bytes());
        table.extend_from_slice(&entry.byte_offset.to_le_bytes());
        table.extend_from_slice(&entry.encoded_len.to_le_bytes());
        table.extend_from_slice(&entry.content_hash.to_le_bytes());
    }
    let mut encoded = Vec::with_capacity(prefix_len as usize);
    encoded.extend_from_slice(&LOD_SHARD_MAGIC);
    encoded.extend_from_slice(&LOD_SHARD_CONTAINER_VERSION.to_le_bytes());
    encoded.extend_from_slice(&0_u16.to_le_bytes());
    encoded.extend_from_slice(&entry_count.to_le_bytes());
    encoded.extend_from_slice(&table_len.to_le_bytes());
    encoded.extend_from_slice(&index.file_len.to_le_bytes());
    encoded.extend_from_slice(&checksum64(&table).to_le_bytes());
    encoded.extend_from_slice(&table);
    Ok(encoded)
}

/// Decodes a bounded prefix previously sized with the header's entry count.
/// The caller supplies the actual file length so a truncated or appended shard
/// cannot make its range table authoritative.
pub fn decode_lod_shard_index(
    encoded_prefix: &[u8],
    actual_file_len: u64,
    max_entries: u32,
) -> Result<LodShardIndex, LodCodecError> {
    if encoded_prefix.len() < LOD_SHARD_HEADER_LEN {
        return Err(LodCodecError::Truncated("shard header"));
    }
    if encoded_prefix[..8] != LOD_SHARD_MAGIC {
        return Err(LodCodecError::InvalidMagic("shard"));
    }
    let version = read_u16(encoded_prefix, 8)?;
    if version != LOD_SHARD_CONTAINER_VERSION {
        return Err(LodCodecError::UnsupportedShardContainerVersion(version));
    }
    if read_u16(encoded_prefix, 10)? != 0 {
        return Err(LodCodecError::NonZeroReservedBytes);
    }
    let entry_count = read_u32(encoded_prefix, 12)?;
    if entry_count == 0 || entry_count > max_entries {
        return Err(LodCodecError::ShardEntryLimitExceeded {
            actual: entry_count,
            limit: max_entries,
        });
    }
    let table_len = read_u64(encoded_prefix, 16)?;
    let expected_table_len = u64::from(entry_count)
        .checked_mul(LOD_SHARD_ENTRY_LEN as u64)
        .ok_or(LodCodecError::LengthOverflow)?;
    if table_len != expected_table_len {
        return Err(LodCodecError::InvalidShardIndex(
            "range-table length does not match entry count".into(),
        ));
    }
    let declared_file_len = read_u64(encoded_prefix, 24)?;
    if declared_file_len != actual_file_len {
        return Err(LodCodecError::LengthMismatch {
            expected: declared_file_len,
            actual: actual_file_len,
        });
    }
    let prefix_len = lod_shard_prefix_len(entry_count)?;
    if encoded_prefix.len() as u64 != prefix_len {
        return Err(LodCodecError::LengthMismatch {
            expected: prefix_len,
            actual: encoded_prefix.len() as u64,
        });
    }
    let table = &encoded_prefix[LOD_SHARD_HEADER_LEN..];
    let expected_checksum = read_u64(encoded_prefix, 32)?;
    let actual_checksum = checksum64(table);
    if actual_checksum != expected_checksum {
        return Err(LodCodecError::ChecksumMismatch {
            expected: expected_checksum,
            actual: actual_checksum,
        });
    }
    let mut entries = Vec::with_capacity(entry_count as usize);
    for entry_index in 0..entry_count as usize {
        let offset = LOD_SHARD_HEADER_LEN + entry_index * LOD_SHARD_ENTRY_LEN;
        entries.push(LodShardEntry {
            page_id: LodPageId(read_u64(encoded_prefix, offset)?),
            byte_offset: read_u64(encoded_prefix, offset + 8)?,
            encoded_len: read_u64(encoded_prefix, offset + 16)?,
            content_hash: read_u64(encoded_prefix, offset + 24)?,
        });
    }
    validate_lod_shard_entries(&entries, prefix_len, declared_file_len)?;
    Ok(LodShardIndex {
        file_len: declared_file_len,
        entries,
    })
}

fn validate_lod_shard_entries(
    entries: &[LodShardEntry],
    prefix_len: u64,
    file_len: u64,
) -> Result<(), LodCodecError> {
    if entries.is_empty() {
        return Err(LodCodecError::InvalidShardIndex(
            "range table is empty".into(),
        ));
    }
    let mut expected_offset = prefix_len;
    let mut previous_page = LodPageId::INVALID;
    for entry in entries {
        if !entry.page_id.is_valid()
            || entry.page_id <= previous_page
            || entry.encoded_len == 0
            || entry.byte_offset != expected_offset
        {
            return Err(LodCodecError::InvalidShardIndex(
                "page IDs or packed ranges are not strictly ordered and contiguous".into(),
            ));
        }
        expected_offset = entry
            .byte_offset
            .checked_add(entry.encoded_len)
            .ok_or(LodCodecError::LengthOverflow)?;
        previous_page = entry.page_id;
    }
    if expected_offset != file_len {
        return Err(LodCodecError::InvalidShardIndex(
            "final packed range does not end at the declared file length".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ManifestEncoding {
    Flexbuffers = 1,
    Json = 2,
}

impl TryFrom<u8> for ManifestEncoding {
    type Error = LodCodecError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Flexbuffers),
            2 => Ok(Self::Json),
            other => Err(LodCodecError::UnsupportedManifestEncoding(other)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodCodecLimits {
    pub max_manifest_bytes: u64,
    pub max_nodes: u32,
    pub max_pages: u32,
    pub max_page_bytes: u64,
    pub max_page_gaussians: u32,
}

impl Default for LodCodecLimits {
    fn default() -> Self {
        Self {
            // This encoded-size gate is the hard pre-deserialization bound for
            // untrusted containers; declared record counts are cross-checked
            // but cannot themselves be trusted before decoding.
            max_manifest_bytes: 64 * 1024 * 1024,
            // These defaults comfortably cover practical >100M-Gaussian
            // packages while preventing multi-million-record allocations
            // unless callers explicitly opt into larger limits.
            max_nodes: 1_000_000,
            max_pages: 262_144,
            max_page_bytes: 64 * 1024 * 1024,
            max_page_gaussians: 1_000_000,
        }
    }
}

impl LodCodecLimits {
    pub fn validate(self) -> Result<Self, LodCodecError> {
        if self.max_manifest_bytes < MANIFEST_HEADER_LEN as u64
            || self.max_nodes == 0
            || self.max_pages == 0
            || self.max_page_bytes < PAGE_HEADER_LEN as u64
            || self.max_page_gaussians == 0
        {
            Err(LodCodecError::InvalidLimits)
        } else {
            Ok(self)
        }
    }
}

#[derive(Asset, Clone, Debug, TypePath)]
pub struct GaussianLodAsset {
    pub manifest: GaussianLodManifest,
}

#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct GaussianLodHandle(pub Handle<GaussianLodAsset>);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GaussianLodManifestLoaderSettings {
    pub max_encoded_bytes: u64,
    pub max_nodes: u32,
    pub max_pages: u32,
}

impl Default for GaussianLodManifestLoaderSettings {
    fn default() -> Self {
        let limits = LodCodecLimits::default();
        Self {
            max_encoded_bytes: limits.max_manifest_bytes,
            max_nodes: limits.max_nodes,
            max_pages: limits.max_pages,
        }
    }
}

#[derive(Default, TypePath)]
pub struct GaussianLodManifestLoader;

impl AssetLoader for GaussianLodManifestLoader {
    type Asset = GaussianLodAsset;
    type Settings = GaussianLodManifestLoaderSettings;
    type Error = LodAssetLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &Self::Settings,
        _: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let bytes = read_bounded(reader, settings.max_encoded_bytes).await?;
        let limits = LodCodecLimits {
            max_manifest_bytes: settings.max_encoded_bytes,
            max_nodes: settings.max_nodes,
            max_pages: settings.max_pages,
            ..Default::default()
        };
        Ok(GaussianLodAsset {
            manifest: decode_manifest(&bytes, limits)?,
        })
    }

    fn extensions(&self) -> &[&str] {
        &["gsplatlod"]
    }
}

async fn read_bounded(
    reader: &mut dyn Reader,
    max_encoded_bytes: u64,
) -> Result<Vec<u8>, LodAssetLoaderError> {
    let max = usize::try_from(max_encoded_bytes).map_err(|_| LodCodecError::InvalidLimits)?;
    let probe_limit = max_encoded_bytes
        .checked_add(1)
        .ok_or(LodCodecError::InvalidLimits)?;
    let mut bytes = Vec::with_capacity(max.min(1024 * 1024));
    let mut bounded = reader.take(probe_limit);
    bounded.read_to_end(&mut bytes).await?;
    if bytes.len() > max {
        return Err(LodCodecError::LimitExceeded {
            field: "encoded bytes",
            actual: bytes.len() as u64,
            limit: max_encoded_bytes,
        }
        .into());
    }
    Ok(bytes)
}

pub fn encode_manifest(manifest: &GaussianLodManifest) -> Result<Vec<u8>, LodCodecError> {
    #[cfg(feature = "io_flexbuffers")]
    let encoding = ManifestEncoding::Flexbuffers;
    #[cfg(not(feature = "io_flexbuffers"))]
    let encoding = ManifestEncoding::Json;
    encode_manifest_with_encoding(manifest, encoding)
}

pub fn encode_manifest_with_encoding(
    manifest: &GaussianLodManifest,
    encoding: ManifestEncoding,
) -> Result<Vec<u8>, LodCodecError> {
    manifest
        .validate()
        .map_err(|error| LodCodecError::ManifestValidation(error.to_string()))?;
    let payload = match encoding {
        ManifestEncoding::Flexbuffers => {
            #[cfg(feature = "io_flexbuffers")]
            {
                let mut serializer = flexbuffers::FlexbufferSerializer::new();
                manifest
                    .serialize(&mut serializer)
                    .map_err(|error| LodCodecError::Serialize(error.to_string()))?;
                serializer.view().to_vec()
            }
            #[cfg(not(feature = "io_flexbuffers"))]
            {
                return Err(LodCodecError::EncodingUnavailable(encoding));
            }
        }
        ManifestEncoding::Json => serde_json::to_vec(manifest)
            .map_err(|error| LodCodecError::Serialize(error.to_string()))?,
    };

    let payload_len = u64::try_from(payload.len()).map_err(|_| LodCodecError::LengthOverflow)?;
    let mut encoded = Vec::with_capacity(MANIFEST_HEADER_LEN + payload.len());
    encoded.extend_from_slice(&MANIFEST_CONTAINER_MAGIC);
    encoded.extend_from_slice(&MANIFEST_CONTAINER_VERSION.to_le_bytes());
    encoded.push(encoding as u8);
    encoded.extend_from_slice(&[0; 5]);
    encoded.extend_from_slice(&manifest.header.node_count.to_le_bytes());
    encoded.extend_from_slice(&manifest.header.page_count.to_le_bytes());
    encoded.extend_from_slice(&payload_len.to_le_bytes());
    encoded.extend_from_slice(&checksum64(&payload).to_le_bytes());
    debug_assert_eq!(encoded.len(), MANIFEST_HEADER_LEN);
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

pub fn decode_manifest(
    encoded: &[u8],
    limits: LodCodecLimits,
) -> Result<GaussianLodManifest, LodCodecError> {
    let limits = limits.validate()?;
    enforce_limit(
        "manifest bytes",
        encoded.len() as u64,
        limits.max_manifest_bytes,
    )?;
    // Read only the fixed prefix before interpreting the versioned remainder.
    // This preserves a useful version error for legacy 32-byte headers.
    if encoded.len() < 10 {
        return Err(LodCodecError::Truncated("manifest header prefix"));
    }
    if encoded[0..8] != MANIFEST_CONTAINER_MAGIC {
        return Err(LodCodecError::InvalidMagic("manifest"));
    }
    let version = read_u16(encoded, 8)?;
    if version != MANIFEST_CONTAINER_VERSION {
        return Err(LodCodecError::UnsupportedContainerVersion(version));
    }
    if encoded.len() < MANIFEST_HEADER_LEN {
        return Err(LodCodecError::Truncated("manifest header"));
    }
    if encoded[11..16].iter().any(|byte| *byte != 0) {
        return Err(LodCodecError::NonZeroReservedBytes);
    }
    let encoding = ManifestEncoding::try_from(encoded[10])?;
    let encoded_node_count = read_u32(encoded, 16)?;
    let encoded_page_count = read_u32(encoded, 20)?;
    enforce_limit(
        "manifest nodes",
        u64::from(encoded_node_count),
        u64::from(limits.max_nodes),
    )?;
    enforce_limit(
        "manifest pages",
        u64::from(encoded_page_count),
        u64::from(limits.max_pages),
    )?;
    let payload_len = read_u64(encoded, 24)?;
    let expected_checksum = read_u64(encoded, 32)?;
    let payload_len = usize::try_from(payload_len).map_err(|_| LodCodecError::LengthOverflow)?;
    let expected_len = MANIFEST_HEADER_LEN
        .checked_add(payload_len)
        .ok_or(LodCodecError::LengthOverflow)?;
    if encoded.len() != expected_len {
        return Err(LodCodecError::LengthMismatch {
            expected: expected_len as u64,
            actual: encoded.len() as u64,
        });
    }
    let payload = &encoded[MANIFEST_HEADER_LEN..];
    let actual_checksum = checksum64(payload);
    if actual_checksum != expected_checksum {
        return Err(LodCodecError::ChecksumMismatch {
            expected: expected_checksum,
            actual: actual_checksum,
        });
    }

    let manifest = match encoding {
        ManifestEncoding::Flexbuffers => {
            #[cfg(feature = "io_flexbuffers")]
            {
                let reader = flexbuffers::Reader::get_root(payload)
                    .map_err(|error| LodCodecError::Deserialize(error.to_string()))?;
                GaussianLodManifest::deserialize(reader)
                    .map_err(|error| LodCodecError::Deserialize(error.to_string()))?
            }
            #[cfg(not(feature = "io_flexbuffers"))]
            {
                return Err(LodCodecError::EncodingUnavailable(encoding));
            }
        }
        ManifestEncoding::Json => serde_json::from_slice(payload)
            .map_err(|error| LodCodecError::Deserialize(error.to_string()))?,
    };
    enforce_limit(
        "manifest nodes",
        manifest.nodes.len() as u64,
        u64::from(limits.max_nodes),
    )?;
    enforce_limit(
        "manifest pages",
        manifest.pages.len() as u64,
        u64::from(limits.max_pages),
    )?;
    check_manifest_count(
        "nodes",
        encoded_node_count,
        manifest.header.node_count,
        manifest.nodes.len(),
    )?;
    check_manifest_count(
        "pages",
        encoded_page_count,
        manifest.header.page_count,
        manifest.pages.len(),
    )?;
    manifest
        .validate()
        .map_err(|error| LodCodecError::ManifestValidation(error.to_string()))?;
    Ok(manifest)
}

pub fn encode_page(page: &PlanarGaussian3dPage) -> Result<Vec<u8>, LodCodecError> {
    encode_page_with_encoding(page, LodPageEncoding::F32Planar)
}

/// Encodes one independently verifiable page. Reduced-degree F16 is intended
/// only for representative pages; manifest validation rejects it for source
/// leaves so the q=1 endpoint remains exact.
pub fn encode_page_with_encoding(
    page: &PlanarGaussian3dPage,
    encoding: LodPageEncoding,
) -> Result<Vec<u8>, LodCodecError> {
    if page.schema_version != LOD_PAGE_SCHEMA_VERSION {
        return Err(LodCodecError::UnsupportedPageSchema(page.schema_version));
    }
    if !page.id.is_valid() {
        return Err(LodCodecError::InvalidPageId(page.id));
    }
    if page.gaussians.is_empty() {
        return Err(LodCodecError::EmptyPage);
    }
    for (index, gaussian) in page.gaussians.iter().enumerate() {
        validate_gaussian(gaussian).map_err(|field| LodCodecError::InvalidGaussian {
            index,
            field: format!("{field:?}"),
        })?;
    }
    let gaussian_count =
        u32::try_from(page.gaussians.len()).map_err(|_| LodCodecError::LengthOverflow)?;
    let mut canonical = page.clone();
    let (container_version, encoded_sh_coefficients) = match encoding {
        LodPageEncoding::F32Planar => (PAGE_CONTAINER_VERSION, SH_COEFF_COUNT),
        LodPageEncoding::F16Sh { degree } => {
            let count = sh_coefficient_count_for_degree(degree)
                .ok_or(LodCodecError::InvalidCompressedShDegree(degree))?;
            for gaussian in &mut canonical.gaussians {
                for coefficient in &mut gaussian.spherical_harmonic.coefficients[..count] {
                    *coefficient = half::f16::from_f32(*coefficient).to_f32();
                }
                gaussian.spherical_harmonic.coefficients[count..].fill(0.0);
            }
            (F16_SH_PAGE_CONTAINER_VERSION, count)
        }
    };
    let payload_len = page_payload_len(gaussian_count, encoding)?;
    let mut encoded = Vec::with_capacity(PAGE_HEADER_LEN + payload_len);
    encoded.extend_from_slice(&PAGE_CONTAINER_MAGIC);
    encoded.extend_from_slice(&container_version.to_le_bytes());
    encoded.extend_from_slice(&page.schema_version.to_le_bytes());
    encoded.extend_from_slice(&page.id.0.to_le_bytes());
    encoded.extend_from_slice(&gaussian_count.to_le_bytes());
    encoded.extend_from_slice(&(encoded_sh_coefficients as u32).to_le_bytes());
    encoded.extend_from_slice(&(payload_len as u64).to_le_bytes());
    encoded.extend_from_slice(&canonical.content_hash().to_le_bytes());
    debug_assert_eq!(encoded.len(), PAGE_HEADER_LEN);
    for gaussian in &canonical.gaussians {
        write_gaussian(&mut encoded, gaussian, encoding, encoded_sh_coefficients);
    }
    debug_assert_eq!(encoded.len(), PAGE_HEADER_LEN + payload_len);
    Ok(encoded)
}

pub fn decode_page(
    encoded: &[u8],
    limits: LodCodecLimits,
) -> Result<PlanarGaussian3dPage, LodCodecError> {
    decode_page_container(encoded, limits).map(|(page, _)| page)
}

fn decode_page_container(
    encoded: &[u8],
    limits: LodCodecLimits,
) -> Result<(PlanarGaussian3dPage, LodPageEncoding), LodCodecError> {
    page_decoder::decode_page_container(encoded, limits)
}

pub fn decode_page_with_descriptor(
    encoded: &[u8],
    descriptor: &LodPageDescriptor,
    limits: LodCodecLimits,
) -> Result<PlanarGaussian3dPage, LodCodecError> {
    page_decoder::decode_page_with_descriptor(encoded, descriptor, limits)
}

fn page_payload_len(count: u32, encoding: LodPageEncoding) -> Result<usize, LodCodecError> {
    let packed_float_bytes = match encoding {
        LodPageEncoding::F32Planar => {
            let bytes = FLOATS_PER_GAUSSIAN
                .checked_mul(size_of::<f32>())
                .ok_or(LodCodecError::LengthOverflow)?;
            debug_assert_eq!(bytes, size_of::<Gaussian3d>());
            bytes
        }
        LodPageEncoding::F16Sh { degree } => {
            let coefficients = sh_coefficient_count_for_degree(degree)
                .ok_or(LodCodecError::InvalidCompressedShDegree(degree))?;
            12_usize
                .checked_mul(size_of::<f32>())
                .and_then(|bytes| {
                    coefficients
                        .checked_mul(size_of::<u16>())
                        .and_then(|sh| bytes.checked_add(sh))
                })
                .ok_or(LodCodecError::LengthOverflow)?
        }
    };
    (count as usize)
        .checked_mul(packed_float_bytes)
        .ok_or(LodCodecError::LengthOverflow)
}

fn sh_coefficient_count_for_degree(degree: u8) -> Option<usize> {
    if degree > SH_DEGREE as u8 {
        return None;
    }
    usize::from(degree)
        .checked_add(1)?
        .checked_pow(2)?
        .checked_mul(SH_CHANNELS)
        .filter(|count| *count <= SH_COEFF_COUNT)
}

fn write_gaussian(
    output: &mut Vec<u8>,
    gaussian: &Gaussian3d,
    encoding: LodPageEncoding,
    encoded_sh_coefficients: usize,
) {
    for value in gaussian
        .position_visibility
        .position
        .into_iter()
        .chain(std::iter::once(gaussian.position_visibility.visibility))
    {
        output.extend_from_slice(&value.to_le_bytes());
    }
    match encoding {
        LodPageEncoding::F32Planar => {
            for value in gaussian.spherical_harmonic.coefficients {
                output.extend_from_slice(&value.to_le_bytes());
            }
        }
        LodPageEncoding::F16Sh { .. } => {
            for value in gaussian
                .spherical_harmonic
                .coefficients
                .iter()
                .take(encoded_sh_coefficients)
            {
                output.extend_from_slice(&half::f16::from_f32(*value).to_bits().to_le_bytes());
            }
        }
    }
    for value in gaussian
        .rotation
        .rotation
        .into_iter()
        .chain(gaussian.scale_opacity.scale)
        .chain(std::iter::once(gaussian.scale_opacity.opacity))
    {
        output.extend_from_slice(&value.to_le_bytes());
    }
}

fn read_gaussian(
    encoded: &[u8],
    offset: &mut usize,
    encoding: LodPageEncoding,
    encoded_sh_coefficients: usize,
) -> Result<Gaussian3d, LodCodecError> {
    let position_visibility = [
        read_next_f32(encoded, offset)?,
        read_next_f32(encoded, offset)?,
        read_next_f32(encoded, offset)?,
        read_next_f32(encoded, offset)?,
    ]
    .into();
    let mut coefficients = [0.0; SH_COEFF_COUNT];
    match encoding {
        LodPageEncoding::F32Planar => {
            for coefficient in &mut coefficients {
                *coefficient = read_next_f32(encoded, offset)?;
            }
        }
        LodPageEncoding::F16Sh { .. } => {
            for coefficient in coefficients.iter_mut().take(encoded_sh_coefficients) {
                let bits = read_u16(encoded, *offset)?;
                *offset += size_of::<u16>();
                *coefficient = half::f16::from_bits(bits).to_f32();
            }
        }
    }
    let rotation = [
        read_next_f32(encoded, offset)?,
        read_next_f32(encoded, offset)?,
        read_next_f32(encoded, offset)?,
        read_next_f32(encoded, offset)?,
    ]
    .into();
    let scale_opacity = [
        read_next_f32(encoded, offset)?,
        read_next_f32(encoded, offset)?,
        read_next_f32(encoded, offset)?,
        read_next_f32(encoded, offset)?,
    ]
    .into();
    Ok(Gaussian3d {
        position_visibility,
        spherical_harmonic: crate::material::spherical_harmonics::SphericalHarmonicCoefficients {
            coefficients,
        },
        rotation,
        scale_opacity,
    })
}

fn read_next_f32(encoded: &[u8], offset: &mut usize) -> Result<f32, LodCodecError> {
    let value = read_f32(encoded, *offset)?;
    *offset = (*offset)
        .checked_add(size_of::<f32>())
        .ok_or(LodCodecError::LengthOverflow)?;
    Ok(value)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, LodCodecError> {
    Ok(u16::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, LodCodecError> {
    Ok(u32::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, LodCodecError> {
    Ok(u64::from_le_bytes(read_array(bytes, offset)?))
}

fn read_f32(bytes: &[u8], offset: usize) -> Result<f32, LodCodecError> {
    Ok(f32::from_le_bytes(read_array(bytes, offset)?))
}

fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], LodCodecError> {
    let end = offset.checked_add(N).ok_or(LodCodecError::LengthOverflow)?;
    bytes
        .get(offset..end)
        .ok_or(LodCodecError::Truncated("numeric field"))?
        .try_into()
        .map_err(|_| LodCodecError::Truncated("numeric field"))
}

fn enforce_limit(field: &'static str, actual: u64, limit: u64) -> Result<(), LodCodecError> {
    if actual > limit {
        Err(LodCodecError::LimitExceeded {
            field,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn check_manifest_count(
    field: &'static str,
    encoded: u32,
    payload_header: u32,
    decoded: usize,
) -> Result<(), LodCodecError> {
    let decoded = u64::try_from(decoded).map_err(|_| LodCodecError::LengthOverflow)?;
    if encoded != payload_header || u64::from(encoded) != decoded {
        Err(LodCodecError::ManifestCountMismatch {
            field,
            encoded,
            payload_header,
            decoded,
        })
    } else {
        Ok(())
    }
}

fn checksum64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LodCodecError {
    InvalidLimits,
    InvalidMagic(&'static str),
    UnsupportedContainerVersion(u16),
    UnsupportedShardContainerVersion(u16),
    UnsupportedManifestEncoding(u8),
    EncodingUnavailable(ManifestEncoding),
    UnsupportedPageSchema(u16),
    InvalidPageId(LodPageId),
    EmptyPage,
    NonZeroReservedBytes,
    IncompatibleSphericalHarmonics {
        encoded_coefficients: u32,
        supported_coefficients: u32,
    },
    InvalidCompressedShDegree(u8),
    InvalidCompressedShCoefficientCount(u32),
    PageEncodingMismatch {
        expected: LodPageEncoding,
        actual: LodPageEncoding,
    },
    ShardEntryLimitExceeded {
        actual: u32,
        limit: u32,
    },
    InvalidShardIndex(String),
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
    ManifestCountMismatch {
        field: &'static str,
        encoded: u32,
        payload_header: u32,
        decoded: u64,
    },
    ChecksumMismatch {
        expected: u64,
        actual: u64,
    },
    Serialize(String),
    Deserialize(String),
    InvalidGaussian {
        index: usize,
        field: String,
    },
    ManifestValidation(String),
    PageValidation(String),
}

impl fmt::Display for LodCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => write!(f, "LoD codec limits are invalid"),
            Self::InvalidMagic(kind) => write!(f, "invalid {kind} container magic"),
            Self::UnsupportedContainerVersion(version) => {
                write!(f, "unsupported LoD container version {version}")
            }
            Self::UnsupportedShardContainerVersion(version) => {
                write!(f, "unsupported LoD shard container version {version}")
            }
            Self::UnsupportedManifestEncoding(encoding) => {
                write!(f, "unsupported LoD manifest encoding {encoding}")
            }
            Self::EncodingUnavailable(encoding) => {
                write!(f, "LoD manifest encoding {encoding:?} is not compiled in")
            }
            Self::UnsupportedPageSchema(version) => {
                write!(f, "unsupported LoD page schema {version}")
            }
            Self::InvalidPageId(id) => write!(f, "LoD page ID {:?} is invalid", id),
            Self::EmptyPage => write!(f, "LoD page must contain at least one Gaussian"),
            Self::NonZeroReservedBytes => write!(f, "LoD reserved header bytes must be zero"),
            Self::IncompatibleSphericalHarmonics {
                encoded_coefficients,
                supported_coefficients,
            } => write!(
                f,
                "LoD page contains {encoded_coefficients} spherical-harmonic coefficients, but this build supports {supported_coefficients}"
            ),
            Self::InvalidCompressedShDegree(degree) => write!(
                f,
                "compressed LoD page SH degree {degree} exceeds the compiled layout"
            ),
            Self::InvalidCompressedShCoefficientCount(count) => write!(
                f,
                "compressed LoD page SH coefficient count {count} does not identify a supported degree"
            ),
            Self::PageEncodingMismatch { expected, actual } => write!(
                f,
                "LoD page container encoding {actual:?} does not match descriptor {expected:?}"
            ),
            Self::ShardEntryLimitExceeded { actual, limit } => write!(
                f,
                "LoD shard range-table entry count {actual} exceeds limit {limit}"
            ),
            Self::InvalidShardIndex(message) => {
                write!(f, "invalid LoD shard range table: {message}")
            }
            Self::Truncated(field) => write!(f, "truncated LoD {field}"),
            Self::LengthOverflow => write!(f, "LoD encoded length overflowed"),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "LoD length is {actual}, expected {expected}")
            }
            Self::LimitExceeded {
                field,
                actual,
                limit,
            } => write!(f, "LoD {field} {actual} exceeds configured limit {limit}"),
            Self::ManifestCountMismatch {
                field,
                encoded,
                payload_header,
                decoded,
            } => write!(
                f,
                "LoD manifest {field} count is {encoded} in the container, {payload_header} in the payload header, and {decoded} after decoding"
            ),
            Self::ChecksumMismatch { expected, actual } => write!(
                f,
                "LoD checksum {actual:#018x} does not match {expected:#018x}"
            ),
            Self::Serialize(error) => write!(f, "failed to serialize LoD data: {error}"),
            Self::Deserialize(error) => write!(f, "failed to deserialize LoD data: {error}"),
            Self::InvalidGaussian { index, field } => {
                write!(f, "LoD Gaussian {index} has invalid {field}")
            }
            Self::ManifestValidation(error) => write!(f, "invalid LoD manifest: {error}"),
            Self::PageValidation(error) => write!(f, "invalid LoD page: {error}"),
        }
    }
}

impl Error for LodCodecError {}

#[derive(Debug)]
pub enum LodAssetLoaderError {
    Io(std::io::Error),
    Codec(LodCodecError),
}

impl fmt::Display for LodAssetLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "LoD asset IO failed: {error}"),
            Self::Codec(error) => write!(f, "LoD asset decode failed: {error}"),
        }
    }
}

impl Error for LodAssetLoaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Codec(error) => Some(error),
        }
    }
}

impl From<std::io::Error> for LodAssetLoaderError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<LodCodecError> for LodAssetLoaderError {
    fn from(value: LodCodecError) -> Self {
        Self::Codec(value)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use bevy::asset::io::VecReader;

    use super::*;
    use crate::{
        gaussian::formats::{
            planar_3d::PlanarGaussian3d,
            planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
        },
        testing::LodTestScene,
    };

    fn fixture() -> crate::gaussian::formats::planar_3d_lod::PlanarGaussian3dLod {
        let scene = LodTestScene::nested_octants(2);
        let cloud = PlanarGaussian3d::from(
            scene
                .gaussians
                .into_iter()
                .map(|entry| entry.gaussian)
                .collect::<Vec<_>>(),
        );
        build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                leaf_capacity: 8,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn manifest_round_trips_with_strict_validation() {
        let built = fixture();
        for encoding in [ManifestEncoding::Json, ManifestEncoding::Flexbuffers] {
            if encoding == ManifestEncoding::Flexbuffers && !cfg!(feature = "io_flexbuffers") {
                continue;
            }
            let encoded = encode_manifest_with_encoding(&built.manifest, encoding).unwrap();
            assert_eq!(read_u16(&encoded, 8).unwrap(), MANIFEST_CONTAINER_VERSION);
            assert_eq!(
                read_u32(&encoded, 16).unwrap(),
                built.manifest.header.node_count
            );
            assert_eq!(
                read_u32(&encoded, 20).unwrap(),
                built.manifest.header.page_count
            );
            let decoded = decode_manifest(&encoded, LodCodecLimits::default()).unwrap();
            assert_eq!(decoded, built.manifest);
        }
    }

    #[test]
    fn page_round_trips_exact_float_bits() {
        let built = fixture();
        for page in &built.pages {
            let encoded = encode_page(page).unwrap();
            let descriptor = built
                .manifest
                .pages
                .iter()
                .find(|descriptor| descriptor.id == page.id)
                .unwrap();
            let decoded =
                decode_page_with_descriptor(&encoded, descriptor, LodCodecLimits::default())
                    .unwrap();
            assert_eq!(&decoded, page);
        }
    }

    #[test]
    fn f16_sh_page_round_trip_expands_to_f32_and_zeros_higher_bands() {
        fn retained_degree(compiled_degree: usize) -> u8 {
            compiled_degree.min(1) as u8
        }

        let mut built = fixture();
        let page = &mut built.pages[0];
        for (index, coefficient) in page.gaussians[0]
            .spherical_harmonic
            .coefficients
            .iter_mut()
            .enumerate()
        {
            *coefficient = index as f32 * 0.03125 + 0.1;
        }
        let degree = retained_degree(SH_DEGREE);
        let encoding = LodPageEncoding::F16Sh { degree };
        let encoded = encode_page_with_encoding(page, encoding).unwrap();
        assert_eq!(
            read_u16(&encoded, 8).unwrap(),
            F16_SH_PAGE_CONTAINER_VERSION
        );
        let decoded = decode_page(&encoded, LodCodecLimits::default()).unwrap();
        let retained = sh_coefficient_count_for_degree(degree).unwrap();
        for (index, actual) in decoded.gaussians[0]
            .spherical_harmonic
            .coefficients
            .iter()
            .enumerate()
        {
            let expected = if index < retained {
                half::f16::from_f32(page.gaussians[0].spherical_harmonic.coefficients[index])
                    .to_f32()
            } else {
                0.0
            };
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        let mut descriptor = built
            .manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == page.id)
            .unwrap()
            .clone();
        descriptor.encoding = encoding;
        descriptor.content_hash = decoded.content_hash();
        descriptor.storage = None;
        assert_eq!(descriptor.effective_sh_degree(), degree);
        assert_eq!(
            decode_page_with_descriptor(&encoded, &descriptor, LodCodecLimits::default()).unwrap(),
            decoded
        );
        let mut wrong = descriptor;
        wrong.encoding = LodPageEncoding::F32Planar;
        assert!(matches!(
            decode_page_with_descriptor(&encoded, &wrong, LodCodecLimits::default()),
            Err(LodCodecError::PageEncodingMismatch { .. })
        ));
    }

    #[test]
    fn shard_index_round_trips_contiguous_absolute_ranges() {
        let prefix_len = lod_shard_prefix_len(2).unwrap();
        let entries = vec![
            LodShardEntry {
                page_id: LodPageId(1),
                byte_offset: prefix_len,
                encoded_len: 101,
                content_hash: 11,
            },
            LodShardEntry {
                page_id: LodPageId(2),
                byte_offset: prefix_len + 101,
                encoded_len: 203,
                content_hash: 22,
            },
        ];
        let index = LodShardIndex {
            file_len: prefix_len + 304,
            entries,
        };
        let encoded = encode_lod_shard_index(&index).unwrap();
        assert_eq!(encoded.len() as u64, prefix_len);
        assert_eq!(
            decode_lod_shard_index(&encoded, index.file_len, 2).unwrap(),
            index
        );
        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(matches!(
            decode_lod_shard_index(&corrupt, index.file_len, 2),
            Err(LodCodecError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn checksum_and_limits_reject_untrusted_data() {
        let built = fixture();
        let mut encoded = encode_page(&built.pages[0]).unwrap();
        *encoded.last_mut().unwrap() ^= 1;
        assert!(matches!(
            decode_page(&encoded, LodCodecLimits::default()),
            Err(LodCodecError::ChecksumMismatch { .. })
                | Err(LodCodecError::InvalidGaussian { .. })
        ));

        let encoded = encode_manifest(&built.manifest).unwrap();
        let mut limits = LodCodecLimits::default();
        limits.max_manifest_bytes = MANIFEST_HEADER_LEN as u64;
        assert!(matches!(
            decode_manifest(&encoded, limits),
            Err(LodCodecError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn manifest_count_limits_are_enforced_before_payload_decoding() {
        let built = fixture();
        let mut encoded = encode_manifest(&built.manifest).unwrap();
        let mut limits = LodCodecLimits::default();
        limits.max_nodes = built.manifest.header.node_count;
        let advertised = limits.max_nodes.checked_add(1).unwrap();
        encoded[16..20].copy_from_slice(&advertised.to_le_bytes());
        // A later checksum failure must not mask the outer-header allocation
        // gate. In particular, serde is never reached for this input.
        encoded[MANIFEST_HEADER_LEN] ^= 0xff;
        assert_eq!(
            decode_manifest(&encoded, limits),
            Err(LodCodecError::LimitExceeded {
                field: "manifest nodes",
                actual: u64::from(advertised),
                limit: u64::from(limits.max_nodes),
            })
        );
    }

    #[test]
    fn manifest_outer_counts_must_match_the_payload() {
        let built = fixture();
        let mut encoded = encode_manifest(&built.manifest).unwrap();
        let encoded_count = built.manifest.header.node_count.checked_add(1).unwrap();
        encoded[16..20].copy_from_slice(&encoded_count.to_le_bytes());
        assert_eq!(
            decode_manifest(&encoded, LodCodecLimits::default()),
            Err(LodCodecError::ManifestCountMismatch {
                field: "nodes",
                encoded: encoded_count,
                payload_header: built.manifest.header.node_count,
                decoded: built.manifest.nodes.len() as u64,
            })
        );
    }

    #[test]
    fn rejects_truncation_and_incompatible_sh_layout() {
        assert!(matches!(
            decode_page(&[0; 4], LodCodecLimits::default()),
            Err(LodCodecError::Truncated(_))
        ));
        let built = fixture();
        let mut page = encode_page(&built.pages[0]).unwrap();
        assert_eq!(
            u32::from_le_bytes(page[24..28].try_into().unwrap()),
            PAGE_SH_COEFFICIENT_COUNT
        );
        // SH0 stores four padded coefficients; SH3 stores forty-eight. Using
        // another real layout here exercises the cross-feature ABI gate
        // without checking a generated fixture into the repository.
        let incompatible_coefficients: u32 = if PAGE_SH_COEFFICIENT_COUNT == 4 {
            48
        } else {
            4
        };
        page[24..28].copy_from_slice(&incompatible_coefficients.to_le_bytes());
        assert_eq!(
            decode_page(&page, LodCodecLimits::default()),
            Err(LodCodecError::IncompatibleSphericalHarmonics {
                encoded_coefficients: incompatible_coefficients,
                supported_coefficients: PAGE_SH_COEFFICIENT_COUNT,
            })
        );
    }

    #[test]
    fn rejects_nonzero_manifest_reserved_fields() {
        let built = fixture();
        let mut manifest = encode_manifest(&built.manifest).unwrap();
        manifest[11] = 1;
        assert_eq!(
            decode_manifest(&manifest, LodCodecLimits::default()),
            Err(LodCodecError::NonZeroReservedBytes)
        );
    }

    #[test]
    fn loader_read_is_bounded_before_allocating_the_full_source() {
        let mut exact = VecReader::new(vec![1, 2, 3, 4]);
        assert_eq!(
            pollster::block_on(read_bounded(&mut exact, 4)).unwrap(),
            [1, 2, 3, 4]
        );

        let mut oversized = VecReader::new(vec![0; 1_000_000]);
        assert!(matches!(
            pollster::block_on(read_bounded(&mut oversized, 32)),
            Err(LodAssetLoaderError::Codec(LodCodecError::LimitExceeded {
                actual: 33,
                ..
            }))
        ));
    }
}
