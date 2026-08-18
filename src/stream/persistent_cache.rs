//! Persistent, content-addressed cache for encoded LoD pages.
//!
//! Cache keys bind package/build identity, logical page identity, decoded-page
//! content hash, and encoded length. The URL is deliberately not an identity:
//! immutable mirrors can share bytes, while a changed manifest cannot reuse a
//! stale page from the same URL. Native records are checksummed, written by
//! same-directory atomic rename, bounded by actual on-disk bytes, and evicted by
//! deterministic LRU. Record corruption is removed during lookup. Full codec
//! and manifest-page validation stays in the bounded page preprocessor; a typed
//! invalidation handoff removes any encoded record rejected there before retry.

use std::{collections::BTreeMap, fmt, path::PathBuf};

use crate::gaussian::formats::planar_3d_lod::GaussianLodManifest;

use super::transport::{
    LodPageId, LodPageTransport, PagePayload, PagePoll, PageRequest, page_checksum64,
};

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{
    NativePersistentCacheConfig, NativePersistentCacheService, NativePersistentPageCache,
    PersistentCachePageTransport, SharedPersistentCachePageTransport,
};

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_contract;
#[cfg(target_arch = "wasm32")]
pub use browser::{
    BROWSER_PERSISTENT_CACHE_GLOBAL_OPERATION_CAPACITY, BrowserPersistentCacheConfig,
    BrowserPersistentCachePageTransport, BrowserPersistentCachePoll, BrowserPersistentPageCache,
    SharedBrowserPersistentCachePageTransport,
};

const CACHE_MAGIC: [u8; 8] = *b"BGSLCHE\0";
const CACHE_FORMAT_VERSION: u16 = 1;
const CACHE_HEADER_LEN: usize = 80;
const CACHE_EXTENSION: &str = "lodpcache";
pub const MAX_PERSISTENT_CACHE_ENTRIES: u32 = 1_000_000;
/// Hard bound applied before native channel allocation or browser queue use.
pub const MAX_PERSISTENT_CACHE_PENDING_OPERATIONS: u32 = 65_536;
/// Process/tab-wide bound on independently named persistent-cache
/// coordinators. Coordinators deliberately outlive package entities so an old
/// operation and a newly spawned package can never mutate one namespace from
/// two independent workers.
pub const MAX_PERSISTENT_CACHE_SERVICES: usize = 64;
const fn record_file_bytes(encoded_len: u64) -> Option<u64> {
    (CACHE_HEADER_LEN as u64).checked_add(encoded_len)
}

/// Stable package/build identity. It intentionally excludes transport URLs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentCachePackageIdentity {
    pub manifest_version: u16,
    pub page_schema_version: u16,
    pub required_features: u64,
    pub source_gaussian_count: u64,
    pub stored_gaussian_count: u64,
    pub source_fingerprint: u64,
    pub config_fingerprint: u64,
    pub builder_abi_version: u32,
    pub reducer_version: u32,
    /// Optional signed-index version, manifest container checksum, or CDN
    /// deployment version supplied by the package owner.
    pub package_version: Option<String>,
}

impl PersistentCachePackageIdentity {
    pub fn from_manifest(manifest: &GaussianLodManifest) -> Result<Self, PersistentCacheError> {
        manifest
            .validate()
            .map_err(|error| PersistentCacheError::InvalidManifest(error.to_string()))?;
        Ok(Self {
            manifest_version: manifest.header.manifest_version,
            page_schema_version: manifest.header.page_schema_version,
            required_features: manifest.header.required_features,
            source_gaussian_count: manifest.header.source_gaussian_count,
            stored_gaussian_count: manifest.header.stored_gaussian_count,
            source_fingerprint: manifest.build.source_fingerprint,
            config_fingerprint: manifest.build.config_fingerprint,
            builder_abi_version: manifest.build.builder_abi_version,
            reducer_version: manifest.build.reducer_version,
            package_version: None,
        })
    }

    pub fn with_package_version(
        mut self,
        version: impl Into<String>,
    ) -> Result<Self, PersistentCacheError> {
        let version = version.into();
        if version.is_empty() || version.bytes().any(|byte| byte.is_ascii_control()) {
            return Err(PersistentCacheError::InvalidPackageVersion);
        }
        self.package_version = Some(version);
        Ok(self)
    }

    pub fn stable_hash(&self) -> u64 {
        let mut hash = CacheHasher::new();
        hash.write(b"bevy-gaussian-splatting persistent LoD package v1");
        hash.write(&self.manifest_version.to_le_bytes());
        hash.write(&self.page_schema_version.to_le_bytes());
        hash.write(&self.required_features.to_le_bytes());
        hash.write(&self.source_gaussian_count.to_le_bytes());
        hash.write(&self.stored_gaussian_count.to_le_bytes());
        hash.write(&self.source_fingerprint.to_le_bytes());
        hash.write(&self.config_fingerprint.to_le_bytes());
        hash.write(&self.builder_abi_version.to_le_bytes());
        hash.write(&self.reducer_version.to_le_bytes());
        match self.package_version.as_deref() {
            Some(version) => {
                hash.write(&[1]);
                hash.write(&(version.len() as u64).to_le_bytes());
                hash.write(version.as_bytes());
            }
            None => hash.write(&[0]),
        }
        hash.finish()
    }
}

/// Full immutable identity required before encoded bytes may be reused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentCachePageIdentity {
    pub package: PersistentCachePackageIdentity,
    pub page_id: LodPageId,
    /// Stable decoded-page checksum from the signed/validated manifest.
    pub content_hash: u64,
    pub encoded_len: u64,
}

impl PersistentCachePageIdentity {
    pub fn key(&self) -> Result<PersistentCacheKey, PersistentCacheError> {
        if !self.page_id.is_valid() {
            return Err(PersistentCacheError::InvalidPageId);
        }
        if self.encoded_len == 0 {
            return Err(PersistentCacheError::ZeroEncodedLength(self.page_id));
        }
        record_file_bytes(self.encoded_len).ok_or(PersistentCacheError::ByteCountOverflow)?;
        Ok(PersistentCacheKey {
            package_hash: self.package.stable_hash(),
            page_id: self.page_id,
            content_hash: self.content_hash,
            encoded_len: self.encoded_len,
        })
    }
}

/// Validated page-identity index copied from a manifest.
#[derive(Clone, Debug)]
pub struct PersistentCachePageIdentities {
    entries: BTreeMap<LodPageId, PersistentCachePageIdentity>,
}

impl PersistentCachePageIdentities {
    pub fn from_manifest(manifest: &GaussianLodManifest) -> Result<Self, PersistentCacheError> {
        Self::from_manifest_with_package_identity(
            manifest,
            PersistentCachePackageIdentity::from_manifest(manifest)?,
        )
    }

    pub fn from_manifest_with_package_identity(
        manifest: &GaussianLodManifest,
        package: PersistentCachePackageIdentity,
    ) -> Result<Self, PersistentCacheError> {
        manifest
            .validate()
            .map_err(|error| PersistentCacheError::InvalidManifest(error.to_string()))?;
        let mut entries = BTreeMap::new();
        for descriptor in &manifest.pages {
            let storage = descriptor
                .storage
                .as_ref()
                .ok_or(PersistentCacheError::MissingStorage(descriptor.id))?;
            entries.insert(
                descriptor.id,
                PersistentCachePageIdentity {
                    package: package.clone(),
                    page_id: descriptor.id,
                    content_hash: descriptor.content_hash,
                    encoded_len: storage.encoded_len,
                },
            );
        }
        Ok(Self { entries })
    }

    pub fn get(&self, page_id: LodPageId) -> Option<&PersistentCachePageIdentity> {
        self.entries.get(&page_id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn validation(&self, page_id: LodPageId) -> Option<PersistentCachePageValidation> {
        Some(PersistentCachePageValidation {
            identity: self.entries.get(&page_id)?.clone(),
        })
    }
}

/// Encoded-byte contract used at the persistent-cache boundary.
///
/// Full page decoding belongs to `LodPagePreprocessor`; this contract checks
/// only immutable identity, length, and the payload's encoded-byte checksum.
#[derive(Clone, Debug)]
struct PersistentCachePageValidation {
    identity: PersistentCachePageIdentity,
}

impl PersistentCachePageValidation {
    fn validate(&self, payload: &PagePayload) -> Result<(), PersistentCacheError> {
        validate_payload_identity(&self.identity, payload)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PersistentCacheKey {
    package_hash: u64,
    page_id: LodPageId,
    content_hash: u64,
    encoded_len: u64,
}

impl PersistentCacheKey {
    pub fn file_name(self) -> String {
        format!(
            "v1-{:016x}-{:016x}-{:016x}-{:016x}.{CACHE_EXTENSION}",
            self.package_hash, self.page_id.0, self.content_hash, self.encoded_len
        )
    }
}

impl fmt::Display for PersistentCacheKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.file_name())
    }
}

fn validate_service_queue_capacity(capacity: u32) -> Result<usize, PersistentCacheError> {
    if capacity == 0 {
        return Err(PersistentCacheError::ZeroServiceQueueCapacity);
    }
    if capacity > MAX_PERSISTENT_CACHE_PENDING_OPERATIONS {
        return Err(PersistentCacheError::ServiceQueueCapacityTooLarge {
            configured: capacity,
            maximum: MAX_PERSISTENT_CACHE_PENDING_OPERATIONS,
        });
    }
    usize::try_from(capacity).map_err(|_| PersistentCacheError::ServiceQueueCapacityOverflow)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PersistentCacheStats {
    pub entries: u32,
    pub bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub writes: u64,
    pub evictions: u64,
    pub corruptions_recovered: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PersistentCacheLookup {
    Hit(PagePayload),
    Miss,
    /// The bad record was successfully removed and should be fetched again.
    CorruptionRecovered(PersistentCacheCorruption),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PersistentCacheInsert {
    Written { evicted: Vec<PersistentCacheKey> },
    AlreadyPresent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentCacheCorruption {
    pub key: PersistentCacheKey,
    pub reason: PersistentCacheCorruptionReason,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PersistentCacheCorruptionReason {
    TruncatedHeader,
    InvalidMagic,
    UnsupportedVersion(u16),
    HeaderKeyMismatch,
    RecordLengthOverflow,
    FileLengthMismatch { expected: u64, actual: u64 },
    PayloadChecksumMismatch { expected: u64, actual: u64 },
}

#[derive(Clone, Copy, Debug)]
struct CacheRecordHeader {
    key: PersistentCacheKey,
    payload_checksum: u64,
    payload_len: u64,
}

impl CacheRecordHeader {
    fn encode(self) -> [u8; CACHE_HEADER_LEN] {
        let mut bytes = [0_u8; CACHE_HEADER_LEN];
        bytes[0..8].copy_from_slice(&CACHE_MAGIC);
        bytes[8..10].copy_from_slice(&CACHE_FORMAT_VERSION.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.key.package_hash.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.key.page_id.0.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.key.content_hash.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.key.encoded_len.to_le_bytes());
        bytes[48..56].copy_from_slice(&self.payload_checksum.to_le_bytes());
        bytes[56..64].copy_from_slice(&self.payload_len.to_le_bytes());
        let header_checksum = page_checksum64(&bytes[..64]);
        bytes[64..72].copy_from_slice(&header_checksum.to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8; CACHE_HEADER_LEN]) -> Result<Self, PersistentCacheCorruptionReason> {
        if bytes[..8] != CACHE_MAGIC {
            return Err(PersistentCacheCorruptionReason::InvalidMagic);
        }
        let version = read_u16(bytes, 8);
        if version != CACHE_FORMAT_VERSION {
            return Err(PersistentCacheCorruptionReason::UnsupportedVersion(version));
        }
        let expected_header_checksum = read_u64(bytes, 64);
        let actual_header_checksum = page_checksum64(&bytes[..64]);
        if expected_header_checksum != actual_header_checksum {
            return Err(PersistentCacheCorruptionReason::HeaderKeyMismatch);
        }
        let key = PersistentCacheKey {
            package_hash: read_u64(bytes, 16),
            page_id: LodPageId(read_u64(bytes, 24)),
            content_hash: read_u64(bytes, 32),
            encoded_len: read_u64(bytes, 40),
        };
        let payload_len = read_u64(bytes, 56);
        if !key.page_id.is_valid() || key.encoded_len == 0 || payload_len != key.encoded_len {
            return Err(PersistentCacheCorruptionReason::HeaderKeyMismatch);
        }
        if record_file_bytes(payload_len).is_none() {
            return Err(PersistentCacheCorruptionReason::RecordLengthOverflow);
        }
        Ok(Self {
            key,
            payload_checksum: read_u64(bytes, 48),
            payload_len,
        })
    }
}
fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("fixed header"))
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("fixed header"))
}

struct CacheHasher(u64);

impl CacheHasher {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}
fn validated_transport_page<UpstreamError>(
    identities: &PersistentCachePageIdentities,
    request: PageRequest,
) -> Result<PersistentCachePageValidation, PersistentCacheTransportError<UpstreamError>> {
    let validation = identities.validation(request.page_id).ok_or(
        PersistentCacheTransportError::MissingIdentity(request.page_id),
    )?;
    let identity = &validation.identity;
    if request
        .expected_bytes
        .is_some_and(|expected| expected != identity.encoded_len)
    {
        return Err(PersistentCacheTransportError::RequestSizeMismatch {
            page: request.page_id,
            expected: request.expected_bytes.unwrap_or_default(),
            identity: identity.encoded_len,
        });
    }
    Ok(validation)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PersistentCacheTransportError<UpstreamError> {
    InvalidTicket(u64),
    MissingIdentity(LodPageId),
    RequestSizeMismatch {
        page: LodPageId,
        expected: u64,
        identity: u64,
    },
    SharedCacheUnavailable,
    Cache(PersistentCacheError),
    Upstream(UpstreamError),
}

impl<UpstreamError: fmt::Debug> fmt::Display for PersistentCacheTransportError<UpstreamError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl<UpstreamError: fmt::Debug> std::error::Error for PersistentCacheTransportError<UpstreamError> {}
fn validate_payload_identity(
    identity: &PersistentCachePageIdentity,
    payload: &PagePayload,
) -> Result<(), PersistentCacheError> {
    identity.key()?;
    if payload.page_id != identity.page_id {
        return Err(PersistentCacheError::PageIdMismatch {
            expected: identity.page_id,
            actual: payload.page_id,
        });
    }
    if payload.bytes.len() as u64 != identity.encoded_len {
        return Err(PersistentCacheError::EncodedLengthMismatch {
            page: identity.page_id,
            expected: identity.encoded_len,
            actual: payload.bytes.len() as u64,
        });
    }
    let actual = page_checksum64(&payload.bytes);
    if payload.checksum != actual {
        return Err(PersistentCacheError::PayloadChecksumMismatch {
            page: identity.page_id,
            expected: payload.checksum,
            actual,
        });
    }
    Ok(())
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PersistentCacheError {
    InvalidManifest(String),
    MissingStorage(LodPageId),
    InvalidPackageVersion,
    InvalidPageId,
    ZeroEncodedLength(LodPageId),
    InvalidRoot,
    RootIsNotDirectory(PathBuf),
    CacheRootAlreadyOwned(PathBuf),
    ZeroByteBudget,
    ZeroEntryBudget,
    EntryBudgetTooLarge {
        configured: u32,
        maximum: u32,
    },
    PageExceedsBudget {
        page: LodPageId,
        record_bytes: u64,
        max_bytes: u64,
    },
    PageIdMismatch {
        expected: LodPageId,
        actual: LodPageId,
    },
    EncodedLengthMismatch {
        page: LodPageId,
        expected: u64,
        actual: u64,
    },
    PayloadChecksumMismatch {
        page: LodPageId,
        expected: u64,
        actual: u64,
    },
    ByteCountOverflow,
    EntryCountOverflow,
    BudgetCannotBeSatisfied,
    ZeroServiceQueueCapacity,
    ServiceQueueCapacityTooLarge {
        configured: u32,
        maximum: u32,
    },
    ServiceQueueCapacityOverflow,
    CacheWorkerSpawn(String),
    CacheServiceInitialization(String),
    CacheServiceQueueFull,
    CacheServiceDisconnected,
    CacheServiceRegistryPoisoned,
    CacheServiceConfigConflict(String),
    CacheServiceRegistryFull {
        maximum: usize,
    },
    InvalidBrowserCacheName(String),
    BrowserStorageUnavailable,
    BrowserCoordinationUnavailable(String),
    BrowserQuotaExceeded(String),
    BrowserStorage(String),
    BrowserIndexCorrupt(String),
    BrowserCacheOperationTimedOut {
        timeout_millis: u32,
    },
    BrowserCacheTemporarilyBypassed,
    BrowserOperationKindMismatch,
    BrowserOperationCapacityExceeded {
        maximum: u32,
    },
    InvalidBrowserTicket(u64),
    IndexAllocationFailed(u64),
    Io(String),
}

impl fmt::Display for PersistentCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PersistentCacheError {}
