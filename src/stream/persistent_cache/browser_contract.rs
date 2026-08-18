use super::*;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct BrowserCacheIndex {
    pub(super) next_epoch: u64,
    pub(super) entries: BTreeMap<PersistentCacheKey, BrowserCacheIndexEntry>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BrowserCacheIndexEntry {
    pub(super) file_bytes: u64,
    pub(super) last_used: u64,
}

impl BrowserCacheIndex {
    pub(super) fn bytes(&self) -> Result<u64, PersistentCacheError> {
        self.entries.values().try_fold(0_u64, |total, entry| {
            total
                .checked_add(entry.file_bytes)
                .ok_or(PersistentCacheError::ByteCountOverflow)
        })
    }

    pub(super) fn take_epoch(&mut self) -> u64 {
        let epoch = self.next_epoch.max(1);
        self.next_epoch = epoch.wrapping_add(1).max(1);
        epoch
    }
}

pub(super) const BROWSER_INDEX_MAGIC: [u8; 8] = *b"BGSLIDX\0";
pub(super) const BROWSER_INDEX_HEADER_LEN: usize = 32;
pub(super) const BROWSER_INDEX_ENTRY_LEN: usize = 56;
pub(super) const BROWSER_INDEX_DIRTY_FLAG: u16 = 1;
#[cfg(test)]
pub(super) fn encode_browser_index(
    index: &BrowserCacheIndex,
) -> Result<Vec<u8>, PersistentCacheError> {
    encode_browser_index_with_flags(index, 0)
}

pub(super) fn encode_browser_index_with_flags(
    index: &BrowserCacheIndex,
    flags: u16,
) -> Result<Vec<u8>, PersistentCacheError> {
    let entry_bytes = index
        .entries
        .len()
        .checked_mul(BROWSER_INDEX_ENTRY_LEN)
        .ok_or(PersistentCacheError::ByteCountOverflow)?;
    let capacity = BROWSER_INDEX_HEADER_LEN
        .checked_add(entry_bytes)
        .and_then(|value| value.checked_add(8))
        .ok_or(PersistentCacheError::ByteCountOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| PersistentCacheError::IndexAllocationFailed(capacity as u64))?;
    bytes.extend_from_slice(&BROWSER_INDEX_MAGIC);
    bytes.extend_from_slice(&CACHE_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&flags.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&index.next_epoch.to_le_bytes());
    bytes.extend_from_slice(&(index.entries.len() as u64).to_le_bytes());
    for (key, entry) in &index.entries {
        bytes.extend_from_slice(&key.package_hash.to_le_bytes());
        bytes.extend_from_slice(&key.page_id.0.to_le_bytes());
        bytes.extend_from_slice(&key.content_hash.to_le_bytes());
        bytes.extend_from_slice(&key.encoded_len.to_le_bytes());
        bytes.extend_from_slice(&entry.file_bytes.to_le_bytes());
        bytes.extend_from_slice(&entry.last_used.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
    }
    debug_assert_eq!(bytes.len(), capacity - 8);
    let checksum = page_checksum64(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    Ok(bytes)
}

pub(super) fn decode_browser_index(
    bytes: &[u8],
    max_entries: u32,
) -> Result<BrowserCacheIndex, PersistentCacheError> {
    if bytes.len() < BROWSER_INDEX_HEADER_LEN + 8 || bytes[..8] != BROWSER_INDEX_MAGIC {
        return Err(PersistentCacheError::BrowserIndexCorrupt(
            "missing or invalid index header".to_owned(),
        ));
    }
    let version = read_u16(bytes, 8);
    if version != CACHE_FORMAT_VERSION {
        return Err(PersistentCacheError::BrowserIndexCorrupt(format!(
            "unsupported index version {version}"
        )));
    }
    let flags = read_u16(bytes, 10);
    if flags != 0 {
        return Err(PersistentCacheError::BrowserIndexCorrupt(format!(
            "index is marked incomplete with flags {flags:#06x}"
        )));
    }
    let entry_count = read_u64(bytes, 24);
    if entry_count > u64::from(max_entries) {
        return Err(PersistentCacheError::BrowserIndexCorrupt(format!(
            "index has {entry_count} entries above {max_entries} bound"
        )));
    }
    let entry_count = usize::try_from(entry_count).map_err(|_| {
        PersistentCacheError::BrowserIndexCorrupt("entry count overflow".to_owned())
    })?;
    let expected = BROWSER_INDEX_HEADER_LEN
        .checked_add(
            entry_count
                .checked_mul(BROWSER_INDEX_ENTRY_LEN)
                .ok_or_else(|| {
                    PersistentCacheError::BrowserIndexCorrupt("index length overflow".to_owned())
                })?,
        )
        .and_then(|value| value.checked_add(8))
        .ok_or_else(|| {
            PersistentCacheError::BrowserIndexCorrupt("index length overflow".to_owned())
        })?;
    if bytes.len() != expected {
        return Err(PersistentCacheError::BrowserIndexCorrupt(format!(
            "index length {} does not match {expected}",
            bytes.len()
        )));
    }
    let expected_checksum = read_u64(bytes, expected - 8);
    let actual_checksum = page_checksum64(&bytes[..expected - 8]);
    if actual_checksum != expected_checksum {
        return Err(PersistentCacheError::BrowserIndexCorrupt(
            "index checksum mismatch".to_owned(),
        ));
    }
    let mut entries = BTreeMap::new();
    let mut offset = BROWSER_INDEX_HEADER_LEN;
    for _ in 0..entry_count {
        let key = PersistentCacheKey {
            package_hash: read_u64(bytes, offset),
            page_id: LodPageId(read_u64(bytes, offset + 8)),
            content_hash: read_u64(bytes, offset + 16),
            encoded_len: read_u64(bytes, offset + 24),
        };
        let entry = BrowserCacheIndexEntry {
            file_bytes: read_u64(bytes, offset + 32),
            last_used: read_u64(bytes, offset + 40),
        };
        if !key.page_id.is_valid()
            || key.encoded_len == 0
            || record_file_bytes(key.encoded_len) != Some(entry.file_bytes)
            || entries.insert(key, entry).is_some()
        {
            return Err(PersistentCacheError::BrowserIndexCorrupt(
                "index entry is invalid or duplicated".to_owned(),
            ));
        }
        offset += BROWSER_INDEX_ENTRY_LEN;
    }
    Ok(BrowserCacheIndex {
        next_epoch: read_u64(bytes, 16).max(1),
        entries,
    })
}
pub(super) fn is_browser_cache_bypass_error(error: &PersistentCacheError) -> bool {
    matches!(
        error,
        PersistentCacheError::BrowserCacheOperationTimedOut { .. }
            | PersistentCacheError::BrowserCacheTemporarilyBypassed
    )
}

pub(super) fn browser_cache_queue_is_full(pending: usize, active: bool, maximum: u32) -> bool {
    pending.saturating_add(usize::from(active)) >= maximum as usize
}
pub(super) fn bounded_cache_chunk_end(
    current: usize,
    chunk: usize,
    maximum: u64,
) -> Result<usize, PersistentCacheCorruptionReason> {
    let end =
        current
            .checked_add(chunk)
            .ok_or(PersistentCacheCorruptionReason::FileLengthMismatch {
                expected: maximum,
                actual: u64::MAX,
            })?;
    if end as u64 > maximum {
        return Err(PersistentCacheCorruptionReason::FileLengthMismatch {
            expected: maximum,
            actual: end as u64,
        });
    }
    Ok(end)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::super::{
        CACHE_HEADER_LEN, CacheRecordHeader, LodPageId, MAX_PERSISTENT_CACHE_PENDING_OPERATIONS,
        PersistentCacheCorruptionReason, PersistentCacheError, PersistentCacheKey,
        PersistentCachePackageIdentity, PersistentCachePageIdentity,
        validate_service_queue_capacity,
    };
    use super::*;

    fn identity(encoded_len: u64) -> PersistentCachePageIdentity {
        PersistentCachePageIdentity {
            package: PersistentCachePackageIdentity {
                manifest_version: 1,
                page_schema_version: 1,
                required_features: 0,
                source_gaussian_count: 1,
                stored_gaussian_count: 1,
                source_fingerprint: 2,
                config_fingerprint: 3,
                builder_abi_version: 1,
                reducer_version: 1,
                package_version: None,
            },
            page_id: LodPageId(1),
            content_hash: 4,
            encoded_len,
        }
    }

    #[test]
    fn browser_index_codec_is_bounded_deterministic_and_checksummed() {
        let base = identity(4);
        let first = PersistentCachePageIdentity {
            page_id: LodPageId(1),
            content_hash: 11,
            ..base.clone()
        }
        .key()
        .unwrap();
        let second = PersistentCachePageIdentity {
            page_id: LodPageId(2),
            content_hash: 22,
            ..base
        }
        .key()
        .unwrap();
        let index = BrowserCacheIndex {
            next_epoch: 17,
            entries: BTreeMap::from([
                (
                    second,
                    BrowserCacheIndexEntry {
                        file_bytes: CACHE_HEADER_LEN as u64 + second.encoded_len,
                        last_used: 9,
                    },
                ),
                (
                    first,
                    BrowserCacheIndexEntry {
                        file_bytes: CACHE_HEADER_LEN as u64 + first.encoded_len,
                        last_used: 4,
                    },
                ),
            ]),
        };
        assert_eq!(
            index.bytes().unwrap(),
            2 * (CACHE_HEADER_LEN as u64 + first.encoded_len)
        );
        let mut epoch_index = index.clone();
        assert_eq!(epoch_index.take_epoch(), 17);
        assert_eq!(epoch_index.take_epoch(), 18);
        let encoded = encode_browser_index(&index).unwrap();
        assert_eq!(decode_browser_index(&encoded, 2).unwrap(), index);
        assert_eq!(encoded, encode_browser_index(&index).unwrap());
        let dirty = encode_browser_index_with_flags(&index, BROWSER_INDEX_DIRTY_FLAG).unwrap();
        assert!(matches!(
            decode_browser_index(&dirty, 2),
            Err(PersistentCacheError::BrowserIndexCorrupt(message))
                if message.contains("incomplete")
        ));
        let mut corrupt = encoded;
        corrupt[BROWSER_INDEX_HEADER_LEN + 1] ^= 1;
        assert!(matches!(
            decode_browser_index(&corrupt, 2),
            Err(PersistentCacheError::BrowserIndexCorrupt(_))
        ));
        assert!(matches!(
            decode_browser_index(&encode_browser_index(&index).unwrap(), 1),
            Err(PersistentCacheError::BrowserIndexCorrupt(_))
        ));
    }

    #[test]
    fn record_lengths_fail_typed_on_u64_overflow() {
        let identity = identity(u64::MAX);
        assert_eq!(identity.key(), Err(PersistentCacheError::ByteCountOverflow));

        let overflowing_key = PersistentCacheKey {
            package_hash: 1,
            page_id: LodPageId(1),
            content_hash: 2,
            encoded_len: u64::MAX,
        };
        let header = CacheRecordHeader {
            key: overflowing_key,
            payload_checksum: 3,
            payload_len: u64::MAX,
        }
        .encode();
        assert!(matches!(
            CacheRecordHeader::decode(&header),
            Err(PersistentCacheCorruptionReason::RecordLengthOverflow)
        ));

        let index = BrowserCacheIndex {
            next_epoch: 1,
            entries: BTreeMap::from([(
                overflowing_key,
                BrowserCacheIndexEntry {
                    file_bytes: u64::MAX,
                    last_used: 1,
                },
            )]),
        };
        let encoded = encode_browser_index(&index).unwrap();
        assert!(matches!(
            decode_browser_index(&encoded, 1),
            Err(PersistentCacheError::BrowserIndexCorrupt(_))
        ));
    }

    #[test]
    fn browser_cache_chunk_accounting_rejects_oversized_corrupt_record() {
        assert_eq!(bounded_cache_chunk_end(2, 2, 4).unwrap(), 4);
        assert_eq!(
            bounded_cache_chunk_end(4, 1, 4),
            Err(PersistentCacheCorruptionReason::FileLengthMismatch {
                expected: 4,
                actual: 5,
            })
        );
    }

    #[test]
    fn browser_cache_queue_counts_active_and_pending_operations() {
        assert!(!browser_cache_queue_is_full(0, false, 2));
        assert!(!browser_cache_queue_is_full(0, true, 2));
        assert!(browser_cache_queue_is_full(1, true, 2));
        assert!(browser_cache_queue_is_full(2, false, 2));
        assert_eq!(
            validate_service_queue_capacity(MAX_PERSISTENT_CACHE_PENDING_OPERATIONS + 1),
            Err(PersistentCacheError::ServiceQueueCapacityTooLarge {
                configured: MAX_PERSISTENT_CACHE_PENDING_OPERATIONS + 1,
                maximum: MAX_PERSISTENT_CACHE_PENDING_OPERATIONS,
            })
        );
    }

    #[test]
    fn browser_cache_bypass_is_narrowly_typed() {
        assert!(is_browser_cache_bypass_error(
            &PersistentCacheError::BrowserCacheOperationTimedOut {
                timeout_millis: 30_000,
            }
        ));
        assert!(is_browser_cache_bypass_error(
            &PersistentCacheError::BrowserCacheTemporarilyBypassed
        ));
        assert!(!is_browser_cache_bypass_error(
            &PersistentCacheError::BrowserCoordinationUnavailable("missing Web Locks".to_owned())
        ));
        assert!(!is_browser_cache_bypass_error(
            &PersistentCacheError::BrowserStorage("quota backend failed".to_owned())
        ));
    }
}
