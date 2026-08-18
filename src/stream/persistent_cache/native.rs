use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap},
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::SystemTime,
};

use super::*;

const TEMP_PREFIX: &str = ".lod-cache-tmp-";
static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(1);

impl PersistentCacheError {
    fn io(error: impl fmt::Display) -> Self {
        Self::Io(error.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativePersistentCacheConfig {
    /// Explicit cache directory. Creation only occurs after the caller opts in.
    pub root: PathBuf,
    /// Hard bound over complete record files, including cache headers.
    pub max_bytes: u64,
    /// Hard metadata/file-count bound in addition to the byte budget.
    pub max_entries: u32,
}

impl NativePersistentCacheConfig {
    pub fn validate(&self) -> Result<(), PersistentCacheError> {
        if self.root.as_os_str().is_empty() {
            return Err(PersistentCacheError::InvalidRoot);
        }
        if self.max_bytes == 0 {
            return Err(PersistentCacheError::ZeroByteBudget);
        }
        if self.max_entries == 0 {
            return Err(PersistentCacheError::ZeroEntryBudget);
        }
        if self.max_entries > MAX_PERSISTENT_CACHE_ENTRIES {
            return Err(PersistentCacheError::EntryBudgetTooLarge {
                configured: self.max_entries,
                maximum: MAX_PERSISTENT_CACHE_ENTRIES,
            });
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
struct CacheEntry {
    path: PathBuf,
    file_bytes: u64,
    last_used: u64,
}

/// Native persistent cache. It is deliberately owned by one orchestration
/// object; callers should not create multiple writers for the same root.
pub struct NativePersistentPageCache {
    config: NativePersistentCacheConfig,
    // Held for the cache lifetime. This prevents a second process (or a path
    // alias that bypasses an in-process registry) from becoming another writer
    // for the same canonical directory.
    _lock_file: File,
    entries: BTreeMap<PersistentCacheKey, CacheEntry>,
    stats: PersistentCacheStats,
    next_epoch: u64,
}

impl Drop for NativePersistentPageCache {
    fn drop(&mut self) {
        let _ = self._lock_file.unlock();
    }
}

impl NativePersistentPageCache {
    pub fn open(config: NativePersistentCacheConfig) -> Result<Self, PersistentCacheError> {
        config.validate()?;
        fs::create_dir_all(&config.root).map_err(PersistentCacheError::io)?;
        let canonical_root = fs::canonicalize(&config.root).map_err(PersistentCacheError::io)?;
        if !canonical_root.is_dir() {
            return Err(PersistentCacheError::RootIsNotDirectory(canonical_root));
        }
        let lock_path = canonical_root.join(".lod-cache.lock");
        let lock_file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(PersistentCacheError::io)?;
        match lock_file.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(PersistentCacheError::CacheRootAlreadyOwned(canonical_root));
            }
            Err(std::fs::TryLockError::Error(error)) => {
                return Err(PersistentCacheError::io(error));
            }
        }
        let mut candidates = BinaryHeap::new();
        let candidate_limit = config.max_entries as usize;
        let mut recovered = 0_u64;
        let mut startup_evictions = 0_u64;
        for entry in fs::read_dir(&canonical_root).map_err(PersistentCacheError::io)? {
            let entry = entry.map_err(PersistentCacheError::io)?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name.starts_with(TEMP_PREFIX) {
                if path.is_file() {
                    fs::remove_file(&path).map_err(PersistentCacheError::io)?;
                    recovered += 1;
                }
                continue;
            }
            if path.extension().and_then(|value| value.to_str()) != Some(CACHE_EXTENSION) {
                continue;
            }
            let metadata = entry.metadata().map_err(PersistentCacheError::io)?;
            if !metadata.is_file() {
                continue;
            }
            let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            match read_header(&path) {
                Ok(header) if header.key.file_name() == name => {
                    let Some(expected) = record_file_bytes(header.payload_len) else {
                        fs::remove_file(&path).map_err(PersistentCacheError::io)?;
                        recovered += 1;
                        continue;
                    };
                    if metadata.len() != expected {
                        fs::remove_file(&path).map_err(PersistentCacheError::io)?;
                        recovered += 1;
                        continue;
                    }
                    candidates.push(Reverse((modified, header.key, path, metadata.len())));
                    if candidates.len() > candidate_limit
                        && let Some(Reverse((_, _, stale_path, _))) = candidates.pop()
                    {
                        fs::remove_file(stale_path).map_err(PersistentCacheError::io)?;
                        startup_evictions = startup_evictions.saturating_add(1);
                    }
                }
                Ok(_) | Err(ReadRecordError::Corrupt(_)) => {
                    fs::remove_file(&path).map_err(PersistentCacheError::io)?;
                    recovered += 1;
                }
                Err(ReadRecordError::Io(error)) => {
                    return Err(PersistentCacheError::Io(error));
                }
            }
        }
        let mut candidates = candidates
            .into_vec()
            .into_iter()
            .map(|Reverse(candidate)| candidate)
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let mut cache = Self {
            config: NativePersistentCacheConfig {
                root: canonical_root,
                ..config
            },
            _lock_file: lock_file,
            entries: BTreeMap::new(),
            stats: PersistentCacheStats {
                corruptions_recovered: recovered,
                evictions: startup_evictions,
                ..Default::default()
            },
            next_epoch: 1,
        };
        for (_, key, path, file_bytes) in candidates {
            let epoch = cache.take_epoch();
            cache.entries.insert(
                key,
                CacheEntry {
                    path,
                    file_bytes,
                    last_used: epoch,
                },
            );
            cache.stats.bytes = cache
                .stats
                .bytes
                .checked_add(file_bytes)
                .ok_or(PersistentCacheError::ByteCountOverflow)?;
        }
        cache.update_entry_count()?;
        cache.evict_to_limits(None)?;
        Ok(cache)
    }

    pub fn config(&self) -> &NativePersistentCacheConfig {
        &self.config
    }

    pub fn stats(&self) -> PersistentCacheStats {
        self.stats
    }

    pub fn contains(&self, identity: &PersistentCachePageIdentity) -> bool {
        identity
            .key()
            .ok()
            .is_some_and(|key| self.entries.contains_key(&key))
    }

    pub fn lookup(
        &mut self,
        identity: &PersistentCachePageIdentity,
    ) -> Result<PersistentCacheLookup, PersistentCacheError> {
        let key = identity.key()?;
        let Some(entry) = self.entries.get(&key).cloned() else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            return Ok(PersistentCacheLookup::Miss);
        };
        match read_record(&entry.path, key) {
            Ok(payload) => {
                let epoch = self.take_epoch();
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.last_used = epoch;
                    touch_file(&entry.path)?;
                }
                self.stats.hits = self.stats.hits.saturating_add(1);
                Ok(PersistentCacheLookup::Hit(payload))
            }
            Err(ReadRecordError::Corrupt(reason)) => {
                self.remove_entry(key)?;
                self.stats.misses = self.stats.misses.saturating_add(1);
                self.stats.corruptions_recovered =
                    self.stats.corruptions_recovered.saturating_add(1);
                Ok(PersistentCacheLookup::CorruptionRecovered(
                    PersistentCacheCorruption { key, reason },
                ))
            }
            Err(ReadRecordError::Io(error)) => Err(PersistentCacheError::Io(error)),
        }
    }

    fn lookup_validated(
        &mut self,
        validation: &PersistentCachePageValidation,
    ) -> Result<PersistentCacheLookup, PersistentCacheError> {
        let lookup = self.lookup(&validation.identity)?;
        let PersistentCacheLookup::Hit(payload) = lookup else {
            return Ok(lookup);
        };
        validation.validate(&payload)?;
        Ok(PersistentCacheLookup::Hit(payload))
    }

    pub fn insert(
        &mut self,
        identity: &PersistentCachePageIdentity,
        payload: &PagePayload,
    ) -> Result<PersistentCacheInsert, PersistentCacheError> {
        let key = identity.key()?;
        validate_payload_identity(identity, payload)?;
        if self.entries.contains_key(&key) {
            match self.lookup(identity)? {
                PersistentCacheLookup::Hit(existing) if existing == *payload => {
                    return Ok(PersistentCacheInsert::AlreadyPresent);
                }
                PersistentCacheLookup::Hit(_) => {
                    self.remove_entry(key)?;
                }
                PersistentCacheLookup::Miss | PersistentCacheLookup::CorruptionRecovered(_) => {}
            }
        }
        let file_bytes = (CACHE_HEADER_LEN as u64)
            .checked_add(identity.encoded_len)
            .ok_or(PersistentCacheError::ByteCountOverflow)?;
        if file_bytes > self.config.max_bytes {
            return Err(PersistentCacheError::PageExceedsBudget {
                page: identity.page_id,
                record_bytes: file_bytes,
                max_bytes: self.config.max_bytes,
            });
        }
        let evicted = self.evict_to_limits(Some(file_bytes))?;
        let final_path = self.config.root.join(key.file_name());
        let header = CacheRecordHeader {
            key,
            payload_checksum: payload.checksum,
            payload_len: identity.encoded_len,
        };
        atomic_write_record(&self.config.root, &final_path, &header, &payload.bytes)?;
        let epoch = self.take_epoch();
        self.entries.insert(
            key,
            CacheEntry {
                path: final_path,
                file_bytes,
                last_used: epoch,
            },
        );
        self.stats.bytes = self
            .stats
            .bytes
            .checked_add(file_bytes)
            .ok_or(PersistentCacheError::ByteCountOverflow)?;
        self.stats.writes = self.stats.writes.saturating_add(1);
        self.update_entry_count()?;
        Ok(PersistentCacheInsert::Written { evicted })
    }

    pub fn invalidate(
        &mut self,
        identity: &PersistentCachePageIdentity,
    ) -> Result<bool, PersistentCacheError> {
        let key = identity.key()?;
        if !self.entries.contains_key(&key) {
            return Ok(false);
        }
        self.remove_entry(key)?;
        Ok(true)
    }

    pub fn clear(&mut self) -> Result<(), PersistentCacheError> {
        let keys = self.entries.keys().copied().collect::<Vec<_>>();
        for key in keys {
            self.remove_entry(key)?;
        }
        Ok(())
    }

    fn take_epoch(&mut self) -> u64 {
        let epoch = self.next_epoch;
        self.next_epoch = self.next_epoch.wrapping_add(1).max(1);
        epoch
    }

    fn update_entry_count(&mut self) -> Result<(), PersistentCacheError> {
        self.stats.entries = self
            .entries
            .len()
            .try_into()
            .map_err(|_| PersistentCacheError::EntryCountOverflow)?;
        Ok(())
    }

    fn evict_to_limits(
        &mut self,
        incoming_file_bytes: Option<u64>,
    ) -> Result<Vec<PersistentCacheKey>, PersistentCacheError> {
        let incoming = incoming_file_bytes.unwrap_or(0);
        let target_entries = self.entries.len() as u64 + u64::from(incoming_file_bytes.is_some());
        let target_bytes = self
            .stats
            .bytes
            .checked_add(incoming)
            .ok_or(PersistentCacheError::ByteCountOverflow)?;
        if target_entries <= u64::from(self.config.max_entries)
            && target_bytes <= self.config.max_bytes
        {
            return Ok(Vec::new());
        }
        let mut candidates = self
            .entries
            .iter()
            .map(|(&key, entry)| (entry.last_used, key, entry.file_bytes))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(last_used, key, _)| (*last_used, *key));
        let mut remaining_entries = target_entries;
        let mut remaining_bytes = target_bytes;
        let mut evicted = Vec::new();
        for (_, key, bytes) in candidates {
            if remaining_entries <= u64::from(self.config.max_entries)
                && remaining_bytes <= self.config.max_bytes
            {
                break;
            }
            self.remove_entry(key)?;
            remaining_entries -= 1;
            remaining_bytes -= bytes;
            evicted.push(key);
        }
        if remaining_entries > u64::from(self.config.max_entries)
            || remaining_bytes > self.config.max_bytes
        {
            return Err(PersistentCacheError::BudgetCannotBeSatisfied);
        }
        Ok(evicted)
    }

    fn remove_entry(&mut self, key: PersistentCacheKey) -> Result<(), PersistentCacheError> {
        let Some(entry) = self.entries.remove(&key) else {
            return Ok(());
        };
        match fs::remove_file(&entry.path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                self.entries.insert(key, entry);
                return Err(PersistentCacheError::io(error));
            }
        }
        self.stats.bytes = self.stats.bytes.saturating_sub(entry.file_bytes);
        self.stats.evictions = self.stats.evictions.saturating_add(1);
        self.update_entry_count()?;
        sync_directory(&self.config.root)?;
        Ok(())
    }
}
fn read_header(path: &Path) -> Result<CacheRecordHeader, ReadRecordError> {
    let mut file = File::open(path).map_err(|error| ReadRecordError::Io(error.to_string()))?;
    let mut bytes = [0_u8; CACHE_HEADER_LEN];
    file.read_exact(&mut bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            ReadRecordError::Corrupt(PersistentCacheCorruptionReason::TruncatedHeader)
        } else {
            ReadRecordError::Io(error.to_string())
        }
    })?;
    CacheRecordHeader::decode(&bytes).map_err(ReadRecordError::Corrupt)
}

fn read_record(
    path: &Path,
    expected_key: PersistentCacheKey,
) -> Result<PagePayload, ReadRecordError> {
    let metadata = fs::metadata(path).map_err(|error| ReadRecordError::Io(error.to_string()))?;
    let header = read_header(path)?;
    if header.key != expected_key {
        return Err(ReadRecordError::Corrupt(
            PersistentCacheCorruptionReason::HeaderKeyMismatch,
        ));
    }
    let expected_file_len = (CACHE_HEADER_LEN as u64)
        .checked_add(header.payload_len)
        .ok_or({
            ReadRecordError::Corrupt(PersistentCacheCorruptionReason::FileLengthMismatch {
                expected: u64::MAX,
                actual: metadata.len(),
            })
        })?;
    if metadata.len() != expected_file_len {
        return Err(ReadRecordError::Corrupt(
            PersistentCacheCorruptionReason::FileLengthMismatch {
                expected: expected_file_len,
                actual: metadata.len(),
            },
        ));
    }
    let capacity = usize::try_from(header.payload_len).map_err(|_| {
        ReadRecordError::Corrupt(PersistentCacheCorruptionReason::FileLengthMismatch {
            expected: header.payload_len,
            actual: metadata.len(),
        })
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|_| {
        ReadRecordError::Io(format!(
            "failed to allocate {} cache payload bytes",
            header.payload_len
        ))
    })?;
    bytes.resize(capacity, 0);
    let mut file = File::open(path).map_err(|error| ReadRecordError::Io(error.to_string()))?;
    file.read_exact(&mut [0_u8; CACHE_HEADER_LEN])
        .map_err(|error| ReadRecordError::Io(error.to_string()))?;
    file.read_exact(&mut bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            ReadRecordError::Corrupt(PersistentCacheCorruptionReason::FileLengthMismatch {
                expected: expected_file_len,
                actual: metadata.len(),
            })
        } else {
            ReadRecordError::Io(error.to_string())
        }
    })?;
    let mut probe = [0_u8; 1];
    if file
        .read(&mut probe)
        .map_err(|error| ReadRecordError::Io(error.to_string()))?
        != 0
    {
        return Err(ReadRecordError::Corrupt(
            PersistentCacheCorruptionReason::FileLengthMismatch {
                expected: expected_file_len,
                actual: expected_file_len.saturating_add(1),
            },
        ));
    }
    let actual = page_checksum64(&bytes);
    if actual != header.payload_checksum {
        return Err(ReadRecordError::Corrupt(
            PersistentCacheCorruptionReason::PayloadChecksumMismatch {
                expected: header.payload_checksum,
                actual,
            },
        ));
    }
    Ok(PagePayload {
        page_id: expected_key.page_id,
        bytes,
        checksum: header.payload_checksum,
    })
}

#[derive(Debug)]
enum ReadRecordError {
    Corrupt(PersistentCacheCorruptionReason),
    Io(String),
}

fn atomic_write_record(
    root: &Path,
    final_path: &Path,
    header: &CacheRecordHeader,
    payload: &[u8],
) -> Result<(), PersistentCacheError> {
    let temporary = root.join(format!(
        "{TEMP_PREFIX}{}-{}",
        std::process::id(),
        NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(PersistentCacheError::io)?;
        file.write_all(&header.encode())
            .map_err(PersistentCacheError::io)?;
        file.write_all(payload).map_err(PersistentCacheError::io)?;
        file.sync_all().map_err(PersistentCacheError::io)?;
        drop(file);
        fs::rename(&temporary, final_path).map_err(PersistentCacheError::io)?;
        sync_directory(root)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(unix)]
fn sync_directory(root: &Path) -> Result<(), PersistentCacheError> {
    File::open(root)
        .and_then(|directory| directory.sync_all())
        .map_err(PersistentCacheError::io)
}

#[cfg(not(unix))]
fn sync_directory(_root: &Path) -> Result<(), PersistentCacheError> {
    Ok(())
}

fn touch_file(path: &Path) -> Result<(), PersistentCacheError> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(PersistentCacheError::io)?;
    file.set_times(std::fs::FileTimes::new().set_modified(SystemTime::now()))
        .map_err(PersistentCacheError::io)
}

/// Cache-first transport wrapper. A cache hit is usable without touching the
/// upstream transport, while misses and recovered corruption populate the cache
/// after a successful bounded fetch.
pub struct PersistentCachePageTransport<Upstream: LodPageTransport> {
    // Options preserve the public `into_parts` API while still allowing Drop
    // to cancel every live upstream ticket without unsafe field extraction.
    upstream: Option<Upstream>,
    cache: Option<NativePersistentPageCache>,
    identities: PersistentCachePageIdentities,
    tickets: BTreeMap<u64, PersistentCacheTicket<Upstream::Ticket>>,
    next_ticket: u64,
}

enum PersistentCacheTicket<Ticket> {
    Ready(PagePayload),
    Upstream {
        ticket: Ticket,
        validation: PersistentCachePageValidation,
    },
}

impl<Upstream: LodPageTransport> PersistentCachePageTransport<Upstream> {
    pub fn new(
        upstream: Upstream,
        cache: NativePersistentPageCache,
        identities: PersistentCachePageIdentities,
    ) -> Self {
        Self {
            upstream: Some(upstream),
            cache: Some(cache),
            identities,
            tickets: BTreeMap::new(),
            next_ticket: 1,
        }
    }

    pub fn cache(&self) -> &NativePersistentPageCache {
        self.cache
            .as_ref()
            .expect("persistent cache is present until into_parts")
    }

    pub fn cache_mut(&mut self) -> &mut NativePersistentPageCache {
        self.cache
            .as_mut()
            .expect("persistent cache is present until into_parts")
    }

    pub fn upstream(&self) -> &Upstream {
        self.upstream
            .as_ref()
            .expect("upstream is present until into_parts")
    }

    pub fn upstream_mut(&mut self) -> &mut Upstream {
        self.upstream
            .as_mut()
            .expect("upstream is present until into_parts")
    }

    /// Removes a page whose downstream codec/preprocess validation failed.
    pub fn invalidate_page(
        &mut self,
        page: LodPageId,
    ) -> Result<bool, PersistentCacheTransportError<Upstream::Error>> {
        let identity = self
            .identities
            .get(page)
            .cloned()
            .ok_or(PersistentCacheTransportError::MissingIdentity(page))?;
        self.cache_mut()
            .invalidate(&identity)
            .map_err(PersistentCacheTransportError::Cache)
    }

    pub fn into_parts(mut self) -> (Upstream, NativePersistentPageCache) {
        self.cancel_all();
        (
            self.upstream
                .take()
                .expect("upstream is present until into_parts"),
            self.cache
                .take()
                .expect("persistent cache is present until into_parts"),
        )
    }

    fn cancel_all(&mut self) {
        let tickets = self.tickets.keys().copied().collect::<Vec<_>>();
        for ticket in tickets {
            <Self as LodPageTransport>::cancel(self, &ticket);
        }
    }
}

impl<Upstream: LodPageTransport> Drop for PersistentCachePageTransport<Upstream> {
    fn drop(&mut self) {
        self.cancel_all();
    }
}

impl<Upstream: LodPageTransport> LodPageTransport for PersistentCachePageTransport<Upstream> {
    type Ticket = u64;
    type Error = PersistentCacheTransportError<Upstream::Error>;

    fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
        let validation = validated_transport_page(&self.identities, request)?;
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
        let state = match self
            .cache_mut()
            .lookup_validated(&validation)
            .map_err(PersistentCacheTransportError::Cache)?
        {
            PersistentCacheLookup::Hit(payload) => PersistentCacheTicket::Ready(payload),
            PersistentCacheLookup::Miss | PersistentCacheLookup::CorruptionRecovered(_) => {
                let ticket = self
                    .upstream_mut()
                    .begin(request)
                    .map_err(PersistentCacheTransportError::Upstream)?;
                PersistentCacheTicket::Upstream { ticket, validation }
            }
        };
        let ticket = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
        self.tickets.insert(ticket, state);
        Ok(ticket)
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
        let Some(state) = self.tickets.remove(ticket) else {
            return PagePoll::Failed(PersistentCacheTransportError::InvalidTicket(*ticket));
        };
        match state {
            PersistentCacheTicket::Ready(payload) => PagePoll::Ready(payload),
            PersistentCacheTicket::Upstream {
                ticket: upstream_ticket,
                validation,
            } => match self.upstream_mut().poll(&upstream_ticket) {
                PagePoll::Pending => {
                    self.tickets.insert(
                        *ticket,
                        PersistentCacheTicket::Upstream {
                            ticket: upstream_ticket,
                            validation,
                        },
                    );
                    PagePoll::Pending
                }
                PagePoll::Ready(payload) => {
                    if let Err(error) = validation.validate(&payload) {
                        return PagePoll::Failed(PersistentCacheTransportError::Cache(error));
                    }
                    match self.cache_mut().insert(&validation.identity, &payload) {
                        Ok(_) => PagePoll::Ready(payload),
                        Err(error) => PagePoll::Failed(PersistentCacheTransportError::Cache(error)),
                    }
                }
                PagePoll::Failed(error) => {
                    PagePoll::Failed(PersistentCacheTransportError::Upstream(error))
                }
            },
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        if let Some(PersistentCacheTicket::Upstream { ticket, .. }) = self.tickets.remove(ticket)
            && let Some(upstream) = self.upstream.as_mut()
        {
            upstream.cancel(&ticket);
        }
    }
}

/// Serialized native cache service. One bounded worker owns the filesystem
/// index and is therefore the only writer for its canonical cache root. Cloned
/// service handles only enqueue bounded commands; cache reads, checksum scans,
/// eviction, writes, and fsync never execute on the caller's frame thread.
#[derive(Clone)]
pub struct NativePersistentCacheService {
    inner: Arc<NativePersistentCacheServiceInner>,
}

struct NativePersistentCacheServiceInner {
    sender: std::sync::mpsc::SyncSender<NativeCacheServiceCommand>,
}

struct RegisteredNativePersistentCacheService {
    config: NativePersistentCacheConfig,
    max_pending_operations: u32,
    service: NativePersistentCacheService,
}

fn native_persistent_cache_services()
-> &'static Mutex<BTreeMap<PathBuf, RegisteredNativePersistentCacheService>> {
    static SERVICES: std::sync::OnceLock<
        Mutex<BTreeMap<PathBuf, RegisteredNativePersistentCacheService>>,
    > = std::sync::OnceLock::new();
    SERVICES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

enum NativeCacheServiceCommand {
    Lookup {
        validation: PersistentCachePageValidation,
        reply: std::sync::mpsc::SyncSender<Result<PersistentCacheLookup, PersistentCacheError>>,
    },
    Insert {
        identity: PersistentCachePageIdentity,
        payload: PagePayload,
        reply: std::sync::mpsc::SyncSender<Result<PersistentCacheInsert, PersistentCacheError>>,
    },
    Invalidate {
        identity: PersistentCachePageIdentity,
        reply: std::sync::mpsc::SyncSender<Result<bool, PersistentCacheError>>,
    },
    #[cfg(test)]
    BlockUntil(std::sync::mpsc::Receiver<()>),
}

impl NativePersistentCacheService {
    pub fn spawn(
        cache: NativePersistentPageCache,
        max_pending_operations: u32,
    ) -> Result<Self, PersistentCacheError> {
        let config = cache.config().clone();
        acquire_native_persistent_cache_service(config, max_pending_operations, move |_| {
            Self::spawn_with_cache_unregistered(cache, max_pending_operations)
        })
    }

    fn spawn_with_cache_unregistered(
        cache: NativePersistentPageCache,
        max_pending_operations: u32,
    ) -> Result<Self, PersistentCacheError> {
        let capacity = validate_service_queue_capacity(max_pending_operations)?;
        let (sender, receiver) = std::sync::mpsc::sync_channel(capacity);
        std::thread::Builder::new()
            .name("gaussian-lod-cache".to_owned())
            .spawn(move || run_native_cache_service(cache, receiver))
            .map_err(|error| PersistentCacheError::CacheWorkerSpawn(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(NativePersistentCacheServiceInner { sender }),
        })
    }

    /// Starts cache directory creation, recovery scanning, and eviction on the
    /// service worker. Commands may be enqueued immediately and are answered
    /// only after open succeeds or with a typed initialization failure. A later
    /// command retries initialization, so a temporary external lock or storage
    /// outage cannot permanently poison the process-lifetime service.
    pub fn spawn_from_config(
        config: NativePersistentCacheConfig,
        max_pending_operations: u32,
    ) -> Result<Self, PersistentCacheError> {
        acquire_native_persistent_cache_service(config, max_pending_operations, |config| {
            Self::spawn_from_config_unregistered(config, max_pending_operations)
        })
    }

    fn spawn_from_config_unregistered(
        config: NativePersistentCacheConfig,
        max_pending_operations: u32,
    ) -> Result<Self, PersistentCacheError> {
        let capacity = validate_service_queue_capacity(max_pending_operations)?;
        let (sender, receiver) = std::sync::mpsc::sync_channel(capacity);
        std::thread::Builder::new()
            .name("gaussian-lod-cache".to_owned())
            .spawn(move || run_native_cache_service_open(config, receiver))
            .map_err(|error| PersistentCacheError::CacheWorkerSpawn(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(NativePersistentCacheServiceInner { sender }),
        })
    }

    fn begin_lookup(
        &self,
        validation: PersistentCachePageValidation,
    ) -> Result<NativeCacheLookupReceiver, PersistentCacheError> {
        let (reply, receiver) = std::sync::mpsc::sync_channel(1);
        self.inner
            .sender
            .try_send(NativeCacheServiceCommand::Lookup { validation, reply })
            .map_err(map_cache_service_send_error)?;
        Ok(receiver)
    }

    fn begin_insert(
        &self,
        identity: PersistentCachePageIdentity,
        payload: PagePayload,
    ) -> Result<NativeCacheInsertReceiver, PersistentCacheError> {
        let (reply, receiver) = std::sync::mpsc::sync_channel(1);
        self.inner
            .sender
            .try_send(NativeCacheServiceCommand::Insert {
                identity,
                payload,
                reply,
            })
            .map_err(map_cache_service_send_error)?;
        Ok(receiver)
    }

    fn begin_invalidate(
        &self,
        identity: PersistentCachePageIdentity,
    ) -> Result<NativeCacheInvalidateReceiver, PersistentCacheError> {
        let (reply, receiver) = std::sync::mpsc::sync_channel(1);
        self.inner
            .sender
            .try_send(NativeCacheServiceCommand::Invalidate { identity, reply })
            .map_err(map_cache_service_send_error)?;
        Ok(receiver)
    }

    #[cfg(test)]
    fn block_until(
        &self,
        release: std::sync::mpsc::Receiver<()>,
    ) -> Result<(), PersistentCacheError> {
        self.inner
            .sender
            .try_send(NativeCacheServiceCommand::BlockUntil(release))
            .map_err(map_cache_service_send_error)
    }
}

fn acquire_native_persistent_cache_service(
    mut config: NativePersistentCacheConfig,
    max_pending_operations: u32,
    start: impl FnOnce(
        NativePersistentCacheConfig,
    ) -> Result<NativePersistentCacheService, PersistentCacheError>,
) -> Result<NativePersistentCacheService, PersistentCacheError> {
    config.validate()?;
    validate_service_queue_capacity(max_pending_operations)?;
    // Resolve the actual directory identity before consulting the process-wide
    // registry. Lexical paths are insufficient here: a symlink or `..` alias
    // must not create a second writer for the same cache namespace.
    fs::create_dir_all(&config.root).map_err(PersistentCacheError::io)?;
    config.root = fs::canonicalize(&config.root).map_err(PersistentCacheError::io)?;
    if !config.root.is_dir() {
        return Err(PersistentCacheError::RootIsNotDirectory(config.root));
    }
    let key = config.root.clone();
    let mut services = native_persistent_cache_services()
        .lock()
        .map_err(|_| PersistentCacheError::CacheServiceRegistryPoisoned)?;
    if let Some(registered) = services.get(&key) {
        if registered.config != config
            || registered.max_pending_operations != max_pending_operations
        {
            return Err(PersistentCacheError::CacheServiceConfigConflict(
                key.display().to_string(),
            ));
        }
        return Ok(registered.service.clone());
    }
    if services.len() >= MAX_PERSISTENT_CACHE_SERVICES {
        return Err(PersistentCacheError::CacheServiceRegistryFull {
            maximum: MAX_PERSISTENT_CACHE_SERVICES,
        });
    }
    let service = start(config.clone())?;
    services.insert(
        key,
        RegisteredNativePersistentCacheService {
            config,
            max_pending_operations,
            service: service.clone(),
        },
    );
    Ok(service)
}

type NativeCacheLookupReceiver =
    std::sync::mpsc::Receiver<Result<PersistentCacheLookup, PersistentCacheError>>;
type NativeCacheInsertReceiver =
    std::sync::mpsc::Receiver<Result<PersistentCacheInsert, PersistentCacheError>>;
type NativeCacheInvalidateReceiver = std::sync::mpsc::Receiver<Result<bool, PersistentCacheError>>;

fn run_native_cache_service(
    mut cache: NativePersistentPageCache,
    receiver: std::sync::mpsc::Receiver<NativeCacheServiceCommand>,
) {
    while let Ok(command) = receiver.recv() {
        run_native_cache_service_command(&mut cache, command);
    }
}

fn run_native_cache_service_open(
    config: NativePersistentCacheConfig,
    receiver: std::sync::mpsc::Receiver<NativeCacheServiceCommand>,
) {
    match NativePersistentPageCache::open(config.clone()) {
        Ok(cache) => run_native_cache_service(cache, receiver),
        Err(_) => {
            // Initialization failures are deliberately non-sticky. The
            // process registry retains this single worker for the canonical
            // root, and every subsequent command gives temporarily
            // unavailable storage (notably another process's file lock) an
            // opportunity to recover. All retries and scans stay on this
            // worker rather than the caller's frame thread.
            while let Ok(command) = receiver.recv() {
                match NativePersistentPageCache::open(config.clone()) {
                    Ok(mut cache) => {
                        run_native_cache_service_command(&mut cache, command);
                        run_native_cache_service(cache, receiver);
                        return;
                    }
                    Err(error) => reply_native_cache_initialization_failure(command, error),
                }
            }
        }
    }
}

fn run_native_cache_service_command(
    cache: &mut NativePersistentPageCache,
    command: NativeCacheServiceCommand,
) {
    match command {
        NativeCacheServiceCommand::Lookup { validation, reply } => {
            let _ = reply.send(cache.lookup_validated(&validation));
        }
        NativeCacheServiceCommand::Insert {
            identity,
            payload,
            reply,
        } => {
            let _ = reply.send(cache.insert(&identity, &payload));
        }
        NativeCacheServiceCommand::Invalidate { identity, reply } => {
            let _ = reply.send(cache.invalidate(&identity));
        }
        #[cfg(test)]
        NativeCacheServiceCommand::BlockUntil(release) => {
            let _ = release.recv();
        }
    }
}

fn reply_native_cache_initialization_failure(
    command: NativeCacheServiceCommand,
    error: PersistentCacheError,
) {
    match command {
        NativeCacheServiceCommand::Lookup { reply, .. } => {
            let _ = reply.send(Err(error));
        }
        NativeCacheServiceCommand::Insert { reply, .. } => {
            let _ = reply.send(Err(error));
        }
        NativeCacheServiceCommand::Invalidate { reply, .. } => {
            let _ = reply.send(Err(error));
        }
        #[cfg(test)]
        NativeCacheServiceCommand::BlockUntil(release) => {
            let _ = release.recv();
        }
    }
}

fn map_cache_service_send_error<T>(
    error: std::sync::mpsc::TrySendError<T>,
) -> PersistentCacheError {
    match error {
        std::sync::mpsc::TrySendError::Full(_) => PersistentCacheError::CacheServiceQueueFull,
        std::sync::mpsc::TrySendError::Disconnected(_) => {
            PersistentCacheError::CacheServiceDisconnected
        }
    }
}

/// Shared, cache-first native transport. The `Arc<Mutex<_>>` coordinates any
/// number of package runtimes while the enclosed service serializes all actual
/// filesystem work on one bounded worker. Lock hold times cover enqueue only.
pub struct SharedPersistentCachePageTransport<Upstream: LodPageTransport> {
    upstream: Upstream,
    cache: Arc<Mutex<NativePersistentCacheService>>,
    identities: PersistentCachePageIdentities,
    tickets: BTreeMap<u64, SharedPersistentCacheTicket<Upstream::Ticket>>,
    invalidations: BTreeMap<LodPageId, NativeCacheInvalidation>,
    next_ticket: u64,
}

impl<Upstream: LodPageTransport> Drop for SharedPersistentCachePageTransport<Upstream> {
    fn drop(&mut self) {
        let tickets = self.tickets.keys().copied().collect::<Vec<_>>();
        for ticket in tickets {
            <Self as LodPageTransport>::cancel(self, &ticket);
        }
    }
}

enum SharedPersistentCacheTicket<Ticket> {
    Lookup {
        receiver: NativeCacheLookupReceiver,
        request: PageRequest,
        validation: PersistentCachePageValidation,
    },
    Upstream {
        ticket: Ticket,
        validation: PersistentCachePageValidation,
        bypass_store: bool,
    },
    Store {
        receiver: NativeCacheInsertReceiver,
        payload: PagePayload,
    },
}

enum NativeCacheInvalidation {
    Queued(PersistentCachePageIdentity),
    InFlight {
        identity: PersistentCachePageIdentity,
        receiver: NativeCacheInvalidateReceiver,
    },
}

impl<Upstream: LodPageTransport> SharedPersistentCachePageTransport<Upstream> {
    pub fn new(
        upstream: Upstream,
        cache: Arc<Mutex<NativePersistentCacheService>>,
        identities: PersistentCachePageIdentities,
    ) -> Self {
        Self {
            upstream,
            cache,
            identities,
            tickets: BTreeMap::new(),
            invalidations: BTreeMap::new(),
            next_ticket: 1,
        }
    }

    pub fn shared_cache(&self) -> &Arc<Mutex<NativePersistentCacheService>> {
        &self.cache
    }

    /// Queues removal of a page whose downstream codec/preprocess validation
    /// failed. Commands share the cache worker's bounded FIFO, so an earlier
    /// insert always completes before its invalidation.
    pub fn invalidate_page(
        &mut self,
        page: LodPageId,
    ) -> Result<(), PersistentCacheTransportError<Upstream::Error>> {
        let _ = self.maintain_cache()?;
        if self.invalidations.contains_key(&page) {
            return Ok(());
        }
        let identity = self
            .identities
            .get(page)
            .cloned()
            .ok_or(PersistentCacheTransportError::MissingIdentity(page))?;
        self.invalidations
            .insert(page, NativeCacheInvalidation::Queued(identity));
        self.maintain_cache().map(|_| ())
    }

    /// Polls queued invalidations without blocking the application thread.
    /// Returns true once no invalidation remains outstanding.
    pub fn maintain_cache(
        &mut self,
    ) -> Result<bool, PersistentCacheTransportError<Upstream::Error>> {
        let pages = self.invalidations.keys().copied().collect::<Vec<_>>();
        for page in pages {
            let Some(state) = self.invalidations.remove(&page) else {
                continue;
            };
            match state {
                NativeCacheInvalidation::Queued(identity) => {
                    self.invalidations
                        .insert(page, NativeCacheInvalidation::Queued(identity));
                }
                NativeCacheInvalidation::InFlight { identity, receiver } => {
                    match receiver.try_recv() {
                        Err(std::sync::mpsc::TryRecvError::Empty) => {
                            self.invalidations.insert(
                                page,
                                NativeCacheInvalidation::InFlight { identity, receiver },
                            );
                        }
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            self.invalidations
                                .insert(page, NativeCacheInvalidation::Queued(identity));
                            return Err(PersistentCacheTransportError::Cache(
                                PersistentCacheError::CacheServiceDisconnected,
                            ));
                        }
                        Ok(Err(error)) => {
                            self.invalidations
                                .insert(page, NativeCacheInvalidation::Queued(identity));
                            return Err(PersistentCacheTransportError::Cache(error));
                        }
                        Ok(Ok(_)) => {}
                    }
                }
            }
        }

        let queued = self
            .invalidations
            .iter()
            .filter_map(|(&page, state)| match state {
                NativeCacheInvalidation::Queued(identity) => Some((page, identity.clone())),
                NativeCacheInvalidation::InFlight { .. } => None,
            })
            .collect::<Vec<_>>();
        for (page, identity) in queued {
            let receiver = match self.cache.lock() {
                Ok(cache) => match cache.begin_invalidate(identity.clone()) {
                    Ok(receiver) => receiver,
                    Err(PersistentCacheError::CacheServiceQueueFull) => continue,
                    Err(error) => return Err(PersistentCacheTransportError::Cache(error)),
                },
                Err(_) => return Err(PersistentCacheTransportError::SharedCacheUnavailable),
            };
            self.invalidations.insert(
                page,
                NativeCacheInvalidation::InFlight { identity, receiver },
            );
        }
        Ok(self.invalidations.is_empty())
    }
}

impl<Upstream: LodPageTransport> LodPageTransport for SharedPersistentCachePageTransport<Upstream> {
    type Ticket = u64;
    type Error = PersistentCacheTransportError<Upstream::Error>;

    fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
        let validation = validated_transport_page(&self.identities, request)?;
        let state = if self.invalidations.contains_key(&request.page_id) {
            let upstream_ticket = self
                .upstream
                .begin(request)
                .map_err(PersistentCacheTransportError::Upstream)?;
            SharedPersistentCacheTicket::Upstream {
                ticket: upstream_ticket,
                validation,
                bypass_store: true,
            }
        } else {
            let receiver = self
                .cache
                .lock()
                .map_err(|_| PersistentCacheTransportError::SharedCacheUnavailable)?
                .begin_lookup(validation.clone())
                .map_err(PersistentCacheTransportError::Cache)?;
            SharedPersistentCacheTicket::Lookup {
                receiver,
                request,
                validation,
            }
        };
        let ticket = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
        self.tickets.insert(ticket, state);
        Ok(ticket)
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
        let Some(state) = self.tickets.remove(ticket) else {
            return PagePoll::Failed(PersistentCacheTransportError::InvalidTicket(*ticket));
        };
        match state {
            SharedPersistentCacheTicket::Lookup {
                receiver,
                request,
                validation,
            } => match receiver.try_recv() {
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    self.tickets.insert(
                        *ticket,
                        SharedPersistentCacheTicket::Lookup {
                            receiver,
                            request,
                            validation,
                        },
                    );
                    PagePoll::Pending
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    PagePoll::Failed(PersistentCacheTransportError::Cache(
                        PersistentCacheError::CacheServiceDisconnected,
                    ))
                }
                Ok(Err(error)) => PagePoll::Failed(PersistentCacheTransportError::Cache(error)),
                Ok(Ok(PersistentCacheLookup::Hit(payload))) => PagePoll::Ready(payload),
                Ok(Ok(
                    PersistentCacheLookup::Miss | PersistentCacheLookup::CorruptionRecovered(_),
                )) => match self.upstream.begin(request) {
                    Ok(upstream_ticket) => {
                        self.tickets.insert(
                            *ticket,
                            SharedPersistentCacheTicket::Upstream {
                                ticket: upstream_ticket,
                                validation,
                                bypass_store: false,
                            },
                        );
                        PagePoll::Pending
                    }
                    Err(error) => PagePoll::Failed(PersistentCacheTransportError::Upstream(error)),
                },
            },
            SharedPersistentCacheTicket::Upstream {
                ticket: upstream_ticket,
                validation,
                bypass_store,
            } => match self.upstream.poll(&upstream_ticket) {
                PagePoll::Pending => {
                    self.tickets.insert(
                        *ticket,
                        SharedPersistentCacheTicket::Upstream {
                            ticket: upstream_ticket,
                            validation,
                            bypass_store,
                        },
                    );
                    PagePoll::Pending
                }
                PagePoll::Failed(error) => {
                    PagePoll::Failed(PersistentCacheTransportError::Upstream(error))
                }
                PagePoll::Ready(payload) => {
                    if let Err(error) = validation.validate(&payload) {
                        return PagePoll::Failed(PersistentCacheTransportError::Cache(error));
                    }
                    if bypass_store {
                        return PagePoll::Ready(payload);
                    }
                    let receiver = match self
                        .cache
                        .lock()
                        .map_err(|_| PersistentCacheTransportError::SharedCacheUnavailable)
                        .and_then(|cache| {
                            cache
                                .begin_insert(validation.identity, payload.clone())
                                .map_err(PersistentCacheTransportError::Cache)
                        }) {
                        Ok(receiver) => receiver,
                        Err(error) => return PagePoll::Failed(error),
                    };
                    self.tickets.insert(
                        *ticket,
                        SharedPersistentCacheTicket::Store { receiver, payload },
                    );
                    PagePoll::Pending
                }
            },
            SharedPersistentCacheTicket::Store { receiver, payload } => match receiver.try_recv() {
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    self.tickets.insert(
                        *ticket,
                        SharedPersistentCacheTicket::Store { receiver, payload },
                    );
                    PagePoll::Pending
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    PagePoll::Failed(PersistentCacheTransportError::Cache(
                        PersistentCacheError::CacheServiceDisconnected,
                    ))
                }
                Ok(Err(error)) => PagePoll::Failed(PersistentCacheTransportError::Cache(error)),
                Ok(Ok(_)) => PagePoll::Ready(payload),
            },
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        if let Some(SharedPersistentCacheTicket::Upstream { ticket, .. }) =
            self.tickets.remove(ticket)
        {
            self.upstream.cancel(&ticket);
        }
    }
}

#[cfg(test)]
mod tests;
