//! Async-runtime-neutral page request and transport contracts.
//!
//! Browser fetch, native HTTP, memory maps, and tests can all implement the
//! begin/poll interface without imposing `Send`, an executor, or a futures crate
//! on the renderer.

use std::{
    borrow::Borrow,
    collections::{BTreeMap, HashMap},
    fmt,
    ops::Deref,
    sync::Arc,
};

use bevy::prelude::Reflect;
use bevy_args::{Deserialize, Serialize};

pub use crate::gaussian::formats::planar_3d_chunked::LodPageId;
use crate::gaussian::formats::planar_3d_lod::GaussianLodManifest;

/// Allocation-free clone of one immutable manifest object name.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ManifestPageUri(Arc<str>);

impl ManifestPageUri {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[cfg(test)]
    fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Deref for ManifestPageUri {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl AsRef<str> for ManifestPageUri {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for ManifestPageUri {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for ManifestPageUri {
    fn from(uri: &str) -> Self {
        Self(Arc::from(uri))
    }
}

impl From<String> for ManifestPageUri {
    fn from(uri: String) -> Self {
        Self(Arc::from(uri))
    }
}

impl fmt::Display for ManifestPageUri {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl PartialEq<String> for ManifestPageUri {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<ManifestPageUri> for String {
    fn eq(&self, other: &ManifestPageUri) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Serialize for ManifestPageUri {
    fn serialize<Serializer>(
        &self,
        serializer: Serializer,
    ) -> Result<Serializer::Ok, Serializer::Error>
    where
        Serializer: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ManifestPageUri {
    fn deserialize<Deserializer>(deserializer: Deserializer) -> Result<Self, Deserializer::Error>
    where
        Deserializer: serde::Deserializer<'de>,
    {
        String::deserialize(deserializer).map(Self::from)
    }
}

/// Transport location copied from a validated manifest page descriptor.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestPageLocation {
    /// Shared immutable object name. Packed packages commonly reference one
    /// shard from many pages, so cloning a location remains allocation-free.
    pub uri: ManifestPageUri,
    pub byte_range: Option<(u64, u64)>,
    pub encoded_len: u64,
}

/// Compact page-ID lookup shared by transport and persistent-cache indexes.
/// Canonical packages require no index allocation; arbitrary portable IDs use
/// one expected-linear-time fallback that remains cheap to clone.
#[derive(Clone, Debug)]
pub(crate) enum CompiledPageIndex {
    DenseOneBased,
    Sparse(Arc<HashMap<LodPageId, usize>>),
}

impl CompiledPageIndex {
    pub(crate) fn compile(len: usize, mut page_id_at: impl FnMut(usize) -> LodPageId) -> Self {
        let dense_one_based = (0..len).all(|index| {
            let expected = u64::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(1));
            expected.is_some_and(|expected| page_id_at(index) == LodPageId(expected))
        });
        if dense_one_based {
            Self::DenseOneBased
        } else {
            Self::Sparse(Arc::new(
                (0..len).map(|index| (page_id_at(index), index)).collect(),
            ))
        }
    }

    pub(crate) fn get(&self, page_id: LodPageId, len: usize) -> Option<usize> {
        match self {
            Self::DenseOneBased => {
                let index = usize::try_from(page_id.0.checked_sub(1)?).ok()?;
                (index < len).then_some(index)
            }
            Self::Sparse(indices) => indices.get(&page_id).copied(),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_dense_one_based(&self) -> bool {
        matches!(self, Self::DenseOneBased)
    }
}

#[derive(Clone, Debug)]
struct ManifestPageLocationEntry {
    page_id: LodPageId,
    location: ManifestPageLocation,
}

/// Resolves immutable page IDs without coupling selection to a particular URL,
/// filesystem, CDN, signed-URL service, or pack-file implementation.
#[derive(Clone, Debug)]
pub struct ManifestPageLocations {
    entries: Arc<[ManifestPageLocationEntry]>,
    index: CompiledPageIndex,
}

impl Default for ManifestPageLocations {
    fn default() -> Self {
        Self {
            entries: Arc::from([]),
            index: CompiledPageIndex::DenseOneBased,
        }
    }
}

impl ManifestPageLocations {
    pub fn from_manifest(manifest: &GaussianLodManifest) -> Result<Self, PageLocationError> {
        manifest
            .validate()
            .map_err(|error| PageLocationError::InvalidManifest(error.to_string()))?;
        Self::from_validated_manifest(manifest)
    }

    /// Copies transport locations from an immutable manifest whose complete
    /// semantic contract has already been validated by an in-crate owner.
    pub(crate) fn from_validated_manifest(
        manifest: &GaussianLodManifest,
    ) -> Result<Self, PageLocationError> {
        let index =
            CompiledPageIndex::compile(manifest.pages.len(), |index| manifest.pages[index].id);
        let mut shared_uris = HashMap::<&str, ManifestPageUri>::new();
        let mut entries = Vec::with_capacity(manifest.pages.len());
        for descriptor in &manifest.pages {
            let storage = descriptor
                .storage
                .as_ref()
                .ok_or(PageLocationError::MissingStorage(descriptor.id))?;
            let uri = if let Some(uri) = shared_uris.get(storage.uri.as_str()) {
                uri.clone()
            } else {
                let uri = ManifestPageUri::from(storage.uri.as_str());
                shared_uris.insert(storage.uri.as_str(), uri.clone());
                uri
            };
            entries.push(ManifestPageLocationEntry {
                page_id: descriptor.id,
                location: ManifestPageLocation {
                    uri,
                    byte_range: storage.byte_range,
                    encoded_len: storage.encoded_len,
                },
            });
        }
        Ok(Self {
            entries: entries.into(),
            index,
        })
    }

    #[cfg(test)]
    fn from_entries(entries: impl IntoIterator<Item = (LodPageId, ManifestPageLocation)>) -> Self {
        let entries = entries
            .into_iter()
            .map(|(page_id, location)| ManifestPageLocationEntry { page_id, location })
            .collect::<Vec<_>>();
        let index = CompiledPageIndex::compile(entries.len(), |index| entries[index].page_id);
        Self {
            entries: entries.into(),
            index,
        }
    }

    pub fn get(&self, page_id: LodPageId) -> Option<&ManifestPageLocation> {
        let index = self.index.get(page_id, self.entries.len())?;
        let entry = self.entries.get(index)?;
        (entry.page_id == page_id).then_some(&entry.location)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Stable manifest page order for transport construction and cache indexes.
    pub fn page_ids(&self) -> impl Iterator<Item = LodPageId> + '_ {
        self.entries.iter().map(|entry| entry.page_id)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn iter(&self) -> impl Iterator<Item = (LodPageId, &ManifestPageLocation)> {
        self.entries
            .iter()
            .map(|entry| (entry.page_id, &entry.location))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PageLocationError {
    InvalidManifest(String),
    MissingStorage(LodPageId),
}

impl fmt::Display for PageLocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidManifest(error) => write!(formatter, "invalid LoD manifest: {error}"),
            Self::MissingStorage(page) => {
                write!(formatter, "LoD page {} has no transport location", page.0)
            }
        }
    }
}

impl std::error::Error for PageLocationError {}

/// Broad request class. Variants are ordered from least to most urgent.
#[derive(
    Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Reflect, Serialize, Deserialize,
)]
pub enum PageRequestClass {
    Prefetch,
    Visible,
    #[default]
    FallbackCritical,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Reflect, Serialize, Deserialize,
)]
pub struct PageRequestPriority {
    pub class: PageRequestClass,
    /// Quantized screen-space urgency. Higher values are serviced first.
    pub urgency: u32,
}

impl PageRequestPriority {
    pub const fn prefetch(urgency: u32) -> Self {
        Self {
            class: PageRequestClass::Prefetch,
            urgency,
        }
    }

    pub const fn visible(urgency: u32) -> Self {
        Self {
            class: PageRequestClass::Visible,
            urgency,
        }
    }

    pub const fn fallback_critical(urgency: u32) -> Self {
        Self {
            class: PageRequestClass::FallbackCritical,
            urgency,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PageRequest {
    pub page_id: LodPageId,
    pub priority: PageRequestPriority,
    /// Expected encoded transport size, when known from the manifest.
    pub expected_bytes: Option<u64>,
    /// The page to retain as a visible fallback while this request is pending.
    pub fallback_page: Option<LodPageId>,
}

impl PageRequest {
    pub const fn new(page_id: LodPageId, priority: PageRequestPriority) -> Self {
        Self {
            page_id,
            priority,
            expected_bytes: None,
            fallback_page: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestEnqueue {
    Enqueued,
    Promoted,
    Duplicate,
    Replaced(LodPageId),
    Rejected,
}

#[derive(Clone, Copy, Debug)]
struct QueuedRequest {
    request: PageRequest,
    sequence: u64,
}

/// Bounded deterministic priority queue with page-id deduplication.
#[derive(Clone, Debug)]
pub struct PageRequestQueue {
    capacity: usize,
    next_sequence: u64,
    entries: BTreeMap<LodPageId, QueuedRequest>,
}

impl PageRequestQueue {
    pub fn new(capacity: usize) -> Result<Self, RequestQueueError> {
        if capacity == 0 {
            return Err(RequestQueueError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            next_sequence: 0,
            entries: BTreeMap::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, page_id: LodPageId) -> bool {
        self.entries.contains_key(&page_id)
    }

    /// Stable page IDs currently waiting for transport admission.
    ///
    /// This read-only view lets demand reconciliation identify stale queued
    /// work while keeping priority, sequence, and capacity invariants private.
    pub fn page_ids(&self) -> impl Iterator<Item = LodPageId> + '_ {
        self.entries.keys().copied()
    }

    pub fn enqueue(&mut self, request: PageRequest) -> RequestEnqueue {
        if !request.page_id.is_valid()
            || request
                .fallback_page
                .is_some_and(|fallback| !fallback.is_valid())
        {
            return RequestEnqueue::Rejected;
        }
        if let Some(existing) = self.entries.get_mut(&request.page_id) {
            if request.priority > existing.request.priority {
                existing.request.priority = request.priority;
                existing.request.expected_bytes =
                    request.expected_bytes.or(existing.request.expected_bytes);
                existing.request.fallback_page =
                    request.fallback_page.or(existing.request.fallback_page);
                return RequestEnqueue::Promoted;
            }
            return RequestEnqueue::Duplicate;
        }

        let replaced = if self.entries.len() >= self.capacity {
            let (&victim_id, victim) = self
                .entries
                .iter()
                .min_by(|(left_id, left), (right_id, right)| {
                    left.request
                        .priority
                        .cmp(&right.request.priority)
                        // Newer low-priority work is discarded before older work.
                        .then_with(|| right.sequence.cmp(&left.sequence))
                        .then_with(|| right_id.cmp(left_id))
                })
                .expect("a full non-zero queue has a victim");
            if request.priority <= victim.request.priority {
                return RequestEnqueue::Rejected;
            }
            self.entries.remove(&victim_id);
            Some(victim_id)
        } else {
            None
        };

        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.entries
            .insert(request.page_id, QueuedRequest { request, sequence });
        replaced.map_or(RequestEnqueue::Enqueued, RequestEnqueue::Replaced)
    }

    pub fn pop(&mut self) -> Option<PageRequest> {
        let (&page_id, _) = self
            .entries
            .iter()
            .max_by(|(left_id, left), (right_id, right)| {
                left.request
                    .priority
                    .cmp(&right.request.priority)
                    // Older work wins priority ties.
                    .then_with(|| right.sequence.cmp(&left.sequence))
                    // Lower stable page IDs win complete ties.
                    .then_with(|| right_id.cmp(left_id))
            })?;
        self.entries.remove(&page_id).map(|entry| entry.request)
    }

    pub fn remove(&mut self, page_id: LodPageId) -> Option<PageRequest> {
        self.entries.remove(&page_id).map(|entry| entry.request)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestQueueError {
    ZeroCapacity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PagePayload {
    pub page_id: LodPageId,
    pub bytes: Vec<u8>,
    /// Stable checksum of the uncompressed bytes.
    pub checksum: u64,
}

impl PagePayload {
    pub fn new(page_id: LodPageId, bytes: Vec<u8>) -> Self {
        let checksum = page_checksum64(&bytes);
        Self {
            page_id,
            bytes,
            checksum,
        }
    }

    pub fn verify(&self) -> bool {
        page_checksum64(&self.bytes) == self.checksum
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PagePoll<Error> {
    Pending,
    Ready(PagePayload),
    Failed(Error),
}

/// Stable runtime-facing class for an owned transport failure.
///
/// Concrete transports retain their backend-specific error type, while the
/// streaming runtime stores this small normalized value across retries. This
/// prevents terminal package status from erasing cache failures into generic
/// transport exhaustion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Reflect)]
#[non_exhaustive]
pub enum LodPageTransportFailureKind {
    Transport,
    Cache,
}

/// Normalized transport failure retained for the last failed page attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LodPageTransportFailure {
    kind: LodPageTransportFailureKind,
    detail: String,
}

impl LodPageTransportFailure {
    pub fn new(kind: LodPageTransportFailureKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }

    pub fn transport(detail: impl Into<String>) -> Self {
        Self::new(LodPageTransportFailureKind::Transport, detail)
    }

    pub fn cache(detail: impl Into<String>) -> Self {
        Self::new(LodPageTransportFailureKind::Cache, detail)
    }

    pub fn kind(&self) -> LodPageTransportFailureKind {
        self.kind
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

/// Transport implemented as a non-blocking state machine.
pub trait LodPageTransport {
    type Ticket: Clone + Eq;
    type Error;

    fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error>;
    fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error>;
    fn cancel(&mut self, ticket: &Self::Ticket);

    /// Converts a backend error into the portable cause retained by the
    /// orchestration layer. Implementations with cache-aware errors should
    /// override this method; the default remains useful for simple/custom
    /// transports that intentionally expose no diagnostic detail.
    fn classify_error(_error: &Self::Error) -> LodPageTransportFailure {
        LodPageTransportFailure::transport("page transport request failed")
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;

    /// Default hard limit for one encoded page read by the native transport.
    pub const DEFAULT_NATIVE_MAX_ENCODED_PAGE_BYTES: u64 = 64 * 1024 * 1024;

    /// Number of filesystem workers shared by every native page transport in
    /// this process. This is deliberately fixed rather than environment-driven
    /// so admission and resource use are reproducible in applications and tests.
    pub const DEFAULT_NATIVE_FILE_IO_WORKERS: usize = 4;

    /// Maximum running plus queued filesystem page reads across the process.
    /// Cancelled tickets continue to occupy a slot until their worker finishes.
    pub const DEFAULT_NATIVE_FILE_IO_IN_FLIGHT_LIMIT: usize = 68;

    /// Keep lazy cancellation O(1) in the common case while bounding stale FIFO
    /// tokens during a long-running filesystem operation.
    const NATIVE_FILE_IO_WAITER_COMPACTION_FLOOR: usize = 64;

    type NativeFileIoJob = Box<dyn FnOnce() + Send + 'static>;
    type NativeFileReadResult = Result<PagePayload, NativeFileTransportError>;
    type NativeFileReadReceiver = std::sync::mpsc::Receiver<NativeFileReadResult>;

    struct NativeFileRead {
        root: std::path::PathBuf,
        path: std::path::PathBuf,
        page_id: LodPageId,
        location: ManifestPageLocation,
        max_encoded_page_bytes: u64,
    }

    enum NativeFileTicketState {
        /// Capacity was unavailable or an earlier waiter owns the next turn.
        /// The pool owns the queued closure and promotes it automatically; the
        /// transport retains only the receiver and cancellation token, so an
        /// owner that temporarily skips polling cannot block later admissions.
        AdmissionDeferred {
            receiver: NativeFileReadReceiver,
            waiter: NativeFileIoWaiter,
        },
        Reading(NativeFileReadReceiver),
    }

    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    struct NativeFileIoWaiter {
        id: u64,
        owner_id: u64,
    }

    enum NativeFileIoSubmission {
        Submitted,
        Deferred(NativeFileIoWaiter),
    }

    struct NativeFileIoScheduledJob {
        owner_id: u64,
        run: NativeFileIoJob,
    }

    struct NativeFileIoPoolState {
        jobs: std::collections::VecDeque<NativeFileIoScheduledJob>,
        ready_owners: std::collections::VecDeque<u64>,
        admission_waiters: std::collections::VecDeque<NativeFileIoWaiter>,
        deferred_jobs: std::collections::HashMap<NativeFileIoWaiter, NativeFileIoJob>,
        in_flight: usize,
        in_flight_limit: usize,
        next_owner_id: u64,
        next_waiter_id: u64,
        shutting_down: bool,
    }

    struct NativeFileIoPoolShared {
        state: std::sync::Mutex<NativeFileIoPoolState>,
        work_available: std::sync::Condvar,
    }

    /// Fixed-size worker pool with FIFO service per transport, deterministic
    /// round-robin service between transports, and pool-driven FIFO admission
    /// after saturation. `in_flight` is incremented before a job is published
    /// and decremented only after its closure returns, so the bound includes
    /// admitted queued, running, cancelled, and receiver-dropped work. Deferred
    /// jobs do not consume that bound and are promoted as soon as capacity frees.
    struct NativeFileIoPool {
        shared: std::sync::Arc<NativeFileIoPoolShared>,
        workers: Vec<std::thread::JoinHandle<()>>,
        in_flight_limit: usize,
    }

    impl NativeFileIoPool {
        fn new(
            worker_count: usize,
            in_flight_limit: usize,
        ) -> Result<Self, NativeFileIoPoolInitError> {
            if worker_count == 0 {
                return Err(NativeFileIoPoolInitError::ZeroWorkers);
            }
            if in_flight_limit < worker_count {
                return Err(NativeFileIoPoolInitError::InFlightLimitBelowWorkerCount {
                    worker_count,
                    in_flight_limit,
                });
            }

            // These queues can contain at most `in_flight_limit` admitted jobs.
            // Reserve that small fixed bound once so automatic promotion cannot
            // encounter a fallible allocation after the requesting call returns.
            let jobs = std::collections::VecDeque::with_capacity(in_flight_limit);
            let ready_owners = std::collections::VecDeque::with_capacity(in_flight_limit);
            let shared = std::sync::Arc::new(NativeFileIoPoolShared {
                state: std::sync::Mutex::new(NativeFileIoPoolState {
                    jobs,
                    ready_owners,
                    admission_waiters: std::collections::VecDeque::new(),
                    deferred_jobs: std::collections::HashMap::new(),
                    in_flight: 0,
                    in_flight_limit,
                    next_owner_id: 1,
                    next_waiter_id: 1,
                    shutting_down: false,
                }),
                work_available: std::sync::Condvar::new(),
            });
            let mut workers = Vec::with_capacity(worker_count);
            for worker_index in 0..worker_count {
                let worker_shared = std::sync::Arc::clone(&shared);
                match std::thread::Builder::new()
                    .name(format!("gaussian-lod-file-io-{worker_index}"))
                    .spawn(move || native_file_io_worker(&worker_shared))
                {
                    Ok(worker) => workers.push(worker),
                    Err(error) => {
                        {
                            let mut state = lock_native_file_io_state(&shared.state);
                            state.shutting_down = true;
                        }
                        shared.work_available.notify_all();
                        for worker in workers {
                            let _ = worker.join();
                        }
                        return Err(NativeFileIoPoolInitError::WorkerSpawn {
                            worker_index,
                            message: error.to_string(),
                        });
                    }
                }
            }

            Ok(Self {
                shared,
                workers,
                in_flight_limit,
            })
        }

        fn allocate_owner_id(&self) -> u64 {
            let mut state = lock_native_file_io_state(&self.shared.state);
            let owner_id = state.next_owner_id;
            state.next_owner_id = state.next_owner_id.wrapping_add(1).max(1);
            owner_id
        }

        #[cfg(test)]
        fn submit(
            &self,
            owner_id: u64,
            job: NativeFileIoJob,
        ) -> Result<(), NativeFileIoPoolAdmissionError> {
            let mut state = lock_native_file_io_state(&self.shared.state);
            if state.shutting_down {
                return Err(NativeFileIoPoolAdmissionError::Unavailable);
            }
            Self::prune_cancelled_waiters(&mut state);
            if state.in_flight >= self.in_flight_limit || !state.deferred_jobs.is_empty() {
                return Err(NativeFileIoPoolAdmissionError::Saturated {
                    in_flight_limit: self.in_flight_limit,
                });
            }
            Self::enqueue(&mut state, owner_id, job);
            drop(state);
            self.shared.work_available.notify_one();
            Ok(())
        }

        fn submit_or_defer(
            &self,
            owner_id: u64,
            job: NativeFileIoJob,
        ) -> Result<NativeFileIoSubmission, NativeFileIoPoolAdmissionError> {
            let mut state = lock_native_file_io_state(&self.shared.state);
            if state.shutting_down {
                return Err(NativeFileIoPoolAdmissionError::Unavailable);
            }
            Self::prune_cancelled_waiters(&mut state);
            if state.in_flight < self.in_flight_limit && state.deferred_jobs.is_empty() {
                Self::enqueue(&mut state, owner_id, job);
                drop(state);
                self.shared.work_available.notify_one();
                return Ok(NativeFileIoSubmission::Submitted);
            }

            state
                .admission_waiters
                .try_reserve(1)
                .map_err(|_| NativeFileIoPoolAdmissionError::QueueAllocationFailed)?;
            state
                .deferred_jobs
                .try_reserve(1)
                .map_err(|_| NativeFileIoPoolAdmissionError::QueueAllocationFailed)?;
            let waiter = NativeFileIoWaiter {
                id: state.next_waiter_id,
                owner_id,
            };
            state.next_waiter_id = state.next_waiter_id.wrapping_add(1).max(1);
            if state.deferred_jobs.contains_key(&waiter) {
                return Err(NativeFileIoPoolAdmissionError::QueueAllocationFailed);
            }
            let replaced = state.deferred_jobs.insert(waiter, job);
            debug_assert!(replaced.is_none());
            state.admission_waiters.push_back(waiter);
            Ok(NativeFileIoSubmission::Deferred(waiter))
        }

        fn cancel_waiter(&self, waiter: NativeFileIoWaiter) -> bool {
            let mut state = lock_native_file_io_state(&self.shared.state);
            if state.deferred_jobs.remove(&waiter).is_none() {
                return false;
            }
            Self::prune_cancelled_waiters(&mut state);
            let promoted = Self::promote_waiters(&mut state);
            drop(state);
            if promoted {
                self.shared.work_available.notify_all();
            }
            true
        }

        fn prune_cancelled_waiters(state: &mut NativeFileIoPoolState) {
            while state
                .admission_waiters
                .front()
                .is_some_and(|waiter| !state.deferred_jobs.contains_key(waiter))
            {
                state.admission_waiters.pop_front();
            }
            if state.deferred_jobs.is_empty() {
                state.admission_waiters.clear();
                return;
            }
            let compact_above = state
                .deferred_jobs
                .len()
                .saturating_mul(2)
                .saturating_add(NATIVE_FILE_IO_WAITER_COMPACTION_FLOOR);
            if state.admission_waiters.len() > compact_above {
                let deferred_jobs = &state.deferred_jobs;
                state
                    .admission_waiters
                    .retain(|waiter| deferred_jobs.contains_key(waiter));
            }
        }

        fn promote_waiters(state: &mut NativeFileIoPoolState) -> bool {
            let mut promoted = false;
            Self::prune_cancelled_waiters(state);
            while state.in_flight < state.in_flight_limit {
                let Some(waiter) = state.admission_waiters.pop_front() else {
                    break;
                };
                let Some(job) = state.deferred_jobs.remove(&waiter) else {
                    continue;
                };
                Self::enqueue(state, waiter.owner_id, job);
                promoted = true;
                Self::prune_cancelled_waiters(state);
            }
            promoted
        }

        fn enqueue(state: &mut NativeFileIoPoolState, owner_id: u64, job: NativeFileIoJob) {
            let owner_already_queued = state.jobs.iter().any(|job| job.owner_id == owner_id);
            debug_assert!(state.jobs.len() < state.in_flight_limit);
            debug_assert!(state.ready_owners.len() < state.in_flight_limit);
            state.in_flight += 1;
            state
                .jobs
                .push_back(NativeFileIoScheduledJob { owner_id, run: job });
            if !owner_already_queued {
                state.ready_owners.push_back(owner_id);
            }
        }

        #[cfg(test)]
        fn in_flight(&self) -> usize {
            lock_native_file_io_state(&self.shared.state).in_flight
        }
    }

    impl Drop for NativeFileIoPool {
        fn drop(&mut self) {
            {
                let mut state = lock_native_file_io_state(&self.shared.state);
                state.shutting_down = true;
            }
            self.shared.work_available.notify_all();
            for worker in self.workers.drain(..) {
                let _ = worker.join();
            }
        }
    }

    fn lock_native_file_io_state(
        state: &std::sync::Mutex<NativeFileIoPoolState>,
    ) -> std::sync::MutexGuard<'_, NativeFileIoPoolState> {
        state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn native_file_io_worker(shared: &NativeFileIoPoolShared) {
        loop {
            let job = {
                let mut state = lock_native_file_io_state(&shared.state);
                loop {
                    if state.shutting_down {
                        return;
                    }
                    if let Some(owner_id) = state.ready_owners.pop_front() {
                        let job_index = state
                            .jobs
                            .iter()
                            .position(|job| job.owner_id == owner_id)
                            .expect("a ready native I/O owner has a queued job");
                        let job = state
                            .jobs
                            .remove(job_index)
                            .expect("the selected native I/O job exists");
                        if state.jobs.iter().any(|job| job.owner_id == owner_id) {
                            state.ready_owners.push_back(owner_id);
                        }
                        break job.run;
                    }
                    state = shared
                        .work_available
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            };

            // A malformed job must not permanently reduce global I/O capacity.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
            let mut state = lock_native_file_io_state(&shared.state);
            state.in_flight = state.in_flight.saturating_sub(1);
            if !state.shutting_down {
                NativeFileIoPool::promote_waiters(&mut state);
            }
            drop(state);
            shared.work_available.notify_all();
        }
    }

    fn shared_native_file_io_pool() -> Result<&'static NativeFileIoPool, NativeFileIoPoolInitError>
    {
        static POOL: std::sync::OnceLock<Result<NativeFileIoPool, NativeFileIoPoolInitError>> =
            std::sync::OnceLock::new();
        POOL.get_or_init(|| {
            NativeFileIoPool::new(
                DEFAULT_NATIVE_FILE_IO_WORKERS,
                DEFAULT_NATIVE_FILE_IO_IN_FLIGHT_LIMIT,
            )
        })
        .as_ref()
        .map_err(Clone::clone)
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum NativeFileIoPoolInitError {
        ZeroWorkers,
        InFlightLimitBelowWorkerCount {
            worker_count: usize,
            in_flight_limit: usize,
        },
        WorkerSpawn {
            worker_index: usize,
            message: String,
        },
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NativeFileIoPoolAdmissionError {
        Saturated { in_flight_limit: usize },
        QueueAllocationFailed,
        Unavailable,
    }

    impl fmt::Display for NativeFileIoPoolInitError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "{self:?}")
        }
    }

    impl std::error::Error for NativeFileIoPoolInitError {}

    impl fmt::Display for NativeFileIoPoolAdmissionError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "{self:?}")
        }
    }

    impl std::error::Error for NativeFileIoPoolAdmissionError {}

    /// Non-blocking native filesystem/range-pack transport. File reads run on
    /// a bounded process-wide fair worker pool; [`LodPageTransport::poll`] never
    /// blocks.
    ///
    /// Manifest URIs must be relative paths below `root`. Absolute paths, URL
    /// schemes, and parent traversal are rejected before any request starts.
    pub struct NativeFilePageTransport {
        root: std::path::PathBuf,
        locations: ManifestPageLocations,
        max_encoded_page_bytes: u64,
        io_pool: &'static NativeFileIoPool,
        io_owner_id: u64,
        tickets: BTreeMap<u64, NativeFileTicketState>,
        next_ticket: u64,
    }

    impl NativeFilePageTransport {
        pub fn new(
            root: impl Into<std::path::PathBuf>,
            locations: ManifestPageLocations,
        ) -> Result<Self, NativeFileTransportError> {
            Self::with_max_encoded_page_bytes(
                root,
                locations,
                DEFAULT_NATIVE_MAX_ENCODED_PAGE_BYTES,
            )
        }

        /// Constructs a transport with an explicit hard bound for one encoded page.
        /// All manifest locations are checked before any worker can be spawned.
        pub fn with_max_encoded_page_bytes(
            root: impl Into<std::path::PathBuf>,
            locations: ManifestPageLocations,
            max_encoded_page_bytes: u64,
        ) -> Result<Self, NativeFileTransportError> {
            let root = root.into();
            if root.as_os_str().is_empty() {
                return Err(NativeFileTransportError::InvalidRoot);
            }
            if max_encoded_page_bytes == 0 {
                return Err(NativeFileTransportError::ZeroMaxEncodedPageBytes);
            }
            for (page_id, location) in locations.iter() {
                validate_native_page_location(page_id, location, max_encoded_page_bytes)?;
            }
            let io_pool = shared_native_file_io_pool()
                .map_err(NativeFileTransportError::IoPoolInitialization)?;
            let io_owner_id = io_pool.allocate_owner_id();
            Ok(Self {
                root,
                locations,
                max_encoded_page_bytes,
                io_pool,
                io_owner_id,
                tickets: BTreeMap::new(),
                next_ticket: 1,
            })
        }

        pub fn from_manifest(
            root: impl Into<std::path::PathBuf>,
            manifest: &GaussianLodManifest,
        ) -> Result<Self, NativeFileTransportError> {
            let locations = ManifestPageLocations::from_manifest(manifest)
                .map_err(NativeFileTransportError::Locations)?;
            Self::new(root, locations)
        }

        /// Manifest constructor paired with [`Self::with_max_encoded_page_bytes`].
        pub fn from_manifest_with_max_encoded_page_bytes(
            root: impl Into<std::path::PathBuf>,
            manifest: &GaussianLodManifest,
            max_encoded_page_bytes: u64,
        ) -> Result<Self, NativeFileTransportError> {
            let locations = ManifestPageLocations::from_manifest(manifest)
                .map_err(NativeFileTransportError::Locations)?;
            Self::with_max_encoded_page_bytes(root, locations, max_encoded_page_bytes)
        }

        pub const fn max_encoded_page_bytes(&self) -> u64 {
            self.max_encoded_page_bytes
        }
    }

    impl LodPageTransport for NativeFilePageTransport {
        type Ticket = u64;
        type Error = NativeFileTransportError;

        fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
            let location = self
                .locations
                .get(request.page_id)
                .cloned()
                .ok_or(NativeFileTransportError::MissingPage(request.page_id))?;
            if request
                .expected_bytes
                .is_some_and(|expected| expected != location.encoded_len)
            {
                return Err(NativeFileTransportError::SizeMismatch {
                    expected: request.expected_bytes.unwrap_or_default(),
                    actual: location.encoded_len,
                });
            }
            validate_native_page_location(request.page_id, &location, self.max_encoded_page_bytes)?;
            let read = std::sync::Arc::new(NativeFileRead {
                path: self.root.join(location.uri.as_ref()),
                root: self.root.clone(),
                page_id: request.page_id,
                location,
                max_encoded_page_bytes: self.max_encoded_page_bytes,
            });
            let state = begin_native_file_read(self.io_pool, self.io_owner_id, read)?;
            let ticket = self.next_ticket;
            self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
            self.tickets.insert(ticket, state);
            Ok(ticket)
        }

        fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
            let Some(state) = self.tickets.remove(ticket) else {
                return PagePoll::Failed(NativeFileTransportError::InvalidTicket(*ticket));
            };
            let (poll, pending_state) = poll_native_file_read(state);
            if let Some(state) = pending_state {
                self.tickets.insert(*ticket, state);
            }
            poll
        }

        fn cancel(&mut self, ticket: &Self::Ticket) {
            if let Some(NativeFileTicketState::AdmissionDeferred { waiter, .. }) =
                self.tickets.remove(ticket)
            {
                self.io_pool.cancel_waiter(waiter);
            }
        }
    }

    impl Drop for NativeFilePageTransport {
        fn drop(&mut self) {
            for state in self.tickets.values() {
                if let NativeFileTicketState::AdmissionDeferred { waiter, .. } = state {
                    self.io_pool.cancel_waiter(*waiter);
                }
            }
        }
    }

    fn begin_native_file_read(
        io_pool: &NativeFileIoPool,
        io_owner_id: u64,
        read: std::sync::Arc<NativeFileRead>,
    ) -> Result<NativeFileTicketState, NativeFileTransportError> {
        let (job, receiver) = native_file_read_job(&read);
        match io_pool.submit_or_defer(io_owner_id, job) {
            Ok(NativeFileIoSubmission::Submitted) => Ok(NativeFileTicketState::Reading(receiver)),
            Ok(NativeFileIoSubmission::Deferred(waiter)) => {
                Ok(NativeFileTicketState::AdmissionDeferred { receiver, waiter })
            }
            Err(error) => Err(NativeFileTransportError::IoPoolAdmission(error)),
        }
    }

    /// Advances one native ticket without blocking. The pool promotes deferred
    /// jobs independently of owner polling, so saturation remains ordinary
    /// transport backpressure rather than a failed page attempt. Admission
    /// setup and completed filesystem errors remain ordinary failures and retain
    /// runtime retry accounting.
    fn poll_native_file_read(
        state: NativeFileTicketState,
    ) -> (
        PagePoll<NativeFileTransportError>,
        Option<NativeFileTicketState>,
    ) {
        match state {
            NativeFileTicketState::AdmissionDeferred { receiver, waiter } => {
                poll_native_file_receiver(receiver, |receiver| {
                    NativeFileTicketState::AdmissionDeferred { receiver, waiter }
                })
            }
            NativeFileTicketState::Reading(receiver) => {
                poll_native_file_receiver(receiver, NativeFileTicketState::Reading)
            }
        }
    }

    fn poll_native_file_receiver(
        receiver: NativeFileReadReceiver,
        pending: impl FnOnce(NativeFileReadReceiver) -> NativeFileTicketState,
    ) -> (
        PagePoll<NativeFileTransportError>,
        Option<NativeFileTicketState>,
    ) {
        match receiver.try_recv() {
            Ok(Ok(payload)) => (PagePoll::Ready(payload), None),
            Ok(Err(error)) => (PagePoll::Failed(error), None),
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                (PagePoll::Pending, Some(pending(receiver)))
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => (
                PagePoll::Failed(NativeFileTransportError::WorkerDisconnected),
                None,
            ),
        }
    }

    fn native_file_read_job(
        read: &std::sync::Arc<NativeFileRead>,
    ) -> (NativeFileIoJob, NativeFileReadReceiver) {
        let read = std::sync::Arc::clone(read);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let job = Box::new(move || {
            let result = read_native_page(
                &read.root,
                &read.path,
                read.page_id,
                &read.location,
                read.max_encoded_page_bytes,
            );
            let _ = sender.send(result);
        });
        (job, receiver)
    }

    fn validate_relative_page_uri(uri: &str) -> Result<(), NativeFileTransportError> {
        use std::path::{Component, Path};

        if uri.is_empty() || uri.contains("://") {
            return Err(NativeFileTransportError::UnsafeUri(uri.to_owned()));
        }
        let path = Path::new(uri);
        if path.is_absolute()
            || path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(NativeFileTransportError::UnsafeUri(uri.to_owned()));
        }
        Ok(())
    }

    fn validate_native_page_location(
        page_id: LodPageId,
        location: &ManifestPageLocation,
        max_encoded_page_bytes: u64,
    ) -> Result<(), NativeFileTransportError> {
        validate_relative_page_uri(&location.uri)?;
        if location.encoded_len == 0 {
            return Err(NativeFileTransportError::ZeroEncodedPageBytes(page_id));
        }
        if location.encoded_len > max_encoded_page_bytes {
            return Err(NativeFileTransportError::EncodedPageTooLarge {
                page: page_id,
                encoded_len: location.encoded_len,
                max_encoded_page_bytes,
            });
        }
        if let Some((start, len)) = location.byte_range {
            if len == 0 || start.checked_add(len).is_none() {
                return Err(NativeFileTransportError::InvalidByteRange {
                    page: page_id,
                    start,
                    len,
                });
            }
            if len != location.encoded_len {
                return Err(NativeFileTransportError::ByteRangeLengthMismatch {
                    page: page_id,
                    range_len: len,
                    encoded_len: location.encoded_len,
                });
            }
        }
        let read_limit = if location.byte_range.is_some() {
            location.encoded_len
        } else {
            location
                .encoded_len
                .checked_add(1)
                .ok_or(NativeFileTransportError::EncodedLengthOverflow)?
        };
        usize::try_from(read_limit).map_err(|_| NativeFileTransportError::EncodedLengthOverflow)?;
        Ok(())
    }

    fn read_native_page(
        root: &std::path::Path,
        path: &std::path::Path,
        page_id: LodPageId,
        location: &ManifestPageLocation,
        max_encoded_page_bytes: u64,
    ) -> Result<PagePayload, NativeFileTransportError> {
        use std::io::Seek;

        validate_native_page_location(page_id, location, max_encoded_page_bytes)?;

        let canonical_root = std::fs::canonicalize(root)
            .map_err(|error| NativeFileTransportError::Io(error.to_string()))?;
        let canonical_path = std::fs::canonicalize(path)
            .map_err(|error| NativeFileTransportError::Io(error.to_string()))?;
        if !canonical_path.starts_with(&canonical_root) {
            return Err(NativeFileTransportError::PathEscapesRoot(page_id));
        }
        let mut file = std::fs::File::open(&canonical_path)
            .map_err(|error| NativeFileTransportError::Io(error.to_string()))?;
        let bytes = if let Some((start, len)) = location.byte_range {
            file.seek(std::io::SeekFrom::Start(start))
                .map_err(|error| NativeFileTransportError::Io(error.to_string()))?;
            read_native_bytes_at_most(&mut file, len)?
        } else {
            let probe_len = location
                .encoded_len
                .checked_add(1)
                .ok_or(NativeFileTransportError::EncodedLengthOverflow)?;
            let bytes = read_native_bytes_at_most(&mut file, probe_len)?;
            if bytes.len() as u64 > location.encoded_len {
                return Err(NativeFileTransportError::ObjectTooLong {
                    page: page_id,
                    expected: location.encoded_len,
                    probed: bytes.len() as u64,
                });
            }
            bytes
        };
        if (bytes.len() as u64) < location.encoded_len {
            return Err(NativeFileTransportError::TruncatedPage {
                page: page_id,
                expected: location.encoded_len,
                actual: bytes.len() as u64,
            });
        }
        Ok(PagePayload::new(page_id, bytes))
    }

    fn read_native_bytes_at_most(
        reader: &mut impl std::io::Read,
        limit: u64,
    ) -> Result<Vec<u8>, NativeFileTransportError> {
        use std::io::Read as _;

        let capacity =
            usize::try_from(limit).map_err(|_| NativeFileTransportError::EncodedLengthOverflow)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(capacity).map_err(|_| {
            NativeFileTransportError::BufferAllocationFailed {
                requested_bytes: limit,
            }
        })?;
        reader
            .take(limit)
            .read_to_end(&mut bytes)
            .map_err(|error| NativeFileTransportError::Io(error.to_string()))?;
        Ok(bytes)
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum NativeFileTransportError {
        InvalidRoot,
        ZeroMaxEncodedPageBytes,
        Locations(PageLocationError),
        IoPoolInitialization(NativeFileIoPoolInitError),
        IoPoolAdmission(NativeFileIoPoolAdmissionError),
        UnsafeUri(String),
        MissingPage(LodPageId),
        InvalidTicket(u64),
        SizeMismatch {
            expected: u64,
            actual: u64,
        },
        ZeroEncodedPageBytes(LodPageId),
        EncodedPageTooLarge {
            page: LodPageId,
            encoded_len: u64,
            max_encoded_page_bytes: u64,
        },
        InvalidByteRange {
            page: LodPageId,
            start: u64,
            len: u64,
        },
        ByteRangeLengthMismatch {
            page: LodPageId,
            range_len: u64,
            encoded_len: u64,
        },
        PathEscapesRoot(LodPageId),
        TruncatedPage {
            page: LodPageId,
            expected: u64,
            actual: u64,
        },
        ObjectTooLong {
            page: LodPageId,
            expected: u64,
            probed: u64,
        },
        BufferAllocationFailed {
            requested_bytes: u64,
        },
        EncodedLengthOverflow,
        WorkerDisconnected,
        Io(String),
    }

    impl fmt::Display for NativeFileTransportError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "{self:?}")
        }
    }

    impl std::error::Error for NativeFileTransportError {}

    #[cfg(test)]
    mod pool_tests {
        use super::*;

        const WAIT: std::time::Duration = std::time::Duration::from_secs(5);

        type TestGate = std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>;

        fn test_gate() -> TestGate {
            std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()))
        }

        fn wait_at_gate(gate: &TestGate) {
            let (released, wake) = &**gate;
            let released = released
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _wait_result = wake
                .wait_timeout_while(released, WAIT, |released| !*released)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }

        fn release_gate(gate: &TestGate) {
            let (released, wake) = &**gate;
            *released
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
            wake.notify_all();
        }

        fn wait_until_idle(pool: &NativeFileIoPool) -> bool {
            let deadline = std::time::Instant::now() + WAIT;
            while pool.in_flight() != 0 {
                if std::time::Instant::now() >= deadline {
                    return false;
                }
                std::thread::yield_now();
            }
            true
        }

        #[test]
        fn pool_configuration_errors_are_typed() {
            assert!(matches!(
                NativeFileIoPool::new(0, 1),
                Err(NativeFileIoPoolInitError::ZeroWorkers)
            ));
            assert!(matches!(
                NativeFileIoPool::new(2, 1),
                Err(NativeFileIoPoolInitError::InFlightLimitBelowWorkerCount {
                    worker_count: 2,
                    in_flight_limit: 1,
                })
            ));
        }

        #[test]
        fn pool_runs_only_its_fixed_worker_count_concurrently() {
            let pool = NativeFileIoPool::new(2, 2).unwrap();
            let gate = test_gate();
            let (started_sender, started_receiver) = std::sync::mpsc::channel();
            let (done_sender, done_receiver) = std::sync::mpsc::channel();

            for id in 1_u8..=2 {
                let gate = std::sync::Arc::clone(&gate);
                let started_sender = started_sender.clone();
                let done_sender = done_sender.clone();
                pool.submit(
                    u64::from(id),
                    Box::new(move || {
                        let _ = started_sender.send(id);
                        wait_at_gate(&gate);
                        let _ = done_sender.send(id);
                    }),
                )
                .unwrap();
            }

            let first_started = started_receiver.recv_timeout(WAIT);
            let second_started = started_receiver.recv_timeout(WAIT);
            let saturated = pool.submit(3, Box::new(|| {}));
            release_gate(&gate);
            let first_done = done_receiver.recv_timeout(WAIT);
            let second_done = done_receiver.recv_timeout(WAIT);

            assert!(first_started.is_ok() && second_started.is_ok());
            assert!(matches!(
                saturated,
                Err(NativeFileIoPoolAdmissionError::Saturated { in_flight_limit: 2 })
            ));
            assert!(first_done.is_ok() && second_done.is_ok());
        }

        #[test]
        fn deferred_native_read_completes_while_its_owner_skips_polling() {
            use std::sync::atomic::{AtomicU64, Ordering};

            static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);

            let pool = NativeFileIoPool::new(1, 1).unwrap();
            let gate = test_gate();
            let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(1);
            let blocker_gate = std::sync::Arc::clone(&gate);
            pool.submit(
                1,
                Box::new(move || {
                    let _ = started_sender.send(());
                    wait_at_gate(&blocker_gate);
                }),
            )
            .unwrap();
            started_receiver.recv_timeout(WAIT).unwrap();

            let unique = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "bevy-gaussian-lod-admission-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir(&root).unwrap();
            std::fs::write(root.join("page.gspage"), [1_u8, 2, 3, 4]).unwrap();
            let read = std::sync::Arc::new(NativeFileRead {
                root: root.clone(),
                path: root.join("page.gspage"),
                page_id: LodPageId(7),
                location: ManifestPageLocation {
                    uri: "page.gspage".into(),
                    byte_range: None,
                    encoded_len: 4,
                },
                max_encoded_page_bytes: 4,
            });

            let state = begin_native_file_read(&pool, 2, read).unwrap();
            let initially_deferred =
                matches!(&state, NativeFileTicketState::AdmissionDeferred { .. });
            let (saturated_poll, state) = poll_native_file_read(state);
            let state = state.expect("a pending native read must retain its ticket state");
            let remained_deferred =
                matches!(&state, NativeFileTicketState::AdmissionDeferred { .. });
            let saturated_in_flight = pool.in_flight();

            release_gate(&gate);
            assert!(wait_until_idle(&pool));
            assert!(initially_deferred);
            assert!(matches!(saturated_poll, PagePoll::Pending));
            assert!(remained_deferred);
            assert_eq!(saturated_in_flight, 1);

            // The owner deliberately did not poll while capacity recovered.
            // Pool-driven promotion must still complete the read and release
            // the global slot before the owner resumes.
            let (completed_poll, next_state) = poll_native_file_read(state);
            let payload = match completed_poll {
                PagePoll::Ready(payload) => payload,
                PagePoll::Pending => panic!("auto-promoted native read did not complete"),
                PagePoll::Failed(error) => panic!("deferred native read failed: {error:?}"),
            };
            assert!(next_state.is_none());
            assert_eq!(payload.page_id, LodPageId(7));
            assert_eq!(payload.bytes, [1, 2, 3, 4]);

            std::fs::remove_file(root.join("page.gspage")).unwrap();
            std::fs::remove_dir(root).unwrap();
        }

        #[test]
        fn saturated_admission_waiters_cannot_be_leapfrogged_by_a_busy_owner() {
            let pool = NativeFileIoPool::new(1, 1).unwrap();
            let gate = test_gate();
            let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(1);
            let blocker_gate = std::sync::Arc::clone(&gate);
            pool.submit(
                1,
                Box::new(move || {
                    let _ = started_sender.send(());
                    wait_at_gate(&blocker_gate);
                }),
            )
            .unwrap();
            started_receiver.recv_timeout(WAIT).unwrap();

            let (order_sender, order_receiver) = std::sync::mpsc::channel();
            let waiting_sender = order_sender.clone();
            assert!(matches!(
                pool.submit_or_defer(
                    2,
                    Box::new(move || {
                        let _ = waiting_sender.send(2_u8);
                    })
                )
                .unwrap(),
                NativeFileIoSubmission::Deferred(_)
            ));
            assert!(matches!(
                pool.submit_or_defer(
                    1,
                    Box::new(move || {
                        let _ = order_sender.send(1_u8);
                    })
                )
                .unwrap(),
                NativeFileIoSubmission::Deferred(_)
            ));

            release_gate(&gate);
            assert_eq!(order_receiver.recv_timeout(WAIT).unwrap(), 2);
            assert_eq!(order_receiver.recv_timeout(WAIT).unwrap(), 1);
            assert!(wait_until_idle(&pool));
        }

        #[test]
        fn cancelling_an_admission_waiter_releases_the_next_reservation() {
            let pool = NativeFileIoPool::new(1, 1).unwrap();
            let gate = test_gate();
            let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(1);
            let blocker_gate = std::sync::Arc::clone(&gate);
            pool.submit(
                1,
                Box::new(move || {
                    let _ = started_sender.send(());
                    wait_at_gate(&blocker_gate);
                }),
            )
            .unwrap();
            started_receiver.recv_timeout(WAIT).unwrap();

            let (completed_sender, completed_receiver) = std::sync::mpsc::channel();
            let first_sender = completed_sender.clone();
            let first = match pool
                .submit_or_defer(
                    2,
                    Box::new(move || {
                        let _ = first_sender.send(1_u8);
                    }),
                )
                .unwrap()
            {
                NativeFileIoSubmission::Deferred(waiter) => waiter,
                NativeFileIoSubmission::Submitted => panic!("the full pool must defer owner 2"),
            };
            assert!(matches!(
                pool.submit_or_defer(
                    3,
                    Box::new(move || {
                        let _ = completed_sender.send(2_u8);
                    })
                )
                .unwrap(),
                NativeFileIoSubmission::Deferred(_)
            ));
            for owner_id in 4..260 {
                let waiter = match pool.submit_or_defer(owner_id, Box::new(|| {})).unwrap() {
                    NativeFileIoSubmission::Deferred(waiter) => waiter,
                    NativeFileIoSubmission::Submitted => {
                        panic!("the full pool must defer cancellation churn")
                    }
                };
                assert!(pool.cancel_waiter(waiter));
            }
            {
                let state = lock_native_file_io_state(&pool.shared.state);
                assert_eq!(state.deferred_jobs.len(), 2);
                assert!(
                    state.admission_waiters.len()
                        <= state.deferred_jobs.len() * 2 + NATIVE_FILE_IO_WAITER_COMPACTION_FLOOR
                );
            }
            assert!(pool.cancel_waiter(first));

            release_gate(&gate);
            assert_eq!(completed_receiver.recv_timeout(WAIT).unwrap(), 2);
            assert!(wait_until_idle(&pool));
            assert!(completed_receiver.try_recv().is_err());
        }

        #[test]
        fn pool_is_fifo_per_owner_and_round_robin_between_owners() {
            let pool = NativeFileIoPool::new(1, 5).unwrap();
            let gate = test_gate();
            let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(1);
            let (order_sender, order_receiver) = std::sync::mpsc::channel();

            let first_gate = std::sync::Arc::clone(&gate);
            let first_order = order_sender.clone();
            pool.submit(
                1,
                Box::new(move || {
                    let _ = started_sender.send(());
                    wait_at_gate(&first_gate);
                    let _ = first_order.send(1_u8);
                }),
            )
            .unwrap();
            started_receiver.recv_timeout(WAIT).unwrap();

            for id in [2_u8, 4] {
                let order_sender = order_sender.clone();
                pool.submit(
                    1,
                    Box::new(move || {
                        let _ = order_sender.send(id);
                    }),
                )
                .unwrap();
            }
            for id in [3_u8, 5] {
                let order_sender = order_sender.clone();
                pool.submit(
                    2,
                    Box::new(move || {
                        let _ = order_sender.send(id);
                    }),
                )
                .unwrap();
            }
            release_gate(&gate);

            let order = (0..5)
                .map(|_| order_receiver.recv_timeout(WAIT).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(order, [1, 2, 3, 4, 5]);
        }

        #[test]
        fn dropped_receiver_stays_charged_until_completion_then_capacity_recovers() {
            let pool = NativeFileIoPool::new(1, 2).unwrap();
            let gate = test_gate();
            let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(1);
            let (done_sender, done_receiver) = std::sync::mpsc::channel();

            let first_gate = std::sync::Arc::clone(&gate);
            let first_done = done_sender.clone();
            pool.submit(
                1,
                Box::new(move || {
                    let _ = started_sender.send(());
                    wait_at_gate(&first_gate);
                    let _ = first_done.send(1_u8);
                }),
            )
            .unwrap();
            started_receiver.recv_timeout(WAIT).unwrap();

            let (result_sender, result_receiver) = std::sync::mpsc::sync_channel::<u8>(1);
            let second_done = done_sender.clone();
            pool.submit(
                1,
                Box::new(move || {
                    let _ = result_sender.send(2);
                    let _ = second_done.send(2_u8);
                }),
            )
            .unwrap();
            drop(result_receiver);

            assert!(matches!(
                pool.submit(2, Box::new(|| {})),
                Err(NativeFileIoPoolAdmissionError::Saturated { in_flight_limit: 2 })
            ));

            release_gate(&gate);
            assert!(done_receiver.recv_timeout(WAIT).is_ok());
            assert!(done_receiver.recv_timeout(WAIT).is_ok());
            assert!(wait_until_idle(&pool));

            let (recovered_sender, recovered_receiver) = std::sync::mpsc::sync_channel(1);
            pool.submit(
                2,
                Box::new(move || {
                    let _ = recovered_sender.send(());
                }),
            )
            .unwrap();
            recovered_receiver.recv_timeout(WAIT).unwrap();
        }

        #[test]
        fn native_transports_use_one_process_wide_pool() {
            let first = shared_native_file_io_pool().unwrap();
            let second = shared_native_file_io_pool().unwrap();
            assert!(std::ptr::eq(first, second));
            assert_eq!(
                first.in_flight_limit,
                DEFAULT_NATIVE_FILE_IO_IN_FLIGHT_LIMIT
            );
            assert_eq!(first.workers.len(), DEFAULT_NATIVE_FILE_IO_WORKERS);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::{
    DEFAULT_NATIVE_FILE_IO_IN_FLIGHT_LIMIT, DEFAULT_NATIVE_FILE_IO_WORKERS,
    DEFAULT_NATIVE_MAX_ENCODED_PAGE_BYTES, NativeFileIoPoolAdmissionError,
    NativeFileIoPoolInitError, NativeFilePageTransport, NativeFileTransportError,
};

/// Small deterministic source for unit tests, generated scenes, and embedded pages.
#[derive(Clone, Debug, Default)]
pub struct MemoryPageTransport {
    pages: BTreeMap<LodPageId, Vec<u8>>,
    tickets: BTreeMap<u64, LodPageId>,
    next_ticket: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryTransportError {
    InvalidPageId,
    MissingPage(LodPageId),
    InvalidTicket(u64),
    SizeMismatch { expected: u64, actual: u64 },
}

impl MemoryPageTransport {
    pub fn insert(&mut self, page_id: LodPageId, bytes: Vec<u8>) -> Option<Vec<u8>> {
        self.pages.insert(page_id, bytes)
    }
}

impl LodPageTransport for MemoryPageTransport {
    type Ticket = u64;
    type Error = MemoryTransportError;

    fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
        if !request.page_id.is_valid() {
            return Err(MemoryTransportError::InvalidPageId);
        }
        let bytes = self
            .pages
            .get(&request.page_id)
            .ok_or(MemoryTransportError::MissingPage(request.page_id))?;
        if let Some(expected) = request.expected_bytes {
            let actual = bytes.len() as u64;
            if actual != expected {
                return Err(MemoryTransportError::SizeMismatch { expected, actual });
            }
        }
        let ticket = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1);
        self.tickets.insert(ticket, request.page_id);
        Ok(ticket)
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
        let Some(page_id) = self.tickets.remove(ticket) else {
            return PagePoll::Failed(MemoryTransportError::InvalidTicket(*ticket));
        };
        match self.pages.get(&page_id) {
            Some(bytes) => PagePoll::Ready(PagePayload::new(page_id, bytes.clone())),
            None => PagePoll::Failed(MemoryTransportError::MissingPage(page_id)),
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        self.tickets.remove(ticket);
    }
}

/// FNV-1a is intentionally simple and stable here. A manifest codec may use a
/// stronger content hash in addition; this catches corruption without a new runtime dependency.
pub fn page_checksum64(bytes: &[u8]) -> u64 {
    let mut checksum = PageChecksum64::new();
    checksum.update(bytes);
    checksum.finish()
}

/// Incremental form of [`page_checksum64`] used by the caller-thread Wasm
/// preprocessor. Keeping the state here guarantees that chunked verification
/// remains byte-for-byte identical to transport and persistent-cache checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PageChecksum64(u64);

impl PageChecksum64 {
    pub(crate) const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub(crate) const fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gaussian::formats::{
            planar_3d_chunked::LodPageStorage,
            planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
        },
        testing::LodTestScene,
    };

    fn request(id: u64, priority: PageRequestPriority) -> PageRequest {
        PageRequest::new(LodPageId(id), priority)
    }

    #[test]
    fn manifest_locations_use_dense_shared_storage_and_sparse_fallback() {
        let mut built = build_planar_3d_lod(
            &LodTestScene::nested_octants(2).cloud(),
            GaussianLodBuildSettings {
                leaf_capacity: 8,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(built.manifest.pages.len() > 1);
        for (index, descriptor) in built.manifest.pages.iter_mut().enumerate() {
            descriptor.storage = Some(LodPageStorage {
                uri: "scene.pack".to_owned(),
                byte_range: Some((index as u64 * 4, 4)),
                encoded_len: 4,
            });
        }
        built.manifest.validate().unwrap();

        let dense = ManifestPageLocations::from_manifest(&built.manifest).unwrap();
        assert!(dense.index.is_dense_one_based());
        let cloned = dense.clone();
        assert!(Arc::ptr_eq(&dense.entries, &cloned.entries));
        let first = dense.get(built.manifest.pages[0].id).unwrap();
        let second = dense.get(built.manifest.pages[1].id).unwrap();
        assert!(first.uri.ptr_eq(&second.uri));
        assert_eq!(
            dense.page_ids().collect::<Vec<_>>(),
            built
                .manifest
                .pages
                .iter()
                .map(|descriptor| descriptor.id)
                .collect::<Vec<_>>()
        );

        let sparse = ManifestPageLocations::from_entries([
            (
                LodPageId(9),
                ManifestPageLocation {
                    uri: "a.gspage".into(),
                    byte_range: None,
                    encoded_len: 1,
                },
            ),
            (
                LodPageId(3),
                ManifestPageLocation {
                    uri: "b.gspage".into(),
                    byte_range: None,
                    encoded_len: 1,
                },
            ),
        ]);
        assert!(!sparse.index.is_dense_one_based());
        assert_eq!(sparse.get(LodPageId(3)).unwrap().uri.as_ref(), "b.gspage");
        assert!(sparse.get(LodPageId(1)).is_none());

        let encoded = serde_json::to_string(first).unwrap();
        let decoded: ManifestPageLocation = serde_json::from_str(&encoded).unwrap();
        assert_eq!(&decoded, first);
    }

    #[test]
    fn incremental_page_checksum_matches_one_shot_across_chunk_boundaries() {
        let bytes = (0_u16..1_025)
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let expected = page_checksum64(&bytes);
        for chunk_size in [1, 3, 44, 257, bytes.len()] {
            let mut checksum = PageChecksum64::new();
            for chunk in bytes.chunks(chunk_size) {
                checksum.update(chunk);
            }
            assert_eq!(checksum.finish(), expected);
        }
    }

    #[test]
    fn queue_deduplicates_promotes_and_orders_deterministically() {
        let mut queue = PageRequestQueue::new(4).unwrap();
        assert_eq!(
            queue.enqueue(request(2, PageRequestPriority::prefetch(3))),
            RequestEnqueue::Enqueued
        );
        assert_eq!(
            queue.enqueue(request(1, PageRequestPriority::visible(1))),
            RequestEnqueue::Enqueued
        );
        assert_eq!(
            queue.enqueue(request(2, PageRequestPriority::prefetch(2))),
            RequestEnqueue::Duplicate
        );
        assert_eq!(
            queue.enqueue(request(2, PageRequestPriority::fallback_critical(0))),
            RequestEnqueue::Promoted
        );
        assert_eq!(queue.len(), 2);
        assert_eq!(
            queue.page_ids().collect::<Vec<_>>(),
            [LodPageId(1), LodPageId(2)]
        );
        assert_eq!(queue.pop().unwrap().page_id, LodPageId(2));
        assert_eq!(queue.pop().unwrap().page_id, LodPageId(1));
    }

    #[test]
    fn bounded_queue_only_replaces_lower_priority_work() {
        let mut queue = PageRequestQueue::new(2).unwrap();
        queue.enqueue(request(1, PageRequestPriority::prefetch(1)));
        queue.enqueue(request(2, PageRequestPriority::visible(1)));
        assert_eq!(
            queue.enqueue(request(3, PageRequestPriority::prefetch(0))),
            RequestEnqueue::Rejected
        );
        assert_eq!(
            queue.enqueue(request(4, PageRequestPriority::fallback_critical(1))),
            RequestEnqueue::Replaced(LodPageId(1))
        );
        assert!(!queue.contains(LodPageId(1)));
        assert!(queue.contains(LodPageId(4)));
        assert_eq!(
            queue.enqueue(request(0, PageRequestPriority::fallback_critical(u32::MAX))),
            RequestEnqueue::Rejected
        );
    }

    #[test]
    fn memory_transport_validates_size_and_ticket_lifecycle() {
        let mut transport = MemoryPageTransport::default();
        transport.insert(LodPageId(7), vec![1, 2, 3]);
        let mut page_request = request(7, PageRequestPriority::visible(1));
        page_request.expected_bytes = Some(3);
        let ticket = transport.begin(page_request).unwrap();
        let PagePoll::Ready(payload) = transport.poll(&ticket) else {
            panic!("expected ready payload");
        };
        assert!(payload.verify());
        assert!(matches!(
            transport.poll(&ticket),
            PagePoll::Failed(MemoryTransportError::InvalidTicket(_))
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_transport_validates_encoded_page_limits_before_workers() {
        let page_id = LodPageId(3);
        let locations = ManifestPageLocations::from_entries([(
            page_id,
            ManifestPageLocation {
                uri: "page.gspage".into(),
                byte_range: None,
                encoded_len: 5,
            },
        )]);
        assert!(matches!(
            NativeFilePageTransport::with_max_encoded_page_bytes("safe-root", locations.clone(), 0,),
            Err(NativeFileTransportError::ZeroMaxEncodedPageBytes)
        ));
        assert!(matches!(
            NativeFilePageTransport::with_max_encoded_page_bytes(
                "safe-root",
                locations.clone(),
                4,
            ),
            Err(NativeFileTransportError::EncodedPageTooLarge {
                page,
                encoded_len: 5,
                max_encoded_page_bytes: 4,
            }) if page == page_id
        ));

        let transport = NativeFilePageTransport::new("safe-root", locations).unwrap();
        assert_eq!(
            transport.max_encoded_page_bytes(),
            DEFAULT_NATIVE_MAX_ENCODED_PAGE_BYTES
        );

        let mismatched_range = ManifestPageLocations::from_entries([(
            page_id,
            ManifestPageLocation {
                uri: "pack.bin".into(),
                byte_range: Some((7, 6)),
                encoded_len: 5,
            },
        )]);
        assert!(matches!(
            NativeFilePageTransport::with_max_encoded_page_bytes(
                "safe-root",
                mismatched_range,
                8,
            ),
            Err(NativeFileTransportError::ByteRangeLengthMismatch {
                page,
                range_len: 6,
                encoded_len: 5,
            }) if page == page_id
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_transport_reads_bounded_pack_ranges_without_blocking_poll() {
        use std::{
            fs,
            sync::atomic::{AtomicU64, Ordering},
        };

        static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
        let unique = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "bevy-gaussian-lod-transport-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("scene.pack"), [99, 98, 1, 2, 3, 4, 97]).unwrap();

        let locations = ManifestPageLocations::from_entries([(
            LodPageId(9),
            ManifestPageLocation {
                uri: "scene.pack".into(),
                byte_range: Some((2, 4)),
                encoded_len: 4,
            },
        )]);
        let mut transport = NativeFilePageTransport::new(&root, locations).unwrap();
        let mut request = PageRequest::new(LodPageId(9), PageRequestPriority::visible(1));
        request.expected_bytes = Some(4);
        let ticket = transport.begin(request).unwrap();
        let payload = loop {
            match transport.poll(&ticket) {
                PagePoll::Pending => std::thread::yield_now(),
                PagePoll::Ready(payload) => break payload,
                PagePoll::Failed(error) => panic!("native page read failed: {error:?}"),
            }
        };
        assert_eq!(payload.bytes, [1, 2, 3, 4]);
        assert!(payload.verify());
        fs::remove_file(root.join("scene.pack")).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_transport_object_reads_probe_only_one_extra_byte() {
        use std::{
            fs,
            sync::atomic::{AtomicU64, Ordering},
        };

        static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
        let unique = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "bevy-gaussian-lod-object-bound-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let page_path = root.join("page.gspage");
        fs::write(&page_path, [1, 2, 3, 4, 5, 6]).unwrap();

        let page_id = LodPageId(11);
        let locations = ManifestPageLocations::from_entries([(
            page_id,
            ManifestPageLocation {
                uri: "page.gspage".into(),
                byte_range: None,
                encoded_len: 4,
            },
        )]);
        let mut transport =
            NativeFilePageTransport::with_max_encoded_page_bytes(&root, locations, 4).unwrap();
        let ticket = transport
            .begin(PageRequest::new(page_id, PageRequestPriority::visible(1)))
            .unwrap();
        let error = loop {
            match transport.poll(&ticket) {
                PagePoll::Pending => std::thread::yield_now(),
                PagePoll::Ready(_) => panic!("oversized object unexpectedly succeeded"),
                PagePoll::Failed(error) => break error,
            }
        };
        assert_eq!(
            error,
            NativeFileTransportError::ObjectTooLong {
                page: page_id,
                expected: 4,
                probed: 5,
            }
        );

        fs::write(&page_path, [1, 2, 3]).unwrap();
        let ticket = transport
            .begin(PageRequest::new(page_id, PageRequestPriority::visible(1)))
            .unwrap();
        let error = loop {
            match transport.poll(&ticket) {
                PagePoll::Pending => std::thread::yield_now(),
                PagePoll::Ready(_) => panic!("truncated object unexpectedly succeeded"),
                PagePoll::Failed(error) => break error,
            }
        };
        assert_eq!(
            error,
            NativeFileTransportError::TruncatedPage {
                page: page_id,
                expected: 4,
                actual: 3,
            }
        );

        fs::remove_file(page_path).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_transport_rejects_absolute_parent_and_url_uris() {
        for uri in ["../page.gspage", "/tmp/page.gspage", "https://example/page"] {
            let locations = ManifestPageLocations::from_entries([(
                LodPageId(1),
                ManifestPageLocation {
                    uri: uri.into(),
                    byte_range: None,
                    encoded_len: 1,
                },
            )]);
            assert!(matches!(
                NativeFilePageTransport::new("safe-root", locations),
                Err(NativeFileTransportError::UnsafeUri(_))
            ));
        }
    }

    #[cfg(all(not(target_arch = "wasm32"), unix))]
    #[test]
    fn native_transport_rejects_symlink_escape_from_package_root() {
        use std::{
            fs,
            sync::atomic::{AtomicU64, Ordering},
        };

        static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
        let unique = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let base = std::env::temp_dir().join(format!(
            "bevy-gaussian-lod-symlink-bound-{}-{unique}",
            std::process::id()
        ));
        let root = base.join("package");
        let outside = base.join("outside.gspage");
        fs::create_dir_all(&root).unwrap();
        fs::write(&outside, [7]).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("page.gspage")).unwrap();

        let page_id = LodPageId(13);
        let locations = ManifestPageLocations::from_entries([(
            page_id,
            ManifestPageLocation {
                uri: "page.gspage".into(),
                byte_range: None,
                encoded_len: 1,
            },
        )]);
        let mut transport = NativeFilePageTransport::new(&root, locations).unwrap();
        let ticket = transport
            .begin(PageRequest::new(page_id, PageRequestPriority::visible(1)))
            .unwrap();
        let error = loop {
            match transport.poll(&ticket) {
                PagePoll::Pending => std::thread::yield_now(),
                PagePoll::Ready(_) => panic!("symlink escape unexpectedly succeeded"),
                PagePoll::Failed(error) => break error,
            }
        };
        assert_eq!(error, NativeFileTransportError::PathEscapesRoot(page_id));

        fs::remove_file(root.join("page.gspage")).unwrap();
        fs::remove_file(outside).unwrap();
        fs::remove_dir(root).unwrap();
        fs::remove_dir(base).unwrap();
    }
}
