//! Bounded LoD page verification, decoding, and support-bound validation.
//!
//! Native builds submit CPU-heavy page work to one fixed-size process-wide
//! worker pool. The pool has hard running-plus-queued job and byte limits and
//! does not consult environment variables, so library behavior is reproducible
//! and safe to compile for Wasm. Browser builds use a bounded cooperative
//! backend that incrementally verifies and decodes at most one page slice per
//! application frame. [`LodPagePreprocessStats`] exposes both the configured
//! slice budget and active decode progress instead of claiming work was moved
//! off the browser main thread.

use std::{
    collections::{BTreeMap, VecDeque},
    num::NonZeroU32,
};

#[cfg(any(test, target_arch = "wasm32"))]
mod cooperative;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
use native as platform;
#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(target_arch = "wasm32")]
use wasm as platform;

use crate::{
    gaussian::formats::{
        planar_3d::Gaussian3d,
        planar_3d_chunked::{LodBounds, LodPageDescriptor, LodPageId, PlanarGaussian3dPage},
        planar_3d_lod::gaussian_support_bounds,
    },
    io::lod::{LodCodecError, LodCodecLimits},
    stream::transport::{PagePayload, PageRequest},
};

#[cfg(not(target_arch = "wasm32"))]
use crate::io::lod::decode_page_with_descriptor;

/// Native worker count shared by every LoD streaming runtime in the process.
pub const DEFAULT_NATIVE_PAGE_PREPROCESS_WORKERS: usize = 4;

/// Maximum running plus queued native page jobs across the process.
pub const DEFAULT_NATIVE_PAGE_PREPROCESS_IN_FLIGHT_LIMIT: usize = 68;

/// Maximum encoded-payload plus declared-decoded bytes charged to running and
/// queued native page jobs across the process.
pub const DEFAULT_NATIVE_PAGE_PREPROCESS_PENDING_BYTES_LIMIT: u64 = 512 * 1024 * 1024;

/// Execution mode used by the page preprocessing stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodPagePreprocessBackend {
    /// Fixed-size, process-wide native worker pool.
    NativeWorkerPool,
    /// Record-bounded incremental work on the caller thread.
    CooperativeWasm,
}

impl Default for LodPagePreprocessBackend {
    fn default() -> Self {
        if cfg!(target_arch = "wasm32") {
            Self::CooperativeWasm
        } else {
            Self::NativeWorkerPool
        }
    }
}

/// Observable bounded-state counters for one streaming runtime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LodPagePreprocessStats {
    pub backend: LodPagePreprocessBackend,
    pub capacity: u32,
    pub waiting: u32,
    /// Jobs handed to the platform backend. This includes native queued and
    /// running work, or the single active cooperative decoder.
    pub submitted: u32,
    pub ready: u32,
    /// Per-runtime encoded-payload plus declared-decoded byte capacity.
    pub byte_capacity: u64,
    /// Bytes charged by waiting, submitted (including cancelled but not yet
    /// acknowledged), and ready pages.
    pub pending_bytes: u64,
    /// Number of native submissions deferred because the process-wide pool was
    /// at a hard job/byte bound or could not reserve queue storage.
    pub deferred_admissions: u64,
    pub cancellations: u64,
    /// Records decoded by the active cooperative job. Zero on native and when
    /// no cooperative job is active.
    pub cooperative_decoded_gaussians: u32,
    /// Total records declared by the active cooperative job. Zero on native
    /// and when no cooperative job is active.
    pub cooperative_total_gaussians: u32,
    /// Record budget supplied to the most recent cooperative advance. Zero for
    /// the native worker-pool backend and before the first browser advance.
    pub cooperative_budget_gaussians_per_frame: u32,
}

impl LodPagePreprocessStats {
    pub fn pending(self) -> u32 {
        self.waiting
            .saturating_add(self.submitted)
            .saturating_add(self.ready)
    }
}

/// A typed validation failure produced before a page may enter residency.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LodPagePreprocessError {
    PayloadPageMismatch {
        expected: LodPageId,
        actual: LodPageId,
    },
    EncodedPageLimitExceeded {
        actual: u64,
        limit: u64,
    },
    PayloadChecksumMismatch,
    Codec(LodCodecError),
    InvalidSupportBounds(LodPageId),
    PayloadOutsideDescriptor(LodPageId),
    /// A decoded physical page was valid as a whole, but records assigned to a
    /// logical node escaped that node's culling bounds.
    PayloadOutsideNodeBounds {
        page: LodPageId,
        node: crate::gaussian::formats::planar_3d_chunked::LodNodeId,
    },
    WorkerPanicked,
}

impl std::fmt::Display for LodPagePreprocessError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LodPagePreprocessError {}

/// Failure to construct or admit work to a bounded preprocessing stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LodPagePreprocessAdmissionError {
    ZeroCapacity,
    ZeroByteCapacity,
    DuplicatePage(LodPageId),
    CapacityExhausted {
        capacity: usize,
    },
    PendingByteCapacityExceeded {
        requested: u64,
        pending: u64,
        capacity: u64,
    },
    NativeJobByteCapacityExceeded {
        requested: u64,
        capacity: u64,
    },
    ByteLengthOverflow,
    NativePoolInitialization(String),
}

impl std::fmt::Display for LodPagePreprocessAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LodPagePreprocessAdmissionError {}

pub(crate) struct LodPagePreprocessInput {
    pub request: PageRequest,
    pub payload: PagePayload,
    pub descriptor: LodPageDescriptor,
    pub limits: LodCodecLimits,
    pub max_encoded_page_bytes: u64,
    pub support_sigma: f32,
}

impl LodPagePreprocessInput {
    pub(crate) fn pending_bytes(&self) -> Result<u64, LodPagePreprocessAdmissionError> {
        (self.payload.bytes.len() as u64)
            .checked_add(self.descriptor.decoded_len)
            .ok_or(LodPagePreprocessAdmissionError::ByteLengthOverflow)
    }
}

pub(crate) struct LodPagePreprocessOutput {
    pub request: PageRequest,
    pub result: Result<PlanarGaussian3dPage, LodPagePreprocessError>,
}

pub(super) struct WaitingJob {
    pub(super) input: LodPagePreprocessInput,
    pub(super) pending_bytes: u64,
}

pub(super) struct ReadyJob {
    pub(super) output: LodPagePreprocessOutput,
    pub(super) pending_bytes: u64,
}

/// Per-runtime bounded admission and completion state.
pub(crate) struct LodPagePreprocessor {
    capacity: usize,
    byte_capacity: u64,
    pending_bytes: u64,
    waiting: VecDeque<WaitingJob>,
    ready: BTreeMap<LodPageId, ReadyJob>,
    backend: platform::BackendState,
    deferred_admissions: u64,
    cancellations: u64,
}

impl LodPagePreprocessor {
    pub(crate) fn with_byte_capacity(
        capacity: usize,
        byte_capacity: u64,
    ) -> Result<Self, LodPagePreprocessAdmissionError> {
        Self::validate_capacity(capacity, byte_capacity)?;
        let backend = platform::BackendState::new()
            .map_err(LodPagePreprocessAdmissionError::NativePoolInitialization)?;
        Ok(Self::with_backend(capacity, byte_capacity, backend))
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn new_native_for_tests(
        capacity: usize,
        byte_capacity: u64,
    ) -> Result<Self, LodPagePreprocessAdmissionError> {
        Self::validate_capacity(capacity, byte_capacity)?;
        let backend = platform::BackendState::new_native_for_tests()
            .map_err(LodPagePreprocessAdmissionError::NativePoolInitialization)?;
        Ok(Self::with_backend(capacity, byte_capacity, backend))
    }

    #[cfg(test)]
    pub(crate) fn new_cooperative_for_tests(
        capacity: usize,
    ) -> Result<Self, LodPagePreprocessAdmissionError> {
        Self::new_cooperative_with_byte_capacity_for_tests(
            capacity,
            DEFAULT_NATIVE_PAGE_PREPROCESS_PENDING_BYTES_LIMIT,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_cooperative_with_byte_capacity_for_tests(
        capacity: usize,
        byte_capacity: u64,
    ) -> Result<Self, LodPagePreprocessAdmissionError> {
        Self::validate_capacity(capacity, byte_capacity)?;
        Ok(Self::with_backend(
            capacity,
            byte_capacity,
            platform::BackendState::new_cooperative_for_tests(),
        ))
    }

    pub(crate) fn submit(
        &mut self,
        input: LodPagePreprocessInput,
    ) -> Result<(), LodPagePreprocessAdmissionError> {
        let page = input.request.page_id;
        if self.contains(page) {
            return Err(LodPagePreprocessAdmissionError::DuplicatePage(page));
        }
        if self.len() >= self.capacity {
            return Err(LodPagePreprocessAdmissionError::CapacityExhausted {
                capacity: self.capacity,
            });
        }
        let pending_bytes = input.pending_bytes()?;
        self.validate_native_job_bytes(pending_bytes)?;
        let next_pending = self
            .pending_bytes
            .checked_add(pending_bytes)
            .ok_or(LodPagePreprocessAdmissionError::ByteLengthOverflow)?;
        if next_pending > self.byte_capacity {
            return Err(
                LodPagePreprocessAdmissionError::PendingByteCapacityExceeded {
                    requested: pending_bytes,
                    pending: self.pending_bytes,
                    capacity: self.byte_capacity,
                },
            );
        }
        self.pending_bytes = next_pending;
        self.waiting.push_back(WaitingJob {
            input,
            pending_bytes,
        });
        Ok(())
    }

    /// Advances completion delivery and admission without blocking. The
    /// cooperative backend processes at most one bounded record slice for a
    /// given application-frame sequence, regardless of camera count.
    pub(crate) fn advance(&mut self, frame_sequence: u64, cooperative_budget: NonZeroU32) {
        self.backend.advance(
            frame_sequence,
            cooperative_budget,
            &mut self.waiting,
            &mut self.ready,
            &mut self.pending_bytes,
            &mut self.deferred_admissions,
        );
    }

    pub(crate) fn validate_job_bytes(
        &self,
        encoded_bytes: u64,
        decoded_bytes: u64,
    ) -> Result<u64, LodPagePreprocessAdmissionError> {
        let pending_bytes = encoded_bytes
            .checked_add(decoded_bytes)
            .ok_or(LodPagePreprocessAdmissionError::ByteLengthOverflow)?;
        self.validate_native_job_bytes(pending_bytes)?;
        if pending_bytes > self.byte_capacity {
            return Err(
                LodPagePreprocessAdmissionError::PendingByteCapacityExceeded {
                    requested: pending_bytes,
                    pending: 0,
                    capacity: self.byte_capacity,
                },
            );
        }
        Ok(pending_bytes)
    }

    pub(crate) fn has_capacity_for(&self, encoded_bytes: u64, decoded_bytes: u64) -> bool {
        if self.len() >= self.capacity {
            return false;
        }
        encoded_bytes
            .checked_add(decoded_bytes)
            .and_then(|requested| self.pending_bytes.checked_add(requested))
            .is_some_and(|pending| pending <= self.byte_capacity)
    }

    pub(crate) fn contains(&self, page: LodPageId) -> bool {
        self.waiting
            .iter()
            .any(|waiting| waiting.input.request.page_id == page)
            || self.ready.contains_key(&page)
            || self.backend.is_running(page)
    }

    pub(crate) fn len(&self) -> usize {
        self.waiting
            .len()
            .saturating_add(self.ready.len())
            .saturating_add(self.backend.tracked_len())
    }

    pub(crate) fn page_ids(&self) -> Vec<LodPageId> {
        let mut pages = self
            .waiting
            .iter()
            .map(|waiting| waiting.input.request.page_id)
            .chain(self.ready.keys().copied())
            .chain(self.backend.running_page_ids())
            .collect::<Vec<_>>();
        pages.sort_unstable();
        pages.dedup();
        pages
    }

    pub(crate) fn ready_page_ids(&self) -> Vec<LodPageId> {
        self.ready.keys().copied().collect()
    }

    pub(crate) fn take_ready(&mut self, page: LodPageId) -> Option<LodPagePreprocessOutput> {
        let ready = self.ready.remove(&page)?;
        release_pending_bytes(&mut self.pending_bytes, ready.pending_bytes);
        Some(ready.output)
    }

    pub(crate) fn cancel(&mut self, page: LodPageId) -> bool {
        let mut cancelled = false;
        if let Some(index) = self
            .waiting
            .iter()
            .position(|waiting| waiting.input.request.page_id == page)
        {
            if let Some(waiting) = self.waiting.remove(index) {
                release_pending_bytes(&mut self.pending_bytes, waiting.pending_bytes);
            }
            cancelled = true;
        }
        if let Some(ready) = self.ready.remove(&page) {
            release_pending_bytes(&mut self.pending_bytes, ready.pending_bytes);
            cancelled = true;
        }
        if self.backend.cancel_running(page, &mut self.pending_bytes) {
            cancelled = true;
        }
        if cancelled {
            self.cancellations = self.cancellations.saturating_add(1);
        }
        cancelled
    }

    pub(crate) fn stats(&self) -> LodPagePreprocessStats {
        let (cooperative_decoded_gaussians, cooperative_total_gaussians) =
            self.backend.cooperative_progress();
        LodPagePreprocessStats {
            backend: self.backend.kind(),
            capacity: self.capacity.try_into().unwrap_or(u32::MAX),
            waiting: self.waiting.len().try_into().unwrap_or(u32::MAX),
            submitted: self.backend.tracked_len().try_into().unwrap_or(u32::MAX),
            ready: self.ready.len().try_into().unwrap_or(u32::MAX),
            byte_capacity: self.byte_capacity,
            pending_bytes: self.pending_bytes,
            deferred_admissions: self.deferred_admissions,
            cancellations: self.cancellations,
            cooperative_decoded_gaussians,
            cooperative_total_gaussians,
            cooperative_budget_gaussians_per_frame: self.backend.cooperative_budget(),
        }
    }

    fn validate_native_job_bytes(
        &self,
        pending_bytes: u64,
    ) -> Result<(), LodPagePreprocessAdmissionError> {
        if let Some(capacity) = self.backend.native_job_byte_capacity()
            && pending_bytes > capacity
        {
            return Err(
                LodPagePreprocessAdmissionError::NativeJobByteCapacityExceeded {
                    requested: pending_bytes,
                    capacity,
                },
            );
        }
        Ok(())
    }

    fn validate_capacity(
        capacity: usize,
        byte_capacity: u64,
    ) -> Result<(), LodPagePreprocessAdmissionError> {
        if capacity == 0 {
            return Err(LodPagePreprocessAdmissionError::ZeroCapacity);
        }
        if byte_capacity == 0 {
            return Err(LodPagePreprocessAdmissionError::ZeroByteCapacity);
        }
        Ok(())
    }

    fn with_backend(capacity: usize, byte_capacity: u64, backend: platform::BackendState) -> Self {
        Self {
            capacity,
            byte_capacity,
            pending_bytes: 0,
            waiting: VecDeque::new(),
            ready: BTreeMap::new(),
            backend,
            deferred_admissions: 0,
            cancellations: 0,
        }
    }
}

impl Drop for LodPagePreprocessor {
    fn drop(&mut self) {
        for page in self.page_ids() {
            self.cancel(page);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn process_input(input: LodPagePreprocessInput) -> LodPagePreprocessOutput {
    let request = input.request;
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| validate_and_decode(input)))
            .unwrap_or(Err(LodPagePreprocessError::WorkerPanicked));
    LodPagePreprocessOutput { request, result }
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_and_decode(
    input: LodPagePreprocessInput,
) -> Result<PlanarGaussian3dPage, LodPagePreprocessError> {
    validate_input_envelope(&input)?;
    if !input.payload.verify() {
        return Err(LodPagePreprocessError::PayloadChecksumMismatch);
    }
    let page = decode_page_with_descriptor(&input.payload.bytes, &input.descriptor, input.limits)
        .map_err(LodPagePreprocessError::Codec)?;
    validate_decoded_page_bounds(&page, &input.descriptor, input.support_sigma)?;
    Ok(page)
}

pub(super) fn validate_input_envelope(
    input: &LodPagePreprocessInput,
) -> Result<(), LodPagePreprocessError> {
    if input.payload.page_id != input.request.page_id {
        return Err(LodPagePreprocessError::PayloadPageMismatch {
            expected: input.request.page_id,
            actual: input.payload.page_id,
        });
    }
    let encoded_len = input.payload.bytes.len() as u64;
    if encoded_len > input.max_encoded_page_bytes {
        return Err(LodPagePreprocessError::EncodedPageLimitExceeded {
            actual: encoded_len,
            limit: input.max_encoded_page_bytes,
        });
    }
    Ok(())
}

#[cfg(any(test, not(target_arch = "wasm32")))]
pub(crate) fn validate_decoded_page_bounds(
    page: &PlanarGaussian3dPage,
    descriptor: &LodPageDescriptor,
    support_sigma: f32,
) -> Result<(), LodPagePreprocessError> {
    let mut actual_bounds: Option<LodBounds> = None;
    extend_decoded_page_bounds(
        &mut actual_bounds,
        &page.gaussians,
        descriptor.id,
        support_sigma,
    )?;
    validate_accumulated_page_bounds(actual_bounds, descriptor)
}

pub(super) fn extend_decoded_page_bounds(
    actual_bounds: &mut Option<LodBounds>,
    gaussians: &[Gaussian3d],
    page: LodPageId,
    support_sigma: f32,
) -> Result<(), LodPagePreprocessError> {
    for gaussian in gaussians {
        let bounds = gaussian_support_bounds(gaussian, support_sigma)
            .map_err(|_| LodPagePreprocessError::InvalidSupportBounds(page))?;
        *actual_bounds = Some(match *actual_bounds {
            Some(current) => current.union(bounds),
            None => bounds,
        });
    }
    Ok(())
}

pub(super) fn validate_accumulated_page_bounds(
    actual_bounds: Option<LodBounds>,
    descriptor: &LodPageDescriptor,
) -> Result<(), LodPagePreprocessError> {
    let actual_bounds =
        actual_bounds.ok_or(LodPagePreprocessError::InvalidSupportBounds(descriptor.id))?;
    let epsilon = 1e-5
        * descriptor
            .bounds
            .radius()
            .max(actual_bounds.radius())
            .max(1.0);
    if descriptor
        .bounds
        .contains_with_epsilon(&actual_bounds, epsilon)
    {
        Ok(())
    } else {
        Err(LodPagePreprocessError::PayloadOutsideDescriptor(
            descriptor.id,
        ))
    }
}

pub(super) fn release_pending_bytes(pending_bytes: &mut u64, bytes: u64) {
    *pending_bytes = pending_bytes
        .checked_sub(bytes)
        .expect("preprocessing pending-byte accounting underflow");
}
