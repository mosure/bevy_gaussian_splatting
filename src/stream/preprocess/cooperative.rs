use std::{
    collections::{BTreeMap, VecDeque},
    num::NonZeroU32,
    panic::{AssertUnwindSafe, catch_unwind},
};

use crate::{
    gaussian::formats::planar_3d_chunked::{
        LodBounds, LodPageDescriptor, LodPageId, PlanarGaussian3dPage,
    },
    io::lod::{
        IncrementalLodPageDecoder, LodCodecLimits, LodPageDecodeProgress,
        MAX_ENCODED_PAGE_GAUSSIAN_BYTES,
    },
    stream::transport::{PageChecksum64, PagePayload, PageRequest},
};

use super::{
    LodPagePreprocessBackend, LodPagePreprocessError, LodPagePreprocessInput,
    LodPagePreprocessOutput, ReadyJob, WaitingJob, extend_decoded_page_bounds,
    release_pending_bytes, validate_accumulated_page_bounds, validate_input_envelope,
};

/// Record-bounded preprocessing used by browsers and deterministic native
/// tests. Exactly one job owns partially verified/decoded state at a time.
pub(super) struct CooperativeBackend {
    last_frame: Option<u64>,
    last_budget: u32,
    active: Option<CooperativeJob>,
}

impl CooperativeBackend {
    pub(super) fn new() -> Self {
        Self {
            last_frame: None,
            last_budget: 0,
            active: None,
        }
    }

    pub(super) fn kind(&self) -> LodPagePreprocessBackend {
        LodPagePreprocessBackend::CooperativeWasm
    }

    pub(super) fn advance(
        &mut self,
        frame_sequence: u64,
        budget: NonZeroU32,
        waiting: &mut VecDeque<WaitingJob>,
        ready: &mut BTreeMap<LodPageId, ReadyJob>,
    ) {
        if self.last_frame == Some(frame_sequence) {
            return;
        }
        self.last_budget = budget.get();
        if self.active.is_none() && waiting.is_empty() {
            return;
        }

        // Mark the frame before executing user-controlled data processing so a
        // caught panic cannot permit a second camera to run another slice.
        self.last_frame = Some(frame_sequence);

        if self.active.is_none() {
            let waiting = waiting
                .pop_front()
                .expect("cooperative work availability was checked above");
            let request = waiting.input.request;
            let pending_bytes = waiting.pending_bytes;
            match catch_unwind(AssertUnwindSafe(|| CooperativeJob::new(waiting))) {
                Ok(Ok(job)) => self.active = Some(job),
                Ok(Err(error)) => {
                    insert_ready(ready, request, pending_bytes, Err(error));
                    return;
                }
                Err(_) => {
                    insert_ready(
                        ready,
                        request,
                        pending_bytes,
                        Err(LodPagePreprocessError::WorkerPanicked),
                    );
                    return;
                }
            }
        }

        let active = self
            .active
            .as_mut()
            .expect("cooperative job was started above");
        let progress = catch_unwind(AssertUnwindSafe(|| active.advance(budget))).unwrap_or(
            CooperativeJobProgress::Complete(Err(LodPagePreprocessError::WorkerPanicked)),
        );
        if let CooperativeJobProgress::Complete(result) = progress {
            let completed = self
                .active
                .take()
                .expect("completed cooperative job remains active until publication");
            insert_ready(ready, completed.request, completed.pending_bytes, result);
        }
    }

    pub(super) fn cancel(&mut self, page: LodPageId, pending_bytes: &mut u64) -> bool {
        if self
            .active
            .as_ref()
            .is_none_or(|active| active.page() != page)
        {
            return false;
        }
        let active = self
            .active
            .take()
            .expect("the active cooperative page was checked above");
        release_pending_bytes(pending_bytes, active.pending_bytes);
        true
    }

    pub(super) fn contains(&self, page: LodPageId) -> bool {
        self.active
            .as_ref()
            .is_some_and(|active| active.page() == page)
    }

    pub(super) fn tracked_len(&self) -> usize {
        usize::from(self.active.is_some())
    }

    pub(super) fn page_ids(&self) -> Vec<LodPageId> {
        self.active
            .as_ref()
            .map(|active| vec![active.page()])
            .unwrap_or_default()
    }

    pub(super) fn progress(&self) -> (u32, u32) {
        self.active
            .as_ref()
            .map(CooperativeJob::progress)
            .unwrap_or_default()
    }

    pub(super) fn budget(&self) -> u32 {
        self.last_budget
    }
}

fn insert_ready(
    ready: &mut BTreeMap<LodPageId, ReadyJob>,
    request: PageRequest,
    pending_bytes: u64,
    result: Result<PlanarGaussian3dPage, LodPagePreprocessError>,
) {
    let previous = ready.insert(
        request.page_id,
        ReadyJob {
            output: LodPagePreprocessOutput { request, result },
            pending_bytes,
        },
    );
    debug_assert!(previous.is_none(), "a cooperative page became ready twice");
}

struct CooperativeJob {
    request: PageRequest,
    pending_bytes: u64,
    state: CooperativeJobState,
}

impl CooperativeJob {
    fn new(waiting: WaitingJob) -> Result<Self, LodPagePreprocessError> {
        validate_input_envelope(&waiting.input)?;
        let LodPagePreprocessInput {
            request,
            payload,
            descriptor,
            limits,
            max_encoded_page_bytes: _,
            support_sigma,
        } = waiting.input;
        let PagePayload {
            page_id: _,
            bytes,
            checksum: expected_checksum,
        } = payload;
        Ok(Self {
            request,
            pending_bytes: waiting.pending_bytes,
            state: CooperativeJobState::Checksum(ChecksumState {
                bytes,
                expected_checksum,
                offset: 0,
                checksum: PageChecksum64::new(),
                descriptor,
                limits,
                support_sigma,
            }),
        })
    }

    fn page(&self) -> LodPageId {
        self.request.page_id
    }

    fn progress(&self) -> (u32, u32) {
        match &self.state {
            CooperativeJobState::Checksum(state) => (0, state.descriptor.gaussian_count),
            CooperativeJobState::Decode(state) => {
                (state.decoder.decoded_count(), state.decoder.total_count())
            }
            CooperativeJobState::Poisoned => (0, 0),
        }
    }

    fn advance(&mut self, budget: NonZeroU32) -> CooperativeJobProgress {
        let state = std::mem::replace(&mut self.state, CooperativeJobState::Poisoned);
        match state {
            CooperativeJobState::Checksum(mut state) => {
                let raw_budget = usize::try_from(budget.get())
                    .unwrap_or(usize::MAX)
                    .saturating_mul(MAX_ENCODED_PAGE_GAUSSIAN_BYTES);
                let end = state
                    .offset
                    .saturating_add(raw_budget)
                    .min(state.bytes.len());
                state.checksum.update(&state.bytes[state.offset..end]);
                state.offset = end;
                if state.offset < state.bytes.len() {
                    self.state = CooperativeJobState::Checksum(state);
                    return CooperativeJobProgress::Pending;
                }
                if state.checksum.finish() != state.expected_checksum {
                    return CooperativeJobProgress::Complete(Err(
                        LodPagePreprocessError::PayloadChecksumMismatch,
                    ));
                }

                let decoder = match IncrementalLodPageDecoder::new(
                    state.bytes,
                    state.descriptor,
                    state.limits,
                ) {
                    Ok(decoder) => decoder,
                    Err(error) => {
                        return CooperativeJobProgress::Complete(Err(
                            LodPagePreprocessError::Codec(error),
                        ));
                    }
                };
                self.state = CooperativeJobState::Decode(DecodeState {
                    decoder,
                    support_sigma: state.support_sigma,
                    actual_bounds: None,
                    first_support_error: None,
                });
                CooperativeJobProgress::Pending
            }
            CooperativeJobState::Decode(mut state) => {
                let progress = match state.decoder.advance(budget) {
                    Ok(progress) => progress,
                    Err(error) => {
                        return CooperativeJobProgress::Complete(Err(
                            LodPagePreprocessError::Codec(error),
                        ));
                    }
                };
                match progress {
                    LodPageDecodeProgress::Pending { decoded_range } => {
                        if state.first_support_error.is_none()
                            && let Err(error) = extend_decoded_page_bounds(
                                &mut state.actual_bounds,
                                &state.decoder.decoded_gaussians()[decoded_range],
                                state.decoder.descriptor().id,
                                state.support_sigma,
                            )
                        {
                            state.first_support_error = Some(error);
                        }
                        self.state = CooperativeJobState::Decode(state);
                        CooperativeJobProgress::Pending
                    }
                    LodPageDecodeProgress::Complete {
                        page,
                        decoded_range,
                    } => {
                        if state.first_support_error.is_none()
                            && let Err(error) = extend_decoded_page_bounds(
                                &mut state.actual_bounds,
                                &page.gaussians[decoded_range],
                                state.decoder.descriptor().id,
                                state.support_sigma,
                            )
                        {
                            state.first_support_error = Some(error);
                        }
                        if let Some(error) = state.first_support_error {
                            return CooperativeJobProgress::Complete(Err(error));
                        }
                        match validate_accumulated_page_bounds(
                            state.actual_bounds,
                            state.decoder.descriptor(),
                        ) {
                            Ok(()) => CooperativeJobProgress::Complete(Ok(page)),
                            Err(error) => CooperativeJobProgress::Complete(Err(error)),
                        }
                    }
                }
            }
            CooperativeJobState::Poisoned => {
                unreachable!("a cooperative job cannot be advanced from a poisoned state")
            }
        }
    }
}

enum CooperativeJobState {
    Checksum(ChecksumState),
    Decode(DecodeState),
    /// A panic while advancing leaves no reusable partial state. The backend
    /// catches it, publishes `WorkerPanicked`, and drops the active job.
    Poisoned,
}

struct ChecksumState {
    bytes: Vec<u8>,
    expected_checksum: u64,
    offset: usize,
    checksum: PageChecksum64,
    descriptor: LodPageDescriptor,
    limits: LodCodecLimits,
    support_sigma: f32,
}

struct DecodeState {
    decoder: IncrementalLodPageDecoder,
    support_sigma: f32,
    actual_bounds: Option<LodBounds>,
    /// Bounds validation follows successful codec validation in the public
    /// error order, so retain the first failure while decoding later records.
    first_support_error: Option<LodPagePreprocessError>,
}

enum CooperativeJobProgress {
    Pending,
    Complete(Result<PlanarGaussian3dPage, LodPagePreprocessError>),
}
