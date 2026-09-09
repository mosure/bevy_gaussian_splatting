use std::{
    collections::{BTreeMap, VecDeque},
    num::NonZeroU32,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use crate::{
    gaussian::formats::planar_3d_chunked::{LodPageDescriptor, LodPageId, PlanarGaussian3dPage},
    io::lod::{
        IncrementalLodPageDecoder, LodCodecLimits, LodPageDecodeProgress,
        MAX_ENCODED_PAGE_GAUSSIAN_BYTES,
    },
    stream::transport::{PageChecksum64, PagePayload, PageRequest},
};

use super::{
    DecodedPageBounds, LodPagePreprocessBackend, LodPagePreprocessError, LodPagePreprocessInput,
    LodPagePreprocessOutput, ReadyJob, SharedPageNodeRange, WaitingJob, release_pending_bytes,
    validate_input_envelope,
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
            node_ranges,
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
                node_ranges,
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
                    bounds: DecodedPageBounds::new(state.node_ranges),
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
                            && let Err(error) = state.bounds.extend(
                                &state.decoder.decoded_gaussians()[decoded_range],
                                state.decoder.descriptor(),
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
                            && let Err(error) = state.bounds.extend(
                                &page.gaussians[decoded_range],
                                state.decoder.descriptor(),
                                state.support_sigma,
                            )
                        {
                            state.first_support_error = Some(error);
                        }
                        if let Some(error) = state.first_support_error {
                            return CooperativeJobProgress::Complete(Err(error));
                        }
                        match state.bounds.finish(state.decoder.descriptor()) {
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
    node_ranges: Option<Arc<Vec<SharedPageNodeRange>>>,
}

struct DecodeState {
    decoder: IncrementalLodPageDecoder,
    support_sigma: f32,
    bounds: DecodedPageBounds,
    /// Bounds validation follows successful codec validation in the public
    /// error order, so retain the first failure while decoding later records.
    first_support_error: Option<LodPagePreprocessError>,
}

enum CooperativeJobProgress {
    Pending,
    Complete(Result<PlanarGaussian3dPage, LodPagePreprocessError>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gaussian::formats::{
            planar_3d::Gaussian3d,
            planar_3d_chunked::{LodNodeId, LodPageEncoding, LodPageKind, LodPageRange},
            planar_3d_lod::gaussian_support_bounds,
        },
        io::lod::encode_page,
        stream::transport::PageRequestPriority,
    };

    fn shared_page_input() -> (PlanarGaussian3dPage, LodPagePreprocessInput) {
        let page_id = LodPageId(7);
        let support_sigma = 3.0;
        let page = PlanarGaussian3dPage::new(
            page_id,
            [-1.0, -0.9, -0.8, 1.0, 1.1]
                .into_iter()
                .map(|x| Gaussian3d {
                    position_visibility: [x, 0.0, 0.0, 1.0].into(),
                    rotation: [1.0, 0.0, 0.0, 0.0].into(),
                    scale_opacity: [0.05, 0.06, 0.07, 0.8].into(),
                    ..Gaussian3d::default()
                })
                .collect(),
        );
        let bounds = |start: usize, end: usize| {
            page.gaussians[start..end]
                .iter()
                .map(|gaussian| gaussian_support_bounds(gaussian, support_sigma).unwrap())
                .reduce(|current, bounds| current.union(bounds))
                .unwrap()
        };
        let descriptor = LodPageDescriptor {
            id: page_id,
            kind: LodPageKind::SourceLeaves,
            encoding: LodPageEncoding::F32Planar,
            gaussian_count: page.gaussians.len() as u32,
            decoded_len: (page.gaussians.len() * std::mem::size_of::<Gaussian3d>()) as u64,
            content_hash: page.content_hash(),
            bounds: bounds(0, 5),
            storage: None,
        };
        let node_ranges = Some(Arc::new(vec![
            SharedPageNodeRange {
                node: LodNodeId(10),
                range: LodPageRange {
                    page: page_id,
                    offset: 0,
                    count: 3,
                },
                bounds: bounds(0, 3),
            },
            SharedPageNodeRange {
                node: LodNodeId(11),
                range: LodPageRange {
                    page: page_id,
                    offset: 3,
                    count: 2,
                },
                bounds: bounds(3, 5),
            },
        ]));
        let encoded = encode_page(&page).unwrap();
        let encoded_len = encoded.len() as u64;
        let input = LodPagePreprocessInput {
            request: PageRequest::new(page_id, PageRequestPriority::visible(1)),
            payload: PagePayload::new(page_id, encoded),
            descriptor,
            limits: LodCodecLimits {
                max_page_bytes: encoded_len,
                ..Default::default()
            },
            max_encoded_page_bytes: encoded_len,
            support_sigma,
            node_ranges,
        };
        (page, input)
    }

    fn run_cooperative(
        input: LodPagePreprocessInput,
        budget: NonZeroU32,
    ) -> Result<PlanarGaussian3dPage, LodPagePreprocessError> {
        let pending_bytes = input.pending_bytes().unwrap();
        let mut job = CooperativeJob::new(WaitingJob {
            input,
            pending_bytes,
        })
        .unwrap();
        for _ in 0..100 {
            if let CooperativeJobProgress::Complete(result) = job.advance(budget) {
                return result;
            }
        }
        panic!("the bounded fixture must complete");
    }

    #[test]
    fn shared_page_bounds_follow_decode_slices_without_completion_tail() {
        let (page, input) = shared_page_input();
        let pending_bytes = input.pending_bytes().unwrap();
        let page_id = input.request.page_id;
        let mut waiting = VecDeque::from([WaitingJob {
            input,
            pending_bytes,
        }]);
        let mut ready = BTreeMap::new();
        let mut backend = CooperativeBackend::new();
        let budget = NonZeroU32::new(2).unwrap();
        let mut previous_count = 0;
        let mut decode_slices = 0;
        for frame in 1..100 {
            backend.advance(frame, budget, &mut waiting, &mut ready);
            if let Some(completed) = ready.remove(&page_id) {
                // Only the last record remained. Its node and page checks
                // completed in the very same slice that decoded it.
                assert_eq!(previous_count, 4);
                assert_eq!(decode_slices, 2);
                assert_eq!(completed.output.result.unwrap(), page);
                assert!(backend.active.is_none());
                return;
            }
            let Some(CooperativeJob {
                state: CooperativeJobState::Decode(state),
                ..
            }) = backend.active.as_ref()
            else {
                continue;
            };
            let processed = state.bounds.processed_gaussians;
            assert_eq!(processed, state.decoder.decoded_count());
            assert!(processed - previous_count <= budget.get());
            if processed != previous_count {
                decode_slices += 1;
            }
            assert_eq!(state.bounds.node_index, usize::from(processed >= 3));
            previous_count = processed;
            backend.advance(frame, budget, &mut waiting, &mut ready);
            let CooperativeJobState::Decode(state) = &backend.active.as_ref().unwrap().state else {
                panic!("same frame must retain the current decode state")
            };
            assert_eq!(state.bounds.processed_gaussians, processed);
        }
        panic!("the bounded fixture must complete");
    }

    #[test]
    fn shared_page_preprocessing_accepts_ranges_across_multiple_slices() {
        for budget in [1, 2, 3, 5, 100] {
            let (page, input) = shared_page_input();
            assert_eq!(
                run_cooperative(input, NonZeroU32::new(budget).unwrap()),
                Ok(page)
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (page, input) = shared_page_input();
            assert_eq!(super::super::process_input(input).result, Ok(page));
        }
    }

    #[test]
    fn shared_page_bounds_failure_prevents_preprocessor_publication_on_both_backends() {
        for failing_node in 0..2 {
            for native in [false, true] {
                let (_, mut input) = shared_page_input();
                let ranges = Arc::make_mut(input.node_ranges.as_mut().unwrap());
                ranges[failing_node].bounds = ranges[1 - failing_node].bounds;
                let expected = Err(LodPagePreprocessError::PayloadOutsideNodeBounds {
                    page: input.request.page_id,
                    node: ranges[failing_node].node,
                });
                if native {
                    #[cfg(not(target_arch = "wasm32"))]
                    assert_eq!(super::super::process_input(input).result, expected);
                } else {
                    assert_eq!(run_cooperative(input, NonZeroU32::MIN), expected);
                }
            }
        }
    }

    #[test]
    fn shared_page_malformed_ranges_fail_before_residency() {
        for malformed in 0..6 {
            let (_, mut input) = shared_page_input();
            let ranges = Arc::make_mut(input.node_ranges.as_mut().unwrap());
            match malformed {
                0 => ranges[1].range.offset = 2, // overlap
                1 => ranges[1].range.offset = 4, // gap
                2 => ranges[1].range.count = 0,
                3 => ranges[1].range.count = u32::MAX,
                4 => ranges[1].range.page = LodPageId(999),
                5 => ranges[1].range.count = 3, // outside decoded page
                _ => unreachable!(),
            }
            assert_eq!(
                run_cooperative(input, NonZeroU32::MIN),
                Err(LodPagePreprocessError::PayloadOutsideNodeBounds {
                    page: LodPageId(7),
                    node: LodNodeId(11),
                })
            );
        }
    }

    #[test]
    fn shared_page_descriptor_error_retains_precedence_over_node_failure() {
        let (_, mut input) = shared_page_input();
        let ranges = Arc::make_mut(input.node_ranges.as_mut().unwrap());
        input.descriptor.bounds = ranges[1].bounds;
        ranges[0].bounds = ranges[1].bounds;
        assert_eq!(
            run_cooperative(input, NonZeroU32::MIN),
            Err(LodPagePreprocessError::PayloadOutsideDescriptor(LodPageId(
                7
            )))
        );
    }
}
