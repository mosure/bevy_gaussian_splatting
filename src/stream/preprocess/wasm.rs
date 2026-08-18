use std::{
    collections::{BTreeMap, VecDeque},
    num::NonZeroU32,
};

use crate::gaussian::formats::planar_3d_chunked::LodPageId;

use super::{LodPagePreprocessBackend, ReadyJob, WaitingJob, cooperative::CooperativeBackend};

/// Browser platform adapter. Keeping it behind the module boundary means the
/// common bounded-state implementation has no target-specific fields or
/// branches.
pub(super) struct BackendState(CooperativeBackend);

impl BackendState {
    pub(super) fn new() -> Result<Self, String> {
        Ok(Self(CooperativeBackend::new()))
    }

    #[cfg(test)]
    pub(super) fn new_cooperative_for_tests() -> Self {
        Self(CooperativeBackend::new())
    }

    pub(super) fn kind(&self) -> LodPagePreprocessBackend {
        self.0.kind()
    }

    pub(super) fn advance(
        &mut self,
        frame_sequence: u64,
        cooperative_budget: NonZeroU32,
        waiting: &mut VecDeque<WaitingJob>,
        ready: &mut BTreeMap<LodPageId, ReadyJob>,
        _pending_bytes: &mut u64,
        _deferred_admissions: &mut u64,
    ) {
        self.0
            .advance(frame_sequence, cooperative_budget, waiting, ready);
    }

    pub(super) fn cancel_running(&mut self, page: LodPageId, pending_bytes: &mut u64) -> bool {
        self.0.cancel(page, pending_bytes)
    }

    pub(super) fn is_running(&self, page: LodPageId) -> bool {
        self.0.contains(page)
    }

    pub(super) fn tracked_len(&self) -> usize {
        self.0.tracked_len()
    }

    pub(super) fn running_page_ids(&self) -> Vec<LodPageId> {
        self.0.page_ids()
    }

    pub(super) fn cooperative_progress(&self) -> (u32, u32) {
        self.0.progress()
    }

    pub(super) fn cooperative_budget(&self) -> u32 {
        self.0.budget()
    }

    pub(super) fn native_job_byte_capacity(&self) -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::{mem::size_of, num::NonZeroU32};

    use wasm_bindgen_test::wasm_bindgen_test;

    use super::super::{LodPagePreprocessInput, LodPagePreprocessor};
    use crate::{
        gaussian::formats::{
            planar_3d::Gaussian3d,
            planar_3d_chunked::{
                LodPageDescriptor, LodPageEncoding, LodPageId, LodPageKind, PlanarGaussian3dPage,
            },
            planar_3d_lod::gaussian_support_bounds,
        },
        io::lod::{LodCodecLimits, encode_page},
        stream::transport::{PagePayload, PageRequest, PageRequestPriority},
    };

    #[wasm_bindgen_test]
    fn cooperative_decoder_is_record_bounded_across_application_frames() {
        let page_id = LodPageId(7);
        let support_sigma = 3.0;
        let gaussians = (0..3)
            .map(|index| {
                let x = index as f32 * 0.25;
                Gaussian3d {
                    position_visibility: [x, 0.0, 0.0, 1.0].into(),
                    rotation: [1.0, 0.0, 0.0, 0.0].into(),
                    scale_opacity: [0.05, 0.06, 0.07, 0.8].into(),
                    ..Gaussian3d::default()
                }
            })
            .collect::<Vec<_>>();
        let page = PlanarGaussian3dPage::new(page_id, gaussians);
        let bounds = page
            .gaussians
            .iter()
            .map(|gaussian| gaussian_support_bounds(gaussian, support_sigma).unwrap())
            .reduce(|current, bounds| current.union(bounds))
            .unwrap();
        let descriptor = LodPageDescriptor {
            id: page_id,
            kind: LodPageKind::SourceLeaves,
            encoding: LodPageEncoding::F32Planar,
            gaussian_count: page.gaussians.len() as u32,
            decoded_len: (page.gaussians.len() * size_of::<Gaussian3d>()) as u64,
            content_hash: page.content_hash(),
            bounds,
            storage: None,
        };
        let encoded = encode_page(&page).unwrap();
        let encoded_len = encoded.len() as u64;
        let mut request = PageRequest::new(page_id, PageRequestPriority::visible(1));
        request.expected_bytes = Some(encoded_len);
        let input = LodPagePreprocessInput {
            request,
            payload: PagePayload::new(page_id, encoded),
            descriptor,
            limits: LodCodecLimits {
                max_page_bytes: encoded_len,
                ..Default::default()
            },
            max_encoded_page_bytes: encoded_len,
            support_sigma,
        };
        let pending_bytes = input.pending_bytes().unwrap();
        let mut preprocessor =
            LodPagePreprocessor::new_cooperative_with_byte_capacity_for_tests(1, pending_bytes)
                .unwrap();
        preprocessor.submit(input).unwrap();
        let budget = NonZeroU32::new(1).unwrap();

        preprocessor.advance(11, budget);
        let first_slice = preprocessor.stats();
        assert_eq!(first_slice.waiting, 0);
        assert_eq!(first_slice.submitted, 1);
        assert_eq!(first_slice.ready, 0);
        assert_eq!(first_slice.cooperative_decoded_gaussians, 0);
        assert_eq!(first_slice.cooperative_total_gaussians, 3);
        assert_eq!(first_slice.cooperative_budget_gaussians_per_frame, 1);
        assert_eq!(first_slice.pending_bytes, pending_bytes);

        preprocessor.advance(11, budget);
        assert_eq!(preprocessor.stats(), first_slice);

        let mut completed_frame = None;
        for frame in 12..32 {
            preprocessor.advance(frame, budget);
            let stats = preprocessor.stats();
            assert!(stats.cooperative_decoded_gaussians <= 3);
            if stats.ready == 1 {
                completed_frame = Some(frame);
                break;
            }
        }
        assert!(completed_frame.is_some_and(|frame| frame > 12));
        assert_eq!(preprocessor.stats().pending_bytes, pending_bytes);

        let output = preprocessor.take_ready(page_id).unwrap();
        assert_eq!(output.result.unwrap(), page);
        assert_eq!(preprocessor.stats().pending_bytes, 0);
        assert_eq!(preprocessor.len(), 0);
    }
}
