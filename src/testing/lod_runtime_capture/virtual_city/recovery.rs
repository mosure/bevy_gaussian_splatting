//! Same-submission visible-work recovery, independent of the first proxy draw.

use super::*;

pub(super) const REQUIRED_SAMPLES: u32 = 10;

#[derive(Clone, Debug, Serialize)]
pub(super) struct DetailSample {
    stamp: LodCaptureStamp,
    path_frame: u64,
    camera: LodCaptureCamera,
    selected: u64,
    candidates: u64,
    compacted: u64,
    drawn: u64,
}

impl DetailSample {
    fn same_work(&self, other: &Self) -> bool {
        self.camera == other.camera
            && self.selected == other.selected
            && self.candidates == other.candidates
            && self.compacted == other.compacted
            && self.drawn == other.drawn
    }

    pub(super) fn from_capture(capture: &CompletedCapture) -> Option<Self> {
        let counts = &capture.record.counts;
        let drawn = counts.drawn?;
        let compacted = counts.compacted?;
        let proof = &capture.evidence;
        let path_frame = proof["path_frame"].as_u64()?;
        (drawn > 0
            && compacted == drawn
            && counts.stamp == capture.record.stamp
            && proof["draw_command_attested"] == true
            && proof["complete_package_frontier"] == true
            && proof["source"] == "post_render_same_submission_copy"
            && proof["frame"].as_u64() == Some(counts.stamp.frame)
            && proof["compaction_generation"].as_u64() == Some(counts.stamp.generation)
            && proof["indirect"]["instance_count"].as_u64() == Some(drawn)
            && proof["indirect"]["overflow_count"].as_u64() == Some(0))
        .then(|| Self {
            stamp: counts.stamp.clone(),
            path_frame,
            camera: capture.record.camera.clone(),
            selected: counts.selected,
            candidates: counts.candidates,
            compacted,
            drawn,
        })
    }
}

#[derive(Default, Serialize)]
pub(super) struct DetailRecovery {
    stationary_reference: Option<DetailSample>,
    first_nonzero_return: Option<DetailSample>,
    restored_window_start: Option<DetailSample>,
    restored_window_end: Option<DetailSample>,
    #[serde(skip)]
    previous: Option<DetailSample>,
    #[serde(skip)]
    stationary_samples: u32,
    #[serde(skip)]
    returned_samples: u32,
}

impl DetailRecovery {
    pub(super) fn observe(&mut self, scenario: &str, sample: Option<DetailSample>, cadence: u32) {
        if !matches!(scenario, "stationary" | "rapid_return") {
            return;
        }
        if scenario == "rapid_return" && self.recovered() {
            return;
        }
        let consecutive = sample.as_ref().is_some_and(|sample| {
            self.previous.as_ref().is_some_and(|previous| {
                sample.stamp.frame == previous.stamp.frame + u64::from(cadence)
                    && sample.same_work(previous)
            })
        });
        if scenario == "stationary" {
            self.stationary_samples = if sample.is_none() {
                0
            } else if consecutive {
                self.stationary_samples.saturating_add(1)
            } else {
                1
            };
            self.stationary_reference = (self.stationary_samples >= REQUIRED_SAMPLES)
                .then(|| sample.clone())
                .flatten();
        } else {
            if self.first_nonzero_return.is_none() {
                self.first_nonzero_return = sample.clone();
            }
            let matches_reference = sample.as_ref().is_some_and(|sample| {
                self.stationary_reference
                    .as_ref()
                    .is_some_and(|reference| sample.same_work(reference))
            });
            self.returned_samples = if !matches_reference {
                0
            } else if consecutive {
                self.returned_samples.saturating_add(1)
            } else {
                1
            };
            if self.returned_samples <= 1 {
                self.restored_window_start = matches_reference.then(|| sample.clone()).flatten();
            }
            self.restored_window_end = (self.returned_samples >= REQUIRED_SAMPLES)
                .then(|| sample.clone())
                .flatten();
        }
        self.previous = sample;
    }

    pub(super) fn stationary_ready(&self) -> bool {
        self.stationary_reference.is_some()
    }

    pub(super) fn recovered(&self) -> bool {
        self.restored_window_end.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(frame: u64, drawn: u64) -> DetailSample {
        DetailSample {
            stamp: LodCaptureStamp {
                run_id: "test".into(),
                view_id: "1".into(),
                frame,
                generation: 1,
            },
            path_frame: frame,
            camera: LodCaptureCamera {
                world_to_view: [0.0; 16],
                projection: [0.0; 16],
                viewport: [1, 1],
                pixel_scale: 1.0,
            },
            selected: 200,
            candidates: 200,
            compacted: drawn,
            drawn,
        }
    }

    #[test]
    fn return_requires_consecutive_reference_work_after_first_proxy_draw() {
        let mut recovery = DetailRecovery::default();
        for frame in 0..10 {
            recovery.observe("stationary", Some(sample(frame, 100)), 1);
        }
        assert!(recovery.stationary_ready());
        recovery.observe("rapid_return", Some(sample(100, 1)), 1);
        assert!(!recovery.recovered());
        for frame in 101..110 {
            recovery.observe("rapid_return", Some(sample(frame, 100)), 1);
        }
        assert!(!recovery.recovered());
        recovery.observe("rapid_return", None, 1);
        for frame in 111..121 {
            recovery.observe("rapid_return", Some(sample(frame, 100)), 1);
        }
        assert!(recovery.recovered());
        assert_eq!(recovery.first_nonzero_return.unwrap().drawn, 1);
        assert_eq!(recovery.restored_window_start.unwrap().stamp.frame, 111);
        assert_eq!(recovery.restored_window_end.unwrap().stamp.frame, 120);
    }
}
