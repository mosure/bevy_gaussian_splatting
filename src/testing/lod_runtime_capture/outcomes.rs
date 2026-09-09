//! Bounded accounting for capture requests, separate from image/draw evidence.

use std::{collections::BTreeMap, io::Write};

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum Outcome {
    Completed {
        image_written: bool,
        draw_attested: bool,
    },
    MissingDrawable {
        reason: String,
    },
    RingFull,
    MappingFailure {
        error: String,
    },
    DeliveryFailure {
        error: String,
    },
    WriteFailure {
        error: String,
    },
    CaptureFailure {
        error: String,
    },
    Timeout,
    Unsubmitted,
    Aborted,
}

#[derive(Serialize)]
struct Request {
    frame: u64,
    path_frame: u64,
    scenario: String,
    readback_encoded: bool,
    outcome: Option<Outcome>,
}

pub(super) struct RequestOutcomes {
    maximum_requests: usize,
    requests: BTreeMap<u64, Request>,
    accounting_errors: u64,
    unresolved_at_shutdown: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct Summary {
    pub requested: usize,
    pub terminal: usize,
    pub completed_captures: usize,
    pub attested_captures: usize,
    pub unresolved: usize,
    pub unresolved_at_shutdown: usize,
    pub accounting_errors: u64,
    pub outcomes: BTreeMap<&'static str, usize>,
}

impl RequestOutcomes {
    pub fn new(maximum_requests: usize) -> Self {
        Self {
            maximum_requests,
            requests: BTreeMap::new(),
            accounting_errors: 0,
            unresolved_at_shutdown: 0,
        }
    }

    pub fn register(
        &mut self,
        frame: u64,
        path_frame: u64,
        scenario: &str,
    ) -> Result<(), &'static str> {
        if self.requests.len() >= self.maximum_requests || self.requests.contains_key(&frame) {
            self.accounting_errors += 1;
            return Err("capture request accounting capacity exceeded or duplicate frame");
        }
        self.requests.insert(
            frame,
            Request {
                frame,
                path_frame,
                scenario: scenario.to_owned(),
                readback_encoded: false,
                outcome: None,
            },
        );
        Ok(())
    }

    pub fn can_encode(&self, frame: u64) -> bool {
        self.requests
            .get(&frame)
            .is_some_and(|request| !request.readback_encoded && request.outcome.is_none())
    }

    pub fn encoded(&mut self, frame: u64) -> Result<(), &'static str> {
        if !self.can_encode(frame) {
            self.accounting_errors += 1;
            return Err("capture encoded an unknown, terminal, or already encoded request");
        }
        self.requests.get_mut(&frame).unwrap().readback_encoded = true;
        Ok(())
    }

    pub fn finish(&mut self, frame: u64, outcome: Outcome) -> Result<(), &'static str> {
        match self.requests.get_mut(&frame) {
            Some(request) if request.outcome.is_none() => {
                request.outcome = Some(outcome);
                Ok(())
            }
            _ => {
                self.accounting_errors += 1;
                Err("capture received a duplicate or unknown terminal request outcome")
            }
        }
    }

    pub fn finish_pending(&mut self, timed_out: bool) {
        for request in self
            .requests
            .values_mut()
            .filter(|request| request.outcome.is_none())
        {
            self.unresolved_at_shutdown += 1;
            request.outcome = Some(if timed_out {
                Outcome::Timeout
            } else if request.readback_encoded {
                Outcome::Aborted
            } else {
                Outcome::Unsubmitted
            });
        }
    }

    pub fn summary(&self) -> Summary {
        let mut result = Summary {
            requested: self.requests.len(),
            terminal: 0,
            completed_captures: 0,
            attested_captures: 0,
            unresolved: 0,
            unresolved_at_shutdown: self.unresolved_at_shutdown,
            accounting_errors: self.accounting_errors,
            outcomes: BTreeMap::new(),
        };
        for request in self.requests.values() {
            let Some(outcome) = &request.outcome else {
                result.unresolved += 1;
                continue;
            };
            result.terminal += 1;
            let kind = match outcome {
                Outcome::Completed { draw_attested, .. } => {
                    result.completed_captures += 1;
                    result.attested_captures += usize::from(*draw_attested);
                    "completed"
                }
                Outcome::MissingDrawable { .. } => "missing_drawable",
                Outcome::RingFull => "ring_full",
                Outcome::MappingFailure { .. } => "mapping_failure",
                Outcome::DeliveryFailure { .. } => "delivery_failure",
                Outcome::WriteFailure { .. } => "write_failure",
                Outcome::CaptureFailure { .. } => "capture_failure",
                Outcome::Timeout => "timeout",
                Outcome::Unsubmitted => "unsubmitted",
                Outcome::Aborted => "aborted",
            };
            *result.outcomes.entry(kind).or_default() += 1;
        }
        result
    }

    pub fn write(&self, mut output: impl Write, run_id: &str) -> std::io::Result<()> {
        for request in self.requests.values() {
            #[derive(Serialize)]
            struct Row<'a> {
                schema_version: u32,
                run_id: &'a str,
                #[serde(flatten)]
                request: &'a Request,
            }
            serde_json::to_writer(
                &mut output,
                &Row {
                    schema_version: 1,
                    run_id,
                    request,
                },
            )
            .map_err(std::io::Error::other)?;
            output.write_all(b"\n")?;
        }
        output.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_outcomes_preserve_skips_failures_and_exactly_once_completion() {
        let mut tracker = RequestOutcomes::new(6);
        for frame in 0..6 {
            tracker.register(frame, frame * 2, "route").unwrap();
        }
        tracker.encoded(0).unwrap();
        tracker
            .finish(
                0,
                Outcome::Completed {
                    image_written: true,
                    draw_attested: true,
                },
            )
            .unwrap();
        tracker
            .finish(
                1,
                Outcome::MissingDrawable {
                    reason: "no receipt".into(),
                },
            )
            .unwrap();
        tracker.finish(2, Outcome::RingFull).unwrap();
        tracker.encoded(3).unwrap();
        tracker
            .finish(
                3,
                Outcome::MappingFailure {
                    error: "device mapping failed".into(),
                },
            )
            .unwrap();
        tracker.encoded(4).unwrap();
        assert_eq!(tracker.summary().unresolved, 2);
        tracker.finish_pending(false);
        let summary = tracker.summary();
        assert_eq!(summary.terminal, 6);
        assert_eq!(summary.completed_captures, 1);
        assert_eq!(summary.attested_captures, 1);
        assert_eq!(summary.unresolved_at_shutdown, 2);
        assert_eq!(summary.outcomes["aborted"], 1);
        assert_eq!(summary.outcomes["unsubmitted"], 1);
        assert!(tracker.finish(0, Outcome::RingFull).is_err());
        assert!(tracker.register(6, 12, "route").is_err());
        assert_eq!(tracker.summary().accounting_errors, 2);
        let mut bytes = Vec::new();
        tracker.write(&mut bytes, "run").unwrap();
        let rows = String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 6);
        assert_eq!(rows[0]["outcome"]["kind"], "completed");
        assert_eq!(rows[5]["path_frame"], 10);
        let mut timeout = RequestOutcomes::new(1);
        timeout.register(0, 0, "cold").unwrap();
        timeout.finish_pending(true);
        assert_eq!(timeout.summary().outcomes["timeout"], 1);
    }
}
