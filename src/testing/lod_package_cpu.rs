//! Opt-in wall-clock measurements of named package orchestration functions.
//!
//! These scopes measure caller-thread elapsed time, including waits inside the
//! named function. They do not measure worker CPU, total renderer CPU, or GPU
//! execution. Nested scopes overlap and must not be summed. No clock is read
//! without an explicitly installed collector and an active package update.
//!
//! The collector retains one completed update. Capture extraction consumes it
//! once and attaches the current submission's frame identity. An absent sample
//! means unobserved work, never a measured zero. No history grows with frames,
//! cameras, nodes, or source Gaussian count.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use bevy::{platform::time::Instant, prelude::Resource};

const SCOPE_COUNT: usize = 14;

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub(crate) enum PackageCpuScope {
    UpdateSystem,
    DestinationPlan,
    CanonicalSelectionMiss,
    DestinationCompile,
    DiscreteWavePlan,
    DiscreteResidentPlan,
    PublishStagedCut,
    /// Transport polling and bounded preprocessor advancement; worker execution
    /// outside this call is excluded.
    RuntimePollPages,
    /// Ready-result admission, decoded-cache insertion/eviction, and pin updates.
    RuntimeCommitPreprocessedPages,
    /// Full package_target_candidates call, including nested destination lookup.
    TargetCandidates,
    /// Canonical CPU slot materialization/enqueue and staged-range validation.
    AdvanceStagedCut,
    /// Snapshot retirement, current GPU feedback, and normalized page demand.
    GpuFeedback,
    /// Selector-free runtime page admission, polling, and request starts.
    GpuPageDemand,
    /// Resident slot staging, immutable snapshot publication, and shared pins.
    GpuPublication,
}

impl PackageCpuScope {
    const ALL: [Self; SCOPE_COUNT] = [
        Self::UpdateSystem,
        Self::DestinationPlan,
        Self::CanonicalSelectionMiss,
        Self::DestinationCompile,
        Self::DiscreteWavePlan,
        Self::DiscreteResidentPlan,
        Self::PublishStagedCut,
        Self::RuntimePollPages,
        Self::RuntimeCommitPreprocessedPages,
        Self::TargetCandidates,
        Self::AdvanceStagedCut,
        Self::GpuFeedback,
        Self::GpuPageDemand,
        Self::GpuPublication,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::UpdateSystem => "lod_package_update_system",
            Self::DestinationPlan => "lod_package_destination_plan",
            Self::CanonicalSelectionMiss => "lod_package_canonical_selection_miss",
            Self::DestinationCompile => "lod_package_destination_compile",
            Self::DiscreteWavePlan => "lod_package_discrete_wave_plan",
            Self::DiscreteResidentPlan => "lod_package_discrete_resident_plan",
            Self::PublishStagedCut => "lod_package_publish_staged_cut",
            Self::RuntimePollPages => "lod_package_runtime_poll_pages",
            Self::RuntimeCommitPreprocessedPages => "lod_package_runtime_commit_preprocessed_pages",
            Self::TargetCandidates => "lod_package_target_candidates",
            Self::AdvanceStagedCut => "lod_package_advance_staged_cut",
            Self::GpuFeedback => "lod_package_gpu_feedback",
            Self::GpuPageDemand => "lod_package_gpu_page_demand",
            Self::GpuPublication => "lod_package_gpu_publication",
        }
    }
}

/// One completed package update, consumed once through [`LodPackageCpuTelemetry`].
#[derive(Clone, Debug, Default)]
pub struct PackageCpuSample {
    /// Per-function caller-thread elapsed milliseconds. Names denote the exact
    /// instrumented scope; some are nested inside others.
    pub cpu_ms: BTreeMap<String, f64>,
    pub calls: BTreeMap<String, u64>,
    pub destination_cache_hits: u64,
    pub destination_compilations: u64,
    pub canonical_visited_nodes: u64,
}

#[derive(Default)]
struct WorkingSample {
    elapsed: [Duration; SCOPE_COUNT],
    calls: [u64; SCOPE_COUNT],
    destination_cache_hits: u64,
    destination_compilations: u64,
    canonical_visited_nodes: u64,
}

impl WorkingSample {
    fn observe(&mut self, scope: PackageCpuScope, duration: Duration) {
        let index = scope as usize;
        self.elapsed[index] = self.elapsed[index].saturating_add(duration);
        self.calls[index] = self.calls[index].saturating_add(1);
    }

    fn snapshot(&self) -> PackageCpuSample {
        let mut sample = PackageCpuSample {
            destination_cache_hits: self.destination_cache_hits,
            destination_compilations: self.destination_compilations,
            canonical_visited_nodes: self.canonical_visited_nodes,
            ..Default::default()
        };
        for scope in PackageCpuScope::ALL {
            let index = scope as usize;
            if self.calls[index] != 0 {
                sample.cpu_ms.insert(
                    scope.name().into(),
                    self.elapsed[index].as_secs_f64() * 1000.0,
                );
                sample.calls.insert(scope.name().into(), self.calls[index]);
            }
        }
        sample
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Arc<Mutex<WorkingSample>>>> = const { RefCell::new(None) };
}

/// Installing this resource explicitly opts the package system into CPU timing.
/// Removing it disables subsequent updates. Completed samples are bounded and
/// consumed once by the caller; this resource starts empty.
#[derive(Resource, Clone, Default)]
pub struct LodPackageCpuTelemetry {
    completed: Arc<Mutex<Option<PackageCpuSample>>>,
}

impl LodPackageCpuTelemetry {
    pub(crate) fn begin_update(&self) -> Option<PackageCpuUpdate> {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            // A nested invocation cannot claim the outer update's samples.
            if active.is_some() {
                return None;
            }
            let sample = Arc::new(Mutex::new(WorkingSample::default()));
            *active = Some(sample.clone());
            Some(PackageCpuUpdate {
                collector: self.clone(),
                sample,
                started: Instant::now(),
            })
        })
    }

    /// Consume the latest update. Absence means unobserved work, not zero cost.
    pub fn take_completed(&self) -> Option<PackageCpuSample> {
        self.completed.lock().ok()?.take()
    }
}

pub(crate) struct PackageCpuUpdate {
    collector: LodPackageCpuTelemetry,
    sample: Arc<Mutex<WorkingSample>>,
    started: Instant,
}

impl Drop for PackageCpuUpdate {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        ACTIVE.with(|active| *active.borrow_mut() = None);
        let Ok(mut sample) = self.sample.lock() else {
            return;
        };
        sample.observe(PackageCpuScope::UpdateSystem, elapsed);
        if let Ok(mut completed) = self.collector.completed.lock() {
            *completed = Some(sample.snapshot());
        }
    }
}

pub(crate) struct PackageCpuTimer {
    sample: Arc<Mutex<WorkingSample>>,
    scope: PackageCpuScope,
    started: Instant,
}

impl Drop for PackageCpuTimer {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        if let Ok(mut sample) = self.sample.lock() {
            sample.observe(self.scope, elapsed);
        }
    }
}

/// Disabled path: one optional-context check, no clock, lock, or allocation.
pub(crate) fn scope(scope: PackageCpuScope) -> Option<PackageCpuTimer> {
    ACTIVE.with(|active| {
        active.borrow().as_ref().map(|sample| PackageCpuTimer {
            sample: sample.clone(),
            scope,
            started: Instant::now(),
        })
    })
}

/// One call at a destination cache hit/miss, never inside the node traversal.
pub(crate) fn destination_observed(cache_hit: bool) {
    ACTIVE.with(|active| {
        let Some(sample) = active.borrow().as_ref().cloned() else {
            return;
        };
        if let Ok(mut sample) = sample.lock() {
            if cache_hit {
                sample.destination_cache_hits = sample.destination_cache_hits.saturating_add(1);
            } else {
                sample.destination_compilations = sample.destination_compilations.saturating_add(1);
            }
        }
    });
}

pub(crate) fn canonical_selected(visited_nodes: u32) {
    ACTIVE.with(|active| {
        let Some(sample) = active.borrow().as_ref().cloned() else {
            return;
        };
        if let Ok(mut sample) = sample.lock() {
            sample.canonical_visited_nodes = sample
                .canonical_visited_nodes
                .saturating_add(u64::from(visited_nodes));
        }
    });
}

#[cfg(any(
    test,
    all(feature = "headless", lod_render_path, not(target_arch = "wasm32"))
))]
#[derive(Clone, Debug)]
pub(crate) struct PackageCpuFrame {
    pub frame: u64,
    pub camera: u64,
    pub sample: PackageCpuSample,
}

#[cfg(any(
    test,
    all(feature = "headless", lod_render_path, not(target_arch = "wasm32"))
))]
impl LodPackageCpuTelemetry {
    /// Called once at the main-to-render extraction boundary. Consuming an
    /// update prevents accidental reuse for a later frame or another camera.
    pub(crate) fn take_for_frame(&self, frame: u64, camera: u64) -> Option<PackageCpuFrame> {
        self.take_completed().map(|sample| PackageCpuFrame {
            frame,
            camera,
            sample,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_scopes_emit_nothing_and_completed_updates_are_consumed_once() {
        let collector = LodPackageCpuTelemetry::default();
        assert!(scope(PackageCpuScope::DestinationPlan).is_none());
        assert!(collector.take_completed().is_none());
        {
            let _update = collector.begin_update().unwrap();
            let _plan = scope(PackageCpuScope::DestinationPlan).unwrap();
            destination_observed(true);
            destination_observed(false);
            canonical_selected(27);
        }
        let sample = collector.take_completed().unwrap();
        assert_eq!(sample.calls["lod_package_update_system"], 1);
        assert_eq!(sample.calls["lod_package_destination_plan"], 1);
        assert_eq!(sample.destination_cache_hits, 1);
        assert_eq!(sample.destination_compilations, 1);
        assert_eq!(sample.canonical_visited_nodes, 27);
        assert!(
            sample
                .cpu_ms
                .values()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!(
            !sample
                .cpu_ms
                .contains_key("lod_package_destination_compile")
        );
        assert!(collector.take_completed().is_none());
        assert!(scope(PackageCpuScope::DestinationPlan).is_none());
    }

    #[test]
    fn samples_do_not_accumulate_across_updates_or_collectors() {
        let first = LodPackageCpuTelemetry::default();
        let second = LodPackageCpuTelemetry::default();
        {
            let _update = first.begin_update().unwrap();
            assert!(second.begin_update().is_none());
            destination_observed(false);
            canonical_selected(13);
        }
        assert!(second.take_completed().is_none());
        {
            let _update = first.begin_update().unwrap();
            destination_observed(true);
        }
        let sample = first.take_completed().unwrap();
        assert_eq!(sample.destination_compilations, 0);
        assert_eq!(sample.canonical_visited_nodes, 0);
        assert_eq!(sample.destination_cache_hits, 1);
        assert_eq!(sample.calls["lod_package_update_system"], 1);
    }

    #[test]
    fn extended_scopes_remain_unobserved_until_called_and_preserve_nested_totals() {
        let mut working = WorkingSample::default();
        working.observe(PackageCpuScope::UpdateSystem, Duration::from_millis(8));
        let before = working.snapshot();
        for scope in [
            PackageCpuScope::RuntimePollPages,
            PackageCpuScope::RuntimeCommitPreprocessedPages,
            PackageCpuScope::TargetCandidates,
            PackageCpuScope::AdvanceStagedCut,
        ] {
            assert!(!before.calls.contains_key(scope.name()));
            assert!(!before.cpu_ms.contains_key(scope.name()));
        }
        working.observe(PackageCpuScope::TargetCandidates, Duration::from_millis(3));
        working.observe(PackageCpuScope::DestinationPlan, Duration::from_millis(1));
        working.observe(
            PackageCpuScope::RuntimePollPages,
            Duration::from_micros(100),
        );
        working.observe(
            PackageCpuScope::RuntimePollPages,
            Duration::from_micros(200),
        );
        let after = working.snapshot();
        assert_eq!(after.cpu_ms["lod_package_update_system"], 8.0);
        assert_eq!(after.cpu_ms["lod_package_target_candidates"], 3.0);
        assert_eq!(after.cpu_ms["lod_package_destination_plan"], 1.0);
        assert_eq!(after.calls["lod_package_runtime_poll_pages"], 2);
        assert!((after.cpu_ms["lod_package_runtime_poll_pages"] - 0.3).abs() < 1e-12);
        assert!(
            !after
                .calls
                .contains_key("lod_package_runtime_commit_preprocessed_pages")
        );
        assert!(!after.cpu_ms.contains_key("lod_package_advance_staged_cut"));
    }

    #[test]
    fn observation_counts_and_durations_saturate_without_wrapping() {
        let mut working = WorkingSample::default();
        working.calls[PackageCpuScope::DiscreteWavePlan as usize] = u64::MAX;
        working.elapsed[PackageCpuScope::DiscreteWavePlan as usize] = Duration::MAX;
        working.observe(PackageCpuScope::DiscreteWavePlan, Duration::from_secs(1));
        let sample = working.snapshot();
        assert_eq!(sample.calls["lod_package_discrete_wave_plan"], u64::MAX);
        assert!(sample.cpu_ms["lod_package_discrete_wave_plan"].is_finite());
    }
    #[test]
    fn extraction_stamps_exact_frame_and_camera_without_reusing_a_stale_update() {
        let collector = LodPackageCpuTelemetry::default();
        assert!(collector.take_for_frame(1, 7).is_none());
        drop(collector.begin_update().unwrap());
        let observed = collector.take_for_frame(11, 23).unwrap();
        assert_eq!((observed.frame, observed.camera), (11, 23));
        assert!(collector.take_for_frame(12, 24).is_none());
        drop(collector.begin_update().unwrap());
        let next = collector.take_for_frame(14, 29).unwrap();
        assert_eq!((next.frame, next.camera), (14, 29));
        assert_eq!(next.sample.calls["lod_package_update_system"], 1);
    }
}
