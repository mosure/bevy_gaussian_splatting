use std::{
    collections::{BTreeMap, VecDeque},
    num::NonZeroU32,
    sync::{
        Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{Receiver, Sender},
    },
    thread::JoinHandle,
};

use crate::gaussian::formats::planar_3d_chunked::LodPageId;

#[cfg(test)]
use super::cooperative::CooperativeBackend;
use super::{
    DEFAULT_NATIVE_PAGE_PREPROCESS_IN_FLIGHT_LIMIT,
    DEFAULT_NATIVE_PAGE_PREPROCESS_PENDING_BYTES_LIMIT, DEFAULT_NATIVE_PAGE_PREPROCESS_WORKERS,
    LodPagePreprocessBackend, LodPagePreprocessOutput, ReadyJob, WaitingJob, process_input,
    release_pending_bytes,
};

/// Native platform state. Production uses the process-wide worker pool, while
/// unit tests may select the same cooperative backend used by browsers.
pub(super) enum BackendState {
    Native(NativeBackend),
    #[cfg(test)]
    Cooperative(Box<CooperativeBackend>),
}

impl BackendState {
    pub(super) fn new() -> Result<Self, String> {
        #[cfg(test)]
        {
            Ok(Self::Cooperative(Box::new(CooperativeBackend::new())))
        }

        #[cfg(not(test))]
        {
            NativeBackend::new().map(Self::Native)
        }
    }

    #[cfg(test)]
    pub(super) fn new_native_for_tests() -> Result<Self, String> {
        NativeBackend::new().map(Self::Native)
    }

    #[cfg(test)]
    pub(super) fn new_cooperative_for_tests() -> Self {
        Self::Cooperative(Box::new(CooperativeBackend::new()))
    }

    pub(super) fn kind(&self) -> LodPagePreprocessBackend {
        match self {
            Self::Native(_) => LodPagePreprocessBackend::NativeWorkerPool,
            #[cfg(test)]
            Self::Cooperative(backend) => backend.kind(),
        }
    }

    pub(super) fn advance(
        &mut self,
        _frame_sequence: u64,
        _cooperative_budget: NonZeroU32,
        waiting: &mut VecDeque<WaitingJob>,
        ready: &mut BTreeMap<LodPageId, ReadyJob>,
        pending_bytes: &mut u64,
        deferred_admissions: &mut u64,
    ) {
        match self {
            Self::Native(backend) => {
                backend.drain_completions(ready, pending_bytes);
                backend.pump_admission(waiting, deferred_admissions);
                backend.drain_completions(ready, pending_bytes);
            }
            #[cfg(test)]
            Self::Cooperative(backend) => {
                backend.advance(_frame_sequence, _cooperative_budget, waiting, ready);
            }
        }
    }

    pub(super) fn cancel_running(&mut self, page: LodPageId, pending_bytes: &mut u64) -> bool {
        match self {
            Self::Native(backend) => backend.cancel(page, pending_bytes),
            #[cfg(test)]
            Self::Cooperative(backend) => backend.cancel(page, pending_bytes),
        }
    }

    pub(super) fn is_running(&self, page: LodPageId) -> bool {
        match self {
            Self::Native(backend) => backend.running.contains_key(&page),
            #[cfg(test)]
            Self::Cooperative(backend) => backend.contains(page),
        }
    }

    pub(super) fn tracked_len(&self) -> usize {
        match self {
            Self::Native(backend) => backend
                .running
                .len()
                .saturating_add(backend.cancelled_running.len()),
            #[cfg(test)]
            Self::Cooperative(backend) => backend.tracked_len(),
        }
    }

    pub(super) fn running_page_ids(&self) -> Vec<LodPageId> {
        match self {
            Self::Native(backend) => backend.running.keys().copied().collect(),
            #[cfg(test)]
            Self::Cooperative(backend) => backend.page_ids(),
        }
    }

    pub(super) fn cooperative_progress(&self) -> (u32, u32) {
        match self {
            Self::Native(_) => (0, 0),
            #[cfg(test)]
            Self::Cooperative(backend) => backend.progress(),
        }
    }

    pub(super) fn cooperative_budget(&self) -> u32 {
        match self {
            Self::Native(_) => 0,
            #[cfg(test)]
            Self::Cooperative(backend) => backend.budget(),
        }
    }

    pub(super) fn native_job_byte_capacity(&self) -> Option<u64> {
        match self {
            Self::Native(_) => Some(DEFAULT_NATIVE_PAGE_PREPROCESS_PENDING_BYTES_LIMIT),
            #[cfg(test)]
            Self::Cooperative(_) => None,
        }
    }
}

struct RunningJob {
    job_id: u64,
    pending_bytes: u64,
    cancelled: Arc<AtomicBool>,
}

struct CancelledRunningJob {
    pending_bytes: u64,
}

pub(super) struct NativeBackend {
    running: BTreeMap<LodPageId, RunningJob>,
    cancelled_running: BTreeMap<u64, CancelledRunningJob>,
    owner_id: u64,
    next_job_id: u64,
    completion_tx: Sender<NativeCompletion>,
    completion_rx: Mutex<Receiver<NativeCompletion>>,
}

impl NativeBackend {
    fn new() -> Result<Self, String> {
        native_pool()?;
        Ok(Self::detached())
    }

    fn detached() -> Self {
        let (completion_tx, completion_rx) = std::sync::mpsc::channel();
        Self {
            running: BTreeMap::new(),
            cancelled_running: BTreeMap::new(),
            owner_id: next_owner_id(),
            next_job_id: 1,
            completion_tx,
            completion_rx: Mutex::new(completion_rx),
        }
    }

    fn pump_admission(
        &mut self,
        waiting: &mut VecDeque<WaitingJob>,
        deferred_admissions: &mut u64,
    ) {
        let Ok(pool) = native_pool() else {
            return;
        };
        while let Some(waiting_job) = waiting.pop_front() {
            let page = waiting_job.input.request.page_id;
            let pending_bytes = waiting_job.pending_bytes;
            let job_id = self.next_job_id;
            self.next_job_id = self.next_job_id.wrapping_add(1).max(1);
            let cancelled = Arc::new(AtomicBool::new(false));
            let job = NativeJob {
                owner_id: self.owner_id,
                job_id,
                pending_bytes,
                input: waiting_job.input,
                cancelled: cancelled.clone(),
                completion_tx: self.completion_tx.clone(),
            };
            match pool.try_submit(job) {
                Ok(()) => {
                    self.running.insert(
                        page,
                        RunningJob {
                            job_id,
                            pending_bytes,
                            cancelled,
                        },
                    );
                }
                Err(job) => {
                    waiting.push_front(WaitingJob {
                        input: job.input,
                        pending_bytes: job.pending_bytes,
                    });
                    *deferred_admissions = deferred_admissions.saturating_add(1);
                    break;
                }
            }
        }
    }

    fn drain_completions(
        &mut self,
        ready: &mut BTreeMap<LodPageId, ReadyJob>,
        pending_bytes: &mut u64,
    ) {
        let completions = {
            let completion_rx = self
                .completion_rx
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner);
            completion_rx.try_iter().collect::<Vec<_>>()
        };
        for completion in completions {
            // Cancellation can race after a worker observed `false` but before
            // its completion is drained. The local cancelled-job ledger is
            // authoritative regardless of the worker's sampled flag.
            if let Some(cancelled) = self.cancelled_running.remove(&completion.job_id) {
                release_pending_bytes(pending_bytes, cancelled.pending_bytes);
                continue;
            }
            if completion.cancelled {
                continue;
            }
            let Some(output) = completion.output else {
                continue;
            };
            let page = output.request.page_id;
            let Some(running) = self.running.get(&page) else {
                continue;
            };
            if running.job_id != completion.job_id {
                continue;
            }
            let running = self
                .running
                .remove(&page)
                .expect("running page was checked above");
            ready.insert(
                page,
                ReadyJob {
                    output,
                    pending_bytes: running.pending_bytes,
                },
            );
        }
    }

    fn cancel(&mut self, page: LodPageId, pending_bytes: &mut u64) -> bool {
        let Some(running) = self.running.remove(&page) else {
            return false;
        };
        running.cancelled.store(true, Ordering::Release);
        if native_pool().is_ok_and(|pool| pool.cancel_queued(self.owner_id, running.job_id)) {
            release_pending_bytes(pending_bytes, running.pending_bytes);
        } else {
            self.cancelled_running.insert(
                running.job_id,
                CancelledRunningJob {
                    pending_bytes: running.pending_bytes,
                },
            );
        }
        true
    }
}

struct NativeCompletion {
    job_id: u64,
    cancelled: bool,
    output: Option<LodPagePreprocessOutput>,
}

struct NativeJob {
    owner_id: u64,
    job_id: u64,
    pending_bytes: u64,
    input: super::LodPagePreprocessInput,
    cancelled: Arc<AtomicBool>,
    completion_tx: Sender<NativeCompletion>,
}

impl NativeJob {
    fn run(self) {
        if self.cancelled.load(Ordering::Acquire) {
            let _ = self.completion_tx.send(NativeCompletion {
                job_id: self.job_id,
                cancelled: true,
                output: None,
            });
            return;
        }
        let output = process_input(self.input);
        let cancelled = self.cancelled.load(Ordering::Acquire);
        let _ = self.completion_tx.send(NativeCompletion {
            job_id: self.job_id,
            cancelled,
            output: (!cancelled).then_some(output),
        });
    }
}

struct OwnedJob<T> {
    owner_id: u64,
    value: T,
}

/// FIFO within one owner and deterministic round-robin between owners.
struct FairOwnerQueue<T> {
    jobs: VecDeque<OwnedJob<T>>,
    ready_owners: VecDeque<u64>,
}

impl<T> Default for FairOwnerQueue<T> {
    fn default() -> Self {
        Self {
            jobs: VecDeque::new(),
            ready_owners: VecDeque::new(),
        }
    }
}

impl<T> FairOwnerQueue<T> {
    fn try_push(&mut self, owner_id: u64, value: T) -> Result<(), T> {
        let owner_already_queued = self.jobs.iter().any(|job| job.owner_id == owner_id);
        if self.jobs.try_reserve(1).is_err()
            || (!owner_already_queued && self.ready_owners.try_reserve(1).is_err())
        {
            return Err(value);
        }
        self.jobs.push_back(OwnedJob { owner_id, value });
        if !owner_already_queued {
            self.ready_owners.push_back(owner_id);
        }
        Ok(())
    }

    fn pop(&mut self) -> Option<T> {
        let owner_id = self.ready_owners.pop_front()?;
        let index = self
            .jobs
            .iter()
            .position(|job| job.owner_id == owner_id)
            .expect("a ready preprocessing owner has a queued job");
        let job = self
            .jobs
            .remove(index)
            .expect("the selected preprocessing job exists");
        if self.jobs.iter().any(|job| job.owner_id == owner_id) {
            self.ready_owners.push_back(owner_id);
        }
        Some(job.value)
    }

    fn remove_matching(&mut self, owner_id: u64, predicate: impl Fn(&T) -> bool) -> Option<T> {
        let index = self
            .jobs
            .iter()
            .position(|job| job.owner_id == owner_id && predicate(&job.value))?;
        let job = self
            .jobs
            .remove(index)
            .expect("the matched preprocessing job exists");
        if !self.jobs.iter().any(|job| job.owner_id == owner_id) {
            self.ready_owners.retain(|queued| *queued != owner_id);
        }
        Some(job.value)
    }
}

struct NativePoolState {
    jobs: FairOwnerQueue<NativeJob>,
    in_flight: usize,
    pending_bytes: u64,
    shutting_down: bool,
}

impl NativePoolState {
    fn can_admit(&self, job_bytes: u64, in_flight_limit: usize, pending_bytes_limit: u64) -> bool {
        !self.shutting_down
            && self.in_flight < in_flight_limit
            && self
                .pending_bytes
                .checked_add(job_bytes)
                .is_some_and(|bytes| bytes <= pending_bytes_limit)
    }
}

struct NativePoolShared {
    state: Mutex<NativePoolState>,
    work_available: Condvar,
}

struct NativePool {
    shared: Arc<NativePoolShared>,
    workers: Vec<JoinHandle<()>>,
    in_flight_limit: usize,
    pending_bytes_limit: u64,
}

impl NativePool {
    fn new() -> Result<Self, String> {
        let shared = Arc::new(NativePoolShared {
            state: Mutex::new(NativePoolState {
                jobs: FairOwnerQueue::default(),
                in_flight: 0,
                pending_bytes: 0,
                shutting_down: false,
            }),
            work_available: Condvar::new(),
        });
        let mut workers = Vec::with_capacity(DEFAULT_NATIVE_PAGE_PREPROCESS_WORKERS);
        for worker in 0..DEFAULT_NATIVE_PAGE_PREPROCESS_WORKERS {
            let worker_shared = shared.clone();
            match std::thread::Builder::new()
                .name(format!("lod-page-preprocess-{worker}"))
                .spawn(move || native_worker(worker_shared))
            {
                Ok(handle) => workers.push(handle),
                Err(error) => {
                    {
                        let mut state = lock_native_pool_state(&shared.state);
                        state.shutting_down = true;
                    }
                    shared.work_available.notify_all();
                    for handle in workers {
                        let _ = handle.join();
                    }
                    return Err(format!(
                        "failed to spawn preprocessing worker {worker}: {error}"
                    ));
                }
            }
        }
        Ok(Self {
            shared,
            workers,
            in_flight_limit: DEFAULT_NATIVE_PAGE_PREPROCESS_IN_FLIGHT_LIMIT,
            pending_bytes_limit: DEFAULT_NATIVE_PAGE_PREPROCESS_PENDING_BYTES_LIMIT,
        })
    }

    // Returning the owned job avoids an additional allocation precisely on
    // the bounded backpressure path; the caller immediately restores it to
    // the per-runtime waiting queue.
    #[allow(clippy::result_large_err)]
    fn try_submit(&self, job: NativeJob) -> Result<(), NativeJob> {
        let mut state = lock_native_pool_state(&self.shared.state);
        let Some(next_pending_bytes) = state.pending_bytes.checked_add(job.pending_bytes) else {
            return Err(job);
        };
        if !state.can_admit(
            job.pending_bytes,
            self.in_flight_limit,
            self.pending_bytes_limit,
        ) {
            return Err(job);
        }
        let owner_id = job.owner_id;
        state.jobs.try_push(owner_id, job)?;
        state.in_flight += 1;
        state.pending_bytes = next_pending_bytes;
        drop(state);
        self.shared.work_available.notify_one();
        Ok(())
    }

    fn cancel_queued(&self, owner_id: u64, job_id: u64) -> bool {
        let mut state = lock_native_pool_state(&self.shared.state);
        let Some(job) = state
            .jobs
            .remove_matching(owner_id, |job| job.job_id == job_id)
        else {
            return false;
        };
        state.in_flight = state
            .in_flight
            .checked_sub(1)
            .expect("native preprocessing job accounting underflow");
        state.pending_bytes = state
            .pending_bytes
            .checked_sub(job.pending_bytes)
            .expect("native preprocessing byte accounting underflow");
        true
    }
}

impl Drop for NativePool {
    fn drop(&mut self) {
        {
            let mut state = lock_native_pool_state(&self.shared.state);
            state.shutting_down = true;
        }
        self.shared.work_available.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn lock_native_pool_state(state: &Mutex<NativePoolState>) -> MutexGuard<'_, NativePoolState> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

fn native_worker(shared: Arc<NativePoolShared>) {
    loop {
        let job = {
            let mut state = lock_native_pool_state(&shared.state);
            loop {
                if state.shutting_down {
                    return;
                }
                if let Some(job) = state.jobs.pop() {
                    break job;
                }
                state = shared
                    .work_available
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        let pending_bytes = job.pending_bytes;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run()));
        let mut state = lock_native_pool_state(&shared.state);
        state.in_flight = state
            .in_flight
            .checked_sub(1)
            .expect("native preprocessing job accounting underflow");
        state.pending_bytes = state
            .pending_bytes
            .checked_sub(pending_bytes)
            .expect("native preprocessing byte accounting underflow");
        drop(state);
        shared.work_available.notify_all();
    }
}

fn native_pool() -> Result<&'static NativePool, String> {
    static POOL: OnceLock<Result<NativePool, String>> = OnceLock::new();
    POOL.get_or_init(NativePool::new)
        .as_ref()
        .map_err(Clone::clone)
}

fn next_owner_id() -> u64 {
    static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
    NEXT_OWNER.fetch_add(1, Ordering::Relaxed).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::preprocess::{LodPagePreprocessAdmissionError, LodPagePreprocessor};

    #[test]
    fn fair_owner_queue_is_fifo_per_owner_and_round_robin_between_owners() {
        let mut queue = FairOwnerQueue::default();
        queue.try_push(1, 1_u8).unwrap();
        queue.try_push(1, 2_u8).unwrap();
        queue.try_push(2, 3_u8).unwrap();
        queue.try_push(2, 4_u8).unwrap();
        queue.try_push(3, 5_u8).unwrap();

        let order = (0..5).map(|_| queue.pop().unwrap()).collect::<Vec<_>>();
        assert_eq!(order, [1, 3, 5, 2, 4]);
        assert!(queue.pop().is_none());
    }

    #[test]
    fn fair_owner_queue_removal_does_not_leave_a_stale_ready_owner() {
        let mut queue = FairOwnerQueue::default();
        queue.try_push(1, 1_u8).unwrap();
        queue.try_push(2, 2_u8).unwrap();
        assert_eq!(queue.remove_matching(1, |value| *value == 1), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert!(queue.pop().is_none());
    }

    #[test]
    fn native_state_enforces_count_byte_overflow_and_shutdown_boundaries() {
        let mut state = NativePoolState {
            jobs: FairOwnerQueue::default(),
            in_flight: 3,
            pending_bytes: 300,
            shutting_down: false,
        };
        assert!(state.can_admit(100, 4, 400));
        assert!(!state.can_admit(101, 4, 400));
        state.in_flight = 4;
        assert!(!state.can_admit(1, 4, 1_000));
        state.in_flight = 0;
        state.pending_bytes = u64::MAX;
        assert!(!state.can_admit(1, 4, u64::MAX));
        state.pending_bytes = 0;
        state.shutting_down = true;
        assert!(!state.can_admit(1, 4, 1_000));
    }

    #[test]
    fn oversized_native_job_is_rejected_before_process_pool_admission() {
        let mut preprocessor =
            LodPagePreprocessor::new_cooperative_with_byte_capacity_for_tests(1, u64::MAX).unwrap();
        preprocessor.backend = BackendState::Native(NativeBackend::detached());
        assert_eq!(
            preprocessor
                .validate_job_bytes(DEFAULT_NATIVE_PAGE_PREPROCESS_PENDING_BYTES_LIMIT + 1, 0,),
            Err(
                LodPagePreprocessAdmissionError::NativeJobByteCapacityExceeded {
                    requested: DEFAULT_NATIVE_PAGE_PREPROCESS_PENDING_BYTES_LIMIT + 1,
                    capacity: DEFAULT_NATIVE_PAGE_PREPROCESS_PENDING_BYTES_LIMIT,
                }
            )
        );
        // Keep Drop state-local; this test intentionally never constructs the
        // process-wide worker pool.
        preprocessor.backend = BackendState::Cooperative(Box::new(CooperativeBackend::new()));
    }

    #[test]
    fn cancellation_ledger_releases_a_racing_non_cancelled_completion() {
        let mut backend = NativeBackend::detached();
        let mut pending_bytes = 50;
        backend
            .cancelled_running
            .insert(7, CancelledRunningJob { pending_bytes: 50 });
        backend
            .completion_tx
            .send(NativeCompletion {
                job_id: 7,
                cancelled: false,
                output: None,
            })
            .unwrap();

        backend.drain_completions(&mut BTreeMap::new(), &mut pending_bytes);
        assert_eq!(pending_bytes, 0);
        assert!(backend.cancelled_running.is_empty());
    }
}
