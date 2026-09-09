//! Process-local admission for explicitly owned LoD allocations.
//!
//! One resource is shared between application and render worlds. Reservations
//! retain allocation identity across `Arc` sharing, and atomic multi-category
//! admission prevents several packages/views from each spending a global limit.
//! These are owned capacity reservations, not measurements of RSS or device
//! memory. Driver allocations, Bevy/wgpu staging, unrelated renderer resources
//! and persistent cache files are outside this ledger.

use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use bevy::prelude::Resource;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct LodMemoryLimits {
    pub max_cpu_bytes: u64,
    pub max_gpu_bytes: u64,
}

impl Default for LodMemoryLimits {
    fn default() -> Self {
        Self {
            max_cpu_bytes: 4 * 1024 * 1024 * 1024,
            max_gpu_bytes: 4 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[repr(usize)]
pub enum LodMemoryCategory {
    AtlasGpu,
    CompactionGpu,
    DecodedPagesCpu,
    RecoveryStagingCpu,
    TransportCpu,
    PreprocessCpu,
    MetadataCpu,
    TransitionCpu,
    UploadStagingCpu,
}

impl LodMemoryCategory {
    pub const ALL: [Self; 9] = [
        Self::AtlasGpu,
        Self::CompactionGpu,
        Self::DecodedPagesCpu,
        Self::RecoveryStagingCpu,
        Self::TransportCpu,
        Self::PreprocessCpu,
        Self::MetadataCpu,
        Self::TransitionCpu,
        Self::UploadStagingCpu,
    ];

    pub const fn is_gpu(self) -> bool {
        matches!(self, Self::AtlasGpu | Self::CompactionGpu)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LodMemoryAllocationId(u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct LodMemoryCategoryUsage {
    pub category: LodMemoryCategory,
    pub bytes: u64,
    pub allocations: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct LodMemorySnapshot {
    pub limits: LodMemoryLimits,
    pub categories: [LodMemoryCategoryUsage; 9],
    pub cpu_bytes: u64,
    pub gpu_bytes: u64,
    /// Saturates if independently representable CPU/GPU totals exceed u64.
    pub total_bytes: u64,
    pub allocations: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodMemoryAdmissionError {
    ByteOverflow,
    IdentityExhausted,
    LimitExceeded {
        gpu: bool,
        requested: u64,
        available: u64,
    },
    SharedReservationCannotSplit,
    SplitExceedsReservation,
}

impl fmt::Display for LodMemoryAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ByteOverflow => write!(f, "LoD memory reservation byte count overflowed"),
            Self::IdentityExhausted => write!(f, "LoD allocation identity space exhausted"),
            Self::LimitExceeded {
                gpu,
                requested,
                available,
            } => write!(
                f,
                "LoD {} reservation of {requested} bytes exceeds {available} available bytes",
                if *gpu { "GPU" } else { "CPU" }
            ),
            Self::SharedReservationCannotSplit => {
                write!(f, "a shared LoD reservation cannot be split")
            }
            Self::SplitExceedsReservation => write!(f, "LoD reservation split exceeds its bytes"),
        }
    }
}
impl std::error::Error for LodMemoryAdmissionError {}

struct LedgerState {
    limits: LodMemoryLimits,
    next_id: u64,
    gpu_release_epoch: u64,
    bytes: [u64; 9],
    allocations: BTreeMap<LodMemoryAllocationId, LodMemoryCategory>,
}

/// Clone this resource into the render world; constructing another ledger
/// creates a separate budget. A lease releases only after its final owner drops.
#[derive(Resource, Clone)]
pub struct LodMemoryLedger(Arc<Mutex<LedgerState>>);

impl Default for LodMemoryLedger {
    fn default() -> Self {
        Self::new(LodMemoryLimits::default())
    }
}

impl LodMemoryLedger {
    pub fn new(limits: LodMemoryLimits) -> Self {
        Self(Arc::new(Mutex::new(LedgerState {
            limits,
            next_id: 1,
            gpu_release_epoch: 0,
            bytes: [0; 9],
            allocations: BTreeMap::new(),
        })))
    }

    pub fn limits(&self) -> LodMemoryLimits {
        self.0.lock().expect("LoD memory ledger poisoned").limits
    }

    /// Existing owners remain valid when a ceiling is lowered. New allocation
    /// admission resumes only once the relevant total fits the new ceiling.
    pub fn set_limits(&self, limits: LodMemoryLimits) {
        self.0.lock().expect("LoD memory ledger poisoned").limits = limits;
    }

    /// Changes only when positive, materialized GPU capacity is returned. Render policy can
    /// retry a rejected replacement after submission retirement frees memory,
    /// without treating stationary zero-byte reuse as an environment change.
    pub fn gpu_release_epoch(&self) -> u64 {
        self.0
            .lock()
            .expect("LoD memory ledger poisoned")
            .gpu_release_epoch
    }

    pub fn snapshot(&self) -> LodMemorySnapshot {
        let state = self.0.lock().expect("LoD memory ledger poisoned");
        let categories = LodMemoryCategory::ALL.map(|category| LodMemoryCategoryUsage {
            category,
            bytes: state.bytes[category as usize],
            allocations: state
                .allocations
                .values()
                .filter(|value| **value == category)
                .count() as u64,
        });
        let cpu_bytes = categories
            .iter()
            .filter(|entry| !entry.category.is_gpu())
            .fold(0u64, |total, entry| total.saturating_add(entry.bytes));
        let gpu_bytes = categories
            .iter()
            .filter(|entry| entry.category.is_gpu())
            .fold(0u64, |total, entry| total.saturating_add(entry.bytes));
        LodMemorySnapshot {
            limits: state.limits,
            categories,
            cpu_bytes,
            gpu_bytes,
            total_bytes: cpu_bytes.saturating_add(gpu_bytes),
            allocations: state.allocations.len() as u64,
        }
    }

    pub fn try_reserve(
        &self,
        category: LodMemoryCategory,
        bytes: u64,
    ) -> Result<LodMemoryLease, LodMemoryAdmissionError> {
        Ok(self
            .try_reserve_many(&[(category, bytes)])?
            .pop()
            .expect("one reservation"))
    }

    /// All requested allocations acquire capacity together, or none changes
    /// the ledger. Existing allocations remain charged throughout admission.
    pub fn try_reserve_many(
        &self,
        charges: &[(LodMemoryCategory, u64)],
    ) -> Result<Vec<LodMemoryLease>, LodMemoryAdmissionError> {
        let mut state = self.0.lock().expect("LoD memory ledger poisoned");
        let mut requested = [0u64; 2];
        let mut current = [0u64; 2];
        for category in LodMemoryCategory::ALL {
            let index = usize::from(category.is_gpu());
            current[index] = current[index]
                .checked_add(state.bytes[category as usize])
                .ok_or(LodMemoryAdmissionError::ByteOverflow)?;
        }
        for &(category, bytes) in charges {
            let index = usize::from(category.is_gpu());
            requested[index] = requested[index]
                .checked_add(bytes)
                .ok_or(LodMemoryAdmissionError::ByteOverflow)?;
        }
        for (index, limit) in [state.limits.max_cpu_bytes, state.limits.max_gpu_bytes]
            .into_iter()
            .enumerate()
        {
            let available = limit.saturating_sub(current[index]);
            if requested[index] > available {
                return Err(LodMemoryAdmissionError::LimitExceeded {
                    gpu: index == 1,
                    requested: requested[index],
                    available,
                });
            }
        }
        let next_id = state
            .next_id
            .checked_add(charges.len() as u64)
            .ok_or(LodMemoryAdmissionError::IdentityExhausted)?;
        let mut leases = Vec::with_capacity(charges.len());
        for &(category, bytes) in charges {
            let id = LodMemoryAllocationId(state.next_id);
            state.next_id += 1;
            let inner = Arc::new(LeaseInner {
                ledger: Arc::clone(&self.0),
                id,
                category,
                bytes,
                gpu_materialized: AtomicBool::new(false),
            });
            state.bytes[category as usize] += bytes;
            state.allocations.insert(id, category);
            leases.push(LodMemoryLease(inner));
        }
        debug_assert_eq!(state.next_id, next_id);
        Ok(leases)
    }
}

struct LeaseInner {
    ledger: Arc<Mutex<LedgerState>>,
    id: LodMemoryAllocationId,
    category: LodMemoryCategory,
    bytes: u64,
    gpu_materialized: AtomicBool,
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        let mut state = self.ledger.lock().expect("LoD memory ledger poisoned");
        state.bytes[self.category as usize] -= self.bytes;
        if self.category.is_gpu()
            && self.bytes != 0
            && self.gpu_materialized.load(Ordering::Acquire)
        {
            state.gpu_release_epoch = state.gpu_release_epoch.wrapping_add(1);
        }
        state.allocations.remove(&self.id);
    }
}

/// Identity-preserving allocation ownership. Cloning adds an owner, not another
/// byte charge. GPU callers retain a lease through queue completion, including
/// after the last CPU buffer handle or per-view state has been removed.
#[derive(Clone)]
pub struct LodMemoryLease(Arc<LeaseInner>);

impl fmt::Debug for LodMemoryLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LodMemoryLease")
            .field("id", &self.0.id)
            .field("category", &self.0.category)
            .field("bytes", &self.0.bytes)
            .finish()
    }
}

impl LodMemoryLease {
    /// Supplies the same process-local authority to dynamically allocated
    /// runtime payloads owned by an already-admitted package.
    pub(crate) fn ledger(&self) -> LodMemoryLedger {
        LodMemoryLedger(Arc::clone(&self.0.ledger))
    }

    pub fn allocation_id(&self) -> LodMemoryAllocationId {
        self.0.id
    }
    pub fn bytes(&self) -> u64 {
        self.0.bytes
    }
    pub fn category(&self) -> LodMemoryCategory {
        self.0.category
    }

    /// Attests that this GPU reservation backed an actual allocation. Owners
    /// may mark it when transferring storage into a submission-retirement
    /// fence. Releasing unused planning capacity never advances the render
    /// retry epoch; this avoids replan churn while atlas staging is pending.
    pub fn mark_gpu_materialized(&self) {
        if self.0.category.is_gpu() {
            self.0.gpu_materialized.store(true, Ordering::Release);
        }
    }

    /// Splits exclusive aggregate ownership without briefly releasing capacity.
    /// Useful when one physical buffer retires from an admitted allocation group.
    pub fn split_off(&mut self, bytes: u64) -> Result<Self, LodMemoryAdmissionError> {
        if bytes > self.0.bytes {
            return Err(LodMemoryAdmissionError::SplitExceedsReservation);
        }
        if Arc::strong_count(&self.0) != 1 {
            return Err(LodMemoryAdmissionError::SharedReservationCannotSplit);
        }
        let ledger = Arc::clone(&self.0.ledger);
        let mut state = ledger.lock().expect("LoD memory ledger poisoned");
        let next_id = state
            .next_id
            .checked_add(1)
            .ok_or(LodMemoryAdmissionError::IdentityExhausted)?;
        let id = LodMemoryAllocationId(state.next_id);
        state.next_id = next_id;
        let inner =
            Arc::get_mut(&mut self.0).expect("exclusive allocation identity was checked above");
        inner.bytes -= bytes;
        let split = Arc::new(LeaseInner {
            ledger: Arc::clone(&ledger),
            id,
            category: inner.category,
            bytes,
            gpu_materialized: AtomicBool::new(inner.gpu_materialized.load(Ordering::Acquire)),
        });
        state.allocations.insert(id, inner.category);
        drop(state);
        Ok(Self(split))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_identity_charges_once_until_final_owner_drops() {
        let ledger = LodMemoryLedger::new(LodMemoryLimits {
            max_cpu_bytes: 100,
            max_gpu_bytes: 100,
        });
        let first = ledger
            .try_reserve(LodMemoryCategory::RecoveryStagingCpu, 80)
            .unwrap();
        let shared = first.clone();
        assert_eq!(first.allocation_id(), shared.allocation_id());
        assert_eq!(ledger.snapshot().cpu_bytes, 80);
        drop(first);
        assert!(
            ledger
                .try_reserve(LodMemoryCategory::DecodedPagesCpu, 21)
                .is_err()
        );
        drop(shared);
        assert_eq!(ledger.snapshot().cpu_bytes, 0);
    }

    #[test]
    fn category_admission_is_atomic_and_successor_overlap_stays_charged() {
        let ledger = LodMemoryLedger::new(LodMemoryLimits {
            max_cpu_bytes: 100,
            max_gpu_bytes: 100,
        });
        let old = ledger
            .try_reserve(LodMemoryCategory::CompactionGpu, 60)
            .unwrap();
        assert!(
            ledger
                .try_reserve_many(&[
                    (LodMemoryCategory::MetadataCpu, 50),
                    (LodMemoryCategory::AtlasGpu, 41)
                ])
                .is_err()
        );
        assert_eq!(ledger.snapshot().cpu_bytes, 0);
        let successor = ledger
            .try_reserve(LodMemoryCategory::CompactionGpu, 40)
            .unwrap();
        assert_eq!(ledger.snapshot().gpu_bytes, 100);
        drop(old);
        assert_eq!(ledger.snapshot().gpu_bytes, 40);
        drop(successor);
        assert_eq!(ledger.snapshot().allocations, 0);
    }

    #[test]
    fn lowering_limits_retains_owned_capacity_and_gpu_release_reopens_retry_epoch() {
        let ledger = LodMemoryLedger::new(LodMemoryLimits {
            max_cpu_bytes: 100,
            max_gpu_bytes: 100,
        });
        let gpu = ledger.try_reserve(LodMemoryCategory::AtlasGpu, 80).unwrap();
        ledger.set_limits(LodMemoryLimits {
            max_cpu_bytes: 100,
            max_gpu_bytes: 20,
        });
        assert_eq!(ledger.snapshot().gpu_bytes, 80);
        assert!(
            ledger
                .try_reserve(LodMemoryCategory::CompactionGpu, 1)
                .is_err()
        );
        drop(
            ledger
                .try_reserve(LodMemoryCategory::CompactionGpu, 0)
                .unwrap(),
        );
        assert_eq!(ledger.gpu_release_epoch(), 0);
        gpu.mark_gpu_materialized();
        drop(gpu);
        assert_eq!(ledger.gpu_release_epoch(), 1);
        assert!(
            ledger
                .try_reserve(LodMemoryCategory::CompactionGpu, 20)
                .is_ok()
        );
    }

    #[test]
    fn pending_growth_quota_reuse_does_not_invalidate_stationary_render_environment() {
        let ledger = LodMemoryLedger::new(LodMemoryLimits {
            max_cpu_bytes: 100,
            max_gpu_bytes: 100,
        });
        let mut live_and_spare = ledger
            .try_reserve(LodMemoryCategory::CompactionGpu, 100)
            .unwrap();
        let unused = live_and_spare.split_off(40).unwrap();
        drop(unused);
        for _ in 0..8 {
            drop(
                ledger
                    .try_reserve(LodMemoryCategory::CompactionGpu, 40)
                    .unwrap(),
            );
        }
        assert_eq!(
            ledger.gpu_release_epoch(),
            0,
            "unused descriptor/morph planning capacity is not GPU retirement"
        );
        live_and_spare.mark_gpu_materialized();
        let retired_clone = live_and_spare.clone();
        drop(live_and_spare);
        assert_eq!(
            ledger.gpu_release_epoch(),
            0,
            "submitted owner still retains storage"
        );
        drop(retired_clone);
        assert_eq!(ledger.gpu_release_epoch(), 1);
    }

    #[test]
    fn split_preserves_charge_and_rejects_shared_owner_mutation() {
        let ledger = LodMemoryLedger::new(LodMemoryLimits {
            max_cpu_bytes: 100,
            max_gpu_bytes: 100,
        });
        let mut allocation = ledger
            .try_reserve(LodMemoryCategory::CompactionGpu, 100)
            .unwrap();
        let clone = allocation.clone();
        assert!(matches!(
            allocation.split_off(40),
            Err(LodMemoryAdmissionError::SharedReservationCannotSplit)
        ));
        assert_eq!(ledger.snapshot().gpu_bytes, 100);
        drop(clone);
        let retired = allocation.split_off(40).unwrap();
        assert_ne!(retired.allocation_id(), allocation.allocation_id());
        assert_eq!(allocation.bytes(), 60);
        assert_eq!(ledger.snapshot().gpu_bytes, 100);
        assert_eq!(ledger.snapshot().allocations, 2);
        drop(allocation);
        assert_eq!(ledger.snapshot().gpu_bytes, 40);
        drop(retired);
        assert_eq!(ledger.snapshot().gpu_bytes, 0);
    }

    #[test]
    fn concurrent_worlds_cannot_each_spend_the_same_available_capacity() {
        let ledger = LodMemoryLedger::new(LodMemoryLimits {
            max_cpu_bytes: 100,
            max_gpu_bytes: 100,
        });
        let mut owners = Vec::new();
        std::thread::scope(|scope| {
            let handles = (0..4)
                .map(|_| {
                    let ledger = ledger.clone();
                    scope.spawn(move || ledger.try_reserve(LodMemoryCategory::AtlasGpu, 60))
                })
                .collect::<Vec<_>>();
            for handle in handles {
                if let Ok(lease) = handle.join().unwrap() {
                    owners.push(lease);
                }
            }
        });
        assert_eq!(owners.len(), 1);
        assert_eq!(ledger.snapshot().gpu_bytes, 60);
        drop(owners);
        assert_eq!(ledger.snapshot().total_bytes, 0);
    }

    #[test]
    fn overflow_and_zero_limits_never_reopen_admission() {
        let ledger = LodMemoryLedger::new(LodMemoryLimits {
            max_cpu_bytes: u64::MAX,
            max_gpu_bytes: 0,
        });
        assert!(matches!(
            ledger.try_reserve_many(&[
                (LodMemoryCategory::MetadataCpu, u64::MAX),
                (LodMemoryCategory::DecodedPagesCpu, 1)
            ]),
            Err(LodMemoryAdmissionError::ByteOverflow)
        ));
        assert!(ledger.try_reserve(LodMemoryCategory::AtlasGpu, 1).is_err());
        assert_eq!(ledger.snapshot().total_bytes, 0);
    }
}
