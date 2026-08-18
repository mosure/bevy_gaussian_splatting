//! Bounded CPU bookkeeping for a GPU page atlas.
//!
//! The allocator uses generations so queued draw/traversal work can detect slots
//! that were evicted and reused. The cache is deterministic LRU with explicit
//! pins for visible ancestor fallbacks.

use std::collections::{BTreeMap, BTreeSet};

use bevy::prelude::Reflect;
use bevy_args::{Deserialize, Serialize};

use crate::gaussian::lod_settings::LodBudgets;

use super::transport::LodPageId;

#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Reflect, Serialize, Deserialize,
)]
pub struct AtlasSlot {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug)]
struct SlotState {
    generation: u32,
    page_id: Option<LodPageId>,
}

/// Fixed-capacity generation-safe slot allocator.
#[derive(Clone, Debug)]
pub struct BoundedAtlasAllocator {
    capacity: u32,
    /// Slot state is created lazily. A serialized capacity must never cause a
    /// source-sized host allocation before any page is resident.
    slots: BTreeMap<u32, SlotState>,
    free: BTreeSet<u32>,
    next_unused: u32,
    allocated: u32,
}

impl BoundedAtlasAllocator {
    pub fn new(capacity: u32) -> Result<Self, AtlasAllocatorError> {
        if capacity == 0 {
            return Err(AtlasAllocatorError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            slots: BTreeMap::new(),
            free: BTreeSet::new(),
            next_unused: 0,
            allocated: 0,
        })
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn allocated(&self) -> u32 {
        self.allocated
    }

    pub fn allocate(&mut self, page_id: LodPageId) -> Result<AtlasSlot, AtlasAllocatorError> {
        if !page_id.is_valid() {
            return Err(AtlasAllocatorError::InvalidPageId);
        }
        let index = if let Some(index) = self.free.pop_first() {
            index
        } else if self.next_unused < self.capacity {
            let index = self.next_unused;
            self.next_unused += 1;
            index
        } else {
            return Err(AtlasAllocatorError::Full);
        };
        let state = self.slots.entry(index).or_insert(SlotState {
            generation: 0,
            page_id: None,
        });
        debug_assert!(state.page_id.is_none());
        state.generation = next_generation(state.generation);
        state.page_id = Some(page_id);
        self.allocated += 1;
        Ok(AtlasSlot {
            index,
            generation: state.generation,
        })
    }

    pub fn free(&mut self, slot: AtlasSlot) -> Result<LodPageId, AtlasAllocatorError> {
        let state = self
            .slots
            .get_mut(&slot.index)
            .ok_or(AtlasAllocatorError::InvalidIndex(slot.index))?;
        if state.generation != slot.generation || state.page_id.is_none() {
            return Err(AtlasAllocatorError::StaleGeneration(slot));
        }
        let page_id = state.page_id.take().expect("checked occupied slot");
        self.free.insert(slot.index);
        self.allocated -= 1;
        Ok(page_id)
    }

    pub fn is_current(&self, slot: AtlasSlot) -> bool {
        self.slots
            .get(&slot.index)
            .is_some_and(|state| state.generation == slot.generation && state.page_id.is_some())
    }

    pub fn page(&self, slot: AtlasSlot) -> Option<LodPageId> {
        self.slots.get(&slot.index).and_then(|state| {
            (state.generation == slot.generation)
                .then_some(state.page_id)
                .flatten()
        })
    }
}

fn next_generation(generation: u32) -> u32 {
    let next = generation.wrapping_add(1);
    if next == 0 { 1 } else { next }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtlasAllocatorError {
    ZeroCapacity,
    InvalidPageId,
    Full,
    InvalidIndex(u32),
    StaleGeneration(AtlasSlot),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageCacheLimits {
    pub max_pages: u32,
    pub max_bytes: u64,
    pub max_gaussians: u64,
}

impl From<&LodBudgets> for PageCacheLimits {
    fn from(budgets: &LodBudgets) -> Self {
        Self {
            max_pages: budgets.max_resident_pages,
            max_bytes: budgets.max_resident_bytes,
            max_gaussians: budgets.max_resident_gaussians,
        }
    }
}

impl PageCacheLimits {
    pub fn validate(self) -> Result<Self, PageCacheError> {
        if self.max_pages == 0 || self.max_bytes == 0 || self.max_gaussians == 0 {
            Err(PageCacheError::ZeroLimit)
        } else {
            Ok(self)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentPage {
    pub page_id: LodPageId,
    pub slot: AtlasSlot,
    pub byte_len: u64,
    pub gaussian_count: u64,
    pub last_used_epoch: u64,
    /// Explicit holds, normally visible ancestors used as fallbacks.
    pub pin_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheInsert {
    pub slot: AtlasSlot,
    pub evicted: Vec<LodPageId>,
    pub already_resident: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheResolution {
    Exact {
        page_id: LodPageId,
        slot: AtlasSlot,
    },
    Ancestor {
        requested: LodPageId,
        page_id: LodPageId,
        slot: AtlasSlot,
    },
    Missing {
        requested: LodPageId,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PageCacheStats {
    pub resident_pages: u32,
    pub resident_bytes: u64,
    pub resident_gaussians: u64,
    pub pinned_pages: u32,
}

#[derive(Clone, Debug)]
pub struct LodPageCache {
    limits: PageCacheLimits,
    atlas: BoundedAtlasAllocator,
    entries: BTreeMap<LodPageId, ResidentPage>,
    resident_bytes: u64,
    resident_gaussians: u64,
}

impl LodPageCache {
    pub fn new(limits: PageCacheLimits) -> Result<Self, PageCacheError> {
        let limits = limits.validate()?;
        let atlas =
            BoundedAtlasAllocator::new(limits.max_pages).map_err(PageCacheError::Allocator)?;
        Ok(Self {
            limits,
            atlas,
            entries: BTreeMap::new(),
            resident_bytes: 0,
            resident_gaussians: 0,
        })
    }

    pub fn limits(&self) -> PageCacheLimits {
        self.limits
    }

    pub fn get(&self, page_id: LodPageId) -> Option<&ResidentPage> {
        self.entries.get(&page_id)
    }

    pub fn contains(&self, page_id: LodPageId) -> bool {
        self.entries.contains_key(&page_id)
    }

    pub fn is_slot_current(&self, slot: AtlasSlot) -> bool {
        self.atlas.is_current(slot)
    }

    pub fn stats(&self) -> PageCacheStats {
        PageCacheStats {
            resident_pages: self.entries.len().try_into().unwrap_or(u32::MAX),
            resident_bytes: self.resident_bytes,
            resident_gaussians: self.resident_gaussians,
            pinned_pages: self
                .entries
                .values()
                .filter(|entry| entry.pin_count > 0)
                .count()
                .try_into()
                .unwrap_or(u32::MAX),
        }
    }

    /// Commits a page after upload. Evictions are decided before mutation, so a
    /// failed insertion never partially flushes the current working set.
    pub fn insert(
        &mut self,
        page_id: LodPageId,
        byte_len: u64,
        gaussian_count: u64,
        epoch: u64,
    ) -> Result<CacheInsert, PageCacheError> {
        if !page_id.is_valid() {
            return Err(PageCacheError::InvalidPageId);
        }
        if byte_len == 0 || gaussian_count == 0 {
            return Err(PageCacheError::EmptyPage(page_id));
        }
        if byte_len > self.limits.max_bytes || gaussian_count > self.limits.max_gaussians {
            return Err(PageCacheError::PageExceedsLimits(page_id));
        }
        if let Some(entry) = self.entries.get_mut(&page_id) {
            if entry.byte_len != byte_len || entry.gaussian_count != gaussian_count {
                return Err(PageCacheError::MetadataMismatch(page_id));
            }
            entry.last_used_epoch = epoch;
            return Ok(CacheInsert {
                slot: entry.slot,
                evicted: Vec::new(),
                already_resident: true,
            });
        }

        let mut pages = self.entries.len() as u64 + 1;
        let mut bytes = self
            .resident_bytes
            .checked_add(byte_len)
            .ok_or(PageCacheError::CountOverflow)?;
        let mut gaussians = self
            .resident_gaussians
            .checked_add(gaussian_count)
            .ok_or(PageCacheError::CountOverflow)?;

        let mut candidates: Vec<_> = self
            .entries
            .values()
            .filter(|entry| entry.pin_count == 0)
            .copied()
            .collect();
        candidates.sort_by_key(|entry| (entry.last_used_epoch, entry.page_id));
        let mut victims = Vec::new();
        for candidate in candidates {
            if pages <= u64::from(self.limits.max_pages)
                && bytes <= self.limits.max_bytes
                && gaussians <= self.limits.max_gaussians
            {
                break;
            }
            pages -= 1;
            bytes -= candidate.byte_len;
            gaussians -= candidate.gaussian_count;
            victims.push(candidate.page_id);
        }
        if pages > u64::from(self.limits.max_pages)
            || bytes > self.limits.max_bytes
            || gaussians > self.limits.max_gaussians
        {
            return Err(PageCacheError::InsufficientEvictableCapacity);
        }

        for &victim in &victims {
            self.remove_unchecked(victim)?;
        }
        let slot = self
            .atlas
            .allocate(page_id)
            .map_err(PageCacheError::Allocator)?;
        self.entries.insert(
            page_id,
            ResidentPage {
                page_id,
                slot,
                byte_len,
                gaussian_count,
                last_used_epoch: epoch,
                pin_count: 0,
            },
        );
        self.resident_bytes = self
            .resident_bytes
            .checked_add(byte_len)
            .ok_or(PageCacheError::CountOverflow)?;
        self.resident_gaussians = self
            .resident_gaussians
            .checked_add(gaussian_count)
            .ok_or(PageCacheError::CountOverflow)?;
        Ok(CacheInsert {
            slot,
            evicted: victims,
            already_resident: false,
        })
    }

    pub fn touch(&mut self, page_id: LodPageId, epoch: u64) -> bool {
        if let Some(entry) = self.entries.get_mut(&page_id) {
            entry.last_used_epoch = epoch;
            true
        } else {
            false
        }
    }

    /// Holds a page against eviction while it is the visible ancestor fallback.
    pub fn pin_fallback(&mut self, page_id: LodPageId) -> Result<AtlasSlot, PageCacheError> {
        let entry = self
            .entries
            .get_mut(&page_id)
            .ok_or(PageCacheError::NotResident(page_id))?;
        entry.pin_count = entry
            .pin_count
            .checked_add(1)
            .ok_or(PageCacheError::CountOverflow)?;
        Ok(entry.slot)
    }

    pub fn unpin_fallback(&mut self, page_id: LodPageId) -> Result<(), PageCacheError> {
        let entry = self
            .entries
            .get_mut(&page_id)
            .ok_or(PageCacheError::NotResident(page_id))?;
        if entry.pin_count == 0 {
            return Err(PageCacheError::NotPinned(page_id));
        }
        entry.pin_count -= 1;
        Ok(())
    }

    pub fn remove(&mut self, page_id: LodPageId) -> Result<ResidentPage, PageCacheError> {
        if self
            .entries
            .get(&page_id)
            .is_some_and(|entry| entry.pin_count > 0)
        {
            return Err(PageCacheError::Pinned(page_id));
        }
        self.remove_unchecked(page_id)
    }

    /// Resolves exact data or the first resident ancestor supplied nearest-first.
    pub fn resolve_with_ancestors(
        &self,
        requested: LodPageId,
        ancestors_nearest_first: impl IntoIterator<Item = LodPageId>,
    ) -> CacheResolution {
        if let Some(entry) = self.entries.get(&requested) {
            return CacheResolution::Exact {
                page_id: requested,
                slot: entry.slot,
            };
        }
        for ancestor in ancestors_nearest_first {
            if let Some(entry) = self.entries.get(&ancestor) {
                return CacheResolution::Ancestor {
                    requested,
                    page_id: ancestor,
                    slot: entry.slot,
                };
            }
        }
        CacheResolution::Missing { requested }
    }

    fn remove_unchecked(&mut self, page_id: LodPageId) -> Result<ResidentPage, PageCacheError> {
        let entry = self
            .entries
            .remove(&page_id)
            .ok_or(PageCacheError::NotResident(page_id))?;
        let freed_page = self
            .atlas
            .free(entry.slot)
            .map_err(PageCacheError::Allocator)?;
        debug_assert_eq!(freed_page, page_id);
        self.resident_bytes -= entry.byte_len;
        self.resident_gaussians -= entry.gaussian_count;
        Ok(entry)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageCacheError {
    ZeroLimit,
    InvalidPageId,
    EmptyPage(LodPageId),
    PageExceedsLimits(LodPageId),
    MetadataMismatch(LodPageId),
    InsufficientEvictableCapacity,
    NotResident(LodPageId),
    Pinned(LodPageId),
    NotPinned(LodPageId),
    CountOverflow,
    Allocator(AtlasAllocatorError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(pages: u32) -> PageCacheLimits {
        PageCacheLimits {
            max_pages: pages,
            max_bytes: u64::from(pages) * 100,
            max_gaussians: u64::from(pages) * 10,
        }
    }

    #[test]
    fn atlas_slots_reject_stale_generations_after_reuse() {
        let mut atlas = BoundedAtlasAllocator::new(1).unwrap();
        let first = atlas.allocate(LodPageId(1)).unwrap();
        assert!(atlas.is_current(first));
        atlas.free(first).unwrap();
        assert!(!atlas.is_current(first));

        let second = atlas.allocate(LodPageId(2)).unwrap();
        assert_eq!(first.index, second.index);
        assert_ne!(first.generation, second.generation);
        assert!(matches!(
            atlas.free(first),
            Err(AtlasAllocatorError::StaleGeneration(_))
        ));
        assert_eq!(atlas.page(second), Some(LodPageId(2)));
    }

    #[test]
    fn atlas_capacity_is_lazy_even_at_u32_max() {
        let mut atlas = BoundedAtlasAllocator::new(u32::MAX).unwrap();
        assert_eq!(atlas.capacity(), u32::MAX);
        assert_eq!(atlas.allocated(), 0);
        assert!(atlas.slots.is_empty());
        assert!(atlas.free.is_empty());

        let slot = atlas.allocate(LodPageId(1)).unwrap();
        assert_eq!(slot.index, 0);
        assert_eq!(atlas.allocated(), 1);
        assert_eq!(atlas.slots.len(), 1);
        atlas.free(slot).unwrap();
        assert_eq!(atlas.allocated(), 0);
        assert_eq!(atlas.slots.len(), 1);
        assert_eq!(atlas.free.len(), 1);
    }

    #[test]
    fn cache_evicts_lru_then_stable_page_id() {
        let mut cache = LodPageCache::new(limits(2)).unwrap();
        let first = cache.insert(LodPageId(2), 50, 5, 1).unwrap().slot;
        cache.insert(LodPageId(1), 50, 5, 1).unwrap();
        let result = cache.insert(LodPageId(3), 50, 5, 2).unwrap();
        assert_eq!(result.evicted, vec![LodPageId(1)]);
        assert!(cache.contains(LodPageId(2)));
        assert!(!cache.contains(LodPageId(1)));
        assert!(cache.is_slot_current(first));
    }

    #[test]
    fn touch_changes_lru_order() {
        let mut cache = LodPageCache::new(limits(2)).unwrap();
        cache.insert(LodPageId(1), 50, 5, 1).unwrap();
        cache.insert(LodPageId(2), 50, 5, 2).unwrap();
        cache.touch(LodPageId(1), 3);
        let result = cache.insert(LodPageId(3), 50, 5, 4).unwrap();
        assert_eq!(result.evicted, vec![LodPageId(2)]);
    }

    #[test]
    fn pinned_fallback_survives_and_failed_insert_is_atomic() {
        let mut cache = LodPageCache::new(limits(1)).unwrap();
        let root_slot = cache.insert(LodPageId(10), 50, 5, 1).unwrap().slot;
        cache.pin_fallback(LodPageId(10)).unwrap();
        assert_eq!(
            cache.insert(LodPageId(1), 50, 5, 2),
            Err(PageCacheError::InsufficientEvictableCapacity)
        );
        assert!(cache.contains(LodPageId(10)));
        assert!(cache.is_slot_current(root_slot));

        cache.unpin_fallback(LodPageId(10)).unwrap();
        let inserted = cache.insert(LodPageId(1), 50, 5, 2).unwrap();
        assert_eq!(inserted.evicted, vec![LodPageId(10)]);
        assert!(!cache.is_slot_current(root_slot));
    }

    #[test]
    fn resolution_prefers_exact_then_nearest_ancestor() {
        let mut cache = LodPageCache::new(limits(3)).unwrap();
        let root = cache.insert(LodPageId(10), 50, 5, 1).unwrap().slot;
        let parent = cache.insert(LodPageId(1), 50, 5, 1).unwrap().slot;
        assert_eq!(
            cache.resolve_with_ancestors(LodPageId(2), [LodPageId(1), LodPageId(10)]),
            CacheResolution::Ancestor {
                requested: LodPageId(2),
                page_id: LodPageId(1),
                slot: parent,
            }
        );
        assert_eq!(
            cache.resolve_with_ancestors(LodPageId(10), []),
            CacheResolution::Exact {
                page_id: LodPageId(10),
                slot: root,
            }
        );
    }

    #[test]
    fn all_limits_participate_in_eviction() {
        let limits = PageCacheLimits {
            max_pages: 4,
            max_bytes: 100,
            max_gaussians: 10,
        };
        let mut cache = LodPageCache::new(limits).unwrap();
        cache.insert(LodPageId(1), 60, 2, 1).unwrap();
        let result = cache.insert(LodPageId(2), 50, 2, 2).unwrap();
        assert_eq!(result.evicted, vec![LodPageId(1)]);
        assert_eq!(cache.stats().resident_bytes, 50);

        let result = cache.insert(LodPageId(3), 10, 9, 3).unwrap();
        assert_eq!(result.evicted, vec![LodPageId(2)]);
        assert_eq!(cache.stats().resident_gaussians, 9);
    }
}
