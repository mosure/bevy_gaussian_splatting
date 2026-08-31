//! Bounded GPU subrange uploads for streamed 3D Gaussian atlas slots.
//!
//! Ordinary [`PlanarGaussian3d`] asset changes continue through Bevy's
//! [`RenderAsset`](bevy::render::render_asset::RenderAsset) preparation path.
//! The streaming bridge owns fixed-size atlas assets, however, and can update a
//! single physical slot without cloning and recreating the complete GPU asset.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    mem::size_of,
    num::{NonZeroU32, NonZeroU64},
    sync::{
        Arc, RwLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use bevy::{
    asset::{AssetId, Assets, UntypedAssetId},
    ecs::system::SystemParam,
    prelude::*,
    render::{
        ExtractSchedule, GpuResourceAppExt, MainWorld, Render, RenderApp, RenderSystems,
        render_asset::{RenderAssets, prepare_assets},
        render_resource::{
            BufferDescriptor, BufferInitDescriptor, BufferUsages, CommandEncoderDescriptor,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use bevy_interleave::prelude::Planar;

use bytemuck::Pod;
#[cfg(feature = "precompute_covariance_3d")]
use bytemuck::Zeroable;

use crate::{
    gaussian::{
        f32::{PositionVisibility, Rotation, ScaleOpacity},
        formats::planar_3d::{PlanarGaussian3d, PlanarStorageGaussian3d},
    },
    material::spherical_harmonics::SphericalHarmonicCoefficients,
    stream::cache::AtlasSlot,
};

#[cfg(feature = "precompute_covariance_3d")]
use crate::gaussian::f32::Covariance3dOpacity;

#[cfg(any(test, lod_render_path))]
use crate::stream::runtime::LodPhysicalRange;

/// A main-world request to upload the final CPU contents of one physical
/// atlas slot.
///
/// Requests contain no Gaussian payload. Extraction snapshots the final atlas
/// contents after all main-world systems have run, so an ordinary asset edit
/// that happens earlier in the frame cannot be overwritten by an older page
/// payload. Repeated writes to the same physical slot are coalesced; the last
/// request (and therefore its generation) wins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodAtlasSlotUpload {
    pub atlas: AssetId<PlanarGaussian3d>,
    pub slot: AtlasSlot,
    pub gaussians_per_slot: u32,
}

impl LodAtlasSlotUpload {
    fn physical_start(self) -> Result<u32, LodAtlasUploadError> {
        self.slot
            .index
            .checked_mul(self.gaussians_per_slot)
            .ok_or(LodAtlasUploadError::AddressOverflow)
    }

    fn physical_end(self) -> Result<u32, LodAtlasUploadError> {
        self.physical_start()?
            .checked_add(self.gaussians_per_slot)
            .ok_or(LodAtlasUploadError::AddressOverflow)
    }

    fn validate_address(self) -> Result<(), LodAtlasUploadError> {
        if self.gaussians_per_slot == 0 {
            return Err(LodAtlasUploadError::ZeroSlotStride);
        }
        self.physical_end()?;
        Ok(())
    }

    fn validate_resident(self) -> Result<(), LodAtlasUploadError> {
        self.validate_address()?;
        if self.slot.generation == 0 {
            return Err(LodAtlasUploadError::ZeroGeneration);
        }
        Ok(())
    }
}

/// Coalescing main-world queue for bridge- or package-owned atlas writes.
#[derive(Resource, Default, Debug)]
pub struct LodAtlasUploadQueue {
    slots: HashMap<(AssetId<PlanarGaussian3d>, u32), LodAtlasSlotUpload>,
}

impl LodAtlasUploadQueue {
    /// Queues one complete fixed-stride physical slot. The atlas is sampled at
    /// extraction time; callers must mutate its CPU mirror before enqueueing.
    pub fn enqueue_slot(
        &mut self,
        atlas: AssetId<PlanarGaussian3d>,
        slot: AtlasSlot,
        gaussians_per_slot: u32,
    ) -> Result<(), LodAtlasUploadError> {
        let upload = LodAtlasSlotUpload {
            atlas,
            slot,
            gaussians_per_slot,
        };
        upload.validate_resident()?;
        self.slots.insert((atlas, slot.index), upload);
        Ok(())
    }

    /// Queues every physical slot without publishing allocator-generation
    /// proofs. This is the bounded fallback for an in-place source mutation:
    /// source-covered and padded slots may all have changed, including slots
    /// that were never resident in the current hierarchy cut.
    pub fn enqueue_complete_atlas(
        &mut self,
        atlas: AssetId<PlanarGaussian3d>,
        physical_gaussians: u32,
        gaussians_per_slot: u32,
    ) -> Result<(), LodAtlasUploadError> {
        if gaussians_per_slot == 0 {
            return Err(LodAtlasUploadError::ZeroSlotStride);
        }
        if physical_gaussians == 0 || !physical_gaussians.is_multiple_of(gaussians_per_slot) {
            return Err(LodAtlasUploadError::InvalidAtlasLength {
                physical_gaussians,
                gaussians_per_slot,
            });
        }
        let slot_count = physical_gaussians / gaussians_per_slot;
        let reserved_slots = usize::try_from(slot_count)
            .map_err(|_| LodAtlasUploadError::QueueAllocationFailed { slot_count })?;
        self.slots
            .try_reserve(reserved_slots)
            .map_err(|_| LodAtlasUploadError::QueueAllocationFailed { slot_count })?;
        for index in 0..slot_count {
            self.enqueue_cleared_slot(atlas, index, gaussians_per_slot)?;
        }
        Ok(())
    }

    /// Queues one physical slot while invalidating (rather than publishing) a
    /// residency generation. This is used when a cut clears a formerly active
    /// slot and has no allocator generation that may safely be rendered.
    pub fn enqueue_cleared_slot(
        &mut self,
        atlas: AssetId<PlanarGaussian3d>,
        slot_index: u32,
        gaussians_per_slot: u32,
    ) -> Result<(), LodAtlasUploadError> {
        let upload = LodAtlasSlotUpload {
            atlas,
            slot: AtlasSlot {
                index: slot_index,
                // Zero explicitly means "invalidate only". Future ACTIVE
                // frontiers still require an allocator-issued generation.
                generation: 0,
            },
            gaussians_per_slot,
        };
        upload.validate_address()?;
        self.slots.insert((atlas, slot_index), upload);
        Ok(())
    }

    pub fn queued_slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Read-only inspection for integration tests and package orchestration.
    pub fn queued_slots(&self) -> impl Iterator<Item = LodAtlasSlotUpload> + '_ {
        self.slots.values().copied()
    }

    pub(crate) fn remove_atlas(&mut self, atlas: AssetId<PlanarGaussian3d>) {
        self.slots
            .retain(|(queued_atlas, _), _| *queued_atlas != atlas);
    }

    /// Removes one queued write only when both its physical key and allocator
    /// generation still match the canceled transaction.
    ///
    /// A newer page may reuse the same slot index before stale cleanup runs.
    /// In that case its descriptor must survive so the new generation can land.
    #[cfg(test)]
    pub(crate) fn remove_slot(
        &mut self,
        atlas: AssetId<PlanarGaussian3d>,
        slot: AtlasSlot,
    ) -> bool {
        let key = (atlas, slot.index);
        if self
            .slots
            .get(&key)
            .is_some_and(|queued| queued.slot == slot)
        {
            self.slots.remove(&key);
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
struct LodTransientAtlasTicketInner {
    generation: AtomicU64,
    ready_generation: AtomicU64,
    failed_generation: AtomicU64,
    canceled: AtomicBool,
}

/// Shared allocation-generation proof for a CPU-only transient atlas.
///
/// Readiness proves only that bounded GPU storage exists for the current
/// generation. Individual atlas slots remain unusable until an ordinary page
/// upload publishes its allocator generation through [`LodAtlasGpuGenerations`].
#[derive(Clone, Debug)]
pub(crate) struct LodTransientAtlasTicket(Arc<LodTransientAtlasTicketInner>);

impl Default for LodTransientAtlasTicket {
    fn default() -> Self {
        Self(Arc::new(LodTransientAtlasTicketInner {
            generation: AtomicU64::new(1),
            ready_generation: AtomicU64::new(0),
            failed_generation: AtomicU64::new(0),
            canceled: AtomicBool::new(false),
        }))
    }
}

impl LodTransientAtlasTicket {
    pub(crate) fn generation(&self) -> u64 {
        self.0.generation.load(Ordering::Acquire)
    }

    pub(crate) fn is_ready(&self) -> bool {
        let generation = self.generation();
        !self.is_canceled() && self.0.ready_generation.load(Ordering::Acquire) == generation
    }

    pub(crate) fn is_failed(&self) -> bool {
        let generation = self.generation();
        !self.is_canceled() && self.0.failed_generation.load(Ordering::Acquire) == generation
    }

    fn is_canceled(&self) -> bool {
        self.0.canceled.load(Ordering::Acquire)
    }

    pub(crate) fn acknowledge(&self, generation: u64) -> bool {
        if self.is_canceled() || self.generation() != generation {
            return false;
        }
        self.0.ready_generation.store(generation, Ordering::Release);
        true
    }

    fn fail(&self, generation: u64) {
        if !self.is_canceled() && self.generation() == generation {
            self.0
                .failed_generation
                .store(generation, Ordering::Release);
        }
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn fail_current_for_test(&self) {
        self.fail(self.generation());
    }

    fn request_reupload(&self) -> u64 {
        if self.is_canceled() {
            return self.generation();
        }
        let previous = self
            .0
            .generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |generation| {
                Some(generation.wrapping_add(1).max(1))
            })
            .expect("generation update closure always returns a value");
        previous.wrapping_add(1).max(1)
    }

    #[cfg(test)]
    pub(crate) fn request_reupload_for_test(&self) -> u64 {
        self.request_reupload()
    }

    fn cancel(&self) {
        self.0.canceled.store(true, Ordering::Release);
    }
}

/// Strong main-world owner of the worker-produced planar atlas. It is never
/// inserted into [`Assets`], so Bevy's generic RenderAsset extraction cannot
/// clone or upload the complete allocation in one frame.
pub(crate) struct LodTransientAtlas {
    physical_gaussians: u32,
    // Kept only for the legacy dense constructor while bridge callers migrate
    // to `new_empty` + `write_slot`. Empty transient atlases never reserve
    // capacity in these planes.
    planes: Arc<RwLock<PlanarGaussian3d>>,
    slots: Arc<RwLock<HashMap<u32, PlanarGaussian3d>>>,
    ticket: LodTransientAtlasTicket,
}

impl LodTransientAtlas {
    /// Wraps an already materialized bounded atlas.
    ///
    /// New transient bridges should use [`Self::new_empty`] so cold
    /// initialization does not allocate or zero every physical slot.
    #[cfg(test)]
    pub(crate) fn new(planes: PlanarGaussian3d) -> Self {
        let physical_gaussians = planes
            .len()
            .try_into()
            .expect("bounded transient atlas length fits u32");
        Self {
            physical_gaussians,
            planes: Arc::new(RwLock::new(planes)),
            slots: Arc::new(RwLock::new(HashMap::new())),
            ticket: default(),
        }
    }

    /// Creates a fixed-size GPU atlas owner with no CPU Gaussian payload.
    ///
    /// CPU memory grows only when [`Self::write_slot`] materializes a page and
    /// is bounded by the number of distinct physical slots written. The render
    /// world still allocates `physical_gaussians` entries on the GPU.
    pub(crate) fn new_empty(physical_gaussians: u32) -> Result<Self, LodAtlasUploadError> {
        if physical_gaussians == 0 {
            return Err(LodAtlasUploadError::InvalidAtlasLength {
                physical_gaussians,
                gaussians_per_slot: 1,
            });
        }
        Ok(Self {
            physical_gaussians,
            planes: Arc::new(RwLock::new(PlanarGaussian3d::default())),
            slots: Arc::new(RwLock::new(HashMap::new())),
            ticket: default(),
        })
    }

    #[cfg(test)]
    pub(crate) fn physical_gaussians(&self) -> u32 {
        self.physical_gaussians
    }

    /// Replaces one complete, padded physical slot in the sparse CPU staging
    /// cache. No other slot is allocated or touched.
    pub(crate) fn write_slot(
        &self,
        slot_index: u32,
        gaussians_per_slot: u32,
        planes: PlanarGaussian3d,
    ) -> Result<(), LodAtlasUploadError> {
        validate_transient_slot(
            self.physical_gaussians,
            slot_index,
            gaussians_per_slot,
            &planes,
        )?;
        self.slots
            .write()
            .map_err(|_| LodAtlasUploadError::TransientAtlasLockPoisoned)?
            .insert(slot_index, planes);
        Ok(())
    }

    /// Drops a CPU staging payload after it can no longer be queued for GPU
    /// upload. GPU residency proofs are managed separately by the uploader.
    #[cfg(test)]
    pub(crate) fn discard_slot(&self, slot_index: u32) -> Result<bool, LodAtlasUploadError> {
        Ok(self
            .slots
            .write()
            .map_err(|_| LodAtlasUploadError::TransientAtlasLockPoisoned)?
            .remove(&slot_index)
            .is_some())
    }

    #[cfg(test)]
    pub(crate) fn snapshot_slot(
        &self,
        descriptor: LodAtlasSlotUpload,
    ) -> Result<PlanarGaussian3d, LodAtlasUploadError> {
        snapshot_transient_slot(
            self.physical_gaussians,
            &self.planes,
            &self.slots,
            descriptor,
        )
    }

    #[cfg(test)]
    pub(crate) fn materialized_slot_count(&self) -> Result<usize, LodAtlasUploadError> {
        Ok(self
            .slots
            .read()
            .map_err(|_| LodAtlasUploadError::TransientAtlasLockPoisoned)?
            .len())
    }

    #[cfg(test)]
    pub(crate) fn materialized_gaussian_count(&self) -> Result<usize, LodAtlasUploadError> {
        self.slots
            .read()
            .map_err(|_| LodAtlasUploadError::TransientAtlasLockPoisoned)?
            .values()
            .try_fold(0_usize, |total, planes| {
                total
                    .checked_add(planes.len())
                    .ok_or(LodAtlasUploadError::AddressOverflow)
            })
    }

    /// Legacy mutable dense mirror access. Empty transient atlases deliberately
    /// return zero-length planes; production page writes use `write_slot`.
    #[cfg(test)]
    pub(crate) fn planes(&self) -> Arc<RwLock<PlanarGaussian3d>> {
        Arc::clone(&self.planes)
    }

    pub(crate) fn ticket(&self) -> &LodTransientAtlasTicket {
        &self.ticket
    }
}

impl Drop for LodTransientAtlas {
    fn drop(&mut self) {
        self.ticket.cancel();
    }
}

fn validate_transient_slot(
    physical_gaussians: u32,
    slot_index: u32,
    gaussians_per_slot: u32,
    planes: &PlanarGaussian3d,
) -> Result<(), LodAtlasUploadError> {
    let descriptor = LodAtlasSlotUpload {
        atlas: AssetId::default(),
        slot: AtlasSlot {
            index: slot_index,
            // CPU staging does not publish residency; use a non-zero value
            // solely to share the checked fixed-stride address validation.
            generation: 1,
        },
        gaussians_per_slot,
    };
    descriptor.validate_address()?;
    let end = descriptor.physical_end()?;
    if end > physical_gaussians {
        return Err(LodAtlasUploadError::SlotOutOfRange {
            start: u64::from(descriptor.physical_start()?),
            end: u64::from(end),
            atlas_len: u64::from(physical_gaussians),
        });
    }
    let expected =
        usize::try_from(gaussians_per_slot).map_err(|_| LodAtlasUploadError::AddressOverflow)?;
    if planes.len() != expected {
        return Err(LodAtlasUploadError::TransientSlotLengthMismatch {
            slot_index,
            expected: gaussians_per_slot,
            actual: planes
                .len()
                .try_into()
                .map_err(|_| LodAtlasUploadError::AddressOverflow)?,
        });
    }
    if planes.spherical_harmonic.len() != planes.len()
        || planes.rotation.len() != planes.len()
        || planes.scale_opacity.len() != planes.len()
    {
        return Err(LodAtlasUploadError::InconsistentPlaneLengths);
    }
    Ok(())
}

fn snapshot_transient_slot(
    physical_gaussians: u32,
    dense_planes: &Arc<RwLock<PlanarGaussian3d>>,
    slots: &Arc<RwLock<HashMap<u32, PlanarGaussian3d>>>,
    descriptor: LodAtlasSlotUpload,
) -> Result<PlanarGaussian3d, LodAtlasUploadError> {
    descriptor.validate_address()?;
    let end = descriptor.physical_end()?;
    if end > physical_gaussians {
        return Err(LodAtlasUploadError::SlotOutOfRange {
            start: u64::from(descriptor.physical_start()?),
            end: u64::from(end),
            atlas_len: u64::from(physical_gaussians),
        });
    }
    if let Some(planes) = slots
        .read()
        .map_err(|_| LodAtlasUploadError::TransientAtlasLockPoisoned)?
        .get(&descriptor.slot.index)
    {
        validate_transient_slot(
            physical_gaussians,
            descriptor.slot.index,
            descriptor.gaussians_per_slot,
            planes,
        )?;
        return Ok(planes.clone());
    }

    let dense = dense_planes
        .read()
        .map_err(|_| LodAtlasUploadError::TransientAtlasLockPoisoned)?;
    if dense.len() == physical_gaussians as usize {
        return snapshot_slot(Some(&dense), descriptor, None).map(|upload| upload.planes);
    }
    Err(LodAtlasUploadError::MissingTransientAtlasSlot {
        slot_index: descriptor.slot.index,
    })
}

struct LodTransientAtlasRegistryEntry {
    source: AssetId<PlanarGaussian3d>,
    physical_gaussians: u32,
    gaussians_per_slot: u32,
    planes: Weak<RwLock<PlanarGaussian3d>>,
    slots: Weak<RwLock<HashMap<u32, PlanarGaussian3d>>>,
    ticket: Weak<LodTransientAtlasTicketInner>,
}

#[derive(Clone)]
struct LiveLodTransientAtlas {
    source: AssetId<PlanarGaussian3d>,
    physical_gaussians: u32,
    gaussians_per_slot: u32,
    planes: Arc<RwLock<PlanarGaussian3d>>,
    slots: Arc<RwLock<HashMap<u32, PlanarGaussian3d>>>,
    ticket: LodTransientAtlasTicket,
    generation: u64,
}

impl LiveLodTransientAtlas {
    fn snapshot_slot(
        &self,
        descriptor: LodAtlasSlotUpload,
    ) -> Result<PlanarGaussian3d, LodAtlasUploadError> {
        snapshot_transient_slot(
            self.physical_gaussians,
            &self.planes,
            &self.slots,
            descriptor,
        )
    }
}

/// Main-world registry for reserved-handle atlases whose large CPU allocation
/// stays outside Bevy's generic asset extraction path.
#[derive(Resource, Default)]
pub(crate) struct LodTransientAtlasRegistry {
    entries: HashMap<AssetId<PlanarGaussian3d>, LodTransientAtlasRegistryEntry>,
}

impl LodTransientAtlasRegistry {
    /// Returns the fixed GPU length of one live transient 3D atlas.
    ///
    /// Transient atlases deliberately have no dense main-world asset, so
    /// consumers that size per-cloud auxiliary storage must use this bounded
    /// allocation metadata instead. Dead, canceled, and non-3D handles are not
    /// reported as live allocations.
    pub(crate) fn physical_gaussians(&self, atlas: UntypedAssetId) -> Option<u32> {
        let atlas = atlas.try_typed::<PlanarGaussian3d>().ok()?;
        let entry = self.entries.get(&atlas)?;
        let ticket = entry.ticket.upgrade().map(LodTransientAtlasTicket)?;
        if ticket.is_canceled()
            || entry.planes.upgrade().is_none()
            || entry.slots.upgrade().is_none()
        {
            return None;
        }
        Some(entry.physical_gaussians)
    }

    pub(crate) fn register(
        &mut self,
        atlas: AssetId<PlanarGaussian3d>,
        source: AssetId<PlanarGaussian3d>,
        _source_gaussians: u32,
        gaussians_per_slot: u32,
        owner: &LodTransientAtlas,
    ) -> Result<(), LodAtlasUploadError> {
        let physical_gaussians = owner.physical_gaussians;
        if gaussians_per_slot == 0
            || physical_gaussians == 0
            || !physical_gaussians.is_multiple_of(gaussians_per_slot)
        {
            return Err(LodAtlasUploadError::InvalidAtlasLength {
                physical_gaussians,
                gaussians_per_slot,
            });
        }
        self.entries.insert(
            atlas,
            LodTransientAtlasRegistryEntry {
                source,
                physical_gaussians,
                gaussians_per_slot,
                planes: Arc::downgrade(&owner.planes),
                slots: Arc::downgrade(&owner.slots),
                ticket: Arc::downgrade(&owner.ticket.0),
            },
        );
        Ok(())
    }

    /// Removes one reserved-handle atlas immediately during owner teardown.
    ///
    /// The upload queue is canceled separately by the orchestrator before its
    /// sparse payload owner is dropped, so no descriptor can outlive the weak
    /// registry entry that identifies its transient allocation.
    pub(crate) fn unregister(&mut self, atlas: AssetId<PlanarGaussian3d>) -> bool {
        self.entries.remove(&atlas).is_some()
    }

    #[cfg(all(
        test,
        not(target_arch = "wasm32"),
        feature = "sort_radix",
        not(feature = "buffer_texture")
    ))]
    pub(crate) fn contains(&self, atlas: AssetId<PlanarGaussian3d>) -> bool {
        self.entries.contains_key(&atlas)
    }

    /// Prunes dead transient registrations without scheduling atlas-wide work.
    ///
    /// GPU allocation is published empty. Only actual resident page writes
    /// enter the bounded upload queue, so initialization cost is independent
    /// of both source size and physical atlas capacity.
    pub(crate) fn queue_pending_initialization(
        &mut self,
        uploads: &mut LodAtlasUploadQueue,
    ) -> Result<(), LodAtlasUploadError> {
        let mut stale = Vec::new();
        for (&atlas, entry) in &self.entries {
            let Some(ticket) = entry.ticket.upgrade().map(LodTransientAtlasTicket) else {
                stale.push(atlas);
                continue;
            };
            if ticket.is_canceled()
                || entry.planes.upgrade().is_none()
                || entry.slots.upgrade().is_none()
            {
                stale.push(atlas);
                continue;
            }
        }
        for atlas in stale {
            self.entries.remove(&atlas);
            uploads.remove_atlas(atlas);
        }
        Ok(())
    }

    fn live(&self) -> HashMap<AssetId<PlanarGaussian3d>, LiveLodTransientAtlas> {
        self.entries
            .iter()
            .filter_map(|(&atlas, entry)| {
                let planes = entry.planes.upgrade()?;
                let slots = entry.slots.upgrade()?;
                let ticket = entry.ticket.upgrade().map(LodTransientAtlasTicket)?;
                if ticket.is_canceled() {
                    return None;
                }
                Some((
                    atlas,
                    LiveLodTransientAtlas {
                        source: entry.source,
                        physical_gaussians: entry.physical_gaussians,
                        gaussians_per_slot: entry.gaussians_per_slot,
                        planes,
                        slots,
                        generation: ticket.generation(),
                        ticket,
                    },
                ))
            })
            .collect()
    }
}

/// Global canonical-atlas work admitted to one render frame.
///
/// This global bound is intentionally separate from each cloud's staging-step
/// bound. Multiple clouds may stage in the same application frame, while the
/// render bridge must keep their aggregate CPU snapshots and GPU work finite.
/// A physical slot remains atomic: if one slot is larger than the byte limit it
/// is deferred and reported through [`LodAtlasUploadBudgetStatus`].
#[derive(Resource, Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodAtlasUploadBudget {
    max_canonical_bytes_per_frame: NonZeroU64,
    max_slots_per_frame: NonZeroU32,
}

impl LodAtlasUploadBudget {
    pub const DEFAULT_MAX_CANONICAL_BYTES_PER_FRAME: u64 = 64 * 1024 * 1024;
    pub const DEFAULT_MAX_SLOTS_PER_FRAME: u32 = 256;

    pub const fn try_new(
        max_canonical_bytes_per_frame: u64,
        max_slots_per_frame: u32,
    ) -> Result<Self, LodAtlasUploadBudgetError> {
        let Some(max_canonical_bytes_per_frame) = NonZeroU64::new(max_canonical_bytes_per_frame)
        else {
            return Err(LodAtlasUploadBudgetError::ZeroCanonicalByteLimit);
        };
        let Some(max_slots_per_frame) = NonZeroU32::new(max_slots_per_frame) else {
            return Err(LodAtlasUploadBudgetError::ZeroSlotLimit);
        };
        Ok(Self {
            max_canonical_bytes_per_frame,
            max_slots_per_frame,
        })
    }

    pub const fn max_canonical_bytes_per_frame(self) -> u64 {
        self.max_canonical_bytes_per_frame.get()
    }

    pub const fn max_slots_per_frame(self) -> u32 {
        self.max_slots_per_frame.get()
    }

    pub fn set_max_canonical_bytes_per_frame(
        &mut self,
        value: u64,
    ) -> Result<(), LodAtlasUploadBudgetError> {
        self.max_canonical_bytes_per_frame =
            NonZeroU64::new(value).ok_or(LodAtlasUploadBudgetError::ZeroCanonicalByteLimit)?;
        Ok(())
    }

    pub fn set_max_slots_per_frame(&mut self, value: u32) -> Result<(), LodAtlasUploadBudgetError> {
        self.max_slots_per_frame =
            NonZeroU32::new(value).ok_or(LodAtlasUploadBudgetError::ZeroSlotLimit)?;
        Ok(())
    }
}

impl Default for LodAtlasUploadBudget {
    fn default() -> Self {
        Self::try_new(
            Self::DEFAULT_MAX_CANONICAL_BYTES_PER_FRAME,
            Self::DEFAULT_MAX_SLOTS_PER_FRAME,
        )
        .expect("default LoD atlas upload limits are non-zero")
    }
}

/// Typed budget fault observed by the deterministic render-world scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodAtlasUploadBudgetError {
    ZeroCanonicalByteLimit,
    ZeroSlotLimit,
    SlotCanonicalByteOverflow {
        atlas: AssetId<PlanarGaussian3d>,
        slot_index: u32,
    },
    SlotExceedsCanonicalByteLimit {
        atlas: AssetId<PlanarGaussian3d>,
        slot_index: u32,
        required: u64,
        limit: u64,
    },
}

impl std::fmt::Display for LodAtlasUploadBudgetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCanonicalByteLimit => {
                write!(
                    formatter,
                    "LoD atlas per-frame canonical byte limit is zero"
                )
            }
            Self::ZeroSlotLimit => write!(formatter, "LoD atlas per-frame slot limit is zero"),
            Self::SlotCanonicalByteOverflow { atlas, slot_index } => write!(
                formatter,
                "LoD atlas {atlas:?} slot {slot_index} canonical byte count overflowed"
            ),
            Self::SlotExceedsCanonicalByteLimit {
                atlas,
                slot_index,
                required,
                limit,
            } => write!(
                formatter,
                "LoD atlas {atlas:?} slot {slot_index} requires {required} canonical bytes, exceeding the per-frame limit {limit}"
            ),
        }
    }
}

impl std::error::Error for LodAtlasUploadBudgetError {}

/// Main-world status for configuration UIs and orchestration. Oversized
/// atomic slots remain queued and expose a typed error instead of disappearing
/// or bypassing the configured bound.
#[derive(Resource, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LodAtlasUploadBudgetStatus {
    last_error: Option<LodAtlasUploadBudgetError>,
}

impl LodAtlasUploadBudgetStatus {
    pub const fn last_error(self) -> Option<LodAtlasUploadBudgetError> {
        self.last_error
    }
}

#[derive(Debug)]
struct ExtractedLodAtlasSlotUpload {
    descriptor: LodAtlasSlotUpload,
    planes: PlanarGaussian3d,
    transient_generation: Option<u64>,
}

#[derive(Debug)]
struct CoalescedLodAtlasUpload {
    descriptors: Vec<LodAtlasSlotUpload>,
    planes: PlanarGaussian3d,
    transient_generation: Option<u64>,
}

impl CoalescedLodAtlasUpload {
    fn start(&self) -> Result<usize, LodAtlasUploadError> {
        self.descriptors
            .first()
            .ok_or(LodAtlasUploadError::EmptyCoalescedRange)?
            .physical_start()
            .map(|start| start as usize)
    }
}

#[cfg(feature = "precompute_covariance_3d")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LodCovarianceUploadRange {
    start: u32,
    count: u32,
}

#[cfg(feature = "precompute_covariance_3d")]
#[derive(Resource)]
struct LodCovariancePipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    max_workgroups_per_dimension: u32,
}

#[cfg(feature = "precompute_covariance_3d")]
impl FromWorld for LodCovariancePipeline {
    fn from_world(world: &mut World) -> Self {
        Self::new(world.resource::<RenderDevice>().wgpu_device())
    }
}

#[cfg(feature = "precompute_covariance_3d")]
impl LodCovariancePipeline {
    fn new(device: &wgpu::Device) -> Self {
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lod_atlas_covariance_layout"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                storage(3, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lod_atlas_covariance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lod_atlas_covariance_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("atlas_covariance.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lod_atlas_covariance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("derive_covariance"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            max_workgroups_per_dimension: device.limits().max_compute_workgroups_per_dimension,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_dispatch_buffers(
        &self,
        render_device: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        rotation: &wgpu::Buffer,
        scale_opacity: &wgpu::Buffer,
        covariance_3d_opacity: &wgpu::Buffer,
        ranges: &[LodCovarianceUploadRange],
    ) -> Result<u32, LodAtlasUploadError> {
        if ranges.is_empty() {
            return Ok(0);
        }
        let max_count = ranges.iter().map(|range| range.count).max().unwrap_or(0);
        let workgroups_x = max_count.div_ceil(64);
        if workgroups_x == 0 || workgroups_x > self.max_workgroups_per_dimension {
            return Err(LodAtlasUploadError::CovarianceDispatchLimit);
        }
        let chunk_size = usize::try_from(self.max_workgroups_per_dimension)
            .map_err(|_| LodAtlasUploadError::AddressOverflow)?;
        let mut prepared = Vec::new();
        prepared
            .try_reserve(ranges.len().div_ceil(chunk_size))
            .map_err(|_| LodAtlasUploadError::CoalescedAllocationFailed)?;
        for (chunk_index, chunk) in ranges.chunks(chunk_size).enumerate() {
            let range_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("lod_atlas_covariance_ranges"),
                contents: bytemuck::cast_slice(chunk),
                usage: BufferUsages::STORAGE,
            });
            let bind_group =
                render_device
                    .wgpu_device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("lod_atlas_covariance_bind_group"),
                        layout: &self.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: rotation.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: scale_opacity.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: covariance_3d_opacity.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: range_buffer.as_entire_binding(),
                            },
                        ],
                    });
            prepared.push((chunk_index, chunk.len() as u32, range_buffer, bind_group));
        }
        for (_, range_count, _, bind_group) in &prepared {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lod_atlas_covariance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, *range_count, 1);
        }
        u32::try_from(prepared.len()).map_err(|_| LodAtlasUploadError::AddressOverflow)
    }
}

#[derive(Debug)]
enum PendingLodAtlasSlotUpload {
    Ready(ExtractedLodAtlasSlotUpload),
}

type LodAtlasUploadKey = (AssetId<PlanarGaussian3d>, u32);

#[derive(Resource, Default, Debug)]
struct LodAtlasUploadScheduler {
    /// First cloud considered on the next admission pass. The cursor stores an
    /// asset id rather than a transient vector offset so insertions/removals do
    /// not make fairness depend on hash-map iteration order.
    next_atlas: Option<AssetId<PlanarGaussian3d>>,
}

#[derive(Default, Debug)]
struct PlannedLodAtlasUploads {
    admitted: Vec<LodAtlasSlotUpload>,
    deferred: Vec<LodAtlasSlotUpload>,
    deferred_canonical_bytes: u64,
    deferred_atlases: BTreeSet<AssetId<PlanarGaussian3d>>,
    oversized_slots: u64,
    first_error: Option<LodAtlasUploadBudgetError>,
}

fn canonical_slot_bytes(descriptor: LodAtlasSlotUpload) -> Result<u64, LodAtlasUploadBudgetError> {
    u64::from(descriptor.gaussians_per_slot)
        .checked_mul(std::mem::size_of::<crate::gaussian::formats::planar_3d::Gaussian3d>() as u64)
        .ok_or(LodAtlasUploadBudgetError::SlotCanonicalByteOverflow {
            atlas: descriptor.atlas,
            slot_index: descriptor.slot.index,
        })
}

fn record_deferred_descriptor(
    plan: &mut PlannedLodAtlasUploads,
    descriptor: LodAtlasSlotUpload,
    budget: LodAtlasUploadBudget,
) {
    plan.deferred_atlases.insert(descriptor.atlas);
    match canonical_slot_bytes(descriptor) {
        Ok(bytes) => {
            plan.deferred_canonical_bytes = plan.deferred_canonical_bytes.saturating_add(bytes);
            if bytes > budget.max_canonical_bytes_per_frame() {
                plan.oversized_slots = plan.oversized_slots.saturating_add(1);
                plan.first_error.get_or_insert(
                    LodAtlasUploadBudgetError::SlotExceedsCanonicalByteLimit {
                        atlas: descriptor.atlas,
                        slot_index: descriptor.slot.index,
                        required: bytes,
                        limit: budget.max_canonical_bytes_per_frame(),
                    },
                );
            }
        }
        Err(error) => {
            plan.oversized_slots = plan.oversized_slots.saturating_add(1);
            plan.first_error.get_or_insert(error);
            plan.deferred_canonical_bytes = u64::MAX;
        }
    }
    plan.deferred.push(descriptor);
}

fn plan_lod_atlas_uploads(
    scheduler: &mut LodAtlasUploadScheduler,
    descriptors: impl IntoIterator<Item = LodAtlasSlotUpload>,
    budget: LodAtlasUploadBudget,
) -> PlannedLodAtlasUploads {
    let mut groups = BTreeMap::<AssetId<PlanarGaussian3d>, VecDeque<LodAtlasSlotUpload>>::new();
    for descriptor in descriptors {
        groups
            .entry(descriptor.atlas)
            .or_default()
            .push_back(descriptor);
    }
    for group in groups.values_mut() {
        group
            .make_contiguous()
            .sort_unstable_by_key(|descriptor| descriptor.slot.index);
    }

    let order = groups.keys().copied().collect::<Vec<_>>();
    let mut plan = PlannedLodAtlasUploads::default();
    if order.is_empty() {
        return plan;
    }
    let mut cursor = scheduler.next_atlas.map_or(0, |next| {
        let offset = order.partition_point(|atlas| *atlas < next);
        if offset == order.len() { 0 } else { offset }
    });
    let mut remaining_bytes = budget.max_canonical_bytes_per_frame();
    let mut remaining_slots = u64::from(budget.max_slots_per_frame());
    let mut blocked = BTreeSet::new();
    let mut last_admitted_index = None;

    while remaining_slots != 0 {
        let mut admitted_in_cycle = false;
        for _ in 0..order.len() {
            let index = cursor;
            cursor = (cursor + 1) % order.len();
            let atlas = order[index];
            if blocked.contains(&atlas) {
                continue;
            }
            let Some(descriptor) = groups
                .get(&atlas)
                .and_then(|descriptors| descriptors.front())
                .copied()
            else {
                continue;
            };
            let bytes = match canonical_slot_bytes(descriptor) {
                Ok(bytes) => bytes,
                Err(error) => {
                    plan.first_error.get_or_insert(error);
                    blocked.insert(atlas);
                    continue;
                }
            };
            if bytes > budget.max_canonical_bytes_per_frame() {
                plan.first_error.get_or_insert(
                    LodAtlasUploadBudgetError::SlotExceedsCanonicalByteLimit {
                        atlas,
                        slot_index: descriptor.slot.index,
                        required: bytes,
                        limit: budget.max_canonical_bytes_per_frame(),
                    },
                );
                blocked.insert(atlas);
                continue;
            }
            if bytes > remaining_bytes {
                // Slot order is a generation/order proof within one atlas, so
                // do not bypass this head with a later, potentially newer slot.
                blocked.insert(atlas);
                continue;
            }
            groups
                .get_mut(&atlas)
                .expect("planned atlas group exists")
                .pop_front();
            plan.admitted.push(descriptor);
            remaining_bytes -= bytes;
            remaining_slots -= 1;
            last_admitted_index = Some(index);
            admitted_in_cycle = true;
            if remaining_slots == 0 {
                break;
            }
        }
        if !admitted_in_cycle {
            break;
        }
    }

    if let Some(index) = last_admitted_index {
        scheduler.next_atlas = Some(order[(index + 1) % order.len()]);
    }
    for (_, descriptors) in groups {
        for descriptor in descriptors {
            record_deferred_descriptor(&mut plan, descriptor, budget);
        }
    }
    plan
}

impl PendingLodAtlasSlotUpload {
    fn descriptor(&self) -> LodAtlasSlotUpload {
        match self {
            Self::Ready(upload) => upload.descriptor,
        }
    }
}

/// Uploads that have crossed into the render world for this preparation pass.
#[derive(Resource, Default)]
struct ExtractedLodAtlasUploads {
    slots: HashMap<LodAtlasUploadKey, PendingLodAtlasSlotUpload>,
    admitted: BTreeSet<LodAtlasUploadKey>,
    invalidations: BTreeSet<LodAtlasUploadKey>,
    frame_budget: LodAtlasUploadBudget,
    deferred_slots: u64,
    deferred_canonical_bytes: u64,
    oversized_slots: u64,
}

#[derive(Clone)]
struct ExtractedLodTransientAtlas {
    source: AssetId<PlanarGaussian3d>,
    physical_gaussians: u32,
    gaussians_per_slot: u32,
    generation: u64,
    ticket: LodTransientAtlasTicket,
}

#[derive(Resource, Default)]
struct ExtractedLodTransientAtlases {
    atlases: BTreeMap<AssetId<PlanarGaussian3d>, ExtractedLodTransientAtlas>,
}

struct LodTransientGpuAtlas {
    source: AssetId<PlanarGaussian3d>,
    physical_gaussians: u32,
    gaussians_per_slot: u32,
    generation: u64,
    ticket: LodTransientAtlasTicket,
    /// Authoritative ownership of the fixed GPU allocation.
    ///
    /// The generic `RenderAssets` map is only the lookup surface consumed by
    /// the renderer. Retaining the buffer handles here lets us repair an
    /// out-of-band map loss without reallocating storage or invalidating every
    /// streamed slot. This resource is initialized through
    /// `init_gpu_resource`, so a real render-device restart drops these handles
    /// and still takes the generation-bumped reupload path below.
    storage: PlanarStorageGaussian3d,
    render_asset_restores: u64,
}

#[derive(Resource, Default)]
struct LodTransientGpuAtlases {
    atlases: HashMap<AssetId<PlanarGaussian3d>, LodTransientGpuAtlas>,
}

impl LodTransientGpuAtlases {
    fn accepts_upload_generation(
        &self,
        atlas: AssetId<PlanarGaussian3d>,
        generation: Option<u64>,
    ) -> bool {
        self.atlases.get(&atlas).is_some_and(|state| {
            transient_upload_generation_is_current(state.generation, &state.ticket, generation)
        }) || generation.is_none()
    }
}

fn transient_upload_generation_is_current(
    state_generation: u64,
    ticket: &LodTransientAtlasTicket,
    upload_generation: Option<u64>,
) -> bool {
    let Some(upload_generation) = upload_generation else {
        // Package-owned atlases are ordinary render assets and do not
        // participate in transient allocation generations.
        return true;
    };
    state_generation == upload_generation && ticket.generation() == upload_generation
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LodTransientAtlasMaintenance {
    Keep,
    RestoreOwned,
    AllocateCurrentGeneration,
    AllocateNewGeneration,
}

const fn transient_atlas_maintenance(
    previous_compatible: bool,
    gpu_asset_exists: bool,
    requires_new_generation: bool,
) -> LodTransientAtlasMaintenance {
    if previous_compatible {
        if gpu_asset_exists {
            LodTransientAtlasMaintenance::Keep
        } else {
            LodTransientAtlasMaintenance::RestoreOwned
        }
    } else if requires_new_generation {
        // `LodTransientGpuAtlases` is render-device scoped. Losing that owner
        // while the main-world ticket is still ready means RenderStartup has
        // replaced the device resources, so every page must be reuploaded.
        LodTransientAtlasMaintenance::AllocateNewGeneration
    } else {
        LodTransientAtlasMaintenance::AllocateCurrentGeneration
    }
}

/// Render-world proof that a physical atlas slot contains a particular
/// allocator generation.
///
/// LoD compaction consults this registry before accepting an ACTIVE bridge
/// frontier. The uploader invalidates a slot before every attempted write, so
/// a missing GPU asset or invalid CPU range cannot accidentally reuse an older
/// proof with the same allocator generation.
#[derive(Resource, Default, Debug)]
pub(crate) struct LodAtlasGpuGenerations {
    slots: HashMap<(UntypedAssetId, u32), u32>,
    /// Render-allocation identity, independent of logical slot generations.
    /// Recreating storage can reuse the same ticket/slot values, so consumers
    /// must not treat those logical generations as proof that an old indirect
    /// output still addresses the currently bound buffers.
    allocation_epochs: HashMap<UntypedAssetId, u64>,
    next_allocation_epoch: u64,
    /// Monotonic per-atlas content epoch for direct GPU subrange writes.
    /// These writes deliberately bypass Bevy asset replacement, so renderer
    /// caches cannot infer them from a storage bind group's change tick.
    content_revisions: HashMap<UntypedAssetId, u64>,
    slot_content_revisions: HashMap<(UntypedAssetId, u32), u64>,
}

impl LodAtlasGpuGenerations {
    #[cfg(any(test, lod_render_path))]
    pub(crate) fn allocation_epoch(&self, atlas: UntypedAssetId) -> Option<u64> {
        self.allocation_epochs.get(&atlas).copied()
    }

    #[cfg(any(test, lod_render_path))]
    pub(crate) fn is_current(&self, atlas: UntypedAssetId, slot: AtlasSlot) -> bool {
        self.slots.get(&(atlas, slot.index)).copied() == Some(slot.generation)
    }

    #[cfg(any(test, lod_render_path))]
    pub(crate) fn frontier_is_current(
        &self,
        atlas: UntypedAssetId,
        ranges: &[LodPhysicalRange],
    ) -> bool {
        ranges
            .iter()
            .all(|range| self.is_current(atlas, range.slot))
    }

    #[cfg(any(test, lod_render_path))]
    pub(crate) fn content_revision(&self, atlas: UntypedAssetId) -> u64 {
        self.content_revisions.get(&atlas).copied().unwrap_or(0)
    }

    #[cfg(any(test, lod_render_path))]
    pub(crate) fn frontier_content_signature(
        &self,
        atlas: UntypedAssetId,
        ranges: &[LodPhysicalRange],
    ) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        ranges.iter().fold(FNV_OFFSET, |hash, range| {
            let revision = self
                .slot_content_revisions
                .get(&(atlas, range.slot.index))
                .copied()
                .unwrap_or(0);
            [
                u64::from(range.slot.index),
                u64::from(range.slot.generation),
                revision,
            ]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .fold(hash, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
            })
        })
    }

    fn invalidate(&mut self, atlas: AssetId<PlanarGaussian3d>, slot_index: u32) {
        self.slots.remove(&(atlas.untyped(), slot_index));
    }

    fn mark_current(&mut self, descriptor: LodAtlasSlotUpload) {
        let atlas = descriptor.atlas.untyped();
        let revision = self.content_revisions.entry(atlas).or_default();
        *revision = revision.wrapping_add(1).max(1);
        self.slot_content_revisions
            .insert((atlas, descriptor.slot.index), *revision);
        if descriptor.slot.generation == 0 {
            return;
        }
        self.slots
            .insert((atlas, descriptor.slot.index), descriptor.slot.generation);
    }

    fn mark_new_allocation(&mut self, atlas: AssetId<PlanarGaussian3d>) {
        self.next_allocation_epoch = self.next_allocation_epoch.wrapping_add(1).max(1);
        self.allocation_epochs
            .insert(atlas.untyped(), self.next_allocation_epoch);
    }

    fn invalidate_atlas(&mut self, atlas: AssetId<PlanarGaussian3d>) {
        let atlas = atlas.untyped();
        self.slots
            .retain(|(resident_atlas, _), _| *resident_atlas != atlas);
        self.content_revisions.remove(&atlas);
        self.slot_content_revisions
            .retain(|(resident_atlas, _), _| *resident_atlas != atlas);
        self.allocation_epochs.remove(&atlas);
    }
}

/// Installs extraction and render preparation for bounded atlas subranges.
#[derive(Default)]
pub struct GaussianLodAtlasUploadPlugin;

impl Plugin for GaussianLodAtlasUploadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LodAtlasUploadQueue>()
            .init_resource::<LodAtlasUploadBudget>()
            .init_resource::<LodAtlasUploadBudgetStatus>()
            .init_resource::<LodAtlasUploadScheduler>()
            .init_resource::<LodTransientAtlasRegistry>();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ExtractedLodAtlasUploads>()
                .init_resource::<ExtractedLodTransientAtlases>()
                .init_gpu_resource::<LodTransientGpuAtlases>()
                .init_gpu_resource::<LodAtlasGpuGenerations>()
                .add_systems(ExtractSchedule, extract_lod_atlas_uploads)
                .add_systems(
                    Render,
                    (prepare_transient_lod_atlases, apply_lod_atlas_uploads)
                        .chain()
                        .in_set(RenderSystems::PrepareAssets)
                        .after(prepare_assets::<PlanarStorageGaussian3d>),
                );
            #[cfg(feature = "precompute_covariance_3d")]
            render_app.init_gpu_resource::<LodCovariancePipeline>();
        }
    }
}

fn extract_lod_atlas_uploads(
    mut extracted: ResMut<ExtractedLodAtlasUploads>,
    mut transient_atlases: ResMut<ExtractedLodTransientAtlases>,
    mut main_world: ResMut<MainWorld>,
) {
    let transient_sources = main_world.resource::<LodTransientAtlasRegistry>().live();
    let next_transient = transient_sources
        .iter()
        .map(|(&atlas, source)| {
            (
                atlas,
                ExtractedLodTransientAtlas {
                    source: source.source,
                    physical_gaussians: source.physical_gaussians,
                    gaussians_per_slot: source.gaussians_per_slot,
                    generation: source.generation,
                    ticket: source.ticket.clone(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let reset_atlases = transient_atlases
        .atlases
        .iter()
        .filter_map(|(&atlas, previous)| {
            (next_transient.get(&atlas).map(|next| next.generation) != Some(previous.generation))
                .then_some(atlas)
        })
        .collect::<BTreeSet<_>>();
    if !reset_atlases.is_empty() {
        extracted
            .slots
            .retain(|(atlas, _), _| !reset_atlases.contains(atlas));
        extracted
            .admitted
            .retain(|(atlas, _)| !reset_atlases.contains(atlas));
    }
    transient_atlases.atlases = next_transient;

    let queued = {
        let mut queue = main_world.resource_mut::<LodAtlasUploadQueue>();
        std::mem::take(&mut queue.slots)
    };
    let budget = *main_world.resource::<LodAtlasUploadBudget>();

    // A newer main-world request supersedes any render-world retry for the
    // same physical slot. Invalidate every queued key immediately, including
    // budget-deferred work, so an in-place source mutation can never retain a
    // stale same-generation GPU proof while waiting for admission.
    for key in queued.keys().copied() {
        extracted.invalidations.insert(key);
        extracted.slots.remove(&key);
    }

    let carried = extracted
        .slots
        .values()
        .map(PendingLodAtlasSlotUpload::descriptor)
        .collect::<Vec<_>>();
    let mut queued_descriptors = queued.into_values().collect::<Vec<_>>();
    queued_descriptors.sort_unstable_by_key(|descriptor| (descriptor.atlas, descriptor.slot.index));
    let queued_keys = queued_descriptors
        .iter()
        .map(|descriptor| (descriptor.atlas, descriptor.slot.index))
        .collect::<BTreeSet<_>>();

    let mut plan = {
        let mut scheduler = main_world.resource_mut::<LodAtlasUploadScheduler>();
        if carried.is_empty() {
            plan_lod_atlas_uploads(&mut scheduler, queued_descriptors.iter().copied(), budget)
        } else {
            // Render-world retries already own bounded CPU snapshots. Drain
            // that recovery backlog before cloning any newly queued planes,
            // preventing payload growth while a device/asset is unavailable.
            plan_lod_atlas_uploads(&mut scheduler, carried, budget)
        }
    };
    if !extracted.slots.is_empty() {
        for descriptor in queued_descriptors.iter().copied() {
            record_deferred_descriptor(&mut plan, descriptor, budget);
        }
    }

    extracted.frame_budget = budget;
    extracted.admitted.clear();
    extracted.admitted.extend(
        plan.admitted
            .iter()
            .map(|descriptor| (descriptor.atlas, descriptor.slot.index)),
    );
    extracted.deferred_slots = plan.deferred.len() as u64;
    extracted.deferred_canonical_bytes = plan.deferred_canonical_bytes;
    extracted.oversized_slots = plan.oversized_slots;
    main_world
        .resource_mut::<LodAtlasUploadBudgetStatus>()
        .last_error = plan.first_error;

    // Only main-world descriptors are requeued. Carried render-world payloads
    // already remain in `extracted.slots` and are reconsidered next frame.
    {
        let mut queue = main_world.resource_mut::<LodAtlasUploadQueue>();
        for descriptor in plan.deferred {
            let key = (descriptor.atlas, descriptor.slot.index);
            if queued_keys.contains(&key) {
                queue.slots.insert(key, descriptor);
            }
        }
    }

    let assets = main_world.resource::<Assets<PlanarGaussian3d>>();
    for descriptor in plan.admitted {
        let key = (descriptor.atlas, descriptor.slot.index);
        if extracted.slots.contains_key(&key) {
            continue;
        }
        let transient_source = transient_sources.get(&descriptor.atlas);
        let upload = if let Some(source) = transient_source {
            source
                .snapshot_slot(descriptor)
                .map(|planes| ExtractedLodAtlasSlotUpload {
                    descriptor,
                    planes,
                    transient_generation: Some(source.generation),
                })
        } else {
            snapshot_slot(assets.get(descriptor.atlas), descriptor, None)
        };
        match upload {
            Ok(upload) => {
                extracted
                    .slots
                    .insert(key, PendingLodAtlasSlotUpload::Ready(upload));
            }
            Err(error) => {
                error!(
                    "failed to snapshot LoD atlas {:?} slot {}: {error}",
                    descriptor.atlas, descriptor.slot.index
                );
                if let Some(source) = transient_source {
                    // Missing/poisoned sparse payloads are an orchestration
                    // invariant failure, not retryable render work. Publish the
                    // failure through the owner ticket so the main-world bridge
                    // can fail visibly instead of forgetting an Invalid entry.
                    source.ticket.fail(source.generation);
                }
            }
        }
    }
}

fn snapshot_slot(
    atlas: Option<&PlanarGaussian3d>,
    descriptor: LodAtlasSlotUpload,
    transient_generation: Option<u64>,
) -> Result<ExtractedLodAtlasSlotUpload, LodAtlasUploadError> {
    descriptor.validate_address()?;
    let atlas = atlas.ok_or(LodAtlasUploadError::MissingAtlasAsset)?;
    let start = descriptor.physical_start()? as usize;
    let end = descriptor.physical_end()? as usize;
    if end > atlas.len() {
        return Err(LodAtlasUploadError::SlotOutOfRange {
            start: start as u64,
            end: end as u64,
            atlas_len: atlas.len() as u64,
        });
    }
    if atlas.spherical_harmonic.len() != atlas.len()
        || atlas.rotation.len() != atlas.len()
        || atlas.scale_opacity.len() != atlas.len()
    {
        return Err(LodAtlasUploadError::InconsistentPlaneLengths);
    }

    Ok(ExtractedLodAtlasSlotUpload {
        descriptor,
        transient_generation,
        planes: PlanarGaussian3d {
            position_visibility: atlas.position_visibility[start..end].to_vec(),
            spherical_harmonic: atlas.spherical_harmonic[start..end].to_vec(),
            rotation: atlas.rotation[start..end].to_vec(),
            scale_opacity: atlas.scale_opacity[start..end].to_vec(),
        },
    })
}

fn encode_planar_copy<T: Pod>(
    render_device: &RenderDevice,
    encoder: &mut wgpu::CommandEncoder,
    destination: &wgpu::Buffer,
    start: usize,
    values: &[T],
    label: &'static str,
    staging: &mut Vec<bevy::render::render_resource::Buffer>,
) -> Result<(), LodAtlasUploadError> {
    let element_bytes = u64::try_from(std::mem::size_of::<T>())
        .map_err(|_| LodAtlasUploadError::AddressOverflow)?;
    let destination_offset = u64::try_from(start)
        .map_err(|_| LodAtlasUploadError::AddressOverflow)?
        .checked_mul(element_bytes)
        .ok_or(LodAtlasUploadError::AddressOverflow)?;
    let copy_bytes = u64::try_from(values.len())
        .map_err(|_| LodAtlasUploadError::AddressOverflow)?
        .checked_mul(element_bytes)
        .ok_or(LodAtlasUploadError::AddressOverflow)?;
    if copy_bytes == 0 {
        return Err(LodAtlasUploadError::EmptyCoalescedRange);
    }
    let source = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_buffer(&source, 0, destination, destination_offset, copy_bytes);
    staging.push(source);
    Ok(())
}

#[derive(Clone, Copy)]
struct LodAtlasCanonicalBuffers<'a> {
    position_visibility: &'a wgpu::Buffer,
    spherical_harmonic: &'a wgpu::Buffer,
    rotation: &'a wgpu::Buffer,
    scale_opacity: &'a wgpu::Buffer,
    count: usize,
}

fn encode_canonical_buffer_copies(
    render_device: &RenderDevice,
    encoder: &mut wgpu::CommandEncoder,
    atlas: LodAtlasCanonicalBuffers<'_>,
    upload: &CoalescedLodAtlasUpload,
    staging: &mut Vec<bevy::render::render_resource::Buffer>,
) -> Result<(), LodAtlasUploadError> {
    let start = upload.start()?;
    let count = upload.planes.len();
    if count == 0
        || upload.planes.spherical_harmonic.len() != count
        || upload.planes.rotation.len() != count
        || upload.planes.scale_opacity.len() != count
    {
        return Err(LodAtlasUploadError::InconsistentPlaneLengths);
    }
    let end = start
        .checked_add(count)
        .ok_or(LodAtlasUploadError::AddressOverflow)?;
    if end > atlas.count {
        return Err(LodAtlasUploadError::SlotOutOfRange {
            start: start as u64,
            end: end as u64,
            atlas_len: atlas.count as u64,
        });
    }
    staging
        .try_reserve(4)
        .map_err(|_| LodAtlasUploadError::CoalescedAllocationFailed)?;
    encode_planar_copy(
        render_device,
        encoder,
        atlas.position_visibility,
        start,
        &upload.planes.position_visibility,
        "lod_atlas_position_staging",
        staging,
    )?;
    encode_planar_copy(
        render_device,
        encoder,
        atlas.spherical_harmonic,
        start,
        &upload.planes.spherical_harmonic,
        "lod_atlas_sh_staging",
        staging,
    )?;
    encode_planar_copy(
        render_device,
        encoder,
        atlas.rotation,
        start,
        &upload.planes.rotation,
        "lod_atlas_rotation_staging",
        staging,
    )?;
    encode_planar_copy(
        render_device,
        encoder,
        atlas.scale_opacity,
        start,
        &upload.planes.scale_opacity,
        "lod_atlas_scale_opacity_staging",
        staging,
    )?;
    Ok(())
}

fn encode_canonical_atlas_copies(
    render_device: &RenderDevice,
    encoder: &mut wgpu::CommandEncoder,
    atlas: &PlanarStorageGaussian3d,
    upload: &CoalescedLodAtlasUpload,
    staging: &mut Vec<bevy::render::render_resource::Buffer>,
) -> Result<(), LodAtlasUploadError> {
    encode_canonical_buffer_copies(
        render_device,
        encoder,
        LodAtlasCanonicalBuffers {
            position_visibility: &atlas.position_visibility,
            spherical_harmonic: &atlas.spherical_harmonic,
            rotation: &atlas.rotation,
            scale_opacity: &atlas.scale_opacity,
            count: atlas.count,
        },
        upload,
        staging,
    )
}

fn submit_lod_atlas_batch(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    atlas: &PlanarStorageGaussian3d,
    uploads: &[CoalescedLodAtlasUpload],
    #[cfg(feature = "precompute_covariance_3d")] covariance_pipeline: Option<
        &LodCovariancePipeline,
    >,
) -> Result<(), LodAtlasUploadError> {
    if uploads.is_empty() {
        return Err(LodAtlasUploadError::EmptyCoalescedRange);
    }
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("lod_atlas_upload_encoder"),
    });
    let mut staging = Vec::new();
    let result: Result<u32, LodAtlasUploadError> = (|| {
        for upload in uploads {
            encode_canonical_atlas_copies(
                render_device,
                &mut encoder,
                atlas,
                upload,
                &mut staging,
            )?;
        }
        #[cfg(feature = "precompute_covariance_3d")]
        {
            let ranges = uploads
                .iter()
                .map(|upload| {
                    Ok(LodCovarianceUploadRange {
                        start: u32::try_from(upload.start()?)
                            .map_err(|_| LodAtlasUploadError::AddressOverflow)?,
                        count: u32::try_from(upload.planes.len())
                            .map_err(|_| LodAtlasUploadError::AddressOverflow)?,
                    })
                })
                .collect::<Result<Vec<_>, LodAtlasUploadError>>()?;
            covariance_pipeline
                .ok_or(LodAtlasUploadError::CovariancePipelineUnavailable)?
                .encode_dispatch_buffers(
                    render_device,
                    &mut encoder,
                    &atlas.rotation,
                    &atlas.scale_opacity,
                    &atlas.covariance_3d_opacity,
                    &ranges,
                )
        }
        #[cfg(not(feature = "precompute_covariance_3d"))]
        {
            Ok(0_u32)
        }
    })();
    result?;
    render_queue.submit([encoder.finish()]);
    Ok(())
}

#[derive(SystemParam)]
struct LodAtlasUploadGpuParams<'w> {
    gpu_assets: ResMut<'w, RenderAssets<PlanarStorageGaussian3d>>,
    render_queue: Res<'w, RenderQueue>,
    render_device: Res<'w, RenderDevice>,
    #[cfg(feature = "precompute_covariance_3d")]
    covariance_pipeline: Option<Res<'w, LodCovariancePipeline>>,
}

fn transient_plane_bytes<T>(count: u32) -> Result<u64, LodAtlasUploadError> {
    u64::from(count)
        .checked_mul(size_of::<T>() as u64)
        .ok_or(LodAtlasUploadError::AddressOverflow)
}

fn create_transient_lod_atlas(
    render_device: &RenderDevice,
    physical_gaussians: u32,
) -> Result<PlanarStorageGaussian3d, LodAtlasUploadError> {
    if physical_gaussians == 0 {
        return Err(LodAtlasUploadError::InvalidAtlasLength {
            physical_gaussians,
            gaussians_per_slot: 1,
        });
    }
    let limits = render_device.limits();
    let storage_limit = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size);
    let checked_size = |label, size: u64| {
        if size > storage_limit {
            Err(LodAtlasUploadError::GpuPlaneExceedsLimit {
                label,
                required: size,
                limit: storage_limit,
            })
        } else {
            Ok(size)
        }
    };
    let create_plane = |label: &'static str, size| {
        Ok(render_device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: checked_size(label, size)?,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        }))
    };

    let position_visibility = create_plane(
        "lod_transient_position_visibility",
        transient_plane_bytes::<PositionVisibility>(physical_gaussians)?,
    )?;
    let spherical_harmonic = create_plane(
        "lod_transient_spherical_harmonic",
        transient_plane_bytes::<SphericalHarmonicCoefficients>(physical_gaussians)?,
    )?;
    let rotation = create_plane(
        "lod_transient_rotation",
        transient_plane_bytes::<Rotation>(physical_gaussians)?,
    )?;
    let scale_opacity = create_plane(
        "lod_transient_scale_opacity",
        transient_plane_bytes::<ScaleOpacity>(physical_gaussians)?,
    )?;
    #[cfg(feature = "precompute_covariance_3d")]
    let covariance_3d_opacity = create_plane(
        "lod_transient_covariance_3d_opacity",
        transient_plane_bytes::<Covariance3dOpacity>(physical_gaussians)?,
    )?;
    let draw_indirect_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("lod_transient_draw_indirect"),
        contents: transient_draw_indirect_args().as_bytes(),
        usage: BufferUsages::INDIRECT
            | BufferUsages::COPY_DST
            | BufferUsages::STORAGE
            | BufferUsages::COPY_SRC,
    });
    Ok(PlanarStorageGaussian3d {
        position_visibility,
        spherical_harmonic,
        rotation,
        scale_opacity,
        #[cfg(feature = "precompute_covariance_3d")]
        covariance_3d_opacity,
        count: physical_gaussians as usize,
        draw_indirect_buffer,
    })
}

fn transient_draw_indirect_args() -> wgpu::util::DrawIndirectArgs {
    wgpu::util::DrawIndirectArgs {
        vertex_count: 4,
        // A transient atlas is sparse storage, never a directly drawable
        // cloud. Only the LoD candidate path may provide a non-zero draw count
        // after validating every referenced slot generation.
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    }
}

fn prepare_transient_lod_atlases(
    desired: Res<ExtractedLodTransientAtlases>,
    mut owned: ResMut<LodTransientGpuAtlases>,
    mut generations: ResMut<LodAtlasGpuGenerations>,
    mut gpu_assets: ResMut<RenderAssets<PlanarStorageGaussian3d>>,
    render_device: Res<RenderDevice>,
) {
    let stale = owned
        .atlases
        .keys()
        .filter(|atlas| !desired.atlases.contains_key(atlas))
        .copied()
        .collect::<Vec<_>>();
    for atlas in stale {
        owned.atlases.remove(&atlas);
        gpu_assets.remove(atlas);
        generations.invalidate_atlas(atlas);
    }

    for (&atlas, spec) in &desired.atlases {
        if spec.ticket.is_canceled() {
            owned.atlases.remove(&atlas);
            gpu_assets.remove(atlas);
            generations.invalidate_atlas(atlas);
            continue;
        }
        let ticket_generation = spec.ticket.generation();
        let previous = owned.atlases.get(&atlas);
        let previous_definition_compatible = previous.is_some_and(|previous| {
            previous.source == spec.source
                && previous.physical_gaussians == spec.physical_gaussians
                && previous.gaussians_per_slot == spec.gaussians_per_slot
        });
        let previous_generation_matches =
            previous.is_some_and(|previous| previous.generation == ticket_generation);
        let previous_compatible = previous_definition_compatible && previous_generation_matches;
        let requires_new_generation = spec.ticket.is_ready()
            && (previous.is_none()
                || (!previous_definition_compatible && previous_generation_matches));
        let maintenance = transient_atlas_maintenance(
            previous_compatible,
            gpu_assets.get(atlas).is_some(),
            requires_new_generation,
        );
        match maintenance {
            LodTransientAtlasMaintenance::Keep => continue,
            LodTransientAtlasMaintenance::RestoreOwned => {
                let previous = owned
                    .atlases
                    .get_mut(&atlas)
                    .expect("compatible transient atlas has an authoritative owner");
                gpu_assets.insert(atlas, previous.storage.clone());
                previous.render_asset_restores =
                    previous.render_asset_restores.wrapping_add(1).max(1);
                if previous.render_asset_restores == 1 {
                    warn!(
                        "restored transient LoD atlas {atlas:?} into RenderAssets without reallocating or invalidating resident pages"
                    );
                }
                continue;
            }
            LodTransientAtlasMaintenance::AllocateCurrentGeneration
            | LodTransientAtlasMaintenance::AllocateNewGeneration => {}
        }
        let generation = if maintenance == LodTransientAtlasMaintenance::AllocateNewGeneration {
            spec.ticket.request_reupload()
        } else {
            ticket_generation
        };
        gpu_assets.remove(atlas);
        // A recreated allocation contains no resident pages, even when the
        // allocator happens to reuse the same logical slot generations.
        generations.invalidate_atlas(atlas);
        let allocation = create_transient_lod_atlas(&render_device, spec.physical_gaussians);
        match allocation {
            Ok(gpu_atlas) => {
                gpu_assets.insert(atlas, gpu_atlas.clone());
                generations.mark_new_allocation(atlas);
                owned.atlases.insert(
                    atlas,
                    LodTransientGpuAtlas {
                        source: spec.source,
                        physical_gaussians: spec.physical_gaussians,
                        gaussians_per_slot: spec.gaussians_per_slot,
                        generation,
                        ticket: spec.ticket.clone(),
                        storage: gpu_atlas,
                        render_asset_restores: 0,
                    },
                );
                // Storage readiness is deliberately independent of its page
                // contents. `LodAtlasGpuGenerations` remains empty until real
                // resident-page uploads complete.
                spec.ticket.acknowledge(generation);
            }
            Err(error) => {
                error!("failed to allocate transient LoD atlas {atlas:?}: {error}");
                spec.ticket.fail(generation);
                owned.atlases.remove(&atlas);
                gpu_assets.remove(atlas);
            }
        }
    }
}

fn apply_lod_atlas_uploads(
    mut uploads: ResMut<ExtractedLodAtlasUploads>,
    mut generations: ResMut<LodAtlasGpuGenerations>,
    transient: Res<LodTransientGpuAtlases>,
    gpu: LodAtlasUploadGpuParams,
) {
    uploads.deferred_slots = 0;
    uploads.deferred_canonical_bytes = 0;
    uploads.oversized_slots = 0;
    generations.slots.retain(|(atlas, _), _| {
        atlas
            .try_typed::<PlanarGaussian3d>()
            .is_ok_and(|atlas| gpu.gpu_assets.get(atlas).is_some())
    });
    generations.content_revisions.retain(|atlas, _| {
        atlas
            .try_typed::<PlanarGaussian3d>()
            .is_ok_and(|atlas| gpu.gpu_assets.get(atlas).is_some())
    });
    generations.slot_content_revisions.retain(|(atlas, _), _| {
        atlas
            .try_typed::<PlanarGaussian3d>()
            .is_ok_and(|atlas| gpu.gpu_assets.get(atlas).is_some())
    });
    for (atlas, slot_index) in std::mem::take(&mut uploads.invalidations) {
        generations.invalidate(atlas, slot_index);
    }

    let mut ready = Vec::new();
    for key in std::mem::take(&mut uploads.admitted) {
        let Some(pending) = uploads.slots.remove(&key) else {
            continue;
        };
        let descriptor = pending.descriptor();
        // Invalidate before every attempt. This matters when fallback and page
        // data reuse the same allocator generation in consecutive frames.
        generations.invalidate(descriptor.atlas, descriptor.slot.index);

        let PendingLodAtlasSlotUpload::Ready(upload) = pending;
        ready.push(upload);
    }
    let mut gpu_ready = Vec::new();
    for upload in ready {
        let descriptor = upload.descriptor;
        if gpu.gpu_assets.get(descriptor.atlas).is_none() {
            uploads.slots.insert(
                (descriptor.atlas, descriptor.slot.index),
                PendingLodAtlasSlotUpload::Ready(upload),
            );
        } else {
            gpu_ready.push(upload);
        }
    }
    // Coalescing moves the admitted snapshots into contiguous ranges. A range
    // is decomposed back into its exact fixed-stride slot payloads only if both
    // batch submission and its per-range fallback fail, avoiding a full hot-
    // path clone of every admitted plane.
    let coalesced = coalesce_atlas_uploads(gpu_ready);
    let mut batches = BTreeMap::<AssetId<PlanarGaussian3d>, Vec<CoalescedLodAtlasUpload>>::new();
    for upload in coalesced {
        let descriptor = upload.descriptors[0];
        batches.entry(descriptor.atlas).or_default().push(upload);
    }

    for (atlas_id, atlas_uploads) in batches {
        let Some(gpu_atlas) = gpu.gpu_assets.get(atlas_id) else {
            // This system runs after RenderAsset preparation. Keep every slot
            // invalid until its fixed storage allocation exists.
            for upload in atlas_uploads {
                retain_unsubmitted_coalesced_lod_atlas_upload(
                    &mut uploads.slots,
                    upload,
                    |generation| transient.accepts_upload_generation(atlas_id, generation),
                );
            }
            continue;
        };
        let batch_succeeded = submit_lod_atlas_batch(
            &gpu.render_device,
            &gpu.render_queue,
            gpu_atlas,
            &atlas_uploads,
            #[cfg(feature = "precompute_covariance_3d")]
            gpu.covariance_pipeline.as_deref(),
        )
        .is_ok();
        for upload in atlas_uploads {
            let range_succeeded = batch_succeeded
                || upload.start().is_ok_and(|start| {
                    if gpu_atlas
                        .write_gaussian_3d_range(&gpu.render_queue, start, &upload.planes)
                        .is_err()
                    {
                        return false;
                    }
                    #[cfg(feature = "precompute_covariance_3d")]
                    if gpu_atlas
                        .write_gaussian_3d_covariance_range_cpu(
                            &gpu.render_queue,
                            start,
                            &upload.planes,
                        )
                        .is_err()
                    {
                        return false;
                    }
                    true
                });
            if range_succeeded {
                for descriptor in &upload.descriptors {
                    // Submission owns this payload now, even if a transient
                    // allocation generation raced it. A stale generation is
                    // discarded rather than retried against the replacement.
                    if transient
                        .accepts_upload_generation(descriptor.atlas, upload.transient_generation)
                    {
                        generations.mark_current(*descriptor);
                    }
                }
            } else {
                retain_unsubmitted_coalesced_lod_atlas_upload(
                    &mut uploads.slots,
                    upload,
                    |generation| transient.accepts_upload_generation(atlas_id, generation),
                );
            }
        }
    }
}

/// Decomposes one failed contiguous upload back into its exact slot snapshots.
///
/// `is_current` deliberately runs before any splitting/allocation so an
/// obsolete transient allocation generation remains fail-closed at no extra
/// cost. Package atlases pass `None` and are always current. The next extraction
/// pass subjects every restored slot to the ordinary global byte/slot budgets.
fn retain_unsubmitted_coalesced_lod_atlas_upload(
    slots: &mut HashMap<LodAtlasUploadKey, PendingLodAtlasSlotUpload>,
    upload: CoalescedLodAtlasUpload,
    is_current: impl FnOnce(Option<u64>) -> bool,
) {
    if !is_current(upload.transient_generation) {
        return;
    }
    let CoalescedLodAtlasUpload {
        descriptors,
        mut planes,
        transient_generation,
    } = upload;
    let expected = descriptors.iter().try_fold(0_usize, |total, descriptor| {
        total.checked_add(descriptor.gaussians_per_slot as usize)
    });
    let Some(expected) = expected else {
        return;
    };
    if descriptors.is_empty()
        || planes.position_visibility.len() != expected
        || planes.spherical_harmonic.len() != expected
        || planes.rotation.len() != expected
        || planes.scale_opacity.len() != expected
    {
        return;
    }
    for descriptor in descriptors.into_iter().rev() {
        let count = descriptor.gaussians_per_slot as usize;
        let start = planes.position_visibility.len() - count;
        let slot_planes = PlanarGaussian3d {
            position_visibility: planes.position_visibility.split_off(start),
            spherical_harmonic: planes.spherical_harmonic.split_off(start),
            rotation: planes.rotation.split_off(start),
            scale_opacity: planes.scale_opacity.split_off(start),
        };
        slots.insert(
            (descriptor.atlas, descriptor.slot.index),
            PendingLodAtlasSlotUpload::Ready(ExtractedLodAtlasSlotUpload {
                descriptor,
                planes: slot_planes,
                transient_generation,
            }),
        );
    }
}

fn coalesce_atlas_uploads(
    uploads: Vec<ExtractedLodAtlasSlotUpload>,
) -> Vec<CoalescedLodAtlasUpload> {
    let mut groups = BTreeMap::<(AssetId<PlanarGaussian3d>, u32), Vec<_>>::new();
    for upload in uploads {
        groups
            .entry((
                upload.descriptor.atlas,
                upload.descriptor.gaussians_per_slot,
            ))
            .or_default()
            .push(upload);
    }
    let mut coalesced = Vec::new();
    for (_, mut group) in groups {
        group.sort_unstable_by_key(|upload| upload.descriptor.slot.index);
        let mut current: Option<CoalescedLodAtlasUpload> = None;
        for mut upload in group {
            let contiguous = current.as_ref().is_some_and(|current| {
                current.transient_generation == upload.transient_generation
                    && current
                        .descriptors
                        .last()
                        .and_then(|descriptor| descriptor.slot.index.checked_add(1))
                        == Some(upload.descriptor.slot.index)
            });
            if !contiguous {
                if let Some(current) = current.take() {
                    coalesced.push(current);
                }
                current = Some(CoalescedLodAtlasUpload {
                    descriptors: Vec::new(),
                    planes: PlanarGaussian3d::default(),
                    transient_generation: upload.transient_generation,
                });
            }
            let current = current.as_mut().expect("coalesced range initialized");
            current.descriptors.push(upload.descriptor);
            current
                .planes
                .position_visibility
                .append(&mut upload.planes.position_visibility);
            current
                .planes
                .spherical_harmonic
                .append(&mut upload.planes.spherical_harmonic);
            current.planes.rotation.append(&mut upload.planes.rotation);
            current
                .planes
                .scale_opacity
                .append(&mut upload.planes.scale_opacity);
        }
        if let Some(current) = current {
            coalesced.push(current);
        }
    }
    coalesced
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodAtlasUploadError {
    ZeroSlotStride,
    ZeroGeneration,
    AddressOverflow,
    MissingAtlasAsset,
    InconsistentPlaneLengths,
    EmptyCoalescedRange,
    CoalescedAllocationFailed,
    CovariancePipelineUnavailable,
    CovarianceDispatchLimit,
    TransientAtlasLockPoisoned,
    GpuPlaneExceedsLimit {
        label: &'static str,
        required: u64,
        limit: u64,
    },
    InvalidAtlasLength {
        physical_gaussians: u32,
        gaussians_per_slot: u32,
    },
    QueueAllocationFailed {
        slot_count: u32,
    },
    MissingTransientAtlasSlot {
        slot_index: u32,
    },
    TransientSlotLengthMismatch {
        slot_index: u32,
        expected: u32,
        actual: u32,
    },
    SlotOutOfRange {
        start: u64,
        end: u64,
        atlas_len: u64,
    },
}

impl std::fmt::Display for LodAtlasUploadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSlotStride => write!(formatter, "LoD atlas slot stride is zero"),
            Self::ZeroGeneration => write!(formatter, "LoD atlas slot generation is zero"),
            Self::AddressOverflow => write!(formatter, "LoD atlas slot address overflow"),
            Self::MissingAtlasAsset => write!(formatter, "LoD atlas asset is missing"),
            Self::InconsistentPlaneLengths => {
                write!(formatter, "LoD atlas planes have inconsistent lengths")
            }
            Self::EmptyCoalescedRange => {
                write!(formatter, "LoD atlas coalesced range is empty")
            }
            Self::CoalescedAllocationFailed => {
                write!(
                    formatter,
                    "failed to allocate bounded LoD atlas upload batch"
                )
            }
            Self::CovariancePipelineUnavailable => {
                write!(formatter, "LoD covariance compute pipeline is unavailable")
            }
            Self::CovarianceDispatchLimit => write!(
                formatter,
                "LoD covariance upload exceeds adapter dispatch dimensions"
            ),
            Self::TransientAtlasLockPoisoned => {
                write!(formatter, "transient LoD atlas lock is poisoned")
            }
            Self::GpuPlaneExceedsLimit {
                label,
                required,
                limit,
            } => write!(
                formatter,
                "transient LoD atlas plane {label} requires {required} bytes, exceeding the GPU storage-buffer limit {limit}"
            ),
            Self::InvalidAtlasLength {
                physical_gaussians,
                gaussians_per_slot,
            } => write!(
                formatter,
                "LoD atlas length {physical_gaussians} is not a positive multiple of slot stride {gaussians_per_slot}"
            ),
            Self::QueueAllocationFailed { slot_count } => write!(
                formatter,
                "failed to reserve LoD atlas upload queue for {slot_count} physical slots"
            ),
            Self::MissingTransientAtlasSlot { slot_index } => write!(
                formatter,
                "transient LoD atlas slot {slot_index} has no materialized CPU payload"
            ),
            Self::TransientSlotLengthMismatch {
                slot_index,
                expected,
                actual,
            } => write!(
                formatter,
                "transient LoD atlas slot {slot_index} has {actual} Gaussians, expected the fixed stride {expected}"
            ),
            Self::SlotOutOfRange {
                start,
                end,
                atlas_len,
            } => write!(
                formatter,
                "LoD atlas slot range {start}..{end} exceeds atlas length {atlas_len}"
            ),
        }
    }
}

impl std::error::Error for LodAtlasUploadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::formats::planar_3d::Gaussian3d;

    fn atlas_id(value: u128) -> AssetId<PlanarGaussian3d> {
        AssetId::Uuid {
            uuid: bevy::asset::uuid::Uuid::from_u128(value),
        }
    }

    fn descriptor(
        atlas: AssetId<PlanarGaussian3d>,
        slot_index: u32,
        gaussians_per_slot: u32,
    ) -> LodAtlasSlotUpload {
        LodAtlasSlotUpload {
            atlas,
            slot: AtlasSlot {
                index: slot_index,
                generation: slot_index + 1,
            },
            gaussians_per_slot,
        }
    }

    fn gaussian(x: f32) -> Gaussian3d {
        Gaussian3d {
            position_visibility: [x, x + 1.0, x + 2.0, 1.0].into(),
            rotation: [1.0, 0.0, 0.0, x].into(),
            scale_opacity: [x + 0.1, x + 0.2, x + 0.3, 0.5].into(),
            ..default()
        }
    }

    #[test]
    fn transient_initialization_is_source_independent_and_queues_no_full_upload() {
        let assets = Assets::<PlanarGaussian3d>::default();
        let atlas = assets.reserve_handle();
        let source = assets.reserve_handle();
        let owner = LodTransientAtlas::new_empty(4).unwrap();
        let mut registry = LodTransientAtlasRegistry::default();
        registry
            .register(atlas.id(), source.id(), 100_000_000, 2, &owner)
            .unwrap();
        let mut queue = LodAtlasUploadQueue::default();
        registry.queue_pending_initialization(&mut queue).unwrap();

        assert!(assets.get(&atlas).is_none());
        assert_eq!(owner.physical_gaussians(), 4);
        assert_eq!(owner.planes().read().unwrap().len(), 0);
        assert_eq!(owner.materialized_slot_count().unwrap(), 0);
        assert_eq!(owner.materialized_gaussian_count().unwrap(), 0);
        assert_eq!(queue.queued_slot_count(), 0);
        assert!(!owner.ticket().is_ready());
        let budget = LodAtlasUploadBudget::default();
        let mut scheduler = LodAtlasUploadScheduler::default();
        let plan = plan_lod_atlas_uploads(&mut scheduler, queue.queued_slots(), budget);
        assert!(plan.admitted.is_empty());
        assert!(plan.deferred.is_empty());
        assert_eq!(plan.deferred_canonical_bytes, 0);

        owner.ticket().request_reupload_for_test();
        registry.queue_pending_initialization(&mut queue).unwrap();
        assert_eq!(queue.queued_slot_count(), 0);
    }

    #[test]
    fn hundred_million_entry_transient_has_zero_cold_cpu_materialization() {
        const PHYSICAL_GAUSSIANS: u32 = 100_000_000;
        const STRIDE: u32 = 1_000;
        let owner = LodTransientAtlas::new_empty(PHYSICAL_GAUSSIANS).unwrap();

        assert_eq!(owner.physical_gaussians(), PHYSICAL_GAUSSIANS);
        assert_eq!(owner.materialized_slot_count().unwrap(), 0);
        assert_eq!(owner.materialized_gaussian_count().unwrap(), 0);
        let dense_planes = owner.planes();
        let dense = dense_planes.read().unwrap();
        assert_eq!(dense.len(), 0);
        assert_eq!(dense.position_visibility.capacity(), 0);
        assert_eq!(dense.spherical_harmonic.capacity(), 0);
        assert_eq!(dense.rotation.capacity(), 0);
        assert_eq!(dense.scale_opacity.capacity(), 0);
        drop(dense);

        let slot_index = PHYSICAL_GAUSSIANS / STRIDE - 1;
        let slot = PlanarGaussian3d::from(
            (0..STRIDE)
                .map(|index| gaussian(index as f32))
                .collect::<Vec<_>>(),
        );
        owner.write_slot(slot_index, STRIDE, slot).unwrap();
        assert_eq!(owner.materialized_slot_count().unwrap(), 1);
        assert_eq!(
            owner.materialized_gaussian_count().unwrap(),
            STRIDE as usize
        );

        let descriptor = LodAtlasSlotUpload {
            atlas: atlas_id(0x100_000_000),
            slot: AtlasSlot {
                index: slot_index,
                generation: 17,
            },
            gaussians_per_slot: STRIDE,
        };
        let snapshot = owner.snapshot_slot(descriptor).unwrap();
        assert_eq!(snapshot.len(), STRIDE as usize);
        assert_eq!(snapshot.position_visibility[0].position[0], 0.0);
        assert_eq!(
            snapshot.position_visibility[STRIDE as usize - 1].position[0],
            (STRIDE - 1) as f32
        );

        assert!(owner.discard_slot(slot_index).unwrap());
        assert_eq!(owner.materialized_slot_count().unwrap(), 0);
        assert_eq!(
            owner.snapshot_slot(descriptor).unwrap_err(),
            LodAtlasUploadError::MissingTransientAtlasSlot { slot_index }
        );
    }

    #[test]
    fn sparse_transient_slot_writes_validate_stride_and_bounds() {
        let owner = LodTransientAtlas::new_empty(8).unwrap();
        assert_eq!(
            owner
                .write_slot(0, 4, PlanarGaussian3d::from(vec![Gaussian3d::default(); 3]),)
                .unwrap_err(),
            LodAtlasUploadError::TransientSlotLengthMismatch {
                slot_index: 0,
                expected: 4,
                actual: 3,
            }
        );
        assert_eq!(
            owner
                .write_slot(2, 4, PlanarGaussian3d::from(vec![Gaussian3d::default(); 4]),)
                .unwrap_err(),
            LodAtlasUploadError::SlotOutOfRange {
                start: 8,
                end: 12,
                atlas_len: 8,
            }
        );
        assert_eq!(owner.materialized_slot_count().unwrap(), 0);
    }

    #[test]
    fn transient_storage_ack_does_not_publish_page_generations() {
        let atlas = atlas_id(0xA71A5);
        let ticket = LodTransientAtlasTicket::default();
        let mut state_generation = ticket.generation();

        assert!(ticket.acknowledge(state_generation));
        assert!(ticket.is_ready());
        let page = descriptor(atlas, 0, 1);
        let mut pages = LodAtlasGpuGenerations::default();
        assert!(!pages.is_current(atlas.untyped(), page.slot));
        assert!(transient_upload_generation_is_current(
            state_generation,
            &ticket,
            Some(state_generation)
        ));
        pages.mark_current(page);
        assert!(pages.is_current(atlas.untyped(), page.slot));

        let next = ticket.request_reupload();
        assert!(!ticket.is_ready());
        assert!(!ticket.acknowledge(state_generation));
        assert!(!transient_upload_generation_is_current(
            state_generation,
            &ticket,
            Some(state_generation)
        ));

        pages.invalidate_atlas(atlas);
        assert!(!pages.is_current(atlas.untyped(), page.slot));
        state_generation = next;
        assert!(ticket.acknowledge(next));
        assert!(ticket.is_ready());
        assert!(transient_upload_generation_is_current(
            state_generation,
            &ticket,
            Some(next)
        ));
        assert!(transient_upload_generation_is_current(
            state_generation,
            &ticket,
            None
        ));
        assert!(
            !pages.is_current(atlas.untyped(), page.slot),
            "recreated storage remains unusable until the retained page is uploaded again"
        );
    }

    #[test]
    fn transient_render_asset_loss_restores_owned_buffers_without_generation_bump() {
        assert_eq!(
            transient_atlas_maintenance(true, true, false),
            LodTransientAtlasMaintenance::Keep
        );
        assert_eq!(
            transient_atlas_maintenance(true, false, false),
            LodTransientAtlasMaintenance::RestoreOwned
        );
        assert_eq!(
            transient_atlas_maintenance(false, false, false),
            LodTransientAtlasMaintenance::AllocateCurrentGeneration
        );
        assert_eq!(
            transient_atlas_maintenance(false, false, true),
            LodTransientAtlasMaintenance::AllocateNewGeneration
        );
        assert_eq!(
            transient_atlas_maintenance(false, true, true),
            LodTransientAtlasMaintenance::AllocateNewGeneration,
            "device-scoped owner loss supersedes any stale generic map entry"
        );
    }

    #[test]
    fn transient_raw_draw_is_fail_closed() {
        let args = transient_draw_indirect_args();
        assert_eq!(args.vertex_count, 4);
        assert_eq!(args.instance_count, 0);
    }

    #[test]
    fn dropping_transient_owner_cancels_late_completion() {
        let owner = LodTransientAtlas::new(PlanarGaussian3d::from(vec![gaussian(1.0)]));
        let ticket = owner.ticket().clone();
        let generation = ticket.generation();
        drop(owner);
        assert!(!ticket.acknowledge(generation));
        assert!(!ticket.is_ready());
    }

    #[test]
    fn queue_coalesces_physical_slot_and_keeps_latest_generation() {
        let atlas = AssetId::<PlanarGaussian3d>::default();
        let mut queue = LodAtlasUploadQueue::default();
        queue
            .enqueue_slot(
                atlas,
                AtlasSlot {
                    index: 2,
                    generation: 7,
                },
                16,
            )
            .unwrap();
        queue
            .enqueue_slot(
                atlas,
                AtlasSlot {
                    index: 2,
                    generation: 8,
                },
                16,
            )
            .unwrap();
        assert_eq!(queue.queued_slot_count(), 1);
        assert_eq!(queue.slots.values().next().unwrap().slot.generation, 8);
    }

    #[test]
    fn exact_slot_cancellation_preserves_a_newer_reused_generation() {
        let atlas = AssetId::<PlanarGaussian3d>::default();
        let old = AtlasSlot {
            index: 2,
            generation: 7,
        };
        let current = AtlasSlot {
            index: 2,
            generation: 8,
        };
        let mut queue = LodAtlasUploadQueue::default();
        queue.enqueue_slot(atlas, old, 16).unwrap();
        queue.enqueue_slot(atlas, current, 16).unwrap();

        assert!(!queue.remove_slot(atlas, old));
        assert_eq!(queue.queued_slots().next().unwrap().slot, current);
        assert!(queue.remove_slot(atlas, current));
        assert_eq!(queue.queued_slot_count(), 0);
    }

    #[test]
    fn aggregate_budget_is_nonzero_and_rejects_invalid_updates_without_mutation() {
        assert_eq!(
            LodAtlasUploadBudget::try_new(0, 1),
            Err(LodAtlasUploadBudgetError::ZeroCanonicalByteLimit)
        );
        assert_eq!(
            LodAtlasUploadBudget::try_new(1, 0),
            Err(LodAtlasUploadBudgetError::ZeroSlotLimit)
        );
        let mut budget = LodAtlasUploadBudget::try_new(4096, 4).unwrap();
        assert_eq!(
            budget.set_max_canonical_bytes_per_frame(0),
            Err(LodAtlasUploadBudgetError::ZeroCanonicalByteLimit)
        );
        assert_eq!(budget.max_canonical_bytes_per_frame(), 4096);
        assert_eq!(
            budget.set_max_slots_per_frame(0),
            Err(LodAtlasUploadBudgetError::ZeroSlotLimit)
        );
        assert_eq!(budget.max_slots_per_frame(), 4);
    }

    #[test]
    fn global_planner_is_deterministic_fair_and_preserves_slot_order() {
        let atlases = [atlas_id(1), atlas_id(2), atlas_id(3)];
        let record_bytes = std::mem::size_of::<Gaussian3d>() as u64;
        let budget = LodAtlasUploadBudget::try_new(record_bytes * 2, 2).unwrap();
        let input = vec![
            descriptor(atlases[2], 1, 1),
            descriptor(atlases[0], 1, 1),
            descriptor(atlases[1], 0, 1),
            descriptor(atlases[2], 0, 1),
            descriptor(atlases[0], 0, 1),
            descriptor(atlases[1], 1, 1),
        ];
        let mut scheduler = LodAtlasUploadScheduler::default();
        let first = plan_lod_atlas_uploads(&mut scheduler, input, budget);
        assert_eq!(
            first
                .admitted
                .iter()
                .map(|upload| (upload.atlas, upload.slot.index))
                .collect::<Vec<_>>(),
            vec![(atlases[0], 0), (atlases[1], 0)]
        );
        assert_eq!(first.deferred.len(), 4);
        assert_eq!(first.deferred_canonical_bytes, record_bytes * 4);
        assert_eq!(first.deferred_atlases.len(), 3);

        let second = plan_lod_atlas_uploads(&mut scheduler, first.deferred, budget);
        assert_eq!(
            second
                .admitted
                .iter()
                .map(|upload| (upload.atlas, upload.slot.index))
                .collect::<Vec<_>>(),
            vec![(atlases[2], 0), (atlases[0], 1)],
            "the next frame resumes after the last admitted cloud"
        );
        assert!(
            second
                .deferred
                .iter()
                .filter(|upload| upload.atlas == atlases[2])
                .all(|upload| upload.slot.index > 0),
            "later slots never bypass the per-cloud head"
        );
    }

    #[test]
    fn oversized_atomic_slot_is_deferred_with_typed_status() {
        let atlas = atlas_id(11);
        let record_bytes = std::mem::size_of::<Gaussian3d>() as u64;
        let budget = LodAtlasUploadBudget::try_new(record_bytes, 8).unwrap();
        let mut scheduler = LodAtlasUploadScheduler::default();
        let plan = plan_lod_atlas_uploads(&mut scheduler, [descriptor(atlas, 4, 2)], budget);
        assert!(plan.admitted.is_empty());
        assert_eq!(plan.deferred.len(), 1);
        assert_eq!(plan.deferred_canonical_bytes, record_bytes * 2);
        assert_eq!(plan.oversized_slots, 1);
        assert_eq!(
            plan.first_error,
            Some(LodAtlasUploadBudgetError::SlotExceedsCanonicalByteLimit {
                atlas,
                slot_index: 4,
                required: record_bytes * 2,
                limit: record_bytes,
            })
        );
    }

    #[test]
    fn complete_atlas_queue_is_bounded_and_publishes_no_residency_generation() {
        let atlas = AssetId::<PlanarGaussian3d>::default();
        let mut queue = LodAtlasUploadQueue::default();
        queue.enqueue_complete_atlas(atlas, 32, 8).unwrap();
        let mut slots = queue.queued_slots().collect::<Vec<_>>();
        slots.sort_by_key(|upload| upload.slot.index);
        assert_eq!(slots.len(), 4);
        assert_eq!(
            slots
                .iter()
                .map(|upload| upload.slot.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(slots.iter().all(|upload| upload.slot.generation == 0));

        let mut generations = LodAtlasGpuGenerations::default();
        generations.mark_current(slots[0]);
        assert!(!generations.is_current(
            atlas.untyped(),
            AtlasSlot {
                index: 0,
                generation: 1,
            }
        ));
    }

    #[test]
    fn successful_gpu_writes_advance_atlas_and_referenced_slot_revisions() {
        let atlas = atlas_id(12);
        let other = atlas_id(13);
        let mut generations = LodAtlasGpuGenerations::default();
        let range = |slot: AtlasSlot| LodPhysicalRange {
            node: crate::LodNodeId(u64::from(slot.index)),
            page: crate::LodPageId(u64::from(slot.index)),
            slot,
            physical_start: slot.index * 4,
            count: 4,
        };
        let slot_zero = descriptor(atlas, 0, 4);
        let slot_one = descriptor(atlas, 1, 4);
        let slot_zero_ranges = [range(slot_zero.slot)];
        let slot_one_ranges = [range(slot_one.slot)];
        let initial_zero =
            generations.frontier_content_signature(atlas.untyped(), &slot_zero_ranges);
        let initial_one = generations.frontier_content_signature(atlas.untyped(), &slot_one_ranges);
        assert_eq!(generations.content_revision(atlas.untyped()), 0);
        assert_eq!(generations.content_revision(other.untyped()), 0);

        generations.mark_current(slot_zero);
        let first = generations.content_revision(atlas.untyped());
        let written_zero =
            generations.frontier_content_signature(atlas.untyped(), &slot_zero_ranges);
        assert_eq!(first, 1);
        assert_ne!(written_zero, initial_zero);
        assert_eq!(
            generations.frontier_content_signature(atlas.untyped(), &slot_one_ranges),
            initial_one
        );
        assert_eq!(generations.content_revision(other.untyped()), 0);

        generations.mark_current(slot_one);
        assert_eq!(generations.content_revision(atlas.untyped()), first + 1);
        assert_eq!(
            generations.frontier_content_signature(atlas.untyped(), &slot_zero_ranges),
            written_zero,
            "a disjoint staged upload must not invalidate this frontier"
        );
        assert_ne!(
            generations.frontier_content_signature(atlas.untyped(), &slot_one_ranges),
            initial_one
        );

        generations.mark_current(slot_zero);
        assert_ne!(
            generations.frontier_content_signature(atlas.untyped(), &slot_zero_ranges),
            written_zero,
            "rewriting an overlapping slot must invalidate this frontier"
        );

        let complete_atlas_write = LodAtlasSlotUpload {
            atlas,
            slot: AtlasSlot {
                index: 2,
                generation: 0,
            },
            gaussians_per_slot: 4,
        };
        generations.mark_current(complete_atlas_write);
        assert_eq!(generations.content_revision(atlas.untyped()), first + 3);
        assert!(!generations.is_current(atlas.untyped(), complete_atlas_write.slot));
    }

    #[test]
    fn recreated_storage_gets_a_new_epoch_even_when_logical_generations_repeat() {
        let atlas = atlas_id(14);
        let mut generations = LodAtlasGpuGenerations::default();
        assert_eq!(generations.allocation_epoch(atlas.untyped()), None);

        generations.mark_new_allocation(atlas);
        let first = generations
            .allocation_epoch(atlas.untyped())
            .expect("first physical allocation epoch");
        generations.mark_current(descriptor(atlas, 0, 4));
        generations.invalidate_atlas(atlas);
        assert_eq!(generations.allocation_epoch(atlas.untyped()), None);

        generations.mark_new_allocation(atlas);
        let recreated = generations
            .allocation_epoch(atlas.untyped())
            .expect("recreated physical allocation epoch");
        assert_ne!(recreated, first);
        assert!(recreated > first);
    }

    #[test]
    fn snapshot_contains_exact_final_slot_planes() {
        let mut assets = Assets::<PlanarGaussian3d>::default();
        let handle = assets.add(PlanarGaussian3d::from(
            (0..12)
                .map(|index| gaussian(index as f32))
                .collect::<Vec<_>>(),
        ));
        let descriptor = LodAtlasSlotUpload {
            atlas: handle.id(),
            slot: AtlasSlot {
                index: 2,
                generation: 3,
            },
            gaussians_per_slot: 4,
        };
        let atlas = assets.get(&handle).unwrap();
        let upload = snapshot_slot(Some(atlas), descriptor, None).unwrap();
        assert_eq!(upload.planes.len(), 4);
        assert_eq!(
            upload
                .planes
                .position_visibility
                .iter()
                .map(|position| position.position[0])
                .collect::<Vec<_>>(),
            vec![8.0, 9.0, 10.0, 11.0]
        );
        assert_eq!(upload.planes.rotation, atlas.rotation[8..12]);
        assert_eq!(upload.planes.scale_opacity, atlas.scale_opacity[8..12]);
    }

    #[test]
    fn snapshot_rejects_overflow_and_out_of_bounds_without_allocating() {
        let atlas = PlanarGaussian3d::from(vec![Gaussian3d::default(); 4]);
        let mut queue = LodAtlasUploadQueue::default();
        assert_eq!(
            queue.enqueue_slot(
                AssetId::default(),
                AtlasSlot {
                    index: 0,
                    generation: 0,
                },
                4,
            ),
            Err(LodAtlasUploadError::ZeroGeneration)
        );
        let overflow = LodAtlasSlotUpload {
            atlas: AssetId::default(),
            slot: AtlasSlot {
                index: u32::MAX,
                generation: 1,
            },
            gaussians_per_slot: 2,
        };
        assert_eq!(
            snapshot_slot(Some(&atlas), overflow, None).unwrap_err(),
            LodAtlasUploadError::AddressOverflow
        );

        let outside = LodAtlasSlotUpload {
            atlas: AssetId::default(),
            slot: AtlasSlot {
                index: 1,
                generation: 1,
            },
            gaussians_per_slot: 4,
        };
        assert_eq!(
            snapshot_slot(Some(&atlas), outside, None).unwrap_err(),
            LodAtlasUploadError::SlotOutOfRange {
                start: 4,
                end: 8,
                atlas_len: 4,
            }
        );
    }

    #[test]
    fn generation_registry_rejects_reused_and_unuploaded_slots() {
        let atlas = AssetId::<PlanarGaussian3d>::default();
        let first = LodAtlasSlotUpload {
            atlas,
            slot: AtlasSlot {
                index: 0,
                generation: 1,
            },
            gaussians_per_slot: 4,
        };
        let mut generations = LodAtlasGpuGenerations::default();
        assert!(!generations.is_current(atlas.untyped(), first.slot));
        let ranges = [LodPhysicalRange {
            node: crate::LodNodeId(1),
            page: crate::LodPageId(1),
            slot: first.slot,
            physical_start: 0,
            count: 4,
        }];
        assert!(!generations.frontier_is_current(atlas.untyped(), &ranges));
        generations.mark_current(first);
        assert!(generations.is_current(atlas.untyped(), first.slot));
        assert!(generations.frontier_is_current(atlas.untyped(), &ranges));
        assert!(!generations.is_current(
            atlas.untyped(),
            AtlasSlot {
                index: 0,
                generation: 2,
            }
        ));
        generations.invalidate(atlas, 0);
        assert!(!generations.is_current(atlas.untyped(), first.slot));
    }

    #[test]
    fn adjacent_slots_coalesce_into_exact_planar_ranges_and_fewer_queue_writes() {
        let atlas = PlanarGaussian3d::from(
            (0..12)
                .map(|index| gaussian(index as f32))
                .collect::<Vec<_>>(),
        );
        let atlas_id = AssetId::<PlanarGaussian3d>::default();
        let uploads = [2_u32, 0, 4, 1]
            .into_iter()
            .map(|index| {
                snapshot_slot(
                    Some(&atlas),
                    LodAtlasSlotUpload {
                        atlas: atlas_id,
                        slot: AtlasSlot {
                            index,
                            generation: index + 1,
                        },
                        gaussians_per_slot: 2,
                    },
                    None,
                )
                .unwrap()
            })
            .collect();
        let coalesced = coalesce_atlas_uploads(uploads);
        assert_eq!(coalesced.len(), 2);
        assert_eq!(coalesced[0].start().unwrap(), 0);
        assert_eq!(coalesced[0].descriptors.len(), 3);
        assert_eq!(coalesced[0].planes.len(), 6);
        assert_eq!(coalesced[1].start().unwrap(), 8);
        assert_eq!(coalesced[1].planes.len(), 2);
        assert_eq!(
            coalesced[0]
                .planes
                .position_visibility
                .iter()
                .map(|position| position.position[0])
                .collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
        assert_eq!(coalesced.len() * 4, 8, "four planar writes per range");
        assert_eq!(4 * 4, 16, "uncoalesced baseline writes");
    }

    #[test]
    fn failed_coalesced_gpu_ranges_retain_exact_slot_payloads_for_retry() {
        let atlas = PlanarGaussian3d::from(
            (0..8)
                .map(|index| gaussian(index as f32))
                .collect::<Vec<_>>(),
        );
        let atlas_id = atlas_id(0xfeed);
        let coalesced = coalesce_atlas_uploads(
            [0_u32, 1, 3]
                .into_iter()
                .map(|slot_index| {
                    snapshot_slot(Some(&atlas), descriptor(atlas_id, slot_index, 2), None).unwrap()
                })
                .collect(),
        );
        assert_eq!(coalesced.len(), 2);

        // A batch fault followed by failure of every per-range queue write
        // decomposes adjacent ranges back into independently schedulable slots.
        let mut retries = HashMap::new();
        for upload in coalesced {
            retain_unsubmitted_coalesced_lod_atlas_upload(&mut retries, upload, |_| true);
        }
        assert_eq!(retries.len(), 3);
        for slot_index in [0_u32, 1, 3] {
            let expected = descriptor(atlas_id, slot_index, 2);
            let key = (expected.atlas, expected.slot.index);
            let PendingLodAtlasSlotUpload::Ready(actual) = &retries[&key];
            let start = slot_index as usize * 2;
            let end = start + 2;
            assert_eq!(actual.descriptor, expected);
            assert_eq!(
                actual.planes.position_visibility,
                atlas.position_visibility[start..end]
            );
            assert_eq!(
                actual.planes.spherical_harmonic,
                atlas.spherical_harmonic[start..end]
            );
            assert_eq!(actual.planes.rotation, atlas.rotation[start..end]);
            assert_eq!(actual.planes.scale_opacity, atlas.scale_opacity[start..end]);
            assert_eq!(actual.transient_generation, None);
        }

        // A mixed fallback result restores only the failed adjacent range; the
        // nonadjacent range that reached the queue gives up its CPU snapshot.
        retries.clear();
        let coalesced = coalesce_atlas_uploads(
            [0_u32, 1, 3]
                .into_iter()
                .map(|slot_index| {
                    snapshot_slot(Some(&atlas), descriptor(atlas_id, slot_index, 2), None).unwrap()
                })
                .collect(),
        );
        for upload in coalesced {
            if upload.start().unwrap() == 0 {
                retain_unsubmitted_coalesced_lod_atlas_upload(&mut retries, upload, |_| true);
            }
        }
        assert_eq!(
            retries
                .keys()
                .map(|(_, slot)| *slot)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 1])
        );
    }

    #[test]
    fn failed_upload_does_not_retry_an_obsolete_transient_generation() {
        let atlas = PlanarGaussian3d::from(vec![gaussian(0.0), gaussian(1.0)]);
        let upload =
            snapshot_slot(Some(&atlas), descriptor(atlas_id(0xbeef), 0, 2), Some(7)).unwrap();
        let mut retries = HashMap::new();
        let upload = coalesce_atlas_uploads(vec![upload]).pop().unwrap();
        retain_unsubmitted_coalesced_lod_atlas_upload(&mut retries, upload, |generation| {
            generation == Some(8)
        });
        assert!(retries.is_empty());
    }

    /// Opt in with:
    /// `RUN_GPU_LOD_ATLAS_TESTS=1 cargo test --no-default-features --features 'planar buffer_storage lod sh0 sort_std io_flexbuffers' gpu_atlas_copy_matches_cpu_oracle -- --ignored --nocapture`
    /// Add `precompute_covariance_3d` to the feature set to verify the ordered
    /// derived covariance plane in the same upload submission.
    #[test]
    #[ignore = "requires an explicitly requested wgpu adapter"]
    fn gpu_atlas_copy_matches_cpu_oracle() {
        use std::{
            sync::{Arc, mpsc},
            time::Duration,
        };

        #[cfg(feature = "precompute_covariance_3d")]
        use crate::gaussian::f32::Covariance3dOpacity;
        use bevy::render::{
            render_resource::{BufferDescriptor, BufferUsages},
            renderer::{RenderDevice, RenderQueue, WgpuWrapper},
        };

        if std::env::var("RUN_GPU_LOD_ATLAS_TESTS").as_deref() != Ok("1") {
            eprintln!("set RUN_GPU_LOD_ATLAS_TESTS=1 to execute the adapter test");
            return;
        }

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .expect("GPU atlas test requires an adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpu_covariance_dispatch_test"),
            ..Default::default()
        }))
        .expect("GPU atlas test could not create a device");
        let render_device = RenderDevice::from(device);
        let render_queue = RenderQueue(Arc::new(WgpuWrapper::new(queue)));

        let cpu_atlas = PlanarGaussian3d::from(
            (0..4)
                .map(|index| gaussian(index as f32 + 0.25))
                .collect::<Vec<_>>(),
        );
        let atlas_id = AssetId::<PlanarGaussian3d>::default();
        let adjacent = [1_u32, 0]
            .into_iter()
            .map(|index| {
                snapshot_slot(
                    Some(&cpu_atlas),
                    LodAtlasSlotUpload {
                        atlas: atlas_id,
                        slot: AtlasSlot {
                            index,
                            generation: index + 1,
                        },
                        gaussians_per_slot: 2,
                    },
                    None,
                )
                .unwrap()
            })
            .collect();
        let coalesced = coalesce_atlas_uploads(adjacent);
        assert_eq!(
            coalesced.len(),
            1,
            "adjacent slots must form one upload range"
        );
        let upload = &coalesced[0];
        assert_eq!(upload.start().unwrap(), 0);
        assert_eq!(upload.planes.len(), 4);

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let destination = |label, byte_len| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: byte_len,
                usage: storage | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let position_bytes = bytemuck::cast_slice::<_, u8>(&upload.planes.position_visibility);
        let sh_bytes = bytemuck::cast_slice::<_, u8>(&upload.planes.spherical_harmonic);
        let rotation_bytes = bytemuck::cast_slice::<_, u8>(&upload.planes.rotation);
        let scale_bytes = bytemuck::cast_slice::<_, u8>(&upload.planes.scale_opacity);
        let position_visibility = destination(
            "test_atlas_position_visibility",
            position_bytes.len() as u64,
        );
        let spherical_harmonic =
            destination("test_atlas_spherical_harmonic", sh_bytes.len() as u64);
        let rotation = destination("test_atlas_rotation", rotation_bytes.len() as u64);
        let scale_opacity = destination("test_atlas_scale_opacity", scale_bytes.len() as u64);
        #[cfg(feature = "precompute_covariance_3d")]
        let covariance_bytes = upload.planes.len() * std::mem::size_of::<Covariance3dOpacity>();
        #[cfg(feature = "precompute_covariance_3d")]
        let covariance = {
            let descriptor = BufferDescriptor {
                label: Some("test_atlas_covariance"),
                size: covariance_bytes as u64,
                usage: storage | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            };
            render_device.create_buffer(&descriptor)
        };
        let canonical_bytes =
            position_bytes.len() + sh_bytes.len() + rotation_bytes.len() + scale_bytes.len();
        #[cfg(feature = "precompute_covariance_3d")]
        let output_bytes = canonical_bytes + covariance_bytes;
        #[cfg(not(feature = "precompute_covariance_3d"))]
        let output_bytes = canonical_bytes;
        let readback = render_device.create_buffer(&BufferDescriptor {
            label: Some("test_atlas_readback"),
            size: output_bytes as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let error_scope = render_device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        #[cfg(feature = "precompute_covariance_3d")]
        let pipeline = LodCovariancePipeline::new(render_device.wgpu_device());
        let mut upload_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("test_atlas_upload_encoder"),
        });
        let mut staging = Vec::new();
        encode_canonical_buffer_copies(
            &render_device,
            &mut upload_encoder,
            LodAtlasCanonicalBuffers {
                position_visibility: &position_visibility,
                spherical_harmonic: &spherical_harmonic,
                rotation: &rotation,
                scale_opacity: &scale_opacity,
                count: upload.planes.len(),
            },
            upload,
            &mut staging,
        )
        .unwrap();
        #[cfg(feature = "precompute_covariance_3d")]
        assert_eq!(
            pipeline
                .encode_dispatch_buffers(
                    &render_device,
                    &mut upload_encoder,
                    &rotation,
                    &scale_opacity,
                    &covariance,
                    &[LodCovarianceUploadRange {
                        start: 0,
                        count: upload.planes.len() as u32,
                    }],
                )
                .unwrap(),
            1
        );
        assert_eq!(staging.len(), 4, "one staging buffer per canonical plane");
        let mut generations = LodAtlasGpuGenerations::default();
        assert!(
            upload
                .descriptors
                .iter()
                .all(|descriptor| !generations.is_current(atlas_id.untyped(), descriptor.slot))
        );
        render_queue.submit([upload_encoder.finish()]);
        for descriptor in &upload.descriptors {
            generations.mark_current(*descriptor);
        }
        assert!(
            upload
                .descriptors
                .iter()
                .all(|descriptor| generations.is_current(atlas_id.untyped(), descriptor.slot)),
            "generation proofs must publish only after the ordered upload is submitted"
        );

        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("test_covariance_readback_encoder"),
        });
        let mut readback_offset = 0_u64;
        for (buffer, len) in [
            (&position_visibility, position_bytes.len()),
            (&spherical_harmonic, sh_bytes.len()),
            (&rotation, rotation_bytes.len()),
            (&scale_opacity, scale_bytes.len()),
        ] {
            encoder.copy_buffer_to_buffer(buffer, 0, &readback, readback_offset, len as u64);
            readback_offset += len as u64;
        }
        #[cfg(feature = "precompute_covariance_3d")]
        encoder.copy_buffer_to_buffer(
            &covariance,
            0,
            &readback,
            readback_offset,
            covariance_bytes as u64,
        );
        let submission = render_queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (map_sender, map_receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = map_sender.send(result);
        });
        render_device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(30)),
            })
            .expect("GPU atlas test device poll failed");
        map_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("GPU atlas map callback did not run")
            .expect("GPU atlas output failed to map");
        let shader_error = pollster::block_on(error_scope.pop());
        assert!(
            shader_error.is_none(),
            "covariance dispatch validation failed: {shader_error:?}"
        );

        let mapped = slice.get_mapped_range();
        let mut expected_canonical = Vec::with_capacity(canonical_bytes);
        expected_canonical.extend_from_slice(position_bytes);
        expected_canonical.extend_from_slice(sh_bytes);
        expected_canonical.extend_from_slice(rotation_bytes);
        expected_canonical.extend_from_slice(scale_bytes);
        assert_eq!(&mapped[..canonical_bytes], expected_canonical);
        #[cfg(feature = "precompute_covariance_3d")]
        let actual = bytemuck::cast_slice::<u8, Covariance3dOpacity>(
            &mapped[canonical_bytes..canonical_bytes + covariance_bytes],
        );
        #[cfg(feature = "precompute_covariance_3d")]
        for ((actual, rotation), scale_opacity) in actual
            .iter()
            .zip(&upload.planes.rotation)
            .zip(&upload.planes.scale_opacity)
        {
            let expected = Covariance3dOpacity {
                cov3d: crate::gaussian::covariance::compute_covariance_3d(
                    Vec4::from_array(rotation.rotation),
                    Vec3::from_array(scale_opacity.scale),
                ),
                opacity: scale_opacity.opacity,
                pad: 0.0,
            };
            for (actual, expected) in actual.cov3d.iter().zip(expected.cov3d) {
                let tolerance = 2.0e-5 + expected.abs() * 5.0e-6;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{actual} != {expected} (tolerance {tolerance})"
                );
            }
            assert_eq!(actual.opacity, expected.opacity);
            assert_eq!(actual.pad, expected.pad);
        }
        drop(mapped);
        readback.unmap();
    }
}
