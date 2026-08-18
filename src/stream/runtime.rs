//! Bounded runtime orchestration for hierarchy selection, page requests, cache
//! residency, decoded pages, and physical candidate ranges.
//!
//! The controller is deliberately renderer- and async-runtime-neutral. A game
//! supplies a [`LodPageTransport`], calls [`LodStreamingRuntime::update`] once
//! per view/update epoch, and uploads newly decoded pages outside the render
//! pass. No operation allocates in proportion to the manifest's virtual source
//! Gaussian count.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    num::NonZeroU32,
};

use crate::{
    gaussian::{
        formats::{
            planar_3d_chunked::{
                LodBounds, LodNodeId, LodPageDescriptor, LodPageId, LodPageRange,
                PlanarGaussian3dPage,
            },
            planar_3d_lod::{GaussianLodManifest, gaussian_support_bounds},
        },
        lod_settings::{
            GaussianLodSettings, GaussianStreamingSettings, LodEffectiveStatus, LodQualityTarget,
            LodSelectionMode,
        },
    },
    io::lod::LodCodecLimits,
    stream::{
        cache::{AtlasSlot, LodPageCache, PageCacheError, PageCacheLimits, PageCacheStats},
        hierarchy::{
            CompiledManifestLodHierarchy, LodFrontier, LodHierarchy, LodSelectionError, LodView,
            ManifestHierarchyError, select_frontier_with_previous_and_visibility,
        },
        preprocess::{
            LodPagePreprocessAdmissionError, LodPagePreprocessError, LodPagePreprocessInput,
            LodPagePreprocessStats, LodPagePreprocessor,
        },
        transport::{
            LodPageTransport, LodPageTransportFailure, PagePoll, PageRequest, PageRequestPriority,
            PageRequestQueue, RequestEnqueue, RequestQueueError,
        },
    },
};

/// Fixed-stride physical addressing used by a bounded decoded GPU page atlas.
/// The stride is independent of the virtual source size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageAtlasLayout {
    pub gaussians_per_slot: u32,
}

impl PageAtlasLayout {
    pub fn new(gaussians_per_slot: u32) -> Result<Self, LodRuntimeError> {
        if gaussians_per_slot == 0 {
            Err(LodRuntimeError::ZeroAtlasStride)
        } else {
            Ok(Self { gaussians_per_slot })
        }
    }

    pub fn physical_index(self, slot: AtlasSlot, page_offset: u32) -> Result<u32, LodRuntimeError> {
        if page_offset >= self.gaussians_per_slot {
            return Err(LodRuntimeError::PageRangeExceedsAtlasStride {
                offset: page_offset,
                count: 1,
                stride: self.gaussians_per_slot,
            });
        }
        slot.index
            .checked_mul(self.gaussians_per_slot)
            .and_then(|start| start.checked_add(page_offset))
            .ok_or(LodRuntimeError::PhysicalIndexOverflow)
    }
}

/// A contiguous active representation in a generation-safe physical atlas.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodPhysicalRange {
    pub node: LodNodeId,
    pub page: LodPageId,
    pub slot: AtlasSlot,
    pub physical_start: u32,
    pub count: u32,
}

/// A complete, resident, generation-safe frontier represented by bounded
/// physical ranges. Construction is restricted to
/// [`LodStreamFrame::candidate_frontier`] so render code cannot accidentally
/// commit an arbitrary prefix or duplicate list as a complete scene
/// representation.
#[derive(Clone, Debug, PartialEq)]
pub struct LodCandidateFrontier {
    view: LodRuntimeViewId,
    physical_ranges: Vec<LodPhysicalRange>,
    candidate_count: u32,
    quality_status: LodEffectiveStatus,
    selection_view_frozen: bool,
}

impl LodCandidateFrontier {
    pub fn view(&self) -> LodRuntimeViewId {
        self.view
    }

    pub fn physical_ranges(&self) -> &[LodPhysicalRange] {
        &self.physical_ranges
    }

    pub fn candidate_count(&self) -> u32 {
        self.candidate_count
    }

    /// Quality target and achieved error of the complete resident cut used to
    /// construct this render candidate.
    pub fn quality_status(&self) -> &LodEffectiveStatus {
        &self.quality_status
    }

    /// True when this candidate was selected against the view snapshot captured
    /// on entry into [`LodSelectionMode::Frozen`].
    pub fn selection_view_frozen(&self) -> bool {
        self.selection_view_frozen
    }
}

/// Stable application-provided identity for independent camera/subview state.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LodRuntimeViewId(pub u64);

/// Opaque token that groups every camera update belonging to one application
/// frame. Per-frame request and decoded-byte budgets are shared by all updates
/// made with the same token.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LodRuntimeFrameId(u64);

impl LodRuntimeFrameId {
    pub fn sequence(self) -> u64 {
        self.0
    }
}

impl LodPhysicalRange {
    pub fn end(self) -> Option<u32> {
        self.physical_start.checked_add(self.count)
    }
}

/// Observable result for one controller update.
#[derive(Clone, Debug, PartialEq)]
pub struct LodStreamFrame {
    view: LodRuntimeViewId,
    frontier: LodFrontier<LodNodeId>,
    physical_ranges: Vec<LodPhysicalRange>,
    selection_view_frozen: bool,
    /// True when every visible root is covered by a resident selected node.
    complete_resident_cut: bool,
    cache: PageCacheStats,
    queued_requests: u32,
    /// Transport requests plus admitted preprocessing work. Keeping both in a
    /// single pipeline count prevents completion checks from declaring idle
    /// while validated pages are still pending publication.
    in_flight_requests: u32,
    preprocess: LodPagePreprocessStats,
    /// Requests paused after a decoded page could not displace pinned fallbacks.
    capacity_blocked_requests: u32,
    started_pages: Vec<LodPageId>,
    completed_pages: Vec<LodPageId>,
    /// Pages whose encoded payload reached preprocessing but failed checksum,
    /// codec, or support-bound validation during this update. Unlike
    /// `failed_pages`, these are emitted on the first rejection rather than
    /// only after retry exhaustion.
    preprocess_failed_pages: Vec<LodPageId>,
    failed_pages: Vec<LodPageId>,
}

impl LodStreamFrame {
    pub fn view(&self) -> LodRuntimeViewId {
        self.view
    }

    pub fn frontier(&self) -> &LodFrontier<LodNodeId> {
        &self.frontier
    }

    pub fn physical_ranges(&self) -> &[LodPhysicalRange] {
        &self.physical_ranges
    }

    /// True when selection and streaming demand used this view's frozen camera
    /// snapshot. Residency and physical ranges are still current for this frame.
    pub fn selection_view_frozen(&self) -> bool {
        self.selection_view_frozen
    }

    pub fn has_complete_resident_cut(&self) -> bool {
        self.complete_resident_cut
    }

    pub fn cache_stats(&self) -> PageCacheStats {
        self.cache
    }

    pub fn queued_requests(&self) -> u32 {
        self.queued_requests
    }

    pub fn in_flight_requests(&self) -> u32 {
        self.in_flight_requests
    }

    pub fn preprocess_stats(&self) -> LodPagePreprocessStats {
        self.preprocess
    }

    pub fn capacity_blocked_requests(&self) -> u32 {
        self.capacity_blocked_requests
    }

    pub fn started_pages(&self) -> &[LodPageId] {
        &self.started_pages
    }

    pub fn completed_pages(&self) -> &[LodPageId] {
        &self.completed_pages
    }

    pub fn preprocess_failed_pages(&self) -> &[LodPageId] {
        &self.preprocess_failed_pages
    }

    pub fn failed_pages(&self) -> &[LodPageId] {
        &self.failed_pages
    }

    /// Exact count represented by the physical ranges emitted this update.
    pub fn candidate_count(&self) -> u64 {
        self.physical_ranges
            .iter()
            .map(|range| u64::from(range.count))
            .sum()
    }

    /// Validates that this update contains a complete resident cut with exact,
    /// non-overlapping physical ranges and a representable bounded count, then
    /// freezes it without expanding a candidate-sized index vector.
    pub fn candidate_frontier(&self, limit: u32) -> Result<LodCandidateFrontier, LodRuntimeError> {
        if !self.complete_resident_cut {
            return Err(LodRuntimeError::NoResidentFrontier);
        }
        let count = self.candidate_count();
        if count != self.frontier.status.active_gaussians {
            return Err(LodRuntimeError::CandidateCountMismatch {
                frontier: self.frontier.status.active_gaussians,
                physical: count,
            });
        }
        if count > u64::from(limit) {
            return Err(LodRuntimeError::CandidateExpansionLimit { count, limit });
        }
        let mut intervals = self
            .physical_ranges
            .iter()
            .map(|range| {
                range
                    .end()
                    .map(|end| (range.physical_start, end))
                    .ok_or(LodRuntimeError::PhysicalIndexOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        intervals.sort_unstable();
        for pair in intervals.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(LodRuntimeError::OverlappingPhysicalRanges {
                    previous_end: pair[0].1,
                    next_start: pair[1].0,
                });
            }
        }

        Ok(LodCandidateFrontier {
            view: self.view,
            physical_ranges: self.physical_ranges.clone(),
            candidate_count: count as u32,
            quality_status: self.frontier.status,
            selection_view_frozen: self.selection_view_frozen,
        })
    }
}

#[derive(Clone, Debug)]
struct InFlight<Ticket> {
    ticket: Ticket,
    request: PageRequest,
}

#[derive(Clone, Debug, Default)]
struct LodRuntimeViewState {
    previous_frontier: Vec<LodNodeId>,
    /// Policy that produced `previous_frontier`. Hysteresis history is valid
    /// only while this remains exactly unchanged.
    previous_lod_policy: Option<LodHysteresisPolicy>,
    /// Camera snapshot used only for selection and page-demand priority. It does
    /// not contain or freeze residency/physical availability.
    frozen_selection_view: Option<LodView>,
    selected_frontier: BTreeSet<LodPageId>,
    pinned_frontier: BTreeSet<LodPageId>,
    requested_pages: BTreeSet<LodPageId>,
    requested_pages_frame: LodRuntimeFrameId,
}

impl LodRuntimeViewState {
    fn selection_view(&mut self, current: LodView, mode: LodSelectionMode) -> LodView {
        match mode {
            LodSelectionMode::Dynamic => {
                self.frozen_selection_view = None;
                current
            }
            LodSelectionMode::Frozen => *self.frozen_selection_view.get_or_insert(current),
        }
    }

    fn hysteresis_frontier(&self, lod_settings: &GaussianLodSettings) -> &[LodNodeId] {
        if self.previous_lod_policy == Some(LodHysteresisPolicy::from(lod_settings)) {
            &self.previous_frontier
        } else {
            &[]
        }
    }

    fn commit_frontier(&mut self, frontier: &[LodNodeId], lod_settings: &GaussianLodSettings) {
        self.previous_frontier.clear();
        self.previous_frontier.extend_from_slice(frontier);
        self.previous_lod_policy = Some(LodHysteresisPolicy::from(lod_settings));
    }
}

/// Canonical subset of settings that can change hierarchy selection or the
/// interpretation of its previous cut.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LodHysteresisPolicy {
    target: LodQualityTarget,
    hysteresis: f32,
    frustum_culling: bool,
    frustum_margin: f32,
    max_active_gaussians: u64,
    max_traversal_nodes_per_view: u32,
}

impl From<&GaussianLodSettings> for LodHysteresisPolicy {
    fn from(settings: &GaussianLodSettings) -> Self {
        Self {
            target: settings.quality_target(),
            hysteresis: settings.hysteresis,
            frustum_culling: settings.frustum_culling,
            frustum_margin: settings.frustum_margin,
            max_active_gaussians: settings.budgets.max_active_gaussians,
            max_traversal_nodes_per_view: settings.budgets.max_traversal_nodes_per_view,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LodRuntimeStructuralSettings {
    max_resident_gaussians: u64,
    max_resident_bytes: u64,
    max_resident_pages: u32,
    max_pending_requests: u32,
    max_encoded_page_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct SharedPageNodeRange {
    node: LodNodeId,
    range: LodPageRange,
    bounds: LodBounds,
}

impl LodRuntimeStructuralSettings {
    fn new(lod: &GaussianLodSettings, streaming: &GaussianStreamingSettings) -> Self {
        Self {
            max_resident_gaussians: lod.budgets.max_resident_gaussians,
            max_resident_bytes: lod.budgets.max_resident_bytes,
            max_resident_pages: lod.budgets.max_resident_pages,
            max_pending_requests: lod.budgets.max_pending_requests,
            max_encoded_page_bytes: streaming.effective_max_encoded_page_bytes(),
        }
    }

    fn validate_compatible(self, next: Self) -> Result<(), LodRuntimeError> {
        for (field, matches) in [
            (
                "budgets.max_resident_gaussians",
                self.max_resident_gaussians == next.max_resident_gaussians,
            ),
            (
                "budgets.max_resident_bytes",
                self.max_resident_bytes == next.max_resident_bytes,
            ),
            (
                "budgets.max_resident_pages",
                self.max_resident_pages == next.max_resident_pages,
            ),
            (
                "budgets.max_pending_requests",
                self.max_pending_requests == next.max_pending_requests,
            ),
            (
                "streaming.effective_max_encoded_page_bytes",
                self.max_encoded_page_bytes == next.max_encoded_page_bytes,
            ),
        ] {
            if !matches {
                return Err(LodRuntimeError::StructuralSettingsChanged(field));
            }
        }
        Ok(())
    }
}

/// Long-lived bounded state for one virtual Gaussian cloud.
pub struct LodStreamingRuntime<T: LodPageTransport> {
    hierarchy: CompiledManifestLodHierarchy,
    descriptors: BTreeMap<LodPageId, LodPageDescriptor>,
    /// Slice-local validation plans for physical pages shared by logical nodes.
    /// Single-node pages retain the cheaper descriptor-wide preprocessing path.
    shared_page_node_ranges: BTreeMap<LodPageId, Vec<SharedPageNodeRange>>,
    cache: LodPageCache,
    decoded_pages: BTreeMap<LodPageId, PlanarGaussian3dPage>,
    queue: PageRequestQueue,
    transport: T,
    in_flight: BTreeMap<LodPageId, InFlight<T::Ticket>>,
    preprocessor: LodPagePreprocessor,
    preprocess_failures: BTreeMap<LodPageId, LodPagePreprocessError>,
    /// Frame in which a rejected decoded payload queued its bounded retry.
    /// The retry remains in the ordinary request queue, but cannot start until
    /// the next frame. Package transports use that boundary to invalidate a
    /// cached encoded payload before any retry can observe it again.
    preprocess_retry_deferred_frame: BTreeMap<LodPageId, LodRuntimeFrameId>,
    transport_failures: BTreeMap<LodPageId, LodPageTransportFailure>,
    attempts: BTreeMap<LodPageId, u32>,
    terminal_failures: BTreeSet<LodPageId>,
    terminal_requests: BTreeMap<LodPageId, PageRequest>,
    capacity_blocked: BTreeMap<LodPageId, PageRequest>,
    views: BTreeMap<LodRuntimeViewId, LodRuntimeViewState>,
    atlas_layout: PageAtlasLayout,
    pending_request_capacity: usize,
    structural_settings: LodRuntimeStructuralSettings,
    largest_decoded_page: (LodPageId, u64),
    epoch: u64,
    frame_decoded_bytes: u64,
    frame_request_starts: u32,
    frame_finished: bool,
}

impl<T: LodPageTransport> Drop for LodStreamingRuntime<T> {
    fn drop(&mut self) {
        let in_flight = std::mem::take(&mut self.in_flight);
        for request in in_flight.into_values() {
            self.transport.cancel(&request.ticket);
        }
    }
}

impl<T: LodPageTransport> LodStreamingRuntime<T> {
    pub fn new(
        manifest: GaussianLodManifest,
        transport: T,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
    ) -> Result<Self, LodRuntimeError> {
        lod_settings
            .validate()
            .map_err(|error| LodRuntimeError::InvalidSettings(error.to_string()))?;
        streaming_settings
            .validate()
            .map_err(|error| LodRuntimeError::InvalidSettings(error.to_string()))?;
        let hierarchy = CompiledManifestLodHierarchy::new(manifest)
            .map_err(LodRuntimeError::InvalidManifest)?;
        let descriptors = hierarchy
            .manifest()
            .pages
            .iter()
            .cloned()
            .map(|descriptor| (descriptor.id, descriptor))
            .collect::<BTreeMap<_, _>>();
        let mut shared_page_node_ranges = BTreeMap::<_, Vec<_>>::new();
        for node in &hierarchy.manifest().nodes {
            shared_page_node_ranges
                .entry(node.representation.page)
                .or_default()
                .push(SharedPageNodeRange {
                    node: node.id,
                    range: node.representation,
                    bounds: node.bounds,
                });
        }
        shared_page_node_ranges.retain(|_, ranges| {
            ranges.sort_unstable_by_key(|entry| entry.range.offset);
            ranges.len() > 1
        });
        let max_decoded_page_bytes = lod_settings
            .budgets
            .max_resident_bytes
            .min(lod_settings.budgets.max_upload_bytes_per_frame);
        let max_encoded_page_bytes = streaming_settings.effective_max_encoded_page_bytes();
        if max_encoded_page_bytes < 44 {
            return Err(LodRuntimeError::EncodedPageLimitTooSmall {
                limit: max_encoded_page_bytes,
                minimum: 44,
            });
        }
        for descriptor in descriptors.values() {
            if descriptor.decoded_len > max_decoded_page_bytes {
                return Err(LodRuntimeError::PageDecodedBytesExceedLimit {
                    page: descriptor.id,
                    actual: descriptor.decoded_len,
                    limit: max_decoded_page_bytes,
                });
            }
            if u64::from(descriptor.gaussian_count) > lod_settings.budgets.max_resident_gaussians {
                return Err(LodRuntimeError::PageGaussiansExceedLimit {
                    page: descriptor.id,
                    actual: u64::from(descriptor.gaussian_count),
                    limit: lod_settings.budgets.max_resident_gaussians,
                });
            }
            if let Some(storage) = &descriptor.storage
                && storage.encoded_len > max_encoded_page_bytes
            {
                return Err(LodRuntimeError::PageEncodedBytesExceedLimit {
                    page: descriptor.id,
                    actual: storage.encoded_len,
                    limit: max_encoded_page_bytes,
                });
            }
        }
        let maximum_page_gaussians = descriptors
            .values()
            .map(|descriptor| descriptor.gaussian_count)
            .max()
            .ok_or(LodRuntimeError::ManifestHasNoPages)?;
        let largest_decoded_page = descriptors
            .values()
            .map(|descriptor| (descriptor.id, descriptor.decoded_len))
            .max_by_key(|(_, decoded_len)| *decoded_len)
            .ok_or(LodRuntimeError::ManifestHasNoPages)?;
        let physical_address_count = u64::from(lod_settings.budgets.max_resident_pages)
            .checked_mul(u64::from(maximum_page_gaussians))
            .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
        if physical_address_count > u64::from(u32::MAX) + 1 {
            return Err(LodRuntimeError::AtlasAddressSpaceOverflow {
                slots: lod_settings.budgets.max_resident_pages,
                stride: maximum_page_gaussians,
            });
        }
        let cache = LodPageCache::new(PageCacheLimits::from(&lod_settings.budgets))
            .map_err(LodRuntimeError::Cache)?;
        let queue_capacity = usize::try_from(lod_settings.budgets.max_pending_requests)
            .map_err(|_| LodRuntimeError::RequestCapacityOverflow)?;
        let queue = PageRequestQueue::new(queue_capacity).map_err(LodRuntimeError::Queue)?;
        let preprocess_capacity = queue_capacity.min(
            usize::try_from(streaming_settings.max_concurrent_requests)
                .map_err(|_| LodRuntimeError::RequestCapacityOverflow)?,
        );
        let preprocess_byte_capacity = max_encoded_page_bytes
            .checked_add(lod_settings.budgets.max_upload_bytes_per_frame)
            .ok_or(LodRuntimeError::PreprocessAdmission(
                LodPagePreprocessAdmissionError::ByteLengthOverflow,
            ))?;
        let preprocessor =
            LodPagePreprocessor::with_byte_capacity(preprocess_capacity, preprocess_byte_capacity)
                .map_err(LodRuntimeError::PreprocessAdmission)?;
        for descriptor in descriptors.values() {
            let encoded_bytes = descriptor
                .storage
                .as_ref()
                .map_or(max_encoded_page_bytes, |storage| storage.encoded_len);
            preprocessor
                .validate_job_bytes(encoded_bytes, descriptor.decoded_len)
                .map_err(LodRuntimeError::PreprocessAdmission)?;
        }
        Ok(Self {
            hierarchy,
            descriptors,
            shared_page_node_ranges,
            cache,
            decoded_pages: BTreeMap::new(),
            queue,
            transport,
            in_flight: BTreeMap::new(),
            preprocessor,
            preprocess_failures: BTreeMap::new(),
            preprocess_retry_deferred_frame: BTreeMap::new(),
            transport_failures: BTreeMap::new(),
            attempts: BTreeMap::new(),
            terminal_failures: BTreeSet::new(),
            terminal_requests: BTreeMap::new(),
            capacity_blocked: BTreeMap::new(),
            views: BTreeMap::new(),
            atlas_layout: PageAtlasLayout::new(maximum_page_gaussians)?,
            pending_request_capacity: queue_capacity,
            structural_settings: LodRuntimeStructuralSettings::new(
                lod_settings,
                streaming_settings,
            ),
            largest_decoded_page,
            epoch: 0,
            frame_decoded_bytes: 0,
            frame_request_starts: 0,
            frame_finished: true,
        })
    }

    pub fn hierarchy(&self) -> &CompiledManifestLodHierarchy {
        &self.hierarchy
    }

    pub fn cache(&self) -> &LodPageCache {
        &self.cache
    }

    pub fn decoded_page(&self, page: LodPageId) -> Option<&PlanarGaussian3dPage> {
        self.decoded_pages.get(&page)
    }

    pub fn atlas_layout(&self) -> PageAtlasLayout {
        self.atlas_layout
    }

    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Pages whose retry budget is exhausted. Terminal pages are not
    /// automatically enqueued by later updates, even if they remain visible.
    pub fn terminal_failures(&self) -> &BTreeSet<LodPageId> {
        &self.terminal_failures
    }

    pub fn is_terminal_failure(&self, page: LodPageId) -> bool {
        self.terminal_failures.contains(&page)
    }

    /// Last typed preprocessing failure observed for a page. The entry is
    /// retained across bounded retries and cleared after success or an explicit
    /// terminal-page retry.
    pub fn page_preprocess_error(&self, page: LodPageId) -> Option<&LodPagePreprocessError> {
        self.preprocess_failures.get(&page)
    }

    /// Last normalized transport failure observed for a page. The entry is
    /// retained across bounded retries and cleared after a payload succeeds or
    /// an explicit terminal-page retry.
    pub fn page_transport_failure(&self, page: LodPageId) -> Option<&LodPageTransportFailure> {
        self.transport_failures.get(&page)
    }

    /// Number of transport starts attempted since the last success or explicit
    /// terminal-page retry.
    pub fn page_attempts(&self, page: LodPageId) -> Option<u32> {
        self.attempts.get(&page).copied()
    }

    /// Explicitly requeues a terminal page using its last request priority.
    /// Attempt accounting restarts at zero; transport begin occurs on the next
    /// bounded update. Returns `false` when the page is not terminal.
    pub fn retry_terminal_failure(&mut self, page: LodPageId) -> Result<bool, LodRuntimeError> {
        if !self.terminal_failures.contains(&page) {
            return Ok(false);
        }
        let request = self
            .terminal_requests
            .remove(&page)
            .ok_or(LodRuntimeError::MissingTerminalRequest(page))?;
        let attempts = self.attempts.remove(&page);
        self.terminal_failures.remove(&page);
        match self.enqueue_pending_request(request) {
            RequestEnqueue::Rejected => {
                self.terminal_failures.insert(page);
                self.terminal_requests.insert(page, request);
                if let Some(attempts) = attempts {
                    self.attempts.insert(page, attempts);
                }
                Err(LodRuntimeError::RetryQueueRejected(page))
            }
            RequestEnqueue::Enqueued
            | RequestEnqueue::Promoted
            | RequestEnqueue::Duplicate
            | RequestEnqueue::Replaced(_) => {
                self.preprocess_failures.remove(&page);
                self.preprocess_retry_deferred_frame.remove(&page);
                self.transport_failures.remove(&page);
                Ok(true)
            }
        }
    }

    /// Polls bounded page work, selects a complete resident frontier, schedules
    /// missing pages, and emits generation-safe physical ranges for this view.
    pub fn update(
        &mut self,
        view: LodView,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
    ) -> Result<LodStreamFrame, LodRuntimeError> {
        let frame = self.begin_frame();
        let result = self.update_view_in_frame(
            frame,
            LodRuntimeViewId::default(),
            view,
            lod_settings,
            streaming_settings,
        );
        let _ = self.finish_frame(frame);
        result
    }

    /// Updates one camera as one complete orchestration frame. For multiple
    /// cameras in the same application frame, call [`Self::begin_frame`] once
    /// and use [`Self::update_view_in_frame`] so work budgets are shared.
    pub fn update_view(
        &mut self,
        view_id: LodRuntimeViewId,
        view: LodView,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
    ) -> Result<LodStreamFrame, LodRuntimeError> {
        let frame = self.begin_frame();
        let result =
            self.update_view_in_frame(frame, view_id, view, lod_settings, streaming_settings);
        let _ = self.finish_frame(frame);
        result
    }

    /// Starts one application frame and resets its shared decoded-byte and
    /// transport-start accounting. Pass the returned token to every camera via
    /// [`Self::update_view_in_frame`].
    pub fn begin_frame(&mut self) -> LodRuntimeFrameId {
        if self.epoch != 0 && !self.frame_finished {
            self.reconcile_frame_demand(LodRuntimeFrameId(self.epoch));
        }
        self.epoch = self.epoch.wrapping_add(1).max(1);
        self.frame_decoded_bytes = 0;
        self.frame_request_starts = 0;
        self.frame_finished = false;
        LodRuntimeFrameId(self.epoch)
    }

    /// Cancels page work that no view requested in `frame`.
    ///
    /// Multi-view callers should invoke this once after their final
    /// [`Self::update_view_in_frame`] call. Starting the next frame also
    /// performs this reconciliation as a fail-safe, but explicit completion
    /// releases stale camera-cut work one frame earlier.
    pub fn finish_frame(&mut self, frame: LodRuntimeFrameId) -> Result<(), LodRuntimeError> {
        let current = LodRuntimeFrameId(self.epoch);
        if frame != current || frame.0 == 0 {
            return Err(LodRuntimeError::InvalidFrameToken {
                expected: current,
                actual: frame,
            });
        }
        if !self.frame_finished {
            self.reconcile_frame_demand(frame);
        }
        Ok(())
    }

    /// Multi-camera update that shares per-frame work budgets with every other
    /// update using `frame`. A stale token is rejected so callers cannot
    /// accidentally reset or reuse budget accounting.
    pub fn update_view_in_frame(
        &mut self,
        frame: LodRuntimeFrameId,
        view_id: LodRuntimeViewId,
        view: LodView,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
    ) -> Result<LodStreamFrame, LodRuntimeError> {
        let current = LodRuntimeFrameId(self.epoch);
        if frame != current || frame.0 == 0 {
            return Err(LodRuntimeError::InvalidFrameToken {
                expected: current,
                actual: frame,
            });
        }
        if self.frame_finished {
            return Err(LodRuntimeError::FrameAlreadyFinished(frame));
        }
        lod_settings
            .validate()
            .map_err(|error| LodRuntimeError::InvalidSettings(error.to_string()))?;
        streaming_settings
            .validate()
            .map_err(|error| LodRuntimeError::InvalidSettings(error.to_string()))?;
        self.structural_settings
            .validate_compatible(LodRuntimeStructuralSettings::new(
                lod_settings,
                streaming_settings,
            ))?;
        if self.largest_decoded_page.1 > lod_settings.budgets.max_upload_bytes_per_frame {
            return Err(LodRuntimeError::PageDecodedBytesExceedLimit {
                page: self.largest_decoded_page.0,
                actual: self.largest_decoded_page.1,
                limit: lod_settings.budgets.max_upload_bytes_per_frame,
            });
        }

        let selection_view = self
            .views
            .entry(view_id)
            .or_default()
            .selection_view(view, lod_settings.selection_mode);
        let selection_view_frozen = lod_settings.selection_mode == LodSelectionMode::Frozen;

        let mut completed_pages = Vec::new();
        let mut preprocess_failed_pages = Vec::new();
        let mut failed_pages = Vec::new();
        self.poll_pages(frame, lod_settings, streaming_settings, &mut failed_pages)?;

        // Establish this view's current-frame demand before publishing any
        // completed preprocessing result. This prevents a camera cut from
        // committing a stale page merely because its worker won a race with
        // demand reconciliation.
        let mut frontier = self.select_frontier(view_id, selection_view, lod_settings)?;
        let requested_pages = self.record_frame_demand(frame, view_id, &frontier);
        frontier.status.requested_pages = requested_pages;
        self.commit_preprocessed_pages(
            frame,
            lod_settings,
            streaming_settings,
            &mut completed_pages,
            &mut preprocess_failed_pages,
            &mut failed_pages,
        )?;
        if !completed_pages.is_empty() {
            // Newly resident pages may refine the cut immediately, while all
            // expensive verification/decode work still occurred outside this
            // application-thread call on native builds.
            frontier = self.select_frontier(view_id, selection_view, lod_settings)?;
            let requested_pages = self.record_frame_demand(frame, view_id, &frontier);
            frontier.status.requested_pages = requested_pages;
        }

        if self.update_frontier_pins(view_id, &frontier)? {
            self.wake_capacity_blocked();
        }
        let physical_ranges = self.physical_ranges(&frontier)?;
        let represented_count: u64 = physical_ranges
            .iter()
            .map(|range| u64::from(range.count))
            .sum();
        if represented_count != frontier.status.active_gaussians {
            return Err(LodRuntimeError::CandidateCountMismatch {
                frontier: frontier.status.active_gaussians,
                physical: represented_count,
            });
        }
        // A missing child is covered by its selected resident ancestor, but a
        // missing visible root has no possible fallback. Do not let a partial
        // multi-root forest acquire the private GPU commit capability.
        let complete_resident_cut = !frontier.requested_nodes.iter().any(|node| {
            self.hierarchy.parent(*node).is_none()
                && !frontier.nodes.iter().any(|selected| selected == node)
        });

        self.enqueue_missing(&frontier, selection_view)?;
        let started_pages =
            self.start_requests(lod_settings, streaming_settings, &mut failed_pages);
        self.views
            .entry(view_id)
            .or_default()
            .commit_frontier(&frontier.nodes, lod_settings);

        Ok(LodStreamFrame {
            view: view_id,
            frontier,
            physical_ranges,
            selection_view_frozen,
            complete_resident_cut,
            cache: self.cache.stats(),
            queued_requests: self.queue.len().try_into().unwrap_or(u32::MAX),
            in_flight_requests: self
                .in_flight
                .len()
                .saturating_add(self.preprocessor.len())
                .try_into()
                .unwrap_or(u32::MAX),
            preprocess: self.preprocessor.stats(),
            capacity_blocked_requests: self.capacity_blocked.len().try_into().unwrap_or(u32::MAX),
            started_pages,
            completed_pages,
            preprocess_failed_pages,
            failed_pages,
        })
    }

    fn reconcile_frame_demand(&mut self, frame: LodRuntimeFrameId) {
        let demanded = self
            .views
            .values()
            .filter(|view| view.requested_pages_frame == frame)
            .flat_map(|view| view.requested_pages.iter().copied())
            .collect::<BTreeSet<_>>();

        let cancelled_queued = self
            .queue
            .page_ids()
            .filter(|page| !demanded.contains(page))
            .collect::<Vec<_>>();
        for page in &cancelled_queued {
            self.queue.remove(*page);
            self.attempts.remove(page);
            self.preprocess_retry_deferred_frame.remove(page);
            self.transport_failures.remove(page);
        }

        let cancelled_in_flight = self
            .in_flight
            .keys()
            .filter(|page| !demanded.contains(page))
            .copied()
            .collect::<Vec<_>>();
        for page in &cancelled_in_flight {
            if let Some(request) = self.in_flight.remove(page) {
                self.transport.cancel(&request.ticket);
            }
            self.attempts.remove(page);
            self.preprocess_retry_deferred_frame.remove(page);
            self.transport_failures.remove(page);
        }

        let cancelled_preprocessing = self
            .preprocessor
            .page_ids()
            .into_iter()
            .filter(|page| !demanded.contains(page))
            .collect::<Vec<_>>();
        for page in &cancelled_preprocessing {
            self.preprocessor.cancel(*page);
            self.preprocess_failures.remove(page);
            self.preprocess_retry_deferred_frame.remove(page);
            self.transport_failures.remove(page);
            self.attempts.remove(page);
        }

        let cancelled_capacity_blocked = self
            .capacity_blocked
            .keys()
            .filter(|page| !demanded.contains(page))
            .copied()
            .collect::<Vec<_>>();
        for page in &cancelled_capacity_blocked {
            self.capacity_blocked.remove(page);
            self.attempts.remove(page);
            self.preprocess_retry_deferred_frame.remove(page);
            self.transport_failures.remove(page);
        }

        self.frame_finished = true;
    }

    /// Releases a view's fallback holds without affecting other cameras.
    pub fn remove_view(&mut self, view_id: LodRuntimeViewId) -> Result<bool, LodRuntimeError> {
        let Some(state) = self.views.remove(&view_id) else {
            return Ok(false);
        };
        for page in state.pinned_frontier {
            self.cache
                .unpin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
        }
        self.wake_capacity_blocked();
        Ok(true)
    }

    fn select_frontier(
        &self,
        view_id: LodRuntimeViewId,
        view: LodView,
        lod_settings: &GaussianLodSettings,
    ) -> Result<LodFrontier<LodNodeId>, LodRuntimeError> {
        let previous_frontier = self
            .views
            .get(&view_id)
            .map(|state| state.hysteresis_frontier(lod_settings))
            .unwrap_or_default();
        select_frontier_with_previous_and_visibility(
            &self.hierarchy,
            &|node| {
                self.hierarchy
                    .page(node)
                    .is_some_and(|page| self.cache.contains(page))
            },
            view,
            lod_settings,
            previous_frontier,
            |_, metrics| {
                !lod_settings.frustum_culling
                    || view.node_is_visible(metrics, lod_settings.frustum_margin)
            },
        )
        .map_err(LodRuntimeError::Selection)
    }

    fn record_frame_demand(
        &mut self,
        frame: LodRuntimeFrameId,
        view_id: LodRuntimeViewId,
        frontier: &LodFrontier<LodNodeId>,
    ) -> u32 {
        let requested_pages = frontier
            .requested_nodes
            .iter()
            .filter_map(|node| self.hierarchy.page(*node))
            .collect::<BTreeSet<_>>();
        let requested_page_count = requested_pages.len().try_into().unwrap_or(u32::MAX);
        let view_state = self.views.entry(view_id).or_default();
        view_state.requested_pages = requested_pages;
        view_state.requested_pages_frame = frame;
        requested_page_count
    }

    fn demanded_in_frame(&self, frame: LodRuntimeFrameId, page: LodPageId) -> bool {
        self.views
            .values()
            .any(|view| view.requested_pages_frame == frame && view.requested_pages.contains(&page))
    }

    fn poll_pages(
        &mut self,
        frame: LodRuntimeFrameId,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
        failed_pages: &mut Vec<LodPageId>,
    ) -> Result<(), LodRuntimeError> {
        let pages = self.in_flight.keys().copied().collect::<Vec<_>>();
        for page_id in pages {
            let Some(in_flight) = self.in_flight.get(&page_id).cloned() else {
                continue;
            };
            let descriptor = self
                .descriptors
                .get(&page_id)
                .cloned()
                .ok_or(LodRuntimeError::MissingPageDescriptor(page_id))?;
            let encoded_bytes = in_flight
                .request
                .expected_bytes
                .unwrap_or_else(|| streaming_settings.effective_max_encoded_page_bytes());
            if !self
                .preprocessor
                .has_capacity_for(encoded_bytes, descriptor.decoded_len)
            {
                // Leave the transport ticket untouched until exact count and
                // byte admission is available.
                continue;
            }
            match self.transport.poll(&in_flight.ticket) {
                PagePoll::Pending => {}
                PagePoll::Ready(payload) => {
                    self.in_flight.remove(&page_id);
                    self.preprocess_failures.remove(&page_id);
                    self.transport_failures.remove(&page_id);
                    let limits = page_codec_limits(
                        &descriptor,
                        streaming_settings.effective_max_encoded_page_bytes(),
                    );
                    self.preprocessor
                        .submit(LodPagePreprocessInput {
                            request: in_flight.request,
                            payload,
                            descriptor,
                            limits,
                            max_encoded_page_bytes: streaming_settings
                                .effective_max_encoded_page_bytes(),
                            support_sigma: self.hierarchy.manifest().build.settings.support_sigma,
                        })
                        .map_err(LodRuntimeError::PreprocessAdmission)?;
                }
                PagePoll::Failed(error) => {
                    self.in_flight.remove(&page_id);
                    self.transport_failures
                        .insert(page_id, T::classify_error(&error));
                    self.retry_or_fail(
                        in_flight.request,
                        streaming_settings.retry_limit,
                        failed_pages,
                    );
                }
            }
        }
        self.preprocessor.advance(
            frame.sequence(),
            NonZeroU32::new(
                lod_settings
                    .budgets
                    .max_cooperative_preprocess_gaussians_per_frame,
            )
            .expect("validated cooperative preprocessing budget is non-zero"),
        );
        Ok(())
    }

    fn commit_preprocessed_pages(
        &mut self,
        frame: LodRuntimeFrameId,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
        completed_pages: &mut Vec<LodPageId>,
        preprocess_failed_pages: &mut Vec<LodPageId>,
        failed_pages: &mut Vec<LodPageId>,
    ) -> Result<(), LodRuntimeError> {
        let ready_pages = self.preprocessor.ready_page_ids();
        for page_id in ready_pages {
            // A later view in this frame may still demand the page. Leave it
            // ready until demand is known instead of publishing or discarding
            // it speculatively.
            if !self.demanded_in_frame(frame, page_id) {
                continue;
            }
            let descriptor = self
                .descriptors
                .get(&page_id)
                .cloned()
                .ok_or(LodRuntimeError::MissingPageDescriptor(page_id))?;
            if self
                .frame_decoded_bytes
                .checked_add(descriptor.decoded_len)
                .is_none_or(|bytes| bytes > lod_settings.budgets.max_upload_bytes_per_frame)
            {
                continue;
            }
            let Some(output) = self.preprocessor.take_ready(page_id) else {
                continue;
            };
            let page = match output.result {
                Ok(page) => page,
                Err(error) => {
                    self.preprocess_failures.insert(page_id, error);
                    preprocess_failed_pages.push(page_id);
                    self.preprocess_retry_deferred_frame.insert(page_id, frame);
                    self.retry_or_fail(
                        output.request,
                        streaming_settings.retry_limit,
                        failed_pages,
                    );
                    continue;
                }
            };
            if let Some(ranges) = self.shared_page_node_ranges.get(&page_id)
                && let Err(error) = validate_shared_page_node_ranges(
                    &page,
                    ranges,
                    self.hierarchy.manifest().build.settings.support_sigma,
                )
            {
                self.preprocess_failures.insert(page_id, error);
                preprocess_failed_pages.push(page_id);
                self.preprocess_retry_deferred_frame.insert(page_id, frame);
                self.retry_or_fail(output.request, streaming_settings.retry_limit, failed_pages);
                continue;
            }
            self.frame_decoded_bytes = self
                .frame_decoded_bytes
                .checked_add(descriptor.decoded_len)
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
            let insertion = match self.cache.insert(
                page_id,
                descriptor.decoded_len,
                u64::from(descriptor.gaussian_count),
                self.epoch,
            ) {
                Ok(insertion) => insertion,
                Err(PageCacheError::InsufficientEvictableCapacity) => {
                    // Keep only bounded request metadata. Retrying the same
                    // payload before pin/cache state changes would redownload
                    // and decode it forever.
                    self.capacity_blocked.insert(page_id, output.request);
                    continue;
                }
                Err(error) => return Err(LodRuntimeError::Cache(error)),
            };
            for evicted in insertion.evicted {
                self.decoded_pages.remove(&evicted);
            }
            self.decoded_pages.insert(page_id, page);
            self.clear_failure_state(page_id);
            self.capacity_blocked.remove(&page_id);
            self.wake_capacity_blocked();
            completed_pages.push(page_id);
        }
        Ok(())
    }

    fn retry_or_fail(
        &mut self,
        request: PageRequest,
        retry_limit: u32,
        failed_pages: &mut Vec<LodPageId>,
    ) {
        if self.terminal_failures.contains(&request.page_id) {
            self.preprocess_retry_deferred_frame
                .remove(&request.page_id);
            return;
        }
        let attempts = self.attempts.get(&request.page_id).copied().unwrap_or(0);
        let maximum_attempts = retry_limit.saturating_add(1);
        if attempts < maximum_attempts {
            self.queue.enqueue(request);
        } else if self.terminal_failures.insert(request.page_id) {
            self.queue.remove(request.page_id);
            self.preprocess_retry_deferred_frame
                .remove(&request.page_id);
            self.terminal_requests.insert(request.page_id, request);
            failed_pages.push(request.page_id);
        }
    }

    fn clear_failure_state(&mut self, page: LodPageId) {
        self.attempts.remove(&page);
        self.terminal_failures.remove(&page);
        self.terminal_requests.remove(&page);
        self.preprocess_failures.remove(&page);
        self.preprocess_retry_deferred_frame.remove(&page);
        self.transport_failures.remove(&page);
    }

    fn wake_capacity_blocked(&mut self) {
        let requests = self.capacity_blocked.values().copied().collect::<Vec<_>>();
        for request in requests {
            if !matches!(self.queue.enqueue(request), RequestEnqueue::Rejected) {
                self.capacity_blocked.remove(&request.page_id);
            }
        }
    }

    fn pending_request_count(&self) -> usize {
        self.queue
            .len()
            .saturating_add(self.in_flight.len())
            .saturating_add(self.preprocessor.len())
            .saturating_add(self.capacity_blocked.len())
    }

    /// Enqueues without allowing queued, transport/preprocessing in-flight,
    /// and capacity-blocked state to exceed the single configured
    /// pending-request budget. Existing queued pages may still be promoted
    /// without increasing aggregate state.
    fn enqueue_pending_request(&mut self, request: PageRequest) -> RequestEnqueue {
        if self.pending_request_count() < self.pending_request_capacity
            || self.queue.contains(request.page_id)
        {
            self.queue.enqueue(request)
        } else {
            RequestEnqueue::Rejected
        }
    }

    fn update_frontier_pins(
        &mut self,
        view_id: LodRuntimeViewId,
        frontier: &LodFrontier<LodNodeId>,
    ) -> Result<bool, LodRuntimeError> {
        let selected = frontier
            .nodes
            .iter()
            .filter_map(|node| self.hierarchy.page(*node))
            .collect::<BTreeSet<_>>();
        self.views.entry(view_id).or_default().selected_frontier = selected;
        self.synchronize_view_pins(view_id)
    }

    fn synchronize_view_pins(
        &mut self,
        view_id: LodRuntimeViewId,
    ) -> Result<bool, LodRuntimeError> {
        let next = self
            .views
            .get(&view_id)
            .map(|state| state.selected_frontier.clone())
            .unwrap_or_default();
        let previous = self
            .views
            .get(&view_id)
            .map(|state| state.pinned_frontier.clone())
            .unwrap_or_default();
        let changed = next != previous;
        for &page in next.difference(&previous) {
            self.cache
                .pin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
        }
        for &page in previous.difference(&next) {
            self.cache
                .unpin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
        }
        self.views.entry(view_id).or_default().pinned_frontier = next;
        Ok(changed)
    }

    fn physical_ranges(
        &mut self,
        frontier: &LodFrontier<LodNodeId>,
    ) -> Result<Vec<LodPhysicalRange>, LodRuntimeError> {
        let mut ranges = Vec::with_capacity(frontier.nodes.len());
        for &node in &frontier.nodes {
            let representation = self
                .hierarchy
                .representation(node)
                .ok_or(LodRuntimeError::MissingNode(node))?;
            let resident = self.cache.get(representation.page).copied().ok_or(
                LodRuntimeError::SelectedPageNotResident(representation.page),
            )?;
            let decoded = self
                .decoded_pages
                .get(&representation.page)
                .ok_or(LodRuntimeError::SelectedPageNotDecoded(representation.page))?;
            validate_page_range(representation, decoded)?;
            let range_end = representation
                .offset
                .checked_add(representation.count)
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
            if range_end > self.atlas_layout.gaussians_per_slot {
                return Err(LodRuntimeError::PageRangeExceedsAtlasStride {
                    offset: representation.offset,
                    count: representation.count,
                    stride: self.atlas_layout.gaussians_per_slot,
                });
            }
            let physical_start = self
                .atlas_layout
                .physical_index(resident.slot, representation.offset)?;
            self.cache.touch(representation.page, self.epoch);
            ranges.push(LodPhysicalRange {
                node,
                page: representation.page,
                slot: resident.slot,
                physical_start,
                count: representation.count,
            });
        }
        Ok(ranges)
    }

    fn enqueue_missing(
        &mut self,
        frontier: &LodFrontier<LodNodeId>,
        view: LodView,
    ) -> Result<(), LodRuntimeError> {
        for &node in &frontier.requested_nodes {
            let page_id = self
                .hierarchy
                .page(node)
                .ok_or(LodRuntimeError::MissingNode(node))?;
            if self.cache.contains(page_id)
                || self.in_flight.contains_key(&page_id)
                || self.preprocessor.contains(page_id)
                || self.queue.contains(page_id)
                || self.terminal_failures.contains(&page_id)
                || self.capacity_blocked.contains_key(&page_id)
            {
                continue;
            }
            let descriptor = self
                .descriptors
                .get(&page_id)
                .ok_or(LodRuntimeError::MissingPageDescriptor(page_id))?;
            let manifest_node = self
                .hierarchy
                .node(node)
                .ok_or(LodRuntimeError::MissingNode(node))?;
            let metrics = self
                .hierarchy
                .metrics(node)
                .ok_or(LodRuntimeError::MissingNode(node))?;
            let distance = view.distance_to_center(metrics);
            let urgency = if distance <= 0.0 {
                u32::MAX
            } else {
                (1_000_000.0 / distance).clamp(0.0, u32::MAX as f32) as u32
            };
            let mut request = PageRequest::new(
                page_id,
                if manifest_node.parent.is_none() {
                    PageRequestPriority::fallback_critical(urgency)
                } else {
                    PageRequestPriority::visible(urgency)
                },
            );
            request.expected_bytes = descriptor
                .storage
                .as_ref()
                .map(|storage| storage.encoded_len);
            match self.enqueue_pending_request(request) {
                RequestEnqueue::Enqueued
                | RequestEnqueue::Promoted
                | RequestEnqueue::Duplicate
                | RequestEnqueue::Replaced(_) => {}
                RequestEnqueue::Rejected => {
                    // Bounded queue rejection is observable through requested_pages
                    // and a non-zero queue; the complete resident frontier remains valid.
                }
            }
        }
        Ok(())
    }

    fn start_requests(
        &mut self,
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
        failed_pages: &mut Vec<LodPageId>,
    ) -> Vec<LodPageId> {
        let concurrency = streaming_settings.max_concurrent_requests as usize;
        let available = concurrency.saturating_sub(self.in_flight.len());
        let frame_limit = lod_settings
            .budgets
            .max_requests_per_frame
            .saturating_sub(self.frame_request_starts) as usize;
        let attempt_limit = available.min(frame_limit);
        let scan_limit = self.queue.len();
        let mut started = Vec::new();
        let mut attempted = 0;
        let mut deferred = Vec::new();
        for _ in 0..scan_limit {
            if attempted >= attempt_limit {
                break;
            }
            let Some(request) = self.queue.pop() else {
                break;
            };
            if self.terminal_failures.contains(&request.page_id) {
                continue;
            }
            if self
                .preprocess_retry_deferred_frame
                .get(&request.page_id)
                .is_some_and(|deferred_frame| deferred_frame.0 == self.epoch)
            {
                deferred.push(request);
                continue;
            }
            self.preprocess_retry_deferred_frame
                .remove(&request.page_id);
            attempted += 1;
            let attempts = self.attempts.entry(request.page_id).or_default();
            *attempts = attempts.saturating_add(1);
            self.frame_request_starts = self.frame_request_starts.saturating_add(1);
            match self.transport.begin(request) {
                Ok(ticket) => {
                    self.in_flight
                        .insert(request.page_id, InFlight { ticket, request });
                    started.push(request.page_id);
                }
                Err(error) => {
                    self.transport_failures
                        .insert(request.page_id, T::classify_error(&error));
                    self.retry_or_fail(request, streaming_settings.retry_limit, failed_pages);
                }
            }
        }
        for request in deferred {
            let outcome = self.queue.enqueue(request);
            debug_assert!(
                !matches!(outcome, RequestEnqueue::Rejected),
                "a deferred preprocessing retry came from this bounded queue"
            );
        }
        started
    }
}

fn validate_page_range(
    range: LodPageRange,
    page: &PlanarGaussian3dPage,
) -> Result<(), LodRuntimeError> {
    let end = range.end().ok_or(LodRuntimeError::PhysicalIndexOverflow)? as usize;
    if end > page.gaussians.len() {
        Err(LodRuntimeError::PageRangeOutOfBounds {
            page: range.page,
            end: end as u64,
            count: page.gaussians.len() as u64,
        })
    } else {
        Ok(())
    }
}

fn validate_shared_page_node_ranges(
    page: &PlanarGaussian3dPage,
    ranges: &[SharedPageNodeRange],
    support_sigma: f32,
) -> Result<(), LodPagePreprocessError> {
    for entry in ranges {
        let end = entry
            .range
            .end()
            .ok_or(LodPagePreprocessError::PayloadOutsideNodeBounds {
                page: page.id,
                node: entry.node,
            })? as usize;
        let start = entry.range.offset as usize;
        let gaussians = page.gaussians.get(start..end).ok_or(
            LodPagePreprocessError::PayloadOutsideNodeBounds {
                page: page.id,
                node: entry.node,
            },
        )?;
        let mut actual_bounds: Option<LodBounds> = None;
        for gaussian in gaussians {
            let bounds = gaussian_support_bounds(gaussian, support_sigma)
                .map_err(|_| LodPagePreprocessError::InvalidSupportBounds(page.id))?;
            actual_bounds = Some(match actual_bounds {
                Some(current) => current.union(bounds),
                None => bounds,
            });
        }
        let actual_bounds =
            actual_bounds.ok_or(LodPagePreprocessError::PayloadOutsideNodeBounds {
                page: page.id,
                node: entry.node,
            })?;
        let epsilon = 1e-5 * entry.bounds.radius().max(actual_bounds.radius()).max(1.0);
        if !entry.bounds.contains_with_epsilon(&actual_bounds, epsilon) {
            return Err(LodPagePreprocessError::PayloadOutsideNodeBounds {
                page: page.id,
                node: entry.node,
            });
        }
    }
    Ok(())
}

fn page_codec_limits(
    descriptor: &LodPageDescriptor,
    max_encoded_page_bytes: u64,
) -> LodCodecLimits {
    LodCodecLimits {
        max_page_bytes: max_encoded_page_bytes,
        max_page_gaussians: descriptor.gaussian_count,
        ..Default::default()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LodRuntimeError {
    InvalidSettings(String),
    StructuralSettingsChanged(&'static str),
    InvalidManifest(ManifestHierarchyError),
    ManifestHasNoPages,
    ZeroAtlasStride,
    EncodedPageLimitTooSmall {
        limit: u64,
        minimum: u64,
    },
    PageEncodedBytesExceedLimit {
        page: LodPageId,
        actual: u64,
        limit: u64,
    },
    PageDecodedBytesExceedLimit {
        page: LodPageId,
        actual: u64,
        limit: u64,
    },
    PageGaussiansExceedLimit {
        page: LodPageId,
        actual: u64,
        limit: u64,
    },
    InvalidPageSupportBounds {
        page: LodPageId,
    },
    PagePayloadOutsideDescriptor(LodPageId),
    RequestCapacityOverflow,
    PreprocessAdmission(LodPagePreprocessAdmissionError),
    Queue(RequestQueueError),
    Cache(PageCacheError),
    Selection(LodSelectionError<LodNodeId>),
    MissingNode(LodNodeId),
    MissingPageDescriptor(LodPageId),
    MissingTerminalRequest(LodPageId),
    RetryQueueRejected(LodPageId),
    InvalidFrameToken {
        expected: LodRuntimeFrameId,
        actual: LodRuntimeFrameId,
    },
    FrameAlreadyFinished(LodRuntimeFrameId),
    SelectedPageNotResident(LodPageId),
    SelectedPageNotDecoded(LodPageId),
    PageRangeOutOfBounds {
        page: LodPageId,
        end: u64,
        count: u64,
    },
    PageRangeExceedsAtlasStride {
        offset: u32,
        count: u32,
        stride: u32,
    },
    PhysicalIndexOverflow,
    AtlasAddressSpaceOverflow {
        slots: u32,
        stride: u32,
    },
    CandidateCountMismatch {
        frontier: u64,
        physical: u64,
    },
    NoResidentFrontier,
    OverlappingPhysicalRanges {
        previous_end: u32,
        next_start: u32,
    },
    CandidateExpansionLimit {
        count: u64,
        limit: u32,
    },
}

impl fmt::Display for LodRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LodRuntimeError {}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use std::sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    };

    use super::*;
    use crate::{
        gaussian::formats::{
            planar_3d::{Gaussian3d, PlanarGaussian3d},
            planar_3d_chunked::{
                LOD_PAGE_SCHEMA_VERSION, LodBounds, LodIndexRange, LodPageEncoding, LodPageKind,
                LodPageStorage, LodSourceRange,
            },
            planar_3d_lod::{
                GaussianLodBuildMetadata, GaussianLodBuildSettings, GaussianLodManifestHeader,
                GaussianLodNode, GaussianLodQualityMetadata, LOD_CURRENT_REQUIRED_FEATURES,
                LOD_MANIFEST_MAGIC, LOD_MANIFEST_VERSION, LodError, LodMortonRange,
                LodQualityInterval, LodReducerKind, MOMENT_MERGE_VERSION, build_planar_3d_lod,
                lod_config_fingerprint,
            },
        },
        io::lod::{LodCodecError, decode_page, encode_page},
        material::spherical_harmonics::SphericalHarmonicCoefficients,
        stream::transport::{MemoryPageTransport, MemoryTransportError, PagePayload},
        testing::{LodTestScene, VirtualCityScene},
    };

    #[cfg(not(target_arch = "wasm32"))]
    use crate::stream::preprocess::LodPagePreprocessBackend;

    const VIRTUAL_TREE_DEPTH: u16 = 3;
    const VIRTUAL_BRANCHING_FACTOR: u32 = 32;
    // The virtual fixture uses a wide, non-progressive topology like the
    // supported external CPU package builder.
    const VIRTUAL_BUILDER_ABI_VERSION: u32 = 5;
    const VIRTUAL_LEVEL_STARTS: [u32; 4] = [0, 1, 33, 1_057];
    const VIRTUAL_LEVEL_COUNTS: [u32; 4] = [1, 32, 1_024, 32_768];
    const VIRTUAL_NODE_COUNT: u32 = 33_825;

    struct VirtualRuntimeFixture {
        manifest: GaussianLodManifest,
        transport: MemoryPageTransport,
        lod_settings: GaussianLodSettings,
        streaming_settings: GaussianStreamingSettings,
        encoded_root_bytes: usize,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum ToggleTransportError {
        ForcedBeginFailure,
        Memory(MemoryTransportError),
    }

    struct ToggleMemoryTransport {
        inner: MemoryPageTransport,
        fail_begin: bool,
        begin_count: u32,
    }

    struct CancelCountingTransport {
        inner: MemoryPageTransport,
        cancellations: Arc<AtomicU32>,
    }

    impl LodPageTransport for CancelCountingTransport {
        type Ticket = u64;
        type Error = MemoryTransportError;

        fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
            self.inner.begin(request)
        }

        fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
            self.inner.poll(ticket)
        }

        fn cancel(&mut self, ticket: &Self::Ticket) {
            self.cancellations.fetch_add(1, Ordering::Relaxed);
            self.inner.cancel(ticket);
        }
    }

    impl ToggleMemoryTransport {
        fn failing(inner: MemoryPageTransport) -> Self {
            Self {
                inner,
                fail_begin: true,
                begin_count: 0,
            }
        }
    }

    impl LodPageTransport for ToggleMemoryTransport {
        type Ticket = u64;
        type Error = ToggleTransportError;

        fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
            self.begin_count = self.begin_count.saturating_add(1);
            if self.fail_begin {
                Err(ToggleTransportError::ForcedBeginFailure)
            } else {
                self.inner
                    .begin(request)
                    .map_err(ToggleTransportError::Memory)
            }
        }

        fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
            match self.inner.poll(ticket) {
                PagePoll::Pending => PagePoll::Pending,
                PagePoll::Ready(payload) => PagePoll::Ready(payload),
                PagePoll::Failed(error) => PagePoll::Failed(ToggleTransportError::Memory(error)),
            }
        }

        fn cancel(&mut self, ticket: &Self::Ticket) {
            self.inner.cancel(ticket);
        }

        fn classify_error(error: &Self::Error) -> LodPageTransportFailure {
            match error {
                ToggleTransportError::ForcedBeginFailure => {
                    LodPageTransportFailure::transport("forced begin failure")
                }
                ToggleTransportError::Memory(_) => {
                    LodPageTransportFailure::transport("memory transport failure")
                }
            }
        }
    }

    fn fixture() -> (
        GaussianLodManifest,
        MemoryPageTransport,
        GaussianLodSettings,
        GaussianStreamingSettings,
    ) {
        let scene = LodTestScene::screen_space_ladder();
        let mut lod = build_planar_3d_lod(
            &scene.cloud(),
            GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 16,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        let mut transport = MemoryPageTransport::default();
        for page in &lod.pages {
            let encoded = encode_page(page).unwrap();
            let descriptor = lod
                .manifest
                .pages
                .iter_mut()
                .find(|descriptor| descriptor.id == page.id)
                .unwrap();
            descriptor.storage = Some(
                crate::gaussian::formats::planar_3d_chunked::LodPageStorage {
                    uri: format!("memory://{}", page.id.0),
                    byte_range: None,
                    encoded_len: encoded.len() as u64,
                },
            );
            transport.insert(page.id, encoded);
        }
        lod.manifest.validate().unwrap();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        settings.budgets.max_active_gaussians = 1_000_000;
        settings.budgets.max_resident_gaussians = 1_000_000;
        settings.budgets.max_resident_bytes = 256 * 1024 * 1024;
        settings.budgets.max_resident_pages = 1024;
        settings.budgets.max_requests_per_frame = 1024;
        let streaming = GaussianStreamingSettings {
            max_concurrent_requests:
                crate::gaussian::lod_settings::MAX_STREAMING_CONCURRENT_REQUESTS,
            ..Default::default()
        };
        (lod.manifest, transport, settings, streaming)
    }

    #[test]
    fn dropping_runtime_cancels_all_in_flight_transport_work() {
        let (manifest, transport, settings, streaming) = fixture();
        let cancellations = Arc::new(AtomicU32::new(0));
        let mut runtime = LodStreamingRuntime::new(
            manifest,
            CancelCountingTransport {
                inner: transport,
                cancellations: cancellations.clone(),
            },
            &settings,
            &streaming,
        )
        .unwrap();
        let frame = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(frame.in_flight_requests > 0);
        let in_flight = frame.in_flight_requests;
        drop(runtime);
        assert_eq!(cancellations.load(Ordering::Relaxed), in_flight);
    }

    #[test]
    fn finishing_frame_cancels_only_work_no_active_view_demands() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_pending_requests = 8;
        let pages = manifest
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .take(5)
            .collect::<Vec<_>>();
        assert_eq!(pages.len(), 5);
        let cancellations = Arc::new(AtomicU32::new(0));
        let mut runtime = LodStreamingRuntime::new(
            manifest,
            CancelCountingTransport {
                inner: transport,
                cancellations: cancellations.clone(),
            },
            &settings,
            &streaming,
        )
        .unwrap();
        let frame = runtime.begin_frame();
        let request = |page| PageRequest::new(page, PageRequestPriority::visible(1));
        runtime
            .views
            .entry(LodRuntimeViewId(7))
            .or_default()
            .requested_pages = BTreeSet::from([pages[0]]);
        runtime
            .views
            .get_mut(&LodRuntimeViewId(7))
            .unwrap()
            .requested_pages_frame = frame;
        assert_eq!(
            runtime.queue.enqueue(request(pages[0])),
            RequestEnqueue::Enqueued
        );
        assert_eq!(
            runtime.queue.enqueue(request(pages[1])),
            RequestEnqueue::Enqueued
        );
        runtime.in_flight.insert(
            pages[2],
            InFlight {
                ticket: 123,
                request: request(pages[2]),
            },
        );
        runtime.capacity_blocked.insert(pages[3], request(pages[3]));
        let preprocess_descriptor = runtime.descriptors[&pages[4]].clone();
        runtime
            .preprocessor
            .submit(LodPagePreprocessInput {
                request: request(pages[4]),
                payload: PagePayload::new(pages[4], Vec::new()),
                limits: page_codec_limits(
                    &preprocess_descriptor,
                    streaming.effective_max_encoded_page_bytes(),
                ),
                descriptor: preprocess_descriptor,
                max_encoded_page_bytes: streaming.effective_max_encoded_page_bytes(),
                support_sigma: runtime.hierarchy.manifest().build.settings.support_sigma,
            })
            .unwrap();
        for page in &pages {
            runtime.attempts.insert(*page, 1);
        }

        runtime.finish_frame(frame).unwrap();
        assert_eq!(cancellations.load(Ordering::Relaxed), 1);
        assert!(runtime.queue.contains(pages[0]));
        assert!(!runtime.queue.contains(pages[1]));
        assert!(!runtime.in_flight.contains_key(&pages[2]));
        assert!(!runtime.capacity_blocked.contains_key(&pages[3]));
        assert!(!runtime.preprocessor.contains(pages[4]));
        assert_eq!(runtime.attempts.get(&pages[0]), Some(&1));
        for page in &pages[1..] {
            assert!(!runtime.attempts.contains_key(page));
        }
        runtime.finish_frame(frame).unwrap();
        assert_eq!(cancellations.load(Ordering::Relaxed), 1);
    }

    fn memory_preprocess_input(
        manifest: &GaussianLodManifest,
        transport: &mut MemoryPageTransport,
        page: LodPageId,
        max_encoded_page_bytes: u64,
    ) -> LodPagePreprocessInput {
        let descriptor = manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == page)
            .unwrap()
            .clone();
        let mut request = PageRequest::new(page, PageRequestPriority::visible(1));
        request.expected_bytes = descriptor
            .storage
            .as_ref()
            .map(|storage| storage.encoded_len);
        let ticket = transport.begin(request).unwrap();
        let PagePoll::Ready(payload) = transport.poll(&ticket) else {
            panic!("memory transport must return an inserted page")
        };
        LodPagePreprocessInput {
            request,
            payload,
            limits: page_codec_limits(&descriptor, max_encoded_page_bytes),
            descriptor,
            max_encoded_page_bytes,
            support_sigma: manifest.build.settings.support_sigma,
        }
    }

    #[test]
    fn preprocessing_admission_success_and_capacity_are_deterministic() {
        assert!(matches!(
            LodPagePreprocessor::with_byte_capacity(0, 1),
            Err(LodPagePreprocessAdmissionError::ZeroCapacity)
        ));
        let (manifest, mut transport, _, streaming) = fixture();
        let pages = manifest
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .take(2)
            .collect::<Vec<_>>();
        let mut preprocessor = LodPagePreprocessor::new_cooperative_for_tests(1).unwrap();
        preprocessor
            .submit(memory_preprocess_input(
                &manifest,
                &mut transport,
                pages[0],
                streaming.effective_max_encoded_page_bytes(),
            ))
            .unwrap();
        assert_eq!(preprocessor.stats().capacity, 1);
        assert_eq!(preprocessor.stats().waiting, 1);
        let pending_bytes = preprocessor.stats().pending_bytes;
        assert!(pending_bytes > 0);
        assert!(matches!(
            preprocessor.submit(memory_preprocess_input(
                &manifest,
                &mut transport,
                pages[0],
                streaming.effective_max_encoded_page_bytes(),
            )),
            Err(LodPagePreprocessAdmissionError::DuplicatePage(duplicate)) if duplicate == pages[0]
        ));
        assert!(matches!(
            preprocessor.submit(memory_preprocess_input(
                &manifest,
                &mut transport,
                pages[1],
                streaming.effective_max_encoded_page_bytes(),
            )),
            Err(LodPagePreprocessAdmissionError::CapacityExhausted { capacity: 1 })
        ));

        let full_page_budget = NonZeroU32::new(u32::MAX).unwrap();
        preprocessor.advance(1, full_page_budget);
        assert_eq!(preprocessor.stats().ready, 0);
        assert_eq!(preprocessor.stats().submitted, 1);
        preprocessor.advance(2, full_page_budget);
        assert_eq!(preprocessor.stats().ready, 1);
        assert_eq!(preprocessor.stats().pending_bytes, pending_bytes);
        let output = preprocessor.take_ready(pages[0]).unwrap();
        assert_eq!(output.request.page_id, pages[0]);
        assert_eq!(output.result.unwrap().id, pages[0]);
        assert_eq!(preprocessor.len(), 0);
        assert_eq!(preprocessor.stats().pending_bytes, 0);
    }

    #[test]
    fn preprocessing_byte_admission_and_cancellation_are_exact() {
        let (manifest, mut transport, _, streaming) = fixture();
        let pages = manifest
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .take(2)
            .collect::<Vec<_>>();
        let first = memory_preprocess_input(
            &manifest,
            &mut transport,
            pages[0],
            streaming.effective_max_encoded_page_bytes(),
        );
        let second = memory_preprocess_input(
            &manifest,
            &mut transport,
            pages[1],
            streaming.effective_max_encoded_page_bytes(),
        );
        let first_bytes = first.pending_bytes().unwrap();
        let second_bytes = second.pending_bytes().unwrap();
        assert!(matches!(
            LodPagePreprocessor::new_cooperative_with_byte_capacity_for_tests(2, 0),
            Err(LodPagePreprocessAdmissionError::ZeroByteCapacity)
        ));
        let mut preprocessor =
            LodPagePreprocessor::new_cooperative_with_byte_capacity_for_tests(2, first_bytes)
                .unwrap();
        preprocessor.submit(first).unwrap();
        assert_eq!(preprocessor.stats().byte_capacity, first_bytes);
        assert_eq!(preprocessor.stats().pending_bytes, first_bytes);
        assert_eq!(
            preprocessor.submit(second),
            Err(
                LodPagePreprocessAdmissionError::PendingByteCapacityExceeded {
                    requested: second_bytes,
                    pending: first_bytes,
                    capacity: first_bytes,
                }
            )
        );
        assert!(preprocessor.cancel(pages[0]));
        assert_eq!(preprocessor.stats().pending_bytes, 0);
        assert_eq!(preprocessor.len(), 0);
    }

    #[test]
    fn cooperative_preprocessing_runs_at_most_one_page_slice_per_application_frame() {
        let (manifest, mut transport, _, streaming) = fixture();
        let pages = manifest
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .take(2)
            .collect::<Vec<_>>();
        let first = memory_preprocess_input(
            &manifest,
            &mut transport,
            pages[0],
            streaming.effective_max_encoded_page_bytes(),
        );
        let second = memory_preprocess_input(
            &manifest,
            &mut transport,
            pages[1],
            streaming.effective_max_encoded_page_bytes(),
        );
        let byte_capacity = first
            .pending_bytes()
            .unwrap()
            .checked_add(second.pending_bytes().unwrap())
            .unwrap();
        let mut preprocessor =
            LodPagePreprocessor::new_cooperative_with_byte_capacity_for_tests(2, byte_capacity)
                .unwrap();
        preprocessor.submit(first).unwrap();
        preprocessor.submit(second).unwrap();
        let one_record_budget = NonZeroU32::MIN;
        preprocessor.advance(41, one_record_budget);
        let first_slice = preprocessor.stats();
        preprocessor.advance(41, one_record_budget);
        assert_eq!(preprocessor.stats(), first_slice);
        assert_eq!(preprocessor.stats().ready, 0);
        assert_eq!(preprocessor.stats().waiting, 1);
        assert_eq!(preprocessor.stats().submitted, 1);

        let mut frame = 42;
        while preprocessor.stats().ready == 0 {
            let before = preprocessor.stats();
            preprocessor.advance(frame, one_record_budget);
            let after = preprocessor.stats();
            if before.submitted == 1 && after.submitted == 1 {
                assert!(
                    after.cooperative_decoded_gaussians
                        <= before.cooperative_decoded_gaussians.saturating_add(1)
                );
            }
            frame += 1;
            assert!(frame < 1_000, "bounded first page did not complete");
        }
        assert_eq!(preprocessor.stats().ready, 1);
        assert_eq!(preprocessor.stats().waiting, 1);
        assert_eq!(preprocessor.stats().submitted, 0);

        let completed_first_frame = frame - 1;
        let after_first_completion = preprocessor.stats();
        preprocessor.advance(completed_first_frame, one_record_budget);
        assert_eq!(preprocessor.stats(), after_first_completion);
        preprocessor.advance(frame, one_record_budget);
        assert_eq!(preprocessor.stats().ready, 1);
        assert_eq!(preprocessor.stats().waiting, 0);
        assert_eq!(preprocessor.stats().submitted, 1);

        frame += 1;
        while preprocessor.stats().ready < 2 {
            preprocessor.advance(frame, one_record_budget);
            frame += 1;
            assert!(frame < 2_000, "bounded second page did not complete");
        }
        assert_eq!(preprocessor.stats().ready, 2);
        assert_eq!(preprocessor.stats().waiting, 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_preprocessing_worker_pool_executes_the_production_backend() {
        let (manifest, mut transport, _, streaming) = fixture();
        let page = manifest.pages[0].id;
        let input = memory_preprocess_input(
            &manifest,
            &mut transport,
            page,
            streaming.effective_max_encoded_page_bytes(),
        );
        let pending_bytes = input.pending_bytes().unwrap();
        let mut preprocessor = LodPagePreprocessor::new_native_for_tests(1, pending_bytes).unwrap();
        assert_eq!(
            preprocessor.stats().backend,
            LodPagePreprocessBackend::NativeWorkerPool
        );
        preprocessor.submit(input).unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while preprocessor.stats().ready == 0 && std::time::Instant::now() < deadline {
            preprocessor.advance(1, NonZeroU32::MIN);
            std::thread::yield_now();
        }

        let output = preprocessor
            .take_ready(page)
            .expect("native preprocessing worker should complete within the test deadline");
        assert_eq!(output.result.unwrap().id, page);
        assert_eq!(preprocessor.stats().pending_bytes, 0);
    }

    #[test]
    fn malformed_preprocessing_payload_is_typed_and_bounded() {
        let (manifest, mut transport, _, streaming) = fixture();
        let page = manifest.pages[0].id;
        let mut input = memory_preprocess_input(
            &manifest,
            &mut transport,
            page,
            streaming.effective_max_encoded_page_bytes(),
        );
        input.payload.bytes[0] ^= 0xff;
        let mut preprocessor = LodPagePreprocessor::new_cooperative_for_tests(1).unwrap();
        preprocessor.submit(input).unwrap();
        preprocessor.advance(1, NonZeroU32::new(u32::MAX).unwrap());
        assert_eq!(
            preprocessor.take_ready(page).unwrap().result,
            Err(LodPagePreprocessError::PayloadChecksumMismatch)
        );
        assert_eq!(preprocessor.len(), 0);
    }

    #[test]
    fn cooperative_codec_failure_precedes_an_earlier_support_failure() {
        let (manifest, mut transport, _, streaming) = fixture();
        let page_id = manifest.pages[0].id;
        let mut input = memory_preprocess_input(
            &manifest,
            &mut transport,
            page_id,
            streaming.effective_max_encoded_page_bytes(),
        );
        let mut page = decode_page(&input.payload.bytes, input.limits).unwrap();
        page.gaussians[0].scale_opacity.scale.fill(f32::MAX);
        let mut encoded = encode_page(&page).unwrap();
        encoded[36] ^= 1;
        input.payload = PagePayload::new(page_id, encoded);

        let mut preprocessor = LodPagePreprocessor::new_cooperative_for_tests(1).unwrap();
        preprocessor.submit(input).unwrap();
        let mut frame = 1;
        while preprocessor.stats().ready == 0 {
            preprocessor.advance(frame, NonZeroU32::MIN);
            frame += 1;
            assert!(frame < 1_000, "adversarial page did not terminate");
        }
        assert!(matches!(
            preprocessor.take_ready(page_id).unwrap().result,
            Err(LodPagePreprocessError::Codec(
                LodCodecError::ChecksumMismatch { .. }
            ))
        ));
    }

    #[test]
    fn cancelled_preprocessing_never_publishes_a_stale_result() {
        let (manifest, mut transport, _, streaming) = fixture();
        let page = manifest.pages[0].id;
        let input = memory_preprocess_input(
            &manifest,
            &mut transport,
            page,
            streaming.effective_max_encoded_page_bytes(),
        );
        let mut preprocessor = LodPagePreprocessor::new_cooperative_for_tests(1).unwrap();
        let pending_bytes = input.pending_bytes().unwrap();
        preprocessor.submit(input).unwrap();
        let budget = NonZeroU32::MIN;
        preprocessor.advance(1, budget);
        let active = preprocessor.stats();
        assert_eq!(active.waiting, 0);
        assert_eq!(active.submitted, 1);
        assert_eq!(active.pending_bytes, pending_bytes);
        assert_eq!(active.cooperative_budget_gaussians_per_frame, 1);
        assert!(preprocessor.contains(page));
        assert!(preprocessor.cancel(page));
        preprocessor.advance(2, budget);
        assert!(preprocessor.take_ready(page).is_none());
        assert_eq!(preprocessor.stats().cancellations, 1);
        assert_eq!(preprocessor.stats().pending_bytes, 0);
        assert_eq!(preprocessor.len(), 0);
    }

    #[test]
    fn runtime_retains_typed_preprocess_failure_through_terminal_retry_state() {
        let (manifest, mut transport, mut settings, mut streaming) = fixture();
        settings.quality = 0.0;
        settings.budgets.max_requests_per_frame = 1;
        streaming.retry_limit = 0;
        let page = root_page(&manifest);
        let encoded_len = manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == page)
            .and_then(|descriptor| descriptor.storage.as_ref())
            .unwrap()
            .encoded_len as usize;
        transport.insert(page, vec![0; encoded_len]);
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        runtime.update(view(), &settings, &streaming).unwrap();
        let failed = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(failed.preprocess_failed_pages(), &[page]);
        assert_eq!(failed.failed_pages(), &[page]);
        assert!(matches!(
            runtime.page_preprocess_error(page),
            Some(LodPagePreprocessError::Codec(_))
        ));
        assert!(runtime.is_terminal_failure(page));
    }

    #[test]
    fn updates_are_rejected_after_frame_demand_is_reconciled() {
        let (manifest, transport, settings, streaming) = fixture();
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let frame = runtime.begin_frame();
        runtime.finish_frame(frame).unwrap();
        assert_eq!(
            runtime
                .update_view_in_frame(frame, LodRuntimeViewId(5), view(), &settings, &streaming,),
            Err(LodRuntimeError::FrameAlreadyFinished(frame))
        );
    }

    #[test]
    fn shared_physical_page_validates_each_logical_node_slice() {
        let gaussian = |x| Gaussian3d {
            position_visibility: [x, 0.0, 0.0, 1.0].into(),
            spherical_harmonic: SphericalHarmonicCoefficients::default(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.1, 0.1, 0.1, 1.0].into(),
        };
        let page_id = LodPageId(1);
        let page = PlanarGaussian3dPage::new(page_id, vec![gaussian(-1.0), gaussian(1.0)]);
        let left = gaussian_support_bounds(&page.gaussians[0], 3.0).unwrap();
        let right = gaussian_support_bounds(&page.gaussians[1], 3.0).unwrap();
        let ranges = [
            SharedPageNodeRange {
                node: LodNodeId(1),
                range: LodPageRange {
                    page: page_id,
                    offset: 0,
                    count: 1,
                },
                bounds: left,
            },
            SharedPageNodeRange {
                node: LodNodeId(2),
                range: LodPageRange {
                    page: page_id,
                    offset: 1,
                    count: 1,
                },
                bounds: right,
            },
        ];
        assert_eq!(
            validate_shared_page_node_ranges(&page, &ranges, 3.0),
            Ok(())
        );

        let swapped = [
            SharedPageNodeRange {
                bounds: right,
                ..ranges[0]
            },
            SharedPageNodeRange {
                bounds: left,
                ..ranges[1]
            },
        ];
        assert_eq!(
            validate_shared_page_node_ranges(&page, &swapped, 3.0),
            Err(LodPagePreprocessError::PayloadOutsideNodeBounds {
                page: page_id,
                node: LodNodeId(1),
            })
        );
    }

    fn two_root_fixture() -> (
        GaussianLodManifest,
        MemoryPageTransport,
        GaussianLodSettings,
        GaussianStreamingSettings,
    ) {
        let gaussian = |x| Gaussian3d {
            position_visibility: [x, 0.0, 0.0, 1.0].into(),
            spherical_harmonic: SphericalHarmonicCoefficients::default(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.1, 0.1, 0.1, 1.0].into(),
        };
        let cloud: PlanarGaussian3d = vec![gaussian(-1.0), gaussian(1.0)].into();
        let mut lod = build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                branching_factor: 2,
                leaf_capacity: 1,
                support_sigma: 3.0,
            },
        )
        .unwrap();

        let original_root = lod.manifest.roots[0];
        let root_page = lod
            .manifest
            .nodes
            .iter()
            .find(|node| node.id == original_root)
            .unwrap()
            .representation
            .page;
        lod.manifest.nodes.retain(|node| node.id != original_root);
        lod.manifest.pages.retain(|page| page.id != root_page);
        lod.pages.retain(|page| page.id != root_page);
        for node in &mut lod.manifest.nodes {
            node.parent = None;
            node.depth = 0;
            node.quality.min = 0.0;
        }
        lod.manifest.roots = lod.manifest.nodes.iter().map(|node| node.id).collect();
        lod.manifest.header.node_count = lod.manifest.nodes.len() as u32;
        lod.manifest.header.page_count = lod.manifest.pages.len() as u32;
        lod.manifest.header.stored_gaussian_count = lod
            .manifest
            .pages
            .iter()
            .map(|page| u64::from(page.gaussian_count))
            .sum();
        lod.manifest.quality = GaussianLodQualityMetadata {
            max_depth: 0,
            coarsest_gaussian_count: 2,
            finest_gaussian_count: 2,
            max_error: lod
                .manifest
                .nodes
                .iter()
                .fold(LodError::ZERO, |error, node| error.max(node.error)),
        };

        let mut transport = MemoryPageTransport::default();
        for page in &lod.pages {
            let encoded = encode_page(page).unwrap();
            let descriptor = lod
                .manifest
                .pages
                .iter_mut()
                .find(|descriptor| descriptor.id == page.id)
                .unwrap();
            descriptor.storage = Some(LodPageStorage {
                uri: format!("memory://two-root-{}", page.id.0),
                byte_range: None,
                encoded_len: encoded.len() as u64,
            });
            transport.insert(page.id, encoded);
        }
        lod.validate().unwrap();

        let mut settings = GaussianLodSettings::default();
        settings.quality = 0.0;
        settings.budgets.max_requests_per_frame = 1;
        let streaming = GaussianStreamingSettings {
            max_concurrent_requests: 1,
            ..Default::default()
        };
        (lod.manifest, transport, settings, streaming)
    }

    fn view() -> LodView {
        LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, 8.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        )
    }

    fn root_page(manifest: &GaussianLodManifest) -> LodPageId {
        let root = manifest.roots[0];
        manifest
            .nodes
            .iter()
            .find(|node| node.id == root)
            .expect("fixture root must exist")
            .representation
            .page
    }

    /// Builds only breadth-first topology and page descriptors for the lazy
    /// virtual city. The 32,768 leaf page counts remain logical: this helper
    /// materializes and encodes exactly one Gaussian, the root representative.
    fn virtual_runtime_fixture() -> VirtualRuntimeFixture {
        let city = VirtualCityScene::default();
        assert_eq!(city.page_count, VIRTUAL_LEVEL_COUNTS[3]);

        let root_id = LodNodeId(1);
        let root_page_id = LodPageId(1);
        let root_page = PlanarGaussian3dPage::new(
            root_page_id,
            vec![Gaussian3d {
                position_visibility: [0.0, 0.0, 0.0, 1.0].into(),
                spherical_harmonic: SphericalHarmonicCoefficients::default(),
                rotation: [1.0, 0.0, 0.0, 0.0].into(),
                scale_opacity: [0.25, 0.25, 0.25, 1.0].into(),
            }],
        );
        let encoded_root = encode_page(&root_page).unwrap();
        let bounds = LodBounds::new([-1.0; 3], [1.0; 3]).unwrap();
        let build_settings = GaussianLodBuildSettings {
            branching_factor: VIRTUAL_BRANCHING_FACTOR as u8,
            leaf_capacity: city.gaussians_per_page,
            support_sigma: 3.0,
        };
        let mut nodes = Vec::with_capacity(VIRTUAL_NODE_COUNT as usize);
        let mut pages = Vec::with_capacity(VIRTUAL_NODE_COUNT as usize);
        let mut stored_gaussian_count = 0_u64;

        for depth in 0..=VIRTUAL_TREE_DEPTH {
            let depth_index = usize::from(depth);
            let level_start = VIRTUAL_LEVEL_STARTS[depth_index];
            let level_count = VIRTUAL_LEVEL_COUNTS[depth_index];
            let descendant_leaf_count =
                VIRTUAL_BRANCHING_FACTOR.pow(u32::from(VIRTUAL_TREE_DEPTH - depth));
            for ordinal in 0..level_count {
                let flat_index = level_start + ordinal;
                let node_id = LodNodeId(u64::from(flat_index) + 1);
                let page_id = LodPageId(node_id.0);
                let first_leaf = u64::from(ordinal) * u64::from(descendant_leaf_count);
                let source = LodSourceRange {
                    start: first_leaf * u64::from(city.gaussians_per_page),
                    count: u64::from(descendant_leaf_count) * u64::from(city.gaussians_per_page),
                };
                let is_leaf = depth == VIRTUAL_TREE_DEPTH;
                let gaussian_count = if is_leaf { city.gaussians_per_page } else { 1 };
                let geometric_error = f32::from(VIRTUAL_TREE_DEPTH - depth);
                let error = LodError {
                    geometric: geometric_error,
                    appearance: 0.0,
                    opacity: 0.0,
                    combined: geometric_error,
                };
                let quality = LodQualityInterval {
                    min: f32::from(depth) / f32::from(VIRTUAL_TREE_DEPTH),
                    max: if is_leaf {
                        1.0
                    } else {
                        f32::from(depth + 1) / f32::from(VIRTUAL_TREE_DEPTH)
                    },
                };
                nodes.push(GaussianLodNode {
                    id: node_id,
                    parent: (depth > 0).then(|| {
                        let parent_index = VIRTUAL_LEVEL_STARTS[depth_index - 1]
                            + ordinal / VIRTUAL_BRANCHING_FACTOR;
                        LodNodeId(u64::from(parent_index) + 1)
                    }),
                    depth,
                    bounds,
                    children: if is_leaf {
                        LodIndexRange::empty()
                    } else {
                        LodIndexRange {
                            start: VIRTUAL_LEVEL_STARTS[depth_index + 1]
                                + ordinal * VIRTUAL_BRANCHING_FACTOR,
                            count: VIRTUAL_BRANCHING_FACTOR,
                        }
                    },
                    source,
                    morton: LodMortonRange {
                        min: first_leaf,
                        max: first_leaf + u64::from(descendant_leaf_count) - 1,
                    },
                    representation: LodPageRange {
                        page: page_id,
                        offset: 0,
                        count: gaussian_count,
                    },
                    error,
                    quality,
                    high_fidelity_certificate: if is_leaf { 1.0 } else { 0.0 },
                });
                pages.push(LodPageDescriptor {
                    id: page_id,
                    kind: if is_leaf {
                        LodPageKind::SourceLeaves
                    } else {
                        LodPageKind::Representatives
                    },
                    encoding: LodPageEncoding::F32Planar,
                    gaussian_count,
                    decoded_len: u64::from(gaussian_count)
                        * std::mem::size_of::<Gaussian3d>() as u64,
                    content_hash: if page_id == root_page_id {
                        root_page.content_hash()
                    } else {
                        0x9e37_79b9_7f4a_7c15_u64 ^ page_id.0
                    },
                    bounds,
                    storage: (page_id == root_page_id).then(|| LodPageStorage {
                        uri: "memory://virtual-city-root".to_owned(),
                        byte_range: None,
                        encoded_len: encoded_root.len() as u64,
                    }),
                });
                stored_gaussian_count += u64::from(gaussian_count);
            }
        }
        assert_eq!(nodes.len(), VIRTUAL_NODE_COUNT as usize);
        assert_eq!(pages.len(), VIRTUAL_NODE_COUNT as usize);

        let source_gaussian_count = city.source_gaussian_count();
        let root_error = nodes[0].error;
        let manifest = GaussianLodManifest {
            header: GaussianLodManifestHeader {
                magic: LOD_MANIFEST_MAGIC,
                manifest_version: LOD_MANIFEST_VERSION,
                page_schema_version: LOD_PAGE_SCHEMA_VERSION,
                required_features: LOD_CURRENT_REQUIRED_FEATURES,
                source_gaussian_count,
                stored_gaussian_count,
                node_count: VIRTUAL_NODE_COUNT,
                page_count: VIRTUAL_NODE_COUNT,
            },
            scene_bounds: Some(bounds),
            roots: vec![root_id],
            nodes,
            pages,
            build: GaussianLodBuildMetadata {
                settings: build_settings,
                reducer: LodReducerKind::MomentMerge,
                builder_abi_version: VIRTUAL_BUILDER_ABI_VERSION,
                reducer_version: MOMENT_MERGE_VERSION,
                source_fingerprint: 0x4f9a_2be3_a561_903d,
                config_fingerprint: lod_config_fingerprint(build_settings, None),
            },
            quality: GaussianLodQualityMetadata {
                max_depth: VIRTUAL_TREE_DEPTH,
                coarsest_gaussian_count: 1,
                finest_gaussian_count: source_gaussian_count,
                max_error: root_error,
            },
        };
        manifest.validate().unwrap();

        let encoded_root_bytes = encoded_root.len();
        let mut transport = MemoryPageTransport::default();
        assert!(transport.insert(root_page_id, encoded_root).is_none());

        let mut lod_settings = GaussianLodSettings::default();
        lod_settings.quality = 0.0;
        lod_settings.budgets.max_active_gaussians = 2;
        lod_settings.budgets.max_resident_gaussians = u64::from(city.gaussians_per_page);
        let max_page_decoded_bytes =
            u64::from(city.gaussians_per_page) * std::mem::size_of::<Gaussian3d>() as u64;
        lod_settings.budgets.max_resident_bytes = max_page_decoded_bytes;
        lod_settings.budgets.max_resident_pages = 2;
        lod_settings.budgets.max_pending_requests = 2;
        lod_settings.budgets.max_requests_per_frame = 1;
        lod_settings.budgets.max_upload_bytes_per_frame = max_page_decoded_bytes;
        lod_settings.budgets.max_traversal_nodes_per_view = 4;
        let streaming_settings = GaussianStreamingSettings {
            max_concurrent_requests: 1,
            max_compressed_cache_bytes: 4 * 1024,
            ..Default::default()
        };

        VirtualRuntimeFixture {
            manifest,
            transport,
            lod_settings,
            streaming_settings,
            encoded_root_bytes,
        }
    }

    #[test]
    fn streams_from_roots_to_exact_frontier_without_holes() {
        let (manifest, transport, settings, streaming) = fixture();
        let source_count = manifest.header.source_gaussian_count;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();

        let first = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(first.candidate_count(), 0);
        assert!(!first.started_pages.is_empty());

        let mut final_frame = None;
        for _ in 0..64 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert_eq!(
                frame.candidate_count(),
                frame.frontier.status.active_gaussians
            );
            assert!(frame.failed_pages.is_empty());
            if frame.frontier.requested_nodes.is_empty()
                && frame.in_flight_requests == 0
                && frame.queued_requests == 0
            {
                final_frame = Some(frame);
                break;
            }
        }
        let final_frame = final_frame.expect("fixture should become fully resident");
        assert_eq!(final_frame.candidate_count(), source_count);
    }

    #[test]
    fn terminal_failure_does_not_restart_per_frame_and_explicit_retry_succeeds() {
        let (manifest, transport, mut settings, mut streaming) = fixture();
        settings.quality = 0.0;
        settings.budgets.max_requests_per_frame = 1;
        streaming.retry_limit = 1;
        let page = root_page(&manifest);
        let mut runtime = LodStreamingRuntime::new(
            manifest,
            ToggleMemoryTransport::failing(transport),
            &settings,
            &streaming,
        )
        .unwrap();

        // retry_limit counts retries after the initial start: one retry means
        // exactly two failed transport starts before the page is terminal.
        let first = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(first.failed_pages.is_empty());
        assert_eq!(first.queued_requests, 1);
        assert_eq!(runtime.page_attempts(page), Some(1));
        let terminal = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(terminal.failed_pages, vec![page]);
        assert_eq!(terminal.queued_requests, 0);
        assert_eq!(terminal.in_flight_requests, 0);
        assert_eq!(runtime.page_attempts(page), Some(2));
        assert!(runtime.is_terminal_failure(page));
        assert_eq!(runtime.terminal_failures(), &BTreeSet::from([page]));
        assert_eq!(
            runtime.page_transport_failure(page),
            Some(&LodPageTransportFailure::transport("forced begin failure"))
        );
        assert_eq!(runtime.transport_mut().begin_count, 2);

        // Visibility keeps requesting the page, but terminal state suppresses
        // enqueue/start and does not repeatedly report the same transition.
        for _ in 0..8 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert!(frame.started_pages.is_empty());
            assert!(frame.failed_pages.is_empty());
            assert_eq!(frame.queued_requests, 0);
            assert_eq!(frame.in_flight_requests, 0);
        }
        assert_eq!(runtime.transport_mut().begin_count, 2);

        runtime.transport_mut().fail_begin = false;
        assert!(runtime.retry_terminal_failure(page).unwrap());
        assert!(!runtime.is_terminal_failure(page));
        assert_eq!(runtime.page_attempts(page), None);
        assert_eq!(runtime.page_transport_failure(page), None);
        let restarted = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(restarted.started_pages, vec![page]);
        assert_eq!(runtime.page_attempts(page), Some(1));
        assert_eq!(runtime.transport_mut().begin_count, 3);

        let verifying = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(verifying.completed_pages.is_empty());
        assert_eq!(verifying.preprocess_stats().submitted, 1);
        assert_eq!(
            verifying.preprocess_stats().cooperative_decoded_gaussians,
            0
        );
        let completed = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(completed.completed_pages, vec![page]);
        assert_eq!(runtime.page_attempts(page), None);
        assert!(!runtime.is_terminal_failure(page));
        assert!(runtime.terminal_failures().is_empty());
        assert!(!runtime.retry_terminal_failure(page).unwrap());
    }

    #[test]
    fn coarsest_streams_one_candidate_for_over_100m_logical_gaussians() {
        let mut fixture = virtual_runtime_fixture();
        fixture.lod_settings.selection_mode = LodSelectionMode::Frozen;
        let source_gaussian_count = fixture.manifest.header.source_gaussian_count;
        assert!(source_gaussian_count > 100_000_000);
        assert_eq!(fixture.manifest.nodes.len(), VIRTUAL_NODE_COUNT as usize);
        assert_eq!(fixture.manifest.pages.len(), VIRTUAL_NODE_COUNT as usize);
        assert!(u64::from(VIRTUAL_NODE_COUNT) * 1_000 < source_gaussian_count);
        assert_eq!(
            fixture
                .manifest
                .pages
                .iter()
                .filter(|page| page.storage.is_some())
                .count(),
            1
        );
        assert!(fixture.encoded_root_bytes < 4 * 1024);

        let root_node = fixture.manifest.roots[0];
        let root_page = fixture.manifest.nodes[0].representation.page;
        let mut runtime = LodStreamingRuntime::new(
            fixture.manifest,
            fixture.transport,
            &fixture.lod_settings,
            &fixture.streaming_settings,
        )
        .unwrap();
        assert_eq!(
            runtime.hierarchy().manifest().header.source_gaussian_count,
            source_gaussian_count
        );
        assert_eq!(runtime.descriptors.len(), VIRTUAL_NODE_COUNT as usize);
        assert_eq!(
            runtime.atlas_layout().gaussians_per_slot,
            VirtualCityScene::default().gaussians_per_page
        );
        assert!(runtime.decoded_pages.is_empty());

        let captured_view = view();
        let moved_view = LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, 800.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        let first = runtime
            .update(
                captured_view,
                &fixture.lod_settings,
                &fixture.streaming_settings,
            )
            .unwrap();
        assert!(first.selection_view_frozen());
        assert_eq!(
            runtime.views[&LodRuntimeViewId::default()].frozen_selection_view,
            Some(captured_view)
        );
        assert_eq!(first.candidate_count(), 0);
        assert_eq!(first.started_pages, vec![root_page]);
        assert_eq!(first.in_flight_requests, 1);
        assert_eq!(first.cache.resident_gaussians, 0);

        let verifying = runtime
            .update(
                moved_view,
                &fixture.lod_settings,
                &fixture.streaming_settings,
            )
            .unwrap();
        assert!(verifying.selection_view_frozen());
        assert_eq!(
            runtime.views[&LodRuntimeViewId::default()].frozen_selection_view,
            Some(captured_view)
        );
        assert!(verifying.completed_pages.is_empty());
        assert_eq!(verifying.preprocess_stats().submitted, 1);
        assert_eq!(
            verifying.preprocess_stats().cooperative_decoded_gaussians,
            0
        );
        let resident = runtime
            .update(
                moved_view,
                &fixture.lod_settings,
                &fixture.streaming_settings,
            )
            .unwrap();
        assert!(resident.selection_view_frozen());
        assert_eq!(resident.completed_pages, vec![root_page]);
        assert_eq!(resident.frontier.nodes, vec![root_node]);
        assert!(resident.frontier.requested_nodes.is_empty());
        assert_eq!(
            resident.frontier.status.requested_target,
            crate::gaussian::lod_settings::LodQualityTarget::Coarsest
        );
        assert_eq!(resident.frontier.status.achieved_max_target_ratio, 0.0);
        assert_eq!(resident.frontier.status.active_gaussians, 1);
        assert_eq!(resident.candidate_count(), 1);
        assert!(
            resident
                .candidate_frontier(1)
                .unwrap()
                .selection_view_frozen()
        );
        assert_eq!(resident.cache.resident_pages, 1);
        assert_eq!(resident.cache.resident_gaussians, 1);
        assert_eq!(runtime.decoded_pages.len(), 1);
        assert_eq!(runtime.cache().limits().max_pages, 2);
        assert_eq!(
            runtime.cache().limits().max_gaussians,
            u64::from(VirtualCityScene::default().gaussians_per_page)
        );
        let materialized_gaussians = runtime
            .decoded_pages
            .values()
            .map(|page| page.gaussians.len())
            .sum::<usize>();
        assert_eq!(materialized_gaussians, 1);
        assert!(materialized_gaussians as u64 * 100_000_000 < source_gaussian_count);

        let mut dynamic_settings = fixture.lod_settings.clone();
        dynamic_settings.selection_mode = LodSelectionMode::Dynamic;
        let unfrozen = runtime
            .update(moved_view, &dynamic_settings, &fixture.streaming_settings)
            .unwrap();
        assert!(!unfrozen.selection_view_frozen());
        assert_eq!(
            runtime.views[&LodRuntimeViewId::default()].frozen_selection_view,
            None
        );
    }

    #[test]
    fn candidate_frontier_is_bounded_and_non_overlapping() {
        let frame = LodStreamFrame {
            view: LodRuntimeViewId::default(),
            frontier: LodFrontier {
                nodes: Vec::new(),
                requested_nodes: Vec::new(),
                status: crate::gaussian::lod_settings::LodEffectiveStatus {
                    active_gaussians: 10,
                    ..Default::default()
                },
            },
            physical_ranges: vec![LodPhysicalRange {
                node: LodNodeId(1),
                page: LodPageId(1),
                slot: AtlasSlot {
                    index: 0,
                    generation: 1,
                },
                physical_start: 0,
                count: 10,
            }],
            selection_view_frozen: true,
            complete_resident_cut: true,
            cache: Default::default(),
            queued_requests: 0,
            in_flight_requests: 0,
            preprocess: Default::default(),
            capacity_blocked_requests: 0,
            started_pages: Vec::new(),
            completed_pages: Vec::new(),
            preprocess_failed_pages: Vec::new(),
            failed_pages: Vec::new(),
        };
        assert!(matches!(
            frame.candidate_frontier(9),
            Err(LodRuntimeError::CandidateExpansionLimit { .. })
        ));
        let frontier = frame.candidate_frontier(10).unwrap();
        assert_eq!(frontier.candidate_count(), 10);
        assert_eq!(frontier.quality_status(), &frame.frontier.status);
        assert!(frontier.selection_view_frozen());

        let mut overlapping = frame.clone();
        overlapping.physical_ranges.push(LodPhysicalRange {
            node: LodNodeId(2),
            page: LodPageId(2),
            slot: AtlasSlot {
                index: 1,
                generation: 1,
            },
            physical_start: 5,
            count: 5,
        });
        overlapping.frontier.status.active_gaussians = 15;
        assert!(matches!(
            overlapping.candidate_frontier(15),
            Err(LodRuntimeError::OverlappingPhysicalRanges { .. })
        ));
    }

    #[test]
    fn runtime_hysteresis_history_requires_an_unchanged_lod_policy() {
        let original = GaussianLodSettings {
            quality: 0.5,
            ..Default::default()
        };
        let mut state = LodRuntimeViewState::default();
        let frontier = [LodNodeId(3), LodNodeId(4)];
        state.commit_frontier(&frontier, &original);

        assert_eq!(state.hysteresis_frontier(&original), frontier);

        let mut changed_quality = original.clone();
        changed_quality.quality = 0.75;
        assert!(state.hysteresis_frontier(&changed_quality).is_empty());

        // Committing a successful selection resets history to the new policy.
        state.commit_frontier(&[LodNodeId(1)], &changed_quality);
        assert_eq!(state.hysteresis_frontier(&changed_quality), [LodNodeId(1)]);

        let mut ignored_residency_policy = changed_quality.clone();
        ignored_residency_policy.budgets.max_resident_bytes /= 2;
        ignored_residency_policy.budgets.max_pending_requests /= 2;
        assert_eq!(
            state.hysteresis_frontier(&ignored_residency_policy),
            [LodNodeId(1)],
            "non-selection residency budgets must not churn hysteresis history"
        );

        let mut changed_policy = changed_quality.clone();
        changed_policy.hysteresis *= 0.5;
        assert!(state.hysteresis_frontier(&changed_policy).is_empty());
    }

    #[test]
    fn frozen_selection_captures_moves_invariantly_and_recaptures_after_unfreeze() {
        let initial = view();
        let moved = LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, 80.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        let recapture =
            LodView::orthographic(bevy::math::Vec3::new(4.0, 2.0, 12.0), 720.0, 20.0, 0.01);
        let mut state = LodRuntimeViewState::default();

        assert_eq!(
            state.selection_view(initial, LodSelectionMode::Frozen),
            initial
        );
        assert_eq!(
            state.selection_view(moved, LodSelectionMode::Frozen),
            initial
        );
        assert_eq!(state.frozen_selection_view, Some(initial));

        assert_eq!(
            state.selection_view(moved, LodSelectionMode::Dynamic),
            moved
        );
        assert_eq!(state.frozen_selection_view, None);
        assert_eq!(
            state.selection_view(recapture, LodSelectionMode::Frozen),
            recapture
        );
        assert_eq!(state.frozen_selection_view, Some(recapture));
    }

    #[test]
    fn frozen_views_capture_independently_by_runtime_view_id() {
        let left = LodRuntimeViewId(21);
        let right = LodRuntimeViewId(22);
        let left_view = view();
        let right_view = LodView::perspective(
            bevy::math::Vec3::new(100.0, 0.0, 8.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        let moved = LodView::perspective(
            bevy::math::Vec3::new(0.0, 0.0, -100.0),
            720.0,
            60_f32.to_radians(),
            0.01,
        );
        let mut views = BTreeMap::<LodRuntimeViewId, LodRuntimeViewState>::new();

        assert_eq!(
            views
                .entry(left)
                .or_default()
                .selection_view(left_view, LodSelectionMode::Frozen),
            left_view
        );
        assert_eq!(
            views
                .entry(right)
                .or_default()
                .selection_view(right_view, LodSelectionMode::Frozen),
            right_view
        );
        assert_eq!(
            views
                .get_mut(&left)
                .unwrap()
                .selection_view(moved, LodSelectionMode::Frozen),
            left_view
        );
        assert_eq!(
            views
                .get_mut(&right)
                .unwrap()
                .selection_view(moved, LodSelectionMode::Frozen),
            right_view
        );
    }

    #[test]
    fn frozen_view_keeps_frontier_and_residency_progress_mutable() {
        let mut settings = GaussianLodSettings {
            quality: 0.5,
            selection_mode: LodSelectionMode::Frozen,
            ..Default::default()
        };
        let captured = view();
        let mut state = LodRuntimeViewState::default();
        state.selection_view(captured, settings.selection_mode);

        state.commit_frontier(&[LodNodeId(1)], &settings);
        state.selected_frontier.insert(LodPageId(10));
        state.requested_pages.insert(LodPageId(11));

        // A later residency publication may refine the frontier without
        // changing the captured camera snapshot.
        state.commit_frontier(&[LodNodeId(2), LodNodeId(3)], &settings);
        state.selected_frontier.insert(LodPageId(11));
        state.requested_pages.clear();
        assert_eq!(state.previous_frontier, [LodNodeId(2), LodNodeId(3)]);
        assert_eq!(
            state.selected_frontier,
            BTreeSet::from([LodPageId(10), LodPageId(11)])
        );
        assert!(state.requested_pages.is_empty());
        assert_eq!(state.frozen_selection_view, Some(captured));

        // Quality remains independently editable and resets hysteresis history;
        // it does not discard the frozen camera until Dynamic is requested.
        settings.quality = 0.75;
        assert!(state.hysteresis_frontier(&settings).is_empty());
        assert_eq!(
            state.selection_view(captured, settings.selection_mode),
            captured
        );
    }

    #[test]
    fn two_million_candidate_frontier_stores_one_physical_range() {
        let mut frame = LodStreamFrame {
            view: LodRuntimeViewId(7),
            frontier: LodFrontier {
                nodes: vec![LodNodeId(1)],
                requested_nodes: Vec::new(),
                status: crate::gaussian::lod_settings::LodEffectiveStatus {
                    active_gaussians: 2_000_000,
                    ..Default::default()
                },
            },
            physical_ranges: vec![LodPhysicalRange {
                node: LodNodeId(1),
                page: LodPageId(1),
                slot: AtlasSlot {
                    index: 0,
                    generation: 1,
                },
                physical_start: 0,
                count: 2_000_000,
            }],
            selection_view_frozen: false,
            complete_resident_cut: true,
            cache: Default::default(),
            queued_requests: 0,
            in_flight_requests: 0,
            preprocess: Default::default(),
            capacity_blocked_requests: 0,
            started_pages: Vec::new(),
            completed_pages: Vec::new(),
            preprocess_failed_pages: Vec::new(),
            failed_pages: Vec::new(),
        };
        let frontier = frame.candidate_frontier(2_000_000).unwrap();
        assert_eq!(frontier.candidate_count(), 2_000_000);
        assert_eq!(frontier.physical_ranges().len(), 1);
        assert_eq!(
            std::mem::size_of_val(frontier.physical_ranges()),
            std::mem::size_of::<LodPhysicalRange>()
        );
        frame.complete_resident_cut = false;
        assert_eq!(
            frame.candidate_frontier(2_000_000),
            Err(LodRuntimeError::NoResidentFrontier)
        );
    }

    #[test]
    fn candidate_capability_rejects_a_partial_multi_root_forest() {
        let (manifest, transport, settings, streaming) = two_root_fixture();
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();

        let empty = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(!empty.has_complete_resident_cut());
        assert_eq!(
            empty.candidate_frontier(2),
            Err(LodRuntimeError::NoResidentFrontier)
        );

        let verifying = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(verifying.completed_pages().is_empty());
        assert_eq!(verifying.preprocess_stats().submitted, 1);
        assert_eq!(
            verifying.preprocess_stats().cooperative_decoded_gaussians,
            0
        );
        let partial = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(partial.completed_pages().len(), 1);
        assert_eq!(partial.physical_ranges().len(), 1);
        assert_eq!(partial.frontier().requested_nodes.len(), 1);
        assert!(!partial.has_complete_resident_cut());
        assert_eq!(
            partial.candidate_frontier(2),
            Err(LodRuntimeError::NoResidentFrontier)
        );
    }

    #[test]
    fn cameras_share_residency_but_keep_independent_fallback_holds() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.quality = 0.0;
        let root_node = manifest.roots[0];
        let root = manifest
            .nodes
            .iter()
            .find(|node| node.id == root_node)
            .unwrap();
        let root_page = root.representation.page;
        let root_candidate_count = u64::from(root.representation.count);
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let left = LodRuntimeViewId(11);
        let right = LodRuntimeViewId(12);

        runtime
            .update_view(left, view(), &settings, &streaming)
            .unwrap();
        let verifying = runtime
            .update_view(left, view(), &settings, &streaming)
            .unwrap();
        assert_eq!(verifying.candidate_count(), 0);
        assert_eq!(verifying.preprocess_stats().submitted, 1);
        assert_eq!(
            verifying.preprocess_stats().cooperative_decoded_gaussians,
            0
        );
        let left_frame = runtime
            .update_view(left, view(), &settings, &streaming)
            .unwrap();
        assert_eq!(left_frame.candidate_count(), root_candidate_count);
        let right_frame = runtime
            .update_view(right, view(), &settings, &streaming)
            .unwrap();
        assert_eq!(right_frame.candidate_count(), root_candidate_count);
        assert_eq!(runtime.cache().get(root_page).unwrap().pin_count, 2);

        assert!(runtime.remove_view(left).unwrap());
        assert_eq!(runtime.cache().get(root_page).unwrap().pin_count, 1);
        assert!(runtime.remove_view(right).unwrap());
        assert_eq!(runtime.cache().get(root_page).unwrap().pin_count, 0);
        assert!(!runtime.remove_view(right).unwrap());
    }

    #[test]
    fn cameras_in_one_frame_share_request_and_decoded_byte_budgets() {
        let (manifest, transport, mut settings, streaming) = fixture();
        let page_bytes = manifest
            .pages
            .iter()
            .map(|descriptor| (descriptor.id, descriptor.decoded_len))
            .collect::<BTreeMap<_, _>>();
        let decoded_budget = page_bytes.values().copied().max().unwrap();
        settings.budgets.max_requests_per_frame = 1;
        settings.budgets.max_upload_bytes_per_frame = decoded_budget;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let left = LodRuntimeViewId(21);
        let right = LodRuntimeViewId(22);
        let mut previous_frame = None;

        for _ in 0..16 {
            let frame = runtime.begin_frame();
            previous_frame = Some(frame);
            let left_frame = runtime
                .update_view_in_frame(frame, left, view(), &settings, &streaming)
                .unwrap();
            let right_frame = runtime
                .update_view_in_frame(frame, right, view(), &settings, &streaming)
                .unwrap();
            assert!(left_frame.started_pages.len() + right_frame.started_pages.len() <= 1);
            let completed_bytes = left_frame
                .completed_pages
                .iter()
                .chain(&right_frame.completed_pages)
                .map(|page| page_bytes[page])
                .sum::<u64>();
            assert!(completed_bytes <= decoded_budget);
        }

        let stale = previous_frame.unwrap();
        let current = runtime.begin_frame();
        assert!(matches!(
            runtime.update_view_in_frame(stale, left, view(), &settings, &streaming),
            Err(LodRuntimeError::InvalidFrameToken {
                expected,
                actual
            }) if expected == current && actual == stale
        ));
    }

    #[test]
    fn atlas_addressing_is_rejected_before_any_gpu_sized_allocation() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_resident_pages = u32::MAX;
        assert!(matches!(
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming),
            Err(LodRuntimeError::AtlasAddressSpaceOverflow { .. })
        ));
    }

    #[test]
    fn aggregate_pending_budget_includes_queue_in_flight_and_capacity_blocked() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_pending_requests = 4;
        let pages = manifest
            .pages
            .iter()
            .map(|descriptor| descriptor.id)
            .take(5)
            .collect::<Vec<_>>();
        assert_eq!(pages.len(), 5);
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let request = |page| PageRequest::new(page, PageRequestPriority::visible(1));

        assert_eq!(
            runtime.queue.enqueue(request(pages[0])),
            RequestEnqueue::Enqueued
        );
        runtime.in_flight.insert(
            pages[1],
            InFlight {
                ticket: 101,
                request: request(pages[1]),
            },
        );
        runtime.in_flight.insert(
            pages[2],
            InFlight {
                ticket: 102,
                request: request(pages[2]),
            },
        );
        runtime.capacity_blocked.insert(pages[3], request(pages[3]));
        assert_eq!(runtime.pending_request_count(), 4);
        assert_eq!(
            runtime.enqueue_pending_request(request(pages[4])),
            RequestEnqueue::Rejected
        );
        assert_eq!(runtime.pending_request_count(), 4);
    }

    #[test]
    fn manifest_pages_are_rejected_before_transport_or_decode_when_over_budget() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.budgets.max_upload_bytes_per_frame = 1;
        assert!(matches!(
            LodStreamingRuntime::new(manifest.clone(), transport.clone(), &settings, &streaming),
            Err(LodRuntimeError::PageDecodedBytesExceedLimit { .. })
        ));

        let (_, _, settings, mut streaming) = fixture();
        streaming.max_encoded_page_bytes = 44;
        assert!(matches!(
            LodStreamingRuntime::new(manifest.clone(), transport.clone(), &settings, &streaming),
            Err(LodRuntimeError::PageEncodedBytesExceedLimit { .. })
        ));

        streaming.max_encoded_page_bytes = 43;
        assert!(matches!(
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming),
            Err(LodRuntimeError::EncodedPageLimitTooSmall {
                limit: 43,
                minimum: 44
            })
        ));
    }

    #[test]
    fn structural_reconfiguration_is_explicit_and_too_small_frame_budget_cannot_stall() {
        let (manifest, transport, mut settings, streaming) = fixture();
        let original_upload_budget = settings.budgets.max_upload_bytes_per_frame;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();
        let started = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(!started.started_pages.is_empty());

        settings.budgets.max_upload_bytes_per_frame = 1;
        assert!(matches!(
            runtime.update(view(), &settings, &streaming),
            Err(LodRuntimeError::PageDecodedBytesExceedLimit { limit: 1, .. })
        ));
        settings.budgets.max_upload_bytes_per_frame = original_upload_budget;

        settings.budgets.max_resident_pages -= 1;
        assert_eq!(
            runtime.update(view(), &settings, &streaming),
            Err(LodRuntimeError::StructuralSettingsChanged(
                "budgets.max_resident_pages"
            ))
        );
    }

    #[test]
    fn ready_pages_respect_the_cumulative_decoded_byte_budget_per_update() {
        let (manifest, transport, mut settings, streaming) = fixture();
        let page_bytes = manifest
            .pages
            .iter()
            .map(|descriptor| (descriptor.id, descriptor.decoded_len))
            .collect::<BTreeMap<_, _>>();
        let frame_budget = page_bytes.values().copied().max().unwrap();
        settings.budgets.max_upload_bytes_per_frame = frame_budget;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();

        for _ in 0..16 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            let completed_bytes = frame
                .completed_pages
                .iter()
                .map(|page| page_bytes[page])
                .sum::<u64>();
            assert!(completed_bytes <= frame_budget);
        }
    }

    #[test]
    fn pinned_cache_pressure_pauses_requests_until_pin_state_changes() {
        let (manifest, transport, mut settings, streaming) = fixture();
        settings.quality = 0.0;
        settings.budgets.max_resident_pages = 1;
        settings.budgets.max_pending_requests = 2;
        settings.budgets.max_resident_gaussians = manifest
            .pages
            .iter()
            .map(|page| u64::from(page.gaussian_count))
            .max()
            .unwrap();
        let max_page_bytes = manifest
            .pages
            .iter()
            .map(|page| page.decoded_len)
            .max()
            .unwrap();
        settings.budgets.max_resident_bytes = max_page_bytes;
        settings.budgets.max_upload_bytes_per_frame = max_page_bytes;
        let mut runtime =
            LodStreamingRuntime::new(manifest, transport, &settings, &streaming).unwrap();

        runtime.update(view(), &settings, &streaming).unwrap();
        runtime.update(view(), &settings, &streaming).unwrap();
        settings.quality = 1.0;

        let mut blocked = None;
        for _ in 0..64 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert!(
                frame.queued_requests + frame.in_flight_requests + frame.capacity_blocked_requests
                    <= settings.budgets.max_pending_requests
            );
            if frame.capacity_blocked_requests > 0
                && frame.in_flight_requests == 0
                && frame.queued_requests == 0
            {
                blocked = Some(frame.capacity_blocked_requests);
                break;
            }
        }
        let blocked = blocked.expect("all pinned-capacity requests should eventually pause");
        assert_eq!(runtime.capacity_blocked.len(), blocked as usize);
        for _ in 0..8 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert!(frame.started_pages.is_empty());
            assert_eq!(frame.capacity_blocked_requests, blocked);
        }

        assert!(runtime.remove_view(LodRuntimeViewId::default()).unwrap());
        let restarted = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(!restarted.started_pages.is_empty());
    }

    #[test]
    fn invalid_support_payload_exhausts_retry_budget_instead_of_redownloading_forever() {
        let scene = LodTestScene::screen_space_ladder();
        let mut lod = build_planar_3d_lod(
            &scene.cloud(),
            GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 16,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        let page_id = root_page(&lod.manifest);
        let page = lod
            .pages
            .iter_mut()
            .find(|page| page.id == page_id)
            .unwrap();
        page.gaussians[0].position_visibility.position[0] += 1_000_000.0;
        let encoded = encode_page(page).unwrap();
        let descriptor = lod
            .manifest
            .pages
            .iter_mut()
            .find(|descriptor| descriptor.id == page_id)
            .unwrap();
        descriptor.content_hash = page.content_hash();
        descriptor.storage = Some(LodPageStorage {
            uri: "memory://invalid-support".to_owned(),
            byte_range: None,
            encoded_len: encoded.len() as u64,
        });
        lod.manifest.validate().unwrap();

        let mut transport = MemoryPageTransport::default();
        transport.insert(page_id, encoded);
        let mut settings = GaussianLodSettings::default();
        settings.quality = 0.0;
        let streaming = GaussianStreamingSettings {
            max_concurrent_requests: 1,
            retry_limit: 0,
            ..Default::default()
        };
        let mut runtime =
            LodStreamingRuntime::new(lod.manifest, transport, &settings, &streaming).unwrap();

        let first = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(first.started_pages, vec![page_id]);
        let verifying = runtime.update(view(), &settings, &streaming).unwrap();
        assert!(verifying.failed_pages.is_empty());
        assert_eq!(verifying.preprocess_stats().submitted, 1);
        assert_eq!(
            verifying.preprocess_stats().cooperative_decoded_gaussians,
            0
        );
        let failed = runtime.update(view(), &settings, &streaming).unwrap();
        assert_eq!(failed.failed_pages, vec![page_id]);
        assert!(runtime.is_terminal_failure(page_id));
        for _ in 0..4 {
            let frame = runtime.update(view(), &settings, &streaming).unwrap();
            assert!(frame.started_pages.is_empty());
            assert!(frame.failed_pages.is_empty());
        }
    }

    #[test]
    fn decoded_support_must_stay_inside_advertised_page_bounds() {
        let scene = LodTestScene::screen_space_ladder();
        let mut lod = build_planar_3d_lod(
            &scene.cloud(),
            GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 16,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        let page = &mut lod.pages[0];
        let descriptor = lod
            .manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == page.id)
            .unwrap();
        page.gaussians[0].position_visibility.position[0] += 1_000_000.0;
        assert_eq!(
            crate::stream::preprocess::validate_decoded_page_bounds(page, descriptor, 3.0),
            Err(LodPagePreprocessError::PayloadOutsideDescriptor(page.id))
        );
    }
}
