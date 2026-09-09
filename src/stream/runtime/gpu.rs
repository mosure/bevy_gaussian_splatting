//! Authenticated paging driven by bounded GPU page IDs, without CPU selection.

use super::*;

#[derive(Default)]
pub(crate) struct GpuPageUpdate {
    pub completed_pages: Vec<LodPageId>,
    pub preprocess_failed_pages: Vec<LodPageId>,
    pub queued_requests: usize,
    pub in_flight_requests: usize,
    pub capacity_blocked_requests: usize,
}

impl<T: LodPageTransport> LodStreamingRuntime<T> {
    /// Called only on a newly instantiated GPU-only package. Its authored root
    /// pages replace the CPU selector's bootstrap/coverage guard; no CPU cut has
    /// been published and no selector history is transferred between modes.
    pub(crate) fn initialize_gpu_page_demands(&mut self) -> Result<(), LodRuntimeError> {
        debug_assert!(self.views.is_empty());
        while let Some(page) = self.coverage_guard.pinned_pages.first().copied() {
            self.cache
                .unpin_fallback(page)
                .map_err(LodRuntimeError::Cache)?;
            self.coverage_guard.pinned_pages.remove(&page);
        }
        self.coverage_guard.package_bootstrap_released = true;
        self.split_cohort_capacity_stall = None;
        self.wake_capacity_blocked();
        Ok(())
    }

    pub(crate) fn gpu_residency_revision(&self) -> u64 {
        self.residency_revision
    }

    /// Reacquire cached pages made absent by an earlier snapshot exclusion.
    /// Touching LRU age is insufficient: another completion may need the only
    /// unpinned slot before this demanded page is published again.
    pub(crate) fn retain_gpu_pending_pages(
        &mut self,
        request_order: &[LodPageId],
        published: &BTreeMap<LodPageId, AtlasSlot>,
        pending: &mut BTreeSet<LodPageId>,
    ) -> Result<(), LodRuntimeError> {
        for &page in request_order {
            if !published.contains_key(&page)
                && !pending.contains(&page)
                && self.cache.contains(page)
                && self.decoded_pages.contains_key(&page)
            {
                self.retain_resident_page(page)?;
                pending.insert(page);
            }
        }
        Ok(())
    }

    /// Runs the existing bounded transport/decode/admission pipeline once for a
    /// complete frame of GPU demand. Callers authenticate the feedback's source
    /// and generation; this boundary independently validates every page ID.
    pub(crate) fn update_gpu_page_demands(
        &mut self,
        demands: &[(LodRuntimeViewId, &BTreeSet<LodPageId>)],
        request_order: &[LodPageId],
        lod_settings: &GaussianLodSettings,
        streaming_settings: &GaussianStreamingSettings,
    ) -> Result<GpuPageUpdate, LodRuntimeError> {
        Self::validate_creation_settings(lod_settings, streaming_settings)?;
        self.structural_settings
            .validate_compatible(LodRuntimeStructuralSettings::new(
                lod_settings,
                streaming_settings,
            ))?;
        // GPU packages validate spatial opt-in before this selector-free page
        // drive. Missing transition metadata keeps the complete discrete cut.
        if !matches!(
            lod_settings.presentation_mode,
            LodPresentationMode::Discrete | LodPresentationMode::ContinuousMorph
        ) {
            return Err(LodRuntimeError::InvalidSettings(
                "GPU page demand requires discrete or spatial-morph presentation".into(),
            ));
        }
        for (_, pages) in demands {
            for &page in *pages {
                if self.hierarchy.page_descriptor(page).is_none() {
                    return Err(LodRuntimeError::MissingPageDescriptor(page));
                }
            }
        }
        if self.largest_decoded_page.1 > lod_settings.budgets.max_upload_bytes_per_frame {
            return Err(LodRuntimeError::PageDecodedBytesExceedLimit {
                page: self.largest_decoded_page.0,
                actual: self.largest_decoded_page.1,
                limit: lod_settings.budgets.max_upload_bytes_per_frame,
            });
        }
        let frame = self.begin_frame();
        let result = (|| {
            for &(view, pages) in demands {
                self.prime_package_pages_in_order(
                    frame,
                    view,
                    pages,
                    request_order
                        .iter()
                        .copied()
                        .filter(|page| pages.contains(page)),
                )?;
            }
            let mut update = GpuPageUpdate::default();
            let mut failed_pages = Vec::new();
            self.poll_pages(frame, lod_settings, streaming_settings, &mut failed_pages)?;
            update.completed_pages = self.commit_preprocessed_pages(
                frame,
                Some(request_order),
                lod_settings,
                streaming_settings,
                &mut update.preprocess_failed_pages,
                &mut failed_pages,
            )?;
            // All current views are primed. Obsolete requests must not consume
            // this frame's IO starts and then be cancelled by finish_frame.
            // Completion holds remain intact until the normal end of frame.
            self.cancel_undemanded_page_work(frame);
            // Refresh after polling, which can requeue retries or wake blocked
            // requests. Demotion is as important as promotion after camera motion.
            self.reprioritize_gpu_page_requests(request_order);
            self.start_requests(lod_settings, streaming_settings, &mut failed_pages);
            update.queued_requests = self.queue.len();
            update.in_flight_requests = self.in_flight.len();
            update.capacity_blocked_requests = self.capacity_blocked.len();
            Ok(update)
        })();
        let finished = self.finish_frame(frame);
        match (result, finished) {
            (Ok(update), Ok(())) => Ok(update),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }

    pub(super) fn reprioritize_gpu_page_requests(&mut self, request_order: &[LodPageId]) {
        if self.queue.is_empty() && self.capacity_blocked.is_empty() {
            return;
        }
        for (rank, &page) in request_order.iter().enumerate() {
            let priority = PageRequestPriority::visible(
                u32::MAX.saturating_sub(u32::try_from(rank).unwrap_or(u32::MAX)),
            );
            self.queue.set_priority(page, priority);
            if let Some(request) = self.capacity_blocked.get_mut(&page) {
                request.priority = priority;
            }
        }
    }
}
