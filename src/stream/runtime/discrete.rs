//! Package-only adjacent categorical transactions. The canonical selector still
//! owns the destination; these helpers bound demand and resolve independent
//! complete child cohorts without requiring the complete deep target to arrive.

pub(super) mod budget;

use super::*;
use crate::stream::hierarchy::apply_temporal_substitution_step_with_admission;

#[derive(Debug)]
pub(super) struct PreparedDiscreteWave {
    previous_nodes: Vec<LodNodeId>,
    active_gaussians: u64,
    substitutions: Vec<LodTemporalSubstitution<LodNodeId>>,
    eligible: BTreeSet<LodTemporalSubstitutionKey<LodNodeId>>,
}

impl PreparedDiscreteWave {
    fn new(
        previous_nodes: &[LodNodeId],
        active_gaussians: u64,
        substitutions: Vec<LodTemporalSubstitution<LodNodeId>>,
    ) -> Self {
        let eligible = substitutions
            .iter()
            .map(|substitution| substitution.key)
            .collect();
        Self {
            previous_nodes: previous_nodes.to_vec(),
            active_gaussians,
            substitutions,
            eligible,
        }
    }
}

impl From<LodSelectionError<LodNodeId>> for LodRuntimeError {
    fn from(error: LodSelectionError<LodNodeId>) -> Self {
        Self::Selection(error)
    }
}

impl<T: LodPageTransport> LodStreamingRuntime<T> {
    /// Plan one adjacent wave from the last ACTIVE package cut. Physical
    /// admission includes the complete retained cut and permanent root guard;
    /// shared pages are counted once. No page from a deeper rung is demanded.
    pub(crate) fn package_discrete_target_plan(
        &mut self,
        views: &[(LodRuntimeViewId, LodView)],
        current: &[(LodRuntimeViewId, Vec<LodNodeId>)],
        retained_pages: &BTreeSet<LodPageId>,
        settings: &GaussianLodSettings,
    ) -> Result<LodPackageTargetPlan, LodRuntimeError> {
        let destination = self.package_discrete_destination_plan(views, settings)?;
        if current.iter().any(|(_, nodes)| {
            self.discrete_active_count(nodes)
                .is_ok_and(|count| count > settings.budgets.max_active_gaussians)
        }) {
            return self.discrete_root_recovery(destination, views, current, settings, true);
        }
        self.discrete_wave_from_destination(destination, views, current, retained_pages, settings)
    }

    fn discrete_wave_from_destination(
        &mut self,
        mut destination: LodPackageTargetPlan,
        views: &[(LodRuntimeViewId, LodView)],
        current: &[(LodRuntimeViewId, Vec<LodNodeId>)],
        retained_pages: &BTreeSet<LodPageId>,
        settings: &GaussianLodSettings,
    ) -> Result<LodPackageTargetPlan, LodRuntimeError> {
        #[cfg(feature = "testing")]
        let _cpu_scope = crate::testing::lod_package_cpu::scope(
            crate::testing::lod_package_cpu::PackageCpuScope::DiscreteWavePlan,
        );

        if let Some(cursor) = self.package_discrete_view_cursor {
            let start = destination
                .views
                .partition_point(|target| target.view <= cursor);
            destination.views.rotate_left(start);
        }
        let mut capacity_rejected = false;
        let mut first_admitted_view = None;
        let mut admitted_pages = retained_pages.clone();
        admitted_pages.extend(
            self.hierarchy
                .roots()
                .iter()
                .filter_map(|node| self.hierarchy.page(*node)),
        );
        let (mut admitted_bytes, mut admitted_gaussians, _) =
            LodRuntimeCoverageGuard::page_footprint(&self.hierarchy, &admitted_pages)?;
        let limits = self.cache.limits();
        let mut target_views = Vec::with_capacity(views.len());
        // A bounded quota per view prevents one camera from consuming the
        // complete request window before another can admit its first cohort.
        let quota =
            (settings.budgets.max_requests_per_frame as usize / views.len().max(1)).clamp(1, 64);
        for target in destination.views {
            let nodes = current
                .iter()
                .find(|(id, _)| *id == target.view)
                .map(|(_, nodes)| nodes)
                .ok_or(LodRuntimeError::NoResidentFrontier)?;
            let view = self
                .views
                .get(&target.view)
                .and_then(|state| state.frozen_selection_view)
                .filter(|_| target.selection_view_frozen)
                .unwrap_or_else(|| views.iter().find(|(id, _)| *id == target.view).unwrap().1);
            let mut substitutions =
                temporal_substitution_candidates(&self.hierarchy, nodes, &target.frontier.nodes)
                    .map_err(LodRuntimeError::Selection)?;
            // Visible work must not wait behind unrelated off-screen merges
            // after a camera return. Within each visibility class, release
            // capacity before refining; admission still proves the full union.
            // Evaluate projection once per cohort, not per sort comparison.
            substitutions.sort_by_cached_key(|substitution| {
                let priority = self.discrete_priority(substitution.key.parent, view, settings);
                (
                    std::cmp::Reverse(priority.0),
                    substitution.key.direction,
                    std::cmp::Reverse(priority),
                    substitution.key.parent,
                )
            });
            let eligible = substitutions
                .iter()
                .map(|substitution| substitution.key)
                .collect();
            let active = self.discrete_active_count(nodes)?;
            let step = apply_temporal_substitution_step_with_admission(
                nodes,
                &target.frontier.nodes,
                active,
                &substitutions,
                &eligible,
                |_| true,
                LodTemporalStepBudget {
                    max_active_gaussians: settings.budgets.max_active_gaussians,
                    max_changed_gaussians: (settings.budgets.max_upload_bytes_per_frame
                        / std::mem::size_of::<crate::Gaussian3d>() as u64)
                        .max(1),
                    max_substitutions: quota,
                },
                true,
                |substitution| -> Result<bool, LodRuntimeError> {
                    let mut new_pages = BTreeSet::new();
                    for &node in &substitution.next_nodes {
                        let page = self
                            .hierarchy
                            .page(node)
                            .ok_or(LodRuntimeError::MissingNode(node))?;
                        if self.terminal_failures.contains(&page) {
                            return Ok(false);
                        }
                        if !admitted_pages.contains(&page) {
                            new_pages.insert(page);
                        }
                    }
                    let (bytes, gaussians, _) =
                        LodRuntimeCoverageGuard::page_footprint(&self.hierarchy, &new_pages)?;
                    let bytes = admitted_bytes
                        .checked_add(bytes)
                        .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
                    let gaussians = admitted_gaussians
                        .checked_add(gaussians)
                        .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
                    if admitted_pages.len() + new_pages.len() > limits.max_pages as usize
                        || bytes > limits.max_bytes
                        || gaussians > limits.max_gaussians
                    {
                        capacity_rejected = true;
                        return Ok(false);
                    }
                    // Only a cohort which will enter this complete cut can
                    // reserve physical capacity from subsequent cohorts/views.
                    admitted_pages.extend(new_pages);
                    admitted_bytes = bytes;
                    admitted_gaussians = gaussians;
                    Ok(true)
                },
            )?;
            if !step.substitutions.is_empty() && first_admitted_view.is_none() {
                first_admitted_view = Some(target.view);
            }

            let view = views.iter().find(|(id, _)| *id == target.view).unwrap().1;
            let mut target = self.discrete_plan_view(target, &step, view, settings)?;
            target.discrete_wave = Some(Arc::new(PreparedDiscreteWave::new(
                nodes,
                active,
                step.substitutions,
            )));
            target_views.push(target);
        }
        let stalled = capacity_rejected
            && first_admitted_view.is_none()
            && target_views.iter().any(|target| {
                let current = current
                    .iter()
                    .find(|(id, _)| *id == target.view)
                    .expect("current view");
                // discrete_plan_view already replaced the destination nodes with
                // the admitted wave, so compare against the cached destination.
                self.package_discrete_destination_cache
                    .as_ref()
                    .is_some_and(|cached| {
                        cached
                            .plan
                            .views
                            .iter()
                            .find(|ideal| ideal.view == target.view)
                            .is_some_and(|ideal| ideal.frontier.nodes != current.1)
                    })
            });
        let plan = self.discrete_plan_from_views(target_views)?;
        if stalled {
            let destination = self
                .package_discrete_destination_cache
                .as_ref()
                .expect("stalled destination")
                .plan
                .clone();
            if let Some(direct) = self.discrete_exact_destination_if_admissible(
                destination,
                &admitted_pages,
                settings,
            )? {
                return Ok(direct);
            }
            return self.discrete_root_recovery(plan, views, current, settings, false);
        }
        if let Some(view) = first_admitted_view {
            self.package_discrete_view_cursor = Some(view);
        }
        Ok(plan)
    }

    /// Resolve any ready, independent cohorts. A delayed sibling retains its
    /// complete parent while other branches/views can publish their own step.
    pub(crate) fn package_discrete_resident_plan(
        &self,
        admitted: &LodPackageTargetPlan,
        views: &[(LodRuntimeViewId, LodView)],
        current: &[(LodRuntimeViewId, Vec<LodNodeId>)],
        settings: &GaussianLodSettings,
    ) -> Result<Option<LodPackageTargetPlan>, LodRuntimeError> {
        #[cfg(feature = "testing")]
        let _cpu_scope = crate::testing::lod_package_cpu::scope(
            crate::testing::lod_package_cpu::PackageCpuScope::DiscreteResidentPlan,
        );

        if admitted.discrete_direct_transaction {
            return Ok(admitted
                .pages
                .iter()
                .all(|page| {
                    self.cache.contains(*page)
                        && self.decoded_pages.contains_key(page)
                        && !self.terminal_failures.contains(page)
                })
                .then(|| admitted.clone()));
        }
        let is_resident = |node| {
            self.hierarchy.page(node).is_some_and(|page| {
                self.cache.contains(page)
                    && self.decoded_pages.contains_key(&page)
                    && !self.terminal_failures.contains(&page)
            })
        };
        let mut waiting = false;
        let only_waiting = admitted.views.iter().all(|target| {
            let Some(wave) = target.discrete_wave.as_deref() else {
                return false;
            };
            if !current
                .iter()
                .any(|(id, nodes)| *id == target.view && *nodes == wave.previous_nodes)
            {
                return false;
            }
            waiting |= !wave.substitutions.is_empty();
            !wave
                .substitutions
                .iter()
                .any(|substitution| substitution.next_nodes.iter().copied().all(is_resident))
        });
        if only_waiting && waiting {
            // No admitted cohort can advance. Preserve the complete current
            // cut without rebuilding its sets or projecting every node on each
            // transport poll. A ready independent cohort exits this fast path.
            return Ok(None);
        }
        let mut changed = false;
        let mut already_equal = true;
        let mut ready_views = Vec::with_capacity(admitted.views.len());
        for target in &admitted.views {
            let nodes = current
                .iter()
                .find(|(id, _)| *id == target.view)
                .map(|(_, nodes)| nodes)
                .ok_or(LodRuntimeError::NoResidentFrontier)?;
            already_equal &= *nodes == target.frontier.nodes;
            let rebuilt;
            let wave = if let Some(wave) = target
                .discrete_wave
                .as_deref()
                .filter(|wave| wave.previous_nodes == *nodes)
            {
                wave
            } else {
                rebuilt = PreparedDiscreteWave::new(
                    nodes,
                    self.discrete_active_count(nodes)?,
                    temporal_substitution_candidates(
                        &self.hierarchy,
                        nodes,
                        &target.frontier.nodes,
                    )
                    .map_err(LodRuntimeError::Selection)?,
                );
                &rebuilt
            };
            let step = apply_temporal_substitution_step_with_admission(
                nodes,
                &target.frontier.nodes,
                wave.active_gaussians,
                &wave.substitutions,
                &wave.eligible,
                is_resident,
                LodTemporalStepBudget {
                    max_active_gaussians: settings.budgets.max_active_gaussians,
                    max_changed_gaussians: u64::MAX,
                    max_substitutions: usize::MAX,
                },
                true,
                |_| Ok::<_, LodRuntimeError>(true),
            )?;
            changed |= !step.substitutions.is_empty();
            let view = views
                .iter()
                .find(|(id, _)| *id == target.view)
                .map(|(_, view)| *view)
                .ok_or(LodRuntimeError::NoResidentFrontier)?;
            ready_views.push(self.discrete_plan_view(target.clone(), &step, view, settings)?);
        }
        if !changed && !already_equal {
            return Ok(None);
        }
        self.discrete_plan_from_views(ready_views).map(Some)
    }

    /// Shared packing can make the complete deep destination cheaper than
    /// its intermediate hierarchy path. At an adjacent-capacity stall, use the
    /// exact retained-old + target + root page union as the admission proof.
    /// Existing package materialization/upload budgets still span this one
    /// complete categorical transaction over as many frames as necessary.
    fn discrete_exact_destination_if_admissible(
        &self,
        mut destination: LodPackageTargetPlan,
        retained_with_roots: &BTreeSet<LodPageId>,
        settings: &GaussianLodSettings,
    ) -> Result<Option<LodPackageTargetPlan>, LodRuntimeError> {
        if destination.views.iter().any(|target| {
            target.frontier.status.active_gaussians > settings.budgets.max_active_gaussians
        }) || destination
            .pages
            .iter()
            .any(|page| self.terminal_failures.contains(page))
        {
            return Ok(None);
        }
        let union = retained_with_roots
            .union(&destination.pages)
            .copied()
            .collect::<BTreeSet<_>>();
        let limits = self.cache.limits();
        let (bytes, gaussians, _) =
            LodRuntimeCoverageGuard::page_footprint(&self.hierarchy, &union)?;
        if union.len() > limits.max_pages as usize
            || bytes > limits.max_bytes
            || gaussians > limits.max_gaussians
        {
            return Ok(None);
        }
        destination.discrete_direct_transaction = true;
        Ok(Some(destination))
    }

    /// The permanent decoded root guard can free an inherited fully pinned
    /// cut without allocating a missing intermediate parent. Normal waves keep
    /// measured replacement headroom and never use this recovery transaction.
    fn discrete_root_recovery(
        &self,
        mut destination: LodPackageTargetPlan,
        views: &[(LodRuntimeViewId, LodView)],
        current: &[(LodRuntimeViewId, Vec<LodNodeId>)],
        settings: &GaussianLodSettings,
        only_over_active: bool,
    ) -> Result<LodPackageTargetPlan, LodRuntimeError> {
        let roots = self.hierarchy.roots().to_vec();
        let roots_ready = roots.iter().all(|node| {
            self.hierarchy.page(*node).is_some_and(|page| {
                self.cache.contains(page)
                    && self.decoded_pages.contains_key(&page)
                    && !self.terminal_failures.contains(&page)
            })
        });
        if !roots_ready {
            return Ok(destination);
        }
        let mut changed = false;
        for target in &mut destination.views {
            let nodes = &current
                .iter()
                .find(|(id, _)| *id == target.view)
                .ok_or(LodRuntimeError::NoResidentFrontier)?
                .1;
            if !only_over_active
                || self.discrete_active_count(nodes)? > settings.budgets.max_active_gaussians
            {
                changed |= *nodes != roots;
                let step = crate::stream::hierarchy::LodTemporalFrontierStep {
                    nodes: roots.clone(),
                    substitutions: Vec::new(),
                    requested_nodes: Vec::new(),
                    changed_gaussians: 0,
                    atomic_budget_overshoot: 0,
                    reached_target: false,
                };
                let view = views
                    .iter()
                    .find(|(id, _)| *id == target.view)
                    .ok_or(LodRuntimeError::NoResidentFrontier)?
                    .1;
                *target = self.discrete_plan_view(target.clone(), &step, view, settings)?;
            } else {
                // This recovery transaction only releases ownership. Unrelated
                // views retain their current cut, never unrequested new pages.
                let step = crate::stream::hierarchy::LodTemporalFrontierStep {
                    nodes: nodes.clone(),
                    substitutions: Vec::new(),
                    requested_nodes: Vec::new(),
                    changed_gaussians: 0,
                    atomic_budget_overshoot: 0,
                    reached_target: false,
                };
                let view = views
                    .iter()
                    .find(|(id, _)| *id == target.view)
                    .ok_or(LodRuntimeError::NoResidentFrontier)?
                    .1;
                *target = self.discrete_plan_view(target.clone(), &step, view, settings)?;
            }
        }
        let mut plan = self.discrete_plan_from_views(destination.views)?;
        plan.discrete_direct_transaction = changed;
        Ok(plan)
    }

    fn discrete_active_count(&self, nodes: &[LodNodeId]) -> Result<u64, LodRuntimeError> {
        nodes.iter().try_fold(0_u64, |count, &node| {
            let metrics = self
                .hierarchy
                .metrics(node)
                .ok_or(LodRuntimeError::MissingNode(node))?;
            count
                .checked_add(u64::from(metrics.representative_count))
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)
        })
    }

    fn discrete_plan_view(
        &self,
        mut target: LodPackageTargetView,
        step: &crate::stream::hierarchy::LodTemporalFrontierStep<LodNodeId>,
        view: LodView,
        settings: &GaussianLodSettings,
    ) -> Result<LodPackageTargetView, LodRuntimeError> {
        target.discrete_wave = None;
        let visibility = crate::stream::hierarchy::LodViewEvaluator::from_view(view);
        target.frontier = temporal_frontier_with_visibility(
            &self.hierarchy,
            &target.frontier,
            step,
            view,
            settings,
            |node, _| {
                !settings.frustum_culling
                    || self.hierarchy.node(node).is_none_or(|node| {
                        visibility.bounds_are_visible(node.bounds, settings.frustum_margin)
                    })
            },
        )
        .map_err(LodRuntimeError::Selection)?;
        // This is an immutable complete-cut plan. Demand is its physical page
        // set; missing cohorts are represented by the still-retained parent.
        target.frontier.requested_nodes.clear();
        Ok(target)
    }

    fn discrete_plan_from_views(
        &self,
        views: Vec<LodPackageTargetView>,
    ) -> Result<LodPackageTargetPlan, LodRuntimeError> {
        let mut pages = BTreeSet::new();
        for view in &views {
            for &node in &view.frontier.nodes {
                pages.insert(
                    self.hierarchy
                        .page(node)
                        .ok_or(LodRuntimeError::MissingNode(node))?,
                );
            }
        }
        Ok(LodPackageTargetPlan {
            pages,
            views,
            discrete_direct_transaction: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gaussian::formats::{
            planar_3d_chunked::{LodPageKind, LodPageStorage},
            planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
        },
        io::lod::encode_page,
        stream::transport::PagePayload,
        testing::LodTestScene,
    };

    struct DelayedPages {
        pages: BTreeMap<LodPageId, Vec<u8>>,
        blocked: BTreeSet<LodPageId>,
        started: BTreeSet<LodPageId>,
        canceled: BTreeSet<LodPageId>,
    }
    impl LodPageTransport for DelayedPages {
        type Ticket = LodPageId;
        type Error = ();
        fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
            self.started.insert(request.page_id);
            Ok(request.page_id)
        }
        fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
            if self.blocked.contains(ticket) {
                PagePoll::Pending
            } else {
                PagePoll::Ready(PagePayload::new(*ticket, self.pages[ticket].clone()))
            }
        }
        fn cancel(&mut self, ticket: &Self::Ticket) {
            self.canceled.insert(*ticket);
        }
    }

    fn fixture() -> (
        LodStreamingRuntime<DelayedPages>,
        GaussianLodSettings,
        GaussianStreamingSettings,
        LodView,
    ) {
        fixture_with_page_limit(2048)
    }

    fn fixture_with_page_limit(
        max_pages: u32,
    ) -> (
        LodStreamingRuntime<DelayedPages>,
        GaussianLodSettings,
        GaussianStreamingSettings,
        LodView,
    ) {
        let mut built = build_planar_3d_lod(
            &LodTestScene::nested_octants(3).cloud(),
            GaussianLodBuildSettings {
                branching_factor: 2,
                leaf_capacity: 8,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        // Give each logical node its own authenticated page, so delaying one
        // branch cannot accidentally delay a sibling sharing the same page.
        let mut pages = BTreeMap::new();
        let mut descriptors = Vec::new();
        let mut blocked = BTreeSet::new();
        for node in &mut built.manifest.nodes {
            let source = built
                .pages
                .iter()
                .find(|page| page.id == node.representation.page)
                .unwrap();
            let first = node.representation.offset as usize;
            let count = node.representation.count as usize;
            let id = LodPageId(node.id.0);
            let page =
                PlanarGaussian3dPage::new(id, source.gaussians[first..first + count].to_vec());
            let encoded = encode_page(&page).unwrap();
            let leaf = node.is_leaf();
            if leaf {
                blocked.insert(id);
            }
            descriptors.push(LodPageDescriptor {
                id,
                kind: if leaf {
                    LodPageKind::SourceLeaves
                } else {
                    LodPageKind::Representatives
                },
                encoding: crate::gaussian::formats::planar_3d_chunked::LodPageEncoding::F32Planar,
                gaussian_count: count as u32,
                decoded_len: count as u64 * std::mem::size_of::<crate::Gaussian3d>() as u64,
                content_hash: page.content_hash(),
                bounds: node.bounds,
                storage: Some(LodPageStorage {
                    uri: format!("page-{}", id.0),
                    byte_range: None,
                    encoded_len: encoded.len() as u64,
                }),
            });
            pages.insert(id, encoded);
            node.representation.page = id;
            node.representation.offset = 0;
        }
        built.manifest.pages = descriptors;
        built.manifest.header.page_count = built.manifest.pages.len() as u32;
        built.manifest.header.stored_gaussian_count = built
            .manifest
            .pages
            .iter()
            .map(|page| u64::from(page.gaussian_count))
            .sum();
        built.manifest.header.required_features &= !LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES;
        built.manifest.validate().unwrap();
        let mut settings = GaussianLodSettings {
            quality: 1.0,
            presentation_mode: LodPresentationMode::Discrete,
            frustum_culling: false,
            ..Default::default()
        };
        settings.budgets.max_active_gaussians = 4096;
        settings.budgets.max_resident_pages = max_pages;
        settings.budgets.max_resident_gaussians = 32768;
        settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
        settings.budgets.max_requests_per_frame = 64;
        let streaming = GaussianStreamingSettings {
            max_concurrent_requests: 8,
            ..Default::default()
        };
        let view = LodView::perspective(bevy::math::Vec3::new(0.0, 0.0, 10.0), 540.0, 1.0, 0.1);
        let mut runtime = LodStreamingRuntime::new(
            built.manifest,
            DelayedPages {
                pages,
                blocked,
                started: BTreeSet::new(),
                canceled: BTreeSet::new(),
            },
            &settings,
            &streaming,
        )
        .unwrap();
        let mut coarse = settings.clone();
        coarse.quality = 0.0;
        for _ in 0..64 {
            runtime.update(view, &coarse, &streaming).unwrap();
            if runtime.hierarchy.roots().iter().all(|node| {
                runtime
                    .decoded_page(runtime.hierarchy.page(*node).unwrap())
                    .is_some()
            }) {
                break;
            }
        }
        (runtime, settings, streaming, view)
    }

    fn assert_complete(runtime: &LodStreamingRuntime<DelayedPages>, nodes: &[LodNodeId]) {
        let mut source = nodes
            .iter()
            .map(|&node| runtime.hierarchy.node(node).unwrap().source)
            .collect::<Vec<_>>();
        source.sort_by_key(|range| range.start);
        let mut end = 0;
        for range in source {
            assert_eq!(range.start, end);
            end += range.count;
        }
        assert_eq!(
            end,
            runtime.hierarchy.manifest().header.source_gaussian_count
        );
    }

    fn advance(
        runtime: &mut LodStreamingRuntime<DelayedPages>,
        plan: &LodPackageTargetPlan,
        settings: &GaussianLodSettings,
        streaming: &GaussianStreamingSettings,
        view: LodView,
    ) {
        let frame = runtime.begin_frame();
        runtime
            .prime_package_pages_in_frame(frame, LodRuntimeViewId(7), plan.pages())
            .unwrap();
        let mut coarse = settings.clone();
        coarse.quality = 0.0;
        runtime
            .update_view_in_frame(frame, LodRuntimeViewId(99), view, &coarse, streaming)
            .unwrap();
        runtime.finish_frame(frame).unwrap();
    }

    #[test]
    fn discrete_delayed_final_pages_publish_multiple_complete_intermediate_cuts() {
        let (mut runtime, settings, streaming, view) = fixture();
        let views = [(LodRuntimeViewId(7), view)];
        let mut current = vec![(LodRuntimeViewId(7), runtime.hierarchy.roots().to_vec())];
        let mut counts = vec![runtime.discrete_active_count(&current[0].1).unwrap()];
        for _ in 0..3 {
            let retained = current[0]
                .1
                .iter()
                .map(|node| runtime.hierarchy.page(*node).unwrap())
                .collect();
            let plan = runtime
                .package_discrete_target_plan(&views, &current, &retained, &settings)
                .unwrap();
            let mut ready = None;
            for _ in 0..128 {
                advance(&mut runtime, &plan, &settings, &streaming, view);
                ready = runtime
                    .package_discrete_resident_plan(&plan, &views, &current, &settings)
                    .unwrap();
                if ready.is_some() {
                    break;
                }
            }
            let ready =
                ready.expect("a complete intermediate cohort arrives before blocked final leaves");
            let next = ready.views[0].frontier.nodes.clone();
            assert_complete(&runtime, &next);
            let candidates = runtime
                .package_target_candidates(&ready, &views, &settings)
                .unwrap()
                .unwrap();
            assert!(!candidates.matches_live_target);
            let count = runtime.discrete_active_count(&next).unwrap();
            assert!(count > *counts.last().unwrap());
            counts.push(count);
            current[0].1 = next;
        }
        assert!(counts.len() >= 3);
        assert!(
            runtime
                .transport
                .blocked
                .iter()
                .all(|page| runtime.decoded_page(*page).is_none())
        );
    }

    #[test]
    fn discrete_missing_sibling_keeps_parent_while_independent_branch_advances() {
        let (mut runtime, settings, streaming, view) = fixture();
        let views = [(LodRuntimeViewId(7), view)];
        let root = runtime.hierarchy.roots()[0];
        let current = vec![(LodRuntimeViewId(7), vec![root])];
        let plan = runtime
            .package_discrete_target_plan(&views, &current, &BTreeSet::new(), &settings)
            .unwrap();
        for _ in 0..64 {
            advance(&mut runtime, &plan, &settings, &streaming, view);
        }
        let next = runtime
            .package_discrete_resident_plan(&plan, &views, &current, &settings)
            .unwrap()
            .unwrap();
        let parents = next.views[0].frontier.nodes.clone();
        assert_eq!(parents.len(), 2);
        let blocked_node = runtime.hierarchy.children(parents[0])[0];
        let blocked_page = runtime.hierarchy.page(blocked_node).unwrap();
        runtime.transport.blocked.insert(blocked_page);
        let current = vec![(LodRuntimeViewId(7), parents.clone())];
        let retained = parents
            .iter()
            .map(|node| runtime.hierarchy.page(*node).unwrap())
            .collect();
        let plan = runtime
            .package_discrete_target_plan(&views, &current, &retained, &settings)
            .unwrap();
        let mut uncached = plan.clone();
        for target in &mut uncached.views {
            target.discrete_wave = None;
        }
        assert!(
            runtime
                .package_discrete_resident_plan(&plan, &views, &current, &settings)
                .unwrap()
                .is_none()
        );
        let mut ready = None;
        for _ in 0..128 {
            advance(&mut runtime, &plan, &settings, &streaming, view);
            ready = runtime
                .package_discrete_resident_plan(&plan, &views, &current, &settings)
                .unwrap();
            let rebuilt = runtime
                .package_discrete_resident_plan(&uncached, &views, &current, &settings)
                .unwrap();
            assert_eq!(
                ready.as_ref().map(|plan| &plan.views[0].frontier),
                rebuilt.as_ref().map(|plan| &plan.views[0].frontier),
                "prepared cohort polling preserves the uncached complete-cut result"
            );
            if ready.is_some() {
                break;
            }
        }
        let ready = ready.expect("unrelated branch must not wait for the blocked sibling");
        let nodes = &ready.views[0].frontier.nodes;
        assert!(nodes.contains(&parents[0]));
        assert!(!nodes.contains(&parents[1]));
        assert_complete(&runtime, nodes);
        assert!(runtime.decoded_page(blocked_page).is_none());
        runtime
            .cancel_package_view_work(&[LodRuntimeViewId(7)])
            .unwrap();
        assert!(runtime.transport.canceled.contains(&blocked_page));
        assert!(runtime.in_flight.is_empty());
    }

    #[test]
    fn discrete_camera_return_refines_visible_branch_before_unrelated_offscreen_merges() {
        let (mut runtime, mut settings, _, _) = fixture();
        settings.frustum_culling = true;
        settings.budgets.max_requests_per_frame = 1;
        let parents = runtime
            .hierarchy
            .children(runtime.hierarchy.roots()[0])
            .to_vec();
        let near = runtime.hierarchy.node(parents[0]).unwrap().bounds;
        let far = runtime.hierarchy.node(parents[1]).unwrap().bounds;
        // Parent bounds overlap at their centers. Put the returning camera
        // inside the first box's exclusive interval, then size the test frustum
        // from its actual separation from the other box.
        let (axis, lower, gap) = (0..3)
            .flat_map(|axis| {
                [
                    (axis, true, far.min[axis] - near.min[axis]),
                    (axis, false, near.max[axis] - far.max[axis]),
                ]
            })
            .max_by(|left, right| left.2.total_cmp(&right.2))
            .unwrap();
        assert!(gap > 0.0, "fixture requires distinct parent support");
        let inset = 0.25 * gap.min(near.max[axis] - near.min[axis]);
        let mut position = bevy::math::Vec3::from_array(near.center());
        position[axis] = if lower {
            near.min[axis] + inset
        } else {
            near.max[axis] - inset
        };
        let separation = if lower {
            far.min[axis] - position[axis]
        } else {
            position[axis] - far.max[axis]
        };
        assert!(separation > 0.0);
        let view = LodView::perspective(position, 540.0, 1.0, 0.01).with_clip_from_world(
            bevy::math::Mat4::from_scale(bevy::math::Vec3::splat(4.0 / separation))
                * bevy::math::Mat4::from_translation(-position),
        );
        assert!(view.bounds_are_visible(near, settings.frustum_margin));
        assert!(!view.bounds_are_visible(far, settings.frustum_margin));
        let near_children = runtime.hierarchy.children(parents[0]).to_vec();
        let far_children = runtime.hierarchy.children(parents[1]).to_vec();
        let id = LodRuntimeViewId(7);
        let mut nodes = vec![parents[0]];
        nodes.extend_from_slice(&far_children);
        nodes.sort_unstable();
        let current = vec![(id, nodes)];
        let mut destination = near_children.clone();
        destination.push(parents[1]);
        set_destination(&mut runtime, id, view, &settings, destination);
        let retained = retain_nodes(&runtime, &current);
        let plan = runtime
            .unconstrained_wave(&[(id, view)], &current, &retained, &settings)
            .unwrap();
        let next = &plan.views[0].frontier.nodes;
        assert!(
            !next.contains(&parents[0]),
            "visible refinement owns the one-cohort quota"
        );
        assert!(near_children.iter().all(|node| next.contains(node)));
        assert!(far_children.iter().all(|node| next.contains(node)));
        assert_complete(&runtime, next);
    }
    // Inject complete canonical cuts to isolate physical orchestration from
    // camera projection; the production cache key and planner remain in use.
    fn set_destination(
        runtime: &mut LodStreamingRuntime<DelayedPages>,
        id: LodRuntimeViewId,
        view: LodView,
        settings: &GaussianLodSettings,
        mut nodes: Vec<LodNodeId>,
    ) {
        nodes.sort_unstable();
        assert_complete(runtime, &nodes);
        let active = runtime.discrete_active_count(&nodes).unwrap();
        runtime
            .all_resident_target_frontier(id, view, false, settings)
            .unwrap();
        let cached = runtime
            .views
            .get_mut(&id)
            .unwrap()
            .all_resident_selection
            .as_mut()
            .unwrap();
        cached.frontier.nodes = nodes;
        cached.frontier.status.active_gaussians = active;
    }

    fn retain_nodes(
        runtime: &LodStreamingRuntime<DelayedPages>,
        current: &[(LodRuntimeViewId, Vec<LodNodeId>)],
    ) -> BTreeSet<LodPageId> {
        current
            .iter()
            .flat_map(|(_, nodes)| nodes)
            .chain(runtime.hierarchy.roots())
            .map(|node| runtime.hierarchy.page(*node).unwrap())
            .collect()
    }

    impl LodStreamingRuntime<DelayedPages> {
        fn unconstrained_wave(
            &mut self,
            views: &[(LodRuntimeViewId, LodView)],
            current: &[(LodRuntimeViewId, Vec<LodNodeId>)],
            retained: &BTreeSet<LodPageId>,
            settings: &GaussianLodSettings,
        ) -> Result<LodPackageTargetPlan, LodRuntimeError> {
            let destination = self.package_all_resident_target_plan(views, settings)?;
            self.discrete_wave_from_destination(destination, views, current, retained, settings)
        }
    }

    #[test]
    fn discrete_rejected_work_does_not_reserve_pages_needed_by_another_view() {
        let (mut runtime, mut settings, _, view) = fixture();
        settings.budgets.max_upload_bytes_per_frame = 1;
        let mut children = runtime
            .hierarchy
            .children(runtime.hierarchy.roots()[0])
            .to_vec();
        children.sort_by(|left, right| {
            runtime
                .discrete_priority(*right, view, &settings)
                .cmp(&runtime.discrete_priority(*left, view, &settings))
                .then_with(|| left.cmp(right))
        });
        let left = runtime.hierarchy.children(children[0]).to_vec();
        let right = runtime.hierarchy.children(children[1]).to_vec();
        let deeper = runtime.hierarchy.children(left[0]).to_vec();
        let ids = [LodRuntimeViewId(7), LodRuntimeViewId(8)];
        let current = vec![
            (ids[0], children.clone()),
            (ids[1], vec![left[0], left[1], children[1]]),
        ];
        set_destination(
            &mut runtime,
            ids[0],
            view,
            &settings,
            left.iter().chain(&right).copied().collect(),
        );
        set_destination(
            &mut runtime,
            ids[1],
            view,
            &settings,
            vec![deeper[0], deeper[1], left[1], children[1]],
        );
        let retained = retain_nodes(&runtime, &current);
        // The first view can replace L using pages retained by view2. Its R
        // split exceeds the changed-record budget and must not consume the
        // two free slots needed by view2's independent deeper split.
        runtime.cache = LodPageCache::new(PageCacheLimits {
            max_pages: retained.len() as u32 + 2,
            ..runtime.cache.limits()
        })
        .unwrap();
        let plan = runtime
            .unconstrained_wave(
                &[(ids[0], view), (ids[1], view)],
                &current,
                &retained,
                &settings,
            )
            .unwrap();
        let second = plan
            .views
            .iter()
            .find(|target| target.view == ids[1])
            .unwrap();
        assert!(
            deeper
                .iter()
                .all(|node| second.frontier.nodes.contains(node))
        );
        assert!(!second.frontier.nodes.contains(&left[0]));
        assert_complete(&runtime, &second.frontier.nodes);
    }

    #[test]
    fn discrete_competing_views_rotate_the_first_physical_admission() {
        let (mut runtime, settings, _, view) = fixture();
        let children = runtime
            .hierarchy
            .children(runtime.hierarchy.roots()[0])
            .to_vec();
        let left = runtime.hierarchy.children(children[0]).to_vec();
        let right = runtime.hierarchy.children(children[1]).to_vec();
        let ids = [LodRuntimeViewId(7), LodRuntimeViewId(8)];
        let current = vec![(ids[0], children.clone()), (ids[1], children.clone())];
        set_destination(
            &mut runtime,
            ids[0],
            view,
            &settings,
            vec![left[0], left[1], children[1]],
        );
        set_destination(
            &mut runtime,
            ids[1],
            view,
            &settings,
            vec![children[0], right[0], right[1]],
        );
        let retained = retain_nodes(&runtime, &current);
        runtime.cache = LodPageCache::new(PageCacheLimits {
            max_pages: retained.len() as u32 + 2,
            ..runtime.cache.limits()
        })
        .unwrap();
        for expected in [ids[0], ids[1], ids[0]] {
            let plan = runtime
                .unconstrained_wave(
                    &[(ids[0], view), (ids[1], view)],
                    &current,
                    &retained,
                    &settings,
                )
                .unwrap();
            let changed = plan
                .views
                .iter()
                .filter(|target| target.frontier.nodes != children)
                .map(|target| target.view)
                .collect::<Vec<_>>();
            assert_eq!(changed, vec![expected]);
            for target in plan.views {
                assert_complete(&runtime, &target.frontier.nodes);
            }
        }
    }

    #[test]
    fn discrete_lowered_active_budget_coarsens_through_multiple_over_limit_cuts() {
        let (mut runtime, mut settings, streaming, view) = fixture();
        let roots = runtime.hierarchy.roots().to_vec();
        let mut nodes = roots.clone();
        for _ in 0..3 {
            nodes = nodes
                .iter()
                .flat_map(|node| runtime.hierarchy.children(*node))
                .copied()
                .collect();
        }
        let initial_count = runtime.discrete_active_count(&nodes).unwrap();
        settings.quality = 0.0;
        settings.budgets.max_active_gaussians = runtime.discrete_active_count(&roots).unwrap();
        assert!(initial_count > settings.budgets.max_active_gaussians);
        let views = [(LodRuntimeViewId(7), view)];
        let mut current = vec![(views[0].0, nodes)];
        let mut counts = vec![initial_count];
        // Independent page completion may publish only one sibling group at
        // a time; bound transactions rather than assuming one wave per depth.
        for _ in 0..current[0].1.len() * 2 {
            if current[0].1 == roots {
                break;
            }
            let retained = retain_nodes(&runtime, &current);
            let plan = runtime
                .unconstrained_wave(&views, &current, &retained, &settings)
                .unwrap();
            let mut ready = None;
            for _ in 0..128 {
                advance(&mut runtime, &plan, &settings, &streaming, view);
                ready = runtime
                    .package_discrete_resident_plan(&plan, &views, &current, &settings)
                    .unwrap();
                if ready.is_some() {
                    break;
                }
            }
            current[0].1 = ready
                .expect("decreasing complete cut must remain admissible above the new limit")
                .views[0]
                .frontier
                .nodes
                .clone();
            assert_complete(&runtime, &current[0].1);
            counts.push(runtime.discrete_active_count(&current[0].1).unwrap());
        }
        assert_eq!(current[0].1, roots);
        assert!(counts.len() >= 3);
        assert!(counts.windows(2).all(|pair| pair[1] < pair[0]));
        assert!(counts[1] > settings.budgets.max_active_gaussians);
    }
    #[test]
    fn discrete_full_old_cut_recovers_then_refines_visible_detail_and_settles_under_residency_budget()
     {
        let (mut runtime, mut settings, streaming, _) = fixture_with_page_limit(9);
        settings.frustum_culling = true;
        let roots = runtime.hierarchy.roots().to_vec();
        let mut old = roots.clone();
        for _ in 0..3 {
            old = old
                .iter()
                .flat_map(|node| runtime.hierarchy.children(*node))
                .copied()
                .collect();
        }
        let hot_leaf = runtime
            .hierarchy
            .manifest()
            .nodes
            .iter()
            .find(|node| node.is_leaf())
            .unwrap();
        let center = (bevy::math::Vec3::from_array(hot_leaf.bounds.min)
            + bevy::math::Vec3::from_array(hot_leaf.bounds.max))
            * 0.5;
        let oriented_view = |target: bevy::math::Vec3| {
            let position = target + bevy::math::Vec3::new(0.0, 0.0, 0.5);
            let view_from_world =
                bevy::math::Mat4::look_at_rh(position, target, bevy::math::Vec3::Y);
            let projection = bevy::math::Mat4::perspective_infinite_reverse_rh(1.0, 1.0, 0.01);
            LodView::perspective(position, 540.0, 1.0, 0.01)
                .with_view_projection(projection * view_from_world, bevy::math::Vec2::splat(540.0))
        };
        let view = oriented_view(center);
        assert!(view.frustum.is_some());
        let views = [(LodRuntimeViewId(7), view)];
        let mut initial = runtime
            .package_all_resident_target_plan(&views, &settings)
            .unwrap();
        initial.views[0].frontier.nodes = old.clone();
        initial.views[0].frontier.status.active_gaussians =
            runtime.discrete_active_count(&old).unwrap();
        initial = runtime.discrete_plan_from_views(initial.views).unwrap();
        for _ in 0..256 {
            advance(&mut runtime, &initial, &settings, &streaming, view);
            if initial
                .pages
                .iter()
                .all(|page| runtime.decoded_page(*page).is_some())
            {
                break;
            }
        }
        let mut held = initial.pages.clone();
        for &page in &held {
            runtime.retain_resident_page(page).unwrap();
        }
        assert_eq!(
            runtime.cache.stats().pinned_pages,
            9,
            "inherited cut and permanent root occupy every slot"
        );
        let mut current = vec![(views[0].0, old)];
        let recovery = runtime
            .package_discrete_target_plan(&views, &current, &held, &settings)
            .unwrap();
        assert!(recovery.discrete_direct_transaction);
        let ready = runtime
            .package_discrete_resident_plan(&recovery, &views, &current, &settings)
            .unwrap()
            .unwrap();
        assert_eq!(ready.views[0].frontier.nodes, roots);
        let mut publications = 0;
        let mut deepest = 0;
        let mut next_ready = Some(ready);
        for _ in 0..32 {
            let ready = next_ready.take().unwrap();
            for &page in ready.pages.difference(&held) {
                runtime.retain_resident_page(page).unwrap();
            }
            for &page in held.difference(&ready.pages) {
                runtime.release_resident_page(page).unwrap();
            }
            held = ready.pages.clone();
            current[0].1 = ready.views[0].frontier.nodes.clone();
            assert_complete(&runtime, &current[0].1);
            publications += 1;
            deepest = deepest.max(
                current[0]
                    .1
                    .iter()
                    .map(|node| runtime.hierarchy.node(*node).unwrap().depth)
                    .max()
                    .unwrap(),
            );
            let candidates = runtime
                .package_target_candidates(&ready, &views, &settings)
                .unwrap()
                .unwrap();
            if candidates.matches_live_target {
                assert!(
                    publications >= 3,
                    "recovery must be followed by useful refinement"
                );
                assert!(
                    deepest > 3,
                    "visible branch should refine beyond the inherited breadth-first cut"
                );
                assert_eq!(
                    candidates.views[0].1.quality_status().degradation,
                    LodDegradation::Residency
                );
                assert!(
                    held.len() + 1 + 2 <= runtime.cache.limits().max_pages as usize,
                    "settled cut plus root plus two-page replacement reserve fits"
                );
                let traversals = runtime.all_resident_selection_traversals;
                for _ in 0..4 {
                    let settled = runtime
                        .package_discrete_target_plan(&views, &current, &held, &settings)
                        .unwrap();
                    assert_eq!(settled.views[0].frontier.nodes, current[0].1);
                    assert!(!settled.discrete_direct_transaction);
                }
                assert_eq!(runtime.all_resident_selection_traversals, traversals);
                // Moving to another octant invalidates the physical target,
                // even though the constrained old cut remains drawable.
                let moved = oriented_view(-center);
                let moved_views = [(views[0].0, moved)];
                let moved_plan = runtime
                    .package_discrete_destination_plan(&moved_views, &settings)
                    .unwrap();
                assert_ne!(moved_plan.views[0].frontier.nodes, current[0].1);
                return;
            }
            let plan = runtime
                .package_discrete_target_plan(&views, &current, &held, &settings)
                .unwrap();
            for _ in 0..256 {
                advance(&mut runtime, &plan, &settings, &streaming, view);
                next_ready = runtime
                    .package_discrete_resident_plan(&plan, &views, &current, &settings)
                    .unwrap();
                if next_ready.is_some() {
                    break;
                }
            }
            assert!(
                next_ready.is_some(),
                "one concrete reserved cohort must continue to make progress"
            );
        }
        panic!("constrained destination failed to settle");
    }
    #[test]
    fn discrete_packed_leaf_destination_bypasses_unreachable_intermediate_path_in_four_slots() {
        let (original, mut settings, streaming, view) = fixture();
        let mut manifest = original.hierarchy.manifest().clone();
        let mut pages = original.transport.pages.clone();
        let shared_id = LodPageId(manifest.pages.iter().map(|page| page.id.0).max().unwrap() + 1);
        let mut records = Vec::new();
        let mut old_leaf_pages = BTreeSet::new();
        let mut leaf_count = 0;
        for node in manifest.nodes.iter_mut().filter(|node| node.is_leaf()) {
            let page = crate::io::lod::decode_page(
                &pages[&node.representation.page],
                LodCodecLimits::default(),
            )
            .unwrap();
            old_leaf_pages.insert(node.representation.page);
            node.representation.page = shared_id;
            node.representation.offset = records.len() as u32;
            records.extend(page.gaussians);
            leaf_count += 1;
        }
        assert!(leaf_count >= 16);
        let shared = PlanarGaussian3dPage::new(shared_id, records);
        let encoded = encode_page(&shared).unwrap();
        manifest
            .pages
            .retain(|page| !old_leaf_pages.contains(&page.id));
        manifest.pages.push(LodPageDescriptor {
            id: shared_id,
            kind: LodPageKind::SourceLeaves,
            encoding: crate::gaussian::formats::planar_3d_chunked::LodPageEncoding::F32Planar,
            gaussian_count: shared.gaussians.len() as u32,
            decoded_len: shared.gaussians.len() as u64
                * std::mem::size_of::<crate::Gaussian3d>() as u64,
            content_hash: shared.content_hash(),
            bounds: manifest.scene_bounds.unwrap(),
            storage: Some(LodPageStorage {
                uri: "shared-leaves".into(),
                byte_range: None,
                encoded_len: encoded.len() as u64,
            }),
        });
        pages.retain(|page, _| !old_leaf_pages.contains(page));
        pages.insert(shared_id, encoded);
        manifest.header.page_count = manifest.pages.len() as u32;
        manifest.header.required_features |= LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES;
        manifest.validate().unwrap();
        settings.budgets.max_resident_pages = 4;
        let mut runtime = LodStreamingRuntime::new(
            manifest,
            DelayedPages {
                pages,
                blocked: BTreeSet::new(),
                started: BTreeSet::new(),
                canceled: BTreeSet::new(),
            },
            &settings,
            &streaming,
        )
        .unwrap();
        let mut coarse = settings.clone();
        coarse.quality = 0.0;
        for _ in 0..64 {
            runtime.update(view, &coarse, &streaming).unwrap();
            if runtime.hierarchy.roots().iter().all(|node| {
                runtime
                    .decoded_page(runtime.hierarchy.page(*node).unwrap())
                    .is_some()
            }) {
                break;
            }
        }
        let roots = runtime.hierarchy.roots().to_vec();
        let views = [(LodRuntimeViewId(7), view)];
        let mut current = vec![(views[0].0, roots.clone())];
        let mut held = BTreeSet::new();
        let mut saw_direct = false;
        for wave in 0..8 {
            let plan = runtime
                .package_discrete_target_plan(&views, &current, &held, &settings)
                .unwrap();
            saw_direct |= plan.discrete_direct_transaction;
            if wave > 0 {
                assert_ne!(
                    plan.views[0].frontier.nodes, roots,
                    "shared packing must not cause repeated root recovery"
                );
            }
            let mut ready = None;
            for _ in 0..256 {
                advance(&mut runtime, &plan, &settings, &streaming, view);
                ready = runtime
                    .package_discrete_resident_plan(&plan, &views, &current, &settings)
                    .unwrap();
                if ready.is_some() {
                    break;
                }
            }
            let ready = ready.expect("admitted exact destination union must complete");
            for &page in ready.pages.difference(&held) {
                runtime.retain_resident_page(page).unwrap();
            }
            for &page in held.difference(&ready.pages) {
                runtime.release_resident_page(page).unwrap();
            }
            held = ready.pages.clone();
            current[0].1 = ready.views[0].frontier.nodes.clone();
            assert_complete(&runtime, &current[0].1);
            assert!(runtime.cache.stats().resident_pages <= 4);
            if current[0]
                .1
                .iter()
                .all(|node| runtime.hierarchy.children(*node).is_empty())
            {
                assert!(saw_direct);
                assert_eq!(held, BTreeSet::from([shared_id]));
                let candidates = runtime
                    .package_target_candidates(&ready, &views, &settings)
                    .unwrap()
                    .unwrap();
                assert!(candidates.matches_live_target);
                assert_eq!(
                    candidates.views[0].1.candidate_count(),
                    shared.gaussians.len() as u32
                );
                return;
            }
        }
        panic!("packed complete destination did not converge");
    }
    #[test]
    fn discrete_two_homogeneous_pages_need_no_extra_navigation_reserve() {
        let (_, mut settings, streaming, view) = fixture();
        let cloud = LodTestScene::nested_octants(1).cloud();
        let built = build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                branching_factor: 2,
                leaf_capacity: 8,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        let mut manifest = built.manifest;
        assert_eq!(manifest.nodes.len(), 3);
        // The format requires shared pages to have one depth and leaf kind.
        // This two-level tree therefore occupies one representative page and
        // one source page; every possible navigation cut fits their union.
        let mut grouped = BTreeMap::<LodPageId, Vec<crate::Gaussian3d>>::new();
        for node in &mut manifest.nodes {
            let source = built
                .pages
                .iter()
                .find(|page| page.id == node.representation.page)
                .unwrap();
            let first = node.representation.offset as usize;
            let count = node.representation.count as usize;
            let id = LodPageId(if node.is_leaf() { 2 } else { 1 });
            let records = grouped.entry(id).or_default();
            node.representation.page = id;
            node.representation.offset = records.len() as u32;
            records.extend_from_slice(&source.gaussians[first..first + count]);
        }
        let source_count = manifest.header.source_gaussian_count;
        let mut pages = BTreeMap::new();
        manifest.pages.clear();
        let mut decoded_len = 0;
        let mut stored_count = 0;
        for (id, records) in grouped {
            let shared = PlanarGaussian3dPage::new(id, records);
            let encoded = encode_page(&shared).unwrap();
            let page_decoded_len =
                shared.gaussians.len() as u64 * std::mem::size_of::<crate::Gaussian3d>() as u64;
            decoded_len += page_decoded_len;
            stored_count += shared.gaussians.len() as u64;
            manifest.pages.push(LodPageDescriptor {
                id,
                kind: if id == LodPageId(2) {
                    LodPageKind::SourceLeaves
                } else {
                    LodPageKind::Representatives
                },
                encoding: crate::gaussian::formats::planar_3d_chunked::LodPageEncoding::F32Planar,
                gaussian_count: shared.gaussians.len() as u32,
                decoded_len: page_decoded_len,
                content_hash: shared.content_hash(),
                bounds: manifest.scene_bounds.unwrap(),
                storage: Some(LodPageStorage {
                    uri: format!("shared-hierarchy-{}", id.0),
                    byte_range: None,
                    encoded_len: encoded.len() as u64,
                }),
            });
            pages.insert(id, encoded);
        }
        manifest.header.page_count = 2;
        manifest.header.required_features |= LOD_REQUIRED_FEATURE_SHARED_NODE_PAGES;
        manifest.validate().unwrap();
        settings.budgets.max_resident_pages = 2;
        settings.budgets.max_resident_bytes = decoded_len;
        settings.budgets.max_resident_gaussians = stored_count;
        let mut runtime = LodStreamingRuntime::new(
            manifest,
            DelayedPages {
                pages,
                blocked: BTreeSet::new(),
                started: BTreeSet::new(),
                canceled: BTreeSet::new(),
            },
            &settings,
            &streaming,
        )
        .unwrap();
        let envelope = BTreeSet::from([LodPageId(1), LodPageId(2)]);
        for _ in 0..128 {
            runtime.update(view, &settings, &streaming).unwrap();
            if envelope
                .iter()
                .all(|page| runtime.decoded_page(*page).is_some())
            {
                break;
            }
        }
        assert!(
            envelope
                .iter()
                .all(|page| runtime.decoded_page(*page).is_some())
        );
        let views = [(LodRuntimeViewId(7), view)];
        let current = vec![(views[0].0, runtime.hierarchy.roots().to_vec())];
        let plan = runtime
            .package_discrete_target_plan(&views, &current, &envelope, &settings)
            .unwrap();
        assert!(plan.pages.is_subset(&envelope));
        assert!(!plan.discrete_direct_transaction);
        let ready = runtime
            .package_discrete_resident_plan(&plan, &views, &current, &settings)
            .unwrap()
            .unwrap();
        assert_complete(&runtime, &ready.views[0].frontier.nodes);
        let candidates = runtime
            .package_target_candidates(&ready, &views, &settings)
            .unwrap()
            .unwrap();
        assert_eq!(runtime.cache.stats().resident_pages, 2);
        assert!(candidates.matches_live_target);
        assert_eq!(
            u64::from(candidates.views[0].1.candidate_count()),
            source_count,
            "resident original detail must not be suppressed by a fictitious extra page reserve"
        );
        assert_eq!(
            candidates.views[0].1.quality_status().degradation,
            LodDegradation::None
        );
    }
}
