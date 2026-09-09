//! Physical destination budgeting uses complete sibling collapses, never
//! visibility holes. The reserve is the measured largest immediate cohort on
//! the ideal cut's ancestor paths, not a percentage of the cache.

use std::{cmp::Reverse, collections::BinaryHeap};

use super::*;

pub(crate) struct DiscreteDestinationCache {
    keys: Vec<(LodRuntimeViewId, AllResidentSelectionKey)>,
    limits: PageCacheLimits,
    pub(super) plan: LodPackageTargetPlan,
}

#[derive(Default)]
struct PageUnion {
    references: BTreeMap<LodPageId, usize>,
    bytes: u64,
    gaussians: u64,
}

impl PageUnion {
    fn add(
        &mut self,
        hierarchy: &CompiledManifestLodHierarchy,
        page: LodPageId,
    ) -> Result<(), LodRuntimeError> {
        let count = self.references.entry(page).or_default();
        if *count == 0 {
            let descriptor = hierarchy
                .page_descriptor(page)
                .ok_or(LodRuntimeError::MissingPageDescriptor(page))?;
            self.bytes = self
                .bytes
                .checked_add(descriptor.decoded_len)
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
            self.gaussians = self
                .gaussians
                .checked_add(u64::from(descriptor.gaussian_count))
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
        }
        *count += 1;
        Ok(())
    }

    fn remove(
        &mut self,
        hierarchy: &CompiledManifestLodHierarchy,
        page: LodPageId,
    ) -> Result<(), LodRuntimeError> {
        let count = self
            .references
            .get_mut(&page)
            .ok_or(LodRuntimeError::MissingPageDescriptor(page))?;
        *count -= 1;
        if *count == 0 {
            self.references.remove(&page);
            let descriptor = hierarchy
                .page_descriptor(page)
                .ok_or(LodRuntimeError::MissingPageDescriptor(page))?;
            self.bytes -= descriptor.decoded_len;
            self.gaussians -= u64::from(descriptor.gaussian_count);
        }
        Ok(())
    }
}

fn priority_for_metrics(
    metrics: LodNodeMetrics,
    bounds: crate::gaussian::formats::planar_3d_chunked::LodBounds,
    leaf: bool,
    view: LodView,
    settings: &GaussianLodSettings,
) -> (bool, u32, u32, Reverse<u64>) {
    let evaluator = crate::stream::hierarchy::LodViewEvaluator::from_view(view);
    let visible =
        !settings.frustum_culling || evaluator.bounds_are_visible(bounds, settings.frustum_margin);
    if !visible {
        return (false, 0, 0, Reverse(u64::MAX));
    }
    // This final tie-break schedules nearer visible regions when conservative
    // near-plane bounds saturate. It changes no error bound or quality test.
    // f64 arithmetic avoids overflow across the validated f32 transform range.
    let center = view
        .world_from_local
        .as_dmat4()
        .transform_point3(metrics.center.as_dvec3());
    let distance_squared = view.camera_position.as_dvec3().distance_squared(center);
    let (error_px, pressure) =
        evaluator.projected_quality(metrics, settings.quality_target(), leaf);
    (
        true,
        pressure.max(0.0).to_bits(),
        error_px.max(0.0).to_bits(),
        Reverse(distance_squared.to_bits()),
    )
}

impl<T: LodPageTransport> LodStreamingRuntime<T> {
    pub(crate) fn package_discrete_destination_plan(
        &mut self,
        views: &[(LodRuntimeViewId, LodView)],
        settings: &GaussianLodSettings,
    ) -> Result<LodPackageTargetPlan, LodRuntimeError> {
        #[cfg(feature = "testing")]
        let _cpu_scope = crate::testing::lod_package_cpu::scope(
            crate::testing::lod_package_cpu::PackageCpuScope::DestinationPlan,
        );

        settings
            .validate()
            .map_err(|error| LodRuntimeError::InvalidSettings(error.to_string()))?;
        let mut keys = Vec::with_capacity(views.len());
        for &(id, live) in views {
            let view = self
                .views
                .entry(id)
                .or_default()
                .selection_view(live, settings.selection_mode);
            view.validate().map_err(|error| match error {
                LodSelectionError::InvalidView(field) => {
                    LodRuntimeError::Selection(LodSelectionError::InvalidView(field))
                }
                _ => unreachable!("LodView::validate only emits InvalidView"),
            })?;
            keys.push((
                id,
                AllResidentSelectionKey {
                    view,
                    policy: LodHysteresisPolicy::from(settings),
                    selection_view_frozen: settings.selection_mode == LodSelectionMode::Frozen,
                },
            ));
        }
        keys.sort_unstable_by_key(|(id, _)| *id);
        let limits = self.cache.limits();
        if let Some(cached) = self.package_discrete_destination_cache.as_ref()
            && cached.keys == keys
            && cached.limits == limits
        {
            #[cfg(feature = "testing")]
            crate::testing::lod_package_cpu::destination_observed(true);
            return Ok(cached.plan.clone());
        }
        #[cfg(feature = "testing")]
        crate::testing::lod_package_cpu::destination_observed(false);
        let mut ideal = self.package_all_resident_target_plan(views, settings)?;
        ideal.views.sort_unstable_by_key(|target| target.view);
        let plan = self.discrete_budget_destination(ideal, views, settings)?;
        self.package_discrete_destination_cache = Some(DiscreteDestinationCache {
            keys,
            limits,
            plan: plan.clone(),
        });
        Ok(plan)
    }

    pub(super) fn discrete_priority(
        &self,
        node: LodNodeId,
        view: LodView,
        settings: &GaussianLodSettings,
    ) -> (bool, u32, u32, Reverse<u64>) {
        priority_for_metrics(
            self.hierarchy.metrics(node).expect("compiled node"),
            self.hierarchy.node(node).expect("compiled node").bounds,
            self.hierarchy.children(node).is_empty(),
            view,
            settings,
        )
    }

    pub(super) fn discrete_budget_destination(
        &self,
        mut ideal: LodPackageTargetPlan,
        views: &[(LodRuntimeViewId, LodView)],
        settings: &GaussianLodSettings,
    ) -> Result<LodPackageTargetPlan, LodRuntimeError> {
        #[cfg(feature = "testing")]
        let _cpu_scope = crate::testing::lod_package_cpu::scope(
            crate::testing::lod_package_cpu::PackageCpuScope::DestinationCompile,
        );

        let mut cuts: Vec<BTreeSet<LodNodeId>> = ideal
            .views
            .iter()
            .map(|target| target.frontier.nodes.iter().copied().collect())
            .collect();
        let selection_views = ideal
            .views
            .iter()
            .map(|target| {
                self.views
                    .get(&target.view)
                    .and_then(|state| state.frozen_selection_view)
                    .filter(|_| target.selection_view_frozen)
                    .unwrap_or_else(|| {
                        views
                            .iter()
                            .find(|(id, _)| *id == target.view)
                            .expect("target view")
                            .1
                    })
            })
            .collect::<Vec<_>>();
        let page = |node| {
            self.hierarchy
                .page(node)
                .ok_or(LodRuntimeError::MissingNode(node))
        };
        let mut union = PageUnion::default();
        for &root in self.hierarchy.roots() {
            union.add(&self.hierarchy, page(root)?)?;
        }
        for cut in &cuts {
            for &node in cut {
                union.add(&self.hierarchy, page(node)?)?;
            }
        }

        // Every path is deduplicated before walking further. Work is linear in
        // the ideal selector's visited topology, independent of source records.
        let mut ancestors = BTreeSet::new();
        for cut in &cuts {
            for &node in cut {
                let mut cursor = self.hierarchy.parent(node);
                while let Some(parent) = cursor {
                    if !ancestors.insert(parent) {
                        break;
                    }
                    cursor = self.hierarchy.parent(parent);
                }
            }
        }
        // Shared packing can place every intermediate and final logical node
        // in the same physical allocation. If the complete navigation envelope
        // fits, every adjacent path is safe without any additional reserve.
        // Reuse the visited ancestry; never scan the whole manifest here.
        let mut navigation_pages = union.references.keys().copied().collect::<BTreeSet<_>>();
        for &ancestor in &ancestors {
            navigation_pages.insert(page(ancestor)?);
        }
        let (navigation_bytes, navigation_gaussians, _) =
            LodRuntimeCoverageGuard::page_footprint(&self.hierarchy, &navigation_pages)?;
        let limits = self.cache.limits();
        if navigation_pages.len() <= limits.max_pages as usize
            && navigation_bytes <= limits.max_bytes
            && navigation_gaussians <= limits.max_gaussians
        {
            return Ok(ideal);
        }

        drop(navigation_pages);
        let mut reserve_pages = 0;
        let mut reserve_bytes = 0;
        let mut reserve_gaussians = 0;
        for &parent in &ancestors {
            let children = self
                .hierarchy
                .children(parent)
                .iter()
                .map(|&child| page(child))
                .collect::<Result<BTreeSet<_>, _>>()?;
            for cohort in [&children, &BTreeSet::from([page(parent)?])] {
                let (bytes, gaussians, _) =
                    LodRuntimeCoverageGuard::page_footprint(&self.hierarchy, cohort)?;
                reserve_pages = reserve_pages.max(cohort.len());
                reserve_bytes = reserve_bytes.max(bytes);
                reserve_gaussians = reserve_gaussians.max(gaussians);
            }
        }
        let limits = self.cache.limits();
        let fits = |union: &PageUnion| {
            union
                .references
                .len()
                .checked_add(reserve_pages)
                .is_some_and(|pages| pages <= limits.max_pages as usize)
                && union
                    .bytes
                    .checked_add(reserve_bytes)
                    .is_some_and(|bytes| bytes <= limits.max_bytes)
                && union
                    .gaussians
                    .checked_add(reserve_gaussians)
                    .is_some_and(|gaussians| gaussians <= limits.max_gaussians)
        };
        if fits(&union) {
            return Ok(ideal);
        }

        // A parent enters the heap only when all immediate children are in
        // this view's complete cut. Collapsing it can enable its own parent.
        let mut candidates = BinaryHeap::new();
        for (index, cut) in cuts.iter().enumerate() {
            let parents = cut
                .iter()
                .filter_map(|node| self.hierarchy.parent(*node))
                .collect::<BTreeSet<_>>();
            for parent in parents {
                if self
                    .hierarchy
                    .children(parent)
                    .iter()
                    .all(|child| cut.contains(child))
                {
                    candidates.push(Reverse((
                        self.discrete_priority(parent, selection_views[index], settings),
                        Reverse(index),
                        parent,
                    )));
                }
            }
        }
        while !fits(&union) {
            let Some(Reverse((_, Reverse(index), parent))) = candidates.pop() else {
                break;
            };
            let children = self.hierarchy.children(parent);
            if !children.iter().all(|child| cuts[index].contains(child)) {
                continue;
            }
            union.add(&self.hierarchy, page(parent)?)?;
            for &child in children {
                cuts[index].remove(&child);
                union.remove(&self.hierarchy, page(child)?)?;
            }
            cuts[index].insert(parent);
            if let Some(parent) = self.hierarchy.parent(parent)
                && self
                    .hierarchy
                    .children(parent)
                    .iter()
                    .all(|child| cuts[index].contains(child))
            {
                candidates.push(Reverse((
                    self.discrete_priority(parent, selection_views[index], settings),
                    Reverse(index),
                    parent,
                )));
            }
        }
        for (index, target) in ideal.views.iter_mut().enumerate() {
            let nodes = cuts[index].iter().copied().collect::<Vec<_>>();
            if nodes != target.frontier.nodes {
                let step = crate::stream::hierarchy::LodTemporalFrontierStep {
                    nodes,
                    substitutions: Vec::new(),
                    requested_nodes: Vec::new(),
                    changed_gaussians: 0,
                    atomic_budget_overshoot: 0,
                    reached_target: false,
                };
                *target = self.discrete_plan_view(
                    target.clone(),
                    &step,
                    selection_views[index],
                    settings,
                )?;
                target.frontier.status.degradation = target
                    .frontier
                    .status
                    .degradation
                    .merge(LodDegradation::Residency);
            }
        }
        self.discrete_plan_from_views(ideal.views)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saturated_visible_bounds_prioritize_camera_near_detail_before_node_identity() {
        let metrics = |x| LodNodeMetrics {
            center: bevy::math::Vec3::new(x, 0.0, -5.0),
            radius: 100.0,
            geometric_error: 1.0,
            appearance_error: 0.0,
            opacity_error: 0.0,
            quality_min: 0.0,
            quality_max: 1.0,
            high_fidelity_certificate: 0.0,
            representative_count: 1,
        };
        let settings = GaussianLodSettings {
            quality: 1.0,
            frustum_culling: true,
            ..Default::default()
        };
        for (x, left_is_nearer) in [(-10.0, true), (10.0, false)] {
            let position = bevy::math::Vec3::new(x, 0.0, 0.0);
            let view_from_world = bevy::math::Mat4::look_at_rh(
                position,
                position - bevy::math::Vec3::Z,
                bevy::math::Vec3::Y,
            );
            let projection = bevy::math::Mat4::perspective_infinite_reverse_rh(1.0, 1.0, 0.01);
            let view = LodView::perspective(position, 540.0, 1.0, 0.01)
                .with_view_projection(projection * view_from_world, bevy::math::Vec2::splat(540.0));
            let bounds = |x| crate::gaussian::formats::planar_3d_chunked::LodBounds {
                min: (metrics(x).center - bevy::math::Vec3::splat(100.0)).to_array(),
                max: (metrics(x).center + bevy::math::Vec3::splat(100.0)).to_array(),
            };
            let left = priority_for_metrics(metrics(-10.0), bounds(-10.0), false, view, &settings);
            let right = priority_for_metrics(metrics(10.0), bounds(10.0), false, view, &settings);
            assert_eq!((left.0, left.1, left.2), (right.0, right.1, right.2));
            assert!(left.0);
            assert_eq!(left.2, f32::MAX.to_bits());
            assert_eq!(left > right, left_is_nearer);
        }
    }
}
