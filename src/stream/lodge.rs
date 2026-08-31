//! Pure CPU planning for LODGE external active-set packages.
//!
//! This module deliberately owns no transport, cache, atlas, or render state.
//! It turns authenticated, sorted memberships into one deterministic classified
//! union, then guards that union with a retained/pending publication state. The
//! package layer remains responsible for proving that every required page and
//! atlas generation is drawable before committing a pending union.

use std::{collections::BTreeSet, error::Error, fmt, sync::Arc};

use crate::gaussian::formats::{
    lodge::{LodgeCameraCluster, LodgeClusterId, LodgeGaussianId, LodgeRecordRun},
    planar_3d_chunked::LodPageId,
};

/// Stable, direction-independent identity for the two active sets in one
/// camera-conditioned LODGE presentation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LodgePairIdentity {
    pub first: LodgeClusterId,
    pub second: LodgeClusterId,
}

impl LodgePairIdentity {
    fn canonical(first: LodgeClusterId, second: LodgeClusterId) -> Self {
        if first <= second {
            Self { first, second }
        } else {
            Self {
                first: second,
                second: first,
            }
        }
    }
}

/// Deterministic two-nearest selection and the weight of the canonical second
/// cluster. The pair remains content-stable when the nearest cluster changes
/// inside the same two-cluster region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodgePairSelection {
    pub identity: LodgePairIdentity,
    pub nearest: LodgeClusterId,
    pub second_weight: f32,
}

/// How one catalog Gaussian participates in the selected pair.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LodgeMembershipClass {
    Shared = 0,
    FirstOnly = 1,
    SecondOnly = 2,
}

impl LodgeMembershipClass {
    /// Opacity multiplier for this class at the selected pair weight.
    pub fn opacity_weight(self, second_weight: f32) -> f32 {
        let second_weight = if second_weight.is_finite() {
            second_weight.clamp(0.0, 1.0)
        } else {
            0.0
        };
        match self {
            Self::Shared => 1.0,
            Self::FirstOnly => 1.0 - second_weight,
            Self::SecondOnly => second_weight,
        }
    }
}

/// Authenticated membership IDs after codec-level bounds and ordering checks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LodgeMembership {
    cluster: LodgeClusterId,
    ids: Arc<[LodgeGaussianId]>,
}

impl LodgeMembership {
    /// Retains a decoded membership only if its IDs are strictly increasing.
    pub fn new(cluster: LodgeClusterId, ids: Vec<LodgeGaussianId>) -> Result<Self, LodgePlanError> {
        if !cluster.is_valid() {
            return Err(LodgePlanError::InvalidCluster);
        }
        if ids.is_empty() {
            return Err(LodgePlanError::EmptyMembership);
        }
        if ids.iter().any(|id| !id.is_valid()) || ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(LodgePlanError::MembershipNotStrictlySorted);
        }
        Ok(Self {
            cluster,
            ids: ids.into(),
        })
    }

    pub fn cluster(&self) -> LodgeClusterId {
        self.cluster
    }

    pub fn ids(&self) -> &[LodgeGaussianId] {
        &self.ids
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// One stable Gaussian ID and its pair-membership class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodgeClassifiedGaussian {
    pub id: LodgeGaussianId,
    pub class: LodgeMembershipClass,
}

/// One contiguous page-local run with a uniform pair-membership class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodgeClassifiedPageRun {
    /// First stable catalog ID represented by this run. A canonical resident
    /// catalog maps this directly to source index `id - 1` without inventing
    /// page or atlas identity.
    pub first_id: LodgeGaussianId,
    pub page: LodPageId,
    pub page_offset: u32,
    pub count: u32,
    pub class: LodgeMembershipClass,
}

/// Validated, allocation-free stable-ID to page-local record resolver.
#[derive(Clone, Copy, Debug)]
pub struct LodgeRecordLocationResolver<'a> {
    runs: &'a [LodgeRecordRun],
}

impl<'a> LodgeRecordLocationResolver<'a> {
    /// Validates the manifest-global dense run index once. Runs may change page
    /// or page-local offset arbitrarily, but stable IDs must cover one dense,
    /// strictly ascending interval without overlap.
    pub fn new(runs: &'a [LodgeRecordRun]) -> Result<Self, LodgePlanError> {
        let mut expected_first = None;
        for run in runs {
            if !run.first_id.is_valid() || run.count == 0 || !run.page.is_valid() {
                return Err(LodgePlanError::InvalidRecordRun);
            }
            if expected_first.is_some_and(|expected| run.first_id.0 != expected) {
                return Err(LodgePlanError::InvalidRecordRun);
            }
            let end = run
                .first_id
                .0
                .checked_add(u64::from(run.count))
                .ok_or(LodgePlanError::CountOverflow)?;
            run.page_offset
                .checked_add(run.count)
                .ok_or(LodgePlanError::CountOverflow)?;
            expected_first = Some(end);
        }
        Ok(Self { runs })
    }

    /// Constructs a resolver for a manifest which has already crossed
    /// [`GaussianLodgeManifest::validate`](crate::GaussianLodgeManifest::validate).
    /// This avoids rescanning a potentially large run table on every camera
    /// update in the resident integration.
    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) const fn from_validated(runs: &'a [LodgeRecordRun]) -> Self {
        Self { runs }
    }

    pub fn resolve(&self, id: LodgeGaussianId) -> Option<(LodPageId, u32)> {
        let upper = self.runs.partition_point(|run| run.first_id.0 <= id.0);
        let run = self.runs.get(upper.checked_sub(1)?)?;
        let relative = id.0.checked_sub(run.first_id.0)?;
        if relative >= u64::from(run.count) {
            return None;
        }
        let relative = u32::try_from(relative).ok()?;
        Some((run.page, run.page_offset.checked_add(relative)?))
    }
}

/// Hard limits applied before a classified union can become a package
/// candidate. These are independent so no zero value means "unbounded".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodgePairLimits {
    pub max_union_gaussians: u64,
    pub max_classified_runs: u32,
    pub max_required_pages: u32,
}

impl LodgePairLimits {
    fn validate(self) -> Result<Self, LodgePlanError> {
        if self.max_union_gaussians == 0
            || self.max_classified_runs == 0
            || self.max_required_pages == 0
        {
            Err(LodgePlanError::ZeroLimit)
        } else {
            Ok(self)
        }
    }
}

/// Public, renderer-facing facts for one fully classified pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodgePairCounts {
    pub shared: u64,
    pub first_only: u64,
    pub second_only: u64,
    pub union: u64,
    pub required_pages: u32,
}

/// Coefficients consumed by an active-set-aware raster path. Shared records are
/// emitted exactly once at full authored opacity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodgePairOpacityWeights {
    pub shared: f32,
    pub first_only: f32,
    pub second_only: f32,
}

/// Constant-size public observation for one complete pair candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodgePairStatus {
    pub identity: LodgePairIdentity,
    pub nearest: LodgeClusterId,
    pub second_weight: f32,
    pub counts: LodgePairCounts,
}

/// Complete immutable union for one selected pair. Camera-only weight changes
/// preserve `runs` and `required_pages` and therefore need no restaging.
#[derive(Clone, Debug, PartialEq)]
pub struct LodgePairCandidate {
    selection: LodgePairSelection,
    runs: Arc<[LodgeClassifiedPageRun]>,
    required_pages: Arc<[LodPageId]>,
    counts: LodgePairCounts,
}

impl LodgePairCandidate {
    pub fn selection(&self) -> LodgePairSelection {
        self.selection
    }

    pub fn identity(&self) -> LodgePairIdentity {
        self.selection.identity
    }

    pub fn second_weight(&self) -> f32 {
        self.selection.second_weight
    }

    pub fn runs(&self) -> &[LodgeClassifiedPageRun] {
        &self.runs
    }

    pub fn required_pages(&self) -> &[LodPageId] {
        &self.required_pages
    }

    pub fn counts(&self) -> LodgePairCounts {
        self.counts
    }

    pub fn opacity_weights(&self) -> LodgePairOpacityWeights {
        LodgePairOpacityWeights {
            shared: 1.0,
            first_only: 1.0 - self.selection.second_weight,
            second_only: self.selection.second_weight,
        }
    }

    pub fn status(&self) -> LodgePairStatus {
        LodgePairStatus {
            identity: self.identity(),
            nearest: self.selection.nearest,
            second_weight: self.selection.second_weight,
            counts: self.counts,
        }
    }

    /// Re-evaluates camera telemetry for the same immutable pair without
    /// rebuilding its classified union or page demand.
    pub fn retarget(&self, selection: LodgePairSelection) -> Result<Self, LodgePlanError> {
        if selection.identity != self.identity()
            || !matches!(
                selection.nearest,
                nearest if nearest == selection.identity.first
                    || nearest == selection.identity.second
            )
            || !selection.second_weight.is_finite()
            || !(0.0..=1.0).contains(&selection.second_weight)
        {
            return Err(LodgePlanError::PairMembershipMismatch);
        }
        Ok(Self {
            selection,
            runs: Arc::clone(&self.runs),
            required_pages: Arc::clone(&self.required_pages),
            counts: self.counts,
        })
    }

    /// True when two candidates share the exact GPU range/class payload and
    /// differ at most in camera-conditioned weight/nearest-cluster telemetry.
    pub fn same_union(&self, other: &Self) -> bool {
        self.identity() == other.identity()
            && self.runs == other.runs
            && self.required_pages == other.required_pages
            && self.counts == other.counts
    }
}

/// Observable retained/pending lifecycle without exposing a partial candidate
/// as drawable state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodgePairPublicationPhase {
    Empty,
    PendingWithoutRetained,
    Retained,
    PendingWithRetained,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodgePairStageResult {
    Staged,
    RetargetedRetained,
    RetargetedPending,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodgePairCommitResult {
    NoPending,
    Waiting { missing_pages: u32 },
    Published,
}

/// Retains the last complete pair while a replacement is incomplete.
#[derive(Clone, Debug, Default)]
pub struct LodgePairPublicationState {
    retained: Option<Arc<LodgePairCandidate>>,
    pending: Option<Arc<LodgePairCandidate>>,
}

impl LodgePairPublicationState {
    pub fn phase(&self) -> LodgePairPublicationPhase {
        match (self.retained.is_some(), self.pending.is_some()) {
            (false, false) => LodgePairPublicationPhase::Empty,
            (false, true) => LodgePairPublicationPhase::PendingWithoutRetained,
            (true, false) => LodgePairPublicationPhase::Retained,
            (true, true) => LodgePairPublicationPhase::PendingWithRetained,
        }
    }

    /// The only candidate this CPU lifecycle authorizes as drawable.
    pub fn retained(&self) -> Option<&Arc<LodgePairCandidate>> {
        self.retained.as_ref()
    }

    /// Complete replacement content which still lacks a package-owned drawable
    /// residency proof.
    pub fn pending(&self) -> Option<&Arc<LodgePairCandidate>> {
        self.pending.as_ref()
    }

    pub fn stage(&mut self, candidate: Arc<LodgePairCandidate>) -> LodgePairStageResult {
        if self
            .retained
            .as_ref()
            .is_some_and(|retained| retained.same_union(&candidate))
        {
            self.retained = Some(candidate);
            self.pending = None;
            return LodgePairStageResult::RetargetedRetained;
        }
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.same_union(&candidate))
        {
            self.pending = Some(candidate);
            return LodgePairStageResult::RetargetedPending;
        }
        self.pending = Some(candidate);
        LodgePairStageResult::Staged
    }

    /// Atomically promotes a pending pair only when every page is generation-
    /// current and drawable according to the package-provided predicate.
    pub fn commit_pending_if_drawable(
        &mut self,
        mut page_is_drawable: impl FnMut(LodPageId) -> bool,
    ) -> LodgePairCommitResult {
        let Some(pending) = self.pending.as_ref() else {
            return LodgePairCommitResult::NoPending;
        };
        let missing_pages = pending
            .required_pages()
            .iter()
            .filter(|&&page| !page_is_drawable(page))
            .count()
            .try_into()
            .unwrap_or(u32::MAX);
        if missing_pages != 0 {
            return LodgePairCommitResult::Waiting { missing_pages };
        }
        self.retained = self.pending.take();
        LodgePairCommitResult::Published
    }

    pub fn discard_pending(&mut self) {
        self.pending = None;
    }
}

/// Deterministically selects the two nearest clusters. Distance ties use the
/// stable cluster ID, while the returned pair identity is always canonical by
/// ID so crossing the pair midpoint does not rebuild an identical union.
pub fn select_lodge_pair(
    view_position: [f32; 3],
    clusters: &[LodgeCameraCluster],
) -> Result<LodgePairSelection, LodgePlanError> {
    select_lodge_pair_impl(view_position, clusters, true)
}

/// Allocation-free pair selection for a sidecar whose cluster table has
/// already crossed semantic validation.
#[cfg_attr(not(lod_render_path), allow(dead_code))]
pub(crate) fn select_lodge_pair_from_validated_clusters(
    view_position: [f32; 3],
    clusters: &[LodgeCameraCluster],
) -> Result<LodgePairSelection, LodgePlanError> {
    select_lodge_pair_impl(view_position, clusters, false)
}

fn select_lodge_pair_impl(
    view_position: [f32; 3],
    clusters: &[LodgeCameraCluster],
    validate_cluster_ids: bool,
) -> Result<LodgePairSelection, LodgePlanError> {
    if view_position.iter().any(|value| !value.is_finite()) {
        return Err(LodgePlanError::NonFiniteView);
    }
    if clusters.len() < 2 {
        return Err(LodgePlanError::InsufficientClusters);
    }
    let mut seen = validate_cluster_ids.then(BTreeSet::new);
    let mut nearest = None;
    let mut second = None;
    for (index, cluster) in clusters.iter().enumerate() {
        if !cluster.id.is_valid()
            || cluster.center.iter().any(|value| !value.is_finite())
            || seen.as_mut().is_some_and(|seen| !seen.insert(cluster.id))
        {
            return Err(LodgePlanError::InvalidCluster);
        }
        let distance = squared_distance(view_position, cluster.center);
        let candidate = (distance, cluster.id, index);
        if nearest.is_none_or(|current| lodge_cluster_rank_before(candidate, current)) {
            second = nearest;
            nearest = Some(candidate);
        } else if second.is_none_or(|current| lodge_cluster_rank_before(candidate, current)) {
            second = Some(candidate);
        }
    }
    let nearest = &clusters[nearest.expect("two clusters were checked").2];
    let other = &clusters[second.expect("two clusters were checked").2];
    let identity = LodgePairIdentity::canonical(nearest.id, other.id);
    let (first_center, second_center) = if nearest.id == identity.first {
        (nearest.center, other.center)
    } else {
        (other.center, nearest.center)
    };
    Ok(LodgePairSelection {
        identity,
        nearest: nearest.id,
        second_weight: projected_center_line_weight(view_position, first_center, second_center)?,
    })
}

fn lodge_cluster_rank_before(
    left: (f64, LodgeClusterId, usize),
    right: (f64, LodgeClusterId, usize),
) -> bool {
    left.0
        .total_cmp(&right.0)
        .then_with(|| left.1.cmp(&right.1))
        .is_lt()
}

/// Projects the camera center onto the canonical pair center line and clamps
/// the result to exact f32 endpoints.
pub fn projected_center_line_weight(
    view_position: [f32; 3],
    first_center: [f32; 3],
    second_center: [f32; 3],
) -> Result<f32, LodgePlanError> {
    if view_position
        .iter()
        .chain(first_center.iter())
        .chain(second_center.iter())
        .any(|value| !value.is_finite())
    {
        return Err(LodgePlanError::NonFiniteView);
    }
    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    for axis in 0..3 {
        let direction = f64::from(second_center[axis]) - f64::from(first_center[axis]);
        numerator += (f64::from(view_position[axis]) - f64::from(first_center[axis])) * direction;
        denominator += direction * direction;
    }
    if denominator == 0.0 || !denominator.is_finite() || !numerator.is_finite() {
        return Err(LodgePlanError::CoincidentClusterCenters);
    }
    let weight = numerator / denominator;
    if weight <= 0.0 {
        Ok(0.0)
    } else if weight >= 1.0 {
        Ok(1.0)
    } else {
        Ok(weight as f32)
    }
}

/// Linear-time merge of two validated sorted memberships.
pub fn classify_lodge_membership_union(
    first: &LodgeMembership,
    second: &LodgeMembership,
    max_union_gaussians: u64,
) -> Result<Vec<LodgeClassifiedGaussian>, LodgePlanError> {
    if max_union_gaussians == 0 {
        return Err(LodgePlanError::ZeroLimit);
    }
    let first = first.ids();
    let second = second.ids();
    let maximum = first
        .len()
        .checked_add(second.len())
        .ok_or(LodgePlanError::CountOverflow)?;
    let bounded_capacity = usize::try_from(max_union_gaussians)
        .unwrap_or(usize::MAX)
        .min(maximum);
    let mut union = Vec::with_capacity(bounded_capacity);
    let (mut left, mut right) = (0, 0);
    while left < first.len() || right < second.len() {
        let classified = match (first.get(left), second.get(right)) {
            (Some(&first_id), Some(&second_id)) if first_id == second_id => {
                left += 1;
                right += 1;
                LodgeClassifiedGaussian {
                    id: first_id,
                    class: LodgeMembershipClass::Shared,
                }
            }
            (Some(&first_id), Some(&second_id)) if first_id < second_id => {
                left += 1;
                LodgeClassifiedGaussian {
                    id: first_id,
                    class: LodgeMembershipClass::FirstOnly,
                }
            }
            (Some(_), Some(&second_id)) => {
                right += 1;
                LodgeClassifiedGaussian {
                    id: second_id,
                    class: LodgeMembershipClass::SecondOnly,
                }
            }
            (Some(&first_id), None) => {
                left += 1;
                LodgeClassifiedGaussian {
                    id: first_id,
                    class: LodgeMembershipClass::FirstOnly,
                }
            }
            (None, Some(&second_id)) => {
                right += 1;
                LodgeClassifiedGaussian {
                    id: second_id,
                    class: LodgeMembershipClass::SecondOnly,
                }
            }
            (None, None) => break,
        };
        union.push(classified);
        if union.len() as u64 > max_union_gaussians {
            return Err(LodgePlanError::UnionLimitExceeded {
                limit: max_union_gaussians,
            });
        }
    }
    Ok(union)
}

/// Maps a classified stable-ID union to page-local runs. Coalescing requires
/// the same page, class, and immediately adjacent local index.
pub fn coalesce_lodge_classified_runs(
    classified: &[LodgeClassifiedGaussian],
    locations: &LodgeRecordLocationResolver<'_>,
    max_runs: u32,
) -> Result<Vec<LodgeClassifiedPageRun>, LodgePlanError> {
    if max_runs == 0 {
        return Err(LodgePlanError::ZeroLimit);
    }
    if classified.windows(2).any(|pair| pair[0].id >= pair[1].id) {
        return Err(LodgePlanError::MembershipNotStrictlySorted);
    }
    let mut runs: Vec<LodgeClassifiedPageRun> = Vec::new();
    let mut location_run = 0_usize;
    for record in classified {
        while let Some(run) = locations.runs.get(location_run)
            && run.stable_end().is_some_and(|end| record.id.0 >= end)
        {
            location_run += 1;
        }
        let location = locations
            .runs
            .get(location_run)
            .and_then(|run| run.locate(record.id))
            .ok_or(LodgePlanError::MissingRecordLocation(record.id))?;
        let (page, page_offset) = (location.page, location.offset);
        let coalesced = runs.last_mut().is_some_and(|run| {
            if run.page != page || run.class != record.class {
                return false;
            }
            if run.first_id.0.checked_add(u64::from(run.count)) != Some(record.id.0) {
                return false;
            }
            if run.page_offset.checked_add(run.count) != Some(page_offset) {
                return false;
            }
            let Some(count) = run.count.checked_add(1) else {
                return false;
            };
            run.count = count;
            true
        });
        if !coalesced {
            runs.push(LodgeClassifiedPageRun {
                first_id: record.id,
                page,
                page_offset,
                count: 1,
                class: record.class,
            });
            if runs.len() as u64 > u64::from(max_runs) {
                return Err(LodgePlanError::RangeLimitExceeded { limit: max_runs });
            }
        }
    }
    Ok(runs)
}

/// Builds a fully classified pair candidate without consulting transport or
/// residency. Publication remains the responsibility of
/// [`LodgePairPublicationState`].
pub fn build_lodge_pair_candidate(
    selection: LodgePairSelection,
    first: &LodgeMembership,
    second: &LodgeMembership,
    locations: &LodgeRecordLocationResolver<'_>,
    limits: LodgePairLimits,
) -> Result<LodgePairCandidate, LodgePlanError> {
    let limits = limits.validate()?;
    if selection.identity.first >= selection.identity.second
        || !selection.identity.first.is_valid()
        || !selection.identity.second.is_valid()
        || !matches!(
            selection.nearest,
            nearest if nearest == selection.identity.first || nearest == selection.identity.second
        )
        || !selection.second_weight.is_finite()
        || !(0.0..=1.0).contains(&selection.second_weight)
        || first.cluster() != selection.identity.first
        || second.cluster() != selection.identity.second
    {
        return Err(LodgePlanError::PairMembershipMismatch);
    }
    let classified = classify_lodge_membership_union(first, second, limits.max_union_gaussians)?;
    let runs = coalesce_lodge_classified_runs(&classified, locations, limits.max_classified_runs)?;
    let required_pages = runs.iter().map(|run| run.page).collect::<BTreeSet<_>>();
    if required_pages.len() as u64 > u64::from(limits.max_required_pages) {
        return Err(LodgePlanError::PageLimitExceeded {
            limit: limits.max_required_pages,
        });
    }
    let mut counts = LodgePairCounts {
        shared: 0,
        first_only: 0,
        second_only: 0,
        union: classified.len() as u64,
        required_pages: required_pages.len().try_into().unwrap_or(u32::MAX),
    };
    for record in &classified {
        let count = match record.class {
            LodgeMembershipClass::Shared => &mut counts.shared,
            LodgeMembershipClass::FirstOnly => &mut counts.first_only,
            LodgeMembershipClass::SecondOnly => &mut counts.second_only,
        };
        *count = count.checked_add(1).ok_or(LodgePlanError::CountOverflow)?;
    }
    Ok(LodgePairCandidate {
        selection,
        runs: runs.into(),
        required_pages: required_pages.into_iter().collect::<Vec<_>>().into(),
        counts,
    })
}

/// Union of required pages across private camera candidates. Shared catalog
/// pages are charged once. The caller supplies the package-wide residency bound.
pub fn lodge_multi_view_page_demand<'a>(
    candidates: impl IntoIterator<Item = &'a LodgePairCandidate>,
    max_pages: u32,
) -> Result<Vec<LodPageId>, LodgePlanError> {
    if max_pages == 0 {
        return Err(LodgePlanError::ZeroLimit);
    }
    let mut pages = BTreeSet::new();
    for candidate in candidates {
        for &page in candidate.required_pages() {
            pages.insert(page);
            if pages.len() as u64 > u64::from(max_pages) {
                return Err(LodgePlanError::PageLimitExceeded { limit: max_pages });
            }
        }
    }
    Ok(pages.into_iter().collect())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LodgePlanError {
    NonFiniteView,
    InsufficientClusters,
    InvalidCluster,
    EmptyMembership,
    PairMembershipMismatch,
    CoincidentClusterCenters,
    MembershipNotStrictlySorted,
    InvalidRecordRun,
    MissingRecordLocation(LodgeGaussianId),
    ZeroLimit,
    CountOverflow,
    UnionLimitExceeded { limit: u64 },
    RangeLimitExceeded { limit: u32 },
    PageLimitExceeded { limit: u32 },
}

impl fmt::Display for LodgePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteView => write!(formatter, "LODGE view/cluster center must be finite"),
            Self::InsufficientClusters => {
                write!(
                    formatter,
                    "LODGE pair selection requires at least two clusters"
                )
            }
            Self::InvalidCluster => {
                write!(formatter, "LODGE cluster IDs must be valid and unique")
            }
            Self::EmptyMembership => write!(formatter, "LODGE memberships must be nonempty"),
            Self::PairMembershipMismatch => write!(
                formatter,
                "LODGE pair memberships do not match the canonical selected clusters"
            ),
            Self::CoincidentClusterCenters => {
                write!(formatter, "LODGE pair cluster centers must be distinct")
            }
            Self::MembershipNotStrictlySorted => {
                write!(
                    formatter,
                    "LODGE membership IDs must be strictly increasing"
                )
            }
            Self::InvalidRecordRun => {
                write!(formatter, "LODGE record runs are not dense and valid")
            }
            Self::MissingRecordLocation(id) => {
                write!(formatter, "LODGE Gaussian {} has no page location", id.0)
            }
            Self::ZeroLimit => write!(formatter, "LODGE planning limits must be non-zero"),
            Self::CountOverflow => write!(formatter, "LODGE planning count overflow"),
            Self::UnionLimitExceeded { limit } => {
                write!(formatter, "LODGE union exceeds Gaussian limit {limit}")
            }
            Self::RangeLimitExceeded { limit } => {
                write!(
                    formatter,
                    "LODGE union exceeds classified-run limit {limit}"
                )
            }
            Self::PageLimitExceeded { limit } => {
                write!(formatter, "LODGE union exceeds required-page limit {limit}")
            }
        }
    }
}

impl Error for LodgePlanError {}

fn squared_distance(left: [f32; 3], right: [f32; 3]) -> f64 {
    (0..3)
        .map(|axis| {
            let delta = f64::from(left[axis]) - f64::from(right[axis]);
            delta * delta
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::formats::planar_3d_chunked::LodIndexRange;

    fn cluster(id: u32, center: [f32; 3]) -> LodgeCameraCluster {
        LodgeCameraCluster {
            id: LodgeClusterId(id),
            center,
            radius: 1.0,
            neighbors: LodIndexRange { start: 0, count: 0 },
            membership_entry: id - 1,
        }
    }

    fn membership(cluster: u32, ids: &[u64]) -> LodgeMembership {
        LodgeMembership::new(
            LodgeClusterId(cluster),
            ids.iter().copied().map(LodgeGaussianId).collect(),
        )
        .unwrap()
    }

    fn resolver(runs: &[LodgeRecordRun]) -> LodgeRecordLocationResolver<'_> {
        LodgeRecordLocationResolver::new(runs).unwrap()
    }

    fn limits() -> LodgePairLimits {
        LodgePairLimits {
            max_union_gaussians: 64,
            max_classified_runs: 64,
            max_required_pages: 8,
        }
    }

    #[test]
    fn two_nearest_selection_is_tie_stable_and_pair_canonical() {
        let clusters = [
            cluster(9, [2.0, 0.0, 0.0]),
            cluster(5, [0.0, 0.0, 0.0]),
            cluster(7, [4.0, 0.0, 0.0]),
        ];
        let selection = select_lodge_pair([1.0, 0.0, 0.0], &clusters).unwrap();
        assert_eq!(selection.nearest, LodgeClusterId(5));
        assert_eq!(selection.identity.first, LodgeClusterId(5));
        assert_eq!(selection.identity.second, LodgeClusterId(9));
        assert_eq!(selection.second_weight.to_bits(), 0.5_f32.to_bits());

        let crossed = select_lodge_pair([1.75, 0.0, 0.0], &clusters).unwrap();
        assert_eq!(crossed.nearest, LodgeClusterId(9));
        assert_eq!(crossed.identity, selection.identity);
        assert_eq!(crossed.second_weight, 0.875);
        assert_eq!(
            select_lodge_pair_from_validated_clusters([1.75, 0.0, 0.0], &clusters).unwrap(),
            crossed
        );
    }

    #[test]
    fn projected_center_line_weight_has_exact_endpoints_and_rejects_degenerate_pairs() {
        assert_eq!(
            projected_center_line_weight([-2.0, 0.0, 0.0], [0.0; 3], [2.0, 0.0, 0.0])
                .unwrap()
                .to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(
            projected_center_line_weight([3.0, 0.0, 0.0], [0.0; 3], [2.0, 0.0, 0.0])
                .unwrap()
                .to_bits(),
            1.0_f32.to_bits()
        );
        assert_eq!(
            projected_center_line_weight([1.0, 4.0, 0.0], [0.0; 3], [2.0, 0.0, 0.0]).unwrap(),
            0.5
        );
        assert_eq!(
            projected_center_line_weight([0.0; 3], [1.0; 3], [1.0; 3]),
            Err(LodgePlanError::CoincidentClusterCenters)
        );
    }

    #[test]
    fn membership_union_is_sorted_deduplicated_and_classified() {
        let union = classify_lodge_membership_union(
            &membership(1, &[1, 2, 4, 7]),
            &membership(2, &[2, 3, 4, 8]),
            8,
        )
        .unwrap();
        assert_eq!(
            union,
            vec![
                LodgeClassifiedGaussian {
                    id: LodgeGaussianId(1),
                    class: LodgeMembershipClass::FirstOnly
                },
                LodgeClassifiedGaussian {
                    id: LodgeGaussianId(2),
                    class: LodgeMembershipClass::Shared
                },
                LodgeClassifiedGaussian {
                    id: LodgeGaussianId(3),
                    class: LodgeMembershipClass::SecondOnly
                },
                LodgeClassifiedGaussian {
                    id: LodgeGaussianId(4),
                    class: LodgeMembershipClass::Shared
                },
                LodgeClassifiedGaussian {
                    id: LodgeGaussianId(7),
                    class: LodgeMembershipClass::FirstOnly
                },
                LodgeClassifiedGaussian {
                    id: LodgeGaussianId(8),
                    class: LodgeMembershipClass::SecondOnly
                },
            ]
        );
        assert_eq!(
            LodgeMembership::new(
                LodgeClusterId(1),
                vec![LodgeGaussianId(2), LodgeGaussianId(2)],
            ),
            Err(LodgePlanError::MembershipNotStrictlySorted)
        );
    }

    #[test]
    fn empty_membership_is_rejected_before_candidate_construction() {
        assert_eq!(
            LodgeMembership::new(LodgeClusterId(1), Vec::new()),
            Err(LodgePlanError::EmptyMembership)
        );
    }

    #[test]
    fn classified_runs_never_cross_class_or_page_boundaries() {
        let runs = [
            LodgeRecordRun {
                first_id: LodgeGaussianId(1),
                count: 4,
                page: LodPageId(10),
                page_offset: 3,
            },
            LodgeRecordRun {
                first_id: LodgeGaussianId(5),
                count: 4,
                page: LodPageId(11),
                page_offset: 0,
            },
        ];
        let classified = [
            LodgeClassifiedGaussian {
                id: LodgeGaussianId(1),
                class: LodgeMembershipClass::FirstOnly,
            },
            LodgeClassifiedGaussian {
                id: LodgeGaussianId(2),
                class: LodgeMembershipClass::FirstOnly,
            },
            LodgeClassifiedGaussian {
                id: LodgeGaussianId(3),
                class: LodgeMembershipClass::Shared,
            },
            LodgeClassifiedGaussian {
                id: LodgeGaussianId(4),
                class: LodgeMembershipClass::Shared,
            },
            LodgeClassifiedGaussian {
                id: LodgeGaussianId(5),
                class: LodgeMembershipClass::Shared,
            },
            LodgeClassifiedGaussian {
                id: LodgeGaussianId(6),
                class: LodgeMembershipClass::SecondOnly,
            },
        ];
        let coalesced = coalesce_lodge_classified_runs(&classified, &resolver(&runs), 8).unwrap();
        assert_eq!(
            coalesced,
            vec![
                LodgeClassifiedPageRun {
                    first_id: LodgeGaussianId(1),
                    page: LodPageId(10),
                    page_offset: 3,
                    count: 2,
                    class: LodgeMembershipClass::FirstOnly
                },
                LodgeClassifiedPageRun {
                    first_id: LodgeGaussianId(3),
                    page: LodPageId(10),
                    page_offset: 5,
                    count: 2,
                    class: LodgeMembershipClass::Shared
                },
                LodgeClassifiedPageRun {
                    first_id: LodgeGaussianId(5),
                    page: LodPageId(11),
                    page_offset: 0,
                    count: 1,
                    class: LodgeMembershipClass::Shared
                },
                LodgeClassifiedPageRun {
                    first_id: LodgeGaussianId(6),
                    page: LodPageId(11),
                    page_offset: 1,
                    count: 1,
                    class: LodgeMembershipClass::SecondOnly
                },
            ]
        );
    }

    fn candidate(
        selection: LodgePairSelection,
        first: &[u64],
        second: &[u64],
    ) -> LodgePairCandidate {
        let runs = [LodgeRecordRun {
            first_id: LodgeGaussianId(1),
            count: 8,
            page: LodPageId(10),
            page_offset: 0,
        }];
        build_lodge_pair_candidate(
            selection,
            &LodgeMembership::new(
                selection.identity.first,
                first.iter().copied().map(LodgeGaussianId).collect(),
            )
            .unwrap(),
            &LodgeMembership::new(
                selection.identity.second,
                second.iter().copied().map(LodgeGaussianId).collect(),
            )
            .unwrap(),
            &resolver(&runs),
            limits(),
        )
        .unwrap()
    }

    #[test]
    fn retained_pair_is_never_replaced_by_partial_residency() {
        let clusters = [cluster(1, [0.0; 3]), cluster(2, [2.0, 0.0, 0.0])];
        let first = Arc::new(candidate(
            select_lodge_pair([0.5, 0.0, 0.0], &clusters).unwrap(),
            &[1, 2],
            &[2, 3],
        ));
        let mut publication = LodgePairPublicationState::default();
        assert_eq!(
            publication.stage(Arc::clone(&first)),
            LodgePairStageResult::Staged
        );
        assert_eq!(
            publication.phase(),
            LodgePairPublicationPhase::PendingWithoutRetained
        );
        assert_eq!(
            publication.commit_pending_if_drawable(|_| false),
            LodgePairCommitResult::Waiting { missing_pages: 1 }
        );
        assert!(publication.retained().is_none());
        assert_eq!(
            publication.commit_pending_if_drawable(|page| page == LodPageId(10)),
            LodgePairCommitResult::Published
        );
        assert!(Arc::ptr_eq(publication.retained().unwrap(), &first));

        let replacement_clusters = [cluster(3, [0.0; 3]), cluster(4, [2.0, 0.0, 0.0])];
        let replacement = Arc::new(candidate(
            select_lodge_pair([0.5, 0.0, 0.0], &replacement_clusters).unwrap(),
            &[1, 2],
            &[2, 3],
        ));
        assert_eq!(publication.stage(replacement), LodgePairStageResult::Staged);
        assert_eq!(
            publication.phase(),
            LodgePairPublicationPhase::PendingWithRetained
        );
        assert_eq!(
            publication.commit_pending_if_drawable(|_| false),
            LodgePairCommitResult::Waiting { missing_pages: 1 }
        );
        assert!(Arc::ptr_eq(publication.retained().unwrap(), &first));
        publication.discard_pending();

        let retarget = Arc::new(candidate(
            select_lodge_pair([1.5, 0.0, 0.0], &clusters).unwrap(),
            &[1, 2],
            &[2, 3],
        ));
        assert_eq!(
            publication.stage(Arc::clone(&retarget)),
            LodgePairStageResult::RetargetedRetained
        );
        assert_eq!(publication.phase(), LodgePairPublicationPhase::Retained);
        assert_eq!(publication.retained().unwrap().second_weight(), 0.75);
    }

    #[test]
    fn stable_pair_retargets_without_rebuilding_its_union() {
        let clusters = [cluster(1, [0.0; 3]), cluster(2, [2.0, 0.0, 0.0])];
        let original = candidate(
            select_lodge_pair([0.25, 0.0, 0.0], &clusters).unwrap(),
            &[1, 2, 4],
            &[2, 3],
        );
        let moved = original
            .retarget(select_lodge_pair([1.5, 0.0, 0.0], &clusters).unwrap())
            .unwrap();
        assert_eq!(moved.second_weight(), 0.75);
        assert_eq!(moved.counts(), original.counts());
        assert!(Arc::ptr_eq(&moved.runs, &original.runs));
        assert!(Arc::ptr_eq(&moved.required_pages, &original.required_pages));

        let other = [cluster(3, [0.0; 3]), cluster(4, [2.0, 0.0, 0.0])];
        assert_eq!(
            original.retarget(select_lodge_pair([0.5, 0.0, 0.0], &other).unwrap()),
            Err(LodgePlanError::PairMembershipMismatch)
        );
    }

    #[test]
    fn multi_view_page_demand_deduplicates_shared_catalog_pages_and_is_bounded() {
        let first_clusters = [cluster(1, [0.0; 3]), cluster(2, [2.0, 0.0, 0.0])];
        let first = candidate(
            select_lodge_pair([0.5, 0.0, 0.0], &first_clusters).unwrap(),
            &[1],
            &[2],
        );
        let second_clusters = [cluster(3, [0.0; 3]), cluster(4, [2.0, 0.0, 0.0])];
        let second = candidate(
            select_lodge_pair([0.5, 0.0, 0.0], &second_clusters).unwrap(),
            &[2],
            &[3],
        );
        assert_eq!(
            lodge_multi_view_page_demand([&first, &second], 1).unwrap(),
            vec![LodPageId(10)]
        );
        assert_eq!(
            lodge_multi_view_page_demand([&first, &second], 0),
            Err(LodgePlanError::ZeroLimit)
        );
    }
}
