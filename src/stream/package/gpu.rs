//! GPU-owned selection over authenticated, fenced whole-page atlas snapshots.

use super::*;
use crate::render::{
    ordered::GaussianGlobalOrderSettings,
    point::GaussianPointSplattingSettings,
    spatial_morph::{GaussianLodSpatialTransitionSettings, GpuLodSpatialMorph},
    traversal::{
        GpuLodDrawAcknowledgements, GpuLodDrawRenderer, GpuLodHierarchy, GpuLodHierarchyTree,
        GpuLodTraversalFeedback, GpuLodTraversalFeedbacks, GpuLodTraversalSettings,
    },
};
use std::sync::{Weak, atomic::AtomicU64};

static NEXT_SNAPSHOT: AtomicU64 = AtomicU64::new(1);
const MAX_LIVE_SNAPSHOTS: usize = 4;

/// GPU feedback is selection evidence; only successful image completion from
/// the camera's selected renderer acknowledges a published residency generation.
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct GaussianGpuLodPackageStatus {
    pub residency_generation: u64,
    pub snapshot_pages: u32,
    /// Decoded demanded pages leased while staging or snapshot fences defer
    /// publication. These occupy existing bounded resident slots.
    pub pending_publication_pages: u32,
    /// Admitted shared authored mapping; individual views enforce their own cap.
    pub spatial_mapping_bytes: u64,
    /// Spatial mapping admission can fall back to a complete discrete image.
    pub spatial_mapping_error: Option<String>,
    pub visible_views: u32,
    pub acknowledged_views: u32,
    pub selected_gaussians: u64,
    pub queued_requests: u32,
    pub in_flight_requests: u32,
    pub capacity_blocked_requests: u32,
    pub record_limited: bool,
    pub frontier_limited: bool,
    pub visit_limited: bool,
    pub request_overflow: bool,
    pub cutoff_unavailable: bool,
}

struct SnapshotPins {
    lifetime: Weak<()>,
    pages: Vec<LodPageId>,
}

struct SnapshotPageReferences(HashMap<LodPageId, u32>);

impl SnapshotPageReferences {
    fn retain(
        &mut self,
        page: LodPageId,
        acquire: impl FnOnce() -> Result<(), LodRuntimeError>,
    ) -> Result<(), LodRuntimeError> {
        if let Some(references) = self.0.get_mut(&page) {
            *references = references
                .checked_add(1)
                .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
        } else {
            acquire()?;
            self.0.insert(page, 1);
        }
        Ok(())
    }

    fn release(
        &mut self,
        page: LodPageId,
        release: impl FnOnce() -> Result<(), LodRuntimeError>,
    ) -> Result<(), LodRuntimeError> {
        let references = self
            .0
            .get_mut(&page)
            .ok_or(LodRuntimeError::PhysicalIndexOverflow)?;
        if *references > 1 {
            *references -= 1;
        } else {
            release()?;
            self.0.remove(&page);
        }
        Ok(())
    }
}

struct SnapshotOwner {
    _lifetime: Arc<()>,
    _tree_memory: Arc<LodMemoryLease>,
    _snapshot_memory: LodMemoryLease,
}

/// Page membership changes independently of traversal submission counters.
/// Retain the normalized demand sets across held frames and identical cuts.
#[derive(Default)]
struct GpuPageDemands {
    keep: BTreeSet<LodPageId>,
    requested: BTreeSet<LodPageId>,
    /// First occurrence wins: roots, each view's GPU demand, then retained pages.
    order: Vec<LodPageId>,
    /// Rebuilt with demand membership; pin changes do not change missing data.
    missing: MissingPageFootprintCache,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MissingPageFootprint {
    pages: u64,
    bytes: u64,
    gaussians: u64,
}

#[derive(Default)]
struct MissingPageFootprintCache {
    observed: Option<(u64, MissingPageFootprint)>,
}

impl MissingPageFootprintCache {
    /// This cache belongs to one immutable normalized demand set. Explicit
    /// retries and lease retirement still run normally; only actual resident
    /// membership changes require measuring its missing footprint again.
    fn get_or_update(
        &mut self,
        revision: u64,
        requested: &BTreeSet<LodPageId>,
        mut missing: impl FnMut(LodPageId) -> Option<(u64, u64)>,
    ) -> Result<MissingPageFootprint, GaussianLodPackageError> {
        if let Some((observed, footprint)) = self.observed
            && observed == revision
        {
            return Ok(footprint);
        }
        let mut footprint = MissingPageFootprint::default();
        for &page in requested {
            if let Some((bytes, gaussians)) = missing(page) {
                footprint.pages = footprint
                    .pages
                    .checked_add(1)
                    .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
                footprint.bytes = footprint
                    .bytes
                    .checked_add(bytes)
                    .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
                footprint.gaussians = footprint
                    .gaussians
                    .checked_add(gaussians)
                    .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
            }
        }
        self.observed = Some((revision, footprint));
        Ok(footprint)
    }
}

/// Preserve the exact published-ancestor closure, canonicalizing just once.
/// The selected-page vector and at most the resident ancestor set bound this
/// scratch; the per-view feedback reservation covers both collections.
fn resident_navigation(
    selected: &[LodPageId],
    parents: &HashMap<LodPageId, BTreeSet<LodPageId>>,
    mut published: impl FnMut(LodPageId) -> bool,
) -> BTreeSet<LodPageId> {
    let mut pages = selected.to_vec();
    let mut seen = selected.iter().copied().collect::<HashSet<_>>();
    let mut cursor = 0;
    while cursor < pages.len() {
        let page = pages[cursor];
        cursor += 1;
        for &parent in parents.get(&page).into_iter().flatten() {
            if !seen.contains(&parent) && published(parent) {
                seen.insert(parent);
                pages.push(parent);
            }
        }
    }
    drop(seen);
    pages.sort_unstable();
    pages.dedup();
    pages.into_iter().collect()
}

impl GpuPageDemands {
    fn from_views(
        roots: &BTreeSet<LodPageId>,
        navigation: &BTreeMap<Entity, BTreeSet<LodPageId>>,
        feedback: &BTreeMap<Entity, GpuLodTraversalFeedback>,
    ) -> Result<Self, GaussianLodPackageError> {
        let mut keep = roots.clone();
        for pages in navigation.values() {
            keep.extend(pages.iter().copied());
        }
        let capacity = feedback
            .values()
            .try_fold(keep.len(), |count, feedback| {
                count.checked_add(feedback.requested_pages.len())
            })
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        let mut requested = roots.clone();
        let mut order = Vec::with_capacity(capacity);
        order.extend(roots.iter().copied());
        for feedback in feedback.values() {
            for &page in &feedback.requested_pages {
                if requested.insert(page) {
                    order.push(page);
                }
            }
        }
        for &page in &keep {
            if requested.insert(page) {
                order.push(page);
            }
        }
        Ok(Self {
            keep,
            requested,
            order,
            missing: MissingPageFootprintCache::default(),
        })
    }
}

struct SpatialMappingAdmission {
    limit: u64,
    mapping: Option<GpuLodSpatialMorph>,
    error: Option<String>,
    retry: Option<SpatialMappingCapacityRetry>,
}

struct SpatialMappingCapacityRetry {
    required: u64,
    observed_headroom: u64,
}

fn cpu_headroom(ledger: &LodMemoryLedger) -> u64 {
    let usage = ledger.snapshot();
    usage.limits.max_cpu_bytes.saturating_sub(usage.cpu_bytes)
}

impl SpatialMappingAdmission {
    fn should_retry(&self, limit: u64, ledger: &LodMemoryLedger) -> bool {
        self.limit != limit
            || self.retry.as_ref().is_some_and(|retry| {
                let available = cpu_headroom(ledger);
                available > retry.observed_headroom && available >= retry.required
            })
    }

    fn new(
        tree: Arc<GpuLodHierarchyTree>,
        manifest: &crate::GaussianLodManifest,
        limit: u64,
        ledger: &LodMemoryLedger,
    ) -> Self {
        let mut admission = Self {
            limit,
            mapping: None,
            error: None,
            retry: None,
        };
        let required = match GpuLodSpatialMorph::required_mapping_bytes(manifest) {
            Ok(required) if required <= limit => required,
            Ok(required) => {
                admission.error = Some(format!(
                    "spatial mapping requires {required} bytes, limit {limit}"
                ));
                return admission;
            }
            Err(error) => {
                admission.error = Some(error);
                return admission;
            }
        };
        let available = cpu_headroom(ledger);
        if available < required {
            admission.error = Some(format!(
                "spatial mapping requires {required} CPU bytes, {available} available"
            ));
            admission.retry = Some(SpatialMappingCapacityRetry {
                required,
                observed_headroom: available,
            });
            return admission;
        }
        match GpuLodSpatialMorph::for_validated_tree(tree, manifest, limit, ledger) {
            Ok(mapping) => admission.mapping = Some(mapping),
            Err(error) => {
                admission.error = Some(error);
                // Another owner can reserve between the cheap preflight and
                // atomic admission. Retry only after enough capacity returns.
                let available = cpu_headroom(ledger);
                if available < required {
                    admission.retry = Some(SpatialMappingCapacityRetry {
                        required,
                        observed_headroom: available,
                    });
                }
            }
        }
        admission
    }
}

pub(super) struct GpuPackageState {
    tree: Arc<GpuLodHierarchyTree>,
    page_indices: HashMap<LodPageId, usize>,
    parents_by_page: HashMap<LodPageId, BTreeSet<LodPageId>>,
    navigation: BTreeMap<Entity, BTreeSet<LodPageId>>,
    feedback_memory: BTreeMap<Entity, LodMemoryLease>,
    roots: BTreeSet<LodPageId>,
    root_pins: BTreeSet<LodPageId>,
    feedback: BTreeMap<Entity, GpuLodTraversalFeedback>,
    demands: GpuPageDemands,
    spatial_mapping: Option<SpatialMappingAdmission>,
    snapshot: Option<GpuLodHierarchy>,
    placements: BTreeMap<LodPageId, AtlasSlot>,
    pending_publication: BTreeSet<LodPageId>,
    pins: Vec<SnapshotPins>,
    // Overlapping immutable snapshots share one runtime eviction lease per
    // page. Publication/retirement updates this bounded reference count rather
    // than mutating the runtime's ordered cache and caller-pin trees repeatedly.
    snapshot_page_references: SnapshotPageReferences,
    excluded: BTreeSet<LodPageId>,
    observed_revision: u64,
    first_generation: Option<u64>,
    tree_memory: Arc<LodMemoryLease>,
    ledger: LodMemoryLedger,
    status: GaussianGpuLodPackageStatus,
}

impl GpuPackageState {
    fn new(
        state: &mut PackageInstantiation,
        ledger: &LodMemoryLedger,
    ) -> Result<Self, GaussianLodPackageError> {
        let mut runtime = state
            .runtime
            .lock()
            .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
        let manifest = runtime.hierarchy().manifest();
        // Typed topology vectors include duplicated roots and construction
        // scratch. Reserve package ancestry/index metadata separately, plus
        // bounded residency, per-view feedback and navigation sets.
        let topology_bytes = GpuLodHierarchyTree::compilation_bytes(
            manifest.nodes.len(),
            manifest.roots.len(),
            manifest.pages.len(),
        )
        .map_err(GaussianLodPackageError::InvalidManifest)?;
        let bytes = (manifest.nodes.len() as u64)
            .checked_mul(80)
            .and_then(|n| n.checked_add(topology_bytes))
            .and_then(|n| n.checked_add((manifest.pages.len() as u64).saturating_mul(192)))
            .and_then(|n| {
                n.checked_add(
                    u64::from(state.plan.slot_count)
                        .saturating_mul(
                            u64::from(state.config.max_views_per_cloud).saturating_add(2),
                        )
                        .saturating_mul(256),
                )
            })
            // Pending publication set plus bounded release/transfer scratch.
            .and_then(|n| n.checked_add(u64::from(state.plan.slot_count).saturating_mul(64)))
            // Snapshot reference table, including hash capacity/resize slack.
            .and_then(|n| n.checked_add(u64::from(state.plan.slot_count).saturating_mul(64)))
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        let tree_memory = Arc::new(
            ledger
                .try_reserve(LodMemoryCategory::MetadataCpu, bytes)
                .map_err(GaussianLodPackageError::MemoryBudget)?,
        );
        let tree = Arc::new(
            GpuLodHierarchyTree::from_validated_manifest(manifest)
                .map_err(GaussianLodPackageError::InvalidManifest)?,
        );
        let page_indices = tree
            .page_ids()
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i))
            .collect();
        let mut parents_by_page = HashMap::<LodPageId, BTreeSet<LodPageId>>::new();
        for node in &manifest.nodes {
            if let Some(parent_page) = node
                .parent
                .and_then(|parent| runtime.hierarchy().page(parent))
                && parent_page != node.representation.page
            {
                parents_by_page
                    .entry(node.representation.page)
                    .or_default()
                    .insert(parent_page);
            }
        }
        let roots = manifest
            .roots
            .iter()
            .filter_map(|&n| runtime.hierarchy().page(n))
            .collect::<BTreeSet<_>>();
        if roots.len() > state.plan.slot_count as usize {
            return Err(GaussianLodPackageError::RootFallbackExceedsAtlas {
                root_pages: roots.len() as u64,
                slots: state.plan.slot_count,
            });
        }
        runtime
            .initialize_gpu_page_demands()
            .map_err(GaussianLodPackageError::Runtime)?;
        Ok(Self {
            tree,
            page_indices,
            parents_by_page,
            navigation: BTreeMap::new(),
            feedback_memory: BTreeMap::new(),
            roots,
            root_pins: BTreeSet::new(),
            feedback: BTreeMap::new(),
            demands: GpuPageDemands::default(),
            spatial_mapping: None,
            snapshot: None,
            placements: BTreeMap::new(),
            pending_publication: BTreeSet::new(),
            pins: Vec::new(),
            snapshot_page_references: SnapshotPageReferences(HashMap::with_capacity(
                state.plan.slot_count as usize,
            )),
            excluded: BTreeSet::new(),
            observed_revision: u64::MAX,
            first_generation: None,
            tree_memory,
            ledger: ledger.clone(),
            status: GaussianGpuLodPackageStatus::default(),
        })
    }

    fn collect_retired(
        &mut self,
        runtime: &mut LodStreamingRuntime<PackagePageTransport>,
    ) -> Result<(), GaussianLodPackageError> {
        let mut i = 0;
        while i < self.pins.len() {
            if self.pins[i].lifetime.upgrade().is_some() {
                i += 1;
                continue;
            }
            let pins = self.pins.swap_remove(i);
            for page in pins.pages {
                self.release_snapshot_page(runtime, page)?;
            }
        }
        Ok(())
    }

    fn retain_snapshot_page(
        &mut self,
        runtime: &mut LodStreamingRuntime<PackagePageTransport>,
        page: LodPageId,
    ) -> Result<(), GaussianLodPackageError> {
        self.snapshot_page_references
            .retain(page, || runtime.retain_resident_page(page).map(|_| ()))
            .map_err(GaussianLodPackageError::Runtime)
    }

    fn release_snapshot_page(
        &mut self,
        runtime: &mut LodStreamingRuntime<PackagePageTransport>,
        page: LodPageId,
    ) -> Result<(), GaussianLodPackageError> {
        self.snapshot_page_references
            .release(page, || runtime.release_resident_page(page))
            .map_err(GaussianLodPackageError::Runtime)
    }

    fn release_pending_publication(
        &mut self,
        runtime: &mut LodStreamingRuntime<PackagePageTransport>,
    ) -> Result<(), GaussianLodPackageError> {
        while let Some(&page) = self.pending_publication.first() {
            runtime
                .release_resident_page(page)
                .map_err(GaussianLodPackageError::Runtime)?;
            self.pending_publication.remove(&page);
        }
        Ok(())
    }
}

pub(super) type GpuCameraQueryItem = (
    Option<&'static GaussianPointSplattingSettings>,
    Option<&'static GaussianGlobalOrderSettings>,
    &'static GpuLodTraversalSettings,
    &'static Msaa,
    Option<&'static GaussianLodSpatialTransitionSettings>,
);

fn camera_renderer(
    point: Option<&GaussianPointSplattingSettings>,
    ordered: Option<&GaussianGlobalOrderSettings>,
) -> Result<GpuLodDrawRenderer, &'static str> {
    match (point.is_some(), ordered.is_some()) {
        (true, false) => Ok(GpuLodDrawRenderer::GaussianPoints),
        (false, true) => Ok(GpuLodDrawRenderer::OrderedQuads),
        _ => Err(
            "GPU package cameras require exactly one of point splatting or global quad ordering",
        ),
    }
}

fn camera_spatial_mapping_limit(
    presentation: LodPresentationMode,
    ordered: bool,
    spatial: Option<&GaussianLodSpatialTransitionSettings>,
) -> Result<Option<u64>, &'static str> {
    if presentation != LodPresentationMode::ContinuousMorph || !ordered {
        return Ok(None);
    }
    let Some(spatial) = spatial else {
        return Ok(None);
    };
    spatial.validate()?;
    Ok(Some(spatial.max_mapping_bytes))
}

fn prepare_spatial_mapping(
    state: &PackageInstantiation,
    gpu: &mut GpuPackageState,
    presentation: LodPresentationMode,
    limit: Option<u64>,
    ledger: &LodMemoryLedger,
) -> Result<(), GaussianLodPackageError> {
    if presentation != LodPresentationMode::ContinuousMorph {
        gpu.spatial_mapping = None;
        return Ok(());
    }
    let Some(limit) = limit else {
        return Ok(());
    };
    if gpu
        .spatial_mapping
        .as_ref()
        .is_some_and(|old| !old.should_retry(limit, ledger))
    {
        return Ok(());
    }
    // Structural/map-cap failures remain cached. Temporary CPU admission
    // retries only when enough headroom returns, without periodic compilation.
    let retained = gpu
        .spatial_mapping
        .take()
        .and_then(|old| old.mapping)
        .filter(|mapping| mapping.mapping_bytes() <= limit);
    let admission = match retained {
        Some(mapping) => SpatialMappingAdmission {
            limit,
            mapping: Some(mapping),
            error: None,
            retry: None,
        },
        None => {
            let runtime = state
                .runtime
                .lock()
                .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
            SpatialMappingAdmission::new(
                gpu.tree.clone(),
                runtime.hierarchy().manifest(),
                limit,
                ledger,
            )
        }
    };
    gpu.spatial_mapping = Some(admission);
    Ok(())
}

// Page IDs refer to the immutable package, not transient atlas placements. A
// newer residency publication must not discard an in-flight loading request.
// Draw completion and selected-count reporting still require the exact current
// generation; accepting asynchronous demand never authorizes an old cut to draw.
fn accepts_page_feedback(
    feedback: &GpuLodTraversalFeedback,
    source: AssetId<PlanarGaussian3d>,
    first_generation: Option<u64>,
    current_generation: u64,
    previous: Option<&GpuLodTraversalFeedback>,
) -> bool {
    feedback.complete
        && feedback.source == source
        && first_generation
            .is_some_and(|first| (first..=current_generation).contains(&feedback.generation))
        && previous.is_none_or(|old| {
            (feedback.generation, feedback.submission) > (old.generation, old.submission)
        })
}

#[derive(bevy::ecs::system::SystemParam)]
pub(super) struct GpuPackageInputs<'w, 's> {
    pub cameras: Query<'w, 's, GpuCameraQueryItem>,
    pub feedbacks: Option<Res<'w, GpuLodTraversalFeedbacks>>,
    pub acknowledgements: Option<Res<'w, GpuLodDrawAcknowledgements>>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update(
    entity: Entity,
    state: &mut PackageInstantiation,
    settings: &GaussianLodSettings,
    cloud: &CloudSettings,
    views: &[PackageCameraView],
    cameras: &Query<GpuCameraQueryItem>,
    feedbacks: Option<&GpuLodTraversalFeedbacks>,
    acknowledgements: Option<&GpuLodDrawAcknowledgements>,
    ledger: &LodMemoryLedger,
    uploads: &mut LodAtlasUploadQueue,
    staging: &mut PackageStagingPermit<'_>,
    commands: &mut Commands,
) -> Result<(), GaussianLodPackageError> {
    if settings.selection_mode == LodSelectionMode::Frozen {
        return Err(GaussianLodPackageError::InvalidLodSettings(
            "GPU package traversal does not support Frozen selection; use Dynamic".into(),
        ));
    }
    if !matches!(
        settings.presentation_mode,
        LodPresentationMode::Discrete | LodPresentationMode::ContinuousMorph
    ) || cloud.gaussian_mode != GaussianMode::Gaussian3d
        || cloud.rasterize_mode != crate::RasterizeMode::Color
        || cloud.additive
        || cloud.lod_debug.requires_metadata()
        || cloud.visualize_bounding_box
        || !matches!(
            cloud.sort_mode,
            crate::sort::SortMode::Radix | crate::sort::SortMode::None
        )
    {
        return Err(GaussianLodPackageError::InvalidLodSettings(
            "GPU package traversal requires discrete or spatial-morph planar 3D Color, no debug/additive presentation, and Radix or None sort mode".into()));
    }
    let mut mapping_limit = None::<u64>;
    for view in views {
        let (point, ordered, traversal, msaa, spatial) =
            cameras.get(view.entity).map_err(|_| {
                GaussianLodPackageError::InvalidLodSettings(
                    "every visible GPU package camera requires GPU traversal and Msaa::Off".into(),
                )
            })?;
        camera_renderer(point, ordered)
            .map_err(|error| GaussianLodPackageError::InvalidLodSettings(error.into()))?;
        if let Some(limit) =
            camera_spatial_mapping_limit(settings.presentation_mode, ordered.is_some(), spatial)
                .map_err(|error| GaussianLodPackageError::InvalidLodSettings(error.into()))?
        {
            // One immutable map serves the largest eligible view. Rendering
            // independently falls back for cameras with smaller limits.
            mapping_limit = Some(mapping_limit.map_or(limit, |old| old.max(limit)));
        }
        if let Some(point) = point {
            point
                .validate()
                .map_err(|error| GaussianLodPackageError::InvalidLodSettings(error.to_string()))?;
        }
        if let Some(ordered) = ordered {
            ordered
                .validate()
                .map_err(|error| GaussianLodPackageError::InvalidLodSettings(error.into()))?;
        }
        traversal
            .validate()
            .map_err(GaussianLodPackageError::InvalidLodSettings)?;
        if *msaa != Msaa::Off {
            return Err(GaussianLodPackageError::InvalidLodSettings(
                "GPU package rendering requires Msaa::Off".into(),
            ));
        }
    }
    if state.gpu.is_none() {
        state.gpu = Some(GpuPackageState::new(state, ledger)?);
    }
    let mut gpu = state.gpu.take().expect("initialized GPU package");
    let result = prepare_spatial_mapping(
        state,
        &mut gpu,
        settings.presentation_mode,
        mapping_limit,
        ledger,
    )
    .and_then(|_| {
        drive(
            entity,
            state,
            &mut gpu,
            settings,
            views,
            cameras,
            feedbacks,
            acknowledgements,
            uploads,
            staging,
        )
    });
    if result.is_ok() {
        let mut owner = commands.entity(entity);
        owner.insert((LodRenderCandidates::package_required(), gpu.status.clone()));
        if let Some(mapping) = gpu
            .spatial_mapping
            .as_ref()
            .and_then(|admission| admission.mapping.as_ref())
        {
            owner.insert(mapping.clone());
        } else {
            owner.remove::<GpuLodSpatialMorph>();
        }
        if let Some(snapshot) = &gpu.snapshot {
            owner.insert(snapshot.clone());
        } else {
            owner.remove::<GpuLodHierarchy>();
        }
        let active = gpu.status.visible_views != 0
            && gpu.status.acknowledged_views == gpu.status.visible_views
            && views.iter().all(|view| {
                gpu.feedback.get(&view.entity).is_some_and(|feedback| {
                    feedback.generation == gpu.status.residency_generation && feedback.complete
                })
            });
        owner.insert(GaussianLodPackageStatus {
            phase: if active {
                GaussianLodPackagePhase::Active
            } else {
                GaussianLodPackagePhase::Loading
            },
            resident_pages: state.resident_pages,
            active_gaussians: if active {
                gpu.status.selected_gaussians
            } else {
                0
            },
            terminal_failures: state.terminal_failures,
            failure: None,
        });
    }
    state.gpu = Some(gpu);
    result
}

#[allow(clippy::too_many_arguments)]
fn drive(
    entity: Entity,
    state: &mut PackageInstantiation,
    gpu: &mut GpuPackageState,
    settings: &GaussianLodSettings,
    views: &[PackageCameraView],
    cameras: &Query<GpuCameraQueryItem>,
    feedbacks: Option<&GpuLodTraversalFeedbacks>,
    acknowledgements: Option<&GpuLodDrawAcknowledgements>,
    uploads: &mut LodAtlasUploadQueue,
    staging: &mut PackageStagingPermit<'_>,
) -> Result<(), GaussianLodPackageError> {
    if state.transient_atlas.ticket().is_failed() {
        return Err(GaussianLodPackageError::AtlasUpload(
            "GPU package atlas initialization failed".into(),
        ));
    }
    let generation = state.transient_atlas.ticket().generation();
    let replay = generation != state.transient_atlas_generation;
    if replay {
        enqueue_package_materialized_slots(state, uploads)?;
        state.transient_atlas_generation = generation;
    }
    let effective = state.structural.apply(settings);
    let live = views.iter().map(|v| v.entity).collect::<BTreeSet<_>>();
    let mut runtime = state
        .runtime
        .lock()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    #[cfg(feature = "testing")]
    let feedback_timer = crate::testing::lod_package_cpu::scope(
        crate::testing::lod_package_cpu::PackageCpuScope::GpuFeedback,
    );
    gpu.collect_retired(&mut runtime)?;
    for removed in state.views.difference(&live) {
        runtime
            .remove_view(LodRuntimeViewId(removed.to_bits()))
            .map_err(GaussianLodPackageError::Runtime)?;
    }
    let mut changed_demand = gpu.feedback.keys().any(|view| !live.contains(view));
    gpu.feedback.retain(|view, _| live.contains(view));
    gpu.navigation.retain(|view, _| live.contains(view));
    gpu.feedback_memory.retain(|view, _| live.contains(view));
    state.views = live;
    if views.is_empty() {
        gpu.release_pending_publication(&mut runtime)?;
        gpu.snapshot = None;
        gpu.placements.clear();
        gpu.feedback.clear();
        gpu.demands = GpuPageDemands::default();
        gpu.navigation.clear();
        gpu.feedback_memory.clear();
        gpu.excluded.clear();
        gpu.observed_revision = u64::MAX;
        gpu.first_generation = None;
        gpu.status = GaussianGpuLodPackageStatus::default();
        for page in std::mem::take(&mut gpu.root_pins) {
            runtime
                .release_resident_page(page)
                .map_err(GaussianLodPackageError::Runtime)?;
        }
        runtime
            .remove_view(PACKAGE_ROOT_FALLBACK_VIEW)
            .map_err(GaussianLodPackageError::Runtime)?;
        runtime
            .update_gpu_page_demands(&[], &[], &effective, &state.runtime_streaming)
            .map_err(GaussianLodPackageError::Runtime)?;
        state.resident_pages = runtime.cache().stats().resident_pages;
        return Ok(());
    }
    for view in views {
        let Some(mut feedback) = feedbacks.and_then(|f| f.take(view.entity, entity)) else {
            continue;
        };
        if !accepts_page_feedback(
            &feedback,
            state.atlas.id(),
            gpu.first_generation,
            gpu.status.residency_generation,
            gpu.feedback.get(&view.entity),
        ) || feedback
            .requested_pages
            .iter()
            .chain(&feedback.selected_pages)
            .any(|p| !gpu.page_indices.contains_key(p))
        {
            continue;
        }
        let required = ((feedback.requested_pages.len() + feedback.selected_pages.len()) as u64)
            .saturating_mul(144)
            .saturating_add(u64::from(state.plan.slot_count).saturating_mul(256));
        if gpu
            .feedback_memory
            .get(&view.entity)
            .is_none_or(|lease| lease.bytes() < required)
        {
            let lease = gpu
                .ledger
                .try_reserve(LodMemoryCategory::MetadataCpu, required)
                .map_err(GaussianLodPackageError::MemoryBudget)?;
            gpu.feedback_memory.insert(view.entity, lease);
        }
        feedback.selected_pages.sort_unstable();
        feedback.selected_pages.dedup();
        // The collector preserves the GPU's required-before-prefetch prefix.
        // Membership sets below deduplicate without changing that order.
        changed_demand |= gpu.feedback.get(&view.entity).is_none_or(|old| {
            old.selected_pages != feedback.selected_pages
                || old.requested_pages != feedback.requested_pages
        });
        if gpu.feedback.get(&view.entity).is_none_or(|old| {
            old.generation != feedback.generation || old.selected_pages != feedback.selected_pages
        }) {
            let navigation =
                resident_navigation(&feedback.selected_pages, &gpu.parents_by_page, |parent| {
                    // Selected-page feedback has page granularity. Unused nodes
                    // packed on that page must not demand absent ancestry.
                    gpu.snapshot.as_ref().is_some_and(|snapshot| {
                        gpu.page_indices
                            .get(&parent)
                            .is_some_and(|&index| snapshot.page_is_resident(index))
                    })
                });
            if gpu.navigation.get(&view.entity) != Some(&navigation) {
                changed_demand = true;
                gpu.navigation.insert(view.entity, navigation);
            }
        }
        gpu.feedback.insert(view.entity, feedback);
    }
    if changed_demand || gpu.demands.keep.is_empty() {
        gpu.demands = GpuPageDemands::from_views(&gpu.roots, &gpu.navigation, &gpu.feedback)?;
    }
    #[cfg(feature = "testing")]
    drop(feedback_timer);
    let GpuPageDemands {
        keep,
        requested,
        order,
        missing,
    } = &mut gpu.demands;
    let (keep, requested, order) = (&*keep, &*requested, order.as_slice());
    if changed_demand {
        let excluded_before = gpu.excluded.len();
        gpu.excluded
            .retain(|p| !requested.contains(p) && !keep.contains(p));
        if gpu.excluded.len() != excluded_before {
            gpu.observed_revision = u64::MAX;
        }
        let cancelled = gpu
            .pending_publication
            .iter()
            .filter(|page| !requested.contains(page) && !keep.contains(page))
            .copied()
            .collect::<Vec<_>>();
        for page in cancelled {
            runtime
                .release_resident_page(page)
                .map_err(GaussianLodPackageError::Runtime)?;
            gpu.pending_publication.remove(&page);
        }
    }
    // Selection and navigation stay per camera above. The selector-free page
    // runtime needs only their union: copying the retained set into every view
    // multiplied validation, cache touching, and admission bookkeeping.
    let demands = [(PACKAGE_ROOT_FALLBACK_VIEW, requested)];
    #[cfg(feature = "testing")]
    let demand_timer = crate::testing::lod_package_cpu::scope(
        crate::testing::lod_package_cpu::PackageCpuScope::GpuPageDemand,
    );
    // Previously excluded pages can already be decoded and therefore will not
    // appear in completed_pages. Protect them before other commits can evict
    // their slots, then transfer these same bounded leases to the new snapshot.
    if changed_demand || gpu.snapshot.is_none() {
        runtime
            .retain_gpu_pending_pages(order, &gpu.placements, &mut gpu.pending_publication)
            .map_err(GaussianLodPackageError::Runtime)?;
    }
    let update = runtime
        .update_gpu_page_demands(&demands, order, &effective, &state.runtime_streaming)
        .map_err(GaussianLodPackageError::Runtime)?;
    #[cfg(feature = "testing")]
    drop(demand_timer);
    #[cfg(feature = "testing")]
    let _publication_timer = crate::testing::lod_package_cpu::scope(
        crate::testing::lod_package_cpu::PackageCpuScope::GpuPublication,
    );
    for page in update.preprocess_failed_pages {
        runtime.transport_mut().invalidate_cached_page(page)?;
        if state.streaming.persistent_cache
            && state.preprocess_cache_repairs.insert(page)
            && runtime.is_terminal_failure(page)
        {
            runtime
                .retry_terminal_failure(page)
                .map_err(GaussianLodPackageError::Runtime)?;
        }
    }
    let _ = runtime.transport_mut().maintain_cache()?;
    for page in update.completed_pages {
        state.preprocess_cache_repairs.remove(&page);
        // Runtime completion holds end before this call returns. Transfer
        // still-demanded decoded pages into bounded publication ownership
        // before the next frame can evict a sibling whose staging was deferred.
        // Metadata is covered by the pending-publication slot reservation.
        if (requested.contains(&page) || keep.contains(&page))
            && !gpu.placements.contains_key(&page)
            && !gpu.pending_publication.contains(&page)
        {
            runtime
                .retain_resident_page(page)
                .map_err(GaussianLodPackageError::Runtime)?;
            gpu.pending_publication.insert(page);
        }
    }
    for &page in &gpu.roots {
        if !gpu.root_pins.contains(&page)
            && runtime.decoded_page(page).is_some()
            && runtime.cache().contains(page)
        {
            runtime
                .retain_resident_page(page)
                .map_err(GaussianLodPackageError::Runtime)?;
            gpu.root_pins.insert(page);
        }
    }
    let stats = runtime.cache().stats();
    let missing = missing.get_or_update(runtime.gpu_residency_revision(), requested, |page| {
        if runtime.cache().contains(page) {
            None
        } else {
            let descriptor = runtime
                .hierarchy()
                .page_descriptor(page)
                .expect("validated GPU page demand");
            Some((descriptor.decoded_len, u64::from(descriptor.gaussian_count)))
        }
    })?;
    let pressure = update.capacity_blocked_requests > 0
        || (missing.pages != 0
            && !runtime.cache().can_admit_with_eviction(
                missing.pages,
                missing.bytes,
                missing.gaussians,
            ));
    if pressure {
        for &page in gpu.placements.keys() {
            // Request feedback includes resident siblings of incomplete splits;
            // evicting one here would undo another sibling's loading progress.
            if !keep.contains(&page) && !requested.contains(&page) {
                gpu.excluded.insert(page);
            }
        }
    }
    let revision = runtime.gpu_residency_revision();
    // At most four immutable generations may own pages at once. A slow GPU
    // fences further publication; host snapshots never grow without a bound.
    if gpu.pins.len() < MAX_LIVE_SNAPSHOTS
        && (revision != gpu.observed_revision || pressure || replay || gpu.snapshot.is_none())
    {
        // Reuse the existing slot-bounded placement scratch for dirty pages.
        // Canonical resident iteration leaves it sorted without a rank table.
        let mut placement_entries = Vec::with_capacity(stats.resident_pages as usize);
        for (page, resident) in runtime.cache().resident_pages() {
            if !gpu.excluded.contains(&page)
                && !state.mirror.is_page_current(page, resident.slot)
                && runtime.decoded_page(page).is_some()
            {
                placement_entries.push((page, resident.slot));
            }
        }
        let mut all_materialized = true;
        if !placement_entries.is_empty() {
            let mut materialize = |page, slot: crate::stream::cache::AtlasSlot| {
                if !staging.try_consume_slot(
                    state.atlas.id(),
                    slot.index,
                    state.plan.gaussians_per_slot,
                    effective.budgets.max_upload_bytes_per_frame,
                )? {
                    return Ok::<_, GaussianLodPackageError>(false);
                }
                state
                    .mirror
                    .stage_page(page, slot)
                    .map_err(GaussianLodPackageError::RenderCommit)?;
                let payload = state
                    .mirror
                    .materialize_page_payload(
                        runtime.decoded_page(page).expect("checked decoded page"),
                        slot,
                    )
                    .map_err(GaussianLodPackageError::RenderCommit)?;
                state
                    .transient_atlas
                    .write_slot(slot.index, state.plan.gaussians_per_slot, payload)
                    .map_err(|e| GaussianLodPackageError::AtlasUpload(e.to_string()))?;
                uploads
                    .enqueue_slot(state.atlas.id(), slot, state.plan.gaussians_per_slot)
                    .map_err(|e| GaussianLodPackageError::AtlasUpload(e.to_string()))?;
                Ok(true)
            };
            // Only dirty pages participate in priority lookup. A held snapshot
            // performs no demand-sized cache lookup or placement sort.
            for &page in order {
                if let Ok(index) = placement_entries.binary_search_by_key(&page, |&(page, _)| page)
                {
                    all_materialized &= materialize(page, placement_entries[index].1)?;
                }
            }
            // Preserve replay/uncached work outside current demand, after all
            // required cohorts and speculative demand have had their turn.
            for &(page, slot) in &placement_entries {
                if !requested.contains(&page) {
                    all_materialized &= materialize(page, slot)?;
                }
            }
        }
        placement_entries.clear();
        for (page, resident) in runtime.cache().resident_pages() {
            if gpu.excluded.contains(&page) || runtime.decoded_page(page).is_none() {
                continue;
            }
            if state.mirror.is_page_current(page, resident.slot) {
                placement_entries.push((page, resident.slot));
            } else {
                all_materialized = false;
            }
        }
        // Bulk construction consumes the canonical resident order directly.
        let placements = placement_entries.into_iter().collect::<BTreeMap<_, _>>();
        if gpu.roots.iter().all(|p| placements.contains_key(p))
            && (placements != gpu.placements || replay || gpu.snapshot.is_none())
        {
            let reservation = gpu
                .ledger
                .try_reserve(
                    LodMemoryCategory::MetadataCpu,
                    (gpu.tree.page_ids().len() as u64)
                        .saturating_mul(32)
                        .saturating_add((placements.len() as u64).saturating_mul(192))
                        // Fixed-slot overlap validation uses bounded bit scratch.
                        .saturating_add(u64::from(state.plan.slot_count).div_ceil(64) * 8),
                )
                .map_err(GaussianLodPackageError::MemoryBudget)?;
            let mut pinned = Vec::with_capacity(placements.len());
            let lifetime = Arc::new(());
            let ownership = Arc::new(SnapshotOwner {
                _lifetime: lifetime.clone(),
                _tree_memory: gpu.tree_memory.clone(),
                _snapshot_memory: reservation,
            });
            let generation = NEXT_SNAPSHOT.fetch_add(1, Ordering::Relaxed);
            gpu.first_generation.get_or_insert(generation);
            let snapshot = GpuLodHierarchy::from_fixed_slots(
                gpu.tree.clone(),
                generation,
                state.atlas.id(),
                state.plan.slot_count,
                state.plan.gaussians_per_slot,
                placements
                    .iter()
                    .map(|(page, &slot)| (gpu.page_indices[page], slot)),
                ownership,
            )
            .map_err(GaussianLodPackageError::InvalidManifest)?;
            for &page in placements.keys() {
                if let Err(error) = gpu.retain_snapshot_page(&mut runtime, page) {
                    for old in pinned {
                        let _ = gpu.release_snapshot_page(&mut runtime, old);
                    }
                    return Err(error);
                }
                pinned.push(page);
            }
            gpu.pins.push(SnapshotPins {
                lifetime: Arc::downgrade(&lifetime),
                pages: pinned,
            });
            gpu.snapshot = Some(snapshot);
            gpu.placements = placements;
            // Transfer only pages this snapshot actually includes; another
            // bounded staging frame may still be needed for remaining pages.
            let published = gpu
                .pending_publication
                .iter()
                .filter(|page| gpu.placements.contains_key(page))
                .copied()
                .collect::<Vec<_>>();
            for page in published {
                runtime
                    .release_resident_page(page)
                    .map_err(GaussianLodPackageError::Runtime)?;
                gpu.pending_publication.remove(&page);
            }
            gpu.status.residency_generation = generation;
            gpu.status.snapshot_pages = gpu.placements.len() as u32;
        }
        if all_materialized {
            gpu.observed_revision = revision;
        }
    }
    gpu.status.visible_views = views.len() as u32;
    gpu.status.pending_publication_pages = gpu.pending_publication.len() as u32;
    gpu.status.spatial_mapping_bytes = gpu
        .spatial_mapping
        .as_ref()
        .and_then(|admission| admission.mapping.as_ref())
        .map_or(0, GpuLodSpatialMorph::mapping_bytes);
    gpu.status.spatial_mapping_error = gpu
        .spatial_mapping
        .as_ref()
        .and_then(|admission| admission.error.clone());
    gpu.status.acknowledged_views = views
        .iter()
        .filter(|view| {
            acknowledgements
                .and_then(|acks| acks.get(view.entity, entity))
                .is_some_and(|ack| {
                    cameras
                        .get(view.entity)
                        .ok()
                        .and_then(|(point, ordered, _, _, _)| camera_renderer(point, ordered).ok())
                        == Some(ack.renderer)
                        && ack.residency_generation == gpu.status.residency_generation
                        && ack.source == state.atlas.id()
                })
        })
        .count() as u32;
    gpu.status.selected_gaussians = gpu
        .feedback
        .values()
        .filter(|f| f.generation == gpu.status.residency_generation)
        .map(|f| u64::from(f.selected_gaussians))
        .sum();
    gpu.status.queued_requests = update.queued_requests as u32;
    gpu.status.in_flight_requests = update.in_flight_requests as u32;
    gpu.status.capacity_blocked_requests = update.capacity_blocked_requests as u32;
    gpu.status.record_limited = gpu.feedback.values().any(|f| f.record_limited);
    gpu.status.frontier_limited = gpu.feedback.values().any(|f| f.frontier_limited);
    gpu.status.visit_limited = gpu.feedback.values().any(|f| f.visit_limited);
    gpu.status.request_overflow = gpu.feedback.values().any(|f| f.request_overflow);
    gpu.status.cutoff_unavailable = gpu.feedback.values().any(|f| f.cutoff_unavailable);
    state.resident_pages = runtime.cache().stats().resident_pages;
    state.terminal_failures = runtime.terminal_failures().len() as u32;
    Ok(())
}

/// Stop publishing on invalid policy/admission/transport state without releasing
/// any snapshot page while a render-world command can still address its slot.
pub(super) fn fail(
    entity: Entity,
    state: &mut PackageInstantiation,
    error: GaussianLodPackageError,
    commands: &mut Commands,
) {
    if let Some(gpu) = state.gpu.as_mut() {
        gpu.snapshot = None;
        gpu.placements.clear();
        gpu.feedback.clear();
        gpu.demands = GpuPageDemands::default();
        gpu.spatial_mapping = None;
        gpu.navigation.clear();
        gpu.feedback_memory.clear();
        gpu.excluded.clear();
        gpu.observed_revision = u64::MAX;
        gpu.first_generation = None;
        gpu.status = GaussianGpuLodPackageStatus::default();
        if let Ok(mut runtime) = state.runtime.lock() {
            let _ = gpu.release_pending_publication(&mut runtime);
            for view in std::mem::take(&mut state.views) {
                let _ = runtime.remove_view(LodRuntimeViewId(view.to_bits()));
            }
            let _ = runtime.remove_view(PACKAGE_ROOT_FALLBACK_VIEW);
            for page in std::mem::take(&mut gpu.root_pins) {
                let _ = runtime.release_resident_page(page);
            }
            let _ = gpu.collect_retired(&mut runtime);
            let frame = runtime.begin_frame();
            let _ = runtime.finish_frame(frame);
        }
    }
    commands
        .entity(entity)
        .remove::<(GpuLodHierarchy, GpuLodSpatialMorph)>()
        .insert((
            LodRenderCandidates::package_required(),
            GaussianGpuLodPackageStatus::default(),
            GaussianLodPackageStatus::failed(error),
        ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_snapshots_share_one_runtime_lease_until_last_retirement() {
        let page = LodPageId(1);
        let mut references = SnapshotPageReferences(HashMap::with_capacity(1));
        let mut leases = 0;
        references
            .retain(page, || {
                leases += 1;
                Ok(())
            })
            .unwrap();
        references
            .retain(page, || panic!("shared page was pinned twice"))
            .unwrap();
        references
            .release(page, || panic!("page still belongs to another snapshot"))
            .unwrap();
        assert_eq!(leases, 1);
        assert!(
            references
                .release(page, || Err(LodRuntimeError::PhysicalIndexOverflow))
                .is_err()
        );
        assert_eq!(references.0[&page], 1, "failed release retains ownership");
        references
            .release(page, || {
                leases -= 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(leases, 0);
        assert!(references.0.is_empty());
        assert!(
            references
                .retain(page, || Err(LodRuntimeError::PhysicalIndexOverflow))
                .is_err()
        );
        assert!(
            references.0.is_empty(),
            "failed pin must not acquire ownership"
        );
    }

    #[test]
    fn spatial_mapping_admission_requires_current_ordered_camera_opt_in() {
        let spatial = GaussianLodSpatialTransitionSettings::default();
        assert_eq!(
            camera_spatial_mapping_limit(LodPresentationMode::Discrete, true, Some(&spatial)),
            Ok(None)
        );
        assert_eq!(
            camera_spatial_mapping_limit(
                LodPresentationMode::ContinuousMorph,
                false,
                Some(&spatial)
            ),
            Ok(None)
        );
        assert_eq!(
            camera_spatial_mapping_limit(LodPresentationMode::ContinuousMorph, true, None),
            Ok(None)
        );
        assert_eq!(
            camera_spatial_mapping_limit(
                LodPresentationMode::ContinuousMorph,
                true,
                Some(&spatial)
            ),
            Ok(Some(spatial.max_mapping_bytes))
        );
        let invalid = GaussianLodSpatialTransitionSettings {
            max_mapping_bytes: 0,
            ..spatial
        };
        assert!(
            camera_spatial_mapping_limit(
                LodPresentationMode::ContinuousMorph,
                true,
                Some(&invalid)
            )
            .is_err()
        );

        let package = crate::gaussian::formats::planar_3d_lod::build_planar_3d_lod(
            &crate::testing::LodTestScene::nested_octants(1).cloud(),
            crate::GaussianLodBuildSettings {
                leaf_capacity: 2,
                ..default()
            },
        )
        .unwrap();
        let manifest =
            crate::testing::upgrade_manifest_to_synthetic_abi16_lifecycle_fixture(package.manifest)
                .unwrap();
        let tree = Arc::new(GpuLodHierarchyTree::from_validated_manifest(&manifest).unwrap());
        let required = GpuLodSpatialMorph::required_mapping_bytes(&manifest).unwrap();
        let ledger = LodMemoryLedger::new(crate::stream::memory::LodMemoryLimits {
            max_cpu_bytes: required + 16,
            max_gpu_bytes: 0,
        });
        let busy = ledger
            .try_reserve(LodMemoryCategory::MetadataCpu, 32)
            .unwrap();
        let blocked = SpatialMappingAdmission::new(tree.clone(), &manifest, required, &ledger);
        assert!(blocked.mapping.is_none() && blocked.retry.is_some());
        assert!(!blocked.should_retry(required, &ledger));
        let capped = SpatialMappingAdmission::new(tree.clone(), &manifest, required - 4, &ledger);
        assert!(capped.mapping.is_none() && capped.retry.is_none());
        drop(busy);
        assert!(blocked.should_retry(required, &ledger));
        assert!(!capped.should_retry(required - 4, &ledger));
        let admitted = SpatialMappingAdmission::new(tree, &manifest, required, &ledger);
        assert_eq!(admitted.mapping.as_ref().unwrap().mapping_bytes(), required);
        assert!(admitted.error.is_none() && admitted.retry.is_none());
        assert!(!admitted.should_retry(required, &ledger));
        drop(admitted);
        assert_eq!(ledger.snapshot().cpu_bytes, 0);
    }

    #[test]
    fn page_demand_survives_residency_publication_without_accepting_stale_results() {
        let source = AssetId::<PlanarGaussian3d>::default();
        let mut feedback = GpuLodTraversalFeedback {
            generation: 20,
            submission: 7,
            source,
            requested_pages: vec![LodPageId(1)],
            selected_pages: vec![LodPageId(0)],
            selected_gaussians: 1,
            visited_nodes: 1,
            complete: true,
            record_limited: false,
            frontier_limited: false,
            visit_limited: false,
            request_overflow: false,
            cutoff_unavailable: false,
        };
        let previous = feedback.clone();
        feedback.submission += 1;
        // Uploading unrelated pages advanced residency while this demand was
        // being read back. It remains usable without acknowledging generation 21.
        assert!(accepts_page_feedback(
            &feedback,
            source,
            Some(10),
            21,
            Some(&previous)
        ));
        assert!(!accepts_page_feedback(
            &previous,
            source,
            Some(10),
            21,
            Some(&feedback)
        ));
        assert!(!accepts_page_feedback(
            &feedback,
            source,
            Some(10),
            21,
            Some(&feedback)
        ));
        assert!(!accepts_page_feedback(
            &feedback,
            source,
            Some(21),
            21,
            None
        ));
        assert!(!accepts_page_feedback(
            &feedback,
            source,
            Some(10),
            19,
            None
        ));
        assert!(!accepts_page_feedback(&feedback, source, None, 21, None));
        let other_source = AssetId::Uuid {
            uuid: bevy::asset::uuid::Uuid::from_u128(42),
        };
        assert!(!accepts_page_feedback(
            &feedback,
            other_source,
            Some(10),
            21,
            None
        ));
        feedback.complete = false;
        assert!(!accepts_page_feedback(
            &feedback,
            source,
            Some(10),
            21,
            None
        ));

        let roots = BTreeSet::from([LodPageId(1)]);
        let first = Entity::from_bits(11);
        let second = Entity::from_bits(12);
        let mut navigation = BTreeMap::from([
            (first, BTreeSet::from([LodPageId(2)])),
            (second, BTreeSet::from([LodPageId(3)])),
        ]);
        feedback.complete = true;
        feedback.requested_pages = [4, 5, 7].map(LodPageId).to_vec();
        let mut other = feedback.clone();
        other.requested_pages = [6, 5, 8].map(LodPageId).to_vec();
        let mut views = BTreeMap::from([(first, feedback), (second, other)]);
        let demand = GpuPageDemands::from_views(&roots, &navigation, &views).unwrap();
        assert_eq!(
            demand.order,
            [1, 4, 5, 7, 6, 8, 2, 3].map(LodPageId),
            "each view's demand order survives stable deduplication of shared pages"
        );
        views.remove(&first);
        navigation.remove(&first);
        let demand = GpuPageDemands::from_views(&roots, &navigation, &views).unwrap();
        assert_eq!(demand.order, [1, 6, 5, 8, 3].map(LodPageId));
        assert_eq!(
            demand.requested,
            BTreeSet::from([1, 3, 5, 6, 8].map(LodPageId))
        );

        let parents = HashMap::from([
            (LodPageId(4), BTreeSet::from([LodPageId(2)])),
            (LodPageId(5), BTreeSet::from([LodPageId(2)])),
            (LodPageId(2), BTreeSet::from([LodPageId(1)])),
        ]);
        assert_eq!(
            resident_navigation(&[LodPageId(4), LodPageId(5)], &parents, |_| true),
            BTreeSet::from([1, 2, 4, 5].map(LodPageId)),
        );
        assert_eq!(
            resident_navigation(&[LodPageId(4), LodPageId(5)], &parents, |page| {
                page == LodPageId(1)
            }),
            BTreeSet::from([4, 5].map(LodPageId)),
            "a missing intermediate page must not introduce unrelated packed-node ancestry"
        );

        let mut demand = demand;
        let footprint = demand
            .missing
            .get_or_update(7, &demand.requested, |page| {
                (page == LodPageId(6)).then_some((64, 4))
            })
            .unwrap();
        assert_eq!(
            footprint,
            MissingPageFootprint {
                pages: 1,
                bytes: 64,
                gaussians: 4
            }
        );
        assert_eq!(
            demand
                .missing
                .get_or_update(7, &demand.requested, |_| panic!(
                    "unchanged residency must reuse missing footprint"
                ))
                .unwrap(),
            footprint,
        );
        assert_eq!(
            demand
                .missing
                .get_or_update(8, &demand.requested, |_| None)
                .unwrap(),
            MissingPageFootprint::default(),
            "completed residency invalidates the footprint"
        );
        views.clear();
        navigation.clear();
        let mut removed = GpuPageDemands::from_views(&roots, &navigation, &views).unwrap();
        assert_eq!(
            removed
                .missing
                .get_or_update(8, &removed.requested, |_| Some((32, 2)))
                .unwrap(),
            MissingPageFootprint {
                pages: 1,
                bytes: 32,
                gaussians: 2
            },
            "view removal changes demand even when residency revision is unchanged"
        );
    }
}
