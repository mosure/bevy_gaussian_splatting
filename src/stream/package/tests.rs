#![allow(clippy::field_reassign_with_default)]

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
use std::num::NonZeroU32;

use super::platform::validate_native_root;
use super::*;

#[test]
fn package_poll_preserves_backend_failure_category() {
    let poll = map_package_poll(PagePoll::Failed("timeout"), |detail| {
        GaussianLodPackageTransportError::Http(detail.to_owned())
    });
    let PagePoll::Failed(error) = poll else {
        panic!("failed transport poll must remain failed");
    };
    assert!(matches!(
        error,
        GaussianLodPackageTransportError::Http(ref detail) if detail == "timeout"
    ));
    let failure = LodOrchestrationFailure::from(&error);
    assert_eq!(
        failure.code(),
        LodOrchestrationFailureCode::TransportRequestFailed
    );
}
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
use crate::stream::atlas_upload::LodAtlasSlotUpload;
use crate::{
    GaussianLodBuildSettings, LodNodeId,
    gaussian::formats::planar_3d_lod::build_planar_3d_lod,
    io::lod::encode_page,
    stream::{
        cache::AtlasSlot,
        runtime::{LodRuntimeError, LodRuntimeViewId, LodStreamingRuntime},
        transport::MemoryPageTransport,
    },
    testing::LodTestScene,
};

fn sparse_selection_test_plan(
    slot_count: u32,
    gaussians_per_slot: u32,
) -> GaussianLodPackageAtlasPlan {
    GaussianLodPackageAtlasPlan {
        virtual_source_gaussians: u64::from(slot_count) * u64::from(gaussians_per_slot),
        gaussians_per_slot,
        slot_count,
        physical_gaussians: slot_count.checked_mul(gaussians_per_slot).unwrap(),
        physical_bytes: 0,
    }
}

fn sparse_selection_test_range(
    node: u64,
    page: u64,
    slot: u32,
    generation: u32,
    physical_start: u32,
    count: u32,
) -> LodPhysicalRange {
    LodPhysicalRange {
        node: LodNodeId(node),
        page: LodPageId(page),
        slot: AtlasSlot {
            index: slot,
            generation,
        },
        physical_start,
        count,
    }
}

#[test]
fn package_staging_active_owner_uses_full_budget_before_idle_tail() {
    let canonical_slot_bytes = size_of::<Gaussian3d>() as u64;
    let gpu_slot_bytes = gaussian_3d_gpu_bytes_per_record();
    let budget = LodAtlasUploadBudget::try_new(canonical_slot_bytes * 3, 3).unwrap();
    let mut frame = PackageStagingFrame::new(budget);
    let atlas = AssetId::default();

    {
        let mut active = frame.begin_owner();
        for slot_index in 0..3 {
            assert!(
                active
                    .try_consume_slot(atlas, slot_index, 1, gpu_slot_bytes * 3)
                    .unwrap(),
                "the sole staging-active owner must use capacity that queried idle tails do not need"
            );
        }
        assert!(
            !active
                .try_consume_slot(atlas, 3, 1, gpu_slot_bytes * 4)
                .unwrap(),
            "the active owner must remain bounded by the aggregate slot and byte budget"
        );
    }
    drop(frame.begin_owner());
    drop(frame.begin_owner());
    assert_eq!(frame.remaining_canonical_bytes, 0);
    assert_eq!(frame.remaining_slots, 0);
}

#[test]
fn package_staging_atomic_floor_and_rotation_are_starvation_free() {
    let canonical_slot_bytes = size_of::<Gaussian3d>() as u64;
    let gpu_slot_bytes = gaussian_3d_gpu_bytes_per_record();
    let budget = LodAtlasUploadBudget::try_new(canonical_slot_bytes, 1).unwrap();
    let atlas = AssetId::default();
    let mut frame = PackageStagingFrame::new(budget);
    {
        let mut first = frame.begin_owner();
        assert!(
            first.try_consume_slot(atlas, 0, 1, gpu_slot_bytes).unwrap(),
            "one atomic slot must fit whenever the live aggregate budget can hold it"
        );
    }
    {
        let mut second = frame.begin_owner();
        assert!(
            !second
                .try_consume_slot(atlas, 1, 1, gpu_slot_bytes)
                .unwrap()
        );
    }
    drop(frame.begin_owner());

    let mut world = World::new();
    let mut owners = [
        world.spawn_empty().id(),
        world.spawn_empty().id(),
        world.spawn_empty().id(),
    ];
    owners.sort_unstable_by_key(|entity| entity.to_bits());
    let mut scheduler = GaussianLodPackageStagingScheduler::default();
    let mut observed = Vec::new();
    for _ in 0..owners.len() {
        let mut order = owners.to_vec();
        scheduler.rotate_to_next_owner(&mut order, |entity| *entity);
        observed.push(order[0]);
    }
    assert_eq!(observed, owners);
}

#[test]
fn package_staging_rejects_a_slot_larger_than_the_live_global_budget() {
    let canonical_slot_bytes = size_of::<Gaussian3d>() as u64;
    let budget = LodAtlasUploadBudget::try_new(canonical_slot_bytes - 1, 1).unwrap();
    let atlas = AssetId::default();
    let mut frame = PackageStagingFrame::new(budget);
    let error = frame
        .begin_owner()
        .try_consume_slot(atlas, 7, 1, gaussian_3d_gpu_bytes_per_record())
        .unwrap_err();
    assert_eq!(
        error,
        GaussianLodPackageError::MainWorldStagingBudget(
            LodAtlasUploadBudgetError::SlotExceedsCanonicalByteLimit {
                atlas,
                slot_index: 7,
                required: canonical_slot_bytes,
                limit: canonical_slot_bytes - 1,
            }
        )
    );
}

#[test]
fn sparse_atlas_selection_matches_dense_reference_and_preserves_gaps() {
    let plan = sparse_selection_test_plan(3, 8);
    let ranges = [
        sparse_selection_test_range(1, 10, 1, 3, 9, 2),
        sparse_selection_test_range(2, 10, 1, 3, 13, 2),
        sparse_selection_test_range(3, 20, 2, 4, 16, 1),
    ];
    let selection = plan_package_atlas_selection(plan, &ranges).unwrap();

    let mut dense_reference = vec![false; plan.physical_gaussians as usize];
    for range in ranges {
        dense_reference[range.physical_start as usize..range.end().unwrap() as usize].fill(true);
    }
    let mut sparse_result = vec![false; plan.physical_gaussians as usize];
    for intervals in selection.intervals_by_slot.values() {
        for interval in intervals {
            sparse_result[interval.start as usize..interval.end as usize].fill(true);
        }
    }

    assert_eq!(sparse_result, dense_reference);
    assert!(!sparse_result[8]);
    assert!(!sparse_result[11]);
    assert!(!sparse_result[12]);
    assert!(!sparse_result[15]);
    assert_eq!(selection.scratch().slots, 2);
    assert_eq!(selection.scratch().intervals, 3);
}

#[test]
fn sparse_atlas_selection_scratch_is_independent_of_physical_capacity() {
    let plan = sparse_selection_test_plan(1_000_000, 4);
    let ranges = [
        sparse_selection_test_range(1, 10, 7, 1, 28, 2),
        sparse_selection_test_range(2, 10, 7, 1, 30, 2),
        sparse_selection_test_range(3, 20, 999_999, 9, 3_999_999, 1),
    ];
    let selection = plan_package_atlas_selection(plan, &ranges).unwrap();

    assert_eq!(plan.physical_gaussians, 4_000_000);
    assert_eq!(
        selection.scratch(),
        PackageAtlasSelectionScratch {
            slots: 2,
            intervals: 3,
            materializations: 2,
        }
    );
}

#[test]
fn sparse_atlas_selection_rejects_overlap_and_inconsistent_ranges() {
    let plan = sparse_selection_test_plan(2, 8);
    let overlapping = [
        sparse_selection_test_range(1, 10, 0, 1, 1, 3),
        sparse_selection_test_range(2, 10, 0, 1, 3, 2),
    ];
    assert!(matches!(
        plan_package_atlas_selection(plan, &overlapping),
        Err(GaussianLodPackageError::Runtime(
            LodRuntimeError::OverlappingPhysicalRanges {
                previous_end: 4,
                next_start: 3,
            }
        ))
    ));

    let conflicting_generation = [
        sparse_selection_test_range(1, 10, 0, 1, 0, 1),
        sparse_selection_test_range(2, 10, 0, 2, 1, 1),
    ];
    assert!(matches!(
        plan_package_atlas_selection(plan, &conflicting_generation),
        Err(GaussianLodPackageError::ConflictingAtlasSlot { index: 0, .. })
    ));

    let conflicting_page = [
        sparse_selection_test_range(1, 10, 0, 1, 0, 1),
        sparse_selection_test_range(2, 11, 0, 1, 1, 1),
    ];
    assert!(matches!(
        plan_package_atlas_selection(plan, &conflicting_page),
        Err(GaussianLodPackageError::ConflictingAtlasPage { index: 0, .. })
    ));

    let outside_declared_slot = [sparse_selection_test_range(1, 10, 1, 1, 0, 1)];
    assert!(matches!(
        plan_package_atlas_selection(plan, &outside_declared_slot),
        Err(GaussianLodPackageError::RenderCommit(
            LodRenderCommitError::FrontierReferencesUnsynchronizedPage { .. }
        ))
    ));
}

#[test]
fn morph_union_over_atlas_headroom_downgrades_to_the_valid_target_cut() {
    let plan = sparse_selection_test_plan(1, 8);
    let target = vec![sparse_selection_test_range(1, 10, 0, 1, 0, 1)];
    let required_union = vec![
        target[0],
        // A parent-only morph source lies outside this deliberately tight
        // one-slot package atlas, while the exact target remains admissible.
        sparse_selection_test_range(2, 11, 1, 1, 8, 1),
    ];

    let (resolved, downgraded) =
        resolve_package_staging_ranges(plan, required_union.clone(), target.clone(), true).unwrap();
    assert!(downgraded);
    assert_eq!(resolved, target);
    assert!(
        resolve_package_staging_ranges(plan, required_union, resolved, false).is_err(),
        "a non-morph invalid cut must still fail closed"
    );
}

#[test]
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn downgraded_view_blend_cannot_use_progressive_admission() {
    let settings = GaussianLodSettings::default();
    let frontier =
        LodCandidateFrontier::complete_empty_for_test(LodRuntimeViewId::default(), &settings)
            .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing);
    let candidate = LodRenderCandidate::new(frontier);
    assert!(!package_candidate_has_downgraded_view_blend(&candidate));
    package_author_hard_candidate_mode(&candidate);
    assert!(package_candidate_has_downgraded_view_blend(&candidate));
    assert!(!package_progressive_view_blend_is_allowed(
        true, true, false
    ));
    assert!(!package_progressive_view_blend_is_allowed(
        true, false, true
    ));
}

#[test]
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn sticky_hard_same_payload_inherits_the_active_commit() {
    let settings = GaussianLodSettings::default();
    let frontier = || {
        LodCandidateFrontier::complete_empty_for_test(LodRuntimeViewId::default(), &settings)
            .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing)
    };
    let previous = LodRenderCandidate::new(frontier());
    package_author_hard_candidate_mode(&previous);
    assert_eq!(
        previous.active_presentation(),
        Some(LodRenderActivePresentation::HardTarget)
    );

    let mut replacement = LodRenderCandidate::new(frontier());
    package_author_hard_candidate_mode(&replacement);
    assert!(previous.same_payload(&replacement));
    replacement.inherit_active_payload_state(&previous);
    assert!(Arc::ptr_eq(&replacement.phase, &previous.phase));
    assert_eq!(
        replacement.active_presentation(),
        Some(LodRenderActivePresentation::HardTarget)
    );
}

#[test]
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn invalid_active_view_blend_pressure_is_explicit_and_auto_clears() {
    let valid = LodViewBlendStatusSnapshot {
        edge_count: 1,
        lagging_count: 0,
        invalid_pressure_count: 0,
        missing_consumer_count: 0,
        max_lag: 0.0,
        max_delta: 0.0,
        weighted_record_energy: 0.0,
        all_at_target: false,
    };
    assert!(!package_view_blend_status_has_invalid_pressure(Some(valid)));

    let invalid = LodViewBlendStatusSnapshot {
        invalid_pressure_count: 1,
        ..valid
    };
    assert!(package_view_blend_status_has_invalid_pressure(Some(
        invalid
    )));
    assert_eq!(
        invalid_view_blend_pressure_failure().code(),
        LodOrchestrationFailureCode::UnsupportedConfiguration
    );
    assert!(!package_view_blend_status_has_invalid_pressure(None));
}

#[test]
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn missing_view_blend_consumers_are_not_fixed_point_evidence() {
    let complete = LodViewBlendStatusSnapshot {
        edge_count: 2,
        lagging_count: 0,
        invalid_pressure_count: 0,
        missing_consumer_count: 0,
        max_lag: 0.0,
        max_delta: 0.0,
        weighted_record_energy: 0.0,
        all_at_target: true,
    };
    assert!(!package_view_blend_status_has_missing_consumers(Some(
        complete
    )));
    assert!(package_view_blend_status_has_missing_consumers(Some(
        LodViewBlendStatusSnapshot {
            missing_consumer_count: 1,
            all_at_target: false,
            ..complete
        }
    )));
    assert!(!package_view_blend_status_has_missing_consumers(None));
}

struct ViewBlendRetirementHierarchy {
    roots: Vec<LodNodeId>,
    parents: BTreeMap<LodNodeId, LodNodeId>,
    children: BTreeMap<LodNodeId, Vec<LodNodeId>>,
}

impl LodHierarchy for ViewBlendRetirementHierarchy {
    type NodeId = LodNodeId;

    fn roots(&self) -> &[Self::NodeId] {
        &self.roots
    }

    fn parent(&self, node: Self::NodeId) -> Option<Self::NodeId> {
        self.parents.get(&node).copied()
    }

    fn children(&self, node: Self::NodeId) -> &[Self::NodeId] {
        self.children
            .get(&node)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn metrics(&self, _node: Self::NodeId) -> Option<crate::stream::hierarchy::LodNodeMetrics> {
        None
    }
}

fn view_blend_retirement_hierarchy() -> ViewBlendRetirementHierarchy {
    let children = BTreeMap::from([
        (LodNodeId(1), vec![LodNodeId(2), LodNodeId(3)]),
        (LodNodeId(2), vec![LodNodeId(4), LodNodeId(5)]),
        (LodNodeId(3), vec![LodNodeId(6), LodNodeId(7)]),
        (LodNodeId(5), vec![LodNodeId(10), LodNodeId(11)]),
    ]);
    let parents = children
        .iter()
        .flat_map(|(parent, children)| children.iter().map(move |child| (*child, *parent)))
        .collect();
    ViewBlendRetirementHierarchy {
        roots: vec![LodNodeId(1)],
        parents,
        children,
    }
}

#[test]
fn adjacent_ancestor_coarsen_retires_parent_exact_edges_at_the_authored_child_endpoint() {
    let hierarchy = view_blend_retirement_hierarchy();
    let replacement_target = BTreeSet::from([LodNodeId(2), LodNodeId(3)]);
    let mut replacement_initial = replacement_target.clone();

    assert!(package_apply_provable_initial_view_blend_endpoint(
        &hierarchy,
        &replacement_target,
        &mut replacement_initial,
        LodNodeId(2),
        &[LodNodeId(4), LodNodeId(5)],
        1.0_f32.to_bits(),
        false,
    ));
    assert_eq!(
        replacement_initial,
        BTreeSet::from([LodNodeId(3), LodNodeId(4), LodNodeId(5)]),
        "the first drawable ancestor-coarsen frame must remain the retained 3/4/5 cut"
    );
    assert!(package_replacement_matches_removed_blend_endpoint(
        &hierarchy,
        LodNodeId(5),
        &[LodNodeId(10), LodNodeId(11)],
        LodViewBlendEndpoint::ParentExact,
        &replacement_initial,
    ));
}

#[test]
fn adjacent_ancestor_coarsen_retirement_rejects_unproven_initial_presentations() {
    let hierarchy = view_blend_retirement_hierarchy();
    let replacement_target = BTreeSet::from([LodNodeId(2), LodNodeId(3)]);

    let mut parent_initial = replacement_target.clone();
    assert!(package_apply_provable_initial_view_blend_endpoint(
        &hierarchy,
        &replacement_target,
        &mut parent_initial,
        LodNodeId(2),
        &[LodNodeId(4), LodNodeId(5)],
        0.0_f32.to_bits(),
        false,
    ));
    assert!(!package_replacement_matches_removed_blend_endpoint(
        &hierarchy,
        LodNodeId(5),
        &[LodNodeId(10), LodNodeId(11)],
        LodViewBlendEndpoint::ParentExact,
        &parent_initial,
    ));

    let child_initial = BTreeSet::from([LodNodeId(3), LodNodeId(4), LodNodeId(5)]);
    assert!(!package_replacement_matches_removed_blend_endpoint(
        &hierarchy,
        LodNodeId(5),
        &[LodNodeId(10), LodNodeId(11)],
        LodViewBlendEndpoint::Fractional,
        &child_initial,
    ));

    let mut incomplete_cohort = replacement_target.clone();
    assert!(!package_apply_provable_initial_view_blend_endpoint(
        &hierarchy,
        &replacement_target,
        &mut incomplete_cohort,
        LodNodeId(2),
        &[LodNodeId(4)],
        1.0_f32.to_bits(),
        false,
    ));
    assert_eq!(incomplete_cohort, replacement_target);

    let mut fractional_initial = replacement_target.clone();
    assert!(!package_apply_provable_initial_view_blend_endpoint(
        &hierarchy,
        &replacement_target,
        &mut fractional_initial,
        LodNodeId(2),
        &[LodNodeId(4), LodNodeId(5)],
        0.5_f32.to_bits(),
        false,
    ));
    assert_eq!(fractional_initial, replacement_target);

    let mixed_target = BTreeSet::from([LodNodeId(2), LodNodeId(3), LodNodeId(4)]);
    let mut mixed_initial = mixed_target.clone();
    assert!(!package_apply_provable_initial_view_blend_endpoint(
        &hierarchy,
        &mixed_target,
        &mut mixed_initial,
        LodNodeId(2),
        &[LodNodeId(4), LodNodeId(5)],
        1.0_f32.to_bits(),
        false,
    ));
    assert_eq!(mixed_initial, mixed_target);

    let partial_children_target = BTreeSet::from([LodNodeId(3), LodNodeId(4)]);
    let mut partial_children_initial = partial_children_target.clone();
    assert!(!package_apply_provable_initial_view_blend_endpoint(
        &hierarchy,
        &partial_children_target,
        &mut partial_children_initial,
        LodNodeId(2),
        &[LodNodeId(4), LodNodeId(5)],
        1.0_f32.to_bits(),
        false,
    ));
    assert_eq!(partial_children_initial, partial_children_target);

    let mut inherited = replacement_target.clone();
    assert!(package_apply_provable_initial_view_blend_endpoint(
        &hierarchy,
        &replacement_target,
        &mut inherited,
        LodNodeId(2),
        &[LodNodeId(4), LodNodeId(5)],
        1.0_f32.to_bits(),
        true,
    ));
    assert_eq!(
        inherited,
        BTreeSet::from([LodNodeId(3)]),
        "a common edge inherits render-owned weights, so its immutable initial endpoint is not retirement evidence"
    );
}

#[test]
fn active_view_blend_union_retains_both_endpoint_ranges_and_recovery_slots() {
    let parent = sparse_selection_test_range(1, 10, 0, 1, 0, 1);
    let child = sparse_selection_test_range(2, 11, 1, 1, 8, 2);
    let staged = PackageStagedCut {
        ranges: vec![parent, child],
        slots: BTreeMap::from([(0, parent.slot), (1, child.slot)]),
        materializations: Vec::new(),
        next_materialization: 0,
        complete: true,
        fallback_nodes: BTreeSet::new(),
        debug: PackageStagedDebugPreparation {
            complete: true,
            ..default()
        },
    };

    assert_eq!(staged.ranges, [parent, child]);
    assert_eq!(
        staged
            .ranges
            .iter()
            .map(|range| range.page)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([parent.page, child.page]),
        "ACTIVE camera-conditioned presentation retains both endpoint leases"
    );
    assert_eq!(
        staged.slots,
        BTreeMap::from([
            (parent.slot.index, parent.slot),
            (child.slot.index, child.slot),
        ]),
        "generation-loss recovery retains every slot referenced by the endpoint union"
    );
}
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
use crate::{
    gaussian::{
        formats::{
            planar_3d_chunked::{LodPageEncoding, LodPageKind, LodPageStorage},
            planar_3d_lod::{
                EXTERNAL_MOMENT_MERGE_VERSION, lod_config_fingerprint,
                lod_config_fingerprint_for_reducer,
            },
        },
        lod_debug::LodDebugPreset,
    },
    io::lod::{
        LodCodecLimits, LodShardEntry, LodShardIndex, decode_manifest, decode_page,
        decode_page_with_descriptor, encode_lod_shard_index, encode_manifest,
        encode_page_with_encoding, lod_shard_prefix_len,
    },
    stream::{
        preprocess::{LodPagePreprocessError, LodPagePreprocessInput, LodPagePreprocessor},
        render_commit::LOD_RENDER_PREPARED,
        transport::PageRequest,
    },
};
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
use std::sync::Arc;

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
struct NativeTestPackage {
    root: std::path::PathBuf,
    manifest: crate::GaussianLodManifest,
    source_count: usize,
    omitted_page: Option<LodPageId>,
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
impl Drop for NativeTestPackage {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
struct LocalPackageHttpServer {
    address: std::net::SocketAddr,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    requests: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ranges: RequestedByteRanges,
    worker: Option<std::thread::JoinHandle<()>>,
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
type RequestedByteRanges = std::sync::Arc<std::sync::Mutex<Vec<Option<(u64, u64)>>>>;

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
impl LocalPackageHttpServer {
    fn start(root: std::path::PathBuf) -> Self {
        use std::io::{Read as _, Write as _};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let requests = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let ranges = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let worker_stop = stop.clone();
        let worker_requests = requests.clone();
        let worker_ranges = ranges.clone();
        let worker = std::thread::spawn(move || {
            while !worker_stop.load(std::sync::atomic::Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(_) => break,
                };
                let mut request = [0_u8; 8192];
                let Ok(read) = stream.read(&mut request) else {
                    continue;
                };
                let line = String::from_utf8_lossy(&request[..read]);
                let Some(uri) = line
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                else {
                    continue;
                };
                let relative = uri.trim_start_matches('/');
                if relative.split('/').any(|part| part == "..") {
                    continue;
                }
                let byte_range = line.lines().find_map(|header| {
                    let (name, value) = header.split_once(':')?;
                    if !name.eq_ignore_ascii_case("range") {
                        return None;
                    }
                    let (start, end) = value.trim().strip_prefix("bytes=")?.split_once('-')?;
                    Some((start.parse::<u64>().ok()?, end.parse::<u64>().ok()?))
                });
                worker_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                worker_ranges.lock().unwrap().push(byte_range);
                match std::fs::read(root.join(relative)) {
                    Ok(bytes) => {
                        if let Some((start, end)) = byte_range {
                            let range = usize::try_from(start)
                                .ok()
                                .zip(usize::try_from(end).ok())
                                .filter(|(start, end)| start <= end && *end < bytes.len());
                            if let Some((start, end)) = range {
                                let payload = &bytes[start..=end];
                                let header = format!(
                                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\nETag: \"fixture-v1\"\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                                    payload.len(),
                                    bytes.len()
                                );
                                let _ = stream.write_all(header.as_bytes());
                                let _ = stream.write_all(payload);
                            } else {
                                let _ = stream.write_all(
                                    b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nETag: \"fixture-v1\"\r\nConnection: close\r\n\r\n",
                                );
                            }
                        } else {
                            let header = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: \"fixture-v1\"\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                                bytes.len()
                            );
                            let _ = stream.write_all(header.as_bytes());
                            let _ = stream.write_all(&bytes);
                        }
                    }
                    Err(_) => {
                        let _ = stream.write_all(
                            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nETag: \"fixture-v1\"\r\nConnection: close\r\n\r\n",
                        );
                    }
                }
            }
        });
        Self {
            address,
            stop,
            requests,
            ranges,
            worker: Some(worker),
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}/", self.address)
    }
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
impl Drop for LocalPackageHttpServer {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
fn poll_package_transport(
    transport: &mut PackagePageTransport,
    request: PageRequest,
) -> crate::stream::transport::PagePayload {
    let ticket = transport.begin(request).unwrap();
    for _ in 0..10_000 {
        match transport.poll(&ticket) {
            PagePoll::Pending => std::thread::sleep(Duration::from_millis(1)),
            PagePoll::Ready(payload) => return payload,
            PagePoll::Failed(error) => panic!("package transport failed: {error}"),
        }
    }
    panic!("package transport timed out")
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn write_native_test_package(omit_leaf: bool) -> NativeTestPackage {
    write_native_test_package_with_degree(omit_leaf, None)
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn write_native_test_package_with_levels(levels: u32) -> NativeTestPackage {
    write_native_test_package_with_degree_and_levels(false, None, levels)
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn write_native_test_package_with_levels_and_leaf_capacity(
    levels: u32,
    leaf_capacity: u32,
) -> NativeTestPackage {
    write_native_test_package_with_build(false, None, levels, leaf_capacity)
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn write_native_test_package_with_degree(
    omit_leaf: bool,
    representative_degree: Option<u8>,
) -> NativeTestPackage {
    write_native_test_package_with_degree_and_levels(omit_leaf, representative_degree, 2)
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn write_native_test_package_with_degree_and_levels(
    omit_leaf: bool,
    representative_degree: Option<u8>,
    levels: u32,
) -> NativeTestPackage {
    write_native_test_package_with_build(omit_leaf, representative_degree, levels, 8)
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn write_native_test_package_with_build(
    omit_leaf: bool,
    representative_degree: Option<u8>,
    levels: u32,
    leaf_capacity: u32,
) -> NativeTestPackage {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PACKAGE: AtomicU64 = AtomicU64::new(1);
    let unique = NEXT_PACKAGE.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "bevy-gaussian-lod-package-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir(&root).unwrap();
    std::fs::create_dir(root.join("pages")).unwrap();

    let source = LodTestScene::nested_octants(levels).cloud();
    let mut built = build_planar_3d_lod(
        &source,
        GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity,
            support_sigma: 3.0,
        },
    )
    .unwrap();
    let omitted_page = omit_leaf.then(|| {
        built
            .manifest
            .pages
            .iter()
            .find(|page| page.kind == LodPageKind::SourceLeaves)
            .expect("fixture must contain a source leaf")
            .id
    });
    if let Some(degree) = representative_degree {
        built.manifest.build.config_fingerprint =
            lod_config_fingerprint(built.manifest.build.settings, Some(degree));
    }
    let mut encoded_pages = Vec::new();
    for page in &built.pages {
        let descriptor = built
            .manifest
            .pages
            .iter_mut()
            .find(|descriptor| descriptor.id == page.id)
            .unwrap();
        let encoding = if descriptor.kind == LodPageKind::Representatives {
            representative_degree.map_or(LodPageEncoding::F32Planar, |degree| {
                LodPageEncoding::F16Sh { degree }
            })
        } else {
            LodPageEncoding::F32Planar
        };
        let encoded = encode_page_with_encoding(page, encoding).unwrap();
        let canonical = decode_page(&encoded, LodCodecLimits::default()).unwrap();
        descriptor.encoding = encoding;
        descriptor.content_hash = canonical.content_hash();
        if Some(page.id) != omitted_page {
            encoded_pages.push((page.id, encoded));
        }
    }
    encoded_pages.sort_unstable_by_key(|(page_id, _)| *page_id);
    let prefix_len = lod_shard_prefix_len(encoded_pages.len() as u32).unwrap();
    let mut cursor = prefix_len;
    let entries = encoded_pages
        .iter()
        .map(|(page_id, encoded)| {
            let descriptor = built
                .manifest
                .pages
                .iter()
                .find(|descriptor| descriptor.id == *page_id)
                .unwrap();
            let entry = LodShardEntry {
                page_id: *page_id,
                byte_offset: cursor,
                encoded_len: encoded.len() as u64,
                content_hash: descriptor.content_hash,
            };
            cursor += encoded.len() as u64;
            entry
        })
        .collect::<Vec<_>>();
    let shard_uri = "pages/shard-000000.bgslodpack";
    let mut shard = encode_lod_shard_index(&LodShardIndex {
        file_len: cursor,
        entries: entries.clone(),
    })
    .unwrap();
    for (_, encoded) in &encoded_pages {
        shard.extend_from_slice(encoded);
    }
    assert_eq!(shard.len() as u64, cursor);
    std::fs::write(root.join(shard_uri), shard).unwrap();

    for descriptor in &mut built.manifest.pages {
        if Some(descriptor.id) == omitted_page {
            let encoded_len = built
                .pages
                .iter()
                .find(|page| page.id == descriptor.id)
                .map(|page| {
                    encode_page_with_encoding(page, descriptor.encoding)
                        .unwrap()
                        .len() as u64
                })
                .unwrap();
            descriptor.storage = Some(LodPageStorage {
                uri: format!("pages/missing-page-{}.gspage", descriptor.id.0),
                byte_range: None,
                encoded_len,
            });
            continue;
        }
        let entry = entries
            .iter()
            .find(|entry| entry.page_id == descriptor.id)
            .unwrap();
        descriptor.storage = Some(LodPageStorage {
            uri: shard_uri.to_owned(),
            byte_range: Some((entry.byte_offset, entry.encoded_len)),
            encoded_len: entry.encoded_len,
        });
    }
    built.manifest.validate().unwrap();
    let encoded_manifest = encode_manifest(&built.manifest).unwrap();
    std::fs::write(root.join("scene.gsplatlod"), &encoded_manifest).unwrap();
    let manifest = decode_manifest(&encoded_manifest, LodCodecLimits::default()).unwrap();
    assert_eq!(manifest, built.manifest);
    NativeTestPackage {
        root,
        manifest,
        source_count: source.len(),
        omitted_page,
    }
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn package_test_settings(quality: f32) -> GaussianLodSettings {
    let mut settings = GaussianLodSettings::default();
    settings.quality = quality;
    settings.frustum_culling = false;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_pending_requests = 512;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
    settings
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn mark_package_cloud_visible(world: &mut World, camera: Entity, cloud: Entity) {
    let mut visible = world
        .get_mut::<VisibleEntities>(camera)
        .expect("Camera must provide VisibleEntities");
    if !visible
        .iter(TypeId::of::<CloudVisibilityClass>())
        .any(|entity| *entity == cloud)
    {
        visible.push(cloud, TypeId::of::<CloudVisibilityClass>());
    }
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn package_test_world(
    package: &NativeTestPackage,
    settings: GaussianLodSettings,
    debug_metadata: bool,
    retry_limit: u32,
) -> (World, Entity, Entity, Handle<GaussianLodAsset>) {
    let mut world = World::new();
    world.init_resource::<Assets<GaussianLodAsset>>();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<GaussianLodAsset>>>();
    let mut config = GaussianLodPackageConfig {
        max_atlas_gaussians: 4096,
        max_atlas_bytes: 64 * 1024 * 1024,
        max_views_per_cloud: 4,
        ..default()
    };
    config.streaming.retry_limit = retry_limit;
    world.insert_resource(config);
    world.init_resource::<GaussianLodPackageManager>();
    world.init_resource::<GaussianLodPackageStagingScheduler>();
    world.init_resource::<LodAtlasUploadBudget>();
    world.init_resource::<LodAtlasUploadQueue>();
    world.init_resource::<LodTransientAtlasRegistry>();
    let manifest_handle = world
        .resource_mut::<Assets<GaussianLodAsset>>()
        .add(GaussianLodAsset::new(package.manifest.clone()).unwrap());
    let cloud = spawn_package_test_cloud(
        &mut world,
        package,
        manifest_handle.clone(),
        settings,
        debug_metadata,
    );
    let camera = world
        .spawn((
            Camera {
                viewport: Some(bevy::camera::Viewport {
                    physical_size: UVec2::new(1280, 720),
                    ..default()
                }),
                ..default()
            },
            Projection::Perspective(default()),
            GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 5.0)),
            crate::GaussianCamera::default(),
        ))
        .id();
    mark_package_cloud_visible(&mut world, camera, cloud);
    (world, cloud, camera, manifest_handle)
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn spawn_package_test_cloud(
    world: &mut World,
    package: &NativeTestPackage,
    manifest_handle: Handle<GaussianLodAsset>,
    settings: GaussianLodSettings,
    debug_metadata: bool,
) -> Entity {
    let mut cloud_settings = CloudSettings::default();
    cloud_settings.sort_mode = crate::sort::SortMode::Radix;
    if debug_metadata {
        cloud_settings
            .lod_debug
            .apply_preset(LodDebugPreset::Residency);
    }
    world
        .spawn((
            GaussianLodHandle(manifest_handle),
            GaussianLodPackageSource::native_directory(package.root.to_string_lossy().into_owned()),
            settings,
            cloud_settings,
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id()
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn run_package_frame(schedule: &mut Schedule, world: &mut World, cloud: Entity) -> usize {
    run_package_frame_for_clouds(schedule, world, &[cloud])
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn run_package_frame_for_clouds(
    schedule: &mut Schedule,
    world: &mut World,
    clouds: &[Entity],
) -> usize {
    for &cloud in clouds {
        advance_package_render_candidates(world, cloud);
    }
    // The real extraction schedule consumes this queue once per frame.
    // Replacing it here models that boundary without constructing a GPU.
    world.insert_resource(LodAtlasUploadQueue::default());
    schedule.run(world);
    std::thread::yield_now();
    world.resource::<LodAtlasUploadQueue>().queued_slot_count()
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn advance_package_render_candidates(world: &mut World, cloud: Entity) {
    let pending_is_activation_ready = world
        .resource::<GaussianLodPackageManager>()
        .clouds
        .get(&cloud)
        .is_some_and(|state| {
            state
                .staged
                .as_ref()
                .is_some_and(|staged| staged.complete && staged.debug.complete)
        });
    if let Some(candidates) = world.get::<LodRenderCandidates>(cloud) {
        for candidate in candidates.by_camera.values() {
            if candidate.failed() {
                continue;
            }
            match candidate.phase.load(Ordering::Acquire) {
                LOD_RENDER_WAITING => candidate
                    .phase
                    .store(LOD_RENDER_PREPARED, Ordering::Release),
                LOD_RENDER_PREPARED if pending_is_activation_ready => {
                    candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release)
                }
                _ => {}
            }
        }
    }
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn sparse_package_atlas_snapshot(state: &PackageInstantiation) -> Vec<Gaussian3d> {
    let mut snapshot = vec![Gaussian3d::default(); state.plan.physical_gaussians as usize];
    for slot in state.mirror.materialized_slots() {
        let payload = state
            .transient_atlas
            .snapshot_slot(LodAtlasSlotUpload {
                atlas: state.atlas.id(),
                slot,
                gaussians_per_slot: state.plan.gaussians_per_slot,
            })
            .unwrap();
        let start = slot.index as usize * state.plan.gaussians_per_slot as usize;
        for (offset, gaussian) in payload.iter().enumerate() {
            snapshot[start + offset] = gaussian;
        }
    }
    snapshot
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn drive_package_to_active_count(
    schedule: &mut Schedule,
    world: &mut World,
    cloud: Entity,
    camera: Entity,
    expected: u32,
) -> usize {
    let mut maximum_queued = 0;
    for _ in 0..2048 {
        let queued = run_package_frame(schedule, world, cloud);
        maximum_queued = maximum_queued.max(queued);
        let active = world
            .get::<GaussianLodPackageStatus>(cloud)
            .is_some_and(|status| status.phase == GaussianLodPackagePhase::Active);
        let exact = world
            .get::<LodRenderCandidates>(cloud)
            .and_then(|candidates| candidates.get(camera))
            .is_some_and(|candidate| {
                candidate.frontier().candidate_count() == expected
                    && candidate.phase.load(Ordering::Acquire) == LOD_RENDER_ACTIVE
            });
        let manager_committed = world
            .resource::<GaussianLodPackageManager>()
            .clouds
            .get(&cloud)
            .and_then(|state| state.current.as_ref())
            .is_some_and(package_candidate_set_is_active);
        if active && exact && manager_committed {
            return maximum_queued;
        }
    }
    panic!(
        "native package did not reach {expected} active Gaussians; status={:?}, candidates={:?}",
        world.get::<GaussianLodPackageStatus>(cloud),
        world
            .get::<LodRenderCandidates>(cloud)
            .and_then(|candidates| candidates.get(camera))
            .map(|candidate| candidate.frontier().candidate_count())
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn package_test_camera_view(entity: Entity, position: Vec3) -> PackageCameraView {
    PackageCameraView {
        entity,
        view: LodView::perspective(position, 720.0, 60.0_f32.to_radians(), 0.1),
    }
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn install_complete_empty_test_pending(
    state: &mut PackageInstantiation,
    settings: &GaussianLodSettings,
    transform: &GlobalTransform,
    camera: PackageCameraView,
) -> Arc<std::sync::atomic::AtomicU8> {
    let effective = state.structural.apply(settings);
    let mut candidates = LodRenderCandidates::default();
    candidates.insert(
        camera.entity,
        LodCandidateFrontier::complete_empty_for_test(
            LodRuntimeViewId(camera.entity.to_bits()),
            &effective,
        ),
    );
    let phase = Arc::clone(&candidates.get(camera.entity).unwrap().phase);
    let staged = prepare_package_staged_cut(state, &[], &BTreeSet::new()).unwrap();
    state.staged = Some(staged);
    state.pending = Some(candidates);
    state.pending_request = Some(PackageCutRequestSignature::new(
        &effective,
        transform,
        std::slice::from_ref(&camera),
    ));
    state.pending_request_fixed_point = false;
    state.pending_progressive_view_blend = false;
    state.pending_presentation_modes = state
        .pending
        .as_ref()
        .map(package_candidate_presentation_modes)
        .unwrap_or_default();
    state.pending_fallback_nodes.clear();
    replace_package_pending_page_leases(state, &BTreeSet::new()).unwrap();
    phase
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn package_transition_candidate_set(
    settings: &GaussianLodSettings,
    candidates: &[(Entity, u8)],
) -> LodRenderCandidates {
    let mut pending = LodRenderCandidates::default();
    for (camera, phase) in candidates {
        pending.insert(
            *camera,
            LodCandidateFrontier::complete_empty_for_test(
                LodRuntimeViewId(camera.to_bits()),
                settings,
            ),
        );
        pending
            .get(*camera)
            .expect("inserted transition fixture")
            .phase
            .store(*phase, Ordering::Release);
    }
    pending
}

#[test]
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn two_bounded_steps_leave_request_unowned_until_the_stable_followup() {
    let settings = GaussianLodSettings::default();
    let camera = Entity::from_bits(103);
    let view = package_test_camera_view(camera, Vec3::new(0.0, 0.0, 5.0));
    let request = PackageCutRequestSignature::new(
        &settings,
        &GlobalTransform::IDENTITY,
        std::slice::from_ref(&view),
    );

    // Model two separately activated bounded cohorts for one unchanged camera
    // request. Neither endpoint is the selector's stationary fixed point.
    for step in 1..=2 {
        assert_eq!(
            package_request_ownership_after_commit(Some(request.clone()), false, false),
            None,
            "bounded temporal step {step} claimed the full stationary request"
        );
    }
    assert_eq!(
        package_request_ownership_after_commit(Some(request.clone()), false, true),
        Some(request.clone()),
        "only the stable no-transition follow-up may claim the request"
    );
    assert_eq!(
        package_request_ownership_after_commit(Some(request), true, true),
        None,
        "a bootstrap remains an intermediate even when its own cut is stable"
    );
}

#[test]
fn same_payload_frozen_resume_waits_for_one_render_publication() {
    assert!(!package_same_payload_request_fixed_point(true, true, false));
    assert!(
        !package_same_payload_request_fixed_point(true, false, true),
        "render-owned recovery lag also keeps the request unowned"
    );
    assert!(
        package_same_payload_request_fixed_point(true, false, false),
        "the unchanged follow-up may claim ownership after render convergence"
    );
    assert!(!package_same_payload_request_fixed_point(
        false, false, false
    ));
}

#[test]
fn stationary_owned_request_keeps_driving_predictive_maintenance() {
    assert!(package_current_request_can_short_circuit(true, true, false));
    assert!(
        !package_current_request_can_short_circuit(true, true, true),
        "an owned stationary request must still poll/start its optional predictive cohort"
    );
    assert!(!package_current_request_can_short_circuit(
        false, true, false
    ));
    assert!(!package_current_request_can_short_circuit(
        true, false, false
    ));
}

#[test]
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn stale_settings_finish_live_transition_and_companion_before_superseding() {
    use crate::stream::render_commit::{
        LOD_RENDER_ACTIVE, LOD_RENDER_PREPARED, LOD_RENDER_TRANSITIONING,
    };

    let settings = GaussianLodSettings::default();
    let owner = Entity::from_bits(101);
    let companion = Entity::from_bits(102);
    let mut pending = package_transition_candidate_set(
        &settings,
        &[
            (owner, LOD_RENDER_TRANSITIONING),
            (companion, LOD_RENDER_PREPARED),
        ],
    );
    let live = BTreeSet::from([owner, companion]);
    let staged_union_leases = BTreeSet::from([LodPageId(11), LodPageId(12), LodPageId(13)]);

    assert_eq!(
        reconcile_stale_package_pending_transition(&mut pending, &live, false),
        PackagePendingStaleDisposition::FinishAndCommit,
    );
    assert_eq!(pending.len(), 2);
    assert_eq!(
        staged_union_leases,
        BTreeSet::from([LodPageId(11), LodPageId(12), LodPageId(13)]),
        "FinishAndCommit leaves the caller's staged union ownership untouched",
    );

    pending
        .get(owner)
        .unwrap()
        .phase
        .store(LOD_RENDER_ACTIVE, Ordering::Release);
    pending
        .get(companion)
        .unwrap()
        .phase
        .store(LOD_RENDER_ACTIVE, Ordering::Release);
    assert_eq!(
        reconcile_stale_package_pending_transition(&mut pending, &live, true),
        PackagePendingStaleDisposition::FinishAndCommit,
        "the stale request remains latched through exact ACTIVE publication so drive_package_state commits it before selecting a replacement",
    );
    assert!(package_candidate_set_is_active(&pending));
}

#[test]
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn stale_transition_camera_removal_retires_only_nondrawable_consumers() {
    use crate::stream::render_commit::{
        LOD_RENDER_PREPARED, LOD_RENDER_TRANSITIONING, LOD_RENDER_WAITING,
    };

    let settings = GaussianLodSettings::default();
    let owner = Entity::from_bits(201);
    let companion = Entity::from_bits(202);
    let mut pending = package_transition_candidate_set(
        &settings,
        &[
            (owner, LOD_RENDER_TRANSITIONING),
            (companion, LOD_RENDER_PREPARED),
        ],
    );
    let companion_phase = Arc::clone(&pending.get(companion).unwrap().phase);

    assert_eq!(
        reconcile_stale_package_pending_transition(&mut pending, &BTreeSet::from([owner]), true,),
        PackagePendingStaleDisposition::FinishAndCommit,
    );
    assert_eq!(pending.len(), 1);
    assert!(pending.get(owner).is_some());
    assert_eq!(
        companion_phase.load(Ordering::Acquire),
        LOD_RENDER_WAITING,
        "the removed sibling is revoked before its candidate entry is retired",
    );

    assert_eq!(
        reconcile_stale_package_pending_transition(&mut pending, &BTreeSet::new(), true),
        PackagePendingStaleDisposition::CancelSafe,
        "with every participating RenderView gone, no drawable bit-28 owner can block cancellation",
    );
    assert!(pending.is_empty());

    let mut removed_owner = package_transition_candidate_set(
        &settings,
        &[
            (owner, LOD_RENDER_TRANSITIONING),
            (companion, LOD_RENDER_PREPARED),
        ],
    );
    assert_eq!(
        reconcile_stale_package_pending_transition(
            &mut removed_owner,
            &BTreeSet::from([companion]),
            false,
        ),
        PackagePendingStaleDisposition::CancelSafe,
        "a transition owned only by a removed view has no live GPU consumer",
    );
}

#[test]
fn native_roots_reject_url_schemes_while_http_sources_validate() {
    let error = validate_native_root("https://cdn.example/scene/").unwrap_err();
    assert_eq!(
        error,
        GaussianLodPackageError::UnsupportedUrlScheme("https".to_owned())
    );
    assert!(
        package_http_config(
            "https://cdn.example/scene/",
            &GaussianStreamingSettings::default(),
        )
        .is_ok()
    );
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
#[test]
fn two_http_packages_share_one_writer_and_reuse_it_offline() {
    let package = write_native_test_package(false);
    let server = LocalPackageHttpServer::start(package.root.clone());
    let request_count = server.requests.clone();
    let source = GaussianLodPackageSource::url(server.base_url());
    let descriptor = package.manifest.pages.first().unwrap();
    let request = PageRequest {
        page_id: descriptor.id,
        priority: crate::stream::transport::PageRequestPriority::fallback_critical(u32::MAX),
        expected_bytes: descriptor
            .storage
            .as_ref()
            .map(|storage| storage.encoded_len),
        fallback_page: None,
    };
    let mut config = GaussianLodPackageConfig::default();
    config.persistent_cache_root = Some(
        package
            .root
            .join("persistent-cache")
            .to_string_lossy()
            .into_owned(),
    );
    config.persistent_cache_namespace = Some("http-offline-fixture".to_owned());
    config.persistent_cache_max_entries = 32;
    let requested = GaussianStreamingSettings {
        persistent_cache: true,
        retry_limit: 0,
        retry_base_delay_seconds: 0.0,
        max_compressed_cache_bytes: 8 * 1024 * 1024,
        ..default()
    };
    let effective = package_streaming_settings(&requested).unwrap();
    assert!(effective.persistent_cache);

    let mut manager = GaussianLodPackageManager::default();
    let mut first = package_page_transport(
        &package.manifest,
        &source,
        &config,
        &effective,
        &mut manager.caches,
    )
    .unwrap();
    let expected = poll_package_transport(&mut first, request);
    assert_eq!(request_count.load(Ordering::Acquire), 1);
    let (range_start, range_len) = descriptor.storage.as_ref().unwrap().byte_range.unwrap();
    assert_eq!(
        server.ranges.lock().unwrap().as_slice(),
        [Some((range_start, range_start + range_len - 1))],
        "the HTTP package transport must consume the shard range from the manifest"
    );

    let mut second = package_page_transport(
        &package.manifest,
        &source,
        &config,
        &effective,
        &mut manager.caches,
    )
    .unwrap();
    assert!(Arc::ptr_eq(
        first.shared_native_cache_service().unwrap(),
        second.shared_native_cache_service().unwrap(),
    ));
    assert_eq!(manager.caches.len(), 1);

    let mut conflicting = config.clone();
    conflicting.persistent_cache_max_entries += 1;
    assert!(matches!(
        package_page_transport(
            &package.manifest,
            &source,
            &conflicting,
            &effective,
            &mut manager.caches,
        ),
        Err(GaussianLodPackageError::PersistentCacheConfigConflict { .. })
    ));

    // The second package remains cache-backed after the shared origin goes
    // offline and must not issue another request.
    drop(server);
    let actual = poll_package_transport(&mut second, request);
    assert_eq!(actual, expected);
    assert_eq!(request_count.load(Ordering::Acquire), 1);
    drop(first);
    drop(second);
    manager.prune_unused_caches();
    assert!(manager.caches.is_empty());
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
#[test]
fn corrupt_http_package_page_reaches_preprocessor_without_codec_retry() {
    let package = write_native_test_package(false);
    let descriptor = package.manifest.pages.first().unwrap().clone();
    let storage = descriptor.storage.as_ref().unwrap();
    let (range_start, range_len) = storage.byte_range.unwrap();
    let shard_path = package.root.join(&storage.uri);
    let mut shard = std::fs::read(&shard_path).unwrap();
    let last_byte = range_start
        .checked_add(range_len)
        .and_then(|end| end.checked_sub(1))
        .and_then(|index| usize::try_from(index).ok())
        .expect("fixture page range must fit memory");
    shard[last_byte] ^= 0x5a;
    std::fs::write(&shard_path, shard).unwrap();

    let server = LocalPackageHttpServer::start(package.root.clone());
    let request_count = server.requests.clone();
    let source = GaussianLodPackageSource::url(server.base_url());
    let streaming = GaussianStreamingSettings {
        persistent_cache: true,
        retry_limit: 3,
        retry_base_delay_seconds: 0.0,
        ..default()
    };
    let mut config = GaussianLodPackageConfig::default();
    config.persistent_cache_root = Some(
        package
            .root
            .join("corrupt-handoff-cache")
            .to_string_lossy()
            .into_owned(),
    );
    config.persistent_cache_namespace = Some("corrupt-handoff".to_owned());
    let mut manager = GaussianLodPackageManager::default();
    let mut transport = package_page_transport(
        &package.manifest,
        &source,
        &config,
        &streaming,
        &mut manager.caches,
    )
    .unwrap();
    let request = PageRequest {
        page_id: descriptor.id,
        priority: crate::stream::transport::PageRequestPriority::fallback_critical(u32::MAX),
        expected_bytes: Some(storage.encoded_len),
        fallback_page: None,
    };
    let payload = poll_package_transport(&mut transport, request);

    assert_eq!(
        request_count.load(Ordering::Acquire),
        1,
        "package HTTP must not synchronously decode and retry encoded page bytes"
    );
    let mut limits = LodCodecLimits::default();
    limits.max_page_bytes = limits.max_page_bytes.max(storage.encoded_len);
    limits.max_page_gaussians = descriptor.gaussian_count;
    let mut preprocessor = LodPagePreprocessor::new_cooperative_for_tests(1).unwrap();
    preprocessor
        .submit(LodPagePreprocessInput {
            request,
            payload,
            descriptor: descriptor.clone(),
            limits,
            max_encoded_page_bytes: streaming.effective_max_encoded_page_bytes(),
            support_sigma: package.manifest.build.settings.support_sigma,
        })
        .unwrap();
    let full_page_budget = NonZeroU32::new(u32::MAX).unwrap();
    preprocessor.advance(1, full_page_budget);
    preprocessor.advance(2, full_page_budget);
    let output = preprocessor.take_ready(descriptor.id).unwrap();
    assert!(matches!(
        output.result,
        Err(LodPagePreprocessError::Codec(_))
    ));
    assert_eq!(
        request_count.load(Ordering::Acquire),
        1,
        "preprocess rejection must not create a second HTTP retry layer"
    );

    transport.invalidate_cached_page(descriptor.id).unwrap();
    for _ in 0..10_000 {
        if transport.maintain_cache().unwrap() {
            break;
        }
        std::thread::yield_now();
    }
    assert!(transport.maintain_cache().unwrap());
    let _ = poll_package_transport(&mut transport, request);
    assert_eq!(
        request_count.load(Ordering::Acquire),
        2,
        "a preprocess-rejected cache entry must be evicted before retry"
    );
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
#[test]
fn package_runtime_invalidates_rejected_cache_before_bounded_retry() {
    let package = write_native_test_package(false);
    let root_node = package
        .manifest
        .nodes
        .iter()
        .find(|node| node.id == package.manifest.roots[0])
        .unwrap();
    let root_page = root_node.representation.page;
    let descriptor = package
        .manifest
        .pages
        .iter()
        .find(|descriptor| descriptor.id == root_page)
        .unwrap()
        .clone();
    let storage = descriptor.storage.as_ref().unwrap();
    let (range_start, range_len) = storage.byte_range.unwrap();
    let shard_path = package.root.join(&storage.uri);
    let canonical_shard = std::fs::read(&shard_path).unwrap();
    let mut corrupt_shard = canonical_shard.clone();
    let last_byte = range_start
        .checked_add(range_len)
        .and_then(|end| end.checked_sub(1))
        .and_then(|index| usize::try_from(index).ok())
        .expect("fixture page range must fit memory");
    corrupt_shard[last_byte] ^= 0x5a;
    std::fs::write(&shard_path, &corrupt_shard).unwrap();

    let cache_root = package.root.join("preprocess-retry-order-cache");
    let streaming = GaussianStreamingSettings {
        persistent_cache: true,
        retry_limit: 0,
        retry_base_delay_seconds: 0.0,
        ..default()
    };
    let mut cache_config = GaussianLodPackageConfig::default();
    cache_config.persistent_cache_root = Some(cache_root.to_string_lossy().into_owned());
    cache_config.persistent_cache_namespace = Some("preprocess-retry-order".to_owned());
    cache_config.streaming = streaming.clone();
    let source =
        GaussianLodPackageSource::native_directory(package.root.to_string_lossy().into_owned());
    let mut seed_manager = GaussianLodPackageManager::default();
    let mut seed_transport = package_page_transport(
        &package.manifest,
        &source,
        &cache_config,
        &streaming,
        &mut seed_manager.caches,
    )
    .unwrap();
    let request = PageRequest {
        page_id: root_page,
        priority: crate::stream::transport::PageRequestPriority::fallback_critical(u32::MAX),
        expected_bytes: Some(storage.encoded_len),
        fallback_page: None,
    };
    let corrupt_payload = poll_package_transport(&mut seed_transport, request);
    assert!(
        decode_page_with_descriptor(
            &corrupt_payload.bytes,
            &descriptor,
            LodCodecLimits::default(),
        )
        .is_err(),
        "the seeded cache record must pass encoded-cache integrity but fail preprocessing"
    );
    std::fs::write(&shard_path, &canonical_shard).unwrap();

    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    {
        let mut config = world.resource_mut::<GaussianLodPackageConfig>();
        config.persistent_cache_root = cache_config.persistent_cache_root.clone();
        config.persistent_cache_namespace = cache_config.persistent_cache_namespace.clone();
        config.streaming = streaming;
    }
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let mut observed_rejection = false;
    for _ in 0..2048 {
        run_package_frame(&mut schedule, &mut world, cloud);
        let manager = world.resource::<GaussianLodPackageManager>();
        let Some(state) = manager.clouds.get(&cloud) else {
            continue;
        };
        assert_eq!(
            state.runtime_streaming.retry_limit, 0,
            "the regression must exercise the zero ordinary-retry budget"
        );
        let runtime = state.runtime.lock().unwrap();
        if state.preprocess_cache_repairs.contains(&root_page) {
            assert_eq!(
                runtime.page_attempts(root_page),
                None,
                "the cache-repair attempt must remain queued until the next frame"
            );
            assert!(!runtime.is_terminal_failure(root_page));
            observed_rejection = true;
            break;
        }
    }
    assert!(
        observed_rejection,
        "the full package runtime must observe the seeded preprocessing rejection"
    );

    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.manifest.quality.coarsest_gaussian_count as u32,
    );
    let manager = world.resource::<GaussianLodPackageManager>();
    let state = &manager.clouds[&cloud];
    let runtime = state.runtime.lock().unwrap();
    assert!(runtime.page_preprocess_error(root_page).is_none());
    assert!(!runtime.is_terminal_failure(root_page));
    assert!(!state.preprocess_cache_repairs.contains(&root_page));
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn mirror_current_package_replacement_can_activate_before_next_main_poll() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, true, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let coarse_count = package.manifest.quality.coarsest_gaussian_count as u32;
    drive_package_to_active_count(&mut schedule, &mut world, cloud, camera, coarse_count);
    let (visible_ranges, visible_slots, debug_page, debug_slot) = {
        let mut manager = world.resource_mut::<GaussianLodPackageManager>();
        let state = manager.clouds.get_mut(&cloud).unwrap();
        let range = *state
            .visible_ranges
            .first()
            .expect("the coarse package cut references a physical page");
        let debug = state.debug.as_mut().expect("debug annotations are enabled");
        debug.atlas.clear_slot(range.slot).unwrap();
        assert_eq!(debug.atlas.page(range.slot), None);
        (
            state.visible_ranges.clone(),
            state.visible_slots.clone(),
            range.page,
            range.slot,
        )
    };

    // A second identical view creates a distinct pending candidate set whose
    // complete range union is already resident in the retained coarse cut.
    // Real render work can therefore pass PREPARED and publish ACTIVE within
    // one render frame, before another main-world package update observes it.
    let second_camera = world
        .spawn((
            Camera {
                viewport: Some(bevy::camera::Viewport {
                    physical_size: UVec2::new(1280, 720),
                    ..default()
                }),
                ..default()
            },
            Projection::Perspective(default()),
            GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 5.0)),
            crate::GaussianCamera::default(),
        ))
        .id();
    mark_package_cloud_visible(&mut world, second_camera, cloud);

    let pending_phases = (0..64)
        .find_map(|_| {
            // Run only the package update: the ordinary CPU helper deliberately
            // inserts a PREPARED-only frame and would mask this race.
            world.insert_resource(LodAtlasUploadQueue::default());
            schedule.run(&mut world);
            std::thread::yield_now();

            let manager = world.resource::<GaussianLodPackageManager>();
            let state = manager.clouds.get(&cloud)?;
            let pending = state.pending.as_ref()?;
            (pending.len() == 2).then(|| {
                assert!(
                    pending
                        .by_camera
                        .values()
                        .flat_map(LodRenderCandidate::render_ranges)
                        .all(|range| state.mirror.is_range_current(*range)),
                    "the regression requires a wholly mirror-current replacement"
                );
                let staged = state
                    .staged
                    .as_ref()
                    .expect("every extracted pending cut owns staged state");
                assert!(staged.complete);
                assert!(staged.materializations.is_empty());
                assert_eq!(state.visible_ranges, visible_ranges);
                assert_eq!(state.visible_slots, visible_slots);
                assert_eq!(
                    state
                        .debug
                        .as_ref()
                        .expect("debug annotations remain enabled")
                        .atlas
                        .page(debug_slot),
                    None,
                    "mirror-current staging must not publish pending debug metadata into the retained cut"
                );
                pending
                    .by_camera
                    .values()
                    .map(|candidate| Arc::clone(&candidate.phase))
                    .collect::<Vec<_>>()
            })
        })
        .expect("an identical second view must produce an all-current pending cut");
    assert_eq!(
        world.resource::<LodAtlasUploadQueue>().queued_slot_count(),
        0,
        "mirror-current staging must not enqueue redundant atlas uploads"
    );

    // Model same-frame render preparation, compaction, and radix publication.
    for phase in &pending_phases {
        phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    let blocked_target = AtlasSlot {
        index: debug_slot.index,
        generation: debug_slot.generation.wrapping_add(1).max(1),
    };
    {
        let mut manager = world.resource_mut::<GaussianLodPackageManager>();
        let state = manager.clouds.get_mut(&cloud).unwrap();
        let staged = state.staged.as_mut().unwrap();
        staged
            .debug
            .pending
            .push_front((debug_page, blocked_target));
        staged.debug.targets.insert((debug_page, blocked_target));
        staged.debug.complete = false;
        state.debug.as_mut().unwrap().atlas.set_complete(true);
    }
    world.insert_resource(LodAtlasUploadQueue::default());
    schedule.run(&mut world);

    {
        let mut manager = world.resource_mut::<GaussianLodPackageManager>();
        let state = manager.clouds.get_mut(&cloud).unwrap();
        assert!(state.pending.is_some());
        assert!(state.staged.is_some());
        assert!(
            !state.debug.as_ref().unwrap().atlas.metadata().is_complete(),
            "an active replacement must gate retained debug metadata while its provenance is incomplete"
        );
        let staged = state.staged.as_mut().unwrap();
        assert_eq!(
            staged.debug.pending.pop_front(),
            Some((debug_page, blocked_target))
        );
        staged.debug.targets.remove(&(debug_page, blocked_target));
        staged.debug.complete = staged.debug.pending.is_empty();
    }
    world.insert_resource(LodAtlasUploadQueue::default());
    schedule.run(&mut world);

    let manager = world.resource::<GaussianLodPackageManager>();
    let state = &manager.clouds[&cloud];
    assert!(state.pending.is_none());
    assert!(state.staged.is_none());
    assert_eq!(state.current.as_ref().unwrap().len(), 2);
    assert!(package_candidate_set_is_active(
        state.current.as_ref().unwrap()
    ));
    assert_eq!(state.active_gaussians, u64::from(coarse_count));
    assert_eq!(state.visible_ranges, visible_ranges);
    assert_eq!(state.visible_slots, visible_slots);
    let debug = state
        .debug
        .as_ref()
        .expect("debug annotations remain enabled");
    assert_eq!(debug.atlas.page(debug_slot), Some(debug_page));
    assert!(debug.atlas.metadata().is_complete());
    assert!(state.last_failure.is_none());
    assert_eq!(package_status_phase(state), GaussianLodPackagePhase::Active);
    assert_eq!(
        world.resource::<LodAtlasUploadQueue>().queued_slot_count(),
        0
    );
}

#[cfg(lod_render_path)]
#[test]
fn retained_debug_replacement_stages_before_sync_and_radix_activation() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, true, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let coarse_count = package.manifest.quality.coarsest_gaussian_count as u32;
    drive_package_to_active_count(&mut schedule, &mut world, cloud, camera, coarse_count);
    let (current_phase, current_ranges, visible_ranges, visible_slots) = {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        let current = state.current.as_ref().expect("retained current cut");
        let candidate = current.get(camera).expect("retained camera candidate");
        assert!(candidate.render_is_active());
        (
            Arc::clone(&candidate.phase),
            candidate.render_ranges().to_vec(),
            state.visible_ranges.clone(),
            state.visible_slots.clone(),
        )
    };

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    let pending_phases = (0..2048)
        .find_map(|_| {
            // Drive only package orchestration. Render acknowledgement is
            // injected below so the test can observe the exact PREPARED
            // boundary rather than the generic CPU helper auto-activating it.
            world.insert_resource(LodAtlasUploadQueue::default());
            schedule.run(&mut world);
            std::thread::yield_now();
            let manager = world.resource::<GaussianLodPackageManager>();
            let state = manager.clouds.get(&cloud)?;
            let pending = state.pending.as_ref()?;
            let staged = state.staged.as_ref().expect("pending owns staged cut");
            assert!(
                !staged.complete || !staged.debug.complete,
                "the regression requires bounded page or debug staging work"
            );
            Some(
                pending
                    .by_camera
                    .values()
                    .map(|candidate| Arc::clone(&candidate.phase))
                    .collect::<Vec<_>>(),
            )
        })
        .expect("higher quality request must publish a retained replacement");

    for phase in &pending_phases {
        phase.store(
            crate::render::lod::retained_candidate_preparation_phase(true, true, false, true, true),
            Ordering::Release,
        );
        assert_eq!(phase.load(Ordering::Acquire), LOD_RENDER_PREPARED);
    }

    let mut observed_page_progress = false;
    let mut observed_debug_progress = false;
    let mut previous = {
        let manager = world.resource::<GaussianLodPackageManager>();
        let staged = manager.clouds[&cloud].staged.as_ref().unwrap();
        (staged.next_materialization, staged.debug.pending.len())
    };
    for _ in 0..128 {
        world.insert_resource(LodAtlasUploadQueue::default());
        schedule.run(&mut world);
        std::thread::yield_now();
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert_eq!(
            current_phase.load(Ordering::Acquire),
            LOD_RENDER_ACTIVE,
            "bounded replacement staging revoked the retained current output"
        );
        assert_eq!(
            state
                .current
                .as_ref()
                .unwrap()
                .get(camera)
                .unwrap()
                .render_ranges(),
            current_ranges,
            "bounded replacement staging changed the retained logical cut"
        );
        assert_eq!(state.visible_ranges, visible_ranges);
        assert_eq!(state.visible_slots, visible_slots);
        assert!(
            pending_phases
                .iter()
                .all(|phase| phase.load(Ordering::Acquire) == LOD_RENDER_PREPARED)
        );
        let staged = state.staged.as_ref().expect("staging remains owned");
        let next = (staged.next_materialization, staged.debug.pending.len());
        observed_page_progress |= next.0 > previous.0;
        observed_debug_progress |= next.1 < previous.1;
        previous = next;
        if staged.complete && staged.debug.complete {
            break;
        }
    }
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        let staged = state.staged.as_ref().expect("prepared staged cut");
        assert!(staged.complete && staged.debug.complete);
        assert!(
            observed_page_progress || observed_debug_progress,
            "PREPARED did not advance bounded package/debug staging"
        );
        assert!(state.pending.is_some());
        assert!(
            pending_phases
                .iter()
                .all(|phase| phase.load(Ordering::Acquire) == LOD_RENDER_PREPARED)
        );
    }

    // Model the one production publication point after the now-ready debug
    // binding, synchronized descriptors, compaction, and radix output.
    for phase in &pending_phases {
        assert!(crate::render::lod::publish_bridge_activation_after_radix(
            phase
        ));
    }
    world.insert_resource(LodAtlasUploadQueue::default());
    schedule.run(&mut world);

    let manager = world.resource::<GaussianLodPackageManager>();
    let state = &manager.clouds[&cloud];
    assert!(state.pending.is_none());
    assert!(state.staged.is_none());
    assert!(package_candidate_set_is_active(
        state.current.as_ref().expect("replacement committed")
    ));
    assert_ne!(state.visible_ranges, visible_ranges);
    assert!(state.last_failure.is_none());
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn retired_resident_package_slots_are_reused_without_upload() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let coarse_count = package.manifest.quality.coarsest_gaussian_count as u32;
    drive_package_to_active_count(&mut schedule, &mut world, cloud, camera, coarse_count);
    let (coarse_ranges, coarse_slots) = {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        (state.visible_ranges.clone(), state.visible_slots.clone())
    };

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.source_count as u32,
    );
    {
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        let retired = coarse_slots
            .iter()
            .filter(|(index, slot)| state.visible_slots.get(index) != Some(slot))
            .collect::<Vec<_>>();
        assert!(
            !retired.is_empty(),
            "the exact cut must retire at least one coarse visible slot"
        );
        for (&index, &slot) in retired {
            let page = coarse_ranges
                .iter()
                .find(|range| range.slot == slot)
                .map(|range| range.page)
                .expect("every coarse visible slot backs a coarse range");
            assert!(
                state.mirror.is_page_current(page, slot),
                "retired runtime-resident slot {index} lost its CPU mirror proof"
            );
            assert!(
                state
                    .transient_atlas
                    .snapshot_slot(LodAtlasSlotUpload {
                        atlas: state.atlas.id(),
                        slot,
                        gaussians_per_slot: state.plan.gaussians_per_slot,
                    })
                    .is_ok(),
                "retired runtime-resident slot {index} lost its recovery payload"
            );
        }
    }

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 0.0;
    let pending_phases = (0..64)
        .find_map(|_| {
            world.insert_resource(LodAtlasUploadQueue::default());
            schedule.run(&mut world);
            std::thread::yield_now();
            let queued = world
                .resource::<LodAtlasUploadQueue>()
                .queued_slot_count();
            let manager = world.resource::<GaussianLodPackageManager>();
            let state = manager.clouds.get(&cloud)?;
            let pending = state.pending.as_ref()?;
            let candidate = pending.get(camera)?;
            (candidate.rendered_candidate_count() == coarse_count).then(|| {
                assert_eq!(
                    normalize_package_ranges(candidate.render_ranges()),
                    normalize_package_ranges(&coarse_ranges),
                    "coarsening must reuse the exact allocator generations retained from the coarse cut"
                );
                let staged = state.staged.as_ref().unwrap();
                assert!(staged.complete);
                assert!(staged.materializations.is_empty());
                assert_eq!(
                    queued, 0,
                    "a retained cache-resident coarse cut must require no upload"
                );
                pending
                    .by_camera
                    .values()
                    .map(|candidate| Arc::clone(&candidate.phase))
                    .collect::<Vec<_>>()
            })
        })
        .expect("the retained coarse cut must be selectable without streaming");

    // A generation-current GPU cache can complete this replacement in the
    // same render frame; commit it without the CPU helper's PREPARED-only turn.
    for phase in pending_phases {
        phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    world.insert_resource(LodAtlasUploadQueue::default());
    schedule.run(&mut world);

    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert!(state.pending.is_none());
    assert!(state.last_failure.is_none());
    assert_eq!(state.active_gaussians, u64::from(coarse_count));
    assert_eq!(state.visible_ranges, coarse_ranges);
    assert_eq!(state.visible_slots, coarse_slots);
    assert_eq!(
        world.resource::<LodAtlasUploadQueue>().queued_slot_count(),
        0
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn automatic_native_package_bridge_streams_rebuilds_and_cleans_up() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, manifest_handle) =
        package_test_world(&package, settings, true, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let coarse_count = package.manifest.quality.coarsest_gaussian_count as u32;
    let coarse_uploads =
        drive_package_to_active_count(&mut schedule, &mut world, cloud, camera, coarse_count);
    assert!(coarse_uploads > 0);
    let (
        first_atlas,
        first_plan,
        coarse_visible_ranges,
        coarse_visible_slots,
        coarse_atlas_snapshot,
    ) = {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        let debug = state.debug.as_ref().expect("debug annotations are enabled");
        assert_eq!(
            state.current_page_leases,
            package_candidate_pages(state.current.as_ref().unwrap())
        );
        assert!(
            debug
                .index
                .descriptor(package.manifest.pages[0].id)
                .is_some()
        );
        (
            state.atlas.clone(),
            state.plan,
            state.visible_ranges.clone(),
            state.visible_slots.clone(),
            sparse_package_atlas_snapshot(state),
        )
    };
    assert!(first_plan.physical_gaussians <= 4096);
    assert!(first_plan.physical_bytes <= 64 * 1024 * 1024);
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&first_atlas)
            .is_none(),
        "the package fast path must not insert a dense CPU atlas asset"
    );
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert_eq!(
            state.transient_atlas.physical_gaussians(),
            first_plan.physical_gaussians
        );
        assert_eq!(
            state.transient_atlas.materialized_gaussian_count().unwrap(),
            state.visible_slots.len() * first_plan.gaussians_per_slot as usize
        );
    }
    assert!(world.get::<LodDebugMetadata>(cloud).is_some());
    assert!(
        world
            .resource::<LodAtlasUploadQueue>()
            .queued_slots()
            .all(|upload| {
                upload.atlas == first_atlas.id()
                    && upload.slot.index < first_plan.slot_count
                    && upload.gaussians_per_slot == first_plan.gaussians_per_slot
            })
    );

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    let pending_uploads = (0..2048)
        .find_map(|_| {
            let queued = run_package_frame(&mut schedule, &mut world, cloud);
            let manager = world.resource::<GaussianLodPackageManager>();
            manager.clouds[&cloud].pending.is_some().then_some(queued)
        })
        .expect("quality churn must eventually stage a replacement cut");
    assert_eq!(
        pending_uploads, 0,
        "staging must retain the previous atlas instead of rewriting root"
    );
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.current.is_some());
        assert!(state.pending.is_some());
        assert!(!state.root_fallback);
        assert_eq!(state.active_gaussians, u64::from(coarse_count));
        assert_eq!(state.visible_ranges, coarse_visible_ranges);
        assert_eq!(state.visible_slots, coarse_visible_slots);
        assert_eq!(
            state.current_page_leases,
            package_candidate_pages(state.current.as_ref().unwrap())
        );
        assert_eq!(sparse_package_atlas_snapshot(state), coarse_atlas_snapshot);
    }

    let staged_uploads = run_package_frame(&mut schedule, &mut world, cloud);
    assert!(staged_uploads > 0);
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.staged.is_some());
        assert_eq!(state.visible_ranges, coarse_visible_ranges);
        assert_eq!(state.visible_slots, coarse_visible_slots);
        let staged_snapshot = sparse_package_atlas_snapshot(state);
        for range in &coarse_visible_ranges {
            let start = range.physical_start as usize;
            let end = range.end().unwrap() as usize;
            assert_eq!(
                &staged_snapshot[start..end],
                &coarse_atlas_snapshot[start..end],
                "additive staging must not mutate any index used by the current cut"
            );
        }
    }

    let exact_commit_uploads = drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.source_count as u32,
    );
    assert!(staged_uploads + exact_commit_uploads > 1);
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.pending.is_none());
        assert!(state.current.is_some());
        assert!(!state.root_fallback);
        assert_ne!(state.visible_ranges, coarse_visible_ranges);
        assert_eq!(
            state.current_page_leases,
            package_candidate_pages(state.current.as_ref().unwrap())
        );
        let retired = coarse_visible_slots
            .keys()
            .filter(|index| !state.visible_slots.contains_key(index))
            .copied()
            .collect::<Vec<_>>();
        for slot_index in retired {
            assert!(
                state
                    .mirror
                    .materialized_slots()
                    .iter()
                    .any(|slot| slot.index == slot_index),
                "a retired but runtime-resident slot {slot_index} must retain its reusable CPU mirror proof"
            );
        }
    }
    let status = world.get::<GaussianLodPackageStatus>(cloud).unwrap();
    assert_eq!(status.active_gaussians, package.source_count as u64);
    assert_eq!(status.phase, GaussianLodPackagePhase::Active);

    world.clear_trackers();
    assert_eq!(run_package_frame(&mut schedule, &mut world, cloud), 0);

    // A runtime-structural budget change must retire the old atlas even
    // when the manifest/source/config handles are unchanged.
    let old_atlas = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    world
        .get_mut::<GaussianLodSettings>(cloud)
        .unwrap()
        .budgets
        .max_pending_requests -= 1;
    run_package_frame(&mut schedule, &mut world, cloud);
    let structural_atlas = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(structural_atlas.id(), old_atlas.id());
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&old_atlas)
            .is_none()
    );

    // A same-ID manifest reload is a new package generation and must not
    // inherit atlas residency or render handshakes from the old asset.
    let reloaded_manifest = package.manifest.clone();
    *world
        .resource_mut::<Assets<GaussianLodAsset>>()
        .get_mut_untracked(&manifest_handle)
        .unwrap() = GaussianLodAsset::new(reloaded_manifest).unwrap();
    world
        .resource_mut::<Messages<AssetEvent<GaussianLodAsset>>>()
        .write(AssetEvent::Modified {
            id: manifest_handle.id(),
        });
    run_package_frame(&mut schedule, &mut world, cloud);
    let reload_atlas = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(reload_atlas.id(), structural_atlas.id());
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&structural_atlas)
            .is_none()
    );

    // Removal and re-addition under the same AssetId are also generation
    // changes. The removed package must release its atlas and remain in a
    // loading state until the replacement asset is present.
    let replacement_manifest = package.manifest.clone();
    world
        .resource_mut::<Assets<GaussianLodAsset>>()
        .remove(manifest_handle.id());
    world
        .resource_mut::<Messages<AssetEvent<GaussianLodAsset>>>()
        .write(AssetEvent::Removed {
            id: manifest_handle.id(),
        });
    run_package_frame(&mut schedule, &mut world, cloud);
    assert!(
        world
            .resource::<GaussianLodPackageManager>()
            .clouds
            .is_empty()
    );
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&reload_atlas)
            .is_none()
    );
    assert!(world.get::<PlanarGaussian3dHandle>(cloud).is_none());
    assert_eq!(
        world.get::<GaussianLodPackageStatus>(cloud).unwrap().phase,
        GaussianLodPackagePhase::Loading
    );

    world
        .resource_mut::<Assets<GaussianLodAsset>>()
        .insert(
            manifest_handle.id(),
            GaussianLodAsset::new(replacement_manifest).unwrap(),
        )
        .expect("removed manifest ID can be reinserted");
    world
        .resource_mut::<Messages<AssetEvent<GaussianLodAsset>>>()
        .write(AssetEvent::Added {
            id: manifest_handle.id(),
        });
    run_package_frame(&mut schedule, &mut world, cloud);
    let readded_atlas = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(readded_atlas.id(), reload_atlas.id());
    assert!(
        world
            .resource::<GaussianLodPackageManager>()
            .clouds
            .contains_key(&cloud)
    );
    let (readded_transient_owner, readded_transient_ticket, readded_generation, readded_physical) = {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        (
            std::ptr::from_ref(&state.transient_atlas) as usize,
            std::ptr::from_ref(state.transient_atlas.ticket()) as usize,
            state.transient_atlas.ticket().generation(),
            state.transient_atlas.physical_gaussians(),
        )
    };
    {
        let registry = world.resource::<LodTransientAtlasRegistry>();
        assert!(registry.contains(readded_atlas.id()));
        assert_eq!(
            registry.physical_gaussians(readded_atlas.id().untyped()),
            Some(readded_physical)
        );
    }

    // Debug metadata is genuinely lazy and presentation-only. Disabling the
    // only metadata user drops its sidecar without replacing the package
    // runtime, physical atlas, or retained render state.
    world
        .get_mut::<CloudSettings>(cloud)
        .unwrap()
        .lod_debug
        .apply_preset(LodDebugPreset::Off);
    run_package_frame(&mut schedule, &mut world, cloud);
    let disabled_atlas = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    assert_eq!(disabled_atlas.id(), readded_atlas.id());
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&readded_atlas)
            .is_none(),
        "the persistent package fast path must remain outside the dense CPU asset collection"
    );
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert_eq!(
            std::ptr::from_ref(&state.transient_atlas) as usize,
            readded_transient_owner
        );
        assert_eq!(
            std::ptr::from_ref(state.transient_atlas.ticket()) as usize,
            readded_transient_ticket
        );
        assert_eq!(
            state.transient_atlas.ticket().generation(),
            readded_generation
        );
        assert_eq!(state.transient_atlas.physical_gaussians(), readded_physical);
    }
    {
        let registry = world.resource::<LodTransientAtlasRegistry>();
        assert!(registry.contains(readded_atlas.id()));
        assert_eq!(
            registry.physical_gaussians(readded_atlas.id().untyped()),
            Some(readded_physical)
        );
    }
    assert!(
        world.resource::<GaussianLodPackageManager>().clouds[&cloud]
            .debug
            .is_none()
    );
    assert!(world.get::<LodDebugMetadata>(cloud).is_none());

    // Package rendering requires the GPU radix compaction path. An
    // incompatible cloud fails visibly and releases package-owned state.
    world.get_mut::<CloudSettings>(cloud).unwrap().sort_mode = crate::sort::SortMode::None;
    run_package_frame(&mut schedule, &mut world, cloud);
    let status = world.get::<GaussianLodPackageStatus>(cloud).unwrap();
    assert_eq!(status.phase, GaussianLodPackagePhase::Failed);
    assert_eq!(
        status.failure.as_ref().map(LodOrchestrationFailure::code),
        Some(LodOrchestrationFailureCode::UnsupportedConfiguration)
    );
    assert!(
        status
            .error_detail()
            .is_some_and(|error| error.contains("UnsupportedSortMode"))
    );
    assert!(
        world
            .resource::<GaussianLodPackageManager>()
            .clouds
            .is_empty()
    );
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&disabled_atlas)
            .is_none()
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn debug_sidecar_toggle_preserves_active_package_and_primes_boundedly() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(1.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, true, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.source_count as u32,
    );
    assert_eq!(
        run_package_frame(&mut schedule, &mut world, cloud),
        0,
        "settling the exact active cut must not enqueue another Gaussian upload"
    );

    let package_identity = |world: &World| {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        let candidate_phases = state
            .current
            .as_ref()
            .unwrap()
            .by_camera
            .values()
            .map(|candidate| Arc::as_ptr(&candidate.phase) as usize)
            .collect::<Vec<_>>();
        let candidate_ranges = state
            .current
            .as_ref()
            .unwrap()
            .by_camera
            .iter()
            .map(|(&camera, candidate)| (camera, candidate.render_ranges().to_vec()))
            .collect::<Vec<_>>();
        let request_starts = state
            .runtime
            .lock()
            .unwrap()
            .transport_request_starts_for_test();
        (
            state.atlas.id(),
            std::ptr::from_ref(&state.runtime) as usize,
            Arc::as_ptr(&state.debug_index) as usize,
            candidate_phases,
            candidate_ranges,
            state.visible_ranges.clone(),
            state.visible_fallback_nodes.clone(),
            state.active_gaussians,
            request_starts,
        )
    };
    let initial_identity = package_identity(&world);
    let manifest_validations_before = LodDebugManifestIndex::manifest_validation_count_for_test();
    let (initial_page_bases, visible_target_work) = {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        let debug = state.debug.as_ref().expect("initial debug sidecar exists");
        let mut bases = debug
            .page_bases
            .iter()
            .map(|(&slot, basis)| (slot, basis.page, basis.records.as_ptr() as usize))
            .collect::<Vec<_>>();
        bases.sort_unstable_by_key(|basis| basis.0);
        assert!(!bases.is_empty(), "the active cut must prime page bases");
        let target_work = state
            .visible_ranges
            .iter()
            .map(|range| (range.page, range.slot))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|(page, _)| {
                if !state.visible_fallback_nodes.is_empty()
                    && debug.index.node_ids(page).is_some_and(|mut nodes| {
                        nodes.any(|node| state.visible_fallback_nodes.contains(&node))
                    })
                {
                    state.plan.gaussians_per_slot as usize
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        (bases, target_work)
    };
    let expected_residency_patch_records = visible_target_work.iter().sum::<usize>();
    assert_eq!(
        expected_residency_patch_records, 0,
        "the settled exact cut must reuse all-Resident bases without Residency patch work"
    );
    let cached_slot_frame_bound = visible_target_work
        .len()
        .div_ceil(PACKAGE_DEBUG_MAX_SLOTS_PER_FRAME);
    let expected_second_toggle_frames = {
        let mut frames = 0_usize;
        let mut work = PackageDebugPreparationWork::default();
        for &records in &visible_target_work {
            if !work.can_consume(records) {
                frames += 1;
                work = PackageDebugPreparationWork::default();
            }
            assert!(work.can_consume(records));
            work.consume(records);
        }
        frames + usize::from(work.slots > 0)
    };

    world
        .get_mut::<CloudSettings>(cloud)
        .unwrap()
        .lod_debug
        .apply_preset(LodDebugPreset::Off);
    assert_eq!(run_package_frame(&mut schedule, &mut world, cloud), 0);
    assert_eq!(package_identity(&world), initial_identity);
    assert!(
        world.resource::<GaussianLodPackageManager>().clouds[&cloud]
            .debug
            .is_none()
    );
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        let mut retained = state
            .retained_debug_page_bases
            .iter()
            .map(|(&slot, basis)| (slot, basis.page, basis.records.as_ptr() as usize))
            .collect::<Vec<_>>();
        retained.sort_unstable_by_key(|basis| basis.0);
        assert_eq!(retained, initial_page_bases);
    }
    assert!(world.get::<LodDebugMetadata>(cloud).is_none());

    world
        .get_mut::<CloudSettings>(cloud)
        .unwrap()
        .lod_debug
        .apply_preset(LodDebugPreset::Level);
    {
        let mut manager = world.resource_mut::<GaussianLodPackageManager>();
        let state = manager.clouds.get_mut(&cloud).unwrap();
        sync_package_debug_annotations(state, true).unwrap();
        let debug = state.debug.as_ref().unwrap();
        assert!(Arc::ptr_eq(&debug.index, &state.debug_index));
        let mut restored = debug
            .page_bases
            .iter()
            .map(|(&slot, basis)| (slot, basis.page, basis.records.as_ptr() as usize))
            .collect::<Vec<_>>();
        restored.sort_unstable_by_key(|basis| basis.0);
        assert_eq!(restored, initial_page_bases);
        assert!(state.retained_debug_page_bases.is_empty());
        let initialization_before = debug.initialization.len();
        assert!(initialization_before > 0);
        assert!(!debug.atlas.metadata().is_complete());
        let populated_before = debug
            .atlas
            .metadata()
            .sparse()
            .unwrap()
            .slots()
            .iter()
            .filter(|slot| slot.records().is_some())
            .count();

        let validation_count_before = state.debug_index.page_payload_validation_count_for_test();
        let gaussian_validation_count_before = crate::gaussian::formats::planar_3d_lod::
            gaussian_support_full_validation_count_for_test();
        let mut work = PackageDebugPreparationWork::default();
        advance_package_debug_initialization(state, &mut work).unwrap();
        let debug = state.debug.as_ref().unwrap();
        let populated_after = debug
            .atlas
            .metadata()
            .sparse()
            .unwrap()
            .slots()
            .iter()
            .filter(|slot| slot.records().is_some())
            .count();
        assert!(
            work.slots <= PACKAGE_DEBUG_MAX_SLOTS_PER_FRAME,
            "debug initialization exceeded its slot work cap"
        );
        assert!(
            work.records <= PACKAGE_DEBUG_MAX_RECORDS_PER_FRAME
                || (work.slots == 1
                    && state.plan.gaussians_per_slot as usize
                        > PACKAGE_DEBUG_MAX_RECORDS_PER_FRAME),
            "debug initialization exceeded its record work cap"
        );
        assert_eq!(
            work.regenerated_records, 0,
            "restoring cached all-Resident Arcs must regenerate zero basis records"
        );
        assert!(
            work.records <= expected_residency_patch_records,
            "only bounded cut-dependent Residency patches may consume record work"
        );
        assert_eq!(
            initialization_before.saturating_sub(debug.initialization.len()),
            populated_after.saturating_sub(populated_before)
        );
        assert_eq!(
            state.debug_index.page_payload_validation_count_for_test(),
            validation_count_before,
            "trusted runtime-decoded pages must not be revalidated or rehashed during debug initialization"
        );
        assert_eq!(
            crate::gaussian::formats::planar_3d_lod::
                gaussian_support_full_validation_count_for_test(),
            gaussian_validation_count_before,
            "trusted runtime-decoded annotations must not rescan Gaussian SH/rotation/opacity fields"
        );
        assert!(debug.page_bases.len() <= state.plan.slot_count as usize);
        assert_eq!(
            debug.atlas.metadata().is_complete(),
            debug.initialization.is_empty()
        );

        // Lazy initialization and a simultaneous camera/cut replacement share
        // this one frame's budget. Exhaust the remaining allowance and prove a
        // staged Residency variant cannot start a second 32K-record wave.
        let slot_records = state.plan.gaussians_per_slot as usize;
        while work.can_consume(slot_records) {
            work.consume(slot_records);
        }
        let ranges = state.visible_ranges.clone();
        let mut pending_fallback = state.visible_fallback_nodes.clone();
        pending_fallback.insert(ranges[0].node);
        let mut staged = prepare_package_staged_cut(state, &ranges, &pending_fallback).unwrap();
        let pending_before = staged.debug.pending.len();
        assert!(pending_before > 0);
        advance_package_staged_debug_preparation(state, &mut staged, &mut work).unwrap();
        assert_eq!(
            staged.debug.pending.len(),
            pending_before,
            "staged debug preparation must not receive a fresh budget after lazy initialization"
        );
    }
    assert_eq!(
        world.resource::<LodAtlasUploadQueue>().queued_slot_count(),
        0,
        "debug initialization must not enqueue Gaussian atlas writes"
    );
    assert_eq!(run_package_frame(&mut schedule, &mut world, cloud), 0);
    assert_eq!(package_identity(&world), initial_identity);
    assert!(world.get::<LodDebugMetadata>(cloud).is_some());

    world
        .get_mut::<CloudSettings>(cloud)
        .unwrap()
        .lod_debug
        .apply_preset(LodDebugPreset::Off);
    assert_eq!(run_package_frame(&mut schedule, &mut world, cloud), 0);
    assert!(
        world.resource::<GaussianLodPackageManager>().clouds[&cloud]
            .debug
            .is_none()
    );
    world
        .get_mut::<CloudSettings>(cloud)
        .unwrap()
        .lod_debug
        .apply_preset(LodDebugPreset::Page);
    let page_validations_before = world.resource::<GaussianLodPackageManager>().clouds[&cloud]
        .debug_index
        .page_payload_validation_count_for_test();
    let gaussian_validations_before =
        crate::gaussian::formats::planar_3d_lod::gaussian_support_full_validation_count_for_test();
    let mut second_toggle_frames = 0_usize;
    let mut second_toggle_records = 0_usize;
    let mut second_toggle_regenerated_records = 0_usize;
    {
        let mut manager = world.resource_mut::<GaussianLodPackageManager>();
        let state = manager.clouds.get_mut(&cloud).unwrap();
        sync_package_debug_annotations(state, true).unwrap();
        while !state.debug.as_ref().unwrap().initialization.is_empty() {
            let mut work = PackageDebugPreparationWork::default();
            advance_package_debug_initialization(state, &mut work).unwrap();
            second_toggle_frames += 1;
            second_toggle_records = second_toggle_records.saturating_add(work.records);
            second_toggle_regenerated_records =
                second_toggle_regenerated_records.saturating_add(work.regenerated_records);
            assert!(work.slots <= PACKAGE_DEBUG_MAX_SLOTS_PER_FRAME);
            assert!(
                work.slots > 0,
                "cached second-toggle initialization must make bounded slot progress"
            );
        }
        assert!(state.debug.as_ref().unwrap().atlas.metadata().is_complete());
        let mut restored = state
            .debug
            .as_ref()
            .unwrap()
            .page_bases
            .iter()
            .map(|(&slot, basis)| (slot, basis.page, basis.records.as_ptr() as usize))
            .collect::<Vec<_>>();
        restored.sort_unstable_by_key(|basis| basis.0);
        assert_eq!(restored, initial_page_bases);
    }
    assert_eq!(
        second_toggle_regenerated_records, 0,
        "the second Off-to-On toggle must not regenerate cached basis records"
    );
    assert_eq!(
        second_toggle_records, expected_residency_patch_records,
        "the second toggle may only perform the current cut's Residency patches"
    );
    assert_eq!(
        second_toggle_frames, expected_second_toggle_frames,
        "cached second-toggle initialization did not drain under the shared slot/record budget"
    );
    assert_eq!(
        second_toggle_frames, cached_slot_frame_bound,
        "cached all-Resident bases must drain at the 256-slot bound, independent of the 32K record-generation budget"
    );
    assert_eq!(run_package_frame(&mut schedule, &mut world, cloud), 0);
    let manager = world.resource::<GaussianLodPackageManager>();
    let state = &manager.clouds[&cloud];
    assert!(Arc::ptr_eq(
        &state.debug.as_ref().unwrap().index,
        &state.debug_index
    ));
    assert_eq!(package_identity(&world), initial_identity);
    assert!(
        world
            .get::<LodDebugMetadata>(cloud)
            .is_some_and(LodDebugMetadata::is_complete),
        "the second cached Off-to-On toggle must publish ready metadata"
    );
    assert_eq!(
        LodDebugManifestIndex::manifest_validation_count_for_test(),
        manifest_validations_before,
        "debug Off/On toggles must never revalidate the immutable package manifest"
    );
    assert_eq!(
        state.debug_index.page_payload_validation_count_for_test(),
        page_validations_before,
        "the second cached toggle must not revalidate or rehash decoded pages"
    );
    assert_eq!(
        crate::gaussian::formats::planar_3d_lod::gaussian_support_full_validation_count_for_test(),
        gaussian_validations_before,
        "the second cached toggle must not rescan full Gaussian payloads"
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn pending_residency_provenance_commits_atomically_with_the_visible_cut() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, true, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.manifest.quality.coarsest_gaussian_count as u32,
    );

    let mut manager = world.resource_mut::<GaussianLodPackageManager>();
    let state = manager.clouds.get_mut(&cloud).unwrap();
    let ranges = state.visible_ranges.clone();
    assert!(!ranges.is_empty());
    let targets = ranges
        .iter()
        .map(|range| (range.page, range.slot))
        .collect::<BTreeSet<_>>();
    let current_fallback = state.visible_fallback_nodes.clone();
    assert_eq!(current_fallback, state.current_fallback_nodes);
    let toggled_node = ranges[0].node;
    let mut pending_fallback = current_fallback.clone();
    if !pending_fallback.insert(toggled_node) {
        pending_fallback.remove(&toggled_node);
    }

    let debug = state.debug.as_ref().expect("debug annotations are enabled");
    assert!(targets.iter().all(|&(page, slot)| {
        debug
            .atlas
            .page_matches_indexed_node_residency(&debug.index, page, slot, |node| {
                if current_fallback.contains(&node) {
                    LodDebugResidency::AncestorFallback
                } else {
                    LodDebugResidency::Resident
                }
            })
    }));
    let metadata_before = debug.atlas.metadata();
    let sparse_before = metadata_before.sparse().unwrap();
    let identity_before = sparse_before.identity();
    let revisions_before = targets
        .iter()
        .map(|(_, slot)| sparse_before.slots()[slot.index as usize].revision())
        .collect::<Vec<_>>();
    let invariant_revisions_before = targets
        .iter()
        .map(|(_, slot)| sparse_before.slots()[slot.index as usize].invariant_revision())
        .collect::<Vec<_>>();

    let validation_count_before = state.debug_index.page_payload_validation_count_for_test();
    let mut staged = prepare_package_staged_cut(state, &ranges, &pending_fallback).unwrap();
    assert!(staged.complete);
    assert_eq!(state.visible_fallback_nodes, current_fallback);
    assert_eq!(
        staged.debug.targets.len(),
        staged.debug.pending.len() + staged.debug.prepared.len(),
        "staged debug target membership must be unique and indexed"
    );
    let debug = state.debug.as_ref().unwrap();
    assert!(
        debug.atlas.metadata().is_complete(),
        "pending-only provenance must not disable the fully prepared current debug epoch"
    );
    assert!(targets.iter().all(|&(page, slot)| {
        debug
            .atlas
            .page_matches_indexed_node_residency(&debug.index, page, slot, |node| {
                if current_fallback.contains(&node) {
                    LodDebugResidency::AncestorFallback
                } else {
                    LodDebugResidency::Resident
                }
            })
    }));
    let metadata_staged = debug.atlas.metadata();
    let sparse_staged = metadata_staged.sparse().unwrap();
    assert_eq!(sparse_staged.identity(), identity_before);
    assert_eq!(
        targets
            .iter()
            .map(|(_, slot)| sparse_staged.slots()[slot.index as usize].revision())
            .collect::<Vec<_>>(),
        revisions_before,
        "preparing a pending cut must not publish its residency provenance into current slots"
    );

    let mut preparation_frames = 0;
    while !staged.debug.complete {
        let mut work = PackageDebugPreparationWork::default();
        advance_package_staged_debug_preparation(state, &mut staged, &mut work).unwrap();
        assert!(work.slots <= PACKAGE_DEBUG_MAX_SLOTS_PER_FRAME);
        assert!(
            work.records <= PACKAGE_DEBUG_MAX_RECORDS_PER_FRAME
                || (work.slots == 1
                    && state.plan.gaussians_per_slot as usize
                        > PACKAGE_DEBUG_MAX_RECORDS_PER_FRAME)
        );
        assert!(
            work.slots > 0,
            "a decoded mirror-current fixture must advance staged debug preparation"
        );
        assert_eq!(
            staged.debug.targets.len(),
            staged.debug.pending.len() + staged.debug.prepared.len(),
            "moving a target from pending to prepared must preserve unique indexed membership"
        );
        let debug = state.debug.as_ref().unwrap();
        assert!(
            debug.atlas.metadata().is_complete(),
            "staged Arc construction must preserve current-epoch readiness"
        );
        let sparse = debug.atlas.metadata();
        let sparse = sparse.sparse().unwrap();
        assert_eq!(
            targets
                .iter()
                .map(|(_, slot)| sparse.slots()[slot.index as usize].revision())
                .collect::<Vec<_>>(),
            revisions_before,
            "staged Arc construction must not revise live sparse slots"
        );
        preparation_frames += 1;
        assert!(preparation_frames <= targets.len());
    }
    assert_eq!(
        state.debug_index.page_payload_validation_count_for_test(),
        validation_count_before,
        "staged debug preparation must trust pages already validated by the decoder"
    );
    assert!(state.debug.as_ref().unwrap().page_bases.len() <= state.plan.slot_count as usize);

    commit_package_staged_debug_annotations(
        state,
        &staged.ranges,
        &staged.fallback_nodes,
        &staged.debug,
    )
    .unwrap();
    publish_package_staged_cut(state, staged);
    assert_eq!(state.visible_fallback_nodes, pending_fallback);
    let debug = state.debug.as_ref().unwrap();
    assert!(debug.atlas.metadata().is_complete());
    assert!(targets.iter().all(|&(page, slot)| {
        debug
            .atlas
            .page_matches_indexed_node_residency(&debug.index, page, slot, |node| {
                if pending_fallback.contains(&node) {
                    LodDebugResidency::AncestorFallback
                } else {
                    LodDebugResidency::Resident
                }
            })
    }));
    let metadata_committed = debug.atlas.metadata();
    let sparse_committed = metadata_committed.sparse().unwrap();
    assert_eq!(sparse_committed.identity(), identity_before);
    assert_ne!(
        targets
            .iter()
            .map(|(_, slot)| sparse_committed.slots()[slot.index as usize].revision())
            .collect::<Vec<_>>(),
        revisions_before,
        "logical cut publication must expose a new sparse revision for changed residency provenance"
    );
    assert_eq!(
        targets
            .iter()
            .map(|(_, slot)| sparse_committed.slots()[slot.index as usize].invariant_revision())
            .collect::<Vec<_>>(),
        invariant_revisions_before,
        "a same-page logical commit may not invalidate candidate-invariant debug modes"
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn debug_page_basis_cache_replaces_the_entry_when_a_physical_slot_is_reused() {
    let built = build_planar_3d_lod(
        &LodTestScene::nested_octants(2).cloud(),
        GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 8,
            support_sigma: 3.0,
        },
    )
    .unwrap();
    assert!(built.pages.len() >= 2);
    let records_per_slot = built
        .manifest
        .pages
        .iter()
        .map(|page| page.gaussian_count)
        .max()
        .unwrap();
    let index = Arc::new(LodDebugManifestIndex::new(&built.manifest).unwrap());
    let mut debug = PackageDebugAnnotations {
        atlas: LodDebugAnnotationAtlas::new_sparse(1, records_per_slot).unwrap(),
        index: index.clone(),
        initialization: VecDeque::new(),
        page_bases: HashMap::new(),
    };
    let fallback_nodes = BTreeSet::new();
    let first_slot = AtlasSlot {
        index: 0,
        generation: 1,
    };
    let second_slot = AtlasSlot {
        index: 0,
        generation: 2,
    };
    let validations_before = index.page_payload_validation_count_for_test();

    let first = debug
        .prepared_page_records(
            &built.pages[0],
            first_slot,
            records_per_slot,
            &fallback_nodes,
        )
        .unwrap();
    assert_eq!(debug.page_bases.len(), 1);
    assert_eq!(debug.page_bases[&0].page, built.pages[0].id);

    let second = debug
        .prepared_page_records(
            &built.pages[1],
            second_slot,
            records_per_slot,
            &fallback_nodes,
        )
        .unwrap();
    assert_eq!(debug.page_bases.len(), 1);
    assert_eq!(debug.page_bases[&0].page, built.pages[1].id);
    assert!(!Arc::ptr_eq(&first, &second));
    let second_again = debug
        .prepared_page_records(
            &built.pages[1],
            second_slot,
            records_per_slot,
            &fallback_nodes,
        )
        .unwrap();
    assert!(Arc::ptr_eq(&second, &second_again));
    assert_eq!(
        index.page_payload_validation_count_for_test(),
        validations_before,
        "slot-basis creation must trust pages already validated by the decoder"
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn lost_gpu_generations_requeue_the_retained_current_cut_once_per_recovery() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.manifest.quality.coarsest_gaussian_count as u32,
    );
    let (phase, atlas, plan, visible_slots, current_pages) = {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        (
            Arc::clone(&state.current.as_ref().unwrap().get(camera).unwrap().phase),
            state.atlas.id(),
            state.plan,
            state.visible_slots.clone(),
            state.current_page_leases.clone(),
        )
    };
    assert!(!visible_slots.is_empty());

    // This is the render-world signal produced after device loss clears every
    // atlas generation proof. The CPU atlas and package-owned page leases are
    // still the immutable source for a bounded slot replay.
    phase.store(LOD_RENDER_WAITING, Ordering::Release);
    let recovery_uploads = run_package_frame(&mut schedule, &mut world, cloud);
    assert_eq!(recovery_uploads, visible_slots.len());
    let queued = world
        .resource::<LodAtlasUploadQueue>()
        .queued_slots()
        .collect::<Vec<_>>();
    assert_eq!(queued.len(), visible_slots.len());
    assert!(queued.iter().all(|upload| {
        upload.atlas == atlas
            && upload.gaussians_per_slot == plan.gaussians_per_slot
            && visible_slots.get(&upload.slot.index) == Some(&upload.slot)
    }));
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.current_recovery_queued);
        assert_eq!(state.current_page_leases, current_pages);
        assert_eq!(state.visible_slots, visible_slots);
    }
    let recovering_status = world.get::<GaussianLodPackageStatus>(cloud).unwrap();
    assert_eq!(recovering_status.phase, GaussianLodPackagePhase::Degraded);
    assert_eq!(
        recovering_status
            .failure
            .as_ref()
            .map(LodOrchestrationFailure::code),
        Some(LodOrchestrationFailureCode::AtlasCommitFailed)
    );
    assert!(
        recovering_status
            .error_detail()
            .is_some_and(|detail| detail.contains("recovery"))
    );

    // Extraction owns and drains the queued slots under its per-frame limits.
    // Re-enqueueing every main frame would invalidate that backlog and starve
    // slots above the first admitted batch, so the package must remain quiet.
    assert_eq!(run_package_frame(&mut schedule, &mut world, cloud), 0);
    assert!(world.resource::<GaussianLodPackageManager>().clouds[&cloud].current_recovery_queued);

    phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    assert_eq!(run_package_frame(&mut schedule, &mut world, cloud), 0);
    assert!(!world.resource::<GaussianLodPackageManager>().clouds[&cloud].current_recovery_queued);
    let recovered_status = world.get::<GaussianLodPackageStatus>(cloud).unwrap();
    assert_eq!(recovered_status.phase, GaussianLodPackagePhase::Active);
    assert!(recovered_status.failure.is_none());

    // A later independent generation loss starts one fresh bounded replay.
    let recovered_generation = {
        let manager = world.resource::<GaussianLodPackageManager>();
        manager.clouds[&cloud]
            .transient_atlas
            .ticket()
            .request_reupload_for_test()
    };
    assert_eq!(
        run_package_frame(&mut schedule, &mut world, cloud),
        visible_slots.len()
    );
    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert_eq!(state.transient_atlas_generation, recovered_generation);
    assert!(state.current_recovery_queued);
    assert_eq!(phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn globally_covering_offscreen_camera_does_not_stall_visible_camera_transaction() {
    let package = write_native_test_package(false);
    let mut settings = package_test_settings(0.0);
    settings.frustum_culling = true;
    let (mut world, cloud, visible_camera, _) = package_test_world(&package, settings, false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        visible_camera,
        package.manifest.quality.coarsest_gaussian_count as u32,
    );

    let offscreen_camera = world
        .spawn((
            Camera {
                viewport: Some(bevy::camera::Viewport {
                    physical_size: UVec2::new(1280, 720),
                    ..default()
                }),
                ..default()
            },
            Projection::Perspective(default()),
            GlobalTransform::from(
                Transform::from_xyz(0.0, 0.0, 100.0)
                    .looking_at(Vec3::new(0.0, 0.0, 110.0), Vec3::Y),
            ),
            crate::GaussianCamera::default(),
        ))
        .id();

    run_package_frame(&mut schedule, &mut world, cloud);
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert!(
            state.pending.is_none(),
            "an offscreen camera must not invalidate the active visible-camera cut"
        );
        let current = state.current.as_ref().expect("visible cut remains active");
        assert_eq!(current.len(), 1);
        assert!(current.get(visible_camera).is_some());
        assert!(current.get(offscreen_camera).is_none());
        assert!(package_candidate_set_is_active(current));
    }

    let published = world
        .get::<LodRenderCandidates>(cloud)
        .expect("active visible-camera cut remains published");
    assert_eq!(published.len(), 1);
    assert!(published.get(visible_camera).is_some());
    assert!(published.get(offscreen_camera).is_none());
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn cold_complete_empty_package_cut_activates_atomically() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings.clone(), false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    run_package_frame(&mut schedule, &mut world, cloud);
    let camera_view = package_test_camera_view(camera, Vec3::new(0.0, 0.0, 5.0));
    let transform = GlobalTransform::IDENTITY;

    world.resource_scope(|world, mut atlas_uploads: Mut<LodAtlasUploadQueue>| {
        let mut manager = world.resource_mut::<GaussianLodPackageManager>();
        let state = manager.clouds.get_mut(&cloud).unwrap();
        assert!(state.current.is_none());
        clear_package_pending_transaction(state).unwrap();
        install_complete_empty_test_pending(state, &settings, &transform, camera_view);
        drive_package_state_for_test(
            state,
            &settings,
            &transform,
            std::slice::from_ref(&camera_view),
            &mut atlas_uploads,
        )
        .unwrap();
    });

    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert!(state.pending.is_none());
    assert!(state.staged.is_none());
    assert!(state.current_page_leases.is_empty());
    assert!(state.visible_ranges.is_empty());
    assert!(state.visible_slots.is_empty());
    assert_eq!(state.transient_atlas.materialized_slot_count().unwrap(), 0);
    assert!(state.last_failure.is_none());
    assert_eq!(state.active_gaussians, 0);
    assert_eq!(package_status_phase(state), GaussianLodPackagePhase::Active);

    let recovered_generation = state.transient_atlas.ticket().request_reupload_for_test();
    world.resource_scope(|world, mut atlas_uploads: Mut<LodAtlasUploadQueue>| {
        let mut manager = world.resource_mut::<GaussianLodPackageManager>();
        drive_package_state_for_test(
            manager.clouds.get_mut(&cloud).unwrap(),
            &settings,
            &transform,
            std::slice::from_ref(&camera_view),
            &mut atlas_uploads,
        )
        .unwrap();
    });
    assert_eq!(
        world.resource::<LodAtlasUploadQueue>().queued_slot_count(),
        0
    );
    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert_eq!(state.transient_atlas_generation, recovered_generation);
    assert!(!state.current_recovery_queued);
    assert!(package_candidate_set_is_active(
        state.current.as_ref().unwrap()
    ));
    assert_eq!(package_status_phase(state), GaussianLodPackagePhase::Active);
    assert!(state.last_failure.is_none());
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn nonempty_to_complete_empty_package_cut_retains_runtime_cache() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings.clone(), false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.manifest.quality.coarsest_gaussian_count as u32,
    );
    let (retained_ranges, retained_slots, retained_leases) = {
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        (
            state.visible_ranges.clone(),
            state.visible_slots.clone(),
            state.current_page_leases.clone(),
        )
    };
    assert!(!retained_slots.is_empty());
    let camera_view = package_test_camera_view(camera, Vec3::new(0.0, 0.0, 5.0));
    let transform = GlobalTransform::IDENTITY;

    world.resource_scope(|world, mut atlas_uploads: Mut<LodAtlasUploadQueue>| {
        let mut manager = world.resource_mut::<GaussianLodPackageManager>();
        let state = manager.clouds.get_mut(&cloud).unwrap();
        install_complete_empty_test_pending(state, &settings, &transform, camera_view);
        let staged = state.staged.as_ref().unwrap();
        assert!(staged.complete);
        assert!(staged.ranges.is_empty());
        assert!(staged.slots.is_empty());
        assert!(package_candidate_set_is_complete_empty(
            state.pending.as_ref().unwrap()
        ));
        assert_eq!(state.visible_ranges, retained_ranges);
        assert_eq!(state.visible_slots, retained_slots);
        assert_eq!(state.current_page_leases, retained_leases);
        drive_package_state_for_test(
            state,
            &settings,
            &transform,
            std::slice::from_ref(&camera_view),
            &mut atlas_uploads,
        )
        .unwrap();
    });

    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert!(state.pending.is_none());
    assert!(state.staged.is_none());
    assert!(state.current_page_leases.is_empty());
    assert!(state.visible_ranges.is_empty());
    assert!(state.visible_slots.is_empty());
    assert!(
        state.transient_atlas.materialized_slot_count().unwrap() >= retained_slots.len(),
        "an empty visible cut must not discard bounded runtime-cache payloads"
    );
    for range in &retained_ranges {
        assert!(state.mirror.is_range_current(*range));
    }
    let current = state.current.as_ref().unwrap().get(camera).unwrap();
    assert!(current.render_is_active());
    assert!(!package_candidate_requires_atlas(current));
    assert_eq!(state.active_gaussians, 0);
    assert_eq!(package_status_phase(state), GaussianLodPackagePhase::Active);
    assert!(state.last_failure.is_none());
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn stale_empty_motion_candidate_cannot_blank_a_visible_current_cut() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings.clone(), false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.manifest.quality.coarsest_gaussian_count as u32,
    );
    let (retained_ranges, retained_slots) = {
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        (state.visible_ranges.clone(), state.visible_slots.clone())
    };
    let staged_view = package_test_camera_view(camera, Vec3::new(0.0, 0.0, 100.0));
    let live_view = package_test_camera_view(camera, Vec3::new(0.0, 0.0, 5.0));
    let transform = GlobalTransform::IDENTITY;

    let stale_empty_phase =
        world.resource_scope(|world, mut atlas_uploads: Mut<LodAtlasUploadQueue>| {
            let mut manager = world.resource_mut::<GaussianLodPackageManager>();
            let state = manager.clouds.get_mut(&cloud).unwrap();
            let phase =
                install_complete_empty_test_pending(state, &settings, &transform, staged_view);
            assert_eq!(state.visible_ranges, retained_ranges);
            assert_eq!(state.visible_slots, retained_slots);
            drive_package_state_for_test(
                state,
                &settings,
                &transform,
                std::slice::from_ref(&live_view),
                &mut atlas_uploads,
            )
            .unwrap();
            phase
        });

    assert_eq!(
        stale_empty_phase.load(Ordering::Acquire),
        LOD_RENDER_WAITING
    );
    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert_eq!(state.visible_ranges, retained_ranges);
    assert_eq!(state.visible_slots, retained_slots);
    let current = state.current.as_ref().unwrap().get(camera).unwrap();
    assert!(current.render_is_active());
    assert!(package_candidate_requires_atlas(current));
    assert_eq!(package_status_phase(state), GaussianLodPackagePhase::Active);
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn per_cloud_camera_limit_isolated_for_disjoint_visibility_sets() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, first_cloud, first_camera, manifest) =
        package_test_world(&package, settings.clone(), false, 0);
    world
        .resource_mut::<GaussianLodPackageConfig>()
        .max_views_per_cloud = 1;
    let second_cloud = spawn_package_test_cloud(&mut world, &package, manifest, settings, false);
    let second_camera = world
        .spawn((
            Camera {
                viewport: Some(bevy::camera::Viewport {
                    physical_size: UVec2::new(1280, 720),
                    ..default()
                }),
                ..default()
            },
            Projection::Perspective(default()),
            GlobalTransform::from(Transform::from_xyz(2.0, 0.0, 5.0)),
            crate::GaussianCamera::default(),
        ))
        .id();
    mark_package_cloud_visible(&mut world, second_camera, second_cloud);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let active = (0..2048).any(|_| {
        run_package_frame_for_clouds(&mut schedule, &mut world, &[first_cloud, second_cloud]);
        [first_cloud, second_cloud].iter().all(|cloud| {
            world
                .get::<GaussianLodPackageStatus>(*cloud)
                .is_some_and(|status| status.phase == GaussianLodPackagePhase::Active)
        })
    });
    assert!(
        active,
        "disjoint one-camera clouds must stream independently"
    );

    let manager = world.resource::<GaussianLodPackageManager>();
    let first = manager.clouds[&first_cloud].current.as_ref().unwrap();
    let second = manager.clouds[&second_cloud].current.as_ref().unwrap();
    assert_eq!(first.len(), 1);
    assert!(first.get(first_camera).is_some());
    assert!(first.get(second_camera).is_none());
    assert_eq!(second.len(), 1);
    assert!(second.get(second_camera).is_some());
    assert!(second.get(first_camera).is_none());
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn unsupported_camera_only_fails_its_visible_package() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, supported_cloud, supported_camera, manifest) =
        package_test_world(&package, settings.clone(), false, 0);
    let unsupported_cloud =
        spawn_package_test_cloud(&mut world, &package, manifest, settings, false);
    let unsupported_camera = world
        .spawn((
            Camera {
                viewport: Some(bevy::camera::Viewport {
                    physical_size: UVec2::new(1280, 720),
                    ..default()
                }),
                ..default()
            },
            Projection::custom(bevy::camera::PerspectiveProjection::default()),
            GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 5.0)),
            crate::GaussianCamera::default(),
        ))
        .id();
    mark_package_cloud_visible(&mut world, unsupported_camera, unsupported_cloud);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let supported_active = (0..2048).any(|_| {
        run_package_frame_for_clouds(
            &mut schedule,
            &mut world,
            &[supported_cloud, unsupported_cloud],
        );
        world
            .get::<GaussianLodPackageStatus>(supported_cloud)
            .is_some_and(|status| status.phase == GaussianLodPackagePhase::Active)
    });
    assert!(supported_active);
    let supported = world
        .get::<LodRenderCandidates>(supported_cloud)
        .and_then(|candidates| candidates.get(supported_camera))
        .expect("the unrelated supported camera must remain published");
    assert!(supported.render_is_active());
    let unsupported = world
        .get::<GaussianLodPackageStatus>(unsupported_cloud)
        .unwrap();
    assert_eq!(unsupported.phase, GaussianLodPackagePhase::Failed);
    assert_eq!(
        unsupported
            .failure
            .as_ref()
            .map(LodOrchestrationFailure::code),
        Some(LodOrchestrationFailureCode::UnsupportedConfiguration)
    );
    assert!(
        unsupported
            .error_detail()
            .is_some_and(|detail| detail.contains("UnsupportedCamera"))
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn failed_stale_and_multiview_pending_cuts_retain_the_current_transaction() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let coarse_count = package.manifest.quality.coarsest_gaussian_count as u32;
    drive_package_to_active_count(&mut schedule, &mut world, cloud, camera, coarse_count);
    let (coarse_ranges, coarse_slots, coarse_atlas) = {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        (
            state.visible_ranges.clone(),
            state.visible_slots.clone(),
            sparse_package_atlas_snapshot(state),
        )
    };

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    let failed_phase = (0..2048)
        .find_map(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            let manager = world.resource::<GaussianLodPackageManager>();
            manager.clouds[&cloud]
                .pending
                .as_ref()?
                .get(camera)
                .map(|candidate| Arc::clone(&candidate.phase))
        })
        .expect("quality churn must stage a replacement cut");
    {
        let published = world
            .get::<LodRenderCandidates>(cloud)
            .expect("pending replacement is extracted for rendering");
        assert!(
            published.retained_current,
            "a pending package candidate must preserve the current GPU output"
        );
        assert!(
            !published.candidates_are_current,
            "pending descriptors must not be replayed as the retained current cut"
        );
    }
    failed_phase.store(
        crate::stream::render_commit::LOD_RENDER_FAILED,
        Ordering::Release,
    );
    assert_eq!(run_package_frame(&mut schedule, &mut world, cloud), 0);
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.current.is_some());
        assert!(state.pending.is_none());
        assert!(!state.root_fallback);
        assert_eq!(state.visible_ranges, coarse_ranges);
        assert_eq!(state.visible_slots, coarse_slots);
        assert!(package_candidate_set_is_active(
            state.current.as_ref().unwrap()
        ));
        assert_eq!(
            state
                .last_failure
                .as_ref()
                .map(LodOrchestrationFailure::code),
            Some(LodOrchestrationFailureCode::RenderCommitFailed)
        );
        assert_eq!(sparse_package_atlas_snapshot(state), coarse_atlas);
    }
    {
        let published = world
            .get::<LodRenderCandidates>(cloud)
            .expect("retained current candidate remains published");
        assert!(published.retained_current);
        assert!(published.candidates_are_current);
    }

    // The pending descriptor never replaced the current GPU payload, so
    // recovery requires no atlas rewrite or candidate re-stage.
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 0.0;
    let recovery_uploads = run_package_frame(&mut schedule, &mut world, cloud);
    assert_eq!(recovery_uploads, 0);
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.pending.is_none());
        assert!(package_candidate_set_is_active(
            state.current.as_ref().unwrap()
        ));
        assert!(state.last_failure.is_none());
        assert_eq!(state.visible_ranges, coarse_ranges);
        assert_eq!(state.visible_slots, coarse_slots);
    }

    // A settings change after extraction must revoke the stale pending token
    // and keep the current cut, rather than activating an old quality request.
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    let stale_phase = (0..2048)
        .find_map(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            let manager = world.resource::<GaussianLodPackageManager>();
            manager.clouds[&cloud]
                .pending
                .as_ref()?
                .get(camera)
                .map(|candidate| Arc::clone(&candidate.phase))
        })
        .expect("second quality churn must stage a replacement cut");
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 0.0;
    let churn_uploads = run_package_frame(&mut schedule, &mut world, cloud);
    assert!(churn_uploads <= coarse_slots.len());
    assert_eq!(stale_phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.pending.is_none());
        assert!(package_candidate_set_is_active(
            state.current.as_ref().unwrap()
        ));
        assert!(!state.root_fallback);
        assert_eq!(state.visible_ranges, coarse_ranges);
        assert_eq!(state.visible_slots, coarse_slots);
    }

    // Critical policy remains exact, but a view-only mismatch must not starve
    // the two-phase handshake under continuous camera motion. Render extraction
    // claims this exact identity before private GPU preparation; successive
    // camera updates must preserve it even while phase remains WAITING.
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    let camera_motion_phase = (0..2048)
        .find_map(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            let manager = world.resource::<GaussianLodPackageManager>();
            manager.clouds[&cloud]
                .pending
                .as_ref()?
                .get(camera)
                .map(|candidate| Arc::clone(&candidate.phase))
        })
        .expect("camera-motion fixture must stage a replacement cut");
    world
        .get::<LodRenderCandidates>(cloud)
        .and_then(|candidates| candidates.get(camera))
        .expect("visible pending candidate was published for extraction")
        .publish_render_claimed();
    assert_eq!(
        camera_motion_phase.load(Ordering::Acquire),
        LOD_RENDER_WAITING,
    );
    for step in 0..4 {
        *world.get_mut::<GlobalTransform>(camera).unwrap() =
            GlobalTransform::from(Transform::from_xyz(1.0 + step as f32 * 0.125, 0.0, 6.0));
        // Hold the render-side state before PREPARED while the main package
        // advances. This is the N/N+1 extraction race which a helper that
        // advances phase before every package frame would otherwise mask.
        world.insert_resource(LodAtlasUploadQueue::default());
        schedule.run(&mut world);
        std::thread::yield_now();
        let manager = world.resource::<GaussianLodPackageManager>();
        let retained = manager.clouds[&cloud]
            .pending
            .as_ref()
            .and_then(|pending| pending.get(camera))
            .expect("render-claimed WAITING candidate must survive camera-only churn");
        assert!(Arc::ptr_eq(&retained.phase, &camera_motion_phase));
        assert_eq!(retained.phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    }

    run_package_frame(&mut schedule, &mut world, cloud);
    assert_eq!(
        camera_motion_phase.load(Ordering::Acquire),
        LOD_RENDER_PREPARED
    );
    run_package_frame(&mut schedule, &mut world, cloud);
    assert_eq!(
        camera_motion_phase.load(Ordering::Acquire),
        LOD_RENDER_ACTIVE
    );
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.pending.is_none());
        assert!(state.pending_request.is_none());
        assert!(
            !state.current_request_matches_live,
            "a prepared old-camera cut may activate, but must remain diagnostically stale"
        );
        let candidate = state.current.as_ref().unwrap().get(camera).unwrap();
        assert!(Arc::ptr_eq(&candidate.phase, &camera_motion_phase));
        assert!(candidate.frontier().candidate_count() > coarse_count);
        assert!(!state.root_fallback);
    }
    let moving_camera_reached_exact = (0..2048).any(|step| {
        *world.get_mut::<GlobalTransform>(camera).unwrap() =
            GlobalTransform::from(Transform::from_xyz(1.0 + step as f32 * 0.001, 0.0, 6.0));
        run_package_frame(&mut schedule, &mut world, cloud);
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        state.pending.is_none()
            && state.current.as_ref().is_some_and(|current| {
                current.get(camera).is_some_and(|candidate| {
                    candidate.phase.load(Ordering::Acquire) == LOD_RENDER_ACTIVE
                        && candidate.frontier().candidate_count() == package.source_count as u32
                })
            })
    });
    assert!(
        moving_camera_reached_exact,
        "continuous camera motion must not starve prepared LoD activation"
    );

    // Active-list capacity is part of pending request identity. Lowering it
    // must reject an already prepared exact cut whose count exceeds the new
    // cap, while the retained coarse cut stays valid.
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 0.0;
    drive_package_to_active_count(&mut schedule, &mut world, cloud, camera, coarse_count);
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    let over_budget_phase = (0..2048)
        .find_map(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            let manager = world.resource::<GaussianLodPackageManager>();
            let candidate = manager.clouds[&cloud].pending.as_ref()?.get(camera)?;
            (candidate.frontier().candidate_count() > coarse_count)
                .then(|| Arc::clone(&candidate.phase))
        })
        .expect("active-budget fixture must stage an over-budget exact cut");
    world
        .get_mut::<GaussianLodSettings>(cloud)
        .unwrap()
        .budgets
        .max_active_gaussians = u64::from(coarse_count);
    assert_eq!(run_package_frame(&mut schedule, &mut world, cloud), 0);
    assert_eq!(
        over_budget_phase.load(Ordering::Acquire),
        LOD_RENDER_WAITING
    );
    {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.pending.is_none());
        assert!(
            state
                .current
                .as_ref()
                .unwrap()
                .by_camera
                .values()
                .all(
                    |candidate| u64::from(candidate.frontier().candidate_count())
                        <= u64::from(coarse_count)
                )
        );
        assert!(state.pending_request.is_none());
        assert!(!state.root_fallback);
    }
    world
        .get_mut::<GaussianLodSettings>(cloud)
        .unwrap()
        .budgets
        .max_active_gaussians = 4096;

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    let camera_churn_phase = (0..2048)
        .find_map(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            let manager = world.resource::<GaussianLodPackageManager>();
            manager.clouds[&cloud]
                .pending
                .as_ref()?
                .get(camera)
                .map(|candidate| Arc::clone(&candidate.phase))
        })
        .expect("camera churn fixture must stage a replacement cut");
    let second_camera = world
        .spawn((
            Camera {
                viewport: Some(bevy::camera::Viewport {
                    physical_size: UVec2::new(1280, 720),
                    ..default()
                }),
                ..default()
            },
            Projection::Perspective(default()),
            GlobalTransform::from(Transform::from_xyz(4.0, 0.0, 5.0)),
            crate::GaussianCamera::default(),
        ))
        .id();
    mark_package_cloud_visible(&mut world, second_camera, cloud);
    let two_camera_pending = (0..2048).any(|_| {
        run_package_frame(&mut schedule, &mut world, cloud);
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.current.is_some());
        assert_eq!(state.visible_ranges, coarse_ranges);
        assert_eq!(state.visible_slots, coarse_slots);
        state
            .pending
            .as_ref()
            .is_some_and(|pending| pending.len() == 2)
    });
    assert!(
        two_camera_pending,
        "camera-set replacement must eventually stage one complete two-view cut"
    );
    assert_eq!(
        camera_churn_phase.load(Ordering::Acquire),
        LOD_RENDER_WAITING
    );
    let manager = world.resource::<GaussianLodPackageManager>();
    let state = &manager.clouds[&cloud];
    assert!(state.current.is_some());
    assert!(!state.root_fallback);
    assert_eq!(
        state.pending.as_ref().map(LodRenderCandidates::len),
        Some(2)
    );
    assert!(state.pending_request.is_some());
    assert_eq!(
        state
            .visible_ranges
            .iter()
            .map(|range| u64::from(range.count))
            .sum::<u64>(),
        u64::from(coarse_count)
    );
    assert_eq!(state.visible_ranges, coarse_ranges);
    assert_eq!(state.visible_slots, coarse_slots);
    // Once a two-camera cut is active, each view retains its filtered output.
    // The next replacement publishes WAITING candidates without rewriting or
    // exposing the atlas union through an unfiltered draw.
    let active_two_camera =
        (0..2048).any(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            let manager = world.resource::<GaussianLodPackageManager>();
            let state = &manager.clouds[&cloud];
            state.current.as_ref().is_some_and(|current| {
                current.len() == 2 && package_candidate_set_is_active(current)
            }) && state.pending.is_none()
                && !state.root_fallback
        });
    assert!(active_two_camera, "two-camera cut must activate");
    let (two_camera_ranges, two_camera_slots) = {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        (state.visible_ranges.clone(), state.visible_slots.clone())
    };
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 0.0;
    let coarse_two_camera_pending = (0..2048).any(|_| {
        run_package_frame(&mut schedule, &mut world, cloud);
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        assert_eq!(state.visible_ranges, two_camera_ranges);
        assert_eq!(state.visible_slots, two_camera_slots);
        state
            .pending
            .as_ref()
            .is_some_and(|pending| pending.len() == 2)
    });
    assert!(
        coarse_two_camera_pending,
        "quality replacement must eventually stage one complete two-view cut"
    );
    let manager = world.resource::<GaussianLodPackageManager>();
    let state = &manager.clouds[&cloud];
    assert!(state.current.is_some());
    assert!(!state.root_fallback);
    assert_eq!(
        state.pending.as_ref().map(LodRenderCandidates::len),
        Some(2)
    );
    assert_eq!(state.visible_ranges, two_camera_ranges);
    assert_eq!(state.visible_slots, two_camera_slots);
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn compressed_representative_pages_stream_through_the_canonical_atlas() {
    let representative_degree =
        crate::material::spherical_harmonics::SH_DEGREE.saturating_sub(1) as u8;
    let package = write_native_test_package_with_degree(false, Some(representative_degree));
    assert!(package.manifest.pages.iter().any(|descriptor| {
        descriptor.kind == LodPageKind::Representatives
            && descriptor.encoding
                == LodPageEncoding::F16Sh {
                    degree: representative_degree,
                }
    }));
    assert!(package.manifest.pages.iter().all(|descriptor| {
        descriptor.kind != LodPageKind::SourceLeaves
            || descriptor.encoding == LodPageEncoding::F32Planar
    }));

    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.manifest.quality.coarsest_gaussian_count as u32,
    );
    let candidates = world.get::<LodRenderCandidates>(cloud).unwrap();
    assert!(
        candidates
            .by_camera
            .values()
            .flat_map(LodRenderCandidate::render_ranges)
            .any(|range| {
                package.manifest.pages.iter().any(|descriptor| {
                    descriptor.id == range.page && descriptor.kind == LodPageKind::Representatives
                })
            })
    );

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.source_count as u32,
    );
    let candidates = world.get::<LodRenderCandidates>(cloud).unwrap();
    assert!(
        candidates
            .by_camera
            .values()
            .flat_map(LodRenderCandidate::render_ranges)
            .all(|range| {
                package.manifest.pages.iter().any(|descriptor| {
                    descriptor.id == range.page && descriptor.kind == LodPageKind::SourceLeaves
                })
            })
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn native_package_missing_leaf_marks_ancestor_fallback_and_despawn_cleans_atlas() {
    let package = write_native_test_package(true);
    assert!(package.omitted_page.is_some());
    let settings = package_test_settings(1.0);
    let (mut world, cloud, _camera, _) = package_test_world(&package, settings, true, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let atlas = (0..4096)
        .find_map(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            let status = world.get::<GaussianLodPackageStatus>(cloud)?;
            let metadata = world.get::<LodDebugMetadata>(cloud)?;
            (status.phase == GaussianLodPackagePhase::Degraded
                && status.terminal_failures > 0
                && status.active_gaussians > 0
                && metadata.any_record(|record| {
                    record.residency_code() == LodDebugResidency::AncestorFallback as u32
                }))
            .then(|| {
                world
                    .get::<PlanarGaussian3dHandle>(cloud)
                    .unwrap()
                    .handle()
                    .clone()
            })
        })
        .unwrap_or_else(|| {
            panic!(
                "missing native leaf did not publish fallback provenance; status={:?}",
                world.get::<GaussianLodPackageStatus>(cloud)
            )
        });
    let status = world.get::<GaussianLodPackageStatus>(cloud).unwrap();
    assert_eq!(status.phase, GaussianLodPackagePhase::Degraded);
    assert!(status.active_gaussians > 0);
    assert!((status.active_gaussians as usize) < package.source_count);

    let (deferred_slot, gaussians_per_slot, ticket, ticket_generation) = {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        let slot = *state
            .visible_slots
            .values()
            .next()
            .expect("the fallback cut materialized at least one slot");
        (
            slot,
            state.plan.gaussians_per_slot,
            state.transient_atlas.ticket().clone(),
            state.transient_atlas.ticket().generation(),
        )
    };
    world
        .resource_mut::<LodAtlasUploadQueue>()
        .enqueue_slot(atlas.id(), deferred_slot, gaussians_per_slot)
        .unwrap();
    assert!(
        world
            .resource::<LodTransientAtlasRegistry>()
            .contains(atlas.id())
    );
    assert!(
        world
            .resource::<LodAtlasUploadQueue>()
            .queued_slots()
            .any(|upload| upload.atlas == atlas.id())
    );

    world.despawn(cloud);
    schedule.run(&mut world);
    assert!(
        world
            .resource::<GaussianLodPackageManager>()
            .clouds
            .is_empty()
    );
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&atlas)
            .is_none()
    );
    assert_eq!(
        world.resource::<LodAtlasUploadQueue>().queued_slot_count(),
        0
    );
    assert!(
        !world
            .resource::<LodTransientAtlasRegistry>()
            .contains(atlas.id())
    );
    assert!(!ticket.acknowledge(ticket_generation));
    assert!(!ticket.is_ready());
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn preprocessed_package_open_allocates_no_capacity_sized_cpu_payload() {
    let package = write_native_test_package(false);
    let mut settings = package_test_settings(0.0);
    settings.budgets.max_resident_gaussians = 1_000_000;
    settings.budgets.max_resident_bytes = 256 * 1024 * 1024;
    settings.budgets.max_resident_pages = 100_000;
    let mut config = GaussianLodPackageConfig {
        max_atlas_gaussians: 1_000_000,
        max_atlas_bytes: 256 * 1024 * 1024,
        ..default()
    };
    config.streaming.retry_limit = 0;
    let mut manager = GaussianLodPackageManager::default();
    let mut clouds = Assets::<PlanarGaussian3d>::default();
    let mut transient_atlases = LodTransientAtlasRegistry::default();
    let mut atlas_uploads = LodAtlasUploadQueue::default();
    let asset = GaussianLodAsset::new(package.manifest.clone()).unwrap();
    let state = instantiate_package(
        &asset,
        &GaussianLodPackageSource::native_directory(package.root.to_string_lossy().into_owned()),
        &settings,
        &config,
        &config.streaming,
        false,
        &mut manager,
        &mut clouds,
        &mut transient_atlases,
        &mut atlas_uploads,
    )
    .unwrap();

    assert!(state.plan.physical_gaussians > package.source_count as u32);
    assert!(clouds.get(&state.atlas).is_none());
    assert_eq!(state.transient_atlas.materialized_slot_count().unwrap(), 0);
    assert_eq!(
        state.transient_atlas.materialized_gaussian_count().unwrap(),
        0
    );
    assert_eq!(atlas_uploads.queued_slot_count(), 0);
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn stationary_cold_package_publishes_one_bootstrap_then_one_fixed_point_cut() {
    let package = write_native_test_package_with_levels(3);
    let mut settings = package_test_settings(1.0);
    settings.budgets.max_resident_pages = 1024;
    settings.budgets.max_requests_per_frame = 1;
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    {
        let mut config = world.resource_mut::<GaussianLodPackageConfig>();
        config.max_atlas_gaussians = 8192;
        config.streaming.max_concurrent_requests = 1;
    }
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let exact_count = package.source_count as u32;
    let mut saw_resident_loading = false;
    let mut saw_bootstrap_pending = false;
    let mut saw_exact_pending = false;
    let mut activated_counts = Vec::new();
    let mut previous_current_phase = None;
    for _ in 0..4096 {
        run_package_frame(&mut schedule, &mut world, cloud);
        std::thread::sleep(Duration::from_millis(1));
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        if state.current.is_none() {
            saw_resident_loading |= state.resident_pages > 0;
            assert!(
                state.visible_ranges.is_empty() && state.visible_slots.is_empty(),
                "cold unresolved page waves must not become visible atlas ownership"
            );
        }
        if let Some(pending) = &state.pending {
            assert_eq!(
                state.pending_fallback_nodes,
                package_candidate_fallback_nodes(pending),
                "pending CPU Residency metadata must match per-view candidate provenance"
            );
            let candidate = pending
                .get(camera)
                .expect("camera pending candidate exists");
            if candidate.frontier().is_coverage_guard() {
                assert!(candidate.rendered_candidate_count() < exact_count);
                assert!(
                    candidate
                        .render_ranges()
                        .iter()
                        .map(|range| range.page)
                        .collect::<BTreeSet<_>>()
                        .len()
                        <= PACKAGE_BOOTSTRAP_MAX_PAGES as usize
                );
                saw_bootstrap_pending = true;
            } else {
                assert_eq!(candidate.rendered_candidate_count(), exact_count);
                saw_exact_pending = true;
            }
        }
        if let Some(candidate) = state
            .current
            .as_ref()
            .and_then(|current| current.get(camera))
        {
            let current = state.current.as_ref().unwrap();
            let expected_fallback = package_candidate_fallback_nodes(current);
            assert_eq!(state.current_fallback_nodes, expected_fallback);
            assert_eq!(state.visible_fallback_nodes, expected_fallback);
            if previous_current_phase
                .as_ref()
                .is_none_or(|phase| !Arc::ptr_eq(phase, &candidate.phase))
            {
                activated_counts.push(candidate.rendered_candidate_count());
                previous_current_phase = Some(Arc::clone(&candidate.phase));
            }
            if candidate.rendered_candidate_count() == exact_count && candidate.render_is_active() {
                break;
            }
        }
    }

    assert!(
        saw_resident_loading,
        "the regression must observe resident pages arriving before publication"
    );
    assert!(
        saw_bootstrap_pending,
        "the progressive package must stage one complete bounded bootstrap"
    );
    assert!(saw_exact_pending, "the fixed-point target must be staged");
    assert_eq!(activated_counts.len(), 2);
    assert!(activated_counts[0] < exact_count);
    assert_eq!(activated_counts[1], exact_count);
    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert!(state.pending.is_none());
    assert_eq!(state.active_gaussians, u64::from(exact_count));
    assert!(package_candidate_set_is_active(
        state.current.as_ref().unwrap()
    ));
    let (final_phase, final_ranges, final_slots, final_resident, request_starts) = {
        let candidate = state.current.as_ref().unwrap().get(camera).unwrap();
        let runtime = state.runtime.lock().unwrap();
        (
            Arc::clone(&candidate.phase),
            state.visible_ranges.clone(),
            state.visible_slots.clone(),
            state.resident_pages,
            runtime.transport_request_starts_for_test(),
        )
    };
    for _ in 0..256 {
        run_package_frame(&mut schedule, &mut world, cloud);
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        let candidate = state.current.as_ref().unwrap().get(camera).unwrap();
        assert!(Arc::ptr_eq(&final_phase, &candidate.phase));
        assert_eq!(state.visible_ranges, final_ranges);
        assert_eq!(state.visible_slots, final_slots);
        assert_eq!(state.resident_pages, final_resident);
        assert!(state.pending.is_none());
        let runtime = state.runtime.lock().unwrap();
        assert_eq!(runtime.transport_request_starts_for_test(), request_starts);
    }
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn coarsest_package_bootstrap_cpu_metadata_and_cached_toggle_stay_resident() {
    let package = write_native_test_package_with_levels(3);
    let mut settings = package_test_settings(0.0);
    settings.budgets.max_requests_per_frame = 1;
    let (mut world, cloud, _, _) = package_test_world(&package, settings, true, 0);
    world
        .resource_mut::<GaussianLodPackageConfig>()
        .streaming
        .max_concurrent_requests = 1;
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let mut saw_pending_bootstrap = false;
    let mut saw_current_bootstrap = false;
    for _ in 0..512 {
        run_package_frame(&mut schedule, &mut world, cloud);
        std::thread::sleep(Duration::from_millis(1));
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        if let Some(pending) = state.pending.as_ref().filter(|pending| {
            pending
                .by_camera
                .values()
                .all(|candidate| candidate.frontier().is_coverage_guard())
        }) {
            let expected = package_candidate_fallback_nodes(pending);
            assert!(
                expected.is_empty(),
                "coarsest all-resident roots are not ancestor fallbacks"
            );
            assert_eq!(state.pending_fallback_nodes, expected);
            saw_pending_bootstrap = true;
        }
        if let Some(current) = state.current.as_ref().filter(|current| {
            current
                .by_camera
                .values()
                .all(|candidate| candidate.frontier().is_coverage_guard())
        }) {
            let expected = package_candidate_fallback_nodes(current);
            assert!(expected.is_empty());
            assert_eq!(state.current_fallback_nodes, expected);
            assert_eq!(state.visible_fallback_nodes, expected);
            saw_current_bootstrap = true;
            break;
        }
    }
    assert!(
        saw_pending_bootstrap && saw_current_bootstrap,
        "the progressive package must stage and retain its bounded coarsest bootstrap"
    );

    let mut manager = world.resource_mut::<GaussianLodPackageManager>();
    let state = manager.clouds.get_mut(&cloud).unwrap();
    let cached_slots = state
        .visible_ranges
        .iter()
        .map(|range| (range.page, range.slot))
        .collect::<BTreeSet<_>>()
        .len();
    assert!(cached_slots > 0);
    sync_package_debug_annotations(state, false).unwrap();
    assert!(state.debug.is_none());
    sync_package_debug_annotations(state, true).unwrap();

    let mut frames = 0_usize;
    let mut records = 0_usize;
    let mut regenerated_records = 0_usize;
    while !state.debug.as_ref().unwrap().initialization.is_empty() {
        let mut work = PackageDebugPreparationWork::default();
        advance_package_debug_initialization(state, &mut work).unwrap();
        assert!(work.slots > 0);
        frames += 1;
        records = records.saturating_add(work.records);
        regenerated_records = regenerated_records.saturating_add(work.regenerated_records);
    }
    assert_eq!(
        records, 0,
        "a Resident bootstrap must reuse cached page bases without Residency patch copies"
    );
    assert_eq!(regenerated_records, 0);
    assert_eq!(
        frames,
        cached_slots.div_ceil(PACKAGE_DEBUG_MAX_SLOTS_PER_FRAME),
        "cached Resident bootstrap initialization must be slot-bound, not record-bound"
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn tiny_resident_capacity_retains_one_quiescent_bootstrap_with_typed_failure() {
    // Sixteen leaf pages are the exact target, while the package bootstrap cap
    // admits the preceding four-page antichain. The exact target fits by itself,
    // but target + bootstrap + independent root cannot coexist for one atomic
    // handoff. Keep the useful bootstrap stable instead of cycling pressure cuts.
    let package = write_native_test_package_with_levels_and_leaf_capacity(2, 4);
    let leaf_pages = package
        .manifest
        .pages
        .iter()
        .filter(|page| page.kind == LodPageKind::SourceLeaves)
        .count() as u32;
    let root_pages = package
        .manifest
        .roots
        .iter()
        .filter_map(|root| {
            package
                .manifest
                .nodes
                .iter()
                .find(|node| node.id == *root)
                .map(|node| node.representation.page)
        })
        .collect::<BTreeSet<_>>()
        .len() as u32;
    assert_eq!(leaf_pages, 16);
    assert_eq!(root_pages, 1);

    let mut settings = package_test_settings(1.0);
    let resident_page_limit = leaf_pages + root_pages + 1;
    settings.budgets.max_resident_pages = resident_page_limit;
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let exact_count = package.source_count as u32;
    for _ in 0..4096 {
        run_package_frame(&mut schedule, &mut world, cloud);
        std::thread::sleep(Duration::from_millis(1));
        if world
            .get::<GaussianLodPackageStatus>(cloud)
            .is_some_and(|status| {
                status.phase == GaussianLodPackagePhase::Degraded
                    && status.failure.as_ref().is_some_and(|failure| {
                        failure.code() == LodOrchestrationFailureCode::CapacityExceeded
                    })
            })
        {
            break;
        }
    }

    let status = world.get::<GaussianLodPackageStatus>(cloud).unwrap();
    assert_eq!(status.phase, GaussianLodPackagePhase::Degraded);
    assert_eq!(
        status.failure.as_ref().map(LodOrchestrationFailure::code),
        Some(LodOrchestrationFailureCode::CapacityExceeded)
    );
    assert!(
        status
            .error_detail()
            .is_some_and(|detail| detail.contains("AtomicHandoffCapacityExceeded"))
    );
    let (phase, candidate_ranges, visible_ranges, visible_slots, request_starts, stalled_request) = {
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        assert!(state.pending.is_none());
        let candidate = state.current.as_ref().unwrap().get(camera).unwrap();
        assert!(candidate.render_is_active());
        assert!(candidate.frontier().is_coverage_guard());
        assert!(candidate.rendered_candidate_count() < exact_count);
        let stalled_request = match state.bootstrap_handoff.as_ref().unwrap() {
            PackageBootstrapHandoff::CapacityExceeded { request, .. } => request.clone(),
            PackageBootstrapHandoff::Admitted(_) => panic!("tiny target must not be admitted"),
        };
        let runtime = state.runtime.lock().unwrap();
        assert_eq!(runtime.pending_request_count_for_test(), 0);
        assert!(runtime.package_bootstrap_pages_for_test().is_none());
        (
            Arc::clone(&candidate.phase),
            candidate.render_ranges().to_vec(),
            state.visible_ranges.clone(),
            state.visible_slots.clone(),
            runtime.transport_request_starts_for_test(),
            stalled_request,
        )
    };

    for _ in 0..256 {
        run_package_frame(&mut schedule, &mut world, cloud);
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        assert!(state.pending.is_none());
        let candidate = state.current.as_ref().unwrap().get(camera).unwrap();
        assert!(candidate.render_is_active());
        assert!(Arc::ptr_eq(&phase, &candidate.phase));
        assert_eq!(candidate.render_ranges(), candidate_ranges);
        assert_eq!(state.visible_ranges, visible_ranges);
        assert_eq!(state.visible_slots, visible_slots);
        let runtime = state.runtime.lock().unwrap();
        assert_eq!(runtime.pending_request_count_for_test(), 0);
        assert_eq!(runtime.transport_request_starts_for_test(), request_starts);
    }

    world
        .get_mut::<GaussianLodSettings>(cloud)
        .unwrap()
        .selection_mode = LodSelectionMode::Frozen;
    run_package_frame(&mut schedule, &mut world, cloud);
    let frozen_request = {
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        let request = state.bootstrap_handoff.as_ref().unwrap().request().clone();
        assert_ne!(request, stalled_request);
        let runtime = state.runtime.lock().unwrap();
        assert_eq!(runtime.pending_request_count_for_test(), 0);
        assert_eq!(runtime.transport_request_starts_for_test(), request_starts);
        request
    };

    // A camera-set change must first atomically rebind the already-resident
    // bootstrap. It must not return through the old capacity memoization with
    // the added camera missing, nor restart any page work.
    let added_camera = world
        .spawn((
            Camera {
                viewport: Some(bevy::camera::Viewport {
                    physical_size: UVec2::new(1280, 720),
                    ..default()
                }),
                ..default()
            },
            Projection::Perspective(default()),
            GlobalTransform::from(Transform::from_xyz(2.0, 0.0, 5.0)),
            crate::GaussianCamera::default(),
        ))
        .id();
    mark_package_cloud_visible(&mut world, added_camera, cloud);
    run_package_frame(&mut schedule, &mut world, cloud);
    let added_frozen_view = {
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        let pending = state
            .pending
            .as_ref()
            .expect("camera-set edit must stage the resident bootstrap before preflight");
        assert_eq!(pending.len(), 2);
        assert!(
            pending
                .by_camera
                .values()
                .all(|candidate| candidate.frontier().selection_view_frozen())
        );
        state
            .runtime
            .lock()
            .unwrap()
            .frozen_selection_view_for_test(LodRuntimeViewId(added_camera.to_bits()))
            .expect("rebind must persist the new camera's frozen snapshot")
    };
    world
        .entity_mut(added_camera)
        .insert(GlobalTransform::from(Transform::from_xyz(20.0, 0.0, 25.0)));
    let added_request = (0..16)
        .find_map(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
            let current = state.current.as_ref()?;
            (state.pending.is_none()
                && current.len() == 2
                && current
                    .by_camera
                    .values()
                    .all(LodRenderCandidate::render_is_active))
            .then(|| match state.bootstrap_handoff.as_ref()? {
                PackageBootstrapHandoff::CapacityExceeded { request, .. } => Some(request.clone()),
                PackageBootstrapHandoff::Admitted(_) => None,
            })
            .flatten()
        })
        .expect("added camera must receive one resident bootstrap before restalling");
    assert_ne!(added_request, frozen_request);
    {
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        let current = state.current.as_ref().unwrap();
        assert!(Arc::ptr_eq(&phase, &current.get(camera).unwrap().phase));
        assert!(current.by_camera.values().all(|candidate| {
            candidate.frontier().is_coverage_guard() && candidate.frontier().selection_view_frozen()
        }));
        let expected_fallback = package_candidate_fallback_nodes(current);
        assert_eq!(state.current_fallback_nodes, expected_fallback);
        assert_eq!(state.visible_fallback_nodes, expected_fallback);
        assert_eq!(state.visible_ranges, visible_ranges);
        assert_eq!(state.visible_slots, visible_slots);
        let runtime = state.runtime.lock().unwrap();
        assert!(runtime.contains_view_for_test(LodRuntimeViewId(added_camera.to_bits())));
        assert_eq!(
            runtime.frozen_selection_view_for_test(LodRuntimeViewId(added_camera.to_bits())),
            Some(added_frozen_view)
        );
        assert_eq!(runtime.pending_request_count_for_test(), 0);
        assert_eq!(runtime.transport_request_starts_for_test(), request_starts);
    }

    // Removing that camera releases its runtime state before the matching
    // stalled request can return, then rebinds the bootstrap to the remaining
    // complete live set without transport churn.
    world.get_mut::<Camera>(added_camera).unwrap().is_active = false;
    let removed_request = (0..16)
        .find_map(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
            let current = state.current.as_ref()?;
            (state.pending.is_none()
                && current.len() == 1
                && current
                    .get(camera)
                    .is_some_and(LodRenderCandidate::render_is_active))
            .then(|| match state.bootstrap_handoff.as_ref()? {
                PackageBootstrapHandoff::CapacityExceeded { request, .. } => Some(request.clone()),
                PackageBootstrapHandoff::Admitted(_) => None,
            })
            .flatten()
        })
        .expect("removed camera must be dropped before the remaining view restalls");
    assert_ne!(added_request, removed_request);
    {
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        let current = state.current.as_ref().unwrap();
        assert!(Arc::ptr_eq(&phase, &current.get(camera).unwrap().phase));
        let runtime = state.runtime.lock().unwrap();
        assert!(!runtime.contains_view_for_test(LodRuntimeViewId(added_camera.to_bits())));
        assert_eq!(runtime.pending_request_count_for_test(), 0);
        assert_eq!(runtime.transport_request_starts_for_test(), request_starts);
    }

    // Dynamic camera identity invalidates the memoized rejection and performs
    // one new exact preflight, but the still-oversized target remains quiescent.
    world
        .get_mut::<GaussianLodSettings>(cloud)
        .unwrap()
        .selection_mode = LodSelectionMode::Dynamic;
    world
        .entity_mut(camera)
        .insert(GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 25.0)));
    run_package_frame(&mut schedule, &mut world, cloud);
    {
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        let next_request = state.bootstrap_handoff.as_ref().unwrap().request();
        assert_ne!(next_request, &removed_request);
        let runtime = state.runtime.lock().unwrap();
        assert_eq!(runtime.pending_request_count_for_test(), 0);
        assert_eq!(runtime.transport_request_starts_for_test(), request_starts);
    }

    // A policy edit also invalidates the rejection. The coarsest target fits
    // atomically, so the package is allowed to replace the bootstrap and recover.
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 0.0;
    let mut recovered = false;
    for _ in 0..256 {
        run_package_frame(&mut schedule, &mut world, cloud);
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        recovered = state
            .current
            .as_ref()
            .and_then(|current| current.get(camera))
            .is_some_and(|candidate| {
                candidate.render_is_active() && !candidate.frontier().is_coverage_guard()
            });
        if recovered {
            break;
        }
    }
    assert!(
        recovered,
        "a fitting settings edit must clear the capacity stall"
    );
    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert!(state.bootstrap_handoff.is_none());
    assert!(state.last_failure.is_none());
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn terminal_bootstrap_page_skips_bootstrap_and_publishes_one_honest_ancestor_cut() {
    let mut package = write_native_test_package_with_levels(3);
    let probe_settings = package_test_settings(1.0);
    let bootstrap_page = {
        let (mut probe_world, probe_cloud, _, _) =
            package_test_world(&package, probe_settings, false, 0);
        let mut probe_schedule = Schedule::default();
        probe_schedule.add_systems(update_lod_packages);
        run_package_frame(&mut probe_schedule, &mut probe_world, probe_cloud);
        let manager = probe_world.resource::<GaussianLodPackageManager>();
        let runtime = manager.clouds[&probe_cloud].runtime.lock().unwrap();
        *runtime
            .package_bootstrap_pages_for_test()
            .expect("the progressive fixture must admit a bootstrap")
            .first()
            .unwrap()
    };

    let descriptor = package
        .manifest
        .pages
        .iter_mut()
        .find(|descriptor| descriptor.id == bootstrap_page)
        .unwrap();
    let encoded_len = descriptor.storage.as_ref().unwrap().encoded_len;
    descriptor.storage = Some(LodPageStorage {
        uri: "pages/missing-bootstrap.gspage".to_owned(),
        byte_range: None,
        encoded_len,
    });
    package.manifest.validate().unwrap();

    let mut settings = package_test_settings(1.0);
    settings.budgets.max_requests_per_frame = 1;
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    world
        .resource_mut::<GaussianLodPackageConfig>()
        .streaming
        .max_concurrent_requests = 1;
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let mut activated = Vec::new();
    let mut previous_phase = None;
    for _ in 0..4096 {
        run_package_frame(&mut schedule, &mut world, cloud);
        std::thread::sleep(Duration::from_millis(1));
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        if let Some(candidate) = state
            .pending
            .as_ref()
            .and_then(|pending| pending.get(camera))
        {
            assert!(
                !candidate.frontier().is_coverage_guard(),
                "an incomplete bootstrap must never enter the render transaction"
            );
        }
        if let Some(candidate) = state
            .current
            .as_ref()
            .and_then(|current| current.get(camera))
        {
            assert!(
                !candidate.frontier().is_coverage_guard(),
                "a terminally incomplete bootstrap must never become current"
            );
            if previous_phase
                .as_ref()
                .is_none_or(|phase| !Arc::ptr_eq(phase, &candidate.phase))
            {
                activated.push(candidate.rendered_candidate_count());
                previous_phase = Some(Arc::clone(&candidate.phase));
            }
            if candidate.render_is_active() && state.terminal_failures > 0 {
                break;
            }
        }
    }

    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert!(state.terminal_failures > 0);
    assert_eq!(activated.len(), 1);
    assert!(activated[0] > 0);
    assert!(package_candidate_set_is_active(
        state.current.as_ref().unwrap()
    ));
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn legacy_external_abi_cold_package_retains_exact_only_publication() {
    let mut package = write_native_test_package_with_levels(3);
    package.manifest.build.builder_abi_version = 5;
    package.manifest.build.reducer_version = EXTERNAL_MOMENT_MERGE_VERSION;
    package.manifest.build.config_fingerprint = lod_config_fingerprint_for_reducer(
        package.manifest.build.settings,
        None,
        EXTERNAL_MOMENT_MERGE_VERSION,
    );
    for node in &mut package.manifest.nodes {
        if !node.is_leaf() {
            node.high_fidelity_certificate = 0.0;
        }
    }
    package.manifest.validate().unwrap();
    assert!(
        !package
            .manifest
            .build
            .has_bounded_refinement_amplification()
    );

    let mut settings = package_test_settings(1.0);
    settings.budgets.max_requests_per_frame = 1;
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    world
        .resource_mut::<GaussianLodPackageConfig>()
        .streaming
        .max_concurrent_requests = 1;
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    let exact = package.source_count as u32;

    let (initial_direct_request, request_starts_before_motion) = (0..32)
        .find_map(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            std::thread::sleep(Duration::from_millis(1));
            let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
            let target = state.cold_direct_target.as_ref()?;
            let starts = state
                .runtime
                .lock()
                .unwrap()
                .transport_request_starts_for_test();
            (starts > 0).then(|| (target.request.clone(), starts))
        })
        .expect("legacy direct target must start bounded cold page work");
    world
        .entity_mut(camera)
        .insert(GlobalTransform::from(Transform::from_xyz(4.0, 0.0, 12.0)));
    run_package_frame(&mut schedule, &mut world, cloud);
    {
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        let target = state
            .cold_direct_target
            .as_ref()
            .expect("camera motion must replace, not drop, the eligible direct target");
        assert_ne!(target.request, initial_direct_request);
        let runtime = state.runtime.lock().unwrap();
        let pending = runtime.pending_request_count_for_test();
        let mut direct_footprint = target.plan.pages().clone();
        direct_footprint.extend(runtime.active_coverage_guard_pages().iter().copied());
        direct_footprint.extend(
            runtime
                .hierarchy()
                .roots()
                .iter()
                .filter_map(|root| runtime.hierarchy().page(*root)),
        );
        assert!(
            pending <= direct_footprint.len(),
            "replaced direct demand must contain only target/root/guard work: pending={pending}, footprint_pages={}",
            direct_footprint.len()
        );
    }

    let mut activated = Vec::new();
    let mut previous_phase = None;
    for _ in 0..4096 {
        run_package_frame(&mut schedule, &mut world, cloud);
        std::thread::sleep(Duration::from_millis(1));
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        for candidate in state
            .pending
            .iter()
            .chain(state.current.iter())
            .flat_map(|candidates| candidates.by_camera.values())
        {
            assert!(
                !candidate.frontier().is_coverage_guard(),
                "legacy singleton coarse rungs must not use the cold bootstrap path"
            );
        }
        if let Some(candidate) = state
            .current
            .as_ref()
            .and_then(|current| current.get(camera))
        {
            if previous_phase
                .as_ref()
                .is_none_or(|phase| !Arc::ptr_eq(phase, &candidate.phase))
            {
                activated.push(candidate.rendered_candidate_count());
                previous_phase = Some(Arc::clone(&candidate.phase));
            }
            if candidate.rendered_candidate_count() == exact && candidate.render_is_active() {
                break;
            }
        }
    }

    assert_eq!(activated, [exact]);
    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert!(
        state
            .runtime
            .lock()
            .unwrap()
            .transport_request_starts_for_test()
            > request_starts_before_motion,
        "the replacement direct target must resume bounded request progress"
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn stationary_replacement_retains_current_until_one_fixed_point_activation() {
    let package = write_native_test_package(false);
    let mut settings = package_test_settings(0.0);
    settings.budgets.max_requests_per_frame = 1;
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    world
        .resource_mut::<GaussianLodPackageConfig>()
        .streaming
        .max_concurrent_requests = 1;
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let coarse_count = package.manifest.quality.coarsest_gaussian_count as u32;
    let exact_count = package.source_count as u32;
    drive_package_to_active_count(&mut schedule, &mut world, cloud, camera, coarse_count);
    let (coarse_phase, coarse_ranges, coarse_slots, resident_before) = {
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        (
            Arc::clone(&state.current.as_ref().unwrap().get(camera).unwrap().phase),
            state.visible_ranges.clone(),
            state.visible_slots.clone(),
            state.resident_pages,
        )
    };

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    let mut saw_resident_growth = false;
    let mut saw_exact_pending = false;
    let mut exact_activations = 0;
    let mut exact_was_current = false;
    for _ in 0..4096 {
        run_package_frame(&mut schedule, &mut world, cloud);
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        let current = state.current.as_ref().expect("coarse cut stays retained");
        let current_candidate = current
            .get(camera)
            .expect("camera current candidate exists");
        let current_count = current_candidate.rendered_candidate_count();
        assert!(
            current_count == coarse_count || current_count == exact_count,
            "stationary streaming published an intermediate ancestor wave of {current_count} records"
        );
        if current_count == coarse_count {
            saw_resident_growth |= state.resident_pages > resident_before;
            assert!(Arc::ptr_eq(&current_candidate.phase, &coarse_phase));
            assert_eq!(state.visible_ranges, coarse_ranges);
            assert_eq!(state.visible_slots, coarse_slots);
        }
        if let Some(pending) = &state.pending {
            assert_eq!(
                pending.get(camera).unwrap().rendered_candidate_count(),
                exact_count,
                "only the drained replacement may enter the render handshake"
            );
            saw_exact_pending = true;
        }
        let exact_is_current = current_count == exact_count;
        if exact_is_current && !exact_was_current {
            exact_activations += 1;
        }
        exact_was_current = exact_is_current;
        if exact_is_current && current_candidate.render_is_active() {
            break;
        }
    }

    assert!(
        saw_resident_growth,
        "detail pages must stream while the current cut remains unchanged"
    );
    assert!(
        saw_exact_pending,
        "the drained exact replacement must stage"
    );
    assert_eq!(exact_activations, 1);
    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert!(state.pending.is_none());
    assert_eq!(state.active_gaussians, u64::from(exact_count));
    assert!(package_candidate_set_is_active(
        state.current.as_ref().unwrap()
    ));
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn package_cap_plus_one_cut_stages_bounded_prefix_and_activates_once() {
    let package = write_native_test_package(false);
    let stride = package
        .manifest
        .pages
        .iter()
        .map(|page| page.gaussian_count)
        .max()
        .unwrap();
    let bytes_per_slot = u64::from(stride) * gaussian_3d_gpu_bytes_per_record();
    let mut settings = package_test_settings(0.0);
    settings.budgets.max_gpu_upload_bytes_per_commit = bytes_per_slot;
    settings.budgets.max_upload_bytes_per_frame = bytes_per_slot;
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    let coarse = package.manifest.quality.coarsest_gaussian_count as u32;
    drive_package_to_active_count(&mut schedule, &mut world, cloud, camera, coarse);
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    let mut transaction_snapshot = None;
    let mut transaction_upload_frames = 0;
    let mut found_multiframe_transaction = false;
    let mut exact_activation_count = 0;
    let mut exact_was_current = false;
    for _ in 0..2048 {
        let queued = run_package_frame(&mut schedule, &mut world, cloud);
        assert!(queued <= 1, "one-slot cap admitted {queued} package slots");
        let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
        if let Some(pending) = &state.pending {
            transaction_upload_frames += usize::from(queued != 0);
            let retained = transaction_snapshot.get_or_insert_with(|| {
                (
                    state.current_page_leases.clone(),
                    state.visible_ranges.clone(),
                    state.visible_slots.clone(),
                )
            });
            assert_eq!(
                state.pending_page_leases,
                package_candidate_pages(pending),
                "the full replacement must be leased before any prefix is staged"
            );
            assert_eq!(state.current_page_leases, retained.0);
            assert_eq!(state.visible_ranges, retained.1);
            assert_eq!(state.visible_slots, retained.2);
        } else if transaction_snapshot.take().is_some() {
            found_multiframe_transaction |= transaction_upload_frames > 1;
            transaction_upload_frames = 0;
        }
        let exact_is_current = state
            .current
            .as_ref()
            .and_then(|current| current.get(camera))
            .is_some_and(|candidate| {
                candidate.frontier().candidate_count() == package.source_count as u32
            });
        if exact_is_current && !exact_was_current {
            exact_activation_count += 1;
        }
        exact_was_current = exact_is_current;
        if exact_is_current
            && state
                .current
                .as_ref()
                .and_then(|current| current.get(camera))
                .is_some_and(LodRenderCandidate::render_is_active)
        {
            break;
        }
    }

    assert!(
        found_multiframe_transaction,
        "at least one cap+1 replacement must span frames"
    );
    assert_eq!(exact_activation_count, 1);
    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert!(state.pending.is_none());
    assert!(state.pending_page_leases.is_empty());
    assert!(state.staged.is_none());
    assert_eq!(
        state.current_page_leases,
        package_candidate_pages(state.current.as_ref().unwrap())
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn canceling_partial_staging_retains_resident_sparse_upload() {
    let package = write_native_test_package(false);
    let stride = package
        .manifest
        .pages
        .iter()
        .map(|page| page.gaussian_count)
        .max()
        .unwrap();
    let bytes_per_slot = u64::from(stride) * gaussian_3d_gpu_bytes_per_record();
    let mut settings = package_test_settings(0.0);
    settings.budgets.max_gpu_upload_bytes_per_commit = bytes_per_slot;
    settings.budgets.max_upload_bytes_per_frame = bytes_per_slot;
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.manifest.quality.coarsest_gaussian_count as u32,
    );

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    let (deferred, ticket, retained_slots) = (0..2048)
        .find_map(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            let descriptor = world
                .resource::<LodAtlasUploadQueue>()
                .queued_slots()
                .next()?;
            let manager = world.resource::<GaussianLodPackageManager>();
            let state = &manager.clouds[&cloud];
            let staged = state.staged.as_ref()?;
            (!staged.complete
                && state.visible_slots.get(&descriptor.slot.index) != Some(&descriptor.slot))
            .then(|| {
                assert!(state.transient_atlas.snapshot_slot(descriptor).is_ok());
                (
                    descriptor,
                    state.transient_atlas.ticket().clone(),
                    state.visible_slots.clone(),
                )
            })
        })
        .expect("one-slot staging must leave a deferred sparse descriptor");

    // Do not cross the extraction boundary. Cancellation drops only pending
    // cut ownership; the runtime-resident payload and its queued GPU cache
    // write remain valid and reusable.
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 0.0;
    schedule.run(&mut world);
    assert!(
        world
            .resource::<LodAtlasUploadQueue>()
            .queued_slots()
            .any(|queued| queued == deferred)
    );
    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert_eq!(state.visible_slots, retained_slots);
    assert!(package_candidate_set_is_active(
        state.current.as_ref().unwrap()
    ));
    assert!(
        state.staged.as_ref().is_none_or(|staged| {
            staged.slots.get(&deferred.slot.index) != Some(&deferred.slot)
        })
    );
    assert!(state.transient_atlas.snapshot_slot(deferred).is_ok());
    assert!(state.mirror.materialized_slots().contains(&deferred.slot));
    assert_eq!(
        ticket.generation(),
        state.transient_atlas.ticket().generation()
    );
    assert!(!ticket.is_failed());

    // A fresh transaction may reuse the physical index under a newer allocator
    // generation and still reaches ACTIVE without a stale snapshot poisoning
    // the atlas ticket.
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.source_count as u32,
    );
    assert!(!ticket.is_failed());
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn native_package_stages_whole_slots_and_rejects_subpage_cap() {
    let package = write_native_test_package(false);
    let mut settings = package_test_settings(0.0);
    let stride = package
        .manifest
        .pages
        .iter()
        .map(|page| page.gaussian_count)
        .max()
        .unwrap();
    let bytes_per_slot = u64::from(stride) * gaussian_3d_gpu_bytes_per_record();
    settings.budgets.max_gpu_upload_bytes_per_commit = bytes_per_slot;
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    let coarse = package.manifest.quality.coarsest_gaussian_count as u32;
    drive_package_to_active_count(&mut schedule, &mut world, cloud, camera, coarse);
    let plan = world.resource::<GaussianLodPackageManager>().clouds[&cloud].plan;
    assert!(plan.slot_count > 1);
    assert!(plan.physical_bytes > bytes_per_slot);
    assert!(world.resource::<LodAtlasUploadQueue>().queued_slot_count() <= 1);

    let mut too_small = package_test_settings(0.0);
    too_small.budgets.max_gpu_upload_bytes_per_commit = bytes_per_slot - 1;
    let config = world.resource::<GaussianLodPackageConfig>();
    assert_eq!(
        GaussianLodPackageAtlasPlan::from_manifest(&package.manifest, &too_small, config),
        Err(GaussianLodPackageError::GpuUploadCommitTooLarge {
            dirty_slots: 1,
            bytes: bytes_per_slot,
            limit: bytes_per_slot - 1,
        })
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn multiple_packages_share_one_live_main_world_staging_budget_fairly() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, first, camera, manifest_handle) =
        package_test_world(&package, settings.clone(), false, 0);
    let second = spawn_package_test_cloud(&mut world, &package, manifest_handle, settings, false);
    mark_package_cloud_visible(&mut world, camera, second);

    let gaussians_per_slot = package
        .manifest
        .pages
        .iter()
        .map(|page| page.gaussian_count)
        .max()
        .unwrap();
    let canonical_slot_bytes = u64::from(gaussians_per_slot) * size_of::<Gaussian3d>() as u64;
    world.insert_resource(LodAtlasUploadBudget::try_new(canonical_slot_bytes, 1).unwrap());
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let mut first_materialized_frame = None;
    let mut second_materialized_frame = None;
    for frame in 0_u32..2048 {
        let queued = run_package_frame_for_clouds(&mut schedule, &mut world, &[first, second]);
        assert!(
            queued <= 1,
            "one package frame queued {queued} slots despite the live one-slot global budget"
        );
        let manager = world.resource::<GaussianLodPackageManager>();
        if first_materialized_frame.is_none()
            && manager
                .clouds
                .get(&first)
                .is_some_and(|state| state.transient_atlas.materialized_slot_count().unwrap() > 0)
        {
            first_materialized_frame = Some(frame);
        }
        if second_materialized_frame.is_none()
            && manager
                .clouds
                .get(&second)
                .is_some_and(|state| state.transient_atlas.materialized_slot_count().unwrap() > 0)
        {
            second_materialized_frame = Some(frame);
        }
        if first_materialized_frame.is_some() && second_materialized_frame.is_some() {
            break;
        }
    }
    let first_frame = first_materialized_frame.expect("first package must receive staging tokens");
    let second_frame =
        second_materialized_frame.expect("second package must receive staging tokens fairly");
    assert_ne!(
        first_frame, second_frame,
        "a one-slot global frame budget cannot materialize both package clouds together"
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn package_reports_typed_capacity_failure_for_globally_oversized_slot() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, _, _) = package_test_world(&package, settings, false, 0);
    let gaussians_per_slot = package
        .manifest
        .pages
        .iter()
        .map(|page| page.gaussian_count)
        .max()
        .unwrap();
    let canonical_slot_bytes = u64::from(gaussians_per_slot) * size_of::<Gaussian3d>() as u64;
    world.insert_resource(
        LodAtlasUploadBudget::try_new(canonical_slot_bytes.saturating_sub(1), 1).unwrap(),
    );
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    for _ in 0..2048 {
        run_package_frame(&mut schedule, &mut world, cloud);
        if world
            .get::<GaussianLodPackageStatus>(cloud)
            .and_then(|status| status.failure.as_ref())
            .is_some_and(|failure| failure.code() == LodOrchestrationFailureCode::CapacityExceeded)
        {
            break;
        }
    }
    let status = world.get::<GaussianLodPackageStatus>(cloud).unwrap();
    assert_eq!(
        status.failure.as_ref().map(LodOrchestrationFailure::code),
        Some(LodOrchestrationFailureCode::CapacityExceeded)
    );
    assert!(
        status
            .error_detail()
            .is_some_and(|detail| detail.contains("MainWorldStagingBudget"))
    );
    let state = &world.resource::<GaussianLodPackageManager>().clouds[&cloud];
    assert_eq!(state.transient_atlas.materialized_slot_count().unwrap(), 0);
}

#[test]
fn package_streaming_settings_validate_and_rebuild_signatures_track_config() {
    let required = GaussianStreamingSettings::default();
    assert_eq!(
        package_streaming_settings(&required).unwrap(),
        required.clone()
    );
    let invalid = GaussianStreamingSettings {
        max_concurrent_requests: 0,
        ..required.clone()
    };
    assert!(matches!(
        package_streaming_settings(&invalid),
        Err(GaussianLodPackageError::InvalidStreaming(_))
    ));
    let render_path = validate_package_render_path(&crate::sort::SortMode::default());
    assert_eq!(
        render_path.is_ok(),
        crate::stream::lod_render_path_is_supported()
    );
    if !crate::stream::lod_render_path_is_supported() {
        assert_eq!(
            render_path,
            Err(GaussianLodPackageError::UnsupportedRenderPath(
                LodRenderPathSupportError::UnsupportedBuildConfiguration
            ))
        );
    }
    let cached = GaussianStreamingSettings {
        persistent_cache: true,
        ..required.clone()
    };
    let effective = package_streaming_settings(&cached).unwrap();
    assert!(effective.persistent_cache);

    let manifest = AssetId::<GaussianLodAsset>::default();
    let source = GaussianLodPackageSource::native_directory("scene");
    let config = GaussianLodPackageConfig::default();
    let lod_settings = GaussianLodSettings::default();
    let structural = PackageStructuralSignature::new(&lod_settings);
    let current = PackageBuildSignature {
        manifest,
        source: &source,
        config: &config,
        streaming: &required,
        structural,
    };
    assert!(
        current
            == PackageBuildSignature {
                manifest,
                source: &source,
                config: &config,
                streaming: &required,
                structural,
            }
    );

    let mut structural_change = config.clone();
    structural_change.max_atlas_gaussians /= 2;
    assert!(
        current
            != PackageBuildSignature {
                manifest,
                source: &source,
                config: &structural_change,
                streaming: &required,
                structural,
            }
    );
    let effective_streaming_change = GaussianStreamingSettings {
        retry_limit: required.retry_limit + 1,
        ..required.clone()
    };
    assert!(
        current
            != PackageBuildSignature {
                manifest,
                source: &source,
                config: &config,
                streaming: &effective_streaming_change,
                structural,
            }
    );
}

#[test]
fn http_package_has_one_authoritative_retry_budget() {
    let streaming = GaussianStreamingSettings {
        retry_limit: 3,
        ..default()
    };
    let http = package_runtime_streaming_settings(
        &GaussianLodPackageSource::url("https://cdn.example/scene/"),
        &streaming,
    );
    assert_eq!(http.retry_limit, 0);
    let native = package_runtime_streaming_settings(
        &GaussianLodPackageSource::native_directory("scene"),
        &streaming,
    );
    assert_eq!(native.retry_limit, 3);
    // HttpRangePageTransport's request-count regression separately proves
    // its configured retry budget emits only R + 1 bounded attempts.
}

#[test]
fn hundred_million_virtual_source_does_not_scale_physical_allocation() {
    let settings = GaussianLodSettings::default();
    let mut config = GaussianLodPackageConfig::default();
    config.max_atlas_gaussians = 16_384;
    config.max_atlas_bytes = u64::MAX;
    let plan =
        GaussianLodPackageAtlasPlan::from_limits(134_217_728, 4_096, &settings, &config).unwrap();
    assert_eq!(plan.virtual_source_gaussians, 134_217_728);
    assert_eq!(plan.physical_gaussians, 16_384);
    assert_eq!(plan.slot_count, 4);
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn package_atlas_plan_preserves_manifest_root_validation() {
    let package = write_native_test_package(false);
    let mut manifest = package.manifest.clone();
    manifest.roots.push(LodNodeId(u64::MAX));

    assert!(matches!(
        GaussianLodPackageAtlasPlan::from_manifest(
            &manifest,
            &package_test_settings(0.0),
            &GaussianLodPackageConfig::default(),
        ),
        Err(GaussianLodPackageError::InvalidManifest(_))
    ));
}

#[test]
fn memory_backed_package_reaches_q0_and_exact_q1_frontiers() {
    let source = LodTestScene::nested_octants(3).cloud();
    let built = build_planar_3d_lod(
        &source,
        GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 8,
            support_sigma: 3.0,
        },
    )
    .unwrap();
    let mut transport = MemoryPageTransport::default();
    for page in &built.pages {
        transport.insert(page.id, encode_page(page).unwrap());
    }
    // Cooperative preprocessing performs a bounded checksum slice and a
    // bounded decode slice on separate application frames. Scale the eventual
    // bound with the physical package instead of assuming the older one-page
    // decode cadence.
    let max_updates = built.manifest.pages.len().saturating_mul(3) + 16;

    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
    settings.budgets.max_resident_pages = 256;
    settings.budgets.max_pending_requests = 512;
    settings.budgets.max_requests_per_frame = 256;
    settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
    let mut streaming = GaussianStreamingSettings::default();
    streaming.persistent_cache = false;
    streaming.retry_limit = 0;
    let mut runtime =
        LodStreamingRuntime::new(built.manifest, transport, &settings, &streaming).unwrap();
    let view = LodView::perspective(Vec3::new(0.0, 0.0, 5.0), 720.0, 1.0, 0.1);

    let drive = |runtime: &mut LodStreamingRuntime<MemoryPageTransport>,
                 settings: &GaussianLodSettings,
                 expected: Option<u32>| {
        let mut last_summary = String::new();
        for _ in 0..max_updates {
            let frame = runtime
                .update_view(LodRuntimeViewId(7), view, settings, &streaming)
                .unwrap();
            match frame.candidate_frontier(settings.max_active_gaussians_u32()) {
                Ok(candidate) => {
                    last_summary = format!(
                        "candidate={} requested={:?} cache={:?} preprocess={:?}",
                        candidate.candidate_count(),
                        frame.frontier().requested_nodes,
                        frame.cache_stats(),
                        frame.preprocess_stats(),
                    );
                    if expected.is_none_or(|count| candidate.candidate_count() == count) {
                        return candidate;
                    }
                }
                Err(error) => {
                    last_summary = format!(
                        "candidate_error={error:?} requested={:?} cache={:?} preprocess={:?}",
                        frame.frontier().requested_nodes,
                        frame.cache_stats(),
                        frame.preprocess_stats(),
                    );
                }
            }
        }
        let terminal = runtime
            .terminal_failures()
            .iter()
            .map(|&page| (page, runtime.page_preprocess_error(page).cloned()))
            .collect::<Vec<_>>();
        panic!(
            "package runtime did not reach the requested complete cut; last={last_summary}; terminal={terminal:?}"
        )
    };

    let coarse = drive(&mut runtime, &settings, None);
    assert!(coarse.candidate_count() > 0);
    assert!((coarse.candidate_count() as usize) < source.len());

    settings.quality = 1.0;
    let exact = drive(&mut runtime, &settings, Some(source.len() as u32));
    assert_eq!(exact.candidate_count() as usize, source.len());
    assert!(exact.candidate_count() > coarse.candidate_count());
}

#[test]
fn corrupt_leaf_retains_a_complete_ancestor_cut() {
    let source = LodTestScene::nested_octants(3).cloud();
    let built = build_planar_3d_lod(
        &source,
        GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 8,
            support_sigma: 3.0,
        },
    )
    .unwrap();
    let root_pages = built
        .manifest
        .roots
        .iter()
        .filter_map(|root| {
            built
                .manifest
                .nodes
                .iter()
                .find(|node| node.id == *root)
                .map(|node| node.representation.page)
        })
        .collect::<BTreeSet<_>>();
    let corrupt_page = built
        .manifest
        .nodes
        .iter()
        .find(|node| node.children.is_empty() && !root_pages.contains(&node.representation.page))
        .map(|node| node.representation.page)
        .expect("fixture must contain a non-root leaf page");
    let mut transport = MemoryPageTransport::default();
    for page in &built.pages {
        let mut bytes = encode_page(page).unwrap();
        if page.id == corrupt_page {
            let last = bytes.last_mut().expect("encoded page is non-empty");
            *last ^= 0x5a;
        }
        transport.insert(page.id, bytes);
    }
    let max_updates = built.manifest.pages.len().saturating_mul(3) + 16;

    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.frustum_culling = false;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
    settings.budgets.max_resident_pages = 256;
    settings.budgets.max_pending_requests = 512;
    settings.budgets.max_requests_per_frame = 256;
    settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
    let mut streaming = GaussianStreamingSettings::default();
    streaming.retry_limit = 0;
    let mut runtime =
        LodStreamingRuntime::new(built.manifest, transport, &settings, &streaming).unwrap();
    let view = LodView::perspective(Vec3::new(0.0, 0.0, 5.0), 720.0, 1.0, 0.1);

    let coarse = (0..max_updates)
        .find_map(|_| {
            runtime
                .update_view(LodRuntimeViewId(17), view, &settings, &streaming)
                .ok()?
                .candidate_frontier(settings.max_active_gaussians_u32())
                .ok()
        })
        .expect("root cut must become resident before refinement");
    settings.quality = 1.0;
    let degraded = (0..max_updates)
        .find_map(|_| {
            let frame = runtime
                .update_view(LodRuntimeViewId(17), view, &settings, &streaming)
                .ok()?;
            runtime.is_terminal_failure(corrupt_page).then(|| {
                frame
                    .candidate_frontier(settings.max_active_gaussians_u32())
                    .expect("ancestor fallback must remain a complete resident cut")
            })
        })
        .expect("corrupt leaf must exhaust its retry budget");
    assert!(runtime.is_terminal_failure(corrupt_page));
    assert!(degraded.candidate_count() >= coarse.candidate_count());
    assert!((degraded.candidate_count() as usize) < source.len());
    assert!(
        degraded
            .physical_ranges()
            .iter()
            .all(|range| range.page != corrupt_page)
    );
}

#[test]
fn sparse_multi_camera_atlas_rewrites_are_bounded_and_generation_safe() {
    let source = LodTestScene::nested_octants(3).cloud();
    let built = build_planar_3d_lod(
        &source,
        GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 8,
            support_sigma: 3.0,
        },
    )
    .unwrap();
    let manifest = built.manifest.clone();
    let max_updates = manifest.pages.len().saturating_mul(3) + 16;
    let mut transport = MemoryPageTransport::default();
    for page in &built.pages {
        transport.insert(page.id, encode_page(page).unwrap());
    }
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.frustum_culling = false;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
    settings.budgets.max_resident_pages = 256;
    settings.budgets.max_pending_requests = 512;
    settings.budgets.max_requests_per_frame = 256;
    settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
    let mut streaming = GaussianStreamingSettings::default();
    streaming.persistent_cache = false;
    streaming.retry_limit = 0;
    let mut runtime =
        LodStreamingRuntime::new(manifest.clone(), transport, &settings, &streaming).unwrap();
    let stride = runtime.atlas_layout().gaussians_per_slot;
    let slot_count = settings.budgets.max_resident_pages;
    let physical_gaussians = slot_count * stride;
    let plan = GaussianLodPackageAtlasPlan {
        virtual_source_gaussians: source.len() as u64,
        gaussians_per_slot: stride,
        slot_count,
        physical_gaussians,
        physical_bytes: u64::from(physical_gaussians) * gaussian_3d_gpu_bytes_per_record(),
    };
    let mut mirror = LodPageAtlasMirror::new(runtime.atlas_layout(), slot_count).unwrap();
    let mut debug = PackageDebugAnnotations {
        atlas: LodDebugAnnotationAtlas::new(slot_count, stride).unwrap(),
        index: Arc::new(LodDebugManifestIndex::new(&manifest).unwrap()),
        initialization: VecDeque::new(),
        page_bases: HashMap::new(),
    };

    let mut drive = |view_id: LodRuntimeViewId,
                     view: LodView,
                     settings: &GaussianLodSettings,
                     expected: Option<u32>| {
        for _ in 0..max_updates {
            let frame = runtime
                .update_view(view_id, view, settings, &streaming)
                .unwrap();
            for &page in frame.completed_pages() {
                let slot = runtime.cache().get(page).unwrap().slot;
                mirror.stage_page(page, slot).unwrap();
            }
            if let Ok(candidate) = frame.candidate_frontier(settings.max_active_gaussians_u32())
                && expected.is_none_or(|count| candidate.candidate_count() == count)
            {
                return candidate;
            }
        }
        panic!("runtime did not reach requested candidate")
    };

    let root = drive(
        PACKAGE_ROOT_FALLBACK_VIEW,
        LodView::perspective(Vec3::ZERO, 1.0, 1.0, 0.1),
        &settings,
        None,
    );
    settings.quality = 1.0;
    let left = drive(
        LodRuntimeViewId(11),
        LodView::perspective(Vec3::new(-4.0, 0.0, 5.0), 720.0, 1.0, 0.1),
        &settings,
        Some(source.len() as u32),
    );
    let right = drive(
        LodRuntimeViewId(12),
        LodView::perspective(Vec3::new(4.0, 0.0, 5.0), 720.0, 1.0, 0.1),
        &settings,
        Some(source.len() as u32),
    );
    assert_eq!(left.candidate_count(), right.candidate_count());

    // Exercise an exact multi-camera cut, the cold-start root fallback, and a
    // later replacement directly. Upload work must scale with old/new visible
    // slots rather than the configured atlas capacity. Runtime orchestration
    // retains an active current cut instead of using this root rewrite between
    // ordinary replacements; that behavior is covered by the integration tests.
    let mut atlas =
        PlanarGaussian3d::from(vec![Gaussian3d::default(); physical_gaussians as usize]);
    let exact_rewrite = rewrite_atlas_to_frontiers(
        &runtime,
        &mut mirror,
        Some(&mut debug),
        plan,
        &[left.clone(), right.clone()],
        &BTreeSet::new(),
        &BTreeMap::new(),
        &mut atlas,
    )
    .unwrap();
    assert!(!exact_rewrite.selected_slots.is_empty());
    assert_eq!(
        exact_rewrite.selection_scratch.slots,
        exact_rewrite.selected_slots.len()
    );
    assert!(
        exact_rewrite.selection_scratch.intervals
            <= left.physical_ranges().len() + right.physical_ranges().len()
    );
    assert!(exact_rewrite.dirty_slots.len() < plan.slot_count as usize);
    let exact_slots = exact_rewrite.selected_slots.clone();
    let exact_atlas_id = AssetId::<PlanarGaussian3d>::default();
    let mut exact_uploads = LodAtlasUploadQueue::default();
    enqueue_package_atlas_uploads(&mut exact_uploads, exact_atlas_id, plan, &exact_rewrite)
        .unwrap();
    assert_eq!(
        exact_uploads.queued_slot_count(),
        exact_rewrite.dirty_slots.len()
    );
    assert!(exact_uploads.queued_slots().all(|upload| {
        upload.slot.generation == exact_rewrite.selected_slots[&upload.slot.index].generation
    }));
    let exact_materialized = atlas
        .iter()
        .enumerate()
        .filter_map(|(index, gaussian)| {
            (gaussian.scale_opacity.opacity != 0.0).then_some(index as u32)
        })
        .collect::<BTreeSet<_>>();

    let root_rewrite = rewrite_atlas_to_frontiers(
        &runtime,
        &mut mirror,
        Some(&mut debug),
        plan,
        std::slice::from_ref(&root),
        &BTreeSet::new(),
        &exact_slots,
        &mut atlas,
    )
    .unwrap();
    let actual = atlas
        .iter()
        .enumerate()
        .filter_map(|(index, gaussian)| {
            (gaussian.scale_opacity.opacity != 0.0).then_some(index as u32)
        })
        .collect::<BTreeSet<_>>();
    let expected = root
        .physical_ranges()
        .iter()
        .flat_map(|range| range.physical_start..range.end().unwrap())
        .collect::<BTreeSet<_>>();
    let expected_additive = exact_materialized
        .union(&expected)
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected_additive);
    assert!(expected.is_subset(&actual));

    let atlas_id = AssetId::<PlanarGaussian3d>::default();
    let mut uploads = LodAtlasUploadQueue::default();
    enqueue_package_atlas_uploads(&mut uploads, atlas_id, plan, &root_rewrite).unwrap();
    let queued = uploads.queued_slots().collect::<Vec<_>>();
    assert_eq!(queued.len(), root_rewrite.dirty_slots.len());
    assert!(queued.len() < plan.slot_count as usize);
    for upload in queued {
        assert_eq!(upload.atlas, atlas_id);
        assert_eq!(upload.gaussians_per_slot, stride);
        assert!(upload.slot.index < plan.slot_count);
        assert_eq!(
            upload.slot.generation,
            root_rewrite
                .selected_slots
                .get(&upload.slot.index)
                .map_or(0, |slot| slot.generation)
        );
        assert!(
            upload.slot.index * stride + stride <= plan.physical_gaussians,
            "queued slot range must stay within the bounded atlas"
        );
    }

    // Caller-provided prior-cut metadata does not force an identical,
    // materialized page generation to upload again. The mirror is the
    // authoritative full-slot cache record.
    let mut previous_generation_slots = root_rewrite.selected_slots.clone();
    for slot in previous_generation_slots.values_mut() {
        slot.generation = if slot.generation == 1 { 2 } else { 1 };
    }
    let generation_rewrite = rewrite_atlas_to_frontiers(
        &runtime,
        &mut mirror,
        Some(&mut debug),
        plan,
        std::slice::from_ref(&root),
        &BTreeSet::new(),
        &previous_generation_slots,
        &mut atlas,
    )
    .unwrap();
    assert_eq!(
        generation_rewrite.selected_slots,
        root_rewrite.selected_slots
    );
    assert!(generation_rewrite.dirty_slots.is_empty());

    let mut churn_uploads = LodAtlasUploadQueue::default();
    let churn_rewrite = rewrite_atlas_to_frontiers(
        &runtime,
        &mut mirror,
        Some(&mut debug),
        plan,
        &[left.clone(), right],
        &BTreeSet::new(),
        &root_rewrite.selected_slots,
        &mut atlas,
    )
    .unwrap();
    enqueue_package_atlas_uploads(&mut churn_uploads, atlas_id, plan, &churn_rewrite).unwrap();
    assert_eq!(
        churn_uploads.queued_slot_count(),
        churn_rewrite.dirty_slots.len()
    );
    assert!(churn_rewrite.dirty_slots.is_empty());
    assert!(churn_uploads.queued_slot_count() < plan.slot_count as usize);

    let camera = Entity::from_bits(11);
    let mut current = LodRenderCandidates::default();
    current.insert(camera, left);
    current
        .get(camera)
        .unwrap()
        .phase
        .store(LOD_RENDER_ACTIVE, Ordering::Release);
    assert!(package_candidate_set_is_active(&current));
    current.get(camera).unwrap().phase.store(
        crate::stream::render_commit::LOD_RENDER_WAITING,
        Ordering::Release,
    );
    assert!(!package_candidate_set_is_active(&current));
}
