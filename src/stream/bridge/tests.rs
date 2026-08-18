#![allow(clippy::field_reassign_with_default)]

use super::*;
use crate::{
    gaussian::formats::planar_3d::Gaussian3d,
    stream::runtime::{LodPhysicalRange, PageAtlasLayout},
    testing::LodTestScene,
};

fn test_config() -> GaussianLodBridgeConfig {
    GaussianLodBridgeConfig {
        max_ephemeral_source_gaussians: 4096,
        max_ephemeral_stored_gaussians: 8192,
        max_atlas_gaussians: 8192,
        max_atlas_bytes: 16 * 1024 * 1024,
        build_settings: GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 16,
            support_sigma: 3.0,
        },
        ..Default::default()
    }
}

fn test_streaming() -> GaussianStreamingSettings {
    GaussianStreamingSettings {
        max_concurrent_requests: 3,
        ..Default::default()
    }
}

#[test]
fn default_bridge_stored_budget_covers_progressive_hierarchy_overhead() {
    let config = GaussianLodBridgeConfig::default();
    assert_eq!(config.build_settings, GaussianLodBuildSettings::default());
    let source = u64::from(config.max_ephemeral_source_gaussians);
    // Binary topology and the validated amplification factor (at least 2)
    // make all internal representation levels a geometric series below N.
    let maximum_progressive_storage = source * 2;
    assert!(config.max_ephemeral_stored_gaussians >= maximum_progressive_storage);
}

#[test]
fn active_failure_is_reported_as_degraded_not_failed() {
    assert_eq!(
        bridge_status_transition_kind(GaussianLodBridgePhase::Active, true, false),
        Some(LodOrchestrationTransitionKind::Degraded)
    );
    assert_eq!(
        bridge_status_transition_kind(GaussianLodBridgePhase::CompleteFallback, true, false),
        Some(LodOrchestrationTransitionKind::Failed)
    );
    assert_eq!(
        bridge_status_transition_kind(GaussianLodBridgePhase::Active, false, true),
        Some(LodOrchestrationTransitionKind::Recovered)
    );
}

#[test]
fn fallback_atlas_is_complete_and_padded() {
    let source = LodTestScene::nested_octants(2).cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_resident_pages = 32;
    let source_handle = Handle::default();
    let config = test_config();
    let (state, mut atlas) = create_ephemeral_bridge(
        source_handle,
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    assert!(state.debug_atlas.is_none());
    assert!(state.debug_manifest_index.is_none());
    assert!(state.debug_slots.is_empty());
    assert_eq!(atlas.len(), state.mirror.physical_gaussians() as usize);
    assert!(atlas.len() >= source.len());
    assert_eq!(
        atlas.iter().take(source.len()).collect::<Vec<_>>(),
        source.iter().collect::<Vec<_>>()
    );

    Planar::set(&mut atlas, 0, Gaussian3d::default());
    restore_complete_flat_fallback(&mut atlas, &source).unwrap();
    assert_eq!(atlas.get(0), source.get(0));
}

#[test]
fn atlas_capacity_stops_at_the_complete_manifest_union() {
    let source = LodTestScene::nested_octants(2).cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.5;
    settings.budgets.max_resident_pages = 4096;
    let config = test_config();
    let built = build_planar_3d_lod(&source, config.build_settings).unwrap();
    let stride = built
        .manifest
        .pages
        .iter()
        .map(|page| page.gaussian_count)
        .max()
        .unwrap();
    let fallback_slots = (source.len() as u32).div_ceil(stride);
    let complete_manifest_union = built.manifest.header.page_count;
    let maximum_useful_slots = fallback_slots.max(complete_manifest_union);

    let (state, atlas) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();

    assert!(complete_manifest_union >= fallback_slots);
    assert_eq!(state.mirror.slot_count(), maximum_useful_slots);
    assert_eq!(atlas.len(), (maximum_useful_slots * stride) as usize);
    assert!(
        state.mirror.slot_count() < settings.budgets.max_resident_pages,
        "the manifest bound, rather than the source-independent resident cap, must size the atlas"
    );
    assert!(
        state.mirror.slot_count() >= complete_manifest_union,
        "every possible multi-view cut must fit when no physical budget binds"
    );
}

#[test]
fn flat_atlas_capacity_and_complete_fallback_obey_gpu_byte_cap() {
    let source = LodTestScene::nested_octants(2).cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_resident_pages = 32;
    let config = test_config();
    let built = build_planar_3d_lod(&source, config.build_settings).unwrap();
    let stride = built
        .manifest
        .pages
        .iter()
        .map(|page| page.gaussian_count)
        .max()
        .unwrap();
    let required_slots = (source.len() as u32).div_ceil(stride);
    let bytes_per_slot = u64::from(stride) * gaussian_3d_gpu_bytes_per_record();
    let required = u64::from(required_slots) * bytes_per_slot;

    settings.budgets.max_gpu_upload_bytes_per_commit = required - 1;
    let error = match create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    ) {
        Ok(_) => panic!("sub-fallback GPU cap unexpectedly built an atlas"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        LodBridgeError::CompleteFallbackExceedsGpuUploadBudget {
            required,
            limit: required - 1,
        }
    );

    settings.budgets.max_gpu_upload_bytes_per_commit = required;
    let (state, _) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    let physical_gpu_bytes =
        u64::from(state.mirror.physical_gaussians()) * gaussian_3d_gpu_bytes_per_record();
    assert!(physical_gpu_bytes <= required);
    assert!(physical_gpu_bytes <= config.max_atlas_bytes);
    assert_eq!(
        gaussian_3d_gpu_bytes_per_record(),
        size_of::<Gaussian3d>() as u64
            + if cfg!(feature = "precompute_covariance_3d") {
                size_of::<crate::gaussian::f32::Covariance3dOpacity>() as u64
            } else {
                0
            }
    );
}

#[test]
fn atlas_generation_rejects_residency_churn() {
    let layout = PageAtlasLayout::new(8).unwrap();
    let mut mirror = LodPageAtlasMirror::new(layout, 2).unwrap();
    let first = AtlasSlot {
        index: 0,
        generation: 1,
    };
    mirror.stage_page(LodPageId(1), first).unwrap();
    let mut atlas = PlanarGaussian3d::from(vec![Gaussian3d::default(); 16]);
    let first_page = PlanarGaussian3dPage {
        schema_version: crate::gaussian::formats::planar_3d_chunked::LOD_PAGE_SCHEMA_VERSION,
        id: LodPageId(1),
        gaussians: vec![Gaussian3d::default()],
    };
    mirror
        .materialize_page(&mut atlas, &first_page, first)
        .unwrap();
    let old_range = LodPhysicalRange {
        node: crate::LodNodeId(1),
        page: LodPageId(1),
        slot: first,
        physical_start: 0,
        count: 1,
    };
    assert!(mirror.is_range_current(old_range));

    mirror
        .stage_page(
            LodPageId(2),
            AtlasSlot {
                index: 0,
                generation: 2,
            },
        )
        .unwrap();
    assert!(!mirror.is_range_current(old_range));
}

#[test]
fn render_handshake_requires_every_camera_before_activation() {
    let left = Arc::new(AtomicU8::new(LOD_RENDER_PREPARED));
    let right = Arc::new(AtomicU8::new(LOD_RENDER_WAITING));
    let phases = [Arc::clone(&left), Arc::clone(&right)];
    assert!(!phases.iter().all(|phase| matches!(
        phase.load(Ordering::Acquire),
        LOD_RENDER_PREPARED | LOD_RENDER_ACTIVE
    )));
    right.store(LOD_RENDER_PREPARED, Ordering::Release);
    assert!(phases.iter().all(|phase| matches!(
        phase.load(Ordering::Acquire),
        LOD_RENDER_PREPARED | LOD_RENDER_ACTIVE
    )));
}

#[test]
fn ephemeral_runtime_multi_camera_reaches_q0_and_q1_complete_cuts() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, _) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    let views = [
        (
            LodRuntimeViewId(1),
            LodView::perspective(Vec3::ZERO, 720.0, 1.0, 0.1),
        ),
        (
            LodRuntimeViewId(2),
            LodView::perspective(Vec3::new(10.0, 0.0, 0.0), 720.0, 1.0, 0.1),
        ),
    ];

    let drive = |state: &mut BridgeCloudState,
                 settings: &GaussianLodSettings,
                 required_count: Option<usize>|
     -> Vec<LodCandidateFrontier> {
        let effective = state.structural.apply(settings);
        for _ in 0..64 {
            let frame = state.runtime.begin_frame();
            let mut candidates = Vec::new();
            for (id, view) in views {
                let result = state
                    .runtime
                    .update_view_in_frame(frame, id, view, &effective, &state.streaming)
                    .unwrap();
                for &page in result.completed_pages() {
                    let slot = state.runtime.resident_slot(page).unwrap();
                    state.mirror.stage_page(page, slot).unwrap();
                }
                if let Ok(candidate) =
                    result.candidate_frontier(settings.max_active_gaussians_u32())
                {
                    candidates.push(candidate);
                }
            }
            if candidates.len() == views.len()
                && required_count.is_none_or(|count| {
                    candidates
                        .iter()
                        .all(|candidate| candidate.candidate_count() as usize == count)
                })
            {
                return candidates;
            }
        }
        panic!("multi-camera runtime did not reach a complete resident cut")
    };

    let coarse = drive(&mut state, &settings, None);
    assert!(
        coarse
            .iter()
            .all(|candidate| candidate.candidate_count() > 0)
    );
    let coarse_count = coarse[0].candidate_count();
    settings.quality = 1.0;
    let exact = drive(&mut state, &settings, Some(source.len()));
    assert!(
        exact
            .iter()
            .all(|candidate| candidate.candidate_count() as usize == source.len())
    );
    assert!(exact[0].candidate_count() >= coarse_count);

    let camera = Entity::from_bits(1);
    let coarse_phase = state.handshake_for(camera, &coarse[0]);
    coarse_phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    let exact_phase = state.handshake_for(camera, &exact[0]);
    assert!(!Arc::ptr_eq(&coarse_phase, &exact_phase));
    assert_eq!(coarse_phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    assert_eq!(exact_phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    assert!(Arc::ptr_eq(
        &exact_phase,
        &state.handshake_for(camera, &exact[0])
    ));
}

#[test]
fn gpu_upload_commit_budget_is_exact_and_deduplicates_camera_slots() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, _) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    let effective = state.structural.apply(&settings);
    let view = LodView::perspective(Vec3::ZERO, 720.0, 1.0, 0.1);
    let frontier = (0..64)
        .find_map(|_| {
            let frame = state.runtime.begin_frame();
            let stream_frame = state
                .runtime
                .update_view_in_frame(
                    frame,
                    LodRuntimeViewId(41),
                    view,
                    &effective,
                    &state.streaming,
                )
                .unwrap();
            for &page in stream_frame.completed_pages() {
                let slot = state.runtime.resident_slot(page).unwrap();
                state.mirror.stage_page(page, slot).unwrap();
            }
            stream_frame
                .candidate_frontier(settings.max_active_gaussians_u32())
                .ok()
        })
        .expect("runtime should produce a complete resident frontier");

    let dirty_slot_count = frontier
        .physical_ranges()
        .iter()
        .map(|range| range.slot.index)
        .collect::<BTreeSet<_>>()
        .len() as u64;
    let candidate_count = frontier.candidate_count();
    assert!(dirty_slot_count > 0);
    let required = dirty_slot_count
        * u64::from(state.mirror.layout().gaussians_per_slot)
        * gaussian_3d_gpu_bytes_per_record();

    let mut candidates = LodRenderCandidates::default();
    candidates.insert(Entity::from_bits(1), frontier.clone());
    candidates.insert(Entity::from_bits(2), frontier);
    assert!(
        candidates
            .by_camera
            .values()
            .all(|candidate| candidate.rendered_candidate_count() == candidate_count)
    );
    assert_eq!(
        pending_gpu_upload_bytes(&state, &candidates).unwrap(),
        required
    );
    assert_eq!(
        validate_gpu_upload_commit_budget(&state, &candidates, required).unwrap(),
        required
    );
    assert_eq!(
        validate_gpu_upload_commit_budget(&state, &candidates, required - 1),
        Err(LodBridgeError::GpuUploadCommitBudgetExceeded {
            required,
            limit: required - 1,
        })
    );
}

#[test]
fn shared_page_bridge_sync_acquires_once_and_invalidates_sibling_fallback_swap() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 1.0;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, mut atlas) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        true,
    )
    .unwrap();
    let effective = state.structural.apply(&settings);
    let view = LodView::perspective(Vec3::ZERO, 720.0, 1.0, 0.1);
    let frontier = (0..128)
        .find_map(|_| {
            let frame = state.runtime.begin_frame();
            let stream_frame = state
                .runtime
                .update_view_in_frame(
                    frame,
                    LodRuntimeViewId(71),
                    view,
                    &effective,
                    &state.streaming,
                )
                .unwrap();
            for &page in stream_frame.completed_pages() {
                let slot = state.runtime.resident_slot(page).unwrap();
                state.mirror.stage_page(page, slot).unwrap();
            }
            let candidate = stream_frame
                .candidate_frontier(settings.max_active_gaussians_u32())
                .ok()?;
            let mut pages = BTreeSet::new();
            candidate
                .physical_ranges()
                .iter()
                .any(|range| !pages.insert(range.page))
                .then_some(candidate)
        })
        .expect("runtime should produce the exact shared-page frontier");
    let mut ranges_by_page = BTreeMap::<LodPageId, Vec<LodPhysicalRange>>::new();
    for &range in frontier.physical_ranges() {
        ranges_by_page.entry(range.page).or_default().push(range);
    }
    let shared_ranges = ranges_by_page
        .values()
        .find(|ranges| ranges.len() >= 2)
        .expect("fixture must select sibling ranges from one physical page");
    let first = shared_ranges[0];
    let second = shared_ranges[1];
    assert_eq!(first.page, second.page);
    assert_eq!(first.slot, second.slot);

    let unique_pages = ranges_by_page.len() as u64;
    let mut candidates = LodRenderCandidates::default();
    candidates.insert(Entity::from_bits(71), frontier);
    let mut atlas_uploads = LodAtlasUploadQueue::default();
    let atlas_id = state.atlas.id();
    state.decoded_page_acquisitions = 0;
    synchronize_bridge_candidate_pages(
        &mut state,
        &candidates,
        &BTreeSet::from([first.node]),
        atlas_id,
        &mut atlas,
        &mut atlas_uploads,
    )
    .unwrap();
    assert_eq!(state.decoded_page_acquisitions, unique_pages);
    assert_eq!(atlas_uploads.queued_slot_count(), ranges_by_page.len());
    let first_revision = state.debug_revision;
    let first_metadata = state.debug_atlas.as_ref().unwrap().metadata();
    assert!(
        first_metadata.records()[first.physical_start as usize..first.end().unwrap() as usize]
            .iter()
            .all(|record| record.residency_code() == LodDebugResidency::AncestorFallback as u32)
    );
    assert!(
        first_metadata.records()[second.physical_start as usize..second.end().unwrap() as usize]
            .iter()
            .all(|record| record.residency_code() == LodDebugResidency::Resident as u32)
    );

    synchronize_bridge_candidate_pages(
        &mut state,
        &candidates,
        &BTreeSet::from([first.node]),
        atlas_id,
        &mut atlas,
        &mut atlas_uploads,
    )
    .unwrap();
    assert_eq!(state.decoded_page_acquisitions, unique_pages);
    assert_eq!(state.debug_revision, first_revision);

    synchronize_bridge_candidate_pages(
        &mut state,
        &candidates,
        &BTreeSet::from([second.node]),
        atlas_id,
        &mut atlas,
        &mut atlas_uploads,
    )
    .unwrap();
    assert_eq!(state.decoded_page_acquisitions, unique_pages + 1);
    assert_eq!(state.debug_revision, first_revision + 1);
    let second_metadata = state.debug_atlas.as_ref().unwrap().metadata();
    assert!(
        second_metadata.records()[first.physical_start as usize..first.end().unwrap() as usize]
            .iter()
            .all(|record| record.residency_code() == LodDebugResidency::Resident as u32)
    );
    assert!(
        second_metadata.records()[second.physical_start as usize..second.end().unwrap() as usize]
            .iter()
            .all(|record| record.residency_code() == LodDebugResidency::AncestorFallback as u32)
    );
}

#[test]
fn bridge_debug_atlas_tracks_slot_generation_and_keeps_fallback_unannotated() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, _) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        true,
    )
    .unwrap();
    assert!(state.debug_atlas.is_some());
    assert!(state.debug_manifest_index.is_some());
    assert!(state.fallback_debug_metadata.is_empty());

    let view = LodView::perspective(Vec3::ZERO, 720.0, 1.0, 0.1);
    let (page_id, first_slot) = (0..64)
        .find_map(|_| {
            let frame = state.runtime.begin_frame();
            let result = state
                .runtime
                .update_view_in_frame(
                    frame,
                    LodRuntimeViewId(7),
                    view,
                    &state.structural.apply(&settings),
                    &state.streaming,
                )
                .unwrap();
            result.completed_pages().first().map(|&page| {
                let slot = state.runtime.resident_slot(page).unwrap();
                state.stage_completed_page(page, slot).unwrap();
                (page, slot)
            })
        })
        .expect("runtime should decode a page");
    assert_eq!(
        state.debug_atlas.as_ref().unwrap().page(first_slot),
        Some(page_id)
    );
    let indexed_nodes = state
        .debug_manifest_index
        .as_ref()
        .unwrap()
        .node_indices(page_id)
        .unwrap();
    assert!(!indexed_nodes.is_empty());
    let indexed_nodes_ptr = indexed_nodes.as_ptr();

    let page = state.runtime.decoded_page(page_id).unwrap();
    let next_slot = AtlasSlot {
        index: first_slot.index,
        generation: first_slot.generation.wrapping_add(1).max(1),
    };
    let fallback_nodes = state
        .debug_manifest_index
        .as_ref()
        .unwrap()
        .node_ids(page_id)
        .unwrap()
        .collect::<BTreeSet<_>>();
    state
        .sync_debug_page(
            &page,
            next_slot,
            &fallback_nodes.into_iter().collect::<Vec<_>>(),
        )
        .unwrap();
    let debug_atlas = state.debug_atlas.as_ref().unwrap();
    assert_eq!(debug_atlas.page(first_slot), None);
    assert_eq!(debug_atlas.page(next_slot), Some(page_id));
    let offset = next_slot.index as usize * debug_atlas.records_per_slot() as usize;
    assert_eq!(
        debug_atlas.metadata().records()[offset].residency_code(),
        LodDebugResidency::AncestorFallback as u32
    );
    assert_eq!(
        state
            .debug_manifest_index
            .as_ref()
            .unwrap()
            .node_indices(page_id)
            .unwrap()
            .as_ptr(),
        indexed_nodes_ptr
    );
}

#[test]
fn flat_streaming_settings_are_cloned_per_cloud() {
    let config = test_config();
    let original = GaussianLodSettings::default();
    assert_eq!(original.quality_endpoint(), LodQualityEndpoint::Original);
    assert_eq!(validate_flat_lod_render_path(&original), Ok(()));
    let continuous = GaussianLodSettings {
        quality: 0.5,
        ..default()
    };
    assert_eq!(
        validate_flat_lod_render_path(&continuous).is_ok(),
        crate::stream::lod_render_path_is_supported()
    );
    if !crate::stream::lod_render_path_is_supported() {
        assert_eq!(
            validate_flat_lod_render_path(&continuous),
            Err(LodBridgeError::UnsupportedRenderPath(
                LodRenderPathSupportError::UnsupportedBuildConfiguration
            ))
        );
    }
    assert_eq!(
        flat_streaming_settings(None, &config).unwrap(),
        config.streaming_settings
    );
    let per_cloud = test_streaming();
    assert_eq!(
        flat_streaming_settings(Some(&per_cloud), &config).unwrap(),
        per_cloud
    );

    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(config);
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();

    let source = LodTestScene::nested_octants(2).cloud();
    let source_handle = world.resource_mut::<Assets<PlanarGaussian3d>>().add(source);
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_resident_pages = 128;
    let spawn = |world: &mut World, streaming: Option<GaussianStreamingSettings>| {
        let mut entity = world.spawn((
            PlanarGaussian3dHandle(source_handle.clone()),
            settings.clone(),
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ));
        if let Some(streaming) = streaming {
            entity.insert(streaming);
        }
        entity.id()
    };
    let default_ephemeral = spawn(&mut world, None);
    let custom_ephemeral = spawn(&mut world, Some(test_streaming()));

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    schedule.run(&mut world);

    for entity in [default_ephemeral, custom_ephemeral] {
        let actual = world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id();
        if crate::stream::lod_render_path_is_supported() {
            assert_ne!(actual, source_handle.id());
        } else {
            assert_eq!(actual, source_handle.id());
            let status = world.get::<GaussianLodBridgeStatus>(entity).unwrap();
            assert_eq!(status.phase, GaussianLodBridgePhase::CompleteFallback);
            assert_eq!(
                status.failure.as_ref().map(LodOrchestrationFailure::code),
                Some(LodOrchestrationFailureCode::UnsupportedConfiguration)
            );
            assert_eq!(
                status.error_detail(),
                Some(
                    LodBridgeError::UnsupportedRenderPath(
                        LodRenderPathSupportError::UnsupportedBuildConfiguration
                    )
                    .to_string()
                    .as_str()
                )
            );
        }
    }
    assert_eq!(
        world.resource::<GaussianLodBridgeManager>().clouds.len(),
        if crate::stream::lod_render_path_is_supported() {
            2
        } else {
            0
        }
    );
}

#[test]
fn structural_setting_changes_retire_before_rebuilding() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();

    let source = LodTestScene::nested_octants(2).cloud();
    let source_handle = world.resource_mut::<Assets<PlanarGaussian3d>>().add(source);
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_resident_pages = 128;
    let streaming = test_streaming();
    let entity = world
        .spawn((
            PlanarGaussian3dHandle(source_handle.clone()),
            settings,
            streaming,
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();
    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    schedule.run(&mut world);
    let old_atlas = world
        .get::<PlanarGaussian3dHandle>(entity)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(old_atlas.id(), source_handle.id());

    world
        .get_mut::<GaussianLodSettings>(entity)
        .unwrap()
        .budgets
        .max_resident_pages = 64;
    world
        .get_mut::<GaussianStreamingSettings>(entity)
        .unwrap()
        .max_encoded_page_bytes /= 2;
    schedule.run(&mut world);

    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    assert!(world.get::<LodRenderCandidates>(entity).is_none());
    assert!(
        world
            .resource::<GaussianLodBridgeManager>()
            .clouds
            .is_empty()
    );
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(entity).unwrap().phase,
        GaussianLodBridgePhase::Building
    );

    schedule.run(&mut world);
    let new_atlas = world
        .get::<PlanarGaussian3dHandle>(entity)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(new_atlas.id(), source_handle.id());
    assert_ne!(new_atlas.id(), old_atlas.id());
    let state = &world.resource::<GaussianLodBridgeManager>().clouds[&entity];
    assert_eq!(state.signature.max_resident_pages, 64);
    assert_eq!(
        state.signature.max_encoded_page_bytes,
        world
            .get::<GaussianStreamingSettings>(entity)
            .unwrap()
            .max_encoded_page_bytes
    );

    let current_settings = world.get::<GaussianLodSettings>(entity).unwrap();
    let current_streaming = world.get::<GaussianStreamingSettings>(entity).unwrap();
    let config = world.resource::<GaussianLodBridgeConfig>();
    let current_signature =
        BridgeStructuralSignature::new(current_settings, current_streaming, config, false);
    let mut lower_commit_cap = current_settings.clone();
    lower_commit_cap.budgets.max_gpu_upload_bytes_per_commit -= 1;
    assert_ne!(
        current_signature,
        BridgeStructuralSignature::new(&lower_commit_cap, current_streaming, config, false),
        "the atomic GPU cap changes the physical atlas and must rebuild it"
    );
}

#[test]
fn flat_source_remove_and_readd_same_id_releases_the_old_bridge() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();

    let source = LodTestScene::nested_octants(2).cloud();
    let source_handle = world.resource_mut::<Assets<PlanarGaussian3d>>().add(source);
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_resident_pages = 128;
    let entity = world
        .spawn((
            PlanarGaussian3dHandle(source_handle.clone()),
            settings,
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    schedule.run(&mut world);
    let old_atlas = world
        .get::<PlanarGaussian3dHandle>(entity)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(old_atlas.id(), source_handle.id());

    let removed_source = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .remove(source_handle.id())
        .expect("flat source exists before removal");
    world
        .resource_mut::<Messages<AssetEvent<PlanarGaussian3d>>>()
        .write(AssetEvent::Removed {
            id: source_handle.id(),
        });
    schedule.run(&mut world);

    assert!(
        world
            .resource::<GaussianLodBridgeManager>()
            .clouds
            .is_empty()
    );
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&old_atlas)
            .is_none()
    );
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    assert!(world.get::<LodRenderCandidates>(entity).is_none());
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(entity).unwrap().phase,
        GaussianLodBridgePhase::Building
    );

    world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .insert(source_handle.id(), removed_source)
        .expect("removed flat-source ID can be reinserted");
    world
        .resource_mut::<Messages<AssetEvent<PlanarGaussian3d>>>()
        .write(AssetEvent::Added {
            id: source_handle.id(),
        });
    schedule.run(&mut world);

    let new_atlas = world
        .get::<PlanarGaussian3dHandle>(entity)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(new_atlas.id(), source_handle.id());
    assert_ne!(new_atlas.id(), old_atlas.id());
    assert!(
        world
            .resource::<GaussianLodBridgeManager>()
            .clouds
            .contains_key(&entity)
    );
}

#[test]
fn in_place_source_mutation_restores_every_slot_and_invalidates_the_cut() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();

    let source = LodTestScene::screen_space_ladder().cloud();
    let source_handle = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(source.clone());
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let cloud = world
        .spawn((
            PlanarGaussian3dHandle(source_handle.clone()),
            settings,
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();
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
            GaussianCamera::default(),
        ))
        .id();

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    let candidate = (0..64)
        .find_map(|_| {
            schedule.run(&mut world);
            world
                .get::<LodRenderCandidates>(cloud)
                .and_then(|candidates| candidates.get(camera))
                .cloned()
        })
        .expect("a complete resident cut should be staged before mutation");
    let atlas_handle = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    let (stride, physical_gaussians) = {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.mirror.materialized_slots().is_empty());
        (
            state.mirror.layout().gaussians_per_slot,
            state.mirror.physical_gaussians(),
        )
    };
    let target_index = source.len() - 1;
    let target_slot = target_index as u32 / stride;
    assert!(target_slot > 0, "the regression must cover a later slot");
    let replacement = source.get(0);
    assert_ne!(source.get(target_index), replacement);
    candidate
        .phase
        .store(LOD_RENDER_PREPARED, Ordering::Release);

    {
        let mut assets = world.resource_mut::<Assets<PlanarGaussian3d>>();
        let mut source_asset = assets.get_mut(&source_handle).unwrap();
        Planar::set(&mut *source_asset, target_index, replacement);
    }
    world
        .resource_mut::<Messages<AssetEvent<PlanarGaussian3d>>>()
        .write(AssetEvent::Modified {
            id: source_handle.id(),
        });
    schedule.run(&mut world);

    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    assert!(world.get::<LodRenderCandidates>(cloud).is_none());
    assert_eq!(candidate.phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    assert!(
        world
            .resource::<GaussianLodBridgeManager>()
            .clouds
            .is_empty()
    );
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Building
    );
    assert_eq!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&atlas_handle)
            .unwrap()
            .get(target_index),
        replacement
    );
    let mut queued = world
        .resource::<LodAtlasUploadQueue>()
        .queued_slots()
        .filter(|upload| upload.atlas == atlas_handle.id())
        .collect::<Vec<_>>();
    queued.sort_by_key(|upload| upload.slot.index);
    assert_eq!(
        queued.len(),
        (physical_gaussians / stride) as usize,
        "source mutation must dirty even slots never materialized by LoD"
    );
    assert!(queued.iter().all(|upload| upload.slot.generation == 0));
    assert!(queued.iter().any(|upload| upload.slot.index == target_slot));

    schedule.run(&mut world);
    let rebuilt_atlas = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(rebuilt_atlas.id(), source_handle.id());
    assert_ne!(rebuilt_atlas.id(), atlas_handle.id());
    assert_eq!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&rebuilt_atlas)
            .unwrap()
            .get(target_index),
        replacement
    );
}

#[test]
fn automatic_bridge_builds_bounded_fallback_and_quality_one_restores_source() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();

    let source = LodTestScene::nested_octants(2).cloud();
    let source_handle = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(source.clone());
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_resident_pages = 128;
    let entity = world
        .spawn((
            PlanarGaussian3dHandle(source_handle.clone()),
            settings,
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();
    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    schedule.run(&mut world);

    let bridged = world.get::<PlanarGaussian3dHandle>(entity).unwrap();
    assert_ne!(bridged.handle().id(), source_handle.id());
    let atlas = world
        .resource::<Assets<PlanarGaussian3d>>()
        .get(bridged.handle())
        .unwrap();
    assert!(atlas.len() >= source.len());
    assert!(atlas.len() <= test_config().max_atlas_gaussians as usize);
    assert_eq!(
        atlas.iter().take(source.len()).collect::<Vec<_>>(),
        source.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(entity).unwrap().phase,
        GaussianLodBridgePhase::StreamingFallback
    );
    assert!(world.get::<LodRenderCandidates>(entity).is_none());
    assert!(world.get::<LodDebugMetadata>(entity).is_some());

    world
        .get_mut::<GaussianLodSettings>(entity)
        .unwrap()
        .quality = 1.0;
    schedule.run(&mut world);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    assert!(world.get::<GaussianLodBridgeStatus>(entity).is_none());
    assert!(world.get::<LodRenderCandidates>(entity).is_none());
    assert!(world.get::<LodDebugMetadata>(entity).is_none());
    assert!(
        world
            .resource::<GaussianLodBridgeManager>()
            .clouds
            .is_empty()
    );
}

#[test]
fn automatic_bridge_limit_error_preserves_complete_source() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    let mut config = test_config();
    config.max_ephemeral_source_gaussians = 1;
    world.insert_resource(config);
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();

    let source = LodTestScene::nested_octants(2).cloud();
    let source_handle = world.resource_mut::<Assets<PlanarGaussian3d>>().add(source);
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    let entity = world
        .spawn((
            PlanarGaussian3dHandle(source_handle.clone()),
            settings,
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();
    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    schedule.run(&mut world);

    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    let status = world.get::<GaussianLodBridgeStatus>(entity).unwrap();
    assert_eq!(status.phase, GaussianLodBridgePhase::CompleteFallback);
    assert!(status.failure.is_some());
    assert!(world.get::<LodRenderCandidates>(entity).is_none());
    assert!(world.get::<LodDebugMetadata>(entity).is_none());
}

#[test]
fn automatic_bridge_requires_exact_multi_camera_commit_before_materializing_pages() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();

    let source = LodTestScene::screen_space_ladder().cloud();
    let source_handle = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(source.clone());
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let cloud = world
        .spawn((
            PlanarGaussian3dHandle(source_handle.clone()),
            settings,
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();

    let spawn_camera = |world: &mut World, position: Vec3| {
        world
            .spawn((
                Camera {
                    viewport: Some(bevy::camera::Viewport {
                        physical_size: UVec2::new(1280, 720),
                        ..default()
                    }),
                    ..default()
                },
                Projection::Perspective(default()),
                GlobalTransform::from(Transform::from_translation(position)),
                GaussianCamera::default(),
            ))
            .id()
    };
    let left = spawn_camera(&mut world, Vec3::new(0.0, 0.0, 5.0));
    let right = spawn_camera(&mut world, Vec3::new(8.0, 0.0, 5.0));

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    let candidates = (0..64)
        .find_map(|_| {
            schedule.run(&mut world);
            world
                .get::<LodRenderCandidates>(cloud)
                .filter(|candidates| candidates.len() == 2)
                .cloned()
        })
        .unwrap_or_else(|| {
            panic!(
                "both camera cuts should become resident; status={:?}, candidate_count={:?}",
                world.get::<GaussianLodBridgeStatus>(cloud),
                world
                    .get::<LodRenderCandidates>(cloud)
                    .map(LodRenderCandidates::len)
            )
        });
    assert!(candidates.get(left).is_some());
    assert!(candidates.get(right).is_some());
    candidates
        .get(left)
        .unwrap()
        .phase
        .store(LOD_RENDER_PREPARED, Ordering::Release);
    schedule.run(&mut world);

    assert_ne!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );
    let atlas_handle = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    let atlas = world
        .resource::<Assets<PlanarGaussian3d>>()
        .get(&atlas_handle)
        .unwrap();
    assert_eq!(
        atlas.iter().take(source.len()).collect::<Vec<_>>(),
        source.iter().collect::<Vec<_>>()
    );

    let active_candidates = (0..64)
        .find_map(|_| {
            let candidates = world.get::<LodRenderCandidates>(cloud)?.clone();
            if candidates.len() != 2 {
                schedule.run(&mut world);
                return None;
            }
            for candidate in candidates.by_camera.values() {
                candidate
                    .phase
                    .store(LOD_RENDER_PREPARED, Ordering::Release);
            }
            schedule.run(&mut world);
            (world.get::<GaussianLodBridgeStatus>(cloud)?.phase == GaussianLodBridgePhase::Active)
                .then(|| world.get::<LodRenderCandidates>(cloud).unwrap().clone())
        })
        .expect("all-camera prepared cut should activate");
    assert_eq!(active_candidates.len(), 2);
    let queued_uploads = world
        .resource::<LodAtlasUploadQueue>()
        .queued_slots()
        .collect::<Vec<_>>();
    assert!(!queued_uploads.is_empty());
    for (camera, candidate) in &active_candidates.by_camera {
        assert!(matches!(*camera, value if value == left || value == right));
        assert_eq!(
            candidate.frontier.candidate_count() as u64,
            candidate
                .frontier
                .physical_ranges()
                .iter()
                .map(|range| u64::from(range.count))
                .sum::<u64>()
        );
        assert_eq!(candidate.phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);
        for range in candidate.frontier.physical_ranges() {
            assert!(queued_uploads.iter().any(|upload| {
                upload.atlas == atlas_handle.id()
                    && upload.slot == range.slot
                    && upload.gaussians_per_slot
                        == world
                            .resource::<GaussianLodBridgeManager>()
                            .clouds
                            .get(&cloud)
                            .unwrap()
                            .mirror
                            .layout()
                            .gaussians_per_slot
            }));
        }
    }

    world.entity_mut(cloud).insert(ViewVisibility::HIDDEN);
    schedule.run(&mut world);
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::StreamingFallback
    );
    assert!(world.get::<LodRenderCandidates>(cloud).is_none());
    let suspended_state = world
        .resource::<GaussianLodBridgeManager>()
        .clouds
        .get(&cloud)
        .unwrap();
    assert!(suspended_state.views.is_empty());
    assert!(suspended_state.handshakes.is_empty());
    assert!(!suspended_state.active);
    let suspended_atlas = world
        .resource::<Assets<PlanarGaussian3d>>()
        .get(&atlas_handle)
        .unwrap();
    assert_eq!(
        suspended_atlas
            .iter()
            .take(source.len())
            .collect::<Vec<_>>(),
        source.iter().collect::<Vec<_>>()
    );

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    schedule.run(&mut world);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    let retired_atlas = world
        .resource::<Assets<PlanarGaussian3d>>()
        .get(&atlas_handle)
        .unwrap();
    assert_eq!(
        retired_atlas.iter().take(source.len()).collect::<Vec<_>>(),
        source.iter().collect::<Vec<_>>()
    );
    assert!(world.get::<LodRenderCandidates>(cloud).is_none());
}

#[test]
fn interior_high_cut_renders_retained_source_and_keeps_status_candidate() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();

    let source = LodTestScene::checkerboard_facade(8, 8).cloud();
    let source_handle = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(source.clone());
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.99;
    settings.selection_mode = crate::LodSelectionMode::Frozen;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let cloud = world
        .spawn((
            PlanarGaussian3dHandle(source_handle.clone()),
            settings,
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();
    let camera = world
        .spawn((
            Camera {
                viewport: Some(bevy::camera::Viewport {
                    physical_size: UVec2::new(3840, 2160),
                    ..default()
                }),
                ..default()
            },
            Projection::Perspective(default()),
            GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 1.0)),
            GaussianCamera::default(),
        ))
        .id();

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    let candidate = (0..128)
        .find_map(|_| {
            schedule.run(&mut world);
            let candidate = world.get::<LodRenderCandidates>(cloud)?.get(camera)?;
            (!candidate.requires_compaction()).then(|| candidate.clone())
        })
        .unwrap_or_else(|| {
            panic!(
                "interior high-quality cut did not reach retained-source bypass; status={:?}, count={:?}",
                world.get::<GaussianLodBridgeStatus>(cloud),
                world
                    .get::<LodRenderCandidates>(cloud)
                    .and_then(|candidates| candidates.get(camera))
                    .map(LodRenderCandidate::rendered_candidate_count)
            )
        });

    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );
    assert_eq!(candidate.rendered_candidate_count() as usize, source.len());
    let rendered_quality = candidate.rendered_quality_status();
    assert_eq!(rendered_quality.achieved_max_error_px, 0.0);
    assert_eq!(rendered_quality.achieved_max_target_ratio, 0.0);
    assert_eq!(rendered_quality.degradation, crate::LodDegradation::None);
    assert_eq!(rendered_quality.active_gaussians as usize, source.len());
    assert!(candidate.render_is_prepared());
    assert!(candidate.frontier().selection_view_frozen());
    assert_eq!(
        candidate.frontier().quality_status().requested_target,
        world
            .get::<GaussianLodSettings>(cloud)
            .unwrap()
            .quality_target()
    );
    let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
    assert!(state.flat_source_bypass);
    assert!(!state.active);
    assert!(state.owns_render_handle(source_handle.id()));

    let mut cloud_settings = CloudSettings::default();
    cloud_settings
        .lod_debug
        .apply_preset(crate::LodDebugPreset::Level);
    world.entity_mut(cloud).insert(cloud_settings);
    let debug_candidate = (0..128)
        .find_map(|_| {
            schedule.run(&mut world);
            world
                .get::<LodRenderCandidates>(cloud)?
                .get(camera)
                .filter(|candidate| candidate.requires_compaction())
                .cloned()
        })
        .expect("debug annotations must retain the atlas compaction path");
    assert!(debug_candidate.requires_compaction());
    assert_ne!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    assert!(!world.resource::<GaussianLodBridgeManager>().clouds[&cloud].flat_source_bypass);
}

#[test]
fn retained_source_bypass_threshold_is_inclusive_at_five_percent_savings() {
    assert!(!retained_flat_source_bypass_is_eligible(
        1_000,
        1,
        false,
        [949].into_iter(),
    ));
    assert!(retained_flat_source_bypass_is_eligible(
        1_000,
        1,
        false,
        [950].into_iter(),
    ));
}

#[test]
fn retained_source_bypass_requires_every_active_camera_and_debug_off() {
    assert!(!retained_flat_source_bypass_is_eligible(
        1_000,
        2,
        false,
        [1_000, 949].into_iter(),
    ));
    assert!(retained_flat_source_bypass_is_eligible(
        1_000,
        2,
        false,
        [950, 1_000].into_iter(),
    ));
    assert!(!retained_flat_source_bypass_is_eligible(
        1_000,
        2,
        false,
        [1_000].into_iter(),
    ));
    assert!(!retained_flat_source_bypass_is_eligible(
        1_000,
        2,
        true,
        [1_000, 1_000].into_iter(),
    ));
}
