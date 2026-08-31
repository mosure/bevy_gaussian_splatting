#![allow(clippy::field_reassign_with_default)]

use std::sync::atomic::AtomicU32;

use super::*;
use crate::{
    gaussian::formats::{planar_3d::Gaussian3d, planar_3d_chunked::LodPageStorage},
    stream::{
        runtime::{LodPhysicalRange, PageAtlasLayout},
        transport::{PagePoll, PageRequest},
    },
    testing::LodTestScene,
};

#[cfg(all(not(target_arch = "wasm32"), lod_render_path))]
use crate::render::lod::cold_staging_candidate_phase;

#[cfg(not(target_arch = "wasm32"))]
use crate::gaussian::formats::planar_3d_lod::{GaussianLodQualityMetadata, LodError};

struct FirstPollsThenPendingTransport {
    inner: MemoryPageTransport,
    remaining_ready: Arc<AtomicU32>,
}

impl LodPageTransport for FirstPollsThenPendingTransport {
    type Ticket = <MemoryPageTransport as LodPageTransport>::Ticket;
    type Error = <MemoryPageTransport as LodPageTransport>::Error;

    fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
        self.inner.begin(request)
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
        let permitted = self
            .remaining_ready
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok();
        if permitted {
            self.inner.poll(ticket)
        } else {
            PagePoll::Pending
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        self.inner.cancel(ticket);
    }
}

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

fn run_bridge_frame(world: &mut World, schedule: &mut Schedule) {
    schedule.run(world);
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::sleep(std::time::Duration::from_millis(1));
}

fn mark_cloud_visible(world: &mut World, camera: Entity, cloud: Entity) {
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

fn mark_cloud_hidden(world: &mut World, camera: Entity, cloud: Entity) {
    world
        .get_mut::<VisibleEntities>(camera)
        .expect("Camera must provide VisibleEntities")
        .get_mut(TypeId::of::<CloudVisibilityClass>())
        .retain(|entity| *entity != cloud);
}

fn run_until_bridge_initialization_finishes(
    world: &mut World,
    schedule: &mut Schedule,
    entities: &[Entity],
) {
    for _ in 0..2_000 {
        run_bridge_frame(world, schedule);
        let manager = world.resource::<GaussianLodBridgeManager>();
        if entities.iter().all(|entity| {
            manager.clouds.contains_key(entity)
                || world
                    .get::<GaussianLodBridgeStatus>(*entity)
                    .is_some_and(|status| status.failure.is_some())
        }) {
            return;
        }
    }
    panic!("ephemeral bridge initialization did not finish");
}

#[cfg(not(target_arch = "wasm32"))]
fn cold_prepared_bridge_fixture() -> (
    World,
    Schedule,
    Entity,
    Entity,
    Handle<PlanarGaussian3d>,
    Handle<PlanarGaussian3d>,
    LodRenderCandidates,
) {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();
    world.init_resource::<LodTransientAtlasRegistry>();

    let source = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(LodTestScene::screen_space_ladder().cloud());
    let mut settings = GaussianLodSettings {
        quality: 0.0,
        ..Default::default()
    };
    settings.budgets.max_requests_per_frame = 1;
    settings.budgets.max_pending_requests = 128;
    let cloud = world
        .spawn((
            PlanarGaussian3dHandle(source.clone()),
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
    mark_cloud_visible(&mut world, camera, cloud);

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[cloud]);
    let (atlas, ticket) = {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        (
            state.atlas.clone(),
            state.transient_atlas.as_ref().unwrap().ticket().clone(),
        )
    };
    assert!(ticket.acknowledge(ticket.generation()));
    let waiting = (0..256)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            world.get::<LodRenderCandidates>(cloud).cloned()
        })
        .expect("the cold coarse target should publish a complete transaction");
    assert!(!waiting.candidate_draw_required);
    assert!(
        world.resource::<GaussianLodBridgeManager>().clouds[&cloud]
            .current
            .is_none()
    );
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source.id()
    );
    for candidate in waiting.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    (world, schedule, cloud, camera, source, atlas, waiting)
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_cold_source_bound(world: &World, cloud: Entity, source: &Handle<PlanarGaussian3d>) {
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source.id(),
        "a revoked cold transaction must keep the complete source bound"
    );
    assert!(
        world
            .get::<LodRenderCandidates>(cloud)
            .is_none_or(|candidates| !candidates.candidate_draw_required),
        "the sparse atlas cannot become draw-required before a new PREPARED transaction"
    );
    let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
    assert!(state.current.is_none());
    assert!(state.flat_source_bypass);
}

#[cfg(all(not(target_arch = "wasm32"), lod_render_path))]
#[test]
fn cold_raster_pipeline_pending_retains_source_until_candidate_variant_is_ready() {
    let (mut world, mut schedule, cloud, _camera, source, atlas, candidates) =
        cold_prepared_bridge_fixture();

    for candidate in candidates.by_camera.values() {
        candidate.phase.store(
            cold_staging_candidate_phase(true, true, false, true, true),
            Ordering::Release,
        );
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_cold_source_bound(&world, cloud, &source);
    let waiting = world
        .get::<LodRenderCandidates>(cloud)
        .expect("the raster-pending transaction remains published for staging");
    assert!(
        waiting
            .by_camera
            .values()
            .all(|candidate| { candidate.phase.load(Ordering::Acquire) == LOD_RENDER_WAITING })
    );

    for candidate in waiting.by_camera.values() {
        candidate.phase.store(
            cold_staging_candidate_phase(true, true, true, false, true),
            Ordering::Release,
        );
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_cold_source_bound(&world, cloud, &source);
    let waiting = world
        .get::<LodRenderCandidates>(cloud)
        .expect("the debug-binding-pending transaction remains staged");

    for candidate in waiting.by_camera.values() {
        candidate.phase.store(
            cold_staging_candidate_phase(true, true, true, true, true),
            Ordering::Release,
        );
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        atlas.id(),
        "only the compiled LOD_CANDIDATE raster permutation may lower the source bypass"
    );
    assert!(
        world
            .get::<LodRenderCandidates>(cloud)
            .unwrap()
            .candidate_draw_required
    );
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
fn wasm_transient_build_contract_rejects_event_loop_blocking_sources() {
    let mut config = test_config();
    config.max_ephemeral_source_gaussians = 4096;
    assert!(wasm_synchronous_ephemeral_source_is_supported(
        1_024, &config
    ));
    assert!(!wasm_synchronous_ephemeral_source_is_supported(
        1_025, &config
    ));

    config.max_ephemeral_source_gaussians = 512;
    assert!(wasm_synchronous_ephemeral_source_is_supported(512, &config));
    assert!(!wasm_synchronous_ephemeral_source_is_supported(
        513, &config
    ));
}

#[test]
fn preflight_allows_virtual_source_larger_than_physical_page_cache() {
    let mut config = test_config();
    config.max_ephemeral_source_gaussians = 8_192;
    config.max_ephemeral_stored_gaussians = 16_384;
    config.max_atlas_gaussians = 16_384;
    config.max_atlas_bytes = 64 * 1024 * 1024;
    config.build_settings.leaf_capacity = 1;
    let source = PlanarGaussian3d::from(vec![Gaussian3d::default(); 4_097]);
    let mut settings = GaussianLodSettings::default();
    settings.budgets.max_resident_pages = 4_096;

    assert_eq!(
        preflight_ephemeral_source(&source, &settings, &config),
        Ok(4_097)
    );
}

#[test]
fn preflight_rejects_source_larger_than_stored_budget() {
    let mut config = test_config();
    config.max_ephemeral_stored_gaussians = 3;
    let source = PlanarGaussian3d::from(vec![Gaussian3d::default(); 4]);
    assert_eq!(
        preflight_ephemeral_source(&source, &GaussianLodSettings::default(), &config),
        Err(LodBridgeError::StoredGaussianLimit {
            actual: 4,
            limit: 3,
        })
    );
}

#[test]
fn preflight_allows_partial_final_page_for_small_source() {
    let mut config = test_config();
    config.max_ephemeral_source_gaussians = 4;
    config.max_ephemeral_stored_gaussians = 8;
    config.max_atlas_gaussians = 4;
    config.max_atlas_bytes = 4 * gaussian_3d_gpu_bytes_per_record();
    config.build_settings.leaf_capacity = 16;
    let source = PlanarGaussian3d::from(vec![Gaussian3d::default(); 4]);
    let mut settings = GaussianLodSettings::default();
    settings.budgets.max_resident_pages = 1;

    assert_eq!(
        preflight_ephemeral_source(&source, &settings, &config),
        Ok(4)
    );
}

#[test]
fn transient_sort_capacity_covers_padding_after_perfect_square_source() {
    assert_eq!(sorted_entry_capacity_for_count(16), 16);
    assert_eq!(sorted_entry_capacity_for_count(20), 25);
    assert!(sorted_entry_capacity_for_count(20) > sorted_entry_capacity_for_count(16));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn corrected_live_setting_retries_after_validation_failure() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();
    world.init_resource::<LodTransientAtlasRegistry>();
    let source = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(LodTestScene::nested_octants(1).cloud());
    let entity = world
        .spawn((
            PlanarGaussian3dHandle(source),
            GaussianLodSettings {
                quality: 0.5,
                hysteresis: 2.0,
                ..Default::default()
            },
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
    mark_cloud_visible(&mut world, camera, entity);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);

    run_bridge_frame(&mut world, &mut schedule);
    assert!(
        world
            .get::<GaussianLodBridgeStatus>(entity)
            .unwrap()
            .failure
            .is_some()
    );
    assert!(
        world
            .resource::<GaussianLodBridgeManager>()
            .blocked
            .is_empty()
    );

    world
        .get_mut::<GaussianLodSettings>(entity)
        .unwrap()
        .hysteresis = 0.1;
    run_bridge_frame(&mut world, &mut schedule);
    let manager = world.resource::<GaussianLodBridgeManager>();
    assert!(manager.pending.contains_key(&entity) || manager.clouds.contains_key(&entity));
    assert!(manager.blocked.is_empty());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn transient_snapshot_is_frame_bounded_survives_hidden_and_cancels_at_original() {
    const SOURCE_LEN: usize = EPHEMERAL_SNAPSHOT_RECORDS_PER_UPDATE * 2 + 17;
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();
    let mut config = test_config();
    config.max_ephemeral_source_gaussians = SOURCE_LEN as u32;
    config.max_ephemeral_stored_gaussians = (SOURCE_LEN as u64) * 2;
    config.max_atlas_gaussians = (SOURCE_LEN as u32) * 2;
    config.max_atlas_bytes = 64 * 1024 * 1024;
    let physical_page_capacity = ephemeral_physical_page_capacity(config.build_settings);
    world.insert_resource(config);

    let source = PlanarGaussian3d::from(vec![Gaussian3d::default(); SOURCE_LEN]);
    let source_handle = world.resource_mut::<Assets<PlanarGaussian3d>>().add(source);
    let mut settings = GaussianLodSettings {
        quality: 0.5,
        ..Default::default()
    };
    settings.budgets.max_resident_pages = (SOURCE_LEN as u32).div_ceil(physical_page_capacity);
    settings.budgets.max_gpu_upload_bytes_per_commit = 64 * 1024 * 1024;
    let first = world
        .spawn((
            PlanarGaussian3dHandle(source_handle.clone()),
            settings.clone(),
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();
    let second = world
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

    for entity in [first, second] {
        assert_eq!(
            world
                .get::<PlanarGaussian3dHandle>(entity)
                .unwrap()
                .handle()
                .id(),
            source_handle.id(),
            "snapshotting must keep the exact resident source drawable"
        );
        assert_eq!(
            world.get::<GaussianLodBridgeStatus>(entity).unwrap().phase,
            GaussianLodBridgePhase::Building
        );
    }
    let (pending_entity, request_id) = {
        let manager = world.resource::<GaussianLodBridgeManager>();
        assert_eq!(
            manager.pending.len(),
            1,
            "only one giant source is admitted"
        );
        let (&pending_entity, pending) = manager.pending.iter().next().unwrap();
        let PendingEphemeralBridgePhase::Snapshot(snapshot) = &pending.phase else {
            panic!("the first frame must not launch an incomplete snapshot")
        };
        assert_eq!(snapshot.next_index, EPHEMERAL_SNAPSHOT_RECORDS_PER_UPDATE);
        assert_eq!(snapshot.records.len(), snapshot.next_index);
        (pending_entity, pending.request.id)
    };

    world
        .entity_mut(pending_entity)
        .insert(ViewVisibility::HIDDEN);
    schedule.run(&mut world);
    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let pending = manager
            .pending
            .get(&pending_entity)
            .expect("offscreen snapshot initialization must not be canceled or restarted");
        assert_eq!(pending.request.id, request_id);
        let PendingEphemeralBridgePhase::Snapshot(snapshot) = &pending.phase else {
            panic!("the bounded snapshot should still need its final source chunk")
        };
        assert_eq!(
            snapshot.next_index,
            EPHEMERAL_SNAPSHOT_RECORDS_PER_UPDATE * 2
        );
        assert_eq!(snapshot.records.len(), snapshot.next_index);
        assert_eq!(
            world
                .get::<PlanarGaussian3dHandle>(pending_entity)
                .unwrap()
                .handle()
                .id(),
            source_handle.id()
        );
    }

    world.get_mut::<GaussianLodSettings>(first).unwrap().quality = 1.0;
    world
        .get_mut::<GaussianLodSettings>(second)
        .unwrap()
        .quality = 1.0;
    schedule.run(&mut world);
    assert!(
        world
            .resource::<GaussianLodBridgeManager>()
            .pending
            .is_empty()
    );
    assert!(world.get::<GaussianLodBridgeStatus>(first).is_none());
    assert!(world.get::<GaussianLodBridgeStatus>(second).is_none());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn transient_completion_preserves_debug_metadata_changed_while_building() {
    use crate::gaussian::lod_debug::LodDebugRecord;

    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();

    let source = LodTestScene::nested_octants(2).cloud();
    let source_handle = world.resource_mut::<Assets<PlanarGaussian3d>>().add(source);
    let initial = LodDebugMetadata::new(vec![LodDebugRecord::default()]);
    let current = LodDebugMetadata::new(vec![LodDebugRecord::default(); 2]);
    let entity = world
        .spawn((
            PlanarGaussian3dHandle(source_handle),
            GaussianLodSettings {
                quality: 0.5,
                ..Default::default()
            },
            initial,
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();
    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);

    run_bridge_frame(&mut world, &mut schedule);
    assert!(
        world
            .resource::<GaussianLodBridgeManager>()
            .pending
            .contains_key(&entity)
    );
    world.entity_mut(entity).insert(current);
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[entity]);

    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&entity];
        assert_eq!(state.source_debug_metadata.as_ref().unwrap().len(), 2);
        assert_eq!(state.fallback_debug_metadata.len(), 2);
    }

    world
        .get_mut::<GaussianLodSettings>(entity)
        .unwrap()
        .quality = 1.0;
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(world.get::<LodDebugMetadata>(entity).unwrap().len(), 2);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn transient_cold_generation_bump_reuploads_before_active_hide_show() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();
    world.init_resource::<LodTransientAtlasRegistry>();

    let source_handle = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(LodTestScene::screen_space_ladder().cloud());
    let mut settings = GaussianLodSettings {
        quality: 0.5,
        ..Default::default()
    };
    settings.budgets.max_requests_per_frame = 1;
    let entity = world
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
    mark_cloud_visible(&mut world, camera, entity);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[entity]);

    let (atlas, ticket) = {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&entity];
        (
            state.atlas.clone(),
            state.transient_atlas.as_ref().unwrap().ticket().clone(),
        )
    };
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&atlas)
            .is_none(),
        "the bounded transient atlas must never enter generic Assets"
    );
    assert!(!ticket.is_ready());

    assert!(ticket.acknowledge(ticket.generation()));
    let mut waiting = None;
    for _ in 0..128 {
        run_bridge_frame(&mut world, &mut schedule);
        if let Some(candidates) = world.get::<LodRenderCandidates>(entity).cloned() {
            waiting = Some(candidates);
            break;
        }
        assert_eq!(
            world
                .get::<PlanarGaussian3dHandle>(entity)
                .unwrap()
                .handle()
                .id(),
            source_handle.id(),
            "progressive resident ancestor waves must not replace the cold source"
        );
        assert!(
            world.resource::<GaussianLodBridgeManager>().clouds[&entity]
                .current
                .is_none()
        );
    }
    let waiting = waiting.expect("the target cut should publish once streaming becomes quiescent");
    {
        let state = &world.resource::<GaussianLodBridgeManager>().clouds[&entity];
        assert!(state.deferred_ordinary_publications > 0);
        assert!(
            waiting
                .by_camera
                .values()
                .all(|candidate| !candidate.frontier().is_coverage_guard()),
            "normal cold presentation must not substitute the roots-only emergency guard"
        );
    }
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        source_handle.id(),
        "WAITING cannot expose a partially prepared atlas"
    );
    assert!(!waiting.candidate_draw_required);
    assert!(
        world.resource::<GaussianLodBridgeManager>().clouds[&entity]
            .current
            .is_none()
    );

    world
        .resource_mut::<LodAtlasUploadQueue>()
        .remove_atlas(atlas.id());
    let replacement_generation = ticket.request_reupload_for_test();
    assert!(ticket.acknowledge(replacement_generation));
    let restaged = (0..64)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            world.get::<LodRenderCandidates>(entity).cloned()
        })
        .expect("the cold guard should be restaged after atlas recreation");
    assert_eq!(
        world.resource::<GaussianLodBridgeManager>().clouds[&entity].transient_atlas_generation,
        Some(replacement_generation)
    );
    assert!(
        restaged
            .by_camera
            .values()
            .all(|candidate| candidate.phase.load(Ordering::Acquire) == LOD_RENDER_WAITING)
    );
    let replacement_uploads = world
        .resource::<LodAtlasUploadQueue>()
        .queued_slots()
        .collect::<Vec<_>>();
    for candidate in restaged.by_camera.values() {
        for range in candidate.render_ranges() {
            assert!(
                replacement_uploads
                    .iter()
                    .any(|upload| { upload.atlas == atlas.id() && upload.slot == range.slot })
            );
        }
    }
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        source_handle.id(),
        "generation recovery must remain atomic while the new atlas is WAITING"
    );

    for candidate in restaged.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        atlas.id(),
        "PREPARED atomically hands rendering to the bounded atlas"
    );
    assert!(
        world
            .get::<LodRenderCandidates>(entity)
            .unwrap()
            .candidate_draw_required
    );

    // PREPARED has lowered the cold source bypass, but there is still no ACTIVE
    // cut. Recreating the allocation in this exact window must atomically
    // restore the source and restage the same bounded payload.
    world
        .resource_mut::<LodAtlasUploadQueue>()
        .remove_atlas(atlas.id());
    let prepared_replacement_generation = ticket.request_reupload_for_test();
    assert!(ticket.acknowledge(prepared_replacement_generation));
    let recovered = (0..64)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            world.get::<LodRenderCandidates>(entity).cloned()
        })
        .expect("a cold PREPARED cut should be restaged after atlas recreation");
    assert_eq!(
        world.resource::<GaussianLodBridgeManager>().clouds[&entity].transient_atlas_generation,
        Some(prepared_replacement_generation)
    );
    assert!(
        recovered
            .by_camera
            .values()
            .all(|candidate| candidate.phase.load(Ordering::Acquire) == LOD_RENDER_WAITING)
    );
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        source_handle.id(),
        "cold generation loss after PREPARED must restore the complete source"
    );
    let prepared_replacement_uploads = world
        .resource::<LodAtlasUploadQueue>()
        .queued_slots()
        .collect::<Vec<_>>();
    for candidate in recovered.by_camera.values() {
        for range in candidate.render_ranges() {
            assert!(
                prepared_replacement_uploads
                    .iter()
                    .any(|upload| upload.atlas == atlas.id() && upload.slot == range.slot)
            );
        }
    }

    for candidate in recovered.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        atlas.id(),
        "the recreated atlas may replace the source only after PREPARED"
    );

    for candidate in recovered.by_camera.values() {
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    let (active_phase, active_leases) = {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&entity];
        assert!(state.active);
        (
            Arc::clone(&state.current.as_ref().unwrap().get(camera).unwrap().phase),
            state.current_page_leases.clone(),
        )
    };
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(entity).unwrap().phase,
        GaussianLodBridgePhase::Active
    );

    mark_cloud_hidden(&mut world, camera, entity);
    world.entity_mut(entity).insert(ViewVisibility::HIDDEN);
    run_bridge_frame(&mut world, &mut schedule);
    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&entity];
        assert!(state.active);
        assert!(state.views.is_empty());
        assert!(state.handshakes.is_empty());
        assert_eq!(state.current_page_leases, active_leases);
        assert!(Arc::ptr_eq(
            &state.current.as_ref().unwrap().get(camera).unwrap().phase,
            &active_phase
        ));
        assert!(!state.flat_source_bypass);
        assert!(manager.pending.is_empty());
    }
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        atlas.id(),
        "aggregate visibility must not rebind an initialized bridge to its source"
    );

    mark_cloud_visible(&mut world, camera, entity);
    world.entity_mut(entity).insert(ViewVisibility::VISIBLE);
    run_bridge_frame(&mut world, &mut schedule);
    let state = &world.resource::<GaussianLodBridgeManager>().clouds[&entity];
    assert!(state.active);
    assert_eq!(state.current_page_leases, active_leases);
    assert!(Arc::ptr_eq(
        &state.current.as_ref().unwrap().get(camera).unwrap().phase,
        &active_phase
    ));
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        atlas.id()
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn cold_candidate_materializes_one_bounded_step_before_atomic_handoff() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let config = test_config();
    let built = build_planar_3d_lod(&source, config.build_settings).unwrap();
    let gaussians_per_slot = built
        .manifest
        .pages
        .iter()
        .map(|page| page.gaussian_count)
        .max()
        .unwrap();
    let one_slot_bytes = u64::from(gaussians_per_slot) * gaussian_3d_gpu_bytes_per_record();

    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(config);
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();
    world.init_resource::<LodTransientAtlasRegistry>();

    let source_handle = world.resource_mut::<Assets<PlanarGaussian3d>>().add(source);
    let mut settings = GaussianLodSettings {
        // Stay below the exact-source endpoint while requiring a multi-page
        // ordinary cut from this eight-page fixture.
        quality: 0.999,
        ..Default::default()
    };
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_pending_requests = 128;
    settings.budgets.max_upload_bytes_per_frame = one_slot_bytes;
    settings.budgets.max_gpu_upload_bytes_per_commit = one_slot_bytes * 16;
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
    mark_cloud_visible(&mut world, camera, cloud);

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[cloud]);
    let (atlas, ticket) = {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        assert_eq!(state.mirror.layout().gaussians_per_slot, gaussians_per_slot);
        assert_eq!(
            gpu_staging_step_byte_limit(
                &state
                    .structural
                    .apply(world.get::<GaussianLodSettings>(cloud).unwrap())
            ),
            one_slot_bytes
        );
        (
            state.atlas.clone(),
            state.transient_atlas.as_ref().unwrap().ticket().clone(),
        )
    };
    let atlas_generation = ticket.generation();
    assert!(ticket.acknowledge(atlas_generation));

    let mut waiting = None;
    for _ in 0..256 {
        let before = world.resource::<GaussianLodBridgeManager>().clouds[&cloud]
            .mirror
            .materialized_slots()
            .len();
        run_bridge_frame(&mut world, &mut schedule);
        let manager = world.resource::<GaussianLodBridgeManager>();
        assert!(manager.pending.is_empty());
        let state = &manager.clouds[&cloud];
        assert_eq!(state.atlas.id(), atlas.id());
        assert_eq!(state.transient_atlas_generation, Some(atlas_generation));
        assert!(state.current.is_none());
        assert!(state.current_page_leases.is_empty());
        assert!(state.flat_source_bypass);
        let after = state.mirror.materialized_slots().len();
        assert!(after >= before);
        assert!(
            (after - before) as u64 * one_slot_bytes <= one_slot_bytes,
            "one main-world update must not materialize more than the effective staging-step cap"
        );
        if let Some(candidates) = world.get::<LodRenderCandidates>(cloud).cloned() {
            waiting = Some(candidates);
            break;
        }
        assert!(state.pending_page_leases.is_empty());
        assert_eq!(
            world
                .get::<PlanarGaussian3dHandle>(cloud)
                .unwrap()
                .handle()
                .id(),
            source_handle.id()
        );
    }
    let waiting = waiting.expect("the drained ordinary cut should become publishable");
    let waiting_candidate = waiting.get(camera).unwrap();
    assert!(!waiting_candidate.frontier().is_coverage_guard());
    assert_eq!(
        waiting_candidate.phase.load(Ordering::Acquire),
        LOD_RENDER_WAITING
    );
    assert_eq!(waiting.staging_atlas, Some(atlas.id()));
    assert!(!waiting.candidate_draw_required);
    let candidate_pages = bridge_candidate_pages(&waiting);
    let candidate_slots = waiting
        .by_camera
        .values()
        .flat_map(LodRenderCandidate::render_ranges)
        .map(|range| range.slot)
        .collect::<BTreeSet<_>>();
    assert!(
        candidate_slots.len() > 1,
        "the regression requires a complete cut larger than one staging step"
    );
    assert!(candidate_pages.len() > 1);
    let phase = Arc::clone(&waiting_candidate.phase);
    let expected_ranges = waiting_candidate.render_ranges().to_vec();
    let (mut materialized, pre_frame_lease_acquisitions) = {
        let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
        let materialized = state
            .mirror
            .materialized_slots()
            .into_iter()
            .filter(|slot| candidate_slots.contains(slot))
            .collect::<BTreeSet<_>>();
        assert_eq!(materialized.len(), 1);
        assert_eq!(state.pending_page_leases, candidate_pages);
        assert!(!state.handshakes[&camera].staged);
        (materialized, state.pre_frame_pending_lease_acquisitions)
    };

    for _ in 0..candidate_slots.len() + 2 {
        if world.resource::<GaussianLodBridgeManager>().clouds[&cloud].handshakes[&camera].staged {
            break;
        }
        let before = materialized.len();
        run_bridge_frame(&mut world, &mut schedule);
        let published = world.get::<LodRenderCandidates>(cloud).unwrap();
        let published_candidate = published.get(camera).unwrap();
        assert!(Arc::ptr_eq(&published_candidate.phase, &phase));
        assert_eq!(
            published_candidate.render_ranges(),
            expected_ranges.as_slice()
        );
        let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
        assert_eq!(state.atlas.id(), atlas.id());
        assert_eq!(state.transient_atlas_generation, Some(atlas_generation));
        assert_eq!(state.pending_page_leases, candidate_pages);
        assert_eq!(
            state.pre_frame_pending_lease_acquisitions, pre_frame_lease_acquisitions,
            "the complete candidate union must be leased once, not once per staging step"
        );
        materialized = state
            .mirror
            .materialized_slots()
            .into_iter()
            .filter(|slot| candidate_slots.contains(slot))
            .collect();
        let newly_materialized = materialized.len() - before;
        assert!(newly_materialized > 0);
        assert!(newly_materialized as u64 * one_slot_bytes <= one_slot_bytes);
        assert!(state.current.is_none());
        assert!(state.current_page_leases.is_empty());
        assert!(state.flat_source_bypass);
        assert_eq!(
            world
                .get::<PlanarGaussian3dHandle>(cloud)
                .unwrap()
                .handle()
                .id(),
            source_handle.id()
        );
    }

    {
        let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
        assert!(state.handshakes[&camera].staged);
        assert_eq!(materialized, candidate_slots);
        assert_eq!(state.pending_page_leases, candidate_pages);
        assert!(state.current.is_none());
    }
    let queued_candidate_slots = world
        .resource::<LodAtlasUploadQueue>()
        .queued_slots()
        .filter(|upload| upload.atlas == atlas.id() && candidate_slots.contains(&upload.slot))
        .map(|upload| upload.slot)
        .collect::<BTreeSet<_>>();
    assert_eq!(queued_candidate_slots, candidate_slots);

    let decoded_before_idle =
        world.resource::<GaussianLodBridgeManager>().clouds[&cloud].decoded_page_acquisitions;
    run_bridge_frame(&mut world, &mut schedule);
    {
        let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
        assert_eq!(state.decoded_page_acquisitions, decoded_before_idle);
        assert_eq!(state.pending_page_leases, candidate_pages);
        assert!(Arc::ptr_eq(
            &world
                .get::<LodRenderCandidates>(cloud)
                .unwrap()
                .get(camera)
                .unwrap()
                .phase,
            &phase
        ));
    }

    phase.store(LOD_RENDER_PREPARED, Ordering::Release);
    run_bridge_frame(&mut world, &mut schedule);
    {
        let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
        assert!(state.current.is_none());
        assert_eq!(state.pending_page_leases, candidate_pages);
        assert!(!state.flat_source_bypass);
        assert_eq!(state.atlas.id(), atlas.id());
    }
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        atlas.id()
    );
    let prepared = world.get::<LodRenderCandidates>(cloud).unwrap();
    assert!(prepared.candidate_draw_required);
    assert!(Arc::ptr_eq(&prepared.get(camera).unwrap().phase, &phase));
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::WaitingForRender
    );

    phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    run_bridge_frame(&mut world, &mut schedule);
    let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
    assert!(state.active);
    assert!(!state.flat_source_bypass);
    assert!(state.pending_page_leases.is_empty());
    assert_eq!(state.current_page_leases, candidate_pages);
    assert!(Arc::ptr_eq(
        &state.current.as_ref().unwrap().get(camera).unwrap().phase,
        &phase
    ));
    assert_eq!(state.atlas.id(), atlas.id());
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn cold_prepared_motion_invalidation_restores_source_until_reprepared() {
    let (mut world, mut schedule, cloud, camera, source, atlas, prepared) =
        cold_prepared_bridge_fixture();

    // Change both the effective view and its requested detail before the render
    // world's PREPARED token can become ACTIVE. The old transaction must be
    // revoked without exposing its sparse atlas.
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 0.75;
    *world.get_mut::<GlobalTransform>(camera).unwrap() =
        GlobalTransform::from(Transform::from_xyz(8.0, 0.0, 1.0));
    run_bridge_frame(&mut world, &mut schedule);
    assert!(
        prepared
            .by_camera
            .values()
            .all(|candidate| candidate.phase.load(Ordering::Acquire) == LOD_RENDER_WAITING)
    );
    assert_cold_source_bound(&world, cloud, &source);

    let replacement = (0..256)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            assert_cold_source_bound(&world, cloud, &source);
            world
                .get::<LodRenderCandidates>(cloud)
                .filter(|candidates| !candidates.is_empty())
                .cloned()
        })
        .expect("the moved view should eventually reach a new quiescent transaction");
    assert!(!replacement.candidate_draw_required);
    for candidate in replacement.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        atlas.id(),
        "only the replacement PREPARED token may lower the source bypass"
    );
    assert!(
        world
            .get::<LodRenderCandidates>(cloud)
            .unwrap()
            .candidate_draw_required
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn cold_prepared_camera_membership_change_restores_source_until_reprepared() {
    let (mut world, mut schedule, cloud, _camera_a, source, atlas, prepared) =
        cold_prepared_bridge_fixture();
    let camera_b = world
        .spawn((
            Camera {
                viewport: Some(bevy::camera::Viewport {
                    physical_size: UVec2::new(1280, 720),
                    ..default()
                }),
                ..default()
            },
            Projection::Perspective(default()),
            GlobalTransform::from(Transform::from_xyz(8.0, 0.0, 5.0)),
            GaussianCamera::default(),
        ))
        .id();
    mark_cloud_visible(&mut world, camera_b, cloud);

    run_bridge_frame(&mut world, &mut schedule);
    assert!(
        prepared
            .by_camera
            .values()
            .all(|candidate| candidate.phase.load(Ordering::Acquire) == LOD_RENDER_WAITING)
    );
    assert_cold_source_bound(&world, cloud, &source);

    let replacement = (0..256)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            assert_cold_source_bound(&world, cloud, &source);
            world
                .get::<LodRenderCandidates>(cloud)
                .filter(|candidates| candidates.len() == 2 && candidates.get(camera_b).is_some())
                .cloned()
        })
        .expect("both cameras should eventually join a new cold transaction");
    assert!(!replacement.candidate_draw_required);
    for candidate in replacement.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        atlas.id(),
        "the all-camera PREPARED replacement may atomically bind the atlas"
    );
    assert!(
        world
            .get::<LodRenderCandidates>(cloud)
            .unwrap()
            .candidate_draw_required
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn partial_multi_root_guard_keeps_source_bound_until_complete_handoff() {
    let gaussian = |x| Gaussian3d {
        position_visibility: [x, 0.0, 0.0, 1.0].into(),
        rotation: [1.0, 0.0, 0.0, 0.0].into(),
        scale_opacity: [0.1, 0.1, 0.1, 1.0].into(),
        ..default()
    };
    let source = PlanarGaussian3d::from(vec![gaussian(-1.0), gaussian(1.0)]);
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.frustum_culling = false;
    settings.budgets.max_active_gaussians = 2;
    settings.budgets.max_resident_gaussians = 2;
    settings.budgets.max_resident_pages = 2;
    settings.budgets.max_pending_requests = 2;
    settings.budgets.max_requests_per_frame = 1;
    let streaming = GaussianStreamingSettings {
        max_concurrent_requests: 1,
        ..default()
    };
    let mut config = test_config();
    config.max_ephemeral_source_gaussians = 2;
    config.max_ephemeral_stored_gaussians = 4;
    config.max_atlas_gaussians = 2;
    config.build_settings = GaussianLodBuildSettings {
        branching_factor: 2,
        leaf_capacity: 1,
        support_sigma: 3.0,
    };
    config.streaming_settings = streaming.clone();

    // Convert the ordinary one-root fixture into a two-root forest backed by
    // two independent pages. With one request admitted per frame, the bridge
    // must observe a genuinely partial global guard before both roots arrive.
    let mut built = build_planar_3d_lod(&source, config.build_settings).unwrap();
    let original_root = built.manifest.roots[0];
    let original_root_page = built
        .manifest
        .nodes
        .iter()
        .find(|node| node.id == original_root)
        .unwrap()
        .representation
        .page;
    built.manifest.nodes.retain(|node| node.id != original_root);
    built
        .manifest
        .pages
        .retain(|page| page.id != original_root_page);
    built.pages.retain(|page| page.id != original_root_page);
    for node in &mut built.manifest.nodes {
        node.parent = None;
        node.depth = 0;
        node.quality.min = 0.0;
    }
    built.manifest.roots = built.manifest.nodes.iter().map(|node| node.id).collect();
    built.manifest.header.node_count = built.manifest.nodes.len() as u32;
    built.manifest.header.page_count = built.manifest.pages.len() as u32;
    built.manifest.header.stored_gaussian_count = built
        .manifest
        .pages
        .iter()
        .map(|page| u64::from(page.gaussian_count))
        .sum();
    built.manifest.quality = GaussianLodQualityMetadata {
        max_depth: 0,
        coarsest_gaussian_count: 2,
        finest_gaussian_count: 2,
        max_error: built
            .manifest
            .nodes
            .iter()
            .fold(LodError::ZERO, |error, node| error.max(node.error)),
    };
    assert_eq!(built.manifest.roots.len(), 2);
    assert_eq!(
        built
            .manifest
            .nodes
            .iter()
            .map(|node| node.representation.page)
            .collect::<BTreeSet<_>>()
            .len(),
        2,
        "the fixture requires independently streamable guard pages"
    );
    let partial_guard_page = built.pages[0].clone();
    let mut transport = MemoryPageTransport::default();
    for page in &built.pages {
        let encoded = encode_page(page).unwrap();
        let descriptor = built
            .manifest
            .pages
            .iter_mut()
            .find(|descriptor| descriptor.id == page.id)
            .unwrap();
        descriptor.storage = Some(LodPageStorage {
            uri: format!("memory://bridge-two-root-{}", page.id.0),
            byte_range: None,
            encoded_len: encoded.len() as u64,
        });
        transport.insert(page.id, encoded);
    }
    built.validate().unwrap();

    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(config.clone());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();
    world.init_resource::<LodTransientAtlasRegistry>();
    let source_handle = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(source.clone());
    let cloud = world
        .spawn((
            PlanarGaussian3dHandle(source_handle.clone()),
            settings.clone(),
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
    mark_cloud_visible(&mut world, camera, cloud);

    let (mut state, mut atlas_cloud) = create_ephemeral_bridge(
        source_handle.clone(),
        &source,
        None,
        &settings,
        &streaming,
        &config,
        false,
    )
    .unwrap();
    let effective = state.structural.apply(&settings);
    let runtime =
        LodStreamingRuntime::new(built.manifest, transport, &effective, &streaming).unwrap();
    assert_eq!(runtime.atlas_layout(), state.mirror.layout());
    state.runtime = Box::new(runtime);
    let partial_slot = AtlasSlot {
        index: 0,
        generation: 1,
    };
    state
        .mirror
        .stage_page(partial_guard_page.id, partial_slot)
        .unwrap();
    state
        .mirror
        .materialize_page(&mut atlas_cloud, &partial_guard_page, partial_slot)
        .unwrap();
    assert_eq!(state.mirror.materialized_slots(), vec![partial_slot]);
    let atlas = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .reserve_handle();
    let transient = LodTransientAtlas::new(atlas_cloud);
    let generation = transient.ticket().generation();
    assert!(transient.ticket().acknowledge(generation));
    assert!(transient.ticket().is_ready());
    state.atlas = atlas.clone();
    state.transient_atlas_generation = Some(generation);
    state.transient_atlas = Some(transient);
    world
        .resource_mut::<GaussianLodBridgeManager>()
        .clouds
        .insert(cloud, state);

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    run_bridge_frame(&mut world, &mut schedule);
    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        assert!(state.current.is_none());
        assert!(state.flat_source_bypass);
        assert_eq!(state.mirror.materialized_slots(), vec![partial_slot]);
    }
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id(),
        "a partial guard cannot replace the globally complete source"
    );
    assert!(
        world.get::<LodRenderCandidates>(cloud).is_none(),
        "partial guard residency must not emit a candidate-required component"
    );

    let waiting = (0..64)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            world
                .get::<LodRenderCandidates>(cloud)
                .filter(|candidates| {
                    candidates
                        .get(camera)
                        .is_some_and(|candidate| candidate.rendered_candidate_count() == 2)
                })
                .cloned()
        })
        .expect("both guard roots should become resident");
    assert!(!waiting.candidate_draw_required);
    assert!(world.resource::<GaussianLodBridgeManager>().clouds[&cloud].flat_source_bypass);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id(),
        "a complete but WAITING guard cannot replace the source"
    );
    for candidate in waiting.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        atlas.id()
    );
    let prepared = world.get::<LodRenderCandidates>(cloud).unwrap().clone();
    assert!(prepared.candidate_draw_required);
    assert!(!world.resource::<GaussianLodBridgeManager>().clouds[&cloud].flat_source_bypass);
    assert_eq!(
        world.resource::<GaussianLodBridgeManager>().clouds[&cloud]
            .mirror
            .materialized_slots()
            .len(),
        2
    );
    for candidate in prepared.by_camera.values() {
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn transient_atlas_reupload_retains_and_restages_current_cut() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();
    world.init_resource::<LodTransientAtlasRegistry>();

    let source_handle = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(LodTestScene::screen_space_ladder().cloud());
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
    mark_cloud_visible(&mut world, camera, cloud);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[cloud]);

    let (atlas, ticket) = {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        (
            state.atlas.clone(),
            state.transient_atlas.as_ref().unwrap().ticket().clone(),
        )
    };
    assert!(ticket.acknowledge(ticket.generation()));

    let waiting = (0..128)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            world.get::<LodRenderCandidates>(cloud).cloned()
        })
        .expect("the transient bridge should publish a camera cut");
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    assert!(!waiting.candidate_draw_required);
    for candidate in waiting.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    let prepared = world.get::<LodRenderCandidates>(cloud).unwrap().clone();
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        atlas.id()
    );
    assert!(prepared.candidate_draw_required);
    for candidate in prepared.by_camera.values() {
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);

    let current = world.resource::<GaussianLodBridgeManager>().clouds[&cloud]
        .current
        .clone()
        .expect("the initial transient cut should commit");
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );
    *world.resource_mut::<LodAtlasUploadQueue>() = LodAtlasUploadQueue::default();

    let reupload_generation = ticket.request_reupload_for_test();
    run_bridge_frame(&mut world, &mut schedule);

    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        atlas.id(),
        "device recreation must not swap a live bridge back to its source handle"
    );
    let retained = world.get::<LodRenderCandidates>(cloud).unwrap();
    assert!(retained.candidate_draw_required);
    assert!(bridge_candidate_sets_match(&current, retained));
    assert!(
        retained
            .by_camera
            .values()
            .all(|candidate| candidate.phase.load(Ordering::Acquire) == LOD_RENDER_WAITING)
    );
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::WaitingForRender
    );
    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        assert!(!state.active);
        assert!(state.handshakes.values().all(|handshake| !handshake.staged));
        assert!(
            state
                .current
                .as_ref()
                .is_some_and(|candidate| { bridge_candidate_sets_match(candidate, &current) })
        );
    }

    assert!(ticket.acknowledge(reupload_generation));
    for candidate in retained.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    let uploads = world
        .resource::<LodAtlasUploadQueue>()
        .queued_slots()
        .collect::<Vec<_>>();
    for candidate in current.by_camera.values() {
        for range in candidate.render_ranges() {
            assert!(
                uploads
                    .iter()
                    .any(|upload| upload.atlas == atlas.id() && upload.slot == range.slot)
            );
        }
    }

    let restaged = world.get::<LodRenderCandidates>(cloud).unwrap().clone();
    for candidate in restaged.by_camera.values() {
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );
    assert!(world.resource::<GaussianLodBridgeManager>().clouds[&cloud].active);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn transient_gpu_failure_stays_blocked_until_source_or_structure_changes() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();
    world.init_resource::<LodTransientAtlasRegistry>();

    let source_handle = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(LodTestScene::nested_octants(1).cloud());
    let entity = world
        .spawn((
            PlanarGaussian3dHandle(source_handle.clone()),
            GaussianLodSettings {
                quality: 0.5,
                ..Default::default()
            },
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();
    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[entity]);

    let (ticket, first_signature) = {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&entity];
        (
            state.transient_atlas.as_ref().unwrap().ticket().clone(),
            state.signature,
        )
    };
    ticket.fail_current_for_test();
    run_bridge_frame(&mut world, &mut schedule);

    let first_request_id = {
        let manager = world.resource::<GaussianLodBridgeManager>();
        assert!(manager.clouds.is_empty());
        assert!(manager.pending.is_empty());
        assert_eq!(
            manager.blocked.get(&entity),
            Some(&(source_handle.id(), first_signature))
        );
        manager.next_request_id
    };
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );

    world
        .get_mut::<GaussianLodSettings>(entity)
        .unwrap()
        .hysteresis = 0.2;
    run_bridge_frame(&mut world, &mut schedule);
    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        assert!(manager.clouds.is_empty());
        assert!(manager.pending.is_empty());
        assert_eq!(manager.next_request_id, first_request_id);
        assert_eq!(
            manager.blocked.get(&entity),
            Some(&(source_handle.id(), first_signature)),
            "a live-only settings change must not restart a failed giant build"
        );
    }

    world
        .get_mut::<GaussianLodSettings>(entity)
        .unwrap()
        .budgets
        .max_resident_pages -= 1;
    run_bridge_frame(&mut world, &mut schedule);
    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        assert!(manager.pending.contains_key(&entity) || manager.clouds.contains_key(&entity));
        assert!(manager.blocked.is_empty());
        assert!(manager.next_request_id > first_request_id);
    }
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[entity]);

    let second_ticket = {
        let manager = world.resource::<GaussianLodBridgeManager>();
        manager.clouds[&entity]
            .transient_atlas
            .as_ref()
            .unwrap()
            .ticket()
            .clone()
    };
    second_ticket.fail_current_for_test();
    run_bridge_frame(&mut world, &mut schedule);
    let second_request_id = world.resource::<GaussianLodBridgeManager>().next_request_id;

    let replacement = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(LodTestScene::nested_octants(2).cloud());
    *world.get_mut::<PlanarGaussian3dHandle>(entity).unwrap() = PlanarGaussian3dHandle(replacement);
    run_bridge_frame(&mut world, &mut schedule);
    let manager = world.resource::<GaussianLodBridgeManager>();
    assert!(manager.pending.contains_key(&entity) || manager.clouds.contains_key(&entity));
    assert!(manager.blocked.is_empty());
    assert!(manager.next_request_id > second_request_id);
}

#[cfg(all(not(target_arch = "wasm32"), feature = "sort_rayon"))]
#[test]
fn transient_worker_pool_reserves_cpu_for_rendering() {
    assert_eq!(transient_lod_worker_threads(1), 1);
    assert_eq!(transient_lod_worker_threads(4), 1);
    assert_eq!(transient_lod_worker_threads(16), 4);
    assert_eq!(transient_lod_worker_threads(64), 4);
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
    assert_eq!(
        bridge_status_transition_kind(GaussianLodBridgePhase::Building, false, true),
        None,
        "a background retry has not recovered until its cut is active"
    );
    assert_eq!(
        bridge_status_transition_kind(GaussianLodBridgePhase::WaitingForRender, false, true),
        None
    );
}

#[test]
fn transient_atlas_is_bounded_empty_page_cache() {
    let source = LodTestScene::nested_octants(2).cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_resident_pages = 32;
    let source_handle = Handle::default();
    let config = test_config();
    let (state, atlas) = create_ephemeral_bridge(
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
    assert!(atlas.iter().all(|record| record == Gaussian3d::default()));
    assert!(state.mirror.slot_count() <= settings.budgets.max_resident_pages);
    assert!(state.flat_source_bypass, "the source is cold-start only");
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
    let complete_manifest_union = built.manifest.header.page_count;

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

    assert_eq!(state.mirror.slot_count(), complete_manifest_union);
    assert_eq!(atlas.len(), (complete_manifest_union * stride) as usize);
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
fn bounded_atlas_allocation_is_not_an_atomic_page_commit() {
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
    let bytes_per_slot = u64::from(stride) * gaussian_3d_gpu_bytes_per_record();
    let required = bytes_per_slot;

    settings.budgets.max_gpu_upload_bytes_per_commit = required - 1;
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
    assert!(physical_gpu_bytes > settings.budgets.max_gpu_upload_bytes_per_commit);
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
    let coarse_phase = state.handshake_for(camera, &coarse[0], views[0].1);
    coarse_phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    let exact_phase = state.handshake_for(camera, &exact[0], views[0].1);
    assert!(!Arc::ptr_eq(&coarse_phase, &exact_phase));
    assert_eq!(coarse_phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    assert_eq!(exact_phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    assert!(Arc::ptr_eq(
        &exact_phase,
        &state.handshake_for(camera, &exact[0], views[0].1)
    ));
    exact_phase.store(LOD_RENDER_FAILED, Ordering::Release);
    assert!(Arc::ptr_eq(
        &exact_phase,
        &state.handshake_for(camera, &exact[0], views[0].1)
    ));
    assert_eq!(
        exact_phase.load(Ordering::Acquire),
        LOD_RENDER_FAILED,
        "a cached failed nonempty cut must not be revived"
    );
    let moved_view = LodView::perspective(Vec3::new(1.0, 0.0, 0.0), 720.0, 1.0, 0.1);
    let moved_phase = state.handshake_for(camera, &exact[0], moved_view);
    assert!(!Arc::ptr_eq(&moved_phase, &exact_phase));
    assert_eq!(exact_phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    assert_eq!(state.handshakes[&camera].selected_view, moved_view);

    settings.frustum_culling = true;
    let effective = state.structural.apply(&settings);
    let empty_view =
        LodView::perspective(Vec3::ZERO, 720.0, 1.0, 0.1).with_clip_from_world(Mat4::IDENTITY);
    let offscreen = (0..64)
        .find_map(|_| {
            let frame = state.runtime.begin_frame();
            let result = state
                .runtime
                .update_view_in_frame(
                    frame,
                    LodRuntimeViewId(3),
                    empty_view,
                    &effective,
                    &state.streaming,
                )
                .unwrap();
            for &page in result.completed_pages() {
                let slot = state.runtime.resident_slot(page).unwrap();
                state.mirror.stage_page(page, slot).unwrap();
            }
            result
                .candidate_frontier(settings.max_active_gaussians_u32())
                .ok()
                .filter(|candidate| {
                    candidate.candidate_count() > 0 && !candidate.physical_ranges().is_empty()
                })
        })
        .expect("offscreen view must retain a complete global root frontier");
    let offscreen_camera = Entity::from_bits(3);
    let offscreen_phase = state.handshake_for(offscreen_camera, &offscreen, empty_view);
    assert_eq!(offscreen_phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    let reused_phase = state.handshake_for(offscreen_camera, &offscreen, empty_view);
    assert!(Arc::ptr_eq(&offscreen_phase, &reused_phase));
    assert_eq!(
        reused_phase.load(Ordering::Acquire),
        LOD_RENDER_WAITING,
        "global offscreen coverage still requires explicit render publication"
    );
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
    let (mut state, mut atlas) = create_ephemeral_bridge(
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

    let bytes_per_slot = gpu_upload_bytes_per_slot(&state).unwrap();
    let too_small = bytes_per_slot - 1;
    let atlas_id = state.atlas.id();
    let mut uploads = LodAtlasUploadQueue::default();
    assert_eq!(
        synchronize_bridge_candidate_pages_bounded(
            &mut state,
            &candidates,
            &BTreeSet::new(),
            atlas_id,
            BridgeAtlasMaterialization::Dense(&mut atlas),
            &mut uploads,
            too_small,
        ),
        Err(LodBridgeError::GpuUploadCommitBudgetExceeded {
            required: bytes_per_slot,
            limit: too_small,
        }),
        "a staging limit smaller than one physical slot must remain a typed fatal configuration error"
    );
    assert!(state.mirror.materialized_slots().is_empty());
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
        BridgeAtlasMaterialization::Dense(&mut atlas),
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
        BridgeAtlasMaterialization::Dense(&mut atlas),
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
        BridgeAtlasMaterialization::Dense(&mut atlas),
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
    run_until_bridge_initialization_finishes(
        &mut world,
        &mut schedule,
        &[default_ephemeral, custom_ephemeral],
    );

    for entity in [default_ephemeral, custom_ephemeral] {
        let actual = world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id();
        if crate::stream::lod_render_path_is_supported() {
            assert_eq!(actual, source_handle.id());
            assert_ne!(
                world.resource::<GaussianLodBridgeManager>().clouds[&entity]
                    .atlas
                    .id(),
                source_handle.id()
            );
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
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[entity]);
    let old_atlas = world.resource::<GaussianLodBridgeManager>().clouds[&entity]
        .atlas
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
    run_bridge_frame(&mut world, &mut schedule);

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

    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[entity]);
    let new_atlas = world.resource::<GaussianLodBridgeManager>().clouds[&entity]
        .atlas
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
    let mut lower_frame_upload = current_settings.clone();
    lower_frame_upload.budgets.max_upload_bytes_per_frame -= 1;
    assert_ne!(
        current_signature,
        BridgeStructuralSignature::new(&lower_frame_upload, current_streaming, config, false),
        "the runtime preprocessor byte capacity is fixed at construction"
    );
    let mut lower_concurrency = current_streaming.clone();
    lower_concurrency.max_concurrent_requests -= 1;
    assert_ne!(
        current_signature,
        BridgeStructuralSignature::new(current_settings, &lower_concurrency, config, false),
        "the runtime preprocessor job capacity is fixed at construction"
    );
}

#[test]
fn lodge_strategy_switch_restores_the_ephemeral_source_before_retiring_its_atlas() {
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
            test_streaming(),
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();
    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[entity]);
    let old_atlas = world.resource::<GaussianLodBridgeManager>().clouds[&entity]
        .atlas
        .clone();
    assert_ne!(old_atlas.id(), source_handle.id());

    world.entity_mut(entity).insert((
        PlanarGaussian3dHandle(old_atlas.clone()),
        GaussianLodgeHandle::default(),
        LodRenderCandidates::default(),
    ));
    run_bridge_frame(&mut world, &mut schedule);

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
    assert!(world.get::<GaussianLodBridgeStatus>(entity).is_none());

    // Canceling the external switch leaves a valid finest source from which
    // the existing hierarchy strategy can be built again.
    world.entity_mut(entity).remove::<GaussianLodgeHandle>();
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[entity]);
    assert!(
        world
            .resource::<GaussianLodBridgeManager>()
            .clouds
            .contains_key(&entity)
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
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[entity]);
    let old_atlas = world.resource::<GaussianLodBridgeManager>().clouds[&entity]
        .atlas
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
    run_bridge_frame(&mut world, &mut schedule);

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
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[entity]);

    let new_atlas = world.resource::<GaussianLodBridgeManager>().clouds[&entity]
        .atlas
        .clone();
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(entity)
            .unwrap()
            .handle()
            .id(),
        source_handle.id(),
        "without a camera the immutable source remains the cold-start draw"
    );
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
fn in_place_source_mutation_discards_bounded_atlas_and_invalidates_the_cut() {
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
    mark_cloud_visible(&mut world, camera, cloud);

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    let candidate = (0..64)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            world
                .get::<LodRenderCandidates>(cloud)
                .and_then(|candidates| candidates.get(camera))
                .cloned()
        })
        .expect("a complete resident cut should be staged before mutation");
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    assert!(
        !world
            .get::<LodRenderCandidates>(cloud)
            .unwrap()
            .candidate_draw_required
    );
    candidate
        .phase
        .store(LOD_RENDER_PREPARED, Ordering::Release);
    run_bridge_frame(&mut world, &mut schedule);
    let atlas_handle = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(atlas_handle.id(), source_handle.id());
    assert!(
        world
            .get::<LodRenderCandidates>(cloud)
            .unwrap()
            .candidate_draw_required
    );
    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        assert!(!state.mirror.materialized_slots().is_empty());
    }
    let target_index = source.len() - 1;
    let replacement = source.get(0);
    assert_ne!(source.get(target_index), replacement);

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
    run_bridge_frame(&mut world, &mut schedule);

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
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&atlas_handle)
            .is_none(),
        "a source mutation retires the bounded cache instead of rewriting it as a flat source"
    );
    let queued = world
        .resource::<LodAtlasUploadQueue>()
        .queued_slots()
        .filter(|upload| upload.atlas == atlas_handle.id())
        .collect::<Vec<_>>();
    assert!(queued.is_empty());

    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[cloud]);
    let rebuilt_waiting = (0..64)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            world
                .get::<LodRenderCandidates>(cloud)
                .and_then(|candidates| candidates.get(camera))
                .cloned()
        })
        .expect("the rebuilt current-view cut should become resident");
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    rebuilt_waiting
        .phase
        .store(LOD_RENDER_PREPARED, Ordering::Release);
    run_bridge_frame(&mut world, &mut schedule);
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
            .get(&source_handle)
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
    run_until_bridge_initialization_finishes(&mut world, &mut schedule, &[entity]);
    run_bridge_frame(&mut world, &mut schedule);

    let bridged = world.get::<PlanarGaussian3dHandle>(entity).unwrap();
    assert_eq!(bridged.handle().id(), source_handle.id());
    let atlas_handle = world.resource::<GaussianLodBridgeManager>().clouds[&entity]
        .atlas
        .clone();
    let atlas = world
        .resource::<Assets<PlanarGaussian3d>>()
        .get(&atlas_handle)
        .unwrap();
    assert!(atlas.len() <= test_config().max_atlas_gaussians as usize);
    assert!(
        atlas
            .iter()
            .all(|gaussian| gaussian == Gaussian3d::default())
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
    run_bridge_frame(&mut world, &mut schedule);
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
    run_bridge_frame(&mut world, &mut schedule);

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
    mark_cloud_visible(&mut world, left, cloud);
    mark_cloud_visible(&mut world, right, cloud);

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    let candidates = (0..64)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
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
    assert!(
        !candidates.candidate_draw_required,
        "WAITING candidates must leave the complete source draw enabled"
    );
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    candidates
        .get(left)
        .unwrap()
        .phase
        .store(LOD_RENDER_PREPARED, Ordering::Release);
    run_bridge_frame(&mut world, &mut schedule);

    assert_ne!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id(),
        "one PREPARED camera cannot expose an all-camera atlas transaction"
    );
    assert!(
        !world
            .get::<LodRenderCandidates>(cloud)
            .unwrap()
            .candidate_draw_required
    );
    let atlas_handle = world.resource::<GaussianLodBridgeManager>().clouds[&cloud]
        .atlas
        .clone();
    let atlas = world
        .resource::<Assets<PlanarGaussian3d>>()
        .get(&atlas_handle)
        .unwrap();
    assert_eq!(
        atlas.len(),
        world.resource::<GaussianLodBridgeManager>().clouds[&cloud]
            .mirror
            .physical_gaussians() as usize
    );

    let staged_candidates = world.get::<LodRenderCandidates>(cloud).unwrap().clone();
    for candidate in staged_candidates.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        atlas_handle.id()
    );
    assert!(
        world
            .get::<LodRenderCandidates>(cloud)
            .unwrap()
            .candidate_draw_required,
        "the first materialized page must disable unfiltered atlas fallback"
    );
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::WaitingForRender
    );
    let queued_before = world.resource::<LodAtlasUploadQueue>().queued_slot_count();
    assert!(queued_before > 0);

    // A deferred GPU upload keeps PREPARED without causing the next main
    // frame to restore/rewrite the fallback or enqueue the same cut again.
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world.resource::<LodAtlasUploadQueue>().queued_slot_count(),
        queued_before
    );
    let active_candidates = world.get::<LodRenderCandidates>(cloud).unwrap().clone();
    for candidate in active_candidates.by_camera.values() {
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    let active_candidates = world.get::<LodRenderCandidates>(cloud).unwrap().clone();
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );
    assert_eq!(active_candidates.len(), 2);
    assert!(active_candidates.candidate_draw_required);
    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        assert!(
            state.current.as_ref().is_some_and(|current| {
                bridge_candidate_sets_match(current, &active_candidates)
            })
        );
        assert_eq!(
            state.current_page_leases,
            bridge_candidate_pages(&active_candidates)
        );
        assert!(state.pending_page_leases.is_empty());
    }
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

    // Device recreation revokes the retained compaction output without
    // changing its logical cut. PREPARED is not drawable yet, so status waits
    // while the already-materialized generationful slots are requeued.
    let retained_active_count = active_candidates
        .by_camera
        .values()
        .map(|candidate| u64::from(candidate.rendered_candidate_count()))
        .max()
        .unwrap();
    *world.resource_mut::<LodAtlasUploadQueue>() = LodAtlasUploadQueue::default();
    for candidate in active_candidates.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    let rebuilding_status = world.get::<GaussianLodBridgeStatus>(cloud).unwrap();
    assert_eq!(
        rebuilding_status.phase,
        GaussianLodBridgePhase::WaitingForRender
    );
    assert_eq!(rebuilding_status.active_gaussians, retained_active_count);
    let recovery_uploads = world
        .resource::<LodAtlasUploadQueue>()
        .queued_slots()
        .collect::<Vec<_>>();
    for candidate in active_candidates.by_camera.values() {
        for range in candidate.render_ranges() {
            assert!(
                recovery_uploads.iter().any(|upload| {
                    upload.atlas == atlas_handle.id() && upload.slot == range.slot
                })
            );
        }
    }
    assert!(
        world.resource::<GaussianLodBridgeManager>().clouds[&cloud]
            .current
            .as_ref()
            .is_some_and(|current| bridge_candidate_sets_match(current, &active_candidates))
    );
    assert!(
        world.resource::<GaussianLodBridgeManager>().clouds[&cloud]
            .handshakes
            .values()
            .all(|handshake| handshake.staged)
    );
    for candidate in active_candidates.by_camera.values() {
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );
    assert!(world.resource::<GaussianLodBridgeManager>().clouds[&cloud].active);

    // Prepare a finer replacement at this exact two-camera pose.
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 0.5;
    let pending = (0..128)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            let candidates = world.get::<LodRenderCandidates>(cloud)?.clone();
            (!candidates.is_empty()).then_some(candidates)
        })
        .expect("a finer camera cut should become a pending replacement");
    assert!(pending.candidate_draw_required);
    let pending_phases = pending
        .by_camera
        .iter()
        .map(|(&camera, candidate)| (camera, Arc::clone(&candidate.phase)))
        .collect::<BTreeMap<_, _>>();
    for phase in pending_phases.values() {
        phase.store(LOD_RENDER_PREPARED, Ordering::Release);
    }

    // Every published cut covers the global hierarchy. Continuous camera
    // motion therefore retains an explicit bounded atlas candidate and never
    // falls back to the virtual source.
    for step in 0..12 {
        let swap = step % 2 == 0;
        *world.get_mut::<GlobalTransform>(left).unwrap() =
            GlobalTransform::from(Transform::from_xyz(if swap { 8.0 } else { 0.0 }, 0.0, 5.0));
        *world.get_mut::<GlobalTransform>(right).unwrap() =
            GlobalTransform::from(Transform::from_xyz(if swap { 0.0 } else { 8.0 }, 0.0, 5.0));
        run_bridge_frame(&mut world, &mut schedule);

        let moving = world
            .get::<LodRenderCandidates>(cloud)
            .expect("motion must retain a bounded candidate");
        assert_eq!(moving.len(), 2);
        assert!(moving.candidate_draw_required);
        assert_ne!(
            world
                .get::<PlanarGaussian3dHandle>(cloud)
                .unwrap()
                .handle()
                .id(),
            source_handle.id()
        );
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        assert!(!state.flat_source_bypass);
        assert!(state.current.is_some());
        assert!(!state.current_page_leases.is_empty());
        assert!(moving.by_camera.values().all(|candidate| {
            u64::from(candidate.rendered_candidate_count())
                <= world
                    .get::<GaussianLodSettings>(cloud)
                    .unwrap()
                    .budgets
                    .max_active_gaussians
        }));
    }
    assert!(pending_phases.values().all(|phase| matches!(
        phase.load(Ordering::Acquire),
        LOD_RENDER_PREPARED | LOD_RENDER_ACTIVE
    )));

    // A prepared globally covering cut may safely activate after the camera
    // changes; the following frame immediately retargets quality to the live
    // pose without a source-sized transition.
    for phase in pending_phases.values() {
        phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    *world.get_mut::<GlobalTransform>(left).unwrap() =
        GlobalTransform::from(Transform::from_xyz(4.0, 0.0, 5.0));
    *world.get_mut::<GlobalTransform>(right).unwrap() =
        GlobalTransform::from(Transform::from_xyz(-4.0, 0.0, 5.0));
    run_bridge_frame(&mut world, &mut schedule);

    let settled = (0..128)
        .find_map(|_| {
            let candidates = world.get::<LodRenderCandidates>(cloud)?.clone();
            if candidates.len() == 2
                && candidates.by_camera.values().all(|candidate| {
                    pending_phases
                        .values()
                        .all(|phase| !Arc::ptr_eq(phase, &candidate.phase))
                })
            {
                Some(candidates)
            } else {
                run_bridge_frame(&mut world, &mut schedule);
                None
            }
        })
        .expect("the settled current-view cut should resume staging");
    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        assert!(!state.flat_source_bypass);
        assert!(state.current.is_some());
    }
    for candidate in settled.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    for candidate in settled.by_camera.values() {
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
    assert!(
        state
            .current
            .as_ref()
            .is_some_and(|current| { bridge_candidate_sets_match(current, &settled) })
    );
    assert_eq!(state.current_views.len(), 2);
    assert!(pending_phases.values().all(|phase| {
        !state
            .current
            .as_ref()
            .unwrap()
            .by_camera
            .values()
            .any(|candidate| Arc::ptr_eq(phase, &candidate.phase))
    }));
    let active_phases = state
        .current
        .as_ref()
        .unwrap()
        .by_camera
        .iter()
        .map(|(&camera, candidate)| (camera, Arc::clone(&candidate.phase)))
        .collect::<BTreeMap<_, _>>();
    let active_leases = state.current_page_leases.clone();

    let atlas_before_visibility_change = world
        .resource::<Assets<PlanarGaussian3d>>()
        .get(&atlas_handle)
        .unwrap()
        .iter()
        .collect::<Vec<_>>();

    world.entity_mut(cloud).insert(ViewVisibility::HIDDEN);
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        atlas_handle.id()
    );
    assert!(world.get::<LodRenderCandidates>(cloud).is_some());
    let hidden_state = world
        .resource::<GaussianLodBridgeManager>()
        .clouds
        .get(&cloud)
        .unwrap();
    assert_eq!(hidden_state.views.len(), 2);
    assert_eq!(hidden_state.current_page_leases, active_leases);
    assert!(hidden_state.pending_page_leases.is_empty());
    assert!(hidden_state.active);
    assert!(!hidden_state.flat_source_bypass);
    for (&camera, phase) in &active_phases {
        assert!(Arc::ptr_eq(
            &hidden_state
                .current
                .as_ref()
                .unwrap()
                .get(camera)
                .unwrap()
                .phase,
            phase
        ));
    }
    let unchanged_atlas = world
        .resource::<Assets<PlanarGaussian3d>>()
        .get(&atlas_handle)
        .unwrap();
    assert_eq!(
        unchanged_atlas.iter().collect::<Vec<_>>(),
        atlas_before_visibility_change
    );

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&atlas_handle)
            .is_none(),
        "quality one retires the bounded owned page-cache atlas"
    );
    assert!(world.get::<LodRenderCandidates>(cloud).is_none());
}

#[test]
fn camera_visibility_membership_commits_active_old_transaction_before_retarget() {
    let mut world = World::new();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>();
    world.insert_resource(test_config());
    world.init_resource::<GaussianLodBridgeManager>();
    world.init_resource::<LodAtlasUploadQueue>();

    let source_handle = world
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(LodTestScene::screen_space_ladder().cloud());
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
    let spawn_camera = |world: &mut World, x: f32| {
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
                GlobalTransform::from(Transform::from_xyz(x, 0.0, 5.0)),
                GaussianCamera::default(),
            ))
            .id()
    };
    let camera_a = spawn_camera(&mut world, 0.0);
    let camera_b = spawn_camera(&mut world, 8.0);
    mark_cloud_visible(&mut world, camera_a, cloud);

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    let a_only = (0..128)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            world
                .get::<LodRenderCandidates>(cloud)
                .filter(|candidates| {
                    candidates.len() == 1
                        && candidates.get(camera_a).is_some()
                        && candidates.get(camera_b).is_none()
                })
                .cloned()
        })
        .expect("only the camera that lists the cloud should join the transaction");
    assert!(!a_only.candidate_draw_required);
    for candidate in a_only.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    let prepared_a = world.get::<LodRenderCandidates>(cloud).unwrap().clone();
    for candidate in prepared_a.by_camera.values() {
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world.resource::<GaussianLodBridgeManager>().clouds[&cloud]
            .current
            .as_ref()
            .unwrap()
            .len(),
        1
    );

    // Stage a distinct A-only replacement, then let render publish ACTIVE in
    // the same main-world interval in which camera B gains cloud membership.
    // The old-membership transaction has already rendered and must commit
    // before the bridge clears its handshakes to retarget A+B.
    let current_a_phase = Arc::clone(
        &world.resource::<GaussianLodBridgeManager>().clouds[&cloud]
            .current
            .as_ref()
            .unwrap()
            .get(camera_a)
            .unwrap()
            .phase,
    );
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 0.5;
    let replacement_a = (0..256)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            let candidates = world.get::<LodRenderCandidates>(cloud)?;
            let candidate = candidates.get(camera_a)?;
            (candidates.len() == 1 && !Arc::ptr_eq(&candidate.phase, &current_a_phase))
                .then(|| candidate.clone())
        })
        .expect("camera A should reach a distinct detailed replacement");
    let replacement_a_pages = replacement_a
        .render_ranges()
        .iter()
        .map(|range| range.page)
        .collect::<BTreeSet<_>>();
    replacement_a
        .phase
        .store(LOD_RENDER_PREPARED, Ordering::Release);
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world.resource::<GaussianLodBridgeManager>().clouds[&cloud].pending_page_leases,
        replacement_a_pages
    );
    replacement_a
        .phase
        .store(LOD_RENDER_ACTIVE, Ordering::Release);

    mark_cloud_visible(&mut world, camera_b, cloud);
    run_bridge_frame(&mut world, &mut schedule);
    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        let committed = state
            .current
            .as_ref()
            .expect("the rendered A-only replacement must commit");
        assert_eq!(state.views, BTreeSet::from([camera_a, camera_b]));
        assert_eq!(state.current_views.len(), 1);
        assert!(state.current_views.contains_key(&camera_a));
        assert!(Arc::ptr_eq(
            &committed.get(camera_a).unwrap().phase,
            &replacement_a.phase
        ));
        assert_eq!(state.current_page_leases, replacement_a_pages);
        assert_eq!(
            replacement_a.phase.load(Ordering::Acquire),
            LOD_RENDER_ACTIVE
        );
    }
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );

    let both = (0..64)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
            assert!(Arc::ptr_eq(
                &state
                    .current
                    .as_ref()
                    .expect("camera A must retain its last rendered output")
                    .get(camera_a)
                    .unwrap()
                    .phase,
                &replacement_a.phase
            ));
            world
                .get::<LodRenderCandidates>(cloud)
                .filter(|candidates| {
                    candidates.len() == 2
                        && candidates.get(camera_a).is_some()
                        && candidates.get(camera_b).is_some()
                })
                .cloned()
        })
        .expect("newly visible camera must join the next all-camera transaction");
    for candidate in both.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    let prepared_both = world.get::<LodRenderCandidates>(cloud).unwrap().clone();
    assert_eq!(prepared_both.len(), 2);
    for candidate in prepared_both.by_camera.values() {
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    let current = world.resource::<GaussianLodBridgeManager>().clouds[&cloud]
        .current
        .as_ref()
        .unwrap()
        .clone();
    assert_eq!(current.len(), 2);
    assert!(current.get(camera_a).is_some());
    assert!(current.get(camera_b).is_some());
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );

    // Exercise the inverse race. A complete A+B replacement which render has
    // already published ACTIVE must commit before camera B is removed. The
    // full rendered page union stays leased until the next A-only transaction
    // becomes ACTIVE, while camera A never rolls back to the older cut.
    let previous_both = current;
    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 0.0;
    let replacement_both = (0..256)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            let candidates = world.get::<LodRenderCandidates>(cloud)?;
            (candidates.len() == 2
                && candidates.by_camera.iter().any(|(camera, candidate)| {
                    previous_both
                        .get(*camera)
                        .is_some_and(|previous| !Arc::ptr_eq(&candidate.phase, &previous.phase))
                }))
            .then(|| candidates.clone())
        })
        .expect("both cameras should reach a distinct coarse replacement");
    let replacement_both_pages = bridge_candidate_pages(&replacement_both);
    for candidate in replacement_both.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    assert_eq!(
        world.resource::<GaussianLodBridgeManager>().clouds[&cloud].pending_page_leases,
        replacement_both_pages
    );
    for candidate in replacement_both.by_camera.values() {
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    mark_cloud_hidden(&mut world, camera_b, cloud);
    run_bridge_frame(&mut world, &mut schedule);
    {
        let manager = world.resource::<GaussianLodBridgeManager>();
        let state = &manager.clouds[&cloud];
        let committed = state
            .current
            .as_ref()
            .expect("the rendered A+B replacement must commit before removal");
        assert_eq!(state.views, BTreeSet::from([camera_a]));
        assert_eq!(state.current_views.len(), 2);
        for (&camera, candidate) in &replacement_both.by_camera {
            assert!(Arc::ptr_eq(
                &committed.get(camera).unwrap().phase,
                &candidate.phase
            ));
            assert_eq!(candidate.phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);
        }
        assert_eq!(state.current_page_leases, replacement_both_pages);
    }
    assert_eq!(
        world.get::<GaussianLodBridgeStatus>(cloud).unwrap().phase,
        GaussianLodBridgePhase::Active
    );

    let a_after_removal = (0..64)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
            assert!(Arc::ptr_eq(
                &state.current.as_ref().unwrap().get(camera_a).unwrap().phase,
                &replacement_both.get(camera_a).unwrap().phase
            ));
            world
                .get::<LodRenderCandidates>(cloud)
                .filter(|candidates| {
                    candidates.len() == 1
                        && candidates.get(camera_a).is_some()
                        && candidates.get(camera_b).is_none()
                })
                .cloned()
        })
        .expect("camera B must leave the next all-camera transaction");
    for candidate in a_after_removal.by_camera.values() {
        candidate
            .phase
            .store(LOD_RENDER_PREPARED, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    for candidate in a_after_removal.by_camera.values() {
        candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    }
    run_bridge_frame(&mut world, &mut schedule);
    let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
    assert_eq!(state.current.as_ref().unwrap().len(), 1);
    assert!(state.current.as_ref().unwrap().get(camera_a).is_some());
    assert!(state.current.as_ref().unwrap().get(camera_b).is_none());
    assert_eq!(
        state.current_page_leases,
        bridge_candidate_pages(&a_after_removal)
    );
}

#[test]
fn frozen_camera_motion_preserves_selector_provenance_without_duplicate_leases() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.selection_mode = LodSelectionMode::Frozen;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_pending_requests = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, atlas) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    state.transient_atlas = Some(LodTransientAtlas::new(atlas));

    let camera = Entity::from_bits(76);
    let frozen_view = LodView::perspective(Vec3::new(0.0, 0.0, 5.0), 720.0, 1.0, 0.1);
    let mut views = [BridgeCameraView {
        entity: camera,
        view: frozen_view,
    }];
    let mut effective = state.structural.apply(&settings);
    let mut assets = Assets::<PlanarGaussian3d>::default();
    let mut uploads = LodAtlasUploadQueue::default();

    let initial = (0..128)
        .find_map(|_| {
            let (candidates, _) = update_bridge_cloud(
                &mut state,
                &effective,
                &GlobalTransform::IDENTITY,
                &views,
                &mut assets,
                &mut uploads,
            )
            .unwrap();
            candidates.get(camera).cloned()
        })
        .expect("the initial frozen-view cut should become resident");
    let frozen_phase = Arc::clone(&initial.phase);
    frozen_phase.store(LOD_RENDER_PREPARED, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    frozen_phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();

    let frozen_pages = state.current_page_leases.clone();
    assert!(!frozen_pages.is_empty());
    assert!(state.pending_page_leases.is_empty());
    assert_eq!(state.current_views.get(&camera), Some(&frozen_view));
    assert_eq!(
        state.frozen_selection_views.get(&camera),
        Some(&frozen_view)
    );
    let presentation_guard_pages = state
        .runtime
        .coverage_guard_candidate(runtime_view_id(camera), frozen_view, &effective)
        .unwrap()
        .expect("the presentation guard should remain resident")
        .physical_ranges()
        .iter()
        .map(|range| range.page)
        .collect::<BTreeSet<_>>();
    for &page in &presentation_guard_pages {
        assert!(state.runtime.resident_pin_count(page).unwrap_or(0) >= 1);
    }
    for &page in &frozen_pages {
        assert_eq!(
            state.runtime.resident_pin_count(page),
            Some(2 + u32::from(presentation_guard_pages.contains(&page))),
            "runtime frontier and current-render leases are expected, plus a guard lease only when pages overlap"
        );
    }
    let pending_acquisitions_after_activation = state.pre_frame_pending_lease_acquisitions;

    for step in 1..=32 {
        views[0].view = LodView::perspective(Vec3::new(step as f32, 0.0, 5.0), 720.0, 1.0, 0.1);
        let (candidates, status) = update_bridge_cloud(
            &mut state,
            &effective,
            &GlobalTransform::IDENTITY,
            &views,
            &mut assets,
            &mut uploads,
        )
        .unwrap();
        let candidate = candidates
            .get(camera)
            .expect("frozen motion must keep publishing the captured cut");
        assert!(Arc::ptr_eq(&candidate.phase, &frozen_phase));
        assert!(candidate.frontier().selection_view_frozen());
        assert_eq!(status.phase, GaussianLodBridgePhase::Active);
        assert_eq!(state.current_page_leases, frozen_pages);
        assert!(state.pending_page_leases.is_empty());
        assert_eq!(state.current_views.get(&camera), Some(&frozen_view));
        assert_eq!(
            state.frozen_selection_views.get(&camera),
            Some(&frozen_view)
        );
        assert_eq!(
            state.pre_frame_pending_lease_acquisitions, pending_acquisitions_after_activation,
            "live-camera motion in Frozen mode must not open a replacement transaction"
        );
        for &page in &frozen_pages {
            assert_eq!(
                state.runtime.resident_pin_count(page),
                Some(2 + u32::from(presentation_guard_pages.contains(&page))),
                "Frozen motion must not add a second render lease"
            );
        }
    }

    // Unfreezing retargets selector provenance without revoking the globally
    // covering cut or opening a duplicate lease transaction.
    settings.selection_mode = LodSelectionMode::Dynamic;
    effective = state.structural.apply(&settings);
    let (dynamic_candidates, status) = update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    let dynamic = dynamic_candidates
        .get(camera)
        .expect("the globally covering cut remains drawable while unfreezing");
    assert_eq!(status.phase, GaussianLodBridgePhase::Active);
    assert_eq!(frozen_phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);
    assert!(state.current.is_some());
    assert_eq!(state.current_page_leases, frozen_pages);
    assert!(state.pending_page_leases.is_empty());
    assert!(state.frozen_selection_views.is_empty());
    for &page in &frozen_pages {
        assert_eq!(
            state.runtime.resident_pin_count(page),
            Some(2 + u32::from(presentation_guard_pages.contains(&page)))
        );
    }
    assert!(!dynamic.frontier().selection_view_frozen());
    assert!(Arc::ptr_eq(&dynamic.phase, &frozen_phase));
    assert_eq!(state.handshakes[&camera].selected_view, views[0].view);
    assert_eq!(state.current_views.get(&camera), Some(&views[0].view));
    assert!(state.pending_page_leases.is_empty());
    for &page in &state.current_page_leases {
        assert_eq!(
            state.runtime.resident_pin_count(page),
            Some(2 + u32::from(presentation_guard_pages.contains(&page)))
        );
    }
}

#[test]
fn dynamic_single_camera_motion_retains_globally_covering_atlas_cut() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_pending_requests = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, atlas) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    state.transient_atlas = Some(LodTransientAtlas::new(atlas));

    let camera = Entity::from_bits(77);
    let mut views = [BridgeCameraView {
        entity: camera,
        view: LodView::perspective(Vec3::new(0.0, 0.0, 5.0), 720.0, 1.0, 0.1),
    }];
    let effective = state.structural.apply(&settings);
    let mut assets = Assets::<PlanarGaussian3d>::default();
    let mut uploads = LodAtlasUploadQueue::default();

    let initial = (0..128)
        .find_map(|_| {
            let (candidates, _) = update_bridge_cloud(
                &mut state,
                &effective,
                &GlobalTransform::IDENTITY,
                &views,
                &mut assets,
                &mut uploads,
            )
            .unwrap();
            candidates.get(camera).cloned()
        })
        .expect("the initial exact-view root cut should become resident");
    let stale_phase = Arc::clone(&initial.phase);
    stale_phase.store(LOD_RENDER_PREPARED, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    stale_phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    assert!(state.current.is_some());

    for step in 1..=12 {
        views[0].view = LodView::perspective(Vec3::new(step as f32, 0.0, 5.0), 720.0, 1.0, 0.1);
        let (candidates, status) = update_bridge_cloud(
            &mut state,
            &effective,
            &GlobalTransform::IDENTITY,
            &views,
            &mut assets,
            &mut uploads,
        )
        .unwrap();
        let candidate = candidates
            .get(camera)
            .expect("motion keeps an explicit bounded atlas cut");
        assert!(
            u64::from(candidate.rendered_candidate_count())
                <= settings.budgets.max_active_gaussians
        );
        assert_eq!(status.phase, GaussianLodBridgePhase::Active);
        assert!(status.active_gaussians <= settings.budgets.max_active_gaussians);
        assert!(!state.flat_source_bypass);
        assert!(state.current.is_some());
        assert!(!state.current_page_leases.is_empty());
        assert!(state.pending_page_leases.is_empty());
    }
    assert_eq!(stale_phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);

    let (settled, _) = update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    let settled_candidate = settled
        .get(camera)
        .expect("the same stable frame remains drawable without debounce");
    assert!(Arc::ptr_eq(&settled_candidate.phase, &stale_phase));
    assert_eq!(state.handshakes[&camera].selected_view, views[0].view);
}

#[test]
fn active_current_token_survives_progressive_page_waves() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 1;
    settings.budgets.max_pending_requests = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, atlas) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    state.transient_atlas = Some(LodTransientAtlas::new(atlas));

    let camera = Entity::from_bits(78);
    let mut views = [BridgeCameraView {
        entity: camera,
        view: LodView::perspective(Vec3::new(0.0, 0.0, 5.0), 720.0, 1.0, 0.1),
    }];
    let mut assets = Assets::<PlanarGaussian3d>::default();
    let mut uploads = LodAtlasUploadQueue::default();
    let mut effective = state.structural.apply(&settings);

    let initial = (0..128)
        .find_map(|_| {
            let (candidates, _) = update_bridge_cloud(
                &mut state,
                &effective,
                &GlobalTransform::IDENTITY,
                &views,
                &mut assets,
                &mut uploads,
            )
            .unwrap();
            candidates.get(camera).cloned()
        })
        .expect("the initial coarse target should reach a quiescent cut");
    initial.phase.store(LOD_RENDER_PREPARED, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    initial.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();

    let current_phase = Arc::clone(
        &state
            .current
            .as_ref()
            .expect("the coarse cut should be active")
            .get(camera)
            .unwrap()
            .phase,
    );
    let current_leases = state.current_page_leases.clone();
    let deferred_before = state.deferred_ordinary_publications;

    settings.quality = 0.5;
    effective = state.structural.apply(&settings);
    let mut saw_deferred_page_wave = false;
    let replacement = (0..256)
        .find_map(|_| {
            let (candidates, status) = update_bridge_cloud(
                &mut state,
                &effective,
                &GlobalTransform::IDENTITY,
                &views,
                &mut assets,
                &mut uploads,
            )
            .unwrap();
            let candidate = candidates
                .get(camera)
                .expect("the active bounded cut must remain explicit");
            if Arc::ptr_eq(&candidate.phase, &current_phase) {
                assert_eq!(status.phase, GaussianLodBridgePhase::Active);
                assert!(Arc::ptr_eq(
                    &state.current.as_ref().unwrap().get(camera).unwrap().phase,
                    &current_phase
                ));
                assert_eq!(state.current_page_leases, current_leases);
                assert!(state.pending_page_leases.is_empty());
                saw_deferred_page_wave |= state.deferred_ordinary_publications > deferred_before;
                None
            } else {
                assert_eq!(status.phase, GaussianLodBridgePhase::Active);
                assert_eq!(
                    candidate.phase.load(Ordering::Acquire),
                    LOD_RENDER_WAITING,
                    "the first distinct replacement must remain pending while the retained cut is ACTIVE"
                );
                assert!(Arc::ptr_eq(
                    &state.current.as_ref().unwrap().get(camera).unwrap().phase,
                    &current_phase
                ));
                assert_eq!(state.current_page_leases, current_leases);
                Some(candidate.clone())
            }
        })
        .expect("the finer target should publish exactly after its demand drains");

    assert!(saw_deferred_page_wave);
    assert_eq!(current_phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);
    assert!(Arc::ptr_eq(
        &state.current.as_ref().unwrap().get(camera).unwrap().phase,
        &current_phase
    ));
    assert_ne!(replacement.phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);

    // The render world may publish ACTIVE after the preceding main-world
    // update. Before the bridge consumes that phase, a moved camera can request
    // the next refinement wave. The already-rendered replacement must commit
    // first instead of being revoked by the new ordinary-demand gate.
    let replacement_pages = replacement
        .render_ranges()
        .iter()
        .map(|range| range.page)
        .collect::<BTreeSet<_>>();
    let retained_before = state.pre_frame_staged_replacement_retentions;
    replacement
        .phase
        .store(LOD_RENDER_ACTIVE, Ordering::Release);
    settings.quality = 1.0;
    effective = state.structural.apply(&settings);
    views[0].view = LodView::perspective(Vec3::new(8.0, 0.0, 1.0), 720.0, 1.0, 0.1);
    let (published, status) = update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    assert!(state.pre_frame_staged_replacement_retentions > retained_before);
    assert_eq!(status.phase, GaussianLodBridgePhase::Active);
    assert!(Arc::ptr_eq(
        &published.get(camera).unwrap().phase,
        &replacement.phase
    ));
    assert!(Arc::ptr_eq(
        &state.current.as_ref().unwrap().get(camera).unwrap().phase,
        &replacement.phase
    ));
    assert_eq!(state.current_page_leases, replacement_pages);
    assert!(state.pending_page_leases.is_empty());
}

#[test]
fn active_replacement_commits_before_same_policy_pose_demand() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 1;
    settings.budgets.max_pending_requests = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, atlas) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    state.transient_atlas = Some(LodTransientAtlas::new(atlas));

    let camera = Entity::from_bits(79);
    let mut views = [BridgeCameraView {
        entity: camera,
        view: LodView::perspective(Vec3::new(0.0, 0.0, 5.0), 720.0, 1.0, 0.1),
    }];
    let mut assets = Assets::<PlanarGaussian3d>::default();
    let mut uploads = LodAtlasUploadQueue::default();
    let mut effective = state.structural.apply(&settings);

    let coarse = (0..128)
        .find_map(|_| {
            let (candidates, _) = update_bridge_cloud(
                &mut state,
                &effective,
                &GlobalTransform::IDENTITY,
                &views,
                &mut assets,
                &mut uploads,
            )
            .unwrap();
            candidates.get(camera).cloned()
        })
        .expect("the initial coarse cut should become resident");
    coarse.phase.store(LOD_RENDER_PREPARED, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    coarse.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    let coarse_phase = Arc::clone(&state.current.as_ref().unwrap().get(camera).unwrap().phase);

    settings.quality = 0.5;
    effective = state.structural.apply(&settings);
    let replacement = (0..256)
        .find_map(|_| {
            let (candidates, _) = update_bridge_cloud(
                &mut state,
                &effective,
                &GlobalTransform::IDENTITY,
                &views,
                &mut assets,
                &mut uploads,
            )
            .unwrap();
            let candidate = candidates.get(camera)?;
            (!Arc::ptr_eq(&candidate.phase, &coarse_phase)).then(|| candidate.clone())
        })
        .expect("the same-policy pose test needs a distinct detailed replacement");
    let replacement_pages = replacement
        .render_ranges()
        .iter()
        .map(|range| range.page)
        .collect::<BTreeSet<_>>();
    let retained_before = state.pre_frame_staged_replacement_retentions;
    replacement
        .phase
        .store(LOD_RENDER_ACTIVE, Ordering::Release);

    // Keep the quality target, cap, and selection mode identical. Only camera
    // pose changes, which may immediately request finer pages but does not make
    // the already-rendered globally covering replacement incompatible.
    views[0].view = LodView::perspective(Vec3::ZERO, 720.0, 1.0, 0.1);
    let (published, status) = update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    assert!(state.pre_frame_staged_replacement_retentions > retained_before);
    assert_eq!(status.phase, GaussianLodBridgePhase::Active);
    assert!(Arc::ptr_eq(
        &published.get(camera).unwrap().phase,
        &replacement.phase
    ));
    assert!(Arc::ptr_eq(
        &state.current.as_ref().unwrap().get(camera).unwrap().phase,
        &replacement.phase
    ));
    assert_eq!(state.current_page_leases, replacement_pages);
    assert!(state.pending_page_leases.is_empty());
}

#[test]
fn cold_capacity_stall_suppresses_degraded_guard_and_keeps_source_drawable() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 1.0;
    settings.frustum_culling = false;
    // The packed promoted guard occupies one of two physical slots but consumes
    // the independently bounded resident-Gaussian budget. Requested detail is
    // therefore terminally blocked while the bridge still has no render lease.
    settings.budgets.max_resident_pages = 2;
    settings.budgets.max_resident_gaussians = 16;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_pending_requests = 1;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, atlas) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    assert_eq!(state.mirror.slot_count(), 2);
    state.transient_atlas = Some(LodTransientAtlas::new(atlas));

    let camera = Entity::from_bits(98);
    let views = [BridgeCameraView {
        entity: camera,
        view: LodView::perspective(Vec3::ZERO, 720.0, 1.0, 0.1),
    }];
    let effective = state.structural.apply(&settings);
    let mut assets = Assets::<PlanarGaussian3d>::default();
    let mut uploads = LodAtlasUploadQueue::default();
    let mut saw_deferred_ancestor = false;

    for _ in 0..256 {
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(1));
        let (candidates, status) = update_bridge_cloud(
            &mut state,
            &effective,
            &GlobalTransform::IDENTITY,
            &views,
            &mut assets,
            &mut uploads,
        )
        .unwrap();
        saw_deferred_ancestor |= state.deferred_ordinary_publications > 0;
        assert!(candidates.is_empty());
        assert!(!candidates.candidate_draw_required);
        assert_eq!(status.phase, GaussianLodBridgePhase::StreamingFallback);
        assert!(state.current.is_none());
        assert!(state.current_page_leases.is_empty());
        assert!(state.pending_page_leases.is_empty());
        assert!(state.flat_source_bypass);
    }

    assert!(saw_deferred_ancestor);
    assert!(state.deferred_ordinary_publications > 0);
    assert_eq!(state.capacity_pressure_stable_frames, 0);
    assert!(state.capacity_pressure_total_frames >= CAPACITY_PRESSURE_ESCAPE_FRAMES);

    let degraded_guard = state
        .runtime
        .coverage_guard_candidate(runtime_view_id(camera), views[0].view, &effective)
        .unwrap()
        .expect("the suppressed permanent guard should remain resident");
    assert!(degraded_guard.is_coverage_guard());
    assert!(degraded_guard.physical_ranges().len() > 1);
    assert!(degraded_guard.candidate_count() < source.len() as u32);
    assert!(matches!(
        degraded_guard.quality_status().degradation,
        crate::LodDegradation::Residency | crate::LodDegradation::Multiple
    ));
    assert!(degraded_guard.quality_status().achieved_max_target_ratio > 1.0);
    assert!(!coverage_guard_frontiers_satisfy_requested_quality(&[(
        camera,
        degraded_guard,
    )]));

    let mut coarse_settings = effective.clone();
    coarse_settings.quality = 0.0;
    let satisfied_guard = state
        .runtime
        .coverage_guard_candidate(runtime_view_id(camera), views[0].view, &coarse_settings)
        .unwrap()
        .expect("the same permanent guard should satisfy the coarsest target");
    assert_eq!(
        satisfied_guard.quality_status().degradation,
        crate::LodDegradation::None
    );
    assert!(
        satisfied_guard
            .quality_status()
            .achieved_max_target_ratio
            .is_finite()
    );
    assert!(satisfied_guard.quality_status().achieved_max_target_ratio <= 1.0);
    assert!(coverage_guard_frontiers_satisfy_requested_quality(&[(
        camera,
        satisfied_guard,
    )]));
}

#[test]
fn cold_saturated_pipeline_suppresses_degraded_guard_without_draining() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 1.0;
    settings.frustum_culling = false;
    settings.budgets.max_active_gaussians = 4_096;
    settings.budgets.max_resident_pages = 2;
    settings.budgets.max_resident_gaussians = 32;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_pending_requests = 32;
    settings.budgets.max_requests_per_frame = 1;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let mut config = test_config();
    config.streaming_settings.max_concurrent_requests = 1;
    let streaming = config.streaming_settings.clone();
    let (mut state, atlas) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &streaming,
        &config,
        false,
    )
    .unwrap();
    assert_eq!(state.mirror.slot_count(), 2);
    state.transient_atlas = Some(LodTransientAtlas::new(atlas));
    let effective = state.structural.apply(&settings);

    // Permit exactly the promoted guard and root requests to complete. The
    // next detail request remains live forever, reproducing Garden's full-cache
    // pipeline churn without relying on worker timing or a drained block.
    let mut built = build_planar_3d_lod(&source, config.build_settings).unwrap();
    let root = built.manifest.roots[0];
    let root_page = built
        .manifest
        .nodes
        .iter()
        .find(|node| node.id == root)
        .unwrap()
        .representation
        .page;
    let mut memory = MemoryPageTransport::default();
    for page in &built.pages {
        let encoded = encode_page(page).unwrap();
        let descriptor = built
            .manifest
            .pages
            .iter_mut()
            .find(|descriptor| descriptor.id == page.id)
            .unwrap();
        descriptor.storage = Some(LodPageStorage {
            uri: format!("memory://cold-saturated/{}", page.id.0),
            byte_range: None,
            encoded_len: encoded.len() as u64,
        });
        memory.insert(page.id, encoded);
    }
    let remaining_ready = Arc::new(AtomicU32::new(2));
    state.runtime = Box::new(
        LodStreamingRuntime::new(
            built.manifest,
            FirstPollsThenPendingTransport {
                inner: memory,
                remaining_ready: Arc::clone(&remaining_ready),
            },
            &effective,
            &streaming,
        )
        .unwrap(),
    );

    let camera = Entity::from_bits(101);
    let views = [BridgeCameraView {
        entity: camera,
        view: LodView::perspective(Vec3::ZERO, 720.0, 1.0, 0.1),
    }];
    let mut assets = Assets::<PlanarGaussian3d>::default();
    let mut uploads = LodAtlasUploadQueue::default();
    let mut pressure_started = false;

    for _ in 0..256 {
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(1));
        let (candidates, status) = update_bridge_cloud(
            &mut state,
            &effective,
            &GlobalTransform::IDENTITY,
            &views,
            &mut assets,
            &mut uploads,
        )
        .unwrap();
        pressure_started |= state.capacity_pressure_total_frames > 0;
        assert!(candidates.is_empty());
        assert!(!candidates.candidate_draw_required);
        assert_eq!(status.phase, GaussianLodBridgePhase::StreamingFallback);
        assert!(state.current.is_none());
        assert!(state.current_page_leases.is_empty());
        assert!(state.pending_page_leases.is_empty());
        assert!(state.flat_source_bypass);
    }

    assert!(pressure_started);
    assert_eq!(state.capacity_pressure_stable_frames, 0);
    assert!(state.capacity_pressure_total_frames >= CAPACITY_PRESSURE_ESCAPE_FRAMES);
    assert_eq!(remaining_ready.load(Ordering::Acquire), 0);

    // Suppression must persist while the ordinary selector still has a
    // complete ancestor, unresolved demand, and a non-draining pipeline.
    let runtime_frame = state.runtime.begin_frame();
    let probe = state
        .runtime
        .update_view_in_frame(
            runtime_frame,
            runtime_view_id(camera),
            views[0].view,
            &effective,
            &state.streaming,
        )
        .unwrap();
    state.runtime.finish_frame(runtime_frame).unwrap();
    assert!(probe.has_complete_resident_cut());
    assert!(
        probe
            .candidate_frontier(effective.max_active_gaussians_u32())
            .is_ok()
    );
    assert!(!probe.frontier().requested_nodes.is_empty());
    assert_eq!(probe.cache_stats().resident_pages, 2);
    assert!(
        probe
            .queued_requests()
            .saturating_add(probe.in_flight_requests())
            > 0
    );
    assert_eq!(probe.capacity_blocked_requests(), 0);

    let guard_candidate = state
        .runtime
        .coverage_guard_candidate(runtime_view_id(camera), views[0].view, &effective)
        .unwrap()
        .expect("the suppressed permanent guard should remain resident");
    let guard_pages = guard_candidate
        .physical_ranges()
        .iter()
        .map(|range| range.page)
        .collect::<BTreeSet<_>>();
    assert_eq!(guard_pages.len(), 1);
    assert!(!guard_pages.contains(&root_page));
    assert!(guard_candidate.is_coverage_guard());
    assert!(guard_candidate.physical_ranges().len() > 1);
    assert!(guard_candidate.quality_status().achieved_max_target_ratio > 1.0);
    assert!(!coverage_guard_frontiers_satisfy_requested_quality(&[(
        camera,
        guard_candidate,
    )]));
    assert!(state.current.is_none());
    assert!(state.current_page_leases.is_empty());
    assert!(state.pending_page_leases.is_empty());
    assert!(state.flat_source_bypass);
}

#[test]
fn saturated_refinement_publishes_stable_resident_relief_before_guard() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.frustum_culling = false;
    // Eight 16-record slots match this 125-record scene's exact leaf-page
    // footprint, but the independently pinned guard makes exact coexistence
    // require a ninth resident slot.
    settings.budgets.max_resident_pages = 8;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_pending_requests = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, atlas) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    assert_eq!(state.mirror.slot_count(), 8);
    state.transient_atlas = Some(LodTransientAtlas::new(atlas));

    let camera = Entity::from_bits(99);
    let mut views = [BridgeCameraView {
        entity: camera,
        view: LodView::perspective(Vec3::ZERO, 720.0, 1.0, 0.1),
    }];
    let mut assets = Assets::<PlanarGaussian3d>::default();
    let mut uploads = LodAtlasUploadQueue::default();
    let mut effective = state.structural.apply(&settings);

    let coarse = (0..64)
        .find_map(|_| {
            let (candidates, _) = update_bridge_cloud(
                &mut state,
                &effective,
                &GlobalTransform::IDENTITY,
                &views,
                &mut assets,
                &mut uploads,
            )
            .unwrap();
            candidates.get(camera).is_some().then_some(candidates)
        })
        .expect("coarse root should become resident");
    let coarse_phase = Arc::clone(&coarse.get(camera).unwrap().phase);
    coarse_phase.store(LOD_RENDER_PREPARED, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    coarse_phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    assert!(state.current.is_some());
    assert!(!state.current_page_leases.is_empty());
    assert!(!state.flat_source_bypass);
    let coarse_pages = state.current_page_leases.clone();

    settings.quality = 1.0;
    effective = state.structural.apply(&settings);
    let deferred_before = state.deferred_ordinary_publications;
    let relief = (0..256)
        .find_map(|step| {
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::sleep(std::time::Duration::from_millis(1));
            // Garden-like live motion can change quality status every frame
            // while leaving the exact globally covering physical cut stable.
            views[0].view = LodView::perspective(
                Vec3::new(if step % 2 == 0 { -0.01 } else { 0.01 }, 0.0, 0.0),
                720.0,
                1.0,
                0.1,
            );
            let (candidates, status) = update_bridge_cloud(
                &mut state,
                &effective,
                &GlobalTransform::IDENTITY,
                &views,
                &mut assets,
                &mut uploads,
            )
            .unwrap();
            assert!(!state.flat_source_bypass);
            assert!(status.active_gaussians <= settings.budgets.max_active_gaussians);
            let candidate = candidates.get(camera)?;
            if Arc::ptr_eq(&candidate.phase, &coarse_phase) {
                assert!(Arc::ptr_eq(&candidate.phase, &coarse_phase));
                assert_eq!(bridge_candidate_pages(&candidates), coarse_pages);
                assert!(Arc::ptr_eq(
                    &state.current.as_ref().unwrap().get(camera).unwrap().phase,
                    &coarse_phase
                ));
                assert_eq!(state.current_page_leases, coarse_pages);
                assert!(state.pending_page_leases.is_empty());
                None
            } else {
                assert!(
                    !candidate.frontier().is_coverage_guard(),
                    "a stable resident ancestor cut must be tried before the emergency guard"
                );
                Some(candidates)
            }
        })
        .unwrap_or_else(|| {
            panic!(
                "stable capacity pressure should publish a slot-relieving resident cut; total_frames={} stable_frames={} payload={:?} current_leases={:?} pending_leases={:?} deferred={}",
                state.capacity_pressure_total_frames,
                state.capacity_pressure_stable_frames,
                state.capacity_pressure_payload,
                state.current_page_leases,
                state.pending_page_leases,
                state.deferred_ordinary_publications,
            )
        });
    assert!(state.deferred_ordinary_publications > deferred_before);
    let relief_candidate = relief.get(camera).unwrap();
    assert!(relief_candidate.rendered_candidate_count() < source.len() as u32);
    assert!(matches!(
        relief_candidate.rendered_quality_status().degradation,
        crate::LodDegradation::Residency | crate::LodDegradation::Multiple
    ));
    let relief_phase = Arc::clone(&relief_candidate.phase);
    let relief_pages = bridge_candidate_pages(&relief);
    let releasable_pages = coarse_pages
        .difference(&relief_pages)
        .copied()
        .filter(|page| state.runtime.resident_pin_count(*page) == Some(1))
        .collect::<BTreeSet<_>>();
    assert!(!releasable_pages.is_empty());

    relief_phase.store(LOD_RENDER_PREPARED, Ordering::Release);
    let (prepared, _) = update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    assert!(Arc::ptr_eq(
        &prepared.get(camera).unwrap().phase,
        &relief_phase
    ));
    assert!(!prepared.get(camera).unwrap().frontier().is_coverage_guard());
    assert_eq!(state.current_page_leases, coarse_pages);
    assert_eq!(state.pending_page_leases, relief_pages);

    relief_phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    let (active, _) = update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    assert!(Arc::ptr_eq(
        &active.get(camera).unwrap().phase,
        &relief_phase
    ));
    assert!(!active.get(camera).unwrap().frontier().is_coverage_guard());
    assert_eq!(state.current_page_leases, relief_pages);
    assert!(state.pending_page_leases.is_empty());
    assert!(!state.flat_source_bypass);
    assert_eq!(state.capacity_pressure_total_frames, 0);

    // ACTIVE relief starts a fresh pressure epoch. A changed/newly progressing
    // frontier must not be replaced by the guard on the immediately next frame.
    for step in 0..2 {
        views[0].view = LodView::perspective(
            Vec3::new(0.02 + step as f32 * 0.01, 0.0, 0.0),
            720.0,
            1.0,
            0.1,
        );
        let (published, _) = update_bridge_cloud(
            &mut state,
            &effective,
            &GlobalTransform::IDENTITY,
            &views,
            &mut assets,
            &mut uploads,
        )
        .unwrap();
        let candidate = published.get(camera).unwrap();
        assert!(Arc::ptr_eq(&candidate.phase, &relief_phase));
        assert!(!candidate.frontier().is_coverage_guard());
    }
    for page in releasable_pages {
        assert_eq!(state.runtime.resident_pin_count(page).unwrap_or(0), 0);
    }
}

#[test]
fn degraded_guard_never_replaces_active_nonrelieving_capacity_frontier() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.frustum_culling = false;
    // One guard-pinned slot leaves no distinct old-only page for an ordinary
    // capacity cut to release. The degraded guard still must not replace the
    // valid coarse capability merely to break lease coexistence.
    settings.budgets.max_resident_pages = 1;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_pending_requests = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, atlas) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    assert_eq!(state.mirror.slot_count(), 1);
    state.transient_atlas = Some(LodTransientAtlas::new(atlas));

    let camera = Entity::from_bits(100);
    let views = [BridgeCameraView {
        entity: camera,
        view: LodView::perspective(Vec3::ZERO, 720.0, 1.0, 0.1),
    }];
    let mut assets = Assets::<PlanarGaussian3d>::default();
    let mut uploads = LodAtlasUploadQueue::default();
    let mut effective = state.structural.apply(&settings);

    let coarse = (0..64)
        .find_map(|_| {
            let (candidates, _) = update_bridge_cloud(
                &mut state,
                &effective,
                &GlobalTransform::IDENTITY,
                &views,
                &mut assets,
                &mut uploads,
            )
            .unwrap();
            candidates.get(camera).is_some().then_some(candidates)
        })
        .expect("coarse root should become resident beside the guard");
    let coarse_phase = Arc::clone(&coarse.get(camera).unwrap().phase);
    coarse_phase.store(LOD_RENDER_PREPARED, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    coarse_phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    let coarse_pages = state.current_page_leases.clone();
    assert!(!coarse_pages.is_empty());
    assert!(!state.flat_source_bypass);

    settings.quality = 1.0;
    effective = state.structural.apply(&settings);
    for _ in 0..256 {
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(1));
        let (candidates, status) = update_bridge_cloud(
            &mut state,
            &effective,
            &GlobalTransform::IDENTITY,
            &views,
            &mut assets,
            &mut uploads,
        )
        .unwrap();
        let candidate = candidates
            .get(camera)
            .expect("the ACTIVE coarse capability must remain published");
        assert_eq!(status.phase, GaussianLodBridgePhase::Active);
        assert!(!candidate.frontier().is_coverage_guard());
        assert!(Arc::ptr_eq(&candidate.phase, &coarse_phase));
        assert_eq!(bridge_candidate_pages(&candidates), coarse_pages);
        assert!(Arc::ptr_eq(
            &state.current.as_ref().unwrap().get(camera).unwrap().phase,
            &coarse_phase
        ));
        assert_eq!(state.current_page_leases, coarse_pages);
        assert!(state.pending_page_leases.is_empty());
        assert!(!state.flat_source_bypass);
    }
    assert!(state.capacity_pressure_stable_frames >= CAPACITY_PRESSURE_STABLE_FRAMES);
    assert!(state.capacity_pressure_total_frames >= CAPACITY_PRESSURE_ESCAPE_FRAMES);

    let guard_candidate = state
        .runtime
        .coverage_guard_candidate(runtime_view_id(camera), views[0].view, &effective)
        .unwrap()
        .expect("the suppressed permanent guard should remain resident");
    assert!(guard_candidate.is_coverage_guard());
    assert!(guard_candidate.quality_status().achieved_max_target_ratio > 1.0);
    assert!(!coverage_guard_frontiers_satisfy_requested_quality(&[(
        camera,
        guard_candidate,
    )]));
    let current = state
        .current
        .as_ref()
        .expect("coarse cut should stay current");
    assert!(Arc::ptr_eq(
        &current.get(camera).unwrap().phase,
        &coarse_phase
    ));
    assert_eq!(state.current_page_leases, coarse_pages);
    assert!(state.pending_page_leases.is_empty());
    assert!(!state.flat_source_bypass);
}

#[test]
fn active_budget_saturation_renders_bounded_degraded_cut_without_source() {
    let source = LodTestScene::screen_space_ladder().cloud();
    let mut settings = GaussianLodSettings::default();
    settings.quality = 1.0;
    // This is the small-scene analogue of Garden's 2M cap: a complete coarse
    // cut fits, but the exact requested cut cannot fit the active budget.
    settings.budgets.max_active_gaussians = 8;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 16 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_pending_requests = 128;
    settings.budgets.max_upload_bytes_per_frame = 16 * 1024 * 1024;
    let config = test_config();
    let (mut state, atlas) = create_ephemeral_bridge(
        Handle::default(),
        &source,
        None,
        &settings,
        &config.streaming_settings,
        &config,
        false,
    )
    .unwrap();
    state.transient_atlas = Some(LodTransientAtlas::new(atlas));

    let camera = Entity::from_bits(101);
    let views = [BridgeCameraView {
        entity: camera,
        view: LodView::perspective(Vec3::new(0.0, 0.0, 5.0), 720.0, 1.0, 0.1),
    }];
    let effective = state.structural.apply(&settings);
    let mut assets = Assets::<PlanarGaussian3d>::default();
    let mut uploads = LodAtlasUploadQueue::default();

    let candidate = (0..64)
        .find_map(|_| {
            let (candidates, status) = update_bridge_cloud(
                &mut state,
                &effective,
                &GlobalTransform::IDENTITY,
                &views,
                &mut assets,
                &mut uploads,
            )
            .unwrap();
            assert!(status.active_gaussians <= settings.budgets.max_active_gaussians);
            candidates.get(camera).cloned()
        })
        .expect("a bounded complete ancestor cut should become resident");
    assert!(
        u64::from(candidate.rendered_candidate_count()) <= settings.budgets.max_active_gaussians
    );
    assert!(matches!(
        candidate.frontier().quality_status().degradation,
        crate::LodDegradation::ActiveBudget
            | crate::LodDegradation::Multiple
            | crate::LodDegradation::Residency
    ));
    assert!(
        state.flat_source_bypass,
        "a WAITING cold candidate must leave the complete source enabled"
    );
    candidate
        .phase
        .store(LOD_RENDER_PREPARED, Ordering::Release);
    update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    assert!(!state.flat_source_bypass);
    candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
    let (_, status) = update_bridge_cloud(
        &mut state,
        &effective,
        &GlobalTransform::IDENTITY,
        &views,
        &mut assets,
        &mut uploads,
    )
    .unwrap();
    assert_eq!(status.phase, GaussianLodBridgePhase::Active);
    assert!(status.active_gaussians <= settings.budgets.max_active_gaussians);
    assert!(state.current.is_some());
    assert!(!state.flat_source_bypass);
}

#[test]
fn interior_high_cut_never_bypasses_bounded_atlas() {
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
    mark_cloud_visible(&mut world, camera, cloud);

    let mut schedule = Schedule::default();
    schedule.add_systems(update_gaussian_lod_bridges);
    let candidate = (0..128)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            world
                .get::<LodRenderCandidates>(cloud)?
                .get(camera)
                .cloned()
        })
        .unwrap_or_else(|| {
            panic!(
                "interior high-quality cut did not reach bounded atlas staging; status={:?}, count={:?}",
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
        source_handle.id(),
        "a complete WAITING high-detail cut must keep the source drawable"
    );
    assert!(
        !world
            .get::<LodRenderCandidates>(cloud)
            .unwrap()
            .candidate_draw_required
    );
    assert!(u64::from(candidate.rendered_candidate_count()) <= 4096);
    assert!(candidate.frontier().selection_view_frozen());
    assert_eq!(
        candidate.frontier().quality_status().requested_target,
        world
            .get::<GaussianLodSettings>(cloud)
            .unwrap()
            .quality_target()
    );
    assert!(world.resource::<GaussianLodBridgeManager>().clouds[&cloud].flat_source_bypass);
    candidate
        .phase
        .store(LOD_RENDER_PREPARED, Ordering::Release);
    run_bridge_frame(&mut world, &mut schedule);
    let state = &world.resource::<GaussianLodBridgeManager>().clouds[&cloud];
    assert!(!state.flat_source_bypass);
    assert!(!state.active);
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        state.atlas.id()
    );

    let mut cloud_settings = CloudSettings::default();
    cloud_settings
        .lod_debug
        .apply_preset(crate::LodDebugPreset::Level);
    world.entity_mut(cloud).insert(cloud_settings);
    let debug_candidate = (0..128)
        .find_map(|_| {
            run_bridge_frame(&mut world, &mut schedule);
            world
                .get::<LodRenderCandidates>(cloud)?
                .get(camera)
                .cloned()
        })
        .expect("debug annotations keep the same bounded atlas compaction path");
    assert_eq!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id(),
        "debug rebuild also retains the source through WAITING"
    );
    assert!(
        !world
            .get::<LodRenderCandidates>(cloud)
            .unwrap()
            .candidate_draw_required
    );
    debug_candidate
        .phase
        .store(LOD_RENDER_PREPARED, Ordering::Release);
    run_bridge_frame(&mut world, &mut schedule);
    assert_ne!(
        world
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .handle()
            .id(),
        source_handle.id()
    );
    assert!(
        world
            .get::<LodRenderCandidates>(cloud)
            .unwrap()
            .candidate_draw_required
    );
    assert!(!world.resource::<GaussianLodBridgeManager>().clouds[&cloud].flat_source_bypass);
}
