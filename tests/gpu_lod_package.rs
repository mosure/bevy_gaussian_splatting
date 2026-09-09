//! Bounded native page transport -> GPU hierarchy -> complete point image smoke,
//! including two independent package atlases sharing one view projection budget.
//! This is a tiny integration/lifetime contract, not a scene-scale benchmark.
#![cfg(all(
    feature = "headless",
    feature = "testing",
    lod_render_path,
    not(target_arch = "wasm32")
))]

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::{RenderTarget, visibility::NoFrustumCulling},
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{
            BufferId, CachedPipelineState, PipelineCache, TextureFormat, WgpuFeatures,
        },
        renderer::RenderDevice,
        view::{
            ExtractedView,
            screenshot::{Screenshot, ScreenshotCaptured},
        },
    },
    shader::ShaderCacheError,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, GaussianCamera, GaussianLodBuildSettings, GaussianLodHandle,
    GaussianLodSettings, GaussianSplattingPlugin, LodPresentationMode, PlanarGaussian3d,
    gaussian::{
        f32::Rotation,
        formats::{planar_3d_chunked::LodPageStorage, planar_3d_lod::build_planar_3d_lod},
    },
    io::lod::{GaussianLodAsset, encode_page},
    render::{
        point::{
            GaussianPointSplattingDiagnostics, GaussianPointSplattingSettings,
            GaussianPointSplattingViewBudget, GaussianPointSplattingViewBudgetDiagnostics,
        },
        traversal::{
            GpuLodDrawAcknowledgements, GpuLodHierarchy, GpuLodHierarchySnapshot,
            GpuLodTraversalOutputs, GpuLodTraversalSettings,
        },
    },
    stream::{
        atlas_upload::LodAtlasUploadBudget,
        bridge::GaussianLodBridgeConfig,
        memory::{LodMemoryLedger, LodMemorySnapshot},
        package::{
            GaussianGpuLodPackage, GaussianGpuLodPackageStatus, GaussianLodPackageConfig,
            GaussianLodPackageSource, GaussianLodPackageStatus,
        },
    },
};
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

struct TempPackage(PathBuf);
impl Drop for TempPackage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "requires an explicitly requested wgpu adapter"]
fn native_page_demand_reaches_a_generation_acknowledged_point_image() {
    if std::env::var("RUN_GPU_RENDER_TESTS").as_deref() != Ok("1") {
        return;
    }
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = TempPackage(std::env::temp_dir().join(format!(
        "bgs_gpu_package_{}_{}",
        std::process::id(),
        nonce
    )));
    fs::create_dir_all(&directory.0).unwrap();
    let source = PlanarGaussian3d::from(
        (0..32)
            .map(|i| Gaussian3d {
                position_visibility: [
                    ((i % 4) as f32 - 1.5) * 0.15,
                    (((i / 4) % 4) as f32 - 1.5) * 0.15,
                    ((i / 16) as f32 - 0.5) * 0.2,
                    1.0,
                ]
                .into(),
                rotation: Rotation {
                    rotation: [1.0, 0.0, 0.0, 0.0],
                },
                scale_opacity: [0.025, 0.02, 0.015, 0.65].into(),
                spherical_harmonic: default(),
            })
            .collect::<Vec<_>>(),
    );
    let mut lod = build_planar_3d_lod(
        &source,
        GaussianLodBuildSettings {
            leaf_capacity: 4,
            ..default()
        },
    )
    .unwrap();
    assert!(lod.manifest.quality.max_depth >= 2);
    for (descriptor, page) in lod.manifest.pages.iter_mut().zip(&lod.pages) {
        let encoded = encode_page(page).unwrap();
        let filename = format!("{}.gspage", page.id.0);
        fs::write(directory.0.join(&filename), &encoded).unwrap();
        descriptor.storage = Some(LodPageStorage {
            uri: filename,
            byte_range: None,
            encoded_len: encoded.len() as u64,
        });
    }
    lod.validate().unwrap();
    let coarse_count = lod.manifest.quality.coarsest_gaussian_count as u32;
    let mut app = App::new();
    let timing_support = TimingSupport::default();
    let traversal_frames = TraversalFrames::default();
    app.insert_resource(ClearColor(Color::BLACK))
        .insert_resource(timing_support.clone())
        .insert_resource(traversal_frames.clone())
        .insert_resource(GaussianLodBridgeConfig {
            auto_build_flat_clouds: false,
            ..default()
        })
        .insert_resource(GaussianLodPackageConfig {
            max_atlas_gaussians: 256,
            max_atlas_bytes: 4 * 1024 * 1024,
            ..default()
        })
        .insert_resource(State {
            asset: Some(GaussianLodAsset::new(lod.manifest).unwrap()),
            root: directory.0.to_string_lossy().into_owned(),
            started: Instant::now(),
            camera: None,
            cloud: None,
            second_cloud: None,
            target: None,
            coarse_count,
            coarse_generation: None,
            coarse_workspace: None,
            held_snapshots: Vec::new(),
            observed_pending_publication: false,
            exact_acknowledged: false,
            multiple_acknowledged: false,
            requested: false,
            lifecycle: Lifecycle::PublicationBaseline,
            phase_frames: 0,
            last_submission: 0,
            baseline_image: Vec::new(),
            initial_memory: None,
            retiring: None,
        })
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    meta_check: bevy::asset::AssetMetaCheck::Never,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>()
                .disable::<PipelinedRenderingPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 120.0,
        )))
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, setup)
        .add_systems(First, restore_staging_budget)
        .add_systems(Last, throttle_extracted_uploads)
        .add_systems(Update, (await_image, await_cleanup))
        .add_observer(captured);
    app.sub_app_mut(RenderApp)
        .insert_resource(timing_support)
        .insert_resource(traversal_frames)
        .add_systems(Render, observe_traversal.in_set(RenderSystems::Cleanup))
        .add_systems(Render, fail_shader_errors.in_set(RenderSystems::Cleanup));
    assert_eq!(app.run(), AppExit::Success);
}

#[derive(Resource)]
struct State {
    asset: Option<GaussianLodAsset>,
    root: String,
    started: Instant,
    camera: Option<Entity>,
    cloud: Option<Entity>,
    second_cloud: Option<Entity>,
    target: Option<Handle<Image>>,
    coarse_count: u32,
    coarse_generation: Option<u64>,
    coarse_workspace: Option<TraversalFrame>,
    held_snapshots: Vec<Arc<GpuLodHierarchySnapshot>>,
    observed_pending_publication: bool,
    exact_acknowledged: bool,
    multiple_acknowledged: bool,
    requested: bool,
    lifecycle: Lifecycle,
    phase_frames: u32,
    last_submission: u64,
    baseline_image: Vec<u8>,
    initial_memory: Option<LodMemorySnapshot>,
    retiring: Option<RetiringResources>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Lifecycle {
    PublicationBaseline,
    PublicationMotion,
    PublicationFence,
    Baseline,
    Away,
    Revisit,
    Budget,
}

struct RetiringResources {
    snapshots: Vec<Weak<GpuLodHierarchySnapshot>>,
    atlases: Vec<AssetId<PlanarGaussian3d>>,
}

#[derive(Resource, Clone, Default)]
struct TimingSupport(Arc<AtomicBool>);

#[derive(Clone, Copy, Debug)]
struct TraversalFrame {
    allocation: u64,
    residency: u64,
    desired_residency: u64,
    submission: u64,
    ready: bool,
    buffers: [BufferId; 3],
}

#[derive(Resource, Clone, Default)]
struct TraversalFrames(Arc<Mutex<HashMap<(Entity, Entity), TraversalFrame>>>);

fn observe_traversal(
    frames: Res<TraversalFrames>,
    outputs: Res<GpuLodTraversalOutputs>,
    views: Query<&ExtractedView>,
    clouds: Query<(Entity, &GpuLodHierarchy)>,
) {
    let mut frames = frames.0.lock().unwrap();
    for view in &views {
        for (cloud, hierarchy) in &clouds {
            if let Some(output) = outputs.get(view.retained_view_entity, cloud) {
                frames.insert(
                    (
                        view.retained_view_entity.main_entity.id(),
                        output.main_cloud,
                    ),
                    TraversalFrame {
                        allocation: output.generation,
                        residency: output.residency_generation,
                        desired_residency: hierarchy.0.generation,
                        submission: output.submission,
                        ready: output.is_ready(),
                        buffers: [
                            output.entries.id(),
                            output.indirect.id(),
                            output.feedback.id(),
                        ],
                    },
                );
            }
        }
    }
}

// Main-world staging sees a valid budget; only subsequent extraction is
// throttled. The fixture disables pipelined rendering, so First restores it
// before the next staging update, while deferred payloads remain queued.
fn restore_staging_budget(state: Res<State>, mut budget: ResMut<LodAtlasUploadBudget>) {
    *budget = LodAtlasUploadBudget::default();
    if matches!(
        state.lifecycle,
        Lifecycle::PublicationMotion | Lifecycle::PublicationFence
    ) {
        // One published slot per frame makes four distinct generations
        // observable even when several worker completions arrive together.
        budget.set_max_slots_per_frame(1).unwrap();
    }
}

fn throttle_extracted_uploads(state: Res<State>, mut budget: ResMut<LodAtlasUploadBudget>) {
    if state.lifecycle == Lifecycle::PublicationMotion {
        budget.set_max_canonical_bytes_per_frame(1).unwrap();
    }
}

fn setup(
    mut commands: Commands,
    mut state: ResMut<State>,
    mut manifests: ResMut<Assets<GaussianLodAsset>>,
    mut images: ResMut<Assets<Image>>,
    ledger: Res<LodMemoryLedger>,
) {
    state.initial_memory = Some(ledger.snapshot());
    let manifest = manifests.add(state.asset.take().unwrap());
    let mut quality = GaussianLodSettings {
        quality: 0.0,
        presentation_mode: LodPresentationMode::Discrete,
        ..default()
    };
    quality.budgets.max_resident_gaussians = 256;
    quality.budgets.max_resident_pages = 64;
    quality.budgets.max_requests_per_frame = 1;
    state.cloud = Some(
        commands
            .spawn((
                GaussianLodHandle(manifest),
                GaussianLodPackageSource::native_directory(state.root.clone()),
                GaussianGpuLodPackage,
                quality,
                CloudSettings {
                    opacity_adaptive_radius: false,
                    ..default()
                },
                Transform::from_rotation(Quat::from_rotation_y(0.2)),
                Visibility::Visible,
                NoFrustumCulling,
            ))
            .id(),
    );
    let target = images.add(Image::new_target_texture(
        64,
        64,
        TextureFormat::Rgba8UnormSrgb,
        None,
    ));
    state.target = Some(target.clone());
    state.camera = Some(
        commands
            .spawn((
                Camera3d::default(),
                Camera::default(),
                RenderTarget::Image(target.into()),
                Transform::from_xyz(0.0, 0.0, 2.0),
                Tonemapping::None,
                Msaa::Off,
                GaussianCamera::default(),
                GpuLodTraversalSettings {
                    max_selected_gaussians: 64,
                    max_frontier_nodes: 64,
                    max_visited_nodes: 256,
                    max_page_requests: 16,
                    max_gpu_bytes: 1024 * 1024,
                },
                GaussianPointSplattingSettings {
                    samples_per_pixel: 1,
                    max_projected_gaussians: 64,
                    max_points_per_frame: 65_536,
                    max_gpu_bytes: 4 * 1024 * 1024,
                    temporal_sampling: false,
                    ..default()
                },
            ))
            .id(),
    );
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn await_image(
    mut commands: Commands,
    mut state: ResMut<State>,
    diagnostics: Res<GaussianPointSplattingDiagnostics>,
    acknowledgements: Res<GpuLodDrawAcknowledgements>,
    timing: Res<GaussianPointSplattingViewBudgetDiagnostics>,
    traversal_frames: Res<TraversalFrames>,
    mut cameras: Query<(
        &mut GpuLodTraversalSettings,
        &mut GaussianPointSplattingSettings,
    )>,
    mut clouds: Query<(
        &mut GaussianLodSettings,
        Option<&GpuLodHierarchy>,
        Option<&GaussianLodPackageStatus>,
        &GaussianLodHandle,
        Option<&GaussianGpuLodPackageStatus>,
    )>,
) {
    if state.retiring.is_some() {
        return;
    }
    state.phase_frames += 1;
    if state.started.elapsed() >= Duration::from_secs(45) {
        let package = state.cloud.and_then(|cloud| clouds.get(cloud).ok()).map(
            |(_, hierarchy, package, _, gpu)| {
                (
                    hierarchy.map(|h| h.0.generation),
                    package.cloned(),
                    gpu.cloned(),
                )
            },
        );
        let frame = state.camera.and_then(|camera| diagnostics.get(camera));
        let ack = state
            .camera
            .zip(state.cloud)
            .and_then(|(camera, cloud)| acknowledgements.get(camera, cloud));
        let traversal = state
            .camera
            .zip(state.cloud)
            .and_then(|key| traversal_frames.0.lock().unwrap().get(&key).copied());
        panic!(
            "native GPU package timed out: phase={:?} phase_frames={} held={:?} pending_seen={} package={package:?} point={frame:?} ack={ack:?} traversal={traversal:?}",
            state.lifecycle,
            state.phase_frames,
            state
                .held_snapshots
                .iter()
                .map(|s| s.generation)
                .collect::<Vec<_>>(),
            state.observed_pending_publication
        );
    }
    let (Some(camera), Some(cloud)) = (state.camera, state.cloud) else {
        return;
    };
    let (quality, hierarchy, package, manifest, gpu_package) = clouds.get_mut(cloud).unwrap();
    if let Some(package) = package {
        assert!(
            package.failure.is_none(),
            "package failed: {:?}",
            package.failure
        );
    }
    let (Some(hierarchy), Some(ack), Some(frame)) = (
        hierarchy,
        acknowledgements.get(camera, cloud),
        diagnostics.get(camera),
    ) else {
        return;
    };
    let Some(observed) = traversal_frames
        .0
        .lock()
        .unwrap()
        .get(&(camera, cloud))
        .copied()
    else {
        return;
    };
    if matches!(
        state.lifecycle,
        Lifecycle::PublicationMotion | Lifecycle::PublicationFence
    ) {
        // Keep four immutable generations alive, allowing extraction after the
        // initial camera-motion image so deeper cohorts can become requested.
        if state.held_snapshots.len() < 4
            && state
                .held_snapshots
                .last()
                .is_none_or(|held| held.generation != hierarchy.0.generation)
        {
            state.held_snapshots.push(hierarchy.0.clone());
        }
        if state.held_snapshots.len() == 4
            && gpu_package.is_some_and(|status| status.pending_publication_pages > 0)
        {
            state.observed_pending_publication = true;
        }
        if state.lifecycle == Lifecycle::PublicationFence {
            assert!(
                observed.ready,
                "snapshot fence hid the complete resident cut"
            );
            if state.observed_pending_publication {
                state.held_snapshots.clear();
                state.lifecycle = Lifecycle::Baseline;
                state.phase_frames = 0;
            }
            return;
        }
        let coarse = state.coarse_workspace.unwrap();
        if observed.desired_residency <= coarse.residency {
            return;
        }
        assert!(
            observed.ready,
            "pending uploads hid a complete resident snapshot"
        );
        assert_eq!(observed.residency, coarse.residency);
        assert_eq!(observed.allocation, coarse.allocation);
        assert_eq!(observed.buffers, coarse.buffers);
        if !state.requested
            && state.phase_frames >= 12
            && observed.submission > coarse.submission + 2
        {
            assert_eq!(ack.residency_generation, coarse.residency);
            state.requested = true;
            commands.spawn(Screenshot::image(state.target.clone().unwrap()));
        }
        return;
    }
    if ack.residency_generation != hierarchy.0.generation
        || ack.source != hierarchy.0.source
        || ack.submission != frame.submission
        || frame.submission <= state.last_submission
    {
        return;
    }
    assert!(
        !frame.traversal_failed && !frame.sampling_failed && !frame.overflow,
        "point frame failed: {frame:?}"
    );
    if state.coarse_generation.is_none() {
        if frame.projected_gaussians != state.coarse_count {
            return;
        }
        if observed.residency != ack.residency_generation {
            return;
        }
        state.coarse_generation = Some(ack.residency_generation);
        state.coarse_workspace = Some(observed);
        state.requested = true;
        commands.spawn(Screenshot::image(state.target.clone().unwrap()));
    } else if !state.exact_acknowledged && frame.projected_gaussians == 32 {
        assert!(
            ack.residency_generation > state.coarse_generation.unwrap(),
            "refinement did not publish newly demanded page placements"
        );
        if observed.residency != ack.residency_generation {
            return;
        }
        let coarse = state.coarse_workspace.unwrap();
        assert_eq!(
            observed.allocation, coarse.allocation,
            "residency upgrade recreated traversal workspace"
        );
        assert_eq!(
            observed.buffers, coarse.buffers,
            "residency upgrade replaced traversal buffers"
        );
        state.exact_acknowledged = true;
        let (mut traversal, mut point) = cameras.get_mut(camera).unwrap();
        traversal.max_selected_gaussians = 32;
        point.max_projected_gaussians = 32;
        state.second_cloud = Some(
            commands
                .spawn((
                    manifest.clone(),
                    GaussianLodPackageSource::native_directory(state.root.clone()),
                    GaussianGpuLodPackage,
                    quality.clone(),
                    CloudSettings {
                        opacity_adaptive_radius: false,
                        ..default()
                    },
                    Transform::from_rotation(Quat::from_rotation_y(0.2)),
                    Visibility::Visible,
                    NoFrustumCulling,
                ))
                .id(),
        );
    } else if state.exact_acknowledged && !state.requested {
        let (_, second_hierarchy, second_package, _, _) =
            clouds.get_mut(state.second_cloud.unwrap()).unwrap();
        if let Some(package) = second_package {
            assert!(
                package.failure.is_none(),
                "second package failed: {:?}",
                package.failure
            );
        }
        let (Some(second_hierarchy), Some(second_ack)) = (
            second_hierarchy,
            acknowledgements.get(camera, state.second_cloud.unwrap()),
        ) else {
            return;
        };
        if second_ack.residency_generation != second_hierarchy.0.generation
            || second_ack.source != second_hierarchy.0.source
            || second_ack.submission != frame.submission
        {
            return;
        }
        assert_ne!(
            ack.source, second_ack.source,
            "package instances must own separate resident atlases"
        );
        assert!(
            (state.coarse_count * 2..=32).contains(&frame.projected_gaussians),
            "both root covers must fit one shared projection budget: {frame:?}"
        );
        if state.multiple_acknowledged && state.lifecycle == Lifecycle::Budget {
            if let Some(timing) = timing.get(camera) {
                assert!(
                    timing.error.is_none(),
                    "outer budget timing failed: {timing:?}"
                );
                if let Some(ms) = timing.view_gpu_ms {
                    assert!(ms.is_finite() && ms >= 0.0);
                    assert_eq!(timing.samples_per_pixel, 1);
                    let cap = cameras.get(camera).unwrap().0.max_selected_gaussians;
                    assert!(cap >= state.coarse_count * 2);
                    if timing.selected_gaussian_limit < 32 && cap < 32 {
                        state.last_submission = frame.submission;
                        state.requested = true;
                        commands.spawn(Screenshot::image(state.target.clone().unwrap()));
                    }
                }
            }
            return;
        }
        state.multiple_acknowledged = true;
        if state.phase_frames >= 12 {
            state.last_submission = frame.submission;
            state.requested = true;
            commands.spawn(Screenshot::image(state.target.clone().unwrap()));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn captured(
    event: On<ScreenshotCaptured>,
    mut commands: Commands,
    mut state: ResMut<State>,
    mut cameras: Query<&mut Transform, With<GaussianCamera>>,
    hierarchies: Query<&GpuLodHierarchy>,
    timing_support: Res<TimingSupport>,
    mut qualities: Query<&mut GaussianLodSettings>,
) {
    let data = event.image.data.as_ref().unwrap();
    assert!(
        data.chunks_exact(4)
            .any(|pixel| pixel[..3].iter().any(|channel| *channel != 0)),
        "acknowledged point image is empty"
    );
    let camera = state.camera.unwrap();
    match state.lifecycle {
        Lifecycle::PublicationBaseline => {
            state.baseline_image = data.clone();
            let snapshot = hierarchies.get(state.cloud.unwrap()).unwrap().0.clone();
            state.held_snapshots.push(snapshot);
            // Last will defer extraction of new slots after normal staging.
            qualities.get_mut(state.cloud.unwrap()).unwrap().quality = 1.0;
            cameras.get_mut(camera).unwrap().translation.x = 0.3;
            state.lifecycle = Lifecycle::PublicationMotion;
        }
        Lifecycle::PublicationMotion => {
            assert_ne!(
                data, &state.baseline_image,
                "pending atlas uploads retained the old camera image"
            );
            cameras.get_mut(camera).unwrap().translation.x = 0.0;
            state.lifecycle = Lifecycle::PublicationFence;
        }
        Lifecycle::PublicationFence => unreachable!("snapshot fence does not request a screenshot"),
        Lifecycle::Baseline => {
            state.baseline_image = data.clone();
            cameras.get_mut(camera).unwrap().translation.x = 0.3;
            state.lifecycle = Lifecycle::Away;
        }
        Lifecycle::Away => {
            assert_ne!(
                data, &state.baseline_image,
                "camera motion retained the previous package image"
            );
            cameras.get_mut(camera).unwrap().translation.x = 0.0;
            state.lifecycle = Lifecycle::Revisit;
        }
        Lifecycle::Revisit if timing_support.0.load(Ordering::Relaxed) => {
            commands
                .entity(camera)
                .insert(GaussianPointSplattingViewBudget {
                    target_view_gpu_ms: 0.000_001,
                    min_selected_gaussians: 1,
                    max_selected_gaussians: 32,
                });
            state.lifecycle = Lifecycle::Budget;
        }
        Lifecycle::Revisit | Lifecycle::Budget => {
            if state.lifecycle == Lifecycle::Revisit {
                eprintln!(
                    "GPU timestamp queries unavailable; package image/ACK, shared cap, motion and revisit verified; timing assertion skipped"
                );
            }
            let mut retiring = RetiringResources {
                snapshots: Vec::new(),
                atlases: Vec::new(),
            };
            for cloud in [state.cloud.unwrap(), state.second_cloud.unwrap()] {
                let hierarchy = hierarchies.get(cloud).unwrap();
                retiring.snapshots.push(Arc::downgrade(&hierarchy.0));
                retiring.atlases.push(hierarchy.0.source);
                commands.entity(cloud).despawn();
            }
            commands.entity(camera).despawn();
            state.retiring = Some(retiring);
        }
    }
    state.requested = false;
    state.phase_frames = 0;
}

fn await_cleanup(
    state: Res<State>,
    ledger: Res<LodMemoryLedger>,
    atlases: Res<Assets<PlanarGaussian3d>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(retiring) = &state.retiring else {
        return;
    };
    let memory = ledger.snapshot();
    assert!(
        state.started.elapsed() < Duration::from_secs(45),
        "GPU package resources did not retire after unload: {memory:?}"
    );
    let initial = state.initial_memory.unwrap();
    if retiring
        .snapshots
        .iter()
        .any(|snapshot| snapshot.strong_count() != 0)
        || retiring
            .atlases
            .iter()
            .any(|atlas| atlases.contains(*atlas))
        || memory.total_bytes != initial.total_bytes
        || memory.allocations != initial.allocations
    {
        return;
    }
    exit.write(AppExit::Success);
}

fn fail_shader_errors(
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    timing: Res<TimingSupport>,
) {
    timing.0.store(
        device.features().contains(WgpuFeatures::TIMESTAMP_QUERY),
        Ordering::Relaxed,
    );
    for pipeline in cache.pipelines() {
        if let CachedPipelineState::Err(error) = &pipeline.state {
            match error {
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable => {}
                _ => panic!("GPU package shader failed: {error}"),
            }
        }
    }
}
