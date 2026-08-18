#[cfg(not(all(feature = "headless", feature = "testing")))]
#[test]
fn lod_quality_render_test_requires_headless_and_testing_features() {}

#[cfg(all(feature = "headless", feature = "testing"))]
mod headless {
    use std::{
        env,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use bevy::{
        app::{AppExit, ScheduleRunnerPlugin},
        camera::{PerspectiveProjection, Projection, RenderTarget},
        core_pipeline::tonemapping::Tonemapping,
        prelude::*,
        render::{
            Render, RenderApp, RenderSystems,
            extract_resource::{ExtractResource, ExtractResourcePlugin},
            render_resource::TextureFormat,
            renderer::{RenderDevice, RenderQueue},
            view::ExtractedView,
            view::screenshot::{Screenshot, ScreenshotCaptured},
        },
        window::ExitCondition,
        winit::WinitPlugin,
    };
    use bevy_gaussian_splatting::{
        CloudSettings, Gaussian3d, GaussianCamera, GaussianLodBridgeConfig,
        GaussianLodBuildSettings, GaussianLodSettings, GaussianMode, GaussianSplattingPlugin,
        PlanarGaussian3d, PlanarGaussian3dHandle, PlanarHandle, RadixSortDepthBits,
        render::{
            ShaderDefines,
            lod::{
                LodCompactionBuffers, LodIndirectArgs, finalized_indirect_args,
                read_lod_indirect_args_for_testing,
            },
        },
        sort::SortMode,
        stream::{
            bridge::{GaussianLodBridgePhase, GaussianLodBridgeStatus},
            hierarchy::{AllResident, LodView, ManifestLodHierarchy, select_frontier},
            render_commit::LodRenderCandidates,
        },
        testing::{ImageMetrics, LodTestScene, compare_linear_rgba},
    };

    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 128;
    const VERTICAL_FOV: f32 = 60.0_f32.to_radians();
    const NEAR_PLANE: f32 = 0.01;
    const FAR_PLANE: f32 = 10_000.0;
    const NEAR_CAMERA_Z: f32 = 8.0;
    const FAR_CAMERA_Z: f32 = 32.0;
    const DISTANT_CAMERA_Z: f32 = 1_024.0;
    const QUALITIES: [f32; 4] = [0.0, 0.25, 0.50, 0.60];
    const CAMERA_PROBE_QUALITY: f32 = 0.50;
    const REFERENCE_WARMUP_FRAMES: u32 = 45;
    const RESTORED_WARMUP_FRAMES: u32 = 18;
    const STABLE_ACTIVE_FRAMES: u32 = 6;
    const MAX_FRAMES: u32 = 960;

    #[test]
    fn structured_fixture_has_strict_quality_and_perspective_cuts() {
        let fixture = quality_fixture();
        eprintln!(
            "LoD fixture counts: near={:?}, near_probe={}, far_probe={}, distant_probe={}, source={}",
            fixture.expected_near,
            fixture.expected_near_probe,
            fixture.expected_far_probe,
            fixture.expected_distant_probe,
            fixture.source_count
        );
        assert_eq!(fixture.expected_near[0], fixture.coarsest_count);
        assert_eq!(fixture.source_count, 320);
        assert_eq!(
            fixture.expected_near,
            [1, 3, 43, 198],
            "production-default quality samples drifted from their calibrated cuts"
        );
        assert_eq!(
            (
                fixture.expected_near_probe,
                fixture.expected_far_probe,
                fixture.expected_distant_probe,
            ),
            (43, 2, 1),
            "production-default perspective response drifted"
        );
        assert_ne!(
            fixture.expected_near_probe, fixture.expected_near[3],
            "camera-probe and upper quality cuts must differ"
        );
        assert!(
            fixture
                .expected_near
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "quality samples must exercise distinct cuts: {:?}",
            fixture.expected_near
        );
        assert!(
            fixture.expected_far_probe < fixture.expected_near_probe,
            "perspective probe must coarsen with distance: near={}, far={}",
            fixture.expected_near_probe,
            fixture.expected_far_probe
        );
        assert!(
            fixture.expected_distant_probe > 0
                && fixture.expected_distant_probe <= fixture.expected_far_probe,
            "distant perspective probe must retain a nonempty cut no finer than the far probe: far={}, distant={}",
            fixture.expected_far_probe,
            fixture.expected_distant_probe
        );
        assert!(fixture.expected_near.last().copied().unwrap() < fixture.source_count);
    }

    #[test]
    fn automatic_bridge_renders_camera_aware_quality_sweep() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping GPU LoD quality render test; set RUN_GPU_RENDER_TESTS=1 to enable");
            return;
        }

        let fixture = quality_fixture();
        let mut app = App::new();
        let indirect_probe = IndirectProbe::default();
        app.insert_resource(ClearColor(Color::BLACK))
            .insert_resource(quality_bridge_config(fixture.build_settings))
            .insert_resource(indirect_probe)
            .insert_resource(QualityRenderState::new(&fixture));
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: "assets".to_string(),
                    processed_file_path: "assets".to_string(),
                    meta_check: bevy::asset::AssetMetaCheck::Never,
                    unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow,
                    ..default()
                })
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>()
                .disable::<bevy::log::LogPlugin>(),
        );
        app.add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )));
        app.add_plugins((
            GaussianSplattingPlugin,
            ExtractResourcePlugin::<IndirectProbe>::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, drive_capture)
        .add_observer(on_capture);
        app.sub_app_mut(RenderApp).add_systems(
            Render,
            read_ready_lod_indirect_args.in_set(RenderSystems::Cleanup),
        );
        app.run();
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct IndirectObservation {
        candidate_count: u32,
        args: LodIndirectArgs,
        expected: LodIndirectArgs,
    }

    #[derive(Debug, Default)]
    struct IndirectProbeState {
        requested_candidate_count: Option<u32>,
        observation: Option<IndirectObservation>,
        error: Option<String>,
    }

    #[derive(Resource, Clone, ExtractResource, Default)]
    struct IndirectProbe(Arc<Mutex<IndirectProbeState>>);

    impl IndirectProbe {
        fn request(&self, candidate_count: u32) -> bool {
            let mut state = self.0.lock().expect("indirect probe mutex is not poisoned");
            if state.requested_candidate_count != Some(candidate_count) {
                state.requested_candidate_count = Some(candidate_count);
                state.observation = None;
                state.error = None;
            }
            if let Some(error) = &state.error {
                panic!("GPU indirect readback failed: {error}");
            }
            state.observation.is_some_and(|observation| {
                assert_eq!(observation.candidate_count, candidate_count);
                assert_eq!(observation.args, observation.expected);
                true
            })
        }
    }

    fn read_ready_lod_indirect_args(
        render_device: Res<RenderDevice>,
        render_queue: Res<RenderQueue>,
        buffers: Res<LodCompactionBuffers<Gaussian3d>>,
        views: Query<&ExtractedView, With<GaussianCamera>>,
        clouds: Query<(Entity, &PlanarGaussian3dHandle)>,
        probe: Res<IndirectProbe>,
    ) {
        let requested = probe
            .0
            .lock()
            .expect("indirect probe mutex is not poisoned")
            .requested_candidate_count;
        let Some(requested) = requested else {
            return;
        };
        let defines = ShaderDefines::for_radix_depth_bits(RadixSortDepthBits::Bits32);
        for view in &views {
            for (entity, handle) in &clouds {
                let Some(state) =
                    buffers.get_ready(view.retained_view_entity, entity, handle.handle().id())
                else {
                    continue;
                };
                if state.candidate_count() != requested {
                    continue;
                }
                match read_lod_indirect_args_for_testing(&render_device, &render_queue, state) {
                    Ok(args) => {
                        let expected = finalized_indirect_args(
                            requested,
                            state.output_capacity(),
                            defines.workgroup_entries_a,
                            defines.workgroup_entries_c,
                        );
                        let mut probe = probe
                            .0
                            .lock()
                            .expect("indirect probe mutex is not poisoned");
                        probe.observation = Some(IndirectObservation {
                            candidate_count: requested,
                            args,
                            expected,
                        });
                        probe.error = None;
                    }
                    Err(error) => {
                        probe
                            .0
                            .lock()
                            .expect("indirect probe mutex is not poisoned")
                            .error = Some(error.to_string());
                    }
                }
                return;
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Phase {
        ReferenceWarmup,
        ReferencePending,
        LowerQualityWaiting,
        LowerQualityPending,
        NearProbeWaiting,
        FarProbeWaiting,
        DistantProbeWaiting,
        NearRecoveryWaiting,
        RestoredReferenceWaiting,
        RestoredReferencePending,
    }

    #[derive(Resource)]
    struct QualityRenderState {
        phase: Phase,
        phase_frames: u32,
        total_frames: u32,
        stable_active_frames: u32,
        sample_index: usize,
        source_count: u64,
        coarsest_count: u64,
        expected_near: Vec<u64>,
        expected_near_probe: u64,
        expected_far_probe: u64,
        expected_distant_probe: u64,
        target: Option<Handle<Image>>,
        cloud: Option<Entity>,
        camera: Option<Entity>,
        source_handle: Option<Handle<PlanarGaussian3d>>,
        reference: Option<Vec<[f32; 4]>>,
        captures: Vec<QualityCapture>,
    }

    impl QualityRenderState {
        fn new(fixture: &QualityFixture) -> Self {
            Self {
                phase: Phase::ReferenceWarmup,
                phase_frames: 0,
                total_frames: 0,
                stable_active_frames: 0,
                sample_index: 0,
                source_count: fixture.source_count,
                coarsest_count: fixture.coarsest_count,
                expected_near: fixture.expected_near.clone(),
                expected_near_probe: fixture.expected_near_probe,
                expected_far_probe: fixture.expected_far_probe,
                expected_distant_probe: fixture.expected_distant_probe,
                target: None,
                cloud: None,
                camera: None,
                source_handle: None,
                reference: None,
                captures: Vec::with_capacity(QUALITIES.len()),
            }
        }

        fn enter(&mut self, phase: Phase) {
            self.phase = phase;
            self.phase_frames = 0;
            self.stable_active_frames = 0;
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct QualityCapture {
        quality: f32,
        active_gaussians: u64,
        metrics: ImageMetrics,
    }

    struct QualityFixture {
        cloud: PlanarGaussian3d,
        settings: GaussianLodSettings,
        build_settings: GaussianLodBuildSettings,
        source_count: u64,
        coarsest_count: u64,
        expected_near: Vec<u64>,
        expected_near_probe: u64,
        expected_far_probe: u64,
        expected_distant_probe: u64,
    }

    fn quality_fixture() -> QualityFixture {
        let cloud = LodTestScene::checkerboard_facade(20, 16).cloud();
        let build_settings = GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 8,
            support_sigma: 3.0,
        };
        let mut settings = GaussianLodSettings {
            quality: 1.0,
            hysteresis: 0.0,
            frustum_culling: false,
            ..default()
        };
        settings.budgets.max_active_gaussians = 4_096;
        settings.budgets.max_resident_gaussians = 8_192;
        settings.budgets.max_resident_bytes = 32 * 1024 * 1024;
        settings.budgets.max_resident_pages = 256;
        settings.budgets.max_pending_requests = 512;
        settings.budgets.max_requests_per_frame = 256;
        settings.budgets.max_upload_bytes_per_frame = 32 * 1024 * 1024;

        let built = bevy_gaussian_splatting::build_planar_3d_lod(&cloud, build_settings)
            .expect("quality fixture hierarchy builds");
        let hierarchy =
            ManifestLodHierarchy::new(&built.manifest).expect("quality fixture manifest is valid");
        let expected_near = QUALITIES
            .iter()
            .map(|&quality| {
                expected_count(
                    &hierarchy,
                    &settings,
                    quality,
                    Vec3::new(0.0, 0.0, NEAR_CAMERA_Z),
                )
            })
            .collect::<Vec<_>>();
        let expected_near_probe = expected_count(
            &hierarchy,
            &settings,
            CAMERA_PROBE_QUALITY,
            Vec3::new(0.0, 0.0, NEAR_CAMERA_Z),
        );
        let expected_far_probe = expected_count(
            &hierarchy,
            &settings,
            CAMERA_PROBE_QUALITY,
            Vec3::new(0.0, 0.0, FAR_CAMERA_Z),
        );
        let expected_distant_probe = expected_count(
            &hierarchy,
            &settings,
            CAMERA_PROBE_QUALITY,
            Vec3::new(0.0, 0.0, DISTANT_CAMERA_Z),
        );

        QualityFixture {
            source_count: built.manifest.header.source_gaussian_count,
            coarsest_count: built.manifest.quality.coarsest_gaussian_count,
            cloud,
            settings,
            build_settings,
            expected_near,
            expected_near_probe,
            expected_far_probe,
            expected_distant_probe,
        }
    }

    fn expected_count(
        hierarchy: &ManifestLodHierarchy<'_>,
        base: &GaussianLodSettings,
        quality: f32,
        camera_position: Vec3,
    ) -> u64 {
        let mut settings = base.clone();
        settings.quality = quality;
        select_frontier(
            hierarchy,
            &AllResident,
            LodView::perspective(camera_position, HEIGHT as f32, VERTICAL_FOV, NEAR_PLANE),
            &settings,
        )
        .expect("quality fixture selection succeeds")
        .status
        .active_gaussians
    }

    fn quality_bridge_config(build_settings: GaussianLodBuildSettings) -> GaussianLodBridgeConfig {
        GaussianLodBridgeConfig {
            max_ephemeral_source_gaussians: 4_096,
            max_ephemeral_stored_gaussians: 8_192,
            max_atlas_gaussians: 8_192,
            max_atlas_bytes: 32 * 1024 * 1024,
            build_settings,
            ..default()
        }
    }

    fn setup(
        mut commands: Commands,
        mut state: ResMut<QualityRenderState>,
        mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let fixture = quality_fixture();
        let target = images.add(Image::new_target_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let source = gaussian_assets.add(fixture.cloud);
        let cloud = commands
            .spawn((
                PlanarGaussian3dHandle(source.clone()),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::Radix,
                    global_opacity: 1.5,
                    global_scale: 1.0,
                    opacity_adaptive_radius: false,
                    ..default()
                },
                fixture.settings,
                Transform::IDENTITY,
                Visibility::Visible,
                Name::new("lod_quality_patterned_cloud"),
            ))
            .id();
        let camera = commands
            .spawn((
                Camera3d::default(),
                Camera::default(),
                Projection::Perspective(PerspectiveProjection {
                    fov: VERTICAL_FOV,
                    near: NEAR_PLANE,
                    far: FAR_PLANE,
                    ..default()
                }),
                RenderTarget::Image(target.clone().into()),
                Transform::from_translation(Vec3::new(0.0, 0.0, NEAR_CAMERA_Z)),
                Tonemapping::None,
                GaussianCamera::default(),
                Name::new("lod_quality_perspective_camera"),
            ))
            .id();

        state.target = Some(target);
        state.cloud = Some(cloud);
        state.camera = Some(camera);
        state.source_handle = Some(source);
    }

    #[allow(clippy::too_many_arguments)]
    fn drive_capture(
        mut commands: Commands,
        mut state: ResMut<QualityRenderState>,
        statuses: Query<&GaussianLodBridgeStatus>,
        candidates: Query<&LodRenderCandidates>,
        handles: Query<&PlanarGaussian3dHandle>,
        mut lod_settings: Query<&mut GaussianLodSettings>,
        mut transforms: Query<&mut Transform>,
        indirect_probe: Res<IndirectProbe>,
    ) {
        state.total_frames += 1;
        state.phase_frames += 1;
        assert!(
            state.total_frames <= MAX_FRAMES,
            "LoD quality GPU test timed out in {:?}; status={:?}",
            state.phase,
            state.cloud.and_then(|cloud| statuses.get(cloud).ok())
        );

        let cloud = state.cloud.expect("cloud entity exists");
        let camera = state.camera.expect("camera entity exists");
        let capture = |commands: &mut Commands, state: &QualityRenderState| {
            commands.spawn(Screenshot::image(
                state.target.clone().expect("render target exists"),
            ));
        };

        match state.phase {
            Phase::ReferenceWarmup if state.phase_frames >= REFERENCE_WARMUP_FRAMES => {
                assert!(
                    statuses.get(cloud).is_err(),
                    "quality one unexpectedly retained bridge status"
                );
                assert!(
                    candidates.get(cloud).is_err(),
                    "quality one unexpectedly retained render candidates"
                );
                assert_source_handle_restored(&state, cloud, &handles);
                capture(&mut commands, &state);
                state.enter(Phase::ReferencePending);
            }
            Phase::LowerQualityWaiting => {
                let expected = state.expected_near[state.sample_index];
                if observe_exact_active_cut(
                    cloud,
                    camera,
                    expected,
                    &statuses,
                    &candidates,
                    &indirect_probe,
                ) {
                    state.stable_active_frames += 1;
                } else {
                    state.stable_active_frames = 0;
                }
                if state.stable_active_frames >= STABLE_ACTIVE_FRAMES {
                    capture(&mut commands, &state);
                    state.enter(Phase::LowerQualityPending);
                }
            }
            Phase::NearProbeWaiting => {
                if observe_exact_active_cut(
                    cloud,
                    camera,
                    state.expected_near_probe,
                    &statuses,
                    &candidates,
                    &indirect_probe,
                ) {
                    state.stable_active_frames += 1;
                } else {
                    state.stable_active_frames = 0;
                }
                if state.stable_active_frames >= STABLE_ACTIVE_FRAMES {
                    transforms
                        .get_mut(camera)
                        .expect("camera transform exists")
                        .translation = Vec3::new(0.0, 0.0, FAR_CAMERA_Z);
                    state.enter(Phase::FarProbeWaiting);
                }
            }
            Phase::FarProbeWaiting => {
                if observe_exact_active_cut(
                    cloud,
                    camera,
                    state.expected_far_probe,
                    &statuses,
                    &candidates,
                    &indirect_probe,
                ) {
                    state.stable_active_frames += 1;
                } else {
                    state.stable_active_frames = 0;
                }
                if state.stable_active_frames >= STABLE_ACTIVE_FRAMES {
                    assert!(
                        state.expected_far_probe < state.expected_near_probe,
                        "far perspective cut did not strictly coarsen"
                    );
                    transforms
                        .get_mut(camera)
                        .expect("camera transform exists")
                        .translation = Vec3::new(0.0, 0.0, DISTANT_CAMERA_Z);
                    state.enter(Phase::DistantProbeWaiting);
                }
            }
            Phase::DistantProbeWaiting => {
                if observe_exact_active_cut(
                    cloud,
                    camera,
                    state.expected_distant_probe,
                    &statuses,
                    &candidates,
                    &indirect_probe,
                ) {
                    state.stable_active_frames += 1;
                } else {
                    state.stable_active_frames = 0;
                }
                if state.stable_active_frames >= STABLE_ACTIVE_FRAMES {
                    assert!(
                        state.expected_distant_probe > 0
                            && state.expected_distant_probe <= state.expected_far_probe,
                        "distant perspective cut disappeared or refined: far={}, distant={}",
                        state.expected_far_probe,
                        state.expected_distant_probe
                    );
                    transforms
                        .get_mut(camera)
                        .expect("camera transform exists")
                        .translation = Vec3::new(0.0, 0.0, NEAR_CAMERA_Z);
                    state.enter(Phase::NearRecoveryWaiting);
                }
            }
            Phase::NearRecoveryWaiting => {
                if observe_exact_active_cut(
                    cloud,
                    camera,
                    state.expected_near_probe,
                    &statuses,
                    &candidates,
                    &indirect_probe,
                ) {
                    state.stable_active_frames += 1;
                } else {
                    state.stable_active_frames = 0;
                }
                if state.stable_active_frames >= STABLE_ACTIVE_FRAMES {
                    lod_settings
                        .get_mut(cloud)
                        .expect("LoD settings exist")
                        .quality = 1.0;
                    state.enter(Phase::RestoredReferenceWaiting);
                }
            }
            Phase::RestoredReferenceWaiting => {
                let restored = statuses.get(cloud).is_err()
                    && candidates.get(cloud).is_err()
                    && source_handle_is_restored(&state, cloud, &handles);
                if restored {
                    state.stable_active_frames += 1;
                } else {
                    state.stable_active_frames = 0;
                }
                if state.stable_active_frames >= RESTORED_WARMUP_FRAMES {
                    assert_source_handle_restored(&state, cloud, &handles);
                    capture(&mut commands, &state);
                    state.enter(Phase::RestoredReferencePending);
                }
            }
            Phase::ReferenceWarmup
            | Phase::ReferencePending
            | Phase::LowerQualityPending
            | Phase::RestoredReferencePending => {}
        }
    }

    fn observe_exact_active_cut(
        cloud: Entity,
        camera: Entity,
        expected: u64,
        statuses: &Query<&GaussianLodBridgeStatus>,
        candidates: &Query<&LodRenderCandidates>,
        indirect_probe: &IndirectProbe,
    ) -> bool {
        let Ok(status) = statuses.get(cloud) else {
            return false;
        };
        assert!(
            status.failure.is_none(),
            "automatic LoD bridge failed: {status:?}"
        );
        if status.phase != GaussianLodBridgePhase::Active {
            return false;
        }
        assert_eq!(status.active_views, 1);

        let candidates = candidates
            .get(cloud)
            .expect("active bridge publishes render candidates");
        assert_eq!(candidates.len(), 1);
        let candidate = candidates
            .get(camera)
            .expect("active bridge publishes the perspective camera cut");
        assert!(candidate.render_is_prepared());
        let explicit_count = candidate.frontier().candidate_count() as u64;
        let range_count = candidate
            .frontier()
            .physical_ranges()
            .iter()
            .map(|range| u64::from(range.count))
            .sum::<u64>();
        assert_eq!(explicit_count, range_count);
        assert_eq!(status.active_gaussians, explicit_count);
        status.active_gaussians == expected
            && indirect_probe
                .request(u32::try_from(expected).expect("quality fixture count is representable"))
    }

    fn source_handle_is_restored(
        state: &QualityRenderState,
        cloud: Entity,
        handles: &Query<&PlanarGaussian3dHandle>,
    ) -> bool {
        let Some(source) = &state.source_handle else {
            return false;
        };
        handles
            .get(cloud)
            .is_ok_and(|current| current.handle().id() == source.id())
    }

    fn assert_source_handle_restored(
        state: &QualityRenderState,
        cloud: Entity,
        handles: &Query<&PlanarGaussian3dHandle>,
    ) {
        assert!(
            source_handle_is_restored(state, cloud, handles),
            "quality one did not restore the original flat source handle"
        );
    }

    fn on_capture(
        trigger: On<ScreenshotCaptured>,
        mut state: ResMut<QualityRenderState>,
        mut lod_settings: Query<&mut GaussianLodSettings>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("screenshot converts")
            .to_rgba8();
        let linear = linear_rgba(rgba.as_raw());
        let foreground = linear.iter().filter(|pixel| pixel[3] > 0.5).count();
        assert!(
            foreground > 0,
            "LoD quality capture stayed empty: phase={:?}",
            state.phase
        );

        let cloud = state.cloud.expect("cloud entity exists");
        match state.phase {
            Phase::ReferencePending => {
                assert_eq!(state.source_count, 320);
                state.reference = Some(linear);
                lod_settings
                    .get_mut(cloud)
                    .expect("LoD settings exist")
                    .quality = QUALITIES[0];
                state.enter(Phase::LowerQualityWaiting);
            }
            Phase::LowerQualityPending => {
                let quality = QUALITIES[state.sample_index];
                let active_gaussians = state.expected_near[state.sample_index];
                let metrics = compare_linear_rgba(
                    state.reference.as_ref().expect("reference image exists"),
                    &linear,
                    0.5,
                )
                .expect("quality image metrics are valid");
                assert!(!metrics.psnr_rgb.is_nan());
                assert!(!metrics.foreground_psnr_rgb.is_nan());
                assert!(metrics.luminance_ssim.is_finite());
                assert!(
                    metrics.foreground_iou > 0.0,
                    "quality {quality} produced no foreground overlap: {metrics:?}"
                );
                state.captures.push(QualityCapture {
                    quality,
                    active_gaussians,
                    metrics,
                });

                state.sample_index += 1;
                if state.sample_index < QUALITIES.len() {
                    lod_settings
                        .get_mut(cloud)
                        .expect("LoD settings exist")
                        .quality = QUALITIES[state.sample_index];
                    state.enter(Phase::LowerQualityWaiting);
                } else {
                    assert_quality_results(&state);
                    lod_settings
                        .get_mut(cloud)
                        .expect("LoD settings exist")
                        .quality = CAMERA_PROBE_QUALITY;
                    state.enter(Phase::NearProbeWaiting);
                }
            }
            Phase::RestoredReferencePending => {
                let reference = state.reference.as_ref().expect("reference image exists");
                let restored = compare_linear_rgba(reference, &linear, 0.5)
                    .expect("restored reference metrics are valid");
                assert!(
                    restored.psnr_rgb.is_infinite(),
                    "quality-one source restoration changed pixels: {restored:?}"
                );
                assert_eq!(restored.max_abs_error, 0.0);
                exit.write(AppExit::Success);
            }
            phase => panic!("screenshot captured during unexpected phase {phase:?}"),
        }
    }

    fn assert_quality_results(state: &QualityRenderState) {
        eprintln!(
            "GPU LoD quality sweep: {:?}",
            state
                .captures
                .iter()
                .map(|capture| (
                    capture.quality,
                    capture.active_gaussians,
                    capture.metrics.psnr_rgb,
                    capture.metrics.luminance_ssim,
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(state.captures.len(), QUALITIES.len());
        assert_eq!(state.captures[0].quality, 0.0);
        assert_eq!(state.captures[0].active_gaussians, state.coarsest_count);
        assert!(
            state
                .captures
                .windows(2)
                .all(|pair| pair[0].active_gaussians < pair[1].active_gaussians),
            "GPU-observed quality counts were not strictly monotonic: {:?}",
            state.captures
        );
        for pair in state.captures.windows(2) {
            assert!(
                pair[1].metrics.psnr_rgb + 0.15 >= pair[0].metrics.psnr_rgb,
                "increasing quality regressed GPU PSNR: lower={:?}, higher={:?}",
                pair[0],
                pair[1]
            );
            assert!(
                pair[1].metrics.luminance_ssim + 0.002 >= pair[0].metrics.luminance_ssim,
                "increasing quality regressed GPU SSIM: lower={:?}, higher={:?}",
                pair[0],
                pair[1]
            );
        }
        assert!(
            state.captures.last().unwrap().active_gaussians < state.source_count,
            "interior quality unexpectedly bypassed LoD"
        );

        let coarse = state.captures.first().unwrap().metrics;
        let high = state.captures.last().unwrap().metrics;
        assert!(
            coarse.psnr_rgb.is_finite(),
            "coarsest representation was pixel-identical to source"
        );
        assert!(
            high.psnr_rgb >= coarse.psnr_rgb + 0.75,
            "quality sweep did not produce a meaningful PSNR improvement: coarse={coarse:?}, high={high:?}"
        );
        assert!(
            high.luminance_ssim >= coarse.luminance_ssim + 0.05,
            "quality sweep did not produce a meaningful structural-similarity improvement: coarse={coarse:?}, high={high:?}"
        );
        assert!(
            high.foreground_iou + 0.05 >= coarse.foreground_iou,
            "high-quality foreground coverage regressed: coarse={coarse:?}, high={high:?}"
        );
    }

    fn linear_rgba(bytes: &[u8]) -> Vec<[f32; 4]> {
        bytes
            .chunks_exact(4)
            .map(|pixel| {
                let max_rgb = pixel[0].max(pixel[1]).max(pixel[2]);
                [
                    srgb_to_linear(pixel[0]),
                    srgb_to_linear(pixel[1]),
                    srgb_to_linear(pixel[2]),
                    f32::from(max_rgb > 8),
                ]
            })
            .collect()
    }

    fn srgb_to_linear(value: u8) -> f32 {
        let value = f32::from(value) / 255.0;
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }
}
