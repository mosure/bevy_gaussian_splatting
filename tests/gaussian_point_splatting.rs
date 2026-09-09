//! One small, opt-in GPU fixture for the point renderer's image and work contract.

#[cfg(all(feature = "headless", feature = "testing", lod_render_path))]
mod headless {
    use std::{
        env,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::{Duration, Instant},
    };

    use bevy::{
        app::{AppExit, ScheduleRunnerPlugin},
        camera::{RenderTarget, ScalingMode, Viewport, visibility::NoFrustumCulling},
        core_pipeline::tonemapping::Tonemapping,
        prelude::*,
        render::{
            Render, RenderApp, RenderSystems,
            pipelined_rendering::PipelinedRenderingPlugin,
            render_resource::{CachedPipelineState, PipelineCache, TextureFormat, WgpuFeatures},
            renderer::RenderDevice,
            view::screenshot::{Screenshot, ScreenshotCaptured},
        },
        shader::ShaderCacheError,
        window::ExitCondition,
        winit::WinitPlugin,
    };
    use bevy_gaussian_splatting::{
        CloudSettings, Gaussian3d, GaussianCamera, GaussianLodBridgeConfig,
        GaussianLodBuildSettings, GaussianLodSettings, GaussianSplattingPlugin,
        LodPresentationMode, PlanarGaussian3d, PlanarGaussian3dHandle,
        SphericalHarmonicCoefficients,
        gaussian::{
            f32::Rotation, formats::planar_3d_lod::build_planar_3d_lod,
            settings::GaussianColorSpace,
        },
        render::{
            point::{
                GaussianPointSplattingAvailability, GaussianPointSplattingDiagnostics,
                GaussianPointSplattingSettings,
            },
            traversal::{
                GpuLodDrawAcknowledgements, GpuLodHierarchy, GpuLodHierarchyTree,
                GpuLodPagePlacement, GpuLodTraversalSettings,
            },
        },
        sort::SortMode,
        stream::{
            bridge::{GaussianLodBridgePhase, GaussianLodBridgeStatus},
            render_commit::LodRenderCandidates,
        },
        testing::{
            lod_scenes::{LodProjection, LodTestCamera},
            render_oracle::point::{
                PointExpectation, PointOracleOptions, render_point_expectation,
            },
        },
    };

    const TARGET: UVec2 = UVec2::new(96, 80);
    const ORIGIN: UVec2 = UVec2::new(16, 8);
    const EXTENT: u32 = 64;
    const SIGMA_WORLD: f32 = 0.2;
    const SIGMA_DEPTH_WORLD: f32 = 0.001;
    const OPACITY: f32 = 0.65;
    const RADIANCE: f32 = 0.8;
    const SAMPLES: u32 = 8;
    const MIN_AUTOMATIC_SAMPLES: u32 = 3;
    // Exactly sixteen independent fixed streams. No retry or seed expansion
    // follows a failed comparison; seed zero is the existing frozen baseline.
    const ENSEMBLE_SEEDS: [u32; 16] = [
        0x4750_5331,
        0x91a7_3e05,
        0x28cb_d649,
        0xb643_8f21,
        0x05ed_72b3,
        0xd2a1_49c7,
        0x6f38_b05d,
        0x3c97_ea61,
        0xe51d_269b,
        0x7a04_c8f3,
        0xa962_1d47,
        0x143f_b8ad,
        0xc80b_5379,
        0x52e9_a613,
        0x8d71_04ef,
        0xf329_6bc5,
    ];

    #[test]
    fn shared_point_visibility_matches_opacity_and_retains_complete_images_on_overflow() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping Gaussian point GPU fixture; set RUN_GPU_RENDER_TESTS=1");
            return;
        }
        let mut app = App::new();
        let timing_support = TimingSupport::default();
        app.insert_resource(ClearColor(Color::BLACK))
            .insert_resource(timing_support.clone())
            .insert_resource(GaussianLodBridgeConfig {
                auto_build_flat_clouds: false,
                ..default()
            })
            .insert_resource(CaptureState::default())
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
            .add_systems(Update, capture)
            .add_observer(captured);
        app.sub_app_mut(RenderApp)
            .insert_resource(timing_support)
            .add_systems(Render, fail_shader_errors.in_set(RenderSystems::Cleanup));
        assert_eq!(app.run(), AppExit::Success);
    }

    #[derive(Clone, Resource, Default)]
    struct TimingSupport(Arc<AtomicBool>);

    fn fail_shader_errors(
        cache: Res<PipelineCache>,
        device: Res<RenderDevice>,
        timing_support: Res<TimingSupport>,
    ) {
        timing_support.0.store(
            device.features().contains(WgpuFeatures::TIMESTAMP_QUERY),
            Ordering::Relaxed,
        );
        for pipeline in cache.pipelines() {
            if let CachedPipelineState::Err(error) = &pipeline.state {
                match error {
                    ShaderCacheError::ShaderNotLoaded(_)
                    | ShaderCacheError::ShaderImportNotYetAvailable => {}
                    _ => panic!("GPU fixture shader failed: {error}"),
                }
            }
        }
    }

    #[derive(Component)]
    struct TestCloud {
        center_z: f32,
    }

    #[derive(Component)]
    struct TestOccluder;

    #[derive(Component)]
    struct ReorderedCloud;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    enum Case {
        #[default]
        ViewportClipped,
        Layers,
        Ensemble,
        Frozen,
        FeedbackPaused,
        MeshBehind,
        MeshFront,
        MeshBetween,
        EqualDepth,
        EqualDepthReordered,
        Zero,
        Restored,
        AtlasBaseline,
        AtlasRelocated,
        Overflow,
        AdmissionFailure,
        MovingLod,
        AutomaticTiming,
        AutomaticOverflow,
    }

    #[derive(Resource)]
    struct CaptureState {
        case: Case,
        phase_frames: u32,
        frames: u32,
        started: Instant,
        pending: bool,
        camera: Option<Entity>,
        target: Option<Handle<Image>>,
        baseline: Vec<u8>,
        equal_depth_image: Vec<u8>,
        last_submission: u64,
        lod_cloud: Option<Entity>,
        active_moving_frames: u32,
        ensemble: Option<Ensemble>,
        relocated_hierarchies: Vec<(Entity, GpuLodHierarchy)>,
    }

    impl Default for CaptureState {
        fn default() -> Self {
            Self {
                case: Case::default(),
                phase_frames: 0,
                frames: 0,
                started: Instant::now(),
                pending: false,
                camera: None,
                target: None,
                baseline: Vec::new(),
                equal_depth_image: Vec::new(),
                last_submission: 0,
                lod_cloud: None,
                active_moving_frames: 0,
                ensemble: None,
                relocated_hierarchies: Vec::new(),
            }
        }
    }

    fn setup(
        mut commands: Commands,
        mut state: ResMut<CaptureState>,
        mut clouds: ResMut<Assets<PlanarGaussian3d>>,
        mut images: ResMut<Assets<Image>>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
    ) {
        let target = images.add(Image::new_target_texture(
            TARGET.x,
            TARGET.y,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        state.target = Some(target.clone());
        for (z, color) in [(0.2, [RADIANCE, 0.0, 0.0]), (-0.2, [0.0, 0.0, RADIANCE])] {
            let gaussian = layer_gaussian(z, color);
            commands.spawn((
                TestCloud { center_z: z },
                PlanarGaussian3dHandle(clouds.add(PlanarGaussian3d::from(vec![gaussian]))),
                CloudSettings {
                    sort_mode: SortMode::Radix,
                    color_space: GaussianColorSpace::LinRec709Display,
                    global_scale: 100.0,
                    global_opacity: 1.0,
                    opacity_adaptive_radius: false,
                    ..default()
                },
                Transform::from_xyz(8.0, 0.0, 0.0),
                Visibility::Visible,
                NoFrustumCulling,
            ));
        }
        // Queue the opaque pipeline outside the viewport before its first
        // capture, so mesh shader compilation cannot masquerade as occlusion.
        commands.spawn((
            TestOccluder,
            Mesh3d(meshes.add(Rectangle::new(2.4, 2.4))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::linear_rgb(0.0, RADIANCE, 0.0),
                unlit: true,
                ..default()
            })),
            Transform::from_xyz(4.0, 0.0, -0.6),
            NoFrustumCulling,
        ));
        state.camera = Some(
            commands
                .spawn((
                    Camera3d::default(),
                    Camera {
                        viewport: Some(Viewport {
                            physical_position: ORIGIN,
                            physical_size: UVec2::splat(EXTENT),
                            ..default()
                        }),
                        ..default()
                    },
                    Projection::Orthographic(OrthographicProjection {
                        near: 0.1,
                        far: 10.0,
                        scaling_mode: ScalingMode::FixedVertical {
                            viewport_height: 2.0,
                        },
                        ..OrthographicProjection::default_3d()
                    }),
                    RenderTarget::Image(target.into()),
                    Transform::from_xyz(0.0, 0.0, 3.0),
                    Tonemapping::None,
                    Msaa::Off,
                    GaussianCamera::default(),
                    GaussianPointSplattingSettings {
                        samples_per_pixel: SAMPLES,
                        max_projected_gaussians: 64,
                        max_points_per_frame: 131_072,
                        max_gpu_bytes: 4 * 1024 * 1024,
                        temporal_sampling: false,
                        seed: ENSEMBLE_SEEDS[0],
                        ..default()
                    },
                ))
                .id(),
        );
    }

    fn capture(
        mut commands: Commands,
        mut state: ResMut<CaptureState>,
        diagnostics: Res<GaussianPointSplattingDiagnostics>,
        mut cameras: Query<&mut Transform, With<GaussianCamera>>,
        lod_clouds: Query<(&GaussianLodBridgeStatus, &LodRenderCandidates)>,
        hierarchies: Query<(Entity, &GpuLodHierarchy), With<TestCloud>>,
        acknowledgements: Res<GpuLodDrawAcknowledgements>,
    ) {
        state.frames += 1;
        state.phase_frames += 1;
        if state.case == Case::FeedbackPaused && state.phase_frames >= 8 {
            // The three-slot receipt ring is already full before camera motion.
            cameras
                .get_mut(state.camera.unwrap())
                .unwrap()
                .translation
                .x = 0.5;
        } else if state.case == Case::MeshBehind && state.phase_frames == 1 {
            cameras
                .get_mut(state.camera.unwrap())
                .unwrap()
                .translation
                .x = 0.0;
        }
        if state.case == Case::MovingLod {
            cameras
                .get_mut(state.camera.unwrap())
                .unwrap()
                .translation
                .x = 0.15 * (state.phase_frames as f32 * 0.07).sin();
            let active = state
                .lod_cloud
                .and_then(|entity| lod_clouds.get(entity).ok())
                .is_some_and(|(status, candidates)| {
                    assert!(
                        status.failure.is_none(),
                        "moving LoD bridge failed: {status:?}"
                    );
                    status.phase == GaussianLodBridgePhase::Active
                        && status.active_gaussians > 0
                        && status.active_gaussians < 4
                        && candidates
                            .get(state.camera.unwrap())
                            .is_some_and(|candidate| {
                                candidate.render_is_active_for_testing()
                                    && candidate.frontier().presentation_mode()
                                        == LodPresentationMode::Discrete
                            })
                });
            state.active_moving_frames = if active {
                state.active_moving_frames + 1
            } else {
                0
            };
        }
        assert!(
            state.frames <= 5_000 && state.started.elapsed() < Duration::from_secs(45),
            "point fixture timed out in {:?} after {} frames; latest={:?}",
            state.case,
            state.frames,
            state.camera.and_then(|camera| diagnostics.get(camera))
        );
        if state.pending || state.phase_frames < 12 {
            return;
        }
        if state.case == Case::MovingLod && state.active_moving_frames < 12 {
            return;
        }
        let Some(frame) = state.camera.and_then(|camera| diagnostics.get(camera)) else {
            return;
        };
        if state.case == Case::FeedbackPaused {
            assert_eq!(
                frame.submission, state.last_submission,
                "paused telemetry published an uncaptured receipt"
            );
            state.pending = true;
            commands.spawn(Screenshot::image(state.target.clone().unwrap()));
            return;
        }
        if state.case == Case::AdmissionFailure {
            if frame.availability != GaussianPointSplattingAvailability::Retained
                || frame.error.is_none()
            {
                return;
            }
            state.pending = true;
            commands.spawn(Screenshot::image(state.target.clone().unwrap()));
            return;
        }
        if frame.submission <= state.last_submission {
            return;
        }
        if matches!(state.case, Case::AtlasBaseline | Case::AtlasRelocated)
            && hierarchies.iter().any(|(entity, hierarchy)| {
                acknowledgements
                    .get(state.camera.unwrap(), entity)
                    .is_none_or(|ack| {
                        ack.residency_generation != hierarchy.0.generation
                            || ack.source != hierarchy.0.source
                            || ack.submission != frame.submission
                    })
            })
        {
            return;
        }
        assert!(frame.error.is_none(), "point renderer failed: {frame:?}");
        let expected = match state.case {
            Case::Zero => !frame.overflow && !frame.sampling_failed && frame.requested_points == 0,
            Case::Overflow | Case::AutomaticOverflow => frame.overflow && !frame.sampling_failed,
            _ => !frame.overflow && !frame.sampling_failed && frame.requested_points > 0,
        };
        if !expected {
            return;
        }
        if matches!(state.case, Case::AutomaticTiming | Case::AutomaticOverflow) {
            let Some(gpu_ms) = frame.gpu_ms else {
                return;
            };
            assert!(
                gpu_ms.is_finite() && gpu_ms >= 0.0,
                "invalid GPU timestamp: {frame:?}"
            );
            // Reaching three proves feedback reduced eight without crossing
            // the configured floor. Overflow must still dispatch nothing.
            assert!(
                frame.samples_per_pixel >= MIN_AUTOMATIC_SAMPLES,
                "automatic feedback crossed the configured sample floor: {frame:?}"
            );
            if frame.samples_per_pixel != MIN_AUTOMATIC_SAMPLES {
                return;
            }
            // Allocation shrinks at half capacity to avoid resize churn;
            // three active layers may retain four, but cannot retain eight.
            assert!(
                (MIN_AUTOMATIC_SAMPLES..2 * MIN_AUTOMATIC_SAMPLES)
                    .contains(&frame.allocated_samples_per_pixel),
                "automatic downshift violated layer-capacity hysteresis: {frame:?}"
            );
            eprintln!(
                "GPS {:?}: samples={}, allocated_layers={}, gpu_ms={gpu_ms}, requested={}, dispatched={}",
                state.case,
                frame.samples_per_pixel,
                frame.allocated_samples_per_pixel,
                frame.requested_points,
                frame.dispatched_points
            );
        } else {
            assert_eq!(frame.samples_per_pixel, SAMPLES);
        }
        match state.case {
            Case::Zero => assert_eq!(frame.projected_gaussians, 0),
            Case::MovingLod | Case::AutomaticTiming | Case::AutomaticOverflow => {
                assert!((1..4).contains(&frame.projected_gaussians));
            }
            _ => assert_eq!(frame.projected_gaussians, 2),
        }
        if frame.overflow {
            assert_eq!(
                frame.dispatched_points, 0,
                "overflow dispatched a partial point process"
            );
        } else {
            assert_eq!(frame.dispatched_points, frame.requested_points);
        }
        assert!(frame.gpu_bytes > 0 && frame.gpu_bytes <= 4 * 1024 * 1024);
        state.last_submission = frame.submission;
        state.pending = true;
        commands.spawn(Screenshot::image(state.target.clone().unwrap()));
    }

    #[allow(clippy::too_many_arguments)]
    fn captured(
        event: On<ScreenshotCaptured>,
        mut commands: Commands,
        mut state: ResMut<CaptureState>,
        mut clouds: Query<(Entity, &TestCloud, &mut CloudSettings, &mut Transform)>,
        mut occluders: Query<&mut Transform, (With<TestOccluder>, Without<TestCloud>)>,
        mut settings: Query<&mut GaussianPointSplattingSettings>,
        mut bridge: ResMut<GaussianLodBridgeConfig>,
        mut assets: ResMut<Assets<PlanarGaussian3d>>,
        mut exit: MessageWriter<AppExit>,
        timing_support: Res<TimingSupport>,
        diagnostics: Res<GaussianPointSplattingDiagnostics>,
    ) {
        let image = event.image.clone().try_into_dynamic().unwrap().to_rgba8();
        assert_eq!(image.dimensions(), (TARGET.x, TARGET.y));
        let bytes = image.into_raw();
        for (i, pixel) in bytes.chunks_exact(4).enumerate() {
            let x = i as u32 % TARGET.x;
            let y = i as u32 / TARGET.x;
            if !(ORIGIN.x..ORIGIN.x + EXTENT).contains(&x)
                || !(ORIGIN.y..ORIGIN.y + EXTENT).contains(&y)
            {
                assert_eq!(
                    &pixel[..3],
                    &[0, 0, 0],
                    "point composite escaped its viewport at {x},{y}"
                );
            }
        }
        state.case = match state.case {
            Case::ViewportClipped => {
                // Each full radial process exceeds the 1M numerical component
                // bound. Its mean is offscreen, but its huge support covers the
                // viewport. Clipped thinning must render the intended opacity.
                assert_layer_opacity_geometry(&bytes, SIGMA_WORLD * 100.0, 8.0);
                for (_, _, mut cloud, mut transform) in &mut clouds {
                    cloud.global_scale = 1.0;
                    transform.translation = Vec3::ZERO;
                }
                settings
                    .get_mut(state.camera.unwrap())
                    .unwrap()
                    .max_points_per_frame = 65_536;
                Case::Layers
            }
            Case::Layers => {
                assert_layer_opacity(&bytes);
                let mut ensemble = Ensemble::new();
                ensemble.observe(&bytes);
                state.ensemble = Some(ensemble);
                state.baseline = bytes;
                settings.get_mut(state.camera.unwrap()).unwrap().seed = ENSEMBLE_SEEDS[1];
                Case::Ensemble
            }
            Case::Ensemble => {
                let ensemble = state.ensemble.as_mut().unwrap();
                ensemble.observe(&bytes);
                if ensemble.count == ENSEMBLE_SEEDS.len() {
                    state.ensemble.take().unwrap().finish();
                    settings.get_mut(state.camera.unwrap()).unwrap().seed = ENSEMBLE_SEEDS[0];
                    Case::Frozen
                } else {
                    let seed = ENSEMBLE_SEEDS[ensemble.count];
                    settings.get_mut(state.camera.unwrap()).unwrap().seed = seed;
                    Case::Ensemble
                }
            }
            Case::Frozen => {
                assert_eq!(
                    bytes, state.baseline,
                    "fixed-seed samples changed across frames"
                );
                state.last_submission = diagnostics.get(state.camera.unwrap()).unwrap().submission;
                diagnostics.set_feedback_paused_for_testing(true);
                Case::FeedbackPaused
            }
            Case::FeedbackPaused => {
                let center_x = |image: &[u8]| {
                    let (moment, mass) = image.chunks_exact(4).enumerate().fold(
                        (0.0, 0.0),
                        |(moment, mass), (index, pixel)| {
                            let weight = pixel[..3]
                                .iter()
                                .map(|value| f64::from(*value))
                                .sum::<f64>();
                            (
                                moment + (index as u32 % TARGET.x) as f64 * weight,
                                mass + weight,
                            )
                        },
                    );
                    assert!(mass > 0.0, "camera motion lost the admitted complete image");
                    moment / mass
                };
                assert!(
                    center_x(&state.baseline) - center_x(&bytes) > 8.0,
                    "a full telemetry ring retained the old camera image"
                );
                diagnostics.set_feedback_paused_for_testing(false);
                occluders.single_mut().unwrap().translation = Vec3::new(0.0, 0.0, -0.6);
                Case::MeshBehind
            }
            Case::MeshBehind => {
                assert_mesh_occlusion(&bytes, &state.baseline, true, true);
                occluders.single_mut().unwrap().translation.z = 0.6;
                Case::MeshFront
            }
            Case::MeshFront => {
                assert_mesh_occlusion(&bytes, &state.baseline, false, false);
                occluders.single_mut().unwrap().translation.z = 0.0;
                Case::MeshBetween
            }
            Case::MeshBetween => {
                // GPS uses Gaussian center depth: a plane between the centers
                // clips the blue layer while the red layer remains in front.
                assert_mesh_occlusion(&bytes, &state.baseline, true, false);
                occluders.single_mut().unwrap().translation.x = 4.0;
                for (_, cloud, _, mut transform) in &mut clouds {
                    transform.translation.z = -cloud.center_z;
                }
                Case::EqualDepth
            }
            Case::EqualDepth => {
                assert_equal_depth_coverage(&bytes, &state.baseline);
                state.equal_depth_image = bytes;
                let entity = clouds.iter().next().unwrap().0;
                commands.entity(entity).insert(ReorderedCloud);
                Case::EqualDepthReordered
            }
            Case::EqualDepthReordered => {
                assert_eq!(
                    bytes, state.equal_depth_image,
                    "equal-depth cloud ties changed after ECS query order changed"
                );
                for (_, _, mut cloud, mut transform) in &mut clouds {
                    cloud.global_opacity = 0.0;
                    transform.translation.z = 0.0;
                }
                Case::Zero
            }
            Case::Zero => {
                assert!(
                    bytes.chunks_exact(4).all(|pixel| pixel[..3] == [0, 0, 0]),
                    "zero-point success retained a stale image"
                );
                for (_, _, mut cloud, _) in &mut clouds {
                    cloud.global_opacity = 1.0;
                }
                Case::Restored
            }
            Case::Restored => {
                assert_eq!(
                    bytes, state.baseline,
                    "restoring opacity changed the fixed-seed image"
                );
                // Two immutable copies model a page moving to another atlas
                // slot without changing its authenticated logical records.
                // The same entity, asset and camera persist across snapshots.
                for (entity, cloud, _, _) in &mut clouds {
                    let color = if cloud.center_z > 0.0 {
                        [RADIANCE, 0.0, 0.0]
                    } else {
                        [0.0, 0.0, RADIANCE]
                    };
                    let gaussian = layer_gaussian(cloud.center_z, color);
                    let built = build_planar_3d_lod(
                        &PlanarGaussian3d::from(vec![gaussian]),
                        GaussianLodBuildSettings::default(),
                    )
                    .unwrap();
                    assert_eq!(built.manifest.pages.len(), 1);
                    assert_eq!(built.pages[0].gaussians, vec![gaussian]);
                    let tree =
                        Arc::new(GpuLodHierarchyTree::from_manifest(&built.manifest).unwrap());
                    let handle = assets.add(PlanarGaussian3d::from(vec![gaussian, gaussian]));
                    let snapshot = |generation, start| {
                        GpuLodHierarchy::new(
                            tree.clone(),
                            generation,
                            handle.id(),
                            vec![Some(GpuLodPagePlacement { start, count: 1 })],
                            Arc::new(()),
                        )
                        .unwrap()
                    };
                    state.relocated_hierarchies.push((entity, snapshot(2, 1)));
                    commands.entity(entity).insert((
                        snapshot(1, 0),
                        PlanarGaussian3dHandle(handle),
                        GaussianLodSettings {
                            quality: 1.0,
                            presentation_mode: LodPresentationMode::Discrete,
                            ..default()
                        },
                    ));
                }
                commands
                    .entity(state.camera.unwrap())
                    .insert(GpuLodTraversalSettings {
                        max_selected_gaussians: 4,
                        max_frontier_nodes: 4,
                        max_visited_nodes: 16,
                        max_page_requests: 4,
                        max_gpu_bytes: 1024 * 1024,
                    });
                Case::AtlasBaseline
            }
            Case::AtlasBaseline => {
                assert!(bytes.chunks_exact(4).any(|pixel| pixel[..3] != [0, 0, 0]));
                state.baseline = bytes;
                for (entity, hierarchy) in state.relocated_hierarchies.drain(..) {
                    commands.entity(entity).insert(hierarchy);
                }
                Case::AtlasRelocated
            }
            Case::AtlasRelocated => {
                assert_eq!(
                    bytes, state.baseline,
                    "relocating unchanged hierarchy pages changed frozen point samples"
                );
                settings
                    .get_mut(state.camera.unwrap())
                    .unwrap()
                    .max_points_per_frame = 1;
                Case::Overflow
            }
            Case::Overflow => {
                assert_eq!(
                    bytes, state.baseline,
                    "overflow replaced the complete image with a prefix"
                );
                settings
                    .get_mut(state.camera.unwrap())
                    .unwrap()
                    .max_gpu_bytes = 1;
                Case::AdmissionFailure
            }
            Case::AdmissionFailure => {
                assert_eq!(
                    bytes, state.baseline,
                    "failed memory admission discarded the complete image or enabled per-cloud rendering"
                );
                settings
                    .get_mut(state.camera.unwrap())
                    .unwrap()
                    .max_gpu_bytes = 4 * 1024 * 1024;
                settings
                    .get_mut(state.camera.unwrap())
                    .unwrap()
                    .max_points_per_frame = 65_536;
                for (entity, _, _, _) in &mut clouds {
                    commands.entity(entity).despawn();
                }
                commands
                    .entity(state.camera.unwrap())
                    .remove::<GpuLodTraversalSettings>();
                bridge.auto_build_flat_clouds = true;
                bridge.max_ephemeral_source_gaussians = 64;
                bridge.max_ephemeral_stored_gaussians = 256;
                bridge.max_atlas_gaussians = 64;
                bridge.max_atlas_bytes = 4 * 1024 * 1024;
                bridge.build_settings = GaussianLodBuildSettings {
                    branching_factor: 2,
                    leaf_capacity: 1,
                    ..default()
                };
                let mut records = Vec::new();
                for x in [-0.1, 0.1] {
                    for y in [-0.1, 0.1] {
                        records.push(Gaussian3d {
                            position_visibility: [x, y, 0.0, 1.0].into(),
                            rotation: Rotation {
                                rotation: [1.0, 0.0, 0.0, 0.0],
                            },
                            scale_opacity: [0.08, 0.08, 0.08, 0.7].into(),
                            spherical_harmonic: default(),
                        });
                    }
                }
                let mut lod = GaussianLodSettings {
                    quality: 0.0,
                    presentation_mode: LodPresentationMode::Discrete,
                    frustum_culling: false,
                    ..default()
                };
                lod.budgets.max_active_gaussians = 4;
                state.lod_cloud = Some(
                    commands
                        .spawn((
                            TestCloud { center_z: 0.0 },
                            PlanarGaussian3dHandle(assets.add(PlanarGaussian3d::from(records))),
                            CloudSettings {
                                sort_mode: SortMode::Radix,
                                color_space: GaussianColorSpace::LinRec709Display,
                                global_scale: 1.0,
                                global_opacity: 1.0,
                                ..default()
                            },
                            lod,
                            Transform::default(),
                            Visibility::Visible,
                            NoFrustumCulling,
                        ))
                        .id(),
                );
                // A recreated per-view workspace may restart its submission
                // counter; acceptance still requires twelve moving ACTIVE frames.
                state.last_submission = 0;
                Case::MovingLod
            }
            Case::MovingLod => {
                assert!(state.active_moving_frames >= 12);
                assert!(
                    bytes.chunks_exact(4).any(|pixel| pixel[..3] != [0, 0, 0]),
                    "moving discrete candidate activated without a visible image"
                );
                if timing_support.0.load(Ordering::Relaxed) {
                    let mut policy = settings.get_mut(state.camera.unwrap()).unwrap();
                    policy.min_samples_per_pixel = MIN_AUTOMATIC_SAMPLES;
                    policy.target_gpu_ms = Some(0.000_001);
                    Case::AutomaticTiming
                } else {
                    eprintln!(
                        "automatic GPU timing phase skipped: RenderDevice has no TIMESTAMP_QUERY"
                    );
                    exit.write(AppExit::Success);
                    Case::MovingLod
                }
            }
            Case::AutomaticTiming => {
                assert!(
                    bytes.chunks_exact(4).any(|pixel| pixel[..3] != [0, 0, 0]),
                    "automatic sampling produced no image"
                );
                state.baseline = bytes;
                settings
                    .get_mut(state.camera.unwrap())
                    .unwrap()
                    .max_points_per_frame = 1;
                Case::AutomaticOverflow
            }
            Case::AutomaticOverflow => {
                assert_eq!(
                    bytes, state.baseline,
                    "automatic overflow replaced the complete image with partial sampling"
                );
                exit.write(AppExit::Success);
                Case::AutomaticOverflow
            }
        };
        state.pending = false;
        state.phase_frames = 0;
    }

    fn linear(channel: u8) -> f64 {
        linear_value(f64::from(channel) / 255.0)
    }

    fn linear_value(value: f64) -> f64 {
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    fn layer_gaussian(z: f32, color: [f32; 3]) -> Gaussian3d {
        let mut sh = SphericalHarmonicCoefficients::default();
        for (channel, value) in color.into_iter().enumerate() {
            sh.set(channel, (value - 0.5) / 0.282_094_8);
        }
        Gaussian3d {
            position_visibility: [0.0, 0.0, z, 1.0].into(),
            rotation: Rotation {
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
            // These orthographic layers test XY intensity and center-depth
            // occlusion. Even the 100x offscreen-support case stays wholly
            // beyond the near plane: depth support is only 3 * 0.001 * 100.
            // The ensemble oracle and relocated atlas copies reuse this exact
            // record; near_clip_render covers isotropic near-camera support.
            scale_opacity: [SIGMA_WORLD, SIGMA_WORLD, SIGMA_DEPTH_WORLD, OPACITY].into(),
            spherical_harmonic: sh,
        }
    }

    /// Per-pixel mean/M2 and oracle intervals together occupy under 512 KiB.
    /// Displayed-frame RMSE is kept separate from the sixteen-seed ensemble.
    struct Ensemble {
        oracle: PointExpectation,
        mean: Vec<[f64; 3]>,
        m2: Vec<[f64; 3]>,
        count: usize,
        frame_rmse: Vec<f64>,
        channel_mean: [f64; 3],
        channel_m2: [f64; 3],
        quantization_sum: [f64; 3],
        quantization_squared: f64,
    }

    impl Ensemble {
        fn new() -> Self {
            let records = [
                layer_gaussian(0.2, [RADIANCE, 0.0, 0.0]),
                layer_gaussian(-0.2, [0.0, 0.0, RADIANCE]),
            ];
            let oracle = render_point_expectation(
                &records,
                LodTestCamera {
                    position: Vec3::new(0.0, 0.0, 3.0),
                    target: Vec3::ZERO,
                    projection: LodProjection::Orthographic {
                        vertical_world_size: 2.0,
                    },
                    near: 0.1,
                    far: 10.0,
                    viewport: [EXTENT; 2],
                    ..default()
                },
                GaussianColorSpace::LinRec709Display,
                PointOracleOptions::default(),
            )
            .expect("bounded two-record GPS expectation must complete");
            assert!(
                oracle.converged,
                "GPS expectation interval remained unresolved: width={}, work={:?}",
                oracle.maximum_interval_width, oracle.work
            );
            let pixels = (EXTENT * EXTENT) as usize;
            Self {
                oracle,
                mean: vec![[0.0; 3]; pixels],
                m2: vec![[0.0; 3]; pixels],
                count: 0,
                frame_rmse: Vec::with_capacity(ENSEMBLE_SEEDS.len()),
                channel_mean: [0.0; 3],
                channel_m2: [0.0; 3],
                quantization_sum: [0.0; 3],
                quantization_squared: 0.0,
            }
        }

        fn observe(&mut self, bytes: &[u8]) {
            assert!(self.count < ENSEMBLE_SEEDS.len());
            self.count += 1;
            let n = self.count as f64;
            let mut error_squared = 0.0;
            let mut channel_mass = [0.0; 3];
            for y in 0..EXTENT {
                for x in 0..EXTENT {
                    let pixel = (y * EXTENT + x) as usize;
                    let offset = (((y + ORIGIN.y) * TARGET.x + x + ORIGIN.x) * 4) as usize;
                    for (channel, mass) in channel_mass.iter_mut().enumerate() {
                        let code = bytes[offset + channel];
                        let value = linear(code);
                        let expected = (self.oracle.lower[pixel][channel]
                            + self.oracle.upper[pixel][channel])
                            * 0.5;
                        error_squared += (value - expected).powi(2);
                        *mass += value;
                        let delta = value - self.mean[pixel][channel];
                        self.mean[pixel][channel] += delta / n;
                        self.m2[pixel][channel] += delta * (value - self.mean[pixel][channel]);
                        // Decode the complete half-code sRGB bin into linear light.
                        // Quantization is systematic and does not shrink as 1/sqrt(N).
                        let low = linear_value((f64::from(code) - 0.5).max(0.0) / 255.0);
                        let high = linear_value((f64::from(code) + 0.5).min(255.0) / 255.0);
                        let allowance = (value - low).max(high - value) + 1e-5;
                        self.quantization_sum[channel] += allowance;
                        self.quantization_squared += allowance * allowance;
                    }
                }
            }
            let pixels = f64::from(EXTENT * EXTENT);
            self.frame_rmse
                .push((error_squared / (3.0 * pixels)).sqrt());
            for (channel, mass) in channel_mass.into_iter().enumerate() {
                let value = mass / pixels;
                let delta = value - self.channel_mean[channel];
                self.channel_mean[channel] += delta / n;
                self.channel_m2[channel] += delta * (value - self.channel_mean[channel]);
            }
        }

        fn finish(self) {
            assert_eq!(self.count, ENSEMBLE_SEEDS.len());
            let n = self.count as f64;
            let pixels = f64::from(EXTENT * EXTENT);
            let mut bias_squared = 0.0;
            let mut se_squared = 0.0;
            let mut oracle_squared = 0.0;
            let mut expected_mean = [0.0; 3];
            let mut oracle_allowance = [0.0; 3];
            for pixel in 0..self.mean.len() {
                for (channel, channel_expected) in expected_mean.iter_mut().enumerate() {
                    let expected = (self.oracle.lower[pixel][channel]
                        + self.oracle.upper[pixel][channel])
                        * 0.5;
                    let half_width = (self.oracle.upper[pixel][channel]
                        - self.oracle.lower[pixel][channel])
                        * 0.5;
                    *channel_expected += expected / pixels;
                    oracle_allowance[channel] += half_width / pixels;
                    bias_squared += (self.mean[pixel][channel] - expected).powi(2);
                    se_squared += self.m2[pixel][channel] / ((n - 1.0) * n);
                    oracle_squared += half_width * half_width;
                }
            }
            let bias_rmse = (bias_squared / (3.0 * pixels)).sqrt();
            let se_rmse = (se_squared / (3.0 * pixels)).sqrt();
            let quantization_rmse = (self.quantization_squared / (n * 3.0 * pixels)).sqrt();
            let oracle_rmse = (oracle_squared / (3.0 * pixels)).sqrt();
            eprintln!(
                "GPS expectation: seeds={}, spp={SAMPLES}, displayed_frame_rmse={:?}, ensemble_bias_rmse={bias_rmse:.7}, estimated_mc_se_rmse={se_rmse:.7}, quantization_allowance={quantization_rmse:.7}, oracle_allowance={oracle_rmse:.7}, oracle_work={:?}",
                self.count, self.frame_rmse, self.oracle.work
            );
            // A spatial RMS screen plus six-SE channel-mean checks detects a
            // repeated stream or systematic layer bias without treating a
            // sixteen-frame average as delivered-frame quality qualification.
            assert!(
                bias_rmse <= 1.35 * se_rmse + quantization_rmse + oracle_rmse,
                "GPS ensemble has excess spatial bias: bias={bias_rmse}, SE={se_rmse}"
            );
            for channel in [0, 2] {
                let se = (self.channel_m2[channel] / ((n - 1.0) * n)).sqrt();
                let quantization = self.quantization_sum[channel] / (n * pixels);
                let bias = (self.channel_mean[channel] - expected_mean[channel]).abs();
                eprintln!(
                    "GPS channel {channel}: mean_bias={bias:.8}, ensemble_SE={se:.8}, quantization={quantization:.8}, oracle={:.8}",
                    oracle_allowance[channel]
                );
                assert!(
                    bias <= 6.0 * se + quantization + oracle_allowance[channel],
                    "GPS channel {channel} mean disagrees with integrated occupancy: bias={bias}, SE={se}"
                );
            }
        }
    }

    fn assert_mesh_occlusion(bytes: &[u8], baseline: &[u8], red_visible: bool, blue_visible: bool) {
        let mut green = 0.0;
        for y in ORIGIN.y..ORIGIN.y + EXTENT {
            for x in ORIGIN.x..ORIGIN.x + EXTENT {
                let offset = ((y * TARGET.x + x) * 4) as usize;
                for (channel, visible) in [(0, red_visible), (2, blue_visible)] {
                    let expected = if visible {
                        baseline[offset + channel]
                    } else {
                        0
                    };
                    assert!(
                        bytes[offset + channel].abs_diff(expected) <= 1,
                        "opaque depth failed at {x},{y}, channel {channel}: red={red_visible}, blue={blue_visible}"
                    );
                }
                green += linear(bytes[offset + 1]);
            }
        }
        assert!(
            green > f64::from(EXTENT * EXTENT) * 0.4,
            "opaque mesh did not contribute its visible green background"
        );
    }

    fn assert_equal_depth_coverage(bytes: &[u8], baseline: &[u8]) {
        let mut channels = [0.0; 2];
        let mut baseline_mass = 0.0;
        for (pixel, original) in bytes.chunks_exact(4).zip(baseline.chunks_exact(4)) {
            channels[0] += linear(pixel[0]);
            channels[1] += linear(pixel[2]);
            baseline_mass += linear(original[0]) + linear(original[2]);
            assert_eq!(pixel[1], 0, "occluder remained visible in the tie case");
        }
        assert!(
            (channels.iter().sum::<f64>() - baseline_mass).abs() <= baseline_mass * 0.01 + 1.0,
            "equal-depth winner resolution lost point coverage: {channels:?}, baseline={baseline_mass}"
        );
        assert!(
            channels[0].max(channels[1]) > channels[0].min(channels[1]) * 1.15,
            "equal-depth overlaps did not consistently prefer one cloud: {channels:?}"
        );
    }

    fn assert_layer_opacity(bytes: &[u8]) {
        assert_layer_opacity_geometry(bytes, SIGMA_WORLD, 0.0);
    }

    fn assert_layer_opacity_geometry(bytes: &[u8], sigma_world: f32, center_x: f32) {
        // Orthographic projection spans two world units. Include the renderer's
        // physical-pixel mip variance and determinant opacity compensation.
        let original_variance = (f64::from(sigma_world) * f64::from(EXTENT) / 2.0).powi(2);
        let variance = original_variance + 0.3;
        let peak = f64::from(OPACITY) * original_variance / variance;
        let mut expected = [0.0; 3];
        let mut actual = [0.0; 3];
        for y in 0..EXTENT {
            for x in 0..EXTENT {
                let dx = f64::from(x) + 0.5 - (1.0 + f64::from(center_x)) * f64::from(EXTENT) / 2.0;
                let dy = f64::from(y) + 0.5 - f64::from(EXTENT) / 2.0;
                let radius_squared = (dx * dx + dy * dy) / variance;
                if radius_squared <= 9.0 {
                    let alpha = peak * (-0.5 * radius_squared).exp();
                    expected[0] += f64::from(RADIANCE) * alpha;
                    expected[2] += f64::from(RADIANCE) * (1.0 - alpha) * alpha;
                }
                let offset = (((y + ORIGIN.y) * TARGET.x + x + ORIGIN.x) * 4) as usize;
                for channel in 0..3 {
                    actual[channel] += linear(bytes[offset + channel]);
                }
            }
        }
        for channel in [0, 2] {
            assert!(
                (actual[channel] - expected[channel]).abs() <= expected[channel] * 0.12 + 1.0,
                "shared point visibility disagrees with alpha composition: actual={actual:?}, expected={expected:?}"
            );
        }
        assert!(
            actual[0] > actual[2] * 1.15,
            "the nearer red cloud did not occlude blue: {actual:?}"
        );
        assert!(actual[1] < 0.1, "unexpected green radiance: {actual:?}");
    }
}
