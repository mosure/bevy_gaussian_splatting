#[cfg(not(all(feature = "headless", feature = "testing")))]
#[test]
fn lod_quality_render_test_requires_headless_and_testing_features() {}

#[cfg(all(feature = "headless", feature = "testing"))]
mod headless {
    use std::{
        any::TypeId,
        collections::{BTreeMap, BTreeSet, VecDeque},
        env, fs,
        io::{BufReader, Read},
        mem::size_of,
        path::{Path, PathBuf},
        sync::{Arc, Mutex},
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use bevy::{
        app::{AppExit, PluginsState, ScheduleRunnerPlugin},
        asset::AssetId,
        camera::{PerspectiveProjection, Projection, RenderTarget},
        core_pipeline::tonemapping::Tonemapping,
        prelude::*,
        render::{
            Render, RenderApp, RenderSystems,
            extract_resource::{ExtractResource, ExtractResourcePlugin},
            pipelined_rendering::PipelinedRenderingPlugin,
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
        GaussianLodBuildSettings, GaussianLodDebugAvailability, GaussianLodHandle,
        GaussianLodLifecycle, GaussianLodManifest, GaussianLodPackageConfig,
        GaussianLodPackageSource, GaussianLodSettings, GaussianLodSourceKind, GaussianLodStatus,
        GaussianMode, GaussianSplattingPlugin, GaussianStreamingSettings, LodDebugPreset,
        LodDebugSettings, LodDegradation, LodEffectiveStatus, LodPageStorage, LodQualityTarget,
        LodReducerKind, LodSelectionMode, PlanarGaussian3d, PlanarGaussian3dHandle, PlanarHandle,
        RadixSortDepthBits,
        gaussian::{
            cloud::CloudVisibilityClass,
            lod_debug::{LodDebugMetadata, LodDebugRecord},
        },
        io::{
            lod::{
                GaussianLodAsset, LodCodecLimits, decode_manifest, encode_manifest, encode_page,
            },
            ply::parse_ply_3d,
        },
        render::{
            LodDebugBindGroup, LodDebugGpuUploadStats, ShaderDefines,
            lod::{
                LodCompactionBuffers, LodIndirectArgs, LodLastRadixDrawableForTesting,
                LodViewBlendPublicationLabel, LodViewBlendUploadStats, finalized_indirect_args,
                lod_view_blend_pressures_for_testing, lod_view_blend_view_for_testing,
                lod_view_blend_weight_for_testing, read_lod_indirect_args_for_testing,
            },
        },
        sort::SortMode,
        stream::{
            atlas_upload::{LodAtlasUploadBudget, LodAtlasUploadBudgetStatus, LodAtlasUploadQueue},
            bridge::{GaussianLodBridgePhase, GaussianLodBridgeStatus},
            hierarchy::{
                AllResident, LodView, ManifestLodHierarchy, select_frontier,
                select_frontier_with_visibility,
            },
            package::{
                GaussianLodPackagePhase, GaussianLodPackageStatus,
                GaussianLodPackageTestingSnapshot,
            },
            render_commit::{
                LodRenderCandidate, LodRenderCandidates, LodViewBlendEndpoint,
                LodViewBlendTestingSnapshot,
            },
            runtime::{
                LodTemporalTransitionMode, LodViewBlendEdge, LodViewBlendIdentity,
                LodViewBlendMetric,
            },
        },
        testing::{
            ImageMetrics, LodTestScene, compare_linear_rgba,
            upgrade_manifest_to_synthetic_abi16_lifecycle_fixture,
        },
        utils::VIEWER_DEFAULT_LOD_HYSTERESIS,
    };
    use sha2::{Digest, Sha256};

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
    const PACKAGE_MAX_FRAMES: u32 = 2_400;

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
        // This fixture is owned by the in-memory `build_planar_3d_lod`
        // ABI-14/MomentMerge-v3 path, not the external Garden ABI-16 fitter.
        // Keep its exact selector outputs as a separately calibrated contract.
        assert_eq!(
            fixture.expected_near,
            [1, 3, 7, 11],
            "the calibrated in-memory builder quality contract drifted"
        );
        assert_eq!(
            (
                fixture.expected_near_probe,
                fixture.expected_far_probe,
                fixture.expected_distant_probe,
            ),
            (7, 2, 1),
            "the calibrated in-memory builder perspective contract drifted"
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

    #[test]
    fn native_preprocessed_package_streams_bounded_camera_aware_cuts() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!(
                "skipping native package GPU quality test; set RUN_GPU_RENDER_TESTS=1 to enable"
            );
            return;
        }

        let package = NativePackageFixture::write();
        let asset_root = package.root.to_string_lossy().into_owned();
        let upload_budget = LodAtlasUploadBudget::try_new(package.canonical_slot_bytes, 1)
            .expect("one package slot is a valid global staging budget");
        let package_config = GaussianLodPackageConfig {
            max_atlas_gaussians: 8_192,
            max_atlas_bytes: 32 * 1024 * 1024,
            streaming: GaussianStreamingSettings {
                max_concurrent_requests: 1,
                retry_limit: 0,
                ..default()
            },
            ..default()
        };

        let mut app = App::new();
        app.insert_resource(ClearColor(Color::BLACK))
            .insert_resource(GaussianLodBridgeConfig {
                auto_build_flat_clouds: false,
                ..default()
            })
            .insert_resource(package_config)
            .insert_resource(upload_budget)
            .insert_resource(IndirectProbe::default())
            .insert_resource(GardenViewBlendRenderProbe::default())
            .insert_resource(PackageQualityRenderState::new(&package));
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: asset_root.clone(),
                    processed_file_path: asset_root,
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
                // Keep one app update equal to one completed render update so
                // package activation, indirect readback, and the in-memory
                // target capture describe the same frame. The pipelined
                // renderer may legitimately capture the pre-activation clear
                // while the main world has already observed the shared ACTIVE
                // token.
                .disable::<PipelinedRenderingPlugin>()
                .disable::<bevy::log::LogPlugin>(),
        );
        app.add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 120.0,
        )));
        app.add_plugins((
            GaussianSplattingPlugin,
            ExtractResourcePlugin::<IndirectProbe>::default(),
            ExtractResourcePlugin::<GardenViewBlendRenderProbe>::default(),
        ))
        .add_systems(Startup, setup_native_package_quality)
        .add_systems(Update, drive_native_package_quality)
        .add_observer(on_native_package_capture);
        app.sub_app_mut(RenderApp).add_systems(
            Render,
            (
                read_ready_lod_indirect_args,
                capture_garden_view_blend_render_state.after(LodViewBlendPublicationLabel),
            )
                .in_set(RenderSystems::Cleanup),
        );
        let exit = app.run();
        assert!(
            exit.is_success(),
            "native package quality app failed: {exit:?}"
        );
    }

    #[test]
    #[ignore = "requires the canonical Garden package via BGS_GARDEN_LOD"]
    fn canonical_garden_manifest_cpu_selector_has_useful_distance_response() {
        let manifest_path = PathBuf::from(
            env::var_os("BGS_GARDEN_LOD")
                .unwrap_or_else(|| panic!("set BGS_GARDEN_LOD to Garden's scene.gsplatlod")),
        );
        let encoded = fs::read(&manifest_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", manifest_path.display()));
        assert_canonical_garden_manifest_bytes(&encoded);
        let manifest = decode_manifest(&encoded, LodCodecLimits::default())
            .expect("Garden manifest decodes and validates");
        assert_canonical_garden_manifest(&manifest);
        let hierarchy = ManifestLodHierarchy::new(&manifest).expect("Garden hierarchy compiles");
        let bounds = manifest.scene_bounds.expect("Garden has scene bounds");
        let center = Vec3::from_array(bounds.center());
        let radius = bounds.radius();
        let view_direction = Vec3::new(0.0, 1.5, 5.0).normalize();
        const VIEWPORT_HEIGHT: f32 = 1080.0;
        const VIEWPORT_ASPECT: f32 = 16.0 / 9.0;
        const VIEWER_FOV: f32 = std::f32::consts::FRAC_PI_4;
        const VIEWER_NEAR: f32 = 0.1;
        assert_eq!(bounds.min, GARDEN_SCENE_MIN);
        assert_eq!(bounds.max, GARDEN_SCENE_MAX);
        assert_eq!(center, Vec3::from_array(GARDEN_SCENE_CENTER));
        assert_eq!(radius, GARDEN_SCENE_RADIUS);
        assert!((GARDEN_AUTO_FRAME_DISTANCE / radius - 2.177_309_3).abs() < 1e-6);
        eprintln!(
            "Garden ABI16 authenticated bounds: min=[{:.9}, {:.9}, {:.9}], max=[{:.9}, {:.9}, {:.9}], center=[{:.9}, {:.9}, {:.9}], radius={:.9}, auto_frame={:.9}, auto_frame/R={:.9}",
            bounds.min[0],
            bounds.min[1],
            bounds.min[2],
            bounds.max[0],
            bounds.max[1],
            bounds.max[2],
            center.x,
            center.y,
            center.z,
            radius,
            GARDEN_AUTO_FRAME_DISTANCE,
            GARDEN_AUTO_FRAME_DISTANCE / radius,
        );

        // Exact ABI-16 / MomentMerge-v4 counts measured from the authenticated
        // fresh package. These deliberately replace, rather than inherit, the
        // old ABI-15/v3 count table.
        let cases = [
            (
                GARDEN_AUTO_FRAME_DISTANCE,
                [377_333_u64, 3_140_911, 5_806_112],
            ),
            (2.4 * radius, [334_325, 2_875_744, 5_784_608]),
            (4.0 * radius, [80_388, 1_474_692, 5_254_176]),
            (6.0 * radius, [6_801, 773_288, 4_673_631]),
            (10.0 * radius, [179, 556_519, 2_783_512]),
        ];
        let mut count_mismatches = Vec::new();
        let mut previous_counts: Option<[u64; 3]> = None;
        for (distance, expected_counts) in cases {
            let camera_position = center + view_direction * distance;
            let clip_from_world =
                Mat4::perspective_infinite_reverse_rh(VIEWER_FOV, VIEWPORT_ASPECT, VIEWER_NEAR)
                    * Mat4::look_at_rh(camera_position, center, Vec3::Y);
            let view =
                LodView::perspective(camera_position, VIEWPORT_HEIGHT, VIEWER_FOV, VIEWER_NEAR)
                    .with_clip_from_world(clip_from_world);
            let mut counts = [0_u64; 3];
            for (index, quality) in [0.35_f32, 0.50, 0.65].into_iter().enumerate() {
                let mut settings = GaussianLodSettings {
                    quality,
                    hysteresis: 0.0,
                    ..default()
                };
                settings.budgets.max_active_gaussians = 8_000_000;
                settings.budgets.max_traversal_nodes_per_view = manifest.header.node_count;
                let selected = select_frontier_with_visibility(
                    &hierarchy,
                    &AllResident,
                    view,
                    &settings,
                    |_, metrics| view.node_is_visible(metrics, 0.0),
                )
                .expect("Garden CPU selection succeeds");
                counts[index] = selected.status.active_gaussians;
                assert!(selected.requested_nodes.is_empty());
            }
            eprintln!(
                "Garden ABI16 CPU selector: distance/R={:.6}, q35={}, q50={}, q65={}",
                distance / radius,
                counts[0],
                counts[1],
                counts[2],
            );
            if counts != expected_counts {
                count_mismatches.push((distance / radius, expected_counts, counts));
            }
            assert!(
                counts.windows(2).all(|pair| pair[0] <= pair[1]),
                "Garden quality response regressed at distance/R={}: {counts:?}",
                distance / radius,
            );
            assert!(
                counts[2] > 0 && counts[2] < GARDEN_SOURCE_GAUSSIANS,
                "Garden q=.65 must retain a nonempty reduced cut at distance/R={}: {}",
                distance / radius,
                counts[2],
            );
            if let Some(previous_counts) = previous_counts {
                assert!(
                    counts
                        .iter()
                        .zip(previous_counts)
                        .all(|(current, previous)| *current <= previous),
                    "Garden distance response refined while moving away: previous={previous_counts:?}, current={counts:?}",
                );
            }
            previous_counts = Some(counts);
        }
        assert!(
            count_mismatches.is_empty(),
            "Garden ABI-16 count table drifted after printing every distance row: {count_mismatches:#?}"
        );
    }

    #[test]
    #[ignore = "requires the canonical Garden package and PLY via BGS_GARDEN_LOD/BGS_GARDEN_PLY"]
    fn canonical_garden_package_static_camera_converges_without_cut_churn() {
        let manifest_path = PathBuf::from(
            env::var_os("BGS_GARDEN_LOD")
                .unwrap_or_else(|| panic!("set BGS_GARDEN_LOD to Garden's scene.gsplatlod")),
        );
        let source_path = PathBuf::from(
            env::var_os("BGS_GARDEN_PLY")
                .unwrap_or_else(|| panic!("set BGS_GARDEN_PLY to the canonical Garden PLY")),
        );
        assert!(
            manifest_path.is_file(),
            "missing {}",
            manifest_path.display()
        );
        assert_canonical_garden_source(&source_path);
        let manifest = load_canonical_garden_manifest(&manifest_path);
        let scene_frame = garden_scene_frame(&manifest);
        let node_parents = garden_node_parents(&manifest);
        let package_root = manifest_path
            .parent()
            .expect("Garden manifest has a package directory")
            .to_path_buf();
        let manifest_name = manifest_path
            .file_name()
            .expect("Garden manifest has a file name")
            .to_string_lossy()
            .into_owned();

        let mut settings = GaussianLodSettings {
            quality: garden_env_f32("BGS_GARDEN_QUALITY", 0.65),
            ..default()
        };
        settings.budgets.max_active_gaussians = garden_env_u64(
            "BGS_GARDEN_MAX_ACTIVE_GAUSSIANS",
            GARDEN_VIEWER_MAX_ACTIVE_GAUSSIANS,
        );
        assert!(
            settings.budgets.max_active_gaussians <= settings.budgets.max_resident_gaussians,
            "Garden max-active override cannot exceed the authenticated test atlas capacity"
        );
        const RECORDS_PER_PAGE: u64 = 1_024;
        settings.budgets.max_resident_pages = (settings.budgets.max_resident_gaussians
            / RECORDS_PER_PAGE)
            .try_into()
            .expect("Garden viewer page capacity fits u32");
        let package_config = GaussianLodPackageConfig {
            max_atlas_gaussians: settings
                .budgets
                .max_resident_gaussians
                .try_into()
                .expect("Garden viewer atlas record capacity fits u32"),
            max_atlas_bytes: settings.budgets.max_resident_bytes,
            ..default()
        };
        let mut app = App::new();
        app.insert_resource(ClearColor(Color::BLACK))
            .insert_resource(GaussianLodBridgeConfig {
                auto_build_flat_clouds: false,
                ..default()
            })
            .insert_resource(package_config)
            .insert_resource(GardenViewBlendRenderProbe::default())
            .insert_resource(GardenPackageStaticState::new(
                package_root.clone(),
                manifest_name,
                source_path,
                settings,
                scene_frame,
                node_parents,
            ));
        let asset_root = package_root.to_string_lossy().into_owned();
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: asset_root.clone(),
                    processed_file_path: asset_root,
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
            1.0 / 120.0,
        )))
        .add_plugins((
            GaussianSplattingPlugin,
            ExtractResourcePlugin::<GardenViewBlendRenderProbe>::default(),
        ))
        .add_systems(Startup, setup_garden_package_static)
        // Package orchestration runs in PostUpdate. Inspect the fully updated
        // cut and upload queue in Last so a same-frame replacement cannot race
        // an offscreen sample requested from an earlier schedule.
        .add_systems(Last, drive_garden_package_static)
        .add_observer(on_garden_package_capture);
        app.sub_app_mut(RenderApp).add_systems(
            Render,
            capture_garden_view_blend_render_state
                .after(LodViewBlendPublicationLabel)
                .in_set(RenderSystems::Cleanup),
        );
        let exit = app.run();
        assert!(exit.is_success(), "Garden package app failed: {exit:?}");
    }

    #[test]
    #[ignore = "requires the canonical Garden package and PLY via BGS_GARDEN_LOD/BGS_GARDEN_PLY"]
    fn canonical_garden_package_interactive_lod_is_monotonic_stable_and_spatially_faithful() {
        let manifest_path = PathBuf::from(
            env::var_os("BGS_GARDEN_LOD")
                .unwrap_or_else(|| panic!("set BGS_GARDEN_LOD to Garden's scene.gsplatlod")),
        );
        let source_path = PathBuf::from(
            env::var_os("BGS_GARDEN_PLY")
                .unwrap_or_else(|| panic!("set BGS_GARDEN_PLY to the canonical Garden PLY")),
        );
        assert!(
            manifest_path.is_file(),
            "missing {}",
            manifest_path.display()
        );
        assert_canonical_garden_source(&source_path);
        let _authenticated_manifest = load_canonical_garden_manifest(&manifest_path);
        let package_root = manifest_path
            .parent()
            .expect("Garden manifest has a package directory")
            .to_path_buf();
        let manifest_name = manifest_path
            .file_name()
            .expect("Garden manifest has a file name")
            .to_string_lossy()
            .into_owned();

        let mut settings = GaussianLodSettings {
            quality: GardenInteractiveScenario::NearLowCold.quality(),
            hysteresis: VIEWER_DEFAULT_LOD_HYSTERESIS,
            ..default()
        };
        settings.hysteresis = garden_env_f32("BGS_GARDEN_HYSTERESIS", settings.hysteresis);
        assert_eq!(
            settings.hysteresis, 0.0,
            "interactive Garden exact-reversibility qualification requires zero hysteresis"
        );
        settings.budgets.max_active_gaussians = garden_env_u64(
            "BGS_GARDEN_MAX_ACTIVE_GAUSSIANS",
            GARDEN_INTERACTIVE_MAX_ACTIVE_GAUSSIANS,
        );
        assert!(
            settings.budgets.max_active_gaussians <= settings.budgets.max_resident_gaussians,
            "Garden max-active override cannot exceed the authenticated test atlas capacity"
        );
        const RECORDS_PER_PAGE: u64 = 1_024;
        settings.budgets.max_resident_pages = (settings.budgets.max_resident_gaussians
            / RECORDS_PER_PAGE)
            .try_into()
            .expect("Garden viewer page capacity fits u32");
        let max_concurrent_requests = garden_env_u64("BGS_GARDEN_MAX_CONCURRENT_REQUESTS", 64)
            .try_into()
            .expect("Garden request concurrency fits u32");
        let package_config = GaussianLodPackageConfig {
            max_atlas_gaussians: settings
                .budgets
                .max_resident_gaussians
                .try_into()
                .expect("Garden viewer atlas record capacity fits u32"),
            max_atlas_bytes: settings.budgets.max_resident_bytes,
            streaming: GaussianStreamingSettings {
                max_concurrent_requests,
                ..default()
            },
            ..default()
        };
        eprintln!(
            "Garden interactive package policy: hysteresis={}, max_active={}, max_concurrent_requests={max_concurrent_requests}",
            settings.hysteresis, settings.budgets.max_active_gaussians
        );

        let mut app = App::new();
        app.insert_resource(ClearColor(Color::BLACK))
            .insert_resource(GaussianLodBridgeConfig {
                auto_build_flat_clouds: false,
                ..default()
            })
            .insert_resource(package_config)
            .insert_resource(GardenViewBlendRenderProbe::ordered())
            .insert_resource(GardenInteractiveState::new(
                package_root.clone(),
                manifest_name,
                source_path,
                settings,
            ));
        let asset_root = package_root.to_string_lossy().into_owned();
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: asset_root.clone(),
                    processed_file_path: asset_root,
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
            1.0 / 120.0,
        )))
        .add_plugins((
            GaussianSplattingPlugin,
            ExtractResourcePlugin::<GardenViewBlendRenderProbe>::default(),
        ))
        .add_systems(Startup, setup_garden_interactive)
        .add_systems(Last, drive_garden_interactive)
        .add_observer(on_garden_interactive_capture);
        app.sub_app_mut(RenderApp).add_systems(
            Render,
            capture_garden_view_blend_render_state
                .after(LodViewBlendPublicationLabel)
                .in_set(RenderSystems::Cleanup),
        );
        let exit = app.run();
        assert!(
            exit.is_success(),
            "interactive Garden package app failed: {exit:?}"
        );
    }

    #[test]
    #[ignore = "requires a real GPU plus the canonical Garden package and PLY"]
    fn canonical_garden_continuous_dolly_has_no_one_frame_lod_transition_spike() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping Garden temporal-dolly gate; set RUN_GPU_RENDER_TESTS=1");
            return;
        }
        let manifest_path = PathBuf::from(
            env::var_os("BGS_GARDEN_LOD")
                .unwrap_or_else(|| panic!("set BGS_GARDEN_LOD to Garden's scene.gsplatlod")),
        );
        let source_path = PathBuf::from(
            env::var_os("BGS_GARDEN_PLY")
                .unwrap_or_else(|| panic!("set BGS_GARDEN_PLY to the canonical Garden PLY")),
        );
        assert!(
            manifest_path.is_file(),
            "missing {}",
            manifest_path.display()
        );
        assert_canonical_garden_source(&source_path);
        let temporal_phase = env::var("BGS_GARDEN_TEMPORAL_PHASE").ok();
        if temporal_phase.as_deref() == Some("roundtrip") {
            assert_garden_dynamic_view_blend_roundtrip(&manifest_path);
            return;
        }

        // Matching flat and fixed-cut traces in both zoom directions are
        // intentionally evaluated before the dynamic post-fix traces.
        // `baseline` mode reports characterization evidence without claiming
        // that the provisional limits below have already been calibrated on
        // this package and adapter.
        let baselines = GardenTemporalDollyDirection::ALL.map(|direction| {
            let flat = capture_garden_flat_temporal_dolly(&source_path, &manifest_path, direction);
            let frozen = capture_garden_package_temporal_dolly(
                &manifest_path,
                LodSelectionMode::Frozen,
                direction,
            );
            eprintln!(
                "Garden temporal {direction:?} flat baseline: {}",
                flat.summary()
            );
            eprintln!(
                "Garden temporal {direction:?} fixed-cut (LodSelectionMode::Frozen) baseline: {}",
                frozen.summary()
            );
            (direction, flat, frozen)
        });
        if temporal_phase.as_deref() == Some("baseline") {
            return;
        }

        for (direction, flat, frozen) in baselines {
            let dynamic = capture_garden_package_temporal_dolly(
                &manifest_path,
                LodSelectionMode::Dynamic,
                direction,
            );
            eprintln!(
                "Garden temporal {direction:?} Dynamic-LoD trace: {}",
                dynamic.summary()
            );
            assert_garden_dynamic_temporal_trace(direction, &flat, &frozen, &dynamic);
        }
        assert_garden_dynamic_view_blend_roundtrip(&manifest_path);
    }

    #[test]
    #[ignore = "requires a real GPU plus the canonical Garden package and PLY"]
    fn canonical_garden_debug_colors_use_bounded_sparse_uploads_without_rebuilding_the_cut() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping Garden debug-color GPU gate; set RUN_GPU_RENDER_TESTS=1 to enable");
            return;
        }

        let manifest_path = PathBuf::from(
            env::var_os("BGS_GARDEN_LOD")
                .unwrap_or_else(|| panic!("set BGS_GARDEN_LOD to Garden's scene.gsplatlod")),
        );
        let source_path = PathBuf::from(
            env::var_os("BGS_GARDEN_PLY")
                .unwrap_or_else(|| panic!("set BGS_GARDEN_PLY to the canonical Garden PLY")),
        );
        assert!(
            manifest_path.is_file(),
            "missing {}",
            manifest_path.display()
        );
        assert_canonical_garden_source(&source_path);
        let _authenticated_manifest = load_canonical_garden_manifest(&manifest_path);
        let package_root = manifest_path
            .parent()
            .expect("Garden manifest has a package directory")
            .to_path_buf();
        let manifest_name = manifest_path
            .file_name()
            .expect("Garden manifest has a file name")
            .to_string_lossy()
            .into_owned();

        let mut settings = GaussianLodSettings {
            quality: GARDEN_INTERACTIVE_REVIEW_QUALITY,
            hysteresis: VIEWER_DEFAULT_LOD_HYSTERESIS,
            ..default()
        };
        settings.hysteresis = garden_env_f32("BGS_GARDEN_HYSTERESIS", settings.hysteresis);
        settings.budgets.max_active_gaussians = garden_env_u64(
            "BGS_GARDEN_MAX_ACTIVE_GAUSSIANS",
            GARDEN_INTERACTIVE_MAX_ACTIVE_GAUSSIANS,
        );
        assert!(
            settings.budgets.max_active_gaussians <= settings.budgets.max_resident_gaussians,
            "Garden max-active override cannot exceed the authenticated test atlas capacity"
        );
        const RECORDS_PER_PAGE: u64 = 1_024;
        settings.budgets.max_resident_pages = (settings.budgets.max_resident_gaussians
            / RECORDS_PER_PAGE)
            .try_into()
            .expect("Garden viewer page capacity fits u32");
        let max_concurrent_requests = garden_env_u64("BGS_GARDEN_MAX_CONCURRENT_REQUESTS", 64)
            .try_into()
            .expect("Garden request concurrency fits u32");
        let package_config = GaussianLodPackageConfig {
            max_atlas_gaussians: settings
                .budgets
                .max_resident_gaussians
                .try_into()
                .expect("Garden viewer atlas record capacity fits u32"),
            max_atlas_bytes: settings.budgets.max_resident_bytes,
            streaming: GaussianStreamingSettings {
                max_concurrent_requests,
                ..default()
            },
            ..default()
        };

        let probe = GardenDebugIndirectProbe::default();
        let mut app = App::new();
        app.insert_resource(ClearColor(Color::NONE))
            .insert_resource(GaussianLodBridgeConfig {
                auto_build_flat_clouds: false,
                ..default()
            })
            .insert_resource(package_config)
            .insert_resource(probe.clone())
            .init_resource::<GardenDebugPixelCaptureSink>();
        let asset_root = package_root.to_string_lossy().into_owned();
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: asset_root.clone(),
                    processed_file_path: asset_root,
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
                .disable::<PipelinedRenderingPlugin>()
                .disable::<bevy::log::LogPlugin>(),
        );
        app.add_plugins((
            GaussianSplattingPlugin,
            ExtractResourcePlugin::<GardenDebugIndirectProbe>::default(),
        ));
        app.add_observer(on_garden_debug_pixel_capture);
        while app.plugins_state() == PluginsState::Adding {
            std::thread::yield_now();
        }
        app.finish();
        app.cleanup();
        // The manually-driven headless app creates and finalizes RenderApp as
        // part of plugin completion. Register the probe only after that
        // lifecycle boundary instead of assuming the sub-app exists while the
        // plugin group is still Adding.
        app.sub_app_mut(RenderApp).add_systems(
            Render,
            read_garden_debug_indirect_args.in_set(RenderSystems::Cleanup),
        );

        let target =
            app.world_mut()
                .resource_mut::<Assets<Image>>()
                .add(Image::new_target_texture(
                    GARDEN_TARGET_WIDTH,
                    GARDEN_TARGET_HEIGHT,
                    TextureFormat::Rgba8UnormSrgb,
                    None,
                ));
        let manifest: Handle<GaussianLodAsset> =
            app.world().resource::<AssetServer>().load(manifest_name);
        let cloud = app
            .world_mut()
            .spawn((
                GaussianLodHandle(manifest.clone()),
                GaussianLodPackageSource::native_directory(
                    package_root.to_string_lossy().into_owned(),
                ),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::Radix,
                    radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                    lod_debug: LodDebugSettings::from_preset(LodDebugPreset::Off),
                    ..default()
                },
                settings,
                Transform::IDENTITY,
                Visibility::Visible,
                Name::new("canonical_garden_debug_upload_package"),
            ))
            .id();
        let camera = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                Camera {
                    is_active: false,
                    ..default()
                },
                Projection::Perspective(PerspectiveProjection {
                    far: 1_000_000.0,
                    ..default()
                }),
                RenderTarget::Image(target.clone().into()),
                Transform::IDENTITY,
                Tonemapping::None,
                GaussianCamera::default(),
                Name::new("canonical_garden_debug_upload_camera"),
            ))
            .id();

        let mut manifest_frames = 0_u32;
        let (scene_frame, records_per_slot) = loop {
            manifest_frames += 1;
            assert!(
                manifest_frames <= 7_200,
                "Garden debug manifest did not load within the frame bound"
            );
            garden_debug_step(&mut app, None, "manifest loading");
            let manifest_info = app
                .world()
                .resource::<Assets<GaussianLodAsset>>()
                .get(&manifest)
                .map(|asset| {
                    assert_canonical_garden_manifest(asset.manifest());
                    let bounds = asset
                        .manifest()
                        .scene_bounds
                        .expect("authenticated Garden manifest carries scene bounds");
                    let records_per_slot = asset
                        .manifest()
                        .pages
                        .iter()
                        .map(|page| page.gaussian_count)
                        .max()
                        .expect("Garden manifest has pages");
                    (
                        GardenSceneFrame {
                            center: Vec3::from_array(bounds.center()),
                            radius: bounds.radius(),
                        },
                        records_per_slot,
                    )
                });
            if let Some(info) = manifest_info {
                break info;
            }
        };
        assert!(
            scene_frame.center.is_finite()
                && scene_frame.radius.is_finite()
                && scene_frame.radius > 0.0,
            "Garden scene frame is invalid: {scene_frame:?}"
        );
        *app.world_mut()
            .get_mut::<Transform>(camera)
            .expect("Garden debug camera transform exists") =
            scene_frame.transform(GardenInteractivePose::Overview);
        app.world_mut()
            .get_mut::<Camera>(camera)
            .expect("Garden debug camera exists")
            .is_active = true;
        let debug_slot_bytes = u64::from(records_per_slot)
            .checked_mul(size_of::<LodDebugRecord>() as u64)
            .expect("Garden debug slot byte count fits u64");

        let (off_cut, _) = wait_for_stable_garden_debug_cut(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Off,
            debug_slot_bytes,
            None,
            None,
            false,
            "initial debug-off cut",
        );
        assert_stationary_garden_debug_window(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Off,
            debug_slot_bytes,
            &off_cut,
            "debug-off stationary baseline",
        );

        app.world_mut()
            .get_mut::<CloudSettings>(cloud)
            .expect("Garden cloud settings exist")
            .lod_debug
            .apply_preset(LodDebugPreset::Level);
        let (level_cut, level_enable) = wait_for_stable_garden_debug_cut(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Level,
            debug_slot_bytes,
            Some(&off_cut),
            None,
            false,
            "debug Off-to-Level initialization",
        );
        assert_eq!(level_enable.record_buffer_allocations, 1);
        assert_eq!(
            level_enable.config_bytes_written,
            LodDebugGpuUploadStats::config_bytes_per_write()
        );
        assert!(
            level_enable.record_bytes_written > 0,
            "Garden debug initialization uploaded no annotation records"
        );
        let metadata_record_count = {
            let metadata = app
                .world()
                .get::<LodDebugMetadata>(cloud)
                .expect("active Garden debug view publishes metadata");
            assert!(
                metadata.records().is_empty(),
                "streamed Garden debug metadata regressed to a dense CPU atlas"
            );
            metadata.len()
        };
        let full_atlas_bytes = (metadata_record_count as u64)
            .checked_mul(size_of::<LodDebugRecord>() as u64)
            .expect("Garden full debug-atlas byte count fits u64");
        assert!(
            level_enable.record_bytes_written < full_atlas_bytes,
            "Garden debug initialization rewrote the full atlas: wrote={}, capacity={full_atlas_bytes}",
            level_enable.record_bytes_written
        );
        assert_stationary_garden_debug_window(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Level,
            debug_slot_bytes,
            &level_cut,
            "Level stationary window",
        );
        assert_garden_debug_pipeline_active(&mut app, debug_slot_bytes, "Level debug pipeline");
        let mut indirect_request = 0_u64;
        let level_drawn = assert_garden_debug_indirect_draw(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Level,
            debug_slot_bytes,
            &level_cut,
            &probe,
            &mut indirect_request,
        );

        let page_cut = apply_garden_debug_preset_without_rebuilding_cut(
            &mut app,
            cloud,
            camera,
            debug_slot_bytes,
            &level_cut,
            LodDebugPreset::Page,
        );
        assert_stationary_garden_debug_window(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Page,
            debug_slot_bytes,
            &page_cut,
            "Page stationary window",
        );
        assert_garden_debug_pipeline_active(&mut app, debug_slot_bytes, "Page debug pipeline");

        *app.world_mut()
            .get_mut::<Transform>(camera)
            .expect("Garden debug camera transform exists") =
            scene_frame.transform(GardenInteractivePose::Closer);
        let (closer_cut, camera_move) = wait_for_stable_garden_debug_cut(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Page,
            debug_slot_bytes,
            None,
            Some(&page_cut),
            true,
            "Page debug camera movement",
        );
        assert_eq!(
            camera_move.record_buffer_allocations, 0,
            "camera movement recreated the Garden debug record buffer"
        );
        assert_eq!(
            camera_move.config_bytes_written, 0,
            "camera movement rewrote the presentation-only debug uniform"
        );
        assert!(
            camera_move.record_bytes_written > 0,
            "camera movement changed the Garden cut without sparse annotation uploads"
        );
        assert_eq!(
            closer_cut.atlas, page_cut.atlas,
            "camera movement replaced the package atlas"
        );
        assert_ne!(
            closer_cut.logical_signature(),
            page_cut.logical_signature(),
            "scripted closer camera did not exercise a different Garden cut"
        );
        assert_stationary_garden_debug_window(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Page,
            debug_slot_bytes,
            &closer_cut,
            "settled closer Page window",
        );
        let closer_drawn = assert_garden_debug_indirect_draw(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Page,
            debug_slot_bytes,
            &closer_cut,
            &probe,
            &mut indirect_request,
        );

        let residency_closer_cut = apply_garden_debug_preset_without_rebuilding_cut(
            &mut app,
            cloud,
            camera,
            debug_slot_bytes,
            &closer_cut,
            LodDebugPreset::Residency,
        );
        assert_stationary_garden_debug_window(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Residency,
            debug_slot_bytes,
            &residency_closer_cut,
            "Residency stationary window",
        );
        assert_garden_debug_pipeline_active(&mut app, debug_slot_bytes, "Residency debug pipeline");

        *app.world_mut()
            .get_mut::<Transform>(camera)
            .expect("Garden debug camera transform exists") =
            scene_frame.transform(GardenInteractivePose::Overview);
        let (residency_overview_cut, residency_camera_move) = wait_for_stable_garden_debug_cut(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Residency,
            debug_slot_bytes,
            None,
            Some(&residency_closer_cut),
            true,
            "Residency debug camera movement",
        );
        assert_eq!(
            residency_camera_move.record_buffer_allocations, 0,
            "Residency camera movement recreated the Garden debug record buffer"
        );
        assert_eq!(
            residency_camera_move.config_bytes_written, 0,
            "Residency camera movement rewrote the presentation-only debug uniform"
        );
        assert_eq!(
            residency_overview_cut.atlas, residency_closer_cut.atlas,
            "Residency camera movement replaced the package atlas"
        );
        assert_ne!(
            residency_overview_cut.logical_signature(),
            residency_closer_cut.logical_signature(),
            "scripted Residency camera movement did not activate a different Garden candidate"
        );
        assert_stationary_garden_debug_window(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Residency,
            debug_slot_bytes,
            &residency_overview_cut,
            "settled overview Residency window",
        );
        assert_garden_debug_pipeline_active(
            &mut app,
            debug_slot_bytes,
            "settled Residency debug pipeline",
        );

        let pressure_cut = apply_garden_debug_preset_without_rebuilding_cut(
            &mut app,
            cloud,
            camera,
            debug_slot_bytes,
            &residency_overview_cut,
            LodDebugPreset::SelectionPressure,
        );
        let _pressure_drawn = assert_garden_debug_indirect_draw(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::SelectionPressure,
            debug_slot_bytes,
            &pressure_cut,
            &probe,
            &mut indirect_request,
        );
        assert_stationary_garden_debug_window(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::SelectionPressure,
            debug_slot_bytes,
            &pressure_cut,
            "SelectionPressure stationary window",
        );
        assert_garden_debug_pipeline_active(
            &mut app,
            debug_slot_bytes,
            "SelectionPressure debug pipeline",
        );

        let off_before_page_cut = apply_garden_debug_preset_without_rebuilding_cut(
            &mut app,
            cloud,
            camera,
            debug_slot_bytes,
            &pressure_cut,
            LodDebugPreset::Off,
        );
        assert_stationary_garden_debug_window(
            &mut app,
            cloud,
            camera,
            LodDebugPreset::Off,
            debug_slot_bytes,
            &off_before_page_cut,
            "Off baseline before Page pixel transition",
        );
        let debug_pixel_contract = assert_garden_debug_off_page_off_pixel_contract(
            &mut app,
            cloud,
            camera,
            &target,
            debug_slot_bytes,
            &off_before_page_cut,
            &probe,
            &mut indirect_request,
        );

        eprintln!(
            "Garden debug sparse gate: atlas_records={}, initial_record_bytes={}, page_camera_record_bytes={}, residency_camera_record_bytes={}, per_frame_cap={}, slot_cap={}, overview_pre_cull={}, overview_post_frustum={}, closer_pre_cull={}, closer_post_frustum={}, pixel_frames={}, page_transition_frames={}, off_restore_frames={}, page_supported_hue_bins={}, page_changed_pixels={}, page_changed_fraction={:.6}, page_stability={:?}, off_restore={:?}",
            metadata_record_count,
            level_enable.record_bytes_written,
            camera_move.record_bytes_written,
            residency_camera_move.record_bytes_written,
            LodDebugGpuUploadStats::max_sparse_record_bytes_per_frame(),
            LodDebugGpuUploadStats::max_sparse_record_slots_per_frame(),
            page_cut.candidate_count,
            level_drawn,
            closer_cut.candidate_count,
            closer_drawn,
            debug_pixel_contract.captured_frames,
            debug_pixel_contract.page_transition_frames,
            debug_pixel_contract.off_restore_frames,
            debug_pixel_contract.page_supported_hue_bins,
            debug_pixel_contract.page_changed_pixels,
            debug_pixel_contract.page_changed_fraction,
            debug_pixel_contract.page_stability,
            debug_pixel_contract.off_restore,
        );
    }

    type GardenDebugRangeSignature = (u64, u64, u32, u32, u32, u32);

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct GardenDebugCutObservation {
        atlas: AssetId<PlanarGaussian3d>,
        render_commit: usize,
        candidate_count: u32,
        resident_pages: u32,
        ranges: Vec<GardenDebugRangeSignature>,
        required_ranges: Vec<GardenDebugRangeSignature>,
        blend_weights: Vec<(u64, Vec<u64>, u32, u32)>,
    }

    impl GardenDebugCutObservation {
        fn logical_signature(&self) -> (u32, &[GardenDebugRangeSignature]) {
            (self.candidate_count, &self.ranges)
        }
    }

    #[derive(Clone, Debug)]
    struct GardenDebugPendingTelemetry {
        camera_active: bool,
        camera_visible_entities_present: bool,
        cloud_visible_to_camera: bool,
        cloud_view_visible: Option<bool>,
        phase: &'static str,
        render_claimed: bool,
        authored_temporal_mode: Option<LodTemporalTransitionMode>,
        effective_temporal_mode: Option<LodTemporalTransitionMode>,
        temporal_progress: Option<f32>,
        transition_identity: Option<String>,
        target_ranges: (usize, u64, u64),
        required_ranges: (usize, u64, u64),
        current_ranges: (usize, u64, u64),
    }

    impl GardenDebugPendingTelemetry {
        fn summary(&self) -> String {
            format!(
                "camera_active={}, visible_entities={}, cloud_visible={}, view_visible={:?}, phase={}, claimed={}, authored_temporal={:?}, effective_temporal={:?}, progress={:?}, transition={}, target={:?}, required={:?}, current={:?}",
                self.camera_active,
                self.camera_visible_entities_present,
                self.cloud_visible_to_camera,
                self.cloud_view_visible,
                self.phase,
                self.render_claimed,
                self.authored_temporal_mode,
                self.effective_temporal_mode,
                self.temporal_progress,
                self.transition_identity.as_deref().unwrap_or("none"),
                self.target_ranges,
                self.required_ranges,
                self.current_ranges,
            )
        }
    }

    fn garden_debug_pending_telemetry(
        app: &App,
        cloud: Entity,
        camera: Entity,
    ) -> Option<GardenDebugPendingTelemetry> {
        let world = app.world();
        let camera_component = world.get::<Camera>(camera)?;
        let visible_entities = world.get::<bevy::camera::visibility::VisibleEntities>(camera);
        let candidate = world.get::<LodRenderCandidates>(cloud)?.get(camera)?;
        let phase = if candidate.failed() {
            "FAILED"
        } else if candidate.render_is_active_for_testing() {
            "ACTIVE"
        } else if candidate.render_is_transitioning_for_testing() {
            "TRANSITIONING"
        } else if candidate.render_is_prepared() {
            "PREPARED"
        } else {
            "WAITING"
        };
        let transition = candidate.temporal_transition();
        Some(GardenDebugPendingTelemetry {
            camera_active: camera_component.is_active,
            camera_visible_entities_present: visible_entities.is_some(),
            cloud_visible_to_camera: visible_entities.is_none_or(|visible| {
                visible
                    .iter(TypeId::of::<CloudVisibilityClass>())
                    .any(|visible_cloud| *visible_cloud == cloud)
            }),
            cloud_view_visible: world
                .get::<ViewVisibility>(cloud)
                .map(|visible| visible.get()),
            phase,
            render_claimed: candidate.render_is_claimed_for_testing(),
            authored_temporal_mode: transition.map(|transition| transition.mode()),
            effective_temporal_mode: candidate.temporal_transition_mode(),
            temporal_progress: candidate.temporal_transition_progress(),
            transition_identity: transition
                .and_then(|transition| transition.morph())
                .map(|morph| format!("{:?}", morph.identity())),
            target_ranges: garden_debug_range_set_telemetry(candidate.frontier().physical_ranges()),
            required_ranges: garden_debug_range_set_telemetry(
                candidate.required_atlas_ranges_for_testing(),
            ),
            current_ranges: garden_debug_range_set_telemetry(candidate.render_ranges()),
        })
    }

    fn garden_debug_range_set_telemetry(
        ranges: &[bevy_gaussian_splatting::stream::runtime::LodPhysicalRange],
    ) -> (usize, u64, u64) {
        let mut digest = 0xcbf2_9ce4_8422_2325_u64;
        let mut count = 0_u64;
        for range in ranges {
            count = count.saturating_add(u64::from(range.count));
            for value in [
                range.node.0,
                range.page.0,
                u64::from(range.slot.index),
                u64::from(range.slot.generation),
                u64::from(range.physical_start),
                u64::from(range.count),
            ] {
                digest ^= value;
                digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        (ranges.len(), count, digest)
    }

    #[derive(Resource, Default)]
    struct GardenDebugPixelCaptureSink {
        pending: BTreeMap<Entity, u32>,
        images: BTreeMap<u32, Vec<u8>>,
        discard: BTreeSet<u32>,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum GardenDebugPixelPhase {
        OffBefore,
        PageRequested,
        OffRestored,
    }

    #[derive(Clone, Copy, Debug)]
    struct GardenDebugPixelFrameEvidence {
        sample: u32,
        phase: GardenDebugPixelPhase,
        status_preset: LodDebugPreset,
        availability: GaussianLodDebugAvailability,
        render_preset: LodDebugPreset,
        debug_binding_ready: bool,
        lod_debug_queued: bool,
        indirect_draw: u32,
    }

    #[derive(Debug)]
    struct GardenDebugPixelContractSummary {
        captured_frames: usize,
        page_transition_frames: usize,
        off_restore_frames: usize,
        page_supported_hue_bins: usize,
        page_changed_pixels: usize,
        page_changed_fraction: f64,
        page_stability: ImageMetrics,
        off_restore: ImageMetrics,
    }

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    struct GardenDebugUploadDelta {
        record_buffer_allocations: u64,
        record_bytes_written: u64,
        config_bytes_written: u64,
    }

    impl GardenDebugUploadDelta {
        fn between(before: LodDebugGpuUploadStats, after: LodDebugGpuUploadStats) -> Self {
            Self {
                record_buffer_allocations: after
                    .record_buffer_allocations()
                    .checked_sub(before.record_buffer_allocations())
                    .expect("Garden debug allocation counter is monotonic"),
                record_bytes_written: after
                    .record_bytes_written()
                    .checked_sub(before.record_bytes_written())
                    .expect("Garden debug record-byte counter is monotonic"),
                config_bytes_written: after
                    .config_bytes_written()
                    .checked_sub(before.config_bytes_written())
                    .expect("Garden debug config-byte counter is monotonic"),
            }
        }

        fn accumulate(&mut self, next: Self) {
            self.record_buffer_allocations = self
                .record_buffer_allocations
                .checked_add(next.record_buffer_allocations)
                .expect("Garden debug allocation delta fits u64");
            self.record_bytes_written = self
                .record_bytes_written
                .checked_add(next.record_bytes_written)
                .expect("Garden debug record-byte delta fits u64");
            self.config_bytes_written = self
                .config_bytes_written
                .checked_add(next.config_bytes_written)
                .expect("Garden debug config-byte delta fits u64");
        }
    }

    fn garden_debug_gpu_stats(app: &App) -> LodDebugGpuUploadStats {
        *app.sub_app(RenderApp)
            .world()
            .resource::<LodDebugGpuUploadStats>()
    }

    fn assert_garden_debug_pipeline_active(app: &mut App, debug_slot_bytes: u64, context: &str) {
        let before = garden_debug_gpu_stats(app);
        let upload = garden_debug_step(app, Some(debug_slot_bytes), context);
        assert_eq!(
            upload,
            GardenDebugUploadDelta::default(),
            "settled {context} performed unexpected upload/allocation work"
        );
        let after = garden_debug_gpu_stats(app);
        assert!(
            after.ready_bind_group_queues() > before.ready_bind_group_queues(),
            "{context} did not queue an upload-ready LoD debug bind group"
        );
        assert!(
            after.specialized_pipeline_queues() > before.specialized_pipeline_queues(),
            "{context} did not queue a pipeline specialized with LOD_DEBUG"
        );
    }

    fn garden_debug_step(
        app: &mut App,
        debug_slot_bytes: Option<u64>,
        context: &str,
    ) -> GardenDebugUploadDelta {
        let before = garden_debug_gpu_stats(app);
        app.update();
        let after = garden_debug_gpu_stats(app);
        let delta = GardenDebugUploadDelta::between(before, after);
        assert!(
            delta.record_bytes_written
                <= LodDebugGpuUploadStats::max_sparse_record_bytes_per_frame(),
            "Garden debug record upload exceeded the per-frame byte cap during {context}: {delta:?}"
        );
        if let (Some(slot_bytes), true) = (debug_slot_bytes, delta.record_bytes_written != 0) {
            assert_eq!(
                delta.record_bytes_written % slot_bytes,
                0,
                "Garden debug upload was not composed of exact sparse slots during {context}: {delta:?}, slot_bytes={slot_bytes}"
            );
            assert!(
                delta.record_bytes_written / slot_bytes
                    <= LodDebugGpuUploadStats::max_sparse_record_slots_per_frame() as u64,
                "Garden debug upload exceeded the per-frame sparse-slot cap during {context}: {delta:?}, slot_bytes={slot_bytes}"
            );
        }
        delta
    }

    fn observe_garden_debug_cut(
        app: &App,
        cloud: Entity,
        camera: Entity,
        expected_preset: LodDebugPreset,
    ) -> Option<GardenDebugCutObservation> {
        let world = app.world();
        let lod_status = world.get::<GaussianLodStatus>(cloud)?;
        if lod_status.debug_preset != expected_preset {
            return None;
        }
        let expected_availability = if expected_preset == LodDebugPreset::Off {
            GaussianLodDebugAvailability::Disabled
        } else {
            GaussianLodDebugAvailability::MetadataReady
        };
        if lod_status.debug_availability != expected_availability {
            return None;
        }
        observe_garden_debug_candidate(world, cloud, camera)
    }

    fn observe_garden_debug_active_cut(
        app: &App,
        cloud: Entity,
        camera: Entity,
    ) -> Option<GardenDebugCutObservation> {
        observe_garden_debug_candidate(app.world(), cloud, camera)
    }

    fn observe_garden_debug_candidate(
        world: &World,
        cloud: Entity,
        camera: Entity,
    ) -> Option<GardenDebugCutObservation> {
        let status = world.get::<GaussianLodPackageStatus>(cloud)?;
        assert!(
            status.failure.is_none(),
            "Garden debug package failed: {status:?}"
        );
        assert_eq!(status.terminal_failures, 0);
        if status.phase != GaussianLodPackagePhase::Active {
            return None;
        }
        if world.resource::<LodAtlasUploadQueue>().queued_slot_count() != 0 {
            return None;
        }
        let candidates = world.get::<LodRenderCandidates>(cloud)?;
        let candidate = candidates.get(camera)?;
        assert!(!candidate.failed(), "Garden debug candidate failed");
        if !candidate.render_is_active_for_testing()
            || candidate.rendered_candidate_count() as u64 != status.active_gaussians
            || candidate.rendered_quality_status().active_gaussians != status.active_gaussians
        {
            return None;
        }
        let handle = world.get::<PlanarGaussian3dHandle>(cloud)?;
        let blend = candidate.view_blend_testing_snapshot()?;
        let edges = candidate.view_blend()?.morph()?.edges();
        assert_eq!(edges.len(), blend.weights.len());
        if blend.status.lagging_count != 0
            || blend.status.missing_consumer_count != 0
            || blend
                .weights
                .iter()
                .any(|weight| weight.displayed.to_bits() != weight.desired.to_bits())
        {
            return None;
        }
        assert_eq!(
            blend.status.invalid_pressure_count, 0,
            "settled Garden debug cut reported invalid active pressure edges"
        );
        assert_eq!(
            blend.status.missing_consumer_count, 0,
            "settled Garden debug cut omitted a private render consumer"
        );
        assert_eq!(
            world
                .get::<GaussianLodStatus>(cloud)?
                .view_blend_invalid_pressure_evaluations,
            0,
            "settled Garden debug public status reported invalid pressure edges"
        );
        assert_eq!(
            world
                .get::<GaussianLodStatus>(cloud)?
                .view_blend_missing_consumers,
            0,
            "settled Garden debug public status reported a missing private consumer"
        );
        let ranges = candidate
            .render_ranges()
            .iter()
            .map(|range| {
                (
                    range.node.0,
                    range.page.0,
                    range.slot.index,
                    range.slot.generation,
                    range.physical_start,
                    range.count,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            ranges.iter().map(|range| u64::from(range.5)).sum::<u64>(),
            status.active_gaussians,
            "Garden debug candidate ranges and active count diverged"
        );
        Some(GardenDebugCutObservation {
            atlas: handle.0.id(),
            render_commit: candidate.render_commit_identity_for_testing(),
            candidate_count: candidate.rendered_candidate_count(),
            resident_pages: status.resident_pages,
            ranges,
            required_ranges: candidate
                .required_atlas_ranges_for_testing()
                .iter()
                .map(|range| {
                    (
                        range.node.0,
                        range.page.0,
                        range.slot.index,
                        range.slot.generation,
                        range.physical_start,
                        range.count,
                    )
                })
                .collect(),
            blend_weights: edges
                .iter()
                .zip(blend.weights)
                .map(|(edge, weight)| {
                    (
                        edge.parent().0,
                        edge.children().iter().map(|child| child.0).collect(),
                        weight.displayed.to_bits(),
                        weight.desired.to_bits(),
                    )
                })
                .collect(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn wait_for_stable_garden_debug_cut(
        app: &mut App,
        cloud: Entity,
        camera: Entity,
        expected_preset: LodDebugPreset,
        debug_slot_bytes: u64,
        preserve: Option<&GardenDebugCutObservation>,
        differ_from: Option<&GardenDebugCutObservation>,
        require_debug_pipeline_each_frame: bool,
        context: &str,
    ) -> (GardenDebugCutObservation, GardenDebugUploadDelta) {
        const REQUIRED_STABLE_FRAMES: u32 = 16;
        const MAX_PHASE_FRAMES: u32 = 7_200;
        let mut totals = GardenDebugUploadDelta::default();
        let mut stable_frames = 0_u32;
        let mut last = None;
        for frame in 0..MAX_PHASE_FRAMES {
            let before = garden_debug_gpu_stats(app);
            let delta = garden_debug_step(app, Some(debug_slot_bytes), context);
            let after = garden_debug_gpu_stats(app);
            if require_debug_pipeline_each_frame {
                assert!(
                    after.ready_bind_group_queues() > before.ready_bind_group_queues(),
                    "{context} dropped its upload-ready debug binding on transition frame {frame}: delta={delta:?}, render={:#?}",
                    garden_debug_render_telemetry(app),
                );
                assert!(
                    after.specialized_pipeline_queues() > before.specialized_pipeline_queues(),
                    "{context} dropped its LOD_DEBUG pipeline on transition frame {frame}: delta={delta:?}, render={:#?}",
                    garden_debug_render_telemetry(app),
                );
            }
            totals.accumulate(delta);
            if frame != 0 && frame % 600 == 0 {
                eprintln!(
                    "Garden debug pending telemetry during {context} at frame {frame}: {}",
                    garden_debug_pending_telemetry(app, cloud, camera)
                        .map_or_else(|| "none".to_owned(), |telemetry| telemetry.summary()),
                );
            }
            let Some(observation) = observe_garden_debug_cut(app, cloud, camera, expected_preset)
            else {
                stable_frames = 0;
                continue;
            };
            if let Some(expected) = preserve {
                assert_eq!(
                    &observation, expected,
                    "presentation-only debug work replaced the Garden package/candidate during {context}"
                );
            }
            if differ_from.is_some_and(|previous| {
                observation.logical_signature() == previous.logical_signature()
            }) {
                stable_frames = 0;
                last = Some(observation);
                continue;
            }
            let unchanged = last.as_ref() == Some(&observation);
            if unchanged && delta == GardenDebugUploadDelta::default() {
                stable_frames += 1;
            } else {
                stable_frames = 0;
            }
            last = Some(observation);
            if stable_frames >= REQUIRED_STABLE_FRAMES {
                return (last.expect("stable Garden debug cut exists"), totals);
            }
        }
        panic!(
            "Garden debug cut did not stabilize during {context}: last={last:?}, totals={totals:?}, package={:?}, lod={:?}, pending={}",
            app.world().get::<GaussianLodPackageStatus>(cloud),
            app.world().get::<GaussianLodStatus>(cloud),
            garden_debug_pending_telemetry(app, cloud, camera)
                .map_or_else(|| "none".to_owned(), |telemetry| telemetry.summary()),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_stationary_garden_debug_window(
        app: &mut App,
        cloud: Entity,
        camera: Entity,
        expected_preset: LodDebugPreset,
        debug_slot_bytes: u64,
        expected: &GardenDebugCutObservation,
        context: &str,
    ) {
        for _ in 0..16 {
            let delta = garden_debug_step(app, Some(debug_slot_bytes), context);
            assert_eq!(
                delta,
                GardenDebugUploadDelta::default(),
                "stationary Garden debug view performed upload/allocation work during {context}"
            );
            let observed = observe_garden_debug_cut(app, cloud, camera, expected_preset)
                .unwrap_or_else(|| panic!("Garden debug candidate disappeared during {context}"));
            assert_eq!(
                &observed, expected,
                "stationary Garden debug cut changed during {context}"
            );
        }
    }

    fn apply_garden_debug_preset_without_rebuilding_cut(
        app: &mut App,
        cloud: Entity,
        camera: Entity,
        debug_slot_bytes: u64,
        expected: &GardenDebugCutObservation,
        preset: LodDebugPreset,
    ) -> GardenDebugCutObservation {
        app.world_mut()
            .get_mut::<CloudSettings>(cloud)
            .expect("Garden cloud settings exist")
            .lod_debug
            .apply_preset(preset);
        let context = format!("config-only preset change to {preset:?}");
        let mut totals = GardenDebugUploadDelta::default();
        let mut observed = None;
        let mut step_telemetry = Vec::new();
        for frame in 0..4 {
            let before = garden_debug_gpu_stats(app);
            let delta = garden_debug_step(app, Some(debug_slot_bytes), &context);
            let after = garden_debug_gpu_stats(app);
            totals.accumulate(delta);
            let next = observe_garden_debug_cut(app, cloud, camera, preset)
                .unwrap_or_else(|| panic!("Garden candidate disappeared during {context}"));
            assert_eq!(
                &next, expected,
                "config-only debug preset change replaced the Garden package/current candidate"
            );
            observed = Some(next);
            step_telemetry.push((
                frame,
                before,
                delta,
                after,
                garden_debug_render_telemetry(app),
            ));
            if preset == LodDebugPreset::Off || totals.config_bytes_written != 0 {
                break;
            }
        }
        assert_eq!(totals.record_buffer_allocations, 0, "{context}");
        assert_eq!(totals.record_bytes_written, 0, "{context}");
        let expected_config_bytes = if preset == LodDebugPreset::Off {
            0
        } else {
            LodDebugGpuUploadStats::config_bytes_per_write()
        };
        assert_eq!(
            totals.config_bytes_written,
            expected_config_bytes,
            "{context} wrote an unexpected debug-uniform byte count; main_preset={:?}; main_visible={:?}; main_metadata_complete={:?}; steps={step_telemetry:#?}",
            app.world()
                .get::<CloudSettings>(cloud)
                .map(|settings| settings.lod_debug.preset),
            app.world()
                .get::<ViewVisibility>(cloud)
                .map(|visibility| (*visibility).get()),
            app.world()
                .get::<LodDebugMetadata>(cloud)
                .map(LodDebugMetadata::is_complete_for_testing),
        );
        observed.expect("config-only Garden preset change was observed")
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_garden_debug_off_page_off_pixel_contract(
        app: &mut App,
        cloud: Entity,
        camera: Entity,
        target: &Handle<Image>,
        debug_slot_bytes: u64,
        expected: &GardenDebugCutObservation,
        probe: &GardenDebugIndirectProbe,
        indirect_request: &mut u64,
    ) -> GardenDebugPixelContractSummary {
        {
            let sink = app.world().resource::<GardenDebugPixelCaptureSink>();
            assert!(
                sink.pending.is_empty() && sink.images.is_empty() && sink.discard.is_empty(),
                "Garden debug pixel sink was not empty before its one-shot transition"
            );
        }

        let mut frames = Vec::new();
        let mut next_sample = 0_u32;
        let (off_before, off_before_upload) = capture_garden_debug_pixel_frame(
            app,
            cloud,
            camera,
            target,
            debug_slot_bytes,
            expected,
            probe,
            indirect_request,
            &mut next_sample,
            GardenDebugPixelPhase::OffBefore,
        );
        assert_eq!(
            off_before_upload,
            GardenDebugUploadDelta::default(),
            "stable Off pixel baseline performed debug upload work"
        );
        assert_eq!(off_before.status_preset, LodDebugPreset::Off);
        assert_eq!(
            off_before.availability,
            GaussianLodDebugAvailability::Disabled
        );
        assert_eq!(off_before.render_preset, LodDebugPreset::Off);
        assert_eq!(off_before.phase, GardenDebugPixelPhase::OffBefore);
        assert!(off_before.indirect_draw > 0);
        let off_before_sample = off_before.sample;
        frames.push(off_before);

        app.world_mut()
            .get_mut::<CloudSettings>(cloud)
            .expect("Garden cloud settings exist")
            .lod_debug
            .apply_preset(LodDebugPreset::Page);
        let mut page_upload = GardenDebugUploadDelta::default();
        let mut ready_page_samples = Vec::new();
        let mut page_transition_frames = 0_usize;
        let required_slots = expected
            .ranges
            .iter()
            .map(|range| (range.2, range.3))
            .collect::<BTreeSet<_>>()
            .len();
        let gpu_slots_per_frame = LodDebugGpuUploadStats::max_sparse_record_slots_per_frame().min(
            usize::try_from(
                LodDebugGpuUploadStats::max_sparse_record_bytes_per_frame() / debug_slot_bytes,
            )
            .expect("Garden debug byte-limited slot count fits usize")
            .max(1),
        );
        let cpu_candidate_count = usize::try_from(expected.candidate_count)
            .expect("Garden debug candidate count fits usize");
        let cpu_frames = cpu_candidate_count.div_ceil(GARDEN_DEBUG_PIXEL_CPU_RECORDS_PER_FRAME);
        let gpu_frames = required_slots.div_ceil(gpu_slots_per_frame);
        let page_transition_frame_bound = cpu_frames
            .checked_add(gpu_frames)
            .and_then(|frames| frames.checked_add(GARDEN_DEBUG_PIXEL_TRANSITION_HEADROOM_FRAMES))
            .expect("Garden debug transition-frame bound fits usize");
        for _ in 0..page_transition_frame_bound {
            let (frame, upload) = capture_garden_debug_pixel_frame(
                app,
                cloud,
                camera,
                target,
                debug_slot_bytes,
                expected,
                probe,
                indirect_request,
                &mut next_sample,
                GardenDebugPixelPhase::PageRequested,
            );
            assert_eq!(frame.phase, GardenDebugPixelPhase::PageRequested);
            assert!(frame.indirect_draw > 0);
            page_transition_frames += 1;
            assert!(
                upload.record_buffer_allocations <= 1,
                "Off-to-Page pixel transition allocated several debug record buffers in one frame: {upload:?}"
            );
            page_upload.accumulate(upload);
            let render_page_ready =
                frame.render_preset == LodDebugPreset::Page && frame.debug_binding_ready;
            if render_page_ready {
                assert!(
                    frame.lod_debug_queued,
                    "Page binding became ready without same-frame LOD_DEBUG specialization: frame={frame:?}, render={:#?}",
                    garden_debug_render_telemetry(app),
                );
            }
            let page_ready = render_page_ready
                && frame.status_preset == LodDebugPreset::Page
                && frame.availability == GaussianLodDebugAvailability::MetadataReady;
            if page_ready {
                ready_page_samples.push(frame.sample);
            } else {
                for sample in ready_page_samples.drain(..) {
                    discard_garden_debug_pixel_sample(app, sample);
                }
                discard_garden_debug_pixel_sample(app, frame.sample);
                ready_page_samples.clear();
            }
            frames.push(frame);
            if ready_page_samples.len() == GARDEN_DEBUG_PIXEL_STABLE_FRAMES {
                break;
            }
        }
        assert_eq!(
            page_upload.record_buffer_allocations, 1,
            "cold Off-to-Page enable must allocate exactly one bounded debug record buffer"
        );
        assert!(
            page_upload.record_bytes_written > 0,
            "cold Off-to-Page enable published no sparse debug records"
        );
        assert_eq!(
            page_upload.config_bytes_written,
            LodDebugGpuUploadStats::config_bytes_per_write(),
            "Off-to-Page pixel transition must update exactly one debug uniform"
        );
        assert_eq!(
            ready_page_samples.len(),
            GARDEN_DEBUG_PIXEL_STABLE_FRAMES,
            "Page did not produce two consecutive frames with a ready exact binding and LOD_DEBUG specialization: frames={frames:#?}, render={:#?}",
            garden_debug_render_telemetry(app),
        );

        app.world_mut()
            .get_mut::<CloudSettings>(cloud)
            .expect("Garden cloud settings exist")
            .lod_debug
            .apply_preset(LodDebugPreset::Off);
        let mut off_upload = GardenDebugUploadDelta::default();
        let mut restored_off_samples = Vec::new();
        let mut off_restore_frames = 0_usize;
        for _ in 0..GARDEN_DEBUG_PIXEL_MAX_OFF_TRANSITION_FRAMES {
            let (frame, upload) = capture_garden_debug_pixel_frame(
                app,
                cloud,
                camera,
                target,
                debug_slot_bytes,
                expected,
                probe,
                indirect_request,
                &mut next_sample,
                GardenDebugPixelPhase::OffRestored,
            );
            assert_eq!(frame.phase, GardenDebugPixelPhase::OffRestored);
            assert!(frame.indirect_draw > 0);
            off_restore_frames += 1;
            assert_eq!(
                upload,
                GardenDebugUploadDelta::default(),
                "Page-to-Off pixel transition must remove the binding without buffer writes"
            );
            off_upload.accumulate(upload);
            let off_ready = frame.status_preset == LodDebugPreset::Off
                && frame.availability == GaussianLodDebugAvailability::Disabled
                && frame.render_preset == LodDebugPreset::Off;
            if off_ready {
                assert!(
                    !frame.lod_debug_queued,
                    "Off frame unexpectedly queued LOD_DEBUG: {frame:?}"
                );
                restored_off_samples.push(frame.sample);
            } else {
                for sample in restored_off_samples.drain(..) {
                    discard_garden_debug_pixel_sample(app, sample);
                }
                discard_garden_debug_pixel_sample(app, frame.sample);
                restored_off_samples.clear();
            }
            frames.push(frame);
            if restored_off_samples.len() == GARDEN_DEBUG_PIXEL_STABLE_FRAMES {
                break;
            }
        }
        assert_eq!(
            off_upload,
            GardenDebugUploadDelta::default(),
            "Page-to-Off transition performed buffer work"
        );
        assert_eq!(
            restored_off_samples.len(),
            GARDEN_DEBUG_PIXEL_STABLE_FRAMES,
            "Off did not settle for two consecutive frames after Page: frames={frames:#?}, render={:#?}",
            garden_debug_render_telemetry(app),
        );
        let retained_samples = [
            off_before_sample,
            ready_page_samples[0],
            ready_page_samples[1],
            restored_off_samples[0],
            restored_off_samples[1],
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();

        for drain_frame in 0..GARDEN_DEBUG_PIXEL_READBACK_DRAIN_FRAMES {
            if app
                .world()
                .resource::<GardenDebugPixelCaptureSink>()
                .pending
                .is_empty()
            {
                break;
            }
            let upload = garden_debug_step(
                app,
                Some(debug_slot_bytes),
                "Garden debug pixel readback drain",
            );
            assert_eq!(
                upload,
                GardenDebugUploadDelta::default(),
                "settled Off state performed debug work while draining pixel readbacks"
            );
            let observed = observe_garden_debug_active_cut(app, cloud, camera).unwrap_or_else(|| {
                panic!(
                    "Garden ACTIVE cut disappeared while draining pixel readbacks at frame {drain_frame}"
                )
            });
            assert_eq!(
                &observed, expected,
                "Garden cut changed while draining pixel readbacks"
            );
        }
        let images = {
            let mut sink = app
                .world_mut()
                .resource_mut::<GardenDebugPixelCaptureSink>();
            assert!(
                sink.pending.is_empty(),
                "Garden debug pixel readbacks did not complete: pending={:?}, captured={}, expected={}",
                sink.pending,
                sink.images.len(),
                frames.len(),
            );
            assert!(
                sink.discard.is_empty(),
                "discarded Garden debug pixel readbacks did not complete: {:?}",
                sink.discard,
            );
            assert_eq!(
                sink.images.keys().copied().collect::<BTreeSet<_>>(),
                retained_samples,
                "Garden debug pixel sink retained the wrong endpoint frames"
            );
            std::mem::take(&mut sink.images)
        };

        let get_image = |sample| {
            garden_debug_linear_rgba_with_alpha(
                images
                    .get(&sample)
                    .unwrap_or_else(|| panic!("missing Garden debug pixel sample {sample}")),
            )
        };
        let off_before_image = get_image(off_before_sample);
        let page_first = get_image(ready_page_samples[0]);
        let page_second = get_image(ready_page_samples[1]);
        let off_after_image = get_image(restored_off_samples[1]);

        // Page is intentionally categorical across authored page boundaries.
        // Require only stationary consecutive Page frames, never spatial RGB
        // continuity across two differently colored pages.
        assert_garden_temporal_stability(
            &page_first,
            &page_second,
            "two consecutive ready Page debug frames",
        );
        let page_stability = compare_linear_rgba(
            &page_first,
            &page_second,
            GARDEN_DEBUG_PIXEL_ALPHA_THRESHOLD,
        )
        .expect("Page debug stability metrics are valid");

        let off_restore = compare_linear_rgba(
            &off_before_image,
            &off_after_image,
            GARDEN_DEBUG_PIXEL_ALPHA_THRESHOLD,
        )
        .expect("debug Off restoration metrics are valid");
        assert!(
            off_restore.foreground_iou >= 0.9999 && off_restore.alpha_mae <= 0.0001,
            "Off-before/after alpha silhouette changed across Page: {off_restore:?}"
        );

        let page_supported_hue_bins = garden_debug_page_supported_hue_bins(&page_second);
        assert!(
            page_supported_hue_bins >= GARDEN_DEBUG_MIN_PAGE_HUE_BINS,
            "Page debug output did not expose a nontrivial categorical palette: supported_hue_bins={page_supported_hue_bins}"
        );
        let (page_changed_pixels, page_foreground_pixels, page_changed_fraction) =
            garden_debug_rgb_difference(&off_before_image, &page_second);
        assert!(
            page_changed_pixels >= GARDEN_DEBUG_MIN_PAGE_CHANGED_PIXELS
                && page_changed_fraction >= GARDEN_DEBUG_MIN_PAGE_CHANGED_FRACTION,
            "Page debug output did not differ meaningfully from Off: changed={page_changed_pixels}, foreground={page_foreground_pixels}, fraction={page_changed_fraction:.6}"
        );

        GardenDebugPixelContractSummary {
            captured_frames: frames.len(),
            page_transition_frames,
            off_restore_frames,
            page_supported_hue_bins,
            page_changed_pixels,
            page_changed_fraction,
            page_stability,
            off_restore,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_garden_debug_pixel_frame(
        app: &mut App,
        cloud: Entity,
        camera: Entity,
        target: &Handle<Image>,
        debug_slot_bytes: u64,
        expected: &GardenDebugCutObservation,
        probe: &GardenDebugIndirectProbe,
        indirect_request: &mut u64,
        next_sample: &mut u32,
        phase: GardenDebugPixelPhase,
    ) -> (GardenDebugPixelFrameEvidence, GardenDebugUploadDelta) {
        let sample = *next_sample;
        *next_sample = next_sample
            .checked_add(1)
            .expect("Garden debug pixel sample fits u32");
        let screenshot = app
            .world_mut()
            .spawn(Screenshot::image(target.clone()))
            .id();
        let previous = app
            .world_mut()
            .resource_mut::<GardenDebugPixelCaptureSink>()
            .pending
            .insert(screenshot, sample);
        assert!(
            previous.is_none(),
            "Garden debug screenshot entity was reused"
        );

        *indirect_request = indirect_request
            .checked_add(1)
            .expect("Garden debug indirect request id fits u64");
        probe.request(*indirect_request, expected.candidate_count);
        let before = garden_debug_gpu_stats(app);
        let upload = garden_debug_step(app, Some(debug_slot_bytes), "debug pixel transition");
        let after = garden_debug_gpu_stats(app);

        let observed = observe_garden_debug_active_cut(app, cloud, camera).unwrap_or_else(|| {
            panic!(
                "Garden pixel frame {sample} lost its ACTIVE cut during {phase:?}: package={:?}, lod={:?}, render={:#?}",
                app.world().get::<GaussianLodPackageStatus>(cloud),
                app.world().get::<GaussianLodStatus>(cloud),
                garden_debug_render_telemetry(app),
            )
        });
        assert_eq!(
            &observed, expected,
            "Garden pixel frame {sample} changed its ACTIVE cut during {phase:?}"
        );
        let indirect = probe.result(*indirect_request).unwrap_or_else(|| {
            panic!(
                "Garden pixel frame {sample} did not publish same-frame indirect evidence during {phase:?}: render={:#?}",
                garden_debug_render_telemetry(app),
            )
        });
        let indirect_draw = assert_garden_debug_indirect_observation(
            indirect,
            expected,
            &format!("debug pixel sample {sample} during {phase:?}"),
        );
        let lod_status = app
            .world()
            .get::<GaussianLodStatus>(cloud)
            .expect("Garden debug pixel frame has LoD status");
        let status_preset = lod_status.debug_preset;
        let availability = lod_status.debug_availability;
        let (render_preset, debug_binding_ready) = garden_debug_render_pixel_state(app);
        let lod_debug_queued = after.ready_bind_group_queues() > before.ready_bind_group_queues()
            && after.specialized_pipeline_queues() > before.specialized_pipeline_queues();

        (
            GardenDebugPixelFrameEvidence {
                sample,
                phase,
                status_preset,
                availability,
                render_preset,
                debug_binding_ready,
                lod_debug_queued,
                indirect_draw,
            },
            upload,
        )
    }

    fn garden_debug_render_pixel_state(app: &App) -> (LodDebugPreset, bool) {
        let states = app
            .sub_app(RenderApp)
            .world()
            .iter_entities()
            .filter_map(|entity| {
                let settings = entity.get::<CloudSettings>()?;
                let preset = settings.lod_debug.preset;
                let binding_ready =
                    entity
                        .get::<LodDebugBindGroup<Gaussian3d>>()
                        .is_some_and(|binding| {
                            binding.ready_for_testing() && binding.preset_for_testing() == preset
                        });
                Some((preset, binding_ready))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            states.len(),
            1,
            "Garden debug pixel gate expected one extracted cloud: {states:?}"
        );
        states[0]
    }

    fn garden_debug_page_supported_hue_bins(image: &[[f32; 4]]) -> usize {
        const HUE_BINS: usize = 12;
        const MIN_SATURATION: f32 = 0.2;
        const MIN_PIXELS_PER_BIN: u32 = 16;
        let mut bins = [0_u32; HUE_BINS];
        for pixel in image
            .iter()
            .filter(|pixel| pixel[3] > GARDEN_DEBUG_PIXEL_ALPHA_THRESHOLD)
        {
            let [red, green, blue] = [pixel[0], pixel[1], pixel[2]];
            let maximum = red.max(green).max(blue);
            let minimum = red.min(green).min(blue);
            let chroma = maximum - minimum;
            if maximum <= 0.0 || chroma / maximum < MIN_SATURATION {
                continue;
            }
            let hue_sector = if red >= green && red >= blue {
                (green - blue) / chroma
            } else if green >= blue {
                (blue - red) / chroma + 2.0
            } else {
                (red - green) / chroma + 4.0
            };
            let hue_turn = (hue_sector / 6.0).rem_euclid(1.0);
            let bin = ((hue_turn * HUE_BINS as f32).floor() as usize).min(HUE_BINS - 1);
            bins[bin] += 1;
        }
        bins.into_iter()
            .filter(|pixels| *pixels >= MIN_PIXELS_PER_BIN)
            .count()
    }

    fn garden_debug_rgb_difference(off: &[[f32; 4]], page: &[[f32; 4]]) -> (usize, usize, f64) {
        assert_eq!(off.len(), page.len());
        let mut foreground_pixels = 0_usize;
        let mut changed_pixels = 0_usize;
        for (off, page) in off.iter().zip(page) {
            if off[3] <= GARDEN_DEBUG_PIXEL_ALPHA_THRESHOLD
                && page[3] <= GARDEN_DEBUG_PIXEL_ALPHA_THRESHOLD
            {
                continue;
            }
            foreground_pixels += 1;
            let max_rgb_difference = off[..3]
                .iter()
                .zip(&page[..3])
                .map(|(off, page)| (off - page).abs())
                .fold(0.0_f32, f32::max);
            changed_pixels += usize::from(max_rgb_difference >= GARDEN_DEBUG_MIN_PAGE_RGB_DELTA);
        }
        let changed_fraction = changed_pixels as f64 / foreground_pixels.max(1) as f64;
        (changed_pixels, foreground_pixels, changed_fraction)
    }

    fn discard_garden_debug_pixel_sample(app: &mut App, sample: u32) {
        let mut sink = app
            .world_mut()
            .resource_mut::<GardenDebugPixelCaptureSink>();
        if sink.images.remove(&sample).is_some() {
            return;
        }
        assert!(
            sink.pending.values().any(|pending| *pending == sample),
            "Garden debug pixel sample {sample} was neither captured nor pending"
        );
        assert!(
            sink.discard.insert(sample),
            "Garden debug pixel sample {sample} was discarded twice"
        );
    }

    fn on_garden_debug_pixel_capture(
        trigger: On<ScreenshotCaptured>,
        mut sink: ResMut<GardenDebugPixelCaptureSink>,
    ) {
        let sample = sink.pending.remove(&trigger.entity).unwrap_or_else(|| {
            panic!(
                "unregistered Garden debug pixel screenshot {:?}",
                trigger.entity
            )
        });
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("Garden debug pixel screenshot converts")
            .to_rgba8();
        assert_eq!(rgba.width(), GARDEN_TARGET_WIDTH);
        assert_eq!(rgba.height(), GARDEN_TARGET_HEIGHT);
        // Unlike the broader Garden RGB oracles, this contract needs the real
        // render-target alpha silhouette. Preserve screenshot alpha instead of
        // synthesizing a binary foreground channel from RGB luminance.
        let bytes = rgba.into_raw();
        let image = garden_debug_linear_rgba_with_alpha(&bytes);
        assert_garden_bounds_fit_image_nonblank(garden_image_sanity(
            &image,
            GARDEN_TARGET_WIDTH as usize,
            GARDEN_TARGET_HEIGHT as usize,
        ));
        let alpha_foreground = image
            .iter()
            .filter(|pixel| pixel[3] > GARDEN_DEBUG_PIXEL_ALPHA_THRESHOLD)
            .count();
        assert!(
            alpha_foreground >= 1_024 && alpha_foreground < image.len(),
            "Garden debug pixel frame has an empty or full alpha silhouette: sample={sample}, alpha_foreground={alpha_foreground}, pixels={}",
            image.len(),
        );
        if sink.discard.remove(&sample) {
            return;
        }
        assert!(
            sink.images.insert(sample, bytes).is_none(),
            "Garden debug pixel sample {sample} was captured twice"
        );
    }

    fn garden_debug_linear_rgba_with_alpha(bytes: &[u8]) -> Vec<[f32; 4]> {
        bytes
            .chunks_exact(4)
            .map(|pixel| {
                [
                    srgb_to_linear(pixel[0]),
                    srgb_to_linear(pixel[1]),
                    srgb_to_linear(pixel[2]),
                    f32::from(pixel[3]) / 255.0,
                ]
            })
            .collect()
    }

    fn garden_debug_render_telemetry(app: &App) -> Vec<String> {
        app.sub_app(RenderApp)
            .world()
            .iter_entities()
            .filter_map(|entity| {
                let settings = entity.get::<CloudSettings>()?;
                let binding = entity.get::<LodDebugBindGroup<Gaussian3d>>();
                Some(format!(
                    "entity={:?}, extracted={:?}, metadata_complete={:?}, binding_preset={:?}, ready={:?}, sparse_identity={:?}, records={:?}",
                    entity.id(),
                    settings.lod_debug.preset,
                    entity
                        .get::<LodDebugMetadata>()
                        .map(LodDebugMetadata::is_complete_for_testing),
                    binding.map(LodDebugBindGroup::preset_for_testing),
                    binding.map(LodDebugBindGroup::ready_for_testing),
                    binding.and_then(LodDebugBindGroup::sparse_identity_for_testing),
                    binding.map(LodDebugBindGroup::record_count_for_testing),
                ))
            })
            .collect()
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct GardenDebugIndirectObservation {
        request: u64,
        candidate_count: u32,
        args: LodIndirectArgs,
    }

    #[derive(Debug, Default)]
    struct GardenDebugIndirectProbeState {
        requested: Option<(u64, u32)>,
        observation: Option<GardenDebugIndirectObservation>,
        error: Option<String>,
    }

    #[derive(Resource, Clone, ExtractResource, Default)]
    struct GardenDebugIndirectProbe(Arc<Mutex<GardenDebugIndirectProbeState>>);

    impl GardenDebugIndirectProbe {
        fn request(&self, request: u64, candidate_count: u32) {
            let mut state = self
                .0
                .lock()
                .expect("Garden debug indirect probe mutex is not poisoned");
            state.requested = Some((request, candidate_count));
            state.observation = None;
            state.error = None;
        }

        fn result(&self, request: u64) -> Option<GardenDebugIndirectObservation> {
            let state = self
                .0
                .lock()
                .expect("Garden debug indirect probe mutex is not poisoned");
            if let Some(error) = &state.error {
                panic!("Garden debug indirect readback failed: {error}");
            }
            state
                .observation
                .filter(|observation| observation.request == request)
        }
    }

    fn read_garden_debug_indirect_args(
        render_device: Res<RenderDevice>,
        render_queue: Res<RenderQueue>,
        buffers: Res<LodCompactionBuffers<Gaussian3d>>,
        views: Query<&ExtractedView, With<GaussianCamera>>,
        clouds: Query<(Entity, &PlanarGaussian3dHandle)>,
        probe: Res<GardenDebugIndirectProbe>,
    ) {
        let request = {
            let state = probe
                .0
                .lock()
                .expect("Garden debug indirect probe mutex is not poisoned");
            let Some(requested) = state.requested else {
                return;
            };
            if state
                .observation
                .is_some_and(|observation| observation.request == requested.0)
            {
                return;
            }
            requested
        };
        for view in &views {
            for (entity, handle) in &clouds {
                let Some(state) =
                    buffers.get_ready(view.retained_view_entity, entity, handle.handle().id())
                else {
                    continue;
                };
                if state.candidate_count() != request.1 {
                    continue;
                }
                match read_lod_indirect_args_for_testing(&render_device, &render_queue, state) {
                    Ok(args) => {
                        let mut probe = probe
                            .0
                            .lock()
                            .expect("Garden debug indirect probe mutex is not poisoned");
                        probe.observation = Some(GardenDebugIndirectObservation {
                            request: request.0,
                            candidate_count: request.1,
                            args,
                        });
                        probe.error = None;
                    }
                    Err(error) => {
                        probe
                            .0
                            .lock()
                            .expect("Garden debug indirect probe mutex is not poisoned")
                            .error = Some(error.to_string());
                    }
                }
                return;
            }
        }
    }

    fn assert_garden_debug_indirect_observation(
        observation: GardenDebugIndirectObservation,
        expected: &GardenDebugCutObservation,
        context: &str,
    ) -> u32 {
        assert_eq!(
            observation.candidate_count, expected.candidate_count,
            "Garden indirect candidate count changed during {context}"
        );
        assert_eq!(
            observation.args.vertex_count, 4,
            "Garden indirect quad topology changed during {context}"
        );
        assert!(
            observation.args.instance_count > 0,
            "Garden output became empty during {context}"
        );
        assert!(
            observation.args.instance_count <= observation.candidate_count,
            "Garden indirect draw exceeded its pre-cull candidate count during {context}"
        );
        assert_eq!(
            observation.args.instance_count, observation.args.candidate_hits,
            "Garden debug compaction lost accepted candidates during {context}"
        );
        assert_eq!(
            observation.args.overflow_count, 0,
            "Garden indirect draw overflowed during {context}"
        );
        observation.args.instance_count
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_garden_debug_indirect_draw(
        app: &mut App,
        cloud: Entity,
        camera: Entity,
        expected_preset: LodDebugPreset,
        debug_slot_bytes: u64,
        expected: &GardenDebugCutObservation,
        probe: &GardenDebugIndirectProbe,
        request: &mut u64,
    ) -> u32 {
        *request = request
            .checked_add(1)
            .expect("indirect request id fits u64");
        probe.request(*request, expected.candidate_count);
        for _ in 0..16 {
            let delta = garden_debug_step(app, Some(debug_slot_bytes), "debug indirect probe");
            assert_eq!(
                delta,
                GardenDebugUploadDelta::default(),
                "settled Garden debug draw performed upload/allocation work during indirect probe"
            );
            let observed = observe_garden_debug_cut(app, cloud, camera, expected_preset)
                .expect("Garden debug cut remains observable during indirect probe");
            assert_eq!(&observed, expected);
            let Some(observation) = probe.result(*request) else {
                continue;
            };
            return assert_garden_debug_indirect_observation(
                observation,
                expected,
                "settled debug indirect probe",
            );
        }
        panic!("Garden debug indirect draw was not readable after 16 settled frames");
    }

    type GardenCutSignature = (u64, Vec<(u64, u64, u32)>);
    type GardenLogicalCutSignature = (u64, Vec<(u64, u64, u32)>);

    fn garden_logical_cut_digest(signature: &GardenLogicalCutSignature) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut digest = FNV_OFFSET_BASIS;
        let mut absorb = |bytes: &[u8]| {
            for byte in bytes {
                digest ^= u64::from(*byte);
                digest = digest.wrapping_mul(FNV_PRIME);
            }
        };
        absorb(&signature.0.to_le_bytes());
        absorb(&(signature.1.len() as u64).to_le_bytes());
        for &(node, page, count) in &signature.1 {
            absorb(&node.to_le_bytes());
            absorb(&page.to_le_bytes());
            absorb(&count.to_le_bytes());
        }
        digest
    }

    const GARDEN_TARGET_WIDTH: u32 = 1_920;
    const GARDEN_TARGET_HEIGHT: u32 = 1_080;
    const GARDEN_DEBUG_PIXEL_ALPHA_THRESHOLD: f32 = 8.0 / 255.0;
    const GARDEN_DEBUG_PIXEL_STABLE_FRAMES: usize = 2;
    // Mirror the package's public behavior through an acceptance-test bound:
    // CPU annotation creation admits at most 32K candidate records per frame.
    const GARDEN_DEBUG_PIXEL_CPU_RECORDS_PER_FRAME: usize = 32 * 1_024;
    const GARDEN_DEBUG_PIXEL_TRANSITION_HEADROOM_FRAMES: usize = 8;
    const GARDEN_DEBUG_PIXEL_MAX_OFF_TRANSITION_FRAMES: u32 = 8;
    const GARDEN_DEBUG_PIXEL_READBACK_DRAIN_FRAMES: u32 = 240;
    const GARDEN_DEBUG_MIN_PAGE_HUE_BINS: usize = 4;
    const GARDEN_DEBUG_MIN_PAGE_RGB_DELTA: f32 = 0.02;
    const GARDEN_DEBUG_MIN_PAGE_CHANGED_PIXELS: usize = 1_024;
    const GARDEN_DEBUG_MIN_PAGE_CHANGED_FRACTION: f64 = 0.05;
    const GARDEN_EXTERNAL_BUILDER_ABI: u32 = 16;
    const GARDEN_MOMENT_MERGE_REDUCER_VERSION: u32 = 4;
    const GARDEN_SOURCE_GAUSSIANS: u64 = 5_834_784;
    const GARDEN_SOURCE_SHA256: &str =
        "16701d5e0630dfaca74f8794ed7ce2aa23fa922f87dc09a7e37484e8d3f82d5a";
    const GARDEN_MANIFEST_SHA256: &str =
        "67b9119222e1435fb88755698dcd916e608c9cd21c1417b687a7cce663729600";
    const GARDEN_SCENE_MIN: [f32; 3] = [-118.729_54, -130.432_02, -121.283_48];
    const GARDEN_SCENE_MAX: [f32; 3] = [137.847_32, 109.880_554, 136.600_8];
    const GARDEN_SCENE_CENTER: [f32; 3] = [9.558_891, -10.275_734, 7.658_661];
    const GARDEN_SCENE_RADIUS: f32 = 217.994_34;
    const GARDEN_AUTO_FRAME_DISTANCE: f32 = 474.641_1;
    const GARDEN_VIEWER_MAX_ACTIVE_GAUSSIANS: u64 = 8_000_000;
    const GARDEN_NODE_PAGE_COUNT: usize = 6_517;
    const GARDEN_MAX_REFINEMENT_AMPLIFICATION: u64 = 8;
    const GARDEN_FOREGROUND_LUMINANCE: f64 = 1.0e-4;
    // The spatial oracle uses a visible 1%-linear-luminance silhouette rather
    // than treating a single nonzero quantized tail pixel as scene morphology.
    const GARDEN_ORACLE_FOREGROUND_LUMINANCE: f64 = 0.01;
    const GARDEN_CAPTURE_SAMPLES: u32 = 16;
    const GARDEN_CAPTURE_INTERVAL_FRAMES: u32 = 8;
    const GARDEN_PACKAGE_RETIRE_FRAMES: u32 = 8;
    const GARDEN_FLAT_REFERENCE_WARMUP_FRAMES: u32 = 240;
    const GARDEN_TEMPORAL_WIDTH: u32 = 640;
    const GARDEN_TEMPORAL_HEIGHT: u32 = 360;
    const GARDEN_TEMPORAL_DOLLY_SAMPLES: u32 = 48;
    // A candidate can remain ACTIVE while its next preferred pages are still
    // arriving. Match the interactive gate's one-second quiescence proof so a
    // measured dolly never inherits pre-sample bootstrap/refinement work.
    const GARDEN_TEMPORAL_STABLE_FRAMES: u32 = 120;
    const GARDEN_TEMPORAL_MAX_SETUP_FRAMES: u32 = 7_200;

    // Provisional pre-characterization limits. The baseline-only phase prints
    // the flat and fixed-cut evidence needed to replace these with measured,
    // adapter-qualified values. The comparisons still reject gross one-frame
    // topology spikes while that calibration is pending.
    const GARDEN_TEMPORAL_DELTA_RATIO: f64 = 2.0;
    const GARDEN_TEMPORAL_DELTA_RMS_FLOOR: f64 = 0.002;
    const GARDEN_TEMPORAL_DELTA_MAX_FLOOR: f64 = 0.05;
    const GARDEN_TEMPORAL_SECOND_RATIO: f64 = 2.0;
    const GARDEN_TEMPORAL_SECOND_RMS_FLOOR: f64 = 0.003;
    const GARDEN_TEMPORAL_SECOND_MAX_FLOOR: f64 = 0.08;

    // An interior LoD cut deliberately renders fewer records than the flat
    // source, so exact pixels are neither expected nor a useful acceptance
    // contract. These bounds reject scene-scale morphology/color corruption:
    // 20 dB full-frame and 18 dB foreground PSNR cap normalized RMS error at
    // 10% and 12.6%, while SSIM/IoU independently require structure and the
    // luminance-defined silhouette to remain close to the canonical source.
    const GARDEN_MIN_FULL_FRAME_PSNR: f64 = 20.0;
    const GARDEN_MIN_FOREGROUND_PSNR: f64 = 18.0;
    const GARDEN_MIN_LUMINANCE_SSIM: f64 = 0.90;
    const GARDEN_MIN_FOREGROUND_IOU: f64 = 0.90;
    const GARDEN_MAX_FOREGROUND_RGB_MAE: f64 = 0.08;
    const GARDEN_MAX_FOREGROUND_LUMINANCE_BIAS: f64 = 0.03;

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    enum GardenCapturePhase {
        #[default]
        AwaitingStableCut,
        PackageCapturePending,
        BetweenPackageCaptures,
        RetiringPackage,
        FlatReferenceWarmup,
        FlatReferencePending,
    }

    #[derive(Clone, Copy, Debug)]
    struct GardenImageSanity {
        foreground_pixels: usize,
        foreground_fraction: f64,
        foreground_mean_luminance: f64,
        foreground_luminance_stddev: f64,
        foreground_luminance_range: f64,
        occupied_macro_tiles: usize,
        horizontal_extent: f64,
        vertical_extent: f64,
    }

    #[derive(Clone, Copy, Debug)]
    struct GardenSpatialMetrics {
        full_frame_psnr_rgb: f64,
        foreground_psnr_rgb: f64,
        luminance_ssim: f64,
        foreground_iou: f64,
        foreground_rgb_mae: f64,
        foreground_luminance_bias: f64,
        foreground_union_pixels: usize,
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct GardenTemporalDifference {
        full_rgb_rms: f64,
        foreground_rgb_rms: f64,
        max_abs_rgb: f64,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum GardenTemporalDollyDirection {
        Refining,
        Coarsening,
    }

    impl GardenTemporalDollyDirection {
        const ALL: [Self; 2] = [Self::Refining, Self::Coarsening];

        fn path_sample(self, sample: u32) -> u32 {
            match self {
                Self::Refining => sample,
                Self::Coarsening => GARDEN_TEMPORAL_DOLLY_SAMPLES - 1 - sample,
            }
        }
    }

    #[derive(Debug, Default)]
    struct GardenTemporalTrace {
        deltas: Vec<GardenTemporalDifference>,
        second_differences: Vec<GardenTemporalDifference>,
        initial_active_candidate_count: Option<u32>,
        final_active_candidate_count: Option<u32>,
        active_endpoint_changes: u32,
        blend_frames: u32,
        fractional_blend_frames: u32,
        peak_blend_edges: u32,
        lagging_blend_frames: u32,
        immutable_table_uploads: u64,
        weight_writes: u64,
        buffer_allocations: u64,
        bounded_hard_frames: u32,
    }

    impl GardenTemporalTrace {
        fn summary(&self) -> String {
            fn maxima(values: &[GardenTemporalDifference]) -> GardenTemporalDifference {
                values.iter().copied().fold(
                    GardenTemporalDifference::default(),
                    |maximum, value| GardenTemporalDifference {
                        full_rgb_rms: maximum.full_rgb_rms.max(value.full_rgb_rms),
                        foreground_rgb_rms: maximum
                            .foreground_rgb_rms
                            .max(value.foreground_rgb_rms),
                        max_abs_rgb: maximum.max_abs_rgb.max(value.max_abs_rgb),
                    },
                )
            }
            format!(
                "samples={}, active_candidate_count={:?}->{:?}, active_endpoint_changes={}, blend_frames={}, fractional_blend_frames={}, peak_blend_edges={}, lagging_blend_frames={}, immutable_table_uploads={}, weight_writes={}, buffer_allocations={}, bounded_hard_frames={}, delta_max={:?}, second_max={:?}",
                self.deltas.len() + 1,
                self.initial_active_candidate_count,
                self.final_active_candidate_count,
                self.active_endpoint_changes,
                self.blend_frames,
                self.fractional_blend_frames,
                self.peak_blend_edges,
                self.lagging_blend_frames,
                self.immutable_table_uploads,
                self.weight_writes,
                self.buffer_allocations,
                self.bounded_hard_frames,
                maxima(&self.deltas),
                maxima(&self.second_differences),
            )
        }
    }

    #[derive(Default)]
    struct GardenTemporalTraceAccumulator {
        previous_previous: Option<Vec<[f32; 4]>>,
        previous: Option<Vec<[f32; 4]>>,
        trace: GardenTemporalTrace,
    }

    impl GardenTemporalTraceAccumulator {
        fn push(&mut self, image: Vec<[f32; 4]>) {
            if let Some(previous) = self.previous.as_ref() {
                self.trace
                    .deltas
                    .push(garden_temporal_difference(previous, &image));
            }
            if let (Some(previous_previous), Some(previous)) =
                (self.previous_previous.as_ref(), self.previous.as_ref())
            {
                self.trace
                    .second_differences
                    .push(garden_temporal_second_difference(
                        previous_previous,
                        previous,
                        &image,
                    ));
            }
            self.previous_previous = self.previous.take();
            self.previous = Some(image);
        }

        fn finish(self) -> GardenTemporalTrace {
            self.trace
        }
    }

    type GardenTemporalCutSignature = (u32, Vec<(u64, u64, u32)>);
    type GardenTemporalSettledBaseline = GardenTemporalCutSignature;

    #[derive(Clone, Debug, PartialEq)]
    struct GardenRoundtripSettledSignature {
        logical_cut: GardenTemporalCutSignature,
        presentation: GardenViewBlendPresentationSignature,
    }

    #[derive(Resource, Default)]
    struct GardenTemporalCaptureSink {
        pending: BTreeMap<Entity, u32>,
        images: BTreeMap<u32, Vec<[f32; 4]>>,
    }

    #[derive(Default)]
    struct GardenTemporalTelemetry {
        selection_mode: LodSelectionMode,
        last_active_cut: Option<GardenTemporalCutSignature>,
        initial_active_candidate_count: Option<u32>,
        final_active_candidate_count: Option<u32>,
        active_endpoint_changes: u32,
        initial_blend_signature: Option<GardenViewBlendPresentationSignature>,
        last_blend_signature: Option<GardenViewBlendPresentationSignature>,
        initial_upload: Option<LodViewBlendUploadStats>,
        last_upload: Option<LodViewBlendUploadStats>,
        last_blend: Option<GardenViewBlendObservation>,
        last_physical_drawable: Option<LodLastRadixDrawableForTesting>,
        promoted_drawable: GardenPromotedDrawableTracker,
        node_parents: GardenNodeParents,
        blend_frames: u32,
        fractional_blend_frames: u32,
        peak_blend_edges: u32,
        lagging_blend_frames: u32,
        authored_publication_hold: GardenAuthoredPublicationHold,
        bounded_hard_frames: u32,
    }

    fn capture_garden_flat_temporal_dolly(
        source_path: &Path,
        manifest_path: &Path,
        direction: GardenTemporalDollyDirection,
    ) -> GardenTemporalTrace {
        let manifest = load_canonical_garden_manifest(manifest_path);
        let scene_frame = garden_scene_frame(&manifest);
        let package_root = manifest_path
            .parent()
            .expect("Garden manifest has a package directory");
        let settings = garden_temporal_lod_settings(LodSelectionMode::Dynamic);
        let mut app = garden_temporal_app(package_root, garden_temporal_package_config(&settings));
        let source = load_canonical_garden_source(source_path);
        let source = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(source);
        app.world_mut().spawn((
            PlanarGaussian3dHandle(source),
            CloudSettings {
                gaussian_mode: GaussianMode::Gaussian3d,
                sort_mode: SortMode::Radix,
                radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                ..default()
            },
            Transform::IDENTITY,
            Visibility::Visible,
            Name::new("canonical_garden_temporal_flat_reference"),
        ));
        let (target, camera) = spawn_garden_temporal_camera(&mut app, scene_frame, direction);

        // One initial settle at the first dolly pose. No settle or duplicate
        // render is inserted between the 48 measured camera samples below.
        for _ in 0..GARDEN_FLAT_REFERENCE_WARMUP_FRAMES {
            app.update();
        }
        capture_garden_temporal_frames(&mut app, camera, &target, scene_frame, direction, None)
    }

    fn capture_garden_package_temporal_dolly(
        manifest_path: &Path,
        selection_mode: LodSelectionMode,
        direction: GardenTemporalDollyDirection,
    ) -> GardenTemporalTrace {
        let manifest = load_canonical_garden_manifest(manifest_path);
        let scene_frame = garden_scene_frame(&manifest);
        let package_root = manifest_path
            .parent()
            .expect("Garden manifest has a package directory");
        let manifest_name = manifest_path
            .file_name()
            .expect("Garden manifest has a file name")
            .to_string_lossy()
            .into_owned();
        let settings = garden_temporal_lod_settings(selection_mode);
        let mut app = garden_temporal_app(package_root, garden_temporal_package_config(&settings));
        let manifest_handle: Handle<GaussianLodAsset> =
            app.world().resource::<AssetServer>().load(manifest_name);
        let cloud = app
            .world_mut()
            .spawn((
                GaussianLodHandle(manifest_handle.clone()),
                GaussianLodPackageSource::native_directory(
                    package_root.to_string_lossy().into_owned(),
                ),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::Radix,
                    radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                    ..default()
                },
                settings,
                Transform::IDENTITY,
                Visibility::Visible,
                Name::new(match selection_mode {
                    LodSelectionMode::Frozen => "canonical_garden_temporal_frozen_package",
                    LodSelectionMode::Dynamic => "canonical_garden_temporal_dynamic_package",
                }),
            ))
            .id();
        let (target, camera) = spawn_garden_temporal_camera(&mut app, scene_frame, direction);
        let mut initial_cut = wait_for_garden_temporal_package_settle(
            &mut app,
            cloud,
            camera,
            &manifest_handle,
            selection_mode,
        );
        if selection_mode == LodSelectionMode::Dynamic {
            // Warm every camera pose once before measurement. The qualified
            // Dynamic trace is the ordinary all-resident class: camera motion
            // must evaluate displayed == desired == f(current view) without
            // borrowing the exceptional late-readiness slew.
            for sample in 0..GARDEN_TEMPORAL_DOLLY_SAMPLES {
                *app.world_mut()
                    .get_mut::<Transform>(camera)
                    .expect("Garden temporal prewarm camera transform exists") =
                    garden_temporal_dolly_transform(scene_frame, direction, sample);
                app.update();
            }
            *app.world_mut()
                .get_mut::<Transform>(camera)
                .expect("Garden temporal prewarm camera transform exists") =
                garden_temporal_dolly_transform(scene_frame, direction, 0);
            initial_cut = wait_for_garden_temporal_package_settle(
                &mut app,
                cloud,
                camera,
                &manifest_handle,
                selection_mode,
            );
        }
        let initial_active_candidate_count = initial_cut.0;
        let initial_render = app
            .world()
            .resource::<GardenViewBlendRenderProbe>()
            .latest_snapshot()
            .expect("initial Garden temporal cut has an ordered promoted drawable");
        let mut promoted_drawable = GardenPromotedDrawableTracker::default();
        assert!(
            matches!(
                promoted_drawable.classify(&initial_render, "initial Garden temporal cut"),
                GardenPromotedDrawableClass::CurrentCandidate
            ),
            "initial Garden temporal cut retained an older drawable"
        );
        let initial_render_candidate = initial_render.candidate.clone();
        let initial_physical_drawable = initial_render.drawable.clone();
        let initial_blend = observe_garden_view_blend_with_render_state(
            &initial_render_candidate,
            initial_render,
            true,
            "initial Garden temporal cut",
        );
        match selection_mode {
            LodSelectionMode::Dynamic => {
                initial_blend.assert_stationary_fixed_point("initial Garden temporal cut")
            }
            LodSelectionMode::Frozen => {
                initial_blend.assert_frozen_fixed_point("initial Garden temporal cut")
            }
        }
        initial_blend.assert_manifest_edge_topology(
            &garden_node_parents(&manifest),
            "initial Garden temporal cut",
        );
        let mut telemetry = GardenTemporalTelemetry {
            selection_mode,
            last_active_cut: Some(initial_cut),
            initial_active_candidate_count: Some(initial_active_candidate_count),
            final_active_candidate_count: Some(initial_active_candidate_count),
            initial_blend_signature: Some(initial_blend.presentation_signature()),
            last_blend_signature: Some(initial_blend.presentation_signature()),
            initial_upload: Some(initial_blend.upload),
            last_upload: Some(initial_blend.upload),
            last_blend: Some(initial_blend.clone()),
            last_physical_drawable: Some(initial_physical_drawable),
            promoted_drawable,
            node_parents: garden_node_parents(&manifest),
            ..default()
        };
        let trace = capture_garden_temporal_frames(
            &mut app,
            camera,
            &target,
            scene_frame,
            direction,
            Some((cloud, &mut telemetry)),
        );
        if selection_mode == LodSelectionMode::Frozen {
            assert_eq!(
                telemetry.active_endpoint_changes, 0,
                "Frozen-LoD changed its logical cut during the sort-only dolly"
            );
            assert_eq!(
                telemetry.last_blend_signature, telemetry.initial_blend_signature,
                "Frozen-LoD changed its topology or displayed/desired weights during the dolly"
            );
        }
        assert_eq!(
            telemetry.bounded_hard_frames, 0,
            "authenticated ABI-16 Garden used a hard cohort during temporal qualification"
        );
        if selection_mode == LodSelectionMode::Dynamic {
            telemetry
                .authored_publication_hold
                .assert_no_pending_incomplete_publication(&format!(
                    "Garden temporal {direction:?} trace"
                ));
            eprintln!(
                "Garden temporal {direction:?} authored publications: distinct={}, max_consecutive_hold_frames={}",
                telemetry.authored_publication_hold.distinct_publications,
                telemetry.authored_publication_hold.max_consecutive_frames,
            );
        }
        trace
    }

    fn garden_temporal_roundtrip_positions() -> Vec<u32> {
        let mut positions = (0..GARDEN_TEMPORAL_DOLLY_SAMPLES).collect::<Vec<_>>();
        positions
            .extend((GARDEN_TEMPORAL_DOLLY_SAMPLES / 2..GARDEN_TEMPORAL_DOLLY_SAMPLES - 1).rev());
        positions.extend(GARDEN_TEMPORAL_DOLLY_SAMPLES / 2 + 1..GARDEN_TEMPORAL_DOLLY_SAMPLES);
        positions.extend((0..GARDEN_TEMPORAL_DOLLY_SAMPLES - 1).rev());
        positions
    }

    fn garden_view_blend_weight_map(
        observation: &GardenViewBlendObservation,
    ) -> BTreeMap<GardenViewBlendEdgeKey, f32> {
        observation
            .edges
            .iter()
            .map(|edge| (edge.key.clone(), f32::from_bits(edge.displayed_weight_bits)))
            .collect()
    }

    fn assert_garden_view_blend_reversal(
        before: &GardenViewBlendObservation,
        turning: &GardenViewBlendObservation,
        after: &GardenViewBlendObservation,
        context: &str,
    ) {
        let before = garden_view_blend_weight_map(before);
        let turning = garden_view_blend_weight_map(turning);
        let after = garden_view_blend_weight_map(after);
        let mut reversed_edges = 0_usize;
        for (key, turning_weight) in &turning {
            let (Some(before_weight), Some(after_weight)) = (before.get(key), after.get(key))
            else {
                continue;
            };
            let incoming = turning_weight - before_weight;
            let outgoing = after_weight - turning_weight;
            if incoming.abs() > f32::EPSILON
                && outgoing.abs() > f32::EPSILON
                && incoming.signum() != outgoing.signum()
            {
                reversed_edges += 1;
            }
        }
        assert!(
            reversed_edges > 0,
            "{context} did not reverse any common edge directly from the current view: before={}, turning={}, after={}",
            before.len(),
            turning.len(),
            after.len(),
        );
    }

    fn garden_view_blend_edge<'a>(
        observation: &'a GardenViewBlendObservation,
        key: &GardenViewBlendEdgeKey,
    ) -> Option<&'a GardenViewBlendEdgeObservation> {
        observation
            .edges
            .binary_search_by(|edge| edge.key.cmp(key))
            .ok()
            .map(|index| &observation.edges[index])
    }

    fn garden_view_blend_edge_key(edge: &LodViewBlendEdge) -> GardenViewBlendEdgeKey {
        GardenViewBlendEdgeKey {
            parent: edge.parent().0,
            children: edge.children().iter().map(|node| node.0).collect(),
            parent_metric: edge.parent_metric().into(),
            child_metrics: edge
                .child_metrics()
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
        }
    }

    fn garden_evaluation_sample_index(
        observations: &[GardenViewBlendObservation],
        destination: usize,
        context: &str,
    ) -> usize {
        let observation = &observations[destination];
        if observation.evaluation_view == Some(observation.current_render_view)
            && observation.evaluation_target == Some(observation.current_render_target)
        {
            return destination;
        }
        if destination > 0
            && observation.evaluation_view
                == Some(observations[destination - 1].current_render_view)
            && observation.evaluation_target
                == Some(observations[destination - 1].current_render_target)
        {
            return destination - 1;
        }
        panic!(
            "{context} complete reversal publication used an evaluation older than its current/previous render view at sample {destination}"
        );
    }

    fn assert_garden_view_blend_reversal_with_activation_window(
        observations: &[GardenViewBlendObservation],
        activations: &[GardenViewBlendActivationEvidence],
        turning_index: usize,
        context: &str,
    ) {
        assert_eq!(
            observations.len(),
            activations.len(),
            "{context} observation/activation traces are torn"
        );
        assert!(
            turning_index > 0 && turning_index + 1 < observations.len(),
            "{context} turning sample has no immediate neighbors"
        );
        if activations[turning_index - 1..=turning_index + 1]
            .iter()
            .all(|activation| !activation.activation_frame)
        {
            assert_garden_view_blend_reversal(
                &observations[turning_index - 1],
                &observations[turning_index],
                &observations[turning_index + 1],
                context,
            );
            return;
        }

        assert!(
            turning_index >= 3 && turning_index + 3 < observations.len(),
            "{context} authored-boundary reversal has no symmetric seven-sample window"
        );
        let mut stable_keys = observations[turning_index - 3]
            .edges
            .iter()
            .map(|edge| edge.key.clone())
            .collect::<BTreeSet<_>>();
        for observation in &observations[turning_index - 2..=turning_index + 3] {
            let keys = observation
                .edges
                .iter()
                .map(|edge| edge.key.clone())
                .collect::<BTreeSet<_>>();
            stable_keys.retain(|key| keys.contains(key));
        }

        let mut reversed_key = None;
        for key in &stable_keys {
            let raw_before = f32::from_bits(
                garden_view_blend_edge(&observations[turning_index - 1], key)
                    .expect("stable reversal key exists before the turn")
                    .current_render_weight_bits,
            );
            let raw_turning = f32::from_bits(
                garden_view_blend_edge(&observations[turning_index], key)
                    .expect("stable reversal key exists at the turn")
                    .current_render_weight_bits,
            );
            let raw_after = f32::from_bits(
                garden_view_blend_edge(&observations[turning_index + 1], key)
                    .expect("stable reversal key exists after the turn")
                    .current_render_weight_bits,
            );
            let raw_incoming = raw_turning - raw_before;
            let raw_outgoing = raw_after - raw_turning;
            if raw_incoming.abs() <= f32::EPSILON
                || raw_outgoing.abs() <= f32::EPSILON
                || raw_incoming.signum() == raw_outgoing.signum()
            {
                continue;
            }

            let mut incoming = Vec::new();
            let mut outgoing = Vec::new();
            for destination in turning_index - 2..=turning_index + 3 {
                if activations[destination].activation_frame {
                    continue;
                }
                let current_observation = &observations[destination];
                if !current_observation.desired_evaluation_complete
                    || current_observation.status.lagging_count != 0
                    || current_observation.status.invalid_pressure_count != 0
                    || current_observation.status.missing_consumer_count != 0
                {
                    continue;
                }
                let previous_edge = garden_view_blend_edge(&observations[destination - 1], key)
                    .expect("stable reversal key exists at the previous sample");
                let current_edge = garden_view_blend_edge(current_observation, key)
                    .expect("stable reversal key exists at the destination sample");
                if current_edge.recovery_lag
                    || current_edge.displayed_weight_bits != current_edge.desired_weight_bits
                    || current_edge.evaluation_weight_bits
                        != Some(current_edge.displayed_weight_bits)
                {
                    continue;
                }
                let delta = f32::from_bits(current_edge.displayed_weight_bits)
                    - f32::from_bits(previous_edge.displayed_weight_bits);
                if delta.abs() <= f32::EPSILON {
                    continue;
                }
                let evaluation_sample =
                    garden_evaluation_sample_index(observations, destination, context);
                if evaluation_sample <= turning_index {
                    incoming.push(delta);
                } else {
                    outgoing.push(delta);
                }
            }
            if incoming
                .iter()
                .any(|incoming| incoming.signum() == raw_incoming.signum())
                && outgoing
                    .iter()
                    .any(|outgoing| outgoing.signum() == raw_outgoing.signum())
            {
                reversed_key = Some(key.clone());
                break;
            }
        }
        assert!(
            reversed_key.is_some(),
            "{context} authored-boundary window did not show one stable, oracle-backed edge reverse within three samples: stable_keys={}, activations={:?}",
            stable_keys.len(),
            &activations[turning_index - 3..=turning_index + 3],
        );
    }

    fn assert_garden_retained_dynamic_recovery_overlay(
        render: &GardenViewBlendRenderSnapshot,
        retained: &GardenExtractedCandidateProof,
        frozen: &GardenViewBlendObservation,
        context: &str,
    ) -> GardenViewBlendEdgeKey {
        assert_eq!(
            render.candidate.selection_mode,
            LodSelectionMode::Dynamic,
            "{context} retained recovery overlay did not come from extracted Dynamic settings"
        );
        assert!(
            render.candidate.prepared
                && !render.candidate.active
                && !render.candidate.transitioning
                && !render.candidate.selection_view_frozen
                && !render.candidate.failed
                && !render.candidate.view_blend_replan_requested
                && render.candidate.temporal_mode == Some(LodTemporalTransitionMode::Morphing),
            "{context} retained recovery overlay escaped its pending Dynamic phase: {:?}",
            render.candidate,
        );
        let (retained_current, candidates_are_current, _) = render.candidate.retention;
        assert!(
            retained_current && !candidates_are_current,
            "{context} recovery overlay was not protected by retained-current ownership"
        );
        assert!(
            retained.active
                && retained.selection_mode == LodSelectionMode::Frozen
                && retained.selection_view_frozen
                && !retained.failed
                && !retained.view_blend_replan_requested
                && retained.temporal_mode == Some(LodTemporalTransitionMode::Morphing),
            "{context} recovery overlay did not retain the accepted Frozen drawable"
        );
        assert_eq!(
            (
                retained.target_ranges.as_slice(),
                retained.presentation_ranges.as_slice(),
                retained.required_ranges.as_slice(),
            ),
            (
                frozen.target_ranges.as_slice(),
                frozen.presentation_ranges.as_slice(),
                frozen.required_ranges.as_slice(),
            ),
            "{context} retained recovery overlay changed its physical range backing"
        );
        assert_eq!(
            (
                render.drawable.compaction_generation,
                render.drawable.radix_publication_generation,
                render.drawable.compute_input_generation,
                (
                    render.drawable.candidate_fingerprint_primary,
                    render.drawable.candidate_fingerprint_secondary,
                    render.drawable.candidate_range_count,
                ),
                render.drawable.candidate_content_signature,
                render.drawable.candidate_atlas_allocation_epoch,
            ),
            (
                frozen.compaction_generation,
                frozen.publication_generation,
                frozen.compute_input_generation,
                frozen.candidate_fingerprint,
                frozen.candidate_content_signature,
                frozen.candidate_atlas_allocation_epoch,
            ),
            "{context} metadata-only recovery overlay changed its promoted physical identity/generation"
        );
        let view_blend = render
            .drawable
            .view_blend
            .as_ref()
            .expect("retained Dynamic recovery overlay has one promoted blend table");
        assert!(
            view_blend.desired_evaluation_complete
                && view_blend.evaluation_view == Some(render.current_render_view)
                && view_blend.evaluation_target == Some(render.current_render_target)
                && view_blend.invalid_pressure.iter().all(|invalid| !invalid),
            "{context} retained recovery overlay did not attach one complete valid current-render oracle"
        );
        assert_eq!(
            (
                view_blend.upload.immutable_table_upload_count,
                view_blend.upload.weight_write_count,
                view_blend.upload.buffer_allocation_count,
                view_blend.upload.weight_bytes_written,
                view_blend.upload.edge_count,
                view_blend.upload.word_capacity,
                view_blend.upload.last_max_delta.to_bits(),
                view_blend.upload.last_weighted_record_energy.to_bits(),
                view_blend.upload.max_weight_delta_per_frame.to_bits(),
            ),
            (
                frozen.upload.immutable_table_upload_count,
                frozen.upload.weight_write_count,
                frozen.upload.buffer_allocation_count,
                frozen.upload.weight_bytes_written,
                frozen.upload.edge_count,
                frozen.upload.word_capacity,
                frozen.upload.last_max_delta.to_bits(),
                frozen.upload.last_weighted_record_energy.to_bits(),
                frozen.upload.max_weight_delta_per_frame.to_bits(),
            ),
            "{context} desired-only recovery overlay reported physical table/suffix work"
        );
        assert_eq!(
            (
                view_blend.edges.len(),
                view_blend.weights.len(),
                view_blend.endpoints.len(),
                view_blend.recovery_lag.len(),
                view_blend.invalid_pressure.len(),
            ),
            (
                frozen.edges.len(),
                frozen.edges.len(),
                frozen.edges.len(),
                frozen.edges.len(),
                frozen.edges.len(),
            ),
            "{context} retained recovery overlay published a torn edge table"
        );
        let frozen_edges = frozen
            .edges
            .iter()
            .map(|edge| (&edge.key, edge))
            .collect::<BTreeMap<_, _>>();
        let mut lagging = 0_u32;
        let mut witness = None;
        for (((edge, weight), endpoint), recovery_lag) in view_blend
            .edges
            .iter()
            .zip(&view_blend.weights)
            .zip(&view_blend.endpoints)
            .zip(&view_blend.recovery_lag)
        {
            let key = garden_view_blend_edge_key(edge);
            let frozen_edge = frozen_edges.get(&key).unwrap_or_else(|| {
                panic!("{context} recovery overlay changed a Frozen edge key: {key:?}")
            });
            let displayed_bits = weight.displayed.to_bits();
            let desired_bits = weight.desired.to_bits();
            let (parent_pressure, child_pressure) = lod_view_blend_pressures_for_testing(
                render.current_render_view,
                render.current_render_target,
                edge,
            )
            .unwrap_or_else(|| {
                panic!(
                    "{context} retained recovery overlay used a malformed or threshold-contradictory edge: {key:?}"
                )
            });
            assert!(
                parent_pressure.is_finite()
                    && child_pressure.is_finite()
                    && !(parent_pressure <= 1.0 && child_pressure > 1.0),
                "{context} retained recovery overlay used non-finite or threshold-contradictory selector pressures for {key:?}: parent={parent_pressure}, child={child_pressure}"
            );
            let oracle_bits = lod_view_blend_weight_for_testing(
                render.current_render_view,
                render.current_render_target,
                edge,
            )
            .to_bits();
            assert_eq!(
                (
                    displayed_bits,
                    garden_view_blend_endpoint_tag(*endpoint),
                    desired_bits,
                ),
                (
                    frozen_edge.displayed_weight_bits,
                    frozen_edge.endpoint,
                    oracle_bits,
                ),
                "{context} recovery overlay changed displayed/endpoint bits or missed its exact current oracle for {key:?}"
            );
            if displayed_bits != desired_bits {
                lagging = lagging.saturating_add(1);
                assert!(
                    *recovery_lag,
                    "{context} lagging Frozen common edge omitted recovery provenance: {key:?}"
                );
                witness.get_or_insert(key);
            }
        }
        assert_eq!(
            view_blend.upload.lagging_edge_count, lagging,
            "{context} recovery overlay lag aggregate disagreed with its exact common-edge bits"
        );
        witness.unwrap_or_else(|| {
            panic!("{context} retained Dynamic overlay did not expose a Frozen common-edge lag")
        })
    }

    fn assert_garden_exact_active_dynamic_recovery_overlay(
        render: &GardenViewBlendRenderSnapshot,
        context: &str,
    ) {
        assert!(
            render.drawable.candidate_token_matches
                && render.drawable.candidate_content_matches
                && render.candidate.selection_mode == LodSelectionMode::Dynamic
                && render.candidate.prepared
                && render.candidate.active
                && !render.candidate.transitioning
                && render.candidate.selection_view_frozen
                && !render.candidate.failed
                && !render.candidate.view_blend_replan_requested
                && render.candidate.temporal_mode == Some(LodTemporalTransitionMode::Morphing),
            "{context} escaped the exact ACTIVE Frozen-frontier Dynamic-policy overlay class: {:?}",
            render.candidate,
        );
        let view_blend = render
            .drawable
            .view_blend
            .as_ref()
            .expect("exact ACTIVE Dynamic recovery overlay has one promoted blend table");
        assert!(
            view_blend.desired_evaluation_complete
                && view_blend.evaluation_view == Some(render.current_render_view)
                && view_blend.evaluation_target == Some(render.current_render_target)
                && view_blend.invalid_pressure.iter().all(|invalid| !invalid),
            "{context} exact ACTIVE recovery overlay did not attach one complete valid current-render oracle"
        );
        let aggregate = render
            .candidate
            .view_blend
            .as_ref()
            .expect("exact ACTIVE Dynamic recovery overlay has one candidate aggregate");
        assert_eq!(
            (
                aggregate.status.missing_consumer_count,
                aggregate.status.invalid_pressure_count,
            ),
            (0, 0),
            "{context} exact ACTIVE recovery overlay was not coherently aggregated"
        );
        assert!(
            view_blend.edges.iter().all(|edge| {
                lod_view_blend_pressures_for_testing(
                    render.current_render_view,
                    render.current_render_target,
                    edge,
                )
                .is_some_and(|(parent, child)| {
                    parent.is_finite() && child.is_finite() && !(parent <= 1.0 && child > 1.0)
                })
            }),
            "{context} exact ACTIVE recovery overlay contained a malformed or threshold-contradictory selector edge"
        );
    }

    fn garden_exact_frozen_resume_witness(
        resumed: &GardenViewBlendObservation,
        frozen_displayed_bits: &BTreeMap<GardenViewBlendEdgeKey, u32>,
        context: &str,
    ) -> (GardenViewBlendEdgeKey, bool) {
        resumed
            .edges
            .iter()
            .find_map(|edge| {
                let frozen_bits = *frozen_displayed_bits.get(&edge.key)?;
                if edge.current_render_weight_bits == frozen_bits
                    || edge.desired_weight_bits != edge.current_render_weight_bits
                    || edge.evaluation_weight_bits != Some(edge.current_render_weight_bits)
                {
                    return None;
                }
                let frozen_weight = f32::from_bits(frozen_bits);
                let desired = f32::from_bits(edge.desired_weight_bits);
                let mut expected = (frozen_weight
                    + (desired - frozen_weight).clamp(
                        -resumed.upload.max_weight_delta_per_frame,
                        resumed.upload.max_weight_delta_per_frame,
                    ))
                .clamp(0.0, 1.0);
                if (expected - desired).abs() <= f32::EPSILON {
                    expected = desired;
                }
                if edge.displayed_weight_bits != frozen_bits
                    && edge.displayed_weight_bits != expected.to_bits()
                {
                    return None;
                }
                let caught_up = edge.displayed_weight_bits == edge.desired_weight_bits;
                assert!(
                    caught_up || edge.recovery_lag,
                    "{context} Frozen common edge moved or retargeted without recovery provenance: {:?}",
                    edge.key,
                );
                Some((edge.key.clone(), caught_up))
            })
            .unwrap_or_else(|| {
                panic!(
                    "{context} exposed no Frozen common edge at either the retained suffix or its first exact bounded recovery step"
                )
            })
    }

    fn assert_garden_frozen_view_blend_hold_and_resume(
        app: &mut App,
        cloud: Entity,
        camera: Entity,
        scene_frame: GardenSceneFrame,
        telemetry: &mut GardenTemporalTelemetry,
    ) {
        let midpoint = GARDEN_TEMPORAL_DOLLY_SAMPLES / 2;
        *app.world_mut()
            .get_mut::<Transform>(camera)
            .expect("Garden Frozen camera transform exists") = garden_temporal_dolly_transform(
            scene_frame,
            GardenTemporalDollyDirection::Refining,
            midpoint,
        );
        assert_eq!(
            telemetry.selection_mode,
            LodSelectionMode::Dynamic,
            "Garden pre-Frozen telemetry did not begin Dynamic"
        );
        let dynamic_settled = settle_garden_roundtrip_pose(
            app,
            cloud,
            camera,
            telemetry,
            "Garden pre-Frozen midpoint",
        );
        let dynamic_midpoint = telemetry
            .last_blend
            .as_ref()
            .expect("Garden pre-Frozen midpoint has one promoted blend")
            .clone();
        dynamic_midpoint.assert_stationary_fixed_point("Garden pre-Frozen midpoint");
        assert_eq!(
            dynamic_midpoint.presentation_signature(),
            dynamic_settled.presentation,
            "Garden pre-Frozen midpoint settled a torn presentation"
        );
        let frozen_signature = dynamic_midpoint.presentation_signature();
        let frozen_upload = dynamic_midpoint.upload;
        let frozen_compaction_generation = dynamic_midpoint.compaction_generation;
        telemetry.initial_blend_signature = Some(frozen_signature.clone());
        telemetry.initial_upload = Some(frozen_upload);
        app.world_mut()
            .get_mut::<GaussianLodSettings>(cloud)
            .expect("Garden Frozen LoD settings exist")
            .selection_mode = LodSelectionMode::Frozen;

        let mut consecutive_frozen_owned = 0_u32;
        let mut frozen = None;
        for frame in 0..240 {
            app.update();
            let render = app
                .world()
                .resource::<GardenViewBlendRenderProbe>()
                .latest_snapshot()
                .expect("Garden Frozen entry has an ordered promoted drawable");
            if telemetry.selection_mode == LodSelectionMode::Dynamic
                && render.drawable.candidate_token_matches
                && render.drawable.candidate_content_matches
                && render.candidate.selection_mode == LodSelectionMode::Frozen
                && render.candidate.selection_view_frozen
            {
                telemetry.selection_mode = LodSelectionMode::Frozen;
            }
            observe_garden_temporal_promoted_frame(app, telemetry, frame, false);
            let observation = telemetry
                .last_blend
                .as_ref()
                .expect("Garden Frozen entry retains one promoted blend")
                .clone();
            observation.assert_frozen_fixed_point(&format!("Garden Frozen entry frame {frame}"));
            assert_eq!(
                observation.presentation_signature(),
                frozen_signature,
                "Garden Frozen entry changed topology or weights at frame {frame}"
            );
            assert_eq!(
                observation.upload, frozen_upload,
                "Garden Frozen entry changed blend resources at frame {frame}"
            );
            assert_eq!(
                observation.compaction_generation, frozen_compaction_generation,
                "Garden Frozen entry recreated its compaction state at frame {frame}"
            );
            if telemetry.selection_mode == LodSelectionMode::Frozen
                && garden_temporal_final_pose_is_coherent(app, cloud, camera, telemetry)
            {
                consecutive_frozen_owned = consecutive_frozen_owned.saturating_add(1);
                if consecutive_frozen_owned == 2 {
                    frozen = Some(observation);
                    break;
                }
            } else {
                consecutive_frozen_owned = 0;
            }
        }
        let frozen = frozen.expect(
            "Garden Frozen entry did not publish two consecutive exact-token/current Frozen-owned frames in 240 updates",
        );

        *app.world_mut()
            .get_mut::<Transform>(camera)
            .expect("Garden Frozen camera transform exists") = garden_temporal_dolly_transform(
            scene_frame,
            GardenTemporalDollyDirection::Refining,
            midpoint + 2,
        );
        for frame in 0..GARDEN_TEMPORAL_STABLE_FRAMES {
            app.update();
            observe_garden_temporal_promoted_frame(app, telemetry, frame, false);
            assert!(
                garden_temporal_final_pose_is_coherent(app, cloud, camera, telemetry),
                "Garden Frozen hold frame {frame} lost exact-token/current Frozen ownership"
            );
            let held = telemetry
                .last_blend
                .as_ref()
                .expect("Garden Frozen hold retains one promoted blend");
            held.assert_frozen_fixed_point(&format!("Garden Frozen hold frame {frame}"));
            assert_eq!(
                held.presentation_signature(),
                frozen_signature,
                "Frozen Garden changed topology/displayed/desired bits at frame {frame}"
            );
            assert_eq!(
                held.upload, frozen_upload,
                "Frozen Garden wrote or reallocated its blend table at frame {frame}"
            );
            assert_eq!(
                held.compaction_generation, frozen_compaction_generation,
                "Frozen Garden recreated its compaction state at frame {frame}"
            );
            assert_eq!(
                held.status.lagging_count, 0,
                "Frozen Garden reported catch-up lag at frame {frame}"
            );
        }

        app.world_mut()
            .get_mut::<GaussianLodSettings>(cloud)
            .expect("Garden Dynamic LoD settings exist")
            .selection_mode = LodSelectionMode::Dynamic;
        let mut previous = garden_view_blend_weight_map(&frozen);
        let frozen_displayed_bits = frozen
            .edges
            .iter()
            .map(|edge| (edge.key.clone(), edge.displayed_weight_bits))
            .collect::<BTreeMap<_, _>>();
        let mut recovery_witness = None;
        let mut recovery_witness_caught_up = false;
        let mut caught_up = false;
        let mut consecutive_dynamic_owned = 0_u32;
        for frame in 0..24 {
            app.update();
            let render = app
                .world()
                .resource::<GardenViewBlendRenderProbe>()
                .latest_snapshot()
                .expect("Garden Frozen-to-Dynamic recovery has an ordered promoted drawable");
            let exact_candidate = render.drawable.candidate_token_matches
                && render.drawable.candidate_content_matches;
            let exact_active_dynamic_overlay = telemetry.selection_mode == LodSelectionMode::Frozen
                && exact_candidate
                && render.candidate.selection_mode == LodSelectionMode::Dynamic
                && render.candidate.prepared
                && render.candidate.active
                && !render.candidate.transitioning
                && render.candidate.selection_view_frozen
                && !render.candidate.failed
                && !render.candidate.view_blend_replan_requested
                && render.candidate.temporal_mode == Some(LodTemporalTransitionMode::Morphing)
                && render.drawable.view_blend.as_ref().is_some_and(|blend| {
                    blend.desired_evaluation_complete
                        && blend.evaluation_view == Some(render.current_render_view)
                        && blend.evaluation_target == Some(render.current_render_target)
                });
            let retained_dynamic_overlay = telemetry.selection_mode == LodSelectionMode::Frozen
                && !exact_candidate
                && render.candidate.selection_mode == LodSelectionMode::Dynamic
                && render.drawable.view_blend.as_ref().is_some_and(|blend| {
                    blend.desired_evaluation_complete
                        && blend.evaluation_view == Some(render.current_render_view)
                        && blend.evaluation_target == Some(render.current_render_target)
                        && blend.upload.lagging_edge_count != 0
                });
            if retained_dynamic_overlay {
                let context =
                    format!("Garden Frozen-to-Dynamic retained recovery overlay frame {frame}");
                let retained = match telemetry.promoted_drawable.classify(&render, &context) {
                    GardenPromotedDrawableClass::CurrentCandidate => {
                        panic!("{context} unexpectedly classified a mismatched overlay current")
                    }
                    GardenPromotedDrawableClass::RetainedCurrent(retained) => retained,
                };
                let witness = assert_garden_retained_dynamic_recovery_overlay(
                    &render, &retained, &frozen, &context,
                );
                if let Some(previous_witness) = recovery_witness.as_ref() {
                    assert_eq!(
                        previous_witness, &witness,
                        "{context} changed the common-edge recovery witness"
                    );
                } else {
                    recovery_witness = Some(witness);
                }
                telemetry.authored_publication_hold.recovery_edges =
                    frozen.edges.iter().map(|edge| edge.key.clone()).collect();
                continue;
            }
            if exact_active_dynamic_overlay {
                let context =
                    format!("Garden Frozen-to-Dynamic exact ACTIVE recovery overlay frame {frame}");
                assert_garden_exact_active_dynamic_recovery_overlay(&render, &context);
                telemetry.authored_publication_hold.recovery_edges =
                    frozen.edges.iter().map(|edge| edge.key.clone()).collect();
                telemetry.selection_mode = LodSelectionMode::Dynamic;
            }
            if telemetry.selection_mode == LodSelectionMode::Frozen
                && exact_candidate
                && render.candidate.selection_mode == LodSelectionMode::Dynamic
                && !render.candidate.selection_view_frozen
            {
                telemetry.authored_publication_hold.recovery_edges =
                    frozen.edges.iter().map(|edge| edge.key.clone()).collect();
                telemetry.selection_mode = LodSelectionMode::Dynamic;
            }
            let bounded_recovery_edges = telemetry.authored_publication_hold.recovery_edges.clone();
            observe_garden_temporal_promoted_frame(app, telemetry, frame, false);
            let resumed = telemetry
                .last_blend
                .as_ref()
                .expect("Garden Frozen-to-Dynamic recovery retains one promoted blend")
                .clone();
            resumed.assert_dynamic_coherent(&format!(
                "Garden Frozen-to-Dynamic recovery frame {frame}"
            ));
            assert_eq!(
                resumed.upload.buffer_allocation_count, frozen_upload.buffer_allocation_count,
                "Garden Frozen-to-Dynamic recovery reallocated its blend table"
            );
            let current = garden_view_blend_weight_map(&resumed);
            for (key, displayed) in &current {
                if !bounded_recovery_edges.contains(key) {
                    continue;
                }
                if let Some(previous) = previous.get(key) {
                    assert!(
                        (displayed - previous).abs()
                            <= resumed.upload.max_weight_delta_per_frame + f32::EPSILON,
                        "Garden Frozen-to-Dynamic edge exceeded its published slew bound: key={key:?}, previous={previous}, current={displayed}, stats={:?}",
                        resumed.upload,
                    );
                }
            }
            previous = current;
            let exact_dynamic_presentation = render.drawable.candidate_token_matches
                && render.drawable.candidate_content_matches
                && render.candidate.active
                && render.candidate.selection_mode == LodSelectionMode::Dynamic
                && !render.candidate.failed
                && !render.candidate.view_blend_replan_requested
                && render.candidate.temporal_mode == Some(LodTemporalTransitionMode::Morphing);
            if exact_dynamic_presentation && resumed.desired_evaluation_complete {
                if recovery_witness.is_none() {
                    let (witness, witness_caught_up) = garden_exact_frozen_resume_witness(
                        &resumed,
                        &frozen_displayed_bits,
                        &format!("Garden Frozen-to-Dynamic recovery frame {frame}"),
                    );
                    recovery_witness = Some(witness);
                    recovery_witness_caught_up |= witness_caught_up;
                }
                let witness = recovery_witness
                    .as_ref()
                    .expect("Garden Frozen recovery witness was just established");
                if let Some(edge) = resumed.edges.iter().find(|edge| &edge.key == witness) {
                    recovery_witness_caught_up |= edge.displayed_weight_bits
                        == edge.desired_weight_bits
                        && edge.evaluation_weight_bits == Some(edge.desired_weight_bits)
                        && edge.desired_weight_bits == edge.current_render_weight_bits;
                }
            }
            let exact_dynamic_owned =
                exact_dynamic_presentation && !render.candidate.selection_view_frozen;
            if exact_dynamic_owned
                && garden_temporal_final_pose_is_coherent(app, cloud, camera, telemetry)
                && recovery_witness_caught_up
            {
                resumed.assert_stationary_fixed_point("Garden Frozen-to-Dynamic fixed point");
                consecutive_dynamic_owned = consecutive_dynamic_owned.saturating_add(1);
                if consecutive_dynamic_owned == 2 {
                    caught_up = true;
                    break;
                }
            } else {
                consecutive_dynamic_owned = 0;
            }
        }
        assert!(
            recovery_witness.is_some(),
            "Garden Dynamic resume bypassed common-edge Frozen recovery"
        );
        assert!(
            recovery_witness_caught_up,
            "Garden Frozen common-edge recovery witness never caught up exactly"
        );
        assert!(
            caught_up,
            "Garden Frozen-to-Dynamic weights did not catch up within 24 frames"
        );
    }

    fn assert_garden_dynamic_view_blend_roundtrip(manifest_path: &Path) {
        let manifest = load_canonical_garden_manifest(manifest_path);
        let node_parents = garden_node_parents(&manifest);
        let scene_frame = garden_scene_frame(&manifest);
        let package_root = manifest_path
            .parent()
            .expect("Garden manifest has a package directory");
        let manifest_name = manifest_path
            .file_name()
            .expect("Garden manifest has a file name")
            .to_string_lossy()
            .into_owned();
        let settings = garden_temporal_lod_settings(LodSelectionMode::Dynamic);
        let mut app = garden_temporal_app(package_root, garden_temporal_package_config(&settings));
        let manifest_handle: Handle<GaussianLodAsset> =
            app.world().resource::<AssetServer>().load(manifest_name);
        let cloud = app
            .world_mut()
            .spawn((
                GaussianLodHandle(manifest_handle.clone()),
                GaussianLodPackageSource::native_directory(
                    package_root.to_string_lossy().into_owned(),
                ),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::Radix,
                    radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                    ..default()
                },
                settings,
                Transform::IDENTITY,
                Visibility::Visible,
                Name::new("canonical_garden_view_blend_roundtrip"),
            ))
            .id();
        let (target, camera) = spawn_garden_temporal_camera(
            &mut app,
            scene_frame,
            GardenTemporalDollyDirection::Refining,
        );
        wait_for_garden_temporal_package_settle(
            &mut app,
            cloud,
            camera,
            &manifest_handle,
            LodSelectionMode::Dynamic,
        );

        let positions = garden_temporal_roundtrip_positions();
        // Populate the entire authenticated package path once. The 8M atlas
        // can retain all 6.67M stored records, so the measured replay below is
        // the normal all-resident class rather than the explicit late-delivery
        // fallback qualified by the native one-slot fixture.
        let mut prewarmed_required = BTreeMap::<u32, GardenPhysicalCutSignature>::new();
        let mut prewarmed_pages = BTreeSet::<u64>::new();
        for _pass in 0..4 {
            for &position in &positions {
                *app.world_mut()
                    .get_mut::<Transform>(camera)
                    .expect("Garden roundtrip camera transform exists") =
                    garden_temporal_dolly_transform(
                        scene_frame,
                        GardenTemporalDollyDirection::Refining,
                        position,
                    );
                app.update();
                if let Some(candidate) = app
                    .world()
                    .get::<LodRenderCandidates>(cloud)
                    .and_then(|candidates| candidates.get(camera))
                    .filter(|candidate| candidate.render_is_active_for_testing())
                {
                    let required = garden_physical_range_signature(
                        candidate.required_atlas_ranges_for_testing(),
                    );
                    prewarmed_pages.extend(required.iter().map(|range| range.1));
                    prewarmed_required.insert(position, required);
                }
            }
            if prewarmed_required.len() == GARDEN_TEMPORAL_DOLLY_SAMPLES as usize
                && app
                    .world()
                    .resource::<LodAtlasUploadQueue>()
                    .queued_slot_count()
                    == 0
            {
                break;
            }
        }
        let missing_positions = (0..GARDEN_TEMPORAL_DOLLY_SAMPLES)
            .filter(|position| !prewarmed_required.contains_key(position))
            .collect::<Vec<_>>();
        for position in missing_positions {
            *app.world_mut()
                .get_mut::<Transform>(camera)
                .expect("Garden roundtrip camera transform exists") =
                garden_temporal_dolly_transform(
                    scene_frame,
                    GardenTemporalDollyDirection::Refining,
                    position,
                );
            wait_for_garden_temporal_package_settle(
                &mut app,
                cloud,
                camera,
                &manifest_handle,
                LodSelectionMode::Dynamic,
            );
            let candidate = app
                .world()
                .get::<LodRenderCandidates>(cloud)
                .and_then(|candidates| candidates.get(camera))
                .filter(|candidate| candidate.render_is_active_for_testing())
                .expect("strictly settled Garden prewarm pose has one ACTIVE candidate");
            let required =
                garden_physical_range_signature(candidate.required_atlas_ranges_for_testing());
            prewarmed_pages.extend(required.iter().map(|range| range.1));
            prewarmed_required.insert(position, required);
        }
        *app.world_mut()
            .get_mut::<Transform>(camera)
            .expect("Garden roundtrip camera transform exists") =
            garden_temporal_dolly_transform(scene_frame, GardenTemporalDollyDirection::Refining, 0);
        wait_for_garden_temporal_package_settle(
            &mut app,
            cloud,
            camera,
            &manifest_handle,
            LodSelectionMode::Dynamic,
        );
        let pose_zero_candidate = app
            .world()
            .get::<LodRenderCandidates>(cloud)
            .and_then(|candidates| candidates.get(camera))
            .filter(|candidate| candidate.render_is_active_for_testing())
            .expect("final strictly settled Garden prewarm pose has one ACTIVE candidate");
        prewarmed_pages.extend(
            pose_zero_candidate
                .required_atlas_ranges_for_testing()
                .iter()
                .map(|range| range.page.0),
        );
        assert_eq!(
            prewarmed_required.len(),
            GARDEN_TEMPORAL_DOLLY_SAMPLES as usize,
            "Garden warm sweep did not observe a complete ACTIVE required union at every path pose"
        );
        assert_eq!(
            app.world()
                .resource::<LodAtlasUploadQueue>()
                .queued_slot_count(),
            0,
            "Garden warm sweep left atlas uploads queued before all-resident replay"
        );
        let resident_pages = app
            .world()
            .get::<GaussianLodPackageStatus>(cloud)
            .expect("Garden warm sweep package status exists")
            .resident_pages;
        assert!(
            resident_pages as usize >= prewarmed_pages.len(),
            "Garden warm sweep retained fewer pages than its required path union: resident={resident_pages}, required={}",
            prewarmed_pages.len(),
        );
        let initial =
            observe_garden_view_blend(&app, cloud, camera, "Garden roundtrip initial fixed point");
        initial.assert_stationary_fixed_point("Garden roundtrip initial fixed point");
        initial
            .assert_manifest_edge_topology(&node_parents, "Garden roundtrip initial fixed point");
        let initial_upload = initial.upload;
        let mut previous_observation = initial.clone();

        let mut observations = Vec::with_capacity(positions.len());
        let mut activation_evidence = Vec::with_capacity(positions.len());
        let mut authored_publication_hold = GardenAuthoredPublicationHold::default();
        let mut saw_fractional_overlap_replacement = false;
        let midpoint = GARDEN_TEMPORAL_DOLLY_SAMPLES / 2;
        for (sample, &position) in positions.iter().enumerate() {
            *app.world_mut()
                .get_mut::<Transform>(camera)
                .expect("Garden roundtrip camera transform exists") =
                garden_temporal_dolly_transform(
                    scene_frame,
                    GardenTemporalDollyDirection::Refining,
                    position,
                );
            let sample = u32::try_from(sample).expect("Garden roundtrip sample fits u32");
            app.update();
            let observation = observe_garden_view_blend(
                &app,
                cloud,
                camera,
                &format!("Garden roundtrip sample {sample} at path position {position}"),
            );
            authored_publication_hold.assert_recovery_slew_from(
                &observation,
                &previous_observation,
                true,
                &format!("Garden roundtrip sample {sample} at path position {position}"),
            );
            let activation = observation.assert_all_resident_dynamic_frame(
                &previous_observation,
                &authored_publication_hold.recovery_edges,
                &authored_publication_hold.pending_ordinary_edges,
                &node_parents,
                true,
                false,
                &format!("Garden roundtrip sample {sample} at path position {position}"),
            );
            authored_publication_hold.observe(
                &observation,
                &activation,
                &format!("Garden roundtrip sample {sample}"),
            );
            saw_fractional_overlap_replacement |= activation.preserved_fractional_overlap;
            assert_garden_public_view_blend_status(
                &app,
                cloud,
                &observation,
                &format!("Garden roundtrip sample {sample}"),
            );
            assert_eq!(
                observation.upload.buffer_allocation_count, initial_upload.buffer_allocation_count,
                "prewarmed Garden roundtrip reallocated its blend buffer at sample {sample}"
            );
            assert_eq!(
                app.world()
                    .resource::<LodAtlasUploadQueue>()
                    .queued_slot_count(),
                0,
                "Garden all-resident replay queued a late atlas upload at path position {position}"
            );
            let replay_candidate = app
                .world()
                .get::<LodRenderCandidates>(cloud)
                .and_then(|candidates| candidates.get(camera))
                .expect("Garden all-resident replay candidate remains present");
            assert_eq!(
                replay_candidate.rendered_quality_status().requested_pages,
                0,
                "Garden all-resident replay requested a late page at path position {position}"
            );
            for required in &observation.required_ranges {
                assert!(
                    prewarmed_pages.contains(&required.1),
                    "Garden replay required page {} outside the frozen prewarm union at path position {position}",
                    required.1,
                );
            }
            previous_observation = observation.clone();
            observations.push(observation);
            activation_evidence.push(activation);

            if sample as usize
                == (GARDEN_TEMPORAL_DOLLY_SAMPLES + (GARDEN_TEMPORAL_DOLLY_SAMPLES - 2 - midpoint))
                    as usize
            {
                // The midpoint is deliberately fractional and may remain
                // ACTIVE indefinitely. Canonicalize any predecessor-authored
                // PREPARED table first; then 120 frames must preserve weights,
                // leases, and every resource counter exactly.
                let mut stationary_telemetry = seed_garden_roundtrip_telemetry(
                    &app,
                    previous_observation.clone(),
                    initial_upload,
                    &node_parents,
                    std::mem::take(&mut authored_publication_hold),
                    "Garden stationary mid-band seed",
                );
                let settled = settle_garden_roundtrip_pose(
                    &mut app,
                    cloud,
                    camera,
                    &mut stationary_telemetry,
                    "Garden stationary mid-band synchronization",
                );
                let stationary = stationary_telemetry
                    .last_blend
                    .as_ref()
                    .expect("Garden stationary midpoint has one settled promoted blend")
                    .clone();
                assert_eq!(
                    stationary.presentation_signature(),
                    settled.presentation,
                    "Garden stationary midpoint settled helper returned a torn presentation"
                );
                assert!(
                    stationary.edges.iter().any(|edge| edge.endpoint == 0),
                    "Garden stationary midpoint did not exercise a fractional edge"
                );
                let stationary_signature = stationary.presentation_signature();
                let stationary_upload = stationary.upload;
                let mut final_stationary = stationary.clone();
                for frame in 0..GARDEN_TEMPORAL_STABLE_FRAMES {
                    app.update();
                    observe_garden_temporal_promoted_frame(
                        &app,
                        &mut stationary_telemetry,
                        frame,
                        false,
                    );
                    assert!(
                        garden_temporal_final_pose_is_coherent(
                            &app,
                            cloud,
                            camera,
                            &stationary_telemetry,
                        ),
                        "Garden stationary mid-band frame {frame} left its exact current fixed point"
                    );
                    let held = stationary_telemetry
                        .last_blend
                        .as_ref()
                        .expect("Garden stationary hold has one promoted blend")
                        .clone();
                    held.assert_stationary_fixed_point(&format!(
                        "Garden stationary mid-band frame {frame}"
                    ));
                    assert_eq!(
                        held.presentation_signature(),
                        stationary_signature,
                        "Garden stationary fractional topology/weights changed at frame {frame}"
                    );
                    assert_eq!(
                        held.upload, stationary_upload,
                        "Garden stationary fractional resources changed at frame {frame}"
                    );
                    assert_eq!(
                        held.compaction_generation, stationary.compaction_generation,
                        "Garden stationary fractional compaction state was recreated at frame {frame}"
                    );
                    final_stationary = held;
                }
                previous_observation = final_stationary;
                authored_publication_hold =
                    std::mem::take(&mut stationary_telemetry.authored_publication_hold);
            }
        }

        let close = (GARDEN_TEMPORAL_DOLLY_SAMPLES - 1) as usize;
        assert_garden_view_blend_reversal_with_activation_window(
            &observations,
            &activation_evidence,
            close,
            "Garden close-pose reversal",
        );
        let midpoint_turn = GARDEN_TEMPORAL_DOLLY_SAMPLES as usize
            + (GARDEN_TEMPORAL_DOLLY_SAMPLES - 2 - midpoint) as usize;
        assert_garden_view_blend_reversal_with_activation_window(
            &observations,
            &activation_evidence,
            midpoint_turn,
            "Garden mid-band reversal",
        );
        assert!(
            saw_fractional_overlap_replacement,
            "Garden roundtrip never preserved a fractional common edge while admitting a disjoint new edge"
        );
        let mut settle_telemetry = seed_garden_roundtrip_telemetry(
            &app,
            previous_observation,
            initial_upload,
            &node_parents,
            authored_publication_hold,
            "Garden roundtrip fixed-sweep seed",
        );
        // The moving trajectory deliberately gives its final 1 -> 0 cut only
        // one update. That update may be the radix-proven authored first draw;
        // complete its already-bounded two-phase handoff without adding a
        // measured sample before declaring the moving trace quiescent.
        settle_garden_roundtrip_pose(
            &mut app,
            cloud,
            camera,
            &mut settle_telemetry,
            "Garden all-resident roundtrip final-pose synchronization",
        );
        settle_telemetry
            .authored_publication_hold
            .assert_no_pending_incomplete_publication("Garden all-resident roundtrip");
        let roundtrip_authored_publications = settle_telemetry
            .authored_publication_hold
            .distinct_publications;
        let roundtrip_max_authored_hold_frames = settle_telemetry
            .authored_publication_hold
            .max_consecutive_frames;
        let mut settled_pose_signatures = BTreeMap::new();
        let mut settled_capture_positions = BTreeMap::<u32, u32>::new();
        for position in 0..GARDEN_TEMPORAL_DOLLY_SAMPLES {
            *app.world_mut()
                .get_mut::<Transform>(camera)
                .expect("Garden forward fixed-sweep camera transform exists") =
                garden_temporal_dolly_transform(
                    scene_frame,
                    GardenTemporalDollyDirection::Refining,
                    position,
                );
            let signature = settle_garden_roundtrip_pose(
                &mut app,
                cloud,
                camera,
                &mut settle_telemetry,
                &format!("Garden forward fixed-sweep pose {position}"),
            );
            if position == 0
                || position == midpoint
                || position == GARDEN_TEMPORAL_DOLLY_SAMPLES - 1
            {
                capture_garden_roundtrip_settled_pose(
                    &mut app,
                    &target,
                    position,
                    cloud,
                    camera,
                    &mut settle_telemetry,
                    &signature,
                    &format!("Garden forward fixed-sweep capture pose {position}"),
                );
                assert!(
                    settled_capture_positions
                        .insert(position, position)
                        .is_none(),
                    "Garden forward fixed sweep reused capture id {position}"
                );
            }
            assert!(
                settled_pose_signatures
                    .insert(position, signature)
                    .is_none(),
                "Garden forward fixed sweep visited pose {position} twice"
            );
        }
        for position in (0..GARDEN_TEMPORAL_DOLLY_SAMPLES).rev() {
            *app.world_mut()
                .get_mut::<Transform>(camera)
                .expect("Garden reverse fixed-sweep camera transform exists") =
                garden_temporal_dolly_transform(
                    scene_frame,
                    GardenTemporalDollyDirection::Refining,
                    position,
                );
            let signature = settle_garden_roundtrip_pose(
                &mut app,
                cloud,
                camera,
                &mut settle_telemetry,
                &format!("Garden reverse fixed-sweep pose {position}"),
            );
            assert_eq!(
                settled_pose_signatures.get(&position),
                Some(&signature),
                "Garden canonical fixed-pose topology/weights depended on approach direction at pose {position}"
            );
            if position == 0
                || position == midpoint
                || position == GARDEN_TEMPORAL_DOLLY_SAMPLES - 1
            {
                let capture_id = GARDEN_TEMPORAL_DOLLY_SAMPLES + position;
                capture_garden_roundtrip_settled_pose(
                    &mut app,
                    &target,
                    capture_id,
                    cloud,
                    camera,
                    &mut settle_telemetry,
                    &signature,
                    &format!("Garden reverse fixed-sweep capture pose {position}"),
                );
                assert!(
                    settled_capture_positions
                        .insert(capture_id, position)
                        .is_none(),
                    "Garden reverse fixed sweep reused capture id {capture_id}"
                );
            }
        }
        assert_eq!(
            settled_pose_signatures.len(),
            GARDEN_TEMPORAL_DOLLY_SAMPLES as usize,
            "Garden fixed sweeps did not settle one canonical presentation at every path pose"
        );
        assert_garden_frozen_view_blend_hold_and_resume(
            &mut app,
            cloud,
            camera,
            scene_frame,
            &mut settle_telemetry,
        );

        for _ in 0..240 {
            if app
                .world()
                .resource::<GardenTemporalCaptureSink>()
                .pending
                .is_empty()
            {
                break;
            }
            app.update();
        }
        let images = {
            let mut sink = app.world_mut().resource_mut::<GardenTemporalCaptureSink>();
            assert!(
                sink.pending.is_empty(),
                "Garden roundtrip screenshot readbacks did not drain: {:?}",
                sink.pending
            );
            std::mem::take(&mut sink.images)
        };
        let mut first_pose_image = BTreeMap::<u32, Vec<[f32; 4]>>::new();
        for (sample, position) in settled_capture_positions {
            let image = images
                .get(&sample)
                .unwrap_or_else(|| panic!("missing Garden roundtrip keyframe {sample}"));
            if let Some(first) = first_pose_image.get(&position) {
                assert_garden_temporal_stability(
                    first,
                    image,
                    &format!("matched-pose Garden roundtrip position {position}"),
                );
            } else {
                first_pose_image.insert(position, image.clone());
            }
        }
        eprintln!(
            "Garden camera-conditioned roundtrip: samples={}, distinct_poses={}, keyframes={}, blend_edges_peak={}, immutable_uploads={}, weight_writes={}, allocations={}, authored_publications={}, max_authored_hold_frames={}",
            observations.len(),
            settled_pose_signatures.len(),
            images.len(),
            observations
                .iter()
                .map(|observation| observation.status.edge_count)
                .max()
                .unwrap_or_default(),
            observations
                .last()
                .map_or(0, |observation| observation
                    .upload
                    .immutable_table_upload_count)
                .saturating_sub(initial_upload.immutable_table_upload_count),
            observations
                .last()
                .map_or(0, |observation| observation.upload.weight_write_count)
                .saturating_sub(initial_upload.weight_write_count),
            observations
                .last()
                .map_or(0, |observation| observation.upload.buffer_allocation_count)
                .saturating_sub(initial_upload.buffer_allocation_count),
            roundtrip_authored_publications,
            roundtrip_max_authored_hold_frames,
        );
    }

    fn garden_temporal_lod_settings(selection_mode: LodSelectionMode) -> GaussianLodSettings {
        let mut settings = GaussianLodSettings {
            quality: GARDEN_INTERACTIVE_REVIEW_QUALITY,
            selection_mode,
            hysteresis: VIEWER_DEFAULT_LOD_HYSTERESIS,
            ..default()
        };
        // Qualify the viewer's bounded 8M active policy. The authenticated
        // package stores 6.67M records, so the 8M resident atlas still leaves
        // real headroom for one parent/child presentation union.
        settings.budgets.max_active_gaussians = GARDEN_VIEWER_MAX_ACTIVE_GAUSSIANS;
        settings.budgets.max_resident_pages = (settings.budgets.max_resident_gaussians / 1_024)
            .try_into()
            .expect("Garden temporal resident-page capacity fits u32");
        settings
    }

    fn garden_temporal_package_config(settings: &GaussianLodSettings) -> GaussianLodPackageConfig {
        GaussianLodPackageConfig {
            max_atlas_gaussians: settings
                .budgets
                .max_resident_gaussians
                .try_into()
                .expect("Garden temporal atlas capacity fits u32"),
            max_atlas_bytes: settings.budgets.max_resident_bytes,
            streaming: GaussianStreamingSettings {
                max_concurrent_requests: 64,
                ..default()
            },
            ..default()
        }
    }

    fn garden_temporal_app(package_root: &Path, package_config: GaussianLodPackageConfig) -> App {
        let mut app = App::new();
        app.insert_resource(ClearColor(Color::BLACK))
            .insert_resource(GaussianLodBridgeConfig {
                auto_build_flat_clouds: false,
                ..default()
            })
            .insert_resource(package_config)
            .insert_resource(GardenViewBlendRenderProbe::default())
            .init_resource::<GardenTemporalCaptureSink>();
        let asset_root = package_root.to_string_lossy().into_owned();
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: asset_root.clone(),
                    processed_file_path: asset_root,
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
                .disable::<PipelinedRenderingPlugin>()
                .disable::<bevy::log::LogPlugin>(),
        );
        app.add_plugins((
            GaussianSplattingPlugin,
            ExtractResourcePlugin::<GardenViewBlendRenderProbe>::default(),
        ))
        .add_observer(on_garden_temporal_capture);
        app.sub_app_mut(RenderApp).add_systems(
            Render,
            capture_garden_view_blend_render_state
                .after(LodViewBlendPublicationLabel)
                .in_set(RenderSystems::Cleanup),
        );
        while app.plugins_state() == PluginsState::Adding {
            std::thread::yield_now();
        }
        app.finish();
        app.cleanup();
        app
    }

    fn spawn_garden_temporal_camera(
        app: &mut App,
        scene_frame: GardenSceneFrame,
        direction: GardenTemporalDollyDirection,
    ) -> (Handle<Image>, Entity) {
        let target =
            app.world_mut()
                .resource_mut::<Assets<Image>>()
                .add(Image::new_target_texture(
                    GARDEN_TEMPORAL_WIDTH,
                    GARDEN_TEMPORAL_HEIGHT,
                    TextureFormat::Rgba8UnormSrgb,
                    None,
                ));
        let camera = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                Camera::default(),
                Projection::Perspective(PerspectiveProjection {
                    far: 1_000_000.0,
                    ..default()
                }),
                RenderTarget::Image(target.clone().into()),
                garden_temporal_dolly_transform(scene_frame, direction, 0),
                Tonemapping::None,
                GaussianCamera::default(),
                Name::new("canonical_garden_temporal_dolly_camera"),
            ))
            .id();
        (target, camera)
    }

    fn garden_temporal_dolly_transform(
        scene_frame: GardenSceneFrame,
        direction: GardenTemporalDollyDirection,
        sample: u32,
    ) -> Transform {
        assert!(sample < GARDEN_TEMPORAL_DOLLY_SAMPLES);
        let path_sample = direction.path_sample(sample);
        let denominator = GARDEN_TEMPORAL_DOLLY_SAMPLES.saturating_sub(1).max(1);
        let fraction = path_sample as f32 / denominator as f32;
        // Start at the authenticated viewer auto-frame rather than the
        // deliberately distant interactive stress pose. Garden's manifest
        // bounds include sparse outliers, so 4R projects the actual scene to
        // only a few macro tiles and does not represent the normal viewer
        // zoom range this temporal gate is intended to qualify.
        let far = GARDEN_AUTO_FRAME_DISTANCE / GARDEN_SCENE_RADIUS;
        let close = 1.5_f32;
        let distance = (far.ln() + fraction * (close.ln() - far.ln())).exp();
        Transform::from_translation(scene_frame.center + Vec3::Z * distance * scene_frame.radius)
            .looking_at(scene_frame.center, Vec3::Y)
    }

    fn capture_garden_temporal_frames(
        app: &mut App,
        camera: Entity,
        target: &Handle<Image>,
        scene_frame: GardenSceneFrame,
        direction: GardenTemporalDollyDirection,
        mut package: Option<(Entity, &mut GardenTemporalTelemetry)>,
    ) -> GardenTemporalTrace {
        for sample in 0..GARDEN_TEMPORAL_DOLLY_SAMPLES {
            *app.world_mut()
                .get_mut::<Transform>(camera)
                .expect("Garden temporal camera transform exists") =
                garden_temporal_dolly_transform(scene_frame, direction, sample);
            request_garden_temporal_capture(app, target, sample);
            app.update();
            if let Some((cloud, telemetry)) = package.as_mut() {
                observe_garden_temporal_package_frame(app, *cloud, camera, telemetry, sample);
            }
        }

        // Screenshot mapping is asynchronous. Drain only already-requested
        // readbacks with the camera held at its final pose; the measured set is
        // still exactly the 48 consecutive render frames above.
        let mut final_pose_coherent = package.is_none();
        let mut final_pose_observations = 0_u32;
        for drain_frame in 0..240_u32 {
            let images_complete = app
                .world()
                .resource::<GardenTemporalCaptureSink>()
                .images
                .len()
                == GARDEN_TEMPORAL_DOLLY_SAMPLES as usize;
            if images_complete && final_pose_coherent {
                break;
            }
            app.update();
            if let Some((cloud, telemetry)) = package.as_mut() {
                observe_garden_temporal_promoted_frame(
                    app,
                    telemetry,
                    GARDEN_TEMPORAL_DOLLY_SAMPLES.saturating_add(drain_frame),
                    false,
                );
                final_pose_observations = final_pose_observations.saturating_add(1);
                final_pose_coherent =
                    garden_temporal_final_pose_is_coherent(app, *cloud, camera, telemetry);
            }
        }
        if package.is_some() {
            assert!(
                final_pose_observations > 0,
                "Garden temporal package omitted its unmeasured final-pose observation"
            );
            assert!(
                final_pose_coherent,
                "Garden temporal package did not publish one coherent owned final-pose frame during screenshot drain"
            );
        }
        let images = {
            let mut sink = app.world_mut().resource_mut::<GardenTemporalCaptureSink>();
            assert!(
                sink.pending.is_empty(),
                "Garden temporal screenshot readbacks did not complete: pending={:?}, captured={}",
                sink.pending,
                sink.images.len()
            );
            assert_eq!(
                sink.images.len(),
                GARDEN_TEMPORAL_DOLLY_SAMPLES as usize,
                "Garden temporal dolly did not capture every consecutive sample"
            );
            std::mem::take(&mut sink.images)
        };
        let mut accumulator = GardenTemporalTraceAccumulator::default();
        for sample in 0..GARDEN_TEMPORAL_DOLLY_SAMPLES {
            accumulator.push(
                images
                    .get(&sample)
                    .unwrap_or_else(|| panic!("missing Garden temporal sample {sample}"))
                    .clone(),
            );
        }
        let mut trace = accumulator.finish();
        if let Some((_, telemetry)) = package {
            trace.initial_active_candidate_count = telemetry.initial_active_candidate_count;
            trace.final_active_candidate_count = telemetry.final_active_candidate_count;
            trace.active_endpoint_changes = telemetry.active_endpoint_changes;
            trace.blend_frames = telemetry.blend_frames;
            trace.fractional_blend_frames = telemetry.fractional_blend_frames;
            trace.peak_blend_edges = telemetry.peak_blend_edges;
            trace.lagging_blend_frames = telemetry.lagging_blend_frames;
            if let (Some(initial), Some(final_upload)) =
                (telemetry.initial_upload, telemetry.last_upload)
            {
                trace.immutable_table_uploads = final_upload
                    .immutable_table_upload_count
                    .saturating_sub(initial.immutable_table_upload_count);
                trace.weight_writes = final_upload
                    .weight_write_count
                    .saturating_sub(initial.weight_write_count);
                trace.buffer_allocations = final_upload
                    .buffer_allocation_count
                    .saturating_sub(initial.buffer_allocation_count);
            }
            trace.bounded_hard_frames = telemetry.bounded_hard_frames;
        }
        trace
    }

    fn request_garden_temporal_capture(app: &mut App, target: &Handle<Image>, sample: u32) {
        let screenshot = app
            .world_mut()
            .spawn(Screenshot::image(target.clone()))
            .id();
        let previous = app
            .world_mut()
            .resource_mut::<GardenTemporalCaptureSink>()
            .pending
            .insert(screenshot, sample);
        assert!(previous.is_none(), "Garden screenshot entity was reused");
    }

    fn on_garden_temporal_capture(
        trigger: On<ScreenshotCaptured>,
        mut sink: ResMut<GardenTemporalCaptureSink>,
    ) {
        let sample = sink.pending.remove(&trigger.entity).unwrap_or_else(|| {
            panic!(
                "unregistered Garden temporal screenshot {:?}",
                trigger.entity
            )
        });
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("Garden temporal screenshot converts")
            .to_rgba8();
        assert_eq!(rgba.width(), GARDEN_TEMPORAL_WIDTH);
        assert_eq!(rgba.height(), GARDEN_TEMPORAL_HEIGHT);
        let image = linear_rgba(rgba.as_raw());
        assert_garden_bounds_fit_image_nonblank(garden_image_sanity(
            &image,
            GARDEN_TEMPORAL_WIDTH as usize,
            GARDEN_TEMPORAL_HEIGHT as usize,
        ));
        assert!(
            sink.images.insert(sample, image).is_none(),
            "Garden temporal sample {sample} was captured twice"
        );
    }

    fn wait_for_garden_temporal_package_settle(
        app: &mut App,
        cloud: Entity,
        camera: Entity,
        manifest: &Handle<GaussianLodAsset>,
        selection_mode: LodSelectionMode,
    ) -> GardenTemporalSettledBaseline {
        let mut last_quiescent = None;
        let mut stable_frames = 0_u32;
        for frame in 1..=GARDEN_TEMPORAL_MAX_SETUP_FRAMES {
            app.update();
            let world = app.world();
            if let Some(asset) = world.resource::<Assets<GaussianLodAsset>>().get(manifest) {
                assert_canonical_garden_temporal_manifest(asset.manifest());
            }
            let Some(status) = world.get::<GaussianLodPackageStatus>(cloud) else {
                continue;
            };
            assert!(
                status.failure.is_none(),
                "Garden temporal package failed during initial settle: {status:?}"
            );
            assert_eq!(status.terminal_failures, 0);
            assert!(
                world
                    .resource::<LodAtlasUploadBudgetStatus>()
                    .last_error()
                    .is_none(),
                "Garden temporal atlas upload failed: {:?}",
                world.resource::<LodAtlasUploadBudgetStatus>().last_error()
            );
            let stable = (|| {
                if status.phase != GaussianLodPackagePhase::Active
                    || world.resource::<LodAtlasUploadQueue>().queued_slot_count() != 0
                {
                    return None;
                }
                let candidate = world.get::<LodRenderCandidates>(cloud)?.get(camera)?;
                if candidate.failed()
                    || !candidate.render_is_active_for_testing()
                    || u64::from(candidate.rendered_candidate_count()) != status.active_gaussians
                    || candidate.temporal_transition_mode()
                        != Some(LodTemporalTransitionMode::Morphing)
                {
                    return None;
                }
                let blend = candidate.view_blend_testing_snapshot()?;
                if blend.status.edge_count == 0
                    || blend.status.lagging_count != 0
                    || blend.status.invalid_pressure_count != 0
                    || blend.status.missing_consumer_count != 0
                    || blend.status.max_lag.to_bits() != 0.0_f32.to_bits()
                    || blend.weights.len() != blend.status.edge_count as usize
                    || blend
                        .weights
                        .iter()
                        .any(|weight| weight.displayed.to_bits() != weight.desired.to_bits())
                    || !candidate.render_ranges().iter().all(|range| {
                        candidate
                            .required_atlas_ranges_for_testing()
                            .contains(range)
                    })
                    || !candidate.frontier().physical_ranges().iter().all(|range| {
                        candidate
                            .required_atlas_ranges_for_testing()
                            .contains(range)
                    })
                {
                    return None;
                }
                if selection_mode == LodSelectionMode::Dynamic {
                    let render =
                        garden_render_view_blend_state(app, "Garden temporal initial fixed point");
                    if render.drawable.rendered_candidate_count
                        != candidate.rendered_candidate_count()
                        || !render.drawable.candidate_token_matches
                        || !render
                            .drawable
                            .view_blend
                            .as_ref()
                            .is_some_and(|blend| blend.desired_evaluation_complete)
                    {
                        return None;
                    }
                    let render_candidate = render.candidate.clone();
                    observe_garden_view_blend_with_render_state(
                        &render_candidate,
                        render,
                        true,
                        "Garden temporal initial fixed point",
                    )
                    .assert_stationary_fixed_point("Garden temporal initial fixed point");
                }
                let published = world.get::<GaussianLodStatus>(cloud)?;
                let rendered_quality = candidate.rendered_quality_status();
                let requested_target = world.get::<GaussianLodSettings>(cloud)?.quality_target();
                if published.failure.is_some()
                    || published.source != GaussianLodSourceKind::Package
                    || published.lifecycle != GaussianLodLifecycle::Active
                    || published.selection_mode != selection_mode
                    || published.frozen_views
                        != u32::from(selection_mode == LodSelectionMode::Frozen)
                    || published.active_views != 1
                    || published.temporal_transition_mode.is_some()
                    || published.selected_gaussians != status.active_gaussians
                    || published.selected_gaussians != rendered_quality.active_gaussians
                    || published.submitted_candidates != candidate.rendered_candidate_count()
                    || published.resident_pages != status.resident_pages
                    || published.view_blend_edges != blend.status.edge_count
                    || published.view_blend_lagging_edges != 0
                    || published.view_blend_invalid_pressure_evaluations != 0
                    || published.view_blend_missing_consumers != 0
                    || published.view_blend_max_lag.to_bits() != 0.0_f32.to_bits()
                    || published.view_blend_max_delta.to_bits() != blend.status.max_delta.to_bits()
                    || published.view_blend_weighted_record_energy.to_bits()
                        != blend.status.weighted_record_energy.to_bits()
                    || published.requested_target != requested_target
                    || published.requested_target != rendered_quality.requested_target
                    || published.achieved_max_error_px
                        != Some(rendered_quality.achieved_max_error_px)
                    || published.achieved_max_target_ratio
                        != Some(rendered_quality.achieved_max_target_ratio)
                    || published.target_satisfied != Some(true)
                    || published.degradation != LodDegradation::None
                    || published.degradation != rendered_quality.degradation
                    || rendered_quality.degradation != LodDegradation::None
                    || rendered_quality.requested_pages != 0
                {
                    return None;
                }
                Some((
                    garden_temporal_cut_signature(candidate),
                    status.phase,
                    published.revision,
                    published.selected_gaussians,
                    published.submitted_candidates,
                    published.resident_pages,
                ))
            })();
            if stable.is_some() && stable == last_quiescent {
                stable_frames += 1;
            } else {
                stable_frames = u32::from(stable.is_some());
                last_quiescent = stable;
            }
            if stable_frames >= GARDEN_TEMPORAL_STABLE_FRAMES {
                return last_quiescent.expect("Garden temporal stable cut exists").0;
            }
            if frame % 600 == 0 {
                let candidate = world
                    .get::<LodRenderCandidates>(cloud)
                    .and_then(|candidates| candidates.get(camera))
                    .map(|candidate| {
                        (
                            candidate.frontier().candidate_count(),
                            candidate.frontier().physical_ranges().len(),
                            candidate.rendered_candidate_count(),
                            candidate.render_is_prepared(),
                            candidate.render_is_transitioning_for_testing(),
                            candidate.render_is_active_for_testing(),
                            candidate.temporal_transition_mode(),
                            candidate.temporal_transition_progress(),
                            candidate.temporal_transition().is_some(),
                            candidate.rendered_quality_status().requested_pages,
                        )
                    });
                eprintln!(
                    "Garden temporal {:?} initial settle frame {frame}: stable_frames={stable_frames}, status={status:?}, published={:?}, queued_uploads={}, candidate={candidate:?}",
                    selection_mode,
                    world.get::<GaussianLodStatus>(cloud),
                    world.resource::<LodAtlasUploadQueue>().queued_slot_count(),
                );
            }
        }
        panic!(
            "Garden temporal {:?} package did not settle within {} frames: status={:?}",
            selection_mode,
            GARDEN_TEMPORAL_MAX_SETUP_FRAMES,
            app.world().get::<GaussianLodPackageStatus>(cloud)
        )
    }

    fn observe_garden_temporal_package_frame(
        app: &App,
        cloud: Entity,
        camera: Entity,
        telemetry: &mut GardenTemporalTelemetry,
        sample: u32,
    ) {
        let world = app.world();
        let status = world
            .get::<GaussianLodPackageStatus>(cloud)
            .expect("Garden temporal package status remains present");
        assert!(
            status.failure.is_none(),
            "Garden temporal package failed at dolly sample {sample}: {status:?}"
        );
        assert_eq!(status.terminal_failures, 0);
        assert!(
            world
                .resource::<LodAtlasUploadBudgetStatus>()
                .last_error()
                .is_none(),
            "Garden temporal atlas upload failed at sample {sample}: {:?}",
            world.resource::<LodAtlasUploadBudgetStatus>().last_error()
        );
        let candidates = world
            .get::<LodRenderCandidates>(cloud)
            .expect("Garden temporal render candidates remain present");
        let candidate = candidates
            .get(camera)
            .expect("Garden temporal camera candidate remains present");
        assert!(
            !candidate.failed(),
            "Garden temporal candidate failed at dolly sample {sample}"
        );
        if candidate.temporal_transition_mode()
            == Some(LodTemporalTransitionMode::BoundedHardCohort)
        {
            telemetry.bounded_hard_frames += 1;
            panic!("authenticated ABI-16 Garden used a hard cohort at temporal sample {sample}");
        }
        observe_garden_temporal_promoted_frame(app, telemetry, sample, true);
        if candidate.render_is_active_for_testing() {
            if let Some(blend) = telemetry.last_blend.as_ref() {
                assert_garden_public_view_blend_status(
                    app,
                    cloud,
                    blend,
                    &format!("Garden temporal sample {sample}"),
                );
            }

            let signature = garden_temporal_cut_signature(candidate);
            telemetry.final_active_candidate_count = Some(signature.0);
            if telemetry.last_active_cut.as_ref() != Some(&signature) {
                telemetry.active_endpoint_changes += 1;
                telemetry.last_active_cut = Some(signature);
            }
        }
    }

    fn observe_garden_temporal_promoted_frame(
        app: &App,
        telemetry: &mut GardenTemporalTelemetry,
        sample: u32,
        measured_frame: bool,
    ) {
        let context = if measured_frame {
            format!("Garden temporal promoted sample {sample}")
        } else {
            format!("Garden temporal final-pose drain frame {sample}")
        };
        let render = app
            .world()
            .resource::<GardenViewBlendRenderProbe>()
            .latest_snapshot()
            .unwrap_or_else(|| {
                panic!("{context} drawable probe disappeared after initial promotion")
            });
        let drawable_unchanged = telemetry
            .last_physical_drawable
            .as_ref()
            .is_some_and(|previous| garden_promoted_drawable_state_eq(previous, &render.drawable));
        let physical_drawable = render.drawable.clone();
        let drawable_class = telemetry.promoted_drawable.classify(&render, &context);
        let exact_token_aggregate = matches!(
            &drawable_class,
            GardenPromotedDrawableClass::CurrentCandidate
        );
        let render_candidate = match drawable_class {
            GardenPromotedDrawableClass::CurrentCandidate => render.candidate.clone(),
            GardenPromotedDrawableClass::RetainedCurrent(retained) => retained,
        };
        let allow_unevaluated_late_authored = !measured_frame
            && exact_token_aggregate
            && render_candidate.prepared
            && !render_candidate.active
            && !render_candidate.transitioning;
        assert!(!render_candidate.failed, "{context} candidate failed");
        match render_candidate.temporal_mode {
            Some(LodTemporalTransitionMode::BoundedHardCohort) => {
                telemetry.bounded_hard_frames += 1;
                panic!("{context} used a hard cohort")
            }
            None => {
                assert_ne!(
                    telemetry.selection_mode,
                    LodSelectionMode::Frozen,
                    "{context} retired the Frozen view-blend table during camera motion"
                );
                telemetry
                    .authored_publication_hold
                    .assert_no_pending_incomplete_publication(&context);
                telemetry.last_blend = None;
                telemetry.authored_publication_hold.table = None;
                telemetry
                    .authored_publication_hold
                    .pending_ordinary_edges
                    .clear();
                telemetry.authored_publication_hold.recovery_edges.clear();
                telemetry.authored_publication_hold.consecutive_frames = 0;
                telemetry
                    .authored_publication_hold
                    .incomplete_distinct_publications = 0;
                telemetry
                    .authored_publication_hold
                    .last_incomplete_publication = None;
                telemetry.last_physical_drawable = Some(physical_drawable);
            }
            Some(LodTemporalTransitionMode::Morphing) => {
                assert!(
                    render_candidate.prepared,
                    "{context} promoted a Morphing output without a prepared render capability"
                );
                let blend = observe_garden_view_blend_with_render_state(
                    &render_candidate,
                    render,
                    exact_token_aggregate,
                    &context,
                );
                blend.assert_dynamic_coherent(&context);
                blend.assert_no_invalid_pressure_pairs(&context);
                blend.assert_manifest_edge_topology(&telemetry.node_parents, &context);
                match telemetry.selection_mode {
                    LodSelectionMode::Dynamic => {
                        blend.assert_active_dynamic_evaluation_complete(&context);
                        if drawable_unchanged {
                            // The ordered probe may expose one radix-proven
                            // drawable across multiple app frames. Preserve
                            // logical/public accounting, but do not count the
                            // same physical suffix as another authored hold or
                            // recovery step.
                        } else if let Some(previous) = telemetry.last_blend.as_ref() {
                            telemetry
                                .authored_publication_hold
                                .assert_recovery_slew_from(
                                    &blend,
                                    previous,
                                    exact_token_aggregate,
                                    &context,
                                );
                            let evidence = blend.assert_all_resident_dynamic_frame(
                                previous,
                                &telemetry.authored_publication_hold.recovery_edges,
                                &telemetry.authored_publication_hold.pending_ordinary_edges,
                                &telemetry.node_parents,
                                exact_token_aggregate,
                                allow_unevaluated_late_authored,
                                &context,
                            );
                            telemetry
                                .authored_publication_hold
                                .observe(&blend, &evidence, &context);
                        } else {
                            assert!(
                                blend.upload.immutable_table_upload_count > 0,
                                "{context} first blend had no immutable table upload"
                            );
                            for edge in &blend.edges {
                                assert!(
                                    !edge.activation_requires_slew,
                                    "{context} used late-readiness slew in the prewarmed trajectory"
                                );
                                assert_eq!(
                                    edge.displayed_weight_bits, edge.initial_weight_bits,
                                    "{context} first blend did not publish its authored endpoint for {:?}",
                                    edge.key,
                                );
                            }
                            let evidence = GardenViewBlendActivationEvidence {
                                activation_frame: true,
                                new_authored_publication: true,
                                preserved_fractional_overlap: false,
                                new_edge_keys: blend
                                    .edges
                                    .iter()
                                    .map(|edge| edge.key.clone())
                                    .collect(),
                                unevaluated_late_edge_keys: BTreeSet::new(),
                            };
                            telemetry
                                .authored_publication_hold
                                .observe(&blend, &evidence, &context);
                        }
                    }
                    LodSelectionMode::Frozen => {
                        assert_eq!(
                            blend.status.lagging_count, 0,
                            "{context} accumulated Frozen blend lag"
                        );
                        assert!(
                            blend.edges.iter().all(|edge| {
                                edge.displayed_weight_bits == edge.desired_weight_bits
                            }),
                            "{context} changed Frozen displayed/desired ownership"
                        );
                        assert_eq!(
                            Some(blend.presentation_signature()),
                            telemetry.initial_blend_signature,
                            "{context} changed Frozen topology or weights"
                        );
                    }
                }
                if let Some(previous) = telemetry.last_blend.as_ref().map(|blend| blend.upload) {
                    assert!(
                        blend.upload.immutable_table_upload_count
                            >= previous.immutable_table_upload_count
                            && blend.upload.weight_write_count >= previous.weight_write_count
                            && blend.upload.buffer_allocation_count
                                >= previous.buffer_allocation_count,
                        "{context} reset view-blend resource counters: previous={previous:?}, next={:?}",
                        blend.upload,
                    );
                    assert!(
                        blend.upload.word_capacity >= blend.upload.edge_count,
                        "{context} blend-buffer capacity underflowed: {:?}",
                        blend.upload,
                    );
                    if telemetry.selection_mode == LodSelectionMode::Frozen {
                        assert_eq!(
                            blend.upload, previous,
                            "{context} performed Frozen topology/weight/allocation work"
                        );
                    }
                }
                if let Some(initial) = telemetry.initial_upload {
                    assert_eq!(
                        blend.upload.buffer_allocation_count, initial.buffer_allocation_count,
                        "{context} reallocated the prewarmed view-blend buffer"
                    );
                }
                telemetry.last_blend_signature = Some(blend.presentation_signature());
                if measured_frame {
                    telemetry.blend_frames += 1;
                    telemetry.fractional_blend_frames +=
                        u32::from(blend.edges.iter().any(|edge| edge.endpoint == 0));
                    telemetry.peak_blend_edges =
                        telemetry.peak_blend_edges.max(blend.status.edge_count);
                    telemetry.lagging_blend_frames += u32::from(blend.status.lagging_count != 0);
                    telemetry.last_upload = Some(blend.upload);
                }
                telemetry.last_blend = Some(blend);
                telemetry.last_physical_drawable = Some(physical_drawable);
            }
        }
    }

    fn garden_promoted_drawable_state_eq(
        left: &LodLastRadixDrawableForTesting,
        right: &LodLastRadixDrawableForTesting,
    ) -> bool {
        let mut left = left.clone();
        let mut right = right.clone();
        left.candidate_token_matches = false;
        left.candidate_content_matches = false;
        right.candidate_token_matches = false;
        right.candidate_content_matches = false;
        left == right
    }

    fn garden_temporal_final_pose_is_coherent(
        app: &App,
        cloud: Entity,
        camera: Entity,
        telemetry: &GardenTemporalTelemetry,
    ) -> bool {
        let world = app.world();
        let Some(render) = world
            .resource::<GardenViewBlendRenderProbe>()
            .latest_snapshot()
        else {
            return false;
        };
        if !render.drawable.candidate_token_matches
            || !render.drawable.candidate_content_matches
            || !render.candidate.active
            || render.candidate.selection_mode != telemetry.selection_mode
            || render.candidate.selection_view_frozen
                != (telemetry.selection_mode == LodSelectionMode::Frozen)
            || render.candidate.failed
            || render.candidate.view_blend_replan_requested
            || render.candidate.temporal_mode != Some(LodTemporalTransitionMode::Morphing)
        {
            return false;
        }
        let Some(blend) = telemetry.last_blend.as_ref() else {
            return false;
        };
        if !telemetry
            .authored_publication_hold
            .incomplete_publication_is_clear()
            || blend.status.lagging_count != 0
            || blend.status.invalid_pressure_count != 0
            || blend.status.missing_consumer_count != 0
            || blend.status.max_lag_bits != 0.0_f32.to_bits()
        {
            return false;
        }
        match telemetry.selection_mode {
            LodSelectionMode::Dynamic => {
                if !blend.desired_evaluation_complete
                    || blend.evaluation_view != Some(render.current_render_view)
                    || blend.evaluation_target != Some(render.current_render_target)
                {
                    return false;
                }
            }
            LodSelectionMode::Frozen => {
                blend.assert_frozen_fixed_point("Garden temporal Frozen final-pose drain");
                if Some(blend.presentation_signature()) != telemetry.initial_blend_signature
                    || Some(blend.upload) != telemetry.initial_upload
                {
                    return false;
                }
            }
        }
        let Some(package) = world.get::<GaussianLodPackageStatus>(cloud) else {
            return false;
        };
        if package.failure.is_some()
            || package.phase != GaussianLodPackagePhase::Active
            || world.resource::<LodAtlasUploadQueue>().queued_slot_count() != 0
        {
            return false;
        }
        let Some(candidates) = world.get::<LodRenderCandidates>(cloud) else {
            return false;
        };
        let (_, candidates_are_current, retained_current_is_stale) =
            candidates.package_retention_for_testing();
        let Some(candidate) = candidates.get(camera) else {
            return false;
        };
        if !candidates_are_current
            || retained_current_is_stale
            || candidate.failed()
            || !candidate.render_is_active_for_testing()
            || candidate.frontier().selection_view_frozen()
                != (telemetry.selection_mode == LodSelectionMode::Frozen)
            || candidate.temporal_transition_mode() != Some(LodTemporalTransitionMode::Morphing)
            || candidate.view_blend_replan_requested_for_testing()
            || candidate.rendered_quality_status().requested_pages != 0
            || candidate.render_commit_identity_for_testing()
                != render.candidate.render_commit_identity
            || candidate.rendered_candidate_count() != render.candidate.rendered_candidate_count
            || garden_physical_range_signature(candidate.frontier().physical_ranges())
                != render.candidate.target_ranges
            || garden_physical_range_signature(candidate.render_ranges())
                != render.candidate.presentation_ranges
            || garden_physical_range_signature(candidate.required_atlas_ranges_for_testing())
                != render.candidate.required_ranges
        {
            return false;
        }
        let Some(public) = world.get::<GaussianLodStatus>(cloud) else {
            return false;
        };
        public.selection_mode == telemetry.selection_mode
            && public.target_satisfied == Some(true)
            && public.view_blend_lagging_edges == 0
            && public.view_blend_invalid_pressure_evaluations == 0
            && public.view_blend_missing_consumers == 0
    }

    fn settle_garden_roundtrip_pose(
        app: &mut App,
        cloud: Entity,
        camera: Entity,
        telemetry: &mut GardenTemporalTelemetry,
        context: &str,
    ) -> GardenRoundtripSettledSignature {
        let mut previous_qualifying = None;
        for frame in 0..240 {
            app.update();
            observe_garden_temporal_promoted_frame(app, telemetry, frame, false);
            if !garden_temporal_final_pose_is_coherent(app, cloud, camera, telemetry) {
                previous_qualifying = None;
                continue;
            }
            let blend = telemetry
                .last_blend
                .as_ref()
                .expect("coherent Garden fixed pose has a promoted blend");
            blend.assert_stationary_fixed_point(&format!("{context} qualifying frame {frame}"));
            let candidate = app
                .world()
                .get::<LodRenderCandidates>(cloud)
                .and_then(|candidates| candidates.get(camera))
                .expect("coherent Garden fixed pose has one MainWorld candidate");
            let signature = GardenRoundtripSettledSignature {
                logical_cut: garden_temporal_cut_signature(candidate),
                presentation: blend.presentation_signature(),
            };
            let qualifying = (signature.clone(), blend.upload);
            if previous_qualifying.as_ref() == Some(&qualifying) {
                return signature;
            }
            previous_qualifying = Some(qualifying);
        }
        panic!(
            "{context} did not produce two consecutive identical coherent publications in 240 updates"
        );
    }

    fn capture_garden_roundtrip_settled_pose(
        app: &mut App,
        target: &Handle<Image>,
        capture_id: u32,
        cloud: Entity,
        camera: Entity,
        telemetry: &mut GardenTemporalTelemetry,
        settled: &GardenRoundtripSettledSignature,
        context: &str,
    ) {
        let baseline_upload = telemetry
            .last_blend
            .as_ref()
            .expect("settled Garden capture has a baseline blend")
            .upload;
        request_garden_temporal_capture(app, target, capture_id);
        app.update();
        observe_garden_temporal_promoted_frame(app, telemetry, capture_id, false);
        assert!(
            garden_temporal_final_pose_is_coherent(app, cloud, camera, telemetry),
            "{context} left its coherent fixed point during screenshot capture"
        );
        let blend = telemetry
            .last_blend
            .as_ref()
            .expect("settled Garden screenshot has a promoted blend");
        blend.assert_stationary_fixed_point(context);
        let candidate = app
            .world()
            .get::<LodRenderCandidates>(cloud)
            .and_then(|candidates| candidates.get(camera))
            .expect("settled Garden screenshot has one MainWorld candidate");
        assert_eq!(
            (
                garden_temporal_cut_signature(candidate),
                blend.presentation_signature(),
                blend.upload,
            ),
            (
                settled.logical_cut.clone(),
                settled.presentation.clone(),
                baseline_upload,
            ),
            "{context} changed its settled cut/presentation/resources on the captured frame"
        );
    }

    fn seed_garden_roundtrip_telemetry(
        app: &App,
        previous: GardenViewBlendObservation,
        initial_upload: LodViewBlendUploadStats,
        node_parents: &GardenNodeParents,
        authored_publication_hold: GardenAuthoredPublicationHold,
        context: &str,
    ) -> GardenTemporalTelemetry {
        let seed_render = app
            .world()
            .resource::<GardenViewBlendRenderProbe>()
            .latest_snapshot()
            .unwrap_or_else(|| panic!("{context} has no ordered promoted drawable"));
        let mut promoted_drawable = GardenPromotedDrawableTracker::default();
        assert!(
            matches!(
                promoted_drawable.classify(&seed_render, context),
                GardenPromotedDrawableClass::CurrentCandidate
            ),
            "{context} retained an older drawable"
        );
        GardenTemporalTelemetry {
            selection_mode: LodSelectionMode::Dynamic,
            initial_upload: Some(initial_upload),
            last_blend_signature: Some(previous.presentation_signature()),
            last_upload: Some(previous.upload),
            last_blend: Some(previous),
            last_physical_drawable: Some(seed_render.drawable),
            promoted_drawable,
            node_parents: node_parents.clone(),
            authored_publication_hold,
            ..default()
        }
    }

    fn garden_temporal_cut_signature(
        candidate: &bevy_gaussian_splatting::stream::render_commit::LodRenderCandidate,
    ) -> GardenTemporalCutSignature {
        (
            candidate.frontier().candidate_count(),
            candidate
                .frontier()
                .physical_ranges()
                .iter()
                .map(|range| (range.node.0, range.page.0, range.count))
                .collect(),
        )
    }

    fn load_canonical_garden_manifest(manifest_path: &Path) -> GaussianLodManifest {
        let encoded = fs::read(manifest_path).unwrap_or_else(|error| {
            panic!(
                "failed to read canonical Garden manifest {}: {error}",
                manifest_path.display()
            )
        });
        assert_canonical_garden_manifest_bytes(&encoded);
        let manifest = decode_manifest(&encoded, LodCodecLimits::default())
            .expect("canonical Garden manifest decodes and validates");
        assert_canonical_garden_manifest(&manifest);
        manifest
    }

    fn assert_canonical_garden_temporal_manifest(manifest: &GaussianLodManifest) {
        assert_canonical_garden_manifest(manifest);
    }

    fn garden_scene_frame(manifest: &GaussianLodManifest) -> GardenSceneFrame {
        let bounds = manifest
            .scene_bounds
            .expect("authenticated Garden manifest carries scene bounds");
        let frame = GardenSceneFrame {
            center: Vec3::from_array(bounds.center()),
            radius: bounds.radius(),
        };
        assert!(
            frame.center.is_finite() && frame.radius.is_finite() && frame.radius > 0.0,
            "Garden scene frame is invalid: {frame:?}"
        );
        frame
    }

    #[derive(Resource)]
    struct GardenPackageStaticState {
        package_root: PathBuf,
        manifest_name: String,
        source_path: PathBuf,
        settings: GaussianLodSettings,
        scene_frame: GardenSceneFrame,
        node_parents: GardenNodeParents,
        manifest: Option<Handle<GaussianLodAsset>>,
        target: Option<Handle<Image>>,
        cloud: Option<Entity>,
        camera: Option<Entity>,
        total_frames: u32,
        phase_frames: u32,
        stable_target_frames: u32,
        cut_changes: u32,
        bounded_hard_frames: u32,
        peak_resident_pages: u32,
        last_resident_pages: Option<u32>,
        last_status_revision: Option<u64>,
        last_signature: Option<GardenCutSignature>,
        last_blend_signature: Option<GardenViewBlendPresentationSignature>,
        last_blend_status: Option<GardenViewBlendStatusObservation>,
        last_blend_upload: Option<LodViewBlendUploadStats>,
        last_compaction_generation: Option<u64>,
        promoted_drawable: GardenPromotedDrawableTracker,
        capture_phase: GardenCapturePhase,
        capture_gap_frames: u32,
        capture_samples: u32,
        first_capture: Option<Vec<[f32; 4]>>,
        first_sanity: Option<GardenImageSanity>,
    }

    impl GardenPackageStaticState {
        fn new(
            package_root: PathBuf,
            manifest_name: String,
            source_path: PathBuf,
            settings: GaussianLodSettings,
            scene_frame: GardenSceneFrame,
            node_parents: GardenNodeParents,
        ) -> Self {
            Self {
                package_root,
                manifest_name,
                source_path,
                settings,
                scene_frame,
                node_parents,
                manifest: None,
                target: None,
                cloud: None,
                camera: None,
                total_frames: 0,
                phase_frames: 0,
                stable_target_frames: 0,
                cut_changes: 0,
                bounded_hard_frames: 0,
                peak_resident_pages: 0,
                last_resident_pages: None,
                last_status_revision: None,
                last_signature: None,
                last_blend_signature: None,
                last_blend_status: None,
                last_blend_upload: None,
                last_compaction_generation: None,
                promoted_drawable: GardenPromotedDrawableTracker::default(),
                capture_phase: GardenCapturePhase::AwaitingStableCut,
                capture_gap_frames: 0,
                capture_samples: 0,
                first_capture: None,
                first_sanity: None,
            }
        }
    }

    type GardenPhysicalCutSignature = Vec<(u64, u64, u32, u32, u32, u32)>;
    type GardenNodeParents = BTreeMap<u64, Option<u64>>;

    fn garden_node_parents(manifest: &GaussianLodManifest) -> GardenNodeParents {
        manifest
            .nodes
            .iter()
            .map(|node| (node.id.0, node.parent.map(|parent| parent.0)))
            .collect()
    }

    fn garden_node_is_descendant_or_same(
        node: u64,
        ancestor: u64,
        parents: &GardenNodeParents,
    ) -> bool {
        let mut current = Some(node);
        while let Some(node) = current {
            if node == ancestor {
                return true;
            }
            current = parents.get(&node).copied().flatten();
        }
        false
    }

    fn garden_required_node_range(
        ranges: &GardenPhysicalCutSignature,
        node: u64,
        context: &str,
    ) -> (u64, u64, u32, u32, u32, u32) {
        let matches = ranges
            .iter()
            .filter(|range| range.0 == node)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "{context} expected one physical backing range for node {node}: {matches:?}"
        );
        matches[0]
    }

    #[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct GardenViewBlendMetricSignature {
        center_bits: [u32; 3],
        radius_bits: u32,
        geometric_error_bits: u32,
        quality_min_bits: u32,
        quality_max_bits: u32,
        high_fidelity_certificate_bits: u32,
        original_representation: bool,
    }

    impl From<LodViewBlendMetric> for GardenViewBlendMetricSignature {
        fn from(metric: LodViewBlendMetric) -> Self {
            let metrics = metric.node_metrics();
            Self {
                center_bits: metrics.center.to_array().map(f32::to_bits),
                radius_bits: metrics.radius.to_bits(),
                geometric_error_bits: metrics.geometric_error.to_bits(),
                quality_min_bits: metrics.quality_min.to_bits(),
                quality_max_bits: metrics.quality_max.to_bits(),
                high_fidelity_certificate_bits: metrics.high_fidelity_certificate.to_bits(),
                original_representation: metric.is_original_representation(),
            }
        }
    }

    #[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct GardenViewBlendEdgeKey {
        parent: u64,
        children: Vec<u64>,
        parent_metric: GardenViewBlendMetricSignature,
        child_metrics: Vec<GardenViewBlendMetricSignature>,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct GardenViewBlendEdgeObservation {
        key: GardenViewBlendEdgeKey,
        batch_index: u32,
        record_count: u32,
        initial_weight_bits: u32,
        activation_requires_slew: bool,
        recovery_lag: bool,
        endpoint: u8,
        displayed_weight_bits: u32,
        desired_weight_bits: u32,
        pressure_bits: Option<(u32, u32)>,
        evaluation_weight_bits: Option<u32>,
        current_view_weight_bits: Option<u32>,
        current_render_weight_bits: u32,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct GardenViewBlendStatusObservation {
        edge_count: u32,
        lagging_count: u32,
        invalid_pressure_count: u32,
        missing_consumer_count: u32,
        max_lag_bits: u32,
        max_delta_bits: u32,
        weighted_record_energy_bits: u32,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct GardenViewBlendPresentationSignature {
        target_ranges: GardenPhysicalCutSignature,
        presentation_ranges: GardenPhysicalCutSignature,
        required_ranges: GardenPhysicalCutSignature,
        edges: Vec<(GardenViewBlendEdgeKey, u8, u32, u32)>,
    }

    #[derive(Clone, Debug, PartialEq)]
    struct GardenViewBlendObservation {
        compaction_generation: u64,
        publication_generation: u64,
        compute_input_generation: u64,
        candidate_prepared: bool,
        candidate_active: bool,
        candidate_transitioning: bool,
        candidate_fingerprint: (Option<u64>, Option<u64>, Option<u32>),
        candidate_content_signature: Option<u64>,
        candidate_atlas_allocation_epoch: Option<u64>,
        desired_evaluation_complete: bool,
        evaluation_view: Option<LodView>,
        evaluation_target: Option<LodQualityTarget>,
        current_render_view: LodView,
        current_render_target: LodQualityTarget,
        status: GardenViewBlendStatusObservation,
        target_ranges: GardenPhysicalCutSignature,
        presentation_ranges: GardenPhysicalCutSignature,
        required_ranges: GardenPhysicalCutSignature,
        edges: Vec<GardenViewBlendEdgeObservation>,
        upload: LodViewBlendUploadStats,
    }

    #[derive(Clone, Debug, Default, Eq, PartialEq)]
    struct GardenViewBlendActivationEvidence {
        activation_frame: bool,
        new_authored_publication: bool,
        preserved_fractional_overlap: bool,
        new_edge_keys: BTreeSet<GardenViewBlendEdgeKey>,
        unevaluated_late_edge_keys: BTreeSet<GardenViewBlendEdgeKey>,
    }

    type GardenAuthoredTableIdentity = (u64, u64, Vec<GardenViewBlendEdgeKey>);

    #[derive(Clone, Debug, PartialEq)]
    struct GardenIncompletePublicationIdentity {
        compaction_generation: u64,
        publication_generation: u64,
        compute_input_generation: u64,
        candidate_phase: (bool, bool, bool),
        candidate_fingerprint: (Option<u64>, Option<u64>, Option<u32>),
        candidate_content_signature: Option<u64>,
        candidate_atlas_allocation_epoch: Option<u64>,
        evaluation_view: Option<LodView>,
        evaluation_target: Option<LodQualityTarget>,
        status: GardenViewBlendStatusObservation,
        target_ranges: GardenPhysicalCutSignature,
        presentation_ranges: GardenPhysicalCutSignature,
        required_ranges: GardenPhysicalCutSignature,
        upload: (u64, u64, u64, u64, u32, u32, u32, u32, u64),
        edges: Vec<(
            GardenViewBlendEdgeKey,
            bool,
            bool,
            u8,
            u32,
            u32,
            Option<u32>,
        )>,
    }

    impl GardenIncompletePublicationIdentity {
        fn from_observation(observation: &GardenViewBlendObservation) -> Self {
            Self {
                compaction_generation: observation.compaction_generation,
                publication_generation: observation.publication_generation,
                compute_input_generation: observation.compute_input_generation,
                candidate_phase: (
                    observation.candidate_prepared,
                    observation.candidate_active,
                    observation.candidate_transitioning,
                ),
                candidate_fingerprint: observation.candidate_fingerprint,
                candidate_content_signature: observation.candidate_content_signature,
                candidate_atlas_allocation_epoch: observation.candidate_atlas_allocation_epoch,
                evaluation_view: observation.evaluation_view,
                evaluation_target: observation.evaluation_target,
                status: observation.status.clone(),
                target_ranges: observation.target_ranges.clone(),
                presentation_ranges: observation.presentation_ranges.clone(),
                required_ranges: observation.required_ranges.clone(),
                upload: (
                    observation.upload.immutable_table_upload_count,
                    observation.upload.weight_write_count,
                    observation.upload.buffer_allocation_count,
                    observation.upload.weight_bytes_written,
                    observation.upload.edge_count,
                    observation.upload.word_capacity,
                    observation.upload.lagging_edge_count,
                    observation.upload.last_max_delta.to_bits(),
                    observation.upload.last_weighted_record_energy.to_bits(),
                ),
                edges: observation
                    .edges
                    .iter()
                    .map(|edge| {
                        (
                            edge.key.clone(),
                            edge.activation_requires_slew,
                            edge.recovery_lag,
                            edge.endpoint,
                            edge.displayed_weight_bits,
                            edge.desired_weight_bits,
                            edge.evaluation_weight_bits,
                        )
                    })
                    .collect(),
            }
        }
    }

    #[derive(Clone, Debug, Default, PartialEq)]
    struct GardenAuthoredPublicationHold {
        table: Option<GardenAuthoredTableIdentity>,
        pending_ordinary_edges: BTreeMap<GardenViewBlendEdgeKey, u32>,
        recovery_edges: BTreeSet<GardenViewBlendEdgeKey>,
        consecutive_frames: u32,
        max_consecutive_frames: u32,
        distinct_publications: u32,
        incomplete_distinct_publications: u32,
        last_incomplete_publication: Option<GardenIncompletePublicationIdentity>,
    }

    impl GardenAuthoredPublicationHold {
        fn observe(
            &mut self,
            observation: &GardenViewBlendObservation,
            evidence: &GardenViewBlendActivationEvidence,
            context: &str,
        ) {
            if evidence.new_authored_publication {
                self.incomplete_distinct_publications = 0;
                self.last_incomplete_publication = None;
                let current_keys = observation
                    .edges
                    .iter()
                    .map(|edge| edge.key.clone())
                    .collect::<BTreeSet<_>>();
                self.pending_ordinary_edges
                    .retain(|key, _| current_keys.contains(key));
                self.recovery_edges.retain(|key| current_keys.contains(key));
                for key in &evidence.new_edge_keys {
                    let edge = observation
                        .edges
                        .iter()
                        .find(|edge| &edge.key == key)
                        .expect("new authored edge is present in its publication");
                    if !edge.activation_requires_slew {
                        self.pending_ordinary_edges
                            .insert(key.clone(), edge.initial_weight_bits);
                    } else {
                        self.recovery_edges.insert(key.clone());
                    }
                }
            }
            if observation.desired_evaluation_complete {
                self.pending_ordinary_edges.clear();
                self.incomplete_distinct_publications = 0;
                self.last_incomplete_publication = None;
            } else {
                let inherited_only_guard =
                    evidence.new_authored_publication && evidence.new_edge_keys.is_empty();
                let unevaluated_late_guard = !evidence.unevaluated_late_edge_keys.is_empty();
                let incomplete_publication =
                    GardenIncompletePublicationIdentity::from_observation(observation);
                if self.last_incomplete_publication.as_ref() != Some(&incomplete_publication) {
                    self.incomplete_distinct_publications =
                        self.incomplete_distinct_publications.saturating_add(1);
                    self.last_incomplete_publication = Some(incomplete_publication);
                }
                assert!(
                    self.incomplete_distinct_publications <= 2,
                    "{context} published more than two distinct incomplete suffixes for one authored table"
                );
                assert!(
                    evidence.activation_frame,
                    "{context} published an incomplete desired table without classifying its authored guard"
                );
                assert!(
                    !self.pending_ordinary_edges.is_empty()
                        || inherited_only_guard
                        || unevaluated_late_guard,
                    "{context} published an incomplete desired table without an ordinary, inherited-only, or PREPARED late-edge authored guard"
                );
                if inherited_only_guard || unevaluated_late_guard {
                    assert_eq!(
                        (observation.evaluation_view, observation.evaluation_target),
                        (None, None),
                        "{context} unevaluated authored publication exposed partial evaluation metadata"
                    );
                }
                self.pending_ordinary_edges.retain(|key, authored_bits| {
                    let edge = observation
                        .edges
                        .iter()
                        .find(|edge| &edge.key == key)
                        .expect("pending authored edge remains in the current table");
                    if (edge.displayed_weight_bits, edge.desired_weight_bits)
                        == (*authored_bits, *authored_bits)
                    {
                        assert!(
                            !edge.activation_requires_slew,
                            "{context} ordinary authored guard gained late-recovery provenance: {:?}",
                            edge.key,
                        );
                        true
                    } else {
                        assert_eq!(
                            edge.evaluation_weight_bits,
                            Some(edge.desired_weight_bits),
                            "{context} carried authored edge left its endpoint without retargeting the captured-view oracle: {:?}",
                            edge.key,
                        );
                        false
                    }
                });
                for edge in &observation.edges {
                    if self.pending_ordinary_edges.contains_key(&edge.key) {
                        continue;
                    }
                    if let Some(evaluation_weight_bits) = edge.evaluation_weight_bits {
                        assert_eq!(
                            evaluation_weight_bits, edge.desired_weight_bits,
                            "{context} left a non-guarded common/late edge off its captured-view desired oracle: {:?}",
                            edge.key,
                        );
                    } else {
                        assert!(
                            evidence.new_authored_publication
                                && evidence.activation_frame
                                && !observation.desired_evaluation_complete
                                && observation.evaluation_view.is_none()
                                && observation.evaluation_target.is_none()
                                && (!self.pending_ordinary_edges.is_empty()
                                    || inherited_only_guard
                                    || unevaluated_late_guard),
                            "{context} omitted a non-guarded common/late edge oracle outside a coherent unevaluated authored hold: {:?}",
                            edge.key,
                        );
                    }
                }
            }
            self.recovery_edges.retain(|key| {
                observation.edges.iter().any(|edge| {
                    if &edge.key != key {
                        return false;
                    }
                    if edge.displayed_weight_bits != edge.desired_weight_bits {
                        assert!(
                            edge.recovery_lag,
                            "{context} lagging recovery edge lost its mutable recovery marker: {:?}",
                            edge.key,
                        );
                        true
                    } else {
                        edge.recovery_lag && edge.evaluation_weight_bits.is_none()
                    }
                })
            });
            if !evidence.activation_frame {
                self.table = None;
                self.consecutive_frames = 0;
                return;
            }
            let table = (
                observation.compaction_generation,
                observation.upload.immutable_table_upload_count,
                observation
                    .edges
                    .iter()
                    .map(|edge| edge.key.clone())
                    .collect::<Vec<_>>(),
            );
            if self.table.is_none() {
                self.table = Some(table);
                self.consecutive_frames = 1;
                self.distinct_publications += 1;
            } else if self.table.as_ref() == Some(&table) {
                self.consecutive_frames += 1;
            } else {
                assert!(
                    evidence.new_authored_publication,
                    "{context} changed authored table identity without an immutable-table publication"
                );
                self.table = Some(table);
                self.consecutive_frames = 1;
                self.distinct_publications += 1;
            }
            self.max_consecutive_frames = self.max_consecutive_frames.max(self.consecutive_frames);
        }

        fn assert_recovery_slew_from(
            &mut self,
            observation: &GardenViewBlendObservation,
            previous: &GardenViewBlendObservation,
            exact_token_candidate: bool,
            context: &str,
        ) {
            let table_changed = observation.upload.immutable_table_upload_count
                != previous.upload.immutable_table_upload_count;
            let previous_edges = previous
                .edges
                .iter()
                .map(|edge| (&edge.key, edge))
                .collect::<BTreeMap<_, _>>();
            if !table_changed {
                for edge in &observation.edges {
                    if !edge.recovery_lag || self.recovery_edges.contains(&edge.key) {
                        continue;
                    }
                    let previous_edge = previous_edges.get(&edge.key).unwrap_or_else(|| {
                        panic!(
                            "{context} armed recovery on a new edge without an authored table publication: {:?}",
                            edge.key,
                        )
                    });
                    let same_recovery_layout = (
                        edge.batch_index,
                        edge.record_count,
                        edge.initial_weight_bits,
                        edge.activation_requires_slew,
                    ) == (
                        previous_edge.batch_index,
                        previous_edge.record_count,
                        previous_edge.initial_weight_bits,
                        previous_edge.activation_requires_slew,
                    );
                    let previous_unevaluated_hold = previous.candidate_prepared
                        && !previous.candidate_active
                        && !previous.candidate_transitioning
                        && !previous.desired_evaluation_complete
                        && previous.evaluation_view.is_none()
                        && previous.evaluation_target.is_none()
                        && previous_edge.evaluation_weight_bits.is_none()
                        && previous_edge.displayed_weight_bits == previous_edge.desired_weight_bits;
                    let current_unevaluated_hold = observation.candidate_prepared
                        && !observation.candidate_active
                        && !observation.candidate_transitioning
                        && !observation.desired_evaluation_complete
                        && observation.evaluation_view.is_none()
                        && observation.evaluation_target.is_none()
                        && edge.evaluation_weight_bits.is_none()
                        && edge.displayed_weight_bits == edge.desired_weight_bits;
                    if self.pending_ordinary_edges.contains_key(&edge.key) {
                        // A recovered authored first draw may consume its guard
                        // without moving. Its promoted marker becomes the prior
                        // proof when a later publication takes the first step.
                        assert!(
                            edge.displayed_weight_bits == previous_edge.displayed_weight_bits
                                && edge.displayed_weight_bits != edge.desired_weight_bits
                                && exact_token_candidate
                                && observation.candidate_prepared
                                && !observation.candidate_transitioning
                                && observation.evaluation_view
                                    == Some(observation.current_render_view)
                                && observation.evaluation_target
                                    == Some(observation.current_render_target)
                                && observation.status.invalid_pressure_count == 0
                                && observation.status.missing_consumer_count == 0
                                && edge.evaluation_weight_bits == Some(edge.desired_weight_bits)
                                && edge.current_render_weight_bits == edge.desired_weight_bits,
                            "{context} recovered an ordinary authored first draw without an exact current-view lagging oracle: {:?}",
                            edge.key,
                        );
                        self.recovery_edges.insert(edge.key.clone());
                        continue;
                    }
                    if current_unevaluated_hold {
                        // The invalid-pressure live state normally cannot publish,
                        // but another dirty reason may radix-promote its table-wide
                        // recovery marker before the next valid-view oracle. Keep
                        // that exact PREPARED hold as provenance for the later
                        // bounded step.
                        assert!(
                            exact_token_candidate
                                && observation.compaction_generation
                                    == previous.compaction_generation
                                && observation.status.invalid_pressure_count == 0
                                && observation.status.missing_consumer_count == 0
                                && edge.displayed_weight_bits
                                    == previous_edge.displayed_weight_bits
                                && same_recovery_layout,
                            "{context} published an unauthenticated unevaluated recovery hold: key={:?}, previous={previous_edge:?}, current={edge:?}",
                            edge.key,
                        );
                        self.recovery_edges.insert(edge.key.clone());
                        continue;
                    }
                    // An invalid pressure pair can occur in the prior extracted
                    // view while its PREPARED authored suffix remains unevaluated
                    // (with or without its recovery marker). The later valid view
                    // re-arms recovery table-wide. Authenticate exactly that hold
                    // boundary before accepting the promoted bounded step.
                    assert!(
                        exact_token_candidate && previous_unevaluated_hold && same_recovery_layout,
                        "{context} armed recovery without a prior PREPARED unevaluated hold: key={:?}, previous={previous_edge:?}, current={edge:?}",
                        edge.key,
                    );
                    assert!(
                        observation.compaction_generation == previous.compaction_generation
                            && observation.candidate_prepared
                            && !observation.candidate_transitioning
                            && observation.evaluation_view == Some(observation.current_render_view)
                            && observation.evaluation_target
                                == Some(observation.current_render_target)
                            && observation.status.invalid_pressure_count == 0
                            && observation.status.missing_consumer_count == 0
                            && edge.evaluation_weight_bits == Some(edge.desired_weight_bits)
                            && edge.current_render_weight_bits == edge.desired_weight_bits
                            && edge.displayed_weight_bits != edge.desired_weight_bits,
                        "{context} recovered an unpublished invalid hold without an exact current-view oracle: key={:?}, current={edge:?}",
                        edge.key,
                    );
                    self.recovery_edges.insert(edge.key.clone());
                }
            }
            for key in &self.recovery_edges {
                let Some(edge) = observation.edges.iter().find(|edge| &edge.key == key) else {
                    continue;
                };
                let Some(previous) = previous_edges.get(key) else {
                    continue;
                };
                if edge.displayed_weight_bits == previous.displayed_weight_bits {
                    continue;
                }
                assert_eq!(
                    edge.evaluation_weight_bits,
                    Some(edge.desired_weight_bits),
                    "{context} moved a recovery edge without its exact captured-view oracle: key={key:?}"
                );
                assert_eq!(
                    edge.recovery_lag,
                    edge.displayed_weight_bits != edge.desired_weight_bits,
                    "{context} moved a recovery edge with an untruthful mutable lag marker: key={key:?}"
                );
                let previous_displayed = f32::from_bits(previous.displayed_weight_bits);
                let current_desired = f32::from_bits(edge.desired_weight_bits);
                let mut expected = (previous_displayed
                    + (current_desired - previous_displayed).clamp(
                        -observation.upload.max_weight_delta_per_frame,
                        observation.upload.max_weight_delta_per_frame,
                    ))
                .clamp(0.0, 1.0);
                if (expected - current_desired).abs() <= f32::EPSILON {
                    expected = current_desired;
                }
                assert!(
                    (f32::from_bits(edge.displayed_weight_bits) - previous_displayed).abs()
                        <= observation.upload.max_weight_delta_per_frame + f32::EPSILON,
                    "{context} recovery edge exceeded its bounded step: key={key:?}"
                );
                assert_eq!(
                    edge.displayed_weight_bits,
                    expected.to_bits(),
                    "{context} recovery edge did not consume one exact bounded step toward its current captured-view desired weight: key={key:?}, previous_displayed={previous_displayed}, current_desired={current_desired}, current={}",
                    f32::from_bits(edge.displayed_weight_bits),
                );
            }
        }

        fn assert_no_pending_incomplete_publication(&self, context: &str) {
            assert!(
                self.pending_ordinary_edges.is_empty(),
                "{context} ended before its authored edge guards became fully evaluated: {:?}",
                self.pending_ordinary_edges.keys().collect::<Vec<_>>(),
            );
            assert_eq!(
                self.incomplete_distinct_publications, 0,
                "{context} ended with incomplete authored-publication cadence state"
            );
            assert!(
                self.last_incomplete_publication.is_none(),
                "{context} ended with a retained incomplete publication identity"
            );
        }

        fn incomplete_publication_is_clear(&self) -> bool {
            self.pending_ordinary_edges.is_empty()
                && self.incomplete_distinct_publications == 0
                && self.last_incomplete_publication.is_none()
        }
    }

    impl GardenViewBlendObservation {
        fn presentation_signature(&self) -> GardenViewBlendPresentationSignature {
            GardenViewBlendPresentationSignature {
                target_ranges: self.target_ranges.clone(),
                presentation_ranges: self.presentation_ranges.clone(),
                required_ranges: self.required_ranges.clone(),
                edges: self
                    .edges
                    .iter()
                    .map(|edge| {
                        (
                            edge.key.clone(),
                            edge.endpoint,
                            edge.displayed_weight_bits,
                            edge.desired_weight_bits,
                        )
                    })
                    .collect(),
            }
        }

        fn assert_live_dynamic_exact(&self, context: &str) {
            self.assert_dynamic_coherent(context);
            assert!(
                self.desired_evaluation_complete,
                "{context} used the initial retained-endpoint publication instead of a live evaluated Dynamic suffix"
            );
            self.assert_no_invalid_pressure_pairs(context);
            assert_eq!(
                self.status.lagging_count, 0,
                "{context} unexpectedly used late-readiness/Frozen-resume slew"
            );
            assert_eq!(
                self.status.max_lag_bits,
                0.0_f32.to_bits(),
                "{context} published nonzero blend lag"
            );
        }

        fn assert_active_dynamic_evaluation_complete(&self, context: &str) {
            if !self.candidate_active {
                return;
            }
            assert!(
                self.candidate_prepared
                    && !self.candidate_transitioning
                    && self.desired_evaluation_complete
                    && self.evaluation_view.is_some()
                    && self.evaluation_target.is_some(),
                "{context} exposed an ACTIVE Dynamic drawable without one complete coherent evaluation tuple"
            );
            assert_eq!(
                (
                    self.status.invalid_pressure_count,
                    self.status.missing_consumer_count,
                ),
                (0, 0),
                "{context} exposed an ACTIVE Dynamic drawable with invalid/missing private evaluation"
            );
            for edge in &self.edges {
                assert_eq!(
                    edge.evaluation_weight_bits,
                    Some(edge.desired_weight_bits),
                    "{context} exposed an ACTIVE Dynamic edge without its exact desired oracle: {:?}",
                    edge.key,
                );
            }
        }

        fn assert_current_render_evaluation(&self, context: &str) {
            assert_eq!(
                self.evaluation_view,
                Some(self.current_render_view),
                "{context} did not evaluate the exact current extracted view"
            );
            assert_eq!(
                self.evaluation_target,
                Some(self.current_render_target),
                "{context} did not evaluate the exact current render-owned quality target"
            );
        }

        fn assert_manifest_edge_topology(&self, parents: &GardenNodeParents, context: &str) {
            for edge in &self.edges {
                let expected_children = parents
                    .iter()
                    .filter_map(|(node, parent)| {
                        (*parent == Some(edge.key.parent)).then_some(*node)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    edge.key.children, expected_children,
                    "{context} edge omitted, duplicated, or reordered an immediate sibling for parent {}",
                    edge.key.parent,
                );
                assert_eq!(
                    edge.key.child_metrics.len(),
                    expected_children.len(),
                    "{context} edge metric table did not cover every immediate child of parent {}",
                    edge.key.parent,
                );
            }
        }

        fn assert_evaluation_no_older_than_previous(&self, previous: &Self, context: &str) {
            let evaluation_view = self
                .evaluation_view
                .unwrap_or_else(|| panic!("{context} has no checked drawable evaluation view"));
            assert!(
                evaluation_view == self.current_render_view
                    || evaluation_view == previous.current_render_view,
                "{context} evaluated a view older than one render frame: evaluation={evaluation_view:?}, previous={:?}, current={:?}",
                previous.current_render_view,
                self.current_render_view,
            );
            let evaluation_target = self
                .evaluation_target
                .unwrap_or_else(|| panic!("{context} has no checked drawable evaluation target"));
            assert!(
                evaluation_target == self.current_render_target
                    || evaluation_target == previous.current_render_target,
                "{context} evaluated a quality target older than one render frame: evaluation={evaluation_target:?}, previous={:?}, current={:?}",
                previous.current_render_target,
                self.current_render_target,
            );
        }

        /// Returns true for the one legal retained-endpoint publication which
        /// atomically installs a new immutable table. Every steady common-edge
        /// frame returns false and must be exactly at the current-view weight.
        fn assert_dynamic_frame_transition(
            &self,
            previous: &Self,
            recovery_edges: &BTreeSet<GardenViewBlendEdgeKey>,
            pending_ordinary_edges: &BTreeMap<GardenViewBlendEdgeKey, u32>,
            parents: &GardenNodeParents,
            exact_token_candidate: bool,
            allow_unevaluated_late_authored: bool,
            context: &str,
        ) -> GardenViewBlendActivationEvidence {
            self.assert_dynamic_coherent(context);
            self.assert_active_dynamic_evaluation_complete(context);
            self.assert_no_invalid_pressure_pairs(context);
            self.assert_manifest_edge_topology(parents, context);
            assert_eq!(
                self.compaction_generation, previous.compaction_generation,
                "{context} destroyed/recreated the drawable compaction state"
            );
            assert!(
                self.publication_generation >= previous.publication_generation,
                "{context} drawable publication generation regressed"
            );
            assert!(
                self.upload.immutable_table_upload_count
                    >= previous.upload.immutable_table_upload_count
                    && self.upload.weight_write_count >= previous.upload.weight_write_count
                    && self.upload.buffer_allocation_count
                        >= previous.upload.buffer_allocation_count,
                "{context} reset a per-compaction resource counter"
            );
            let previous_edges_by_key = previous
                .edges
                .iter()
                .map(|edge| (&edge.key, edge))
                .collect::<BTreeMap<_, _>>();
            for key in recovery_edges {
                let Some(previous_edge) = previous_edges_by_key.get(key) else {
                    continue;
                };
                let previous_unevaluated_late_first_draw = previous_edge.activation_requires_slew
                    && previous.candidate_prepared
                    && !previous.candidate_active
                    && !previous.candidate_transitioning
                    && !previous.desired_evaluation_complete
                    && previous.evaluation_view.is_none()
                    && previous.evaluation_target.is_none()
                    && previous_edge.evaluation_weight_bits.is_none()
                    && previous_edge.displayed_weight_bits == previous_edge.desired_weight_bits
                    && previous_edge.displayed_weight_bits == previous_edge.initial_weight_bits;
                if !previous_unevaluated_late_first_draw {
                    continue;
                }
                let edge = self.edges.iter().find(|edge| &edge.key == key).unwrap_or_else(|| {
                    panic!(
                        "{context} retired a PREPARED late edge before publishing its current-view oracle: {key:?}"
                    )
                });
                let oracle = edge.evaluation_weight_bits.unwrap_or_else(|| {
                    panic!(
                        "{context} repeated a distinct PREPARED late-edge publication without attaching its current-view oracle: {key:?}"
                    )
                });
                assert_eq!(
                    edge.desired_weight_bits, oracle,
                    "{context} PREPARED late edge attached an oracle without retargeting desired: {key:?}"
                );
                self.assert_active_dynamic_evaluation_complete(context);
                if oracle != previous_edge.displayed_weight_bits {
                    assert_ne!(
                        edge.displayed_weight_bits, edge.desired_weight_bits,
                        "{context} PREPARED late edge caught up before exposing truthful recovery lag: {key:?}"
                    );
                    assert!(
                        edge.recovery_lag,
                        "{context} PREPARED late edge exposed lag without its mutable recovery marker: {key:?}"
                    );
                }
            }
            let table_changed = self.upload.immutable_table_upload_count
                > previous.upload.immutable_table_upload_count;
            if !self.desired_evaluation_complete && !table_changed {
                let previous_edges = previous
                    .edges
                    .iter()
                    .map(|edge| (&edge.key, edge))
                    .collect::<BTreeMap<_, _>>();
                assert_eq!(
                    self.edges.len(),
                    previous.edges.len(),
                    "{context} changed immutable edge cardinality without a table upload"
                );
                assert_eq!(
                    (
                        self.target_ranges.as_slice(),
                        self.presentation_ranges.as_slice(),
                        self.required_ranges.as_slice()
                    ),
                    (
                        previous.target_ranges.as_slice(),
                        previous.presentation_ranges.as_slice(),
                        previous.required_ranges.as_slice()
                    ),
                    "{context} changed immutable presentation/range topology without a table upload"
                );
                assert_eq!(
                    self.upload.buffer_allocation_count, previous.upload.buffer_allocation_count,
                    "{context} reallocated the blend buffer while retaining an incomplete authored guard"
                );
                assert_eq!(
                    self.status.invalid_pressure_count, 0,
                    "{context} used an incomplete authored guard to hide invalid pressure pairs"
                );
                assert!(
                    !pending_ordinary_edges.is_empty(),
                    "{context} retained an incomplete table without an ordinary authored guard"
                );
                for (key, authored_bits) in pending_ordinary_edges {
                    let edge = self.edges.iter().find(|edge| &edge.key == key).unwrap_or_else(|| {
                        panic!(
                            "{context} dropped a pending ordinary authored edge without a table upload: {key:?}",
                        )
                    });
                    assert_eq!(
                        (edge.displayed_weight_bits, edge.desired_weight_bits),
                        (*authored_bits, *authored_bits),
                        "{context} moved a pending ordinary authored edge before its table became fully evaluated: {:?}",
                        edge.key,
                    );
                    let previous = previous_edges.get(&edge.key).unwrap_or_else(|| {
                        panic!(
                            "{context} introduced a pending ordinary edge without a table upload: {:?}",
                            edge.key,
                        )
                    });
                    assert_eq!(
                        edge.displayed_weight_bits, previous.displayed_weight_bits,
                        "{context} moved a pending ordinary edge between incomplete suffixes: {:?}",
                        edge.key,
                    );
                }
            }
            if self.evaluation_view.is_some() || self.evaluation_target.is_some() {
                self.assert_evaluation_no_older_than_previous(previous, context);
            } else {
                assert!(
                    !self.desired_evaluation_complete,
                    "{context} omitted evaluation metadata from a complete radix publication"
                );
            }
            if !table_changed {
                let previous_edges = previous
                    .edges
                    .iter()
                    .map(|edge| (&edge.key, edge))
                    .collect::<BTreeMap<_, _>>();
                let mut by_batch_index = self.edges.iter().collect::<Vec<_>>();
                by_batch_index.sort_by_key(|edge| edge.batch_index);
                let mut displayed_changed = false;
                let mut max_delta = 0.0_f32;
                let mut weighted_record_energy = 0.0_f64;
                for edge in by_batch_index {
                    let previous_edge = previous_edges.get(&edge.key).unwrap_or_else(|| {
                        panic!(
                            "{context} changed an edge key without an immutable table upload: {:?}",
                            edge.key,
                        )
                    });
                    let delta = (f32::from_bits(edge.displayed_weight_bits)
                        - f32::from_bits(previous_edge.displayed_weight_bits))
                    .abs();
                    displayed_changed |= delta > 0.0;
                    max_delta = max_delta.max(delta);
                    weighted_record_energy += f64::from(delta) * f64::from(edge.record_count);
                    if delta > 0.0 && !recovery_edges.contains(&edge.key) {
                        assert_eq!(
                            (
                                edge.displayed_weight_bits,
                                edge.desired_weight_bits,
                                edge.evaluation_weight_bits,
                            ),
                            (
                                edge.desired_weight_bits,
                                edge.desired_weight_bits,
                                Some(edge.desired_weight_bits),
                            ),
                            "{context} ordinary all-resident edge did not follow the exact captured-view oracle in the same radix-promoted frame: key={:?}, previous={previous_edge:?}, current={edge:?}, pending_ordinary={}",
                            edge.key,
                            pending_ordinary_edges.contains_key(&edge.key),
                        );
                    }
                }
                if displayed_changed {
                    assert_eq!(
                        self.upload.weight_write_count,
                        previous.upload.weight_write_count.saturating_add(1),
                        "{context} changed same-table suffix did not advance exactly one GPU weight-write generation"
                    );
                    assert_eq!(
                        self.upload.weight_bytes_written,
                        previous.upload.weight_bytes_written.saturating_add(
                            u64::try_from(self.edges.len())
                                .expect("view-blend edge count fits u64")
                                * std::mem::size_of::<f32>() as u64,
                        ),
                        "{context} changed same-table suffix wrote an unexpected number of weight bytes"
                    );
                    assert_eq!(
                        self.upload.last_max_delta.to_bits(),
                        max_delta.to_bits(),
                        "{context} changed same-table suffix reported a torn max displayed-weight delta"
                    );
                    assert_eq!(
                        self.upload.last_weighted_record_energy.to_bits(),
                        weighted_record_energy.to_bits(),
                        "{context} changed same-table suffix reported torn weighted-record energy"
                    );
                } else {
                    assert_eq!(
                        (
                            self.upload.weight_write_count,
                            self.upload.weight_bytes_written,
                        ),
                        (
                            previous.upload.weight_write_count,
                            previous.upload.weight_bytes_written,
                        ),
                        "{context} camera-only target retarget changed GPU weight-write counters"
                    );
                    let current_event = (
                        self.upload.last_max_delta.to_bits(),
                        self.upload.last_weighted_record_energy.to_bits(),
                    );
                    let previous_event = (
                        previous.upload.last_max_delta.to_bits(),
                        previous.upload.last_weighted_record_energy.to_bits(),
                    );
                    if self.publication_generation == previous.publication_generation {
                        assert_eq!(
                            current_event, previous_event,
                            "{context} retargeted one existing drawable publication but changed its frozen delta/energy telemetry"
                        );
                    } else {
                        assert_eq!(
                            current_event,
                            (0.0_f32.to_bits(), 0.0_f64.to_bits()),
                            "{context} camera-only radix publication reported displayed-weight work"
                        );
                    }
                }
                if self.desired_evaluation_complete && self.status.lagging_count == 0 {
                    self.assert_live_dynamic_exact(context);
                }
                return GardenViewBlendActivationEvidence {
                    activation_frame: !self.desired_evaluation_complete,
                    ..default()
                };
            }
            assert_eq!(
                self.status.invalid_pressure_count, 0,
                "{context} all-resident activation encountered an invalid pressure pair"
            );
            self.assert_table_replacement_continuity(
                previous,
                recovery_edges,
                parents,
                exact_token_candidate,
                allow_unevaluated_late_authored,
                context,
            )
        }

        fn assert_all_resident_dynamic_frame(
            &self,
            previous: &Self,
            recovery_edges: &BTreeSet<GardenViewBlendEdgeKey>,
            pending_ordinary_edges: &BTreeMap<GardenViewBlendEdgeKey, u32>,
            parents: &GardenNodeParents,
            exact_token_candidate: bool,
            allow_unevaluated_late_authored: bool,
            context: &str,
        ) -> GardenViewBlendActivationEvidence {
            let evidence = self.assert_dynamic_frame_transition(
                previous,
                recovery_edges,
                pending_ordinary_edges,
                parents,
                exact_token_candidate,
                allow_unevaluated_late_authored,
                context,
            );
            if !evidence.new_authored_publication {
                return evidence;
            }
            let previous_edges = previous
                .edges
                .iter()
                .map(|edge| (&edge.key, edge))
                .collect::<BTreeMap<_, _>>();
            for edge in self
                .edges
                .iter()
                .filter(|edge| !previous_edges.contains_key(&edge.key))
            {
                if edge.activation_requires_slew {
                    assert!(
                        allow_unevaluated_late_authored
                            && evidence.unevaluated_late_edge_keys.contains(&edge.key)
                            && edge.recovery_lag,
                        "{context} used late-readiness slew outside the exact PREPARED unmeasured authored publication"
                    );
                    continue;
                }
                assert_eq!(
                    edge.displayed_weight_bits, edge.initial_weight_bits,
                    "{context} newly admitted edge did not publish its exact authored endpoint for {:?}",
                    edge.key,
                );
                assert_eq!(
                    edge.desired_weight_bits, edge.initial_weight_bits,
                    "{context} unevaluated new edge published a non-authored desired weight for {:?}",
                    edge.key,
                );
            }
            evidence
        }

        fn assert_table_replacement_continuity(
            &self,
            previous: &Self,
            recovery_edges: &BTreeSet<GardenViewBlendEdgeKey>,
            parents: &GardenNodeParents,
            exact_token_candidate: bool,
            allow_unevaluated_late_authored: bool,
            context: &str,
        ) -> GardenViewBlendActivationEvidence {
            assert!(
                self.upload.immutable_table_upload_count
                    > previous.upload.immutable_table_upload_count,
                "{context} checked replacement continuity without a new compaction/table generation"
            );
            assert_eq!(
                self.upload.immutable_table_upload_count,
                previous
                    .upload
                    .immutable_table_upload_count
                    .saturating_add(1),
                "{context} collapsed multiple immutable-table replacements into one drawable publication"
            );
            let previous_edges = previous
                .edges
                .iter()
                .map(|edge| (&edge.key, edge))
                .collect::<BTreeMap<_, _>>();
            let current_edges = self
                .edges
                .iter()
                .map(|edge| (&edge.key, edge))
                .collect::<BTreeMap<_, _>>();
            let mut new_edge_keys = BTreeSet::new();
            let mut unevaluated_late_edge_keys = BTreeSet::new();
            let mut common_recovery_moved = false;
            for edge in &self.edges {
                if let Some(previous_edge) = previous_edges.get(&edge.key) {
                    let recovery_moved = recovery_edges.contains(&edge.key)
                        && edge.displayed_weight_bits != previous_edge.displayed_weight_bits;
                    if recovery_moved {
                        common_recovery_moved = true;
                    } else if !recovery_edges.contains(&edge.key) {
                        assert_eq!(
                            edge.displayed_weight_bits, previous_edge.displayed_weight_bits,
                            "{context} immutable-table replacement moved a common edge without authenticated recovery for {:?}",
                            edge.key,
                        );
                    }
                    if self.desired_evaluation_complete {
                        assert_eq!(
                            Some(edge.desired_weight_bits),
                            edge.evaluation_weight_bits,
                            "{context} immutable-table replacement reset a common edge's desired weight instead of retargeting the checked drawable view for {:?}",
                            edge.key,
                        );
                    } else if let Some(evaluation_weight_bits) = edge.evaluation_weight_bits {
                        assert_eq!(
                            edge.desired_weight_bits, evaluation_weight_bits,
                            "{context} authored replacement left a common edge off its checked drawable-view desired oracle for {:?}",
                            edge.key,
                        );
                    } else {
                        assert_eq!(
                            edge.desired_weight_bits, previous_edge.desired_weight_bits,
                            "{context} authored replacement changed a common edge's desired state before live evaluation for {:?}",
                            edge.key,
                        );
                    }
                    for node in
                        std::iter::once(edge.key.parent).chain(edge.key.children.iter().copied())
                    {
                        assert_eq!(
                            garden_required_node_range(&self.required_ranges, node, context,),
                            garden_required_node_range(&previous.required_ranges, node, context,),
                            "{context} rematerialized common-edge node {node} instead of preserving its exact physical lease"
                        );
                    }
                } else {
                    new_edge_keys.insert(edge.key.clone());
                    assert_eq!(
                        edge.displayed_weight_bits, edge.initial_weight_bits,
                        "{context} newly admitted edge did not publish its exact authored endpoint for {:?}",
                        edge.key,
                    );
                    if edge.activation_requires_slew {
                        assert!(
                            edge.recovery_lag,
                            "{context} late-admitted edge omitted its mutable recovery marker: {:?}",
                            edge.key,
                        );
                        if let Some(oracle) = edge.evaluation_weight_bits {
                            assert_eq!(
                                edge.desired_weight_bits, oracle,
                                "{context} late-admitted edge did not publish the current drawable-view desired weight: {:?}",
                                edge.key,
                            );
                            if oracle != edge.initial_weight_bits {
                                assert_ne!(
                                    edge.displayed_weight_bits, edge.desired_weight_bits,
                                    "{context} late-admitted edge hid its required recovery lag: {:?}",
                                    edge.key,
                                );
                            }
                        } else {
                            assert!(
                                allow_unevaluated_late_authored,
                                "{context} late-admitted edge has no valid drawable-view oracle outside the exact PREPARED unmeasured authored publication: {:?}",
                                edge.key,
                            );
                            assert_eq!(
                                (edge.displayed_weight_bits, edge.desired_weight_bits),
                                (edge.initial_weight_bits, edge.initial_weight_bits),
                                "{context} unevaluated PREPARED late edge did not hold its authored endpoint: {:?}",
                                edge.key,
                            );
                            unevaluated_late_edge_keys.insert(edge.key.clone());
                        }
                    }
                }
                if !self.desired_evaluation_complete {
                    assert_eq!(
                        edge.current_view_weight_bits, None,
                        "{context} unevaluated activation incorrectly claimed a current-view oracle for {:?}",
                        edge.key,
                    );
                }
            }
            if common_recovery_moved {
                assert!(
                    exact_token_candidate,
                    "{context} compressed a replacement recovery step without exact candidate ownership"
                );
                assert_eq!(
                    self.publication_generation,
                    previous.publication_generation.saturating_add(2),
                    "{context} did not expose exactly one skipped inherited-table publication before its common recovery step"
                );
                assert_eq!(
                    self.upload.weight_write_count,
                    previous.upload.weight_write_count.saturating_add(1),
                    "{context} common replacement recovery did not consume exactly one suffix write"
                );
                assert_eq!(
                    self.upload.weight_bytes_written,
                    previous.upload.weight_bytes_written.saturating_add(
                        u64::try_from(self.edges.len()).expect("view-blend edge count fits u64")
                            * std::mem::size_of::<f32>() as u64,
                    ),
                    "{context} common replacement recovery wrote an unexpected suffix size"
                );
                let mut by_batch_index = self.edges.iter().collect::<Vec<_>>();
                by_batch_index.sort_by_key(|edge| edge.batch_index);
                let (max_delta, weighted_record_energy) = by_batch_index.into_iter().fold(
                    (0.0_f32, 0.0_f64),
                    |(max_delta, energy), edge| {
                        let delta = previous_edges.get(&edge.key).map_or(0.0, |previous_edge| {
                            (f32::from_bits(edge.displayed_weight_bits)
                                - f32::from_bits(previous_edge.displayed_weight_bits))
                            .abs()
                        });
                        (
                            max_delta.max(delta),
                            energy + f64::from(delta) * f64::from(edge.record_count),
                        )
                    },
                );
                assert_eq!(
                    self.upload.last_max_delta.to_bits(),
                    max_delta.to_bits(),
                    "{context} common replacement recovery reported a torn max delta"
                );
                assert_eq!(
                    self.upload.last_weighted_record_energy.to_bits(),
                    weighted_record_energy.to_bits(),
                    "{context} common replacement recovery reported torn weighted-record energy"
                );
            }
            if !unevaluated_late_edge_keys.is_empty() {
                assert!(
                    allow_unevaluated_late_authored
                        && self.candidate_prepared
                        && !self.candidate_active
                        && !self.candidate_transitioning
                        && !self.desired_evaluation_complete,
                    "{context} unevaluated late-edge authored publication escaped its exact PREPARED phase gate"
                );
                assert_eq!(
                    (self.evaluation_view, self.evaluation_target),
                    (None, None),
                    "{context} PREPARED late-edge authored publication exposed partial evaluation metadata"
                );
                assert_eq!(
                    (
                        self.status.invalid_pressure_count,
                        self.status.missing_consumer_count,
                    ),
                    (0, 0),
                    "{context} PREPARED late-edge authored publication was not consumer/pressure safe"
                );
                assert_eq!(
                    (
                        self.upload.weight_write_count,
                        self.upload.weight_bytes_written,
                    ),
                    (
                        previous.upload.weight_write_count,
                        previous.upload.weight_bytes_written,
                    ),
                    "{context} PREPARED late-edge authored publication wrote a drawable weight suffix"
                );
                assert_eq!(
                    (
                        self.upload.last_max_delta.to_bits(),
                        self.upload.last_weighted_record_energy.to_bits(),
                    ),
                    (0.0_f32.to_bits(), 0.0_f64.to_bits()),
                    "{context} PREPARED late-edge authored publication reported displayed-weight work"
                );
            }
            let preserved_fractional_overlap = self.edges.iter().any(|common| {
                previous_edges
                    .get(&common.key)
                    .is_some_and(|previous| common.endpoint == 0 || previous.endpoint == 0)
                    && self.edges.iter().any(|new_edge| {
                        !previous_edges.contains_key(&new_edge.key)
                            && !garden_node_is_descendant_or_same(
                                common.key.parent,
                                new_edge.key.parent,
                                parents,
                            )
                            && !garden_node_is_descendant_or_same(
                                new_edge.key.parent,
                                common.key.parent,
                                parents,
                            )
                    })
            });
            if !new_edge_keys.is_empty() {
                assert!(
                    !self.desired_evaluation_complete,
                    "{context} evaluated a newly authored edge before publishing its retained endpoint"
                );
            }
            let target_nodes = self
                .target_ranges
                .iter()
                .map(|range| range.0)
                .collect::<BTreeSet<_>>();
            let mut replacement_initial_nodes = target_nodes.clone();
            let mut occupied_edge_nodes = BTreeSet::new();
            for edge in &self.edges {
                assert!(
                    occupied_edge_nodes.insert(edge.key.parent)
                        && edge
                            .key
                            .children
                            .iter()
                            .all(|child| occupied_edge_nodes.insert(*child)),
                    "{context} replacement table contained nested/overlapping edge endpoints: {:?}",
                    edge.key,
                );
                let target_has_parent = target_nodes.contains(&edge.key.parent);
                let target_has_all_children = edge
                    .key
                    .children
                    .iter()
                    .all(|child| target_nodes.contains(child));
                assert_ne!(
                    target_has_parent, target_has_all_children,
                    "{context} replacement target was neither exact parent-only nor complete-children-only for {:?}",
                    edge.key,
                );
                replacement_initial_nodes.remove(&edge.key.parent);
                for child in &edge.key.children {
                    replacement_initial_nodes.remove(child);
                }
                if previous_edges.contains_key(&edge.key) {
                    // Common edges inherit render-owned state; their immutable
                    // authored endpoint is not retirement evidence.
                    continue;
                }
                match edge.initial_weight_bits {
                    bits if bits == 0.0_f32.to_bits() => {
                        replacement_initial_nodes.insert(edge.key.parent);
                    }
                    bits if bits == 1.0_f32.to_bits() => {
                        replacement_initial_nodes.extend(edge.key.children.iter().copied());
                    }
                    _ => panic!(
                        "{context} replacement edge used a fractional authored endpoint: {:?}",
                        edge.key,
                    ),
                }
            }
            let removed_edges = previous
                .edges
                .iter()
                .filter(|edge| !current_edges.contains_key(&edge.key))
                .collect::<Vec<_>>();
            if new_edge_keys.is_empty() && !self.desired_evaluation_complete {
                assert_eq!(
                    (self.evaluation_view, self.evaluation_target),
                    (None, None),
                    "{context} inherited-only authored publication exposed partial evaluation metadata"
                );
                assert_eq!(
                    (
                        self.status.invalid_pressure_count,
                        self.status.missing_consumer_count,
                    ),
                    (0, 0),
                    "{context} inherited-only authored publication was not consumer/pressure safe"
                );
                assert_eq!(
                    (
                        self.upload.weight_write_count,
                        self.upload.weight_bytes_written,
                    ),
                    (
                        previous.upload.weight_write_count,
                        previous.upload.weight_bytes_written,
                    ),
                    "{context} inherited-only authored publication wrote a drawable weight suffix"
                );
                assert_eq!(
                    (
                        self.upload.last_max_delta.to_bits(),
                        self.upload.last_weighted_record_energy.to_bits(),
                    ),
                    (0.0_f32.to_bits(), 0.0_f64.to_bits()),
                    "{context} inherited-only authored publication reported displayed-weight work"
                );
                if removed_edges.is_empty() {
                    assert!(
                        exact_token_candidate
                            && self.candidate_prepared
                            && !self.candidate_active
                            && !self.candidate_transitioning,
                        "{context} topology-identical inherited-only publication escaped its exact-token PREPARED phase gate"
                    );
                    assert_eq!(
                        self.edges.len(),
                        previous.edges.len(),
                        "{context} topology-identical inherited-only publication changed edge cardinality"
                    );
                    assert_eq!(
                        (
                            self.presentation_ranges.as_slice(),
                            self.required_ranges.as_slice(),
                        ),
                        (
                            previous.presentation_ranges.as_slice(),
                            previous.required_ranges.as_slice(),
                        ),
                        "{context} topology-identical inherited-only publication changed its physical presentation/backing union"
                    );
                    let common_endpoint_nodes = self
                        .edges
                        .iter()
                        .flat_map(|edge| {
                            std::iter::once(edge.key.parent)
                                .chain(edge.key.children.iter().copied())
                        })
                        .collect::<BTreeSet<_>>();
                    let target_residual = |ranges: &GardenPhysicalCutSignature| {
                        let mut residual = ranges
                            .iter()
                            .filter(|range| !common_endpoint_nodes.contains(&range.0))
                            .copied()
                            .collect::<Vec<_>>();
                        residual.sort_unstable();
                        residual
                    };
                    assert_eq!(
                        target_residual(&self.target_ranges),
                        target_residual(&previous.target_ranges),
                        "{context} topology-identical inherited-only publication changed target ranges outside its common edge endpoints"
                    );
                    let mut admission_metadata_changed = false;
                    for edge in &self.edges {
                        let previous_edge = previous_edges.get(&edge.key).unwrap_or_else(|| {
                            panic!(
                                "{context} topology-identical inherited-only publication changed edge keys: {:?}",
                                edge.key,
                            )
                        });
                        assert!(
                            edge.initial_weight_bits == 0.0_f32.to_bits()
                                || edge.initial_weight_bits == 1.0_f32.to_bits(),
                            "{context} inherited-only publication used a fractional authored endpoint: {:?}",
                            edge.key,
                        );
                        assert_eq!(
                            (
                                edge.endpoint,
                                edge.displayed_weight_bits,
                                edge.desired_weight_bits,
                                edge.recovery_lag,
                                edge.record_count,
                                edge.batch_index,
                            ),
                            (
                                previous_edge.endpoint,
                                previous_edge.displayed_weight_bits,
                                previous_edge.desired_weight_bits,
                                previous_edge.recovery_lag,
                                previous_edge.record_count,
                                previous_edge.batch_index,
                            ),
                            "{context} topology-identical inherited-only publication reset common drawable state/layout: {:?}",
                            edge.key,
                        );
                        assert!(
                            previous_edge.initial_weight_bits == 0.0_f32.to_bits()
                                || previous_edge.initial_weight_bits == 1.0_f32.to_bits(),
                            "{context} previous inherited-only publication used a fractional authored endpoint: {:?}",
                            edge.key,
                        );
                        assert!(
                            previous_edge.activation_requires_slew
                                || !edge.activation_requires_slew,
                            "{context} topology-identical inherited-only publication gained late-readiness provenance on a common key: {:?}",
                            edge.key,
                        );
                        admission_metadata_changed |=
                            (edge.initial_weight_bits, edge.activation_requires_slew)
                                != (
                                    previous_edge.initial_weight_bits,
                                    previous_edge.activation_requires_slew,
                                );
                    }
                    assert!(
                        admission_metadata_changed,
                        "{context} uploaded an unexplained topology-identical inherited table without changing authored admission metadata"
                    );
                }
            }
            for edge in removed_edges {
                let endpoint_bits = match edge.endpoint {
                    1 => 0.0_f32.to_bits(),
                    2 => 1.0_f32.to_bits(),
                    _ => panic!(
                        "{context} retired a fractional edge during table replacement: {:?}",
                        edge.key
                    ),
                };
                assert_eq!(
                    (edge.displayed_weight_bits, edge.desired_weight_bits),
                    (endpoint_bits, endpoint_bits),
                    "{context} retired an edge before its drawable endpoint: {:?}",
                    edge.key,
                );
                match edge.endpoint {
                    1 => assert!(
                        replacement_initial_nodes.contains(&edge.key.parent),
                        "{context} parent-exact retired edge was not preserved by the replacement's first drawable endpoint: {:?}",
                        edge.key,
                    ),
                    2 => {
                        for child in &edge.key.children {
                            assert!(
                                replacement_initial_nodes.iter().any(|node| {
                                    garden_node_is_descendant_or_same(*node, *child, parents)
                                }),
                                "{context} children-exact retired edge lost child/descendant coverage in the replacement's first drawable endpoint: edge={:?}, child={child}, replacement={replacement_initial_nodes:?}",
                                edge.key,
                            );
                        }
                    }
                    _ => unreachable!(),
                }
            }
            GardenViewBlendActivationEvidence {
                activation_frame: true,
                new_authored_publication: true,
                preserved_fractional_overlap,
                new_edge_keys,
                unevaluated_late_edge_keys,
            }
        }

        fn assert_stationary_fixed_point(&self, context: &str) {
            self.assert_live_dynamic_exact(context);
            self.assert_current_render_evaluation(context);
            assert!(
                !self.edges.is_empty(),
                "{context} did not expose any ABI-16 camera-conditioned blend edges"
            );
            assert_eq!(
                self.status.edge_count as usize,
                self.edges.len(),
                "{context} published an incoherent blend edge count"
            );
            assert_eq!(
                self.status.lagging_count, 0,
                "{context} unexpectedly used late-readiness/Frozen-resume slew"
            );
            assert_eq!(
                self.status.max_lag_bits,
                0.0_f32.to_bits(),
                "{context} published nonzero blend lag"
            );
            for edge in &self.edges {
                assert_eq!(
                    edge.displayed_weight_bits, edge.desired_weight_bits,
                    "{context} displayed weight lagged on an ordinary all-resident Dynamic frame for {:?}",
                    edge.key
                );
            }
        }

        fn assert_frozen_fixed_point(&self, context: &str) {
            self.assert_dynamic_coherent(context);
            assert_eq!(
                self.status.invalid_pressure_count, 0,
                "{context} Frozen publication reported invalid pressure edges"
            );
            self.assert_no_invalid_pressure_pairs(context);
            assert_eq!(
                self.status.lagging_count, 0,
                "{context} Frozen publication reported recovery lag"
            );
            assert!(
                self.edges
                    .iter()
                    .all(|edge| { edge.displayed_weight_bits == edge.desired_weight_bits }),
                "{context} Frozen publication did not hold one exact displayed/desired table"
            );
        }

        fn assert_dynamic_coherent(&self, context: &str) {
            assert_ne!(
                self.compaction_generation, 0,
                "{context} published a zero compaction generation"
            );
            assert_ne!(
                self.compute_input_generation, 0,
                "{context} published a zero compute-input generation"
            );
            assert_ne!(
                self.publication_generation, 0,
                "{context} published a zero drawable-publication generation"
            );
            assert!(
                self.candidate_fingerprint.0.is_some()
                    && self.candidate_fingerprint.1.is_some()
                    && self.candidate_fingerprint.2.is_some(),
                "{context} radix drawable omitted its candidate fingerprint"
            );
            assert!(
                self.candidate_content_signature.is_some()
                    && self.candidate_atlas_allocation_epoch.is_some(),
                "{context} radix drawable omitted its atlas content/allocation provenance"
            );
            assert_eq!(
                self.status.edge_count as usize,
                self.edges.len(),
                "{context} published an incoherent blend edge count"
            );
            assert_eq!(
                self.upload.lagging_edge_count, self.status.lagging_count,
                "{context} main/render lag counts diverged"
            );
            assert!(
                self.status.invalid_pressure_count <= self.status.edge_count,
                "{context} reported more invalid pressure edges than immutable edges"
            );
            assert_eq!(
                self.status.missing_consumer_count, 0,
                "{context} accepted a drawable without every retained private consumer"
            );
            assert!(
                self.target_ranges
                    .iter()
                    .all(|range| self.required_ranges.contains(range)),
                "{context} target ranges escaped the generation-safe required union"
            );
            assert!(
                self.presentation_ranges
                    .iter()
                    .all(|range| self.required_ranges.contains(range)),
                "{context} presentation ranges escaped the generation-safe required union"
            );
            let required_nodes = self
                .required_ranges
                .iter()
                .map(|range| range.0)
                .collect::<BTreeSet<_>>();
            for edge in &self.edges {
                assert!(
                    required_nodes.contains(&edge.key.parent),
                    "{context} required union omitted edge parent {}",
                    edge.key.parent,
                );
                for child in &edge.key.children {
                    assert!(
                        required_nodes.contains(child),
                        "{context} required union omitted immediate child {child} of parent {}",
                        edge.key.parent,
                    );
                }
            }
            let mut lagging = 0_u32;
            let mut max_lag = 0.0_f32;
            for edge in &self.edges {
                let displayed = f32::from_bits(edge.displayed_weight_bits);
                let desired = f32::from_bits(edge.desired_weight_bits);
                assert!(
                    displayed.is_finite()
                        && desired.is_finite()
                        && (0.0..=1.0).contains(&displayed)
                        && (0.0..=1.0).contains(&desired),
                    "{context} published an invalid edge weight for {:?}: displayed={displayed}, desired={desired}",
                    edge.key,
                );
                match edge.endpoint {
                    0 => assert!(
                        edge.displayed_weight_bits != 0.0_f32.to_bits()
                            && edge.displayed_weight_bits != 1.0_f32.to_bits(),
                        "{context} classified an exact displayed weight as fractional for {:?}",
                        edge.key,
                    ),
                    1 => assert_eq!(
                        edge.displayed_weight_bits,
                        0.0_f32.to_bits(),
                        "{context} ParentExact endpoint disagreed with displayed weight for {:?}",
                        edge.key,
                    ),
                    2 => assert_eq!(
                        edge.displayed_weight_bits,
                        1.0_f32.to_bits(),
                        "{context} ChildrenExact endpoint disagreed with displayed weight for {:?}",
                        edge.key,
                    ),
                    endpoint => panic!(
                        "{context} published unknown endpoint tag {endpoint} for {:?}",
                        edge.key,
                    ),
                }
                if self.desired_evaluation_complete {
                    assert_eq!(
                        self.status.invalid_pressure_count, 0,
                        "{context} called an invalid-pressure publication fully evaluated"
                    );
                    assert_eq!(
                        Some(edge.desired_weight_bits),
                        edge.current_view_weight_bits,
                        "{context} desired weight was not the stateless drawable-view oracle for {:?}",
                        edge.key
                    );
                } else {
                    assert_eq!(
                        edge.current_view_weight_bits, None,
                        "{context} incomplete drawable publication exposed a partial view oracle for {:?}",
                        edge.key
                    );
                }
                if edge.displayed_weight_bits != edge.desired_weight_bits {
                    lagging += 1;
                }
                max_lag = max_lag.max((displayed - desired).abs());
            }
            assert_eq!(
                lagging, self.status.lagging_count,
                "{context} coherent weight table disagreed with its lag count"
            );
            assert_eq!(
                max_lag.to_bits(),
                self.status.max_lag_bits,
                "{context} coherent weight table disagreed with its maximum lag"
            );
        }

        fn assert_no_invalid_pressure_pairs(&self, context: &str) {
            assert_eq!(
                self.status.invalid_pressure_count, 0,
                "{context} reported invalid active view-blend pressure edges"
            );
            for edge in &self.edges {
                let (parent_bits, child_bits) = edge.pressure_bits.unwrap_or_else(|| {
                    panic!(
                        "{context} active edge has a malformed or threshold-contradictory pressure pair: {:?}",
                        edge.key
                    )
                });
                let parent = f32::from_bits(parent_bits);
                let child = f32::from_bits(child_bits);
                assert!(
                    parent.is_finite() && child.is_finite() && !(parent <= 1.0 && child > 1.0),
                    "{context} active edge has non-finite or threshold-contradictory pressure: key={:?}, parent={parent}, child={child}",
                    edge.key,
                );
            }
        }
    }

    #[derive(Clone, Debug)]
    struct GardenDrawableIndirectObservation {
        compaction_generation: u64,
        radix_publication_generation: u64,
        args: LodIndirectArgs,
        expected: LodIndirectArgs,
    }

    #[derive(Clone, Debug)]
    struct GardenViewBlendRenderSnapshot {
        candidate: GardenExtractedCandidateProof,
        drawable: LodLastRadixDrawableForTesting,
        current_render_view: LodView,
        current_render_target: LodQualityTarget,
        indirect: Option<GardenDrawableIndirectObservation>,
    }

    #[derive(Clone, Debug)]
    struct GardenExtractedCandidateProof {
        rendered_candidate_count: u32,
        render_commit_identity: usize,
        failed: bool,
        prepared: bool,
        transitioning: bool,
        active: bool,
        selection_mode: LodSelectionMode,
        selection_view_frozen: bool,
        temporal_mode: Option<LodTemporalTransitionMode>,
        view_blend_replan_requested: bool,
        view_blend: Option<LodViewBlendTestingSnapshot>,
        retention: (bool, bool, bool),
        morph_identity: Option<LodViewBlendIdentity>,
        target_ranges: GardenPhysicalCutSignature,
        presentation_ranges: GardenPhysicalCutSignature,
        required_ranges: GardenPhysicalCutSignature,
    }

    fn garden_extracted_candidate_proof(
        candidates: &LodRenderCandidates,
        candidate: &LodRenderCandidate,
        selection_mode: LodSelectionMode,
    ) -> GardenExtractedCandidateProof {
        GardenExtractedCandidateProof {
            rendered_candidate_count: candidate.rendered_candidate_count(),
            render_commit_identity: candidate.render_commit_identity_for_testing(),
            failed: candidate.failed(),
            prepared: candidate.render_is_prepared(),
            transitioning: candidate.render_is_transitioning_for_testing(),
            active: candidate.render_is_active_for_testing(),
            selection_mode,
            selection_view_frozen: candidate.frontier().selection_view_frozen(),
            temporal_mode: candidate.temporal_transition_mode(),
            view_blend_replan_requested: candidate.view_blend_replan_requested_for_testing(),
            view_blend: candidate.view_blend_testing_snapshot(),
            retention: candidates.package_retention_for_testing(),
            morph_identity: candidate
                .view_blend()
                .and_then(|blend| blend.morph())
                .map(|morph| morph.identity()),
            target_ranges: garden_physical_range_signature(candidate.frontier().physical_ranges()),
            presentation_ranges: garden_physical_range_signature(candidate.render_ranges()),
            required_ranges: garden_physical_range_signature(
                candidate.required_atlas_ranges_for_testing(),
            ),
        }
    }

    const GARDEN_MAX_PREPARED_RETAINED_DRAWABLE_FRAMES: u32 = 16;

    #[derive(Clone, Debug)]
    enum GardenPromotedDrawableClass {
        CurrentCandidate,
        RetainedCurrent(GardenExtractedCandidateProof),
    }

    #[derive(Clone, Debug, Default)]
    struct GardenPromotedDrawableTracker {
        last_accepted: Option<LodLastRadixDrawableForTesting>,
        last_accepted_candidate: Option<GardenExtractedCandidateProof>,
        armed_handoff_token: Option<usize>,
        prepared_retained_frames: u32,
        max_prepared_retained_frames: u32,
    }

    impl GardenPromotedDrawableTracker {
        fn classify(
            &mut self,
            render: &GardenViewBlendRenderSnapshot,
            context: &str,
        ) -> GardenPromotedDrawableClass {
            assert_garden_promoted_indirect(render, context);
            let token = render.candidate.render_commit_identity;
            if render.candidate.view_blend_replan_requested {
                let (retained_current, candidates_are_current, _) = render.candidate.retention;
                assert!(
                    retained_current
                        && !candidates_are_current
                        && !render.candidate.active
                        && !render.candidate.transitioning,
                    "{context} requested a predecessor replan outside a retained PREPARED handoff"
                );
            }
            if render.drawable.candidate_token_matches && render.drawable.candidate_content_matches
            {
                assert!(
                    !render.candidate.view_blend_replan_requested,
                    "{context} promoted an exact token after its predecessor proof requested replan"
                );
                assert_eq!(
                    render.drawable.rendered_candidate_count,
                    render.candidate.rendered_candidate_count,
                    "{context} exact-token drawable count disagreed with its extracted candidate"
                );
                if render.candidate.temporal_mode == Some(LodTemporalTransitionMode::Morphing) {
                    let aggregate = render.candidate.view_blend.as_ref().unwrap_or_else(|| {
                        panic!("{context} exact-token Morphing candidate omitted its coherent aggregate")
                    });
                    assert_eq!(
                        aggregate.status.missing_consumer_count, 0,
                        "{context} exact-token promoted candidate still reported a missing private consumer"
                    );
                }
                self.last_accepted = Some(render.drawable.clone());
                self.last_accepted_candidate = Some(render.candidate.clone());
                self.armed_handoff_token = None;
                self.prepared_retained_frames = 0;
                return GardenPromotedDrawableClass::CurrentCandidate;
            }

            let (retained_current, candidates_are_current, retained_current_is_stale) =
                render.candidate.retention;
            assert!(
                retained_current && !candidates_are_current,
                "{context} exposed a token/content-mismatched radix drawable outside an authenticated retained-current handoff: retained_current={retained_current}, candidates_are_current={candidates_are_current}, retained_current_is_stale={retained_current_is_stale}, drawable={:?}",
                render.drawable,
            );
            assert!(
                !render.candidate.active && !render.candidate.transitioning,
                "{context} exposed an ACTIVE/TRANSITIONING extracted candidate with a different radix drawable"
            );
            if render.candidate.temporal_mode == Some(LodTemporalTransitionMode::Morphing) {
                let pending = render.candidate.view_blend.as_ref().unwrap_or_else(|| {
                    panic!(
                        "{context} retained Morphing replacement omitted missing-consumer evidence"
                    )
                });
                assert_ne!(
                    pending.status.missing_consumer_count, 0,
                    "{context} retained replacement claimed a complete private-consumer aggregate before its exact token was radix-promoted"
                );
            }
            let previous = self.last_accepted.as_ref().unwrap_or_else(|| {
                panic!(
                    "{context} exposed a retained-old radix drawable before any exact drawable was accepted"
                )
            });
            assert_eq!(
                (
                    render.drawable.compaction_generation,
                    render.drawable.rendered_candidate_count,
                    render.drawable.candidate_fingerprint_primary,
                    render.drawable.candidate_fingerprint_secondary,
                    render.drawable.candidate_range_count,
                    render.drawable.candidate_content_signature,
                    render.drawable.candidate_atlas_allocation_epoch,
                    render.drawable.morph_identity,
                ),
                (
                    previous.compaction_generation,
                    previous.rendered_candidate_count,
                    previous.candidate_fingerprint_primary,
                    previous.candidate_fingerprint_secondary,
                    previous.candidate_range_count,
                    previous.candidate_content_signature,
                    previous.candidate_atlas_allocation_epoch,
                    previous.morph_identity,
                ),
                "{context} retained-current handoff changed immutable drawable identity"
            );
            assert!(
                render.drawable.compute_input_generation >= previous.compute_input_generation
                    && render.drawable.radix_publication_generation
                        >= previous.radix_publication_generation,
                "{context} retained-current drawable generation regressed"
            );
            assert_eq!(
                render
                    .drawable
                    .view_blend
                    .as_ref()
                    .map(|blend| (&blend.identity, &blend.edges)),
                previous
                    .view_blend
                    .as_ref()
                    .map(|blend| (&blend.identity, &blend.edges)),
                "{context} retained-current handoff changed its immutable blend table"
            );

            if self.armed_handoff_token != Some(token) {
                self.armed_handoff_token = render.candidate.prepared.then_some(token);
                self.prepared_retained_frames = 0;
            }
            if self.armed_handoff_token == Some(token) {
                self.prepared_retained_frames = self.prepared_retained_frames.saturating_add(1);
                self.max_prepared_retained_frames = self
                    .max_prepared_retained_frames
                    .max(self.prepared_retained_frames);
                assert!(
                    self.prepared_retained_frames <= GARDEN_MAX_PREPARED_RETAINED_DRAWABLE_FRAMES,
                    "{context} kept a PREPARED replacement behind the retained drawable for more than {GARDEN_MAX_PREPARED_RETAINED_DRAWABLE_FRAMES} frames"
                );
            }
            GardenPromotedDrawableClass::RetainedCurrent(
                self.last_accepted_candidate
                    .as_ref()
                    .expect("a retained drawable has a frozen accepted candidate proof")
                    .clone(),
            )
        }
    }

    fn assert_garden_promoted_indirect(render: &GardenViewBlendRenderSnapshot, context: &str) {
        let indirect = render.indirect.as_ref().unwrap_or_else(|| {
            panic!("{context} omitted exact radix-generation indirect evidence")
        });
        assert_eq!(
            (
                indirect.compaction_generation,
                indirect.radix_publication_generation,
            ),
            (
                render.drawable.compaction_generation,
                render.drawable.radix_publication_generation,
            ),
            "{context} indirect args were not paired with the promoted drawable generation"
        );
        assert_eq!(
            indirect.args, indirect.expected,
            "{context} indirect args disagreed with the promoted drawable count"
        );
        assert_ne!(
            indirect.args.instance_count, 0,
            "{context} promoted drawable produced an empty indirect draw"
        );
    }

    #[derive(Resource, Clone, ExtractResource)]
    struct GardenViewBlendRenderProbe {
        latest: Arc<Mutex<Option<GardenViewBlendRenderSnapshot>>>,
        ordered: Option<Arc<Mutex<GardenOrderedViewBlendSnapshots>>>,
    }

    #[derive(Default)]
    struct GardenOrderedViewBlendSnapshots {
        enabled: bool,
        next_sequence: u64,
        snapshots: VecDeque<(u64, Option<GardenViewBlendRenderSnapshot>)>,
    }

    impl Default for GardenViewBlendRenderProbe {
        fn default() -> Self {
            Self {
                latest: Arc::new(Mutex::new(None)),
                ordered: None,
            }
        }
    }

    impl GardenViewBlendRenderProbe {
        fn ordered() -> Self {
            Self {
                latest: Arc::new(Mutex::new(None)),
                ordered: Some(Arc::new(Mutex::new(
                    GardenOrderedViewBlendSnapshots::default(),
                ))),
            }
        }

        fn latest_snapshot(&self) -> Option<GardenViewBlendRenderSnapshot> {
            self.latest
                .lock()
                .expect("Garden view-blend probe mutex is not poisoned")
                .as_ref()
                .cloned()
        }

        fn next_ordered_snapshot(&self) -> Option<(u64, Option<GardenViewBlendRenderSnapshot>)> {
            self.ordered
                .as_ref()
                .expect("ordered Garden probe was not enabled")
                .lock()
                .expect("Garden ordered-probe mutex is not poisoned")
                .snapshots
                .pop_front()
        }

        fn begin_ordered_capture(&self) {
            let mut ordered = self
                .ordered
                .as_ref()
                .expect("ordered Garden probe was not enabled")
                .lock()
                .expect("Garden ordered-probe mutex is not poisoned");
            assert!(!ordered.enabled && ordered.snapshots.is_empty());
            ordered.enabled = true;
        }

        fn finish_ordered_capture(&self) {
            let mut ordered = self
                .ordered
                .as_ref()
                .expect("ordered Garden probe was not enabled")
                .lock()
                .expect("Garden ordered-probe mutex is not poisoned");
            assert!(ordered.enabled, "ordered Garden probe stopped twice");
            ordered.enabled = false;
        }

        fn assert_ordered_capture_drained(&self) {
            let ordered = self
                .ordered
                .as_ref()
                .expect("ordered Garden probe was not enabled")
                .lock()
                .expect("Garden ordered-probe mutex is not poisoned");
            assert!(
                !ordered.enabled && ordered.snapshots.is_empty(),
                "ordered Garden probe retained package Cleanup evidence after retirement"
            );
        }
    }

    fn capture_garden_view_blend_render_state(
        render_device: Res<RenderDevice>,
        render_queue: Res<RenderQueue>,
        buffers: Res<LodCompactionBuffers<Gaussian3d>>,
        views: Query<&ExtractedView, With<GaussianCamera>>,
        clouds: Query<(
            Entity,
            &PlanarGaussian3dHandle,
            &GlobalTransform,
            &GaussianLodSettings,
            &CloudSettings,
            &LodRenderCandidates,
        )>,
        probe: Res<GardenViewBlendRenderProbe>,
    ) {
        let mut latest = None;
        for view in &views {
            let camera = view.retained_view_entity.main_entity.id();
            for (cloud, handle, world_from_local, settings, cloud_settings, candidates) in &clouds {
                let Some(candidate) = candidates.get(camera) else {
                    continue;
                };
                let Some(state) =
                    buffers.get_ready(view.retained_view_entity, cloud, handle.handle().id())
                else {
                    continue;
                };
                let Some(drawable) = state.last_radix_drawable_for_testing(candidate) else {
                    continue;
                };
                let candidate = garden_extracted_candidate_proof(
                    candidates,
                    candidate,
                    settings.selection_mode,
                );
                let current_render_view = lod_view_blend_view_for_testing(view, world_from_local)
                    .expect("Garden drawable view is constructible");
                let args = read_lod_indirect_args_for_testing(&render_device, &render_queue, state)
                    .unwrap_or_else(|error| {
                        panic!("Garden drawable indirect-args readback failed: {error}")
                    });
                let defines =
                    ShaderDefines::for_radix_depth_bits(cloud_settings.radix_sort_depth_bits);
                let expected = finalized_indirect_args(
                    drawable.rendered_candidate_count,
                    state.output_capacity(),
                    defines.radix_base * defines.entries_per_invocation_a,
                    defines.workgroup_entries_c,
                );
                let compaction_generation = drawable.compaction_generation;
                let radix_publication_generation = drawable.radix_publication_generation;
                assert!(
                    latest.is_none(),
                    "Garden view-blend qualification expected one retained view/cloud consumer"
                );
                latest = Some(GardenViewBlendRenderSnapshot {
                    candidate,
                    drawable,
                    current_render_view,
                    current_render_target: settings.quality_target(),
                    indirect: Some(GardenDrawableIndirectObservation {
                        compaction_generation,
                        radix_publication_generation,
                        args,
                        expected,
                    }),
                });
            }
        }
        if let Some(ordered) = &probe.ordered {
            let mut ordered = ordered
                .lock()
                .expect("Garden ordered-probe mutex is not poisoned");
            if ordered.enabled {
                let sequence = ordered.next_sequence;
                ordered.next_sequence = ordered.next_sequence.saturating_add(1);
                ordered.snapshots.push_back((sequence, latest.clone()));
                assert!(
                    ordered.snapshots.len() <= 8,
                    "Garden ordered drawable probe exceeded its pipelined backlog bound"
                );
            }
        }
        *probe
            .latest
            .lock()
            .expect("Garden view-blend probe mutex is not poisoned") = latest;
    }

    fn garden_view_blend_endpoint_tag(endpoint: LodViewBlendEndpoint) -> u8 {
        match endpoint {
            LodViewBlendEndpoint::Fractional => 0,
            LodViewBlendEndpoint::ParentExact => 1,
            LodViewBlendEndpoint::ChildrenExact => 2,
        }
    }

    fn garden_physical_range_signature(
        ranges: &[bevy_gaussian_splatting::stream::runtime::LodPhysicalRange],
    ) -> GardenPhysicalCutSignature {
        ranges
            .iter()
            .map(|range| {
                (
                    range.node.0,
                    range.page.0,
                    range.slot.index,
                    range.slot.generation,
                    range.physical_start,
                    range.count,
                )
            })
            .collect()
    }

    fn garden_render_view_blend_state(app: &App, context: &str) -> GardenViewBlendRenderSnapshot {
        let world = app.sub_app(RenderApp).world();
        let mut views = world.iter_entities().filter_map(|entity| {
            entity.get::<GaussianCamera>()?;
            entity.get::<ExtractedView>()
        });
        let view = views
            .next()
            .unwrap_or_else(|| panic!("{context} has no extracted Gaussian camera"));
        assert!(
            views.next().is_none(),
            "{context} expected exactly one extracted Gaussian camera"
        );

        let mut clouds = world.iter_entities().filter_map(|entity| {
            let handle = entity.get::<PlanarGaussian3dHandle>()?;
            let world_from_local = entity.get::<GlobalTransform>()?;
            let settings = entity.get::<GaussianLodSettings>()?;
            let cloud_settings = entity.get::<CloudSettings>()?;
            let candidates = entity.get::<LodRenderCandidates>()?;
            Some((
                entity.id(),
                handle.handle().id(),
                world_from_local,
                settings,
                cloud_settings,
                candidates,
            ))
        });
        let (cloud, handle, world_from_local, settings, cloud_settings, candidates) = clouds
            .next()
            .unwrap_or_else(|| panic!("{context} has no extracted planar Garden cloud"));
        assert!(
            clouds.next().is_none(),
            "{context} expected exactly one extracted planar Garden cloud"
        );
        let buffers = world.resource::<LodCompactionBuffers<Gaussian3d>>();
        let state = buffers
            .get_ready(view.retained_view_entity, cloud, handle)
            .unwrap_or_else(|| panic!("{context} has no drawable LoD compaction state"));
        let camera = view.retained_view_entity.main_entity.id();
        let candidate = candidates
            .get(camera)
            .unwrap_or_else(|| panic!("{context} has no extracted Garden candidate"));
        let drawable = state
            .last_radix_drawable_for_testing(candidate)
            .unwrap_or_else(|| panic!("{context} has no last-radix drawable publication"));
        let candidate =
            garden_extracted_candidate_proof(candidates, candidate, settings.selection_mode);
        let args = read_lod_indirect_args_for_testing(
            world.resource::<RenderDevice>(),
            world.resource::<RenderQueue>(),
            state,
        )
        .unwrap_or_else(|error| panic!("{context} indirect-args readback failed: {error}"));
        let defines = ShaderDefines::for_radix_depth_bits(cloud_settings.radix_sort_depth_bits);
        let expected = finalized_indirect_args(
            drawable.rendered_candidate_count,
            state.output_capacity(),
            defines.radix_base * defines.entries_per_invocation_a,
            defines.workgroup_entries_c,
        );
        let current_render_view = lod_view_blend_view_for_testing(view, world_from_local)
            .unwrap_or_else(|| panic!("{context} drawable view is constructible"));
        GardenViewBlendRenderSnapshot {
            candidate,
            indirect: Some(GardenDrawableIndirectObservation {
                compaction_generation: drawable.compaction_generation,
                radix_publication_generation: drawable.radix_publication_generation,
                args,
                expected,
            }),
            drawable,
            current_render_view,
            current_render_target: settings.quality_target(),
        }
    }

    fn observe_garden_view_blend(
        app: &App,
        _cloud: Entity,
        _camera: Entity,
        context: &str,
    ) -> GardenViewBlendObservation {
        let render = garden_render_view_blend_state(app, context);
        assert!(
            render.drawable.candidate_token_matches,
            "{context} expected a fully ready exact-token drawable"
        );
        let candidate = render.candidate.clone();
        observe_garden_view_blend_with_render_state(&candidate, render, true, context)
    }

    fn assert_garden_public_view_blend_status(
        app: &App,
        cloud: Entity,
        blend: &GardenViewBlendObservation,
        context: &str,
    ) {
        // Main-world Last publishes one update before Render Cleanup captures
        // the next radix-proven drawable suffix. Moving frames therefore only
        // compare fail-closed safety invariants here; static/quiescent gates
        // perform the exact edge/lag/delta/energy equality after both plateau.
        let status = app
            .world()
            .get::<GaussianLodStatus>(cloud)
            .unwrap_or_else(|| panic!("{context} has no public LoD status"));
        assert_eq!(
            status.view_blend_invalid_pressure_evaluations, 0,
            "{context} canonical Garden published invalid active pressure edges"
        );
        if status.view_blend_missing_consumers != 0 {
            assert_ne!(
                status.target_satisfied,
                Some(true),
                "{context} one-update-behind public status claimed satisfaction with a missing private consumer"
            );
        }
        assert_eq!(
            blend.status.invalid_pressure_count, 0,
            "{context} drawable Garden publication contained invalid pressure edges"
        );
    }

    fn observe_garden_view_blend_with_render_state(
        candidate: &GardenExtractedCandidateProof,
        render: GardenViewBlendRenderSnapshot,
        exact_token_aggregate: bool,
        context: &str,
    ) -> GardenViewBlendObservation {
        let GardenViewBlendRenderSnapshot {
            candidate: _,
            drawable,
            current_render_view,
            current_render_target,
            indirect,
        } = render;
        assert!(
            candidate.prepared,
            "{context} observed a candidate without a prepared render capability"
        );
        assert_eq!(
            candidate.temporal_mode,
            Some(LodTemporalTransitionMode::Morphing),
            "{context} did not activate camera-conditioned ABI-16 blending"
        );
        assert!(
            !candidate.view_blend_replan_requested,
            "{context} treated a predecessor-replan token as drawable"
        );
        assert_eq!(
            drawable.rendered_candidate_count, candidate.rendered_candidate_count,
            "{context} drawable count did not match its token/content-matched candidate"
        );
        let indirect = indirect.unwrap_or_else(|| {
            panic!("{context} omitted exact radix-generation indirect evidence")
        });
        assert_eq!(
            (
                indirect.compaction_generation,
                indirect.radix_publication_generation,
            ),
            (
                drawable.compaction_generation,
                drawable.radix_publication_generation,
            ),
            "{context} indirect args were not paired with the observed radix drawable generation"
        );
        assert_eq!(
            indirect.args, indirect.expected,
            "{context} indirect args did not match the exact radix-promoted candidate count/generation"
        );
        assert_ne!(
            indirect.args.instance_count, 0,
            "{context} exact radix-promoted candidate produced an empty indirect draw"
        );
        let view_blend = drawable.view_blend.as_ref().unwrap_or_else(|| {
            panic!("{context} radix-promoted Morphing output has no blend table")
        });
        assert_eq!(
            drawable.morph_identity,
            Some(view_blend.identity),
            "{context} radix drawable and its blend snapshot have different identities"
        );
        assert_eq!(
            candidate.morph_identity,
            Some(view_blend.identity),
            "{context} frozen extracted candidate proof and radix drawable have different blend identities"
        );
        let edge_count = view_blend.edges.len();
        assert_eq!(
            (
                view_blend.weights.len(),
                view_blend.endpoints.len(),
                view_blend.recovery_lag.len(),
                view_blend.invalid_pressure.len(),
                view_blend.upload.edge_count as usize,
            ),
            (edge_count, edge_count, edge_count, edge_count, edge_count),
            "{context} radix-promoted blend snapshot is torn"
        );
        let evaluation = match (
            view_blend.desired_evaluation_complete,
            view_blend.evaluation_view,
            view_blend.evaluation_target,
        ) {
            (true, Some(view), Some(target)) => Some((view, target)),
            (false, None, None) => None,
            (false, Some(view), Some(target)) => Some((view, target)),
            state => {
                panic!("{context} published an incoherent drawable evaluation tuple: {state:?}")
            }
        };
        let target_ranges = candidate.target_ranges.clone();
        let presentation_ranges = candidate.presentation_ranges.clone();
        let required_ranges = candidate.required_ranges.clone();
        let record_counts = view_blend
            .edges
            .iter()
            .map(|edge| {
                edge.children().iter().try_fold(0_u32, |count, child| {
                    count.checked_add(
                        garden_required_node_range(&required_ranges, child.0, context).5,
                    )
                })
            })
            .collect::<Option<Vec<_>>>()
            .unwrap_or_else(|| panic!("{context} per-edge mapped record count overflowed"));
        let mut edges = view_blend
            .edges
            .iter()
            .zip(&view_blend.weights)
            .zip(&view_blend.endpoints)
            .zip(&view_blend.recovery_lag)
            .zip(&view_blend.invalid_pressure)
            .zip(record_counts)
            .enumerate()
            .map(
                |(
                    batch_index,
                    (((((edge, weight), endpoint), recovery_lag), invalid_pressure), record_count),
                )| {
                    let evaluation_pressure = evaluation.and_then(|(view, target)| {
                        lod_view_blend_pressures_for_testing(view, target, edge)
                    });
                    let current_render_pressure = lod_view_blend_pressures_for_testing(
                        current_render_view,
                        current_render_target,
                        edge,
                    );
                    assert_eq!(
                        *invalid_pressure,
                        current_render_pressure.is_none(),
                        "{context} radix-latched invalid-pressure mask disagreed with the exact current render view for parent {}",
                        edge.parent().0,
                    );
                    GardenViewBlendEdgeObservation {
                        key: garden_view_blend_edge_key(edge),
                        batch_index: u32::try_from(batch_index)
                            .expect("view-blend edge index fits u32"),
                        record_count,
                        initial_weight_bits: edge.initial_weight_bits(),
                        activation_requires_slew: edge.activation_requires_slew(),
                        recovery_lag: *recovery_lag,
                        endpoint: garden_view_blend_endpoint_tag(*endpoint),
                        displayed_weight_bits: weight.displayed.to_bits(),
                        desired_weight_bits: weight.desired.to_bits(),
                        pressure_bits: current_render_pressure
                            .map(|(parent, child)| (parent.to_bits(), child.to_bits())),
                        evaluation_weight_bits: evaluation_pressure.map(|_| {
                            let (view, target) = evaluation
                                .expect("a pressure pair requires a drawable evaluation tuple");
                            lod_view_blend_weight_for_testing(view, target, edge).to_bits()
                        }),
                        current_view_weight_bits: view_blend
                            .desired_evaluation_complete
                            .then_some(evaluation_pressure)
                            .flatten()
                            .map(|_| {
                                let (view, target) = evaluation.expect(
                                    "a complete evaluation requires a drawable view/target",
                                );
                                lod_view_blend_weight_for_testing(view, target, edge).to_bits()
                            }),
                        current_render_weight_bits: lod_view_blend_weight_for_testing(
                            current_render_view,
                            current_render_target,
                            edge,
                        )
                        .to_bits(),
                    }
                },
            )
            .collect::<Vec<_>>();
        edges.sort_by(|left, right| left.key.cmp(&right.key));
        assert!(
            edges.windows(2).all(|pair| pair[0].key != pair[1].key),
            "{context} published duplicate immutable edge keys"
        );
        assert_eq!(
            view_blend.upload.edge_count as usize,
            edges.len(),
            "{context} radix upload state disagreed with its exact edge table"
        );
        let lagging_count = edges
            .iter()
            .filter(|edge| edge.displayed_weight_bits != edge.desired_weight_bits)
            .count()
            .try_into()
            .unwrap_or(u32::MAX);
        let invalid_pressure_count = view_blend
            .invalid_pressure
            .iter()
            .filter(|invalid| **invalid)
            .count()
            .try_into()
            .unwrap_or(u32::MAX);
        let max_lag = edges.iter().fold(0.0_f32, |max_lag, edge| {
            max_lag.max(
                (f32::from_bits(edge.displayed_weight_bits)
                    - f32::from_bits(edge.desired_weight_bits))
                .abs(),
            )
        });
        assert_eq!(
            view_blend.upload.lagging_edge_count, lagging_count,
            "{context} radix upload lag count disagreed with its exact weights"
        );
        let aggregate = candidate.view_blend.as_ref().unwrap_or_else(|| {
            panic!("{context} frozen candidate proof omitted its post-Cleanup blend aggregate")
        });
        assert_eq!(
            aggregate.status.missing_consumer_count, 0,
            "{context} post-Cleanup aggregate omitted the promoted private consumer"
        );
        assert_eq!(
            aggregate.endpoints, view_blend.endpoints,
            "{context} single-consumer aggregate endpoint mask disagreed with the promoted radix snapshot"
        );
        assert_eq!(
            aggregate
                .weights
                .iter()
                .map(|weight| (weight.displayed.to_bits(), weight.desired.to_bits()))
                .collect::<Vec<_>>(),
            view_blend
                .weights
                .iter()
                .map(|weight| (weight.displayed.to_bits(), weight.desired.to_bits()))
                .collect::<Vec<_>>(),
            "{context} single-consumer aggregate weights disagreed with the promoted radix snapshot"
        );
        assert_eq!(
            (
                aggregate.status.edge_count,
                aggregate.status.lagging_count,
                aggregate.status.invalid_pressure_count,
                aggregate.status.max_lag.to_bits(),
            ),
            (
                view_blend.upload.edge_count,
                lagging_count,
                invalid_pressure_count,
                max_lag.to_bits(),
            ),
            "{context} single-consumer aggregate status disagreed with the promoted radix snapshot"
        );
        if exact_token_aggregate {
            assert_eq!(
                (
                    aggregate.status.max_delta.to_bits(),
                    aggregate.status.weighted_record_energy.to_bits(),
                ),
                (
                    view_blend.upload.last_max_delta.to_bits(),
                    (view_blend
                        .upload
                        .last_weighted_record_energy
                        .min(f64::from(f32::MAX)) as f32)
                        .to_bits(),
                ),
                "{context} exact-token single-consumer aggregate drawable metrics disagreed with the promoted radix snapshot"
            );
        }
        GardenViewBlendObservation {
            compaction_generation: drawable.compaction_generation,
            publication_generation: drawable.radix_publication_generation,
            compute_input_generation: drawable.compute_input_generation,
            candidate_prepared: candidate.prepared,
            candidate_active: candidate.active,
            candidate_transitioning: candidate.transitioning,
            candidate_fingerprint: (
                drawable.candidate_fingerprint_primary,
                drawable.candidate_fingerprint_secondary,
                drawable.candidate_range_count,
            ),
            candidate_content_signature: drawable.candidate_content_signature,
            candidate_atlas_allocation_epoch: drawable.candidate_atlas_allocation_epoch,
            desired_evaluation_complete: view_blend.desired_evaluation_complete,
            evaluation_view: view_blend.evaluation_view,
            evaluation_target: view_blend.evaluation_target,
            current_render_view,
            current_render_target,
            status: GardenViewBlendStatusObservation {
                edge_count: view_blend.upload.edge_count,
                lagging_count,
                invalid_pressure_count,
                missing_consumer_count: aggregate.status.missing_consumer_count,
                max_lag_bits: max_lag.to_bits(),
                max_delta_bits: view_blend.upload.last_max_delta.to_bits(),
                weighted_record_energy_bits: (view_blend
                    .upload
                    .last_weighted_record_energy
                    .min(f64::from(f32::MAX)) as f32)
                    .to_bits(),
            },
            target_ranges,
            presentation_ranges,
            required_ranges,
            edges,
            upload: view_blend.upload,
        }
    }

    const GARDEN_INTERACTIVE_LOW_QUALITY: f32 = 0.35;
    const GARDEN_INTERACTIVE_REVIEW_QUALITY: f32 = 0.65;
    const GARDEN_INTERACTIVE_MAX_ACTIVE_GAUSSIANS: u64 = 8_000_000;
    const GARDEN_INTERACTIVE_STABLE_FRAMES: u32 = 120;
    const GARDEN_INTERACTIVE_CAPTURE_GAP_FRAMES: u32 = 8;
    const GARDEN_INTERACTIVE_TRANSITION_CAPTURE_INTERVAL: u32 = 120;
    // At the fixed 120 Hz runner this remains a thirty-second operational cap
    // for cold native I/O, bounded topology admission, and 120-frame fixed-point
    // evidence. Camera-conditioned weights have no frame-count completion
    // contract; ordinary resident motion is exact and late recovery is gated
    // directly by its published per-edge delta bound.
    const GARDEN_INTERACTIVE_REQUEST_MAX_FRAMES: u32 = 3_600;
    const GARDEN_INTERACTIVE_MAX_FRAMES: u32 = 20_000;

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    enum GardenInteractivePose {
        Overview,
        Closer,
        Farther,
        Orbit,
    }

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    enum GardenInteractiveScenario {
        NearLowCold,
        NearHigh,
        CloserHigh,
        FartherHigh,
        NearHighRecovery,
        OrbitHigh,
        NearHighReturn,
        NearLowReturn,
    }

    impl GardenInteractiveScenario {
        const ALL: [Self; 8] = [
            Self::NearLowCold,
            Self::NearHigh,
            Self::CloserHigh,
            Self::FartherHigh,
            Self::NearHighRecovery,
            Self::OrbitHigh,
            Self::NearHighReturn,
            Self::NearLowReturn,
        ];

        fn pose(self) -> GardenInteractivePose {
            match self {
                Self::CloserHigh => GardenInteractivePose::Closer,
                Self::FartherHigh => GardenInteractivePose::Farther,
                Self::OrbitHigh => GardenInteractivePose::Orbit,
                Self::NearLowCold
                | Self::NearHigh
                | Self::NearHighRecovery
                | Self::NearHighReturn
                | Self::NearLowReturn => GardenInteractivePose::Overview,
            }
        }

        fn quality(self) -> f32 {
            match self {
                Self::NearLowCold | Self::NearLowReturn => GARDEN_INTERACTIVE_LOW_QUALITY,
                Self::NearHigh
                | Self::CloserHigh
                | Self::FartherHigh
                | Self::NearHighRecovery
                | Self::OrbitHigh
                | Self::NearHighReturn => GARDEN_INTERACTIVE_REVIEW_QUALITY,
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct GardenSceneFrame {
        center: Vec3,
        radius: f32,
    }

    impl GardenSceneFrame {
        fn viewer_auto_frame_transform(self) -> Transform {
            let view_direction = Vec3::new(0.0, 1.5, 5.0).normalize();
            Transform::from_translation(self.center + view_direction * GARDEN_AUTO_FRAME_DISTANCE)
                .looking_at(self.center, Vec3::Y)
        }

        fn transform(self, pose: GardenInteractivePose) -> Transform {
            let offset = match pose {
                GardenInteractivePose::Overview => Vec3::Z * (2.4 * self.radius),
                GardenInteractivePose::Closer => Vec3::Z * (1.5 * self.radius),
                GardenInteractivePose::Farther => Vec3::Z * (4.0 * self.radius),
                GardenInteractivePose::Orbit => {
                    Vec3::new(1.0, 0.0, 1.0).normalize() * (2.4 * self.radius)
                }
            };
            Transform::from_translation(self.center + offset).looking_at(self.center, Vec3::Y)
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum GardenInteractivePhase {
        AwaitingManifest,
        PackageWaiting,
        PackageFirstPending,
        PackageSettledGap,
        PackageSecondPending,
        RetiringPackage,
        FlatWarmup,
        FlatPending,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum GardenInteractiveCapture {
        Transition(GardenInteractiveScenario),
        SettledFirst(GardenInteractiveScenario),
        SettledSecond(GardenInteractiveScenario),
        Flat(GardenInteractivePose),
    }

    struct GardenInteractiveSettlement {
        active_gaussians: u64,
        signature: GardenLogicalCutSignature,
        image: Vec<[f32; 4]>,
        start_frame: u32,
        settled_frame: u32,
        request_frames: u32,
        cut_changes: u32,
        blend_edges: u32,
        fractional_blend_edges: u32,
        blend_signature: GardenViewBlendPresentationSignature,
        blend_upload: LodViewBlendUploadStats,
        bounded_hard_frames: u32,
        transition_captures: u32,
        stale_visible_frames: u32,
        active_while_stale_frames: u32,
        retained_origin_frames: u32,
        lifecycle: Vec<GaussianLodPackagePhase>,
        quality_status: LodEffectiveStatus,
        target_satisfied: bool,
    }

    #[derive(Resource)]
    struct GardenInteractiveState {
        package_root: PathBuf,
        manifest_name: String,
        source_path: PathBuf,
        settings: GaussianLodSettings,
        manifest: Option<Handle<GaussianLodAsset>>,
        target: Option<Handle<Image>>,
        cloud: Option<Entity>,
        camera: Option<Entity>,
        scene_frame: Option<GardenSceneFrame>,
        manifest_validated: bool,
        scenario_index: usize,
        phase: GardenInteractivePhase,
        total_frames: u32,
        phase_frames: u32,
        request_frames: u32,
        scenario_start_frame: u32,
        first_drawable_frame: Option<u32>,
        stable_frames: u32,
        request_cut_changes: u32,
        request_bounded_hard_frames: u32,
        request_seen_signatures: BTreeSet<GardenLogicalCutSignature>,
        transition_captures: u32,
        stale_visible_frames: u32,
        active_while_stale_frames: u32,
        retained_origin_frames: u32,
        lifecycle: Vec<GaussianLodPackagePhase>,
        last_resident_pages: Option<u32>,
        last_resident_change_request_frame: u32,
        quiescent_resident_pages: Option<u32>,
        last_observed_signature: Option<GardenLogicalCutSignature>,
        request_origin_signature: Option<GardenLogicalCutSignature>,
        quiescent_signature: Option<GardenLogicalCutSignature>,
        last_blend_signature: Option<GardenViewBlendPresentationSignature>,
        last_blend_upload: Option<LodViewBlendUploadStats>,
        last_active_blend: Option<GardenViewBlendObservation>,
        node_parents: GardenNodeParents,
        fractional_overlap_replacements: u32,
        authored_publication_hold: GardenAuthoredPublicationHold,
        promoted_drawable: GardenPromotedDrawableTracker,
        last_physical_drawable: Option<LodLastRadixDrawableForTesting>,
        last_cleanup_sequence: Option<u64>,
        quiescent_blend_signature: Option<GardenViewBlendPresentationSignature>,
        quiescent_blend_upload: Option<LodViewBlendUploadStats>,
        quiescent_quality_status: Option<LodEffectiveStatus>,
        quiescent_target_satisfied: Option<bool>,
        pending_capture: Option<GardenInteractiveCapture>,
        first_settled_image: Option<Vec<[f32; 4]>>,
        settlements: BTreeMap<GardenInteractiveScenario, GardenInteractiveSettlement>,
        flat_pose_index: usize,
    }

    impl GardenInteractiveState {
        fn new(
            package_root: PathBuf,
            manifest_name: String,
            source_path: PathBuf,
            settings: GaussianLodSettings,
        ) -> Self {
            Self {
                package_root,
                manifest_name,
                source_path,
                settings,
                manifest: None,
                target: None,
                cloud: None,
                camera: None,
                scene_frame: None,
                manifest_validated: false,
                scenario_index: 0,
                phase: GardenInteractivePhase::AwaitingManifest,
                total_frames: 0,
                phase_frames: 0,
                request_frames: 0,
                scenario_start_frame: 0,
                first_drawable_frame: None,
                stable_frames: 0,
                request_cut_changes: 0,
                request_bounded_hard_frames: 0,
                request_seen_signatures: BTreeSet::new(),
                transition_captures: 0,
                stale_visible_frames: 0,
                active_while_stale_frames: 0,
                retained_origin_frames: 0,
                lifecycle: Vec::new(),
                last_resident_pages: None,
                last_resident_change_request_frame: 0,
                quiescent_resident_pages: None,
                last_observed_signature: None,
                request_origin_signature: None,
                quiescent_signature: None,
                last_blend_signature: None,
                last_blend_upload: None,
                last_active_blend: None,
                node_parents: BTreeMap::new(),
                fractional_overlap_replacements: 0,
                authored_publication_hold: GardenAuthoredPublicationHold::default(),
                promoted_drawable: GardenPromotedDrawableTracker::default(),
                last_physical_drawable: None,
                last_cleanup_sequence: None,
                quiescent_blend_signature: None,
                quiescent_blend_upload: None,
                quiescent_quality_status: None,
                quiescent_target_satisfied: None,
                pending_capture: None,
                first_settled_image: None,
                settlements: BTreeMap::new(),
                flat_pose_index: 0,
            }
        }

        fn scenario(&self) -> GardenInteractiveScenario {
            GardenInteractiveScenario::ALL[self.scenario_index]
        }

        fn begin_scenario(&mut self, scenario_index: usize) {
            self.scenario_index = scenario_index;
            self.settings.quality = self.scenario().quality();
            self.phase = GardenInteractivePhase::PackageWaiting;
            self.phase_frames = 0;
            self.request_frames = 0;
            self.scenario_start_frame = self.total_frames;
            self.stable_frames = 0;
            self.request_cut_changes = 0;
            self.request_bounded_hard_frames = 0;
            self.request_seen_signatures.clear();
            if let Some(origin) = self.last_observed_signature.clone() {
                self.request_seen_signatures.insert(origin);
            }
            self.transition_captures = 0;
            self.stale_visible_frames = 0;
            self.active_while_stale_frames = 0;
            self.retained_origin_frames = 0;
            self.lifecycle.clear();
            self.last_resident_pages = None;
            self.last_resident_change_request_frame = 0;
            self.quiescent_resident_pages = None;
            self.quiescent_signature = None;
            self.last_blend_signature = None;
            self.last_blend_upload = None;
            self.quiescent_blend_signature = None;
            self.quiescent_blend_upload = None;
            self.quiescent_quality_status = None;
            self.quiescent_target_satisfied = None;
            self.request_origin_signature = self.last_observed_signature.clone();
            self.pending_capture = None;
            self.first_settled_image = None;
        }
    }

    fn assert_canonical_garden_source(source_path: &Path) {
        let mut source = fs::File::open(source_path).unwrap_or_else(|error| {
            panic!(
                "failed to open canonical Garden source {}: {error}",
                source_path.display()
            )
        });
        let mut digest = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let count = source.read(&mut buffer).unwrap_or_else(|error| {
                panic!(
                    "failed to hash canonical Garden source {}: {error}",
                    source_path.display()
                )
            });
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        let actual = format!("{:x}", digest.finalize());
        assert_eq!(
            actual, GARDEN_SOURCE_SHA256,
            "Garden source hash does not match the package's authenticated canonical PLY"
        );
    }

    fn assert_canonical_garden_manifest_bytes(encoded: &[u8]) {
        let actual = format!("{:x}", Sha256::digest(encoded));
        assert_eq!(
            actual, GARDEN_MANIFEST_SHA256,
            "Garden manifest hash does not match the qualified host-Morton ABI-16 artifact"
        );
    }

    fn garden_env_f32(name: &str, default: f32) -> f32 {
        let Some(value) = env::var_os(name) else {
            return default;
        };
        let value = value
            .to_str()
            .unwrap_or_else(|| panic!("{name} must be valid UTF-8"))
            .parse::<f32>()
            .unwrap_or_else(|error| panic!("invalid {name}: {error}"));
        assert!(
            value.is_finite() && (0.0..=1.0).contains(&value),
            "{name} must be a finite quality in [0, 1]"
        );
        value
    }

    fn garden_env_u64(name: &str, default: u64) -> u64 {
        let Some(value) = env::var_os(name) else {
            return default;
        };
        let value = value
            .to_str()
            .unwrap_or_else(|| panic!("{name} must be valid UTF-8"))
            .parse::<u64>()
            .unwrap_or_else(|error| panic!("invalid {name}: {error}"));
        assert!(value > 0, "{name} must be greater than zero");
        value
    }

    fn load_canonical_garden_source(source_path: &Path) -> PlanarGaussian3d {
        let source = fs::File::open(source_path).unwrap_or_else(|error| {
            panic!(
                "failed to reopen canonical Garden source {}: {error}",
                source_path.display()
            )
        });
        let mut reader = BufReader::with_capacity(1024 * 1024, source);
        let cloud = parse_ply_3d(&mut reader).unwrap_or_else(|error| {
            panic!(
                "failed to parse canonical Garden source {}: {error}",
                source_path.display()
            )
        });
        assert_eq!(
            cloud.position_visibility.len() as u64,
            GARDEN_SOURCE_GAUSSIANS,
            "canonical Garden PLY record count changed after authentication"
        );
        cloud
    }

    fn setup_garden_package_static(
        mut commands: Commands,
        mut state: ResMut<GardenPackageStaticState>,
        asset_server: Res<AssetServer>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let target = images.add(Image::new_target_texture(
            GARDEN_TARGET_WIDTH,
            GARDEN_TARGET_HEIGHT,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let manifest: Handle<GaussianLodAsset> = asset_server.load(state.manifest_name.clone());
        let cloud = commands
            .spawn((
                GaussianLodHandle(manifest.clone()),
                GaussianLodPackageSource::native_directory(
                    state.package_root.to_string_lossy().into_owned(),
                ),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::Radix,
                    radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                    ..default()
                },
                state.settings.clone(),
                Transform::IDENTITY,
                Visibility::Visible,
                Name::new("canonical_garden_lod_package"),
            ))
            .id();
        let camera = commands
            .spawn((
                Camera3d::default(),
                Camera::default(),
                Projection::Perspective(PerspectiveProjection {
                    far: 1_000_000.0,
                    ..default()
                }),
                RenderTarget::Image(target.clone().into()),
                state.scene_frame.viewer_auto_frame_transform(),
                Tonemapping::None,
                GaussianCamera::default(),
                Name::new("canonical_garden_static_camera"),
            ))
            .id();
        state.manifest = Some(manifest);
        state.target = Some(target);
        state.cloud = Some(cloud);
        state.camera = Some(camera);
    }

    // Bevy system parameters intentionally keep each ECS access explicit so
    // schedule conflicts remain visible in this end-to-end acceptance gate.
    #[allow(clippy::too_many_arguments)]
    fn drive_garden_package_static(
        mut commands: Commands,
        mut state: ResMut<GardenPackageStaticState>,
        mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
        manifests: Res<Assets<GaussianLodAsset>>,
        statuses: Query<&GaussianLodPackageStatus>,
        lod_statuses: Query<&GaussianLodStatus>,
        candidates: Query<&LodRenderCandidates>,
        camera_transforms: Query<&Transform, With<GaussianCamera>>,
        upload_budget_status: Res<LodAtlasUploadBudgetStatus>,
        upload_queue: Res<LodAtlasUploadQueue>,
        blend_probe: Res<GardenViewBlendRenderProbe>,
    ) {
        const MAX_FRAMES: u32 = 7_200;
        const REQUIRED_STABLE_TARGET_FRAMES: u32 = 240;

        state.total_frames += 1;
        state.phase_frames += 1;
        assert!(
            state.total_frames <= MAX_FRAMES,
            "Garden package did not stabilize: frames={}, active_endpoints={}, peak_pages={}, status={:?}",
            state.total_frames,
            state.cut_changes,
            state.peak_resident_pages,
            state.cloud.and_then(|cloud| statuses.get(cloud).ok()),
        );
        let camera = state.camera.expect("Garden camera exists");
        assert_eq!(
            camera_transforms
                .get(camera)
                .expect("Garden camera transform exists"),
            &state.scene_frame.viewer_auto_frame_transform(),
            "Garden stability test camera moved"
        );

        match state.capture_phase {
            GardenCapturePhase::RetiringPackage => {
                if state.phase_frames < GARDEN_PACKAGE_RETIRE_FRAMES {
                    return;
                }
                let source = load_canonical_garden_source(&state.source_path);
                let cloud = commands
                    .spawn((
                        PlanarGaussian3dHandle(gaussian_assets.add(source)),
                        CloudSettings {
                            gaussian_mode: GaussianMode::Gaussian3d,
                            sort_mode: SortMode::Radix,
                            radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                            ..default()
                        },
                        Transform::IDENTITY,
                        Visibility::Visible,
                        Name::new("canonical_garden_flat_reference"),
                    ))
                    .id();
                state.cloud = Some(cloud);
                state.phase_frames = 0;
                state.capture_phase = GardenCapturePhase::FlatReferenceWarmup;
                return;
            }
            GardenCapturePhase::FlatReferenceWarmup => {
                if state.phase_frames >= GARDEN_FLAT_REFERENCE_WARMUP_FRAMES {
                    commands.spawn(Screenshot::image(
                        state.target.clone().expect("Garden render target exists"),
                    ));
                    state.capture_phase = GardenCapturePhase::FlatReferencePending;
                }
                return;
            }
            GardenCapturePhase::FlatReferencePending => return,
            GardenCapturePhase::AwaitingStableCut
            | GardenCapturePhase::PackageCapturePending
            | GardenCapturePhase::BetweenPackageCaptures => {}
        }

        let cloud = state.cloud.expect("Garden cloud exists");
        let Ok(status) = statuses.get(cloud) else {
            assert_eq!(
                state.capture_phase,
                GardenCapturePhase::AwaitingStableCut,
                "Garden package status disappeared during the visual-stability window"
            );
            return;
        };
        assert!(
            status.failure.is_none(),
            "Garden package failed: {status:?}"
        );
        assert_eq!(status.terminal_failures, 0);
        assert!(
            status.resident_pages <= state.settings.budgets.max_resident_pages,
            "Garden exceeded its resident-page budget: {status:?}"
        );
        assert!(
            upload_budget_status.last_error().is_none(),
            "Garden atlas upload budget failed: {:?}",
            upload_budget_status.last_error()
        );
        let queued_uploads = upload_queue.queued_slot_count();
        if state.capture_phase != GardenCapturePhase::AwaitingStableCut {
            assert_eq!(
                queued_uploads, 0,
                "Garden queued an atlas upload during the visual-stability window"
            );
        }
        state.peak_resident_pages = state.peak_resident_pages.max(status.resident_pages);
        if status.phase != GaussianLodPackagePhase::Active {
            assert_eq!(
                state.capture_phase,
                GardenCapturePhase::AwaitingStableCut,
                "Garden package left ACTIVE during the visual-stability window: {status:?}"
            );
            state.stable_target_frames = 0;
            return;
        }
        let Some(render_blend) = blend_probe.latest_snapshot() else {
            assert!(
                state.promoted_drawable.last_accepted.is_none(),
                "Garden render-world drawable disappeared after its first promoted output"
            );
            state.stable_target_frames = 0;
            return;
        };
        let drawable_class = state
            .promoted_drawable
            .classify(&render_blend, "static Garden handoff");
        let (render_candidate, retained) = match drawable_class {
            GardenPromotedDrawableClass::CurrentCandidate => {
                (render_blend.candidate.clone(), false)
            }
            GardenPromotedDrawableClass::RetainedCurrent(retained) => (retained, true),
        };
        assert!(
            !render_candidate.failed,
            "static Garden promoted candidate failed"
        );
        if render_candidate.temporal_mode == Some(LodTemporalTransitionMode::BoundedHardCohort) {
            state.bounded_hard_frames += 1;
            panic!(
                "authenticated ABI-16 Garden promoted a hard cohort during static qualification"
            );
        }
        let prepared_blend = observe_garden_view_blend_with_render_state(
            &render_candidate,
            render_blend,
            !retained,
            if retained {
                "static Garden retained handoff"
            } else {
                "static Garden promoted handoff"
            },
        );
        prepared_blend.assert_dynamic_coherent("static Garden promoted handoff");
        if retained {
            assert_eq!(
                state.capture_phase,
                GardenCapturePhase::AwaitingStableCut,
                "Garden retained an old drawable during the visual-stability window"
            );
            state.stable_target_frames = 0;
            return;
        }

        let Ok(candidates) = candidates.get(cloud) else {
            state.stable_target_frames = 0;
            return;
        };
        let Some(candidate) = candidates.get(camera) else {
            state.stable_target_frames = 0;
            return;
        };
        assert!(!candidate.failed(), "Garden main-world candidate failed");

        // A PREPARED candidate can already own the exact radix-promoted
        // authored endpoint. It is qualified above, but only ACTIVE advances
        // the static logical-cut and stationary-window contracts.
        if !candidate.render_is_active_for_testing()
            || u64::from(candidate.rendered_candidate_count()) != status.active_gaussians
            || candidate.rendered_quality_status().active_gaussians != status.active_gaussians
            || garden_physical_range_signature(candidate.frontier().physical_ranges())
                != prepared_blend.target_ranges
            || garden_physical_range_signature(candidate.render_ranges())
                != prepared_blend.presentation_ranges
            || garden_physical_range_signature(candidate.required_atlas_ranges_for_testing())
                != prepared_blend.required_ranges
        {
            assert_eq!(
                state.capture_phase,
                GardenCapturePhase::AwaitingStableCut,
                "Garden candidate became incoherent during the visual-stability window"
            );
            state.stable_target_frames = 0;
            return;
        }
        let blend = prepared_blend;
        blend.assert_stationary_fixed_point("static Garden");
        blend.assert_manifest_edge_topology(&state.node_parents, "static Garden");

        let signature = (
            status.active_gaussians,
            candidate
                .render_ranges()
                .iter()
                .map(|range| (range.node.0, range.page.0, range.count))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            signature
                .1
                .iter()
                .map(|range| u64::from(range.2))
                .sum::<u64>(),
            status.active_gaussians,
            "Garden published incoherent range and active-record counts"
        );
        let range_count = signature.1.len();
        if state.last_signature.as_ref() != Some(&signature) {
            assert_eq!(
                state.capture_phase,
                GardenCapturePhase::AwaitingStableCut,
                "Garden cut changed after visual-stability capture began"
            );
            state.cut_changes += 1;
            if state.cut_changes <= 32 {
                eprintln!(
                    "Garden static ACTIVE endpoint {} at frame {}: active={}, ranges={}, resident={}, ratio={:.3}, degradation={:?}",
                    state.cut_changes,
                    state.total_frames,
                    status.active_gaussians,
                    signature.1.len(),
                    status.resident_pages,
                    candidate
                        .rendered_quality_status()
                        .achieved_max_target_ratio,
                    candidate.rendered_quality_status().degradation,
                );
            }
            state.last_signature = Some(signature);
            state.stable_target_frames = 0;
        }

        let rendered_quality = candidate.rendered_quality_status();
        let requested_target = state.settings.quality_target();
        let fixed_point_revision = lod_statuses.get(cloud).ok().and_then(|published| {
            (published.failure.is_none()
                && published.source == GaussianLodSourceKind::Package
                && published.lifecycle == GaussianLodLifecycle::Active
                && published.selection_mode == state.settings.selection_mode
                && published.frozen_views
                    == u32::from(state.settings.selection_mode == LodSelectionMode::Frozen)
                && published.active_views == 1
                && published.temporal_transition_mode.is_none()
                && published.selected_gaussians == status.active_gaussians
                && published.selected_gaussians == rendered_quality.active_gaussians
                && published.submitted_candidates == candidate.rendered_candidate_count()
                && published.resident_pages == status.resident_pages
                && published.view_blend_edges == blend.status.edge_count
                && published.view_blend_lagging_edges == 0
                && published.view_blend_invalid_pressure_evaluations == 0
                && published.view_blend_missing_consumers == 0
                && published.view_blend_max_lag.to_bits() == 0.0_f32.to_bits()
                && published.view_blend_max_delta.to_bits() == blend.status.max_delta_bits
                && published.view_blend_weighted_record_energy.to_bits()
                    == blend.status.weighted_record_energy_bits
                && published.requested_target == requested_target
                && published.requested_target == rendered_quality.requested_target
                && published.achieved_max_error_px == Some(rendered_quality.achieved_max_error_px)
                && published.achieved_max_target_ratio
                    == Some(rendered_quality.achieved_max_target_ratio)
                && published.target_satisfied == Some(true)
                && published.degradation == LodDegradation::None
                && rendered_quality.degradation == LodDegradation::None
                && rendered_quality.requested_pages == 0)
                .then_some(published.revision)
        });
        if fixed_point_revision.is_none() {
            assert_eq!(
                state.capture_phase,
                GardenCapturePhase::AwaitingStableCut,
                "Garden lost its exact public fixed point during the visual-stability window"
            );
            state.stable_target_frames = 0;
            state.last_status_revision = None;
            state.last_resident_pages = Some(status.resident_pages);
            state.last_blend_signature = None;
            state.last_blend_status = None;
            state.last_blend_upload = None;
            state.last_compaction_generation = None;
            return;
        }

        let resident_plateau = state.last_resident_pages == Some(status.resident_pages);
        let status_revision_plateau = state.last_status_revision == fixed_point_revision;
        let blend_signature = blend.presentation_signature();
        let blend_signature_plateau = state.last_blend_signature.as_ref() == Some(&blend_signature);
        let blend_status_plateau = state.last_blend_status.as_ref() == Some(&blend.status);
        let blend_upload_plateau = state.last_blend_upload == Some(blend.upload);
        let compaction_generation_plateau =
            state.last_compaction_generation == Some(blend.compaction_generation);
        if state.capture_phase != GardenCapturePhase::AwaitingStableCut {
            assert!(
                resident_plateau,
                "Garden residency changed during the visual-stability window: previous={:?}, current={}",
                state.last_resident_pages, status.resident_pages
            );
            assert!(
                status_revision_plateau,
                "Garden public fixed-point observation changed during the visual-stability window: previous={:?}, current={fixed_point_revision:?}",
                state.last_status_revision,
            );
            assert!(
                blend_signature_plateau,
                "Garden fractional topology/weights changed during the visual-stability window"
            );
            assert!(
                blend_status_plateau,
                "Garden fractional status changed during the visual-stability window"
            );
            assert!(
                blend_upload_plateau,
                "Garden stationary fractional state performed immutable/weight/allocation work during the visual-stability window: previous={:?}, current={:?}",
                state.last_blend_upload, blend.upload,
            );
            assert!(
                compaction_generation_plateau,
                "Garden recreated its compaction state during the visual-stability window"
            );
        }
        state.last_resident_pages = Some(status.resident_pages);
        state.last_status_revision = fixed_point_revision;
        state.last_blend_signature = Some(blend_signature);
        state.last_blend_status = Some(blend.status.clone());
        state.last_blend_upload = Some(blend.upload);
        state.last_compaction_generation = Some(blend.compaction_generation);
        if resident_plateau
            && status_revision_plateau
            && blend_signature_plateau
            && blend_status_plateau
            && blend_upload_plateau
            && compaction_generation_plateau
            && queued_uploads == 0
            && status.active_gaussians >= 1_000_000
            && range_count >= 1_000
        {
            state.stable_target_frames += 1;
        } else {
            state.stable_target_frames = 0;
        }
        if state.stable_target_frames < REQUIRED_STABLE_TARGET_FRAMES
            || state.capture_phase == GardenCapturePhase::PackageCapturePending
        {
            return;
        }

        assert_eq!(
            state.bounded_hard_frames, 0,
            "authenticated ABI-16 Garden used a hard cohort during static qualification"
        );
        let fractional_edges = blend.edges.iter().filter(|edge| edge.endpoint == 0).count();
        assert!(
            fractional_edges > 0,
            "static ABI-16 Garden did not exercise a stationary in-band fractional presentation"
        );
        assert!(
            blend
                .presentation_ranges
                .iter()
                .all(|range| blend.required_ranges.contains(range)),
            "static ABI-16 Garden presentation escaped its generation-safe required union"
        );
        assert!(
            blend
                .target_ranges
                .iter()
                .all(|range| blend.required_ranges.contains(range)),
            "static ABI-16 Garden target cut escaped its retained parent/children union"
        );
        match state.capture_phase {
            GardenCapturePhase::AwaitingStableCut => {
                let manifest = manifests
                    .get(
                        state
                            .manifest
                            .as_ref()
                            .expect("Garden manifest handle exists"),
                    )
                    .expect("active Garden package retains its manifest asset");
                assert_canonical_garden_manifest(manifest.manifest());
                commands.spawn(Screenshot::image(
                    state.target.clone().expect("Garden render target exists"),
                ));
                state.capture_phase = GardenCapturePhase::PackageCapturePending;
            }
            GardenCapturePhase::BetweenPackageCaptures => {
                state.capture_gap_frames += 1;
                if state.capture_gap_frames >= GARDEN_CAPTURE_INTERVAL_FRAMES {
                    commands.spawn(Screenshot::image(
                        state.target.clone().expect("Garden render target exists"),
                    ));
                    state.capture_phase = GardenCapturePhase::PackageCapturePending;
                }
            }
            GardenCapturePhase::PackageCapturePending => {}
            GardenCapturePhase::RetiringPackage
            | GardenCapturePhase::FlatReferenceWarmup
            | GardenCapturePhase::FlatReferencePending => {
                unreachable!("flat-reference phases return before package-state inspection")
            }
        }
    }

    fn assert_canonical_garden_manifest(manifest: &GaussianLodManifest) {
        assert_eq!(
            manifest.header.source_gaussian_count, GARDEN_SOURCE_GAUSSIANS,
            "Garden visual acceptance requires the canonical source record count"
        );
        assert_ne!(
            manifest.build.source_fingerprint, 0,
            "Garden package omits its canonical-source fingerprint"
        );
        let bounds = manifest
            .scene_bounds
            .expect("canonical Garden manifest has scene bounds");
        assert_eq!(bounds.min, GARDEN_SCENE_MIN, "Garden scene minimum drifted");
        assert_eq!(bounds.max, GARDEN_SCENE_MAX, "Garden scene maximum drifted");
        assert_eq!(
            bounds.center(),
            GARDEN_SCENE_CENTER,
            "Garden scene center drifted"
        );
        assert_eq!(
            bounds.radius(),
            GARDEN_SCENE_RADIUS,
            "Garden scene radius drifted"
        );
        assert_eq!(
            manifest.header.node_count as usize, GARDEN_NODE_PAGE_COUNT,
            "Garden visual acceptance requires the canonical node topology"
        );
        assert_eq!(manifest.nodes.len(), GARDEN_NODE_PAGE_COUNT);
        assert_eq!(
            manifest.header.page_count as usize, GARDEN_NODE_PAGE_COUNT,
            "Garden visual acceptance requires one logical page per canonical node"
        );
        assert_eq!(manifest.pages.len(), GARDEN_NODE_PAGE_COUNT);
        assert_eq!(
            manifest.build.builder_abi_version, GARDEN_EXTERNAL_BUILDER_ABI,
            "Garden visual acceptance requires the external progressive ABI 16 package"
        );
        assert_eq!(
            manifest.build.reducer,
            LodReducerKind::MomentMerge,
            "Garden visual acceptance requires MomentMerge representatives"
        );
        assert_eq!(
            manifest.build.reducer_version, GARDEN_MOMENT_MERGE_REDUCER_VERSION,
            "Garden visual acceptance requires the boundary-aware MomentMerge v4 reducer"
        );
        let morph = manifest
            .morph_map
            .as_ref()
            .expect("ABI-16 Garden visual acceptance requires the monotone morph sidecar");
        assert_eq!(
            morph.schema_version, 1,
            "Garden visual acceptance requires morph-map schema 1"
        );
        assert_eq!(
            morph.node_runs.len(),
            manifest.nodes.len(),
            "Garden morph map must cover every hierarchy node"
        );
        assert!(
            !morph.child_run_lengths.is_empty(),
            "Garden morph map has no authored child runs"
        );
        assert!(
            manifest.build.has_bounded_refinement_amplification(),
            "Garden package does not advertise bounded refinement amplification"
        );
        assert_eq!(
            u64::from(manifest.build.settings.branching_factor),
            GARDEN_MAX_REFINEMENT_AMPLIFICATION,
            "Garden package uses an unexpected refinement-amplification bound"
        );

        let pages_by_id = manifest
            .pages
            .iter()
            .map(|page| (page.id, page))
            .collect::<BTreeMap<_, _>>();
        let node_pages = manifest
            .nodes
            .iter()
            .map(|node| node.representation.page)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            node_pages.len(),
            GARDEN_NODE_PAGE_COUNT,
            "Garden package must retain one independently addressable logical page per node"
        );

        for node in &manifest.nodes {
            let page = pages_by_id
                .get(&node.representation.page)
                .expect("validated Garden node references a known page");
            assert_eq!(
                node.representation.offset, 0,
                "Garden node {:?} does not own its logical page",
                node.id
            );
            assert_eq!(
                node.representation.count, page.gaussian_count,
                "Garden node {:?} does not cover its complete logical page",
                node.id
            );
            if node.is_leaf() {
                continue;
            }
            assert!(
                node.representation.count > 1,
                "Garden internal node {:?} regressed to a singleton page",
                node.id
            );
            let child_start = node.children.start as usize;
            let child_end = node
                .children
                .end()
                .expect("validated Garden child range does not overflow")
                as usize;
            let child_representatives = manifest.nodes[child_start..child_end]
                .iter()
                .try_fold(0_u64, |count, child| {
                    count.checked_add(u64::from(child.representation.count))
                })
                .expect("Garden child representation count does not overflow");
            let maximum = u64::from(node.representation.count)
                .checked_mul(GARDEN_MAX_REFINEMENT_AMPLIFICATION)
                .expect("Garden refinement bound does not overflow");
            assert!(
                child_representatives <= maximum,
                "Garden node {:?} refinement amplification exceeded 8x: parent={}, children={child_representatives}",
                node.id,
                node.representation.count
            );
        }
    }

    fn on_garden_package_capture(
        trigger: On<ScreenshotCaptured>,
        mut commands: Commands,
        mut state: ResMut<GardenPackageStaticState>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("Garden offscreen capture converts")
            .to_rgba8();
        assert_eq!(rgba.width(), GARDEN_TARGET_WIDTH);
        assert_eq!(rgba.height(), GARDEN_TARGET_HEIGHT);
        let linear = linear_rgba(rgba.as_raw());
        let sanity = garden_image_sanity(
            &linear,
            GARDEN_TARGET_WIDTH as usize,
            GARDEN_TARGET_HEIGHT as usize,
        );
        // The authenticated viewer auto-frame fits conservative manifest
        // support bounds, including sparse outliers. Its scene silhouette is
        // therefore intentionally judged by the shared bounds-fit nonblank
        // contract; the unchanged flat-reference oracle below owns fidelity.
        assert_garden_bounds_fit_image_nonblank(sanity);

        match state.capture_phase {
            GardenCapturePhase::PackageCapturePending => {
                let temporal = if let Some(first_capture) = state.first_capture.as_ref() {
                    let temporal = compare_linear_rgba(first_capture, &linear, 0.5)
                        .expect("Garden temporal image metrics are valid");
                    let foreground_iou = garden_foreground_iou(first_capture, &linear);
                    assert!(
                        temporal.psnr_rgb.is_infinite() || temporal.psnr_rgb >= 55.0,
                        "static Garden RGB changed between stable samples: sample={}, metrics={temporal:?}",
                        state.capture_samples + 1
                    );
                    assert!(
                        temporal.foreground_psnr_rgb.is_infinite()
                            || temporal.foreground_psnr_rgb >= 55.0,
                        "static Garden foreground changed between stable samples: sample={}, metrics={temporal:?}",
                        state.capture_samples + 1
                    );
                    assert!(
                        temporal.luminance_ssim >= 0.9999,
                        "static Garden luminance changed between stable samples: sample={}, metrics={temporal:?}",
                        state.capture_samples + 1
                    );
                    assert!(
                        foreground_iou >= 0.9999 && temporal.alpha_mae <= 0.0001,
                        "static Garden silhouette flickered between stable samples: sample={}, iou={foreground_iou}, metrics={temporal:?}",
                        state.capture_samples + 1
                    );
                    assert!(
                        temporal.max_abs_error <= 0.02,
                        "static Garden pixels have an excessive temporal residual: sample={}, metrics={temporal:?}",
                        state.capture_samples + 1
                    );
                    Some((foreground_iou, temporal))
                } else {
                    None
                };

                if state.first_capture.is_none() {
                    state.first_capture = Some(linear);
                    state.first_sanity = Some(sanity);
                }
                state.capture_samples += 1;
                if state.capture_samples == GARDEN_CAPTURE_SAMPLES {
                    let blend = state
                        .last_blend_signature
                        .as_ref()
                        .expect("stable Garden blend signature exists");
                    let fractional_edges = blend
                        .edges
                        .iter()
                        .filter(|(_, endpoint, _, _)| *endpoint == 0)
                        .count();
                    eprintln!(
                        "Garden ABI 16 package stabilized after {} frames and {} in-memory samples: quality={}, max_active={}, active_ranges={}, resident_pages={}, active_cut_observations={}, blend_edges={}, fractional_edges={}, blend_upload={:?}, first={:?}, final={:?}, temporal={:?}",
                        state.total_frames,
                        state.capture_samples,
                        state.settings.quality,
                        state.settings.budgets.max_active_gaussians,
                        state
                            .last_signature
                            .as_ref()
                            .map_or(0, |signature| signature.1.len()),
                        state.last_resident_pages.unwrap_or_default(),
                        state.cut_changes,
                        blend.edges.len(),
                        fractional_edges,
                        state.last_blend_upload,
                        state
                            .first_sanity
                            .expect("first Garden sanity sample exists"),
                        sanity,
                        temporal,
                    );
                    let package = state.cloud.take().expect("Garden package entity exists");
                    commands.entity(package).despawn();
                    state.manifest = None;
                    state.phase_frames = 0;
                    state.capture_phase = GardenCapturePhase::RetiringPackage;
                } else {
                    state.capture_gap_frames = 0;
                    state.capture_phase = GardenCapturePhase::BetweenPackageCaptures;
                }
            }
            GardenCapturePhase::FlatReferencePending => {
                let package = state
                    .first_capture
                    .as_ref()
                    .expect("stable Garden package readback remains in memory");
                let spatial = garden_spatial_metrics(&linear, package);
                assert_garden_spatial_fidelity(spatial);
                eprintln!(
                    "Garden ABI 16 spatial oracle passed against the authenticated flat source after {} frames: quality={}, max_active={}, flat={sanity:?}, package={:?}, metrics={spatial:?}",
                    state.total_frames,
                    state.settings.quality,
                    state.settings.budgets.max_active_gaussians,
                    state
                        .first_sanity
                        .expect("first Garden package sanity sample exists"),
                );
                exit.write(AppExit::Success);
            }
            phase => panic!("Garden capture arrived during unexpected phase {phase:?}"),
        }
    }

    fn setup_garden_interactive(
        mut commands: Commands,
        mut state: ResMut<GardenInteractiveState>,
        asset_server: Res<AssetServer>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let target = images.add(Image::new_target_texture(
            GARDEN_TARGET_WIDTH,
            GARDEN_TARGET_HEIGHT,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let manifest: Handle<GaussianLodAsset> = asset_server.load(state.manifest_name.clone());
        let cloud = commands
            .spawn((
                GaussianLodHandle(manifest.clone()),
                GaussianLodPackageSource::native_directory(
                    state.package_root.to_string_lossy().into_owned(),
                ),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::Radix,
                    radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                    ..default()
                },
                state.settings.clone(),
                Transform::IDENTITY,
                Visibility::Visible,
                Name::new("canonical_garden_interactive_lod_package"),
            ))
            .id();
        let camera = commands
            .spawn((
                Camera3d::default(),
                Camera {
                    is_active: false,
                    ..default()
                },
                Projection::Perspective(PerspectiveProjection {
                    far: 1_000_000.0,
                    ..default()
                }),
                RenderTarget::Image(target.clone().into()),
                Transform::IDENTITY,
                Tonemapping::None,
                GaussianCamera::default(),
                Name::new("canonical_garden_scripted_interaction_camera"),
            ))
            .id();
        state.manifest = Some(manifest);
        state.target = Some(target);
        state.cloud = Some(cloud);
        state.camera = Some(camera);
    }

    #[allow(clippy::too_many_arguments)]
    fn drive_garden_interactive(
        mut commands: Commands,
        mut state: ResMut<GardenInteractiveState>,
        mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
        manifests: Res<Assets<GaussianLodAsset>>,
        statuses: Query<&GaussianLodPackageStatus>,
        package_testing: Query<&GaussianLodPackageTestingSnapshot>,
        lod_statuses: Query<&GaussianLodStatus>,
        candidates: Query<&LodRenderCandidates>,
        mut cameras: Query<&mut Camera, With<GaussianCamera>>,
        mut camera_transforms: Query<&mut Transform, With<GaussianCamera>>,
        upload_budget_status: Res<LodAtlasUploadBudgetStatus>,
        upload_queue: Res<LodAtlasUploadQueue>,
        blend_probe: Res<GardenViewBlendRenderProbe>,
    ) {
        state.total_frames += 1;
        state.phase_frames += 1;
        assert!(
            state.total_frames <= GARDEN_INTERACTIVE_MAX_FRAMES,
            "interactive Garden gate timed out: phase={:?}, scenario={:?}, total_frames={}, request_frames={}, cuts={}, stale_visible={}, retained_origin={}, status={:?}",
            state.phase,
            state.scenario(),
            state.total_frames,
            state.request_frames,
            state.request_cut_changes,
            state.stale_visible_frames,
            state.retained_origin_frames,
            state.cloud.and_then(|cloud| statuses.get(cloud).ok()),
        );

        assert!(
            upload_budget_status.last_error().is_none(),
            "interactive Garden atlas upload budget failed: {:?}",
            upload_budget_status.last_error()
        );

        // Drain the ordered Render Cleanup cadence before any phase-specific
        // early return. Package phases interpret the payload below; startup,
        // retirement, and flat-reference phases only preserve sequence/order.
        let ordered_snapshot = blend_probe.next_ordered_snapshot();
        let cleanup_observed = ordered_snapshot.is_some();
        let render_snapshot = ordered_snapshot.and_then(|(sequence, snapshot)| {
            if let Some(previous) = state.last_cleanup_sequence {
                assert_eq!(
                    sequence,
                    previous.saturating_add(1),
                    "interactive Garden ordered Cleanup probe skipped or reordered a render observation"
                );
            }
            state.last_cleanup_sequence = Some(sequence);
            snapshot
        });

        match state.phase {
            GardenInteractivePhase::AwaitingManifest => {
                let Some(manifest_asset) = state
                    .manifest
                    .as_ref()
                    .and_then(|manifest| manifests.get(manifest))
                else {
                    return;
                };
                assert_canonical_garden_manifest(manifest_asset.manifest());
                state.node_parents = garden_node_parents(manifest_asset.manifest());
                let bounds = manifest_asset
                    .manifest()
                    .scene_bounds
                    .expect("authenticated Garden manifest carries scene bounds");
                let frame = GardenSceneFrame {
                    center: Vec3::from_array(bounds.center()),
                    radius: bounds.radius(),
                };
                assert!(
                    frame.center.is_finite() && frame.radius.is_finite() && frame.radius > 0.0,
                    "Garden scene frame is invalid: {frame:?}"
                );
                let camera = state.camera.expect("interactive Garden camera exists");
                *camera_transforms
                    .get_mut(camera)
                    .expect("interactive Garden camera transform exists") =
                    frame.transform(GardenInteractivePose::Overview);
                cameras
                    .get_mut(camera)
                    .expect("interactive Garden camera exists")
                    .is_active = true;
                state.scene_frame = Some(frame);
                state.manifest_validated = true;
                blend_probe.begin_ordered_capture();
                state.begin_scenario(0);
                eprintln!(
                    "Garden interactive frame derived from authenticated manifest at app frame {} ({:.3}s): center={:?}, radius={}, overview={:?}",
                    state.total_frames,
                    garden_frames_to_seconds(state.total_frames),
                    frame.center,
                    frame.radius,
                    frame.transform(GardenInteractivePose::Overview),
                );
                return;
            }
            GardenInteractivePhase::RetiringPackage => {
                assert!(
                    !cleanup_observed || render_snapshot.is_some(),
                    "interactive Garden drawable disappeared from an ordered Cleanup snapshot before package retirement"
                );
                if let Some(render) = render_snapshot.as_ref() {
                    let previous = state
                        .last_physical_drawable
                        .as_ref()
                        .expect("retiring Garden package previously qualified a drawable");
                    assert!(
                        garden_promoted_drawable_state_eq(previous, &render.drawable),
                        "Garden package changed its physical drawable after final fixed-point capture"
                    );
                }
                if state.phase_frames < GARDEN_PACKAGE_RETIRE_FRAMES {
                    return;
                }
                blend_probe.assert_ordered_capture_drained();
                let source = load_canonical_garden_source(&state.source_path);
                let cloud = commands
                    .spawn((
                        PlanarGaussian3dHandle(gaussian_assets.add(source)),
                        CloudSettings {
                            gaussian_mode: GaussianMode::Gaussian3d,
                            sort_mode: SortMode::Radix,
                            radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                            ..default()
                        },
                        Transform::IDENTITY,
                        Visibility::Visible,
                        Name::new("canonical_garden_interactive_flat_reference"),
                    ))
                    .id();
                state.cloud = Some(cloud);
                state.flat_pose_index = 0;
                let pose = garden_flat_oracle_poses()[0];
                let camera = state.camera.expect("interactive Garden camera exists");
                *camera_transforms
                    .get_mut(camera)
                    .expect("interactive Garden camera transform exists") = state
                    .scene_frame
                    .expect("Garden scene frame exists")
                    .transform(pose);
                state.phase = GardenInteractivePhase::FlatWarmup;
                state.phase_frames = 0;
                return;
            }
            GardenInteractivePhase::FlatWarmup => {
                if state.phase_frames >= GARDEN_FLAT_REFERENCE_WARMUP_FRAMES
                    && state.pending_capture.is_none()
                {
                    let pose = garden_flat_oracle_poses()[state.flat_pose_index];
                    commands.spawn(Screenshot::image(
                        state.target.clone().expect("Garden render target exists"),
                    ));
                    state.pending_capture = Some(GardenInteractiveCapture::Flat(pose));
                    state.phase = GardenInteractivePhase::FlatPending;
                }
                return;
            }
            GardenInteractivePhase::FlatPending => return,
            GardenInteractivePhase::PackageWaiting
            | GardenInteractivePhase::PackageFirstPending
            | GardenInteractivePhase::PackageSettledGap
            | GardenInteractivePhase::PackageSecondPending => {}
        }

        assert!(state.manifest_validated);
        state.request_frames += 1;
        let cloud = state.cloud.expect("interactive Garden package exists");
        let camera = state.camera.expect("interactive Garden camera exists");
        let diagnostic_candidates = candidates.get(cloud).ok();
        let diagnostic_candidate = diagnostic_candidates.and_then(|set| set.get(camera));
        let candidate_diagnostic = diagnostic_candidate.map(|candidate| {
            (
                diagnostic_candidates
                    .expect("diagnostic candidate belongs to its set")
                    .package_retention_for_testing(),
                candidate.render_commit_identity_for_testing(),
                (
                    candidate.render_is_prepared(),
                    candidate.render_is_active_for_testing(),
                    candidate.render_is_transitioning_for_testing(),
                    candidate.failed(),
                    candidate.view_blend_replan_requested_for_testing(),
                ),
                candidate.temporal_transition_mode(),
                candidate.rendered_quality_status(),
                candidate
                    .view_blend_testing_snapshot()
                    .map(|blend| blend.status),
            )
        });
        let render_diagnostic = blend_probe.latest_snapshot().map(|render| {
            (
                render.candidate.render_commit_identity,
                render.candidate.retention,
                (
                    render.candidate.prepared,
                    render.candidate.active,
                    render.candidate.transitioning,
                    render.candidate.failed,
                    render.candidate.view_blend_replan_requested,
                ),
                render.drawable.compaction_generation,
                render.drawable.radix_publication_generation,
                render
                    .drawable
                    .view_blend
                    .as_ref()
                    .map(|blend| blend.upload),
            )
        });
        assert!(
            state.request_frames <= GARDEN_INTERACTIVE_REQUEST_MAX_FRAMES,
            "Garden request did not quiesce: scenario={:?}, frames={} ({:.3}s), cuts={}, transitions={}, stale_visible={}, active_while_stale={}, retained_origin={}, lifecycle={:?}, status={:?}, work={:?}, public={:?}, candidate={candidate_diagnostic:?}, render={render_diagnostic:?}, last_resident_change_frame={}, last_cleanup_sequence={:?}, queued_uploads={}",
            state.scenario(),
            state.request_frames,
            garden_frames_to_seconds(state.request_frames),
            state.request_cut_changes,
            state.transition_captures,
            state.stale_visible_frames,
            state.active_while_stale_frames,
            state.retained_origin_frames,
            state.lifecycle,
            statuses.get(cloud).ok(),
            package_testing.get(cloud).ok(),
            lod_statuses.get(cloud).ok(),
            state.last_resident_change_request_frame,
            state.last_cleanup_sequence,
            upload_queue.queued_slot_count(),
        );

        let Ok(status) = statuses.get(cloud) else {
            assert!(
                render_snapshot.is_none(),
                "interactive Garden rendered a drawable before publishing package status"
            );
            assert!(
                state.settlements.is_empty(),
                "Garden package status disappeared after its first drawable cut"
            );
            state.stable_frames = 0;
            return;
        };
        if state.request_frames >= GARDEN_INTERACTIVE_REQUEST_MAX_FRAMES.saturating_sub(360)
            && state.request_frames.is_multiple_of(120)
        {
            eprintln!(
                "Garden {:?} late-request progress at frame {}: resident={}, last_resident_change={}, work={:?}, public={:?}, candidate={candidate_diagnostic:?}, render={render_diagnostic:?}",
                state.scenario(),
                state.request_frames,
                status.resident_pages,
                state.last_resident_change_request_frame,
                package_testing.get(cloud).ok(),
                lod_statuses.get(cloud).ok(),
            );
        }
        if state.lifecycle.last().copied() != Some(status.phase) {
            state.lifecycle.push(status.phase);
        }
        assert!(
            status.failure.is_none(),
            "Garden package failed: {status:?}"
        );
        assert_eq!(status.terminal_failures, 0);
        assert_ne!(status.phase, GaussianLodPackagePhase::Failed);
        assert_ne!(status.phase, GaussianLodPackagePhase::Degraded);
        assert!(
            status.resident_pages <= state.settings.budgets.max_resident_pages,
            "Garden exceeded its resident-page budget: {status:?}"
        );
        if !state.settlements.is_empty() {
            assert_eq!(
                status.phase,
                GaussianLodPackagePhase::Active,
                "Garden left ACTIVE after becoming drawable: scenario={:?}, status={status:?}",
                state.scenario()
            );
            assert!(
                status.active_gaussians > 0,
                "Garden discarded its visible cut during {:?}",
                state.scenario()
            );
        }

        let render_candidates = candidates.get(cloud).ok();
        let render_candidate = render_candidates.and_then(|set| set.get(camera));
        if let Some(candidate) = render_candidate {
            assert!(!candidate.failed(), "Garden render candidate failed");
            if candidate.temporal_transition_mode()
                == Some(LodTemporalTransitionMode::BoundedHardCohort)
            {
                state.request_bounded_hard_frames += 1;
                panic!(
                    "authenticated ABI-16 Garden fell back to a hard cohort during interactive qualification: scenario={:?}, request_frame={}",
                    state.scenario(),
                    state.request_frames,
                );
            }
        }
        let render_commit_identity = render_snapshot
            .as_ref()
            .map(|render| render.candidate.render_commit_identity);
        let drawable_unchanged = render_snapshot.as_ref().is_some_and(|render| {
            state
                .last_physical_drawable
                .as_ref()
                .is_some_and(|previous| {
                    garden_promoted_drawable_state_eq(previous, &render.drawable)
                })
        });
        let physical_drawable = render_snapshot
            .as_ref()
            .map(|render| render.drawable.clone());
        if cleanup_observed && render_snapshot.is_none() {
            assert!(
                state.promoted_drawable.last_accepted.is_none(),
                "interactive Garden drawable probe disappeared after its first promoted output"
            );
        }
        let blend_observation = render_snapshot.and_then(|render| {
            let drawable_class = state
                .promoted_drawable
                .classify(&render, "interactive Garden handoff");
            let (render_candidate, retained) = match drawable_class {
                GardenPromotedDrawableClass::CurrentCandidate => (render.candidate.clone(), false),
                GardenPromotedDrawableClass::RetainedCurrent(retained) => (retained, true),
            };
            assert!(
                !render_candidate.failed,
                "interactive Garden promoted drawable candidate failed"
            );
            if render_candidate.temporal_mode == Some(LodTemporalTransitionMode::BoundedHardCohort)
            {
                state.request_bounded_hard_frames += 1;
                panic!("interactive Garden promoted drawable used a hard cohort");
            }
            if !render_candidate.prepared
                || render_candidate.temporal_mode != Some(LodTemporalTransitionMode::Morphing)
            {
                return None;
            }
            let allow_unevaluated_late_authored = !retained
                && !render_candidate.active
                && !render_candidate.transitioning;
            let blend = observe_garden_view_blend_with_render_state(
                &render_candidate,
                render,
                !retained,
                "interactive Garden",
            );
            blend.assert_dynamic_coherent("interactive Garden");
            blend.assert_active_dynamic_evaluation_complete("interactive Garden");
            assert_eq!(
                blend.status.invalid_pressure_count, 0,
                "interactive canonical Garden encountered invalid active pressure edges"
            );
            blend.assert_no_invalid_pressure_pairs("interactive Garden");
            blend.assert_manifest_edge_topology(&state.node_parents, "interactive Garden");
            if !drawable_unchanged {
                if let Some(previous) = state.last_active_blend.clone() {
                state.authored_publication_hold.assert_recovery_slew_from(
                    &blend,
                    &previous,
                    !retained,
                    "interactive Garden recovery",
                );
                let evidence = blend.assert_dynamic_frame_transition(
                    &previous,
                    &state.authored_publication_hold.recovery_edges,
                    &state.authored_publication_hold.pending_ordinary_edges,
                    &state.node_parents,
                    !retained,
                    allow_unevaluated_late_authored,
                    "interactive Garden transition",
                );
                state
                    .authored_publication_hold
                    .observe(&blend, &evidence, "interactive Garden");
                state.fractional_overlap_replacements = state
                    .fractional_overlap_replacements
                    .saturating_add(u32::from(evidence.preserved_fractional_overlap));
                } else if !blend.desired_evaluation_complete {
                    let unevaluated_late_edge_keys = blend
                        .edges
                        .iter()
                        .filter(|edge| {
                            edge.activation_requires_slew && edge.evaluation_weight_bits.is_none()
                        })
                        .map(|edge| edge.key.clone())
                        .collect::<BTreeSet<_>>();
                    assert!(
                        allow_unevaluated_late_authored
                            && !unevaluated_late_edge_keys.is_empty()
                            && blend.evaluation_view.is_none()
                            && blend.evaluation_target.is_none(),
                        "interactive Garden first incomplete blend was not the exact PREPARED late-edge authored publication"
                    );
                    for edge in &blend.edges {
                        assert_eq!(
                            (edge.displayed_weight_bits, edge.desired_weight_bits),
                            (edge.initial_weight_bits, edge.initial_weight_bits),
                            "interactive Garden first PREPARED authored edge left its retained endpoint: {:?}",
                            edge.key,
                        );
                        if unevaluated_late_edge_keys.contains(&edge.key) {
                            assert!(
                                edge.recovery_lag,
                                "interactive Garden first PREPARED late edge omitted recovery provenance: {:?}",
                                edge.key,
                            );
                        }
                    }
                    assert_eq!(
                        (
                            blend.upload.last_max_delta.to_bits(),
                            blend.upload.last_weighted_record_energy.to_bits(),
                        ),
                        (0.0_f32.to_bits(), 0.0_f64.to_bits()),
                        "interactive Garden first PREPARED late publication reported displayed-weight work"
                    );
                    let evidence = GardenViewBlendActivationEvidence {
                        activation_frame: true,
                        new_authored_publication: true,
                        preserved_fractional_overlap: false,
                        new_edge_keys: blend.edges.iter().map(|edge| edge.key.clone()).collect(),
                        unevaluated_late_edge_keys,
                    };
                    state.authored_publication_hold.observe(
                        &blend,
                        &evidence,
                        "interactive Garden first PREPARED late publication",
                    );
                }
            }
            state.last_active_blend = Some(blend.clone());
            (!retained).then_some(blend)
        });
        if let Some(physical_drawable) = physical_drawable {
            state.last_physical_drawable = Some(physical_drawable);
        }
        let observable = render_candidate.and_then(|candidate| {
            garden_observable_logical_cut(status, candidate)
                .map(|signature| (signature, candidate.rendered_quality_status()))
        });
        if observable.is_some() && state.first_drawable_frame.is_none() {
            state.first_drawable_frame = Some(state.total_frames);
            eprintln!(
                "Garden first drawable cut at app frame {} ({:.3}s)",
                state.total_frames,
                garden_frames_to_seconds(state.total_frames)
            );
        }

        let requested_target = state.settings.quality_target();
        let unified_status = lod_statuses.get(cloud).ok();
        let request_is_fresh = match (
            unified_status,
            observable.as_ref(),
            render_candidate,
            blend_observation.as_ref(),
        ) {
            (Some(public), Some((_, rendered)), Some(candidate), Some(blend)) => {
                public.failure.is_none()
                    && public.source == GaussianLodSourceKind::Package
                    && public.lifecycle == GaussianLodLifecycle::Active
                    && public.selection_mode == LodSelectionMode::Dynamic
                    && public.frozen_views == 0
                    && public.resident_pages == status.resident_pages
                    && public.requested_target == requested_target
                    && public.target_satisfied == Some(true)
                    && public.degradation == LodDegradation::None
                    && public.temporal_transition_mode.is_none()
                    && public.active_views == 1
                    && public.selected_gaussians == rendered.active_gaussians
                    && public.submitted_candidates == candidate.rendered_candidate_count()
                    && public.view_blend_edges == blend.status.edge_count
                    && public.view_blend_lagging_edges == 0
                    && public.view_blend_invalid_pressure_evaluations == 0
                    && public.view_blend_missing_consumers == 0
                    && public.view_blend_max_lag.to_bits() == 0.0_f32.to_bits()
                    && public.view_blend_max_delta.to_bits() == blend.status.max_delta_bits
                    && public.view_blend_weighted_record_energy.to_bits()
                        == blend.status.weighted_record_energy_bits
                    && public.achieved_max_error_px == Some(rendered.achieved_max_error_px)
                    && public.achieved_max_target_ratio == Some(rendered.achieved_max_target_ratio)
                    && rendered.requested_target == requested_target
                    && rendered.degradation == LodDegradation::None
                    && rendered.requested_pages == 0
                    && rendered.active_gaussians == status.active_gaussians
                    && render_commit_identity
                        == Some(candidate.render_commit_identity_for_testing())
                    && garden_physical_range_signature(candidate.frontier().physical_ranges())
                        == blend.target_ranges
                    && garden_physical_range_signature(candidate.render_ranges())
                        == blend.presentation_ranges
                    && garden_physical_range_signature(
                        candidate.required_atlas_ranges_for_testing(),
                    ) == blend.required_ranges
                    && blend.desired_evaluation_complete
                    && blend.evaluation_view == Some(blend.current_render_view)
                    && blend.evaluation_target == Some(blend.current_render_target)
                    && blend.status.lagging_count == 0
                    && blend.status.invalid_pressure_count == 0
                    && blend.status.missing_consumer_count == 0
                    && blend
                        .edges
                        .iter()
                        .all(|edge| edge.displayed_weight_bits == edge.desired_weight_bits)
                    && candidate.render_ranges().iter().all(|range| {
                        candidate
                            .required_atlas_ranges_for_testing()
                            .contains(range)
                    })
                    && candidate.frontier().physical_ranges().iter().all(|range| {
                        candidate
                            .required_atlas_ranges_for_testing()
                            .contains(range)
                    })
            }
            _ => false,
        };
        if let Some((signature, rendered_quality)) = observable.as_ref() {
            if !request_is_fresh {
                state.stale_visible_frames += 1;
                if status.phase == GaussianLodPackagePhase::Active {
                    state.active_while_stale_frames += 1;
                }
            }
            if state.request_origin_signature.as_ref() == Some(signature) {
                state.retained_origin_frames += 1;
            }
            if state.last_observed_signature.as_ref() != Some(signature) {
                state.request_cut_changes += 1;
                assert!(
                    state.request_seen_signatures.insert(signature.clone()),
                    "ABI-16 Garden revisited an ACTIVE logical endpoint within one request: scenario={:?}, changes={}, frame={}, signature={signature:?}",
                    state.scenario(),
                    state.request_cut_changes,
                    state.request_frames,
                );
                let cold_initial_publication = if state.request_origin_signature.is_none() {
                    1
                } else {
                    0
                };
                let topology_changes = state
                    .request_cut_changes
                    .saturating_sub(cold_initial_publication);
                eprintln!(
                    "Garden {:?} ACTIVE cut {} ({} topology changes) at request frame {} ({:.3}s): active={}, ranges={}, signature_digest={:#018x}, resident={}, blend_edges={}, lagging_edges={}, max_lag={}, rendered_target={:?}, requested_target={:?}, degradation={:?}, achieved_ratio={}, target_satisfied={:?}, fixed_point={}, lifecycle={:?}",
                    state.scenario(),
                    state.request_cut_changes,
                    topology_changes,
                    state.request_frames,
                    garden_frames_to_seconds(state.request_frames),
                    signature.0,
                    signature.1.len(),
                    garden_logical_cut_digest(signature),
                    status.resident_pages,
                    blend_observation
                        .as_ref()
                        .map_or(0, |blend| blend.status.edge_count),
                    blend_observation
                        .as_ref()
                        .map_or(0, |blend| blend.status.lagging_count),
                    blend_observation
                        .as_ref()
                        .map_or(0.0, |blend| { f32::from_bits(blend.status.max_lag_bits) }),
                    rendered_quality.requested_target,
                    requested_target,
                    rendered_quality.degradation,
                    rendered_quality.achieved_max_target_ratio,
                    unified_status.and_then(|status| status.target_satisfied),
                    request_is_fresh,
                    state.lifecycle,
                );
                state.last_observed_signature = Some(signature.clone());
                state.stable_frames = 0;
            }
        }

        let queued_uploads = upload_queue.queued_slot_count();
        let resident_plateau = state.last_resident_pages == Some(status.resident_pages);
        if !resident_plateau {
            state.last_resident_change_request_frame = state.request_frames;
        }
        state.last_resident_pages = Some(status.resident_pages);
        let blend_signature = blend_observation
            .as_ref()
            .map(GardenViewBlendObservation::presentation_signature);
        let blend_upload = blend_observation.as_ref().map(|blend| blend.upload);
        let blend_plateau = state.last_blend_signature == blend_signature
            && state.last_blend_upload == blend_upload
            && blend_signature.is_some();
        if cleanup_observed {
            state.last_blend_signature = blend_signature.clone();
            state.last_blend_upload = blend_upload;
        }
        let target_signature = observable
            .as_ref()
            .and_then(|(signature, rendered_quality)| {
                (status.phase == GaussianLodPackagePhase::Active
                    && rendered_quality.requested_target == requested_target
                    && request_is_fresh)
                    .then_some(signature)
            });
        if state.phase == GardenInteractivePhase::PackageWaiting
            && state.request_frames >= 2
            && cleanup_observed
            && resident_plateau
            && blend_plateau
            && queued_uploads == 0
            && target_signature.is_some()
        {
            state.stable_frames += 1;
        } else if state.phase == GardenInteractivePhase::PackageWaiting && cleanup_observed {
            state.stable_frames = 0;
        }

        if state.phase == GardenInteractivePhase::PackageWaiting
            && state.scenario_index > 0
            && state.pending_capture.is_none()
            && (state.request_frames == 1
                || state
                    .request_frames
                    .is_multiple_of(GARDEN_INTERACTIVE_TRANSITION_CAPTURE_INTERVAL))
        {
            let scenario = state.scenario();
            commands.spawn(Screenshot::image(
                state.target.clone().expect("Garden render target exists"),
            ));
            state.pending_capture = Some(GardenInteractiveCapture::Transition(scenario));
        }

        if state.phase == GardenInteractivePhase::PackageWaiting
            && state.stable_frames >= GARDEN_INTERACTIVE_STABLE_FRAMES
            && state.pending_capture.is_none()
        {
            let signature = target_signature
                .expect("stable Garden target has an observable logical signature")
                .clone();
            assert_eq!(
                state.request_bounded_hard_frames,
                0,
                "authenticated ABI-16 Garden used a hard cohort during {:?}",
                state.scenario(),
            );
            let settled_blend = blend_observation
                .as_ref()
                .expect("stable Garden target has a coherent view-blend observation");
            state
                .authored_publication_hold
                .assert_no_pending_incomplete_publication(&format!(
                    "interactive Garden {:?} fixed point",
                    state.scenario(),
                ));
            settled_blend.assert_stationary_fixed_point(&format!(
                "interactive Garden {:?} fixed point",
                state.scenario()
            ));
            assert!(
                settled_blend
                    .presentation_ranges
                    .iter()
                    .all(|range| settled_blend.required_ranges.contains(range)),
                "interactive Garden {:?} presentation escaped its generation-safe required union",
                state.scenario(),
            );
            if settled_blend.edges.iter().any(|edge| edge.endpoint == 0) {
                assert!(
                    settled_blend
                        .required_ranges
                        .iter()
                        .any(|range| !settled_blend.presentation_ranges.contains(range)),
                    "interactive Garden {:?} fractional presentation did not retain an additional endpoint range",
                    state.scenario(),
                );
            }
            state.quiescent_signature = Some(signature);
            state.quiescent_blend_signature = Some(settled_blend.presentation_signature());
            state.quiescent_blend_upload = Some(settled_blend.upload);
            state.quiescent_quality_status = Some(
                observable
                    .as_ref()
                    .expect("stable Garden target has observable quality status")
                    .1,
            );
            state.quiescent_target_satisfied =
                unified_status.and_then(|status| status.target_satisfied);
            state.quiescent_resident_pages = Some(status.resident_pages);
            let scenario = state.scenario();
            commands.spawn(Screenshot::image(
                state.target.clone().expect("Garden render target exists"),
            ));
            state.pending_capture = Some(GardenInteractiveCapture::SettledFirst(scenario));
            state.phase = GardenInteractivePhase::PackageFirstPending;
            state.phase_frames = 0;
            return;
        }

        if matches!(
            state.phase,
            GardenInteractivePhase::PackageFirstPending
                | GardenInteractivePhase::PackageSettledGap
                | GardenInteractivePhase::PackageSecondPending
        ) {
            let signature = target_signature
                .expect("quiescent Garden cut remained observable during settled sampling");
            assert_eq!(
                Some(signature),
                state.quiescent_signature.as_ref(),
                "Garden logical cut changed during settled sampling of {:?}",
                state.scenario()
            );
            assert_eq!(
                Some(status.resident_pages),
                state.quiescent_resident_pages,
                "Garden residency changed during settled sampling of {:?}",
                state.scenario()
            );
            assert_eq!(
                queued_uploads,
                0,
                "Garden queued uploads during settled sampling of {:?}",
                state.scenario()
            );
            assert_eq!(
                blend_signature.as_ref(),
                state.quiescent_blend_signature.as_ref(),
                "Garden fractional topology/weights changed during settled sampling of {:?}",
                state.scenario(),
            );
            assert_eq!(
                blend_upload,
                state.quiescent_blend_upload,
                "Garden fractional resources changed during settled sampling of {:?}",
                state.scenario(),
            );
        }

        if state.phase == GardenInteractivePhase::PackageSettledGap
            && state.phase_frames >= GARDEN_INTERACTIVE_CAPTURE_GAP_FRAMES
            && state.pending_capture.is_none()
        {
            let scenario = state.scenario();
            commands.spawn(Screenshot::image(
                state.target.clone().expect("Garden render target exists"),
            ));
            state.pending_capture = Some(GardenInteractiveCapture::SettledSecond(scenario));
            state.phase = GardenInteractivePhase::PackageSecondPending;
            state.phase_frames = 0;
        }
    }

    fn garden_observable_logical_cut(
        status: &GaussianLodPackageStatus,
        candidate: &bevy_gaussian_splatting::stream::render_commit::LodRenderCandidate,
    ) -> Option<GardenLogicalCutSignature> {
        // A PREPARED/TRANSITIONING candidate may expose its immutable target
        // frontier before the renderer has made that endpoint visible. Endpoint
        // and direction evidence therefore begins only at ACTIVE.
        if status.phase != GaussianLodPackagePhase::Active
            || !candidate.render_is_active_for_testing()
            || u64::from(candidate.rendered_candidate_count()) != status.active_gaussians
            || candidate.rendered_quality_status().active_gaussians != status.active_gaussians
        {
            return None;
        }
        let ranges = candidate
            .render_ranges()
            .iter()
            .map(|range| (range.node.0, range.page.0, range.count))
            .collect::<Vec<_>>();
        assert_eq!(
            ranges
                .iter()
                .map(|(_, _, count)| u64::from(*count))
                .sum::<u64>(),
            status.active_gaussians,
            "Garden published incoherent logical ranges"
        );
        Some((status.active_gaussians, ranges))
    }

    fn garden_frames_to_seconds(frames: u32) -> f64 {
        f64::from(frames) / 120.0
    }

    fn garden_temporal_difference(
        previous: &[[f32; 4]],
        current: &[[f32; 4]],
    ) -> GardenTemporalDifference {
        assert_eq!(previous.len(), current.len());
        let mut full_squared = 0.0_f64;
        let mut foreground_squared = 0.0_f64;
        let mut foreground_pixels = 0_usize;
        let mut max_abs_rgb = 0.0_f64;
        for (previous, current) in previous.iter().zip(current) {
            let foreground = linear_luminance(previous) >= GARDEN_ORACLE_FOREGROUND_LUMINANCE
                || linear_luminance(current) >= GARDEN_ORACLE_FOREGROUND_LUMINANCE;
            let mut pixel_squared = 0.0_f64;
            for channel in 0..3 {
                let difference = f64::from(current[channel] - previous[channel]);
                pixel_squared += difference * difference;
                max_abs_rgb = max_abs_rgb.max(difference.abs());
            }
            full_squared += pixel_squared;
            if foreground {
                foreground_squared += pixel_squared;
                foreground_pixels += 1;
            }
        }
        GardenTemporalDifference {
            full_rgb_rms: (full_squared / (previous.len().max(1) * 3) as f64).sqrt(),
            foreground_rgb_rms: (foreground_squared / (foreground_pixels.max(1) * 3) as f64).sqrt(),
            max_abs_rgb,
        }
    }

    fn linear_luminance(pixel: &[f32; 4]) -> f64 {
        garden_luminance(*pixel)
    }

    fn garden_temporal_second_difference(
        previous_previous: &[[f32; 4]],
        previous: &[[f32; 4]],
        current: &[[f32; 4]],
    ) -> GardenTemporalDifference {
        assert_eq!(previous_previous.len(), previous.len());
        assert_eq!(previous.len(), current.len());
        let mut full_squared = 0.0_f64;
        let mut foreground_squared = 0.0_f64;
        let mut foreground_pixels = 0_usize;
        let mut max_abs_rgb = 0.0_f64;
        for ((previous_previous, previous), current) in
            previous_previous.iter().zip(previous).zip(current)
        {
            let foreground = linear_luminance(previous_previous)
                >= GARDEN_ORACLE_FOREGROUND_LUMINANCE
                || linear_luminance(previous) >= GARDEN_ORACLE_FOREGROUND_LUMINANCE
                || linear_luminance(current) >= GARDEN_ORACLE_FOREGROUND_LUMINANCE;
            let mut pixel_squared = 0.0_f64;
            for channel in 0..3 {
                let difference = f64::from(
                    current[channel] - 2.0 * previous[channel] + previous_previous[channel],
                );
                pixel_squared += difference * difference;
                max_abs_rgb = max_abs_rgb.max(difference.abs());
            }
            full_squared += pixel_squared;
            if foreground {
                foreground_squared += pixel_squared;
                foreground_pixels += 1;
            }
        }
        GardenTemporalDifference {
            full_rgb_rms: (full_squared / (previous.len().max(1) * 3) as f64).sqrt(),
            foreground_rgb_rms: (foreground_squared / (foreground_pixels.max(1) * 3) as f64).sqrt(),
            max_abs_rgb,
        }
    }

    fn assert_garden_dynamic_temporal_trace(
        direction: GardenTemporalDollyDirection,
        flat: &GardenTemporalTrace,
        frozen: &GardenTemporalTrace,
        dynamic: &GardenTemporalTrace,
    ) {
        assert_eq!(flat.initial_active_candidate_count, None);
        assert_eq!(flat.final_active_candidate_count, None);
        assert_eq!(
            flat.active_endpoint_changes
                + flat.blend_frames
                + flat.fractional_blend_frames
                + flat.lagging_blend_frames
                + flat.bounded_hard_frames,
            0,
            "flat Garden baseline cannot carry package transition telemetry"
        );
        assert_eq!(
            frozen.active_endpoint_changes
                + frozen.lagging_blend_frames
                + frozen.bounded_hard_frames,
            0,
            "Frozen-LoD baseline must remain one fixed cut throughout the identical dolly"
        );
        assert!(
            frozen.blend_frames > 0,
            "Frozen-LoD baseline did not retain an ACTIVE view-blend presentation"
        );
        assert_eq!(
            (
                frozen.immutable_table_uploads,
                frozen.weight_writes,
                frozen.buffer_allocations,
            ),
            (0, 0, 0),
            "Frozen-LoD changed its immutable table, weights, or allocation during camera motion"
        );
        assert_eq!(
            frozen.initial_active_candidate_count, frozen.final_active_candidate_count,
            "Frozen-LoD {direction:?} baseline changed candidate cardinality"
        );
        dynamic
            .initial_active_candidate_count
            .expect("Dynamic Garden trace publishes its initial ACTIVE candidate count");
        dynamic
            .final_active_candidate_count
            .expect("Dynamic Garden trace publishes its final ACTIVE candidate count");
        assert!(
            dynamic.active_endpoint_changes > 0,
            "Dynamic Garden {direction:?} dolly did not exercise camera-conditioned topology"
        );
        assert!(
            dynamic.blend_frames > 0 && dynamic.fractional_blend_frames > 0,
            "Dynamic ABI 16 Garden {direction:?} dolly did not expose fractional ACTIVE blending: {dynamic:?}"
        );
        assert_eq!(
            dynamic.buffer_allocations, 0,
            "prewarmed Dynamic Garden reallocated its view-blend buffer during the measured trajectory"
        );
        assert!(
            dynamic.weight_writes > 0,
            "Dynamic Garden camera motion never updated its per-edge weight suffix"
        );
        let initial = dynamic.initial_active_candidate_count.unwrap();
        let final_count = dynamic.final_active_candidate_count.unwrap();
        match direction {
            GardenTemporalDollyDirection::Refining => assert!(
                final_count >= initial,
                "refining Garden dolly ended coarser: {initial}->{final_count}"
            ),
            GardenTemporalDollyDirection::Coarsening => assert!(
                final_count <= initial,
                "coarsening Garden dolly ended finer: {initial}->{final_count}"
            ),
        }
        assert_eq!(
            dynamic.bounded_hard_frames, 0,
            "ABI 16 Garden unexpectedly fell back to a hard cohort on the qualified adapter"
        );
        assert_eq!(dynamic.deltas.len(), flat.deltas.len());
        assert_eq!(dynamic.deltas.len(), frozen.deltas.len());
        assert_eq!(
            dynamic.second_differences.len(),
            flat.second_differences.len()
        );
        assert_eq!(
            dynamic.second_differences.len(),
            frozen.second_differences.len()
        );
        for (index, ((dynamic, flat), frozen)) in dynamic
            .deltas
            .iter()
            .zip(&flat.deltas)
            .zip(&frozen.deltas)
            .enumerate()
        {
            let baseline_full = flat.full_rgb_rms.max(frozen.full_rgb_rms);
            let baseline_foreground = flat.foreground_rgb_rms.max(frozen.foreground_rgb_rms);
            let baseline_max = flat.max_abs_rgb.max(frozen.max_abs_rgb);
            assert!(
                dynamic.full_rgb_rms
                    <= baseline_full * GARDEN_TEMPORAL_DELTA_RATIO
                        + GARDEN_TEMPORAL_DELTA_RMS_FLOOR
                    && dynamic.foreground_rgb_rms
                        <= baseline_foreground * GARDEN_TEMPORAL_DELTA_RATIO
                            + GARDEN_TEMPORAL_DELTA_RMS_FLOOR
                    && dynamic.max_abs_rgb
                        <= baseline_max * GARDEN_TEMPORAL_DELTA_RATIO
                            + GARDEN_TEMPORAL_DELTA_MAX_FLOOR,
                "Garden one-frame temporal delta spike at dolly sample {}: dynamic={dynamic:?}, flat={flat:?}, frozen={frozen:?}",
                index + 1,
            );
        }
        for (index, ((dynamic, flat), frozen)) in dynamic
            .second_differences
            .iter()
            .zip(&flat.second_differences)
            .zip(&frozen.second_differences)
            .enumerate()
        {
            let baseline_full = flat.full_rgb_rms.max(frozen.full_rgb_rms);
            let baseline_foreground = flat.foreground_rgb_rms.max(frozen.foreground_rgb_rms);
            let baseline_max = flat.max_abs_rgb.max(frozen.max_abs_rgb);
            assert!(
                dynamic.full_rgb_rms
                    <= baseline_full * GARDEN_TEMPORAL_SECOND_RATIO
                        + GARDEN_TEMPORAL_SECOND_RMS_FLOOR
                    && dynamic.foreground_rgb_rms
                        <= baseline_foreground * GARDEN_TEMPORAL_SECOND_RATIO
                            + GARDEN_TEMPORAL_SECOND_RMS_FLOOR
                    && dynamic.max_abs_rgb
                        <= baseline_max * GARDEN_TEMPORAL_SECOND_RATIO
                            + GARDEN_TEMPORAL_SECOND_MAX_FLOOR,
                "Garden one-frame temporal second-difference spike at dolly sample {}: dynamic={dynamic:?}, flat={flat:?}, frozen={frozen:?}",
                index + 2,
            );
        }
    }

    fn garden_flat_oracle_poses() -> [GardenInteractivePose; 4] {
        [
            GardenInteractivePose::Overview,
            GardenInteractivePose::Closer,
            GardenInteractivePose::Farther,
            GardenInteractivePose::Orbit,
        ]
    }

    fn garden_package_scenario_for_pose(pose: GardenInteractivePose) -> GardenInteractiveScenario {
        match pose {
            GardenInteractivePose::Overview => GardenInteractiveScenario::NearHigh,
            GardenInteractivePose::Closer => GardenInteractiveScenario::CloserHigh,
            GardenInteractivePose::Farther => GardenInteractiveScenario::FartherHigh,
            GardenInteractivePose::Orbit => GardenInteractiveScenario::OrbitHigh,
        }
    }

    fn assert_garden_temporal_stability(first: &[[f32; 4]], second: &[[f32; 4]], context: &str) {
        let temporal = compare_linear_rgba(first, second, 0.5)
            .expect("Garden temporal image metrics are valid");
        let foreground_iou = garden_foreground_iou(first, second);
        assert!(
            temporal.psnr_rgb.is_infinite() || temporal.psnr_rgb >= 55.0,
            "Garden RGB changed during {context}: {temporal:?}"
        );
        assert!(
            temporal.foreground_psnr_rgb.is_infinite() || temporal.foreground_psnr_rgb >= 55.0,
            "Garden foreground changed during {context}: {temporal:?}"
        );
        assert!(
            temporal.luminance_ssim >= 0.9999,
            "Garden luminance changed during {context}: {temporal:?}"
        );
        assert!(
            foreground_iou >= 0.9999 && temporal.alpha_mae <= 0.0001,
            "Garden silhouette changed during {context}: iou={foreground_iou}, metrics={temporal:?}"
        );
        assert!(
            temporal.max_abs_error <= 0.02,
            "Garden pixels changed excessively during {context}: {temporal:?}"
        );
    }

    fn assert_garden_interactive_settlement(
        scenario: GardenInteractiveScenario,
        settlement: &GardenInteractiveSettlement,
        prior: &BTreeMap<GardenInteractiveScenario, GardenInteractiveSettlement>,
    ) {
        assert!(
            settlement.target_satisfied,
            "Garden {:?} settled without satisfying its canonical non-degraded target",
            scenario
        );
        let get = |scenario| {
            prior
                .get(&scenario)
                .unwrap_or_else(|| panic!("missing prior Garden settlement {scenario:?}"))
        };
        assert!(
            settlement.blend_edges > 0,
            "Garden {:?} settled without camera-conditioned blend edges",
            scenario,
        );
        assert!(
            settlement.fractional_blend_edges > 0,
            "Garden {:?} did not exercise a fractional fixed point",
            scenario,
        );
        assert_eq!(
            settlement.bounded_hard_frames, 0,
            "Garden {:?} used a categorical fallback",
            scenario,
        );
        match scenario {
            GardenInteractiveScenario::NearLowCold
            | GardenInteractiveScenario::NearHigh
            | GardenInteractiveScenario::CloserHigh
            | GardenInteractiveScenario::FartherHigh => {}
            GardenInteractiveScenario::NearHighRecovery => {
                assert_garden_reproduces_settlement(
                    settlement,
                    get(GardenInteractiveScenario::NearHigh),
                    "far-to-overview recovery",
                );
            }
            GardenInteractiveScenario::OrbitHigh => {}
            GardenInteractiveScenario::NearHighReturn => {
                assert_garden_reproduces_settlement(
                    settlement,
                    get(GardenInteractiveScenario::NearHigh),
                    "orbit-to-overview return",
                );
            }
            GardenInteractiveScenario::NearLowReturn => {
                assert_garden_reproduces_settlement(
                    settlement,
                    get(GardenInteractiveScenario::NearLowCold),
                    "quality-down return",
                );
            }
        }
    }

    fn assert_garden_reproduces_settlement(
        actual: &GardenInteractiveSettlement,
        expected: &GardenInteractiveSettlement,
        context: &str,
    ) {
        assert_eq!(
            actual.active_gaussians, expected.active_gaussians,
            "Garden active detail was not reversible during {context}"
        );
        assert_eq!(
            actual.signature, expected.signature,
            "Garden logical cut was not reversible during {context}"
        );
        assert_eq!(
            actual.quality_status, expected.quality_status,
            "Garden achieved quality status was not reversible during {context}"
        );
        assert_eq!(
            actual.blend_signature, expected.blend_signature,
            "Garden view-blend topology/weights were history-dependent during {context}"
        );
        assert_garden_temporal_stability(&expected.image, &actual.image, context);
    }

    #[allow(clippy::too_many_arguments)]
    fn on_garden_interactive_capture(
        trigger: On<ScreenshotCaptured>,
        mut commands: Commands,
        mut state: ResMut<GardenInteractiveState>,
        mut lod_settings: Query<&mut GaussianLodSettings>,
        mut camera_transforms: Query<&mut Transform, With<GaussianCamera>>,
        blend_probe: Res<GardenViewBlendRenderProbe>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let capture = state
            .pending_capture
            .take()
            .expect("interactive Garden capture has a registered purpose");
        if matches!(
            capture,
            GardenInteractiveCapture::SettledSecond(GardenInteractiveScenario::NearLowReturn)
        ) {
            // The request-frame screenshot already owns the final ordered
            // Cleanup evidence. Stop the producer before image/oracle work and
            // package retirement, when Last intentionally no longer drains it.
            blend_probe.finish_ordered_capture();
        }
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("Garden offscreen capture converts")
            .to_rgba8();
        assert_eq!(rgba.width(), GARDEN_TARGET_WIDTH);
        assert_eq!(rgba.height(), GARDEN_TARGET_HEIGHT);
        let linear = linear_rgba(rgba.as_raw());
        let sanity = garden_image_sanity(
            &linear,
            GARDEN_TARGET_WIDTH as usize,
            GARDEN_TARGET_HEIGHT as usize,
        );
        assert_garden_bounds_fit_image_nonblank(sanity);

        match capture {
            GardenInteractiveCapture::Transition(scenario) => {
                assert_eq!(state.scenario(), scenario);
                assert_eq!(state.phase, GardenInteractivePhase::PackageWaiting);
                state.transition_captures += 1;
            }
            GardenInteractiveCapture::SettledFirst(scenario) => {
                assert_eq!(state.scenario(), scenario);
                assert_eq!(state.phase, GardenInteractivePhase::PackageFirstPending);
                state.first_settled_image = Some(linear);
                state.phase = GardenInteractivePhase::PackageSettledGap;
                state.phase_frames = 0;
            }
            GardenInteractiveCapture::SettledSecond(scenario) => {
                assert_eq!(state.scenario(), scenario);
                assert_eq!(state.phase, GardenInteractivePhase::PackageSecondPending);
                let first = state
                    .first_settled_image
                    .take()
                    .expect("first settled Garden image exists");
                assert_garden_temporal_stability(
                    &first,
                    &linear,
                    &format!("settled sampling for {scenario:?}"),
                );
                if state.scenario_index > 0 {
                    assert!(
                        state.transition_captures > 0,
                        "Garden transition {:?} reached settlement without a retained-output capture",
                        scenario
                    );
                }
                let blend_signature = state
                    .quiescent_blend_signature
                    .clone()
                    .expect("quiescent Garden blend signature exists");
                let fractional_blend_edges = blend_signature
                    .edges
                    .iter()
                    .filter(|(_, endpoint, _, _)| *endpoint == 0)
                    .count()
                    .try_into()
                    .expect("Garden fractional edge count fits u32");
                let settlement = GardenInteractiveSettlement {
                    active_gaussians: state
                        .quiescent_signature
                        .as_ref()
                        .expect("quiescent Garden signature exists")
                        .0,
                    signature: state
                        .quiescent_signature
                        .clone()
                        .expect("quiescent Garden signature exists"),
                    image: first,
                    start_frame: state.scenario_start_frame,
                    settled_frame: state.total_frames,
                    request_frames: state.request_frames,
                    cut_changes: state.request_cut_changes,
                    blend_edges: blend_signature
                        .edges
                        .len()
                        .try_into()
                        .expect("Garden blend edge count fits u32"),
                    fractional_blend_edges,
                    blend_signature,
                    blend_upload: state
                        .quiescent_blend_upload
                        .expect("quiescent Garden blend upload exists"),
                    bounded_hard_frames: state.request_bounded_hard_frames,
                    transition_captures: state.transition_captures,
                    stale_visible_frames: state.stale_visible_frames,
                    active_while_stale_frames: state.active_while_stale_frames,
                    retained_origin_frames: state.retained_origin_frames,
                    lifecycle: state.lifecycle.clone(),
                    quality_status: state
                        .quiescent_quality_status
                        .expect("quiescent Garden quality status exists"),
                    target_satisfied: state.quiescent_target_satisfied == Some(true),
                };
                assert_garden_interactive_settlement(scenario, &settlement, &state.settlements);
                eprintln!(
                    "Garden {:?} settled at app frame {} ({:.3}s total), {} request frames ({:.3}s): active={}, ranges={}, signature_digest={:#018x}, ACTIVE_cuts={}, blend_edges={}, fractional_edges={}, blend_upload={:?}, bounded_hard_frames={}, transition_captures={}, stale_visible={}, active_while_stale={}, retained_origin={}, lifecycle={:?}, degradation={:?}, achieved_ratio={}, target_satisfied={}, sanity={sanity:?}",
                    scenario,
                    settlement.settled_frame,
                    garden_frames_to_seconds(settlement.settled_frame),
                    settlement.request_frames,
                    garden_frames_to_seconds(settlement.request_frames),
                    settlement.active_gaussians,
                    settlement.signature.1.len(),
                    garden_logical_cut_digest(&settlement.signature),
                    settlement.cut_changes,
                    settlement.blend_edges,
                    settlement.fractional_blend_edges,
                    settlement.blend_upload,
                    settlement.bounded_hard_frames,
                    settlement.transition_captures,
                    settlement.stale_visible_frames,
                    settlement.active_while_stale_frames,
                    settlement.retained_origin_frames,
                    settlement.lifecycle,
                    settlement.quality_status.degradation,
                    settlement.quality_status.achieved_max_target_ratio,
                    settlement.target_satisfied,
                );
                assert_eq!(
                    settlement.start_frame + settlement.request_frames,
                    settlement.settled_frame,
                    "Garden request-frame accounting drifted"
                );
                state.settlements.insert(scenario, settlement);

                let next = state.scenario_index + 1;
                if next == GardenInteractiveScenario::ALL.len() {
                    assert!(
                        state.fractional_overlap_replacements > 0,
                        "interactive Garden never preserved a fractional common edge while admitting a disjoint new edge"
                    );
                    eprintln!(
                        "Garden interactive authored publications: distinct={}, max_consecutive_hold_frames={}",
                        state.authored_publication_hold.distinct_publications,
                        state.authored_publication_hold.max_consecutive_frames,
                    );
                    let package = state.cloud.take().expect("Garden package entity exists");
                    commands.entity(package).despawn();
                    state.manifest = None;
                    state.phase = GardenInteractivePhase::RetiringPackage;
                    state.phase_frames = 0;
                } else {
                    state.begin_scenario(next);
                    let cloud = state.cloud.expect("Garden package entity exists");
                    lod_settings
                        .get_mut(cloud)
                        .expect("Garden package LoD settings exist")
                        .quality = state.settings.quality;
                    let camera = state.camera.expect("Garden camera exists");
                    *camera_transforms
                        .get_mut(camera)
                        .expect("Garden camera transform exists") = state
                        .scene_frame
                        .expect("Garden scene frame exists")
                        .transform(state.scenario().pose());
                    eprintln!(
                        "Garden requested {:?} at app frame {} ({:.3}s): quality={}, transform={:?}",
                        state.scenario(),
                        state.total_frames,
                        garden_frames_to_seconds(state.total_frames),
                        state.settings.quality,
                        state
                            .scene_frame
                            .expect("Garden scene frame exists")
                            .transform(state.scenario().pose()),
                    );
                }
            }
            GardenInteractiveCapture::Flat(pose) => {
                assert_eq!(state.phase, GardenInteractivePhase::FlatPending);
                let scenario = garden_package_scenario_for_pose(pose);
                let package = &state
                    .settlements
                    .get(&scenario)
                    .unwrap_or_else(|| panic!("missing Garden package oracle for {pose:?}"))
                    .image;
                let spatial = garden_spatial_metrics(&linear, package);
                assert_garden_spatial_fidelity(spatial);
                eprintln!(
                    "Garden interactive flat-source oracle passed for {pose:?} at app frame {}: metrics={spatial:?}, flat={sanity:?}",
                    state.total_frames
                );
                state.flat_pose_index += 1;
                if state.flat_pose_index == garden_flat_oracle_poses().len() {
                    exit.write(AppExit::Success);
                } else {
                    let next_pose = garden_flat_oracle_poses()[state.flat_pose_index];
                    let camera = state.camera.expect("Garden camera exists");
                    *camera_transforms
                        .get_mut(camera)
                        .expect("Garden camera transform exists") = state
                        .scene_frame
                        .expect("Garden scene frame exists")
                        .transform(next_pose);
                    state.phase = GardenInteractivePhase::FlatWarmup;
                    state.phase_frames = 0;
                }
            }
        }
    }

    fn garden_image_sanity(pixels: &[[f32; 4]], width: usize, height: usize) -> GardenImageSanity {
        assert_eq!(pixels.len(), width * height);
        const MACRO_TILE_COLUMNS: usize = 8;
        const MACRO_TILE_ROWS: usize = 8;

        let mut foreground_pixels = 0_usize;
        let mut luminance_sum = 0.0_f64;
        let mut luminance_squared_sum = 0.0_f64;
        let mut min_luminance = f64::INFINITY;
        let mut max_luminance = f64::NEG_INFINITY;
        let mut min_x = width;
        let mut max_x = 0_usize;
        let mut min_y = height;
        let mut max_y = 0_usize;
        let mut occupied_tiles = [false; MACRO_TILE_COLUMNS * MACRO_TILE_ROWS];

        for (index, pixel) in pixels.iter().copied().enumerate() {
            let luminance = 0.2126 * f64::from(pixel[0])
                + 0.7152 * f64::from(pixel[1])
                + 0.0722 * f64::from(pixel[2]);
            // The offscreen target clears to opaque black, so alpha cannot
            // distinguish the scene from its background. Classify foreground
            // from linear RGB energy instead.
            if luminance <= GARDEN_FOREGROUND_LUMINANCE {
                continue;
            }
            let x = index % width;
            let y = index / width;
            foreground_pixels += 1;
            luminance_sum += luminance;
            luminance_squared_sum += luminance * luminance;
            min_luminance = min_luminance.min(luminance);
            max_luminance = max_luminance.max(luminance);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            let tile_x = x * MACRO_TILE_COLUMNS / width;
            let tile_y = y * MACRO_TILE_ROWS / height;
            occupied_tiles[tile_y * MACRO_TILE_COLUMNS + tile_x] = true;
        }

        let total_pixels = pixels.len() as f64;
        let foreground_denominator = foreground_pixels.max(1) as f64;
        let foreground_mean_luminance = luminance_sum / foreground_denominator;
        let foreground_luminance_variance = luminance_squared_sum / foreground_denominator
            - foreground_mean_luminance * foreground_mean_luminance;
        let has_foreground = foreground_pixels != 0;
        GardenImageSanity {
            foreground_pixels,
            foreground_fraction: foreground_pixels as f64 / total_pixels,
            foreground_mean_luminance,
            foreground_luminance_stddev: foreground_luminance_variance.max(0.0).sqrt(),
            foreground_luminance_range: if has_foreground {
                max_luminance - min_luminance
            } else {
                0.0
            },
            occupied_macro_tiles: occupied_tiles
                .into_iter()
                .filter(|occupied| *occupied)
                .count(),
            horizontal_extent: if has_foreground {
                (max_x - min_x + 1) as f64 / width as f64
            } else {
                0.0
            },
            vertical_extent: if has_foreground {
                (max_y - min_y + 1) as f64 / height as f64
            } else {
                0.0
            },
        }
    }

    fn assert_garden_bounds_fit_image_nonblank(sanity: GardenImageSanity) {
        // Manifest support bounds are deliberately conservative, so a
        // bounds-fit overview need not satisfy the close-up static oracle's
        // one-percent coverage requirement. Bounds-fit captures still require
        // a clearly drawable, spatially nondegenerate image here; their
        // flat-source or temporal oracles remain responsible for fidelity and
        // retain their fixed PSNR/SSIM/IoU thresholds.
        assert!(
            sanity.foreground_pixels >= 1_024,
            "bounds-fit Garden output became blank: {sanity:?}"
        );
        assert!(
            sanity.foreground_fraction.is_finite()
                && sanity.foreground_mean_luminance >= 0.005
                && sanity.foreground_mean_luminance <= 0.98,
            "bounds-fit Garden output has invalid luminance: {sanity:?}"
        );
        assert!(
            sanity.foreground_luminance_stddev >= 0.005
                && sanity.foreground_luminance_range >= 0.05,
            "bounds-fit Garden output is spatially flat: {sanity:?}"
        );
        assert!(
            sanity.occupied_macro_tiles >= 4
                && sanity.horizontal_extent >= 0.05
                && sanity.vertical_extent >= 0.05,
            "bounds-fit Garden output is implausibly localized: {sanity:?}"
        );
    }

    fn garden_foreground_iou(first: &[[f32; 4]], second: &[[f32; 4]]) -> f64 {
        garden_foreground_iou_at(first, second, GARDEN_FOREGROUND_LUMINANCE)
    }

    fn garden_foreground_iou_at(first: &[[f32; 4]], second: &[[f32; 4]], threshold: f64) -> f64 {
        assert_eq!(first.len(), second.len());
        let mut intersection = 0_usize;
        let mut union = 0_usize;
        for (first, second) in first.iter().zip(second) {
            let foreground = |pixel: &[f32; 4]| {
                0.2126 * f64::from(pixel[0])
                    + 0.7152 * f64::from(pixel[1])
                    + 0.0722 * f64::from(pixel[2])
                    > threshold
            };
            let first_foreground = foreground(first);
            let second_foreground = foreground(second);
            intersection += usize::from(first_foreground && second_foreground);
            union += usize::from(first_foreground || second_foreground);
        }
        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn garden_spatial_metrics(
        flat_reference: &[[f32; 4]],
        package_candidate: &[[f32; 4]],
    ) -> GardenSpatialMetrics {
        assert_eq!(flat_reference.len(), package_candidate.len());
        let full_frame = compare_linear_rgba(flat_reference, package_candidate, 0.5)
            .expect("Garden flat/package image metrics are valid");
        let mut foreground_rgb_squared_error = 0.0_f64;
        let mut foreground_rgb_absolute_error = 0.0_f64;
        let mut signed_luminance_error = 0.0_f64;
        let mut foreground_union_pixels = 0_usize;

        for (reference, candidate) in flat_reference.iter().zip(package_candidate) {
            let reference_luminance = garden_luminance(*reference);
            let candidate_luminance = garden_luminance(*candidate);
            if reference_luminance <= GARDEN_ORACLE_FOREGROUND_LUMINANCE
                && candidate_luminance <= GARDEN_ORACLE_FOREGROUND_LUMINANCE
            {
                continue;
            }
            foreground_union_pixels += 1;
            signed_luminance_error += candidate_luminance - reference_luminance;
            for channel in 0..3 {
                let error = f64::from(candidate[channel] - reference[channel]);
                foreground_rgb_squared_error += error * error;
                foreground_rgb_absolute_error += error.abs();
            }
        }

        let foreground_rgb_samples = foreground_union_pixels
            .checked_mul(3)
            .expect("Garden target RGB sample count fits usize")
            .max(1) as f64;
        let foreground_rgb_mse = foreground_rgb_squared_error / foreground_rgb_samples;
        GardenSpatialMetrics {
            full_frame_psnr_rgb: full_frame.psnr_rgb,
            foreground_psnr_rgb: if foreground_rgb_mse == 0.0 {
                f64::INFINITY
            } else {
                10.0 * (1.0 / foreground_rgb_mse).log10()
            },
            luminance_ssim: full_frame.luminance_ssim,
            foreground_iou: garden_foreground_iou_at(
                flat_reference,
                package_candidate,
                GARDEN_ORACLE_FOREGROUND_LUMINANCE,
            ),
            foreground_rgb_mae: foreground_rgb_absolute_error / foreground_rgb_samples,
            foreground_luminance_bias: (signed_luminance_error
                / foreground_union_pixels.max(1) as f64)
                .abs(),
            foreground_union_pixels,
        }
    }

    fn assert_garden_spatial_fidelity(metrics: GardenSpatialMetrics) {
        assert!(
            metrics.foreground_union_pixels > 0,
            "Garden flat/package oracle found no luminance-defined foreground: {metrics:?}"
        );
        assert!(
            metrics.full_frame_psnr_rgb >= GARDEN_MIN_FULL_FRAME_PSNR,
            "Garden package diverged from flat-source color energy: {metrics:?}"
        );
        assert!(
            metrics.foreground_psnr_rgb >= GARDEN_MIN_FOREGROUND_PSNR,
            "Garden package foreground color diverged from the flat source: {metrics:?}"
        );
        assert!(
            metrics.luminance_ssim >= GARDEN_MIN_LUMINANCE_SSIM,
            "Garden package luminance structure diverged from the flat source: {metrics:?}"
        );
        assert!(
            metrics.foreground_iou >= GARDEN_MIN_FOREGROUND_IOU,
            "Garden package silhouette diverged from the flat source: {metrics:?}"
        );
        assert!(
            metrics.foreground_rgb_mae <= GARDEN_MAX_FOREGROUND_RGB_MAE,
            "Garden package foreground has excessive mean color error: {metrics:?}"
        );
        assert!(
            metrics.foreground_luminance_bias <= GARDEN_MAX_FOREGROUND_LUMINANCE_BIAS,
            "Garden package has excessive signed foreground luminance bias: {metrics:?}"
        );
    }

    fn garden_luminance(pixel: [f32; 4]) -> f64 {
        0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct IndirectObservation {
        candidate_count: u32,
        args: LodIndirectArgs,
        expected: LodIndirectArgs,
    }

    #[derive(Clone, Debug, Default)]
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
                            defines.radix_base * defines.entries_per_invocation_a,
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
                    &state,
                    cloud,
                    camera,
                    expected,
                    &statuses,
                    &candidates,
                    &handles,
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
                    &state,
                    cloud,
                    camera,
                    state.expected_near_probe,
                    &statuses,
                    &candidates,
                    &handles,
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
                    &state,
                    cloud,
                    camera,
                    state.expected_far_probe,
                    &statuses,
                    &candidates,
                    &handles,
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
                    &state,
                    cloud,
                    camera,
                    state.expected_distant_probe,
                    &statuses,
                    &candidates,
                    &handles,
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
                    &state,
                    cloud,
                    camera,
                    state.expected_near_probe,
                    &statuses,
                    &candidates,
                    &handles,
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

    // Keeping the independent status, candidate, source-handle, and GPU-probe
    // inputs explicit makes this cross-world commit assertion auditable.
    #[allow(clippy::too_many_arguments)]
    fn observe_exact_active_cut(
        state: &QualityRenderState,
        cloud: Entity,
        camera: Entity,
        expected: u64,
        statuses: &Query<&GaussianLodBridgeStatus>,
        candidates: &Query<&LodRenderCandidates>,
        handles: &Query<&PlanarGaussian3dHandle>,
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

        let Ok(candidates) = candidates.get(cloud) else {
            // Dynamic-view handoff deliberately revokes every stale cut and
            // draws the immutable exact source while the replacement view is
            // debounced and selected. This is an ACTIVE render capability,
            // but it has no atlas-candidate payload to publish or probe. Only
            // accept that narrow contract; any other missing-candidate ACTIVE
            // state remains an immediate test failure.
            assert_eq!(
                status.active_gaussians, state.source_count,
                "ACTIVE without render candidates must report the exact retained source"
            );
            assert_source_handle_restored(state, cloud, handles);
            return false;
        };
        assert_eq!(candidates.len(), 1);
        let candidate = candidates
            .get(camera)
            .expect("active bridge publishes the perspective camera cut");
        assert!(
            !candidate.failed(),
            "pending LoD replacement failed while the retained cut remained ACTIVE"
        );
        if !candidate.render_is_active_for_testing() {
            return false;
        }
        let explicit_count = candidate.frontier().candidate_count() as u64;
        let range_count = candidate
            .frontier()
            .physical_ranges()
            .iter()
            .map(|range| u64::from(range.count))
            .sum::<u64>();
        assert_eq!(explicit_count, range_count);

        if status.active_gaussians == state.source_count {
            // A complete logical frontier may accompany the exact-source
            // interval for status/freeze provenance. Prove that it is the
            // retained-source draw rather than accepting an uncommitted atlas
            // cut merely because a candidate component happens to exist.
            assert_source_handle_restored(state, cloud, handles);
            assert_eq!(
                u64::from(candidate.rendered_candidate_count()),
                state.source_count
            );
            assert!(candidate.render_ranges().is_empty());
            return false;
        }

        // A retained ACTIVE cut and its WAITING/PREPARED replacement
        // intentionally overlap: status describes the cut still being drawn
        // while the candidate component carries the replacement for
        // render-world preparation. Wait for the bridge's next update to
        // publish one coherent committed snapshot before probing or asserting
        // its count.
        if !committed_cut_is_observable(
            candidate.render_is_active_for_testing(),
            status.active_gaussians,
            explicit_count,
            expected,
        ) {
            return false;
        }
        assert_eq!(status.active_gaussians, explicit_count);
        indirect_probe
            .request(u32::try_from(expected).expect("quality fixture count is representable"))
    }

    fn committed_cut_is_observable(
        candidate_active: bool,
        status: u64,
        candidate: u64,
        expected: u64,
    ) -> bool {
        candidate_active && status == expected && candidate == expected
    }

    #[test]
    fn exact_active_cut_waits_for_pending_replacement_commit() {
        assert!(!committed_cut_is_observable(false, 2, 2, 2));
        assert!(!committed_cut_is_observable(false, 1, 2, 2));
        assert!(!committed_cut_is_observable(true, 1, 2, 2));
        assert!(!committed_cut_is_observable(true, 1, 2, 1));
        assert!(!committed_cut_is_observable(true, 2, 3, 2));
        assert!(committed_cut_is_observable(true, 2, 2, 2));
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
        statuses: Query<&GaussianLodBridgeStatus>,
        candidates: Query<&LodRenderCandidates>,
        indirect_probe: Res<IndirectProbe>,
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
        let cloud = state.cloud.expect("cloud entity exists");
        let candidate_diagnostic = candidates.get(cloud).ok().and_then(|candidates| {
            let camera = state.camera?;
            let candidate = candidates.get(camera)?;
            Some((
                candidate.render_is_prepared(),
                candidate.render_is_transitioning_for_testing(),
                candidate.render_is_active_for_testing(),
                candidate.rendered_candidate_count(),
                candidate.frontier().candidate_count(),
                candidate.temporal_transition_mode(),
                candidate.temporal_transition_progress(),
            ))
        });
        let indirect_diagnostic = indirect_probe
            .0
            .lock()
            .expect("indirect probe mutex is not poisoned")
            .clone();
        assert!(
            foreground > 0,
            "LoD quality capture stayed empty: phase={:?}, status={:?}, candidate={:?}, indirect={:?}",
            state.phase,
            statuses.get(cloud).ok(),
            candidate_diagnostic,
            indirect_diagnostic,
        );

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

    struct NativePackageFixture {
        root: PathBuf,
        settings: GaussianLodSettings,
        expected_coarse: u64,
        expected_near: u64,
        expected_far: u64,
        expected_near_pages: usize,
        canonical_slot_bytes: u64,
        node_parents: GardenNodeParents,
    }

    impl NativePackageFixture {
        fn write() -> Self {
            let fixture = quality_fixture();
            let mut settings = fixture.settings.clone();
            settings.quality = 0.0;
            settings.budgets.max_resident_pages = 16;
            settings.budgets.max_pending_requests = 32;
            settings.budgets.max_requests_per_frame = 1;

            let mut lod = bevy_gaussian_splatting::build_planar_3d_lod(
                &fixture.cloud,
                fixture.build_settings,
            )
            .expect("native package quality hierarchy builds");
            // This synthetic upgrade qualifies package/render lifecycle only:
            // its ABI-14 representative payloads are not spatial-quality or
            // release evidence. Canonical Garden owns every visual oracle.
            lod.manifest = upgrade_manifest_to_synthetic_abi16_lifecycle_fixture(lod.manifest)
                .expect("native package lifecycle fixture upgrades to a validated ABI-16 map");
            assert_eq!(
                lod.manifest.build.builder_abi_version, 16,
                "native late-delivery qualification requires ABI-16 lifecycle semantics"
            );
            assert!(
                lod.manifest.morph_map.is_some(),
                "native late-delivery qualification cannot run without a morph map"
            );
            let hierarchy = ManifestLodHierarchy::new(&lod.manifest)
                .expect("native package quality manifest is valid");
            let coarse = select_frontier(
                &hierarchy,
                &AllResident,
                LodView::perspective(
                    Vec3::new(0.0, 0.0, NEAR_CAMERA_Z),
                    HEIGHT as f32,
                    VERTICAL_FOV,
                    NEAR_PLANE,
                ),
                &settings,
            )
            .expect("coarse package frontier is selectable");
            let mut near_settings = settings.clone();
            near_settings.quality = CAMERA_PROBE_QUALITY;
            let near = select_frontier(
                &hierarchy,
                &AllResident,
                LodView::perspective(
                    Vec3::new(0.0, 0.0, NEAR_CAMERA_Z),
                    HEIGHT as f32,
                    VERTICAL_FOV,
                    NEAR_PLANE,
                ),
                &near_settings,
            )
            .expect("near package frontier is selectable");
            let far = select_frontier(
                &hierarchy,
                &AllResident,
                LodView::perspective(
                    Vec3::new(0.0, 0.0, FAR_CAMERA_Z),
                    HEIGHT as f32,
                    VERTICAL_FOV,
                    NEAR_PLANE,
                ),
                &near_settings,
            )
            .expect("far package frontier is selectable");
            let near_pages = near
                .nodes
                .iter()
                .map(|node_id| {
                    lod.manifest
                        .nodes
                        .iter()
                        .find(|node| node.id == *node_id)
                        .expect("selected package node exists")
                        .representation
                        .page
                })
                .collect::<BTreeSet<_>>()
                .len();
            assert!(
                near_pages > 1,
                "bounded package staging fixture must require several physical pages"
            );
            assert!(
                near_pages <= settings.budgets.max_resident_pages as usize,
                "near package frontier must fit its resident-page budget"
            );
            assert_eq!(coarse.status.active_gaussians, fixture.coarsest_count);
            assert_eq!(near.status.active_gaussians, fixture.expected_near_probe);
            assert_eq!(far.status.active_gaussians, fixture.expected_far_probe);

            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock is after the Unix epoch")
                .as_nanos();
            let root = env::temp_dir().join(format!(
                "bevy_gaussian_lod_quality_package_{}_{}",
                std::process::id(),
                nonce
            ));
            let pages_root = root.join("pages");
            fs::create_dir_all(&pages_root).expect("native package directory is writable");
            for (descriptor, page) in lod.manifest.pages.iter_mut().zip(&lod.pages) {
                assert_eq!(descriptor.id, page.id);
                let encoded = encode_page(page).expect("native package page encodes");
                let file_name = format!("{:016x}.gspage", page.id.0);
                fs::write(pages_root.join(&file_name), &encoded)
                    .expect("native package page is writable");
                descriptor.storage = Some(LodPageStorage {
                    uri: format!("pages/{file_name}"),
                    byte_range: None,
                    encoded_len: encoded.len() as u64,
                });
            }
            lod.validate()
                .expect("native package remains valid after assigning page locations");
            fs::write(
                root.join("scene.gsplatlod"),
                encode_manifest(&lod.manifest).expect("native package manifest encodes"),
            )
            .expect("native package manifest is writable");

            let gaussians_per_slot = lod
                .manifest
                .pages
                .iter()
                .map(|page| page.gaussian_count)
                .max()
                .expect("native package contains pages");
            Self {
                root,
                settings,
                expected_coarse: coarse.status.active_gaussians,
                expected_near: near.status.active_gaussians,
                expected_far: far.status.active_gaussians,
                expected_near_pages: near_pages,
                canonical_slot_bytes: u64::from(gaussians_per_slot)
                    * size_of::<Gaussian3d>() as u64,
                node_parents: garden_node_parents(&lod.manifest),
            }
        }
    }

    impl Drop for NativePackageFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum PackageCapturePhase {
        CoarseWaiting,
        CoarsePending,
        NearWaiting,
        NearPending,
        FarWaiting,
        FarPending,
    }

    #[derive(Resource)]
    struct PackageQualityRenderState {
        package_root: String,
        settings: GaussianLodSettings,
        expected_coarse: u64,
        expected_near: u64,
        expected_far: u64,
        expected_near_pages: usize,
        phase: PackageCapturePhase,
        phase_frames: u32,
        total_frames: u32,
        stable_active_frames: u32,
        retained_coarse_frames: u32,
        refinement_transition_frames: Option<u32>,
        peak_resident_pages: u32,
        node_parents: GardenNodeParents,
        last_blend: Option<GardenViewBlendObservation>,
        late_edge_keys: BTreeSet<GardenViewBlendEdgeKey>,
        late_edge_initial_weight_bits: BTreeMap<GardenViewBlendEdgeKey, u32>,
        active_late_edge_keys: BTreeSet<GardenViewBlendEdgeKey>,
        lagged_late_edge_keys: BTreeSet<GardenViewBlendEdgeKey>,
        caught_up_late_edge_keys: BTreeSet<GardenViewBlendEdgeKey>,
        max_overlapping_lagging_late_edges: u32,
        authored_publication_hold: GardenAuthoredPublicationHold,
        promoted_drawable: GardenPromotedDrawableTracker,
        saw_late_activation: bool,
        saw_late_lag: bool,
        saw_late_public_unsatisfied: bool,
        saw_late_catchup: bool,
        target: Option<Handle<Image>>,
        cloud: Option<Entity>,
        camera: Option<Entity>,
    }

    impl PackageQualityRenderState {
        fn new(package: &NativePackageFixture) -> Self {
            Self {
                package_root: package.root.to_string_lossy().into_owned(),
                settings: package.settings.clone(),
                expected_coarse: package.expected_coarse,
                expected_near: package.expected_near,
                expected_far: package.expected_far,
                expected_near_pages: package.expected_near_pages,
                phase: PackageCapturePhase::CoarseWaiting,
                phase_frames: 0,
                total_frames: 0,
                stable_active_frames: 0,
                retained_coarse_frames: 0,
                refinement_transition_frames: None,
                peak_resident_pages: 0,
                node_parents: package.node_parents.clone(),
                last_blend: None,
                late_edge_keys: BTreeSet::new(),
                late_edge_initial_weight_bits: BTreeMap::new(),
                active_late_edge_keys: BTreeSet::new(),
                lagged_late_edge_keys: BTreeSet::new(),
                caught_up_late_edge_keys: BTreeSet::new(),
                max_overlapping_lagging_late_edges: 0,
                authored_publication_hold: GardenAuthoredPublicationHold::default(),
                promoted_drawable: GardenPromotedDrawableTracker::default(),
                saw_late_activation: false,
                saw_late_lag: false,
                saw_late_public_unsatisfied: false,
                saw_late_catchup: false,
                target: None,
                cloud: None,
                camera: None,
            }
        }

        fn enter(&mut self, phase: PackageCapturePhase) {
            self.phase = phase;
            self.phase_frames = 0;
            self.stable_active_frames = 0;
        }
    }

    fn setup_native_package_quality(
        mut commands: Commands,
        mut state: ResMut<PackageQualityRenderState>,
        asset_server: Res<AssetServer>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let target = images.add(Image::new_target_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let manifest: Handle<GaussianLodAsset> = asset_server.load("scene.gsplatlod");
        let cloud = commands
            .spawn((
                GaussianLodHandle(manifest),
                GaussianLodPackageSource::native_directory(state.package_root.clone()),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::Radix,
                    radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                    global_opacity: 1.5,
                    opacity_adaptive_radius: false,
                    ..default()
                },
                state.settings.clone(),
                Transform::IDENTITY,
                Visibility::Visible,
                Name::new("native_preprocessed_lod_package"),
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
                Name::new("native_package_quality_camera"),
            ))
            .id();
        state.target = Some(target);
        state.cloud = Some(cloud);
        state.camera = Some(camera);
    }

    // This is a Bevy system: explicit resources and queries document its ECS
    // access contract more clearly than a test-only aggregate SystemParam.
    #[allow(clippy::too_many_arguments)]
    fn drive_native_package_quality(
        mut commands: Commands,
        mut state: ResMut<PackageQualityRenderState>,
        package_statuses: Query<&GaussianLodPackageStatus>,
        lod_statuses: Query<&GaussianLodStatus>,
        bridge_statuses: Query<&GaussianLodBridgeStatus>,
        candidates: Query<&LodRenderCandidates>,
        upload_budget_status: Res<LodAtlasUploadBudgetStatus>,
        upload_queue: Res<LodAtlasUploadQueue>,
        indirect_probe: Res<IndirectProbe>,
        blend_probe: Res<GardenViewBlendRenderProbe>,
    ) {
        state.total_frames += 1;
        state.phase_frames += 1;
        if state.total_frames > PACKAGE_MAX_FRAMES {
            panic!(
                "native package quality test timed out in {:?}; expected=({},{},{}); status={:?}; candidates={:?}; last_blend={:?}; late=(keys={:?},lagged={:?},caught={:?},max_overlap={}); queued_uploads={}; upload_budget_error={:?}",
                state.phase,
                state.expected_coarse,
                state.expected_near,
                state.expected_far,
                state
                    .cloud
                    .and_then(|cloud| package_statuses.get(cloud).ok()),
                state.cloud.and_then(|cloud| candidates.get(cloud).ok()),
                state.last_blend,
                state.late_edge_keys,
                state.lagged_late_edge_keys,
                state.caught_up_late_edge_keys,
                state.max_overlapping_lagging_late_edges,
                upload_queue.queued_slot_count(),
                upload_budget_status.last_error(),
            );
        }

        let cloud = state.cloud.expect("native package cloud exists");
        let camera = state.camera.expect("native package camera exists");
        assert!(
            bridge_statuses.get(cloud).is_err(),
            "a preprocessed package must not enter the transient LoD bridge"
        );
        if let Ok(status) = package_statuses.get(cloud) {
            assert!(
                status.failure.is_none(),
                "native package streaming failed in {:?}: {status:?}; candidates={:?}",
                state.phase,
                candidates.get(cloud).ok(),
            );
            assert_eq!(status.terminal_failures, 0);
            assert_ne!(status.phase, GaussianLodPackagePhase::Failed);
            assert_ne!(status.phase, GaussianLodPackagePhase::Degraded);
            assert!(
                status.resident_pages <= state.settings.budgets.max_resident_pages,
                "native package exceeded its resident-page budget: {status:?}"
            );
            state.peak_resident_pages = state.peak_resident_pages.max(status.resident_pages);
            if state.phase == PackageCapturePhase::NearWaiting
                && status.phase == GaussianLodPackagePhase::Active
                && status.active_gaussians == state.expected_coarse
            {
                state.retained_coarse_frames += 1;
            }
        }

        let mut active_blend_fixed = true;
        let render = blend_probe.latest_snapshot();
        if render.is_none() {
            assert!(
                state.promoted_drawable.last_accepted.is_none(),
                "native synthetic ABI-16 drawable probe disappeared after its first promoted output"
            );
        }
        let drawable_class = render.as_ref().map(|render| {
            state
                .promoted_drawable
                .classify(render, "native synthetic ABI-16 handoff")
        });
        if let (Some(drawable_class), Some(render)) = (drawable_class, render) {
            let (render_candidate, exact_token_aggregate) = match drawable_class {
                GardenPromotedDrawableClass::CurrentCandidate => (render.candidate.clone(), true),
                GardenPromotedDrawableClass::RetainedCurrent(retained) => (retained, false),
            };
            if !render_candidate.prepared {
                return;
            }
            let allow_unevaluated_late_authored = exact_token_aggregate
                && !render_candidate.active
                && !render_candidate.transitioning;
            match render_candidate.temporal_mode {
                Some(LodTemporalTransitionMode::BoundedHardCohort) => {
                    panic!("synthetic ABI-16 lifecycle fixture used a hard cohort")
                }
                None => {
                    if !render_candidate.active {
                        return;
                    }
                    state
                        .authored_publication_hold
                        .assert_no_pending_incomplete_publication(
                            "native synthetic ABI-16 categorical endpoint",
                        );
                    state.last_blend = None;
                    state.authored_publication_hold.table = None;
                    state
                        .authored_publication_hold
                        .pending_ordinary_edges
                        .clear();
                    state.authored_publication_hold.recovery_edges.clear();
                    state.authored_publication_hold.consecutive_frames = 0;
                    state
                        .authored_publication_hold
                        .incomplete_distinct_publications = 0;
                    state.authored_publication_hold.last_incomplete_publication = None;
                }
                Some(LodTemporalTransitionMode::Morphing) => {
                    let blend = observe_garden_view_blend_with_render_state(
                        &render_candidate,
                        render,
                        exact_token_aggregate,
                        "native synthetic ABI-16 package",
                    );
                    blend.assert_dynamic_coherent("native synthetic ABI-16 package");
                    blend.assert_active_dynamic_evaluation_complete(
                        "native synthetic ABI-16 package",
                    );
                    blend.assert_no_invalid_pressure_pairs("native synthetic ABI-16 package");
                    blend.assert_manifest_edge_topology(
                        &state.node_parents,
                        "native synthetic ABI-16 package",
                    );
                    if let Some(previous) = state.last_blend.clone() {
                        state.authored_publication_hold.assert_recovery_slew_from(
                            &blend,
                            &previous,
                            exact_token_aggregate,
                            "native synthetic ABI-16 late-delivery recovery",
                        );
                        let evidence = blend.assert_dynamic_frame_transition(
                            &previous,
                            &state.authored_publication_hold.recovery_edges,
                            &state.authored_publication_hold.pending_ordinary_edges,
                            &state.node_parents,
                            exact_token_aggregate,
                            allow_unevaluated_late_authored,
                            "native synthetic ABI-16 transition",
                        );
                        state.authored_publication_hold.observe(
                            &blend,
                            &evidence,
                            "native synthetic ABI-16 package",
                        );
                        if evidence.new_authored_publication {
                            let previous_keys = previous
                                .edges
                                .iter()
                                .map(|edge| &edge.key)
                                .collect::<BTreeSet<_>>();
                            for edge in blend
                                .edges
                                .iter()
                                .filter(|edge| !previous_keys.contains(&edge.key))
                                .filter(|edge| edge.activation_requires_slew)
                            {
                                assert!(
                                    state.late_edge_keys.insert(edge.key.clone()),
                                    "native synthetic ABI-16 fixture admitted the same late edge twice"
                                );
                                assert!(
                                    state
                                        .late_edge_initial_weight_bits
                                        .insert(edge.key.clone(), edge.initial_weight_bits)
                                        .is_none(),
                                    "native synthetic ABI-16 fixture replaced the retained initial endpoint for a late edge"
                                );
                            }
                        }
                    } else {
                        assert!(
                            blend.upload.immutable_table_upload_count > 0,
                            "native synthetic ABI-16 first blend had no immutable table upload"
                        );
                        let late_edges = blend
                            .edges
                            .iter()
                            .filter(|edge| edge.activation_requires_slew)
                            .collect::<Vec<_>>();
                        assert!(
                            !late_edges.is_empty(),
                            "native synthetic ABI-16 first Morphing publication had no late-delivery provenance"
                        );
                        let mut unevaluated_late_edge_keys = BTreeSet::new();
                        for edge in late_edges {
                            assert!(
                                edge.recovery_lag,
                                "native late edge omitted its mutable recovery provenance"
                            );
                            assert_eq!(
                                edge.displayed_weight_bits, edge.initial_weight_bits,
                                "native late edge did not activate at its authored retained endpoint"
                            );
                            if let Some(oracle) = edge.evaluation_weight_bits {
                                assert_eq!(
                                    edge.desired_weight_bits, oracle,
                                    "native late edge desired weight did not use the captured view"
                                );
                                if edge.desired_weight_bits != edge.initial_weight_bits {
                                    assert_ne!(
                                        edge.displayed_weight_bits, edge.desired_weight_bits,
                                        "native late edge hid its required first-publication recovery lag"
                                    );
                                }
                            } else {
                                assert!(
                                    allow_unevaluated_late_authored
                                        && !blend.desired_evaluation_complete
                                        && blend.evaluation_view.is_none()
                                        && blend.evaluation_target.is_none(),
                                    "native late edge omitted its oracle outside the exact PREPARED authored publication"
                                );
                                assert_eq!(
                                    edge.desired_weight_bits, edge.initial_weight_bits,
                                    "native unevaluated late edge changed desired before preflight"
                                );
                                unevaluated_late_edge_keys.insert(edge.key.clone());
                            }
                            let first_late_observation =
                                state.late_edge_keys.insert(edge.key.clone());
                            match state.late_edge_initial_weight_bits.entry(edge.key.clone()) {
                                std::collections::btree_map::Entry::Vacant(entry) => {
                                    entry.insert(edge.initial_weight_bits);
                                }
                                std::collections::btree_map::Entry::Occupied(entry) => {
                                    assert_eq!(
                                        *entry.get(),
                                        edge.initial_weight_bits,
                                        "native synthetic ABI-16 fixture changed a late edge's retained initial endpoint before ACTIVE"
                                    );
                                    assert!(
                                        !first_late_observation,
                                        "native synthetic ABI-16 late-edge key/initial tracking diverged"
                                    );
                                }
                            }
                        }
                        if !unevaluated_late_edge_keys.is_empty() {
                            assert_eq!(
                                (
                                    blend.upload.last_max_delta.to_bits(),
                                    blend.upload.last_weighted_record_energy.to_bits(),
                                ),
                                (0.0_f32.to_bits(), 0.0_f64.to_bits()),
                                "native PREPARED late publication reported displayed-weight work"
                            );
                        }
                        let evidence = GardenViewBlendActivationEvidence {
                            activation_frame: true,
                            new_authored_publication: true,
                            preserved_fractional_overlap: false,
                            new_edge_keys: blend
                                .edges
                                .iter()
                                .map(|edge| edge.key.clone())
                                .collect(),
                            unevaluated_late_edge_keys,
                        };
                        state.authored_publication_hold.observe(
                            &blend,
                            &evidence,
                            "native synthetic ABI-16 first Morphing publication",
                        );
                    }
                    if render_candidate.active && blend.desired_evaluation_complete {
                        let active_late_edges = blend
                            .edges
                            .iter()
                            .filter(|edge| state.late_edge_keys.contains(&edge.key))
                            .collect::<Vec<_>>();
                        for edge in &active_late_edges {
                            assert_eq!(
                                edge.evaluation_weight_bits,
                                Some(edge.desired_weight_bits),
                                "native late edge became ACTIVE without its exact desired oracle"
                            );
                            let first_active = state.active_late_edge_keys.insert(edge.key.clone());
                            if first_active {
                                let retained_initial = state
                                    .late_edge_initial_weight_bits
                                    .get(&edge.key)
                                    .copied()
                                    .expect(
                                        "native late edge retained initial endpoint is tracked",
                                    );
                                if edge.desired_weight_bits != retained_initial {
                                    assert_ne!(
                                        edge.displayed_weight_bits, edge.desired_weight_bits,
                                        "native late edge first became ACTIVE only after hiding its required recovery lag"
                                    );
                                    assert!(
                                        edge.recovery_lag,
                                        "native late edge first became ACTIVE without its mutable recovery marker"
                                    );
                                }
                            }
                            if edge.displayed_weight_bits != edge.desired_weight_bits {
                                state.lagged_late_edge_keys.insert(edge.key.clone());
                                state.saw_late_lag = true;
                            } else if state.lagged_late_edge_keys.contains(&edge.key) {
                                state.caught_up_late_edge_keys.insert(edge.key.clone());
                            }
                        }
                        state.saw_late_activation = !state.active_late_edge_keys.is_empty();
                        let overlapping_lagging_late_edges = active_late_edges
                            .iter()
                            .filter(|edge| edge.displayed_weight_bits != edge.desired_weight_bits)
                            .count()
                            .try_into()
                            .unwrap_or(u32::MAX);
                        state.max_overlapping_lagging_late_edges = state
                            .max_overlapping_lagging_late_edges
                            .max(overlapping_lagging_late_edges);
                        state.saw_late_catchup = !state.late_edge_keys.is_empty()
                            && state.caught_up_late_edge_keys == state.late_edge_keys;
                    }
                    if let Ok(public) = lod_statuses.get(cloud) {
                        assert_eq!(
                            public.view_blend_invalid_pressure_evaluations, 0,
                            "native synthetic ABI-16 public status reported invalid pressure"
                        );
                        if public.view_blend_missing_consumers > 0 {
                            assert_ne!(
                                public.target_satisfied,
                                Some(true),
                                "native synthetic ABI-16 public status claimed a fresh request while its lagged MainWorld status still reported a missing private consumer"
                            );
                        }
                        let active_late_recovery = render_candidate.active
                            && blend.desired_evaluation_complete
                            && blend.edges.iter().any(|edge| {
                                state.late_edge_keys.contains(&edge.key)
                                    && edge.displayed_weight_bits != edge.desired_weight_bits
                            });
                        if public.view_blend_lagging_edges > 0 {
                            assert_ne!(
                                public.target_satisfied,
                                Some(true),
                                "native synthetic ABI-16 status claimed a fresh request while displayed weights lagged"
                            );
                            if active_late_recovery {
                                state.saw_late_public_unsatisfied |=
                                    public.target_satisfied == Some(false);
                            }
                        }
                    }
                    active_blend_fixed = exact_token_aggregate
                        && render_candidate.active
                        && blend.desired_evaluation_complete
                        && blend.evaluation_view == Some(blend.current_render_view)
                        && blend.evaluation_target == Some(blend.current_render_target)
                        && blend.status.lagging_count == 0
                        && blend.status.invalid_pressure_count == 0
                        && blend.status.missing_consumer_count == 0
                        && blend
                            .edges
                            .iter()
                            .all(|edge| edge.displayed_weight_bits == edge.desired_weight_bits);
                    if active_blend_fixed {
                        blend.assert_stationary_fixed_point(
                            "native synthetic ABI-16 fixed package cut",
                        );
                    }
                    state.last_blend = Some(blend);
                }
            }
        }

        let expected = match state.phase {
            PackageCapturePhase::CoarseWaiting => Some(state.expected_coarse),
            PackageCapturePhase::NearWaiting => Some(state.expected_near),
            PackageCapturePhase::FarWaiting => Some(state.expected_far),
            PackageCapturePhase::CoarsePending
            | PackageCapturePhase::NearPending
            | PackageCapturePhase::FarPending => None,
        };
        let Some(expected) = expected else {
            return;
        };
        if active_blend_fixed
            && coherent_package_cut(
                cloud,
                camera,
                expected,
                &package_statuses,
                &candidates,
                &indirect_probe,
            )
        {
            state.stable_active_frames += 1;
        } else {
            state.stable_active_frames = 0;
        }
        if state.stable_active_frames < STABLE_ACTIVE_FRAMES {
            return;
        }

        if state.phase == PackageCapturePhase::NearWaiting {
            state.refinement_transition_frames = Some(state.phase_frames);
        }
        commands.spawn(Screenshot::image(
            state.target.clone().expect("native package target exists"),
        ));
        state.phase = match state.phase {
            PackageCapturePhase::CoarseWaiting => PackageCapturePhase::CoarsePending,
            PackageCapturePhase::NearWaiting => PackageCapturePhase::NearPending,
            PackageCapturePhase::FarWaiting => PackageCapturePhase::FarPending,
            pending => pending,
        };
        state.phase_frames = 0;
        state.stable_active_frames = 0;
    }

    fn coherent_package_cut(
        cloud: Entity,
        camera: Entity,
        expected: u64,
        statuses: &Query<&GaussianLodPackageStatus>,
        candidates: &Query<&LodRenderCandidates>,
        indirect_probe: &IndirectProbe,
    ) -> bool {
        let Ok(status) = statuses.get(cloud) else {
            return false;
        };
        if status.phase != GaussianLodPackagePhase::Active {
            return false;
        }
        let Ok(candidates) = candidates.get(cloud) else {
            return false;
        };
        assert_eq!(candidates.len(), 1);
        if candidates.package_retention_for_testing() != (true, true, false) {
            // Update can observe a radix-ready pending candidate one schedule
            // before package PostUpdate commits it. Until then the candidate
            // component and package status intentionally describe different
            // generations, so count equality is not yet a coherent-cut proof.
            return false;
        }
        let Some(candidate) = candidates.get(camera) else {
            return false;
        };
        assert!(
            !candidate.failed(),
            "native package render candidate failed"
        );
        if !candidate.render_is_active_for_testing() {
            return false;
        }
        let candidate_count = u64::from(candidate.frontier().candidate_count());
        if candidate_count != expected {
            return false;
        }
        let range_count = candidate
            .render_ranges()
            .iter()
            .map(|range| u64::from(range.count))
            .sum::<u64>();
        assert_eq!(
            range_count,
            u64::from(candidate.rendered_candidate_count()),
            "native package presentation ranges and submitted count diverged"
        );
        assert_eq!(status.active_gaussians, range_count);
        assert_eq!(
            candidate.rendered_quality_status().active_gaussians,
            expected
        );
        let resident_candidate_pages = candidate
            .render_ranges()
            .iter()
            .map(|range| range.page)
            .collect::<BTreeSet<_>>()
            .len();
        assert!(
            status.resident_pages as usize >= resident_candidate_pages,
            "active package cut references more pages than status reports: status={status:?}"
        );
        indirect_probe.request(candidate.rendered_candidate_count())
    }

    fn on_native_package_capture(
        trigger: On<ScreenshotCaptured>,
        mut state: ResMut<PackageQualityRenderState>,
        mut lod_settings: Query<&mut GaussianLodSettings>,
        mut transforms: Query<&mut Transform>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("native package offscreen capture converts")
            .to_rgba8();
        let non_black_pixels = rgba
            .as_raw()
            .chunks_exact(4)
            .filter(|pixel| pixel[0].max(pixel[1]).max(pixel[2]) > 8)
            .count();
        assert!(
            non_black_pixels > 0,
            "stable native package cut rendered an empty offscreen target in {:?}",
            state.phase
        );

        let cloud = state.cloud.expect("native package cloud exists");
        match state.phase {
            PackageCapturePhase::CoarsePending => {
                lod_settings
                    .get_mut(cloud)
                    .expect("native package LoD settings exist")
                    .quality = CAMERA_PROBE_QUALITY;
                state.enter(PackageCapturePhase::NearWaiting);
            }
            PackageCapturePhase::NearPending => {
                assert!(
                    state.expected_near_pages > 1
                        && state
                            .refinement_transition_frames
                            .is_some_and(|frames| frames >= 2),
                    "one-slot package budget did not exercise multi-frame staging: pages={}, frames={:?}",
                    state.expected_near_pages,
                    state.refinement_transition_frames
                );
                assert!(
                    state.retained_coarse_frames > 0,
                    "package did not retain its drawable coarse cut while refinement streamed"
                );
                assert!(
                    state.peak_resident_pages > 1,
                    "package refinement never made several pages resident"
                );
                transforms
                    .get_mut(state.camera.expect("native package camera exists"))
                    .expect("native package camera transform exists")
                    .translation = Vec3::new(0.0, 0.0, FAR_CAMERA_Z);
                state.enter(PackageCapturePhase::FarWaiting);
            }
            PackageCapturePhase::FarPending => {
                assert!(
                    state.expected_far < state.expected_near,
                    "far package camera cut did not deterministically coarsen"
                );
                assert!(
                    state.saw_late_activation
                        && !state.late_edge_keys.is_empty()
                        && state.active_late_edge_keys == state.late_edge_keys,
                    "synthetic ABI-16 one-slot fixture did not observe every late-delivery edge in an ACTIVE complete publication: late={:?}, active={:?}",
                    state.late_edge_keys,
                    state.active_late_edge_keys,
                );
                assert!(
                    state.saw_late_lag && state.lagged_late_edge_keys == state.late_edge_keys,
                    "synthetic ABI-16 late edges did not all expose truthful displayed/desired lag: late={:?}, lagged={:?}",
                    state.late_edge_keys,
                    state.lagged_late_edge_keys,
                );
                assert!(
                    state.saw_late_public_unsatisfied,
                    "synthetic ABI-16 public status never withheld request freshness during late recovery"
                );
                assert!(
                    state.saw_late_catchup
                        && state.caught_up_late_edge_keys == state.late_edge_keys,
                    "synthetic ABI-16 late edges did not all catch up: late={:?}, caught={:?}",
                    state.late_edge_keys,
                    state.caught_up_late_edge_keys,
                );
                state
                    .authored_publication_hold
                    .assert_no_pending_incomplete_publication(
                        "synthetic ABI-16 late-delivery lifecycle",
                    );
                eprintln!(
                    "synthetic ABI-16 late-delivery lifecycle: edges={}, authored_publications={}, max_consecutive_hold_frames={}, max_overlapping_lagging_edges={}",
                    state.late_edge_keys.len(),
                    state.authored_publication_hold.distinct_publications,
                    state.authored_publication_hold.max_consecutive_frames,
                    state.max_overlapping_lagging_late_edges,
                );
                exit.write(AppExit::Success);
            }
            phase => panic!("native package capture arrived in unexpected phase {phase:?}"),
        }
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
