#[cfg(not(all(feature = "headless", feature = "testing")))]
#[test]
fn lod_device_recovery_requires_headless_and_testing_features() {}

#[cfg(all(feature = "headless", feature = "testing"))]
mod headless {
    use std::{
        collections::BTreeSet,
        env, fs,
        path::{Path, PathBuf},
        process::Command,
        time::Duration,
    };

    use bevy::{
        app::{AppExit, ScheduleRunnerPlugin},
        camera::{PerspectiveProjection, Projection, RenderTarget},
        core_pipeline::tonemapping::Tonemapping,
        light::cluster::{ClusterConfig, GlobalClusterSettings},
        prelude::*,
        render::{
            render_resource::{PollType, TextureFormat},
            renderer::{RenderAdapterInfo, RenderDevice},
            view::screenshot::{Screenshot, ScreenshotCaptured},
        },
        window::ExitCondition,
        winit::WinitPlugin,
    };
    use bevy_gaussian_splatting::{
        CloudSettings, GaussianCamera, GaussianLodBridgeConfig, GaussianLodBuildSettings,
        GaussianLodSettings, GaussianMode, GaussianRenderRecoveryPhase,
        GaussianRenderRecoveryStatus, GaussianSplattingPlugin, PlanarGaussian3d,
        PlanarGaussian3dHandle, RadixSortDepthBits, build_planar_3d_lod,
        sort::SortMode,
        stream::{
            bridge::{GaussianLodBridgePhase, GaussianLodBridgeStatus},
            hierarchy::{AllResident, LodView, ManifestLodHierarchy, select_frontier},
        },
        testing::{LodTestScene, compare_linear_rgba},
    };
    use serde::{Deserialize, Serialize};

    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 128;
    const EXPECTED_ACTIVE: u64 = 3;
    // Pipeline compilation runs again after recovery; leave enough frames for
    // slower CI Metal/DX12 shader compilers while retaining a bounded failure.
    const MAX_FRAMES: u32 = 3_600;
    const STABLE_FRAMES: u32 = 10;
    const MIN_ADAPTER_GOLDENS_ENV: &str = "LOD_MIN_ADAPTER_GOLDENS";

    #[test]
    fn active_lod_render_recovers_after_injected_device_loss() {
        if env::var("RUN_GPU_DEVICE_LOSS_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping GPU device-loss test; set RUN_GPU_DEVICE_LOSS_TESTS=1 to enable");
            return;
        }
        run_capture(CaptureMode::RecoverDevice);
    }

    #[test]
    fn cross_adapter_goldens() {
        if env::var("RUN_GPU_CROSS_ADAPTER_TESTS").ok().as_deref() != Some("1") {
            eprintln!(
                "skipping cross-adapter golden test; set RUN_GPU_CROSS_ADAPTER_TESTS=1 to enable"
            );
            return;
        }

        let adapters = compatible_adapters();
        let minimum_adapters = env::var(MIN_ADAPTER_GOLDENS_ENV)
            .map(|value| {
                value.parse::<usize>().unwrap_or_else(|error| {
                    panic!(
                        "{MIN_ADAPTER_GOLDENS_ENV} must be a positive integer, got {value:?}: {error}"
                    )
                })
            })
            .unwrap_or(2);
        assert!(
            minimum_adapters >= 2,
            "{MIN_ADAPTER_GOLDENS_ENV} must be at least two for a cross-adapter comparison"
        );
        assert!(
            adapters.len() >= minimum_adapters,
            "found {} adapter(s) satisfying the LoD radix/storage limit contract, but {MIN_ADAPTER_GOLDENS_ENV} requires {minimum_adapters}: {adapters:?}",
            adapters.len()
        );
        eprintln!(
            "discovered {} compatible adapter(s): {adapters:?}",
            adapters.len()
        );
        let output_dir = env::temp_dir().join(format!(
            "bevy_gaussian_lod_adapter_goldens_{}",
            std::process::id()
        ));
        fs::create_dir_all(&output_dir).expect("cross-adapter golden directory is writable");

        let executable = env::current_exe().expect("current test executable is known");
        let mut observations = Vec::with_capacity(adapters.len());
        for (index, adapter) in adapters.iter().enumerate() {
            let output = output_dir.join(format!("adapter_{index}.json"));
            let status = Command::new(&executable)
                .args([
                    "--exact",
                    "headless::cross_adapter_golden_child",
                    "--nocapture",
                ])
                .env("LOD_ADAPTER_GOLDEN_CHILD", "1")
                .env("LOD_ADAPTER_GOLDEN_OUTPUT", &output)
                .env("WGPU_ADAPTER_NAME", &adapter.name)
                .env("WGPU_BACKEND", backend_environment_name(adapter.backend))
                .status()
                .unwrap_or_else(|error| panic!("failed to launch adapter {:?}: {error}", adapter));
            assert!(status.success(), "adapter child failed for {adapter:?}");
            let encoded = fs::read(&output).expect("adapter child wrote its observation");
            let observation: AdapterGolden =
                serde_json::from_slice(&encoded).expect("adapter observation is valid JSON");
            assert_eq!(
                observation.adapter, adapter.name,
                "adapter child did not select the requested adapter"
            );
            assert_eq!(
                observation.backend,
                format!("{:?}", adapter.backend),
                "adapter child did not select the requested backend"
            );
            assert_eq!(observation.active_gaussians, EXPECTED_ACTIVE);
            assert_eq!(observation.width, WIDTH);
            assert_eq!(observation.height, HEIGHT);
            assert!(observation.non_black_pixels >= 64);
            observations.push((output, observation));
        }

        let reference_bytes = read_golden_rgba(&observations[0].0, &observations[0].1);
        let reference = linear_rgba(&reference_bytes);
        let mut comparisons = 0;
        for (path, observation) in observations.iter().skip(1) {
            let candidate_bytes = read_golden_rgba(path, observation);
            let candidate = linear_rgba(&candidate_bytes);
            let metrics = compare_linear_rgba(&reference, &candidate, 0.05)
                .expect("cross-adapter images have compatible dimensions");
            assert!(
                metrics.psnr_rgb >= 48.0 && metrics.luminance_ssim >= 0.995,
                "cross-adapter image drift exceeded the golden tolerance: reference={:?}, candidate={observation:?}, metrics={metrics:?}",
                observations[0].1
            );
            assert!(
                metrics.max_abs_error <= 0.025,
                "cross-adapter maximum channel error exceeded tolerance: {metrics:?}"
            );
            comparisons += 1;
        }
        assert!(
            comparisons >= 1,
            "cross-adapter goldens must compare at least one adapter against the reference"
        );

        eprintln!(
            "validated {} compatible adapter golden(s) and {comparisons} cross-adapter comparison(s): {:?}",
            observations.len(),
            observations
                .iter()
                .map(|(_, observation)| (&observation.adapter, &observation.backend))
                .collect::<Vec<_>>()
        );
        fs::remove_dir_all(&output_dir)
            .expect("cross-adapter golden temporary directory is removable");
    }

    #[test]
    fn cross_adapter_golden_child() {
        if env::var("LOD_ADAPTER_GOLDEN_CHILD").ok().as_deref() != Some("1") {
            return;
        }
        let output = PathBuf::from(
            env::var_os("LOD_ADAPTER_GOLDEN_OUTPUT")
                .expect("adapter golden child output path is configured"),
        );
        run_capture(CaptureMode::WriteGolden(output));
    }

    #[derive(Clone, Debug)]
    enum CaptureMode {
        RecoverDevice,
        WriteGolden(PathBuf),
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum CapturePhase {
        InitialWaiting,
        InitialPending,
        Recovering,
        RecoveredPending,
    }

    #[derive(Resource)]
    struct CaptureState {
        mode: CaptureMode,
        phase: CapturePhase,
        frames: u32,
        stable_frames: u32,
        target: Option<Handle<Image>>,
        cloud: Option<Entity>,
        initial_device_generation: u64,
        initial_rgba: Option<Vec<u8>>,
    }

    impl CaptureState {
        fn new(mode: CaptureMode) -> Self {
            Self {
                mode,
                phase: CapturePhase::InitialWaiting,
                frames: 0,
                stable_frames: 0,
                target: None,
                cloud: None,
                initial_device_generation: 0,
                initial_rgba: None,
            }
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct AdapterGolden {
        adapter: String,
        backend: String,
        width: u32,
        height: u32,
        active_gaussians: u64,
        non_black_pixels: usize,
        rgba_file: String,
        rgba_hash: u64,
    }

    fn run_capture(mode: CaptureMode) {
        let expected_golden = match &mode {
            CaptureMode::WriteGolden(path) => Some(path.clone()),
            CaptureMode::RecoverDevice => None,
        };
        let recovery_status = GaussianRenderRecoveryStatus::default();
        let mut app = App::new();
        app.insert_resource(ClearColor(Color::BLACK))
            .insert_resource(bridge_config())
            .insert_resource(recovery_status.clone())
            .insert_resource(CaptureState::new(mode));
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: "assets".to_owned(),
                    processed_file_path: "assets".to_owned(),
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
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, (disable_gpu_clustering, setup).chain())
        .add_systems(Update, drive_capture)
        .add_observer(on_capture);
        let exit = app.run();
        assert!(
            exit.is_success(),
            "LoD recovery/golden app exited with {exit:?}: {:?}",
            recovery_status.snapshot()
        );
        if let Some(output) = expected_golden {
            assert!(output.is_file(), "golden child did not write {output:?}");
            assert!(
                output.with_extension("rgba").is_file(),
                "golden child did not write the RGBA payload for {output:?}"
            );
        }
    }

    fn disable_gpu_clustering(mut settings: ResMut<GlobalClusterSettings>) {
        // Bevy's adaptive GPU-cluster readback can still have a mapped staging
        // callback pending when this test intentionally destroys the device.
        // Disable that readback; `ClusterConfig::None` keeps the light-free
        // fixture's CPU cluster payload empty as well.
        settings.gpu_clustering = None;
    }

    fn bridge_config() -> GaussianLodBridgeConfig {
        GaussianLodBridgeConfig {
            max_ephemeral_source_gaussians: 4_096,
            max_ephemeral_stored_gaussians: 8_192,
            max_atlas_gaussians: 8_192,
            max_atlas_bytes: 32 * 1024 * 1024,
            build_settings: GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 8,
                support_sigma: 3.0,
            },
            ..default()
        }
    }

    fn lod_settings() -> GaussianLodSettings {
        let mut settings = GaussianLodSettings {
            quality: 0.24,
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
        settings
    }

    fn recovery_cloud_settings() -> CloudSettings {
        CloudSettings {
            gaussian_mode: GaussianMode::Gaussian3d,
            sort_mode: SortMode::Radix,
            radix_sort_depth_bits: RadixSortDepthBits::Bits32,
            global_opacity: 1.5,
            opacity_adaptive_radius: false,
            ..default()
        }
    }

    #[test]
    fn recovery_fixture_uses_an_active_lod_cut() {
        let settings = recovery_cloud_settings();
        assert!(!settings.lod_debug.requires_metadata());

        let source = include_str!("lod_device_recovery.rs");
        let compatible_adapters = &source[source
            .rfind("fn compatible_adapters()")
            .expect("adapter compatibility helper remains present")..];
        assert!(compatible_adapters.contains("max_compute_invocations_per_workgroup >= 256"));
        assert!(compatible_adapters.contains("max_compute_workgroup_size_x >= 256"));
        assert!(!compatible_adapters.contains("max_compute_invocations_per_workgroup >= 1_024"));

        let cloud = LodTestScene::checkerboard_facade(20, 16).cloud();
        let built = build_planar_3d_lod(&cloud, bridge_config().build_settings)
            .expect("recovery fixture hierarchy builds");
        let hierarchy =
            ManifestLodHierarchy::new(&built.manifest).expect("recovery manifest is valid");
        let active = select_frontier(
            &hierarchy,
            &AllResident,
            LodView::perspective(
                Vec3::new(0.0, 0.0, 8.0),
                HEIGHT as f32,
                60.0_f32.to_radians(),
                0.01,
            ),
            &lod_settings(),
        )
        .expect("recovery fixture selection succeeds")
        .status
        .active_gaussians;
        eprintln!("recovery fixture active cut: {active}");
        assert_eq!(active, EXPECTED_ACTIVE);
    }

    fn setup(
        mut commands: Commands,
        mut state: ResMut<CaptureState>,
        mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let target = images.add(Image::new_target_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let cloud = gaussian_assets.add(LodTestScene::checkerboard_facade(20, 16).cloud());
        let cloud_entity = commands
            .spawn((
                PlanarGaussian3dHandle(cloud),
                recovery_cloud_settings(),
                lod_settings(),
                Transform::IDENTITY,
                Visibility::Visible,
            ))
            .id();
        commands.spawn((
            Camera3d::default(),
            Camera::default(),
            Projection::Perspective(PerspectiveProjection {
                fov: 60.0_f32.to_radians(),
                near: 0.01,
                far: 1_000.0,
                ..default()
            }),
            RenderTarget::Image(target.clone().into()),
            Transform::from_translation(Vec3::new(0.0, 0.0, 8.0)),
            Tonemapping::None,
            // This Gaussian-only fixture has no lights. Avoid unrelated
            // per-view cluster work at the intentional device-loss boundary.
            ClusterConfig::None,
            GaussianCamera::default(),
        ));
        state.target = Some(target);
        state.cloud = Some(cloud_entity);
    }

    fn drive_capture(
        mut commands: Commands,
        mut state: ResMut<CaptureState>,
        statuses: Query<&GaussianLodBridgeStatus>,
        recovery: Res<GaussianRenderRecoveryStatus>,
    ) {
        state.frames += 1;
        assert!(
            state.frames <= MAX_FRAMES,
            "LoD recovery/golden capture timed out in {:?}: recovery={:?}",
            state.phase,
            recovery.snapshot()
        );
        let cloud = state.cloud.expect("capture cloud exists");
        let active = statuses.get(cloud).is_ok_and(|status| {
            status.failure.is_none()
                && status.phase == GaussianLodBridgePhase::Active
                && status.active_gaussians == EXPECTED_ACTIVE
        });
        let recovery_snapshot = recovery.snapshot();

        let ready = match state.phase {
            CapturePhase::InitialWaiting => {
                active && recovery_snapshot.phase == GaussianRenderRecoveryPhase::Ready
            }
            CapturePhase::Recovering => {
                active
                    && recovery_snapshot.phase == GaussianRenderRecoveryPhase::Ready
                    && recovery_snapshot.device_generation > state.initial_device_generation
            }
            CapturePhase::InitialPending | CapturePhase::RecoveredPending => false,
        };
        state.stable_frames = if ready {
            state.stable_frames.saturating_add(1)
        } else {
            0
        };
        if state.stable_frames < STABLE_FRAMES {
            return;
        }
        commands.spawn(Screenshot::image(
            state.target.clone().expect("capture target exists"),
        ));
        state.phase = match state.phase {
            CapturePhase::InitialWaiting => CapturePhase::InitialPending,
            CapturePhase::Recovering => CapturePhase::RecoveredPending,
            phase => phase,
        };
        state.stable_frames = 0;
    }

    #[allow(clippy::too_many_arguments)]
    fn on_capture(
        trigger: On<ScreenshotCaptured>,
        mut state: ResMut<CaptureState>,
        statuses: Query<&GaussianLodBridgeStatus>,
        render_device: Res<RenderDevice>,
        adapter: Res<RenderAdapterInfo>,
        recovery: Res<GaussianRenderRecoveryStatus>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("device recovery screenshot converts")
            .to_rgba8()
            .into_raw();
        let non_black_pixels = rgba
            .chunks_exact(4)
            .filter(|pixel| pixel[0].max(pixel[1]).max(pixel[2]) > 8)
            .count();
        assert!(
            non_black_pixels >= 64,
            "active LoD recovery/golden frame was empty"
        );
        let cloud = state.cloud.expect("capture cloud exists");
        let status = statuses
            .get(cloud)
            .expect("capture cloud retains its LoD bridge status");
        assert_eq!(status.phase, GaussianLodBridgePhase::Active);
        assert!(
            status.failure.is_none(),
            "active LoD bridge failed: {status:?}"
        );
        assert_eq!(status.active_gaussians, EXPECTED_ACTIVE);

        match (&state.mode, state.phase) {
            (CaptureMode::WriteGolden(output), CapturePhase::InitialPending) => {
                let rgba_path = output.with_extension("rgba");
                fs::write(&rgba_path, &rgba).expect("adapter RGBA golden is writable");
                let observation = AdapterGolden {
                    adapter: adapter.name.clone(),
                    backend: format!("{:?}", adapter.backend),
                    width: WIDTH,
                    height: HEIGHT,
                    active_gaussians: status.active_gaussians,
                    non_black_pixels,
                    rgba_file: rgba_path
                        .file_name()
                        .expect("RGBA output has a filename")
                        .to_string_lossy()
                        .into_owned(),
                    rgba_hash: fnv1a64(&rgba),
                };
                fs::write(
                    output,
                    serde_json::to_vec_pretty(&observation)
                        .expect("adapter observation serializes"),
                )
                .expect("adapter observation is writable");
                exit.write(AppExit::Success);
            }
            (CaptureMode::RecoverDevice, CapturePhase::InitialPending) => {
                state.initial_device_generation = recovery.snapshot().device_generation;
                state.initial_rgba = Some(rgba);
                state.phase = CapturePhase::Recovering;
                state.stable_frames = 0;
                render_device.wgpu_device().destroy();
                // Force delivery of wgpu's device-lost callback before the
                // next extraction/render cycle can submit work to the old
                // device. A raw `destroy()` alone may defer callback polling
                // until a later render system has already touched it.
                render_device
                    .poll(PollType::wait_indefinitely())
                    .expect("destroyed device loss callback is pollable");
            }
            (CaptureMode::RecoverDevice, CapturePhase::RecoveredPending) => {
                let initial = state
                    .initial_rgba
                    .as_ref()
                    .expect("pre-loss capture exists");
                assert_eq!(
                    initial,
                    &rgba,
                    "device recovery changed the deterministic LoD image; metrics={:?}",
                    compare_linear_rgba(&linear_rgba(initial), &linear_rgba(&rgba), 0.05)
                );
                let snapshot = recovery.snapshot();
                assert_eq!(snapshot.phase, GaussianRenderRecoveryPhase::Ready);
                assert!(snapshot.device_generation > state.initial_device_generation);
                assert!(snapshot.total_device_losses >= 1);
                exit.write(AppExit::Success);
            }
            (_, phase) => panic!("screenshot arrived in unexpected phase {phase:?}"),
        }
    }

    fn compatible_adapters() -> Vec<wgpu::AdapterInfo> {
        let instance = wgpu::Instance::default();
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let mut seen = BTreeSet::new();
        adapters
            .into_iter()
            .filter(|adapter| {
                let limits = adapter.limits();
                limits.max_compute_invocations_per_workgroup >= 256
                    && limits.max_compute_workgroup_size_x >= 256
                    && limits.max_storage_buffers_per_shader_stage >= 9
            })
            .filter_map(|adapter| {
                let info = adapter.get_info();
                let key = (info.name.clone(), format!("{:?}", info.backend));
                if !seen.insert(key) {
                    return None;
                }

                // Bevy's default Functionality priority requests the adapter's
                // full feature/limit set (except mappable primary buffers on a
                // discrete GPU). Some backends advertise the numeric LoD
                // limits but cannot create that device in a headless process,
                // so prove the complete initialization contract up front.
                let mut required_features = adapter.features();
                if info.device_type == wgpu::DeviceType::DiscreteGpu {
                    required_features.remove(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
                }
                required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
                let descriptor = wgpu::DeviceDescriptor {
                    label: Some("LoD cross-adapter compatibility probe"),
                    required_features,
                    required_limits: adapter.limits(),
                    // Match Bevy's native renderer descriptor. No experimental
                    // feature is used by this fixture, but the token itself is
                    // part of Bevy's device-request contract.
                    experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                };
                match pollster::block_on(adapter.request_device(&descriptor)) {
                    Ok((device, queue)) => {
                        drop(queue);
                        drop(device);
                        Some(info)
                    }
                    Err(error) => {
                        eprintln!(
                            "excluding adapter that cannot create the Bevy LoD device: {info:?}: {error}"
                        );
                        None
                    }
                }
            })
            .collect()
    }

    fn backend_environment_name(backend: wgpu::Backend) -> &'static str {
        match backend {
            wgpu::Backend::Vulkan => "vulkan",
            wgpu::Backend::Metal => "metal",
            wgpu::Backend::Dx12 => "dx12",
            wgpu::Backend::Gl => "gl",
            wgpu::Backend::BrowserWebGpu => "webgpu",
            wgpu::Backend::Noop => "noop",
        }
    }

    fn read_golden_rgba(json_path: &Path, observation: &AdapterGolden) -> Vec<u8> {
        let rgba_path = json_path.with_file_name(&observation.rgba_file);
        let bytes = fs::read(&rgba_path).expect("adapter RGBA golden is readable");
        assert_eq!(bytes.len(), (WIDTH * HEIGHT * 4) as usize);
        assert_eq!(fnv1a64(&bytes), observation.rgba_hash);
        bytes
    }

    fn linear_rgba(bytes: &[u8]) -> Vec<[f32; 4]> {
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

    fn srgb_to_linear(value: u8) -> f32 {
        let value = f32::from(value) / 255.0;
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }
}
