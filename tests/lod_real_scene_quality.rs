#[cfg(not(all(feature = "headless", feature = "testing")))]
#[test]
fn lod_real_scene_quality_requires_headless_and_testing_features() {}

#[cfg(all(feature = "headless", feature = "testing"))]
mod headless {
    use std::{
        collections::BTreeSet,
        env,
        fmt::Write as _,
        fs,
        path::{Path, PathBuf},
        thread,
        time::{Duration, Instant},
    };

    use bevy::{
        app::{AppExit, ScheduleRunnerPlugin},
        asset::{
            AssetMetaCheck, AssetPlugin, DependencyLoadState, LoadState,
            RecursiveDependencyLoadState, UnapprovedPathMode,
        },
        camera::{PerspectiveProjection, Projection, RenderTarget},
        core_pipeline::tonemapping::Tonemapping,
        prelude::*,
        render::{
            render_resource::TextureFormat,
            view::screenshot::{Screenshot, ScreenshotCaptured},
        },
        window::ExitCondition,
        winit::WinitPlugin,
    };
    use bevy_gaussian_splatting::{
        CloudSettings, Gaussian3d, GaussianCamera, GaussianLodBridgeConfig,
        GaussianLodBuildSettings, GaussianLodSettings, GaussianMode, GaussianSplattingPlugin,
        PlanarGaussian3d, PlanarGaussian3dHandle, SphericalHarmonicCoefficients,
        build_planar_3d_lod,
        gaussian::{covariance::compute_covariance_3d, settings::GaussianColorSpace},
        io::{IoPlugin, scene::GaussianScene},
        material::spherical_harmonics::SH_DEGREE,
        sort::SortMode,
        stream::{
            bridge::{GaussianLodBridgePhase, GaussianLodBridgeStatus},
            hierarchy::{
                AllResident, LodHierarchy, LodView, ManifestLodHierarchy, select_frontier,
            },
        },
        testing::{ImageMetrics, LodProjection, LodTestCamera, compare_linear_rgba},
    };
    use sha2::{Digest, Sha256};

    const WIDTH: u32 = 192;
    const HEIGHT: u32 = 192;
    const FOV_Y: f32 = 60.0_f32.to_radians();
    const SYNTHETIC_SOURCE_COUNT: usize = 4_096;
    const REFERENCE_WARMUP_FRAMES: u32 = 45;
    const RESTORED_WARMUP_FRAMES: u32 = 18;
    // Require several active frames so image metrics sample a settled cut.
    const STABLE_ACTIVE_FRAMES: u32 = 30;
    const MAX_FRAMES: u32 = 900;
    const FOREGROUND_ALPHA: f32 = 0.02;
    const SPILL_DILATION_PX: usize = 2;

    const TRELLIS_ENV: &str = "BGS_TRELLIS_GLB";
    const TRELLIS_REPORT_ENV: &str = "BGS_LOD_REPORT_PATH";
    const TRELLIS_AUDIT_PROFILE_ENV: &str = "BGS_TRELLIS_AUDIT_PROFILE";
    const TRELLIS_BYTE_LEN: u64 = 112_899_460;
    const TRELLIS_SPLAT_COUNT: usize = 478_368;
    // Active cuts change by whole hierarchy nodes. Allow one default two-leaf
    // (2 * 64 record) domain of integer granularity beyond the 5% curve bound.
    const TRELLIS_CONTINUITY_DISCRETE_RECORD_SLACK: usize = 128;
    const TRELLIS_SHA256_FOR_PREFLIGHT: &str =
        "fbe9d96b6689a78228c121e5f1bc8c5ccc32cef1941294d25f1db66f4a901dc1";
    const TRELLIS_RASTER_SIZE: u32 = 192;
    const TRELLIS_DEPLOYMENT_VIEWPORT_HEIGHT_PX: f32 = 1_080.0;
    const TRELLIS_FOV_Y: f32 = 45.0_f32.to_radians();
    const TRELLIS_MORPHOLOGY_QUALITIES: [f32; 7] = [0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1.0];
    // At the 192px oracle resolution, an opacity-visible 3-sigma footprint
    // wider than 24px with an 8:1 axis ratio is an obvious needle-like
    // representative rather than sub-pixel covariance noise.
    const VISIBLE_ELONGATION_MIN_MAJOR_SIGMA_PX: f32 = 4.0;
    const VISIBLE_ELONGATION_MIN_ASPECT_RATIO: f32 = 8.0;
    const SYNTHETIC_COARSE_QUALITY: f32 = 0.65;
    const SYNTHETIC_COARSE_THRESHOLDS: QualityThresholds = QualityThresholds {
        minimum_foreground_psnr: 30.0,
        minimum_ssim: 0.95,
        minimum_iou: 0.95,
        maximum_alpha_mae: 0.02,
        maximum_spill: 0.01,
    };
    const Q95_THRESHOLDS: QualityThresholds = QualityThresholds {
        minimum_foreground_psnr: 38.0,
        minimum_ssim: 0.99,
        minimum_iou: 0.99,
        maximum_alpha_mae: 0.005,
        maximum_spill: 0.005,
    };
    const Q99_THRESHOLDS: QualityThresholds = QualityThresholds {
        minimum_foreground_psnr: 40.0,
        minimum_ssim: 0.995,
        minimum_iou: 0.995,
        maximum_alpha_mae: 0.003,
        maximum_spill: 0.003,
    };

    /// Exercises the production GPU path with thin, curved diagonal ribbons. The
    /// source splats have strongly oriented covariance, so a transposed or
    /// over-inflated coarse covariance produces visible pixels perpendicular to
    /// the ribbons and fails the spill gate even when a whole-image PSNR is
    /// diluted by the transparent background.
    #[test]
    fn diagonal_ribbons_keep_high_quality_covariance_on_the_gpu() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!(
                "skipping real-pattern LoD GPU regression; set RUN_GPU_RENDER_TESTS=1 to enable"
            );
            return;
        }

        let mut app = App::new();
        app.insert_resource(ClearColor(Color::linear_rgba(0.0, 0.0, 0.0, 0.0)))
            .insert_resource(synthetic_bridge_config())
            .insert_resource(GpuQualityState::default());
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: "assets".to_owned(),
                    processed_file_path: "assets".to_owned(),
                    meta_check: AssetMetaCheck::Never,
                    unapproved_path_mode: UnapprovedPathMode::Allow,
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
        app.add_plugins(GaussianSplattingPlugin)
            .add_systems(Startup, setup_gpu_fixture)
            .add_systems(Update, drive_gpu_capture)
            .add_observer(on_gpu_capture);
        app.run();
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum GpuPhase {
        ReferenceWarmup,
        ReferencePending,
        CoarseWaiting,
        CoarsePending,
        Quality95Waiting,
        Quality95Pending,
        Quality99Waiting,
        Quality99Pending,
        RestoredWaiting,
        RestoredPending,
    }

    #[derive(Resource)]
    struct GpuQualityState {
        phase: GpuPhase,
        phase_frames: u32,
        total_frames: u32,
        stable_frames: u32,
        target: Option<Handle<Image>>,
        cloud: Option<Entity>,
        reference: Option<Vec<[f32; 4]>>,
        coarse: Option<GpuCapture>,
        quality95: Option<GpuCapture>,
        quality99: Option<GpuCapture>,
        pending_active_gaussians: u64,
    }

    impl Default for GpuQualityState {
        fn default() -> Self {
            Self {
                phase: GpuPhase::ReferenceWarmup,
                phase_frames: 0,
                total_frames: 0,
                stable_frames: 0,
                target: None,
                cloud: None,
                reference: None,
                coarse: None,
                quality95: None,
                quality99: None,
                pending_active_gaussians: 0,
            }
        }
    }

    impl GpuQualityState {
        fn enter(&mut self, phase: GpuPhase) {
            self.phase = phase;
            self.phase_frames = 0;
            self.stable_frames = 0;
        }
    }

    struct GpuCapture {
        active_gaussians: u64,
        image: Vec<[f32; 4]>,
    }

    fn synthetic_bridge_config() -> GaussianLodBridgeConfig {
        GaussianLodBridgeConfig {
            max_ephemeral_source_gaussians: 8_192,
            max_ephemeral_stored_gaussians: 16_384,
            max_atlas_gaussians: 16_384,
            max_atlas_bytes: 64 * 1024 * 1024,
            build_settings: GaussianLodBuildSettings {
                branching_factor: 8,
                leaf_capacity: 128,
                support_sigma: 3.0,
            },
            ..default()
        }
    }

    fn synthetic_lod_settings() -> GaussianLodSettings {
        let mut settings = GaussianLodSettings {
            quality: 1.0,
            hysteresis: 0.0,
            frustum_culling: false,
            ..default()
        };
        settings.budgets.max_active_gaussians = 8_192;
        settings.budgets.max_resident_gaussians = 16_384;
        settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
        settings.budgets.max_resident_pages = 2_048;
        settings.budgets.max_pending_requests = 2_048;
        settings.budgets.max_requests_per_frame = 512;
        settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
        settings
    }

    fn setup_gpu_fixture(
        mut commands: Commands,
        mut state: ResMut<GpuQualityState>,
        mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let cloud = diagonal_ribbon_cloud();
        assert_eq!(cloud.position_visibility.len(), SYNTHETIC_SOURCE_COUNT);
        let target = images.add(Image::new_target_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let cloud = commands
            .spawn((
                PlanarGaussian3dHandle(gaussian_assets.add(cloud)),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::Radix,
                    global_opacity: 1.0,
                    global_scale: 1.0,
                    opacity_adaptive_radius: false,
                    ..default()
                },
                synthetic_lod_settings(),
                Transform::IDENTITY,
                Visibility::Visible,
                Name::new("lod_covariance_diagonal_ribbons"),
            ))
            .id();
        commands.spawn((
            Camera3d::default(),
            Camera::default(),
            Projection::Perspective(PerspectiveProjection {
                fov: FOV_Y,
                near: 0.01,
                far: 100.0,
                ..default()
            }),
            RenderTarget::Image(target.clone().into()),
            Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)),
            Tonemapping::None,
            GaussianCamera::default(),
            Name::new("lod_covariance_ribbon_camera"),
        ));
        state.target = Some(target);
        state.cloud = Some(cloud);
    }

    fn drive_gpu_capture(
        mut commands: Commands,
        mut state: ResMut<GpuQualityState>,
        statuses: Query<&GaussianLodBridgeStatus>,
    ) {
        state.total_frames += 1;
        state.phase_frames += 1;
        assert!(
            state.total_frames <= MAX_FRAMES,
            "real-pattern LoD GPU regression timed out in {:?}; status={:?}",
            state.phase,
            state.cloud.and_then(|cloud| statuses.get(cloud).ok())
        );

        let cloud = state.cloud.expect("ribbon cloud exists");
        match state.phase {
            GpuPhase::ReferenceWarmup if state.phase_frames >= REFERENCE_WARMUP_FRAMES => {
                assert!(
                    statuses.get(cloud).is_err(),
                    "quality one unexpectedly retained a LoD bridge"
                );
                request_gpu_capture(&mut commands, &state);
                state.enter(GpuPhase::ReferencePending);
            }
            GpuPhase::CoarseWaiting | GpuPhase::Quality95Waiting | GpuPhase::Quality99Waiting => {
                let Ok(status) = statuses.get(cloud) else {
                    state.stable_frames = 0;
                    return;
                };
                assert!(status.failure.is_none(), "LoD bridge failed: {status:?}");
                if status.phase == GaussianLodBridgePhase::Active && status.active_gaussians > 0 {
                    state.pending_active_gaussians = status.active_gaussians;
                    state.stable_frames += 1;
                } else {
                    state.stable_frames = 0;
                }
                if state.stable_frames >= STABLE_ACTIVE_FRAMES {
                    request_gpu_capture(&mut commands, &state);
                    let pending = match state.phase {
                        GpuPhase::CoarseWaiting => GpuPhase::CoarsePending,
                        GpuPhase::Quality95Waiting => GpuPhase::Quality95Pending,
                        GpuPhase::Quality99Waiting => GpuPhase::Quality99Pending,
                        _ => unreachable!(),
                    };
                    state.enter(pending);
                }
            }
            GpuPhase::RestoredWaiting => {
                if statuses.get(cloud).is_err() {
                    state.stable_frames += 1;
                } else {
                    state.stable_frames = 0;
                }
                if state.stable_frames >= RESTORED_WARMUP_FRAMES {
                    request_gpu_capture(&mut commands, &state);
                    state.enter(GpuPhase::RestoredPending);
                }
            }
            GpuPhase::ReferenceWarmup
            | GpuPhase::ReferencePending
            | GpuPhase::CoarsePending
            | GpuPhase::Quality95Pending
            | GpuPhase::Quality99Pending
            | GpuPhase::RestoredPending => {}
        }
    }

    fn request_gpu_capture(commands: &mut Commands, state: &GpuQualityState) {
        commands.spawn(Screenshot::image(
            state.target.clone().expect("render target exists"),
        ));
    }

    fn on_gpu_capture(
        trigger: On<ScreenshotCaptured>,
        mut state: ResMut<GpuQualityState>,
        mut settings: Query<&mut GaussianLodSettings>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let image = linear_rgba(
            trigger
                .image
                .clone()
                .try_into_dynamic()
                .expect("screenshot converts")
                .to_rgba8()
                .as_raw(),
        );
        assert!(
            image.iter().any(|pixel| pixel[3] > FOREGROUND_ALPHA),
            "real-pattern LoD capture stayed transparent in {:?}",
            state.phase
        );
        let cloud = state.cloud.expect("ribbon cloud exists");
        match state.phase {
            GpuPhase::ReferencePending => {
                state.reference = Some(image);
                settings
                    .get_mut(cloud)
                    .expect("ribbon LoD settings exist")
                    .quality = SYNTHETIC_COARSE_QUALITY;
                state.enter(GpuPhase::CoarseWaiting);
            }
            GpuPhase::CoarsePending => {
                let active_gaussians = state.pending_active_gaussians;
                assert!(
                    active_gaussians < SYNTHETIC_SOURCE_COUNT as u64,
                    "q={SYNTHETIC_COARSE_QUALITY:.2} did not activate a material LoD cut: active={active_gaussians}, source={SYNTHETIC_SOURCE_COUNT}"
                );
                state.coarse = Some(GpuCapture {
                    active_gaussians,
                    image,
                });
                settings
                    .get_mut(cloud)
                    .expect("ribbon LoD settings exist")
                    .quality = 0.95;
                state.enter(GpuPhase::Quality95Waiting);
            }
            GpuPhase::Quality95Pending => {
                let active_gaussians = state.pending_active_gaussians;
                assert!(
                    active_gaussians <= SYNTHETIC_SOURCE_COUNT as u64,
                    "q=.95 exceeded the exact source count: active={active_gaussians}, source={SYNTHETIC_SOURCE_COUNT}"
                );
                state.quality95 = Some(GpuCapture {
                    active_gaussians,
                    image,
                });
                settings
                    .get_mut(cloud)
                    .expect("ribbon LoD settings exist")
                    .quality = 0.99;
                state.enter(GpuPhase::Quality99Waiting);
            }
            GpuPhase::Quality99Pending => {
                let active_gaussians = state.pending_active_gaussians;
                state.quality99 = Some(GpuCapture {
                    active_gaussians,
                    image,
                });
                assert_gpu_quality(&state);
                settings
                    .get_mut(cloud)
                    .expect("ribbon LoD settings exist")
                    .quality = 1.0;
                state.enter(GpuPhase::RestoredWaiting);
            }
            GpuPhase::RestoredPending => {
                let reference = state.reference.as_ref().expect("reference exists");
                let restored = compare_linear_rgba(reference, &image, FOREGROUND_ALPHA)
                    .expect("restored reference dimensions match");
                assert!(
                    restored.psnr_rgb.is_infinite() && restored.max_abs_error == 0.0,
                    "q=1 did not restore the exact source pixels: {restored:?}"
                );
                exit.write(AppExit::Success);
            }
            phase => panic!("screenshot captured during unexpected phase {phase:?}"),
        }
    }

    fn assert_gpu_quality(state: &GpuQualityState) {
        let reference = state.reference.as_ref().expect("reference exists");
        let coarse = state.coarse.as_ref().expect("coarse capture exists");
        let q95 = state.quality95.as_ref().expect("q=.95 capture exists");
        let q99 = state.quality99.as_ref().expect("q=.99 capture exists");
        assert!(coarse.active_gaussians < SYNTHETIC_SOURCE_COUNT as u64);
        assert!(q95.active_gaussians >= coarse.active_gaussians);
        assert!(q99.active_gaussians >= q95.active_gaussians);

        let coarse_observation = quality_observation(reference, &coarse.image, WIDTH, HEIGHT);
        let q95_observation = quality_observation(reference, &q95.image, WIDTH, HEIGHT);
        let q99_observation = quality_observation(reference, &q99.image, WIDTH, HEIGHT);
        eprintln!(
            "diagonal-ribbon LoD quality: q{SYNTHETIC_COARSE_QUALITY:.2} active={} {:?}; q95 active={} {:?}; q99 active={} {:?}",
            coarse.active_gaussians,
            coarse_observation,
            q95.active_gaussians,
            q95_observation,
            q99.active_gaussians,
            q99_observation
        );
        assert_thresholds(
            "synthetic coarse cut",
            coarse_observation,
            SYNTHETIC_COARSE_THRESHOLDS,
        );
        assert_alpha_morphology_bound("synthetic coarse cut", coarse_observation);
        assert_thresholds("synthetic q=.95", q95_observation, Q95_THRESHOLDS);
        assert_thresholds("synthetic q=.99", q99_observation, Q99_THRESHOLDS);
        assert_monotonic(
            "synthetic coarse -> q=.95",
            coarse_observation,
            q95_observation,
        );
        assert_monotonic("synthetic q=.95 -> q=.99", q95_observation, q99_observation);
    }

    fn diagonal_ribbon_cloud() -> PlanarGaussian3d {
        const RIBBONS: usize = 2;
        const LANES: usize = 4;
        const SAMPLES: usize = SYNTHETIC_SOURCE_COUNT / (RIBBONS * LANES);
        let mut gaussians = Vec::with_capacity(SYNTHETIC_SOURCE_COUNT);
        for ribbon in 0..RIBBONS {
            for lane in 0..LANES {
                for sample in 0..SAMPLES {
                    let t = sample as f32 / (SAMPLES - 1) as f32;
                    let x = -1.55 + 3.10 * t;
                    let phase = if ribbon == 0 { 0.0 } else { 1.7 };
                    let offset = if ribbon == 0 { -0.24 } else { 0.24 };
                    let y = 0.48 * x + offset + 0.13 * (t * 8.0 + phase).sin();
                    let tangent =
                        Vec2::new(3.10, 1.488 + 1.04 * (t * 8.0 + phase).cos()).normalize_or_zero();
                    let normal = Vec2::new(-tangent.y, tangent.x);
                    let lane_offset = (lane as f32 - 1.5) * 0.009;
                    let position = Vec3::new(
                        x + normal.x * lane_offset,
                        y + normal.y * lane_offset,
                        (ribbon as f32 - 0.5) * 0.006,
                    );
                    let angle = tangent.y.atan2(tangent.x);
                    let rotation = Quat::from_rotation_z(angle);
                    let color = if ribbon == 0 {
                        [0.95, 0.08, 0.04]
                    } else {
                        [0.04, 0.45, 0.95]
                    };
                    gaussians.push(test_gaussian(
                        position,
                        Vec3::new(0.020, 0.0055, 0.003),
                        rotation,
                        color,
                        0.72,
                    ));
                }
            }
        }
        gaussians.into()
    }

    fn test_gaussian(
        position: Vec3,
        scale: Vec3,
        rotation: Quat,
        color: [f32; 3],
        opacity: f32,
    ) -> Gaussian3d {
        let mut coefficients = SphericalHarmonicCoefficients::default();
        for (index, value) in color.into_iter().enumerate() {
            coefficients.set(index, value);
        }
        Gaussian3d {
            position_visibility: [position.x, position.y, position.z, 1.0].into(),
            // Canonical Gaussian rotations store the scalar component first.
            rotation: [rotation.w, rotation.x, rotation.y, rotation.z].into(),
            scale_opacity: [scale.x, scale.y, scale.z, opacity].into(),
            spherical_harmonic: coefficients,
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum TrellisAuditProfile {
        Pr,
        Full,
    }

    impl TrellisAuditProfile {
        fn from_environment() -> Self {
            match env::var(TRELLIS_AUDIT_PROFILE_ENV) {
                Ok(value) if value == "pr" => Self::Pr,
                Ok(value) if value == "full" => Self::Full,
                Ok(value) => panic!(
                    "unsupported {TRELLIS_AUDIT_PROFILE_ENV}={value:?}; expected `pr` or `full`"
                ),
                Err(env::VarError::NotPresent) => Self::Pr,
                Err(env::VarError::NotUnicode(_)) => {
                    panic!("{TRELLIS_AUDIT_PROFILE_ENV} must be valid Unicode")
                }
            }
        }

        const fn as_str(self) -> &'static str {
            match self {
                Self::Pr => "pr",
                Self::Full => "full",
            }
        }

        fn rendered_qualities(self) -> Vec<f32> {
            let steps = match self {
                Self::Pr => 20,
                Self::Full => 100,
            };
            let mut qualities = (0..=steps)
                .map(|step| step as f32 / steps as f32)
                .collect::<Vec<_>>();
            qualities.push(0.99);
            qualities.sort_by(f32::total_cmp);
            qualities.dedup_by(|left, right| (*left - *right).abs() <= 1e-6);
            qualities
        }

        fn orbit_specs(self) -> &'static [TrellisOrbitSpec] {
            match self {
                Self::Pr => &PR_TRELLIS_ORBIT_SPECS,
                Self::Full => &FULL_TRELLIS_ORBIT_SPECS,
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct TrellisOrbitSpec {
        label: &'static str,
        yaw_degrees: f32,
        pitch_degrees: f32,
    }

    const PR_TRELLIS_ORBIT_SPECS: [TrellisOrbitSpec; 2] = [
        TrellisOrbitSpec {
            label: "orbit-left-40",
            yaw_degrees: -40.0,
            pitch_degrees: 0.0,
        },
        TrellisOrbitSpec {
            label: "orbit-right-40",
            yaw_degrees: 40.0,
            pitch_degrees: 0.0,
        },
    ];

    const FULL_TRELLIS_ORBIT_SPECS: [TrellisOrbitSpec; 6] = [
        PR_TRELLIS_ORBIT_SPECS[0],
        PR_TRELLIS_ORBIT_SPECS[1],
        TrellisOrbitSpec {
            label: "orbit-left-75",
            yaw_degrees: -75.0,
            pitch_degrees: 0.0,
        },
        TrellisOrbitSpec {
            label: "orbit-right-75",
            yaw_degrees: 75.0,
            pitch_degrees: 0.0,
        },
        TrellisOrbitSpec {
            label: "orbit-high-25",
            yaw_degrees: 0.0,
            pitch_degrees: 25.0,
        },
        TrellisOrbitSpec {
            label: "orbit-low-25",
            yaw_degrees: 0.0,
            pitch_degrees: -25.0,
        },
    ];

    /// Full-scene color and covariance audit for the documented Trellis
    /// artifact. It is ignored by default and accepts only a caller-supplied
    /// local path. The test contains no URL and performs no download; CI can
    /// provision the immutable artifact through a non-required cache or
    /// workflow-dispatch step.
    #[test]
    #[ignore = "requires the canonical local Trellis GLB via BGS_TRELLIS_GLB"]
    fn canonical_trellis_high_quality_color_and_covariance_audit() {
        let profile = TrellisAuditProfile::from_environment();
        let path = PathBuf::from(env::var_os(TRELLIS_ENV).unwrap_or_else(|| {
            panic!(
                "set {TRELLIS_ENV} to the canonical local Trellis GLB (sha256 {TRELLIS_SHA256_FOR_PREFLIGHT})"
            )
        }));
        verify_trellis_provenance(&path);
        let loaded = load_local_gaussian_scene(&path);
        assert_eq!(loaded.cloud.position_visibility.len(), TRELLIS_SPLAT_COUNT);

        let lod = build_planar_3d_lod(&loaded.cloud, GaussianLodBuildSettings::default())
            .expect("canonical Trellis hierarchy builds with the production default profile");
        assert_eq!(
            lod.manifest.build.settings,
            GaussianLodBuildSettings::default()
        );
        assert!(
            lod.manifest
                .nodes
                .iter()
                .filter(|node| node.is_leaf())
                .all(|node| node.source.count <= 64),
            "the production progressive profile must retain fine logical leaves"
        );
        assert!(
            lod.manifest
                .pages
                .iter()
                .all(|page| page.gaussian_count <= 1_024),
            "fine logical nodes must remain packed into bounded physical pages"
        );
        assert!(lod.manifest.header.page_count < lod.manifest.header.node_count);
        let hierarchy = ManifestLodHierarchy::new(&lod.manifest)
            .expect("canonical Trellis hierarchy manifest is valid");
        let mut certificates = lod
            .manifest
            .nodes
            .iter()
            .filter(|node| !node.is_leaf())
            .map(|node| node.high_fidelity_certificate)
            .collect::<Vec<_>>();
        certificates.sort_by(f32::total_cmp);
        let quantile = |fraction: f32| {
            certificates[((certificates.len() - 1) as f32 * fraction).round() as usize]
        };
        eprintln!(
            "Trellis internal certificate distribution: count={} min={:.6} p50={:.6} p90={:.6} p95={:.6} p99={:.6} max={:.6}; admitted={:?}",
            certificates.len(),
            quantile(0.0),
            quantile(0.5),
            quantile(0.9),
            quantile(0.95),
            quantile(0.99),
            quantile(1.0),
            [0.0001, 0.001, 0.01, 0.05, 0.1].map(|threshold| certificates
                .iter()
                .filter(|value| **value >= threshold)
                .count())
        );
        let rendered_qualities = profile.rendered_qualities();
        let selection_qualities = trellis_dense_selection_qualities();
        let mut distance_sweeps = Vec::with_capacity(3);
        let distance_cameras = trellis_distance_cameras(&loaded, &hierarchy);
        for audit_camera in distance_cameras {
            let exact = real_scene_quality_sample(
                &lod,
                &hierarchy,
                audit_camera.camera,
                1.0,
                loaded.color_space,
            );
            let rendered = rendered_qualities
                .iter()
                .copied()
                .map(|quality| {
                    if quality_eq(quality, 1.0) {
                        quality_sweep_point(quality, &exact, &exact, audit_camera.camera)
                    } else {
                        let sample = real_scene_quality_sample(
                            &lod,
                            &hierarchy,
                            audit_camera.camera,
                            quality,
                            loaded.color_space,
                        );
                        quality_sweep_point(quality, &sample, &exact, audit_camera.camera)
                    }
                })
                .collect();
            let deployment_selection = selection_qualities
                .iter()
                .copied()
                .map(|quality| SelectionSweepPoint {
                    quality,
                    active_gaussians: real_scene_active_count(
                        &lod,
                        &hierarchy,
                        audit_camera.camera,
                        quality,
                        TRELLIS_DEPLOYMENT_VIEWPORT_HEIGHT_PX,
                    ),
                })
                .collect();
            distance_sweeps.push(TrellisDistanceSweep {
                camera: audit_camera,
                rendered,
                deployment_selection,
            });
        }

        let authored_camera = trellis_authored_camera(&loaded);
        let authored_sweep =
            real_scene_morphology_sweep(&lod, &hierarchy, authored_camera, loaded.color_space);
        let orbit_sweeps = trellis_orbit_cameras(profile, distance_cameras[1].camera)
            .into_iter()
            .map(|audit_camera| TrellisMorphologySweep {
                label: audit_camera.label,
                rendered: real_scene_morphology_sweep(
                    &lod,
                    &hierarchy,
                    audit_camera.camera,
                    loaded.color_space,
                ),
            })
            .collect::<Vec<_>>();

        let report =
            trellis_quality_report(profile, &distance_sweeps, &authored_sweep, &orbit_sweeps);
        eprintln!("BGS_TRELLIS_LOD_REPORT_BEGIN\n{report}BGS_TRELLIS_LOD_REPORT_END");
        if let Some(path) = env::var_os(TRELLIS_REPORT_ENV) {
            let path = PathBuf::from(path);
            fs::write(&path, &report).unwrap_or_else(|error| {
                panic!(
                    "failed to write Trellis LoD report {}: {error}",
                    path.display()
                )
            });
        }

        assert_trellis_distance_graph(&distance_sweeps);
        assert_trellis_authored_safety(&authored_sweep);
        for sweep in &orbit_sweeps {
            assert_trellis_morphology_safety(sweep.label, &sweep.rendered);
        }
    }

    fn verify_trellis_provenance(path: &Path) {
        let bytes = fs::read(path).unwrap_or_else(|error| {
            panic!(
                "failed to read local Trellis GLB {}: {error}",
                path.display()
            )
        });
        assert_eq!(
            bytes.len() as u64,
            TRELLIS_BYTE_LEN,
            "unexpected Trellis byte length; preflight sha256 must be {TRELLIS_SHA256_FOR_PREFLIGHT}"
        );
        let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(
            actual_sha256, TRELLIS_SHA256_FOR_PREFLIGHT,
            "canonical Trellis SHA-256 drifted"
        );
        let gltf = gltf::Gltf::from_slice_without_validation(&bytes)
            .expect("local Trellis artifact parses as GLB");
        let position_counts = gltf
            .document
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .filter_map(|primitive| primitive.get(&gltf::Semantic::Positions))
            .map(|accessor| accessor.count())
            .collect::<Vec<_>>();
        assert_eq!(
            position_counts,
            [TRELLIS_SPLAT_COUNT],
            "canonical Trellis must contain exactly one 478,368-splat primitive"
        );
    }

    struct LoadedRealScene {
        cloud: PlanarGaussian3d,
        cloud_transform: Transform,
        authored_camera: Transform,
        color_space: GaussianColorSpace,
    }

    fn load_local_gaussian_scene(path: &Path) -> LoadedRealScene {
        let canonical = fs::canonicalize(path)
            .unwrap_or_else(|error| panic!("failed to canonicalize {}: {error}", path.display()));
        let root = canonical.parent().expect("Trellis path has a parent");
        let file_name = canonical
            .file_name()
            .expect("Trellis path has a file name")
            .to_string_lossy()
            .into_owned();

        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(AssetPlugin {
            file_path: root.display().to_string(),
            processed_file_path: root.display().to_string(),
            meta_check: AssetMetaCheck::Never,
            unapproved_path_mode: UnapprovedPathMode::Allow,
            ..default()
        });
        app.init_asset::<PlanarGaussian3d>();
        app.add_plugins(IoPlugin);

        let handle: Handle<GaussianScene> = app.world().resource::<AssetServer>().load(file_name);
        let deadline = Instant::now() + Duration::from_secs(60);
        let scene = loop {
            app.update();
            if let Some((load, dependency, recursive)) = app
                .world()
                .resource::<AssetServer>()
                .get_load_states(&handle)
            {
                match (&load, &dependency, &recursive) {
                    (LoadState::Failed(error), _, _)
                    | (_, DependencyLoadState::Failed(error), _)
                    | (_, _, RecursiveDependencyLoadState::Failed(error)) => {
                        panic!("local Trellis asset load failed: {error}")
                    }
                    (LoadState::Loaded, _, RecursiveDependencyLoadState::Loaded) => {
                        if let Some(scene) = app
                            .world()
                            .resource::<Assets<GaussianScene>>()
                            .get(&handle)
                            .cloned()
                        {
                            break scene;
                        }
                    }
                    _ => {}
                }
            }
            assert!(
                Instant::now() < deadline,
                "timed out loading local Trellis asset"
            );
            thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(scene.bundles.len(), 1, "Trellis primitive count drifted");
        assert_eq!(scene.cameras.len(), 1, "Trellis authored camera drifted");
        let bundle = &scene.bundles[0];
        let cloud = app
            .world()
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&bundle.cloud)
            .cloned()
            .expect("Trellis cloud dependency is loaded");
        LoadedRealScene {
            cloud,
            cloud_transform: bundle.transform,
            authored_camera: scene.cameras[0].transform,
            color_space: bundle.settings.color_space,
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct TrellisDistanceCamera {
        label: &'static str,
        root_coverage: f32,
        distance_in_root_radii: f32,
        camera: LodTestCamera,
    }

    #[derive(Clone, Copy, Debug)]
    struct TrellisOrbitCamera {
        label: &'static str,
        camera: LodTestCamera,
    }

    fn trellis_authored_camera(scene: &LoadedRealScene) -> LodTestCamera {
        let world_from_cloud = scene.cloud_transform.to_matrix();
        let cloud_from_world = world_from_cloud.inverse();
        let authored_position =
            cloud_from_world.transform_point3(scene.authored_camera.translation);
        let authored_target = cloud_from_world.transform_point3(
            scene.authored_camera.translation + scene.authored_camera.rotation * Vec3::NEG_Z,
        );
        let authored_up = cloud_from_world
            .transform_vector3(scene.authored_camera.rotation * Vec3::Y)
            .normalize_or_zero();
        LodTestCamera {
            position: authored_position,
            target: authored_target,
            up: authored_up,
            projection: LodProjection::Perspective {
                vertical_fov_radians: TRELLIS_FOV_Y,
            },
            near: 0.01,
            far: 1_000.0,
            viewport: [TRELLIS_RASTER_SIZE, TRELLIS_RASTER_SIZE],
        }
    }

    fn trellis_distance_cameras(
        scene: &LoadedRealScene,
        hierarchy: &ManifestLodHierarchy<'_>,
    ) -> [TrellisDistanceCamera; 3] {
        let [root] = hierarchy.roots() else {
            panic!(
                "canonical Trellis quality graph requires one root, found {}",
                hierarchy.roots().len()
            )
        };
        let metrics = hierarchy
            .metrics(*root)
            .expect("canonical Trellis root metrics exist");
        let radius = metrics.radius;
        assert!(radius.is_finite() && radius > 0.0);
        let tangent = (0.5 * TRELLIS_FOV_Y).tan();
        let cloud_from_world = scene.cloud_transform.to_matrix().inverse();
        let forward = cloud_from_world
            .transform_vector3(scene.authored_camera.rotation * Vec3::NEG_Z)
            .normalize_or_zero();
        let authored_up = cloud_from_world
            .transform_vector3(scene.authored_camera.rotation * Vec3::Y)
            .normalize_or_zero();
        let right = forward.cross(authored_up).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        assert!(forward != Vec3::ZERO && right != Vec3::ZERO && up != Vec3::ZERO);
        [("near", 0.8_f32), ("mid", 0.4_f32), ("far", 0.2_f32)].map(|(label, root_coverage)| {
            // LodView uses distance to the support surface rather than the root
            // center. Solve C = R / (tan(fov / 2) * (D - R)) for D so the
            // named scales have an exact, scene-invariant projected coverage.
            let distance = radius + radius / (root_coverage * tangent);
            let camera = LodTestCamera {
                position: metrics.center - forward * distance,
                target: metrics.center,
                up,
                projection: LodProjection::Perspective {
                    vertical_fov_radians: TRELLIS_FOV_Y,
                },
                near: 0.001,
                far: distance + radius * 4.0,
                viewport: [TRELLIS_RASTER_SIZE, TRELLIS_RASTER_SIZE],
            };
            let actual_coverage = trellis_lod_view(camera, TRELLIS_DEPLOYMENT_VIEWPORT_HEIGHT_PX)
                .projected_coverage(metrics);
            assert!(
                (actual_coverage - root_coverage).abs() <= 1e-5,
                "{label} root coverage drifted: requested={root_coverage}, actual={actual_coverage}"
            );
            TrellisDistanceCamera {
                label,
                root_coverage,
                distance_in_root_radii: distance / radius,
                camera,
            }
        })
    }

    fn trellis_orbit_cameras(
        profile: TrellisAuditProfile,
        mid_camera: LodTestCamera,
    ) -> Vec<TrellisOrbitCamera> {
        let target = mid_camera.target;
        let forward = (target - mid_camera.position).normalize_or_zero();
        let right = forward.cross(mid_camera.up).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let distance = mid_camera.position.distance(target);
        assert!(
            forward != Vec3::ZERO
                && right != Vec3::ZERO
                && up != Vec3::ZERO
                && distance.is_finite()
                && distance > 0.0
        );

        profile
            .orbit_specs()
            .iter()
            .copied()
            .map(|spec| {
                let yaw = Quat::from_axis_angle(up, spec.yaw_degrees.to_radians());
                let yawed_right = yaw * right;
                let pitch = Quat::from_axis_angle(yawed_right, spec.pitch_degrees.to_radians());
                let outward = (pitch * (yaw * -forward)).normalize_or_zero();
                let up_hint = (pitch * up).normalize_or_zero();
                let orbit_forward = -outward;
                let orbit_right = orbit_forward.cross(up_hint).normalize_or_zero();
                let orbit_up = orbit_right.cross(orbit_forward).normalize_or_zero();
                assert!(
                    outward != Vec3::ZERO && orbit_right != Vec3::ZERO && orbit_up != Vec3::ZERO
                );
                TrellisOrbitCamera {
                    label: spec.label,
                    camera: LodTestCamera {
                        position: target + outward * distance,
                        target,
                        up: orbit_up,
                        ..mid_camera
                    },
                }
            })
            .collect()
    }

    struct RealSceneQualitySample {
        active_gaussians: usize,
        achieved_max_error_px: f32,
        image: Vec<[f32; 4]>,
        covariance: CovarianceDiagnostics,
    }

    fn real_scene_quality_sample(
        lod: &bevy_gaussian_splatting::PlanarGaussian3dLod,
        hierarchy: &ManifestLodHierarchy<'_>,
        camera: LodTestCamera,
        quality: f32,
        color_space: GaussianColorSpace,
    ) -> RealSceneQualitySample {
        let settings = trellis_lod_settings(lod, quality);
        let metric_view = trellis_lod_view(camera, camera.viewport[1] as f32);
        let frontier = select_frontier(hierarchy, &AllResident, metric_view, &settings)
            .expect("Trellis quality selection succeeds");
        let gaussians =
            bevy_gaussian_splatting::testing::gather_frontier_gaussians(lod, &frontier.nodes)
                .expect("Trellis frontier resolves");
        assert_eq!(
            gaussians.len() as u64,
            frontier.status.active_gaussians,
            "Trellis gathered frontier count differs from selector status"
        );
        let (image, covariance) = render_color_and_covariance_image(
            &gaussians,
            camera,
            camera.viewport[0],
            camera.viewport[1],
            color_space,
        );
        RealSceneQualitySample {
            active_gaussians: gaussians.len(),
            achieved_max_error_px: frontier.status.achieved_max_error_px,
            image,
            covariance,
        }
    }

    fn real_scene_morphology_sweep(
        lod: &bevy_gaussian_splatting::PlanarGaussian3dLod,
        hierarchy: &ManifestLodHierarchy<'_>,
        camera: LodTestCamera,
        color_space: GaussianColorSpace,
    ) -> Vec<QualitySweepPoint> {
        let exact = real_scene_quality_sample(lod, hierarchy, camera, 1.0, color_space);
        TRELLIS_MORPHOLOGY_QUALITIES
            .into_iter()
            .map(|quality| {
                if quality_eq(quality, 1.0) {
                    quality_sweep_point(quality, &exact, &exact, camera)
                } else {
                    let sample =
                        real_scene_quality_sample(lod, hierarchy, camera, quality, color_space);
                    quality_sweep_point(quality, &sample, &exact, camera)
                }
            })
            .collect()
    }

    fn real_scene_active_count(
        lod: &bevy_gaussian_splatting::PlanarGaussian3dLod,
        hierarchy: &ManifestLodHierarchy<'_>,
        camera: LodTestCamera,
        quality: f32,
        selection_viewport_height_px: f32,
    ) -> usize {
        let settings = trellis_lod_settings(lod, quality);
        let view = trellis_lod_view(camera, selection_viewport_height_px);
        let frontier = select_frontier(hierarchy, &AllResident, view, &settings)
            .expect("Trellis dense quality selection succeeds");
        usize::try_from(frontier.status.active_gaussians)
            .expect("Trellis active count fits host usize")
    }

    fn trellis_lod_settings(
        lod: &bevy_gaussian_splatting::PlanarGaussian3dLod,
        quality: f32,
    ) -> GaussianLodSettings {
        let mut settings = GaussianLodSettings {
            quality,
            hysteresis: 0.0,
            frustum_culling: false,
            ..default()
        };
        settings.budgets.max_active_gaussians = TRELLIS_SPLAT_COUNT as u64 * 2;
        settings.budgets.max_traversal_nodes_per_view = lod.manifest.header.node_count.max(1);
        settings
    }

    fn trellis_lod_view(camera: LodTestCamera, viewport_height_px: f32) -> LodView {
        LodView::perspective(
            camera.position,
            viewport_height_px,
            match camera.projection {
                LodProjection::Perspective {
                    vertical_fov_radians,
                } => vertical_fov_radians,
                LodProjection::Orthographic { .. } => unreachable!(),
            },
            camera.near,
        )
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct CovarianceDiagnostics {
        maximum_major_sigma_px: f32,
        maximum_aspect_ratio: f32,
        projected_splats: usize,
        extreme_projected_splats: usize,
        visible_elongated_splats: usize,
    }

    #[derive(Clone, Copy)]
    struct ProjectedCovariance {
        center: Vec2,
        depth: f32,
        covariance: Vec3,
        opacity: f32,
        linear_rgb: Vec3,
    }

    // Keep these values, basis order, and expressions equivalent to
    // `material/spherical_harmonics.wgsl`. Coefficients are stored as one RGB
    // triplet per basis function, not as three channel-major bands.
    const SH_BASIS_CONSTANTS: [f32; 16] = [
        0.282_094_8,
        -0.488_602_52,
        0.488_602_52,
        -0.488_602_52,
        1.092_548_5,
        -1.092_548_5,
        0.315_391_57,
        -1.092_548_5,
        0.546_274_24,
        -0.590_043_6,
        2.890_611_4,
        -0.457_045_8,
        0.373_176_34,
        -0.457_045_8,
        1.445_305_7,
        -0.590_043_6,
    ];

    fn spherical_harmonic_basis(ray_direction: Vec3) -> [f32; 16] {
        let squared = ray_direction * ray_direction;
        [
            SH_BASIS_CONSTANTS[0],
            SH_BASIS_CONSTANTS[1] * ray_direction.y,
            SH_BASIS_CONSTANTS[2] * ray_direction.z,
            SH_BASIS_CONSTANTS[3] * ray_direction.x,
            SH_BASIS_CONSTANTS[4] * ray_direction.x * ray_direction.y,
            SH_BASIS_CONSTANTS[5] * ray_direction.y * ray_direction.z,
            SH_BASIS_CONSTANTS[6] * (2.0 * squared.z - squared.x - squared.y),
            SH_BASIS_CONSTANTS[7] * ray_direction.x * ray_direction.z,
            SH_BASIS_CONSTANTS[8] * (squared.x - squared.y),
            SH_BASIS_CONSTANTS[9] * ray_direction.y * (3.0 * squared.x - squared.y),
            SH_BASIS_CONSTANTS[10] * ray_direction.x * ray_direction.y * ray_direction.z,
            SH_BASIS_CONSTANTS[11] * ray_direction.y * (4.0 * squared.z - squared.x - squared.y),
            SH_BASIS_CONSTANTS[12]
                * ray_direction.z
                * (2.0 * squared.z - 3.0 * squared.x - 3.0 * squared.y),
            SH_BASIS_CONSTANTS[13] * ray_direction.x * (4.0 * squared.z - squared.x - squared.y),
            SH_BASIS_CONSTANTS[14] * ray_direction.z * (squared.x - squared.y),
            SH_BASIS_CONSTANTS[15] * ray_direction.x * (squared.x - 3.0 * squared.y),
        ]
    }

    fn sh_coefficient_rgb(
        spherical_harmonic: &SphericalHarmonicCoefficients,
        basis: usize,
    ) -> Vec3 {
        let base = basis * 3;
        let coefficient = |offset| {
            spherical_harmonic
                .coefficients
                .get(base + offset)
                .copied()
                .unwrap_or(0.0)
        };
        Vec3::new(coefficient(0), coefficient(1), coefficient(2))
    }

    fn spherical_harmonics_lookup_degree(
        ray_direction: Vec3,
        spherical_harmonic: &SphericalHarmonicCoefficients,
        maximum_degree: usize,
    ) -> Vec3 {
        let squared = ray_direction * ray_direction;
        let mut color =
            Vec3::splat(0.5) + sh_coefficient_rgb(spherical_harmonic, 0) * SH_BASIS_CONSTANTS[0];

        if maximum_degree >= 1 {
            color += (sh_coefficient_rgb(spherical_harmonic, 1) * SH_BASIS_CONSTANTS[1])
                * ray_direction.y;
            color += (sh_coefficient_rgb(spherical_harmonic, 2) * SH_BASIS_CONSTANTS[2])
                * ray_direction.z;
            color += (sh_coefficient_rgb(spherical_harmonic, 3) * SH_BASIS_CONSTANTS[3])
                * ray_direction.x;
        }
        if maximum_degree >= 2 {
            color += (sh_coefficient_rgb(spherical_harmonic, 4) * SH_BASIS_CONSTANTS[4])
                * ray_direction.x
                * ray_direction.y;
            color += (sh_coefficient_rgb(spherical_harmonic, 5) * SH_BASIS_CONSTANTS[5])
                * ray_direction.y
                * ray_direction.z;
            color += (sh_coefficient_rgb(spherical_harmonic, 6) * SH_BASIS_CONSTANTS[6])
                * (2.0 * squared.z - squared.x - squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 7) * SH_BASIS_CONSTANTS[7])
                * ray_direction.x
                * ray_direction.z;
            color += (sh_coefficient_rgb(spherical_harmonic, 8) * SH_BASIS_CONSTANTS[8])
                * (squared.x - squared.y);
        }
        if maximum_degree >= 3 {
            color += (sh_coefficient_rgb(spherical_harmonic, 9) * SH_BASIS_CONSTANTS[9])
                * ray_direction.y
                * (3.0 * squared.x - squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 10) * SH_BASIS_CONSTANTS[10])
                * ray_direction.x
                * ray_direction.y
                * ray_direction.z;
            color += (sh_coefficient_rgb(spherical_harmonic, 11) * SH_BASIS_CONSTANTS[11])
                * ray_direction.y
                * (4.0 * squared.z - squared.x - squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 12) * SH_BASIS_CONSTANTS[12])
                * ray_direction.z
                * (2.0 * squared.z - 3.0 * squared.x - 3.0 * squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 13) * SH_BASIS_CONSTANTS[13])
                * ray_direction.x
                * (4.0 * squared.z - squared.x - squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 14) * SH_BASIS_CONSTANTS[14])
                * ray_direction.z
                * (squared.x - squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 15) * SH_BASIS_CONSTANTS[15])
                * ray_direction.x
                * (squared.x - 3.0 * squared.y);
        }
        color
    }

    fn spherical_harmonics_linear_color(
        ray_direction: Vec3,
        spherical_harmonic: &SphericalHarmonicCoefficients,
        color_space: GaussianColorSpace,
    ) -> Vec3 {
        let display_color =
            spherical_harmonics_lookup_degree(ray_direction, spherical_harmonic, SH_DEGREE.min(3));
        match color_space {
            GaussianColorSpace::LinRec709Display => display_color,
            GaussianColorSpace::SrgbRec709Display => Vec3::new(
                srgb_display_channel_to_linear(display_color.x),
                srgb_display_channel_to_linear(display_color.y),
                srgb_display_channel_to_linear(display_color.z),
            ),
        }
    }

    fn srgb_display_channel_to_linear(value: f32) -> f32 {
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    fn composite_linear_source_over(pixel: &mut [f32; 4], linear_rgb: Vec3, alpha: f32) {
        let transmittance = 1.0 - alpha;
        pixel[0] = linear_rgb.x * alpha + pixel[0] * transmittance;
        pixel[1] = linear_rgb.y * alpha + pixel[1] * transmittance;
        pixel[2] = linear_rgb.z * alpha + pixel[2] * transmittance;
        pixel[3] = alpha + pixel[3] * transmittance;
    }

    fn assert_vec3_close(actual: Vec3, expected: Vec3) {
        let maximum_error = (actual - expected).abs().max_element();
        assert!(
            maximum_error <= 1e-6,
            "vectors differ by {maximum_error}: actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_rgba_close(actual: [f32; 4], expected: [f32; 4]) {
        for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 1e-6,
                "RGBA channel {channel} differs by {error}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn sh_basis_and_dc_match_the_renderer_shader() {
        let basis = spherical_harmonic_basis(Vec3::X);
        assert!((basis[0] - 0.282_094_8).abs() < 1e-7);
        assert!((basis[3] + 0.488_602_52).abs() < 1e-7);
        assert!((basis[8] - 0.546_274_24).abs() < 1e-7);
        assert!((basis[15] + 0.590_043_6).abs() < 1e-7);

        let mut spherical_harmonic = SphericalHarmonicCoefficients::default();
        assert_vec3_close(
            spherical_harmonics_lookup_degree(Vec3::Z, &spherical_harmonic, 0),
            Vec3::splat(0.5),
        );
        spherical_harmonic.set(0, 1.0);
        spherical_harmonic.set(1, -2.0);
        spherical_harmonic.set(2, 0.25);
        assert_vec3_close(
            spherical_harmonics_lookup_degree(Vec3::Z, &spherical_harmonic, 0),
            Vec3::splat(0.5) + Vec3::new(1.0, -2.0, 0.25) * 0.282_094_8,
        );

        let zero = SphericalHarmonicCoefficients::default();
        assert_vec3_close(
            spherical_harmonics_linear_color(Vec3::Z, &zero, GaussianColorSpace::LinRec709Display),
            Vec3::splat(0.5),
        );
        assert_vec3_close(
            spherical_harmonics_linear_color(Vec3::Z, &zero, GaussianColorSpace::SrgbRec709Display),
            Vec3::splat(srgb_display_channel_to_linear(0.5)),
        );
    }

    #[test]
    fn sh_directional_bands_match_the_renderer_shader() {
        let direction = Vec3::new(0.6, 0.8, 0.0);
        let mut degree_one = SphericalHarmonicCoefficients::default();
        degree_one.set(3, 1.0);
        degree_one.set(7, 1.0);
        degree_one.set(11, 1.0);
        assert_vec3_close(
            spherical_harmonics_lookup_degree(direction, &degree_one, 1),
            Vec3::new(0.5 - 0.488_602_52 * 0.8, 0.5, 0.5 - 0.488_602_52 * 0.6),
        );

        let mut upper_bands = SphericalHarmonicCoefficients::default();
        upper_bands.set(18, 1.0);
        upper_bands.set(37, 1.0);
        assert_vec3_close(
            spherical_harmonics_lookup_degree(Vec3::Z, &upper_bands, 3),
            Vec3::new(0.5 + 2.0 * 0.315_391_57, 0.5 + 2.0 * 0.373_176_34, 0.5),
        );
    }

    #[test]
    fn premultiplied_source_over_compositing_matches_the_renderer() {
        let mut pixel = [0.0; 4];
        composite_linear_source_over(&mut pixel, Vec3::new(0.8, 0.2, 0.1), 0.25);
        composite_linear_source_over(&mut pixel, Vec3::new(0.1, 0.4, 0.9), 0.5);
        assert_rgba_close(pixel, [0.15, 0.225, 0.4625, 0.625]);
    }

    fn render_color_and_covariance_image(
        gaussians: &[Gaussian3d],
        camera: LodTestCamera,
        width: u32,
        height: u32,
        color_space: GaussianColorSpace,
    ) -> (Vec<[f32; 4]>, CovarianceDiagnostics) {
        let forward = (camera.target - camera.position).normalize_or_zero();
        let right = forward.cross(camera.up).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        assert!(forward != Vec3::ZERO && right != Vec3::ZERO && up != Vec3::ZERO);
        let fov = match camera.projection {
            LodProjection::Perspective {
                vertical_fov_radians,
            } => vertical_fov_radians,
            LodProjection::Orthographic { .. } => {
                panic!("Trellis color/covariance audit requires perspective cameras")
            }
        };
        let focal_y = 0.5 * height as f32 / (0.5 * fov).tan();
        let focal_x = focal_y;
        let mut diagnostics = CovarianceDiagnostics::default();
        let mut projected = Vec::with_capacity(gaussians.len());
        for gaussian in gaussians {
            let position = Vec3::from_array(gaussian.position_visibility.position);
            let relative = position - camera.position;
            let depth = relative.dot(forward);
            if depth < camera.near || depth > camera.far || !depth.is_finite() {
                continue;
            }
            let view_x = relative.dot(right);
            let view_y = relative.dot(up);
            let center = Vec2::new(
                width as f32 * 0.5 + focal_x * view_x / depth,
                height as f32 * 0.5 - focal_y * view_y / depth,
            );
            let packed = compute_covariance_3d(
                Vec4::from_array(gaussian.rotation.rotation),
                Vec3::from_array(gaussian.scale_opacity.scale.map(f32::abs)),
            );
            let covariance = Mat3::from_cols(
                Vec3::new(packed[0], packed[1], packed[2]),
                Vec3::new(packed[1], packed[3], packed[4]),
                Vec3::new(packed[2], packed[4], packed[5]),
            );
            let reciprocal_depth = 1.0 / depth;
            let reciprocal_depth_squared = reciprocal_depth * reciprocal_depth;
            let jacobian_x = right * (focal_x * reciprocal_depth)
                - forward * (focal_x * view_x * reciprocal_depth_squared);
            let jacobian_y = -up * (focal_y * reciprocal_depth)
                + forward * (focal_y * view_y * reciprocal_depth_squared);
            let covariance_x = jacobian_x.dot(covariance * jacobian_x) + 0.3;
            let covariance_xy = jacobian_x.dot(covariance * jacobian_y);
            let covariance_y = jacobian_y.dot(covariance * jacobian_y) + 0.3;
            let midpoint = 0.5 * (covariance_x + covariance_y);
            let determinant = covariance_x * covariance_y - covariance_xy * covariance_xy;
            let discriminant = (midpoint * midpoint - determinant).max(0.0).sqrt();
            let major = (midpoint + discriminant).max(0.0).sqrt();
            let minor = (midpoint - discriminant).max(1e-8).sqrt();
            let aspect = major / minor;
            if determinant <= 1e-8
                || !center.is_finite()
                || !major.is_finite()
                || !aspect.is_finite()
            {
                continue;
            }
            diagnostics.projected_splats += 1;
            diagnostics.maximum_major_sigma_px = diagnostics.maximum_major_sigma_px.max(major);
            diagnostics.maximum_aspect_ratio = diagnostics.maximum_aspect_ratio.max(aspect);
            diagnostics.extreme_projected_splats += usize::from(major > 8.0 && aspect > 20.0);
            let opacity = (gaussian.scale_opacity.opacity
                * gaussian.position_visibility.visibility)
                .clamp(0.0, 0.999);
            let support_radius = 3.0 * major;
            let intersects_viewport = center.x + support_radius >= 0.0
                && center.x - support_radius < width as f32
                && center.y + support_radius >= 0.0
                && center.y - support_radius < height as f32;
            diagnostics.visible_elongated_splats += usize::from(
                opacity >= FOREGROUND_ALPHA
                    && intersects_viewport
                    && major > VISIBLE_ELONGATION_MIN_MAJOR_SIGMA_PX
                    && aspect > VISIBLE_ELONGATION_MIN_ASPECT_RATIO,
            );
            projected.push(ProjectedCovariance {
                center,
                depth,
                covariance: Vec3::new(covariance_x, covariance_xy, covariance_y),
                opacity,
                // The audit cameras and Gaussian positions are both expressed in
                // cloud-local coordinates, matching the renderer's world-to-local
                // direction before its spherical-harmonic lookup.
                linear_rgb: spherical_harmonics_linear_color(
                    relative.normalize_or_zero(),
                    &gaussian.spherical_harmonic,
                    color_space,
                ),
            });
        }
        projected.sort_by(|left, right| right.depth.total_cmp(&left.depth));

        let mut image = vec![[0.0_f32; 4]; (width * height) as usize];
        for gaussian in projected {
            let covariance_x = gaussian.covariance.x;
            let covariance_xy = gaussian.covariance.y;
            let covariance_y = gaussian.covariance.z;
            let determinant = covariance_x * covariance_y - covariance_xy * covariance_xy;
            if determinant <= 1e-8 {
                continue;
            }
            let midpoint = 0.5 * (covariance_x + covariance_y);
            let discriminant = (midpoint * midpoint - determinant).max(0.0).sqrt();
            let radius = (3.0 * (midpoint + discriminant).max(0.0).sqrt()).ceil() as i32;
            let min_x = (gaussian.center.x.floor() as i32 - radius).max(0);
            let max_x = (gaussian.center.x.ceil() as i32 + radius).min(width as i32 - 1);
            let min_y = (gaussian.center.y.floor() as i32 - radius).max(0);
            let max_y = (gaussian.center.y.ceil() as i32 + radius).min(height as i32 - 1);
            if min_x > max_x || min_y > max_y {
                continue;
            }
            let inverse_x = covariance_y / determinant;
            let inverse_xy = -covariance_xy / determinant;
            let inverse_y = covariance_x / determinant;
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let delta = Vec2::new(
                        x as f32 + 0.5 - gaussian.center.x,
                        y as f32 + 0.5 - gaussian.center.y,
                    );
                    let mahalanobis = inverse_x * delta.x * delta.x
                        + 2.0 * inverse_xy * delta.x * delta.y
                        + inverse_y * delta.y * delta.y;
                    if mahalanobis > 9.0 {
                        continue;
                    }
                    let source_alpha =
                        (gaussian.opacity * (-0.5 * mahalanobis).exp()).clamp(0.0, 0.999);
                    let pixel = &mut image[(y as u32 * width + x as u32) as usize];
                    composite_linear_source_over(pixel, gaussian.linear_rgb, source_alpha);
                }
            }
        }
        (image, diagnostics)
    }

    #[derive(Clone, Copy, Debug)]
    struct QualityThresholds {
        minimum_foreground_psnr: f64,
        minimum_ssim: f64,
        minimum_iou: f64,
        maximum_alpha_mae: f64,
        maximum_spill: f64,
    }

    #[derive(Clone, Copy, Debug)]
    struct QualityObservation {
        full: ImageMetrics,
        foreground_roi: ImageMetrics,
        spill_outside_dilated_reference: f64,
        foreground_spill_pixels_outside_dilated_reference: usize,
        maximum_foreground_distance_px: usize,
    }

    #[derive(Clone, Copy, Debug)]
    struct QualitySweepPoint {
        quality: f32,
        active_gaussians: usize,
        base_target_error_px: Option<f32>,
        error_authority: Option<f32>,
        effective_error_cap_px: Option<f32>,
        achieved_max_error_px: f32,
        covariance: CovarianceDiagnostics,
        observation: QualityObservation,
    }

    #[derive(Clone, Copy, Debug)]
    struct SelectionSweepPoint {
        quality: f32,
        active_gaussians: usize,
    }

    #[derive(Debug)]
    struct TrellisDistanceSweep {
        camera: TrellisDistanceCamera,
        rendered: Vec<QualitySweepPoint>,
        deployment_selection: Vec<SelectionSweepPoint>,
    }

    #[derive(Debug)]
    struct TrellisMorphologySweep {
        label: &'static str,
        rendered: Vec<QualitySweepPoint>,
    }

    fn quality_sweep_point(
        quality: f32,
        sample: &RealSceneQualitySample,
        exact: &RealSceneQualitySample,
        camera: LodTestCamera,
    ) -> QualitySweepPoint {
        let settings = GaussianLodSettings {
            quality,
            ..Default::default()
        };
        let target = settings.quality_target();
        QualitySweepPoint {
            quality,
            active_gaussians: sample.active_gaussians,
            base_target_error_px: target.max_screen_space_error_px(),
            error_authority: target.error_authority(),
            effective_error_cap_px: target.effective_max_screen_space_error_px(),
            achieved_max_error_px: sample.achieved_max_error_px,
            covariance: sample.covariance,
            observation: quality_observation(
                &exact.image,
                &sample.image,
                camera.viewport[0],
                camera.viewport[1],
            ),
        }
    }

    fn quality_observation(
        reference: &[[f32; 4]],
        candidate: &[[f32; 4]],
        width: u32,
        height: u32,
    ) -> QualityObservation {
        let full = compare_linear_rgba(reference, candidate, FOREGROUND_ALPHA)
            .expect("quality images have matching dimensions");
        let (reference_roi, candidate_roi) =
            foreground_union_roi(reference, candidate, width, height, 4);
        let foreground_roi = compare_linear_rgba(&reference_roi, &candidate_roi, FOREGROUND_ALPHA)
            .expect("quality foreground ROIs match");
        let spill = alpha_spill_diagnostics(reference, candidate, width, height, SPILL_DILATION_PX);
        QualityObservation {
            full,
            foreground_roi,
            spill_outside_dilated_reference: spill.alpha_mass_ratio,
            foreground_spill_pixels_outside_dilated_reference: spill
                .foreground_pixels_outside_dilated_reference,
            maximum_foreground_distance_px: spill.maximum_foreground_distance_px,
        }
    }

    fn assert_thresholds(label: &str, observation: QualityObservation, limits: QualityThresholds) {
        assert!(
            observation.full.foreground_psnr_rgb >= limits.minimum_foreground_psnr,
            "{label} foreground PSNR below {} dB: {observation:?}",
            limits.minimum_foreground_psnr
        );
        assert!(
            observation.foreground_roi.luminance_ssim >= limits.minimum_ssim,
            "{label} foreground ROI SSIM below {}: {observation:?}",
            limits.minimum_ssim
        );
        assert!(
            observation.full.foreground_iou >= limits.minimum_iou,
            "{label} foreground IoU below {}: {observation:?}",
            limits.minimum_iou
        );
        assert!(
            observation.foreground_roi.alpha_mae <= limits.maximum_alpha_mae,
            "{label} foreground ROI alpha MAE above {}: {observation:?}",
            limits.maximum_alpha_mae
        );
        assert!(
            observation.spill_outside_dilated_reference <= limits.maximum_spill,
            "{label} covariance spill above {}: {observation:?}",
            limits.maximum_spill
        );
    }

    fn assert_alpha_morphology_bound(label: &str, observation: QualityObservation) {
        assert_eq!(
            observation.foreground_spill_pixels_outside_dilated_reference, 0,
            "{label} renders alpha >= {FOREGROUND_ALPHA} outside the q=1 foreground mask dilated by {SPILL_DILATION_PX}px: {observation:?}"
        );
        assert!(
            observation.maximum_foreground_distance_px <= SPILL_DILATION_PX,
            "{label} foreground extends {}px from the nearest q=1 foreground pixel (limit {SPILL_DILATION_PX}px): {observation:?}",
            observation.maximum_foreground_distance_px
        );
    }

    fn assert_monotonic(label: &str, lower: QualityObservation, higher: QualityObservation) {
        assert!(
            higher.full.foreground_psnr_rgb + 0.25 >= lower.full.foreground_psnr_rgb,
            "{label} foreground PSNR regressed: lower={lower:?}, higher={higher:?}"
        );
        assert!(
            higher.foreground_roi.luminance_ssim + 0.005 >= lower.foreground_roi.luminance_ssim,
            "{label} foreground SSIM regressed: lower={lower:?}, higher={higher:?}"
        );
        assert!(
            higher.full.foreground_iou + 0.005 >= lower.full.foreground_iou,
            "{label} foreground IoU regressed: lower={lower:?}, higher={higher:?}"
        );
        assert!(
            higher.foreground_roi.alpha_mae <= lower.foreground_roi.alpha_mae + 0.005,
            "{label} alpha MAE regressed: lower={lower:?}, higher={higher:?}"
        );
        assert!(
            higher.spill_outside_dilated_reference <= lower.spill_outside_dilated_reference + 0.002,
            "{label} covariance spill regressed: lower={lower:?}, higher={higher:?}"
        );
    }

    fn trellis_dense_selection_qualities() -> Vec<f32> {
        (0..=200).map(|step| step as f32 / 200.0).collect()
    }

    #[derive(Clone, Copy, Debug)]
    struct UtilityGoal {
        label: &'static str,
        minimum_savings_percent: usize,
        minimum_psnr_db: f64,
    }

    const UTILITY_GOALS: [UtilityGoal; 2] = [
        UtilityGoal {
            label: "10% savings at >=33 dB",
            minimum_savings_percent: 10,
            minimum_psnr_db: 33.0,
        },
        UtilityGoal {
            label: "5% savings at >=35 dB",
            minimum_savings_percent: 5,
            minimum_psnr_db: 35.0,
        },
    ];

    #[derive(Clone, Copy, Debug)]
    struct ContinuitySummary {
        distinct_cuts: usize,
        q20: f32,
        q50: f32,
        q80: f32,
        widest_interior_step: usize,
        widest_step_lower_quality: f32,
        widest_step_higher_quality: f32,
        mean_active: f64,
    }

    fn continuity_summary(sweep: &[SelectionSweepPoint]) -> ContinuitySummary {
        let distinct_cuts = sweep
            .iter()
            .map(|point| point.active_gaussians)
            .collect::<BTreeSet<_>>()
            .len();
        let interior = sweep
            .iter()
            .filter(|point| point.quality > 0.0 && point.quality < 1.0)
            .collect::<Vec<_>>();
        let mean_active = interior
            .iter()
            .map(|point| point.active_gaussians as f64)
            .sum::<f64>()
            / interior.len() as f64;
        let (widest_interior_step, widest_step_lower_quality, widest_step_higher_quality) = sweep
            .windows(2)
            .filter_map(|pair| {
                let [lower, higher] = pair else {
                    unreachable!()
                };
                (lower.quality > 0.0 && higher.quality < 1.0).then_some((
                    higher
                        .active_gaussians
                        .saturating_sub(lower.active_gaussians),
                    lower.quality,
                    higher.quality,
                ))
            })
            .max_by(|left, right| left.0.cmp(&right.0))
            .expect("dense deployment sweep has interior steps");
        ContinuitySummary {
            distinct_cuts,
            q20: deployment_crossing_quality(sweep, 20),
            q50: deployment_crossing_quality(sweep, 50),
            q80: deployment_crossing_quality(sweep, 80),
            widest_interior_step,
            widest_step_lower_quality,
            widest_step_higher_quality,
            mean_active,
        }
    }

    fn deployment_crossing_quality(sweep: &[SelectionSweepPoint], percent: usize) -> f32 {
        sweep
            .iter()
            .find(|point| point.active_gaussians * 100 >= TRELLIS_SPLAT_COUNT * percent)
            .unwrap_or_else(|| panic!("deployment sweep never crosses {percent}% active"))
            .quality
    }

    fn meets_utility_goal(point: &QualitySweepPoint, goal: UtilityGoal) -> bool {
        point.quality > 0.0
            && point.quality < 1.0
            && point.active_gaussians * 100
                <= TRELLIS_SPLAT_COUNT * (100 - goal.minimum_savings_percent)
            && point.observation.full.foreground_psnr_rgb >= goal.minimum_psnr_db
    }

    fn utility_anchor(
        sweep: &[QualitySweepPoint],
        goal: UtilityGoal,
    ) -> Option<&QualitySweepPoint> {
        sweep
            .iter()
            .filter(|point| meets_utility_goal(point, goal))
            .min_by(|left, right| {
                left.active_gaussians
                    .cmp(&right.active_gaussians)
                    .then_with(|| left.quality.total_cmp(&right.quality))
            })
    }

    fn common_utility_quality(sweeps: &[TrellisDistanceSweep], goal: UtilityGoal) -> Option<f32> {
        sweeps
            .first()?
            .rendered
            .iter()
            .filter_map(|candidate| {
                let quality = candidate.quality;
                let points = sweeps
                    .iter()
                    .map(|sweep| point_at_quality(&sweep.rendered, quality))
                    .collect::<Vec<_>>();
                points
                    .iter()
                    .all(|point| meets_utility_goal(point, goal))
                    .then_some((
                        quality,
                        points
                            .iter()
                            .map(|point| point.active_gaussians)
                            .sum::<usize>(),
                    ))
            })
            .min_by(|left, right| {
                left.1
                    .cmp(&right.1)
                    .then_with(|| left.0.total_cmp(&right.0))
            })
            .map(|(quality, _)| quality)
    }

    fn trellis_quality_report(
        profile: TrellisAuditProfile,
        sweeps: &[TrellisDistanceSweep],
        authored: &[QualitySweepPoint],
        orbits: &[TrellisMorphologySweep],
    ) -> String {
        assert_eq!(sweeps.len(), 3);
        let mut report = String::new();
        writeln!(report, "# Canonical Trellis LoD quality report").unwrap();
        writeln!(report).unwrap();
        writeln!(report, "- Audit profile: `{}`", profile.as_str()).unwrap();
        writeln!(
            report,
            "- Source: {TRELLIS_SPLAT_COUNT} Gaussians, SHA-256 `{TRELLIS_SHA256_FOR_PREFLIGHT}`"
        )
        .unwrap();
        writeln!(
            report,
            "- Metric cuts and deterministic image oracle: {TRELLIS_RASTER_SIZE}px selection height and {TRELLIS_RASTER_SIZE}x{TRELLIS_RASTER_SIZE} raster"
        )
        .unwrap();
        writeln!(
            report,
            "- Deployment count graph: {:.0}px selection height (reported separately; never paired with 192px PSNR)",
            TRELLIS_DEPLOYMENT_VIEWPORT_HEIGHT_PX
        )
        .unwrap();
        writeln!(
            report,
            "- PSNR: linear-RGB foreground-union PSNR against quality 1 at the same camera"
        )
        .unwrap();
        writeln!(
            report,
            "- Morphology: authored camera plus {} deterministic mid-coverage orbit views; visible elongation means major sigma >{VISIBLE_ELONGATION_MIN_MAJOR_SIGMA_PX:.1}px and aspect >{VISIBLE_ELONGATION_MIN_ASPECT_RATIO:.1}:1",
            orbits.len()
        )
        .unwrap();
        writeln!(report).unwrap();
        writeln!(report, "## Camera-distance contract").unwrap();
        writeln!(report).unwrap();
        writeln!(
            report,
            "| scale | root coverage | center distance / root radius |"
        )
        .unwrap();
        writeln!(report, "|---|---:|---:|").unwrap();
        for sweep in sweeps {
            writeln!(
                report,
                "| {} | {:.0}% | {:.3} R |",
                sweep.camera.label,
                sweep.camera.root_coverage * 100.0,
                sweep.camera.distance_in_root_radii
            )
            .unwrap();
        }
        writeln!(report).unwrap();
        writeln!(report, "## Quality graph").unwrap();
        writeln!(report).unwrap();
        writeln!(
            report,
            "| q | base target px | error authority | effective cap px | near 192px active (% source) | near PSNR dB | mid 192px active (% source) | mid PSNR dB | far 192px active (% source) | far PSNR dB |"
        )
        .unwrap();
        writeln!(
            report,
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        .unwrap();
        for (index, first) in sweeps[0].rendered.iter().enumerate() {
            let points = sweeps
                .iter()
                .map(|sweep| {
                    let point = &sweep.rendered[index];
                    assert!(quality_eq(point.quality, first.quality));
                    assert_eq!(point.base_target_error_px, first.base_target_error_px);
                    assert_eq!(point.error_authority, first.error_authority);
                    assert_eq!(point.effective_error_cap_px, first.effective_error_cap_px);
                    point
                })
                .collect::<Vec<_>>();
            writeln!(
                report,
                "| {:.4} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                first.quality,
                format_base_target_error(first),
                format_error_authority(first),
                format_effective_error_cap(first),
                format_active(points[0].active_gaussians),
                format_psnr(points[0].observation.full.foreground_psnr_rgb),
                format_active(points[1].active_gaussians),
                format_psnr(points[1].observation.full.foreground_psnr_rgb),
                format_active(points[2].active_gaussians),
                format_psnr(points[2].observation.full.foreground_psnr_rgb),
            )
            .unwrap();
        }
        writeln!(report).unwrap();
        writeln!(report, "## 1080p deployment continuity summary").unwrap();
        writeln!(report).unwrap();
        writeln!(
            report,
            "| scale | distinct cuts | q20 | q50 | q80 | widest interior .005 step | mean active (% source) |"
        )
        .unwrap();
        writeln!(report, "|---|---:|---:|---:|---:|---:|---:|").unwrap();
        for sweep in sweeps {
            let summary = continuity_summary(&sweep.deployment_selection);
            writeln!(
                report,
                "| {} | {} | {:.3} | {:.3} | {:.3} | {} ({:.2}%, q={:.3}->{:.3}) | {:.1} ({:.2}%) |",
                sweep.camera.label,
                summary.distinct_cuts,
                summary.q20,
                summary.q50,
                summary.q80,
                summary.widest_interior_step,
                summary.widest_interior_step as f64 * 100.0 / TRELLIS_SPLAT_COUNT as f64,
                summary.widest_step_lower_quality,
                summary.widest_step_higher_quality,
                summary.mean_active,
                summary.mean_active * 100.0 / TRELLIS_SPLAT_COUNT as f64,
            )
            .unwrap();
        }
        writeln!(report).unwrap();
        writeln!(report, "## Common utility anchors and safety diagnostics").unwrap();
        writeln!(report).unwrap();
        for goal in UTILITY_GOALS {
            match common_utility_quality(sweeps, goal) {
                Some(quality) => {
                    writeln!(report, "- {}: q={quality:.4}", goal.label).unwrap();
                }
                None => {
                    writeln!(report, "- {}: no common interior anchor", goal.label).unwrap();
                }
            }
        }
        writeln!(report).unwrap();
        writeln!(
            report,
            "| purpose | camera | q | 192px metric active (% source) | PSNR dB | achieved max px | ROI SSIM | IoU | alpha MAE | spill | alpha>=.02 outside q1+2px | max foreground distance px | max sigma px | max aspect | projected splats | extreme splats (%) | visible elongated splats |"
        )
        .unwrap();
        writeln!(
            report,
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        .unwrap();
        for goal in UTILITY_GOALS {
            if let Some(quality) = common_utility_quality(sweeps, goal) {
                for sweep in sweeps {
                    write_diagnostic_row(
                        &mut report,
                        goal.label,
                        sweep.camera.label,
                        point_at_quality(&sweep.rendered, quality),
                    );
                }
            }
        }
        for sweep in sweeps {
            for quality in [0.95, 0.99, 1.0] {
                write_diagnostic_row(
                    &mut report,
                    "safety",
                    sweep.camera.label,
                    point_at_quality(&sweep.rendered, quality),
                );
            }
        }
        for point in authored {
            write_diagnostic_row(&mut report, "morphology", "authored", point);
        }
        for sweep in orbits {
            for point in &sweep.rendered {
                write_diagnostic_row(&mut report, "morphology", sweep.label, point);
            }
        }
        report
    }

    fn write_diagnostic_row(
        report: &mut String,
        purpose: &str,
        label: &str,
        point: &QualitySweepPoint,
    ) {
        writeln!(
            report,
            "| {purpose} | {label} | {:.4} | {} | {} | {:.3} | {:.5} | {:.5} | {:.6} | {:.6} | {} | {} | {:.3} | {:.3} | {} | {} ({:.4}%) | {} |",
            point.quality,
            format_active(point.active_gaussians),
            format_psnr(point.observation.full.foreground_psnr_rgb),
            point.achieved_max_error_px,
            point.observation.foreground_roi.luminance_ssim,
            point.observation.full.foreground_iou,
            point.observation.foreground_roi.alpha_mae,
            point.observation.spill_outside_dilated_reference,
            point
                .observation
                .foreground_spill_pixels_outside_dilated_reference,
            point.observation.maximum_foreground_distance_px,
            point.covariance.maximum_major_sigma_px,
            point.covariance.maximum_aspect_ratio,
            point.covariance.projected_splats,
            point.covariance.extreme_projected_splats,
            extreme_projected_percentage(point.covariance),
            point.covariance.visible_elongated_splats,
        )
        .unwrap();
    }

    fn format_base_target_error(point: &QualitySweepPoint) -> String {
        match point.base_target_error_px {
            Some(value) => format!("{value:.3}"),
            None if point.quality <= 0.0 => "roots".to_owned(),
            None => "exact".to_owned(),
        }
    }

    fn format_error_authority(point: &QualitySweepPoint) -> String {
        match point.error_authority {
            Some(value) => format!("{value:.6}"),
            None if point.quality <= 0.0 => "roots".to_owned(),
            None => "exact".to_owned(),
        }
    }

    fn format_effective_error_cap(point: &QualitySweepPoint) -> String {
        match point.effective_error_cap_px {
            Some(value) => format!("{value:.3}"),
            None if point.quality <= 0.0 => "roots".to_owned(),
            None => "exact".to_owned(),
        }
    }

    fn extreme_projected_percentage(covariance: CovarianceDiagnostics) -> f64 {
        if covariance.projected_splats == 0 {
            0.0
        } else {
            covariance.extreme_projected_splats as f64 * 100.0 / covariance.projected_splats as f64
        }
    }

    fn format_active(active_gaussians: usize) -> String {
        format!(
            "{} ({:.2}%)",
            active_gaussians,
            active_gaussians as f64 * 100.0 / TRELLIS_SPLAT_COUNT as f64
        )
    }

    fn format_psnr(value: f64) -> String {
        if value.is_infinite() {
            "inf".to_owned()
        } else {
            format!("{value:.2}")
        }
    }

    fn point_at_quality(sweep: &[QualitySweepPoint], quality: f32) -> &QualitySweepPoint {
        sweep
            .iter()
            .find(|point| quality_eq(point.quality, quality))
            .unwrap_or_else(|| panic!("Trellis rendered sweep is missing q={quality:.4}"))
    }

    fn selection_point_at_quality(
        sweep: &[SelectionSweepPoint],
        quality: f32,
    ) -> SelectionSweepPoint {
        sweep
            .iter()
            .copied()
            .find(|point| quality_eq(point.quality, quality))
            .unwrap_or_else(|| panic!("Trellis deployment sweep is missing q={quality:.4}"))
    }

    fn quality_eq(left: f32, right: f32) -> bool {
        (left - right).abs() <= 1e-6
    }

    fn assert_trellis_distance_graph(sweeps: &[TrellisDistanceSweep]) {
        assert_eq!(sweeps.len(), 3);
        assert_eq!(
            sweeps
                .iter()
                .map(|sweep| sweep.camera.label)
                .collect::<Vec<_>>(),
            ["near", "mid", "far"]
        );
        for sweep in sweeps {
            assert_dense_selection_sweep(sweep);
            assert_running_best_quality(sweep.camera.label, &sweep.rendered);
            assert_trellis_scale_quality(sweep);
        }

        let root_active = sweeps[0].deployment_selection[0].active_gaussians;
        assert!(
            sweeps
                .iter()
                .all(|sweep| sweep.deployment_selection[0].active_gaussians == root_active),
            "quality zero must select the same roots at every distance: {sweeps:?}"
        );
        for index in 0..sweeps[0].deployment_selection.len() {
            let near = sweeps[0].deployment_selection[index];
            let mid = sweeps[1].deployment_selection[index];
            let far = sweeps[2].deployment_selection[index];
            assert!(quality_eq(near.quality, mid.quality) && quality_eq(mid.quality, far.quality));
            assert!(
                near.active_gaussians >= mid.active_gaussians
                    && mid.active_gaussians >= far.active_gaussians,
                "Trellis active count increased with distance at q={:.4}: near={near:?}, mid={mid:?}, far={far:?}",
                near.quality
            );
        }
        let metric_root_active = sweeps[0].rendered[0].active_gaussians;
        assert!(
            sweeps
                .iter()
                .all(|sweep| sweep.rendered[0].active_gaussians == metric_root_active),
            "192px quality zero must select the same roots at every distance: {sweeps:?}"
        );
        for index in 0..sweeps[0].rendered.len() {
            let near = sweeps[0].rendered[index];
            let mid = sweeps[1].rendered[index];
            let far = sweeps[2].rendered[index];
            assert!(quality_eq(near.quality, mid.quality) && quality_eq(mid.quality, far.quality));
            assert!(
                near.active_gaussians >= mid.active_gaussians
                    && mid.active_gaussians >= far.active_gaussians,
                "Trellis 192px metric active count increased with distance at q={:.4}: near={near:?}, mid={mid:?}, far={far:?}",
                near.quality
            );
            for (sweep, metric) in sweeps.iter().zip([near, mid, far]) {
                let deployment =
                    selection_point_at_quality(&sweep.deployment_selection, metric.quality);
                assert!(
                    deployment.active_gaussians >= metric.active_gaussians,
                    "Trellis {} 1080p deployment cut is coarser than its 192px metric cut at q={:.4}: deployment={deployment:?}, metric={metric:?}",
                    sweep.camera.label,
                    metric.quality
                );
            }
        }

        for goal in UTILITY_GOALS {
            let quality = common_utility_quality(sweeps, goal).unwrap_or_else(|| {
                panic!(
                    "no common interior Trellis anchor meets {} at every distance: {sweeps:?}",
                    goal.label
                )
            });
            for sweep in sweeps {
                let anchor = point_at_quality(&sweep.rendered, quality);
                let exact = point_at_quality(&sweep.rendered, 1.0);
                assert_covariance_morphology_bound(
                    &format!(
                        "Trellis {} common {} anchor q={quality:.4}",
                        sweep.camera.label, goal.label
                    ),
                    anchor,
                    exact,
                );
            }
        }

        let summaries = sweeps
            .iter()
            .map(|sweep| continuity_summary(&sweep.deployment_selection))
            .collect::<Vec<_>>();
        for (crossing, near, mid, far) in [
            ("q20", summaries[0].q20, summaries[1].q20, summaries[2].q20),
            ("q50", summaries[0].q50, summaries[1].q50, summaries[2].q50),
            ("q80", summaries[0].q80, summaries[1].q80, summaries[2].q80),
        ] {
            assert!(
                near <= mid + 1e-6 && mid <= far + 1e-6,
                "Trellis deployment {crossing} crossings are not ordered near <= mid <= far: near={near}, mid={mid}, far={far}"
            );
        }

        let mut interior_samples = 0_usize;
        let mut strict_distance_samples = 0_usize;
        let mut near_mid_gap = 0_usize;
        let mut mid_far_gap = 0_usize;
        for index in 0..sweeps[0].deployment_selection.len() {
            let near = sweeps[0].deployment_selection[index];
            let mid = sweeps[1].deployment_selection[index];
            let far = sweeps[2].deployment_selection[index];
            if near.quality <= 0.0 || near.quality >= 1.0 {
                continue;
            }
            interior_samples += 1;
            near_mid_gap += near.active_gaussians - mid.active_gaussians;
            mid_far_gap += mid.active_gaussians - far.active_gaussians;
            strict_distance_samples += usize::from(
                near.active_gaussians > mid.active_gaussians
                    && mid.active_gaussians > far.active_gaussians,
            );
        }
        assert!(interior_samples > 0);
        assert!(
            near_mid_gap * 50 >= interior_samples * TRELLIS_SPLAT_COUNT,
            "Trellis average near-to-mid deployment separation is below 2% of source: total_gap={near_mid_gap}, samples={interior_samples}"
        );
        assert!(
            mid_far_gap * 50 >= interior_samples * TRELLIS_SPLAT_COUNT,
            "Trellis average mid-to-far deployment separation is below 2% of source: total_gap={mid_far_gap}, samples={interior_samples}"
        );
        assert!(
            strict_distance_samples * 5 >= interior_samples,
            "Trellis has strict near > mid > far selection on fewer than 20% of interior samples: strict={strict_distance_samples}, samples={interior_samples}"
        );
    }

    fn assert_dense_selection_sweep(sweep: &TrellisDistanceSweep) {
        assert_eq!(sweep.deployment_selection.len(), 201);
        for pair in sweep.deployment_selection.windows(2) {
            let [lower, higher] = pair else {
                unreachable!()
            };
            assert!(higher.quality > lower.quality);
            assert!(higher.quality - lower.quality <= 0.005 + 1e-6);
            assert!(
                higher.active_gaussians >= lower.active_gaussians,
                "Trellis {} active count regressed with quality: lower={lower:?}, higher={higher:?}",
                sweep.camera.label
            );
            if lower.quality > 0.0 && higher.quality < 1.0 {
                assert!(
                    higher.active_gaussians - lower.active_gaussians
                        <= TRELLIS_SPLAT_COUNT / 20 + TRELLIS_CONTINUITY_DISCRETE_RECORD_SLACK,
                    "Trellis {} interior .005 quality step activates more than 5% of the source plus one two-leaf domain: lower={lower:?}, higher={higher:?}",
                    sweep.camera.label
                );
            }
        }
        let summary = continuity_summary(&sweep.deployment_selection);
        assert!(
            summary.q20 <= summary.q50 + 1e-6 && summary.q50 <= summary.q80 + 1e-6,
            "Trellis {} deployment crossings are not ordered q20 <= q50 <= q80: {summary:?}",
            sweep.camera.label
        );
        // At the dense .005 selector sampling rate this requires at least 15
        // independently checked slider intervals between the 20% and 80%
        // workload crossings. The pre-retune Trellis cliff occupied roughly
        // .01 quality, so this remains a strong anti-cliff contract without
        // pretending that active-record count must be linear in quality.
        assert!(
            summary.q80 - summary.q20 >= 0.075 - 1e-6,
            "Trellis {} deployment graph traverses q20 to q80 in less than .075 quality: {summary:?}",
            sweep.camera.label
        );
        assert_eq!(
            sweep
                .deployment_selection
                .last()
                .expect("dense deployment sweep has q=1")
                .active_gaussians,
            TRELLIS_SPLAT_COUNT
        );
    }

    fn assert_running_best_quality(label: &str, sweep: &[QualitySweepPoint]) {
        assert!(!sweep.is_empty());
        for pair in sweep.windows(2) {
            let [previous, point] = pair else {
                unreachable!()
            };
            assert!(point.quality > previous.quality);
            assert!(
                point.active_gaussians >= previous.active_gaussians,
                "Trellis {label} 192px metric active count regressed with quality: previous={previous:?}, point={point:?}"
            );
        }
        let mut best_psnr = f64::NEG_INFINITY;
        let mut best_ssim: Option<f64> = None;
        let mut best_iou: Option<f64> = None;
        let mut best_alpha_mae: Option<f64> = None;
        let mut best_spill: Option<f64> = None;
        for point in sweep {
            let observation = point.observation;
            // Root-level and very coarse cuts contain only a handful of
            // representatives, so exchanging one discrete page can move a
            // local foreground mask without indicating a broken quality
            // curve. Keep that noise bounded to 1 dB; once the useful half of
            // the slider begins, enforce the tighter 0.5 dB envelope.
            let psnr_tolerance = if point.quality < 0.5 - 1e-6 {
                1.0
            } else if point.quality < 0.7 - 1e-6 {
                // Mid-quality cuts still replace coarse discrete
                // representatives. Bound that quantization noise while the
                // image remains below the useful 30+ dB region.
                1.25
            } else {
                0.5
            };
            assert!(
                observation.full.foreground_psnr_rgb + psnr_tolerance >= best_psnr,
                "Trellis {label} q={:.4} PSNR regressed below the running-best envelope: best={best_psnr}, point={point:?}",
                point.quality
            );
            best_psnr = best_psnr.max(observation.full.foreground_psnr_rgb);
            if point.quality < 0.7 - 1e-6 {
                continue;
            }
            if let Some(best) = best_ssim {
                assert!(
                    observation.foreground_roi.luminance_ssim + 0.005 >= best,
                    "Trellis {label} q={:.4} SSIM regressed below the q>=.7 running-best envelope: best={best}, point={point:?}",
                    point.quality
                );
            }
            if let Some(best) = best_iou {
                assert!(
                    observation.full.foreground_iou + 0.01 >= best,
                    "Trellis {label} q={:.4} IoU regressed below the q>=.7 running-best envelope: best={best}, point={point:?}",
                    point.quality
                );
            }
            if let Some(best) = best_alpha_mae {
                assert!(
                    observation.foreground_roi.alpha_mae <= best + 0.005,
                    "Trellis {label} q={:.4} alpha MAE regressed above the q>=.7 running-best envelope: best={best}, point={point:?}",
                    point.quality
                );
            }
            if let Some(best) = best_spill {
                assert!(
                    observation.spill_outside_dilated_reference <= best + 0.002,
                    "Trellis {label} q={:.4} spill regressed above the q>=.7 running-best envelope: best={best}, point={point:?}",
                    point.quality
                );
            }
            best_ssim = Some(
                best_ssim.map_or(observation.foreground_roi.luminance_ssim, |best| {
                    best.max(observation.foreground_roi.luminance_ssim)
                }),
            );
            best_iou = Some(best_iou.map_or(observation.full.foreground_iou, |best| {
                best.max(observation.full.foreground_iou)
            }));
            best_alpha_mae = Some(
                best_alpha_mae.map_or(observation.foreground_roi.alpha_mae, |best| {
                    best.min(observation.foreground_roi.alpha_mae)
                }),
            );
            best_spill = Some(
                best_spill.map_or(observation.spill_outside_dilated_reference, |best| {
                    best.min(observation.spill_outside_dilated_reference)
                }),
            );
        }
    }

    fn assert_trellis_scale_quality(sweep: &TrellisDistanceSweep) {
        let label = sweep.camera.label;
        let exact = point_at_quality(&sweep.rendered, 1.0);
        assert_exact_trellis_point(label, exact);
        let q95 = point_at_quality(&sweep.rendered, 0.95);
        let q99 = point_at_quality(&sweep.rendered, 0.99);
        assert_thresholds(
            &format!("Trellis {label} q=.95"),
            q95.observation,
            Q95_THRESHOLDS,
        );
        assert_thresholds(
            &format!("Trellis {label} q=.99"),
            q99.observation,
            Q99_THRESHOLDS,
        );
        assert_covariance_major_sigma_bound(&format!("Trellis {label} q=.95"), q95, exact);
        assert_covariance_major_sigma_bound(&format!("Trellis {label} q=.99"), q99, exact);

        for goal in UTILITY_GOALS {
            assert!(
                utility_anchor(&sweep.rendered, goal).is_some(),
                "Trellis {label} has no interior cut meeting {} across the full slider: {:?}",
                goal.label,
                sweep.rendered
            );
        }
    }

    fn assert_trellis_authored_safety(sweep: &[QualitySweepPoint]) {
        assert_trellis_morphology_safety("authored", sweep);
        let q95 = point_at_quality(sweep, 0.95);
        let q99 = point_at_quality(sweep, 0.99);
        assert_thresholds("Trellis authored q=.95", q95.observation, Q95_THRESHOLDS);
        assert_thresholds("Trellis authored q=.99", q99.observation, Q99_THRESHOLDS);
        assert_monotonic(
            "Trellis authored q=.95 -> q=.99",
            q95.observation,
            q99.observation,
        );
    }

    fn assert_trellis_morphology_safety(label: &str, sweep: &[QualitySweepPoint]) {
        assert_eq!(
            sweep.len(),
            TRELLIS_MORPHOLOGY_QUALITIES.len(),
            "Trellis {label} morphology sweep length drifted"
        );
        for (point, expected_quality) in sweep.iter().zip(TRELLIS_MORPHOLOGY_QUALITIES) {
            assert!(
                quality_eq(point.quality, expected_quality),
                "Trellis {label} morphology sweep quality drifted: expected={expected_quality}, point={point:?}"
            );
        }
        for pair in sweep.windows(2) {
            let [lower, higher] = pair else {
                unreachable!()
            };
            assert!(
                higher.active_gaussians >= lower.active_gaussians,
                "Trellis {label} morphology active count regressed with quality: lower={lower:?}, higher={higher:?}"
            );
        }

        let exact = point_at_quality(sweep, 1.0);
        assert_exact_trellis_point(label, exact);
        for point in sweep.iter().filter(|point| point.quality < 1.0) {
            let point_label = format!("Trellis {label} q={:.2}", point.quality);
            assert_covariance_morphology_bound(&point_label, point, exact);
            assert!(
                point.covariance.visible_elongated_splats
                    <= exact.covariance.visible_elongated_splats,
                "{point_label} adds opacity-visible elongated splats beyond the same-view q=1 baseline (major sigma >{VISIBLE_ELONGATION_MIN_MAJOR_SIGMA_PX}px and aspect >{VISIBLE_ELONGATION_MIN_ASPECT_RATIO}:1): candidate={}, exact={}, point={point:?}",
                point.covariance.visible_elongated_splats,
                exact.covariance.visible_elongated_splats,
            );
            assert_alpha_morphology_bound(&point_label, point.observation);
        }
    }

    fn assert_covariance_major_sigma_bound(
        label: &str,
        candidate: &QualitySweepPoint,
        exact: &QualitySweepPoint,
    ) {
        let candidate_sigma = candidate.covariance.maximum_major_sigma_px;
        let exact_sigma = exact.covariance.maximum_major_sigma_px;
        assert!(candidate_sigma.is_finite() && exact_sigma.is_finite());
        let conservative_limit = exact_sigma * 1.25 + 1.0;
        assert!(
            candidate_sigma <= conservative_limit,
            "{label} maximum projected major sigma exceeds 1.25x exact plus one pixel: candidate={candidate_sigma}, exact={exact_sigma}, limit={conservative_limit}"
        );
    }

    fn assert_covariance_morphology_bound(
        label: &str,
        candidate: &QualitySweepPoint,
        exact: &QualitySweepPoint,
    ) {
        assert_covariance_major_sigma_bound(label, candidate, exact);
        assert!(
            candidate.covariance.projected_splats > 0 && exact.covariance.projected_splats > 0,
            "{label} has no valid projected covariance samples: candidate={candidate:?}, exact={exact:?}"
        );
        let candidate_aspect = candidate.covariance.maximum_aspect_ratio;
        let exact_aspect = exact.covariance.maximum_aspect_ratio;
        assert!(candidate_aspect.is_finite() && exact_aspect.is_finite());
        let conservative_limit = (exact_aspect * 2.0).max(8.0);
        assert!(
            candidate_aspect <= conservative_limit,
            "{label} maximum projected aspect exceeds max(8, 2x exact): candidate={candidate_aspect}, exact={exact_aspect}, limit={conservative_limit}"
        );
    }

    fn assert_exact_trellis_point(label: &str, point: &QualitySweepPoint) {
        assert_eq!(point.active_gaussians, TRELLIS_SPLAT_COUNT);
        assert!(
            point.observation.full.psnr_rgb.is_infinite()
                && point.observation.full.foreground_psnr_rgb.is_infinite()
                && point.observation.full.max_abs_error == 0.0
                && point.observation.foreground_roi.alpha_mae == 0.0
                && point.observation.spill_outside_dilated_reference == 0.0
                && point
                    .observation
                    .foreground_spill_pixels_outside_dilated_reference
                    == 0
                && point.observation.maximum_foreground_distance_px == 0,
            "Trellis {label} quality one did not restore the exact reference: {point:?}"
        );
    }

    fn foreground_union_roi(
        reference: &[[f32; 4]],
        candidate: &[[f32; 4]],
        width: u32,
        height: u32,
        margin: usize,
    ) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
        assert_eq!(reference.len(), (width * height) as usize);
        assert_eq!(candidate.len(), reference.len());
        let width = width as usize;
        let height = height as usize;
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0;
        let mut max_y = 0;
        let mut found = false;
        for (index, (expected, actual)) in reference.iter().zip(candidate).enumerate() {
            if expected[3] <= FOREGROUND_ALPHA && actual[3] <= FOREGROUND_ALPHA {
                continue;
            }
            let x = index % width;
            let y = index / width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            found = true;
        }
        assert!(found, "quality comparison has no foreground");
        min_x = min_x.saturating_sub(margin);
        min_y = min_y.saturating_sub(margin);
        max_x = (max_x + margin).min(width - 1);
        max_y = (max_y + margin).min(height - 1);

        let mut reference_roi = Vec::with_capacity((max_x - min_x + 1) * (max_y - min_y + 1));
        let mut candidate_roi = Vec::with_capacity(reference_roi.capacity());
        for y in min_y..=max_y {
            let start = y * width + min_x;
            let end = y * width + max_x + 1;
            reference_roi.extend_from_slice(&reference[start..end]);
            candidate_roi.extend_from_slice(&candidate[start..end]);
        }
        (reference_roi, candidate_roi)
    }

    #[derive(Clone, Copy, Debug)]
    struct AlphaSpillDiagnostics {
        alpha_mass_ratio: f64,
        foreground_pixels_outside_dilated_reference: usize,
        maximum_foreground_distance_px: usize,
    }

    fn alpha_spill_diagnostics(
        reference: &[[f32; 4]],
        candidate: &[[f32; 4]],
        width: u32,
        height: u32,
        dilation: usize,
    ) -> AlphaSpillDiagnostics {
        let width = width as usize;
        let height = height as usize;
        assert_eq!(reference.len(), width * height);
        assert_eq!(candidate.len(), reference.len());
        let unreachable = width.max(height) + 1;
        let mut foreground_distance = vec![unreachable; reference.len()];
        for (index, pixel) in reference.iter().enumerate() {
            if pixel[3] >= FOREGROUND_ALPHA {
                foreground_distance[index] = 0;
            }
        }
        assert!(
            foreground_distance.iter().any(|distance| *distance == 0),
            "alpha spill reference has no foreground"
        );

        // Exact two-pass chessboard-distance transform. Its <= dilation mask
        // is identical to the square dilation used by the spill contract, and
        // it also reports how far an opacity-visible candidate reaches beyond
        // the exact q=1 geometry.
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                let mut best = foreground_distance[index];
                if x > 0 {
                    best = best.min(foreground_distance[index - 1].saturating_add(1));
                }
                if y > 0 {
                    for neighbor_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                        best = best.min(
                            foreground_distance[(y - 1) * width + neighbor_x].saturating_add(1),
                        );
                    }
                }
                foreground_distance[index] = best;
            }
        }
        for y in (0..height).rev() {
            for x in (0..width).rev() {
                let index = y * width + x;
                let mut best = foreground_distance[index];
                if x + 1 < width {
                    best = best.min(foreground_distance[index + 1].saturating_add(1));
                }
                if y + 1 < height {
                    for neighbor_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                        best = best.min(
                            foreground_distance[(y + 1) * width + neighbor_x].saturating_add(1),
                        );
                    }
                }
                foreground_distance[index] = best;
            }
        }

        let reference_alpha = reference
            .iter()
            .map(|pixel| f64::from(pixel[3]))
            .sum::<f64>();
        assert!(reference_alpha > 0.0);
        let alpha_mass_ratio = candidate
            .iter()
            .zip(&foreground_distance)
            .filter(|(_, distance)| **distance > dilation)
            .map(|(pixel, _)| f64::from(pixel[3]))
            .sum::<f64>()
            / reference_alpha;
        let foreground_pixels_outside_dilated_reference = candidate
            .iter()
            .zip(&foreground_distance)
            .filter(|(pixel, distance)| pixel[3] >= FOREGROUND_ALPHA && **distance > dilation)
            .count();
        let maximum_foreground_distance_px = candidate
            .iter()
            .zip(foreground_distance)
            .filter_map(|(pixel, distance)| (pixel[3] >= FOREGROUND_ALPHA).then_some(distance))
            .max()
            .unwrap_or(0);
        AlphaSpillDiagnostics {
            alpha_mass_ratio,
            foreground_pixels_outside_dilated_reference,
            maximum_foreground_distance_px,
        }
    }

    #[test]
    fn alpha_spill_diagnostics_tracks_direct_foreground_distance() {
        let mut reference = vec![[0.0; 4]; 49];
        let mut candidate = reference.clone();
        reference[3 * 7 + 3][3] = 1.0;
        candidate[3 * 7 + 3][3] = 1.0;
        candidate[3 * 7 + 5][3] = FOREGROUND_ALPHA;
        candidate[3 * 7 + 6][3] = FOREGROUND_ALPHA;

        let diagnostics = alpha_spill_diagnostics(&reference, &candidate, 7, 7, 2);
        assert_eq!(diagnostics.foreground_pixels_outside_dilated_reference, 1);
        assert_eq!(diagnostics.maximum_foreground_distance_px, 3);
        assert!((diagnostics.alpha_mass_ratio - f64::from(FOREGROUND_ALPHA)).abs() <= 1e-9);
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
        srgb_display_channel_to_linear(f32::from(value) / 255.0)
    }
}
