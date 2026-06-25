#[cfg(not(feature = "headless"))]
#[test]
fn visibility_render_test_requires_headless_feature() {}

#[cfg(feature = "headless")]
mod headless {
    use std::{
        env,
        path::PathBuf,
        time::{Duration, Instant},
    };

    use bevy::{
        app::{AppExit, ScheduleRunnerPlugin},
        camera::RenderTarget,
        core_pipeline::tonemapping::Tonemapping,
        prelude::*,
        render::{
            render_resource::{Extent3d, TextureFormat},
            view::screenshot::{Screenshot, ScreenshotCaptured},
        },
        window::ExitCondition,
        winit::WinitPlugin,
    };
    use bevy_gaussian_splatting::{
        CloudSettings, Gaussian3d, GaussianCamera, GaussianMode, GaussianSplattingPlugin,
        PlanarGaussian3d, PlanarGaussian3dHandle, SphericalHarmonicCoefficients,
        gaussian::f32::Rotation, sort::SortMode,
    };

    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 128;
    const VISIBLE_WARMUP_FRAMES: u32 = 45;
    const HIDDEN_WARMUP_FRAMES: u32 = 18;
    const MAX_FRAMES: u32 = 180;
    const VISIBLE_NON_BLACK_MIN: usize = 64;
    const HIDDEN_NON_BLACK_MAX: usize = 8;

    #[test]
    fn hidden_gaussian_cloud_clears_render_target() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping GPU visibility render test; set RUN_GPU_RENDER_TESTS=1 to enable");
            return;
        }

        let mut app = App::new();
        app.insert_resource(ClearColor(Color::srgb_u8(0, 0, 0)))
            .insert_resource(VisibilityRenderState::default());
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
        app.add_plugins(GaussianSplattingPlugin);
        app.add_systems(Startup, setup_visibility_scene)
            .add_systems(Update, drive_visibility_capture)
            .add_observer(on_screenshot_captured);
        app.run();
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CapturePhase {
        VisibleWarmup,
        VisiblePending,
        HiddenWarmup,
        HiddenPending,
    }

    #[derive(Debug, Resource)]
    struct VisibilityRenderState {
        phase: CapturePhase,
        phase_frames: u32,
        total_frames: u32,
        started_at: Instant,
        cloud: Option<Entity>,
        target: Option<Handle<Image>>,
        visible_stats: Option<ImageStats>,
    }

    impl Default for VisibilityRenderState {
        fn default() -> Self {
            Self {
                phase: CapturePhase::VisibleWarmup,
                phase_frames: 0,
                total_frames: 0,
                started_at: Instant::now(),
                cloud: None,
                target: None,
                visible_stats: None,
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct ImageStats {
        non_black_pixels: usize,
        max_channel: u8,
    }

    fn setup_visibility_scene(
        mut commands: Commands,
        mut state: ResMut<VisibilityRenderState>,
        mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let size = Extent3d {
            width: WIDTH,
            height: HEIGHT,
            ..default()
        };
        let render_target = images.add(Image::new_target_texture(
            size.width,
            size.height,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        state.target = Some(render_target.clone());

        let cloud = gaussian_assets.add(visibility_test_cloud());
        let cloud_entity = commands
            .spawn((
                PlanarGaussian3dHandle(cloud),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::None,
                    global_opacity: 2.0,
                    global_scale: 1.0,
                    opacity_adaptive_radius: false,
                    ..default()
                },
                Transform::default(),
                Visibility::Visible,
                Name::new("visibility_test_cloud"),
            ))
            .id();
        state.cloud = Some(cloud_entity);

        commands.spawn((
            Camera3d::default(),
            Camera::default(),
            RenderTarget::Image(render_target.into()),
            Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)),
            Tonemapping::None,
            GaussianCamera::default(),
        ));
    }

    fn drive_visibility_capture(mut commands: Commands, mut state: ResMut<VisibilityRenderState>) {
        state.total_frames += 1;
        state.phase_frames += 1;

        if state.total_frames > MAX_FRAMES {
            panic!(
                "visibility render test timed out after {} frames in phase {:?} ({:?} elapsed)",
                state.total_frames,
                state.phase,
                state.started_at.elapsed()
            );
        }

        let should_capture = match state.phase {
            CapturePhase::VisibleWarmup => state.phase_frames >= VISIBLE_WARMUP_FRAMES,
            CapturePhase::HiddenWarmup => state.phase_frames >= HIDDEN_WARMUP_FRAMES,
            CapturePhase::VisiblePending | CapturePhase::HiddenPending => false,
        };

        if !should_capture {
            return;
        }

        let Some(target) = state.target.clone() else {
            return;
        };
        commands.spawn(Screenshot::image(target));
        state.phase = match state.phase {
            CapturePhase::VisibleWarmup => CapturePhase::VisiblePending,
            CapturePhase::HiddenWarmup => CapturePhase::HiddenPending,
            phase => phase,
        };
        state.phase_frames = 0;
    }

    fn visibility_test_cloud() -> PlanarGaussian3d {
        let mut red = SphericalHarmonicCoefficients::default();
        red.set(0, 6.0);

        let rotation = Rotation {
            rotation: [1.0, 0.0, 0.0, 0.0],
        };

        let mut gaussians = Vec::new();
        for x in [-0.35, 0.35] {
            for y in [-0.35, 0.35] {
                for z in [-0.35, 0.35] {
                    gaussians.push(Gaussian3d {
                        position_visibility: [x, y, z, 1.0].into(),
                        rotation,
                        scale_opacity: [0.22, 0.22, 0.22, 0.85].into(),
                        spherical_harmonic: red,
                    });
                }
            }
        }
        gaussians.push(gaussians[0]);
        gaussians.into()
    }

    fn on_screenshot_captured(
        trigger: On<ScreenshotCaptured>,
        mut commands: Commands,
        mut state: ResMut<VisibilityRenderState>,
        mut app_exit: MessageWriter<AppExit>,
    ) {
        let image = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("failed to convert screenshot image")
            .to_rgba8();
        let stats = image_stats(image.as_raw());

        match state.phase {
            CapturePhase::VisiblePending => {
                maybe_save_debug_image("visible.png", |path| {
                    image
                        .save(path)
                        .expect("failed to save visible debug image");
                });
                assert!(
                    stats.non_black_pixels >= VISIBLE_NON_BLACK_MIN,
                    "visible cloud did not render enough non-black pixels: {stats:?}"
                );
                assert!(
                    stats.max_channel > 32,
                    "visible cloud render was unexpectedly dim: {stats:?}"
                );

                let Some(cloud) = state.cloud else {
                    panic!("visibility render test cloud entity was not recorded");
                };
                commands.entity(cloud).insert(Visibility::Hidden);
                state.visible_stats = Some(stats);
                state.phase = CapturePhase::HiddenWarmup;
                state.phase_frames = 0;
            }
            CapturePhase::HiddenPending => {
                maybe_save_debug_image("hidden.png", |path| {
                    image.save(path).expect("failed to save hidden debug image");
                });
                let visible_stats = state
                    .visible_stats
                    .expect("hidden capture completed before visible capture");
                let hidden_relative_limit = (visible_stats.non_black_pixels / 20).max(1);
                assert!(
                    stats.non_black_pixels <= HIDDEN_NON_BLACK_MAX
                        || stats.non_black_pixels <= hidden_relative_limit,
                    "hidden cloud left visible pixels: visible={visible_stats:?}, hidden={stats:?}"
                );
                app_exit.write(AppExit::Success);
            }
            phase => panic!("screenshot captured during unexpected phase {phase:?}"),
        }
    }

    fn image_stats(rgba: &[u8]) -> ImageStats {
        let mut non_black_pixels = 0;
        let mut max_channel = 0;

        for pixel in rgba.chunks_exact(4) {
            let rgb_max = pixel[0].max(pixel[1]).max(pixel[2]);
            max_channel = max_channel.max(rgb_max);
            if rgb_max > 8 {
                non_black_pixels += 1;
            }
        }

        ImageStats {
            non_black_pixels,
            max_channel,
        }
    }

    fn maybe_save_debug_image(file_name: &str, save: impl FnOnce(PathBuf)) {
        let Some(dir) = env::var_os("VISIBILITY_RENDER_DEBUG_DIR") else {
            return;
        };

        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("failed to create visibility render debug directory");
        save(dir.join(file_name));
    }
}
