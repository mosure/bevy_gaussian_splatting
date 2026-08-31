#[cfg(not(feature = "headless"))]
#[test]
fn support_overlap_render_test_requires_headless_feature() {}

#[cfg(feature = "headless")]
mod headless {
    use std::{env, sync::Mutex, time::Duration};

    use bevy::{
        app::{AppExit, ScheduleRunnerPlugin},
        camera::{RenderTarget, visibility::NoFrustumCulling},
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
        CloudSettings, Gaussian3d, GaussianCamera, GaussianLodSettings, GaussianMode,
        GaussianSplattingPlugin, PlanarGaussian3d, PlanarGaussian3dHandle,
        SphericalHarmonicCoefficients, gaussian::f32::Rotation, sort::SortMode,
    };

    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 128;
    const WARMUP_FRAMES: u32 = 45;
    const MAX_FRAMES: u32 = 120;
    const RIGHT_EDGE_WIDTH: usize = 16;
    const LEFT_EDGE_WIDTH: usize = 4;
    static GPU_RENDER_TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn gaussian_with_offscreen_center_and_visible_support_reaches_pixels() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!(
                "skipping GPU support-overlap render test; set RUN_GPU_RENDER_TESTS=1 to enable"
            );
            return;
        }
        let _gpu_render_guard = GPU_RENDER_TEST_MUTEX
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let mut app = App::new();
        app.insert_resource(ClearColor(Color::BLACK))
            .insert_resource(SupportOverlapState::default());
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
        app.add_plugins(GaussianSplattingPlugin)
            .add_systems(Startup, setup)
            .add_systems(Update, drive_capture)
            .add_observer(on_capture);
        app.run();
    }

    #[test]
    fn non_unit_global_scale_matches_authored_scale_controls() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!(
                "skipping GPU global-scale parity test; set RUN_GPU_RENDER_TESTS=1 to enable"
            );
            return;
        }
        let _gpu_render_guard = GPU_RENDER_TEST_MUTEX
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let mut app = App::new();
        app.insert_resource(ClearColor(Color::BLACK))
            .insert_resource(ScaleParityState::default());
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
        app.add_plugins(GaussianSplattingPlugin)
            .add_systems(Startup, setup_scale_parity)
            .add_systems(Update, drive_scale_parity_capture)
            .add_observer(on_scale_parity_capture);
        app.run();
    }

    #[derive(Resource, Default)]
    struct SupportOverlapState {
        frames: u32,
        capture_pending: bool,
        target: Option<Handle<Image>>,
    }

    #[derive(Resource, Default)]
    struct ScaleParityState {
        frames: u32,
        capture_pending: bool,
        target: Option<Handle<Image>>,
    }

    fn setup(
        mut commands: Commands,
        mut state: ResMut<SupportOverlapState>,
        mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let target = images.add(Image::new_target_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        state.target = Some(target.clone());

        commands.spawn((
            PlanarGaussian3dHandle(gaussian_assets.add(support_overlap_cloud())),
            CloudSettings {
                gaussian_mode: GaussianMode::Gaussian3d,
                sort_mode: SortMode::Radix,
                global_opacity: 256.0,
                global_scale: 1.0,
                opacity_adaptive_radius: false,
                ..default()
            },
            GaussianLodSettings::default(),
            // Keep main-world position-AABB culling from hiding this deliberate
            // offscreen-center fixture before support-aware GPU culling runs.
            NoFrustumCulling,
            Transform::default(),
            Visibility::Visible,
            Name::new("support_overlap_test_cloud"),
        ));

        commands.spawn((
            Camera3d::default(),
            Camera::default(),
            RenderTarget::Image(target.into()),
            Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)),
            Tonemapping::None,
            GaussianCamera::default(),
        ));
    }

    fn support_overlap_cloud() -> PlanarGaussian3d {
        let mut red = SphericalHarmonicCoefficients::default();
        red.set(0, 6.0);
        let mut blue = SphericalHarmonicCoefficients::default();
        blue.set(2, 6.0);
        let rotation = Rotation {
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        let mut gaussians = [2.2, 2.35]
            .into_iter()
            .flat_map(|x| {
                [-0.2, 0.2].into_iter().map(move |y| Gaussian3d {
                    // At z=0 the default camera's right plane is near x=2.07.
                    // Every center is outside, while three-sigma support stays
                    // well inside the viewport.
                    position_visibility: [x, y, 0.0, 1.0].into(),
                    rotation,
                    scale_opacity: [0.65, 0.65, 0.65, 0.9].into(),
                    spherical_harmonic: red,
                })
            })
            .collect::<Vec<_>>();
        gaussians.push(Gaussian3d {
            // The default camera edge is x=-2.071 at z=0. This center and its
            // 3*0.001 authored sphere are wholly outside, but the fixed
            // 3-sigma mip footprint reaches roughly one pixel into the image.
            position_visibility: [-2.086, 0.0, 0.0, 1.0].into(),
            rotation,
            scale_opacity: [0.001, 0.001, 0.001, 0.9].into(),
            spherical_harmonic: blue,
        });
        gaussians.into()
    }

    fn setup_scale_parity(
        mut commands: Commands,
        mut state: ResMut<ScaleParityState>,
        mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let target = images.add(Image::new_target_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        state.target = Some(target.clone());

        let base_scale = Vec3::new(0.06, 0.10, 0.04);
        for (position, authored_scale, global_scale, name) in [
            (
                Vec3::new(-0.75, -0.8, 0.0),
                base_scale,
                2.0,
                "global_scale_positive_two",
            ),
            (
                Vec3::new(0.75, -0.8, 0.0),
                base_scale * 2.0,
                1.0,
                "authored_scale_positive_control",
            ),
            (
                Vec3::new(-0.75, 0.8, 0.0),
                base_scale,
                -2.0,
                "global_scale_negative_two",
            ),
            (
                Vec3::new(0.75, 0.8, 0.0),
                base_scale * 2.0,
                1.0,
                "authored_scale_negative_control",
            ),
        ] {
            spawn_scale_parity_cloud(
                &mut commands,
                &mut gaussian_assets,
                position,
                authored_scale,
                global_scale,
                name,
            );
        }

        commands.spawn((
            Camera3d::default(),
            Camera::default(),
            RenderTarget::Image(target.into()),
            Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)),
            Tonemapping::None,
            GaussianCamera::default(),
        ));
    }

    fn spawn_scale_parity_cloud(
        commands: &mut Commands,
        gaussian_assets: &mut Assets<PlanarGaussian3d>,
        position: Vec3,
        authored_scale: Vec3,
        global_scale: f32,
        name: &'static str,
    ) {
        let mut green = SphericalHarmonicCoefficients::default();
        green.set(1, 1.0);
        let gaussian = Gaussian3d {
            position_visibility: [position.x, position.y, position.z, 1.0].into(),
            rotation: Rotation {
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
            scale_opacity: [authored_scale.x, authored_scale.y, authored_scale.z, 0.8].into(),
            spherical_harmonic: green,
        };
        commands.spawn((
            PlanarGaussian3dHandle(gaussian_assets.add(PlanarGaussian3d::from(vec![gaussian]))),
            CloudSettings {
                gaussian_mode: GaussianMode::Gaussian3d,
                sort_mode: SortMode::Radix,
                global_opacity: 1.0,
                global_scale,
                opacity_adaptive_radius: false,
                ..default()
            },
            GaussianLodSettings::default(),
            NoFrustumCulling,
            Transform::default(),
            Visibility::Visible,
            Name::new(name),
        ));
    }

    fn drive_capture(mut commands: Commands, mut state: ResMut<SupportOverlapState>) {
        state.frames += 1;
        assert!(
            state.frames <= MAX_FRAMES,
            "support-overlap GPU test timed out"
        );
        if state.capture_pending || state.frames < WARMUP_FRAMES {
            return;
        }
        commands.spawn(Screenshot::image(
            state.target.clone().expect("render target exists"),
        ));
        state.capture_pending = true;
    }

    fn drive_scale_parity_capture(mut commands: Commands, mut state: ResMut<ScaleParityState>) {
        state.frames += 1;
        assert!(
            state.frames <= MAX_FRAMES,
            "global-scale parity GPU test timed out"
        );
        if state.capture_pending || state.frames < WARMUP_FRAMES {
            return;
        }
        commands.spawn(Screenshot::image(
            state.target.clone().expect("render target exists"),
        ));
        state.capture_pending = true;
    }

    fn on_capture(trigger: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>) {
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("screenshot converts")
            .to_rgba8();
        let mut total = 0usize;
        let mut right_edge = 0usize;
        let mut left_edge_mip_blue = 0usize;
        for (index, pixel) in rgba.as_raw().chunks_exact(4).enumerate() {
            let visible = pixel[0].max(pixel[1]).max(pixel[2]) > 8;
            let x = index % WIDTH as usize;
            total += usize::from(visible);
            right_edge += usize::from(visible && x >= WIDTH as usize - RIGHT_EDGE_WIDTH);
            left_edge_mip_blue += usize::from(
                x < LEFT_EDGE_WIDTH && pixel[2] > 12 && pixel[2] > pixel[0].saturating_add(8),
            );
        }
        assert!(total >= 16, "support-overlap render stayed black: {total}");
        assert!(
            right_edge >= 8,
            "offscreen-center Gaussian did not reach the right edge: total={total}, right_edge={right_edge}"
        );
        assert!(
            left_edge_mip_blue >= 1,
            "mip-dominated offscreen Gaussian was support-culled: blue_pixels={left_edge_mip_blue}"
        );
        exit.write(AppExit::Success);
    }

    fn on_scale_parity_capture(trigger: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>) {
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("screenshot converts")
            .to_rgba8();
        let bytes = rgba.as_raw();
        assert_mirrored_scale_pair(bytes, 0, HEIGHT as usize / 2, "first scale pair");
        assert_mirrored_scale_pair(
            bytes,
            HEIGHT as usize / 2,
            HEIGHT as usize,
            "second scale pair",
        );
        exit.write(AppExit::Success);
    }

    fn assert_mirrored_scale_pair(bytes: &[u8], y_start: usize, y_end: usize, label: &str) {
        let mut left_mask = 0usize;
        let mut right_mask = 0usize;
        let mut mask_mismatches = 0usize;
        let mut left_energy = 0u64;
        let mut right_energy = 0u64;
        let mut max_channel_difference = 0u8;
        for y in y_start..y_end {
            for left_x in 0..WIDTH as usize / 2 {
                let right_x = WIDTH as usize - 1 - left_x;
                let left = &bytes[(y * WIDTH as usize + left_x) * 4..][..4];
                let right = &bytes[(y * WIDTH as usize + right_x) * 4..][..4];
                let left_visible = left[1] > 8;
                let right_visible = right[1] > 8;
                left_mask += usize::from(left_visible);
                right_mask += usize::from(right_visible);
                mask_mismatches += usize::from(left_visible != right_visible);
                left_energy += u64::from(left[1]);
                right_energy += u64::from(right[1]);
                for channel in 0..3 {
                    max_channel_difference =
                        max_channel_difference.max(left[channel].abs_diff(right[channel]));
                }
            }
        }

        assert!(
            left_mask >= 64 && right_mask >= 64,
            "{label} did not render enough support: left={left_mask}, right={right_mask}"
        );
        assert!(
            mask_mismatches <= 2,
            "{label} footprint masks diverged: left={left_mask}, right={right_mask}, mismatches={mask_mismatches}"
        );
        let energy_difference = left_energy.abs_diff(right_energy);
        let energy_tolerance = left_energy.max(right_energy) / 100 + 4;
        assert!(
            energy_difference <= energy_tolerance,
            "{label} energy diverged: left={left_energy}, right={right_energy}, difference={energy_difference}, tolerance={energy_tolerance}"
        );
        assert!(
            max_channel_difference <= 4,
            "{label} per-channel parity diverged: max_difference={max_channel_difference}"
        );
    }
}
