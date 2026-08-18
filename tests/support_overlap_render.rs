#[cfg(not(feature = "headless"))]
#[test]
fn support_overlap_render_test_requires_headless_feature() {}

#[cfg(feature = "headless")]
mod headless {
    use std::{env, time::Duration};

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

    #[test]
    fn gaussian_with_offscreen_center_and_visible_support_reaches_pixels() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!(
                "skipping GPU support-overlap render test; set RUN_GPU_RENDER_TESTS=1 to enable"
            );
            return;
        }

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

    #[derive(Resource, Default)]
    struct SupportOverlapState {
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
                global_opacity: 2.0,
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
        let rotation = Rotation {
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        [2.2, 2.35]
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
            .collect::<Vec<_>>()
            .into()
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

    fn on_capture(trigger: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>) {
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("screenshot converts")
            .to_rgba8();
        let mut total = 0usize;
        let mut right_edge = 0usize;
        for (index, pixel) in rgba.as_raw().chunks_exact(4).enumerate() {
            let visible = pixel[0].max(pixel[1]).max(pixel[2]) > 8;
            total += usize::from(visible);
            right_edge +=
                usize::from(visible && index % WIDTH as usize >= WIDTH as usize - RIGHT_EDGE_WIDTH);
        }
        assert!(total >= 16, "support-overlap render stayed black: {total}");
        assert!(
            right_edge >= 8,
            "offscreen-center Gaussian did not reach the right edge: total={total}, right_edge={right_edge}"
        );
        exit.write(AppExit::Success);
    }
}
