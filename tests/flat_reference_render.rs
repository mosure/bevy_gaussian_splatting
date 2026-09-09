//! Portable static-image oracle harness, deliberately usable unchanged on main.
//! Run with BGS_FLAT_REFERENCE_CONFIG and RUN_GPU_RENDER_TESTS=1. It renders the
//! entire source through the flat pipeline; it does not claim draw-count or
//! timing telemetry. The outer experiment runner hashes input/output artifacts.
#[cfg(feature = "headless")]
mod headless {
    use std::{
        fs,
        io::BufReader,
        path::PathBuf,
        time::{Duration, Instant},
    };

    use bevy::{
        app::{AppExit, ScheduleRunnerPlugin},
        camera::{PerspectiveProjection, Projection, RenderTarget},
        core_pipeline::tonemapping::Tonemapping,
        prelude::*,
        render::{
            pipelined_rendering::PipelinedRenderingPlugin,
            render_resource::TextureFormat,
            view::screenshot::{Screenshot, ScreenshotCaptured},
        },
        window::ExitCondition,
        winit::WinitPlugin,
    };
    use bevy_gaussian_splatting::{
        CloudSettings, GaussianCamera, GaussianSplattingPlugin, PlanarGaussian3d,
        PlanarGaussian3dHandle, io::ply::parse_ply_3d, sort::SortMode,
    };
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Pose {
        name: String,
        position: [f32; 3],
        target: [f32; 3],
        #[serde(default = "default_up")]
        up: [f32; 3],
    }
    fn default_up() -> [f32; 3] {
        [0.0, 1.0, 0.0]
    }
    #[derive(Deserialize)]
    struct Config {
        source: PathBuf,
        output: PathBuf,
        viewport: [u32; 2],
        vertical_fov_radians: f32,
        near: f32,
        poses: Vec<Pose>,
        /// Diagnostic shader ablation only. The exact source is copied into
        /// the output so a main/branch comparison cannot hide a changed filter.
        #[serde(default)]
        helpers_override: Option<PathBuf>,
    }
    #[derive(Resource)]
    struct Session {
        config: Config,
        source: Option<PlanarGaussian3d>,
        target: Option<Handle<Image>>,
        camera: Option<Entity>,
        pose: usize,
        frames: u32,
        pending: bool,
        started: Instant,
    }

    #[test]
    fn capture_flat_source_references() {
        if std::env::var("RUN_GPU_RENDER_TESTS").as_deref() != Ok("1") {
            return;
        }
        let Ok(path) = std::env::var("BGS_FLAT_REFERENCE_CONFIG") else {
            return;
        };
        let bytes = fs::read(path).unwrap();
        let config: Config = serde_json::from_slice(&bytes).unwrap();
        assert!(!config.poses.is_empty());
        assert!(
            config
                .viewport
                .into_iter()
                .all(|v| (11..=8192).contains(&v))
        );
        assert!(config.near.is_finite() && config.near > 0.0);
        assert!(
            config.vertical_fov_radians.is_finite()
                && (0.0..std::f32::consts::PI).contains(&config.vertical_fov_radians)
        );
        for pose in &config.poses {
            assert!(
                !pose.name.is_empty()
                    && pose
                        .name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
            );
            assert!(
                Vec3::from_array(pose.position).is_finite()
                    && Vec3::from_array(pose.target).is_finite()
                    && Vec3::from_array(pose.up).is_finite()
                    && Vec3::from_array(pose.up).length_squared() > 1e-8
                    && (Vec3::from_array(pose.target) - Vec3::from_array(pose.position))
                        .normalize_or_zero()
                        .cross(Vec3::from_array(pose.up).normalize_or_zero())
                        .length_squared()
                        > 1e-8
            );
            assert!(
                Vec3::from_array(pose.position).distance_squared(Vec3::from_array(pose.target))
                    > 1e-6
            );
        }
        fs::create_dir(&config.output).expect("reference output directory must be new");
        fs::write(config.output.join("settings.json"), bytes).unwrap();
        let helpers_override = config.helpers_override.as_ref().map(|path| {
            assert!(fs::metadata(path).unwrap().len() <= 1024 * 1024);
            let source = fs::read_to_string(path).unwrap();
            fs::write(config.output.join("helpers-override.wgsl"), &source).unwrap();
            source
        });
        let cloud =
            parse_ply_3d(&mut BufReader::new(fs::File::open(&config.source).unwrap())).unwrap();
        let mut app = App::new();
        #[cfg(feature = "lod")]
        app.insert_resource(
            bevy_gaussian_splatting::stream::bridge::GaussianLodBridgeConfig {
                auto_build_flat_clouds: false,
                ..default()
            },
        );
        app.insert_resource(Session {
            config,
            source: Some(cloud),
            target: None,
            camera: None,
            pose: 0,
            frames: 0,
            pending: false,
            started: Instant::now(),
        })
        .insert_resource(ClearColor(Color::linear_rgba(0.0, 0.0, 0.0, 0.0)))
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>()
                .disable::<PipelinedRenderingPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 240.0,
        )))
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, advance)
        .add_observer(captured);
        if let Some(source) = helpers_override {
            let handle: Handle<Shader> =
                bevy::asset::uuid_handle!("9ca57ab0-07de-4a43-94f8-547c38e292cb");
            app.world_mut()
                .resource_mut::<Assets<Shader>>()
                .insert(
                    handle.id(),
                    Shader::from_wgsl(source, "flat_reference_helpers_override.wgsl"),
                )
                .unwrap();
        }
        assert!(app.run().is_success());
    }

    fn setup(
        mut commands: Commands,
        mut state: ResMut<Session>,
        mut assets: ResMut<Assets<PlanarGaussian3d>>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let cloud = assets.add(state.source.take().unwrap());
        commands.spawn((
            PlanarGaussian3dHandle(cloud),
            CloudSettings {
                sort_mode: SortMode::Radix,
                opacity_adaptive_radius: false,
                ..default()
            },
            Transform::IDENTITY,
            Visibility::Visible,
        ));
        let [width, height] = state.config.viewport;
        let target = images.add(Image::new_target_texture(
            width,
            height,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let pose = &state.config.poses[0];
        let camera = commands
            .spawn((
                Camera3d::default(),
                Camera::default(),
                Projection::Perspective(PerspectiveProjection {
                    fov: state.config.vertical_fov_radians,
                    near: state.config.near,
                    far: 100_000.0,
                    ..default()
                }),
                RenderTarget::Image(target.clone().into()),
                Transform::from_translation(Vec3::from_array(pose.position))
                    .looking_at(Vec3::from_array(pose.target), Vec3::from_array(pose.up)),
                GaussianCamera::default(),
                Tonemapping::None,
                Msaa::Off,
            ))
            .id();
        state.target = Some(target);
        state.camera = Some(camera);
    }

    fn advance(mut commands: Commands, mut state: ResMut<Session>) {
        assert!(
            state.started.elapsed() < Duration::from_secs(180),
            "flat reference capture timed out"
        );
        if state.pending || state.pose >= state.config.poses.len() {
            return;
        }
        state.frames += 1;
        if state.frames < 120 {
            return;
        }
        commands.spawn(Screenshot::image(state.target.clone().unwrap()));
        state.pending = true;
    }

    fn captured(
        event: On<ScreenshotCaptured>,
        mut state: ResMut<Session>,
        mut cameras: Query<&mut Transform, With<GaussianCamera>>,
        mut exit: MessageWriter<AppExit>,
    ) {
        assert!(state.pending);
        let image = event.image.clone().try_into_dynamic().unwrap().to_rgba8();
        let nonblack = image
            .pixels()
            .filter(|p| p.0[..3].iter().any(|&c| c > 2))
            .count();
        if nonblack == 0 {
            state.pending = false;
            state.frames = 100;
            return;
        }
        let pose = &state.config.poses[state.pose];
        image
            .save(state.config.output.join(format!("{}.png", pose.name)))
            .unwrap();
        state.pose += 1;
        state.frames = 0;
        state.pending = false;
        if state.pose == state.config.poses.len() {
            fs::write(state.config.output.join("status.json"),
                "{\"static_images_complete\":true,\"draw_counts_observed\":false,\"performance_observed\":false}\n").unwrap();
            exit.write(AppExit::Success);
        } else {
            let next = &state.config.poses[state.pose];
            *cameras.get_mut(state.camera.unwrap()).unwrap() =
                Transform::from_translation(Vec3::from_array(next.position))
                    .looking_at(Vec3::from_array(next.target), Vec3::from_array(next.up));
        }
    }
}
