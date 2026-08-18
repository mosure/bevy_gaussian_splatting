#[cfg(not(feature = "headless"))]
#[test]
fn lod_debug_render_test_requires_headless_feature() {}

#[cfg(feature = "headless")]
mod headless {
    use std::{env, time::Duration};

    use bevy::{
        app::{AppExit, ScheduleRunnerPlugin},
        camera::RenderTarget,
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
        CloudSettings, Gaussian3d, GaussianCamera, GaussianLodBridgeConfig, GaussianLodSettings,
        GaussianMode, GaussianSplattingPlugin, LodDebugPreset, LodDebugSettings, Planar,
        PlanarGaussian3d, PlanarGaussian3dHandle, SphericalHarmonicCoefficients,
        gaussian::{
            f32::Rotation,
            lod_debug::{
                LodDebugMetadata, LodDebugRecord, LodDebugResidency, lod_debug_level_color,
                lod_debug_page_color, lod_debug_selection_pressure_color,
            },
        },
        sort::SortMode,
    };

    const WIDTH: u32 = 96;
    const HEIGHT: u32 = 96;
    const WARMUP_FRAMES: u32 = 45;
    const MAX_FRAMES: u32 = 720;
    const DEBUG_PAGE_COLOR_KEY: u32 = 0x4d2a_91c7;
    const DEBUG_QUALITY: f32 = 0.95;
    const DEBUG_QUALITY_THRESHOLD: f32 = 1.0;

    #[test]
    fn every_lod_debug_preset_reaches_gpu_pixels() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping GPU LoD debug render test; set RUN_GPU_RENDER_TESTS=1 to enable");
            return;
        }

        let mut app = App::new();
        app.insert_resource(ClearColor(Color::BLACK))
            .insert_resource(GaussianLodBridgeConfig {
                auto_build_flat_clouds: false,
                ..default()
            })
            .insert_resource(DebugRenderState::default());
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

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum DebugCase {
        Off,
        Level,
        Page,
        Residency,
        Boundaries,
        SelectionPressure,
    }

    impl DebugCase {
        const ALL: [Self; 6] = [
            Self::Off,
            Self::Level,
            Self::Page,
            Self::Residency,
            Self::Boundaries,
            Self::SelectionPressure,
        ];

        fn settings(self) -> LodDebugSettings {
            LodDebugSettings::from_preset(match self {
                Self::Off => LodDebugPreset::Off,
                Self::Level => LodDebugPreset::Level,
                Self::Page => LodDebugPreset::Page,
                Self::Residency => LodDebugPreset::Residency,
                Self::Boundaries => LodDebugPreset::Boundaries,
                Self::SelectionPressure => LodDebugPreset::SelectionPressure,
            })
        }
    }

    #[derive(Resource, Default)]
    struct DebugRenderState {
        case_index: usize,
        phase_frames: u32,
        total_frames: u32,
        capture_pending: bool,
        cloud: Option<Entity>,
        target: Option<Handle<Image>>,
    }

    impl DebugRenderState {
        fn current_case(&self) -> DebugCase {
            DebugCase::ALL[self.case_index]
        }
    }

    fn setup(
        mut commands: Commands,
        mut state: ResMut<DebugRenderState>,
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

        let cloud = debug_test_cloud();
        let record_count = cloud.len();
        let cloud_handle = gaussian_assets.add(cloud);
        let metadata = LodDebugMetadata::new(vec![
            LodDebugRecord {
                page_color_key: DEBUG_PAGE_COLOR_KEY,
                hierarchy_depth: 7,
                residency: LodDebugResidency::AncestorFallback as u32,
                boundary_distance_bits: 0.0_f32.to_bits(),
                geometric_error: 0.125,
                quality_threshold: DEBUG_QUALITY_THRESHOLD,
                node_center: [0.0, 0.0, 0.0],
                node_radius: 1.0,
            };
            record_count
        ]);
        let entity = commands
            .spawn((
                PlanarGaussian3dHandle(cloud_handle),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::Radix,
                    global_opacity: 2.0,
                    opacity_adaptive_radius: false,
                    lod_debug: DebugCase::ALL[0].settings(),
                    ..default()
                },
                metadata,
                GaussianLodSettings {
                    quality: DEBUG_QUALITY,
                    ..default()
                },
                Transform::default(),
                Name::new("lod_debug_test_cloud"),
            ))
            .id();
        state.cloud = Some(entity);

        commands.spawn((
            Camera3d::default(),
            Camera::default(),
            RenderTarget::Image(target.into()),
            Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)),
            Tonemapping::None,
            GaussianCamera::default(),
        ));
    }

    fn drive_capture(mut commands: Commands, mut state: ResMut<DebugRenderState>) {
        state.total_frames += 1;
        state.phase_frames += 1;
        assert!(
            state.total_frames <= MAX_FRAMES,
            "LoD debug GPU test timed out"
        );

        if state.capture_pending || state.phase_frames < WARMUP_FRAMES {
            return;
        }
        let target = state.target.clone().expect("render target exists");
        commands.spawn(Screenshot::image(target));
        state.capture_pending = true;
    }

    fn on_capture(
        trigger: On<ScreenshotCaptured>,
        mut state: ResMut<DebugRenderState>,
        mut settings: Query<&mut CloudSettings>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("screenshot converts")
            .to_rgba8();
        let sums = channel_sums(rgba.as_raw());
        assert!(sums.non_black > 32, "debug render stayed black: {sums:?}");

        let current = state.current_case();
        assert_case_pixels(current, &sums);

        state.case_index += 1;
        if state.case_index == DebugCase::ALL.len() {
            exit.write(AppExit::Success);
            return;
        }

        let entity = state.cloud.expect("cloud entity exists");
        settings
            .get_mut(entity)
            .expect("cloud settings exist")
            .lod_debug = state.current_case().settings();
        state.phase_frames = 0;
        state.capture_pending = false;
    }

    fn assert_case_pixels(case: DebugCase, sums: &ChannelSums) {
        match case {
            DebugCase::Off => assert!(
                sums.red > sums.green.saturating_mul(2) && sums.red > sums.blue.saturating_mul(2),
                "off preset did not preserve authored red: {sums:?}"
            ),
            DebugCase::Level => assert_matches_field_color(sums, lod_debug_level_color(7), "level"),
            DebugCase::Page => {
                assert_matches_field_color(sums, lod_debug_page_color(DEBUG_PAGE_COLOR_KEY), "page")
            }
            DebugCase::Residency => assert!(
                sums.red > sums.green && sums.green > sums.blue.saturating_mul(2),
                "residency did not render ancestor fallback orange: {sums:?}"
            ),
            DebugCase::Boundaries => assert!(
                sums.dominant_channel() == 1 && sums.green > sums.blue.saturating_mul(2),
                "boundaries did not render the fixed green overlay: {sums:?}"
            ),
            DebugCase::SelectionPressure => {
                let projection = PerspectiveProjection::default();
                let focal_y_px = 0.5 * HEIGHT as f32 / (0.5 * projection.fov).tan();
                let projected_error_px = 0.125 * focal_y_px / (5.0 - 1.0);
                let projected_support_radius_px = focal_y_px / (5.0 - 1.0);
                let projected_coverage =
                    (2.0 * projected_support_radius_px / HEIGHT as f32).clamp(0.0, 1.0);
                let lod = GaussianLodSettings {
                    quality: DEBUG_QUALITY,
                    ..default()
                };
                let pressure = lod.quality_target().node_pressure(
                    DEBUG_QUALITY_THRESHOLD,
                    projected_error_px,
                    projected_coverage,
                    0.0,
                    false,
                );
                let guarded_color = lod_debug_selection_pressure_color(pressure);
                let unguarded_pressure =
                    DEBUG_QUALITY * projected_coverage / DEBUG_QUALITY_THRESHOLD;
                let unguarded_color = lod_debug_selection_pressure_color(unguarded_pressure);
                assert!(
                    unguarded_color[2] > unguarded_color[1],
                    "fixture must be blue-dominant without the high-quality guard: {unguarded_color:?}"
                );
                assert!(
                    guarded_color[0] > guarded_color[1] && guarded_color[0] > guarded_color[2],
                    "fixture's large near-top pixel error must be red-dominant with the high-quality guard: {guarded_color:?}"
                );
                assert_matches_field_color(sums, guarded_color, "selection-pressure");
            }
        }
    }

    fn assert_matches_field_color(sums: &ChannelSums, expected: [f32; 3], label: &str) {
        let expected_channel = dominant_expected_channel(expected, label);
        assert_eq!(
            sums.dominant_channel(),
            expected_channel,
            "{label} annotation chose the wrong dominant channel; expected={expected:?}, observed={sums:?}"
        );
    }

    fn dominant_expected_channel(color: [f32; 3], label: &str) -> usize {
        let (index, largest) = color
            .into_iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("RGB colors are non-empty");
        let next_largest = color
            .into_iter()
            .enumerate()
            .filter_map(|(candidate, value)| (candidate != index).then_some(value))
            .max_by(f32::total_cmp)
            .expect("RGB has two remaining channels");
        assert!(
            largest - next_largest > 0.05,
            "test fixture must give {label} an unambiguous expected color: {color:?}"
        );
        index
    }

    #[derive(Debug)]
    struct ChannelSums {
        red: u64,
        green: u64,
        blue: u64,
        non_black: usize,
    }

    impl ChannelSums {
        fn dominant_channel(&self) -> usize {
            [self.red, self.green, self.blue]
                .into_iter()
                .enumerate()
                .max_by_key(|(_, value)| *value)
                .map(|(index, _)| index)
                .expect("RGB channel list is non-empty")
        }
    }

    fn channel_sums(bytes: &[u8]) -> ChannelSums {
        let mut sums = ChannelSums {
            red: 0,
            green: 0,
            blue: 0,
            non_black: 0,
        };
        for pixel in bytes.chunks_exact(4) {
            sums.red += u64::from(pixel[0]);
            sums.green += u64::from(pixel[1]);
            sums.blue += u64::from(pixel[2]);
            sums.non_black += usize::from(pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0);
        }
        sums
    }

    fn debug_test_cloud() -> PlanarGaussian3d {
        let mut red = SphericalHarmonicCoefficients::default();
        red.set(0, 6.0);
        let rotation = Rotation {
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        let mut gaussians = Vec::new();
        for x in [-0.35, 0.35] {
            for y in [-0.35, 0.35] {
                gaussians.push(Gaussian3d {
                    position_visibility: [x, y, 0.0, 1.0].into(),
                    rotation,
                    scale_opacity: [0.3, 0.3, 0.3, 0.9].into(),
                    spherical_harmonic: red,
                });
            }
        }
        gaussians.into()
    }
}
