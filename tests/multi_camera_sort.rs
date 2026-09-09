//! One tiny application checks isolated camera ordering for every flat sort mode.
#![cfg(all(
    feature = "headless",
    feature = "testing",
    feature = "sort_radix",
    feature = "sort_std",
    feature = "sort_rayon",
    not(target_arch = "wasm32")
))]

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::{
        RenderTarget, ScalingMode,
        visibility::{NoFrustumCulling, RenderLayers},
    },
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{CachedPipelineState, PipelineCache, TextureFormat},
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    shader::ShaderCacheError,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, GaussianCamera, GaussianLodBridgeConfig, GaussianSplattingPlugin,
    PlanarGaussian3d, PlanarGaussian3dHandle, RadixSortDepthBits, SphericalHarmonicCoefficients,
    gaussian::{f32::Rotation, settings::GaussianColorSpace},
    sort::{SortConfig, SortMode, SortedEntries, SortedEntriesHandle},
};

#[test]
fn flat_multi_camera_sort_survives_capacity_retention_and_camera_removal() {
    if std::env::var("RUN_GPU_RENDER_TESTS").as_deref() != Ok("1") {
        return;
    }
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GaussianLodBridgeConfig {
            auto_build_flat_clouds: false,
            ..default()
        })
        .insert_resource(SortConfig { period_ms: 0 })
        .insert_resource(State::default())
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
        .add_systems(Render, fail_shader_errors.in_set(RenderSystems::Cleanup));
    assert_eq!(app.run(), AppExit::Success);
}

fn fail_shader_errors(cache: Res<PipelineCache>) {
    for pipeline in cache.pipelines() {
        if let CachedPipelineState::Err(error) = &pipeline.state {
            match error {
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable => {}
                _ => panic!("multi-camera shader failed: {error}"),
            }
        }
    }
}

#[derive(Component)]
struct TestCloud;

#[derive(Component)]
struct TestCamera {
    index: u8,
    target: Handle<Image>,
    red_near: bool,
}

#[derive(Component)]
struct Capture {
    index: u8,
    red_near: bool,
}

#[derive(Resource)]
struct State {
    started: Instant,
    phase: u8,
    frames: u32,
    pending: usize,
    incomplete: bool,
    images: HashMap<u8, Vec<u8>>,
    short_source: Handle<PlanarGaussian3d>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            phase: 0,
            frames: 0,
            pending: 0,
            incomplete: false,
            images: HashMap::new(),
            short_source: default(),
        }
    }
}

fn source(count: usize) -> PlanarGaussian3d {
    let mut records = vec![
        Gaussian3d {
            position_visibility: [0.0, 0.0, 0.0, 1.0].into(),
            rotation: Rotation {
                rotation: [1.0, 0.0, 0.0, 0.0]
            },
            scale_opacity: [0.2, 0.2, 0.2, 0.0].into(),
            spherical_harmonic: default(),
        };
        count
    ];
    for (index, z, color) in [(0, 0.2, [0.85, 0.05, 0.05]), (1, -0.2, [0.05, 0.05, 0.85])] {
        let mut sh = SphericalHarmonicCoefficients::default();
        for (channel, value) in color.into_iter().enumerate() {
            sh.set(channel, (value - 0.5) / 0.282_094_8);
        }
        records[index].position_visibility = [0.0, 0.0, z, 1.0].into();
        records[index].scale_opacity = [0.2, 0.2, 0.2, 0.8].into();
        records[index].spherical_harmonic = sh;
    }
    // A bright hidden record catches invalid-marker loss in reduced-precision
    // radix keys and ensures CPU sort modes honor the same visibility contract.
    records[2].position_visibility = [0.0, 0.0, 0.0, 0.0].into();
    records[2].scale_opacity = [0.2, 0.2, 0.2, 0.9].into();
    for (channel, value) in [0.05, 0.95, 0.05].into_iter().enumerate() {
        records[2]
            .spherical_harmonic
            .set(channel, (value - 0.5) / 0.282_094_8);
    }
    PlanarGaussian3d::from(records)
}

fn setup(
    mut commands: Commands,
    mut state: ResMut<State>,
    mut clouds: ResMut<Assets<PlanarGaussian3d>>,
    mut images: ResMut<Assets<Image>>,
) {
    let long_source = clouds.add(source(65));
    state.short_source = clouds.add(source(3));
    for (mode_index, mode) in [SortMode::Radix, SortMode::Std, SortMode::Rayon]
        .into_iter()
        .enumerate()
    {
        commands.spawn((
            TestCloud,
            PlanarGaussian3dHandle(long_source.clone()),
            NoFrustumCulling,
            RenderLayers::layer(mode_index),
            CloudSettings {
                sort_mode: mode,
                radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                color_space: GaussianColorSpace::LinRec709Display,
                ..default()
            },
            Transform::default(),
        ));
        for side in 0..2 {
            let target = images.add(Image::new_target_texture(
                48,
                48,
                TextureFormat::Rgba8UnormSrgb,
                None,
            ));
            commands.spawn((
                TestCamera {
                    index: (mode_index * 2 + side) as u8,
                    target: target.clone(),
                    red_near: side == 0,
                },
                Camera3d::default(),
                GaussianCamera::default(),
                Msaa::Off,
                Tonemapping::None,
                Camera {
                    order: if side == 0 { -7 } else { 19 },
                    ..default()
                },
                RenderTarget::Image(target.into()),
                RenderLayers::layer(mode_index),
                Projection::Orthographic(OrthographicProjection {
                    scaling_mode: ScalingMode::FixedVertical {
                        viewport_height: 1.2,
                    },
                    near: 0.01,
                    far: 20.0,
                    ..OrthographicProjection::default_3d()
                }),
                Transform::from_xyz(0.0, 0.0, if side == 0 { 2.0 } else { -2.0 })
                    .looking_at(Vec3::ZERO, Vec3::Y),
            ));
        }
    }
}

fn capture(
    mut commands: Commands,
    mut state: ResMut<State>,
    mut cameras: Query<(Entity, &mut TestCamera, &mut Transform)>,
    mut clouds: Query<
        (
            &mut PlanarGaussian3dHandle,
            &SortedEntriesHandle,
            &mut CloudSettings,
        ),
        With<TestCloud>,
    >,
    sorted: Res<Assets<SortedEntries>>,
    mut exit: MessageWriter<AppExit>,
) {
    assert!(
        state.started.elapsed() < Duration::from_secs(45),
        "multi-camera phase {} timed out",
        state.phase
    );
    if state.pending != 0 {
        return;
    }
    if state.incomplete {
        state.images.clear();
        state.incomplete = false;
        state.frames = 0;
    }
    if !state.images.is_empty() {
        for side in 0..2u8 {
            let Some(reference) = state.images.get(&side) else {
                continue;
            };
            for mode in 1..3u8 {
                let image = &state.images[&(mode * 2 + side)];
                assert!(
                    image
                        .iter()
                        .zip(reference)
                        .all(|(a, b)| a.abs_diff(*b) <= 1),
                    "sort modes disagree in phase {} side {side}",
                    state.phase
                );
            }
        }
        state.images.clear();
        match state.phase {
            0 => {
                for (mut handle, _, mut settings) in &mut clouds {
                    handle.0 = state.short_source.clone();
                    settings.radix_sort_depth_bits = RadixSortDepthBits::Bits24;
                }
                for (_, mut camera, mut transform) in &mut cameras {
                    camera.red_near = !camera.red_near;
                    *transform =
                        Transform::from_xyz(0.0, 0.0, if camera.red_near { 2.0 } else { -2.0 })
                            .looking_at(Vec3::ZERO, Vec3::Y);
                }
            }
            1 => {
                for (_, _, mut settings) in &mut clouds {
                    settings.radix_sort_depth_bits = RadixSortDepthBits::Bits16;
                }
                for (entity, camera, _) in &cameras {
                    if camera.index % 2 == 0 {
                        commands.entity(entity).despawn();
                    }
                }
            }
            _ => {
                exit.write(AppExit::Success);
                return;
            }
        }
        state.phase += 1;
        state.frames = 0;
    }
    state.frames += 1;
    if state.frames < 48 {
        return;
    }
    if clouds.iter().count() != 3 {
        return;
    }
    for (_, handle, _) in &clouds {
        let Some(entries) = sorted.get(handle) else {
            return;
        };
        assert_eq!(
            entries.entry_count, 65,
            "capacity must survive the 65 -> 3 handle swap"
        );
        assert_eq!(entries.camera_count, if state.phase < 2 { 6 } else { 3 });
    }
    for (_, camera, _) in &cameras {
        commands.spawn((
            Screenshot::image(camera.target.clone()),
            Capture {
                index: camera.index,
                red_near: camera.red_near,
            },
        ));
        state.pending += 1;
    }
    state.frames = 0;
}

fn captured(event: On<ScreenshotCaptured>, requests: Query<&Capture>, mut state: ResMut<State>) {
    let Ok(request) = requests.get(event.entity) else {
        return;
    };
    let data = event.image.data.as_ref().expect("captured pixels");
    let pixel = &data[(24 * 48 + 24) * 4..][..4];
    if pixel[0] <= 20 && pixel[2] <= 20 {
        // Pipeline compilation can outlive the initial warmup. Retry complete
        // capture rounds under the single application deadline.
        state.incomplete = true;
        state.pending -= 1;
        return;
    }
    assert!(
        pixel[1] < 70,
        "hidden record leaked into the image: {pixel:?}"
    );
    if request.red_near {
        assert!(pixel[0] > pixel[2] + 30, "red-near order failed: {pixel:?}");
    } else {
        assert!(
            pixel[2] > pixel[0] + 30,
            "blue-near order failed: {pixel:?}"
        );
    }
    assert!(state.images.insert(request.index, data.clone()).is_none());
    state.pending -= 1;
}
