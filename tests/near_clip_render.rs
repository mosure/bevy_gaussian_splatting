//! One bounded app checks near-plane approach/return in all three renderers.
#![cfg(all(
    feature = "headless",
    feature = "testing",
    lod_render_path,
    not(target_arch = "wasm32")
))]

use std::time::{Duration, Instant};

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::{RenderTarget, ScalingMode, visibility::NoFrustumCulling},
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
    PlanarGaussian3d, PlanarGaussian3dHandle,
    gaussian::{f32::Rotation, settings::GaussianColorSpace},
    render::{
        ordered::{GaussianGlobalOrderDiagnostics, GaussianGlobalOrderSettings},
        point::{
            GaussianPointSplattingAvailability, GaussianPointSplattingDiagnostics,
            GaussianPointSplattingSettings,
        },
    },
    sort::{SortConfig, SortMode},
};

const SIZE: u32 = 48;
const NEAR: f32 = 0.1;
const SIGMA: f32 = 0.04;
// Start with a visible control, approach/cross the center plane, then retrace.
const DEPTHS: [f32; 6] = [
    NEAR + 3.0 * SIGMA + 0.001,
    NEAR + 1.5 * SIGMA,
    NEAR + 0.00001,
    NEAR - 0.00001,
    NEAR + 0.00001,
    NEAR + 3.0 * SIGMA + 0.001,
];

#[test]
fn near_plane_approach_fades_and_return_restores_each_renderer() {
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

#[derive(Component)]
struct TestCamera {
    backend: usize,
    target: Handle<Image>,
}

#[derive(Component)]
struct Capture {
    phase: usize,
    backend: usize,
}

#[derive(Resource)]
struct State {
    started: Instant,
    phase: usize,
    frames: u32,
    pending: usize,
    images: [Option<Vec<u8>>; 3],
    baseline: [Vec<u8>; 3],
    near: [Vec<u8>; 3],
    submissions: [u64; 2],
}

impl Default for State {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            phase: 0,
            frames: 0,
            pending: 0,
            images: default(),
            baseline: default(),
            near: default(),
            submissions: [0; 2],
        }
    }
}

fn projection(orthographic: bool) -> Projection {
    if orthographic {
        Projection::Orthographic(OrthographicProjection {
            near: NEAR,
            far: 10.0,
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 0.3,
            },
            ..OrthographicProjection::default_3d()
        })
    } else {
        Projection::Perspective(PerspectiveProjection {
            near: NEAR,
            far: 10.0,
            fov: 1.2,
            ..default()
        })
    }
}

fn setup(
    mut commands: Commands,
    mut clouds: ResMut<Assets<PlanarGaussian3d>>,
    mut images: ResMut<Assets<Image>>,
) {
    let mut gaussian = Gaussian3d {
        position_visibility: [0.0, 0.0, 0.0, 1.0].into(),
        rotation: Rotation {
            rotation: [1.0, 0.0, 0.0, 0.0],
        },
        scale_opacity: [SIGMA, SIGMA, SIGMA, 0.8].into(),
        spherical_harmonic: default(),
    };
    for channel in 0..3 {
        gaussian.spherical_harmonic.set(channel, 0.5 / 0.282_094_8);
    }
    commands.spawn((
        PlanarGaussian3dHandle(clouds.add(PlanarGaussian3d::from(vec![gaussian]))),
        CloudSettings {
            sort_mode: SortMode::Radix,
            color_space: GaussianColorSpace::LinRec709Display,
            opacity_adaptive_radius: false,
            ..default()
        },
        Transform::IDENTITY,
        NoFrustumCulling,
    ));
    for backend in 0..3 {
        let target = images.add(Image::new_target_texture(
            SIZE,
            SIZE,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let mut camera = commands.spawn((
            TestCamera {
                backend,
                target: target.clone(),
            },
            Camera3d::default(),
            GaussianCamera::default(),
            Camera {
                order: backend as isize,
                ..default()
            },
            RenderTarget::Image(target.into()),
            Msaa::Off,
            Tonemapping::None,
            projection(false),
            Transform::from_xyz(0.0, 0.0, DEPTHS[0]),
        ));
        if backend == 1 {
            camera.insert(GaussianGlobalOrderSettings {
                max_projected_gaussians: 1,
                max_gpu_bytes: 4 * 1024 * 1024,
            });
        } else if backend == 2 {
            camera.insert(GaussianPointSplattingSettings {
                samples_per_pixel: 8,
                min_samples_per_pixel: 8,
                max_projected_gaussians: 1,
                max_points_per_frame: 262_144,
                max_gpu_bytes: 4 * 1024 * 1024,
                seed: 0x4750_5331,
                temporal_sampling: false,
                target_gpu_ms: None,
            });
        }
    }
}

fn fail_shader_errors(cache: Res<PipelineCache>) {
    for pipeline in cache.pipelines() {
        if let CachedPipelineState::Err(error) = &pipeline.state {
            match error {
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable => {}
                _ => panic!("near-clip shader failed: {error}"),
            }
        }
    }
}

// Unit white radiance over black makes linear RGB a premultiplied-alpha probe.
fn center_alpha(bytes: &[u8]) -> f32 {
    let mut sum = 0.0;
    for y in SIZE / 2 - 4..SIZE / 2 + 4 {
        for x in SIZE / 2 - 4..SIZE / 2 + 4 {
            let srgb = f32::from(bytes[((y * SIZE + x) * 4) as usize]) / 255.0;
            sum += if srgb <= 0.04045 {
                srgb / 12.92
            } else {
                ((srgb + 0.055) / 1.055).powf(2.4)
            };
        }
    }
    sum / 64.0
}

fn capture(
    mut commands: Commands,
    mut state: ResMut<State>,
    mut cameras: Query<(Entity, &TestCamera, &mut Transform, &mut Projection)>,
    ordered: Res<GaussianGlobalOrderDiagnostics>,
    points: Res<GaussianPointSplattingDiagnostics>,
    mut exit: MessageWriter<AppExit>,
) {
    assert!(
        state.started.elapsed() < Duration::from_secs(45),
        "near-clip phase {} timed out",
        state.phase
    );
    if state.pending != 0 {
        return;
    }
    if state.images.iter().all(Option::is_some) {
        let images = std::mem::take(&mut state.images).map(Option::unwrap);
        let step = state.phase % DEPTHS.len();
        // Only initial visible controls may retry shader warmup. A black near
        // image is never accepted before every backend has rendered that control.
        if step == 0 && images.iter().any(|bytes| center_alpha(bytes) <= 0.4) {
            state.frames = 0;
            return;
        }
        let difference = images[0]
            .iter()
            .zip(&images[1])
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(
            difference <= 2,
            "flat/ordered near profile differs by {difference} at phase {}",
            state.phase
        );
        for (backend, bytes) in images.iter().enumerate() {
            match step {
                0 => state.baseline[backend] = bytes.clone(),
                1 => {
                    let ratio = center_alpha(bytes) / center_alpha(&state.baseline[backend]);
                    assert!(
                        (0.3..0.75).contains(&ratio),
                        "backend {backend} half-clearance alpha ratio {ratio}"
                    );
                }
                2..=4 => {
                    let peak = bytes
                        .chunks_exact(4)
                        .flat_map(|pixel| pixel[..3].iter())
                        .copied()
                        .max()
                        .unwrap();
                    assert!(
                        peak <= 2,
                        "backend {backend} abruptly appeared at phase {}: peak {peak}",
                        state.phase
                    );
                    if step == 2 {
                        state.near[backend] = bytes.clone();
                    }
                    if step == 4 {
                        assert_eq!(
                            bytes, &state.near[backend],
                            "near return differs, backend {backend}"
                        );
                    }
                }
                5 => assert_eq!(
                    bytes, &state.baseline[backend],
                    "visible return differs, backend {backend}"
                ),
                _ => unreachable!(),
            }
        }
        state.phase += 1;
        if state.phase == DEPTHS.len() * 2 {
            exit.write(AppExit::Success);
            return;
        }
        for (_, _, mut transform, mut camera_projection) in &mut cameras {
            transform.translation.z = DEPTHS[state.phase % DEPTHS.len()];
            if state.phase == DEPTHS.len() {
                *camera_projection = projection(true);
            }
        }
        state.frames = 0;
        return;
    }
    state.frames += 1;
    if state.frames < 8 {
        return;
    }
    let mut submissions = [0; 2];
    for (entity, camera, _, _) in &cameras {
        if camera.backend == 1 {
            let Some(frame) = ordered.get(entity) else {
                return;
            };
            if !frame.ready || frame.submission <= state.submissions[0] {
                return;
            }
            assert!(
                !frame.overflow && !frame.traversal_failed && frame.error.is_none(),
                "ordered failed: {frame:?}"
            );
            assert_eq!(frame.source_clouds, 1);
            submissions[0] = frame.submission;
        } else if camera.backend == 2 {
            let Some(frame) = points.get(entity) else {
                return;
            };
            if frame.availability != GaussianPointSplattingAvailability::Ready
                || frame.submission <= state.submissions[1]
            {
                return;
            }
            assert!(
                !frame.overflow
                    && !frame.sampling_failed
                    && !frame.traversal_failed
                    && frame.error.is_none(),
                "GPS failed: {frame:?}"
            );
            assert_eq!(frame.samples_per_pixel, 8);
            submissions[1] = frame.submission;
        }
    }
    state.submissions = submissions;
    for (_, camera, _, _) in &cameras {
        commands.spawn((
            Screenshot::image(camera.target.clone()),
            Capture {
                phase: state.phase,
                backend: camera.backend,
            },
        ));
        state.pending += 1;
    }
}

fn captured(event: On<ScreenshotCaptured>, captures: Query<&Capture>, mut state: ResMut<State>) {
    let Ok(capture) = captures.get(event.entity) else {
        return;
    };
    assert_eq!(capture.phase, state.phase);
    let image = event.image.clone().try_into_dynamic().unwrap().to_rgba8();
    assert_eq!(image.dimensions(), (SIZE, SIZE));
    assert!(
        state.images[capture.backend]
            .replace(image.into_raw())
            .is_none()
    );
    state.pending -= 1;
}
