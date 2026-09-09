//! Bounded production-raster parity: anisotropic OBB coverage and finite elliptical density.
//!
//! Opt in with RUN_GPU_RENDER_TESTS=1. This is a numerical render contract on
//! tiny authored scenes, not a LoD quality, throughput, or arbitrary-camera claim.

#![cfg(all(
    feature = "headless",
    feature = "testing",
    feature = "lod",
    feature = "buffer_storage",
    feature = "sort_radix",
    not(target_arch = "wasm32")
))]

use std::{
    io::Read,
    time::{Duration, Instant},
};

use bevy::{
    app::PluginsState,
    camera::{Hdr, RenderTarget, Viewport},
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::{
        RenderApp,
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{PipelineCache, TextureFormat},
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, GaussianCamera, GaussianMode, GaussianSplattingPlugin,
    PlanarGaussian3d, PlanarGaussian3dHandle, RadixSortDepthBits, SphericalHarmonicCoefficients,
    gaussian::{
        covariance::compute_covariance_3d, f32::Rotation, formats::planar_3d_chunked::LodNodeId,
        settings::GaussianColorSpace,
    },
    sort::SortMode,
    stream::bridge::GaussianLodBridgeConfig,
    testing::{LodProjection, LodTestCamera, render_production_lod_linear_gaussians},
};
use half::f16;
use sha2::{Digest, Sha256};

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;
const DEADLINE: Duration = Duration::from_secs(45);
const GAUSSIAN_SHADER_HANDLE: Handle<Shader> =
    bevy::asset::uuid_handle!("9a18d83b-137d-4f44-9628-e2defc4b62b0");

struct FrozenGaussianShaderOverride {
    path: String,
    source: String,
}

impl FrozenGaussianShaderOverride {
    fn from_env() -> Option<Self> {
        let path = std::env::var_os("BGS_ORACLE_GAUSSIAN_WGSL")?;
        let path = std::fs::canonicalize(path).expect("Gaussian shader override must exist");
        const MAX_BYTES: u64 = 1024 * 1024;
        let mut bytes = Vec::new();
        std::fs::File::open(&path)
            .expect("Gaussian shader override must be readable")
            .take(MAX_BYTES + 1)
            .read_to_end(&mut bytes)
            .expect("Gaussian shader override read must succeed");
        assert!(
            !bytes.is_empty() && bytes.len() as u64 <= MAX_BYTES,
            "Gaussian shader override must contain 1..=1 MiB of UTF-8 WGSL"
        );
        Some(Self {
            path: path.to_string_lossy().into_owned(),
            source: String::from_utf8(bytes).expect("Gaussian shader override must be UTF-8"),
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum TargetPrecision {
    Float16,
    Srgb8,
}

impl TargetPrecision {
    fn format(self) -> TextureFormat {
        match self {
            Self::Float16 => TextureFormat::Rgba16Float,
            Self::Srgb8 => TextureFormat::Rgba8UnormSrgb,
        }
    }

    fn quantize(self, value: f32, channel: usize) -> f32 {
        match self {
            Self::Float16 => f16::from_f32(value).to_f32(),
            Self::Srgb8 => {
                let encoded = if channel < 3 {
                    linear_to_srgb(value)
                } else {
                    value
                };
                let stored = (encoded.clamp(0.0, 1.0) * 255.0).round() / 255.0;
                if channel < 3 {
                    srgb_to_linear(stored)
                } else {
                    stored
                }
            }
        }
    }

    // One target-storage step, in linear units. The f16 floor covers the
    // subnormal range; sRGB uses the larger adjacent decoded code distance.
    fn step(self, value: f32, channel: usize) -> f32 {
        match self {
            Self::Float16 => {
                let bits = f16::from_f32(value.abs()).to_bits();
                (f16::from_bits(bits + 1).to_f32() - f16::from_bits(bits).to_f32())
                    .max(2.0_f32.powi(-24))
            }
            Self::Srgb8 if channel == 3 => 1.0 / 255.0,
            Self::Srgb8 => {
                let code = (linear_to_srgb(value).clamp(0.0, 1.0) * 255.0).round();
                let center = srgb_to_linear(code / 255.0);
                let below = srgb_to_linear((code - 1.0).max(0.0) / 255.0);
                let above = srgb_to_linear((code + 1.0).min(255.0) / 255.0);
                (center - below).max(above - center)
            }
        }
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn camera() -> LodTestCamera {
    LodTestCamera {
        position: Vec3::new(0.0, 0.0, 5.0),
        target: Vec3::ZERO,
        up: Vec3::Y,
        world_rotation: None,
        projection: LodProjection::Perspective {
            vertical_fov_radians: std::f32::consts::FRAC_PI_2,
        },
        near: 0.1,
        far: 100.0,
        viewport: [WIDTH, HEIGHT],
    }
}

fn gaussian(
    position: [f32; 3],
    scale: [f32; 3],
    angle: f32,
    opacity: f32,
    display_rgb: [f32; 3],
) -> Gaussian3d {
    let mut spherical_harmonic = SphericalHarmonicCoefficients::default();
    spherical_harmonic.coefficients.fill(0.0);
    for (channel, color) in display_rgb.into_iter().enumerate() {
        spherical_harmonic.coefficients[channel] = (color - 0.5) / 0.282_094_8;
    }
    Gaussian3d {
        position_visibility: [position[0], position[1], position[2], 1.0].into(),
        rotation: Rotation {
            rotation: [(angle * 0.5).cos(), 0.0, 0.0, (angle * 0.5).sin()],
        },
        scale_opacity: [scale[0], scale[1], scale[2], opacity].into(),
        spherical_harmonic,
    }
}

fn diagonal_support_witness() -> Gaussian3d {
    // At depth 5, the shader's doubled-pixel Jacobian is 128 / 5.
    // Filtered screen-down covariance is [5, 3, 5]: raw eigenvalues 6.8,
    // 0.8, followed by +1.2 variance. A -45 degree world rotation produces
    // the positive screen-down off-diagonal through the viewport Y reflection.
    let jacobian = 128.0 / 5.0;
    gaussian(
        [0.0; 3],
        [6.8_f32.sqrt() / jacobian, 0.8_f32.sqrt() / jacobian, 0.0001],
        -std::f32::consts::FRAC_PI_4,
        1.0,
        [0.8, 0.4, 0.2],
    )
}

fn raster_diagonal_support_witness() -> Gaussian3d {
    let mut gaussian = diagonal_support_witness();
    // The centered analytic rectangle has pixel centers exactly on an edge.
    // CPU inclusive support cannot predict the GPU triangle top-left rule at
    // those samples. Move its center to (64 + 1/8, 64 - 1/16) physical pixels,
    // keeping every diagonal edge well clear of raster subpixel uncertainty.
    // The independent centered covariance/support witness below is retained.
    gaussian.position_visibility.position = [0.009_765_625, 0.004_882_812_5, 0.0];
    gaussian
}

fn mixed_scene() -> Vec<Gaussian3d> {
    let mut opposite = raster_diagonal_support_witness();
    opposite.rotation.rotation[3] = -opposite.rotation.rotation[3];
    opposite.scale_opacity.opacity = 0.7;
    opposite.spherical_harmonic =
        gaussian([0.0; 3], [0.1; 3], 0.0, 1.0, [0.15, 0.8, 0.35]).spherical_harmonic;
    vec![
        // Exactly equal camera-depth keys, distinct colors, and opposing axes test
        // stable radix order as well as alpha-over composition.
        raster_diagonal_support_witness(),
        opposite,
        gaussian([0.0; 3], [0.16, 0.16, 0.0001], 0.0, 0.35, [0.2, 0.3, 0.9]),
        // Unit peak clamp before the fragment's .999 cap. Opacity deliberately
        // exceeds one, and the center lies exactly on a pixel center.
        gaussian(
            [1.523_437_5, -0.976_562_5, 0.0],
            [0.15; 3],
            0.0,
            2.0,
            [0.7, 0.5, 0.3],
        ),
        // A partially clipped viewport footprint, whose center is in-frustum.
        gaussian(
            [-4.85, 1.0, 0.0],
            [0.22, 0.08, 0.0001],
            0.38,
            0.6,
            [0.4, 0.7, 0.9],
        ),
    ]
}

fn oracle(gaussians: &[Gaussian3d]) -> Vec<[f32; 4]> {
    oracle_in_viewport(gaussians, [0, 0, WIDTH, HEIGHT])
}

fn oracle_in_viewport(gaussians: &[Gaussian3d], viewport: [u32; 4]) -> Vec<[f32; 4]> {
    let [origin_x, origin_y, width, height] = viewport;
    assert!(width > 0 && height > 0 && origin_x + width <= WIDTH && origin_y + height <= HEIGHT);
    let owned = gaussians
        .iter()
        .enumerate()
        .map(|(index, gaussian)| (LodNodeId(index as u64 + 1), *gaussian))
        .collect::<Vec<_>>();
    let mut local_camera = camera();
    local_camera.viewport = [width, height];
    let local = render_production_lod_linear_gaussians(
        &owned,
        local_camera,
        width,
        height,
        GaussianColorSpace::SrgbRec709Display,
    )
    .expect("authored scene is in the CPU oracle's supported domain");
    if viewport == [0, 0, WIDTH, HEIGHT] {
        return local;
    }
    let mut target = vec![[0.0; 4]; (WIDTH * HEIGHT) as usize];
    for y in 0..height {
        let source_start = (y * width) as usize;
        let target_start = ((origin_y + y) * WIDTH + origin_x) as usize;
        target[target_start..target_start + width as usize]
            .copy_from_slice(&local[source_start..source_start + width as usize]);
    }
    target
}

struct ExpectedImage {
    viewport: [u32; 4],
    rgba: Vec<[f32; 4]>,
    // Propagated per-pixel bound: one storage step plus 3e-6 arithmetic error
    // for each actual contributing blend. It is deliberately independent of
    // a measured image metric and does not dilute support errors in black area.
    error_bound: Vec<[f32; 4]>,
}

fn quantized_oracle(
    gaussians: &[Gaussian3d],
    precision: TargetPrecision,
    additive: bool,
) -> ExpectedImage {
    quantized_oracle_in_viewport(gaussians, precision, additive, [0, 0, WIDTH, HEIGHT])
}

fn quantized_oracle_in_viewport(
    gaussians: &[Gaussian3d],
    precision: TargetPrecision,
    additive: bool,
    viewport: [u32; 4],
) -> ExpectedImage {
    let mut ordered = gaussians.iter().copied().enumerate().collect::<Vec<_>>();
    ordered.sort_by(|(left_index, left), (right_index, right)| {
        let left_key = (Vec3::from_array(left.position_visibility.position) - camera().position)
            .length_squared();
        let right_key = (Vec3::from_array(right.position_visibility.position) - camera().position)
            .length_squared();
        right_key
            .total_cmp(&left_key)
            .then(left_index.cmp(right_index))
    });
    let mut result = ExpectedImage {
        viewport,
        rgba: vec![[0.0; 4]; (WIDTH * HEIGHT) as usize],
        error_bound: vec![[0.0; 4]; (WIDTH * HEIGHT) as usize],
    };
    for (_, gaussian) in ordered {
        let source = oracle_in_viewport(&[gaussian], viewport);
        for ((pixel, error), source) in result
            .rgba
            .iter_mut()
            .zip(&mut result.error_bound)
            .zip(source)
        {
            if source[3] == 0.0 {
                continue;
            }
            let destination_factor = if additive { 1.0 } else { 1.0 - source[3] };
            for channel in 0..4 {
                let blended = source[channel] + pixel[channel] * destination_factor;
                pixel[channel] = precision.quantize(blended, channel);
                error[channel] = error[channel] * destination_factor
                    + precision.step(pixel[channel], channel)
                    + 3.0e-6;
            }
        }
    }
    result
}

#[derive(Resource, Default)]
struct CapturedImage(Option<Image>);

fn on_capture(trigger: On<ScreenshotCaptured>, mut captured: ResMut<CapturedImage>) {
    assert!(
        captured.0.is_none(),
        "only one screenshot may be outstanding"
    );
    captured.0 = Some(trigger.image.clone());
}

struct Harness {
    app: App,
    cloud: Entity,
    target: Handle<Image>,
    precision: TargetPrecision,
    shader_identity: serde_json::Value,
}

impl Harness {
    fn new(
        precision: TargetPrecision,
        gaussians: Vec<Gaussian3d>,
        shader_override: Option<&FrozenGaussianShaderOverride>,
        viewport: Option<Viewport>,
    ) -> Self {
        let mut app = App::new();
        app.insert_resource(ClearColor(Color::NONE))
            .insert_resource(GaussianLodBridgeConfig {
                auto_build_flat_clouds: false,
                ..default()
            })
            .init_resource::<CapturedImage>();
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: "assets".into(),
                    processed_file_path: "assets".into(),
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
        app.add_plugins(GaussianSplattingPlugin);
        app.add_observer(on_capture);
        let started = Instant::now();
        while app.plugins_state() == PluginsState::Adding {
            assert!(
                started.elapsed() < DEADLINE,
                "renderer plugin startup timed out"
            );
            std::thread::yield_now();
        }
        app.finish();
        app.cleanup();

        let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
        if let Some(shader_override) = shader_override {
            assert!(
                shaders.contains(GAUSSIAN_SHADER_HANDLE.id()),
                "diagnostic handle must replace the registered production Gaussian shader"
            );
            shaders
                .insert(
                    GAUSSIAN_SHADER_HANDLE.id(),
                    Shader::from_wgsl(shader_override.source.clone(), shader_override.path.clone()),
                )
                .expect("registered Gaussian shader override must be replaceable");
        }
        let installed_shader = shaders
            .get(&GAUSSIAN_SHADER_HANDLE)
            .expect("production Gaussian shader is registered");
        // Hash the actual installed asset, including when manually linked
        // against an older library. Never hash the current source checkout and
        // present it as the identity of an already compiled embedded shader.
        let shader_identity = serde_json::json!({
            "variant": if shader_override.is_some() { "diagnostic_override" } else { "production_embedded" },
            "gaussian_wgsl_sha256": format!("{:x}", Sha256::digest(installed_shader.source.as_str().as_bytes())),
            "gaussian_wgsl_path": installed_shader.path,
            "override_environment": shader_override.map(|_| "BGS_ORACLE_GAUSSIAN_WGSL"),
            "scope": "Gaussian entry shader only; imported modules retain compiled-library identity",
        });
        let target =
            app.world_mut()
                .resource_mut::<Assets<Image>>()
                .add(Image::new_target_texture(
                    WIDTH,
                    HEIGHT,
                    precision.format(),
                    None,
                ));
        let asset = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(gaussians));
        let cloud = app
            .world_mut()
            .spawn((
                PlanarGaussian3dHandle(asset),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::Radix,
                    radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                    opacity_adaptive_radius: false,
                    aabb: false,
                    color_space: GaussianColorSpace::SrgbRec709Display,
                    ..default()
                },
                Transform::IDENTITY,
                Visibility::Visible,
            ))
            .id();
        let mut camera_entity = app.world_mut().spawn((
            Camera3d::default(),
            Camera {
                viewport,
                ..default()
            },
            Projection::Perspective(PerspectiveProjection {
                fov: std::f32::consts::FRAC_PI_2,
                near: 0.1,
                far: 100.0,
                ..default()
            }),
            RenderTarget::Image(target.clone().into()),
            Transform::from_translation(camera().position).looking_at(Vec3::ZERO, Vec3::Y),
            Tonemapping::None,
            Msaa::Off,
            GaussianCamera::default(),
        ));
        if matches!(precision, TargetPrecision::Float16) {
            camera_entity.insert(Hdr);
        }
        Self {
            app,
            cloud,
            target,
            precision,
            shader_identity,
        }
    }

    fn set_scene(&mut self, gaussians: Vec<Gaussian3d>) {
        let asset = self
            .app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(gaussians));
        self.app
            .world_mut()
            .entity_mut(self.cloud)
            .insert(PlanarGaussian3dHandle(asset));
    }

    fn set_additive(&mut self, additive: bool) {
        self.app
            .world_mut()
            .get_mut::<CloudSettings>(self.cloud)
            .expect("fixture cloud settings")
            .additive = additive;
    }

    fn capture(&mut self) -> Vec<[f32; 4]> {
        // Asset replacement, pipeline specialization, extraction, and radix all
        // run through production schedules. An enabled test cannot silently
        // succeed if a pipeline never compiles or the renderer stays blank.
        let started = Instant::now();
        let mut frames = 0;
        let mut ready_frames = 0;
        while frames < 32 || ready_frames < 8 {
            assert!(
                started.elapsed() < DEADLINE,
                "pipeline/asset warmup timed out"
            );
            self.app.update();
            frames += 1;
            let cache = self
                .app
                .sub_app(RenderApp)
                .world()
                .resource::<PipelineCache>();
            if cache.waiting_pipelines().next().is_none() {
                ready_frames += 1;
            } else {
                ready_frames = 0;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        self.app
            .world_mut()
            .spawn(Screenshot::image(self.target.clone()));
        let image = loop {
            assert!(
                started.elapsed() < DEADLINE,
                "screenshot readback timed out"
            );
            self.app.update();
            if let Some(image) = self
                .app
                .world_mut()
                .resource_mut::<CapturedImage>()
                .0
                .take()
            {
                break image;
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(image.texture_descriptor.format, self.precision.format());
        assert_eq!(image.width(), WIDTH);
        assert_eq!(image.height(), HEIGHT);
        let bytes = image
            .data
            .expect("screenshot contains mapped target pixels");
        match self.precision {
            TargetPrecision::Float16 => {
                assert_eq!(bytes.len(), (WIDTH * HEIGHT * 8) as usize);
                bytes
                    .chunks_exact(8)
                    .map(|pixel| {
                        std::array::from_fn(|channel| {
                            f16::from_bits(u16::from_le_bytes([
                                pixel[2 * channel],
                                pixel[2 * channel + 1],
                            ]))
                            .to_f32()
                        })
                    })
                    .collect()
            }
            TargetPrecision::Srgb8 => {
                assert_eq!(bytes.len(), (WIDTH * HEIGHT * 4) as usize);
                bytes
                    .chunks_exact(4)
                    .map(|pixel| {
                        std::array::from_fn(|channel| {
                            let value = pixel[channel] as f32 / 255.0;
                            if channel < 3 {
                                srgb_to_linear(value)
                            } else {
                                value
                            }
                        })
                    })
                    .collect()
            }
        }
    }
}

fn pixel(image: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    image[(y * WIDTH + x) as usize]
}

fn report_parity(
    actual: &[[f32; 4]],
    expected: &ExpectedImage,
    precision: TargetPrecision,
    label: &str,
    shader_identity: &serde_json::Value,
) -> bool {
    assert_eq!(actual.len(), expected.rgba.len());
    let mut max_error = [0.0_f32; 4];
    let mut squared_error = [0.0_f64; 4];
    let mut foreground = 0_usize;
    let mut maximum_bound_fraction = 0.0_f32;
    let mut violations = Vec::new();
    let mut below_reference = [0_usize; 4];
    let mut above_reference = [0_usize; 4];
    for (index, ((actual, expected), bound)) in actual
        .iter()
        .zip(&expected.rgba)
        .zip(&expected.error_bound)
        .enumerate()
    {
        if expected[3] != 0.0 || actual[3] != 0.0 {
            foreground += 1;
        }
        for channel in 0..4 {
            assert!(
                actual[channel].is_finite(),
                "{label}: nonfinite GPU pixel {index}"
            );
            let error = (actual[channel] - expected[channel]).abs();
            below_reference[channel] += usize::from(actual[channel] < expected[channel]);
            above_reference[channel] += usize::from(actual[channel] > expected[channel]);
            max_error[channel] = max_error[channel].max(error);
            squared_error[channel] += f64::from(error).powi(2);
            maximum_bound_fraction = maximum_bound_fraction.max(error / bound[channel].max(1e-20));
            if error > bound[channel] {
                violations.push((
                    error / bound[channel].max(1e-20),
                    serde_json::json!({
                        "pixel": [index % WIDTH as usize, index / WIDTH as usize],
                        "channel": channel,
                        "gpu": actual[channel], "quantized_cpu": expected[channel],
                        "error": error, "frozen_bound": bound[channel],
                        "storage_steps": error / precision.step(expected[channel], channel),
                    }),
                ));
            }
        }
    }
    let visible = foreground >= 20 && actual.iter().any(|pixel| pixel[3] > 0.01);
    let rmse = squared_error.map(|sum| (sum / foreground as f64).sqrt());
    violations.sort_by(|left, right| right.0.total_cmp(&left.0));
    println!(
        "{}",
        serde_json::json!({
            "contract": "fixed_support_identity_cloud_production_cpu_gpu_oracle",
            "scene": label, "target": format!("{precision:?}"),
            "viewport_xywh": expected.viewport,
            "shader": shader_identity,
            "foreground_pixels": foreground, "max_linear_rgba_error": max_error,
            "foreground_linear_rgba_rmse": rmse,
            "max_frozen_quantization_bound_fraction": maximum_bound_fraction,
            "violating_channels": violations.len(),
            "gpu_channels_below_reference": below_reference,
            "gpu_channels_above_reference": above_reference,
            "nonblank_fixture": visible,
            "largest_violations": violations.iter().take(24).map(|(_, record)| record).collect::<Vec<_>>(),
            "scope": "128x128; no MSAA or tonemapping; no performance claim",
        })
    );
    violations.is_empty() && visible
}

fn record_parity(
    failures: &mut Vec<String>,
    actual: &[[f32; 4]],
    expected: &ExpectedImage,
    precision: TargetPrecision,
    label: &str,
    shader_identity: &serde_json::Value,
) {
    if !report_parity(actual, expected, precision, label, shader_identity) {
        failures.push(format!("{precision:?}/{label}"));
    }
}

fn record_contract(
    failures: &mut Vec<String>,
    passed: bool,
    precision: TargetPrecision,
    label: &str,
) {
    if !passed {
        failures.push(format!("{precision:?}/{label}"));
    }
}

fn report_inward_translation(
    clipped: &[[f32; 4]],
    inward: &[[f32; 4]],
    precision: TargetPrecision,
    shader_identity: &serde_json::Value,
) {
    let mut max_error = [0.0_f32; 4];
    let mut changed_channels = [0_usize; 4];
    for y in 0..HEIGHT {
        for x in 0..WIDTH - 32 {
            let clipped = pixel(clipped, x, y);
            let inward = pixel(inward, x + 32, y);
            for channel in 0..4 {
                max_error[channel] =
                    max_error[channel].max((clipped[channel] - inward[channel]).abs());
                changed_channels[channel] += usize::from(clipped[channel] != inward[channel]);
            }
        }
    }
    // This is a discriminator, not another acceptance tolerance. The tiny
    // nonzero Z variance introduces a correspondingly tiny off-axis covariance
    // change, so exact pixel equality is informative but is not required.
    println!(
        "{}",
        serde_json::json!({
            "contract": "diagnostic_exact_32_pixel_inward_translation",
        "target": format!("{precision:?}"),
        "shader": shader_identity,
            "max_shifted_linear_rgba_difference": max_error,
            "changed_shifted_channels": changed_channels,
            "scope": "same splat translated +2.5 world X; shared visible pixels only; no new tolerance",
        })
    );
}

#[test]
fn diagonal_fixture_has_independent_covariance_and_support_witnesses() {
    let gaussian = diagonal_support_witness();
    let covariance = compute_covariance_3d(
        Vec4::from_array(gaussian.rotation.rotation),
        Vec3::from_array(gaussian.scale_opacity.scale),
    );
    let jacobian_squared = (128.0_f32 / 5.0).powi(2);
    let projected = [
        covariance[0] * jacobian_squared + 1.2,
        -covariance[1] * jacobian_squared,
        covariance[3] * jacobian_squared + 1.2,
    ];
    for (actual, expected) in projected.into_iter().zip([5.0, 3.0, 5.0]) {
        assert!((actual - expected).abs() < 2e-6, "covariance {projected:?}");
    }
    let rendered = oracle(&[gaussian]);
    let opacity_scale = 0.34_f32.sqrt();
    let major_axis_alpha = opacity_scale * (-3.125_f32).exp();
    assert!((pixel(&rendered, 66, 66)[3] - major_axis_alpha).abs() < 1e-6);
    assert_eq!(pixel(&rendered, 66, 61), [0.0; 4]);
    // The conservative OBB covers this corner, but its q=10.25 lies outside
    // the finite density ellipse. Raster coverage is not optical support.
    let corner = Vec2::new(7.0, 3.0);
    assert!(corner.dot(Vec2::new(1.0, 1.0).normalize()).abs() < 3.0 * 8.0_f32.sqrt());
    assert!(corner.dot(Vec2::new(-1.0, 1.0).normalize()).abs() < 3.0 * 2.0_f32.sqrt());
    assert_eq!(pixel(&rendered, 67, 65), [0.0; 4]);
    // Analytically d=(3,-3) has q=9: the C1 boundary vanishes within
    // the same f32 covariance/alpha allowance as the interior witness.
    assert!(pixel(&rendered, 65, 62)[3] < 1e-6);
}

#[test]
fn raster_diagonal_edges_avoid_sample_center_tie_rules() {
    let gaussian = raster_diagonal_support_witness();
    let center = Vec2::new(
        64.0 + 12.8 * gaussian.position_visibility.position[0],
        64.0 - 12.8 * gaussian.position_visibility.position[1],
    );
    assert_eq!(center, Vec2::new(64.125, 63.9375));
    let half_extent_shader = Vec2::new(3.0 * 8.0_f32.sqrt(), 3.0 * 2.0_f32.sqrt());
    let mut minimum_edge_distance_px = f32::INFINITY;
    for direction in [-1.0, 1.0] {
        // Check both +/-45 degree splats used by the mixed scene. Distances
        // to the edge's infinite line are conservative for the finite quad.
        let major = Vec2::new(1.0, direction).normalize();
        let minor = Vec2::new(-direction, 1.0).normalize();
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let delta = 2.0 * (Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - center);
                let distance = 0.5
                    * (delta.dot(major).abs() - half_extent_shader.x)
                        .abs()
                        .min((delta.dot(minor).abs() - half_extent_shader.y).abs());
                minimum_edge_distance_px = minimum_edge_distance_px.min(distance);
            }
        }
    }
    assert!(
        minimum_edge_distance_px > 1.0 / 32.0,
        "fixture edge clearance must exceed raster subpixel uncertainty: {minimum_edge_distance_px}"
    );
    let rendered = oracle(&[gaussian]);
    assert_eq!(
        pixel(&rendered, 62, 59),
        [0.0; 4],
        "the formerly tied edge sample is now unambiguously outside"
    );
    assert_eq!(pixel(&rendered, 62, 60), [0.0; 4]);
    assert!(pixel(&rendered, 62, 61)[3] > 0.0);
    assert!(pixel(&rendered, 66, 66)[3] > 0.015);
    assert_eq!(pixel(&rendered, 66, 61), [0.0; 4]);
    let corner_delta = 2.0 * (Vec2::new(67.5, 65.5) - center);
    let corner_mahalanobis = (5.0 * corner_delta.x.powi(2) - 6.0 * corner_delta.x * corner_delta.y
        + 5.0 * corner_delta.y.powi(2))
        / 16.0;
    assert!(
        corner_mahalanobis > 9.0,
        "the translated OBB corner must remain outside finite elliptical density"
    );
    assert_eq!(pixel(&rendered, 67, 65), [0.0; 4]);
    // Independently evaluate the new tail, without calling the production
    // density helper: q=8.8486328125 at this translated pixel center.
    let tail_delta = 2.0 * (Vec2::new(66.5, 63.5) - center);
    let tail_q = (5.0 * tail_delta.x.powi(2) - 6.0 * tail_delta.x * tail_delta.y
        + 5.0 * tail_delta.y.powi(2))
        / 16.0;
    assert!((8.0..9.0).contains(&tail_q));
    let u = tail_q - 8.0;
    let untapered_alpha = 0.34_f32.sqrt() * (-0.5 * tail_q).exp();
    let tapered_alpha = untapered_alpha * (1.0 - u).powi(2) * (1.0 + 2.0 * u);
    assert!(tapered_alpha > 0.0 && tapered_alpha < 0.1 * untapered_alpha);
    assert!((pixel(&rendered, 66, 63)[3] - tapered_alpha).abs() < 1e-6);
}

#[test]
fn production_gpu_matches_cpu_oracle_support_color_alpha_and_additive_toggle() {
    if std::env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping GPU oracle parity; set RUN_GPU_RENDER_TESTS=1 to enable");
        return;
    }
    // Read and freeze a diagnostic source once for both target formats. No
    // production asset or shader file is changed by this opt-in A/B hook.
    let shader_override = FrozenGaussianShaderOverride::from_env();
    let mut failures = Vec::new();
    for precision in [TargetPrecision::Float16, TargetPrecision::Srgb8] {
        let witness = vec![raster_diagonal_support_witness()];
        let mut harness = Harness::new(precision, witness.clone(), shader_override.as_ref(), None);
        let shader_identity = harness.shader_identity.clone();
        let actual = harness.capture();
        record_parity(
            &mut failures,
            &actual,
            &quantized_oracle(&witness, precision, false),
            precision,
            "diagonal_support",
            &shader_identity,
        );
        record_contract(
            &mut failures,
            pixel(&actual, 66, 66)[3] > 0.015,
            precision,
            "rotated major-axis support was lost",
        );
        record_contract(
            &mut failures,
            pixel(&actual, 66, 61) == [0.0; 4],
            precision,
            "minor-axis outside support leaked",
        );
        record_contract(
            &mut failures,
            pixel(&actual, 67, 65) == [0.0; 4],
            precision,
            "conservative OBB corner leaked density beyond q=9",
        );
        record_contract(
            &mut failures,
            pixel(&actual, 66, 63)[3] < 0.5 * 0.34_f32.sqrt() * (-0.5 * 8.848_633_f32).exp()
                && (matches!(precision, TargetPrecision::Srgb8) || pixel(&actual, 66, 63)[3] > 0.0),
            precision,
            "finite tail must taper inside q=9, with nonzero Float16 density",
        );

        let mixed = mixed_scene();
        let mut isolated_clipped = None;
        for (index, gaussian) in mixed.iter().copied().enumerate() {
            harness.set_scene(vec![gaussian]);
            let isolated = harness.capture();
            record_parity(
                &mut failures,
                &isolated,
                &quantized_oracle(&[gaussian], precision, false),
                precision,
                &format!("isolated_gaussian_{index}"),
                &shader_identity,
            );
            if index == mixed.len() - 1 {
                isolated_clipped = Some(isolated);
            }
        }
        let mut translated = *mixed.last().expect("the final source is the clipped splat");
        translated.position_visibility.position[0] += 2.5;
        harness.set_scene(vec![translated]);
        let translated_image = harness.capture();
        record_parity(
            &mut failures,
            &translated_image,
            &quantized_oracle(&[translated], precision, false),
            precision,
            "isolated_clipped_splat_translated_inward_32_pixels",
            &shader_identity,
        );
        report_inward_translation(
            &isolated_clipped.expect("clipped splat was captured"),
            &translated_image,
            precision,
            &shader_identity,
        );

        harness.set_scene(mixed.clone());
        let alpha_over = harness.capture();
        record_parity(
            &mut failures,
            &alpha_over,
            &quantized_oracle(&mixed, precision, false),
            precision,
            "mixed_alpha_over",
            &shader_identity,
        );
        record_contract(
            &mut failures,
            pixel(&alpha_over, 83, 76)[3] > 0.995,
            precision,
            "unit vertex peak or final fragment cap diverged",
        );

        // Change only the setting on the existing entity: specialization must
        // switch the fixed-function blend state and then restore alpha-over.
        harness.set_additive(true);
        let additive = harness.capture();
        record_parity(
            &mut failures,
            &additive,
            &quantized_oracle(&mixed, precision, true),
            precision,
            "mixed_additive",
            &shader_identity,
        );
        record_contract(
            &mut failures,
            pixel(&additive, 64, 64)[3] > pixel(&alpha_over, 64, 64)[3] + 0.05,
            precision,
            "additive setting did not specialize the live pipeline",
        );
        harness.set_additive(false);
        let restored = harness.capture();
        record_parity(
            &mut failures,
            &restored,
            &quantized_oracle(&mixed, precision, false),
            precision,
            "restored_alpha_over",
            &shader_identity,
        );
        record_contract(
            &mut failures,
            restored == alpha_over,
            precision,
            "stationary alpha-over output did not restore exactly",
        );
        drop(harness);

        // A fresh target isolates viewport-origin behavior from any pixels
        // left behind by an earlier camera viewport. The non-square physical
        // viewport also exercises projection aspect/focal scaling explicitly.
        let viewport = [13, 7, 96, 104];
        let mut offset_harness = Harness::new(
            precision,
            mixed.clone(),
            shader_override.as_ref(),
            Some(Viewport {
                physical_position: UVec2::new(viewport[0], viewport[1]),
                physical_size: UVec2::new(viewport[2], viewport[3]),
                ..default()
            }),
        );
        let offset_actual = offset_harness.capture();
        record_parity(
            &mut failures,
            &offset_actual,
            &quantized_oracle_in_viewport(&mixed, precision, false, viewport),
            precision,
            "mixed_nonzero_viewport_origin_and_nonsquare_aspect",
            &offset_harness.shader_identity,
        );
        let outside_is_transparent = (0..HEIGHT).all(|y| {
            (0..WIDTH).all(|x| {
                let inside = x >= viewport[0]
                    && x < viewport[0] + viewport[2]
                    && y >= viewport[1]
                    && y < viewport[1] + viewport[3];
                inside || pixel(&offset_actual, x, y) == [0.0; 4]
            })
        });
        record_contract(
            &mut failures,
            outside_is_transparent,
            precision,
            "pixels outside the offset viewport must remain exactly transparent",
        );
    }
    assert!(
        failures.is_empty(),
        "unchanged oracle contracts failed after all discriminator cases ran: {failures:?}"
    );
}
