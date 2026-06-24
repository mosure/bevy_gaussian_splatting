#![allow(dead_code, unused_imports)]

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    asset::LoadState,
    camera::primitives::Aabb,
    camera::visibility::ViewVisibility,
    camera::{Projection, RenderTarget},
    core_pipeline::tonemapping::Tonemapping,
    image::TextureFormatPixelInfo,
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSystems,
        render_asset::RenderAssets,
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode,
            PollType, TexelCopyBufferInfo, TexelCopyBufferLayout, TextureFormat, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, GaussianMode, GaussianScene, GaussianSceneHandle,
    GaussianSplattingPlugin, PlanarGaussian3d, PlanarGaussian3dHandle, PlanarGaussian4d,
    PlanarGaussian4dHandle,
    gaussian::interface::{CommonCloud, TestCloud},
    io::ply::parse_ply_3d,
    io::scene::GaussianSceneLoaded,
    random_gaussians_3d, random_gaussians_3d_seeded, random_gaussians_4d,
    random_gaussians_4d_seeded,
    sort::{SortMode, SortTrigger, SortedEntriesHandle},
    utils::GaussianSplattingViewer,
};
use bevy_interleave::prelude::Planar;
use crossbeam_channel::{Receiver, Sender};
use serde::Deserialize;
use serde_json::Value;
use std::{
    io::BufReader,
    path::Path,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

const MANIFEST_PATH: &str = "www/examples/examples.json";
const THUMB_WIDTH: u32 = 960;
const THUMB_HEIGHT: u32 = 540;

#[derive(Debug, Deserialize)]
struct ExamplesManifest {
    schema_version: u32,
    examples: Vec<ExampleEntry>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ExampleEntry {
    id: String,
    title: String,
    description: String,
    #[serde(default)]
    tags: Vec<String>,
    thumbnail: String,
    #[serde(default)]
    thumbnail_input_scene: Option<String>,
    #[serde(default)]
    thumbnail_input_cloud: Option<String>,
    #[serde(default)]
    input_scene: Option<String>,
    #[serde(default)]
    input_cloud: Option<String>,
    #[serde(default)]
    args: Value,
}

#[derive(Resource, Deref)]
struct MainWorldReceiver(Receiver<Vec<u8>>);

#[derive(Resource, Deref)]
struct RenderWorldSender(Sender<Vec<u8>>);

#[derive(Debug, Resource)]
struct CaptureController {
    frames_since_ready: u32,
    total_frames: u32,
    warmup_frames_after_ready: u32,
    max_total_frames: u32,
    started_at: Instant,
    max_elapsed: Duration,
    capture_requested: bool,
    width: u32,
    height: u32,
}

impl CaptureController {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            frames_since_ready: 0,
            total_frames: 0,
            warmup_frames_after_ready: 15,
            max_total_frames: 600,
            started_at: Instant::now(),
            max_elapsed: Duration::from_secs(90),
            capture_requested: false,
            width,
            height,
        }
    }
}

#[derive(Resource, Clone)]
struct OutputTarget {
    path: PathBuf,
}

#[derive(Resource, Clone)]
struct ThumbnailRenderConfig {
    sort_mode: SortMode,
}

#[derive(Resource, Clone, Deref)]
struct CaptureRenderTarget(Handle<Image>);

#[derive(Resource, Default)]
struct AutoFrameState {
    done: bool,
}

#[derive(Component, Debug, Default)]
struct SceneCameraApplied;

#[derive(Component, Debug, Default)]
struct SceneRenderModeApplied;

type SceneRenderModeQuery = (Entity, &'static Children);
type SceneRenderModeFilter = (With<GaussianSceneLoaded>, Without<SceneRenderModeApplied>);
type SceneReadyQuery = (
    Entity,
    &'static GaussianSceneHandle,
    &'static Children,
    Option<&'static SceneCameraApplied>,
    Option<&'static SceneRenderModeApplied>,
);
type SceneReadyFilter = With<GaussianSceneLoaded>;

#[test]
fn render_example_thumbnails() {
    if std::env::var("RENDER_EXAMPLE_THUMBNAILS").ok().as_deref() != Some("1") {
        return;
    }

    let manifest = load_manifest();
    assert_eq!(manifest.schema_version, 1, "unexpected manifest version");

    for example in manifest.examples {
        let mut args = apply_args(GaussianSplattingViewer::default(), &example);
        args.width = THUMB_WIDTH as f32;
        args.height = THUMB_HEIGHT as f32;

        let output_path = PathBuf::from("www/examples").join(&example.thumbnail);
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create thumbnail directory");
        }

        let started = Instant::now();
        println!(
            "[thumbnails] rendering '{}' -> {}",
            example.id,
            output_path.display()
        );
        render_example(args, output_path);
        println!(
            "[thumbnails] rendered '{}' in {:?}",
            example.id,
            started.elapsed()
        );
    }
}

fn load_manifest() -> ExamplesManifest {
    let data = std::fs::read_to_string(MANIFEST_PATH).expect("failed to read examples manifest");
    serde_json::from_str(&data).expect("failed to parse examples manifest")
}

fn apply_args(
    mut base: GaussianSplattingViewer,
    example: &ExampleEntry,
) -> GaussianSplattingViewer {
    let effective_scene = example
        .thumbnail_input_scene
        .as_ref()
        .or(example.input_scene.as_ref());
    let effective_cloud = example
        .thumbnail_input_cloud
        .as_ref()
        .or(example.input_cloud.as_ref());

    if effective_scene.is_some() && effective_cloud.is_some() {
        panic!(
            "example '{}' cannot define both input_scene and input_cloud",
            example.id
        );
    }

    let mut base_value = serde_json::to_value(&base).expect("failed to serialize args");
    let Some(base_map) = base_value.as_object_mut() else {
        panic!("expected base args to serialize to object");
    };

    if let Some(args_map) = example.args.as_object() {
        for (key, value) in args_map.iter() {
            if !base_map.contains_key(key) {
                panic!("unknown viewer arg: {key}");
            }
            base_map.insert(key.clone(), value.clone());
        }
    } else if !example.args.is_null() {
        panic!("expected args to be a JSON object");
    }

    base = serde_json::from_value(base_value).expect("failed to deserialize args");

    if let Some(input_scene) = effective_scene {
        let resolved_scene = resolve_thumbnail_scene_input(input_scene);
        base.input_scene = Some(resolved_scene);
        base.input_cloud = None;
    } else if let Some(input_cloud) = effective_cloud {
        base.input_cloud = Some(input_cloud.clone());
        base.input_scene = None;
    }

    base
}

fn resolve_thumbnail_scene_input(input_scene: &str) -> String {
    let is_remote = input_scene.starts_with("https://") || input_scene.starts_with("http://");
    if !is_remote {
        return input_scene.to_owned();
    }

    let strict_cache = std::env::var("THUMBNAIL_SCENE_CACHE_STRICT")
        .ok()
        .as_deref()
        == Some("1");
    let Some(cache_dir) = std::env::var("THUMBNAIL_SCENE_CACHE_DIR").ok() else {
        return input_scene.to_owned();
    };

    let url_without_query = input_scene.split('?').next().unwrap_or(input_scene);
    let Some(file_name) = url_without_query.rsplit('/').next() else {
        return input_scene.to_owned();
    };
    if file_name.is_empty() {
        return input_scene.to_owned();
    }

    let cached_path = PathBuf::from(cache_dir).join(file_name);
    if cached_path.exists() {
        let resolved_path = cached_path
            .canonicalize()
            .unwrap_or_else(|_| cached_path.clone())
            .to_string_lossy()
            .replace('\\', "/");
        println!(
            "[thumbnails] using cached scene for '{}': {}",
            input_scene, resolved_path
        );
        return resolved_path;
    }

    if strict_cache {
        panic!(
            "missing cached thumbnail scene for '{}' at '{}'",
            input_scene,
            cached_path.display()
        );
    }

    println!(
        "[thumbnails] scene cache miss for '{}', falling back to remote URL",
        input_scene
    );
    input_scene.to_owned()
}

fn supported_thumbnail_sort_modes() -> String {
    let mut modes = vec!["default", "none"];
    #[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
    modes.push("radix");
    #[cfg(feature = "sort_rayon")]
    modes.push("rayon");
    #[cfg(feature = "sort_std")]
    modes.push("std");
    modes.join(", ")
}

fn preferred_thumbnail_sort_mode() -> SortMode {
    let requested = std::env::var("THUMBNAIL_SORT_MODE")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty());

    if let Some(value) = requested.as_deref() {
        if value == "default" {
            return SortMode::default();
        }
        if value == "none" {
            return SortMode::None;
        }
        #[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
        if value == "radix" {
            return SortMode::Radix;
        }
        #[cfg(feature = "sort_rayon")]
        if value == "rayon" {
            return SortMode::Rayon;
        }
        #[cfg(feature = "sort_std")]
        if value == "std" {
            return SortMode::Std;
        }

        panic!(
            "unsupported THUMBNAIL_SORT_MODE='{value}', expected one of: {}",
            supported_thumbnail_sort_modes()
        );
    }

    #[cfg(feature = "sort_std")]
    {
        SortMode::Std
    }

    #[cfg(all(not(feature = "sort_std"), feature = "sort_rayon"))]
    {
        SortMode::Rayon
    }

    #[cfg(all(not(feature = "sort_std"), not(feature = "sort_rayon")))]
    {
        SortMode::default()
    }
}

fn render_example(args: GaussianSplattingViewer, output_path: PathBuf) {
    let sort_mode = preferred_thumbnail_sort_mode();
    println!("[thumbnails] thumbnail sort mode: {sort_mode:?}");

    App::new()
        .insert_resource(CaptureController::new(THUMB_WIDTH, THUMB_HEIGHT))
        .insert_resource(OutputTarget { path: output_path })
        .insert_resource(ThumbnailRenderConfig { sort_mode })
        .insert_resource(AutoFrameState::default())
        .insert_resource(ClearColor(Color::srgb_u8(0, 0, 0)))
        .insert_resource(args)
        .add_plugins(
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
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, setup_gaussian_cloud)
        .add_systems(
            Update,
            (
                apply_scene_camera_spawn,
                apply_scene_render_mode_override,
                mark_capture_ready,
                request_screenshot_capture,
            )
                .chain(),
        )
        .add_observer(on_screenshot_captured)
        .run();
}

#[allow(clippy::too_many_arguments)]
fn setup_gaussian_cloud(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    args: Res<GaussianSplattingViewer>,
    render_config: Res<ThumbnailRenderConfig>,
    mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
    mut gaussian_4d_assets: ResMut<Assets<PlanarGaussian4d>>,
    mut images: ResMut<Assets<Image>>,
    controller: Res<CaptureController>,
) {
    let cloud_transform = args.cloud_transform();
    let cloud_settings = CloudSettings {
        gaussian_mode: args.gaussian_mode,
        playback_mode: args.playback_mode,
        rasterize_mode: args.rasterization_mode,
        radix_sort_depth_bits: args.radix_sort_depth_bits,
        sort_mode: render_config.sort_mode.clone(),
        global_scale: 8.0,
        global_opacity: 2.0,
        ..default()
    };

    let size = Extent3d {
        width: controller.width,
        height: controller.height,
        ..default()
    };

    let render_target_handle = images.add(Image::new_target_texture(
        size.width,
        size.height,
        TextureFormat::Rgba8UnormSrgb,
        None,
    ));
    commands.insert_resource(CaptureRenderTarget(render_target_handle.clone()));

    if let Some(input_scene) = &args.input_scene {
        let scene_handle: Handle<GaussianScene> = asset_server.load(input_scene.clone());
        commands.spawn((
            GaussianSceneHandle(scene_handle),
            Name::new("gaussian_scene"),
            cloud_transform,
        ));
    } else {
        match args.gaussian_mode {
            GaussianMode::Gaussian2d | GaussianMode::Gaussian3d => {
                let cloud = if args.gaussian_count > 0 {
                    if let Some(seed) = args.gaussian_seed {
                        gaussian_assets.add(random_gaussians_3d_seeded(args.gaussian_count, seed))
                    } else {
                        gaussian_assets.add(random_gaussians_3d(args.gaussian_count))
                    }
                } else if let Some(input_cloud) = &args.input_cloud {
                    if input_cloud.ends_with(".ply") {
                        gaussian_assets.add(load_ply_cloud(input_cloud))
                    } else {
                        asset_server.load(input_cloud)
                    }
                } else {
                    gaussian_assets.add(PlanarGaussian3d::test_model())
                };

                commands.spawn((
                    PlanarGaussian3dHandle(cloud),
                    cloud_settings.clone(),
                    Name::new("gaussian_cloud"),
                    cloud_transform,
                    Visibility::Visible,
                ));
            }
            GaussianMode::Gaussian4d => {
                let cloud = if args.gaussian_count > 0 {
                    if let Some(seed) = args.gaussian_seed {
                        gaussian_4d_assets
                            .add(random_gaussians_4d_seeded(args.gaussian_count, seed))
                    } else {
                        gaussian_4d_assets.add(random_gaussians_4d(args.gaussian_count))
                    }
                } else if let Some(input_cloud) = &args.input_cloud {
                    asset_server.load(input_cloud)
                } else {
                    gaussian_4d_assets.add(PlanarGaussian4d::test_model())
                };

                commands.spawn((
                    PlanarGaussian4dHandle(cloud),
                    cloud_settings,
                    Name::new("gaussian_cloud"),
                    cloud_transform,
                    Visibility::Visible,
                ));
            }
        }
    }

    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        RenderTarget::Image(render_target_handle.into()),
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
        Tonemapping::None,
        GaussianCamera::default(),
    ));
}

fn apply_scene_camera_spawn(
    mut commands: Commands,
    scene_handles: Query<(Entity, &GaussianSceneHandle), Without<SceneCameraApplied>>,
    asset_server: Res<AssetServer>,
    scenes: Res<Assets<GaussianScene>>,
    mut cameras: Query<&mut Transform, With<GaussianCamera>>,
) {
    for (entity, scene_handle) in scene_handles.iter() {
        if let Some(load_state) = asset_server.get_load_state(&scene_handle.0) {
            match load_state {
                LoadState::Failed(err) => {
                    panic!("failed to load scene asset {:?}: {err}", scene_handle.0);
                }
                state if !state.is_loaded() => continue,
                _ => {}
            }
        }

        let Some(scene) = scenes.get(&scene_handle.0) else {
            continue;
        };

        if let Some(scene_camera) = scene.cameras.first()
            && let Ok(mut camera_transform) = cameras.single_mut()
        {
            *camera_transform = scene_camera.transform;
        }

        commands.entity(entity).insert(SceneCameraApplied);
    }
}

fn apply_scene_render_mode_override(
    mut commands: Commands,
    args: Res<GaussianSplattingViewer>,
    render_config: Res<ThumbnailRenderConfig>,
    scenes: Query<SceneRenderModeQuery, SceneRenderModeFilter>,
    mut cloud_settings: Query<&mut CloudSettings>,
) {
    if args.input_scene.is_none() {
        return;
    }

    for (entity, children) in scenes.iter() {
        for child in children.iter() {
            let child: Entity = child;
            if let Ok(mut settings) = cloud_settings.get_mut(child) {
                settings.rasterize_mode = args.rasterization_mode;
                settings.radix_sort_depth_bits = args.radix_sort_depth_bits;
                settings.sort_mode = render_config.sort_mode.clone();
            }
        }

        commands.entity(entity).insert(SceneRenderModeApplied);
    }
}

#[allow(clippy::too_many_arguments)]
fn mark_capture_ready(
    mut auto_frame: ResMut<AutoFrameState>,
    args: Res<GaussianSplattingViewer>,
    asset_server: Res<AssetServer>,
    scenes: Res<Assets<GaussianScene>>,
    scene_handles: Query<SceneReadyQuery, SceneReadyFilter>,
    cloud_assets: Res<Assets<PlanarGaussian3d>>,
    cloud_assets_4d: Res<Assets<PlanarGaussian4d>>,
    child_cloud_handles: Query<&PlanarGaussian3dHandle>,
    cloud_handles: Query<&PlanarGaussian3dHandle>,
    cloud_handles_4d: Query<&PlanarGaussian4dHandle>,
) {
    if auto_frame.done {
        return;
    }

    if args.input_scene.is_some() {
        for (_, scene_handle, children, camera_applied, render_mode_applied) in scene_handles.iter()
        {
            if let Some(load_state) = asset_server.get_load_state(&scene_handle.0) {
                match load_state {
                    LoadState::Failed(err) => {
                        panic!("failed to load scene asset {:?}: {err}", scene_handle.0);
                    }
                    state if !state.is_loaded() => continue,
                    _ => {}
                }
            }

            if scenes.get(&scene_handle.0).is_none()
                || camera_applied.is_none()
                || render_mode_applied.is_none()
            {
                continue;
            }

            let mut scene_cloud_count = 0usize;
            let mut scene_clouds_ready = true;

            for child in children.iter() {
                let child: Entity = child;
                let Ok(cloud_handle) = child_cloud_handles.get(child) else {
                    continue;
                };

                scene_cloud_count += 1;

                if let Some(load_state) = asset_server.get_load_state(&cloud_handle.0) {
                    match load_state {
                        LoadState::Failed(err) => {
                            panic!(
                                "failed to load scene cloud asset {:?}: {err}",
                                cloud_handle.0
                            );
                        }
                        state if !state.is_loaded() => {
                            scene_clouds_ready = false;
                            break;
                        }
                        _ => {}
                    }
                }

                if cloud_assets.get(&cloud_handle.0).is_none() {
                    scene_clouds_ready = false;
                    break;
                }
            }

            if scene_cloud_count > 0 && scene_clouds_ready {
                println!(
                    "[thumbnails] scene ready (clouds={}, camera_applied={}, render_mode_applied={})",
                    scene_cloud_count,
                    camera_applied.is_some(),
                    render_mode_applied.is_some()
                );
                auto_frame.done = true;
                return;
            }
        }
        return;
    }

    for cloud_handle in cloud_handles.iter() {
        if let Some(load_state) = asset_server.get_load_state(&cloud_handle.0) {
            match load_state {
                LoadState::Failed(err) => {
                    panic!("failed to load cloud asset {:?}: {err}", cloud_handle.0);
                }
                state if !state.is_loaded() => continue,
                _ => {}
            }
        }

        if cloud_assets.get(&cloud_handle.0).is_some() {
            println!("[thumbnails] cloud ready (3d)");
            auto_frame.done = true;
            return;
        }
    }

    for cloud_handle in cloud_handles_4d.iter() {
        if let Some(load_state) = asset_server.get_load_state(&cloud_handle.0) {
            match load_state {
                LoadState::Failed(err) => {
                    panic!("failed to load 4d cloud asset {:?}: {err}", cloud_handle.0);
                }
                state if !state.is_loaded() => continue,
                _ => {}
            }
        }

        if cloud_assets_4d.get(&cloud_handle.0).is_some() {
            println!("[thumbnails] cloud ready (4d)");
            auto_frame.done = true;
            return;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn request_screenshot_capture(
    mut commands: Commands,
    capture_target: Option<Res<CaptureRenderTarget>>,
    output_target: Res<OutputTarget>,
    auto_frame: Res<AutoFrameState>,
    render_config: Res<ThumbnailRenderConfig>,
    cameras: Query<&SortTrigger, With<GaussianCamera>>,
    clouds_3d: Query<Option<&SortedEntriesHandle>, With<PlanarGaussian3dHandle>>,
    clouds_4d: Query<Option<&SortedEntriesHandle>, With<PlanarGaussian4dHandle>>,
    mut controller: ResMut<CaptureController>,
) {
    let elapsed = controller.started_at.elapsed();
    if elapsed > controller.max_elapsed {
        panic!(
            "timed out while generating thumbnail: {:?} (elapsed={:?}, auto_frame.done={}, frames_since_ready={}, capture_requested={})",
            output_target.path,
            elapsed,
            auto_frame.done,
            controller.frames_since_ready,
            controller.capture_requested,
        );
    }

    controller.total_frames += 1;
    if controller.total_frames > controller.max_total_frames {
        panic!(
            "timed out while generating thumbnail: {:?} (elapsed={:?}, total_frames={}, auto_frame.done={}, frames_since_ready={}, capture_requested={})",
            output_target.path,
            elapsed,
            controller.total_frames,
            auto_frame.done,
            controller.frames_since_ready,
            controller.capture_requested,
        );
    }

    if !auto_frame.done {
        controller.frames_since_ready = 0;
        return;
    }

    let requires_cpu_sort = match render_config.sort_mode {
        #[cfg(feature = "sort_std")]
        SortMode::Std => true,
        #[cfg(feature = "sort_rayon")]
        SortMode::Rayon => true,
        _ => false,
    };

    if requires_cpu_sort {
        let mut sort_ready = true;
        let mut saw_camera = false;
        for trigger in cameras.iter() {
            saw_camera = true;
            if trigger.needs_sort {
                sort_ready = false;
                break;
            }
        }
        if !saw_camera {
            sort_ready = false;
        }

        let mut saw_cloud = false;
        if sort_ready {
            for sorted_handle in clouds_3d.iter() {
                saw_cloud = true;
                if sorted_handle.is_none() {
                    sort_ready = false;
                    break;
                }
            }
        }
        if sort_ready {
            for sorted_handle in clouds_4d.iter() {
                saw_cloud = true;
                if sorted_handle.is_none() {
                    sort_ready = false;
                    break;
                }
            }
        }
        if !saw_cloud {
            sort_ready = false;
        }

        if !sort_ready {
            controller.frames_since_ready = 0;
            return;
        }
    }

    controller.frames_since_ready += 1;
    if controller.frames_since_ready < controller.warmup_frames_after_ready {
        return;
    }

    if controller.capture_requested {
        return;
    }

    let Some(capture_target) = capture_target else {
        return;
    };

    println!(
        "[thumbnails] requesting screenshot (elapsed={:?}, frames_since_ready={})",
        elapsed, controller.frames_since_ready
    );
    commands.spawn(Screenshot::image(capture_target.0.clone()));
    controller.capture_requested = true;
}

fn on_screenshot_captured(
    trigger: On<ScreenshotCaptured>,
    output_target: Res<OutputTarget>,
    mut app_exit: MessageWriter<AppExit>,
) {
    println!(
        "[thumbnails] screenshot captured for '{}'",
        output_target.path.display()
    );
    let img = match trigger.image.clone().try_into_dynamic() {
        Ok(img) => img.to_rgba8(),
        Err(e) => panic!("Failed to convert screenshot image: {e:?}"),
    };

    if let Err(e) = img.save(&output_target.path) {
        panic!("Failed to save image: {e}");
    }

    app_exit.write(AppExit::Success);
}

fn load_ply_cloud(input_cloud: &str) -> PlanarGaussian3d {
    let direct_path = PathBuf::from(input_cloud);
    let path = if direct_path.exists() {
        direct_path
    } else {
        Path::new("assets").join(input_cloud)
    };
    let file = std::fs::File::open(&path).unwrap_or_else(|err| {
        panic!("failed to open PLY file for thumbnail render {path:?}: {err}")
    });
    let mut reader = BufReader::new(file);
    parse_ply_3d(&mut reader).unwrap_or_else(|err| {
        panic!("failed to parse PLY file for thumbnail render {path:?}: {err}")
    })
}

pub struct ImageCopyPlugin;

impl Plugin for ImageCopyPlugin {
    fn build(&self, app: &mut App) {
        let (sender, receiver) = crossbeam_channel::unbounded();

        let render_app = app
            .insert_resource(MainWorldReceiver(receiver))
            .sub_app_mut(RenderApp);

        render_app
            .insert_resource(RenderWorldSender(sender))
            .add_systems(ExtractSchedule, extract_image_copiers)
            .add_systems(
                Render,
                (copy_images_to_buffer, receive_image_from_buffer)
                    .chain()
                    .after(RenderSystems::Render),
            );
    }
}

pub struct CaptureFramePlugin;

impl Plugin for CaptureFramePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, save_captured_frame);
    }
}

#[derive(Clone, Component)]
struct ImageCopier {
    buffer: Buffer,
    enabled: Arc<AtomicBool>,
    src_image: Handle<Image>,
}

impl ImageCopier {
    pub fn new(src_image: Handle<Image>, size: Extent3d, render_device: &RenderDevice) -> Self {
        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(size.width as usize) * 4;

        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("image_copier_buffer"),
            size: padded_bytes_per_row as u64 * size.height as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            src_image,
            enabled: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Default, Resource, Deref)]
struct ImageCopiers(Vec<ImageCopier>);

fn extract_image_copiers(mut commands: Commands, image_copiers: Extract<Query<&ImageCopier>>) {
    commands.insert_resource(ImageCopiers(image_copiers.iter().cloned().collect()));
}

fn copy_images_to_buffer(
    render_context: RenderContext,
    image_copiers: Option<Res<ImageCopiers>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    render_queue: Res<RenderQueue>,
) {
    let Some(image_copiers) = image_copiers else {
        return;
    };

    for image_copier in image_copiers.iter() {
        if !image_copier.enabled() {
            continue;
        }

        let Some(src_image) = gpu_images.get(&image_copier.src_image) else {
            continue;
        };

        let mut encoder = render_context
            .render_device()
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let block_dimensions = src_image.texture_descriptor.format.block_dimensions();
        let block_size = src_image
            .texture_descriptor
            .format
            .block_copy_size(None)
            .unwrap();

        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
            (src_image.texture_descriptor.size.width as usize / block_dimensions.0 as usize)
                * block_size as usize,
        );

        encoder.copy_texture_to_buffer(
            src_image.texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &image_copier.buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        std::num::NonZero::<u32>::new(padded_bytes_per_row as u32)
                            .unwrap()
                            .into(),
                    ),
                    rows_per_image: None,
                },
            },
            src_image.texture_descriptor.size,
        );

        render_queue.submit(std::iter::once(encoder.finish()));
    }
}

fn receive_image_from_buffer(
    image_copiers: Res<ImageCopiers>,
    render_device: Res<RenderDevice>,
    sender: Res<RenderWorldSender>,
) {
    for image_copier in image_copiers.0.iter() {
        if !image_copier.enabled() {
            continue;
        }

        let buffer_slice = image_copier.buffer.slice(..);
        let (tx, rx) = crossbeam_channel::bounded(1);

        buffer_slice.map_async(MapMode::Read, move |result| match result {
            Ok(()) => tx.send(()).expect("Failed to send map result"),
            Err(err) => panic!("Failed to map buffer: {err}"),
        });

        render_device
            .poll(PollType::wait_indefinitely())
            .expect("Failed to poll device");

        rx.recv().expect("Failed to receive buffer map");

        let _ = sender.send(buffer_slice.get_mapped_range().to_vec());
        image_copier.buffer.unmap();
    }
}

#[derive(Component, Deref)]
struct ImageToSave(Handle<Image>);

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn save_captured_frame(
    _images_to_save: Query<&ImageToSave>,
    _clouds: Query<
        (
            Option<&Aabb>,
            Option<&SortedEntriesHandle>,
            Option<&ViewVisibility>,
        ),
        With<PlanarGaussian3dHandle>,
    >,
    _cameras: Query<(&Camera, &RenderTarget), With<GaussianCamera>>,
    receiver: Res<MainWorldReceiver>,
    _output_target: Res<OutputTarget>,
    _auto_frame: Res<AutoFrameState>,
    _images: ResMut<Assets<Image>>,
    _controller: ResMut<CaptureController>,
    _app_exit: MessageWriter<AppExit>,
) {
    while receiver.try_recv().is_ok() {}
}
