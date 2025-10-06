//! Headless rendering for gaussian splatting
//!
//! Renders gaussian splatting to images without creating a window.
//! Based on Bevy's headless_renderer example.
//!
//! Usage: cargo run --example headless --no-default-features --features "headless" -- [filename]

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::RenderTarget,
    core_pipeline::tonemapping::Tonemapping,
    image::TextureFormatPixelInfo,
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_graph::{self, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode,
            PollType, TexelCopyBufferInfo, TexelCopyBufferLayout, TextureFormat, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Extract, Render, RenderApp, RenderSystems,
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_args::BevyArgsPlugin;
use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, GaussianSplattingPlugin, PlanarGaussian3d,
    PlanarGaussian3dHandle, gaussian::interface::TestCloud, random_gaussians_3d,
    utils::GaussianSplattingViewer,
};
use crossbeam_channel::{Receiver, Sender};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

#[derive(Resource, Deref)]
struct MainWorldReceiver(Receiver<Vec<u8>>);

#[derive(Resource, Deref)]
struct RenderWorldSender(Sender<Vec<u8>>);

#[derive(Debug, Default, Resource)]
struct CaptureController {
    frames_to_wait: u32,
    width: u32,
    height: u32,
}

impl CaptureController {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            frames_to_wait: 40, 
            width,
            height,
        }
    }
}

fn main() {
    App::new()
        .insert_resource(CaptureController::new(1920, 1080))
        .insert_resource(ClearColor(Color::srgb_u8(0, 0, 0)))
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                // Disable WinitPlugin for headless environments
                .disable::<WinitPlugin>(),
        )
        .add_plugins(BevyArgsPlugin::<GaussianSplattingViewer>::default())
        .add_plugins(ImageCopyPlugin)
        .add_plugins(CaptureFramePlugin)
        .add_plugins(ScheduleRunnerPlugin::run_loop(
            Duration::from_secs_f64(1.0 / 60.0),
        ))
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, setup_gaussian_cloud)
        .run();
}

fn setup_gaussian_cloud(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    args: Res<GaussianSplattingViewer>,
    mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
    mut images: ResMut<Assets<Image>>,
    render_device: Res<RenderDevice>,
    controller: Res<CaptureController>,
) {
    // Load or generate gaussian cloud
    let cloud = if args.gaussian_count > 0 {
        println!("Generating {} gaussians", args.gaussian_count);
        gaussian_assets.add(random_gaussians_3d(args.gaussian_count))
    } else if args.input_cloud.is_some() && !args.input_cloud.as_ref().unwrap().is_empty() {
        println!("Loading {:?}", args.input_cloud);
        asset_server.load(&args.input_cloud.as_ref().unwrap().clone())
    } else {
        gaussian_assets.add(PlanarGaussian3d::test_model())
    };

    // Setup render target
    let size = Extent3d {
        width: controller.width,
        height: controller.height,
        ..default()
    };

    let mut render_target_image = Image::new_target_texture(
        size.width,
        size.height,
        TextureFormat::bevy_default(),
    );
    render_target_image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let render_target_handle = images.add(render_target_image);

    let cpu_image = Image::new_target_texture(
        size.width,
        size.height,
        TextureFormat::bevy_default(),
    );
    let cpu_image_handle = images.add(cpu_image);

    commands.spawn((
        PlanarGaussian3dHandle(cloud),
        CloudSettings::default(),
        Name::new("gaussian_cloud"),
    ));

    commands.spawn((
        Camera3d::default(),
        Camera {
            target: RenderTarget::Image(render_target_handle.clone().into()),
            ..default()
        },
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
        Tonemapping::None,
        GaussianCamera::default(),
    ));

    // Spawn image copier for GPU->CPU transfer
    commands.spawn(ImageCopier::new(
        render_target_handle,
        size,
        &render_device,
    ));

    // Spawn image to save
    commands.spawn(ImageToSave(cpu_image_handle));
}

/// Plugin for copying images from GPU to CPU
pub struct ImageCopyPlugin;

impl Plugin for ImageCopyPlugin {
    fn build(&self, app: &mut App) {
        let (sender, receiver) = crossbeam_channel::unbounded();

        let render_app = app
            .insert_resource(MainWorldReceiver(receiver))
            .sub_app_mut(RenderApp);

        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(ImageCopy, ImageCopyDriver);
        graph.add_node_edge(bevy::render::graph::CameraDriverLabel, ImageCopy);

        render_app
            .insert_resource(RenderWorldSender(sender))
            .add_systems(ExtractSchedule, extract_image_copiers)
            .add_systems(
                Render,
                receive_image_from_buffer.after(RenderSystems::Render),
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
        let padded_bytes_per_row =
            RenderDevice::align_copy_bytes_per_row(size.width as usize) * 4;

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

fn extract_image_copiers(
    mut commands: Commands,
    image_copiers: Extract<Query<&ImageCopier>>,
) {
    commands.insert_resource(ImageCopiers(
        image_copiers.iter().cloned().collect(),
    ));
}

/// RenderGraph label
#[derive(Debug, PartialEq, Eq, Clone, Hash, RenderLabel)]
struct ImageCopy;

#[derive(Default)]
struct ImageCopyDriver;

impl render_graph::Node for ImageCopyDriver {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let image_copiers = world.get_resource::<ImageCopiers>().unwrap();
        let gpu_images = world.get_resource::<RenderAssets<GpuImage>>().unwrap();

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

            let block_dimensions = src_image.texture_format.block_dimensions();
            let block_size = src_image.texture_format.block_copy_size(None).unwrap();

            let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
                (src_image.size.width as usize / block_dimensions.0 as usize) * block_size as usize,
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
                src_image.size,
            );

            let render_queue = world.get_resource::<RenderQueue>().unwrap();
            render_queue.submit(std::iter::once(encoder.finish()));
        }

        Ok(())
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
            .poll(PollType::Wait)
            .expect("Failed to poll device");

        rx.recv().expect("Failed to receive buffer map");

        let _ = sender.send(buffer_slice.get_mapped_range().to_vec());
        image_copier.buffer.unmap();
    }
}

#[derive(Component, Deref)]
struct ImageToSave(Handle<Image>);

fn save_captured_frame(
    images_to_save: Query<&ImageToSave>,
    receiver: Res<MainWorldReceiver>,
    mut images: ResMut<Assets<Image>>,
    mut controller: ResMut<CaptureController>,
    mut app_exit: MessageWriter<AppExit>,
) {
    if controller.frames_to_wait > 0 {
        controller.frames_to_wait -= 1;
        while receiver.try_recv().is_ok() {}
        return;
    }

    // Try to receive image data
    let mut image_data = Vec::new();
    while let Ok(data) = receiver.try_recv() {
        image_data = data; 
    }

    if image_data.is_empty() {
        return;
    }

    for image_handle in images_to_save.iter() {
        let Some(image) = images.get_mut(image_handle.id()) else {
            continue;
        };

        let row_bytes = image.width() as usize
            * image.texture_descriptor.format.pixel_size().unwrap();
        let aligned_row_bytes = RenderDevice::align_copy_bytes_per_row(row_bytes);

        if row_bytes == aligned_row_bytes {
            image.data.as_mut().unwrap().clone_from(&image_data);
        } else {
            // Shrink to original size
            image.data = Some(
                image_data
                    .chunks(aligned_row_bytes)
                    .take(image.height() as usize)
                    .flat_map(|row| &row[..row_bytes.min(row.len())])
                    .cloned()
                    .collect(),
            );
        }

        let img = match image.clone().try_into_dynamic() {
            Ok(img) => img.to_rgba8(),
            Err(e) => panic!("Failed to create image: {e:?}"),
        };

        let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("headless_output");
        std::fs::create_dir_all(&output_dir).unwrap();
        let output_path = output_dir.join("0.png");

        info!("Saving screenshot to {:?}", output_path);
        if let Err(e) = img.save(&output_path) {
            panic!("Failed to save image: {e}");
        }
    }

    app_exit.write(AppExit::Success);
}