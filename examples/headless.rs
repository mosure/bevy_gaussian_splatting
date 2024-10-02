// for running the gaussian splatting viewer without a window ( i.e on a server )
//! ensure the "headless_output" directory exists in the root of the project
// c_rr --example headless --no-default-features --features "headless" -- [filename]

use bevy::{
    prelude::*,
    app::ScheduleRunnerPlugin,
    core::Name,
    core_pipeline::tonemapping::Tonemapping,
    render::renderer::RenderDevice,
};
use bevy_args::BevyArgsPlugin;

use bevy_gaussian_splatting::{
    GaussianCamera,
    GaussianCloud,
    GaussianSplattingBundle,
    GaussianSplattingPlugin,
    random_gaussians,
    utils::GaussianSplattingViewer,
};


/// Derived from: https://github.com/bevyengine/bevy/pull/5550
mod frame_capture {
    pub mod image_copy {
        use std::sync::Arc;

        use bevy::prelude::*;
        use bevy::render::render_asset::RenderAssets;
        use bevy::render::render_graph::{self, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel};
        use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
        use bevy::render::{Extract, RenderApp};
        use bevy::render::texture::GpuImage;

        use bevy::render::render_resource::{
            Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d,
            ImageCopyBuffer, ImageDataLayout, MapMode,
        };
        use pollster::FutureExt;
        use wgpu::Maintain;

        use std::sync::atomic::{AtomicBool, Ordering};

        pub fn receive_images(
            image_copiers: Query<&ImageCopier>,
            mut images: ResMut<Assets<Image>>,
            render_device: Res<RenderDevice>,
        ) {
            for image_copier in image_copiers.iter() {
                if !image_copier.enabled() {
                    continue;
                }
                // Derived from: https://sotrh.github.io/learn-wgpu/showcase/windowless/#a-triangle-without-a-window
                // We need to scope the mapping variables so that we can
                // unmap the buffer
                async {
                    let buffer_slice = image_copier.buffer.slice(..);

                    // NOTE: We have to create the mapping THEN device.poll() before await
                    // the future. Otherwise the application will freeze.
                    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
                    buffer_slice.map_async(MapMode::Read, move |result| {
                        tx.send(result).unwrap();
                    });
                    render_device.poll(Maintain::Wait);
                    rx.receive().await.unwrap().unwrap();
                    if let Some(image) = images.get_mut(&image_copier.dst_image) {
                        image.data = buffer_slice.get_mapped_range().to_vec();
                    }

                    image_copier.buffer.unmap();
                }
                .block_on();
            }
        }

        #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
        pub struct ImageCopyLabel;

        pub struct ImageCopyPlugin;
        impl Plugin for ImageCopyPlugin {
            fn build(&self, app: &mut App) {
                let render_app = app
                    .add_systems(Update, receive_images)
                    .sub_app_mut(RenderApp);

                render_app.add_systems(ExtractSchedule, image_copy_extract);

                let mut graph = render_app.world_mut().get_resource_mut::<RenderGraph>().unwrap();

                graph.add_node(ImageCopyLabel, ImageCopyDriver);
                graph.add_node_edge(ImageCopyLabel, bevy::render::graph::CameraDriverLabel);
            }
        }

        #[derive(Clone, Default, Resource, Deref, DerefMut)]
        pub struct ImageCopiers(pub Vec<ImageCopier>);

        #[derive(Clone, Component)]
        pub struct ImageCopier {
            buffer: Buffer,
            enabled: Arc<AtomicBool>,
            src_image: Handle<Image>,
            dst_image: Handle<Image>,
        }

        impl ImageCopier {
            pub fn new(
                src_image: Handle<Image>,
                dst_image: Handle<Image>,
                size: Extent3d,
                render_device: &RenderDevice,
            ) -> ImageCopier {
                let padded_bytes_per_row =
                    RenderDevice::align_copy_bytes_per_row((size.width) as usize) * 4;

                let cpu_buffer = render_device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: padded_bytes_per_row as u64 * size.height as u64,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                ImageCopier {
                    buffer: cpu_buffer,
                    src_image,
                    dst_image,
                    enabled: Arc::new(AtomicBool::new(true)),
                }
            }

            pub fn enabled(&self) -> bool {
                self.enabled.load(Ordering::Relaxed)
            }
        }

        pub fn image_copy_extract(
            mut commands: Commands,
            image_copiers: Extract<Query<&ImageCopier>>,
        ) {
            commands.insert_resource(ImageCopiers(
                image_copiers.iter().cloned().collect::<Vec<ImageCopier>>(),
            ));
        }

        #[derive(Default)]
        pub struct ImageCopyDriver;

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

                    let src_image = gpu_images.get(&image_copier.src_image).unwrap();

                    let mut encoder = render_context
                        .render_device()
                        .create_command_encoder(&CommandEncoderDescriptor::default());

                    let block_dimensions = src_image.texture_format.block_dimensions();
                    let block_size = src_image.texture_format.block_copy_size(None).unwrap();

                    let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
                        (src_image.size.x as usize / block_dimensions.0 as usize)
                            * block_size as usize,
                    );

                    let texture_extent = Extent3d {
                        width: src_image.size.x as u32,
                        height: src_image.size.y as u32,
                        depth_or_array_layers: 1,
                    };

                    encoder.copy_texture_to_buffer(
                        src_image.texture.as_image_copy(),
                        ImageCopyBuffer {
                            buffer: &image_copier.buffer,
                            layout: ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(
                                    std::num::NonZeroU32::new(padded_bytes_per_row as u32)
                                        .unwrap()
                                        .into(),
                                ),
                                rows_per_image: None,
                            },
                        },
                        texture_extent,
                    );

                    let render_queue = world.get_resource::<RenderQueue>().unwrap();
                    render_queue.submit(std::iter::once(encoder.finish()));
                }

                Ok(())
            }
        }
    }
    pub mod scene {
        use std::path::PathBuf;

        use bevy::{
            app::AppExit,
            prelude::*,
            render::{camera::RenderTarget, renderer::RenderDevice},
        };
        use wgpu::{Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};

        use super::image_copy::ImageCopier;


        #[derive(Component, Default)]
        pub struct CaptureCamera;

        #[derive(Component, Deref, DerefMut)]
        struct ImageToSave(Handle<Image>);

        pub struct CaptureFramePlugin;
        impl Plugin for CaptureFramePlugin {
            fn build(&self, app: &mut App) {
                app.add_systems(PostUpdate, update);
            }
        }

        #[derive(Debug, Default, Resource, Event)]
        pub struct SceneController {
            state: SceneState,
            name: String,
            width: u32,
            height: u32,
            single_image: bool,
        }

        impl SceneController {
            pub fn new(width:u32, height:u32, single_image: bool) -> SceneController {
                SceneController {
                    state: SceneState::BuildScene,
                    name: String::from(""),
                    width,
                    height,
                    single_image
                }
            }
        }

        #[derive(Debug, Default)]
        pub enum SceneState {
            #[default]
            BuildScene,
            Render(u32),
        }

        pub fn setup_render_target(
            commands: &mut Commands,
            images: &mut ResMut<Assets<Image>>,
            render_device: &Res<RenderDevice>,
            scene_controller: &mut ResMut<SceneController>,
            pre_roll_frames: u32,
            scene_name: String,
        ) -> RenderTarget {
            let size = Extent3d {
                width: scene_controller.width,
                height: scene_controller.height,
                ..Default::default()
            };

            // This is the texture that will be rendered to.
            let mut render_target_image = Image {
                texture_descriptor: TextureDescriptor {
                    label: None,
                    size,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8UnormSrgb,
                    mip_level_count: 1,
                    sample_count: 1,
                    usage: TextureUsages::COPY_SRC
                        | TextureUsages::COPY_DST
                        | TextureUsages::TEXTURE_BINDING
                        | TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                },
                ..Default::default()
            };
            render_target_image.resize(size);
            let render_target_image_handle = images.add(render_target_image);

            // This is the texture that will be copied to.
            let mut cpu_image = Image {
                texture_descriptor: TextureDescriptor {
                    label: None,
                    size,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8UnormSrgb,
                    mip_level_count: 1,
                    sample_count: 1,
                    usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                ..Default::default()
            };
            cpu_image.resize(size);
            let cpu_image_handle = images.add(cpu_image);

            commands.spawn(ImageCopier::new(
                render_target_image_handle.clone(),
                cpu_image_handle.clone(),
                size,
                render_device,
            ));

            commands.spawn(ImageToSave(cpu_image_handle));

            scene_controller.state = SceneState::Render(pre_roll_frames);
            scene_controller.name = scene_name;
            RenderTarget::Image(render_target_image_handle)
        }

        fn update(
            images_to_save: Query<&ImageToSave>,
            mut images: ResMut<Assets<Image>>,
            mut scene_controller: ResMut<SceneController>,
            mut app_exit_writer: EventWriter<AppExit>,
        ) {
            if let SceneState::Render(n) = scene_controller.state {
                if n < 1 {
                    for (i, image) in images_to_save.iter().enumerate() {
                        let img_bytes = images.get_mut(image.id()).unwrap();

                        let img = match img_bytes.clone().try_into_dynamic() {
                            Ok(img) => img.to_rgba8(),
                            Err(e) => panic!("Failed to create image buffer {e:?}"),
                        };

                        let images_dir =
                            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("headless_output");
                        std::fs::create_dir_all(&images_dir).unwrap();

                        let image_path = images_dir.join(format!("{i}.png"));
                        if let Err(e) = img.save(image_path){
                            panic!("Failed to save image: {}", e);
                        };
                    }
                    if scene_controller.single_image {
                        app_exit_writer.send(AppExit::Success);
                    }
                } else {
                    scene_controller.state = SceneState::Render(n - 1);
                }
            }
        }
    }
}

fn setup_gaussian_cloud(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gaussian_splatting_viewer: Res<GaussianSplattingViewer>,
    mut gaussian_assets: ResMut<Assets<GaussianCloud>>,
    mut scene_controller: ResMut<frame_capture::scene::SceneController>,
    mut images: ResMut<Assets<Image>>,
    render_device: Res<RenderDevice>,
) {
    let cloud: Handle<GaussianCloud>;

    if gaussian_splatting_viewer.gaussian_count > 0 {
        println!("generating {} gaussians", gaussian_splatting_viewer.gaussian_count);
        cloud = gaussian_assets.add(random_gaussians(gaussian_splatting_viewer.gaussian_count));
    } else if !gaussian_splatting_viewer.input_file.is_empty() {
        println!("loading {}", gaussian_splatting_viewer.input_file);
        cloud = asset_server.load(&gaussian_splatting_viewer.input_file);
    } else {
        cloud = gaussian_assets.add(GaussianCloud::test_model());
    }

    let render_target = frame_capture::scene::setup_render_target(
        &mut commands,
        &mut images,
        &render_device,
        &mut scene_controller,
        15,
        String::from("main_scene"),
    );


    commands.spawn((
        GaussianSplattingBundle { cloud, ..default() },
        Name::new("gaussian_cloud"),
    ));

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
            tonemapping: Tonemapping::None,
            camera: Camera {
                target: render_target,
                ..default()
            },
            ..default()
        },
        GaussianCamera,
    ));
}

pub struct AppConfig {
    width: u32,
    height: u32,
    single_image: bool,
}

fn headless_app() {
    let mut app = App::new();

    let config = AppConfig {
        width: 1920,
        height: 1080,
        single_image: true,
    };

    // setup frame capture
    app.insert_resource(frame_capture::scene::SceneController::new(config.width, config.height, config.single_image));
    app.insert_resource(ClearColor(Color::srgb_u8(0, 0, 0)));

    app.add_plugins(
        DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(WindowPlugin {
            primary_window: None,
            exit_condition: bevy::window::ExitCondition::DontExit,
            close_when_requested: false,
        }),
    );
    app.add_plugins(BevyArgsPlugin::<GaussianSplattingViewer>::default());

    // headless frame capture
    app.add_plugins(frame_capture::image_copy::ImageCopyPlugin);
    app.add_plugins(frame_capture::scene::CaptureFramePlugin);

    app.add_plugins(ScheduleRunnerPlugin::run_loop(
        std::time::Duration::from_secs_f64(1.0 / 60.0),
    ));

    // setup for gaussian splatting
    app.add_plugins(GaussianSplattingPlugin);


    app.init_resource::<frame_capture::scene::SceneController>();
    app.add_event::<frame_capture::scene::SceneController>();

    app.add_systems(Startup, setup_gaussian_cloud);


    app.run();
}

pub fn main() {
    headless_app();
}
