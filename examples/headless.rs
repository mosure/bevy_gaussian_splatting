use bevy::{
    app::ScheduleRunnerPlugin, core::Name, core_pipeline::tonemapping::Tonemapping, prelude::*, render::renderer::RenderDevice,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use bevy_gaussian_splatting::{
    random_gaussians, utils::get_arg, GaussianCloud, GaussianSplattingBundle,
    GaussianSplattingPlugin,
};

#[cfg(feature = "material_noise")]
use bevy_gaussian_splatting::material::noise::NoiseMaterial;

#[cfg(feature = "morph_particles")]
use bevy_gaussian_splatting::morph::particle::{random_particle_behaviors, ParticleBehaviors};

#[cfg(feature = "query_select")]
use bevy_gaussian_splatting::query::select::{InvertSelectionEvent, SaveSelectionEvent};

#[cfg(feature = "query_sparse")]
use bevy_gaussian_splatting::query::sparse::SparseSelect;

// TODO: clean up later, make repo a workspace?
// Derived from: https://github.com/bevyengine/bevy/pull/5550

mod frame_capture {
    pub mod image_copy {
        use std::sync::Arc;

        use bevy::prelude::*;
        use bevy::render::render_asset::RenderAssets;
        use bevy::render::render_graph::{self, NodeRunError, RenderGraph, RenderGraphContext};
        use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
        use bevy::render::{Extract, RenderApp};

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

        pub const IMAGE_COPY: &str = "image_copy";

        pub struct ImageCopyPlugin;
        impl Plugin for ImageCopyPlugin {
            fn build(&self, app: &mut App) {
                let render_app = app
                    .add_systems(Update, receive_images)
                    .sub_app_mut(RenderApp);

                render_app.add_systems(ExtractSchedule, image_copy_extract);

                let mut graph = render_app.world.get_resource_mut::<RenderGraph>().unwrap();

                graph.add_node(IMAGE_COPY, ImageCopyDriver);

                graph.add_node_edge(IMAGE_COPY, bevy::render::main_graph::node::CAMERA_DRIVER);
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
                let gpu_images = world.get_resource::<RenderAssets<Image>>().unwrap();

                for image_copier in image_copiers.iter() {
                    if !image_copier.enabled() {
                        continue;
                    }

                    let src_image = gpu_images.get(&image_copier.src_image).unwrap();

                    let mut encoder = render_context
                        .render_device()
                        .create_command_encoder(&CommandEncoderDescriptor::default());

                    let block_dimensions = src_image.texture_format.block_dimensions();
                    let block_size = src_image.texture_format.block_size(None).unwrap();

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
    pub mod scene_tester {
        use std::path::PathBuf;

        use bevy::{
            app::AppExit,
            log::LogPlugin,
            prelude::*,
            render::{camera::RenderTarget, renderer::RenderDevice},
            window::ExitCondition,
        };
        use wgpu::{Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};

        // use image::{io::Reader, ImageBuffer, Rgba, ImageFormat};

        use super::image_copy::{ImageCopier, ImageCopyPlugin};


        #[derive(Component, Default)]
        pub struct CaptureCamera;

        #[derive(Component, Deref, DerefMut)]
        struct ImageToSave(Handle<Image>);

        pub struct SceneTesterPlugin;
        impl Plugin for SceneTesterPlugin {
            fn build(&self, app: &mut App) {
                app.add_plugins(
                    DefaultPlugins
                        .build()
                        .disable::<LogPlugin>()
                        .set(WindowPlugin {
                            primary_window: None,
                            exit_condition: ExitCondition::DontExit,
                            close_when_requested: false,
                        }),
                )
                .add_plugins(ImageCopyPlugin)
                .init_resource::<SceneController>()
                .add_event::<SceneController>()
                .add_systems(PostUpdate, update);
            }
        }

        #[derive(Debug, Resource, Event)]
        pub struct SceneController {
            state: SceneState,
            name: String,
            width: u32,
            height: u32,
        }

        impl SceneController {
            pub fn new(width:u32, height:u32) -> SceneController {
                SceneController {
                    state: SceneState::BuildScene,
                    name: String::from(""),
                    width,
                    height,
                }
            }
        }

        impl Default for SceneController {
            fn default() -> SceneController {
                SceneController {
                    state: SceneState::BuildScene,
                    name: String::from(""),
                    width: 1920,
                    height: 1080,
                }
            }
        }

        #[derive(Debug)]
        pub enum SceneState {
            BuildScene,
            Render(u32),
        }

        pub fn setup_test(
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
                if n > 0 {
                    scene_controller.state = SceneState::Render(n - 1);
                } else {
                    let x = images_to_save.iter().len();
                    println!("saving {} images", x);
                    for image in images_to_save.iter() {
                        let img_bytes = images.get_mut(image.id()).unwrap();

                        let img = match img_bytes.clone().try_into_dynamic() {
                            Ok(img) => img.to_rgba8(),
                            Err(e) => panic!("Failed to create image buffer {e:?}"),
                        };
                        
                        println!(
                            "\n After: {}x{} ({} channels)",
                            img.width(),
                            img.height(),
                            img.sample_layout().channels
                        );

                        let images_path =
                            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_images");
                        let uuid = bevy::utils::Uuid::new_v4();
                        let image_path = images_path.join(format!("{uuid}.png"));
                        if let Err(e) = img.save(image_path){
                            panic!("Failed to save image: {}", e);
                        };
                    }
                    app_exit_writer.send(AppExit);
                }
            }
        }
    }
}

// --------------------------------------

pub struct HeadlessGaussianSplatViewer {
    pub width: f32,
    pub height: f32,
}

impl Default for HeadlessGaussianSplatViewer {
    fn default() -> HeadlessGaussianSplatViewer {
        HeadlessGaussianSplatViewer {
            width: 1920.0,
            height: 1080.0,
        }
    }
}

fn setup_gaussian_cloud(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut gaussian_assets: ResMut<Assets<GaussianCloud>>,
    mut scene_controller: ResMut<frame_capture::scene_tester::SceneController>,
    mut images: ResMut<Assets<Image>>,
    render_device: Res<RenderDevice>,
) {
    let cloud: Handle<GaussianCloud>;

    // TODO: add proper GaussianSplattingViewer argument parsing
    let file_arg = get_arg(1);
    if let Some(n) = file_arg.clone().and_then(|s| s.parse::<usize>().ok()) {
        println!("generating {} gaussians", n);
        cloud = gaussian_assets.add(random_gaussians(n));
    } else if let Some(filename) = file_arg {
        if filename == "--help" {
            println!("usage: cargo run -- [filename | n]");
            return;
        }

        println!("loading {}", filename);
        cloud = asset_server.load(filename.to_string());
    } else {
        cloud = gaussian_assets.add(GaussianCloud::test_model());
    }

    let render_target = frame_capture::scene_tester::setup_test(
        &mut commands,
        &mut images,
        &render_device,
        &mut scene_controller,
        15,
        String::from("basic_cube_scene"),
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
        PanOrbitCamera {
            allow_upside_down: true,
            orbit_smoothness: 0.0,
            pan_smoothness: 0.0,
            zoom_smoothness: 0.0,
            ..default()
        },
    ));
}

#[cfg(feature = "morph_particles")]
fn setup_particle_behavior(
    mut commands: Commands,
    mut particle_behavior_assets: ResMut<Assets<ParticleBehaviors>>,
    gaussian_cloud: Query<(
        Entity,
        &Handle<GaussianCloud>,
        Without<Handle<ParticleBehaviors>>,
    )>,
) {
    if gaussian_cloud.is_empty() {
        return;
    }

    let mut particle_behaviors = None;

    let file_arg = get_arg(1);
    if let Some(_n) = file_arg.clone().and_then(|s| s.parse::<usize>().ok()) {
        let behavior_arg = get_arg(2);
        if let Some(k) = behavior_arg.clone().and_then(|s| s.parse::<usize>().ok()) {
            println!("generating {} particle behaviors", k);
            particle_behaviors = particle_behavior_assets
                .add(random_particle_behaviors(k))
                .into();
        }
    }

    if let Some(particle_behaviors) = particle_behaviors {
        commands
            .entity(gaussian_cloud.single().0)
            .insert(particle_behaviors);
    }
}

#[cfg(feature = "material_noise")]
fn setup_noise_material(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gaussian_clouds: Query<(Entity, &Handle<GaussianCloud>, Without<NoiseMaterial>)>,
) {
    if gaussian_clouds.is_empty() {
        return;
    }

    for (entity, cloud_handle, _) in gaussian_clouds.iter() {
        if Some(bevy::asset::LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        commands.entity(entity).insert(NoiseMaterial::default());
    }
}

#[cfg(feature = "query_select")]
fn press_i_invert_selection(
    keys: Res<Input<KeyCode>>,
    mut select_inverse_events: EventWriter<InvertSelectionEvent>,
) {
    if keys.just_pressed(KeyCode::I) {
        println!("inverting selection");
        select_inverse_events.send(InvertSelectionEvent);
    }
}

#[cfg(feature = "query_select")]
fn press_o_save_selection(
    keys: Res<Input<KeyCode>>,
    mut select_inverse_events: EventWriter<SaveSelectionEvent>,
) {
    if keys.just_pressed(KeyCode::O) {
        println!("saving selection");
        select_inverse_events.send(SaveSelectionEvent);
    }
}

#[cfg(feature = "query_sparse")]
fn setup_sparse_select(
    mut commands: Commands,
    gaussian_cloud: Query<(Entity, &Handle<GaussianCloud>, Without<SparseSelect>)>,
) {
    if gaussian_cloud.is_empty() {
        return;
    }

    commands
        .entity(gaussian_cloud.single().0)
        .insert(SparseSelect {
            completed: true,
            ..default()
        });
}

fn headless_app() {
    let config = HeadlessGaussianSplatViewer::default();
    let mut app = App::new();

    // setup for gaussian viewer app
    app.insert_resource(frame_capture::scene_tester::SceneController::new(config.width as u32, config.height as u32));
    app.insert_resource(ClearColor(Color::rgb_u8(0, 0, 0)));

    // app.add_plugins(
    //     DefaultPlugins
    //         .set(ImagePlugin::default_nearest())
    //         .build()
    //         .disable::<bevy::winit::WinitPlugin>(),
    // );
    // headless frame capture
    app.add_plugins(frame_capture::scene_tester::SceneTesterPlugin);

    app.add_plugins(ScheduleRunnerPlugin::run_loop(
        std::time::Duration::from_secs_f64(1.0 / 60.0),
    ));

    app.add_plugins(PanOrbitCameraPlugin);

    // setup for gaussian splatting
    app.add_plugins(GaussianSplattingPlugin);


    app.add_systems(Startup, setup_gaussian_cloud);

    #[cfg(feature = "material_noise")]
    app.add_systems(Update, setup_noise_material);

    #[cfg(feature = "morph_particles")]
    app.add_systems(Update, setup_particle_behavior);

    #[cfg(feature = "query_select")]
    {
        app.add_systems(Update, press_i_invert_selection);
        app.add_systems(Update, press_o_save_selection);
    }

    #[cfg(feature = "query_sparse")]
    app.add_systems(Update, setup_sparse_select);

    app.run();
}

pub fn main() {
    headless_app();
}
