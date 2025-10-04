use bevy::{
    app::AppExit, camera::Viewport, core_pipeline::tonemapping::Tonemapping, prelude::*, window::WindowResized
};
use bevy_args::{BevyArgsPlugin, parse_args};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, quick::WorldInspectorPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, GaussianCamera, GaussianMode, GaussianSplattingPlugin,
    PlanarGaussian3d, PlanarGaussian3dHandle, SphericalHarmonicCoefficients,
    gaussian::f32::Rotation,
    utils::{GaussianSplattingViewer, setup_hooks},
};

fn multi_camera_app() {
    let config = parse_args::<GaussianSplattingViewer>();
    let mut app = App::new();

    // setup for gaussian viewer app
    app.insert_resource(ClearColor(Color::srgb_u8(0, 0, 0)));
    app.add_plugins(
        DefaultPlugins
            .set(ImagePlugin::default_nearest())
            .set(WindowPlugin {
                primary_window: Some(Window {
                    mode: bevy::window::WindowMode::Windowed,
                    present_mode: bevy::window::PresentMode::AutoVsync,
                    prevent_default_event_handling: false,
                    resolution: bevy::window::WindowResolution::new(
                        config.width as u32,
                        config.height as u32
                    ),
                    title: config.name.clone(),
                    ..default()
                }),
                ..default()
            }),
    );
    app.add_plugins(BevyArgsPlugin::<GaussianSplattingViewer>::default());
    app.add_plugins(PanOrbitCameraPlugin);

    if config.editor {
        app.add_plugins(EguiPlugin::default());
        app.add_plugins(WorldInspectorPlugin::new());
    }

    if config.press_esc_close {
        app.add_systems(Update, esc_close);
    }

    app.add_plugins(GaussianSplattingPlugin);
    app.add_systems(Startup, setup_multi_camera);
    app.add_systems(Update, (press_s_to_spawn_camera, set_camera_viewports));

    app.run();
}

pub fn setup_multi_camera(
    mut commands: Commands,
    _asset_server: Res<AssetServer>,
    mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
) {
    let grid_size_x = 10;
    let grid_size_y = 10;
    let spacing = 12.0;

    // let mut blue_gaussians = Vec::new();
    // let mut blue_sh = SphericalHarmonicCoefficients::default();
    // blue_sh.set(2, 5.0);

    // for i in 0..grid_size_x {
    //     for j in 0..grid_size_y {
    //         let x = i as f32 * spacing - (grid_size_x as f32 * spacing) / 2.0;
    //         let y = j as f32 * spacing - (grid_size_y as f32 * spacing) / 2.0;
    //         let position = [x, y, 0.0, 1.0];
    //         let scale = [2.0, 1.0, 0.01, 0.5];

    //         let angle = std::f32::consts::PI / 2.0 * i as f32 / grid_size_x as f32;
    //         let rotation = Quat::from_rotation_z(angle).to_array();
    //         let rotation = [3usize, 0usize, 1usize, 2usize]
    //             .iter()
    //             .map(|i| rotation[*i])
    //             .collect::<Vec<_>>()
    //             .try_into()
    //             .unwrap();

    //         let gaussian = Gaussian {
    //             position_visibility: position.into(),
    //             rotation: Rotation {
    //                 rotation,
    //             },
    //             scale_opacity: scale.into(),
    //             spherical_harmonic: blue_sh,
    //         };
    //         blue_gaussians.push(gaussian);
    //     }
    // }

    // let cloud = asset_server.load("office.ply");
    // commands.spawn((
    //     GaussianSplattingBundle {
    //         cloud,//: gaussian_assets.add(blue_gaussians.into()),
    //         ..default()
    //     },
    //     Name::new("gaussian_cloud_3dgs"),
    // ));

    let mut red_gaussians = Vec::new();
    let mut red_sh = SphericalHarmonicCoefficients::default();
    red_sh.set(0, 5.0);

    for i in 0..grid_size_x {
        for j in 0..grid_size_y {
            let x = i as f32 * spacing - (grid_size_x as f32 * spacing) / 2.0;
            let y = j as f32 * spacing - (grid_size_y as f32 * spacing) / 2.0;
            let position = [x, y, 0.0, 1.0];
            let scale = [2.0, 1.0, 0.01, 0.5];

            let angle = std::f32::consts::PI / 2.0 * (i + 1) as f32 / grid_size_x as f32;
            let rotation = Quat::from_rotation_z(angle).to_array();
            let rotation = [3usize, 0usize, 1usize, 2usize]
                .iter()
                .map(|i| rotation[*i])
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            let gaussian = Gaussian3d {
                position_visibility: position.into(),
                rotation: Rotation { rotation },
                scale_opacity: scale.into(),
                spherical_harmonic: red_sh,
            };
            red_gaussians.push(gaussian);
        }
    }

    commands.spawn((
        Transform::from_translation(Vec3::new(spacing, spacing, 0.0)),
        PlanarGaussian3dHandle(gaussian_assets.add(red_gaussians)),
        CloudSettings {
            aabb: true,
            gaussian_mode: GaussianMode::Gaussian2d,
            ..default()
        },
        Name::new("gaussian_cloud_2dgs"),
    ));

    commands.spawn((
        GaussianCamera { warmup: true },
        Camera3d::default(),
        Camera {
            order: 0,
            ..default()
        },
        Transform::from_translation(Vec3::new(0.0, 1.5, 20.0)),
        Tonemapping::None,
        CameraPosition {
            pos: UVec2::new(0, 0),
        },
        PanOrbitCamera {
            allow_upside_down: true,
            ..default()
        },
    ));

    // commands.spawn((
    //     GaussianCamera,
    //     Camera3dBundle {
    //         camera: Camera{
    //             order: 1,
    //             ..default()
    //         },
    //         transform: Transform::from_translation(Vec3::new(0.0, 0.0, 40.0)),
    //         tonemapping: Tonemapping::None,
    //         ..default()
    //     },
    //     CameraPosition {
    //         pos: UVec2::new(1, 0),
    //     },
    //     PanOrbitCamera {
    //         allow_upside_down: true,
    //         ..default()
    //     },
    // ));
}

fn press_s_to_spawn_camera(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    windows: Query<&Window>,
) {
    if keys.just_pressed(KeyCode::KeyS) {
        let window = windows.single().unwrap();
        let size = window.physical_size() / UVec2::new(2, 1);
        let pos = UVec2::new(1, 0);

        commands.spawn((
            GaussianCamera { warmup: true },
            Camera3d::default(),
            Camera {
                order: 1,
                viewport: Viewport {
                    physical_position: pos * size,
                    physical_size: size,
                    ..default()
                }
                .into(),
                ..default()
            },
            Transform::from_translation(Vec3::new(0.0, 0.0, 40.0)),
            Tonemapping::None,
            CameraPosition { pos },
            PanOrbitCamera {
                allow_upside_down: true,
                ..default()
            },
        ));
    }
}

#[derive(Component)]
struct CameraPosition {
    pos: UVec2,
}

fn set_camera_viewports(
    windows: Query<&Window>,
    mut resize_events: MessageReader<WindowResized>,
    mut cameras: Query<(&CameraPosition, &mut Camera), With<GaussianCamera>>,
) {
    for resize_event in resize_events.read() {
        let window = windows.get(resize_event.window).unwrap();
        let size = window.physical_size() / UVec2::new(2, 1);

        for (position, mut camera) in &mut cameras {
            camera.viewport = Some(Viewport {
                physical_position: position.pos * size,
                physical_size: size,
                ..default()
            });
        }
    }
}

fn esc_close(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

pub fn main() {
    setup_hooks();
    multi_camera_app();
}
