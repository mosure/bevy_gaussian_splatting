use bevy::{app::AppExit, core_pipeline::tonemapping::Tonemapping, prelude::*};
use bevy_args::{BevyArgsPlugin, parse_args};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, quick::WorldInspectorPlugin};
use bevy_interleave::prelude::Planar;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, GaussianCamera, GaussianMode, GaussianSplattingPlugin,
    PlanarGaussian3d, PlanarGaussian3dHandle, SphericalHarmonicCoefficients,
    gaussian::f32::Rotation,
    utils::{GaussianSplattingViewer, setup_hooks},
};

pub fn setup_surfel_compare(
    mut commands: Commands,
    mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
) {
    let grid_size_x = 10;
    let grid_size_y = 10;
    let spacing = 12.0;
    let visualize_bounding_box = false;

    let mut blue_gaussians = Vec::new();
    let mut blue_sh = SphericalHarmonicCoefficients::default();
    blue_sh.set(2, 5.0);

    for i in 0..grid_size_x {
        for j in 0..grid_size_y {
            let x = i as f32 * spacing - (grid_size_x as f32 * spacing) / 2.0;
            let y = j as f32 * spacing - (grid_size_y as f32 * spacing) / 2.0;
            let position = [x, y, 0.0, 1.0];
            let scale = [2.0, 1.0, 0.01, 0.5];

            let angle = std::f32::consts::PI / 2.0 * i as f32 / grid_size_x as f32;
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
                spherical_harmonic: blue_sh,
            };
            blue_gaussians.push(gaussian);
        }
    }

    let cloud = gaussian_assets.add(PlanarGaussian3d::from_interleaved(blue_gaussians));
    commands.spawn((
        PlanarGaussian3dHandle(cloud),
        CloudSettings {
            visualize_bounding_box,
            ..default()
        },
        Name::new("gaussian_cloud_3dgs"),
    ));

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

    let cloud = gaussian_assets.add(PlanarGaussian3d::from_interleaved(red_gaussians));
    commands.spawn((
        Transform::from_translation(Vec3::new(spacing, spacing, 0.0)),
        PlanarGaussian3dHandle(cloud),
        CloudSettings {
            visualize_bounding_box,
            aabb: true,
            gaussian_mode: GaussianMode::Gaussian2d,
            ..default()
        },
        Name::new("gaussian_cloud_2dgs"),
    ));

    let camera_transform = Transform::from_translation(Vec3::new(0.0, 1.5, 20.0));
    info!(
        "camera_transform: {:?}",
        camera_transform.compute_matrix().to_cols_array_2d()
    );

    commands.spawn((
        camera_transform,
        Camera3d::default(),
        Tonemapping::None,
        PanOrbitCamera {
            allow_upside_down: true,
            ..default()
        },
        GaussianCamera::default(),
    ));
}

fn compare_surfel_app() {
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
                    resolution: (config.width, config.height).into(),
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

    // setup for gaussian splatting
    app.add_plugins(GaussianSplattingPlugin);
    app.add_systems(Startup, setup_surfel_compare);

    app.run();
}

pub fn esc_close(keys: Res<ButtonInput<KeyCode>>, mut exit: EventWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

pub fn main() {
    setup_hooks();
    compare_surfel_app();
}
