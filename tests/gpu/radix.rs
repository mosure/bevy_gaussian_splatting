use bevy::{
    app::AppExit, core_pipeline::tonemapping::Tonemapping, diagnostic::FrameCount, prelude::*,
};

use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, PlanarGaussian3d, PlanarGaussian3dHandle, random_gaussians_3d,
};

use _harness::{TestHarness, test_harness_app};

mod _harness;

// run with `cargo run --bin test_radix --features "debug_gpu,sort_radix,testing"`
fn main() {
    let mut app = test_harness_app(TestHarness {
        resolution: (512, 512),
    });

    app.add_systems(Startup, setup);
    app.add_systems(Update, exit_after_warmup);

    app.run();
}

fn setup(mut commands: Commands, mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>) {
    let cloud = gaussian_assets.add(random_gaussians_3d(10_000));

    commands.spawn((
        PlanarGaussian3dHandle(cloud),
        CloudSettings::default(),
        Name::new("gaussian_cloud"),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
        Tonemapping::None,
        GaussianCamera::default(),
    ));
}

fn exit_after_warmup(mut exit: MessageWriter<AppExit>, frame_count: Res<FrameCount>) {
    const FRAMES_TO_RUN: u32 = 30;
    if frame_count.0 >= FRAMES_TO_RUN {
        exit.write(AppExit::Success);
    }
}
