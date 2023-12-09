use std::sync::{
    Arc,
    Mutex,
};

use bevy::{
    prelude::*,
    app::AppExit,
    core::FrameCount,
    render::view::screenshot::ScreenshotManager,
    window::PrimaryWindow,
};

use bevy_gaussian_splatting::{
    GaussianCloud,
    GaussianSplattingBundle,
    random_gaussians,
};

use _harness::{
    TestHarness,
    test_harness_app,
    TestStateArc,
};

mod _harness;


fn main() {
    let mut app = test_harness_app(TestHarness {
        resolution: (512.0, 512.0),
    });

    app.add_systems(Startup, setup);
    app.add_systems(Update, verify_radix_stages);

    app.run();
}

fn setup(
    mut commands: Commands,
    mut gaussian_assets: ResMut<Assets<GaussianCloud>>,
) {
    let cloud = gaussian_assets.add(random_gaussians(10000));

    commands.spawn((
        GaussianSplattingBundle {
            cloud,
            ..default()
        },
        Name::new("gaussian_cloud"),
    ));

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
            ..default()
        },
    ));
}


fn verify_radix_stages(
    mut exit: EventWriter<AppExit>,
    frame_count: Res<FrameCount>,
    state: Local<TestStateArc>,
) {

}
