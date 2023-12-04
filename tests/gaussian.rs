use std::sync::Arc;

use bevy::{
    prelude::*,
    app::AppExit,
    asset::LoadState,
    render::view::screenshot::ScreenshotManager,
    window::PrimaryWindow,
};

use bevy_gaussian_splatting::{
    GaussianCloud,
    GaussianSplattingBundle,
    io::codec::GaussianCloudCodec,
    random_gaussians,
};

use _harness::{
    TestHarness,
    test_harness_app,
    TestStateArc,
};

mod _harness;


#[test]
fn test_codec() {
    let count = 100;

    let gaussians = random_gaussians(count);
    let encoded = gaussians.encode();
    let decoded = GaussianCloud::decode(encoded.as_slice());

    assert_eq!(gaussians, decoded);
}



#[test]
fn test_basic_rendering() {
    let mut app = test_harness_app(TestHarness {
        resolution: (512.0, 512.0),
    });

    app.add_systems(Startup, setup);
    fn setup(
        mut commands: Commands,
        mut gaussian_assets: ResMut<Assets<GaussianCloud>>,
    ) {
        let cloud = gaussian_assets.add(random_gaussians(1000));

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

    app.add_systems(Update, capture_ready);
    fn capture_ready(
        gaussian_cloud_assets: Res<Assets<GaussianCloud>>,
        asset_server: Res<AssetServer>,
        gaussian_clouds: Query<
            Entity,
            &Handle<GaussianCloud>,
        >,
        main_window: Query<Entity, With<PrimaryWindow>>,
        mut screenshot_manager: ResMut<ScreenshotManager>,
        mut exit: EventWriter<AppExit>,
        state: Local<TestStateArc>,
    ) {
        let state_clone = Arc::clone(&state);

        let state = state.lock().unwrap();
        if state.test_completed {
            // exit.send(AppExit);
            return;
        }

        if let Ok(window_entity) = main_window.get_single() {
            screenshot_manager.save_screenshot_to_disk(window_entity, "tests/gaussian_splatting.png");
            let mut state = state_clone.lock().unwrap();
            state.test_completed = true;

            // screenshot_manager.take_screenshot(window_entity, move |image: Image| {
            //     let has_non_zero_data = image.data.iter().fold(false, |non_zero, &x| non_zero || x != 0);
            //     assert!(has_non_zero_data, "screenshot is all zeros");

            //     let mut state = state_clone.lock().unwrap();
            //     state.test_completed = true;
            // }).unwrap();
        }
    }


    app.run();
}

