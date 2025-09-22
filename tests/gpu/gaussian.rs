use std::sync::{Arc, Mutex};

use bevy::{
    app::AppExit, core::FrameCount, core_pipeline::tonemapping::Tonemapping, prelude::*,
    render::view::screenshot::ScreenshotManager, window::PrimaryWindow,
};

use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, PlanarGaussian3d, PlanarGaussian3dHandle, random_gaussians_3d,
};

use _harness::{TestHarness, TestStateArc, test_harness_app};

mod _harness;

// run with `cargo run --bin test_gaussian`
fn main() {
    let mut app = test_harness_app(TestHarness {
        resolution: (512.0, 512.0),
    });

    app.add_systems(Startup, setup);
    app.add_systems(Update, capture_ready);

    app.run();
}

fn setup(mut commands: Commands, mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>) {
    let cloud = gaussian_assets.add(random_gaussians_3d(10000));

    commands.spawn((
        PlanarGaussian3dHandle(cloud),
        CloudSettings::default(),
        Name::new("gaussian_cloud"),
    ));

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
            tonemapping: Tonemapping::None,
            ..default()
        },
        GaussianCamera,
    ));
}

fn check_image_equality(image: &Image, other: &Image) -> bool {
    if image.width() != other.width() || image.height() != other.height() {
        return false;
    }

    for (word, other_word) in image.data.iter().zip(other.data.iter()) {
        if word != other_word {
            return false;
        }
    }

    true
}

fn test_stability(captures: Arc<Mutex<Vec<Image>>>) {
    let all_frames_similar = captures
        .lock()
        .unwrap()
        .iter()
        .try_fold(None, |acc, image| match acc {
            Some(acc_image) => {
                if check_image_equality(acc_image, image) {
                    Some(Some(acc_image))
                } else {
                    None
                }
            }
            None => Some(Some(image)),
        })
        .is_some();
    assert!(all_frames_similar, "all frames are not the same");
}

fn save_captures(captures: Arc<Mutex<Vec<Image>>>) {
    captures
        .lock()
        .unwrap()
        .iter()
        .enumerate()
        .for_each(|(i, image)| {
            let path = format!("target/tmp/test_gaussian_frame_{}.png", i);

            let dyn_img = image.clone().try_into_dynamic().unwrap();
            let img = dyn_img.to_rgba8();
            img.save(path).unwrap();
        });
}

fn capture_ready(
    // gaussian_cloud_assets: Res<Assets<PlanarGaussian3d>>,
    // asset_server: Res<AssetServer>,
    // gaussian_clouds: Query<
    //     Entity,
    //     &PlanarGaussian3dHandle,
    // >,
    main_window: Query<Entity, With<PrimaryWindow>>,
    mut screenshot_manager: ResMut<ScreenshotManager>,
    mut exit: EventWriter<AppExit>,
    frame_count: Res<FrameCount>,
    state: Local<TestStateArc>,
    buffer: Local<Arc<Mutex<Vec<Image>>>>,
) {
    let buffer = buffer.to_owned();

    let buffer_frames = 10;
    let wait_frames = 10; // wait for gaussian cloud to load
    if frame_count.0 < wait_frames {
        return;
    }

    let state_clone = Arc::clone(&state);
    let buffer_clone = Arc::clone(&buffer);

    let mut state = state.lock().unwrap();
    state.test_loaded = true;

    if state.test_completed {
        {
            let captures = buffer.lock().unwrap();
            let frame_count = captures.len();
            assert_eq!(
                frame_count, buffer_frames,
                "captured {} frames, expected {}",
                frame_count, buffer_frames
            );
        }

        save_captures(buffer.clone());
        test_stability(buffer);
        // TODO: add correctness test (use CPU gaussian pipeline to compare results)

        exit.write(AppExit);
        return;
    }

    if let Ok(window_entity) = main_window.get_single() {
        screenshot_manager
            .take_screenshot(window_entity, move |image: Image| {
                let has_non_zero_data = image.data.iter().any(|&x| x != 0);
                assert!(has_non_zero_data, "screenshot is all zeros");

                let mut buffer = buffer_clone.lock().unwrap();
                buffer.push(image);

                if buffer.len() >= buffer_frames {
                    let mut state = state_clone.lock().unwrap();
                    state.test_completed = true;
                }
            })
            .unwrap();
    }
}
