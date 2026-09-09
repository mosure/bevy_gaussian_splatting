//! The upstream flycam plugin drives a non-rendered motion entity. Applying its
//! relative motion only after input admission preserves calibrated camera roll
//! and projection, including when upstream consumes uncaptured mouse events.

use bevy::{
    input::InputSystems,
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use bevy_flycam::{FlyCam, KeyBindings, MovementSettings, NoCameraPlayerPlugin};
use bevy_inspector_egui::bevy_egui::{EguiPostUpdateSet, input::EguiWantsInput};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraSystemSet};

#[derive(Component)]
pub(super) struct FlyCamera;

#[derive(Component, Default)]
struct FlyMotion {
    admitted: bool,
    logged_takeover: bool,
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub(super) struct FlyCameraCommit;

pub(super) struct ViewerFlyCameraPlugin {
    pub speed: f32,
    pub sensitivity: f32,
}

impl Plugin for ViewerFlyCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(NoCameraPlayerPlugin)
            .insert_resource(MovementSettings {
                speed: self.speed,
                sensitivity: self.sensitivity,
            })
            .insert_resource(KeyBindings {
                // Escape remains the viewer's exit shortcut.
                toggle_grab_cursor: KeyCode::Tab,
                ..default()
            })
            .add_systems(Startup, spawn_motion)
            .add_systems(PostStartup, release_cursor)
            .add_systems(PreUpdate, prepare_motion.after(InputSystems))
            .add_systems(
                PostUpdate,
                commit_motion
                    .in_set(FlyCameraCommit)
                    .after(EguiPostUpdateSet::ProcessOutput)
                    .after(PanOrbitCameraSystemSet)
                    .before(bevy::transform::TransformSystems::Propagate)
                    .before(bevy::camera::CameraUpdateSystems),
            );
        info!(
            "Flycam: Tab captures/releases the mouse; mouse looks, WASD moves, Space/Left Shift move up/down; Home resets a camera path; F12 saves a screenshot; Esc exits (speed {} world units/s)",
            self.speed
        );
    }
}

fn spawn_motion(mut commands: Commands) {
    commands.spawn((FlyCam, FlyMotion::default(), Transform::IDENTITY));
}

fn release_cursor(mut windows: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    if let Ok(mut cursor) = windows.single_mut() {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
}

fn pointer_available(window: &Window, blocked: bool) -> bool {
    window.focused && window.cursor_position().is_some() && !blocked
}

fn prepare_motion(
    mut motion: Query<(&mut Transform, &mut FlyMotion)>,
    mut windows: Query<(&Window, &mut CursorOptions), With<PrimaryWindow>>,
    keys: Res<ButtonInput<KeyCode>>,
    egui: Option<Res<EguiWantsInput>>,
) {
    let Ok((mut transform, mut motion)) = motion.single_mut() else {
        return;
    };
    *transform = Transform::IDENTITY;
    motion.admitted = false;
    let Ok((window, mut cursor)) = windows.single_mut() else {
        return;
    };
    if !pointer_available(window, egui.is_some_and(|input| input.wants_any_input()))
        || keys.just_pressed(KeyCode::Home)
    {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
        return;
    }
    motion.admitted = cursor.grab_mode != CursorGrabMode::None && !keys.just_pressed(KeyCode::Tab);
}

fn apply_motion(camera: &mut Transform, motion: &Transform, ground_up: Option<Vec3>) -> bool {
    if motion.translation == Vec3::ZERO && motion.rotation == Quat::IDENTITY {
        return false;
    }
    if let Some(up) = ground_up.and_then(Vec3::try_normalize) {
        apply_ground_motion(camera, motion, up);
        return true;
    }
    camera.translation += camera.rotation * motion.translation;
    camera.rotation = (camera.rotation * motion.rotation).normalize();
    true
}

// Ground matching changes the navigation frame, not the model or its calibrated
// projection. Reconstructing from a fixed up axis removes roll accumulation and
// keeps WASD at constant ground-relative height even while looking up or down.
fn apply_ground_motion(camera: &mut Transform, motion: &Transform, up: Vec3) {
    let forward = camera.forward().as_vec3();
    let horizontal = forward - up * forward.dot(up);
    let flat_forward = if horizontal.length_squared() > 1e-8 {
        horizontal.normalize()
    } else {
        let right = camera.right().as_vec3();
        let right = (right - up * right.dot(up))
            .try_normalize()
            .unwrap_or_else(|| up.any_orthonormal_vector());
        up.cross(right)
    };
    let (yaw_delta, pitch_delta, _) = motion.rotation.to_euler(EulerRot::YXZ);
    // Upstream movement may already include this frame's look delta. Map its
    // translation in the pre-look basis so yaw is never applied twice.
    let right = flat_forward.cross(up).normalize();
    camera.translation += right * motion.translation.x + up * motion.translation.y
        - flat_forward * motion.translation.z;

    // Match upstream flycam's pitch limit, expressed about the estimated ground.
    let flat_forward = Quat::from_axis_angle(up, yaw_delta) * flat_forward;
    let pitch = (forward.dot(up).clamp(-1.0, 1.0).asin() + pitch_delta).clamp(-1.54, 1.54);
    camera.look_to(flat_forward * pitch.cos() + up * pitch.sin(), up);
}

#[allow(clippy::too_many_arguments)]
fn commit_motion(
    mut commands: Commands,
    mut motion: Query<(&Transform, &mut FlyMotion), Without<FlyCamera>>,
    mut cameras: Query<(Entity, &mut Transform), With<FlyCamera>>,
    mut windows: Query<(&Window, &mut CursorOptions), With<PrimaryWindow>>,
    egui: Option<Res<EguiWantsInput>>,
    ground: Option<Res<super::ground_plane::GroundPlaneEstimate>>,
    #[cfg(not(target_arch = "wasm32"))] path: Option<ResMut<super::ViewerCameraPath>>,
) {
    let Ok((delta, mut motion)) = motion.single_mut() else {
        return;
    };
    let Ok((window, mut cursor)) = windows.single_mut() else {
        return;
    };
    if !pointer_available(window, egui.is_some_and(|input| input.wants_any_input())) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
        return;
    }
    if !motion.admitted {
        return;
    }
    let Ok((camera_entity, mut camera)) = cameras.single_mut() else {
        return;
    };
    if apply_motion(
        &mut camera,
        delta,
        ground.as_ref().map(|plane| plane.normal),
    ) {
        // Automatic framing may use the disabled orbit controller until first
        // input, but must never overwrite the user's subsequent flycam pose.
        commands.entity(camera_entity).remove::<PanOrbitCamera>();
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(mut path) = path {
            path.frames_per_second = 0.0;
        }
        if !motion.logged_takeover {
            info!("Manual flycam navigation took over; camera calibration retained");
            motion.logged_takeover = true;
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use bevy::input::{InputPlugin, mouse::MouseMotion};

    #[test]
    fn matched_flycam_tracks_tilted_ground_without_roll_drift_or_height_change() {
        let up = Quat::from_euler(EulerRot::XYZ, 0.7, 0.0, 0.4) * Vec3::Y;
        let forward = up.any_orthonormal_vector();
        let mut camera =
            Transform::from_xyz(10.0, 20.0, 30.0).looking_to(forward * 0.8 - up * 0.6, up);
        camera.rotation *= Quat::from_rotation_z(0.7);
        let original = camera;
        assert!(!apply_motion(&mut camera, &Transform::IDENTITY, Some(up)));
        assert_eq!(
            camera, original,
            "matching must not alter a held authored pose"
        );

        let movement =
            Transform::from_xyz(0.0, 0.0, -2.0).with_rotation(Quat::from_rotation_y(0.4));
        assert!(apply_motion(&mut camera, &movement, Some(up)));
        let displacement = camera.translation - original.translation;
        assert!(displacement.dot(up).abs() < 1e-5);
        assert!((displacement.length() - 2.0).abs() < 1e-5);
        assert!(displacement.abs_diff_eq(forward * 2.0, 1e-5));
        assert!(camera.right().dot(up).abs() < 1e-5);
        assert!(camera.up().dot(up) > 0.0);

        // If upstream look runs before movement, translation already contains
        // yaw. The adapter must preserve that one rotation in the ground basis.
        let yaw = Quat::from_rotation_y(0.4);
        let mut looked_camera = original;
        apply_motion(
            &mut looked_camera,
            &Transform::from_translation(yaw * Vec3::NEG_Z * 2.0).with_rotation(yaw),
            Some(up),
        );
        let expected = Quat::from_axis_angle(up, 0.4) * forward * 2.0;
        assert!((looked_camera.translation - original.translation).abs_diff_eq(expected, 1e-5));

        let before_rise = camera.translation;
        apply_motion(&mut camera, &Transform::from_xyz(0.0, 3.0, 0.0), Some(up));
        assert!((camera.translation - before_rise).abs_diff_eq(up * 3.0, 1e-5));
        apply_motion(
            &mut camera,
            &Transform::from_rotation(Quat::from_rotation_x(-1.4)),
            Some(up),
        );
        assert!(camera.forward().dot(up).abs() <= 1.54_f32.sin() + 1e-6);
        assert!(camera.right().dot(up).abs() < 1e-5);
    }

    #[test]
    fn upstream_flycam_moves_only_on_admitted_input_and_retains_calibration() {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            InputPlugin,
            ViewerFlyCameraPlugin {
                speed: 20.0,
                sensitivity: 0.00012,
            },
        ));
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            std::time::Duration::from_millis(16),
        ));
        let mut window = Window {
            focused: true,
            ..default()
        };
        window.set_cursor_position(Some(Vec2::new(100.0, 100.0)));
        let window = app
            .world_mut()
            .spawn((window, CursorOptions::default(), PrimaryWindow))
            .id();
        let initial = Transform::from_xyz(10.0, 20.0, 30.0).with_rotation(Quat::from_euler(
            EulerRot::YXZ,
            0.7,
            -0.3,
            1.1,
        ));
        let path = bevy_gaussian_splatting::camera::path::GaussianCameraPath::from_json(br#"[{"id":0,"width":960,"height":640,"fx":700,"fy":900,"position":[10,20,30],"rotation":[[1,0,0],[0,1,0],[0,0,1]]}]"#).unwrap();
        let projection = path.frames()[0].projection(0.1, 10000.0).unwrap();
        app.insert_resource(super::super::ViewerCameraPath {
            path,
            first: 0,
            current: 0,
            frames_per_second: 30.0,
            elapsed: 0.0,
        });
        let clip = projection.get_clip_from_view();
        let camera = app.world_mut().spawn((FlyCamera, initial, projection)).id();
        app.update();
        app.world_mut().write_message(MouseMotion {
            delta: Vec2::new(25.0, 15.0),
        });
        app.update();
        assert_eq!(*app.world().get::<Transform>(camera).unwrap(), initial);
        assert_eq!(
            app.world()
                .resource::<super::super::ViewerCameraPath>()
                .frames_per_second,
            30.0
        );
        app.world_mut()
            .get_mut::<CursorOptions>(window)
            .unwrap()
            .grab_mode = CursorGrabMode::Confined;
        app.world_mut().get_mut::<Window>(window).unwrap().focused = false;
        app.world_mut().write_message(MouseMotion {
            delta: Vec2::new(25.0, 15.0),
        });
        app.update();
        assert_eq!(*app.world().get::<Transform>(camera).unwrap(), initial);
        app.world_mut().get_mut::<Window>(window).unwrap().focused = true;
        app.world_mut()
            .get_mut::<CursorOptions>(window)
            .unwrap()
            .grab_mode = CursorGrabMode::Confined;
        app.world_mut().write_message(MouseMotion {
            delta: Vec2::new(25.0, 15.0),
        });
        app.update();
        assert_ne!(
            app.world().get::<Transform>(camera).unwrap().rotation,
            initial.rotation
        );
        assert_eq!(
            app.world()
                .get::<Projection>(camera)
                .unwrap()
                .get_clip_from_view(),
            clip
        );
        assert_eq!(
            app.world().get::<Transform>(camera).unwrap().translation,
            initial.translation
        );
        assert_eq!(
            app.world()
                .resource::<super::super::ViewerCameraPath>()
                .frames_per_second,
            0.0
        );
        let before_move = *app.world().get::<Transform>(camera).unwrap();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyW);
        app.update();
        let moved =
            app.world().get::<Transform>(camera).unwrap().translation - before_move.translation;
        assert!((moved.length() - 0.32).abs() < 1e-4);
        assert!(
            moved
                .normalize()
                .abs_diff_eq(before_move.forward().as_vec3(), 1e-4)
        );
    }
}
