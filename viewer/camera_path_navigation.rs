//! Navigation for calibrated path cameras. PanOrbit cannot pan custom projections
//! and rebuilds rotations without authored roll, so these cameras use local axes.

use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
    prelude::*,
    window::PrimaryWindow,
};
use bevy_inspector_egui::bevy_egui::input::EguiWantsInput;

use super::{
    VIEWER_CAMERA_MAX_ORBIT_RADIUS, VIEWER_CAMERA_VISIBILITY_FAR, ViewerCameraPath,
    ViewerMainCamera,
};

#[derive(Component)]
pub(super) struct PathCameraNavigation {
    radius: f32,
    initial_radius: f32,
    manual: bool,
}

impl PathCameraNavigation {
    pub(super) fn new(path: &ViewerCameraPath) -> Self {
        // Scale navigation to the authored route, independent of its world origin.
        let (min, max) = path.path.frames().iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(min, max), frame| {
                let position = frame.transform().translation;
                (min.min(position), max.max(position))
            },
        );
        let radius = ((max - min).length() * 0.02).clamp(1.0, VIEWER_CAMERA_MAX_ORBIT_RADIUS);
        info!(
            "Calibrated camera {}: navigation stops playback; Home restores the starting pose and holds it",
            path.current
        );
        Self {
            radius,
            initial_radius: radius,
            manual: false,
        }
    }

    fn apply(
        &mut self,
        transform: &mut Transform,
        projection: &Projection,
        viewport: Vec2,
        orbit: Vec2,
        pan: Vec2,
        scroll: f32,
    ) -> bool {
        if orbit == Vec2::ZERO && pan == Vec2::ZERO && scroll == 0.0 {
            return false;
        }
        if orbit != Vec2::ZERO {
            let focus = transform.translation + transform.forward() * self.radius;
            let angles = orbit / viewport * Vec2::new(std::f32::consts::TAU, std::f32::consts::PI);
            transform.rotation = (transform.rotation
                * Quat::from_rotation_y(-angles.x)
                * Quat::from_rotation_x(-angles.y))
            .normalize();
            transform.translation = focus - transform.forward() * self.radius;
        }
        if pan != Vec2::ZERO {
            // Independent focal lengths: world displacement subtends exactly the
            // requested logical pixels at the orbit focus depth.
            let clip = projection.get_clip_from_view();
            let focal = Vec2::new(clip.x_axis.x, clip.y_axis.y) * viewport * 0.5;
            let displacement = pan / focal * self.radius;
            let translation = -transform.right() * displacement.x + transform.up() * displacement.y;
            transform.translation += translation;
        }
        if scroll != 0.0 {
            let next =
                (self.radius * (-0.2 * scroll).exp()).clamp(0.05, VIEWER_CAMERA_MAX_ORBIT_RADIUS);
            let translation = transform.back() * (next - self.radius);
            transform.translation += translation;
            self.radius = next;
        }
        true
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn navigate(
    path: Option<ResMut<ViewerCameraPath>>,
    mut cameras: Query<
        (&mut Transform, &mut Projection, &mut PathCameraNavigation),
        With<ViewerMainCamera>,
    >,
    windows: Query<&Window, With<PrimaryWindow>>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    egui_input: Option<Res<EguiWantsInput>>,
    args: Option<Res<super::GaussianSplattingViewer>>,
) {
    let Some(mut path) = path else { return };
    let Ok(window) = windows.single() else { return };
    if !window.focused || window.cursor_position().is_none() {
        return;
    }
    let Ok((mut transform, mut projection, mut navigation)) = cameras.single_mut() else {
        return;
    };
    if keys.just_pressed(KeyCode::Home)
        && !egui_input
            .as_ref()
            .is_some_and(|input| input.wants_any_keyboard_input())
    {
        path.current = path.first;
        path.elapsed = 0.0;
        path.frames_per_second = 0.0;
        let frame = &path.path.frames()[path.first];
        *transform = frame.transform();
        *projection = frame.projection(0.1, VIEWER_CAMERA_VISIBILITY_FAR).unwrap();
        navigation.radius = navigation.initial_radius;
        navigation.manual = false;
        info!("Calibrated camera restored to frame {} (held)", path.first);
        return;
    }
    if egui_input.is_some_and(|input| input.wants_any_pointer_input()) {
        return;
    }
    if args.is_some_and(|args| args.camera_controller == super::CameraController::Flycam) {
        return;
    }
    let viewport = Vec2::new(window.width(), window.height());
    if viewport.min_element() <= 0.0 {
        return;
    }
    let orbit = if buttons.pressed(MouseButton::Left) {
        motion.delta
    } else {
        Vec2::ZERO
    };
    let pan = if buttons.pressed(MouseButton::Right) {
        motion.delta
    } else {
        Vec2::ZERO
    };
    let scroll = scroll.delta.y
        * match scroll.unit {
            MouseScrollUnit::Line => 1.0,
            MouseScrollUnit::Pixel => 0.01,
        };
    if navigation.apply(&mut transform, &projection, viewport, orbit, pan, scroll) {
        if !navigation.manual {
            info!(
                "Manual camera navigation took over at path frame {}",
                path.current
            );
            navigation.manual = true;
        }
        path.frames_per_second = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_gaussian_splatting::camera::path::GaussianCameraPath;

    fn path() -> ViewerCameraPath {
        ViewerCameraPath {
            path: GaussianCameraPath::from_json(br#"[{"id":0,"width":960,"height":640,"fx":700,"fy":900,"position":[10,20,30],"rotation":[[0,-1,0],[1,0,0],[0,0,1]]}]"#).unwrap(),
            first: 0, current: 0, frames_per_second: 30.0, elapsed: 0.0,
        }
    }

    #[test]
    fn calibrated_path_navigation_preserves_idle_pose_and_projection() {
        let path = path();
        let mut navigation = PathCameraNavigation::new(&path);
        let frame = &path.path.frames()[0];
        let original = frame.transform();
        let mut transform = original;
        let projection = frame.projection(0.1, VIEWER_CAMERA_VISIBILITY_FAR).unwrap();
        let clip = projection.get_clip_from_view();
        let viewport = Vec2::new(960.0, 640.0);
        assert!(!navigation.apply(
            &mut transform,
            &projection,
            viewport,
            Vec2::ZERO,
            Vec2::ZERO,
            0.0
        ));
        assert_eq!(transform, original);
        assert!(navigation.apply(
            &mut transform,
            &projection,
            viewport,
            Vec2::ZERO,
            Vec2::new(70.0, 90.0),
            0.0
        ));
        let expected = original.translation - original.right() * 0.1 + original.up() * 0.1;
        assert!(transform.translation.abs_diff_eq(expected, 1e-5));
        assert_eq!(transform.rotation, original.rotation);
        assert!(navigation.apply(
            &mut transform,
            &projection,
            viewport,
            Vec2::new(10.0, 20.0),
            Vec2::ZERO,
            1.0
        ));
        assert_ne!(transform.rotation, original.rotation);
        assert!(transform.is_finite());
        assert_eq!(projection.get_clip_from_view(), clip);
    }

    #[test]
    fn calibrated_path_navigation_takes_over_only_focused_movement_and_resets() {
        let path = path();
        let frame = &path.path.frames()[0];
        let original = frame.transform();
        let projection = frame.projection(0.1, VIEWER_CAMERA_VISIBILITY_FAR).unwrap();
        let clip = projection.get_clip_from_view();
        let navigation = PathCameraNavigation::new(&path);
        let mut app = App::new();
        app.insert_resource(path)
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<AccumulatedMouseMotion>()
            .init_resource::<AccumulatedMouseScroll>()
            .add_systems(Update, navigate);
        let camera = app
            .world_mut()
            .spawn((ViewerMainCamera, original, projection, navigation))
            .id();
        let mut window = Window {
            focused: true,
            ..default()
        };
        window.set_cursor_position(Some(Vec2::new(100.0, 100.0)));
        let window = app.world_mut().spawn((window, PrimaryWindow)).id();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        assert_eq!(
            app.world().resource::<ViewerCameraPath>().frames_per_second,
            30.0
        );
        assert_eq!(*app.world().get::<Transform>(camera).unwrap(), original);
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::new(20.0, 10.0);
        app.world_mut().get_mut::<Window>(window).unwrap().focused = false;
        app.update();
        assert_eq!(*app.world().get::<Transform>(camera).unwrap(), original);
        assert_eq!(
            app.world().resource::<ViewerCameraPath>().frames_per_second,
            30.0
        );
        app.world_mut().get_mut::<Window>(window).unwrap().focused = true;
        app.update();
        assert_ne!(*app.world().get::<Transform>(camera).unwrap(), original);
        assert_eq!(
            app.world().resource::<ViewerCameraPath>().frames_per_second,
            0.0
        );
        assert_eq!(
            app.world()
                .get::<Projection>(camera)
                .unwrap()
                .get_clip_from_view(),
            clip
        );
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Home);
        app.update();
        assert_eq!(*app.world().get::<Transform>(camera).unwrap(), original);
        assert_eq!(app.world().resource::<ViewerCameraPath>().elapsed, 0.0);
        assert_eq!(
            app.world().resource::<ViewerCameraPath>().frames_per_second,
            0.0
        );
    }
}
