//! Bounded geometric navigation aid, not semantic terrain detection. A dominant
//! plane must have broad 2D support without a similarly supported competing tilt.

use bevy::{
    math::DVec3,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, futures::check_ready},
};
use bevy_gaussian_splatting::{PlanarGaussian3d, PlanarGaussian3dHandle};
use rand::{Rng, SeedableRng, rngs::StdRng};

pub(super) const MAX_GROUND_SAMPLES: usize = 4096;
const MIN_GROUND_SAMPLES: usize = 96;
const MIN_INLIER_FRACTION: f64 = 0.35;
const HYPOTHESES: usize = 128;

#[derive(Clone, Copy, Debug, Resource)]
pub(super) struct GroundPlaneEstimate {
    /// Unit normal, signed toward the initial camera when its up is tangent.
    pub normal: Vec3,
    pub point: Vec3,
    pub inliers: usize,
    pub samples: usize,
    pub inlier_fraction: f32,
    pub rms_distance: f32,
}

type SourceKey = (Entity, bevy::asset::AssetId<PlanarGaussian3d>, Mat4);

#[derive(Resource, Default)]
struct GroundPlaneWork {
    source: Option<SourceKey>,
    task: Option<Task<Option<GroundPlaneEstimate>>>,
    attempts: u32,
    next_sample_seconds: f64,
    finished: bool,
}

pub(super) struct GroundPlanePlugin;

impl Plugin for GroundPlanePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GroundPlaneWork>()
            .add_systems(Update, estimate_scene_ground);
    }
}

#[allow(clippy::too_many_arguments)]
fn estimate_scene_ground(
    mut commands: Commands,
    mut work: ResMut<GroundPlaneWork>,
    time: Res<Time>,
    clouds: Query<(Entity, &PlanarGaussian3dHandle, &GlobalTransform)>,
    cameras: Query<&Transform, With<super::ViewerMainCamera>>,
    assets: Res<Assets<PlanarGaussian3d>>,
    mut asset_events: MessageReader<AssetEvent<PlanarGaussian3d>>,
    #[cfg(feature = "lod")] atlases: Option<
        Res<bevy_gaussian_splatting::stream::atlas_upload::LodTransientAtlasRegistry>,
    >,
) {
    let Some((entity, handle, transform)) = clouds.iter().min_by_key(|(entity, _, _)| *entity)
    else {
        asset_events.clear();
        if work.source.is_some() {
            commands.remove_resource::<GroundPlaneEstimate>();
            *work = GroundPlaneWork::default();
        }
        return;
    };
    let key = (entity, handle.0.id(), transform.to_matrix());
    let source_changed = asset_events.read().fold(false, |changed, event| {
        changed
            | matches!(event, AssetEvent::Modified { id } | AssetEvent::Removed { id } if *id == key.1)
    });
    if work.source != Some(key) || source_changed {
        commands.remove_resource::<GroundPlaneEstimate>();
        *work = GroundPlaneWork {
            source: Some(key),
            ..default()
        };
    }
    // Liveness must precede task publication, independently of the expensive
    // sampling cadence. A same-handle reload may temporarily remove its asset.
    #[cfg(feature = "lod")]
    let resident_live = atlases.as_ref().is_some_and(|atlases| {
        atlases
            .sample_positions(handle.0.id(), 0)
            .ok()
            .flatten()
            .is_some()
    });
    #[cfg(not(feature = "lod"))]
    let resident_live = false;
    let cloud = assets.get(&handle.0);
    if !resident_live && cloud.is_none() {
        commands.remove_resource::<GroundPlaneEstimate>();
        *work = GroundPlaneWork {
            source: Some(key),
            ..default()
        };
        return;
    }
    if let Some(result) = work.task.as_mut().and_then(check_ready) {
        work.task = None;
        if let Some(estimate) = result {
            info!(
                "Ground-plane navigation estimate: normal {:?}, point {:?}, {}/{} inliers ({:.1}%), RMS distance {:.4}; frozen for this scene",
                estimate.normal,
                estimate.point,
                estimate.inliers,
                estimate.samples,
                estimate.inlier_fraction * 100.0,
                estimate.rms_distance
            );
            commands.insert_resource(estimate);
            work.finished = true;
        } else if work.attempts == 8 {
            work.finished = true;
            warn!(
                "Ground-plane estimate is ambiguous or inadequately supported after 8 bounded attempts; retaining authored navigation orientation"
            );
        }
    }
    if work.finished
        || work.task.is_some()
        || work.attempts >= 8
        || time.elapsed_secs_f64() < work.next_sample_seconds
    {
        return;
    }
    work.next_sample_seconds = time.elapsed_secs_f64() + 2.0;
    #[cfg(feature = "lod")]
    let resident = atlases.and_then(|atlases| {
        atlases
            .sample_positions(handle.0.id(), MAX_GROUND_SAMPLES)
            .ok()
            .flatten()
    });
    #[cfg(not(feature = "lod"))]
    let resident: Option<Vec<Vec3>> = None;
    let Ok(camera) = cameras.single() else { return };
    let points = if let Some(points) = resident {
        // Do not infer a large scene's up from a handful of startup slots.
        if points.len() < 512 {
            return;
        }
        points
    } else {
        let Some(cloud) = cloud else { return };
        let count = cloud.position_visibility.len().min(MAX_GROUND_SAMPLES);
        (0..count)
            .filter_map(|sample| {
                let index = (sample as u64 * cloud.position_visibility.len() as u64 / count as u64)
                    as usize;
                let position = &cloud.position_visibility[index];
                let point = Vec3::from(position.position);
                (position.visibility > 0.0
                    && cloud.scale_opacity[index].opacity > 0.0
                    && point.is_finite())
                .then_some(point)
            })
            .collect()
    };
    if points.len() < MIN_GROUND_SAMPLES {
        return;
    }
    let points: Vec<_> = points
        .into_iter()
        .map(|point| transform.transform_point(point))
        .collect();
    let up = camera.up().as_vec3();
    let position = camera.translation;
    work.attempts += 1;
    work.task = Some(
        AsyncComputeTaskPool::get()
            .spawn(async move { estimate_ground_plane(&points, up, position) }),
    );
}

/// Uses at most 4096 finite positions, with no full-scene scans or I/O. The
/// caller freezes a successful estimate rather than changing navigation as
/// streamed residency changes. A rejected estimate leaves the prior unchanged.
pub(super) fn estimate_ground_plane(
    points: &[Vec3],
    up_prior: Vec3,
    camera_position: Vec3,
) -> Option<GroundPlaneEstimate> {
    if points.len() < MIN_GROUND_SAMPLES || points.len() > MAX_GROUND_SAMPLES {
        return None;
    }
    let up = up_prior.try_normalize()?.as_dvec3();
    let points: Vec<DVec3> = points
        .iter()
        .filter(|p| p.is_finite())
        .map(|p| p.as_dvec3())
        .collect();
    if points.len() < MIN_GROUND_SAMPLES {
        return None;
    }
    // Coordinate medians and a median radius keep remote outliers from setting
    // the residual tolerance. f64 centered arithmetic handles translated scenes.
    let median = DVec3::from_array(std::array::from_fn(|axis| {
        let mut values: Vec<f64> = points.iter().map(|p| p[axis]).collect();
        *values
            .select_nth_unstable_by(points.len() / 2, f64::total_cmp)
            .1
    }));
    let mut distances: Vec<f64> = points.iter().map(|p| p.distance(median)).collect();
    let extent = *distances
        .select_nth_unstable_by(points.len() / 2, f64::total_cmp)
        .1;
    if !extent.is_finite() || extent <= 1e-8 {
        return None;
    }
    let tolerance = extent * 0.01;
    let mut rng = StdRng::seed_from_u64(0x6772_6f75_6e64);
    let mut best = None;
    let mut best_count = 0;
    let mut best_error = f64::INFINITY;
    let mut candidates = Vec::with_capacity(HYPOTHESES);
    for _ in 0..HYPOTHESES {
        let a = points[rng.random_range(0..points.len())];
        let b = points[rng.random_range(0..points.len())];
        let c = points[rng.random_range(0..points.len())];
        let Some(mut normal) = (b - a).cross(c - a).try_normalize() else {
            continue;
        };
        if normal.dot(up) < 0.0 {
            normal = -normal;
        }
        let (count, error) = support(&points, a, normal, tolerance);
        candidates.push((normal, count));
        if count > best_count || (count == best_count && error < best_error) {
            best = Some((a, normal));
            best_count = count;
            best_error = error;
        }
    }
    if best_count < MIN_GROUND_SAMPLES
        || best_count as f64 / (points.len() as f64) < MIN_INLIER_FRACTION
    {
        return None;
    }
    let (origin, normal) = best?;
    // Parallel floors/roofs share the same navigation up. Distinct strong tilts
    // make automatic ground identification ambiguous, so leave navigation alone.
    if candidates.iter().any(|(candidate, count)| {
        candidate.dot(normal).abs() < 0.966 && *count as f64 >= best_count as f64 * 0.85
    }) {
        return None;
    }
    let inliers: Vec<DVec3> = points
        .iter()
        .copied()
        .filter(|p| (*p - origin).dot(normal).abs() <= tolerance)
        .collect();
    let point = inliers.iter().copied().sum::<DVec3>() / inliers.len() as f64;
    let axis = normal;
    let u = axis.any_orthonormal_vector();
    let v = axis.cross(u);
    let (mut uu, mut uv, mut vv, mut uh, mut vh) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for p in &inliers {
        let delta = *p - point;
        let (x, y, h) = (delta.dot(u), delta.dot(v), delta.dot(axis));
        uu += x * x;
        uv += x * y;
        vv += y * y;
        uh += x * h;
        vh += y * h;
    }
    let determinant = uu * vv - uv * uv;
    // Reject lines and tiny local patches, even if RANSAC finds many duplicates.
    let trace = uu + vv;
    if determinant <= trace * trace * 0.001
        || trace / (inliers.len() as f64) < extent * extent * 0.01
    {
        return None;
    }
    let a = (uh * vv - vh * uv) / determinant;
    let b = (vh * uu - uh * uv) / determinant;
    let mut normal = (axis - u * a - v * b).try_normalize()?;
    let camera_side = normal.dot(camera_position.as_dvec3() - point);
    let sign = if camera_side.abs() > tolerance * 2.0 {
        camera_side
    } else {
        normal.dot(up)
    };
    if sign < 0.0 {
        normal = -normal;
    }
    let (count, error) = support(&points, point, normal, tolerance);
    let fraction = count as f64 / points.len() as f64;
    if count < MIN_GROUND_SAMPLES || fraction < MIN_INLIER_FRACTION {
        return None;
    }
    Some(GroundPlaneEstimate {
        normal: normal.as_vec3(),
        point: point.as_vec3(),
        inliers: count,
        samples: points.len(),
        inlier_fraction: fraction as f32,
        rms_distance: (error / count as f64).sqrt() as f32,
    })
}

fn support(points: &[DVec3], origin: DVec3, normal: DVec3, tolerance: f64) -> (usize, f64) {
    points.iter().fold((0, 0.0), |(count, error), point| {
        let distance = (*point - origin).dot(normal);
        if distance.abs() <= tolerance {
            (count + 1, error + distance * distance)
        } else {
            (count, error)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_estimate_invalidates_on_same_handle_changes_and_missing_source() {
        let mut app = App::new();
        app.init_resource::<Time>()
            .init_resource::<Assets<PlanarGaussian3d>>()
            .add_message::<AssetEvent<PlanarGaussian3d>>()
            .add_plugins(GroundPlanePlugin);
        let handle = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::default());
        let source = app
            .world_mut()
            .spawn((
                PlanarGaussian3dHandle(handle.clone()),
                GlobalTransform::IDENTITY,
            ))
            .id();
        app.world_mut()
            .spawn((super::super::ViewerMainCamera, Transform::IDENTITY));
        let estimate = GroundPlaneEstimate {
            normal: Vec3::Y,
            point: Vec3::ZERO,
            inliers: 96,
            samples: 96,
            inlier_fraction: 1.0,
            rms_distance: 0.0,
        };
        app.insert_resource(estimate)
            .insert_resource(GroundPlaneWork {
                source: Some((source, handle.id(), Mat4::IDENTITY)),
                finished: true,
                next_sample_seconds: 100.0,
                ..default()
            });
        app.update();
        assert!(app.world().contains_resource::<GroundPlaneEstimate>());
        app.world_mut()
            .write_message(AssetEvent::<PlanarGaussian3d>::Modified { id: handle.id() });
        app.update();
        assert!(!app.world().contains_resource::<GroundPlaneEstimate>());
        assert!(!app.world().resource::<GroundPlaneWork>().finished);

        app.insert_resource(estimate);
        app.world_mut().resource_mut::<GroundPlaneWork>().finished = true;
        app.world_mut()
            .resource_mut::<GroundPlaneWork>()
            .next_sample_seconds = 100.0;
        app.world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .remove(handle.id());
        // No asset event or elapsed cadence is needed for a missing source.
        app.update();
        assert!(!app.world().contains_resource::<GroundPlaneEstimate>());
        assert!(!app.world().resource::<GroundPlaneWork>().finished);

        app.world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .insert(handle.id(), PlanarGaussian3d::default())
            .unwrap();
        app.insert_resource(estimate);
        app.world_mut().resource_mut::<GroundPlaneWork>().finished = true;
        *app.world_mut().get_mut::<GlobalTransform>(source).unwrap() =
            GlobalTransform::from_translation(Vec3::X);
        app.update();
        assert!(!app.world().contains_resource::<GroundPlaneEstimate>());
        assert_eq!(
            app.world().resource::<GroundPlaneWork>().source.unwrap().2,
            Mat4::from_translation(Vec3::X)
        );
    }

    #[test]
    #[ignore = "requires the bounded local Poland position sample"]
    fn poland_bounded_ground_sample_matches_navigation_up() {
        #[derive(serde::Deserialize)]
        struct Samples {
            positions: Vec<[f32; 3]>,
            camera_position: [f32; 3],
        }
        let path = std::env::var("BGS_POLAND_GROUND_SAMPLE").expect("set BGS_POLAND_GROUND_SAMPLE");
        let sample: Samples = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let positions: Vec<_> = sample.positions.into_iter().map(Vec3::from_array).collect();
        let estimate = estimate_ground_plane(
            &positions,
            Vec3::Y,
            Vec3::from_array(sample.camera_position),
        )
        .expect("Poland sample must support an unambiguous plane");
        assert!(estimate.normal.dot(Vec3::NEG_Y) > 0.99);
        println!(
            "{}",
            serde_json::json!({
                "normal": estimate.normal.to_array(),
                "point": estimate.point.to_array(),
                "inliers": estimate.inliers,
                "samples": estimate.samples,
                "inlier_fraction": estimate.inlier_fraction,
                "rms_distance": estimate.rms_distance,
            })
        );
    }

    #[test]
    fn tilted_ground_survives_outliers_and_world_translation_with_stable_sign() {
        let rotation = Quat::from_rotation_z(0.35) * Quat::from_rotation_x(-0.2);
        let expected = rotation * Vec3::Y;
        let offset = Vec3::new(12000.0, -8000.0, 24000.0);
        let mut points = Vec::new();
        for x in -10..=10 {
            for z in -10..=10 {
                let noise = ((x * 7 + z * 11) as f32).sin() * 0.01;
                points.push(offset + rotation * Vec3::new(x as f32, noise, z as f32));
            }
        }
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..250 {
            points.push(
                offset
                    + Vec3::new(
                        rng.random_range(-50.0..50.0),
                        rng.random_range(5.0..50.0),
                        rng.random_range(-50.0..50.0),
                    ),
            );
        }
        let fit = estimate_ground_plane(&points, Vec3::Y, offset + Vec3::Y * 100.0).unwrap();
        assert!(fit.normal.dot(expected) > 0.999);
        assert!(fit.inlier_fraction > 0.6);
        assert!(fit.rms_distance < 0.03);
        let reversed = estimate_ground_plane(&points, -Vec3::Y, offset - Vec3::Y * 100.0).unwrap();
        assert!(fit.normal.dot(reversed.normal) < -0.999);
        assert!((fit.point - offset).dot(expected).abs() < 0.05);
        assert_eq!(fit.samples, points.len());
        assert!(fit.inliers >= 441);
    }

    #[test]
    fn handles_top_down_axes_but_rejects_ambiguous_planes_and_lines() {
        let line: Vec<Vec3> = (0..256).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect();
        assert!(estimate_ground_plane(&line, Vec3::Y, Vec3::Y * 10.0).is_none());
        let wall: Vec<Vec3> = (0..256)
            .map(|i| Vec3::new(0.0, (i / 16) as f32, (i % 16) as f32))
            .collect();
        let fit = estimate_ground_plane(&wall, Vec3::Y, Vec3::X * 10.0).unwrap();
        assert!(
            fit.normal.dot(Vec3::X) > 0.999,
            "camera-up may be tangent to a Z/X-up model's ground"
        );
        let mut ambiguous = wall.clone();
        ambiguous.extend((0..256).map(|i| Vec3::new((i / 16) as f32, 0.0, (i % 16) as f32)));
        assert!(estimate_ground_plane(&ambiguous, Vec3::Y, Vec3::splat(10.0)).is_none());
        assert!(estimate_ground_plane(&wall[..32], Vec3::Y, Vec3::ZERO).is_none());
        assert!(estimate_ground_plane(&wall, Vec3::ZERO, Vec3::ZERO).is_none());
    }
}
