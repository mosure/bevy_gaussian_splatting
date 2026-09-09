//! Calibrated camera paths exported by standard 3DGS tools.
//!
//! Input rotations are row-major camera-to-world matrices in the OpenCV
//! right/down/forward convention. Positions and scene coordinates are unchanged;
//! Bevy's right/up/back camera basis is `R * diag(1, -1, -1)`.

use std::{collections::HashSet, fmt};

use bevy::{
    camera::{CameraProjection, Projection, SubCameraView},
    math::Vec3A,
    prelude::*,
};
use serde::Deserialize;

/// Validated frames in file order. Frame indices and authored IDs are distinct.
#[derive(Clone, Debug)]
pub struct GaussianCameraPath {
    frames: Vec<GaussianCameraPathFrame>,
}

#[derive(Clone, Debug)]
pub struct GaussianCameraPathFrame {
    id: u64,
    image_name: String,
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    transform: Transform,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceCamera {
    id: u64,
    #[serde(default)]
    img_name: String,
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    position: [f32; 3],
    rotation: [[f32; 3]; 3],
}

#[derive(Debug)]
pub struct GaussianCameraPathError(String);

impl fmt::Display for GaussianCameraPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for GaussianCameraPathError {}

impl GaussianCameraPath {
    pub fn from_json(bytes: &[u8]) -> Result<Self, GaussianCameraPathError> {
        if bytes.len() > 64 * 1024 * 1024 {
            return Err(GaussianCameraPathError("camera path exceeds 64 MiB".into()));
        }
        let source: Vec<SourceCamera> = serde_json::from_slice(bytes)
            .map_err(|error| GaussianCameraPathError(error.to_string()))?;
        if source.is_empty() || source.len() > 100_000 {
            return Err(GaussianCameraPathError(
                "camera path requires 1..=100000 frames".into(),
            ));
        }
        let mut ids = HashSet::with_capacity(source.len());
        let frames = source
            .into_iter()
            .enumerate()
            .map(|(index, source)| {
                let invalid = || {
                    GaussianCameraPathError(format!(
                        "camera frame {index} has invalid intrinsics, pose, or duplicate ID"
                    ))
                };
                let r = source.rotation;
                let rotation = Mat3::from_cols(
                    Vec3::new(r[0][0], r[1][0], r[2][0]),
                    Vec3::new(r[0][1], r[1][1], r[2][1]),
                    Vec3::new(r[0][2], r[1][2], r[2][2]),
                );
                let gram = rotation.transpose() * rotation;
                let x_scale = 2.0 * source.fx / source.width as f32;
                let y_scale = 2.0 * source.fy / source.height as f32;
                if source.width == 0
                    || source.height == 0
                    || !source.fx.is_finite()
                    || source.fx <= 0.0
                    || !source.fy.is_finite()
                    || source.fy <= 0.0
                    || !x_scale.is_finite()
                    || x_scale <= 0.0
                    || !y_scale.is_finite()
                    || y_scale <= 0.0
                    || !Vec3::from_array(source.position).is_finite()
                    || !rotation.is_finite()
                    || (rotation.determinant() - 1.0).abs() > 1e-4
                    || gram
                        .to_cols_array()
                        .into_iter()
                        .zip(Mat3::IDENTITY.to_cols_array())
                        .any(|(a, b)| (a - b).abs() > 1e-4)
                    || !ids.insert(source.id)
                {
                    return Err(invalid());
                }
                let bevy_rotation =
                    Mat3::from_cols(rotation.x_axis, -rotation.y_axis, -rotation.z_axis);
                Ok(GaussianCameraPathFrame {
                    id: source.id,
                    image_name: source.img_name,
                    width: source.width,
                    height: source.height,
                    fx: source.fx,
                    fy: source.fy,
                    transform: Transform {
                        translation: Vec3::from_array(source.position),
                        rotation: Quat::from_mat3(&bevy_rotation).normalize(),
                        scale: Vec3::ONE,
                    },
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { frames })
    }

    pub fn frames(&self) -> &[GaussianCameraPathFrame] {
        &self.frames
    }
}

impl GaussianCameraPathFrame {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn image_name(&self) -> &str {
        &self.image_name
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn transform(&self) -> Transform {
        self.transform
    }

    pub fn projection(&self, near: f32, far: f32) -> Result<Projection, GaussianCameraPathError> {
        if !near.is_finite() || near <= 0.0 || !far.is_finite() || far <= near {
            return Err(GaussianCameraPathError(
                "camera projection requires 0 < near < far".into(),
            ));
        }
        Ok(Projection::custom(GaussianCameraIntrinsics {
            x_scale: 2.0 * self.fx / self.width as f32,
            y_scale: 2.0 * self.fy / self.height as f32,
            near,
            far,
        }))
    }
}

/// Centered pinhole intrinsics, resized with the full source image.
///
/// Independent focal lengths survive Bevy viewport updates. Resizing scales
/// pixel intrinsics independently in X/Y; use the source image aspect ratio
/// when an anisotropic image resize is undesirable. No lens distortion or
/// unrecorded principal-point offset is inferred.
#[derive(Clone, Debug)]
pub struct GaussianCameraIntrinsics {
    x_scale: f32,
    y_scale: f32,
    near: f32,
    far: f32,
}

impl GaussianCameraIntrinsics {
    pub fn vertical_fov_radians(&self) -> f32 {
        2.0 * self.y_scale.recip().atan()
    }
    pub fn near(&self) -> f32 {
        self.near
    }
}

impl CameraProjection for GaussianCameraIntrinsics {
    fn get_clip_from_view(&self) -> Mat4 {
        Mat4::from_cols(
            Vec4::new(self.x_scale, 0.0, 0.0, 0.0),
            Vec4::new(0.0, self.y_scale, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, self.near, 0.0),
        )
    }

    fn get_clip_from_view_for_sub(&self, sub: &SubCameraView) -> Mat4 {
        let full = sub.full_size.as_vec2();
        let size = sub.size.as_vec2();
        let scale = full / size;
        let offset = (full - 2.0 * sub.offset - size) / size;
        let mut matrix = self.get_clip_from_view();
        matrix.x_axis.x *= scale.x;
        matrix.y_axis.y *= scale.y;
        matrix.z_axis.x = -offset.x;
        matrix.z_axis.y = offset.y;
        matrix
    }

    fn update(&mut self, _width: f32, _height: f32) {}
    fn far(&self) -> f32 {
        self.far
    }

    fn get_frustum_corners(&self, z_near: f32, z_far: f32) -> [Vec3A; 8] {
        std::array::from_fn(|index| {
            let z = if index < 4 { z_near } else { z_far };
            let x = z.abs() / self.x_scale;
            let y = z.abs() / self.y_scale;
            [
                Vec3A::new(x, -y, z),
                Vec3A::new(x, y, z),
                Vec3A::new(-x, y, z),
                Vec3A::new(-x, -y, z),
            ][index % 4]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        serde_json::json!([{"id":7,"img_name":"rolled","width":640,"height":480,
            "fx":500,"fy":450,"position":[4,-8,12],
            "rotation":[[0,-1,0],[0,0,1],[-1,0,0]]}])
    }

    fn pixels(projection: &Projection, transform: Transform, world: Vec3, size: Vec2) -> Vec2 {
        let clip =
            projection.get_clip_from_view() * transform.to_matrix().inverse() * world.extend(1.0);
        (clip.truncate().truncate() / clip.w * Vec2::new(0.5, -0.5) + Vec2::splat(0.5)) * size
    }

    #[test]
    fn standard_3dgs_camera_preserves_roll_and_independent_focal_lengths() {
        let path = GaussianCameraPath::from_json(&serde_json::to_vec(&fixture()).unwrap()).unwrap();
        let frame = &path.frames()[0];
        assert_eq!(frame.id(), 7);
        assert_eq!(frame.image_name(), "rolled");
        let transform = frame.transform();
        assert!(transform.forward().as_vec3().abs_diff_eq(Vec3::Y, 1e-6));
        assert!(transform.up().as_vec3().abs_diff_eq(Vec3::X, 1e-6));
        let mut projection = frame.projection(0.1, 1000.0).unwrap();
        // OpenCV local (0.4,-0.2,4): u=320+500*.4/4, v=240+450*(-.2)/4.
        let world = Vec3::new(4.2, -4.0, 11.6);
        assert!(
            pixels(&projection, transform, world, Vec2::new(640.0, 480.0))
                .abs_diff_eq(Vec2::new(370.0, 217.5), 1e-3)
        );
        projection.update(320.0, 240.0);
        assert!(
            pixels(&projection, transform, world, Vec2::new(320.0, 240.0))
                .abs_diff_eq(Vec2::new(185.0, 108.75), 1e-3)
        );
        let sub = SubCameraView {
            full_size: UVec2::new(640, 480),
            offset: Vec2::new(100.0, 80.0),
            size: UVec2::new(320, 240),
        };
        let clip = projection.get_clip_from_view_for_sub(&sub)
            * transform.to_matrix().inverse()
            * world.extend(1.0);
        let pixel = (clip.truncate().truncate() / clip.w * Vec2::new(0.5, -0.5) + Vec2::splat(0.5))
            * sub.size.as_vec2();
        assert!(pixel.abs_diff_eq(Vec2::new(270.0, 137.5), 1e-3));
    }

    #[test]
    fn standard_3dgs_camera_rejects_invalid_calibration_and_pose() {
        for (field, value) in [
            ("fx", serde_json::json!(0)),
            ("width", serde_json::json!(0)),
            (
                "rotation",
                serde_json::json!([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            ),
        ] {
            let mut fixture = fixture();
            fixture[0][field] = value;
            assert!(GaussianCameraPath::from_json(&serde_json::to_vec(&fixture).unwrap()).is_err());
        }
        let mut duplicate = fixture();
        duplicate.as_array_mut().unwrap().push(fixture()[0].clone());
        assert!(GaussianCameraPath::from_json(&serde_json::to_vec(&duplicate).unwrap()).is_err());
        assert!(GaussianCameraPath::from_json(b"[]").is_err());
    }
}
