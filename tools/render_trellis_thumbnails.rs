use std::{f32::consts::FRAC_PI_4, fs::File, io::BufReader, path::Path};

use bevy::{
    asset::RenderAssetUsages,
    image::Image,
    math::Vec3,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_gaussian_splatting::{
    PlanarGaussian3d, gaussian::interface::CommonCloud, io::ply::parse_ply_3d,
};
use bevy_interleave::prelude::Planar;

const WIDTH: u32 = 960;
const HEIGHT: u32 = 540;
const INPUT_PLY: &str = "assets/trellis.ply";

#[derive(Clone, Copy)]
enum ThumbnailMode {
    Position,
    Depth,
    Normal,
}

#[derive(Clone, Copy)]
struct ProjectedPoint {
    x: u32,
    y: u32,
    depth: f32,
    position: Vec3,
}

fn main() {
    let cloud = load_cloud(INPUT_PLY);
    let (projected, center, min, max, depth_min, depth_max) = project_points(&cloud, WIDTH, HEIGHT);

    render_and_save(
        "www/examples/thumbnails/seeded-scoop.png",
        ThumbnailMode::Position,
        &projected,
        center,
        min,
        max,
        depth_min,
        depth_max,
    );
    render_and_save(
        "www/examples/thumbnails/seeded-depth.png",
        ThumbnailMode::Depth,
        &projected,
        center,
        min,
        max,
        depth_min,
        depth_max,
    );
    render_and_save(
        "www/examples/thumbnails/seeded-normal.png",
        ThumbnailMode::Normal,
        &projected,
        center,
        min,
        max,
        depth_min,
        depth_max,
    );
}

fn load_cloud(path: &str) -> PlanarGaussian3d {
    let file = File::open(path)
        .unwrap_or_else(|err| panic!("failed to open thumbnail input {path}: {err}"));
    let mut reader = BufReader::new(file);
    parse_ply_3d(&mut reader)
        .unwrap_or_else(|err| panic!("failed to parse thumbnail input {path}: {err}"))
}

fn project_points(
    cloud: &PlanarGaussian3d,
    width: u32,
    height: u32,
) -> (Vec<ProjectedPoint>, Vec3, Vec3, Vec3, f32, f32) {
    let aabb = cloud
        .compute_aabb()
        .unwrap_or_else(|| panic!("cloud has no AABB for thumbnail projection"));
    let min = Vec3::from(aabb.min);
    let max = Vec3::from(aabb.max);
    let center = (min + max) * 0.5;
    let half_extents = (max - min) * 0.5;

    let aspect = width as f32 / height as f32;
    let tan_half_fov_y = (FRAC_PI_4 * 0.5).tan();
    let tan_half_fov_x = tan_half_fov_y * aspect;
    let radius = half_extents.length().max(0.1);
    let distance = (radius / tan_half_fov_y.min(tan_half_fov_x).max(1e-4)) * 1.15;

    let camera_dir = Vec3::new(0.6, 0.4, 1.0).normalize_or_zero();
    let camera_pos = center + camera_dir * distance;
    let forward = (center - camera_pos).normalize_or_zero();
    let right = forward.cross(Vec3::Y).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();

    let mut projected = Vec::with_capacity(cloud.len().min(600_000));
    let mut depth_min = f32::INFINITY;
    let mut depth_max = f32::NEG_INFINITY;

    for gaussian in cloud.iter() {
        if gaussian.position_visibility.visibility <= 0.0 || gaussian.scale_opacity.opacity <= 0.001
        {
            continue;
        }

        let position = Vec3::from(gaussian.position_visibility.position);
        let rel = position - camera_pos;

        let z = rel.dot(forward);
        if z <= 1e-4 {
            continue;
        }

        let x = rel.dot(right);
        let y = rel.dot(up);

        let ndc_x = x / (z * tan_half_fov_x.max(1e-4));
        let ndc_y = y / (z * tan_half_fov_y.max(1e-4));
        if ndc_x.abs() > 1.2 || ndc_y.abs() > 1.2 {
            continue;
        }

        let px = ((ndc_x * 0.5 + 0.5) * (width.saturating_sub(1)) as f32).round() as i32;
        let py = ((1.0 - (ndc_y * 0.5 + 0.5)) * (height.saturating_sub(1)) as f32).round() as i32;
        if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
            continue;
        }

        depth_min = depth_min.min(z);
        depth_max = depth_max.max(z);
        projected.push(ProjectedPoint {
            x: px as u32,
            y: py as u32,
            depth: z,
            position,
        });
    }

    if projected.is_empty() {
        panic!("no points projected for thumbnail rendering");
    }

    (projected, center, min, max, depth_min, depth_max)
}

#[allow(clippy::too_many_arguments)]
fn render_and_save(
    output_path: &str,
    mode: ThumbnailMode,
    projected: &[ProjectedPoint],
    center: Vec3,
    min: Vec3,
    max: Vec3,
    depth_min: f32,
    depth_max: f32,
) {
    let background = [10u8, 12u8, 16u8, 255u8];
    let mut pixels = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    for px in pixels.chunks_exact_mut(4) {
        px.copy_from_slice(&background);
    }

    let mut z_buffer = vec![f32::INFINITY; (WIDTH * HEIGHT) as usize];
    let extent = (max - min).max(Vec3::splat(1e-5));
    let depth_extent = (depth_max - depth_min).max(1e-5);

    for point in projected {
        let idx = (point.y * WIDTH + point.x) as usize;
        if point.depth >= z_buffer[idx] {
            continue;
        }
        z_buffer[idx] = point.depth;

        let color = match mode {
            ThumbnailMode::Position => {
                ((point.position - min) / extent).clamp(Vec3::ZERO, Vec3::ONE)
            }
            ThumbnailMode::Depth => {
                let t = ((point.depth - depth_min) / depth_extent).clamp(0.0, 1.0);
                Vec3::new(1.0 - t, (1.0 - (2.0 * (t - 0.5).abs())).max(0.0), t)
            }
            ThumbnailMode::Normal => {
                let n = (point.position - center).normalize_or_zero();
                n * 0.5 + Vec3::splat(0.5)
            }
        };

        let out = &mut pixels[idx * 4..idx * 4 + 4];
        out[0] = (color.x.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[1] = (color.y.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[2] = (color.z.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[3] = 255;
    }

    let non_background = pixels
        .chunks_exact(4)
        .filter(|px| px[0] != background[0] || px[1] != background[1] || px[2] != background[2])
        .count();
    if non_background == 0 {
        panic!("rendered thumbnail is empty for {output_path}");
    }

    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).expect("failed to create thumbnail directory");
    }

    let image = Image::new(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    let dynamic = image
        .try_into_dynamic()
        .expect("failed to convert generated thumbnail image");
    dynamic
        .save(output_path)
        .unwrap_or_else(|err| panic!("failed to save thumbnail {output_path}: {err}"));
}
