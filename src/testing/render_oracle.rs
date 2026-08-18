//! Small deterministic CPU render oracle for LoD quality regression tests.
//!
//! This is intentionally not a production renderer. It provides a stable, linear-color image
//! comparison that exercises hierarchy cuts and representative Gaussians on every CI adapter.

use std::{collections::BTreeMap, fmt};

use bevy::prelude::Vec3;

use crate::{
    gaussian::{
        formats::{
            planar_3d::Gaussian3d,
            planar_3d_chunked::{LodNodeId, LodPageId},
            planar_3d_lod::PlanarGaussian3dLod,
        },
        lod_settings::GaussianLodSettings,
    },
    stream::hierarchy::{AllResident, LodView, ManifestLodHierarchy, select_frontier},
    testing::{
        image_metrics::{ImageMetrics, ImageMetricsError, compare_linear_rgba},
        lod_scenes::{LodProjection, LodTestCamera},
    },
};

#[derive(Clone, Debug, PartialEq)]
pub struct LodQualitySample {
    pub quality: f32,
    pub selected_nodes: usize,
    pub active_gaussians: usize,
    pub metrics: ImageMetrics,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LodRenderOracleError {
    InvalidImageSize,
    InvalidCamera(&'static str),
    InvalidManifest(String),
    Selection(String),
    MissingNode(LodNodeId),
    MissingPage(LodPageId),
    PageRangeOutOfBounds(LodNodeId),
    ImageMetrics(ImageMetricsError),
}

impl fmt::Display for LodRenderOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for LodRenderOracleError {}

/// Resolve a selected frontier into its exact replacement Gaussian records.
pub fn gather_frontier_gaussians(
    lod: &PlanarGaussian3dLod,
    frontier: &[LodNodeId],
) -> Result<Vec<Gaussian3d>, LodRenderOracleError> {
    let nodes = lod
        .manifest
        .nodes
        .iter()
        .map(|node| (node.id, node))
        .collect::<BTreeMap<_, _>>();
    let pages = lod
        .pages
        .iter()
        .map(|page| (page.id, page))
        .collect::<BTreeMap<_, _>>();
    let mut result = Vec::new();
    for &node_id in frontier {
        let node = nodes
            .get(&node_id)
            .ok_or(LodRenderOracleError::MissingNode(node_id))?;
        let page = pages
            .get(&node.representation.page)
            .ok_or(LodRenderOracleError::MissingPage(node.representation.page))?;
        let start = node.representation.offset as usize;
        let end = node
            .representation
            .end()
            .ok_or(LodRenderOracleError::PageRangeOutOfBounds(node_id))? as usize;
        let slice = page
            .gaussians
            .get(start..end)
            .ok_or(LodRenderOracleError::PageRangeOutOfBounds(node_id))?;
        result.extend_from_slice(slice);
    }
    Ok(result)
}

/// Render a deterministic linear premultiplied-RGBA image using a simple projected Gaussian
/// footprint. Gaussians are composited back-to-front, matching the production ordering contract.
pub fn render_linear_gaussians(
    gaussians: &[Gaussian3d],
    camera: LodTestCamera,
    width: u32,
    height: u32,
) -> Result<Vec<[f32; 4]>, LodRenderOracleError> {
    if width == 0 || height == 0 {
        return Err(LodRenderOracleError::InvalidImageSize);
    }
    let forward = (camera.target - camera.position).normalize_or_zero();
    let right = forward.cross(camera.up).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    if forward == Vec3::ZERO || right == Vec3::ZERO || up == Vec3::ZERO {
        return Err(LodRenderOracleError::InvalidCamera("basis"));
    }
    let aspect = width as f32 / height as f32;
    let projection = match camera.projection {
        LodProjection::Perspective {
            vertical_fov_radians,
        } if vertical_fov_radians.is_finite()
            && vertical_fov_radians > 0.0
            && vertical_fov_radians < std::f32::consts::PI =>
        {
            Projection::Perspective {
                focal_y: 0.5 * height as f32 / (0.5 * vertical_fov_radians).tan(),
            }
        }
        LodProjection::Orthographic {
            vertical_world_size,
        } if vertical_world_size.is_finite() && vertical_world_size > 0.0 => {
            Projection::Orthographic {
                pixels_per_world: height as f32 / vertical_world_size,
            }
        }
        _ => return Err(LodRenderOracleError::InvalidCamera("projection")),
    };

    let mut projected = Vec::with_capacity(gaussians.len());
    for gaussian in gaussians {
        if gaussian.position_visibility.visibility <= 0.0 || gaussian.scale_opacity.opacity <= 0.0 {
            continue;
        }
        let world = Vec3::from_array(gaussian.position_visibility.position);
        let relative = world - camera.position;
        let depth = relative.dot(forward);
        if !depth.is_finite() || depth < camera.near || depth > camera.far {
            continue;
        }
        let view_x = relative.dot(right);
        let view_y = relative.dot(up);
        let max_scale = gaussian
            .scale_opacity
            .scale
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0, f32::max)
            .max(1e-5);
        let (pixel_x, pixel_y, sigma_px) = match projection {
            Projection::Perspective { focal_y } => {
                let focal_x = focal_y / aspect * (width as f32 / height as f32);
                (
                    width as f32 * 0.5 + view_x * focal_x / depth,
                    height as f32 * 0.5 - view_y * focal_y / depth,
                    max_scale * focal_y / depth,
                )
            }
            Projection::Orthographic { pixels_per_world } => (
                width as f32 * 0.5 + view_x * pixels_per_world,
                height as f32 * 0.5 - view_y * pixels_per_world,
                max_scale * pixels_per_world,
            ),
        };
        if !pixel_x.is_finite() || !pixel_y.is_finite() || !sigma_px.is_finite() {
            continue;
        }
        let coefficients = &gaussian.spherical_harmonic.coefficients;
        let color = [
            coefficients.first().copied().unwrap_or(0.0).clamp(0.0, 1.0),
            coefficients.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0),
            coefficients.get(2).copied().unwrap_or(0.0).clamp(0.0, 1.0),
        ];
        projected.push(ProjectedGaussian {
            depth,
            pixel_x,
            pixel_y,
            sigma_px: sigma_px.max(0.35),
            color,
            opacity: (gaussian.scale_opacity.opacity * gaussian.position_visibility.visibility)
                .clamp(0.0, 0.999),
        });
    }
    projected.sort_by(|left, right| right.depth.total_cmp(&left.depth));

    let mut image = vec![[0.0; 4]; (width * height) as usize];
    for gaussian in projected {
        let radius = (gaussian.sigma_px * 3.0).ceil() as i32;
        let min_x = (gaussian.pixel_x.floor() as i32 - radius).max(0);
        let max_x = (gaussian.pixel_x.ceil() as i32 + radius).min(width as i32 - 1);
        let min_y = (gaussian.pixel_y.floor() as i32 - radius).max(0);
        let max_y = (gaussian.pixel_y.ceil() as i32 + radius).min(height as i32 - 1);
        if min_x > max_x || min_y > max_y {
            continue;
        }
        let inverse_two_sigma_squared = 0.5 / gaussian.sigma_px.powi(2);
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 + 0.5 - gaussian.pixel_x;
                let dy = y as f32 + 0.5 - gaussian.pixel_y;
                let alpha = (gaussian.opacity
                    * (-(dx * dx + dy * dy) * inverse_two_sigma_squared).exp())
                .clamp(0.0, 0.999);
                if alpha < 1.0 / 1024.0 {
                    continue;
                }
                let pixel = &mut image[(y as u32 * width + x as u32) as usize];
                for (channel, color) in gaussian.color.into_iter().enumerate() {
                    pixel[channel] = color * alpha + pixel[channel] * (1.0 - alpha);
                }
                pixel[3] = alpha + pixel[3] * (1.0 - alpha);
            }
        }
    }
    Ok(image)
}

/// Select and render all requested quality levels against the quality-one reference.
pub fn render_quality_sweep(
    lod: &PlanarGaussian3dLod,
    camera: LodTestCamera,
    settings: &GaussianLodSettings,
    qualities: &[f32],
    width: u32,
    height: u32,
) -> Result<Vec<LodQualitySample>, LodRenderOracleError> {
    let hierarchy = ManifestLodHierarchy::new(&lod.manifest)
        .map_err(|error| LodRenderOracleError::InvalidManifest(format!("{error:?}")))?;
    let view = match camera.projection {
        LodProjection::Perspective {
            vertical_fov_radians,
        } => LodView::perspective(
            camera.position,
            height as f32,
            vertical_fov_radians,
            camera.near,
        ),
        LodProjection::Orthographic {
            vertical_world_size,
        } => LodView::orthographic(
            camera.position,
            height as f32,
            vertical_world_size,
            camera.near,
        ),
    };

    let mut reference_settings = settings.clone();
    reference_settings.quality = 1.0;
    let reference_frontier =
        select_frontier(&hierarchy, &AllResident, view, &reference_settings)
            .map_err(|error| LodRenderOracleError::Selection(format!("{error:?}")))?;
    let reference_gaussians = gather_frontier_gaussians(lod, &reference_frontier.nodes)?;
    let reference = render_linear_gaussians(&reference_gaussians, camera, width, height)?;

    let mut samples = Vec::with_capacity(qualities.len());
    for &quality in qualities {
        let mut settings = settings.clone();
        settings.quality = quality;
        settings
            .validate()
            .map_err(|error| LodRenderOracleError::Selection(error.to_string()))?;
        let frontier = select_frontier(&hierarchy, &AllResident, view, &settings)
            .map_err(|error| LodRenderOracleError::Selection(format!("{error:?}")))?;
        let gaussians = gather_frontier_gaussians(lod, &frontier.nodes)?;
        let rendered = render_linear_gaussians(&gaussians, camera, width, height)?;
        let metrics = compare_linear_rgba(&reference, &rendered, 1.0 / 255.0)
            .map_err(LodRenderOracleError::ImageMetrics)?;
        samples.push(LodQualitySample {
            quality,
            selected_nodes: frontier.nodes.len(),
            active_gaussians: gaussians.len(),
            metrics,
        });
    }
    Ok(samples)
}

#[derive(Clone, Copy, Debug)]
enum Projection {
    Perspective { focal_y: f32 },
    Orthographic { pixels_per_world: f32 },
}

#[derive(Clone, Copy, Debug)]
struct ProjectedGaussian {
    depth: f32,
    pixel_x: f32,
    pixel_y: f32,
    sigma_px: f32,
    color: [f32; 3],
    opacity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gaussian::formats::{
            planar_3d::PlanarGaussian3d,
            planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
        },
        testing::{LodTestScene, all_small_lod_scenes},
    };

    const QUALITIES: [f32; 12] = [
        0.0,
        1e-6,
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        0.99,
        1.0 - 1e-6,
        1.0,
    ];

    #[test]
    fn quality_sweep_exercises_coarsening_and_exact_endpoint() {
        let scene = LodTestScene::checkerboard_facade(16, 16);
        let cloud = PlanarGaussian3d::from(
            scene
                .gaussians
                .iter()
                .map(|entry| entry.gaussian)
                .collect::<Vec<_>>(),
        );
        let lod = build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                leaf_capacity: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let mut settings = GaussianLodSettings::default();
        settings.budgets.max_active_gaussians = 10_000;
        let samples =
            render_quality_sweep(&lod, scene.camera, &settings, &QUALITIES, 96, 96).unwrap();

        assert_eq!(samples.len(), QUALITIES.len());
        assert_eq!(
            samples.first().unwrap().active_gaussians as u64,
            lod.manifest.quality.coarsest_gaussian_count
        );
        assert_eq!(
            samples.last().unwrap().active_gaussians as u64,
            lod.manifest.header.source_gaussian_count
        );
        assert!(samples.last().unwrap().metrics.psnr_rgb.is_infinite());
        assert_eq!(samples.last().unwrap().metrics.luminance_ssim, 1.0);
        assert_eq!(samples.last().unwrap().metrics.alpha_mae, 0.0);
        assert!(
            samples
                .windows(2)
                .all(|pair| pair[1].active_gaussians >= pair[0].active_gaussians),
            "quality sweep counts were not monotonic: {samples:?}"
        );
        assert!(
            samples
                .iter()
                .any(|sample| sample.metrics.psnr_rgb.is_finite()),
            "quality sweep never exercised a visually different coarse representation"
        );
        let coarse = samples.first().unwrap();
        let high = samples.iter().find(|sample| sample.quality == 0.9).unwrap();
        assert!(
            high.metrics.psnr_rgb + 1.0 >= coarse.metrics.psnr_rgb,
            "high quality regressed substantially: coarse={coarse:?}, high={high:?}"
        );
        assert!(high.metrics.luminance_ssim + 0.01 >= coarse.metrics.luminance_ssim);
    }

    #[test]
    fn production_defaults_improve_quality_without_scene_specific_tuning() {
        const DEFAULT_QUALITIES: [f32; 7] = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0];
        let scene = LodTestScene::checkerboard_facade(64, 64);
        let lod = build_planar_3d_lod(&scene.cloud(), GaussianLodBuildSettings::default()).unwrap();
        let settings = GaussianLodSettings::default();
        let samples =
            render_quality_sweep(&lod, scene.camera, &settings, &DEFAULT_QUALITIES, 96, 96)
                .unwrap();

        eprintln!(
            "default LoD quality oracle: {:?}",
            samples
                .iter()
                .map(|sample| (
                    sample.quality,
                    sample.active_gaussians,
                    sample.metrics.psnr_rgb,
                    sample.metrics.luminance_ssim,
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            samples.first().unwrap().active_gaussians as u64,
            lod.manifest.quality.coarsest_gaussian_count
        );
        assert_eq!(
            samples.last().unwrap().active_gaussians as u64,
            lod.manifest.header.source_gaussian_count
        );
        assert!(
            samples
                .windows(2)
                .all(|pair| pair[1].active_gaussians >= pair[0].active_gaussians),
            "default quality counts regressed: {samples:?}"
        );
        for pair in samples.windows(2) {
            assert!(
                pair[1].metrics.psnr_rgb + 0.05 >= pair[0].metrics.psnr_rgb,
                "default quality PSNR regressed: lower={:?}, higher={:?}",
                pair[0],
                pair[1]
            );
            assert!(
                pair[1].metrics.luminance_ssim + 0.002 >= pair[0].metrics.luminance_ssim,
                "default quality SSIM regressed: lower={:?}, higher={:?}",
                pair[0],
                pair[1]
            );
        }
        assert!(
            samples[2].metrics.psnr_rgb >= samples[0].metrics.psnr_rgb + 3.0,
            "the default q=.5 target must materially improve PSNR over q=0: {samples:?}"
        );
        assert!(
            samples[2].metrics.luminance_ssim >= samples[0].metrics.luminance_ssim + 0.1,
            "the default q=.5 target must materially improve SSIM over q=0: {samples:?}"
        );
        assert!(samples.last().unwrap().metrics.psnr_rgb.is_infinite());
        assert_eq!(samples.last().unwrap().metrics.luminance_ssim, 1.0);
    }

    #[test]
    fn orthographic_oracle_is_finite() {
        let scene = LodTestScene::boundary_straddlers();
        let mut camera = scene.camera;
        camera.projection = LodProjection::Orthographic {
            vertical_world_size: 8.0,
        };
        let image = render_linear_gaussians(
            &scene
                .gaussians
                .iter()
                .map(|entry| entry.gaussian)
                .collect::<Vec<_>>(),
            camera,
            64,
            64,
        )
        .unwrap();
        assert!(image.iter().flatten().all(|value| value.is_finite()));
        assert!(image.iter().any(|pixel| pixel[3] > 0.0));
    }

    #[test]
    fn missing_page_is_reported_without_panicking() {
        let scene = LodTestScene::nested_octants(1);
        let cloud = PlanarGaussian3d::from(
            scene
                .gaussians
                .iter()
                .map(|entry| entry.gaussian)
                .collect::<Vec<_>>(),
        );
        let mut lod = build_planar_3d_lod(&cloud, GaussianLodBuildSettings::default()).unwrap();
        let root = lod.manifest.roots[0];
        lod.pages.clear();
        assert!(matches!(
            gather_frontier_gaussians(&lod, &[root]),
            Err(LodRenderOracleError::MissingPage(_))
        ));
    }

    #[test]
    fn every_adversarial_pattern_gets_a_full_quality_image_sweep() {
        const CATALOG_QUALITIES: [f32; 8] = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0];
        for scene in all_small_lod_scenes() {
            let lod = build_planar_3d_lod(
                &scene.cloud(),
                GaussianLodBuildSettings {
                    leaf_capacity: 16,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("{} build failed: {error}", scene.name));
            let mut settings = GaussianLodSettings::default();
            settings.budgets.max_active_gaussians =
                lod.manifest.header.source_gaussian_count.max(1);
            settings.budgets.max_traversal_nodes_per_view = lod.manifest.header.node_count.max(1);
            let samples =
                render_quality_sweep(&lod, scene.camera, &settings, &CATALOG_QUALITIES, 48, 40)
                    .unwrap_or_else(|error| {
                        panic!("{} quality render failed: {error}", scene.name)
                    });

            assert_eq!(
                samples.first().unwrap().active_gaussians as u64,
                lod.manifest.quality.coarsest_gaussian_count,
                "{} violated q=0",
                scene.name
            );
            let exact = samples.last().unwrap();
            assert_eq!(
                exact.active_gaussians as u64, lod.manifest.header.source_gaussian_count,
                "{} violated q=1",
                scene.name
            );
            assert!(
                exact.metrics.psnr_rgb.is_infinite(),
                "{} q=1 RGB",
                scene.name
            );
            assert_eq!(exact.metrics.luminance_ssim, 1.0, "{} q=1 SSIM", scene.name);
            assert_eq!(exact.metrics.alpha_mae, 0.0, "{} q=1 alpha", scene.name);
            assert!(
                samples.first().unwrap().active_gaussians < exact.active_gaussians,
                "{} did not exercise a coarser representation",
                scene.name
            );
            assert!(
                samples
                    .windows(2)
                    .all(|pair| pair[1].active_gaussians >= pair[0].active_gaussians),
                "{} active count is not monotonic: {samples:?}",
                scene.name
            );
            let coarse = samples.first().unwrap();
            let near_original = &samples[samples.len() - 2];
            assert!(
                near_original.metrics.psnr_rgb + 1.0 >= coarse.metrics.psnr_rgb,
                "{} near-original PSNR regressed: coarse={coarse:?}, near={near_original:?}",
                scene.name
            );
            assert!(
                near_original.metrics.luminance_ssim + 0.01 >= coarse.metrics.luminance_ssim,
                "{} near-original SSIM regressed: coarse={coarse:?}, near={near_original:?}",
                scene.name
            );
            for sample in samples {
                let metrics = sample.metrics;
                assert!(
                    metrics.psnr_rgb.is_finite() || metrics.psnr_rgb.is_infinite(),
                    "{} q={} PSNR is NaN",
                    scene.name,
                    sample.quality
                );
                assert!(
                    (-1.0..=1.0).contains(&metrics.luminance_ssim),
                    "{} q={} invalid SSIM",
                    scene.name,
                    sample.quality
                );
                assert!(
                    (0.0..=1.0).contains(&metrics.alpha_mae),
                    "{} q={} invalid alpha MAE",
                    scene.name,
                    sample.quality
                );
                assert!(
                    (0.0..=1.0).contains(&metrics.foreground_iou),
                    "{} q={} invalid foreground IoU",
                    scene.name,
                    sample.quality
                );
            }
        }
    }
}
