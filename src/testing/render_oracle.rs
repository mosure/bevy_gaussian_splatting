//! Small deterministic CPU render oracle for LoD quality regression tests.
//!
//! This is intentionally not a production renderer. It provides a stable, linear-color image
//! comparison that exercises hierarchy cuts and representative Gaussians on every CI adapter.

use std::{collections::BTreeMap, fmt};

use bevy::prelude::{Mat3, Vec2, Vec3, Vec4};

use crate::{
    gaussian::{
        covariance::compute_covariance_3d,
        formats::{
            planar_3d::Gaussian3d,
            planar_3d_chunked::{LodNodeId, LodPageId},
            planar_3d_lod::PlanarGaussian3dLod,
        },
        lod_settings::GaussianLodSettings,
        settings::GaussianColorSpace,
    },
    material::spherical_harmonics::{SH_DEGREE, SphericalHarmonicCoefficients},
    render::{
        GAUSSIAN_AUTHORED_SUPPORT_SIGMA, gaussian_mip_filter_covariance_2d, gaussian_support_cutoff,
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

/// Testing-only render result carrying the logical hierarchy node that made
/// the largest final source-over alpha contribution at each pixel.
///
/// Computing exact dominant attribution is intentionally more expensive than
/// [`render_linear_gaussians`]. It exists to localize thin reconstruction
/// errors around hierarchy boundaries rather than averaging them away in a
/// whole-frame metric.
#[derive(Clone, Debug, PartialEq)]
pub struct LodAttributedImage {
    pub rgba: Vec<[f32; 4]>,
    pub dominant_nodes: Vec<Option<LodNodeId>>,
}

/// Raster-support policy used by the deterministic CPU oracle.
///
/// Existing quality-oracle entry points retain their historical authored
/// three-sigma support. Real flat-vs-LoD comparisons should opt into the two
/// explicit wrappers below so they reproduce the production default: flat
/// source splats use the opacity-adaptive cutoff while LoD representatives
/// retain at least their authored three-sigma support.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodOracleSupport {
    FlatAdaptive,
    LodAuthoredThreeSigma,
}

impl LodOracleSupport {
    #[inline]
    fn cutoff(self, opacity: f32) -> f32 {
        match self {
            Self::FlatAdaptive => gaussian_support_cutoff(opacity, true, false),
            Self::LodAuthoredThreeSigma => GAUSSIAN_AUTHORED_SUPPORT_SIGMA,
        }
    }
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

/// Resolve a selected frontier while retaining the logical owner of every
/// representative record.
pub fn gather_frontier_gaussians_with_nodes(
    lod: &PlanarGaussian3dLod,
    frontier: &[LodNodeId],
) -> Result<Vec<(LodNodeId, Gaussian3d)>, LodRenderOracleError> {
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
        result.extend(slice.iter().copied().map(|gaussian| (node_id, gaussian)));
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
    render_linear_gaussians_internal(
        gaussians.iter().map(|gaussian| (None, gaussian)),
        camera,
        width,
        height,
        false,
        LodOracleSupport::LodAuthoredThreeSigma,
    )
    .map(|rendered| rendered.rgba)
}

/// Render a flat source with the production default opacity-adaptive support.
pub fn render_flat_linear_gaussians(
    gaussians: &[Gaussian3d],
    camera: LodTestCamera,
    width: u32,
    height: u32,
) -> Result<Vec<[f32; 4]>, LodRenderOracleError> {
    render_linear_gaussians_internal(
        gaussians.iter().map(|gaussian| (None, gaussian)),
        camera,
        width,
        height,
        false,
        LodOracleSupport::FlatAdaptive,
    )
    .map(|rendered| rendered.rgba)
}

/// Render a node-attributed frontier and retain the dominant logical node per
/// pixel for boundary-conditioned reconstruction metrics.
pub fn render_linear_gaussians_with_nodes(
    gaussians: &[(LodNodeId, Gaussian3d)],
    camera: LodTestCamera,
    width: u32,
    height: u32,
) -> Result<LodAttributedImage, LodRenderOracleError> {
    render_linear_gaussians_internal(
        gaussians
            .iter()
            .map(|(node, gaussian)| (Some(*node), gaussian)),
        camera,
        width,
        height,
        true,
        LodOracleSupport::LodAuthoredThreeSigma,
    )
}

/// Render a node-attributed LoD cut with the production authored-support
/// floor, even for low-opacity MomentMerge representatives.
pub fn render_lod_linear_gaussians_with_nodes(
    gaussians: &[(LodNodeId, Gaussian3d)],
    camera: LodTestCamera,
    width: u32,
    height: u32,
) -> Result<LodAttributedImage, LodRenderOracleError> {
    render_linear_gaussians_internal(
        gaussians
            .iter()
            .map(|(node, gaussian)| (Some(*node), gaussian)),
        camera,
        width,
        height,
        true,
        LodOracleSupport::LodAuthoredThreeSigma,
    )
}

/// Render a flat source with the production 3D covariance, SH-color, cutoff,
/// and distance-squared ordering contracts.
///
/// The oracle assumes the cloud's world transform, global scale, and global
/// opacity are identity/one, and mirrors the default `DrawMode::All` + OBB
/// raster path. Those are the settings used by the authenticated real-scene
/// gates. Existing lightweight wrappers intentionally retain their historical
/// isotropic/DC behavior for stable synthetic quality baselines.
pub fn render_production_flat_linear_gaussians(
    gaussians: &[Gaussian3d],
    camera: LodTestCamera,
    width: u32,
    height: u32,
    color_space: GaussianColorSpace,
) -> Result<Vec<[f32; 4]>, LodRenderOracleError> {
    render_production_linear_gaussians_internal(
        gaussians.iter().map(|gaussian| (None, gaussian)),
        camera,
        width,
        height,
        false,
        LodOracleSupport::FlatAdaptive,
        color_space,
    )
    .map(|rendered| rendered.rgba)
}

/// Render the flat source with production raster policy while attributing each
/// original record to its selected logical ancestor.
///
/// `owners` is index-aligned with the caller's source order. `None` keeps an
/// out-of-cut or culled record in the flat reference image without allowing it
/// to seed a logical boundary. Ownership affects attribution only; RGBA is
/// bit-identical to [`render_production_flat_linear_gaussians`].
pub fn render_production_flat_linear_gaussians_with_owners(
    gaussians: &[Gaussian3d],
    owners: &[Option<LodNodeId>],
    camera: LodTestCamera,
    width: u32,
    height: u32,
    color_space: GaussianColorSpace,
) -> Result<LodAttributedImage, LodRenderOracleError> {
    if gaussians.len() != owners.len() {
        return Err(LodRenderOracleError::InvalidManifest(
            "flat source and owner lengths differ".to_owned(),
        ));
    }
    render_production_linear_gaussians_internal(
        owners.iter().copied().zip(gaussians),
        camera,
        width,
        height,
        true,
        LodOracleSupport::FlatAdaptive,
        color_space,
    )
}

/// Render an LoD frontier with production policy without constructing a
/// dominant-node attribution map.
pub fn render_production_lod_linear_gaussians(
    gaussians: &[(LodNodeId, Gaussian3d)],
    camera: LodTestCamera,
    width: u32,
    height: u32,
    color_space: GaussianColorSpace,
) -> Result<Vec<[f32; 4]>, LodRenderOracleError> {
    render_production_linear_gaussians_internal(
        gaussians.iter().map(|(_, gaussian)| (None, gaussian)),
        camera,
        width,
        height,
        false,
        LodOracleSupport::LodAuthoredThreeSigma,
        color_space,
    )
    .map(|rendered| rendered.rgba)
}

/// Render an attributed LoD frontier with the production 3D covariance,
/// authored-support floor, SH-color, and distance-squared ordering contracts.
pub fn render_production_lod_linear_gaussians_with_nodes(
    gaussians: &[(LodNodeId, Gaussian3d)],
    camera: LodTestCamera,
    width: u32,
    height: u32,
    color_space: GaussianColorSpace,
) -> Result<LodAttributedImage, LodRenderOracleError> {
    render_production_linear_gaussians_internal(
        gaussians
            .iter()
            .map(|(node, gaussian)| (Some(*node), gaussian)),
        camera,
        width,
        height,
        true,
        LodOracleSupport::LodAuthoredThreeSigma,
        color_space,
    )
}

fn render_production_linear_gaussians_internal<'a>(
    gaussians: impl IntoIterator<Item = (Option<LodNodeId>, &'a Gaussian3d)>,
    camera: LodTestCamera,
    width: u32,
    height: u32,
    attribute_nodes: bool,
    support: LodOracleSupport,
    color_space: GaussianColorSpace,
) -> Result<LodAttributedImage, LodRenderOracleError> {
    if width == 0 || height == 0 {
        return Err(LodRenderOracleError::InvalidImageSize);
    }
    let forward = (camera.target - camera.position).normalize_or_zero();
    let right = forward.cross(camera.up).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    if forward == Vec3::ZERO || right == Vec3::ZERO || up == Vec3::ZERO {
        return Err(LodRenderOracleError::InvalidCamera("basis"));
    }
    let projection = match camera.projection {
        LodProjection::Perspective {
            vertical_fov_radians,
        } if vertical_fov_radians.is_finite()
            && vertical_fov_radians > 0.0
            && vertical_fov_radians < std::f32::consts::PI =>
        {
            ProductionProjection::Perspective {
                pixel_focal: 0.5 * height as f32 / (0.5 * vertical_fov_radians).tan(),
                shader_focal: height as f32 / (0.5 * vertical_fov_radians).tan(),
            }
        }
        LodProjection::Orthographic {
            vertical_world_size,
        } if vertical_world_size.is_finite() && vertical_world_size > 0.0 => {
            ProductionProjection::Orthographic {
                pixels_per_world: height as f32 / vertical_world_size,
                shader_units_per_world: 2.0 * height as f32 / vertical_world_size,
            }
        }
        _ => return Err(LodRenderOracleError::InvalidCamera("projection")),
    };

    let mut projected = Vec::new();
    for (input_order, (node, gaussian)) in gaussians.into_iter().enumerate() {
        // Visibility is selection/classification metadata. The default
        // DrawMode::All color path does not multiply it into opacity.
        if gaussian.scale_opacity.opacity <= 0.0 {
            continue;
        }
        let Some(geometry) = production_projected_geometry(
            gaussian, camera, width, height, forward, right, up, projection,
        ) else {
            continue;
        };
        let opacity = (gaussian.scale_opacity.opacity * geometry.opacity_scale).clamp(0.0, 0.999);
        if !opacity.is_finite() || opacity <= 0.0 {
            continue;
        }
        let support_cutoff = support.cutoff(gaussian.scale_opacity.opacity);
        let Some(obb) = production_obb(geometry.covariance, support_cutoff) else {
            continue;
        };
        projected.push(ProductionProjectedGaussian {
            node,
            input_order,
            distance_squared: geometry.relative.length_squared(),
            center: geometry.center,
            inverse_covariance: geometry.inverse_covariance,
            color: production_spherical_harmonics_linear_color(
                geometry.relative.normalize(),
                &gaussian.spherical_harmonic,
                color_space,
            ),
            opacity,
            obb,
        });
    }

    // The flat radix path and the LoD compactor both key positive finite
    // camera-to-mean distance squared, far to near. The default 32-bit radix
    // setting preserves the complete f32 key; stable input order resolves an
    // equal key in the same way as the GPU passes.
    projected.sort_by(|left, right| {
        right
            .distance_squared
            .total_cmp(&left.distance_squared)
            .then_with(|| left.input_order.cmp(&right.input_order))
    });

    let mut image = vec![[0.0; 4]; (width * height) as usize];
    let mut node_contributions =
        attribute_nodes.then(|| vec![BTreeMap::<LodNodeId, f32>::new(); image.len()]);
    for gaussian in projected {
        let radius_x = gaussian.obb.aabb_radius_pixels.x;
        let radius_y = gaussian.obb.aabb_radius_pixels.y;
        if !radius_x.is_finite() || !radius_y.is_finite() {
            continue;
        }
        let min_x = (gaussian.center.x.floor() as i32 - radius_x.ceil() as i32).max(0);
        let max_x =
            (gaussian.center.x.ceil() as i32 + radius_x.ceil() as i32).min(width as i32 - 1);
        let min_y = (gaussian.center.y.floor() as i32 - radius_y.ceil() as i32).max(0);
        let max_y =
            (gaussian.center.y.ceil() as i32 + radius_y.ceil() as i32).min(height as i32 - 1);
        if min_x > max_x || min_y > max_y {
            continue;
        }
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                // helpers.wgsl covariance is expressed in two units per
                // physical pixel. Vertex bounds divide the radius by the full
                // viewport to reach NDC; the interpolated conic therefore sees
                // the same doubled-pixel displacement here.
                let delta = 2.0
                    * Vec2::new(
                        x as f32 + 0.5 - gaussian.center.x,
                        y as f32 + 0.5 - gaussian.center.y,
                    );
                if !gaussian.obb.contains_shader_delta(delta) {
                    continue;
                }
                let mahalanobis = gaussian.inverse_covariance[0] * delta.x * delta.x
                    + 2.0 * gaussian.inverse_covariance[1] * delta.x * delta.y
                    + gaussian.inverse_covariance[2] * delta.y * delta.y;
                if !mahalanobis.is_finite() || mahalanobis < 0.0 {
                    continue;
                }
                let alpha = (gaussian.opacity * (-0.5 * mahalanobis).exp()).clamp(0.0, 0.999);
                let pixel_index = (y as u32 * width + x as u32) as usize;
                let pixel = &mut image[pixel_index];
                for (channel, color) in gaussian.color.into_iter().enumerate() {
                    pixel[channel] = color * alpha + pixel[channel] * (1.0 - alpha);
                }
                pixel[3] = alpha + pixel[3] * (1.0 - alpha);
                if let Some(contributions) = node_contributions.as_mut() {
                    let contributions = &mut contributions[pixel_index];
                    for contribution in contributions.values_mut() {
                        *contribution *= 1.0 - alpha;
                    }
                    // INVALID represents an unowned source contribution. It
                    // participates in dominance, then maps back to None, so a
                    // culled source splat cannot leave a weaker false label.
                    *contributions
                        .entry(gaussian.node.unwrap_or(LodNodeId::INVALID))
                        .or_default() += alpha;
                }
            }
        }
    }
    let dominant_nodes = dominant_nodes_from_contributions(node_contributions, image.len());
    Ok(LodAttributedImage {
        rgba: image,
        dominant_nodes,
    })
}

fn dominant_nodes_from_contributions(
    node_contributions: Option<Vec<BTreeMap<LodNodeId, f32>>>,
    pixel_count: usize,
) -> Vec<Option<LodNodeId>> {
    node_contributions
        .map(|contributions| {
            contributions
                .into_iter()
                .map(|contributions| {
                    contributions
                        .into_iter()
                        .max_by(|left, right| {
                            left.1
                                .total_cmp(&right.1)
                                .then_with(|| right.0.cmp(&left.0))
                        })
                        .and_then(|(node, _)| node.is_valid().then_some(node))
                })
                .collect()
        })
        .unwrap_or_else(|| vec![None; pixel_count])
}

#[derive(Clone, Copy, Debug)]
enum ProductionProjection {
    Perspective {
        pixel_focal: f32,
        shader_focal: f32,
    },
    Orthographic {
        pixels_per_world: f32,
        shader_units_per_world: f32,
    },
}

#[derive(Clone, Copy, Debug)]
struct ProductionProjectedGeometry {
    relative: Vec3,
    center: Vec2,
    /// Filtered covariance in helpers.wgsl coordinates: two units per pixel.
    covariance: [f32; 3],
    inverse_covariance: [f32; 3],
    opacity_scale: f32,
}

#[allow(clippy::too_many_arguments)]
fn production_projected_geometry(
    gaussian: &Gaussian3d,
    camera: LodTestCamera,
    width: u32,
    height: u32,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    projection: ProductionProjection,
) -> Option<ProductionProjectedGeometry> {
    let world = Vec3::from_array(gaussian.position_visibility.position);
    let relative = world - camera.position;
    let depth = relative.dot(forward);
    if !depth.is_finite() || depth < camera.near || depth > camera.far {
        return None;
    }
    let view_x = relative.dot(right);
    let view_y = relative.dot(up);
    let (center, jacobian_x, jacobian_y) = match projection {
        ProductionProjection::Perspective {
            pixel_focal,
            shader_focal,
        } => {
            let reciprocal_depth = 1.0 / depth;
            let reciprocal_depth_squared = reciprocal_depth * reciprocal_depth;
            (
                Vec2::new(
                    width as f32 * 0.5 + pixel_focal * view_x * reciprocal_depth,
                    height as f32 * 0.5 - pixel_focal * view_y * reciprocal_depth,
                ),
                right * (shader_focal * reciprocal_depth)
                    - forward * (shader_focal * view_x * reciprocal_depth_squared),
                -up * (shader_focal * reciprocal_depth)
                    + forward * (shader_focal * view_y * reciprocal_depth_squared),
            )
        }
        ProductionProjection::Orthographic {
            pixels_per_world,
            shader_units_per_world,
        } => (
            Vec2::new(
                width as f32 * 0.5 + view_x * pixels_per_world,
                height as f32 * 0.5 - view_y * pixels_per_world,
            ),
            right * shader_units_per_world,
            -up * shader_units_per_world,
        ),
    };
    if !center.is_finite() || relative == Vec3::ZERO {
        return None;
    }

    // CPU form of gaussian_3d.wgsl::compute_local_cov3d plus
    // helpers.wgsl::cov2d for an identity cloud transform. Both Jacobian
    // columns have the opposite sign from the literal Bevy -Z view-space port;
    // the common sign cancels in J^T Sigma J, including the xy covariance.
    let packed = compute_covariance_3d(
        Vec4::from_array(gaussian.rotation.rotation),
        Vec3::from_array(gaussian.scale_opacity.scale),
    );
    let covariance_3d = Mat3::from_cols(
        Vec3::new(packed[0], packed[1], packed[2]),
        Vec3::new(packed[1], packed[3], packed[4]),
        Vec3::new(packed[2], packed[4], packed[5]),
    );
    let covariance_shader = [
        jacobian_x.dot(covariance_3d * jacobian_x),
        jacobian_x.dot(covariance_3d * jacobian_y),
        jacobian_y.dot(covariance_3d * jacobian_y),
    ];
    // The public CPU helper is intentionally expressed in physical-pixel
    // covariance (+0.3 px^2), while helpers.wgsl::cov2d uses two coordinate
    // units per pixel (+1.2 shader units^2). Filter after dividing by four,
    // then restore shader coordinates; the determinant opacity ratio is
    // invariant under that common scale.
    let mip = gaussian_mip_filter_covariance_2d(covariance_shader.map(|value| 0.25 * value));
    let covariance = mip.covariance.map(|value| 4.0 * value);
    let [covariance_x, covariance_xy, covariance_y] = covariance;
    let determinant = covariance_x * covariance_y - covariance_xy * covariance_xy;
    if !(determinant > 0.0 && determinant.is_finite()) {
        return None;
    }
    let inverse_covariance = [
        covariance_y / determinant,
        -covariance_xy / determinant,
        covariance_x / determinant,
    ];
    if !inverse_covariance.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(ProductionProjectedGeometry {
        relative,
        center,
        covariance,
        inverse_covariance,
        opacity_scale: mip.opacity_scale,
    })
}

#[derive(Clone, Copy, Debug)]
struct ProductionProjectedGaussian {
    node: Option<LodNodeId>,
    input_order: usize,
    distance_squared: f32,
    center: Vec2,
    inverse_covariance: [f32; 3],
    color: [f32; 3],
    opacity: f32,
    obb: ProductionObb,
}

/// Default OBB raster support in helpers.wgsl's doubled-pixel coordinates.
#[derive(Clone, Copy, Debug)]
struct ProductionObb {
    major_axis: Vec2,
    minor_axis: Vec2,
    half_extent_shader: Vec2,
    aabb_radius_pixels: Vec2,
}

impl ProductionObb {
    #[inline]
    fn contains_shader_delta(self, delta: Vec2) -> bool {
        delta.dot(self.major_axis).abs() <= self.half_extent_shader.x
            && delta.dot(self.minor_axis).abs() <= self.half_extent_shader.y
    }
}

fn production_obb(covariance: [f32; 3], cutoff: f32) -> Option<ProductionObb> {
    let [covariance_x, covariance_xy, covariance_y] = covariance;
    let determinant = covariance_x * covariance_y - covariance_xy * covariance_xy;
    let midpoint = 0.5 * (covariance_x + covariance_y);
    let discriminant = (midpoint * midpoint - determinant).max(0.0);
    let term = discriminant.sqrt();
    let major_radius = (midpoint + term).sqrt();
    let minor_radius = (midpoint - term).max(0.0).sqrt();
    if !cutoff.is_finite()
        || cutoff <= 0.0
        || !major_radius.is_finite()
        || !minor_radius.is_finite()
        || minor_radius <= 0.0
    {
        return None;
    }

    // Keep this zero-safe analytic eigenvector and negative-handed
    // perpendicular in lockstep with helpers.wgsl::get_bounding_box_clip.
    let candidate = Vec2::new(-covariance_xy, midpoint + term - covariance_x);
    let major_axis = if candidate.x.abs() + candidate.y.abs() > 1.0e-12 {
        candidate.normalize()
    } else {
        Vec2::X
    };
    let minor_axis = Vec2::new(major_axis.y, -major_axis.x);
    let half_extent_shader = cutoff * Vec2::new(major_radius, minor_radius);
    let aabb_radius_pixels =
        0.5 * (major_axis.abs() * half_extent_shader.x + minor_axis.abs() * half_extent_shader.y);
    if !major_axis.is_finite()
        || !minor_axis.is_finite()
        || !half_extent_shader.is_finite()
        || !aabb_radius_pixels.is_finite()
    {
        return None;
    }
    Some(ProductionObb {
        major_axis,
        minor_axis,
        half_extent_shader,
        aabb_radius_pixels,
    })
}

// Keep the basis order and expressions in lockstep with
// material/spherical_harmonics.wgsl. Coefficients are interleaved RGB triples.
const PRODUCTION_SH_BASIS_CONSTANTS: [f32; 16] = [
    0.282_094_8,
    -0.488_602_52,
    0.488_602_52,
    -0.488_602_52,
    1.092_548_5,
    -1.092_548_5,
    0.315_391_57,
    -1.092_548_5,
    0.546_274_24,
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_34,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];

fn production_spherical_harmonics_linear_color(
    ray_direction: Vec3,
    spherical_harmonic: &SphericalHarmonicCoefficients,
    color_space: GaussianColorSpace,
) -> [f32; 3] {
    let squared = ray_direction * ray_direction;
    let basis = [
        PRODUCTION_SH_BASIS_CONSTANTS[0],
        PRODUCTION_SH_BASIS_CONSTANTS[1] * ray_direction.y,
        PRODUCTION_SH_BASIS_CONSTANTS[2] * ray_direction.z,
        PRODUCTION_SH_BASIS_CONSTANTS[3] * ray_direction.x,
        PRODUCTION_SH_BASIS_CONSTANTS[4] * ray_direction.x * ray_direction.y,
        PRODUCTION_SH_BASIS_CONSTANTS[5] * ray_direction.y * ray_direction.z,
        PRODUCTION_SH_BASIS_CONSTANTS[6] * (2.0 * squared.z - squared.x - squared.y),
        PRODUCTION_SH_BASIS_CONSTANTS[7] * ray_direction.x * ray_direction.z,
        PRODUCTION_SH_BASIS_CONSTANTS[8] * (squared.x - squared.y),
        PRODUCTION_SH_BASIS_CONSTANTS[9] * ray_direction.y * (3.0 * squared.x - squared.y),
        PRODUCTION_SH_BASIS_CONSTANTS[10] * ray_direction.x * ray_direction.y * ray_direction.z,
        PRODUCTION_SH_BASIS_CONSTANTS[11]
            * ray_direction.y
            * (4.0 * squared.z - squared.x - squared.y),
        PRODUCTION_SH_BASIS_CONSTANTS[12]
            * ray_direction.z
            * (2.0 * squared.z - 3.0 * squared.x - 3.0 * squared.y),
        PRODUCTION_SH_BASIS_CONSTANTS[13]
            * ray_direction.x
            * (4.0 * squared.z - squared.x - squared.y),
        PRODUCTION_SH_BASIS_CONSTANTS[14] * ray_direction.z * (squared.x - squared.y),
        PRODUCTION_SH_BASIS_CONSTANTS[15] * ray_direction.x * (squared.x - 3.0 * squared.y),
    ];
    let basis_count = match SH_DEGREE {
        0 => 1,
        1 => 4,
        2 => 9,
        _ => 16,
    };
    let mut color = [0.5_f32; 3];
    for (basis_index, basis) in basis.into_iter().take(basis_count).enumerate() {
        for (channel, color) in color.iter_mut().enumerate() {
            *color += spherical_harmonic
                .coefficients
                .get(basis_index * 3 + channel)
                .copied()
                .unwrap_or(0.0)
                * basis;
        }
    }
    if color_space == GaussianColorSpace::SrgbRec709Display {
        color = color.map(production_srgb_display_channel_to_linear);
    }
    color
}

#[inline]
fn production_srgb_display_channel_to_linear(value: f32) -> f32 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn render_linear_gaussians_internal<'a>(
    gaussians: impl IntoIterator<Item = (Option<LodNodeId>, &'a Gaussian3d)>,
    camera: LodTestCamera,
    width: u32,
    height: u32,
    attribute_nodes: bool,
    support: LodOracleSupport,
) -> Result<LodAttributedImage, LodRenderOracleError> {
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

    let mut projected = Vec::new();
    for (node, gaussian) in gaussians {
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
        let (pixel_x, pixel_y, unfiltered_sigma_px) = match projection {
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
        if !pixel_x.is_finite() || !pixel_y.is_finite() || !unfiltered_sigma_px.is_finite() {
            continue;
        }
        // This oracle intentionally uses an isotropic projection of the
        // representative's largest authored scale, but its screen filter must
        // still match the production determinant-normalized +0.3 variance
        // contract. Otherwise low-resolution boundary metrics would reward the
        // old opacity-inflating dilation that the renderer no longer uses.
        let variance = unfiltered_sigma_px.max(1e-5).powi(2);
        let mip = gaussian_mip_filter_covariance_2d([variance, 0.0, variance]);
        let sigma_px = mip.covariance[0].max(0.0).sqrt();
        let coefficients = &gaussian.spherical_harmonic.coefficients;
        let color = [
            coefficients.first().copied().unwrap_or(0.0).clamp(0.0, 1.0),
            coefficients.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0),
            coefficients.get(2).copied().unwrap_or(0.0).clamp(0.0, 1.0),
        ];
        projected.push(ProjectedGaussian {
            node,
            depth,
            pixel_x,
            pixel_y,
            sigma_px,
            color,
            opacity: (gaussian.scale_opacity.opacity
                * gaussian.position_visibility.visibility
                * mip.opacity_scale)
                .clamp(0.0, 0.999),
            support_cutoff: support.cutoff(gaussian.scale_opacity.opacity),
        });
    }
    projected.sort_by(|left, right| right.depth.total_cmp(&left.depth));

    let mut image = vec![[0.0; 4]; (width * height) as usize];
    let mut node_contributions =
        attribute_nodes.then(|| vec![BTreeMap::<LodNodeId, f32>::new(); (width * height) as usize]);
    for gaussian in projected {
        let radius = (gaussian.sigma_px * gaussian.support_cutoff).ceil() as i32;
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
                let pixel_index = (y as u32 * width + x as u32) as usize;
                let pixel = &mut image[pixel_index];
                for (channel, color) in gaussian.color.into_iter().enumerate() {
                    pixel[channel] = color * alpha + pixel[channel] * (1.0 - alpha);
                }
                pixel[3] = alpha + pixel[3] * (1.0 - alpha);
                if let Some(contributions) = node_contributions.as_mut() {
                    let contributions = &mut contributions[pixel_index];
                    for contribution in contributions.values_mut() {
                        *contribution *= 1.0 - alpha;
                    }
                    if let Some(node) = gaussian.node {
                        *contributions.entry(node).or_default() += alpha;
                    }
                }
            }
        }
    }
    let dominant_nodes = node_contributions
        .map(|contributions| {
            contributions
                .into_iter()
                .map(|contributions| {
                    contributions
                        .into_iter()
                        .max_by(|left, right| {
                            left.1
                                .total_cmp(&right.1)
                                .then_with(|| right.0.cmp(&left.0))
                        })
                        .map(|(node, _)| node)
                })
                .collect()
        })
        .unwrap_or_else(|| vec![None; image.len()]);
    Ok(LodAttributedImage {
        rgba: image,
        dominant_nodes,
    })
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
    node: Option<LodNodeId>,
    depth: f32,
    pixel_x: f32,
    pixel_y: f32,
    sigma_px: f32,
    color: [f32; 3],
    opacity: f32,
    support_cutoff: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gaussian::{
            formats::{
                planar_3d::PlanarGaussian3d,
                planar_3d_chunked::LodPageStorage,
                planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
            },
            lod_settings::GaussianStreamingSettings,
        },
        io::lod::encode_page,
        stream::{
            hierarchy::{
                LodTemporalStepBudget, apply_temporal_substitution_step,
                select_frontier_with_previous, temporal_substitution_candidates,
            },
            runtime::{LodRuntimeViewId, LodStreamingRuntime, LodTemporalTransitionMode},
            transport::MemoryPageTransport,
        },
        testing::{
            LodTestScene, all_small_lod_scenes, compare_temporal_deltas,
            compare_temporal_second_differences,
        },
    };

    fn assert_close(actual: f32, expected: f32, tolerance: f32, label: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} differs: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    #[test]
    fn production_projection_matches_shader_mean_covariance_mip_cutoff_and_sh_contracts() {
        let scene = LodTestScene::screen_space_ladder();
        let mut gaussian = scene.gaussians[0].gaussian;
        gaussian.position_visibility.position = [0.0, 0.0, 0.0];
        gaussian.position_visibility.visibility = 0.8;
        gaussian.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
        gaussian.scale_opacity.scale = [1.0, 2.0, 0.5];
        gaussian.scale_opacity.opacity = 0.5;
        gaussian.spherical_harmonic.coefficients.fill(0.0);
        gaussian.spherical_harmonic.coefficients[0] = 1.0;
        gaussian.spherical_harmonic.coefficients[1] = -0.5;
        gaussian.spherical_harmonic.coefficients[2] = 0.25;
        let camera = LodTestCamera {
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            projection: LodProjection::Perspective {
                vertical_fov_radians: std::f32::consts::FRAC_PI_2,
            },
            near: 0.1,
            far: 100.0,
            viewport: [200, 100],
        };
        let forward = (camera.target - camera.position).normalize();
        let right = forward.cross(camera.up).normalize();
        let up = right.cross(forward).normalize();
        let projection = ProductionProjection::Perspective {
            pixel_focal: 50.0,
            shader_focal: 100.0,
        };
        let geometry = production_projected_geometry(
            &gaussian, camera, 200, 100, forward, right, up, projection,
        )
        .expect("centered Gaussian projects");
        assert_eq!(geometry.center, Vec2::new(100.0, 50.0));
        // helpers.wgsl uses two covariance units per physical pixel. At depth
        // five, its full-viewport focal 100 yields sigma 20 and 40 before mip;
        // +0.3 physical px^2 is therefore +1.2 in shader covariance.
        assert_close(geometry.covariance[0], 401.2, 2e-4, "shader covariance xx");
        assert_close(geometry.covariance[1], 0.0, 1e-6, "shader covariance xy");
        assert_close(
            geometry.covariance[2],
            1_601.2,
            5e-4,
            "shader covariance yy",
        );
        assert_close(
            geometry.covariance[0] * 0.25,
            100.3,
            1e-4,
            "physical covariance xx",
        );
        assert_close(
            geometry.covariance[2] * 0.25,
            400.3,
            2e-4,
            "physical covariance yy",
        );
        let expected_opacity_scale = ((400.0_f32 * 1_600.0) / (401.2_f32 * 1_601.2)).sqrt();
        assert_close(
            geometry.opacity_scale,
            expected_opacity_scale,
            1e-6,
            "determinant mip opacity",
        );
        assert_close(
            gaussian.scale_opacity.opacity * geometry.opacity_scale,
            0.5 * expected_opacity_scale,
            1e-6,
            "DrawMode::All opacity ignores visibility",
        );
        assert_eq!(
            LodOracleSupport::FlatAdaptive
                .cutoff(gaussian.scale_opacity.opacity)
                .to_bits(),
            gaussian_support_cutoff(gaussian.scale_opacity.opacity, true, false).to_bits(),
        );
        assert_eq!(
            LodOracleSupport::LodAuthoredThreeSigma
                .cutoff(gaussian.scale_opacity.opacity)
                .to_bits(),
            3.0_f32.to_bits(),
        );
        let linear = production_spherical_harmonics_linear_color(
            geometry.relative.normalize(),
            &gaussian.spherical_harmonic,
            GaussianColorSpace::LinRec709Display,
        );
        let expected_display = [
            0.5 + 0.282_094_8,
            0.5 - 0.5 * 0.282_094_8,
            0.5 + 0.25 * 0.282_094_8,
        ];
        for channel in 0..3 {
            assert_close(
                linear[channel],
                expected_display[channel],
                1e-6,
                "linear SH",
            );
        }
        let srgb = production_spherical_harmonics_linear_color(
            geometry.relative.normalize(),
            &gaussian.spherical_harmonic,
            GaussianColorSpace::SrgbRec709Display,
        );
        for channel in 0..3 {
            assert_close(
                srgb[channel],
                production_srgb_display_channel_to_linear(expected_display[channel]),
                1e-6,
                "sRGB SH conversion",
            );
        }

        let mut off_axis = gaussian;
        off_axis.position_visibility.position = [1.0, 2.0, 0.0];
        let off_axis = production_projected_geometry(
            &off_axis, camera, 200, 100, forward, right, up, projection,
        )
        .expect("off-axis Gaussian projects");
        assert_eq!(off_axis.center, Vec2::new(110.0, 30.0));

        let orthographic = production_projected_geometry(
            &off_axis_gaussian(&gaussian),
            camera,
            200,
            100,
            forward,
            right,
            up,
            ProductionProjection::Orthographic {
                pixels_per_world: 10.0,
                shader_units_per_world: 20.0,
            },
        )
        .expect("orthographic Gaussian projects");
        assert_eq!(orthographic.center, Vec2::new(110.0, 30.0));
        assert_close(
            orthographic.covariance[0],
            401.2,
            2e-4,
            "orthographic shader covariance xx",
        );
        assert_close(
            orthographic.covariance[2],
            1_601.2,
            5e-4,
            "orthographic shader covariance yy",
        );

        let mut rotated = gaussian;
        let half_sqrt = 0.5_f32.sqrt();
        rotated.rotation.rotation = [half_sqrt, 0.0, 0.0, half_sqrt];
        let rotated = production_projected_geometry(
            &rotated, camera, 200, 100, forward, right, up, projection,
        )
        .expect("rotated Gaussian projects");
        assert_close(
            rotated.covariance[0],
            1_601.2,
            5e-4,
            "rotated covariance xx",
        );
        assert_close(rotated.covariance[2], 401.2, 2e-4, "rotated covariance yy");

        let obb = production_obb(rotated.covariance, 3.0).expect("diagonal x-major OBB is finite");
        assert_close(obb.major_axis.x.abs(), 1.0, 1e-6, "x-major OBB axis");
        assert_close(obb.major_axis.y.abs(), 0.0, 1e-6, "x-major OBB cross axis");
        assert_close(
            obb.major_axis.perp_dot(obb.minor_axis),
            -1.0,
            1e-6,
            "OBB negative handedness",
        );
        let corner = 0.9
            * (obb.major_axis * obb.half_extent_shader.x
                + obb.minor_axis * obb.half_extent_shader.y);
        assert!(obb.contains_shader_delta(corner));
        let corner_mahalanobis = rotated.inverse_covariance[0] * corner.x * corner.x
            + 2.0 * rotated.inverse_covariance[1] * corner.x * corner.y
            + rotated.inverse_covariance[2] * corner.y * corner.y;
        assert!(
            corner_mahalanobis > 9.0 && corner_mahalanobis < 18.0,
            "the production OBB retains valid anisotropic rectangle corners"
        );
    }

    fn off_axis_gaussian(gaussian: &Gaussian3d) -> Gaussian3d {
        let mut off_axis = *gaussian;
        off_axis.position_visibility.position = [1.0, 2.0, 0.0];
        off_axis
    }

    #[test]
    fn attributed_oracle_tracks_the_dominant_logical_node() {
        let scene = LodTestScene::screen_space_ladder();
        let tagged = scene
            .gaussians
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                (
                    LodNodeId(if index % 2 == 0 { 1 } else { 2 }),
                    entry.gaussian,
                )
            })
            .collect::<Vec<_>>();
        let rendered = render_linear_gaussians_with_nodes(&tagged, scene.camera, 96, 64).unwrap();
        assert_eq!(rendered.rgba.len(), 96 * 64);
        assert_eq!(rendered.dominant_nodes.len(), rendered.rgba.len());
        assert!(rendered.dominant_nodes.contains(&Some(LodNodeId(1))));
        assert!(rendered.dominant_nodes.contains(&Some(LodNodeId(2))));
        assert!(
            rendered
                .dominant_nodes
                .iter()
                .zip(&rendered.rgba)
                .all(|(node, pixel)| node.is_some() || pixel[3] == 0.0),
            "an attributed foreground pixel lost its logical owner"
        );
    }

    #[test]
    fn production_flat_owner_attribution_does_not_change_rgba() {
        let scene = LodTestScene::screen_space_ladder();
        let gaussians = scene
            .gaussians
            .iter()
            .map(|entry| entry.gaussian)
            .collect::<Vec<_>>();
        let owners = gaussians
            .iter()
            .enumerate()
            .map(|(index, _)| (index % 3 != 0).then_some(LodNodeId((index % 2 + 1) as u64)))
            .collect::<Vec<_>>();
        let plain = render_production_flat_linear_gaussians(
            &gaussians,
            scene.camera,
            96,
            64,
            GaussianColorSpace::SrgbRec709Display,
        )
        .unwrap();
        let attributed = render_production_flat_linear_gaussians_with_owners(
            &gaussians,
            &owners,
            scene.camera,
            96,
            64,
            GaussianColorSpace::SrgbRec709Display,
        )
        .unwrap();
        assert_eq!(attributed.rgba, plain);
        assert!(
            attributed.dominant_nodes.iter().any(Option::is_some),
            "attributed production reference did not retain any logical owner"
        );
    }

    #[test]
    fn production_flat_unowned_source_can_dominate_a_selected_owner() {
        let scene = LodTestScene::screen_space_ladder();
        let mut gaussian = scene.gaussians[0].gaussian;
        gaussian.position_visibility.position = scene.camera.target.to_array();
        gaussian.position_visibility.visibility = 1.0;
        gaussian.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
        gaussian.scale_opacity.scale = [0.5; 3];
        gaussian.scale_opacity.opacity = 0.8;
        gaussian.spherical_harmonic.coefficients.fill(0.0);
        let gaussians = [gaussian, gaussian];
        let attributed = render_production_flat_linear_gaussians_with_owners(
            &gaussians,
            &[Some(LodNodeId(7)), None],
            scene.camera,
            96,
            64,
            GaussianColorSpace::SrgbRec709Display,
        )
        .unwrap();
        let foreground = attributed
            .rgba
            .iter()
            .zip(&attributed.dominant_nodes)
            .filter(|(pixel, _)| pixel[3] > 0.0)
            .collect::<Vec<_>>();
        assert!(!foreground.is_empty());
        assert!(
            foreground.into_iter().all(|(_, owner)| owner.is_none()),
            "a stronger unowned flat contribution left a weaker selected-owner label"
        );
    }

    #[test]
    fn flat_and_lod_oracle_support_match_the_production_cutoff_contract() {
        for opacity in [0.001_f32, 0.01, 0.1, 0.5, 1.0] {
            assert_eq!(
                LodOracleSupport::FlatAdaptive.cutoff(opacity).to_bits(),
                gaussian_support_cutoff(opacity, true, false).to_bits()
            );
            assert_eq!(
                LodOracleSupport::LodAuthoredThreeSigma
                    .cutoff(opacity)
                    .to_bits(),
                gaussian_support_cutoff(opacity, true, true).to_bits()
            );
            assert!(
                LodOracleSupport::LodAuthoredThreeSigma.cutoff(opacity)
                    >= LodOracleSupport::FlatAdaptive.cutoff(opacity)
            );
        }

        let scene = LodTestScene::screen_space_ladder();
        let mut gaussian = scene.gaussians[0].gaussian;
        gaussian.position_visibility.position = [0.0, 0.0, 0.0];
        gaussian.position_visibility.visibility = 1.0;
        gaussian.scale_opacity.scale = [0.75; 3];
        gaussian.scale_opacity.opacity = 0.02;
        let camera = LodTestCamera {
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            projection: LodProjection::Perspective {
                vertical_fov_radians: 60.0_f32.to_radians(),
            },
            near: 0.01,
            far: 100.0,
            viewport: [96, 96],
        };
        let flat = render_flat_linear_gaussians(&[gaussian], camera, 96, 96).unwrap();
        let tagged = [(LodNodeId(1), gaussian)];
        let lod = render_lod_linear_gaussians_with_nodes(&tagged, camera, 96, 96).unwrap();
        let legacy = render_linear_gaussians(&[gaussian], camera, 96, 96).unwrap();
        assert_eq!(
            legacy, lod.rgba,
            "legacy wrapper must retain authored 3sigma support"
        );
        let visible = |image: &[[f32; 4]]| image.iter().filter(|pixel| pixel[3] > 0.0).count();
        assert!(
            visible(&lod.rgba) > visible(&flat),
            "low-opacity LoD support floor must retain pixels truncated by flat adaptive support"
        );
    }

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
                // Global SSIM is a smoke metric rather than a selector
                // authority; a substantially larger foreground can trade a
                // tiny SSIM dip for better PSNR, alpha error, and overlap.
                pair[1].metrics.luminance_ssim + 0.005 >= pair[0].metrics.luminance_ssim,
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

    #[derive(Debug)]
    struct TemporalTraceSample {
        camera_z: f32,
        frontier: Vec<LodNodeId>,
        frontier_hash: u64,
        active_gaussians: u64,
        exact: Vec<[f32; 4]>,
        lod: Vec<[f32; 4]>,
    }

    #[test]
    fn stateful_camera_trace_bounds_lod_temporal_discontinuities() {
        const WIDTH: u32 = 80;
        const HEIGHT: u32 = 80;
        const SAMPLE_COUNT: usize = 257;
        const NEAR_Z: f32 = 6.0;
        const FAR_Z: f32 = 48.0;
        const QUALITY: f32 = 0.60;
        const FOREGROUND_ALPHA: f32 = 1.0 / 255.0;

        // These are fixed fixture gates, not thresholds inferred from this
        // run. They deliberately leave modest numerical headroom while still
        // rejecting a broad representation jump or noisy unchanged cut.
        const MAX_NO_CUT_DELTA_RMSE: f64 = 0.025;
        const MAX_CUT_DELTA_RMSE: f64 = 0.075;
        const MAX_NO_CUT_CURVATURE_RMSE: f64 = 0.008;
        const MAX_CUT_CURVATURE_RMSE: f64 = 0.10;

        let scene = LodTestScene::checkerboard_facade(24, 16);
        let lod = build_planar_3d_lod(
            &scene.cloud(),
            GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 8,
                support_sigma: 3.0,
            },
        )
        .expect("temporal checkerboard hierarchy builds");
        let hierarchy = ManifestLodHierarchy::new(&lod.manifest)
            .expect("temporal checkerboard manifest is valid");
        let source_count = lod.manifest.header.source_gaussian_count;
        let mut settings = GaussianLodSettings {
            quality: QUALITY,
            frustum_culling: false,
            ..Default::default()
        };
        settings.budgets.max_active_gaussians = source_count;
        let mut exact_settings = settings.clone();
        exact_settings.quality = 1.0;

        let mut previous_frontier = Vec::new();
        let mut trace = Vec::with_capacity(SAMPLE_COUNT);
        for index in 0..SAMPLE_COUNT {
            let fraction = index as f32 / (SAMPLE_COUNT - 1) as f32;
            let camera_z = NEAR_Z * (FAR_Z / NEAR_Z).powf(fraction);
            let mut camera = scene.camera;
            camera.position = Vec3::new(0.0, 0.0, camera_z);
            camera.viewport = [WIDTH, HEIGHT];
            let view = LodView::perspective(
                camera.position,
                HEIGHT as f32,
                match camera.projection {
                    LodProjection::Perspective {
                        vertical_fov_radians,
                    } => vertical_fov_radians,
                    LodProjection::Orthographic { .. } => unreachable!(),
                },
                camera.near,
            );
            let frontier = select_frontier_with_previous(
                &hierarchy,
                &AllResident,
                view,
                &settings,
                &previous_frontier,
            )
            .expect("stateful temporal selection succeeds");
            let exact_frontier = select_frontier(&hierarchy, &AllResident, view, &exact_settings)
                .expect("quality-one temporal selection succeeds");
            assert_eq!(
                exact_frontier.status.active_gaussians, source_count,
                "quality one must remain the exact endpoint at camera z={camera_z}"
            );

            let lod_gaussians = gather_frontier_gaussians(&lod, &frontier.nodes)
                .expect("temporal LoD frontier resolves");
            let exact_gaussians = gather_frontier_gaussians(&lod, &exact_frontier.nodes)
                .expect("temporal exact frontier resolves");
            let lod_image = render_linear_gaussians(&lod_gaussians, camera, WIDTH, HEIGHT)
                .expect("temporal LoD image renders");
            let exact_image = render_linear_gaussians(&exact_gaussians, camera, WIDTH, HEIGHT)
                .expect("temporal exact image renders");
            assert!(
                exact_image.iter().any(|pixel| pixel[3] > FOREGROUND_ALPHA),
                "quality-one temporal image is empty at camera z={camera_z}"
            );
            assert_eq!(
                frontier.status.active_gaussians,
                lod_gaussians.len() as u64,
                "selector and gathered temporal cut counts differ"
            );

            previous_frontier.clone_from(&frontier.nodes);
            trace.push(TemporalTraceSample {
                camera_z,
                frontier_hash: stable_frontier_hash(&frontier.nodes),
                frontier: frontier.nodes,
                active_gaussians: lod_gaussians.len() as u64,
                exact: exact_image,
                lod: lod_image,
            });
        }

        let mut no_cut_delta_rmse = Vec::new();
        let mut cut_delta_rmse = Vec::new();
        let mut events = Vec::new();
        for (index, pair) in trace.windows(2).enumerate() {
            let [previous, current] = pair else {
                unreachable!()
            };
            let metrics = compare_temporal_deltas(
                &previous.exact,
                &current.exact,
                &previous.lod,
                &current.lod,
                FOREGROUND_ALPHA,
            )
            .expect("temporal delta metrics are valid");
            assert!(metrics.foreground_pixels > 0);
            if previous.frontier == current.frontier {
                no_cut_delta_rmse.push(metrics.foreground_rgb_rmse);
            } else {
                cut_delta_rmse.push(metrics.foreground_rgb_rmse);
                events.push((
                    index + 1,
                    current.camera_z,
                    previous.frontier_hash,
                    current.frontier_hash,
                    previous.active_gaussians,
                    current.active_gaussians,
                    metrics.foreground_rgb_rmse,
                ));
            }
        }

        let mut no_cut_curvature_rmse = Vec::new();
        let mut cut_curvature_rmse = Vec::new();
        for triple in trace.windows(3) {
            let [previous, current, next] = triple else {
                unreachable!()
            };
            let metrics = compare_temporal_second_differences(
                &previous.exact,
                &current.exact,
                &next.exact,
                &previous.lod,
                &current.lod,
                &next.lod,
                FOREGROUND_ALPHA,
            )
            .expect("temporal curvature metrics are valid");
            let cut_changed =
                previous.frontier != current.frontier || current.frontier != next.frontier;
            if cut_changed {
                cut_curvature_rmse.push(metrics.foreground_rgb_rmse);
            } else {
                no_cut_curvature_rmse.push(metrics.foreground_rgb_rmse);
            }
        }

        let no_cut_delta_max = finite_max(&no_cut_delta_rmse);
        let cut_delta_max = finite_max(&cut_delta_rmse);
        let no_cut_curvature_max = finite_max(&no_cut_curvature_rmse);
        let cut_curvature_max = finite_max(&cut_curvature_rmse);
        let distinct_cuts = trace
            .iter()
            .map(|sample| sample.frontier.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        eprintln!(
            "stateful LoD temporal trace: samples={} distinct_cuts={} cut_events={} no_cut_intervals={} no_cut_delta_max={:.6} cut_delta_max={:.6} no_cut_curvature_max={:.6} cut_curvature_max={:.6} events={events:?}",
            trace.len(),
            distinct_cuts,
            cut_delta_rmse.len(),
            no_cut_delta_rmse.len(),
            no_cut_delta_max,
            cut_delta_max,
            no_cut_curvature_max,
            cut_curvature_max,
        );

        assert!(
            distinct_cuts >= 4 && cut_delta_rmse.len() >= 3,
            "temporal fixture stopped exercising enough stateful cut changes: {events:?}"
        );
        assert!(
            no_cut_delta_rmse.len() >= SAMPLE_COUNT / 3,
            "temporal fixture has too few unchanged-cut controls"
        );
        assert!(
            no_cut_delta_max <= MAX_NO_CUT_DELTA_RMSE,
            "unchanged LoD cut was not temporally quiet: max={no_cut_delta_max}, gate={MAX_NO_CUT_DELTA_RMSE}, events={events:?}"
        );
        assert!(
            cut_delta_max <= MAX_CUT_DELTA_RMSE,
            "LoD cut event exceeded the fixed temporal residual gate: max={cut_delta_max}, gate={MAX_CUT_DELTA_RMSE}, events={events:?}"
        );
        assert!(
            no_cut_curvature_max <= MAX_NO_CUT_CURVATURE_RMSE,
            "unchanged LoD cut exceeded the fixed curvature gate: max={no_cut_curvature_max}, gate={MAX_NO_CUT_CURVATURE_RMSE}"
        );
        assert!(
            cut_curvature_max <= MAX_CUT_CURVATURE_RMSE,
            "LoD cut event exceeded the fixed curvature gate: max={cut_curvature_max}, gate={MAX_CUT_CURVATURE_RMSE}, events={events:?}"
        );
    }

    #[test]
    fn legacy_categorical_runtime_bounds_large_camera_jump_event_energy() {
        const WIDTH: u32 = 80;
        const HEIGHT: u32 = 80;
        const QUALITY: f32 = 0.60;
        const FOREGROUND_ALPHA: f32 = 1.0 / 255.0;
        const MAX_STAGGERED_EVENT_RMSE: f64 = 0.06;

        #[derive(Debug)]
        struct StaggeredEvent {
            frame_index: usize,
            previous_hash: u64,
            next_hash: u64,
            previous_nodes: usize,
            next_nodes: usize,
            substitutions: usize,
            changed_gaussians: u64,
            atomic_budget_overshoot: u64,
            foreground_rgb_rmse: f64,
        }

        let scene = LodTestScene::checkerboard_facade(24, 16);
        let lod = build_planar_3d_lod(
            &scene.cloud(),
            GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 8,
                support_sigma: 3.0,
            },
        )
        .expect("runtime temporal checkerboard hierarchy builds");
        let hierarchy = ManifestLodHierarchy::new(&lod.manifest)
            .expect("runtime temporal checkerboard manifest is valid");
        let mut settings = GaussianLodSettings {
            quality: QUALITY,
            frustum_culling: false,
            ..Default::default()
        };
        configure_fully_resident_temporal_budgets(&mut settings, &lod);
        let streaming = GaussianStreamingSettings {
            max_concurrent_requests: lod.manifest.header.page_count,
            ..Default::default()
        };
        let mut runtime = temporal_runtime(&lod, &settings, &streaming);

        let mut near_camera = scene.camera;
        near_camera.position = Vec3::new(0.0, 0.0, 6.0);
        near_camera.viewport = [WIDTH, HEIGHT];
        let mut far_camera = near_camera;
        far_camera.position.z = 48.0;
        let near_view = temporal_view(near_camera, HEIGHT);
        let far_view = temporal_view(far_camera, HEIGHT);

        // Prime the coarse representations with an independent view, then
        // remove it. The measured view starts from the fine near cut while the
        // cache retains both sides of the jump, isolating production stagger
        // from transport or residency delay.
        let priming_view = LodRuntimeViewId(1);
        settle_runtime_view(&mut runtime, priming_view, far_view, &settings, &streaming);
        assert!(runtime.remove_view(priming_view).unwrap());
        let measured_view = LodRuntimeViewId(2);
        let near_nodes = settle_runtime_view(
            &mut runtime,
            measured_view,
            near_view,
            &settings,
            &streaming,
        );

        let immediate = select_frontier_with_previous(
            &hierarchy,
            &AllResident,
            far_view,
            &settings,
            &near_nodes,
        )
        .expect("immediate far selector target succeeds");
        assert_ne!(
            immediate.nodes, near_nodes,
            "runtime stagger fixture must exercise a large coarsening jump"
        );

        // Measure the topology energy of applying the same canonical jump all
        // at once. This is additive over indivisible parent/child cohorts and,
        // unlike endpoint image RMSE, cannot hide work through cancellation
        // between spatially different substitutions.
        let (immediate_topology_energy, immediate_substitutions) = {
            let mut nodes = near_nodes.clone();
            let mut changed_gaussians = 0_u64;
            let mut substitution_count = 0_usize;
            for _ in 0..=lod.manifest.nodes.len() {
                if nodes == immediate.nodes {
                    break;
                }
                let substitutions =
                    temporal_substitution_candidates(&hierarchy, &nodes, &immediate.nodes)
                        .expect("immediate topology substitutions are valid");
                assert!(
                    !substitutions.is_empty(),
                    "immediate topology jump became disconnected: current={nodes:?}, target={:?}",
                    immediate.nodes
                );
                let eligible = substitutions
                    .iter()
                    .map(|substitution| substitution.key)
                    .collect::<std::collections::BTreeSet<_>>();
                let current_active = gather_frontier_gaussians(&lod, &nodes)
                    .expect("immediate topology frontier payload is valid")
                    .len() as u64;
                let step = apply_temporal_substitution_step(
                    &nodes,
                    &immediate.nodes,
                    current_active,
                    &substitutions,
                    &eligible,
                    |_| true,
                    LodTemporalStepBudget {
                        max_active_gaussians: settings.budgets.max_active_gaussians,
                        max_changed_gaussians: u64::MAX,
                        max_substitutions: usize::MAX,
                    },
                )
                .expect("unbounded immediate topology step succeeds");
                assert!(
                    !step.substitutions.is_empty(),
                    "unbounded immediate topology step made no progress"
                );
                changed_gaussians = changed_gaussians
                    .checked_add(step.changed_gaussians)
                    .expect("immediate topology energy fits u64");
                substitution_count += step.substitutions.len();
                nodes = step.nodes;
            }
            assert_eq!(
                nodes, immediate.nodes,
                "unbounded topology plan did not reach the canonical target"
            );
            (changed_gaussians, substitution_count)
        };
        assert!(
            immediate_substitutions >= 2,
            "fixture must exercise multiple indivisible topology cohorts"
        );
        let mut exact_settings = settings.clone();
        exact_settings.quality = 1.0;
        let exact = select_frontier(&hierarchy, &AllResident, far_view, &exact_settings)
            .expect("far exact selector succeeds");
        let exact_image = render_frontier(&lod, &exact.nodes, far_camera, WIDTH, HEIGHT);
        let near_image = render_frontier(&lod, &near_nodes, far_camera, WIDTH, HEIGHT);
        let immediate_image = render_frontier(&lod, &immediate.nodes, far_camera, WIDTH, HEIGHT);
        let immediate_metrics = compare_temporal_deltas(
            &exact_image,
            &exact_image,
            &near_image,
            &immediate_image,
            FOREGROUND_ALPHA,
        )
        .expect("immediate jump metrics are valid");
        assert!(immediate_metrics.foreground_pixels > 0);

        let mut previous_nodes = near_nodes;
        let mut previous_image = near_image;
        let mut staggered_events = Vec::new();
        let mut complete_cuts = std::collections::BTreeSet::from([previous_nodes.clone()]);
        let mut reached_target = false;
        // A pre-ABI16 package has no view-blend payload, so compatibility
        // rendering advances complete categorical cohorts. Bound this trace by
        // hierarchy structure rather than a historical fixed frame count; the
        // cubic quality authority may legitimately expose more adjacent cuts.
        let max_trace_frames = lod.manifest.nodes.len().saturating_mul(2).saturating_add(1);
        for frame_index in 0..max_trace_frames {
            let frame = runtime
                .update_view(measured_view, far_view, &settings, &streaming)
                .expect("staggered runtime update succeeds");
            assert!(
                frame.frontier().requested_nodes.is_empty(),
                "primed stagger trace unexpectedly requested residency: {:?}",
                frame.frontier().requested_nodes
            );
            frame
                .candidate_frontier(settings.max_active_gaussians_u32())
                .expect("every staggered runtime state is one complete resident cut");
            let nodes = frame.frontier().nodes.clone();
            let image = render_frontier(&lod, &nodes, far_camera, WIDTH, HEIGHT);
            assert_eq!(
                frame.candidate_count(),
                gather_frontier_gaussians(&lod, &nodes).unwrap().len() as u64,
                "runtime stagger cut count differs from its complete payload"
            );
            complete_cuts.insert(nodes.clone());
            if nodes != previous_nodes {
                let transition = frame
                    .temporal_transition()
                    .expect("every changed cut carries its topology transaction");
                assert_eq!(
                    transition.mode(),
                    LodTemporalTransitionMode::BoundedHardCohort,
                    "the in-memory ABI14 fixture should exercise the categorical fallback"
                );
                assert!(transition.morph().is_none());
                let substitution_energy = transition
                    .substitutions()
                    .iter()
                    .try_fold(0_u64, |energy, substitution| {
                        energy.checked_add(substitution.changed_gaussians())
                    })
                    .expect("staggered topology energy fits u64");
                assert_eq!(
                    transition.changed_gaussians(),
                    substitution_energy,
                    "runtime transition energy must equal its exact cohort payload"
                );

                let mut reconstructed = previous_nodes
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                for substitution in transition.substitutions() {
                    assert!(
                        substitution
                            .previous_nodes
                            .iter()
                            .all(|node| reconstructed.contains(node)),
                        "runtime transition does not apply to its advertised previous cut"
                    );
                    for node in &substitution.previous_nodes {
                        assert!(reconstructed.remove(node));
                    }
                    for &node in &substitution.next_nodes {
                        assert!(reconstructed.insert(node));
                    }
                }
                assert_eq!(
                    reconstructed.into_iter().collect::<Vec<_>>(),
                    nodes,
                    "runtime transition payload does not reconstruct its emitted cut"
                );

                let metrics = compare_temporal_deltas(
                    &exact_image,
                    &exact_image,
                    &previous_image,
                    &image,
                    FOREGROUND_ALPHA,
                )
                .expect("staggered event metrics are valid");
                staggered_events.push(StaggeredEvent {
                    frame_index,
                    previous_hash: stable_frontier_hash(&previous_nodes),
                    next_hash: stable_frontier_hash(&nodes),
                    previous_nodes: previous_nodes.len(),
                    next_nodes: nodes.len(),
                    substitutions: transition.substitutions().len(),
                    changed_gaussians: transition.changed_gaussians(),
                    atomic_budget_overshoot: transition.atomic_budget_overshoot(),
                    foreground_rgb_rmse: metrics.foreground_rgb_rmse,
                });
            }
            previous_nodes = nodes;
            previous_image = image;
            if previous_nodes == immediate.nodes
                && frame.selection_stable()
                && !frame.temporal_transition_applied()
            {
                reached_target = true;
                break;
            }
        }

        let staggered_max = staggered_events
            .iter()
            .map(|event| event.foreground_rgb_rmse)
            .fold(0.0_f64, f64::max);
        let staggered_topology_energy = staggered_events
            .iter()
            .map(|event| event.changed_gaussians)
            .sum::<u64>();
        let staggered_peak_topology_energy = staggered_events
            .iter()
            .map(|event| event.changed_gaussians)
            .max()
            .unwrap_or(0);
        let staggered_substitutions = staggered_events
            .iter()
            .map(|event| event.substitutions)
            .sum::<usize>();
        let event_trace = staggered_events
            .iter()
            .map(|event| {
                (
                    event.frame_index,
                    event.previous_hash,
                    event.next_hash,
                    event.previous_nodes,
                    event.next_nodes,
                    event.substitutions,
                    event.changed_gaussians,
                    event.atomic_budget_overshoot,
                    event.foreground_rgb_rmse,
                )
            })
            .collect::<Vec<_>>();
        eprintln!(
            "legacy categorical LoD cohort trace: immediate_rmse={:.6} immediate_topology_energy={} bounded_peak_topology_energy={} bounded_max_rmse={:.6} complete_cuts={} events={event_trace:?}",
            immediate_metrics.foreground_rgb_rmse,
            immediate_topology_energy,
            staggered_peak_topology_energy,
            staggered_max,
            complete_cuts.len(),
        );
        assert!(
            reached_target,
            "legacy categorical runtime did not reach the immediate target"
        );
        assert!(
            complete_cuts.len() >= 3 && staggered_events.len() >= 2,
            "legacy categorical coarsening did not distribute the jump: {event_trace:?}"
        );
        assert!(
            immediate_metrics.foreground_rgb_rmse.is_finite()
                && immediate_metrics.foreground_rgb_rmse > 0.0,
            "canonical topology jump must alter the rendered endpoint"
        );
        assert_eq!(
            staggered_topology_energy, immediate_topology_energy,
            "bounded categorical runtime must account for the exact canonical topology energy"
        );
        assert_eq!(
            staggered_substitutions, immediate_substitutions,
            "bounded categorical runtime must apply every canonical topology cohort exactly once"
        );
        assert!(
            staggered_peak_topology_energy < immediate_topology_energy,
            "legacy categorical runtime did not bound the immediate topology energy: immediate={immediate_topology_energy}, peak={staggered_peak_topology_energy}, events={event_trace:?}"
        );
        assert!(
            staggered_max <= MAX_STAGGERED_EVENT_RMSE,
            "legacy categorical runtime exceeded the fixed absolute event gate: max={staggered_max}, gate={MAX_STAGGERED_EVENT_RMSE}, events={event_trace:?}"
        );
    }

    fn configure_fully_resident_temporal_budgets(
        settings: &mut GaussianLodSettings,
        lod: &PlanarGaussian3dLod,
    ) {
        settings.budgets.max_active_gaussians = lod.manifest.header.source_gaussian_count;
        settings.budgets.max_resident_gaussians = lod.manifest.header.stored_gaussian_count;
        settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
        settings.budgets.max_resident_pages = lod.manifest.header.page_count;
        settings.budgets.max_pending_requests = lod.manifest.header.page_count;
        settings.budgets.max_requests_per_frame = lod.manifest.header.page_count;
        settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
    }

    fn temporal_runtime(
        lod: &PlanarGaussian3dLod,
        settings: &GaussianLodSettings,
        streaming: &GaussianStreamingSettings,
    ) -> LodStreamingRuntime<MemoryPageTransport> {
        let mut manifest = lod.manifest.clone();
        let mut transport = MemoryPageTransport::default();
        for page in &lod.pages {
            let encoded = encode_page(page).expect("temporal runtime page encodes");
            let descriptor = manifest
                .pages
                .iter_mut()
                .find(|descriptor| descriptor.id == page.id)
                .expect("temporal runtime page descriptor exists");
            descriptor.storage = Some(LodPageStorage {
                uri: format!("memory://temporal-{}", page.id.0),
                byte_range: None,
                encoded_len: encoded.len() as u64,
            });
            transport.insert(page.id, encoded);
        }
        LodStreamingRuntime::new(manifest, transport, settings, streaming)
            .expect("temporal runtime constructs")
    }

    fn settle_runtime_view(
        runtime: &mut LodStreamingRuntime<MemoryPageTransport>,
        view_id: LodRuntimeViewId,
        view: LodView,
        settings: &GaussianLodSettings,
        streaming: &GaussianStreamingSettings,
    ) -> Vec<LodNodeId> {
        for _ in 0..512 {
            let frame = runtime
                .update_view(view_id, view, settings, streaming)
                .expect("temporal runtime warmup succeeds");
            assert!(frame.failed_pages().is_empty());
            assert!(frame.preprocess_failed_pages().is_empty());
            if frame.has_complete_resident_cut()
                && frame.frontier().requested_nodes.is_empty()
                && frame.in_flight_requests() == 0
                && frame.queued_requests() == 0
                && frame.selection_stable()
                && !frame.temporal_transition_applied()
            {
                frame
                    .candidate_frontier(settings.max_active_gaussians_u32())
                    .expect("settled runtime view has a complete resident cut");
                return frame.frontier().nodes.clone();
            }
        }
        panic!("temporal runtime view did not settle within its bounded warmup");
    }

    fn temporal_view(camera: LodTestCamera, height: u32) -> LodView {
        LodView::perspective(
            camera.position,
            height as f32,
            match camera.projection {
                LodProjection::Perspective {
                    vertical_fov_radians,
                } => vertical_fov_radians,
                LodProjection::Orthographic { .. } => unreachable!(),
            },
            camera.near,
        )
    }

    fn render_frontier(
        lod: &PlanarGaussian3dLod,
        nodes: &[LodNodeId],
        camera: LodTestCamera,
        width: u32,
        height: u32,
    ) -> Vec<[f32; 4]> {
        let gaussians = gather_frontier_gaussians(lod, nodes).expect("temporal frontier resolves");
        render_linear_gaussians(&gaussians, camera, width, height)
            .expect("temporal frontier renders")
    }

    fn stable_frontier_hash(nodes: &[LodNodeId]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        nodes.iter().fold(FNV_OFFSET, |hash, node| {
            node.0.to_le_bytes().into_iter().fold(hash, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
            })
        })
    }

    fn finite_max(values: &[f64]) -> f64 {
        values.iter().copied().fold(0.0_f64, f64::max)
    }
}
