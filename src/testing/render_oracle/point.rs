//! Bounded deterministic expectation of the corrected GPS Poisson process.
//!
//! Each pixel integrates -log(1-alpha(x)) with the shared C1 three-sigma tail,
//! then uses its occupancy probability 1-exp(-integral). Independent occupancies
//! compose in reverse-Z center-depth order, with the lowest input index winning
//! exact depth ties. This is an expectation, not a finite-SPP noise prediction.
//!
//! Intervals bound integration error for the projected f32 parameters, with an
//! explicit f64 arithmetic allowance. They do not certify shader transcendental,
//! radial-inversion, Poisson, or pseudorandom approximation errors. Those remain
//! measurable differences between this ideal process and the GPU implementation.
//! The admitted profile is identity transform, unit global scale/opacity,
//! DrawMode::All, transparent background, and no opaque mesh depth.

use std::{cmp::Ordering, collections::BinaryHeap, fmt};

use super::{
    Gaussian3d, GaussianColorSpace, LodTestCamera, ProductionWorldSupport, Vec2, Vec3,
    production_projected_geometry, production_projection,
    production_spherical_harmonics_linear_color,
};
use crate::{
    render::support::gaussian_support_weight_and_derivative, stream::hierarchy::LodViewProjection,
};

#[derive(Clone, Copy, Debug)]
pub struct PointOracleOptions {
    pub max_pixel_pairs: u64,
    pub max_integration_cells: u64,
    pub max_cells_per_integral: u32,
    pub max_depth: u32,
    /// Absolute interval width requested for one Gaussian/pixel integral.
    pub integral_tolerance: f64,
    /// All final RGBA intervals must be this narrow to claim convergence.
    pub pixel_tolerance: f64,
    /// Absolute final-image allowance for f64 quadrature/compositing arithmetic.
    pub numerical_allowance: f64,
}

impl Default for PointOracleOptions {
    fn default() -> Self {
        Self {
            max_pixel_pairs: 1_000_000,
            max_integration_cells: 2_000_000,
            max_cells_per_integral: 4096,
            max_depth: 16,
            integral_tolerance: 1e-4,
            pixel_tolerance: 1e-3,
            numerical_allowance: 1e-9,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PointOracleWork {
    pub projected_gaussians: u64,
    pub pixel_pairs: u64,
    pub integration_cells: u64,
    pub limited_integrals: u64,
    pub maximum_live_cells: usize,
}

#[derive(Debug)]
pub enum PointOracleError {
    InvalidInput(&'static str),
    Inconclusive {
        reason: &'static str,
        work: PointOracleWork,
    },
}

impl fmt::Display for PointOracleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(reason) => write!(formatter, "GPS oracle input: {reason}"),
            Self::Inconclusive { reason, .. } => {
                write!(formatter, "GPS oracle inconclusive: {reason}")
            }
        }
    }
}

impl std::error::Error for PointOracleError {}

#[derive(Debug)]
pub struct PointExpectation {
    pub lower: Vec<[f64; 4]>,
    pub upper: Vec<[f64; 4]>,
    /// False means the complete image is bounded but insufficiently resolved.
    pub converged: bool,
    pub maximum_interval_width: f64,
    pub numerical_allowance: f64,
    pub work: PointOracleWork,
}

impl PointExpectation {
    /// Display convenience only; f32 conversion adds its usual rounding error.
    pub fn midpoint_rgba(&self) -> Vec<[f32; 4]> {
        self.lower
            .iter()
            .zip(&self.upper)
            .map(|(lower, upper)| {
                std::array::from_fn(|channel| ((lower[channel] + upper[channel]) * 0.5) as f32)
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
struct Projected {
    mean: [f64; 2],
    /// Physical-pixel inverse covariance, derived from shader-rounded Cholesky.
    conic: [f64; 3],
    radius: [f64; 2],
    opacity: f64,
    depth: f32,
    input_index: usize,
    color: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    rectangle: [f64; 4],
    lower: f64,
    upper: f64,
    depth: u32,
    sequence: u64,
}

impl Cell {
    fn gap(self) -> f64 {
        self.upper - self.lower
    }
}
impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Cell {}
impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gap()
            .total_cmp(&other.gap())
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

fn intensity(opacity: f64, radius_squared: f64) -> f64 {
    -(-opacity * gaussian_support_weight_and_derivative(radius_squared, 9.0).0).ln_1p()
}

/// Convex SPD quadratic extrema on a rectangle: maximum at a corner; minimum
/// at the origin if contained, or one of the four constrained edge minima.
fn quadratic_bounds(conic: [f64; 3], rectangle: [f64; 4]) -> [f64; 2] {
    let [a, b, c] = conic;
    let [x0, y0, x1, y1] = rectangle;
    let q = |x: f64, y: f64| a * x * x + 2.0 * b * x * y + c * y * y;
    let corners = [q(x0, y0), q(x0, y1), q(x1, y0), q(x1, y1)];
    let mut lower = if x0 <= 0.0 && x1 >= 0.0 && y0 <= 0.0 && y1 >= 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    for x in [x0, x1] {
        lower = lower.min(q(x, (-b * x / c).clamp(y0, y1)));
    }
    for y in [y0, y1] {
        lower = lower.min(q((-b * y / a).clamp(x0, x1), y));
    }
    let x = x0.abs().max(x1.abs());
    let y = y0.abs().max(y1.abs());
    let guard = 128.0 * f64::EPSILON * (1.0 + a * x * x + 2.0 * b.abs() * x * y + c * y * y);
    [
        (lower - guard).max(0.0),
        corners.into_iter().fold(0.0, f64::max) + guard,
    ]
}

fn cell(projected: &Projected, rectangle: [f64; 4], depth: u32, sequence: u64) -> Cell {
    let [x0, y0, x1, y1] = rectangle;
    let relative = [
        x0 - projected.mean[0],
        y0 - projected.mean[1],
        x1 - projected.mean[0],
        y1 - projected.mean[1],
    ];
    let [q_min, q_max] = quadratic_bounds(projected.conic, relative);
    let area = (x1 - x0) * (y1 - y0);
    let mut lower = if q_max <= 9.0 {
        area * intensity(projected.opacity, q_max)
    } else {
        0.0
    };
    let mut upper = if q_min <= 9.0 {
        area * intensity(projected.opacity, q_min)
    } else {
        0.0
    };
    if q_min < 9.0 {
        // Tensor-product midpoint error is bounded by area*(h_x² H_xx +
        // h_y² H_yy)/24. The C1 tail has bounded piecewise second derivatives,
        // so the remainder bound also holds across its q=8 and q=9 boundaries.
        let [a, b, c] = projected.conic;
        let x = relative[0].abs().max(relative[2].abs());
        let y = relative[1].abs().max(relative[3].abs());
        let g = projected.opacity * (-0.5 * q_min).exp();
        let denominator = 1.0 - g;
        let (first, second) = if q_max <= 8.0 {
            (0.5 * g / denominator, 0.25 * g / denominator.powi(2))
        } else {
            // For g(q)=opacity*exp(-q/2)*T(q): |T'|<=1.5,
            // |T''|<=6, hence |g'|<=2g0 and |g''|<=7.75g0.
            // lambda=-log(1-g) then has these conservative q derivatives.
            let first = 2.0 * g / denominator;
            (first, 7.75 * g / denominator + first * first)
        };
        let h_xx = 4.0 * second * (a * x + b.abs() * y).powi(2) + 2.0 * first * a;
        let h_yy = 4.0 * second * (b.abs() * x + c * y).powi(2) + 2.0 * first * c;
        let dx = (relative[0] + relative[2]) * 0.5;
        let dy = (relative[1] + relative[3]) * 0.5;
        let midpoint_q = quadratic_bounds(projected.conic, [dx, dy, dx, dy]);
        let midpoint_lower = area * intensity(projected.opacity, midpoint_q[1]);
        let midpoint_upper = area * intensity(projected.opacity, midpoint_q[0]);
        let error = area * ((x1 - x0).powi(2) * h_xx + (y1 - y0).powi(2) * h_yy) / 24.0
            * (1.0 + 256.0 * f64::EPSILON);
        lower = lower.max(midpoint_lower - error);
        upper = upper.min(midpoint_upper + error);
    }
    let guard = 256.0 * f64::EPSILON * area * (1.0 + intensity(projected.opacity, 0.0));
    Cell {
        rectangle,
        lower: (lower - guard).max(0.0),
        upper: upper + guard,
        depth,
        sequence,
    }
}

fn integral(
    projected: &Projected,
    rectangle: [f64; 4],
    options: PointOracleOptions,
    work: &mut PointOracleWork,
) -> Result<[f64; 2], PointOracleError> {
    let mut cells = BinaryHeap::new();
    let mut created = 1_u32;
    if work.integration_cells >= options.max_integration_cells {
        return Err(PointOracleError::Inconclusive {
            reason: "global integration-cell limit",
            work: *work,
        });
    }
    work.integration_cells += 1;
    let first = cell(projected, rectangle, 0, 0);
    let mut gap = first.gap();
    cells.push(first);
    work.maximum_live_cells = work.maximum_live_cells.max(cells.len());
    while gap > options.integral_tolerance {
        let parent = *cells.peek().expect("integral retains its covering cells");
        if parent.depth >= options.max_depth
            || created.saturating_add(4) > options.max_cells_per_integral
        {
            work.limited_integrals += 1;
            break;
        }
        if work.integration_cells.saturating_add(4) > options.max_integration_cells {
            return Err(PointOracleError::Inconclusive {
                reason: "global integration-cell limit",
                work: *work,
            });
        }
        cells.pop();
        gap -= parent.gap();
        let [x0, y0, x1, y1] = parent.rectangle;
        let xm = (x0 + x1) * 0.5;
        let ym = (y0 + y1) * 0.5;
        for rectangle in [
            [x0, y0, xm, ym],
            [xm, y0, x1, ym],
            [x0, ym, xm, y1],
            [xm, ym, x1, y1],
        ] {
            let child = cell(projected, rectangle, parent.depth + 1, u64::from(created));
            created += 1;
            work.integration_cells += 1;
            gap += child.gap();
            cells.push(child);
        }
        work.maximum_live_cells = work.maximum_live_cells.max(cells.len());
    }
    work.maximum_live_cells = work.maximum_live_cells.max(cells.len());
    // Re-sum the complete cover, avoiding accumulated subtractive-update error.
    let result = cells.iter().fold([0.0, 0.0], |sum, cell| {
        [sum[0] + cell.lower, sum[1] + cell.upper]
    });
    let guard = 256.0 * f64::EPSILON * f64::from(created) * (1.0 + result[1]);
    Ok([(result[0] - guard).max(0.0), result[1] + guard])
}

fn validate_options(options: PointOracleOptions) -> Result<(), PointOracleError> {
    if !(1..=16_000_000).contains(&options.max_pixel_pairs)
        || !(1..=16_000_000).contains(&options.max_integration_cells)
        || !(1..=65_536).contains(&options.max_cells_per_integral)
        || options.max_depth > 24
        || !options.integral_tolerance.is_finite()
        || options.integral_tolerance <= 0.0
        || !options.pixel_tolerance.is_finite()
        || options.pixel_tolerance <= 0.0
        || !options.numerical_allowance.is_finite()
        || options.numerical_allowance < 1e-12
    {
        return Err(PointOracleError::InvalidInput(
            "invalid integration tolerance/work/depth admission",
        ));
    }
    Ok(())
}

/// Render a complete bounded local input; budget exhaustion returns an explicit
/// inconclusive error, never a prefix image masquerading as a complete teacher.
pub fn render_point_expectation(
    gaussians: &[Gaussian3d],
    camera: LodTestCamera,
    color_space: GaussianColorSpace,
    options: PointOracleOptions,
) -> Result<PointExpectation, PointOracleError> {
    validate_options(options)?;
    let [width, height] = camera.viewport;
    if gaussians.len() > 100_000
        || width == 0
        || height == 0
        || width > 512
        || height > 512
        || !camera.near.is_finite()
        || !camera.far.is_finite()
        || camera.near <= 0.0
        || camera.far <= camera.near
        || !camera.position.is_finite()
    {
        return Err(PointOracleError::InvalidInput(
            "requires <=100k records, <=512² pixels and a finite camera",
        ));
    }
    let (forward, right, up) = camera.basis().map_err(PointOracleError::InvalidInput)?;
    let projection = production_projection(camera.projection, camera.viewport)
        .map_err(|_| PointOracleError::InvalidInput("invalid projection"))?;
    let world_support = ProductionWorldSupport::new(camera, camera.viewport)
        .map_err(|_| PointOracleError::InvalidInput("invalid deployment support projection"))?;
    let view = camera.lod_view(Vec2::new(width as f32, height as f32));
    let LodViewProjection::Matrix {
        clip_from_world, ..
    } = view.projection
    else {
        return Err(PointOracleError::InvalidInput(
            "camera did not provide a projection matrix",
        ));
    };
    let mut projected = Vec::with_capacity(gaussians.len());
    for (input_index, gaussian) in gaussians.iter().enumerate() {
        if !gaussian.scale_opacity.opacity.is_finite()
            || gaussian
                .position_visibility
                .position
                .iter()
                .any(|value| !value.is_finite())
            || gaussian
                .scale_opacity
                .scale
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || gaussian
                .rotation
                .rotation
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(PointOracleError::InvalidInput(
                "nonfinite or invalid Gaussian geometry",
            ));
        }
        if !world_support.contains(gaussian, 3.0) {
            continue;
        }
        let Some(geometry) = production_projected_geometry(
            gaussian, camera, width, height, forward, right, up, projection,
        ) else {
            continue;
        };
        let opacity = (gaussian.scale_opacity.opacity * geometry.opacity_scale).clamp(0.0, 0.999);
        if opacity <= 0.0 {
            continue;
        }
        let position = Vec3::from_array(gaussian.position_visibility.position);
        let clip = clip_from_world * position.extend(1.0);
        let depth = clip.z / clip.w;
        if !(clip.w > 0.0 && depth > 0.0 && depth <= 1.0) {
            continue;
        }
        let [a, b, c] = geometry.covariance.map(|value| value * 0.25);
        let determinant = a * c - b * b;
        if !(a > 0.0 && determinant > 0.0) {
            continue;
        }
        let bx = a.sqrt();
        let bxy = b / bx;
        let by = (determinant / a).sqrt();
        if ![bx, bxy, by].into_iter().all(f32::is_finite) || by <= 0.0 {
            continue;
        }
        let [bx, bxy, by] = [bx, bxy, by].map(f64::from);
        let inverse_x = 1.0 / bx;
        let inverse_y = 1.0 / by;
        let cross = -bxy * inverse_x * inverse_y;
        let color = production_spherical_harmonics_linear_color(
            geometry.relative.normalize(),
            &gaussian.spherical_harmonic,
            color_space,
        );
        if !color.into_iter().all(f32::is_finite) {
            return Err(PointOracleError::InvalidInput("nonfinite linear SH color"));
        }
        projected.push(Projected {
            mean: [f64::from(geometry.center.x), f64::from(geometry.center.y)],
            conic: [
                inverse_x * inverse_x + cross * cross,
                cross * inverse_y,
                inverse_y * inverse_y,
            ],
            radius: [3.0 * bx, 3.0 * (bxy * bxy + by * by).sqrt()],
            opacity: f64::from(opacity),
            depth,
            input_index,
            color: color.map(f64::from),
        });
    }
    render_projected(projected, camera.viewport, options)
}

fn render_projected(
    mut projected: Vec<Projected>,
    viewport: [u32; 2],
    options: PointOracleOptions,
) -> Result<PointExpectation, PointOracleError> {
    let [width, height] = viewport;
    // Back to front: reverse-Z is increasing toward the camera, and the lowest
    // input index is in front among exactly equal stored depth values.
    projected.sort_by(|a, b| {
        a.depth
            .total_cmp(&b.depth)
            .then_with(|| b.input_index.cmp(&a.input_index))
    });
    let mut result = PointExpectation {
        lower: vec![[0.0; 4]; (width * height) as usize],
        upper: vec![[0.0; 4]; (width * height) as usize],
        converged: false,
        maximum_interval_width: 0.0,
        numerical_allowance: options.numerical_allowance,
        work: PointOracleWork {
            projected_gaussians: projected.len() as u64,
            ..Default::default()
        },
    };
    for gaussian in &projected {
        let x0 = (gaussian.mean[0] - gaussian.radius[0])
            .floor()
            .clamp(0.0, f64::from(width)) as u32;
        let y0 = (gaussian.mean[1] - gaussian.radius[1])
            .floor()
            .clamp(0.0, f64::from(height)) as u32;
        let x1 = (gaussian.mean[0] + gaussian.radius[0])
            .ceil()
            .clamp(0.0, f64::from(width)) as u32;
        let y1 = (gaussian.mean[1] + gaussian.radius[1])
            .ceil()
            .clamp(0.0, f64::from(height)) as u32;
        let visits = u64::from(x1 - x0) * u64::from(y1 - y0);
        if result.work.pixel_pairs.saturating_add(visits) > options.max_pixel_pairs {
            return Err(PointOracleError::Inconclusive {
                reason: "Gaussian/pixel pair limit",
                work: result.work,
            });
        }
        for y in y0..y1 {
            for x in x0..x1 {
                result.work.pixel_pairs += 1;
                let bounds = integral(
                    gaussian,
                    [
                        f64::from(x),
                        f64::from(y),
                        f64::from(x + 1),
                        f64::from(y + 1),
                    ],
                    options,
                    &mut result.work,
                )?;
                let occupancy = bounds.map(|value| -(-value).exp_m1());
                let index = (y * width + x) as usize;
                for channel in 0..4 {
                    let color = if channel == 3 {
                        1.0
                    } else {
                        gaussian.color[channel]
                    };
                    let mut lower = f64::INFINITY;
                    let mut upper = f64::NEG_INFINITY;
                    for prior in [result.lower[index][channel], result.upper[index][channel]] {
                        for alpha in occupancy {
                            let composed = color * alpha + prior * (1.0 - alpha);
                            lower = lower.min(composed);
                            upper = upper.max(composed);
                        }
                    }
                    let guard =
                        256.0 * f64::EPSILON * (1.0 + color.abs() + lower.abs().max(upper.abs()));
                    result.lower[index][channel] = lower - guard;
                    result.upper[index][channel] = upper + guard;
                }
            }
        }
    }
    for (lower, upper) in result.lower.iter_mut().zip(&mut result.upper) {
        for channel in 0..4 {
            lower[channel] -= options.numerical_allowance;
            upper[channel] += options.numerical_allowance;
            result.maximum_interval_width = result
                .maximum_interval_width
                .max(upper[channel] - lower[channel]);
        }
    }
    result.converged = result.maximum_interval_width <= options.pixel_tolerance;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::point::math::dilog;

    fn splat(opacity: f64, sigma: f64) -> Projected {
        Projected {
            mean: [0.5, 0.5],
            conic: [1.0 / (sigma * sigma), 0.0, 1.0 / (sigma * sigma)],
            radius: [3.0 * sigma; 2],
            opacity,
            depth: 0.5,
            input_index: 0,
            color: [1.0, 0.0, 0.0],
        }
    }

    #[test]
    fn point_integral_bounds_narrow_opaque_pixel_and_reports_insufficient_work() {
        let gaussian = splat(0.999, 0.04);
        let options = PointOracleOptions {
            max_cells_per_integral: 65_536,
            max_depth: 24,
            integral_tolerance: 1e-5,
            ..Default::default()
        };
        let result = render_projected(vec![gaussian.clone()], [1, 1], options).unwrap();
        // An independent radial reference integrates the unchanged core in
        // closed form and bounds the monotone tapered annulus by rectangles.
        // This is only 128 one-dimensional panels, not a second pixel oracle.
        let core = 2.0 * (dilog(0.999) - dilog(0.999 * (-4.0_f64).exp()));
        let (mut annulus_lower, mut annulus_upper) = (0.0, 0.0);
        for i in 0..128 {
            annulus_upper += intensity(0.999, 8.0 + f64::from(i) / 128.0) / 128.0;
            annulus_lower += intensity(0.999, 8.0 + f64::from(i + 1) / 128.0) / 128.0;
        }
        let scale = std::f64::consts::PI * 0.04_f64.powi(2);
        let exact_lower = -(-(core + annulus_lower) * scale).exp_m1();
        let exact_upper = -(-(core + annulus_upper) * scale).exp_m1();
        assert!(exact_upper - exact_lower < 1e-6);
        assert!(result.lower[0][3] <= exact_upper && exact_lower <= result.upper[0][3]);
        assert!(result.converged, "{:?}", result.work);
        assert!(
            exact_upper < 0.02 && gaussian.opacity > 0.99,
            "pixel-center alpha is a biased occupancy teacher"
        );
        let limited = render_projected(
            vec![gaussian],
            [1, 1],
            PointOracleOptions {
                max_cells_per_integral: 1,
                ..options
            },
        )
        .unwrap();
        assert!(!limited.converged);
        assert_eq!(limited.work.limited_integrals, 1);

        // The public path also preserves production filtering: a narrow,
        // authored-opaque source loses peak opacity through determinant
        // compensation, then still requires physical-pixel integration.
        let mut source = crate::testing::LodTestScene::screen_space_ladder().gaussians[0].gaussian;
        source.position_visibility.position = [0.0; 3];
        source.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
        source.scale_opacity.scale = [1.0; 3];
        source.scale_opacity.opacity = 1.0;
        let camera = LodTestCamera {
            viewport: [1, 1],
            ..Default::default()
        };
        let (forward, right, up) = camera.basis().unwrap();
        let geometry = production_projected_geometry(
            &source,
            camera,
            1,
            1,
            forward,
            right,
            up,
            production_projection(camera.projection, camera.viewport).unwrap(),
        )
        .unwrap();
        let result = render_point_expectation(
            &[source],
            camera,
            GaussianColorSpace::SrgbRec709Display,
            options,
        )
        .unwrap();
        assert!(result.converged);
        assert!(result.upper[0][3] < f64::from(geometry.opacity_scale) * 0.9);
    }

    #[test]
    fn point_expectation_uses_occupancy_center_depth_and_stable_index_ties() {
        let red = splat(0.5, 2.0);
        let mut blue = red.clone();
        blue.color = [0.0, 0.0, 1.0];
        blue.input_index = 1;
        let options = PointOracleOptions {
            integral_tolerance: 1e-6,
            max_cells_per_integral: 65_536,
            ..Default::default()
        };
        let one = render_projected(vec![red.clone()], [1, 1], options).unwrap();
        let p = (one.lower[0][3] + one.upper[0][3]) * 0.5;
        let tied = render_projected(vec![blue.clone(), red.clone()], [1, 1], options).unwrap();
        let expected = [p, 0.0, (1.0 - p) * p, 1.0 - (1.0 - p).powi(2)];
        for (channel, value) in expected.into_iter().enumerate() {
            assert!(tied.lower[0][channel] <= value && value <= tied.upper[0][channel]);
        }
        blue.depth = 0.75;
        let closer = render_projected(vec![red, blue], [1, 1], options).unwrap();
        assert!(
            closer.lower[0][2] > closer.upper[0][0],
            "closer center beats lower input index"
        );
        assert!(matches!(
            render_projected(
                vec![splat(0.5, 2.0)],
                [2, 1],
                PointOracleOptions {
                    max_pixel_pairs: 1,
                    ..options
                }
            ),
            Err(PointOracleError::Inconclusive {
                reason: "Gaussian/pixel pair limit",
                ..
            })
        ));
    }
}
