use std::{collections::VecDeque, fmt};

/// Metrics computed in linear color space. Infinite PSNR denotes identical RGB images.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageMetrics {
    pub psnr_rgb: f64,
    pub foreground_psnr_rgb: f64,
    pub luminance_ssim: f64,
    pub alpha_mae: f64,
    pub foreground_iou: f64,
    pub max_abs_error: f32,
}

/// Signed difference statistics for a temporal image expression.
///
/// RGB aggregates use the union of every input image's foreground mask. Alpha
/// aggregates use the full image, matching [`ImageMetrics::alpha_mae`]. PSNR
/// uses a nominal peak of one and is infinite when the signed RGB expression
/// is exactly zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalResidualMetrics {
    pub signed_rgb_mean: [f64; 3],
    pub foreground_rgb_rmse: f64,
    pub foreground_psnr_rgb: f64,
    pub signed_alpha_mean: f64,
    pub alpha_abs_mean: f64,
    pub max_abs_residual: f32,
    pub foreground_pixels: usize,
}

/// Signed reconstruction error for one spatial region of an image.
///
/// RGB RMSE uses all three channels. The signed luminance and alpha means are
/// `candidate - reference`, so a negative value exposes systematic
/// under-reconstruction rather than hiding it inside an absolute metric.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialResidualMetrics {
    pub pixels: usize,
    pub rgb_rmse: f64,
    pub signed_luminance_mean: f64,
    pub signed_alpha_mean: f64,
    pub alpha_abs_mean: f64,
    pub max_abs_residual: f32,
}

/// Reconstruction error split into a band around logical-node boundaries and
/// the remaining attributed interior.
///
/// `rgb_rmse_enrichment` and `alpha_abs_enrichment` are boundary/interior
/// ratios. Infinite enrichment means the interior is exact while the boundary
/// is not. Callers comparing real scenes should additionally match reference
/// image-gradient distributions because hierarchy boundaries can coincide
/// with genuine scene edges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryBandMetrics {
    pub boundary: SpatialResidualMetrics,
    pub interior: SpatialResidualMetrics,
    pub rgb_rmse_enrichment: f64,
    pub alpha_abs_enrichment: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryMetricsError {
    InvalidDimensions,
    Image(ImageMetricsError),
    LabelLengthMismatch { expected: usize, actual: usize },
    NoBoundaryPixels,
    NoInteriorPixels,
}

impl fmt::Display for BoundaryMetricsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "image dimensions must be non-zero and fit usize"),
            Self::Image(error) => error.fmt(f),
            Self::LabelLengthMismatch { expected, actual } => write!(
                f,
                "node-label length differs from the image: expected={expected}, actual={actual}"
            ),
            Self::NoBoundaryPixels => write!(f, "node labels contain no measurable boundary"),
            Self::NoInteriorPixels => write!(f, "boundary band consumes every attributed pixel"),
        }
    }
}

impl std::error::Error for BoundaryMetricsError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageMetricsError {
    Empty,
    LengthMismatch { reference: usize, candidate: usize },
    NonFinite { pixel: usize, channel: usize },
}

impl fmt::Display for ImageMetricsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "images must contain at least one pixel"),
            Self::LengthMismatch {
                reference,
                candidate,
            } => write!(
                f,
                "image lengths differ: reference={reference}, candidate={candidate}"
            ),
            Self::NonFinite { pixel, channel } => {
                write!(
                    f,
                    "non-finite image value at pixel {pixel}, channel {channel}"
                )
            }
        }
    }
}

impl std::error::Error for ImageMetricsError {}

/// Compare two linear RGBA images.
///
/// `foreground_alpha_threshold` is applied to the union of both alpha masks. RGB values are
/// expected to use a normalized nominal range of `[0, 1]`; values outside the range are accepted
/// (HDR tests need them), but the PSNR peak remains one so runs stay comparable.
pub fn compare_linear_rgba(
    reference: &[[f32; 4]],
    candidate: &[[f32; 4]],
    foreground_alpha_threshold: f32,
) -> Result<ImageMetrics, ImageMetricsError> {
    if reference.is_empty() {
        return Err(ImageMetricsError::Empty);
    }
    if reference.len() != candidate.len() {
        return Err(ImageMetricsError::LengthMismatch {
            reference: reference.len(),
            candidate: candidate.len(),
        });
    }

    let threshold = foreground_alpha_threshold.clamp(0.0, 1.0);
    let mut rgb_squared_error = 0.0_f64;
    let mut foreground_rgb_squared_error = 0.0_f64;
    let mut foreground_rgb_samples = 0_u64;
    let mut alpha_absolute_error = 0.0_f64;
    let mut max_abs_error = 0.0_f32;
    let mut foreground_intersection = 0_u64;
    let mut foreground_union = 0_u64;

    let mut reference_luma = Vec::with_capacity(reference.len());
    let mut candidate_luma = Vec::with_capacity(reference.len());

    for (pixel, (expected, actual)) in reference.iter().zip(candidate).enumerate() {
        for channel in 0..4 {
            if !expected[channel].is_finite() || !actual[channel].is_finite() {
                return Err(ImageMetricsError::NonFinite { pixel, channel });
            }
            max_abs_error = max_abs_error.max((expected[channel] - actual[channel]).abs());
        }

        let expected_foreground = expected[3] > threshold;
        let actual_foreground = actual[3] > threshold;
        foreground_intersection += u64::from(expected_foreground && actual_foreground);
        foreground_union += u64::from(expected_foreground || actual_foreground);

        for channel in 0..3 {
            let error = f64::from(expected[channel] - actual[channel]);
            rgb_squared_error += error * error;
            if expected_foreground || actual_foreground {
                foreground_rgb_squared_error += error * error;
                foreground_rgb_samples += 1;
            }
        }
        alpha_absolute_error += f64::from((expected[3] - actual[3]).abs());

        reference_luma.push(linear_luminance(*expected));
        candidate_luma.push(linear_luminance(*actual));
    }

    let rgb_samples = (reference.len() * 3) as f64;
    let rgb_mse = rgb_squared_error / rgb_samples;
    let foreground_mse = if foreground_rgb_samples == 0 {
        0.0
    } else {
        foreground_rgb_squared_error / foreground_rgb_samples as f64
    };

    Ok(ImageMetrics {
        psnr_rgb: psnr_from_mse(rgb_mse),
        foreground_psnr_rgb: psnr_from_mse(foreground_mse),
        luminance_ssim: global_ssim(&reference_luma, &candidate_luma),
        alpha_mae: alpha_absolute_error / reference.len() as f64,
        foreground_iou: if foreground_union == 0 {
            1.0
        } else {
            foreground_intersection as f64 / foreground_union as f64
        },
        max_abs_error,
    })
}

/// Compare reconstruction error near logical-node boundaries with error in
/// the attributed interiors.
///
/// `node_labels` is row-major and uses `None` for pixels without a dominant
/// hierarchy node. A boundary seed is any attributed pixel with a four-neighbor
/// carrying a different node id. The boundary band is the Manhattan expansion
/// of those seeds by `band_radius` pixels, restricted to attributed pixels.
pub fn compare_node_boundary_bands(
    reference: &[[f32; 4]],
    candidate: &[[f32; 4]],
    node_labels: &[Option<u64>],
    width: u32,
    height: u32,
    band_radius: u32,
) -> Result<BoundaryBandMetrics, BoundaryMetricsError> {
    let pixel_count = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .filter(|&count| count > 0)
        .ok_or(BoundaryMetricsError::InvalidDimensions)?;
    if reference.len() != candidate.len() {
        return Err(BoundaryMetricsError::Image(
            ImageMetricsError::LengthMismatch {
                reference: reference.len(),
                candidate: candidate.len(),
            },
        ));
    }
    if reference.len() != pixel_count {
        return Err(BoundaryMetricsError::Image(
            ImageMetricsError::LengthMismatch {
                reference: pixel_count,
                candidate: reference.len(),
            },
        ));
    }
    if node_labels.len() != pixel_count {
        return Err(BoundaryMetricsError::LabelLengthMismatch {
            expected: pixel_count,
            actual: node_labels.len(),
        });
    }
    for (pixel, (expected, actual)) in reference.iter().zip(candidate).enumerate() {
        for channel in 0..4 {
            if !expected[channel].is_finite() || !actual[channel].is_finite() {
                return Err(BoundaryMetricsError::Image(ImageMetricsError::NonFinite {
                    pixel,
                    channel,
                }));
            }
        }
    }

    let width = width as usize;
    let height = height as usize;
    let mut distance = vec![u32::MAX; pixel_count];
    let mut queue = VecDeque::new();
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let Some(label) = node_labels[index] else {
                continue;
            };
            let differs = (x > 0 && node_labels[index - 1].is_some_and(|other| other != label))
                || (x + 1 < width && node_labels[index + 1].is_some_and(|other| other != label))
                || (y > 0 && node_labels[index - width].is_some_and(|other| other != label))
                || (y + 1 < height
                    && node_labels[index + width].is_some_and(|other| other != label));
            if differs {
                distance[index] = 0;
                queue.push_back(index);
            }
        }
    }
    if queue.is_empty() {
        return Err(BoundaryMetricsError::NoBoundaryPixels);
    }

    while let Some(index) = queue.pop_front() {
        let next_distance = distance[index].saturating_add(1);
        if next_distance > band_radius {
            continue;
        }
        let x = index % width;
        let y = index / width;
        let mut visit = |neighbor: usize| {
            if node_labels[neighbor].is_some() && distance[neighbor] > next_distance {
                distance[neighbor] = next_distance;
                queue.push_back(neighbor);
            }
        };
        if x > 0 {
            visit(index - 1);
        }
        if x + 1 < width {
            visit(index + 1);
        }
        if y > 0 {
            visit(index - width);
        }
        if y + 1 < height {
            visit(index + width);
        }
    }

    let mut boundary = SpatialResidualAccumulator::default();
    let mut interior = SpatialResidualAccumulator::default();
    for index in 0..pixel_count {
        if node_labels[index].is_none() {
            continue;
        }
        let target = if distance[index] <= band_radius {
            &mut boundary
        } else {
            &mut interior
        };
        target.add(reference[index], candidate[index]);
    }
    let boundary = boundary
        .finish()
        .ok_or(BoundaryMetricsError::NoBoundaryPixels)?;
    let interior = interior
        .finish()
        .ok_or(BoundaryMetricsError::NoInteriorPixels)?;
    Ok(BoundaryBandMetrics {
        rgb_rmse_enrichment: ratio_or_infinity(boundary.rgb_rmse, interior.rgb_rmse),
        alpha_abs_enrichment: ratio_or_infinity(boundary.alpha_abs_mean, interior.alpha_abs_mean),
        boundary,
        interior,
    })
}

#[derive(Default)]
struct SpatialResidualAccumulator {
    pixels: usize,
    rgb_squared_sum: f64,
    signed_luminance_sum: f64,
    signed_alpha_sum: f64,
    alpha_abs_sum: f64,
    max_abs_residual: f32,
}

impl SpatialResidualAccumulator {
    fn add(&mut self, reference: [f32; 4], candidate: [f32; 4]) {
        self.pixels += 1;
        for channel in 0..3 {
            let residual = f64::from(candidate[channel] - reference[channel]);
            self.rgb_squared_sum += residual * residual;
            self.max_abs_residual = self.max_abs_residual.max(residual.abs() as f32);
        }
        self.signed_luminance_sum += linear_luminance(candidate) - linear_luminance(reference);
        let alpha = f64::from(candidate[3] - reference[3]);
        self.signed_alpha_sum += alpha;
        self.alpha_abs_sum += alpha.abs();
        self.max_abs_residual = self.max_abs_residual.max(alpha.abs() as f32);
    }

    fn finish(self) -> Option<SpatialResidualMetrics> {
        (self.pixels > 0).then(|| SpatialResidualMetrics {
            pixels: self.pixels,
            rgb_rmse: (self.rgb_squared_sum / (self.pixels * 3) as f64).sqrt(),
            signed_luminance_mean: self.signed_luminance_sum / self.pixels as f64,
            signed_alpha_mean: self.signed_alpha_sum / self.pixels as f64,
            alpha_abs_mean: self.alpha_abs_sum / self.pixels as f64,
            max_abs_residual: self.max_abs_residual,
        })
    }
}

fn ratio_or_infinity(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        if numerator == 0.0 { 1.0 } else { f64::INFINITY }
    } else {
        numerator / denominator
    }
}

/// Measures motion-cancelled temporal error between a candidate and reference.
///
/// For previous/current images `R` and `C`, the signed residual is
/// `(C_current - C_previous) - (R_current - R_previous)`. A static candidate
/// bias therefore cancels, while a representation change remains visible.
pub fn compare_temporal_deltas(
    reference_previous: &[[f32; 4]],
    reference_current: &[[f32; 4]],
    candidate_previous: &[[f32; 4]],
    candidate_current: &[[f32; 4]],
    foreground_alpha_threshold: f32,
) -> Result<TemporalResidualMetrics, ImageMetricsError> {
    signed_image_expression_metrics(
        &[
            reference_previous,
            reference_current,
            candidate_previous,
            candidate_current,
        ],
        &[1.0, -1.0, -1.0, 1.0],
        foreground_alpha_threshold,
    )
}

/// Measures excess temporal curvature in a candidate relative to a reference.
///
/// The signed expression is
/// `(C_next - 2*C_current + C_previous) -
///  (R_next - 2*R_current + R_previous)`.
/// Smooth linear motion cancels; a one-frame discontinuity produces a spike.
#[allow(clippy::too_many_arguments)]
pub fn compare_temporal_second_differences(
    reference_previous: &[[f32; 4]],
    reference_current: &[[f32; 4]],
    reference_next: &[[f32; 4]],
    candidate_previous: &[[f32; 4]],
    candidate_current: &[[f32; 4]],
    candidate_next: &[[f32; 4]],
    foreground_alpha_threshold: f32,
) -> Result<TemporalResidualMetrics, ImageMetricsError> {
    signed_image_expression_metrics(
        &[
            reference_previous,
            reference_current,
            reference_next,
            candidate_previous,
            candidate_current,
            candidate_next,
        ],
        &[-1.0, 2.0, -1.0, 1.0, -2.0, 1.0],
        foreground_alpha_threshold,
    )
}

fn signed_image_expression_metrics(
    images: &[&[[f32; 4]]],
    weights: &[f64],
    foreground_alpha_threshold: f32,
) -> Result<TemporalResidualMetrics, ImageMetricsError> {
    debug_assert_eq!(images.len(), weights.len());
    let Some(reference) = images.first() else {
        return Err(ImageMetricsError::Empty);
    };
    if reference.is_empty() {
        return Err(ImageMetricsError::Empty);
    }
    for image in &images[1..] {
        if image.len() != reference.len() {
            return Err(ImageMetricsError::LengthMismatch {
                reference: reference.len(),
                candidate: image.len(),
            });
        }
    }

    let threshold = foreground_alpha_threshold.clamp(0.0, 1.0);
    let mut signed_rgb_sum = [0.0_f64; 3];
    let mut rgb_squared_sum = 0.0_f64;
    let mut signed_alpha_sum = 0.0_f64;
    let mut alpha_absolute_sum = 0.0_f64;
    let mut max_abs_residual = 0.0_f32;
    let mut foreground_pixels = 0_usize;

    for pixel in 0..reference.len() {
        let mut foreground = false;
        for image in images {
            for channel in 0..4 {
                if !image[pixel][channel].is_finite() {
                    return Err(ImageMetricsError::NonFinite { pixel, channel });
                }
            }
            foreground |= image[pixel][3] > threshold;
        }

        let mut residual = [0.0_f64; 4];
        for (image, &weight) in images.iter().zip(weights) {
            for (channel, output) in residual.iter_mut().enumerate() {
                *output += weight * f64::from(image[pixel][channel]);
            }
        }
        max_abs_residual = max_abs_residual.max(
            residual
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max) as f32,
        );
        signed_alpha_sum += residual[3];
        alpha_absolute_sum += residual[3].abs();
        if foreground {
            foreground_pixels += 1;
            for channel in 0..3 {
                signed_rgb_sum[channel] += residual[channel];
                rgb_squared_sum += residual[channel] * residual[channel];
            }
        }
    }

    let rgb_sample_count = foreground_pixels.saturating_mul(3);
    let rgb_mse = if rgb_sample_count == 0 {
        0.0
    } else {
        rgb_squared_sum / rgb_sample_count as f64
    };
    let rgb_mean_denominator = foreground_pixels.max(1) as f64;
    let image_denominator = reference.len() as f64;
    Ok(TemporalResidualMetrics {
        signed_rgb_mean: signed_rgb_sum.map(|value| value / rgb_mean_denominator),
        foreground_rgb_rmse: rgb_mse.sqrt(),
        foreground_psnr_rgb: psnr_from_mse(rgb_mse),
        signed_alpha_mean: signed_alpha_sum / image_denominator,
        alpha_abs_mean: alpha_absolute_sum / image_denominator,
        max_abs_residual,
        foreground_pixels,
    })
}

fn linear_luminance(rgba: [f32; 4]) -> f64 {
    0.2126 * f64::from(rgba[0]) + 0.7152 * f64::from(rgba[1]) + 0.0722 * f64::from(rgba[2])
}

fn psnr_from_mse(mse: f64) -> f64 {
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

// A global SSIM is deliberately used here: it is deterministic, dependency-free, and useful for
// CI smoke gates. Product-quality render tests may layer a windowed implementation on top.
fn global_ssim(reference: &[f64], candidate: &[f64]) -> f64 {
    let count = reference.len() as f64;
    let mean_reference = reference.iter().sum::<f64>() / count;
    let mean_candidate = candidate.iter().sum::<f64>() / count;

    let mut variance_reference = 0.0;
    let mut variance_candidate = 0.0;
    let mut covariance = 0.0;
    for (&expected, &actual) in reference.iter().zip(candidate) {
        let expected_delta = expected - mean_reference;
        let actual_delta = actual - mean_candidate;
        variance_reference += expected_delta * expected_delta;
        variance_candidate += actual_delta * actual_delta;
        covariance += expected_delta * actual_delta;
    }
    variance_reference /= count;
    variance_candidate /= count;
    covariance /= count;

    let c1 = 0.01_f64.powi(2);
    let c2 = 0.03_f64.powi(2);
    let numerator = (2.0 * mean_reference * mean_candidate + c1) * (2.0 * covariance + c2);
    let denominator = (mean_reference.powi(2) + mean_candidate.powi(2) + c1)
        * (variance_reference + variance_candidate + c2);
    if denominator == 0.0 {
        1.0
    } else {
        (numerator / denominator).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_images_have_perfect_metrics() {
        let image = [[0.2, 0.4, 0.8, 1.0], [0.0, 0.0, 0.0, 0.0]];
        let metrics = compare_linear_rgba(&image, &image, 1.0 / 255.0).unwrap();
        assert!(metrics.psnr_rgb.is_infinite());
        assert!(metrics.foreground_psnr_rgb.is_infinite());
        assert_eq!(metrics.luminance_ssim, 1.0);
        assert_eq!(metrics.alpha_mae, 0.0);
        assert_eq!(metrics.foreground_iou, 1.0);
        assert_eq!(metrics.max_abs_error, 0.0);
    }

    #[test]
    fn metrics_detect_color_alpha_and_mask_changes() {
        let reference = [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]];
        let candidate = [[0.5, 0.0, 0.0, 0.5], [0.1, 0.0, 0.0, 0.25]];
        let metrics = compare_linear_rgba(&reference, &candidate, 0.1).unwrap();
        assert!(metrics.psnr_rgb.is_finite());
        assert!(metrics.luminance_ssim < 1.0);
        assert_eq!(metrics.alpha_mae, 0.375);
        assert_eq!(metrics.foreground_iou, 0.5);
        assert_eq!(metrics.max_abs_error, 0.5);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(
            compare_linear_rgba(&[], &[], 0.0),
            Err(ImageMetricsError::Empty)
        );
        assert!(matches!(
            compare_linear_rgba(&[[0.0; 4]], &[], 0.0),
            Err(ImageMetricsError::LengthMismatch { .. })
        ));
        assert!(matches!(
            compare_linear_rgba(&[[f32::NAN, 0.0, 0.0, 0.0]], &[[0.0; 4]], 0.0),
            Err(ImageMetricsError::NonFinite { .. })
        ));
    }

    #[test]
    fn temporal_delta_cancels_identical_motion_and_static_bias() {
        let reference_previous = [[0.0, 0.25, 0.5, 0.25]];
        let reference_current = [[0.25, 0.5, 0.75, 0.5]];
        let candidate_previous = [[0.5, 0.25, 0.5, 0.5]];
        let candidate_current = [[0.75, 0.5, 0.75, 0.75]];
        let metrics = compare_temporal_deltas(
            &reference_previous,
            &reference_current,
            &candidate_previous,
            &candidate_current,
            0.01,
        )
        .unwrap();

        assert_eq!(metrics.signed_rgb_mean, [0.0; 3]);
        assert_eq!(metrics.foreground_rgb_rmse, 0.0);
        assert!(metrics.foreground_psnr_rgb.is_infinite());
        assert_eq!(metrics.signed_alpha_mean, 0.0);
        assert_eq!(metrics.alpha_abs_mean, 0.0);
        assert_eq!(metrics.max_abs_residual, 0.0);
        assert_eq!(metrics.foreground_pixels, 1);
    }

    #[test]
    fn temporal_delta_preserves_signed_channel_changes() {
        let still = [[0.0, 0.0, 0.0, 1.0]];
        let changed = [[0.5, 0.0, -0.25, 1.25]];
        let metrics = compare_temporal_deltas(&still, &still, &still, &changed, 0.01).unwrap();

        assert_eq!(metrics.signed_rgb_mean, [0.5, 0.0, -0.25]);
        assert!((metrics.foreground_rgb_rmse - (5.0_f64 / 48.0).sqrt()).abs() < 1e-12);
        assert_eq!(metrics.signed_alpha_mean, 0.25);
        assert_eq!(metrics.alpha_abs_mean, 0.25);
        assert_eq!(metrics.max_abs_residual, 0.5);
    }

    #[test]
    fn temporal_second_difference_cancels_linear_motion_and_detects_an_impulse() {
        let zero = [[0.0, 0.0, 0.0, 1.0]];
        let quarter = [[0.25, 0.0, 0.0, 1.0]];
        let half = [[0.5, 0.0, 0.0, 1.0]];
        let linear = compare_temporal_second_differences(
            &zero, &quarter, &half, &zero, &quarter, &half, 0.01,
        )
        .unwrap();
        assert_eq!(linear.foreground_rgb_rmse, 0.0);
        assert!(linear.foreground_psnr_rgb.is_infinite());

        let impulse =
            compare_temporal_second_differences(&zero, &zero, &zero, &zero, &half, &zero, 0.01)
                .unwrap();
        assert_eq!(impulse.signed_rgb_mean, [-1.0, 0.0, 0.0]);
        assert!((impulse.foreground_rgb_rmse - (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(impulse.max_abs_residual, 1.0);
    }

    #[test]
    fn temporal_metrics_reject_mismatched_and_non_finite_sequences() {
        let valid = [[0.0; 4]];
        assert!(matches!(
            compare_temporal_deltas(&valid, &[], &valid, &valid, 0.0),
            Err(ImageMetricsError::LengthMismatch { .. })
        ));
        let invalid = [[0.0, f32::NAN, 0.0, 0.0]];
        assert!(matches!(
            compare_temporal_deltas(&valid, &valid, &valid, &invalid, 0.0),
            Err(ImageMetricsError::NonFinite {
                pixel: 0,
                channel: 1
            })
        ));
    }

    #[test]
    fn boundary_bands_expose_localized_signed_under_reconstruction() {
        let width = 8;
        let height = 3;
        let reference = vec![[0.5, 0.5, 0.5, 0.8]; width * height];
        let mut candidate = reference.clone();
        let mut labels = vec![None; width * height];
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                labels[index] = Some(if x >= width / 2 { 2 } else { 1 });
                if x == width / 2 - 1 || x == width / 2 {
                    candidate[index] = [0.3, 0.3, 0.3, 0.5];
                }
            }
        }

        let metrics = compare_node_boundary_bands(
            &reference,
            &candidate,
            &labels,
            width as u32,
            height as u32,
            0,
        )
        .unwrap();
        assert_eq!(metrics.boundary.pixels, height * 2);
        assert_eq!(metrics.interior.pixels, height * (width - 2));
        assert!((metrics.boundary.rgb_rmse - 0.2).abs() < 1e-7);
        assert!(metrics.boundary.signed_luminance_mean < -0.19);
        assert!((metrics.boundary.signed_alpha_mean + 0.3).abs() < 1e-7);
        assert_eq!(metrics.interior.rgb_rmse, 0.0);
        assert_eq!(metrics.interior.alpha_abs_mean, 0.0);
        assert!(metrics.rgb_rmse_enrichment.is_infinite());
        assert!(metrics.alpha_abs_enrichment.is_infinite());
    }

    #[test]
    fn boundary_bands_reject_missing_boundaries_and_consumed_interiors() {
        let image = vec![[0.0; 4]; 4];
        assert_eq!(
            compare_node_boundary_bands(&image, &image, &[Some(1); 4], 2, 2, 0),
            Err(BoundaryMetricsError::NoBoundaryPixels)
        );
        assert_eq!(
            compare_node_boundary_bands(
                &image,
                &image,
                &[Some(1), Some(2), Some(1), Some(2)],
                2,
                2,
                1,
            ),
            Err(BoundaryMetricsError::NoInteriorPixels)
        );
    }
}
