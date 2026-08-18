use std::fmt;

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
}
