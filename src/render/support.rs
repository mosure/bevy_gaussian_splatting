//! Shared finite Gaussian density. The radial C1 taper occupies the last ninth
//! of squared support radius, leaving q<=8 unchanged for authored 3-sigma
//! support. It never extends raster/world bounds or renormalizes opacity.
//!
//! At 3 sigma the integrated kernel mass is 0.9856464154 of the unbounded
//! Gaussian: 0.3281% below the former truncated disk and 0.9010% below the
//! former OBB square. These are kernel-mass changes, not composited RGB bounds.

/// Current-camera near clipping of a Gaussian's peak opacity. `clearance` is
/// the center's signed distance inside the near plane, and `variance` is its
/// covariance along that plane's normal. A common positive plane scale cancels.
///
/// The C1 filter reaches full opacity when the authored three-sigma depth
/// support is inside the plane. This softens center clipping; it is not an
/// integral of the Gaussian volume in front of the plane.
#[inline]
pub fn gaussian_near_clip_weight(clearance: f32, variance: f32) -> f32 {
    if !(clearance > 0.0 && clearance.is_finite() && variance >= 0.0 && variance.is_finite()) {
        return 0.0;
    }
    if variance == 0.0 {
        return 1.0;
    }
    let x = (clearance / (3.0 * variance.sqrt())).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Density multiplier matching `support.wgsl`, including adaptive cutoffs.
#[inline]
pub fn gaussian_support_weight(q: f32, cutoff_squared: f32) -> f32 {
    if !(q >= 0.0 && cutoff_squared > 0.0 && q < cutoff_squared) {
        return 0.0;
    }
    let u = (9.0 * (q / cutoff_squared) - 8.0).clamp(0.0, 1.0);
    let remaining = 1.0 - u;
    (-0.5 * q).exp() * remaining * remaining * (1.0 + 2.0 * u)
}

/// Analytic density and derivative with respect to Mahalanobis squared radius.
/// Fitting uses the derivative directly, avoiding division by a vanishing tail.
/// The deterministic integration oracle uses f64 to bound the same kernel.
#[inline]
#[cfg(any(test, feature = "testing"))]
pub(crate) fn gaussian_support_weight_and_derivative(q: f64, cutoff_squared: f64) -> (f64, f64) {
    if !(q >= 0.0 && cutoff_squared > 0.0 && q < cutoff_squared) {
        return (0.0, 0.0);
    }
    let u = (9.0 * (q / cutoff_squared) - 8.0).clamp(0.0, 1.0);
    let remaining = 1.0 - u;
    let taper = remaining * remaining * (1.0 + 2.0 * u);
    let taper_derivative = -54.0 * u * remaining / cutoff_squared;
    let density = (-0.5 * q).exp();
    (density * taper, density * (taper_derivative - 0.5 * taper))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_clip_weight_has_support_scaled_continuous_endpoints() {
        assert_eq!(gaussian_near_clip_weight(-1.0, 1.0), 0.0);
        assert_eq!(gaussian_near_clip_weight(0.0, 1.0), 0.0);
        assert_eq!(gaussian_near_clip_weight(1.5, 1.0), 0.5);
        assert_eq!(gaussian_near_clip_weight(3.0, 1.0), 1.0);
        assert_eq!(gaussian_near_clip_weight(4.0, 1.0), 1.0);
        assert_eq!(gaussian_near_clip_weight(0.0, 0.0), 0.0);
        assert_eq!(gaussian_near_clip_weight(0.01, 0.0), 1.0);
        assert_eq!(gaussian_near_clip_weight(3.0, 4.0), 0.5);
        assert_eq!(gaussian_near_clip_weight(1.5, 4.0), 0.15625);
        let epsilon = 0.001;
        assert!(gaussian_near_clip_weight(epsilon, 1.0) / epsilon < 0.001);
        assert!((1.0 - gaussian_near_clip_weight(3.0 - epsilon, 1.0)) / epsilon < 0.001);
        for (clearance, variance) in [(f32::NAN, 1.0), (1.0, f32::INFINITY), (1.0, -1.0)] {
            assert_eq!(gaussian_near_clip_weight(clearance, variance), 0.0);
        }
    }

    #[test]
    fn finite_support_preserves_core_and_has_continuous_endpoint_derivatives() {
        for cutoff_squared in [1.0, 9.0] {
            let inner = cutoff_squared * 8.0 / 9.0;
            for q in [0.0, inner * 0.5, inner] {
                let (weight, derivative) =
                    gaussian_support_weight_and_derivative(q, cutoff_squared);
                assert!((weight - (-0.5 * q).exp()).abs() < 1e-14);
                assert!((derivative + 0.5 * weight).abs() < 1e-14);
            }
            for q in [inner, (inner + cutoff_squared) * 0.5, cutoff_squared] {
                let h = cutoff_squared * 1e-6;
                let (weight, derivative) =
                    gaussian_support_weight_and_derivative(q, cutoff_squared);
                let numeric = (gaussian_support_weight_and_derivative(q + h, cutoff_squared).0
                    - gaussian_support_weight_and_derivative(q - h, cutoff_squared).0)
                    / (2.0 * h);
                assert!((derivative - numeric).abs() < 1e-4);
                assert!(
                    (f64::from(gaussian_support_weight(q as f32, cutoff_squared as f32)) - weight)
                        .abs()
                        < 1e-6
                );
            }
            assert_eq!(
                gaussian_support_weight_and_derivative(cutoff_squared, cutoff_squared),
                (0.0, 0.0)
            );
            assert_eq!(
                gaussian_support_weight((cutoff_squared * 2.0) as f32, cutoff_squared as f32),
                0.0
            );
        }
    }
}
