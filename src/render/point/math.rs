//! Numerical reference for Gaussian Point Splatting (Rijsdijk et al., 2026).
//!
//! The f64 oracle uses a converged dilogarithm series and bisection. The f32
//! routines mirror `sampling.wgsl` so its bounded numerical policies can be
//! checked without GPU experiments. Neither count clipping nor replacing
//! discarded support samples preserves the corrected Poisson process.
//!
//! Sampling derivation: <https://jorisar.nl/gaussian_point_splatting/>.
//! PTRS: Hoermann, "The transformed rejection method for generating Poisson
//! random variables" (1993), <https://doi.org/10.1016/0167-6687(93)90997-4>.

pub const MAX_OPACITY: f64 = 0.999;
pub const MAX_POISSON_RATE: f32 = 1_048_576.0;
pub const MAX_POISSON_COMPONENTS: u32 = 1024;
pub const POISSON_FAILURE: u32 = u32::MAX;

/// Li2(alpha), with the rendering opacity domain explicitly limited to .999.
pub fn dilog(alpha: f64) -> f64 {
    let x = alpha.clamp(0.0, MAX_OPACITY);
    let z = if x > 0.5 { 1.0 - x } else { x };
    let mut power = z;
    let mut sum = 0.0;
    for n in 1..=64 {
        let term = power / f64::from(n * n);
        sum += term;
        if term < f64::EPSILON * sum {
            break;
        }
        power *= z;
    }
    if x > 0.5 {
        std::f64::consts::FRAC_PI_6 * std::f64::consts::PI - x.ln() * (-x).ln_1p() - sum
    } else {
        sum
    }
}

/// Expected full-process count when covariance is expressed in sample pixels.
pub fn expected_count(covariance_determinant: f64, opacity: f64) -> f64 {
    std::f64::consts::TAU * covariance_determinant.sqrt() * dilog(opacity)
}

/// Corrected radial inverse CDF; `u` must lie strictly between zero and one.
pub fn radius(opacity: f64, u: f64) -> f64 {
    assert!(u > 0.0 && u < 1.0);
    let alpha = opacity.clamp(0.0, MAX_OPACITY);
    if alpha == 0.0 {
        return 0.0;
    }
    let target = (1.0 - u) * dilog(alpha);
    let (mut lower, mut upper) = (0.0, alpha);
    for _ in 0..64 {
        let middle = (lower + upper) * 0.5;
        if dilog(middle) < target {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    (-2.0 * (((lower + upper) * 0.5) / alpha).ln()).sqrt()
}

fn hash(mut value: u32) -> u32 {
    value = (value ^ (value >> 16)).wrapping_mul(0x7feb_352d);
    value = (value ^ (value >> 15)).wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

pub fn point_seed(cloud: u32, gaussian: u32, frame: u32, sample: u32) -> u32 {
    hash(
        cloud
            ^ hash(gaussian.wrapping_add(0x9e37_79b9))
            ^ hash(frame.wrapping_add(0x85eb_ca6b))
            ^ hash(sample.wrapping_add(0xc2b2_ae35)),
    )
}

/// Open-interval f32 uniforms, identically keyed to the shader's counter RNG.
pub fn point_random(seed: u32, counter: u32) -> f32 {
    let bits = hash(seed ^ hash(counter.wrapping_add(0x9e37_79b9)));
    ((bits >> 9) as f32 + 0.5) * (1.0 / 8_388_608.0)
}

pub fn point_intensity(opacity: f32) -> f32 {
    if opacity < 0.01 {
        opacity * (1.0 + opacity * (0.5 + opacity * (1.0 / 3.0 + opacity * 0.25)))
    } else {
        -(1.0 - opacity).ln()
    }
}

pub fn point_rectangle(mean: [f32; 2], basis: [f32; 3], extent: [f32; 2]) -> [f32; 4] {
    let radius = [
        3.0 * basis[0],
        3.0 * (basis[1] * basis[1] + basis[2] * basis[2]).sqrt(),
    ];
    [
        (mean[0] - radius[0]).clamp(0.0, extent[0]),
        (mean[1] - radius[1]).clamp(0.0, extent[1]),
        (mean[0] + radius[0]).clamp(0.0, extent[0]),
        (mean[1] + radius[1]).clamp(0.0, extent[1]),
    ]
}

/// Conservative rectangle envelope mirrored from WGSL. Each marginal distance
/// divided by its standard deviation lower-bounds the full Mahalanobis radius.
/// Outward coordinate/radius slack and upward opacity/intensity slack prevent
/// f32 rounding from turning the tighter envelope into intensity clipping.
pub fn point_rectangle_peak(
    opacity: f32,
    mean: [f32; 2],
    basis: [f32; 3],
    rectangle: [f32; 4],
) -> f32 {
    const GUARD: f32 = 1.0 / 16384.0;
    let global_peak = point_intensity(opacity) * (1.0 + GUARD);
    let marginal = [basis[0], (basis[1] * basis[1] + basis[2] * basis[2]).sqrt()];
    let normalized = std::array::from_fn::<_, 2, _>(|axis| {
        let coordinate = 1.0f32
            .max(mean[axis].abs())
            .max(rectangle[axis].abs())
            .max(rectangle[axis + 2].abs());
        let distance = (rectangle[axis] - mean[axis]).max(mean[axis] - rectangle[axis + 2]);
        (distance - GUARD * coordinate).max(0.0) / (marginal[axis] * (1.0 + GUARD))
    });
    let lower_bound = ((normalized[0] * normalized[0]).max(normalized[1] * normalized[1])
        * (1.0 - GUARD)
        - GUARD)
        .max(0.0);
    if !(0.0..=9.0).contains(&lower_bound) {
        return global_peak;
    }
    let peak_opacity = opacity.min(opacity * (-0.5 * lower_bound).exp() * (1.0 + GUARD));
    point_intensity(peak_opacity) * (1.0 + GUARD)
}

pub fn point_dilog(alpha: f32) -> f32 {
    let x = alpha.clamp(0.0, MAX_OPACITY as f32);
    let z = if x > 0.5 { 1.0 - x } else { x };
    let mut sum = 0.0;
    for n in (1..=16).rev() {
        sum = z * (sum + 1.0 / (n * n) as f32);
    }
    if x > 0.5 {
        (std::f64::consts::PI.powi(2) / 6.0) as f32 - x.ln() * (1.0 - x).ln() - sum
    } else {
        sum
    }
}

pub fn point_radius(alpha: f32, dilog: f32, u: f32) -> f32 {
    let opacity = alpha.clamp(0.0, MAX_OPACITY as f32);
    if opacity == 0.0 || dilog <= 0.0 {
        return 0.0;
    }
    let target = (1.0 - u) * dilog;
    // Degree-10 least-squares inverse. Fit the smooth residual (x-y)/y^2
    // at 2,049 Chebyshev-spaced y/Li2(.999) nodes, using the f64 oracle.
    // Factorization ensures exact zero and unit slope at the origin.
    let t = target / 1.637_022_6;
    let mut polynomial = -0.643_604_34;
    polynomial = polynomial * t + 2.305_536_7;
    polynomial = polynomial * t - 3.361_320_7;
    polynomial = polynomial * t + 2.545_964;
    polynomial = polynomial * t - 1.066_589_6;
    polynomial = polynomial * t + 0.239_364_98;
    polynomial = polynomial * t - 0.031_230_46;
    polynomial = polynomial * t + 0.023_843_503;
    polynomial = polynomial * t - 0.250_007_48;
    let mut x = (target + target * target * polynomial).clamp(0.0, opacity);
    for _ in 0..2 {
        let derivative = if x < 0.01 {
            1.0 + x * (0.5 + x * (1.0 / 3.0 + x * 0.25))
        } else {
            -(1.0 - x).ln() / x
        };
        x = (x - (point_dilog(x) - target) / derivative).clamp(0.0, opacity);
    }
    (-2.0 * (x / opacity).clamp(1e-30, 1.0).ln())
        .max(0.0)
        .sqrt()
}

fn poisson_log_probability(k: f32, lambda: f32) -> f32 {
    if k == 0.0 {
        return -lambda;
    }
    if k < 10.0 {
        let log_factorial: f32 = (2..=k as u32).map(|n| (n as f32).ln()).sum();
        return k * lambda.ln() - lambda - log_factorial;
    }
    let mut deviance = k * (k / lambda).ln() + lambda - k;
    if (k - lambda).abs() < 0.1 * (k + lambda) {
        let v = (k - lambda) / (k + lambda);
        let square = v * v;
        let mut term = 2.0 * k * v;
        deviance = (k - lambda) * v;
        for j in 1..=6 {
            term *= square;
            deviance += term / (2 * j + 1) as f32;
        }
    }
    let inverse = 1.0 / k;
    let square = inverse * inverse;
    let stirling_error = inverse * (1.0 / 12.0 + square * (-1.0 / 360.0 + square / 1260.0));
    -deviance - stirling_error - 0.5 * (std::f32::consts::TAU * k).ln()
}

/// Bounded f32 Poisson sampler; failure is distinct from a legitimate zero count.
pub fn point_poisson(lambda: f32, seed: u32) -> u32 {
    if !(0.0..=MAX_POISSON_RATE).contains(&lambda) {
        return POISSON_FAILURE;
    }
    if lambda == 0.0 {
        return 0;
    }
    if lambda < 10.0 {
        let u = point_random(seed, 0);
        let mut probability = (-lambda).exp();
        let mut cumulative = probability;
        for k in 0..128 {
            if u <= cumulative {
                return k;
            }
            probability *= lambda / (k + 1) as f32;
            cumulative += probability;
        }
        return POISSON_FAILURE;
    }
    let b = 0.931 + 2.53 * lambda.sqrt();
    let a = -0.059 + 0.02483 * b;
    let inverse_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    for attempt in 0..64 {
        let u = point_random(seed, 2 * attempt) - 0.5;
        let v = point_random(seed, 2 * attempt + 1);
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        if us >= 0.07 && v <= vr {
            return k as u32;
        }
        if (v * inverse_alpha / (a / (us * us) + b)).ln() <= poisson_log_probability(k, lambda) {
            return k as u32;
        }
    }
    POISSON_FAILURE
}

/// Independent Poisson components preserve the full count distribution. The
/// limit sentinel means the already-sampled nonnegative sum exceeds the bound;
/// callers must reject the entire image rather than draw that prefix.
pub fn point_poisson_process(lambda: f32, seed: u32, limit: u32) -> u32 {
    if !(0.0..=MAX_POISSON_RATE * MAX_POISSON_COMPONENTS as f32).contains(&lambda) {
        return POISSON_FAILURE;
    }
    if lambda <= MAX_POISSON_RATE {
        return point_poisson(lambda, seed);
    }
    let mut remaining = lambda;
    let mut total = 0;
    for component in 0..MAX_POISSON_COMPONENTS {
        if remaining == 0.0 {
            break;
        }
        let rate = remaining.min(MAX_POISSON_RATE);
        let count = point_poisson(rate, hash(seed ^ hash(component.wrapping_add(0xd1b5_4a35))));
        if count == POISSON_FAILURE {
            return count;
        }
        if count > limit - total {
            return limit + 1;
        }
        total += count;
        remaining -= rate;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_poisson_preserves_small_stream_and_hard_work_rejection() {
        let rate = 3.5 * MAX_POISSON_RATE;
        let seed = point_seed(17, 31, 0, 0);
        assert_eq!(
            point_poisson_process(7.5, seed, 100),
            point_poisson(7.5, seed)
        );
        assert_eq!(point_poisson_process(rate, seed, 1024), 1025);
        let count = point_poisson_process(rate, seed, 1 << 30);
        assert_ne!(count, POISSON_FAILURE);
        assert!((count as f32 - rate).abs() < 10.0 * rate.sqrt());
        assert_eq!(
            point_poisson_process(f32::INFINITY, 0, 1024),
            POISSON_FAILURE
        );
    }

    #[test]
    fn clipped_rectangle_thinning_matches_offaxis_and_giant_ellipse_intensity() {
        let opacity = 0.8_f32;
        let viewport = [4.0, 4.0];
        for (mean, basis) in [
            ([0.25, 2.0], [2.3, 1.2, 1.7]),
            ([-1000.0, 256.0], [10000.0, 2500.0, 10000.0]),
            ([-1000.0, -500.0], [400.0, 200.0, 200.0]),
        ] {
            let rectangle = point_rectangle(mean, basis, viewport);
            let peak = point_rectangle_peak(opacity, mean, basis, rectangle);
            if basis[0] == 400.0 {
                assert!(
                    peak < point_intensity(opacity) / 32.0,
                    "far off-axis tail kept the unattenuated center peak"
                );
            }
            let rate = (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1]) * peak;
            let full = std::f32::consts::TAU * basis[0] * basis[2] * point_dilog(opacity);
            assert!(rate < full);
            assert!(rate <= viewport[0] * viewport[1] * peak);
            let density = |px: f64, py: f64| {
                let x = (px - f64::from(mean[0])) / f64::from(basis[0]);
                let y = (py - f64::from(mean[1]) - f64::from(basis[1]) * x) / f64::from(basis[2]);
                let radius_squared = x * x + y * y;
                if radius_squared > 9.0 {
                    0.0
                } else {
                    -(-f64::from(opacity)
                        * crate::render::support::gaussian_support_weight_and_derivative(
                            radius_squared,
                            9.0,
                        )
                        .0)
                        .ln_1p()
                }
            };
            // Include the rectangle edges/corners where the maximum tail
            // intensity can occur, and compare directly with the f64 oracle.
            for y in 0..=16 {
                for x in 0..=16 {
                    let px = f64::from(rectangle[0])
                        + f64::from(rectangle[2] - rectangle[0]) * f64::from(x) / 16.0;
                    let py = f64::from(rectangle[1])
                        + f64::from(rectangle[3] - rectangle[1]) * f64::from(y) / 16.0;
                    assert!(
                        f64::from(peak) >= density(px, py),
                        "rectangle envelope underestimated intensity at ({px}, {py})"
                    );
                }
            }
            let (mut expected_mass, mut first_pixel_mass) = (0.0, 0.0);
            for y in 0..128 {
                for x in 0..128 {
                    let mass =
                        density((f64::from(x) + 0.5) / 32.0, (f64::from(y) + 0.5) / 32.0) / 1024.0;
                    expected_mass += mass;
                    if x < 32 && y < 32 {
                        first_pixel_mass += mass;
                    }
                }
            }
            let population = 1024;
            let (mut sum, mut squares, mut empty_pixels) = (0.0, 0.0, 0);
            for frame in 0..population {
                let seed = point_seed(73, 19, frame, 0);
                let count = point_poisson_process(rate, seed, 1024);
                assert_ne!(count, POISSON_FAILURE);
                let (mut accepted, mut first_pixel_hit) = (0, false);
                for point in 0..count {
                    let counter = point * 3 + 4096;
                    let px =
                        rectangle[0] + (rectangle[2] - rectangle[0]) * point_random(seed, counter);
                    let py = rectangle[1]
                        + (rectangle[3] - rectangle[1]) * point_random(seed, counter + 1);
                    let x = (px - mean[0]) / basis[0];
                    let y = (py - mean[1] - basis[1] * x) / basis[2];
                    let radius_squared = x * x + y * y;
                    if radius_squared <= 9.0
                        && point_random(seed, counter + 2) * peak
                            < point_intensity(
                                opacity
                                    * crate::render::support::gaussian_support_weight(
                                        radius_squared,
                                        9.0,
                                    ),
                            )
                    {
                        accepted += 1;
                        first_pixel_hit |= px < 1.0 && py < 1.0;
                    }
                }
                sum += f64::from(accepted);
                squares += f64::from(accepted).powi(2);
                empty_pixels += u32::from(!first_pixel_hit);
            }
            let n = f64::from(population);
            let mean = sum / n;
            let variance = squares / n - mean * mean;
            let empty = (-first_pixel_mass).exp();
            assert!((mean - expected_mass).abs() < 6.0 * (expected_mass / n).sqrt() + 0.01);
            assert!(
                (variance - expected_mass).abs()
                    < 8.0 * ((expected_mass + 2.0 * expected_mass * expected_mass) / n).sqrt()
            );
            assert!(
                (f64::from(empty_pixels) / n - empty).abs()
                    < 6.0 * (empty * (1.0 - empty) / n).sqrt()
            );
        }
        assert!(point_intensity(1e-8) > 0.0);
    }

    #[test]
    fn opacity_mass_and_inverse_match_the_f64_oracle() {
        assert_eq!(dilog(0.0), 0.0);
        assert!((dilog(0.5) - 0.582_240_526_465_012_5).abs() < 1e-15);
        assert_eq!(point_dilog(1.0), point_dilog(MAX_OPACITY as f32));
        for alpha in [1e-8, 0.001, 0.1, 0.5, 0.8, 0.99, 0.999] {
            let approximate_mass = f64::from(point_dilog(alpha as f32));
            assert!((approximate_mass - dilog(alpha)).abs() < 5e-7);
            assert!(expected_count(4.0, alpha) >= std::f64::consts::TAU * 2.0 * alpha);
            for u in [1e-5, 0.001, 0.01, 0.1, 0.5, 0.9, 0.999, 0.999_99] {
                let actual = f64::from(point_radius(
                    alpha as f32,
                    point_dilog(alpha as f32),
                    u as f32,
                ));
                let reference = radius(alpha, u);
                // CDF error is the distribution criterion, including near the origin.
                let cdf = 1.0 - dilog(alpha * (-0.5 * actual * actual).exp()) / dilog(alpha);
                assert!((cdf - u).abs() < 2e-6, "alpha={alpha}, u={u}, cdf={cdf}");
                assert!((actual - reference).abs() < 0.001);
            }
        }
    }

    #[test]
    fn support_rejection_preserves_the_interior_intensity() {
        // Both out-of-support proposals and the tapered annulus are thinned;
        // replacing either rejection would multiply the unchanged core density.
        let alpha = 0.9;
        let total = dilog(alpha);
        let retained = total - dilog(alpha * (-4.5_f64).exp());
        assert!(retained < total);
        let density_at_one = -(1.0 - alpha * (-0.5_f64).exp()).ln();
        let opacity = 1.0 - (-density_at_one).exp();
        assert!((opacity - alpha * (-0.5_f64).exp()).abs() < 1e-15);
        assert!((1.0 - (-density_at_one * total / retained).exp() - opacity).abs() > 1e-3);
        let envelope = -(-alpha * (-0.5_f64 * 8.5).exp()).ln_1p();
        let tail = crate::render::support::gaussian_support_weight_and_derivative(8.5, 9.0).0;
        let target = -(-alpha * tail).ln_1p();
        let acceptance = target / envelope;
        assert!(acceptance > 0.0 && acceptance < 1.0);
        assert!((envelope * acceptance - target).abs() < 1e-15);
        assert_eq!(
            crate::render::support::gaussian_support_weight(9.0, 9.0),
            0.0
        );
    }

    #[test]
    fn poisson_counts_match_mean_variance_and_empty_probability() {
        let population = 16_384;
        for lambda in [0.05, 1.0, 7.5, 10.0, 100.0, 10_000.0] {
            let (mut sum, mut square_sum, mut zeros) = (0.0, 0.0, 0);
            for frame in 0..population {
                let seed = point_seed(7, 13, frame, 3);
                let u = point_random(seed, 5);
                assert!(u > 0.0 && u < 1.0);
                let count = point_poisson(lambda, seed);
                assert_ne!(count, POISSON_FAILURE);
                let count = f64::from(count);
                sum += count;
                square_sum += count * count;
                zeros += u32::from(count == 0.0);
            }
            let n = f64::from(population);
            let rate = f64::from(lambda);
            let mean = sum / n;
            let variance = square_sum / n - mean * mean;
            assert!((mean - rate).abs() < 6.0 * (rate / n).sqrt());
            assert!((variance - rate).abs() < 8.0 * ((rate + 2.0 * rate * rate) / n).sqrt());
            let empty = (-rate).exp();
            assert!(
                (f64::from(zeros) / n - empty).abs()
                    < 6.0 * (empty * (1.0 - empty) / n).sqrt() + 1.0 / n
            );
        }
    }

    #[test]
    fn unsupported_counts_fail_explicitly() {
        assert_eq!(point_poisson(0.0, 1), 0);
        for lambda in [-1.0, f32::NAN, f32::INFINITY, MAX_POISSON_RATE + 1.0] {
            assert_eq!(point_poisson(lambda, 1), POISSON_FAILURE);
        }
        for frame in 0..128 {
            let count = point_poisson(MAX_POISSON_RATE, point_seed(0, 1, frame, 0));
            assert_ne!(count, POISSON_FAILURE);
            assert!((f64::from(count) - f64::from(MAX_POISSON_RATE)).abs() < 10_000.0);
        }
    }
}
