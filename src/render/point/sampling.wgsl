#define_import_path bevy_gaussian_splatting::point_sampling

// Gaussian Point Splatting, Rijsdijk et al. (2026), sections 3.3--3.4.
// Independent implementation: no fitted coefficients from the reference renderer.
// Sample the full corrected process, or thin a clipped rectangle envelope.
// Replacing rejected points or clipping counts changes the expected opacity.
const POINT_MAX_OPACITY: f32 = 0.999;
const POINT_POISSON_FAILURE: u32 = 0xffffffffu;
// Above this rate f32 proposal arithmetic increasingly loses subinteger precision.
// Larger processes are sums of independently keyed components below this bound.
const POINT_MAX_POISSON_RATE: f32 = 1048576.0;
const POINT_MAX_POISSON_COMPONENTS: u32 = 1024u;

fn point_hash(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

fn point_seed(cloud: u32, gaussian: u32, frame: u32, sample: u32) -> u32 {
    return point_hash(cloud ^ point_hash(gaussian + 0x9e3779b9u)
        ^ point_hash(frame + 0x85ebca6bu) ^ point_hash(sample + 0xc2b2ae35u));
}

fn point_random(seed: u32, counter: u32) -> f32 {
    // Exactly representable midpoints: neither zero nor one, including after f32 rounding.
    let bits = point_hash(seed ^ point_hash(counter + 0x9e3779b9u));
    return (f32(bits >> 9u) + 0.5) * (1.0 / 8388608.0);
}

fn point_intensity(opacity: f32) -> f32 {
    // Stable -log(1-opacity), including opacities below one f32 ULP at one.
    if opacity < 0.01 {
        return opacity * (1.0 + opacity * (0.5 + opacity * (1.0 / 3.0 + opacity * 0.25)));
    }
    return -log(1.0 - opacity);
}

fn point_rectangle(mean: vec2<f32>, basis: vec2<f32>, basis_y: f32, extent: vec2<f32>) -> vec4<f32> {
    let radius = 3.0 * vec2<f32>(basis.x, length(vec2<f32>(basis.y, basis_y)));
    return vec4<f32>(clamp(mean - radius, vec2<f32>(0.0), extent),
        clamp(mean + radius, vec2<f32>(0.0), extent));
}

// A marginal displacement cannot exceed the Mahalanobis radius: for
// d = L*z, |d_i| <= length(L_i)*length(z). Each rectangle-axis distance thus
// supplies a lower bound on z.z, even for a correlated projected Gaussian.
fn point_rectangle_peak(opacity: f32, mean: vec2<f32>, basis: vec2<f32>, basis_y: f32, rectangle: vec4<f32>) -> f32 {
    // Exact power-of-two slack covers f32 distance, norm, exponential and log
    // rounding. Enlarge the proposal envelope; thinning keeps its original
    // intensity, so this slack adds only rejected proposals.
    let guard = 0.00006103515625; // 2^-14
    let global_peak = point_intensity(opacity) * (1.0 + guard);
    let coordinates = max(vec2<f32>(1.0), max(abs(mean), max(abs(rectangle.xy), abs(rectangle.zw))));
    let distance = max(vec2<f32>(0.0), max(rectangle.xy - mean, mean - rectangle.zw) - guard * coordinates);
    let marginal = vec2<f32>(basis.x, length(vec2<f32>(basis.y, basis_y))) * (1.0 + guard);
    let normalized = distance / marginal;
    let lower_bound = max(0.0, max(normalized.x * normalized.x, normalized.y * normalized.y) * (1.0 - guard) - guard);
    // The rectangle intersects the three-sigma support. Unusable arithmetic
    // must retain the original global envelope, never underestimate a tail.
    if !(lower_bound >= 0.0 && lower_bound <= 9.0) { return global_peak; }
    let peak_opacity = min(opacity, opacity * exp(-0.5 * lower_bound) * (1.0 + guard));
    return point_intensity(peak_opacity) * (1.0 + guard);
}

fn point_dilog(alpha: f32) -> f32 {
    let x = clamp(alpha, 0.0, POINT_MAX_OPACITY);
    let reflected = x > 0.5;
    let z = select(x, 1.0 - x, reflected);
    // Horner evaluation of sum(z^k/k^2), with no divisions or dynamic loops.
    // z <= 1/2: the omitted series tail is below 5.3e-8.
    var sum = 1.0 / 256.0;
    sum = sum * z + 1.0 / 225.0;
    sum = sum * z + 1.0 / 196.0;
    sum = sum * z + 1.0 / 169.0;
    sum = sum * z + 1.0 / 144.0;
    sum = sum * z + 1.0 / 121.0;
    sum = sum * z + 1.0 / 100.0;
    sum = sum * z + 1.0 / 81.0;
    sum = sum * z + 1.0 / 64.0;
    sum = sum * z + 1.0 / 49.0;
    sum = sum * z + 1.0 / 36.0;
    sum = sum * z + 1.0 / 25.0;
    sum = sum * z + 1.0 / 16.0;
    sum = sum * z + 1.0 / 9.0;
    sum = sum * z + 0.25;
    sum = z * (sum * z + 1.0);
    if reflected {
        return 1.6449340668482264 - log(x) * log(1.0 - x) - sum;
    }
    return sum;
}

fn point_radius(alpha: f32, dilog: f32, u: f32) -> f32 {
    let opacity = clamp(alpha, 0.0, POINT_MAX_OPACITY);
    if opacity == 0.0 || dilog <= 0.0 {
        return 0.0;
    }
    let cdf_target = (1.0 - u) * dilog;
    // Independently fitted degree-10 inverse, factored as y+y^2*P8(y/Li2(.999)).
    // The factorization preserves small-opacity behavior, unlike a free intercept.
    // Two corrections control f32 CDF error without a per-point search loop.
    let t = cdf_target / 1.6370226052761176;
    var polynomial = -0.6436043220444363;
    polynomial = polynomial * t + 2.3055367025740328;
    polynomial = polynomial * t - 3.3613206236720226;
    polynomial = polynomial * t + 2.545964004001259;
    polynomial = polynomial * t - 1.0665895817036013;
    polynomial = polynomial * t + 0.23936498051552585;
    polynomial = polynomial * t - 0.031230460838329886;
    polynomial = polynomial * t + 0.023843501944397688;
    polynomial = polynomial * t - 0.25000746556146003;
    var x = clamp(cdf_target + cdf_target * cdf_target * polynomial, 0.0, opacity);
    for (var iteration = 0u; iteration < 2u; iteration += 1u) {
        var derivative = 1.0 + x * (0.5 + x * (1.0 / 3.0 + x * 0.25));
        if x >= 0.01 {
            derivative = -log(1.0 - x) / x;
        }
        x = clamp(x - (point_dilog(x) - cdf_target) / derivative, 0.0, opacity);
    }
    return sqrt(max(0.0, -2.0 * log(clamp(x / opacity, 1e-30, 1.0))));
}

fn point_poisson_log_probability(k: f32, lambda: f32) -> f32 {
    if k == 0.0 {
        return -lambda;
    }
    if k < 10.0 {
        var log_factorial = 0.0;
        for (var n = 2.0; n <= k; n += 1.0) {
            log_factorial += log(n);
        }
        return k * log(lambda) - lambda - log_factorial;
    }
    // Avoid cancellation in k*log(lambda)-lambda-log(k!) for large nearby k, lambda.
    var deviance = k * log(k / lambda) + lambda - k;
    if abs(k - lambda) < 0.1 * (k + lambda) {
        let v = (k - lambda) / (k + lambda);
        let square = v * v;
        var term = 2.0 * k * v;
        deviance = (k - lambda) * v;
        for (var j = 1u; j <= 6u; j += 1u) {
            term *= square;
            deviance += term / f32(2u * j + 1u);
        }
    }
    let inverse = 1.0 / k;
    let square = inverse * inverse;
    let stirling_error = inverse * (1.0 / 12.0
        + square * (-1.0 / 360.0 + square / 1260.0));
    return -deviance - stirling_error - 0.5 * log(6.283185307179586 * k);
}

fn point_poisson(lambda: f32, seed: u32) -> u32 {
    if !(lambda >= 0.0 && lambda <= POINT_MAX_POISSON_RATE) {
        return POINT_POISSON_FAILURE;
    }
    if lambda == 0.0 {
        return 0u;
    }
    if lambda < 10.0 {
        // Inverse CDF, not a normal approximation: accurate also for subpixel splats.
        let u = point_random(seed, 0u);
        var probability = exp(-lambda);
        var cumulative = probability;
        for (var k = 0u; k < 128u; k += 1u) {
            if u <= cumulative {
                return k;
            }
            probability *= lambda / f32(k + 1u);
            cumulative += probability;
        }
        return POINT_POISSON_FAILURE;
    }
    // Hoermann's PTRS (1993), doi:10.1016/0167-6687(93)90997-4.
    // Bounded dispatch time, with an explicit failure instead of a biased count.
    let b = 0.931 + 2.53 * sqrt(lambda);
    let a = -0.059 + 0.02483 * b;
    let inverse_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    for (var attempt = 0u; attempt < 64u; attempt += 1u) {
        let u = point_random(seed, 2u * attempt) - 0.5;
        let v = point_random(seed, 2u * attempt + 1u);
        let us = 0.5 - abs(u);
        let k = floor((2.0 * a / us + b) * u + lambda + 0.43);
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        if us >= 0.07 && v <= vr {
            return u32(k);
        }
        if log(v * inverse_alpha / (a / (us * us) + b))
            <= point_poisson_log_probability(k, lambda) {
            return u32(k);
        }
    }
    return POINT_POISSON_FAILURE;
}

fn point_poisson_process(lambda: f32, seed: u32, limit: u32) -> u32 {
    if !(lambda >= 0.0 && lambda <= POINT_MAX_POISSON_RATE * f32(POINT_MAX_POISSON_COMPONENTS)) {
        return POINT_POISSON_FAILURE;
    }
    // Keep the established small-process random stream unchanged.
    if lambda <= POINT_MAX_POISSON_RATE { return point_poisson(lambda, seed); }
    var remaining = lambda;
    var total = 0u;
    for (var component = 0u; component < POINT_MAX_POISSON_COMPONENTS; component += 1u) {
        if remaining == 0.0 { break; }
        let rate = min(remaining, POINT_MAX_POISSON_RATE);
        let count = point_poisson(rate, point_hash(seed ^ point_hash(component + 0xd1b54a35u)));
        if count == POINT_POISSON_FAILURE { return count; }
        // Remaining components are nonnegative: this outcome is already known
        // to exceed the hard work ceiling. The whole image will be rejected.
        if count > limit - total { return limit + 1u; }
        total += count;
        remaining -= rate;
    }
    return total;
}
