// Complete bounded GPU LoD hierarchy construction.
//
// The host emits one command per bitonic stage and one command per hierarchy
// level. Separate compute passes provide the global memory dependency between
// stages without relying on non-portable device-scope barriers. Reduction is
// deliberately one invocation per node: leaf_capacity/branching_factor bound
// each sequential loop while independent nodes remain parallel.

const SH_PLANES: u32 = __SH_VEC4_PLANES__u;
const SH_COEFFICIENTS: f32 = __SH_COEFF_COUNT__.0;
const MORTON_BITS_PER_AXIS: u32 = __LOD_MORTON_BITS_PER_AXIS__u;
const MORTON_AXIS_MAX: f32 = __LOD_MORTON_AXIS_MAX__.0;
const EPSILON: f32 = 1.1920929e-7;

const STATUS_POSITION_NON_FINITE: u32 = 1u;
const STATUS_VISIBILITY_NON_FINITE: u32 = 2u;
const STATUS_SH_NON_FINITE: u32 = 4u;
const STATUS_ROTATION_NON_FINITE: u32 = 8u;
const STATUS_DEGENERATE_ROTATION: u32 = 16u;
const STATUS_SCALE_NON_FINITE: u32 = 32u;
const STATUS_NEGATIVE_SCALE: u32 = 64u;
const STATUS_OPACITY_NON_FINITE: u32 = 128u;
const STATUS_OUTSIDE_NORMALIZATION_BOUNDS: u32 = 256u;
const STATUS_DERIVED_NON_FINITE: u32 = 512u;

struct GaussianInput {
    position_visibility: vec4<f32>,
    spherical_harmonic: array<vec4<f32>, __SH_VEC4_PLANES__>,
    rotation: vec4<f32>,
    scale_opacity: vec4<f32>,
}

struct GlobalParams {
    // record_count, padded_count, source_index_base low/high
    counts: vec4<u32>,
    normalization_min: vec4<f32>,
    normalization_max: vec4<f32>,
    // support_sigma, leaf_capacity, branching_factor, reserved
    build: vec4<f32>,
}

struct StageParams {
    // Sort: k, j, 0, 0. Reduction: output start/count, input start/count.
    first: vec4<u32>,
    // Reduction: is_leaf, reserved.
    second: vec4<u32>,
}

struct SortEntry {
    // Morton low/high, source index low/high.
    key_and_source: vec4<u32>,
    // Source input index, valid flag, reserved.
    input_and_valid: vec4<u32>,
}

struct NodeRaw {
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    representative_support_min: vec4<f32>,
    representative_support_max: vec4<f32>,
    // geometric, appearance, opacity, combined
    error: vec4<f32>,
    // Error inherited by the next reduction level.
    summary_error: vec4<f32>,
    // child start/count, canonical source start/count
    topology: vec4<u32>,
    // Morton min low/high, max low/high
    morton: vec4<u32>,
    // page record count, is_leaf, status, reserved
    page: vec4<u32>,
    representative: GaussianInput,
}

struct BoundsPair {
    min: vec3<f32>,
    max: vec3<f32>,
}

struct EigenResult {
    values: vec3<f32>,
    row0: vec3<f32>,
    row1: vec3<f32>,
    row2: vec3<f32>,
}

struct ReductionResult {
    representative: GaussianInput,
    source_min: vec3<f32>,
    source_max: vec3<f32>,
    support_min: vec3<f32>,
    support_max: vec3<f32>,
    error: vec4<f32>,
    status: u32,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<uniform> stage: StageParams;
@group(0) @binding(2) var<storage, read> inputs: array<GaussianInput>;
@group(0) @binding(3) var<storage, read_write> entries: array<SortEntry>;
@group(0) @binding(4) var<storage, read_write> sorted: array<GaussianInput>;
@group(0) @binding(5) var<storage, read_write> statuses: array<u32>;
@group(0) @binding(6) var<storage, read_write> nodes: array<NodeRaw>;

fn finite_scalar(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn finite_vec3(value: vec3<f32>) -> bool {
    return finite_scalar(value.x) && finite_scalar(value.y) && finite_scalar(value.z);
}

fn finite_vec4(value: vec4<f32>) -> bool {
    return finite_scalar(value.x) && finite_scalar(value.y)
        && finite_scalar(value.z) && finite_scalar(value.w);
}

fn next_up(value: f32) -> f32 {
    if (value == 0.0) { return bitcast<f32>(1u); }
    let bits = bitcast<u32>(value);
    return bitcast<f32>(select(bits - 1u, bits + 1u, value > 0.0));
}

fn next_down(value: f32) -> f32 {
    if (value == 0.0) { return bitcast<f32>(0x80000001u); }
    let bits = bitcast<u32>(value);
    return bitcast<f32>(select(bits + 1u, bits - 1u, value > 0.0));
}

fn support_bounds(gaussian: GaussianInput) -> BoundsPair {
    let maximum_scale = max(
        gaussian.scale_opacity.x,
        max(gaussian.scale_opacity.y, gaussian.scale_opacity.z),
    );
    let radius = next_up(globals.build.x * maximum_scale);
    var result: BoundsPair;
    result.min = vec3<f32>(
        next_down(gaussian.position_visibility.x - radius),
        next_down(gaussian.position_visibility.y - radius),
        next_down(gaussian.position_visibility.z - radius),
    );
    result.max = vec3<f32>(
        next_up(gaussian.position_visibility.x + radius),
        next_up(gaussian.position_visibility.y + radius),
        next_up(gaussian.position_visibility.z + radius),
    );
    return result;
}

fn validate_gaussian(gaussian: GaussianInput) -> u32 {
    var result = 0u;
    if (!finite_vec3(gaussian.position_visibility.xyz)) {
        result = result | STATUS_POSITION_NON_FINITE;
    }
    if (!finite_scalar(gaussian.position_visibility.w)) {
        result = result | STATUS_VISIBILITY_NON_FINITE;
    }
    for (var plane = 0u; plane < SH_PLANES; plane = plane + 1u) {
        if (!finite_vec4(gaussian.spherical_harmonic[plane])) {
            result = result | STATUS_SH_NON_FINITE;
        }
    }
    if (!finite_vec4(gaussian.rotation)) {
        result = result | STATUS_ROTATION_NON_FINITE;
    } else if (dot(gaussian.rotation, gaussian.rotation) <= EPSILON) {
        result = result | STATUS_DEGENERATE_ROTATION;
    }
    if (!finite_vec3(gaussian.scale_opacity.xyz)) {
        result = result | STATUS_SCALE_NON_FINITE;
    } else if (any(gaussian.scale_opacity.xyz < vec3<f32>(0.0))) {
        result = result | STATUS_NEGATIVE_SCALE;
    }
    if (!finite_scalar(gaussian.scale_opacity.w)) {
        result = result | STATUS_OPACITY_NON_FINITE;
    }
    if ((result & STATUS_POSITION_NON_FINITE) == 0u
        && (any(gaussian.position_visibility.xyz < globals.normalization_min.xyz)
            || any(gaussian.position_visibility.xyz > globals.normalization_max.xyz))) {
        result = result | STATUS_OUTSIDE_NORMALIZATION_BOUNDS;
    }
    return result;
}

fn quantize_axis(value: f32, lower: f32, upper: f32) -> u32 {
    let extent = upper - lower;
    if (extent <= 0.0) { return 0u; }
    let normalized = clamp((value - lower) / extent, 0.0, 1.0);
    return u32(floor(normalized * MORTON_AXIS_MAX));
}

fn morton_key(position: vec3<f32>) -> vec2<u32> {
    let quantized = vec3<u32>(
        quantize_axis(position.x, globals.normalization_min.x, globals.normalization_max.x),
        quantize_axis(position.y, globals.normalization_min.y, globals.normalization_max.y),
        quantize_axis(position.z, globals.normalization_min.z, globals.normalization_max.z),
    );
    var key = vec2<u32>(0u);
    for (var bit = 0u; bit < MORTON_BITS_PER_AXIS; bit = bit + 1u) {
        for (var axis = 0u; axis < 3u; axis = axis + 1u) {
            let output_bit = 3u * bit + axis;
            let source_bit = (quantized[axis] >> bit) & 1u;
            if (output_bit < 32u) {
                key.x = key.x | (source_bit << output_bit);
            } else {
                key.y = key.y | (source_bit << (output_bit - 32u));
            }
        }
    }
    return key;
}

fn compare_u32(left: u32, right: u32) -> i32 {
    if (left < right) { return -1; }
    if (left > right) { return 1; }
    return 0;
}

fn ordered_float(value: f32) -> u32 {
    var bits = bitcast<u32>(value);
    if (value == 0.0) { bits = 0u; }
    return bits ^ select(0x80000000u, 0xffffffffu, (bits & 0x80000000u) != 0u);
}

fn compare_float(left: f32, right: f32) -> i32 {
    return compare_u32(ordered_float(left), ordered_float(right));
}

fn compare_gaussians(left: GaussianInput, right: GaussianInput) -> i32 {
    for (var component = 0u; component < 4u; component = component + 1u) {
        let ordering = compare_float(left.position_visibility[component], right.position_visibility[component]);
        if (ordering != 0) { return ordering; }
    }
    for (var plane = 0u; plane < SH_PLANES; plane = plane + 1u) {
        for (var component = 0u; component < 4u; component = component + 1u) {
            let ordering = compare_float(
                left.spherical_harmonic[plane][component],
                right.spherical_harmonic[plane][component],
            );
            if (ordering != 0) { return ordering; }
        }
    }
    for (var component = 0u; component < 4u; component = component + 1u) {
        let ordering = compare_float(left.rotation[component], right.rotation[component]);
        if (ordering != 0) { return ordering; }
    }
    for (var component = 0u; component < 4u; component = component + 1u) {
        let ordering = compare_float(left.scale_opacity[component], right.scale_opacity[component]);
        if (ordering != 0) { return ordering; }
    }
    return 0;
}

fn compare_entries(left: SortEntry, right: SortEntry) -> i32 {
    if (left.input_and_valid.y != right.input_and_valid.y) {
        return select(1, -1, left.input_and_valid.y != 0u);
    }
    if (left.input_and_valid.y == 0u) {
        return compare_u32(left.input_and_valid.x, right.input_and_valid.x);
    }
    var ordering = compare_u32(left.key_and_source.y, right.key_and_source.y);
    if (ordering != 0) { return ordering; }
    ordering = compare_u32(left.key_and_source.x, right.key_and_source.x);
    if (ordering != 0) { return ordering; }
    ordering = compare_gaussians(inputs[left.input_and_valid.x], inputs[right.input_and_valid.x]);
    if (ordering != 0) { return ordering; }
    ordering = compare_u32(left.key_and_source.w, right.key_and_source.w);
    if (ordering != 0) { return ordering; }
    return compare_u32(left.key_and_source.z, right.key_and_source.z);
}

@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= globals.counts.y) { return; }
    var entry: SortEntry;
    entry.key_and_source = vec4<u32>(0xffffffffu);
    entry.input_and_valid = vec4<u32>(index, 0u, 0u, 0u);
    if (index < globals.counts.x) {
        let source_low = globals.counts.z + index;
        let carry = select(0u, 1u, source_low < globals.counts.z);
        let source_high = globals.counts.w + carry;
        let gaussian = inputs[index];
        let status = validate_gaussian(gaussian);
        statuses[index] = status;
        let key = morton_key(gaussian.position_visibility.xyz);
        entry.key_and_source = vec4<u32>(key, source_low, source_high);
        entry.input_and_valid.y = 1u;
    }
    entries[index] = entry;
}

@compute @workgroup_size(256)
fn bitonic_stage(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= globals.counts.y) { return; }
    let partner = index ^ stage.first.y;
    if (partner <= index || partner >= globals.counts.y) { return; }
    let left = entries[index];
    let right = entries[partner];
    let ascending = (index & stage.first.x) == 0u;
    let ordering = compare_entries(left, right);
    if ((ascending && ordering > 0) || (!ascending && ordering < 0)) {
        entries[index] = right;
        entries[partner] = left;
    }
}

@compute @workgroup_size(256)
fn gather_sorted(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= globals.counts.x) { return; }
    sorted[index] = inputs[entries[index].input_and_valid.x];
}

fn gaussian_covariance(gaussian: GaussianInput) -> array<vec3<f32>, 3> {
    let q = normalize(gaussian.rotation);
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    var rotation: array<vec3<f32>, 3>;
    rotation[0] = vec3<f32>(1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - w*z), 2.0 * (x*z + w*y));
    rotation[1] = vec3<f32>(2.0 * (x*y + w*z), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - w*x));
    rotation[2] = vec3<f32>(2.0 * (x*z - w*y), 2.0 * (y*z + w*x), 1.0 - 2.0 * (x*x + y*y));
    let scale2 = gaussian.scale_opacity.xyz * gaussian.scale_opacity.xyz;
    var covariance: array<vec3<f32>, 3>;
    covariance[0] = vec3<f32>(0.0);
    covariance[1] = vec3<f32>(0.0);
    covariance[2] = vec3<f32>(0.0);
    for (var row = 0u; row < 3u; row = row + 1u) {
        for (var column = 0u; column < 3u; column = column + 1u) {
            var value = 0.0;
            for (var axis = 0u; axis < 3u; axis = axis + 1u) {
                value = value + rotation[row][axis] * scale2[axis] * rotation[column][axis];
            }
            covariance[row][column] = value;
        }
    }
    return covariance;
}

fn eigendecompose(input0: vec3<f32>, input1: vec3<f32>, input2: vec3<f32>) -> EigenResult {
    var matrix: array<vec3<f32>, 3>;
    matrix[0] = input0;
    matrix[1] = input1;
    matrix[2] = input2;
    var vectors: array<vec3<f32>, 3>;
    vectors[0] = vec3<f32>(1.0, 0.0, 0.0);
    vectors[1] = vec3<f32>(0.0, 1.0, 0.0);
    vectors[2] = vec3<f32>(0.0, 0.0, 1.0);
    for (var iteration = 0u; iteration < 32u; iteration = iteration + 1u) {
        var p = 0u;
        var q = 1u;
        var largest = abs(matrix[0][1]);
        if (abs(matrix[0][2]) > largest) { p = 0u; q = 2u; largest = abs(matrix[0][2]); }
        if (abs(matrix[1][2]) > largest) { p = 1u; q = 2u; }
        let off_diagonal = matrix[p][q];
        let threshold_scale = max(max(abs(matrix[p][p]), abs(matrix[q][q])), 1.0);
        if (abs(off_diagonal) <= 1e-6 * threshold_scale) { break; }
        let tau = (matrix[q][q] - matrix[p][p]) / (2.0 * off_diagonal);
        var t = 1.0;
        if (tau != 0.0) {
            t = sign(tau) / (abs(tau) + sqrt(1.0 + tau * tau));
        }
        let cosine = inverseSqrt(1.0 + t * t);
        let sine = t * cosine;
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        matrix[p][p] = cosine*cosine*app - 2.0*sine*cosine*off_diagonal + sine*sine*aqq;
        matrix[q][q] = sine*sine*app + 2.0*sine*cosine*off_diagonal + cosine*cosine*aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for (var index = 0u; index < 3u; index = index + 1u) {
            if (index != p && index != q) {
                let aip = matrix[index][p];
                let aiq = matrix[index][q];
                matrix[index][p] = cosine * aip - sine * aiq;
                matrix[p][index] = matrix[index][p];
                matrix[index][q] = sine * aip + cosine * aiq;
                matrix[q][index] = matrix[index][q];
            }
            let vip = vectors[index][p];
            let viq = vectors[index][q];
            vectors[index][p] = cosine * vip - sine * viq;
            vectors[index][q] = sine * vip + cosine * viq;
        }
    }

    var values = vec3<f32>(matrix[0][0], matrix[1][1], matrix[2][2]);
    var order = vec3<u32>(0u, 1u, 2u);
    for (var left = 0u; left < 2u; left = left + 1u) {
        var best = left;
        for (var right = left + 1u; right < 3u; right = right + 1u) {
            let rv = values[order[right]];
            let bv = values[order[best]];
            if (rv > bv || (rv == bv && order[right] < order[best])) { best = right; }
        }
        let temporary = order[left];
        order[left] = order[best];
        order[best] = temporary;
    }
    var reordered: array<vec3<f32>, 3>;
    for (var row = 0u; row < 3u; row = row + 1u) {
        reordered[row] = vec3<f32>(
            vectors[row][order.x], vectors[row][order.y], vectors[row][order.z],
        );
    }
    let determinant = dot(reordered[0], cross(reordered[1], reordered[2]));
    if (determinant < 0.0) {
        reordered[0].z = -reordered[0].z;
        reordered[1].z = -reordered[1].z;
        reordered[2].z = -reordered[2].z;
    }
    var result: EigenResult;
    result.values = max(vec3<f32>(values[order.x], values[order.y], values[order.z]), vec3<f32>(0.0));
    result.row0 = reordered[0];
    result.row1 = reordered[1];
    result.row2 = reordered[2];
    return result;
}

// V contains covariance eigenvectors as columns. Encoding V as the stored
// quaternion makes the renderer construct R=V^T and evaluate R^T D R=V D V^T.
fn eigen_rotation(eigen: EigenResult) -> vec4<f32> {
    let r0 = eigen.row0;
    let r1 = eigen.row1;
    let r2 = eigen.row2;
    var w: f32;
    var x: f32;
    var y: f32;
    var z: f32;
    let trace = r0.x + r1.y + r2.z;
    if (trace > 0.0) {
        let s = sqrt(max(trace + 1.0, 0.0)) * 2.0;
        w = 0.25 * s;
        x = (r2.y - r1.z) / s;
        y = (r0.z - r2.x) / s;
        z = (r1.x - r0.y) / s;
    } else if (r0.x > r1.y && r0.x > r2.z) {
        let s = sqrt(max(1.0 + r0.x - r1.y - r2.z, 0.0)) * 2.0;
        w = (r2.y - r1.z) / s;
        x = 0.25 * s;
        y = (r0.y + r1.x) / s;
        z = (r0.z + r2.x) / s;
    } else if (r1.y > r2.z) {
        let s = sqrt(max(1.0 + r1.y - r0.x - r2.z, 0.0)) * 2.0;
        w = (r0.z - r2.x) / s;
        x = (r0.y + r1.x) / s;
        y = 0.25 * s;
        z = (r1.z + r2.y) / s;
    } else {
        let s = sqrt(max(1.0 + r2.z - r0.x - r1.y, 0.0)) * 2.0;
        w = (r1.x - r0.y) / s;
        x = (r0.z + r2.x) / s;
        y = (r1.z + r2.y) / s;
        z = 0.25 * s;
    }
    var quaternion = normalize(vec4<f32>(w, x, y, z));
    if (quaternion.x < 0.0
        || (quaternion.x == 0.0 && (quaternion.y < 0.0
            || (quaternion.y == 0.0 && (quaternion.z < 0.0
                || (quaternion.z == 0.0 && quaternion.w < 0.0)))))) {
        quaternion = -quaternion;
    }
    return quaternion;
}

fn reduction_gaussian(index: u32, leaf: bool) -> GaussianInput {
    if (leaf) { return sorted[index]; }
    return nodes[index].representative;
}

fn reduce_range(start: u32, count: u32, leaf: bool) -> ReductionResult {
    var weight = 0.0;
    var weighted_position = vec3<f32>(0.0);
    var second: array<vec3<f32>, 3>;
    second[0] = vec3<f32>(0.0);
    second[1] = vec3<f32>(0.0);
    second[2] = vec3<f32>(0.0);
    var sh_sum: array<vec4<f32>, __SH_VEC4_PLANES__>;
    var sh_squared: array<vec4<f32>, __SH_VEC4_PLANES__>;
    for (var plane = 0u; plane < SH_PLANES; plane = plane + 1u) {
        sh_sum[plane] = vec4<f32>(0.0);
        sh_squared[plane] = vec4<f32>(0.0);
    }
    var optical_depth = 0.0;
    var min_opacity = 3.4028235e38;
    var max_opacity = -3.4028235e38;
    var max_visibility = -3.4028235e38;
    var source_min = vec3<f32>(3.4028235e38);
    var source_max = vec3<f32>(-3.4028235e38);

    for (var offset = 0u; offset < count; offset = offset + 1u) {
        let gaussian = reduction_gaussian(start + offset, leaf);
        let bounds = support_bounds(gaussian);
        source_min = min(source_min, bounds.min);
        source_max = max(source_max, bounds.max);
        let opacity = clamp(gaussian.scale_opacity.w, 0.0, 1.0);
        let visibility = clamp(gaussian.position_visibility.w, 0.0, 1.0);
        let sample_weight = max(opacity * visibility, 1e-12);
        weight = weight + sample_weight;
        let position = gaussian.position_visibility.xyz;
        weighted_position = weighted_position + sample_weight * position;
        let covariance = gaussian_covariance(gaussian);
        for (var row = 0u; row < 3u; row = row + 1u) {
            for (var column = 0u; column < 3u; column = column + 1u) {
                second[row][column] = second[row][column]
                    + sample_weight * (covariance[row][column] + position[row] * position[column]);
            }
        }
        for (var plane = 0u; plane < SH_PLANES; plane = plane + 1u) {
            let coefficients = gaussian.spherical_harmonic[plane];
            sh_sum[plane] = sh_sum[plane] + sample_weight * coefficients;
            sh_squared[plane] = sh_squared[plane] + sample_weight * coefficients * coefficients;
        }
        let effective_opacity = min(opacity * visibility, 1.0 - EPSILON);
        optical_depth = optical_depth - log(1.0 - effective_opacity);
        min_opacity = min(min_opacity, opacity);
        max_opacity = max(max_opacity, opacity);
        max_visibility = max(max_visibility, gaussian.position_visibility.w);
    }

    let mean = weighted_position / weight;
    var covariance: array<vec3<f32>, 3>;
    for (var row = 0u; row < 3u; row = row + 1u) {
        for (var column = 0u; column < 3u; column = column + 1u) {
            covariance[row][column] = second[row][column] / weight - mean[row] * mean[column];
        }
    }
    covariance[0][1] = 0.5 * (covariance[0][1] + covariance[1][0]);
    covariance[1][0] = covariance[0][1];
    covariance[0][2] = 0.5 * (covariance[0][2] + covariance[2][0]);
    covariance[2][0] = covariance[0][2];
    covariance[1][2] = 0.5 * (covariance[1][2] + covariance[2][1]);
    covariance[2][1] = covariance[1][2];
    let eigen = eigendecompose(covariance[0], covariance[1], covariance[2]);

    var representative: GaussianInput;
    representative.position_visibility = vec4<f32>(mean, max_visibility);
    var appearance_variance = 0.0;
    for (var plane = 0u; plane < SH_PLANES; plane = plane + 1u) {
        let coefficient_mean = sh_sum[plane] / weight;
        representative.spherical_harmonic[plane] = coefficient_mean;
        appearance_variance = appearance_variance
            + dot(max(sh_squared[plane] / weight - coefficient_mean * coefficient_mean, vec4<f32>(0.0)), vec4<f32>(1.0));
    }
    representative.rotation = eigen_rotation(eigen);
    representative.scale_opacity = vec4<f32>(sqrt(eigen.values), clamp(1.0 - exp(-optical_depth), 0.0, 1.0));
    let representative_support = support_bounds(representative);
    let delta = max(abs(mean - source_min), abs(mean - source_max));
    let geometric = next_up(length(delta));
    let appearance = next_up(sqrt(max(appearance_variance / SH_COEFFICIENTS, 0.0)));
    let opacity_error = next_up(max(
        abs(representative.scale_opacity.w - min_opacity),
        abs(representative.scale_opacity.w - max_opacity),
    ));
    let combined = next_up(max(geometric, max(appearance, opacity_error)));
    var status = validate_gaussian(representative);
    if (!finite_vec3(source_min) || !finite_vec3(source_max)
        || !finite_vec3(representative_support.min) || !finite_vec3(representative_support.max)
        || !finite_vec4(vec4<f32>(geometric, appearance, opacity_error, combined))) {
        status = status | STATUS_DERIVED_NON_FINITE;
    }
    var result: ReductionResult;
    result.representative = representative;
    result.source_min = source_min;
    result.source_max = source_max;
    result.support_min = representative_support.min;
    result.support_max = representative_support.max;
    result.error = vec4<f32>(geometric, appearance, opacity_error, combined);
    result.status = status;
    return result;
}

fn add_error(left: vec4<f32>, right: vec4<f32>) -> vec4<f32> {
    var result = vec4<f32>(
        next_up(left.x + right.x),
        next_up(left.y + right.y),
        next_up(left.z + right.z),
        next_up(left.w + right.w),
    );
    result.w = max(result.w, max(result.x, max(result.y, result.z)));
    return result;
}

// Reduce host-planned groups from a globally merged Morton stream. The host
// stores each explicit local (start,count) descriptor in `entries` and batches
// only whole groups, so dispatch boundaries cannot change global topology.
// Leaf inputs live in `sorted`; internal inputs are NodeRaw summaries in
// `nodes[0..stage.first.z]`. Outputs start at stage.first.x, after internal
// inputs, which makes every invocation race-free without device-wide barriers.
@compute @workgroup_size(64)
fn reduce_external_groups(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let output_index = invocation.x;
    if (output_index >= stage.first.y) { return; }
    let descriptor = entries[output_index].input_and_valid;
    let range_start = descriptor.x;
    let range_count = descriptor.y;
    let leaf = stage.second.x != 0u;
    let reduction = reduce_range(range_start, range_count, leaf);

    var node: NodeRaw;
    var node_min = reduction.source_min;
    var node_max = reduction.source_max;
    var inherited_error = vec4<f32>(0.0);
    if (!leaf) {
        let first_child = nodes[range_start];
        node_min = first_child.bounds_min.xyz;
        node_max = first_child.bounds_max.xyz;
        for (var offset = 0u; offset < range_count; offset = offset + 1u) {
            let child = nodes[range_start + offset];
            node_min = min(node_min, child.bounds_min.xyz);
            node_max = max(node_max, child.bounds_max.xyz);
            inherited_error = max(inherited_error, child.summary_error);
        }
    }
    node_min = min(node_min, reduction.support_min);
    node_max = max(node_max, reduction.support_max);
    let accumulated_error = select(
        reduction.error,
        add_error(inherited_error, reduction.error),
        !leaf,
    );
    node.bounds_min = vec4<f32>(node_min, 0.0);
    node.bounds_max = vec4<f32>(node_max, 0.0);
    node.representative_support_min = vec4<f32>(reduction.support_min, 0.0);
    node.representative_support_max = vec4<f32>(reduction.support_max, 0.0);
    // Keep local and accumulated errors independently observable on readback.
    node.error = reduction.error;
    node.summary_error = accumulated_error;
    node.topology = vec4<u32>(range_start, range_count, 0u, 0u);
    node.morton = vec4<u32>(0u);
    // External reduction does not quantize Morton keys, so its placeholder
    // normalization bounds are irrelevant. Preserve every numeric/field status
    // while excluding only that sort-stage domain check.
    let external_status = reduction.status & ~STATUS_OUTSIDE_NORMALIZATION_BOUNDS;
    node.page = vec4<u32>(1u, select(0u, 1u, leaf), external_status, 0u);
    node.representative = reduction.representative;
    nodes[stage.first.x + output_index] = node;
}
