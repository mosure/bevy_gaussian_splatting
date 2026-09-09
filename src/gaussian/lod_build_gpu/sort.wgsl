// Bounded canonical Morton sort. Separate compute passes provide the global
// memory dependency between bitonic stages without device-scope barriers.

const SH_PLANES: u32 = __SH_VEC4_PLANES__u;
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

struct GaussianInput {
    position_visibility: vec4<f32>,
    spherical_harmonic: array<vec4<f32>, __SH_VEC4_PLANES__>,
    rotation: vec4<f32>,
    scale_opacity: vec4<f32>,
}

struct GlobalParams {
    // record_count, padded_count, reserved
    counts: vec4<u32>,
    normalization_min: vec4<f32>,
    normalization_max: vec4<f32>,
}

struct StageParams {
    // k, j, reserved
    first: vec4<u32>,
}

struct SortEntry {
    // Morton low/high, source index low/high.
    key_and_source: vec4<u32>,
    // Source input index, valid flag, reserved.
    input_and_valid: vec4<u32>,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<uniform> stage: StageParams;
@group(0) @binding(2) var<storage, read> inputs: array<GaussianInput>;
@group(0) @binding(3) var<storage, read_write> entries: array<SortEntry>;
@group(0) @binding(4) var<storage, read_write> sorted: array<GaussianInput>;
@group(0) @binding(5) var<storage, read_write> statuses: array<u32>;

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

fn compare_u32(left: u32, right: u32) -> i32 {
    if (left < right) { return -1; }
    if (left > right) { return 1; }
    return 0;
}

fn ordered_float(value: f32) -> u32 {
    // The host canonicalizes signed zero before upload. Keep this comparison
    // bit-only: arithmetic/equality on a subnormal may be flushed to zero by
    // the device, collapsing distinct canonical CPU payload keys.
    let bits = bitcast<u32>(value);
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
    if (index >= globals.counts.x) { return; }
    // Sort entries, including invalid padding, are initialized by the host.
    // The GPU consumes only the uploaded integer Morton/source tuple and must
    // never derive a package-ordering key from adapter floating-point math.
    statuses[index] = validate_gaussian(inputs[index]);
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
