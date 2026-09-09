#define_import_path bevy_gaussian_splatting::lod_spatial_metrics

// Shared current-view projection and quality pressure. Components are
// {pressure, scheduling priority, near-support crossing, conservative visibility}.
// The same function supplies discrete admission and spatial transition weights.
const FINITE_MAX: f32 = 3.402823e+38;

// Whole-node rejection mirrors the production world-support sphere, including
// its Mip footprint. The host admits only authenticated >=3-sigma sphere-union
// bounds and supplies a conservative transform inflation. Invalid arithmetic
// retains the node. `envelope` is two for a parent/child covariance transition.
fn lod_support_outside_view(
    center_radius: vec4<f32>, half_extents: vec4<f32>,
    world_from_local: mat4x4<f32>, clip_from_world: mat4x4<f32>,
    view: vec4<f32>, omission: vec4<f32>, planes: array<vec4<f32>, 6>, envelope: f32,
) -> bool {
    if !(omission.x >= 1.0) { return false; }
    let tolerance = 0.00002 * max(center_radius.w, 1.0);
    let extent = (half_extents.xyz + vec3<f32>(tolerance)) * (omission.x * envelope);
    let rows = transpose(clip_from_world);
    let transform = transpose(world_from_local);
    let center = vec4<f32>(center_radius.xyz, 1.0);
    var mip = omission.y;
    if omission.z != 0.0 {
        let depth_plane = transform * rows[3];
        let depth = abs(dot(depth_plane, center)) + dot(abs(depth_plane.xyz), extent);
        mip *= depth;
    }
    let margin = mip + max(view.w, 0.0);
    if !(margin >= 0.0 && margin <= FINITE_MAX)
        || any(!(extent >= vec3<f32>(0.0))) || any(!(extent <= vec3<f32>(FINITE_MAX))) { return false; }
    let absolute_transform = transpose(mat4x4<f32>(abs(world_from_local[0]),
        abs(world_from_local[1]), abs(world_from_local[2]), abs(world_from_local[3])));
    for (var plane = 0u; plane < 6u; plane += 1u) {
        let normal_length = length(planes[plane].xyz);
        if !(normal_length > 0.0 && normal_length <= FINITE_MAX) { continue; }
        let world_plane = planes[plane] / normal_length;
        let local_plane = transform * world_plane;
        let distance = dot(local_plane, center);
        let support = dot(abs(local_plane.xyz), extent) + margin;
        let magnitude = absolute_transform * abs(world_plane);
        let guard = 0.00000762939453125 * max(dot(magnitude,
            vec4<f32>(abs(center_radius.xyz) + extent, 1.0)) + margin, 1.0);
        if distance + support < -guard { return true; }
    }
    return false;
}

fn lod_safe_product(extent: f32, scale: f32) -> f32 {
    if extent == 0.0 { return 0.0; }
    return min(extent * scale, FINITE_MAX);
}

fn lod_ratio(numerator: f32, denominator: f32) -> f32 {
    if denominator <= 0.0 { return select(FINITE_MAX, 0.0, numerator <= 0.0); }
    return min(numerator / denominator, FINITE_MAX);
}

fn lod_screen_norm(x: vec3<f32>, y: vec3<f32>) -> f32 {
    let scale = max(max(max(abs(x.x), abs(x.y)), abs(x.z)), max(max(abs(y.x), abs(y.y)), abs(y.z)));
    if scale == 0.0 { return 0.0; }
    let a = x / scale;
    let b = y / scale;
    let aa = dot(a, a);
    let ab = dot(a, b);
    let bb = dot(b, b);
    return scale * sqrt(0.5 * (aa + bb + sqrt((aa - bb) * (aa - bb) + 4.0 * ab * ab)));
}

// Continuous physical scheduling estimate, independent of quality certificates.
// Near-bounded projection avoids saturating merely because an enclosing sphere
// crosses the camera. This coordinate ranks work; it is not an image-error bound.
fn lod_finite_scheduling_score(
    center_radius: vec4<f32>, half_extents: vec4<f32>, error_quality: vec4<f32>,
    world_from_local: mat4x4<f32>, clip_from_world: mat4x4<f32>, view: vec4<f32>,
) -> f32 {
    let score_max = 1.0e20;
    let center = world_from_local * vec4<f32>(center_radius.xyz, 1.0);
    let rows = transpose(clip_from_world);
    let pixel_x = rows[0] * (0.5 * view.x);
    let pixel_y = rows[1] * (0.5 * view.y);
    let gradient = rows[3].xyz;
    let gradient_squared = dot(gradient, gradient);
    let w = dot(rows[3], center);
    var support_scale = lod_screen_norm(pixel_x.xyz, pixel_y.xyz) / max(abs(w), 1.0e-12);
    var error_scale = support_scale;
    if gradient_squared > 0.0 {
        // For perspective projection z_clip = depth_ratio*w_clip + offset.
        // The near plane z_clip=w_clip therefore has this positive w value.
        let depth_ratio = dot(rows[2].xyz, gradient) / gradient_squared;
        let near_w = max(abs((rows[2].w - depth_ratio * rows[3].w) / (1.0 - depth_ratio)), 1.0e-12);
        let center_w = max(w, near_w);
        let local_gradient = transpose(world_from_local) * vec4<f32>(gradient, 0.0);
        let support_w = dot(abs(local_gradient.xyz), half_extents.xyz);
        let minimum_w = max(w - support_w, near_w);
        let row_x = pixel_x.xyz - gradient * (dot(pixel_x, center) / center_w);
        let row_y = pixel_y.xyz - gradient * (dot(pixel_y, center) / center_w);
        let norm = lod_screen_norm(row_x, row_y);
        support_scale = norm / minimum_w;
        let error_minimum_w = max(minimum_w - error_quality.x * view.z * sqrt(gradient_squared), near_w);
        error_scale = (norm / error_minimum_w) * (center_w / error_minimum_w);
    }
    let severity = max(lod_safe_product(error_quality.x * view.z, error_scale),
        lod_safe_product(center_radius.w * view.z, support_scale) * 0.125);
    if !(severity >= 0.0 && severity <= FINITE_MAX) { return score_max; }
    // A positive floor remains representable through all 64 ancestry halvings.
    return clamp(severity, 1.0e-12, score_max);
}

fn lod_projection_metric(
    center_radius: vec4<f32>, half_extents: vec4<f32>, error_quality: vec4<f32>, original: u32,
    world_from_local: mat4x4<f32>, clip_from_world: mat4x4<f32>, view: vec4<f32>, quality: vec4<f32>,
) -> vec4<f32> {
    if quality.z == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let center = world_from_local * vec4<f32>(center_radius.xyz, 1.0);
    let radius = center_radius.w * view.z;
    let error = error_quality.x * view.z;
    let rows = transpose(clip_from_world);
    let planes = array<vec4<f32>, 6>(rows[3] + rows[0], rows[3] - rows[0],
        rows[3] + rows[1], rows[3] - rows[1], rows[3] - rows[2], rows[2]);
    let local_center = vec4<f32>(center_radius.xyz, 1.0);
    let extent_magnitude = vec4<f32>(abs(center_radius.xyz) + half_extents.xyz, 1.0);
    let margin = max(view.w, 0.0);
    let local_from_world_plane = transpose(world_from_local);
    let absolute_transform = transpose(mat4x4<f32>(
        abs(world_from_local[0]), abs(world_from_local[1]),
        abs(world_from_local[2]), abs(world_from_local[3])));
    var crosses_near = false;
    for (var plane = 0u; plane < 6u; plane += 1u) {
        if quality.w == 0.0 && plane != 4u { continue; }
        let normal_length = length(planes[plane].xyz);
        // Infinite reverse-Z has no far plane. Invalid plane arithmetic
        // retains refinement; the ordinary conservative projection still ranks it.
        if !(normal_length > 0.0 && normal_length <= FINITE_MAX) { continue; }
        let world_plane = planes[plane] / normal_length;
        let local_plane = local_from_world_plane * world_plane;
        let magnitude = absolute_transform * abs(world_plane);
        let support = dot(abs(local_plane.xyz), half_extents.xyz) + margin;
        let distance = dot(local_plane, local_center);
        // Match LodLocalFrustum's conservative rounding guard. A sphere crossing
        // this plane does not suffice: thin/deep AABBs can miss it entirely.
        let allowance = 0.00000762939453125 * max(dot(magnitude, extent_magnitude) + margin, 1.0);
        if quality.w != 0.0 && distance + support < -allowance { return vec4<f32>(0.0); }
        if plane == 4u { crosses_near = abs(distance) <= support + allowance; }
    }
    let pixel_x = rows[0] * (0.5 * view.x);
    let pixel_y = rows[1] * (0.5 * view.y);
    let gradient = rows[3].xyz;
    let gradient_length = length(gradient);
    let w = dot(rows[3], center);
    var support_scale = FINITE_MAX;
    var error_scale = FINITE_MAX;
    if gradient_length == 0.0 {
        support_scale = lod_screen_norm(pixel_x.xyz, pixel_y.xyz) / abs(w);
        error_scale = support_scale;
    } else if w > 0.0 {
        let row_x = pixel_x.xyz - gradient * (dot(pixel_x, center) / w);
        let row_y = pixel_y.xyz - gradient * (dot(pixel_y, center) / w);
        let norm = lod_screen_norm(row_x, row_y);
        let near_distance = dot(planes[4], center) / length(planes[4].xyz);
        let minimum_w = w - radius * gradient_length;
        if minimum_w > 0.0 && near_distance > radius { support_scale = norm / minimum_w; }
        let error_minimum = w - (radius + error) * gradient_length;
        if error_minimum > 0.0 && near_distance > radius + error {
            error_scale = (norm / error_minimum) * (w / error_minimum);
        }
    }
    let projected_error = lod_safe_product(error, error_scale);
    // Footprint breaks otherwise-zero error ties: 16 support-diameter pixels
    // count as one error pixel for scheduling, without changing eligibility.
    let severity = max(projected_error, lod_safe_product(radius, support_scale) * 0.125);
    let severity_bucket = 1u + u32(clamp(floor(log2(max(severity, 1.0))) * 0.5, 0.0, 5.0));
    let priority = select(severity_bucket, 7u, crosses_near);
    if quality.z == 2.0 { return vec4<f32>(FINITE_MAX, f32(priority), f32(crosses_near), 1.0); }
    let coverage = clamp(2.0 * lod_safe_product(radius, support_scale) / view.y, 0.0, 1.0);
    let detail = quality.x;
    let fidelity = smoothstep(0.90, 0.99, detail);
    let structural = lod_ratio(detail * (coverage + (1.0 - coverage) * fidelity), error_quality.y);
    let projected = lod_ratio(projected_error, quality.y);
    let normalized = clamp(detail / 0.99, 0.0, 1.0);
    var pressure = max(min(structural, projected), normalized * normalized * normalized * projected);
    let certificate = error_quality.z;
    if certificate <= 1.0 / 65535.0 {
        if detail >= 0.95 && original == 0u { pressure = FINITE_MAX; }
    } else {
        let authority = clamp(detail / 0.95, 0.0, 1.0);
        let effective_coverage = coverage + (1.0 - coverage) * authority * authority * authority;
        let demand = smoothstep(0.90, 0.95, detail) * detail * authority * effective_coverage;
        pressure = max(pressure, lod_ratio(demand, certificate));
    }
    return vec4<f32>(pressure, f32(priority), f32(crosses_near), 1.0);
}
