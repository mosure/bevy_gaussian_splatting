#define_import_path bevy_gaussian_splatting::helpers

#import bevy_gaussian_splatting::bindings::{
    view,
    gaussian_uniforms,
}

// `cov2d` uses the full-viewport focal below, so its coordinates contain two
// units per physical pixel. Covariance therefore scales by four. Multiplying
// the public 0.3 physical-pixel variance in render/mod.rs to 1.2 here corrects
// the old coordinate mismatch (which applied only 0.075 physical px^2).
const GAUSSIAN_SHADER_COORDINATE_UNITS_PER_PIXEL: f32 = 2.0;
const GAUSSIAN_MIP_FILTER_VARIANCE_2D_PHYSICAL: f32 = 0.3;
const GAUSSIAN_MIP_FILTER_VARIANCE_2D_SHADER: f32 =
    GAUSSIAN_MIP_FILTER_VARIANCE_2D_PHYSICAL
        * GAUSSIAN_SHADER_COORDINATE_UNITS_PER_PIXEL
        * GAUSSIAN_SHADER_COORDINATE_UNITS_PER_PIXEL;
const GAUSSIAN_FINITE_F32_MAX: f32 = 3.402823e+38;

// Back-to-front center depth, shared by dense radix, LoD compaction, and
// cross-cloud ordering. Signed depths also support orthographic near planes
// behind the eye. Non-finite centers sort to the invalid sentinel.
fn gaussian_depth_sort_key(position_world: vec3<f32>) -> u32 {
    let depth = -(view.view_from_world * vec4<f32>(position_world, 1.0)).z;
    let raw_bits = bitcast<u32>(depth);
    if (raw_bits & 0x7f800000u) == 0x7f800000u { return 0xffffffffu; }
    let bits = select(raw_bits, 0u, depth == 0.0);
    let ascending = select(bits ^ 0x80000000u, ~bits, depth < 0.0);
    return ~ascending;
}

// Soften center-based near clipping over the Gaussian's own three-sigma
// depth support. A center entering the near plane starts with zero opacity;
// wholly interior support retains its authored opacity and covariance. This
// is a current-camera clipping filter, not truncated-volume integration.
fn gaussian_near_clip_weight(position_world: vec3<f32>, covariance: array<f32, 6>) -> f32 {
    let rows = transpose(view.clip_from_view);
    let plane_view = rows[3] - rows[2];
    let position_view = view.view_from_world * vec4<f32>(position_world, 1.0);
    let clearance = dot(plane_view, position_view);
    // Evaluate clearance in view space to avoid a large translated world-plane
    // constant. Plane normalization cancels between clearance and support.
    let plane_world = transpose(view.view_from_world) * vec4<f32>(plane_view.xyz, 0.0);
    let normal = plane_world.xyz;
    let sigma = mat3x3<f32>(
        vec3<f32>(covariance[0], covariance[1], covariance[2]),
        vec3<f32>(covariance[1], covariance[3], covariance[4]),
        vec3<f32>(covariance[2], covariance[4], covariance[5]),
    );
    let variance = dot(normal, sigma * normal);
    if !(clearance > 0.0 && clearance <= GAUSSIAN_FINITE_F32_MAX)
        || !(variance >= 0.0 && variance <= GAUSSIAN_FINITE_F32_MAX) { return 0.0; }
    if variance == 0.0 { return 1.0; }
    let weight = clamp(clearance / (3.0 * sqrt(variance)), 0.0, 1.0);
    return weight * weight * (3.0 - 2.0 * weight);
}

// Converts the filtered Gaussian's screen-space cutoff to a conservative
// world-space sphere margin. `cov2d` measures projected covariance in doubled
// shader coordinates, so the fixed LoD 3-sigma footprint is
// `3 * sqrt(1.2)` shader units. At a perspective depth one shader unit spans
// `abs(view_z) / focal`; orthographic projection omits the depth term. Using
// the smaller focal covers both viewport axes. Invalid projection state returns
// a negative sentinel so both callers retain the splat rather than false-cull.
fn gaussian_mip_support_radius_world(
    position_world: vec3<f32>,
    cutoff: f32,
) -> f32 {
    let viewport_size = view.viewport.zw;
    let focal = abs(vec2<f32>(
        view.clip_from_view[0].x * viewport_size.x,
        view.clip_from_view[1].y * viewport_size.y,
    ));
    let min_focal = min(focal.x, focal.y);
    let mip_radius_shader = cutoff * sqrt(GAUSSIAN_MIP_FILTER_VARIANCE_2D_SHADER);
    if !(viewport_size.x > 0.0 && viewport_size.x <= GAUSSIAN_FINITE_F32_MAX)
        || !(viewport_size.y > 0.0 && viewport_size.y <= GAUSSIAN_FINITE_F32_MAX)
        || !(min_focal > 0.0 && min_focal <= GAUSSIAN_FINITE_F32_MAX)
        || !(mip_radius_shader >= 0.0 && mip_radius_shader <= GAUSSIAN_FINITE_F32_MAX)
    {
        return -1.0;
    }

    var radius_world = mip_radius_shader / min_focal;
    let projection_w = view.clip_from_view[3].w;
    if projection_w == 0.0 {
        let position_view = view.view_from_world * vec4<f32>(position_world, 1.0);
        let depth = abs(position_view.z);
        if !(depth > 0.0 && depth <= GAUSSIAN_FINITE_F32_MAX) {
            return -1.0;
        }
        radius_world = radius_world * depth;
    } else if projection_w != 1.0 {
        return -1.0;
    }
    if !(radius_world >= 0.0 && radius_world <= GAUSSIAN_FINITE_F32_MAX) {
        return -1.0;
    }
    return radius_world;
}

fn gaussian_mip_filter_covariance_2d(covariance: vec3<f32>) -> vec4<f32> {
    let filtered_covariance = vec3<f32>(
        covariance.x + GAUSSIAN_MIP_FILTER_VARIANCE_2D_SHADER,
        covariance.y,
        covariance.z + GAUSSIAN_MIP_FILTER_VARIANCE_2D_SHADER,
    );
    let original_determinant = covariance.x * covariance.z - covariance.y * covariance.y;
    let filtered_determinant = filtered_covariance.x * filtered_covariance.z
        - filtered_covariance.y * filtered_covariance.y;
    let determinant_ratio = original_determinant / filtered_determinant;
    var opacity_scale = 0.0;
    if original_determinant > 0.0
        && filtered_determinant > 0.0
        && determinant_ratio >= 0.0
    {
        opacity_scale = sqrt(clamp(determinant_ratio, 0.0, 1.0));
    }

    return vec4<f32>(filtered_covariance, opacity_scale);
}

fn cov2d(
    position: vec3<f32>,
    cov3d: array<f32, 6>,
) -> vec4<f32> {
    let Vrk = mat3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    );

    var t = view.view_from_world * vec4<f32>(position, 1.0);

    let focal = vec2<f32>(
        view.clip_from_view[0].x * view.viewport.z,
        view.clip_from_view[1].y * view.viewport.w,
    );

    var J: mat3x3<f32>;
    if view.clip_from_view[3].w == 1.0 {
        // Orthographic NDC is affine in view x/y. The full-viewport focal is
        // still in doubled shader coordinates, but it must not vary with depth
        // or couple view-space z into the projected covariance.
        J = mat3x3(
            focal.x, 0.0, 0.0,
            0.0, -focal.y, 0.0,
            0.0, 0.0, 0.0,
        );
    } else {
        let s = 1.0 / (t.z * t.z);
        J = mat3x3(
            focal.x / t.z, 0.0, -(focal.x * t.x) * s,
            0.0, -focal.y / t.z, (focal.y * t.y) * s,
            0.0, 0.0, 0.0,
        );
    }

    let W = transpose(
        mat3x3<f32>(
            view.view_from_world[0].xyz,
            view.view_from_world[1].xyz,
            view.view_from_world[2].xyz,
        )
    );

    let T = W * J;

    let cov = transpose(T) * transpose(Vrk) * T;

    return gaussian_mip_filter_covariance_2d(
        vec3<f32>(cov[0][0], cov[0][1], cov[1][1]),
    );
}

fn get_bounding_box_clip(
    cov2d: vec3<f32>,
    direction: vec2<f32>,
    cutoff: f32,
) -> vec4<f32> {
    // return vec4<f32>(offset, uv);

    // Use one stable eigensystem for both lengths and orientation. Computing
    // lambda-a through mid^2-det loses the small component of an x-major
    // eigenvector and can rotate its support box by 90 degrees after one ULP.
    let covariance_scale = max(max(abs(cov2d.x), abs(cov2d.z)), abs(cov2d.y));
    let covariance = cov2d / select(1.0, covariance_scale, covariance_scale > 0.0);
    let mid = 0.5 * covariance.x + 0.5 * covariance.z;
    let half_difference = 0.5 * covariance.x - 0.5 * covariance.z;
    let difference_scale = max(abs(half_difference), abs(covariance.y));
    var spectral_radius = 0.0;
    if difference_scale > 0.0 {
        spectral_radius = difference_scale * length(vec2<f32>(half_difference, covariance.y) / difference_scale);
    }
    let lambda1 = max(mid + spectral_radius, 0.0);
    let lambda2 = max(mid - spectral_radius, 0.0);
    // Multiplying square roots avoids overflowing a finite support radius
    // merely because the corresponding variance exceeds the f32 range.
    let radius_scale = sqrt(covariance_scale);
    let x_axis_length = radius_scale * sqrt(lambda1);
    let y_axis_length = radius_scale * sqrt(lambda2);

#ifdef USE_AABB
    let radius_px = cutoff * max(x_axis_length, y_axis_length);
    let radius_ndc = vec2<f32>(
        radius_px / view.viewport.zw,
    );

    return vec4<f32>(
        radius_ndc * direction,
        radius_px * direction,
    );
#endif

#ifdef USE_OBB

    let bounds = cutoff * vec2<f32>(x_axis_length, y_axis_length);

    // Choose the well-conditioned eigenvector formula on either side of the
    // diagonal. Only an exactly isotropic covariance needs a canonical axis.
    var major_axis_candidate = vec2<f32>(-covariance.y, spectral_radius - half_difference);
    if covariance.x >= covariance.z {
        major_axis_candidate = vec2<f32>(spectral_radius + half_difference, -covariance.y);
    }
    let vector_scale = max(abs(major_axis_candidate.x), abs(major_axis_candidate.y));
    var eigvec1 = vec2<f32>(1.0, 0.0);
    if vector_scale > 0.0 {
        eigvec1 = normalize(major_axis_candidate / vector_scale);
    }
    let eigvec2 = vec2<f32>(
        eigvec1.y,
        -eigvec1.x
    );

    let rotation_matrix = transpose(
        mat2x2(
            eigvec1,
            eigvec2,
        )
    );

    let scaled_vertex = direction * bounds;
    let rotated_vertex = scaled_vertex * rotation_matrix;

    let scaling_factor = 1.0 / view.viewport.zw;
    let ndc_vertex = rotated_vertex * scaling_factor;

    return vec4<f32>(
        ndc_vertex,
        rotated_vertex,
    );
#endif
}

fn intrinsic_matrix() -> mat3x4<f32> {
    let focal = vec2<f32>(
        view.clip_from_view[0].x * view.viewport.z / 2.0,
        view.clip_from_view[1].y * view.viewport.w / 2.0,
    );

    let Ks = mat3x4<f32>(
        vec4<f32>(focal.x, 0.0, 0.0, (view.viewport.z - 1.0) / 2.0),
        vec4<f32>(0.0, focal.y, 0.0, (view.viewport.w - 1.0) / 2.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );

    return Ks;
}

fn get_rotation_matrix(
    rotation: vec4<f32>,
) -> mat3x3<f32> {
    let r = rotation.x;
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;

    return mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - r * z),
        2.0 * (x * z + r * y),

        2.0 * (x * y + r * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - r * x),

        2.0 * (x * z - r * y),
        2.0 * (y * z + r * x),
        1.0 - 2.0 * (x * x + y * y),
    );
}

fn get_scale_matrix(
    scale: vec3<f32>,
) -> mat3x3<f32> {
    return mat3x3<f32>(
        scale.x * gaussian_uniforms.global_scale, 0.0, 0.0,
        0.0, scale.y * gaussian_uniforms.global_scale, 0.0,
        0.0, 0.0, scale.z * gaussian_uniforms.global_scale,
    );
}
