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

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let trace = cov2d.x + cov2d.z;
    let mid = 0.5 * trace;
    let discriminant = max(0.0, mid * mid - det);

    let term = sqrt(discriminant);

    let lambda1 = mid + term;
    let lambda2 = max(mid - term, 0.0);

    let x_axis_length = sqrt(lambda1);
    let y_axis_length = sqrt(lambda2);

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

    let a = (cov2d.x - cov2d.z) * (cov2d.x - cov2d.z);
    let b = sqrt(a + 4.0 * cov2d.y * cov2d.y);
    let major_radius = sqrt((cov2d.x + cov2d.z + b) * 0.5);
    let minor_radius = sqrt((cov2d.x + cov2d.z - b) * 0.5);

    let bounds = cutoff * vec2<f32>(
        major_radius,
        minor_radius,
    );

    let major_axis_candidate = vec2<f32>(
        -cov2d.y,
        lambda1 - cov2d.x,
    );
    // The analytic eigenvector above is exactly zero for a diagonal
    // x-major (and isotropic) covariance. Choose the canonical x axis in
    // that case instead of normalizing zero to NaN. The perpendicular below
    // deliberately retains the historical negative handedness.
    var eigvec1 = vec2<f32>(1.0, 0.0);
    if abs(major_axis_candidate.x) + abs(major_axis_candidate.y) > 1.0e-12 {
        eigvec1 = normalize(major_axis_candidate);
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
