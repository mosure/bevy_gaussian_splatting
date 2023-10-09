#import bevy_render::globals    Globals
#import bevy_render::view       View

#import bevy_gaussian_splatting::spherical_harmonics    compute_color_from_sh_3_degree


struct GaussianInput {
    @location(0) position: vec3<f32>,
    @location(1) log_scale: vec3<f32>,
    @location(2) rot: vec4<f32>,
    @location(3) opacity_logit: f32,
    sh: array<vec3<f32>, #{MAX_SH_COEFF_COUNT}>,
};

struct GaussianOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) conic_and_opacity: vec4<f32>,
};

struct GaussianUniforms {
    global_scale: f32,
    transform: mat4x4<f32>,
};


@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;

@group(1) @binding(0) var<uniform> uniforms: GaussianUniforms;
@group(1) @binding(1) var<storage, read> points: array<GaussianInput>;


fn sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        let z = exp(x);
        return z / (1.0 + z);
    }
}

// TODO: precompute cov3d
fn compute_cov3d(log_scale: vec3<f32>, rot: vec4<f32>) -> array<f32, 6> {
    let modifier = uniforms.global_scale;
    let S = mat3x3<f32>(
        exp(log_scale.x) * modifier, 0.0, 0.0,
        0.0, exp(log_scale.y) * modifier, 0.0,
        0.0, 0.0, exp(log_scale.z) * modifier,
    );

    let r = rot.x;
    let x = rot.y;
    let y = rot.z;
    let z = rot.w;

    let R = mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
        2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y),
    );

    let M = S * R;
    let Sigma = transpose(M) * M;

    return array<f32, 6>(
        Sigma[0][0],
        Sigma[0][1],
        Sigma[0][2],
        Sigma[1][1],
        Sigma[1][2],
        Sigma[2][2],
    );
}

fn compute_cov2d(position: vec3<f32>, log_scale: vec3<f32>, rot: vec4<f32>) -> vec3<f32> {
    let cov3d = compute_cov3d(log_scale, rot);

    var t = view.view * vec4<f32>(position, 1.0);

    let focal_x = view.projection[0][0];
    let focal_y = view.projection[1][1];

    let aspect_ratio = focal_x / focal_y;
    let tan_fovy = 1.0 / focal_y;
    let tan_fovx = tan_fovy * aspect_ratio;

    let limx = 1.3 * tan_fovx;
    let limy = 1.3 * tan_fovy;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;

    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    let J = mat4x4(
        focal_x / t.z, 0.0, -(focal_x * t.x) / (t.z * t.z), 0.0,
        0.0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z), 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    );

    let W = transpose(view.view);

    let T = W * J;

    let Vrk = mat4x4(
        cov3d[0], cov3d[1], cov3d[2], 0.0,
        cov3d[1], cov3d[3], cov3d[4], 0.0,
        cov3d[2], cov3d[4], cov3d[5], 0.0,
        0.0, 0.0, 0.0, 0.0,
    );

    var cov = transpose(T) * transpose(Vrk) * T;

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;

    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}


@vertex
fn vs_points(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> GaussianOutput {
    // TODO: size may need to be 6 for aabb?
    var quad_vertices = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );

    var output: GaussianOutput;
    let quad_index = vertex_index % 4u;
    let quad_offset = quad_vertices[quad_index];
    let point = points[instance_index];

    let cov2d = compute_cov2d(point.position, point.log_scale, point.rot);
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let lambda_1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda_2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius_px = ceil(3.0 * sqrt(max(lambda_1, lambda_2)));
    let radius_ndc = vec2<f32>(
        radius_px / (view.viewport.w), // TODO: test viewport.z swap
        radius_px / (view.viewport.z),
    );
    output.conic_and_opacity = vec4<f32>(conic, sigmoid(point.opacity_logit));

    var projPosition = view.projection * vec4<f32>(point.position, 1.0);
    projPosition = projPosition / projPosition.w;
    output.position = vec4<f32>(projPosition.xy + 2.0 * radius_ndc * quad_offset, projPosition.zw);
    output.color = compute_color_from_sh_3_degree(point.position, point.sh, view.world_position);
    output.uv = radius_px * quad_offset;

    return output;
}

@fragment
fn fs_main(input: GaussianOutput) -> @location(0) vec4<f32> {
    // we want the distance from the gaussian to the fragment while uv
    // is the reverse
    let d = -input.uv;
    let conic = input.conic_and_opacity.xyz;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;
    let opacity = input.conic_and_opacity.w;

    if (power > 0.0) {
        discard;
    }

    let alpha = min(0.99, opacity * exp(power));

    return vec4<f32>(input.color * alpha, alpha);
}
