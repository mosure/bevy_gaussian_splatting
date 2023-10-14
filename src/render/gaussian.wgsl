#import bevy_render::globals    Globals
#import bevy_render::view       View

#import bevy_gaussian_splatting::spherical_harmonics    spherical_harmonics_lookup


struct GaussianInput {
    @location(0) rotation: vec4<f32>,
    @location(1) position: vec3<f32>,
    @location(2) scale: vec3<f32>,
    @location(3) opacity: f32,
    sh: array<f32, #{MAX_SH_COEFF_COUNT}>,
};

struct GaussianOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(linear) uv: vec2<f32>,
};

struct GaussianUniforms {
    transform: mat4x4<f32>,
    global_scale: f32,
};


@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;

@group(1) @binding(0) var<uniform> uniforms: GaussianUniforms;

@group(2) @binding(0) var<storage, read> points: array<GaussianInput>;


// https://github.com/cvlab-epfl/gaussian-splatting-web/blob/905b3c0fb8961e42c79ef97e64609e82383ca1c2/src/shaders.ts#L185
// TODO: precompute
// fn compute_cov3d(scale: vec3<f32>, rot: vec4<f32>) -> array<f32, 6> {
//     let modifier = uniforms.global_scale;
//     let S = mat3x3<f32>(
//         scale.x * modifier, 0., 0.,
//         0., scale.y * modifier, 0.,
//         0., 0., scale.z * modifier,
//     );

//     let r = rot.x;
//     let x = rot.y;
//     let y = rot.z;
//     let z = rot.w;

//     let R = mat3x3<f32>(
//         1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
//         2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
//         2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y),
//     );

//     let M = S * R;
//     let Sigma = transpose(M) * M;

//     return array<f32, 6>(
//         Sigma[0][0],
//         Sigma[0][1],
//         Sigma[0][2],
//         Sigma[1][1],
//         Sigma[1][2],
//         Sigma[2][2],
//     );
// }
fn compute_cov3d(scale: vec3<f32>, rot: vec4<f32>) -> array<f32, 6> {
    let S = scale * uniforms.global_scale;

    let r = rot.x;
    let x = rot.y;
    let y = rot.z;
    let z = rot.w;

    let R = mat3x3<f32>(
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

    let M = mat3x3<f32>(
        S[0] * R.x,
        S[1] * R.y,
        S[2] * R.z,
    );

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

fn compute_cov2d(position: vec3<f32>, scale: vec3<f32>, rot: vec4<f32>) -> vec3<f32> {
    let cov3d = compute_cov3d(scale, rot);
    let Vrk = mat3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    );

    var t = view.inverse_view * vec4<f32>(position, 1.0);

    let focal_x = 500.0;
    let focal_y = 500.0;

    let limx = 1.3 * 0.5 * view.viewport.z / focal_x;
    let limy = 1.3 * 0.5 * view.viewport.w / focal_y;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;

    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    let J = mat3x3(
        focal_x / t.z,
        0.0,
        -(focal_x * t.x) / (t.z * t.z),

        0.0,
        -focal_y / t.z,
        (focal_y * t.y) / (t.z * t.z),

        0.0, 0.0, 0.0,
    );

    let W = transpose(
        mat3x3<f32>(
            view.view.x.xyz,
            view.view.y.xyz,
            view.view.z.xyz,
        )
    );

    let T = W * J;

    var cov = transpose(T) * transpose(Vrk) * T;

    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}


fn world_to_clip(world_pos: vec3<f32>) -> vec4<f32> {
    let homogenous_pos = view.view_proj * vec4<f32>(world_pos, 1.0);
    return vec4<f32>(homogenous_pos.xyz, 1.0) / (homogenous_pos.w + 0.0000001);
}

fn in_frustum(clip_space_pos: vec3<f32>) -> bool {
    return abs(clip_space_pos.x) < 1.1
        && abs(clip_space_pos.y) < 1.1
        && abs(clip_space_pos.z - 0.5) < 0.5;
}


fn get_bounding_box_corner(
    cov2d: vec3<f32>,
    direction: vec2<f32>,
) -> vec2<f32> {
    let T = cov2d.x + cov2d.z;
    let D = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

    let lambda1 = 0.5 * (T + sqrt(T * T - 4.0 * D));
    let lambda2 = 0.5 * (T - sqrt(T * T - 4.0 * D));

    let x_axis_length = sqrt(lambda1);
    let y_axis_length = sqrt(lambda2);

    // let threshold = 0.1;
    // if (abs(lambda1 - lambda2) < threshold) {
    //     return vec2<f32>(
    //         direction.x * (x_axis_length + y_axis_length) * 0.5,
    //         direction.y * x_axis_length
    //     ) / view.viewport.zw;
    // }

    let eigvec1 = normalize(vec2<f32>(
        cov2d.y,
        lambda1 - cov2d.x
    ));

    let scaled_vertex = vec2<f32>(
        direction.x * x_axis_length,
        direction.y * y_axis_length
    ) / view.viewport.zw;

    let rotated_vertex = vec2<f32>(
        eigvec1.x * scaled_vertex.x - eigvec1.y * scaled_vertex.y,
        eigvec1.y * scaled_vertex.x + eigvec1.x * scaled_vertex.y
    );

    return rotated_vertex;
}


@vertex
fn vs_points(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> GaussianOutput {
    var output: GaussianOutput;
    let point = points[instance_index];

    if (!in_frustum(world_to_clip(point.position).xyz)) {
        output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return output;
    }

    var quad_vertices = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    let quad_index = vertex_index % 4u;
    let quad_offset = quad_vertices[quad_index];

    let ray_direction = normalize(point.position - view.world_position);
    output.color = vec4<f32>(
        spherical_harmonics_lookup(ray_direction, point.sh),
        point.opacity
    );

    let cov2d = compute_cov2d(point.position, point.scale, point.rotation);

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let det_inv = 1.0 / det;

    let conic = vec3<f32>(
        cov2d.z * det_inv,
        -cov2d.y * det_inv,
        cov2d.x * det_inv
    );
    output.conic = conic;

    let mid = 0.5 * (cov2d.x + cov2d.z);
    let lambda_1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda_2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius_px = 3.5 * sqrt(max(lambda_1, lambda_2));
    let radius_ndc = vec2<f32>(
        radius_px / view.viewport.z,
        radius_px / view.viewport.w,
    );

    output.uv = (quad_offset + 1.0) / 2.0;

    var projected_position = view.view_proj * vec4<f32>(point.position, 1.0);
    projected_position = projected_position / projected_position.w;

    // output.position = vec4<f32>(
    //     projected_position.xy + 2.0 * radius_ndc * quad_offset,
    //     projected_position.zw,
    // );

    output.position = vec4<f32>(
        projected_position.xy + get_bounding_box_corner(
            cov2d,
            quad_offset,
        ),
        0.0,
        1.0,
    );

    return output;
}

@fragment
fn fs_main(input: GaussianOutput) -> @location(0) vec4<f32> {
    let d = -input.uv;
    let conic = input.conic;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;

    if (power > 0.0) {
        discard;
    }

    if (abs(input.uv.x) > 0.97 || abs(input.uv.y) > 0.97 || abs(input.uv.x) < 0.03 || abs(input.uv.y) < 0.03) {
        return vec4<f32>(
            0.0,
            0.0,
            0.0,
            1.0
        );
    }

    let alpha = min(0.99, input.color.a * exp(power));
    return vec4<f32>(
        input.color.rgb * alpha,
        alpha,
    );
}
