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
    global_scale: f32,
    transform: f32,
};


@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;

@group(1) @binding(0) var<uniform> uniforms: GaussianUniforms;

@group(2) @binding(0) var<storage, read> points: array<GaussianInput>;


// https://github.com/cvlab-epfl/gaussian-splatting-web/blob/905b3c0fb8961e42c79ef97e64609e82383ca1c2/src/shaders.ts#L185
// TODO: precompute
fn compute_cov3d(scale: vec3<f32>, rot: vec4<f32>) -> array<f32, 6> {
    let modifier = uniforms.global_scale;
    let S = mat3x3<f32>(
        scale.x * modifier, 0.0, 0.0,
        0.0, scale.y * modifier, 0.0,
        0.0, 0.0, scale.z * modifier,
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

fn compute_cov2d(position: vec3<f32>, scale: vec3<f32>, rot: vec4<f32>) -> vec3<f32> {
    let cov3d = compute_cov3d(scale, rot);

    var t = view.inverse_view * vec4<f32>(position, 1.0);

    let focal_x = view.projection[0][0];
    let focal_y = view.projection[1][1];

    let limx = 1.3 * 0.5 * view.viewport.z / focal_x;
    let limy = 1.3 * 0.5 * view.viewport.w / focal_y;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;

    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    let J = mat4x4(
        focal_x / t.z, 0.0, -(focal_x * t.x) / (t.z * t.z), 0.0,
        0.0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z), 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    );

    let W = transpose(view.inverse_view);

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



// https://github.com/Lichtso/splatter/blob/c6b7a3894c25578cd29c9761619e4f194449e389/src/shaders.wgsl#L125-L169
fn quat_to_mat(p: vec4<f32>) -> mat3x3<f32> {
  var q = p * sqrt(2.0);
  var yy = q.y * q.y;
  var yz = q.y * q.z;
  var yw = q.y * q.w;
  var yx = q.y * q.x;
  var zz = q.z * q.z;
  var zw = q.z * q.w;
  var zx = q.z * q.x;
  var ww = q.w * q.w;
  var wx = q.w * q.x;
  return mat3x3<f32>(
    1.0 - zz - ww, yz + wx, yw - zx,
    yz - wx, 1.0 - yy - ww, zw + yx,
    yw + zx, zw - yx, 1.0 - yy - zz,
  );
}

fn projected_covariance_of_ellipsoid(scale: vec3<f32>, rotation: vec4<f32>, translation: vec3<f32>) -> mat3x3<f32> {
    let camera_matrix = mat3x3<f32>(
        view.view.x.xyz,
        view.view.y.xyz,
        view.view.z.xyz
    );
    var transform = quat_to_mat(rotation);
    transform.x *= scale.x;
    transform.y *= scale.y;
    transform.z *= scale.z;

    // 3D Covariance
    var view_pos = view.view * vec4<f32>(translation, 1.0);
    view_pos.x = clamp(view_pos.x / view_pos.z, -1.0, 1.0) * view_pos.z;
    view_pos.y = clamp(view_pos.y / view_pos.z, -1.0, 1.0) * view_pos.z;
    let T = transpose(transform) * camera_matrix * mat3x3(
        1.0 / view_pos.z, 0.0, -view_pos.x / (view_pos.z * view_pos.z),
        0.0, 1.0 / view_pos.z, -view_pos.y / (view_pos.z * view_pos.z),
        0.0, 0.0, 0.0,
    );
    let covariance_matrix = transpose(T) * T;

    return covariance_matrix;
}

fn projected_contour_of_ellipsoid(scale: vec3<f32>, rotation: vec4<f32>, translation: vec3<f32>) -> mat3x3<f32> {
    let camera_matrix = mat3x3<f32>(
        view.inverse_view.x.xyz,
        view.inverse_view.y.xyz,
        view.inverse_view.z.xyz
    );

    var transform = quat_to_mat(rotation);
    transform.x /= scale.x;
    transform.y /= scale.y;
    transform.z /= scale.z;

    let ray_origin = view.world_position - translation;
    let local_ray_origin = ray_origin * transform;
    let local_ray_origin_squared = local_ray_origin * local_ray_origin;

    let diagonal = 1.0 - local_ray_origin_squared.yxx - local_ray_origin_squared.zzy;
    let triangle = local_ray_origin.yxx * local_ray_origin.zzy;

    let A = mat3x3<f32>(
        diagonal.x, triangle.z, triangle.y,
        triangle.z, diagonal.y, triangle.x,
        triangle.y, triangle.x, diagonal.z,
    );

    transform = transpose(camera_matrix) * transform;
    let M = transform * A * transpose(transform);

    return M;
}

fn extract_translation_of_ellipse(M: mat3x3<f32>) -> vec2<f32> {
    let discriminant = M.x.x * M.y.y - M.x.y * M.x.y;
    let inverse_discriminant = 1.0 / discriminant;
    return vec2<f32>(
        M.x.y * M.y.z - M.y.y * M.x.z,
        M.x.y * M.x.z - M.x.x * M.y.z,
    ) * inverse_discriminant;
}

fn extract_rotation_of_ellipse(M: mat3x3<f32>) -> vec2<f32> {
    let a = (M.x.x - M.y.y) * (M.x.x - M.y.y);
    let b = a + 4.0 * M.x.y * M.x.y;
    let c = 0.5 * sqrt(a / b);
    var j = sqrt(0.5 - c);
    var k = -sqrt(0.5 + c) * sign(M.x.y) * sign(M.x.x - M.y.y);
    if(M.x.y < 0.0 || M.x.x - M.y.y < 0.0) {
        k = -k;
        j = -j;
    }
    if(M.x.x - M.y.y < 0.0) {
        let t = j;
        j = -k;
        k = t;
    }
    return vec2<f32>(j, k);
}

fn extract_scale_of_ellipse(M: mat3x3<f32>, translation: vec2<f32>, rotation: vec2<f32>) -> vec2<f32> {
    let d = 2.0 * M.x.y * rotation.x * rotation.y;
    let e = M.z.z - (M.x.x * translation.x * translation.x + M.y.y * translation.y * translation.y + 2.0 * M.x.y * translation.x * translation.y);
    let semi_major_axis = sqrt(abs(e / (M.x.x * rotation.y * rotation.y + M.y.y * rotation.x * rotation.x - d)));
    let semi_minor_axis = sqrt(abs(e / (M.x.x * rotation.x * rotation.x + M.y.y * rotation.y * rotation.y + d)));

    return vec2<f32>(semi_major_axis, semi_minor_axis);
}

fn extract_scale_of_covariance(M: mat3x3<f32>) -> vec2<f32> {
    let a = (M.x.x - M.y.y) * (M.x.x - M.y.y);
    let b = sqrt(a + 4.0 * M.x.y * M.x.y);
    let semi_major_axis = sqrt((M.x.x + M.y.y + b) * 0.5);
    let semi_minor_axis = sqrt((M.x.x + M.y.y - b) * 0.5);
    return vec2<f32>(semi_major_axis, semi_minor_axis);
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

fn view_dimensions(projection: mat4x4<f32>) -> vec2<f32> {
    let near = projection[2][3] / (projection[2][2] + 1.0);
    let right = near / projection[0][0];
    let top = near / projection[1][1];

    return vec2<f32>(2.0 * right, 2.0 * top);
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
    let radius_px = ceil(3.0 * sqrt(max(lambda_1, lambda_2)));
    let radius_ndc = vec2<f32>(
        radius_px / f32(view.viewport.w),
        radius_px / f32(view.viewport.z),
    );

    output.uv = radius_px * quad_offset;

    var projected_position = view.view_proj * vec4<f32>(point.position, 1.0);
    projected_position = projected_position / projected_position.w;

    output.position = vec4<f32>(
        projected_position.xy + 2.0 * radius_ndc * quad_offset,
        projected_position.zw,
    );

    // let M = projected_contour_of_ellipsoid(
    //     point.scale * uniforms.global_scale,
    //     point.rotation,
    //     point.position,
    // );
    // let translation = extract_translation_of_ellipse(M);
    // let rotation = extract_rotation_of_ellipse(M);
    // //let semi_axes = extract_scale_of_ellipse(M, translation, rotation);

    // let covariance = projected_covariance_of_ellipsoid(
    //     point.scale * uniforms.global_scale,
    //     point.rotation,
    //     point.position
    // );
    // let semi_axes = extract_scale_of_covariance(covariance);

    // let view_dimensions = view_dimensions(view.projection);
    // let ellipse_size_bias = 0.2 * view_dimensions.x / f32(view.viewport.z);

    // let transformation = mat3x2<f32>(
    //     vec2<f32>(rotation.y, -rotation.x) * (ellipse_size_bias + semi_axes.x),
    //     vec2<f32>(rotation.x, rotation.y) * (ellipse_size_bias + semi_axes.y),
    //     translation,
    // );

    // let T = mat3x3(
    //     vec3<f32>(transformation.x, 0.0),
    //     vec3<f32>(transformation.y, 0.0),
    //     vec3<f32>(transformation.z, 1.0),
    // );

    // let ellipse_margin = 3.3;  // should be 2.0
    // output.uv = quad_offset * ellipse_margin;
    // output.position = vec4<f32>(
    //     (T * vec3<f32>(output.uv, 1.0)).xy / view_dimensions,
    //     0.0,
    //     1.0,
    // );

    return output;
}

@fragment
fn fs_main(input: GaussianOutput) -> @location(0) vec4<f32> {
    // let power = dot(input.uv, input.uv);
    // let alpha = input.color.a * exp(-0.5 * power);

    // if (alpha < 1.0 / 255.0) {
    //     discard;
    // }

    // return vec4<f32>(input.color.rgb * alpha, alpha);


    let d = -input.uv;
    let conic = input.conic;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;

    if (power > 0.0) {
        discard;
    }

    let alpha = min(0.99, input.color.a * exp(power));
    return vec4<f32>(
        input.color.rgb * alpha,
        alpha,
    );
}
