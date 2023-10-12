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
    @location(1) @interpolate(linear) uv: vec2<f32>,
};

struct GaussianUniforms {
    global_scale: f32,
    transform: f32,
};


@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;

@group(1) @binding(0) var<uniform> uniforms: GaussianUniforms;

@group(2) @binding(0) var<storage, read> points: array<GaussianInput>;


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

fn projected_contour_of_ellipsoid(scale: vec3<f32>, rotation: vec4<f32>, translation: vec3<f32>) -> mat3x3<f32> {
    let camera_matrix = mat3x3<f32>(view.inverse_view_proj.x.xyz, view.inverse_view_proj.y.xyz, view.inverse_view_proj.z.xyz);
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


@vertex
fn vs_points(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> GaussianOutput {
    var quad_vertices = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    var output: GaussianOutput;
    let quad_index = vertex_index % 4u;
    let quad_offset = quad_vertices[quad_index];
    let point = points[instance_index];

    let ray_direction = normalize(point.position - view.world_position);
    output.color = vec4<f32>(
        spherical_harmonics_lookup(ray_direction, point.sh),
        point.opacity
    );

    let M = projected_contour_of_ellipsoid(
        point.scale * uniforms.global_scale,
        point.rotation,
        point.position,
    );
    let translation = extract_translation_of_ellipse(M);
    let rotation = extract_rotation_of_ellipse(M);
    let semi_axes = extract_scale_of_ellipse(M, translation, rotation);

    let field_of_view_y = 2.0 * atan(1.0 / view.projection[1][1]);
    let view_height = tan(field_of_view_y / 2.0);
    let view_width = (f32(view.viewport.z) / f32(view.viewport.w)) / view_height;
    let ellipse_size_bias = 0.2 * view_width / f32(view.viewport.z);

    let transformation = mat3x2<f32>(
        vec2<f32>(rotation.y, -rotation.x) * (ellipse_size_bias + semi_axes.x),
        vec2<f32>(rotation.x, rotation.y) * (ellipse_size_bias + semi_axes.y),
        translation,
    );

    let T = mat3x3(
        vec3<f32>(transformation.x, 0.0),
        vec3<f32>(transformation.y, 0.0),
        vec3<f32>(transformation.z, 1.0),
    );

    let ellipse_margin = 3.3;  // should be 2.0
    output.uv = quad_offset * ellipse_margin;
    output.position = vec4<f32>(
        (T * vec3<f32>(output.uv, 1.0)).xy / view.viewport.zw,
        0.0,
        1.0,
    );

    return output;
}

@fragment
fn fs_main(input: GaussianOutput) -> @location(0) vec4<f32> {
    let power = dot(input.uv, input.uv);
    let alpha = input.color.a * exp(-0.5 * power);

    if (alpha < 1.0 / 255.0) {
        discard;
    }

    return vec4<f32>(input.color.rgb * alpha, alpha);
}
