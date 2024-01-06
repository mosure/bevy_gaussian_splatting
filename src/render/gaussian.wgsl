#import bevy_gaussian_splatting::bindings::{
    view,
    globals,
    gaussian_uniforms,
    sorting_pass_index,
    sorting,
    draw_indirect,
    input_entries,
    output_entries,
    Entry,
}
#import bevy_gaussian_splatting::depth::{
    depth_to_rgb,
}
#import bevy_gaussian_splatting::transform::{
    world_to_clip,
    in_frustum,
}

#ifdef PACKED
#ifdef PRECOMPUTE_COVARIANCE_3D
#import bevy_gaussian_splatting::packed::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_cov3d,
}
#else
#import bevy_gaussian_splatting::packed::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_rotation,
    get_scale,
}
#endif
#endif

#ifdef BUFFER_STORAGE
#ifdef PRECOMPUTE_COVARIANCE_3D
#import bevy_gaussian_splatting::planar::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_cov3d,
}
#else
#import bevy_gaussian_splatting::planar::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_rotation,
    get_scale,
}
#endif
#endif

#ifdef BUFFER_TEXTURE
#ifdef PRECOMPUTE_COVARIANCE_3D
#import bevy_gaussian_splatting::texture::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_cov3d,
}
#else
#import bevy_gaussian_splatting::texture::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_rotation,
    get_scale,
}
#endif
#endif


#ifdef BUFFER_STORAGE
@group(3) @binding(0) var<storage, read> sorted_entries: array<Entry>;

fn get_entry(index: u32) -> Entry {
    return sorted_entries[index];
}
#endif

#ifdef BUFFER_TEXTURE
@group(3) @binding(0) var sorted_entries: texture_2d<u32>;

fn get_entry(index: u32) -> Entry {
    let sample = textureLoad(
        sorted_entries,
        location(index),
        0,
    );

    return Entry(
        sample.r,
        sample.g,
    );
}
#endif

#ifdef WEBGL2
struct GaussianVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) conic: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) major_minor: vec2<f32>,
};
#else
struct GaussianVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(linear) uv: vec2<f32>,
    @location(3) @interpolate(linear) major_minor: vec2<f32>,
};
#endif


// https://github.com/cvlab-epfl/gaussian-splatting-web/blob/905b3c0fb8961e42c79ef97e64609e82383ca1c2/src/shaders.ts#L185
// TODO: precompute
fn compute_cov3d(scale: vec3<f32>, rotation: vec4<f32>) -> array<f32, 6> {
    let S = mat3x3<f32>(
        scale.x * gaussian_uniforms.global_scale, 0.0, 0.0,
        0.0, scale.y * gaussian_uniforms.global_scale, 0.0,
        0.0, 0.0, scale.z * gaussian_uniforms.global_scale,
    );

    let r = rotation.x;
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;

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

fn compute_cov2d(
    position: vec3<f32>,
    index: u32,
) -> vec3<f32> {
#ifdef PRECOMPUTE_COVARIANCE_3D
    let cov3d = get_cov3d(index);
#else
    let rotation = get_rotation(index);
    let scale = get_scale(index);

    let cov3d = compute_cov3d(scale, rotation);
#endif

    let Vrk = mat3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    );

    var t = view.inverse_view * vec4<f32>(position, 1.0);

    let focal = vec2<f32>(
        view.projection.x.x * view.viewport.z,
        view.projection.y.y * view.viewport.w,
    );

    let s = 1.0 / (t.z * t.z);
    let J = mat3x3(
        focal.x / t.z, 0.0, -(focal.x * t.x) * s,
        0.0, -focal.y / t.z, (focal.y * t.y) * s,
        0.0, 0.0, 0.0,
    );

    let W = transpose(
        mat3x3<f32>(
            view.inverse_view.x.xyz,
            view.inverse_view.y.xyz,
            view.inverse_view.z.xyz,
        )
    );

    let T = W * J;

    var cov = transpose(T) * transpose(Vrk) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}

fn get_bounding_box(
    cov2d: vec3<f32>,
    direction: vec2<f32>,
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
    let radius_px = 3.5 * max(x_axis_length, y_axis_length);
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

    let bounds = 3.5 * vec2<f32>(
        major_radius,
        minor_radius,
    );

    let eigvec1 = normalize(vec2<f32>(
        -cov2d.y,
        lambda1 - cov2d.x,
    ));
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


// @compute @workgroup_size(#{RADIX_BASE}, #{RADIX_DIGIT_PLACES})
// fn gaussian_compute(
//     @builtin(local_invocation_id) gl_LocalInvocationID: vec3<u32>,
//     @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
// ) {
//     // TODO: compute cov2d, color (any non-quad gaussian property)
// }


@vertex
fn vs_points(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> GaussianVertexOutput {
    var output: GaussianVertexOutput;

    let entry = get_entry(instance_index);
    let splat_index = entry.value;

    var discard_quad = false;

    discard_quad |= entry.key == 0xFFFFFFFFu; // || splat_index == 0u;

    let position = vec4<f32>(get_position(splat_index), 1.0);

    let transformed_position = (gaussian_uniforms.global_transform * position).xyz;
    let projected_position = world_to_clip(transformed_position);

    discard_quad |= !in_frustum(projected_position.xyz);

#ifdef DRAW_SELECTED
    discard_quad |= get_visibility(splat_index) < 0.5;
#endif

    if (discard_quad) {
        output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
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

    let ray_direction = normalize(transformed_position - view.world_position);

    var rgb = vec3<f32>(0.0);

#ifdef VISUALIZE_DEPTH
    let first_position = vec4<f32>(get_position(get_entry(1u).value), 1.0);
    let last_position = vec4<f32>(get_position(get_entry(gaussian_uniforms.count - 1u).value), 1.0);

    let min_position = (gaussian_uniforms.global_transform * first_position).xyz;
    let max_position = (gaussian_uniforms.global_transform * last_position).xyz;

    let camera_position = view.world_position;

    let min_distance = length(min_position - camera_position);
    let max_distance = length(max_position - camera_position);

    let depth = length(transformed_position - camera_position);
    rgb = depth_to_rgb(
        depth,
        min_distance,
        max_distance,
    );
#else
    rgb = get_color(splat_index, ray_direction);
#endif

    // TODO: precompute color, cov2d for every gaussian. cov2d only needs a single evaluation, while color needs to be evaluated every frame in SH degree > 0 mode
    output.color = vec4<f32>(
        rgb,
        get_opacity(splat_index),
    );

#ifdef HIGHLIGHT_SELECTED
    if (get_visibility(splat_index) > 0.5) {
        output.color = vec4<f32>(0.3, 1.0, 0.1, 1.0);
    }
#endif

    let cov2d = compute_cov2d(transformed_position, splat_index);

#ifdef USE_AABB
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(
        cov2d.z * det_inv,
        -cov2d.y * det_inv,
        cov2d.x * det_inv
    );
    output.conic = conic;
#endif

    let bb = get_bounding_box(
        cov2d,
        quad_offset,
    );

    output.uv = quad_offset;
    output.major_minor = bb.zw;
    output.position = vec4<f32>(
        projected_position.xy + bb.xy,
        projected_position.zw
    );

    return output;
}

@fragment
fn fs_main(input: GaussianVertexOutput) -> @location(0) vec4<f32> {
#ifdef USE_AABB
    let d = -input.major_minor;
    let conic = input.conic;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;

    if (power > 0.0) {
        discard;
    }
#endif

#ifdef USE_OBB
    let sigma = 1.0 / 3.5;
    let sigma_squared = 2.0 * sigma * sigma;
    let distance_squared = dot(input.uv, input.uv);

    let power = -distance_squared / sigma_squared;

    if (distance_squared > 3.5 * 3.5) {
        discard;
    }
#endif

#ifdef VISUALIZE_BOUNDING_BOX
    let uv = (input.uv + 1.0) / 2.0;
    let edge_width = 0.08;
    if (
        (uv.x < edge_width || uv.x > 1.0 - edge_width) ||
        (uv.y < edge_width || uv.y > 1.0 - edge_width)
    ) {
        return vec4<f32>(0.3, 1.0, 0.1, 1.0);
    }
#endif

    let alpha = exp(power);
    let final_alpha = alpha * input.color.a;
    return vec4<f32>(
        input.color.rgb * final_alpha,
        final_alpha,
    );
}
