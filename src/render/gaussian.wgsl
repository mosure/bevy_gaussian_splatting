#import bevy_gaussian_splatting::bindings::{
    view,
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
#import bevy_gaussian_splatting::helpers::{
    get_rotation_matrix,
    get_scale_matrix,
}
#import bevy_gaussian_splatting::surfel::{
    compute_cov2d_surfel,
    get_bounding_box_cov2d,
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
#else

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

#endif


#ifdef BUFFER_TEXTURE
#ifdef PRECOMPUTE_COVARIANCE_3D
#import bevy_gaussian_splatting::texture::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_cov3d,
    location,
}
#else
#import bevy_gaussian_splatting::texture::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_rotation,
    get_scale,
    location,
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
    let S = get_scale_matrix(scale);

    let T = mat3x3<f32>(
        gaussian_uniforms.transform[0].xyz,
        gaussian_uniforms.transform[1].xyz,
        gaussian_uniforms.transform[2].xyz,
    );

    let R = get_rotation_matrix(rotation);

    let M = S * R;
    let Sigma = transpose(M) * M;
    let TS = T * Sigma * transpose(T);

    return array<f32, 6>(
        TS[0][0],
        TS[0][1],
        TS[0][2],
        TS[1][1],
        TS[1][2],
        TS[2][2],
    );
}

fn compute_cov2d_3dgs(
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

    var t = view.view_from_world * vec4<f32>(position, 1.0);

    let focal = vec2<f32>(
        view.clip_from_view.x.x * view.viewport.z,
        view.clip_from_view.y.y * view.viewport.w,
    );

    let s = 1.0 / (t.z * t.z);
    let J = mat3x3(
        focal.x / t.z, 0.0, -(focal.x * t.x) * s,
        0.0, -focal.y / t.z, (focal.y * t.y) * s,
        0.0, 0.0, 0.0,
    );

    let W = transpose(
        mat3x3<f32>(
            view.view_from_world.x.xyz,
            view.view_from_world.y.xyz,
            view.view_from_world.z.xyz,
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

fn inverted_infinity_norm(v: vec3<f32>) -> vec3<f32> {
    let min_value = min(v.x, min(v.y, v.z));
    let min_vec = vec3<f32>(min_value);

    return select(
        vec3<f32>(0.0),
        vec3<f32>(1.0),
        v == min_vec,
    );
}


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

    let transformed_position = (gaussian_uniforms.transform * position).xyz;
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

#ifdef RASTERIZE_DEPTH
    // TODO: unbiased depth rendering, see: https://zju3dv.github.io/pgsr/
    let first_position = vec4<f32>(get_position(get_entry(1u).value), 1.0);
    let last_position = vec4<f32>(get_position(get_entry(gaussian_uniforms.count - 1u).value), 1.0);

    let min_position = (gaussian_uniforms.transform * first_position).xyz;
    let max_position = (gaussian_uniforms.transform * last_position).xyz;

    let camera_position = view.world_position;

    let min_distance = length(min_position - camera_position);
    let max_distance = length(max_position - camera_position);

    let depth = length(transformed_position - camera_position);
    rgb = depth_to_rgb(
        depth,
        min_distance,
        max_distance,
    );
#else ifdef RASTERIZE_NORMAL
    // TODO: support surfel normal rendering
    let T = mat3x3<f32>(
        gaussian_uniforms.transform[0].xyz,
        gaussian_uniforms.transform[1].xyz,
        gaussian_uniforms.transform[2].xyz,
    );

    let R = get_rotation_matrix(get_rotation(splat_index));
    let scale = get_scale(splat_index);
    let scale_inf = inverted_infinity_norm(scale);
    let S = get_scale_matrix(scale_inf);

    let M = S * R;
    let Sigma = transpose(M) * M;

    let N = T * Sigma * transpose(T);
    let normal = vec3<f32>(
        N[0][0],
        N[0][1],
        N[1][1],
    );

    let t = normalize(normal);

    rgb = vec3<f32>(
        0.5 * (t.x + 1.0),
        0.5 * (t.y + 1.0),
        0.5 * (t.z + 1.0)
    );
#else
    rgb = get_color(splat_index, ray_direction);
#endif

    let opacity = get_opacity(splat_index);

#ifdef OPACITY_ADAPTIVE_RADIUS
    let cutoff = sqrt(max(9.0 + 2.0 * log(opacity), 0.000001));
#else
    let cutoff = 3.0;
#endif

    // TODO: verify color benefit for ray_direction computed at quad verticies instead of gaussian center (same as current complexity)
    output.color = vec4<f32>(
        rgb,
        opacity,
    );

#ifdef HIGHLIGHT_SELECTED
    if (get_visibility(splat_index) > 0.5) {
        output.color = vec4<f32>(0.3, 1.0, 0.1, 1.0);
    }
#endif

#ifdef GAUSSIAN_3D
    let cov2d = compute_cov2d_3dgs(
        transformed_position,
        splat_index,
    );
    let bb = get_bounding_box(
        cov2d,
        quad_offset,
        cutoff,
    );
#else ifdef GAUSSIAN_SURFEL
    let cov2d = compute_cov2d_surfel(
        transformed_position,
        splat_index,
        cutoff,
    );
    let bb = get_bounding_box_cov2d(
        cov2d,
        quad_offset,
        cutoff,
    );
#endif

#ifdef USE_AABB
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(
        cov2d.z * det_inv,
        -cov2d.y * det_inv,
        cov2d.x * det_inv
    );
    // TODO: this conic seems only valid in 3dgs
    output.conic = conic;
    output.major_minor = bb.zw;
#endif

    output.uv = quad_offset;
    output.position = vec4<f32>(
        projected_position.xy + bb.xy,
        projected_position.zw
    );

    return output;
}

@fragment
fn fs_main(input: GaussianVertexOutput) -> @location(0) vec4<f32> {
    // TODO: surfel accumulation

#ifdef USE_AABB
    let d = -input.major_minor;
    let conic = input.conic;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;

    if (power > 0.0) {
        discard;
    }
#endif

#ifdef USE_OBB
    let sigma = 1.0 / 3.0;
    let sigma_squared = 2.0 * sigma * sigma;
    let distance_squared = dot(input.uv, input.uv);

    let power = -distance_squared / sigma_squared;

    if (distance_squared > 3.0 * 3.0) {
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

    // TODO: round final_alpha to terminate depth test?

    return vec4<f32>(
        input.color.rgb * final_alpha,
        final_alpha,
    );
}
