#import bevy_gaussian_splatting::bindings::{view, gaussian_uniforms, Entry}
#import bevy_gaussian_splatting::projection::{projected_world_position, projected_support_in_frustum, projected_covariance, projected_color}
#import bevy_gaussian_splatting::planar::{get_opacity, get_visibility}
#import bevy_gaussian_splatting::helpers::{get_bounding_box_clip, gaussian_depth_sort_key}
#import bevy_gaussian_splatting::ordered_types::OrderedGaussian
#ifdef LOD_SPATIAL_MORPH
    #import bevy_gaussian_splatting::lod_spatial_projection::{spatial_entry, spatial_draw_word, spatial_sample, spatial_projection, spatial_support_in_frustum}
#endif

struct ProjectConfig {
    output_start: u32,
    capacity: u32,
    identity: u32,
    draw_mode: u32,
    adaptive_radius: u32,
    candidate: u32,
    unsorted_source: u32,
    traversed: u32,
}
@group(3) @binding(0) var<uniform> config: ProjectConfig;
#ifndef LOD_SPATIAL_MORPH
    @group(3) @binding(1) var<storage, read> entries: array<Entry>;
    @group(3) @binding(2) var<storage, read> indirect: array<u32>;
#endif
@group(3) @binding(3) var<storage, read_write> projected: array<OrderedGaussian>;
@group(3) @binding(4) var<storage, read_write> header: array<atomic<u32>>;

@compute @workgroup_size(256)
fn project(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let local = (group.y * groups.x + group.x) * 256u + lane;
    if local >= config.capacity { return; }
    let output = config.output_start + local;
    projected[output].cutoff_squared = 0.0;
    #ifdef LOD_SPATIAL_MORPH
        let spatial = (config.traversed & 2u) != 0u;
        let input_count = spatial_draw_word(config.capacity, spatial, 1u);
        var input_flags = 0u;
        if config.traversed != 0u { input_flags = spatial_draw_word(config.capacity, spatial, 4u); }
        if spatial && local == 0u {
            atomicAdd(&header[7], spatial_draw_word(config.capacity, true, 3u));
            atomicAdd(&header[14], spatial_draw_word(config.capacity, true, 6u));
            atomicOr(&header[15], spatial_draw_word(config.capacity, true, 5u));
            atomicAdd(&header[16], spatial_draw_word(config.capacity, true, 0u));
            atomicAdd(&header[17], spatial_draw_word(config.capacity, true, 7u));
        }
    #else
        let input_count = indirect[1];
        var input_flags = 0u;
        if config.traversed != 0u { input_flags = indirect[4]; }
    #endif
    // A missing covering root or an overflowing traversal invalidates the
    // complete shared image. The following gather pass suppresses its draw.
    if config.traversed != 0u && ((input_flags & 1u) != 0u || input_count > config.capacity) {
        if local == 0u { atomicOr(&header[13], 2u); }
        return;
    }
    let count = select(input_count, gaussian_uniforms.count, config.identity != 0u);
    if local >= count { return; }
    var index = local;
    if config.identity == 0u {
        #ifdef LOD_SPATIAL_MORPH
            index = spatial_entry(local).y;
        #else
            index = entries[local].value;
        #endif
        if config.traversed == 0u { index &= 0x0fffffffu; }
    }
    if index >= gaussian_uniforms.count {
        if config.traversed != 0u { atomicOr(&header[13], 2u); }
        return;
    }
    let child_visibility = get_visibility(index);
    var visibility = child_visibility;
    #ifdef LOD_SPATIAL_MORPH
        let sample = spatial_sample(local, config.capacity, spatial, index);
        let parent_visibility = get_visibility(sample.parent);
        if sample.enabled { visibility = max(parent_visibility, child_visibility); }
    #endif
    if config.draw_mode == 1u && visibility < 0.5 { return; }
    // Match the flat radix path's visibility sentinel. Discrete compaction
    // treats this channel as selection metadata and has no universal gate.
    if config.identity != 0u && config.unsorted_source == 0u && visibility <= 0.0 { return; }
    #ifdef LOD_SPATIAL_MORPH
        let projection = spatial_projection(index, sample);
        let position = projection.position;
    #else
        let position = projected_world_position(index);
    #endif
    let clip_h = view.unjittered_clip_from_world * vec4<f32>(position, 1.0);
    let clip = clip_h / (clip_h.w + 0.000000001);
    if !(clip.z >= 0.0 && clip.z <= 1.0) { return; }
    let authored_opacity = get_opacity(index);
    var cutoff = 3.0;
    if config.adaptive_radius != 0u && config.candidate == 0u {
        cutoff = sqrt(max(9.0 + 2.0 * log(max(authored_opacity, 0.000001)), 0.000001));
    }
    #ifdef LOD_SPATIAL_MORPH
        if !spatial_support_in_frustum(index, sample, position, cutoff) { return; }
        let covariance = projection.covariance;
    #else
        if !projected_support_in_frustum(index, position, cutoff) { return; }
        let covariance = projected_covariance(index, position);
    #endif
    let det = covariance.x * covariance.z - covariance.y * covariance.y;
    if !(det > 0.0) { return; }
    var opacity = clamp(authored_opacity * gaussian_uniforms.global_opacity * covariance.w, 0.0, 1.0);
    var color = projected_color(index, position);
    if config.draw_mode == 2u && child_visibility > 0.5 { color = vec3<f32>(0.3, 1.0, 0.1); }
    #ifdef LOD_SPATIAL_MORPH
        var parent_color = vec4<f32>(0.0);
        var morph = vec4<f32>(0.0);
        if sample.enabled {
            opacity = clamp(authored_opacity * gaussian_uniforms.global_opacity * projection.child_opacity_scale, 0.0, 1.0);
            color = projected_color(index, projection.child_position);
            parent_color = vec4<f32>(projected_color(sample.parent, projection.parent_position),
                clamp(get_opacity(sample.parent) * gaussian_uniforms.global_opacity * projection.parent_opacity_scale, 0.0, 1.0));
            morph = vec4<f32>(projection.parent_coefficient, projection.child_coefficient, sample.weight, 1.0);
            if config.draw_mode == 1u {
                morph.x *= select(0.0, 1.0, parent_visibility >= 0.5);
                morph.y *= select(0.0, 1.0, child_visibility >= 0.5);
            }
            if config.draw_mode == 2u {
                if parent_visibility > 0.5 { parent_color = vec4<f32>(0.3, 1.0, 0.1, parent_color.a); }
                if child_visibility > 0.5 { color = vec3<f32>(0.3, 1.0, 0.1); }
            }
        }
        if !(opacity > 0.0 || parent_color.a > 0.0) { return; }
    #else
        if !(opacity > 0.0) { return; }
    #endif
    let depth_key = gaussian_depth_sort_key(position);
    if depth_key == 0xffffffffu { return; }
    var conic_xy = vec2<f32>(covariance.z, -covariance.y) / det;
    #ifdef USE_AABB
        // The per-cloud AABB renderer evaluates density in its interpolated NDC-oriented
        // offset; framebuffer Y has the opposite sign.
        conic_xy.y = -conic_xy.y;
    #endif
    projected[output] = OrderedGaussian(
        (clip.xy * vec2<f32>(0.5, -0.5) + 0.5) * view.viewport.zw + view.viewport.xy,
        get_bounding_box_clip(covariance.xyz, vec2<f32>(1.0, 0.0), cutoff).xy,
        get_bounding_box_clip(covariance.xyz, vec2<f32>(0.0, 1.0), cutoff).xy,
        conic_xy,
        vec4<f32>(color, opacity), covariance.x / det, clip.z,
        depth_key, cutoff * cutoff
        #ifdef LOD_SPATIAL_MORPH
            , parent_color, morph
        #endif
    );
}
