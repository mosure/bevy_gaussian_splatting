#import bevy_gaussian_splatting::bindings::{
    view,
    gaussian_uniforms,
    Entry,
}
#import bevy_gaussian_splatting::classification::class_to_rgb
#import bevy_gaussian_splatting::depth::depth_to_rgb
#import bevy_gaussian_splatting::optical_flow::{
    calculate_motion_vector,
    optical_flow_to_rgb,
}
#import bevy_gaussian_splatting::helpers::{
    gaussian_mip_support_radius_world,
    get_rotation_matrix,
    get_scale_matrix,
}
#import bevy_gaussian_splatting::transform::{
    world_to_clip,
    in_frustum,
}

#ifdef LOD_MORPH
    #import bevy_gaussian_splatting::lod_morph::{
        lod_external_active_set_opacity_coefficient,
        lod_morph_fragment_color,
        lod_morph_log_scale,
        lod_morph_position,
        lod_morph_rotation,
        lod_morph_sample,
        lod_morph_support_cutoff,
        lod_morph_support_max_scale,
        lod_morph_visibility,
    }
#endif

#ifdef LOD_DEBUG
    #import bevy_gaussian_splatting::lod_debug::{
        apply_lod_debug_annotation,
        apply_lod_debug_morph_annotation,
        lod_debug_morph_requires_authored_color,
        lod_debug_requires_authored_color,
    }
#endif

#ifdef GAUSSIAN_2D
    #import bevy_gaussian_splatting::gaussian_2d::{
        compute_cov2d_surfel,
        get_bounding_box_cov2d,
        surfel_fragment_power,
    }
#else ifdef GAUSSIAN_3D
    #import bevy_gaussian_splatting::gaussian_3d::{
        compute_cov2d_3dgs,
    }
    #import bevy_gaussian_splatting::helpers::{
        get_bounding_box_clip,
    }
#else ifdef GAUSSIAN_4D
    #import bevy_gaussian_splatting::gaussian_4d::{
        conditional_cov3d,
    }
    #import bevy_gaussian_splatting::helpers::{
        cov2d,
        get_bounding_box_clip,
    }
#endif

#ifdef PACKED
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::packed::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_cov3d,
            get_rotation,
            get_scale,
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
#else ifdef BUFFER_STORAGE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::planar::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_cov3d,
            get_rotation,
            get_scale,
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
#else ifdef BUFFER_TEXTURE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::texture::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_cov3d,
            get_rotation,
            get_scale,
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
#else ifdef BUFFER_TEXTURE
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

// LoD compaction carries a two-bit mode-qualified presentation class and two
// bits of per-view Residency provenance in the sorted entry itself. In morph
// mode class 1 is the parent-map fast path; in external-active-set mode the
// classes are Shared/FirstOnly/SecondOnly. Ordinary/flat entries leave them
// zero. A position record occupies at least 16 bytes, so WebGPU's u32-sized
// storage-buffer limit keeps every physical index below this 28-bit mask.
const LOD_ENTRY_SOURCE_INDEX_MASK: u32 = 0x0fffffffu;
const LOD_ENTRY_PRESENTATION_CLASS_SHIFT: u32 = 28u;
const LOD_ENTRY_PRESENTATION_CLASS_MASK: u32 = 3u << LOD_ENTRY_PRESENTATION_CLASS_SHIFT;
const LOD_ENTRY_RESIDENCY_SHIFT: u32 = 30u;
const LOD_EXTERNAL_ACTIVE_SET_FIRST_ONLY: u32 = 1u;

fn source_index_from_entry(entry: Entry) -> u32 {
    return entry.value & LOD_ENTRY_SOURCE_INDEX_MASK;
}

fn lod_residency_from_entry(entry: Entry) -> u32 {
    return entry.value >> LOD_ENTRY_RESIDENCY_SHIFT;
}

fn lod_presentation_class_from_entry(entry: Entry) -> u32 {
    return (entry.value & LOD_ENTRY_PRESENTATION_CLASS_MASK)
        >> LOD_ENTRY_PRESENTATION_CLASS_SHIFT;
}

fn lod_morph_from_entry(entry: Entry) -> bool {
    return lod_presentation_class_from_entry(entry)
        == LOD_EXTERNAL_ACTIVE_SET_FIRST_ONLY;
}

const GAUSSIAN_AUTHORED_SUPPORT_SIGMA: f32 = 3.0;
const GAUSSIAN_OPACITY_RADIUS_LOG_FLOOR: f32 = 0.000001;

fn gaussian_support_cutoff(opacity: f32) -> f32 {
#ifdef LOD_CANDIDATE
    // Portable pages accept every finite authored opacity. Candidate
    // compaction and the offline LoD support oracle both use the authored
    // three-sigma footprint, so raster must not grow a separate opacity > 1
    // tail (or shrink a low-opacity representative) after compaction.
    return GAUSSIAN_AUTHORED_SUPPORT_SIGMA;
#else
    #ifdef OPACITY_ADAPTIVE_RADIUS
        return sqrt(max(
            GAUSSIAN_AUTHORED_SUPPORT_SIGMA * GAUSSIAN_AUTHORED_SUPPORT_SIGMA
                + 2.0 * log(max(opacity, GAUSSIAN_OPACITY_RADIUS_LOG_FLOOR)),
            GAUSSIAN_OPACITY_RADIUS_LOG_FLOOR,
        ));
    #else
        return GAUSSIAN_AUTHORED_SUPPORT_SIGMA;
    #endif
#endif
}

#ifdef WEBGL2
    struct GaussianVertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
        @location(1) uv: vec2<f32>,
    #ifdef GAUSSIAN_2D
        @location(2) local_to_pixel_u: vec3<f32>,
        @location(3) local_to_pixel_v: vec3<f32>,
        @location(4) local_to_pixel_w: vec3<f32>,
        @location(5) mean_2d: vec2<f32>,
        @location(6) radius: vec2<f32>,
        @location(8) cutoff_squared: f32,
    #else #ifdef GAUSSIAN_3D
        @location(2) conic: vec3<f32>,
        @location(3) major_minor: vec2<f32>,
        @location(4) cutoff_squared: f32,
    #else #ifdef GAUSSIAN_4D
        @location(2) conic: vec3<f32>,
        @location(3) major_minor: vec2<f32>,
        @location(4) cutoff_squared: f32,
    #endif
    #ifdef LOD_MORPH
        @location(7) lod_morph_alpha: vec4<f32>,
        @location(9) lod_morph_parent_color: vec4<f32>,
    #endif
    };
#else
    struct GaussianVertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) @interpolate(flat) color: vec4<f32>,
        @location(1) @interpolate(linear) uv: vec2<f32>,
    #ifdef GAUSSIAN_2D
        @location(2) @interpolate(flat) local_to_pixel_u: vec3<f32>,
        @location(3) @interpolate(flat) local_to_pixel_v: vec3<f32>,
        @location(4) @interpolate(flat) local_to_pixel_w: vec3<f32>,
        @location(5) @interpolate(flat) mean_2d: vec2<f32>,
        @location(6) @interpolate(flat) radius: vec2<f32>,
        @location(8) @interpolate(flat) cutoff_squared: f32,
    #else ifdef GAUSSIAN_3D
        @location(2) @interpolate(flat) conic: vec3<f32>,
        @location(3) @interpolate(linear) major_minor: vec2<f32>,
        @location(4) @interpolate(flat) cutoff_squared: f32,
    #else ifdef GAUSSIAN_4D
        @location(2) @interpolate(flat) conic: vec3<f32>,
        @location(3) @interpolate(linear) major_minor: vec2<f32>,
        @location(4) @interpolate(flat) cutoff_squared: f32,
    #endif
    #ifdef LOD_MORPH
        // {parent peak alpha, child peak alpha, parent optical-depth
        // coefficient, child optical-depth coefficient}. The coefficients
        // include per-edge weight, parent run splitting, and filtered
        // projected-area conservation.
        @location(7) @interpolate(flat) lod_morph_alpha: vec4<f32>,
        // Parent endpoint radiance in linear light plus the exact blend weight
        // in `.w`. Child endpoint radiance occupies `color.rgb`; the fragment
        // combines both by optical depth and uses `.w` only to identify exact
        // endpoint fast paths.
        @location(9) @interpolate(flat) lod_morph_parent_color: vec4<f32>,
    #endif
    };
#endif

fn world_to_local_direction(ray_direction_world: vec3<f32>, transform: mat4x4<f32>) -> vec3<f32> {
    let basis = mat3x3<f32>(
        transform[0].xyz,
        transform[1].xyz,
        transform[2].xyz,
    );
    let basis_x = normalize(basis[0]);
    let basis_y = normalize(basis[1]);
    let basis_z = normalize(basis[2]);

    let local = vec3<f32>(
        dot(basis_x, ray_direction_world),
        dot(basis_y, ray_direction_world),
        dot(basis_z, ray_direction_world),
    );

    return normalize(local);
}

#ifdef LOD_MORPH
fn lod_morph_visibility_contributes(visibility: f32) -> bool {
    #ifdef DRAW_SELECTED
        return visibility >= 0.5;
    #else
        // Visibility stores selection/classification metadata. DrawMode::All
        // and HighlightSelected render every endpoint; HighlightSelected only
        // changes the selected endpoint's color below.
        return true;
    #endif
}

fn gaussian_render_color_at(
    index: u32,
    transformed_position: vec3<f32>,
) -> vec3<f32> {
    let ray_direction_world = normalize(transformed_position - view.world_position);
    let ray_direction_local = world_to_local_direction(
        ray_direction_world,
        gaussian_uniforms.transform,
    );
    // `get_color` evaluates this endpoint's SH and performs its own declared
    // sRGB-to-linear conversion. Endpoint colors must reach the optical-depth
    // mixer independently; interpolating encoded color or SH first is wrong.
    return get_color(index, ray_direction_local);
}
#endif

// Conservative world-space sphere for the visible Gaussian support. This is
// intentionally shared in policy with LoD compaction so the raster stage never
// reintroduces a center-only false negative after compaction retained an
// edge-overlapping splat.
fn gaussian_support_radius_world(
    index: u32,
    cutoff: f32,
    morph_parent_index: u32,
    morph_blend_t: f32,
    morph_active: bool,
    position_world: vec3<f32>,
) -> f32 {
    let child_scale = get_scale(index);
    var max_scale = max(
        abs(child_scale.x),
        max(abs(child_scale.y), abs(child_scale.z)),
    );
    #ifdef LOD_MORPH
        if morph_active {
            max_scale = lod_morph_support_max_scale(
                get_scale(morph_parent_index),
                child_scale,
                morph_blend_t,
            );
        }
    #endif
    let local_radius = cutoff * abs(gaussian_uniforms.global_scale) * max_scale;
    let authored_radius_world = local_radius * gaussian_uniforms.transform_scale_bound;
    #ifdef GAUSSIAN_3D
        let mip_radius_world = gaussian_mip_support_radius_world(
            position_world,
            cutoff,
        );
        if !(mip_radius_world >= 0.0) {
            return mip_radius_world;
        }
        return authored_radius_world + mip_radius_world;
    #else
        return authored_radius_world;
    #endif
}

fn gaussian_support_sphere_in_frustum(center: vec3<f32>, radius: f32) -> bool {
    if !(radius >= 0.0) {
        return true;
    }
    for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
        let plane = view.frustum[plane_index];
        let signed_distance = dot(plane.xyz, center) + plane.w;
        // Bevy ViewUniform frustum half-spaces have unit normals.
        if signed_distance < -radius {
            return false;
        }
    }
    return true;
}

@vertex
fn vs_points(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> GaussianVertexOutput {
    var output: GaussianVertexOutput;
    #ifdef LOD_MORPH
        output.lod_morph_alpha = vec4<f32>(0.0);
        output.lod_morph_parent_color = vec4<f32>(0.0);
    #endif

    let entry = get_entry(instance_index);
    let splat_index = source_index_from_entry(entry);

    var morph_parent_index = splat_index;
    var morph_run_length = 1u;
    var morph_blend_t = 1.0;
    var morph_active = false;
    #ifdef LOD_MORPH
        // The packed entry bit is the common-case fast path: millions of
        // unchanged records do not touch or search the morph table.
        if lod_morph_from_entry(entry) {
            let morph = lod_morph_sample(splat_index, gaussian_uniforms.count);
            morph_parent_index = morph.parent_physical_index;
            morph_run_length = morph.run_length;
            morph_blend_t = morph.blend_t;
            morph_active = morph.enabled;
        }
    #endif

    let child_visibility = get_visibility(splat_index);
    var parent_visibility = child_visibility;
    var displayed_visibility = child_visibility;
    #ifdef LOD_MORPH
        if morph_active {
            parent_visibility = get_visibility(morph_parent_index);
            displayed_visibility = lod_morph_visibility(
                parent_visibility,
                child_visibility,
                morph_blend_t,
            );
        }
    #endif

    var discard_quad = false;

    discard_quad |= entry.key == 0xFFFFFFFFu; // || splat_index == 0u;

    let child_position_local = get_position(splat_index);
    var parent_position_local = child_position_local;
    var position_local = child_position_local;
    #ifdef LOD_MORPH
        if morph_active {
            parent_position_local = get_position(morph_parent_index);
            position_local = lod_morph_position(
                parent_position_local,
                position_local,
                morph_blend_t,
            );
        }
    #endif
    let position = vec4<f32>(position_local, 1.0);

    var transformed_position = (gaussian_uniforms.transform * position).xyz;
    var parent_transformed_position = transformed_position;
    var child_transformed_position = transformed_position;
    #ifdef LOD_MORPH
        if morph_active {
            parent_transformed_position = (
                gaussian_uniforms.transform * vec4<f32>(parent_position_local, 1.0)
            ).xyz;
            child_transformed_position = (
                gaussian_uniforms.transform * vec4<f32>(child_position_local, 1.0)
            ).xyz;
        }
    #endif
    var previous_transformed_position = transformed_position;

    var opacity = get_opacity(splat_index);

    var cutoff = gaussian_support_cutoff(opacity);
    #ifdef LOD_MORPH
        if morph_active {
            let parent_cutoff = gaussian_support_cutoff(
                get_opacity(morph_parent_index),
            );
            cutoff = lod_morph_support_cutoff(
                parent_cutoff,
                cutoff,
                morph_blend_t,
            );
        }
    #endif
    output.cutoff_squared = cutoff * cutoff;

#ifdef DRAW_SELECTED
    discard_quad |= displayed_visibility < 0.5;
#endif

#ifdef GAUSSIAN_4D
#else
    let projected_position = world_to_clip(transformed_position);
    let support_radius = gaussian_support_radius_world(
        splat_index,
        cutoff,
        morph_parent_index,
        morph_blend_t,
        morph_active,
        transformed_position,
    );
    discard_quad |= !gaussian_support_sphere_in_frustum(
        transformed_position,
        support_radius,
    );
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

#ifdef GAUSSIAN_2D
    let surfel = compute_cov2d_surfel(
        transformed_position,
        splat_index,
        cutoff,
    );

    output.local_to_pixel_u = surfel.local_to_pixel[0];
    output.local_to_pixel_v = surfel.local_to_pixel[1];
    output.local_to_pixel_w = surfel.local_to_pixel[2];
    output.mean_2d = surfel.mean_2d;

    let bb = get_bounding_box_cov2d(
        surfel.extent,
        quad_offset,
        cutoff,
    );
    output.radius = bb.zw;
#else
    #ifdef GAUSSIAN_3D
        let gaussian_mip = compute_cov2d_3dgs(
            transformed_position,
            parent_transformed_position,
            child_transformed_position,
            splat_index,
            morph_parent_index,
            morph_blend_t,
            morph_active,
        );
        let gaussian_cov2d = gaussian_mip.filtered.xyz;
        #ifdef LOD_MORPH
            if morph_active {
                let run_count = f32(max(morph_run_length, 1u));
                var parent_coefficient = 0.0;
                var child_coefficient = 0.0;
                if morph_blend_t <= 0.0 {
                    // Exact duplicated-parent endpoint: K coincident proxies
                    // each carry 1/K of the parent's optical depth.
                    parent_coefficient = 1.0 / run_count;
                } else if morph_blend_t >= 1.0 {
                    // Exact authored child endpoint.
                    child_coefficient = 1.0;
                } else {
                    parent_coefficient = (1.0 - morph_blend_t)
                        * gaussian_mip.parent_projected_area_ratio
                        / run_count;
                    child_coefficient = morph_blend_t
                        * gaussian_mip.child_projected_area_ratio;
                }
                // Compaction retains the endpoint visibility union in the open
                // interval. Remove only the invisible endpoint's optical-depth
                // term so visibility changes fade with the authored blend and
                // both exact cuts retain their ordinary-render semantics.
                parent_coefficient = parent_coefficient * select(
                    0.0,
                    1.0,
                    lod_morph_visibility_contributes(parent_visibility),
                );
                child_coefficient = child_coefficient * select(
                    0.0,
                    1.0,
                    lod_morph_visibility_contributes(child_visibility),
                );
                output.lod_morph_alpha = vec4<f32>(
                    clamp(
                        gaussian_uniforms.global_opacity
                            * get_opacity(morph_parent_index)
                            * gaussian_mip.parent_opacity_scale,
                        0.0,
                        1.0,
                    ),
                    clamp(
                        gaussian_uniforms.global_opacity
                            * opacity
                            * gaussian_mip.child_opacity_scale,
                        0.0,
                        1.0,
                    ),
                    parent_coefficient,
                    child_coefficient,
                );
            }
        #endif
        opacity = opacity * gaussian_mip.filtered.w;
    #else ifdef GAUSSIAN_4D
        let gaussian_4d = conditional_cov3d(
            transformed_position,
            splat_index,
            gaussian_uniforms.time,
        );

        if !gaussian_4d.mask {
            output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            return output;
        }

        let position_t = vec4<f32>(position.xyz + gaussian_4d.delta_mean, 1.0);
        transformed_position = (gaussian_uniforms.transform * position_t).xyz;
        // TODO: set previous_transformed_position based on temporal position delta
        let projected_position = world_to_clip(transformed_position);

        if !in_frustum(projected_position.xyz) {
            output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            return output;
        }

        opacity = opacity * gaussian_4d.opacity_modifier;

        let gaussian_mip = cov2d(
            transformed_position,
            gaussian_4d.cov3d,
        );
        let gaussian_cov2d = gaussian_mip.xyz;
        opacity = opacity * gaussian_mip.w;
    #endif

    let bb = get_bounding_box_clip(
        gaussian_cov2d,
        quad_offset,
        cutoff,
    );

    #ifdef USE_AABB
        let det = gaussian_cov2d.x * gaussian_cov2d.z - gaussian_cov2d.y * gaussian_cov2d.y;
        let det_inv = 1.0 / det;
        let conic = vec3<f32>(
            gaussian_cov2d.z * det_inv,
            -gaussian_cov2d.y * det_inv,
            gaussian_cov2d.x * det_inv
        );
        output.conic = conic;
        output.major_minor = bb.zw;
    #endif
#endif

    var rgb = vec3<f32>(0.0);
    #ifdef LOD_MORPH
        var lod_parent_rgb = vec3<f32>(0.0);
        var lod_child_rgb = vec3<f32>(0.0);
        var lod_endpoint_colors_valid = false;
    #endif

// TODO: RASTERIZE_ACCELERATION
#ifdef RASTERIZE_CLASSIFICATION
    let ray_direction_world = normalize(transformed_position - view.world_position);
    let ray_direction_local = world_to_local_direction(ray_direction_world, gaussian_uniforms.transform);

    #ifdef LOD_DEBUG
        var debug_requires_authored_color = lod_debug_requires_authored_color(splat_index);
        #ifdef LOD_MORPH
            if morph_active {
                debug_requires_authored_color = lod_debug_morph_requires_authored_color(
                    splat_index,
                    morph_parent_index,
                );
            }
        #endif
        if debug_requires_authored_color {
    #endif
    #ifdef GAUSSIAN_3D_STRUCTURE
        #ifdef LOD_MORPH
            if morph_active {
                lod_parent_rgb = gaussian_render_color_at(
                    morph_parent_index,
                    parent_transformed_position,
                );
                lod_child_rgb = gaussian_render_color_at(
                    splat_index,
                    child_transformed_position,
                );
                lod_endpoint_colors_valid = true;
            } else {
                rgb = get_color(splat_index, ray_direction_local);
            }
        #else
            rgb = get_color(splat_index, ray_direction_local);
        #endif
    #else ifdef GAUSSIAN_4D
        rgb = get_color(splat_index, gaussian_4d.dir_t, ray_direction_local);
    #endif
    #ifdef LOD_DEBUG
        }
    #endif

    #ifdef LOD_MORPH
        if morph_active && lod_endpoint_colors_valid {
            lod_parent_rgb = class_to_rgb(parent_visibility, lod_parent_rgb);
            lod_child_rgb = class_to_rgb(child_visibility, lod_child_rgb);
            rgb = lod_child_rgb;
        } else {
            rgb = class_to_rgb(child_visibility, rgb);
        }
    #else
        rgb = class_to_rgb(child_visibility, rgb);
    #endif
#else ifdef RASTERIZE_DEPTH
    // TODO: unbiased depth rendering, see: https://zju3dv.github.io/pgsr/
    let first_position = vec4<f32>(get_position(source_index_from_entry(get_entry(1u))), 1.0);
    let last_position = vec4<f32>(get_position(source_index_from_entry(get_entry(gaussian_uniforms.count - 1u))), 1.0);

    let min_position = (gaussian_uniforms.transform * last_position).xyz;
    let max_position = (gaussian_uniforms.transform * first_position).xyz;

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
    // TODO: support rotation decomposition for 4d gaussians
    var raster_rotation = get_rotation(splat_index);
    var raster_scale = get_scale(splat_index);
    #ifdef LOD_MORPH
        if morph_active {
            raster_rotation = lod_morph_rotation(
                get_rotation(morph_parent_index),
                raster_rotation,
                morph_blend_t,
            );
            raster_scale = lod_morph_log_scale(
                get_scale(morph_parent_index),
                raster_scale,
                morph_blend_t,
            );
        }
    #endif
    let R = get_rotation_matrix(raster_rotation);
    let S = get_scale_matrix(raster_scale);
    let T = mat3x3<f32>(
        gaussian_uniforms.transform[0].xyz,
        gaussian_uniforms.transform[1].xyz,
        gaussian_uniforms.transform[2].xyz,
    );
    let L = T * S * R;

    let local_normal = vec4<f32>(L[2], 0.0);
    let world_normal = view.view_from_world * local_normal;

    let t = normalize(world_normal);

    rgb = vec3<f32>(
        0.5 * (t.x + 1.0),
        0.5 * (t.y + 1.0),
        0.5 * (t.z + 1.0)
    );
#else ifdef RASTERIZE_OPTICAL_FLOW
    let motion_vector = calculate_motion_vector(
        transformed_position,
        previous_transformed_position,
    );

    rgb = optical_flow_to_rgb(motion_vector);
#else ifdef RASTERIZE_POSITION
    rgb = (transformed_position - gaussian_uniforms.min.xyz) / (gaussian_uniforms.max.xyz - gaussian_uniforms.min.xyz);
#else ifdef RASTERIZE_VELOCITY
    let time_delta = 1e-3;
    let future_gaussian_4d = conditional_cov3d(
        transformed_position,
        splat_index,
        gaussian_uniforms.time + time_delta,
    );
    let position_delta = future_gaussian_4d.delta_mean - gaussian_4d.delta_mean;
    let velocity = position_delta / time_delta;
    let velocity_magnitude = length(velocity);
    let velocity_normalized = normalize(velocity);

    // TODO: magnitude normalization
    let min_magnitude = 1.0;
    let max_magnitude = 2.0;

    let scaled_mag = clamp(
        (velocity_magnitude - min_magnitude) / (max_magnitude - min_magnitude),
        0.0,
        1.0
    );

    if scaled_mag < 1e-2 {
        opacity = 0.0;
    }

    let base_color = 0.5 * (velocity_normalized + vec3<f32>(1.0, 1.0, 1.0));
    rgb = base_color * scaled_mag;
#else ifdef RASTERIZE_COLOR
    // TODO: verify color benefit for ray_direction computed at quad verticies instead of gaussian center (same as current complexity)
    let ray_direction_world = normalize(transformed_position - view.world_position);
    let ray_direction_local = world_to_local_direction(ray_direction_world, gaussian_uniforms.transform);

    #ifdef LOD_DEBUG
        var debug_requires_authored_color = lod_debug_requires_authored_color(splat_index);
        #ifdef LOD_MORPH
            if morph_active {
                debug_requires_authored_color = lod_debug_morph_requires_authored_color(
                    splat_index,
                    morph_parent_index,
                );
            }
        #endif
        if debug_requires_authored_color {
    #endif
    #ifdef GAUSSIAN_3D_STRUCTURE
        #ifdef LOD_MORPH
            if morph_active {
                lod_parent_rgb = gaussian_render_color_at(
                    morph_parent_index,
                    parent_transformed_position,
                );
                lod_child_rgb = gaussian_render_color_at(
                    splat_index,
                    child_transformed_position,
                );
                lod_endpoint_colors_valid = true;
            } else {
                rgb = get_color(splat_index, ray_direction_local);
            }
        #else
            rgb = get_color(splat_index, ray_direction_local);
        #endif
    #else ifdef GAUSSIAN_4D
        rgb = get_color(splat_index, gaussian_4d.dir_t, ray_direction_local);
    #endif
    #ifdef LOD_DEBUG
        }
    #endif
#endif

#ifdef LOD_MORPH
    if morph_active {
        // Diagnostic raster modes have one geometry-derived color. Giving both
        // optical-depth endpoints that same value preserves the diagnostic;
        // authored Color/Classification paths supplied distinct endpoint
        // radiance above.
        if !lod_endpoint_colors_valid {
            lod_parent_rgb = rgb;
            lod_child_rgb = rgb;
        }
        rgb = lod_child_rgb;
    }
#endif

#ifdef LOD_DEBUG
    #ifdef LOD_MORPH
        if morph_active {
            // Apply node-authored diagnostics to each radiance endpoint before
            // the fragment's tau mixture. Calling the established helper at its
            // exact endpoints also preserves Residency's child-entry policy.
            lod_parent_rgb = apply_lod_debug_morph_annotation(
                splat_index,
                morph_parent_index,
                0.0,
                lod_residency_from_entry(entry),
                lod_parent_rgb,
            );
            lod_child_rgb = apply_lod_debug_morph_annotation(
                splat_index,
                morph_parent_index,
                1.0,
                lod_residency_from_entry(entry),
                lod_child_rgb,
            );
            rgb = lod_child_rgb;
        } else {
            rgb = apply_lod_debug_annotation(
                splat_index,
                lod_residency_from_entry(entry),
                rgb,
            );
        }
    #else
        rgb = apply_lod_debug_annotation(
            splat_index,
            lod_residency_from_entry(entry),
            rgb,
        );
    #endif
#endif

    var output_opacity = clamp(opacity * gaussian_uniforms.global_opacity, 0.0, 1.0);
    #ifdef LOD_MORPH
        // External active-set interpolation scales the final authored/global/
        // Mip-corrected peak opacity only. It never changes geometry, support,
        // color, visibility, or the ABI-16 optical-depth path.
        output_opacity = output_opacity * lod_external_active_set_opacity_coefficient(
            lod_presentation_class_from_entry(entry),
        );
    #endif
    output.color = vec4<f32>(rgb, output_opacity);
    #ifdef LOD_MORPH
        if morph_active {
            output.lod_morph_parent_color = vec4<f32>(lod_parent_rgb, morph_blend_t);
        }
    #endif

#ifdef HIGHLIGHT_SELECTED
    #ifdef LOD_MORPH
        if morph_active {
            if parent_visibility > 0.5 {
                output.lod_morph_parent_color = vec4<f32>(
                    vec3<f32>(0.3, 1.0, 0.1),
                    output.lod_morph_parent_color.w,
                );
            }
            if child_visibility > 0.5 {
                output.color = vec4<f32>(0.3, 1.0, 0.1, output_opacity);
            }
        } else if child_visibility > 0.5 {
            output.color = vec4<f32>(0.3, 1.0, 0.1, output_opacity);
        }
    #else
        if child_visibility > 0.5 {
            output.color = vec4<f32>(0.3, 1.0, 0.1, output_opacity);
        }
    #endif
#endif

    output.uv = quad_offset;
    output.position = vec4<f32>(
        projected_position.xy + bb.xy,
        projected_position.zw,
    );

    return output;
}

@fragment
fn fs_main(input: GaussianVertexOutput) -> @location(0) vec4<f32> {
#ifdef USE_AABB
#ifdef GAUSSIAN_2D
    let radius = input.radius;
    let mean_2d = input.mean_2d;
    let aspect = vec2<f32>(
        1.0,
        view.viewport.z / view.viewport.w,
    );
    let pixel_coord = input.uv * radius * aspect + mean_2d;

    let power = surfel_fragment_power(
        mat3x3<f32>(
            input.local_to_pixel_u,
            input.local_to_pixel_v,
            input.local_to_pixel_w,
        ),
        pixel_coord,
        mean_2d,
    );
#else ifdef GAUSSIAN_3D
    let d = -input.major_minor;
    let conic = input.conic;
    let power = -0.5 * (
        conic.x * d.x * d.x
            + 2.0 * conic.y * d.x * d.y
            + conic.z * d.y * d.y
    );
#else ifdef GAUSSIAN_4D
    let d = -input.major_minor;
    let conic = input.conic;
    let power = -0.5 * (
        conic.x * d.x * d.x
            + 2.0 * conic.y * d.x * d.y
            + conic.z * d.y * d.y
    );
#endif

    if (power > 0.0) {
        discard;
    }
#endif

#ifdef USE_OBB
    let distance_squared = dot(input.uv, input.uv);
    let power = -0.5 * input.cutoff_squared * distance_squared;

    // The OBB remains a conservative rectangle. Clipping its [-1, 1] quad to
    // a unit circle would drop valid anisotropic support at its corners.
#endif

#ifdef VISUALIZE_BOUNDING_BOX
    let uv = input.uv * 0.5 + 0.5;
    let edge_width = 0.08;
    if (
        (uv.x < edge_width || uv.x > 1.0 - edge_width) ||
        (uv.y < edge_width || uv.y > 1.0 - edge_width)
    ) {
        return vec4<f32>(0.3, 1.0, 0.1, 1.0);
    }
#endif

    let gaussian_weight = exp(power);
    #ifdef LOD_MORPH
        if input.lod_morph_alpha.z > 0.0 || input.lod_morph_alpha.w > 0.0 {
            return lod_morph_fragment_color(
                input.lod_morph_alpha.x,
                input.lod_morph_alpha.y,
                gaussian_weight,
                input.lod_morph_parent_color.w,
                input.lod_morph_alpha.z,
                input.lod_morph_alpha.w,
                input.lod_morph_parent_color.rgb,
                input.color.rgb,
            );
        }
    #endif
    let alpha = min(gaussian_weight * input.color.a, 0.999);

    // TODO: round alpha to terminate depth test?

    return vec4<f32>(
        input.color.rgb * alpha,
        alpha,
    );
}
