#import bevy_gaussian_splatting::bindings::{view, gaussian_uniforms, Entry}
#import bevy_gaussian_splatting::planar::{get_opacity, get_visibility}
#import bevy_gaussian_splatting::projection::{projected_world_position, projected_support_in_frustum, projected_covariance, projected_color}
#import bevy_gaussian_splatting::point_types::PointGaussian
#import bevy_gaussian_splatting::point_sampling::{point_dilog, point_rectangle, point_rectangle_peak}

struct ProjectConfig {
    output_start: u32,
    input_capacity: u32,
    source_seed: u32,
    frame: u32,
    identity_source: u32,
    draw_mode: u32,
    traversed_source: u32,
    padding: u32,
}
@group(3) @binding(0) var<uniform> config: ProjectConfig;
@group(3) @binding(1) var<storage, read> entries: array<Entry>;
@group(3) @binding(2) var<storage, read> indirect: array<u32>;
@group(3) @binding(3) var<storage, read_write> projected: array<PointGaussian>;

@compute @workgroup_size(256)
fn project(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let local = invocation.x;
    // A missing root is a failed complete cut, not an empty visible scene.
    // Only traversal sources carry status after their four draw arguments.
    if config.traversed_source != 0u && (indirect[4] & 1u) != 0u {
        if local == 0u {
            projected[config.output_start].epoch = config.frame;
            projected[config.output_start].failure = 4u;
            projected[config.output_start].opacity = 0.0;
        }
        return;
    }
    let count = select(indirect[1], gaussian_uniforms.count, config.identity_source != 0u);
    if local >= count || local >= config.input_capacity { return; }
    var index = local;
    if config.identity_source == 0u { index = entries[local].value & 0x0fffffffu; }
    // Traversal preserves logical representative identity across page placement
    // and priority changes. Flat and CPU-compacted inputs keep their own seeds.
    var sampling_identity = index;
    if config.traversed_source != 0u { sampling_identity = entries[local].key; }
    let output_index = config.output_start + local;
    projected[output_index].epoch = 0u;
    if index >= gaussian_uniforms.count { return; }
    let visibility = get_visibility(index);
    if config.draw_mode == 1u && visibility < 0.5 { return; }
    let position = projected_world_position(index);
    if !projected_support_in_frustum(index, position, 3.0) { return; }
    let clip = view.clip_from_world * vec4<f32>(position, 1.0);
    if !(clip.w > 0.0) { return; }
    let depth = clip.z / clip.w;
    if !(depth > 0.0 && depth <= 1.0) { return; }
    let covariance = projected_covariance(index, position);
    // The shared projection helpers use two coordinate units per physical pixel.
    let cov = covariance.xyz * 0.25;
    let determinant = cov.x * cov.z - cov.y * cov.y;
    if !(cov.x > 0.0 && determinant > 0.0) { return; }
    let opacity = clamp(get_opacity(index) * gaussian_uniforms.global_opacity * covariance.w, 0.0, 0.999);
    if !(opacity > 0.0) { return; }
    let mean = (clip.xy / clip.w * vec2<f32>(0.5, -0.5) + 0.5) * view.viewport.zw;
    let radius = 3.0 * sqrt(vec2<f32>(cov.x, cov.z));
    if any(mean + radius < vec2<f32>(0.0)) || any(mean - radius >= view.viewport.zw) { return; }
    let basis_x = sqrt(cov.x);
    let basis_xy = cov.y / basis_x;
    let basis_y = sqrt(determinant / cov.x);
    let rectangle = point_rectangle(mean, vec2<f32>(basis_x, basis_xy), basis_y, view.viewport.zw);
    let extent = rectangle.zw - rectangle.xy;
    let peak = point_rectangle_peak(opacity, mean, vec2<f32>(basis_x, basis_xy), basis_y, rectangle);
    let rectangle_rate = extent.x * extent.y * peak;
    let dilog = point_dilog(opacity);
    let full_rate = 6.283185307179586 * basis_x * basis_y * dilog;
    let use_rectangle = rectangle_rate < full_rate;
    var color = projected_color(index, position);
    if config.draw_mode == 2u && visibility >= 0.5 { color = vec3<f32>(1.0, 0.0, 0.0); }
    projected[output_index] = PointGaussian(mean, vec2<f32>(basis_x, basis_xy), basis_y,
        opacity, dilog, depth, color,
        config.source_seed ^ (sampling_identity * 0x9e3779b9u), config.frame, 0u,
        select(full_rate, rectangle_rate, use_rectangle), select(0.0, peak, use_rectangle));
}
