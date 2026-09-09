#define_import_path bevy_gaussian_splatting::lod_spatial_projection

#import bevy_gaussian_splatting::bindings::{view, gaussian_uniforms}
#import bevy_gaussian_splatting::projection::projected_world_position
#import bevy_gaussian_splatting::planar::get_scale
#import bevy_gaussian_splatting::gaussian_3d::compute_cov2d_3dgs
#import bevy_gaussian_splatting::helpers::gaussian_mip_support_radius_world
#import bevy_gaussian_splatting::lod_morph::lod_morph_support_max_scale

// Reuse the ordinary projection's two input storage bindings. For spatial
// traversal the first has a bounded metadata tail and the second is immutable
// correspondence. Nonspatial sources keep their original indirect header.
@group(3) @binding(1) var<storage, read> spatial_entries: array<u32>;
@group(3) @binding(2) var<storage, read> spatial_mapping: array<u32>;

struct SpatialSample {
    parent: u32,
    split_count: u32,
    weight: f32,
    enabled: bool,
}

struct SpatialProjection {
    position: vec3<f32>,
    parent_position: vec3<f32>,
    child_position: vec3<f32>,
    covariance: vec4<f32>,
    parent_opacity_scale: f32,
    child_opacity_scale: f32,
    parent_coefficient: f32,
    child_coefficient: f32,
}

fn spatial_entry(local: u32) -> vec2<u32> {
    return vec2<u32>(spatial_entries[local * 2u], spatial_entries[local * 2u + 1u]);
}

fn spatial_draw_word(capacity: u32, enabled: bool, word: u32) -> u32 {
    if enabled { return spatial_entries[capacity * 2u + word]; }
    return spatial_mapping[word];
}

fn spatial_sample(local: u32, capacity: u32, enabled: bool, child: u32) -> SpatialSample {
    let inactive = SpatialSample(child, 1u, 1.0, false);
    if !enabled { return inactive; }
    let tail = 2u * capacity;
    let ranges = spatial_entries[tail + 2u];
    if ranges == 0u { return inactive; }
    var low = 0u;
    var high = ranges;
    while low < high {
        let middle = low + (high - low) / 2u;
        if spatial_entries[tail + 8u + middle * 8u] <= local { low = middle + 1u; }
        else { high = middle; }
    }
    if low == 0u { return inactive; }
    let base = tail + 8u + (low - 1u) * 8u;
    let relative = local - spatial_entries[base];
    if relative >= spatial_entries[base + 1u] || spatial_entries[base + 7u] == 0u { return inactive; }
    let ordinal = spatial_entries[base + 3u] + relative;
    let runs = spatial_entries[base + 5u];
    let start = spatial_mapping[2] + spatial_entries[base + 4u];
    low = 0u;
    high = runs;
    while low < high {
        let middle = low + (high - low) / 2u;
        if spatial_mapping[start + middle] <= ordinal { low = middle + 1u; }
        else { high = middle; }
    }
    if low >= runs { return inactive; }
    var previous = 0u;
    if low > 0u { previous = spatial_mapping[start + low - 1u]; }
    let count = spatial_mapping[start + low] - previous;
    let parent = spatial_entries[base + 2u] + low;
    let weight = bitcast<f32>(spatial_entries[base + 6u]);
    if parent >= gaussian_uniforms.count || count == 0u || !(weight > 0.0 && weight < 1.0) { return inactive; }
    return SpatialSample(parent, count, weight, true);
}

fn spatial_projection(child: u32, sample: SpatialSample) -> SpatialProjection {
    let child_position = projected_world_position(child);
    let parent_position = projected_world_position(sample.parent);
    var position = child_position;
    if sample.enabled { position = mix(parent_position, child_position, sample.weight); }
    let covariance = compute_cov2d_3dgs(position, parent_position, child_position,
        child, sample.parent, sample.weight, sample.enabled);
    return SpatialProjection(position, parent_position, child_position, covariance.filtered,
        covariance.parent_opacity_scale, covariance.child_opacity_scale,
        (1.0 - sample.weight) * covariance.parent_projected_area_ratio / f32(sample.split_count),
        sample.weight * covariance.child_projected_area_ratio);
}

fn spatial_support_in_frustum(child: u32, sample: SpatialSample, position: vec3<f32>, cutoff: f32) -> bool {
    let mip = gaussian_mip_support_radius_world(position, cutoff);
    let scale = lod_morph_support_max_scale(get_scale(sample.parent), get_scale(child), sample.weight);
    let radius = cutoff * abs(gaussian_uniforms.global_scale) * scale
        * gaussian_uniforms.transform_scale_bound + mip;
    if !(mip >= 0.0 && radius >= 0.0) { return true; }
    for (var plane = 0u; plane < 6u; plane += 1u) {
        if dot(view.frustum[plane].xyz, position) + view.frustum[plane].w < -radius { return false; }
    }
    return true;
}
