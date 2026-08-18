#define_import_path bevy_gaussian_splatting::lod_debug

#import bevy_gaussian_splatting::bindings::{
    view,
    gaussian_uniforms,
}

// Must match gaussian::lod_debug::LodDebugRecord (40 bytes).
struct LodDebugRecord {
    page_color_key: u32,
    hierarchy_level: u32,
    residency: u32,
    boundary_distance_bits: u32,
    geometric_error: f32,
    quality_threshold: f32,
    node_center: array<f32, 3>,
    node_radius: f32,
};

@group(4) @binding(0) var<storage, read> lod_debug_records: array<LodDebugRecord>;

// Must match render::LodDebugGpuUniform (32 bytes).
struct LodDebugUniforms {
    // preset code, metadata count, reserved, reserved
    flags: vec4<u32>,
    // max error px, requested detail, reserved, reserved
    quality_params: vec4<f32>,
};

@group(4) @binding(1) var<uniform> lod_debug_uniforms: LodDebugUniforms;

// Must match gaussian::lod_settings quality-policy constants.
const HIGH_QUALITY_FIDELITY_GUARD_START: f32 = 0.90;
const HIGH_QUALITY_FIDELITY_GUARD_FULL: f32 = 0.99;
const HIGH_QUALITY_CERTIFICATE_GUARD_FULL: f32 = 0.95;
const PROJECTED_ERROR_AUTHORITY_FULL: f32 = 0.99;

const LOD_DEBUG_ORIGINAL_REPRESENTATION_BIT: u32 = 0x80000000u;
const LOD_DEBUG_RESIDENCY_MASK: u32 = 0x0000ffffu;
const LOD_DEBUG_CERTIFICATE_SHIFT: u32 = 16u;
const LOD_DEBUG_CERTIFICATE_MAX: f32 = 65535.0;
const LOD_DEBUG_BOUNDARY_WIDTH: f32 = 0.05;
const LOD_DEBUG_BOUNDARY_COLOR: vec3<f32> = vec3<f32>(0.05, 1.0, 0.2);
const LOD_DEBUG_RESIDENT_COLOR: vec3<f32> = vec3<f32>(0.1, 0.85, 0.25);
const LOD_DEBUG_FALLBACK_COLOR: vec3<f32> = vec3<f32>(1.0, 0.65, 0.05);
const LOD_DEBUG_UNKNOWN_COLOR: vec3<f32> = vec3<f32>(0.45, 0.45, 0.5);

fn lod_debug_residency(record: LodDebugRecord) -> u32 {
    return record.residency & LOD_DEBUG_RESIDENCY_MASK;
}

fn lod_debug_high_fidelity_certificate(record: LodDebugRecord) -> f32 {
    return f32(record.residency >> LOD_DEBUG_CERTIFICATE_SHIFT)
        / LOD_DEBUG_CERTIFICATE_MAX;
}

fn lod_debug_hash32(input: u32) -> u32 {
    var value = input ^ 0x9e3779b9u;
    value = value ^ (value >> 16u);
    value = value * 0x7feb352du;
    value = value ^ (value >> 15u);
    value = value * 0x846ca68bu;
    return value ^ (value >> 16u);
}

fn lod_debug_hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> vec3<f32> {
    let h = fract(hue) * 6.0;
    let sector = u32(floor(h)) % 6u;
    let fraction = h - floor(h);
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * fraction);
    let t = value * (1.0 - saturation * (1.0 - fraction));
    switch sector {
        case 0u: { return vec3<f32>(value, t, p); }
        case 1u: { return vec3<f32>(q, value, p); }
        case 2u: { return vec3<f32>(p, value, t); }
        case 3u: { return vec3<f32>(p, q, value); }
        case 4u: { return vec3<f32>(t, p, value); }
        default: { return vec3<f32>(value, p, q); }
    }
}

fn lod_debug_page_color(page_key: u32) -> vec3<f32> {
    let hash = lod_debug_hash32(page_key);
    let hue = f32(hash & 0x00ffffffu) / 16777215.0;
    return lod_debug_hsv_to_rgb(hue, 0.72, 0.95);
}

// Fixed, seed-independent level order: purple, cyan, green, yellow, orange,
// red, blue, pink for levels 0..7, then repeat for deeper hierarchies.
fn lod_debug_level_color(level: u32) -> vec3<f32> {
    switch level % 8u {
        case 0u: { return vec3<f32>(0.72, 0.32, 0.95); }
        case 1u: { return vec3<f32>(0.05, 0.78, 0.95); }
        case 2u: { return vec3<f32>(0.15, 0.85, 0.35); }
        case 3u: { return vec3<f32>(0.95, 0.85, 0.10); }
        case 4u: { return vec3<f32>(1.00, 0.48, 0.08); }
        case 5u: { return vec3<f32>(0.95, 0.12, 0.16); }
        case 6u: { return vec3<f32>(0.18, 0.35, 1.00); }
        default: { return vec3<f32>(1.00, 0.22, 0.65); }
    }
}

fn lod_debug_mix_pressure_color(
    pressure: f32,
    lower_pressure: f32,
    upper_pressure: f32,
    lower_color: vec3<f32>,
    upper_color: vec3<f32>,
) -> vec3<f32> {
    let amount = clamp(
        (pressure - lower_pressure) / max(upper_pressure - lower_pressure, 1e-20),
        0.0,
        1.0,
    );
    return mix(lower_color, upper_color, amount);
}

// Blue/cyan is safely below the requested target, green is exactly on target,
// and yellow/orange/red is increasingly too coarse.
fn lod_debug_selection_pressure_color(raw_pressure: f32) -> vec3<f32> {
    let blue = vec3<f32>(0.05, 0.20, 0.90);
    let cyan = vec3<f32>(0.00, 0.82, 1.00);
    let green = vec3<f32>(0.10, 0.90, 0.20);
    let yellow = vec3<f32>(1.00, 0.90, 0.00);
    let orange = vec3<f32>(1.00, 0.45, 0.00);
    let red = vec3<f32>(0.95, 0.05, 0.05);
    let pressure = select(max(raw_pressure, 0.0), 0.0, !(raw_pressure >= 0.0));
    if pressure <= 0.5 {
        return lod_debug_mix_pressure_color(pressure, 0.0, 0.5, blue, cyan);
    }
    if pressure <= 1.0 {
        return lod_debug_mix_pressure_color(pressure, 0.5, 1.0, cyan, green);
    }
    if pressure <= 1.5 {
        return lod_debug_mix_pressure_color(pressure, 1.0, 1.5, green, yellow);
    }
    if pressure <= 2.0 {
        return lod_debug_mix_pressure_color(pressure, 1.5, 2.0, yellow, orange);
    }
    if pressure <= 4.0 {
        return lod_debug_mix_pressure_color(pressure, 2.0, 4.0, orange, red);
    }
    return red;
}

struct LodDebugProjectedNode {
    geometric_error_px: f32,
    support_radius_px: f32,
};

// Reproduces LodView's projected error and support radius from the exact
// owning-node sphere. This remains O(1) per Gaussian and does not approximate
// node support from the representative Gaussian's position.
fn lod_debug_projected_node(record: LodDebugRecord) -> LodDebugProjectedNode {
    let transform_scale = gaussian_uniforms.transform_scale_bound;
    let geometric_error_world = max(record.geometric_error, 0.0) * transform_scale;
    let node_center_local = vec3<f32>(
        record.node_center[0],
        record.node_center[1],
        record.node_center[2],
    );
    let node_center_world = (
        gaussian_uniforms.transform * vec4<f32>(node_center_local, 1.0)
    ).xyz;
    let node_radius_world = max(record.node_radius, 0.0) * transform_scale;
    let focal_y_px = 0.5 * view.viewport.w * abs(view.clip_from_view[1][1]);
    var projection_scale_px_per_world = focal_y_px;
    if view.clip_from_view[3][3] == 1.0 {
        return LodDebugProjectedNode(
            geometric_error_world * projection_scale_px_per_world,
            node_radius_world * projection_scale_px_per_world,
        );
    }
    let near_plane = max(view.clip_from_view[3][2], 1e-20);
    let distance_to_surface = max(
        distance(view.world_position, node_center_world) - node_radius_world,
        near_plane,
    );
    projection_scale_px_per_world = focal_y_px / distance_to_surface;
    return LodDebugProjectedNode(
        geometric_error_world * projection_scale_px_per_world,
        node_radius_world * projection_scale_px_per_world,
    );
}

fn lod_debug_pressure_ratio(numerator: f32, denominator: f32) -> f32 {
    if denominator <= 0.0 {
        if numerator <= 0.0 {
            return 0.0;
        }
        return 3.402823e+38;
    }
    return min(numerator / denominator, 3.402823e+38);
}

fn lod_debug_smooth_quality_guard(
    detail_fraction: f32,
    start: f32,
    full: f32,
) -> f32 {
    if detail_fraction <= start {
        return 0.0;
    }
    if detail_fraction >= full {
        return 1.0;
    }
    let t = (detail_fraction - start) / (full - start);
    return t * t * (3.0 - 2.0 * t);
}

fn lod_debug_high_quality_fidelity_guard(detail_fraction: f32) -> f32 {
    return lod_debug_smooth_quality_guard(
        detail_fraction,
        HIGH_QUALITY_FIDELITY_GUARD_START,
        HIGH_QUALITY_FIDELITY_GUARD_FULL,
    );
}

fn lod_debug_high_quality_certificate_guard(detail_fraction: f32) -> f32 {
    let normalized = clamp(
        clamp(detail_fraction, 0.0, 1.0) / HIGH_QUALITY_CERTIFICATE_GUARD_FULL,
        0.0,
        1.0,
    );
    return normalized * normalized * normalized;
}

fn lod_debug_high_quality_certificate_demand(
    detail_fraction: f32,
    projected_coverage: f32,
) -> f32 {
    let detail = clamp(detail_fraction, 0.0, 1.0);
    let normalized = clamp(
        detail / HIGH_QUALITY_CERTIFICATE_GUARD_FULL,
        0.0,
        1.0,
    );
    let base_demand = detail * normalized;
    let authority = lod_debug_high_quality_certificate_guard(detail);
    let coverage = clamp(projected_coverage, 0.0, 1.0);
    let effective_coverage = coverage + (1.0 - coverage) * authority;
    return base_demand * effective_coverage;
}

fn lod_debug_projected_error_authority(detail_fraction: f32) -> f32 {
    let normalized = clamp(
        clamp(detail_fraction, 0.0, 1.0) / PROJECTED_ERROR_AUTHORITY_FULL,
        0.0,
        1.0,
    );
    return normalized * normalized * normalized;
}

fn lod_debug_selection_pressure(record: LodDebugRecord) -> f32 {
    let projected = lod_debug_projected_node(record);
    let selection_error = projected.geometric_error_px;
    let max_error_px = lod_debug_uniforms.quality_params.x;
    let is_original =
        (record.boundary_distance_bits & LOD_DEBUG_ORIGINAL_REPRESENTATION_BIT) != 0u;

    // Zero is the exact Original endpoint. Metadata carries an explicit leaf
    // bit because representative error alone cannot distinguish an exact leaf
    // from a zero-error internal fallback.
    if max_error_px == 0.0 {
        return select(3.402823e+38, 0.0, is_original);
    }

    let viewport_height_px = max(view.viewport.w, 1e-20);
    let projected_coverage = clamp(
        2.0 * projected.support_radius_px / viewport_height_px,
        0.0,
        1.0,
    );
    let requested_detail = max(lod_debug_uniforms.quality_params.y, 0.0);
    let fidelity_guard = lod_debug_high_quality_fidelity_guard(requested_detail);
    let effective_coverage = projected_coverage
        + (1.0 - projected_coverage) * fidelity_guard;
    let structural_demand = requested_detail * effective_coverage;
    let structural_pressure = lod_debug_pressure_ratio(
        structural_demand,
        max(record.quality_threshold, 0.0),
    );
    let error_pressure = lod_debug_pressure_ratio(selection_error, max_error_px);

    // Balanced selection accepts a node once either target is met, while the
    // continuous authority term limits how far the structural shortcut may
    // exceed the advertised pixel target. Positive quantized-compatible
    // certificates carry coverage-aware demand; legacy zero/tiny values stay
    // compatible below .95 and fail closed for non-original nodes at .95+.
    let balanced_pressure = min(structural_pressure, error_pressure);
    let error_authority = lod_debug_projected_error_authority(requested_detail);
    let guarded_error_pressure = max(
        balanced_pressure,
        error_authority * error_pressure,
    );
    let certificate = lod_debug_high_fidelity_certificate(record);
    var certificate_pressure = 0.0;
    if certificate <= 1.0 / LOD_DEBUG_CERTIFICATE_MAX {
        if requested_detail >= HIGH_QUALITY_CERTIFICATE_GUARD_FULL && !is_original {
            certificate_pressure = 3.402823e+38;
        }
    } else {
        let certificate_demand = lod_debug_high_quality_certificate_demand(
            requested_detail,
            projected_coverage,
        );
        certificate_pressure = lod_debug_pressure_ratio(
            certificate_demand,
            certificate,
        );
    }
    return max(guarded_error_pressure, certificate_pressure);
}

fn lod_debug_residency_color(residency: u32) -> vec3<f32> {
    switch residency {
        case 1u: { return LOD_DEBUG_RESIDENT_COLOR; }
        case 2u: { return LOD_DEBUG_FALLBACK_COLOR; }
        default: { return LOD_DEBUG_UNKNOWN_COLOR; }
    }
}

// Applies field coloring and the support-aware chunk boundary overlay. Missing
// metadata deliberately preserves authored color rather than reading out of
// bounds or fabricating a hierarchy value for an ordinary flat cloud.
fn apply_lod_debug_annotation(splat_index: u32, authored: vec3<f32>) -> vec3<f32> {
    let mode = lod_debug_uniforms.flags.x;
    let metadata_count = lod_debug_uniforms.flags.y;
    if (mode == 0u || splat_index >= metadata_count) {
        return authored;
    }

    let record = lod_debug_records[splat_index];
    var annotated = authored;

    switch mode {
        case 1u: {
            annotated = lod_debug_level_color(record.hierarchy_level);
        }
        case 2u: {
            annotated = lod_debug_page_color(record.page_color_key);
        }
        case 3u: {
            annotated = lod_debug_residency_color(lod_debug_residency(record));
        }
        case 4u: {
            // Boundaries retain authored appearance before the overlay below.
        }
        case 5u: {
            if lod_debug_uniforms.quality_params.x >= 0.0 {
                annotated = lod_debug_selection_pressure_color(
                    lod_debug_selection_pressure(record),
                );
            } else {
                // Metadata can be attached to a flat/non-LoD render entity.
                // Without extracted policy there is no truthful target.
                annotated = authored;
            }
        }
        default: {}
    }

    if mode == 4u {
        let distance_bits = record.boundary_distance_bits & 0x7fffffffu;
        let distance = bitcast<f32>(distance_bits);
        let boundary_factor = 1.0 - smoothstep(0.0, LOD_DEBUG_BOUNDARY_WIDTH, distance);
        annotated = mix(authored, LOD_DEBUG_BOUNDARY_COLOR, clamp(boundary_factor * 0.9, 0.0, 1.0));
    }

    return annotated;
}
