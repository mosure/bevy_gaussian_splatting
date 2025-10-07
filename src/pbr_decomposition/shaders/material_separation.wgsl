#define_import_path bevy_gaussian_splatting::pbr_decomposition::material_separation

#import bevy_gaussian_splatting::bindings::{
    position_visibility,
}

#ifdef PACKED
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::packed::{
            get_position,
        }
    #else
        #import bevy_gaussian_splatting::packed::{
            get_position,
        }
    #endif
#else ifdef BUFFER_STORAGE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::planar::{
            get_position,
        }
    #else
        #import bevy_gaussian_splatting::planar::{
            get_position,
        }
    #endif
#else ifdef BUFFER_TEXTURE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::texture::{
            get_position,
        }
    #else
        #import bevy_gaussian_splatting::texture::{
            get_position,
        }
    #endif
#endif

struct StreamingStats {
    mean_rgb: vec3<f32>,
    count: u32,

    M2_rgb: vec3<f32>,
    near_normal_count: u32,

    near_normal_mean: vec3<f32>,
    topk_count: u32,

    topk_directions: array<vec3<f32>, 8>,
    topk_intensities: array<f32, 8>,

    residual_direction_sum: vec3<f32>,
    residual_direction_M2: f32,

    _pad: array<f32, 3>,
}

struct PbrMaterialData {
    base_color: vec3<f32>,
    metallic: f32,
    perceptual_roughness: f32,
    reflectance: f32,
    ambient_occlusion: f32,
    _pad: f32,
}

struct MaterialSettings {
    roughness_min: f32,
    roughness_max: f32,
    metallic_saturation_threshold: f32,
    metallic_min_threshold: f32,
}

struct GaussianMaterialOverride {
    base_color_factor: vec4<f32>,
    bounds_min: vec4<f32>,
    bounds_size: vec4<f32>,
    flags: vec4<u32>,
}

// Group 3: IO buffers for this pipeline
@group(3) @binding(0) var<storage, read> stats: array<StreamingStats>;
@group(3) @binding(1) var<storage, read_write> materials: array<PbrMaterialData>;

// Group 4: settings
@group(4) @binding(0) var<uniform> settings: MaterialSettings;

// Group 5: material overrides
@group(5) @binding(0) var<uniform> material_override: GaussianMaterialOverride;
@group(5) @binding(1) var base_color_texture: texture_2d<f32>;
@group(5) @binding(2) var base_color_sampler: sampler;

const COLOR_EPSILON: f32 = 1e-6;

fn compute_saturation_robust(color: vec3<f32>) -> f32 {
    let c = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    let max_c = max(c.r, max(c.g, c.b));

    if (max_c < COLOR_EPSILON) {
        return 0.0;
    }

    let min_c = min(c.r, min(c.g, c.b));
    let delta = max_c - min_c;

    if (delta < COLOR_EPSILON) {
        return 0.0;
    }

    return delta / max_c;
}

fn smoothstep_custom(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(256)
fn estimate_material_properties(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let gaussian_count = arrayLength(&stats);
    if (idx >= gaussian_count) { return; }

    let stat = stats[idx];

    var base_color = select(
        stat.mean_rgb,
        stat.near_normal_mean,
        stat.near_normal_count > 8u
    );

    let base_color_factor = material_override.base_color_factor.rgb;
    base_color *= base_color_factor;

    if (material_override.flags.x > 0u) {
        let axis = material_override.flags.y;
        let position = get_position(idx);

        var projected_position = vec2<f32>(0.0);
        var bounds_projected_min = vec2<f32>(0.0);
        var bounds_projected_size = vec2<f32>(1.0);

        if (axis == 0u) {
            projected_position = position.xy;
            bounds_projected_min = material_override.bounds_min.xy;
            bounds_projected_size = material_override.bounds_size.xy;
        } else if (axis == 1u) {
            projected_position = vec2<f32>(position.x, position.z);
            bounds_projected_min = vec2<f32>(material_override.bounds_min.x, material_override.bounds_min.z);
            bounds_projected_size = vec2<f32>(material_override.bounds_size.x, material_override.bounds_size.z);
        } else {
            projected_position = position.yz;
            bounds_projected_min = material_override.bounds_min.yz;
            bounds_projected_size = material_override.bounds_size.yz;
        }

        bounds_projected_size = max(bounds_projected_size, vec2<f32>(1e-3));
        var uv = (projected_position - bounds_projected_min) / bounds_projected_size;
        uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));

        let texture_color = textureSampleLevel(base_color_texture, base_color_sampler, uv, 0.0).rgb;
        base_color = texture_color * base_color;
    }

    base_color = clamp(base_color, vec3<f32>(0.0), vec3<f32>(1.0));

    var roughness = 0.5;

    if (stat.topk_count > 3u) {
        let angular_std_dev = sqrt(stat.residual_direction_M2);

        roughness = mix(
            settings.roughness_min,
            settings.roughness_max,
            saturate(angular_std_dev / 1.2)
        );
    } else {
        roughness = 0.8;
    }

    let base_saturation = compute_saturation_robust(base_color);

    let base_energy = length(stat.near_normal_mean);
    let spec_energy = length(stat.mean_rgb - stat.near_normal_mean);
    let total_energy = base_energy + spec_energy;
    let spec_ratio = spec_energy / max(0.0001, total_energy);

    var metallic = 0.0;

    if (base_saturation < settings.metallic_saturation_threshold && spec_ratio > 0.4) {
        metallic = smoothstep_custom(0.4, 0.9, spec_ratio);
    } else if (base_saturation > 0.3 && spec_ratio < 0.3) {
        metallic = 0.0;
    } else {
        metallic = smoothstep_custom(0.5, 0.9, spec_ratio);
    }

    metallic = select(0.0, metallic, metallic >= settings.metallic_min_threshold);

    let reflectance = 0.5;
    let ao = 1.0;

    materials[idx] = PbrMaterialData(
        base_color,
        metallic,
        roughness,
        reflectance,
        ao,
        0.0
    );
}
