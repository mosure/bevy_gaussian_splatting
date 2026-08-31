#define_import_path bevy_gaussian_splatting::lod_morph

// One compact physical-source keyed table is shared by compaction and raster.
// Compaction binds it after its existing four resources; raster appends it to
// the sorted-entry group. Keeping one additional storage binding preserves the
// WebGPU minimum of eight vertex storage buffers for precomputed covariance +
// debug + morph.
#ifdef LOD_MORPH_COMPACTION
    @group(3) @binding(4) var<storage, read> lod_morph_words: array<u32>;
#else
    @group(3) @binding(1) var<storage, read> lod_morph_words: array<u32>;
#endif

const LOD_PRESENTATION_HEADER_WORDS: u32 = 8u;
const LOD_MORPH_HEADER_WORDS: u32 = LOD_PRESENTATION_HEADER_WORDS;
const LOD_MORPH_DESCRIPTOR_WORDS: u32 = 8u;
const LOD_MORPH_MAPPING_WORDS: u32 = 2u;
const LOD_PRESENTATION_MODE_NONE: u32 = 0u;
const LOD_PRESENTATION_MODE_MORPH: u32 = 1u;
const LOD_PRESENTATION_MODE_EXTERNAL_ACTIVE_SET: u32 = 2u;
const LOD_PRESENTATION_MODE_WORD: u32 = 5u;
const LOD_PRESENTATION_FIRST_WEIGHT_WORD: u32 = 6u;
const LOD_PRESENTATION_SECOND_WEIGHT_WORD: u32 = 7u;
const LOD_EXTERNAL_ACTIVE_SET_SHARED: u32 = 0u;
const LOD_EXTERNAL_ACTIVE_SET_FIRST_ONLY: u32 = 1u;
const LOD_EXTERNAL_ACTIVE_SET_SECOND_ONLY: u32 = 2u;
const LOD_MORPH_ALPHA_LIMIT: f32 = 0.999999;
const LOD_MORPH_FRAGMENT_ALPHA_LIMIT: f32 = 0.999;
const LOD_MORPH_SCALE_FLOOR: f32 = 0.00000001;
const LOD_MORPH_QUATERNION_NORM_FLOOR: f32 = 0.000000000001;

struct LodMorphSample {
    parent_physical_index: u32,
    run_length: u32,
    blend_t: f32,
    enabled: bool,
}

fn lod_morph_inactive(child_physical_index: u32) -> LodMorphSample {
    return LodMorphSample(child_physical_index, 1u, 1.0, false);
}

fn lod_presentation_mode() -> u32 {
    if arrayLength(&lod_morph_words) < LOD_PRESENTATION_HEADER_WORDS {
        return LOD_PRESENTATION_MODE_NONE;
    }
    return lod_morph_words[LOD_PRESENTATION_MODE_WORD];
}

// The external path uses host-produced f32 bits exactly. Invalid class or
// weight payloads contribute no exclusive opacity; Shared never needs either
// weight and remains exact at both endpoints. Other presentation modes retain
// their ordinary authored opacity.
fn lod_external_active_set_opacity_coefficient(active_set_class: u32) -> f32 {
    if lod_presentation_mode() != LOD_PRESENTATION_MODE_EXTERNAL_ACTIVE_SET {
        return 1.0;
    }
    if active_set_class == LOD_EXTERNAL_ACTIVE_SET_SHARED {
        return 1.0;
    }
    if active_set_class == LOD_EXTERNAL_ACTIVE_SET_FIRST_ONLY {
        let first_weight = bitcast<f32>(
            lod_morph_words[LOD_PRESENTATION_FIRST_WEIGHT_WORD],
        );
        return select(0.0, first_weight, first_weight >= 0.0 && first_weight <= 1.0);
    }
    if active_set_class == LOD_EXTERNAL_ACTIVE_SET_SECOND_ONLY {
        let second_weight = bitcast<f32>(
            lod_morph_words[LOD_PRESENTATION_SECOND_WEIGHT_WORD],
        );
        return select(0.0, second_weight, second_weight >= 0.0 && second_weight <= 1.0);
    }
    return 0.0;
}

// Header words:
//   descriptor_count, mapping_record_start_words, mapping_record_count,
//   weight_start_words, weight_count, presentation_mode,
//   external_first_weight_bits, external_second_weight_bits.
// Each 8-word descriptor stores child_physical_start, child_count, and a
// mapping_start relative to the direct-record region. Descriptor word 3 is the
// dense immutable edge index. Each direct record is
// {parent_physical_index, run_length}; the final compact region stores exactly
// one host-derived displayed f32 weight per edge. Compaction and raster bind
// this same table, so they consume identical per-view bits.
// Host validation is authoritative, but every offset and record is checked
// again here so malformed/stale transition state falls back to the authored
// child instead of issuing an OOB read.
fn lod_morph_sample(
    child_physical_index: u32,
    source_count: u32,
) -> LodMorphSample {
    let table_word_count = arrayLength(&lod_morph_words);
    if table_word_count < LOD_MORPH_HEADER_WORDS
        || lod_presentation_mode() != LOD_PRESENTATION_MODE_MORPH
    {
        return lod_morph_inactive(child_physical_index);
    }

    let descriptor_count = lod_morph_words[0u];
    let mapping_record_start = lod_morph_words[1u];
    let mapping_record_count = lod_morph_words[2u];
    let weight_start = lod_morph_words[3u];
    let weight_count = lod_morph_words[4u];
    if descriptor_count == 0u
        || mapping_record_start < LOD_MORPH_HEADER_WORDS
        || mapping_record_start > table_word_count
        || weight_start < mapping_record_start
        || weight_start > table_word_count
    {
        return lod_morph_inactive(child_physical_index);
    }

    let descriptor_capacity =
        (mapping_record_start - LOD_MORPH_HEADER_WORDS) / LOD_MORPH_DESCRIPTOR_WORDS;
    let mapping_capacity = (weight_start - mapping_record_start) / LOD_MORPH_MAPPING_WORDS;
    let weight_capacity = table_word_count - weight_start;
    if descriptor_count > descriptor_capacity
        || mapping_record_count > mapping_capacity
        || weight_count > weight_capacity
    {
        return lod_morph_inactive(child_physical_index);
    }

    // Find the last sorted descriptor whose physical start is <= the child.
    var low = 0u;
    var high = descriptor_count;
    while low < high {
        let middle = low + (high - low) / 2u;
        let descriptor_word =
            LOD_MORPH_HEADER_WORDS + middle * LOD_MORPH_DESCRIPTOR_WORDS;
        let child_start = lod_morph_words[descriptor_word];
        if child_start <= child_physical_index {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    if low == 0u {
        return lod_morph_inactive(child_physical_index);
    }

    let descriptor_word =
        LOD_MORPH_HEADER_WORDS + (low - 1u) * LOD_MORPH_DESCRIPTOR_WORDS;
    let child_start = lod_morph_words[descriptor_word];
    let child_count = lod_morph_words[descriptor_word + 1u];
    let mapping_start = lod_morph_words[descriptor_word + 2u];
    let edge_index = lod_morph_words[descriptor_word + 3u];
    let child_relative = child_physical_index - child_start;
    if child_relative >= child_count
        || edge_index >= weight_count
        || mapping_start > mapping_record_count
        || child_relative >= mapping_record_count - mapping_start
    {
        return lod_morph_inactive(child_physical_index);
    }

    let mapping_index = mapping_start + child_relative;
    let mapping_word = mapping_record_start + mapping_index * LOD_MORPH_MAPPING_WORDS;
    let parent_physical_index = lod_morph_words[mapping_word];
    let run_length = lod_morph_words[mapping_word + 1u];
    if parent_physical_index >= source_count || run_length == 0u {
        return lod_morph_inactive(child_physical_index);
    }

    let blend_t = bitcast<f32>(lod_morph_words[weight_start + edge_index]);
    if !(blend_t >= 0.0 && blend_t <= 1.0) {
        return lod_morph_inactive(child_physical_index);
    }
    return LodMorphSample(parent_physical_index, run_length, blend_t, true);
}

fn lod_morph_position(
    parent_position: vec3<f32>,
    child_position: vec3<f32>,
    blend_t: f32,
) -> vec3<f32> {
    if blend_t <= 0.0 {
        return parent_position;
    }
    if blend_t >= 1.0 {
        return child_position;
    }
    return mix(parent_position, child_position, blend_t);
}

// Visibility is endpoint-authored state, not a field that may be averaged.
// Retain the union throughout the open blend interval so a proxy needed by
// either endpoint reaches raster, while preserving both exact endpoint cuts.
fn lod_morph_visibility(
    parent_visibility: f32,
    child_visibility: f32,
    blend_t: f32,
) -> f32 {
    if blend_t <= 0.0 {
        return parent_visibility;
    }
    if blend_t >= 1.0 {
        return child_visibility;
    }
    return max(parent_visibility, child_visibility);
}

// Both endpoint optical-depth terms remain nonzero in the open interval. The
// quad and fragment support must therefore enclose both endpoint cutoffs rather
// than interpolate to a smaller, lossy support.
fn lod_morph_support_cutoff(
    parent_cutoff: f32,
    child_cutoff: f32,
    blend_t: f32,
) -> f32 {
    if blend_t <= 0.0 {
        return parent_cutoff;
    }
    if blend_t >= 1.0 {
        return child_cutoff;
    }
    return max(parent_cutoff, child_cutoff);
}

fn lod_morph_log_scale(
    parent_scale: vec3<f32>,
    child_scale: vec3<f32>,
    blend_t: f32,
) -> vec3<f32> {
    if blend_t <= 0.0 {
        return parent_scale;
    }
    if blend_t >= 1.0 {
        return child_scale;
    }
    let safe_parent = max(abs(parent_scale), vec3<f32>(LOD_MORPH_SCALE_FLOOR));
    let safe_child = max(abs(child_scale), vec3<f32>(LOD_MORPH_SCALE_FLOOR));
    return exp(mix(log(safe_parent), log(safe_child), blend_t));
}

// A convex covariance blend can be wider than a log-scale interpolation when
// the endpoint principal axes differ. This spectral upper bound encloses that
// blend without requiring an eigenvalue solve in compaction or raster.
fn lod_morph_support_max_scale(
    parent_scale: vec3<f32>,
    child_scale: vec3<f32>,
    blend_t: f32,
) -> f32 {
    let parent_abs = abs(parent_scale);
    let child_abs = abs(child_scale);
    let parent_max = max(parent_abs.x, max(parent_abs.y, parent_abs.z));
    let child_max = max(child_abs.x, max(child_abs.y, child_abs.z));
    if blend_t <= 0.0 {
        return parent_max;
    }
    if blend_t >= 1.0 {
        return child_max;
    }
    return sqrt(max(mix(
        parent_max * parent_max,
        child_max * child_max,
        blend_t,
    ), 0.0));
}

fn lod_morph_rotation(
    parent_rotation: vec4<f32>,
    child_rotation: vec4<f32>,
    blend_t: f32,
) -> vec4<f32> {
    if blend_t <= 0.0 {
        return parent_rotation;
    }
    if blend_t >= 1.0 {
        return child_rotation;
    }
    var aligned_parent = parent_rotation;
    if dot(aligned_parent, child_rotation) < 0.0 {
        aligned_parent = -aligned_parent;
    }
    let interpolated = mix(aligned_parent, child_rotation, blend_t);
    let norm_squared = dot(interpolated, interpolated);
    if !(norm_squared > LOD_MORPH_QUATERNION_NORM_FLOOR) {
        return child_rotation;
    }
    return interpolated * inverseSqrt(norm_squared);
}

fn lod_morph_covariance(
    parent_covariance: array<f32, 6>,
    child_covariance: array<f32, 6>,
    blend_t: f32,
) -> array<f32, 6> {
    if blend_t <= 0.0 {
        return parent_covariance;
    }
    if blend_t >= 1.0 {
        return child_covariance;
    }
    var covariance: array<f32, 6>;
    for (var i = 0u; i < 6u; i += 1u) {
        covariance[i] = mix(parent_covariance[i], child_covariance[i], blend_t);
    }
    return covariance;
}

fn lod_morph_opacity_to_optical_depth(opacity: f32) -> f32 {
    let bounded = clamp(opacity, 0.0, LOD_MORPH_ALPHA_LIMIT);
    return -log(max(1.0 - bounded, 1.0 - LOD_MORPH_ALPHA_LIMIT));
}

fn lod_morph_fragment_color(
    parent_peak_alpha: f32,
    child_peak_alpha: f32,
    gaussian_weight: f32,
    morph_blend_t: f32,
    parent_optical_depth_coefficient: f32,
    child_optical_depth_coefficient: f32,
    parent_linear_rgb: vec3<f32>,
    child_linear_rgb: vec3<f32>,
) -> vec4<f32> {
    let parent_alpha = clamp(
        gaussian_weight * parent_peak_alpha,
        0.0,
        LOD_MORPH_FRAGMENT_ALPHA_LIMIT,
    );
    let child_alpha = clamp(
        gaussian_weight * child_peak_alpha,
        0.0,
        LOD_MORPH_FRAGMENT_ALPHA_LIMIT,
    );
    if morph_blend_t >= 1.0
        && parent_optical_depth_coefficient <= 0.0
        && child_optical_depth_coefficient >= 1.0
    {
        return vec4<f32>(child_linear_rgb * child_alpha, child_alpha);
    }
    if morph_blend_t <= 0.0
        && child_optical_depth_coefficient <= 0.0
        && parent_optical_depth_coefficient >= 1.0
    {
        return vec4<f32>(parent_linear_rgb * parent_alpha, parent_alpha);
    }

    // The vertex stage folds both endpoint blend factors and filtered
    // projected-area ratios into these coefficients. Applying them to optical
    // depth after evaluating the current interpolated Gaussian makes its
    // integrated optical-depth mass linear between endpoints. At the parent
    // endpoint each of K coincident child-cardinality proxies receives exactly
    // 1/K of the parent's per-covered-pixel optical depth; no second endpoint
    // set is rendered, so there is no double-density halo.
    let parent_tau = max(parent_optical_depth_coefficient, 0.0)
        * lod_morph_opacity_to_optical_depth(parent_alpha);
    let child_tau = max(child_optical_depth_coefficient, 0.0)
        * lod_morph_opacity_to_optical_depth(child_alpha);
    let total_tau = parent_tau + child_tau;
    if !(total_tau > 0.0) {
        return vec4<f32>(0.0);
    }
    let alpha = min(
        1.0 - exp(-(parent_tau + child_tau)),
        LOD_MORPH_FRAGMENT_ALPHA_LIMIT,
    );
    // Endpoint SH radiance is evaluated and converted to linear light in the
    // vertex stage. Mixing that linear radiance by the same optical depths used
    // for alpha preserves colored energy; interpolating SH (or sRGB) by t would
    // not, especially when endpoint opacity and projected area differ.
    let linear_rgb = (
        parent_linear_rgb * parent_tau + child_linear_rgb * child_tau
    ) / total_tau;
    return vec4<f32>(linear_rgb * alpha, alpha);
}
