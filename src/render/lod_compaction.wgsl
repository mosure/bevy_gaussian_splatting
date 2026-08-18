#define_import_path bevy_gaussian_splatting::lod_compaction

#import bevy_gaussian_splatting::bindings::{
    gaussian_uniforms,
    view,
    Entry,
}
#ifdef PACKED_F32
#import bevy_gaussian_splatting::packed::{
    get_position,
    get_visibility,
    get_scale,
}
#else

#ifdef BUFFER_STORAGE
#import bevy_gaussian_splatting::planar::{
    get_position,
    get_visibility,
    get_scale,
}
#endif

#endif

struct LodCompactionUniform {
    source_count: u32,
    candidate_count: u32,
    output_capacity: u32,
    candidate_source_mode: u32,
    consumer_entries_a: u32,
    consumer_entries_c: u32,
    quality_endpoint: u32,
    frustum_culling: u32,
    frustum_margin: f32,
    candidate_range_count: u32,
    transform_scale_bound: f32,
    candidate_source_word_capacity: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
    padding_3: u32,
}

// The first 16 bytes are DrawIndirectArgs and bytes 16..28 are
// DispatchIndirectArgs. Atomic words retain the same four-byte ABI.
struct LodIndirectArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
    dispatch_c_x: u32,
    dispatch_c_y: u32,
    dispatch_c_z: u32,
    candidate_hits: atomic<u32>,
    overflow_count: atomic<u32>,
}

struct LodCandidateEvaluation {
    entry: Entry,
    accepted: u32,
}

@group(3) @binding(0) var<uniform> lod_config: LodCompactionUniform;
// Candidate words/range descriptors occupy only
// [0, candidate_source_word_capacity). A cached two-word evaluation for every
// candidate follows, then stable-scan records. This dynamic prefix avoids a
// resident 4*C candidate-word reserve for steady range frontiers while keeping
// group 3 to three storage buffers for the five-plane Gaussian4d layout.
@group(3) @binding(1) var<storage, read_write> candidate_and_scan_words: array<u32>;
@group(3) @binding(2) var<storage, read_write> active_entries: array<Entry>;
@group(3) @binding(3) var<storage, read_write> lod_indirect: LodIndirectArgs;

var<workgroup> scan_values: array<u32, 256>;

fn scan_group_capacity() -> u32 {
    return (lod_config.output_capacity + 255u) / 256u;
}

fn scan_block_record_index(block_index: u32) -> u32 {
    return scan_group_capacity() + block_index;
}

fn scan_record_word_index(record_index: u32, member: u32) -> u32 {
    return lod_config.candidate_source_word_capacity
        + lod_config.output_capacity * 2u
        + record_index * 2u
        + member;
}

fn evaluation_word_index(candidate_offset: u32, member: u32) -> u32 {
    return lod_config.candidate_source_word_capacity + candidate_offset * 2u + member;
}

fn store_candidate_evaluation(candidate_offset: u32, evaluation: LodCandidateEvaluation) {
    candidate_and_scan_words[evaluation_word_index(candidate_offset, 0u)] = evaluation.entry.key;
    candidate_and_scan_words[evaluation_word_index(candidate_offset, 1u)] = select(
        0xFFFFFFFFu,
        evaluation.entry.value,
        evaluation.accepted != 0u,
    );
}

fn load_candidate_evaluation(candidate_offset: u32) -> LodCandidateEvaluation {
    let value = candidate_and_scan_words[evaluation_word_index(candidate_offset, 1u)];
    return LodCandidateEvaluation(
        Entry(candidate_and_scan_words[evaluation_word_index(candidate_offset, 0u)], value),
        select(1u, 0u, value == 0xFFFFFFFFu),
    );
}

fn candidate_from_physical_ranges(candidate_offset: u32) -> u32 {
    var low = 0u;
    var high = lod_config.candidate_range_count;
    while low < high {
        let middle = low + (high - low) / 2u;
        let candidate_start = candidate_and_scan_words[middle * 4u];
        if candidate_start <= candidate_offset {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    if low == 0u {
        return 0xFFFFFFFFu;
    }
    let descriptor = low - 1u;
    let word = descriptor * 4u;
    let candidate_start = candidate_and_scan_words[word];
    let physical_start = candidate_and_scan_words[word + 1u];
    let count = candidate_and_scan_words[word + 2u];
    let relative = candidate_offset - candidate_start;
    if relative >= count {
        return 0xFFFFFFFFu;
    }
    return physical_start + relative;
}

fn scan_record_count(record_index: u32) -> u32 {
    return candidate_and_scan_words[scan_record_word_index(record_index, 0u)];
}

fn scan_record_offset(record_index: u32) -> u32 {
    return candidate_and_scan_words[scan_record_word_index(record_index, 1u)];
}

fn store_scan_record_count(record_index: u32, value: u32) {
    candidate_and_scan_words[scan_record_word_index(record_index, 0u)] = value;
}

fn store_scan_record_offset(record_index: u32, value: u32) {
    candidate_and_scan_words[scan_record_word_index(record_index, 1u)] = value;
}

// The renderer's non-adaptive Gaussian cutoff is 3 sigma. A sphere using the
// largest local scale, expanded by an upper bound on the cloud transform's
// largest singular value, conservatively contains the transformed ellipsoid
// even under non-uniform scale or shear.
fn support_radius_world(index: u32) -> f32 {
    let scale = abs(get_scale(index));
    let local_radius = 3.0 * abs(gaussian_uniforms.global_scale) * max(
        scale.x,
        max(scale.y, scale.z),
    );
    return local_radius * lod_config.transform_scale_bound;
}

fn support_sphere_in_frustum(center: vec3<f32>, support_radius: f32) -> bool {
    // Invalid support data is retained. Dropping it here would turn malformed
    // metadata into a non-conservative visibility decision.
    if !(support_radius >= 0.0) {
        return true;
    }

    let expanded_radius = support_radius + max(lod_config.frustum_margin, 0.0);
    for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
        let half_space = view.frustum[plane_index];
        let signed_distance = dot(half_space.xyz, center) + half_space.w;
        // Bevy ViewUniform frustum half-spaces have unit normals.
        if signed_distance < -expanded_radius {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(1)
fn lod_reset() {
    lod_indirect.vertex_count = 4u;
    atomicStore(&lod_indirect.instance_count, 0u);
    lod_indirect.first_vertex = 0u;
    lod_indirect.first_instance = 0u;
    lod_indirect.dispatch_x = 0u;
    lod_indirect.dispatch_y = 1u;
    lod_indirect.dispatch_z = 1u;
    lod_indirect.dispatch_c_x = 1u;
    lod_indirect.dispatch_c_y = 0u;
    lod_indirect.dispatch_c_z = 1u;
    atomicStore(&lod_indirect.candidate_hits, 0u);
    atomicStore(&lod_indirect.overflow_count, 0u);
}

fn evaluate_candidate(candidate_offset: u32) -> LodCandidateEvaluation {
    let rejected = LodCandidateEvaluation(Entry(0u, 0u), 0u);
    if candidate_offset >= lod_config.candidate_count {
        return rejected;
    }
    var source_index = candidate_offset;
    if lod_config.candidate_source_mode == 2u {
        source_index = candidate_from_physical_ranges(candidate_offset);
    }
    if (source_index >= lod_config.source_count || source_index >= gaussian_uniforms.count) {
        return rejected;
    }

    let position = vec4<f32>(get_position(source_index), 1.0);
    let transformed_position = (gaussian_uniforms.transform * position).xyz;
    let support_radius = support_radius_world(source_index);
    let visibility = get_visibility(source_index);
    if ((lod_config.frustum_culling != 0u &&
            !support_sphere_in_frustum(transformed_position, support_radius)) ||
        visibility <= 0.0) {
        return rejected;
    }
    let diff = transformed_position - view.world_position;
    let dist2 = dot(diff, diff);
    let key = (0xFFFFFFFFu - bitcast<u32>(dist2)) >> #{RADIX_KEY_SHIFT}u;
    return LodCandidateEvaluation(Entry(key, source_index), 1u);
}

// Inclusive Hillis-Steele scan. Every caller invokes this uniformly with one
// value per lane, so the barriers are workgroup-uniform and input order is
// retained. No invocation performs a serial scan over the frontier.
fn inclusive_workgroup_scan(local_index: u32) {
    workgroupBarrier();
    var offset = 1u;
    while offset < 256u {
        var addend = 0u;
        if local_index >= offset {
            addend = scan_values[local_index - offset];
        }
        workgroupBarrier();
        scan_values[local_index] = scan_values[local_index] + addend;
        workgroupBarrier();
        offset = offset * 2u;
    }
}

// Pass 1: reduce each 256-candidate workgroup to an accepted count. Candidate
// placement is intentionally deferred until all earlier workgroup counts have
// stable prefix offsets.
@compute @workgroup_size(256)
fn lod_count(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let evaluation = evaluate_candidate(global_id.x);
    if global_id.x < lod_config.candidate_count {
        store_candidate_evaluation(global_id.x, evaluation);
    }
    scan_values[local_index] = evaluation.accepted;
    inclusive_workgroup_scan(local_index);
    if local_index == 255u {
        store_scan_record_count(workgroup_id.x, scan_values[local_index]);
    }
}

// Pass 2: independently scan blocks of 256 workgroup counts. The final lane
// stores the block total for the bounded second scan level.
@compute @workgroup_size(256)
fn lod_scan_groups(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let group_count = (lod_config.candidate_count + 255u) / 256u;
    let group_index = workgroup_id.x * 256u + local_index;
    var count = 0u;
    if group_index < group_count {
        count = scan_record_count(group_index);
    }
    scan_values[local_index] = count;
    inclusive_workgroup_scan(local_index);
    if group_index < group_count {
        store_scan_record_offset(group_index, scan_values[local_index] - count);
    }
    if local_index == 255u {
        store_scan_record_count(
            scan_block_record_index(workgroup_id.x),
            scan_values[local_index],
        );
    }
}

// Pass 3: at most 256 block totals are scanned in parallel by one workgroup.
// This is the only fixed-size root of the bounded scan hierarchy.
@compute @workgroup_size(256)
fn lod_scan_blocks(@builtin(local_invocation_index) local_index: u32) {
    let group_count = (lod_config.candidate_count + 255u) / 256u;
    let block_count = (group_count + 255u) / 256u;
    var count = 0u;
    if local_index < block_count {
        count = scan_record_count(scan_block_record_index(local_index));
    }
    scan_values[local_index] = count;
    inclusive_workgroup_scan(local_index);
    if local_index < block_count {
        store_scan_record_offset(
            scan_block_record_index(local_index),
            scan_values[local_index] - count,
        );
    }
    if local_index + 1u == block_count {
        atomicStore(&lod_indirect.candidate_hits, scan_values[local_index]);
    }
}

// Pass 4: turn block-local group offsets into global exclusive offsets.
@compute @workgroup_size(256)
fn lod_add_block_offsets(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let group_count = (lod_config.candidate_count + 255u) / 256u;
    let group_index = global_id.x;
    if group_index >= group_count {
        return;
    }
    let block_offset = scan_record_offset(scan_block_record_index(group_index / 256u));
    store_scan_record_offset(group_index, scan_record_offset(group_index) + block_offset);
}

// Pass 5: scatter cached first-pass evaluations using the global group offset
// plus a stable local prefix. Rejected candidates are never
// written or sorted, and equal-key entries retain candidate input order.
@compute @workgroup_size(256)
fn lod_scatter(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    var evaluation = LodCandidateEvaluation(Entry(0u, 0u), 0u);
    if global_id.x < lod_config.candidate_count {
        evaluation = load_candidate_evaluation(global_id.x);
    }
    scan_values[local_index] = evaluation.accepted;
    inclusive_workgroup_scan(local_index);
    if evaluation.accepted != 0u {
        let output_index = scan_record_offset(workgroup_id.x)
            + scan_values[local_index] - 1u;
        if output_index < lod_config.output_capacity {
            active_entries[output_index] = evaluation.entry;
        } else {
            // The host contract guarantees candidate_count <= output_capacity;
            // keep this diagnostic defensive without using an atomic for
            // placement or ordering.
            atomicAdd(&lod_indirect.overflow_count, 1u);
        }
    }
}

@compute @workgroup_size(1)
fn lod_finalize() {
    let hits = atomicLoad(&lod_indirect.candidate_hits);
    let active_count = min(hits, lod_config.output_capacity);
    atomicStore(&lod_indirect.instance_count, active_count);
    let entries_a = max(lod_config.consumer_entries_a, 1u);
    let entries_c = max(lod_config.consumer_entries_c, 1u);
    lod_indirect.dispatch_x = (active_count + entries_a - 1u) / entries_a;
    lod_indirect.dispatch_y = 1u;
    lod_indirect.dispatch_z = 1u;
    lod_indirect.dispatch_c_x = 1u;
    lod_indirect.dispatch_c_y = (active_count + entries_c - 1u) / entries_c;
    lod_indirect.dispatch_c_z = 1u;
}
