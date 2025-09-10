#import bevy_gaussian_splatting::bindings::{
    view,
    globals,
    gaussian_uniforms,
    sorting_pass_index,
    sorting,
    status_counters,
    draw_indirect,
    input_entries,
    output_entries,
    sorted_entries,
    DrawIndirect,
    Entry,
}
#import bevy_gaussian_splatting::transform::{
    world_to_clip,
    in_frustum,
}

#ifdef PACKED_F32
#import bevy_gaussian_splatting::packed::get_position
#else

#ifdef BUFFER_STORAGE
#import bevy_gaussian_splatting::planar::get_position
#endif

#endif

#ifdef BUFFER_TEXTURE
#import bevy_gaussian_splatting::texture::get_position
#endif


struct SortingGlobal {
    digit_histogram: array<array<atomic<u32>, #{RADIX_BASE}>, #{RADIX_DIGIT_PLACES}>,
    assignment_counter: atomic<u32>,
}

@group(3) @binding(0) var<uniform> sorting_pass_index: u32;
@group(3) @binding(1) var<storage, read_write> sorting: SortingGlobal;
@group(3) @binding(2) var<storage, read_write> status_counters: array<array<atomic<u32>, #{RADIX_BASE}>>;
@group(3) @binding(3) var<storage, read_write> draw_indirect: DrawIndirect;
@group(3) @binding(4) var<storage, read_write> input_entries: array<Entry>;
@group(3) @binding(5) var<storage, read_write> output_entries: array<Entry>;


struct SortingSharedA {
    digit_histogram: array<array<atomic<u32>, #{RADIX_BASE}>, #{RADIX_DIGIT_PLACES}>,
}
var<workgroup> sorting_shared_a: SortingSharedA;

@compute @workgroup_size(#{RADIX_BASE}, #{RADIX_DIGIT_PLACES})
fn radix_sort_a(
    @builtin(local_invocation_id) gl_LocalInvocationID: vec3<u32>,
    @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
) {
    if (gl_LocalInvocationID.x == 0u && gl_LocalInvocationID.y == 0u && gl_GlobalInvocationID.x == 0u) {
        // Initialize draw counts early so the draw call doesn't get zeroed if later passes stall
        draw_indirect.vertex_count = 4u;
        atomicStore(&draw_indirect.instance_count, gaussian_uniforms.count);
    }
    sorting_shared_a.digit_histogram[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0u;
    workgroupBarrier();

    let thread_index = gl_GlobalInvocationID.x * #{RADIX_DIGIT_PLACES}u + gl_GlobalInvocationID.y;
    let start_entry_index = thread_index * #{ENTRIES_PER_INVOCATION_A}u;
    let end_entry_index = start_entry_index + #{ENTRIES_PER_INVOCATION_A}u;

    for (var entry_index = start_entry_index; entry_index < end_entry_index; entry_index += 1u) {
        if (entry_index >= gaussian_uniforms.count) {
            continue;
        }

        var key: u32 = 0xFFFFFFFFu;
        let position = vec4<f32>(get_position(entry_index), 1.0);
        let transformed_position = (gaussian_uniforms.transform * position).xyz;
        let clip_space_pos = world_to_clip(transformed_position);

        // Use full-precision squared distance (monotonic with true distance for positive values)
        // to avoid quantization artifacts. We invert the float bit pattern so that an ascending
        // integer radix sort produces farthest-first ordering. For positive finite f32 values the
        // bit pattern ordering matches numeric ordering, so inverting achieves the desired sort.
        // (We deliberately avoid sqrt to save cycles and keep higher relative precision.)
        let diff = transformed_position - view.world_position;
        let dist2 = dot(diff, diff); // squared distance
        let dist_bits = bitcast<u32>(dist2);
        let key_distance = 0xFFFFFFFFu - dist_bits;

        if (in_frustum(clip_space_pos.xyz)) {
            key = key_distance;
        }

        output_entries[entry_index].key = key;
        output_entries[entry_index].value = entry_index;

        for(var shift = 0u; shift < #{RADIX_DIGIT_PLACES}u; shift += 1u) {
            let digit = (key >> (shift * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
            atomicAdd(&sorting_shared_a.digit_histogram[shift][digit], 1u);
        }
    }
    workgroupBarrier();

    atomicAdd(&sorting.digit_histogram[gl_LocalInvocationID.y][gl_LocalInvocationID.x], sorting_shared_a.digit_histogram[gl_LocalInvocationID.y][gl_LocalInvocationID.x]);
}

@compute @workgroup_size(1)
fn radix_sort_b(
    @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
) {
    // Exclusive scan of per-digit counts for each digit place
    var sum = 0u;
    for(var digit = 0u; digit < #{RADIX_BASE}u; digit += 1u) {
        let tmp = atomicLoad(&sorting.digit_histogram[gl_GlobalInvocationID.y][digit]);
        atomicStore(&sorting.digit_histogram[gl_GlobalInvocationID.y][digit], sum);
        sum += tmp;
    }
}

struct SortingSharedC {
    // Legacy fields (not relied on for algorithmic correctness)
    entries: array<atomic<u32>, #{WORKGROUP_ENTRIES_C}>,
    gather_sources: array<atomic<u32>, #{WORKGROUP_ENTRIES_C}>,
    // Pad scan array to avoid bank-conflict offset running out-of-bounds
    scan: array<atomic<u32>, #{WORKGROUP_INVOCATIONS_C} + (#{WORKGROUP_INVOCATIONS_C} >> LOG_NUM_BANKS)>,
    total: u32,
}
var<workgroup> sorting_shared_c: SortingSharedC;

// Additional shared arrays for stable multi-split within a tile
var<workgroup> tile_entries: array<u32, #{WORKGROUP_ENTRIES_C}>;
var<workgroup> counts_ws: array<u32, #{RADIX_BASE}u * #{WORKGROUP_INVOCATIONS_C}u>;
var<workgroup> digit_totals_ws: array<u32, #{RADIX_BASE}u>;
var<workgroup> digit_offsets_ws: array<u32, #{RADIX_BASE}u>;
var<workgroup> digit_global_base_ws: array<u32, #{RADIX_BASE}u>;
// New: per-iteration per-digit totals and prefixes to ensure stability
var<workgroup> digit_iter_totals_ws: array<u32, #{RADIX_BASE}u * #{ENTRIES_PER_INVOCATION_C}u>;
var<workgroup> iter_prefix_ws: array<u32, #{RADIX_BASE}u * #{ENTRIES_PER_INVOCATION_C}u>;
const INVALID_DIGIT: u32 = #{RADIX_BASE}u;

// Reset pass to clear per-frame counters and histograms
@compute @workgroup_size(#{RADIX_BASE}, #{RADIX_DIGIT_PLACES})
fn radix_reset(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
){
    let b = local_id.x;
    let p = local_id.y;

    atomicStore(&sorting.digit_histogram[p][b], 0u);
    atomicStore(&status_counters[p][b], 0u);

    if (global_id.x == 0u && global_id.y == 0u) {
        atomicStore(&sorting.assignment_counter, 0u);
        draw_indirect.instance_count = 0u;
    }
}

const NUM_BANKS: u32 = 16u;
const LOG_NUM_BANKS: u32 = 4u;
fn conflict_free_offset(n: u32) -> u32 {
    // Simple bank-conflict padding to reduce contention
    return n >> LOG_NUM_BANKS;
}

// Note: kept here for completeness; the stable multi-split below does not rely on this scan.
fn exclusive_scan(local_invocation_index: u32, value: u32) -> u32 {
    sorting_shared_c.scan[local_invocation_index + conflict_free_offset(local_invocation_index)] = value;

    var offset = 1u;
    for (var d = #{WORKGROUP_INVOCATIONS_C}u >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if(local_invocation_index < d) {
            var ai = offset * (2u * local_invocation_index + 1u) - 1u;
            var bi = offset * (2u * local_invocation_index + 2u) - 1u;
            ai += conflict_free_offset(ai);
            bi += conflict_free_offset(bi);
            sorting_shared_c.scan[bi] += sorting_shared_c.scan[ai];
        }

        offset <<= 1u;
    }

    if (local_invocation_index == 0u) {
      var i = #{WORKGROUP_INVOCATIONS_C}u - 1u;
      i += conflict_free_offset(i);
      sorting_shared_c.total = sorting_shared_c.scan[i];
      sorting_shared_c.scan[i] = 0u;
    }

    for (var d = 1u; d < #{WORKGROUP_INVOCATIONS_C}u; d <<= 1u) {
        workgroupBarrier();
        offset >>= 1u;
        if(local_invocation_index < d) {
            var ai = offset * (2u * local_invocation_index + 1u) - 1u;
            var bi = offset * (2u * local_invocation_index + 2u) - 1u;
            ai += conflict_free_offset(ai);
            bi += conflict_free_offset(bi);
            let t = sorting_shared_c.scan[ai];
            sorting_shared_c.scan[ai] = sorting_shared_c.scan[bi];
            sorting_shared_c.scan[bi] += t;
        }
    }

    workgroupBarrier();
    return sorting_shared_c.scan[local_invocation_index + conflict_free_offset(local_invocation_index)];
}

@compute @workgroup_size(#{WORKGROUP_INVOCATIONS_C})
fn radix_sort_c(
    @builtin(local_invocation_id) gl_LocalInvocationID: vec3<u32>,
    @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
    @builtin(workgroup_id) gl_WorkgroupID: vec3<u32>,
) {
    let tid = gl_LocalInvocationID.x;
    let tile_index = gl_WorkgroupID.y;

    // Compute global offset for this tile
    let global_entry_offset = tile_index * #{WORKGROUP_ENTRIES_C}u;
    if (global_entry_offset >= gaussian_uniforms.count) { return; }

    // Load input and compute deterministic local ranks
    var keys: array<u32, #{ENTRIES_PER_INVOCATION_C}>;
    var values: array<u32, #{ENTRIES_PER_INVOCATION_C}>;
    var digit_of: array<u32, #{ENTRIES_PER_INVOCATION_C}>;
    var local_rank_in_thread: array<u32, #{ENTRIES_PER_INVOCATION_C}>;

    // Zero per-thread per-digit counts
    for (var d = 0u; d < #{RADIX_BASE}u; d += 1u) {
        counts_ws[d * #{WORKGROUP_INVOCATIONS_C}u + tid] = 0u;
    }
    workgroupBarrier();

    // Load & compute local ranks in input order; also record per-iteration digits for stability
    for (var i = 0u; i < #{ENTRIES_PER_INVOCATION_C}u; i += 1u) {
        let idx = global_entry_offset + #{WORKGROUP_INVOCATIONS_C}u * i + tid;
        if (idx < gaussian_uniforms.count) {
            let k = input_entries[idx].key;
            let v = input_entries[idx].value;
            keys[i] = k;
            values[i] = v;

            let d = (k >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
            digit_of[i] = d;

            let off = d * #{WORKGROUP_INVOCATIONS_C}u + tid;
            let lr = counts_ws[off];
            local_rank_in_thread[i] = lr;
            counts_ws[off] = lr + 1u;

            // Record digit in tile input order for stable placement
            tile_entries[i * #{WORKGROUP_INVOCATIONS_C}u + tid] = d;
        } else {
            keys[i] = 0xFFFFFFFFu;
            values[i] = 0xFFFFFFFFu;
            digit_of[i] = 0u;
            local_rank_in_thread[i] = 0u;
            tile_entries[i * #{WORKGROUP_INVOCATIONS_C}u + tid] = INVALID_DIGIT;
        }
    }
    workgroupBarrier();

    // Per-digit totals for this tile (across all iterations)
    if (tid < #{RADIX_BASE}u) {
        var total = 0u;
        for (var t = 0u; t < #{WORKGROUP_INVOCATIONS_C}u; t += 1u) {
            total += counts_ws[tid * #{WORKGROUP_INVOCATIONS_C}u + t];
        }
        digit_totals_ws[tid] = total;
    }
    workgroupBarrier();

    // Compute per-iteration per-digit totals: digit_iter_totals_ws[d][i]
    if (tid < #{RADIX_BASE}u) {
        for (var i = 0u; i < #{ENTRIES_PER_INVOCATION_C}u; i += 1u) {
            var tcount = 0u;
            let base_index = i * #{WORKGROUP_INVOCATIONS_C}u;
            for (var t = 0u; t < #{WORKGROUP_INVOCATIONS_C}u; t += 1u) {
                let dd = tile_entries[base_index + t];
                if (dd == tid) { tcount += 1u; }
            }
            digit_iter_totals_ws[tid * #{ENTRIES_PER_INVOCATION_C}u + i] = tcount;
        }
    }
    workgroupBarrier();

    // Compute per-iteration exclusive prefix across iterations for each digit: iter_prefix_ws[d][i]
    if (tid < #{RADIX_BASE}u) {
        var acc = 0u;
        for (var i = 0u; i < #{ENTRIES_PER_INVOCATION_C}u; i += 1u) {
            let idxp = tid * #{ENTRIES_PER_INVOCATION_C}u + i;
            let t = digit_iter_totals_ws[idxp];
            iter_prefix_ws[idxp] = acc;
            acc += t;
        }
    }
    workgroupBarrier();

    // Publish per-digit global base via lookback; also set draw indirect if final pass
    if (tid < #{RADIX_BASE}u) {
        let local_total = digit_totals_ws[tid];
        atomicStore(&status_counters[tile_index][tid], 0x40000000u | local_total);
        storageBarrier();

        var global_digit_count = 0u;
        var prev = tile_index;
        loop {
            if (prev == 0u) {
                // Add global base (exclusive) for this digit across the whole array
                global_digit_count += atomicLoad(&sorting.digit_histogram[sorting_pass_index][tid]);
                break;
            }
            prev -= 1u;

            // Spin until prior tile publishes its local total for this digit
            var word = 0u;
            loop {
                word = atomicLoad(&status_counters[prev][tid]);
                if ((word & 0xC0000000u) != 0u) { break; }
            }
            global_digit_count += word & 0x3FFFFFFFu;
            if ((word & 0x80000000u) != 0u) { break; }
        }

        digit_global_base_ws[tid] = global_digit_count;
        storageBarrier();
        atomicStore(&status_counters[tile_index][tid], 0x80000000u | (global_digit_count + local_total));

        if (sorting_pass_index == #{RADIX_DIGIT_PLACES}u - 1u && tid == 0u) {
            draw_indirect.vertex_count = 4u;
            atomicStore(&draw_indirect.instance_count, gaussian_uniforms.count);
        }
    }
    workgroupBarrier();

    // Write keys to global memory at final stable positions (stable within tile)
    for (var i = 0u; i < #{ENTRIES_PER_INVOCATION_C}u; i += 1u) {
        let k = keys[i];
        let d = tile_entries[i * #{WORKGROUP_INVOCATIONS_C}u + tid];
        if (d == INVALID_DIGIT) { continue; }
        // Count threads before me in this iteration with the same digit
        var thread_prefix = 0u;
        let base_index = i * #{WORKGROUP_INVOCATIONS_C}u;
        for (var t = 0u; t < tid; t += 1u) {
            if (tile_entries[base_index + t] == d) { thread_prefix += 1u; }
        }
        let pos_in_tile_for_digit = iter_prefix_ws[d * #{ENTRIES_PER_INVOCATION_C}u + i] + thread_prefix;
        let dst = digit_global_base_ws[d] + pos_in_tile_for_digit;
        if (dst < gaussian_uniforms.count) {
            output_entries[dst].key = k;
        }
    }
    workgroupBarrier();

    // Write values to global memory to match keys
    for (var i = 0u; i < #{ENTRIES_PER_INVOCATION_C}u; i += 1u) {
        let v = values[i];
        let d = tile_entries[i * #{WORKGROUP_INVOCATIONS_C}u + tid];
        if (d == INVALID_DIGIT) { continue; }
        var thread_prefix = 0u;
        let base_index = i * #{WORKGROUP_INVOCATIONS_C}u;
        for (var t = 0u; t < tid; t += 1u) {
            if (tile_entries[base_index + t] == d) { thread_prefix += 1u; }
        }
        let pos_in_tile_for_digit = iter_prefix_ws[d * #{ENTRIES_PER_INVOCATION_C}u + i] + thread_prefix;
        let dst = digit_global_base_ws[d] + pos_in_tile_for_digit;
        if (dst < gaussian_uniforms.count) {
            output_entries[dst].value = v;
        }
    }
}
