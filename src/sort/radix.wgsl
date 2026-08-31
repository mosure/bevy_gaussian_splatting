#import bevy_gaussian_splatting::bindings::{
    view,
    gaussian_uniforms,
    sorting_pass_index,
    sorting,
    status_counters,
    draw_indirect,
    input_entries,
    output_entries,
    Entry,
}
#ifdef PACKED_F32
#import bevy_gaussian_splatting::packed::{get_position, get_visibility}
#else

#ifdef BUFFER_STORAGE
#import bevy_gaussian_splatting::planar::{get_position, get_visibility}
#endif

#endif

#ifdef BUFFER_TEXTURE
#import bevy_gaussian_splatting::texture::{get_position, get_visibility}
#endif

struct SortingGlobal {
    digit_histogram: array<array<atomic<u32>, #{RADIX_BASE}>, #{RADIX_DIGIT_PLACES}>,
}

@group(3) @binding(0) var<uniform> sorting_pass_index: u32;
@group(3) @binding(1) var<storage, read_write> sorting: SortingGlobal;
// Per-tile temporary storage for radix pass C.
@group(3) @binding(2) var<storage, read_write> status_counters: array<array<atomic<u32>, #{RADIX_BASE}>>;
// Layout-compatible read-only view of the draw/dispatch record. The compactor
// owns all writes. Radix reads the exact instance count while the same buffer
// is also consumed as an indirect-dispatch source, which requires an inclusive
// read-only storage usage rather than exclusive read-write storage.
struct RadixDrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_vertex: u32,
    base_instance: u32,
}
@group(3) @binding(3) var<storage, read> draw_indirect: RadixDrawIndirect;
@group(3) @binding(4) var<storage, read_write> input_entries: array<Entry>;
@group(3) @binding(5) var<storage, read_write> output_entries: array<Entry>;

fn radix_entry_count() -> u32 {
    return draw_indirect.instance_count;
}

// All portable radix entry points stay at or below 256 invocations. Pass A
// aggregates into a workgroup histogram before touching global atomics.
var<workgroup> workgroup_digit_histogram: array<array<atomic<u32>, #{RADIX_BASE}>, #{RADIX_DIGIT_PLACES}>;
var<workgroup> radix_scan_values: array<u32, #{RADIX_BASE}>;
var<workgroup> radix_scan_total: u32;

// Work-efficient exclusive Blelloch scan for one RADIX_BASE-wide workgroup.
// RADIX_BASE is 256, so every entry point remains WebGPU-portable while the
// total scan work is O(n), rather than O(n log n) iterative-doubling work.
fn exclusive_radix_scan(local_index: u32) {
    workgroupBarrier();
    var stride = 1u;
    while stride < #{RADIX_BASE}u {
        let index = (local_index + 1u) * stride * 2u - 1u;
        if index < #{RADIX_BASE}u {
            radix_scan_values[index] += radix_scan_values[index - stride];
        }
        workgroupBarrier();
        stride *= 2u;
    }

    if local_index == 0u {
        radix_scan_total = radix_scan_values[#{RADIX_BASE}u - 1u];
        radix_scan_values[#{RADIX_BASE}u - 1u] = 0u;
    }
    workgroupBarrier();

    stride = #{RADIX_BASE}u / 2u;
    loop {
        let index = (local_index + 1u) * stride * 2u - 1u;
        if index < #{RADIX_BASE}u {
            let left_index = index - stride;
            let left_value = radix_scan_values[left_index];
            let parent_value = radix_scan_values[index];
            radix_scan_values[left_index] = parent_value;
            radix_scan_values[index] = parent_value + left_value;
        }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
}

@compute @workgroup_size(#{RADIX_BASE})
fn radix_reset(@builtin(local_invocation_id) local_id: vec3<u32>) {
    for (var digit_pass = 0u; digit_pass < #{RADIX_DIGIT_PLACES}u; digit_pass += 1u) {
        atomicStore(&sorting.digit_histogram[digit_pass][local_id.x], 0u);
    }
}

@compute @workgroup_size(#{RADIX_BASE})
fn radix_sort_a(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    for (var digit_pass = 0u; digit_pass < #{RADIX_DIGIT_PLACES}u; digit_pass += 1u) {
        atomicStore(&workgroup_digit_histogram[digit_pass][local_id.x], 0u);
    }
    workgroupBarrier();
    let thread_index = global_id.x;
    let start_entry_index = thread_index * #{ENTRIES_PER_INVOCATION_A}u;
    let end_entry_index = start_entry_index + #{ENTRIES_PER_INVOCATION_A}u;

    for (var entry_index = start_entry_index; entry_index < end_entry_index; entry_index += 1u) {
        if (entry_index >= radix_entry_count()) { continue; }
        var key: u32 = 0xFFFFFFFFu;
        let position = vec4<f32>(get_position(entry_index), 1.0);
        let transformed_position = (gaussian_uniforms.transform * position).xyz;
        let diff = transformed_position - view.world_position;
        let dist2 = dot(diff, diff);
        let dist_bits = bitcast<u32>(dist2);
        let key_distance = 0xFFFFFFFFu - dist_bits;
        // Rotation-stable global pre-sort only: per-frame support-frustum
        // culling belongs to the raster vertex stage. This is not per-pixel
        // StopThePop ordering.
        if (get_visibility(entry_index) > 0.0) {
            key = key_distance;
        }
        key = key >> #{RADIX_KEY_SHIFT}u;
        input_entries[entry_index].key = key;
        input_entries[entry_index].value = entry_index;
        for(var shift = 0u; shift < #{RADIX_DIGIT_PLACES}u; shift += 1u) {
            let digit = (key >> (shift * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
            atomicAdd(&workgroup_digit_histogram[shift][digit], 1u);
        }
    }
    workgroupBarrier();
    for (var digit_pass = 0u; digit_pass < #{RADIX_DIGIT_PLACES}u; digit_pass += 1u) {
        let count = atomicLoad(&workgroup_digit_histogram[digit_pass][local_id.x]);
        if count != 0u {
            atomicAdd(&sorting.digit_histogram[digit_pass][local_id.x], count);
        }
    }
}

// LoD compaction has already generated keys and a dense active list. This pass
// only builds digit histograms and is dispatched from the exact active count.
@compute @workgroup_size(#{RADIX_BASE})
fn radix_sort_active_a(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    for (var digit_pass = 0u; digit_pass < #{RADIX_DIGIT_PLACES}u; digit_pass += 1u) {
        atomicStore(&workgroup_digit_histogram[digit_pass][local_id.x], 0u);
    }
    workgroupBarrier();
    let thread_index = global_id.x;
    let start_entry_index = thread_index * #{ENTRIES_PER_INVOCATION_A}u;
    let end_entry_index = start_entry_index + #{ENTRIES_PER_INVOCATION_A}u;
    let count = radix_entry_count();
    for (var entry_index = start_entry_index; entry_index < end_entry_index; entry_index += 1u) {
        if (entry_index >= count) { continue; }
        let key = input_entries[entry_index].key;
        for (var shift = 0u; shift < #{RADIX_DIGIT_PLACES}u; shift += 1u) {
            let digit = (key >> (shift * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
            atomicAdd(&workgroup_digit_histogram[shift][digit], 1u);
        }
    }
    workgroupBarrier();
    for (var digit_pass = 0u; digit_pass < #{RADIX_DIGIT_PLACES}u; digit_pass += 1u) {
        let histogram_count = atomicLoad(&workgroup_digit_histogram[digit_pass][local_id.x]);
        if histogram_count != 0u {
            atomicAdd(&sorting.digit_histogram[digit_pass][local_id.x], histogram_count);
        }
    }
}

@compute @workgroup_size(#{RADIX_BASE})
fn radix_sort_b(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let digit = local_id.x;
    let digit_pass = workgroup_id.y;
    let count = atomicLoad(&sorting.digit_histogram[digit_pass][digit]);
    radix_scan_values[digit] = count;
    exclusive_radix_scan(digit);
    atomicStore(&sorting.digit_histogram[digit_pass][digit], radix_scan_values[digit]);
}

// --- SHARED MEMORY for radix pass C ---
var<workgroup> tile_input_entries: array<Entry, #{WORKGROUP_ENTRIES_C}>;
var<workgroup> sorted_tile_entries: array<Entry, #{WORKGROUP_ENTRIES_C}>;
var<workgroup> tile_digit_counts: array<atomic<u32>, #{RADIX_BASE}>;
var<workgroup> local_digit_offsets: array<u32, #{RADIX_BASE}>;
var<workgroup> tile_entry_count_ws: u32;
var<workgroup> tile_prefix_values: array<u32, #{WORKGROUP_ENTRIES_C}>;
var<workgroup> tile_scan_carry: u32;
var<workgroup> tile_scan_total: u32;
const INVALID_KEY: u32 = 0xFFFFFFFFu;

// Work-efficient exclusive Blelloch scan over a 1024-entry tile using 256
// lanes. Each lane owns at most four tree nodes at the lowest level; the total
// number of additions remains linear in WORKGROUP_ENTRIES_C.
fn exclusive_tile_scan(local_index: u32) {
    let tile_size = #{WORKGROUP_ENTRIES_C}u;
    let threads = #{WORKGROUP_INVOCATIONS_C}u;
    workgroupBarrier();
    var stride = 1u;
    while stride < tile_size {
        let span = stride * 2u;
        for (
            var index = (local_index + 1u) * span - 1u;
            index < tile_size;
            index += threads * span
        ) {
            tile_prefix_values[index] += tile_prefix_values[index - stride];
        }
        workgroupBarrier();
        stride = span;
    }

    if local_index == 0u {
        tile_scan_total = tile_prefix_values[tile_size - 1u];
        tile_prefix_values[tile_size - 1u] = 0u;
    }
    workgroupBarrier();

    stride = tile_size / 2u;
    loop {
        let span = stride * 2u;
        for (
            var index = (local_index + 1u) * span - 1u;
            index < tile_size;
            index += threads * span
        ) {
            let left_index = index - stride;
            let left_value = tile_prefix_values[left_index];
            let parent_value = tile_prefix_values[index];
            tile_prefix_values[left_index] = parent_value;
            tile_prefix_values[index] = parent_value + left_value;
        }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
}

@compute @workgroup_size(#{WORKGROUP_INVOCATIONS_C})
fn radix_sort_c_count_tiles(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let tile_size = #{WORKGROUP_ENTRIES_C}u;
    let threads = #{WORKGROUP_INVOCATIONS_C}u;
    let global_entry_offset = workgroup_id.y * tile_size;

    if (tid < #{RADIX_BASE}u) {
        atomicStore(&tile_digit_counts[tid], 0u);
    }
    workgroupBarrier();

    for (var i = tid; i < tile_size; i += threads) {
        let idx = global_entry_offset + i;
        if (idx >= radix_entry_count()) {
            continue;
        }

        let entry = input_entries[idx];
        let digit = (entry.key >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
        atomicAdd(&tile_digit_counts[digit], 1u);
    }
    workgroupBarrier();

    if (tid < #{RADIX_BASE}u) {
        let count = atomicLoad(&tile_digit_counts[tid]);
        atomicStore(&status_counters[workgroup_id.y][tid], count);
    }
}

// One workgroup of RADIX_BASE lanes, lane = digit. Each digit's tile-prefix is
// independent (no cross-lane data), so this packs 256 lanes into ~4 full waves instead
// of dispatching 256 single-lane workgroups (1/64 wave occupancy on RDNA).
@compute @workgroup_size(#{RADIX_BASE})
fn radix_sort_c_scan_tiles(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let lane = local_id.x;
    let digit = workgroup_id.y;
    let tile_size = #{WORKGROUP_ENTRIES_C}u;
    let tile_count = (radix_entry_count() + tile_size - 1u) / tile_size;
    if lane == 0u {
        tile_scan_carry = atomicLoad(&sorting.digit_histogram[sorting_pass_index][digit]);
    }
    workgroupBarrier();
    for (var base = 0u; base < tile_count; base += #{RADIX_BASE}u) {
        let tile = base + lane;
        var count = 0u;
        if tile < tile_count {
            count = atomicLoad(&status_counters[tile][digit]);
        }
        radix_scan_values[lane] = count;
        exclusive_radix_scan(lane);
        let block_base = tile_scan_carry;
        if tile < tile_count {
            atomicStore(
                &status_counters[tile][digit],
                block_base + radix_scan_values[lane],
            );
        }
        workgroupBarrier();
        if lane == 0u {
            tile_scan_carry = block_base + radix_scan_total;
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(#{WORKGROUP_INVOCATIONS_C})
fn radix_sort_c_scatter(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let tile_size = #{WORKGROUP_ENTRIES_C}u;
    let threads = #{WORKGROUP_INVOCATIONS_C}u;
    let global_entry_offset = workgroup_id.y * tile_size;

    // Step 1: Parallel load.
    for (var i = tid; i < tile_size; i += threads) {
        let idx = global_entry_offset + i;
        if (idx < radix_entry_count()) {
            tile_input_entries[i] = input_entries[idx];
        } else {
            tile_input_entries[i] = Entry(INVALID_KEY, INVALID_KEY);
        }
    }
    workgroupBarrier();

    if tid == 0u {
        tile_entry_count_ws = min(
            tile_size,
            radix_entry_count() - min(global_entry_offset, radix_entry_count()),
        );
    }
    workgroupBarrier();

    // Eight stable binary partitions replace the serial tile placement. The
    // fallback uses portable workgroup scans and exactly preserves LSD order.
    for (var bit = 0u; bit < #{RADIX_BITS_PER_DIGIT}u; bit += 1u) {
        for (var i = tid; i < tile_size; i += threads) {
            // WGSL `select` is restricted to scalar/vector values, so keep the
            // ping-pong source choice explicit for the Entry structure.
            var entry = tile_input_entries[i];
            if (bit & 1u) != 0u {
                entry = sorted_tile_entries[i];
            }
            let digit = (entry.key >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u))
                & (#{RADIX_BASE}u - 1u);
            tile_prefix_values[i] = select(
                0u,
                1u,
                i < tile_entry_count_ws && ((digit >> bit) & 1u) == 0u,
            );
        }
        exclusive_tile_scan(tid);
        let zero_count = tile_scan_total;
        for (var i = tid; i < tile_entry_count_ws; i += threads) {
            var entry = tile_input_entries[i];
            if (bit & 1u) != 0u {
                entry = sorted_tile_entries[i];
            }
            let digit = (entry.key >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u))
                & (#{RADIX_BASE}u - 1u);
            let zeros_before = tile_prefix_values[i];
            let is_zero = ((digit >> bit) & 1u) == 0u;
            let destination = select(
                zero_count + i - zeros_before,
                zeros_before,
                is_zero,
            );
            if (bit & 1u) == 0u {
                sorted_tile_entries[destination] = entry;
            } else {
                tile_input_entries[destination] = entry;
            }
        }
        workgroupBarrier();
    }

    // Parallel local histogram + scan provides each digit's tile-local base.
    atomicStore(&tile_digit_counts[tid], 0u);
    workgroupBarrier();
    for (var i = tid; i < tile_entry_count_ws; i += threads) {
        let entry = tile_input_entries[i];
        let digit = (entry.key >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u))
            & (#{RADIX_BASE}u - 1u);
        atomicAdd(&tile_digit_counts[digit], 1u);
    }
    workgroupBarrier();
    let digit_count = atomicLoad(&tile_digit_counts[tid]);
    radix_scan_values[tid] = digit_count;
    exclusive_radix_scan(tid);
    local_digit_offsets[tid] = radix_scan_values[tid];
    workgroupBarrier();

    for (var i = tid; i < tile_size; i += threads) {
        if (i < tile_entry_count_ws) {
            let entry = tile_input_entries[i];
            let digit = (entry.key >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);

            let bin_start_offset = local_digit_offsets[digit];
            let rank_in_bin = i - bin_start_offset;
            let global_base = atomicLoad(&status_counters[workgroup_id.y][digit]);
            let dst = global_base + rank_in_bin;

            if (dst < radix_entry_count()) {
                output_entries[dst] = entry;
            }
        }
    }
}
