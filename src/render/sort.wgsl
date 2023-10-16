#define_import_path bevy_gaussian_splatting::sort


@group(0) @binding(0) var<storage, read> points: array<GaussianInput>;

@group(0) @binding(1) var<storage, write> sorted_points: array<GaussianInput>;


fn sort() {
    let num_points = points.length();

    for (let i = 0; i < num_points; i++) {
        sorted_points[i] = points[i];
    }

    for (let i = 0; i < num_points; i++) {
        let min_index = i;
        let min_value = sorted_points[i].position.z;

        for (let j = i + 1; j < num_points; j++) {
            if (sorted_points[j].position.z < min_value) {
                min_index = j;
                min_value = sorted_points[j].position.z;
            }
        }

        let temp = sorted_points[i];
        sorted_points[i] = sorted_points[min_index];
        sorted_points[min_index] = temp;
    }

    workgroupBarrier();
}




// Onesweep Radix Sort

struct SortingSharedA {
    digit_histogram: array<array<atomic<u32>, RADIX_BASE>, RADIX_DIGIT_PLACES>,
}
var<workgroup> sorting_shared_a: SortingSharedA;

@compute @workgroup_size(RADIX_BASE, RADIX_DIGIT_PLACES)
fn radixSortA(
    @builtin(local_invocation_id) gl_LocalInvocationID: vec3<u32>,
    @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
) {
    sorting_shared_a.digit_histogram[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0u;
    workgroupBarrier();

    let thread_index = gl_GlobalInvocationID.x * RADIX_DIGIT_PLACES + gl_GlobalInvocationID.y;
    let start_entry_index = thread_index * ENTRIES_PER_INVOCATION_A;
    let end_entry_index = start_entry_index + ENTRIES_PER_INVOCATION_A;
    for(var entry_index = start_entry_index; entry_index < end_entry_index; entry_index += 1u) {
        if(entry_index >= arrayLength(&splats)) {
            continue;
        }
        var key: u32 = 0xFFFFFFFFu; // Stream compaction for frustum culling
        let clip_space_pos = worldToClipSpace(splats[entry_index].center);
        if(isInFrustum(clip_space_pos.xyz)) {
            // key = bitcast<u32>(clip_space_pos.z);
            key = u32(clip_space_pos.z * 0xFFFF.0) << 16u;
            key |= u32((clip_space_pos.x * 0.5 + 0.5) * 0xFF.0) << 8u;
            key |= u32((clip_space_pos.y * 0.5 + 0.5) * 0xFF.0);
        }
        output_entries[entry_index].key = key;
        output_entries[entry_index].value = entry_index;
        for(var shift = 0u; shift < RADIX_DIGIT_PLACES; shift += 1u) {
            let digit = (key >> (shift * RADIX_BITS_PER_DIGIT)) & (RADIX_BASE - 1u);
            atomicAdd(&sorting_shared_a.digit_histogram[shift][digit], 1u);
        }
    }
    workgroupBarrier();

    atomicAdd(&sorting.digit_histogram[gl_LocalInvocationID.y][gl_LocalInvocationID.x], sorting_shared_a.digit_histogram[gl_LocalInvocationID.y][gl_LocalInvocationID.x]);
}

@compute @workgroup_size(1)
fn radixSortB(
    @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
) {
    var sum = 0u;
    for(var digit = 0u; digit < RADIX_BASE; digit += 1u) {
        let tmp = sorting.digit_histogram[gl_GlobalInvocationID.y][digit];
        sorting.digit_histogram[gl_GlobalInvocationID.y][digit] = sum;
        sum += tmp;
    }
}

struct SortingSharedC {
    entries: array<atomic<u32>, WORKGROUP_ENTRIES_C>,
    gather_sources: array<atomic<u32>, WORKGROUP_ENTRIES_C>,
    scan: array<atomic<u32>, WORKGROUP_INVOCATIONS_C>,
    total: u32,
}
var<workgroup> sorting_shared_c: SortingSharedC;

const NUM_BANKS: u32 = 16u;
const LOG_NUM_BANKS: u32 = 4u;
fn conflicFreeOffset(n: u32) -> u32 {
    return 0u; // n >> NUM_BANKS + n >> (2u * LOG_NUM_BANKS);
}

fn exclusiveScan(gl_LocalInvocationID: vec3<u32>) -> u32 {
    var offset = 1u;
    for(var d = WORKGROUP_INVOCATIONS_C >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if(gl_LocalInvocationID.x < d) {
            var ai = offset * (2u * gl_LocalInvocationID.x + 1u) - 1u;
            var bi = offset * (2u * gl_LocalInvocationID.x + 2u) - 1u;
            ai += conflicFreeOffset(ai);
            bi += conflicFreeOffset(bi);
            sorting_shared_c.scan[bi] += sorting_shared_c.scan[ai];
        }
        offset <<= 1u;
    }
    if(gl_LocalInvocationID.x == 0u) {
      var i = WORKGROUP_INVOCATIONS_C - 1u;
      i += conflicFreeOffset(i);
      sorting_shared_c.total = sorting_shared_c.scan[i];
      sorting_shared_c.scan[i] = 0u;
    }
    for(var d = 1u; d < WORKGROUP_INVOCATIONS_C; d <<= 1u) {
        workgroupBarrier();
        offset >>= 1u;
        if(gl_LocalInvocationID.x < d) {
            var ai = offset * (2u * gl_LocalInvocationID.x + 1u) - 1u;
            var bi = offset * (2u * gl_LocalInvocationID.x + 2u) - 1u;
            ai += conflicFreeOffset(ai);
            bi += conflicFreeOffset(bi);
            let t = sorting_shared_c.scan[ai];
            sorting_shared_c.scan[ai] = sorting_shared_c.scan[bi];
            sorting_shared_c.scan[bi] += t;
        }
    }
    workgroupBarrier();
    return sorting_shared_c.total;
}

@compute @workgroup_size(WORKGROUP_INVOCATIONS_C)
fn radixSortC(
    @builtin(local_invocation_id) gl_LocalInvocationID: vec3<u32>,
    @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
) {
    // Draw an assignment number
    if(gl_LocalInvocationID.x == 0u) {
        sorting_shared_c.entries[0] = atomicAdd(&sorting.assignment_counter, 1u);
    }
    workgroupBarrier();

    let assignment = sorting_shared_c.entries[0];
    var scatter_targets: array<u32, ENTRIES_PER_INVOCATION_C>;
    var gather_sources: array<u32, ENTRIES_PER_INVOCATION_C>;
    let local_entry_offset = gl_LocalInvocationID.x * ENTRIES_PER_INVOCATION_C;
    let global_entry_offset = assignment * WORKGROUP_ENTRIES_C + local_entry_offset;
    /* TODO: Specialize end shader
    let end_entry_index = ENTRIES_PER_INVOCATION_C;
    if(global_entry_offset + end_entry_index > arrayLength(&splats)) {
        if(arrayLength(&splats) <= global_entry_offset) {
            end_entry_index = 0u;
        } else {
            end_entry_index = arrayLength(&splats) - global_entry_offset;
        }
    }*/
    if(gl_LocalInvocationID.x == 0u && global_entry_offset + WORKGROUP_ENTRIES_C >= arrayLength(&splats)) {
        // Last workgroup resets the assignment number for the next pass
        sorting.assignment_counter = 0u;
    }

    for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
        // Load keys from global memory into shared memory
        let key = input_entries[global_entry_offset + entry_index][0];
        sorting_shared_c.entries[local_entry_offset + entry_index] = key;
        // Extract digit from key and initialize gather_sources
        let digit = (key >> (sorting_pass_index * RADIX_BITS_PER_DIGIT)) & (RADIX_BASE - 1u);
        gather_sources[entry_index] = (digit << 16u) | (local_entry_offset + entry_index);
    }

    // Workgroup wide ranking
    // Warp-level multi-split (WLMS) can not be implemented,
    // because there is no subgroup ballot support in WebGPU yet: https://github.com/gpuweb/gpuweb/issues/3950
    // Alternative: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    for(var bit_shift = 0u; bit_shift < RADIX_BITS_PER_DIGIT; bit_shift += 1u) {
        var rank = 0u;
        for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
            let bit = (gather_sources[entry_index] >> (16u + bit_shift)) & 1u;
            scatter_targets[entry_index] = rank;
            rank += 1u - bit;
        }
        sorting_shared_c.scan[gl_LocalInvocationID.x + conflicFreeOffset(gl_LocalInvocationID.x)] = rank;
        let total = exclusiveScan(gl_LocalInvocationID);
        rank = sorting_shared_c.scan[gl_LocalInvocationID.x + conflicFreeOffset(gl_LocalInvocationID.x)];
        for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
            scatter_targets[entry_index] += rank;
            let bit = (gather_sources[entry_index] >> (16u + bit_shift)) & 1u;
            if(bit == 1u) {
                scatter_targets[entry_index] = local_entry_offset + entry_index - scatter_targets[entry_index] + total;
            }
        }

        // Scatter the gather_sources
        for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
            sorting_shared_c.gather_sources[scatter_targets[entry_index]] = gather_sources[entry_index];
        }
        workgroupBarrier();
        for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
            gather_sources[entry_index] = sorting_shared_c.gather_sources[local_entry_offset + entry_index];
        }
    }

    // Reset histogram
    sorting_shared_c.scan[gl_LocalInvocationID.x + conflicFreeOffset(gl_LocalInvocationID.x)] = 0u;
    workgroupBarrier();

    // Build tile histogram in shared memory
    for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
        let digit = gather_sources[entry_index] >> 16u;
        atomicAdd(&sorting_shared_c.scan[digit + conflicFreeOffset(digit)], 1u);
    }
    workgroupBarrier();

    // Store histogram in global table
    var local_digit_count = sorting_shared_c.scan[gl_LocalInvocationID.x + conflicFreeOffset(gl_LocalInvocationID.x)];
    atomicStore(&sorting.status_counters[assignment][gl_LocalInvocationID.x], 0x40000000u | local_digit_count);

    // Chained decoupling lookback
    var global_digit_count = 0u;
    var previous_tile = assignment;
    while true {
        if(previous_tile == 0u) {
            global_digit_count += sorting.digit_histogram[sorting_pass_index][gl_LocalInvocationID.x];
            break;
        }
        previous_tile -= 1u;
        var status_counter = 0u;
        while((status_counter & 0xC0000000u) == 0u) {
            status_counter = atomicLoad(&sorting.status_counters[previous_tile][gl_LocalInvocationID.x]);
        }
        global_digit_count += status_counter & 0x3FFFFFFFu;
        if((status_counter & 0x80000000u) != 0u) {
            break;
        }
    }
    atomicStore(&sorting.status_counters[assignment][gl_LocalInvocationID.x], 0x80000000u | (global_digit_count + local_digit_count));
    if(sorting_pass_index == RADIX_DIGIT_PLACES - 1u && gl_LocalInvocationID.x == WORKGROUP_INVOCATIONS_C - 2u && global_entry_offset + WORKGROUP_ENTRIES_C >= arrayLength(&splats)) {
        sorting.draw_indirect.vertex_count = 4u;
        sorting.draw_indirect.instance_count = global_digit_count + local_digit_count;
    }
    exclusiveScan(gl_LocalInvocationID);
    sorting_shared_c.scan[gl_LocalInvocationID.x + conflicFreeOffset(gl_LocalInvocationID.x)] = global_digit_count - sorting_shared_c.scan[gl_LocalInvocationID.x + conflicFreeOffset(gl_LocalInvocationID.x)];
    workgroupBarrier();

    // Store keys from shared memory into global memory
    for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
        let digit = gather_sources[entry_index] >> 16u;
        output_entries[sorting_shared_c.scan[digit + conflicFreeOffset(digit)] + local_entry_offset + entry_index][0] = sorting_shared_c.entries[gather_sources[entry_index] & 0xFFFFu];
    }
    workgroupBarrier();

    // Load values from global memory into shared memory
    for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
        sorting_shared_c.entries[local_entry_offset + entry_index] = input_entries[global_entry_offset + entry_index][1];
    }
    workgroupBarrier();

    // Store values from shared memory into global memory
    for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
        let digit = gather_sources[entry_index] >> 16u;
        output_entries[sorting_shared_c.scan[digit + conflicFreeOffset(digit)] + local_entry_offset + entry_index][1] = sorting_shared_c.entries[gather_sources[entry_index] & 0xFFFFu];
    }
}

