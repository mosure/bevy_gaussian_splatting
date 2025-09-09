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

        let distance = distance(transformed_position, view.world_position);
        let distance_wide = 0x0FFFFFFF - u32(distance * 1.0e4);

        // TODO: use 4d transformed position, from gaussian node

        if (in_frustum(clip_space_pos.xyz)) {
            // key = bitcast<u32>(1.0 - clip_space_pos.z);
            // key = u32(clip_space_pos.z * 0xFFFF.0) << 16u;
            // key |= u32((clip_space_pos.x * 0.5 + 0.5) * 0xFF.0) << 8u;
            // key |= u32((clip_space_pos.y * 0.5 + 0.5) * 0xFF.0);
            key = distance_wide;
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
    var sum = 0u;
    for(var digit = 0u; digit < #{RADIX_BASE}u; digit += 1u) {
        let tmp = sorting.digit_histogram[gl_GlobalInvocationID.y][digit];
        sorting.digit_histogram[gl_GlobalInvocationID.y][digit] = sum;
        sum += tmp;
    }
}

struct SortingSharedC {
    entries: array<atomic<u32>, #{WORKGROUP_ENTRIES_C}>,
    gather_sources: array<atomic<u32>, #{WORKGROUP_ENTRIES_C}>,
    scan: array<atomic<u32>, #{WORKGROUP_INVOCATIONS_C}>,
    total: u32,
}
var<workgroup> sorting_shared_c: SortingSharedC;

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
) {
    // Draw an assignment number
    if(gl_LocalInvocationID.x == 0u) {
        sorting_shared_c.entries[0] = atomicAdd(&sorting.assignment_counter, 1u);
    }

    // Reset histogram
    sorting_shared_c.scan[gl_LocalInvocationID.x + conflict_free_offset(gl_LocalInvocationID.x)] = 0u;
    workgroupBarrier();

    let assignment = sorting_shared_c.entries[0];
    let global_entry_offset = assignment * #{WORKGROUP_ENTRIES_C}u;
    // TODO: Specialize end shader
    if(gl_LocalInvocationID.x == 0u && assignment * #{WORKGROUP_ENTRIES_C}u + #{WORKGROUP_ENTRIES_C}u >= gaussian_uniforms.count) {
        // Last workgroup resets the assignment number for the next pass
        atomicStore(&sorting.assignment_counter, 0u);
    }

    // Load keys from global memory into registers and rank them
    var keys: array<u32, #{ENTRIES_PER_INVOCATION_C}>;
    var ranks: array<u32, #{ENTRIES_PER_INVOCATION_C}>;
    for(var entry_index = 0u; entry_index < #{ENTRIES_PER_INVOCATION_C}u; entry_index += 1u) {
        keys[entry_index] = input_entries[global_entry_offset + #{WORKGROUP_INVOCATIONS_C}u * entry_index + gl_LocalInvocationID.x].key;
        let digit = (keys[entry_index] >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
        // TODO: Implement warp-level multi-split (WLMS) once WebGPU supports subgroup operations
        ranks[entry_index] = atomicAdd(&sorting_shared_c.scan[digit + conflict_free_offset(digit)], 1u);
    }
    workgroupBarrier();

    // Cumulate histogram
    let local_digit_count = sorting_shared_c.scan[gl_LocalInvocationID.x + conflict_free_offset(gl_LocalInvocationID.x)];
    let local_digit_offset = exclusive_scan(gl_LocalInvocationID.x, local_digit_count);
    sorting_shared_c.scan[gl_LocalInvocationID.x + conflict_free_offset(gl_LocalInvocationID.x)] = local_digit_offset;

    // Chained decoupling lookback
    atomicStore(&status_counters[assignment][gl_LocalInvocationID.x], 0x40000000u | local_digit_count);
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
            status_counter = atomicLoad(&status_counters[previous_tile][gl_LocalInvocationID.x]);
        }
        global_digit_count += status_counter & 0x3FFFFFFFu;
        if((status_counter & 0x80000000u) != 0u) {
            break;
        }
    }
    atomicStore(&status_counters[assignment][gl_LocalInvocationID.x], 0x80000000u | (global_digit_count + local_digit_count));
    // On the final digit pass, set indirect draw counts once (robust gate)
    if (sorting_pass_index == #{RADIX_DIGIT_PLACES}u - 1u && gl_LocalInvocationID.x == 0u) {
        draw_indirect.vertex_count = 4u;
        // Use the total gaussian count to avoid edge-dependent undercounting
        atomicStore(&draw_indirect.instance_count, gaussian_uniforms.count);
    }

    // Scatter keys inside shared memory
    for(var entry_index = 0u; entry_index < #{ENTRIES_PER_INVOCATION_C}u; entry_index += 1u) {
        let key = keys[entry_index];
        let digit = (key >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
        ranks[entry_index] += sorting_shared_c.scan[digit + conflict_free_offset(digit)];
        sorting_shared_c.entries[ranks[entry_index]] = key;
    }
    workgroupBarrier();

    // Add global offset
    sorting_shared_c.scan[gl_LocalInvocationID.x + conflict_free_offset(gl_LocalInvocationID.x)] = global_digit_count - local_digit_offset;
    workgroupBarrier();

    // Store keys from shared memory into global memory
    for(var entry_index = 0u; entry_index < #{ENTRIES_PER_INVOCATION_C}u; entry_index += 1u) {
        let key = sorting_shared_c.entries[#{WORKGROUP_INVOCATIONS_C}u * entry_index + gl_LocalInvocationID.x];
        let digit = (key >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
        keys[entry_index] = digit;
        output_entries[sorting_shared_c.scan[digit + conflict_free_offset(digit)] + #{WORKGROUP_INVOCATIONS_C}u * entry_index + gl_LocalInvocationID.x].key = key;
    }
    workgroupBarrier();

    // Load values from global memory and scatter them inside shared memory
    for(var entry_index = 0u; entry_index < #{ENTRIES_PER_INVOCATION_C}u; entry_index += 1u) {
        let value = input_entries[global_entry_offset + #{WORKGROUP_INVOCATIONS_C}u * entry_index + gl_LocalInvocationID.x].value;
        sorting_shared_c.entries[ranks[entry_index]] = value;
    }
    workgroupBarrier();

    // Store values from shared memory into global memory
    for(var entry_index = 0u; entry_index < #{ENTRIES_PER_INVOCATION_C}u; entry_index += 1u) {
        let value = sorting_shared_c.entries[#{WORKGROUP_INVOCATIONS_C}u * entry_index + gl_LocalInvocationID.x];
        let digit = keys[entry_index];
    output_entries[sorting_shared_c.scan[digit + conflict_free_offset(digit)] + #{WORKGROUP_INVOCATIONS_C}u * entry_index + gl_LocalInvocationID.x].value = value;
    }
}
