#import bevy_gaussian_splatting::ordered_types::{OrderedGaussian, OrderedEntry, OrderedConfig}

@group(0) @binding(0) var<uniform> config: OrderedConfig;
@group(0) @binding(1) var<storage, read> projected: array<OrderedGaussian>;
@group(0) @binding(2) var<storage, read_write> prefix: array<u32>;
@group(0) @binding(3) var<storage, read_write> header: array<u32>;
@group(0) @binding(4) var<storage, read_write> entries: array<OrderedEntry>;
var<workgroup> scan: array<u32, 256>;
var<workgroup> carry: u32;

fn inclusive_scan(lane: u32) {
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var previous = 0u;
        if lane >= stride { previous = scan[lane - stride]; }
        workgroupBarrier();
        scan[lane] += previous;
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn classify(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let linear_group = group.y * groups.x + group.x;
    // The final 2D row may contain padding. All lanes leave together before
    // scan barriers, and no padding group writes beyond the prefix allocation.
    if linear_group >= config.groups { return; }
    let index = linear_group * 256u + lane;
    var valid = 0u;
    if index < config.capacity { valid = u32(projected[index].cutoff_squared > 0.0); }
    scan[lane] = valid;
    inclusive_scan(lane);
    if index < config.capacity { prefix[config.groups * 2u + index] = scan[lane] - valid; }
    if lane == 255u { prefix[linear_group] = scan[lane]; }
}

@compute @workgroup_size(256)
fn scan_groups(@builtin(local_invocation_index) lane: u32) {
    if lane == 0u { carry = 0u; }
    workgroupBarrier();
    for (var base = 0u; base < config.groups; base += 256u) {
        var count = 0u;
        if base + lane < config.groups { count = prefix[base + lane]; }
        scan[lane] = count;
        inclusive_scan(lane);
        if base + lane < config.groups { prefix[config.groups + base + lane] = carry + scan[lane] - count; }
        workgroupBarrier();
        if lane == 0u { carry += scan[255]; }
        workgroupBarrier();
    }
    if lane == 0u {
        header[12] = select(0u, 1u, carry > config.capacity) | header[13];
        let count = select(carry, 0u, header[12] != 0u);
        header[0] = 4u; header[1] = count; header[2] = 0u; header[3] = 0u;
        header[4] = (count + 1023u) / 1024u; header[5] = 1u; header[6] = 1u;
        header[8] = 1u; header[9] = (count + 1023u) / 1024u; header[10] = 1u;
    }
}

@compute @workgroup_size(256)
fn gather(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let input = (group.y * groups.x + group.x) * 256u + lane;
    if header[12] != 0u || input >= config.capacity || !(projected[input].cutoff_squared > 0.0) { return; }
    let index = prefix[config.groups * 2u + input] + prefix[config.groups + input / 256u];
    entries[index] = OrderedEntry(projected[input].key, input);
}
