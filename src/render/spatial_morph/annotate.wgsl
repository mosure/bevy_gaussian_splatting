#import bevy_gaussian_splatting::lod_spatial_metrics::{lod_projection_metric, lod_support_outside_view}

// The traversal's 352-byte view ABI; field order is shared with its Config.
struct Config {
    world_from_local: mat4x4<f32>, clip_from_world: mat4x4<f32>,
    view: vec4<f32>, quality: vec4<f32>, counts: vec4<u32>, limits: vec4<u32>,
    offsets: vec4<u32>, feedback_offsets: vec4<u32>, spatial: vec4<u32>, omission: vec4<f32>,
    frustum: array<vec4<f32>, 6>,
}
struct Limits {
    capacity: u32, frontier: u32, selected_nodes: u32, selected_pages: u32,
    edges: u32, records: u32, padding: vec2<u32>,
}
struct Node {
    center_radius: vec4<f32>, half_extents: vec4<f32>, error_quality: vec4<f32>,
    topology: vec4<u32>, counts: vec4<u32>,
}
struct Page { start: u32, count: u32 }
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<uniform> limits: Limits;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> pages: array<Page>;
@group(0) @binding(4) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> mapping: array<u32>;
@group(0) @binding(7) var<storage, read_write> scratch: array<vec2<u32>>;

const NO_PARENT: u32 = 0xffffffffu;
const INVALID_PRESSURE: u32 = 2u;
const NEAR_BYPASS: u32 = 4u;
const MISSING_PARENT: u32 = 8u;
const BAND_UNAVAILABLE: u32 = 16u;
const SELECTION_UNAVAILABLE: u32 = 32u;
var<workgroup> scan: array<vec2<u32>, 256>;

fn tail() -> u32 { return 2u * limits.capacity; }
fn descriptor(range: u32) -> u32 { return tail() + 8u + range * 8u; }
fn range_count() -> u32 { return atomicLoad(&state[5]); }
fn range_node(range: u32) -> u32 { return atomicLoad(&state[16u + config.offsets.z + range * 4u]); }
fn range_start(range: u32) -> u32 { return atomicLoad(&state[16u + config.offsets.z + range * 4u + 1u]); }
fn map_node(index: u32) -> vec4<u32> {
    let base = mapping[1] + 4u * index;
    return vec4<u32>(mapping[base], mapping[base + 1u], mapping[base + 2u], mapping[base + 3u]);
}
fn selected(index: u32) -> bool {
    return (atomicLoad(&state[16u + limits.selected_nodes + index / 32u]) & (1u << (index & 31u))) != 0u;
}
fn metric(node: Node) -> vec4<f32> {
    return lod_projection_metric(node.center_radius, node.half_extents, node.error_quality,
        node.counts.z, config.world_from_local, config.clip_from_world, config.view, config.quality);
}

// Endpoint boxes alone do not enclose convex covariance at interpolated means.
// The mapping records 1 + 3/support_sigma: a box half extent for the mean plus
// the directional covariance radius. Affine plane transformation covers shear.
// Return one outside the approach band, smoothly falling to zero at the
// conservative unsafe boundary. A negative value denotes invalid arithmetic.
// The renderer admits this mapping only for abs(global_scale)<=1, so the
// authored support envelope also bounds its scaled covariance.
fn envelope_near_factor(node: Node) -> f32 {
    let rows = transpose(config.clip_from_world);
    let plane = rows[3] - rows[2];
    let normal_length = length(plane.xyz);
    if !(normal_length > 0.0 && normal_length < 3.402823e+38) { return -1.0; }
    let world_plane = plane / normal_length;
    let local_plane = transpose(config.world_from_local) * world_plane;
    let scale = bitcast<f32>(mapping[5]);
    // Authenticated page bounds permit a small relative containment tolerance.
    let tolerance = 0.00002 * max(node.center_radius.w, 1.0);
    let extent = (node.half_extents.xyz + vec3<f32>(tolerance)) * scale;
    let margin = max(config.view.w, 0.0);
    let support = dot(abs(local_plane.xyz), extent) + margin;
    let distance = dot(local_plane, vec4<f32>(node.center_radius.xyz, 1.0));
    let absolute_transform = transpose(mat4x4<f32>(abs(config.world_from_local[0]),
        abs(config.world_from_local[1]), abs(config.world_from_local[2]), abs(config.world_from_local[3])));
    let magnitude = absolute_transform * abs(world_plane);
    let allowance = 0.00000762939453125 * max(
        dot(magnitude, vec4<f32>(abs(node.center_radius.xyz) + extent, 1.0)) + margin, 1.0);
    let envelope = support + allowance;
    let clearance = distance - envelope;
    if !(envelope >= 0.0 && envelope < 3.402823e+38)
        || !(abs(clearance) < 3.402823e+38) { return -1.0; }
    return smoothstep(0.0, max(envelope, 1.0e-12), clearance);
}

fn inclusive_scan(lane: u32) {
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var prior = vec2<u32>(0u);
        if lane >= stride { prior = scan[lane - stride]; }
        workgroupBarrier();
        scan[lane] += prior;
        workgroupBarrier();
    }
}

// One representative range starts each adjacent edge. Its siblings are
// contiguous in the canonical node order, and the selected-node bitmap proves
// that none has been replaced by grandchildren or a resident ancestor.
fn classify_edge(range: u32) -> vec2<u32> {
    let node_index = range_node(range);
    let info = map_node(node_index);
    if info.x == NO_PARENT { return vec2<u32>(0u); }
    let parent = nodes[info.x];
    if lod_support_outside_view(parent.center_radius, parent.half_extents,
        config.world_from_local, config.clip_from_world, config.view, config.omission, config.frustum, 2.0) {
        return vec2<u32>(0u);
    }
    if node_index != parent.topology.x || parent.topology.y == 0u { return vec2<u32>(0u); }
    for (var child = 0u; child < parent.topology.y; child += 1u) {
        if !selected(parent.topology.x + child) { return vec2<u32>(0u); }
    }
    let parent_info = map_node(info.x);
    if parent_info.w != parent.counts.x || parent_info.w == 0u { return vec2<u32>(0u); }
    let cutoff = 16u + config.spatial.z;
    let tau = bitcast<f32>(atomicLoad(&state[cutoff]));
    // No excluded candidate means the actual discrete child endpoint. The
    // selector also emits actual parents, rather than child duplicates, at ties.
    if tau == 0.0 { return vec2<u32>(0u); }
    let score = bitcast<f32>(atomicLoad(&state[cutoff + 8u + info.x]));
    var weight = clamp((score - tau) / tau, 0.0, 1.0);
    if !(weight > 0.0 && weight < 1.0) {
        if !(weight >= 0.0 && weight <= 1.0) { atomicOr(&output[tail() + 5u], INVALID_PRESSURE); }
        return vec2<u32>(0u);
    }
    let parent_metric = metric(parent);
    let near_factor = envelope_near_factor(parent);
    // The actual child endpoint is reached before an interpolated envelope
    // could cross near. Missing/invalid safety evidence still bypasses the
    // complete edge; no child or opacity is independently discarded.
    if parent_metric.z != 0.0 || config.quality.z == 2.0 || !(near_factor > 0.0) {
        atomicOr(&output[tail() + 5u], NEAR_BYPASS);
        return vec2<u32>(0u);
    }
    for (var child = 0u; child < parent.topology.y; child += 1u) {
        let child_metric = metric(nodes[parent.topology.x + child]);
        if child_metric.z != 0.0 {
            atomicOr(&output[tail() + 5u], NEAR_BYPASS);
            return vec2<u32>(0u);
        }
    }
    // Keep the exact budget endpoints while advancing an active edge to its
    // child as camera clearance vanishes. max(weight, near_weight) would jump
    // at the budget's parent endpoint. The simultaneous zero-weight/unsafe
    // corner remains categorical; no stateless map can make that limit unique.
    let denominator = weight + (1.0 - weight) * near_factor;
    if !(denominator > 0.0 && denominator <= 1.0) {
        atomicOr(&output[tail() + 5u], INVALID_PRESSURE);
        return vec2<u32>(0u);
    }
    weight = weight / denominator;
    if weight >= 1.0 {
        atomicOr(&output[tail() + 5u], NEAR_BYPASS);
        return vec2<u32>(0u);
    }
    let page = pages[parent.topology.z];
    if page.count == 0u || parent.topology.w > page.count
        || parent.counts.x > page.count - parent.topology.w
    {
        atomicOr(&output[tail() + 5u], MISSING_PARENT);
        return vec2<u32>(0u);
    }
    let base = descriptor(range);
    atomicStore(&output[base + 2u], page.start + parent.topology.w);
    atomicStore(&output[base + 4u], parent_info.z);
    atomicStore(&output[base + 5u], parent_info.w);
    atomicStore(&output[base + 6u], bitcast<u32>(weight));
    atomicStore(&output[base + 7u], 1u);
    return vec2<u32>(1u, parent.counts.y);
}

@compute @workgroup_size(256)
fn classify(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    var cost = vec2<u32>(0u);
    var selection_available = config.spatial.z != 0u;
    if selection_available {
        selection_available = atomicLoad(&state[16u + config.spatial.z + 1u]) == 1u;
    }
    if id.x == 0u && !selection_available {
        atomicOr(&output[tail() + 5u], SELECTION_UNAVAILABLE);
    }
    if id.x < limits.frontier {
        atomicStore(&output[descriptor(id.x) + 7u], 0u);
    }
    // Each invocation owns its descriptor. Header initialization happened in
    // the preceding command copy, so no cross-workgroup initialization race.
    if id.x < range_count() && selection_available && (atomicLoad(&state[12]) & 1u) == 0u {
        let base = descriptor(id.x);
        let node = nodes[range_node(id.x)];
        atomicStore(&output[base], range_start(id.x));
        atomicStore(&output[base + 1u], node.counts.x);
        cost = classify_edge(id.x);
    }
    scan[lane] = cost;
    inclusive_scan(lane);
    if id.x < limits.frontier { scratch[id.x] = scan[lane]; }
    if lane == 255u { scratch[limits.frontier + group.x] = scan[lane]; }
}

@compute @workgroup_size(256)
fn scan_groups(@builtin(local_invocation_index) lane: u32) {
    let groups = (limits.frontier + 255u) / 256u;
    var cost = vec2<u32>(0u);
    if lane < groups { cost = scratch[limits.frontier + lane]; }
    scan[lane] = cost;
    inclusive_scan(lane);
    if lane < groups { scratch[limits.frontier + lane] = scan[lane] - cost; }
    if lane == 0u {
        atomicStore(&output[tail() + 2u], range_count());
        let required = scan[groups - 1u];
        // The copied indirect vertex-count word is not used by projection.
        atomicStore(&output[tail()], required.x);
        atomicStore(&output[tail() + 7u], required.y);
        if required.x > limits.edges || required.y > limits.records {
            atomicOr(&output[tail() + 5u], BAND_UNAVAILABLE);
        }
    }
}

@compute @workgroup_size(256)
fn emit(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= range_count() { return; }
    if (atomicLoad(&output[tail() + 5u]) & BAND_UNAVAILABLE) != 0u {
        // A partial canonical prefix would drop nonzero weights as the camera
        // moves. Expose unavailable and retain this whole source's discrete cut.
        atomicStore(&output[descriptor(id.x) + 7u], 0u);
        return;
    }
    let node_index = range_node(id.x);
    let info = map_node(node_index);
    if info.x == NO_PARENT { return; }
    let parent = nodes[info.x];
    let sibling = node_index - parent.topology.x;
    if sibling > id.x { return; }
    let first = id.x - sibling;
    if range_node(first) != parent.topology.x { return; }
    let source = descriptor(first);
    if atomicLoad(&output[source + 7u]) == 0u { return; }
    let base = descriptor(id.x);
    atomicStore(&output[base + 2u], atomicLoad(&output[source + 2u]));
    atomicStore(&output[base + 3u], info.y);
    atomicStore(&output[base + 4u], atomicLoad(&output[source + 4u]));
    atomicStore(&output[base + 5u], atomicLoad(&output[source + 5u]));
    atomicStore(&output[base + 6u], atomicLoad(&output[source + 6u]));
    atomicStore(&output[base + 7u], 1u);
    if sibling == 0u {
        atomicAdd(&output[tail() + 3u], 1u);
        atomicAdd(&output[tail() + 6u], parent.counts.y);
        let bit = 1u << (parent.topology.z & 31u);
        let word = 16u + config.feedback_offsets.x + parent.topology.z / 32u;
        if (atomicOr(&state[word], bit) & bit) == 0u {
            let slot = atomicAdd(&state[11], 1u);
            // Host allocation adds at least one slot per admitted edge.
            if slot < limits.selected_pages {
                atomicStore(&state[16u + config.feedback_offsets.z + slot], parent.topology.z);
            }
        }
    }
}
