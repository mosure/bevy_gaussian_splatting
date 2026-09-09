#import bevy_gaussian_splatting::lod_spatial_metrics::{lod_projection_metric, lod_finite_scheduling_score, lod_support_outside_view}

// Bounded, current-camera desired selection is independent of page residency.
// A second walk resolves that target to a complete resident ancestor cover.
// The logical antichain always covers the source. Proven invisible ranges
// retain their node identity while physical count and page demand become zero.
struct Node {
    center_radius: vec4<f32>,
    half_extents: vec4<f32>,
    error_quality: vec4<f32>,
    topology: vec4<u32>,
    counts: vec4<u32>,
}
struct Page { start: u32, count: u32 }
struct Entry { key: u32, value: u32 }
struct Config {
    world_from_local: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    // Width, height, conservative world scale, frustum margin.
    view: vec4<f32>,
    // Detail, error limit, endpoint (coarse=0, balanced=1, original=2), frustum enabled.
    quality: vec4<f32>,
    // Nodes, pages, roots, initial root representative count.
    counts: vec4<u32>,
    // Record, frontier-node, visit and request bounds.
    limits: vec4<u32>,
    // Queue A, queue B, accepted ranges, request bitmap offsets (in words).
    offsets: vec4<u32>,
    // Selected bitmap, requests, selected pages, bitmap word count.
    feedback_offsets: vec4<u32>,
    // Selected page capacity, transition edge capacity, cutoff scratch, candidate count.
    spatial: vec4<u32>,
    // Support inflation (0 disables), Mip factor, perspective, active annotation.
    omission: vec4<f32>,
    frustum: array<vec4<f32>, 6>,
}
struct State {
    dispatch_x: u32, dispatch_y: u32, dispatch_z: u32, current_count: u32,
    next_count: atomic<u32>, selected_count: atomic<u32>, active_records: atomic<u32>, active_nodes: atomic<u32>,
    output_count: atomic<u32>, visits: atomic<u32>, request_count: atomic<u32>, selected_page_count: atomic<u32>,
    flags: atomic<u32>, level: u32, phase: u32,
    admission_boundary: u32,
    words: array<atomic<u32>>,
}
struct Draw { vertices: u32, instances: u32, first_vertex: u32, first_instance: u32,
    flags: u32, padding_a: u32, padding_b: u32, padding_c: u32 }
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> pages: array<Page>;
@group(0) @binding(3) var<storage, read_write> state: State;
@group(0) @binding(4) var<storage, read_write> entries: array<Entry>;
@group(0) @binding(5) var<storage, read_write> draw: Draw;
@group(0) @binding(6) var<storage, read> source_order: array<u32>;

const OMITTED_OUTSIDE_VIEW: u32 = 1u;
const MISSING_ROOT: u32 = 1u;
const RECORD_LIMIT: u32 = 2u;
const FRONTIER_LIMIT: u32 = 4u;
const REQUEST_LIMIT: u32 = 8u;
const VISIT_LIMIT: u32 = 16u;
const CUTOFF_UNAVAILABLE: u32 = 32u;
// Inclusive per-node scans use five words: extra records, extra frontier
// nodes, child visits, replaced parent records, and input parent records.
const SCAN_WORDS: u32 = 5u;
var<workgroup> scan_cost: array<vec4<u32>, 256>;
var<workgroup> scan_records: array<u32, 256>;
const PRIORITY_BUCKETS: u32 = 8u;
var<workgroup> rank_upper: array<vec4<u32>, 256>;
fn prefix_base() -> u32 { return config.feedback_offsets.z + config.spatial.x; }
fn group_base() -> u32 { return prefix_base() + SCAN_WORDS * config.limits.y; }
fn summary_base() -> u32 {
    return group_base() + PRIORITY_BUCKETS * ((config.limits.y + 255u) / 256u);
}
fn ordered_input_base() -> u32 { return summary_base() + SCAN_WORDS; }
fn node_bitmap_words() -> u32 { return (config.counts.x + 31u) / 32u; }
fn rank_totals_base() -> u32 { return ordered_input_base() + config.limits.y; }
fn pending_base() -> u32 { return rank_totals_base() + PRIORITY_BUCKETS; }
fn desired_base() -> u32 { return pending_base() + PRIORITY_BUCKETS * node_bitmap_words(); }
fn selected_base() -> u32 { return desired_base() + node_bitmap_words(); }
fn canonical_base() -> u32 { return selected_base() + node_bitmap_words(); }
fn bitmap_prefix_base() -> u32 { return canonical_base() + node_bitmap_words(); }
fn bitmap_group_base() -> u32 { return bitmap_prefix_base() + 2u * node_bitmap_words(); }
fn visibility_base() -> u32 { return bitmap_group_base() + 2u * ((node_bitmap_words() + 255u) / 256u); }
fn outside_node(index: u32, envelope: f32) -> bool {
    let node = nodes[index];
    return lod_support_outside_view(node.center_radius, node.half_extents,
        config.world_from_local, config.clip_from_world, config.view, config.omission, config.frustum, envelope);
}
fn own_record_count(index: u32) -> u32 {
    return select(nodes[index].counts.x, 0u, node_bit(visibility_base(), index));
}
fn range_base(range: u32) -> u32 { return config.offsets.z + range * 4u; }
fn entry_word(word: u32) -> u32 {
    if (word & 1u) == 0u { return entries[word / 2u].key; }
    return entries[word / 2u].value;
}
fn store_entry_word(word: u32, value: u32) {
    if (word & 1u) == 0u { entries[word / 2u].key = value; }
    else { entries[word / 2u].value = value; }
}
fn descriptor_base(range: u32) -> u32 { return config.limits.x * 2u + 8u + range * 8u; }

fn bitmap_base() -> u32 {
    if state.phase == 3u { return canonical_base(); }
    return pending_base() + atomicLoad(&state.next_count) * node_bitmap_words();
}
fn node_bit(base: u32, index: u32) -> bool {
    return (atomicLoad(&state.words[base + index / 32u]) & (1u << (index & 31u))) != 0u;
}
fn set_node_bit(base: u32, index: u32) {
    atomicOr(&state.words[base + index / 32u], 1u << (index & 31u));
}
fn load_cost(base: u32) -> vec4<u32> {
    return vec4<u32>(atomicLoad(&state.words[base]), atomicLoad(&state.words[base + 1u]),
        atomicLoad(&state.words[base + 2u]), atomicLoad(&state.words[base + 3u]));
}
fn store_cost(base: u32, cost: vec4<u32>) {
    for (var axis = 0u; axis < 4u; axis += 1u) { atomicStore(&state.words[base + axis], cost[axis]); }
}
fn add_cost(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    return min(a, vec4<u32>(0xffffffffu) - b) + b;
}
fn input_node(index: u32) -> u32 {
    if state.phase == 1u {
        let offset = select(config.offsets.x, config.offsets.y, (state.level & 1u) != 0u);
        return atomicLoad(&state.words[offset + index]);
    }
    return atomicLoad(&state.words[ordered_input_base() + index]);
}
fn fits(cost: vec4<u32>) -> bool {
    if state.phase == 1u { return true; }
    if state.phase == 2u { return cost.z <= config.limits.w - atomicLoad(&state.request_count); }
    return cost.x <= config.limits.x - atomicLoad(&state.active_records)
        && cost.y <= config.limits.y - atomicLoad(&state.active_nodes)
        && cost.z <= config.limits.z - atomicLoad(&state.visits);
}

fn request_page(page: u32) {
    let bit = 1u << (page & 31u);
    let old = atomicOr(&state.words[config.offsets.w + page / 32u], bit);
    if (old & bit) == 0u {
        let index = atomicAdd(&state.request_count, 1u);
        if index < config.limits.w {
            atomicStore(&state.words[config.feedback_offsets.y + index], page);
        } else { atomicOr(&state.flags, REQUEST_LIMIT); }
    }
}

// All accepted ranges have deterministic offsets. Only the deduplicated
// telemetry page list uses atomic append; it never determines the visible cut.
fn accept_node(index: u32, slot: u32, start: u32) {
    let node = nodes[index];
    set_node_bit(selected_base(), index);
    set_node_bit(canonical_base(), node.counts.w);
    let base = range_base(slot);
    atomicStore(&state.words[base], index);
    atomicStore(&state.words[base + 1u], start);
    let count = nodes[index].counts.x;
    atomicStore(&state.words[base + 2u], count);
    atomicStore(&state.words[base + 3u], select(0u, OMITTED_OUTSIDE_VIEW, count == 0u));
}

// Zero means no refinement. Eligibility retains the quality policy, while
// priority uses physical projected severity; certificate saturation must not
// make a distant proxy outrank a near-camera proxy by its topology index.
fn refinement_priority(node: Node) -> u32 {
    if node.topology.y == 0u || config.quality.z == 0.0 { return 0u; }
    let projected = lod_projection_metric(node.center_radius, node.half_extents,
        node.error_quality, node.counts.z, config.world_from_local,
        config.clip_from_world, config.view, config.quality);
    return select(0u, u32(projected.y), projected.w != 0.0
        && (config.quality.z == 2.0 || projected.x > 1.0));
}

@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x == 0u {
        state.dispatch_x = 0u; state.dispatch_y = 1u; state.dispatch_z = 1u; state.current_count = 0u;
        atomicStore(&state.next_count, 7u); atomicStore(&state.selected_count, 0u);
        atomicStore(&state.active_records, 0u); atomicStore(&state.active_nodes, config.counts.z);
        atomicStore(&state.output_count, 0u); atomicStore(&state.visits, config.counts.z);
        atomicStore(&state.request_count, 0u); atomicStore(&state.selected_page_count, 0u);
        var initial_flags = 0u;
        if config.quality.z == 1.0 && config.spatial.y != 0u
            && config.spatial.z == 0u && config.spatial.w != 0u { initial_flags = CUTOFF_UNAVAILABLE; }
        atomicStore(&state.flags, initial_flags); state.level = 0u; state.phase = 0u;
        if config.spatial.z != 0u {
            for (var word = 0u; word < 8u; word += 1u) { atomicStore(&state.words[config.spatial.z + word], 0u); }
        }
        draw.vertices = 4u; draw.instances = 0u; draw.first_vertex = 0u; draw.first_instance = 0u;
        draw.flags = 0u;
    }
    if id.x < config.feedback_offsets.w {
        atomicStore(&state.words[config.offsets.w + id.x], 0u);
        atomicStore(&state.words[config.feedback_offsets.x + id.x], 0u);
    }
    if id.x < node_bitmap_words() {
        atomicStore(&state.words[visibility_base() + id.x], 0u);
        for (var bucket = 0u; bucket < PRIORITY_BUCKETS + 3u; bucket += 1u) {
            atomicStore(&state.words[pending_base() + bucket * node_bitmap_words() + id.x], 0u);
        }
    }
}

@compute @workgroup_size(256)
fn classify_visibility(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < config.counts.x && outside_node(id.x, 1.0) { set_node_bit(visibility_base(), id.x); }
}

@compute @workgroup_size(256)
fn bootstrap(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= config.counts.z { return; }
    atomicAdd(&state.active_records, nodes[id.x].counts.x);
    let page = nodes[id.x].topology.z;
    if pages[page].count == 0u { request_page(page); atomicOr(&state.flags, MISSING_ROOT); }
    atomicStore(&state.words[config.offsets.x + id.x], id.x);
    let priority = refinement_priority(nodes[id.x]);
    if priority != 0u { set_node_bit(pending_base() + priority * node_bitmap_words(), id.x); }
}

// Each bitmap word is scanned once. Desired bits retain immutable node order;
// canonical output bits follow source domains across parent/child substitutions.
// The same scratch handles desired queues, blocked demand and canonical output.
@compute @workgroup_size(256)
fn bitmap_scan(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var count = 0u;
    var records = 0u;
    if id.x < node_bitmap_words() && (atomicLoad(&state.flags) & MISSING_ROOT) == 0u {
        var bits = atomicLoad(&state.words[bitmap_base() + id.x]);
        count = countOneBits(bits);
        if state.phase == 3u {
            while bits != 0u {
                let bit = firstTrailingBit(bits);
                records += nodes[source_order[id.x * 32u + bit]].counts.x;
                bits &= bits - 1u;
            }
        }
    }
    scan_cost[lane] = vec4<u32>(count, records, 0u, 0u);
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var previous = vec4<u32>(0u);
        if lane >= stride { previous = scan_cost[lane - stride]; }
        workgroupBarrier();
        scan_cost[lane] += previous;
        workgroupBarrier();
    }
    if id.x < node_bitmap_words() {
        atomicStore(&state.words[bitmap_prefix_base() + id.x * 2u], scan_cost[lane].x);
        atomicStore(&state.words[bitmap_prefix_base() + id.x * 2u + 1u], scan_cost[lane].y);
    }
    if lane == 255u {
        atomicStore(&state.words[bitmap_group_base() + group.x * 2u], scan_cost[lane].x);
        atomicStore(&state.words[bitmap_group_base() + group.x * 2u + 1u], scan_cost[lane].y);
    }
}

@compute @workgroup_size(1)
fn bitmap_groups() {
    var count = 0u;
    var records = 0u;
    for (var group = 0u; group < (node_bitmap_words() + 255u) / 256u; group += 1u) {
        let base = bitmap_group_base() + group * 2u;
        let next_count = count + atomicLoad(&state.words[base]);
        let next_records = records + atomicLoad(&state.words[base + 1u]);
        atomicStore(&state.words[base], count);
        atomicStore(&state.words[base + 1u], records);
        count = next_count;
        records = next_records;
    }
    // Pending queues are subsets of the admitted logical frontier. Fail closed
    // if a future producer violates that invariant rather than overwrite scratch.
    if count > config.limits.y || records > config.limits.x {
        atomicOr(&state.flags, MISSING_ROOT);
        count = 0u;
    }
    state.current_count = count;
    state.dispatch_x = (count + 255u) / 256u;
    if state.phase == 3u {
        if count != atomicLoad(&state.selected_count) || records != atomicLoad(&state.output_count) {
            atomicOr(&state.flags, MISSING_ROOT);
        }
    }
}

@compute @workgroup_size(256)
fn bitmap_scatter(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    if id.x >= node_bitmap_words() || (atomicLoad(&state.flags) & MISSING_ROOT) != 0u { return; }
    var bits = atomicLoad(&state.words[bitmap_base() + id.x]);
    var count = atomicLoad(&state.words[bitmap_group_base() + group.x * 2u]);
    var records = atomicLoad(&state.words[bitmap_group_base() + group.x * 2u + 1u]);
    if lane != 0u {
        count += atomicLoad(&state.words[bitmap_prefix_base() + (id.x - 1u) * 2u]);
        records += atomicLoad(&state.words[bitmap_prefix_base() + (id.x - 1u) * 2u + 1u]);
    }
    // Clear before a subsequent dispatch enqueues descendants in this bucket.
    if state.phase != 3u { atomicStore(&state.words[bitmap_base() + id.x], 0u); }
    while bits != 0u {
        let bit = firstTrailingBit(bits);
        var index = id.x * 32u + bit;
        if state.phase == 3u {
            index = source_order[index];
            let base = range_base(count);
            let physical_count = nodes[index].counts.x;
            atomicStore(&state.words[base], index);
            atomicStore(&state.words[base + 1u], records);
            atomicStore(&state.words[base + 2u], physical_count);
            atomicStore(&state.words[base + 3u], select(0u, OMITTED_OUTSIDE_VIEW, physical_count == 0u));
            records += physical_count;
        } else {
            atomicStore(&state.words[ordered_input_base() + count], index);
        }
        count += 1u;
        bits &= bits - 1u;
    }
}

// Necessary missing cohorts precede bounded speculative demand. The latter
// preserves complete sibling pages before the spatial band reaches this node;
// it never changes the logical cut, selected records, or traversal visit budget.
fn demand_priority(index: u32) -> u32 {
    let node = nodes[index];
    if node.topology.y == 0u { return 0u; }
    let priority = refinement_priority(node);
    if node_bit(desired_base(), index) {
        if config.spatial.z != 0u && config.quality.z == 1.0 { return 3u + (priority + 1u) / 2u; }
        return priority;
    }
    if config.spatial.z == 0u || config.quality.z != 1.0 { return 0u; }
    let tau = bitcast<f32>(atomicLoad(&state.words[config.spatial.z]));
    let score = bitcast<f32>(atomicLoad(&state.words[score_base() + index]));
    if tau > 0.0 && score > 0.0 && score >= 0.5 * tau {
        return min(3u, (priority + 2u) / 3u);
    }
    return 0u;
}

// Desired scheduling is global across depth within descending priority bins.
// Child priorities are capped by their admitted parent's scheduling priority;
// therefore each bucket drains within the authenticated tree depth bound.
@compute @workgroup_size(256)
fn traverse(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var cost = vec4<u32>(0u);
    var records = 0u;
    if id.x < state.current_count && (atomicLoad(&state.flags) & MISSING_ROOT) == 0u {
        let index = input_node(id.x);
        let node = nodes[index];
        records = nodes[index].counts.x;
        var split = state.phase == 0u || (state.phase == 2u && demand_priority(index) != 0u);
        if state.phase == 1u && node_bit(desired_base(), index) {
            split = true;
            for (var child = 0u; child < node.topology.y; child += 1u) {
                let child_index = node.topology.x + child;
                if pages[nodes[child_index].topology.z].count == 0u { split = false; break; }
            }
        }
        if split {
            var child_records = 0u;
            for (var child = 0u; child < node.topology.y; child += 1u) {
                child_records += nodes[node.topology.x + child].counts.x;
            }
            let parent_records = nodes[index].counts.x;
            // Never refund a negative split cost: this monotone reserve funds
            // every resident fallback cut, even when children need fewer records.
            cost = vec4<u32>(child_records - min(child_records, parent_records),
                node.topology.y - 1u, node.topology.y, records);
        }
    }
    scan_cost[lane] = cost;
    scan_records[lane] = records;
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var previous_cost = vec4<u32>(0u);
        var previous_records = 0u;
        if lane >= stride {
            previous_cost = scan_cost[lane - stride];
            previous_records = scan_records[lane - stride];
        }
        workgroupBarrier();
        scan_cost[lane] = add_cost(scan_cost[lane], previous_cost);
        scan_records[lane] += previous_records;
        workgroupBarrier();
    }
    if id.x < state.current_count {
        let base = prefix_base() + id.x * SCAN_WORDS;
        store_cost(base, scan_cost[lane]);
        atomicStore(&state.words[base + 4u], scan_records[lane]);
    }
    if lane == 255u {
        let base = group_base() + group.x * SCAN_WORDS;
        store_cost(base, scan_cost[lane]);
        atomicStore(&state.words[base + 4u], scan_records[lane]);
    }
}

@compute @workgroup_size(1)
fn scan_groups() {
    if (atomicLoad(&state.flags) & MISSING_ROOT) != 0u { return; }
    var total = vec4<u32>(0u);
    var total_records = 0u;
    var admitted = vec4<u32>(0u);
    var boundary = state.current_count;
    let groups = (state.current_count + 255u) / 256u;
    for (var group = 0u; group < groups; group += 1u) {
        let base = group_base() + group * SCAN_WORDS;
        let next = add_cost(total, load_cost(base));
        let next_records = total_records + atomicLoad(&state.words[base + 4u]);
        if boundary == state.current_count && !fits(next) {
            for (var lane = 0u; lane < 256u; lane += 1u) {
                let index = group * 256u + lane;
                if index >= state.current_count { break; }
                let cost = add_cost(total, load_cost(prefix_base() + index * SCAN_WORDS));
                if !fits(cost) {
                    boundary = index;
                    var flags = 0u;
                    if state.phase == 2u { flags = REQUEST_LIMIT; }
                    else {
                        if cost.x > config.limits.x - atomicLoad(&state.active_records) { flags |= RECORD_LIMIT; }
                        if cost.y > config.limits.y - atomicLoad(&state.active_nodes) { flags |= FRONTIER_LIMIT; }
                        if cost.z > config.limits.z - atomicLoad(&state.visits) { flags |= VISIT_LIMIT; }
                    }
                    atomicOr(&state.flags, flags);
                    break;
                }
                admitted = cost;
            }
        }
        if boundary == state.current_count { admitted = next; }
        store_cost(base, total);
        atomicStore(&state.words[base + 4u], total_records);
        total = next;
        total_records = next_records;
    }
    store_cost(summary_base(), admitted);
    atomicStore(&state.words[summary_base() + 4u], total_records);
    state.admission_boundary = boundary;
    if state.phase == 0u {
        atomicAdd(&state.active_records, admitted.x);
        atomicAdd(&state.active_nodes, admitted.y);
        atomicAdd(&state.visits, admitted.z);
    }
    if state.phase == 2u { atomicAdd(&state.request_count, admitted.z); }
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32) {
    if id.x >= state.current_count || (atomicLoad(&state.flags) & MISSING_ROOT) != 0u { return; }
    let index = input_node(id.x);
    let node = nodes[index];
    let base = prefix_base() + id.x * SCAN_WORDS;
    let local = load_cost(base);
    var previous_visits = 0u;
    if lane != 0u { previous_visits = atomicLoad(&state.words[base - SCAN_WORDS + 2u]); }
    let candidate = local.z != previous_visits;
    let group_offset = group_base() + group.x * SCAN_WORDS;
    let prefix = add_cost(load_cost(group_offset), local);
    if state.phase == 0u {
        if id.x < state.admission_boundary {
            set_node_bit(desired_base(), index);
            for (var child = 0u; child < node.topology.y; child += 1u) {
                let child_index = node.topology.x + child;
                let priority = min(refinement_priority(nodes[child_index]), atomicLoad(&state.next_count));
                if priority != 0u { set_node_bit(pending_base() + priority * node_bitmap_words(), child_index); }
            }
        }
        return;
    }
    if state.phase == 2u {
        if candidate && id.x < state.admission_boundary {
            // Demand belongs to the fixed logical target, not spare display
            // capacity. Preserve all siblings while their parent remains visible.
            var start = atomicLoad(&state.request_count) - load_cost(summary_base()).z + prefix.z - node.topology.y;
            for (var child = 0u; child < node.topology.y; child += 1u) {
                let child_index = node.topology.x + child;
                atomicStore(&state.words[config.feedback_offsets.y + start], nodes[child_index].topology.z);
                start += 1u;
            }
        }
        return;
    }
    if candidate {
        let write_offset = select(config.offsets.y, config.offsets.x, (state.level & 1u) != 0u);
        let start = prefix.z - node.topology.y;
        for (var child = 0u; child < node.topology.y; child += 1u) {
            atomicStore(&state.words[write_offset + start + child], node.topology.x + child);
        }
    } else {
        let previous_splits = prefix.z - prefix.y;
        let previous_records = atomicLoad(&state.words[group_offset + 4u])
            + atomicLoad(&state.words[base + 4u]) - nodes[index].counts.x;
        accept_node(index, atomicLoad(&state.selected_count) + id.x - previous_splits,
            atomicLoad(&state.output_count) + previous_records - prefix.w);
    }
}

// Demand is one stable eight-bin scan over the actual resident ranges.
// Reuse cost-prefix scratch after fallback, avoiding seven empty bitmap rounds.
@compute @workgroup_size(256)
fn rank_demand(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var priority = 0u;
    var lower = vec4<u32>(0u);
    var upper = vec4<u32>(0u);
    if id.x < state.current_count && (atomicLoad(&state.flags) & MISSING_ROOT) == 0u {
        let index = atomicLoad(&state.words[range_base(id.x)]);
        priority = demand_priority(index);
        if priority < 4u { lower[priority] = 1u; } else { upper[priority - 4u] = 1u; }
    }
    scan_cost[lane] = lower;
    rank_upper[lane] = upper;
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        lower = vec4<u32>(0u); upper = vec4<u32>(0u);
        if lane >= stride { lower = scan_cost[lane - stride]; upper = rank_upper[lane - stride]; }
        workgroupBarrier();
        scan_cost[lane] += lower; rank_upper[lane] += upper;
        workgroupBarrier();
    }
    if id.x < state.current_count {
        var prefix = 0u;
        if priority < 4u { prefix = scan_cost[lane][priority]; }
        else { prefix = rank_upper[lane][priority - 4u]; }
        atomicStore(&state.words[prefix_base() + id.x * 2u], priority);
        atomicStore(&state.words[prefix_base() + id.x * 2u + 1u], prefix - 1u);
    }
    if lane == 255u {
        store_cost(group_base() + group.x * 8u, scan_cost[lane]);
        store_cost(group_base() + group.x * 8u + 4u, rank_upper[lane]);
    }
}

@compute @workgroup_size(1)
fn rank_demand_groups() {
    if (atomicLoad(&state.flags) & MISSING_ROOT) != 0u { return; }
    var lower = vec4<u32>(0u); var upper = vec4<u32>(0u);
    for (var group = 0u; group < (state.current_count + 255u) / 256u; group += 1u) {
        let base = group_base() + group * 8u;
        let next_lower = lower + load_cost(base);
        let next_upper = upper + load_cost(base + 4u);
        store_cost(base, lower); store_cost(base + 4u, upper);
        lower = next_lower; upper = next_upper;
    }
    var start = 0u;
    for (var reverse = 0u; reverse < PRIORITY_BUCKETS; reverse += 1u) {
        let priority = PRIORITY_BUCKETS - reverse - 1u;
        atomicStore(&state.words[rank_totals_base() + priority], start);
        if priority < 4u { start += lower[priority]; } else { start += upper[priority - 4u]; }
    }
}

@compute @workgroup_size(256)
fn order_demand(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    if id.x >= state.current_count || (atomicLoad(&state.flags) & MISSING_ROOT) != 0u { return; }
    let priority = atomicLoad(&state.words[prefix_base() + id.x * 2u]);
    let output = atomicLoad(&state.words[rank_totals_base() + priority])
        + atomicLoad(&state.words[group_base() + group.x * 8u + priority])
        + atomicLoad(&state.words[prefix_base() + id.x * 2u + 1u]);
    atomicStore(&state.words[ordered_input_base() + output],
        atomicLoad(&state.words[range_base(id.x)]));
}

@compute @workgroup_size(1)
fn next_bucket() { atomicSub(&state.next_count, 1u); }

@compute @workgroup_size(1)
fn begin_resident() {
    state.phase = 1u; state.level = 0u;
    state.current_count = config.counts.z;
    state.dispatch_x = (state.current_count + 255u) / 256u;
}

@compute @workgroup_size(1)
fn begin_demand() {
    state.phase = 2u;
    state.current_count = atomicLoad(&state.selected_count);
    state.dispatch_x = (state.current_count + 255u) / 256u;
}

@compute @workgroup_size(1)
fn begin_canonical() { state.phase = 3u; }

@compute @workgroup_size(1)
fn advance() {
    if (atomicLoad(&state.flags) & MISSING_ROOT) != 0u {
        state.current_count = 0u; state.dispatch_x = 0u; return;
    }
    let admitted = load_cost(summary_base());
    atomicAdd(&state.selected_count, state.current_count - (admitted.z - admitted.y));
    atomicAdd(&state.output_count, atomicLoad(&state.words[summary_base() + 4u]) - admitted.w);
    state.current_count = admitted.z;
    state.level += 1u;
    state.dispatch_x = (state.current_count + 255u) / 256u;
}

@compute @workgroup_size(1)
fn prepare_expand() { state.dispatch_x = atomicLoad(&state.selected_count); }

fn retain_page(page: u32) {
    let bit = 1u << (page & 31u);
    if (atomicOr(&state.words[config.feedback_offsets.x + page / 32u], bit) & bit) == 0u {
        let slot = atomicAdd(&state.selected_page_count, 1u);
        if slot < config.spatial.x {
            atomicStore(&state.words[config.feedback_offsets.z + slot], page);
        } else { atomicOr(&state.flags, MISSING_ROOT); }
    }
}

// Annotation has now identified the actual fractional band. Only these edges
// expand all immediate children. Logical selection, residency and complete
// cohort demand remain independent of this final physical omission.
@compute @workgroup_size(256)
fn physical_counts(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    if id.x == 0u { atomicStore(&state.selected_page_count, 0u); }
    if id.x < config.feedback_offsets.w {
        atomicStore(&state.words[config.feedback_offsets.x + id.x], 0u);
    }
    var count = 0u;
    if id.x < atomicLoad(&state.selected_count) && (atomicLoad(&state.flags) & MISSING_ROOT) == 0u {
        let base = range_base(id.x);
        let index = atomicLoad(&state.words[base]);
        count = own_record_count(index);
        if config.omission.w != 0.0 && entry_word(descriptor_base(id.x) + 7u) != 0u {
            // Active annotation already rejects the whole edge if the expanded
            // parent envelope is outside. Every sibling therefore keeps its
            // complete cardinality, including offscreen endpoint children.
            count = nodes[index].counts.x;
        }
        let page = pages[nodes[index].topology.z];
        if count != 0u && (nodes[index].topology.w > page.count
            || count > page.count - min(nodes[index].topology.w, page.count)) {
            atomicOr(&state.flags, MISSING_ROOT);
            count = 0u;
        }
        atomicStore(&state.words[base + 2u], count);
        atomicStore(&state.words[base + 3u], select(0u, OMITTED_OUTSIDE_VIEW, count == 0u));
    }
    scan_records[lane] = count;
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var previous = 0u;
        if lane >= stride { previous = scan_records[lane - stride]; }
        workgroupBarrier();
        scan_records[lane] += previous;
        workgroupBarrier();
    }
    if id.x < config.limits.y {
        atomicStore(&state.words[prefix_base() + id.x], scan_records[lane]);
    }
    if lane == 255u && group.x < (config.limits.y + 255u) / 256u {
        atomicStore(&state.words[group_base() + group.x], scan_records[lane]);
    }
}

@compute @workgroup_size(1)
fn physical_groups() {
    var count = 0u;
    let groups = (config.limits.y + 255u) / 256u;
    for (var group = 0u; group < groups; group += 1u) {
        let next = count + atomicLoad(&state.words[group_base() + group]);
        atomicStore(&state.words[group_base() + group], count);
        count = next;
    }
    if count > config.limits.x { atomicOr(&state.flags, MISSING_ROOT); }
    if (atomicLoad(&state.flags) & MISSING_ROOT) != 0u { count = 0u; }
    atomicStore(&state.output_count, count);
    if config.omission.w != 0.0 { store_entry_word(config.limits.x * 2u + 1u, count); }
    state.dispatch_x = atomicLoad(&state.selected_count);
}

@compute @workgroup_size(256)
fn physical_ranges(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    if id.x >= atomicLoad(&state.selected_count) || (atomicLoad(&state.flags) & MISSING_ROOT) != 0u { return; }
    let base = range_base(id.x);
    let index = atomicLoad(&state.words[base]);
    let count = atomicLoad(&state.words[base + 2u]);
    let start = atomicLoad(&state.words[group_base() + group.x])
        + atomicLoad(&state.words[prefix_base() + id.x]) - count;
    atomicStore(&state.words[base + 1u], start);
    if config.omission.w != 0.0 {
        let descriptor = descriptor_base(id.x);
        store_entry_word(descriptor, start);
        store_entry_word(descriptor + 1u, count);
        if count != 0u && entry_word(descriptor + 7u) != 0u {
            retain_page(nodes[bitcast<u32>(nodes[index].error_quality.w)].topology.z);
        }
    }
    // Every logical range was resolved from a complete resident cohort. Keep
    // its page even when final physical expansion omits all endpoint records.
    retain_page(nodes[index].topology.z);
}

fn sampling_identity(node: u32, local: u32) -> u32 {
    var key = node * 0x9e3779b9u ^ local;
    key = (key ^ (key >> 16u)) * 0x7feb352du;
    key = (key ^ (key >> 15u)) * 0x846ca68bu;
    return key ^ (key >> 16u);
}

@compute @workgroup_size(256)
fn expand(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if group.x >= atomicLoad(&state.selected_count) || (atomicLoad(&state.flags) & MISSING_ROOT) != 0u { return; }
    let index = atomicLoad(&state.words[range_base(group.x)]);
    let node = nodes[index];
    let output_start = atomicLoad(&state.words[range_base(group.x) + 1u]);
    let count = atomicLoad(&state.words[range_base(group.x) + 2u]);
    let source_start = pages[node.topology.z].start + node.topology.w;
    for (var local = lane; local < count; local += 256u) {
        entries[output_start + local] = Entry(sampling_identity(index, local), source_start + local);
    }
}

@compute @workgroup_size(1)
fn finish() {
    draw.flags = atomicLoad(&state.flags);
    draw.instances = select(0u, atomicLoad(&state.output_count), (draw.flags & MISSING_ROOT) == 0u);
    if config.omission.w != 0.0 {
        store_entry_word(config.limits.x * 2u + 1u, draw.instances);
        store_entry_word(config.limits.x * 2u + 4u, draw.flags);
    }
}


// Exact budget threshold for a complete, bounded internal candidate graph.
// All arrays share the admitted traversal workspace; no Gaussian records sort.
fn score_base() -> u32 { return config.spatial.z + 8u; }
fn cutoff_a() -> u32 { return score_base() + config.counts.x; }
fn cutoff_b() -> u32 { return cutoff_a() + 2u * config.spatial.w; }
fn cutoff_hist() -> u32 { return cutoff_b() + 2u * config.spatial.w; }
fn cutoff_groups_count() -> u32 { return (config.spatial.w + 255u) / 256u; }
fn cutoff_cost_base() -> u32 { return cutoff_hist() + 256u * cutoff_groups_count(); }
fn cutoff_cost_groups() -> u32 { return cutoff_cost_base() + 4u * config.spatial.w; }
fn cutoff_input() -> u32 {
    return select(cutoff_a(), cutoff_b(), (atomicLoad(&state.words[config.spatial.z + 2u]) & 1u) != 0u);
}
fn candidate_score(index: u32) -> f32 {
    return bitcast<f32>(~atomicLoad(&state.words[cutoff_a() + index * 2u]));
}

@compute @workgroup_size(256)
fn score_candidates(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= config.spatial.w { return; }
    let base = config.counts.x + id.x * 4u;
    if source_order[base + 2u] != state.level { return; }
    let index = source_order[base];
    let parent = source_order[base + 1u];
    let node = nodes[index];
    var score = 0.0;
    if refinement_priority(node) != 0u {
        score = lod_finite_scheduling_score(node.center_radius, node.half_extents,
            node.error_quality, config.world_from_local, config.clip_from_world, config.view);
        if parent != 0xffffffffu {
            score = min(score, 0.5 * bitcast<f32>(atomicLoad(&state.words[score_base() + parent])));
        }
    }
    atomicStore(&state.words[score_base() + index], bitcast<u32>(score));
    let root_alias = source_order[base + 3u];
    if root_alias != 0xffffffffu { atomicStore(&state.words[score_base() + root_alias], bitcast<u32>(score)); }
    // Positive finite floats have monotone unsigned bits. Complement for
    // descending order; stable radix retains source-domain ties.
    atomicStore(&state.words[cutoff_a() + id.x * 2u], ~bitcast<u32>(score));
    atomicStore(&state.words[cutoff_a() + id.x * 2u + 1u], index);
}

@compute @workgroup_size(1)
fn score_next_depth() { state.level += 1u; }

@compute @workgroup_size(256)
fn cutoff_radix_histogram(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var digit = 256u;
    let shift = atomicLoad(&state.words[config.spatial.z + 2u]) * 8u;
    if id.x < config.spatial.w { digit = (atomicLoad(&state.words[cutoff_input() + id.x * 2u]) >> shift) & 255u; }
    scan_records[lane] = digit;
    workgroupBarrier();
    var count = 0u;
    for (var other = 0u; other < 256u; other += 1u) {
        count += u32(scan_records[other] == lane);
    }
    atomicStore(&state.words[cutoff_hist() + group.x * 256u + lane], count);
}

@compute @workgroup_size(256)
fn cutoff_radix_groups(@builtin(local_invocation_index) lane: u32) {
    var total = 0u;
    for (var group = 0u; group < cutoff_groups_count(); group += 1u) {
        let index = cutoff_hist() + group * 256u + lane;
        let next = total + atomicLoad(&state.words[index]);
        atomicStore(&state.words[index], total);
        total = next;
    }
    scan_records[lane] = total;
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var previous = 0u;
        if lane >= stride { previous = scan_records[lane - stride]; }
        workgroupBarrier();
        scan_records[lane] += previous;
        workgroupBarrier();
    }
    let offset = scan_records[lane] - total;
    for (var group = 0u; group < cutoff_groups_count(); group += 1u) {
        atomicAdd(&state.words[cutoff_hist() + group * 256u + lane], offset);
    }
}

@compute @workgroup_size(256)
fn cutoff_radix_scatter(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let input = cutoff_input();
    let output = select(cutoff_b(), cutoff_a(), input == cutoff_b());
    let shift = atomicLoad(&state.words[config.spatial.z + 2u]) * 8u;
    var key = 0xffffffffu;
    var digit = 256u;
    if id.x < config.spatial.w {
        key = atomicLoad(&state.words[input + id.x * 2u]); digit = (key >> shift) & 255u;
    }
    scan_records[lane] = digit;
    workgroupBarrier();
    if id.x >= config.spatial.w { return; }
    var rank = atomicLoad(&state.words[cutoff_hist() + group.x * 256u + digit]);
    for (var previous = 0u; previous < lane; previous += 1u) { rank += u32(scan_records[previous] == digit); }
    atomicStore(&state.words[output + rank * 2u], key);
    atomicStore(&state.words[output + rank * 2u + 1u], atomicLoad(&state.words[input + id.x * 2u + 1u]));
}

@compute @workgroup_size(1)
fn cutoff_next_radix() { atomicAdd(&state.words[config.spatial.z + 2u], 1u); }

@compute @workgroup_size(256)
fn cutoff_costs(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var cost = vec4<u32>(0u);
    if id.x < config.spatial.w && candidate_score(id.x) > 0.0 {
        let node = nodes[atomicLoad(&state.words[cutoff_a() + id.x * 2u + 1u])];
        cost = vec4<u32>(node.counts.y - node.counts.x, node.topology.y - 1u, node.topology.y, 0u);
    }
    scan_cost[lane] = cost;
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var previous = vec4<u32>(0u);
        if lane >= stride { previous = scan_cost[lane - stride]; }
        workgroupBarrier();
        scan_cost[lane] = add_cost(scan_cost[lane], previous);
        workgroupBarrier();
    }
    if id.x < config.spatial.w { store_cost(cutoff_cost_base() + id.x * 4u, scan_cost[lane]); }
    if lane == 255u { store_cost(cutoff_cost_groups() + group.x * 4u, scan_cost[lane]); }
}

@compute @workgroup_size(1)
fn cutoff_boundary() {
    var total = vec4<u32>(0u);
    var boundary = config.spatial.w;
    for (var group = 0u; group < cutoff_groups_count(); group += 1u) {
        let base = cutoff_cost_groups() + group * 4u;
        let next = add_cost(total, load_cost(base));
        if boundary == config.spatial.w && !fits(next) {
            for (var lane = 0u; lane < 256u; lane += 1u) {
                let index = group * 256u + lane;
                if index >= config.spatial.w { break; }
                let cost = add_cost(total, load_cost(cutoff_cost_base() + index * 4u));
                if !fits(cost) {
                    boundary = index;
                    if cost.x > config.limits.x - config.counts.w { atomicOr(&state.flags, RECORD_LIMIT); }
                    if cost.y > config.limits.y - config.counts.z { atomicOr(&state.flags, FRONTIER_LIMIT); }
                    if cost.z > config.limits.z - config.counts.z { atomicOr(&state.flags, VISIT_LIMIT); }
                    break;
                }
            }
        }
        store_cost(base, total);
        total = next;
    }
    var tau = 0.0;
    if boundary < config.spatial.w { tau = candidate_score(boundary); }
    // A threshold tie renders actual parents, never duplicated t=0 children.
    while boundary != 0u && candidate_score(boundary - 1u) <= tau { boundary -= 1u; }
    var admitted = vec4<u32>(0u);
    if boundary != 0u {
        let last = boundary - 1u;
        admitted = add_cost(load_cost(cutoff_cost_groups() + (last / 256u) * 4u),
            load_cost(cutoff_cost_base() + last * 4u));
    }
    atomicStore(&state.words[config.spatial.z], bitcast<u32>(tau));
    atomicStore(&state.words[config.spatial.z + 1u], 1u);
    atomicStore(&state.words[config.spatial.z + 3u], boundary);
    atomicStore(&state.active_records, config.counts.w + admitted.x);
    atomicStore(&state.active_nodes, config.counts.z + admitted.y);
    // Work admission covers the exhaustive graph, rather than reporting only
    // the smaller logical cut as if undisplayed candidate scoring were free.
    atomicStore(&state.visits, config.counts.x - config.counts.z);
}

@compute @workgroup_size(256)
fn cutoff_materialize(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= atomicLoad(&state.words[config.spatial.z + 3u]) { return; }
    let node = atomicLoad(&state.words[cutoff_a() + id.x * 2u + 1u]);
    set_node_bit(desired_base(), node);
}
