#define_import_path bevy_gaussian_splatting::pbr_decomposition::spatial_hash

struct SpatialHashConfig {
    cell_size: f32,
    table_size: u32,
    gaussian_count: u32,
    _pad: u32,
}

struct GridCell {
    start: u32,
    count: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> cell_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> cell_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_ranges: array<GridCell>;

@group(1) @binding(0) var<uniform> config: SpatialHashConfig;

fn hash_position(pos: vec3<f32>) -> u32 {
    let cell = vec3<i32>(floor(pos / config.cell_size));
    let p1 = u32(cell.x) * 73856093u;
    let p2 = u32(cell.y) * 19349663u;
    let p3 = u32(cell.z) * 83492791u;
    return (p1 ^ p2 ^ p3) % config.table_size;
}

@compute @workgroup_size(256)
fn compute_cell_keys(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= config.gaussian_count) { return; }

    let pos = positions[idx];
    let hash = hash_position(pos);

    cell_keys[idx] = hash;
    cell_indices[idx] = idx;
}

@compute @workgroup_size(256)
fn build_cell_ranges(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= config.gaussian_count) { return; }

    let key = cell_keys[idx];
    let prev_key = select(0xFFFFFFFFu, cell_keys[idx - 1u], idx > 0u);

    if (key != prev_key) {
        cell_ranges[key].start = idx;

        if (prev_key != 0xFFFFFFFFu) {
            let prev_start = cell_ranges[prev_key].start;
            cell_ranges[prev_key].count = idx - prev_start;
        }
    }

    if (idx == config.gaussian_count - 1u) {
        let start = cell_ranges[key].start;
        cell_ranges[key].count = config.gaussian_count - start;
    }
}

fn query_neighbors_27(
    query_pos: vec3<f32>,
    max_neighbors: u32,
    radius: f32,
    neighbors: ptr<function, array<u32, 64>>
) -> u32 {
    let query_cell = vec3<i32>(floor(query_pos / config.cell_size));
    var count = 0u;

    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let neighbor_cell = query_cell + vec3<i32>(dx, dy, dz);

                let cell_pos = vec3<f32>(neighbor_cell) * config.cell_size;
                let hash = hash_position(cell_pos);

                let range = cell_ranges[hash];

                for (var i = 0u; i < range.count; i++) {
                    if (count >= max_neighbors) { return count; }

                    let candidate_idx = cell_indices[range.start + i];
                    let candidate_pos = positions[candidate_idx];

                    let diff = query_pos - candidate_pos;
                    let dist_sq = dot(diff, diff);

                    if (dist_sq <= radius * radius) {
                        (*neighbors)[count] = candidate_idx;
                        count++;
                    }
                }
            }
        }
    }

    return count;
}
