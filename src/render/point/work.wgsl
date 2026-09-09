#import bevy_gaussian_splatting::support::gaussian_support_weight
#import bevy_gaussian_splatting::point_types::{PointGaussian, PointConfig}
#import bevy_gaussian_splatting::point_sampling::{point_poisson_process, point_random, point_radius, point_intensity, point_rectangle}

struct PointWork {
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
    total: u32,
    failure: atomic<u32>,
    projected_count: atomic<u32>,
    padding: vec2<u32>,
    words: array<u32>,
}
struct PixelSample { depth: atomic<u32>, winner: atomic<u32> }

@group(0) @binding(0) var<uniform> config: PointConfig;
@group(0) @binding(1) var<storage, read> records: array<PointGaussian>;
@group(0) @binding(2) var<storage, read_write> work: PointWork;
@group(0) @binding(3) var<storage, read_write> pixels: array<PixelSample>;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var scene_depth: texture_depth_2d;

var<workgroup> scan: array<u32, 256>;

fn scan_sum(value: u32, lane: u32) -> u32 {
    scan[lane] = min(value, config.max_points + 1u);
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset *= 2u) {
        var left = 0u;
        if lane >= offset { left = scan[lane - offset]; }
        workgroupBarrier();
        scan[lane] += min(left, config.max_points + 1u - scan[lane]);
        workgroupBarrier();
    }
    return scan[lane];
}

fn group_word(group: u32) -> u32 { return config.item_count + group * 2u; }
fn block_word(block: u32) -> u32 { return config.item_count + config.group_count * 2u + block * 2u; }

@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x == 0u && id.y == 0u {
        work.dispatch_x = 0u; work.dispatch_y = 1u; work.dispatch_z = 1u;
        work.total = 0u;
        atomicStore(&work.failure, 0u);
        atomicStore(&work.projected_count, 0u);
    }
    let stride = 65535u * 256u;
    let index = id.x + id.y * stride;
    if index < config.width * config.height * config.samples {
        atomicStore(&pixels[index].depth, 0u);
        atomicStore(&pixels[index].winner, 0xffffffffu);
    }
}

@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let index = id.x;
    var points = 0u;
    if index < config.item_count {
        let record_index = index / config.samples;
        let layer = index % config.samples;
        let record = records[record_index];
        if record.epoch == config.frame && record.failure != 0u {
            atomicOr(&work.failure, record.failure);
        }
        if record.epoch == config.frame && record.opacity > 0.0 {
            if layer == 0u { atomicAdd(&work.projected_count, 1u); }
            let seed = record.seed ^ config.seed ^ (layer * 0x85ebca6bu) ^ (config.noise_frame * 0xc2b2ae35u);
            points = point_poisson_process(record.proposal_rate, seed, config.max_points);
            if points == 0xffffffffu {
                atomicOr(&work.failure, 2u);
                points = 0u;
            }
            // Saturated prefix sums cannot overflow: a single rate above the
            // total budget invalidates the whole image, not a suffix of clouds.
            if points > config.max_points {
                atomicOr(&work.failure, 1u);
                points = config.max_points + 1u;
            }
        }
    }
    let inclusive = scan_sum(points, lane);
    if index < config.item_count { work.words[index] = inclusive; }
    if lane == 255u { work.words[group_word(group.x)] = inclusive; }
}

@compute @workgroup_size(256)
fn scan_groups(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var value = 0u;
    if id.x < config.group_count { value = work.words[group_word(id.x)]; }
    let inclusive = scan_sum(value, lane);
    if id.x < config.group_count { work.words[group_word(id.x) + 1u] = inclusive - value; }
    if lane == 255u { work.words[block_word(group.x)] = inclusive; }
}

@compute @workgroup_size(256)
fn scan_blocks(@builtin(local_invocation_index) lane: u32) {
    var value = 0u;
    if lane < config.block_count { value = work.words[block_word(lane)]; }
    let inclusive = scan_sum(value, lane);
    if lane < config.block_count { work.words[block_word(lane) + 1u] = inclusive - value; }
    if lane == 255u {
        work.total = inclusive;
        if inclusive > config.max_points { atomicOr(&work.failure, 1u); }
        if atomicLoad(&work.failure) == 0u {
            let groups = (inclusive + 255u) / 256u;
            work.dispatch_x = min(groups, 65535u);
            work.dispatch_y = max((groups + 65534u) / 65535u, 1u);
        }
    }
}

@compute @workgroup_size(256)
fn add_offsets(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= config.item_count { return; }
    let group = id.x / 256u;
    if atomicLoad(&work.failure) == 0u {
        work.words[id.x] += work.words[group_word(group) + 1u]
            + work.words[block_word(group / 256u) + 1u];
    }
}

struct PointHit { pixel: u32, record: u32, depth: u32 }

fn sample_point(point: u32) -> PointHit {
    var miss = PointHit(0xffffffffu, 0u, 0u);
    if point >= work.total || atomicLoad(&work.failure) != 0u { return miss; }
    // Inclusive prefix maps one invocation to exactly one point. This costs
    // logarithmic reads but needs no point-count-sized allocation or CPU list.
    var low = 0u;
    var high = config.item_count;
    while low < high {
        let mid = low + (high - low) / 2u;
        if work.words[mid] <= point { low = mid + 1u; } else { high = mid; }
    }
    if low >= config.item_count { return miss; }
    var start = 0u;
    if low > 0u { start = work.words[low - 1u]; }
    let record_index = low / config.samples;
    let layer = low % config.samples;
    let record = records[record_index];
    let seed = record.seed ^ config.seed ^ (layer * 0x85ebca6bu) ^ (config.noise_frame * 0xc2b2ae35u);
    var position: vec2<f32>;
    if record.proposal_peak > 0.0 {
        let counter = (point - start) * 3u + 4096u;
        let rectangle = point_rectangle(record.mean, record.basis, record.basis_y,
            vec2<f32>(f32(config.width), f32(config.height)));
        position = rectangle.xy + (rectangle.zw - rectangle.xy)
            * vec2<f32>(point_random(seed, counter), point_random(seed, counter + 1u));
        let delta = position - record.mean;
        let x = delta.x / record.basis.x;
        let y = (delta.y - record.basis.y * x) / record.basis_y;
        let radius_squared = x * x + y * y;
        if radius_squared > 9.0 { return miss; }
        // Homogeneous Poisson envelope thinned to the same corrected intensity
        // as the radial process. Rejections are never replaced or renormalized.
        let intensity = point_intensity(record.opacity * gaussian_support_weight(radius_squared, 9.0));
        if point_random(seed, counter + 2u) * record.proposal_peak >= intensity { return miss; }
    } else {
        let counter = (point - start) * 3u + 4096u;
        let radius = point_radius(record.opacity, record.dilog, point_random(seed, counter));
        if radius >= 3.0 { return miss; }
        let radius_squared = radius * radius;
        if radius_squared > 8.0 {
            // The analytic radial process is an envelope. Independent thinning
            // gives the shared finite kernel without replacing rejected points.
            let envelope = point_intensity(record.opacity * exp(-0.5 * radius_squared));
            let density_target = point_intensity(record.opacity * gaussian_support_weight(radius_squared, 9.0));
            if point_random(seed, counter + 2u) * envelope >= density_target { return miss; }
        }
        let angle = 6.283185307179586 * point_random(seed, counter + 1u);
        let disk = radius * vec2<f32>(cos(angle), sin(angle));
        position = record.mean + vec2<f32>(
            record.basis.x * disk.x, record.basis.y * disk.x + record.basis_y * disk.y);
    }
    if any(position < vec2<f32>(0.0)) || any(position >= vec2<f32>(f32(config.width), f32(config.height))) { return miss; }
    let xy = vec2<u32>(position);
    let pixel = (layer * config.height + xy.y) * config.width + xy.x;
    return PointHit(pixel, record_index, bitcast<u32>(record.depth));
}

@compute @workgroup_size(256)
fn depth(@builtin(global_invocation_id) id: vec3<u32>) {
    let hit = sample_point(id.x + id.y * 65535u * 256u);
    if hit.pixel != 0xffffffffu { atomicMax(&pixels[hit.pixel].depth, hit.depth); }
}

@compute @workgroup_size(256)
fn winner(@builtin(global_invocation_id) id: vec3<u32>) {
    let hit = sample_point(id.x + id.y * 65535u * 256u);
    if hit.pixel != 0xffffffffu && atomicLoad(&pixels[hit.pixel].depth) == hit.depth {
        atomicMin(&pixels[hit.pixel].winner, hit.record);
    }
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= config.width || id.y >= config.height || atomicLoad(&work.failure) != 0u { return; }
    let opaque = textureLoad(scene_depth, vec2<i32>(id.xy + config.viewport_origin), 0);
    var color = vec4<f32>(0.0);
    for (var layer = 0u; layer < config.samples; layer += 1u) {
        let pixel = (layer * config.height + id.y) * config.width + id.x;
        let selected = atomicLoad(&pixels[pixel].winner);
        if selected != 0xffffffffu {
            let record = records[selected];
            if record.depth >= opaque { color += vec4<f32>(record.color, 1.0); }
        }
    }
    textureStore(output, vec2<i32>(id.xy), color / f32(config.samples));
}
