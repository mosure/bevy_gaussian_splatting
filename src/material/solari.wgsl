#define_import_path bevy_gaussian_splatting::solari

#import bevy_render::maths::PI

const SOLARI_WORLD_CACHE_MAX_SEARCH_STEPS: u32 = 3u;
const SOLARI_WORLD_CACHE_POSITION_BASE_CELL_SIZE: f32 = 0.25;
const SOLARI_WORLD_CACHE_POSITION_LOD_SCALE: f32 = 30.0;
const SOLARI_WORLD_CACHE_EMPTY_CELL: u32 = 0u;
const SOLARI_WORLD_CACHE_SIZE: u32 = #{SOLARI_WORLD_CACHE_SIZE};
const SOLARI_WORLD_CACHE_MASK: u32 = SOLARI_WORLD_CACHE_SIZE - 1u;

@group(4) @binding(0) var<storage, read> solari_world_cache_checksums:
    array<atomic<u32>, #{SOLARI_WORLD_CACHE_SIZE}>;
@group(4) @binding(1) var<storage, read> solari_world_cache_radiance:
    array<vec4<f32>, #{SOLARI_WORLD_CACHE_SIZE}>;

fn solari_shade_diffuse(
    base_color: vec3<f32>,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    view_position: vec3<f32>,
    exposure: f32,
) -> vec3<f32> {
    let radiance = solari_world_cache_lookup(world_position, world_normal, view_position);
    if all(radiance == vec3<f32>(0.0)) {
        return base_color;
    }

    let diffuse_brdf = base_color / PI;
    return radiance * exposure * diffuse_brdf;
}

fn solari_world_cache_lookup(
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    view_position: vec3<f32>,
) -> vec3<f32> {
    let cell_size = solari_get_cell_size(world_position, view_position);
    let world_position_quantized =
        bitcast<vec3<u32>>(solari_quantize_position(world_position, cell_size));
    let world_normal_quantized =
        bitcast<vec3<u32>>(solari_quantize_normal(world_normal));

    var key = solari_compute_key(world_position_quantized, world_normal_quantized);
    let checksum = solari_compute_checksum(world_position_quantized, world_normal_quantized);

    for (var i = 0u; i < SOLARI_WORLD_CACHE_MAX_SEARCH_STEPS; i = i + 1u) {
        let existing_checksum = atomicLoad(&solari_world_cache_checksums[key]);
        if existing_checksum == checksum {
            return solari_world_cache_radiance[key].rgb;
        }

        if existing_checksum == SOLARI_WORLD_CACHE_EMPTY_CELL {
            return vec3<f32>(0.0);
        }

        key = solari_wrap_key(solari_pcg_hash(key));
    }

    return vec3<f32>(0.0);
}

fn solari_get_cell_size(world_position: vec3<f32>, view_position: vec3<f32>) -> f32 {
    let camera_distance = distance(view_position, world_position) /
        SOLARI_WORLD_CACHE_POSITION_LOD_SCALE;
    let lod = exp2(floor(log2(1.0 + camera_distance)));
    return SOLARI_WORLD_CACHE_POSITION_BASE_CELL_SIZE * lod;
}

fn solari_quantize_position(
    world_position: vec3<f32>,
    quantization_factor: f32,
) -> vec3<f32> {
    return floor(world_position / quantization_factor + 0.0001);
}

fn solari_quantize_normal(world_normal: vec3<f32>) -> vec3<f32> {
    return floor(normalize(world_normal) + 0.0001);
}

fn solari_compute_key(world_position: vec3<u32>, world_normal: vec3<u32>) -> u32 {
    var key = solari_pcg_hash(world_position.x);
    key = solari_pcg_hash(key + world_position.y);
    key = solari_pcg_hash(key + world_position.z);
    key = solari_pcg_hash(key + world_normal.x);
    key = solari_pcg_hash(key + world_normal.y);
    key = solari_pcg_hash(key + world_normal.z);
    return solari_wrap_key(key);
}

fn solari_compute_checksum(world_position: vec3<u32>, world_normal: vec3<u32>) -> u32 {
    var key = solari_iqint_hash(world_position.x);
    key = solari_iqint_hash(key + world_position.y);
    key = solari_iqint_hash(key + world_position.z);
    key = solari_iqint_hash(key + world_normal.x);
    key = solari_iqint_hash(key + world_normal.y);
    key = solari_iqint_hash(key + world_normal.z);
    return key;
}

fn solari_wrap_key(key: u32) -> u32 {
    return key & SOLARI_WORLD_CACHE_MASK;
}

fn solari_pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn solari_iqint_hash(input: u32) -> u32 {
    let n = (input << 13u) ^ input;
    return n * (n * n * 15731u + 789221u) + 1376312589u;
}
