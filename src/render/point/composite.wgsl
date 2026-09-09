@group(0) @binding(0) var color: texture_2d<f32>;
@group(0) @binding(1) var<uniform> viewport: vec4<u32>;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(color, vec2<i32>(position.xy) - vec2<i32>(viewport.xy), 0);
}
