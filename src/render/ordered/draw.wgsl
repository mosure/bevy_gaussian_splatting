#import bevy_gaussian_splatting::support::gaussian_support_weight
#import bevy_gaussian_splatting::ordered_types::{OrderedGaussian, OrderedEntry, OrderedConfig}
#ifdef LOD_SPATIAL_MORPH
    #import bevy_gaussian_splatting::lod_morph::lod_morph_fragment_color
#endif
@group(0) @binding(0) var<uniform> config: OrderedConfig;
@group(0) @binding(1) var<storage, read> projected: array<OrderedGaussian>;
@group(0) @binding(2) var<storage, read> entries: array<OrderedEntry>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) mean: vec2<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
    @location(5) @interpolate(flat) cutoff_squared: f32,
    #ifdef LOD_SPATIAL_MORPH
        @location(3) @interpolate(flat) parent_color: vec4<f32>,
        @location(4) @interpolate(flat) morph: vec4<f32>,
    #endif
}

@vertex
fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let record = projected[entries[instance].value];
    let corner = vec2<f32>(select(-1.0, 1.0, (vertex & 2u) != 0u), select(-1.0, 1.0, (vertex & 1u) != 0u));
    let mean = (record.mean - vec2<f32>(config.viewport.xy)) / vec2<f32>(config.viewport.zw);
    let position = mean * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0)
        + record.axis_x * corner.x + record.axis_y * corner.y;
    return VertexOutput(vec4<f32>(position, record.depth, 1.0), record.mean,
        vec3<f32>(record.conic_xy, record.conic_z), record.color, record.cutoff_squared
        #ifdef LOD_SPATIAL_MORPH
            , record.parent_color, record.morph
        #endif
    );
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    let d = 2.0 * (input.position.xy - input.mean);
    let power = -0.5 * (input.conic.x * d.x * d.x + 2.0 * input.conic.y * d.x * d.y + input.conic.z * d.y * d.y);
    if power > 0.0 { discard; }
    let gaussian_weight = gaussian_support_weight(-2.0 * power, input.cutoff_squared);
    #ifdef LOD_SPATIAL_MORPH
        if input.morph.w != 0.0 {
            return lod_morph_fragment_color(input.parent_color.a, input.color.a,
                gaussian_weight, input.morph.z, input.morph.x, input.morph.y,
                input.parent_color.rgb, input.color.rgb);
        }
    #endif
    let alpha = min(gaussian_weight * input.color.a, 0.999);
    return vec4<f32>(input.color.rgb * alpha, alpha);
}
