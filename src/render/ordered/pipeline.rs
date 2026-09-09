use super::{DRAW_SHADER, GATHER_SHADER, PROJECT_SHADER};
use crate::{
    Gaussian3d, RadixSortDepthBits,
    render::{
        CloudPipeline, CloudPipelineKey, ShaderDefines, shader_defs, shader_defs_with_defines,
    },
};
use bevy::{
    prelude::*,
    render::{render_resource::*, renderer::RenderDevice},
    shader::ShaderDefVal,
};
use std::collections::HashMap;

pub(super) const RADIX_STAGES: [&str; 6] = [
    "radix_reset",
    "radix_sort_active_a",
    "radix_sort_b",
    "radix_sort_c_count_tiles",
    "radix_sort_c_scan_tiles",
    "radix_sort_c_scatter",
];

#[derive(Resource, Default)]
pub(super) struct Pipelines(pub [Option<Pipeline>; 2]);

pub(super) struct Pipeline {
    pub project_layout: BindGroupLayout,
    pub gather_layout: BindGroupLayout,
    pub draw_layout: BindGroupLayout,
    pub radix_layout: BindGroupLayout,
    pub empty: BindGroup,
    pub project: [CachedComputePipelineId; 2],
    pub gather: [CachedComputePipelineId; 3],
    pub radix: [CachedComputePipelineId; 6],
    pub draws: HashMap<TextureFormat, CachedRenderPipelineId>,
    draw_desc: BindGroupLayoutDescriptor,
    spatial: bool,
}

fn entry(binding: u32, visibility: ShaderStages, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Pipeline {
    pub fn new(
        device: &RenderDevice,
        cache: &PipelineCache,
        cloud: &CloudPipeline<Gaussian3d>,
        spatial: bool,
    ) -> Self {
        let uniform = BufferBindingType::Uniform;
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let compute = ShaderStages::COMPUTE;
        let project_entries = [
            entry(0, compute, uniform),
            entry(1, compute, read),
            entry(2, compute, read),
            entry(3, compute, write),
            entry(4, compute, write),
        ];
        let project_desc = BindGroupLayoutDescriptor::new("global_quad_project", &project_entries);
        let project_layout =
            device.create_bind_group_layout(Some("global_quad_project"), &project_entries);
        let project = std::array::from_fn(|aabb| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("global_quad_project".into()),
                layout: vec![
                    cloud.compute_view_layout_desc.clone(),
                    cloud.gaussian_uniform_layout_desc.clone(),
                    cloud.gaussian_cloud_layout_desc.clone(),
                    project_desc.clone(),
                ],
                shader: PROJECT_SHADER,
                shader_defs: shader_defs(CloudPipelineKey {
                    aabb: aabb != 0,
                    ..default()
                })
                .into_iter()
                .chain(spatial_defines(spatial))
                .collect(),
                entry_point: Some("project".into()),
                ..default()
            })
        });
        let gather_entries = [
            entry(0, compute, uniform),
            entry(1, compute, read),
            entry(2, compute, write),
            entry(3, compute, write),
            entry(4, compute, write),
        ];
        let gather_desc = BindGroupLayoutDescriptor::new("global_quad_gather", &gather_entries);
        let gather_layout =
            device.create_bind_group_layout(Some("global_quad_gather"), &gather_entries);
        let gather = ["classify", "scan_groups", "gather"].map(|stage| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("global_quad_{stage}").into()),
                layout: vec![gather_desc.clone()],
                shader: GATHER_SHADER,
                shader_defs: spatial_defines(spatial),
                entry_point: Some(stage.into()),
                ..default()
            })
        });
        let draw_entries = [
            entry(0, ShaderStages::VERTEX, uniform),
            entry(1, ShaderStages::VERTEX, read),
            entry(2, ShaderStages::VERTEX, read),
        ];
        let draw_desc = BindGroupLayoutDescriptor::new("global_quad_draw", &draw_entries);
        let draw_layout = device.create_bind_group_layout(Some("global_quad_draw"), &draw_entries);
        let radix_entries = [
            entry(0, compute, uniform),
            entry(1, compute, write),
            entry(2, compute, write),
            entry(3, compute, read),
            entry(4, compute, write),
            entry(5, compute, write),
        ];
        let radix_desc = BindGroupLayoutDescriptor::new("global_quad_radix", &radix_entries);
        let radix_layout =
            device.create_bind_group_layout(Some("global_quad_radix"), &radix_entries);
        let empty_desc = BindGroupLayoutDescriptor::new("global_quad_empty", &[]);
        let empty_layout = device.create_bind_group_layout(Some("global_quad_empty"), &[]);
        let empty = device.create_bind_group("global_quad_empty", &empty_layout, &[]);
        let defines = shader_defs_with_defines(
            CloudPipelineKey::default(),
            ShaderDefines::for_radix_depth_bits(RadixSortDepthBits::Bits32),
        );
        let radix = RADIX_STAGES.map(|stage| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("global_quad_{stage}").into()),
                layout: vec![
                    empty_desc.clone(),
                    empty_desc.clone(),
                    empty_desc.clone(),
                    radix_desc.clone(),
                ],
                shader: crate::sort::radix::RADIX_SHADER_HANDLE,
                shader_defs: defines.clone(),
                entry_point: Some(stage.into()),
                ..default()
            })
        });
        Self {
            project_layout,
            gather_layout,
            draw_layout,
            radix_layout,
            empty,
            project,
            gather,
            radix,
            draws: default(),
            draw_desc,
            spatial,
        }
    }

    pub fn specialize(&mut self, format: TextureFormat, cache: &PipelineCache) {
        self.draws.entry(format).or_insert_with(|| {
            cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("globally_ordered_gaussian_quads".into()),
                layout: vec![self.draw_desc.clone()],
                vertex: VertexState {
                    shader: DRAW_SHADER,
                    shader_defs: spatial_defines(self.spatial),
                    entry_point: Some("vertex".into()),
                    ..default()
                },
                fragment: Some(FragmentState {
                    shader: DRAW_SHADER,
                    shader_defs: spatial_defines(self.spatial),
                    entry_point: Some("fragment".into()),
                    targets: vec![Some(ColorTargetState {
                        format,
                        blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleStrip,
                    cull_mode: None,
                    ..default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(CompareFunction::GreaterEqual),
                    stencil: default(),
                    bias: default(),
                }),
                ..default()
            })
        });
    }

    pub fn loaded(&self, format: TextureFormat, cache: &PipelineCache) -> bool {
        self.project
            .iter()
            .chain(&self.gather)
            .chain(&self.radix)
            .all(|id| cache.get_compute_pipeline(*id).is_some())
            && self
                .draws
                .get(&format)
                .is_some_and(|id| cache.get_render_pipeline(*id).is_some())
    }

    pub fn error(&self, cache: &PipelineCache) -> Option<String> {
        self.project
            .iter()
            .chain(&self.gather)
            .chain(&self.radix)
            .find_map(|id| match cache.get_compute_pipeline_state(*id) {
                CachedPipelineState::Err(error) => Some(error.to_string()),
                _ => None,
            })
    }
}

fn spatial_defines(spatial: bool) -> Vec<ShaderDefVal> {
    if spatial {
        vec!["LOD_SPATIAL_MORPH".into(), "LOD_MORPH".into()]
    } else {
        Vec::new()
    }
}
