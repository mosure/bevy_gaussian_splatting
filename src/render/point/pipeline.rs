//! Lazy point-renderer pipeline compilation and target specialization.
//!
//! The resource starts empty, including after device recovery. Creating the
//! first valid point camera is the only path that allocates layouts and queues
//! shaders; ordinary Gaussian applications pay neither cost.

use super::{COMPOSITE_SHADER, PROJECT_SHADER, WORK_SHADER};
use crate::{
    Gaussian3d,
    render::{CloudPipeline, CloudPipelineKey, shader_defs},
};
use bevy::{
    prelude::*,
    render::{render_resource::*, renderer::RenderDevice},
};
use std::collections::HashMap;

#[derive(Resource, Default)]
pub(super) struct PointPipeline(Option<PointPipelines>);

impl PointPipeline {
    pub(super) fn get(&self) -> Option<&PointPipelines> {
        self.0.as_ref()
    }

    pub(super) fn initialize(
        &mut self,
        device: &RenderDevice,
        cache: &PipelineCache,
        cloud: &CloudPipeline<Gaussian3d>,
    ) -> &mut PointPipelines {
        self.0
            .get_or_insert_with(|| PointPipelines::new(device, cache, cloud))
    }
}

pub(super) const STAGES: [&str; 8] = [
    "reset",
    "count",
    "scan_groups",
    "scan_blocks",
    "add_offsets",
    "depth",
    "winner",
    "resolve",
];

pub(super) struct PointPipelines {
    pub(super) project_layout: BindGroupLayout,
    pub(super) work_layout: BindGroupLayout,
    pub(super) composite_layout: BindGroupLayout,
    pub(super) project: CachedComputePipelineId,
    pub(super) work: [CachedComputePipelineId; 8],
    composite_layout_desc: BindGroupLayoutDescriptor,
    pub(super) composites: HashMap<TextureFormat, CachedRenderPipelineId>,
}

fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl PointPipelines {
    fn new(
        device: &RenderDevice,
        cache: &PipelineCache,
        cloud: &CloudPipeline<Gaussian3d>,
    ) -> Self {
        let uniform = BufferBindingType::Uniform;
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let project_entries = [
            buffer_entry(0, uniform),
            buffer_entry(1, read),
            buffer_entry(2, read),
            buffer_entry(3, write),
        ];
        let project_desc = BindGroupLayoutDescriptor::new("point_project", &project_entries);
        let project_layout =
            device.create_bind_group_layout(Some("point_project"), &project_entries);
        let project = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("gaussian_point_project".into()),
            layout: vec![
                cloud.compute_view_layout_desc.clone(),
                cloud.gaussian_uniform_layout_desc.clone(),
                cloud.gaussian_cloud_layout_desc.clone(),
                project_desc,
            ],
            shader: PROJECT_SHADER,
            shader_defs: shader_defs(CloudPipelineKey::default()),
            entry_point: Some("project".into()),
            ..default()
        });
        let work_entries = [
            buffer_entry(0, uniform),
            buffer_entry(1, read),
            buffer_entry(2, write),
            buffer_entry(3, write),
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ];
        let work_desc = BindGroupLayoutDescriptor::new("point_work", &work_entries);
        let work_layout = device.create_bind_group_layout(Some("point_work"), &work_entries);
        let work = STAGES.map(|stage| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("gaussian_point_{stage}").into()),
                layout: vec![work_desc.clone()],
                shader: WORK_SHADER,
                entry_point: Some(stage.into()),
                ..default()
            })
        });
        let composite_entries = [
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                visibility: ShaderStages::FRAGMENT,
                ..buffer_entry(1, uniform)
            },
        ];
        Self {
            project_layout,
            work_layout,
            composite_layout: device
                .create_bind_group_layout(Some("point_composite"), &composite_entries),
            composite_layout_desc: BindGroupLayoutDescriptor::new(
                "point_composite",
                &composite_entries,
            ),
            project,
            work,
            composites: HashMap::new(),
        }
    }
}

impl PointPipelines {
    pub(super) fn composite(
        &mut self,
        format: TextureFormat,
        cache: &PipelineCache,
    ) -> CachedRenderPipelineId {
        *self.composites.entry(format).or_insert_with(|| {
            cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("gaussian_point_composite".into()),
                layout: vec![self.composite_layout_desc.clone()],
                vertex: VertexState {
                    shader: COMPOSITE_SHADER,
                    entry_point: Some("vertex".into()),
                    ..default()
                },
                fragment: Some(FragmentState {
                    shader: COMPOSITE_SHADER,
                    entry_point: Some("fragment".into()),
                    targets: vec![Some(ColorTargetState {
                        format,
                        blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                    ..default()
                }),
                ..default()
            })
        })
    }

    pub(super) fn loaded(&self, format: TextureFormat, cache: &PipelineCache) -> bool {
        cache.get_compute_pipeline(self.project).is_some()
            && self
                .work
                .iter()
                .all(|id| cache.get_compute_pipeline(*id).is_some())
            && self
                .composites
                .get(&format)
                .is_some_and(|id| cache.get_render_pipeline(*id).is_some())
    }
}
