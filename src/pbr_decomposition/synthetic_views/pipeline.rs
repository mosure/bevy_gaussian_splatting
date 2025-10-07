use bevy::{
    asset::{load_internal_asset, uuid_handle},
    prelude::*,
    render::{
        render_resource::*,
        renderer::RenderDevice,
    },
};
use std::collections::HashMap;
use bevy::asset::AssetId;
use crate::gaussian::formats::planar_3d::PlanarGaussian3d;

use crate::pbr_decomposition::types::*;
use crate::render::{CloudPipeline, CloudPipelineKey, shader_defs};
use crate::gaussian::formats::planar_3d::Gaussian3d;

const SYNTHETIC_VIEWS_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("c3d4e5f6-a7b8-9012-cdef-123456789012");

#[derive(Resource)]
pub struct SyntheticViewsBuffers {
    pub asset_map: HashMap<AssetId<PlanarGaussian3d>, GpuSyntheticViewsBuffers>,
}

impl Default for SyntheticViewsBuffers {
    fn default() -> Self {
        SyntheticViewsBuffers {
            asset_map: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuSyntheticViewsBuffers {
    pub stats: Buffer,
}

impl GpuSyntheticViewsBuffers {
    pub fn new(count: usize, render_device: &RenderDevice) -> Self {
        let stats = render_device.create_buffer(&BufferDescriptor {
            label: Some("streaming_stats"),
            size: (count as u64) * StreamingStats::min_size().get(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self { stats }
    }
}

#[derive(Resource)]
pub struct SyntheticViewsPipeline {
    pub pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
    pub settings_layout: BindGroupLayout,
}

impl FromWorld for SyntheticViewsPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // Group 3: normals (read) + stats (read_write)
        let bind_group_layout = render_device.create_bind_group_layout(
            Some("synthetic_views_bind_group_layout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NormalData::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(StreamingStats::min_size()),
                    },
                    count: None,
                },
            ],
        );

        let settings_layout = render_device.create_bind_group_layout(
            Some("synthetic_views_settings_layout"),
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<super::super::orchestrator::SyntheticViewSettingsUniform>() as u64),
                },
                count: None,
            }],
        );

        let pipeline_cache = world.resource::<PipelineCache>();
        let cloud_pipeline = world.resource::<CloudPipeline<Gaussian3d>>();
        let shader_defs = shader_defs(CloudPipelineKey::default());

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("synthetic_views".into()),
            layout: vec![
                cloud_pipeline.compute_view_layout.clone(),
                cloud_pipeline.gaussian_uniform_layout.clone(),
                cloud_pipeline.gaussian_cloud_layout.clone(),
                bind_group_layout.clone(),
                settings_layout.clone(),
            ],
            push_constant_ranges: vec![],
            shader: SYNTHETIC_VIEWS_SHADER_HANDLE,
            shader_defs,
            entry_point: Some("evaluate_synthetic_views".into()),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            settings_layout,
        }
    }
}

pub fn load_synthetic_views_shader(app: &mut App) {
    load_internal_asset!(
        app,
        SYNTHETIC_VIEWS_SHADER_HANDLE,
        "../shaders/synthetic_views.wgsl",
        Shader::from_wgsl
    );
}
