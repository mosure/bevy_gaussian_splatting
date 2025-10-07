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

const NORMAL_ESTIMATION_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("b2c3d4e5-f6a7-8901-bcde-f12345678901");

#[derive(Resource)]
pub struct NormalEstimationBuffers {
    pub asset_map: HashMap<AssetId<PlanarGaussian3d>, GpuNormalBuffers>,
}

impl Default for NormalEstimationBuffers {
    fn default() -> Self {
        NormalEstimationBuffers {
            asset_map: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuNormalBuffers {
    pub normals: Buffer,
}

impl GpuNormalBuffers {
    pub fn new(count: usize, render_device: &RenderDevice) -> Self {
        let normals = render_device.create_buffer(&BufferDescriptor {
            label: Some("gaussian normals"),
            size: (count as u64) * NormalData::min_size().get(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self { normals }
    }
}

#[derive(Resource)]
pub struct NormalEstimationPipeline {
    pub pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
    pub settings_layout: BindGroupLayout,
}

impl FromWorld for NormalEstimationPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // Group 3: output normals
        let bind_group_layout = render_device.create_bind_group_layout(
            Some("normal_estimation_bind_group_layout"),
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(NormalData::min_size()),
                },
                count: None,
            }],
        );

        let settings_layout = render_device.create_bind_group_layout(
            Some("normal_estimation_settings_layout"),
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<super::super::orchestrator::NormalSettingsUniform>() as u64),
                },
                count: None,
            }],
        );

        let pipeline_cache = world.resource::<PipelineCache>();
        let cloud_pipeline = world.resource::<CloudPipeline<Gaussian3d>>();
        let shader_defs = shader_defs(CloudPipelineKey::default());

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("normal_estimation".into()),
            layout: vec![
                cloud_pipeline.compute_view_layout.clone(),
                cloud_pipeline.gaussian_uniform_layout.clone(),
                cloud_pipeline.gaussian_cloud_layout.clone(),
                bind_group_layout.clone(),
                settings_layout.clone(),
            ],
            push_constant_ranges: vec![],
            shader: NORMAL_ESTIMATION_SHADER_HANDLE,
            shader_defs,
            entry_point: Some("estimate_normals".into()),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            settings_layout,
        }
    }
}

pub fn load_normal_estimation_shader(app: &mut App) {
    load_internal_asset!(
        app,
        NORMAL_ESTIMATION_SHADER_HANDLE,
        "../shaders/normal_estimation.wgsl",
        Shader::from_wgsl
    );
}
