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

const MATERIAL_SEPARATION_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("d4e5f6a7-b8c9-0123-def0-234567890123");

#[derive(Resource)]
pub struct MaterialSeparationBuffers {
    pub asset_map: HashMap<AssetId<PlanarGaussian3d>, GpuMaterialBuffers>,
}

impl Default for MaterialSeparationBuffers {
    fn default() -> Self {
        MaterialSeparationBuffers {
            asset_map: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuMaterialBuffers {
    pub materials: Buffer,
}

impl GpuMaterialBuffers {
    pub fn new(count: usize, render_device: &RenderDevice) -> Self {
        let materials = render_device.create_buffer(&BufferDescriptor {
            label: Some("pbr_materials"),
            size: (count as u64) * PbrMaterialData::min_size().get(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self { materials }
    }
}

#[derive(Resource)]
pub struct MaterialSeparationPipeline {
    pub pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
    pub settings_layout: BindGroupLayout,
}

impl FromWorld for MaterialSeparationPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            Some("material_separation_bind_group_layout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(StreamingStats::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(PbrMaterialData::min_size()),
                    },
                    count: None,
                },
            ],
        );

        let settings_layout = render_device.create_bind_group_layout(
            Some("material_separation_settings_layout"),
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<super::super::orchestrator::MaterialSettingsUniform>() as u64),
                },
                count: None,
            }],
        );

        let pipeline_cache = world.resource::<PipelineCache>();
        let cloud_pipeline = world.resource::<CloudPipeline<Gaussian3d>>();
        let shader_defs = shader_defs(CloudPipelineKey::default());

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("material_separation".into()),
            layout: vec![
                cloud_pipeline.compute_view_layout.clone(),
                cloud_pipeline.gaussian_uniform_layout.clone(),
                cloud_pipeline.gaussian_cloud_layout.clone(),
                bind_group_layout.clone(),
                settings_layout.clone(),
                cloud_pipeline.gaussian_material_layout.clone(),
            ],
            push_constant_ranges: vec![],
            shader: MATERIAL_SEPARATION_SHADER_HANDLE,
            shader_defs,
            entry_point: Some("estimate_material_properties".into()),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            settings_layout,
        }
    }
}

pub fn load_material_separation_shader(app: &mut App) {
    load_internal_asset!(
        app,
        MATERIAL_SEPARATION_SHADER_HANDLE,
        "../shaders/material_separation.wgsl",
        Shader::from_wgsl
    );
}
