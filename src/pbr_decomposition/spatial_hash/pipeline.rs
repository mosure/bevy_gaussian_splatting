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

use crate::pbr_decomposition::types::*;

const SPATIAL_HASH_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("a1b2c3d4-e5f6-7890-abcd-ef1234567890");

#[derive(Resource)]
pub struct SpatialHashBuffers {
    pub asset_map: HashMap<AssetId<()>, GpuSpatialHashBuffers>,
}

impl Default for SpatialHashBuffers {
    fn default() -> Self {
        SpatialHashBuffers {
            asset_map: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuSpatialHashBuffers {
    pub cell_keys: Buffer,
    pub cell_indices: Buffer,
    pub cell_ranges: Buffer,
    pub config: SpatialHashConfig,
}

impl GpuSpatialHashBuffers {
    pub fn new(config: SpatialHashConfig, render_device: &RenderDevice) -> Self {
        let count = config.gaussian_count as usize;
        let table_size = config.table_size as usize;

        let cell_keys = render_device.create_buffer(&BufferDescriptor {
            label: Some("spatial hash cell keys"),
            size: (count * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cell_indices = render_device.create_buffer(&BufferDescriptor {
            label: Some("spatial hash cell indices"),
            size: (count * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cell_ranges = render_device.create_buffer(&BufferDescriptor {
            label: Some("spatial hash cell ranges"),
            size: (table_size * std::mem::size_of::<GridCell>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            cell_keys,
            cell_indices,
            cell_ranges,
            config,
        }
    }
}

#[derive(Resource)]
pub struct SpatialHashPipeline {
    pub compute_keys_pipeline: CachedComputePipelineId,
    pub build_ranges_pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
    pub config_layout: BindGroupLayout,
}

impl FromWorld for SpatialHashPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            Some("spatial_hash_bind_group_layout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let config_layout = render_device.create_bind_group_layout(
            Some("spatial_hash_config_layout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(std::mem::size_of::<SpatialHashConfig>() as u64),
                    },
                    count: None,
                },
            ],
        );

        let pipeline_cache = world.resource::<PipelineCache>();

        let compute_keys_pipeline = pipeline_cache.queue_compute_pipeline(
            ComputePipelineDescriptor {
                label: Some("spatial_hash_compute_keys".into()),
                layout: vec![bind_group_layout.clone(), config_layout.clone()],
                push_constant_ranges: vec![],
                shader: SPATIAL_HASH_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: Some("compute_cell_keys".into()),
                zero_initialize_workgroup_memory: false,
            },
        );

        let build_ranges_pipeline = pipeline_cache.queue_compute_pipeline(
            ComputePipelineDescriptor {
                label: Some("spatial_hash_build_ranges".into()),
                layout: vec![bind_group_layout.clone(), config_layout.clone()],
                push_constant_ranges: vec![],
                shader: SPATIAL_HASH_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: Some("build_cell_ranges".into()),
                zero_initialize_workgroup_memory: false,
            },
        );

        Self {
            compute_keys_pipeline,
            build_ranges_pipeline,
            bind_group_layout,
            config_layout,
        }
    }
}

pub fn load_spatial_hash_shader(app: &mut App) {
    load_internal_asset!(
        app,
        SPATIAL_HASH_SHADER_HANDLE,
        "../shaders/spatial_hash.wgsl",
        Shader::from_wgsl
    );
}
