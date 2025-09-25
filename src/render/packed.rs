use bevy::render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
        BindingType, Buffer, BufferBinding, BufferBindingType, BufferInitDescriptor, BufferSize,
        BufferUsages, ShaderStages,
    },
    renderer::RenderDevice,
};

use crate::{
    gaussian::{cloud::Cloud, packed::Gaussian},
    render::{CloudPipeline, GpuCloud},
};

#[derive(Debug, Clone)]
pub struct PackedBuffers {
    gaussians: Buffer,
}

pub fn prepare_cloud(render_device: &RenderDevice, cloud: &Cloud) -> PackedBuffers {
    let gaussians = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("packed_gaussian_cloud_buffer"),
        contents: bytemuck::cast_slice(cloud.gaussian_iter().collect::<Vec<Gaussian>>().as_slice()),
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    PackedBuffers { gaussians }
}

pub fn get_bind_group_layout(render_device: &RenderDevice, read_only: bool) -> BindGroupLayout {
    render_device.create_bind_group_layout(
        Some("packed_gaussian_cloud_layout"),
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::all(),
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(std::mem::size_of::<Gaussian>() as u64),
            },
            count: None,
        }],
    )
}

#[cfg(feature = "packed")]
pub fn get_bind_group(
    render_device: &RenderDevice,
    gaussian_cloud_pipeline: &CloudPipeline,
    cloud: &GpuCloud,
) -> BindGroup {
    render_device.create_bind_group(
        "packed_gaussian_cloud_bind_group",
        &gaussian_cloud_pipeline.gaussian_cloud_layout,
        &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &cloud.packed.gaussians,
                offset: 0,
                size: BufferSize::new(cloud.packed.gaussians.size()),
            }),
        }],
    )
}
