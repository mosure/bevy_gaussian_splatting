#[allow(unused_imports)]
use bevy::render::{
    render_resource::*,
    renderer::RenderDevice,
};

#[allow(unused_imports)]
use crate::{
    gaussian::{
        cloud::GaussianCloud,
        f32::{
            PositionVisibility,
            Rotation,
            ScaleOpacity,
        },
    },
    render::{
        GaussianCloudPipeline,
        GpuGaussianCloud,
    },
    material::spherical_harmonics::SphericalHarmonicCoefficients,
};

#[cfg(feature = "f16")]
use crate::gaussian::f16::RotationScaleOpacityPacked128;

#[derive(Debug, Clone)]
pub struct PlanarBuffers {
    position_visibility: Buffer,
    spherical_harmonics: Buffer,

    #[cfg(feature = "f16")]
    rotation_scale_opacity: Buffer,

    #[cfg(feature = "f32")]
    rotation: Buffer,
    #[cfg(feature = "f32")]
    scale_opacity: Buffer,
}


#[cfg(feature = "f16")]
pub fn prepare_cloud(
    render_device: &RenderDevice,
    cloud: &GaussianCloud,
) -> PlanarBuffers {
    let position_visibility = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("planar_position_visibility_buffer"),
        contents: bytemuck::cast_slice(cloud.position_visibility.as_slice()),
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    let rotation_scale_opacity = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("planar_rotation_scale_opacity_buffer"),
        contents: bytemuck::cast_slice(cloud.rotation_scale_opacity_packed128.as_slice()),
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    let spherical_harmonics = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("planar_spherical_harmonics_buffer"),
        contents: bytemuck::cast_slice(cloud.spherical_harmonic.as_slice()),
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    PlanarBuffers {
        position_visibility,
        rotation_scale_opacity,
        spherical_harmonics,
    }
}


#[cfg(feature = "f32")]
pub fn prepare_cloud(
    render_device: &RenderDevice,
    cloud: &GaussianCloud,
) -> PlanarBuffers {
    let position_visibility = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("planar_f32_position_visibility_buffer"),
        contents: bytemuck::cast_slice(cloud.position_visibility.as_slice()),
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    let rotation = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("planar_f32_rotation_buffer"),
        contents: bytemuck::cast_slice(cloud.rotation.as_slice()),
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    let scale_opacity = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("planar_f32_scale_opacity_buffer"),
        contents: bytemuck::cast_slice(cloud.scale_opacity.as_slice()),
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    let spherical_harmonics = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("planar_f32_spherical_harmonics_buffer"),
        contents: bytemuck::cast_slice(cloud.spherical_harmonic.as_slice()),
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    });

    PlanarBuffers {
        position_visibility,
        rotation,
        scale_opacity,
        spherical_harmonics,
    }
}


#[cfg(feature = "f16")]
pub fn get_bind_group_layout(
    render_device: &RenderDevice,
    read_only: bool
) -> BindGroupLayout {
    render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("planar_f16_gaussian_cloud_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<PositionVisibility>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<SphericalHarmonicCoefficients>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<RotationScaleOpacityPacked128>() as u64),
                },
                count: None,
            },
        ],
    })
}


#[cfg(feature = "f32")]
pub fn get_bind_group_layout(
    render_device: &RenderDevice,
    read_only: bool
) -> BindGroupLayout {
    render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("planar_f32_gaussian_cloud_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<PositionVisibility>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<SphericalHarmonicCoefficients>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<Rotation>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<ScaleOpacity>() as u64),
                },
                count: None,
            },
        ],
    })
}


#[cfg(all(feature = "planar", feature = "f16"))]
pub fn get_bind_group(
    render_device: &RenderDevice,
    gaussian_cloud_pipeline: &GaussianCloudPipeline,
    cloud: &GpuGaussianCloud,
) -> BindGroup {
    render_device.create_bind_group(
        "planar_gaussian_cloud_bind_group",
        &gaussian_cloud_pipeline.gaussian_cloud_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &cloud.planar.position_visibility,
                    offset: 0,
                    size: BufferSize::new(cloud.planar.position_visibility.size()),
                }),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &cloud.planar.spherical_harmonics,
                    offset: 0,
                    size: BufferSize::new(cloud.planar.spherical_harmonics.size()),
                }),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &cloud.planar.rotation_scale_opacity,
                    offset: 0,
                    size: BufferSize::new(cloud.planar.rotation_scale_opacity.size()),
                }),
            },
        ],
    )
}


#[cfg(all(feature = "planar", feature = "f32"))]
pub fn get_bind_group(
    render_device: &RenderDevice,
    gaussian_cloud_pipeline: &GaussianCloudPipeline,
    cloud: &GpuGaussianCloud,
) -> BindGroup {
    render_device.create_bind_group(
        "planar_gaussian_cloud_bind_group",
        &gaussian_cloud_pipeline.gaussian_cloud_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &cloud.planar.position_visibility,
                    offset: 0,
                    size: BufferSize::new(cloud.planar.position_visibility.size()),
                }),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &cloud.planar.spherical_harmonics,
                    offset: 0,
                    size: BufferSize::new(cloud.planar.spherical_harmonics.size()),
                }),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &cloud.planar.rotation,
                    offset: 0,
                    size: BufferSize::new(cloud.planar.rotation.size()),
                }),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &cloud.planar.scale_opacity,
                    offset: 0,
                    size: BufferSize::new(cloud.planar.scale_opacity.size()),
                }),
            },
        ],
    )
}
