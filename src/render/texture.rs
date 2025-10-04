use bevy::{
    asset::LoadState,
    ecs::query::QueryItem,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::{RenderAssetUsages, RenderAssets},
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
            BindingType, Extent3d, ShaderStages, TextureDimension, TextureFormat,
            TextureSampleType, TextureUsages, TextureViewDimension,
        },
        renderer::RenderDevice,
    },
};
use static_assertions::assert_cfg;

#[allow(unused_imports)]
use crate::{
    gaussian::{
        cloud::{Cloud, PlanarGaussian3dHandle},
        f32::{PositionVisibility, Rotation, ScaleOpacity},
    },
    material::spherical_harmonics::{
        SH_COEFF_COUNT, SH_VEC4_PLANES, SphericalHarmonicCoefficients,
    },
    render::{CloudPipeline, GpuCloud},
};

// TODO: support loading from directory of images

assert_cfg!(
    feature = "planar",
    "texture rendering is only supported with the `planar` feature enabled",
);

// TODO: migrate to auto-generated GPU buffers using bevy_interleave
// #[cfg(feature = "f16")]
// #[derive(Component, Clone, Debug, Reflect)]
// pub struct TextureBuffers {
//     position_visibility: Handle<Image>,
//     spherical_harmonics: Handle<Image>,

//     #[cfg(feature = "precompute_covariance_3d")]
//     covariance_3d_opacity: Handle<Image>,
//     #[cfg(not(feature = "precompute_covariance_3d"))]
//     rotation_scale_opacity: Handle<Image>,
// }

#[derive(Component, Clone, Debug, Reflect)]
pub struct TextureBuffers {
    position_visibility: Handle<Image>,
    spherical_harmonics: Handle<Image>,

    #[cfg(feature = "precompute_covariance_3d")]
    covariance_3d_opacity: Handle<Image>,
    #[cfg(not(feature = "precompute_covariance_3d"))]
    rotation: Handle<Image>,
    #[cfg(not(feature = "precompute_covariance_3d"))]
    scale_opacity: Handle<Image>,
}

impl ExtractComponent for TextureBuffers {
    type QueryData = &'static Self;

    type QueryFilter = ();
    type Out = Self;

    fn extract_component(texture_buffers: QueryItem<'_, Self::QueryData>) -> Option<Self::Out> {
        texture_buffers.clone().into()
    }
}

#[derive(Default)]
pub struct BufferTexturePlugin;

impl Plugin for BufferTexturePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<TextureBuffers>();
        app.add_plugins(ExtractComponentPlugin::<TextureBuffers>::default());

        app.add_systems(Update, queue_textures);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            queue_gpu_texture_buffers.in_set(RenderSystems::PrepareAssets),
        );
    }
}

#[derive(Component, Clone, Debug)]
pub struct GpuTextureBuffers {
    pub bind_group: BindGroup,
}

pub fn queue_gpu_texture_buffers(
    mut commands: Commands,
    // gaussian_cloud_pipeline: Res<CloudPipeline>,
    pipeline: ResMut<CloudPipeline>,
    render_device: ResMut<RenderDevice>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    clouds: Query<(Entity, &TextureBuffers)>,
) {
    // TODO: verify gpu_images are loaded

    for (entity, texture_buffers) in clouds.iter() {
        // #[cfg(feature = "f16")]
        // let bind_group = render_device.create_bind_group(
        //     Some("texture_gaussian_cloud_bind_group"),
        //     &pipeline.gaussian_cloud_layout,
        //     &[
        //         BindGroupEntry {
        //             binding: 0,
        //             resource: BindingResource::TextureView(
        //                 &gpu_images.get(&texture_buffers.position_visibility).unwrap().texture_view
        //             ),
        //         },
        //         BindGroupEntry {
        //             binding: 1,
        //             resource: BindingResource::TextureView(
        //                 &gpu_images.get(&texture_buffers.spherical_harmonics).unwrap().texture_view
        //             ),
        //         },
        //         #[cfg(feature = "precompute_covariance_3d")]
        //         BindGroupEntry {
        //             binding: 2,
        //             resource: BindingResource::TextureView(
        //                 &gpu_images.get(&texture_buffers.covariance_3d_opacity).unwrap().texture_view
        //             ),
        //         },
        //         #[cfg(not(feature = "precompute_covariance_3d"))]
        //         BindGroupEntry {
        //             binding: 2,
        //             resource: BindingResource::TextureView(
        //                 &gpu_images.get(&texture_buffers.rotation_scale_opacity).unwrap().texture_view
        //             ),
        //         },
        //     ],
        // );

        let bind_group = render_device.create_bind_group(
            Some("texture_gaussian_cloud_bind_group"),
            &pipeline.gaussian_cloud_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &gpu_images
                            .get(&texture_buffers.position_visibility)
                            .unwrap()
                            .texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &gpu_images
                            .get(&texture_buffers.spherical_harmonics)
                            .unwrap()
                            .texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &gpu_images
                            .get(&texture_buffers.rotation)
                            .unwrap()
                            .texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(
                        &gpu_images
                            .get(&texture_buffers.scale_opacity)
                            .unwrap()
                            .texture_view,
                    ),
                },
            ],
        );

        commands
            .entity(entity)
            .insert(GpuTextureBuffers { bind_group });
    }
}

// TODO: support asset change detection and reupload
fn queue_textures(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gaussian_cloud_res: Res<Assets<Cloud>>,
    mut images: ResMut<Assets<Image>>,
    clouds: Query<(Entity, &PlanarGaussian3dHandle), Without<TextureBuffers>>,
) {
    for (entity, cloud_handle) in clouds.iter() {
        if Some(LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        if gaussian_cloud_res.get(cloud_handle).is_none() {
            continue;
        }

        let cloud = gaussian_cloud_res.get(cloud_handle).unwrap();

        let square = cloud.len_sqrt_ceil() as u32;
        let extent_1d = Extent3d {
            width: square,
            height: square, // TODO: shrink height to save memory (consider fixed width)
            depth_or_array_layers: 1,
        };

        let mut position_visibility = Image::new(
            extent_1d,
            TextureDimension::D2,
            bytemuck::cast_slice(cloud.position_visibility.as_slice()).to_vec(),
            TextureFormat::Rgba32Float,
            RenderAssetUsages::default(), // TODO: if there are no CPU image derived features, set to render only
        );
        position_visibility.texture_descriptor.usage =
            TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
        let position_visibility = images.add(position_visibility);

        let texture_buffers: TextureBuffers;

        // #[cfg(feature = "f16")]
        // {
        //     let planar_spherical_harmonics: Vec<u32> = (0..SH_VEC4_PLANES)
        //         .flat_map(|plane_index| {
        //             cloud.spherical_harmonic.iter()
        //                 .flat_map(move |sh| {
        //                     let start_index = plane_index * 4;
        //                     let end_index = std::cmp::min(start_index + 4, sh.coefficients.len());

        //                     let mut depthwise = sh.coefficients[start_index..end_index].to_vec();
        //                     depthwise.resize(4, 0);

        //                     depthwise
        //                 })
        //         })
        //         .collect();

        //     let mut spherical_harmonics = Image::new(
        //         Extent3d {
        //             width: square,
        //             height: square,
        //             depth_or_array_layers: SH_VEC4_PLANES as u32,
        //         },
        //         TextureDimension::D2,
        //         bytemuck::cast_slice(planar_spherical_harmonics.as_slice()).to_vec(),
        //         TextureFormat::Rgba32Uint,
        //         RenderAssetUsages::default(),
        //     );
        //     spherical_harmonics.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
        //     let spherical_harmonics = images.add(spherical_harmonics);

        //     #[cfg(feature = "precompute_covariance_3d")]
        //     {
        //         let mut covariance_3d_opacity = Image::new(
        //             extent_1d,
        //             TextureDimension::D2,
        //             bytemuck::cast_slice(cloud.covariance_3d_opacity_packed128.as_slice()).to_vec(),
        //             TextureFormat::Rgba32Uint,
        //             RenderAssetUsages::default(),
        //         );
        //         covariance_3d_opacity.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
        //         let covariance_3d_opacity = images.add(covariance_3d_opacity);

        //         texture_buffers = TextureBuffers {
        //             position_visibility,
        //             spherical_harmonics,
        //             covariance_3d_opacity,
        //         };
        //     }

        //     #[cfg(not(feature = "precompute_covariance_3d"))]
        //     {
        //         let mut rotation_scale_opacity = Image::new(
        //             extent_1d,
        //             TextureDimension::D2,
        //             bytemuck::cast_slice(cloud.rotation_scale_opacity_packed128.as_slice()).to_vec(),
        //             TextureFormat::Rgba32Uint,
        //             RenderAssetUsages::default(),
        //         );
        //         rotation_scale_opacity.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
        //         let rotation_scale_opacity = images.add(rotation_scale_opacity);

        //         texture_buffers = TextureBuffers {
        //             position_visibility,
        //             spherical_harmonics,
        //             rotation_scale_opacity,
        //         };
        //     }
        // }

        {
            texture_buffers = TextureBuffers {
                position_visibility,
                spherical_harmonics: todo!(),
                rotation: todo!(),
                scale_opacity: todo!(),
            };
        }

        commands.entity(entity).insert(texture_buffers);
    }
}

pub fn get_sorted_bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
    render_device.create_bind_group_layout(
        Some("texture_sorted_layout"),
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::all(),
            ty: BindingType::Texture {
                view_dimension: TextureViewDimension::D2,
                sample_type: TextureSampleType::Uint,
                multisampled: false,
            },
            count: None,
        }],
    )
}

// #[cfg(feature = "f16")]
// pub fn get_bind_group_layout(
//     render_device: &RenderDevice,
//     _read_only: bool
// ) -> BindGroupLayout {
//     let sh_view_dimension = if SH_VEC4_PLANES == 1 {
//         TextureViewDimension::D2
//     } else {
//         TextureViewDimension::D2Array
//     };

//     render_device.create_bind_group_layout(
//         Some("texture_f16_gaussian_cloud_layout"),
//         &[
//             BindGroupLayoutEntry {
//                 binding: 0,
//                 visibility: ShaderStages::all(),
//                 ty: BindingType::Texture {
//                     view_dimension: TextureViewDimension::D2,
//                     sample_type: TextureSampleType::Float {
//                         filterable: false,
//                     },
//                     multisampled: false,
//                 },
//                 count: None,
//             },
//             BindGroupLayoutEntry {
//                 binding: 1,
//                 visibility: ShaderStages::all(),
//                 ty: BindingType::Texture {
//                     view_dimension: sh_view_dimension,
//                     sample_type: TextureSampleType::Uint,
//                     multisampled: false,
//                 },
//                 count: None,
//             },
//             BindGroupLayoutEntry {
//                 binding: 2,
//                 visibility: ShaderStages::all(),
//                 ty: BindingType::Texture {
//                     view_dimension: TextureViewDimension::D2,
//                     sample_type: TextureSampleType::Uint,
//                     multisampled: false,
//                 },
//                 count: None,
//             },
//         ],
//     )
// }

pub fn get_bind_group_layout(render_device: &RenderDevice, read_only: bool) -> BindGroupLayout {
    render_device.create_bind_group_layout(
        Some("texture_f32_gaussian_cloud_layout"),
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(
                        std::mem::size_of::<PositionVisibility>() as u64
                    ),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<
                        SphericalHarmonicCoefficients,
                    >() as u64),
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
    )
}
