use rand::{
    Rng, SeedableRng,
    distr::{Distribution, StandardUniform},
    rng,
    rngs::StdRng,
    seq::SliceRandom,
};
use std::marker::Copy;
#[cfg(any(feature = "lod", feature = "precompute_covariance_3d"))]
use std::mem::size_of;

use bevy::prelude::*;
use bevy_interleave::prelude::*;
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::{
    gaussian::{
        f32::{Covariance3dOpacity, PositionVisibility, Rotation, ScaleOpacity},
        interface::{CommonCloud, TestCloud},
        iter::PositionIter,
        settings::CloudSettings,
    },
    material::spherical_harmonics::{
        HALF_SH_COEFF_COUNT, SH_COEFF_COUNT, SphericalHarmonicCoefficients,
    },
};

#[derive(
    Clone,
    Debug,
    Default,
    Copy,
    PartialEq,
    Planar,
    ReflectInterleaved,
    Reflect,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[cfg_attr(not(feature = "precompute_covariance_3d"), derive(StorageBindings))]
#[serde(default)]
#[repr(C)]
pub struct Gaussian3d {
    #[serde(default)]
    pub position_visibility: PositionVisibility,
    #[serde(default)]
    pub spherical_harmonic: SphericalHarmonicCoefficients,
    #[serde(default)]
    pub rotation: Rotation,
    #[serde(default)]
    pub scale_opacity: ScaleOpacity,
}

/// Total bytes written to GPU storage for one canonical 3D Gaussian.
///
/// The serialized/runtime page ABI remains [`Gaussian3d`], but the
/// `precompute_covariance_3d` render layout owns one additional derived plane.
/// Atlas allocation and atomic upload budgets must include both.
#[cfg(feature = "lod")]
pub(crate) const fn gaussian_3d_gpu_bytes_per_record() -> u64 {
    let canonical = size_of::<Gaussian3d>() as u64;
    #[cfg(feature = "precompute_covariance_3d")]
    {
        canonical + size_of::<Covariance3dOpacity>() as u64
    }
    #[cfg(not(feature = "precompute_covariance_3d"))]
    {
        canonical
    }
}

/// GPU storage for the canonical planar 3D asset.
///
/// `precompute_covariance_3d` deliberately augments the canonical rotation and
/// scale planes instead of replacing them. Gaussian2d, normal rendering, LoD
/// support bounds, and tooling still need the original transform, while the
/// Gaussian3d raster path can consume the additional covariance plane without
/// changing the serialized asset ABI.
#[derive(Debug, Clone)]
#[cfg(feature = "precompute_covariance_3d")]
pub struct PlanarStorageGaussian3d {
    pub position_visibility: bevy::render::render_resource::Buffer,
    pub spherical_harmonic: bevy::render::render_resource::Buffer,
    pub rotation: bevy::render::render_resource::Buffer,
    pub scale_opacity: bevy::render::render_resource::Buffer,
    pub covariance_3d_opacity: bevy::render::render_resource::Buffer,
    pub count: usize,
    pub draw_indirect_buffer: bevy::render::render_resource::Buffer,
}

#[cfg(feature = "precompute_covariance_3d")]
impl bevy::render::render_asset::RenderAsset for PlanarStorageGaussian3d {
    type SourceAsset = PlanarGaussian3d;
    type Param = bevy::ecs::system::lifetimeless::SRes<bevy::render::renderer::RenderDevice>;

    fn prepare_asset(
        source: Self::SourceAsset,
        _: bevy::asset::AssetId<Self::SourceAsset>,
        render_device: &mut bevy::ecs::system::SystemParamItem<Self::Param>,
        _: Option<&Self>,
    ) -> Result<Self, bevy::render::render_asset::PrepareAssetError<Self::SourceAsset>> {
        use bevy::render::render_resource::{BufferInitDescriptor, BufferUsages};

        let count = source.len();
        let draw_indirect_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian_3d_draw_indirect"),
            contents: wgpu::util::DrawIndirectArgs {
                vertex_count: 4,
                instance_count: count.min(u32::MAX as usize) as u32,
                first_vertex: 0,
                first_instance: 0,
            }
            .as_bytes(),
            usage: BufferUsages::INDIRECT
                | BufferUsages::COPY_DST
                | BufferUsages::STORAGE
                | BufferUsages::COPY_SRC,
        });

        let storage_usage = BufferUsages::COPY_DST | BufferUsages::STORAGE;
        let position_visibility = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("position_visibility_buffer"),
            contents: bytemuck::cast_slice(source.position_visibility.as_slice()),
            usage: storage_usage,
        });
        let spherical_harmonic = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("spherical_harmonic_buffer"),
            contents: bytemuck::cast_slice(source.spherical_harmonic.as_slice()),
            usage: storage_usage,
        });
        let rotation = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("rotation_buffer"),
            contents: bytemuck::cast_slice(source.rotation.as_slice()),
            usage: storage_usage,
        });
        let scale_opacity = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("scale_opacity_buffer"),
            contents: bytemuck::cast_slice(source.scale_opacity.as_slice()),
            usage: storage_usage,
        });

        #[cfg(feature = "precompute_covariance_3d")]
        let covariance_3d_opacity = {
            let values = source
                .rotation
                .iter()
                .zip(source.scale_opacity.iter())
                .map(|(rotation, scale_opacity)| Covariance3dOpacity {
                    cov3d: crate::gaussian::covariance::compute_covariance_3d(
                        Vec4::from_array(rotation.rotation),
                        Vec3::from_array(scale_opacity.scale),
                    ),
                    opacity: scale_opacity.opacity,
                    pad: 0.0,
                })
                .collect::<Vec<_>>();
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("covariance_3d_opacity_buffer"),
                contents: bytemuck::cast_slice(values.as_slice()),
                usage: storage_usage,
            })
        };

        Ok(Self {
            position_visibility,
            spherical_harmonic,
            rotation,
            scale_opacity,
            #[cfg(feature = "precompute_covariance_3d")]
            covariance_3d_opacity,
            count,
            draw_indirect_buffer,
        })
    }

    fn asset_usage(_: &Self::SourceAsset) -> bevy::asset::RenderAssetUsages {
        bevy::asset::RenderAssetUsages::default()
    }
}

#[cfg(feature = "precompute_covariance_3d")]
impl GpuPlanar for PlanarStorageGaussian3d {
    type PackedType = Gaussian3d;
    type PlanarType = PlanarGaussian3d;

    fn len(&self) -> usize {
        self.count
    }
}

#[cfg(feature = "precompute_covariance_3d")]
impl GpuPlanarStorage for PlanarStorageGaussian3d {
    fn draw_indirect_buffer(&self) -> &bevy::render::render_resource::Buffer {
        &self.draw_indirect_buffer
    }

    fn bind_group(
        &self,
        render_device: &bevy::render::renderer::RenderDevice,
        layout: &bevy::render::render_resource::BindGroupLayout,
    ) -> bevy::render::render_resource::BindGroup {
        use bevy::render::render_resource::{BindGroupEntry, BindingResource, BufferBinding};

        fn binding<'a>(
            binding: u32,
            buffer: &'a bevy::render::render_resource::Buffer,
        ) -> BindGroupEntry<'a> {
            BindGroupEntry {
                binding,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer,
                    offset: 0,
                    size: bevy::render::render_resource::BufferSize::new(buffer.size()),
                }),
            }
        }
        let entries = vec![
            binding(0, &self.position_visibility),
            binding(1, &self.spherical_harmonic),
            binding(2, &self.rotation),
            binding(3, &self.scale_opacity),
        ];
        #[cfg(feature = "precompute_covariance_3d")]
        let entries = {
            let mut entries = entries;
            entries.push(binding(4, &self.covariance_3d_opacity));
            entries
        };

        render_device.create_bind_group("storage_gaussian_3d_bind_group", layout, &entries)
    }

    fn bind_group_layout(
        render_device: &bevy::render::renderer::RenderDevice,
        read_only: bool,
    ) -> bevy::render::render_resource::BindGroupLayout {
        use bevy::render::render_resource::{
            BindGroupLayoutEntry, BindingType, BufferBindingType, BufferSize, ShaderStages,
        };

        let sizes = [
            size_of::<PositionVisibility>(),
            size_of::<SphericalHarmonicCoefficients>(),
            size_of::<Rotation>(),
            size_of::<ScaleOpacity>(),
            #[cfg(feature = "precompute_covariance_3d")]
            size_of::<Covariance3dOpacity>(),
        ];
        let entries = sizes
            .iter()
            .enumerate()
            .map(|(binding, size)| BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(*size as u64),
                },
                count: None,
            })
            .collect::<Vec<_>>();
        render_device
            .create_bind_group_layout(Some("storage_gaussian_3d_bind_group_layout"), &entries)
    }
}

/// Byte ranges for one aligned planar GPU subrange.
#[cfg(feature = "lod")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Gaussian3dStorageRangeLayout {
    pub position_visibility_offset: u64,
    pub spherical_harmonic_offset: u64,
    pub rotation_offset: u64,
    pub scale_opacity_offset: u64,
    #[cfg(feature = "precompute_covariance_3d")]
    pub covariance_3d_opacity_offset: u64,
    pub count: usize,
}

#[cfg(feature = "lod")]
fn plane_byte_offset<T>(start: usize) -> Result<u64, Gaussian3dStorageWriteError> {
    let start = u64::try_from(start).map_err(|_| Gaussian3dStorageWriteError::AddressOverflow)?;
    let element_size = u64::try_from(std::mem::size_of::<T>())
        .map_err(|_| Gaussian3dStorageWriteError::AddressOverflow)?;
    start
        .checked_mul(element_size)
        .ok_or(Gaussian3dStorageWriteError::AddressOverflow)
}

#[cfg(feature = "lod")]
fn gaussian_3d_storage_range_layout(
    storage_count: usize,
    start: usize,
    count: usize,
) -> Result<Gaussian3dStorageRangeLayout, Gaussian3dStorageWriteError> {
    if count == 0 {
        return Err(Gaussian3dStorageWriteError::EmptyRange);
    }
    let end = start
        .checked_add(count)
        .ok_or(Gaussian3dStorageWriteError::AddressOverflow)?;
    if end > storage_count {
        return Err(Gaussian3dStorageWriteError::RangeOutOfBounds {
            start,
            end,
            storage_count,
        });
    }
    Ok(Gaussian3dStorageRangeLayout {
        position_visibility_offset: plane_byte_offset::<PositionVisibility>(start)?,
        spherical_harmonic_offset: plane_byte_offset::<SphericalHarmonicCoefficients>(start)?,
        rotation_offset: plane_byte_offset::<Rotation>(start)?,
        scale_opacity_offset: plane_byte_offset::<ScaleOpacity>(start)?,
        #[cfg(feature = "precompute_covariance_3d")]
        covariance_3d_opacity_offset: plane_byte_offset::<Covariance3dOpacity>(start)?,
        count,
    })
}

#[cfg(feature = "lod")]
impl PlanarStorageGaussian3d {
    /// Writes one already-allocated planar subrange without replacing buffers
    /// or invalidating bind groups. This is intentionally separate from the
    /// normal RenderAsset update path and is used only by fixed-capacity LoD
    /// atlases whose CPU mirror has already been updated.
    pub(crate) fn write_gaussian_3d_range(
        &self,
        render_queue: &bevy::render::renderer::RenderQueue,
        start: usize,
        planes: &PlanarGaussian3d,
    ) -> Result<(), Gaussian3dStorageWriteError> {
        let count = planes.len();
        if planes.spherical_harmonic.len() != count
            || planes.rotation.len() != count
            || planes.scale_opacity.len() != count
        {
            return Err(Gaussian3dStorageWriteError::InconsistentPlaneLengths);
        }
        let layout = gaussian_3d_storage_range_layout(self.count, start, count)?;

        render_queue.write_buffer(
            &self.position_visibility,
            layout.position_visibility_offset,
            bytemuck::cast_slice(planes.position_visibility.as_slice()),
        );
        render_queue.write_buffer(
            &self.spherical_harmonic,
            layout.spherical_harmonic_offset,
            bytemuck::cast_slice(planes.spherical_harmonic.as_slice()),
        );
        render_queue.write_buffer(
            &self.rotation,
            layout.rotation_offset,
            bytemuck::cast_slice(planes.rotation.as_slice()),
        );
        render_queue.write_buffer(
            &self.scale_opacity,
            layout.scale_opacity_offset,
            bytemuck::cast_slice(planes.scale_opacity.as_slice()),
        );

        Ok(())
    }

    /// Recovery fallback for the derived covariance plane. Normal streamed
    /// atlas uploads dispatch [`crate::stream::atlas_upload`]'s GPU derivation
    /// pass after adjacent canonical writes have been coalesced. This path is
    /// retained for a temporarily unavailable compute resource during device
    /// recreation and deliberately never changes serialized page data.
    #[cfg(feature = "precompute_covariance_3d")]
    pub(crate) fn write_gaussian_3d_covariance_range_cpu(
        &self,
        render_queue: &bevy::render::renderer::RenderQueue,
        start: usize,
        planes: &PlanarGaussian3d,
    ) -> Result<(), Gaussian3dStorageWriteError> {
        let count = planes.len();
        if planes.rotation.len() != count || planes.scale_opacity.len() != count {
            return Err(Gaussian3dStorageWriteError::InconsistentPlaneLengths);
        }
        let layout = gaussian_3d_storage_range_layout(self.count, start, count)?;
        let covariance = planes
            .rotation
            .iter()
            .zip(planes.scale_opacity.iter())
            .map(|(rotation, scale_opacity)| Covariance3dOpacity {
                cov3d: crate::gaussian::covariance::compute_covariance_3d(
                    Vec4::from_array(rotation.rotation),
                    Vec3::from_array(scale_opacity.scale),
                ),
                opacity: scale_opacity.opacity,
                pad: 0.0,
            })
            .collect::<Vec<_>>();
        render_queue.write_buffer(
            &self.covariance_3d_opacity,
            layout.covariance_3d_opacity_offset,
            bytemuck::cast_slice(covariance.as_slice()),
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(feature = "lod")]
pub(crate) enum Gaussian3dStorageWriteError {
    EmptyRange,
    AddressOverflow,
    InconsistentPlaneLengths,
    RangeOutOfBounds {
        start: usize,
        end: usize,
        storage_count: usize,
    },
}

#[cfg(feature = "precompute_covariance_3d")]
impl PlanarSync for Gaussian3d {
    type PackedType = Gaussian3d;
    type PlanarType = PlanarGaussian3d;
    type PlanarTypeHandle = PlanarGaussian3dHandle;
    type GpuPlanarType = PlanarStorageGaussian3d;
}

pub type Gaussian2d = Gaussian3d; // GaussianMode::Gaussian2d /w Gaussian3d structure

// #[allow(unused_imports)]
// #[cfg(feature = "f16")]
// use crate::gaussian::f16::{
//     Covariance3dOpacityPacked128,
//     RotationScaleOpacityPacked128,
//     pack_f32s_to_u32,
// };

// #[cfg(feature = "f16")]
// #[derive(
//     Debug,
//     Default,
//     PartialEq,
//     Reflect,
//     Serialize,
//     Deserialize,
// )]
// pub struct Cloud3d {
//     pub position_visibility: Vec<PositionVisibility>,

//     pub spherical_harmonic: Vec<SphericalHarmonicCoefficients>,

//     #[cfg(not(feature = "precompute_covariance_3d"))]
//     pub rotation_scale_opacity_packed128: Vec<RotationScaleOpacityPacked128>,

//     #[cfg(feature = "precompute_covariance_3d")]
//     pub covariance_3d_opacity_packed128: Vec<Covariance3dOpacityPacked128>,
// }

impl CommonCloud for PlanarGaussian3d {
    type PackedType = Gaussian3d;

    fn visibility(&self, index: usize) -> f32 {
        self.position_visibility[index].visibility
    }

    fn visibility_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.position_visibility[index].visibility
    }

    fn support_radius(&self, index: usize) -> Vec3 {
        let scale = Vec3::from_array(self.scale_opacity[index].scale).abs();
        if scale.is_finite() {
            Vec3::splat(scale.max_element() * 3.0)
        } else {
            Vec3::splat(0.1)
        }
    }

    fn position_iter(&self) -> PositionIter<'_> {
        PositionIter::new(&self.position_visibility)
    }

    #[cfg(feature = "sort_rayon")]
    fn position_par_iter(&self) -> crate::gaussian::iter::PositionParIter<'_> {
        crate::gaussian::iter::PositionParIter::new(&self.position_visibility)
    }
}

impl FromIterator<Gaussian3d> for PlanarGaussian3d {
    fn from_iter<I: IntoIterator<Item = Gaussian3d>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<Gaussian3d>>().into()
    }
}

impl From<Vec<Gaussian3d>> for PlanarGaussian3d {
    fn from(packed: Vec<Gaussian3d>) -> Self {
        Self::from_interleaved(packed)
    }
}

impl Distribution<Gaussian3d> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gaussian3d {
        Gaussian3d {
            rotation: [
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ]
            .into(),
            position_visibility: [
                rng.random_range(-20.0..20.0),
                rng.random_range(-20.0..20.0),
                rng.random_range(-20.0..20.0),
                1.0,
            ]
            .into(),
            scale_opacity: [
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..0.8),
            ]
            .into(),
            spherical_harmonic: SphericalHarmonicCoefficients {
                coefficients: {
                    // #[cfg(feature = "f16")]
                    // {
                    //     let mut coefficients: [u32; HALF_SH_COEFF_COUNT] = [0; HALF_SH_COEFF_COUNT];
                    //     for coefficient in coefficients.iter_mut() {
                    //         let upper = rng.gen_range(-1.0..1.0);
                    //         let lower = rng.gen_range(-1.0..1.0);

                    //         *coefficient = pack_f32s_to_u32(upper, lower);
                    //     }
                    //     coefficients
                    // }

                    {
                        let mut coefficients = [0.0; SH_COEFF_COUNT];
                        for coefficient in coefficients.iter_mut() {
                            *coefficient = rng.random_range(-1.0..1.0);
                        }
                        coefficients
                    }
                },
            },
        }
    }
}

pub fn random_gaussians_3d(n: usize) -> PlanarGaussian3d {
    let mut rng = rng();
    let mut gaussians: Vec<Gaussian3d> = Vec::with_capacity(n);

    for _ in 0..n {
        gaussians.push(rng.random());
    }

    PlanarGaussian3d::from_interleaved(gaussians)
}

pub fn random_gaussians_3d_seeded(n: usize, seed: u64) -> PlanarGaussian3d {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut gaussians: Vec<Gaussian3d> = Vec::with_capacity(n);

    for _ in 0..n {
        gaussians.push(StandardUniform.sample(&mut rng));
    }

    PlanarGaussian3d::from_interleaved(gaussians)
}

impl TestCloud for PlanarGaussian3d {
    fn test_model() -> Self {
        let mut rng = rng();

        let origin = Gaussian3d {
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            position_visibility: [0.0, 0.0, 0.0, 1.0].into(),
            scale_opacity: [0.125, 0.125, 0.125, 0.125].into(),
            spherical_harmonic: SphericalHarmonicCoefficients {
                coefficients: {
                    // #[cfg(feature = "f16")]
                    // {
                    //     let mut coefficients = [0_u32; HALF_SH_COEFF_COUNT];

                    //     for coefficient in coefficients.iter_mut() {
                    //         let upper = rng.gen_range(-1.0..1.0);
                    //         let lower = rng.gen_range(-1.0..1.0);

                    //         *coefficient = pack_f32s_to_u32(upper, lower);
                    //     }

                    //     coefficients
                    // }

                    {
                        let mut coefficients = [0.0; SH_COEFF_COUNT];

                        for coefficient in coefficients.iter_mut() {
                            *coefficient = rng.random_range(-1.0..1.0);
                        }

                        coefficients
                    }
                },
            },
        };
        let mut gaussians: Vec<Gaussian3d> = Vec::new();

        for &x in [-0.5, 0.5].iter() {
            for &y in [-0.5, 0.5].iter() {
                for &z in [-0.5, 0.5].iter() {
                    let mut g = origin;
                    g.position_visibility = [x, y, z, 1.0].into();
                    gaussians.push(g);

                    gaussians
                        .last_mut()
                        .unwrap()
                        .spherical_harmonic
                        .coefficients
                        .shuffle(&mut rng);
                }
            }
        }

        gaussians.push(gaussians[0]);
        gaussians.into()
    }
}

// TODO: attempt iter() on the Planar trait
impl PlanarGaussian3d {
    pub fn iter(&self) -> impl Iterator<Item = Gaussian3d> + '_ {
        self.position_visibility
            .iter()
            .zip(self.spherical_harmonic.iter())
            .zip(self.rotation.iter())
            .zip(self.scale_opacity.iter())
            .map(
                |(((position_visibility, spherical_harmonic), rotation), scale_opacity)| {
                    Gaussian3d {
                        position_visibility: *position_visibility,
                        spherical_harmonic: *spherical_harmonic,

                        rotation: *rotation,
                        scale_opacity: *scale_opacity,
                    }
                },
            )
    }
}

#[cfg(test)]
mod bounds_tests {
    use super::*;

    #[test]
    fn cloud_bounds_include_anisotropic_three_sigma_support() {
        let gaussian = Gaussian3d {
            position_visibility: [10.0, -2.0, 3.0, 1.0].into(),
            scale_opacity: [2.0, 0.5, 0.25, 1.0].into(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            ..Default::default()
        };
        let cloud = PlanarGaussian3d::from(vec![gaussian]);
        let bounds = cloud.compute_aabb().unwrap();
        assert_eq!(Vec3::from(bounds.min), Vec3::new(4.0, -8.0, -3.0));
        assert_eq!(Vec3::from(bounds.max), Vec3::new(16.0, 4.0, 9.0));
    }

    #[test]
    fn cloud_bounds_fall_back_for_non_finite_scale() {
        let gaussian = Gaussian3d {
            position_visibility: [1.0, 2.0, 3.0, 1.0].into(),
            scale_opacity: [f32::NAN, 1.0, 1.0, 1.0].into(),
            ..Default::default()
        };
        let cloud = PlanarGaussian3d::from(vec![gaussian]);
        let bounds = cloud.compute_aabb().unwrap();
        assert_eq!(Vec3::from(bounds.min), Vec3::new(0.9, 1.9, 2.9));
        assert_eq!(Vec3::from(bounds.max), Vec3::new(1.1, 2.1, 3.1));
    }

    #[test]
    #[cfg(feature = "lod")]
    fn gpu_subrange_layout_uses_each_planar_element_stride() {
        let layout = gaussian_3d_storage_range_layout(32, 3, 5).unwrap();
        assert_eq!(layout.count, 5);
        assert_eq!(
            layout.position_visibility_offset,
            (3 * std::mem::size_of::<PositionVisibility>()) as u64
        );
        assert_eq!(
            layout.spherical_harmonic_offset,
            (3 * std::mem::size_of::<SphericalHarmonicCoefficients>()) as u64
        );
        assert_eq!(
            layout.rotation_offset,
            (3 * std::mem::size_of::<Rotation>()) as u64
        );
        assert_eq!(
            layout.scale_opacity_offset,
            (3 * std::mem::size_of::<ScaleOpacity>()) as u64
        );
        #[cfg(feature = "precompute_covariance_3d")]
        assert_eq!(
            layout.covariance_3d_opacity_offset,
            (3 * std::mem::size_of::<Covariance3dOpacity>()) as u64
        );
    }

    #[test]
    #[cfg(feature = "lod")]
    fn gpu_subrange_layout_rejects_empty_overflow_and_out_of_bounds_ranges() {
        assert_eq!(
            gaussian_3d_storage_range_layout(8, 0, 0),
            Err(Gaussian3dStorageWriteError::EmptyRange)
        );
        assert_eq!(
            gaussian_3d_storage_range_layout(usize::MAX, usize::MAX, 1),
            Err(Gaussian3dStorageWriteError::AddressOverflow)
        );
        assert_eq!(
            gaussian_3d_storage_range_layout(8, 6, 3),
            Err(Gaussian3dStorageWriteError::RangeOutOfBounds {
                start: 6,
                end: 9,
                storage_count: 8,
            })
        );
    }
}
