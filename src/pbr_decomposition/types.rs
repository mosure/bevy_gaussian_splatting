use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[derive(
    Clone,
    Debug,
    Copy,
    PartialEq,
    Reflect,
    ShaderType,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct PbrMaterialData {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub perceptual_roughness: f32,
    pub reflectance: f32,
    pub ambient_occlusion: f32,
    pub _pad: f32,
}

impl Default for PbrMaterialData {
    fn default() -> Self {
        Self {
            base_color: [0.5, 0.5, 0.5],
            metallic: 0.0,
            perceptual_roughness: 0.5,
            reflectance: 0.5,
            ambient_occlusion: 1.0,
            _pad: 0.0,
        }
    }
}

#[derive(
    Clone,
    Debug,
    Copy,
    PartialEq,
    Reflect,
    ShaderType,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct NormalData {
    pub normal: [f32; 3],
    pub confidence: f32,
}

impl Default for NormalData {
    fn default() -> Self {
        Self {
            normal: [0.0, 0.0, 1.0],
            confidence: 0.0,
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Reflect, ShaderType)]
#[repr(C)]
pub struct StreamingStats {
    pub mean_rgb: Vec3,
    pub count: u32,

    pub M2_rgb: Vec3,
    pub near_normal_count: u32,

    pub near_normal_mean: Vec3,
    pub topk_count: u32,

    pub topk_directions: [Vec3; 8],
    pub topk_intensities: [f32; 8],

    pub residual_direction_sum: Vec3,
    pub residual_direction_M2: f32,

    pub _pad: [f32; 3],
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self {
            mean_rgb: Vec3::ZERO,
            count: 0,
            M2_rgb: Vec3::ZERO,
            near_normal_count: 0,
            near_normal_mean: Vec3::ZERO,
            topk_count: 0,
            topk_directions: [Vec3::ZERO; 8],
            topk_intensities: [0.0; 8],
            residual_direction_sum: Vec3::ZERO,
            residual_direction_M2: 0.0,
            _pad: [0.0; 3],
        }
    }
}

#[derive(
    Clone,
    Debug,
    Copy,
    PartialEq,
    Reflect,
    ShaderType,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct SpatialHashConfig {
    pub cell_size: f32,
    pub table_size: u32,
    pub gaussian_count: u32,
    pub _pad: u32,
}

impl Default for SpatialHashConfig {
    fn default() -> Self {
        Self {
            cell_size: 0.5,
            table_size: 1024,
            gaussian_count: 0,
            _pad: 0,
        }
    }
}

impl SpatialHashConfig {
    pub fn from_point_cloud(positions: &[[f32; 3]], target_neighbors: u32) -> Self {
        if positions.is_empty() {
            return Self::default();
        }

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for pos in positions {
            for i in 0..3 {
                min[i] = min[i].min(pos[i]);
                max[i] = max[i].max(pos[i]);
            }
        }

        let volume = (max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2]);
        let density = positions.len() as f32 / volume.max(1e-6);

        let cell_size = (target_neighbors as f32 / density).cbrt().max(0.01);

        let table_size = (positions.len() * 2).next_power_of_two();

        Self {
            cell_size,
            table_size: table_size as u32,
            gaussian_count: positions.len() as u32,
            _pad: 0,
        }
    }
}

#[derive(
    Clone,
    Debug,
    Copy,
    PartialEq,
    Reflect,
    ShaderType,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct GridCell {
    pub start: u32,
    pub count: u32,
}

impl Default for GridCell {
    fn default() -> Self {
        Self { start: 0, count: 0 }
    }
}
