use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect, Serialize, Deserialize)]
pub enum SHCoordinateFrame {
    World,
    Local,
}

impl Default for SHCoordinateFrame {
    fn default() -> Self {
        Self::World
    }
}

#[derive(Resource, Reflect, Clone, Serialize, Deserialize)]
#[reflect(Resource)]
pub struct PbrDecompositionSettings {
    pub use_gpu_hashed_grid: bool,
    pub neighbor_search_radius: f32,
    pub max_neighbors: u32,
    pub hash_table_load_factor: f32,

    pub use_scale_axis_method: bool,
    pub normal_spatial_sigma: f32,
    pub normal_color_sigma: f32,
    pub normal_confidence_threshold: f32,

    pub use_streaming_stats: bool,
    pub num_synthetic_views: u32,
    pub view_near_normal_angle: f32,
    pub sh_coordinate_frame: SHCoordinateFrame,
    pub topk_residuals: u32,

    pub enable_spectral_analysis: bool,
    pub enable_chroma_filtering: bool,
    pub specular_intensity_threshold: f32,
    pub specular_hue_tolerance: f32,

    pub validated_highlight_min_count: u32,
    pub metallic_saturation_threshold: f32,
    pub metallic_min_threshold: f32,
    pub roughness_min: f32,
    pub roughness_max: f32,
    pub ao_radius: f32,
    pub ao_samples: u32,

    pub enable_material_propagation: bool,
    pub propagation_radius: f32,
    pub propagation_normal_threshold: f32,
    pub propagation_blend_max: f32,
    pub high_confidence_roughness_max: f32,

    pub enable_energy_validation: bool,
    pub energy_error_threshold: f32,
    pub ggx_lut_path: String,

    pub use_gbuffer_path: bool,
    pub reconstruct_position_from_depth: bool,
    pub enable_ssao: bool,
    pub enable_intrinsic_ao: bool,

    pub cache_decomposed_materials: bool,
    pub cache_path: String,
}

impl Default for PbrDecompositionSettings {
    fn default() -> Self {
        Self {
            use_gpu_hashed_grid: true,
            neighbor_search_radius: 0.5,
            max_neighbors: 64,
            hash_table_load_factor: 0.5,

            use_scale_axis_method: true,
            normal_spatial_sigma: 0.1,
            normal_color_sigma: 0.2,
            normal_confidence_threshold: 0.5,

            use_streaming_stats: true,
            num_synthetic_views: 128,
            view_near_normal_angle: 30.0,
            sh_coordinate_frame: SHCoordinateFrame::World,
            topk_residuals: 8,

            enable_spectral_analysis: true,
            enable_chroma_filtering: true,
            specular_intensity_threshold: 1.5,
            specular_hue_tolerance: 0.1,

            validated_highlight_min_count: 3,
            metallic_saturation_threshold: 0.15,
            metallic_min_threshold: 0.1,
            roughness_min: 0.089,
            roughness_max: 1.0,
            ao_radius: 0.5,
            ao_samples: 16,

            enable_material_propagation: true,
            propagation_radius: 1.0,
            propagation_normal_threshold: 0.85,
            propagation_blend_max: 0.5,
            high_confidence_roughness_max: 0.3,

            enable_energy_validation: true,
            energy_error_threshold: 0.1,
            ggx_lut_path: "assets/textures/ggx_energy_lut.png".to_string(),

            use_gbuffer_path: true,
            reconstruct_position_from_depth: true,
            enable_ssao: true,
            enable_intrinsic_ao: false,

            cache_decomposed_materials: true,
            cache_path: "assets/cache/".to_string(),
        }
    }
}
