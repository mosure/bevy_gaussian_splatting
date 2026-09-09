use bevy::prelude::*;
use bevy_args::{Deserialize, Parser, Serialize};

use crate::gaussian::settings::{GaussianMode, PlaybackMode, RadixSortDepthBits, RasterizeMode};

#[cfg(feature = "lod")]
use crate::gaussian::lod_debug::{LodDebugPreset, LodDebugSettings};
#[cfg(feature = "lod")]
use crate::gaussian::lod_settings::{
    GaussianLodSettings, GaussianStreamingSettings, LodBudgets, LodSelectionMode, LodSettingsError,
};
#[cfg(feature = "lod")]
use crate::stream::memory::LodMemoryLimits;

/// Default active-record ceiling used by the standalone viewer.
#[cfg(feature = "lod")]
pub const VIEWER_DEFAULT_LOD_MAX_ACTIVE_GAUSSIANS: u64 = 8_000_000;
/// Default page-transport concurrency used by standalone viewer packages.
#[cfg(feature = "lod")]
pub const VIEWER_DEFAULT_LOD_MAX_CONCURRENT_REQUESTS: u32 = 64;
/// Stateless selector policy used by the standalone viewer.
///
/// The reusable library keeps its temporal hysteresis default. The viewer uses
/// a canonical cut for an identical camera and quality so returning to a pose
/// cannot settle on a history-dependent frontier.
#[cfg(feature = "lod")]
pub const VIEWER_DEFAULT_LOD_HYSTERESIS: f32 = 0.0;

#[cfg(feature = "lod")]
#[derive(Debug, Serialize, Deserialize, clap::Args)]
#[serde(default)]
pub struct GaussianLodViewerArgs {
    #[arg(
        long,
        default_value = "1.0",
        help = "detail quality in [0,1]: 0 is coarsest, 1 is exact, and intermediate detail scales with projected node size and pixel error"
    )]
    pub lod_quality: f32,

    #[arg(
        long,
        default_value_t = VIEWER_DEFAULT_LOD_MAX_ACTIVE_GAUSSIANS,
        help = "maximum active Gaussians in one LoD cut; values above the viewer's resident-record capacity are clamped"
    )]
    pub lod_max_active_gaussians: u64,

    /// Maximum resident Gaussian records per cloud, including refinement headroom.
    #[arg(long, default_value_t = LodBudgets::default().max_resident_gaussians)]
    pub lod_max_resident_gaussians: u64,

    /// Maximum resident page/atlas bytes per cloud; the shared ledger also applies.
    #[arg(long, default_value_t = LodBudgets::default().max_resident_bytes)]
    pub lod_max_resident_bytes: u64,

    /// Shared CPU admission for owned LoD allocations, excluding unrelated process memory.
    #[arg(long, default_value_t = LodMemoryLimits::default().max_cpu_bytes)]
    pub lod_max_cpu_bytes: u64,

    /// Shared GPU admission for owned LoD allocations, excluding unrelated device memory.
    #[arg(long, default_value_t = LodMemoryLimits::default().max_gpu_bytes)]
    pub lod_max_gpu_bytes: u64,

    #[arg(
        long,
        default_value_t = VIEWER_DEFAULT_LOD_MAX_CONCURRENT_REQUESTS,
        help = "maximum concurrent LoD page transport requests for standalone packages; lower this for constrained HTTP origins"
    )]
    pub lod_max_concurrent_requests: u32,

    /// Encoded manifest admission for the selected package; page limits are separate.
    #[arg(long, default_value_t = crate::io::lod::LodCodecLimits::DEFAULT_MAX_MANIFEST_BYTES,
        value_parser = clap::value_parser!(u64).range(1..=536_870_912))]
    pub lod_max_manifest_bytes: u64,

    #[arg(
        long,
        action = clap::ArgAction::SetTrue,
        help = "freeze LoD selection to the current camera (press F in the viewer to toggle)"
    )]
    #[serde(default)]
    pub lod_freeze: bool,

    #[arg(
        long,
        value_enum,
        help = "LoD visualization: off, level, page, residency, boundaries, or selection-pressure"
    )]
    pub lod_debug: Option<LodDebugPreset>,
}

#[cfg(feature = "lod")]
impl Default for GaussianLodViewerArgs {
    fn default() -> Self {
        Self {
            lod_quality: 1.0,
            lod_max_active_gaussians: VIEWER_DEFAULT_LOD_MAX_ACTIVE_GAUSSIANS,
            lod_max_resident_gaussians: LodBudgets::default().max_resident_gaussians,
            lod_max_resident_bytes: LodBudgets::default().max_resident_bytes,
            lod_max_cpu_bytes: LodMemoryLimits::default().max_cpu_bytes,
            lod_max_gpu_bytes: LodMemoryLimits::default().max_gpu_bytes,
            lod_max_concurrent_requests: VIEWER_DEFAULT_LOD_MAX_CONCURRENT_REQUESTS,
            lod_max_manifest_bytes: crate::io::lod::LodCodecLimits::DEFAULT_MAX_MANIFEST_BYTES,
            lod_freeze: false,
            lod_debug: None,
        }
    }
}

/// Optional standalone-viewer controls for the point-splatting camera backend.
#[cfg(lod_render_path)]
#[derive(Debug, Serialize, Deserialize, clap::Args)]
#[serde(default)]
pub struct GaussianPointSplattingViewerArgs {
    #[arg(long, action = clap::ArgAction::SetTrue, conflicts_with = "global_order",
        help = "render supported 3D color clouds with Gaussian point splatting; uses MSAA off and discrete LoD cuts")]
    pub point_splatting: bool,

    #[arg(
        long,
        default_value_t = 4,
        help = "initial complete point-sampling layers per pixel, from 1 to 8"
    )]
    pub point_samples_per_pixel: u32,

    #[arg(
        long,
        default_value_t = 1,
        help = "minimum complete sampling layers retained by automatic control"
    )]
    pub point_min_samples_per_pixel: u32,

    #[arg(
        long,
        default_value_t = 16_777_216,
        help = "maximum generated points per frame before viewport/support rejection"
    )]
    pub point_max_points: u32,

    #[arg(
        long,
        help = "optional GPU millisecond target for automatic whole-layer sampling; requires supported timestamp feedback for timing control"
    )]
    pub point_target_gpu_ms: Option<f32>,

    #[arg(long, default_value_t = 1_048_576)]
    pub point_max_projected_gaussians: u32,

    #[arg(long, default_value_t = 268_435_456)]
    pub point_max_gpu_bytes: u64,

    #[arg(
        long,
        help = "select and request package pages on the GPU; requires --input-lod and either --point-splatting or --global-order"
    )]
    pub lod_gpu_traversal: bool,

    #[arg(long, requires = "lod_gpu_traversal",
        default_value_t = crate::render::traversal::GpuLodTraversalSettings::default().max_page_requests,
        help = "maximum GPU traversal page references per view before host deduplication")]
    pub lod_max_page_requests: u32,
}

#[cfg(lod_render_path)]
impl Default for GaussianPointSplattingViewerArgs {
    fn default() -> Self {
        let settings = crate::render::point::GaussianPointSplattingSettings::default();
        Self {
            point_splatting: false,
            point_samples_per_pixel: settings.samples_per_pixel,
            point_min_samples_per_pixel: settings.min_samples_per_pixel,
            point_max_points: settings.max_points_per_frame,
            point_target_gpu_ms: settings.target_gpu_ms,
            point_max_projected_gaussians: settings.max_projected_gaussians,
            point_max_gpu_bytes: settings.max_gpu_bytes,
            lod_gpu_traversal: false,
            lod_max_page_requests: crate::render::traversal::GpuLodTraversalSettings::default()
                .max_page_requests,
        }
    }
}

/// Optional shared depth order for supported opaque-depth-tested alpha quads.
#[cfg(lod_render_path)]
#[derive(Debug, Serialize, Deserialize, clap::Args)]
#[serde(default)]
pub struct GaussianGlobalOrderViewerArgs {
    #[arg(
        long,
        conflicts_with = "point_splatting",
        help = "sort supported 3D color clouds together; uses MSAA off"
    )]
    pub global_order: bool,
    #[arg(long, default_value_t = 1_048_576)]
    pub global_order_max_gaussians: u32,
    #[arg(long, default_value_t = 268_435_456)]
    pub global_order_max_gpu_bytes: u64,
    #[arg(long, requires_all = ["global_order", "lod_gpu_traversal", "input_lod"],
        help = "blend adjacent resident LoD representations using the current camera")]
    pub lod_spatial_transitions: bool,
    #[arg(long, default_value_t = 256)]
    pub lod_max_transition_nodes: u32,
    #[arg(long, default_value_t = 65_536)]
    pub lod_max_transition_records: u32,
    #[arg(long, default_value_t = 33_554_432)]
    pub lod_max_mapping_bytes: u64,
}

#[cfg(lod_render_path)]
impl Default for GaussianGlobalOrderViewerArgs {
    fn default() -> Self {
        let settings = crate::render::ordered::GaussianGlobalOrderSettings::default();
        let transitions =
            crate::render::spatial_morph::GaussianLodSpatialTransitionSettings::default();
        Self {
            global_order: false,
            global_order_max_gaussians: settings.max_projected_gaussians,
            global_order_max_gpu_bytes: settings.max_gpu_bytes,
            lod_spatial_transitions: false,
            lod_max_transition_nodes: transitions.max_transition_nodes,
            lod_max_transition_records: transitions.max_transition_records,
            lod_max_mapping_bytes: transitions.max_mapping_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum CameraController {
    #[default]
    Orbit,
    Flycam,
}

fn default_camera_speed() -> f32 {
    12.0
}
fn default_camera_sensitivity() -> f32 {
    0.00012
}

#[derive(Debug, Resource, Serialize, Deserialize, Parser)]
#[command(about = "bevy_gaussian_splatting viewer", version, long_about = None)]
pub struct GaussianSplattingViewer {
    /// Camera navigation: orbit for object inspection, flycam for moving through scenes.
    #[arg(long, value_enum, default_value = "orbit")]
    #[serde(default)]
    pub camera_controller: CameraController,

    /// Flycam movement speed in world units per second.
    #[arg(long, default_value_t = default_camera_speed())]
    #[serde(default = "default_camera_speed")]
    pub camera_speed: f32,

    /// Flycam mouse sensitivity (bevy_flycam units).
    #[arg(long, default_value_t = default_camera_sensitivity())]
    #[serde(default = "default_camera_sensitivity")]
    pub camera_sensitivity: f32,

    /// Estimate a ground-plane up direction once from bounded scene samples for flycam navigation.
    #[arg(long, action = clap::ArgAction::SetTrue)]
    #[serde(default)]
    pub match_ground_plane: bool,

    #[arg(
        long,
        default_value = "true",
        action = clap::ArgAction::Set,
        help = "show the world inspector (enabled by default)"
    )]
    pub editor: bool,

    #[arg(long, default_value = "true")]
    pub press_esc_close: bool,

    #[arg(long, default_value = "true")]
    pub press_s_screenshot: bool,

    #[arg(long, default_value = "false")]
    pub show_axes: bool,

    #[arg(long, default_value = "true")]
    pub show_fps: bool,

    #[arg(long, default_value = "1920.0")]
    pub width: f32,

    #[arg(long, default_value = "1080.0")]
    pub height: f32,

    #[arg(long, default_value = "bevy_gaussian_splatting")]
    pub name: String,

    #[arg(long, default_value = "1")]
    pub msaa_samples: u8,

    #[arg(long, default_value = None, help = "input file path (or url/base64_url if web_asset feature is enabled)")]
    pub input_cloud: Option<String>,

    #[arg(
        long,
        default_value = None,
        help = "secondary input file used when morph_interpolate is enabled",
    )]
    pub input_cloud_target: Option<String>,

    #[arg(long, default_value = None, help = "input glTF/GLB scene path (or url/base64_url if web_asset feature is enabled)")]
    pub input_scene: Option<String>,

    #[cfg(not(target_arch = "wasm32"))]
    #[arg(
        long,
        help = "standard 3DGS camera JSON path; preserves authored orientation and intrinsics"
    )]
    pub camera_path: Option<String>,

    #[cfg(not(target_arch = "wasm32"))]
    #[arg(
        long,
        default_value_t = 0,
        help = "zero-based camera-path frame to open"
    )]
    pub camera_path_index: usize,

    #[cfg(not(target_arch = "wasm32"))]
    #[arg(
        long,
        default_value_t = 0.0,
        help = "camera-path playback frames per second; zero holds the selected frame"
    )]
    pub camera_path_fps: f64,

    #[cfg(feature = "lod")]
    #[arg(
        long,
        default_value = None,
        conflicts_with_all = ["input_cloud", "input_scene"],
        help = "prebuilt .gsplatlod manifest path or URL (pages resolve beside it)"
    )]
    pub input_lod: Option<String>,

    #[arg(long, default_value = None, help = "cloud translation as x,y,z")]
    pub cloud_translation: Option<String>,

    #[arg(long, default_value = None, help = "cloud rotation in degrees as x,y,z")]
    pub cloud_rotation: Option<String>,

    #[arg(long, default_value = None, help = "cloud scale as uniform or x,y,z")]
    pub cloud_scale: Option<String>,

    #[arg(long, default_value = "0")]
    pub gaussian_count: usize,

    #[arg(long, default_value = None, help = "seed for random gaussian generation")]
    pub gaussian_seed: Option<u64>,

    #[arg(long, value_enum, default_value_t = GaussianMode::Gaussian3d)]
    pub gaussian_mode: GaussianMode,

    #[arg(long, value_enum, default_value_t = PlaybackMode::Still)]
    pub playback_mode: PlaybackMode,

    #[arg(long, value_enum, default_value_t = RasterizeMode::Color)]
    pub rasterization_mode: RasterizeMode,

    #[arg(long, value_enum, default_value_t = RadixSortDepthBits::Bits32)]
    pub radix_sort_depth_bits: RadixSortDepthBits,

    #[cfg(feature = "lod")]
    #[command(flatten)]
    #[serde(flatten)]
    pub lod: GaussianLodViewerArgs,

    #[cfg(lod_render_path)]
    #[command(flatten)]
    #[serde(flatten)]
    pub point: GaussianPointSplattingViewerArgs,

    #[cfg(lod_render_path)]
    #[command(flatten)]
    #[serde(flatten)]
    pub ordered: GaussianGlobalOrderViewerArgs,

    #[arg(long, default_value = "0")]
    pub particle_count: usize,
}

impl Default for GaussianSplattingViewer {
    fn default() -> GaussianSplattingViewer {
        GaussianSplattingViewer {
            camera_controller: CameraController::default(),
            camera_speed: default_camera_speed(),
            camera_sensitivity: default_camera_sensitivity(),
            match_ground_plane: false,
            editor: true,
            press_esc_close: true,
            press_s_screenshot: true,
            show_axes: false,
            show_fps: true,
            width: 1920.0,
            height: 1080.0,
            name: "bevy_gaussian_splatting".to_string(),
            msaa_samples: 1,
            input_cloud: None,
            input_cloud_target: None,
            input_scene: None,
            #[cfg(not(target_arch = "wasm32"))]
            camera_path: None,
            #[cfg(not(target_arch = "wasm32"))]
            camera_path_index: 0,
            #[cfg(not(target_arch = "wasm32"))]
            camera_path_fps: 0.0,
            #[cfg(feature = "lod")]
            input_lod: None,
            cloud_translation: None,
            cloud_rotation: None,
            cloud_scale: None,
            gaussian_count: 0,
            gaussian_seed: None,
            gaussian_mode: GaussianMode::Gaussian3d,
            playback_mode: PlaybackMode::Still,
            rasterization_mode: RasterizeMode::Color,
            radix_sort_depth_bits: RadixSortDepthBits::Bits32,
            #[cfg(feature = "lod")]
            lod: GaussianLodViewerArgs::default(),
            #[cfg(lod_render_path)]
            point: GaussianPointSplattingViewerArgs::default(),
            #[cfg(lod_render_path)]
            ordered: GaussianGlobalOrderViewerArgs::default(),
            particle_count: 0,
        }
    }
}

impl GaussianSplattingViewer {
    /// Validate CLI and deserialized query/config values before installing camera input.
    pub fn camera_control_settings(&self) -> Result<(f32, f32), String> {
        if self.match_ground_plane && self.camera_controller != CameraController::Flycam {
            return Err("--match-ground-plane requires --camera-controller flycam".into());
        }
        if !self.camera_speed.is_finite()
            || !(0.0..=1_000_000.0).contains(&self.camera_speed)
            || self.camera_speed == 0.0
        {
            return Err("camera-speed must be finite and in (0, 1000000]".into());
        }
        if !self.camera_sensitivity.is_finite()
            || !(0.0..=1.0).contains(&self.camera_sensitivity)
            || self.camera_sensitivity == 0.0
        {
            return Err("camera-sensitivity must be finite and in (0, 1]".into());
        }
        Ok((self.camera_speed, self.camera_sensitivity))
    }

    /// Builds the cloud component represented by the viewer CLI and validates
    /// it before any render-world allocations can observe it.
    #[cfg(feature = "lod")]
    pub fn lod_settings(&self) -> Result<GaussianLodSettings, LodSettingsError> {
        let lod = &self.lod;
        let mut settings = GaussianLodSettings {
            quality: lod.lod_quality,
            hysteresis: VIEWER_DEFAULT_LOD_HYSTERESIS,
            selection_mode: if lod.lod_freeze {
                LodSelectionMode::Frozen
            } else {
                LodSelectionMode::Dynamic
            },
            ..default()
        };
        settings.budgets.max_resident_gaussians = lod.lod_max_resident_gaussians;
        settings.budgets.max_resident_bytes = lod.lod_max_resident_bytes;
        settings.budgets.max_active_gaussians = lod
            .lod_max_active_gaussians
            .min(settings.budgets.max_resident_gaussians);
        #[cfg(lod_render_path)]
        if self.point.point_splatting || self.ordered.global_order {
            settings.presentation_mode = if self.ordered.lod_spatial_transitions {
                crate::gaussian::lod_settings::LodPresentationMode::ContinuousMorph
            } else {
                crate::gaussian::lod_settings::LodPresentationMode::Discrete
            };
        }
        settings.validate()?;
        Ok(settings)
    }

    /// Shared allocation ceilings validated before either world creates LoD resources.
    #[cfg(feature = "lod")]
    pub fn lod_memory_limits(&self) -> Result<LodMemoryLimits, LodSettingsError> {
        let limits = LodMemoryLimits {
            max_cpu_bytes: self.lod.lod_max_cpu_bytes,
            max_gpu_bytes: self.lod.lod_max_gpu_bytes,
        };
        for (field, value) in [
            ("lod_max_cpu_bytes", limits.max_cpu_bytes),
            ("lod_max_gpu_bytes", limits.max_gpu_bytes),
        ] {
            if value == 0 {
                return Err(LodSettingsError::ZeroBudget(field));
            }
        }
        Ok(limits)
    }

    /// Validates the opt-in camera backend before the viewer creates a device.
    #[cfg(lod_render_path)]
    pub fn point_splatting_settings(
        &self,
    ) -> Result<Option<crate::render::point::GaussianPointSplattingSettings>, String> {
        if !self.point.point_splatting {
            return Ok(None);
        }
        if self.ordered.global_order {
            return Err("--point-splatting conflicts with --global-order".to_owned());
        }
        if self.point.lod_gpu_traversal && self.input_lod.is_none() {
            return Err("--lod-gpu-traversal requires --input-lod".to_owned());
        }
        if self.gaussian_mode != GaussianMode::Gaussian3d {
            return Err("--point-splatting requires --gaussian-mode gaussian3d".to_owned());
        }
        if self.rasterization_mode != RasterizeMode::Color {
            return Err("--point-splatting requires --rasterization-mode color".to_owned());
        }
        if self.lod_debug_settings() != LodDebugSettings::default() {
            return Err("--point-splatting requires --lod-debug off".to_owned());
        }
        if self.input_cloud_target.is_some() {
            return Err(
                "--point-splatting does not support --input-cloud-target interpolation".to_owned(),
            );
        }
        let settings = crate::render::point::GaussianPointSplattingSettings {
            samples_per_pixel: self.point.point_samples_per_pixel,
            min_samples_per_pixel: self.point.point_min_samples_per_pixel,
            max_points_per_frame: self.point.point_max_points,
            max_projected_gaussians: self.point.point_max_projected_gaussians,
            max_gpu_bytes: self.point.point_max_gpu_bytes,
            target_gpu_ms: self.point.point_target_gpu_ms,
            ..default()
        };
        settings.validate().map_err(|error| error.to_string())?;
        Ok(Some(settings))
    }

    #[cfg(lod_render_path)]
    pub fn global_order_settings(
        &self,
    ) -> Result<Option<crate::render::ordered::GaussianGlobalOrderSettings>, String> {
        if !self.ordered.global_order {
            return Ok(None);
        }
        if self.point.point_splatting {
            return Err("--global-order conflicts with --point-splatting".to_owned());
        }
        if self.gaussian_mode != GaussianMode::Gaussian3d
            || self.rasterization_mode != RasterizeMode::Color
            || self.lod_debug_settings() != LodDebugSettings::default()
            || self.input_cloud_target.is_some()
        {
            return Err("--global-order requires 3D color clouds, --lod-debug off and no interpolation target".to_owned());
        }
        let settings = crate::render::ordered::GaussianGlobalOrderSettings {
            max_projected_gaussians: self.ordered.global_order_max_gaussians,
            max_gpu_bytes: self.ordered.global_order_max_gpu_bytes,
        };
        settings.validate().map_err(str::to_owned)?;
        Ok(Some(settings))
    }

    #[cfg(lod_render_path)]
    pub fn gpu_lod_traversal_settings(
        &self,
    ) -> Result<Option<crate::render::traversal::GpuLodTraversalSettings>, String> {
        if !self.point.lod_gpu_traversal {
            return Ok(None);
        }
        if self.input_lod.is_none() {
            return Err("--lod-gpu-traversal requires --input-lod".to_owned());
        }
        let projected_capacity = self
            .point_splatting_settings()?
            .map(|settings| settings.max_projected_gaussians)
            .or(self
                .global_order_settings()?
                .map(|settings| settings.max_projected_gaussians))
            .ok_or_else(|| {
                "--lod-gpu-traversal requires --point-splatting or --global-order".to_owned()
            })?;
        let lod = self.lod_settings().map_err(|error| error.to_string())?;
        if lod.selection_mode == LodSelectionMode::Frozen {
            return Err("--lod-gpu-traversal does not support --lod-freeze".to_owned());
        }
        let settings = crate::render::traversal::GpuLodTraversalSettings {
            max_selected_gaussians: lod
                .budgets
                .max_active_gaussians
                .min(u64::from(projected_capacity)) as u32,
            max_page_requests: self.point.lod_max_page_requests,
            ..default()
        };
        settings.validate().map_err(|error| error.to_string())?;
        Ok(Some(settings))
    }

    #[cfg(lod_render_path)]
    pub fn spatial_transition_settings(
        &self,
    ) -> Result<Option<crate::render::spatial_morph::GaussianLodSpatialTransitionSettings>, String>
    {
        if !self.ordered.lod_spatial_transitions {
            return Ok(None);
        }
        if !self.ordered.global_order || !self.point.lod_gpu_traversal || self.input_lod.is_none() {
            return Err("--lod-spatial-transitions requires --global-order, --lod-gpu-traversal and --input-lod".into());
        }
        self.global_order_settings()?;
        self.gpu_lod_traversal_settings()?;
        let settings = crate::render::spatial_morph::GaussianLodSpatialTransitionSettings {
            max_transition_nodes: self.ordered.lod_max_transition_nodes,
            max_transition_records: self.ordered.lod_max_transition_records,
            max_mapping_bytes: self.ordered.lod_max_mapping_bytes,
        };
        settings.validate().map_err(str::to_owned)?;
        Ok(Some(settings))
    }

    /// Builds the viewer's per-package transport policy without changing the
    /// reusable library default.
    #[cfg(feature = "lod")]
    pub fn lod_streaming_settings(&self) -> Result<GaussianStreamingSettings, LodSettingsError> {
        let settings = GaussianStreamingSettings {
            max_concurrent_requests: self.lod.lod_max_concurrent_requests,
            ..default()
        };
        settings.validate()?;
        Ok(settings)
    }

    /// Builds the optional named cloud annotation preset.
    #[cfg(feature = "lod")]
    pub fn lod_debug_settings(&self) -> LodDebugSettings {
        LodDebugSettings::from_preset(self.lod.lod_debug.unwrap_or_default())
    }

    pub fn cloud_transform(&self) -> Transform {
        let mut transform = Transform::default();

        if let Some(translation) = self.cloud_translation.as_deref().and_then(parse_vec3) {
            transform.translation = translation;
        }

        if let Some(rotation) = self.cloud_rotation.as_deref().and_then(parse_vec3) {
            transform.rotation = Quat::from_euler(
                EulerRot::XYZ,
                rotation.x.to_radians(),
                rotation.y.to_radians(),
                rotation.z.to_radians(),
            );
        }

        if let Some(scale) = self.cloud_scale.as_deref().and_then(parse_scale) {
            transform.scale = scale;
        }

        transform
    }
}

fn parse_vec3(value: &str) -> Option<Vec3> {
    let parts: Vec<&str> = value
        .split(&[',', ' ', '\t'][..])
        .filter(|part| !part.is_empty())
        .collect();
    if parts.len() != 3 {
        return None;
    }

    let x = parts[0].parse::<f32>().ok()?;
    let y = parts[1].parse::<f32>().ok()?;
    let z = parts[2].parse::<f32>().ok()?;

    Some(Vec3::new(x, y, z))
}

fn parse_scale(value: &str) -> Option<Vec3> {
    let parts: Vec<&str> = value
        .split(&[',', ' ', '\t'][..])
        .filter(|part| !part.is_empty())
        .collect();
    if parts.is_empty() {
        return None;
    }

    if parts.len() == 1 {
        let v = parts[0].parse::<f32>().ok()?;
        return Some(Vec3::splat(v));
    }

    if parts.len() != 3 {
        return None;
    }

    let x = parts[0].parse::<f32>().ok()?;
    let y = parts[1].parse::<f32>().ok()?;
    let z = parts[2].parse::<f32>().ok()?;

    Some(Vec3::new(x, y, z))
}

#[cfg(test)]
mod viewer_cli_tests {
    use super::*;

    #[test]
    fn camera_controls_round_trip_and_validate_cli_and_config_values() {
        let defaults = GaussianSplattingViewer::try_parse_from(["viewer"]).unwrap();
        assert_eq!(defaults.camera_controller, CameraController::Orbit);
        assert!(!defaults.match_ground_plane);
        assert!(
            GaussianSplattingViewer::try_parse_from(["viewer", "--match-ground-plane"])
                .unwrap()
                .camera_control_settings()
                .unwrap_err()
                .contains("requires --camera-controller flycam")
        );
        assert_eq!(defaults.camera_control_settings().unwrap(), (12.0, 0.00012));
        let configured = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--camera-controller=flycam",
            "--camera-speed=200",
            "--camera-sensitivity=0.0002",
            "--match-ground-plane",
        ])
        .unwrap();
        let mut value = serde_json::to_value(configured).unwrap();
        assert_eq!(value["camera_controller"], "flycam");
        let restored: GaussianSplattingViewer = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(restored.camera_controller, CameraController::Flycam);
        assert!(restored.match_ground_plane);
        assert_eq!(restored.camera_control_settings().unwrap(), (200.0, 0.0002));
        for field in ["camera_speed", "camera_sensitivity"] {
            value[field] = serde_json::json!(0.0);
            assert!(
                serde_json::from_value::<GaussianSplattingViewer>(value.clone())
                    .unwrap()
                    .camera_control_settings()
                    .is_err()
            );
            value[field] = serde_json::to_value(if field == "camera_speed" {
                200.0
            } else {
                0.0002
            })
            .unwrap();
        }
        for argument in [
            "--camera-speed=NaN",
            "--camera-speed=1000001",
            "--camera-sensitivity=inf",
            "--camera-sensitivity=2",
        ] {
            assert!(
                GaussianSplattingViewer::try_parse_from(["viewer", argument])
                    .unwrap()
                    .camera_control_settings()
                    .is_err()
            );
        }
        let object = value.as_object_mut().unwrap();
        for field in [
            "camera_controller",
            "camera_speed",
            "camera_sensitivity",
            "match_ground_plane",
        ] {
            object.remove(field);
        }
        let old_config: GaussianSplattingViewer = serde_json::from_value(value).unwrap();
        assert_eq!(old_config.camera_controller, CameraController::Orbit);
        assert!(!old_config.match_ground_plane);
        assert_eq!(
            old_config.camera_control_settings().unwrap(),
            defaults.camera_control_settings().unwrap()
        );
    }

    #[test]
    fn editor_cli_defaults_on_and_accepts_explicit_disable() {
        let defaults = GaussianSplattingViewer::try_parse_from(["viewer"])
            .expect("viewer defaults should parse");
        assert!(defaults.editor);

        let disabled = GaussianSplattingViewer::try_parse_from(["viewer", "--editor=false"])
            .expect("the default-on editor should accept an explicit opt-out");
        assert!(!disabled.editor);
    }

    #[cfg(lod_render_path)]
    #[test]
    fn point_cli_opt_in_round_trips_and_preserves_lod_quality() {
        let defaults = GaussianSplattingViewer::try_parse_from(["viewer"]).unwrap();
        assert!(defaults.point_splatting_settings().unwrap().is_none());
        assert_eq!(defaults.point.point_samples_per_pixel, 4);
        assert_eq!(defaults.point.point_min_samples_per_pixel, 1);
        assert_eq!(defaults.point.point_max_points, 16_777_216);
        assert_eq!(defaults.point.point_target_gpu_ms, None);
        assert_eq!(defaults.point.lod_max_page_requests, 1024);

        let configured = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--point-splatting",
            "--point-samples-per-pixel",
            "2",
            "--point-min-samples-per-pixel",
            "2",
            "--point-max-points",
            "1000000",
            "--point-target-gpu-ms",
            "6.5",
            "--lod-quality",
            "0.375",
            "--lod-max-active-gaussians",
            "500000",
        ])
        .unwrap();
        let point = configured.point_splatting_settings().unwrap().unwrap();
        assert_eq!(point.samples_per_pixel, 2);
        assert_eq!(point.min_samples_per_pixel, 2);
        assert_eq!(point.max_points_per_frame, 1_000_000);
        assert_eq!(point.target_gpu_ms, Some(6.5));
        let lod = configured.lod_settings().unwrap();
        assert_eq!(
            lod.presentation_mode,
            crate::gaussian::lod_settings::LodPresentationMode::Discrete
        );
        assert_eq!(lod.quality, 0.375);
        assert_eq!(lod.budgets.max_active_gaussians, 500_000);

        let value = serde_json::to_value(&configured).unwrap();
        assert!(!value.as_object().unwrap().contains_key("point"));
        assert_eq!(value["point_splatting"], true);
        let restored: GaussianSplattingViewer = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(restored.point_splatting_settings().unwrap(), Some(point));
        let mut omitted_options = value;
        omitted_options
            .as_object_mut()
            .unwrap()
            .retain(|key, _| !key.starts_with("point_"));
        let restored: GaussianSplattingViewer = serde_json::from_value(omitted_options).unwrap();
        assert!(restored.point_splatting_settings().unwrap().is_none());
        assert_eq!(restored.point.point_samples_per_pixel, 4);

        let gpu = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--point-splatting",
            "--lod-gpu-traversal",
            "--input-lod",
            "scene.gsplatlod",
            "--lod-max-active-gaussians",
            "500000",
            "--point-max-projected-gaussians",
            "250000",
        ])
        .unwrap();
        assert_eq!(
            gpu.gpu_lod_traversal_settings()
                .unwrap()
                .unwrap()
                .max_selected_gaussians,
            250_000
        );
        let missing_package = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--point-splatting",
            "--lod-gpu-traversal",
        ])
        .unwrap();
        assert!(missing_package.gpu_lod_traversal_settings().is_err());
        let missing_renderer = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--lod-gpu-traversal",
            "--input-lod",
            "scene.gsplatlod",
        ])
        .unwrap();
        assert!(missing_renderer.gpu_lod_traversal_settings().is_err());
        let frozen = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--point-splatting",
            "--lod-gpu-traversal",
            "--input-lod",
            "scene.gsplatlod",
            "--lod-freeze",
        ])
        .unwrap();
        assert_eq!(
            frozen.gpu_lod_traversal_settings().unwrap_err(),
            "--lod-gpu-traversal does not support --lod-freeze"
        );
    }

    #[cfg(lod_render_path)]
    #[test]
    fn global_order_cli_selects_discrete_and_rejects_backend_conflicts() {
        let configured = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--global-order",
            "--global-order-max-gaussians",
            "4096",
        ])
        .unwrap();
        assert_eq!(
            configured
                .global_order_settings()
                .unwrap()
                .unwrap()
                .max_projected_gaussians,
            4096
        );
        assert_eq!(
            configured.lod_settings().unwrap().presentation_mode,
            crate::gaussian::lod_settings::LodPresentationMode::Discrete
        );
        let gpu = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--global-order",
            "--lod-gpu-traversal",
            "--input-lod",
            "scene.gsplatlod",
            "--global-order-max-gaussians",
            "16000000",
            "--lod-max-active-gaussians=16000000",
            "--lod-max-resident-gaussians=24000000",
        ])
        .unwrap();
        assert_eq!(
            gpu.gpu_lod_traversal_settings()
                .unwrap()
                .unwrap()
                .max_selected_gaussians,
            16_000_000
        );
        let spatial = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--global-order",
            "--lod-gpu-traversal",
            "--input-lod",
            "scene.gsplatlod",
            "--lod-spatial-transitions",
            "--lod-max-transition-records",
            "262144",
            "--lod-max-mapping-bytes",
            "134217728",
            "--lod-max-page-requests",
            "4096",
        ])
        .unwrap();
        assert_eq!(
            spatial.lod_settings().unwrap().presentation_mode,
            crate::gaussian::lod_settings::LodPresentationMode::ContinuousMorph
        );
        let transitions = spatial.spatial_transition_settings().unwrap().unwrap();
        assert_eq!(transitions.max_transition_records, 262144);
        assert_eq!(transitions.max_mapping_bytes, 134217728);
        assert_eq!(
            spatial
                .gpu_lod_traversal_settings()
                .unwrap()
                .unwrap()
                .max_page_requests,
            4096
        );
        let mut roundtrip: GaussianSplattingViewer =
            serde_json::from_value(serde_json::to_value(&spatial).unwrap()).unwrap();
        assert_eq!(
            roundtrip.spatial_transition_settings().unwrap(),
            Some(transitions)
        );
        assert_eq!(
            roundtrip
                .gpu_lod_traversal_settings()
                .unwrap()
                .unwrap()
                .max_page_requests,
            4096
        );
        let mut omitted = serde_json::to_value(&roundtrip).unwrap();
        omitted
            .as_object_mut()
            .unwrap()
            .remove("lod_max_page_requests");
        let omitted: GaussianSplattingViewer = serde_json::from_value(omitted).unwrap();
        assert_eq!(
            omitted
                .gpu_lod_traversal_settings()
                .unwrap()
                .unwrap()
                .max_page_requests,
            1024
        );
        assert!(
            GaussianSplattingViewer::try_parse_from(["viewer", "--lod-max-page-requests", "4096"])
                .is_err()
        );
        roundtrip.point.lod_gpu_traversal = false;
        assert!(roundtrip.spatial_transition_settings().is_err());
        assert!(
            GaussianSplattingViewer::try_parse_from(["viewer", "--lod-spatial-transitions",])
                .is_err()
        );
        assert!(gpu.spatial_transition_settings().unwrap().is_none());
        assert!(
            GaussianSplattingViewer::try_parse_from([
                "viewer",
                "--global-order",
                "--point-splatting",
            ])
            .is_err()
        );
        let mut restored: GaussianSplattingViewer =
            serde_json::from_value(serde_json::to_value(configured).unwrap()).unwrap();
        assert!(restored.global_order_settings().unwrap().is_some());
        restored.point.point_splatting = true;
        assert!(restored.global_order_settings().is_err());
        assert!(restored.point_splatting_settings().is_err());
    }

    #[cfg(lod_render_path)]
    #[test]
    fn point_cli_rejects_invalid_budgets_and_unsupported_presentations() {
        for args in [
            ["--point-samples-per-pixel", "0"],
            ["--point-samples-per-pixel", "9"],
            ["--point-max-points", "0"],
            ["--point-max-points", "1073741825"],
            ["--point-target-gpu-ms", "NaN"],
            ["--point-target-gpu-ms", "0"],
            ["--gaussian-mode", "gaussian4d"],
            ["--rasterization-mode", "depth"],
            ["--lod-debug", "page"],
            ["--input-cloud-target", "target.ply"],
        ] {
            let viewer = GaussianSplattingViewer::try_parse_from([
                "viewer",
                "--point-splatting",
                args[0],
                args[1],
            ])
            .unwrap();
            assert!(
                viewer.point_splatting_settings().is_err(),
                "accepted {args:?}"
            );
        }
    }
}

pub fn setup_hooks() {
    #[cfg(debug_assertions)]
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
    }
}

pub fn log(_msg: &str) {
    #[cfg(debug_assertions)]
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::log_1(&_msg.into());
    }
    #[cfg(debug_assertions)]
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("{_msg}");
    }
}

#[cfg(all(test, feature = "lod"))]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn lod_cli_defaults_promote_only_the_viewer_active_budget() {
        let viewer = GaussianSplattingViewer::try_parse_from(["viewer"])
            .expect("viewer defaults should parse");
        let settings = viewer
            .lod_settings()
            .expect("default viewer LoD policy is valid");
        let library_default = GaussianLodSettings::default();
        assert_eq!(library_default.budgets.max_active_gaussians, 2_000_000);
        assert_ne!(
            library_default.hysteresis, VIEWER_DEFAULT_LOD_HYSTERESIS,
            "the viewer policy must not change the reusable library default"
        );
        let mut expected = library_default;
        expected.hysteresis = VIEWER_DEFAULT_LOD_HYSTERESIS;
        expected.budgets.max_active_gaussians = VIEWER_DEFAULT_LOD_MAX_ACTIVE_GAUSSIANS;
        assert_eq!(settings, expected);
        assert_eq!(
            viewer.lod_memory_limits().unwrap(),
            LodMemoryLimits::default()
        );
        assert!(settings.budgets.max_active_gaussians <= settings.budgets.max_resident_gaussians);
        let streaming = viewer
            .lod_streaming_settings()
            .expect("default viewer streaming policy is valid");
        assert_eq!(
            GaussianStreamingSettings::default().max_concurrent_requests,
            8
        );
        assert_eq!(
            streaming.max_concurrent_requests,
            VIEWER_DEFAULT_LOD_MAX_CONCURRENT_REQUESTS
        );
    }

    #[test]
    fn lod_query_fields_remain_flattened_and_round_trip() {
        const LOD_QUERY_FIELDS: [&str; 10] = [
            "lod_quality",
            "lod_max_active_gaussians",
            "lod_max_resident_gaussians",
            "lod_max_resident_bytes",
            "lod_max_cpu_bytes",
            "lod_max_gpu_bytes",
            "lod_max_concurrent_requests",
            "lod_max_manifest_bytes",
            "lod_freeze",
            "lod_debug",
        ];

        let mut viewer = GaussianSplattingViewer::default();
        viewer.lod.lod_quality = 0.375;
        viewer.lod.lod_max_active_gaussians = 3_500_000;
        viewer.lod.lod_max_resident_gaussians = 24_000_000;
        viewer.lod.lod_max_resident_bytes = 4_294_967_296;
        viewer.lod.lod_max_cpu_bytes = 8_589_934_592;
        viewer.lod.lod_max_gpu_bytes = 8_589_934_592;
        viewer.lod.lod_max_concurrent_requests = 24;
        let serialized = serde_json::to_value(&viewer).expect("viewer should serialize");
        let object = serialized
            .as_object()
            .expect("viewer should serialize as a JSON object");

        assert!(!object.contains_key("lod"));
        assert_eq!(
            object
                .keys()
                .filter(|field| field.starts_with("lod_"))
                .count(),
            LOD_QUERY_FIELDS.len() + usize::from(cfg!(lod_render_path))
        );
        for field in LOD_QUERY_FIELDS {
            assert!(
                object.contains_key(field),
                "missing flattened field {field}"
            );
        }

        let decoded: GaussianSplattingViewer = serde_json::from_value(serialized.clone())
            .expect("flattened viewer should deserialize");
        assert_eq!(decoded.lod.lod_quality, 0.375);
        assert_eq!(decoded.lod.lod_max_active_gaussians, 3_500_000);
        assert_eq!(decoded.lod.lod_max_resident_gaussians, 24_000_000);
        assert_eq!(decoded.lod.lod_max_resident_bytes, 4_294_967_296);
        assert_eq!(
            decoded.lod_memory_limits().unwrap(),
            viewer.lod_memory_limits().unwrap()
        );
        assert_eq!(decoded.lod.lod_max_concurrent_requests, 24);

        let mut omitted_options = serialized;
        let omitted_options = omitted_options
            .as_object_mut()
            .expect("viewer should remain an object");
        for field in LOD_QUERY_FIELDS {
            omitted_options.remove(field);
        }
        let omitted_options = serde_json::Value::Object(omitted_options.clone());
        let decoded: GaussianSplattingViewer = serde_json::from_value(omitted_options)
            .expect("omitted LoD options should use defaults");
        assert_eq!(decoded.lod.lod_quality, 1.0);
        assert_eq!(
            decoded.lod_memory_limits().unwrap(),
            LodMemoryLimits::default()
        );
        assert_eq!(
            decoded.lod_settings().unwrap().budgets,
            GaussianSplattingViewer::default()
                .lod_settings()
                .unwrap()
                .budgets
        );
        assert_eq!(
            decoded.lod.lod_max_active_gaussians,
            VIEWER_DEFAULT_LOD_MAX_ACTIVE_GAUSSIANS
        );
        assert_eq!(
            decoded.lod.lod_max_concurrent_requests,
            VIEWER_DEFAULT_LOD_MAX_CONCURRENT_REQUESTS
        );
        assert!(!decoded.lod.lod_freeze);
        assert_eq!(decoded.lod.lod_debug, None);
    }

    #[test]
    fn lod_cli_overrides_construct_the_promoted_policy() {
        let viewer = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--lod-quality=0.25",
            "--lod-max-active-gaussians=16000000",
            "--lod-max-resident-gaussians=24000000",
            "--lod-max-resident-bytes=4294967296",
            "--lod-max-cpu-bytes=8589934592",
            "--lod-max-gpu-bytes=8589934592",
            "--lod-max-concurrent-requests=32",
            "--lod-max-manifest-bytes=268435456",
            "--lod-freeze",
        ])
        .expect("valid LoD CLI overrides should parse");
        let settings = viewer.lod_settings().expect("overrides should validate");

        assert_eq!(settings.quality, 0.25);
        assert_eq!(viewer.lod.lod_max_manifest_bytes, 268_435_456);
        assert_eq!(settings.budgets.max_active_gaussians, 16_000_000);
        assert_eq!(settings.budgets.max_resident_gaussians, 24_000_000);
        assert_eq!(settings.budgets.max_resident_bytes, 4_294_967_296);
        assert_eq!(
            viewer.lod_memory_limits().unwrap(),
            LodMemoryLimits {
                max_cpu_bytes: 8_589_934_592,
                max_gpu_bytes: 8_589_934_592,
            }
        );
        assert_eq!(settings.selection_mode, LodSelectionMode::Frozen);
        assert_eq!(
            viewer
                .lod_streaming_settings()
                .expect("streaming override should validate")
                .max_concurrent_requests,
            32
        );
    }

    #[test]
    fn lod_active_budget_override_is_validated_and_bounded_by_residency() {
        let resident_capacity = GaussianLodSettings::default()
            .budgets
            .max_resident_gaussians;
        let clamped = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--lod-max-active-gaussians=18446744073709551615",
        ])
        .expect("u64 override should parse")
        .lod_settings()
        .expect("an oversized override should clamp to resident capacity");
        assert_eq!(clamped.budgets.max_active_gaussians, resident_capacity);

        let zero =
            GaussianSplattingViewer::try_parse_from(["viewer", "--lod-max-active-gaussians=0"])
                .expect("zero is syntactically an integer");
        assert!(matches!(
            zero.lod_settings(),
            Err(LodSettingsError::ZeroBudget("budgets.max_active_gaussians"))
        ));
    }

    #[test]
    fn lod_transport_concurrency_is_validated_in_the_documented_range() {
        let zero =
            GaussianSplattingViewer::try_parse_from(["viewer", "--lod-max-concurrent-requests=0"])
                .expect("zero is syntactically an integer");
        assert!(matches!(
            zero.lod_streaming_settings(),
            Err(LodSettingsError::ZeroBudget(
                "streaming.max_concurrent_requests"
            ))
        ));

        let oversized = GaussianSplattingViewer::try_parse_from([
            "viewer",
            "--lod-max-concurrent-requests=257",
        ])
        .expect("257 is syntactically an integer");
        assert!(matches!(
            oversized.lod_streaming_settings(),
            Err(LodSettingsError::OutOfRange {
                field: "streaming.max_concurrent_requests",
                min: "1",
                max: "256",
            })
        ));

        for valid in [1, 256] {
            let argument = format!("--lod-max-concurrent-requests={valid}");
            let viewer = GaussianSplattingViewer::try_parse_from(["viewer", argument.as_str()])
                .expect("concurrency is syntactically an integer");
            assert_eq!(
                viewer
                    .lod_streaming_settings()
                    .expect("boundary concurrency should validate")
                    .max_concurrent_requests,
                valid
            );
        }
    }

    #[test]
    fn lod_help_exposes_only_promoted_controls() {
        let help = GaussianSplattingViewer::command()
            .render_long_help()
            .to_string();
        for visible in [
            "--lod-quality",
            "--lod-max-active-gaussians",
            "--lod-max-resident-gaussians",
            "--lod-max-resident-bytes",
            "--lod-max-cpu-bytes",
            "--lod-max-gpu-bytes",
            "--lod-max-concurrent-requests",
            "--lod-freeze",
            "--lod-debug",
        ] {
            assert!(help.contains(visible), "missing primary control {visible}");
        }
        assert!(help.contains("[default: 8000000]"));
        assert!(help.contains("[default: 64]"));
        assert!(help.contains("transport requests"));
        assert!(help.contains("resident-record capacity"));
        for hidden in ["--lod-enabled", "--lod-debug-color", "--lod-hysteresis"] {
            assert!(
                !help.contains(hidden),
                "removed control leaked into help: {hidden}"
            );
        }
    }

    #[test]
    fn lod_cli_semantic_errors_are_rejected_before_attachment() {
        let invalid_quality =
            GaussianSplattingViewer::try_parse_from(["viewer", "--lod-quality=NaN"])
                .expect("NaN is syntactically a float");
        assert!(matches!(
            invalid_quality.lod_settings(),
            Err(LodSettingsError::NonFinite("quality"))
        ));
        let mut invalid = GaussianSplattingViewer::default();
        invalid.lod.lod_max_resident_bytes = 0;
        assert!(matches!(
            invalid.lod_settings(),
            Err(LodSettingsError::ZeroBudget("budgets.max_resident_bytes"))
        ));
        invalid.lod.lod_max_resident_bytes = LodBudgets::default().max_resident_bytes;
        invalid.lod.lod_max_resident_gaussians = 0;
        assert!(matches!(
            invalid.lod_settings(),
            Err(LodSettingsError::ZeroBudget(_))
        ));
        invalid.lod.lod_max_cpu_bytes = 0;
        assert!(matches!(
            invalid.lod_memory_limits(),
            Err(LodSettingsError::ZeroBudget("lod_max_cpu_bytes"))
        ));
        invalid.lod.lod_max_cpu_bytes = LodMemoryLimits::default().max_cpu_bytes;
        invalid.lod.lod_max_gpu_bytes = 0;
        assert!(matches!(
            invalid.lod_memory_limits(),
            Err(LodSettingsError::ZeroBudget("lod_max_gpu_bytes"))
        ));
    }

    #[test]
    fn lod_debug_cli_uses_named_presets() {
        let defaults = GaussianSplattingViewer::try_parse_from(["viewer"]).unwrap();
        assert_eq!(defaults.lod_debug_settings(), LodDebugSettings::default());

        let preset = GaussianSplattingViewer::try_parse_from(["viewer", "--lod-debug=level"])
            .unwrap()
            .lod_debug_settings();
        assert_eq!(preset.preset, LodDebugPreset::Level);

        let pressure =
            GaussianSplattingViewer::try_parse_from(["viewer", "--lod-debug=selection-pressure"])
                .unwrap()
                .lod_debug_settings();
        assert_eq!(pressure.preset, LodDebugPreset::SelectionPressure);
    }
}
