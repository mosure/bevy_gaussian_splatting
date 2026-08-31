use bevy::prelude::*;
use bevy_args::{Deserialize, Parser, Serialize};

use crate::gaussian::settings::{GaussianMode, PlaybackMode, RadixSortDepthBits, RasterizeMode};

#[cfg(feature = "lod")]
use crate::gaussian::lod_debug::{LodDebugPreset, LodDebugSettings};
#[cfg(feature = "lod")]
use crate::gaussian::lod_settings::{
    GaussianLodSettings, GaussianStreamingSettings, LodSelectionMode, LodSettingsError,
};

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

    #[arg(
        long,
        default_value_t = VIEWER_DEFAULT_LOD_MAX_CONCURRENT_REQUESTS,
        help = "maximum concurrent LoD page transport requests for standalone packages; lower this for constrained HTTP origins"
    )]
    pub lod_max_concurrent_requests: u32,

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
            lod_max_concurrent_requests: VIEWER_DEFAULT_LOD_MAX_CONCURRENT_REQUESTS,
            lod_freeze: false,
            lod_debug: None,
        }
    }
}

#[derive(Debug, Resource, Serialize, Deserialize, Parser)]
#[command(about = "bevy_gaussian_splatting viewer", version, long_about = None)]
pub struct GaussianSplattingViewer {
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

    #[arg(long, default_value = "0")]
    pub particle_count: usize,
}

impl Default for GaussianSplattingViewer {
    fn default() -> GaussianSplattingViewer {
        GaussianSplattingViewer {
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
            particle_count: 0,
        }
    }
}

impl GaussianSplattingViewer {
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
        settings.budgets.max_active_gaussians = lod
            .lod_max_active_gaussians
            .min(settings.budgets.max_resident_gaussians);
        settings.validate()?;
        Ok(settings)
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
    fn editor_cli_defaults_on_and_accepts_explicit_disable() {
        let defaults = GaussianSplattingViewer::try_parse_from(["viewer"])
            .expect("viewer defaults should parse");
        assert!(defaults.editor);

        let disabled = GaussianSplattingViewer::try_parse_from(["viewer", "--editor=false"])
            .expect("the default-on editor should accept an explicit opt-out");
        assert!(!disabled.editor);
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
        const LOD_QUERY_FIELDS: [&str; 5] = [
            "lod_quality",
            "lod_max_active_gaussians",
            "lod_max_concurrent_requests",
            "lod_freeze",
            "lod_debug",
        ];

        let mut viewer = GaussianSplattingViewer::default();
        viewer.lod.lod_quality = 0.375;
        viewer.lod.lod_max_active_gaussians = 3_500_000;
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
            LOD_QUERY_FIELDS.len()
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
        assert_eq!(decoded.lod.lod_max_concurrent_requests, 24);

        let mut legacy = serialized;
        let legacy = legacy
            .as_object_mut()
            .expect("viewer should remain an object");
        for field in LOD_QUERY_FIELDS {
            legacy.remove(field);
        }
        let legacy = serde_json::Value::Object(legacy.clone());
        let decoded: GaussianSplattingViewer = serde_json::from_value(legacy)
            .expect("pre-LoD flattened viewer should still deserialize");
        assert_eq!(decoded.lod.lod_quality, 1.0);
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
            "--lod-max-active-gaussians=3500000",
            "--lod-max-concurrent-requests=32",
            "--lod-freeze",
        ])
        .expect("valid LoD CLI overrides should parse");
        let settings = viewer.lod_settings().expect("overrides should validate");

        assert_eq!(settings.quality, 0.25);
        assert_eq!(settings.budgets.max_active_gaussians, 3_500_000);
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
