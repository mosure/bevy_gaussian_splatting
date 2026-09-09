//! Opt-in native capture of the actual LoD renderer. No systems are installed
//! unless [`run_capture`] is called by the `capture_lod` tool.
//!
//! Readback storage and queued results have fixed capacities. The camera path
//! advances independently of GPU mapping; overload is reported as missing samples.
//! Images, counters and timestamps are copied in the same view submission.

mod cadence;
mod gpu;
mod hierarchy;
mod ordered;
mod outcomes;
mod point;
pub(crate) mod source;
pub mod virtual_city;

pub use ordered::RuntimeOrderedGpuConfig;
pub use point::RuntimePointGpuConfig;
pub use source::{
    SourceMetadata, attribute_cut, convert_glb_to_ply, export_cohort, export_rung, fit_rung,
};

use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    asset::{AssetMetaCheck, AssetPlugin, RenderAssetUsages, UnapprovedPathMode},
    camera::{PerspectiveProjection, Projection, RenderTarget},
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        settings::{RenderCreation, WgpuSettings},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_interleave::prelude::Planar;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    CloudSettings, GaussianCamera, GaussianLodBuildSettings, GaussianLodHandle,
    GaussianLodPackageConfig, GaussianLodPackageSource, GaussianLodSettings,
    GaussianSplattingPlugin, GaussianStreamingSettings, LodPresentationMode, PlanarGaussian3d,
    build_planar_3d_lod,
    gaussian::formats::planar_3d_chunked::LodPageStorage,
    io::lod::{GaussianLodAsset, LodCodecLimits, decode_manifest, encode_manifest, encode_page},
    sort::SortMode,
    testing::{LodTestScene, lod_capture::*},
};

#[doc(hidden)]
pub use gpu::LodRuntimeDrawProbe;

type CaptureResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureCameraSegment {
    pub scenario: String,
    pub frames: u32,
    /// Exact zero-based row in `camera_path`; retains authored roll and fx/fy.
    #[serde(default)]
    pub camera_frame: Option<usize>,
    #[serde(default)]
    pub from: [f32; 3],
    #[serde(default)]
    pub to: [f32; 3],
    #[serde(default)]
    pub target: [f32; 3],
    /// Camera roll/up convention. Defaults to world Y for existing paths.
    #[serde(default = "default_camera_up")]
    pub up: [f32; 3],
}

/// Paths are relative to the configuration file. `synthetic` creates a small,
/// deterministic package and saves its source bytes beside the capture.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeCaptureConfig {
    #[serde(default)]
    pub capture_mode: RuntimeCaptureMode,
    /// Minimum headless schedule period. Zero preserves unpaced throughput
    /// captures; a positive period gives streaming real time between poses.
    /// This is CPU schedule pacing, not presentation or display-latency evidence.
    #[serde(default)]
    pub frame_period_ms: f64,
    /// First logical path frame eligible for regular readback. Startup probes
    /// remain enabled so loading readiness never depends on this threshold.
    #[serde(default)]
    pub capture_start_path_frame: u64,
    /// Flat cadence has no render attestation; wait this fixed time after the
    /// loop starts in addition to observing its resident main-world asset.
    #[serde(default = "default_cadence_warmup")]
    pub cadence_minimum_warmup_seconds: f64,
    pub output: PathBuf,
    pub manifest: Option<PathBuf>,
    pub source: Option<PathBuf>,
    pub synthetic: bool,
    #[serde(default)]
    pub render_mode: LodCapturePipeline,
    #[serde(default)]
    pub point_gpu: Option<RuntimePointGpuConfig>,
    #[serde(default)]
    pub ordered_gpu: Option<RuntimeOrderedGpuConfig>,
    #[serde(default)]
    pub spatial_transitions:
        Option<crate::render::spatial_morph::GaussianLodSpatialTransitionSettings>,
    #[serde(default)]
    pub camera_path: Option<PathBuf>,
    /// Per-run decoder admission; never changes the library's default limit.
    #[serde(default = "default_max_manifest_bytes")]
    pub max_manifest_bytes: u64,
    #[serde(default = "default_ledger_bytes")]
    pub max_cpu_bytes: u64,
    #[serde(default = "default_ledger_bytes")]
    pub max_gpu_bytes: u64,
    /// Package transport concurrency; omitted configs retain the library default.
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: u32,
    #[serde(default)]
    pub source_metadata: Option<PathBuf>,
    #[serde(default = "default_max_source_gaussians")]
    pub max_source_gaussians: u64,
    #[serde(default = "default_true")]
    pub capture_images: bool,
    /// Copy the bounded selected-node list beside the same-submission receipt.
    #[serde(default)]
    pub capture_hierarchy_cut: bool,
    pub builder_revision: String,
    pub viewport: [u32; 2],
    /// Render a physical-pixel tile of a calibrated full image. `viewport`
    /// remains the output tile size; focal lengths and pixel rays are preserved.
    #[serde(default)]
    pub camera_crop: Option<RuntimeCameraCrop>,
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub quality: f32,
    #[serde(default)]
    pub presentation_mode: LodPresentationMode,
    pub max_active_gaussians: u64,
    pub max_resident_gaussians: u64,
    pub max_resident_bytes: u64,
    pub max_resident_pages: u32,
    pub max_upload_bytes_per_frame: u64,
    pub capture_every: u32,
    pub readback_slots: u32,
    pub timeout_seconds: u64,
    pub request_timestamps: bool,
    pub segments: Vec<CaptureCameraSegment>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeCameraCrop {
    pub full_viewport: [u32; 2],
    pub origin: [u32; 2],
}

impl RuntimeCameraCrop {
    fn sub_view(self, size: [u32; 2]) -> bevy::camera::SubCameraView {
        bevy::camera::SubCameraView {
            full_size: UVec2::from_array(self.full_viewport),
            offset: UVec2::from_array(self.origin).as_vec2(),
            size: UVec2::from_array(size),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeCaptureMode {
    #[default]
    Instrumented,
    CadenceOnly,
}

fn default_camera_up() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

fn default_cadence_warmup() -> f64 {
    2.0
}

fn default_true() -> bool {
    true
}
fn default_max_source_gaussians() -> u64 {
    8_000_000
}

fn default_max_manifest_bytes() -> u64 {
    LodCodecLimits::default().max_manifest_bytes
}

fn default_ledger_bytes() -> u64 {
    4 * 1024 * 1024 * 1024
}

fn default_max_concurrent_requests() -> u32 {
    GaussianStreamingSettings::default().max_concurrent_requests
}

fn segment_min_length_squared(from: Vec3, to: Vec3) -> f32 {
    let delta = to - from;
    let t = if delta.length_squared() > 0.0 {
        (-from.dot(delta) / delta.length_squared()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    from.lerp(to, t).length_squared()
}

impl RuntimeCaptureConfig {
    fn validate(&self) -> CaptureResult<()> {
        if self.viewport.iter().any(|&v| v == 0 || v > 8192)
            || self.capture_every == 0
            || !(1..=8).contains(&self.readback_slots)
            || self.timeout_seconds == 0
            || self.timeout_seconds > 3600
            || !self.frame_period_ms.is_finite()
            || !(0.0..=1000.0).contains(&self.frame_period_ms)
            || !self.cadence_minimum_warmup_seconds.is_finite()
            || !(0.0..=60.0).contains(&self.cadence_minimum_warmup_seconds)
            || self
                .segments
                .iter()
                .map(|segment| u64::from(segment.frames))
                .sum::<u64>()
                > 100_000
            || self.segments.is_empty()
            || self.capture_start_path_frame
                >= self
                    .segments
                    .iter()
                    .map(|segment| u64::from(segment.frames))
                    .sum::<u64>()
            || self.builder_revision.trim().is_empty()
            || self.builder_revision.starts_with("REPLACE_")
            || !self.vertical_fov_radians.is_finite()
            || !(0.0..std::f32::consts::PI).contains(&self.vertical_fov_radians)
            || !self.near.is_finite()
            || self.near <= 0.0
            || self.max_source_gaussians == 0
            || (self.synthetic && self.manifest.is_some())
            || (!self.synthetic
                && self.render_mode != LodCapturePipeline::FlatSource
                && self.manifest.is_none())
            || (!self.synthetic && self.source.is_none())
            || !(default_max_manifest_bytes()..=256 * 1024 * 1024)
                .contains(&self.max_manifest_bytes)
            || self.max_cpu_bytes == 0
            || self.max_gpu_bytes == 0
            || (self.render_mode == LodCapturePipeline::HierarchyPoint) != self.point_gpu.is_some()
            || (self.render_mode == LodCapturePipeline::HierarchyOrdered)
                != self.ordered_gpu.is_some()
            || (self.render_mode.uses_gpu_hierarchy()
                && (self.capture_mode != RuntimeCaptureMode::Instrumented
                    || (self.presentation_mode != LodPresentationMode::Discrete
                        && self.spatial_transitions.is_none())))
            || (self.spatial_transitions.is_some()
                && (self.render_mode != LodCapturePipeline::HierarchyOrdered
                    || self.presentation_mode != LodPresentationMode::ContinuousMorph))
            || (self.capture_hierarchy_cut && !self.render_mode.uses_gpu_hierarchy())
        {
            return Err("invalid capture configuration".into());
        }
        for segment in &self.segments {
            if segment.camera_frame.is_some() {
                if self.camera_path.is_none()
                    || segment.scenario.trim().is_empty()
                    || segment.frames == 0
                {
                    return Err("camera_frame requires a camera_path and a nonempty segment".into());
                }
                continue;
            }
            let from = Vec3::from_array(segment.from);
            let to = Vec3::from_array(segment.to);
            let target = Vec3::from_array(segment.target);
            let up = Vec3::from_array(segment.up);
            let unit_up = up.try_normalize();
            if segment.scenario.trim().is_empty()
                || segment.frames == 0
                || !from.is_finite()
                || !to.is_finite()
                || !target.is_finite()
                || !up.is_finite()
                || unit_up.is_none()
                // Check the whole linear path, including an interior point
                // at the target or parallel to up; endpoint checks miss both.
                || segment_min_length_squared(from - target, to - target) < 1e-8
                || unit_up.is_some_and(|up| {
                    segment_min_length_squared((target - from).cross(up), (target - to).cross(up)) < 1e-8
                })
            {
                return Err("invalid camera segment".into());
            }
        }
        self.settings().validate()?;
        self.streaming().validate()?;
        if let Some(crop) = self.camera_crop {
            if crop
                .full_viewport
                .iter()
                .any(|&size| size == 0 || size > 8192)
                || self.camera_path.is_none()
                || self
                    .segments
                    .iter()
                    .any(|segment| segment.camera_frame.is_none())
            {
                return Err("capture crop requires bounded calibrated camera frames".into());
            }
            crate::testing::lod_scenes::LodPixelCrop {
                origin: crop.origin,
                size: self.viewport,
            }
            .validate(crop.full_viewport)?;
        }
        if let Some(transitions) = &self.spatial_transitions {
            transitions.validate()?;
        }
        if let Some(ordered) = &self.ordered_gpu {
            ordered.ordered().validate()?;
            ordered.traversal(self.max_active_gaussians).validate()?;
        }
        if let Some(point) = &self.point_gpu {
            point.point().validate()?;
            point.traversal(self.max_active_gaussians).validate()?;
            if let Some(budget) = point.view_budget(self.max_active_gaussians) {
                budget.validate()?;
            }
            if (point.target_gpu_ms.is_some() || point.target_view_gpu_ms.is_some())
                && !self.request_timestamps
            {
                return Err("automatic point budgets require request_timestamps".into());
            }
        }
        // Eight 8192² RGBA images would be too large even though each image is
        // legal. Capture allocation admission is independent of scene budgets.
        let row = u64::from(self.viewport[0] * 4).div_ceil(256) * 256;
        let bytes = (if self.capture_images && self.capture_mode == RuntimeCaptureMode::Instrumented
        {
            row * u64::from(self.viewport[1])
        } else {
            0
        } + gpu::IMAGE_OFFSET
            + u64::from(self.cut_capacity()) * hierarchy::SELECTED_RANGE_BYTES)
            * u64::from(self.readback_slots);
        if bytes > 256 * 1024 * 1024 {
            return Err("capture readback ring exceeds 256 MiB".into());
        }
        Ok(())
    }

    fn cut_capacity(&self) -> u32 {
        if !self.capture_hierarchy_cut {
            return 0;
        }
        self.ordered_gpu
            .as_ref()
            .map(|settings| settings.max_frontier_nodes)
            .or_else(|| {
                self.point_gpu
                    .as_ref()
                    .map(|settings| settings.max_frontier_nodes)
            })
            .unwrap_or(0)
    }

    fn streaming(&self) -> GaussianStreamingSettings {
        GaussianStreamingSettings {
            max_concurrent_requests: self.max_concurrent_requests,
            ..Default::default()
        }
    }

    fn settings(&self) -> GaussianLodSettings {
        let mut settings = GaussianLodSettings {
            quality: self.quality,
            presentation_mode: self.presentation_mode,
            ..Default::default()
        };
        settings.budgets.max_active_gaussians = self.max_active_gaussians;
        settings.budgets.max_resident_gaussians = self.max_resident_gaussians;
        settings.budgets.max_resident_bytes = self.max_resident_bytes;
        settings.budgets.max_resident_pages = self.max_resident_pages;
        settings.budgets.max_upload_bytes_per_frame = self.max_upload_bytes_per_frame;
        settings
    }
}

#[derive(Clone)]
pub(super) struct FrameRequest {
    frame: u64,
    path_frame: u64,
    scenario: String,
    camera: Entity,
    target: Handle<Image>,
    started: Instant,
    // The next main-world First schedule records start-to-start frame cadence.
    // This Arc remains attached to the original submission through GPU mapping.
    frame_wall_ms: Arc<Mutex<Option<f64>>>,
}

#[derive(Resource, Clone, ExtractResource)]
pub(super) struct CaptureRequest {
    current: Option<FrameRequest>,
    identity: LodCaptureIdentity,
    run_id: String,
    viewport: [u32; 2],
    slots: u32,
    pipeline: LodCapturePipeline,
    point_gpu: Option<RuntimePointGpuConfig>,
    ordered_gpu: Option<RuntimeOrderedGpuConfig>,
    capture_images: bool,
    cut_capacity: u32,
    sender: mpsc::SyncSender<CompletedCapture>,
    stats: Arc<Mutex<CaptureStats>>,
}

#[derive(Default, Serialize)]
pub(super) struct CaptureStats {
    requested: u64,
    submitted: u64,
    completed: u64,
    dropped_ring_full: u64,
    missing_drawable: u64,
    unattested_draws: u64,
    attested_submissions: u64,
    /// Completed copies of an attested command with nonzero vertex and
    /// instance counts. Encoded zero-instance commands cannot release startup.
    nonzero_drawn_submissions: u64,
    mapping_errors: Vec<String>,
    last_drawable_diagnostic: Option<serde_json::Value>,
    /// Present only when the caller supplies a run-entry clock before loading
    /// sources. An absent origin must never become a misleading startup zero.
    startup_timing: Option<CaptureStartupTiming>,
    /// First execution of the capture's Core3d system, after a RenderDevice
    /// exists. Shared with virtual-city without changing its generator identity.
    renderer_loop_timing: Option<CaptureStartupTiming>,
    /// Native request accounting is separate from strict image receipts.
    /// Virtual-city retains its existing lifecycle protocol when this is None.
    #[serde(skip)]
    outcomes: Option<outcomes::RequestOutcomes>,
}

#[derive(Serialize)]
struct CaptureStartupTiming {
    schema_version: u32,
    origin: &'static str,
    origin_unix_ms: Option<u64>,
    #[serde(skip)]
    started: Instant,
    first_nonzero_attested_frame: Option<CaptureStartupFrame>,
    first_complete_package_nonzero_attested_frame: Option<CaptureStartupFrame>,
}

#[derive(Clone, Serialize)]
struct CaptureStartupFrame {
    stamp: LodCaptureStamp,
    scenario: String,
    frame_started_seconds: f64,
    frame_started_before_origin: bool,
    readback_observed_seconds: f64,
    selected_gaussians: u64,
    candidate_gaussians: u64,
    drawn_gaussians: u64,
}

impl CaptureStartupTiming {
    fn new(started: Instant) -> Self {
        Self::with_origin(
            started,
            "run_capture_entry_before_config_source_load_identity_hashing_and_app_initialization;excludes_process_launch_before_function_entry;sampled_frame_start_and_readback_upper_bound_not_exact_first_GPU_draw",
        )
    }

    fn with_origin(started: Instant, origin: &'static str) -> Self {
        Self {
            schema_version: 1,
            origin,
            origin_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()
                .and_then(|time| time.as_millis().try_into().ok()),
            started,
            first_nonzero_attested_frame: None,
            first_complete_package_nonzero_attested_frame: None,
        }
    }

    fn observe(
        &mut self,
        counts: &LodCaptureCounts,
        scenario: &str,
        frame_started: Instant,
        readback_observed: Instant,
        complete_package_frontier: bool,
    ) {
        let Some(drawn_gaussians) = counts.drawn.filter(|count| *count > 0) else {
            return;
        };
        if counts.source != LodCountSource::GpuReadback {
            return;
        }
        let Some(observed_elapsed) = readback_observed.checked_duration_since(self.started) else {
            return;
        };
        let sample = CaptureStartupFrame {
            stamp: counts.stamp.clone(),
            scenario: scenario.to_owned(),
            frame_started_seconds: frame_started
                .saturating_duration_since(self.started)
                .as_secs_f64(),
            frame_started_before_origin: frame_started < self.started,
            readback_observed_seconds: observed_elapsed.as_secs_f64(),
            selected_gaussians: counts.selected,
            candidate_gaussians: counts.candidates,
            drawn_gaussians,
        };
        // Asynchronous mappings can finish out of order. Keep the earliest
        // observed sampled frame, with that frame's own completion timestamp.
        for destination in [
            Some(&mut self.first_nonzero_attested_frame),
            (complete_package_frontier && counts.pipeline != LodCapturePipeline::FlatSource)
                .then_some(&mut self.first_complete_package_nonzero_attested_frame),
        ]
        .into_iter()
        .flatten()
        {
            if destination
                .as_ref()
                .is_none_or(|first| sample.frame_started_seconds < first.frame_started_seconds)
            {
                *destination = Some(sample.clone());
            }
        }
    }
}

impl CaptureStats {
    fn register_request(&mut self, frame: &FrameRequest) {
        self.requested += 1;
        if let Some(outcomes) = &mut self.outcomes
            && let Err(error) = outcomes.register(frame.frame, frame.path_frame, &frame.scenario)
        {
            self.mapping_errors.push(error.to_owned());
        }
    }

    fn finish_request(&mut self, frame: u64, outcome: outcomes::Outcome) {
        if let Some(outcomes) = &mut self.outcomes
            && let Err(error) = outcomes.finish(frame, outcome)
        {
            self.mapping_errors.push(error.to_owned());
        }
    }

    fn fail_request(&mut self, frame: u64, error: String) {
        self.finish_request(
            frame,
            outcomes::Outcome::CaptureFailure {
                error: error.clone(),
            },
        );
        self.mapping_errors.push(error);
    }

    fn observe_renderer_loop(&mut self, now: Instant) {
        self.renderer_loop_timing.get_or_insert_with(|| CaptureStartupTiming::with_origin(
            now,
            "first_capture_Core3d_system_execution_with_RenderDevice;renderer_initialized_proxy_not_exact_device_initialization;excludes_prior_source_load_and_app_startup;sampled_readback_completion_is_an_upper_bound",
        ));
    }

    fn observe_draw_readback(&mut self, attested: bool, vertices: u32, instances: u32) -> bool {
        let nonzero = attested && vertices > 0 && instances > 0;
        if nonzero {
            self.nonzero_drawn_submissions = self.nonzero_drawn_submissions.saturating_add(1);
        }
        nonzero
    }

    fn has_observed_drawable(&self) -> bool {
        self.nonzero_drawn_submissions > 0
    }
}

pub(super) struct CompletedCapture {
    record: LodFrameCapture,
    rgba: Vec<u8>,
    frame_wall_ms: Arc<Mutex<Option<f64>>>,
    evidence: serde_json::Value,
}

#[derive(Resource)]
struct CaptureSession {
    config: RuntimeCaptureConfig,
    camera_path: Option<crate::camera::path::GaussianCameraPath>,
    output: Option<BufWriter<File>>,
    evidence: Option<BufWriter<File>>,
    cadence: Vec<cadence::CadenceSample>,
    last_cadence: Option<(Instant, cadence::CadenceSample)>,
    first_loop_frame: Option<Instant>,
    last_cadence_startup_sample: Option<Instant>,
    output_directory: PathBuf,
    receiver: Mutex<mpsc::Receiver<CompletedCapture>>,
    pending_write: Vec<CompletedCapture>,
    camera: Option<Entity>,
    target: Option<Handle<Image>>,
    next_frame: u64,
    path_frame: u64,
    last_frame_started: Option<(Instant, Arc<Mutex<Option<f64>>>)>,
    last_startup_capture: Option<Instant>,
    started: Instant,
    manifest: Option<crate::GaussianLodManifest>,
    flat_cloud: Option<source::LoadedSource>,
    source_metadata: Option<SourceMetadata>,
    package_root: PathBuf,
    finished_path: bool,
    drain_frames: u32,
}

/// Launch a headless native run. This is an explicitly expensive entrypoint:
/// it hashes supplied assets, opens a GPU, streams the package and saves images.
pub fn run_capture(config_path: &Path) -> CaptureResult<()> {
    let startup_timing = CaptureStartupTiming::new(Instant::now());
    let config_bytes = fs::read(config_path)?;
    let mut config: RuntimeCaptureConfig = serde_json::from_slice(&config_bytes)?;
    config.validate()?;
    if config.capture_mode == RuntimeCaptureMode::CadenceOnly {
        config.capture_images = false;
        config.request_timestamps = false;
    }
    let base = config_path.parent().unwrap_or(Path::new("."));
    config.output = base.join(&config.output);
    if let Some(path) = &mut config.manifest {
        *path = base.join(&*path);
    }
    if let Some(path) = &mut config.source {
        *path = base.join(&*path);
    }
    if let Some(path) = &mut config.source_metadata {
        *path = base.join(&*path);
    }
    if let Some(path) = &mut config.camera_path {
        *path = base.join(&*path);
    }
    let camera_source = config
        .camera_path
        .as_ref()
        .map(|path| read_bounded_file(path, 64 * 1024 * 1024))
        .transpose()?;
    let camera_path = camera_source
        .as_ref()
        .map(|bytes| crate::camera::path::GaussianCameraPath::from_json(bytes))
        .transpose()?;
    for index in config
        .segments
        .iter()
        .filter_map(|segment| segment.camera_frame)
    {
        let frame = camera_path
            .as_ref()
            .and_then(|path| path.frames().get(index))
            .ok_or("camera_frame exceeds the supplied camera path")?;
        frame.projection(config.near, 100_000.0)?;
    }
    // Never overwrite an earlier experiment or mix its image identities.
    if let Some(parent) = config.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&config.output)?;
    let config_canonical = serde_json::to_vec_pretty(&config)?;
    fs::write(config.output.join("settings.json"), &config_canonical)?;
    let mut camera_bytes = serde_json::to_vec(&config.segments)?;
    fs::write(config.output.join("camera_path.json"), &camera_bytes)?;
    if let Some(bytes) = camera_source {
        camera_bytes.extend_from_slice(&bytes);
        fs::write(config.output.join("camera_path_source.json"), bytes)?;
    }
    let renderer_revision = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable-see-executable-sha256".to_owned());
    let source_metadata: Option<SourceMetadata> = config
        .source_metadata
        .as_ref()
        .map(|path| -> CaptureResult<_> { Ok(serde_json::from_slice(&fs::read(path)?)?) })
        .transpose()?;
    let mut flat_cloud =
        if config.render_mode == LodCapturePipeline::FlatSource && !config.synthetic {
            Some(source::load_source(
                config.source.as_ref().unwrap(),
                config.max_source_gaussians,
            )?)
        } else {
            None
        };
    let (manifest_bytes, package_root, source_sha256) = if config.synthetic {
        if config.render_mode == LodCapturePipeline::FlatSource {
            flat_cloud = Some(source::LoadedSource {
                cloud: LodTestScene::checkerboard_facade(32, 32).cloud(),
                transform: Transform::IDENTITY,
                color_space: Default::default(),
            });
        }
        create_synthetic_package(&config.output)?
    } else {
        let source_sha256 = hash_file(config.source.as_ref().unwrap())?;
        if let Some(metadata) = &source_metadata {
            if metadata.original_sha256 != source_sha256 {
                return Err("source metadata identity does not match the original source".into());
            }
            fs::write(
                config.output.join("source_metadata.json"),
                serde_json::to_vec_pretty(metadata)?,
            )?;
        }
        if let Some(manifest) = &config.manifest {
            (
                read_bounded_file(manifest, config.max_manifest_bytes)?,
                manifest.parent().unwrap_or(Path::new(".")).to_path_buf(),
                source_sha256,
            )
        } else {
            let loaded = flat_cloud.as_ref().ok_or("flat source missing")?;
            let descriptor = serde_json::to_vec_pretty(&serde_json::json!({
                "pipeline": "flat_source", "source_sha256": source_sha256,
                "gaussian_count": loaded.cloud.len(), "world_from_local": loaded.transform.to_matrix().to_cols_array(),
                "color_space": loaded.color_space,
            }))?;
            fs::write(config.output.join("flat_source.json"), &descriptor)?;
            (descriptor, config.output.clone(), source_sha256)
        }
    };
    let manifest = if config.render_mode != LodCapturePipeline::FlatSource {
        Some(decode_manifest(
            &manifest_bytes,
            LodCodecLimits {
                max_manifest_bytes: config.max_manifest_bytes,
                ..Default::default()
            },
        )?)
    } else {
        None
    };
    if config.capture_hierarchy_cut {
        let hierarchy = manifest.as_ref().ok_or("cut capture requires a manifest")?;
        if hierarchy.nodes.len() > 262_144 || hierarchy.roots.len() > 65_535 {
            return Err("cut capture metadata exceeds its node/root admission".into());
        }
        #[derive(Serialize)]
        struct HierarchyIndex<'a> {
            kind: &'static str,
            manifest_sha256: String,
            index_layout: &'static str,
            roots: &'a [crate::gaussian::formats::planar_3d_chunked::LodNodeId],
            nodes: &'a [crate::gaussian::formats::planar_3d_lod::GaussianLodNode],
        }
        // Stream metadata already admitted by the manifest decoder; do not
        // build a second serde Value tree or retain a per-frame node map.
        let mut writer = BufWriter::new(File::create(config.output.join("hierarchy_index.json"))?);
        serde_json::to_writer(
            &mut writer,
            &HierarchyIndex {
                kind: "capture_hierarchy_index_v1",
                manifest_sha256: hash_bytes(&manifest_bytes),
                index_layout: "root aliases in roots order, then nodes in manifest order",
                roots: &hierarchy.roots,
                nodes: &hierarchy.nodes,
            },
        )?;
        writer.flush()?;
    }
    let renderer_sha256 = hash_file(&std::env::current_exe()?)?;
    let mut settings_identity = config_canonical.clone();
    if let Some(metadata) = &source_metadata {
        settings_identity.extend(serde_json::to_vec(metadata)?);
    }
    let settings_sha256 = hash_bytes(&settings_identity);
    let run_id = hash_bytes(
        format!(
            "{renderer_sha256}:{settings_sha256}:{}",
            hash_bytes(&manifest_bytes)
        )
        .as_bytes(),
    );
    let identity = LodCaptureIdentity {
        manifest_sha256: hash_bytes(&manifest_bytes),
        source_sha256,
        builder_revision: config.builder_revision.clone(),
        renderer_revision,
        renderer_sha256,
        features: capture_features(),
        backend: "vulkan".to_owned(),
        adapter: None,
        driver: None,
        camera_path_sha256: hash_bytes(&camera_bytes),
        settings_sha256,
        instrumentation: if config.capture_mode == RuntimeCaptureMode::CadenceOnly {
            "cadence_only;no_GPU_copy_timestamp_draw_probe_or_image;CPU_readiness_not_draw_attestation".to_owned()
        } else if config.point_gpu.is_some() {
            format!(
                "bounded_same_submission_point_image_receipt+traversal_feedback+optional_GPU_timestamps;images={};projected_Gaussians_separate_from_point_attempts;memory_partial;frame_wall_includes_capture_encoding",
                config.capture_images
            )
        } else if config.ordered_gpu.is_some() {
            format!(
                "bounded_same_submission_ordered_draw_receipt+traversal_feedback+optional_GPU_timestamps;images={};memory_partial;frame_wall_includes_capture_encoding",
                config.capture_images
            )
        } else {
            format!(
                "bounded_async_indirect+draw_command_attestation+optional_gpu_timestamps_v1;images={};memory_partial;frame_wall_includes_main_thread_capture_encoding",
                config.capture_images
            )
        },
    };
    let (sender, receiver) = mpsc::sync_channel(config.readback_slots as usize);
    let request = CaptureRequest {
        current: None,
        identity,
        run_id,
        viewport: config.viewport,
        slots: config.readback_slots,
        pipeline: config.render_mode,
        point_gpu: config.point_gpu.clone(),
        ordered_gpu: config.ordered_gpu.clone(),
        capture_images: config.capture_images,
        cut_capacity: config.cut_capacity(),
        sender,
        stats: Arc::new(Mutex::new(CaptureStats {
            startup_timing: Some(startup_timing),
            // <=100k logical frames plus at most four startup requests/second.
            // Extra headroom covers the initial request and deadline boundary.
            outcomes: (config.capture_mode == RuntimeCaptureMode::Instrumented).then(|| {
                outcomes::RequestOutcomes::new(
                    config
                        .segments
                        .iter()
                        .map(|segment| segment.frames as usize)
                        .sum::<usize>()
                        + config.timeout_seconds as usize * 4
                        + 8,
                )
            }),
            ..Default::default()
        })),
    };
    let session = CaptureSession {
        camera_path,
        output: (config.capture_mode == RuntimeCaptureMode::Instrumented)
            .then(|| File::create(config.output.join("capture.jsonl")).map(BufWriter::new))
            .transpose()?,
        evidence: (config.capture_mode == RuntimeCaptureMode::Instrumented)
            .then(|| {
                File::create(config.output.join("submission_evidence.jsonl")).map(BufWriter::new)
            })
            .transpose()?,
        cadence: Vec::new(),
        last_cadence: None,
        first_loop_frame: None,
        last_cadence_startup_sample: None,
        output_directory: config.output.clone(),
        receiver: Mutex::new(receiver),
        pending_write: Vec::new(),
        camera: None,
        target: None,
        next_frame: 0,
        path_frame: 0,
        last_frame_started: None,
        last_startup_capture: None,
        started: Instant::now(),
        manifest,
        flat_cloud,
        source_metadata,
        package_root,
        finished_path: false,
        drain_frames: 0,
        config: config.clone(),
    };
    let mut wgpu = WgpuSettings::default();
    if config.request_timestamps {
        wgpu.features |=
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    }
    let mut app = App::new();
    app.insert_resource(session)
        .insert_resource(request)
        .insert_resource(crate::stream::memory::LodMemoryLedger::new(
            crate::stream::memory::LodMemoryLimits {
                max_cpu_bytes: config.max_cpu_bytes,
                max_gpu_bytes: config.max_gpu_bytes,
            },
        ))
        .insert_resource(ClearColor(Color::linear_rgba(0.0, 0.0, 0.0, 0.0)))
        .insert_resource(GaussianLodPackageConfig {
            max_atlas_gaussians: config.max_resident_gaussians.try_into()?,
            max_atlas_bytes: config.max_resident_bytes,
            ..Default::default()
        })
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    unapproved_path_mode: UnapprovedPathMode::Allow,
                    ..Default::default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(wgpu)),
                    ..Default::default()
                })
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..Default::default()
                })
                .disable::<WinitPlugin>()
                .disable::<PipelinedRenderingPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            config.frame_period_ms / 1000.0,
        )))
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, setup_capture)
        .add_systems(First, advance_capture)
        .add_systems(Last, receive_captures);
    if config.capture_mode == RuntimeCaptureMode::Instrumented {
        app.add_plugins(ExtractResourcePlugin::<CaptureRequest>::default());
        gpu::install(app.sub_app_mut(RenderApp));
    }
    let exit = app.run();
    if exit.is_success() {
        Ok(())
    } else {
        Err("capture failed; inspect capture_status.json".into())
    }
}

fn setup_capture(
    mut commands: Commands,
    mut session: ResMut<CaptureSession>,
    mut request: ResMut<CaptureRequest>,
    (adapter, device): (
        Res<bevy::render::renderer::RenderAdapterInfo>,
        Res<bevy::render::renderer::RenderDevice>,
    ),
    (mut assets, mut flat_assets): (
        ResMut<Assets<GaussianLodAsset>>,
        ResMut<Assets<PlanarGaussian3d>>,
    ),
    mut images: ResMut<Assets<Image>>,
    mut exit: MessageWriter<AppExit>,
) {
    request.identity.backend = match adapter.backend {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gl",
        _ => "unsupported",
    }
    .to_owned();
    request.identity.adapter = Some(adapter.name.clone());
    request.identity.driver = Some(format!("{} {}", adapter.driver, adapter.driver_info));
    if let Some(policy) = &session.config.ordered_gpu {
        let capacity = session
            .config
            .max_active_gaussians
            .min(u64::from(policy.max_projected_gaussians)) as u32;
        let limits = device.limits();
        let spatial = session.config.spatial_transitions.is_some();
        let admission = crate::render::ordered::preflight_global_order(
            &policy.ordered(),
            capacity,
            1,
            spatial,
            &limits,
        );
        let report = serde_json::json!({
            "scope":"single-cloud ordered renderer allocation before package streaming; excludes atlas, traversal and untracked device allocations",
            "adapter":adapter.name, "capacity":capacity, "spatial_requested":spatial,
            "projected_record_bytes":if spatial {96} else {64},
            "device_limits":{
                "max_storage_buffer_binding_size":limits.max_storage_buffer_binding_size,
                "max_buffer_size":limits.max_buffer_size,
                "max_compute_workgroups_per_dimension":limits.max_compute_workgroups_per_dimension,
            },
            "admitted_gpu_bytes":admission.as_ref().ok(),
            "error":admission.as_ref().err(),
        });
        let saved = fs::write(
            session.output_directory.join("device_preflight.json"),
            serde_json::to_vec_pretty(&report).expect("finite device admission report"),
        );
        let failure = admission
            .err()
            .or_else(|| saved.err().map(|error| error.to_string()));
        if let Some(reason) = failure {
            eprintln!("capture device admission failed: {reason}");
            let status = serde_json::json!({
                "release_qualified":false, "failed_before_streaming":true, "reason":reason,
                "device_preflight":"device_preflight.json",
            });
            if let Err(error) = fs::write(
                session.output_directory.join("capture_status.json"),
                serde_json::to_vec_pretty(&status).expect("finite failure report"),
            ) {
                eprintln!("capture failure report could not be saved: {error}");
            }
            exit.write(AppExit::error());
            return;
        }
    }
    let mut cloud_settings = CloudSettings {
        sort_mode: SortMode::Radix,
        opacity_adaptive_radius: false,
        ..Default::default()
    };
    if let Some(source) = session.flat_cloud.take() {
        cloud_settings.color_space = source.color_space;
        commands.spawn((
            crate::PlanarGaussian3dHandle(flat_assets.add(source.cloud)),
            cloud_settings,
            source.transform,
            Visibility::Visible,
        ));
    } else {
        let transform = if let Some(metadata) = &session.source_metadata {
            cloud_settings.color_space = metadata.color_space;
            Transform::from_matrix(Mat4::from_cols_array(&metadata.world_from_local))
        } else {
            Transform::IDENTITY
        };
        let manifest = assets.add(GaussianLodAsset::new(session.manifest.take().unwrap()).unwrap());
        let mut entity = commands.spawn((
            GaussianLodHandle(manifest),
            GaussianLodPackageSource::native_directory(
                session.package_root.to_string_lossy().into_owned(),
            ),
            session.config.settings(),
            session.config.streaming(),
            cloud_settings,
            transform,
            Visibility::Visible,
        ));
        if session.config.render_mode.uses_gpu_hierarchy() {
            entity.insert(crate::stream::package::GaussianGpuLodPackage);
        }
    }
    let [width, height] = session.config.viewport;
    let mut image = Image::new_target_texture(width, height, TextureFormat::Rgba8UnormSrgb, None);
    if session.config.capture_images {
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    }
    let target = images.add(image);
    let segment = &session.config.segments[0];
    let (transform, projection) = capture_camera_pose(&session, segment, 0);
    let mut camera_entity = commands.spawn((
        Camera3d::default(),
        Camera {
            sub_camera_view: session
                .config
                .camera_crop
                .map(|crop| crop.sub_view(session.config.viewport)),
            ..Default::default()
        },
        projection,
        RenderTarget::Image(target.clone().into()),
        transform,
        GaussianCamera::default(),
        Tonemapping::None,
        Msaa::Off,
    ));
    if let Some(point) = &session.config.point_gpu {
        camera_entity.insert((
            point.point(),
            point.traversal(session.config.max_active_gaussians),
        ));
        if let Some(budget) = point.view_budget(session.config.max_active_gaussians) {
            camera_entity.insert(budget);
        }
    }
    if let Some(ordered) = &session.config.ordered_gpu {
        camera_entity.insert((
            ordered.ordered(),
            ordered.traversal(session.config.max_active_gaussians),
        ));
    }
    if let Some(transitions) = &session.config.spatial_transitions {
        camera_entity.insert(transitions.clone());
    }
    let camera = camera_entity.id();
    session.camera = Some(camera);
    session.target = Some(target);
}

fn capture_camera_transform(segment: &CaptureCameraSegment, local_frame: u64) -> Transform {
    let from = Vec3::from_array(segment.from);
    let position = if segment.from == segment.to {
        // Weighted lerp of equal floats can still round differently as t
        // changes. A held pose must preserve Transform change detection and
        // exact view/cache identity across every frame of the segment.
        from
    } else {
        let t = local_frame as f32 / segment.frames.saturating_sub(1).max(1) as f32;
        from.lerp(Vec3::from_array(segment.to), t)
    };
    Transform::from_translation(position).looking_at(
        Vec3::from_array(segment.target),
        Vec3::from_array(segment.up),
    )
}

fn capture_camera_pose(
    session: &CaptureSession,
    segment: &CaptureCameraSegment,
    local: u64,
) -> (Transform, Projection) {
    if let Some(index) = segment.camera_frame {
        let frame = &session
            .camera_path
            .as_ref()
            .expect("validated camera path")
            .frames()[index];
        return (
            frame.transform(),
            frame
                .projection(session.config.near, 100_000.0)
                .expect("validated camera projection"),
        );
    }
    (
        capture_camera_transform(segment, local),
        Projection::Perspective(PerspectiveProjection {
            fov: session.config.vertical_fov_radians,
            near: session.config.near,
            far: 100_000.0,
            ..Default::default()
        }),
    )
}

fn advance_capture(
    mut session: ResMut<CaptureSession>,
    mut request: ResMut<CaptureRequest>,
    mut cameras: Query<(&mut Transform, &mut Projection), With<GaussianCamera>>,
    package_statuses: Query<&crate::stream::package::GaussianLodPackageStatus>,
    flat_sources: Query<&crate::PlanarGaussian3dHandle, Without<GaussianLodHandle>>,
    flat_assets: Res<Assets<PlanarGaussian3d>>,
) {
    let now = Instant::now();
    if let Some((started, mut sample)) = session.last_cadence.take() {
        sample.frame_wall_ms = now.duration_since(started).as_secs_f64() * 1_000.0;
        session.cadence.push(sample);
    }
    if let Some((started, duration)) = session.last_frame_started.take() {
        *duration.lock().unwrap() = Some(now.duration_since(started).as_secs_f64() * 1_000.0);
    }
    request.current = None;
    let Some(camera) = session.camera else {
        return;
    };
    let first_loop_frame = *session.first_loop_frame.get_or_insert(now);
    let frame = session.next_frame;
    let mut local = session.path_frame;
    let segment = session.config.segments.iter().find(|segment| {
        if local < u64::from(segment.frames) {
            true
        } else {
            local -= u64::from(segment.frames);
            false
        }
    });
    let Some(segment) = segment else {
        session.finished_path = true;
        session.drain_frames += 1;
        return;
    };
    let (next, next_projection) = capture_camera_pose(&session, segment, local);
    let world_from_view = next.to_matrix().to_cols_array();
    if let Ok((mut transform, mut projection)) = cameras.get_mut(camera) {
        if *transform != next {
            *transform = next;
        }
        if local == 0 {
            *projection = next_projection;
        }
    }
    let scenario = segment.scenario.clone();
    let cadence_only = session.config.capture_mode == RuntimeCaptureMode::CadenceOnly;
    let ready = if cadence_only {
        if session.config.render_mode != LodCapturePipeline::FlatSource {
            package_statuses.iter().any(|status| {
                status.phase == crate::stream::package::GaussianLodPackagePhase::Active
            })
        } else {
            flat_sources
                .iter()
                .any(|handle| flat_assets.contains(handle.0.id()))
                && now.duration_since(first_loop_frame).as_secs_f64()
                    >= session.config.cadence_minimum_warmup_seconds
        }
    } else {
        request.stats.lock().unwrap().has_observed_drawable()
    };
    let at_startup_boundary =
        session.path_frame + 1 == u64::from(session.config.segments[0].frames);
    if cadence_only {
        // A finite logical path plus four samples/second during the bounded
        // startup pause caps sample storage independently of renderer cadence.
        let record = !at_startup_boundary
            || ready
            || session
                .last_cadence_startup_sample
                .is_none_or(|last| now.duration_since(last) >= Duration::from_millis(250));
        if record {
            if at_startup_boundary {
                session.last_cadence_startup_sample = Some(now);
            }
            session.last_cadence = Some((
                now,
                cadence::CadenceSample {
                    frame,
                    path_frame: session.path_frame,
                    scenario,
                    world_from_view,
                    frame_wall_ms: 0.0,
                    cpu_readiness_observed: ready,
                },
            ));
        }
    } else {
        let duration = Arc::new(Mutex::new(None));
        session.last_frame_started = Some((now, duration.clone()));
        let startup_due = ready
            || session
                .last_startup_capture
                .is_none_or(|last| now.duration_since(last) >= Duration::from_millis(250));
        if (!ready
            || (session.path_frame >= session.config.capture_start_path_frame
                && session
                    .path_frame
                    .is_multiple_of(u64::from(session.config.capture_every))))
            && startup_due
        {
            if !ready {
                session.last_startup_capture = Some(now);
            }
            request.current = Some(FrameRequest {
                frame,
                path_frame: session.path_frame,
                scenario,
                camera,
                target: session.target.clone().unwrap(),
                started: now,
                frame_wall_ms: duration,
            });
            request
                .stats
                .lock()
                .unwrap()
                .register_request(request.current.as_ref().unwrap());
        }
    }
    session.next_frame += 1;
    // Loading and shader compilation must not consume the complete camera path
    // before a drawable exists. The initial stationary segment may extend;
    // subsequent camera poses advance by their deterministic logical path frame.
    if !at_startup_boundary || ready {
        session.path_frame += 1;
    }
}

fn receive_captures(
    mut session: ResMut<CaptureSession>,
    request: Res<CaptureRequest>,
    mut exit: MessageWriter<AppExit>,
    package_statuses: Query<(
        &crate::stream::package::GaussianLodPackageStatus,
        Option<&crate::GaussianLodStatus>,
    )>,
) {
    let received: Vec<_> = if session.config.capture_mode == RuntimeCaptureMode::Instrumented {
        session.receiver.lock().unwrap().try_iter().collect()
    } else {
        Vec::new()
    };
    session.pending_write.extend(received);
    let mut waiting = Vec::new();
    for mut capture in std::mem::take(&mut session.pending_write) {
        let frame_wall_ms = *capture.frame_wall_ms.lock().unwrap();
        if frame_wall_ms.is_none() {
            waiting.push(capture);
            continue;
        }
        if let Some(timing) = &mut capture.record.timings {
            timing.frame_wall_ms = frame_wall_ms;
        }
        let result = write_capture(&mut session, &mut capture);
        if let Err(error) = result {
            let mut stats = request.stats.lock().unwrap();
            stats.finish_request(
                capture.record.stamp.frame,
                outcomes::Outcome::WriteFailure {
                    error: error.to_string(),
                },
            );
            stats.mapping_errors.push(error.to_string());
        } else {
            let mut stats = request.stats.lock().unwrap();
            stats.completed += 1;
            stats.finish_request(
                capture.record.stamp.frame,
                outcomes::Outcome::Completed {
                    image_written: capture.record.image.is_some(),
                    draw_attested: capture.record.counts.drawn.is_some(),
                },
            );
        }
    }
    session.pending_write = waiting;
    let mut stats = request.stats.lock().unwrap();
    let timed_out = session.started.elapsed().as_secs() > session.config.timeout_seconds;
    let drained =
        session.finished_path && session.drain_frames >= 3 && stats.completed >= stats.submitted;
    let failed = !stats.mapping_errors.is_empty();
    if drained || timed_out || failed {
        let cadence_only = session.config.capture_mode == RuntimeCaptureMode::CadenceOnly;
        let cadence_result = if cadence_only {
            cadence::write(
                &session.output_directory,
                &request.run_id,
                &request.identity,
                &session.cadence,
                if session.config.render_mode != LodCapturePipeline::FlatSource {
                    "main_world_package_phase_active;not_draw_attestation"
                } else {
                    "main_world_source_asset_present_and_fixed_warmup;not_GPU_readiness_or_draw_attestation"
                },
            )
        } else {
            Ok(())
        };
        if let Err(error) = &cadence_result {
            eprintln!("capture cadence write failed: {error}");
        }
        let outcomes_result = if let Some(outcomes) = &mut stats.outcomes {
            outcomes.finish_pending(timed_out);
            File::create(session.output_directory.join("request_outcomes.jsonl"))
                .and_then(|file| outcomes.write(BufWriter::new(file), &request.run_id))
        } else {
            Ok(())
        };
        if let Err(error) = &outcomes_result {
            eprintln!("capture request outcomes write failed: {error}");
        }
        let outcome_summary = stats
            .outcomes
            .as_ref()
            .map(outcomes::RequestOutcomes::summary);
        let requests_accounted = outcomes_result.is_ok()
            && outcome_summary.as_ref().is_some_and(|summary| {
                summary.requested as u64 == stats.requested
                    && summary.terminal == summary.requested
                    && summary.unresolved == 0
                    && summary.accounting_errors == 0
            });
        let status = serde_json::json!({"capture_mode": session.config.capture_mode,
            "cadence_samples": session.cadence.len(), "stats": &*stats, "timed_out": timed_out,
            "request_outcomes": outcome_summary,
            "request_accounting_complete": requests_accounted,
            "request_outcomes_file": (!cadence_only).then_some("request_outcomes.jsonl"),
            "package_status": package_statuses.iter().map(|(package, lod)| format!("package={package:?};lod={lod:?}")).collect::<Vec<_>>(),
            "release_qualified": false, "memory_audit_complete": false,
            "scenario_samples_complete": session.finished_path && stats.completed == stats.requested
                && (cadence_only || requests_accounted),
            "attested_counts_complete": stats.completed > 0 && stats.attested_submissions == stats.submitted
                && outcome_summary.as_ref().is_some_and(|summary| summary.attested_captures as u64 == stats.requested)
                && requests_accounted,
            "reason": "Runtime observations require matched reference quality and complete residency/device memory audit."});
        let _ = fs::write(
            session.output_directory.join("capture_status.json"),
            serde_json::to_vec_pretty(&status).unwrap(),
        );
        if let Some(output) = &mut session.output {
            let _ = output.flush();
        }
        if let Some(evidence) = &mut session.evidence {
            let _ = evidence.flush();
        }
        let empty = if cadence_only {
            session.cadence.is_empty()
        } else {
            stats.completed == 0
        };
        exit.write(
            if timed_out || failed || empty || cadence_result.is_err() || outcomes_result.is_err() {
                AppExit::error()
            } else {
                AppExit::Success
            },
        );
    }
}

fn write_capture(
    session: &mut CaptureSession,
    capture: &mut CompletedCapture,
) -> CaptureResult<()> {
    if session.config.capture_images {
        let name = format!("frame-{:08}.png", capture.record.stamp.frame);
        let image = Image::new(
            Extent3d {
                width: capture.record.camera.viewport[0],
                height: capture.record.camera.viewport[1],
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            std::mem::take(&mut capture.rgba),
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        image
            .try_into_dynamic()?
            .save(session.output_directory.join(&name))?;
        capture.record.image = Some(LodCaptureImage {
            stamp: capture.record.stamp.clone(),
            path: name.clone(),
            sha256: hash_file(&session.output_directory.join(name))?,
            viewport: capture.record.camera.viewport,
        });
    }
    capture
        .record
        .write_jsonl(session.output.as_mut().expect("instrumented output"))?;
    serde_json::to_writer(
        session.evidence.as_mut().expect("instrumented evidence"),
        &capture.evidence,
    )?;
    session.evidence.as_mut().unwrap().write_all(b"\n")?;
    Ok(())
}

fn create_synthetic_package(output: &Path) -> CaptureResult<(Vec<u8>, PathBuf, String)> {
    let scene = LodTestScene::checkerboard_facade(32, 32);
    let cloud: PlanarGaussian3d = scene.cloud();
    let records: Vec<_> = scene.gaussians.iter().map(|entry| entry.gaussian).collect();
    let source = bytemuck::cast_slice(&records);
    fs::write(output.join("source.gaussians.bin"), source)?;
    let mut lod = build_planar_3d_lod(
        &cloud,
        GaussianLodBuildSettings {
            branching_factor: 8,
            leaf_capacity: 64,
            support_sigma: 3.0,
        },
    )?;
    let root = output.join("package");
    fs::create_dir(&root)?;
    for (page, descriptor) in lod.pages.iter().zip(&mut lod.manifest.pages) {
        if page.id != descriptor.id {
            return Err("synthetic page order mismatch".into());
        }
        let bytes = encode_page(page)?;
        let name = format!("page-{}.bgspage", page.id.0);
        fs::write(root.join(&name), &bytes)?;
        descriptor.storage = Some(LodPageStorage {
            uri: name,
            byte_range: None,
            encoded_len: bytes.len() as u64,
        });
    }
    let manifest = encode_manifest(&lod.manifest)?;
    fs::write(root.join("manifest.bgslod"), &manifest)?;
    Ok((manifest, root, hash_bytes(source)))
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_bounded_file(path: &Path, maximum: u64) -> CaptureResult<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(maximum + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err(format!(
            "{} exceeds its {maximum}-byte capture limit",
            path.display()
        )
        .into());
    }
    Ok(bytes)
}
pub(crate) fn hash_file(path: &Path) -> CaptureResult<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut block = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut block)?;
        if read == 0 {
            break;
        }
        hasher.update(&block[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
fn capture_features() -> Vec<String> {
    [
        ("buffer_storage", cfg!(feature = "buffer_storage")),
        ("debug_gpu", cfg!(feature = "debug_gpu")),
        ("debug_tooling", cfg!(feature = "debug_tooling")),
        ("default", cfg!(feature = "default")),
        ("file_asset", cfg!(feature = "file_asset")),
        ("headless", cfg!(feature = "headless")),
        ("io_bincode2", cfg!(feature = "io_bincode2")),
        ("io_flexbuffers", cfg!(feature = "io_flexbuffers")),
        ("io_ply", cfg!(feature = "io_ply")),
        ("lod", cfg!(feature = "lod")),
        ("lod_build", cfg!(feature = "lod_build")),
        ("lod_build_sh0", cfg!(feature = "lod_build_sh0")),
        ("lod_build_sh3", cfg!(feature = "lod_build_sh3")),
        ("lod_render", cfg!(feature = "lod_render")),
        ("material_noise", cfg!(feature = "material_noise")),
        ("morph_interpolate", cfg!(feature = "morph_interpolate")),
        ("morph_particles", cfg!(feature = "morph_particles")),
        ("noise", cfg!(feature = "noise")),
        ("planar", cfg!(feature = "planar")),
        (
            "precompute_covariance_3d",
            cfg!(feature = "precompute_covariance_3d"),
        ),
        ("query_raycast", cfg!(feature = "query_raycast")),
        ("query_select", cfg!(feature = "query_select")),
        ("query_sparse", cfg!(feature = "query_sparse")),
        ("sh0", cfg!(feature = "sh0")),
        ("sh1", cfg!(feature = "sh1")),
        ("sh2", cfg!(feature = "sh2")),
        ("sh3", cfg!(feature = "sh3")),
        ("sh4", cfg!(feature = "sh4")),
        ("sort_radix", cfg!(feature = "sort_radix")),
        ("sort_rayon", cfg!(feature = "sort_rayon")),
        ("sort_std", cfg!(feature = "sort_std")),
        ("testing", cfg!(feature = "testing")),
        ("tooling", cfg!(feature = "tooling")),
        ("viewer", cfg!(feature = "viewer")),
        ("web", cfg!(feature = "web")),
        ("web_asset", cfg!(feature = "web_asset")),
        ("webgpu", cfg!(feature = "webgpu")),
    ]
    .into_iter()
    .filter(|(_, enabled)| *enabled)
    .map(|(name, _)| name.to_owned())
    .collect()
}

pub(super) fn rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmRSS:").and_then(|rest| {
            rest.split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()?
                .checked_mul(1024)
        })
    })
}

#[cfg(test)]
mod presentation_tests {
    use super::*;

    #[test]
    fn stationary_capture_segment_has_bit_identical_matrices_at_every_frame() {
        let segment = CaptureCameraSegment {
            scenario: "heldout12".into(),
            frames: 4096,
            camera_frame: None,
            from: [1.234567, -6.765432, 12.345678],
            to: [1.234567, -6.765432, 12.345678],
            target: [0.1234567, 2.345678, -3.456789],
            up: [0.0, 1.0, 0.0],
        };
        let first = capture_camera_transform(&segment, 0)
            .to_matrix()
            .to_cols_array()
            .map(f32::to_bits);
        for frame in 0..u64::from(segment.frames) {
            assert_eq!(
                capture_camera_transform(&segment, frame)
                    .to_matrix()
                    .to_cols_array()
                    .map(f32::to_bits),
                first,
                "stationary camera identity drifted at logical frame {frame}"
            );
        }
    }

    #[test]
    fn startup_waits_for_a_nonzero_attested_gpu_readback() {
        let mut stats = CaptureStats {
            attested_submissions: 100,
            ..Default::default()
        };
        assert!(
            !stats.has_observed_drawable(),
            "encoding a command is not completed draw evidence"
        );
        for (attested, vertices, instances) in [(false, 4, 64), (true, 4, 0), (true, 0, 64)] {
            stats.observe_draw_readback(attested, vertices, instances);
            assert!(!stats.has_observed_drawable());
        }
        stats.observe_draw_readback(true, 4, 64);
        assert!(stats.has_observed_drawable());
        assert_eq!(stats.nonzero_drawn_submissions, 1);
    }

    #[test]
    fn startup_timing_preserves_run_origin_and_original_frame_through_reordered_maps() {
        let origin = Instant::now();
        let mut startup = CaptureStartupTiming::new(origin);
        let mut counts = LodCaptureCounts {
            stamp: LodCaptureStamp {
                run_id: "run".into(),
                view_id: "view".into(),
                frame: 90,
                generation: 8,
            },
            pipeline: LodCapturePipeline::Hierarchy,
            source: LodCountSource::GpuReadback,
            selected: 100,
            transition_extra: 0,
            candidates: 100,
            output_capacity: 128,
            compacted: Some(80),
            drawn: Some(80),
        };
        startup.observe(
            &counts,
            "startup",
            origin + Duration::from_secs(12),
            origin + Duration::from_secs(13),
            true,
        );
        // The prior complete allocation's map arrives later. Its own stamp,
        // counts and original frame time must replace the newer observation.
        counts.stamp.frame = 80;
        counts.stamp.generation = 7;
        counts.selected = 2;
        counts.candidates = 2;
        counts.compacted = Some(1);
        counts.drawn = Some(1);
        startup.observe(
            &counts,
            "startup",
            origin + Duration::from_secs(10),
            origin + Duration::from_secs(14),
            true,
        );
        let first = startup
            .first_complete_package_nonzero_attested_frame
            .as_ref()
            .unwrap();
        assert_eq!(first.stamp, counts.stamp);
        assert_eq!(first.selected_gaussians, 2);
        assert_eq!(first.candidate_gaussians, 2);
        assert_eq!(first.drawn_gaussians, 1);
        assert_eq!(first.frame_started_seconds, 10.0);
        assert_eq!(first.readback_observed_seconds, 14.0);
        assert_eq!(
            startup.first_nonzero_attested_frame.as_ref().unwrap().stamp,
            counts.stamp
        );
        let json = serde_json::to_value(&startup).unwrap();
        assert!(
            json["origin"]
                .as_str()
                .unwrap()
                .contains("before_config_source_load")
        );
        assert!(json.get("started").is_none());
    }

    #[test]
    fn startup_timing_requires_draw_evidence_and_separates_flat_from_complete_package() {
        let origin = Instant::now();
        let mut startup = CaptureStartupTiming::new(origin);
        let mut counts = LodCaptureCounts {
            stamp: LodCaptureStamp {
                run_id: "run".into(),
                view_id: "view".into(),
                frame: 1,
                generation: 0,
            },
            pipeline: LodCapturePipeline::FlatSource,
            source: LodCountSource::GpuReadback,
            selected: 1,
            transition_extra: 0,
            candidates: 1,
            output_capacity: 1,
            compacted: None,
            drawn: None,
        };
        startup.observe(&counts, "startup", origin, origin, true);
        counts.drawn = Some(0);
        startup.observe(&counts, "startup", origin, origin, true);
        counts.drawn = Some(1);
        counts.source = LodCountSource::CpuSelection;
        startup.observe(&counts, "startup", origin, origin, true);
        assert!(startup.first_nonzero_attested_frame.is_none());
        counts.source = LodCountSource::GpuReadback;
        startup.observe(&counts, "startup", origin, origin, true);
        assert!(startup.first_nonzero_attested_frame.is_some());
        assert!(
            startup
                .first_complete_package_nonzero_attested_frame
                .is_none()
        );
        assert!(CaptureStats::default().startup_timing.is_none());
    }

    #[test]
    fn renderer_loop_origin_is_independent_and_retains_first_frame_draw() {
        let run_entry = Instant::now();
        let first_frame = run_entry + Duration::from_secs(5);
        let renderer_loop = first_frame + Duration::from_millis(10);
        let observed = renderer_loop + Duration::from_millis(200);
        let mut stats = CaptureStats::default();
        stats.observe_renderer_loop(renderer_loop);
        stats.observe_renderer_loop(renderer_loop + Duration::from_secs(1));
        assert!(
            stats.startup_timing.is_none(),
            "virtual runs have no run-entry origin"
        );
        let timing = stats.renderer_loop_timing.as_mut().unwrap();
        assert_eq!(timing.started, renderer_loop);
        let counts = LodCaptureCounts {
            stamp: LodCaptureStamp {
                run_id: "run".into(),
                view_id: "view".into(),
                frame: 0,
                generation: 1,
            },
            pipeline: LodCapturePipeline::Hierarchy,
            source: LodCountSource::GpuReadback,
            selected: 2,
            transition_extra: 0,
            candidates: 2,
            output_capacity: 2,
            compacted: Some(1),
            drawn: Some(1),
        };
        timing.observe(&counts, "startup", first_frame, observed, true);
        let first = timing
            .first_complete_package_nonzero_attested_frame
            .as_ref()
            .unwrap();
        assert_eq!(first.stamp.frame, 0);
        assert!(first.frame_started_before_origin);
        assert_eq!(first.frame_started_seconds, 0.0);
        assert_eq!(first.readback_observed_seconds, 0.2);
    }

    #[test]
    fn runtime_capture_preserves_default_and_explicit_discrete_presentation() {
        let default: RuntimeCaptureConfig = serde_json::from_str(include_str!(
            "../../tools/fixtures/lod_runtime_synthetic.json"
        ))
        .unwrap();
        assert_eq!(
            default.settings().presentation_mode,
            LodPresentationMode::ContinuousMorph
        );
        assert_eq!(
            default.max_concurrent_requests,
            default_max_concurrent_requests()
        );
        assert_eq!(default.frame_period_ms, 0.0);
        assert_eq!(default.capture_start_path_frame, 0);
        let mut discrete: RuntimeCaptureConfig = serde_json::from_str(include_str!(
            "../../tools/fixtures/lod_runtime_synthetic_discrete.json"
        ))
        .unwrap();
        discrete.validate().unwrap();
        discrete.capture_start_path_frame = discrete
            .segments
            .iter()
            .map(|segment| u64::from(segment.frames))
            .sum();
        assert!(
            discrete.validate().is_err(),
            "regular capture must include a path frame"
        );
        discrete.capture_start_path_frame -= 1;
        discrete.validate().unwrap();
        discrete.capture_start_path_frame = 0;
        for period in [0.0, 1000.0 / 30.0, 1000.0] {
            discrete.frame_period_ms = period;
            discrete.validate().unwrap();
            assert_eq!(
                serde_json::to_value(&discrete).unwrap()["frame_period_ms"],
                period
            );
        }
        for period in [-1.0, f64::NAN, f64::INFINITY, 1000.1] {
            discrete.frame_period_ms = period;
            assert!(discrete.validate().is_err());
        }
        discrete.frame_period_ms = 0.0;
        assert_eq!(
            discrete.settings().presentation_mode,
            LodPresentationMode::Discrete
        );
        assert_eq!(
            default.settings().quality_target(),
            discrete.settings().quality_target()
        );
        for concurrency in [
            1,
            64,
            crate::gaussian::lod_settings::MAX_STREAMING_CONCURRENT_REQUESTS,
        ] {
            discrete.max_concurrent_requests = concurrency;
            discrete.validate().unwrap();
            assert_eq!(discrete.streaming().max_concurrent_requests, concurrency);
            let resolved = serde_json::to_value(&discrete).unwrap();
            assert_eq!(resolved["max_concurrent_requests"], concurrency);
        }
        for concurrency in [
            0,
            crate::gaussian::lod_settings::MAX_STREAMING_CONCURRENT_REQUESTS + 1,
        ] {
            discrete.max_concurrent_requests = concurrency;
            assert!(discrete.validate().is_err());
        }
        discrete.max_concurrent_requests = default_max_concurrent_requests();
        discrete.render_mode = LodCapturePipeline::HierarchyOrdered;
        assert!(
            discrete.validate().is_err(),
            "backend requires explicit bounds"
        );
        discrete.ordered_gpu = Some(RuntimeOrderedGpuConfig {
            max_projected_gaussians: 1024,
            max_gpu_bytes: 1 << 20,
            max_traversal_gpu_bytes: 1 << 20,
            max_frontier_nodes: 64,
            max_visited_nodes: 128,
            max_page_requests: 16,
        });
        discrete.validate().unwrap();
        discrete.spatial_transitions = Some(Default::default());
        assert!(
            discrete.validate().is_err(),
            "spatial capture requires morph presentation"
        );
        discrete.presentation_mode = LodPresentationMode::ContinuousMorph;
        discrete.validate().unwrap();
        discrete
            .spatial_transitions
            .as_mut()
            .unwrap()
            .max_transition_nodes = 0;
        assert!(
            discrete.validate().is_err(),
            "spatial capacity must be bounded and nonzero"
        );
        discrete.spatial_transitions = None;
        assert!(
            discrete.validate().is_err(),
            "GPU morph capture requires explicit spatial policy"
        );
        discrete.presentation_mode = LodPresentationMode::Discrete;
        discrete.render_mode = LodCapturePipeline::HierarchyPoint;
        assert!(
            discrete.validate().is_err(),
            "backend config must match the renderer"
        );
    }

    #[test]
    fn camera_up_preserves_roll_and_rejects_singular_paths() {
        let mut config: RuntimeCaptureConfig = serde_json::from_str(include_str!(
            "../../tools/fixtures/lod_runtime_synthetic.json"
        ))
        .unwrap();
        assert_eq!(config.segments[0].up, [0.0, 1.0, 0.0]);
        config.segments.truncate(1);
        config.segments[0].from = [0.0, 0.0, 5.0];
        config.segments[0].to = [0.0, 0.0, 5.0];
        config.segments[0].target = [0.0, 0.0, 0.0];
        config.segments[0].up = [1.0, 0.0, 0.0];
        config.validate().unwrap();
        let camera = Transform::from_translation(Vec3::new(0.0, 0.0, 5.0))
            .looking_at(Vec3::ZERO, Vec3::from_array(config.segments[0].up));
        assert!((camera.rotation * Vec3::Y - Vec3::X).length() < 1e-6);
        for invalid in [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [f32::NAN, 1.0, 0.0]] {
            config.segments[0].up = invalid;
            assert!(config.validate().is_err());
        }
        config.segments[0].up = [0.0, 1.0, 0.0];
        config.segments[0].from = [-1.0, 1.0, 0.0];
        config.segments[0].to = [1.0, 1.0, 0.0];
        assert!(
            config.validate().is_err(),
            "mid-path view is parallel to up"
        );
        config.segments[0].from = [-1.0, 0.0, 0.0];
        config.segments[0].to = [1.0, 0.0, 0.0];
        assert!(config.validate().is_err(), "mid-path camera reaches target");
    }

    #[test]
    fn physical_capture_crop_requires_calibration_and_preserves_pixel_rays() {
        let mut config: RuntimeCaptureConfig = serde_json::from_str(include_str!(
            "../../tools/fixtures/lod_runtime_synthetic_discrete.json"
        ))
        .unwrap();
        config.viewport = [480, 319];
        config.camera_crop = Some(RuntimeCameraCrop {
            full_viewport: [960, 638],
            origin: [480, 319],
        });
        assert!(
            config.validate().is_err(),
            "uncalibrated segments cannot be cropped"
        );
        config.camera_path = Some("cameras.json".into());
        for segment in &mut config.segments {
            segment.camera_frame = Some(0);
        }
        config.validate().unwrap();
        let sub = config.camera_crop.unwrap().sub_view(config.viewport);
        let path = crate::camera::path::GaussianCameraPath::from_json(
            br#"[
            {"id":0,"img_name":"crop","width":960,"height":638,"fx":900,"fy":880,
             "position":[0,0,0],"rotation":[[1,0,0],[0,1,0],[0,0,1]]}
        ]"#,
        )
        .unwrap();
        let projection = path.frames()[0].projection(0.1, 1000.0).unwrap();
        let pixel = |matrix: Mat4, size: Vec2, point: Vec4| {
            let clip = matrix * point;
            (clip.truncate().truncate() / clip.w * Vec2::new(0.5, -0.5) + Vec2::splat(0.5)) * size
        };
        for point in [
            Vec4::new(0.2, -0.4, -4.0, 1.0),
            Vec4::new(1.3, -0.7, -7.0, 1.0),
        ] {
            let full = pixel(
                projection.get_clip_from_view(),
                sub.full_size.as_vec2(),
                point,
            );
            let tile = pixel(
                projection.get_clip_from_view_for_sub(&sub),
                sub.size.as_vec2(),
                point,
            );
            assert!((full - sub.offset).abs_diff_eq(tile, 1e-4));
        }
        config.camera_crop.as_mut().unwrap().origin[0] += 1;
        assert!(config.validate().is_err(), "tile must fit the full image");
    }
}
