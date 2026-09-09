//! Opt-in browser lifecycle qualification of the real package renderer.
//!
//! The page supplies a hashed artifact/configuration before starting this app.
//! Counts come from asynchronous copies of the buffer actually drawn. This
//! runner establishes execution evidence, not image quality or release approval.

mod gpu;

pub use gpu::BrowserLodDrawProbe;

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use bevy::{
    asset::{AssetMetaCheck, UnapprovedPathMode, io::web::WebAssetPlugin},
    core_pipeline::tonemapping::Tonemapping,
    platform::time::Instant,
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        settings::{RenderCreation, WgpuSettings},
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use wasm_bindgen::prelude::*;

use crate::{
    CloudSettings, GaussianCamera, GaussianLodHandle, GaussianLodPackageConfig,
    GaussianLodPackageSource, GaussianLodSettings, GaussianSplattingPlugin,
    GaussianStreamingSettings, LodPresentationMode,
    render::{
        point::GaussianPointSplattingSettings,
        traversal::{GpuLodHierarchy, GpuLodTraversalSettings},
    },
    sort::SortMode,
    stream::{
        memory::{LodMemoryLedger, LodMemoryLimits},
        package::{
            GaussianGpuLodPackage, GaussianLodPackagePhase, GaussianLodPackageStatus,
            GaussianLodPackageTestingSnapshot,
        },
    },
};

#[wasm_bindgen(inline_js = r#"
export function bgs_browser_capture_config() {
    if (!globalThis.__bgsBrowserQualificationConfig) throw new Error("qualification config missing");
    return JSON.stringify(globalThis.__bgsBrowserQualificationConfig);
}
export function bgs_browser_capture_emit(record) {
    globalThis.__bgsBrowserQualificationEmit(JSON.parse(record));
}
"#)]
extern "C" {
    #[wasm_bindgen(catch)]
    fn bgs_browser_capture_config() -> Result<String, JsValue>;
    fn bgs_browser_capture_emit(record: &str);
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BrowserConfig {
    manifest_url: String,
    manifest_sha256: String,
    wasm_sha256: String,
    renderer_revision: String,
    viewport: [u32; 2],
    from: [f32; 3],
    to: [f32; 3],
    target: [f32; 3],
    quality: f32,
    max_active_gaussians: u64,
    max_atlas_gaussians: u32,
    max_atlas_bytes: u64,
    max_cpu_bytes: u64,
    max_gpu_bytes: u64,
    max_encoded_page_bytes: u64,
    max_concurrent_requests: u32,
    preparation_records_per_frame: u32,
    sample_every_frames: u32,
    phase_frames: u32,
    timeout_seconds: u32,
    /// Explicit GPS + GPU hierarchy opt-in; absent preserves the quad protocol.
    #[serde(default)]
    point_gpu: Option<BrowserPointGpuConfig>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BrowserPointGpuConfig {
    samples_per_pixel: u32,
    max_projected_gaussians: u32,
    max_points_per_frame: u32,
    max_gpu_bytes: u64,
    max_traversal_gpu_bytes: u64,
    max_frontier_nodes: u32,
    max_visited_nodes: u32,
    max_page_requests: u32,
}

impl BrowserPointGpuConfig {
    fn point(&self) -> GaussianPointSplattingSettings {
        GaussianPointSplattingSettings {
            samples_per_pixel: self.samples_per_pixel,
            max_projected_gaussians: self.max_projected_gaussians,
            max_points_per_frame: self.max_points_per_frame,
            max_gpu_bytes: self.max_gpu_bytes,
            temporal_sampling: false,
            ..default()
        }
    }

    fn traversal(&self, max_active: u64) -> GpuLodTraversalSettings {
        GpuLodTraversalSettings {
            max_selected_gaussians: max_active.min(u64::from(self.max_projected_gaussians)) as u32,
            max_frontier_nodes: self.max_frontier_nodes,
            max_visited_nodes: self.max_visited_nodes,
            max_page_requests: self.max_page_requests,
            max_gpu_bytes: self.max_traversal_gpu_bytes,
        }
    }
}

impl BrowserConfig {
    fn validate(&self) -> Result<(), &'static str> {
        let hash =
            |value: &str| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit());
        if !hash(&self.manifest_sha256)
            || !hash(&self.wasm_sha256)
            || self.renderer_revision.trim().is_empty()
            || self.renderer_revision.starts_with("REPLACE_")
            || !(self.manifest_url.starts_with("http://")
                || self.manifest_url.starts_with("https://"))
            || self.viewport.iter().any(|&size| size == 0 || size > 4096)
            || !self.quality.is_finite()
            || !(0.0..=1.0).contains(&self.quality)
            || self.sample_every_frames == 0
            || self.sample_every_frames > 120
            || self.phase_frames < self.sample_every_frames
            || self.phase_frames > 3600
            || self.timeout_seconds == 0
            || self.timeout_seconds > 1800
            || self.max_cpu_bytes == 0
            || self.max_gpu_bytes == 0
            || self.max_atlas_gaussians == 0
            || self.max_active_gaussians == 0
            || self.max_atlas_bytes == 0
            || self.preparation_records_per_frame == 0
        {
            return Err("invalid bounded browser qualification configuration");
        }
        let target = Vec3::from_array(self.target);
        let from = Vec3::from_array(self.from);
        let to = Vec3::from_array(self.to);
        if !target.is_finite()
            || !from.is_finite()
            || !to.is_finite()
            || from.distance_squared(target) < 1e-8
            || to.distance_squared(target) < 1e-8
            || from.distance_squared(to) < 1e-8
        {
            return Err("camera path must move and avoid the look-at target");
        }
        if let Some(point) = &self.point_gpu {
            point
                .point()
                .validate()
                .map_err(|_| "invalid point renderer limits")?;
            point
                .traversal(self.max_active_gaussians)
                .validate()
                .map_err(|_| "invalid GPU traversal limits")?;
        }
        Ok(())
    }

    fn lod(&self) -> GaussianLodSettings {
        let mut settings = GaussianLodSettings {
            quality: self.quality,
            presentation_mode: if self.point_gpu.is_some() {
                LodPresentationMode::Discrete
            } else {
                GaussianLodSettings::default().presentation_mode
            },
            ..default()
        };
        settings.budgets.max_active_gaussians = self.max_active_gaussians;
        settings.budgets.max_resident_gaussians = u64::from(self.max_atlas_gaussians);
        settings.budgets.max_resident_bytes = self.max_atlas_bytes;
        settings.budgets.max_resident_pages = self.max_atlas_gaussians.min(4096);
        // A single upload cannot exceed this fixture's complete atlas budget.
        // Bound its staging reservation too, instead of charging the desktop
        // upload default against a deliberately small browser CPU ledger.
        settings.budgets.max_upload_bytes_per_frame = settings
            .budgets
            .max_upload_bytes_per_frame
            .min(self.max_atlas_bytes);
        settings
    }

    fn streaming(&self) -> GaussianStreamingSettings {
        GaussianStreamingSettings {
            max_encoded_page_bytes: self.max_encoded_page_bytes,
            max_concurrent_requests: self.max_concurrent_requests,
            persistent_cache: false,
            ..default()
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    ColdLoad,
    Stationary,
    Move,
    Unload,
    Reload,
    Complete,
}
impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::ColdLoad => "cold_load",
            Self::Stationary => "stationary",
            Self::Move => "move",
            Self::Unload => "unload",
            Self::Reload => "reload",
            Self::Complete => "complete",
        }
    }
}

#[derive(Default)]
struct SharedEvidence {
    records: VecDeque<Value>,
    attested_phases: std::collections::BTreeSet<String>,
    mapping_errors: u32,
    dropped_readbacks: u32,
    invalid_counts: u32,
}

#[derive(Clone)]
struct BrowserFrame {
    frame: u64,
    phase: &'static str,
    camera: Entity,
    point_gpu: Option<BrowserPointGpuConfig>,
}

#[derive(Resource, Clone, ExtractResource)]
struct BrowserCaptureRequest {
    frame: Option<BrowserFrame>,
    evidence: Arc<Mutex<SharedEvidence>>,
}

#[derive(Resource)]
struct BrowserSession {
    config: BrowserConfig,
    started: Instant,
    last_frame: Instant,
    frame: u64,
    phase: Phase,
    phase_started: u64,
    cloud: Option<Entity>,
    camera: Option<Entity>,
    unload_complete: bool,
    terminal: bool,
}

fn emit(record: Value) {
    bgs_browser_capture_emit(&record.to_string());
}

fn memory_json(ledger: &LodMemoryLedger) -> Value {
    let snapshot = ledger.snapshot();
    json!({"scope":"owned_capacity_reservations_not_RSS_or_driver_memory",
        "cpu_bytes":snapshot.cpu_bytes, "gpu_bytes":snapshot.gpu_bytes,
        "allocations":snapshot.allocations,
        "max_cpu_bytes":snapshot.limits.max_cpu_bytes,
        "max_gpu_bytes":snapshot.limits.max_gpu_bytes,
        "categories":snapshot.categories.iter().map(|entry| json!({
            "category":format!("{:?}",entry.category), "bytes":entry.bytes,
            "allocations":entry.allocations})).collect::<Vec<_>>()})
}

/// Starts only from the dedicated qualification page/binary.
pub fn run() {
    let config: BrowserConfig = serde_json::from_str(
        &bgs_browser_capture_config().expect("qualification page must supply configuration"),
    )
    .expect("valid configuration JSON");
    config.validate().expect("valid qualification limits");
    config.lod().validate().expect("valid LoD settings");
    config
        .streaming()
        .validate()
        .expect("valid streaming settings");
    emit(
        json!({"kind":"session", "schema":"bgs-browser-lod-qualification-v2",
        "manifest_url":config.manifest_url, "manifest_sha256":config.manifest_sha256,
        "wasm_sha256":config.wasm_sha256, "renderer_revision":config.renderer_revision,
        "feature_profile":format!("planar,lod_render,sh{},io_flexbuffers,web_asset,webgpu,testing",
            if cfg!(feature="sh3") {3} else if cfg!(feature="sh2") {2} else if cfg!(feature="sh1") {1} else if cfg!(feature="sh4") {4} else {0}),
        "viewport":config.viewport, "quality":config.quality,
        "camera_path":{"from":config.from,"to":config.to,"target":config.target},
        "sample_every_frames":config.sample_every_frames, "phase_frames":config.phase_frames,
        "renderer":if config.point_gpu.is_some() {"point_gpu_lod"} else {"quad"},
        "point_gpu":config.point_gpu,
        "readback_ring_slots":3, "readback_ring_bytes":3 * gpu::READBACK_BYTES,
        "readback_scope":"instrumentation_excluded_from_owned_scene_ledger",
        "release_qualified":false}),
    );
    let mut app = App::new();
    app.insert_resource(LodMemoryLedger::new(LodMemoryLimits {
        max_cpu_bytes: config.max_cpu_bytes,
        max_gpu_bytes: config.max_gpu_bytes,
    }));
    app.insert_resource(GaussianLodPackageConfig {
        max_atlas_gaussians: config.max_atlas_gaussians,
        max_atlas_bytes: config.max_atlas_bytes,
        preparation_records_per_frame: config.preparation_records_per_frame,
        streaming: config.streaming(),
        ..default()
    });
    app.add_plugins(
        DefaultPlugins
            .set(AssetPlugin {
                meta_check: AssetMetaCheck::Never,
                unapproved_path_mode: UnapprovedPathMode::Allow,
                ..default()
            })
            .set(WebAssetPlugin {
                silence_startup_warning: true,
            })
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                    backends: Some(wgpu::Backends::BROWSER_WEBGPU),
                    force_fallback_adapter: false,
                    ..default()
                })),
                ..default()
            })
            .set(WindowPlugin {
                primary_window: Some(Window {
                    canvas: Some("#bevy".to_owned()),
                    resize_constraints: bevy::window::WindowResizeConstraints {
                        min_width: 1.0,
                        min_height: 1.0,
                        ..default()
                    },
                    resolution: bevy::window::WindowResolution::new(
                        config.viewport[0],
                        config.viewport[1],
                    ),
                    title: "LoD browser qualification".to_owned(),
                    ..default()
                }),
                ..default()
            }),
    );
    app.add_plugins(GaussianSplattingPlugin);
    let evidence = Arc::new(Mutex::new(SharedEvidence::default()));
    app.insert_resource(BrowserCaptureRequest {
        frame: None,
        evidence,
    });
    app.add_plugins(ExtractResourcePlugin::<BrowserCaptureRequest>::default());
    app.insert_resource(BrowserSession {
        config,
        started: Instant::now(),
        last_frame: Instant::now(),
        frame: 0,
        phase: Phase::ColdLoad,
        phase_started: 0,
        cloud: None,
        camera: None,
        unload_complete: false,
        terminal: false,
    });
    app.add_systems(Startup, setup);
    app.add_systems(Update, drive_lifecycle);
    app.add_systems(Last, observe_main);
    gpu::install(app.sub_app_mut(RenderApp));
    app.run();
}

fn spawn_package(commands: &mut Commands, assets: &AssetServer, config: &BrowserConfig) -> Entity {
    let mut entity = commands.spawn((
        GaussianLodHandle(assets.load(config.manifest_url.clone())),
        GaussianLodPackageSource::try_from_manifest_uri(&config.manifest_url)
            .expect("absolute manifest URL"),
        config.lod(),
        config.streaming(),
        Transform::default(),
        CloudSettings {
            sort_mode: SortMode::Radix,
            ..default()
        },
    ));
    if config.point_gpu.is_some() {
        entity.insert(GaussianGpuLodPackage);
    }
    entity.id()
}

fn install_camera_backend(commands: &mut Commands, camera: Entity, config: &BrowserConfig) {
    if let Some(point) = &config.point_gpu {
        commands.entity(camera).insert((
            Msaa::Off,
            point.point(),
            point.traversal(config.max_active_gaussians),
        ));
    }
}

fn setup(mut commands: Commands, assets: Res<AssetServer>, mut session: ResMut<BrowserSession>) {
    let config = &session.config;
    let camera = commands
        .spawn((
            Camera3d::default(),
            GaussianCamera::default(),
            Tonemapping::None,
            Projection::Perspective(PerspectiveProjection {
                near: 0.01,
                far: 1_000_000.0,
                ..default()
            }),
            Transform::from_translation(Vec3::from_array(config.from))
                .looking_at(Vec3::from_array(config.target), Vec3::Y),
        ))
        .id();
    let cloud = spawn_package(&mut commands, &assets, config);
    install_camera_backend(&mut commands, camera, config);
    session.camera = Some(camera);
    session.cloud = Some(cloud);
}

fn drive_lifecycle(
    mut commands: Commands,
    assets: Res<AssetServer>,
    ledger: Res<LodMemoryLedger>,
    request: Res<BrowserCaptureRequest>,
    mut session: ResMut<BrowserSession>,
    mut cameras: Query<&mut Transform, With<GaussianCamera>>,
) {
    if session.terminal {
        return;
    }
    session.frame += 1;
    let elapsed = session.frame - session.phase_started;
    let attested = request
        .evidence
        .lock()
        .unwrap()
        .attested_phases
        .contains(session.phase.name());
    let next = match session.phase {
        Phase::ColdLoad if attested => Some(Phase::Stationary),
        Phase::Stationary if elapsed >= u64::from(session.config.phase_frames) && attested => {
            Some(Phase::Move)
        }
        Phase::Move if elapsed >= u64::from(session.config.phase_frames) && attested => {
            if let Some(cloud) = session.cloud.take() {
                commands.entity(cloud).despawn();
            }
            if let Some(camera) = session.camera {
                commands
                    .entity(camera)
                    .remove::<(GaussianPointSplattingSettings, GpuLodTraversalSettings)>();
            }
            Some(Phase::Unload)
        }
        Phase::Unload
            if elapsed >= u64::from(session.config.sample_every_frames)
                && ledger.snapshot().total_bytes == 0 =>
        {
            session.unload_complete = true;
            session.cloud = Some(spawn_package(&mut commands, &assets, &session.config));
            install_camera_backend(&mut commands, session.camera.unwrap(), &session.config);
            Some(Phase::Reload)
        }
        Phase::Reload if elapsed >= u64::from(session.config.phase_frames) && attested => {
            Some(Phase::Complete)
        }
        _ => None,
    };
    if let Some(next) = next {
        emit(json!({"kind":"transition", "frame":session.frame,
            "from":session.phase.name(), "to":next.name(), "memory":memory_json(&ledger)}));
        session.phase = next;
        session.phase_started = session.frame;
    }
    if let Some(camera) = session.camera
        && let Ok(mut transform) = cameras.get_mut(camera)
    {
        let fraction = if session.phase == Phase::Move {
            ((session.frame - session.phase_started) as f32 / session.config.phase_frames as f32)
                .min(1.0)
        } else if matches!(
            session.phase,
            Phase::Unload | Phase::Reload | Phase::Complete
        ) {
            1.0
        } else {
            0.0
        };
        let position = Vec3::from_array(session.config.from)
            .lerp(Vec3::from_array(session.config.to), fraction);
        *transform = Transform::from_translation(position)
            .looking_at(Vec3::from_array(session.config.target), Vec3::Y);
    }
}

fn observe_main(
    mut session: ResMut<BrowserSession>,
    mut request: ResMut<BrowserCaptureRequest>,
    ledger: Res<LodMemoryLedger>,
    statuses: Query<(
        &GaussianLodPackageStatus,
        Option<&GaussianLodPackageTestingSnapshot>,
        Option<&GpuLodHierarchy>,
    )>,
    cameras: Query<&GlobalTransform, With<GaussianCamera>>,
) {
    let mut evidence = request.evidence.lock().unwrap();
    for record in evidence.records.drain(..) {
        emit(record);
    }
    if session.terminal {
        drop(evidence);
        request.frame = None;
        return;
    }
    let package_failed = session
        .cloud
        .and_then(|cloud| statuses.get(cloud).ok())
        .is_some_and(|(status, _, _)| status.phase == GaussianLodPackagePhase::Failed);
    let failed = evidence.mapping_errors != 0 || evidence.invalid_counts != 0 || package_failed;
    let timeout = session.started.elapsed().as_secs() >= u64::from(session.config.timeout_seconds);
    if failed || timeout || session.phase == Phase::Complete {
        let passed =
            !failed && !timeout && session.unload_complete && session.phase == Phase::Complete;
        emit(json!({"kind":"terminal", "frame":session.frame,
            "execution_complete":passed, "release_qualified":false,
            "reason":if package_failed {"package_failure"} else if failed {"readback_or_count_failure"} else if timeout {"timeout"} else {"lifecycle_complete"},
            "attested_phases":evidence.attested_phases, "unload_complete":session.unload_complete,
            "mapping_errors":evidence.mapping_errors, "invalid_counts":evidence.invalid_counts,
            "dropped_readbacks":evidence.dropped_readbacks, "memory":memory_json(&ledger)}));
        session.terminal = true;
        drop(evidence);
        request.frame = None;
        return;
    }
    drop(evidence);
    let now = Instant::now();
    let wall_ms = now.duration_since(session.last_frame).as_secs_f64() * 1000.0;
    session.last_frame = now;
    let sampled = session
        .frame
        .is_multiple_of(u64::from(session.config.sample_every_frames));
    request.frame = sampled.then(|| BrowserFrame {
        frame: session.frame,
        phase: session.phase.name(),
        camera: session.camera.expect("startup camera"),
        point_gpu: session.config.point_gpu.clone(),
    });
    if !sampled {
        return;
    }
    let status = session.cloud.and_then(|cloud| statuses.get(cloud).ok());
    emit(
        json!({"kind":"main_frame", "frame":session.frame, "phase":session.phase.name(),
        "elapsed_ms":session.started.elapsed().as_secs_f64()*1000.0, "wall_frame_ms":wall_ms,
        "camera_world":session.camera.and_then(|camera|cameras.get(camera).ok()).map(|t|t.to_matrix().to_cols_array()),
        "gpu_snapshot":status.and_then(|(_, _, hierarchy)|hierarchy).map(|hierarchy|json!({
            "generation":hierarchy.0.generation.to_string(),
            "source_asset":format!("{:?}",hierarchy.0.source),
            "cloud":session.cloud.unwrap().to_bits().to_string()})),
        "package":status.map(|(status, work, _)|json!({"phase":format!("{:?}",status.phase),
            "resident_pages":status.resident_pages,"selected_gaussians":status.active_gaussians,
            "terminal_failures":status.terminal_failures,"error":status.error_detail(),
            "work":work.map(|work|json!({"available":work.runtime_work_available,
                "queued":work.runtime_request_queue_len,"transport_in_flight":work.runtime_transport_in_flight_requests,
                "preprocess_waiting":work.preprocess_waiting_jobs,"preprocess_running":work.preprocess_backend_tracked_jobs,
                "preprocess_ready":work.preprocess_ready_pages}))})),
        "memory":memory_json(&ledger)}),
    );
}
