//! Bounded, sequential real-source quad/GPS comparison with submission-stamped
//! GPU feedback. This is an instrumented screen, not a LoD acceptance result.

mod gpu;

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    asset::{AssetMetaCheck, RenderAssetUsages},
    camera::{RenderTarget, visibility::NoFrustumCulling},
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::{
        RenderApp,
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_interleave::prelude::Planar;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    CloudSettings, GaussianCamera, GaussianLodBridgeConfig, GaussianSplattingPlugin,
    PlanarGaussian3d, PlanarGaussian3dHandle,
    render::point::GaussianPointSplattingSettings,
    sort::SortMode,
    testing::lod_runtime_capture::{
        hash_file,
        source::{LoadedSource, load_source},
    },
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
const TIMED_FRAMES: u32 = 16;
const READBACK_SLOTS: usize = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    source: PathBuf,
    source_sha256: String,
    output: PathBuf,
    max_source_gaussians: u64,
    camera_from: [f32; 3],
    camera_target: [f32; 3],
    #[serde(default = "default_up")]
    camera_up: [f32; 3],
    #[serde(default = "default_fov")]
    vertical_fov_radians: f32,
    #[serde(default = "default_viewport")]
    viewport: [u32; 2],
    #[serde(default = "default_warmup")]
    warmup_frames: u32,
    #[serde(default = "default_timeout")]
    timeout_seconds: u32,
    #[serde(default = "default_true")]
    capture_images: bool,
    #[serde(default)]
    point_settings: GaussianPointSplattingSettings,
}
fn default_up() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}
fn default_fov() -> f32 {
    std::f32::consts::FRAC_PI_4
}
fn default_viewport() -> [u32; 2] {
    [1280, 720]
}
fn default_warmup() -> u32 {
    32
}
fn default_timeout() -> u32 {
    120
}
fn default_true() -> bool {
    true
}

impl Config {
    fn validate(&self) -> Result<()> {
        let from = Vec3::from_array(self.camera_from);
        let target = Vec3::from_array(self.camera_target);
        let up = Vec3::from_array(self.camera_up);
        let orientation_valid = (target - from)
            .try_normalize()
            .zip(up.try_normalize())
            .is_some_and(|(forward, up)| forward.cross(up).length_squared() > 1e-10);
        if self.viewport.contains(&0)
            || self.viewport[0] > 1280
            || self.viewport[1] > 720
            || !(1..=120).contains(&self.warmup_frames)
            || !(1..=300).contains(&self.timeout_seconds)
            || self.max_source_gaussians == 0
            || self.max_source_gaussians > 8_000_000
            || self.source.extension().and_then(|s| s.to_str()) != Some("ply")
            || self.source_sha256.len() != 64
            || !self.source_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || !from.is_finite()
            || !target.is_finite()
            || !up.is_finite()
            || !orientation_valid
            || !self.vertical_fov_radians.is_finite()
            || !(0.01..3.13).contains(&self.vertical_fov_radians)
            || self.point_settings.target_gpu_ms.is_some()
            || self.point_settings.max_gpu_bytes > 2 * 1024 * 1024 * 1024
        {
            return Err("invalid point comparison config: one PLY, <=8M records, <=1280x720, 1..120 warmup frames, <=300s, fixed sample layers and <=2GiB point allocations".into());
        }
        self.point_settings.validate()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Case {
    Quad,
    Gps1,
    Gps4,
}
impl Case {
    fn from_index(index: usize) -> Option<Self> {
        [Self::Quad, Self::Gps1, Self::Gps4].get(index).copied()
    }
    fn name(self) -> &'static str {
        match self {
            Self::Quad => "quad",
            Self::Gps1 => "gps1",
            Self::Gps4 => "gps4",
        }
    }
    fn samples(self) -> Option<u32> {
        match self {
            Self::Quad => None,
            Self::Gps1 => Some(1),
            Self::Gps4 => Some(4),
        }
    }
}

#[derive(Clone)]
struct Request {
    frame: u64,
    case: Case,
    camera: Entity,
    target: Handle<Image>,
}

#[derive(Default)]
struct SharedState {
    request: Option<Request>,
    dropped_ring_full: u64,
    unavailable_work: u64,
    errors: Vec<String>,
}

#[derive(Clone, Resource)]
struct Shared {
    config: Config,
    state: Arc<Mutex<SharedState>>,
    sender: mpsc::SyncSender<Completed>,
}

struct Completed {
    case: Case,
    index: u32,
    row: serde_json::Value,
    rgba: Option<Vec<u8>>,
}

#[derive(Resource)]
struct Session {
    receiver: Mutex<mpsc::Receiver<Completed>>,
    source: Option<LoadedSource>,
    camera: Option<Entity>,
    target: Option<Handle<Image>>,
    rows: Vec<serde_json::Value>,
    seen: [[bool; TIMED_FRAMES as usize]; 3],
    case_index: usize,
    frame: u64,
    started: Instant,
    identity: serde_json::Value,
    finished: bool,
}

/// Runs one bounded opt-in experiment. Output creation refuses prior directories.
pub fn run_point_comparison(config_path: &Path) -> Result<()> {
    if fs::metadata(config_path)?.len() > 64 * 1024 {
        return Err("point comparison config exceeds64KiB".into());
    }
    let input = fs::read(config_path)?;
    let mut config: Config = serde_json::from_slice(&input)?;
    config.validate()?;
    let base = config_path.parent().unwrap_or(Path::new("."));
    config.source = base.join(&config.source);
    config.output = base.join(&config.output);
    let max_bytes = config
        .max_source_gaussians
        .checked_mul(1024)
        .and_then(|v| v.checked_add(16 * 1024 * 1024))
        .ok_or("source byte bound overflow")?;
    if fs::metadata(&config.source)?.len() > max_bytes {
        return Err("PLY exceeds source byte ceiling".into());
    }
    if hash_file(&config.source)? != config.source_sha256 {
        return Err("source SHA256 mismatch".into());
    }
    let source = load_source(&config.source, config.max_source_gaussians)?;
    let count = source.cloud.len();
    if count == 0 {
        return Err("point comparison requires a nonempty source".into());
    }
    if hash_file(&config.source)? != config.source_sha256 {
        return Err("source changed during decoding".into());
    }
    if let Some(parent) = config.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&config.output)?;
    fs::write(config.output.join("config.input.json"), &input)?;
    fs::write(
        config.output.join("config.resolved.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    let executable = std::env::current_exe()?;
    let identity = serde_json::json!({
        "source_path":config.source,"source_sha256":config.source_sha256,"source_gaussians":count,
        "renderer_path":executable,"renderer_sha256":hash_file(&executable)?,
        "config_sha256":format!("{:x}",Sha256::digest(&input)),
        "world_from_local":source.transform.to_matrix().to_cols_array(),"color_space":source.color_space,
        "sh_degree":crate::material::spherical_harmonics::SH_DEGREE,
        "support":"quad fixed3sigma OBB rectangle; GPS radial3sigma rejection",
        "depth_metric":"quad forward camera depth (-view-space Z); GPS reverse-Z projected center",
        "timing_scope":"two same-encoder timestamps before Gaussian compaction/sort and after upscaling; excludes readback copies/PNG encoding; includes intervening per-view passes",
        "gpu_timing_unavailable":"null; no CPU-wall substitution",
        "count_scope":"quad actual indirect draw; GPS same-submission requested/dispatched attempts before viewport/support rejection",
        "quality_gate":false,"lod_gate":false,"readback_slots":READBACK_SLOTS,
    });
    let (sender, receiver) = mpsc::sync_channel(READBACK_SLOTS);
    let shared = Shared {
        config: config.clone(),
        state: Default::default(),
        sender,
    };
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::BLACK))
        .insert_resource(shared.clone())
        .insert_resource(Session {
            receiver: Mutex::new(receiver),
            source: Some(source),
            camera: None,
            target: None,
            rows: Vec::with_capacity(48),
            seen: [[false; 16]; 3],
            case_index: 0,
            frame: 0,
            started: Instant::now(),
            identity,
            finished: false,
        })
        .insert_resource(GaussianLodBridgeConfig {
            auto_build_flat_clouds: false,
            ..default()
        })
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>()
                .disable::<PipelinedRenderingPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_millis(1)))
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, advance);
    gpu::install(app.sub_app_mut(RenderApp), shared);
    let exit = app.run();
    if exit != AppExit::Success {
        return Err("point comparison did not complete; see report.json".into());
    }
    Ok(())
}

fn setup(
    mut commands: Commands,
    mut session: ResMut<Session>,
    shared: Res<Shared>,
    mut clouds: ResMut<Assets<PlanarGaussian3d>>,
    mut images: ResMut<Assets<Image>>,
) {
    let source = session.source.take().unwrap();
    commands.spawn((
        PlanarGaussian3dHandle(clouds.add(source.cloud)),
        CloudSettings {
            sort_mode: SortMode::Radix,
            color_space: source.color_space,
            opacity_adaptive_radius: false,
            ..default()
        },
        source.transform,
        Visibility::Visible,
        NoFrustumCulling,
    ));
    let mut image = Image::new_target_texture(
        shared.config.viewport[0],
        shared.config.viewport[1],
        TextureFormat::Rgba8UnormSrgb,
        None,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = images.add(image);
    let camera = commands
        .spawn((
            Camera3d::default(),
            RenderTarget::Image(target.clone().into()),
            Projection::Perspective(PerspectiveProjection {
                fov: shared.config.vertical_fov_radians,
                near: 0.01,
                far: 100_000.0,
                ..default()
            }),
            GaussianCamera::default(),
            Msaa::Off,
            Tonemapping::None,
            Transform::from_translation(Vec3::from_array(shared.config.camera_from)).looking_at(
                Vec3::from_array(shared.config.camera_target),
                Vec3::from_array(shared.config.camera_up),
            ),
        ))
        .id();
    session.camera = Some(camera);
    session.target = Some(target);
}

fn advance(
    mut commands: Commands,
    mut session: ResMut<Session>,
    shared: Res<Shared>,
    mut exit: MessageWriter<AppExit>,
) {
    if session.finished {
        return;
    }
    let completed: Vec<_> = session.receiver.lock().unwrap().try_iter().collect();
    for mut result in completed {
        let index = match result.case {
            Case::Quad => 0,
            Case::Gps1 => 1,
            Case::Gps4 => 2,
        };
        if result.index >= TIMED_FRAMES || session.seen[index][result.index as usize] {
            shared
                .state
                .lock()
                .unwrap()
                .errors
                .push("duplicate/out-of-range submitted sample".into());
            continue;
        }
        session.seen[index][result.index as usize] = true;
        if let Some(rgba) = result.rgba.take() {
            let name = format!("{}-{:02}.png", result.case.name(), result.index);
            let path = shared.config.output.join(&name);
            match write_image(&path, rgba, shared.config.viewport) {
                Ok(hash) => {
                    result.row["image"] = serde_json::json!({"path":name,"sha256":hash,"scope":"same-submission final camera target"})
                }
                Err(error) => shared.state.lock().unwrap().errors.push(error.to_string()),
            }
        }
        session.rows.push(result.row);
    }
    if session.seen[session.case_index].iter().all(|seen| *seen) {
        session.case_index += 1;
        if let Some(case) = Case::from_index(session.case_index) {
            let mut settings = shared.config.point_settings.clone();
            settings.samples_per_pixel = case.samples().unwrap();
            commands.entity(session.camera.unwrap()).insert(settings);
        }
    }
    let errors = shared.state.lock().unwrap().errors.clone();
    let done = session.case_index == 3;
    let timeout =
        session.started.elapsed() > Duration::from_secs(u64::from(shared.config.timeout_seconds));
    if done || timeout || !errors.is_empty() {
        shared.state.lock().unwrap().request = None;
        let valid = session.rows.iter().all(|row| row["complete_work"] == true);
        let status = if done && valid && errors.is_empty() {
            "complete"
        } else {
            "failed"
        };
        let shared_state = shared.state.lock().unwrap();
        let report = serde_json::json!({"schema_version":1,"kind":"instrumented_quad_point_real_scene_comparison",
            "status":status,"identity":session.identity,"configuration":shared.config,"frames":session.rows,
            "timed_frames_per_case":TIMED_FRAMES,"warmup_rendered_frames_per_case":shared.config.warmup_frames,
            "gpu_timing_complete":session.rows.len()==48 && session.rows.iter().all(|row|row["gpu_ms"].is_number()),
            "timeout":timeout,"errors":errors,"dropped_ring_full":shared_state.dropped_ring_full,
            "unavailable_work_frames":shared_state.unavailable_work,"elapsed_seconds":session.started.elapsed().as_secs_f64(),
            "limits":["one stationary camera; no hierarchy/streaming or LoD-quality qualification","three stochastic images per GPS case are a noise screen, not convergence proof","timestamps include instrumentation scheduling but exclude feedback/image copies","source backend and support/order differences must be retained in image interpretation"]});
        drop(shared_state);
        let written = (|| -> Result<()> {
            let mut writer =
                BufWriter::new(File::create(shared.config.output.join("report.json"))?);
            serde_json::to_writer_pretty(&mut writer, &report)?;
            writeln!(writer)?;
            writer.flush()?;
            Ok(())
        })();
        session.finished = true;
        exit.write(if status == "complete" && written.is_ok() {
            AppExit::Success
        } else {
            AppExit::error()
        });
        return;
    }
    session.frame += 1;
    if let (Some(camera), Some(target)) = (session.camera, session.target.clone()) {
        shared.state.lock().unwrap().request = Some(Request {
            frame: session.frame,
            case: Case::from_index(session.case_index).unwrap(),
            camera,
            target,
        });
    }
}

fn write_image(path: &Path, rgba: Vec<u8>, viewport: [u32; 2]) -> Result<String> {
    Image::new(
        Extent3d {
            width: viewport[0],
            height: viewport[1],
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
    .try_into_dynamic()?
    .save(path)?;
    hash_file(path)
}
