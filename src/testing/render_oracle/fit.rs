//! Bounded diagnostic representative fitting. The forward pass shares the CPU
//! production oracle's projection, support, SH and color conversion. GPU parity
//! is deliberately not implied. Support boundaries and sort changes are handled
//! by accepting a proposal only after a complete fresh forward pass.

use std::{
    mem::size_of,
    time::{Duration, Instant},
};

use bevy::prelude::Quat;
use serde::{Deserialize, Serialize};

use super::*;
use crate::{
    gaussian::formats::planar_3d_chunked::LodBounds, material::spherical_harmonics::SH_COEFF_COUNT,
};

const PARAMS: usize = SH_COEFF_COUNT + 10;
const OPACITY: usize = SH_COEFF_COUNT;
const GEOMETRY: usize = OPACITY + 1;
pub const MAX_TAPE_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_FIT_VIEWS: usize = 16;
pub const MAX_FIT_WORKING_BYTES: usize = 256 * 1024 * 1024;
/// Two 512x512 images containing composed RGB and owned transmittance.
pub const MAX_TEACHER_BYTES: usize = 2 * 512 * 512 * size_of::<[f32; 4]>();
type FitResult<T> = Result<T, String>;
type Gradient = [f64; PARAMS];
const LOCAL_GEOMETRY_ATTEMPTS: u32 = 6;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryFeasibilityPolicy {
    #[default]
    RejectWholeProposal,
    PerRepresentativeBacktracking,
}

/// Coordinates of the three Adam mean parameters; all other parameters retain
/// their existing coordinates. Local axes/scales are frozen from the seed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeanCoordinates {
    #[default]
    World,
    RepresentativeLocal {
        trust_region_fraction: f32,
    },
}

type MeanFrame = [[f32; 3]; 3];

fn initial_mean_frame(gaussian: &Gaussian3d) -> MeanFrame {
    let [w, x, y, z] = gaussian.rotation.rotation;
    let rotation = Quat::from_xyzw(x, y, z, w).normalize();
    std::array::from_fn(|axis| {
        (rotation * [Vec3::X, Vec3::Y, Vec3::Z][axis] * gaussian.scale_opacity.scale[axis])
            .to_array()
    })
}

fn mean_displacement(frame: &MeanFrame, local: Vec3) -> Vec3 {
    Vec3::from_array(frame[0]) * local.x
        + Vec3::from_array(frame[1]) * local.y
        + Vec3::from_array(frame[2]) * local.z
}

/// Supervise a fixed grid of actual pixels from a larger deployment viewport.
/// Projection and Mip filtering use `viewport`; only the sampled RGBA grid is
/// allocated. Both viewport axes must be the same integer multiple of each
/// fit view's allocated size, and `offset` must lie inside that stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitReferencePixels {
    pub viewport: [u32; 2],
    pub offset: [u32; 2],
}

impl FitReferencePixels {
    pub(crate) fn sample_stride(self, allocated: [u32; 2]) -> FitResult<u32> {
        let [width, height] = allocated;
        let [reference_width, reference_height] = self.viewport;
        if width == 0 || height == 0 {
            return Err("reference pixel sampling requires a nonempty allocated viewport".into());
        }
        let stride = reference_width / width;
        if reference_width == 0
            || reference_height == 0
            || reference_width > 8192
            || reference_height > 8192
            || stride == 0
            || stride > 256
            || reference_width % width != 0
            || reference_height % height != 0
            || reference_height / height != stride
            || self.offset.iter().any(|offset| *offset >= stride)
        {
            return Err("reference pixels require a uniform integer stride in 1..=256, viewport <=8192 per axis, and an offset inside the stride".into());
        }
        Ok(stride)
    }
}

/// Admit the entire retained training image set before rendering any teacher.
/// Each target is composed RGB plus owned-only transmittance (16 bytes/pixel).
/// This bounds retained target storage, not total fitter working set or RSS.
pub(crate) fn training_teacher_bytes(
    viewports: impl IntoIterator<Item = [u32; 2]>,
) -> FitResult<usize> {
    let mut total = 0_usize;
    for [width, height] in viewports {
        let bytes = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(size_of::<[f32; 4]>()))
            .ok_or("training teacher RGBA allocation overflow")?;
        total = total
            .checked_add(bytes)
            .ok_or("training teacher RGBA allocation overflow")?;
    }
    if total > MAX_TEACHER_BYTES {
        return Err("training teacher RGBA storage exceeds the unchanged 8 MiB ceiling".into());
    }
    Ok(total)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FitOptions {
    pub max_steps: u32,
    pub max_seconds: u32,
    /// Optional deadline for teachers/optimization within max_seconds. Zero
    /// preserves the shared deadline; heldout evaluation gets remaining time.
    pub max_training_seconds: u32,
    pub max_tape_bytes: usize,
    pub max_pixel_visits_per_forward: u64,
    pub learning_rate: f64,
    /// Weight of owned-only transmittance MSE; with no context this equals
    /// opacity MSE because T = 1 - alpha.
    pub alpha_weight: f64,
    /// None renders the allocated viewport directly. Some preserves the native
    /// pixel filter at a larger viewport without retaining full-size teachers.
    pub reference_pixels: Option<FitReferencePixels>,
    /// Highest SH degree the optimizer may change. Higher authored coefficients
    /// remain active in rendering and are preserved exactly in every proposal.
    pub max_fitted_sh_degree: u32,
    pub fit_geometry: bool,
    pub geometry_feasibility: GeometryFeasibilityPolicy,
    pub mean_coordinates: MeanCoordinates,
    /// World-mode maximum displacement per axis/update; unused in local mode.
    pub mean_step_limit: f32,
    /// Stop after this many complete updates without a strict training-loss
    /// decrease. Zero preserves the original step/deadline-only behavior.
    pub training_stagnation_patience: u32,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            max_steps: 256,
            max_seconds: 180,
            max_training_seconds: 0,
            max_tape_bytes: MAX_TAPE_BYTES,
            max_pixel_visits_per_forward: 16_000_000,
            learning_rate: 0.03,
            alpha_weight: 1.0,
            reference_pixels: None,
            max_fitted_sh_degree: SH_DEGREE as u32,
            fit_geometry: false,
            geometry_feasibility: GeometryFeasibilityPolicy::RejectWholeProposal,
            mean_coordinates: MeanCoordinates::World,
            mean_step_limit: 0.002,
            training_stagnation_patience: 0,
        }
    }
}

impl FitOptions {
    pub fn validate(&self) -> FitResult<()> {
        if self.max_steps == 0
            || self.max_steps > 256
            || self.training_stagnation_patience > 256
            || self.max_seconds == 0
            || self.max_seconds > 180
            || self.max_training_seconds > self.max_seconds
            || self.max_tape_bytes < size_of::<Contribution>()
            || self.max_tape_bytes > MAX_TAPE_BYTES
            || self.max_pixel_visits_per_forward == 0
            || self.max_pixel_visits_per_forward > 128_000_000
            || !self.learning_rate.is_finite()
            || !(0.0..=0.1).contains(&self.learning_rate)
            || self.learning_rate == 0.0
            || !self.alpha_weight.is_finite()
            || !(0.0..=10.0).contains(&self.alpha_weight)
            || !self.mean_step_limit.is_finite()
            || !(0.0..=0.01).contains(&self.mean_step_limit)
            || self.mean_step_limit == 0.0
            || matches!(self.mean_coordinates, MeanCoordinates::RepresentativeLocal { trust_region_fraction } if !trust_region_fraction.is_finite() || !(0.0..=0.25).contains(&trust_region_fraction) || trust_region_fraction == 0.0)
            || !matches!(SH_DEGREE, 0..=3)
            || self.max_fitted_sh_degree > SH_DEGREE as u32
            || self.max_fitted_sh_degree > 3
            || self.reference_pixels.is_some_and(|pixels| {
                pixels.viewport.contains(&0)
                    || pixels.viewport.iter().any(|size| *size > 8192)
                    || pixels.offset.iter().any(|offset| *offset >= 8192)
            })
        {
            return Err("invalid fitter options: hard limits 256 steps, 180 seconds, 128 MiB tape, 128M visits; fitted SH degree must not exceed the compiled degree or 3".into());
        }
        Ok(())
    }

    fn fitted_sh_coefficients(&self) -> usize {
        // validate() bounds the degree before any optimizer operation.
        3 * (self.max_fitted_sh_degree as usize + 1).pow(2)
    }
}

#[derive(Clone, Debug)]
pub struct FitView {
    pub id: String,
    pub camera: LodTestCamera,
}

#[derive(Clone, Debug, Serialize)]
pub struct FitStep {
    pub step: u32,
    pub loss: f64,
    pub accepted_scale: Option<f64>,
    pub ownership_rejections: u32,
    /// At most four global proposals; counts belong to their stated outcome.
    pub trials: Vec<FitTrial>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct FitGeometryFeasibility {
    /// An ownership-rejected whole-proposal trial may stop before visiting every owner.
    pub representatives_considered: u32,
    pub requested_representatives: u32,
    /// Feasible at a factor below one; excludes fully frozen geometry.
    pub limited_representatives: u32,
    pub frozen_representatives: u32,
    pub ownership_checks: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FitTrialOutcome {
    Accepted,
    TrainingLossRejected,
    OwnershipRejected,
    TrainingDeadline,
}

#[derive(Clone, Debug, Serialize)]
pub struct FitTrial {
    pub scale: f64,
    pub outcome: FitTrialOutcome,
    pub geometry: FitGeometryFeasibility,
}

struct FitProposal {
    gaussians: Vec<Gaussian3d>,
    geometry: FitGeometryFeasibility,
}

#[derive(Clone, Debug, Serialize)]
pub struct HeldoutLoss {
    pub id: String,
    pub initial_loss: f64,
    pub final_loss: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FitWork {
    /// Admitted retained RGB/transmittance targets, excluding tape and scratch.
    pub training_teacher_bytes: usize,
    /// Retained source/seed images in an explicit no-optimization diagnostic.
    pub diagnostic_image_bytes: usize,
    pub owned_source_records: usize,
    pub owned_candidate_records: usize,
    pub immutable_context_records: usize,
    /// Two Adam moments per trainable record; context has no optimizer state.
    pub optimizer_moment_bytes: usize,
    /// Checked upper bound for simultaneously owned fit arrays, including the
    /// admitted tape and teacher ceiling. Excludes borrowed input storage/RSS.
    pub admitted_working_bytes: usize,
    /// Logical full-view passes, not the number of gradient tiles.
    pub forward_passes: u64,
    pub projection_passes: u64,
    /// Includes immutable context on every fresh forward/gradient projection.
    pub projected_records: u64,
    pub gradient_tiles: u64,
    pub gradient_tile_splits: u64,
    pub tile_bound_tests: u64,
    pub pixel_visits: u64,
    pub contributions: u64,
    pub peak_tape_bytes: usize,
    pub local_projection_evaluations: u64,
    pub elapsed_seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct FitReport {
    pub mean_coordinates: MeanCoordinates,
    /// World-unit basis columns in owned-seed order, frozen for all updates.
    /// Empty in world mode. The corresponding mode is recorded in options.
    pub mean_coordinate_frames: Vec<[[f32; 3]; 3]>,
    pub initial_training_loss: f64,
    pub final_training_loss: f64,
    pub steps: Vec<FitStep>,
    /// Unpublished proposals from a final interrupted update, separate from
    /// completed steps and from every accepted-trial count.
    pub interrupted_step_trials: Vec<FitTrial>,
    pub stop_reason: String,
    pub heldout: Vec<HeldoutLoss>,
    pub heldout_complete: bool,
    pub work: FitWork,
}

pub struct FitOutput {
    pub gaussians: Vec<Gaussian3d>,
    pub report: FitReport,
}

/// Frozen supervision and explicit trainability. Only `initial` is optimized
/// and returned; `source` and `context` are immutable throughout the call.
/// Context participates in the same depth order for RGB, but is excluded from
/// the owned-transmittance target and from all optimizer parameters.
#[derive(Clone, Copy)]
pub struct FitProblem<'a> {
    pub source: &'a [Gaussian3d],
    pub initial: &'a [Gaussian3d],
    pub context: &'a [Gaussian3d],
    pub domains: &'a [LodBounds],
    /// Optional deployment order for exact depth ties. Supply all nonempty
    /// inputs together; keys must be unique within each composed scene.
    pub source_order: Option<&'a [u64]>,
    pub initial_order: Option<&'a [u64]>,
    pub context_order: Option<&'a [u64]>,
}

#[derive(Serialize)]
pub struct FitDiagnosticView {
    pub id: String,
    pub viewport: [u32; 2],
    pub objective_loss: f64,
    pub composed_rgb_mse: f64,
    pub owned_transmittance_mse: f64,
    /// Linear composed RGB plus owned-only T; no display transfer is applied.
    #[serde(skip_serializing)]
    pub source: Vec<[f32; 4]>,
    #[serde(skip_serializing)]
    pub initial: Vec<[f32; 4]>,
}

#[derive(Serialize)]
pub struct FitDiagnosticOutput {
    pub views: Vec<FitDiagnosticView>,
    pub work: FitWork,
}

/// Render unchanged seed/source substitutions with immutable context. No
/// optimizer state, parameter proposal, or training/heldout decision is made.
/// Both image sets are retained under the target and working-array ceilings.
pub fn diagnose_representatives(
    problem: FitProblem<'_>,
    views: &[FitView],
    color_space: GaussianColorSpace,
    options: &FitOptions,
) -> FitResult<FitDiagnosticOutput> {
    compare_representatives_inner(problem, views, color_space, options, true)
}

/// Compare an accepted native candidate to its artifact reload. Both-blank
/// views remain useful equality witnesses; this is not a training objective.
pub fn compare_representatives(
    problem: FitProblem<'_>,
    views: &[FitView],
    color_space: GaussianColorSpace,
    options: &FitOptions,
) -> FitResult<FitDiagnosticOutput> {
    compare_representatives_inner(problem, views, color_space, options, false)
}

fn compare_representatives_inner(
    problem: FitProblem<'_>,
    views: &[FitView],
    color_space: GaussianColorSpace,
    options: &FitOptions,
    require_owned_foreground: bool,
) -> FitResult<FitDiagnosticOutput> {
    validate_problem(&problem, views, &[], options)?;
    let image_bytes = training_teacher_bytes(views.iter().map(|view| view.camera.viewport))?
        .checked_mul(2)
        .ok_or("diagnostic image allocation overflow")?;
    let mut budget = Budget::new(options);
    budget.work.owned_source_records = problem.source.len();
    budget.work.owned_candidate_records = problem.initial.len();
    budget.work.immutable_context_records = problem.context.len();
    budget.work.diagnostic_image_bytes = image_bytes;
    budget.work.admitted_working_bytes = admit_working_arrays(
        &problem,
        image_bytes,
        views
            .iter()
            .map(|view| (view.camera.viewport[0] * view.camera.viewport[1]) as usize)
            .max()
            .unwrap_or(0),
        options,
    )?;
    let mut images = Vec::with_capacity(views.len());
    for view in views {
        let projection = ViewContext::with_reference_pixels(view.camera, options.reference_pixels)?;
        let source = forward_scene(
            FitScene {
                owned: problem.source,
                context: problem.context,
                owned_order: problem.source_order,
                context_order: problem.context_order,
            },
            projection,
            color_space,
            false,
            &mut budget,
        )?
        .image;
        let initial = forward_scene(
            FitScene {
                owned: problem.initial,
                context: problem.context,
                owned_order: problem.initial_order,
                context_order: problem.context_order,
            },
            projection,
            color_space,
            false,
            &mut budget,
        )?
        .image;
        if require_owned_foreground && !source.iter().chain(&initial).any(|pixel| pixel[3] < 0.98) {
            return Err("diagnostic source and seed have no owned foreground".into());
        }
        let composed_rgb_mse = loss(&initial, &source, 0.0);
        let owned_transmittance_mse = initial
            .iter()
            .zip(&source)
            .map(|(a, b)| (f64::from(a[3]) - f64::from(b[3])).powi(2))
            .sum::<f64>()
            / initial.len() as f64;
        if !composed_rgb_mse.is_finite() || !owned_transmittance_mse.is_finite() {
            return Err("nonfinite diagnostic objective".into());
        }
        images.push(FitDiagnosticView {
            id: view.id.clone(),
            viewport: view.camera.viewport,
            objective_loss: composed_rgb_mse + options.alpha_weight * owned_transmittance_mse,
            composed_rgb_mse,
            owned_transmittance_mse,
            source,
            initial,
        });
    }
    budget.work.elapsed_seconds = budget.start.elapsed().as_secs_f64();
    Ok(FitDiagnosticOutput {
        views: images,
        work: budget.work,
    })
}

#[derive(Clone, Copy)]
struct FitScene<'a> {
    owned: &'a [Gaussian3d],
    context: &'a [Gaussian3d],
    owned_order: Option<&'a [u64]>,
    context_order: Option<&'a [u64]>,
}

impl<'a> FitScene<'a> {
    fn order_key(self, index: usize) -> u64 {
        if index < self.owned.len() {
            self.owned_order.map_or(index as u64, |order| order[index])
        } else {
            self.context_order
                .map_or(index as u64, |order| order[index - self.owned.len()])
        }
    }

    #[cfg(test)]
    fn owned(gaussians: &'a [Gaussian3d]) -> Self {
        Self {
            owned: gaussians,
            context: &[],
            owned_order: None,
            context_order: None,
        }
    }
}

fn admit_working_arrays(
    problem: &FitProblem<'_>,
    teacher_bytes: usize,
    pixels: usize,
    options: &FitOptions,
) -> FitResult<usize> {
    let owned = problem.initial.len();
    let candidate = owned
        .checked_add(problem.context.len())
        .ok_or("fit count overflow")?;
    let projected = problem
        .source
        .len()
        .max(owned)
        .checked_add(problem.context.len())
        .ok_or("fit count overflow")?;
    // Conservative simultaneous capacities: projection merge/sort scratch,
    // candidate/current arrays, two Adam moments plus aggregate/view gradients,
    // per-projection adjoints/bounds, heldout teacher/before/after images, and
    // one reusable tape. No optimizer array is allocated for context.
    let arrays = [
        (1, teacher_bytes),
        (1, options.max_tape_bytes),
        (projected, 4 * size_of::<ProductionProjectedGaussian>()),
        (projected, size_of::<u64>()), // explicit-order validation scratch
        (
            owned,
            2 * size_of::<Gaussian3d>() + 4 * size_of::<Gradient>(),
        ),
        (
            owned,
            if matches!(
                options.mean_coordinates,
                MeanCoordinates::RepresentativeLocal { .. }
            ) {
                size_of::<MeanFrame>()
            } else {
                0
            },
        ),
        (
            candidate,
            size_of::<PixelTile>() + size_of::<ProjectedGradient>(),
        ),
        (pixels, 3 * size_of::<[f32; 4]>()),
        (32 * 32, size_of::<[f64; 4]>()),
        (16, size_of::<PixelTile>()),
        (
            options.max_steps as usize,
            size_of::<FitStep>() + 4 * size_of::<FitTrial>(),
        ),
    ];
    let bytes = arrays
        .into_iter()
        .try_fold(0usize, |total, (count, stride)| {
            count
                .checked_mul(stride)
                .and_then(|bytes| total.checked_add(bytes))
        })
        .ok_or("fit working allocation overflow")?;
    if bytes > MAX_FIT_WORKING_BYTES {
        return Err("fit working arrays exceed the 256 MiB ceiling".into());
    }
    Ok(bytes)
}

struct Budget {
    start: Instant,
    deadline: Instant,
    options: FitOptions,
    work: FitWork,
    mean_frames: Vec<MeanFrame>,
}
impl Budget {
    fn new(options: &FitOptions) -> Self {
        let start = Instant::now();
        Self {
            start,
            deadline: start + Duration::from_secs(options.max_seconds.into()),
            options: options.clone(),
            work: FitWork::default(),
            mean_frames: Vec::new(),
        }
    }
    fn check(&self) -> FitResult<()> {
        self.check_at(Instant::now())
    }
    fn check_at(&self, now: Instant) -> FitResult<()> {
        if now >= self.deadline {
            Err("wall_time_budget".into())
        } else {
            Ok(())
        }
    }
    fn begin_training(&mut self) {
        if self.options.max_training_seconds > 0 {
            self.deadline =
                self.start + Duration::from_secs(self.options.max_training_seconds.into());
        }
    }
    fn begin_evaluation(&mut self) {
        self.deadline = self.start + Duration::from_secs(self.options.max_seconds.into());
    }
}

#[derive(Clone, Copy)]
struct ViewContext {
    camera: LodTestCamera,
    width: u32,
    height: u32,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    projection: ProductionProjection,
    world_support: ProductionWorldSupport,
    projection_viewport: [u32; 2],
    sample_stride: f32,
    sample_origin: Vec2,
}
impl ViewContext {
    #[cfg(test)]
    fn new(camera: LodTestCamera) -> FitResult<Self> {
        Self::with_reference_pixels(camera, None)
    }

    fn with_reference_pixels(
        camera: LodTestCamera,
        reference_pixels: Option<FitReferencePixels>,
    ) -> FitResult<Self> {
        let [width, height] = camera.viewport;
        if width == 0
            || height == 0
            || width > 512
            || height > 512
            || !camera.position.is_finite()
            || !camera.target.is_finite()
            || !camera.up.is_finite()
            || !camera.near.is_finite()
            || !camera.far.is_finite()
            || camera.near <= 0.0
            || camera.far <= camera.near
        {
            return Err("invalid fit camera or image size (maximum 512x512)".into());
        }
        let (forward, right, up) = camera.basis().map_err(str::to_owned)?;
        let (projection_viewport, sample_stride, sample_origin) = match reference_pixels {
            Some(pixels) => {
                let stride = pixels.sample_stride(camera.viewport)?;
                (
                    pixels.viewport,
                    stride as f32,
                    Vec2::from_array(pixels.offset.map(|offset| offset as f32))
                        + Vec2::splat(0.5 - 0.5 * stride as f32),
                )
            }
            None => (camera.viewport, 1.0, Vec2::ZERO),
        };
        let projection = super::production_projection(camera.projection, projection_viewport)
            .map_err(|error| error.to_string())?;
        let world_support = ProductionWorldSupport::new(camera, projection_viewport)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            camera,
            width,
            height,
            forward,
            right,
            up,
            projection,
            world_support,
            projection_viewport,
            sample_stride,
            sample_origin,
        })
    }
    fn geometry(self, gaussian: &Gaussian3d) -> Option<ProductionProjectedGeometry> {
        if !self
            .world_support
            .contains(gaussian, GAUSSIAN_AUTHORED_SUPPORT_SIGMA)
        {
            return None;
        }
        let mut geometry = production_projected_geometry(
            gaussian,
            self.camera,
            self.projection_viewport[0],
            self.projection_viewport[1],
            self.forward,
            self.right,
            self.up,
            self.projection,
        )?;
        // Preserve the deployment projection's determinant opacity correction.
        // Mapping its conic to the sample grid is a coordinate change only; it
        // must not apply the low-resolution pixel filter a second time.
        geometry.center = (geometry.center - self.sample_origin) / self.sample_stride;
        let scale_squared = self.sample_stride * self.sample_stride;
        geometry.covariance = geometry.covariance.map(|value| value / scale_squared);
        geometry.inverse_covariance = geometry
            .inverse_covariance
            .map(|value| value * scale_squared);
        Some(geometry)
    }
}

#[cfg(test)]
mod calibrated_projection_tests {
    use super::*;
    use crate::testing::lod_scenes::LodPixelCrop;

    #[test]
    fn physical_crop_retains_deployment_frustum_and_rejects_offscreen_jacobian_tail() {
        let camera = LodTestCamera {
            position: Vec3::ZERO,
            target: -Vec3::Z,
            projection: LodProjection::Calibrated {
                focal_length_px: Vec2::new(95.0, 130.0),
                principal_point_px: Vec2::new(64.0, 48.0),
                image_size: [128, 96],
                crop: None,
            },
            near: 0.1,
            viewport: [128, 96],
            ..Default::default()
        };
        let full = ViewContext::new(camera).unwrap();
        let crop = ViewContext::new(
            camera
                .with_crop(LodPixelCrop {
                    origin: [64, 48],
                    size: [1, 1],
                })
                .unwrap(),
        )
        .unwrap();
        assert_eq!(full.world_support.frustum, crop.world_support.frustum);
        assert_eq!(
            full.world_support.min_shader_focal,
            crop.world_support.min_shader_focal
        );
        let mut offscreen = Gaussian3d::default();
        offscreen.position_visibility.position = [4.0, 0.0, -1.0];
        offscreen.scale_opacity.scale = [0.1, 0.1, 0.5];
        offscreen.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
        let raw = production_projected_geometry(
            &offscreen,
            crop.camera,
            1,
            1,
            crop.forward,
            crop.right,
            crop.up,
            crop.projection,
        )
        .unwrap();
        let delta = 2.0 * (Vec2::splat(0.5) - raw.center);
        assert!(
            production_obb(raw.covariance, 3.0)
                .unwrap()
                .contains_shader_delta(delta),
            "fixture's offscreen perspective Jacobian must reach the cropped pixel"
        );
        assert!(full.geometry(&offscreen).is_none());
        assert!(crop.geometry(&offscreen).is_none());

        // Infinite reverse-Z admits a perspective center beyond finite far
        // while its support overlaps that plane, including in a physical crop.
        let far_camera = LodTestCamera { far: 2.0, ..camera };
        let far_full = ViewContext::new(far_camera).unwrap();
        let far_crop = ViewContext::new(
            far_camera
                .with_crop(LodPixelCrop {
                    origin: [64, 48],
                    size: [1, 1],
                })
                .unwrap(),
        )
        .unwrap();
        let mut crossing_far = offscreen;
        crossing_far.position_visibility.position = [0.0, 0.0, -2.2];
        crossing_far.scale_opacity.scale = [0.2; 3];
        assert!(far_full.world_support.contains(&crossing_far, 3.0));
        assert!(far_full.geometry(&crossing_far).is_some());
        assert!(far_crop.geometry(&crossing_far).is_some());
        let orthographic = ViewContext::new(LodTestCamera {
            projection: LodProjection::Orthographic {
                vertical_world_size: 2.0,
            },
            ..far_camera
        })
        .unwrap();
        assert!(orthographic.world_support.contains(&crossing_far, 3.0));
        assert!(orthographic.geometry(&crossing_far).is_none());
        crossing_far.position_visibility.position[2] = -3.0;
        assert!(!far_full.world_support.contains(&crossing_far, 3.0));
        assert!(far_full.geometry(&crossing_far).is_none());
        assert!(far_crop.geometry(&crossing_far).is_none());
    }

    #[test]
    fn calibrated_fit_crop_samples_native_filter_at_reference_pixel_centers() {
        let camera = LodTestCamera {
            position: Vec3::ZERO,
            target: -Vec3::Z,
            projection: LodProjection::Calibrated {
                focal_length_px: Vec2::new(95.0, 130.0),
                principal_point_px: Vec2::new(53.0, 47.0),
                image_size: [128, 96],
                crop: None,
            },
            viewport: [128, 96],
            ..Default::default()
        }
        .with_crop(LodPixelCrop {
            origin: [40, 30],
            size: [40, 32],
        })
        .unwrap();
        let native = ViewContext::new(camera).unwrap();
        let sampled = ViewContext::with_reference_pixels(
            LodTestCamera {
                viewport: [20, 16],
                ..camera
            },
            Some(FitReferencePixels {
                viewport: [40, 32],
                offset: [1, 0],
            }),
        )
        .unwrap();
        let mut gaussian =
            crate::testing::LodTestScene::screen_space_ladder().gaussians[0].gaussian;
        gaussian.position_visibility.position = [0.1, -0.05, -3.0];
        gaussian.scale_opacity.scale = [0.09, 0.17, 0.04];
        let native = native.geometry(&gaussian).unwrap();
        let sampled = sampled.geometry(&gaussian).unwrap();
        assert_eq!(sampled.opacity_scale, native.opacity_scale);
        assert_eq!(sampled.center, (native.center - Vec2::new(0.5, -0.5)) / 2.0);
        assert_eq!(sampled.covariance, native.covariance.map(|v| v / 4.0));
        assert_eq!(
            sampled.inverse_covariance,
            native.inverse_covariance.map(|v| v * 4.0)
        );
        assert!(
            ViewContext::with_reference_pixels(
                LodTestCamera {
                    viewport: [20, 16],
                    ..camera
                },
                Some(FitReferencePixels {
                    viewport: [80, 64],
                    offset: [0, 0]
                })
            )
            .is_err()
        );
    }
}

#[derive(Clone, Copy)]
struct Contribution {
    pixel: u32,
    projected: u32,
    alpha: f32,
    before: [f32; 4],
}
#[cfg_attr(not(test), allow(dead_code))]
struct Forward {
    image: Vec<[f32; 4]>,
    projected: Vec<ProductionProjectedGaussian>,
    tape: Vec<Contribution>,
}

fn prepare_projected(
    scene: FitScene<'_>,
    view: ViewContext,
    color_space: GaussianColorSpace,
    budget: &mut Budget,
) -> FitResult<Vec<ProductionProjectedGaussian>> {
    budget.check()?;
    budget.work.projection_passes += 1;
    let mut projected = Vec::with_capacity(scene.owned.len() + scene.context.len());
    for (input_order, gaussian) in scene.owned.iter().chain(scene.context).enumerate() {
        if input_order % 1024 == 0 {
            budget.check()?;
        }
        if gaussian.scale_opacity.opacity <= 0.0 {
            continue;
        }
        let Some(geometry) = view.geometry(gaussian) else {
            continue;
        };
        let opacity = (gaussian.scale_opacity.opacity * geometry.opacity_scale).clamp(0.0, 1.0);
        if !opacity.is_finite() || opacity <= 0.0 {
            continue;
        }
        let Some(obb) = production_obb(geometry.covariance, GAUSSIAN_AUTHORED_SUPPORT_SIGMA) else {
            continue;
        };
        let color = production_spherical_harmonics_linear_color(
            geometry.relative.normalize(),
            &gaussian.spherical_harmonic,
            color_space,
        );
        if color.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite projected color".into());
        }
        projected.push(ProductionProjectedGaussian {
            node: None,
            input_order,
            view_depth: geometry.view_depth,
            center: geometry.center,
            inverse_covariance: geometry.inverse_covariance,
            color,
            opacity,
            cutoff_squared: GAUSSIAN_AUTHORED_SUPPORT_SIGMA.powi(2),
            obb,
        });
    }
    projected.sort_by(|a, b| {
        b.view_depth.total_cmp(&a.view_depth).then_with(|| {
            scene
                .order_key(a.input_order)
                .cmp(&scene.order_key(b.input_order))
        })
    });
    Ok(projected)
}

fn prepare_scene_projected(
    scene: FitScene<'_>,
    view: ViewContext,
    color_space: GaussianColorSpace,
    budget: &mut Budget,
) -> FitResult<Vec<ProductionProjectedGaussian>> {
    budget.work.projected_records += (scene.owned.len() + scene.context.len()) as u64;
    prepare_projected(scene, view, color_space, budget)
}

#[cfg(test)]
fn forward(
    gaussians: &[Gaussian3d],
    view: ViewContext,
    color_space: GaussianColorSpace,
    record: bool,
    budget: &mut Budget,
) -> FitResult<Forward> {
    forward_scene(
        FitScene::owned(gaussians),
        view,
        color_space,
        record,
        budget,
    )
}

fn forward_scene(
    scene: FitScene<'_>,
    view: ViewContext,
    color_space: GaussianColorSpace,
    record: bool,
    budget: &mut Budget,
) -> FitResult<Forward> {
    budget.check()?;
    budget.work.forward_passes += 1;
    let projected = prepare_scene_projected(scene, view, color_space, budget)?;
    let mut image = vec![[0.0, 0.0, 0.0, 1.0]; (view.width * view.height) as usize];
    let mut tape = Vec::<Contribution>::new();
    let max_entries = budget.options.max_tape_bytes / size_of::<Contribution>();
    let mut visits = 0_u64;
    for (index, gaussian) in projected.iter().enumerate() {
        budget.check()?;
        let radius = gaussian.obb.aabb_radius_pixels;
        if !radius.is_finite() {
            return Err("nonfinite support radius".into());
        }
        // Float clipping before integer conversion avoids overflow for large off-axis supports.
        let min_x = (gaussian.center.x.floor() - radius.x.ceil()).max(0.0) as i64;
        let max_x =
            (gaussian.center.x.ceil() + radius.x.ceil()).min(view.width as f32 - 1.0) as i64;
        let min_y = (gaussian.center.y.floor() - radius.y.ceil()).max(0.0) as i64;
        let max_y =
            (gaussian.center.y.ceil() + radius.y.ceil()).min(view.height as f32 - 1.0) as i64;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                visits += 1;
                budget.work.pixel_visits += 1;
                if visits > budget.options.max_pixel_visits_per_forward {
                    return Err(
                        "pixel visit budget exceeded; no omitted raster tail is permitted".into(),
                    );
                }
                if visits.is_multiple_of(4096) {
                    budget.check()?;
                }
                let delta = 2.0
                    * Vec2::new(
                        x as f32 + 0.5 - gaussian.center.x,
                        y as f32 + 0.5 - gaussian.center.y,
                    );
                if !gaussian.obb.contains_shader_delta(delta) {
                    continue;
                }
                let q = gaussian.inverse_covariance;
                let mahalanobis = q[0] * delta.x * delta.x
                    + 2.0 * q[1] * delta.x * delta.y
                    + q[2] * delta.y * delta.y;
                if !mahalanobis.is_finite() || mahalanobis < 0.0 {
                    continue;
                }
                let alpha = (gaussian.opacity
                    * gaussian_support_weight(mahalanobis, gaussian.cutoff_squared))
                .clamp(0.0, 0.999);
                let pixel_index = (y as u32 * view.width + x as u32) as usize;
                let pixel = &mut image[pixel_index];
                if record {
                    if tape.len() == max_entries {
                        return Err("contribution tape budget exceeded; no omitted compositing tail is permitted".into());
                    }
                    if tape.len() == tape.capacity() {
                        tape.try_reserve_exact((max_entries - tape.len()).min(4096))
                            .map_err(|_| "contribution tape allocation refused")?;
                        if tape.capacity() > max_entries {
                            return Err("allocator exceeded admitted tape capacity".into());
                        }
                        budget.work.peak_tape_bytes = budget
                            .work
                            .peak_tape_bytes
                            .max(tape.capacity() * size_of::<Contribution>());
                    }
                    tape.push(Contribution {
                        pixel: pixel_index as u32,
                        projected: index as u32,
                        alpha,
                        before: *pixel,
                    });
                }
                budget.work.contributions += 1;
                for (channel, color) in gaussian.color.into_iter().enumerate() {
                    pixel[channel] = color * alpha + pixel[channel] * (1.0 - alpha);
                }
                if gaussian.input_order < scene.owned.len() {
                    pixel[3] *= 1.0 - alpha;
                }
            }
        }
    }
    budget.check()?;
    Ok(Forward {
        image,
        projected,
        tape,
    })
}

fn loss(image: &[[f32; 4]], teacher: &[[f32; 4]], alpha_weight: f64) -> f64 {
    image
        .iter()
        .zip(teacher)
        .map(|(a, b)| {
            (0..4)
                .map(|c| {
                    let delta = f64::from(a[c]) - f64::from(b[c]);
                    delta * delta * if c == 3 { alpha_weight } else { 1.0 / 3.0 }
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / image.len() as f64
}

/// Non-overlapping half-open physical-pixel rectangle.
#[derive(Clone, Copy, Debug)]
struct PixelTile {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}
impl PixelTile {
    #[cfg(test)]
    fn full(view: ViewContext) -> Self {
        Self {
            x: 0,
            y: 0,
            width: view.width,
            height: view.height,
        }
    }
    fn area(self) -> usize {
        (self.width * self.height) as usize
    }
    fn global_index(self, local: usize, full_width: u32) -> usize {
        ((self.y + local as u32 / self.width) * full_width + self.x + local as u32 % self.width)
            as usize
    }
    fn local_index(self, global: u32, full_width: u32) -> usize {
        ((global / full_width - self.y) * self.width + global % full_width - self.x) as usize
    }
    fn intersect(self, other: Self) -> Self {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        Self {
            x,
            y,
            width: (self.x + self.width)
                .min(other.x + other.width)
                .saturating_sub(x),
            height: (self.y + self.height)
                .min(other.y + other.height)
                .saturating_sub(y),
        }
    }
    fn split(self) -> Option<[Self; 2]> {
        if self.width >= self.height && self.width > 1 {
            let half = self.width / 2;
            Some([
                Self {
                    width: half,
                    ..self
                },
                Self {
                    x: self.x + half,
                    width: self.width - half,
                    ..self
                },
            ])
        } else if self.height > 1 {
            let half = self.height / 2;
            Some([
                Self {
                    height: half,
                    ..self
                },
                Self {
                    y: self.y + half,
                    height: self.height - half,
                    ..self
                },
            ])
        } else {
            None
        }
    }
}

fn projected_pixel_bounds(
    gaussian: &ProductionProjectedGaussian,
    view: ViewContext,
) -> FitResult<PixelTile> {
    let radius = gaussian.obb.aabb_radius_pixels;
    if !radius.is_finite() {
        return Err("nonfinite support radius".into());
    }
    // Match the inclusive full-forward loop, represented as a clipped half-open rectangle.
    let min_x = (gaussian.center.x.floor() - radius.x.ceil()).clamp(0.0, view.width as f32) as u32;
    let min_y = (gaussian.center.y.floor() - radius.y.ceil()).clamp(0.0, view.height as f32) as u32;
    let end_x =
        (gaussian.center.x.ceil() + radius.x.ceil() + 1.0).clamp(0.0, view.width as f32) as u32;
    let end_y =
        (gaussian.center.y.ceil() + radius.y.ceil() + 1.0).clamp(0.0, view.height as f32) as u32;
    Ok(PixelTile {
        x: min_x,
        y: min_y,
        width: end_x.saturating_sub(min_x),
        height: end_y.saturating_sub(min_y),
    })
}

#[cfg_attr(not(test), allow(dead_code))]
struct TiledGradient {
    gradients: Vec<Gradient>,
    image: Vec<[f32; 4]>,
}

/// One projection/sort and one logical full-view forward; only the reverse
/// tape is tiled. Every pixel sees the same complete Gaussian depth order.
#[cfg(test)]
fn tiled_gradient(
    gaussians: &[Gaussian3d],
    teacher: &[[f32; 4]],
    view: ViewContext,
    color_space: GaussianColorSpace,
    budget: &mut Budget,
) -> FitResult<TiledGradient> {
    tiled_scene_gradient(
        FitScene::owned(gaussians),
        teacher,
        view,
        color_space,
        budget,
    )
}

fn tiled_scene_gradient(
    scene: FitScene<'_>,
    teacher: &[[f32; 4]],
    view: ViewContext,
    color_space: GaussianColorSpace,
    budget: &mut Budget,
) -> FitResult<TiledGradient> {
    budget.check()?;
    if teacher.len() != (view.width * view.height) as usize {
        return Err("teacher image size mismatch".into());
    }
    budget.work.forward_passes += 1;
    let projected = prepare_scene_projected(scene, view, color_space, budget)?;
    let bounds = projected
        .iter()
        .map(|g| projected_pixel_bounds(g, view))
        .collect::<FitResult<Vec<_>>>()?;
    let mut image = vec![[0.0, 0.0, 0.0, 1.0]; teacher.len()];
    let mut projected_gradient = vec![[0.0; 9]; projected.len()];
    let mut tape = Vec::<Contribution>::new();
    let max_entries = budget.options.max_tape_bytes / size_of::<Contribution>();
    let mut visits = 0_u64;
    // Depth-first subdivision keeps pending metadata bounded by log(tile area),
    // and a single tape allocation is reused across all accepted tiles.
    for y in (0..view.height).step_by(32) {
        for x in (0..view.width).step_by(32) {
            let mut pending = vec![PixelTile {
                x,
                y,
                width: (view.width - x).min(32),
                height: (view.height - y).min(32),
            }];
            while let Some(tile) = pending.pop() {
                budget.check()?;
                let mut upper_entries = 0_u64;
                for (index, bounds) in bounds.iter().enumerate() {
                    if index % 1024 == 0 {
                        budget.check()?;
                    }
                    budget.work.tile_bound_tests += 1;
                    upper_entries += tile.intersect(*bounds).area() as u64;
                }
                if upper_entries > max_entries as u64
                    && let Some([first, second]) = tile.split()
                {
                    budget.work.gradient_tile_splits += 1;
                    pending.push(second);
                    pending.push(first);
                    continue;
                }
                // A one-pixel tile may have a loose rectangular bound. Attempt its
                // actual OBB contributions; a real overflow still fails explicitly.
                tape.clear();
                let tile_capacity = upper_entries.min(max_entries as u64) as usize;
                if tape.capacity() < tile_capacity {
                    // Drop the empty smaller allocation before growing so two
                    // owned tape allocations never overlap at the byte ceiling.
                    drop(std::mem::take(&mut tape));
                    tape.try_reserve_exact(tile_capacity)
                        .map_err(|_| "contribution tape allocation refused")?;
                    if tape.capacity() > max_entries {
                        return Err("allocator exceeded admitted tape capacity".into());
                    }
                    budget.work.peak_tape_bytes = budget
                        .work
                        .peak_tape_bytes
                        .max(tape.capacity() * size_of::<Contribution>());
                }
                raster_gradient_tile(
                    &projected,
                    scene.owned.len(),
                    &bounds,
                    tile,
                    view,
                    &mut image,
                    &mut tape,
                    &mut visits,
                    budget,
                )?;
                accumulate_projected_gradient(
                    &projected,
                    scene.owned.len(),
                    &image,
                    &tape,
                    teacher,
                    view,
                    tile,
                    &mut projected_gradient,
                    budget,
                )?;
                budget.work.gradient_tiles += 1;
            }
        }
    }
    let gradients = chain_rule_gradients(
        scene.owned,
        &projected,
        &projected_gradient,
        view,
        color_space,
        budget,
    )?;
    budget.check()?;
    Ok(TiledGradient { gradients, image })
}

#[allow(clippy::too_many_arguments)]
fn raster_gradient_tile(
    projected: &[ProductionProjectedGaussian],
    owned_count: usize,
    bounds: &[PixelTile],
    tile: PixelTile,
    view: ViewContext,
    image: &mut [[f32; 4]],
    tape: &mut Vec<Contribution>,
    visits: &mut u64,
    budget: &mut Budget,
) -> FitResult<()> {
    let max_entries = budget.options.max_tape_bytes / size_of::<Contribution>();
    for (index, (gaussian, bounds)) in projected.iter().zip(bounds).enumerate() {
        if index % 1024 == 0 {
            budget.check()?;
        }
        let region = tile.intersect(*bounds);
        for y in region.y..region.y + region.height {
            for x in region.x..region.x + region.width {
                *visits += 1;
                budget.work.pixel_visits += 1;
                if *visits > budget.options.max_pixel_visits_per_forward {
                    return Err("pixel visit budget exceeded across gradient tiles; no omitted raster tail is permitted".into());
                }
                if visits.is_multiple_of(4096) {
                    budget.check()?;
                }
                let delta = 2.0
                    * Vec2::new(
                        x as f32 + 0.5 - gaussian.center.x,
                        y as f32 + 0.5 - gaussian.center.y,
                    );
                if !gaussian.obb.contains_shader_delta(delta) {
                    continue;
                }
                let q = gaussian.inverse_covariance;
                let mahalanobis = q[0] * delta.x * delta.x
                    + 2.0 * q[1] * delta.x * delta.y
                    + q[2] * delta.y * delta.y;
                if !mahalanobis.is_finite() || mahalanobis < 0.0 {
                    continue;
                }
                let alpha = (gaussian.opacity
                    * gaussian_support_weight(mahalanobis, gaussian.cutoff_squared))
                .clamp(0.0, 0.999);
                let pixel_index = y * view.width + x;
                let pixel = &mut image[pixel_index as usize];
                if tape.len() == max_entries {
                    return Err("contribution tape budget exceeded in one pixel; no omitted compositing tail is permitted".into());
                }
                // The clipped rectangular bound reserved enough entries before
                // this tile started; never permit implicit Vec growth here.
                if tape.len() == tape.capacity() {
                    return Err("tile contribution bound underestimated actual raster work".into());
                }
                tape.push(Contribution {
                    pixel: pixel_index,
                    projected: index as u32,
                    alpha,
                    before: *pixel,
                });
                budget.work.contributions += 1;
                for (channel, color) in gaussian.color.into_iter().enumerate() {
                    pixel[channel] = color * alpha + pixel[channel] * (1.0 - alpha);
                }
                if gaussian.input_order < owned_count {
                    pixel[3] *= 1.0 - alpha;
                }
            }
        }
    }
    Ok(())
}

// RGB color, pixel center xy, inverse covariance xx/xy/yy, peak opacity.
type ProjectedGradient = [f64; 9];

#[cfg(test)]
fn backward(
    gaussians: &[Gaussian3d],
    rendered: &Forward,
    teacher: &[[f32; 4]],
    view: ViewContext,
    color_space: GaussianColorSpace,
    budget: &mut Budget,
) -> FitResult<Vec<Gradient>> {
    let mut projected_gradient = vec![[0.0; 9]; rendered.projected.len()];
    accumulate_projected_gradient(
        &rendered.projected,
        gaussians.len(),
        &rendered.image,
        &rendered.tape,
        teacher,
        view,
        PixelTile::full(view),
        &mut projected_gradient,
        budget,
    )?;
    chain_rule_gradients(
        gaussians,
        &rendered.projected,
        &projected_gradient,
        view,
        color_space,
        budget,
    )
}

#[allow(clippy::too_many_arguments)]
fn accumulate_projected_gradient(
    projected: &[ProductionProjectedGaussian],
    owned_count: usize,
    image: &[[f32; 4]],
    tape: &[Contribution],
    teacher: &[[f32; 4]],
    view: ViewContext,
    tile: PixelTile,
    projected_gradient: &mut [ProjectedGradient],
    budget: &mut Budget,
) -> FitResult<()> {
    // Normalization is always the FULL view, regardless of tile dimensions.
    let count = teacher.len() as f64;
    let mut pixel_gradient: Vec<[f64; 4]> = (0..tile.area())
        .map(|index| {
            let pixel = tile.global_index(index, view.width);
            std::array::from_fn(|c| {
                2.0 * (f64::from(image[pixel][c]) - f64::from(teacher[pixel][c])) / count
                    * if c == 3 {
                        budget.options.alpha_weight
                    } else {
                        1.0 / 3.0
                    }
            })
        })
        .collect();
    for (tape_index, entry) in tape.iter().rev().enumerate() {
        if tape_index % 4096 == 0 {
            budget.check()?;
        }
        let gaussian = &projected[entry.projected as usize];
        let gp = &mut pixel_gradient[tile.local_index(entry.pixel, view.width)];
        let gradient = &mut projected_gradient[entry.projected as usize];
        let alpha = f64::from(entry.alpha);
        let owned = gaussian.input_order < owned_count;
        let mut alpha_gradient = if owned {
            -gp[3] * f64::from(entry.before[3])
        } else {
            0.0
        };
        for c in 0..3 {
            if owned {
                gradient[c] += alpha * gp[c];
                alpha_gradient += gp[c] * f64::from(gaussian.color[c] - entry.before[c]);
            }
        }
        if owned && entry.alpha > 0.0 && entry.alpha < 0.999 {
            let delta = 2.0
                * Vec2::new(
                    (entry.pixel % view.width) as f32 + 0.5 - gaussian.center.x,
                    (entry.pixel / view.width) as f32 + 0.5 - gaussian.center.y,
                );
            let [x, y] = delta.to_array().map(f64::from);
            let q = gaussian.inverse_covariance.map(f64::from);
            let mahalanobis = q[0] * x * x + 2.0 * q[1] * x * y + q[2] * y * y;
            let (_, density_derivative) = gaussian_support_weight_and_derivative(
                mahalanobis,
                f64::from(gaussian.cutoff_squared),
            );
            let gq = alpha_gradient * f64::from(gaussian.opacity) * density_derivative;
            // Center coordinates are physical pixels; the conic sees twice
            // their displacement. Use d(weight)/dq directly at the zero tail.
            gradient[3] -= 4.0 * gq * (q[0] * x + q[1] * y);
            gradient[4] -= 4.0 * gq * (q[1] * x + q[2] * y);
            gradient[5] += gq * x * x;
            gradient[6] += 2.0 * gq * x * y;
            gradient[7] += gq * y * y;
            gradient[8] += alpha_gradient * alpha / f64::from(gaussian.opacity);
        }
        // Frozen context still attenuates gradients to all records behind it.
        for value in &mut gp[..3] {
            *value *= 1.0 - alpha;
        }
        if owned {
            gp[3] *= 1.0 - alpha;
        }
    }
    Ok(())
}

fn chain_rule_gradients(
    gaussians: &[Gaussian3d],
    projected: &[ProductionProjectedGaussian],
    projected_gradient: &[ProjectedGradient],
    view: ViewContext,
    color_space: GaussianColorSpace,
    budget: &mut Budget,
) -> FitResult<Vec<Gradient>> {
    let mut gradients = vec![[0.0; PARAMS]; gaussians.len()];
    for (projected, gp) in projected.iter().zip(projected_gradient) {
        budget.check()?;
        if projected.input_order >= gaussians.len() {
            continue;
        }
        let source = &gaussians[projected.input_order];
        let geometry = view
            .geometry(source)
            .ok_or("projected Gaussian changed during backward")?;
        let direction = geometry.relative.normalize();
        let raw = production_spherical_harmonics_linear_color(
            direction,
            &source.spherical_harmonic,
            GaussianColorSpace::LinRec709Display,
        );
        let gradient = &mut gradients[projected.input_order];
        // Evaluating the linear SH basis through the shared helper keeps shader
        // signs/channel ordering authoritative without a second polynomial table.
        for (coefficient, value) in gradient
            .iter_mut()
            .take(budget.options.fitted_sh_coefficients())
            .enumerate()
        {
            let mut unit = SphericalHarmonicCoefficients::default();
            unit.coefficients[coefficient] = 1.0;
            let channel = coefficient % 3;
            let basis = production_spherical_harmonics_linear_color(
                direction,
                &unit,
                GaussianColorSpace::LinRec709Display,
            )[channel]
                - 0.5;
            let transfer = if color_space == GaussianColorSpace::SrgbRec709Display {
                srgb_derivative(raw[channel])
            } else {
                1.0
            };
            *value = gp[channel] * f64::from(basis) * transfer;
        }
        let opacity = f64::from(source.scale_opacity.opacity);
        if source.scale_opacity.opacity * geometry.opacity_scale < 1.0 {
            gradient[OPACITY] =
                gp[8] * f64::from(geometry.opacity_scale) * opacity * (1.0 - opacity);
        }
        if budget.options.fit_geometry {
            let mean_frame = match budget.options.mean_coordinates {
                MeanCoordinates::World => None,
                MeanCoordinates::RepresentativeLocal { .. } => Some(
                    *budget
                        .mean_frames
                        .get(projected.input_order)
                        .ok_or("missing frozen mean frame")?,
                ),
            };
            for parameter in 0..9 {
                // Dimensionless log-scale/rotation differences need a wider
                // step: subtracting f32 conics and near-unit Mip opacity
                // compensation at .001 amplifies quantization in weak,
                // cancelling gradients. .004 balances centered-difference
                // truncation against that cancellation; complete-image tests
                // check independent .002/.004/.008 steps without looser bounds.
                let epsilon = if parameter < 3 && mean_frame.is_none() {
                    0.0001
                } else {
                    0.004
                };
                let plus =
                    perturb_fitting_geometry(*source, parameter, epsilon, mean_frame.as_ref());
                let minus =
                    perturb_fitting_geometry(*source, parameter, -epsilon, mean_frame.as_ref());
                budget.work.local_projection_evaluations += 2;
                if let (Some(a), Some(b)) = (
                    projected_parameters(&plus, view, color_space),
                    projected_parameters(&minus, view, color_space),
                ) {
                    gradient[GEOMETRY + parameter] = gp
                        .iter()
                        .zip(a.iter().zip(b))
                        .map(|(g, (a, b))| {
                            g * (f64::from(*a) - f64::from(b)) / (2.0 * f64::from(epsilon))
                        })
                        .sum();
                }
            }
        }
    }
    if gradients.iter().flatten().any(|v| !v.is_finite()) {
        return Err("nonfinite fit gradient".into());
    }
    Ok(gradients)
}

fn srgb_derivative(value: f32) -> f64 {
    if value <= 0.04045 {
        1.0 / 12.92
    } else {
        (2.4 / 1.055) * ((f64::from(value) + 0.055) / 1.055).powf(1.4)
    }
}

fn projected_parameters(
    gaussian: &Gaussian3d,
    view: ViewContext,
    color_space: GaussianColorSpace,
) -> Option<[f32; 9]> {
    let geometry = view.geometry(gaussian)?;
    let color = production_spherical_harmonics_linear_color(
        geometry.relative.normalize(),
        &gaussian.spherical_harmonic,
        color_space,
    );
    Some([
        color[0],
        color[1],
        color[2],
        geometry.center.x,
        geometry.center.y,
        geometry.inverse_covariance[0],
        geometry.inverse_covariance[1],
        geometry.inverse_covariance[2],
        (gaussian.scale_opacity.opacity * geometry.opacity_scale).clamp(0.0, 1.0),
    ])
}

fn perturb_geometry(mut gaussian: Gaussian3d, parameter: usize, delta: f32) -> Gaussian3d {
    match parameter {
        0..=2 => gaussian.position_visibility.position[parameter] += delta,
        3..=5 => gaussian.scale_opacity.scale[parameter - 3] *= delta.exp(),
        _ => {
            let r = gaussian.rotation.rotation;
            let q = Quat::from_xyzw(r[1], r[2], r[3], r[0]);
            let rotation =
                (Quat::from_axis_angle([Vec3::X, Vec3::Y, Vec3::Z][parameter - 6], delta) * q)
                    .normalize();
            gaussian.rotation.rotation = [rotation.w, rotation.x, rotation.y, rotation.z];
        }
    }
    gaussian
}

fn perturb_fitting_geometry(
    mut gaussian: Gaussian3d,
    parameter: usize,
    delta: f32,
    frame: Option<&MeanFrame>,
) -> Gaussian3d {
    if parameter < 3
        && let Some(frame) = frame
    {
        gaussian.position_visibility.position =
            (Vec3::from_array(gaussian.position_visibility.position)
                + Vec3::from_array(frame[parameter]) * delta)
                .to_array();
        gaussian
    } else {
        perturb_geometry(gaussian, parameter, delta)
    }
}

/// Preserve each frozen representative's original owner domain and cardinality.
/// Bounds constrain complete authored three-sigma support, not merely the mean.
pub fn within_owner(gaussian: &Gaussian3d, bounds: LodBounds) -> bool {
    if bounds.validate().is_err()
        || gaussian
            .position_visibility
            .position
            .iter()
            .any(|v| !v.is_finite())
        || gaussian
            .scale_opacity
            .scale
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        || gaussian.rotation.rotation.iter().any(|v| !v.is_finite())
        || !gaussian.scale_opacity.opacity.is_finite()
        || !(0.0..=1.0).contains(&gaussian.scale_opacity.opacity)
        || gaussian
            .spherical_harmonic
            .coefficients
            .iter()
            .any(|v| !v.is_finite())
    {
        return false;
    }
    let covariance = compute_covariance_3d(
        Vec4::from_array(gaussian.rotation.rotation),
        Vec3::from_array(gaussian.scale_opacity.scale),
    );
    [covariance[0], covariance[3], covariance[5]]
        .into_iter()
        .enumerate()
        .all(|(axis, variance)| {
            let radius = 3.0 * variance.sqrt();
            let mean = gaussian.position_visibility.position[axis];
            let epsilon = 1e-5 * (bounds.max[axis] - bounds.min[axis]).abs().max(1.0);
            radius.is_finite()
                && mean - radius >= bounds.min[axis] - epsilon
                && mean + radius <= bounds.max[axis] + epsilon
        })
}

fn proposed_geometry(
    mut value: Gaussian3d,
    step: &Gradient,
    scale: f64,
    options: &FitOptions,
    mean_frame: Option<&MeanFrame>,
) -> Gaussian3d {
    let first = if let MeanCoordinates::RepresentativeLocal {
        trust_region_fraction,
    } = options.mean_coordinates
    {
        let norm = step[GEOMETRY]
            .hypot(step[GEOMETRY + 1])
            .hypot(step[GEOMETRY + 2]);
        let factor = if norm > f64::from(trust_region_fraction) {
            f64::from(trust_region_fraction) / norm
        } else {
            1.0
        };
        let local = Vec3::from_array(std::array::from_fn(|axis| {
            (-scale * factor * step[GEOMETRY + axis]) as f32
        }));
        value.position_visibility.position = (Vec3::from_array(value.position_visibility.position)
            + mean_displacement(mean_frame.expect("validated frozen mean frame"), local))
        .to_array();
        3
    } else {
        0
    };
    for parameter in first..9 {
        let limit = if parameter < 3 {
            f64::from(options.mean_step_limit)
        } else if parameter < 6 {
            0.02
        } else {
            0.01
        };
        let delta = (-scale * step[GEOMETRY + parameter].clamp(-limit, limit)) as f32;
        if delta != 0.0 {
            value = perturb_geometry(value, parameter, delta);
        }
    }
    value
}

#[cfg(test)]
fn proposal(
    current: &[Gaussian3d],
    directions: &[Gradient],
    domains: &[LodBounds],
    scale: f64,
    options: &FitOptions,
) -> Result<FitProposal, FitGeometryFeasibility> {
    proposal_with_frames(current, directions, domains, scale, options, &[])
}

fn proposal_with_frames(
    current: &[Gaussian3d],
    directions: &[Gradient],
    domains: &[LodBounds],
    scale: f64,
    options: &FitOptions,
    mean_frames: &[MeanFrame],
) -> Result<FitProposal, FitGeometryFeasibility> {
    let mut geometry = FitGeometryFeasibility::default();
    let mut gaussians = Vec::with_capacity(current.len());
    for (index, ((source, step), domain)) in current.iter().zip(directions).zip(domains).enumerate()
    {
        geometry.representatives_considered += 1;
        // Appearance always gets its ordinary proposal, independently of
        // whether this representative can move its support inside its owner.
        let mut appearance = *source;
        for (coefficient, direction) in appearance
            .spherical_harmonic
            .coefficients
            .iter_mut()
            .take(options.fitted_sh_coefficients())
            .zip(step.iter().take(SH_COEFF_COUNT))
        {
            *coefficient -= (scale * direction.clamp(-0.02, 0.02)) as f32;
        }
        if step[OPACITY] != 0.0 {
            let opacity = f64::from(appearance.scale_opacity.opacity).clamp(1e-6, 1.0 - 1e-6);
            let logit = (opacity / (1.0 - opacity)).ln() - scale * step[OPACITY].clamp(-0.1, 0.1);
            appearance.scale_opacity.opacity = (1.0 / (1.0 + (-logit).exp())) as f32;
        }
        let requested = options.fit_geometry && step[GEOMETRY..].iter().any(|value| *value != 0.0);
        geometry.requested_representatives += u32::from(requested);
        if !requested
            || options.geometry_feasibility == GeometryFeasibilityPolicy::RejectWholeProposal
        {
            let value = if requested {
                proposed_geometry(appearance, step, scale, options, mean_frames.get(index))
            } else {
                appearance
            };
            geometry.ownership_checks += 1;
            if !within_owner(&value, *domain) {
                return Err(geometry);
            }
            gaussians.push(value);
            continue;
        }
        let mut feasible = None;
        for attempt in 0..LOCAL_GEOMETRY_ATTEMPTS {
            // Every trial starts from the same appearance-only value. Never
            // compound failed geometry or alter the authenticated owner bounds.
            let value = proposed_geometry(
                appearance,
                step,
                scale * 0.5_f64.powi(attempt as i32),
                options,
                mean_frames.get(index),
            );
            geometry.ownership_checks += 1;
            if within_owner(&value, *domain) {
                geometry.limited_representatives += u32::from(attempt > 0);
                feasible = Some(value);
                break;
            }
        }
        if let Some(value) = feasible {
            gaussians.push(value);
        } else {
            geometry.ownership_checks += 1;
            if !within_owner(&appearance, *domain) {
                return Err(geometry);
            }
            geometry.frozen_representatives += 1;
            gaussians.push(appearance);
        }
    }
    Ok(FitProposal {
        gaussians,
        geometry,
    })
}

fn mean_loss(
    scene: FitScene<'_>,
    views: &[(ViewContext, Vec<[f32; 4]>)],
    color_space: GaussianColorSpace,
    budget: &mut Budget,
) -> FitResult<f64> {
    let mut total = 0.0;
    for (view, teacher) in views {
        let rendered = forward_scene(scene, *view, color_space, false, budget)?;
        total += loss(&rendered.image, teacher, budget.options.alpha_weight);
    }
    let result = total / views.len() as f64;
    if result.is_finite() {
        Ok(result)
    } else {
        Err("nonfinite fit loss".into())
    }
}

/// Optimize using training views only. Heldout images/losses are first generated
/// after all updates stop; they cannot change acceptance, learning rates or stopping.
/// Deadline is shared across teachers, fitting and heldout evaluation. Overflow
/// fails the whole call; a deadline returns the last fully accepted candidate.
pub fn fit_representatives(
    source: &[Gaussian3d],
    initial: &[Gaussian3d],
    domains: &[LodBounds],
    training: &[FitView],
    heldout: &[FitView],
    color_space: GaussianColorSpace,
    options: &FitOptions,
) -> FitResult<FitOutput> {
    fit_representatives_with_context(
        FitProblem {
            source,
            initial,
            context: &[],
            domains,
            source_order: None,
            initial_order: None,
            context_order: None,
        },
        training,
        heldout,
        color_space,
        options,
    )
}

fn validate_problem(
    problem: &FitProblem<'_>,
    training: &[FitView],
    heldout: &[FitView],
    options: &FitOptions,
) -> FitResult<()> {
    options.validate()?;
    let FitProblem {
        source,
        initial,
        context,
        domains,
        source_order,
        initial_order,
        context_order,
    } = *problem;
    if source.is_empty()
        || source
            .len()
            .checked_add(context.len())
            .is_none_or(|count| count > 100_000)
        || initial.is_empty()
        || initial
            .len()
            .checked_add(context.len())
            .is_none_or(|count| count > 30_000)
        || initial.len() != domains.len()
        || training.is_empty()
        || training.len() > MAX_FIT_VIEWS
        || heldout.len() > MAX_FIT_VIEWS
    {
        return Err(
            "fitter admits <=100k source+context, <=30k seed+context, <=16 train and <=16 heldout views"
                .into(),
        );
    }
    for (order, records) in [
        (source_order, source.len()),
        (initial_order, initial.len()),
        (context_order, context.len()),
    ] {
        if order.is_some_and(|order| order.len() != records) {
            return Err("fit stable-order length disagrees with its records".into());
        }
    }
    let has_order = source_order.is_some() || initial_order.is_some() || context_order.is_some();
    if has_order {
        let (Some(source_order), Some(initial_order)) = (source_order, initial_order) else {
            return Err("fit stable order must cover source, initial, and nonempty context".into());
        };
        if !context.is_empty() && context_order.is_none() {
            return Err("fit stable order must cover source, initial, and nonempty context".into());
        }
        // Validate teacher and candidate independently: substitution deliberately
        // reuses owned keys, while frozen context must never alias an owned key.
        let mut keys = Vec::with_capacity(source.len().max(initial.len()) + context.len());
        for owned_order in [source_order, initial_order] {
            keys.clear();
            keys.extend_from_slice(owned_order);
            keys.extend_from_slice(context_order.unwrap_or_default());
            keys.sort_unstable();
            if keys.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err("fit stable-order keys must be unique in each composed scene".into());
            }
        }
    }
    // This diagnostic subset excludes the flat radix visibility sentinel path.
    // Reject malformed records instead of silently removing their supervision.
    if source.iter().chain(initial).chain(context).any(|g| {
        !g.position_visibility.visibility.is_finite()
            || g.position_visibility.visibility <= 0.0
            || g.position_visibility
                .position
                .iter()
                .any(|v| !v.is_finite())
            || g.spherical_harmonic
                .coefficients
                .iter()
                .any(|v| !v.is_finite())
            || g.scale_opacity
                .scale
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || !g.scale_opacity.opacity.is_finite()
            || !(0.0..=1.0).contains(&g.scale_opacity.opacity)
            || g.rotation.rotation.iter().any(|v| !v.is_finite())
            || Vec4::from_array(g.rotation.rotation).length_squared() < 1e-12
    }) {
        return Err(
            "fit requires finite, positive-scale, positive-visibility source/candidate/context records"
                .into(),
        );
    }
    let mut ids = std::collections::BTreeSet::new();
    for view in training.iter().chain(heldout) {
        if view.id.is_empty() || !ids.insert(&view.id) {
            return Err("view IDs must be unique across frozen train/heldout".into());
        }
        ViewContext::with_reference_pixels(view.camera, options.reference_pixels)?;
    }
    if initial
        .iter()
        .zip(domains)
        .any(|(g, b)| !within_owner(g, *b))
    {
        return Err(
            "initial representative escapes its frozen owner bounds or is nonfinite".into(),
        );
    }
    Ok(())
}

/// Fit only the owned seed against composed RGB and owned transmittance.
/// Frozen context is reprojected in every fresh forward; it is never flattened
/// into a background image or included in optimizer moments/proposals.
pub fn fit_representatives_with_context(
    problem: FitProblem<'_>,
    training: &[FitView],
    heldout: &[FitView],
    color_space: GaussianColorSpace,
    options: &FitOptions,
) -> FitResult<FitOutput> {
    validate_problem(&problem, training, heldout, options)?;
    let FitProblem {
        source,
        initial,
        context,
        domains,
        source_order,
        initial_order,
        context_order,
    } = problem;
    let mut budget = Budget::new(options);
    budget.begin_training();
    budget.work.training_teacher_bytes =
        training_teacher_bytes(training.iter().map(|view| view.camera.viewport))?;
    budget.work.owned_source_records = source.len();
    budget.work.owned_candidate_records = initial.len();
    budget.work.immutable_context_records = context.len();
    budget.work.optimizer_moment_bytes = initial.len() * 2 * size_of::<Gradient>();
    budget.work.admitted_working_bytes = admit_working_arrays(
        &problem,
        budget.work.training_teacher_bytes,
        training
            .iter()
            .chain(heldout)
            .map(|view| (view.camera.viewport[0] * view.camera.viewport[1]) as usize)
            .max()
            .unwrap_or(0),
        options,
    )?;
    if matches!(
        options.mean_coordinates,
        MeanCoordinates::RepresentativeLocal { .. }
    ) {
        budget.mean_frames = initial.iter().map(initial_mean_frame).collect();
    }
    let teacher_scene = FitScene {
        owned: source,
        context,
        owned_order: source_order,
        context_order,
    };
    let mut teachers = Vec::new();
    let mut has_owned_foreground = false;
    for view in training {
        let view = ViewContext::with_reference_pixels(view.camera, options.reference_pixels)?;
        let image = forward_scene(teacher_scene, view, color_space, false, &mut budget)?.image;
        // Negative views constrain seed spill; immutable context cannot make
        // an otherwise empty owned training set a valid fitting objective.
        has_owned_foreground |= image.iter().any(|pixel| pixel[3] < 0.98);
        teachers.push((view, image));
    }
    if !has_owned_foreground {
        return Err(
            "training teachers have no owned foreground; refusing a blank fit objective".into(),
        );
    }
    let mut current = initial.to_vec();
    let initial_loss = mean_loss(
        FitScene {
            owned: &current,
            context,
            owned_order: initial_order,
            context_order,
        },
        &teachers,
        color_space,
        &mut budget,
    )?;
    let mut current_loss = initial_loss;
    let mut moments = vec![([0.0; PARAMS], [0.0; PARAMS]); initial.len()];
    let mut steps = Vec::new();
    let mut interrupted_step_trials = Vec::new();
    let mut stop_reason = "step_budget".to_owned();
    let mut stagnant_steps = 0;
    let mut ownership_stagnant_steps = 0;
    'steps: for step in 1..=options.max_steps {
        let mut gradient = vec![[0.0; PARAMS]; current.len()];
        for (view, teacher) in &teachers {
            let result = tiled_scene_gradient(
                FitScene {
                    owned: &current,
                    context,
                    owned_order: initial_order,
                    context_order,
                },
                teacher,
                *view,
                color_space,
                &mut budget,
            )
            .map(|result| result.gradients);
            let gradients = match result {
                Ok(value) => value,
                Err(error) if error == "wall_time_budget" => {
                    stop_reason = error;
                    break 'steps;
                }
                Err(error) => return Err(error),
            };
            for (total, sample) in gradient.iter_mut().zip(gradients) {
                for (a, b) in total.iter_mut().zip(sample) {
                    *a += b / teachers.len() as f64;
                }
            }
        }
        // Adam moments count complete training-gradient evaluations, including
        // rejected proposals; acceptance only changes the published parameters.
        for ((m, v), gradient) in moments.iter_mut().zip(&mut gradient) {
            for parameter in 0..PARAMS {
                m[parameter] = 0.9 * m[parameter] + 0.1 * gradient[parameter];
                v[parameter] = 0.999 * v[parameter] + 0.001 * gradient[parameter].powi(2);
                gradient[parameter] = options.learning_rate
                    * (m[parameter] / (1.0 - 0.9_f64.powi(step as i32)))
                    / ((v[parameter] / (1.0 - 0.999_f64.powi(step as i32))).sqrt() + 1e-8);
            }
        }
        let previous_loss = current_loss;
        let mut accepted_scale = None;
        let mut ownership_rejections = 0;
        let mut trials = Vec::with_capacity(4);
        for attempt in 0..4 {
            let scale = 0.5_f64.powi(attempt);
            let candidate = match proposal_with_frames(
                &current,
                &gradient,
                domains,
                scale,
                options,
                &budget.mean_frames,
            ) {
                Ok(candidate) => candidate,
                Err(geometry) => {
                    ownership_rejections += 1;
                    trials.push(FitTrial {
                        scale,
                        outcome: FitTrialOutcome::OwnershipRejected,
                        geometry,
                    });
                    continue;
                }
            };
            let mut trial = FitTrial {
                scale,
                outcome: FitTrialOutcome::TrainingLossRejected,
                geometry: candidate.geometry,
            };
            match mean_loss(
                FitScene {
                    owned: &candidate.gaussians,
                    context,
                    owned_order: initial_order,
                    context_order,
                },
                &teachers,
                color_space,
                &mut budget,
            ) {
                Ok(value) if value <= current_loss => {
                    current = candidate.gaussians;
                    current_loss = value;
                    accepted_scale = Some(scale);
                    trial.outcome = FitTrialOutcome::Accepted;
                    trials.push(trial);
                    break;
                }
                Ok(_) => trials.push(trial),
                Err(error) if error == "wall_time_budget" => {
                    trial.outcome = FitTrialOutcome::TrainingDeadline;
                    trials.push(trial);
                    interrupted_step_trials = trials;
                    stop_reason = error;
                    break 'steps;
                }
                Err(error) => return Err(error),
            }
        }
        steps.push(FitStep {
            step,
            loss: current_loss,
            accepted_scale,
            ownership_rejections,
            trials,
        });
        stagnant_steps = if current_loss < previous_loss {
            0
        } else {
            stagnant_steps + 1
        };
        ownership_stagnant_steps = if ownership_rejections == 4 {
            ownership_stagnant_steps + 1
        } else {
            0
        };
        if options.training_stagnation_patience > 0
            && stagnant_steps >= options.training_stagnation_patience
        {
            stop_reason = if ownership_stagnant_steps >= options.training_stagnation_patience {
                "ownership_stagnation"
            } else {
                "training_stagnation"
            }
            .into();
            break;
        }
    }
    if stop_reason == "wall_time_budget" && options.max_training_seconds > 0 {
        stop_reason = "training_time_budget".into();
    }
    // Heldout images are evaluated sequentially only after the optimizer stops;
    // their storage must not overlap the retained training teacher set.
    drop(teachers);
    budget.begin_evaluation();
    let mut heldout_losses = Vec::new();
    for view in heldout {
        let result = (|| {
            let view_context =
                ViewContext::with_reference_pixels(view.camera, options.reference_pixels)?;
            let teacher =
                forward_scene(teacher_scene, view_context, color_space, false, &mut budget)?.image;
            let before = forward_scene(
                FitScene {
                    owned: initial,
                    context,
                    owned_order: initial_order,
                    context_order,
                },
                view_context,
                color_space,
                false,
                &mut budget,
            )?
            .image;
            let after = forward_scene(
                FitScene {
                    owned: &current,
                    context,
                    owned_order: initial_order,
                    context_order,
                },
                view_context,
                color_space,
                false,
                &mut budget,
            )?
            .image;
            Ok::<_, String>(HeldoutLoss {
                id: view.id.clone(),
                initial_loss: loss(&before, &teacher, options.alpha_weight),
                final_loss: loss(&after, &teacher, options.alpha_weight),
            })
        })();
        match result {
            Ok(value) => heldout_losses.push(value),
            Err(error) if error == "wall_time_budget" => break,
            Err(error) => return Err(error),
        }
    }
    budget.work.elapsed_seconds = budget.start.elapsed().as_secs_f64();
    Ok(FitOutput {
        gaussians: current,
        report: FitReport {
            mean_coordinates: options.mean_coordinates,
            mean_coordinate_frames: budget.mean_frames,
            initial_training_loss: initial_loss,
            final_training_loss: current_loss,
            steps,
            interrupted_step_trials,
            stop_reason,
            heldout_complete: heldout_losses.len() == heldout.len(),
            heldout: heldout_losses,
            work: budget.work,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Gaussian3d, ViewContext) {
        let mut gaussian = Gaussian3d::default();
        gaussian.position_visibility.position = [0.13, -0.07, 0.0];
        gaussian.position_visibility.visibility = 1.0;
        gaussian.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
        gaussian.scale_opacity.scale = [0.23, 0.31, 0.17];
        gaussian.scale_opacity.opacity = 0.47;
        gaussian.spherical_harmonic.coefficients[0] = 0.35;
        gaussian.spherical_harmonic.coefficients[1] = -0.13;
        let view = ViewContext::new(LodTestCamera {
            world_rotation: None,
            position: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            projection: LodProjection::Perspective {
                vertical_fov_radians: 1.1,
            },
            near: 0.01,
            far: 100.0,
            viewport: [20, 16],
        })
        .unwrap();
        (gaussian, view)
    }

    #[test]
    fn framebuffer_obb_contains_the_covariance_major_axis() {
        let obb = production_obb([5.0, 3.0, 5.0], 3.0).unwrap();
        assert!(obb.contains_shader_delta(Vec2::new(5.0, 5.0)));
        assert!(!obb.contains_shader_delta(Vec2::new(-5.0, 5.0)));
    }

    #[test]
    fn taped_forward_matches_oracle_rgb_owned_transmittance_and_order() {
        let (a, view) = fixture();
        let mut b = a;
        b.position_visibility.position[2] = -0.17;
        b.spherical_harmonic.coefficients[1] = 0.8;
        let gaussians = [a, b];
        let color = GaussianColorSpace::SrgbRec709Display;
        let mut budget = Budget::new(&FitOptions::default());
        let actual = forward(&gaussians, view, color, true, &mut budget).unwrap();
        let expected = render_production_linear_gaussians_internal(
            gaussians.iter().map(|g| (None, g)),
            view.camera,
            view.width,
            view.height,
            false,
            LodOracleSupport::LodAuthoredThreeSigma,
            color,
        )
        .unwrap();
        for (actual, expected) in actual.image.iter().zip(&expected.rgba) {
            assert_eq!(actual[..3], expected[..3]);
            assert!((actual[3] - (1.0 - expected[3])).abs() < 2e-6);
        }
        assert!(!actual.tape.is_empty());
        assert_eq!(actual.projected[0].input_order, 1);
    }

    #[test]
    fn reference_pixel_grid_matches_native_forward_and_translation_gradient() {
        let (a, ordinary) = fixture();
        let pixels = FitReferencePixels {
            viewport: [80, 64],
            offset: [2, 1],
        };
        let view = ViewContext::with_reference_pixels(ordinary.camera, Some(pixels)).unwrap();
        let mut b = a;
        b.position_visibility.position = [-0.09, 0.12, -0.17];
        b.scale_opacity.scale = [0.04, 0.025, 0.02];
        b.spherical_harmonic.coefficients[1] = 0.8;
        let gaussians = [a, b];
        let options = FitOptions {
            fit_geometry: true,
            reference_pixels: Some(pixels),
            ..Default::default()
        };
        let color = GaussianColorSpace::SrgbRec709Display;
        let mut budget = Budget::new(&options);
        let sample_native = |gaussians: &[Gaussian3d]| {
            let camera = LodTestCamera {
                viewport: pixels.viewport,
                ..ordinary.camera
            };
            let rendered = render_production_linear_gaussians_internal(
                gaussians.iter().map(|gaussian| (None, gaussian)),
                camera,
                pixels.viewport[0],
                pixels.viewport[1],
                false,
                LodOracleSupport::LodAuthoredThreeSigma,
                color,
            )
            .unwrap();
            (0..view.height)
                .flat_map(|y| {
                    let rgba = &rendered.rgba;
                    (0..view.width).map(move |x| {
                        let mut pixel = rgba[((4 * y + pixels.offset[1]) * pixels.viewport[0]
                            + 4 * x
                            + pixels.offset[0])
                            as usize];
                        pixel[3] = 1.0 - pixel[3];
                        pixel
                    })
                })
                .collect::<Vec<_>>()
        };
        let actual = forward(&gaussians, view, color, true, &mut budget).unwrap();
        let expected = sample_native(&gaussians);
        assert_eq!(actual.image.len(), 20 * 16);
        for (sample, native) in actual.image.iter().flatten().zip(expected.iter().flatten()) {
            assert!((sample - native).abs() < 2e-6);
        }
        let low_resolution = forward(&gaussians, ordinary, color, false, &mut budget).unwrap();
        assert!(loss(&low_resolution.image, &expected, 1.0) > 1e-7);

        let teacher = vec![[0.0; 4]; expected.len()];
        let gradient = tiled_gradient(&gaussians, &teacher, view, color, &mut budget)
            .unwrap()
            .gradients[0][GEOMETRY];
        let epsilon = 0.0001;
        let mut plus = gaussians;
        let mut minus = gaussians;
        plus[0] = perturb_geometry(a, 0, epsilon);
        minus[0] = perturb_geometry(a, 0, -epsilon);
        let numerical = (loss(&sample_native(&plus), &teacher, 1.0)
            - loss(&sample_native(&minus), &teacher, 1.0))
            / (2.0 * f64::from(epsilon));
        assert!(
            (gradient - numerical).abs() < 2e-6 + numerical.abs() * 0.02,
            "sampled translation gradient {gradient} vs native finite difference {numerical}"
        );
        assert!(
            FitReferencePixels {
                offset: [4, 0],
                ..pixels
            }
            .sample_stride([20, 16])
            .is_err()
        );
        assert!(pixels.sample_stride([20, 15]).is_err());
    }

    #[test]
    fn immutable_context_preserves_shared_order_owned_transmittance_and_gradients() {
        let (base, view) = fixture();
        let make = |z: f32, color: f32| {
            let mut gaussian = base;
            gaussian.position_visibility.position[2] = z;
            gaussian.spherical_harmonic.coefficients[0] = color;
            gaussian
        };
        let owned = [make(0.2, 0.1), make(-0.2, 0.8)];
        // Owned[0] and context[1] have exactly equal depth. Their captured
        // order is opposite to the optimizer's owned-then-context storage.
        let context = [make(-0.4, 0.3), make(0.2, -0.2), make(0.4, 0.9)];
        let owned_order = [40, 20];
        let context_order = [10, 30, 50];
        let scene = FitScene {
            owned: &owned,
            context: &context,
            owned_order: Some(&owned_order),
            context_order: Some(&context_order),
        };
        let color = GaussianColorSpace::LinRec709Display;
        let mut budget = Budget::new(&FitOptions::default());
        let composed = forward_scene(scene, view, color, true, &mut budget).unwrap();
        assert_eq!(
            composed
                .projected
                .iter()
                .map(|g| g.input_order)
                .collect::<Vec<_>>(),
            [2, 1, 3, 0, 4]
        );
        let all = [context[0], owned[1], context[1], owned[0], context[2]];
        let complete = forward(&all, view, color, false, &mut budget).unwrap();
        let reordered = forward_scene(
            FitScene {
                owned_order: None,
                context_order: None,
                ..scene
            },
            view,
            color,
            false,
            &mut budget,
        )
        .unwrap();
        assert!(
            composed
                .image
                .iter()
                .zip(&reordered.image)
                .any(|(a, b)| a[..3] != b[..3])
        );
        let isolated = forward(&owned, view, color, false, &mut budget).unwrap();
        for ((actual, complete), isolated) in composed
            .image
            .iter()
            .zip(&complete.image)
            .zip(&isolated.image)
        {
            assert_eq!(actual[..3], complete[..3]);
            assert_eq!(actual[3], isolated[3]);
        }
        let teacher = vec![[0.12, 0.07, 0.02, 0.73]; composed.image.len()];
        let gradient = tiled_scene_gradient(scene, &teacher, view, color, &mut budget).unwrap();
        assert_eq!(gradient.gradients.len(), owned.len());
        assert_eq!(gradient.image, composed.image);
        for index in 0..owned.len() {
            for parameter in [0, OPACITY] {
                let epsilon = 0.002_f64;
                let mut plus = owned;
                let mut minus = owned;
                if parameter == OPACITY {
                    let opacity = f64::from(owned[index].scale_opacity.opacity);
                    let logit = (opacity / (1.0 - opacity)).ln();
                    plus[index].scale_opacity.opacity =
                        (1.0 / (1.0 + (-logit - epsilon).exp())) as f32;
                    minus[index].scale_opacity.opacity =
                        (1.0 / (1.0 + (-logit + epsilon).exp())) as f32;
                } else {
                    plus[index].spherical_harmonic.coefficients[parameter] += epsilon as f32;
                    minus[index].spherical_harmonic.coefficients[parameter] -= epsilon as f32;
                }
                let positive = forward_scene(
                    FitScene {
                        owned: &plus,
                        ..scene
                    },
                    view,
                    color,
                    false,
                    &mut budget,
                )
                .unwrap();
                let negative = forward_scene(
                    FitScene {
                        owned: &minus,
                        ..scene
                    },
                    view,
                    color,
                    false,
                    &mut budget,
                )
                .unwrap();
                let numerical = (loss(&positive.image, &teacher, 1.0)
                    - loss(&negative.image, &teacher, 1.0))
                    / (2.0 * epsilon);
                let actual = gradient.gradients[index][parameter];
                assert!(
                    (actual - numerical).abs() < 2e-6 + 0.002 * numerical.abs(),
                    "owned {index} parameter {parameter}: {actual} vs {numerical}"
                );
            }
        }
    }

    #[test]
    fn context_fit_keeps_optimizer_and_output_owned_and_rejects_unowned_foreground() {
        let (source, view) = fixture();
        let mut initial = source;
        initial.spherical_harmonic.coefficients[0] -= 0.1;
        let mut context = source;
        context.position_visibility.position[2] = 0.3;
        let source = [source];
        let initial = [initial];
        let context = [context];
        let domains = [LodBounds::new([-2.0; 3], [2.0; 3]).unwrap()];
        let views = [FitView {
            id: "train".into(),
            camera: view.camera,
        }];
        let options = FitOptions {
            max_steps: 1,
            ..Default::default()
        };
        let ordered_problem = FitProblem {
            source: &source,
            initial: &initial,
            context: &context,
            domains: &domains,
            source_order: Some(&[1]),
            initial_order: Some(&[1]),
            context_order: Some(&[0]),
        };
        validate_problem(&ordered_problem, &views, &[], &options).unwrap();
        for (invalid, expected) in [
            (
                FitProblem {
                    source_order: Some(&[]),
                    ..ordered_problem
                },
                "length",
            ),
            (
                FitProblem {
                    context_order: None,
                    ..ordered_problem
                },
                "must cover",
            ),
            (
                FitProblem {
                    context_order: Some(&[1]),
                    ..ordered_problem
                },
                "unique",
            ),
        ] {
            assert!(
                validate_problem(&invalid, &views, &[], &options)
                    .unwrap_err()
                    .contains(expected)
            );
        }
        let result = fit_representatives_with_context(
            ordered_problem,
            &views,
            &[],
            GaussianColorSpace::LinRec709Display,
            &options,
        )
        .unwrap();
        assert_eq!(result.gaussians.len(), initial.len());
        assert_eq!(result.report.work.immutable_context_records, 1);
        assert_eq!(
            result.report.work.optimizer_moment_bytes,
            2 * size_of::<Gradient>()
        );
        assert_eq!(result.report.work.training_teacher_bytes, 20 * 16 * 16);
        assert!(result.report.work.admitted_working_bytes <= MAX_FIT_WORKING_BYTES);
        assert!(result.report.final_training_loss <= result.report.initial_training_loss);
        let diagnostic = diagnose_representatives(
            FitProblem {
                source: &source,
                initial: &initial,
                context: &context,
                domains: &domains,
                source_order: None,
                initial_order: None,
                context_order: None,
            },
            &views,
            GaussianColorSpace::LinRec709Display,
            &options,
        )
        .unwrap();
        assert_eq!(diagnostic.work.forward_passes, 2);
        assert_eq!(diagnostic.work.optimizer_moment_bytes, 0);
        assert_eq!(diagnostic.work.training_teacher_bytes, 0);
        assert_eq!(diagnostic.work.diagnostic_image_bytes, 2 * 20 * 16 * 16);
        assert_eq!(diagnostic.views.len(), 1);
        assert_eq!(
            diagnostic.views[0].objective_loss,
            result.report.initial_training_loss
        );
        let mut empty_owned = source;
        empty_owned[0].scale_opacity.opacity = 0.0;
        let error = fit_representatives_with_context(
            FitProblem {
                source: &empty_owned,
                initial: &initial,
                context: &context,
                domains: &domains,
                source_order: None,
                initial_order: None,
                context_order: None,
            },
            &views,
            &[],
            GaussianColorSpace::LinRec709Display,
            &options,
        )
        .err()
        .unwrap();
        assert!(error.contains("no owned foreground"));
        let negative = diagnose_representatives(
            FitProblem {
                source: &empty_owned,
                initial: &initial,
                context: &context,
                domains: &domains,
                source_order: None,
                initial_order: None,
                context_order: None,
            },
            &views,
            GaussianColorSpace::LinRec709Display,
            &options,
        )
        .unwrap();
        assert!(negative.views[0].owned_transmittance_mse > 0.0);
        assert!(negative.views[0].source.iter().all(|pixel| pixel[3] == 1.0));
        assert!(
            negative.views[0]
                .source
                .iter()
                .any(|pixel| pixel[..3].iter().any(|value| *value > 0.0))
        );
        let blank = diagnose_representatives(
            FitProblem {
                source: &empty_owned,
                initial: &empty_owned,
                context: &context,
                domains: &domains,
                source_order: None,
                initial_order: None,
                context_order: None,
            },
            &views,
            GaussianColorSpace::LinRec709Display,
            &options,
        )
        .err()
        .unwrap();
        assert!(blank.contains("no owned foreground"));

        let training = [
            views[0].clone(),
            FitView {
                id: "negative".into(),
                camera: LodTestCamera {
                    target: Vec3::new(0.0, 0.0, 6.0),
                    ..view.camera
                },
            },
        ];
        let mixed = fit_representatives_with_context(
            FitProblem {
                source: &source,
                initial: &initial,
                context: &context,
                domains: &domains,
                source_order: None,
                initial_order: None,
                context_order: None,
            },
            &training,
            &[],
            GaussianColorSpace::LinRec709Display,
            &options,
        )
        .unwrap();
        assert_eq!(mixed.report.work.training_teacher_bytes, 2 * 20 * 16 * 16);
        assert_eq!(mixed.report.steps.len(), 1);
    }

    #[test]
    fn local_mean_coordinates_preserve_units_rotation_and_frozen_seed_metric() {
        let (mut gaussian, view) = fixture();
        let seed_rotation = Quat::from_rotation_z(0.6);
        gaussian.rotation.rotation = [
            seed_rotation.w,
            seed_rotation.x,
            seed_rotation.y,
            seed_rotation.z,
        ];
        let frame = initial_mean_frame(&gaussian);
        let options = FitOptions {
            fit_geometry: true,
            mean_coordinates: MeanCoordinates::RepresentativeLocal {
                trust_region_fraction: 0.05,
            },
            ..Default::default()
        };
        options.validate().unwrap();
        let mut step = [0.0; PARAMS];
        step[GEOMETRY..GEOMETRY + 3].copy_from_slice(&[0.3, -0.4, 0.2]);
        let moved = proposed_geometry(gaussian, &step, 1.0, &options, Some(&frame));
        let displacement = Vec3::from_array(moved.position_visibility.position)
            - Vec3::from_array(gaussian.position_visibility.position);
        let local = (seed_rotation.inverse() * displacement)
            / Vec3::from_array(gaussian.scale_opacity.scale);
        assert!((local.length() - 0.05).abs() < 1e-6);
        let mut changed = gaussian;
        changed.scale_opacity.scale = [4.0, 2.0, 3.0];
        changed.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(
            proposed_geometry(changed, &step, 1.0, &options, Some(&frame))
                .position_visibility
                .position,
            moved.position_visibility.position
        );

        let rotation = Quat::from_rotation_y(0.4);
        let units = 1_000.0;
        let position = |value: Vec3| rotation * value * units;
        let mut transformed = gaussian;
        transformed.position_visibility.position =
            position(Vec3::from_array(gaussian.position_visibility.position)).to_array();
        transformed.scale_opacity.scale = gaussian.scale_opacity.scale.map(|scale| scale * units);
        let orientation = rotation * seed_rotation;
        transformed.rotation.rotation =
            [orientation.w, orientation.x, orientation.y, orientation.z];
        let transformed_frame = initial_mean_frame(&transformed);
        let moved_transformed =
            proposed_geometry(transformed, &step, 1.0, &options, Some(&transformed_frame));
        assert!(
            (Vec3::from_array(moved_transformed.position_visibility.position)
                - position(Vec3::from_array(moved.position_visibility.position)))
            .length()
                < 2e-5
        );
        let transformed_view = ViewContext::new(LodTestCamera {
            position: position(view.camera.position),
            target: position(view.camera.target),
            up: rotation * view.camera.up,
            world_rotation: Some(rotation),
            near: view.camera.near * units,
            far: view.camera.far * units,
            ..view.camera
        })
        .unwrap();
        let gradient = |gaussian: Gaussian3d, view: ViewContext, frame: MeanFrame| {
            let mut budget = Budget::new(&options);
            budget.mean_frames.push(frame);
            let target = vec![[0.12, 0.07, 0.02, 0.73]; (view.width * view.height) as usize];
            tiled_gradient(
                &[gaussian],
                &target,
                view,
                GaussianColorSpace::LinRec709Display,
                &mut budget,
            )
            .unwrap()
            .gradients[0]
        };
        let before = gradient(gaussian, view, frame);
        let after = gradient(transformed, transformed_view, transformed_frame);
        for axis in 0..3 {
            let a = before[GEOMETRY + axis];
            let b = after[GEOMETRY + axis];
            assert!(
                (a - b).abs() < 2e-6 + a.abs() * 0.01,
                "local mean axis {axis}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn reverse_compositing_sh_and_opacity_match_full_forward_differences() {
        let (a, view) = fixture();
        let mut b = a;
        b.position_visibility.position[2] = -0.3;
        b.scale_opacity.opacity = 0.6;
        let gaussians = [a, b];
        let options = FitOptions::default();
        let mut budget = Budget::new(&options);
        let color = GaussianColorSpace::SrgbRec709Display;
        let teacher = vec![[0.12, 0.07, 0.02, 0.18]; (view.width * view.height) as usize];
        let rendered = forward(&gaussians, view, color, true, &mut budget).unwrap();
        let gradients =
            backward(&gaussians, &rendered, &teacher, view, color, &mut budget).unwrap();
        for owner in 0..2 {
            for parameter in [0, 1, OPACITY] {
                let epsilon = 0.002_f64;
                let mut a = gaussians;
                let mut b = gaussians;
                if parameter == OPACITY {
                    let o = f64::from(gaussians[owner].scale_opacity.opacity);
                    let logit = (o / (1.0 - o)).ln();
                    a[owner].scale_opacity.opacity =
                        (1.0 / (1.0 + (-logit - epsilon).exp())) as f32;
                    b[owner].scale_opacity.opacity =
                        (1.0 / (1.0 + (-logit + epsilon).exp())) as f32;
                } else {
                    a[owner].spherical_harmonic.coefficients[parameter] += epsilon as f32;
                    b[owner].spherical_harmonic.coefficients[parameter] -= epsilon as f32;
                }
                let positive = forward(&a, view, color, false, &mut budget).unwrap();
                let negative = forward(&b, view, color, false, &mut budget).unwrap();
                let numerical = (loss(&positive.image, &teacher, 1.0)
                    - loss(&negative.image, &teacher, 1.0))
                    / (2.0 * epsilon);
                let analytic = gradients[owner][parameter];
                assert!(
                    (analytic - numerical).abs() < 2e-6 + numerical.abs() * 0.002,
                    "owner {owner} parameter {parameter}: {analytic} vs {numerical}"
                );
            }
        }
    }

    #[test]
    fn local_geometry_jacobian_predicts_full_image_translation_derivative() {
        let (gaussian, view) = fixture();
        let options = FitOptions {
            fit_geometry: true,
            ..Default::default()
        };
        let mut budget = Budget::new(&options);
        let color = GaussianColorSpace::LinRec709Display;
        let teacher = vec![[0.0; 4]; (view.width * view.height) as usize];
        let rendered = forward(&[gaussian], view, color, true, &mut budget).unwrap();
        let gradient =
            backward(&[gaussian], &rendered, &teacher, view, color, &mut budget).unwrap();
        let epsilon = 0.0001;
        let plus = forward(
            &[perturb_geometry(gaussian, 0, epsilon)],
            view,
            color,
            false,
            &mut budget,
        )
        .unwrap();
        let minus = forward(
            &[perturb_geometry(gaussian, 0, -epsilon)],
            view,
            color,
            false,
            &mut budget,
        )
        .unwrap();
        // Chosen away from OBB boundary and depth-sort discontinuities.
        let numerical = (loss(&plus.image, &teacher, 1.0) - loss(&minus.image, &teacher, 1.0))
            / (2.0 * f64::from(epsilon));
        assert!((gradient[0][GEOMETRY] - numerical).abs() < 2e-5);
        assert_eq!(budget.work.local_projection_evaluations, 18);
    }

    #[test]
    fn all_nine_geometry_derivatives_match_independent_complete_image_differences() {
        let (mut front, _) = fixture();
        front.position_visibility.position = [0.13, -0.07, 0.1];
        front.scale_opacity.scale = [3.6, 2.8, 2.2];
        let rotation = Quat::from_scaled_axis(Vec3::new(0.37, -0.29, 0.23));
        front.rotation.rotation = [rotation.w, rotation.x, rotation.y, rotation.z];
        if let Some(coefficient) = front.spherical_harmonic.coefficients.get_mut(3) {
            *coefficient = 0.21;
        }
        if let Some(coefficient) = front.spherical_harmonic.coefficients.get_mut(7) {
            *coefficient = -0.18;
        }
        let mut back = front;
        back.position_visibility.position = [-0.35, 0.22, -0.75];
        back.scale_opacity.opacity = 0.63;
        back.spherical_harmonic.coefficients[0] = -0.27;
        back.spherical_harmonic.coefficients[1] = 0.41;
        let view = ViewContext::new(LodTestCamera {
            world_rotation: None,
            position: Vec3::new(1.3, 0.9, 5.7),
            target: Vec3::new(0.1, -0.15, 0.05),
            up: Vec3::new(0.1, 1.0, 0.05),
            projection: LodProjection::Perspective {
                vertical_fov_radians: 1.05,
            },
            near: 0.01,
            far: 100.0,
            viewport: [37, 31],
        })
        .unwrap();
        let gaussians = [front, back];
        let options = FitOptions {
            fit_geometry: true,
            ..Default::default()
        };
        let color = GaussianColorSpace::SrgbRec709Display;
        let mut budget = Budget::new(&options);
        let teacher = vec![[0.12, 0.07, 0.19, 0.24]; (view.width * view.height) as usize];
        let rendered = forward(&gaussians, view, color, true, &mut budget).unwrap();
        // Both anisotropic splats cover the entire viewport, away from cutoff
        // and clipping edges, and have distinct depth keys. Check this for all
        // perturbations rather than presuming the derivative is continuous.
        let membership = |render: &Forward| {
            render
                .tape
                .iter()
                .map(|entry| {
                    (
                        render.projected[entry.projected as usize].input_order,
                        entry.pixel,
                    )
                })
                .collect::<Vec<_>>()
        };
        let expected_membership = membership(&rendered);
        assert_eq!(expected_membership.len(), 2 * teacher.len());
        let gradients =
            backward(&gaussians, &rendered, &teacher, view, color, &mut budget).unwrap();
        let independent_image = |values: &[Gaussian3d]| {
            render_production_linear_gaussians_internal(
                values.iter().map(|gaussian| (None, gaussian)),
                view.camera,
                view.width,
                view.height,
                false,
                LodOracleSupport::LodAuthoredThreeSigma,
                color,
            )
            .unwrap()
            .rgba
            .into_iter()
            .map(|mut pixel| {
                pixel[3] = 1.0 - pixel[3];
                pixel
            })
            .collect::<Vec<_>>()
        };
        let mut mismatches = Vec::new();
        for owner in 0..gaussians.len() {
            for parameter in 0..9 {
                let perturb = |delta: f32| {
                    let mut values = gaussians;
                    let gaussian = &mut values[owner];
                    match parameter {
                        0..=2 => gaussian.position_visibility.position[parameter] += delta,
                        3..=5 => gaussian.scale_opacity.scale[parameter - 3] *= delta.exp(),
                        _ => {
                            let [w, x, y, z] = gaussian.rotation.rotation;
                            let mut axis = Vec3::ZERO;
                            axis[parameter - 6] = delta;
                            let q = (Quat::from_scaled_axis(axis) * Quat::from_xyzw(x, y, z, w))
                                .normalize();
                            gaussian.rotation.rotation = [q.w, q.x, q.y, q.z];
                        }
                    }
                    values
                };
                // Multiple independent full-image steps check convergence;
                // matching only the local Jacobian's step could conceal f32
                // cancellation shared by both sides of the comparison.
                for epsilon in [0.002_f32, 0.004, 0.008] {
                    let plus = perturb(epsilon);
                    let minus = perturb(-epsilon);
                    for values in [&plus, &minus] {
                        let check = forward(values, view, color, true, &mut budget).unwrap();
                        assert_eq!(
                            membership(&check),
                            expected_membership,
                            "owner {owner}, geometry {parameter}: support/order discontinuity"
                        );
                    }
                    let numerical = (loss(&independent_image(&plus), &teacher, 1.0)
                        - loss(&independent_image(&minus), &teacher, 1.0))
                        / (2.0 * f64::from(epsilon));
                    let actual = gradients[owner][GEOMETRY + parameter];
                    if numerical.abs() <= 1e-7 {
                        mismatches.push(format!("owner {owner}, geometry {parameter}, step {epsilon}: degenerate derivative witness"));
                    }
                    if (actual - numerical).abs() > 2e-6 + numerical.abs() * 0.01 {
                        mismatches.push(format!("owner {owner}, geometry {parameter}, step {epsilon}: backward {actual}, independent {numerical}"));
                        let mut projected_gradients = vec![[0.0; 9]; rendered.projected.len()];
                        accumulate_projected_gradient(
                            &rendered.projected,
                            gaussians.len(),
                            &rendered.image,
                            &rendered.tape,
                            &teacher,
                            view,
                            PixelTile::full(view),
                            &mut projected_gradients,
                            &mut budget,
                        )
                        .unwrap();
                        let index = rendered
                            .projected
                            .iter()
                            .position(|g| g.input_order == owner)
                            .unwrap();
                        for epsilon in [0.00025_f32, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016] {
                            let a = projected_parameters(
                                &perturb_geometry(gaussians[owner], parameter, epsilon),
                                view,
                                color,
                            )
                            .unwrap();
                            let b = projected_parameters(
                                &perturb_geometry(gaussians[owner], parameter, -epsilon),
                                view,
                                color,
                            )
                            .unwrap();
                            let contributions: Vec<f64> = projected_gradients[index]
                                .iter()
                                .zip(a.iter().zip(b))
                                .map(|(gradient, (a, b))| {
                                    gradient * (f64::from(*a) - f64::from(b))
                                        / (2.0 * f64::from(epsilon))
                                })
                                .collect();
                            let image_slope =
                                (loss(&independent_image(&perturb(epsilon)), &teacher, 1.0)
                                    - loss(&independent_image(&perturb(-epsilon)), &teacher, 1.0))
                                    / (2.0 * f64::from(epsilon));
                            eprintln!(
                                "geometry diagnostic owner={owner} parameter={parameter} epsilon={epsilon} local={} image={image_slope} terms={contributions:?} opacity_pair={:?}",
                                contributions.iter().sum::<f64>(),
                                [a[8], b[8]]
                            );
                        }
                    }
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "complete-image derivative mismatches:\n{}",
            mismatches.join("\n")
        );
    }

    #[test]
    fn broader_training_preserves_the_original_teacher_byte_ceiling() {
        assert_eq!(
            training_teacher_bytes([[512, 512]; 2]).unwrap(),
            MAX_TEACHER_BYTES
        );
        assert_eq!(training_teacher_bytes([[256, 144]; 12]).unwrap(), 7_077_888);
        assert!(
            training_teacher_bytes([[384, 216]; 12])
                .unwrap_err()
                .contains("8 MiB")
        );
        assert!(
            training_teacher_bytes([[u32::MAX, u32::MAX]])
                .unwrap_err()
                .contains("overflow")
        );
        assert!(
            training_teacher_bytes([[u32::MAX, 268_435_456]; 2])
                .unwrap_err()
                .contains("overflow")
        );
    }

    fn exact_support_owner(gaussian: &Gaussian3d) -> LodBounds {
        let covariance = compute_covariance_3d(
            Vec4::from_array(gaussian.rotation.rotation),
            Vec3::from_array(gaussian.scale_opacity.scale),
        );
        let radius = [covariance[0], covariance[3], covariance[5]].map(|v| 3.0 * v.sqrt());
        let center = gaussian.position_visibility.position;
        LodBounds::new(
            std::array::from_fn(|axis| center[axis] - radius[axis]),
            std::array::from_fn(|axis| center[axis] + radius[axis]),
        )
        .unwrap()
    }

    #[test]
    fn per_owner_feasibility_preserves_appearance_and_other_owners_geometry() {
        let (blocked, _) = fixture();
        let mut free = blocked;
        free.position_visibility.position[0] += 0.8;
        let initial = [blocked, free];
        let domains = [
            exact_support_owner(&blocked),
            LodBounds::new([-2.0; 3], [2.0; 3]).unwrap(),
        ];
        let mut directions = [[0.0; PARAMS]; 2];
        for step in &mut directions {
            step[0] = 0.01;
            step[OPACITY] = 0.05;
            step[GEOMETRY] = -0.1;
        }
        directions[1][GEOMETRY + 3] = -0.01;
        let whole_proposal = FitOptions {
            fit_geometry: true,
            ..Default::default()
        };
        assert_eq!(
            whole_proposal.geometry_feasibility,
            GeometryFeasibilityPolicy::RejectWholeProposal
        );
        assert!(proposal(&initial, &directions, &domains, 1.0, &whole_proposal).is_err());
        let bounded = FitOptions {
            geometry_feasibility: GeometryFeasibilityPolicy::PerRepresentativeBacktracking,
            ..whole_proposal
        };
        let result = proposal(&initial, &directions, &domains, 1.0, &bounded).unwrap();
        assert_eq!(result.gaussians.len(), initial.len());
        for (gaussian, domain) in result.gaussians.iter().zip(domains) {
            assert!(within_owner(gaussian, domain));
        }
        assert_eq!(
            result.gaussians[0].position_visibility,
            blocked.position_visibility
        );
        assert_eq!(result.gaussians[0].rotation, blocked.rotation);
        assert_eq!(
            result.gaussians[0].scale_opacity.scale,
            blocked.scale_opacity.scale
        );
        assert_ne!(
            result.gaussians[0].scale_opacity.opacity,
            blocked.scale_opacity.opacity
        );
        assert_ne!(
            result.gaussians[0].spherical_harmonic,
            blocked.spherical_harmonic
        );
        assert_ne!(
            result.gaussians[1].position_visibility,
            free.position_visibility
        );
        assert_ne!(
            result.gaussians[1].scale_opacity.scale,
            free.scale_opacity.scale
        );
        assert_eq!(result.geometry.requested_representatives, 2);
        assert_eq!(result.geometry.frozen_representatives, 1);
        assert_eq!(result.geometry.limited_representatives, 0);
        assert!(result.geometry.ownership_checks <= 2 * (LOCAL_GEOMETRY_ATTEMPTS + 1));
    }

    #[test]
    fn per_owner_backtracking_retains_a_feasible_fraction_without_expanding_bounds() {
        let (gaussian, _) = fixture();
        let mut domain = exact_support_owner(&gaussian);
        domain.max[0] += 0.0011;
        let original_domain = domain;
        let mut direction = [0.0; PARAMS];
        direction[GEOMETRY] = -0.1;
        let options = FitOptions {
            fit_geometry: true,
            geometry_feasibility: GeometryFeasibilityPolicy::PerRepresentativeBacktracking,
            ..Default::default()
        };
        let result = proposal(&[gaussian], &[direction], &[domain], 1.0, &options).unwrap();
        assert_eq!(domain, original_domain);
        assert!(within_owner(&result.gaussians[0], domain));
        let displacement = result.gaussians[0].position_visibility.position[0]
            - gaussian.position_visibility.position[0];
        assert!((displacement - 0.001).abs() < 1e-7);
        assert_eq!(result.geometry.limited_representatives, 1);
        assert_eq!(result.geometry.frozen_representatives, 0);
    }

    #[test]
    fn geometry_policy_defaults_preserve_whole_proposal_rejection_and_feasible_values() {
        let decoded: FitOptions = serde_json::from_str(r#"{"fit_geometry":true}"#).unwrap();
        assert_eq!(
            decoded.geometry_feasibility,
            GeometryFeasibilityPolicy::RejectWholeProposal
        );
        let (gaussian, _) = fixture();
        let mut direction = [0.0; PARAMS];
        direction[0] = 0.01;
        direction[GEOMETRY] = -0.1;
        let tight = exact_support_owner(&gaussian);
        assert!(proposal(&[gaussian], &[direction], &[tight], 1.0, &decoded).is_err());
        let loose = LodBounds::new([-2.0; 3], [2.0; 3]).unwrap();
        let whole_proposal = proposal(&[gaussian], &[direction], &[loose], 1.0, &decoded).unwrap();
        let bounded = proposal(
            &[gaussian],
            &[direction],
            &[loose],
            1.0,
            &FitOptions {
                geometry_feasibility: GeometryFeasibilityPolicy::PerRepresentativeBacktracking,
                ..decoded
            },
        )
        .unwrap();
        assert_eq!(whole_proposal.gaussians, bounded.gaussians);
        assert_eq!(whole_proposal.geometry.limited_representatives, 0);
        assert_eq!(whole_proposal.geometry.frozen_representatives, 0);
    }

    #[test]
    fn reported_geometry_counts_keep_accepted_trials_separate_from_rejections() {
        let (teacher, view) = fixture();
        let mut initial = teacher;
        initial.spherical_harmonic.coefficients[0] -= 0.3;
        let domains = [exact_support_owner(&initial)];
        let training = [FitView {
            id: "train".into(),
            camera: view.camera,
        }];
        let options = FitOptions {
            max_steps: 4,
            fit_geometry: true,
            geometry_feasibility: GeometryFeasibilityPolicy::PerRepresentativeBacktracking,
            ..Default::default()
        };
        let result = fit_representatives(
            &[teacher],
            &[initial],
            &domains,
            &training,
            &[],
            GaussianColorSpace::SrgbRec709Display,
            &options,
        )
        .unwrap();
        assert!(result.report.final_training_loss < result.report.initial_training_loss);
        assert!(result.report.interrupted_step_trials.is_empty());
        assert!(within_owner(&result.gaussians[0], domains[0]));
        for step in &result.report.steps {
            assert!(step.trials.len() <= 4);
            let accepted = step
                .trials
                .iter()
                .filter(|trial| trial.outcome == FitTrialOutcome::Accepted)
                .collect::<Vec<_>>();
            assert_eq!(accepted.len(), usize::from(step.accepted_scale.is_some()));
            assert_eq!(
                accepted.first().map(|trial| trial.scale),
                step.accepted_scale
            );
            assert_eq!(
                step.ownership_rejections as usize,
                step.trials
                    .iter()
                    .filter(|trial| trial.outcome == FitTrialOutcome::OwnershipRejected)
                    .count()
            );
            for trial in &step.trials {
                assert!(
                    trial.geometry.limited_representatives + trial.geometry.frozen_representatives
                        <= trial.geometry.requested_representatives
                );
                assert!(trial.geometry.representatives_considered <= 1);
                assert!(trial.geometry.ownership_checks <= LOCAL_GEOMETRY_ATTEMPTS + 1);
            }
        }
    }

    #[test]
    fn restricted_sh_fit_preserves_authored_higher_coefficients() {
        let (mut teacher, view) = fixture();
        for (index, coefficient) in teacher
            .spherical_harmonic
            .coefficients
            .iter_mut()
            .enumerate()
            .skip(3)
        {
            *coefficient = 0.001 * index as f32;
        }
        let mut initial = teacher;
        initial.spherical_harmonic.coefficients[0] -= 0.3;
        let options = FitOptions {
            max_steps: 4,
            max_fitted_sh_degree: 0,
            ..Default::default()
        };
        let result = fit_representatives(
            &[teacher],
            &[initial],
            &[exact_support_owner(&initial)],
            &[FitView {
                id: "train".into(),
                camera: view.camera,
            }],
            &[],
            GaussianColorSpace::SrgbRec709Display,
            &options,
        )
        .unwrap();
        assert!(result.report.final_training_loss < result.report.initial_training_loss);
        assert_ne!(
            result.gaussians[0].spherical_harmonic.coefficients[0],
            initial.spherical_harmonic.coefficients[0]
        );
        for (actual, expected) in result.gaussians[0]
            .spherical_harmonic
            .coefficients
            .iter()
            .zip(initial.spherical_harmonic.coefficients.iter())
            .skip(3)
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        // Even a supplied nonzero direction cannot update excluded coordinates.
        let proposed = proposal(
            &[initial],
            &[[0.01; PARAMS]],
            &[exact_support_owner(&initial)],
            1.0,
            &options,
        )
        .unwrap();
        for (actual, expected) in proposed.gaussians[0]
            .spherical_harmonic
            .coefficients
            .iter()
            .zip(initial.spherical_harmonic.coefficients.iter())
            .skip(3)
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        let mut invalid = options.clone();
        invalid.max_fitted_sh_degree = SH_DEGREE as u32 + 1;
        assert!(invalid.validate().is_err());
        let defaults: FitOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(defaults.max_fitted_sh_degree, SH_DEGREE as u32);
    }

    #[test]
    fn training_deadline_reserves_evaluation_time_without_extending_total_budget() {
        let options = FitOptions {
            max_seconds: 180,
            max_training_seconds: 120,
            ..Default::default()
        };
        options.validate().unwrap();
        let mut budget = Budget::new(&options);
        budget.begin_training();
        assert!(
            budget
                .check_at(budget.start + Duration::from_secs(119))
                .is_ok()
        );
        assert!(
            budget
                .check_at(budget.start + Duration::from_secs(120))
                .is_err()
        );
        budget.begin_evaluation();
        assert!(
            budget
                .check_at(budget.start + Duration::from_secs(120))
                .is_ok()
        );
        assert!(
            budget
                .check_at(budget.start + Duration::from_secs(180))
                .is_err()
        );
        let mut shared_deadline = Budget::new(&FitOptions::default());
        let original_deadline = shared_deadline.deadline;
        shared_deadline.begin_training();
        assert_eq!(shared_deadline.deadline, original_deadline);
        shared_deadline.begin_evaluation();
        assert_eq!(shared_deadline.deadline, original_deadline);
        assert!(
            FitOptions {
                max_training_seconds: 181,
                ..options
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn seventeenth_view_is_rejected_before_fitting() {
        let (gaussian, view) = fixture();
        let bounds = LodBounds::new([-2.0; 3], [2.0; 3]).unwrap();
        let views = vec![
            FitView {
                id: "unused".into(),
                camera: view.camera
            };
            17
        ];
        for (training, heldout) in [(&views[..], &views[..0]), (&views[..1], &views[..])] {
            let error = fit_representatives(
                &[gaussian],
                &[gaussian],
                &[bounds],
                training,
                heldout,
                GaussianColorSpace::SrgbRec709Display,
                &FitOptions::default(),
            )
            .err()
            .unwrap();
            assert!(error.contains("<=16 train and <=16 heldout"));
        }
    }

    #[test]
    fn training_stagnation_stops_without_consulting_heldout_views() {
        let (gaussian, view) = fixture();
        let bounds = LodBounds::new([-2.0; 3], [2.0; 3]).unwrap();
        let train = [FitView {
            id: "train".into(),
            camera: view.camera,
        }];
        let mut camera = view.camera;
        camera.position.x += 0.2;
        let heldout = [FitView {
            id: "heldout".into(),
            camera,
        }];
        let options = FitOptions {
            max_steps: 8,
            training_stagnation_patience: 2,
            ..Default::default()
        };
        let run = |heldout: &[FitView], options: &FitOptions| {
            fit_representatives(
                &[gaussian],
                &[gaussian],
                &[bounds],
                &train,
                heldout,
                GaussianColorSpace::SrgbRec709Display,
                options,
            )
            .unwrap()
        };
        let a = run(&[], &options);
        let b = run(&heldout, &options);
        assert_eq!(a.report.stop_reason, "training_stagnation");
        assert_eq!(a.report.steps.len(), 2);
        assert_eq!(a.gaussians, b.gaussians);
        assert_eq!(a.report.steps.len(), b.report.steps.len());
        assert_eq!(a.report.final_training_loss, b.report.final_training_loss);
        assert!(b.report.heldout_complete);
        let unchanged_default = run(
            &[],
            &FitOptions {
                training_stagnation_patience: 0,
                ..options
            },
        );
        assert_eq!(unchanged_default.report.steps.len(), 8);
        assert_eq!(unchanged_default.report.stop_reason, "step_budget");
    }

    fn tiled_fixture() -> (Vec<Gaussian3d>, ViewContext, Vec<[f32; 4]>) {
        let (mut a, view) = fixture();
        a.scale_opacity.scale = [0.72, 0.91, 0.17];
        let rotation = Quat::from_rotation_z(0.4);
        a.rotation.rotation = [rotation.w, rotation.x, rotation.y, rotation.z];
        let mut b = a;
        b.position_visibility.position[2] = -0.3;
        b.scale_opacity.opacity = 0.6;
        b.spherical_harmonic.coefficients[1] = 0.8;
        // Equal camera-depth keys retain input order even when a tile reverses only
        // its own contribution tape.
        let mut c = a;
        c.scale_opacity.opacity = 0.2;
        c.spherical_harmonic.coefficients[0] = -0.1;
        let mut clipped = a;
        clipped.position_visibility.position = [-1.7, 0.3, 0.0];
        clipped.scale_opacity.scale = [0.3, 0.45, 0.1];
        let view = ViewContext::new(LodTestCamera {
            world_rotation: None,
            viewport: [37, 35],
            ..view.camera
        })
        .unwrap();
        let teacher = (0..view.width * view.height)
            .map(|i| {
                let x = (i % view.width) as f32 / view.width as f32;
                let y = (i / view.width) as f32 / view.height as f32;
                [0.12 * x, 0.07 * y, 0.02, 0.18 + 0.1 * x]
            })
            .collect();
        (vec![a, b, c, clipped], view, teacher)
    }

    fn assert_gradients_equal(actual: &[Gradient], expected: &[Gradient]) {
        assert_eq!(actual.len(), expected.len());
        for (owner, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            for (parameter, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                // Tiling changes only the order of f64 reductions over pixels.
                assert!(
                    (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
                    "owner {owner}, parameter {parameter}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn tiled_reverse_matches_untiled_rgba_and_gradients_across_partial_tiles() {
        let (gaussians, view, teacher) = tiled_fixture();
        let options = FitOptions {
            fit_geometry: true,
            ..Default::default()
        };
        let color = GaussianColorSpace::SrgbRec709Display;
        let mut untiled_budget = Budget::new(&options);
        let rendered = forward(&gaussians, view, color, true, &mut untiled_budget).unwrap();
        let expected = backward(
            &gaussians,
            &rendered,
            &teacher,
            view,
            color,
            &mut untiled_budget,
        )
        .unwrap();
        let mut tiled_budget = Budget::new(&options);
        let actual = tiled_gradient(&gaussians, &teacher, view, color, &mut tiled_budget).unwrap();
        assert_eq!(actual.image, rendered.image);
        assert_gradients_equal(&actual.gradients, &expected);
        assert_eq!(tiled_budget.work.gradient_tiles, 4);
        assert_eq!(tiled_budget.work.gradient_tile_splits, 0);
        assert_eq!(tiled_budget.work.forward_passes, 1);
        assert_eq!(tiled_budget.work.projection_passes, 1);
        assert_eq!(
            tiled_budget.work.pixel_visits,
            untiled_budget.work.pixel_visits
        );
        assert_eq!(
            tiled_budget.work.contributions,
            untiled_budget.work.contributions
        );
        assert_eq!(
            tiled_budget.work.local_projection_evaluations,
            rendered.projected.len() as u64 * 18
        );
    }

    #[test]
    fn adaptive_tiles_preserve_complete_gradients_with_a_small_tape() {
        let (gaussians, view, teacher) = tiled_fixture();
        let options = FitOptions {
            // Intentionally not a multiple of the entry size: admitted
            // allocation must round down, never above the byte ceiling.
            max_tape_bytes: 64 * size_of::<Contribution>() + 3,
            fit_geometry: true,
            ..Default::default()
        };
        let color = GaussianColorSpace::LinRec709Display;
        let mut untiled_budget = Budget::new(&FitOptions {
            max_tape_bytes: MAX_TAPE_BYTES,
            ..options.clone()
        });
        let rendered = forward(&gaussians, view, color, true, &mut untiled_budget).unwrap();
        let expected = backward(
            &gaussians,
            &rendered,
            &teacher,
            view,
            color,
            &mut untiled_budget,
        )
        .unwrap();
        let mut tiled_budget = Budget::new(&options);
        let actual = tiled_gradient(&gaussians, &teacher, view, color, &mut tiled_budget).unwrap();
        assert_eq!(actual.image, rendered.image);
        assert_gradients_equal(&actual.gradients, &expected);
        assert!(tiled_budget.work.gradient_tiles > 4);
        assert!(tiled_budget.work.gradient_tile_splits > 0);
        assert!(tiled_budget.work.peak_tape_bytes > 0);
        assert!(tiled_budget.work.peak_tape_bytes <= options.max_tape_bytes);
        assert_eq!(tiled_budget.work.forward_passes, 1);
        assert_eq!(tiled_budget.work.projection_passes, 1);
        assert_eq!(
            tiled_budget.work.pixel_visits,
            untiled_budget.work.pixel_visits
        );
        assert_eq!(
            tiled_budget.work.contributions,
            untiled_budget.work.contributions
        );
        assert_eq!(
            tiled_budget.work.local_projection_evaluations,
            untiled_budget.work.local_projection_evaluations
        );
    }

    #[test]
    fn tiled_reverse_refuses_a_single_pixel_with_an_overfull_contribution_tape() {
        let (gaussian, view) = fixture();
        let options = FitOptions {
            max_tape_bytes: size_of::<Contribution>(),
            ..Default::default()
        };
        let mut budget = Budget::new(&options);
        let teacher = vec![[0.0; 4]; (view.width * view.height) as usize];
        let error = tiled_gradient(
            &[gaussian, gaussian],
            &teacher,
            view,
            GaussianColorSpace::LinRec709Display,
            &mut budget,
        )
        .err()
        .unwrap();
        assert!(error.contains("contribution tape budget exceeded in one pixel"));
        assert!(budget.work.gradient_tile_splits > 0);
        assert!(budget.work.peak_tape_bytes <= options.max_tape_bytes);
    }

    #[test]
    fn tiled_reverse_applies_visit_limit_to_the_complete_view() {
        let (gaussians, view, teacher) = tiled_fixture();
        let color = GaussianColorSpace::LinRec709Display;
        let mut untiled_budget = Budget::new(&FitOptions::default());
        forward(&gaussians, view, color, false, &mut untiled_budget).unwrap();
        let options = FitOptions {
            max_pixel_visits_per_forward: untiled_budget.work.pixel_visits - 1,
            max_tape_bytes: 64 * size_of::<Contribution>(),
            ..Default::default()
        };
        let mut budget = Budget::new(&options);
        let error = tiled_gradient(&gaussians, &teacher, view, color, &mut budget)
            .err()
            .unwrap();
        assert!(error.contains("pixel visit budget exceeded across gradient tiles"));
        assert!(budget.work.gradient_tiles > 1);
        assert_eq!(budget.work.forward_passes, 1);
        assert_eq!(
            budget.work.pixel_visits,
            options.max_pixel_visits_per_forward + 1
        );
        assert!(budget.work.peak_tape_bytes <= options.max_tape_bytes);
    }

    #[test]
    fn tape_overflow_fails_instead_of_dropping_contributions() {
        let (gaussian, view) = fixture();
        let options = FitOptions {
            max_tape_bytes: size_of::<Contribution>(),
            ..Default::default()
        };
        let mut budget = Budget::new(&options);
        let error = forward(
            &[gaussian],
            view,
            GaussianColorSpace::LinRec709Display,
            true,
            &mut budget,
        )
        .err()
        .unwrap();
        assert!(error.contains("contribution tape budget exceeded"));
        assert!(budget.work.peak_tape_bytes <= options.max_tape_bytes);
    }

    #[test]
    fn fitter_improves_training_without_changing_cardinality_or_owner() {
        let (source, view) = fixture();
        let mut initial = source;
        initial.spherical_harmonic.coefficients[0] -= 0.3;
        initial.scale_opacity.opacity -= 0.1;
        let bounds = LodBounds::new([-2.0; 3], [2.0; 3]).unwrap();
        let train = [FitView {
            id: "train".into(),
            camera: view.camera,
        }];
        let options = FitOptions {
            max_steps: 8,
            ..Default::default()
        };
        let output = fit_representatives(
            &[source],
            &[initial],
            &[bounds],
            &train,
            &[],
            GaussianColorSpace::SrgbRec709Display,
            &options,
        )
        .unwrap();
        assert_eq!(output.gaussians.len(), 1);
        assert!(within_owner(&output.gaussians[0], bounds));
        assert!(output.report.final_training_loss < output.report.initial_training_loss);
        assert_eq!(
            output.gaussians[0].position_visibility,
            initial.position_visibility
        );
        assert_eq!(output.gaussians[0].rotation, initial.rotation);
        assert_eq!(
            output.gaussians[0].scale_opacity.scale,
            initial.scale_opacity.scale
        );
        let mut escaped = output.gaussians[0];
        escaped.scale_opacity.scale = [10.0; 3];
        assert!(!within_owner(&escaped, bounds));
    }

    #[test]
    fn heldout_views_cannot_change_any_optimizer_update() {
        let (source, view) = fixture();
        let mut initial = source;
        initial.spherical_harmonic.coefficients[0] -= 0.3;
        let bounds = LodBounds::new([-2.0; 3], [2.0; 3]).unwrap();
        let train = [FitView {
            id: "train".into(),
            camera: view.camera,
        }];
        let mut held_camera = view.camera;
        held_camera.position.x = 0.7;
        let heldout = [FitView {
            id: "heldout".into(),
            camera: held_camera,
        }];
        let options = FitOptions {
            max_steps: 2,
            ..Default::default()
        };
        let a = fit_representatives(
            &[source],
            &[initial],
            &[bounds],
            &train,
            &[],
            GaussianColorSpace::SrgbRec709Display,
            &options,
        )
        .unwrap();
        let b = fit_representatives(
            &[source],
            &[initial],
            &[bounds],
            &train,
            &heldout,
            GaussianColorSpace::SrgbRec709Display,
            &options,
        )
        .unwrap();
        assert_eq!(a.gaussians, b.gaussians);
        assert_eq!(a.report.final_training_loss, b.report.final_training_loss);
        assert!(b.report.heldout_complete);
    }
}
