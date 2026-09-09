//! Authenticated local-cohort fitting through the shared optimizer/publication path.

use super::*;
use crate::io::ply::{PlyShCompatibility, stream_ply_3d_with_sh_compatibility};
use crate::{
    camera::path::GaussianCameraPath,
    testing::{
        lod_scenes::LodPixelCrop,
        render_oracle::fit::{
            FitDiagnosticOutput, FitProblem, compare_representatives, diagnose_representatives,
            fit_representatives_with_context,
        },
    },
};
use std::io::BufRead;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ContextRepresentation {
    #[default]
    Selected,
    Original,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct PlyRoundtripPolicy {
    /// Reserved from options.max_seconds, never added to the fitting ceiling.
    reserve_seconds: u32,
    max_rgb_absolute_error: f64,
    max_transmittance_absolute_error: f64,
}

impl Default for PlyRoundtripPolicy {
    fn default() -> Self {
        Self {
            reserve_seconds: 10,
            max_rgb_absolute_error: 1e-4,
            max_transmittance_absolute_error: 1e-4,
        }
    }
}

impl PlyRoundtripPolicy {
    fn validate(&self, total_seconds: u32) -> CaptureResult<()> {
        if self.reserve_seconds == 0
            || self.reserve_seconds >= total_seconds
            || [
                self.max_rgb_absolute_error,
                self.max_transmittance_absolute_error,
            ]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0 || value > 0.01)
        {
            return Err("PLY roundtrip requires a reserve within max_seconds and positive image tolerances <=0.01".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CalibratedCamera {
    id: String,
    camera_path: PinnedFile,
    frame_index: usize,
    /// Physical viewport at which the imported calibration is defined.
    calibration_viewport: [u32; 2],
    #[serde(default)]
    crop: Option<LodPixelCrop>,
    near: f32,
    far: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CohortConfig {
    cohort_sidecar: PinnedFile,
    #[serde(default)]
    context_representation: ContextRepresentation,
    /// Evaluate unchanged source/seed images without allocating optimizer state.
    #[serde(default)]
    diagnostic_only: bool,
    /// Optional fixed-cardinality deployment artifact for evaluation or fitting.
    #[serde(default)]
    candidate_ply: Option<PinnedFile>,
    #[serde(default)]
    ply_roundtrip: PlyRoundtripPolicy,
    output_directory: PathBuf,
    /// Allocated RGB/transmittance grid, independent of physical crop size.
    viewport: [u32; 2],
    training: Vec<CalibratedCamera>,
    heldout: Vec<CalibratedCamera>,
    #[serde(default)]
    options: FitOptions,
}

impl CohortConfig {
    fn validate(&self) -> CaptureResult<()> {
        self.options.validate()?;
        if !self.diagnostic_only {
            self.ply_roundtrip.validate(self.options.max_seconds)?;
        }
        if self.viewport.contains(&0)
            || self.viewport.iter().any(|&size| size > 512)
            || self.training.is_empty()
            || (self.heldout.is_empty() && !self.diagnostic_only)
            || self.training.len() > MAX_FIT_VIEWS
            || self.heldout.len() > MAX_FIT_VIEWS
        {
            return Err(
                "cohort fit requires <=512x512 targets and 1..=16 train/heldout views".into(),
            );
        }
        if self.diagnostic_only && self.training.len() + self.heldout.len() > MAX_FIT_VIEWS {
            return Err("unchanged-cohort diagnostic admits at most 16 total views".into());
        }
        training_teacher_bytes(self.training.iter().map(|_| self.viewport))?;
        if let Some(pixels) = self.options.reference_pixels {
            pixels.sample_stride(self.viewport)?;
        }
        let mut ids = BTreeSet::new();
        let training_frames = self
            .training
            .iter()
            .map(|camera| (camera.camera_path.sha256.as_str(), camera.frame_index))
            .collect::<BTreeSet<_>>();
        for (heldout, camera) in self
            .training
            .iter()
            .map(|camera| (false, camera))
            .chain(self.heldout.iter().map(|camera| (true, camera)))
        {
            if camera.id.is_empty() || !ids.insert(&camera.id) {
                return Err("train/heldout camera IDs overlap".into());
            }
            if heldout
                && training_frames
                    .contains(&(camera.camera_path.sha256.as_str(), camera.frame_index))
            {
                return Err(
                    "train/heldout reuse a physical camera frame, including a different crop"
                        .into(),
                );
            }
            if camera.calibration_viewport.contains(&0)
                || camera.calibration_viewport.iter().any(|&size| size > 8192)
                || !camera.near.is_finite()
                || camera.near <= 0.0
                || !camera.far.is_finite()
                || camera.far <= camera.near
            {
                return Err("invalid calibrated fit viewport or clipping interval".into());
            }
            if let Some(crop) = camera.crop {
                crop.validate(camera.calibration_viewport)?;
                let projected_viewport = self
                    .options
                    .reference_pixels
                    .map_or(self.viewport, |pixels| pixels.viewport);
                if crop.size != projected_viewport {
                    return Err("physical crop size must match the reference pixel viewport".into());
                }
            }
        }
        Ok(())
    }
}

fn views(config: &CohortConfig) -> CaptureResult<(Vec<FitView>, Vec<FitView>)> {
    let mut paths = std::collections::BTreeMap::new();
    let mut result = Vec::<FitView>::new();
    for camera in config.training.iter().chain(&config.heldout) {
        check_file(&camera.camera_path, 4 * 1024 * 1024)?;
        let key = (
            camera.camera_path.path.clone(),
            camera.camera_path.sha256.clone(),
        );
        if !paths.contains_key(&key) {
            let path = GaussianCameraPath::from_json(&fs::read(&camera.camera_path.path)?)?;
            paths.insert(key.clone(), path);
        }
        let frame = paths[&key]
            .frames()
            .get(camera.frame_index)
            .ok_or("calibrated fit frame index is outside the pinned camera path")?;
        let mut view = LodTestCamera::from_camera_path_frame(
            frame,
            camera.calibration_viewport,
            camera.near,
            camera.far,
        )?;
        if let Some(crop) = camera.crop {
            view = view.with_crop(crop)?;
        }
        view.viewport = config.viewport;
        if result.iter().any(|old| old.camera == view) {
            return Err("duplicate train/heldout calibrated physical view".into());
        }
        result.push(FitView {
            id: camera.id.clone(),
            camera: view,
        });
    }
    let heldout = result.split_off(config.training.len());
    Ok((result, heldout))
}

pub(super) fn fit_cohort(config_bytes: &[u8], started: Instant) -> CaptureResult<()> {
    let config: CohortConfig = serde_json::from_slice(config_bytes)?;
    config.validate()?;
    if config.output_directory.exists() {
        return Err("fit output directory already exists; refusing overwrite".into());
    }
    let (training, heldout) = views(&config)?;
    let input = super::super::cohort::load_cohort(
        &config.cohort_sidecar.path,
        &config.cohort_sidecar.sha256,
        matches!(
            config.context_representation,
            ContextRepresentation::Original
        ),
    )?;
    input.validate_views(&training)?;
    input.validate_views(&heldout)?;
    let candidate_byte_limit = input.initial.len() as u64 * 1024 + 64 * 1024;
    let candidate = config
        .candidate_ply
        .as_ref()
        .map(|pinned| -> CaptureResult<_> {
            check_file(pinned, candidate_byte_limit)?;
            let loaded = reload_candidate(
                &mut std::io::BufReader::new(fs::File::open(&pinned.path)?),
                input.initial.len(),
            )?;
            check_file(pinned, candidate_byte_limit)?;
            Ok(loaded)
        })
        .transpose()?;
    let admission_seconds = started.elapsed().as_secs_f64();
    if config.diagnostic_only {
        let all_views = training.into_iter().chain(heldout).collect::<Vec<_>>();
        // Both-blank output is a valid result when evaluating a fitted artifact.
        // The unchanged-seed P0 diagnostic keeps its owned-foreground guard.
        let evaluate = if candidate.is_some() {
            compare_representatives
        } else {
            diagnose_representatives
        };
        let diagnostic = evaluate(
            FitProblem {
                source: &input.source,
                initial: candidate.as_deref().unwrap_or(&input.initial),
                context: &input.context,
                domains: &input.domains,
                source_order: input.source_order.as_deref(),
                initial_order: input.initial_order.as_deref(),
                context_order: input.context_order.as_deref(),
            },
            &all_views,
            input.color_space,
            &config.options,
        )?;
        input.verify_inputs()?;
        verify_cameras(&config)?;
        if let Some(pinned) = &config.candidate_ply {
            check_file(pinned, candidate_byte_limit)?;
        }
        return publish_diagnostic(
            &config,
            config_bytes,
            &input,
            &diagnostic,
            admission_seconds,
            candidate.as_deref(),
        );
    }
    let mut optimization_options = config.options.clone();
    optimization_options.max_seconds -= config.ply_roundtrip.reserve_seconds;
    optimization_options.max_training_seconds = optimization_options
        .max_training_seconds
        .min(optimization_options.max_seconds);
    let optimization_started = Instant::now();
    let result = fit_representatives_with_context(
        FitProblem {
            source: &input.source,
            initial: candidate.as_deref().unwrap_or(&input.initial),
            context: &input.context,
            domains: &input.domains,
            source_order: input.source_order.as_deref(),
            initial_order: input.initial_order.as_deref(),
            context_order: input.context_order.as_deref(),
        },
        &training,
        &heldout,
        input.color_space,
        &optimization_options,
    )?;
    let optimization_seconds = optimization_started.elapsed().as_secs_f64();
    let verification_started = Instant::now();
    input.verify_inputs()?;
    verify_cameras(&config)?;
    if let Some(pinned) = &config.candidate_ply {
        check_file(pinned, candidate_byte_limit)?;
    }
    let input_verification_seconds = verification_started.elapsed().as_secs_f64();
    let report = json!({
        "schema_version": 1,
        "kind": "diagnostic_cpu_oracle_cohort_fit",
        "gpu_forward_parity_qualified": false,
        "quality_certificate": false,
        "package_or_error_policy_modified": false,
        "configuration": config, "optimization_options": optimization_options,
        "config_sha256": format!("{:x}", Sha256::digest(config_bytes)),
        "fitter_executable_sha256": hash_file(&std::env::current_exe()?)?,
        "cohort_lineage": input.sidecar,
        "fixed_cardinality": result.gaussians.len(),
        "output_order": "unchanged owned seed indices; context is immutable and omitted from fitted.ply",
        "ownership": "every emitted full three-sigma support remains within its authenticated owner bounds",
        "objective": "mean train linear composed RGB MSE + alpha_weight * owned-only transmittance MSE; targets contain four f32 values per pixel",
        "context": "complete shared forward camera-depth order; context attenuates RGB gradients but has no optimizer state or owned-transmittance contribution",
        "optimizer_values": "authenticated native source/context; initial owned records use the declared native seed or SHA-verified deployment candidate",
        "projection": "pinned calibrated fx/fy and exact imported rotation; physical crops and reference pixel grids preserve native Mip covariance/filter",
        "support": "shared smooth finite three-sigma support for native original leaves, seed and context; identity transform, sRGB display SH, linear premultiplied RGB, transparent black background, no tonemap; adaptive flat-teacher equivalence is not claimed",
        "heldout": "no heldout-driven optimizer updates, candidate selection or stopping; artifact fidelity is checked on every configured view after fitting",
        "admission_seconds": admission_seconds, "postfit_input_verification_seconds": input_verification_seconds,
        "fit": result.report,
        "limits": [
            "100k owned-source plus context, 30k owned-seed plus context; metadata/page authentication is admitted separately",
            "8 MiB retained RGB/transmittance targets, 128 MiB tape and 256 MiB fit working arrays; borrowed inputs and process RSS are separate",
            "context scope is the authenticated export selection, not a claim of complete full-scene supervision",
            "deadline publishes only the last fully accepted candidate; overflow fails without partial output"
        ]
    });
    publish_checked_fit(
        &config,
        config_bytes,
        &input,
        &result.gaussians,
        &training,
        &heldout,
        optimization_seconds,
        report,
    )
}

fn verify_cameras(config: &CohortConfig) -> CaptureResult<()> {
    for camera in config.training.iter().chain(&config.heldout) {
        check_file(&camera.camera_path, 4 * 1024 * 1024)?;
    }
    Ok(())
}

/// Uses the same normalization/SH checks as deployment PLY loading, with a
/// fixed record capacity instead of constructing and copying a planar cloud.
fn reload_candidate(
    reader: &mut dyn BufRead,
    count: usize,
) -> CaptureResult<Vec<crate::Gaussian3d>> {
    let mut records = Vec::new();
    records.try_reserve_exact(count)?;
    stream_ply_3d_with_sh_compatibility(
        reader,
        1024,
        PlyShCompatibility::RequireRepresentable,
        |batch| {
            if records.len() + batch.len() > count {
                return Err(std::io::Error::other(
                    "candidate PLY exceeds fixed cardinality",
                ));
            }
            records.extend_from_slice(batch);
            Ok(())
        },
    )?;
    if records.len() != count {
        return Err("candidate PLY changed fixed cardinality".into());
    }
    Ok(records)
}

fn maximum_image_drift(reference: &[[f32; 4]], reloaded: &[[f32; 4]]) -> [f64; 2] {
    reference
        .iter()
        .zip(reloaded)
        .fold([0.0_f64; 2], |mut maximum, (a, b)| {
            for channel in 0..4 {
                let kind = usize::from(channel == 3);
                maximum[kind] =
                    maximum[kind].max((f64::from(a[channel]) - f64::from(b[channel])).abs());
            }
            maximum
        })
}

#[allow(clippy::too_many_arguments)]
fn publish_checked_fit(
    config: &CohortConfig,
    config_bytes: &[u8],
    input: &super::super::cohort::LoadedCohort,
    native: &[crate::Gaussian3d],
    training: &[FitView],
    heldout: &[FitView],
    optimization_seconds: f64,
    mut report: serde_json::Value,
) -> CaptureResult<()> {
    fs::create_dir(&config.output_directory)?;
    let candidate = config.output_directory.join("candidate.ply");
    let io_started = Instant::now();
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&candidate)?,
    );
    write_ply_header(&mut writer, native.len() as u64)?;
    for gaussian in native {
        write_ply_gaussian(&mut writer, gaussian)?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    let candidate_hash = hash_file(&candidate)?;
    let reload = reload_candidate(
        &mut std::io::BufReader::new(fs::File::open(&candidate)?),
        native.len(),
    );
    let mut status: &str;
    let mut reason = None;
    let mut validation_seconds = 0.0;
    let mut comparisons = Vec::new();
    let mut maximum_scalar_difference = None;
    let mut reloaded_record_sha256 = None;
    let mut reload_capacity_bytes = 0usize;
    let io_seconds;
    match reload {
        Err(error) => {
            io_seconds = io_started.elapsed().as_secs_f64();
            reason = Some(format!("deployment PLY reload failed: {error}"));
            status = "rejected";
        }
        Ok(reloaded) => {
            reload_capacity_bytes =
                (reloaded.capacity() + 1024) * std::mem::size_of::<crate::Gaussian3d>();
            io_seconds = io_started.elapsed().as_secs_f64();
            let validation_started = Instant::now();
            if reloaded.len() != native.len() {
                reason = Some("deployment reload changed record count".into());
                status = "rejected";
            } else {
                maximum_scalar_difference = Some(
                    bytemuck::cast_slice::<_, f32>(native)
                        .iter()
                        .zip(bytemuck::cast_slice::<_, f32>(&reloaded))
                        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                        .fold(0.0, f64::max),
                );
                reloaded_record_sha256 = Some(format!(
                    "{:x}",
                    Sha256::digest(bytemuck::cast_slice::<_, u8>(&reloaded))
                ));
                status = "passed";
                for view in training.iter().chain(heldout) {
                    let remaining = f64::from(config.options.max_seconds)
                        - optimization_seconds
                        - validation_started.elapsed().as_secs_f64();
                    if remaining < 1.0 {
                        status = "inconclusive";
                        reason =
                            Some("shared optimizer/roundtrip compute deadline exhausted".into());
                        break;
                    }
                    let mut options = config.options.clone();
                    options.max_seconds = remaining.floor() as u32;
                    options.max_training_seconds = 0;
                    options.max_steps = 1;
                    options.fit_geometry = false;
                    let compared = compare_representatives(
                        FitProblem {
                            source: native,
                            initial: &reloaded,
                            context: &input.context,
                            domains: &input.domains,
                            source_order: input.initial_order.as_deref(),
                            initial_order: input.initial_order.as_deref(),
                            context_order: input.context_order.as_deref(),
                        },
                        std::slice::from_ref(view),
                        input.color_space,
                        &options,
                    );
                    let compared = match compared {
                        Ok(compared) => compared,
                        Err(error) => {
                            status = if error == "wall_time_budget"
                                || error.contains("budget")
                                || error.contains("ceiling")
                            {
                                "inconclusive"
                            } else {
                                "rejected"
                            };
                            reason = Some(error);
                            break;
                        }
                    };
                    let image = &compared.views[0];
                    let maximum = maximum_image_drift(&image.source, &image.initial);
                    let passed = maximum[0] <= config.ply_roundtrip.max_rgb_absolute_error
                        && maximum[1] <= config.ply_roundtrip.max_transmittance_absolute_error;
                    comparisons.push(json!({"view": view.id, "maximum_rgb_absolute_error": maximum[0],
                        "maximum_owned_transmittance_absolute_error": maximum[1],
                        "composed_rgb_mse": image.composed_rgb_mse, "owned_transmittance_mse": image.owned_transmittance_mse,
                        "passed": passed, "work": compared.work}));
                    if !passed {
                        status = "rejected";
                        reason = Some(format!(
                            "PLY roundtrip image drift exceeds tolerance in {}",
                            view.id
                        ));
                        break;
                    }
                }
            }
            validation_seconds = validation_started.elapsed().as_secs_f64();
            if status == "passed"
                && optimization_seconds + validation_seconds > f64::from(config.options.max_seconds)
            {
                status = "inconclusive";
                reason = Some("shared optimizer/roundtrip compute deadline exhausted".into());
            }
        }
    }
    let accepted = status == "passed" && comparisons.len() == training.len() + heldout.len();
    let artifact = config.output_directory.join(if accepted {
        "fitted.ply"
    } else {
        "unaccepted-candidate.ply"
    });
    report["output_path"] = serde_json::to_value(&artifact)?;
    report["output_sha256"] = candidate_hash.into();
    report["artifact_accepted"] = accepted.into();
    report["ply_logit_logscale_roundtrip_error_measured"] =
        (comparisons.len() == training.len() + heldout.len()).into();
    report["ply_roundtrip"] = json!({"status": status, "reason": reason, "policy": config.ply_roundtrip,
        "native_record_sha256": format!("{:x}", Sha256::digest(bytemuck::cast_slice::<_, u8>(native))),
        "reloaded_record_sha256": reloaded_record_sha256, "maximum_scalar_difference": maximum_scalar_difference,
        "optimization_seconds": optimization_seconds, "validation_compute_seconds": validation_seconds,
        "shared_compute_limit_seconds": config.options.max_seconds, "write_hash_reload_io_seconds": io_seconds,
        "additional_reload_record_capacity_bytes": reload_capacity_bytes,
        "reload_allocation_scope": "one preallocated owned-record vector plus 1024-record production parser batch; header, buffered I/O and allocator overhead are separate; no planar cloud copy", "views": comparisons,
        "expected_views": training.len() + heldout.len(),
        "scope": "artifact image fidelity on the same configured pixel samples and immutable native context; no full-scene or source-quality promotion; scalar equality alone never passes the gate",
        "failure_policy": "native optimized records are unchanged; unaccepted-candidate.ply is diagnostic evidence only, never fitted.ply"});
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(config.output_directory.join("fit.json"))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    let mut writer = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(config.output_directory.join("config.json"))?;
    writer.write_all(config_bytes)?;
    writer.sync_all()?;
    // An accepted filename appears only after its complete gate report is durable.
    fs::rename(&candidate, &artifact)?;
    if accepted {
        Ok(())
    } else {
        Err(format!("PLY roundtrip {status}; candidate retained as unaccepted evidence").into())
    }
}

fn diagnostic_png(
    path: &Path,
    viewport: [u32; 2],
    pixels: &[[f32; 4]],
    transmittance: bool,
) -> CaptureResult<()> {
    use bevy::{
        asset::RenderAssetUsages,
        prelude::Image,
        render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    };
    let encode = |linear: f32| {
        let linear = linear.clamp(0.0, 1.0);
        let display = if linear <= 0.0031308 {
            12.92 * linear
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        (display * 255.0).round() as u8
    };
    let rgba = pixels
        .iter()
        .flat_map(|pixel| {
            if transmittance {
                let value = (pixel[3].clamp(0.0, 1.0) * 255.0).round() as u8;
                [value, value, value, 255]
            } else {
                [encode(pixel[0]), encode(pixel[1]), encode(pixel[2]), 255]
            }
        })
        .collect();
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
    Ok(())
}

fn publish_diagnostic(
    config: &CohortConfig,
    config_bytes: &[u8],
    input: &super::super::cohort::LoadedCohort,
    diagnostic: &FitDiagnosticOutput,
    admission_seconds: f64,
    candidate: Option<&[crate::Gaussian3d]>,
) -> CaptureResult<()> {
    fs::create_dir(&config.output_directory)?;
    let mut created = Vec::new();
    let publish = (|| -> CaptureResult<()> {
        let mut artifacts = Vec::new();
        for (index, view) in diagnostic.views.iter().enumerate() {
            let initial_role = if candidate.is_some() {
                "candidate"
            } else {
                "seed"
            };
            for (role, pixels) in [("source", &view.source), (initial_role, &view.initial)] {
                let stem = format!("view-{index:02}-{role}");
                let raw = config.output_directory.join(format!("{stem}.rgbt32"));
                created.push(raw.clone());
                let mut writer =
                    BufWriter::new(OpenOptions::new().create_new(true).write(true).open(&raw)?);
                for pixel in pixels {
                    for value in pixel {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                }
                writer.flush()?;
                writer.get_ref().sync_all()?;
                let rgb = config.output_directory.join(format!("{stem}.png"));
                let transmittance = config.output_directory.join(format!("{stem}-owned-t.png"));
                created.push(rgb.clone());
                diagnostic_png(&rgb, view.viewport, pixels, false)?;
                created.push(transmittance.clone());
                diagnostic_png(&transmittance, view.viewport, pixels, true)?;
                artifacts.push(
                    json!({ "view": view.id, "role": role, "viewport": view.viewport,
                    "rgb_png": rgb, "owned_transmittance_png": transmittance,
                    "linear_rgbt_f32le": raw, "linear_rgbt_sha256": hash_file(&raw)? }),
                );
            }
        }
        let report = json!({
            "schema_version": 1, "kind": if candidate.is_some() { "candidate_cohort_substitution_diagnostic" } else { "unchanged_cohort_substitution_diagnostic" },
            "candidate_ply": config.candidate_ply,
            "candidate_order": "unchanged owned seed indices and owner bounds; candidate records are deployment PLY reload values when supplied",
            "optimizer_steps": 0, "parameters_changed": false,
            "gpu_forward_parity_qualified": false, "quality_certificate": false,
            "configuration": config, "config_sha256": format!("{:x}", Sha256::digest(config_bytes)),
            "fitter_executable_sha256": hash_file(&std::env::current_exe()?)?,
            "cohort_lineage": input.sidecar, "admission_seconds": admission_seconds,
            "diagnostic": diagnostic, "artifacts": artifacts,
            "target_layout": "row-major little-endian [linear premultiplied composed R,G,B,owned T], four f32 per pixel",
            "display": "RGB PNG uses sRGB display conversion on black; owned-T PNG is linear grayscale; metrics use unclipped linear arrays",
            "context": "identical frozen records in one shared forward camera-depth order for source and evaluated seed/candidate; excluded from owned transmittance",
            "input_values": "source/context and default seed use authenticated native pages; optional candidate_ply uses its SHA-verified deployment reload",
            "support": "shared smooth finite three-sigma support for native original leaves, seed and context; no adaptive flat-teacher equivalence claimed",
            "scope": "authenticated selected context and configured pixel samples; no full-scene coverage, training or quality qualification implied"
        });
        let report_path = config.output_directory.join("diagnostic.json");
        created.push(report_path.clone());
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(report_path)?,
        );
        serde_json::to_writer_pretty(&mut writer, &report)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        let frozen = config.output_directory.join("config.json");
        created.push(frozen.clone());
        let mut writer = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(frozen)?;
        writer.write_all(config_bytes)?;
        writer.sync_all()?;
        Ok(())
    })();
    if publish.is_err() {
        for path in created {
            let _ = fs::remove_file(path);
        }
        let _ = fs::remove_dir(&config.output_directory);
    }
    publish
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohort_ply_reload_measures_anisotropic_image_drift_with_frozen_context() {
        let mut native = crate::Gaussian3d::default();
        native.position_visibility.visibility = 1.0;
        native.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
        native.scale_opacity.scale = [0.01, 10.0, 0.01];
        native.scale_opacity.opacity = 0.8;
        let mut context = native;
        context.position_visibility.position[2] = -0.5;
        context.scale_opacity.scale = [0.3; 3];
        let mut bytes = Vec::new();
        write_ply_header(&mut bytes, 1).unwrap();
        write_ply_gaussian(&mut bytes, &native).unwrap();
        let reloaded = reload_candidate(&mut std::io::Cursor::new(bytes), 1).unwrap();
        assert!(reloaded[0].scale_opacity.scale[1] < native.scale_opacity.scale[1]);
        let domain = LodBounds::new([-100.0; 3], [100.0; 3]).unwrap();
        let view = FitView {
            id: "reload".into(),
            camera: LodTestCamera {
                world_rotation: None,
                position: Vec3::new(0.0, 0.0, 3.0),
                target: Vec3::ZERO,
                up: Vec3::Y,
                projection: LodProjection::Perspective {
                    vertical_fov_radians: 1.0,
                },
                near: 0.01,
                far: 100.0,
                viewport: [32, 32],
            },
        };
        let compare = |initial: &[crate::Gaussian3d]| {
            compare_representatives(
                FitProblem {
                    source: std::slice::from_ref(&native),
                    initial,
                    context: std::slice::from_ref(&context),
                    domains: std::slice::from_ref(&domain),
                    source_order: None,
                    initial_order: None,
                    context_order: None,
                },
                std::slice::from_ref(&view),
                crate::gaussian::settings::GaussianColorSpace::SrgbRec709Display,
                &FitOptions::default(),
            )
            .unwrap()
        };
        let identity = compare(std::slice::from_ref(&native));
        assert_eq!(
            maximum_image_drift(&identity.views[0].source, &identity.views[0].initial),
            [0.0; 2]
        );
        let changed = compare(&reloaded);
        let drift = maximum_image_drift(&changed.views[0].source, &changed.views[0].initial);
        assert!(drift[0] > PlyRoundtripPolicy::default().max_rgb_absolute_error);
        assert!(drift[1] > PlyRoundtripPolicy::default().max_transmittance_absolute_error);
    }

    #[test]
    fn cohort_fit_rejects_cross_split_frame_reuse_and_rescaled_crops() {
        let camera = CalibratedCamera {
            id: "train".into(),
            camera_path: PinnedFile {
                path: "cameras.json".into(),
                sha256: "0".repeat(64),
            },
            frame_index: 0,
            calibration_viewport: [960, 640],
            crop: Some(LodPixelCrop {
                origin: [20, 30],
                size: [32, 32],
            }),
            near: 0.1,
            far: 100.0,
        };
        let mut config = CohortConfig {
            cohort_sidecar: PinnedFile {
                path: "cohort.json".into(),
                sha256: "1".repeat(64),
            },
            context_representation: ContextRepresentation::Selected,
            diagnostic_only: false,
            candidate_ply: None,
            ply_roundtrip: PlyRoundtripPolicy::default(),
            output_directory: "fit".into(),
            viewport: [32, 32],
            training: vec![camera.clone()],
            heldout: vec![CalibratedCamera {
                id: "heldout".into(),
                frame_index: 1,
                ..camera
            }],
            options: FitOptions::default(),
        };
        config.validate().unwrap();
        config.candidate_ply = Some(PinnedFile {
            path: "fitted.ply".into(),
            sha256: "2".repeat(64),
        });
        config.validate().unwrap();
        config.candidate_ply = None;
        config.heldout[0].frame_index = 0;
        config.heldout[0].crop.as_mut().unwrap().origin[0] += 1;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("physical camera frame")
        );
        config.heldout[0].frame_index = 1;
        config.viewport = [16, 16];
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("physical crop size")
        );
    }
}
