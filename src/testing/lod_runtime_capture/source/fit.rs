//! CPU-only fitting/export admission. All source identities remain distinct:
//! the original PLY, frozen rung PLY, authenticated manifest and fit result.

use std::{
    collections::BTreeSet,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use bevy::prelude::Vec3;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::{CaptureResult, hash_file, load_source, write_ply_gaussian, write_ply_header};

mod cohort;
use crate::{
    gaussian::formats::planar_3d_chunked::{LodBounds, LodNodeId, LodPageId, LodSourceRange},
    io::lod::{LodCodecLimits, decode_manifest},
    testing::{
        lod_scenes::{LodProjection, LodTestCamera},
        render_oracle::fit::{
            FitOptions, FitView, MAX_FIT_VIEWS, fit_representatives, training_teacher_bytes,
        },
    },
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PinnedFile {
    path: PathBuf,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Camera {
    id: String,
    from: [f32; 3],
    target: [f32; 3],
    #[serde(default = "up_y")]
    up: [f32; 3],
    vertical_fov_radians: f32,
    near: f32,
    far: f32,
}
fn up_y() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

impl Camera {
    fn view(&self, viewport: [u32; 2]) -> FitView {
        FitView {
            id: self.id.clone(),
            camera: LodTestCamera {
                world_rotation: None,
                position: Vec3::from_array(self.from),
                target: Vec3::from_array(self.target),
                up: Vec3::from_array(self.up),
                projection: LodProjection::Perspective {
                    vertical_fov_radians: self.vertical_fov_radians,
                },
                near: self.near,
                far: self.far,
                viewport,
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    source: PinnedFile,
    initial_rung: PinnedFile,
    rung_sidecar: PinnedFile,
    /// Pinned successful builder argv record joins original file path to package.
    /// It is provenance evidence, not an independent canonical-source recomputation.
    build_record: PinnedFile,
    output_directory: PathBuf,
    expected_source_gaussians: u64,
    expected_rung_gaussians: u64,
    viewport: [u32; 2],
    training: Vec<Camera>,
    heldout: Vec<Camera>,
    #[serde(default)]
    options: FitOptions,
}

#[derive(Debug, Deserialize)]
struct RungOwner {
    node: LodNodeId,
    source: LodSourceRange,
    conservative_node_bounds: LodBounds,
    page: LodPageId,
    decoded_page_offset: u32,
    output_start: u64,
    output_count: u32,
}

#[derive(Debug, Deserialize)]
struct RungSidecar {
    kind: String,
    complete_source_antichain_validated: bool,
    manifest_path: PathBuf,
    manifest_sha256: String,
    canonical_decoded_source_fingerprint: String,
    source_gaussian_count: u64,
    output_gaussian_count: u64,
    output_sha256: String,
    owners: Vec<RungOwner>,
}

fn check_file(file: &PinnedFile, max_bytes: u64) -> CaptureResult<()> {
    if fs::metadata(&file.path)?.len() > max_bytes {
        return Err(format!(
            "fit input exceeds {} byte admission: {}",
            max_bytes,
            file.path.display()
        )
        .into());
    }
    if file.sha256.len() != 64
        || !file.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        || hash_file(&file.path)? != file.sha256
    {
        return Err(format!("fit input SHA256 mismatch: {}", file.path.display()).into());
    }
    Ok(())
}

fn validate_split(config: &Config) -> CaptureResult<()> {
    config.options.validate()?;
    if config.expected_source_gaussians == 0
        || config.expected_source_gaussians > 100_000
        || config.expected_rung_gaussians == 0
        || config.expected_rung_gaussians > 30_000
        || config.training.is_empty()
        || config.training.len() > MAX_FIT_VIEWS
        || config.heldout.is_empty()
        || config.heldout.len() > MAX_FIT_VIEWS
        || config.viewport.contains(&0)
        || config.viewport.iter().any(|n| *n > 512)
    {
        return Err("fit config requires 1..=16 training and 1..=16 heldout views; <=100k sources, <=30k frozen representatives, <=512x512 image".into());
    }
    training_teacher_bytes(config.training.iter().map(|_| config.viewport))?;
    if let Some(pixels) = config.options.reference_pixels {
        pixels.sample_stride(config.viewport)?;
    }
    let mut ids = BTreeSet::new();
    let mut poses = Vec::new();
    for camera in config.training.iter().chain(&config.heldout) {
        if camera.id.is_empty() || !ids.insert(&camera.id) {
            return Err("train/heldout camera IDs overlap".into());
        }
        let view = camera.view(config.viewport).camera;
        let direction = (view.target - view.position).normalize_or_zero();
        let up = view.up.normalize_or_zero();
        if !view.position.is_finite()
            || !view.target.is_finite()
            || !view.up.is_finite()
            || direction == Vec3::ZERO
            || up == Vec3::ZERO
            || direction.cross(up).length_squared() < 1e-10
            || !camera.vertical_fov_radians.is_finite()
            || camera.vertical_fov_radians <= 0.0
            || camera.vertical_fov_radians >= std::f32::consts::PI
            || !camera.near.is_finite()
            || !camera.far.is_finite()
            || camera.near <= 0.0
            || camera.far <= camera.near
        {
            return Err("fit camera has nonfinite/singular pose or projection".into());
        }
        let pose = (view.position, direction, up, camera.vertical_fov_radians);
        if poses.contains(&pose) {
            return Err("duplicate train/heldout physical camera pose".into());
        }
        poses.push(pose);
    }
    Ok(())
}

fn domains(sidecar: &RungSidecar, config: &Config) -> CaptureResult<Vec<LodBounds>> {
    if sidecar.kind != "authored_rung_export"
        || !sidecar.complete_source_antichain_validated
        || sidecar.source_gaussian_count != config.expected_source_gaussians
        || sidecar.output_gaussian_count != config.expected_rung_gaussians
        || sidecar.output_sha256 != config.initial_rung.sha256
    {
        return Err("fit seed sidecar does not describe the pinned complete authored rung".into());
    }
    check_file(
        &PinnedFile {
            path: sidecar.manifest_path.clone(),
            sha256: sidecar.manifest_sha256.clone(),
        },
        64 * 1024 * 1024,
    )?;
    let manifest = decode_manifest(
        &fs::read(&sidecar.manifest_path)?,
        LodCodecLimits {
            max_manifest_bytes: 64 * 1024 * 1024,
            max_nodes: 262_144,
            max_pages: 65_536,
            max_page_bytes: 16 * 1024 * 1024,
            max_page_gaussians: 65_535,
        },
    )?;
    if manifest.header.source_gaussian_count != config.expected_source_gaussians
        || format!("{:016x}", manifest.build.source_fingerprint)
            != sidecar.canonical_decoded_source_fingerprint
    {
        return Err(
            "fit source canonical fingerprint disagrees with authenticated manifest".into(),
        );
    }
    let nodes = manifest
        .nodes
        .iter()
        .map(|node| (node.id, node))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut source_end = 0_u64;
    let mut output_end = 0_u64;
    let mut result = Vec::with_capacity(config.expected_rung_gaussians as usize);
    let mut physical = std::collections::BTreeMap::<LodPageId, Vec<(u32, u32)>>::new();
    for owner in &sidecar.owners {
        let node = nodes
            .get(&owner.node)
            .ok_or("fit owner absent from manifest")?;
        if owner.source.start != source_end
            || owner.source.count == 0
            || owner.output_start != output_end
            || owner.output_count == 0
            || owner.source != node.source
            || owner.conservative_node_bounds != node.bounds
            || owner.page != node.representation.page
            || owner.decoded_page_offset != node.representation.offset
            || owner.output_count != node.representation.count
        {
            return Err("fit owner range/bounds does not match complete manifest antichain".into());
        }
        source_end = owner.source.end().ok_or("fit source ownership overflow")?;
        output_end = output_end
            .checked_add(u64::from(owner.output_count))
            .ok_or("fit output ownership overflow")?;
        if output_end > config.expected_rung_gaussians {
            return Err("fit ownership exceeds admitted cardinality".into());
        }
        let end = owner
            .decoded_page_offset
            .checked_add(owner.output_count)
            .ok_or("fit physical range overflow")?;
        physical
            .entry(owner.page)
            .or_default()
            .push((owner.decoded_page_offset, end));
        result.extend(std::iter::repeat_n(
            owner.conservative_node_bounds,
            owner.output_count as usize,
        ));
    }
    for ranges in physical.values_mut() {
        ranges.sort_unstable();
        if ranges.windows(2).any(|v| v[0].1 > v[1].0) {
            return Err("fit seed repeats physical payload range".into());
        }
    }
    if source_end != config.expected_source_gaussians
        || output_end != config.expected_rung_gaussians
    {
        return Err("fit seed does not cover complete source or output".into());
    }
    Ok(result)
}

fn validate_build_lineage(config: &Config, sidecar: &RungSidecar) -> CaptureResult<()> {
    let build: serde_json::Value = serde_json::from_slice(&fs::read(&config.build_record.path)?)?;
    if build["exit_code"].as_u64() != Some(0) {
        return Err("fit source build record did not succeed".into());
    }
    let argv = build["argv"]
        .as_array()
        .ok_or("build record missing argv")?;
    let flag_path = |flag: &str| -> CaptureResult<PathBuf> {
        let matches = argv
            .windows(2)
            .filter(|pair| pair[0].as_str() == Some(flag))
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err("ambiguous source build argv".into());
        }
        Ok(fs::canonicalize(
            matches[0][1]
                .as_str()
                .ok_or("nonstring source build argv")?,
        )?)
    };
    if flag_path("--input")? != fs::canonicalize(&config.source.path)?
        || flag_path("--output")?
            != fs::canonicalize(
                sidecar
                    .manifest_path
                    .parent()
                    .ok_or("manifest has no parent")?,
            )?
    {
        return Err("recorded builder does not connect pinned source path to seed package".into());
    }
    Ok(())
}

/// Read a frozen config, fit within hard CPU ceilings, and publish a new PLY plus
/// a diagnostic report. Never rewrites the package, seed payload, or certificate.
pub fn fit_rung(config_path: &Path) -> CaptureResult<()> {
    let started = Instant::now();
    if fs::metadata(config_path)?.len() > 1024 * 1024 {
        return Err("fit config exceeds 1 MiB".into());
    }
    let config_bytes = fs::read(config_path)?;
    let config_sha256 = format!("{:x}", Sha256::digest(&config_bytes));
    if serde_json::from_slice::<serde_json::Value>(&config_bytes)?
        .get("cohort_sidecar")
        .is_some()
    {
        return cohort::fit_cohort(&config_bytes, started);
    }
    let config: Config = serde_json::from_slice(&config_bytes)?;
    validate_split(&config)?;
    if config.output_directory.exists() {
        return Err("fit output directory already exists; refusing overwrite".into());
    }
    for (file, limit) in [
        (&config.source, 100 * 1024 * 1024),
        (&config.initial_rung, 32 * 1024 * 1024),
        (&config.rung_sidecar, 16 * 1024 * 1024),
        (&config.build_record, 1024 * 1024),
    ] {
        check_file(file, limit)?;
    }
    if config.source.path.extension().and_then(|v| v.to_str()) != Some("ply")
        || config
            .initial_rung
            .path
            .extension()
            .and_then(|v| v.to_str())
            != Some("ply")
    {
        return Err(
            "diagnostic fit admits PLY with identity cloud transform and sRGB display SH only"
                .into(),
        );
    }
    let sidecar: RungSidecar = serde_json::from_slice(&fs::read(&config.rung_sidecar.path)?)?;
    let domains = domains(&sidecar, &config)?;
    validate_build_lineage(&config, &sidecar)?;
    let source = load_source(&config.source.path, config.expected_source_gaussians)?;
    let initial = load_source(&config.initial_rung.path, config.expected_rung_gaussians)?;
    let source_records = source.cloud.iter().collect::<Vec<_>>();
    let initial_records = initial.cloud.iter().collect::<Vec<_>>();
    if source_records.len() as u64 != config.expected_source_gaussians
        || initial_records.len() as u64 != config.expected_rung_gaussians
    {
        return Err("fit decoded source/candidate count differs from pinned cardinality".into());
    }
    let color_space = source.color_space;
    drop(source);
    drop(initial);
    let training = config
        .training
        .iter()
        .map(|v| v.view(config.viewport))
        .collect::<Vec<_>>();
    let heldout = config
        .heldout
        .iter()
        .map(|v| v.view(config.viewport))
        .collect::<Vec<_>>();
    let admission_seconds = started.elapsed().as_secs_f64();
    let result = fit_representatives(
        &source_records,
        &initial_records,
        &domains,
        &training,
        &heldout,
        color_space,
        &config.options,
    )?;
    // Detect external input changes during fitting. This is not a source hash
    // relabel: the resulting PLY receives its own SHA below.
    for (file, limit) in [
        (&config.source, 100 * 1024 * 1024),
        (&config.initial_rung, 32 * 1024 * 1024),
        (&config.rung_sidecar, 16 * 1024 * 1024),
        (&config.build_record, 1024 * 1024),
    ] {
        check_file(file, limit)?;
    }
    check_file(
        &PinnedFile {
            path: sidecar.manifest_path.clone(),
            sha256: sidecar.manifest_sha256.clone(),
        },
        64 * 1024 * 1024,
    )?;
    let report = json!({
        "schema_version":1,"kind":"diagnostic_cpu_oracle_representative_fit",
        "gpu_forward_parity_qualified":false,"quality_certificate":false,
        "package_or_error_policy_modified":false,"fixed_cardinality":result.gaussians.len(),
        "source":config.source,"initial_rung":config.initial_rung,"rung_sidecar":config.rung_sidecar,
        "build_record":config.build_record,"manifest_path":sidecar.manifest_path,
        "manifest_sha256":sidecar.manifest_sha256,
        "canonical_decoded_source_fingerprint":sidecar.canonical_decoded_source_fingerprint,
        "source_lineage_scope":"pinned successful builder argv and authenticated manifest fingerprint; original canonical fingerprint is not independently recomputed here",
        "fitter_executable_sha256":hash_file(&std::env::current_exe()?)?,
        "config_sha256":config_sha256,"configuration":config,
        "output_order":"unchanged seed representative indices; inherit exact seed node/source/output owner ranges",
        "ownership":"every published full three-sigma support remains within authenticated original node bounds",
        "appearance_subspace": {
            "max_fitted_sh_degree":config.options.max_fitted_sh_degree,
            "higher_coefficients":"rendered unchanged; preserved exactly from the authenticated seed, not zeroed",
            "scope":"optimizer coordinates only; output SH layout and renderer ABI unchanged"
        },
        "projection_sampling": {
            "allocated_viewport":config.viewport,
            "reference_viewport":config.options.reference_pixels.map_or(config.viewport, |pixels| pixels.viewport),
            "pixel_stride":config.options.reference_pixels.map_or(1, |pixels| pixels.sample_stride(config.viewport).expect("validated sampling")),
            "pixel_offset":config.options.reference_pixels.map_or([0,0], |pixels| pixels.offset),
            "filter":"Mip covariance and determinant opacity compensation evaluated at the reference viewport, before mapping to the sampled grid",
            "scope":"fixed actual pixel centers; sparse grid is not downsampled pixel integration or full deployment-image qualification"
        },
        "geometry_feasibility":{
            "policy":config.options.geometry_feasibility,
            "per_representative_factors":[1.0,0.5,0.25,0.125,0.0625,0.03125],
            "fallback":"under per_representative_backtracking, only infeasible geometry freezes; SH/opacity still participate in global training acceptance",
            "counts":"per global trial with accepted/rejected/deadline outcome; interrupted_step_trials are unpublished, never accepted updates"
        },
        "support":"fixed authored 3 sigma OBB, identity transform, no tonemap, CPU f32 premultiplied linear RGB and owned transmittance; actual GPU parity unqualified",
        "objective":"mean train sampled-pixel linear premultiplied RGB MSE + alpha_weight * owned-transmittance MSE; heldout is evaluated only after optimizer stops",
        "derivatives":"analytic reverse source-over, SH and logit opacity; optional local projection finite differences for mean/log-scale/local rotation; four fresh-forward backtracking attempts",
        "admission_seconds":admission_seconds,"fit":result.report,
        "ply_logit_logscale_roundtrip_error_measured":false,
        "limits":["bounded teacher-relative pixel-grid diagnostic, not photo supervision or full deployment quality","no heldout-based model selection","wall budget covers CPU teachers, optimizer and heldout; bounded input hashing/decoding/export reported separately","overflow fails without a partial published PLY; deadline retains last fully accepted candidate"]
    });
    publish_fit(
        &config.output_directory,
        &result.gaussians,
        &config_bytes,
        report,
    )
}

fn publish_fit(
    output_directory: &Path,
    gaussians: &[crate::Gaussian3d],
    config_bytes: &[u8],
    mut report: serde_json::Value,
) -> CaptureResult<()> {
    fs::create_dir(output_directory)?;
    let output = output_directory.join("fitted.ply");
    let publish = (|| -> CaptureResult<()> {
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&output)?,
        );
        write_ply_header(&mut writer, gaussians.len() as u64)?;
        for gaussian in gaussians {
            write_ply_gaussian(&mut writer, gaussian)?;
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        report["output_path"] = serde_json::to_value(&output)?;
        report["output_sha256"] = hash_file(&output)?.into();
        let mut record = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output_directory.join("fit.json"))?;
        serde_json::to_writer_pretty(&mut record, &report)?;
        writeln!(record)?;
        record.sync_all()?;
        let mut frozen = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output_directory.join("config.json"))?;
        frozen.write_all(config_bytes)?;
        frozen.sync_all()?;
        Ok(())
    })();
    if publish.is_err() {
        // This directory was created by this invocation; clean only its known
        // files, never a caller-owned directory tree.
        for name in ["fitted.ply", "fit.json", "config.json"] {
            let _ = fs::remove_file(output_directory.join(name));
        }
        let _ = fs::remove_dir(output_directory);
    }
    publish
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Config {
        let pin = PinnedFile {
            path: "unused".into(),
            sha256: "0".repeat(64),
        };
        let camera = Camera {
            id: "train".into(),
            from: [0.0, 0.0, 3.0],
            target: [0.0; 3],
            up: up_y(),
            vertical_fov_radians: 1.0,
            near: 0.1,
            far: 100.0,
        };
        let mut config = Config {
            source: pin.clone(),
            initial_rung: pin.clone(),
            rung_sidecar: pin.clone(),
            build_record: pin,
            output_directory: "unused".into(),
            expected_source_gaussians: 100,
            expected_rung_gaussians: 25,
            viewport: [32, 32],
            training: vec![camera.clone(); 2],
            heldout: vec![camera; 6],
            options: FitOptions::default(),
        };
        for (i, v) in config
            .training
            .iter_mut()
            .chain(&mut config.heldout)
            .enumerate()
        {
            v.id = format!("view_{i}");
            v.from[0] = i as f32 * 0.1;
        }
        config
    }

    #[test]
    fn split_rejects_pose_leakage_even_with_different_ids() {
        let mut config = fixture();
        validate_split(&config).unwrap();
        config.heldout[0].from = config.training[0].from;
        assert!(
            validate_split(&config)
                .unwrap_err()
                .to_string()
                .contains("physical camera")
        );
    }

    #[test]
    fn broader_split_admits_bounded_teachers_and_rejects_seventeenth_views() {
        let mut config = fixture();
        config.training.resize(12, config.training[0].clone());
        config.heldout.resize(12, config.heldout[0].clone());
        for (i, camera) in config
            .training
            .iter_mut()
            .chain(&mut config.heldout)
            .enumerate()
        {
            camera.id = format!("view_{i}");
            camera.from[0] = i as f32 * 0.1;
        }
        config.viewport = [256, 144];
        validate_split(&config).unwrap();
        config.viewport = [384, 216];
        assert!(
            validate_split(&config)
                .unwrap_err()
                .to_string()
                .contains("8 MiB")
        );
        config.viewport = [256, 144];
        let mut too_many = config.clone();
        too_many.training.resize(17, config.training[0].clone());
        assert!(
            validate_split(&too_many)
                .unwrap_err()
                .to_string()
                .contains("1..=16")
        );
        config.heldout.resize(17, config.heldout[0].clone());
        assert!(
            validate_split(&config)
                .unwrap_err()
                .to_string()
                .contains("1..=16")
        );
    }
}
