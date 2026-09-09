//! Bounded admission for flat source loading and GLB-to-PLY teacher conversion.
//! GLB loading reuses the production KHR_gaussian_splatting decoder, including
//! authored cloud transform and color conventions. Multi-primitive scenes are
//! rejected because this capture schema describes one ordered stream.

use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::Path,
    time::{Duration, Instant},
};

use bevy::{
    asset::{
        AssetMetaCheck, AssetPlugin, DependencyLoadState, LoadState, RecursiveDependencyLoadState,
        UnapprovedPathMode,
    },
    prelude::*,
};
use bevy_interleave::prelude::Planar;
use serde::{Deserialize, Serialize};

use crate::{
    PlanarGaussian3d,
    gaussian::{formats::planar_3d::Gaussian3d, settings::GaussianColorSpace},
    io::{
        IoPlugin,
        ply::{PlyShCompatibility, stream_ply_3d_with_sh_compatibility},
        scene::GaussianScene,
    },
    material::spherical_harmonics::{SH_CHANNELS, SH_COEFF_COUNT_PER_CHANNEL},
};

use super::{CaptureResult, hash_file};

mod attribute_cut;
pub use attribute_cut::attribute_cut;
mod export_rung;
pub use export_rung::export_rung;
pub(crate) mod cohort;
pub use cohort::export_cohort;
mod fit;
pub use fit::fit_rung;

pub(crate) struct LoadedSource {
    pub cloud: PlanarGaussian3d,
    pub transform: Transform,
    pub color_space: GaussianColorSpace,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub original_sha256: String,
    pub gaussian_count: u64,
    pub world_from_local: [f32; 16],
    pub color_space: GaussianColorSpace,
    pub converted_ply_sha256: Option<String>,
    /// PLY stores opacity logits/log scales; their float roundtrip is measured.
    pub maximum_scalar_roundtrip_error: Option<f64>,
}

pub(crate) fn load_source(path: &Path, max_gaussians: u64) -> CaptureResult<LoadedSource> {
    match path.extension().and_then(|value| value.to_str()) {
        Some("glb") => load_glb(path, max_gaussians),
        Some("ply") => {
            let mut records = Vec::new();
            stream_ply_3d_with_sh_compatibility(
                &mut BufReader::new(File::open(path)?),
                1024,
                PlyShCompatibility::RequireRepresentable,
                |batch| {
                    if records.len() as u64 + batch.len() as u64 > max_gaussians {
                        return Err(std::io::Error::other(
                            "flat source exceeds max_source_gaussians",
                        ));
                    }
                    records.extend_from_slice(batch);
                    Ok(())
                },
            )?;
            Ok(LoadedSource {
                cloud: PlanarGaussian3d::from_interleaved(records),
                transform: Transform::IDENTITY,
                color_space: GaussianColorSpace::SrgbRec709Display,
            })
        }
        _ => Err("capture source must be .ply or one self-contained .glb".into()),
    }
}

fn load_glb(path: &Path, max_gaussians: u64) -> CaptureResult<LoadedSource> {
    let maximum_file_bytes = max_gaussians
        .checked_mul(1024)
        .and_then(|v| v.checked_add(16 * 1024 * 1024))
        .ok_or("GLB byte admission overflow")?;
    if fs::metadata(path)?.len() > maximum_file_bytes {
        return Err("GLB exceeds bounded source byte admission".into());
    }
    // Inspect untrusted counts/instance fanout before the loader allocates the
    // canonical planes. External buffers would escape this byte identity/bound.
    let bytes = fs::read(path)?;
    let gltf = gltf::Gltf::from_slice_without_validation(&bytes)?;
    if gltf
        .buffers()
        .any(|buffer| !matches!(buffer.source(), gltf::buffer::Source::Bin))
    {
        return Err("capture GLB must contain only its own binary buffer".into());
    }
    let primitives: Vec<_> = gltf.meshes().flat_map(|mesh| mesh.primitives()).collect();
    if primitives.len() != 1 || gltf.nodes().filter(|node| node.mesh().is_some()).count() != 1 {
        return Err("capture GLB requires exactly one mesh primitive and one instance".into());
    }
    let count = primitives[0]
        .get(&gltf::Semantic::Positions)
        .ok_or("GLB missing POSITION")?
        .count() as u64;
    if count == 0 || count > max_gaussians {
        return Err("GLB exceeds max_source_gaussians".into());
    }
    drop(primitives);
    drop(gltf);
    drop(bytes);
    let path = fs::canonicalize(path)?;
    let directory = path.parent().ok_or("source has no parent")?;
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin {
            file_path: directory.to_string_lossy().into_owned(),
            processed_file_path: directory.to_string_lossy().into_owned(),
            meta_check: AssetMetaCheck::Never,
            unapproved_path_mode: UnapprovedPathMode::Allow,
            ..Default::default()
        })
        .init_asset::<PlanarGaussian3d>()
        .add_plugins(IoPlugin);
    let handle: Handle<GaussianScene> = app
        .world()
        .resource::<AssetServer>()
        .load(path.file_name().unwrap().to_string_lossy().into_owned());
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        app.update();
        if let Some((load, dependency, recursive)) = app
            .world()
            .resource::<AssetServer>()
            .get_load_states(&handle)
        {
            match (&load, &dependency, &recursive) {
                (LoadState::Failed(error), _, _)
                | (_, DependencyLoadState::Failed(error), _)
                | (_, _, RecursiveDependencyLoadState::Failed(error)) => {
                    return Err(format!("GLB decoder: {error}").into());
                }
                (LoadState::Loaded, _, RecursiveDependencyLoadState::Loaded) => break,
                _ => {}
            }
        }
        if Instant::now() > deadline {
            return Err("GLB decoder timed out".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let bundle = {
        let scene = app
            .world()
            .resource::<Assets<GaussianScene>>()
            .get(&handle)
            .ok_or("missing loaded GLB scene")?;
        if scene.bundles.len() != 1 {
            return Err("GLB decoder returned more than one ordered stream".into());
        }
        scene.bundles[0].clone()
    };
    let cloud = app
        .world_mut()
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .remove(bundle.cloud.id())
        .ok_or("GLB cloud missing")?;
    if cloud.len() as u64 != count {
        return Err("decoded GLB count disagrees with admitted count".into());
    }
    Ok(LoadedSource {
        cloud,
        transform: bundle.transform,
        color_space: bundle.settings.color_space,
    })
}

// PLY emits actual RGB SH coefficients; the in-memory coefficient array may
// have vec4 padding (notably SH0). The twelve other fields include visibility.
const PLY_RECORD_BYTES: u64 = (12 + SH_CHANNELS * SH_COEFF_COUNT_PER_CHANNEL) as u64 * 4;

fn write_ply_header(writer: &mut impl Write, count: u64) -> std::io::Result<()> {
    writeln!(
        writer,
        "ply\nformat binary_little_endian 1.0\nelement vertex {}",
        count
    )?;
    for property in ["x", "y", "z", "visibility", "f_dc_0", "f_dc_1", "f_dc_2"] {
        writeln!(writer, "property float {property}")?;
    }
    let rest_per_channel = SH_COEFF_COUNT_PER_CHANNEL - 1;
    for coefficient in 0..rest_per_channel * SH_CHANNELS {
        writeln!(writer, "property float f_rest_{coefficient}")?;
    }
    for property in [
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        writeln!(writer, "property float {property}")?;
    }
    writeln!(writer, "end_header")?;
    Ok(())
}

fn write_ply_gaussian(writer: &mut impl Write, gaussian: &Gaussian3d) -> std::io::Result<()> {
    let rest_per_channel = SH_COEFF_COUNT_PER_CHANNEL - 1;
    let mut write = |value: f32| writer.write_all(&value.to_le_bytes());
    for value in gaussian.position_visibility.position {
        write(value)?;
    }
    write(gaussian.position_visibility.visibility)?;
    for channel in 0..SH_CHANNELS {
        write(gaussian.spherical_harmonic.coefficients[channel])?;
    }
    for channel in 0..SH_CHANNELS {
        for coefficient in 1..=rest_per_channel {
            write(gaussian.spherical_harmonic.coefficients[coefficient * SH_CHANNELS + channel])?;
        }
    }
    let opacity = gaussian.scale_opacity.opacity;
    // Infinite logits round-trip exact endpoints through the production
    // sigmoid. Intermediate finite values use a stable f64 logit.
    let logit = (f64::from(opacity) / (1.0 - f64::from(opacity))).ln() as f32;
    write(logit)?;
    for scale in gaussian.scale_opacity.scale {
        write(f64::from(scale).ln() as f32)?;
    }
    for rotation in gaussian.rotation.rotation {
        write(rotation)?;
    }
    Ok(())
}

/// Decode a bounded single-primitive GLB into the external builder's PLY input.
/// Save an adjacent `.metadata.json` sidecar for applying the exact same instance
/// transform/color space to flat and package captures. The source is not modified.
pub fn convert_glb_to_ply(input: &Path, output: &Path, max_gaussians: u64) -> CaptureResult<()> {
    if max_gaussians == 0 {
        return Err("max_gaussians must be positive".into());
    }
    let source = load_glb(input, max_gaussians)?;
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    write_ply_header(&mut writer, source.cloud.len() as u64)?;
    for gaussian in source.cloud.iter() {
        write_ply_gaussian(&mut writer, &gaussian)?;
    }
    writer.flush()?;
    let mut original = source.cloud.iter();
    let mut maximum_error = 0.0f64;
    stream_ply_3d_with_sh_compatibility(
        &mut BufReader::new(File::open(output)?),
        1024,
        PlyShCompatibility::RequireRepresentable,
        |batch| {
            for roundtrip in batch {
                let authored = original
                    .next()
                    .ok_or_else(|| std::io::Error::other("converted PLY has excess records"))?;
                let authored: &[f32] = bytemuck::cast_slice(std::slice::from_ref(&authored));
                let decoded: &[f32] = bytemuck::cast_slice(std::slice::from_ref(roundtrip));
                for (&a, &b) in authored.iter().zip(decoded) {
                    maximum_error = maximum_error.max((f64::from(a) - f64::from(b)).abs());
                }
            }
            Ok(())
        },
    )?;
    if original.next().is_some() {
        return Err("converted PLY lost records".into());
    }
    let metadata = SourceMetadata {
        original_sha256: hash_file(input)?,
        gaussian_count: source.cloud.len() as u64,
        world_from_local: source.transform.to_matrix().to_cols_array(),
        color_space: source.color_space,
        converted_ply_sha256: Some(hash_file(output)?),
        maximum_scalar_roundtrip_error: Some(maximum_error),
    };
    let metadata_path = output.with_extension("metadata.json");
    let mut metadata_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(metadata_path)?;
    serde_json::to_writer_pretty(&mut metadata_file, &metadata)?;
    metadata_file.write_all(b"\n")?;
    println!(
        "converted {} Gaussians; maximum scalar roundtrip error {maximum_error:.9e}",
        source.cloud.len()
    );
    Ok(())
}
