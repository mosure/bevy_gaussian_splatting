//! Decoder for SOG (Self-Organizing Gaussians) version 2 scenes.
//!
//! Spec: <https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/>
//!
//! Only the 3D Gaussian model is supported. The `2dgs` model is rejected.

use std::{
    f32::consts::SQRT_2,
    io::{self, Cursor, Read},
    path::Path,
};

use bevy_interleave::prelude::Planar;
use serde::Deserialize;

use crate::{
    gaussian::formats::planar_3d::{Gaussian3d, PlanarGaussian3d},
    material::spherical_harmonics::{SH_CHANNELS, SH_COEFF_COUNT},
};

// meta.json

#[derive(Deserialize)]
struct Meta {
    version: u32,
    count: usize,
    #[serde(default)]
    model: Option<String>,
    means: MeansMeta,
    scales: CodebookMeta,
    quats: FilesMeta,
    sh0: CodebookMeta,
    #[serde(rename = "shN")]
    sh_n: Option<ShNMeta>,
}

#[derive(Deserialize)]
struct MeansMeta {
    mins: [f32; 3],
    maxs: [f32; 3],
    files: Vec<String>,
}

#[derive(Deserialize)]
struct CodebookMeta {
    codebook: Vec<f32>,
    files: Vec<String>,
}

#[derive(Deserialize)]
struct FilesMeta {
    files: Vec<String>,
}

#[derive(Deserialize)]
struct ShNMeta {
    count: usize,
    bands: u32,
    codebook: Vec<f32>,
    files: Vec<String>,
}

fn invalid(err: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err.to_string())
}

/// Number of higher-order (AC) coefficients per color channel for a SOG band count.
fn ac_coefficients(bands: u32) -> io::Result<usize> {
    match bands {
        1 => Ok(3),
        2 => Ok(8),
        3 => Ok(15),
        _ => Err(invalid("SOG shN bands must be 1, 2, or 3")),
    }
}

/// Files named by meta.json must be plain names inside the scene, never paths.
fn is_plain_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['/', '\\']) && name != "." && name != ".."
}

fn file_name(files: &[String], index: usize) -> io::Result<&str> {
    files
        .get(index)
        .map(String::as_str)
        .ok_or_else(|| invalid("SOG meta.json is missing a required file entry"))
}

fn load_image(
    read: &mut impl FnMut(&str) -> io::Result<Vec<u8>>,
    name: &str,
) -> io::Result<image::RgbaImage> {
    let bytes = read(name)?;
    image::load_from_memory(&bytes)
        .map(|img| img.to_rgba8())
        .map_err(invalid)
}

fn lookup(codebook: &[f32], index: u8) -> io::Result<f32> {
    codebook
        .get(index as usize)
        .copied()
        .ok_or_else(|| invalid("SOG codebook index out of range"))
}

/// Inverts the log-domain position transform: `sign(n) * (exp(|n|) - 1)`.
fn unlog(n: f32) -> f32 {
    n.signum() * (n.abs().exp() - 1.0)
}

/// Decodes a smallest-three quaternion from its RGBA texel. Output is `[w, x, y, z]`.
fn decode_quat(texel: [u8; 4]) -> io::Result<[f32; 4]> {
    if !(252..=255).contains(&texel[3]) {
        return Err(invalid("SOG quaternion has a reserved mode value"));
    }
    let mode = texel[3] - 252;

    let comp = |c: u8| (c as f32 / 255.0 - 0.5) * 2.0 / SQRT_2;
    let (a, b, c) = (comp(texel[0]), comp(texel[1]), comp(texel[2]));
    let d = (1.0 - (a * a + b * b + c * c)).max(0.0).sqrt();

    let q = match mode {
        0 => [d, a, b, c],
        1 => [a, d, b, c],
        2 => [a, b, d, c],
        _ => [a, b, c, d],
    };

    let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        Ok(q.map(|v| v / norm))
    } else {
        Ok(q)
    }
}

/// Decodes a SOG scene. `read` returns the bytes of a named file in the scene, which is
/// how both the zip and directory layouts are supported.
pub fn decode_sog(
    mut read: impl FnMut(&str) -> io::Result<Vec<u8>>,
) -> io::Result<PlanarGaussian3d> {
    let mut read = move |name: &str| {
        if !is_plain_name(name) {
            return Err(invalid("SOG file references must be plain file names"));
        }
        read(name)
    };

    let meta: Meta = serde_json::from_slice(&read("meta.json")?).map_err(invalid)?;

    if meta.version != 2 {
        return Err(invalid(format!("unsupported SOG version {}", meta.version)));
    }
    if matches!(meta.model.as_deref(), Some("2dgs")) {
        return Err(invalid("SOG 2dgs models are not supported"));
    }

    let means_l = load_image(&mut read, file_name(&meta.means.files, 0)?)?;
    let means_u = load_image(&mut read, file_name(&meta.means.files, 1)?)?;
    let scales = load_image(&mut read, file_name(&meta.scales.files, 0)?)?;
    let quats = load_image(&mut read, file_name(&meta.quats.files, 0)?)?;
    let sh0 = load_image(&mut read, file_name(&meta.sh0.files, 0)?)?;

    let (width, height) = means_l.dimensions();
    for image in [&means_u, &scales, &quats, &sh0] {
        if image.dimensions() != (width, height) {
            return Err(invalid(
                "SOG per-Gaussian images have mismatched dimensions",
            ));
        }
    }
    if meta.count > width as usize * height as usize {
        return Err(invalid("SOG count exceeds the image capacity"));
    }

    let palette = match &meta.sh_n {
        Some(sh_n) => {
            let coeffs = ac_coefficients(sh_n.bands)?;
            let centroids = load_image(&mut read, file_name(&sh_n.files, 0)?)?;
            let labels = load_image(&mut read, file_name(&sh_n.files, 1)?)?;
            if labels.dimensions() != (width, height) {
                return Err(invalid("SOG shN labels have mismatched dimensions"));
            }
            if centroids.width() as usize != 64 * coeffs {
                return Err(invalid(
                    "SOG shN palette width does not match its band count",
                ));
            }
            Some((sh_n, coeffs, centroids, labels))
        }
        None => None,
    };

    let mut gaussians = Vec::with_capacity(meta.count + 32);

    for i in 0..meta.count {
        let x = (i as u32) % width;
        let y = (i as u32) / width;

        let mut gaussian = Gaussian3d::default();

        // position: 16 bits per axis, split across the two means images
        let low = means_l.get_pixel(x, y).0;
        let high = means_u.get_pixel(x, y).0;
        for axis in 0..3 {
            let q = ((high[axis] as u32) << 8) | low[axis] as u32;
            let t = q as f32 / 65535.0;
            let n = meta.means.mins[axis] + (meta.means.maxs[axis] - meta.means.mins[axis]) * t;
            gaussian.position_visibility.position[axis] = unlog(n);
        }

        // orientation
        gaussian.rotation.rotation = decode_quat(quats.get_pixel(x, y).0)?;

        // scale: codebook holds log-domain values
        let scale_texel = scales.get_pixel(x, y).0;
        for (scale, code) in gaussian.scale_opacity.scale.iter_mut().zip(scale_texel) {
            *scale = lookup(&meta.scales.codebook, code)?.exp();
        }

        // DC color (SH coefficient 0) and linear opacity
        let dc_texel = sh0.get_pixel(x, y).0;
        for (channel, &code) in dc_texel.iter().enumerate().take(SH_CHANNELS) {
            let value = lookup(&meta.sh0.codebook, code)?;
            gaussian.spherical_harmonic.set(channel, value);
        }
        gaussian.scale_opacity.opacity = dc_texel[3] as f32 / 255.0;

        // higher-order SH from the palette
        if let Some((sh_n, coeffs, centroids, labels)) = &palette {
            let label_texel = labels.get_pixel(x, y).0;
            let label = label_texel[0] as usize + ((label_texel[1] as usize) << 8);
            if label >= sh_n.count {
                return Err(invalid("SOG shN label is out of range"));
            }

            let u_base = (label % 64) * coeffs;
            let v = (label / 64) as u32;
            if v >= centroids.height() {
                return Err(invalid("SOG shN palette is too short for its label count"));
            }

            for c in 0..*coeffs {
                let texel = centroids.get_pixel((u_base + c) as u32, v).0;
                for (channel, &code) in texel.iter().enumerate().take(SH_CHANNELS) {
                    // AC coefficient `c` is SH coefficient `c + 1`
                    let index = (c + 1) * SH_CHANNELS + channel;
                    if index < SH_COEFF_COUNT {
                        let value = lookup(&sh_n.codebook, code)?;
                        gaussian.spherical_harmonic.set(index, value);
                    }
                }
            }
        }

        gaussians.push(gaussian);
    }

    // pad to a multiple of 32, matching the PLY loader
    let pad = 32 - (gaussians.len() % 32);
    gaussians.extend(std::iter::repeat_n(Gaussian3d::default(), pad));

    Ok(PlanarGaussian3d::from_interleaved(gaussians))
}

fn read_zip_entry(archive: &mut zip::ZipArchive<Cursor<&[u8]>>, name: &str) -> io::Result<Vec<u8>> {
    let mut file = archive.by_name(name).map_err(invalid)?;
    let mut buf = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Decodes a single-archive `.sog` file (a zip with the scene files at its root).
pub fn parse_sog_zip(bytes: &[u8]) -> io::Result<PlanarGaussian3d> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(invalid)?;
    decode_sog(|name| read_zip_entry(&mut archive, name))
}

/// Decodes a zip archive that holds either a single SOG scene (`meta.json` at the root) or
/// a streamed SOG dataset (`lod-meta.json` at the root). Used for `.sog` and `.ssog` files.
pub fn parse_sog_archive(bytes: &[u8]) -> io::Result<PlanarGaussian3d> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(invalid)?;

    let is_streamed = archive.by_name("lod-meta.json").is_ok();
    if is_streamed {
        #[cfg(feature = "io_sog_streamed")]
        {
            crate::io::sog_streamed::decode_streamed(|name| read_zip_entry(&mut archive, name))
        }

        #[cfg(not(feature = "io_sog_streamed"))]
        {
            Err(io::Error::other(
                "streamed SOG support not enabled, enable with io_sog_streamed feature",
            ))
        }
    } else {
        decode_sog(|name| read_zip_entry(&mut archive, name))
    }
}

/// Decodes a SOG scene stored as a directory containing `meta.json` and the image files.
pub fn parse_sog_dir(dir: &Path) -> io::Result<PlanarGaussian3d> {
    decode_sog(|name| std::fs::read(dir.join(name)))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::io::Write;

    use image::{ExtendedColorType, ImageEncoder, codecs::webp::WebPEncoder};

    use super::*;

    const TOLERANCE: f32 = 1e-3;

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    pub(crate) fn encode_lossless_webp(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        WebPEncoder::new_lossless(&mut out)
            .write_image(rgba, width, height, ExtendedColorType::Rgba8)
            .expect("encode webp");
        out
    }

    pub(crate) fn build_zip(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, data) in files {
            writer
                .start_file(*name, zip::write::SimpleFileOptions::default())
                .expect("start file");
            writer.write_all(data).expect("write file");
        }
        writer.finish().expect("finish zip").into_inner()
    }

    /// A 2-Gaussian scene: no SH palette unless `sh_bands` is given.
    pub(crate) fn sample_scene(sh_bands: Option<u32>) -> Vec<u8> {
        build_zip(&sample_scene_files(sh_bands))
    }

    /// The unbundled files of the sample scene, named as in a SOG dataset directory.
    pub(crate) fn sample_scene_files(sh_bands: Option<u32>) -> Vec<(&'static str, Vec<u8>)> {
        // Two pixels in a 2x1 image. Gaussian 0 is at the far corner of the means range,
        // gaussian 1 at the near corner.
        let means_l = [255, 0, 0, 255, 0, 0, 0, 255];
        let means_u = [255, 128, 128, 255, 0, 0, 0, 255];
        let quats = [128, 128, 236, 252, 128, 128, 128, 252];
        let scales = [10, 20, 10, 255, 20, 20, 20, 255];
        let sh0 = [50, 50, 60, 255, 60, 60, 60, 128];

        let mut scale_codebook = vec![0.0f32; 256];
        scale_codebook[10] = 0.1f32.ln();
        scale_codebook[20] = 0.5f32.ln();

        let mut dc_codebook = vec![0.0f32; 256];
        dc_codebook[50] = 0.25;
        dc_codebook[60] = -0.5;

        let mut meta = serde_json::json!({
            "version": 2,
            "count": 2,
            "means": {
                "mins": [-1.0, -1.0, -1.0],
                "maxs": [1.0, 1.0, 1.0],
                "files": ["means_l.webp", "means_u.webp"],
            },
            "scales": { "codebook": scale_codebook, "files": ["scales.webp"] },
            "quats": { "files": ["quats.webp"] },
            "sh0": { "codebook": dc_codebook, "files": ["sh0.webp"] },
        });

        let mut files = vec![
            ("means_l.webp", encode_lossless_webp(2, 1, &means_l)),
            ("means_u.webp", encode_lossless_webp(2, 1, &means_u)),
            ("scales.webp", encode_lossless_webp(2, 1, &scales)),
            ("quats.webp", encode_lossless_webp(2, 1, &quats)),
            ("sh0.webp", encode_lossless_webp(2, 1, &sh0)),
        ];

        if let Some(bands) = sh_bands {
            let coeffs = ac_coefficients(bands).unwrap();
            let mut sh_codebook = vec![0.0f32; 256];
            sh_codebook[7] = 0.1;
            sh_codebook[8] = 0.2;
            sh_codebook[9] = 0.3;

            // palette entry 0: (7, 8, 9); entry 1 starts at column `coeffs`: all 9s
            let width = 64 * coeffs;
            let mut palette = vec![0u8; width * 4];
            palette[0..4].copy_from_slice(&[7, 8, 9, 255]);
            palette[coeffs * 4..coeffs * 4 + 4].copy_from_slice(&[9, 9, 9, 255]);

            // label 0 for gaussian 0, label 1 for gaussian 1
            let labels = [0, 0, 0, 255, 1, 0, 0, 255];

            meta["shN"] = serde_json::json!({
                "count": 2,
                "bands": bands,
                "codebook": sh_codebook,
                "files": ["shN_centroids.webp", "shN_labels.webp"],
            });
            files.push((
                "shN_centroids.webp",
                encode_lossless_webp(width as u32, 1, &palette),
            ));
            files.push(("shN_labels.webp", encode_lossless_webp(2, 1, &labels)));
        }

        files.push(("meta.json", serde_json::to_vec(&meta).unwrap()));
        files
    }

    #[test]
    fn decodes_positions_scales_rotations_opacity_and_dc_color() {
        let cloud = parse_sog_zip(&sample_scene(None)).expect("decode sog");
        let gaussians: Vec<Gaussian3d> = cloud.iter().collect();

        // padded to a multiple of 32
        assert_eq!(gaussians.len(), 32);

        let g0 = &gaussians[0];
        assert_close(
            g0.position_visibility.position[0],
            1.0f32.exp() - 1.0,
            TOLERANCE,
        );
        assert_close(g0.position_visibility.position[1], 0.0, 1e-3);
        assert_close(g0.position_visibility.position[2], 0.0, 1e-3);

        let g1 = &gaussians[1];
        for axis in g1.position_visibility.position {
            assert_close(axis, -(1.0f32.exp() - 1.0), TOLERANCE);
        }

        // scales are exponentiated from the log codebook
        assert_close(g0.scale_opacity.scale[0], 0.1, TOLERANCE);
        assert_close(g0.scale_opacity.scale[1], 0.5, TOLERANCE);
        assert_close(g1.scale_opacity.scale[0], 0.5, TOLERANCE);

        // opacity is linear
        assert_close(g0.scale_opacity.opacity, 1.0, TOLERANCE);
        assert_close(g1.scale_opacity.opacity, 128.0 / 255.0, TOLERANCE);

        // DC color
        assert_close(g0.spherical_harmonic.coefficients[0], 0.25, TOLERANCE);
        assert_close(g0.spherical_harmonic.coefficients[2], -0.5, TOLERANCE);
        assert_close(g1.spherical_harmonic.coefficients[0], -0.5, TOLERANCE);

        // identity-ish quaternion for gaussian 1: w = 1, rest ~ 0
        assert_close(g1.rotation.rotation[0], 1.0, 0.01);
        assert_close(g1.rotation.rotation[1], 0.0, 0.01);

        // gaussian 0: w = 0.8, z = 0.6 (mode 0, omitted w is largest)
        assert_close(g0.rotation.rotation[0], 0.8, 0.01);
        assert_close(g0.rotation.rotation[3], 0.6, 0.01);
    }

    #[test]
    fn decodes_band_one_palette_into_interleaved_sh_layout() {
        let cloud = parse_sog_zip(&sample_scene(Some(1))).expect("decode sog");
        let gaussians: Vec<Gaussian3d> = cloud.iter().collect();

        // gaussian 0 uses palette entry 0: coefficient 1 holds (0.1, 0.2, 0.3)
        let g0 = &gaussians[0].spherical_harmonic.coefficients;
        assert_close(g0[3], 0.1, TOLERANCE);
        assert_close(g0[4], 0.2, TOLERANCE);
        assert_close(g0[5], 0.3, TOLERANCE);

        // gaussian 1 uses palette entry 1, which is all 0.3
        let g1 = &gaussians[1].spherical_harmonic.coefficients;
        assert_close(g1[3], 0.3, TOLERANCE);
        assert_close(g1[4], 0.3, TOLERANCE);
        assert_close(g1[5], 0.3, TOLERANCE);
    }

    #[test]
    fn rejects_unsupported_version() {
        let files = {
            let meta = serde_json::json!({ "version": 3, "count": 0, "means": {"mins":[0,0,0],"maxs":[0,0,0],"files":["a","b"]}, "scales": {"codebook":[], "files":["c"]}, "quats": {"files":["d"]}, "sh0": {"codebook":[], "files":["e"]} });
            build_zip(&[("meta.json", serde_json::to_vec(&meta).unwrap())])
        };
        match parse_sog_zip(&files) {
            Ok(_) => panic!("version 3 must be rejected"),
            Err(err) => assert!(err.to_string().contains("unsupported SOG version")),
        }
    }

    #[test]
    fn rejects_path_traversal_in_file_references() {
        let meta = serde_json::json!({
            "version": 2, "count": 0,
            "means": {"mins":[0,0,0],"maxs":[0,0,0],"files":["../secret.webp","b"]},
            "scales": {"codebook":[], "files":["c"]}, "quats": {"files":["d"]}, "sh0": {"codebook":[], "files":["e"]}
        });
        let zip_bytes = build_zip(&[("meta.json", serde_json::to_vec(&meta).unwrap())]);
        assert!(
            parse_sog_zip(&zip_bytes).is_err(),
            "traversal must be rejected"
        );
    }
}
