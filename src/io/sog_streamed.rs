//! Decoder for streamed SOG datasets: a `lod-meta.json` index plus one unbundled SOG
//! dataset per chunk directory. Also used for the `.ssog` zip packaging, where the same
//! files sit inside a zip archive.
//!
//! Spec: <https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/>
//!
//! Scope: every leaf is loaded at LOD level 0 (plus the environment, if present) into a
//! single cloud. Streaming and camera-distance LOD selection are not implemented.

use std::{
    collections::HashMap,
    io,
    path::{Component, Path},
};

use bevy_interleave::prelude::Planar;
use serde::Deserialize;

use crate::{
    gaussian::formats::planar_3d::{Gaussian3d, PlanarGaussian3d},
    io::sog::decode_sog,
};

#[derive(Deserialize)]
struct LodMeta {
    /// Pre-versioned datasets omit this field.
    #[serde(default)]
    version: Option<u32>,
    #[serde(default)]
    environment: Option<String>,
    filenames: Vec<String>,
    tree: Node,
}

#[derive(Deserialize)]
struct Node {
    #[serde(default)]
    children: Option<Vec<Node>>,
    #[serde(default)]
    lods: Option<HashMap<String, LodRun>>,
}

#[derive(Clone, Copy, Deserialize)]
struct LodRun {
    /// Index into `filenames`.
    file: usize,
    offset: usize,
    count: usize,
}

fn invalid(err: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err.to_string())
}

/// Validates an index-provided relative path so it cannot escape the dataset.
fn checked_relative(name: &str) -> io::Result<&str> {
    let safe = !name.is_empty()
        && Path::new(name)
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
    if safe {
        Ok(name)
    } else {
        Err(invalid("streamed SOG paths must be relative and stay in the dataset"))
    }
}

fn collect_level0_runs(node: &Node, out: &mut Vec<LodRun>) -> io::Result<()> {
    match (&node.children, &node.lods) {
        (Some(children), None) => {
            for child in children {
                collect_level0_runs(child, out)?;
            }
            Ok(())
        }
        (None, Some(lods)) => {
            if let Some(run) = lods.get("0") {
                out.push(*run);
            }
            Ok(())
        }
        _ => Err(invalid("streamed SOG nodes must be either interior or leaf")),
    }
}

/// Decodes the SOG dataset whose `meta.json` is at `meta_path`. Sibling files are resolved
/// relative to that path's directory.
fn decode_dataset(
    read: &mut impl FnMut(&str) -> io::Result<Vec<u8>>,
    meta_path: &str,
) -> io::Result<Vec<Gaussian3d>> {
    let meta_path = checked_relative(meta_path)?;
    let prefix = match meta_path.rsplit_once('/') {
        Some((dir, _)) => format!("{dir}/"),
        None => String::new(),
    };

    let cloud = decode_sog(|file| {
        let name = if file == "meta.json" {
            meta_path.to_string()
        } else {
            format!("{prefix}{file}")
        };
        read(&name)
    })?;

    Ok(cloud.iter().collect())
}

/// Decodes a streamed SOG dataset. `read` returns the bytes of a file named relative to the
/// dataset root, for example `lod-meta.json` or `0_0/means_l.webp`.
pub fn decode_streamed(
    mut read: impl FnMut(&str) -> io::Result<Vec<u8>>,
) -> io::Result<PlanarGaussian3d> {
    let index: LodMeta = serde_json::from_slice(&read("lod-meta.json")?).map_err(invalid)?;

    let version = index.version.unwrap_or(1);
    if version > 1 {
        return Err(invalid(format!("unsupported streamed SOG version {version}")));
    }

    let mut runs = Vec::new();
    collect_level0_runs(&index.tree, &mut runs)?;

    // a chunk is decoded on first use and dropped after its last run, so only the chunks
    // currently in use are held in memory
    let mut remaining: HashMap<usize, usize> = HashMap::new();
    for run in &runs {
        *remaining.entry(run.file).or_default() += 1;
    }

    let mut chunks: HashMap<usize, Vec<Gaussian3d>> = HashMap::new();
    let mut gaussians = Vec::new();

    for run in runs {
        if !chunks.contains_key(&run.file) {
            let name = index
                .filenames
                .get(run.file)
                .ok_or_else(|| invalid("streamed SOG run references a missing chunk"))?;
            chunks.insert(run.file, decode_dataset(&mut read, name)?);
        }

        let chunk = &chunks[&run.file];
        let end = run
            .offset
            .checked_add(run.count)
            .filter(|&end| end <= chunk.len())
            .ok_or_else(|| invalid("streamed SOG run exceeds its chunk"))?;
        gaussians.extend_from_slice(&chunk[run.offset..end]);

        let left = remaining.get_mut(&run.file).expect("counted above");
        *left -= 1;
        if *left == 0 {
            chunks.remove(&run.file);
        }
    }

    if let Some(environment) = &index.environment {
        gaussians.extend(decode_dataset(&mut read, environment)?);
    }

    // pad to a multiple of 32, matching the PLY loader
    let pad = 32 - (gaussians.len() % 32);
    gaussians.extend(std::iter::repeat_n(Gaussian3d::default(), pad));

    Ok(PlanarGaussian3d::from_interleaved(gaussians))
}

/// Decodes a streamed SOG dataset stored as a directory containing `lod-meta.json`.
pub fn parse_streamed_sog_dir(dir: &Path) -> io::Result<PlanarGaussian3d> {
    decode_streamed(|name| std::fs::read(dir.join(name)))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::io::sog::tests::{build_zip, sample_scene_files};

    use super::*;

    const TOLERANCE: f32 = 1e-3;

    /// The streamed index with two chunks: leaf 0 takes splat 0 of chunk 0, and leaf 1
    /// takes splat 1 of chunk 1.
    fn two_leaf_index() -> serde_json::Value {
        serde_json::json!({
            "version": 1,
            "count": 2,
            "counts": [2],
            "lodLevels": 1,
            "filenames": ["0_0/meta.json", "1_0/meta.json"],
            "tree": {
                "bound": { "min": [-2, -2, -2], "max": [2, 2, 2] },
                "children": [
                    {
                        "bound": { "min": [-2, -2, -2], "max": [0, 2, 2] },
                        "lods": { "0": { "file": 0, "offset": 0, "count": 1 } }
                    },
                    {
                        "bound": { "min": [0, -2, -2], "max": [2, 2, 2] },
                        "lods": { "0": { "file": 1, "offset": 1, "count": 1 } }
                    }
                ]
            }
        })
    }

    /// The dataset as name → bytes, with the index and both chunks.
    fn dataset_files(index: &serde_json::Value) -> Vec<(String, Vec<u8>)> {
        let mut files = vec![(
            "lod-meta.json".to_string(),
            serde_json::to_vec(index).unwrap(),
        )];
        for chunk in ["0_0", "1_0"] {
            for (file, data) in sample_scene_files(None) {
                files.push((format!("{chunk}/{file}"), data));
            }
        }
        files
    }

    fn write_dataset(name: &str, index: &serde_json::Value) -> PathBuf {
        let root = std::env::temp_dir().join(format!("sog_streamed_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&root);
        for (file, data) in dataset_files(index) {
            let path = root.join(&file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, data).unwrap();
        }
        root
    }

    fn assert_two_leaf_cloud(cloud: &PlanarGaussian3d) {
        let gaussians: Vec<Gaussian3d> = cloud.iter().collect();

        // padded to a multiple of 32
        assert_eq!(gaussians.len(), 32);

        // leaf 0 takes splat 0 of chunk 0 (+e-1), leaf 1 takes splat 1 of chunk 1 (-(e-1))
        let expected = 1.0f32.exp() - 1.0;
        let first = gaussians[0].position_visibility.position[0];
        let second = gaussians[1].position_visibility.position[0];
        assert!((first - expected).abs() <= TOLERANCE, "first = {first}");
        assert!((second + expected).abs() <= TOLERANCE, "second = {second}");
    }

    #[test]
    fn loads_level_zero_runs_from_directory_in_tree_order() {
        let root = write_dataset("runs", &two_leaf_index());
        let cloud = parse_streamed_sog_dir(&root).expect("decode streamed sog");
        assert_two_leaf_cloud(&cloud);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn loads_level_zero_runs_from_zip_archive() {
        let files = dataset_files(&two_leaf_index());
        let borrowed: Vec<(&str, Vec<u8>)> = files
            .iter()
            .map(|(name, data)| (name.as_str(), data.clone()))
            .collect();
        let archive = build_zip(&borrowed);

        let cloud = crate::io::sog::parse_sog_archive(&archive).expect("decode ssog zip");
        assert_two_leaf_cloud(&cloud);
    }

    #[test]
    fn rejects_newer_major_version() {
        let mut index = two_leaf_index();
        index["version"] = serde_json::json!(2);
        let root = write_dataset("newer_version", &index);

        let err = match parse_streamed_sog_dir(&root) {
            Ok(_) => panic!("version 2 must be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("unsupported streamed SOG version"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_paths_that_escape_the_dataset() {
        let mut index = two_leaf_index();
        index["filenames"] = serde_json::json!(["../outside/meta.json", "1_0/meta.json"]);
        let root = write_dataset("traversal", &index);

        assert!(parse_streamed_sog_dir(&root).is_err());

        let _ = std::fs::remove_dir_all(&root);
    }
    /// Loads a real streamed `.ssog` file. Not run by default; see the command in the
    /// summary. Example:
    ///   cargo test --lib loads_sample_ssog -- --ignored --nocapture
    #[test]
    #[ignore]
    fn loads_sample_ssog() {
        let path = std::env::var("SSOG_SAMPLE")
            .unwrap_or_else(|_| "assets/Apartment Pool.ssog".to_string());
        let bytes = std::fs::read(&path).expect("read sample file");

        let started = std::time::Instant::now();
        let cloud = crate::io::sog::parse_sog_archive(&bytes).expect("decode sample ssog");
        let elapsed = started.elapsed();

        let gaussians: Vec<Gaussian3d> = cloud.iter().collect();

        // padding splats are all-zero defaults; every other splat must be well-formed
        let is_padding = |g: &Gaussian3d| g.scale_opacity.scale.iter().all(|v| *v == 0.0);
        let padding = gaussians.iter().filter(|g| is_padding(g)).count();
        let finite = gaussians.iter().filter(|g| !is_padding(g)).all(|g| {
            g.position_visibility.position.iter().all(|v| v.is_finite())
                && g.scale_opacity.scale.iter().all(|v| v.is_finite() && *v > 0.0)
                && g.scale_opacity.opacity.is_finite()
        });

        println!("file: {path}");
        println!("gaussians (padded): {}", gaussians.len());
        println!("decode time: {elapsed:.2?}");
        println!("first position: {:?}", gaussians[0].position_visibility.position);
        println!("padding splats: {padding}");
        println!("all finite with positive scales: {finite}");

        assert!(!gaussians.is_empty());
        assert_eq!(gaussians.len() % 32, 0);
        assert!(padding < 32, "at most 31 padding splats expected");
        assert!(finite, "decoded values must be finite, with positive scales");
    }
}
