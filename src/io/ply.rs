use std::io::{self, BufRead};

use bevy_interleave::prelude::Planar;
use ply_rs::{
    parser::Parser,
    ply::{ElementDef, Encoding, Property, PropertyAccess, PropertyType, ScalarType},
};

use crate::{
    gaussian::formats::{
        planar_3d::{Gaussian3d, PlanarGaussian3d},
        planar_4d::{Gaussian4d, PlanarGaussian4d},
    },
    material::{
        spherical_harmonics::{SH_CHANNELS, SH_COEFF_COUNT, SH_COEFF_COUNT_PER_CHANNEL},
        spherindrical_harmonics::SH_4D_COEFF_COUNT,
    },
};

pub const MAX_SIZE_VARIANCE: f32 = 4.0;
pub const DEFAULT_STREAM_BATCH_SIZE: usize = 16 * 1024;
/// Hard ceiling for the temporary interleaved batch used by [`stream_ply_3d`].
///
/// This is intentionally independent of the caller's source-count limit: a malformed or
/// accidentally enormous batch argument must not turn the first allocation into an OOM abort.
pub const MAX_STREAM_BATCH_ALLOCATION_BYTES: usize = 256 * 1024 * 1024;

/// Controls how 3DGS PLY spherical-harmonic properties are matched to the
/// crate's compiled SH degree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlyShCompatibility {
    /// Read the representable coefficients from each color channel and ignore
    /// higher-order `f_rest_*` coefficients.
    #[default]
    AllowTruncation,
    /// Reject input that contains coefficients above the compiled SH degree.
    /// Offline LoD package construction uses this by default so exact leaves
    /// match the package's declared SH ABI.
    RequireRepresentable,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlyStreamSummary {
    /// Number of source vertices, excluding any storage-alignment padding.
    pub logical_count: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlyShLayout {
    source_rest_coefficients_per_channel: usize,
}

impl PlyStreamSummary {
    pub fn is_empty(self) -> bool {
        self.logical_count == 0
    }

    fn include(&mut self) {
        self.logical_count += 1;
    }
}

impl PropertyAccess for Gaussian3d {
    fn new() -> Self {
        Gaussian3d::default()
    }

    fn set_property(&mut self, key: String, property: Property) {
        match (key.as_ref(), property) {
            ("x", Property::Float(v)) => self.position_visibility.position[0] = v,
            ("y", Property::Float(v)) => self.position_visibility.position[1] = v,
            ("z", Property::Float(v)) => self.position_visibility.position[2] = v,
            ("visibility", Property::Float(v)) => self.position_visibility.visibility = v,
            ("f_dc_0", Property::Float(v)) => self.spherical_harmonic.set(0, v),
            ("f_dc_1", Property::Float(v)) => self.spherical_harmonic.set(1, v),
            ("f_dc_2", Property::Float(v)) => self.spherical_harmonic.set(2, v),
            ("scale_0", Property::Float(v)) => self.scale_opacity.scale[0] = v,
            ("scale_1", Property::Float(v)) => self.scale_opacity.scale[1] = v,
            ("scale_2", Property::Float(v)) => self.scale_opacity.scale[2] = v,
            ("opacity", Property::Float(v)) => {
                self.scale_opacity.opacity = 1.0 / (1.0 + (-v).exp())
            }
            ("rot_0", Property::Float(v)) => self.rotation.rotation[0] = v,
            ("rot_1", Property::Float(v)) => self.rotation.rotation[1] = v,
            ("rot_2", Property::Float(v)) => self.rotation.rotation[2] = v,
            ("rot_3", Property::Float(v)) => self.rotation.rotation[3] = v,
            (_, Property::Float(v)) if key.starts_with("f_rest_") => {
                let Ok(i) = key[7..].parse::<usize>() else {
                    // Header validation reports malformed names before payload
                    // parsing. Keep PropertyAccess panic-free for direct users.
                    return;
                };

                // interleaved
                // if (i + 3) < SH_COEFF_COUNT {
                //     self.spherical_harmonic.coefficients[i + 3] = v;
                // }

                // planar
                let rest_coefficients_per_channel = SH_COEFF_COUNT_PER_CHANNEL.saturating_sub(1);
                if rest_coefficients_per_channel == 0 {
                    return;
                }
                let channel = i / rest_coefficients_per_channel;
                // Streaming PLY loads rewrite source f_rest indices into this
                // compiled channel-major layout once, from the header. Keep
                // this direct PropertyAccess implementation bounded as well.
                if channel >= SH_CHANNELS {
                    return;
                }
                let coefficient = (i % rest_coefficients_per_channel) + 1;

                let Some(interleaved_idx) = coefficient
                    .checked_mul(SH_CHANNELS)
                    .and_then(|base| base.checked_add(channel))
                else {
                    return;
                };

                if interleaved_idx < SH_COEFF_COUNT {
                    self.spherical_harmonic.set(interleaved_idx, v);
                } else {
                    // TODO: convert higher degree SH to lower degree SH
                }
            }
            (_, _) => {}
        }
    }
}

pub fn parse_ply_3d(reader: &mut dyn BufRead) -> Result<PlanarGaussian3d, std::io::Error> {
    let mut cloud = Vec::new();
    stream_ply_3d(reader, DEFAULT_STREAM_BATCH_SIZE, |batch| {
        cloud.extend_from_slice(batch);
        Ok(())
    })?;

    Ok(PlanarGaussian3d::from_interleaved(cloud))
}

/// Stream a PLY payload in bounded normalized batches.
///
/// The callback may persist Morton runs or pages immediately, so the normalized Gaussian batch
/// allocation is controlled by `batch_size` rather than source size. The upstream PLY parser's
/// header and one ASCII input line are separate allocations; this function rejects vertex-list
/// properties but does not claim a hostile-input byte bound for an arbitrarily long ASCII line.
/// Callers should only publish their final manifest after this function returns successfully
/// because earlier callback writes are not rolled back.
pub fn stream_ply_3d(
    reader: &mut dyn BufRead,
    batch_size: usize,
    consume_batch: impl FnMut(&[Gaussian3d]) -> io::Result<()>,
) -> io::Result<PlyStreamSummary> {
    stream_ply_3d_with_sh_compatibility(
        reader,
        batch_size,
        PlyShCompatibility::AllowTruncation,
        consume_batch,
    )
}

/// Stream a PLY payload while explicitly controlling whether higher-order SH
/// coefficients may be truncated to the compiled representation.
pub fn stream_ply_3d_with_sh_compatibility(
    mut reader: &mut dyn BufRead,
    batch_size: usize,
    sh_compatibility: PlyShCompatibility,
    mut consume_batch: impl FnMut(&[Gaussian3d]) -> io::Result<()>,
) -> io::Result<PlyStreamSummary> {
    if batch_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "PLY stream batch size must be greater than zero",
        ));
    }
    let batch_bytes = batch_size
        .checked_mul(size_of::<Gaussian3d>())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "PLY stream batch allocation size overflowed usize",
            )
        })?;
    if batch_bytes > MAX_STREAM_BATCH_ALLOCATION_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "PLY stream batch requires {batch_bytes} bytes, exceeding the {}-byte safety limit",
                MAX_STREAM_BATCH_ALLOCATION_BYTES
            ),
        ));
    }

    let parser = Parser::<Gaussian3d>::new();
    let header = parser.read_header(&mut reader)?;

    // This converter consumes only Gaussian vertices. Validate the complete
    // header before reading any payload so an ignored face/list element cannot
    // make the upstream parser allocate from an attacker-controlled list count.
    let mut remapped_vertex = None;
    for (_, element) in &header.elements {
        if element.name == "vertex" {
            let sh_layout = validate_gaussian_3d_properties(element, sh_compatibility)?;
            remapped_vertex = Some(remap_gaussian_3d_element(element, sh_layout));
        } else if element.count != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported non-vertex PLY element '{}' with {} records",
                    element.name, element.count
                ),
            ));
        }
    }

    let mut summary = PlyStreamSummary::default();
    let mut batch = Vec::new();
    batch.try_reserve_exact(batch_size).map_err(|error| {
        io::Error::new(
            io::ErrorKind::OutOfMemory,
            format!("could not reserve the PLY stream batch: {error}"),
        )
    })?;

    for (_, element) in &header.elements {
        let parse_element = if element.name == "vertex" {
            remapped_vertex.as_ref().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "validated PLY vertex definition is unavailable",
                )
            })?
        } else {
            element
        };
        for element_index in 0..element.count {
            let mut value = read_ply_element(&parser, &mut reader, parse_element, header.encoding)?;
            if element.name != "vertex" {
                continue;
            }

            normalize_gaussian_3d(&mut value).map_err(|message| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid Gaussian vertex {element_index}: {message}"),
                )
            })?;
            summary.include();
            batch.push(value);
            if batch.len() == batch_size {
                consume_batch(&batch)?;
                batch.clear();
            }
        }
    }

    if !batch.is_empty() {
        consume_batch(&batch)?;
    }
    Ok(summary)
}

fn read_ply_element(
    parser: &Parser<Gaussian3d>,
    mut reader: &mut dyn BufRead,
    element: &ElementDef,
    encoding: Encoding,
) -> io::Result<Gaussian3d> {
    match encoding {
        Encoding::Ascii => {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("unexpected EOF while reading '{}' element", element.name),
                ));
            }
            parser.read_ascii_element(&line, element)
        }
        Encoding::BinaryBigEndian => parser.read_big_endian_element(&mut reader, element),
        Encoding::BinaryLittleEndian => parser.read_little_endian_element(&mut reader, element),
    }
}

fn validate_gaussian_3d_properties(
    element: &ElementDef,
    sh_compatibility: PlyShCompatibility,
) -> io::Result<PlyShLayout> {
    const REQUIRED: [&str; 14] = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "scale_0", "scale_1", "scale_2", "opacity",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ];
    let missing = REQUIRED
        .into_iter()
        .filter(|required| !element.properties.contains_key(*required))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "missing required Gaussian properties: {}",
                missing.join(", ")
            ),
        ));
    }
    for (name, property) in &element.properties {
        if matches!(property.data_type, PropertyType::List(_, _)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("list-valued Gaussian vertex property '{name}' is not supported"),
            ));
        }
        let consumed = REQUIRED.contains(&name.as_str())
            || name == "visibility"
            || name.starts_with("f_rest_");
        if consumed && property.data_type != PropertyType::Scalar(ScalarType::Float) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Gaussian property '{name}' must be a scalar float"),
            ));
        }
    }
    let mut rest_indices = Vec::new();
    for name in element
        .properties
        .keys()
        .filter(|name| name.starts_with("f_rest_"))
    {
        let index = name[7..].parse::<usize>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("malformed Gaussian spherical-harmonic property '{name}'"),
            )
        })?;
        rest_indices.push(index);
    }
    rest_indices.sort_unstable();
    if let Some((expected, actual)) = rest_indices
        .iter()
        .copied()
        .enumerate()
        .find(|(expected, actual)| expected != actual)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Gaussian spherical-harmonic properties must be contiguous from f_rest_0; expected f_rest_{expected}, found f_rest_{actual}"
            ),
        ));
    }
    if rest_indices.len() % SH_CHANNELS != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Gaussian spherical-harmonic property count {} is not divisible across {SH_CHANNELS} color channels",
                rest_indices.len()
            ),
        ));
    }

    let source_rest_coefficients_per_channel = rest_indices.len() / SH_CHANNELS;
    let source_coefficients_per_channel = source_rest_coefficients_per_channel
        .checked_add(1)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Gaussian spherical-harmonic coefficient count overflowed usize",
            )
        })?;
    let source_degree_plus_one = source_coefficients_per_channel.isqrt();
    if source_degree_plus_one.checked_mul(source_degree_plus_one)
        != Some(source_coefficients_per_channel)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Gaussian f_rest properties do not form a complete spherical-harmonic degree: each channel has {source_coefficients_per_channel} coefficients including DC"
            ),
        ));
    }

    let compiled_rest_coefficients_per_channel = SH_COEFF_COUNT_PER_CHANNEL.saturating_sub(1);
    if sh_compatibility == PlyShCompatibility::RequireRepresentable
        && source_rest_coefficients_per_channel > compiled_rest_coefficients_per_channel
    {
        let source_degree = source_degree_plus_one.saturating_sub(1);
        let representable_rest = SH_CHANNELS * compiled_rest_coefficients_per_channel;
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Gaussian source SH degree {source_degree} exceeds the compiled spherical-harmonic capacity of {representable_rest} f_rest coefficients; rebuild with the matching SH profile or explicitly allow SH truncation"
            ),
        ));
    }
    Ok(PlyShLayout {
        source_rest_coefficients_per_channel,
    })
}

/// Rewrite source channel-major indices into the compiled channel stride. The
/// PLY parser then fills [`Gaussian3d`] directly, without per-vertex scratch
/// storage. An internal NUL-prefixed key cannot collide with a parsed PLY
/// identifier and marks a coefficient that is intentionally discarded.
fn remap_gaussian_3d_element(element: &ElementDef, sh_layout: PlyShLayout) -> ElementDef {
    let compiled_rest_coefficients_per_channel = SH_COEFF_COUNT_PER_CHANNEL.saturating_sub(1);
    if sh_layout.source_rest_coefficients_per_channel == compiled_rest_coefficients_per_channel {
        return element.clone();
    }

    let mut remapped = ElementDef::new(element.name.clone());
    remapped.count = element.count;

    for (name, property) in &element.properties {
        let Some(source_index) = name
            .strip_prefix("f_rest_")
            .and_then(|index| index.parse::<usize>().ok())
        else {
            remapped.properties.insert(name.clone(), property.clone());
            continue;
        };

        let source_channel = source_index / sh_layout.source_rest_coefficients_per_channel;
        let source_coefficient = source_index % sh_layout.source_rest_coefficients_per_channel;
        let remapped_name = if source_coefficient < compiled_rest_coefficients_per_channel {
            let compiled_index =
                source_channel * compiled_rest_coefficients_per_channel + source_coefficient;
            format!("f_rest_{compiled_index}")
        } else {
            format!("\0bgs_truncated_f_rest_{source_index}")
        };
        let mut remapped_property = property.clone();
        remapped_property.name.clone_from(&remapped_name);
        remapped.properties.insert(remapped_name, remapped_property);
    }

    remapped
}

fn normalize_gaussian_3d(gaussian: &mut Gaussian3d) -> Result<(), &'static str> {
    if !gaussian
        .position_visibility
        .position
        .iter()
        .all(|value| value.is_finite())
    {
        return Err("position is not finite");
    }

    // PLY Gaussian splat scales are logarithmic. Clamp relative outliers before exponentiation,
    // matching the legacy loader while rejecting values that would poison hierarchy bounds.
    let mean_scale = gaussian.scale_opacity.scale.iter().sum::<f32>() / 3.0;
    if !mean_scale.is_finite() {
        return Err("scale is not finite");
    }
    for scale in &mut gaussian.scale_opacity.scale {
        *scale = scale
            .clamp(
                mean_scale - MAX_SIZE_VARIANCE,
                mean_scale + MAX_SIZE_VARIANCE,
            )
            .exp();
        if !scale.is_finite() || *scale <= 0.0 {
            return Err("scale is invalid after exponentiation");
        }
    }

    let norm_squared = gaussian
        .rotation
        .rotation
        .iter()
        .map(|value| value * value)
        .sum::<f32>();
    if !norm_squared.is_finite() {
        return Err("rotation is not finite");
    }
    if norm_squared <= f32::EPSILON {
        gaussian.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
    } else {
        let reciprocal_norm = norm_squared.sqrt().recip();
        for value in &mut gaussian.rotation.rotation {
            *value *= reciprocal_norm;
        }
    }
    if !gaussian.scale_opacity.opacity.is_finite() {
        return Err("opacity is not finite");
    }
    Ok(())
}

impl PropertyAccess for Gaussian4d {
    fn new() -> Self {
        Gaussian4d::default()
    }

    fn set_property(&mut self, key: String, property: Property) {
        match (key.as_ref(), property) {
            ("x", Property::Float(v)) => self.position_visibility.position[0] = v,
            ("y", Property::Float(v)) => self.position_visibility.position[1] = v,
            ("z", Property::Float(v)) => self.position_visibility.position[2] = v,
            ("visibility", Property::Float(v)) => self.position_visibility.visibility = v,

            ("t", Property::Float(v)) => self.timestamp_timescale.timestamp = v,
            ("st", Property::Float(v)) => self.timestamp_timescale.timescale = v,

            (_, Property::Float(v)) if key.starts_with("feat_") => {
                let Some(channel) = key.chars().nth(5).and_then(|channel| match channel {
                    'r' => Some(0),
                    'g' => Some(1),
                    'b' => Some(2),
                    _ => None,
                }) else {
                    return;
                };
                let Ok(i) = key.get(7..).unwrap_or_default().parse::<usize>() else {
                    return;
                };
                let Some(interleaved_idx) = i
                    .checked_mul(SH_CHANNELS)
                    .and_then(|base| base.checked_add(channel))
                else {
                    return;
                };

                if interleaved_idx < SH_4D_COEFF_COUNT {
                    self.spherindrical_harmonic.set(interleaved_idx, v);
                } else {
                    // TODO: handle higher-degree if needed
                }
            }

            ("sx", Property::Float(v)) => self.scale_opacity.scale[0] = v,
            ("sy", Property::Float(v)) => self.scale_opacity.scale[1] = v,
            ("sz", Property::Float(v)) => self.scale_opacity.scale[2] = v,
            ("opacity", Property::Float(v)) => self.scale_opacity.opacity = v,

            ("rot_x", Property::Float(v)) => self.isotropic_rotations.rotation[0] = v,
            ("rot_y", Property::Float(v)) => self.isotropic_rotations.rotation[1] = v,
            ("rot_z", Property::Float(v)) => self.isotropic_rotations.rotation[2] = v,
            ("rot_w", Property::Float(v)) => self.isotropic_rotations.rotation[3] = v,

            ("rot_r_x", Property::Float(v)) => self.isotropic_rotations.rotation_r[0] = v,
            ("rot_r_y", Property::Float(v)) => self.isotropic_rotations.rotation_r[1] = v,
            ("rot_r_z", Property::Float(v)) => self.isotropic_rotations.rotation_r[2] = v,
            ("rot_r_w", Property::Float(v)) => self.isotropic_rotations.rotation_r[3] = v,
            _ => {}
        }
    }
}

pub fn parse_ply_4d(mut reader: &mut dyn BufRead) -> Result<PlanarGaussian4d, std::io::Error> {
    let parser = Parser::<Gaussian4d>::new();
    let header = parser.read_header(&mut reader)?;

    let mut cloud = Vec::new();

    let required_properties = vec![
        "x", "y", "z", "t", "st", "sx", "sy", "sz", "opacity", "rot_x", "rot_y", "rot_z", "rot_w",
        "rot_r_x", "rot_r_y", "rot_r_z", "rot_r_w",
    ];
    let mut required_property_count = required_properties.len();

    for (_key, element) in &header.elements {
        if element.name == "vertex" {
            for (key, _prop) in &element.properties {
                required_property_count -= required_properties.contains(&key.as_str()) as usize;
            }

            if required_property_count > 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "missing required properties",
                ));
            }

            cloud = parser.read_payload_for_element(&mut reader, element, &header)?;
        }
    }

    for g in &mut cloud {
        let norm = g
            .isotropic_rotations
            .rotation
            .iter()
            .map(|v| v.powi(2))
            .sum::<f32>()
            .sqrt();

        for v in &mut g.isotropic_rotations.rotation {
            *v /= norm;
        }

        let norm = g
            .isotropic_rotations
            .rotation_r
            .iter()
            .map(|v| v.powi(2))
            .sum::<f32>()
            .sqrt();

        for v in &mut g.isotropic_rotations.rotation_r {
            *v /= norm;
        }

        // TODO: normalize timescale between 0 and 1
    }

    Ok(PlanarGaussian4d::from_interleaved(cloud))
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, Cursor};

    use super::*;

    fn gaussian_ascii_ply(count: usize, include_scale_2: bool) -> Vec<u8> {
        let mut source = format!(
            "ply\nformat ascii 1.0\nelement vertex {count}\nproperty float x\nproperty float y\nproperty float z\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float scale_0\nproperty float scale_1\n"
        );
        if include_scale_2 {
            source.push_str("property float scale_2\n");
        }
        source.push_str(
            "property float opacity\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n",
        );
        for index in 0..count {
            let mut values = vec![
                index as f32,
                0.0,
                -(index as f32),
                0.1,
                0.2,
                0.3,
                -2.0,
                -2.0,
            ];
            if include_scale_2 {
                values.push(-2.0);
            }
            values.extend([0.0, 1.0, 0.0, 0.0, 0.0]);
            source.push_str(
                &values
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            source.push('\n');
        }
        source.into_bytes()
    }

    fn gaussian_ascii_ply_with_rest(rest_values: &[f32]) -> Vec<u8> {
        let mut source = String::from(
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\n",
        );
        for index in 0..rest_values.len() {
            source.push_str(&format!("property float f_rest_{index}\n"));
        }
        source.push_str(
            "property float opacity\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n0 0 0 0.1 0.2 0.3 -2 -2 -2",
        );
        for value in rest_values {
            source.push(' ');
            source.push_str(&value.to_string());
        }
        source.push_str(" 0 1 0 0 0\n");
        source.into_bytes()
    }

    #[test]
    fn streams_bounded_batches_and_reports_logical_count() {
        let bytes = gaussian_ascii_ply(5, true);
        let mut reader = BufReader::new(Cursor::new(bytes));
        let mut batch_lengths = Vec::new();
        let summary = stream_ply_3d(&mut reader, 2, |batch| {
            batch_lengths.push(batch.len());
            Ok(())
        })
        .unwrap();

        assert_eq!(batch_lengths, [2, 2, 1]);
        assert_eq!(summary.logical_count, 5);
    }

    #[test]
    fn aligned_legacy_parse_does_not_add_an_extra_workgroup() {
        let bytes = gaussian_ascii_ply(32, true);
        let mut reader = BufReader::new(Cursor::new(bytes));
        let cloud = parse_ply_3d(&mut reader).unwrap();
        assert_eq!(cloud.len(), 32);
    }

    #[test]
    fn parse_preserves_unaligned_logical_count() {
        let bytes = gaussian_ascii_ply(33, true);
        let mut reader = BufReader::new(Cursor::new(bytes));
        let cloud = parse_ply_3d(&mut reader).unwrap();
        assert_eq!(cloud.len(), 33);
    }

    #[test]
    fn rejects_missing_scale_axis_and_zero_batch_size() {
        let bytes = gaussian_ascii_ply(1, false);
        let mut reader = BufReader::new(Cursor::new(bytes));
        let error = stream_ply_3d(&mut reader, 1, |_| Ok(())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("scale_2"));

        let bytes = gaussian_ascii_ply(1, true);
        let mut reader = BufReader::new(Cursor::new(bytes));
        let error = stream_ply_3d(&mut reader, 0, |_| Ok(())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        let bytes = gaussian_ascii_ply(1, true);
        let mut reader = BufReader::new(Cursor::new(bytes));
        let oversized = MAX_STREAM_BATCH_ALLOCATION_BYTES / size_of::<Gaussian3d>() + 1;
        let error = stream_ply_3d(&mut reader, oversized, |_| Ok(())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("safety limit"));
    }

    #[test]
    fn explicit_sh_policy_rejects_lossy_profile_conversion() {
        let compiled_degree_plus_one = SH_COEFF_COUNT_PER_CHANNEL.isqrt();
        let source_degree_plus_one = compiled_degree_plus_one + 1;
        let source_rest_per_channel = source_degree_plus_one * source_degree_plus_one - 1;
        let source =
            gaussian_ascii_ply_with_rest(&vec![0.25; SH_CHANNELS * source_rest_per_channel]);

        let mut reader = BufReader::new(Cursor::new(&source));
        let error = stream_ply_3d_with_sh_compatibility(
            &mut reader,
            1,
            PlyShCompatibility::RequireRepresentable,
            |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("exceeds the compiled"));

        let mut reader = BufReader::new(Cursor::new(&source));
        let summary = stream_ply_3d_with_sh_compatibility(
            &mut reader,
            1,
            PlyShCompatibility::AllowTruncation,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(summary.logical_count, 1);
    }

    #[test]
    fn higher_source_degree_truncates_each_channel_without_aliasing() {
        let compiled_rest_per_channel = SH_COEFF_COUNT_PER_CHANNEL.saturating_sub(1);
        if compiled_rest_per_channel == 0 {
            return;
        }

        let compiled_degree_plus_one = SH_COEFF_COUNT_PER_CHANNEL.isqrt();
        let source_degree_plus_one = compiled_degree_plus_one + 1;
        let source_rest_per_channel = source_degree_plus_one * source_degree_plus_one - 1;
        let mut rest_values = (0..SH_CHANNELS * source_rest_per_channel)
            .map(|index| index as f32 + 1.0)
            .collect::<Vec<_>>();
        rest_values[compiled_rest_per_channel] = 1234.0;
        rest_values[source_rest_per_channel] = 20.0;
        rest_values[2 * source_rest_per_channel] = 30.0;
        let source = gaussian_ascii_ply_with_rest(&rest_values);

        let mut gaussian = None;
        let mut reader = BufReader::new(Cursor::new(source));
        stream_ply_3d_with_sh_compatibility(
            &mut reader,
            1,
            PlyShCompatibility::AllowTruncation,
            |batch| {
                gaussian = Some(batch[0]);
                Ok(())
            },
        )
        .unwrap();
        let gaussian = gaussian.unwrap();

        for channel in 0..SH_CHANNELS {
            for source_coefficient in 0..compiled_rest_per_channel {
                let source_index = channel * source_rest_per_channel + source_coefficient;
                let compiled_index = (source_coefficient + 1) * SH_CHANNELS + channel;
                assert_eq!(
                    gaussian.spherical_harmonic.coefficients[compiled_index],
                    rest_values[source_index]
                );
            }
        }
        assert_eq!(
            gaussian
                .spherical_harmonic
                .coefficients
                .get(SH_CHANNELS + 1),
            Some(&20.0)
        );
        assert_eq!(
            gaussian
                .spherical_harmonic
                .coefficients
                .get(SH_CHANNELS + 2),
            Some(&30.0)
        );
    }

    #[test]
    fn lower_source_degree_uses_its_own_channel_stride() {
        let compiled_degree_plus_one = SH_COEFF_COUNT_PER_CHANNEL.isqrt();
        if compiled_degree_plus_one <= 2 {
            return;
        }

        let source_degree_plus_one = compiled_degree_plus_one - 1;
        let source_rest_per_channel = source_degree_plus_one * source_degree_plus_one - 1;
        let rest_values = (0..SH_CHANNELS * source_rest_per_channel)
            .map(|index| index as f32 + 1.0)
            .collect::<Vec<_>>();
        let source = gaussian_ascii_ply_with_rest(&rest_values);

        let mut gaussian = None;
        let mut reader = BufReader::new(Cursor::new(source));
        stream_ply_3d(&mut reader, 1, |batch| {
            gaussian = Some(batch[0]);
            Ok(())
        })
        .unwrap();
        let gaussian = gaussian.unwrap();

        for channel in 0..SH_CHANNELS {
            for source_coefficient in 0..source_rest_per_channel {
                let source_index = channel * source_rest_per_channel + source_coefficient;
                let compiled_index = (source_coefficient + 1) * SH_CHANNELS + channel;
                assert_eq!(
                    gaussian.spherical_harmonic.coefficients[compiled_index],
                    rest_values[source_index]
                );
            }
        }
    }

    #[test]
    fn rejects_incomplete_or_non_degree_rest_property_sets() {
        let complete_degree_one = vec![0.0; SH_CHANNELS * 3];
        let source = String::from_utf8(gaussian_ascii_ply_with_rest(&complete_degree_one))
            .unwrap()
            .replacen("property float f_rest_4\n", "", 1);
        let mut reader = BufReader::new(Cursor::new(source));
        let error = stream_ply_3d(&mut reader, 1, |_| Ok(())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("contiguous"));

        let source = gaussian_ascii_ply_with_rest(&[0.0; SH_CHANNELS]);
        let mut reader = BufReader::new(Cursor::new(source));
        let error = stream_ply_3d(&mut reader, 1, |_| Ok(())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(
            error
                .to_string()
                .contains("complete spherical-harmonic degree")
        );
    }

    #[test]
    fn rejects_wrong_consumed_types_and_vertex_lists_before_payload_decode() {
        let source = String::from_utf8(gaussian_ascii_ply(1, true)).unwrap();
        let wrong_position_type = source.replacen("property float x", "property double x", 1);
        let mut reader = BufReader::new(Cursor::new(wrong_position_type));
        let error = stream_ply_3d(&mut reader, 1, |_| Ok(())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("'x' must be a scalar float"));

        let list_property = source.replacen(
            "property float x",
            "property list uchar float unsupported\nproperty float x",
            1,
        );
        let mut reader = BufReader::new(Cursor::new(list_property));
        let error = stream_ply_3d(&mut reader, 1, |_| Ok(())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("list-valued"));
    }

    #[test]
    fn rejects_non_vertex_payloads_before_list_counts_are_decoded() {
        let source = b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float opacity\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nelement face 1\nproperty list char int vertex_indices\nend_header\n";
        let mut reader = BufReader::new(Cursor::new(source));
        let error = stream_ply_3d(&mut reader, 1, |_| Ok(())).unwrap_err();
        assert!(error.to_string().contains("unsupported non-vertex"));
    }

    #[test]
    fn zero_quaternion_is_repaired_without_nan() {
        let mut gaussian = Gaussian3d::default();
        gaussian.scale_opacity.scale = [0.0; 3];
        gaussian.rotation.rotation = [0.0; 4];
        normalize_gaussian_3d(&mut gaussian).unwrap();
        assert_eq!(gaussian.rotation.rotation, [1.0, 0.0, 0.0, 0.0]);
        assert!(
            gaussian
                .scale_opacity
                .scale
                .iter()
                .all(|scale| *scale == 1.0)
        );
    }

    #[test]
    fn rest_coefficients_use_standard_channel_major_ply_layout() {
        if SH_COEFF_COUNT_PER_CHANNEL <= 1 {
            return;
        }
        let rest_per_channel = SH_COEFF_COUNT_PER_CHANNEL - 1;
        let mut gaussian = Gaussian3d::default();
        gaussian.set_property("f_rest_0".to_owned(), Property::Float(10.0));
        gaussian.set_property(format!("f_rest_{rest_per_channel}"), Property::Float(20.0));
        assert_eq!(gaussian.spherical_harmonic.coefficients[SH_CHANNELS], 10.0);
        assert_eq!(
            gaussian
                .spherical_harmonic
                .coefficients
                .get(SH_CHANNELS + 1)
                .copied(),
            Some(20.0)
        );
    }

    #[test]
    fn malformed_rest_property_is_an_input_error_not_a_panic() {
        let source = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float opacity\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nproperty float f_rest_broken\nend_header\n";
        let mut reader = BufReader::new(Cursor::new(source));
        let error = stream_ply_3d(&mut reader, 1, |_| Ok(())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("f_rest_broken"));
    }

    #[test]
    fn malformed_4d_feature_property_is_ignored_without_panicking() {
        let mut gaussian = Gaussian4d::default();
        gaussian.set_property("feat_".to_owned(), Property::Float(1.0));
        gaussian.set_property("feat_x_0".to_owned(), Property::Float(2.0));
        gaussian.set_property("feat_r_bad".to_owned(), Property::Float(3.0));
        gaussian.set_property(format!("feat_r_{}", usize::MAX), Property::Float(4.0));
        assert!(
            gaussian
                .spherindrical_harmonic
                .coefficients
                .iter()
                .flatten()
                .all(|coefficient| *coefficient == 0.0)
        );

        let mut gaussian = Gaussian3d::default();
        gaussian.set_property(format!("f_rest_{}", usize::MAX), Property::Float(1.0));
        assert!(
            gaussian
                .spherical_harmonic
                .coefficients
                .iter()
                .all(|coefficient| *coefficient == 0.0)
        );
    }
}
