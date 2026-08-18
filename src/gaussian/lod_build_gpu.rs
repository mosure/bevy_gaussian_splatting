//! Canonical preprocessing contracts and bounded GPU hierarchy primitives.
//!
//! [`preprocess_lod_batch_cpu`] validates a bounded source batch and emits the
//! deterministic Morton/support records consumed by the external-memory
//! builder. [`hierarchy`] contains the promoted GPU route: bounded device sort
//! plus explicit MomentMerge reductions for the globally merged hierarchy.

/// Deterministic bounded GPU sort and MomentMerge reduction primitives.
pub mod hierarchy;

use std::fmt;

use crate::gaussian::formats::{
    planar_3d::Gaussian3d,
    planar_3d_chunked::{LodBounds, LodBoundsError},
    planar_3d_lod::{canonical_lod_morton_code, gaussian_support_bounds},
};

/// Validation flags emitted independently for every source record.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct LodPreprocessStatus(pub u32);

impl LodPreprocessStatus {
    pub const VALID: Self = Self(0);
    pub const POSITION_NON_FINITE: Self = Self(1 << 0);
    pub const VISIBILITY_NON_FINITE: Self = Self(1 << 1);
    pub const SH_NON_FINITE: Self = Self(1 << 2);
    pub const ROTATION_NON_FINITE: Self = Self(1 << 3);
    pub const DEGENERATE_ROTATION: Self = Self(1 << 4);
    pub const SCALE_NON_FINITE: Self = Self(1 << 5);
    pub const NEGATIVE_SCALE: Self = Self(1 << 6);
    pub const OPACITY_NON_FINITE: Self = Self(1 << 7);
    pub const OUTSIDE_NORMALIZATION_BOUNDS: Self = Self(1 << 8);
    pub const DERIVED_NON_FINITE: Self = Self(1 << 9);

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn contains(self, flag: Self) -> bool {
        self.0 & flag.0 == flag.0
    }

    #[inline]
    fn insert(&mut self, flag: Self) {
        self.0 |= flag.0;
    }
}

impl fmt::Debug for LodPreprocessStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            formatter.write_str("LodPreprocessStatus::VALID")
        } else {
            write!(formatter, "LodPreprocessStatus({:#x})", self.0)
        }
    }
}

/// One input-order record produced for an external Morton-sort run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodPreprocessRecord {
    pub source_index: u64,
    pub morton: u64,
    pub support_bounds: Option<LodBounds>,
    pub status: LodPreprocessStatus,
}

impl LodPreprocessRecord {
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.status.is_valid()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LodPreprocessBatchOutput {
    pub records: Vec<LodPreprocessRecord>,
}

/// CPU implementation of the canonical validation, conservative support,
/// f32 Morton quantization, and source-index contract.
pub fn preprocess_lod_batch_cpu(
    records: &[Gaussian3d],
    source_index_base: u64,
    normalization_bounds: LodBounds,
    support_sigma: f32,
) -> Result<LodPreprocessBatchOutput, LodPreprocessError> {
    validate_batch_inputs(
        records,
        source_index_base,
        normalization_bounds,
        support_sigma,
    )?;

    let mut output = Vec::with_capacity(records.len());
    for (local_index, gaussian) in records.iter().enumerate() {
        let source_index = source_index_base + local_index as u64;
        let status = validate_record(gaussian, normalization_bounds);
        if !status.is_valid() {
            output.push(LodPreprocessRecord {
                source_index,
                morton: 0,
                support_bounds: None,
                status,
            });
            continue;
        }

        let position = gaussian.position_visibility.position;
        let mut status = status;
        let support_bounds = match gaussian_support_bounds(gaussian, support_sigma) {
            Ok(bounds) => Some(bounds),
            Err(_) => {
                status.insert(LodPreprocessStatus::DERIVED_NON_FINITE);
                None
            }
        };
        output.push(LodPreprocessRecord {
            source_index,
            morton: if status.is_valid() {
                canonical_lod_morton_code(position, normalization_bounds)
            } else {
                0
            },
            support_bounds,
            status,
        });
    }
    Ok(LodPreprocessBatchOutput { records: output })
}

fn validate_batch_inputs(
    records: &[Gaussian3d],
    source_index_base: u64,
    normalization_bounds: LodBounds,
    support_sigma: f32,
) -> Result<(), LodPreprocessError> {
    if records.len() > u32::MAX as usize {
        return Err(LodPreprocessError::BatchTooLarge {
            actual: records.len(),
            limit: u32::MAX,
        });
    }
    if !records.is_empty() {
        source_index_base
            .checked_add(records.len() as u64 - 1)
            .ok_or(LodPreprocessError::SourceIndexOverflow)?;
    }
    normalization_bounds
        .validate()
        .map_err(LodPreprocessError::InvalidNormalizationBounds)?;
    for axis in 0..3 {
        if !(normalization_bounds.max[axis] - normalization_bounds.min[axis]).is_finite() {
            return Err(LodPreprocessError::NormalizationExtentNonFinite { axis });
        }
    }
    if !support_sigma.is_finite() || support_sigma <= 0.0 {
        return Err(LodPreprocessError::InvalidSupportSigma(support_sigma));
    }
    Ok(())
}

fn validate_record(gaussian: &Gaussian3d, normalization_bounds: LodBounds) -> LodPreprocessStatus {
    let mut status = LodPreprocessStatus::VALID;
    let position = gaussian.position_visibility.position;
    if !position.iter().all(|value| value.is_finite()) {
        status.insert(LodPreprocessStatus::POSITION_NON_FINITE);
    }
    if !gaussian.position_visibility.visibility.is_finite() {
        status.insert(LodPreprocessStatus::VISIBILITY_NON_FINITE);
    }
    if !gaussian
        .spherical_harmonic
        .coefficients
        .iter()
        .all(|value| value.is_finite())
    {
        status.insert(LodPreprocessStatus::SH_NON_FINITE);
    }
    if !gaussian
        .rotation
        .rotation
        .iter()
        .all(|value| value.is_finite())
    {
        status.insert(LodPreprocessStatus::ROTATION_NON_FINITE);
    } else {
        let rotation_norm_squared = gaussian
            .rotation
            .rotation
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        if !rotation_norm_squared.is_finite() || rotation_norm_squared <= f32::EPSILON {
            status.insert(LodPreprocessStatus::DEGENERATE_ROTATION);
        }
    }
    if !gaussian
        .scale_opacity
        .scale
        .iter()
        .all(|value| value.is_finite())
    {
        status.insert(LodPreprocessStatus::SCALE_NON_FINITE);
    } else if gaussian
        .scale_opacity
        .scale
        .iter()
        .any(|value| *value < 0.0)
    {
        status.insert(LodPreprocessStatus::NEGATIVE_SCALE);
    }
    if !gaussian.scale_opacity.opacity.is_finite() {
        status.insert(LodPreprocessStatus::OPACITY_NON_FINITE);
    }
    if !status.contains(LodPreprocessStatus::POSITION_NON_FINITE)
        && (0..3).any(|axis| {
            position[axis] < normalization_bounds.min[axis]
                || position[axis] > normalization_bounds.max[axis]
        })
    {
        status.insert(LodPreprocessStatus::OUTSIDE_NORMALIZATION_BOUNDS);
    }
    status
}

#[derive(Debug)]
pub enum LodPreprocessError {
    BatchTooLarge { actual: usize, limit: u32 },
    SourceIndexOverflow,
    InvalidNormalizationBounds(LodBoundsError),
    NormalizationExtentNonFinite { axis: usize },
    InvalidSupportSigma(f32),
}

impl fmt::Display for LodPreprocessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchTooLarge { actual, limit } => {
                write!(
                    formatter,
                    "LoD preprocessing batch has {actual} records, limit is {limit}"
                )
            }
            Self::SourceIndexOverflow => formatter.write_str("LoD source index overflow"),
            Self::InvalidNormalizationBounds(error) => {
                write!(formatter, "invalid LoD normalization bounds: {error}")
            }
            Self::NormalizationExtentNonFinite { axis } => write!(
                formatter,
                "LoD normalization extent for axis {axis} is not finite in f32"
            ),
            Self::InvalidSupportSigma(value) => write!(
                formatter,
                "LoD support sigma must be finite and positive, got {value}"
            ),
        }
    }
}

impl std::error::Error for LodPreprocessError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::spherical_harmonics::SphericalHarmonicCoefficients;

    fn gaussian(position: [f32; 3], scale: [f32; 3]) -> Gaussian3d {
        Gaussian3d {
            position_visibility: [position[0], position[1], position[2], 1.0].into(),
            spherical_harmonic: SphericalHarmonicCoefficients::default(),
            rotation: [0.0, 0.0, 0.0, 1.0].into(),
            scale_opacity: [scale[0], scale[1], scale[2], 0.75].into(),
        }
    }

    fn normalization_bounds() -> LodBounds {
        LodBounds::new([-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]).unwrap()
    }

    #[test]
    fn preprocessing_validates_fields_and_normalization_domain() {
        let mut records = vec![gaussian([0.0, 0.0, 0.0], [0.25, 0.5, 1.0]); 5];
        records[1].spherical_harmonic.coefficients[0] = f32::NAN;
        records[2].rotation.rotation = [0.0; 4];
        records[3].scale_opacity.scale[1] = -0.5;
        records[4].position_visibility.position[0] = 8.0;
        let output = preprocess_lod_batch_cpu(&records, 100, normalization_bounds(), 3.0).unwrap();

        assert!(output.records[0].is_valid());
        assert!(
            output.records[1]
                .status
                .contains(LodPreprocessStatus::SH_NON_FINITE)
        );
        assert!(
            output.records[2]
                .status
                .contains(LodPreprocessStatus::DEGENERATE_ROTATION)
        );
        assert!(
            output.records[3]
                .status
                .contains(LodPreprocessStatus::NEGATIVE_SCALE)
        );
        assert!(
            output.records[4]
                .status
                .contains(LodPreprocessStatus::OUTSIDE_NORMALIZATION_BOUNDS)
        );
        assert_eq!(output.records[4].source_index, 104);
    }

    #[test]
    fn preprocessing_emits_conservative_bounds_and_morton_endpoints() {
        let records = vec![
            gaussian([-4.0, -4.0, -4.0], [0.25, 0.5, 1.0]),
            gaussian([4.0, 4.0, 4.0], [0.5, 0.25, 0.125]),
            gaussian([0.0, 0.0, 0.0], [0.0; 3]),
        ];
        let output =
            preprocess_lod_batch_cpu(&records, u64::from(u32::MAX), normalization_bounds(), 3.0)
                .unwrap();

        assert_eq!(output.records[0].morton, 0);
        assert_eq!(output.records[1].morton, 0x7fff_ffff_ffff_ffff);
        assert_eq!(output.records[1].source_index, u64::from(u32::MAX) + 1);
        for (record, gaussian) in output.records.iter().zip(&records) {
            assert_eq!(
                record.morton,
                canonical_lod_morton_code(
                    gaussian.position_visibility.position,
                    normalization_bounds()
                )
            );
            assert_eq!(
                record.support_bounds,
                Some(gaussian_support_bounds(gaussian, 3.0).unwrap())
            );
            let bounds = record.support_bounds.unwrap();
            for axis in 0..3 {
                assert!(bounds.min[axis] <= gaussian.position_visibility.position[axis]);
                assert!(bounds.max[axis] >= gaussian.position_visibility.position[axis]);
            }
        }
    }

    #[test]
    fn preprocessing_rejects_unrepresentable_batches_and_extents() {
        let records = [gaussian([0.0; 3], [1.0; 3])];
        let extreme_bounds = LodBounds::new([-f32::MAX, -1.0, -1.0], [f32::MAX, 1.0, 1.0]).unwrap();
        let error = preprocess_lod_batch_cpu(&records, 0, extreme_bounds, 3.0).unwrap_err();
        assert!(matches!(
            error,
            LodPreprocessError::NormalizationExtentNonFinite { axis: 0 }
        ));
    }
}
