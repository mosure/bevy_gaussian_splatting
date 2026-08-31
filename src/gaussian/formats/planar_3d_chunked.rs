//! Versioned, independently addressable chunks of planar 3D Gaussians.
//!
//! The types in this module deliberately contain no Bevy asset or renderer
//! state.  They are the portable decoded-page contract shared by offline
//! builders, caches, streamers, and GPU upload code.

use std::{error::Error, fmt, mem::size_of};

use serde::{Deserialize, Serialize};

use crate::{
    gaussian::formats::planar_3d::{Gaussian3d, PlanarGaussian3d},
    material::spherical_harmonics::SH_DEGREE,
};

/// Current decoded page schema.
/// Page schema 2 records the compile-time spherical-harmonic coefficient count
/// in every page header. This prevents a page built for one SH layout from
/// being silently decoded with another layout.
pub const LOD_PAGE_SCHEMA_VERSION: u16 = 2;

const FNV_64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Stable identifier of a node in a LoD manifest.
///
/// Zero is reserved so an all-zero GPU record can represent an invalid node.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct LodNodeId(pub u64);

impl LodNodeId {
    pub const INVALID: Self = Self(0);

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }
}

/// Stable identifier of an independently addressable page.
///
/// Zero is reserved so an all-zero page-table record is always invalid.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct LodPageId(pub u64);

impl LodPageId {
    pub const INVALID: Self = Self(0);

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }
}

/// A finite, inclusive axis-aligned bound in cloud-local coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LodBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl LodBounds {
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Result<Self, LodBoundsError> {
        let bounds = Self { min, max };
        bounds.validate()?;
        Ok(bounds)
    }

    pub fn from_point(point: [f32; 3]) -> Result<Self, LodBoundsError> {
        Self::new(point, point)
    }

    pub fn validate(&self) -> Result<(), LodBoundsError> {
        for axis in 0..3 {
            if !self.min[axis].is_finite() || !self.max[axis].is_finite() {
                return Err(LodBoundsError::NonFinite { axis });
            }
            if self.min[axis] > self.max[axis] {
                return Err(LodBoundsError::Inverted { axis });
            }
        }
        Ok(())
    }

    #[inline]
    pub fn center(&self) -> [f32; 3] {
        [
            midpoint(self.min[0], self.max[0]),
            midpoint(self.min[1], self.max[1]),
            midpoint(self.min[2], self.max[2]),
        ]
    }

    #[inline]
    pub fn extent(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Radius of the sphere centered at [`Self::center`] containing the AABB.
    #[inline]
    pub fn radius(&self) -> f32 {
        let extent = [
            f64::from(self.max[0]) - f64::from(self.min[0]),
            f64::from(self.max[1]) - f64::from(self.min[1]),
            f64::from(self.max[2]) - f64::from(self.min[2]),
        ];
        let radius =
            0.5 * (extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2]).sqrt();
        radius.min(f32::MAX as f64) as f32
    }

    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    #[inline]
    pub fn contains(&self, other: &Self) -> bool {
        self.contains_with_epsilon(other, 0.0)
    }

    pub fn contains_with_epsilon(&self, other: &Self, epsilon: f32) -> bool {
        let epsilon = epsilon.max(0.0);
        (0..3).all(|axis| {
            self.min[axis] - epsilon <= other.min[axis]
                && self.max[axis] + epsilon >= other.max[axis]
        })
    }
}

#[inline]
fn midpoint(a: f32, b: f32) -> f32 {
    // This form avoids overflowing for large, same-sign finite endpoints.
    a * 0.5 + b * 0.5
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LodBoundsError {
    NonFinite { axis: usize },
    Inverted { axis: usize },
}

impl fmt::Display for LodBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { axis } => write!(f, "LoD bound axis {axis} is not finite"),
            Self::Inverted { axis } => write!(f, "LoD bound axis {axis} has min greater than max"),
        }
    }
}

impl Error for LodBoundsError {}

/// Contiguous range in a manifest-owned vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodIndexRange {
    pub start: u32,
    pub count: u32,
}

impl LodIndexRange {
    #[inline]
    pub const fn empty() -> Self {
        Self { start: 0, count: 0 }
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn end(self) -> Option<u32> {
        self.start.checked_add(self.count)
    }
}

/// Contiguous range in the canonical Morton-sorted source sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodSourceRange {
    pub start: u64,
    pub count: u64,
}

impl LodSourceRange {
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn end(self) -> Option<u64> {
        self.start.checked_add(self.count)
    }
}

/// Range of decoded Gaussians inside a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodPageRange {
    pub page: LodPageId,
    pub offset: u32,
    pub count: u32,
}

impl LodPageRange {
    #[inline]
    pub fn end(self) -> Option<u32> {
        self.offset.checked_add(self.count)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LodPageKind {
    /// Coarse replacement records for internal hierarchy nodes.
    Representatives,
    /// Original full-quality source records for leaves.
    SourceLeaves,
    /// A future packer may colocate both kinds in one page.
    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LodPageEncoding {
    /// The crate's current full-precision [`Gaussian3d`] representation.
    F32Planar,
    /// Canonical geometry/opacity remains f32, while the interleaved RGB SH
    /// prefix through the inclusive degree is stored as IEEE binary16. On
    /// decode it is expanded into the ordinary f32 plane and higher bands are
    /// exactly zero. Source-leaf pages deliberately never use this encoding.
    F16Sh { degree: u8 },
}

impl LodPageEncoding {
    /// Highest SH degree which can affect a decoded record. F32 pages carry
    /// the complete degree compiled into this crate.
    pub const fn effective_sh_degree(self) -> u8 {
        match self {
            Self::F32Planar => SH_DEGREE as u8,
            Self::F16Sh { degree } => degree,
        }
    }

    pub const fn is_supported(self) -> bool {
        self.effective_sh_degree() <= SH_DEGREE as u8
    }
}

/// Optional transport metadata. Page identity is independent of its location.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LodPageStorage {
    pub uri: String,
    /// Byte range for a packed file. `None` means the URI is the page object.
    pub byte_range: Option<(u64, u64)>,
    pub encoded_len: u64,
}

/// Manifest-side description of an independently validated page.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LodPageDescriptor {
    pub id: LodPageId,
    pub kind: LodPageKind,
    pub encoding: LodPageEncoding,
    pub gaussian_count: u32,
    pub decoded_len: u64,
    /// Stable checksum of the decoded versioned payload.
    pub content_hash: u64,
    pub bounds: LodBounds,
    pub storage: Option<LodPageStorage>,
}

impl LodPageDescriptor {
    pub const fn effective_sh_degree(&self) -> u8 {
        self.encoding.effective_sh_degree()
    }

    pub fn validate(&self) -> Result<(), LodPageValidationError> {
        if !self.id.is_valid() {
            return Err(LodPageValidationError::InvalidPageId);
        }
        if self.gaussian_count == 0 {
            return Err(LodPageValidationError::EmptyPage);
        }
        if !self.encoding.is_supported() {
            return Err(LodPageValidationError::UnsupportedShDegree {
                encoded: self.effective_sh_degree(),
                supported: SH_DEGREE as u8,
            });
        }
        let expected_len = u64::from(self.gaussian_count)
            .checked_mul(size_of::<Gaussian3d>() as u64)
            .ok_or(LodPageValidationError::DecodedLengthOverflow)?;
        if self.decoded_len != expected_len {
            return Err(LodPageValidationError::DecodedLengthMismatch {
                expected: expected_len,
                actual: self.decoded_len,
            });
        }
        self.bounds
            .validate()
            .map_err(LodPageValidationError::InvalidBounds)?;
        if let Some(storage) = &self.storage {
            if storage.uri.is_empty() {
                return Err(LodPageValidationError::EmptyStorageUri);
            }
            if storage.encoded_len == 0 {
                return Err(LodPageValidationError::EmptyEncodedPage);
            }
            if let Some((start, len)) = storage.byte_range {
                if len == 0 || start.checked_add(len).is_none() {
                    return Err(LodPageValidationError::InvalidByteRange);
                }
                if len != storage.encoded_len {
                    return Err(LodPageValidationError::EncodedLengthMismatch);
                }
            }
        }
        Ok(())
    }
}

/// Decoded, portable page payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanarGaussian3dPage {
    pub schema_version: u16,
    pub id: LodPageId,
    pub gaussians: Vec<Gaussian3d>,
}

impl PlanarGaussian3dPage {
    pub fn new(id: LodPageId, gaussians: Vec<Gaussian3d>) -> Self {
        Self {
            schema_version: LOD_PAGE_SCHEMA_VERSION,
            id,
            gaussians,
        }
    }

    pub fn from_planar(id: LodPageId, cloud: &PlanarGaussian3d) -> Self {
        Self::new(id, cloud.iter().collect())
    }

    pub fn into_planar(self) -> PlanarGaussian3d {
        self.gaussians.into()
    }

    #[inline]
    pub fn content_hash(&self) -> u64 {
        let mut hash = StableHasher::new();
        hash.write(&self.schema_version.to_le_bytes());
        hash.write(&self.id.0.to_le_bytes());
        hash.write(&(self.gaussians.len() as u64).to_le_bytes());
        for gaussian in &self.gaussians {
            hash_gaussian(&mut hash, gaussian);
        }
        hash.finish()
    }

    pub fn validate(&self, descriptor: &LodPageDescriptor) -> Result<(), LodPageValidationError> {
        descriptor.validate()?;
        if self.schema_version != LOD_PAGE_SCHEMA_VERSION {
            return Err(LodPageValidationError::UnsupportedSchemaVersion {
                expected: LOD_PAGE_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        if self.id != descriptor.id {
            return Err(LodPageValidationError::PageIdMismatch {
                expected: descriptor.id,
                actual: self.id,
            });
        }
        let actual_count = u32::try_from(self.gaussians.len())
            .map_err(|_| LodPageValidationError::GaussianCountOverflow)?;
        if actual_count != descriptor.gaussian_count {
            return Err(LodPageValidationError::GaussianCountMismatch {
                expected: descriptor.gaussian_count,
                actual: self.gaussians.len(),
            });
        }
        for (index, gaussian) in self.gaussians.iter().enumerate() {
            validate_gaussian(gaussian)
                .map_err(|field| LodPageValidationError::InvalidGaussian { index, field })?;
        }
        let actual_hash = self.content_hash();
        if actual_hash != descriptor.content_hash {
            return Err(LodPageValidationError::ContentHashMismatch {
                expected: descriptor.content_hash,
                actual: actual_hash,
            });
        }
        Ok(())
    }
}

/// Stable decoded-payload fingerprint used for cache and build identities.
pub fn stable_gaussian_hash(gaussian: &Gaussian3d) -> u64 {
    let mut hash = StableHasher::new();
    hash_gaussian(&mut hash, gaussian);
    hash.finish()
}

pub(crate) fn hash_gaussian(hash: &mut StableHasher, gaussian: &Gaussian3d) {
    for value in gaussian.position_visibility.position {
        hash.write(&canonical_f32_bits(value).to_le_bytes());
    }
    hash.write(&canonical_f32_bits(gaussian.position_visibility.visibility).to_le_bytes());
    for value in gaussian.spherical_harmonic.coefficients {
        hash.write(&canonical_f32_bits(value).to_le_bytes());
    }
    for value in gaussian.rotation.rotation {
        hash.write(&canonical_f32_bits(value).to_le_bytes());
    }
    for value in gaussian.scale_opacity.scale {
        hash.write(&canonical_f32_bits(value).to_le_bytes());
    }
    hash.write(&canonical_f32_bits(gaussian.scale_opacity.opacity).to_le_bytes());
}

#[inline]
fn canonical_f32_bits(value: f32) -> u32 {
    // Treat signed zero identically so a semantically irrelevant sign bit does
    // not invalidate a page cache key.
    if value == 0.0 { 0 } else { value.to_bits() }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaussianField {
    Position(usize),
    Visibility,
    SphericalHarmonic(usize),
    Rotation(usize),
    Scale(usize),
    Opacity,
    NegativeScale(usize),
    DegenerateRotation,
}

pub fn validate_gaussian(gaussian: &Gaussian3d) -> Result<(), GaussianField> {
    for (axis, value) in gaussian.position_visibility.position.iter().enumerate() {
        if !value.is_finite() {
            return Err(GaussianField::Position(axis));
        }
    }
    if !gaussian.position_visibility.visibility.is_finite() {
        return Err(GaussianField::Visibility);
    }
    for (index, value) in gaussian.spherical_harmonic.coefficients.iter().enumerate() {
        if !value.is_finite() {
            return Err(GaussianField::SphericalHarmonic(index));
        }
    }
    let mut rotation_norm_squared = 0.0;
    for (index, value) in gaussian.rotation.rotation.iter().enumerate() {
        if !value.is_finite() {
            return Err(GaussianField::Rotation(index));
        }
        rotation_norm_squared += value * value;
    }
    if !rotation_norm_squared.is_finite() || rotation_norm_squared <= f32::EPSILON {
        return Err(GaussianField::DegenerateRotation);
    }
    for (axis, value) in gaussian.scale_opacity.scale.iter().enumerate() {
        if !value.is_finite() {
            return Err(GaussianField::Scale(axis));
        }
        if *value < 0.0 {
            return Err(GaussianField::NegativeScale(axis));
        }
    }
    if !gaussian.scale_opacity.opacity.is_finite() {
        return Err(GaussianField::Opacity);
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub enum LodPageValidationError {
    InvalidPageId,
    EmptyPage,
    UnsupportedShDegree {
        encoded: u8,
        supported: u8,
    },
    DecodedLengthOverflow,
    DecodedLengthMismatch {
        expected: u64,
        actual: u64,
    },
    InvalidBounds(LodBoundsError),
    EmptyStorageUri,
    EmptyEncodedPage,
    InvalidByteRange,
    EncodedLengthMismatch,
    UnsupportedSchemaVersion {
        expected: u16,
        actual: u16,
    },
    PageIdMismatch {
        expected: LodPageId,
        actual: LodPageId,
    },
    GaussianCountOverflow,
    GaussianCountMismatch {
        expected: u32,
        actual: usize,
    },
    InvalidGaussian {
        index: usize,
        field: GaussianField,
    },
    ContentHashMismatch {
        expected: u64,
        actual: u64,
    },
}

impl fmt::Display for LodPageValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPageId => write!(f, "LoD page ID zero is reserved"),
            Self::EmptyPage => write!(f, "LoD page contains no Gaussians"),
            Self::UnsupportedShDegree { encoded, supported } => write!(
                f,
                "LoD page SH degree {encoded} exceeds compiled degree {supported}"
            ),
            Self::DecodedLengthOverflow => write!(f, "LoD page decoded length overflowed"),
            Self::DecodedLengthMismatch { expected, actual } => write!(
                f,
                "LoD page decoded length is {actual}, expected {expected}"
            ),
            Self::InvalidBounds(error) => write!(f, "invalid LoD page bounds: {error}"),
            Self::EmptyStorageUri => write!(f, "LoD page storage URI is empty"),
            Self::EmptyEncodedPage => write!(f, "LoD page encoded length is zero"),
            Self::InvalidByteRange => write!(f, "LoD page byte range is invalid"),
            Self::EncodedLengthMismatch => {
                write!(
                    f,
                    "LoD page byte-range length does not match encoded length"
                )
            }
            Self::UnsupportedSchemaVersion { expected, actual } => write!(
                f,
                "LoD page schema version {actual} is unsupported (expected {expected})"
            ),
            Self::PageIdMismatch { expected, actual } => write!(
                f,
                "LoD payload page ID {:?} does not match descriptor {:?}",
                actual, expected
            ),
            Self::GaussianCountOverflow => write!(f, "LoD page Gaussian count exceeds u32"),
            Self::GaussianCountMismatch { expected, actual } => write!(
                f,
                "LoD page contains {actual} Gaussians, expected {expected}"
            ),
            Self::InvalidGaussian { index, field } => {
                write!(f, "LoD page Gaussian {index} has invalid {field:?}")
            }
            Self::ContentHashMismatch { expected, actual } => write!(
                f,
                "LoD page checksum {actual:#018x} does not match {expected:#018x}"
            ),
        }
    }
}

impl Error for LodPageValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidBounds(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StableHasher(u64);

impl StableHasher {
    pub(crate) const fn new() -> Self {
        Self(FNV_64_OFFSET_BASIS)
    }

    pub(crate) fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(FNV_64_PRIME);
        }
    }

    pub(crate) const fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::spherical_harmonics::SphericalHarmonicCoefficients;

    fn gaussian(position: [f32; 3]) -> Gaussian3d {
        Gaussian3d {
            position_visibility: [position[0], position[1], position[2], 1.0].into(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.25, 0.5, 1.0, 0.5].into(),
            spherical_harmonic: SphericalHarmonicCoefficients::default(),
        }
    }

    #[test]
    fn bounds_union_contains_inputs() {
        let left = LodBounds::new([-2.0, -1.0, 3.0], [0.0, 4.0, 5.0]).unwrap();
        let right = LodBounds::new([-1.0, -3.0, 4.0], [8.0, 2.0, 7.0]).unwrap();
        let union = left.union(right);
        assert!(union.contains(&left));
        assert!(union.contains(&right));
        assert_eq!(union.min, [-2.0, -3.0, 3.0]);
        assert_eq!(union.max, [8.0, 4.0, 7.0]);
    }

    #[test]
    fn payload_hash_detects_corruption() {
        let page = PlanarGaussian3dPage::new(LodPageId(1), vec![gaussian([1.0, 2.0, 3.0])]);
        let descriptor = LodPageDescriptor {
            id: page.id,
            kind: LodPageKind::SourceLeaves,
            encoding: LodPageEncoding::F32Planar,
            gaussian_count: 1,
            decoded_len: size_of::<Gaussian3d>() as u64,
            content_hash: page.content_hash(),
            bounds: LodBounds::new([0.0; 3], [4.0; 3]).unwrap(),
            storage: None,
        };
        page.validate(&descriptor).unwrap();

        let mut corrupt = page;
        corrupt.gaussians[0].scale_opacity.opacity = 0.25;
        assert!(matches!(
            corrupt.validate(&descriptor),
            Err(LodPageValidationError::ContentHashMismatch { .. })
        ));
    }

    #[test]
    fn signed_zero_has_canonical_hash() {
        let positive = gaussian([0.0, 1.0, 2.0]);
        let negative = gaussian([-0.0, 1.0, 2.0]);
        assert_eq!(
            stable_gaussian_hash(&positive),
            stable_gaussian_hash(&negative)
        );
    }

    #[test]
    fn portable_gaussian_accepts_finite_opacity_above_one() {
        for opacity in [1.0_f32, 2.0, f32::MAX] {
            let mut sample = gaussian([0.0, 1.0, 2.0]);
            sample.scale_opacity.opacity = opacity;
            assert_eq!(validate_gaussian(&sample), Ok(()));
        }
    }

    #[test]
    fn page_round_trips_planar_storage() {
        let cloud: PlanarGaussian3d = vec![gaussian([0.0; 3]), gaussian([1.0; 3])].into();
        let page = PlanarGaussian3dPage::from_planar(LodPageId(9), &cloud);
        assert_eq!(page.gaussians.len(), 2);
        let round_trip = page.into_planar();
        assert_eq!(
            round_trip.iter().collect::<Vec<_>>(),
            cloud.iter().collect::<Vec<_>>()
        );
    }
}
