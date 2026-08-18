//! Record-bounded decoding for independently addressable LoD pages.
//!
//! This state machine is target-independent. It owns no threads, timers,
//! filesystem handles, or browser APIs; the streaming backend decides when to
//! advance it and how many records are allowed in one scheduling quantum.

use std::{num::NonZeroU32, ops::Range};

use crate::{
    gaussian::formats::{
        planar_3d::Gaussian3d,
        planar_3d_chunked::{
            LOD_PAGE_SCHEMA_VERSION, LodPageDescriptor, LodPageEncoding, LodPageId,
            LodPageValidationError, PlanarGaussian3dPage, StableHasher, hash_gaussian,
            validate_gaussian,
        },
    },
    material::spherical_harmonics::{SH_COEFF_COUNT, SH_DEGREE},
};

use super::{
    F16_SH_PAGE_CONTAINER_VERSION, LodCodecError, LodCodecLimits, PAGE_CONTAINER_MAGIC,
    PAGE_CONTAINER_VERSION, PAGE_HEADER_LEN, PAGE_SH_COEFFICIENT_COUNT, enforce_limit,
    page_payload_len, read_gaussian, read_u16, read_u32, read_u64, sh_coefficient_count_for_degree,
};

/// Resumable decoder used by cooperative browser preprocessing.
///
/// Header validation is constant work performed by [`Self::new`]. Each
/// [`Self::advance`] call then decodes, validates, and hashes at most the
/// requested non-zero number of Gaussian records. The encoded input is owned
/// by the decoder so bytes processed by an earlier call cannot be changed
/// while the operation is suspended.
#[cfg(any(test, target_arch = "wasm32"))]
pub(crate) struct IncrementalLodPageDecoder {
    encoded: Vec<u8>,
    descriptor: LodPageDescriptor,
    state: PageDecodeState,
}

/// Result of one bounded incremental page-decode step.
#[cfg(any(test, target_arch = "wasm32"))]
pub(crate) enum LodPageDecodeProgress {
    Pending {
        /// Retained records decoded by this step, indexed into
        /// [`IncrementalLodPageDecoder::decoded_gaussians`].
        decoded_range: Range<usize>,
    },
    Complete {
        page: PlanarGaussian3dPage,
        /// Retained records decoded by the final step, indexed into
        /// `page.gaussians`.
        decoded_range: Range<usize>,
    },
}

#[cfg(any(test, target_arch = "wasm32"))]
impl IncrementalLodPageDecoder {
    pub(crate) fn new(
        encoded: Vec<u8>,
        descriptor: LodPageDescriptor,
        limits: LodCodecLimits,
    ) -> Result<Self, LodCodecError> {
        // The manifest descriptor is the memory-accounted decoded size. A
        // hostile, otherwise self-consistent container may advertise more
        // records; those records still have to be validated and hashed to
        // preserve error precedence, but they never need to be retained.
        let state = PageDecodeState::new(&encoded, limits, descriptor.gaussian_count)?;
        Ok(Self {
            encoded,
            descriptor,
            state,
        })
    }

    pub(crate) fn decoded_gaussians(&self) -> &[Gaussian3d] {
        &self.state.gaussians
    }

    #[cfg(test)]
    pub(crate) fn encoded_bytes(&self) -> &[u8] {
        &self.encoded
    }

    pub(crate) const fn descriptor(&self) -> &LodPageDescriptor {
        &self.descriptor
    }

    /// Number of encoded bytes whose header or records have been decoded.
    /// The fixed header is consumed during construction.
    #[cfg(test)]
    pub(crate) const fn encoded_offset(&self) -> usize {
        self.state.offset
    }

    pub(crate) fn decoded_count(&self) -> u32 {
        self.state.decoded_count as u32
    }

    pub(crate) const fn total_count(&self) -> u32 {
        self.state.gaussian_count
    }

    pub(crate) fn advance(
        &mut self,
        max_gaussians: NonZeroU32,
    ) -> Result<LodPageDecodeProgress, LodCodecError> {
        assert!(
            !self.state.finished,
            "completed incremental LoD page decoder advanced again"
        );
        let decoded_range = self.state.decode_next(&self.encoded, max_gaussians)?;
        if self.state.is_complete() {
            let actual_hash = self.state.validate_container_hash()?;
            if self.state.encoding != self.descriptor.encoding {
                return Err(LodCodecError::PageEncodingMismatch {
                    expected: self.descriptor.encoding,
                    actual: self.state.encoding,
                });
            }
            validate_page_metadata_with_precomputed_hash(
                self.state.schema_version,
                self.state.page_id,
                self.state.decoded_count,
                &self.descriptor,
                actual_hash,
            )
            .map_err(|error| LodCodecError::PageValidation(error.to_string()))?;
            let page = self.state.take_page();
            Ok(LodPageDecodeProgress::Complete {
                page,
                decoded_range,
            })
        } else {
            Ok(LodPageDecodeProgress::Pending { decoded_range })
        }
    }
}

struct PageDecodeState {
    schema_version: u16,
    page_id: LodPageId,
    gaussian_count: u32,
    encoding: LodPageEncoding,
    encoded_sh_coefficients: usize,
    expected_content_hash: u64,
    content_hasher: StableHasher,
    offset: usize,
    gaussians: Vec<Gaussian3d>,
    decoded_count: usize,
    retained_gaussian_limit: usize,
    finished: bool,
}

impl PageDecodeState {
    fn new(
        encoded: &[u8],
        limits: LodCodecLimits,
        retained_gaussian_limit: u32,
    ) -> Result<Self, LodCodecError> {
        let limits = limits.validate()?;
        enforce_limit("page bytes", encoded.len() as u64, limits.max_page_bytes)?;
        if encoded.len() < PAGE_HEADER_LEN {
            return Err(LodCodecError::Truncated("page header"));
        }
        if encoded[0..8] != PAGE_CONTAINER_MAGIC {
            return Err(LodCodecError::InvalidMagic("page"));
        }
        let version = read_u16(encoded, 8)?;
        let schema_version = read_u16(encoded, 10)?;
        if schema_version != LOD_PAGE_SCHEMA_VERSION {
            return Err(LodCodecError::UnsupportedPageSchema(schema_version));
        }
        let page_id = LodPageId(read_u64(encoded, 12)?);
        if !page_id.is_valid() {
            return Err(LodCodecError::InvalidPageId(page_id));
        }
        let gaussian_count = read_u32(encoded, 20)?;
        enforce_limit(
            "page Gaussians",
            u64::from(gaussian_count),
            u64::from(limits.max_page_gaussians),
        )?;
        if gaussian_count == 0 {
            return Err(LodCodecError::EmptyPage);
        }
        let encoded_sh_coefficients = read_u32(encoded, 24)?;
        let (encoding, encoded_sh_coefficients) = match version {
            PAGE_CONTAINER_VERSION => {
                if encoded_sh_coefficients != PAGE_SH_COEFFICIENT_COUNT {
                    return Err(LodCodecError::IncompatibleSphericalHarmonics {
                        encoded_coefficients: encoded_sh_coefficients,
                        supported_coefficients: PAGE_SH_COEFFICIENT_COUNT,
                    });
                }
                (LodPageEncoding::F32Planar, SH_COEFF_COUNT)
            }
            F16_SH_PAGE_CONTAINER_VERSION => {
                let count = usize::try_from(encoded_sh_coefficients)
                    .map_err(|_| LodCodecError::LengthOverflow)?;
                let degree = (0..=SH_DEGREE as u8)
                    .find(|degree| sh_coefficient_count_for_degree(*degree) == Some(count))
                    .ok_or(LodCodecError::InvalidCompressedShCoefficientCount(
                        encoded_sh_coefficients,
                    ))?;
                (LodPageEncoding::F16Sh { degree }, count)
            }
            _ => return Err(LodCodecError::UnsupportedContainerVersion(version)),
        };
        let payload_len = read_u64(encoded, 28)?;
        let expected_content_hash = read_u64(encoded, 36)?;
        let calculated_payload_len = page_payload_len(gaussian_count, encoding)?;
        if payload_len != calculated_payload_len as u64 {
            return Err(LodCodecError::LengthMismatch {
                expected: calculated_payload_len as u64,
                actual: payload_len,
            });
        }
        let expected_len = PAGE_HEADER_LEN
            .checked_add(calculated_payload_len)
            .ok_or(LodCodecError::LengthOverflow)?;
        if encoded.len() != expected_len {
            return Err(LodCodecError::LengthMismatch {
                expected: expected_len as u64,
                actual: encoded.len() as u64,
            });
        }

        let mut content_hasher = StableHasher::new();
        content_hasher.write(&schema_version.to_le_bytes());
        content_hasher.write(&page_id.0.to_le_bytes());
        content_hasher.write(&u64::from(gaussian_count).to_le_bytes());
        Ok(Self {
            schema_version,
            page_id,
            gaussian_count,
            encoding,
            encoded_sh_coefficients,
            expected_content_hash,
            content_hasher,
            offset: PAGE_HEADER_LEN,
            gaussians: Vec::with_capacity(gaussian_count.min(retained_gaussian_limit) as usize),
            decoded_count: 0,
            retained_gaussian_limit: retained_gaussian_limit as usize,
            finished: false,
        })
    }

    fn decode_next(
        &mut self,
        encoded: &[u8],
        max_gaussians: NonZeroU32,
    ) -> Result<Range<usize>, LodCodecError> {
        let start = self.decoded_count;
        let end = start
            .saturating_add(max_gaussians.get() as usize)
            .min(self.gaussian_count as usize);
        let retained_start = self.gaussians.len();
        for index in start..end {
            let mut next_offset = self.offset;
            let gaussian = read_gaussian(
                encoded,
                &mut next_offset,
                self.encoding,
                self.encoded_sh_coefficients,
            )?;
            validate_gaussian(&gaussian).map_err(|field| LodCodecError::InvalidGaussian {
                index,
                field: format!("{field:?}"),
            })?;
            hash_gaussian(&mut self.content_hasher, &gaussian);
            if self.gaussians.len() < self.retained_gaussian_limit {
                self.gaussians.push(gaussian);
            }
            self.offset = next_offset;
            self.decoded_count += 1;
        }
        Ok(retained_start..self.gaussians.len())
    }

    fn is_complete(&self) -> bool {
        self.decoded_count == self.gaussian_count as usize
    }

    fn validate_container_hash(&self) -> Result<u64, LodCodecError> {
        debug_assert!(self.is_complete());
        let actual_hash = self.content_hasher.finish();
        if actual_hash != self.expected_content_hash {
            return Err(LodCodecError::ChecksumMismatch {
                expected: self.expected_content_hash,
                actual: actual_hash,
            });
        }
        Ok(actual_hash)
    }

    fn take_page(&mut self) -> PlanarGaussian3dPage {
        debug_assert!(self.is_complete());
        debug_assert_eq!(self.gaussians.len(), self.decoded_count);
        self.finished = true;
        PlanarGaussian3dPage {
            schema_version: self.schema_version,
            id: self.page_id,
            gaussians: std::mem::take(&mut self.gaussians),
        }
    }

    fn into_page(mut self) -> PlanarGaussian3dPage {
        self.take_page()
    }
}

pub(super) fn decode_page_container(
    encoded: &[u8],
    limits: LodCodecLimits,
) -> Result<(PlanarGaussian3dPage, LodPageEncoding), LodCodecError> {
    let mut state = PageDecodeState::new(encoded, limits, u32::MAX)?;
    state.decode_next(encoded, NonZeroU32::MAX)?;
    state.validate_container_hash()?;
    let encoding = state.encoding;
    let page = state.into_page();
    Ok((page, encoding))
}

pub(super) fn decode_page_with_descriptor(
    encoded: &[u8],
    descriptor: &LodPageDescriptor,
    limits: LodCodecLimits,
) -> Result<PlanarGaussian3dPage, LodCodecError> {
    let mut state = PageDecodeState::new(encoded, limits, u32::MAX)?;
    state.decode_next(encoded, NonZeroU32::MAX)?;
    let actual_hash = state.validate_container_hash()?;
    if state.encoding != descriptor.encoding {
        return Err(LodCodecError::PageEncodingMismatch {
            expected: descriptor.encoding,
            actual: state.encoding,
        });
    }
    validate_page_metadata_with_precomputed_hash(
        state.schema_version,
        state.page_id,
        state.decoded_count,
        descriptor,
        actual_hash,
    )
    .map_err(|error| LodCodecError::PageValidation(error.to_string()))?;
    Ok(state.into_page())
}

fn validate_page_metadata_with_precomputed_hash(
    schema_version: u16,
    page_id: LodPageId,
    gaussian_count: usize,
    descriptor: &LodPageDescriptor,
    content_hash: u64,
) -> Result<(), LodPageValidationError> {
    // Every decoded Gaussian has already passed `validate_gaussian` before it
    // enters the state. Repeating that scan (and `content_hash`) here would
    // make the nominally constant final step proportional to the full page.
    descriptor.validate()?;
    if schema_version != LOD_PAGE_SCHEMA_VERSION {
        return Err(LodPageValidationError::UnsupportedSchemaVersion {
            expected: LOD_PAGE_SCHEMA_VERSION,
            actual: schema_version,
        });
    }
    if page_id != descriptor.id {
        return Err(LodPageValidationError::PageIdMismatch {
            expected: descriptor.id,
            actual: page_id,
        });
    }
    let actual_count =
        u32::try_from(gaussian_count).map_err(|_| LodPageValidationError::GaussianCountOverflow)?;
    if actual_count != descriptor.gaussian_count {
        return Err(LodPageValidationError::GaussianCountMismatch {
            expected: descriptor.gaussian_count,
            actual: gaussian_count,
        });
    }
    if content_hash != descriptor.content_hash {
        return Err(LodPageValidationError::ContentHashMismatch {
            expected: descriptor.content_hash,
            actual: content_hash,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;
    use crate::{
        gaussian::formats::{
            planar_3d::PlanarGaussian3d,
            planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
        },
        io::lod::{decode_page, encode_page, encode_page_with_encoding},
        testing::LodTestScene,
    };

    fn fixture() -> (PlanarGaussian3dPage, LodPageDescriptor) {
        let scene = LodTestScene::nested_octants(2);
        let cloud = PlanarGaussian3d::from(
            scene
                .gaussians
                .into_iter()
                .map(|entry| entry.gaussian)
                .collect::<Vec<_>>(),
        );
        let built = build_planar_3d_lod(
            &cloud,
            GaussianLodBuildSettings {
                leaf_capacity: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let page = built
            .pages
            .iter()
            .max_by_key(|page| page.gaussians.len())
            .unwrap()
            .clone();
        assert!(page.gaussians.len() > 1);
        let descriptor = built
            .manifest
            .pages
            .iter()
            .find(|descriptor| descriptor.id == page.id)
            .unwrap()
            .clone();
        (page, descriptor)
    }

    fn decode_incrementally(
        encoded: Vec<u8>,
        descriptor: LodPageDescriptor,
        max_gaussians: NonZeroU32,
    ) -> Result<PlanarGaussian3dPage, LodCodecError> {
        let mut decoder =
            IncrementalLodPageDecoder::new(encoded, descriptor, LodCodecLimits::default())?;
        let mut decoded = 0;
        let mut encoded_offset = PAGE_HEADER_LEN;
        loop {
            match decoder.advance(max_gaussians)? {
                LodPageDecodeProgress::Pending { decoded_range } => {
                    let next_encoded_offset = decoder.encoded_offset();
                    assert_eq!(decoded_range.start, decoded);
                    assert!(decoded_range.len() <= max_gaussians.get() as usize);
                    assert!(
                        next_encoded_offset - encoded_offset
                            <= max_gaussians.get() as usize
                                * super::super::MAX_ENCODED_PAGE_GAUSSIAN_BYTES
                    );
                    decoded = decoded_range.end;
                    encoded_offset = next_encoded_offset;
                    assert_eq!(decoder.decoded_count() as usize, decoded);
                    assert_eq!(decoder.encoded_offset(), encoded_offset);
                    assert_eq!(decoder.decoded_gaussians().len(), decoded);
                    assert!(decoder.decoded_count() < decoder.total_count());
                }
                LodPageDecodeProgress::Complete {
                    page,
                    decoded_range,
                } => {
                    assert_eq!(decoded_range.start, decoded);
                    assert_eq!(decoded_range.end, page.gaussians.len());
                    assert!(decoder.encoded_offset() >= encoded_offset);
                    assert_eq!(decoder.encoded_offset(), decoder.encoded_bytes().len());
                    return Ok(page);
                }
            }
        }
    }

    #[test]
    fn record_bounded_f32_decode_matches_one_shot() {
        let (page, descriptor) = fixture();
        let encoded = encode_page(&page).unwrap();
        let one_shot =
            decode_page_with_descriptor(&encoded, &descriptor, LodCodecLimits::default()).unwrap();
        let incremental =
            decode_incrementally(encoded, descriptor, NonZeroU32::new(1).unwrap()).unwrap();
        assert_eq!(incremental, one_shot);
    }

    #[test]
    fn record_bounded_f16_decode_matches_one_shot() {
        let (page, mut descriptor) = fixture();
        let encoding = LodPageEncoding::F16Sh {
            degree: (SH_DEGREE as u8).min(1),
        };
        let encoded = encode_page_with_encoding(&page, encoding).unwrap();
        let canonical = decode_page(&encoded, LodCodecLimits::default()).unwrap();
        descriptor.encoding = encoding;
        descriptor.content_hash = canonical.content_hash();
        let one_shot =
            decode_page_with_descriptor(&encoded, &descriptor, LodCodecLimits::default()).unwrap();
        let incremental =
            decode_incrementally(encoded, descriptor, NonZeroU32::new(2).unwrap()).unwrap();
        assert_eq!(incremental, one_shot);
    }

    #[test]
    fn incremental_errors_match_one_shot_error_order() {
        let (page, descriptor) = fixture();
        let encoded = encode_page(&page).unwrap();

        let mut corrupt = encoded.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        let expected =
            decode_page_with_descriptor(&corrupt, &descriptor, LodCodecLimits::default())
                .unwrap_err();
        assert_eq!(
            decode_incrementally(corrupt, descriptor.clone(), NonZeroU32::new(1).unwrap()),
            Err(expected)
        );

        let mut wrong_encoding = descriptor.clone();
        wrong_encoding.encoding = LodPageEncoding::F16Sh { degree: 0 };
        let expected =
            decode_page_with_descriptor(&encoded, &wrong_encoding, LodCodecLimits::default())
                .unwrap_err();
        assert_eq!(
            decode_incrementally(encoded.clone(), wrong_encoding, NonZeroU32::new(1).unwrap()),
            Err(expected)
        );

        let mut wrong_hash = descriptor;
        wrong_hash.content_hash ^= 1;
        let expected =
            decode_page_with_descriptor(&encoded, &wrong_hash, LodCodecLimits::default())
                .unwrap_err();
        assert_eq!(
            decode_incrementally(encoded, wrong_hash, NonZeroU32::new(1).unwrap()),
            Err(expected)
        );
    }

    #[test]
    fn header_errors_match_before_incremental_state_is_created() {
        let (_, descriptor) = fixture();
        let encoded = vec![0; PAGE_HEADER_LEN - 1];
        let expected =
            decode_page_with_descriptor(&encoded, &descriptor, LodCodecLimits::default())
                .unwrap_err();
        let actual =
            match IncrementalLodPageDecoder::new(encoded, descriptor, LodCodecLimits::default()) {
                Ok(_) => panic!("truncated page unexpectedly constructed a decoder"),
                Err(error) => error,
            };
        assert_eq!(actual, expected);
    }

    #[test]
    fn descriptor_count_caps_retained_memory_without_masking_codec_errors() {
        let (page, mut descriptor) = fixture();
        let encoded = encode_page(&page).unwrap();
        descriptor.gaussian_count = 1;
        descriptor.decoded_len = size_of::<Gaussian3d>() as u64;
        let expected =
            decode_page_with_descriptor(&encoded, &descriptor, LodCodecLimits::default())
                .unwrap_err();
        let mut decoder =
            IncrementalLodPageDecoder::new(encoded, descriptor, LodCodecLimits::default()).unwrap();
        let actual = loop {
            match decoder.advance(NonZeroU32::new(1).unwrap()) {
                Ok(LodPageDecodeProgress::Pending { .. }) => {
                    assert!(decoder.decoded_gaussians().len() <= 1);
                }
                Ok(LodPageDecodeProgress::Complete { .. }) => {
                    panic!("descriptor-count mismatch unexpectedly completed")
                }
                Err(error) => break error,
            }
        };
        assert!(decoder.decoded_gaussians().len() <= 1);
        assert_eq!(actual, expected);
    }
}
