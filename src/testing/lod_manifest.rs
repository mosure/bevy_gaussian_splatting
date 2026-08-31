//! Synthetic manifest adapters for lifecycle-only LoD qualification.

use crate::gaussian::formats::{
    planar_3d_chunked::{LodIndexRange, LodPageEncoding, LodPageKind},
    planar_3d_lod::{
        EXTERNAL_SPATIAL_MOMENT_MERGE_BUILDER_ABI_VERSION, GaussianLodManifest,
        GaussianLodMorphMap, LOD_MORPH_MAP_SCHEMA_VERSION, LOD_REQUIRED_FEATURE_MONOTONE_MORPH_MAP,
        LodReducerKind, LodValidationError, SPATIAL_MOMENT_MERGE_VERSION,
        lod_config_fingerprint_for_reducer,
    },
};

/// Upgrades a small, already-valid in-memory builder manifest into a synthetic
/// ABI-16 fixture for package/render lifecycle tests.
///
/// Every internal node receives one positive, monotone run per parent record.
/// Quotient/remainder partitioning covers the complete concatenated immediate
/// child cohort exactly; leaves receive empty, index-aligned run ranges.
///
/// This helper changes only manifest correspondence and version metadata. It
/// does **not** spatially refit representatives or establish ABI-16 image,
/// quality, or error-certificate evidence. Never use its output as a release
/// artifact or as visual-quality qualification.
pub fn upgrade_manifest_to_synthetic_abi16_lifecycle_fixture(
    mut manifest: GaussianLodManifest,
) -> Result<GaussianLodManifest, LodValidationError> {
    manifest.validate()?;

    let mut node_runs = Vec::with_capacity(manifest.nodes.len());
    let mut child_run_lengths = Vec::new();
    for node in &manifest.nodes {
        let start = u32::try_from(child_run_lengths.len())
            .map_err(|_| LodValidationError::CountOverflow("morph child runs"))?;
        if node.is_leaf() {
            node_runs.push(LodIndexRange { start, count: 0 });
            continue;
        }

        let child_end = node
            .children
            .end()
            .ok_or(LodValidationError::InvalidChildRange(node.id))?;
        let child_start = usize::try_from(node.children.start)
            .map_err(|_| LodValidationError::CountOverflow("morph child range"))?;
        let child_end = usize::try_from(child_end)
            .map_err(|_| LodValidationError::CountOverflow("morph child range"))?;
        let children = manifest
            .nodes
            .get(child_start..child_end)
            .ok_or(LodValidationError::InvalidChildRange(node.id))?;
        let child_count = children.iter().try_fold(0_u64, |total, child| {
            total
                .checked_add(u64::from(child.representation.count))
                .ok_or(LodValidationError::CountOverflow(
                    "morph immediate-child records",
                ))
        })?;
        let parent_count = node.representation.count;
        if child_count < u64::from(parent_count) {
            return Err(LodValidationError::MorphChildCoverageMismatch {
                node: node.id,
                expected: child_count,
                actual: u64::from(parent_count),
            });
        }

        let base = child_count / u64::from(parent_count);
        let remainder = child_count % u64::from(parent_count);
        for parent_record in 0..parent_count {
            let run = base + u64::from(u64::from(parent_record) < remainder);
            let run = u16::try_from(run).map_err(|_| {
                LodValidationError::MorphRecordCapacityExceeded(
                    u32::try_from(run).unwrap_or(u32::MAX),
                )
            })?;
            debug_assert!(run > 0);
            child_run_lengths.push(run);
        }
        node_runs.push(LodIndexRange {
            start,
            count: parent_count,
        });
    }

    let compressed_representative_sh_degree = manifest.pages.iter().find_map(|page| {
        matches!(page.kind, LodPageKind::Representatives)
            .then_some(page.encoding)
            .and_then(|encoding| match encoding {
                LodPageEncoding::F16Sh { degree } => Some(degree),
                LodPageEncoding::F32Planar => None,
            })
    });
    manifest.header.required_features |= LOD_REQUIRED_FEATURE_MONOTONE_MORPH_MAP;
    manifest.build.reducer = LodReducerKind::MomentMerge;
    manifest.build.builder_abi_version = EXTERNAL_SPATIAL_MOMENT_MERGE_BUILDER_ABI_VERSION;
    manifest.build.reducer_version = SPATIAL_MOMENT_MERGE_VERSION;
    manifest.build.config_fingerprint = lod_config_fingerprint_for_reducer(
        manifest.build.settings,
        compressed_representative_sh_degree,
        SPATIAL_MOMENT_MERGE_VERSION,
    );
    manifest.morph_map = Some(GaussianLodMorphMap {
        schema_version: LOD_MORPH_MAP_SCHEMA_VERSION,
        node_runs,
        child_run_lengths,
    });
    manifest.validate()?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gaussian::formats::planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
        testing::LodTestScene,
    };

    #[test]
    fn synthetic_abi16_lifecycle_map_is_index_aligned_positive_and_complete() {
        let built = build_planar_3d_lod(
            &LodTestScene::nested_octants(3).cloud(),
            GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 8,
                support_sigma: 3.0,
            },
        )
        .unwrap();
        let manifest =
            upgrade_manifest_to_synthetic_abi16_lifecycle_fixture(built.manifest).unwrap();
        let morph = manifest.morph_map.as_ref().unwrap();

        assert_ne!(
            manifest.header.required_features & LOD_REQUIRED_FEATURE_MONOTONE_MORPH_MAP,
            0
        );
        assert_eq!(
            manifest.build.builder_abi_version,
            EXTERNAL_SPATIAL_MOMENT_MERGE_BUILDER_ABI_VERSION
        );
        assert_eq!(manifest.build.reducer, LodReducerKind::MomentMerge);
        assert_eq!(manifest.build.reducer_version, SPATIAL_MOMENT_MERGE_VERSION);
        assert_eq!(
            manifest.build.config_fingerprint,
            lod_config_fingerprint_for_reducer(
                manifest.build.settings,
                None,
                SPATIAL_MOMENT_MERGE_VERSION,
            )
        );
        assert_eq!(morph.schema_version, LOD_MORPH_MAP_SCHEMA_VERSION);
        assert_eq!(morph.node_runs.len(), manifest.nodes.len());
        for (node_index, node) in manifest.nodes.iter().enumerate() {
            let runs = manifest.morph_child_run_lengths_at(node_index).unwrap();
            if node.is_leaf() {
                assert!(runs.is_empty());
                continue;
            }
            assert_eq!(runs.len(), node.representation.count as usize);
            assert!(runs.iter().all(|run| *run > 0));
            let child_end = node.children.end().unwrap() as usize;
            let child_count = manifest.nodes[node.children.start as usize..child_end]
                .iter()
                .map(|child| u64::from(child.representation.count))
                .sum::<u64>();
            assert_eq!(
                runs.iter().map(|run| u64::from(*run)).sum::<u64>(),
                child_count
            );
        }
        manifest.validate().unwrap();
    }
}
