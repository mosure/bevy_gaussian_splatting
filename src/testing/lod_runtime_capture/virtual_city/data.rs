use super::*;
use crate::gaussian::formats::{planar_3d_chunked::*, planar_3d_lod::*};

/// Bounds include three-sigma support of every record emitted by the fixture.
/// City blocks have distinct world-space origins, including off-route blocks.
fn interval_bounds(city: VirtualCityScene, first: u32, count: u32) -> LodBounds {
    let last = first + count - 1;
    let same_row = first / city.grid_width == last / city.grid_width;
    let min_x = if same_row { first % city.grid_width } else { 0 };
    let max_x = if same_row {
        last % city.grid_width
    } else {
        city.grid_width - 1
    };
    LodBounds::new(
        [
            min_x as f32 * 24.0 - 7.0,
            -1.0,
            (first / city.grid_width) as f32 * 24.0 - 7.0,
        ],
        [
            max_x as f32 * 24.0 + 26.0,
            9.0,
            (last / city.grid_width) as f32 * 24.0 + 26.0,
        ],
    )
    .unwrap()
}

pub(super) fn city(config: &VirtualCityConfig) -> VirtualCityScene {
    VirtualCityScene {
        seed: config.seed,
        page_count: config
            .source_gaussians
            .div_ceil(u64::from(config.records_per_page)) as u32,
        gaussians_per_page: config.records_per_page,
        grid_width: config.grid_width,
    }
}

/// No source-sized allocation: one requested leaf or one internal proxy only.
pub(super) fn page(config: &VirtualCityConfig, node: &GaussianLodNode) -> PlanarGaussian3dPage {
    let records = if node.is_leaf() {
        let index = (node.source.start / u64::from(config.records_per_page)) as u32;
        let mut records: Vec<_> = city(config)
            .generate_page(index)
            .unwrap()
            .into_iter()
            .map(|record| record.gaussian)
            .collect();
        records.truncate(node.source.count as usize);
        records
    } else {
        let center = (Vec3::from_array(node.bounds.min) + Vec3::from_array(node.bounds.max)) * 0.5;
        (0..node.representation.count)
            .map(|index| {
                // Bounded, distinct diagnostic proxies; no spatial fit is claimed.
                let offset = ((index % 16) as f32 / 15.0 - 0.5) * 0.5;
                crate::Gaussian3d {
                    position_visibility: [center.x + offset, center.y, center.z, 1.0].into(),
                    scale_opacity: [0.4, 0.4, 0.4, 0.7].into(),
                    rotation: [1.0, 0.0, 0.0, 0.0].into(),
                    ..Default::default()
                }
            })
            .collect()
    };
    PlanarGaussian3dPage::new(node.representation.page, records)
}

/// Real checksums require an explicit O(source records) CPU pass. Only metadata
/// and the current page live at once, and no encoded payload archive is written.
pub(super) fn build(config: &VirtualCityConfig) -> CaptureResult<GaussianLodManifest> {
    config.validate()?;
    let city = city(config);
    let settings = GaussianLodBuildSettings {
        branching_factor: 32,
        leaf_capacity: config.records_per_page,
        support_sigma: 3.0,
    };
    let mut work = vec![(0_u32, city.page_count, None, 0_u16)];
    let mut nodes = Vec::with_capacity(city.page_count as usize * 2 - 1);
    let mut index = 0;
    while index < work.len() {
        let (first, count, parent, depth) = work[index];
        let id = LodNodeId(index as u64 + 1);
        let source_start = u64::from(first) * u64::from(config.records_per_page);
        let source_end = (u64::from(first + count) * u64::from(config.records_per_page))
            .min(config.source_gaussians);
        let bounds = interval_bounds(city, first, count);
        let leaf = count == 1;
        let children = if leaf {
            LodIndexRange::empty()
        } else {
            let children = LodIndexRange {
                start: work.len() as u32,
                count: 2,
            };
            let left = count / 2;
            work.push((first, left, Some(id), depth + 1));
            work.push((first + left, count - left, Some(id), depth + 1));
            children
        };
        let geometric = if leaf {
            0.0
        } else {
            (Vec3::from_array(bounds.max) - Vec3::from_array(bounds.min)).length()
        };
        nodes.push(GaussianLodNode {
            id,
            parent,
            depth,
            bounds,
            children,
            source: LodSourceRange {
                start: source_start,
                count: source_end - source_start,
            },
            morton: LodMortonRange {
                min: u64::from(first),
                max: u64::from(first + count - 1),
            },
            representation: LodPageRange {
                page: LodPageId(id.0),
                offset: 0,
                count: if leaf {
                    (source_end - source_start) as u32
                } else {
                    1
                },
            },
            error: LodError {
                geometric,
                combined: geometric,
                ..Default::default()
            },
            quality: LodQualityInterval { min: 0.0, max: 1.0 },
            high_fidelity_certificate: if leaf { 1.0 } else { 0.0 },
        });
        index += 1;
    }
    drop(work);
    // ABI16's count amplification is a real structural invariant even for a
    // quality-unqualified synthetic fixture. Binary topology may have up to
    // 32x record amplification per rung; directly above 4096-record leaves this
    // requires 256 parent records, then 16, then one at the next coarser rung.
    for index in (0..nodes.len()).rev() {
        if !nodes[index].is_leaf() {
            let children = nodes[index].children;
            let child_records: u32 = nodes
                [children.start as usize..children.end().unwrap() as usize]
                .iter()
                .map(|node| node.representation.count)
                .sum();
            nodes[index].representation.count = child_records.div_ceil(32).max(1);
        }
    }
    let mut pages = Vec::with_capacity(nodes.len());
    let mut stored = 0;
    for (index, node) in nodes.iter().enumerate() {
        let payload = page(config, node);
        let encoded = encode_page(&payload)?;
        pages.push(LodPageDescriptor {
            id: payload.id,
            kind: if node.is_leaf() {
                LodPageKind::SourceLeaves
            } else {
                LodPageKind::Representatives
            },
            encoding: LodPageEncoding::F32Planar,
            gaussian_count: node.representation.count,
            decoded_len: u64::from(node.representation.count)
                * std::mem::size_of::<crate::Gaussian3d>() as u64,
            content_hash: payload.content_hash(),
            bounds: node.bounds,
            storage: Some(LodPageStorage {
                uri: format!("pages/{}.gspage", payload.id.0),
                byte_range: None,
                encoded_len: encoded.len() as u64,
            }),
        });
        stored += u64::from(node.representation.count);
        if index % 1024 == 0 {
            eprintln!(
                "virtual-city prehash: {index}/{} pages (CPU preparation, no GPU)",
                nodes.len()
            );
        }
    }
    let bounds = nodes[0].bounds;
    let max_depth = nodes.iter().map(|node| node.depth).max().unwrap();
    let max_error = nodes[0].error;
    let coarsest_gaussian_count = u64::from(nodes[0].representation.count);
    let manifest = GaussianLodManifest {
        header: GaussianLodManifestHeader {
            magic: LOD_MANIFEST_MAGIC,
            manifest_version: LOD_MANIFEST_VERSION,
            page_schema_version: LOD_PAGE_SCHEMA_VERSION,
            required_features: LOD_CURRENT_REQUIRED_FEATURES,
            source_gaussian_count: config.source_gaussians,
            stored_gaussian_count: stored,
            node_count: nodes.len() as u32,
            page_count: pages.len() as u32,
        },
        scene_bounds: Some(bounds),
        roots: vec![LodNodeId(1)],
        nodes,
        pages,
        build: GaussianLodBuildMetadata {
            settings,
            reducer: LodReducerKind::MomentMerge,
            builder_abi_version: 5,
            reducer_version: EXTERNAL_MOMENT_MERGE_VERSION,
            source_fingerprint: config.seed ^ config.source_gaussians,
            config_fingerprint: lod_config_fingerprint_for_reducer(
                settings,
                None,
                EXTERNAL_MOMENT_MERGE_VERSION,
            ),
        },
        quality: GaussianLodQualityMetadata {
            max_depth,
            coarsest_gaussian_count,
            finest_gaussian_count: config.source_gaussians,
            max_error,
        },
        morph_map: None,
    };
    // This adapter is explicitly a lifecycle fixture, never fitted quality data.
    Ok(crate::testing::upgrade_manifest_to_synthetic_abi16_lifecycle_fixture(manifest)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_virtual_leaf_has_real_distinct_payload_and_exact_source_coverage() {
        let config = VirtualCityConfig {
            source_gaussians: 113,
            records_per_page: 16,
            grid_width: 4,
            ..Default::default()
        };
        let manifest = build(&config).unwrap();
        manifest.validate().unwrap();
        let leaves: Vec<_> = manifest
            .nodes
            .iter()
            .filter(|node| node.is_leaf())
            .collect();
        assert_eq!(leaves.len(), 8);
        assert_eq!(
            leaves.iter().map(|node| node.source.count).sum::<u64>(),
            113
        );
        let mut hashes = std::collections::HashSet::new();
        for node in &manifest.nodes {
            let payload = page(&config, node);
            let descriptor = &manifest.pages[node.id.0 as usize - 1];
            payload.validate(descriptor).unwrap();
            assert_eq!(
                encode_page(&payload).unwrap().len() as u64,
                descriptor.storage.as_ref().unwrap().encoded_len
            );
            if node.is_leaf() {
                assert!(hashes.insert(payload.content_hash()));
                for gaussian in payload.gaussians {
                    let center = Vec3::from(gaussian.position_visibility.position);
                    assert!((0..3).all(|axis| center[axis] >= node.bounds.min[axis]
                        && center[axis] <= node.bounds.max[axis]));
                }
            }
        }
        assert_ne!(leaves[0].bounds, leaves[1].bounds);
        assert_eq!(leaves.last().unwrap().source.count, 1);
    }

    #[test]
    fn virtual_configuration_rejects_unbounded_sources_or_pages() {
        assert!(
            VirtualCityConfig {
                source_gaussians: 100_000_001,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            VirtualCityConfig {
                records_per_page: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            VirtualCityConfig {
                source_gaussians: 100_000_000,
                records_per_page: 1,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert_eq!(
            city(&VirtualCityConfig {
                source_gaussians: 100_000_000,
                ..Default::default()
            })
            .page_count,
            24_415
        );
    }
}
