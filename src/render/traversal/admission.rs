use bevy::{
    asset::AssetId,
    prelude::Entity,
    render::{render_asset::RenderAssets, view::RetainedViewEntity},
};
use bevy_interleave::prelude::*;

use crate::{
    Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle,
    gaussian::formats::planar_3d::PlanarStorageGaussian3d,
    render::lod::{LodCompactionBuffers, lod_compaction_asset_id},
    stream::render_commit::LodRenderCandidates,
};

/// The fixed reservation and source identity consumed by either shared renderer.
/// Using prepared capacity, rather than asynchronous visible counts, prevents a
/// changing compacted cut from writing into another cloud's projection range.
pub(crate) struct FixedSource {
    pub asset: AssetId<PlanarGaussian3d>,
    pub capacity: u32,
    pub compaction_generation: Option<u64>,
}

pub(crate) fn fixed_source(
    view: RetainedViewEntity,
    entity: Entity,
    handle: &PlanarGaussian3dHandle,
    candidates: Option<&LodRenderCandidates>,
    assets: &RenderAssets<PlanarStorageGaussian3d>,
    compacted: &LodCompactionBuffers<Gaussian3d>,
) -> Result<FixedSource, &'static str> {
    let asset = lod_compaction_asset_id(handle.0.id(), candidates)
        .ok_or("shared renderer is waiting for its candidate source")?;
    let gpu = assets
        .get(asset)
        .ok_or("shared renderer is waiting for a source asset")?;
    let state = compacted.get(view, entity, asset);
    if candidates.is_some_and(|set| set.candidate_draw_required) && state.is_none() {
        return Err("shared renderer is waiting for candidate compaction");
    }
    let capacity = match state {
        Some(state) => state.output_capacity(),
        None => u32::try_from(gpu.len()).map_err(|_| "shared source exceeds u32 indexing")?,
    };
    Ok(FixedSource {
        asset,
        capacity,
        compaction_generation: state.map(|state| state.generation()),
    })
}

/// Reserve fixed inputs, admit every hierarchy root, then share the remaining
/// projected capacity. The selected limit applies to GPU hierarchy records only.
/// Sorting makes capacity and exact-depth tie behavior independent of query order.
pub(crate) fn admit_traversal_records(
    projected_limit: u32,
    selected_limit: u32,
    fixed_records: u32,
    roots: &[(Entity, u32)],
) -> Result<Vec<(Entity, u32)>, &'static str> {
    let mut admitted = roots.to_vec();
    admitted.sort_unstable_by_key(|(entity, _)| entity.to_bits());
    if admitted.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("GPU hierarchy admission contains a duplicate cloud");
    }
    let minimum = admitted.iter().try_fold(0u32, |sum, (_, count)| {
        (*count > 0).then(|| sum.checked_add(*count)).flatten()
    });
    let minimum = minimum.ok_or("GPU hierarchy root record count is invalid or exceeds u32")?;
    let ceiling = projected_limit
        .checked_sub(fixed_records)
        .ok_or("fixed sources exceed the view projected-record budget")?
        .min(selected_limit);
    if minimum > ceiling {
        return Err("visible GPU hierarchy roots exceed the view projected-record budget");
    }
    if admitted.is_empty() {
        return Ok(admitted);
    }
    let count = u32::try_from(admitted.len()).map_err(|_| "too many visible GPU hierarchies")?;
    let remaining = ceiling - minimum;
    for (index, (_, capacity)) in admitted.iter_mut().enumerate() {
        *capacity += remaining / count + u32::from((index as u32) < remaining % count);
    }
    Ok(admitted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiple_hierarchies_share_one_ceiling_without_losing_a_root_cut() {
        let first = Entity::from_bits(1);
        let second = Entity::from_bits(2);
        let roots = [(second, 7), (first, 3)];
        let admitted = admit_traversal_records(100, 31, 0, &roots).unwrap();
        assert_eq!(admitted, vec![(first, 14), (second, 17)]);
        assert_eq!(
            admitted,
            admit_traversal_records(31, 100, 0, &[(first, 3), (second, 7)]).unwrap()
        );
        assert_eq!(
            admit_traversal_records(10, 10, 0, &roots).unwrap(),
            vec![(first, 3), (second, 7)]
        );
        assert_eq!(
            admit_traversal_records(36, 100, 5, &roots).unwrap(),
            admitted
        );
        assert_eq!(
            admit_traversal_records(15, 10, 5, &roots).unwrap(),
            vec![(first, 3), (second, 7)]
        );
        assert!(admit_traversal_records(14, 100, 5, &roots).is_err());
        assert!(admit_traversal_records(4, 100, 5, &roots).is_err());
        assert!(admit_traversal_records(9, 100, 0, &roots).is_err());
        assert!(admit_traversal_records(100, 9, 0, &roots).is_err());
        assert!(
            admit_traversal_records(u32::MAX, u32::MAX, 0, &[(first, u32::MAX), (second, 1)])
                .is_err()
        );
        assert!(admit_traversal_records(100, 100, 0, &[(first, 3), (first, 7)]).is_err());
    }
}
