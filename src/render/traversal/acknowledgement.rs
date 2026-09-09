//! Fenced image receipts shared by hierarchy renderers and package publishers.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use bevy::{asset::AssetId, prelude::*};

use super::GpuLodTraversalSettings;
use crate::{PlanarGaussian3d, PlanarGaussian3dHandle};

/// Submission counters belong to one renderer's diagnostics timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuLodDrawRenderer {
    GaussianPoints,
    OrderedQuads,
}

/// A completed image consumed this immutable, pinned residency generation.
#[derive(Clone, Debug)]
pub struct GpuLodDrawAcknowledgement {
    pub renderer: GpuLodDrawRenderer,
    pub submission: u64,
    pub residency_generation: u64,
    pub source: AssetId<PlanarGaussian3d>,
}

#[derive(Resource, Clone, Default)]
pub struct GpuLodDrawAcknowledgements(
    Arc<Mutex<HashMap<(Entity, Entity), GpuLodDrawAcknowledgement>>>,
);

impl GpuLodDrawAcknowledgements {
    pub fn get(&self, camera: Entity, cloud: Entity) -> Option<GpuLodDrawAcknowledgement> {
        self.0.lock().unwrap().get(&(camera, cloud)).cloned()
    }

    /// Called only after the renderer maps its same-submission success header.
    pub(crate) fn publish(
        &self,
        camera: Entity,
        cloud: Entity,
        receipt: GpuLodDrawAcknowledgement,
    ) {
        let mut receipts = self.0.lock().unwrap();
        let key = (camera, cloud);
        if receipts.get(&key).is_none_or(|old| {
            receipt.residency_generation > old.residency_generation
                || (receipt.residency_generation == old.residency_generation
                    && receipt.source == old.source
                    && (receipt.renderer != old.renderer || receipt.submission >= old.submission))
        }) {
            receipts.insert(key, receipt);
        }
    }

    pub(super) fn clear(&self) {
        self.0.lock().unwrap().clear();
    }
}

pub(super) fn prune(
    receipts: Res<GpuLodDrawAcknowledgements>,
    cameras: Query<(), With<GpuLodTraversalSettings>>,
    clouds: Query<(), With<PlanarGaussian3dHandle>>,
) {
    receipts
        .0
        .lock()
        .unwrap()
        .retain(|(camera, cloud), _| cameras.contains(*camera) && clouds.contains(*cloud));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_switch_preserves_residency_generation_order() {
        let mut world = World::new();
        let camera = world.spawn_empty().id();
        let cloud = world.spawn_empty().id();
        let receipts = GpuLodDrawAcknowledgements::default();
        let receipt = |renderer, submission, residency_generation| GpuLodDrawAcknowledgement {
            renderer,
            submission,
            residency_generation,
            source: Handle::<PlanarGaussian3d>::default().id(),
        };
        receipts.publish(
            camera,
            cloud,
            receipt(GpuLodDrawRenderer::GaussianPoints, 100, 2),
        );
        receipts.publish(
            camera,
            cloud,
            receipt(GpuLodDrawRenderer::OrderedQuads, 1, 2),
        );
        assert_eq!(
            receipts.get(camera, cloud).unwrap().renderer,
            GpuLodDrawRenderer::OrderedQuads
        );
        receipts.publish(
            camera,
            cloud,
            receipt(GpuLodDrawRenderer::GaussianPoints, 101, 1),
        );
        assert_eq!(
            receipts.get(camera, cloud).unwrap().renderer,
            GpuLodDrawRenderer::OrderedQuads
        );
        receipts.publish(
            camera,
            cloud,
            receipt(GpuLodDrawRenderer::OrderedQuads, 2, 3),
        );
        receipts.publish(
            camera,
            cloud,
            receipt(GpuLodDrawRenderer::OrderedQuads, 1, 3),
        );
        let latest = receipts.get(camera, cloud).unwrap();
        assert_eq!((latest.residency_generation, latest.submission), (3, 2));
    }
}
