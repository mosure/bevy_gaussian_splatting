use bevy::{
    prelude::*,
    render::primitives::Aabb,
};

#[cfg(feature = "sort_rayon")]
use rayon::prelude::*;

use crate::gaussian::{
    f32::{
        Position,
        PositionVisibility,
    },
    iter::{
        PositionIter,
        PositionParIter,
    },
};


pub trait CommonCloud {
    type PackedType;
    // type GpuPlanarType;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize;
    fn len_sqrt_ceil(&self) -> usize {
        (self.len() as f32).sqrt().ceil() as usize
    }
    fn square_len(&self) -> usize {
        self.len_sqrt_ceil().pow(2)
    }

    fn compute_aabb(&self) -> Option<Aabb> {
        if self.is_empty() {
            return None;
        }

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        // TODO: find a more correct aabb bound derived from scalar max gaussian scale
        let max_scale = 0.1;

        for position in self.position_iter() {
            min = min.min(Vec3::from(*position) - Vec3::splat(max_scale));
            max = max.max(Vec3::from(*position) + Vec3::splat(max_scale));
        }

        Aabb::from_min_max(min, max).into()
    }

    fn subset(&self, indicies: &[usize]) -> Self;

    fn from_packed(packed_array: Vec<Self::PackedType>) -> Self;

    fn visibility(&self, index: usize) -> f32;
    fn visibility_mut(&mut self, index: usize) -> &mut f32;

    fn resize_to_square(&mut self);


    // TODO: type erasure for position iterators
    fn position_iter(&self) -> PositionIter<'_>;

    #[cfg(feature = "sort_rayon")]
    fn position_par_iter(&self) -> PositionParIter<'_>;
}



pub trait TestCloud {
    fn test_model() -> Self;
}


// TODO: CloudSlice and CloudStream traits
