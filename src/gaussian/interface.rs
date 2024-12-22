use bevy::{
    prelude::*,
    render::primitives::Aabb,
};

#[cfg(feature = "sort_rayon")]
use rayon::prelude::*;

use crate::gaussian::f32::{
    Position,
    Positions,
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

    fn position_iter(&self) -> Positions<'_>;

    #[cfg(feature = "sort_rayon")]
    fn position_par_iter(&self) -> impl IndexedParallelIterator<Item = &Position> + '_;

    fn subset(&self, indicies: &[usize]) -> Self;

    fn from_packed(packed_array: Vec<Self::PackedType>) -> Self;

    fn visibility(&self, index: usize) -> f32;
    fn visibility_mut(&mut self, index: usize) -> &mut f32;

    fn resize_to_square(&mut self);
}

impl<T> FromIterator<T::PackedType> for T
where
    T: CommonCloud,
{
    fn from_iter<I: IntoIterator<Item = T::PackedType>>(iter: I) -> Self {
        let packed = iter.into_iter().collect();
        T::from_packed(packed)
    }
}

impl<T> From<Vec<T::PackedType>> for T
where
    T: CommonCloud,
{
    fn from(packed: Vec<T::PackedType>) -> Self {
        T::from_packed(packed)
    }
}



pub trait TestCloud {
    fn test_model() -> Self;
}


// TODO: CloudSlice and CloudStream traits
