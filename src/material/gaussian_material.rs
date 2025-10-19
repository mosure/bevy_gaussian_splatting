use std::collections::{HashMap, HashSet};

use bevy::{
    asset::Asset,
    ecs::query::QueryItem,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::RenderAssets,
        render_resource::{Sampler, SamplerId, TextureView, TextureViewId},
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};
use bevy::render::texture::{FallbackImage, GpuImage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Default)]
pub enum GaussianTextureProjection {
    Xy,
    Xz,
    Yz,
}

impl Default for GaussianTextureProjection {
    fn default() -> Self {
        Self::Xz
    }
}

#[derive(Default, Clone, Copy, Debug, Serialize, Deserialize, Reflect, PartialEq)]
#[reflect(Default)]
pub struct GaussianMaterialBounds {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Asset, Clone, Debug, Reflect)]
#[reflect(Default)]
pub struct GaussianMaterial {
    pub base_color: LinearRgba,
    pub base_color_texture: Option<Handle<Image>>,
    pub texture_projection: GaussianTextureProjection,
    /// Optional manual bounds. When `None`, the gaussian cloud bounds will be used.
    pub bounds: Option<GaussianMaterialBounds>,
}

impl Default for GaussianMaterial {
    fn default() -> Self {
        Self {
            base_color: LinearRgba::WHITE,
            base_color_texture: None,
            texture_projection: GaussianTextureProjection::default(),
            bounds: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GpuGaussianMaterial {
    pub base_color: Vec4,
    pub projection_axis: u32,
    pub bounds_override: Option<GaussianMaterialBounds>,
    pub texture_view: TextureView,
    pub sampler: Sampler,
    pub use_texture: u32,
    pub texture_view_id: TextureViewId,
    pub sampler_id: SamplerId,
}

impl PartialEq for GpuGaussianMaterial {
    fn eq(&self, other: &Self) -> bool {
        self.base_color == other.base_color
            && self.projection_axis == other.projection_axis
            && self.bounds_override == other.bounds_override
            && self.use_texture == other.use_texture
            && self.texture_view_id == other.texture_view_id
            && self.sampler_id == other.sampler_id
    }
}

#[derive(Clone, Debug)]
pub struct CachedGaussianMaterial {
    pub material: GpuGaussianMaterial,
    pub revision: u64,
}

#[derive(Resource, Default)]
pub struct RenderGaussianMaterials {
    pub map: HashMap<AssetId<GaussianMaterial>, CachedGaussianMaterial>,
    pub revision: u64,
}

#[derive(Resource, Default, Clone)]
pub struct ExtractedGaussianMaterials(pub Vec<(AssetId<GaussianMaterial>, GaussianMaterial)>);

#[derive(Component, Clone, Reflect, Deref, DerefMut)]
#[reflect(Component, Default)]
pub struct GaussianMaterialHandle(pub Handle<GaussianMaterial>);

impl Default for GaussianMaterialHandle {
    fn default() -> Self {
        Self(Handle::default())
    }
}

impl ExtractComponent for GaussianMaterialHandle {
    type QueryData = &'static Self;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(component: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(component.clone())
    }
}

pub struct GaussianMaterialPlugin;

impl Plugin for GaussianMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<GaussianMaterial>()
            .register_type::<GaussianMaterialBounds>()
            .register_type::<GaussianMaterialHandle>();
        app.init_asset::<GaussianMaterial>();
        app.add_plugins(ExtractComponentPlugin::<GaussianMaterialHandle>::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<RenderGaussianMaterials>()
                .init_resource::<ExtractedGaussianMaterials>()
                .add_systems(ExtractSchedule, extract_gaussian_materials)
                .add_systems(
                    Render,
                    prepare_gaussian_materials_system.in_set(RenderSystems::PrepareBindGroups),
                );
        }
    }
}

fn prepare_gaussian_materials_system(
    materials: Res<ExtractedGaussianMaterials>,
    images: Res<RenderAssets<GpuImage>>,
    fallback_image: Res<FallbackImage>,
    mut cache: ResMut<RenderGaussianMaterials>,
) {
    let fallback_texture_view = fallback_image.d2.texture_view.clone();
    let fallback_sampler = fallback_image.d2.sampler.clone();
    let mut seen_ids: HashSet<AssetId<GaussianMaterial>> =
        HashSet::with_capacity(materials.0.len());
    let mut revision = cache.revision;

    for (id, material) in materials.0.iter() {
        seen_ids.insert(*id);
        let (texture_view, sampler, use_texture) = if let Some(handle) = material.base_color_texture.clone() {
            if let Some(image) = images.get(&handle) {
                (image.texture_view.clone(), image.sampler.clone(), 1)
            } else {
                (fallback_texture_view.clone(), fallback_sampler.clone(), 0)
            }
        } else {
            (fallback_texture_view.clone(), fallback_sampler.clone(), 0)
        };

        let projection_axis = match material.texture_projection {
            GaussianTextureProjection::Xy => 0,
            GaussianTextureProjection::Xz => 1,
            GaussianTextureProjection::Yz => 2,
        };

        let texture_view_id = texture_view.id();
        let sampler_id = sampler.id();

        let gpu_material = GpuGaussianMaterial {
            base_color: material.base_color.to_vec4(),
            projection_axis,
            bounds_override: material.bounds,
            texture_view,
            sampler,
            use_texture,
            texture_view_id,
            sampler_id,
        };

        let entry = cache.map.get(id).cloned();

        match entry {
            Some(existing) if existing.material == gpu_material => {}
            _ => {
                revision = revision.wrapping_add(1);
                cache.map.insert(
                    *id,
                    CachedGaussianMaterial {
                        material: gpu_material,
                        revision,
                    },
                );
            }
        }
    }

    cache.map.retain(|id, _| seen_ids.contains(id));
    cache.revision = revision;
}

fn extract_gaussian_materials(
    mut commands: Commands,
    materials: Extract<Res<Assets<GaussianMaterial>>>,
) {
    let mut extracted = Vec::with_capacity(materials.len());
    for (id, material) in materials.iter() {
        extracted.push((id, material.clone()));
    }
    commands.insert_resource(ExtractedGaussianMaterials(extracted));
}
