#![cfg(feature = "solari")]

use std::marker::PhantomData;

use bevy::solari::realtime::prepare::{
    SolariLightingResources, WORLD_CACHE_SIZE as SOLARI_WORLD_CACHE_SIZE,
};
use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        render_phase::{
            PhaseItem, RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType,
            BufferBindingType, ShaderDefVal, ShaderStages,
        },
        renderer::RenderDevice,
    },
};
use bevy_interleave::prelude::PlanarSync;

use super::{
    CloudPipeline, DrawGaussianInstanced, GaussianUniformBindGroups, SetGaussianUniformBindGroup,
    SetPreviousViewBindGroup,
};
use crate::camera::GaussianCamera;

pub(super) struct CloudPipelineSolariExtras {
    pub layout: BindGroupLayout,
}

pub(super) fn init_cloud_pipeline_extras(
    render_device: &RenderDevice,
) -> CloudPipelineSolariExtras {
    let layout = render_device.create_bind_group_layout(
        Some("gaussian_solari_layout"),
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );

    CloudPipelineSolariExtras { layout }
}

pub(super) fn push_shader_defs(shader_defs: &mut Vec<ShaderDefVal>) {
    shader_defs.push("SOLARI".into());
    shader_defs.push(ShaderDefVal::UInt(
        "SOLARI_WORLD_CACHE_SIZE".into(),
        SOLARI_WORLD_CACHE_SIZE as u32,
    ));
}

pub(super) fn pipeline_layout<R: PlanarSync>(
    pipeline: &CloudPipeline<R>,
    mut base: Vec<BindGroupLayout>,
) -> Vec<BindGroupLayout> {
    base.push(pipeline.solari.layout.clone());
    base
}

pub(super) fn configure_render_app<R: PlanarSync>(render_app: &mut App) {
    render_app.add_plugins(RenderSolariPlugin::<R>::default());
}

pub(super) type DrawGaussians<R: PlanarSync> = (
    SetItemPipeline,
    // SetViewBindGroup<0>,
    SetPreviousViewBindGroup<0>,
    SetGaussianUniformBindGroup<1>,
    SetGaussianSolariBindGroup<4>,
    DrawGaussianInstanced<R>,
);

#[derive(Component)]
pub(super) struct GaussianSolariBindGroup {
    value: BindGroup,
}

fn queue_gaussian_solari_bind_groups<R: PlanarSync>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    gaussian_cloud_pipeline: Res<CloudPipeline<R>>,
    views: Query<
        (
            Entity,
            Option<&SolariLightingResources>,
            Option<&GaussianSolariBindGroup>,
        ),
        With<GaussianCamera>,
    >,
) {
    for (entity, solari_resources, existing_bind_group) in &views {
        match solari_resources {
            Some(resources) => {
                let bind_group = render_device.create_bind_group(
                    "gaussian_solari_bind_group",
                    &gaussian_cloud_pipeline.solari.layout,
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: resources.world_cache_checksums.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: resources.world_cache_radiance.as_entire_binding(),
                        },
                    ],
                );

                commands
                    .entity(entity)
                    .insert(GaussianSolariBindGroup { value: bind_group });
            }
            None => {
                if existing_bind_group.is_some() {
                    commands.entity(entity).remove::<GaussianSolariBindGroup>();
                }
            }
        }
    }
}

pub(super) struct SetGaussianSolariBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetGaussianSolariBindGroup<I> {
    type Param = ();
    type ViewQuery = Option<Read<GaussianSolariBindGroup>>;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _: &P,
        solari_bind_group: Option<ROQueryItem<'w, Self::ViewQuery>>,
        _entity: Option<()>,
        _: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        if let Some(Some(bind_group)) = solari_bind_group {
            pass.set_bind_group(I, &bind_group.value, &[]);
            RenderCommandResult::Success
        } else {
            RenderCommandResult::Skip
        }
    }
}

struct RenderSolariPlugin<R: PlanarSync>(PhantomData<R>);

impl<R: PlanarSync> Default for RenderSolariPlugin<R> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<R: PlanarSync> Plugin for RenderSolariPlugin<R> {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Render,
            queue_gaussian_solari_bind_groups::<R>.in_set(RenderSet::PrepareBindGroups),
        );
    }
}
