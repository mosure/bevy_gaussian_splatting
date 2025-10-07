use bevy::prelude::*;
use bevy::render::{
    Render, RenderApp, RenderSystems,
    render_asset::RenderAssets,
    render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel, RenderGraphExt},
    render_resource::{
        BindGroup, BindGroupEntry, BindingResource, BufferBinding, BufferUsages, BufferInitDescriptor,
        ComputePassDescriptor, CachedPipelineState,
    },
    renderer::{RenderContext, RenderDevice},
    view::ViewUniformOffset,
};
use bevy::core_pipeline::prepass::PreviousViewUniformOffset;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_interleave::{interface::storage::PlanarStorageBindGroup, prelude::{PlanarSync, GpuPlanar}};
use bytemuck::{Pod, Zeroable};

use crate::{
    GaussianCamera,
    gaussian::formats::planar_3d::{Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle},
    render::{
        CloudPipeline, CloudUniform, GaussianComputeViewBindGroup, GaussianMaterialOverrideBindGroup,
        GaussianUniformBindGroups,
    },
    sort::radix::RadixSortLabel,
};
use crate::pbr_decomposition::{
    settings::PbrDecompositionSettings,
    normal_estimation::{NormalEstimationPipeline, NormalEstimationBuffers},
    synthetic_views::{SyntheticViewsPipeline, SyntheticViewsBuffers},
    material_separation::{MaterialSeparationPipeline, MaterialSeparationBuffers, GpuMaterialBuffers},
};
use crate::pbr_decomposition::synthetic_views::GpuSyntheticViewsBuffers;
use bevy::render::extract_component::DynamicUniformIndex;

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Component)]
pub enum DecompositionStatus {
    Pending,
    InProgress,
    Complete,
    Failed,
}

impl Default for DecompositionStatus {
    fn default() -> Self {
        Self::Pending
    }
}

#[derive(Component, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct DecomposedPbrMaterial {
    pub status: DecompositionStatus,
    pub progress: f32,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct PbrDecompositionLabel;

// Copy of settings in render world
#[derive(Resource, Clone)]
pub struct RuntimeSettings(pub PbrDecompositionSettings);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct NormalSettingsUniform {
    spatial_sigma: f32,
    color_sigma: f32,
    confidence_threshold: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SyntheticViewSettingsUniform {
    num_views: u32,
    near_normal_angle_cos: f32,
    sh_frame: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MaterialSettingsUniform {
    roughness_min: f32,
    roughness_max: f32,
    metallic_saturation_threshold: f32,
    metallic_min_threshold: f32,
}

pub fn extract_settings_to_render(mut render_commands: Commands, mut main_world: ResMut<bevy::render::MainWorld>) {
    let settings = main_world.resource::<PbrDecompositionSettings>().clone();
    render_commands.insert_resource(RuntimeSettings(settings));
}

pub fn update_pbr_buffers(
    render_device: Res<RenderDevice>,
    gpu_planars: Res<RenderAssets<<Gaussian3d as bevy_interleave::prelude::PlanarSync>::GpuPlanarType>>,
    mut normals_res: ResMut<NormalEstimationBuffers>,
    mut stats_res: ResMut<SyntheticViewsBuffers>,
    mut mats_res: ResMut<MaterialSeparationBuffers>,
) {
    for (asset_id, cloud) in gpu_planars.iter() {
        let count = cloud.len();

        normals_res
            .asset_map
            .entry(asset_id)
            .or_insert_with(|| crate::pbr_decomposition::normal_estimation::GpuNormalBuffers::new(count, &render_device));

        stats_res
            .asset_map
            .entry(asset_id)
            .or_insert_with(|| GpuSyntheticViewsBuffers::new(count, &render_device));

        mats_res
            .asset_map
            .entry(asset_id)
            .or_insert_with(|| GpuMaterialBuffers::new(count, &render_device));
    }
}

pub struct PbrDecompositionNode3d {
    views: QueryState<(
        &'static GaussianComputeViewBindGroup,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
    )>,
    gaussian_clouds: QueryState<(
        &'static PlanarStorageBindGroup<Gaussian3d>,
        &'static PlanarGaussian3dHandle,
        &'static DynamicUniformIndex<CloudUniform>,
        Option<&'static GaussianMaterialOverrideBindGroup>,
    )>,
    initialized: bool,
}

impl FromWorld for PbrDecompositionNode3d {
    fn from_world(world: &mut World) -> Self {
        Self { views: world.query(), gaussian_clouds: world.query(), initialized: false }
    }
}

impl Node for PbrDecompositionNode3d {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<NormalEstimationPipeline>();
        let pipeline_cache = world.resource::<bevy::render::render_resource::PipelineCache>();
        if let CachedPipelineState::Ok(_) = pipeline_cache.get_compute_pipeline_state(pipeline.pipeline) {
            self.initialized = true;
        }
        self.views.update_archetypes(world);
        self.gaussian_clouds.update_archetypes(world);
    }

    fn run(&self, _graph: &mut RenderGraphContext, render_context: &mut RenderContext, world: &World) -> Result<(), NodeRunError> {
        if !self.initialized { return Ok(()); }

        let device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<bevy::render::render_resource::PipelineCache>();
        let cloud_pipeline = world.resource::<CloudPipeline<Gaussian3d>>();

        let normals_pipeline = world.resource::<NormalEstimationPipeline>();
        let synth_pipeline = world.resource::<SyntheticViewsPipeline>();
        let mats_pipeline = world.resource::<MaterialSeparationPipeline>();

        let normals_res = world.resource::<NormalEstimationBuffers>();
        let stats_res = world.resource::<SyntheticViewsBuffers>();
        let mats_res = world.resource::<MaterialSeparationBuffers>();

        let rt_settings = world.resource::<RuntimeSettings>().0.clone();

        let gaussian_uniforms = world.resource::<GaussianUniformBindGroups>();

        let gpu_planars = world.resource::<RenderAssets<<Gaussian3d as PlanarSync>::GpuPlanarType>>();

        for (view_bg, view_offset, prev_view_offset) in self.views.iter_manual(world) {
            for (planar_bind_group, handle, cloud_uniform_index, material_override) in
                self.gaussian_clouds.iter_manual(world)
            {
                let asset_id = handle.0.id();
                let Some(normals_gpu) = normals_res.asset_map.get(&asset_id) else { continue; };
                let Some(stats_gpu) = stats_res.asset_map.get(&asset_id) else { continue; };
                let Some(mats_gpu) = mats_res.asset_map.get(&asset_id) else { continue; };
                let Some(cloud_gpu) = gpu_planars.get(handle.0.id()) else { continue; };
                let gaussian_count = cloud_gpu.len() as u32;
                let workgroups = gaussian_count.div_ceil(256).max(1);

                // Uniform buffers
            let normal_settings = NormalSettingsUniform {
                spatial_sigma: rt_settings.normal_spatial_sigma,
                color_sigma: rt_settings.normal_color_sigma,
                confidence_threshold: rt_settings.normal_confidence_threshold,
                _pad: 0.0,
            };
            let normals_uniform_buf = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("normal_settings"),
                contents: bytemuck::bytes_of(&normal_settings),
                usage: BufferUsages::UNIFORM,
            });

            let synth_settings = SyntheticViewSettingsUniform {
                num_views: rt_settings.num_synthetic_views,
                near_normal_angle_cos: (rt_settings.view_near_normal_angle.to_radians()).cos(),
                sh_frame: match rt_settings.sh_coordinate_frame { crate::pbr_decomposition::settings::SHCoordinateFrame::World => 0, _ => 1 },
                _pad: 0,
            };
            let synth_uniform_buf = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("synth_view_settings"),
                contents: bytemuck::bytes_of(&synth_settings),
                usage: BufferUsages::UNIFORM,
            });

            let mat_settings = MaterialSettingsUniform {
                roughness_min: rt_settings.roughness_min,
                roughness_max: rt_settings.roughness_max,
                metallic_saturation_threshold: rt_settings.metallic_saturation_threshold,
                metallic_min_threshold: rt_settings.metallic_min_threshold,
            };
            let mats_uniform_buf = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("material_settings"),
                contents: bytemuck::bytes_of(&mat_settings),
                usage: BufferUsages::UNIFORM,
            });

            // Bind groups for our pipelines (group 3 and 4)
            let normals_bg = device.create_bind_group(
                Some("pbr_normals_bg"),
                &normals_pipeline.bind_group_layout,
                &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding { buffer: &normals_gpu.normals, offset: 0, size: None }),
                }],
            );
            let normals_settings_bg = device.create_bind_group(
                Some("pbr_normals_settings_bg"),
                &normals_pipeline.settings_layout,
                &[BindGroupEntry { binding: 0, resource: normals_uniform_buf.as_entire_binding() }],
            );

            let synth_bg = device.create_bind_group(
                Some("pbr_synth_bg"),
                &synth_pipeline.bind_group_layout,
                &[
                    BindGroupEntry { binding: 0, resource: BindingResource::Buffer(BufferBinding { buffer: &normals_gpu.normals, offset: 0, size: None }) },
                    BindGroupEntry { binding: 1, resource: BindingResource::Buffer(BufferBinding { buffer: &stats_gpu.stats, offset: 0, size: None }) },
                ],
            );
            let synth_settings_bg = device.create_bind_group(
                Some("pbr_synth_settings_bg"),
                &synth_pipeline.settings_layout,
                &[BindGroupEntry { binding: 0, resource: synth_uniform_buf.as_entire_binding() }],
            );

            let mats_bg = device.create_bind_group(
                Some("pbr_mats_bg"),
                &mats_pipeline.bind_group_layout,
                &[
                    BindGroupEntry { binding: 0, resource: BindingResource::Buffer(BufferBinding { buffer: &stats_gpu.stats, offset: 0, size: None }) },
                    BindGroupEntry { binding: 1, resource: BindingResource::Buffer(BufferBinding { buffer: &mats_gpu.materials, offset: 0, size: None }) },
                ],
            );
            let mats_settings_bg = device.create_bind_group(
                Some("pbr_mats_settings_bg"),
                &mats_pipeline.settings_layout,
                &[BindGroupEntry { binding: 0, resource: mats_uniform_buf.as_entire_binding() }],
            );

                // Dispatch passes sequentially
                debug!(gaussian_count, "dispatching PBR decomposition passes");
                let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor::default());

                // estimate_normals
                if let Some(p) = pipeline_cache.get_compute_pipeline(normals_pipeline.pipeline) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &view_bg.value, &[view_offset.offset, prev_view_offset.offset]);
                    if let Some(gu) = &gaussian_uniforms.base_bind_group { pass.set_bind_group(1, gu, &[cloud_uniform_index.index()]); }
                    pass.set_bind_group(2, &planar_bind_group.bind_group, &[]);
                    pass.set_bind_group(3, &normals_bg, &[]);
                    pass.set_bind_group(4, &normals_settings_bg, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }

                // evaluate_synthetic_views
                if let Some(p) = pipeline_cache.get_compute_pipeline(synth_pipeline.pipeline) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &view_bg.value, &[view_offset.offset, prev_view_offset.offset]);
                    if let Some(gu) = &gaussian_uniforms.base_bind_group { pass.set_bind_group(1, gu, &[cloud_uniform_index.index()]); }
                    pass.set_bind_group(2, &planar_bind_group.bind_group, &[]);
                    pass.set_bind_group(3, &synth_bg, &[]);
                    pass.set_bind_group(4, &synth_settings_bg, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }

                // estimate_material_properties
                if let Some(p) = pipeline_cache.get_compute_pipeline(mats_pipeline.pipeline) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &view_bg.value, &[view_offset.offset, prev_view_offset.offset]);
                    if let Some(gu) = &gaussian_uniforms.base_bind_group { pass.set_bind_group(1, gu, &[cloud_uniform_index.index()]); }
                    pass.set_bind_group(2, &planar_bind_group.bind_group, &[]);
                    pass.set_bind_group(3, &mats_bg, &[]);
                    pass.set_bind_group(4, &mats_settings_bg, &[]);
                    let override_bind_group = material_override
                        .map(|bg| &bg.bind_group)
                        .unwrap_or(&cloud_pipeline.fallback_material_override_bind_group);
                    pass.set_bind_group(5, override_bind_group, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                drop(pass);
            }
        }

        Ok(())
    }
}
