use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};

use bevy::{
    asset::uuid_handle,
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};

use super::{GaussianLodSpatialTransitionSettings, GpuLodSpatialMorph};
use crate::stream::memory::{LodMemoryCategory, LodMemoryLease, LodMemoryLedger};

const SHADER: Handle<Shader> = uuid_handle!("4b1db7a8-b4b1-480b-aaba-9e6a749fd3df");

/// Traversal's allocations and explicit offsets. Offsets address State.words,
/// excluding its 64-byte header; Gaussian capacity excludes the optional tail.
pub(crate) struct SpatialMorphBuffers<'a> {
    pub config: &'a Buffer,
    pub nodes: &'a Buffer,
    pub pages: &'a Buffer,
    pub feedback: &'a Buffer,
    pub entries: &'a Buffer,
    pub indirect: &'a Buffer,
    pub selected_nodes_offset: u32,
    pub selected_page_capacity: u32,
    pub frontier_nodes: u32,
    pub entry_capacity: u32,
}

struct Mapping {
    buffer: Buffer,
    _source: GpuLodSpatialMorph,
    _lease: LodMemoryLease,
}

struct Pipelines {
    layout: BindGroupLayout,
    stages: [CachedComputePipelineId; 3],
}

#[derive(Resource, Default)]
pub(crate) struct SpatialMorphPipelines {
    pipeline: Option<Pipelines>,
    maps: HashMap<usize, Weak<Mapping>>,
}

pub(crate) struct SpatialMorphState {
    bindings: BindGroup,
    mapping: Arc<Mapping>,
    parameters: Buffer,
    scratch: Buffer,
    entries: Buffer,
    indirect: Buffer,
    entry_capacity: u32,
    frontier_nodes: u32,
    _lease: LodMemoryLease,
}

fn scratch_bytes(frontier: u32) -> u64 {
    // Per-range inclusive {edges,records}, followed by per-group exclusive sums.
    (u64::from(frontier) * 8 + u64::from(frontier.div_ceil(256)) * 8).max(16)
}

fn storage(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        ..storage(binding, true)
    }
}

fn buffer(device: &RenderDevice, label: &'static str, bytes: u64, usage: BufferUsages) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage,
        mapped_at_creation: false,
    })
}

impl SpatialMorphPipelines {
    fn prepare(&mut self, device: &RenderDevice, cache: &PipelineCache) {
        self.maps.retain(|_, map| map.strong_count() > 0);
        self.pipeline.get_or_insert_with(|| {
            let entries = [
                uniform(0),
                uniform(1),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                storage(5, false),
                storage(6, true),
                storage(7, false),
            ];
            let desc = BindGroupLayoutDescriptor::new("lod_spatial_morph", &entries);
            let layout = device.create_bind_group_layout(Some("lod_spatial_morph"), &entries);
            let stages = ["classify", "scan_groups", "emit"].map(|entry_point| {
                cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(format!("lod_spatial_{entry_point}").into()),
                    layout: vec![desc.clone()],
                    shader: SHADER,
                    entry_point: Some(entry_point.into()),
                    ..default()
                })
            });
            Pipelines { layout, stages }
        });
    }

    fn mapping(
        &mut self,
        device: &RenderDevice,
        ledger: &LodMemoryLedger,
        source: &GpuLodSpatialMorph,
    ) -> Result<Arc<Mapping>, String> {
        let key = Arc::as_ptr(&source.words) as usize;
        if let Some(mapping) = self.maps.get(&key).and_then(Weak::upgrade) {
            return Ok(mapping);
        }
        let lease = ledger
            .try_reserve(LodMemoryCategory::CompactionGpu, source.mapping_bytes())
            .map_err(|error| error.to_string())?;
        let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("lod_spatial_correspondence"),
            contents: bytemuck::cast_slice(source.words.as_slice()),
            usage: BufferUsages::STORAGE,
        });
        lease.mark_gpu_materialized();
        let mapping = Arc::new(Mapping {
            buffer,
            _source: source.clone(),
            _lease: lease,
        });
        self.maps.insert(key, Arc::downgrade(&mapping));
        Ok(mapping)
    }
}

impl SpatialMorphState {
    /// Conservative per-view charge; shared mapping bytes may already have an
    /// owner in the global ledger. Traversal separately owns the entry tail.
    pub(crate) fn extra_gpu_bytes(
        frontier: u32,
        source: &GpuLodSpatialMorph,
    ) -> Result<u64, String> {
        source
            .mapping_bytes()
            .checked_add(scratch_bytes(frontier))
            .and_then(|bytes| bytes.checked_add(32))
            .ok_or_else(|| "spatial GPU allocation overflow".into())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &RenderDevice,
        queue: &RenderQueue,
        cache: &PipelineCache,
        pipelines: &mut SpatialMorphPipelines,
        ledger: &LodMemoryLedger,
        source: &GpuLodSpatialMorph,
        settings: &GaussianLodSpatialTransitionSettings,
        buffers: SpatialMorphBuffers<'_>,
    ) -> Result<Self, String> {
        settings.validate()?;
        if source.mapping_bytes() > settings.max_mapping_bytes {
            return Err("spatial correspondence exceeds this camera's mapping budget".into());
        }
        let scratch_size = scratch_bytes(buffers.frontier_nodes);
        let limits = device.limits();
        for size in [source.mapping_bytes(), scratch_size] {
            if size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size {
                return Err(
                    "spatial correspondence or scratch exceeds a device buffer limit".into(),
                );
            }
        }
        let lease = ledger
            .try_reserve(LodMemoryCategory::CompactionGpu, scratch_size + 32)
            .map_err(|error| error.to_string())?;
        pipelines.prepare(device, cache);
        let mapping = pipelines.mapping(device, ledger, source)?;
        let parameters = buffer(
            device,
            "lod_spatial_limits",
            32,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        queue.write_buffer(
            &parameters,
            0,
            bytemuck::cast_slice(&[
                buffers.entry_capacity,
                buffers.frontier_nodes,
                buffers.selected_nodes_offset,
                buffers.selected_page_capacity,
                settings.max_transition_nodes,
                settings.max_transition_records,
                0u32,
                0u32,
            ]),
        );
        let scratch = buffer(
            device,
            "lod_spatial_scan",
            scratch_size,
            BufferUsages::STORAGE,
        );
        let bindings = Self::bindings(
            device,
            &pipelines.pipeline.as_ref().unwrap().layout,
            &parameters,
            &scratch,
            &mapping,
            &buffers,
        );
        lease.mark_gpu_materialized();
        Ok(Self {
            bindings,
            mapping,
            parameters,
            scratch,
            entries: buffers.entries.clone(),
            indirect: buffers.indirect.clone(),
            entry_capacity: buffers.entry_capacity,
            frontier_nodes: buffers.frontier_nodes,
            _lease: lease,
        })
    }

    fn bindings(
        device: &RenderDevice,
        layout: &BindGroupLayout,
        parameters: &Buffer,
        scratch: &Buffer,
        mapping: &Mapping,
        buffers: &SpatialMorphBuffers<'_>,
    ) -> BindGroup {
        device.create_bind_group(
            "lod_spatial_morph",
            layout,
            &BindGroupEntries::sequential((
                buffers.config.as_entire_binding(),
                parameters.as_entire_binding(),
                buffers.nodes.as_entire_binding(),
                buffers.pages.as_entire_binding(),
                buffers.feedback.as_entire_binding(),
                buffers.entries.as_entire_binding(),
                mapping.buffer.as_entire_binding(),
                scratch.as_entire_binding(),
            )),
        )
    }

    pub(crate) fn rebind_snapshot(
        &mut self,
        device: &RenderDevice,
        pipelines: &SpatialMorphPipelines,
        buffers: SpatialMorphBuffers<'_>,
    ) {
        self.bindings = Self::bindings(
            device,
            &pipelines.pipeline.as_ref().unwrap().layout,
            &self.parameters,
            &self.scratch,
            &self.mapping,
            &buffers,
        );
    }

    pub(crate) fn mapping_buffer(&self) -> &Buffer {
        &self.mapping.buffer
    }

    pub(crate) fn is_ready(
        &self,
        cache: &PipelineCache,
        pipelines: &SpatialMorphPipelines,
    ) -> bool {
        pipelines.pipeline.as_ref().is_some_and(|pipeline| {
            pipeline
                .stages
                .iter()
                .all(|&id| cache.get_compute_pipeline(id).is_some())
        })
    }

    /// Runs before traversal feedback is copied, so admitted parent reads are
    /// included in the same snapshot's residency/pinning receipt.
    pub(crate) fn encode(
        &self,
        context: &mut RenderContext,
        cache: &PipelineCache,
        pipelines: &SpatialMorphPipelines,
    ) {
        context.command_encoder().copy_buffer_to_buffer(
            &self.indirect,
            0,
            &self.entries,
            u64::from(self.entry_capacity) * 8,
            32,
        );
        let Some(pipeline) = &pipelines.pipeline else {
            return;
        };
        let [Some(classify), Some(scan), Some(emit)] =
            pipeline.stages.map(|id| cache.get_compute_pipeline(id))
        else {
            return;
        };
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("current camera spatial transitions"),
                ..default()
            });
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.set_pipeline(classify);
        pass.dispatch_workgroups(self.frontier_nodes.div_ceil(256), 1, 1);
        pass.set_pipeline(scan);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(emit);
        pass.dispatch_workgroups(self.frontier_nodes.div_ceil(256), 1, 1);
    }
}
