//! Bounded GPU primitives for offline LoD construction.
//!
//! The external builder owns the global hierarchy and uses this module to
//! Morton-sort bounded source batches and reduce explicit MomentMerge groups.
//!
//! WebGPU does not expose portable `f64`, so MomentMerge accumulation is
//! deterministic f32 on the GPU rather than bit-identical to the CPU's f64
//! reference reduction. Ordering and Morton keys are exact; reduction results
//! are conservatively validated on readback.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use std::{borrow::Cow, error::Error, fmt, mem::size_of, num::NonZeroU64, time::Duration};

use bytemuck::{Pod, Zeroable};

use crate::{
    gaussian::formats::{
        planar_3d::Gaussian3d,
        planar_3d_chunked::{GaussianField, LodBounds, validate_gaussian},
        planar_3d_lod::{
            LOD_MORTON_AXIS_MAX, LOD_MORTON_BITS_PER_AXIS, LodError, canonicalize_gaussian_zeros,
            compare_gaussians,
        },
    },
    material::spherical_harmonics::{SH_COEFF_COUNT, SH_VEC4_PLANES},
};

pub const GPU_LOD_HIERARCHY_SORT_WORKGROUP_SIZE: u32 = 256;
pub const GPU_LOD_HIERARCHY_REDUCE_WORKGROUP_SIZE: u32 = 64;

const SHADER_SOURCE: &str = include_str!("hierarchy.wgsl");
const READBACK_ALIGNMENT: u64 = 256;

fn shader_source() -> String {
    SHADER_SOURCE
        .replace("__SH_VEC4_PLANES__", &SH_VEC4_PLANES.to_string())
        .replace("__SH_COEFF_COUNT__", &SH_COEFF_COUNT.to_string())
        .replace(
            "__LOD_MORTON_BITS_PER_AXIS__",
            &LOD_MORTON_BITS_PER_AXIS.to_string(),
        )
        .replace("__LOD_MORTON_AXIS_MAX__", &LOD_MORTON_AXIS_MAX.to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuLodHierarchyLimits {
    pub max_records: u32,
    pub max_nodes: u32,
    pub max_stage_commands: u32,
    pub max_input_bytes: u64,
    pub max_node_bytes: u64,
    pub max_readback_bytes: u64,
    pub poll_timeout: Duration,
}

impl Default for GpuLodHierarchyLimits {
    fn default() -> Self {
        Self {
            max_records: 65_536,
            // Default build settings require only ~75 nodes at this batch size;
            // leave ample room for smaller pages without allocating 2*N nodes.
            max_nodes: 4_096,
            max_stage_commands: 512,
            max_input_bytes: 64 * 1024 * 1024,
            max_node_bytes: 64 * 1024 * 1024,
            max_readback_bytes: 128 * 1024 * 1024,
            poll_timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ValidatedCapacities {
    input_bytes: u64,
    entry_bytes: u64,
    status_bytes: u64,
    node_bytes: u64,
    stage_stride: u64,
    stage_bytes: u64,
    readback: ReadbackLayout,
}

#[derive(Clone, Copy, Debug)]
struct ReadbackLayout {
    status_offset: u64,
    entry_offset: u64,
    sorted_offset: u64,
    node_offset: u64,
    capacity_bytes: u64,
}

impl GpuLodHierarchyLimits {
    fn validate(self, device: &wgpu::Device) -> Result<ValidatedCapacities, GpuLodHierarchyError> {
        for (name, value) in [
            ("max_records", u64::from(self.max_records)),
            ("max_nodes", u64::from(self.max_nodes)),
            ("max_stage_commands", u64::from(self.max_stage_commands)),
            ("max_input_bytes", self.max_input_bytes),
            ("max_node_bytes", self.max_node_bytes),
            ("max_readback_bytes", self.max_readback_bytes),
        ] {
            if value == 0 {
                return Err(GpuLodHierarchyError::ZeroLimit(name));
            }
        }
        if self.poll_timeout.is_zero() {
            return Err(GpuLodHierarchyError::ZeroLimit("poll_timeout"));
        }

        let padded_records = self
            .max_records
            .checked_next_power_of_two()
            .ok_or(GpuLodHierarchyError::CapacityOverflow("padded records"))?;
        let input_bytes = checked_bytes(self.max_records, size_of::<Gaussian3d>(), "input")?;
        let entry_bytes =
            checked_bytes(padded_records, size_of::<GpuSortEntryRaw>(), "sort entries")?;
        let status_bytes = checked_bytes(self.max_records, size_of::<u32>(), "statuses")?;
        let node_bytes = checked_bytes(self.max_nodes, size_of::<GpuNodeRaw>(), "nodes")?;
        if input_bytes > self.max_input_bytes {
            return Err(GpuLodHierarchyError::ConfiguredByteLimit {
                field: "max_input_bytes",
                required: input_bytes,
                configured: self.max_input_bytes,
            });
        }
        if node_bytes > self.max_node_bytes {
            return Err(GpuLodHierarchyError::ConfiguredByteLimit {
                field: "max_node_bytes",
                required: node_bytes,
                configured: self.max_node_bytes,
            });
        }
        let stage_stride = align_up(
            size_of::<GpuStageParams>() as u64,
            u64::from(device.limits().min_uniform_buffer_offset_alignment.max(1)),
        )?;
        let stage_bytes = stage_stride
            .checked_mul(u64::from(self.max_stage_commands))
            .ok_or(GpuLodHierarchyError::CapacityOverflow("stage commands"))?;
        if stage_bytes > u64::from(u32::MAX) {
            return Err(GpuLodHierarchyError::DynamicOffsetOverflow(stage_bytes));
        }

        let status_offset = 0;
        let entry_offset = align_up(status_bytes, READBACK_ALIGNMENT)?;
        let sorted_offset = align_up(
            entry_offset
                .checked_add(entry_bytes)
                .ok_or(GpuLodHierarchyError::CapacityOverflow("readback entries"))?,
            READBACK_ALIGNMENT,
        )?;
        let node_offset = align_up(
            sorted_offset
                .checked_add(input_bytes)
                .ok_or(GpuLodHierarchyError::CapacityOverflow("readback source"))?,
            READBACK_ALIGNMENT,
        )?;
        let capacity_bytes = node_offset
            .checked_add(node_bytes)
            .ok_or(GpuLodHierarchyError::CapacityOverflow("readback"))?;
        if capacity_bytes > self.max_readback_bytes {
            return Err(GpuLodHierarchyError::ConfiguredByteLimit {
                field: "max_readback_bytes",
                required: capacity_bytes,
                configured: self.max_readback_bytes,
            });
        }

        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < 5 {
            return Err(GpuLodHierarchyError::DeviceLimit {
                field: "storage buffers per compute stage",
                required: 5,
                supported: u64::from(limits.max_storage_buffers_per_shader_stage),
            });
        }
        if limits.max_dynamic_uniform_buffers_per_pipeline_layout < 1 {
            return Err(GpuLodHierarchyError::DeviceLimit {
                field: "dynamic uniform buffers",
                required: 1,
                supported: u64::from(limits.max_dynamic_uniform_buffers_per_pipeline_layout),
            });
        }
        if limits.max_compute_invocations_per_workgroup < GPU_LOD_HIERARCHY_SORT_WORKGROUP_SIZE {
            return Err(GpuLodHierarchyError::DeviceLimit {
                field: "compute workgroup invocations",
                required: u64::from(GPU_LOD_HIERARCHY_SORT_WORKGROUP_SIZE),
                supported: u64::from(limits.max_compute_invocations_per_workgroup),
            });
        }
        for (name, bytes) in [
            ("input storage binding", input_bytes),
            ("sorted storage binding", input_bytes),
            ("sort-entry storage binding", entry_bytes),
            ("status storage binding", status_bytes),
            ("node storage binding", node_bytes),
        ] {
            validate_limit(name, bytes, limits.max_storage_buffer_binding_size)?;
            validate_limit(name, bytes, limits.max_buffer_size)?;
        }
        validate_limit("stage buffer", stage_bytes, limits.max_buffer_size)?;
        validate_limit("readback buffer", capacity_bytes, limits.max_buffer_size)?;
        let workgroups = padded_records.div_ceil(GPU_LOD_HIERARCHY_SORT_WORKGROUP_SIZE);
        validate_limit(
            "sort workgroups",
            u64::from(workgroups),
            u64::from(limits.max_compute_workgroups_per_dimension),
        )?;
        validate_limit(
            "reduction workgroups",
            u64::from(
                self.max_nodes
                    .div_ceil(GPU_LOD_HIERARCHY_REDUCE_WORKGROUP_SIZE),
            ),
            u64::from(limits.max_compute_workgroups_per_dimension),
        )?;

        Ok(ValidatedCapacities {
            input_bytes,
            entry_bytes,
            status_bytes,
            node_bytes,
            stage_stride,
            stage_bytes,
            readback: ReadbackLayout {
                status_offset,
                entry_offset,
                sorted_offset,
                node_offset,
                capacity_bytes,
            },
        })
    }
}

fn checked_bytes(
    count: u32,
    stride: usize,
    name: &'static str,
) -> Result<u64, GpuLodHierarchyError> {
    u64::from(count)
        .checked_mul(stride as u64)
        .ok_or(GpuLodHierarchyError::CapacityOverflow(name))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GpuLodHierarchyError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or(GpuLodHierarchyError::CapacityOverflow("alignment"))
}

fn validate_limit(
    field: &'static str,
    required: u64,
    supported: u64,
) -> Result<(), GpuLodHierarchyError> {
    if required > supported {
        Err(GpuLodHierarchyError::DeviceLimit {
            field,
            required,
            supported,
        })
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuLodHierarchySortedRecord {
    pub morton: u64,
    pub source_index: u64,
    pub gaussian: Gaussian3d,
}

/// One bounded input to the external/global MomentMerge reducer.
///
/// Leaf inputs use the source Gaussian, its support bounds, and zero error.
/// Internal inputs use the representative, conservative node bounds, and the
/// accumulated error emitted for the child by the preceding GPU level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuLodHierarchyReductionInput {
    pub representative: Gaussian3d,
    pub bounds: LodBounds,
    pub inherited_error: LodError,
}

/// One explicit contiguous group in a bounded reduction submission. Ranges
/// address [`GpuLodHierarchyReductionInput`] (or the leaf Gaussian slice) and
/// must partition the submitted input exactly, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuLodHierarchyReductionGroup {
    pub start: u32,
    pub count: u32,
}

/// Device-produced summary for one explicit group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuLodHierarchyReductionOutput {
    pub representative: Gaussian3d,
    pub representative_support: LodBounds,
    pub local_error: LodError,
    pub accumulated_error: LodError,
}

fn sort_stages(padded_count: u32) -> Vec<GpuStageParams> {
    let mut result = Vec::new();
    let mut k = 2_u32;
    while k <= padded_count {
        let mut j = k / 2;
        while j > 0 {
            result.push(GpuStageParams {
                first: [k, j, 0, 0],
                second: [0; 4],
            });
            j /= 2;
        }
        match k.checked_mul(2) {
            Some(next) => k = next,
            None => break,
        }
    }
    result
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuGlobalParams {
    counts: [u32; 4],
    normalization_min: [f32; 4],
    normalization_max: [f32; 4],
    build: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuStageParams {
    first: [u32; 4],
    second: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuSortEntryRaw {
    key_and_source: [u32; 4],
    input_and_valid: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuNodeRaw {
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    representative_support_min: [f32; 4],
    representative_support_max: [f32; 4],
    error: [f32; 4],
    summary_error: [f32; 4],
    topology: [u32; 4],
    morton: [u32; 4],
    page: [u32; 4],
    representative: Gaussian3d,
}

struct Slot {
    globals: wgpu::Buffer,
    input: wgpu::Buffer,
    entries: wgpu::Buffer,
    sorted: wgpu::Buffer,
    statuses: wgpu::Buffer,
    nodes: wgpu::Buffer,
    stages: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct Pipelines {
    initialize: wgpu::ComputePipeline,
    bitonic: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    reduce_external: wgpu::ComputePipeline,
}

pub struct GpuLodHierarchyBuilder {
    limits: GpuLodHierarchyLimits,
    capacities: ValidatedCapacities,
    pipelines: Pipelines,
    slot: Slot,
}

impl GpuLodHierarchyBuilder {
    pub fn new(
        device: &wgpu::Device,
        limits: GpuLodHierarchyLimits,
    ) -> Result<Self, GpuLodHierarchyError> {
        let capacities = limits.validate(device)?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_lod_hierarchy_layout"),
            entries: &[
                uniform_layout_entry(0, size_of::<GpuGlobalParams>() as u64, false),
                uniform_layout_entry(1, size_of::<GpuStageParams>() as u64, true),
                storage_layout_entry(2, capacities.input_bytes, true),
                storage_layout_entry(3, capacities.entry_bytes, false),
                storage_layout_entry(4, capacities.input_bytes, false),
                storage_layout_entry(5, capacities.status_bytes, false),
                storage_layout_entry(6, capacities.node_bytes, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gaussian_lod_hierarchy_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gaussian_lod_hierarchy_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source())),
        });
        let pipeline = |label: &'static str, entry_point: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            initialize: pipeline("gaussian_lod_hierarchy_initialize", "initialize"),
            bitonic: pipeline("gaussian_lod_hierarchy_bitonic", "bitonic_stage"),
            gather: pipeline("gaussian_lod_hierarchy_gather", "gather_sorted"),
            reduce_external: pipeline(
                "gaussian_lod_hierarchy_reduce_external",
                "reduce_external_groups",
            ),
        };
        let slot = create_slot(device, &layout, capacities);
        Ok(Self {
            limits,
            capacities,
            pipelines,
            slot,
        })
    }

    pub const fn limits(&self) -> GpuLodHierarchyLimits {
        self.limits
    }

    /// Canonically Morton-sort one bounded source batch without constructing a
    /// hierarchy for that batch. This is the appropriate first stage for an
    /// external/global build: the returned records may be spilled and merged,
    /// while hierarchy reduction happens only after the global merge.
    pub fn sort_morton_batch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        records: &[Gaussian3d],
        source_index_base: u64,
        normalization_bounds: LodBounds,
        support_sigma: f32,
    ) -> Result<Vec<GpuLodHierarchySortedRecord>, GpuLodHierarchyError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (
                device,
                queue,
                records,
                source_index_base,
                normalization_bounds,
                support_sigma,
            );
            Err(GpuLodHierarchyError::BlockingReadbackUnsupported)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.sort_morton_batch_native(
                device,
                queue,
                records,
                source_index_base,
                normalization_bounds,
                support_sigma,
            )
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn sort_morton_batch_native(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        records: &[Gaussian3d],
        source_index_base: u64,
        normalization_bounds: LodBounds,
        support_sigma: f32,
    ) -> Result<Vec<GpuLodHierarchySortedRecord>, GpuLodHierarchyError> {
        if records.is_empty() {
            return Err(GpuLodHierarchyError::EmptySource);
        }
        if records.len() > self.limits.max_records as usize {
            return Err(GpuLodHierarchyError::BatchTooLarge {
                actual: records.len(),
                limit: self.limits.max_records,
            });
        }
        source_index_base
            .checked_add(records.len() as u64 - 1)
            .ok_or(GpuLodHierarchyError::SourceIndexOverflow)?;
        validate_reduction_support_sigma(support_sigma)?;
        validate_normalization_bounds(normalization_bounds)?;
        let canonical = canonical_records(records)?;
        let padded_count = (records.len() as u32).next_power_of_two();
        let commands = sort_stages(padded_count);
        if commands.len() > self.limits.max_stage_commands as usize {
            return Err(GpuLodHierarchyError::StageCapacityExceeded {
                required: commands.len(),
                limit: self.limits.max_stage_commands,
            });
        }
        let slot = &mut self.slot;
        let result = (|| {
            let globals = hierarchy_globals(
                records.len() as u32,
                padded_count,
                source_index_base,
                normalization_bounds,
                support_sigma,
                1,
                2,
            );
            queue.write_buffer(&slot.globals, 0, bytemuck::bytes_of(&globals));
            queue.write_buffer(&slot.input, 0, bytemuck::cast_slice(&canonical));
            write_stage_commands(queue, slot, self.capacities.stage_stride, &commands);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gaussian_lod_global_sort_encoder"),
            });
            dispatch(
                &mut encoder,
                "gaussian_lod_global_sort_initialize",
                &self.pipelines.initialize,
                &slot.bind_group,
                0,
                padded_count.div_ceil(GPU_LOD_HIERARCHY_SORT_WORKGROUP_SIZE),
            );
            for index in 0..commands.len() {
                dispatch(
                    &mut encoder,
                    "gaussian_lod_global_sort_bitonic_stage",
                    &self.pipelines.bitonic,
                    &slot.bind_group,
                    dynamic_offset(index, self.capacities.stage_stride)?,
                    padded_count.div_ceil(GPU_LOD_HIERARCHY_SORT_WORKGROUP_SIZE),
                );
            }
            dispatch(
                &mut encoder,
                "gaussian_lod_global_sort_gather",
                &self.pipelines.gather,
                &slot.bind_group,
                0,
                (records.len() as u32).div_ceil(GPU_LOD_HIERARCHY_SORT_WORKGROUP_SIZE),
            );
            let record_count = records.len() as u32;
            let status_len = checked_bytes(record_count, size_of::<u32>(), "status copy")?;
            let entry_len =
                checked_bytes(record_count, size_of::<GpuSortEntryRaw>(), "entry copy")?;
            let sorted_len = checked_bytes(record_count, size_of::<Gaussian3d>(), "sorted copy")?;
            encoder.copy_buffer_to_buffer(
                &slot.statuses,
                0,
                &slot.readback,
                self.capacities.readback.status_offset,
                status_len,
            );
            encoder.copy_buffer_to_buffer(
                &slot.entries,
                0,
                &slot.readback,
                self.capacities.readback.entry_offset,
                entry_len,
            );
            encoder.copy_buffer_to_buffer(
                &slot.sorted,
                0,
                &slot.readback,
                self.capacities.readback.sorted_offset,
                sorted_len,
            );
            let map_len = self
                .capacities
                .readback
                .sorted_offset
                .checked_add(sorted_len)
                .ok_or(GpuLodHierarchyError::CapacityOverflow("sort readback"))?;
            let submission = queue.submit([encoder.finish()]);
            map_slot(
                device,
                slot,
                submission,
                map_len,
                self.limits.poll_timeout,
                |mapped| {
                    decode_sorted_readback(
                        mapped,
                        self.capacities.readback,
                        source_index_base,
                        record_count,
                    )
                },
            )
        })();
        slot.readback.unmap();
        result
    }

    /// Reduce explicit groups of globally sorted leaf Gaussians. Group ranges
    /// must cover the submitted slice exactly and are independent of dispatch
    /// batch boundaries.
    pub fn reduce_moment_merge_leaf_groups(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        records: &[Gaussian3d],
        groups: &[GpuLodHierarchyReductionGroup],
        support_sigma: f32,
    ) -> Result<Vec<GpuLodHierarchyReductionOutput>, GpuLodHierarchyError> {
        self.reduce_moment_merge_external_groups(
            device,
            queue,
            records,
            &[],
            groups,
            support_sigma,
            true,
        )
    }

    /// Reduce explicit groups of child summaries from a preceding global
    /// hierarchy level. The accumulated device error is checked against every
    /// inherited child error before it is accepted.
    pub fn reduce_moment_merge_summary_groups(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &[GpuLodHierarchyReductionInput],
        groups: &[GpuLodHierarchyReductionGroup],
        support_sigma: f32,
    ) -> Result<Vec<GpuLodHierarchyReductionOutput>, GpuLodHierarchyError> {
        self.reduce_moment_merge_external_groups(
            device,
            queue,
            &[],
            inputs,
            groups,
            support_sigma,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_moment_merge_external_groups(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        leaves: &[Gaussian3d],
        summaries: &[GpuLodHierarchyReductionInput],
        groups: &[GpuLodHierarchyReductionGroup],
        support_sigma: f32,
        leaf: bool,
    ) -> Result<Vec<GpuLodHierarchyReductionOutput>, GpuLodHierarchyError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (
                device,
                queue,
                leaves,
                summaries,
                groups,
                support_sigma,
                leaf,
            );
            Err(GpuLodHierarchyError::BlockingReadbackUnsupported)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.reduce_moment_merge_external_groups_native(
                device,
                queue,
                leaves,
                summaries,
                groups,
                support_sigma,
                leaf,
            )
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::too_many_arguments)]
    fn reduce_moment_merge_external_groups_native(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        leaves: &[Gaussian3d],
        summaries: &[GpuLodHierarchyReductionInput],
        groups: &[GpuLodHierarchyReductionGroup],
        support_sigma: f32,
        leaf: bool,
    ) -> Result<Vec<GpuLodHierarchyReductionOutput>, GpuLodHierarchyError> {
        validate_reduction_support_sigma(support_sigma)?;
        let input_count = if leaf {
            if !summaries.is_empty() {
                return Err(GpuLodHierarchyError::InvalidReductionGroups(
                    "leaf reduction received internal summaries",
                ));
            }
            leaves.len()
        } else {
            if !leaves.is_empty() {
                return Err(GpuLodHierarchyError::InvalidReductionGroups(
                    "internal reduction received leaf records",
                ));
            }
            summaries.len()
        };
        validate_reduction_groups(input_count, groups)?;
        if input_count > self.limits.max_records as usize {
            return Err(GpuLodHierarchyError::BatchTooLarge {
                actual: input_count,
                limit: self.limits.max_records,
            });
        }
        if groups.len() > self.limits.max_records as usize {
            return Err(GpuLodHierarchyError::GroupCapacityExceeded {
                required: groups.len(),
                limit: self.limits.max_records,
            });
        }
        let required_nodes =
            if leaf {
                groups.len()
            } else {
                input_count.checked_add(groups.len()).ok_or(
                    GpuLodHierarchyError::CapacityOverflow("external reduction nodes"),
                )?
            };
        if required_nodes > self.limits.max_nodes as usize {
            return Err(GpuLodHierarchyError::NodeCapacityExceeded {
                required: u32::try_from(required_nodes).unwrap_or(u32::MAX),
                limit: self.limits.max_nodes,
            });
        }
        let canonical_leaves = if leaf {
            canonical_records(leaves)?
        } else {
            Vec::new()
        };
        let mut raw_inputs = Vec::new();
        if !leaf {
            raw_inputs
                .try_reserve_exact(summaries.len())
                .map_err(|_| GpuLodHierarchyError::CapacityOverflow("external reduction inputs"))?;
            for (index, input) in summaries.iter().copied().enumerate() {
                validate_gaussian(&input.representative).map_err(|field| {
                    GpuLodHierarchyError::InvalidRepresentative { index, field }
                })?;
                input.bounds.validate().map_err(|error| {
                    GpuLodHierarchyError::InvalidGpuBounds {
                        index,
                        message: error.to_string(),
                    }
                })?;
                validate_error(index, input.inherited_error)?;
                let mut raw = GpuNodeRaw::zeroed();
                raw.bounds_min = [
                    input.bounds.min[0],
                    input.bounds.min[1],
                    input.bounds.min[2],
                    0.0,
                ];
                raw.bounds_max = [
                    input.bounds.max[0],
                    input.bounds.max[1],
                    input.bounds.max[2],
                    0.0,
                ];
                raw.summary_error = error_raw(input.inherited_error);
                raw.representative = canonicalize_gaussian_zeros(input.representative);
                raw_inputs.push(raw);
            }
        }
        let raw_groups = groups
            .iter()
            .map(|group| GpuSortEntryRaw {
                key_and_source: [0; 4],
                input_and_valid: [group.start, group.count, 0, 0],
            })
            .collect::<Vec<_>>();
        let output_start = if leaf { 0 } else { input_count as u32 };
        let stage = GpuStageParams {
            first: [output_start, groups.len() as u32, input_count as u32, 0],
            second: [u32::from(leaf), 0, 0, 0],
        };
        let slot = &mut self.slot;
        let result = (|| {
            let globals = hierarchy_globals(
                input_count as u32,
                input_count as u32,
                0,
                LodBounds::new([0.0; 3], [0.0; 3]).expect("zero bounds are valid"),
                support_sigma,
                1,
                2,
            );
            queue.write_buffer(&slot.globals, 0, bytemuck::bytes_of(&globals));
            queue.write_buffer(&slot.stages, 0, bytemuck::bytes_of(&stage));
            queue.write_buffer(&slot.entries, 0, bytemuck::cast_slice(&raw_groups));
            if leaf {
                queue.write_buffer(&slot.sorted, 0, bytemuck::cast_slice(&canonical_leaves));
            } else {
                queue.write_buffer(&slot.nodes, 0, bytemuck::cast_slice(&raw_inputs));
            }
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gaussian_lod_global_reduce_encoder"),
            });
            dispatch(
                &mut encoder,
                "gaussian_lod_global_reduce_groups",
                &self.pipelines.reduce_external,
                &slot.bind_group,
                0,
                (groups.len() as u32).div_ceil(GPU_LOD_HIERARCHY_REDUCE_WORKGROUP_SIZE),
            );
            let node_len = checked_bytes(
                groups.len() as u32,
                size_of::<GpuNodeRaw>(),
                "external reduction readback",
            )?;
            encoder.copy_buffer_to_buffer(
                &slot.nodes,
                u64::from(output_start) * size_of::<GpuNodeRaw>() as u64,
                &slot.readback,
                self.capacities.readback.node_offset,
                node_len,
            );
            let map_len = self
                .capacities
                .readback
                .node_offset
                .checked_add(node_len)
                .ok_or(GpuLodHierarchyError::CapacityOverflow(
                    "external reduction readback",
                ))?;
            let submission = queue.submit([encoder.finish()]);
            map_slot(
                device,
                slot,
                submission,
                map_len,
                self.limits.poll_timeout,
                |mapped| {
                    decode_external_reduction(
                        mapped,
                        self.capacities.readback.node_offset,
                        groups,
                        summaries,
                        leaf,
                    )
                },
            )
        })();
        slot.readback.unmap();
        result
    }
}

fn validate_reduction_support_sigma(support_sigma: f32) -> Result<(), GpuLodHierarchyError> {
    if !support_sigma.is_finite() || support_sigma <= 0.0 {
        Err(GpuLodHierarchyError::InvalidSupportSigma(support_sigma))
    } else {
        Ok(())
    }
}

fn validate_normalization_bounds(bounds: LodBounds) -> Result<(), GpuLodHierarchyError> {
    bounds
        .validate()
        .map_err(|error| GpuLodHierarchyError::InvalidBounds(error.to_string()))?;
    for axis in 0..3 {
        if !(bounds.max[axis] - bounds.min[axis]).is_finite() {
            return Err(GpuLodHierarchyError::InvalidBounds(format!(
                "normalization extent on axis {axis} is not finite"
            )));
        }
    }
    Ok(())
}

fn canonical_records(records: &[Gaussian3d]) -> Result<Vec<Gaussian3d>, GpuLodHierarchyError> {
    records
        .iter()
        .copied()
        .enumerate()
        .map(|(index, record)| {
            validate_gaussian(&record)
                .map_err(|field| GpuLodHierarchyError::InvalidGaussian { index, field })?;
            Ok(canonicalize_gaussian_zeros(record))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn hierarchy_globals(
    record_count: u32,
    padded_count: u32,
    source_index_base: u64,
    normalization_bounds: LodBounds,
    support_sigma: f32,
    leaf_capacity: u32,
    branching_factor: u8,
) -> GpuGlobalParams {
    GpuGlobalParams {
        counts: [
            record_count,
            padded_count,
            source_index_base as u32,
            (source_index_base >> 32) as u32,
        ],
        normalization_min: [
            normalization_bounds.min[0],
            normalization_bounds.min[1],
            normalization_bounds.min[2],
            0.0,
        ],
        normalization_max: [
            normalization_bounds.max[0],
            normalization_bounds.max[1],
            normalization_bounds.max[2],
            0.0,
        ],
        build: [
            support_sigma,
            leaf_capacity as f32,
            f32::from(branching_factor),
            0.0,
        ],
    }
}

fn write_stage_commands(
    queue: &wgpu::Queue,
    slot: &Slot,
    stride: u64,
    commands: &[GpuStageParams],
) {
    if commands.is_empty() {
        return;
    }
    let mut bytes = vec![0_u8; commands.len() * stride as usize];
    for (index, command) in commands.iter().enumerate() {
        let offset = index * stride as usize;
        bytes[offset..offset + size_of::<GpuStageParams>()]
            .copy_from_slice(bytemuck::bytes_of(command));
    }
    queue.write_buffer(&slot.stages, 0, &bytes);
}

#[cfg(not(target_arch = "wasm32"))]
fn map_slot<T>(
    device: &wgpu::Device,
    slot: &Slot,
    submission: wgpu::SubmissionIndex,
    map_len: u64,
    timeout: Duration,
    decode: impl FnOnce(&[u8]) -> Result<T, GpuLodHierarchyError>,
) -> Result<T, GpuLodHierarchyError> {
    let slice = slot.readback.slice(..map_len);
    let (sender, receiver) = mpsc::sync_channel(1);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(timeout),
        })
        .map_err(|error| GpuLodHierarchyError::DevicePoll(error.to_string()))?;
    receiver
        .recv_timeout(Duration::from_millis(100))
        .map_err(|_| GpuLodHierarchyError::MapCallbackMissing)?
        .map_err(|error| GpuLodHierarchyError::Map(error.to_string()))?;
    let mapped = slice.get_mapped_range();
    decode(&mapped)
}

fn decode_sorted_readback(
    mapped: &[u8],
    layout: ReadbackLayout,
    source_index_base: u64,
    record_count: u32,
) -> Result<Vec<GpuLodHierarchySortedRecord>, GpuLodHierarchyError> {
    let statuses = pod_vec::<u32>(mapped, layout.status_offset, record_count)?;
    if let Some((local_index, status)) = statuses
        .iter()
        .copied()
        .enumerate()
        .find(|(_, status)| *status != 0)
    {
        return Err(GpuLodHierarchyError::InvalidGpuRecord {
            source_index: source_index_base + local_index as u64,
            status,
        });
    }
    let entries = pod_vec::<GpuSortEntryRaw>(mapped, layout.entry_offset, record_count)?;
    let gaussians = pod_vec::<Gaussian3d>(mapped, layout.sorted_offset, record_count)?;
    let mut seen = vec![false; record_count as usize];
    let mut result = Vec::with_capacity(record_count as usize);
    for (index, (entry, gaussian)) in entries.into_iter().zip(gaussians).enumerate() {
        let local_index = entry.input_and_valid[0] as usize;
        if entry.input_and_valid[1] != 1
            || local_index >= record_count as usize
            || seen[local_index]
        {
            return Err(GpuLodHierarchyError::MalformedReadback(
                "sorted entries are invalid, duplicated, or outside the source batch",
            ));
        }
        seen[local_index] = true;
        let morton =
            u64::from(entry.key_and_source[0]) | (u64::from(entry.key_and_source[1]) << 32);
        let source_index =
            u64::from(entry.key_and_source[2]) | (u64::from(entry.key_and_source[3]) << 32);
        if source_index < source_index_base
            || source_index >= source_index_base + u64::from(record_count)
        {
            return Err(GpuLodHierarchyError::MalformedReadback(
                "sorted source index is outside the submitted batch",
            ));
        }
        validate_gaussian(&gaussian)
            .map_err(|field| GpuLodHierarchyError::InvalidGaussian { index, field })?;
        result.push(GpuLodHierarchySortedRecord {
            morton,
            source_index,
            gaussian,
        });
    }
    if !result.windows(2).all(|pair| {
        pair[0]
            .morton
            .cmp(&pair[1].morton)
            .then_with(|| compare_gaussians(&pair[0].gaussian, &pair[1].gaussian))
            .then_with(|| pair[0].source_index.cmp(&pair[1].source_index))
            .is_le()
    }) {
        return Err(GpuLodHierarchyError::MalformedReadback(
            "GPU Morton output is not in canonical order",
        ));
    }
    Ok(result)
}

fn validate_reduction_groups(
    input_count: usize,
    groups: &[GpuLodHierarchyReductionGroup],
) -> Result<(), GpuLodHierarchyError> {
    if input_count == 0 || groups.is_empty() {
        return Err(GpuLodHierarchyError::EmptySource);
    }
    let mut expected_start = 0_u32;
    for group in groups {
        if group.count == 0 || group.start != expected_start {
            return Err(GpuLodHierarchyError::InvalidReductionGroups(
                "groups must be non-empty, contiguous, and ordered",
            ));
        }
        expected_start = group.start.checked_add(group.count).ok_or(
            GpuLodHierarchyError::InvalidReductionGroups("group range overflowed u32"),
        )?;
    }
    if expected_start as usize != input_count {
        return Err(GpuLodHierarchyError::InvalidReductionGroups(
            "groups must cover the submitted input exactly",
        ));
    }
    Ok(())
}

fn error_raw(error: LodError) -> [f32; 4] {
    [
        error.geometric,
        error.appearance,
        error.opacity,
        error.combined,
    ]
}

fn error_from_raw(values: [f32; 4]) -> LodError {
    LodError {
        geometric: values[0],
        appearance: values[1],
        opacity: values[2],
        combined: values[3],
    }
}

fn error_contains(outer: LodError, inner: LodError) -> bool {
    outer.geometric >= inner.geometric
        && outer.appearance >= inner.appearance
        && outer.opacity >= inner.opacity
        && outer.combined >= inner.combined
}

fn decode_external_reduction(
    mapped: &[u8],
    node_offset: u64,
    groups: &[GpuLodHierarchyReductionGroup],
    inputs: &[GpuLodHierarchyReductionInput],
    leaf: bool,
) -> Result<Vec<GpuLodHierarchyReductionOutput>, GpuLodHierarchyError> {
    let raws = pod_vec::<GpuNodeRaw>(mapped, node_offset, groups.len() as u32)?;
    raws.into_iter()
        .enumerate()
        .map(|(index, raw)| {
            if raw.page[2] != 0 {
                return Err(GpuLodHierarchyError::InvalidGpuNode {
                    index,
                    status: raw.page[2],
                });
            }
            validate_gaussian(&raw.representative)
                .map_err(|field| GpuLodHierarchyError::InvalidRepresentative { index, field })?;
            let representative_support = LodBounds::new(
                [
                    raw.representative_support_min[0],
                    raw.representative_support_min[1],
                    raw.representative_support_min[2],
                ],
                [
                    raw.representative_support_max[0],
                    raw.representative_support_max[1],
                    raw.representative_support_max[2],
                ],
            )
            .map_err(|error| GpuLodHierarchyError::InvalidGpuBounds {
                index,
                message: error.to_string(),
            })?;
            let local_error = error_from_raw(raw.error);
            let accumulated_error = error_from_raw(raw.summary_error);
            validate_error(index, local_error)?;
            validate_error(index, accumulated_error)?;
            if !error_contains(accumulated_error, local_error) {
                return Err(GpuLodHierarchyError::NonConservativeGpuError { index });
            }
            if !leaf {
                let group = groups[index];
                for child in &inputs[group.start as usize..(group.start + group.count) as usize] {
                    if !error_contains(accumulated_error, child.inherited_error) {
                        return Err(GpuLodHierarchyError::NonConservativeGpuError { index });
                    }
                }
            }
            Ok(GpuLodHierarchyReductionOutput {
                representative: canonicalize_gaussian_zeros(raw.representative),
                representative_support,
                local_error,
                accumulated_error,
            })
        })
        .collect()
}

fn uniform_layout_entry(binding: u32, bytes: u64, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: NonZeroU64::new(bytes),
        },
        count: None,
    }
}

fn storage_layout_entry(binding: u32, bytes: u64, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(bytes),
        },
        count: None,
    }
}

fn create_slot(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    capacities: ValidatedCapacities,
) -> Slot {
    let buffer = |label: &'static str, size: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let globals = buffer(
        "gaussian_lod_hierarchy_globals",
        size_of::<GpuGlobalParams>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let input = buffer(
        "gaussian_lod_hierarchy_input",
        capacities.input_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let entries = buffer(
        "gaussian_lod_hierarchy_entries",
        capacities.entry_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let sorted = buffer(
        "gaussian_lod_hierarchy_sorted",
        capacities.input_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let statuses = buffer(
        "gaussian_lod_hierarchy_statuses",
        capacities.status_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let nodes = buffer(
        "gaussian_lod_hierarchy_nodes",
        capacities.node_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let stages = buffer(
        "gaussian_lod_hierarchy_stages",
        capacities.stage_bytes,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let readback = buffer(
        "gaussian_lod_hierarchy_readback",
        capacities.readback.capacity_bytes,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gaussian_lod_hierarchy_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &stages,
                    offset: 0,
                    size: NonZeroU64::new(size_of::<GpuStageParams>() as u64),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: entries.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: sorted.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: statuses.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: nodes.as_entire_binding(),
            },
        ],
    });
    Slot {
        globals,
        input,
        entries,
        sorted,
        statuses,
        nodes,
        stages,
        readback,
        bind_group,
    }
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    label: &'static str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dynamic_offset: u32,
    workgroups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[dynamic_offset]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

fn dynamic_offset(index: usize, stride: u64) -> Result<u32, GpuLodHierarchyError> {
    let offset = (index as u64)
        .checked_mul(stride)
        .ok_or(GpuLodHierarchyError::CapacityOverflow("dynamic offset"))?;
    u32::try_from(offset).map_err(|_| GpuLodHierarchyError::DynamicOffsetOverflow(offset))
}

fn pod_vec<T: Pod + Copy>(
    mapped: &[u8],
    offset: u64,
    count: u32,
) -> Result<Vec<T>, GpuLodHierarchyError> {
    let offset = usize::try_from(offset)
        .map_err(|_| GpuLodHierarchyError::MalformedReadback("offset exceeds usize"))?;
    let bytes = (count as usize)
        .checked_mul(size_of::<T>())
        .ok_or(GpuLodHierarchyError::CapacityOverflow("readback decode"))?;
    let end = offset
        .checked_add(bytes)
        .ok_or(GpuLodHierarchyError::CapacityOverflow("readback range"))?;
    let source = mapped
        .get(offset..end)
        .ok_or(GpuLodHierarchyError::MalformedReadback(
            "mapped range is truncated",
        ))?;
    Ok(source
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned::<T>)
        .collect())
}

fn validate_error(index: usize, error: LodError) -> Result<(), GpuLodHierarchyError> {
    let values = [
        error.geometric,
        error.appearance,
        error.opacity,
        error.combined,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || error.combined + f32::EPSILON < error.geometric
        || error.combined + f32::EPSILON < error.appearance
        || error.combined + f32::EPSILON < error.opacity
    {
        Err(GpuLodHierarchyError::InvalidGpuError { index, values })
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum GpuLodHierarchyError {
    ZeroLimit(&'static str),
    CapacityOverflow(&'static str),
    ConfiguredByteLimit {
        field: &'static str,
        required: u64,
        configured: u64,
    },
    DeviceLimit {
        field: &'static str,
        required: u64,
        supported: u64,
    },
    DynamicOffsetOverflow(u64),
    BatchTooLarge {
        actual: usize,
        limit: u32,
    },
    GroupCapacityExceeded {
        required: usize,
        limit: u32,
    },
    NodeCapacityExceeded {
        required: u32,
        limit: u32,
    },
    StageCapacityExceeded {
        required: usize,
        limit: u32,
    },
    SourceIndexOverflow,
    EmptySource,
    InvalidSupportSigma(f32),
    InvalidBounds(String),
    InvalidReductionGroups(&'static str),
    InvalidGaussian {
        index: usize,
        field: GaussianField,
    },
    InvalidGpuRecord {
        source_index: u64,
        status: u32,
    },
    InvalidGpuNode {
        index: usize,
        status: u32,
    },
    InvalidRepresentative {
        index: usize,
        field: GaussianField,
    },
    InvalidGpuBounds {
        index: usize,
        message: String,
    },
    InvalidGpuError {
        index: usize,
        values: [f32; 4],
    },
    NonConservativeGpuError {
        index: usize,
    },
    DevicePoll(String),
    Map(String),
    MapCallbackMissing,
    MalformedReadback(&'static str),
    BlockingReadbackUnsupported,
}

impl fmt::Display for GpuLodHierarchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroLimit(field) => write!(f, "GPU hierarchy {field} must be non-zero"),
            Self::CapacityOverflow(field) => write!(f, "GPU hierarchy {field} capacity overflowed"),
            Self::ConfiguredByteLimit {
                field,
                required,
                configured,
            } => write!(
                f,
                "GPU hierarchy requires {required} bytes but {field} is {configured}"
            ),
            Self::DeviceLimit {
                field,
                required,
                supported,
            } => write!(
                f,
                "GPU hierarchy {field} requires {required}, device supports {supported}"
            ),
            Self::DynamicOffsetOverflow(bytes) => write!(
                f,
                "GPU hierarchy dynamic uniform offset {bytes} exceeds the u32 API range"
            ),
            Self::BatchTooLarge { actual, limit } => write!(
                f,
                "GPU hierarchy batch has {actual} records, configured limit is {limit}"
            ),
            Self::GroupCapacityExceeded { required, limit } => write!(
                f,
                "GPU hierarchy reduction has {required} groups, configured limit is {limit}"
            ),
            Self::NodeCapacityExceeded { required, limit } => write!(
                f,
                "GPU hierarchy needs {required} nodes, configured limit is {limit}"
            ),
            Self::StageCapacityExceeded { required, limit } => write!(
                f,
                "GPU hierarchy needs {required} stage commands, configured limit is {limit}"
            ),
            Self::SourceIndexOverflow => write!(f, "GPU hierarchy source index overflow"),
            Self::EmptySource => write!(f, "GPU hierarchy submission cannot be empty"),
            Self::InvalidSupportSigma(value) => write!(
                f,
                "GPU hierarchy support sigma must be finite and positive, got {value}"
            ),
            Self::InvalidBounds(error) => write!(f, "invalid GPU hierarchy bounds: {error}"),
            Self::InvalidReductionGroups(message) => {
                write!(f, "invalid GPU hierarchy reduction groups: {message}")
            }
            Self::InvalidGaussian { index, field } => write!(
                f,
                "GPU hierarchy source Gaussian {index} has invalid {field:?}"
            ),
            Self::InvalidGpuRecord {
                source_index,
                status,
            } => write!(
                f,
                "GPU hierarchy record {source_index} failed device validation with status {status:#x}"
            ),
            Self::InvalidGpuNode { index, status } => write!(
                f,
                "GPU hierarchy node {index} failed device reduction with status {status:#x}"
            ),
            Self::InvalidRepresentative { index, field } => write!(
                f,
                "GPU hierarchy representative {index} has invalid {field:?}"
            ),
            Self::InvalidGpuBounds { index, message } => write!(
                f,
                "GPU hierarchy node {index} has invalid bounds: {message}"
            ),
            Self::InvalidGpuError { index, values } => write!(
                f,
                "GPU hierarchy node {index} has invalid error components {values:?}"
            ),
            Self::NonConservativeGpuError { index } => write!(
                f,
                "GPU hierarchy node {index} understates its local or inherited child error"
            ),
            Self::DevicePoll(error) => write!(f, "GPU hierarchy device poll failed: {error}"),
            Self::Map(error) => write!(f, "GPU hierarchy readback map failed: {error}"),
            Self::MapCallbackMissing => {
                write!(f, "GPU hierarchy readback callback did not complete")
            }
            Self::MalformedReadback(message) => {
                write!(f, "GPU hierarchy readback is malformed: {message}")
            }
            Self::BlockingReadbackUnsupported => {
                write!(f, "blocking GPU hierarchy readback is unsupported on wasm")
            }
        }
    }
}

impl Error for GpuLodHierarchyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_layout_exactly_matches_wgsl() {
        assert_eq!(size_of::<GpuGlobalParams>(), 64);
        assert_eq!(size_of::<GpuStageParams>(), 32);
        assert_eq!(size_of::<GpuSortEntryRaw>(), 32);
        assert_eq!(size_of::<GpuNodeRaw>(), 9 * 16 + size_of::<Gaussian3d>());
        assert_eq!(std::mem::offset_of!(GpuNodeRaw, representative), 9 * 16);
        let shader = shader_source();
        assert!(shader.contains("fn reduce_external_groups"));
        assert!(!shader.contains("fn reduce_level"));
        assert!(shader.contains("node.summary_error = accumulated_error"));
        assert!(shader.contains("rotation[row][axis] * scale2[axis] * rotation[column][axis]"));
        assert!(!shader.contains("rotation[axis][row] * scale2[axis] * rotation[axis][column]"));
        assert!(shader.contains("let r0 = eigen.row0;"));
        assert!(shader.contains("let r1 = eigen.row1;"));
        assert!(shader.contains("let r2 = eigen.row2;"));
    }

    #[test]
    fn explicit_global_groups_must_partition_each_bounded_batch() {
        let groups = [
            GpuLodHierarchyReductionGroup { start: 0, count: 3 },
            GpuLodHierarchyReductionGroup { start: 3, count: 2 },
        ];
        validate_reduction_groups(5, &groups).unwrap();
        assert!(matches!(
            validate_reduction_groups(
                5,
                &[
                    GpuLodHierarchyReductionGroup { start: 0, count: 3 },
                    GpuLodHierarchyReductionGroup { start: 4, count: 1 },
                ]
            ),
            Err(GpuLodHierarchyError::InvalidReductionGroups(_))
        ));
        assert!(matches!(
            validate_reduction_groups(5, &[GpuLodHierarchyReductionGroup { start: 0, count: 4 }]),
            Err(GpuLodHierarchyError::InvalidReductionGroups(_))
        ));
    }

    #[test]
    fn bitonic_commands_cover_every_power_of_two_stage() {
        assert!(sort_stages(1).is_empty());
        assert_eq!(sort_stages(2).len(), 1);
        assert_eq!(sort_stages(8).len(), 6);
        assert_eq!(sort_stages(65_536).len(), 136);
    }
}
