//! Bounded GPU Morton sorting for offline LoD construction.
//!
//! Canonical Morton keys are authored on the host and uploaded as integers.
//! Exact CPU payload ordering is repaired inside equal-Morton spans after
//! readback so adapter floating point modes cannot affect package bytes.
//! Hierarchy construction and representative fitting use the CPU builder.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use std::{borrow::Cow, error::Error, fmt, mem::size_of, num::NonZeroU64, time::Duration};

use bytemuck::{Pod, Zeroable};

use crate::{
    gaussian::formats::{
        planar_3d::Gaussian3d,
        planar_3d_chunked::{GaussianField, LodBounds, validate_gaussian},
        planar_3d_lod::{
            canonical_lod_morton_code, canonicalize_gaussian_zeros, compare_gaussians,
        },
    },
    material::spherical_harmonics::SH_VEC4_PLANES,
};

pub const GPU_LOD_SORT_WORKGROUP_SIZE: u32 = 256;

const SHADER_SOURCE: &str = include_str!("sort.wgsl");
const READBACK_ALIGNMENT: u64 = 256;

fn shader_source() -> String {
    SHADER_SOURCE.replace("__SH_VEC4_PLANES__", &SH_VEC4_PLANES.to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuLodSortLimits {
    pub max_records: u32,
    pub max_stage_commands: u32,
    pub max_input_bytes: u64,
    pub max_readback_bytes: u64,
    pub poll_timeout: Duration,
}

impl Default for GpuLodSortLimits {
    fn default() -> Self {
        Self {
            max_records: 65_536,
            max_stage_commands: 512,
            max_input_bytes: 64 * 1024 * 1024,
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
    stage_stride: u64,
    stage_bytes: u64,
    readback: ReadbackLayout,
}

#[derive(Clone, Copy, Debug)]
struct ReadbackLayout {
    status_offset: u64,
    entry_offset: u64,
    sorted_offset: u64,
    capacity_bytes: u64,
}

impl GpuLodSortLimits {
    fn validate(self, device: &wgpu::Device) -> Result<ValidatedCapacities, GpuLodSortError> {
        for (name, value) in [
            ("max_records", u64::from(self.max_records)),
            ("max_stage_commands", u64::from(self.max_stage_commands)),
            ("max_input_bytes", self.max_input_bytes),
            ("max_readback_bytes", self.max_readback_bytes),
        ] {
            if value == 0 {
                return Err(GpuLodSortError::ZeroLimit(name));
            }
        }
        if self.poll_timeout.is_zero() {
            return Err(GpuLodSortError::ZeroLimit("poll_timeout"));
        }

        let padded_records = self
            .max_records
            .checked_next_power_of_two()
            .ok_or(GpuLodSortError::CapacityOverflow("padded records"))?;
        let input_bytes = checked_bytes(self.max_records, size_of::<Gaussian3d>(), "input")?;
        let entry_bytes =
            checked_bytes(padded_records, size_of::<GpuSortEntryRaw>(), "sort entries")?;
        let status_bytes = checked_bytes(self.max_records, size_of::<u32>(), "statuses")?;
        if input_bytes > self.max_input_bytes {
            return Err(GpuLodSortError::ConfiguredByteLimit {
                field: "max_input_bytes",
                required: input_bytes,
                configured: self.max_input_bytes,
            });
        }
        let stage_stride = align_up(
            size_of::<GpuStageParams>() as u64,
            u64::from(device.limits().min_uniform_buffer_offset_alignment.max(1)),
        )?;
        let stage_bytes = stage_stride
            .checked_mul(u64::from(self.max_stage_commands))
            .ok_or(GpuLodSortError::CapacityOverflow("stage commands"))?;
        if stage_bytes > u64::from(u32::MAX) {
            return Err(GpuLodSortError::DynamicOffsetOverflow(stage_bytes));
        }

        let status_offset = 0;
        let entry_offset = align_up(status_bytes, READBACK_ALIGNMENT)?;
        let sorted_offset = align_up(
            entry_offset
                .checked_add(entry_bytes)
                .ok_or(GpuLodSortError::CapacityOverflow("readback entries"))?,
            READBACK_ALIGNMENT,
        )?;
        let capacity_bytes = sorted_offset
            .checked_add(input_bytes)
            .ok_or(GpuLodSortError::CapacityOverflow("readback"))?;
        if capacity_bytes > self.max_readback_bytes {
            return Err(GpuLodSortError::ConfiguredByteLimit {
                field: "max_readback_bytes",
                required: capacity_bytes,
                configured: self.max_readback_bytes,
            });
        }

        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < 4 {
            return Err(GpuLodSortError::DeviceLimit {
                field: "storage buffers per compute stage",
                required: 4,
                supported: u64::from(limits.max_storage_buffers_per_shader_stage),
            });
        }
        if limits.max_dynamic_uniform_buffers_per_pipeline_layout < 1 {
            return Err(GpuLodSortError::DeviceLimit {
                field: "dynamic uniform buffers",
                required: 1,
                supported: u64::from(limits.max_dynamic_uniform_buffers_per_pipeline_layout),
            });
        }
        if limits.max_compute_invocations_per_workgroup < GPU_LOD_SORT_WORKGROUP_SIZE {
            return Err(GpuLodSortError::DeviceLimit {
                field: "compute workgroup invocations",
                required: u64::from(GPU_LOD_SORT_WORKGROUP_SIZE),
                supported: u64::from(limits.max_compute_invocations_per_workgroup),
            });
        }
        for (name, bytes) in [
            ("input storage binding", input_bytes),
            ("sorted storage binding", input_bytes),
            ("sort-entry storage binding", entry_bytes),
            ("status storage binding", status_bytes),
        ] {
            validate_limit(name, bytes, limits.max_storage_buffer_binding_size)?;
            validate_limit(name, bytes, limits.max_buffer_size)?;
        }
        validate_limit("stage buffer", stage_bytes, limits.max_buffer_size)?;
        validate_limit("readback buffer", capacity_bytes, limits.max_buffer_size)?;
        let workgroups = padded_records.div_ceil(GPU_LOD_SORT_WORKGROUP_SIZE);
        validate_limit(
            "sort workgroups",
            u64::from(workgroups),
            u64::from(limits.max_compute_workgroups_per_dimension),
        )?;
        Ok(ValidatedCapacities {
            input_bytes,
            entry_bytes,
            status_bytes,
            stage_stride,
            stage_bytes,
            readback: ReadbackLayout {
                status_offset,
                entry_offset,
                sorted_offset,
                capacity_bytes,
            },
        })
    }
}

fn checked_bytes(count: u32, stride: usize, name: &'static str) -> Result<u64, GpuLodSortError> {
    u64::from(count)
        .checked_mul(stride as u64)
        .ok_or(GpuLodSortError::CapacityOverflow(name))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GpuLodSortError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or(GpuLodSortError::CapacityOverflow("alignment"))
}

fn validate_limit(
    field: &'static str,
    required: u64,
    supported: u64,
) -> Result<(), GpuLodSortError> {
    if required > supported {
        Err(GpuLodSortError::DeviceLimit {
            field,
            required,
            supported,
        })
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuLodSortedRecord {
    pub morton: u64,
    pub source_index: u64,
    pub gaussian: Gaussian3d,
}

fn sort_stages(padded_count: u32) -> Vec<GpuStageParams> {
    let mut result = Vec::new();
    let mut k = 2_u32;
    while k <= padded_count {
        let mut j = k / 2;
        while j > 0 {
            result.push(GpuStageParams {
                first: [k, j, 0, 0],
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
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuStageParams {
    first: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
struct GpuSortEntryRaw {
    key_and_source: [u32; 4],
    input_and_valid: [u32; 4],
}

struct Slot {
    globals: wgpu::Buffer,
    input: wgpu::Buffer,
    entries: wgpu::Buffer,
    sorted: wgpu::Buffer,
    statuses: wgpu::Buffer,
    stages: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct Pipelines {
    initialize: wgpu::ComputePipeline,
    bitonic: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
}

pub struct GpuLodBatchSorter {
    limits: GpuLodSortLimits,
    capacities: ValidatedCapacities,
    pipelines: Pipelines,
    slot: Slot,
}

impl GpuLodBatchSorter {
    pub fn new(device: &wgpu::Device, limits: GpuLodSortLimits) -> Result<Self, GpuLodSortError> {
        let capacities = limits.validate(device)?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_lod_sort_layout"),
            entries: &[
                uniform_layout_entry(0, size_of::<GpuGlobalParams>() as u64, false),
                uniform_layout_entry(1, size_of::<GpuStageParams>() as u64, true),
                storage_layout_entry(2, capacities.input_bytes, true),
                storage_layout_entry(3, capacities.entry_bytes, false),
                storage_layout_entry(4, capacities.input_bytes, false),
                storage_layout_entry(5, capacities.status_bytes, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gaussian_lod_sort_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gaussian_lod_sort_shader"),
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
            initialize: pipeline("gaussian_lod_sort_initialize", "initialize"),
            bitonic: pipeline("gaussian_lod_sort_bitonic", "bitonic_stage"),
            gather: pipeline("gaussian_lod_sort_gather", "gather_sorted"),
        };
        let slot = create_slot(device, &layout, capacities);
        Ok(Self {
            limits,
            capacities,
            pipelines,
            slot,
        })
    }

    pub const fn limits(&self) -> GpuLodSortLimits {
        self.limits
    }

    /// Canonically Morton-sort one bounded source batch for external run
    /// construction. The returned records can be spilled and globally merged
    /// before the CPU builder constructs the hierarchy.
    pub fn sort_morton_batch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        records: &[Gaussian3d],
        source_index_base: u64,
        normalization_bounds: LodBounds,
    ) -> Result<Vec<GpuLodSortedRecord>, GpuLodSortError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (
                device,
                queue,
                records,
                source_index_base,
                normalization_bounds,
            );
            Err(GpuLodSortError::BlockingReadbackUnsupported)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.sort_morton_batch_native(
                device,
                queue,
                records,
                source_index_base,
                normalization_bounds,
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
    ) -> Result<Vec<GpuLodSortedRecord>, GpuLodSortError> {
        if records.is_empty() {
            return Err(GpuLodSortError::EmptySource);
        }
        if records.len() > self.limits.max_records as usize {
            return Err(GpuLodSortError::BatchTooLarge {
                actual: records.len(),
                limit: self.limits.max_records,
            });
        }
        source_index_base
            .checked_add(records.len() as u64 - 1)
            .ok_or(GpuLodSortError::SourceIndexOverflow)?;
        validate_normalization_bounds(normalization_bounds)?;
        let canonical = canonical_records(records)?;
        let padded_count = (records.len() as u32).next_power_of_two();
        let host_entries = canonical_sort_entries(
            &canonical,
            source_index_base,
            normalization_bounds,
            padded_count,
        )?;
        let commands = sort_stages(padded_count);
        if commands.len() > self.limits.max_stage_commands as usize {
            return Err(GpuLodSortError::StageCapacityExceeded {
                required: commands.len(),
                limit: self.limits.max_stage_commands,
            });
        }
        let slot = &mut self.slot;
        let result = (|| {
            let globals = sort_globals(records.len() as u32, padded_count, normalization_bounds);
            queue.write_buffer(&slot.globals, 0, bytemuck::bytes_of(&globals));
            queue.write_buffer(&slot.input, 0, bytemuck::cast_slice(&canonical));
            // The existing sort-entry allocation is also the bounded upload
            // staging target. The shader never derives a key from floating
            // point coordinates, so adapter arithmetic cannot change package
            // ordering at Morton quantization boundaries.
            queue.write_buffer(&slot.entries, 0, bytemuck::cast_slice(&host_entries));
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
                padded_count.div_ceil(GPU_LOD_SORT_WORKGROUP_SIZE),
            );
            for index in 0..commands.len() {
                dispatch(
                    &mut encoder,
                    "gaussian_lod_global_sort_bitonic_stage",
                    &self.pipelines.bitonic,
                    &slot.bind_group,
                    dynamic_offset(index, self.capacities.stage_stride)?,
                    padded_count.div_ceil(GPU_LOD_SORT_WORKGROUP_SIZE),
                );
            }
            dispatch(
                &mut encoder,
                "gaussian_lod_global_sort_gather",
                &self.pipelines.gather,
                &slot.bind_group,
                0,
                (records.len() as u32).div_ceil(GPU_LOD_SORT_WORKGROUP_SIZE),
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
                .ok_or(GpuLodSortError::CapacityOverflow("sort readback"))?;
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
                        &canonical,
                        &host_entries,
                    )
                },
            )
        })();
        slot.readback.unmap();
        result
    }
}

fn validate_normalization_bounds(bounds: LodBounds) -> Result<(), GpuLodSortError> {
    bounds
        .validate()
        .map_err(|error| GpuLodSortError::InvalidBounds(error.to_string()))?;
    for axis in 0..3 {
        if !(bounds.max[axis] - bounds.min[axis]).is_finite() {
            return Err(GpuLodSortError::InvalidBounds(format!(
                "normalization extent on axis {axis} is not finite"
            )));
        }
    }
    Ok(())
}

fn canonical_records(records: &[Gaussian3d]) -> Result<Vec<Gaussian3d>, GpuLodSortError> {
    records
        .iter()
        .copied()
        .enumerate()
        .map(|(index, record)| {
            validate_gaussian(&record)
                .map_err(|field| GpuLodSortError::InvalidGaussian { index, field })?;
            Ok(canonicalize_gaussian_zeros(record))
        })
        .collect()
}

fn canonical_sort_entries(
    canonical: &[Gaussian3d],
    source_index_base: u64,
    normalization_bounds: LodBounds,
    padded_count: u32,
) -> Result<Vec<GpuSortEntryRaw>, GpuLodSortError> {
    if canonical.is_empty() || canonical.len() > padded_count as usize {
        return Err(GpuLodSortError::MalformedReadback(
            "host sort-entry count is invalid",
        ));
    }
    let staging_bytes = checked_bytes(
        padded_count,
        size_of::<GpuSortEntryRaw>(),
        "host sort-entry staging",
    )?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(padded_count as usize)
        .map_err(|_| GpuLodSortError::HostAllocationFailed {
            field: "sort-entry staging",
            bytes: staging_bytes,
        })?;
    for (local_index, gaussian) in canonical.iter().enumerate() {
        let source_index = source_index_base
            .checked_add(local_index as u64)
            .ok_or(GpuLodSortError::SourceIndexOverflow)?;
        let morton =
            canonical_lod_morton_code(gaussian.position_visibility.position, normalization_bounds);
        entries.push(GpuSortEntryRaw {
            key_and_source: [
                morton as u32,
                (morton >> 32) as u32,
                source_index as u32,
                (source_index >> 32) as u32,
            ],
            input_and_valid: [local_index as u32, 1, 0, 0],
        });
    }
    for local_index in canonical.len()..padded_count as usize {
        entries.push(GpuSortEntryRaw {
            key_and_source: [u32::MAX; 4],
            input_and_valid: [local_index as u32, 0, 0, 0],
        });
    }
    Ok(entries)
}

fn sort_globals(
    record_count: u32,
    padded_count: u32,
    normalization_bounds: LodBounds,
) -> GpuGlobalParams {
    GpuGlobalParams {
        counts: [record_count, padded_count, 0, 0],
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
    decode: impl FnOnce(&[u8]) -> Result<T, GpuLodSortError>,
) -> Result<T, GpuLodSortError> {
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
        .map_err(|error| GpuLodSortError::DevicePoll(error.to_string()))?;
    receiver
        .recv_timeout(Duration::from_millis(100))
        .map_err(|_| GpuLodSortError::MapCallbackMissing)?
        .map_err(|error| GpuLodSortError::Map(error.to_string()))?;
    let mapped = slice.get_mapped_range();
    decode(&mapped)
}

fn decode_sorted_readback(
    mapped: &[u8],
    layout: ReadbackLayout,
    source_index_base: u64,
    record_count: u32,
    canonical: &[Gaussian3d],
    host_entries: &[GpuSortEntryRaw],
) -> Result<Vec<GpuLodSortedRecord>, GpuLodSortError> {
    let statuses = pod_vec::<u32>(mapped, layout.status_offset, record_count)?;
    let entries = pod_vec::<GpuSortEntryRaw>(mapped, layout.entry_offset, record_count)?;
    let gaussians = pod_vec::<Gaussian3d>(mapped, layout.sorted_offset, record_count)?;
    validate_sorted_readback(
        &statuses,
        &entries,
        &gaussians,
        source_index_base,
        canonical,
        host_entries,
    )
}

fn validate_sorted_readback(
    statuses: &[u32],
    entries: &[GpuSortEntryRaw],
    gpu_gaussians: &[Gaussian3d],
    source_index_base: u64,
    canonical: &[Gaussian3d],
    host_entries: &[GpuSortEntryRaw],
) -> Result<Vec<GpuLodSortedRecord>, GpuLodSortError> {
    if statuses.len() != canonical.len()
        || entries.len() != canonical.len()
        || gpu_gaussians.len() != canonical.len()
        || host_entries.len() < canonical.len()
    {
        return Err(GpuLodSortError::MalformedReadback(
            "host and device sort record counts differ",
        ));
    }
    if let Some((local_index, status)) = statuses
        .iter()
        .copied()
        .enumerate()
        .find(|(_, status)| *status != 0)
    {
        let source_index = source_index_base
            .checked_add(local_index as u64)
            .ok_or(GpuLodSortError::SourceIndexOverflow)?;
        return Err(GpuLodSortError::InvalidGpuRecord {
            source_index,
            status,
        });
    }
    let mut seen = vec![false; canonical.len()];
    let mut result = Vec::with_capacity(canonical.len());
    for (index, (entry, gpu_gaussian)) in entries.iter().zip(gpu_gaussians).enumerate() {
        let local_index = entry.input_and_valid[0] as usize;
        if entry.input_and_valid[1] != 1 || local_index >= canonical.len() || seen[local_index] {
            return Err(GpuLodSortError::MalformedReadback(
                "sorted entries are invalid, duplicated, or outside the source batch",
            ));
        }
        seen[local_index] = true;
        if *entry != host_entries[local_index] {
            return Err(GpuLodSortError::MalformedReadback(
                "sorted entry differs from its host-authored key/source tuple",
            ));
        }
        let morton =
            u64::from(entry.key_and_source[0]) | (u64::from(entry.key_and_source[1]) << 32);
        let source_index =
            u64::from(entry.key_and_source[2]) | (u64::from(entry.key_and_source[3]) << 32);
        let expected_source_index = source_index_base
            .checked_add(local_index as u64)
            .ok_or(GpuLodSortError::SourceIndexOverflow)?;
        if source_index != expected_source_index {
            return Err(GpuLodSortError::MalformedReadback(
                "sorted source index does not match its source record",
            ));
        }
        validate_gaussian(gpu_gaussian)
            .map_err(|field| GpuLodSortError::InvalidGaussian { index, field })?;
        result.push(GpuLodSortedRecord {
            morton,
            source_index,
            // The gathered GPU payload is diagnostic only. Preserve the
            // canonical host bits so device subnormal handling can affect
            // neither equal-key fixup nor the returned package payload.
            gaussian: canonical[local_index],
        });
    }
    canonicalize_equal_morton_spans(&mut result)?;
    Ok(result)
}

/// Finish the package merge-key order on the host without re-sorting the
/// host-authored Morton sequence sorted by the GPU.
///
/// GPU floating-point modes may flush subnormal payload values to zero during
/// comparisons. Sorting only each collision span with Rust's canonical total
/// order makes the result exact while keeping host work proportional to actual
/// Morton collisions.
fn canonicalize_equal_morton_spans(
    records: &mut [GpuLodSortedRecord],
) -> Result<(), GpuLodSortError> {
    if !records
        .windows(2)
        .all(|pair| pair[0].morton <= pair[1].morton)
    {
        return Err(GpuLodSortError::MalformedReadback(
            "GPU Morton output is not monotonic",
        ));
    }

    let mut start = 0;
    while start < records.len() {
        let morton = records[start].morton;
        let mut end = start + 1;
        while end < records.len() && records[end].morton == morton {
            end += 1;
        }
        if end - start > 1 {
            records[start..end].sort_unstable_by(|left, right| {
                compare_gaussians(&left.gaussian, &right.gaussian)
                    .then_with(|| left.source_index.cmp(&right.source_index))
            });
        }
        start = end;
    }
    Ok(())
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
        "gaussian_lod_sort_globals",
        size_of::<GpuGlobalParams>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let input = buffer(
        "gaussian_lod_sort_input",
        capacities.input_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let entries = buffer(
        "gaussian_lod_sort_entries",
        capacities.entry_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let sorted = buffer(
        "gaussian_lod_sort_sorted",
        capacities.input_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let statuses = buffer(
        "gaussian_lod_sort_statuses",
        capacities.status_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let stages = buffer(
        "gaussian_lod_sort_stages",
        capacities.stage_bytes,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let readback = buffer(
        "gaussian_lod_sort_readback",
        capacities.readback.capacity_bytes,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gaussian_lod_sort_bind_group"),
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
        ],
    });
    Slot {
        globals,
        input,
        entries,
        sorted,
        statuses,
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

fn dynamic_offset(index: usize, stride: u64) -> Result<u32, GpuLodSortError> {
    let offset = (index as u64)
        .checked_mul(stride)
        .ok_or(GpuLodSortError::CapacityOverflow("dynamic offset"))?;
    u32::try_from(offset).map_err(|_| GpuLodSortError::DynamicOffsetOverflow(offset))
}

fn pod_vec<T: Pod + Copy>(
    mapped: &[u8],
    offset: u64,
    count: u32,
) -> Result<Vec<T>, GpuLodSortError> {
    let offset = usize::try_from(offset)
        .map_err(|_| GpuLodSortError::MalformedReadback("offset exceeds usize"))?;
    let bytes = (count as usize)
        .checked_mul(size_of::<T>())
        .ok_or(GpuLodSortError::CapacityOverflow("readback decode"))?;
    let end = offset
        .checked_add(bytes)
        .ok_or(GpuLodSortError::CapacityOverflow("readback range"))?;
    let source = mapped
        .get(offset..end)
        .ok_or(GpuLodSortError::MalformedReadback(
            "mapped range is truncated",
        ))?;
    Ok(source
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned::<T>)
        .collect())
}

#[derive(Debug)]
pub enum GpuLodSortError {
    ZeroLimit(&'static str),
    CapacityOverflow(&'static str),
    ConfiguredByteLimit {
        field: &'static str,
        required: u64,
        configured: u64,
    },
    HostAllocationFailed {
        field: &'static str,
        bytes: u64,
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
    StageCapacityExceeded {
        required: usize,
        limit: u32,
    },
    SourceIndexOverflow,
    EmptySource,
    InvalidBounds(String),
    InvalidGaussian {
        index: usize,
        field: GaussianField,
    },
    InvalidGpuRecord {
        source_index: u64,
        status: u32,
    },
    DevicePoll(String),
    Map(String),
    MapCallbackMissing,
    MalformedReadback(&'static str),
    BlockingReadbackUnsupported,
}

impl fmt::Display for GpuLodSortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroLimit(field) => write!(f, "GPU LoD sort {field} must be non-zero"),
            Self::CapacityOverflow(field) => write!(f, "GPU LoD sort {field} capacity overflowed"),
            Self::ConfiguredByteLimit {
                field,
                required,
                configured,
            } => write!(
                f,
                "GPU LoD sort requires {required} bytes but {field} is {configured}"
            ),
            Self::HostAllocationFailed { field, bytes } => write!(
                f,
                "GPU LoD sort could not reserve {bytes} bounded host bytes for {field}"
            ),
            Self::DeviceLimit {
                field,
                required,
                supported,
            } => write!(
                f,
                "GPU LoD sort {field} requires {required}, device supports {supported}"
            ),
            Self::DynamicOffsetOverflow(bytes) => write!(
                f,
                "GPU LoD sort dynamic uniform offset {bytes} exceeds the u32 API range"
            ),
            Self::BatchTooLarge { actual, limit } => write!(
                f,
                "GPU LoD sort batch has {actual} records, configured limit is {limit}"
            ),
            Self::StageCapacityExceeded { required, limit } => write!(
                f,
                "GPU LoD sort needs {required} stage commands, configured limit is {limit}"
            ),
            Self::SourceIndexOverflow => write!(f, "GPU LoD sort source index overflow"),
            Self::EmptySource => write!(f, "GPU LoD sort submission cannot be empty"),
            Self::InvalidBounds(error) => write!(f, "invalid GPU LoD sort bounds: {error}"),
            Self::InvalidGaussian { index, field } => write!(
                f,
                "GPU LoD sort source Gaussian {index} has invalid {field:?}"
            ),
            Self::InvalidGpuRecord {
                source_index,
                status,
            } => write!(
                f,
                "GPU LoD sort record {source_index} failed device validation with status {status:#x}"
            ),
            Self::DevicePoll(error) => write!(f, "GPU LoD sort device poll failed: {error}"),
            Self::Map(error) => write!(f, "GPU LoD sort readback map failed: {error}"),
            Self::MapCallbackMissing => {
                write!(f, "GPU LoD sort readback callback did not complete")
            }
            Self::MalformedReadback(message) => {
                write!(f, "GPU LoD sort readback is malformed: {message}")
            }
            Self::BlockingReadbackUnsupported => {
                write!(f, "blocking GPU LoD sort readback is unsupported on wasm")
            }
        }
    }
}

impl Error for GpuLodSortError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::spherical_harmonics::SH_COEFF_COUNT;

    fn valid_gaussian() -> Gaussian3d {
        Gaussian3d {
            position_visibility: [0.0, 0.0, 0.0, 1.0].into(),
            spherical_harmonic: Default::default(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.25, 0.5, 1.0, 0.75].into(),
        }
    }

    fn gaussian_at(mut gaussian: Gaussian3d, position: [f32; 3]) -> Gaussian3d {
        gaussian.position_visibility.position = position;
        gaussian
    }

    fn quantization_boundary_pair(
        base: Gaussian3d,
        bounds: LodBounds,
        axis: usize,
        bin: u32,
    ) -> [Gaussian3d; 2] {
        use crate::gaussian::formats::planar_3d_lod::LOD_MORTON_AXIS_MAX;

        let extent = bounds.max[axis] - bounds.min[axis];
        let approximate = bounds.min[axis] + extent * (bin as f32 / LOD_MORTON_AXIS_MAX as f32);
        let mut first = approximate;
        for _ in 0..256 {
            first = first.next_down();
        }
        let mut positions = Vec::with_capacity(513);
        positions.push(first);
        for _ in 0..512 {
            positions.push(positions.last().copied().unwrap().next_up());
        }
        let center = bounds.center();
        positions
            .windows(2)
            .find_map(|pair| {
                let mut left_position = center;
                left_position[axis] = pair[0];
                let mut right_position = center;
                right_position[axis] = pair[1];
                let left = gaussian_at(base, left_position);
                let right = gaussian_at(base, right_position);
                (canonical_lod_morton_code(left_position, bounds)
                    != canonical_lod_morton_code(right_position, bounds))
                .then_some([left, right])
            })
            .expect("fixture straddles a canonical Morton quantization boundary")
    }

    #[test]
    fn host_layout_exactly_matches_wgsl() {
        assert_eq!(
            size_of::<Gaussian3d>(),
            (SH_VEC4_PLANES + 3) * size_of::<[f32; 4]>()
        );
        assert_eq!(std::mem::offset_of!(Gaussian3d, position_visibility), 0);
        assert_eq!(std::mem::offset_of!(Gaussian3d, spherical_harmonic), 16);
        assert_eq!(
            std::mem::offset_of!(Gaussian3d, rotation),
            16 + SH_COEFF_COUNT * size_of::<f32>()
        );
        assert_eq!(
            std::mem::offset_of!(Gaussian3d, scale_opacity),
            32 + SH_COEFF_COUNT * size_of::<f32>()
        );
        assert_eq!(size_of::<GpuGlobalParams>(), 48);
        assert_eq!(size_of::<GpuStageParams>(), 16);
        assert_eq!(size_of::<GpuSortEntryRaw>(), 32);
        let shader = shader_source();
        let ordered_float = shader
            .split_once("fn ordered_float")
            .unwrap()
            .1
            .split_once("fn compare_float")
            .unwrap()
            .0;
        assert!(ordered_float.contains("let bits = bitcast<u32>(value);"));
        assert!(!ordered_float.contains("value == 0.0"));
        let payload_order = shader
            .split_once("fn compare_gaussians")
            .unwrap()
            .1
            .split_once("fn compare_entries")
            .unwrap()
            .0;
        let payload_fields = [
            "left.position_visibility",
            "left.spherical_harmonic",
            "left.rotation",
            "left.scale_opacity",
        ]
        .map(|field| payload_order.find(field).unwrap());
        assert!(payload_fields.is_sorted());
        let entry_order = shader
            .split_once("fn compare_entries")
            .unwrap()
            .1
            .split_once("@compute")
            .unwrap()
            .0;
        let entry_fields = [
            "left.key_and_source.y",
            "left.key_and_source.x",
            "compare_gaussians",
            "left.key_and_source.w",
            "left.key_and_source.z",
        ]
        .map(|field| entry_order.find(field).unwrap());
        assert!(entry_fields.is_sorted());
        assert!(entry_order.contains("left.input_and_valid.y != right.input_and_valid.y"));
        assert!(entry_order.contains("left.input_and_valid.y == 0u"));
        assert!(!shader.contains("fn morton_key"));
        assert!(!shader.contains("fn quantize_axis"));
        let initialize = shader
            .split_once("fn initialize")
            .unwrap()
            .1
            .split_once("fn bitonic_stage")
            .unwrap()
            .0;
        assert!(initialize.contains("statuses[index] = validate_gaussian(inputs[index]);"));
        assert!(!initialize.contains("entries[index]"));
    }

    #[test]
    fn host_sort_entries_author_canonical_keys_source_indices_and_padding() {
        let bounds = LodBounds::new(
            [-118.729_54, -130.432_02, -121.283_48],
            [137.847_32, 109.880_554, 136.600_8],
        )
        .unwrap();
        let base = valid_gaussian();
        let boundary = quantization_boundary_pair(base, bounds, 0, 1_048_575);
        let canonical =
            canonical_records(&[boundary[1], gaussian_at(base, bounds.center()), boundary[0]])
                .unwrap();
        let source_index_base = u64::from(u32::MAX) - 1;
        let entries = canonical_sort_entries(&canonical, source_index_base, bounds, 4).unwrap();

        assert_eq!(entries.len(), 4);
        assert_eq!(entries.len() * size_of::<GpuSortEntryRaw>(), 4 * 32);
        for (local_index, gaussian) in canonical.iter().enumerate() {
            let morton = canonical_lod_morton_code(gaussian.position_visibility.position, bounds);
            let source_index = source_index_base + local_index as u64;
            assert_eq!(
                entries[local_index],
                GpuSortEntryRaw {
                    key_and_source: [
                        morton as u32,
                        (morton >> 32) as u32,
                        source_index as u32,
                        (source_index >> 32) as u32,
                    ],
                    input_and_valid: [local_index as u32, 1, 0, 0],
                }
            );
        }
        assert_eq!(
            entries[3],
            GpuSortEntryRaw {
                key_and_source: [u32::MAX; 4],
                input_and_valid: [3, 0, 0, 0],
            }
        );
        assert_ne!(
            entries[0].key_and_source[..2],
            entries[2].key_and_source[..2]
        );

        let final_source = canonical_sort_entries(&canonical[..1], u64::MAX, bounds, 1).unwrap();
        assert_eq!(final_source[0].key_and_source[2..], [u32::MAX; 2]);
        assert!(matches!(
            canonical_sort_entries(&canonical[..2], u64::MAX, bounds, 2),
            Err(GpuLodSortError::SourceIndexOverflow)
        ));
    }

    #[test]
    fn readback_rejects_a_tampered_host_authored_morton_key() {
        let bounds = LodBounds::new([0.0; 3], [1.0; 3]).unwrap();
        let base = valid_gaussian();
        let canonical = canonical_records(&[
            gaussian_at(base, [0.75, 0.5, 0.25]),
            gaussian_at(base, [0.25, 0.5, 0.75]),
            gaussian_at(base, [0.5; 3]),
        ])
        .unwrap();
        let source_index_base = u64::from(u32::MAX) - 1;
        let host_entries =
            canonical_sort_entries(&canonical, source_index_base, bounds, 4).unwrap();
        let mut order = (0..canonical.len()).collect::<Vec<_>>();
        order.sort_unstable_by(|&left, &right| {
            let left_morton =
                canonical_lod_morton_code(canonical[left].position_visibility.position, bounds);
            let right_morton =
                canonical_lod_morton_code(canonical[right].position_visibility.position, bounds);
            left_morton
                .cmp(&right_morton)
                .then_with(|| compare_gaussians(&canonical[left], &canonical[right]))
                .then_with(|| left.cmp(&right))
        });
        let entries = order
            .iter()
            .map(|&index| host_entries[index])
            .collect::<Vec<_>>();
        let gpu_gaussians = order
            .iter()
            .map(|&index| canonical[index])
            .collect::<Vec<_>>();
        let statuses = vec![0; canonical.len()];

        let valid = validate_sorted_readback(
            &statuses,
            &entries,
            &gpu_gaussians,
            source_index_base,
            &canonical,
            &host_entries,
        )
        .unwrap();
        assert_eq!(valid.len(), canonical.len());
        for (record, &local_index) in valid.iter().zip(&order) {
            assert_eq!(
                bytemuck::bytes_of(&record.gaussian),
                bytemuck::bytes_of(&canonical[local_index])
            );
        }

        let mut tampered = entries;
        tampered[0].key_and_source[0] ^= 1;
        assert!(matches!(
            validate_sorted_readback(
                &statuses,
                &tampered,
                &gpu_gaussians,
                source_index_base,
                &canonical,
                &host_entries,
            ),
            Err(GpuLodSortError::MalformedReadback(
                "sorted entry differs from its host-authored key/source tuple"
            ))
        ));

        let mut tampered = order
            .iter()
            .map(|&index| host_entries[index])
            .collect::<Vec<_>>();
        tampered[1].key_and_source[3] ^= 1;
        assert!(matches!(
            validate_sorted_readback(
                &statuses,
                &tampered,
                &gpu_gaussians,
                source_index_base,
                &canonical,
                &host_entries,
            ),
            Err(GpuLodSortError::MalformedReadback(
                "sorted entry differs from its host-authored key/source tuple"
            ))
        ));

        let mut tampered = order
            .iter()
            .map(|&index| host_entries[index])
            .collect::<Vec<_>>();
        tampered[2].input_and_valid[2] = 1;
        assert!(matches!(
            validate_sorted_readback(
                &statuses,
                &tampered,
                &gpu_gaussians,
                source_index_base,
                &canonical,
                &host_entries,
            ),
            Err(GpuLodSortError::MalformedReadback(
                "sorted entry differs from its host-authored key/source tuple"
            ))
        ));

        let mut diagnostic_gaussians = gpu_gaussians;
        diagnostic_gaussians[0].position_visibility.visibility = 0.25;
        let result = validate_sorted_readback(
            &statuses,
            &order
                .iter()
                .map(|&index| host_entries[index])
                .collect::<Vec<_>>(),
            &diagnostic_gaussians,
            source_index_base,
            &canonical,
            &host_entries,
        )
        .unwrap();
        assert_eq!(
            bytemuck::bytes_of(&result[0].gaussian),
            bytemuck::bytes_of(&canonical[order[0]])
        );
    }

    #[test]
    fn canonical_upload_normalizes_signed_zero_and_rejects_nan() {
        let mut gaussian = valid_gaussian();
        gaussian.position_visibility.position[0] = -0.0;
        gaussian.position_visibility.visibility = -0.0;
        gaussian.spherical_harmonic.coefficients.fill(-0.0);
        gaussian.rotation.rotation[3] = -0.0;
        gaussian.scale_opacity.scale[0] = -0.0;
        gaussian.scale_opacity.opacity = -0.0;
        let canonical = canonical_records(&[gaussian]).unwrap().pop().unwrap();
        let fields = canonical
            .position_visibility
            .position
            .iter()
            .chain(std::iter::once(&canonical.position_visibility.visibility))
            .chain(canonical.spherical_harmonic.coefficients.iter())
            .chain(canonical.rotation.rotation.iter())
            .chain(canonical.scale_opacity.scale.iter())
            .chain(std::iter::once(&canonical.scale_opacity.opacity));
        assert!(
            fields
                .into_iter()
                .all(|value| value.to_bits() != 0x8000_0000)
        );

        gaussian.spherical_harmonic.coefficients[SH_COEFF_COUNT - 1] = f32::NAN;
        assert!(matches!(
            canonical_records(&[gaussian]),
            Err(GpuLodSortError::InvalidGaussian { index: 0, .. })
        ));
    }

    #[test]
    fn host_collision_fixup_repairs_subnormal_device_ordering() {
        let base = valid_gaussian();
        let mut subnormal_x = base;
        subnormal_x.position_visibility.position[0] = f32::from_bits(1);
        let mut subnormal_y = base;
        subnormal_y.position_visibility.position[1] = f32::from_bits(1);
        let source_base = u64::from(u32::MAX) - 1;
        // Mimic a device that flushes both subnormals to zero and therefore
        // leaves this collision span in source-index order.
        let mut actual = vec![
            GpuLodSortedRecord {
                morton: 7,
                source_index: source_base,
                gaussian: subnormal_x,
            },
            GpuLodSortedRecord {
                morton: 7,
                source_index: source_base + 1,
                gaussian: subnormal_y,
            },
            GpuLodSortedRecord {
                morton: 7,
                source_index: source_base + 2,
                gaussian: base,
            },
            GpuLodSortedRecord {
                morton: 8,
                source_index: source_base + 3,
                gaussian: base,
            },
        ];
        let mut expected = actual.clone();
        expected.sort_unstable_by(|left, right| {
            left.morton
                .cmp(&right.morton)
                .then_with(|| compare_gaussians(&left.gaussian, &right.gaussian))
                .then_with(|| left.source_index.cmp(&right.source_index))
        });

        canonicalize_equal_morton_spans(&mut actual).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual[3].morton, 8);
    }

    #[test]
    fn host_collision_fixup_rejects_non_monotonic_gpu_morton_output() {
        let mut records = vec![
            GpuLodSortedRecord {
                morton: 2,
                source_index: 0,
                gaussian: valid_gaussian(),
            },
            GpuLodSortedRecord {
                morton: 1,
                source_index: 1,
                gaussian: valid_gaussian(),
            },
        ];
        assert!(matches!(
            canonicalize_equal_morton_spans(&mut records),
            Err(GpuLodSortError::MalformedReadback(
                "GPU Morton output is not monotonic"
            ))
        ));
    }

    /// Opt in with:
    /// `RUN_GPU_LOD_PREPROCESS_TESTS=1 cargo test --features lod_build gpu_collision_sort_matches_cpu_canonical_order -- --ignored --nocapture`
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "requires an explicitly requested wgpu adapter"]
    fn gpu_collision_sort_matches_cpu_canonical_order() {
        use crate::gaussian::formats::planar_3d_lod::canonical_lod_morton_code;

        if std::env::var("RUN_GPU_LOD_PREPROCESS_TESTS").as_deref() != Ok("1") {
            eprintln!("set RUN_GPU_LOD_PREPROCESS_TESTS=1 to execute the adapter test");
            return;
        }

        let base = valid_gaussian();
        let mut records = vec![base];
        let mut signed_zero = base;
        signed_zero.position_visibility.position = [-0.0, -0.0, -0.0];
        signed_zero.spherical_harmonic.coefficients.fill(-0.0);
        signed_zero.rotation.rotation[1..].fill(-0.0);
        records.push(signed_zero);
        // An identical canonical payload after the low 32-bit source index
        // wraps exercises the final high/low source tiebreaker.
        records.push(base);
        for component in 0..3 {
            let mut gaussian = base;
            gaussian.position_visibility.position[component] = f32::from_bits(1);
            records.push(gaussian);
        }
        let mut gaussian = base;
        gaussian.position_visibility.visibility = 0.5;
        records.push(gaussian);
        for coefficient in 0..SH_COEFF_COUNT {
            let mut gaussian = base;
            gaussian.spherical_harmonic.coefficients[coefficient] = coefficient as f32 * 0.25 - 1.0;
            records.push(gaussian);
        }
        for component in 0..4 {
            let mut gaussian = base;
            gaussian.rotation.rotation[component] = if component == 0 { 0.5 } else { 0.25 };
            records.push(gaussian);
        }
        for component in 0..3 {
            let mut gaussian = base;
            gaussian.scale_opacity.scale[component] *= 0.5;
            records.push(gaussian);
        }
        let mut gaussian = base;
        gaussian.scale_opacity.opacity = 0.5;
        records.push(gaussian);
        let normalization_bounds = LodBounds::new(
            [-118.729_54, -130.432_02, -121.283_48],
            [137.847_32, 109.880_554, 136.600_8],
        )
        .unwrap();
        for (axis, bin) in [262_143, 1_048_575, 1_835_007].into_iter().enumerate() {
            records.extend(quantization_boundary_pair(
                base,
                normalization_bounds,
                axis,
                bin,
            ));
        }
        assert!(!records.len().is_power_of_two());
        assert!(records.len() <= 128);

        let source_index_base = u64::from(u32::MAX) - 1;
        let mut expected = canonical_records(&records)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, gaussian)| GpuLodSortedRecord {
                morton: canonical_lod_morton_code(
                    gaussian.position_visibility.position,
                    normalization_bounds,
                ),
                source_index: source_index_base + index as u64,
                gaussian,
            })
            .collect::<Vec<_>>();
        assert!(
            expected
                .windows(2)
                .any(|pair| pair[0].morton != pair[1].morton)
        );
        expected.sort_unstable_by(|left, right| {
            left.morton
                .cmp(&right.morton)
                .then_with(|| compare_gaussians(&left.gaussian, &right.gaussian))
                .then_with(|| left.source_index.cmp(&right.source_index))
        });

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .expect("collision-sort GPU test requires an adapter");
        eprintln!("collision-sort GPU adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gaussian_lod_collision_sort_test_device"),
            ..Default::default()
        }))
        .expect("collision-sort GPU test could not create a device");
        let mut sorter = GpuLodBatchSorter::new(
            &device,
            GpuLodSortLimits {
                max_records: 128,
                max_input_bytes: 1024 * 1024,
                max_readback_bytes: 4 * 1024 * 1024,
                ..Default::default()
            },
        )
        .unwrap();
        let actual = sorter
            .sort_morton_batch(
                &device,
                &queue,
                &records,
                source_index_base,
                normalization_bounds,
            )
            .unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.morton, expected.morton);
            assert_eq!(actual.source_index, expected.source_index);
            assert_eq!(
                bytemuck::bytes_of(&actual.gaussian),
                bytemuck::bytes_of(&expected.gaussian)
            );
        }

        // Reuse the same slot with a much smaller non-power-of-two batch. A
        // stale valid entry from the first dispatch must never enter the sort.
        let smaller = [
            records[records.len() - 1],
            records[0],
            records[records.len() - 2],
        ];
        let mut smaller_expected = canonical_records(&smaller)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, gaussian)| GpuLodSortedRecord {
                morton: canonical_lod_morton_code(
                    gaussian.position_visibility.position,
                    normalization_bounds,
                ),
                source_index: source_index_base + index as u64,
                gaussian,
            })
            .collect::<Vec<_>>();
        smaller_expected.sort_unstable_by(|left, right| {
            left.morton
                .cmp(&right.morton)
                .then_with(|| compare_gaussians(&left.gaussian, &right.gaussian))
                .then_with(|| left.source_index.cmp(&right.source_index))
        });
        let smaller_actual = sorter
            .sort_morton_batch(
                &device,
                &queue,
                &smaller,
                source_index_base,
                normalization_bounds,
            )
            .unwrap();
        assert_eq!(smaller_actual, smaller_expected);
    }

    #[test]
    fn bitonic_commands_cover_every_power_of_two_stage() {
        assert!(sort_stages(1).is_empty());
        assert_eq!(sort_stages(2).len(), 1);
        assert_eq!(sort_stages(8).len(), 6);
        assert_eq!(sort_stages(65_536).len(), 136);
    }
}
