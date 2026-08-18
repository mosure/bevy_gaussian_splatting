//! Bounded external-memory construction of portable Gaussian LoD packages.
//!
//! The source is replayed twice: one bounded pass establishes the canonical
//! normalization bounds, then bounded canonical preprocessing creates sorted
//! runs. The promoted GPU route sorts those batches and reduces the global
//! hierarchy on-device. Runs are merged with a configurable fan-in and the
//! final stream is consumed directly into leaf pages. Only one source batch,
//! merge heads and buffers, one hierarchy reduction group, and explicitly
//! capped manifest metadata are resident at a time. Publication is a
//! synchronized rename of a fully validated sibling staging directory.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    error::Error,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Seek, Write},
    mem::size_of,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use crate::{
    gaussian::{
        formats::{
            planar_3d::Gaussian3d,
            planar_3d_chunked::{
                LOD_PAGE_SCHEMA_VERSION, LodBounds, LodIndexRange, LodNodeId, LodPageDescriptor,
                LodPageEncoding, LodPageId, LodPageKind, LodPageRange, LodPageStorage,
                LodSourceRange, PlanarGaussian3dPage, StableHasher, stable_gaussian_hash,
                validate_gaussian,
            },
            planar_3d_lod::{
                GaussianLodBuildMetadata, GaussianLodBuildSettings, GaussianLodManifest,
                GaussianLodManifestHeader, GaussianLodNode, GaussianLodQualityMetadata,
                LOD_CURRENT_REQUIRED_FEATURES, LOD_MANIFEST_MAGIC, LOD_MANIFEST_VERSION,
                LodBuildError, LodError, LodMortonRange, LodQualityInterval, LodReducerKind,
                MOMENT_MERGE_VERSION, MomentMergeReducer, canonicalize_gaussian_zeros,
                compare_gaussians, gaussian_support_bounds, lod_config_fingerprint,
            },
        },
        lod_build_gpu::{
            LodPreprocessBatchOutput, LodPreprocessError, LodPreprocessRecord, LodPreprocessStatus,
            hierarchy::{
                GpuLodHierarchyBuilder, GpuLodHierarchyError, GpuLodHierarchyReductionGroup,
                GpuLodHierarchyReductionInput, GpuLodHierarchyReductionOutput,
            },
            preprocess_lod_batch_cpu,
        },
    },
    io::{
        lod::{
            LOD_SHARD_ENTRY_LEN, LOD_SHARD_HEADER_LEN, LodCodecError, LodCodecLimits,
            LodShardEntry, LodShardIndex, decode_lod_shard_index, decode_manifest,
            decode_page_with_descriptor, encode_lod_shard_index, encode_manifest,
            encode_page_with_encoding, lod_shard_prefix_len,
        },
        ply::{MAX_STREAM_BATCH_ALLOCATION_BYTES, stream_ply_3d},
    },
    material::spherical_harmonics::SH_COEFF_COUNT,
};

/// The external hierarchy is a distinct deterministic builder topology. The
/// The canonical Morton/support contract is builder ABI 3; ABI 4 adds
/// external runs and bounded hierarchical reduction; ABI 5 emits the manifest
/// v3 high-fidelity certificate (exact leaves only).
pub const EXTERNAL_LOD_BUILDER_ABI_VERSION: u32 = 5;
/// Global external topology with deterministic f32 GPU MomentMerge summaries.
/// ABI 6 emits the manifest v3 high-fidelity certificate (exact leaves only).
pub const EXTERNAL_GPU_LOD_BUILDER_ABI_VERSION: u32 = 6;

const RUN_MAGIC: [u8; 8] = *b"BGSRUN1\0";
const SUMMARY_MAGIC: [u8; 8] = *b"BGSSUM1\0";
const GAUSSIAN_FLOAT_COUNT: usize = 12 + SH_COEFF_COUNT;
const RUN_RECORD_BYTES: usize = 16 + GAUSSIAN_FLOAT_COUNT * size_of::<f32>();
const SUMMARY_RECORD_BYTES: usize = 80 + GAUSSIAN_FLOAT_COUNT * size_of::<f32>();
const PAGE_CONTAINER_HEADER_BYTES: u64 = 44;

/// Explicit allocation and work limits for an external package build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalLodBuildLimits {
    /// Maximum source records resident for validation, preprocessing and sort.
    pub batch_records: usize,
    /// Maximum sorted runs opened by one merge operation.
    pub merge_fan_in: usize,
    /// Buffer allocated for each merge reader and the merge writer.
    pub run_buffer_bytes: usize,
    /// Aggregate configured merge buffers, excluding the bounded heap heads.
    pub max_merge_buffer_bytes: usize,
    pub max_source_count: u64,
    pub max_run_count: u64,
    /// Peak bytes for owned spill/merge files, excluding final package pages.
    pub max_temporary_bytes: u64,
    /// Caps the only source-size-dependent in-memory output metadata.
    pub max_manifest_nodes: u32,
    pub max_manifest_bytes: u64,
    pub max_encoded_page_bytes: u64,
    /// Hard size of one immutable page shard, including its versioned table.
    pub max_shard_bytes: u64,
    /// Hard range-table bound per shard.
    pub max_pages_per_shard: u32,
    /// Bounded page-read/write overlap. A full channel applies backpressure.
    pub pipeline_depth: usize,
}

impl Default for ExternalLodBuildLimits {
    fn default() -> Self {
        Self {
            batch_records: 65_536,
            merge_fan_in: 32,
            run_buffer_bytes: 64 * 1024,
            max_merge_buffer_bytes: 4 * 1024 * 1024,
            max_source_count: 250_000_000,
            max_run_count: 1_000_000,
            max_temporary_bytes: 256 * 1024 * 1024 * 1024,
            max_manifest_nodes: 2_000_000,
            max_manifest_bytes: 512 * 1024 * 1024,
            max_encoded_page_bytes: 256 * 1024 * 1024,
            max_shard_bytes: 512 * 1024 * 1024,
            max_pages_per_shard: 4096,
            pipeline_depth: 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExternalLodBuildConfig {
    pub settings: GaussianLodBuildSettings,
    /// Optional reduced-degree binary16 SH encoding for representative pages.
    /// Source leaves remain full-degree f32, preserving the exact q=1 cut.
    pub compressed_representative_sh_degree: Option<u8>,
    pub limits: ExternalLodBuildLimits,
}

impl ExternalLodBuildConfig {
    pub fn validate(self) -> Result<Self, ExternalLodBuildError> {
        self.settings
            .validate()
            .map_err(LodBuildError::InvalidSettings)?;
        if self
            .compressed_representative_sh_degree
            .is_some_and(|degree| {
                usize::from(degree) > crate::material::spherical_harmonics::SH_DEGREE
            })
        {
            return Err(ExternalLodBuildError::InvalidConfig(format!(
                "compressed representative SH degree exceeds compiled degree {}",
                crate::material::spherical_harmonics::SH_DEGREE
            )));
        }
        let limits = self.limits;
        if limits.batch_records == 0 {
            return Err(ExternalLodBuildError::InvalidConfig(
                "batch_records must be greater than zero".into(),
            ));
        }
        if limits.batch_records > u32::MAX as usize {
            return Err(ExternalLodBuildError::InvalidConfig(
                "batch_records exceeds the GPU batch index range".into(),
            ));
        }
        let batch_bytes = limits
            .batch_records
            .checked_mul(size_of::<Gaussian3d>())
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("batch allocation size overflow".into())
            })?;
        if batch_bytes > MAX_STREAM_BATCH_ALLOCATION_BYTES {
            return Err(ExternalLodBuildError::InvalidConfig(format!(
                "batch requires {batch_bytes} bytes, exceeding the parser limit {MAX_STREAM_BATCH_ALLOCATION_BYTES}"
            )));
        }
        if limits.merge_fan_in < 2 {
            return Err(ExternalLodBuildError::InvalidConfig(
                "merge_fan_in must be at least two".into(),
            ));
        }
        if limits.run_buffer_bytes == 0 {
            return Err(ExternalLodBuildError::InvalidConfig(
                "run_buffer_bytes must be greater than zero".into(),
            ));
        }
        let merge_buffers = limits
            .merge_fan_in
            .checked_add(1)
            .and_then(|count| count.checked_mul(limits.run_buffer_bytes))
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("merge buffer size overflow".into())
            })?;
        if merge_buffers > limits.max_merge_buffer_bytes {
            return Err(ExternalLodBuildError::InvalidConfig(format!(
                "merge buffers require {merge_buffers} bytes, exceeding max_merge_buffer_bytes {}",
                limits.max_merge_buffer_bytes
            )));
        }
        if limits.max_source_count == 0
            || limits.max_run_count == 0
            || limits.max_temporary_bytes == 0
            || limits.max_manifest_nodes == 0
            || limits.max_manifest_bytes < 40
            || limits.max_encoded_page_bytes < PAGE_CONTAINER_HEADER_BYTES
        {
            return Err(ExternalLodBuildError::InvalidConfig(
                "all external build limits must be non-zero and large enough for their headers"
                    .into(),
            ));
        }
        let minimum_shard_bytes = (LOD_SHARD_HEADER_LEN as u64)
            .checked_add(LOD_SHARD_ENTRY_LEN as u64)
            .and_then(|bytes| bytes.checked_add(PAGE_CONTAINER_HEADER_BYTES))
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("minimum shard size overflow".into())
            })?;
        if limits.max_shard_bytes < minimum_shard_bytes
            || limits.max_pages_per_shard == 0
            || !(1..=64).contains(&limits.pipeline_depth)
        {
            return Err(ExternalLodBuildError::InvalidConfig(
                "shard bytes/pages must be non-zero and pipeline_depth must be in 1..=64".into(),
            ));
        }
        if u64::from(self.settings.leaf_capacity) > limits.batch_records as u64 {
            return Err(ExternalLodBuildError::InvalidConfig(
                "leaf_capacity cannot exceed batch_records; this keeps page/reducer work bounded"
                    .into(),
            ));
        }
        let largest_page = u64::from(self.settings.leaf_capacity)
            .checked_mul(size_of::<Gaussian3d>() as u64)
            .and_then(|bytes| bytes.checked_add(PAGE_CONTAINER_HEADER_BYTES))
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("leaf page size overflow".into())
            })?;
        if largest_page > limits.max_encoded_page_bytes {
            return Err(ExternalLodBuildError::InvalidConfig(format!(
                "leaf pages may require {largest_page} bytes, exceeding max_encoded_page_bytes {}",
                limits.max_encoded_page_bytes
            )));
        }
        Ok(self)
    }
}

const MIN_PARALLEL_MERGE_STREAM_BUFFER_BYTES: usize = 4 * 1024;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MergeParallelLayout {
    workers: usize,
    stream_buffer_bytes: u64,
    aggregate_buffer_bytes: u64,
    head_records: u64,
}

fn merge_parallel_layout(
    run_count: u64,
    config: ExternalLodBuildConfig,
) -> Result<MergeParallelLayout, ExternalLodBuildError> {
    if run_count <= 1 {
        return Ok(MergeParallelLayout::default());
    }
    let fan_in = config.limits.merge_fan_in as u64;
    let group_count = run_count.div_ceil(fan_in);
    let streams_per_worker = fan_in.checked_add(1).ok_or_else(|| {
        ExternalLodBuildError::InvalidConfig("merge stream count overflow".into())
    })?;
    let minimum_stream_buffer = config
        .limits
        .run_buffer_bytes
        .clamp(1, MIN_PARALLEL_MERGE_STREAM_BUFFER_BYTES) as u64;
    let minimum_worker_bytes = streams_per_worker
        .checked_mul(minimum_stream_buffer)
        .ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("minimum merge worker bytes overflow".into())
        })?;
    let workers_by_buffer = (config.limits.max_merge_buffer_bytes as u64)
        .checked_div(minimum_worker_bytes)
        .unwrap_or(0)
        .max(1);
    let workers = group_count
        .min(config.limits.pipeline_depth as u64)
        .min(workers_by_buffer);
    let aggregate_streams = workers.checked_mul(streams_per_worker).ok_or_else(|| {
        ExternalLodBuildError::InvalidConfig("aggregate merge stream count overflow".into())
    })?;
    let stream_buffer_bytes = (config.limits.max_merge_buffer_bytes as u64)
        .checked_div(aggregate_streams)
        .unwrap_or(0)
        .min(config.limits.run_buffer_bytes as u64);
    if stream_buffer_bytes == 0 {
        return Err(ExternalLodBuildError::InvalidConfig(
            "parallel merge leaves no bytes for a stream buffer".into(),
        ));
    }
    let aggregate_buffer_bytes = aggregate_streams
        .checked_mul(stream_buffer_bytes)
        .ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("aggregate merge buffer bytes overflow".into())
        })?;
    let head_records = workers.checked_mul(fan_in).ok_or_else(|| {
        ExternalLodBuildError::InvalidConfig("parallel merge head bound overflow".into())
    })?;
    Ok(MergeParallelLayout {
        workers: workers as usize,
        stream_buffer_bytes,
        aggregate_buffer_bytes,
        head_records,
    })
}

fn stream_handoff_chunk_records(config: ExternalLodBuildConfig) -> usize {
    (config.settings.leaf_capacity as usize)
        .min((config.limits.run_buffer_bytes / RUN_RECORD_BYTES).max(1))
}

/// Pure build descriptor used by capacity checks and very-large-scene tests.
/// Its allocations are logarithmic in source count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalLodBuildPlan {
    pub source_count: u64,
    pub initial_run_count: u64,
    pub merge_pass_count: u32,
    /// Leaf-to-root node counts.
    pub hierarchy_level_counts: Vec<u64>,
    pub total_node_count: u64,
    pub maximum_records_per_batch_buffer: u64,
    pub maximum_reducer_input_records: u64,
    pub maximum_hierarchy_level_summaries: u64,
    pub maximum_page_records: u64,
    /// Maximum merge groups which may execute concurrently. The configured
    /// pipeline depth is additionally constrained by the aggregate merge
    /// buffer budget.
    pub merge_worker_limit: u32,
    pub merge_head_records: u64,
    /// Per-stream buffer selected for the maximum parallel merge layout.
    pub merge_stream_buffer_bytes: u64,
    /// Aggregate reader/writer buffers across every parallel merge worker.
    pub merge_buffer_bytes: u64,
    /// Conservative host allocation bound for PLY batch + canonical batch +
    /// preprocess results + sortable run records.
    pub maximum_spill_host_bytes: u64,
    /// Configured merge buffers plus one full run-record heap head per reader.
    pub maximum_merge_host_bytes: u64,
    /// Bounded records which can coexist across the final-merge producer,
    /// channel, and hierarchy consumer.
    pub maximum_stream_handoff_records: u64,
    pub maximum_stream_handoff_host_bytes: u64,
    /// Final k-way merge state plus the bounded record handoff. This overlaps
    /// hierarchy/page work and is reported separately from barrier merges.
    pub maximum_merge_hierarchy_overlap_host_bytes: u64,
    pub maximum_temporary_run_bytes: u64,
    /// Peak coexistence of one input and one output GPU-summary level spill.
    pub maximum_temporary_summary_bytes: u64,
    /// Aggregate peak of external Morton runs plus global hierarchy summaries.
    pub maximum_temporary_bytes: u64,
}

impl ExternalLodBuildPlan {
    pub fn new(
        source_count: u64,
        config: ExternalLodBuildConfig,
    ) -> Result<Self, ExternalLodBuildError> {
        let config = config.validate()?;
        if source_count == 0 {
            return Err(ExternalLodBuildError::EmptySource);
        }
        if source_count > config.limits.max_source_count {
            return Err(ExternalLodBuildError::LimitExceeded {
                field: "source Gaussians",
                actual: source_count,
                limit: config.limits.max_source_count,
            });
        }
        let batch = config.limits.batch_records as u64;
        let initial_run_count = source_count.div_ceil(batch);
        if initial_run_count > config.limits.max_run_count {
            return Err(ExternalLodBuildError::LimitExceeded {
                field: "sorted runs",
                actual: initial_run_count,
                limit: config.limits.max_run_count,
            });
        }
        let one_run_set_bytes = source_count
            .checked_mul(RUN_RECORD_BYTES as u64)
            .and_then(|bytes| {
                initial_run_count
                    .checked_mul(16)
                    .and_then(|headers| bytes.checked_add(headers))
            })
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("temporary run byte bound overflow".into())
            })?;
        let maximum_temporary_run_bytes = one_run_set_bytes.checked_mul(2).ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("temporary run byte bound overflow".into())
        })?;
        let fan_in = config.limits.merge_fan_in as u64;
        let mut merge_pass_count = 0_u32;
        let mut run_count = initial_run_count;
        while run_count > 1 {
            run_count = run_count.div_ceil(fan_in);
            merge_pass_count = merge_pass_count.checked_add(1).ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("merge pass count overflow".into())
            })?;
        }

        let mut hierarchy_level_counts = Vec::new();
        let mut level_count = balanced_group_count(
            source_count,
            u64::from(config.settings.leaf_capacity),
            source_count > 1,
        );
        let mut total_node_count = 0_u64;
        loop {
            hierarchy_level_counts.push(level_count);
            total_node_count = total_node_count.checked_add(level_count).ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("hierarchy node count overflow".into())
            })?;
            if level_count == 1 {
                break;
            }
            level_count = balanced_group_count(
                level_count,
                u64::from(config.settings.branching_factor),
                false,
            );
        }
        if total_node_count > u64::from(config.limits.max_manifest_nodes) {
            return Err(ExternalLodBuildError::LimitExceeded {
                field: "manifest nodes",
                actual: total_node_count,
                limit: u64::from(config.limits.max_manifest_nodes),
            });
        }
        let maximum_reducer_input_records = u64::from(config.settings.leaf_capacity)
            .max(u64::from(config.settings.branching_factor));
        let maximum_hierarchy_level_summaries =
            hierarchy_level_counts.iter().copied().max().unwrap_or(0);
        let maximum_summary_level_bytes = maximum_hierarchy_level_summaries
            .checked_mul(SUMMARY_RECORD_BYTES as u64)
            .and_then(|bytes| bytes.checked_add(16))
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig(
                    "temporary hierarchy summary byte bound overflow".into(),
                )
            })?;
        let maximum_temporary_summary_bytes =
            maximum_summary_level_bytes.checked_mul(2).ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig(
                    "temporary hierarchy summary byte bound overflow".into(),
                )
            })?;
        let maximum_temporary_bytes = maximum_temporary_run_bytes
            .checked_add(maximum_temporary_summary_bytes)
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("temporary byte bound overflow".into())
            })?;
        if maximum_temporary_bytes > config.limits.max_temporary_bytes {
            return Err(ExternalLodBuildError::LimitExceeded {
                field: "temporary run and hierarchy-summary bytes",
                actual: maximum_temporary_bytes,
                limit: config.limits.max_temporary_bytes,
            });
        }
        let maximum_page_records = u64::from(config.settings.leaf_capacity);
        let merge_layout = merge_parallel_layout(initial_run_count, config)?;
        let merge_buffer_bytes = merge_layout.aggregate_buffer_bytes;
        let spill_record_bytes = (size_of::<Gaussian3d>()
            .checked_add(size_of::<Gaussian3d>())
            .and_then(|bytes| bytes.checked_add(size_of::<LodPreprocessRecord>()))
            .and_then(|bytes| bytes.checked_add(size_of::<RunRecord>()))
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("spill host byte bound overflow".into())
            })?) as u64;
        let maximum_spill_host_bytes = batch.checked_mul(spill_record_bytes).ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("spill host byte bound overflow".into())
        })?;
        let maximum_merge_host_bytes = merge_layout
            .head_records
            .checked_mul(size_of::<RunRecord>() as u64)
            .and_then(|head_bytes| head_bytes.checked_add(merge_buffer_bytes))
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("merge host byte bound overflow".into())
            })?;
        let stream_handoff_chunk_records = stream_handoff_chunk_records(config) as u64;
        let maximum_stream_handoff_records = stream_handoff_chunk_records
            .checked_mul(config.limits.pipeline_depth as u64 + 2)
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig(
                    "final merge handoff record bound overflow".into(),
                )
            })?;
        let maximum_stream_handoff_host_bytes = maximum_stream_handoff_records
            .checked_mul(size_of::<RunRecord>() as u64)
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig(
                    "final merge handoff byte bound overflow".into(),
                )
            })?;
        let final_merge_buffer_bytes = (fan_in + 1)
            .checked_mul(config.limits.run_buffer_bytes as u64)
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("final merge buffer bound overflow".into())
            })?;
        let final_merge_host_bytes = fan_in
            .checked_mul(size_of::<RunRecord>() as u64)
            .and_then(|heads| heads.checked_add(final_merge_buffer_bytes))
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("final merge host bound overflow".into())
            })?;
        let maximum_merge_hierarchy_overlap_host_bytes = final_merge_host_bytes
            .checked_add(maximum_stream_handoff_host_bytes)
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig(
                    "merge/hierarchy overlap host bound overflow".into(),
                )
            })?;
        Ok(Self {
            source_count,
            initial_run_count,
            merge_pass_count,
            hierarchy_level_counts,
            total_node_count,
            maximum_records_per_batch_buffer: batch,
            maximum_reducer_input_records,
            maximum_hierarchy_level_summaries,
            maximum_page_records,
            merge_worker_limit: merge_layout.workers as u32,
            merge_head_records: merge_layout.head_records,
            merge_stream_buffer_bytes: merge_layout.stream_buffer_bytes,
            merge_buffer_bytes,
            maximum_spill_host_bytes,
            maximum_merge_host_bytes,
            maximum_stream_handoff_records,
            maximum_stream_handoff_host_bytes,
            maximum_merge_hierarchy_overlap_host_bytes,
            maximum_temporary_run_bytes,
            maximum_temporary_summary_bytes,
            maximum_temporary_bytes,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalLodBuildReport {
    pub source_count: u64,
    pub node_count: u32,
    pub page_count: u32,
    pub stored_gaussian_count: u64,
    pub initial_run_count: u64,
    pub merge_pass_count: u32,
    pub maximum_encoded_page_bytes: u64,
    pub shard_count: u32,
    pub maximum_shard_bytes: u64,
    pub pipeline_depth: usize,
    /// Total k-way merge groups, including the final streamed group.
    pub merge_group_count: u64,
    /// Largest number of independent merge groups observed in flight.
    pub maximum_concurrent_merge_groups: u32,
    /// Smallest per-stream buffer used by a parallel merge pass.
    pub minimum_merge_stream_buffer_bytes: u64,
    /// Records in one final-merge channel payload.
    pub stream_handoff_chunk_records: u32,
    pub stream_handoff_capacity_batches: u32,
    pub final_merge_streamed: bool,
    pub stage_timings: ExternalLodBuildStageTimings,
    pub preprocessing_stage: &'static str,
    pub hierarchy_stage: &'static str,
    pub maximum_spill_host_bytes: u64,
    pub maximum_merge_host_bytes: u64,
    pub maximum_stream_handoff_host_bytes: u64,
    pub maximum_merge_hierarchy_overlap_host_bytes: u64,
    pub maximum_reducer_input_records: u64,
    /// Effective host/GPU input records in one global reduction dispatch.
    pub maximum_global_reduction_batch_records: u64,
    pub maximum_hierarchy_level_summaries: u64,
    pub maximum_page_records: u64,
    pub maximum_temporary_run_bytes: u64,
    pub maximum_temporary_summary_bytes: u64,
    pub maximum_temporary_bytes: u64,
}

/// Wall-clock benchmark hooks for the bounded external stages. Spill includes
/// source read, preprocess, canonical sort, and overlapped run writes;
/// hierarchy includes reduction and standalone page encoding; shard packing
/// overlaps verified page reads with final shard writes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExternalLodBuildStageTimings {
    pub scan: Duration,
    pub spill: Duration,
    /// Pass barriers required before the remaining runs fit one final k-way
    /// merge. Independent groups inside each pass run concurrently.
    pub merge: Duration,
    /// Sum of all independent barrier-merge group wall times. This may exceed
    /// `merge` because groups execute concurrently.
    pub merge_group_work: Duration,
    /// Exact wall duration for which at least two barrier-merge groups were in
    /// flight, computed from worker start/end intervals.
    pub merge_group_overlap: Duration,
    /// Active k-way merge work in the bounded producer feeding hierarchy.
    pub final_merge_stream_work: Duration,
    /// Producer time blocked by the full bounded handoff channel.
    pub final_merge_backpressure: Duration,
    /// Intersection of final-merge producer lifetime with hierarchy/page work.
    pub merge_hierarchy_overlap: Duration,
    /// Consumer wall time, including any wait for the concurrently streamed
    /// final merge and all dependency-ordered internal hierarchy levels.
    pub hierarchy_and_page_encode: Duration,
    pub shard_pack: Duration,
    pub validate_and_publish: Duration,
    pub total: Duration,
}

/// A replayable source is required because Morton normalization uses global
/// center bounds. Implementations must emit the same normalized records and
/// logical count on every replay.
pub trait ReplayableGaussianSource {
    fn replay(
        &self,
        batch_records: usize,
        consume: &mut dyn FnMut(&[Gaussian3d]) -> Result<(), ExternalLodBuildError>,
    ) -> Result<u64, ExternalLodBuildError>;
}

#[derive(Clone, Debug)]
pub struct PlyGaussianSource {
    path: PathBuf,
}

impl PlyGaussianSource {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl ReplayableGaussianSource for PlyGaussianSource {
    fn replay(
        &self,
        batch_records: usize,
        consume: &mut dyn FnMut(&[Gaussian3d]) -> Result<(), ExternalLodBuildError>,
    ) -> Result<u64, ExternalLodBuildError> {
        let file = File::open(&self.path).map_err(|error| {
            ExternalLodBuildError::Io(io::Error::new(
                error.kind(),
                format!("failed to open '{}': {error}", self.path.display()),
            ))
        })?;
        let mut reader = BufReader::new(file);
        let mut callback_error = None;
        let streamed = stream_ply_3d(&mut reader, batch_records, |batch| {
            if let Err(error) = consume(batch) {
                callback_error = Some(error);
                return Err(io::Error::other("external LoD batch consumer failed"));
            }
            Ok(())
        });
        if let Some(error) = callback_error {
            return Err(error);
        }
        Ok(streamed?.logical_count)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalLodGlobalReductionLimits {
    pub max_records: usize,
    pub max_nodes: usize,
}

/// Bounded post-merge MomentMerge dispatch surface. The production
/// implementation delegates to wgpu; tests can supply a deterministic fake to
/// verify external topology, spill, routing, and atomicity without an adapter.
pub trait ExternalLodGlobalReducer {
    fn limits(&self) -> ExternalLodGlobalReductionLimits;

    fn reduce_leaf_groups(
        &mut self,
        records: &[Gaussian3d],
        groups: &[GpuLodHierarchyReductionGroup],
        support_sigma: f32,
    ) -> Result<Vec<GpuLodHierarchyReductionOutput>, ExternalLodBuildError>;

    fn reduce_summary_groups(
        &mut self,
        inputs: &[GpuLodHierarchyReductionInput],
        groups: &[GpuLodHierarchyReductionGroup],
        support_sigma: f32,
    ) -> Result<Vec<GpuLodHierarchyReductionOutput>, ExternalLodBuildError>;
}

/// Pluggable bounded preprocessor. Implementations may additionally expose a
/// global GPU reducer. Per-run preprocessing must never construct hierarchy
/// products that are discarded by the later external merge.
pub trait ExternalLodBatchPreprocessor {
    fn stage_name(&self) -> &'static str;
    fn output_order(&self) -> ExternalLodPreprocessorOutputOrder {
        ExternalLodPreprocessorOutputOrder::Input
    }
    fn preprocess(
        &mut self,
        records: &[Gaussian3d],
        source_index_base: u64,
        normalization_bounds: LodBounds,
        support_sigma: f32,
    ) -> Result<LodPreprocessBatchOutput, ExternalLodBuildError>;

    fn global_gpu_hierarchy(&mut self) -> Option<&mut dyn ExternalLodGlobalReducer> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExternalLodPreprocessorOutputOrder {
    #[default]
    Input,
    CanonicalMergeKey,
}

#[derive(Default)]
pub struct CpuExternalLodBatchPreprocessor;

impl ExternalLodBatchPreprocessor for CpuExternalLodBatchPreprocessor {
    fn stage_name(&self) -> &'static str {
        "cpu-canonical-preprocess"
    }

    fn preprocess(
        &mut self,
        records: &[Gaussian3d],
        source_index_base: u64,
        normalization_bounds: LodBounds,
        support_sigma: f32,
    ) -> Result<LodPreprocessBatchOutput, ExternalLodBuildError> {
        Ok(preprocess_lod_batch_cpu(
            records,
            source_index_base,
            normalization_bounds,
            support_sigma,
        )?)
    }
}

/// Uses bounded GPU canonical sorting for external runs and exposes the same
/// builder for the actual level-wise hierarchy reduction after the global
/// external merge.
pub struct GpuHierarchyExternalLodBatchPreprocessor<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub builder: &'a mut GpuLodHierarchyBuilder,
    pub settings: GaussianLodBuildSettings,
}

impl ExternalLodBatchPreprocessor for GpuHierarchyExternalLodBatchPreprocessor<'_> {
    fn stage_name(&self) -> &'static str {
        "gpu-canonical-sort-readback"
    }

    fn output_order(&self) -> ExternalLodPreprocessorOutputOrder {
        ExternalLodPreprocessorOutputOrder::CanonicalMergeKey
    }

    fn preprocess(
        &mut self,
        records: &[Gaussian3d],
        source_index_base: u64,
        normalization_bounds: LodBounds,
        support_sigma: f32,
    ) -> Result<LodPreprocessBatchOutput, ExternalLodBuildError> {
        if self.settings.support_sigma.to_bits() != support_sigma.to_bits() {
            return Err(ExternalLodBuildError::PreprocessorContract(
                "GPU hierarchy support sigma differs from external build settings".into(),
            ));
        }
        let sorted = self
            .builder
            .sort_morton_batch(
                self.device,
                self.queue,
                records,
                source_index_base,
                normalization_bounds,
                support_sigma,
            )
            .map_err(ExternalLodBuildError::GpuHierarchy)?;
        let records = sorted
            .into_iter()
            .map(|record| {
                let support_bounds = gaussian_support_bounds(&record.gaussian, support_sigma)
                    .map_err(|error| {
                        ExternalLodBuildError::PreprocessorContract(format!(
                            "GPU hierarchy support bounds failed for source {}: {error}",
                            record.source_index
                        ))
                    })?;
                Ok(LodPreprocessRecord {
                    source_index: record.source_index,
                    morton: record.morton,
                    support_bounds: Some(support_bounds),
                    status: LodPreprocessStatus::VALID,
                })
            })
            .collect::<Result<Vec<_>, ExternalLodBuildError>>()?;
        Ok(LodPreprocessBatchOutput { records })
    }

    fn global_gpu_hierarchy(&mut self) -> Option<&mut dyn ExternalLodGlobalReducer> {
        Some(self)
    }
}

impl ExternalLodGlobalReducer for GpuHierarchyExternalLodBatchPreprocessor<'_> {
    fn limits(&self) -> ExternalLodGlobalReductionLimits {
        let limits = self.builder.limits();
        ExternalLodGlobalReductionLimits {
            max_records: limits.max_records as usize,
            max_nodes: limits.max_nodes as usize,
        }
    }

    fn reduce_leaf_groups(
        &mut self,
        records: &[Gaussian3d],
        groups: &[GpuLodHierarchyReductionGroup],
        support_sigma: f32,
    ) -> Result<Vec<GpuLodHierarchyReductionOutput>, ExternalLodBuildError> {
        Ok(self.builder.reduce_moment_merge_leaf_groups(
            self.device,
            self.queue,
            records,
            groups,
            support_sigma,
        )?)
    }

    fn reduce_summary_groups(
        &mut self,
        inputs: &[GpuLodHierarchyReductionInput],
        groups: &[GpuLodHierarchyReductionGroup],
        support_sigma: f32,
    ) -> Result<Vec<GpuLodHierarchyReductionOutput>, ExternalLodBuildError> {
        Ok(self.builder.reduce_moment_merge_summary_groups(
            self.device,
            self.queue,
            inputs,
            groups,
            support_sigma,
        )?)
    }
}

/// Build and atomically publish one external-memory package.
pub fn build_external_lod_package(
    source: &dyn ReplayableGaussianSource,
    output: &Path,
    config: ExternalLodBuildConfig,
    preprocessor: &mut dyn ExternalLodBatchPreprocessor,
) -> Result<ExternalLodBuildReport, ExternalLodBuildError> {
    let total_started = Instant::now();
    let config = config.validate()?;
    if output.exists() {
        return Err(ExternalLodBuildError::OutputExists(output.to_path_buf()));
    }
    if output.file_name().is_none() {
        return Err(ExternalLodBuildError::InvalidConfig(
            "output must name a package directory".into(),
        ));
    }

    let stage_started = Instant::now();
    let (source_count, normalization_bounds, replay_fingerprint) = scan_source(source, config)?;
    let scan_elapsed = stage_started.elapsed();
    let plan = ExternalLodBuildPlan::new(source_count, config)?;
    let mut staging = StagingDirectory::new(output)?;
    let pages_directory = staging.path().join("pages");
    let work_directory = staging.path().join("work");
    let runs_directory = work_directory.join("runs");
    fs::create_dir(&pages_directory)?;
    fs::create_dir(&work_directory)?;
    fs::create_dir(&runs_directory)?;

    let stage_started = Instant::now();
    let initial_runs = spill_sorted_runs(
        source,
        source_count,
        replay_fingerprint,
        normalization_bounds,
        &runs_directory,
        config,
        preprocessor,
    )?;
    let spill_elapsed = stage_started.elapsed();
    if initial_runs.len() as u64 != plan.initial_run_count {
        return Err(ExternalLodBuildError::InconsistentSource(
            "initial run count changed from the pure build plan".into(),
        ));
    }
    let stage_started = Instant::now();
    let (final_runs, merge_barrier_pass_count, merge_barrier_stats) =
        merge_runs_to_final_inputs(initial_runs, &runs_directory, config)?;
    let merge_elapsed = stage_started.elapsed();
    let final_merge_pending = final_runs.len() > 1;
    let merge_pass_count = merge_barrier_pass_count
        .checked_add(u32::from(final_merge_pending))
        .ok_or_else(|| ExternalLodBuildError::InvalidConfig("merge pass count overflow".into()))?;
    if merge_pass_count != plan.merge_pass_count {
        return Err(ExternalLodBuildError::InconsistentSource(
            "merge pass count changed from the pure build plan".into(),
        ));
    }

    let stage_started = Instant::now();
    let ((hierarchy, hierarchy_stage), final_merge_stats) =
        consume_final_runs(final_runs, source_count, config, |reader| {
            if let Some(gpu) = preprocessor.global_gpu_hierarchy() {
                Ok((
                    build_hierarchy_from_run_gpu(
                        reader,
                        source_count,
                        &pages_directory,
                        &work_directory,
                        config,
                        &plan,
                        gpu,
                    )?,
                    "gpu-global-moment-merge-readback",
                ))
            } else {
                Ok((
                    build_hierarchy_from_run(
                        reader,
                        source_count,
                        &pages_directory,
                        config,
                        &plan,
                    )?,
                    "cpu-moment-merge",
                ))
            }
        })?;
    let hierarchy_elapsed = stage_started.elapsed();
    let stage_started = Instant::now();
    let mut manifest = hierarchy.manifest;
    let shard_report = pack_staged_pages(
        &pages_directory,
        &mut manifest.pages,
        config,
        hierarchy.maximum_encoded_page_bytes,
    )?;
    let shard_pack_elapsed = stage_started.elapsed();
    let stage_started = Instant::now();
    manifest
        .validate()
        .map_err(|error| ExternalLodBuildError::Validation(error.to_string()))?;
    let manifest_bytes = encode_manifest(&manifest)?;
    enforce_limit(
        "encoded manifest bytes",
        manifest_bytes.len() as u64,
        config.limits.max_manifest_bytes,
    )?;
    let codec_limits = package_codec_limits(
        &manifest,
        manifest_bytes.len() as u64,
        hierarchy.maximum_encoded_page_bytes,
    )?;
    let decoded = decode_manifest(&manifest_bytes, codec_limits)?;
    if decoded != manifest {
        return Err(ExternalLodBuildError::Validation(
            "manifest round trip changed the built manifest".into(),
        ));
    }
    write_new_synced(&staging.path().join("scene.gsplatlod"), &manifest_bytes)?;
    verify_staged_package(staging.path(), &manifest, codec_limits)?;

    fs::remove_dir_all(&work_directory)?;
    sync_directory(&pages_directory)?;
    sync_directory(staging.path())?;
    staging.publish(output)?;
    sync_directory(nonempty_parent(output))?;
    let validate_publish_elapsed = stage_started.elapsed();

    Ok(ExternalLodBuildReport {
        source_count,
        node_count: manifest.header.node_count,
        page_count: manifest.header.page_count,
        stored_gaussian_count: manifest.header.stored_gaussian_count,
        initial_run_count: plan.initial_run_count,
        merge_pass_count,
        maximum_encoded_page_bytes: hierarchy.maximum_encoded_page_bytes,
        shard_count: shard_report.shard_count,
        maximum_shard_bytes: shard_report.maximum_shard_bytes,
        pipeline_depth: config.limits.pipeline_depth,
        merge_group_count: merge_barrier_stats
            .group_count
            .saturating_add(u64::from(final_merge_stats.streamed)),
        maximum_concurrent_merge_groups: merge_barrier_stats
            .maximum_concurrent_groups
            .max(u32::from(final_merge_stats.streamed)),
        minimum_merge_stream_buffer_bytes: if merge_barrier_stats.minimum_stream_buffer_bytes == 0 {
            u64::from(final_merge_stats.streamed)
                .saturating_mul(config.limits.run_buffer_bytes as u64)
        } else {
            merge_barrier_stats.minimum_stream_buffer_bytes
        },
        stream_handoff_chunk_records: stream_handoff_chunk_records(config) as u32,
        stream_handoff_capacity_batches: config.limits.pipeline_depth as u32,
        final_merge_streamed: final_merge_stats.streamed,
        stage_timings: ExternalLodBuildStageTimings {
            scan: scan_elapsed,
            spill: spill_elapsed,
            merge: merge_elapsed,
            merge_group_work: merge_barrier_stats.group_work,
            merge_group_overlap: merge_barrier_stats.group_overlap,
            final_merge_stream_work: final_merge_stats.merge_work,
            final_merge_backpressure: final_merge_stats.backpressure,
            merge_hierarchy_overlap: final_merge_stats.hierarchy_overlap,
            hierarchy_and_page_encode: hierarchy_elapsed,
            shard_pack: shard_pack_elapsed,
            validate_and_publish: validate_publish_elapsed,
            total: total_started.elapsed(),
        },
        preprocessing_stage: preprocessor.stage_name(),
        hierarchy_stage,
        maximum_spill_host_bytes: plan.maximum_spill_host_bytes,
        maximum_merge_host_bytes: plan.maximum_merge_host_bytes,
        maximum_stream_handoff_host_bytes: plan.maximum_stream_handoff_host_bytes,
        maximum_merge_hierarchy_overlap_host_bytes: plan.maximum_merge_hierarchy_overlap_host_bytes,
        maximum_reducer_input_records: plan.maximum_reducer_input_records,
        maximum_global_reduction_batch_records: hierarchy.maximum_reduction_batch_records,
        maximum_hierarchy_level_summaries: plan.maximum_hierarchy_level_summaries,
        maximum_page_records: plan.maximum_page_records,
        maximum_temporary_run_bytes: plan.maximum_temporary_run_bytes,
        maximum_temporary_summary_bytes: plan.maximum_temporary_summary_bytes,
        maximum_temporary_bytes: plan.maximum_temporary_bytes,
    })
}

fn scan_source(
    source: &dyn ReplayableGaussianSource,
    config: ExternalLodBuildConfig,
) -> Result<(u64, LodBounds, u64), ExternalLodBuildError> {
    let mut count = 0_u64;
    let mut replay_fingerprint = StableHasher::new();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let replay_count = source.replay(config.limits.batch_records, &mut |batch| {
        if batch.len() > config.limits.batch_records {
            return Err(ExternalLodBuildError::InconsistentSource(format!(
                "source emitted a {}-record batch above the configured {}-record bound",
                batch.len(),
                config.limits.batch_records
            )));
        }
        for gaussian in batch {
            if count >= config.limits.max_source_count {
                return Err(ExternalLodBuildError::LimitExceeded {
                    field: "source Gaussians",
                    actual: count.saturating_add(1),
                    limit: config.limits.max_source_count,
                });
            }
            if !gaussian
                .position_visibility
                .position
                .iter()
                .all(|value| value.is_finite())
            {
                return Err(ExternalLodBuildError::InvalidGaussian {
                    source_index: count,
                    field: "position".into(),
                });
            }
            replay_fingerprint.write(
                &stable_gaussian_hash(&canonicalize_gaussian_zeros(*gaussian)).to_le_bytes(),
            );
            for axis in 0..3 {
                let position = gaussian.position_visibility.position[axis];
                min[axis] = min[axis].min(position);
                max[axis] = max[axis].max(position);
            }
            count = count.checked_add(1).ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("source count overflow".into())
            })?;
        }
        Ok(())
    })?;
    if count == 0 {
        return Err(ExternalLodBuildError::EmptySource);
    }
    if replay_count != count {
        return Err(ExternalLodBuildError::InconsistentSource(format!(
            "source replay reported {replay_count} records after emitting {count}"
        )));
    }
    let bounds = LodBounds::new(min, max)
        .map_err(|error| ExternalLodBuildError::InvalidConfig(error.to_string()))?;
    for axis in 0..3 {
        if !(bounds.max[axis] - bounds.min[axis]).is_finite() {
            return Err(ExternalLodBuildError::InvalidConfig(format!(
                "Morton normalization extent on axis {axis} is not finite"
            )));
        }
    }
    replay_fingerprint.write(&count.to_le_bytes());
    Ok((count, bounds, replay_fingerprint.finish()))
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RunRecord {
    morton: u64,
    source_index: u64,
    gaussian: Gaussian3d,
}

fn run_record_cmp(left: &RunRecord, right: &RunRecord) -> Ordering {
    left.morton
        .cmp(&right.morton)
        .then_with(|| compare_gaussians(&left.gaussian, &right.gaussian))
        .then_with(|| left.source_index.cmp(&right.source_index))
}

fn spill_sorted_runs(
    source: &dyn ReplayableGaussianSource,
    expected_count: u64,
    expected_replay_fingerprint: u64,
    normalization_bounds: LodBounds,
    directory: &Path,
    config: ExternalLodBuildConfig,
    preprocessor: &mut dyn ExternalLodBatchPreprocessor,
) -> Result<Vec<PathBuf>, ExternalLodBuildError> {
    let mut paths = Vec::new();
    let planned_runs = expected_count.div_ceil(config.limits.batch_records as u64);
    reserve_exact(
        &mut paths,
        usize::try_from(planned_runs).map_err(|_| {
            ExternalLodBuildError::InvalidConfig("planned run count exceeds usize".into())
        })?,
        "sorted run paths",
    )?;
    let mut source_index_base = 0_u64;
    let mut emitted_count = 0_u64;
    let mut replay_fingerprint = StableHasher::new();
    let mut canonical = Vec::new();
    reserve_exact(
        &mut canonical,
        config.limits.batch_records,
        "canonical preprocess batch",
    )?;
    let replay_count = std::thread::scope(|scope| {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<(PathBuf, Vec<RunRecord>)>(
            config.limits.pipeline_depth,
        );
        let writer = scope.spawn(move || -> Result<(), ExternalLodBuildError> {
            while let Ok((path, run)) = receiver.recv() {
                write_run(&path, &run, config.limits.run_buffer_bytes)?;
            }
            Ok(())
        });
        let mut replay_result = source.replay(config.limits.batch_records, &mut |batch| {
            if batch.len() > config.limits.batch_records {
                return Err(ExternalLodBuildError::InconsistentSource(format!(
                    "source emitted a {}-record batch above the configured {}-record bound",
                    batch.len(),
                    config.limits.batch_records
                )));
            }
            let mut batch_offset = 0_usize;
            while batch_offset < batch.len() {
                let available = config.limits.batch_records - canonical.len();
                let take = available.min(batch.len() - batch_offset);
                for gaussian in batch[batch_offset..batch_offset + take].iter().copied() {
                    let gaussian = canonicalize_gaussian_zeros(gaussian);
                    replay_fingerprint.write(&stable_gaussian_hash(&gaussian).to_le_bytes());
                    canonical.push(gaussian);
                }
                emitted_count = emitted_count.checked_add(take as u64).ok_or_else(|| {
                    ExternalLodBuildError::InvalidConfig("source count overflow".into())
                })?;
                batch_offset += take;
                if canonical.len() == config.limits.batch_records {
                    let path = directory.join(format!("pass-000-run-{:08}.bgsrun", paths.len()));
                    let run = prepare_canonical_run(
                        &canonical,
                        source_index_base,
                        normalization_bounds,
                        config,
                        preprocessor,
                    )?;
                    paths.push(path.clone());
                    sender.send((path, run)).map_err(|_| {
                        ExternalLodBuildError::Validation(
                            "bounded run-writer pipeline disconnected".into(),
                        )
                    })?;
                    source_index_base = source_index_base
                        .checked_add(canonical.len() as u64)
                        .ok_or_else(|| {
                            ExternalLodBuildError::InvalidConfig("source index overflow".into())
                        })?;
                    canonical.clear();
                }
            }
            Ok(())
        });
        if replay_result.is_ok() && !canonical.is_empty() {
            let path = directory.join(format!("pass-000-run-{:08}.bgsrun", paths.len()));
            let result = prepare_canonical_run(
                &canonical,
                source_index_base,
                normalization_bounds,
                config,
                preprocessor,
            )
            .and_then(|run| {
                paths.push(path.clone());
                sender.send((path, run)).map_err(|_| {
                    ExternalLodBuildError::Validation(
                        "bounded run-writer pipeline disconnected".into(),
                    )
                })
            });
            match result {
                Ok(()) => {
                    match source_index_base
                        .checked_add(canonical.len() as u64)
                        .ok_or_else(|| {
                            ExternalLodBuildError::InvalidConfig("source index overflow".into())
                        }) {
                        Ok(next) => source_index_base = next,
                        Err(error) => replay_result = Err(error),
                    }
                }
                Err(error) => replay_result = Err(error),
            }
        }
        drop(sender);
        let writer_result = writer.join().map_err(|_| {
            ExternalLodBuildError::Validation("bounded run-writer pipeline panicked".into())
        })?;
        match (replay_result, writer_result) {
            (_, Err(writer_error)) => Err(writer_error),
            (Err(replay_error), Ok(())) => Err(replay_error),
            (Ok(count), Ok(())) => Ok(count),
        }
    })?;
    replay_fingerprint.write(&emitted_count.to_le_bytes());
    if replay_count != expected_count
        || emitted_count != expected_count
        || source_index_base != expected_count
        || replay_fingerprint.finish() != expected_replay_fingerprint
    {
        return Err(ExternalLodBuildError::InconsistentSource(format!(
            "second replay count/order/content differs from its validation pass (emitted {emitted_count}, spilled {source_index_base}, reported {replay_count}, expected {expected_count})"
        )));
    }
    Ok(paths)
}

#[allow(clippy::too_many_arguments)]
fn prepare_canonical_run(
    canonical: &[Gaussian3d],
    source_index_base: u64,
    normalization_bounds: LodBounds,
    config: ExternalLodBuildConfig,
    preprocessor: &mut dyn ExternalLodBatchPreprocessor,
) -> Result<Vec<RunRecord>, ExternalLodBuildError> {
    let output_order = preprocessor.output_order();
    let output = preprocessor.preprocess(
        canonical,
        source_index_base,
        normalization_bounds,
        config.settings.support_sigma,
    )?;
    if output.records.len() != canonical.len() {
        return Err(ExternalLodBuildError::PreprocessorContract(
            "preprocessor output length differs from its input".into(),
        ));
    }
    let mut run = Vec::new();
    reserve_exact(&mut run, canonical.len(), "sortable run batch")?;
    let mut seen = vec![false; canonical.len()];
    for (output_index, record) in output.records.into_iter().enumerate() {
        let local_index = record
            .source_index
            .checked_sub(source_index_base)
            .and_then(|index| usize::try_from(index).ok())
            .filter(|&index| index < canonical.len())
            .ok_or_else(|| {
                ExternalLodBuildError::PreprocessorContract(format!(
                    "preprocessor record {} references source {} outside batch {}..{}",
                    output_index,
                    record.source_index,
                    source_index_base,
                    source_index_base + canonical.len() as u64
                ))
            })?;
        if seen[local_index] || !record.is_valid() {
            return Err(ExternalLodBuildError::PreprocessorContract(format!(
                "invalid or duplicate preprocessor record at source {}: status {:?}",
                record.source_index, record.status
            )));
        }
        seen[local_index] = true;
        if record.support_bounds.is_none() {
            return Err(ExternalLodBuildError::PreprocessorContract(format!(
                "valid preprocessor record at source {} has no support bounds",
                record.source_index
            )));
        }
        if output_order == ExternalLodPreprocessorOutputOrder::Input && local_index != output_index
        {
            return Err(ExternalLodBuildError::PreprocessorContract(format!(
                "input-order preprocessor emitted source {} at output {output_index}",
                record.source_index
            )));
        }
        run.push(RunRecord {
            morton: record.morton,
            source_index: record.source_index,
            gaussian: canonical[local_index],
        });
    }
    if seen.iter().any(|seen| !seen) {
        return Err(ExternalLodBuildError::PreprocessorContract(
            "preprocessor omitted one or more source records".into(),
        ));
    }
    match output_order {
        ExternalLodPreprocessorOutputOrder::Input => run.sort_unstable_by(run_record_cmp),
        ExternalLodPreprocessorOutputOrder::CanonicalMergeKey => {
            if !run.is_sorted_by(|left, right| run_record_cmp(left, right).is_le()) {
                return Err(ExternalLodBuildError::PreprocessorContract(
                    "GPU hierarchy output is not in canonical merge-key order".into(),
                ));
            }
        }
    }
    Ok(run)
}

fn write_run(
    path: &Path,
    records: &[RunRecord],
    buffer_bytes: usize,
) -> Result<(), ExternalLodBuildError> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::with_capacity(buffer_bytes, file);
    writer.write_all(&RUN_MAGIC)?;
    writer.write_all(&(records.len() as u64).to_le_bytes())?;
    for record in records {
        write_run_record(&mut writer, record)?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct ParallelTaskInterval {
    started: Instant,
    finished: Instant,
}

#[derive(Clone, Copy, Debug, Default)]
struct MergeBarrierStats {
    group_count: u64,
    maximum_concurrent_groups: u32,
    minimum_stream_buffer_bytes: u64,
    group_work: Duration,
    group_overlap: Duration,
}

impl MergeBarrierStats {
    fn include(&mut self, intervals: &[ParallelTaskInterval], stream_buffer_bytes: u64) {
        self.group_count = self.group_count.saturating_add(intervals.len() as u64);
        if !intervals.is_empty() {
            self.minimum_stream_buffer_bytes = if self.minimum_stream_buffer_bytes == 0 {
                stream_buffer_bytes
            } else {
                self.minimum_stream_buffer_bytes.min(stream_buffer_bytes)
            };
        }
        let (work, overlap, peak) = parallel_interval_stats(intervals);
        self.group_work = self.group_work.saturating_add(work);
        self.group_overlap = self.group_overlap.saturating_add(overlap);
        self.maximum_concurrent_groups = self.maximum_concurrent_groups.max(peak);
    }
}

fn parallel_interval_stats(intervals: &[ParallelTaskInterval]) -> (Duration, Duration, u32) {
    let work = intervals.iter().fold(Duration::ZERO, |total, interval| {
        total.saturating_add(
            interval
                .finished
                .saturating_duration_since(interval.started),
        )
    });
    let mut events = Vec::with_capacity(intervals.len().saturating_mul(2));
    for interval in intervals {
        events.push((interval.started, 1_i8));
        events.push((interval.finished, -1_i8));
    }
    // End events sort before start events at an identical instant so a
    // zero-width handoff is never reported as overlap.
    events.sort_unstable_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let mut active = 0_u32;
    let mut peak = 0_u32;
    let mut overlap = Duration::ZERO;
    let mut previous = events.first().map(|event| event.0);
    for (instant, delta) in events {
        if let Some(previous) = previous
            && active >= 2
        {
            overlap = overlap.saturating_add(instant.saturating_duration_since(previous));
        }
        if delta < 0 {
            active = active.saturating_sub(1);
        } else {
            active = active.saturating_add(1);
            peak = peak.max(active);
        }
        previous = Some(instant);
    }
    (work, overlap, peak)
}

fn run_bounded_indexed_tasks<T, F>(
    task_count: usize,
    worker_count: usize,
    task: F,
) -> Result<(Vec<T>, Vec<ParallelTaskInterval>), ExternalLodBuildError>
where
    T: Send,
    F: Fn(usize) -> Result<T, ExternalLodBuildError> + Sync,
{
    if task_count == 0 || worker_count == 0 || worker_count > task_count {
        return Err(ExternalLodBuildError::InvalidConfig(
            "bounded task worker count is outside 1..=task_count".into(),
        ));
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    let (sender, receiver) = std::sync::mpsc::sync_channel::<
        Result<(usize, T, ParallelTaskInterval), ExternalLodBuildError>,
    >(worker_count);
    let mut values = (0..task_count).map(|_| None).collect::<Vec<_>>();
    let mut intervals = vec![None; task_count];
    let mut first_error = None;
    let mut worker_panicked = false;
    std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let sender = sender.clone();
            let next = &next;
            let cancelled = &cancelled;
            let task = &task;
            workers.push(scope.spawn(move || {
                while !cancelled.load(std::sync::atomic::Ordering::Acquire) {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    if index >= task_count {
                        break;
                    }
                    let started = Instant::now();
                    let result = task(index);
                    let finished = Instant::now();
                    let message = result
                        .map(|value| (index, value, ParallelTaskInterval { started, finished }));
                    if message.is_err() {
                        cancelled.store(true, std::sync::atomic::Ordering::Release);
                    }
                    if sender.send(message).is_err() {
                        break;
                    }
                }
            }));
        }
        drop(sender);
        while let Ok(message) = receiver.recv() {
            match message {
                Ok((index, value, interval)) => {
                    values[index] = Some(value);
                    intervals[index] = Some(interval);
                }
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        for worker in workers {
            worker_panicked |= worker.join().is_err();
        }
    });
    if let Some(error) = first_error {
        return Err(error);
    }
    if worker_panicked {
        return Err(ExternalLodBuildError::Validation(
            "bounded merge worker panicked".into(),
        ));
    }
    if values.iter().any(Option::is_none) || intervals.iter().any(Option::is_none) {
        return Err(ExternalLodBuildError::Validation(
            "bounded merge workers did not complete every indexed task".into(),
        ));
    }
    Ok((
        values.into_iter().map(Option::unwrap).collect(),
        intervals.into_iter().map(Option::unwrap).collect(),
    ))
}

fn merge_runs_to_final_inputs(
    mut paths: Vec<PathBuf>,
    directory: &Path,
    config: ExternalLodBuildConfig,
) -> Result<(Vec<PathBuf>, u32, MergeBarrierStats), ExternalLodBuildError> {
    let mut pass = 0_u32;
    let mut stats = MergeBarrierStats::default();
    while paths.len() > config.limits.merge_fan_in {
        pass = pass.checked_add(1).ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("merge pass count overflow".into())
        })?;
        let group_count = paths.len().div_ceil(config.limits.merge_fan_in);
        let layout = merge_parallel_layout(paths.len() as u64, config)?;
        let stream_buffer_bytes = usize::try_from(layout.stream_buffer_bytes).map_err(|_| {
            ExternalLodBuildError::InvalidConfig("merge stream buffer exceeds usize".into())
        })?;
        let (next, intervals) =
            run_bounded_indexed_tasks(group_count, layout.workers, |output_index| {
                let start = output_index
                    .checked_mul(config.limits.merge_fan_in)
                    .ok_or_else(|| {
                        ExternalLodBuildError::InvalidConfig("merge group index overflow".into())
                    })?;
                let end = start
                    .saturating_add(config.limits.merge_fan_in)
                    .min(paths.len());
                let group = paths.get(start..end).ok_or_else(|| {
                    ExternalLodBuildError::Validation(
                        "parallel merge group falls outside its input pass".into(),
                    )
                })?;
                let output = directory.join(format!("pass-{pass:03}-run-{output_index:08}.bgsrun"));
                merge_run_group(group, &output, stream_buffer_bytes)?;
                Ok(output)
            })?;
        stats.include(&intervals, layout.stream_buffer_bytes);
        for path in paths {
            fs::remove_file(path)?;
        }
        paths = next;
    }
    if paths.is_empty() {
        return Err(ExternalLodBuildError::EmptySource);
    }
    Ok((paths, pass, stats))
}

#[derive(Clone, Copy, Debug)]
struct HeapItem {
    record: RunRecord,
    reader_index: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        run_record_cmp(&self.record, &other.record).is_eq()
            && self.reader_index == other.reader_index
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        run_record_cmp(&other.record, &self.record)
            .then_with(|| other.reader_index.cmp(&self.reader_index))
    }
}

struct RunMerger {
    readers: Vec<RunReader>,
    heap: BinaryHeap<HeapItem>,
    total_count: u64,
    emitted_count: u64,
    previous: Option<RunRecord>,
}

impl RunMerger {
    fn open(inputs: &[PathBuf], buffer_bytes: usize) -> Result<Self, ExternalLodBuildError> {
        let mut readers = Vec::new();
        reserve_exact(&mut readers, inputs.len(), "merge readers")?;
        for path in inputs {
            readers.push(RunReader::open(path, buffer_bytes)?);
        }
        let total_count = readers.iter().try_fold(0_u64, |count, reader| {
            count.checked_add(reader.remaining).ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("merged run count overflow".into())
            })
        })?;
        let mut heap = BinaryHeap::new();
        heap.try_reserve(readers.len()).map_err(|error| {
            ExternalLodBuildError::InvalidConfig(format!(
                "could not reserve bounded merge heap: {error}"
            ))
        })?;
        for (reader_index, reader) in readers.iter_mut().enumerate() {
            if let Some(record) = reader.next_record()? {
                heap.push(HeapItem {
                    record,
                    reader_index,
                });
            }
        }
        Ok(Self {
            readers,
            heap,
            total_count,
            emitted_count: 0,
            previous: None,
        })
    }

    fn total_count(&self) -> u64 {
        self.total_count
    }

    fn next_record(&mut self) -> Result<Option<RunRecord>, ExternalLodBuildError> {
        let Some(item) = self.heap.pop() else {
            if self.emitted_count != self.total_count {
                return Err(ExternalLodBuildError::RunCorrupt(format!(
                    "k-way merge emitted {} records, expected {}",
                    self.emitted_count, self.total_count
                )));
            }
            return Ok(None);
        };
        if self
            .previous
            .is_some_and(|previous| run_record_cmp(&previous, &item.record).is_gt())
        {
            return Err(ExternalLodBuildError::RunCorrupt(
                "k-way merge produced descending records".into(),
            ));
        }
        self.emitted_count = self.emitted_count.checked_add(1).ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("merged run count overflow".into())
        })?;
        self.previous = Some(item.record);
        if let Some(record) = self.readers[item.reader_index].next_record()? {
            self.heap.push(HeapItem {
                record,
                reader_index: item.reader_index,
            });
        }
        Ok(Some(item.record))
    }

    fn finish(self) -> Result<(), ExternalLodBuildError> {
        if !self.heap.is_empty() || self.emitted_count != self.total_count {
            return Err(ExternalLodBuildError::RunCorrupt(format!(
                "k-way merge stopped after {} of {} records",
                self.emitted_count, self.total_count
            )));
        }
        for reader in self.readers {
            reader.finish()?;
        }
        Ok(())
    }
}

fn merge_run_group(
    inputs: &[PathBuf],
    output: &Path,
    buffer_bytes: usize,
) -> Result<(), ExternalLodBuildError> {
    let mut merger = RunMerger::open(inputs, buffer_bytes)?;
    let total_count = merger.total_count();
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut writer = BufWriter::with_capacity(buffer_bytes, file);
    writer.write_all(&RUN_MAGIC)?;
    writer.write_all(&total_count.to_le_bytes())?;
    let mut written = 0_u64;
    while let Some(record) = merger.next_record()? {
        write_run_record(&mut writer, &record)?;
        written = written.checked_add(1).ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("merged run write count overflow".into())
        })?;
    }
    merger.finish()?;
    if written != total_count {
        return Err(ExternalLodBuildError::RunCorrupt(format!(
            "merged run wrote {written} records, expected {total_count}"
        )));
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

struct RunReader {
    reader: BufReader<File>,
    remaining: u64,
}

impl RunReader {
    fn open(path: &Path, buffer_bytes: usize) -> Result<Self, ExternalLodBuildError> {
        let mut reader = BufReader::with_capacity(buffer_bytes, File::open(path)?);
        let mut header = [0_u8; 16];
        reader.read_exact(&mut header)?;
        if header[0..8] != RUN_MAGIC {
            return Err(ExternalLodBuildError::RunCorrupt(format!(
                "'{}' has an invalid run header",
                path.display()
            )));
        }
        let remaining = u64::from_le_bytes(header[8..16].try_into().unwrap());
        Ok(Self { reader, remaining })
    }

    fn next_record(&mut self) -> Result<Option<RunRecord>, ExternalLodBuildError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let mut bytes = [0_u8; RUN_RECORD_BYTES];
        self.reader.read_exact(&mut bytes)?;
        self.remaining -= 1;
        Ok(Some(decode_run_record(&bytes)?))
    }

    fn finish(mut self) -> Result<(), ExternalLodBuildError> {
        if self.remaining != 0 {
            return Err(ExternalLodBuildError::RunCorrupt(
                "run reader was not fully consumed".into(),
            ));
        }
        let mut trailing = [0_u8; 1];
        if self.reader.read(&mut trailing)? != 0 {
            return Err(ExternalLodBuildError::RunCorrupt(
                "run contains trailing bytes".into(),
            ));
        }
        Ok(())
    }
}

struct StreamedRunReader {
    receiver: std::sync::mpsc::Receiver<Vec<RunRecord>>,
    current: std::vec::IntoIter<RunRecord>,
    remaining: u64,
}

impl StreamedRunReader {
    fn new(receiver: std::sync::mpsc::Receiver<Vec<RunRecord>>, remaining: u64) -> Self {
        Self {
            receiver,
            current: Vec::new().into_iter(),
            remaining,
        }
    }

    fn next_record(&mut self) -> Result<Option<RunRecord>, ExternalLodBuildError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        loop {
            if let Some(record) = self.current.next() {
                self.remaining -= 1;
                return Ok(Some(record));
            }
            let batch = self.receiver.recv().map_err(|_| {
                ExternalLodBuildError::RunCorrupt(
                    "final merge stream disconnected before the declared source count".into(),
                )
            })?;
            if batch.is_empty() {
                return Err(ExternalLodBuildError::RunCorrupt(
                    "final merge stream emitted an empty batch".into(),
                ));
            }
            self.current = batch.into_iter();
        }
    }

    fn finish(self) -> Result<(), ExternalLodBuildError> {
        if self.remaining != 0 {
            return Err(ExternalLodBuildError::RunCorrupt(
                "final merge stream was not fully consumed".into(),
            ));
        }
        if self.current.len() != 0 {
            return Err(ExternalLodBuildError::RunCorrupt(
                "final merge stream exceeded the declared source count".into(),
            ));
        }
        while let Ok(batch) = self.receiver.recv() {
            if !batch.is_empty() {
                return Err(ExternalLodBuildError::RunCorrupt(
                    "final merge stream contains trailing records".into(),
                ));
            }
        }
        Ok(())
    }
}

enum FinalRunInput {
    File(RunReader),
    Stream(StreamedRunReader),
}

impl FinalRunInput {
    fn remaining(&self) -> u64 {
        match self {
            Self::File(reader) => reader.remaining,
            Self::Stream(reader) => reader.remaining,
        }
    }

    fn next_record(&mut self) -> Result<Option<RunRecord>, ExternalLodBuildError> {
        match self {
            Self::File(reader) => reader.next_record(),
            Self::Stream(reader) => reader.next_record(),
        }
    }

    fn finish(self) -> Result<(), ExternalLodBuildError> {
        match self {
            Self::File(reader) => reader.finish(),
            Self::Stream(reader) => reader.finish(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FinalMergePipelineStats {
    streamed: bool,
    merge_work: Duration,
    backpressure: Duration,
    hierarchy_overlap: Duration,
}

#[derive(Clone, Copy, Debug)]
struct FinalMergeProducerStats {
    started: Instant,
    finished: Instant,
    merge_work: Duration,
    backpressure: Duration,
    completed: bool,
}

fn produce_final_merged_run(
    paths: &[PathBuf],
    source_count: u64,
    buffer_bytes: usize,
    chunk_records: usize,
    sender: std::sync::mpsc::SyncSender<Vec<RunRecord>>,
) -> Result<FinalMergeProducerStats, ExternalLodBuildError> {
    let started = Instant::now();
    let open_started = Instant::now();
    let mut merger = RunMerger::open(paths, buffer_bytes)?;
    let mut merge_work = open_started.elapsed();
    if merger.total_count() != source_count {
        return Err(ExternalLodBuildError::RunCorrupt(format!(
            "final merge declares {} records, expected {source_count}",
            merger.total_count()
        )));
    }
    let mut backpressure = Duration::ZERO;
    let mut emitted = 0_u64;
    loop {
        let merge_started = Instant::now();
        let mut batch = Vec::with_capacity(chunk_records);
        while batch.len() < chunk_records {
            let Some(record) = merger.next_record()? else {
                break;
            };
            batch.push(record);
        }
        merge_work = merge_work.saturating_add(merge_started.elapsed());
        if batch.is_empty() {
            break;
        }
        emitted = emitted.checked_add(batch.len() as u64).ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("final merge emission count overflow".into())
        })?;
        match sender.try_send(batch) {
            Ok(()) => {}
            Err(std::sync::mpsc::TrySendError::Full(batch)) => {
                let blocked = Instant::now();
                if sender.send(batch).is_err() {
                    backpressure = backpressure.saturating_add(blocked.elapsed());
                    return Ok(FinalMergeProducerStats {
                        started,
                        finished: Instant::now(),
                        merge_work,
                        backpressure,
                        completed: false,
                    });
                }
                backpressure = backpressure.saturating_add(blocked.elapsed());
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                return Ok(FinalMergeProducerStats {
                    started,
                    finished: Instant::now(),
                    merge_work,
                    backpressure,
                    completed: false,
                });
            }
        }
    }
    let finish_started = Instant::now();
    merger.finish()?;
    merge_work = merge_work.saturating_add(finish_started.elapsed());
    if emitted != source_count {
        return Err(ExternalLodBuildError::RunCorrupt(format!(
            "final merge streamed {emitted} records, expected {source_count}"
        )));
    }
    Ok(FinalMergeProducerStats {
        started,
        finished: Instant::now(),
        merge_work,
        backpressure,
        completed: true,
    })
}

fn interval_intersection(
    left_start: Instant,
    left_end: Instant,
    right_start: Instant,
    right_end: Instant,
) -> Duration {
    let start = left_start.max(right_start);
    let end = left_end.min(right_end);
    end.saturating_duration_since(start)
}

fn consume_final_runs<T>(
    paths: Vec<PathBuf>,
    source_count: u64,
    config: ExternalLodBuildConfig,
    consume: impl FnOnce(FinalRunInput) -> Result<T, ExternalLodBuildError>,
) -> Result<(T, FinalMergePipelineStats), ExternalLodBuildError> {
    if paths.is_empty() {
        return Err(ExternalLodBuildError::EmptySource);
    }
    if paths.len() == 1 {
        let path = paths.into_iter().next().unwrap();
        let reader = RunReader::open(&path, config.limits.run_buffer_bytes)?;
        let value = consume(FinalRunInput::File(reader))?;
        fs::remove_file(path)?;
        return Ok((value, FinalMergePipelineStats::default()));
    }

    let chunk_records = stream_handoff_chunk_records(config);
    let (sender, receiver) =
        std::sync::mpsc::sync_channel::<Vec<RunRecord>>(config.limits.pipeline_depth);
    std::thread::scope(|scope| {
        let producer = scope.spawn(move || {
            let result = produce_final_merged_run(
                &paths,
                source_count,
                config.limits.run_buffer_bytes,
                chunk_records,
                sender,
            );
            if result.as_ref().is_ok_and(|stats| stats.completed) {
                for path in paths {
                    fs::remove_file(path)?;
                }
            }
            result
        });
        let hierarchy_started = Instant::now();
        let consumed = consume(FinalRunInput::Stream(StreamedRunReader::new(
            receiver,
            source_count,
        )));
        let hierarchy_finished = Instant::now();
        let produced = producer.join().map_err(|_| {
            ExternalLodBuildError::Validation("final merge producer panicked".into())
        })?;
        let producer_stats = match produced {
            Ok(stats) => stats,
            Err(error) => return Err(error),
        };
        let value = consumed?;
        if !producer_stats.completed {
            return Err(ExternalLodBuildError::Validation(
                "hierarchy completed after cancelling its final merge producer".into(),
            ));
        }
        Ok((
            value,
            FinalMergePipelineStats {
                streamed: true,
                merge_work: producer_stats.merge_work,
                backpressure: producer_stats.backpressure,
                hierarchy_overlap: interval_intersection(
                    producer_stats.started,
                    producer_stats.finished,
                    hierarchy_started,
                    hierarchy_finished,
                ),
            },
        ))
    })
}

fn write_run_record(
    writer: &mut impl Write,
    record: &RunRecord,
) -> Result<(), ExternalLodBuildError> {
    let mut bytes = [0_u8; RUN_RECORD_BYTES];
    bytes[0..8].copy_from_slice(&record.morton.to_le_bytes());
    bytes[8..16].copy_from_slice(&record.source_index.to_le_bytes());
    let mut offset = 16;
    for value in gaussian_floats(&record.gaussian) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        offset += 4;
    }
    debug_assert_eq!(offset, RUN_RECORD_BYTES);
    writer.write_all(&bytes)?;
    Ok(())
}

fn decode_run_record(bytes: &[u8; RUN_RECORD_BYTES]) -> Result<RunRecord, ExternalLodBuildError> {
    let morton = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let source_index = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let mut values = [0.0_f32; GAUSSIAN_FLOAT_COUNT];
    let mut offset = 16;
    for value in &mut values {
        *value = f32::from_bits(u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().unwrap(),
        ));
        offset += 4;
    }
    let gaussian = gaussian_from_floats(values);
    validate_gaussian(&gaussian).map_err(|field| {
        ExternalLodBuildError::RunCorrupt(format!("run contains invalid {field:?}"))
    })?;
    Ok(RunRecord {
        morton,
        source_index,
        gaussian,
    })
}

fn gaussian_floats(gaussian: &Gaussian3d) -> [f32; GAUSSIAN_FLOAT_COUNT] {
    let mut values = [0.0; GAUSSIAN_FLOAT_COUNT];
    let mut offset = 0;
    for value in gaussian
        .position_visibility
        .position
        .into_iter()
        .chain([gaussian.position_visibility.visibility])
        .chain(gaussian.spherical_harmonic.coefficients)
        .chain(gaussian.rotation.rotation)
        .chain(gaussian.scale_opacity.scale)
        .chain([gaussian.scale_opacity.opacity])
    {
        values[offset] = value;
        offset += 1;
    }
    debug_assert_eq!(offset, GAUSSIAN_FLOAT_COUNT);
    values
}

fn gaussian_from_floats(values: [f32; GAUSSIAN_FLOAT_COUNT]) -> Gaussian3d {
    let mut offset = 0;
    let mut take = || {
        let value = values[offset];
        offset += 1;
        value
    };
    let position = [take(), take(), take()];
    let visibility = take();
    let coefficients = std::array::from_fn(|_| take());
    let rotation = [take(), take(), take(), take()];
    let scale = [take(), take(), take()];
    let opacity = take();
    debug_assert_eq!(offset, GAUSSIAN_FLOAT_COUNT);
    Gaussian3d {
        position_visibility: [position[0], position[1], position[2], visibility].into(),
        spherical_harmonic: crate::material::spherical_harmonics::SphericalHarmonicCoefficients {
            coefficients,
        },
        rotation: rotation.into(),
        scale_opacity: [scale[0], scale[1], scale[2], opacity].into(),
    }
}

#[derive(Clone)]
struct ReductionSummary {
    draft_index: usize,
    source: LodSourceRange,
    morton: LodMortonRange,
    bounds: LodBounds,
    representative: Gaussian3d,
    reduction_error: LodError,
}

#[derive(Clone, Copy)]
struct GpuReductionSummary {
    draft_index: usize,
    source: LodSourceRange,
    morton: LodMortonRange,
    bounds: LodBounds,
    representative: Gaussian3d,
    reduction_error: LodError,
}

struct SummaryWriter {
    writer: BufWriter<File>,
    expected: u64,
    written: u64,
}

impl SummaryWriter {
    fn create(
        path: &Path,
        expected: u64,
        buffer_bytes: usize,
    ) -> Result<Self, ExternalLodBuildError> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut writer = BufWriter::with_capacity(buffer_bytes, file);
        writer.write_all(&SUMMARY_MAGIC)?;
        writer.write_all(&expected.to_le_bytes())?;
        Ok(Self {
            writer,
            expected,
            written: 0,
        })
    }

    fn write(&mut self, summary: &GpuReductionSummary) -> Result<(), ExternalLodBuildError> {
        if self.written >= self.expected {
            return Err(ExternalLodBuildError::RunCorrupt(
                "hierarchy summary spill exceeded its planned count".into(),
            ));
        }
        write_summary_record(&mut self.writer, summary)?;
        self.written += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<(), ExternalLodBuildError> {
        if self.written != self.expected {
            return Err(ExternalLodBuildError::RunCorrupt(format!(
                "hierarchy summary spill wrote {}, expected {}",
                self.written, self.expected
            )));
        }
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }
}

struct SummaryReader {
    reader: BufReader<File>,
    remaining: u64,
}

impl SummaryReader {
    fn open(
        path: &Path,
        expected: u64,
        buffer_bytes: usize,
    ) -> Result<Self, ExternalLodBuildError> {
        let mut reader = BufReader::with_capacity(buffer_bytes, File::open(path)?);
        let mut header = [0_u8; 16];
        reader.read_exact(&mut header)?;
        if header[0..8] != SUMMARY_MAGIC {
            return Err(ExternalLodBuildError::RunCorrupt(format!(
                "'{}' has an invalid hierarchy summary header",
                path.display()
            )));
        }
        let remaining = u64::from_le_bytes(header[8..16].try_into().unwrap());
        if remaining != expected {
            return Err(ExternalLodBuildError::RunCorrupt(format!(
                "hierarchy summary declares {remaining} records, expected {expected}"
            )));
        }
        Ok(Self { reader, remaining })
    }

    fn next(&mut self) -> Result<Option<GpuReductionSummary>, ExternalLodBuildError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let mut bytes = [0_u8; SUMMARY_RECORD_BYTES];
        self.reader.read_exact(&mut bytes)?;
        self.remaining -= 1;
        Ok(Some(decode_summary_record(&bytes)?))
    }

    fn finish(mut self) -> Result<(), ExternalLodBuildError> {
        if self.remaining != 0 {
            return Err(ExternalLodBuildError::RunCorrupt(
                "hierarchy summary reader was not fully consumed".into(),
            ));
        }
        let mut trailing = [0_u8; 1];
        if self.reader.read(&mut trailing)? != 0 {
            return Err(ExternalLodBuildError::RunCorrupt(
                "hierarchy summary contains trailing bytes".into(),
            ));
        }
        Ok(())
    }
}

fn write_summary_record(
    writer: &mut impl Write,
    summary: &GpuReductionSummary,
) -> Result<(), ExternalLodBuildError> {
    let mut bytes = [0_u8; SUMMARY_RECORD_BYTES];
    let mut offset = 0_usize;
    for value in [
        u64::try_from(summary.draft_index).map_err(|_| {
            ExternalLodBuildError::InvalidConfig("summary draft index exceeds u64".into())
        })?,
        summary.source.start,
        summary.source.count,
        summary.morton.min,
        summary.morton.max,
    ] {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        offset += 8;
    }
    for value in summary
        .bounds
        .min
        .into_iter()
        .chain(summary.bounds.max)
        .chain(gaussian_floats(&summary.representative))
        .chain([
            summary.reduction_error.geometric,
            summary.reduction_error.appearance,
            summary.reduction_error.opacity,
            summary.reduction_error.combined,
        ])
    {
        bytes[offset..offset + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        offset += 4;
    }
    debug_assert_eq!(offset, SUMMARY_RECORD_BYTES);
    writer.write_all(&bytes)?;
    Ok(())
}

fn decode_summary_record(
    bytes: &[u8; SUMMARY_RECORD_BYTES],
) -> Result<GpuReductionSummary, ExternalLodBuildError> {
    let mut offset = 0_usize;
    let mut integers = [0_u64; 5];
    for value in &mut integers {
        *value = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
    }
    let draft_index = usize::try_from(integers[0]).map_err(|_| {
        ExternalLodBuildError::RunCorrupt("summary draft index exceeds usize".into())
    })?;
    let source = LodSourceRange {
        start: integers[1],
        count: integers[2],
    };
    let morton = LodMortonRange {
        min: integers[3],
        max: integers[4],
    };
    let mut take_f32 = || {
        let value = f32::from_bits(u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().unwrap(),
        ));
        offset += 4;
        value
    };
    let bounds = LodBounds::new(
        [take_f32(), take_f32(), take_f32()],
        [take_f32(), take_f32(), take_f32()],
    )
    .map_err(|error| {
        ExternalLodBuildError::RunCorrupt(format!("hierarchy summary has invalid bounds: {error}"))
    })?;
    let gaussian_values = std::array::from_fn(|_| take_f32());
    let representative = gaussian_from_floats(gaussian_values);
    validate_gaussian(&representative).map_err(|field| {
        ExternalLodBuildError::RunCorrupt(format!(
            "hierarchy summary has invalid representative {field:?}"
        ))
    })?;
    let reduction_error = LodError {
        geometric: take_f32(),
        appearance: take_f32(),
        opacity: take_f32(),
        combined: take_f32(),
    };
    debug_assert_eq!(offset, SUMMARY_RECORD_BYTES);
    validate_external_error(reduction_error)?;
    if source.count == 0 || source.end().is_none() || morton.min > morton.max {
        return Err(ExternalLodBuildError::RunCorrupt(
            "hierarchy summary has an invalid source or Morton range".into(),
        ));
    }
    Ok(GpuReductionSummary {
        draft_index,
        source,
        morton,
        bounds,
        representative,
        reduction_error,
    })
}

fn validate_external_error(error: LodError) -> Result<(), ExternalLodBuildError> {
    let values = [
        error.geometric,
        error.appearance,
        error.opacity,
        error.combined,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || error.combined < error.geometric
        || error.combined < error.appearance
        || error.combined < error.opacity
    {
        return Err(ExternalLodBuildError::RunCorrupt(format!(
            "hierarchy summary has invalid error {values:?}"
        )));
    }
    Ok(())
}

struct NodeDraft {
    children: Option<(usize, usize)>,
    source: LodSourceRange,
    morton: LodMortonRange,
    bounds: LodBounds,
    page_id: LodPageId,
    page_count: u32,
    error: LodError,
}

struct HierarchyBuild {
    manifest: GaussianLodManifest,
    maximum_encoded_page_bytes: u64,
    maximum_reduction_batch_records: u64,
}

#[derive(Clone, Copy)]
struct PendingGpuReduction {
    draft_index: Option<usize>,
    children: Option<(usize, usize)>,
    source: LodSourceRange,
    morton: LodMortonRange,
    bounds: LodBounds,
}

#[allow(clippy::too_many_arguments)]
fn build_hierarchy_from_run_gpu(
    mut reader: FinalRunInput,
    source_count: u64,
    pages_directory: &Path,
    work_directory: &Path,
    config: ExternalLodBuildConfig,
    plan: &ExternalLodBuildPlan,
    gpu: &mut dyn ExternalLodGlobalReducer,
) -> Result<HierarchyBuild, ExternalLodBuildError> {
    let gpu_limits = gpu.limits();
    let effective_max_records =
        effective_gpu_batch_records(config.limits.batch_records, gpu_limits.max_records);
    validate_gpu_reduction_group_capacity(
        config.settings.leaf_capacity as usize,
        true,
        effective_max_records,
        gpu_limits.max_nodes,
    )?;
    validate_gpu_reduction_group_capacity(
        config.settings.branching_factor as usize,
        false,
        effective_max_records,
        gpu_limits.max_nodes,
    )?;

    if reader.remaining() != source_count {
        return Err(ExternalLodBuildError::RunCorrupt(format!(
            "final run declares {} records, expected {source_count}",
            reader.remaining()
        )));
    }
    let planned_nodes = usize::try_from(plan.total_node_count).map_err(|_| {
        ExternalLodBuildError::InvalidConfig("planned node count exceeds usize".into())
    })?;
    let mut drafts = Vec::new();
    let mut descriptors = Vec::new();
    reserve_exact(&mut drafts, planned_nodes, "hierarchy drafts")?;
    reserve_exact(&mut descriptors, planned_nodes, "page descriptors")?;
    let mut levels = Vec::<(usize, usize)>::with_capacity(plan.hierarchy_level_counts.len());
    let mut stored_gaussian_count = 0_u64;
    let mut maximum_encoded_page_bytes = 0_u64;
    let mut source_fingerprint = StableHasher::new();
    source_fingerprint.write(&source_count.to_le_bytes());

    let leaf_count = plan.hierarchy_level_counts[0];
    let leaf_summary_path = work_directory.join("hierarchy-summary-000.bgssum");
    let mut summary_writer = SummaryWriter::create(
        &leaf_summary_path,
        leaf_count,
        config.limits.run_buffer_bytes,
    )?;
    let (leaf_base, leaf_remainder) = balanced_group_sizes(source_count, leaf_count);
    let leaf_level_start = drafts.len();
    let mut batch_gaussians = Vec::new();
    let mut batch_groups = Vec::new();
    let mut batch_pending = Vec::new();
    reserve_exact(
        &mut batch_gaussians,
        config.limits.batch_records,
        "GPU leaf reduction batch",
    )?;
    reserve_exact(
        &mut batch_groups,
        effective_max_records.min(gpu_limits.max_nodes),
        "GPU leaf group descriptors",
    )?;
    reserve_exact(
        &mut batch_pending,
        effective_max_records.min(gpu_limits.max_nodes),
        "GPU leaf output metadata",
    )?;
    let mut canonical_start = 0_u64;
    for leaf_index in 0..leaf_count {
        let count = leaf_base + u64::from(leaf_index < leaf_remainder);
        let count_usize = usize::try_from(count)
            .map_err(|_| ExternalLodBuildError::InvalidConfig("leaf count exceeds usize".into()))?;
        if !batch_groups.is_empty()
            && !gpu_batch_can_add(
                batch_gaussians.len(),
                batch_groups.len(),
                count_usize,
                true,
                effective_max_records,
                gpu_limits.max_nodes,
            )
        {
            flush_gpu_leaf_batch(
                gpu,
                config.settings.support_sigma,
                &mut batch_gaussians,
                &mut batch_groups,
                &mut batch_pending,
                &mut summary_writer,
            )?;
        }
        validate_gpu_reduction_group_capacity(
            count_usize,
            true,
            effective_max_records,
            gpu_limits.max_nodes,
        )?;
        let mut gaussians = Vec::new();
        reserve_exact(&mut gaussians, count_usize, "leaf page records")?;
        let mut morton_min = u64::MAX;
        let mut morton_max = 0_u64;
        let mut bounds: Option<LodBounds> = None;
        for _ in 0..count {
            let record = reader.next_record()?.ok_or_else(|| {
                ExternalLodBuildError::RunCorrupt("final run ended inside a leaf".into())
            })?;
            source_fingerprint.write(&record.morton.to_le_bytes());
            source_fingerprint.write(&stable_gaussian_hash(&record.gaussian).to_le_bytes());
            morton_min = morton_min.min(record.morton);
            morton_max = morton_max.max(record.morton);
            let support = gaussian_support_bounds(&record.gaussian, config.settings.support_sigma)?;
            bounds = Some(match bounds {
                Some(current) => current.union(support),
                None => support,
            });
            gaussians.push(record.gaussian);
        }
        let bounds = bounds.ok_or_else(|| {
            ExternalLodBuildError::RunCorrupt("balanced hierarchy emitted an empty leaf".into())
        })?;
        let source = LodSourceRange {
            start: canonical_start,
            count,
        };
        let morton = LodMortonRange {
            min: morton_min,
            max: morton_max,
        };
        let page_id = LodPageId(drafts.len() as u64 + 1);
        batch_groups.push(GpuLodHierarchyReductionGroup {
            start: batch_gaussians.len() as u32,
            count: count as u32,
        });
        batch_gaussians.extend_from_slice(&gaussians);
        let (descriptor, encoded_bytes, encoding_error) = write_page(
            pages_directory,
            page_id,
            LodPageKind::SourceLeaves,
            gaussians,
            config,
        )?;
        debug_assert_eq!(encoding_error, LodError::ZERO);
        maximum_encoded_page_bytes = maximum_encoded_page_bytes.max(encoded_bytes);
        stored_gaussian_count = stored_gaussian_count
            .checked_add(u64::from(descriptor.gaussian_count))
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("stored Gaussian count overflow".into())
            })?;
        let draft_index = drafts.len();
        drafts.push(NodeDraft {
            children: None,
            source,
            morton,
            bounds,
            page_id,
            page_count: descriptor.gaussian_count,
            error: LodError::ZERO,
        });
        descriptors.push(descriptor);
        batch_pending.push(PendingGpuReduction {
            draft_index: Some(draft_index),
            children: None,
            source,
            morton,
            bounds,
        });
        canonical_start = canonical_start.checked_add(count).ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("canonical source offset overflow".into())
        })?;
    }
    flush_gpu_leaf_batch(
        gpu,
        config.settings.support_sigma,
        &mut batch_gaussians,
        &mut batch_groups,
        &mut batch_pending,
        &mut summary_writer,
    )?;
    summary_writer.finish()?;
    levels.push((leaf_level_start, drafts.len()));
    reader.finish()?;
    if canonical_start != source_count {
        return Err(ExternalLodBuildError::RunCorrupt(
            "leaf partition did not consume the canonical source".into(),
        ));
    }

    let mut previous_path = leaf_summary_path;
    let mut previous_count = leaf_count;
    for (level_index, &next_count) in plan.hierarchy_level_counts.iter().enumerate().skip(1) {
        let level_start = drafts.len();
        let next_path = work_directory.join(format!("hierarchy-summary-{level_index:03}.bgssum"));
        let mut previous = SummaryReader::open(
            &previous_path,
            previous_count,
            config.limits.run_buffer_bytes,
        )?;
        let mut next =
            SummaryWriter::create(&next_path, next_count, config.limits.run_buffer_bytes)?;
        let (base, remainder) = balanced_group_sizes(previous_count, next_count);
        let mut inputs = Vec::new();
        let mut groups = Vec::new();
        let mut pending = Vec::new();
        reserve_exact(
            &mut inputs,
            effective_max_records,
            "GPU internal reduction inputs",
        )?;
        reserve_exact(
            &mut groups,
            effective_max_records.min(gpu_limits.max_nodes),
            "GPU internal group descriptors",
        )?;
        reserve_exact(
            &mut pending,
            effective_max_records.min(gpu_limits.max_nodes),
            "GPU internal output metadata",
        )?;
        for group_index in 0..next_count {
            let child_count = base + u64::from(group_index < remainder);
            let child_count_usize = usize::try_from(child_count).map_err(|_| {
                ExternalLodBuildError::InvalidConfig("hierarchy group exceeds usize".into())
            })?;
            if !groups.is_empty()
                && !gpu_batch_can_add(
                    inputs.len(),
                    groups.len(),
                    child_count_usize,
                    false,
                    effective_max_records,
                    gpu_limits.max_nodes,
                )
            {
                flush_gpu_internal_batch(
                    gpu,
                    config,
                    pages_directory,
                    &mut inputs,
                    &mut groups,
                    &mut pending,
                    &mut next,
                    &mut drafts,
                    &mut descriptors,
                    &mut stored_gaussian_count,
                    &mut maximum_encoded_page_bytes,
                )?;
            }
            validate_gpu_reduction_group_capacity(
                child_count_usize,
                false,
                effective_max_records,
                gpu_limits.max_nodes,
            )?;
            let mut children = Vec::new();
            reserve_exact(
                &mut children,
                child_count_usize,
                "hierarchy child summaries",
            )?;
            for _ in 0..child_count {
                children.push(previous.next()?.ok_or_else(|| {
                    ExternalLodBuildError::RunCorrupt(
                        "hierarchy summary ended inside a planned group".into(),
                    )
                })?);
            }
            let first = children[0];
            let last = *children.last().unwrap();
            let first_child = first.draft_index;
            if children
                .iter()
                .enumerate()
                .any(|(offset, child)| child.draft_index != first_child + offset)
            {
                return Err(ExternalLodBuildError::Validation(
                    "hierarchy child drafts are not contiguous".into(),
                ));
            }
            let mut bounds = first.bounds;
            for child in &children[1..] {
                bounds = bounds.union(child.bounds);
            }
            let source_end = last.source.end().ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("node source range overflow".into())
            })?;
            let source = LodSourceRange {
                start: first.source.start,
                count: source_end - first.source.start,
            };
            let morton = LodMortonRange {
                min: first.morton.min,
                max: last.morton.max,
            };
            let group_start = inputs.len() as u32;
            inputs.extend(children.iter().map(|child| GpuLodHierarchyReductionInput {
                representative: child.representative,
                bounds: child.bounds,
                inherited_error: child.reduction_error,
            }));
            groups.push(GpuLodHierarchyReductionGroup {
                start: group_start,
                count: child_count as u32,
            });
            pending.push(PendingGpuReduction {
                draft_index: None,
                children: Some((first_child, child_count_usize)),
                source,
                morton,
                bounds,
            });
        }
        flush_gpu_internal_batch(
            gpu,
            config,
            pages_directory,
            &mut inputs,
            &mut groups,
            &mut pending,
            &mut next,
            &mut drafts,
            &mut descriptors,
            &mut stored_gaussian_count,
            &mut maximum_encoded_page_bytes,
        )?;
        previous.finish()?;
        next.finish()?;
        levels.push((level_start, drafts.len()));
        fs::remove_file(&previous_path)?;
        previous_path = next_path;
        previous_count = next_count;
    }

    let mut root_reader = SummaryReader::open(&previous_path, 1, config.limits.run_buffer_bytes)?;
    let root_summary = root_reader
        .next()?
        .ok_or_else(|| ExternalLodBuildError::RunCorrupt("missing root summary".into()))?;
    root_reader.finish()?;
    fs::remove_file(previous_path)?;
    if root_summary.draft_index + 1 != drafts.len()
        || root_summary.source.start != 0
        || root_summary.source.count != source_count
    {
        return Err(ExternalLodBuildError::Validation(
            "GPU hierarchy root does not span the canonical source".into(),
        ));
    }
    if drafts.len() as u64 != plan.total_node_count {
        return Err(ExternalLodBuildError::Validation(format!(
            "built {} hierarchy nodes, planned {}",
            drafts.len(),
            plan.total_node_count
        )));
    }
    let manifest = finalize_manifest(
        drafts,
        descriptors,
        levels,
        source_count,
        stored_gaussian_count,
        source_fingerprint.finish(),
        config,
        EXTERNAL_GPU_LOD_BUILDER_ABI_VERSION,
    )?;
    Ok(HierarchyBuild {
        manifest,
        maximum_encoded_page_bytes,
        maximum_reduction_batch_records: effective_max_records as u64,
    })
}

fn effective_gpu_batch_records(configured: usize, gpu_limit: usize) -> usize {
    configured.min(gpu_limit)
}

fn gpu_batch_can_add(
    input_count: usize,
    group_count: usize,
    next_input_count: usize,
    leaf: bool,
    max_records: usize,
    max_nodes: usize,
) -> bool {
    let Some(next_inputs) = input_count.checked_add(next_input_count) else {
        return false;
    };
    let Some(next_groups) = group_count.checked_add(1) else {
        return false;
    };
    next_inputs <= max_records
        && next_groups <= max_records
        && if leaf {
            next_groups <= max_nodes
        } else {
            next_inputs
                .checked_add(next_groups)
                .is_some_and(|required| required <= max_nodes)
        }
}

fn validate_gpu_reduction_group_capacity(
    input_count: usize,
    leaf: bool,
    max_records: usize,
    max_nodes: usize,
) -> Result<(), ExternalLodBuildError> {
    if input_count > max_records {
        return Err(GpuLodHierarchyError::BatchTooLarge {
            actual: input_count,
            limit: u32::try_from(max_records).unwrap_or(u32::MAX),
        }
        .into());
    }
    let required_nodes = input_count.checked_add(usize::from(!leaf)).ok_or_else(|| {
        ExternalLodBuildError::InvalidConfig("GPU reduction node count overflow".into())
    })?;
    if required_nodes > max_nodes {
        return Err(GpuLodHierarchyError::NodeCapacityExceeded {
            required: u32::try_from(required_nodes).unwrap_or(u32::MAX),
            limit: u32::try_from(max_nodes).unwrap_or(u32::MAX),
        }
        .into());
    }
    Ok(())
}

fn flush_gpu_leaf_batch(
    gpu: &mut dyn ExternalLodGlobalReducer,
    support_sigma: f32,
    inputs: &mut Vec<Gaussian3d>,
    groups: &mut Vec<GpuLodHierarchyReductionGroup>,
    pending: &mut Vec<PendingGpuReduction>,
    writer: &mut SummaryWriter,
) -> Result<(), ExternalLodBuildError> {
    if groups.is_empty() {
        return Ok(());
    }
    let outputs = gpu.reduce_leaf_groups(inputs, groups, support_sigma)?;
    if outputs.len() != pending.len() {
        return Err(ExternalLodBuildError::PreprocessorContract(
            "GPU leaf reducer output length differs from its planned groups".into(),
        ));
    }
    for (pending, output) in pending.iter().copied().zip(outputs) {
        writer.write(&GpuReductionSummary {
            draft_index: pending.draft_index.ok_or_else(|| {
                ExternalLodBuildError::Validation("leaf reduction lost its draft index".into())
            })?,
            source: pending.source,
            morton: pending.morton,
            bounds: pending.bounds.union(output.representative_support),
            representative: output.representative,
            reduction_error: output.accumulated_error,
        })?;
    }
    inputs.clear();
    groups.clear();
    pending.clear();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn flush_gpu_internal_batch(
    gpu: &mut dyn ExternalLodGlobalReducer,
    config: ExternalLodBuildConfig,
    pages_directory: &Path,
    inputs: &mut Vec<GpuLodHierarchyReductionInput>,
    groups: &mut Vec<GpuLodHierarchyReductionGroup>,
    pending: &mut Vec<PendingGpuReduction>,
    writer: &mut SummaryWriter,
    drafts: &mut Vec<NodeDraft>,
    descriptors: &mut Vec<LodPageDescriptor>,
    stored_gaussian_count: &mut u64,
    maximum_encoded_page_bytes: &mut u64,
) -> Result<(), ExternalLodBuildError> {
    if groups.is_empty() {
        return Ok(());
    }
    let outputs = gpu.reduce_summary_groups(inputs, groups, config.settings.support_sigma)?;
    if outputs.len() != pending.len() {
        return Err(ExternalLodBuildError::PreprocessorContract(
            "GPU internal reducer output length differs from its planned groups".into(),
        ));
    }
    for (pending, output) in pending.iter().copied().zip(outputs) {
        append_gpu_internal_output(
            pending,
            output,
            config,
            pages_directory,
            writer,
            drafts,
            descriptors,
            stored_gaussian_count,
            maximum_encoded_page_bytes,
        )?;
    }
    inputs.clear();
    groups.clear();
    pending.clear();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_gpu_internal_output(
    pending: PendingGpuReduction,
    output: GpuLodHierarchyReductionOutput,
    config: ExternalLodBuildConfig,
    pages_directory: &Path,
    writer: &mut SummaryWriter,
    drafts: &mut Vec<NodeDraft>,
    descriptors: &mut Vec<LodPageDescriptor>,
    stored_gaussian_count: &mut u64,
    maximum_encoded_page_bytes: &mut u64,
) -> Result<(), ExternalLodBuildError> {
    let bounds = pending.bounds.union(output.representative_support);
    let page_id = LodPageId(drafts.len() as u64 + 1);
    let (descriptor, encoded_bytes, encoding_error) = write_page(
        pages_directory,
        page_id,
        LodPageKind::Representatives,
        vec![output.representative],
        config,
    )?;
    let accumulated_error = compose_hierarchical_error(output.accumulated_error, encoding_error)?;
    *maximum_encoded_page_bytes = (*maximum_encoded_page_bytes).max(encoded_bytes);
    *stored_gaussian_count = stored_gaussian_count
        .checked_add(u64::from(descriptor.gaussian_count))
        .ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("stored Gaussian count overflow".into())
        })?;
    let draft_index = drafts.len();
    drafts.push(NodeDraft {
        children: pending.children,
        source: pending.source,
        morton: pending.morton,
        bounds,
        page_id,
        page_count: descriptor.gaussian_count,
        error: accumulated_error,
    });
    descriptors.push(descriptor);
    writer.write(&GpuReductionSummary {
        draft_index,
        source: pending.source,
        morton: pending.morton,
        bounds,
        representative: output.representative,
        reduction_error: accumulated_error,
    })?;
    Ok(())
}

fn build_hierarchy_from_run(
    mut reader: FinalRunInput,
    source_count: u64,
    pages_directory: &Path,
    config: ExternalLodBuildConfig,
    plan: &ExternalLodBuildPlan,
) -> Result<HierarchyBuild, ExternalLodBuildError> {
    if reader.remaining() != source_count {
        return Err(ExternalLodBuildError::RunCorrupt(format!(
            "final run declares {} records, expected {source_count}",
            reader.remaining()
        )));
    }
    let planned_nodes = usize::try_from(plan.total_node_count).map_err(|_| {
        ExternalLodBuildError::InvalidConfig("planned node count exceeds usize".into())
    })?;
    let mut drafts = Vec::new();
    let mut descriptors = Vec::new();
    reserve_exact(&mut drafts, planned_nodes, "hierarchy drafts")?;
    reserve_exact(&mut descriptors, planned_nodes, "page descriptors")?;
    let mut levels = Vec::<(usize, usize)>::with_capacity(plan.hierarchy_level_counts.len());
    let mut stored_gaussian_count = 0_u64;
    let mut maximum_encoded_page_bytes = 0_u64;
    let mut source_fingerprint = StableHasher::new();
    source_fingerprint.write(&source_count.to_le_bytes());

    let leaf_count = plan.hierarchy_level_counts[0];
    let (leaf_base, leaf_remainder) = balanced_group_sizes(source_count, leaf_count);
    let leaf_level_start = drafts.len();
    let mut current = Vec::new();
    reserve_exact(
        &mut current,
        usize::try_from(leaf_count)
            .map_err(|_| ExternalLodBuildError::InvalidConfig("leaf count exceeds usize".into()))?,
        "leaf reduction summaries",
    )?;
    let mut canonical_start = 0_u64;
    for leaf_index in 0..leaf_count {
        let count = leaf_base + u64::from(leaf_index < leaf_remainder);
        let mut gaussians = Vec::new();
        reserve_exact(
            &mut gaussians,
            usize::try_from(count).map_err(|_| {
                ExternalLodBuildError::InvalidConfig("leaf count exceeds usize".into())
            })?,
            "leaf page records",
        )?;
        let mut morton_min = u64::MAX;
        let mut morton_max = 0_u64;
        let mut bounds: Option<LodBounds> = None;
        for _ in 0..count {
            let record = reader.next_record()?.ok_or_else(|| {
                ExternalLodBuildError::RunCorrupt("final run ended inside a leaf".into())
            })?;
            source_fingerprint.write(&record.morton.to_le_bytes());
            source_fingerprint.write(&stable_gaussian_hash(&record.gaussian).to_le_bytes());
            morton_min = morton_min.min(record.morton);
            morton_max = morton_max.max(record.morton);
            let support = gaussian_support_bounds(&record.gaussian, config.settings.support_sigma)?;
            bounds = Some(match bounds {
                Some(current) => current.union(support),
                None => support,
            });
            gaussians.push(record.gaussian);
        }
        let bounds = bounds.ok_or_else(|| {
            ExternalLodBuildError::RunCorrupt("balanced hierarchy emitted an empty leaf".into())
        })?;
        let (representative, reduction_error) = reduce_bounded(&gaussians, config)?;
        let page_id = LodPageId(drafts.len() as u64 + 1);
        let (descriptor, encoded_bytes, encoding_error) = write_page(
            pages_directory,
            page_id,
            LodPageKind::SourceLeaves,
            gaussians,
            config,
        )?;
        debug_assert_eq!(encoding_error, LodError::ZERO);
        maximum_encoded_page_bytes = maximum_encoded_page_bytes.max(encoded_bytes);
        stored_gaussian_count = stored_gaussian_count
            .checked_add(u64::from(descriptor.gaussian_count))
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("stored Gaussian count overflow".into())
            })?;
        let source = LodSourceRange {
            start: canonical_start,
            count,
        };
        let morton = LodMortonRange {
            min: morton_min,
            max: morton_max,
        };
        let draft_index = drafts.len();
        drafts.push(NodeDraft {
            children: None,
            source,
            morton,
            bounds,
            page_id,
            page_count: descriptor.gaussian_count,
            error: LodError::ZERO,
        });
        descriptors.push(descriptor);
        current.push(ReductionSummary {
            draft_index,
            source,
            morton,
            bounds,
            representative,
            reduction_error,
        });
        canonical_start = canonical_start.checked_add(count).ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("canonical source offset overflow".into())
        })?;
    }
    levels.push((leaf_level_start, drafts.len()));
    reader.finish()?;
    if canonical_start != source_count {
        return Err(ExternalLodBuildError::RunCorrupt(
            "leaf partition did not consume the canonical source".into(),
        ));
    }

    while current.len() > 1 {
        let level_start = drafts.len();
        let group_count = balanced_group_count(
            current.len() as u64,
            u64::from(config.settings.branching_factor),
            false,
        );
        let (base, remainder) = balanced_group_sizes(current.len() as u64, group_count);
        let mut next = Vec::new();
        reserve_exact(
            &mut next,
            usize::try_from(group_count).map_err(|_| {
                ExternalLodBuildError::InvalidConfig("hierarchy group count exceeds usize".into())
            })?,
            "hierarchy reduction summaries",
        )?;
        let mut child_offset = 0_usize;
        for group_index in 0..group_count {
            let child_count = (base + u64::from(group_index < remainder)) as usize;
            let children = &current[child_offset..child_offset + child_count];
            let mut reduction_input = Vec::new();
            reserve_exact(
                &mut reduction_input,
                children.len(),
                "hierarchy reducer input",
            )?;
            let mut bounds = children[0].bounds;
            let mut inherited_error = LodError::ZERO;
            for child in children {
                bounds = bounds.union(child.bounds);
                inherited_error = inherited_error.max(child.reduction_error);
                reduction_input.push(child.representative);
            }
            let (representative, local_error) = reduce_bounded(&reduction_input, config)?;
            let error = compose_hierarchical_error(inherited_error, local_error)?;
            bounds = bounds.union(gaussian_support_bounds(
                &representative,
                config.settings.support_sigma,
            )?);
            let first = &children[0];
            let last = children.last().unwrap();
            let source_end = last.source.end().ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("node source range overflow".into())
            })?;
            let source = LodSourceRange {
                start: first.source.start,
                count: source_end - first.source.start,
            };
            let morton = LodMortonRange {
                min: first.morton.min,
                max: last.morton.max,
            };
            let first_child = children[0].draft_index;
            if children
                .iter()
                .enumerate()
                .any(|(offset, child)| child.draft_index != first_child + offset)
            {
                return Err(ExternalLodBuildError::Validation(
                    "hierarchy child drafts are not contiguous".into(),
                ));
            }
            let page_id = LodPageId(drafts.len() as u64 + 1);
            let (descriptor, encoded_bytes, encoding_error) = write_page(
                pages_directory,
                page_id,
                LodPageKind::Representatives,
                vec![representative],
                config,
            )?;
            let error = compose_hierarchical_error(error, encoding_error)?;
            maximum_encoded_page_bytes = maximum_encoded_page_bytes.max(encoded_bytes);
            stored_gaussian_count = stored_gaussian_count
                .checked_add(u64::from(descriptor.gaussian_count))
                .ok_or_else(|| {
                    ExternalLodBuildError::InvalidConfig("stored Gaussian count overflow".into())
                })?;
            let draft_index = drafts.len();
            drafts.push(NodeDraft {
                children: Some((first_child, child_count)),
                source,
                morton,
                bounds,
                page_id,
                page_count: descriptor.gaussian_count,
                error,
            });
            descriptors.push(descriptor);
            next.push(ReductionSummary {
                draft_index,
                source,
                morton,
                bounds,
                representative,
                reduction_error: error,
            });
            child_offset += child_count;
        }
        if child_offset != current.len() {
            return Err(ExternalLodBuildError::Validation(
                "hierarchy grouping did not consume its child level".into(),
            ));
        }
        levels.push((level_start, drafts.len()));
        current = next;
    }

    if drafts.len() as u64 != plan.total_node_count {
        return Err(ExternalLodBuildError::Validation(format!(
            "built {} hierarchy nodes, planned {}",
            drafts.len(),
            plan.total_node_count
        )));
    }
    let manifest = finalize_manifest(
        drafts,
        descriptors,
        levels,
        source_count,
        stored_gaussian_count,
        source_fingerprint.finish(),
        config,
        EXTERNAL_LOD_BUILDER_ABI_VERSION,
    )?;
    Ok(HierarchyBuild {
        manifest,
        maximum_encoded_page_bytes,
        maximum_reduction_batch_records: plan.maximum_reducer_input_records,
    })
}

fn reduce_bounded(
    input: &[Gaussian3d],
    config: ExternalLodBuildConfig,
) -> Result<(Gaussian3d, LodError), ExternalLodBuildError> {
    let result = MomentMergeReducer::new(config.settings.support_sigma)
        .map_err(LodBuildError::InvalidSettings)?
        .reduce(input)?;
    Ok((result.gaussian, result.error))
}

/// A bounded external reducer sees child representatives rather than every
/// descendant source. Component-wise addition preserves a conservative path
/// bound (triangle inequality); `max` alone would hide accumulated levels.
fn compose_hierarchical_error(
    inherited: LodError,
    local: LodError,
) -> Result<LodError, ExternalLodBuildError> {
    let add = |left: f32, right: f32| {
        let value = f64::from(left) + f64::from(right);
        if value.is_finite() && value <= f32::MAX as f64 {
            Ok(value as f32)
        } else {
            Err(ExternalLodBuildError::Validation(
                "hierarchical reduction error overflowed f32".into(),
            ))
        }
    };
    let geometric = add(inherited.geometric, local.geometric)?;
    let appearance = add(inherited.appearance, local.appearance)?;
    let opacity = add(inherited.opacity, local.opacity)?;
    let combined = add(inherited.combined, local.combined)?
        .max(geometric)
        .max(appearance)
        .max(opacity);
    Ok(LodError {
        geometric,
        appearance,
        opacity,
        combined,
    })
}

fn write_page(
    directory: &Path,
    page_id: LodPageId,
    kind: LodPageKind,
    gaussians: Vec<Gaussian3d>,
    config: ExternalLodBuildConfig,
) -> Result<(LodPageDescriptor, u64, LodError), ExternalLodBuildError> {
    let page = PlanarGaussian3dPage::new(page_id, gaussians);
    let mut bounds: Option<LodBounds> = None;
    for gaussian in &page.gaussians {
        let support = gaussian_support_bounds(gaussian, config.settings.support_sigma)?;
        bounds = Some(match bounds {
            Some(current) => current.union(support),
            None => support,
        });
    }
    let bounds = bounds.ok_or_else(|| {
        ExternalLodBuildError::Validation("attempted to write an empty page".into())
    })?;
    let gaussian_count = u32::try_from(page.gaussians.len()).map_err(|_| {
        ExternalLodBuildError::InvalidConfig("page Gaussian count exceeds u32".into())
    })?;
    let encoding = match (kind, config.compressed_representative_sh_degree) {
        (LodPageKind::Representatives, Some(degree)) => LodPageEncoding::F16Sh { degree },
        _ => LodPageEncoding::F32Planar,
    };
    let encoded = encode_page_with_encoding(&page, encoding)?;
    enforce_limit(
        "encoded page bytes",
        encoded.len() as u64,
        config.limits.max_encoded_page_bytes,
    )?;
    let limits = LodCodecLimits {
        max_page_bytes: config.limits.max_encoded_page_bytes,
        max_page_gaussians: gaussian_count.max(1),
        ..LodCodecLimits::default()
    };
    let canonical_page = crate::io::lod::decode_page(&encoded, limits)?;
    let encoding_error = page_encoding_error(&page, &canonical_page)?;
    let mut descriptor = LodPageDescriptor {
        id: page_id,
        kind,
        encoding,
        gaussian_count,
        decoded_len: u64::from(gaussian_count)
            .checked_mul(size_of::<Gaussian3d>() as u64)
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("decoded page length overflow".into())
            })?,
        content_hash: canonical_page.content_hash(),
        bounds,
        storage: None,
    };
    let encoded_hash = checksum64(&encoded);
    let filename = format!("{encoded_hash:016x}.gspage");
    descriptor.storage = Some(LodPageStorage {
        uri: format!("pages/{filename}"),
        byte_range: None,
        encoded_len: encoded.len() as u64,
    });
    descriptor
        .validate()
        .map_err(|error| ExternalLodBuildError::Validation(error.to_string()))?;
    decode_page_with_descriptor(&encoded, &descriptor, limits)?;
    write_new_synced(&directory.join(filename), &encoded)?;
    Ok((descriptor, encoded.len() as u64, encoding_error))
}

fn page_encoding_error(
    original: &PlanarGaussian3dPage,
    decoded: &PlanarGaussian3dPage,
) -> Result<LodError, ExternalLodBuildError> {
    if original.gaussians.len() != decoded.gaussians.len() {
        return Err(ExternalLodBuildError::Validation(
            "encoded page changed its Gaussian count".into(),
        ));
    }
    let mut appearance = 0.0_f64;
    for (original, decoded) in original.gaussians.iter().zip(&decoded.gaussians) {
        let squared = original
            .spherical_harmonic
            .coefficients
            .iter()
            .zip(decoded.spherical_harmonic.coefficients.iter())
            .map(|(left, right)| {
                let difference = f64::from(*left) - f64::from(*right);
                difference * difference
            })
            .sum::<f64>();
        appearance = appearance.max((squared / SH_COEFF_COUNT.max(1) as f64).sqrt());
    }
    if !appearance.is_finite() || appearance > f64::from(f32::MAX) {
        return Err(ExternalLodBuildError::Validation(
            "representative page encoding error overflowed f32".into(),
        ));
    }
    let appearance = appearance as f32;
    Ok(LodError {
        geometric: 0.0,
        appearance,
        opacity: 0.0,
        combined: appearance,
    })
}

#[allow(clippy::too_many_arguments)]
fn finalize_manifest(
    drafts: Vec<NodeDraft>,
    descriptors: Vec<LodPageDescriptor>,
    levels: Vec<(usize, usize)>,
    source_count: u64,
    stored_gaussian_count: u64,
    source_fingerprint: u64,
    config: ExternalLodBuildConfig,
    builder_abi_version: u32,
) -> Result<GaussianLodManifest, ExternalLodBuildError> {
    let node_count = u32::try_from(drafts.len()).map_err(|_| {
        ExternalLodBuildError::InvalidConfig("manifest node count exceeds u32".into())
    })?;
    let max_depth = u16::try_from(levels.len() - 1)
        .map_err(|_| ExternalLodBuildError::InvalidConfig("hierarchy depth exceeds u16".into()))?;
    let mut old_to_new = Vec::new();
    reserve_exact(&mut old_to_new, drafts.len(), "hierarchy index remap")?;
    old_to_new.resize(drafts.len(), usize::MAX);
    let mut order = Vec::new();
    reserve_exact(&mut order, drafts.len(), "breadth-first node order")?;
    for &(start, end) in levels.iter().rev() {
        for (old_index, new_index) in old_to_new.iter_mut().enumerate().take(end).skip(start) {
            *new_index = order.len();
            order.push(old_index);
        }
    }
    let mut nodes = Vec::new();
    reserve_exact(&mut nodes, drafts.len(), "manifest nodes")?;
    for (new_index, old_index) in order.iter().copied().enumerate() {
        let draft = &drafts[old_index];
        let depth = levels
            .iter()
            .rev()
            .position(|(start, end)| (*start..*end).contains(&old_index))
            .ok_or_else(|| {
                ExternalLodBuildError::Validation("draft is outside hierarchy levels".into())
            })?;
        let depth = u16::try_from(depth).unwrap();
        let children = if let Some((old_start, count)) = draft.children {
            let new_start = old_to_new[old_start];
            if (0..count).any(|offset| old_to_new[old_start + offset] != new_start + offset) {
                return Err(ExternalLodBuildError::Validation(
                    "breadth-first child range is not contiguous".into(),
                ));
            }
            LodIndexRange {
                start: u32::try_from(new_start).map_err(|_| {
                    ExternalLodBuildError::InvalidConfig("child index exceeds u32".into())
                })?,
                count: u32::try_from(count).unwrap(),
            }
        } else {
            LodIndexRange::empty()
        };
        let is_leaf = children.is_empty();
        let quality = if max_depth == 0 {
            LodQualityInterval { min: 0.0, max: 1.0 }
        } else {
            LodQualityInterval {
                min: f32::from(depth) / f32::from(max_depth),
                max: if is_leaf {
                    1.0
                } else {
                    f32::from(depth + 1) / f32::from(max_depth)
                },
            }
        };
        nodes.push(GaussianLodNode {
            id: LodNodeId(new_index as u64 + 1),
            parent: None,
            depth,
            bounds: draft.bounds,
            children,
            source: draft.source,
            morton: draft.morton,
            representation: LodPageRange {
                page: draft.page_id,
                offset: 0,
                count: draft.page_count,
            },
            error: draft.error,
            quality,
            high_fidelity_certificate: if is_leaf { 1.0 } else { 0.0 },
        });
    }
    for parent_index in 0..nodes.len() {
        let children = nodes[parent_index].children;
        for child_index in children.start..children.end().unwrap() {
            nodes[child_index as usize].parent = Some(nodes[parent_index].id);
        }
    }
    let root = nodes.first().ok_or(ExternalLodBuildError::EmptySource)?;
    let root_id = root.id;
    let root_bounds = root.bounds;
    let root_representation_count = root.representation.count;
    let root_error = root.error;
    let compressed_representative_sh_degree = descriptors.iter().find_map(|page| {
        if let LodPageEncoding::F16Sh { degree } = page.encoding {
            Some(degree)
        } else {
            None
        }
    });
    let manifest = GaussianLodManifest {
        header: GaussianLodManifestHeader {
            magic: LOD_MANIFEST_MAGIC,
            manifest_version: LOD_MANIFEST_VERSION,
            page_schema_version: LOD_PAGE_SCHEMA_VERSION,
            required_features: LOD_CURRENT_REQUIRED_FEATURES,
            source_gaussian_count: source_count,
            stored_gaussian_count,
            node_count,
            page_count: node_count,
        },
        scene_bounds: Some(root_bounds),
        roots: vec![root_id],
        nodes,
        pages: descriptors,
        build: GaussianLodBuildMetadata {
            settings: config.settings,
            reducer: LodReducerKind::MomentMerge,
            builder_abi_version,
            reducer_version: MOMENT_MERGE_VERSION,
            source_fingerprint,
            config_fingerprint: lod_config_fingerprint(
                config.settings,
                compressed_representative_sh_degree,
            ),
        },
        quality: GaussianLodQualityMetadata {
            max_depth,
            coarsest_gaussian_count: u64::from(root_representation_count),
            finest_gaussian_count: source_count,
            max_error: root_error,
        },
    };
    manifest
        .validate()
        .map_err(|error| ExternalLodBuildError::Validation(error.to_string()))?;
    Ok(manifest)
}

fn balanced_group_count(len: u64, capacity: u64, force_multiple: bool) -> u64 {
    let mut count = len.div_ceil(capacity);
    if force_multiple && count == 1 {
        count = 2.min(len);
    }
    count
}

fn balanced_group_sizes(len: u64, group_count: u64) -> (u64, u64) {
    (len / group_count, len % group_count)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ShardPackReport {
    shard_count: u32,
    maximum_shard_bytes: u64,
}

/// Packs temporary standalone page objects into immutable shard files. The
/// single ordered reader thread and bounded sync channel overlap page reads
/// and verification with shard writes while imposing hard backpressure.
fn pack_staged_pages(
    pages_directory: &Path,
    descriptors: &mut [LodPageDescriptor],
    config: ExternalLodBuildConfig,
    maximum_encoded_page_bytes: u64,
) -> Result<ShardPackReport, ExternalLodBuildError> {
    if descriptors.is_empty() {
        return Err(ExternalLodBuildError::Validation(
            "cannot create empty page shards".into(),
        ));
    }
    let package_root = pages_directory.parent().ok_or_else(|| {
        ExternalLodBuildError::InvalidConfig("pages directory has no package root".into())
    })?;
    let raw_paths = descriptors
        .iter()
        .map(|descriptor| {
            let storage = descriptor.storage.as_ref().ok_or_else(|| {
                ExternalLodBuildError::Validation("page descriptor has no storage URI".into())
            })?;
            if storage.byte_range.is_some() {
                return Err(ExternalLodBuildError::Validation(
                    "builder page was already range-packed".into(),
                ));
            }
            safe_package_path(package_root, &storage.uri)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let groups = plan_page_shards(descriptors, config.limits)?;
    let codec_limits = LodCodecLimits {
        max_page_bytes: maximum_encoded_page_bytes.max(PAGE_CONTAINER_HEADER_BYTES),
        max_page_gaussians: descriptors
            .iter()
            .map(|page| page.gaussian_count)
            .max()
            .unwrap_or(1),
        ..LodCodecLimits::default()
    };
    let mut maximum_shard_bytes = 0_u64;

    for (shard_index, &(start, end, file_len)) in groups.iter().enumerate() {
        let shard_name = format!("shard-{shard_index:06}.bgslodpack");
        let uri = format!("pages/{shard_name}");
        let path = pages_directory.join(&shard_name);
        let entry_count = u32::try_from(end - start).map_err(|_| {
            ExternalLodBuildError::InvalidConfig("shard entry count exceeds u32".into())
        })?;
        let mut byte_offset = lod_shard_prefix_len(entry_count)?;
        let mut entries = Vec::with_capacity(end - start);
        for descriptor in &descriptors[start..end] {
            let encoded_len = descriptor
                .storage
                .as_ref()
                .expect("validated standalone storage")
                .encoded_len;
            entries.push(LodShardEntry {
                page_id: descriptor.id,
                byte_offset,
                encoded_len,
                content_hash: descriptor.content_hash,
            });
            byte_offset = byte_offset.checked_add(encoded_len).ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("shard size overflow".into())
            })?;
        }
        if byte_offset != file_len {
            return Err(ExternalLodBuildError::Validation(
                "planned shard length changed while packing".into(),
            ));
        }
        let prefix = encode_lod_shard_index(&LodShardIndex { file_len, entries })?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut writer = BufWriter::with_capacity(config.limits.run_buffer_bytes, file);
        writer.write_all(&prefix)?;

        let (sender, receiver) = std::sync::mpsc::sync_channel::<
            Result<(usize, Vec<u8>), ExternalLodBuildError>,
        >(config.limits.pipeline_depth);
        let mut pipeline_error = None;
        std::thread::scope(|scope| {
            let producer = scope.spawn(|| {
                for descriptor_index in start..end {
                    let result = read_standalone_page(
                        &raw_paths[descriptor_index],
                        &descriptors[descriptor_index],
                        codec_limits,
                    )
                    .map(|bytes| (descriptor_index, bytes));
                    if sender.send(result).is_err() {
                        return;
                    }
                }
            });
            for expected_index in start..end {
                match receiver.recv() {
                    Ok(Ok((actual_index, bytes))) if actual_index == expected_index => {
                        if pipeline_error.is_none()
                            && let Err(error) = writer.write_all(&bytes)
                        {
                            pipeline_error = Some(ExternalLodBuildError::Io(error));
                        }
                    }
                    Ok(Ok((actual_index, _))) => {
                        pipeline_error.get_or_insert_with(|| {
                            ExternalLodBuildError::Validation(format!(
                                "page pipeline reordered {actual_index} before {expected_index}"
                            ))
                        });
                    }
                    Ok(Err(error)) => {
                        if pipeline_error.is_none() {
                            pipeline_error = Some(error);
                        }
                    }
                    Err(_) => {
                        pipeline_error.get_or_insert_with(|| {
                            ExternalLodBuildError::Validation(
                                "page pipeline disconnected before completing a shard".into(),
                            )
                        });
                    }
                }
            }
            if producer.join().is_err() && pipeline_error.is_none() {
                pipeline_error = Some(ExternalLodBuildError::Validation(
                    "page pipeline reader panicked".into(),
                ));
            }
        });
        if let Some(error) = pipeline_error {
            return Err(error);
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        let actual_file_len = writer.get_ref().metadata()?.len();
        if actual_file_len != file_len {
            return Err(ExternalLodBuildError::Validation(format!(
                "shard '{}' has {actual_file_len} bytes, expected {file_len}",
                path.display()
            )));
        }
        for (descriptor, entry) in descriptors[start..end].iter_mut().zip(
            decode_lod_shard_index(&prefix, actual_file_len, config.limits.max_pages_per_shard)?
                .entries,
        ) {
            descriptor.storage = Some(LodPageStorage {
                uri: uri.clone(),
                byte_range: Some((entry.byte_offset, entry.encoded_len)),
                encoded_len: entry.encoded_len,
            });
        }
        maximum_shard_bytes = maximum_shard_bytes.max(actual_file_len);
    }

    for path in raw_paths {
        fs::remove_file(path)?;
    }
    sync_directory(pages_directory)?;
    Ok(ShardPackReport {
        shard_count: u32::try_from(groups.len())
            .map_err(|_| ExternalLodBuildError::InvalidConfig("shard count exceeds u32".into()))?,
        maximum_shard_bytes,
    })
}

fn plan_page_shards(
    descriptors: &[LodPageDescriptor],
    limits: ExternalLodBuildLimits,
) -> Result<Vec<(usize, usize, u64)>, ExternalLodBuildError> {
    let mut groups = Vec::new();
    let mut start = 0_usize;
    while start < descriptors.len() {
        let mut end = start;
        let mut payload_bytes = 0_u64;
        let mut file_len = 0_u64;
        while end < descriptors.len() && end - start < limits.max_pages_per_shard as usize {
            let encoded_len = descriptors[end]
                .storage
                .as_ref()
                .ok_or_else(|| {
                    ExternalLodBuildError::Validation("page descriptor has no storage URI".into())
                })?
                .encoded_len;
            let candidate_payload = payload_bytes.checked_add(encoded_len).ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("shard size overflow".into())
            })?;
            let candidate_count = u32::try_from(end - start + 1).map_err(|_| {
                ExternalLodBuildError::InvalidConfig("shard entry count exceeds u32".into())
            })?;
            let candidate_file_len = lod_shard_prefix_len(candidate_count)?
                .checked_add(candidate_payload)
                .ok_or_else(|| {
                    ExternalLodBuildError::InvalidConfig("shard size overflow".into())
                })?;
            if candidate_file_len > limits.max_shard_bytes {
                if end == start {
                    return Err(ExternalLodBuildError::LimitExceeded {
                        field: "single-page shard bytes",
                        actual: candidate_file_len,
                        limit: limits.max_shard_bytes,
                    });
                }
                break;
            }
            payload_bytes = candidate_payload;
            file_len = candidate_file_len;
            end += 1;
        }
        groups.push((start, end, file_len));
        start = end;
    }
    Ok(groups)
}

fn read_standalone_page(
    path: &Path,
    descriptor: &LodPageDescriptor,
    limits: LodCodecLimits,
) -> Result<Vec<u8>, ExternalLodBuildError> {
    let expected = descriptor
        .storage
        .as_ref()
        .ok_or_else(|| {
            ExternalLodBuildError::Validation("page descriptor has no storage URI".into())
        })?
        .encoded_len;
    let probe = expected
        .checked_add(1)
        .ok_or_else(|| ExternalLodBuildError::InvalidConfig("page probe overflow".into()))?;
    let capacity = usize::try_from(probe)
        .map_err(|_| ExternalLodBuildError::InvalidConfig("page length exceeds usize".into()))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|_| {
        ExternalLodBuildError::InvalidConfig("failed to reserve bounded page read".into())
    })?;
    File::open(path)?.take(probe).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != expected {
        return Err(ExternalLodBuildError::Validation(format!(
            "page '{}' has {} bytes, expected {expected}",
            path.display(),
            bytes.len()
        )));
    }
    decode_page_with_descriptor(&bytes, descriptor, limits)?;
    Ok(bytes)
}

fn package_codec_limits(
    manifest: &GaussianLodManifest,
    manifest_bytes: u64,
    maximum_page_bytes: u64,
) -> Result<LodCodecLimits, ExternalLodBuildError> {
    let limits = LodCodecLimits {
        max_manifest_bytes: manifest_bytes.max(40),
        max_nodes: manifest.header.node_count.max(1),
        max_pages: manifest.header.page_count.max(1),
        max_page_bytes: maximum_page_bytes.max(PAGE_CONTAINER_HEADER_BYTES),
        max_page_gaussians: manifest
            .pages
            .iter()
            .map(|page| page.gaussian_count)
            .max()
            .unwrap_or(1),
    };
    Ok(limits.validate()?)
}

fn verify_staged_package(
    root: &Path,
    expected: &GaussianLodManifest,
    limits: LodCodecLimits,
) -> Result<(), ExternalLodBuildError> {
    let manifest_bytes = fs::read(root.join("scene.gsplatlod"))?;
    let decoded = decode_manifest(&manifest_bytes, limits)?;
    if &decoded != expected {
        return Err(ExternalLodBuildError::Validation(
            "staged manifest differs from the built manifest".into(),
        ));
    }
    let mut shard_indices = HashMap::<String, LodShardIndex>::new();
    for descriptor in &decoded.pages {
        let storage = descriptor.storage.as_ref().ok_or_else(|| {
            ExternalLodBuildError::Validation("page descriptor has no storage URI".into())
        })?;
        let path = safe_package_path(root, &storage.uri)?;
        let encoded = if let Some((offset, len)) = storage.byte_range {
            if !shard_indices.contains_key(&storage.uri) {
                let index = read_lod_shard_index(&path, decoded.header.page_count.max(1))?;
                shard_indices.insert(storage.uri.clone(), index);
            }
            let index = &shard_indices[&storage.uri];
            let entry = index
                .entries
                .iter()
                .find(|entry| entry.page_id == descriptor.id)
                .ok_or_else(|| {
                    ExternalLodBuildError::Validation(format!(
                        "shard '{}' omits page {:?}",
                        path.display(),
                        descriptor.id
                    ))
                })?;
            if entry.byte_offset != offset
                || entry.encoded_len != len
                || entry.content_hash != descriptor.content_hash
            {
                return Err(ExternalLodBuildError::Validation(format!(
                    "shard '{}' range table disagrees with page {:?}",
                    path.display(),
                    descriptor.id
                )));
            }
            read_file_range(&path, offset, len)?
        } else {
            let encoded = fs::read(&path)?;
            if encoded.len() as u64 != storage.encoded_len {
                return Err(ExternalLodBuildError::Validation(format!(
                    "page '{}' changed length while staged",
                    path.display()
                )));
            }
            encoded
        };
        decode_page_with_descriptor(&encoded, descriptor, limits)?;
    }
    Ok(())
}

fn read_lod_shard_index(
    path: &Path,
    max_entries: u32,
) -> Result<LodShardIndex, ExternalLodBuildError> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut header = [0_u8; LOD_SHARD_HEADER_LEN];
    file.read_exact(&mut header)?;
    let entry_count = u32::from_le_bytes(header[12..16].try_into().unwrap());
    if entry_count == 0 || entry_count > max_entries {
        return Err(LodCodecError::ShardEntryLimitExceeded {
            actual: entry_count,
            limit: max_entries,
        }
        .into());
    }
    let prefix_len = lod_shard_prefix_len(entry_count)?;
    let prefix_len = usize::try_from(prefix_len).map_err(|_| {
        ExternalLodBuildError::InvalidConfig("shard prefix length exceeds usize".into())
    })?;
    let mut prefix = Vec::new();
    prefix.try_reserve_exact(prefix_len).map_err(|_| {
        ExternalLodBuildError::InvalidConfig("failed to reserve bounded shard table".into())
    })?;
    prefix.extend_from_slice(&header);
    prefix.resize(prefix_len, 0);
    file.read_exact(&mut prefix[LOD_SHARD_HEADER_LEN..])?;
    Ok(decode_lod_shard_index(&prefix, file_len, max_entries)?)
}

fn read_file_range(path: &Path, offset: u64, len: u64) -> Result<Vec<u8>, ExternalLodBuildError> {
    let capacity = usize::try_from(len).map_err(|_| {
        ExternalLodBuildError::InvalidConfig("page range length exceeds usize".into())
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|_| {
        ExternalLodBuildError::InvalidConfig("failed to reserve bounded page range".into())
    })?;
    let mut file = File::open(path)?;
    file.seek(io::SeekFrom::Start(offset))?;
    file.take(len).read_to_end(&mut bytes)?;
    if bytes.len() != capacity {
        return Err(ExternalLodBuildError::Validation(format!(
            "page range in '{}' is truncated to {} of {len} bytes",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes)
}

fn safe_package_path(root: &Path, uri: &str) -> Result<PathBuf, ExternalLodBuildError> {
    let relative = Path::new(uri);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(ExternalLodBuildError::Validation(format!(
            "page URI '{uri}' is not package relative"
        )));
    }
    Ok(root.join(relative))
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), ExternalLodBuildError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), ExternalLodBuildError> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn checksum64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn enforce_limit(
    field: &'static str,
    actual: u64,
    limit: u64,
) -> Result<(), ExternalLodBuildError> {
    if actual > limit {
        Err(ExternalLodBuildError::LimitExceeded {
            field,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn reserve_exact<T>(
    values: &mut Vec<T>,
    additional: usize,
    field: &'static str,
) -> Result<(), ExternalLodBuildError> {
    values.try_reserve_exact(additional).map_err(|error| {
        ExternalLodBuildError::InvalidConfig(format!(
            "could not reserve bounded {field} allocation: {error}"
        ))
    })
}

fn nonempty_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

struct StagingDirectory {
    path: PathBuf,
    published: bool,
}

impl StagingDirectory {
    fn new(output: &Path) -> Result<Self, ExternalLodBuildError> {
        let parent = nonempty_parent(output);
        fs::create_dir_all(parent)?;
        let name = output
            .file_name()
            .ok_or_else(|| ExternalLodBuildError::InvalidConfig("invalid output path".into()))?
            .to_string_lossy();
        for attempt in 0..1024_u32 {
            let candidate =
                parent.join(format!(".{name}.staging-{}-{attempt}", std::process::id()));
            match fs::create_dir(&candidate) {
                Ok(()) => {
                    return Ok(Self {
                        path: candidate,
                        published: false,
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(ExternalLodBuildError::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a unique staging directory",
        )))
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn publish(&mut self, output: &Path) -> Result<(), ExternalLodBuildError> {
        if output.exists() {
            return Err(ExternalLodBuildError::OutputExists(output.to_path_buf()));
        }
        fs::rename(&self.path, output)?;
        self.published = true;
        Ok(())
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[derive(Debug)]
pub enum ExternalLodBuildError {
    Io(io::Error),
    LodBuild(LodBuildError),
    Preprocess(LodPreprocessError),
    GpuHierarchy(GpuLodHierarchyError),
    Codec(LodCodecError),
    InvalidConfig(String),
    EmptySource,
    OutputExists(PathBuf),
    LimitExceeded {
        field: &'static str,
        actual: u64,
        limit: u64,
    },
    InvalidGaussian {
        source_index: u64,
        field: String,
    },
    InconsistentSource(String),
    PreprocessorContract(String),
    RunCorrupt(String),
    Validation(String),
}

impl fmt::Display for ExternalLodBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "external LoD I/O failed: {error}"),
            Self::LodBuild(error) => write!(formatter, "external LoD reduction failed: {error}"),
            Self::Preprocess(error) => {
                write!(formatter, "external LoD preprocessing failed: {error}")
            }
            Self::GpuHierarchy(error) => {
                write!(
                    formatter,
                    "external LoD GPU hierarchy construction failed: {error}"
                )
            }
            Self::Codec(error) => write!(formatter, "external LoD codec failed: {error}"),
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid external LoD config: {message}")
            }
            Self::EmptySource => {
                formatter.write_str("cannot build an external LoD package from an empty source")
            }
            Self::OutputExists(path) => write!(
                formatter,
                "output '{}' already exists; refusing to overwrite it",
                path.display()
            ),
            Self::LimitExceeded {
                field,
                actual,
                limit,
            } => write!(
                formatter,
                "external LoD {field} {actual} exceeds configured limit {limit}"
            ),
            Self::InvalidGaussian {
                source_index,
                field,
            } => write!(
                formatter,
                "source Gaussian {source_index} has invalid {field}"
            ),
            Self::InconsistentSource(message) => write!(
                formatter,
                "replayable source changed between passes: {message}"
            ),
            Self::PreprocessorContract(message) => {
                write!(formatter, "bounded preprocessor contract failed: {message}")
            }
            Self::RunCorrupt(message) => {
                write!(formatter, "external Morton run is corrupt: {message}")
            }
            Self::Validation(message) => write!(
                formatter,
                "external LoD package validation failed: {message}"
            ),
        }
    }
}

impl Error for ExternalLodBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::LodBuild(error) => Some(error),
            Self::Preprocess(error) => Some(error),
            Self::GpuHierarchy(error) => Some(error),
            Self::Codec(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for ExternalLodBuildError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<LodBuildError> for ExternalLodBuildError {
    fn from(error: LodBuildError) -> Self {
        Self::LodBuild(error)
    }
}

impl From<LodPreprocessError> for ExternalLodBuildError {
    fn from(error: LodPreprocessError) -> Self {
        Self::Preprocess(error)
    }
}

impl From<GpuLodHierarchyError> for ExternalLodBuildError {
    fn from(error: GpuLodHierarchyError) -> Self {
        Self::GpuHierarchy(error)
    }
}

impl From<LodCodecError> for ExternalLodBuildError {
    fn from(error: LodCodecError) -> Self {
        Self::Codec(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gaussian::f32::{PositionVisibility, Rotation, ScaleOpacity},
        material::spherical_harmonics::SphericalHarmonicCoefficients,
        stream::transport::{
            LodPageTransport, NativeFilePageTransport, PagePoll, PageRequest, PageRequestPriority,
        },
    };

    const GPU_F32_ABSOLUTE_TOLERANCE: f32 = 1e-5;
    const GPU_F32_RELATIVE_TOLERANCE: f32 = 1e-4;
    use std::{
        cell::Cell,
        sync::{Arc, Barrier},
    };

    #[derive(Clone)]
    struct MemorySource(Vec<Gaussian3d>);

    #[derive(Clone)]
    struct FragmentedSource(MemorySource);

    struct FailingPreprocessor;

    struct FakeGlobalPreprocessor {
        limits: ExternalLodGlobalReductionLimits,
        leaf_dispatches: usize,
        summary_dispatches: usize,
        fail_reduction: bool,
    }

    struct ChangingSource {
        source: MemorySource,
        replay_count: Cell<u32>,
    }

    impl ReplayableGaussianSource for MemorySource {
        fn replay(
            &self,
            batch_records: usize,
            consume: &mut dyn FnMut(&[Gaussian3d]) -> Result<(), ExternalLodBuildError>,
        ) -> Result<u64, ExternalLodBuildError> {
            for batch in self.0.chunks(batch_records) {
                consume(batch)?;
            }
            Ok(self.0.len() as u64)
        }
    }

    impl ReplayableGaussianSource for FragmentedSource {
        fn replay(
            &self,
            _batch_records: usize,
            consume: &mut dyn FnMut(&[Gaussian3d]) -> Result<(), ExternalLodBuildError>,
        ) -> Result<u64, ExternalLodBuildError> {
            // Deliberately ignore the requested target size while staying below
            // it. The builder must aggregate callbacks into its own fixed runs.
            for batch in self.0.0.chunks(3) {
                consume(batch)?;
            }
            Ok(self.0.0.len() as u64)
        }
    }

    impl ExternalLodBatchPreprocessor for FailingPreprocessor {
        fn stage_name(&self) -> &'static str {
            "injected-failure"
        }

        fn preprocess(
            &mut self,
            _records: &[Gaussian3d],
            _source_index_base: u64,
            _normalization_bounds: LodBounds,
            _support_sigma: f32,
        ) -> Result<LodPreprocessBatchOutput, ExternalLodBuildError> {
            Err(ExternalLodBuildError::PreprocessorContract(
                "injected after staging".into(),
            ))
        }
    }

    impl ExternalLodBatchPreprocessor for FakeGlobalPreprocessor {
        fn stage_name(&self) -> &'static str {
            "fake-canonical-sort"
        }

        fn preprocess(
            &mut self,
            records: &[Gaussian3d],
            source_index_base: u64,
            normalization_bounds: LodBounds,
            support_sigma: f32,
        ) -> Result<LodPreprocessBatchOutput, ExternalLodBuildError> {
            Ok(preprocess_lod_batch_cpu(
                records,
                source_index_base,
                normalization_bounds,
                support_sigma,
            )?)
        }

        fn global_gpu_hierarchy(&mut self) -> Option<&mut dyn ExternalLodGlobalReducer> {
            Some(self)
        }
    }

    impl ExternalLodGlobalReducer for FakeGlobalPreprocessor {
        fn limits(&self) -> ExternalLodGlobalReductionLimits {
            self.limits
        }

        fn reduce_leaf_groups(
            &mut self,
            records: &[Gaussian3d],
            groups: &[GpuLodHierarchyReductionGroup],
            support_sigma: f32,
        ) -> Result<Vec<GpuLodHierarchyReductionOutput>, ExternalLodBuildError> {
            self.leaf_dispatches += 1;
            if self.fail_reduction {
                return Err(ExternalLodBuildError::PreprocessorContract(
                    "injected global leaf reduction failure".into(),
                ));
            }
            let reducer =
                MomentMergeReducer::new(support_sigma).map_err(LodBuildError::InvalidSettings)?;
            groups
                .iter()
                .map(|group| {
                    let result = reducer.reduce(
                        &records[group.start as usize..(group.start + group.count) as usize],
                    )?;
                    Ok(GpuLodHierarchyReductionOutput {
                        representative: result.gaussian,
                        representative_support: gaussian_support_bounds(
                            &result.gaussian,
                            support_sigma,
                        )?,
                        local_error: result.error,
                        accumulated_error: result.error,
                    })
                })
                .collect()
        }

        fn reduce_summary_groups(
            &mut self,
            inputs: &[GpuLodHierarchyReductionInput],
            groups: &[GpuLodHierarchyReductionGroup],
            support_sigma: f32,
        ) -> Result<Vec<GpuLodHierarchyReductionOutput>, ExternalLodBuildError> {
            self.summary_dispatches += 1;
            if self.fail_reduction {
                return Err(ExternalLodBuildError::PreprocessorContract(
                    "injected global summary reduction failure".into(),
                ));
            }
            let reducer =
                MomentMergeReducer::new(support_sigma).map_err(LodBuildError::InvalidSettings)?;
            groups
                .iter()
                .map(|group| {
                    let children =
                        &inputs[group.start as usize..(group.start + group.count) as usize];
                    let representatives = children
                        .iter()
                        .map(|child| child.representative)
                        .collect::<Vec<_>>();
                    let result = reducer.reduce(&representatives)?;
                    let inherited = children.iter().fold(LodError::ZERO, |error, child| {
                        error.max(child.inherited_error)
                    });
                    Ok(GpuLodHierarchyReductionOutput {
                        representative: result.gaussian,
                        representative_support: gaussian_support_bounds(
                            &result.gaussian,
                            support_sigma,
                        )?,
                        local_error: result.error,
                        accumulated_error: compose_hierarchical_error(inherited, result.error)?,
                    })
                })
                .collect()
        }
    }

    impl ReplayableGaussianSource for ChangingSource {
        fn replay(
            &self,
            batch_records: usize,
            consume: &mut dyn FnMut(&[Gaussian3d]) -> Result<(), ExternalLodBuildError>,
        ) -> Result<u64, ExternalLodBuildError> {
            let replay = self.replay_count.get();
            self.replay_count.set(replay + 1);
            if replay == 0 {
                for batch in self.source.0.chunks(batch_records) {
                    consume(batch)?;
                }
            } else {
                let mut changed = self.source.0.clone();
                changed[0].position_visibility.visibility -= 0.01;
                for batch in changed.chunks(batch_records) {
                    consume(batch)?;
                }
            }
            Ok(self.source.0.len() as u64)
        }
    }

    fn fixture(count: usize) -> MemorySource {
        MemorySource(
            (0..count)
                .map(|index| {
                    let mut coefficients = [0.0; SH_COEFF_COUNT];
                    coefficients[0] = (index % 11) as f32 * 0.03 - 0.1;
                    Gaussian3d {
                        position_visibility: PositionVisibility {
                            position: [
                                (index % 9) as f32 - 4.0,
                                ((index / 9) % 7) as f32 - 3.0,
                                (index / 63) as f32 * 0.2,
                            ],
                            visibility: 1.0,
                        },
                        spherical_harmonic: SphericalHarmonicCoefficients { coefficients },
                        rotation: Rotation {
                            rotation: [1.0, 0.0, 0.0, 0.0],
                        },
                        scale_opacity: ScaleOpacity {
                            scale: [0.03 + index as f32 * 0.0001, 0.05, 0.08],
                            opacity: 0.2 + (index % 5) as f32 * 0.1,
                        },
                    }
                })
                .collect(),
        )
    }

    fn config(batch_records: usize) -> ExternalLodBuildConfig {
        ExternalLodBuildConfig {
            settings: GaussianLodBuildSettings {
                branching_factor: 4,
                leaf_capacity: 8,
                support_sigma: 3.0,
            },
            limits: ExternalLodBuildLimits {
                batch_records,
                merge_fan_in: 3,
                run_buffer_bytes: 512,
                max_merge_buffer_bytes: 4 * 512,
                max_source_count: 1_000,
                max_run_count: 1_000,
                max_temporary_bytes: 64 * 1024 * 1024,
                max_manifest_nodes: 1_000,
                max_manifest_bytes: 4 * 1024 * 1024,
                max_encoded_page_bytes: 1024 * 1024,
                max_shard_bytes: 2 * 1024 * 1024,
                max_pages_per_shard: 16,
                pipeline_depth: 2,
            },
            ..ExternalLodBuildConfig::default()
        }
    }

    fn temporary_output(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "bgs-external-lod-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    /// Reads exactly the page object named by a manifest descriptor. External
    /// builds range-pack pages into `.bgslodpack` shards, so reading the URI as
    /// a standalone page would feed the shard header to the page decoder.
    fn read_packaged_page_for_oracle(root: &Path, descriptor: &LodPageDescriptor) -> Vec<u8> {
        let storage = descriptor
            .storage
            .as_ref()
            .expect("oracle descriptor must name packaged storage");
        let path = root.join(&storage.uri);
        let bytes = match storage.byte_range {
            Some((offset, len)) => read_file_range(&path, offset, len)
                .expect("oracle descriptor range must be readable"),
            None => fs::read(&path).expect("oracle standalone page must be readable"),
        };
        assert_eq!(
            bytes.len() as u64,
            storage.encoded_len,
            "oracle page read must honor the descriptor's encoded length"
        );
        bytes
    }

    fn remove_if_present(path: &Path) {
        if path.exists() {
            fs::remove_dir_all(path).unwrap();
        }
    }

    fn assert_gpu_f32_close(label: &str, gpu: f32, cpu: f32) {
        let tolerance =
            GPU_F32_ABSOLUTE_TOLERANCE + GPU_F32_RELATIVE_TOLERANCE * gpu.abs().max(cpu.abs());
        assert!(
            (gpu - cpu).abs() <= tolerance,
            "{label}: GPU {gpu} differs from CPU {cpu} by {}, tolerance {tolerance}",
            (gpu - cpu).abs()
        );
    }

    fn assert_gpu_representative_close(gpu: &Gaussian3d, cpu: &Gaussian3d) {
        for axis in 0..3 {
            assert_gpu_f32_close(
                &format!("representative position[{axis}]"),
                gpu.position_visibility.position[axis],
                cpu.position_visibility.position[axis],
            );
        }
        assert_gpu_f32_close(
            "representative visibility",
            gpu.position_visibility.visibility,
            cpu.position_visibility.visibility,
        );
        for coefficient in 0..SH_COEFF_COUNT {
            assert_gpu_f32_close(
                &format!("representative SH[{coefficient}]"),
                gpu.spherical_harmonic.coefficients[coefficient],
                cpu.spherical_harmonic.coefficients[coefficient],
            );
        }
        for component in 0..4 {
            assert_gpu_f32_close(
                &format!("representative rotation[{component}]"),
                gpu.rotation.rotation[component],
                cpu.rotation.rotation[component],
            );
        }
        for axis in 0..3 {
            assert_gpu_f32_close(
                &format!("representative scale[{axis}]"),
                gpu.scale_opacity.scale[axis],
                cpu.scale_opacity.scale[axis],
            );
        }
        assert_gpu_f32_close(
            "representative opacity",
            gpu.scale_opacity.opacity,
            cpu.scale_opacity.opacity,
        );
    }

    fn assert_gpu_error_close(gpu: LodError, cpu: LodError) {
        assert_gpu_f32_close("geometric error", gpu.geometric, cpu.geometric);
        assert_gpu_f32_close("appearance error", gpu.appearance, cpu.appearance);
        assert_gpu_f32_close("opacity error", gpu.opacity, cpu.opacity);
        assert_gpu_f32_close("combined error", gpu.combined, cpu.combined);
    }

    #[test]
    fn planner_describes_more_than_one_hundred_million_without_source_allocation() {
        let source_count = 100_000_001;
        let plan = ExternalLodBuildPlan::new(source_count, ExternalLodBuildConfig::default())
            .expect("default external limits support a virtual 100M+ source");
        assert_eq!(plan.source_count, source_count);
        assert_eq!(plan.initial_run_count, 1_526);
        assert!(plan.merge_pass_count >= 2);
        assert!(plan.hierarchy_level_counts.len() < 16);
        assert!(plan.total_node_count < 200_000);
        assert_eq!(plan.maximum_records_per_batch_buffer, 65_536);
        assert_eq!(plan.merge_worker_limit, 2);
        assert!(plan.merge_stream_buffer_bytes <= 64 * 1024);
        assert!(plan.merge_buffer_bytes <= 4 * 1024 * 1024);
        assert!(plan.maximum_spill_host_bytes < 128 * 1024 * 1024);
        assert!(plan.maximum_merge_host_bytes < 8 * 1024 * 1024);
        assert!(plan.maximum_stream_handoff_host_bytes > 0);
        assert!(
            plan.maximum_merge_hierarchy_overlap_host_bytes
                > plan.maximum_stream_handoff_host_bytes
        );
        assert!(plan.maximum_temporary_run_bytes > source_count * RUN_RECORD_BYTES as u64);
        assert!(plan.maximum_temporary_summary_bytes > 0);
        assert_eq!(
            plan.maximum_temporary_bytes,
            plan.maximum_temporary_run_bytes + plan.maximum_temporary_summary_bytes
        );

        let mut disk_limited = ExternalLodBuildConfig::default();
        disk_limited.limits.max_temporary_bytes = plan.maximum_temporary_bytes - 1;
        assert!(matches!(
            ExternalLodBuildPlan::new(source_count, disk_limited),
            Err(ExternalLodBuildError::LimitExceeded {
                field: "temporary run and hierarchy-summary bytes",
                ..
            })
        ));
    }

    #[test]
    fn bounded_parallel_tasks_preserve_index_order_and_measure_real_overlap() {
        let first_pair = Arc::new(Barrier::new(2));
        let (values, intervals) = run_bounded_indexed_tasks(7, 2, {
            let first_pair = Arc::clone(&first_pair);
            move |index| {
                if index < 2 {
                    first_pair.wait();
                }
                std::thread::yield_now();
                Ok(index * 3)
            }
        })
        .unwrap();
        assert_eq!(values, (0..7).map(|index| index * 3).collect::<Vec<_>>());
        let (_, overlap, peak) = parallel_interval_stats(&intervals);
        assert_eq!(peak, 2);
        assert!(overlap > Duration::ZERO);
    }

    #[test]
    fn global_reduction_batch_respects_both_host_and_gpu_caps() {
        assert_eq!(effective_gpu_batch_records(65_536, 8_192), 8_192);
        assert_eq!(effective_gpu_batch_records(4_096, 65_536), 4_096);
        assert!(gpu_batch_can_add(3_000, 300, 1_000, false, 4_096, 4_500));
        assert!(!gpu_batch_can_add(3_500, 300, 700, false, 4_096, 4_500));
        assert!(!gpu_batch_can_add(4_000, 4, 97, true, 4_096, 4_500));
    }

    #[test]
    fn external_build_is_independent_of_batch_and_run_partitioning() {
        let source = fixture(97);
        let first_output = temporary_output("partition-a");
        let second_output = temporary_output("partition-b");
        remove_if_present(&first_output);
        remove_if_present(&second_output);
        let mut first_config = config(11);
        first_config.limits.max_merge_buffer_bytes = 8 * 512;
        first_config.limits.max_pages_per_shard = 3;
        let mut cpu = CpuExternalLodBatchPreprocessor;
        let first = build_external_lod_package(&source, &first_output, first_config, &mut cpu)
            .expect("first external build succeeds");
        let mut reversed = source.clone();
        reversed.0.reverse();
        let reversed = FragmentedSource(reversed);
        let mut cpu = CpuExternalLodBatchPreprocessor;
        let mut second_config = config(17);
        second_config.limits.merge_fan_in = 4;
        second_config.limits.max_merge_buffer_bytes = 5 * 512;
        second_config.limits.max_pages_per_shard = 3;
        second_config.limits.pipeline_depth = 1;
        let second = build_external_lod_package(&reversed, &second_output, second_config, &mut cpu)
            .expect("second external build succeeds");
        assert_eq!(first.initial_run_count, 9);
        assert_eq!(second.initial_run_count, 6);
        assert_eq!(first.merge_pass_count, 2);
        assert_eq!(second.merge_pass_count, 2);
        assert_eq!(first.pipeline_depth, 2);
        assert_eq!(second.pipeline_depth, 1);
        assert!(first.initial_run_count as usize > first.pipeline_depth);
        assert!(second.initial_run_count as usize > second.pipeline_depth);
        assert_eq!(first.merge_group_count, 4);
        // These fixture merges are intentionally tiny. The bounded scheduler
        // has two workers, but an OS may let one finish both groups before the
        // second worker starts. The barrier-backed scheduler test above proves
        // true overlap deterministically; this end-to-end test verifies that
        // its observed telemetry remains internally consistent.
        assert!((1..=2).contains(&first.maximum_concurrent_merge_groups));
        assert_eq!(second.merge_group_count, 3);
        assert_eq!(second.maximum_concurrent_merge_groups, 1);
        assert!(first.final_merge_streamed);
        assert!(second.final_merge_streamed);
        assert!(first.stage_timings.merge_group_work > Duration::ZERO);
        assert_eq!(
            first.stage_timings.merge_group_overlap > Duration::ZERO,
            first.maximum_concurrent_merge_groups > 1
        );
        assert!(first.stage_timings.final_merge_stream_work > Duration::ZERO);
        assert!(first.stage_timings.merge_hierarchy_overlap > Duration::ZERO);
        assert!(
            first.maximum_merge_host_bytes
                <= first_config.limits.max_merge_buffer_bytes as u64
                    + u64::from(first_config.limits.pipeline_depth as u32)
                        * first_config.limits.merge_fan_in as u64
                        * size_of::<RunRecord>() as u64
        );
        let limits = LodCodecLimits {
            max_manifest_bytes: 4 * 1024 * 1024,
            max_nodes: 1_000,
            max_pages: 1_000,
            max_page_bytes: 1024 * 1024,
            max_page_gaussians: 1_000,
        };
        let first_manifest = decode_manifest(
            &fs::read(first_output.join("scene.gsplatlod")).unwrap(),
            limits,
        )
        .unwrap();
        let second_manifest = decode_manifest(
            &fs::read(second_output.join("scene.gsplatlod")).unwrap(),
            limits,
        )
        .unwrap();
        assert!(first.initial_run_count > 1);
        assert!(first.page_count > first_config.limits.max_pages_per_shard);
        assert!(first.shard_count > 1);
        assert!(first.maximum_shard_bytes <= first_config.limits.max_shard_bytes);
        let measured_stages = first
            .stage_timings
            .scan
            .saturating_add(first.stage_timings.spill)
            .saturating_add(first.stage_timings.merge)
            .saturating_add(first.stage_timings.hierarchy_and_page_encode)
            .saturating_add(first.stage_timings.shard_pack)
            .saturating_add(first.stage_timings.validate_and_publish);
        assert!(first.stage_timings.total >= measured_stages);
        assert!(first_manifest.pages.iter().all(|descriptor| {
            descriptor
                .storage
                .as_ref()
                .is_some_and(|storage| storage.byte_range.is_some())
        }));
        let shard_uris = first_manifest
            .pages
            .iter()
            .map(|descriptor| descriptor.storage.as_ref().unwrap().uri.clone())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(shard_uris.len(), first.shard_count as usize);
        for descriptor in &first_manifest.pages {
            let storage = descriptor.storage.as_ref().unwrap();
            let (offset, len) = storage.byte_range.unwrap();
            let encoded = read_file_range(&first_output.join(&storage.uri), offset, len).unwrap();
            decode_page_with_descriptor(&encoded, descriptor, limits).unwrap();
        }
        let first_descriptor = &first_manifest.pages[0];
        let mut transport =
            NativeFilePageTransport::from_manifest(&first_output, &first_manifest).unwrap();
        let mut request = PageRequest::new(
            first_descriptor.id,
            PageRequestPriority::fallback_critical(u32::MAX),
        );
        request.expected_bytes = Some(first_descriptor.storage.as_ref().unwrap().encoded_len);
        let ticket = transport.begin(request).unwrap();
        let payload = (0..10_000)
            .find_map(|_| match transport.poll(&ticket) {
                PagePoll::Pending => {
                    std::thread::yield_now();
                    None
                }
                PagePoll::Ready(payload) => Some(payload),
                PagePoll::Failed(error) => panic!("packed native transport failed: {error}"),
            })
            .expect("packed native transport should complete");
        decode_page_with_descriptor(&payload.bytes, first_descriptor, limits).unwrap();
        assert!(
            fs::read_dir(first_output.join("pages"))
                .unwrap()
                .all(|entry| entry
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "bgslodpack"))
        );
        assert_eq!(first_manifest, second_manifest);
        assert_eq!(
            first_manifest.build.builder_abi_version,
            EXTERNAL_LOD_BUILDER_ABI_VERSION
        );
        remove_if_present(&first_output);
        remove_if_present(&second_output);
    }

    #[test]
    fn compressed_representatives_are_f16_but_source_leaves_remain_exact_f32() {
        let source = fixture(97);
        let output = temporary_output("compressed-sh");
        remove_if_present(&output);
        let mut build_config = config(11);
        build_config.compressed_representative_sh_degree =
            Some(crate::material::spherical_harmonics::SH_DEGREE.min(1) as u8);
        let mut cpu = CpuExternalLodBatchPreprocessor;
        let report = build_external_lod_package(&source, &output, build_config, &mut cpu)
            .expect("compressed representative package should build");
        assert!(report.initial_run_count > 1);
        let limits = LodCodecLimits {
            max_manifest_bytes: 4 * 1024 * 1024,
            max_nodes: 1_000,
            max_pages: 1_000,
            max_page_bytes: 1024 * 1024,
            max_page_gaussians: 1_000,
        };
        let manifest =
            decode_manifest(&fs::read(output.join("scene.gsplatlod")).unwrap(), limits).unwrap();
        let expected_degree = build_config.compressed_representative_sh_degree.unwrap();
        assert!(manifest.pages.iter().any(|page| matches!(
            page.encoding,
            LodPageEncoding::F16Sh { degree } if degree == expected_degree
        )));
        for descriptor in &manifest.pages {
            match descriptor.kind {
                LodPageKind::SourceLeaves => {
                    assert_eq!(descriptor.encoding, LodPageEncoding::F32Planar)
                }
                LodPageKind::Representatives => assert_eq!(
                    descriptor.encoding,
                    LodPageEncoding::F16Sh {
                        degree: expected_degree
                    }
                ),
                LodPageKind::Mixed => panic!("external builder does not emit mixed pages"),
            }
            let storage = descriptor.storage.as_ref().unwrap();
            let (offset, len) = storage.byte_range.unwrap();
            let encoded = read_file_range(&output.join(&storage.uri), offset, len).unwrap();
            let page = decode_page_with_descriptor(&encoded, descriptor, limits).unwrap();
            if matches!(descriptor.encoding, LodPageEncoding::F16Sh { .. }) {
                let retained = 3 * (usize::from(expected_degree) + 1).pow(2);
                assert!(page.gaussians.iter().all(|gaussian| {
                    gaussian.spherical_harmonic.coefficients[retained..]
                        .iter()
                        .all(|coefficient| *coefficient == 0.0)
                }));
            }
        }
        remove_if_present(&output);
    }

    #[test]
    fn topology_oracle_extracts_page_range_from_packed_shard() {
        let output = temporary_output("packed-oracle-range");
        remove_if_present(&output);
        let mut cpu = CpuExternalLodBatchPreprocessor;
        build_external_lod_package(&fixture(17), &output, config(8), &mut cpu)
            .expect("packed oracle fixture should build");
        let limits = LodCodecLimits {
            max_manifest_bytes: 4 * 1024 * 1024,
            max_nodes: 1_000,
            max_pages: 1_000,
            max_page_bytes: 1024 * 1024,
            max_page_gaussians: 1_000,
        };
        let manifest =
            decode_manifest(&fs::read(output.join("scene.gsplatlod")).unwrap(), limits).unwrap();
        let root = &manifest.nodes[0];
        assert!(!root.is_leaf());
        let descriptor = &manifest.pages[(root.representation.page.0 - 1) as usize];
        let storage = descriptor.storage.as_ref().unwrap();
        let (offset, len) = storage
            .byte_range
            .expect("external packages must range-pack representative pages");
        assert!(offset > 0);
        assert_eq!(len, storage.encoded_len);

        let shard = fs::read(output.join(&storage.uri)).unwrap();
        assert_eq!(&shard[..8], b"BGSSHARD");
        assert!(matches!(
            decode_page_with_descriptor(&shard, descriptor, limits),
            Err(LodCodecError::InvalidMagic("page"))
        ));

        let encoded = read_packaged_page_for_oracle(&output, descriptor);
        assert_eq!(&encoded[..8], b"BGSPAGE\0");
        let decoded = decode_page_with_descriptor(&encoded, descriptor, limits).unwrap();
        assert_eq!(decoded.id, root.representation.page);
        assert_eq!(decoded.gaussians.len(), 1);
        remove_if_present(&output);
    }

    #[test]
    fn fake_global_reducer_runs_after_merge_and_is_partition_invariant() {
        let source = fixture(97);
        let first_output = temporary_output("global-fake-a");
        let second_output = temporary_output("global-fake-b");
        remove_if_present(&first_output);
        remove_if_present(&second_output);
        let limits = ExternalLodGlobalReductionLimits {
            max_records: 19,
            max_nodes: 23,
        };
        let mut first_reducer = FakeGlobalPreprocessor {
            limits,
            leaf_dispatches: 0,
            summary_dispatches: 0,
            fail_reduction: false,
        };
        let first =
            build_external_lod_package(&source, &first_output, config(11), &mut first_reducer)
                .expect("fake global hierarchy succeeds");
        assert_eq!(first.preprocessing_stage, "fake-canonical-sort");
        assert_eq!(first.hierarchy_stage, "gpu-global-moment-merge-readback");
        assert!(first_reducer.leaf_dispatches > 1);
        assert!(first_reducer.summary_dispatches > 0);

        let mut reversed = source.clone();
        reversed.0.reverse();
        let mut second_config = config(17);
        second_config.limits.merge_fan_in = 4;
        second_config.limits.max_merge_buffer_bytes = 5 * 512;
        let mut second_reducer = FakeGlobalPreprocessor {
            limits,
            leaf_dispatches: 0,
            summary_dispatches: 0,
            fail_reduction: false,
        };
        let second = build_external_lod_package(
            &FragmentedSource(reversed),
            &second_output,
            second_config,
            &mut second_reducer,
        )
        .expect("repartitioned fake global hierarchy succeeds");
        assert_ne!(first.initial_run_count, second.initial_run_count);
        assert!(second_reducer.leaf_dispatches > 1);
        assert!(second_reducer.summary_dispatches > 0);

        let codec_limits = LodCodecLimits {
            max_manifest_bytes: 4 * 1024 * 1024,
            max_nodes: 1_000,
            max_pages: 1_000,
            max_page_bytes: 1024 * 1024,
            max_page_gaussians: 1_000,
        };
        let first_manifest = decode_manifest(
            &fs::read(first_output.join("scene.gsplatlod")).unwrap(),
            codec_limits,
        )
        .unwrap();
        let second_manifest = decode_manifest(
            &fs::read(second_output.join("scene.gsplatlod")).unwrap(),
            codec_limits,
        )
        .unwrap();
        assert_eq!(first_manifest, second_manifest);
        assert_eq!(
            first_manifest.build.builder_abi_version,
            EXTERNAL_GPU_LOD_BUILDER_ABI_VERSION
        );
        first_manifest.validate().unwrap();
        assert_eq!(first_manifest.nodes[0].source.start, 0);
        assert_eq!(first_manifest.nodes[0].source.count, source.0.len() as u64);
        for node in &first_manifest.nodes {
            if node.is_leaf() {
                assert!(node.source.count <= 8);
                assert_eq!(node.representation.count as u64, node.source.count);
            } else {
                assert_eq!(node.representation.count, 1);
                for child in node.children.start..node.children.end().unwrap() {
                    let child = &first_manifest.nodes[child as usize];
                    assert!(node.bounds.contains_with_epsilon(&child.bounds, 1e-5));
                    assert!(node.error.geometric >= child.error.geometric);
                    assert!(node.error.appearance >= child.error.appearance);
                    assert!(node.error.opacity >= child.error.opacity);
                    assert!(node.error.combined >= child.error.combined);
                }
            }
        }
        remove_if_present(&first_output);
        remove_if_present(&second_output);
    }

    #[test]
    fn global_reduction_failure_is_typed_and_never_publishes() {
        let output = temporary_output("global-reducer-failure");
        remove_if_present(&output);
        let mut reducer = FakeGlobalPreprocessor {
            limits: ExternalLodGlobalReductionLimits {
                max_records: 19,
                max_nodes: 23,
            },
            leaf_dispatches: 0,
            summary_dispatches: 0,
            fail_reduction: true,
        };
        let error = build_external_lod_package(&fixture(31), &output, config(11), &mut reducer)
            .expect_err("injected global reduction must fail");
        assert!(matches!(
            error,
            ExternalLodBuildError::PreprocessorContract(_)
        ));
        assert_eq!(reducer.leaf_dispatches, 1);
        assert!(!output.exists());
    }

    /// Opt in with:
    /// `RUN_GPU_LOD_HIERARCHY_TESTS=1 cargo test --features lod_build gpu_global_external_multi_run_matches_cpu_topology -- --ignored --nocapture`
    #[test]
    #[ignore = "requires an explicitly requested wgpu adapter"]
    fn gpu_global_external_multi_run_matches_cpu_topology() {
        if std::env::var("RUN_GPU_LOD_HIERARCHY_TESTS").as_deref() != Ok("1") {
            eprintln!("set RUN_GPU_LOD_HIERARCHY_TESTS=1 to execute the adapter test");
            return;
        }
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .expect("global external GPU test requires an adapter");
        eprintln!("global external GPU adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("global_external_lod_test_device"),
            ..Default::default()
        }))
        .expect("global external GPU test could not create a device");
        let source = fixture(97);
        let gpu_output = temporary_output("global-real-gpu");
        let cpu_output = temporary_output("global-real-cpu");
        remove_if_present(&gpu_output);
        remove_if_present(&cpu_output);
        let build_config = config(17);
        let mut builder = GpuLodHierarchyBuilder::new(
            &device,
            crate::gaussian::lod_build_gpu::hierarchy::GpuLodHierarchyLimits {
                max_records: 17,
                max_nodes: 64,
                max_input_bytes: 1024 * 1024,
                max_node_bytes: 1024 * 1024,
                max_readback_bytes: 4 * 1024 * 1024,
                ..Default::default()
            },
        )
        .unwrap();
        let mut gpu = GpuHierarchyExternalLodBatchPreprocessor {
            device: &device,
            queue: &queue,
            builder: &mut builder,
            settings: build_config.settings,
        };
        let gpu_report = build_external_lod_package(&source, &gpu_output, build_config, &mut gpu)
            .expect("multi-run global GPU package should build");
        assert!(gpu_report.initial_run_count > 1);
        assert_eq!(
            gpu_report.preprocessing_stage,
            "gpu-canonical-sort-readback"
        );
        assert_eq!(
            gpu_report.hierarchy_stage,
            "gpu-global-moment-merge-readback"
        );
        assert_eq!(gpu_report.maximum_global_reduction_batch_records, 17);

        let mut cpu = CpuExternalLodBatchPreprocessor;
        build_external_lod_package(&source, &cpu_output, build_config, &mut cpu)
            .expect("CPU topology oracle package should build");
        let codec_limits = LodCodecLimits {
            max_manifest_bytes: 4 * 1024 * 1024,
            max_nodes: 1_000,
            max_pages: 1_000,
            max_page_bytes: 1024 * 1024,
            max_page_gaussians: 1_000,
        };
        let gpu_manifest = decode_manifest(
            &fs::read(gpu_output.join("scene.gsplatlod")).unwrap(),
            codec_limits,
        )
        .unwrap();
        let cpu_manifest = decode_manifest(
            &fs::read(cpu_output.join("scene.gsplatlod")).unwrap(),
            codec_limits,
        )
        .unwrap();
        gpu_manifest.validate().unwrap();
        assert_eq!(
            gpu_manifest.build.builder_abi_version,
            EXTERNAL_GPU_LOD_BUILDER_ABI_VERSION
        );
        assert_eq!(
            gpu_manifest.header.node_count,
            cpu_manifest.header.node_count
        );
        assert_eq!(
            gpu_manifest.quality.max_depth,
            cpu_manifest.quality.max_depth
        );
        assert_eq!(
            gpu_manifest.build.source_fingerprint,
            cpu_manifest.build.source_fingerprint
        );
        for (gpu_node, cpu_node) in gpu_manifest.nodes.iter().zip(&cpu_manifest.nodes) {
            assert_eq!(gpu_node.source, cpu_node.source);
            assert_eq!(gpu_node.morton, cpu_node.morton);
            assert_eq!(gpu_node.children, cpu_node.children);
            if gpu_node.is_leaf() {
                assert_eq!(gpu_node.bounds, cpu_node.bounds);
                let gpu_page = &gpu_manifest.pages[(gpu_node.representation.page.0 - 1) as usize];
                let cpu_page = &cpu_manifest.pages[(cpu_node.representation.page.0 - 1) as usize];
                assert_eq!(gpu_page.content_hash, cpu_page.content_hash);
                assert_eq!(
                    read_packaged_page_for_oracle(&gpu_output, gpu_page),
                    read_packaged_page_for_oracle(&cpu_output, cpu_page)
                );
            } else {
                assert_gpu_error_close(gpu_node.error, cpu_node.error);
                let gpu_descriptor =
                    &gpu_manifest.pages[(gpu_node.representation.page.0 - 1) as usize];
                let cpu_descriptor =
                    &cpu_manifest.pages[(cpu_node.representation.page.0 - 1) as usize];
                assert_eq!(gpu_descriptor.kind, LodPageKind::Representatives);
                assert_eq!(cpu_descriptor.kind, LodPageKind::Representatives);
                assert_eq!(gpu_descriptor.gaussian_count, 1);
                assert_eq!(cpu_descriptor.gaussian_count, 1);
                let gpu_page = decode_page_with_descriptor(
                    &read_packaged_page_for_oracle(&gpu_output, gpu_descriptor),
                    gpu_descriptor,
                    codec_limits,
                )
                .unwrap();
                let cpu_page = decode_page_with_descriptor(
                    &read_packaged_page_for_oracle(&cpu_output, cpu_descriptor),
                    cpu_descriptor,
                    codec_limits,
                )
                .unwrap();
                assert_eq!(gpu_page.gaussians.len(), 1);
                assert_eq!(cpu_page.gaussians.len(), 1);
                assert_gpu_representative_close(&gpu_page.gaussians[0], &cpu_page.gaussians[0]);
                for child in gpu_node.children.start..gpu_node.children.end().unwrap() {
                    let child = &gpu_manifest.nodes[child as usize];
                    assert!(gpu_node.bounds.contains_with_epsilon(&child.bounds, 1e-5));
                    assert!(gpu_node.error.geometric >= child.error.geometric);
                    assert!(gpu_node.error.appearance >= child.error.appearance);
                    assert!(gpu_node.error.opacity >= child.error.opacity);
                    assert!(gpu_node.error.combined >= child.error.combined);
                }
            }
        }
        remove_if_present(&gpu_output);
        remove_if_present(&cpu_output);
    }

    #[test]
    fn failed_build_never_publishes_partial_output() {
        let output = temporary_output("atomic-failure");
        remove_if_present(&output);
        let source = fixture(8);
        let mut preprocessor = FailingPreprocessor;
        let error = build_external_lod_package(&source, &output, config(8), &mut preprocessor)
            .expect_err("injected staged build must fail");
        assert!(matches!(
            error,
            ExternalLodBuildError::PreprocessorContract(_)
        ));
        assert!(!output.exists());
        let output_name = output.file_name().unwrap().to_string_lossy();
        let staging_prefix = format!(".{output_name}.staging-");
        assert!(
            fs::read_dir(nonempty_parent(&output))
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| !entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(&staging_prefix)),
            "failed build leaked its private staging directory"
        );
    }

    #[test]
    fn source_replay_content_changes_are_rejected() {
        let output = temporary_output("changing-source");
        remove_if_present(&output);
        let source = ChangingSource {
            source: fixture(16),
            replay_count: Cell::new(0),
        };
        let mut cpu = CpuExternalLodBatchPreprocessor;
        let error = build_external_lod_package(&source, &output, config(8), &mut cpu)
            .expect_err("changed replay content must fail");
        assert!(matches!(
            error,
            ExternalLodBuildError::InconsistentSource(_)
        ));
        assert!(!output.exists());
    }
}
