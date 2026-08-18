//! Build an atomically published, external-memory Gaussian LoD package.

#[cfg(not(target_arch = "wasm32"))]
use std::{error::Error, io, path::PathBuf, process::ExitCode};

#[cfg(not(target_arch = "wasm32"))]
use bevy_gaussian_splatting::{
    gaussian::{
        formats::planar_3d_lod::GaussianLodBuildSettings,
        lod_build_gpu::hierarchy::{GpuLodHierarchyBuilder, GpuLodHierarchyLimits},
    },
    io::lod_build_external::{
        CpuExternalLodBatchPreprocessor, ExternalLodBuildConfig, ExternalLodBuildLimits,
        GpuHierarchyExternalLodBatchPreprocessor, PlyGaussianSource, build_external_lod_package,
    },
};
#[cfg(not(target_arch = "wasm32"))]
use clap::Parser;

#[cfg(not(target_arch = "wasm32"))]
type AnyError = Box<dyn Error + Send + Sync>;
#[cfg(not(target_arch = "wasm32"))]
type AnyResult<T> = Result<T, AnyError>;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Parser)]
#[command(
    name = "build_lod",
    about = "Build a bounded external-memory Gaussian LoD package",
    long_about = "Replays a PLY in bounded batches, creates deterministic Morton-sorted runs, performs bounded fan-in external merge, streams the canonical result into pages and a hierarchy, validates every artifact, and atomically publishes the completed package. --gpu-hierarchy performs per-run GPU canonical sort and then reduces the globally merged stream level-by-level on the GPU with bounded summary spills."
)]
struct Args {
    /// Input 3D Gaussian Splatting PLY. The file must be replayable.
    #[arg(short, long)]
    input: PathBuf,

    /// New package directory. It must not already exist.
    #[arg(short, long)]
    output: PathBuf,

    /// Maximum children per hierarchy node (2..=32).
    #[arg(long, default_value_t = 8)]
    branching: u8,

    /// Maximum original Gaussians in one leaf/page working set.
    #[arg(long, default_value_t = 1024)]
    leaf_capacity: u32,

    /// Truncated support radius in standard deviations.
    #[arg(long, default_value_t = 3.0)]
    support_sigma: f32,

    /// Maximum source records in one PLY/preprocess/sort allocation.
    #[arg(long, default_value_t = 65_536)]
    batch_size: usize,

    /// Maximum sorted runs opened by one external merge operation.
    #[arg(long, default_value_t = 32)]
    merge_fan_in: usize,

    /// Bytes in each merge reader and writer buffer.
    #[arg(long, default_value_t = 64 * 1024)]
    run_buffer_bytes: usize,

    /// Aggregate hard bound for merge reader/writer buffers.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    max_merge_buffer_bytes: usize,

    /// Hard maximum logical source count.
    #[arg(long, default_value_t = 250_000_000)]
    max_source_count: u64,

    /// Hard maximum number of initial spill runs.
    #[arg(long, default_value_t = 1_000_000)]
    max_run_count: u64,

    /// Hard peak byte limit for owned spill/merge files (final pages excluded).
    #[arg(long, default_value_t = 256 * 1024 * 1024 * 1024)]
    max_temporary_bytes: u64,

    /// Hard maximum hierarchy/page descriptors retained for the manifest.
    #[arg(long, default_value_t = 2_000_000)]
    max_manifest_nodes: u32,

    /// Hard encoded manifest byte limit.
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    max_manifest_bytes: u64,

    /// Hard encoded byte limit for any independently verified page.
    #[arg(long, default_value_t = 256 * 1024 * 1024)]
    max_page_bytes: u64,

    /// Hard maximum bytes in one immutable page shard, including its table.
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    max_shard_bytes: u64,

    /// Hard maximum pages in one shard range table.
    #[arg(long, default_value_t = 4096)]
    max_pages_per_shard: u32,

    /// Bounded in-flight page reads while the shard writer drains payloads.
    #[arg(long, default_value_t = 2)]
    pipeline_depth: usize,

    /// Store representative-page SH through this degree as binary16. Source
    /// leaves remain full-degree f32, preserving the exact finest cut.
    #[arg(long)]
    coarse_sh_degree: Option<u8>,

    /// GPU-sort bounded runs, then GPU-reduce the globally merged hierarchy.
    /// External I/O, exact topology, page encoding, fsync, and publication stay on CPU.
    #[arg(long)]
    gpu_hierarchy: bool,

    /// GPU hierarchy input-buffer safety bound; allocation remains batch-sized.
    #[arg(long, default_value_t = 256 * 1024 * 1024)]
    gpu_max_input_bytes: u64,

    /// GPU hierarchy node-buffer safety bound; allocation remains batch-sized.
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    gpu_max_node_bytes: u64,

    /// GPU input-plus-output summary capacity per global reduction batch.
    #[arg(long, default_value_t = 131_072)]
    gpu_max_hierarchy_nodes: u32,

    /// GPU sort command capacity per bounded run.
    #[arg(long, default_value_t = 1024)]
    gpu_max_hierarchy_commands: u32,

    /// GPU sort/reduction readback safety bound per in-flight slot.
    #[arg(long, default_value_t = 256 * 1024 * 1024)]
    gpu_max_hierarchy_readback_bytes: u64,
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("build_lod is available only on native targets");
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("build_lod: {error}");
            let mut source = error.source();
            while let Some(error) = source {
                eprintln!("  caused by: {error}");
                source = error.source();
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run(args: Args) -> AnyResult<()> {
    if !args.input.is_file() {
        return Err(invalid_input(format!(
            "input '{}' is not a regular file",
            args.input.display()
        )));
    }
    let config = ExternalLodBuildConfig {
        settings: GaussianLodBuildSettings {
            branching_factor: args.branching,
            leaf_capacity: args.leaf_capacity,
            support_sigma: args.support_sigma,
        },
        compressed_representative_sh_degree: args.coarse_sh_degree,
        limits: ExternalLodBuildLimits {
            batch_records: args.batch_size,
            merge_fan_in: args.merge_fan_in,
            run_buffer_bytes: args.run_buffer_bytes,
            max_merge_buffer_bytes: args.max_merge_buffer_bytes,
            max_source_count: args.max_source_count,
            max_run_count: args.max_run_count,
            max_temporary_bytes: args.max_temporary_bytes,
            max_manifest_nodes: args.max_manifest_nodes,
            max_manifest_bytes: args.max_manifest_bytes,
            max_encoded_page_bytes: args.max_page_bytes,
            max_shard_bytes: args.max_shard_bytes,
            max_pages_per_shard: args.max_pages_per_shard,
            pipeline_depth: args.pipeline_depth,
        },
    }
    .validate()?;
    let source = PlyGaussianSource::new(&args.input);

    let report = if args.gpu_hierarchy {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .map_err(|error| {
            invalid_input(format!(
                "--gpu-hierarchy requested but no adapter is available: {error}"
            ))
        })?;
        eprintln!("GPU hierarchy adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("external_gaussian_lod_hierarchy_builder"),
                ..Default::default()
            }))?;
        let limits = GpuLodHierarchyLimits {
            max_records: u32::try_from(args.batch_size)
                .map_err(|_| invalid_input("--batch-size exceeds the GPU u32 record limit"))?,
            max_nodes: args.gpu_max_hierarchy_nodes,
            max_stage_commands: args.gpu_max_hierarchy_commands,
            max_input_bytes: args.gpu_max_input_bytes,
            max_node_bytes: args.gpu_max_node_bytes,
            max_readback_bytes: args.gpu_max_hierarchy_readback_bytes,
            ..GpuLodHierarchyLimits::default()
        };
        let mut builder = GpuLodHierarchyBuilder::new(&device, limits)?;
        let mut preprocessor = GpuHierarchyExternalLodBatchPreprocessor {
            device: &device,
            queue: &queue,
            builder: &mut builder,
            settings: config.settings,
        };
        build_external_lod_package(&source, &args.output, config, &mut preprocessor)?
    } else {
        let mut preprocessor = CpuExternalLodBatchPreprocessor;
        build_external_lod_package(&source, &args.output, config, &mut preprocessor)?
    };

    eprintln!(
        "published '{}' via {} + {}: {} source Gaussians, {} initial runs/{} merge passes, {} nodes, {} pages in {} shards, {} stored records, largest encoded page {} bytes, largest shard {} bytes",
        args.output.display(),
        report.preprocessing_stage,
        report.hierarchy_stage,
        report.source_count,
        report.initial_run_count,
        report.merge_pass_count,
        report.node_count,
        report.page_count,
        report.shard_count,
        report.stored_gaussian_count,
        report.maximum_encoded_page_bytes,
        report.maximum_shard_bytes,
    );
    eprintln!(
        "bounded working sets: spill host <= {} bytes, parallel merge host <= {} bytes, streamed merge/hierarchy handoff <= {} bytes, overlapping final merge+handoff <= {} bytes, reducer group <= {} records, global reduction batch <= {} records, page <= {} records, hierarchy level <= {} summaries, temporary runs <= {} bytes, summary spills <= {} bytes, aggregate temporary <= {} bytes",
        report.maximum_spill_host_bytes,
        report.maximum_merge_host_bytes,
        report.maximum_stream_handoff_host_bytes,
        report.maximum_merge_hierarchy_overlap_host_bytes,
        report.maximum_reducer_input_records,
        report.maximum_global_reduction_batch_records,
        report.maximum_page_records,
        report.maximum_hierarchy_level_summaries,
        report.maximum_temporary_run_bytes,
        report.maximum_temporary_summary_bytes,
        report.maximum_temporary_bytes,
    );
    eprintln!(
        "stage timings: scan {:?}, read/preprocess/sort/spill {:?}, barrier merge {:?} (group work {:?}, true overlap {:?}), final streamed merge work {:?} (backpressure {:?}, hierarchy overlap {:?}), hierarchy/encode {:?}, shard pack {:?}, verify/publish {:?}, total {:?}; {} merge groups, peak {} concurrent, min stream buffer {} bytes, handoff {} records x {} queued batches, streamed final {}; bounded pipeline depth {}",
        report.stage_timings.scan,
        report.stage_timings.spill,
        report.stage_timings.merge,
        report.stage_timings.merge_group_work,
        report.stage_timings.merge_group_overlap,
        report.stage_timings.final_merge_stream_work,
        report.stage_timings.final_merge_backpressure,
        report.stage_timings.merge_hierarchy_overlap,
        report.stage_timings.hierarchy_and_page_encode,
        report.stage_timings.shard_pack,
        report.stage_timings.validate_and_publish,
        report.stage_timings.total,
        report.merge_group_count,
        report.maximum_concurrent_merge_groups,
        report.minimum_merge_stream_buffer_bytes,
        report.stream_handoff_chunk_records,
        report.stream_handoff_capacity_batches,
        report.final_merge_streamed,
        report.pipeline_depth,
    );
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn invalid_input(message: impl Into<String>) -> AnyError {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}
