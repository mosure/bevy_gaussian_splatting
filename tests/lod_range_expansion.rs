//! GPU parity for the production descriptor lookup, including heavily
//! fragmented physical pages. The shader functions below are taken directly
//! from lod_compaction.wgsl; only the Gaussian evaluation is replaced by a
//! source-ID readback so every mapping can be checked independently.
use std::{sync::mpsc, time::Duration};

use wgpu::util::DeviceExt;

fn shader() -> String {
    lookup_shader(include_str!("../src/render/lod_compaction.wgsl"))
}

fn lookup_shader(source: &str) -> String {
    let windowed = source.contains("fn prepare_candidate_range_window(");
    let start = source
        .find(if windowed {
            "fn range_upper_bound("
        } else {
            "fn candidate_from_physical_ranges("
        })
        .expect("production descriptor lookup must exist");
    let end = source.find("\nfn scan_record_count(").unwrap();
    let prepare = if windowed {
        "prepare_candidate_range_window(group.x * 256u, lane);"
    } else {
        ""
    };
    format!(
        "{}\n{}\n{}",
        r#"
struct Config { candidate_count: u32, candidate_range_count: u32, pad0: u32, pad1: u32 }
struct LodCandidateSource { index: u32, residency: u32, presentation_class: u32 }
@group(0) @binding(0) var<uniform> lod_config: Config;
@group(0) @binding(1) var<storage, read> candidate_and_scan_words: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<u32>>;
var<workgroup> range_window_low: u32;
var<workgroup> range_window_high: u32;
"#,
        &source[start..end],
        r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
    PREPARE_RANGE_WINDOW
    if id.x < lod_config.candidate_count {
        let mapped = candidate_from_physical_ranges(id.x);
        output[id.x] = vec4(mapped.index, mapped.residency, mapped.presentation_class, id.x);
    }
}
"#
        .replace("PREPARE_RANGE_WINDOW", prepare)
    )
}

#[test]
fn physical_range_expansion_matches_source_membership() {
    if std::env::var("RUN_GPU_RENDER_TESTS").as_deref() != Ok("1") {
        eprintln!("set RUN_GPU_RENDER_TESTS=1 to execute descriptor expansion on a GPU");
        return;
    }
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .expect("range expansion test requires a GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("lod_range_expansion_parity"),
        ..Default::default()
    }))
    .unwrap();
    eprintln!("range expansion adapter: {:?}", adapter.get_info());
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production_range_lookup"),
        source: wgpu::ShaderSource::Wgsl(shader().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("range_lookup_parity"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    // Small and exact workgroup boundaries, a large page spanning many groups,
    // and 65K one-record ranges. Physical order intentionally differs from
    // descriptor order, with all residency/presentation bit combinations.
    for (count, page_size) in [
        (0, 1),
        (1, 1),
        (255, 17),
        (256, 256),
        (257, 256),
        (65_537, 1),
        (1_048_577, 1024),
        (1_048_577, 257),
    ] {
        let (ranges, expected) = mapping_fixture(count, page_size);
        dispatch(&device, &queue, &pipeline, count, &ranges, &expected);
    }
    // Descriptor corruption is rejected locally rather than aliasing the
    // preceding physical page. Host validation is still authoritative.
    let ranges = [[5, 200, 4, 3], [258, 600, 2, 12]];
    let expected = (0..513)
        .map(|id| match id {
            5..=8 => [200 + id - 5, 3, 0, id],
            258..=259 => [600 + id - 258, 0, 3, id],
            _ => [u32::MAX, 0, 0, id],
        })
        .collect::<Vec<_>>();
    dispatch(&device, &queue, &pipeline, 513, &ranges, &expected);
}

fn mapping_fixture(count: u32, page_size: u32) -> (Vec<[u32; 4]>, Vec<[u32; 4]>) {
    assert!(page_size > 0);
    let mut ranges = Vec::new();
    let mut expected = Vec::with_capacity(count as usize);
    let mut start = 0_u32;
    while start < count {
        let len = page_size.min(count - start);
        let ordinal = ranges.len() as u32;
        let physical = (count - start) * 2;
        let metadata = ordinal % 16;
        ranges.push([start, physical, len, metadata]);
        expected.extend((0..len).map(|offset| {
            [
                physical + offset,
                metadata & 3,
                (metadata >> 2) & 3,
                start + offset,
            ]
        }));
        start += len;
    }
    (ranges, expected)
}

fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    count: u32,
    ranges: &[[u32; 4]],
    expected: &[[u32; 4]],
) {
    let params = [count, ranges.len() as u32, 0, 0];
    let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("range_config"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let empty = [[0_u32; 4]];
    let descriptors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("physical_ranges"),
        contents: bytemuck::cast_slice(if ranges.is_empty() { &empty } else { ranges }),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = u64::from(count.max(1)) * 16;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mapped_sources"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mapped_sources_readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("range_lookup_bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: config.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: descriptors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let mapped = readback.slice(..).get_mapped_range();
    let actual: &[[u32; 4]] = bytemuck::cast_slice(&mapped);
    assert_eq!(
        &actual[..count as usize],
        expected,
        "count={count}, ranges={}",
        ranges.len()
    );
    drop(mapped);
    readback.unmap();
}

/// An isolated lookup-and-output-write microbenchmark, not a frame benchmark.
///
/// RUN_LOD_RANGE_BENCHMARK=1 opts in. LOD_RANGE_BASELINE_WGSL must name the
/// preserved pre-window production shader; LOD_RANGE_BENCHMARK_OUTPUT must name
/// a new JSON file. No adapter/timestamp fallback is allowed after opting in.
#[cfg(all(feature = "lod", not(target_arch = "wasm32")))]
#[test]
fn physical_range_lookup_gpu_timestamp_abba() {
    use sha2::{Digest, Sha256};
    use std::{
        fs::OpenOptions,
        io::{Seek, SeekFrom, Write},
    };

    if std::env::var("RUN_LOD_RANGE_BENCHMARK").as_deref() != Ok("1") {
        eprintln!(
            "set RUN_LOD_RANGE_BENCHMARK=1 and the baseline/output paths to run GPU timestamps"
        );
        return;
    }
    let baseline_path = std::env::var("LOD_RANGE_BASELINE_WGSL")
        .expect("LOD_RANGE_BASELINE_WGSL must name the preserved production WGSL");
    let output_path = std::env::var("LOD_RANGE_BENCHMARK_OUTPUT")
        .expect("LOD_RANGE_BENCHMARK_OUTPUT must name a new JSON file");
    let baseline_source = std::fs::read_to_string(&baseline_path).unwrap();
    // An archived rejected candidate can be reproduced without installing it
    // in the production renderer. Default comparison uses the compiled source.
    let candidate_path = std::env::var("LOD_RANGE_CANDIDATE_WGSL").ok();
    let candidate_source = candidate_path
        .as_ref()
        .map(|path| std::fs::read_to_string(path).unwrap());
    let current_source = candidate_source
        .as_deref()
        .unwrap_or(include_str!("../src/render/lod_compaction.wgsl"));
    let sources = [
        lookup_shader(&baseline_source),
        lookup_shader(current_source),
    ];
    let sha256 = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let identities = serde_json::json!({
        "baseline": {
            "path": std::fs::canonicalize(&baseline_path).unwrap(),
            "production_wgsl_sha256": sha256(baseline_source.as_bytes()),
            "extracted_lookup_test_wgsl_sha256": sha256(sources[0].as_bytes()),
        },
        "current": {
            "path": candidate_path.as_deref().unwrap_or("src/render/lod_compaction.wgsl (compiled include_str)"),
            "production_wgsl_sha256": sha256(current_source.as_bytes()),
            "extracted_lookup_test_wgsl_sha256": sha256(sources[1].as_bytes()),
        },
    });
    // Reserve the output before creating a device. A failed run leaves an
    // explicit incomplete artifact; never spend the GPU budget then overwrite.
    let mut output_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
        .expect("benchmark output must not already exist");
    serde_json::to_writer_pretty(
        &mut output_file,
        &serde_json::json!({
            "schema_version": 1, "status": "incomplete", "identities": identities,
        }),
    )
    .unwrap();
    output_file.flush().unwrap();

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .expect("opt-in lookup benchmark requires a GPU adapter");
    assert!(
        adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        "opt-in benchmark requires GPU timestamps; CPU timing is not a substitute"
    );
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("lod_range_lookup_timestamp_benchmark"),
        required_features: wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    }))
    .unwrap();
    let timestamp_period_ns = f64::from(queue.get_timestamp_period());
    assert!(timestamp_period_ns.is_finite() && timestamp_period_ns > 0.0);
    let pipelines = sources.map(|source| {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("exact_production_lookup_functions"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("range_lookup_timestamp_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    });
    let mut results = Vec::new();
    for (name, count, page_size) in [
        ("million_four_ranges", 1_048_576, 262_144),
        ("million_pages_1024", 1_048_577, 1024),
        ("million_pages_257", 1_048_577, 257),
        ("fragmented_singletons", 65_537, 1),
    ] {
        let (ranges, expected) = mapping_fixture(count, page_size);
        // Exact membership, flags and source offsets are validated on both
        // actual kernels outside timestamp intervals. Neither counts nor an
        // aggregate hash can hide a missing or duplicated member.
        for pipeline in &pipelines {
            dispatch(&device, &queue, pipeline, count, &ranges, &expected);
        }
        let mut result = timestamp_case(
            &device,
            &queue,
            &pipelines,
            count,
            &ranges,
            timestamp_period_ns,
        );
        result["name"] = name.into();
        result["page_size"] = page_size.into();
        result["membership_oracle"] = "passed_both_variants_all_records".into();
        results.push(result);
    }
    let report = serde_json::json!({
        "schema_version": 1, "status": "complete", "release_qualified": false,
        "scope": "warm GPU physical-range lookup plus identical 16-byte output write per candidate",
        "excludes": ["uploads", "pipeline compilation", "CPU validation", "readback", "Gaussian evaluation", "scan/scatter", "sorting", "raster", "whole-frame performance"],
        "identities": identities,
        "adapter": {"name": info.name, "vendor": info.vendor, "device": info.device,
            "backend": format!("{:?}", info.backend), "device_type": format!("{:?}", info.device_type),
            "driver": info.driver, "driver_info": info.driver_info},
        "timestamp_period_ns": timestamp_period_ns,
        "method": {"order": "ABBA", "a": "baseline", "b": "current",
            "blocks": 5, "samples_per_variant": 10, "warmup_batches_per_variant": 2,
            "dispatches_per_sample": 8, "workgroup_size": 256,
            "output_buffer": "shared by both variants; all dispatches write every candidate",
            "statistics": "median is middle-pair average; p95 is nearest-rank; ratios current/baseline"},
        "cases": results,
    });
    output_file.seek(SeekFrom::Start(0)).unwrap();
    output_file.set_len(0).unwrap();
    serde_json::to_writer_pretty(&mut output_file, &report).unwrap();
    writeln!(output_file).unwrap();
    output_file.flush().unwrap();
    eprintln!("lookup timestamp microbenchmark: {output_path}");
}

#[cfg(all(feature = "lod", not(target_arch = "wasm32")))]
fn timestamp_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &[wgpu::ComputePipeline; 2],
    count: u32,
    ranges: &[[u32; 4]],
    timestamp_period_ns: f64,
) -> serde_json::Value {
    const REPEATS: u32 = 8;
    const SAMPLES: u32 = 20;
    let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("benchmark_config"),
        contents: bytemuck::cast_slice(&[count, ranges.len() as u32, 0, 0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let descriptors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("benchmark_ranges"),
        contents: bytemuck::cast_slice(ranges),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("benchmark_identical_output"),
        size: u64::from(count) * 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let groups = pipelines.each_ref().map(|pipeline| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("benchmark_bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: descriptors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    });
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("lookup_timestamp_pairs"),
        ty: wgpu::QueryType::Timestamp,
        count: SAMPLES * 2,
    });
    let query_bytes = u64::from(SAMPLES * 2) * 8;
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lookup_timestamp_resolve"),
        size: query_bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lookup_timestamp_readback"),
        size: query_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut warmup = device.create_command_encoder(&Default::default());
    for variant in [0, 1, 1, 0] {
        let mut pass = warmup.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipelines[variant]);
        pass.set_bind_group(0, &groups[variant], &[]);
        for _ in 0..REPEATS {
            pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        }
    }
    let warmup_submission = queue.submit([warmup.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(warmup_submission),
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();

    let mut encoder = device.create_command_encoder(&Default::default());
    for sample in 0..SAMPLES {
        let variant = [0, 1, 1, 0][sample as usize % 4];
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("timed_lookup_and_identical_output_writes"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &queries,
                beginning_of_pass_write_index: Some(sample * 2),
                end_of_pass_write_index: Some(sample * 2 + 1),
            }),
        });
        pass.set_pipeline(&pipelines[variant]);
        pass.set_bind_group(0, &groups[variant], &[]);
        for _ in 0..REPEATS {
            pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        }
    }
    encoder.resolve_query_set(&queries, 0..SAMPLES * 2, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, query_bytes);
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let mapped = readback.slice(..).get_mapped_range();
    let ticks: &[u64] = bytemuck::cast_slice(&mapped);
    let mut durations = [Vec::new(), Vec::new()];
    let mut samples = Vec::new();
    for sample in 0..SAMPLES as usize {
        let variant = [0, 1, 1, 0][sample % 4];
        let elapsed = ticks[2 * sample + 1]
            .checked_sub(ticks[2 * sample])
            .expect("GPU timestamp pair must be ordered");
        assert!(
            elapsed > 0,
            "timestamp interval must resolve above timer granularity"
        );
        let ns = elapsed as f64 * timestamp_period_ns / f64::from(REPEATS);
        durations[variant].push(ns);
        samples.push(serde_json::json!({"sample": sample, "block": sample / 4,
            "variant": (["baseline", "current"][variant]),
            "begin_tick": ticks[2 * sample], "end_tick": ticks[2 * sample + 1],
            "batch_ticks": elapsed, "ns_per_dispatch": ns}));
    }
    drop(mapped);
    readback.unmap();
    let summaries = durations.map(|mut times| {
        times.sort_by(f64::total_cmp);
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let median = (times[4] + times[5]) * 0.5;
        serde_json::json!({"mean_ns_per_dispatch": mean, "median_ns_per_dispatch": median,
            "p95_ns_per_dispatch": times[9], "minimum_ns_per_dispatch": times[0],
            "maximum_ns_per_dispatch": times[9]})
    });
    serde_json::json!({
        "candidate_count": count, "range_count": ranges.len(), "output_bytes": u64::from(count) * 16,
        "dispatch_workgroups": count.div_ceil(256), "samples": samples,
        "baseline": summaries[0], "current": summaries[1],
        "current_over_baseline_mean": summaries[1]["mean_ns_per_dispatch"].as_f64().unwrap()
            / summaries[0]["mean_ns_per_dispatch"].as_f64().unwrap(),
        "current_over_baseline_median": summaries[1]["median_ns_per_dispatch"].as_f64().unwrap()
            / summaries[0]["median_ns_per_dispatch"].as_f64().unwrap(),
    })
}
