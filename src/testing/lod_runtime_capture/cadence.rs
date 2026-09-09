//! CPU frame cadence without GPU probes, copies, timestamps or screenshots.
//! These samples make no assertion about draw submission or GPU completion.

use super::{CaptureResult, LodCaptureIdentity};
use serde::Serialize;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

#[derive(Serialize)]
pub(super) struct CadenceSample {
    pub frame: u64,
    pub path_frame: u64,
    pub scenario: String,
    pub world_from_view: [f32; 16],
    pub frame_wall_ms: f64,
    pub cpu_readiness_observed: bool,
}

pub(super) fn write(
    output: &Path,
    run_id: &str,
    identity: &LodCaptureIdentity,
    samples: &[CadenceSample],
    readiness_scope: &str,
) -> CaptureResult<()> {
    let mut file = BufWriter::new(File::create(output.join("frame_cadence.jsonl"))?);
    for sample in samples {
        serde_json::to_writer(&mut file, sample)?;
        file.write_all(b"\n")?;
    }
    file.flush()?;
    let scenarios: std::collections::BTreeSet<_> =
        samples.iter().map(|sample| &sample.scenario).collect();
    let mut summary = std::collections::BTreeMap::new();
    for scenario in scenarios {
        let mut times: Vec<_> = samples
            .iter()
            .filter(|sample| &sample.scenario == scenario)
            .map(|sample| sample.frame_wall_ms)
            .collect();
        times.sort_by(f64::total_cmp);
        let percentile = |fraction: f64| {
            times[((times.len() as f64 * fraction).ceil() as usize).saturating_sub(1)]
        };
        summary.insert(scenario, serde_json::json!({"samples": times.len(),
            "p50_ms": percentile(0.50), "p95_ms": percentile(0.95), "p99_ms": percentile(0.99),
            "maximum_ms": times.last(), "frames_over_50_ms": times.iter().filter(|&&value| value > 50.0).count()}));
    }
    let document = serde_json::json!({"schema_version": 1, "mode": "cadence_only",
        "run_id": run_id, "identity": identity,
        "timing_scope": "main_loop_start_to_start;CPU_sample_collection;no_GPU_copy_timestamp_draw_probe_or_image;not_GPU_completion",
        "readiness_scope": readiness_scope, "actual_draw_attested": false,
        "sample_count": samples.len(), "scenarios": summary, "release_qualified": false});
    let mut file = BufWriter::new(File::create(output.join("cadence_summary.json"))?);
    serde_json::to_writer_pretty(&mut file, &document)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}
