//! Opt-in frame evidence interchange. This module records supplied observations;
//! it neither reads GPU state nor certifies renderer performance or image quality.
//!
//! Write one [`LodFrameCapture`] per JSONL line with [`LodFrameCapture::write_jsonl`],
//! then inspect the file with `python3 tools/check_lod_capture.py capture.jsonl`.
//! Delayed observations must retain their original [`LodCaptureStamp`].

use std::{collections::BTreeMap, fmt, io::Write, marker::PhantomData};

use serde::{Deserialize, Serialize};

pub const LOD_CAPTURE_SCHEMA_VERSION: u32 = 1;
pub const CPU_MEMORY_CATEGORIES: &[&str] = &[
    "metadata",
    "decoded_pages",
    "upload_staging",
    "transitions",
    "retired",
    "other",
];
pub const GPU_MEMORY_CATEGORIES: &[&str] = &[
    "atlas",
    "range_descriptors",
    "active_records",
    "sort",
    "transitions",
    "retired",
    "other",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LodCaptureMode {
    Synthetic,
    CpuOracle,
    NativeGpu,
    WebGpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LodCountSource {
    CpuSelection,
    CpuOracle,
    GpuReadback,
    Synthetic,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LodCapturePipeline {
    #[default]
    Hierarchy,
    /// GPU hierarchy selection followed by Gaussian point splatting. Compacted
    /// and drawn count admitted projected Gaussians, not stochastic point attempts.
    HierarchyPoint,
    /// GPU hierarchy selection followed by one globally ordered quad stream.
    HierarchyOrdered,
    /// A flat reference has an actual indirect draw, but no compaction stage.
    FlatSource,
}

impl LodCapturePipeline {
    pub fn uses_gpu_hierarchy(self) -> bool {
        matches!(self, Self::HierarchyPoint | Self::HierarchyOrdered)
    }
}

/// A run ID must identify one immutable artifact/renderer/configuration. Attach
/// this stamp when work is submitted, and preserve it through delayed readback.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LodCaptureStamp {
    pub run_id: String,
    pub view_id: String,
    pub frame: u64,
    pub generation: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LodCaptureIdentity {
    /// Hash the actual manifest bytes and source asset bytes, respectively.
    pub manifest_sha256: String,
    pub source_sha256: String,
    pub builder_revision: String,
    pub renderer_revision: String,
    /// Hash of the exact renderer executable/Wasm module, including local edits.
    pub renderer_sha256: String,
    pub features: Vec<String>,
    /// Canonical backend: synthetic, cpu, vulkan, metal, dx12, gl, or webgpu.
    pub backend: String,
    pub adapter: Option<String>,
    pub driver: Option<String>,
    pub camera_path_sha256: String,
    /// Includes filtering, sort mode, quality, budgets and color conventions.
    pub settings_sha256: String,
    pub instrumentation: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LodCaptureCamera {
    /// Column-major world-to-view and projection matrices used by this frame.
    pub world_to_view: [f64; 16],
    pub projection: [f64; 16],
    pub viewport: [u32; 2],
    pub pixel_scale: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LodCaptureCounts {
    pub stamp: LodCaptureStamp,
    pub source: LodCountSource,
    #[serde(default)]
    pub pipeline: LodCapturePipeline,
    pub selected: u64,
    /// Explicit extra candidate records introduced by transitions.
    pub transition_extra: u64,
    pub candidates: u64,
    pub output_capacity: u64,
    /// None means unobserved. Zero means an observed empty result.
    pub compacted: Option<u64>,
    /// Actual Gaussian records submitted through indirect draw arguments, or
    /// admitted projected Gaussians for `HierarchyPoint`. Point attempts are
    /// reported separately in the same-submission evidence.
    pub drawn: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LodCaptureTimings {
    pub stamp: LodCaptureStamp,
    pub frame_wall_ms: Option<f64>,
    #[serde(deserialize_with = "deserialize_unique_map")]
    pub cpu_ms: BTreeMap<String, f64>,
    /// Device timestamp measurements, never CPU submission durations.
    #[serde(deserialize_with = "deserialize_unique_map")]
    pub gpu_ms: BTreeMap<String, f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LodCaptureAllocation {
    pub reserved: u64,
    pub used: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LodCaptureMemory {
    pub stamp: LodCaptureStamp,
    /// Disjoint allocation categories; shared allocations have one owner.
    #[serde(deserialize_with = "deserialize_unique_map")]
    pub cpu: BTreeMap<String, LodCaptureAllocation>,
    #[serde(deserialize_with = "deserialize_unique_map")]
    pub gpu: BTreeMap<String, LodCaptureAllocation>,
    pub process_rss_bytes: Option<u64>,
    /// Independent device-level observation, not a sum of buffer sizes.
    pub device_used_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LodCaptureImage {
    pub stamp: LodCaptureStamp,
    pub path: String,
    pub sha256: String,
    pub viewport: [u32; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LodFrameCapture {
    pub schema_version: u32,
    pub mode: LodCaptureMode,
    pub identity: LodCaptureIdentity,
    pub stamp: LodCaptureStamp,
    /// Examples: cold_start, warm_stationary, motion, return_after_eviction.
    pub scenario: String,
    pub camera: LodCaptureCamera,
    pub counts: LodCaptureCounts,
    pub timings: Option<LodCaptureTimings>,
    pub memory: Option<LodCaptureMemory>,
    pub image: Option<LodCaptureImage>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LodCaptureError(pub String);

impl fmt::Display for LodCaptureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for LodCaptureError {}

fn require(valid: bool, message: &str) -> Result<(), LodCaptureError> {
    if valid {
        Ok(())
    } else {
        Err(LodCaptureError(message.to_owned()))
    }
}
fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

// Serde structs reject duplicate fields, but ordinary BTreeMap deserialization
// keeps the final value. Evidence maps must reject that ambiguous input too.
fn deserialize_unique_map<'de, D, T>(deserializer: D) -> Result<BTreeMap<String, T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct UniqueMap<T>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for UniqueMap<T> {
        type Value = BTreeMap<String, T>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an object with unique keys")
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut access: A,
        ) -> Result<Self::Value, A::Error> {
            let mut entries = BTreeMap::new();
            while let Some((key, value)) = access.next_entry::<String, T>()? {
                if entries.insert(key.clone(), value).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate evidence key: {key}"
                    )));
                }
            }
            Ok(entries)
        }
    }
    deserializer.deserialize_map(UniqueMap(PhantomData))
}

impl LodFrameCapture {
    /// Validate internal consistency. Success is not performance/quality evidence.
    pub fn validate(&self) -> Result<(), LodCaptureError> {
        require(
            self.schema_version == LOD_CAPTURE_SCHEMA_VERSION,
            "unsupported schema_version",
        )?;
        for (name, value) in [
            ("run_id", &self.stamp.run_id),
            ("view_id", &self.stamp.view_id),
            ("scenario", &self.scenario),
            ("backend", &self.identity.backend),
            ("builder_revision", &self.identity.builder_revision),
            ("renderer_revision", &self.identity.renderer_revision),
            ("instrumentation", &self.identity.instrumentation),
        ] {
            require(!value.trim().is_empty(), &format!("empty {name}"))?;
        }
        for value in [
            &self.identity.manifest_sha256,
            &self.identity.source_sha256,
            &self.identity.renderer_sha256,
            &self.identity.camera_path_sha256,
            &self.identity.settings_sha256,
        ] {
            require(digest(value), "identity hashes must be lowercase SHA-256")?;
        }
        require(
            self.identity
                .features
                .iter()
                .all(|value| !value.trim().is_empty())
                && self
                    .identity
                    .features
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
            "features must be nonempty names in sorted unique order",
        )?;
        require(
            self.camera.viewport.iter().all(|value| *value > 0),
            "empty viewport",
        )?;
        require(
            self.camera.pixel_scale.is_finite() && self.camera.pixel_scale > 0.0,
            "invalid pixel_scale",
        )?;
        require(
            self.camera
                .world_to_view
                .iter()
                .chain(&self.camera.projection)
                .all(|value| value.is_finite()),
            "non-finite camera matrix",
        )?;
        require(self.counts.stamp == self.stamp, "count stamp mismatch")?;
        let candidate_bound = self
            .counts
            .selected
            .checked_add(self.counts.transition_extra)
            .ok_or_else(|| LodCaptureError("selected plus transition_extra overflow".to_owned()))?;
        require(
            self.counts.candidates <= candidate_bound,
            "candidates exceed selected plus transition_extra",
        )?;
        if let Some(compacted) = self.counts.compacted {
            require(
                compacted <= self.counts.candidates && compacted <= self.counts.output_capacity,
                "compacted count exceeds candidates or output capacity",
            )?;
        }
        if let Some(drawn) = self.counts.drawn {
            require(
                match self.counts.pipeline {
                    LodCapturePipeline::Hierarchy
                    | LodCapturePipeline::HierarchyPoint
                    | LodCapturePipeline::HierarchyOrdered => self.counts.compacted == Some(drawn),
                    LodCapturePipeline::FlatSource => {
                        self.counts.compacted.is_none()
                            && drawn <= self.counts.candidates
                            && drawn <= self.counts.output_capacity
                    }
                },
                "drawn count does not match the declared pipeline",
            )?;
        }
        require(
            self.counts.pipeline != LodCapturePipeline::FlatSource
                || self.counts.compacted.is_none(),
            "flat source has no compaction observation",
        )?;
        require(
            self.counts.source != LodCountSource::CpuSelection
                || (self.counts.compacted.is_none() && self.counts.drawn.is_none()),
            "CPU selection cannot report compacted or drawn counts",
        )?;
        let gpu = matches!(
            self.mode,
            LodCaptureMode::NativeGpu | LodCaptureMode::WebGpu
        );
        require(
            match self.mode {
                LodCaptureMode::Synthetic => self.identity.backend == "synthetic",
                LodCaptureMode::CpuOracle => self.identity.backend == "cpu",
                LodCaptureMode::NativeGpu => matches!(
                    self.identity.backend.as_str(),
                    "vulkan" | "metal" | "dx12" | "gl"
                ),
                LodCaptureMode::WebGpu => self.identity.backend == "webgpu",
            },
            "backend does not match capture mode",
        )?;
        require(
            match self.mode {
                LodCaptureMode::Synthetic => self.counts.source == LodCountSource::Synthetic,
                LodCaptureMode::CpuOracle => matches!(
                    self.counts.source,
                    LodCountSource::CpuSelection | LodCountSource::CpuOracle
                ),
                _ => matches!(
                    self.counts.source,
                    LodCountSource::CpuSelection | LodCountSource::GpuReadback
                ),
            },
            "count provenance does not match capture mode",
        )?;
        if let Some(timings) = &self.timings {
            require(timings.stamp == self.stamp, "timing stamp mismatch")?;
            require(
                gpu || timings.gpu_ms.is_empty(),
                "GPU timestamps require a GPU capture",
            )?;
            require(
                timings
                    .cpu_ms
                    .keys()
                    .chain(timings.gpu_ms.keys())
                    .all(|key| !key.trim().is_empty()),
                "empty timing stage",
            )?;
            require(
                timings
                    .frame_wall_ms
                    .iter()
                    .chain(timings.cpu_ms.values())
                    .chain(timings.gpu_ms.values())
                    .all(|value| value.is_finite() && *value >= 0.0),
                "invalid timing value",
            )?;
        }
        if let Some(memory) = &self.memory {
            require(memory.stamp == self.stamp, "memory stamp mismatch")?;
            for entries in [&memory.cpu, &memory.gpu] {
                let mut reserved = 0_u64;
                let mut used = 0_u64;
                for (name, entry) in entries {
                    require(!name.trim().is_empty(), "empty memory category")?;
                    require(entry.used <= entry.reserved, "memory used exceeds reserved")?;
                    reserved = reserved.checked_add(entry.reserved).ok_or_else(|| {
                        LodCaptureError("memory reserved total overflow".to_owned())
                    })?;
                    used = used
                        .checked_add(entry.used)
                        .ok_or_else(|| LodCaptureError("memory used total overflow".to_owned()))?;
                }
            }
        }
        if let Some(image) = &self.image {
            require(image.stamp == self.stamp, "image stamp mismatch")?;
            require(
                image.viewport == self.camera.viewport,
                "image viewport mismatch",
            )?;
            require(
                !image.path.trim().is_empty()
                    && !image.path.contains('\0')
                    && digest(&image.sha256),
                "invalid image identity",
            )?;
        }
        Ok(())
    }

    /// Missing evidence for a GPU measurement capture. An empty list means only
    /// that the observations are present, not that release thresholds passed.
    pub fn missing_gpu_evidence(&self) -> Vec<String> {
        if let Err(error) = self.validate() {
            return vec![format!("invalid capture: {error}")];
        }
        let mut missing = Vec::new();
        if !matches!(
            self.mode,
            LodCaptureMode::NativeGpu | LodCaptureMode::WebGpu
        ) {
            missing.push("GPU capture mode".to_owned());
        }
        if self.counts.source != LodCountSource::GpuReadback
            || (self.counts.pipeline != LodCapturePipeline::FlatSource
                && self.counts.compacted.is_none())
            || self.counts.drawn.is_none()
        {
            missing.push("applicable compacted and drawn GPU readback".to_owned());
        }
        for (name, value) in [
            ("adapter", &self.identity.adapter),
            ("driver", &self.identity.driver),
        ] {
            if value.as_ref().is_none_or(|value| value.trim().is_empty()) {
                missing.push(name.to_owned());
            }
        }
        if self.timings.as_ref().is_none_or(|timings| {
            timings.frame_wall_ms.is_none()
                || timings.cpu_ms.is_empty()
                || timings.gpu_ms.is_empty()
        }) {
            missing.push("frame, CPU stage and GPU timestamp timings".to_owned());
        }
        if let Some(memory) = &self.memory {
            for (domain, entries, categories) in [
                ("cpu", &memory.cpu, CPU_MEMORY_CATEGORIES),
                ("gpu", &memory.gpu, GPU_MEMORY_CATEGORIES),
            ] {
                for category in categories {
                    if !entries.contains_key(*category) {
                        missing.push(format!("memory.{domain}.{category}"));
                    }
                }
            }
            if memory.process_rss_bytes.is_none() {
                missing.push("process RSS observation".to_owned());
            }
            if memory.device_used_bytes.is_none() {
                missing.push("device memory observation".to_owned());
            }
        } else {
            missing.push("memory ledger and independent observations".to_owned());
        }
        if self.image.is_none() {
            missing.push("matching image identity".to_owned());
        }
        missing
    }

    /// Validate before writing, so an invalid capture cannot partially append a
    /// JSON record. I/O failures can still leave a partial final line.
    pub fn write_jsonl(&self, mut writer: impl Write) -> Result<(), LodCaptureError> {
        self.validate()?;
        let mut encoded =
            serde_json::to_vec(self).map_err(|error| LodCaptureError(error.to_string()))?;
        encoded.push(b'\n');
        writer
            .write_all(&encoded)
            .map_err(|error| LodCaptureError(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> LodFrameCapture {
        serde_json::from_str(
            include_str!("../../tools/fixtures/lod_capture_synthetic.jsonl").trim(),
        )
        .unwrap()
    }

    #[test]
    fn cpu_fixture_round_trips_without_qualifying_gpu_work() {
        let capture = fixture();
        let mut bytes = Vec::new();
        capture.write_jsonl(&mut bytes).unwrap();
        assert_eq!(
            serde_json::from_slice::<LodFrameCapture>(&bytes).unwrap(),
            capture
        );
        assert!(!capture.missing_gpu_evidence().is_empty());
    }

    #[test]
    fn mismatched_or_overflowing_counts_cannot_be_written() {
        for case in 0..4 {
            let mut capture = fixture();
            match case {
                0 => capture.counts.drawn = Some(99),
                1 => capture.counts.selected = u64::MAX,
                2 => capture.counts.compacted = Some(capture.counts.output_capacity + 1),
                3 => capture.counts.stamp.generation += 1,
                _ => unreachable!(),
            }
            let mut bytes = Vec::new();
            assert!(capture.write_jsonl(&mut bytes).is_err());
            assert!(bytes.is_empty());
        }
    }

    #[test]
    fn delayed_image_and_nonfinite_timing_are_rejected() {
        let mut capture = fixture();
        capture.image.as_mut().unwrap().stamp.frame += 1;
        assert!(capture.validate().is_err());
        let mut capture = fixture();
        capture.timings.as_mut().unwrap().frame_wall_ms = Some(f64::NAN);
        assert!(capture.validate().is_err());
    }

    #[test]
    fn memory_overflow_is_rejected() {
        let mut capture = fixture();
        let cpu = &mut capture.memory.as_mut().unwrap().cpu;
        cpu.insert(
            "one".to_owned(),
            LodCaptureAllocation {
                reserved: u64::MAX,
                used: 0,
            },
        );
        cpu.insert(
            "two".to_owned(),
            LodCaptureAllocation {
                reserved: 1,
                used: 0,
            },
        );
        assert!(capture.validate().is_err());
    }

    #[test]
    fn completeness_fails_closed_without_prior_validation() {
        let mut capture = fixture();
        capture.counts.drawn = Some(9);
        assert!(capture.missing_gpu_evidence()[0].starts_with("invalid capture:"));
        let mut capture = fixture();
        capture.mode = LodCaptureMode::NativeGpu;
        capture.counts.source = LodCountSource::GpuReadback;
        assert!(capture.validate().unwrap_err().0.contains("backend"));
    }

    #[test]
    fn deserialization_rejects_duplicate_evidence_map_entries() {
        let encoded = serde_json::to_string(&fixture()).unwrap();
        for (original, replacement) in [
            (
                "\"cpu_ms\":{\"fixture\":0.1}",
                "\"cpu_ms\":{\"fixture\":0.1,\"fixture\":0.2}",
            ),
            (
                "\"gpu_ms\":{}",
                "\"gpu_ms\":{\"raster\":0.1,\"raster\":0.2}",
            ),
            (
                "\"cpu\":{}",
                "\"cpu\":{\"a\":{\"reserved\":1,\"used\":0},\"a\":{\"reserved\":2,\"used\":0}}",
            ),
            (
                "\"gpu\":{}",
                "\"gpu\":{\"a\":{\"reserved\":1,\"used\":0},\"a\":{\"reserved\":2,\"used\":0}}",
            ),
        ] {
            assert!(encoded.contains(original));
            let duplicate = encoded.replace(original, replacement);
            assert!(serde_json::from_str::<LodFrameCapture>(&duplicate).is_err());
        }
    }

    #[test]
    fn deserialization_rejects_unknown_fields_and_boolean_numbers() {
        for path in [
            "",
            "/identity",
            "/stamp",
            "/camera",
            "/counts",
            "/timings",
            "/memory",
            "/image",
        ] {
            let mut value = serde_json::to_value(fixture()).unwrap();
            value
                .pointer_mut(path)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("unknown".to_owned(), serde_json::json!(0));
            assert!(serde_json::from_value::<LodFrameCapture>(value).is_err());
        }
        for path in [
            "/schema_version",
            "/stamp/frame",
            "/counts/selected",
            "/timings/frame_wall_ms",
            "/camera/pixel_scale",
            "/memory/process_rss_bytes",
        ] {
            let mut value = serde_json::to_value(fixture()).unwrap();
            *value.pointer_mut(path).unwrap() = serde_json::json!(true);
            assert!(serde_json::from_value::<LodFrameCapture>(value).is_err());
        }
        let encoded = serde_json::to_string(&fixture()).unwrap();
        assert!(encoded.contains("\"selected\":10"));
        assert!(
            serde_json::from_str::<LodFrameCapture>(
                &encoded.replace("\"selected\":10", "\"selected\":18446744073709551616")
            )
            .is_err()
        );
    }
}
