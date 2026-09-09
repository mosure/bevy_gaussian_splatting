#!/usr/bin/env python3
"""Validate and summarize LoD frame JSONL without running a renderer.

This checks supplied evidence consistency and completeness. It cannot establish
that a producer really measured a GPU or that performance/quality gates pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

U64_MAX = (1 << 64) - 1
MAX_CAPTURE_LINE_BYTES = 4 * 1024 * 1024
CPU_CATEGORIES = {"metadata", "decoded_pages", "upload_staging", "transitions", "retired", "other"}
GPU_CATEGORIES = {"atlas", "range_descriptors", "active_records", "sort", "transitions", "retired", "other"}


class CaptureError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CaptureError(message)


def shape(value, required, optional=()):
    require(isinstance(value, dict), "expected an object")
    required, optional = set(required.split()), set(optional.split()) if isinstance(optional, str) else set(optional)
    require(not (required - value.keys()), f"missing fields: {sorted(required - value.keys())}")
    require(not (value.keys() - required - optional), f"unknown fields: {sorted(value.keys() - required - optional)}")


def string(value, allow_empty=False):
    require(isinstance(value, str), "expected string")
    # Python's JSON parser accepts escaped lone surrogates, while Rust strings
    # cannot contain them. Reject them before identity comparison or file I/O.
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CaptureError("string contains an invalid Unicode surrogate") from error
    require(allow_empty or bool(value.strip()), "expected nonempty string")


def uint(value, maximum=U64_MAX):
    require(type(value) is int and 0 <= value <= maximum, "expected unsigned integer within declared width")


def number(value, positive=False):
    require(type(value) in (int, float), "expected finite number")
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    require(valid, "expected finite positive/nonnegative number")


def digest(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "expected lowercase SHA-256")


def stamp(value):
    shape(value, "run_id view_id frame generation")
    string(value["run_id"])
    string(value["view_id"])
    uint(value["frame"])
    uint(value["generation"])


def viewport(value):
    require(isinstance(value, list) and len(value) == 2, "viewport must have two dimensions")
    for dimension in value:
        uint(dimension, (1 << 32) - 1)
        require(dimension > 0, "viewport must be nonempty")


def validate(record):
    shape(record, "schema_version mode identity stamp scenario camera counts", "timings memory image")
    require(type(record["schema_version"]) is int and record["schema_version"] == 1, "unsupported schema_version")
    require(record["mode"] in ("synthetic", "cpu_oracle", "native_gpu", "web_gpu"), "unknown capture mode")
    stamp(record["stamp"])
    string(record["scenario"])
    identity = record["identity"]
    shape(identity, "manifest_sha256 source_sha256 builder_revision renderer_revision renderer_sha256 features backend camera_path_sha256 settings_sha256 instrumentation", "adapter driver")
    for field in ("manifest_sha256", "source_sha256", "renderer_sha256", "camera_path_sha256", "settings_sha256"):
        digest(identity[field])
    for field in ("builder_revision", "renderer_revision", "backend", "instrumentation"):
        string(identity[field])
    backends = {"synthetic": ("synthetic",), "cpu_oracle": ("cpu",), "native_gpu": ("vulkan", "metal", "dx12", "gl"), "web_gpu": ("webgpu",)}
    require(identity["backend"] in backends[record["mode"]], "backend does not match capture mode")
    for field in ("adapter", "driver"):
        if identity.get(field) is not None:
            string(identity[field], allow_empty=True)
    features = identity["features"]
    require(isinstance(features, list), "features must be an array")
    for feature in features:
        string(feature)
    require(features == sorted(set(features)), "features must be sorted and unique")
    camera = record["camera"]
    shape(camera, "world_to_view projection viewport pixel_scale")
    viewport(camera["viewport"])
    number(camera["pixel_scale"], positive=True)
    for field in ("world_to_view", "projection"):
        matrix = camera[field]
        require(isinstance(matrix, list) and len(matrix) == 16, "camera matrix must contain 16 values")
        for value in matrix:
            require(type(value) in (float, int), "camera matrix must be numeric")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            require(finite, "camera matrix must be finite")
    counts = record["counts"]
    shape(counts, "stamp source selected transition_extra candidates output_capacity", "compacted drawn pipeline")
    pipeline = counts.get("pipeline", "hierarchy")
    require(pipeline in ("hierarchy", "hierarchy_point", "hierarchy_ordered", "flat_source"), "unknown count pipeline")
    stamp(counts["stamp"])
    require(counts["stamp"] == record["stamp"], "count stamp mismatch")
    sources = {
        "synthetic": ("synthetic",), "cpu_oracle": ("cpu_selection", "cpu_oracle"),
        "native_gpu": ("cpu_selection", "gpu_readback"), "web_gpu": ("cpu_selection", "gpu_readback"),
    }
    require(counts["source"] in sources[record["mode"]], "count provenance does not match capture mode")
    for field in ("selected", "transition_extra", "candidates", "output_capacity"):
        uint(counts[field])
    candidate_bound = counts["selected"] + counts["transition_extra"]
    require(candidate_bound <= U64_MAX, "selected plus transition_extra overflow")
    require(counts["candidates"] <= candidate_bound, "candidates exceed selected plus transition_extra")
    compacted, drawn = counts.get("compacted"), counts.get("drawn")
    if compacted is not None:
        uint(compacted)
        require(compacted <= min(counts["candidates"], counts["output_capacity"]), "compacted count exceeds candidates or output capacity")
    if drawn is not None:
        uint(drawn)
        require((drawn == compacted if pipeline != "flat_source" else compacted is None and drawn <= min(counts["candidates"], counts["output_capacity"])), "drawn count does not match the declared pipeline")
    require(pipeline != "flat_source" or compacted is None, "flat source has no compaction observation")
    require(counts["source"] != "cpu_selection" or (compacted is None and drawn is None), "CPU selection cannot report compacted or drawn counts")
    timings = record.get("timings")
    if timings is not None:
        shape(timings, "stamp cpu_ms gpu_ms", "frame_wall_ms")
        stamp(timings["stamp"])
        require(timings["stamp"] == record["stamp"], "timing stamp mismatch")
        if timings.get("frame_wall_ms") is not None:
            number(timings["frame_wall_ms"])
        for field in ("cpu_ms", "gpu_ms"):
            require(isinstance(timings[field], dict), "timing stages must be an object")
            for name, value in timings[field].items():
                string(name)
                number(value)
        require(record["mode"] in ("native_gpu", "web_gpu") or not timings["gpu_ms"], "GPU timestamps require a GPU capture")
    memory = record.get("memory")
    if memory is not None:
        shape(memory, "stamp cpu gpu", "process_rss_bytes device_used_bytes")
        stamp(memory["stamp"])
        require(memory["stamp"] == record["stamp"], "memory stamp mismatch")
        for domain in ("cpu", "gpu"):
            require(isinstance(memory[domain], dict), "memory categories must be an object")
            totals = {"reserved": 0, "used": 0}
            for name, entry in memory[domain].items():
                string(name)
                shape(entry, "reserved used")
                for field in totals:
                    uint(entry[field])
                    totals[field] += entry[field]
                    require(totals[field] <= U64_MAX, f"memory {field} total overflow")
                require(entry["used"] <= entry["reserved"], "memory used exceeds reserved")
        for field in ("process_rss_bytes", "device_used_bytes"):
            if memory.get(field) is not None:
                uint(memory[field])
    image = record.get("image")
    if image is not None:
        shape(image, "stamp path sha256 viewport")
        stamp(image["stamp"])
        require(image["stamp"] == record["stamp"], "image stamp mismatch")
        string(image["path"])
        require("\0" not in image["path"], "image path contains a NUL byte")
        digest(image["sha256"])
        viewport(image["viewport"])
        require(image["viewport"] == camera["viewport"], "image viewport mismatch")
    return record


def missing_gpu_evidence(record):
    try:
        validate(record)
    except CaptureError as error:
        return [f"invalid capture: {error}"]
    missing = []
    if record["mode"] not in ("native_gpu", "web_gpu"):
        missing.append("GPU capture mode")
    counts = record["counts"]
    if counts["source"] != "gpu_readback" or (counts.get("pipeline", "hierarchy") != "flat_source" and counts.get("compacted") is None) or counts.get("drawn") is None:
        missing.append("applicable compacted and drawn GPU readback")
    for field in ("adapter", "driver"):
        if not (record["identity"].get(field) or "").strip():
            missing.append(field)
    timings = record.get("timings")
    if not timings or timings.get("frame_wall_ms") is None or not timings["cpu_ms"] or not timings["gpu_ms"]:
        missing.append("frame, CPU stage and GPU timestamp timings")
    memory = record.get("memory")
    if memory is None:
        missing.append("memory ledger and independent observations")
    else:
        for domain, required in (("cpu", CPU_CATEGORIES), ("gpu", GPU_CATEGORIES)):
            missing.extend(f"memory.{domain}.{name}" for name in sorted(required - memory[domain].keys()))
        if memory.get("process_rss_bytes") is None:
            missing.append("process RSS observation")
        if memory.get("device_used_bytes") is None:
            missing.append("device memory observation")
    if record.get("image") is None:
        missing.append("matching image identity")
    return missing


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise CaptureError(f"non-finite JSON constant: {value}")


def register_record(record, seen, runs):
    validate(record)
    stamp_value = record["stamp"]
    key = (stamp_value["run_id"], stamp_value["view_id"], stamp_value["frame"])
    require(key not in seen, "duplicate run/view/frame (including conflicting generation)")
    seen.add(key)
    # Omission and explicit null mean the same thing in Rust Option fields.
    identity = {**record["identity"], "adapter": record["identity"].get("adapter"), "driver": record["identity"].get("driver")}
    run_identity = (identity, record["mode"], record["counts"].get("pipeline", "hierarchy"))
    previous = runs.setdefault(stamp_value["run_id"], run_identity)
    require(previous == run_identity, "run identity changed; use a new run_id")


def read_captures(path):
    records, seen, runs = [], set(), {}
    with Path(path).open("rb") as source:
        for line_number, line in enumerate(iter(lambda: source.readline(MAX_CAPTURE_LINE_BYTES + 1), b""), 1):
            require(len(line) <= MAX_CAPTURE_LINE_BYTES, f"line {line_number}: capture exceeds 4 MiB")
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"), object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
                register_record(record, seen, runs)
                records.append(record)
            except (CaptureError, ValueError, TypeError, KeyError, RecursionError) as error:
                raise CaptureError(f"line {line_number}: {error}") from error
    require(bool(records), "empty capture")
    return records


def percentile(values, fraction):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(records):
    groups, seen, runs = {}, set(), {}
    for record in records:
        register_record(record, seen, runs)
        key = (record["stamp"]["run_id"], record["stamp"]["view_id"], record["scenario"])
        groups.setdefault(key, []).append(record)
    require(bool(groups), "empty capture")
    summary = []
    for (run_id, view_id, scenario), frames in sorted(groups.items()):
        times = [r["timings"]["frame_wall_ms"] for r in frames if r.get("timings") and r["timings"].get("frame_wall_ms") is not None]
        missing = sorted({item for record in frames for item in missing_gpu_evidence(record)})
        peak_memory = {}
        for domain in ("cpu", "gpu"):
            for field in ("reserved", "used"):
                samples = [sum(e[field] for e in r["memory"][domain].values()) for r in frames if r.get("memory") is not None and r["memory"][domain]]
                peak_memory[f"{domain}_{field}_bytes"] = max(samples, default=None)
        summary.append({
            "run_id": run_id, "view_id": view_id, "scenario": scenario,
            "mode": frames[0]["mode"], "frames": len(frames), "timed_frames": len(times),
            "frame_wall_ms": {"p50": percentile(times, .50), "p95": percentile(times, .95), "p99": percentile(times, .99), "over_50ms": sum(t > 50 for t in times)},
            "peak_counts": {field: max((r["counts"][field] for r in frames if r["counts"].get(field) is not None), default=None) for field in ("selected", "transition_extra", "candidates", "compacted", "drawn")},
            "peak_recorded_memory": peak_memory,
            "gpu_evidence_complete": not missing, "missing_gpu_evidence": missing,
        })
    return {"schema_version": 1, "valid_capture": True, "release_qualified": False, "groups": summary}


def verify_images(records, capture_path):
    for record in records:
        image = record.get("image")
        if image is None:
            continue
        path = Path(capture_path).parent / image["path"]
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(block)
        require(hasher.hexdigest() == image["sha256"], f"image SHA-256 mismatch: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--require-gpu-evidence", action="store_true", help="fail if any frame lacks complete GPU measurement fields")
    parser.add_argument("--require-scenario", action="append", default=[], help="require a scenario in every run/view; repeat as needed")
    parser.add_argument("--minimum-frames", type=int, default=1, help="minimum frames in each reported run/view/scenario")
    parser.add_argument("--verify-images", action="store_true", help="read image files relative to capture and verify their hashes")
    args = parser.parse_args(argv)
    try:
        require(args.minimum_frames >= 1, "minimum-frames must be positive")
        records = read_captures(args.capture)
        if args.verify_images:
            verify_images(records, args.capture)
        summary = summarize(records)
        summary["images_verified"] = args.verify_images
        failures = []
        scenarios = {}
        for group in summary["groups"]:
            key = (group["run_id"], group["view_id"])
            scenarios.setdefault(key, set()).add(group["scenario"])
            if group["frames"] < args.minimum_frames:
                failures.append(f"{key}/{group['scenario']}: too few frames")
            if args.require_gpu_evidence and not group["gpu_evidence_complete"]:
                failures.append(f"{key}/{group['scenario']}: missing GPU evidence")
        for key, present in scenarios.items():
            absent = set(args.require_scenario) - present
            if absent:
                failures.append(f"{key}: missing scenarios {sorted(absent)}")
        summary["requested_checks_passed"] = not failures
        summary["failures"] = failures
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 2 if failures else 0
    except (CaptureError, OSError, UnicodeError) as error:
        print(f"invalid capture: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
