#!/usr/bin/env python3
"""Validate actual procedural lifecycle captures and compare matched camera work.

No renderer is launched. Source-size scaling is accepted only after both runs
pass lifecycle evidence and their stationary visible draw work matches.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from check_lod_capture import CaptureError, read_captures, reject_constant, reject_duplicates


def require(condition, message):
    if not condition:
        raise CaptureError(message)


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=reject_constant,
                      object_pairs_hook=reject_duplicates)


def positive(value):
    return type(value) is int and 0 < value <= 2**64 - 1


def validate(directory):
    directory = Path(directory)
    status = load(directory / "status.json")
    config = load(directory / "settings.json")
    prepared = load(directory / "preparation.json")
    require(status.get("schema_version") == 1, "unsupported lifecycle schema")
    require(status.get("execution_verified") is True, "renderer did not complete verified lifecycle")
    require(status.get("release_qualified") is False, "synthetic lifecycle cannot qualify a release")
    require(status.get("failure") is None, "renderer reported a failure")
    require(positive(status.get("source_gaussians")), "invalid source count")
    require(status["source_gaussians"] == config["source_gaussians"] == prepared["source_gaussians"],
            "source extent identity mismatch")
    leaf_pages = (config["source_gaussians"] + config["records_per_page"] - 1) // config["records_per_page"]
    require(status["real_leaf_pages"] == prepared["real_leaf_pages"] == leaf_pages,
            "source extent lacks actual leaf descriptors")
    require(prepared["payload_archive_bytes"] == 0, "fixture materialized a payload archive")
    require(status.get("unload_zero_ledger_observed") is True, "no retirement observation")
    require(positive(status.get("slot_replacements_before_unload")), "no actual atlas slot replacement")
    require(len(status.get("observed_leaf_pages", [])) > 1, "only a proxy/root was observed")
    require(1 <= status["server"]["peak_active_handlers"] <= 2, "server escaped handler bound")
    require(not status["capture_stats"]["mapping_errors"], "GPU map errors")
    records = read_captures(directory / "capture.jsonl")
    require(records, "no capture records")
    evidence = {}
    for line in (directory / "submission_evidence.jsonl").read_text().splitlines():
        row = json.loads(line, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
        require(type(row.get("frame")) is int and row["frame"] not in evidence, "duplicate or invalid evidence frame")
        evidence[row["frame"]] = row
    phases = defaultdict(list)
    for record in records:
        require(record["identity"]["manifest_sha256"] == prepared["manifest_sha256"], "manifest identity mismatch")
        require(record["identity"]["source_sha256"] == prepared["generator_identity"], "generator identity mismatch")
        adapter = record["identity"].get("adapter", "").lower()
        require(not any(word in adapter for word in ("llvmpipe", "swiftshader", "software")), "software adapter")
        frame = record["stamp"]["frame"]
        row = evidence.get(frame, {})
        drawn = record["counts"]["drawn"]
        if positive(drawn):
            require(row.get("draw_command_attested") is True, "positive draw is unattested")
            require(row.get("source") == "post_render_same_submission_copy", "wrong submission provenance")
            require(row.get("indirect", {}).get("instance_count") == drawn, "actual indirect count mismatch")
            require(row.get("indirect", {}).get("overflow_count") == 0, "GPU overflow")
            require(row.get("compaction_generation") == record["stamp"]["generation"], "generation mismatch")
            phases[record["scenario"]].append(record)
    expected = {"cold", "stationary", "move", "rapid_return", "eviction", "reload"}
    require(set(phases) == expected, f"missing or extra nonzero draw phases: {set(phases)}")
    lifecycle = [json.loads(line, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
                 for line in (directory / "lifecycle.jsonl").read_text().splitlines()]
    require(lifecycle, "no lifecycle measurements")
    require(any(row["phase"] == "unload" and row["memory"]["total_bytes"] == 0 for row in lifecycle),
            "retirement summary has no zero-ledger frame")
    for row in lifecycle:
        memory = row["memory"]
        require(memory["cpu_bytes"] <= config["max_cpu_bytes"] and memory["gpu_bytes"] <= config["max_gpu_bytes"], "ledger limit exceeded")
        require(memory["cpu_bytes"] + memory["gpu_bytes"] == memory["total_bytes"], "memory total mismatch")
    return {"directory": str(directory), "source_gaussians": status["source_gaussians"],
            "real_leaf_pages": leaf_pages, "draws_by_phase": {key: len(rows) for key, rows in phases.items()},
            "sampled_peak_rss_bytes": status["sampled_peak_rss_bytes"],
            "cpu_prehash_seconds": prepared["cpu_prehash_seconds"],
            "slot_replacements_before_unload": status["slot_replacements_before_unload"],
            "execution_verified": True, "release_qualified": False}, phases, config


def compare(left, right, tolerance=0.01):
    require(type(tolerance) in (int, float) and math.isfinite(tolerance) and 0 <= tolerance <= 1,
            "invalid draw tolerance")
    a, a_phases, a_config = left
    b, b_phases, b_config = right
    for key in ("records_per_page", "grid_width", "seed", "viewport", "quality", "max_active_gaussians",
                "max_resident_pages", "max_cpu_bytes", "max_gpu_bytes", "phase_frames", "capture_every"):
        require(a_config[key] == b_config[key], f"unmatched workload setting: {key}")
    require(a["source_gaussians"] < b["source_gaussians"], "scaling inputs must increase real source extent")
    comparisons = {}
    for phase in ("stationary", "reload"):
        # Tail removes startup transients while retaining actual same-camera work.
        rows_a, rows_b = a_phases[phase][-10:], b_phases[phase][-10:]
        require(len(rows_a) >= 3 and len(rows_b) >= 3, "too few settled samples")
        for rows in (rows_a, rows_b):
            require(all(row["camera"] == rows[0]["camera"] for row in rows), "stationary camera moved")
        # Camera records include the frame stamp; compare only actual matrices.
        for key in ("world_to_view", "projection", "viewport"):
            require(rows_a[0]["camera"][key] == rows_b[0]["camera"][key], "camera paths differ")
        draw_a = statistics.median(row["counts"]["drawn"] for row in rows_a)
        draw_b = statistics.median(row["counts"]["drawn"] for row in rows_b)
        difference = abs(draw_a - draw_b) / max(draw_a, draw_b)
        require(difference <= tolerance, f"{phase} visible work differs by {difference:.1%}; no scaling claim")
        wall_a = [row["timings"]["frame_wall_ms"] for row in rows_a if row.get("timings") and row["timings"]["frame_wall_ms"] is not None]
        wall_b = [row["timings"]["frame_wall_ms"] for row in rows_b if row.get("timings") and row["timings"]["frame_wall_ms"] is not None]
        comparisons[phase] = {"drawn_median": [draw_a, draw_b], "draw_relative_difference": difference,
                              "instrumented_frame_wall_median_ms": [statistics.median(wall_a) if wall_a else None, statistics.median(wall_b) if wall_b else None]}
    return {"matched_visible_work": True, "draw_relative_tolerance": tolerance,
            "phases": comparisons, "release_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--max-draw-relative-difference", type=float, default=0.01)
    args = parser.parse_args()
    try:
        require(len(args.runs) in (1, 2), "provide one lifecycle run or two scaling runs")
        results = [validate(path) for path in args.runs]
        report = {"runs": [result[0] for result in results]}
        if len(results) == 2:
            report["scaling"] = compare(*results, tolerance=args.max_draw_relative_difference)
        print(json.dumps(report, indent=2))
    except (CaptureError, KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"virtual-city validation failed: {error}\n")


if __name__ == "__main__":
    main()
