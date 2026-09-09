#!/usr/bin/env python3
"""Validate stamped browser execution evidence; never promote it to image/release qualification."""
import argparse
import json
import math
import pathlib
import re

PHASES = ["cold_load", "stationary", "move", "unload", "reload", "complete"]
CATEGORIES = {"AtlasGpu", "CompactionGpu", "DecodedPagesCpu", "RecoveryStagingCpu",
              "TransportCpu", "PreprocessCpu", "MetadataCpu", "TransitionCpu", "UploadStagingCpu"}
MAX_SAFE = (1 << 53) - 1
ADAPTER_KEYS = ("name", "backend", "driver", "driver_info")


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def uint(value, label, maximum=MAX_SAFE):
    require(type(value) is int and 0 <= value <= maximum, f"invalid {label}")
    return value


def fields(value, expected, label):
    require(type(value) is dict and set(value) == set(expected.split()), f"unexpected {label} fields")


def matrix(value, label):
    require(type(value) is list and len(value) == 16, f"invalid {label}")
    require(all(type(x) in (int, float) and math.isfinite(x) and abs(x) <= 1e20 for x in value), f"nonfinite {label}")


def identity_integer(value, label):
    require(type(value) is str and re.fullmatch(r"[1-9][0-9]{0,19}", value) is not None
            and int(value) < 1 << 64, f"invalid exact {label}")


def memory(value):
    fields(value, "scope cpu_bytes gpu_bytes allocations max_cpu_bytes max_gpu_bytes categories", "memory")
    require(value["scope"] == "owned_capacity_reservations_not_RSS_or_driver_memory", "ambiguous memory scope")
    for key in ("cpu_bytes", "gpu_bytes", "allocations", "max_cpu_bytes", "max_gpu_bytes"):
        uint(value[key], key)
    require(value["cpu_bytes"] <= value["max_cpu_bytes"] and value["gpu_bytes"] <= value["max_gpu_bytes"], "global memory limit exceeded")
    seen, cpu, gpu, allocations = set(), 0, 0, 0
    for entry in value["categories"]:
        fields(entry, "category bytes allocations", "memory category")
        name = entry["category"]
        require(name in CATEGORIES and name not in seen, "unknown or duplicate memory category")
        seen.add(name)
        byte_count = uint(entry["bytes"], "category bytes")
        allocations += uint(entry["allocations"], "category allocations")
        if name.endswith("Gpu"):
            gpu += byte_count
        else:
            cpu += byte_count
    require(seen == CATEGORIES and (cpu, gpu, allocations) == (value["cpu_bytes"], value["gpu_bytes"], value["allocations"]), "memory category totals disagree")


def validate(records, *, allow_software=False):
    require(2 <= len(records) <= 10000, "incomplete or oversized evidence")
    session, terminal = records[0], records[-1]
    version2 = session.get("schema") == "bgs-browser-lod-qualification-v2"
    fields(session, "kind schema manifest_url manifest_sha256 wasm_sha256 renderer_revision feature_profile viewport quality camera_path sample_every_frames phase_frames readback_ring_slots readback_ring_bytes readback_scope release_qualified" + (" renderer point_gpu" if version2 else ""), "session")
    require(session["kind"] == "session" and session["schema"] in {"bgs-browser-lod-qualification-v1", "bgs-browser-lod-qualification-v2"}, "missing session identity")
    renderer = session.get("renderer", "quad")
    require(renderer in {"quad", "point_gpu_lod"}, "unknown renderer")
    if renderer == "point_gpu_lod":
        config = session["point_gpu"]
        fields(config, "samples_per_pixel max_projected_gaussians max_points_per_frame max_gpu_bytes max_traversal_gpu_bytes max_frontier_nodes max_visited_nodes max_page_requests", "point configuration")
        for key, value in config.items():
            require(uint(value, key) > 0, "zero point configuration limit")
        require(config["samples_per_pixel"] <= 8 and config["max_projected_gaussians"] <= 0x0fffffff
                and config["max_points_per_frame"] <= 1 << 30 and config["max_frontier_nodes"] <= 65535
                and config["max_visited_nodes"] < 1 << 32 and config["max_page_requests"] <= 1 << 20, "unsupported point/traversal limits")
    elif version2:
        require(session["point_gpu"] is None, "quad session carries point configuration")
    for key in ("manifest_sha256", "wasm_sha256"):
        require(type(session[key]) is str and re.fullmatch(r"[0-9a-f]{64}", session[key]) is not None, f"invalid {key}")
    require(type(session["renderer_revision"]) is str and re.fullmatch(r"[0-9a-f]{40}\+source-sha256-[0-9a-f]{64}", session["renderer_revision"]) is not None, "missing build source identity")
    require(type(session["manifest_url"]) is str and session["manifest_url"].startswith(("http://", "https://")), "invalid manifest URL")
    require(type(session["viewport"]) is list and len(session["viewport"]) == 2
            and all(type(size) is int and 1 <= size <= 4096 for size in session["viewport"]), "invalid session viewport")
    require(type(session["quality"]) in (int, float) and math.isfinite(session["quality"]) and 0 <= session["quality"] <= 1, "invalid quality")
    require(1 <= uint(session["sample_every_frames"], "sample interval") <= 120
            and session["sample_every_frames"] <= uint(session["phase_frames"], "phase duration") <= 3600, "invalid phase sampling")
    fields(session["camera_path"], "from to target", "camera path")
    for position in session["camera_path"].values():
        require(type(position) is list and len(position) == 3
                and all(type(x) in (int, float) and math.isfinite(x) for x in position), "invalid camera path")
    require(session["release_qualified"] is False, "execution evidence cannot qualify a release")
    require(session["readback_ring_slots"] == 3 and session["readback_ring_bytes"] == (288 if version2 else 192), "unrecognized readback capacity")
    require(session["readback_scope"] == "instrumentation_excluded_from_owned_scene_ledger", "ambiguous instrumentation memory scope")
    require(re.fullmatch(r"planar,lod_render,sh[0-4],io_flexbuffers,web_asset,webgpu,testing", session["feature_profile"]) is not None, "feature profile mismatch")
    fields(terminal, "kind frame execution_complete release_qualified reason attested_phases unload_complete mapping_errors invalid_counts dropped_readbacks memory", "terminal")
    require(terminal["kind"] == "terminal" and terminal["execution_complete"] is True
            and terminal["release_qualified"] is False and terminal["unload_complete"] is True
            and terminal["reason"] == "lifecycle_complete", "browser lifecycle incomplete")
    for key in ("mapping_errors", "invalid_counts", "dropped_readbacks"):
        require(uint(terminal[key], key) == 0, f"capture lost evidence: {key}")
    memory(terminal["memory"])
    main, gpu, transitions, adapters, phases = {}, [], [], set(), set()
    adapter_requests, device_requests = [], []
    software = False
    limits = None
    for record in records[1:-1]:
        kind = record.get("kind")
        require(kind in {"main_frame", "gpu_frame", "gpu_point_frame", "gpu_pending", "transition", "browser_adapter", "browser_device"}, f"unsupported or failed evidence: {kind}")
        if kind in ("browser_adapter", "browser_device"):
            require(record.get("wasm_sha256") == session["wasm_sha256"], "adapter/device belongs to different Wasm")
            require(uint(record.get("request_id"), "adapter request identity", 16) > 0, "zero adapter request")
            require(record.get("status") == "returned" and record.get("error") is None, "renderer adapter/device request failed")
            if kind == "browser_adapter":
                fields(record, "kind request_id wasm_sha256 source status options info error", "actual adapter request")
                require(record["source"] == "renderer_request_adapter_promise", "adapter identity came from a separate probe")
                fields(record["options"], "power_preference force_fallback_adapter feature_level", "adapter request options")
                require(record["options"]["power_preference"] in (None, "low-power", "high-performance"), "invalid adapter power preference")
                require(record["options"]["feature_level"] in (None, "core", "compatibility"), "invalid adapter feature level")
                require(type(record["options"]["force_fallback_adapter"]) is bool
                        and (record["options"]["force_fallback_adapter"] is False or allow_software), "renderer requested a fallback adapter")
                fields(record["info"], "vendor architecture device description is_fallback_adapter", "actual adapter info")
                require(type(record["info"]["is_fallback_adapter"]) is bool
                        and (record["info"]["is_fallback_adapter"] is False or allow_software), "hardware adapter fallback identity is missing or true")
                for key in ("vendor", "architecture", "device", "description"):
                    require(type(record["info"][key]) is str and len(record["info"][key]) <= 2048, "invalid actual adapter identity")
                require(any(record["info"][key].strip() for key in ("vendor", "architecture", "device", "description")), "actual adapter identity is empty")
                software |= record["info"]["is_fallback_adapter"] or record["options"]["force_fallback_adapter"] or any(
                    token in " ".join(record["info"][key] for key in ("vendor", "architecture", "device", "description")).lower()
                    for token in ("swiftshader", "llvmpipe", "software"))
                require(not software or allow_software, "actual renderer adapter is software")
                adapter_requests.append(record)
            else:
                fields(record, "kind request_id device_id wasm_sha256 source required_features required_limits status limits error", "actual device request")
                require(record["source"] == "renderer_request_device_promise", "device identity came from a separate probe")
                require(uint(record["device_id"], "device identity", 16) > 0, "zero device identity")
                require(type(record["required_features"]) is list and len(record["required_features"]) <= 128
                        and all(type(feature) is str and 0 < len(feature) <= 256 for feature in record["required_features"])
                        and len(set(record["required_features"])) == len(record["required_features"]), "invalid device features")
                require(type(record["required_limits"]) is dict and len(record["required_limits"]) <= 128, "invalid required device limits")
                for name, limit in record["required_limits"].items():
                    require(type(name) is str and 0 < len(name) <= 128, "invalid device limit name")
                    uint(limit, "required device limit")
                fields(record["limits"], "maxBufferSize maxStorageBufferBindingSize maxStorageBuffersPerShaderStage", "actual device limits")
                for limit in record["limits"].values():
                    require(uint(limit, "actual device limit") > 0, "zero actual device limit")
                device_requests.append(record)
            continue
        frame = uint(record["frame"], "frame")
        if "memory" in record:
            memory(record["memory"])
            observed_limits = (record["memory"]["max_cpu_bytes"], record["memory"]["max_gpu_bytes"])
            require(limits in (None, observed_limits), "memory limits changed during run")
            limits = observed_limits
        if kind == "transition":
            fields(record, "kind frame from to memory", "transition")
            transitions.append(record)
            if record["from"] == "unload":
                require(record["memory"]["cpu_bytes"] == 0 and record["memory"]["gpu_bytes"] == 0, "unload did not retire owned capacity")
            continue
        require(record["phase"] in PHASES[:-1], "unknown observation phase")
        if kind == "main_frame":
            fields(record, "kind frame phase elapsed_ms wall_frame_ms camera_world package memory" + (" gpu_snapshot" if version2 else ""), "main frame")
            if version2 and record["gpu_snapshot"] is not None:
                snapshot = record["gpu_snapshot"]
                require(renderer == "point_gpu_lod", "quad frame carries GPU hierarchy proof")
                fields(snapshot, "generation source_asset cloud", "GPU snapshot")
                identity_integer(snapshot["generation"], "residency generation")
                identity_integer(snapshot["cloud"], "snapshot cloud")
                require(type(snapshot["source_asset"]) is str and 0 < len(snapshot["source_asset"]) < 256, "invalid snapshot source asset")
            require(frame not in main, "duplicate main frame")
            matrix(record["camera_world"], "main camera")
            for key in ("elapsed_ms", "wall_frame_ms"):
                require(type(record[key]) in (int, float) and math.isfinite(record[key]) and record[key] >= 0, "invalid CPU timing")
            package = record["package"]
            if package is not None:
                fields(package, "phase resident_pages selected_gaussians terminal_failures error work", "package")
                require(package["phase"] in {"Loading", "Active", "Degraded", "Failed"}, "invalid package phase")
                for key in ("resident_pages", "selected_gaussians", "terminal_failures"):
                    uint(package[key], key)
                require(package["error"] is None or type(package["error"]) is str, "invalid package error")
                if package["work"] is not None:
                    fields(package["work"], "available queued transport_in_flight preprocess_waiting preprocess_running preprocess_ready", "runtime work")
                    require(type(package["work"]["available"]) is bool, "invalid work availability")
                    for key, count in package["work"].items():
                        if key != "available":
                            uint(count, key)
            main[frame] = record
            continue
        adapter = record["adapter"]
        fields(adapter, "name backend driver driver_info", "adapter")
        require(all(type(value) is str for value in adapter.values()), "invalid adapter strings")
        require(adapter["backend"] == "BrowserWebGpu", "observed backend is not browser WebGPU")
        adapters.add(tuple(adapter[key] for key in ADAPTER_KEYS))
        if kind == "gpu_pending":
            fields(record, "kind frame phase adapter reason", "pending frame")
            continue
        if kind == "gpu_point_frame":
            require(renderer == "point_gpu_lod", "point frame belongs to a quad session")
            validate_point_frame(record, session, phases)
            gpu.append(record)
            continue
        require(renderer == "quad", "quad draw cannot attest a point image")
        fields(record, "kind frame phase view cloud generation compute_input_generation radix_publication_generation candidate_fingerprint_primary candidate_fingerprint_secondary atlas_allocation_epoch source draw_command_attested selected candidates output_capacity camera_world projection viewport adapter memory indirect counts_valid", "GPU frame")
        for key in ("generation", "compute_input_generation", "radix_publication_generation", "atlas_allocation_epoch"):
            identity_integer(record[key], key)
        for key in ("candidate_fingerprint_primary", "candidate_fingerprint_secondary"):
            require(type(record[key]) is str and record[key].isdigit() and int(record[key]) < 1 << 64, f"invalid {key}")
        require(record["source"] == "post_render_same_submission_indirect_copy", "unattested count source")
        matrix(record["camera_world"], "GPU camera")
        matrix(record["projection"], "GPU projection")
        require(record["viewport"] == session["viewport"], "viewport changed")
        selected = uint(record["selected"], "selected", (1 << 32) - 1)
        candidates = uint(record["candidates"], "candidates", (1 << 32) - 1)
        capacity = uint(record["output_capacity"], "output capacity", (1 << 32) - 1)
        indirect = record["indirect"]
        fields(indirect, "vertex_count compacted candidate_hits overflow_count drawn", "indirect")
        compacted = uint(indirect["compacted"], "compacted", (1 << 32) - 1)
        hits = uint(indirect["candidate_hits"], "candidate hits", (1 << 32) - 1)
        require(selected <= candidates and compacted <= hits <= candidates and compacted <= capacity, "selected/compacted/candidate counts disagree")
        require(type(indirect["vertex_count"]) is int and indirect["vertex_count"] == 4
                and type(indirect["overflow_count"]) is int and indirect["overflow_count"] == 0
                and record["counts_valid"] is True, "invalid indirect draw or overflow")
        require(type(record["draw_command_attested"]) is bool, "invalid draw attestation")
        if record["draw_command_attested"]:
            require(uint(indirect["drawn"], "drawn") == compacted, "draw/readback mismatch")
            if compacted > 0:
                phases.add(record["phase"])
        else:
            require(indirect["drawn"] is None, "unattested draw count")
        gpu.append(record)
    require(len(adapters) == 1, "missing or conflicting adapter identity")
    require((not adapter_requests and not device_requests)
            or (len(adapter_requests) == len(device_requests) == 1
                and adapter_requests[0]["request_id"] == device_requests[0]["request_id"]),
            "missing or ambiguous renderer adapter/device request chain")
    require([(r["from"], r["to"]) for r in transitions] == list(zip(PHASES, PHASES[1:])), "missing or reordered lifecycle transitions")
    require(all(a["frame"] < b["frame"] for a, b in zip(transitions, transitions[1:])), "nonmonotonic lifecycle")
    require(uint(terminal["frame"], "terminal frame") >= transitions[-1]["frame"], "terminal precedes completed lifecycle")
    for record in [*main.values(), *gpu]:
        index = PHASES.index(record["phase"])
        first = 0 if index == 0 else transitions[index - 1]["frame"]
        require(first <= record["frame"] < transitions[index]["frame"], "observation contradicts lifecycle phase interval")
    required_phases = {"cold_load", "stationary", "move", "reload"}
    require(required_phases <= phases and set(terminal["attested_phases"]) == phases, "missing observed draws for required phases")
    seen = set()
    point_submissions = set()
    for record in gpu:
        stamp = (record["frame"], record["view"])
        require(stamp not in seen, "duplicate GPU frame/view")
        seen.add(stamp)
        corresponding = main.get(record["frame"])
        require(corresponding is not None and corresponding["phase"] == record["phase"], "GPU/main frame identity mismatch")
        require(corresponding["package"] is not None and corresponding["package"]["terminal_failures"] == 0
                and corresponding["package"]["phase"] != "Failed", "draw lacks successful package ownership")
        if record["kind"] == "gpu_point_frame":
            snapshot = corresponding["gpu_snapshot"]
            require(snapshot == {"generation": record["generation"], "source_asset": record["source_asset"], "cloud": record["cloud"]}, "point image does not match current main-world residency/source")
            stamp = (record["view"], record["point_submission"])
            require(stamp not in point_submissions, "duplicate point image submission")
            point_submissions.add(stamp)
        require(all(abs(a-b) <= 1e-5 * max(1, abs(a), abs(b)) for a,b in zip(record["camera_world"], corresponding["camera_world"])), "GPU/main camera mismatch")
    require(any(r["camera_world"] != gpu[0]["camera_world"] for r in gpu if r["phase"] == "move"), "camera movement was not observed")
    adapter = next(iter(adapters))
    software |= any(token in " ".join(adapter).lower() for token in ("swiftshader", "llvmpipe", "software"))
    require(not software or allow_software, "software adapter cannot qualify hardware WebGPU execution")
    return {"execution_verified": True, "release_qualified": False,
            "renderer": renderer,
            "qualification": "software_diagnostic" if software else "adapter_diagnostic" if allow_software else "hardware_execution" if adapter_requests else "execution_without_adapter_attestation",
            "hardware_execution_verified": bool(adapter_requests) and not software and not allow_software,
            "gpu_frames": len(gpu), "main_frames": len(main), "adapter": dict(zip(ADAPTER_KEYS, adapter)),
            "hardware_adapter_attested": bool(adapter_requests) and not software,
            "hardware_adapter": adapter_requests[0]["info"] if adapter_requests and not software else None,
            "renderer_adapter_request": adapter_requests[0] if adapter_requests else None,
            "renderer_device_request": device_requests[0] if device_requests else None,
            "software_adapter_reported": software,
            "manifest_sha256": session["manifest_sha256"], "wasm_sha256": session["wasm_sha256"],
            "memory_limits": {"cpu": limits[0], "gpu": limits[1]}}


def validate_point_frame(record, session, phases):
    fields(record, "kind frame phase view cloud generation allocation_generation source_asset point_submission traversal_submission source point_image_attested output_capacity samples_per_pixel camera_world projection viewport adapter memory point traversal counts_valid", "point GPU frame")
    for key in ("view", "cloud", "generation", "allocation_generation", "point_submission", "traversal_submission"):
        identity_integer(record[key], key)
    require(type(record["source_asset"]) is str and 0 < len(record["source_asset"]) < 256, "missing point source asset")
    require(record["source"] == "post_render_same_submission_point_and_traversal_copy", "unattested point count source")
    require(record["point_image_attested"] is True and record["counts_valid"] is True, "point image was not completed and attested")
    matrix(record["camera_world"], "GPU camera")
    matrix(record["projection"], "GPU projection")
    require(record["viewport"] == session["viewport"], "viewport changed")
    config = session["point_gpu"]
    require(uint(record["samples_per_pixel"], "sampling layers") == config["samples_per_pixel"], "point sampling policy changed")
    point, traversal = record["point"], record["traversal"]
    fields(point, "projected_gaussians requested_points dispatched_points flags", "point work header")
    fields(traversal, "selected_gaussians visited_nodes requested_pages selected_pages flags", "traversal work header")
    for key, value in [*point.items(), *traversal.items()]:
        uint(value, key, (1 << 32) - 1)
    capacity = uint(record["output_capacity"], "projection output capacity", (1 << 32) - 1)
    require(point["flags"] == 0 and traversal["flags"] & ~31 == 0 and traversal["flags"] & 1 == 0, "point overflow or incomplete hierarchy")
    require(point["projected_gaussians"] <= traversal["selected_gaussians"] <= capacity <= config["max_projected_gaussians"], "point/hierarchy counts exceed admitted records")
    require(point["requested_points"] == point["dispatched_points"] <= config["max_points_per_frame"], "point work is partial or exceeds the point budget")
    require(traversal["visited_nodes"] <= config["max_visited_nodes"]
            and (traversal["requested_pages"] <= config["max_page_requests"] or traversal["flags"] & 8 != 0)
            and traversal["selected_pages"] <= config["max_frontier_nodes"], "traversal feedback exceeds bounded work")
    if point["projected_gaussians"] > 0 and point["dispatched_points"] > 0:
        phases.add(record["phase"])


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def read_records(path):
    records = []
    with path.open(encoding="utf-8") as source:
        while line := source.readline(256 * 1024 + 1):
            require(len(line) <= 256 * 1024 and len(records) < 10000, "capture size ceiling exceeded")
            records.append(json.loads(line, object_pairs_hook=unique_object,
                                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON"))))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=pathlib.Path)
    parser.add_argument("--allow-software", action="store_true", help="Validate a distinctly labeled adapter diagnostic; never hardware execution qualification")
    args = parser.parse_args()
    try:
        print(json.dumps(validate(read_records(args.capture), allow_software=args.allow_software), indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        raise SystemExit(f"browser qualification rejected: {error}") from error


if __name__ == "__main__":
    main()
