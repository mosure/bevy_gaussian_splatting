"""Synthetic CPU-only protocol tests; these fixtures are not browser observations."""
import copy
import pathlib
import tempfile
import unittest

from check_lod_browser_capture import CATEGORIES, PHASES, unique_object, validate
from serve_lod_browser_capture import byte_range, safe_file


def fixture():
    matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1]
    memory = {"scope": "owned_capacity_reservations_not_RSS_or_driver_memory",
              "cpu_bytes": 0, "gpu_bytes": 0, "allocations": 0,
              "max_cpu_bytes": 1000, "max_gpu_bytes": 1000,
              "categories": [{"category": name, "bytes": 0, "allocations": 0} for name in sorted(CATEGORIES)]}
    adapter = {"name": "synthetic protocol fixture", "backend": "BrowserWebGpu", "driver": "", "driver_info": ""}
    session = {"kind": "session", "schema": "bgs-browser-lod-qualification-v1",
               "manifest_url": "http://localhost/package/scene.gsplatlod", "manifest_sha256": "a" * 64,
               "wasm_sha256": "b" * 64, "renderer_revision": "d" * 40 + "+source-sha256-" + "c" * 64,
               "feature_profile": "planar,lod_render,sh3,io_flexbuffers,web_asset,webgpu,testing",
               "viewport": [960, 540], "quality": 0, "camera_path": {"from": [0, 0, 5], "to": [2, 0, 4], "target": [0, 0, 0]},
               "sample_every_frames": 10, "phase_frames": 120, "readback_ring_slots": 3, "readback_ring_bytes": 192,
               "readback_scope": "instrumentation_excluded_from_owned_scene_ledger", "release_qualified": False}
    records = [session]
    for index, phase in enumerate(PHASES[:-1]):
        frame = index * 100 + 10
        camera = matrix.copy()
        camera[12] = index if index > 1 else 0
        package = {"phase": "Active", "resident_pages": 1, "selected_gaussians": 10,
                   "terminal_failures": 0, "error": None, "work": None} if phase != "unload" else None
        records.append({"kind": "main_frame", "frame": frame, "phase": phase, "elapsed_ms": frame,
                        "wall_frame_ms": 16, "camera_world": camera, "package": package, "memory": memory})
        if phase != "unload":
            records.append({"kind": "gpu_frame", "frame": frame, "phase": phase, "view": "1", "cloud": "2",
                            "generation": "1", "compute_input_generation": "2", "radix_publication_generation": "3",
                            "candidate_fingerprint_primary": "12345678901234567890", "candidate_fingerprint_secondary": "4",
                            "atlas_allocation_epoch": "1", "source": "post_render_same_submission_indirect_copy",
                            "draw_command_attested": True, "selected": 10, "candidates": 10, "output_capacity": 16,
                            "camera_world": camera, "projection": matrix, "viewport": [960, 540], "adapter": adapter,
                            "memory": memory, "indirect": {"vertex_count": 4, "compacted": 8, "candidate_hits": 8, "overflow_count": 0, "drawn": 8}, "counts_valid": True})
        records.append({"kind": "transition", "frame": frame + 1, "from": phase, "to": PHASES[index + 1], "memory": memory})
    records.append({"kind": "terminal", "frame": 500, "execution_complete": True, "release_qualified": False,
                    "reason": "lifecycle_complete", "attested_phases": ["cold_load", "stationary", "move", "reload"],
                    "unload_complete": True, "mapping_errors": 0, "invalid_counts": 0, "dropped_readbacks": 0, "memory": memory})
    return copy.deepcopy(records)


def point_fixture():
    records = fixture()
    config = {"samples_per_pixel": 1, "max_projected_gaussians": 32,
              "max_points_per_frame": 262144, "max_gpu_bytes": 4194304,
              "max_traversal_gpu_bytes": 2097152, "max_frontier_nodes": 64,
              "max_visited_nodes": 256, "max_page_requests": 16}
    records[0].update(schema="bgs-browser-lod-qualification-v2", renderer="point_gpu_lod",
                      point_gpu=config, readback_ring_bytes=288)
    for index, record in enumerate(records):
        source = "atlas-reloaded" if record.get("phase") == "reload" else "atlas-first"
        if record["kind"] == "main_frame":
            record["gpu_snapshot"] = ({"generation": "1", "source_asset": source, "cloud": "2"}
                                      if record["package"] is not None else None)
        elif record["kind"] == "gpu_frame":
            common = {key: record[key] for key in ("frame", "phase", "view", "cloud", "generation",
                                                  "camera_world", "projection", "viewport", "adapter", "memory")}
            records[index] = dict(common, kind="gpu_point_frame", allocation_generation="1", source_asset=source,
                                  point_submission=str(record["frame"]), traversal_submission=str(record["frame"]),
                                  source="post_render_same_submission_point_and_traversal_copy",
                                  point_image_attested=True, output_capacity=32, samples_per_pixel=1,
                                  point={"projected_gaussians": 8, "requested_points": 500, "dispatched_points": 500, "flags": 0},
                                  traversal={"selected_gaussians": 10, "visited_nodes": 9, "requested_pages": 2, "selected_pages": 3, "flags": 0},
                                  counts_valid=True)
    return records


class BrowserEvidenceTests(unittest.TestCase):
    def test_empty_point_image_cannot_attest_a_lifecycle_phase(self):
        rows = point_fixture()
        for index, record in enumerate(rows):
            if record["kind"] == "gpu_point_frame" and record["phase"] == "reload":
                rows[index] = {"kind": "gpu_pending", "frame": record["frame"],
                               "phase": "reload", "adapter": record["adapter"],
                               "reason": "no_point_hierarchy_input"}
        with self.assertRaises(ValueError):
            validate(rows)

    def test_point_protocol_attests_current_snapshot_without_quad_draw_claims(self):
        rows = point_fixture()
        # Visiting a parent may request its entire child cohort without visiting
        # those children. Page requests have their own independent work limit.
        rows[0]["point_gpu"]["max_visited_nodes"] = 1
        for record in rows:
            if record["kind"] == "gpu_point_frame":
                record["traversal"].update(visited_nodes=1, requested_pages=8)
        result = validate(rows)
        self.assertEqual(result["renderer"], "point_gpu_lod")
        self.assertTrue(result["execution_verified"])
        self.assertFalse(result["hardware_execution_verified"])
        self.assertFalse(result["release_qualified"])

    def test_point_partial_work_stale_snapshots_and_quad_proofs_fail_closed(self):
        mutations = [
            lambda rows: rows[2].__setitem__("generation", "2"),
            lambda rows: rows[2].__setitem__("source_asset", "stale-atlas"),
            lambda rows: rows[2].__setitem__("source", "post_render_same_submission_indirect_copy"),
            lambda rows: rows[2].__setitem__("draw_command_attested", True),
            lambda rows: rows[2]["point"].__setitem__("dispatched_points", 499),
            lambda rows: rows[2]["point"].__setitem__("flags", 1),
            lambda rows: rows[2]["traversal"].__setitem__("flags", 1),
            lambda rows: rows[2]["point"].__setitem__("projected_gaussians", 33),
            lambda rows: rows[2]["traversal"].__setitem__("visited_nodes", 257),
            lambda rows: rows[2]["traversal"].__setitem__("requested_pages", 17),
            lambda rows: rows[1].__setitem__("gpu_snapshot", None),
            lambda rows: rows[5].__setitem__("point_submission", rows[2]["point_submission"]),
        ]
        for edit in mutations:
            rows = point_fixture()
            edit(rows)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate(rows)

    def test_explicit_software_mode_preserves_provenance_and_never_qualifies_hardware(self):
        rows = point_fixture()
        actual = actual_request_fixture()
        actual[0]["info"].update(vendor="SwiftShader", is_fallback_adapter=True)
        rows[1:1] = actual
        with self.assertRaises(ValueError):
            validate(rows)
        result = validate(rows, allow_software=True)
        self.assertEqual(result["qualification"], "software_diagnostic")
        self.assertFalse(result["hardware_adapter_attested"])
        self.assertFalse(result["hardware_execution_verified"])
        self.assertFalse(result["release_qualified"])
        rows[1]["source"] = "separate_adapter_probe"
        with self.assertRaises(ValueError):
            validate(rows, allow_software=True)

    def test_synthetic_protocol_fixture_has_no_release_promotion(self):
        self.assertTrue(validate(fixture())["execution_verified"])
        self.assertFalse(validate(fixture())["release_qualified"])
        current = fixture()
        current[0].update(schema="bgs-browser-lod-qualification-v2", renderer="quad", point_gpu=None, readback_ring_bytes=288)
        for record in current:
            if record["kind"] == "main_frame":
                record["gpu_snapshot"] = None
        self.assertEqual(validate(current)["renderer"], "quad")

    def test_adapter_summary_uses_keys_independent_of_json_field_order(self):
        records = fixture()
        for record in records:
            if "adapter" in record:
                record["adapter"] = {key: record["adapter"][key] for key in ("backend", "driver", "driver_info", "name")}
        summary = validate(records)
        self.assertEqual(summary["adapter"]["backend"], "BrowserWebGpu")
        self.assertEqual(summary["adapter"]["name"], "synthetic protocol fixture")
        self.assertFalse(summary["hardware_adapter_attested"])

    def test_actual_renderer_adapter_device_chain_attests_hardware(self):
        records = fixture()
        records[1:1] = actual_request_fixture()
        summary = validate(records)
        self.assertTrue(summary["hardware_adapter_attested"])
        self.assertEqual(summary["hardware_adapter"]["vendor"], "synthetic protocol vendor")
        self.assertEqual(summary["renderer_device_request"]["limits"]["maxBufferSize"], 2**30)

    def test_separate_probe_fallback_and_conflicting_request_chains_fail(self):
        mutations = [
            lambda rows: rows[0].__setitem__("source", "separate_adapter_probe"),
            lambda rows: rows[0]["info"].__setitem__("is_fallback_adapter", True),
            lambda rows: rows[0]["info"].__setitem__("is_fallback_adapter", None),
            lambda rows: rows[0]["info"].__setitem__("vendor", "SwiftShader"),
            lambda rows: rows[0].__setitem__("wasm_sha256", "c" * 64),
            lambda rows: rows[1].__setitem__("request_id", 2),
            lambda rows: rows[1]["limits"].__setitem__("maxBufferSize", True),
            lambda rows: rows.pop(),
            lambda rows: rows.append(copy.deepcopy(rows[0])),
        ]
        for edit in mutations:
            records = fixture()
            requests = actual_request_fixture()
            edit(requests)
            records[1:1] = requests
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate(records)

    def test_bad_counts_provenance_identity_and_incomplete_lifecycle(self):
        for path, value in [("draw_command_attested", False), ("generation", 1), ("selected", True),
                            ("candidates", 1), ("source", "cpu_estimate"), ("extra", 1)]:
            records = fixture()
            records[2][path] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate(records)
        records = fixture()
        records[-1]["execution_complete"] = False
        with self.assertRaises(ValueError):
            validate(records)

    def test_nonfinite_memory_and_frame_mismatch_fail_closed(self):
        for edit in (lambda r: r[2]["camera_world"].__setitem__(0, float("nan")),
                     lambda r: r[2].__setitem__("frame", 999),
                     lambda r: r[-1]["memory"].__setitem__("cpu_bytes", 1),
                     lambda r: r[2]["indirect"].__setitem__("overflow_count", 1)):
            records = fixture()
            edit(records)
            with self.assertRaises(ValueError):
                validate(records)
        with self.assertRaises(ValueError):
            unique_object([("frame", 1), ("frame", 2)])

    def test_software_adapter_and_false_package_claims_are_rejected(self):
        records = fixture()
        for record in records:
            if "adapter" in record:
                record["adapter"]["name"] = "SwiftShader Device"
        with self.assertRaises(ValueError):
            validate(records)
        records = fixture()
        records[1]["package"] = None
        with self.assertRaises(ValueError):
            validate(records)

    def test_exact_ranges_and_path_confinement(self):
        self.assertEqual(byte_range("bytes=2-5", 8), (2, 5))
        for value in ("bytes=2-8", "bytes=4-3", "bytes=-5", "bytes=1-2,4-5"):
            with self.assertRaises(ValueError):
                byte_range(value, 8)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "page").write_bytes(b"tiny")
            self.assertEqual(safe_file(root, "page"), root / "page")
            with self.assertRaises(FileNotFoundError):
                safe_file(root, "../escaped")


def actual_request_fixture():
    return [
        {"kind": "browser_adapter", "request_id": 1, "wasm_sha256": "b" * 64,
         "source": "renderer_request_adapter_promise", "status": "returned", "error": None,
         "options": {"power_preference": "high-performance", "force_fallback_adapter": False, "feature_level": None},
         "info": {"vendor": "synthetic protocol vendor", "architecture": "synthetic protocol architecture",
                  "device": "", "description": "", "is_fallback_adapter": False}},
        {"kind": "browser_device", "request_id": 1, "device_id": 1, "wasm_sha256": "b" * 64,
         "source": "renderer_request_device_promise", "status": "returned", "error": None,
         "required_features": [], "required_limits": {"maxBindGroups": 4},
         "limits": {"maxBufferSize": 2**30, "maxStorageBufferBindingSize": 2**28, "maxStorageBuffersPerShaderStage": 8}},
    ]


if __name__ == "__main__":
    unittest.main()
