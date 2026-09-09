"""CPU-only contract tests; every observation below is synthetic test data."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_lod_capture as capture

FIXTURE = Path(__file__).parent / "fixtures/lod_capture_synthetic.jsonl"


def fixture():
    return json.loads(FIXTURE.read_text())


def structurally_complete_gpu_fixture():
    # This producer label is deliberately exercised only as a schema fixture.
    # It is never saved as a measured result or used to qualify the renderer.
    record = fixture()
    record["mode"] = "native_gpu"
    record["counts"]["source"] = "gpu_readback"
    record["identity"].update(backend="vulkan", adapter="test adapter", driver="test driver", instrumentation="test readback")
    record["timings"]["gpu_ms"] = {"raster": 0.2}
    record["memory"]["cpu"] = {name: {"reserved": 0, "used": 0} for name in capture.CPU_CATEGORIES}
    record["memory"]["gpu"] = {name: {"reserved": 0, "used": 0} for name in capture.GPU_CATEGORIES}
    record["memory"].update(process_rss_bytes=1, device_used_bytes=1)
    return record


class CaptureTests(unittest.TestCase):
    def test_gpu_hierarchy_backends_require_matching_compacted_and_drawn_counts(self):
        for pipeline in ("hierarchy_point", "hierarchy_ordered"):
            with self.subTest(pipeline=pipeline):
                record = structurally_complete_gpu_fixture()
                record["counts"]["pipeline"] = pipeline
                capture.validate(record)
                record["counts"]["drawn"] = 4096
                with self.assertRaises(capture.CaptureError):
                    capture.validate(record)

    def test_flat_source_reports_draw_without_inventing_compaction(self):
        record = structurally_complete_gpu_fixture()
        record["counts"].update(pipeline="flat_source", compacted=None)
        self.assertEqual(capture.missing_gpu_evidence(record), [])
        record["counts"]["compacted"] = record["counts"]["drawn"]
        with self.assertRaises(capture.CaptureError):
            capture.validate(record)

    def test_shared_synthetic_fixture_is_diagnostic_only(self):
        records = capture.read_captures(FIXTURE)
        summary = capture.summarize(records)
        self.assertTrue(summary["valid_capture"])
        self.assertFalse(summary["release_qualified"])
        self.assertFalse(summary["groups"][0]["gpu_evidence_complete"])
        self.assertIsNone(summary["groups"][0]["peak_recorded_memory"]["cpu_reserved_bytes"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(capture.main([str(FIXTURE)]), 0)
            self.assertEqual(capture.main([str(FIXTURE), "--require-gpu-evidence"]), 2)
            self.assertEqual(capture.main([str(FIXTURE), "--require-scenario", "motion"]), 2)
            self.assertEqual(capture.main([str(FIXTURE), "--minimum-frames", "2"]), 2)

    def test_gpu_field_completeness_still_does_not_qualify_release(self):
        record = capture.validate(structurally_complete_gpu_fixture())
        self.assertEqual(capture.missing_gpu_evidence(record), [])
        self.assertFalse(capture.summarize([record])["release_qualified"])
        del record["memory"]["gpu"]["retired"]
        self.assertIn("memory.gpu.retired", capture.missing_gpu_evidence(record))

    def test_completeness_and_summary_fail_closed_without_prior_validation(self):
        record = structurally_complete_gpu_fixture()
        record["counts"]["drawn"] += 1
        self.assertTrue(capture.missing_gpu_evidence(record)[0].startswith("invalid capture:"))
        with self.assertRaises(capture.CaptureError):
            capture.summarize([record])
        with self.assertRaises(capture.CaptureError):
            capture.summarize([])
        record = fixture()
        with self.assertRaises(capture.CaptureError):
            capture.summarize([record, record])

    def test_unknown_fields_are_rejected_at_every_object_level(self):
        for field in (None, "identity", "stamp", "camera", "counts", "timings", "memory", "image"):
            record = fixture()
            (record if field is None else record[field])["unknown"] = 0
            with self.subTest(field=field), self.assertRaises(capture.CaptureError):
                capture.validate(record)
        record = fixture()
        record["memory"]["cpu"]["test"] = {"reserved": 0, "used": 0, "unknown": 0}
        with self.assertRaises(capture.CaptureError):
            capture.validate(record)

    def test_boolean_and_oversized_numbers_do_not_coerce(self):
        for section, field, values in (
            (None, "schema_version", [True]),
            ("stamp", "frame", [True, capture.U64_MAX + 1]),
            ("counts", "selected", [True, capture.U64_MAX + 1]),
            ("timings", "frame_wall_ms", [True, 10 ** 400]),
            ("camera", "pixel_scale", [True, 10 ** 400]),
            ("memory", "process_rss_bytes", [True, capture.U64_MAX + 1]),
        ):
            for value in values:
                record = fixture()
                (record if section is None else record[section])[field] = value
                with self.subTest(section=section, field=field, value=value), self.assertRaises(capture.CaptureError):
                    capture.validate(record)
        for value in (True, float("inf"), 10 ** 400):
            record = fixture()
            record["camera"]["projection"][0] = value
            with self.assertRaises(capture.CaptureError):
                capture.validate(record)
        for value in (True, 1 << 32):
            record = fixture()
            record["camera"]["viewport"][0] = value
            with self.assertRaises(capture.CaptureError):
                capture.validate(record)

    def test_conflicting_backend_and_invalid_unicode_are_rejected(self):
        record = structurally_complete_gpu_fixture()
        record["identity"]["backend"] = "synthetic"
        with self.assertRaisesRegex(capture.CaptureError, "backend"):
            capture.validate(record)
        for field in ("adapter", "backend"):
            record = fixture()
            record["identity"][field] = "\ud800"
            with self.assertRaises(capture.CaptureError):
                capture.validate(record)
        record = fixture()
        record["image"]["path"] = "contains\0nul"
        with self.assertRaises(capture.CaptureError):
            capture.validate(record)

    def test_invalid_count_relationships_and_integer_overflow(self):
        mutations = [
            {"drawn": 9}, {"selected": capture.U64_MAX}, {"compacted": 13},
            {"compacted": None}, {"candidates": -1}, {"candidates": True},
            {"source": "gpu_readback"}, {"output_capacity": 7},
        ]
        for fields in mutations:
            with self.subTest(fields=fields):
                record = fixture()
                record["counts"].update(fields)
                with self.assertRaises(capture.CaptureError):
                    capture.validate(record)

    def test_stale_observations_and_nonfinite_values(self):
        for field in ("counts", "timings", "memory", "image"):
            with self.subTest(field=field):
                record = fixture()
                record[field]["stamp"]["generation"] += 1
                with self.assertRaisesRegex(capture.CaptureError, "stamp mismatch"):
                    capture.validate(record)
        for value in (float("nan"), float("inf"), -1, True):
            record = fixture()
            record["timings"]["frame_wall_ms"] = value
            with self.assertRaises(capture.CaptureError):
                capture.validate(record)

    def test_memory_ledger_rejects_overflow_and_double_capacity(self):
        for cpu in (
            {"a": {"reserved": capture.U64_MAX, "used": 0}, "b": {"reserved": 1, "used": 0}},
            {"a": {"reserved": 1, "used": 2}},
        ):
            record = fixture()
            record["memory"]["cpu"] = cpu
            with self.assertRaises(capture.CaptureError):
                capture.validate(record)

    def test_duplicate_keys_frames_and_run_identity_are_rejected(self):
        original = fixture()
        changed = copy.deepcopy(original)
        changed["stamp"]["frame"] += 1
        for field in ("counts", "timings", "memory", "image"):
            changed[field]["stamp"] = copy.deepcopy(changed["stamp"])
        changed["identity"]["manifest_sha256"] = "f" * 64
        lines = [
            json.dumps(original) + "\n" + json.dumps(original),
            json.dumps(original) + "\n" + json.dumps(changed),
            '{"schema_version":1,"schema_version":1}',
            json.dumps(original).replace('"fixture": 0.1', '"fixture": 0.1, "fixture": 0.2'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            for value in lines:
                path.write_text(value)
                with self.assertRaises(capture.CaptureError):
                    capture.read_captures(path)

    def test_null_optional_identity_matches_omitted_and_read_size_is_bounded(self):
        first, second = fixture(), fixture()
        second["stamp"]["frame"] = 1
        for field in ("counts", "timings", "memory", "image"):
            second[field]["stamp"] = copy.deepcopy(second["stamp"])
        second["identity"].pop("adapter")
        second["identity"].pop("driver")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text(json.dumps(first) + "\n" + json.dumps(second))
            self.assertEqual(len(capture.read_captures(path)), 2)
            with patch.object(capture, "MAX_CAPTURE_LINE_BYTES", 32):
                with self.assertRaisesRegex(capture.CaptureError, "exceeds"):
                    capture.read_captures(path)
            for malformed in ("[" * 2000, '{"x":NaN}', '{"x":Infinity}', '{"x":1e400}'):
                path.write_text(malformed)
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(capture.main([str(path)]), 1)

    def test_image_identity_can_be_verified_without_decoding_or_gpu(self):
        record = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            image = Path(directory) / record["image"]["path"]
            image.write_bytes(b"synthetic image bytes")
            record["image"]["sha256"] = capture.hashlib.sha256(image.read_bytes()).hexdigest()
            capture.verify_images([record], path)
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(capture.CaptureError, "SHA-256 mismatch"):
                capture.verify_images([record], path)

    def test_percentiles_separate_view_and_scenario(self):
        records = []
        for frame, duration in enumerate((1.0, 2.0, 100.0)):
            record = fixture()
            record["stamp"]["frame"] = frame
            for field in ("counts", "timings", "memory", "image"):
                record[field]["stamp"] = copy.deepcopy(record["stamp"])
            record["timings"]["frame_wall_ms"] = duration
            records.append(record)
        other = fixture()
        other["stamp"]["frame"] = 3
        for field in ("counts", "timings", "memory", "image"):
            other[field]["stamp"] = copy.deepcopy(other["stamp"])
        other["scenario"] = "cold_start"
        records.append(other)
        summary = capture.summarize(records)
        warm = next(group for group in summary["groups"] if group["scenario"] == "warm_stationary")
        self.assertEqual(warm["frame_wall_ms"], {"p50": 2.0, "p95": 100.0, "p99": 100.0, "over_50ms": 1})
        self.assertEqual(len(summary["groups"]), 2)


if __name__ == "__main__":
    unittest.main()
