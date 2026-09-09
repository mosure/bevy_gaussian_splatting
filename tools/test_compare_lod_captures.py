from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from check_lod_capture import CaptureError
from compare_lod_captures import compare, index_capture, metrics, require_attested_image


def renderer_receipt(prefix):
    identity = {"cloud": "cloud", "source_asset": "asset", "residency_generation": 7,
                "allocation_generation": 3}
    producer = identity | {"submission": 17}
    receipt = "ordered_draw_receipt" if prefix == "ordered" else "point_image_receipt"
    return identity | {
        "frame": 11, "path_frame": 3, "pipeline": f"hierarchy_{prefix}",
        "counts_valid": True, receipt: True, f"{prefix}_image_attested": True,
        f"{prefix}_submission": 9, "traversal_submission": 17,
        "renderer_traversal_mapping": {
            "schema_version": 1, "association": "current_render_encoded_inputs",
            "renderer_submission": 9, "renderer_inputs": [producer.copy()],
            "traversal_output": producer.copy(),
        },
    }


class ImageMetricsTests(unittest.TestCase):
    def test_multiview_pairing_requires_unambiguous_receipts_and_retains_each_view(self):
        records = [{"scenario": "held", "stamp": {"run_id": "run", "view_id": view,
                    "frame": 11, "generation": 7}, "image": {"path": "frame.png"},
                    "counts": {"pipeline": "hierarchy_ordered", "source": "gpu_readback", "drawn": 1}}
                   for view in ("left", "right")]
        base = renderer_receipt("ordered")
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "capture.jsonl"
            receipts = capture.with_name("submission_evidence.jsonl")
            with patch("compare_lod_captures.read_captures", return_value=records), \
                    patch("compare_lod_captures.verify_images"):
                receipts.write_text(json.dumps(base) + "\n")
                with self.assertRaisesRegex(CaptureError, "ambiguous submission receipt"):
                    index_capture(capture)
                receipts.write_text("".join(json.dumps(base | {"view_id": view}) + "\n"
                                            for view in ("right", "left")))
                indexed = index_capture(capture)
                self.assertEqual(set(indexed), {("held", 3, "left"), ("held", 3, "right")})
                self.assertEqual(indexed[("held", 3, "left")]["stamp"]["view_id"], "left")

    def test_complete_pair_gate_detects_missing_views_and_allows_sampled_diagnostics(self):
        def record(view):
            return {"stamp": {"run_id": "run", "view_id": view}, "camera": {"viewport": [21, 21]},
                    "image": {"path": "image.png"}, "counts": {"drawn": 1},
                    "identity": {"source_sha256": "a" * 64, "camera_path_sha256": "b" * 64}}
        reference = {("held", 3, view): record(view) for view in ("left", "right")}
        candidate = {("held", 3, "left"): record("left"), ("held", 4, "right"): record("right")}
        image = (np.full((21, 21, 3), .5), np.ones((21, 21)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.jsonl"
            path.write_text("capture identity")
            with patch("compare_lod_captures.index_capture", side_effect=[reference, candidate]):
                with self.assertRaisesRegex(CaptureError, "incomplete logical sample pairs"):
                    compare(path, path, require_complete_pairs=True)
            with patch("compare_lod_captures.index_capture", side_effect=[reference, candidate]), \
                    patch("compare_lod_captures.rgba", return_value=image):
                report = compare(path, path)
            self.assertEqual(report["paired_samples"], 1)
            self.assertEqual(report["view_pairing"], "matching_view_id")
            self.assertFalse(report["complete_pairs"])
            self.assertFalse(report["release_qualified"])
            with patch("compare_lod_captures.index_capture", side_effect=[reference, reference]), \
                    patch("compare_lod_captures.rgba", return_value=image):
                report = compare(path, path, require_complete_pairs=True)
            self.assertEqual(report["paired_samples"], 2)
            self.assertTrue(report["complete_pairs"])

    def test_pipeline_receipts_reject_cross_submission_and_unattested_images(self):
        for prefix in ("ordered", "point"):
            pipeline = f"hierarchy_{prefix}"
            record = {"counts": {"pipeline": pipeline, "source": "gpu_readback"},
                      "stamp": {"frame": 11, "generation": 7}}
            receipt = "ordered_draw_receipt" if prefix == "ordered" else "point_image_receipt"
            evidence = renderer_receipt(prefix)
            # The renderer and traversal counters advance independently.
            require_attested_image(record, evidence)
            for field, invalid in (("frame", 12), ("residency_generation", 8),
                                   ("traversal_submission", 10), ("pipeline", "hierarchy"),
                                   (f"{prefix}_submission", 10), ("allocation_generation", 4),
                                   ("cloud", "other"), ("source_asset", "other"),
                                   ("renderer_traversal_mapping", None),
                                   ("counts_valid", False), (receipt, False),
                                   (f"{prefix}_image_attested", False)):
                with self.subTest(pipeline=pipeline, field=field), self.assertRaises(CaptureError):
                    require_attested_image(record, evidence | {field: invalid})
            for field, invalid in (("submission", 16), ("residency_generation", 6),
                                   ("allocation_generation", 2), ("cloud", "other"),
                                   ("source_asset", "other")):
                stale = deepcopy(evidence)
                stale["renderer_traversal_mapping"]["renderer_inputs"][0][field] = invalid
                with self.subTest(pipeline=pipeline, input_field=field), self.assertRaises(CaptureError):
                    require_attested_image(record, stale)
            # Equal numeric counters do not qualify older receipts lacking the
            # actual encoded input association, even with a valid count header.
            historical = evidence | {"traversal_submission": 9}
            historical.pop("renderer_traversal_mapping")
            with self.assertRaisesRegex(CaptureError, "missing explicit"):
                require_attested_image(record, historical)

    def test_exact_output_has_zero_error_and_unit_similarity(self):
        rng = np.random.default_rng(614)
        rgb = rng.random((23, 29, 3))
        alpha = np.ones((23, 29))
        result = metrics(rgb, rgb, alpha, alpha)
        self.assertEqual(result["foreground_mse"], 0)
        self.assertIsNone(result["foreground_psnr_db"])
        self.assertEqual(result["foreground_ssim"], 1)

    def test_constant_fields_match_analytic_psnr_and_ssim(self):
        reference = np.full((21, 21, 3), 0.25)
        candidate = np.full_like(reference, 0.5)
        alpha = np.ones((21, 21))
        result = metrics(reference, candidate, alpha, alpha)
        self.assertAlmostEqual(result["foreground_psnr_db"], 12.041199826559248)
        self.assertAlmostEqual(result["foreground_ssim"],
                               (2 * 0.25 * 0.5 + 0.01**2) /
                               (0.25**2 + 0.5**2 + 0.01**2))

    def test_union_mask_penalizes_missing_and_spilled_silhouettes(self):
        reference = np.zeros((21, 21, 3))
        candidate = np.zeros_like(reference)
        reference[8:12, 8:12] = 0.5
        candidate[8:12, 10:14] = 0.5
        reference_alpha = (reference[..., 0] > 0).astype(float)
        candidate_alpha = (candidate[..., 0] > 0).astype(float)
        result = metrics(reference, candidate, reference_alpha, candidate_alpha)
        self.assertEqual(result["foreground_pixels"], 24)
        self.assertAlmostEqual(result["silhouette_iou"], 1 / 3)
        self.assertAlmostEqual(result["foreground_mse"], 1 / 6)

    def test_empty_render_is_rejected(self):
        rgb = np.zeros((21, 21, 3))
        alpha = np.zeros((21, 21))
        with self.assertRaisesRegex(CaptureError, "empty foreground"):
            metrics(rgb, rgb, alpha, alpha)


if __name__ == "__main__":
    unittest.main()
