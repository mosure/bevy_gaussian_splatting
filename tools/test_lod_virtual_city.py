import copy
import unittest

from check_lod_capture import CaptureError
from check_lod_virtual_city import compare


def fixture(source, drawn=100):
    config = dict(records_per_page=4096, grid_width=128, seed=12, viewport=[960, 540],
                  quality=.9, max_active_gaussians=65536, max_resident_pages=64,
                  max_cpu_bytes=2**30, max_gpu_bytes=2**29, phase_frames=180,
                  capture_every=8)
    camera = dict(world_to_view=[1.] * 16, projection=[2.] * 16, viewport=[960, 540])
    row = dict(camera=camera, counts=dict(drawn=drawn), timings=dict(frame_wall_ms=4.))
    return dict(source_gaussians=source), {phase: [copy.deepcopy(row) for _ in range(3)]
                                         for phase in ("stationary", "reload")}, config


class ScalingTests(unittest.TestCase):
    def test_matched_draws_compare_without_claiming_release_qualification(self):
        result = compare(fixture(10_000_000), fixture(100_000_000))
        self.assertTrue(result["matched_visible_work"])
        self.assertFalse(result["release_qualified"])

    def test_larger_source_with_less_draw_work_is_not_a_scaling_result(self):
        with self.assertRaises(CaptureError):
            compare(fixture(10_000_000), fixture(100_000_000, 60))

    def test_mismatched_cameras_and_budgets_fail_closed(self):
        changed = fixture(100_000_000)
        changed[2]["max_resident_pages"] *= 2
        with self.assertRaises(CaptureError):
            compare(fixture(10_000_000), changed)
        changed = fixture(100_000_000)
        changed[1]["reload"][0]["camera"]["world_to_view"][0] = 9
        with self.assertRaises(CaptureError):
            compare(fixture(10_000_000), changed)

    def test_nonfinite_tolerance_and_wrong_extent_order_fail(self):
        for tolerance in (float("nan"), float("inf"), True, -1):
            with self.assertRaises(CaptureError):
                compare(fixture(10), fixture(100), tolerance)
        with self.assertRaises(CaptureError):
            compare(fixture(100), fixture(10))


if __name__ == "__main__":
    unittest.main()
