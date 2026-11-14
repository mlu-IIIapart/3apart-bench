"""Small correctness controls for the additive CPU100M runner; no performance test."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("cpu_xl_e2e", Path(__file__).with_name("cpu_xl_e2e.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
np = B.np


class FullOutputControls(unittest.TestCase):
    def setUp(self):
        self.layer, self.times, self.values = B.E.inputs(41)
        self.output = np.full((41, 3), np.nan)
        self.ranges = B.place_batches(self.layer, self.times, self.values, self.output, 13)

    def check(self, output=None, ranges=None):
        return B.validate_output(self.layer, self.times, self.values,
                                 self.output if output is None else output,
                                 self.ranges if ranges is None else ranges, chunk=7)

    def test_full_reference_and_short_last_batch(self):
        result = self.check()
        self.assertTrue(result["passed"])
        self.assertEqual(result["markers_checked"], 41)
        self.assertEqual(result["xyz_coordinates_checked"], 123)
        self.assertEqual(result["coverage_ranges"][-1], [39, 2])
        self.assertLess(result["max_abs_error"], 1e-14)

    def test_nonfinite_coordinate_rejected_even_with_valid_z(self):
        self.output[25, 0] = np.nan
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertEqual(result["nonfinite_output_coordinates"], 1)

    def test_finite_wrong_coordinate_rejected(self):
        self.output[40, 1] += 1e-4
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertGreater(result["coordinates_over_tolerance"], 0)

    def test_wrong_global_offset_rejected(self):
        # A batch that resets to marker0 can be finite and in valid bands.
        # The full global-input oracle must still reject the misplaced rows.
        self.output[13:26] = self.output[:13]
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertGreater(result["coordinates_over_tolerance"], 0)

    def test_missing_write_and_duplicate_write_rejected(self):
        for ranges in [self.ranges[:-1], self.ranges + [self.ranges[-1]],
                       [(0, 13), (0, 13), (26, 13), (39, 2)]]:
            with self.subTest(ranges=ranges), self.assertRaisesRegex(ValueError, "coverage"):
                self.check(ranges=ranges)

    def test_silent_unwritten_output_rejected(self):
        self.output[39:] = np.nan
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertEqual(result["nonfinite_output_coordinates"], 6)

    def test_outside_band_and_zero_radius_rejected(self):
        self.output[0] = (0, 0, 1)
        self.output[1] = (0, 0, 0)
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertGreaterEqual(result["outside_band"], 2)
        self.assertEqual(result["invalid_radius"], 1)

    def test_invalid_input_rejected(self):
        self.times[8] = np.inf
        self.assertFalse(self.check()["passed"])

    def test_result_preserves_raw_clocks_and_does_not_overwrite(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = ["--machine", "test", "--output", str(output), "--markers", "41",
                    "--batch-markers", "13", "--warmups", "1", "--runs", "2",
                    "--cold-runs", "2", "--reference-chunk", "7"]
            self.assertEqual(B.main(argv), 0)
            data = json.loads(output.read_text())
            self.assertTrue(data["passed"])
            self.assertEqual(len(data["resident"]["samples"]), 2)
            self.assertEqual(len(data["cold_pipelines"]), 2)
            self.assertTrue(data["resident"]["validation"]["passed"])
            for item in data["cold_pipelines"]:
                self.assertTrue(item["validation"]["passed"])
                self.assertGreater(item["end_to_end_ms"], 0)
                self.assertGreaterEqual(item["end_to_end_ms"], item["placement_and_complete_output_materialization_ms"])
            original = output.read_bytes()
            with self.assertRaises(SystemExit):
                B.main(argv)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
