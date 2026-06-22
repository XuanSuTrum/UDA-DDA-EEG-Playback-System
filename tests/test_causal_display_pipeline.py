import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "app"))

import generate_upper_demo_predictions_lds as generator
import model_adapters
import test_offline_lds as offline_diagnostic


DISPLAY_FIELDS = [
    "display_prob_negative",
    "display_prob_non_negative",
    "display_negative_score",
    "display_state",
]


def prediction_frame(probabilities, trial_ids, labels=None, scaler_mode="calibration_feature"):
    if labels is None:
        labels = ["negative"] * len(probabilities)
    return pd.DataFrame(
        {
            "window_index": np.arange(len(probabilities)),
            "trial_id": trial_ids,
            "true_label_name": labels,
            "prob_negative_calibrated": probabilities,
            "scaler_mode": scaler_mode,
        }
    )


class CausalDisplayTests(unittest.TestCase):
    def test_future_windows_do_not_change_past_display_results(self):
        base = prediction_frame(
            [0.10, 0.20, 0.30, 0.40, 0.90, 0.95],
            [4, 4, 4, 4, 4, 4],
        )
        changed = base.copy()
        changed.loc[4:, "prob_negative_calibrated"] = [0.01, 0.02]

        base_display = generator.build_display_df(base)
        changed_display = generator.build_display_df(changed)

        pd.testing.assert_frame_equal(
            base_display.loc[:3, DISPLAY_FIELDS],
            changed_display.loc[:3, DISPLAY_FIELDS],
        )

    def test_new_trial_resets_rolling_history(self):
        frame = prediction_frame(
            [0.90, 0.80, 0.70, 0.20, 0.30],
            [4, 4, 4, 5, 5],
        )

        display = generator.build_display_df(frame)

        self.assertAlmostEqual(display.loc[3, "display_prob_negative"], 0.20)
        self.assertEqual(display.loc[3, "display_state"], "非负性")

    def test_true_label_shuffle_does_not_change_display_fields(self):
        frame = prediction_frame(
            [0.20, 0.80, 0.60, 0.10],
            [4, 4, 5, 5],
            ["negative", "negative", "neutral", "neutral"],
        )
        shuffled = frame.copy()
        shuffled["true_label_name"] = ["positive", "neutral", "negative", "positive"]

        original_display = generator.build_display_df(frame)
        shuffled_display = generator.build_display_df(shuffled)

        pd.testing.assert_frame_equal(
            original_display[DISPLAY_FIELDS],
            shuffled_display[DISPLAY_FIELDS],
        )

    def test_state_matches_probability_reference_rule(self):
        frame = prediction_frame([0.49, 0.51, 0.50], [4, 5, 6])
        display = generator.build_display_df(frame)

        expected = np.where(
            display["display_prob_negative"] >= generator.NEGATIVE_THRESHOLD,
            "负性",
            "非负性",
        )
        self.assertListEqual(display["display_state"].tolist(), expected.tolist())
        np.testing.assert_allclose(
            display["display_prob_non_negative"],
            1.0 - display["display_prob_negative"],
        )
        np.testing.assert_allclose(
            display["display_negative_score"],
            100.0 * display["display_prob_negative"],
        )

    def test_formal_output_is_fixed_to_calibration_feature(self):
        compare_df = pd.DataFrame(
            [
                {
                    "scaler_mode": "match_training_test",
                    "output_role": "diagnostic_only",
                    "binary_acc_median": 100.0,
                    "selected_display_csv": "diagnostic.csv",
                },
                {
                    "scaler_mode": "calibration_feature",
                    "output_role": "formal_display",
                    "binary_acc_median": 0.0,
                    "selected_display_csv": "formal.csv",
                },
            ]
        )

        selected = generator.select_formal_output_row(compare_df)

        self.assertEqual(selected["scaler_mode"], "calibration_feature")
        self.assertEqual(selected["selected_display_csv"], "formal.csv")

    def test_label_shuffle_does_not_change_formal_mode_selection(self):
        compare_df = pd.DataFrame(
            [
                {
                    "scaler_mode": "calibration_feature",
                    "output_role": "formal_display",
                    "true_label_name": "negative",
                    "selected_display_csv": "formal.csv",
                },
                {
                    "scaler_mode": "match_training_test",
                    "output_role": "diagnostic_only",
                    "true_label_name": "positive",
                    "selected_display_csv": "diagnostic.csv",
                },
            ]
        )
        shuffled = compare_df.copy()
        shuffled["true_label_name"] = list(reversed(shuffled["true_label_name"]))

        original = generator.select_formal_output_row(compare_df)
        changed = generator.select_formal_output_row(shuffled)

        self.assertEqual(original["selected_display_csv"], "formal.csv")
        self.assertEqual(changed["selected_display_csv"], "formal.csv")

    def test_leaky_diagnostic_is_opt_in_and_never_formal(self):
        self.assertEqual(
            generator.scaler_modes_to_run(False),
            ["calibration_feature"],
        )
        self.assertEqual(
            generator.scaler_modes_to_run(True),
            ["calibration_feature", "match_training_test"],
        )
        self.assertNotIn(
            ("match_training_test", "none"),
            offline_diagnostic.diagnostic_test_configs(False),
        )
        self.assertIn(
            ("match_training_test", "none"),
            offline_diagnostic.diagnostic_test_configs(True),
        )
        diagnostic_paths = generator.output_paths("match_training_test")
        self.assertTrue(all("diagnostic_only" in path for path in diagnostic_paths))


class AdapterImportTests(unittest.TestCase):
    def test_named_adapter_classes_are_distinct(self):
        module = importlib.reload(model_adapters)
        self.assertTrue(hasattr(module, "GenericDeepLearningAdapter"))
        self.assertTrue(hasattr(module, "EEGNetAdapter"))
        self.assertTrue(hasattr(module, "UDADDAOnlineAdapter"))
        self.assertFalse(hasattr(module, "DeepLearningAdapter"))

    def test_adapter_factory_mapping(self):
        with mock.patch.object(
            model_adapters,
            "EEGNetAdapter",
            return_value="eegnet-adapter",
        ):
            result = model_adapters.AdapterFactory.create_adapter(
                "eegnet",
                model_path="unused.pth",
            )
            self.assertEqual(result, "eegnet-adapter")

        with mock.patch.object(
            model_adapters,
            "GenericDeepLearningAdapter",
            return_value="generic-adapter",
        ):
            result = model_adapters.AdapterFactory.create_adapter(
                "deep",
                model_path="unused.pth",
                model_class=object,
            )
            self.assertEqual(result, "generic-adapter")

        with mock.patch.object(
            model_adapters.os.path,
            "exists",
            return_value=True,
        ), mock.patch.object(
            model_adapters,
            "UDADDAOnlineAdapter",
            return_value="uda-adapter",
        ):
            result = model_adapters.AdapterFactory.create_adapter(
                "uda-dda-online",
                model_path="unused.pth",
                scaler_path="unused.pkl",
            )
            self.assertEqual(result, "uda-adapter")

    def test_generic_adapter_uses_four_value_interface(self):
        adapter = model_adapters.GenericDeepLearningAdapter(
            model_path="missing.pth",
            model_class=None,
        )
        result = adapter.predict(np.zeros((200, 8), dtype=np.float32))
        self.assertEqual(len(result), 4)
        self.assertEqual(result[3].shape, (3,))

    def test_eeg_viewer_module_imports(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    f"sys.path.insert(0, {str(ROOT / 'app')!r}); "
                    "import eeg_viewer2"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
