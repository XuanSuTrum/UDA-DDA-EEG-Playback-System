import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from plot_upper_demo_negative_history_from_log import parse_prediction_sync_lines


class PredictionArtifactTests(unittest.TestCase):
    def test_parse_prediction_sync_log_rows(self):
        lines = [
            "[16:15:28] [预测同步] t=0.5s | trial=4 | 真实标签=negative | 显示状态=负性 | 负性得分=61.0 | 负性概率=0.610 | 非负性概率=0.390",
            "[16:15:31] unrelated line",
            "[16:15:34] [预测同步] t=1.5s | trial=4 | 真实标签=negative | 显示状态=负性 | 负性得分=59.5 | 负性概率=0.595 | 非负性概率=0.405",
        ]

        df = parse_prediction_sync_lines(lines)

        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, "trial_id"], 4)
        self.assertAlmostEqual(df.loc[1, "prob_negative"], 0.595)
        self.assertAlmostEqual(df.loc[1, "prob_non_negative"], 0.405)

    def test_predictions_display_example_fields(self):
        example_path = ROOT / "examples" / "predictions_display.example.csv"
        df = pd.read_csv(example_path)
        required = {
            "time_sec",
            "trial_id",
            "true_label_name",
            "prob_negative",
            "prob_neutral",
            "prob_positive",
            "prob_non_negative",
            "display_state",
            "display_prob_negative",
            "display_prob_non_negative",
            "display_negative_score",
            "feature_source",
        }

        self.assertTrue(required.issubset(df.columns))
        for _, row in df.iterrows():
            self.assertAlmostEqual(
                row["display_prob_non_negative"],
                1.0 - row["display_prob_negative"],
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
