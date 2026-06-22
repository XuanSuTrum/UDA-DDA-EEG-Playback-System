#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate upper-computer demo prediction files from verified 62-channel DE+LDS features.

The replay MAT is still used by eeg_viewer2.py for waveform display. This script
uses the more stable offline DE+LDS feature chain to generate prediction CSV
files that eeg_viewer2.py can synchronize during playback.

The display files are based on model predictions aggregated at trial level, not
on true labels.
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import test_offline_lds as lds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = str(PROJECT_ROOT / "outputs" / "upper_demo")
META_CSV = str(PROJECT_ROOT / "data" / "upper_demo" / "subject15_trial4_15_replay_meta.csv")

DEFAULT_PRED_CSV = os.path.join(OUT_DIR, "subject15_trial4_15_predictions.csv")
DEFAULT_DISPLAY_CSV = os.path.join(OUT_DIR, "subject15_trial4_15_predictions_display.csv")
COMPARE_SUMMARY_CSV = os.path.join(OUT_DIR, "subject15_trial4_15_lds_compare_summary.csv")

FS = 200
WINDOW_SAMPLES = 200
NEGATIVE_THRESHOLD = 0.5
DISPLAY_TEMPERATURE = 3.0
DISPLAY_PROB_MODE = "segment_median"
DISPLAY_SEGMENT_WINDOW = 10
CALIBRATION_PREFERENCE_TOLERANCE = 0.5
FEATURE_SOURCE = "DE+LDS"

# Keep output filenames exactly keyed by scaler_mode. The postprocess choice is
# still recorded inside the CSV.
SCALER_POSTPROCESS_BY_MODE = {
    "match_training_test": "none",
    "calibration_feature": "clip",
}

THREE_CLASS_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def ensure_output_writable(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, "a", encoding="utf-8-sig"):
                pass
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot write output file because it may be open or read-only: {path}\n"
                "Please close Excel/WPS/Pandas viewers or remove the read-only attribute, then rerun."
            ) from exc


def configure_paths(args):
    """Apply CLI paths while keeping repository defaults local and non-personal."""
    global OUT_DIR, META_CSV, DEFAULT_PRED_CSV, DEFAULT_DISPLAY_CSV, COMPARE_SUMMARY_CSV

    OUT_DIR = str(Path(args.output_dir))
    META_CSV = str(Path(args.meta_csv))
    DEFAULT_PRED_CSV = str(Path(OUT_DIR) / "subject15_trial4_15_predictions.csv")
    DEFAULT_DISPLAY_CSV = str(Path(OUT_DIR) / "subject15_trial4_15_predictions_display.csv")
    COMPARE_SUMMARY_CSV = str(Path(OUT_DIR) / "subject15_trial4_15_lds_compare_summary.csv")

    lds.FEATURE_DIR = Path(args.feature_dir)
    lds.CALIB_FEATURE_PATH = str(Path(args.calib_feature))
    lds.ONLINE_FEATURE_PATH = str(Path(args.online_feature))
    lds.MODEL_PATH = str(Path(args.model_path))
    lds.OUTPUT_DIR = OUT_DIR
    lds.COMPARE_SUMMARY_PATH = str(Path(OUT_DIR) / "test_offline_lds_postprocess_compare_summary.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate timestamped predictions_display.csv files for offline upper-computer playback."
    )
    parser.add_argument("--feature-dir", default=str(lds.FEATURE_DIR), help="Directory containing derived DE+LDS feature MAT files.")
    parser.add_argument("--calib-feature", default=lds.CALIB_FEATURE_PATH, help="Calibration feature MAT path.")
    parser.add_argument("--online-feature", default=lds.ONLINE_FEATURE_PATH, help="Online/demo feature MAT path.")
    parser.add_argument("--model-path", default=lds.MODEL_PATH, help="UDA-DDA model weight path. Weights are not distributed in this repository.")
    parser.add_argument("--meta-csv", default=META_CSV, help="Replay metadata CSV with time_sec/trial/label rows.")
    parser.add_argument("--output-dir", default=OUT_DIR, help="Directory for generated prediction CSV files.")
    return parser.parse_args(argv)


def true_binary_label(true_label_name):
    return "负性" if str(true_label_name).lower() == "negative" else "非负性"


def pred_binary_label(prob_negative):
    return "负性" if float(prob_negative) >= NEGATIVE_THRESHOLD else "非负性"


def temperature_calibrate_probs(probs, temperature=DISPLAY_TEMPERATURE, eps=1e-8):
    """Calibrate already-softmax probabilities for display without changing labels."""
    probs = np.asarray(probs, dtype=np.float64)
    clipped = np.clip(probs, eps, 1.0)
    powered = clipped ** (1.0 / float(temperature))
    return powered / powered.sum(axis=-1, keepdims=True)


def majority_vote(values):
    counts = Counter(values)
    return counts.most_common(1)[0][0] if counts else ""


def load_window_meta(n_windows):
    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"Meta CSV not found: {META_CSV}")

    meta_df = pd.read_csv(META_CSV)
    expected_samples = n_windows * WINDOW_SAMPLES
    if len(meta_df) < expected_samples:
        print(
            f"WARNING: meta rows ({len(meta_df)}) < expected samples ({expected_samples}); "
            "using nearest available meta row at the end."
        )

    rows = []
    for window_index in range(n_windows):
        start_sample = window_index * WINDOW_SAMPLES
        end_sample = start_sample + WINDOW_SAMPLES
        center_sample = start_sample + WINDOW_SAMPLES // 2
        meta_idx = min(center_sample, len(meta_df) - 1)
        meta_row = meta_df.iloc[meta_idx]
        rows.append({
            "window_index": window_index,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "center_sample": center_sample,
            "time_sec": center_sample / FS,
            "trial_id": int(meta_row["trial_id"]),
            "trial_time_sec": float(meta_row["trial_time_sec"]),
            "raw_label": int(meta_row["raw_label"]),
            "true_label_name": str(meta_row["true_label_name"]),
        })
    return pd.DataFrame(rows)


def run_model_probs(model, features, device):
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = model.predict(x)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
    if probs.shape[1] != 3:
        raise ValueError(f"Expected probability shape (N,3), got {probs.shape}")
    return probs


def output_paths(scaler_mode):
    pred_csv = os.path.join(OUT_DIR, f"subject15_trial4_15_predictions_lds_{scaler_mode}.csv")
    trial_summary_csv = os.path.join(OUT_DIR, f"subject15_trial4_15_trial_summary_lds_{scaler_mode}.csv")
    display_csv = os.path.join(OUT_DIR, f"subject15_trial4_15_predictions_display_lds_{scaler_mode}.csv")
    return pred_csv, trial_summary_csv, display_csv


def build_prediction_df(probs, window_meta_df, scaler_mode, scaler_postprocess):
    rows = []
    probs_calibrated = temperature_calibrate_probs(probs, temperature=DISPLAY_TEMPERATURE)
    for i, meta in window_meta_df.iterrows():
        prob_negative = float(probs[i, 0])
        prob_neutral = float(probs[i, 1])
        prob_positive = float(probs[i, 2])
        prob_non_negative = prob_neutral + prob_positive
        prob_negative_calibrated = float(probs_calibrated[i, 0])
        prob_neutral_calibrated = float(probs_calibrated[i, 1])
        prob_positive_calibrated = float(probs_calibrated[i, 2])
        prob_non_negative_calibrated = prob_neutral_calibrated + prob_positive_calibrated
        pred_three_idx = int(np.argmax(probs[i]))
        rows.append({
            **meta.to_dict(),
            "prob_negative": prob_negative,
            "prob_neutral": prob_neutral,
            "prob_positive": prob_positive,
            "prob_non_negative": prob_non_negative,
            "negative_score": prob_negative * 100.0,
            "prob_negative_calibrated": prob_negative_calibrated,
            "prob_neutral_calibrated": prob_neutral_calibrated,
            "prob_positive_calibrated": prob_positive_calibrated,
            "prob_non_negative_calibrated": prob_non_negative_calibrated,
            "negative_score_calibrated": prob_negative_calibrated * 100.0,
            "display_temperature": DISPLAY_TEMPERATURE,
            "pred_binary_label": pred_binary_label(prob_negative),
            "pred_three_class_label": THREE_CLASS_LABELS[pred_three_idx],
            "scaler_mode": scaler_mode,
            "scaler_postprocess": scaler_postprocess,
            "feature_source": FEATURE_SOURCE,
        })
    return pd.DataFrame(rows)


def build_trial_summary(pred_df, scaler_mode):
    rows = []
    for trial_id, g in pred_df.groupby("trial_id", sort=True):
        mean_prob_negative = float(g["prob_negative"].mean())
        median_prob_negative = float(g["prob_negative"].median())
        mean_prob_neutral = float(g["prob_neutral"].mean())
        mean_prob_positive = float(g["prob_positive"].mean())
        mean_prob_non_negative = mean_prob_neutral + mean_prob_positive
        mean_prob_negative_calibrated = float(g["prob_negative_calibrated"].mean())
        median_prob_negative_calibrated = float(g["prob_negative_calibrated"].median())
        mean_negative_score_calibrated = mean_prob_negative_calibrated * 100.0
        median_negative_score_calibrated = median_prob_negative_calibrated * 100.0
        negative_window_ratio = float((g["pred_binary_label"] == "负性").mean())

        mean_prob_binary_pred = pred_binary_label(mean_prob_negative)
        median_prob_binary_pred = pred_binary_label(median_prob_negative)
        window_majority_binary_pred = majority_vote(g["pred_binary_label"].tolist())

        three_probs = np.array([mean_prob_negative, mean_prob_neutral, mean_prob_positive])
        three_class_mean_prob_pred = THREE_CLASS_LABELS[int(np.argmax(three_probs))]
        three_class_majority_pred = majority_vote(g["pred_three_class_label"].tolist())

        true_name = str(g["true_label_name"].iloc[0])
        true_binary = true_binary_label(true_name)

        rows.append({
            "trial_id": int(trial_id),
            "true_label_name": true_name,
            "true_binary_label": true_binary,
            "n_windows": int(len(g)),
            "mean_prob_negative": mean_prob_negative,
            "median_prob_negative": median_prob_negative,
            "mean_prob_non_negative": mean_prob_non_negative,
            "mean_prob_neutral": mean_prob_neutral,
            "mean_prob_positive": mean_prob_positive,
            "mean_negative_score": mean_prob_negative * 100.0,
            "mean_prob_negative_calibrated": mean_prob_negative_calibrated,
            "median_prob_negative_calibrated": median_prob_negative_calibrated,
            "mean_negative_score_calibrated": mean_negative_score_calibrated,
            "median_negative_score_calibrated": median_negative_score_calibrated,
            "negative_window_ratio": negative_window_ratio,
            "mean_prob_binary_pred": mean_prob_binary_pred,
            "median_prob_binary_pred": median_prob_binary_pred,
            "window_majority_binary_pred": window_majority_binary_pred,
            "binary_correct_mean": int(mean_prob_binary_pred == true_binary),
            "binary_correct_median": int(median_prob_binary_pred == true_binary),
            "binary_correct_majority": int(window_majority_binary_pred == true_binary),
            "three_class_mean_prob_pred": three_class_mean_prob_pred,
            "three_class_majority_pred": three_class_majority_pred,
            "scaler_mode": scaler_mode,
            "feature_source": FEATURE_SOURCE,
        })
    return pd.DataFrame(rows)


def choose_display_strategy(trial_summary_df):
    acc = {
        "mean": float(trial_summary_df["binary_correct_mean"].mean() * 100.0),
        "median": float(trial_summary_df["binary_correct_median"].mean() * 100.0),
        "majority": float(trial_summary_df["binary_correct_majority"].mean() * 100.0),
    }
    # For the upper-computer display, keep state stable with trial-level median.
    # Dynamic window/segment probabilities below only affect the displayed curve.
    return "median", acc


def compute_dynamic_display_probabilities(display_df):
    if DISPLAY_PROB_MODE not in {"window", "segment_mean", "segment_median"}:
        raise ValueError(f"Unknown DISPLAY_PROB_MODE: {DISPLAY_PROB_MODE}")

    dynamic_probs = pd.Series(index=display_df.index, dtype=float)

    for _, group in display_df.groupby("trial_id", sort=False):
        calibrated = group["prob_negative_calibrated"].astype(float)
        if DISPLAY_PROB_MODE == "window":
            values = calibrated
        elif DISPLAY_PROB_MODE == "segment_mean":
            values = calibrated.rolling(DISPLAY_SEGMENT_WINDOW, min_periods=1).mean()
        else:
            values = calibrated.rolling(DISPLAY_SEGMENT_WINDOW, min_periods=1).median()
        dynamic_probs.loc[group.index] = values

    return dynamic_probs.clip(0.0, 1.0).astype(float)


def build_display_df(pred_df, trial_summary_df, strategy):
    display_df = pred_df.copy()
    summary_by_trial = {int(r["trial_id"]): r for _, r in trial_summary_df.iterrows()}
    dynamic_display_probs = compute_dynamic_display_probabilities(display_df)

    for trial_id, row in summary_by_trial.items():
        mask = display_df["trial_id"] == trial_id
        if strategy == "mean":
            raw_p_neg = float(row["mean_prob_negative"])
            state = row["mean_prob_binary_pred"]
        elif strategy == "median":
            raw_p_neg = float(row["median_prob_negative"])
            state = row["median_prob_binary_pred"]
        elif strategy == "majority":
            raw_p_neg = float(row["negative_window_ratio"])
            state = row["window_majority_binary_pred"]
        else:
            raise ValueError(f"Unknown display strategy: {strategy}")

        raw_p_neg = float(np.clip(raw_p_neg, 0.0, 1.0))
        display_df.loc[mask, "display_state"] = state
        display_df.loc[mask, "raw_display_prob_negative"] = raw_p_neg
        display_df.loc[mask, "display_strategy"] = strategy
        display_df.loc[mask, "display_temperature"] = DISPLAY_TEMPERATURE
        display_df.loc[mask, "display_probability_source"] = (
            f"temperature_calibrated_softmax_{DISPLAY_PROB_MODE}"
        )
        display_df.loc[mask, "display_prob_mode"] = DISPLAY_PROB_MODE
        display_df.loc[mask, "display_segment_window"] = DISPLAY_SEGMENT_WINDOW

    display_df["display_prob_negative"] = dynamic_display_probs
    display_df["display_prob_non_negative"] = 1.0 - display_df["display_prob_negative"]
    display_df["display_negative_score"] = display_df["display_prob_negative"] * 100.0

    return display_df


def run_one_mode(model, calib_features, online_features, window_meta_df, scaler_mode):
    scaler_postprocess = SCALER_POSTPROCESS_BY_MODE[scaler_mode]
    print("\n" + "=" * 80)
    print(f"Generate LDS predictions | scaler_mode={scaler_mode} | postprocess={scaler_postprocess}")
    print("=" * 80)

    scaler = lds.build_scaler(scaler_mode, calib_features, online_features)
    online_scaled_raw = scaler.transform(online_features).astype(np.float32)
    online_scaled = lds.postprocess_scaled_feature(online_scaled_raw, mode=scaler_postprocess)
    print(
        f"Scaled range raw=[{online_scaled_raw.min():.3f}, {online_scaled_raw.max():.3f}] "
        f"post=[{online_scaled.min():.3f}, {online_scaled.max():.3f}]"
    )

    probs = run_model_probs(model, online_scaled, lds.DEVICE)
    pred_df = build_prediction_df(probs, window_meta_df, scaler_mode, scaler_postprocess)
    trial_summary_df = build_trial_summary(pred_df, scaler_mode)
    display_strategy, binary_acc = choose_display_strategy(trial_summary_df)
    display_df = build_display_df(pred_df, trial_summary_df, display_strategy)
    raw_display_min = float(display_df["raw_display_prob_negative"].min())
    raw_display_max = float(display_df["raw_display_prob_negative"].max())
    calibrated_display_min = float(display_df["display_prob_negative"].min())
    calibrated_display_max = float(display_df["display_prob_negative"].max())

    pred_csv, trial_summary_csv, display_csv = output_paths(scaler_mode)
    for path in (pred_csv, trial_summary_csv, display_csv):
        ensure_output_writable(path)
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    trial_summary_df.to_csv(trial_summary_csv, index=False, encoding="utf-8-sig")
    display_df.to_csv(display_csv, index=False, encoding="utf-8-sig")

    three_class_acc_mean = float(
        (trial_summary_df["three_class_mean_prob_pred"] == trial_summary_df["true_label_name"]).mean() * 100.0
    )

    print("Trial summary:")
    print(
        trial_summary_df[
            [
                "trial_id",
                "true_label_name",
                "mean_prob_binary_pred",
                "median_prob_binary_pred",
                "window_majority_binary_pred",
                "binary_correct_mean",
                "binary_correct_median",
                "binary_correct_majority",
            ]
        ].to_string(index=False)
    )
    print(
        f"Binary trial acc mean={binary_acc['mean']:.2f}% | "
        f"median={binary_acc['median']:.2f}% | majority={binary_acc['majority']:.2f}%"
    )
    print(f"Selected display strategy: {display_strategy}")
    print(f"Saved: {pred_csv}")
    print(f"Saved: {trial_summary_csv}")
    print(f"Saved: {display_csv}")

    return {
        "scaler_mode": scaler_mode,
        "binary_acc_mean": binary_acc["mean"],
        "binary_acc_median": binary_acc["median"],
        "binary_acc_majority": binary_acc["majority"],
        "three_class_acc_mean": three_class_acc_mean,
        "selected_display_strategy": display_strategy,
        "selected_display_csv": display_csv,
        "selected_prediction_csv": pred_csv,
        "selected_trial_summary_csv": trial_summary_csv,
        "raw_display_prob_negative_min": raw_display_min,
        "raw_display_prob_negative_max": raw_display_max,
        "calibrated_display_prob_negative_min": calibrated_display_min,
        "calibrated_display_prob_negative_max": calibrated_display_max,
        "display_prob_mode": DISPLAY_PROB_MODE,
        "display_segment_window": DISPLAY_SEGMENT_WINDOW,
    }


def main(argv=None):
    args = parse_args(argv)
    configure_paths(args)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading DE+LDS features using test_offline_lds.py paths...")
    calib_features, _, _ = lds.load_mat_feature_label(lds.CALIB_FEATURE_PATH, session=lds.SESSION)
    online_features, online_labels_raw, online_labels_idx = lds.load_mat_feature_label(
        lds.ONLINE_FEATURE_PATH,
        session=lds.SESSION,
    )
    print(f"Calibration DE+LDS feature shape: {calib_features.shape}")
    print(f"Online/demo DE+LDS feature shape: {online_features.shape}")
    if calib_features.shape != (485, 310):
        print(f"WARNING: expected calibration shape (485, 310), got {calib_features.shape}")
    if online_features.shape != (1964, 310):
        print(f"WARNING: expected online/demo shape (1964, 310), got {online_features.shape}")

    window_meta_df = load_window_meta(len(online_features))
    # Feature labels are still checked against meta to catch accidental ordering drift.
    feature_label_names = [lds.INDEX_TO_NAME[int(v)] for v in online_labels_idx]
    mismatch = np.mean(window_meta_df["true_label_name"].values != np.asarray(feature_label_names))
    if mismatch > 0:
        print(f"WARNING: meta labels and DE+LDS labels mismatch ratio: {mismatch:.4f}")

    print("Loading UDA-DDA model...")
    model = lds.load_model(lds.MODEL_PATH, lds.DEVICE)

    summary_rows = []
    for scaler_mode in ("match_training_test", "calibration_feature"):
        summary_rows.append(
            run_one_mode(
                model=model,
                calib_features=calib_features,
                online_features=online_features,
                window_meta_df=window_meta_df,
                scaler_mode=scaler_mode,
            )
        )

    compare_df = pd.DataFrame(summary_rows)
    # Pick the default display source by selected display strategy accuracy.
    # If calibration_feature is tied or within a small tolerance, prefer it
    # because it matches the deployment protocol: trials 1-3 calibration,
    # trials 4-15 detection. match_training_test remains a diagnostic upper bound.
    def selected_acc(row):
        return float(row[f"binary_acc_{row['selected_display_strategy']}"])

    compare_df["_selected_acc"] = compare_df.apply(selected_acc, axis=1)
    compare_df["selected_reason"] = ""

    best_idx_by_acc = int(compare_df["_selected_acc"].idxmax())
    best_row_by_acc = compare_df.loc[best_idx_by_acc]
    calibration_rows = compare_df[compare_df["scaler_mode"] == "calibration_feature"]
    if not calibration_rows.empty:
        calibration_idx = int(calibration_rows.index[0])
        calibration_acc = float(compare_df.loc[calibration_idx, "_selected_acc"])
        best_acc = float(best_row_by_acc["_selected_acc"])
        if best_acc - calibration_acc <= CALIBRATION_PREFERENCE_TOLERANCE:
            best_idx = calibration_idx
            selected_reason = (
                "prefer calibration_feature under tied accuracy because it matches deployment protocol"
            )
        else:
            best_idx = best_idx_by_acc
            selected_reason = "select highest selected display accuracy"
    else:
        best_idx = best_idx_by_acc
        selected_reason = "select highest selected display accuracy"

    compare_df.loc[best_idx, "selected_reason"] = selected_reason
    best_row = compare_df.loc[best_idx]

    ensure_output_writable(COMPARE_SUMMARY_CSV)
    compare_df.to_csv(COMPARE_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    ensure_output_writable(DEFAULT_DISPLAY_CSV)
    ensure_output_writable(DEFAULT_PRED_CSV)
    shutil.copyfile(best_row["selected_display_csv"], DEFAULT_DISPLAY_CSV)
    shutil.copyfile(best_row["selected_prediction_csv"], DEFAULT_PRED_CSV)

    print("\n" + "=" * 80)
    print("LDS compare summary:")
    print(
        compare_df[
            [
                "scaler_mode",
                "binary_acc_mean",
                "binary_acc_median",
                "binary_acc_majority",
                "three_class_acc_mean",
                "selected_display_strategy",
                "_selected_acc",
                "raw_display_prob_negative_min",
                "raw_display_prob_negative_max",
                "calibrated_display_prob_negative_min",
                "calibrated_display_prob_negative_max",
                "display_prob_mode",
                "display_segment_window",
                "selected_reason",
            ]
        ].to_string(index=False)
    )
    print("\nDefault display file selected for eeg_viewer2.py:")
    print(f"  scaler_mode: {best_row['scaler_mode']}")
    print(f"  strategy: {best_row['selected_display_strategy']}")
    print(f"  source display CSV: {best_row['selected_display_csv']}")
    print(f"  copied to: {DEFAULT_DISPLAY_CSV}")
    print(f"  copied prediction CSV to: {DEFAULT_PRED_CSV}")
    print("\nDisplay probability calibration:")
    print(f"  temperature = {DISPLAY_TEMPERATURE}")
    print(
        "  raw trial-level display negative range = "
        f"[{best_row['raw_display_prob_negative_min']:.3f}, "
        f"{best_row['raw_display_prob_negative_max']:.3f}]"
    )
    print(
        "  dynamic display negative range = "
        f"[{best_row['calibrated_display_prob_negative_min']:.3f}, "
        f"{best_row['calibrated_display_prob_negative_max']:.3f}]"
    )
    print("\nDisplay probability mode:")
    print(f"  DISPLAY_PROB_MODE = {DISPLAY_PROB_MODE}")
    print(f"  DISPLAY_SEGMENT_WINDOW = {DISPLAY_SEGMENT_WINDOW}")
    print("  Original softmax probabilities remain in prob_* fields.")
    print("  display_state comes from model trial-level aggregation.")
    print("  display_prob_negative comes from window/segment calibrated probabilities.")
    print("  true labels are only used for evaluation logs, not for display prediction.")
    if best_row["scaler_mode"] == "calibration_feature":
        print(
            "最终默认展示文件选择 calibration_feature + clip，因为其准确率与 "
            "match_training_test 相同或差距很小，且更符合前3个trial标定、"
            "后12个trial检测的部署流程。"
        )
    print(f"  selected_reason: {selected_reason}")
    print("IMPORTANT: display fields come from model prediction aggregation, not true-label substitution.")
    print(f"Compare summary saved: {COMPARE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
