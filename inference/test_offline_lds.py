"""
Offline DE+LDS feature-chain diagnosis.

This script compares two scaler modes without changing training code or model
structure:

1. match_training_test:
   Fit MinMaxScaler on subject 15 online remaining DE+LDS features. This leaks
   test-set statistics and is used only to reproduce the offline upper bound.

2. calibration_feature:
   Fit MinMaxScaler on subject 15 calibration DE+LDS features, then transform
   the online remaining DE+LDS features. This is closer to deployment.

The goal is to separate the impact of LDS smoothing from scaler/calibration
distribution mismatch.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import scipy.io as scio
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import SDA_DDA


# ================= Config =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_DIR = PROJECT_ROOT / "data" / "seed_features"
CALIB_FEATURE_PATH = str(FEATURE_DIR / "15_calibration_3trials.mat")
ONLINE_FEATURE_PATH = str(FEATURE_DIR / "15_online_remaining_trials.mat")

MODEL_PATH = str(PROJECT_ROOT / "models" / "subject15_calib_supervised_best_model.pth")

OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
COMPARE_SUMMARY_PATH = str(Path(OUTPUT_DIR) / "test_offline_lds_postprocess_compare_summary.csv")

SESSION = 1
N_CLASS = 3
BASE_NET = "simple_net"
TRANSFER_LOSS = "mmd"

# Scaler postprocess options: "none", "clip", "tanh".
SCALER_POSTPROCESS = "clip"
CLIP_MIN = -1.0
CLIP_MAX = 1.0

RAW_LABELS_SESSION1 = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
ONLINE_TRIAL_IDS = list(range(4, 16))
ONLINE_TRIAL_RAW_LABELS = RAW_LABELS_SESSION1[3:15]

LABEL_MAP_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_NAME = {
    0: "negative",
    1: "neutral",
    2: "positive",
}
TARGET_NAMES = ["negative", "neutral", "positive"]


def dataset_key(session):
    return f"dataset_session{session}"


def load_mat_feature_label(mat_path, session=1):
    key = dataset_key(session)
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Feature file not found: {mat_path}")

    mat_data = scio.loadmat(mat_path)
    if key not in mat_data:
        raise KeyError(f"{mat_path} does not contain {key}")

    features = mat_data[key]["feature"][0, 0].astype(np.float32)
    labels_raw = mat_data[key]["label"][0, 0].reshape(-1)
    labels_idx = np.array([LABEL_MAP_TO_INDEX[int(v)] for v in labels_raw], dtype=np.int64)
    return features, labels_raw.astype(np.int64), labels_idx


def load_model(model_path, device):
    model = SDA_DDA.Transfer_Net(
        N_CLASS,
        transfer_loss=TRANSFER_LOSS,
        base_net=BASE_NET,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Model loaded: {model_path}")
    return model


def build_scaler(scaler_mode, calib_features, online_features):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    if scaler_mode == "match_training_test":
        scaler.fit(online_features)
    elif scaler_mode == "calibration_feature":
        scaler.fit(calib_features)
    else:
        raise ValueError(
            "scaler_mode must be 'match_training_test' or 'calibration_feature', "
            f"got {scaler_mode}"
        )
    return scaler


def postprocess_scaled_feature(x, mode="clip"):
    if mode == "none":
        return x.astype(np.float32)
    if mode == "clip":
        return np.clip(x, CLIP_MIN, CLIP_MAX).astype(np.float32)
    if mode == "tanh":
        return np.tanh(x).astype(np.float32)
    raise ValueError(f"Unknown scaler postprocess mode: {mode}")


def infer_trial_ids_from_labels(labels_raw):
    """
    Reconstruct trial ids for subject 15 online trials.

    The feature .mat stores concatenated windows and labels, but not trial ids.
    We use the known SEED session-1 trial label order. Consecutive trials with
    the same label are ambiguous from labels alone, so that same-label run is
    split as evenly as possible among those trials.
    """
    labels_raw = [int(v) for v in labels_raw]
    trial_ids = np.zeros(len(labels_raw), dtype=np.int64)
    pos = 0
    group_start = 0

    while group_start < len(ONLINE_TRIAL_RAW_LABELS):
        label = ONLINE_TRIAL_RAW_LABELS[group_start]
        group_end = group_start + 1
        while (
            group_end < len(ONLINE_TRIAL_RAW_LABELS)
            and ONLINE_TRIAL_RAW_LABELS[group_end] == label
        ):
            group_end += 1

        run_start = pos
        while pos < len(labels_raw) and labels_raw[pos] == label:
            pos += 1
        run_end = pos
        run_len = run_end - run_start
        n_trials = group_end - group_start

        if run_len <= 0:
            raise ValueError(
                "Cannot reconstruct trial ids: label sequence does not match "
                f"expected online trial label {label} at trial {group_start + 4}."
            )

        split_sizes = [run_len // n_trials] * n_trials
        for i in range(run_len % n_trials):
            split_sizes[i] += 1

        cursor = run_start
        for offset, size in enumerate(split_sizes):
            trial_id = ONLINE_TRIAL_IDS[group_start + offset]
            trial_ids[cursor : cursor + size] = trial_id
            cursor += size

        group_start = group_end

    if pos != len(labels_raw):
        raise ValueError(
            "Cannot reconstruct trial ids: extra labels remain after expected "
            f"online trials. consumed={pos}, total={len(labels_raw)}"
        )

    return trial_ids


def run_model_inference(model, features, labels_idx, labels_raw, trial_ids, device):
    results = []
    with torch.no_grad():
        x_tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = model.predict(x_tensor)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()

        for i in range(len(features)):
            true_idx = int(labels_idx[i])
            pred_idx = int(preds[i])
            results.append(
                {
                    "sample_idx": i,
                    "trial_id": int(trial_ids[i]),
                    "true_label_raw": int(labels_raw[i]),
                    "true_label_idx": true_idx,
                    "true_label_name": INDEX_TO_NAME[true_idx],
                    "pred_label_idx": pred_idx,
                    "pred_label_name": INDEX_TO_NAME[pred_idx],
                    "prob_negative": float(probs[i, 0].cpu().item()),
                    "prob_neutral": float(probs[i, 1].cpu().item()),
                    "prob_positive": float(probs[i, 2].cpu().item()),
                    "correct": int(pred_idx == true_idx),
                }
            )

    return pd.DataFrame(results)


def compute_metrics(results_df):
    y_true = results_df["true_label_idx"].values
    y_pred = results_df["pred_label_idx"].values

    overall_acc = float(results_df["correct"].mean() * 100.0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=TARGET_NAMES,
        zero_division=0,
    )

    class_acc = {}
    for idx, name in INDEX_TO_NAME.items():
        mask = y_true == idx
        class_acc[name] = float((y_pred[mask] == idx).mean() * 100.0) if mask.any() else np.nan

    return overall_acc, class_acc, cm, report


def build_trial_summary(results_df):
    rows = []
    for trial_id, g in results_df.groupby("trial_id"):
        majority_pred = int(g["pred_label_idx"].mode().iloc[0])
        true_idx = int(g["true_label_idx"].iloc[0])
        rows.append(
            {
                "trial_id": int(trial_id),
                "n_windows": int(len(g)),
                "true_label_name": g["true_label_name"].iloc[0],
                "majority_pred_name": INDEX_TO_NAME[majority_pred],
                "majority_correct": int(majority_pred == true_idx),
                "window_acc_percent": float(g["correct"].mean() * 100.0),
                "mean_prob_negative": float(g["prob_negative"].mean()),
                "mean_prob_neutral": float(g["prob_neutral"].mean()),
                "mean_prob_positive": float(g["prob_positive"].mean()),
            }
        )
    return pd.DataFrame(rows)


def output_paths_for_mode(scaler_mode, scaler_postprocess):
    output_dir = Path(OUTPUT_DIR)
    result_csv = str(output_dir / f"test_offline_lds_{scaler_mode}_{scaler_postprocess}_result.csv")
    trial_summary_csv = str(output_dir / f"test_offline_lds_{scaler_mode}_{scaler_postprocess}_trial_summary.csv")
    confusion_csv = str(output_dir / f"test_offline_lds_{scaler_mode}_{scaler_postprocess}_confusion.csv")
    return result_csv, trial_summary_csv, confusion_csv


def run_lds_feature_test(scaler_mode, scaler_postprocess=SCALER_POSTPROCESS):
    print("\n" + "=" * 70)
    print(
        "Offline DE+LDS test | "
        f"scaler_mode={scaler_mode} | scaler_postprocess={scaler_postprocess}"
    )
    print("=" * 70)

    print("[1/5] Load DE+LDS feature files...")
    calib_features, _, _ = load_mat_feature_label(CALIB_FEATURE_PATH, session=SESSION)
    online_features, online_labels_raw, online_labels_idx = load_mat_feature_label(
        ONLINE_FEATURE_PATH,
        session=SESSION,
    )
    trial_ids = infer_trial_ids_from_labels(online_labels_raw)

    print(f"Calibration features: {calib_features.shape}")
    print(f"Online features: {online_features.shape}")

    print("[2/5] Fit scaler...")
    scaler = build_scaler(scaler_mode, calib_features, online_features)
    online_features_scaled_raw = scaler.transform(online_features).astype(np.float32)
    raw_scaled_min = float(online_features_scaled_raw.min())
    raw_scaled_max = float(online_features_scaled_raw.max())
    print(
        "Raw scaled online range: "
        f"[{raw_scaled_min:.3f}, {raw_scaled_max:.3f}]"
    )
    online_features_scaled = postprocess_scaled_feature(
        online_features_scaled_raw,
        mode=scaler_postprocess,
    )
    post_scaled_min = float(online_features_scaled.min())
    post_scaled_max = float(online_features_scaled.max())
    print(
        "Postprocessed scaled online range: "
        f"[{post_scaled_min:.3f}, {post_scaled_max:.3f}]"
    )

    print("[3/5] Load model...")
    model = load_model(MODEL_PATH, DEVICE)

    print("[4/5] Run inference...")
    results_df = run_model_inference(
        model=model,
        features=online_features_scaled,
        labels_idx=online_labels_idx,
        labels_raw=online_labels_raw,
        trial_ids=trial_ids,
        device=DEVICE,
    )

    overall_acc, class_acc, cm, report = compute_metrics(results_df)
    trial_summary_df = build_trial_summary(results_df)
    trial_majority_acc = float(trial_summary_df["majority_correct"].mean() * 100.0)

    print("[5/5] Save outputs...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_csv, trial_summary_csv, confusion_csv = output_paths_for_mode(
        scaler_mode,
        scaler_postprocess,
    )
    results_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    trial_summary_df.to_csv(trial_summary_csv, index=False, encoding="utf-8-sig")

    cm_df = pd.DataFrame(
        cm,
        index=["True_Neg", "True_Neu", "True_Pos"],
        columns=["Pred_Neg", "Pred_Neu", "Pred_Pos"],
    )
    cm_df.to_csv(confusion_csv, encoding="utf-8-sig")

    print("\nResult")
    print("-" * 70)
    print(f"Samples: {len(results_df)}")
    print(f"Overall accuracy: {overall_acc:.2f}%")
    print(f"Trial majority accuracy: {trial_majority_acc:.2f}%")
    print("Class accuracy:")
    for name in TARGET_NAMES:
        print(f"  {name}: {class_acc[name]:.2f}%")
    print("\nConfusion matrix:")
    print(cm_df)
    print("\nClassification report:")
    print(report)
    print("Saved:")
    print(f"  Result CSV: {result_csv}")
    print(f"  Trial summary CSV: {trial_summary_csv}")
    print(f"  Confusion CSV: {confusion_csv}")

    summary_row = {
        "scaler_mode": scaler_mode,
        "scaler_postprocess": scaler_postprocess,
        "raw_scaled_min": raw_scaled_min,
        "raw_scaled_max": raw_scaled_max,
        "post_scaled_min": post_scaled_min,
        "post_scaled_max": post_scaled_max,
        "overall_acc": overall_acc,
        "negative_acc": class_acc["negative"],
        "neutral_acc": class_acc["neutral"],
        "positive_acc": class_acc["positive"],
        "trial_majority_acc": trial_majority_acc,
        "result_csv": result_csv,
        "trial_summary_csv": trial_summary_csv,
        "confusion_csv": confusion_csv,
    }
    return results_df, trial_summary_df, summary_row


def build_diagnosis_note(compare_df):
    def pick_acc(scaler_mode, scaler_postprocess):
        row = compare_df[
            (compare_df["scaler_mode"] == scaler_mode)
            & (compare_df["scaler_postprocess"] == scaler_postprocess)
        ]
        if row.empty:
            return np.nan
        return float(row.iloc[0]["overall_acc"])

    match_acc = pick_acc("match_training_test", "none")
    calib_none_acc = pick_acc("calibration_feature", "none")
    calib_clip_acc = pick_acc("calibration_feature", "clip")

    high_threshold = 75.0
    large_drop = 8.0

    if match_acc >= high_threshold and calib_clip_acc >= high_threshold:
        return (
            "DE+LDS remains high when calibration-scaled inputs are clipped. "
            "The previous calibration_feature collapse was mainly caused by "
            "out-of-range model inputs; if raw-DE causal smoothing is still low, "
            "missing LDS-like stabilization remains the next bottleneck."
        )
    if match_acc >= high_threshold and (match_acc - calib_none_acc) >= large_drop:
        return (
            "DE+LDS is high with test-fitted scaler but drops with calibration "
            "scaler before postprocess. Scaler/calibration range mismatch is a "
            "major bottleneck; compare clip/tanh rows to see whether bounding "
            "the model input recovers neutral performance."
        )
    return (
        "DE+LDS is not high under these scaler modes. This points to model "
        "training, target-subject adaptation, or feature-file mismatch issues."
    )


def configure_paths(args):
    """Apply CLI paths while keeping import-time defaults safe for public repos."""
    global FEATURE_DIR, CALIB_FEATURE_PATH, ONLINE_FEATURE_PATH, MODEL_PATH, OUTPUT_DIR, COMPARE_SUMMARY_PATH

    FEATURE_DIR = Path(args.feature_dir)
    CALIB_FEATURE_PATH = str(Path(args.calib_feature))
    ONLINE_FEATURE_PATH = str(Path(args.online_feature))
    MODEL_PATH = str(Path(args.model_path))
    OUTPUT_DIR = str(Path(args.output_dir))
    COMPARE_SUMMARY_PATH = str(Path(OUTPUT_DIR) / "test_offline_lds_postprocess_compare_summary.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the offline DE+LDS UDA-DDA feature-chain diagnosis."
    )
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR), help="Directory containing derived DE+LDS feature MAT files.")
    parser.add_argument("--calib-feature", default=CALIB_FEATURE_PATH, help="Calibration feature MAT path.")
    parser.add_argument("--online-feature", default=ONLINE_FEATURE_PATH, help="Online/demo feature MAT path.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="UDA-DDA model weight path. Weights are not distributed in this repository.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for diagnostic CSV outputs.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_paths(args)

    print("=" * 70)
    print("Offline DE+LDS feature-chain diagnosis")
    print("=" * 70)

    summary_rows = []
    test_configs = [
        ("match_training_test", "none"),
        ("calibration_feature", "none"),
        ("calibration_feature", "clip"),
        ("calibration_feature", "tanh"),
    ]
    for scaler_mode, scaler_postprocess in test_configs:
        _, _, summary_row = run_lds_feature_test(scaler_mode, scaler_postprocess)
        summary_rows.append(summary_row)

    compare_df = pd.DataFrame(summary_rows)
    diagnosis_note = build_diagnosis_note(compare_df)
    compare_df["diagnosis_note"] = diagnosis_note

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    compare_df.to_csv(COMPARE_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("Feature-chain diagnosis")
    print("=" * 70)
    print(
        compare_df[
            [
                "scaler_mode",
                "scaler_postprocess",
                "overall_acc",
                "neutral_acc",
                "raw_scaled_min",
                "raw_scaled_max",
                "post_scaled_min",
                "post_scaled_max",
                "trial_majority_acc",
            ]
        ]
    )
    print("\nDiagnosis:")
    print(diagnosis_note)
    print(f"\nCompare summary saved to: {COMPARE_SUMMARY_PATH}")

    return compare_df


if __name__ == "__main__":
    compare_df = main()
