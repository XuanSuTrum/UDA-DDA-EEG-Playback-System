#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot P(negative) history from upper-computer prediction-sync logs."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PREDICTION_SYNC_RE = re.compile(
    r"\[预测同步\]\s*"
    r"t=(?P<time_sec>[-+]?\d+(?:\.\d+)?)s\s*\|\s*"
    r"trial=(?P<trial_id>\d+)\s*\|\s*"
    r"真实标签=(?P<true_label>[^|]+)\|\s*"
    r"显示状态=(?P<display_state>[^|]+)\|\s*"
    r"负性得分=(?P<negative_score>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"负性概率=(?P<prob_negative>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"非负性概率=(?P<prob_non_negative>[-+]?\d+(?:\.\d+)?)"
)

LABEL_COLORS = {
    "negative": "#fee2e2",
    "neutral": "#f1f5f9",
    "positive": "#dcfce7",
}


def parse_prediction_sync_lines(lines):
    """Parse observed prediction-sync log rows without filling missing time points."""
    rows = []
    for line in lines:
        match = PREDICTION_SYNC_RE.search(line)
        if not match:
            continue
        item = match.groupdict()
        rows.append(
            {
                "time_sec": float(item["time_sec"]),
                "trial_id": int(item["trial_id"]),
                "true_label": item["true_label"].strip(),
                "display_state": item["display_state"].strip(),
                "negative_score": float(item["negative_score"]),
                "prob_negative": float(item["prob_negative"]),
                "prob_non_negative": float(item["prob_non_negative"]),
            }
        )
    return pd.DataFrame(rows)


def read_prediction_log(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return parse_prediction_sync_lines(f)


def _label_color(label):
    return LABEL_COLORS.get(str(label).strip().lower(), "#e0f2fe")


def _plot_trial_background(ax, df):
    for trial_id, group in df.groupby("trial_id", sort=True):
        start = float(group["time_sec"].min())
        end = float(group["time_sec"].max())
        label = str(group["true_label"].iloc[0]).strip()
        ax.axvspan(start, end, color=_label_color(label), alpha=0.42, linewidth=0)
        ax.text(
            (start + end) / 2.0,
            1.04,
            f"Trial {trial_id}\n{label}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )


def _iter_continuous_segments(group, max_gap_sec):
    group = group.sort_values("time_sec")
    if max_gap_sec is None:
        yield group
        return

    start = 0
    times = group["time_sec"].tolist()
    for idx in range(1, len(times)):
        if times[idx] - times[idx - 1] > max_gap_sec:
            yield group.iloc[start:idx]
            start = idx
    yield group.iloc[start:]


def plot_negative_history(df, output_path, title=None, max_gap_sec=2.0):
    if df.empty:
        raise ValueError("No [预测同步] rows were parsed from the log.")

    df = df.sort_values(["trial_id", "time_sec"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=150)

    _plot_trial_background(ax, df)

    first_segment = True
    for _, group in df.groupby("trial_id", sort=True):
        for segment in _iter_continuous_segments(group, max_gap_sec=max_gap_sec):
            if segment.empty:
                continue
            ax.plot(
                segment["time_sec"],
                segment["prob_negative"],
                color="#2563eb",
                linewidth=2.0,
                marker="o",
                markersize=2.6,
                label="P(negative)" if first_segment else None,
            )
            first_segment = False

    ax.axhline(
        0.5,
        color="#64748b",
        linestyle="--",
        linewidth=1.2,
        label="50% equal-probability reference",
    )
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Playback time (s)")
    ax.set_ylabel("Probability")
    ax.set_title(title or "Offline Playback Negative-Class Probability")
    ax.grid(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Parse upper-computer logs and plot observed P(negative) over offline playback time."
    )
    parser.add_argument("--log", required=True, help="Upper-computer text log path.")
    parser.add_argument("--output", default="outputs/negative_history_from_log.png", help="Output image path.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=2.0,
        help="Break a line segment when adjacent log times are farther apart. Use 0 to disable gap splitting.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    max_gap_sec = None if args.max_gap_sec == 0 else args.max_gap_sec
    df = read_prediction_log(args.log)
    output_path = plot_negative_history(df, args.output, title=args.title, max_gap_sec=max_gap_sec)
    print(f"Parsed rows: {len(df)}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
