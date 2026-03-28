"""Compare prediction and ground-truth CSVs on id; write report, CM image, and optional per-topic summary."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def _find_label_column(df: pd.DataFrame, path: str, label_col: str = None) -> str:
    """Find the column containing labels ('label' or 'prediction')."""
    if label_col and label_col in df.columns:
        return label_col
    for col in ("label", "prediction"):
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find 'label' or 'prediction' column in {path}. "
        f"Available columns: {list(df.columns)}"
    )


def _find_topic_column(df: pd.DataFrame, hint: str = None) -> str | None:
    """Return the topic column name, or None if not found."""
    if hint and hint in df.columns:
        return hint
    for col in ("topic", "Topic", "target", "Target"):
        if col in df.columns:
            return col
    return None


def _build_summary(
    overall: dict,
    per_topic: dict | None,
    pred_path: str,
    true_path: str,
    n_samples: int,
) -> str:
    """Build a readable summary string for overall + per-topic performance."""
    metric_labels = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
    }
    lines = []
    lines.append("=" * 65)
    lines.append("EVALUATION SUMMARY")
    lines.append("=" * 65)
    lines.append(f"  Predictions : {pred_path}")
    lines.append(f"  Ground truth: {true_path}")
    lines.append(f"  Samples     : {n_samples}")
    lines.append("")
    lines.append("OVERALL PERFORMANCE")
    lines.append("-" * 45)
    for key, label in metric_labels.items():
        val = overall.get(key)
        if val is not None:
            lines.append(f"  {label:<20s} {val:.4f}")

    if per_topic:
        lines.append("")
        lines.append("")
        lines.append("PER-TOPIC PERFORMANCE")
        lines.append("=" * 65)
        for topic in sorted(per_topic.keys()):
            t = per_topic[topic]
            lines.append("")
            lines.append(f"  Topic: {topic}  (n={t['support']})")
            lines.append(f"  {'-' * 45}")
            for key, label in metric_labels.items():
                val = t.get(key)
                if val is not None:
                    lines.append(f"    {label:<20s} {val:.4f}")
            if t.get("classification_report"):
                lines.append("")
                lines.append(t["classification_report"])

    lines.append("")
    return "\n".join(lines)


def evaluate(pred_path: str, true_path: str, output_dir: str = None,
             topic_col: str = None, label_col: str = None):
    """Run evaluation and save results."""

    pred_df = pd.read_csv(pred_path)
    true_df = pd.read_csv(true_path)

    pred_col = _find_label_column(pred_df, pred_path, label_col)
    true_col = _find_label_column(true_df, true_path)

    print(f"Predictions: {pred_path}  (column='{pred_col}', rows={len(pred_df)})")
    print(f"Ground truth: {true_path}  (column='{true_col}', rows={len(true_df)})")

    merged = pred_df[["id", pred_col]].merge(
        true_df[["id", true_col]], on="id", suffixes=("_pred", "_true")
    )

    if len(merged) == 0:
        raise ValueError("No matching ids found between prediction and ground truth files.")

    if len(merged) < len(pred_df):
        print(f"  WARNING: Only {len(merged)}/{len(pred_df)} prediction ids matched ground truth")

    y_pred_col = f"{pred_col}_pred" if f"{pred_col}_pred" in merged.columns else pred_col
    y_true_col = f"{true_col}_true" if f"{true_col}_true" in merged.columns else true_col

    y_pred = merged[y_pred_col].astype(str).values
    y_true = merged[y_true_col].astype(str).values

    labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 60)
    lines.append(f"Predictions file : {pred_path}")
    lines.append(f"Ground truth file: {true_path}")
    lines.append(f"Matched samples  : {len(merged)}")
    lines.append("")
    lines.append(f"Accuracy     : {acc:.4f}")
    lines.append(f"Macro F1     : {macro_f1:.4f}")
    lines.append(f"Weighted F1  : {weighted_f1:.4f}")
    lines.append("")
    lines.append("Classification Report:")
    lines.append(report)
    lines.append("Confusion Matrix (rows=true, cols=pred):")
    lines.append(f"Labels: {labels}")
    lines.append(np.array2string(cm))
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)

    if output_dir:
        out = Path(output_dir)
    else:
        out = Path(pred_path).parent

    out.mkdir(parents=True, exist_ok=True)

    results_path = out / "results.txt"
    results_path.write_text(report_text, encoding="utf-8")
    print(f"\nSaved report  → {results_path}")

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), max(5, len(labels) * 1.2)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix  (Macro F1={macro_f1:.4f})")
    plt.tight_layout()

    cm_path = out / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved CM image → {cm_path}")

    topic_col_name = _find_topic_column(true_df, hint=topic_col)
    per_topic = None

    if topic_col_name:
        merged_with_topic = pred_df[["id", pred_col]].merge(
            true_df[["id", true_col, topic_col_name]], on="id", suffixes=("_pred", "_true")
        )
        topics = merged_with_topic[topic_col_name].values
        per_topic = {}

        for t in sorted(set(topics)):
            mask = topics == t
            tp = y_pred[mask]
            tt = y_true[mask]
            t_labels = sorted(set(tt) | set(tp))

            per_topic[str(t)] = {
                "accuracy": float(accuracy_score(tt, tp)),
                "macro_f1": float(f1_score(tt, tp, average="macro", labels=t_labels, zero_division=0)),
                "macro_precision": float(precision_score(tt, tp, average="macro", labels=t_labels, zero_division=0)),
                "macro_recall": float(recall_score(tt, tp, average="macro", labels=t_labels, zero_division=0)),
                "support": int(mask.sum()),
                "classification_report": classification_report(
                    tt, tp, labels=t_labels, digits=4, zero_division=0
                ),
            }

        print("\nPer-topic results:")
        for tname, tm in sorted(per_topic.items()):
            print(f"  {tname} (n={tm['support']}):  "
                  f"Acc={tm['accuracy']:.4f}  F1={tm['macro_f1']:.4f}  "
                  f"P={tm['macro_precision']:.4f}  R={tm['macro_recall']:.4f}")

    overall = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
    }

    summary_text = _build_summary(overall, per_topic, pred_path, true_path, len(merged))
    summary_path = out / "eval_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved summary → {summary_path}")

    return overall


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions vs ground truth")
    parser.add_argument(
        "-p",
        "--pred", type=str, required=True,
        help="Predictions CSV (needs id + label/prediction columns)",
    )
    parser.add_argument(
        "-l",
        "--label-col", type=str, default=None,
        help="Label column in predictions CSV (auto-detected)",
    )
    parser.add_argument(
        "-t", "--true", type=str, required=True,
        help="Ground truth CSV (needs id + label/prediction columns)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory (default: same as --pred)",
    )
    parser.add_argument(
        "-tc",
        "--topic-col", type=str, default=None,
        help="Topic column in ground truth CSV (auto-detected)",
    )
    args = parser.parse_args()

    evaluate(args.pred, args.true, args.output_dir, topic_col=args.topic_col, label_col=args.label_col)


if __name__ == "__main__":
    main()
