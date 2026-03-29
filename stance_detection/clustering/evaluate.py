"""Evaluate a trained clustering model on a labelled CSV dataset."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .classifier import StanceClassifier


def _save_evaluation_results(y_true, y_pred, df, output_path, title, verbose):
    """Generate and save evaluation outputs (report, confusion matrix, metrics)."""
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_path = output_path / "predictions.csv"
    df_predictions = df.copy()
    df_predictions["predicted_stance"] = y_pred
    df_predictions.to_csv(predictions_path, index=False)
    if verbose:
        print(f"Predictions saved to: {predictions_path}")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"CLASSIFICATION REPORT — {title}")
        print(f"{'=' * 60}")

    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, zero_division=0)

    if verbose:
        print(report_str)

    report_txt_path = output_path / "classification_report.txt"
    with open(report_txt_path, "w") as f:
        f.write(f"CLASSIFICATION REPORT — {title}\n")
        f.write("=" * 60 + "\n")
        f.write(report_str)
    if verbose:
        print(f"Classification report saved to: {report_txt_path}")

    metrics_path = output_path / "metrics.json"
    metrics = {
        "accuracy": report_dict["accuracy"],
        "macro_avg": report_dict["macro avg"],
        "weighted_avg": report_dict["weighted avg"],
        "per_class": {k: v for k, v in report_dict.items()
                      if k not in ["accuracy", "macro avg", "weighted avg"]},
        "title": title,
        "num_samples": len(y_true),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print(f"Metrics saved to: {metrics_path}")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"CONFUSION MATRIX — {title}")
        print(f"{'=' * 60}")

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if verbose:
        print(f"Labels: {labels}")
        print(cm)

    cm_txt_path = output_path / "confusion_matrix.txt"
    with open(cm_txt_path, "w") as f:
        f.write(f"CONFUSION MATRIX — {title}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Labels: {labels}\n")
        f.write(str(cm))
    if verbose:
        print(f"Confusion matrix saved to: {cm_txt_path}")

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Stance Detection Confusion Matrix\n{title}")
    plt.tight_layout()
    cm_img_path = output_path / "confusion_matrix.png"
    plt.savefig(cm_img_path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Confusion matrix plot saved to: {cm_img_path}")

    return metrics


def evaluate_model(
    dataset_path: str,
    model_path: str,
    output_dir: str,
    topic: str = None,
    text_column: str = "sentence",
    topic_column: str = "topic",
    label_column: str = "prediction",
    verbose: bool = True,
):
    """Evaluate a trained clustering model on a labelled CSV.

    Predicts each row using its topic column (after normalisation) and
    generates per-topic and overall reports.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    for col in [text_column, topic_column, label_column]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

    if verbose:
        print(f"Loading model from: {model_path}")
    classifier = StanceClassifier.load(model_path)

    available_topics = classifier.get_topics()
    if verbose:
        print(f"Available model topics: {available_topics}")

    df["_normalized_topic"] = (
        df[topic_column].astype(str)
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    if topic is not None:
        topic = topic.replace(" ", "_").replace("/", "_")
        if topic not in available_topics:
            raise ValueError(
                f"Topic '{topic}' not found in model. Available topics: {available_topics}"
            )
        df = df[df["_normalized_topic"] == topic]
        if df.empty:
            raise ValueError(f"No rows in dataset match topic '{topic}'")
        if verbose:
            print(f"Filtered dataset to topic '{topic}': {len(df)} rows")

    dataset_topics = df["_normalized_topic"].unique().tolist()
    if verbose:
        print(f"Topics in dataset: {dataset_topics}")

    for t in dataset_topics:
        if t not in available_topics:
            raise ValueError(
                f"Topic '{t}' from dataset not found in model. "
                f"Available model topics: {available_topics}"
            )

    if verbose:
        print(f"\nPredicting {len(df)} samples across {len(dataset_topics)} topic(s)...")

    y_pred = []
    for _, row in df.iterrows():
        pred = classifier.predict(row[text_column], row["_normalized_topic"])
        y_pred.append(pred)

    y_true = df[label_column].tolist()

    all_metrics = {}
    for i, t in enumerate(dataset_topics):
        if verbose and len(dataset_topics) > 1:
            print(f"\n{'#' * 60}")
            print(f"# TOPIC {i + 1}/{len(dataset_topics)}: {t}")
            print(f"{'#' * 60}")

        mask = df["_normalized_topic"] == t
        topic_df = df[mask]
        topic_y_true = [y_true[j] for j in topic_df.index]
        topic_y_pred = [y_pred[j] for j in topic_df.index]

        topic_output_path = (
            output_path / t if len(dataset_topics) > 1 else output_path
        )
        metrics = _save_evaluation_results(
            y_true=topic_y_true,
            y_pred=topic_y_pred,
            df=topic_df,
            output_path=topic_output_path,
            title=f"Topic: {t}",
            verbose=verbose,
        )
        all_metrics[t] = metrics

    overall_metrics = None
    if len(dataset_topics) > 1:
        if verbose:
            print(f"\n{'#' * 60}")
            print("# OVERALL (all topics combined)")
            print(f"{'#' * 60}")
        overall_metrics = _save_evaluation_results(
            y_true=y_true,
            y_pred=y_pred,
            df=df,
            output_path=output_path / "overall",
            title="Overall (all topics)",
            verbose=verbose,
        )

    if verbose:
        print(f"\n{'=' * 60}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"All outputs saved to: {output_path}")

    if len(dataset_topics) == 1:
        return all_metrics[dataset_topics[0]]
    return {"per_topic": all_metrics, "overall": overall_metrics}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate clustering model on a labelled CSV")
    parser.add_argument("dataset", help="Path to CSV file with evaluation data")
    parser.add_argument("model", help="Path to trained model (.pkl)")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("-t", "--topic",
                        help="Evaluate a single topic (default: all topics in dataset)")
    parser.add_argument("--text-column", default="sentence",
                        help="Name of text column in CSV")
    parser.add_argument("--topic-column", default="topic",
                        help="Name of topic column in CSV")
    parser.add_argument("--label-column", default="prediction",
                        help="Name of label column in CSV")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output")
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Error: Dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    try:
        evaluate_model(
            dataset_path=args.dataset,
            model_path=args.model,
            output_dir=args.output_dir,
            topic=args.topic,
            text_column=args.text_column,
            topic_column=args.topic_column,
            label_column=args.label_column,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
