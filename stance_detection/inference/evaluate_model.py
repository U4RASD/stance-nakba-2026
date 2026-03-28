"""Load a saved stance (or multitask) model, run inference on a CSV, optionally eval and save plots/summary."""

import argparse
import json
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

from ..data_loader import StanceDataset, normalize_arabic


_ARABIC_FONT_PATH = "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"
_font_fallback_configured = False


def _configure_arabic_fallback():
    """Register Noto Sans Arabic in matplotlib's font fallback chain.

    Uses matplotlib >=3.2 font-fallback: DejaVu Sans renders Latin/punctuation,
    Noto Sans Arabic renders Arabic glyphs — no per-element fontproperties needed.
    """
    global _font_fallback_configured
    if _font_fallback_configured:
        return
    _font_fallback_configured = True
    if Path(_ARABIC_FONT_PATH).exists():
        fm.fontManager.addfont(_ARABIC_FONT_PATH)
        matplotlib.rcParams["font.family"] = "sans-serif"
        sans = matplotlib.rcParams.get("font.sans-serif", [])
        if "Noto Sans Arabic" not in sans:
            matplotlib.rcParams["font.sans-serif"] = [
                "DejaVu Sans", "Noto Sans Arabic",
            ] + [f for f in sans if f not in ("DejaVu Sans", "Noto Sans Arabic")]


def _ar(text: str) -> str:
    """Reshape and reorder Arabic text so matplotlib renders it correctly."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def _detect_text_col(df: pd.DataFrame, hint: str = None) -> str:
    if hint and hint in df.columns:
        return hint
    for col in ("sentence", "text", "Text", "Sentence"):
        if col in df.columns:
            return col
    raise ValueError(
        f"Cannot detect text column. Available: {list(df.columns)}. "
        "Pass --text-col explicitly."
    )


def _detect_topic_col(df: pd.DataFrame, hint: str = None) -> str | None:
    if hint and hint in df.columns:
        return hint
    for col in ("topic", "Topic", "target", "Target"):
        if col in df.columns:
            return col
    return None


def _detect_label_col(df: pd.DataFrame, hint: str = None) -> str:
    if hint and hint in df.columns:
        return hint
    for col in ("label", "prediction", "Label", "Prediction"):
        if col in df.columns:
            return col
    raise ValueError(
        f"Cannot detect label column. Available: {list(df.columns)}. "
        "Pass --label-col explicitly."
    )


def _is_multitask(model_dir: Path) -> bool:
    return (model_dir / "label_config.json").exists()


def _load_standard_model(model_dir: Path):
    """Load a standard AutoModelForSequenceClassification + its label maps."""
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), trust_remote_code=True,
    )
    id2label = model.config.id2label
    label2id = model.config.label2id
    id2label = {int(k): str(v) for k, v in id2label.items()}
    label2id = {str(k): int(v) for k, v in label2id.items()}
    return model, id2label, label2id


def _load_multitask_model(model_dir: Path):
    """Load a MultitaskModel and label maps from label_config.json."""
    from ..training.multitask import load_multitask_model

    with open(model_dir / "label_config.json") as f:
        label_cfg = json.load(f)

    label2id = label_cfg["label2id"]
    id2label_raw = label_cfg["id2label"]
    id2label = {int(k): str(v) for k, v in id2label_raw.items()}
    label2id = {str(k): int(v) for k, v in label2id.items()}

    num_stance = label_cfg["num_stance_labels"]
    num_sarcasm = label_cfg.get("num_sarcasm_labels", 2)
    num_sentiment = label_cfg.get("num_sentiment_labels", 3)

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    from ..training.multitask import MultitaskModel
    model = MultitaskModel(config, num_stance, num_sarcasm, num_sentiment)

    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model weights found in {model_dir}. "
            "Expected model.safetensors or pytorch_model.bin"
        )
    model.load_state_dict(state_dict)
    return model, id2label, label2id


def _run_inference(model, dataset, is_multitask_model: bool) -> np.ndarray:
    """Run the model on dataset and return predicted label indices."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if is_multitask_model:
        from ..training.multitask import MultitaskTrainer
        trainer = MultitaskTrainer(
            model=model,
            args=_dummy_training_args(),
        )
    else:
        trainer = Trainer(
            model=model,
            args=_dummy_training_args(),
        )

    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return np.argmax(logits, axis=1)


def _dummy_training_args():
    """Minimal TrainingArguments for inference only."""
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir="/tmp/_eval_model_tmp",
        per_device_eval_batch_size=32,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )


def _compute_metrics(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro",
                                                 labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro",
                                           labels=labels, zero_division=0)),
    }


def _per_topic_metrics(y_true, y_pred, topics):
    results = {}
    for topic in sorted(set(topics)):
        mask = np.array(topics) == topic
        tt, tp = np.array(y_true)[mask], np.array(y_pred)[mask]
        t_labels = sorted(set(tt) | set(tp))
        cm = confusion_matrix(tt, tp, labels=t_labels)
        results[str(topic)] = {
            **_compute_metrics(tt, tp, t_labels),
            "support": int(mask.sum()),
            "labels": t_labels,
            "confusion_matrix": cm,
            "classification_report": classification_report(
                tt, tp, labels=t_labels, digits=4, zero_division=0,
            ),
        }
    return results


def _safe_dirname(topic: str) -> str:
    """Sanitise a topic string for use as a directory name."""
    return topic.replace("/", "_").replace("\\", "_").replace(" ", "_").strip("_") or "unknown"


def _save_per_topic_artifacts(per_topic: dict, out_dir: Path):
    """Save confusion matrix image + classification report for each topic."""
    _configure_arabic_fallback()
    topic_dir = out_dir / "per_topic"
    topic_dir.mkdir(parents=True, exist_ok=True)

    for topic, data in sorted(per_topic.items()):
        safe = _safe_dirname(topic)
        t_dir = topic_dir / safe
        t_dir.mkdir(parents=True, exist_ok=True)

        t_labels = data["labels"]
        cm = data["confusion_matrix"]

        report_path = t_dir / "classification_report.txt"
        lines = [
            f"Topic   : {topic}",
            f"Support : {data['support']}",
            "",
        ]
        for key, label in METRIC_LABELS.items():
            val = data.get(key)
            if val is not None:
                lines.append(f"  {label:<20s} {val:.4f}")
        lines.append("")
        lines.append(data["classification_report"])
        lines.append("")
        lines.append("Confusion Matrix (rows=true, cols=pred):")
        lines.append(f"Labels: {t_labels}")
        lines.append(np.array2string(cm))
        report_path.write_text("\n".join(lines), encoding="utf-8")

        fig, ax = plt.subplots(
            figsize=(max(6, len(t_labels) * 1.5), max(5, len(t_labels) * 1.2))
        )
        ar_t_labels = [_ar(str(l)) for l in t_labels]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=ar_t_labels, yticklabels=ar_t_labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        title = f"{_ar(topic)}  (Macro F1={data['macro_f1']:.4f})"
        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        fig.savefig(t_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    print(f"Saved per-topic results → {topic_dir}")


def _save_combined_confusion_matrices(overall_cm, overall_labels, overall_f1,
                                      per_topic, out_dir: Path):
    """Save a single image with overall + per-topic confusion matrices side by side."""
    _configure_arabic_fallback()

    sorted_topics = sorted(per_topic.keys())
    n_panels = 1 + len(sorted_topics)

    panel_w = max(5, max(len(overall_labels), *(len(per_topic[t]["labels"]) for t in sorted_topics)) * 1.4)
    panel_h = panel_w * 0.85
    fig, axes = plt.subplots(1, n_panels, figsize=(panel_w * n_panels, panel_h))
    if n_panels == 1:
        axes = [axes]

    ar_overall = [_ar(str(l)) for l in overall_labels]
    sns.heatmap(overall_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ar_overall, yticklabels=ar_overall, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Overall  (F1={overall_f1:.4f})", fontsize=11)

    for idx, topic in enumerate(sorted_topics, start=1):
        data = per_topic[topic]
        t_labels = data["labels"]
        cm = data["confusion_matrix"]
        ar_t = [_ar(str(l)) for l in t_labels]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=ar_t, yticklabels=ar_t, ax=axes[idx])
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
        axes[idx].set_title(f"{_ar(topic)}  (F1={data['macro_f1']:.4f})", fontsize=11)

    plt.tight_layout()
    path = out_dir / "confusion_matrices_combined.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved combined CM image → {path}")


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
}


def _build_summary(overall, per_topic, model_dir, data_path, n_samples, labels,
                   report_text, cm):
    lines = []
    lines.append("=" * 65)
    lines.append(f"  Model    : {model_dir}")
    lines.append(f"  Data     : {data_path}")
    lines.append(f"  Samples  : {n_samples}")
    lines.append("=" * 65)

    lines.append("")
    lines.append("OVERALL PERFORMANCE")
    lines.append("-" * 45)
    for key, label in METRIC_LABELS.items():
        val = overall.get(key)
        if val is not None:
            lines.append(f"  {label:<20s} {val:.4f}")

    lines.append("")
    lines.append("Classification Report:")
    lines.append(report_text)

    lines.append("Confusion Matrix (rows=true, cols=pred):")
    lines.append(f"Labels: {labels}")
    lines.append(np.array2string(cm))

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
            for key, label in METRIC_LABELS.items():
                val = t.get(key)
                if val is not None:
                    lines.append(f"    {label:<20s} {val:.4f}")
            if t.get("classification_report"):
                lines.append("")
                lines.append(t["classification_report"])

    lines.append("")
    return "\n".join(lines)


def evaluate_model(model_dir: str, data_path: str, output_dir: str = None,
                   text_col: str = None, topic_col: str = None,
                   label_col: str = None, max_length: int = 512,
                   normalize: bool = True, infer_only: bool = False):
    """Infer on CSV; if not infer_only, compute metrics and write summary/plots."""

    model_dir = Path(model_dir)
    data_path = Path(data_path)

    print(f"Model dir : {model_dir}")
    print(f"Data path : {data_path}")
    print(f"Mode      : {'inference only' if infer_only else 'evaluate'}")

    df = pd.read_csv(data_path)
    text_col = _detect_text_col(df, text_col)
    topic_col = _detect_topic_col(df, topic_col)

    has_labels = not infer_only
    if has_labels:
        label_col = _detect_label_col(df, label_col)

    print(f"Text col  : {text_col}")
    print(f"Topic col : {topic_col or '(none)'}")
    if has_labels:
        print(f"Label col : {label_col}")
    print(f"Samples   : {len(df)}")

    if normalize:
        df[text_col] = df[text_col].apply(normalize_arabic)
        if topic_col:
            df[topic_col] = df[topic_col].apply(normalize_arabic)

    multitask = _is_multitask(model_dir)
    print(f"Model type: {'multitask' if multitask else 'standard'}")

    if multitask:
        model, id2label, label2id = _load_multitask_model(model_dir)
    else:
        model, id2label, label2id = _load_standard_model(model_dir)

    print(f"Labels    : {label2id}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    if has_labels:
        gt_labels_str = df[label_col].astype(str).tolist()
        unknown = set(gt_labels_str) - set(label2id.keys())
        if unknown:
            print(f"WARNING: {len(unknown)} unknown label values will be kept as-is "
                  f"for evaluation: {unknown}")

    texts = df[text_col].tolist()
    topics = df[topic_col].tolist() if topic_col else None
    dataset = StanceDataset(
        texts=texts,
        topics=topics,
        labels=None,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    print(f"\nRunning inference on {len(dataset)} samples ...")
    pred_ids = _run_inference(model, dataset, multitask)
    pred_labels_str = [id2label[int(p)] for p in pred_ids]

    print(f"Prediction distribution:")
    print(pd.Series(pred_labels_str).value_counts().to_string())

    out = Path(output_dir) if output_dir else model_dir
    out.mkdir(parents=True, exist_ok=True)

    pred_df = df.copy()
    pred_df["prediction"] = pred_labels_str
    pred_csv_path = out / "predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved predictions → {pred_csv_path}")

    if infer_only:
        return None, None

    y_true = gt_labels_str
    y_pred = pred_labels_str
    all_labels = sorted(set(y_true) | set(y_pred))

    overall = _compute_metrics(y_true, y_pred, all_labels)
    report_text = classification_report(y_true, y_pred, labels=all_labels, digits=4,
                                        zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    per_topic = None
    if topic_col:
        per_topic = _per_topic_metrics(y_true, y_pred, df[topic_col].tolist())

    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    for key, label in METRIC_LABELS.items():
        print(f"  {label:<20s} {overall[key]:.4f}")
    print(f"\n{report_text}")

    if per_topic:
        print("Per-topic results:")
        for tname, tm in sorted(per_topic.items()):
            print(f"  {tname} (n={tm['support']}):  "
                  f"Acc={tm['accuracy']:.4f}  F1={tm['macro_f1']:.4f}  "
                  f"P={tm['macro_precision']:.4f}  R={tm['macro_recall']:.4f}")

    summary_text = _build_summary(
        overall, per_topic,
        model_dir=str(model_dir),
        data_path=str(data_path),
        n_samples=len(df),
        labels=all_labels,
        report_text=report_text,
        cm=cm,
    )
    summary_path = out / "eval_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\nSaved summary → {summary_path}")

    _configure_arabic_fallback()
    fig, ax = plt.subplots(
        figsize=(max(6, len(all_labels) * 1.5), max(5, len(all_labels) * 1.2))
    )
    ar_labels = [_ar(str(l)) for l in all_labels]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ar_labels, yticklabels=ar_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix  (Macro F1={overall['macro_f1']:.4f})")
    plt.tight_layout()
    cm_path = out / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved CM image → {cm_path}")

    if per_topic:
        _save_per_topic_artifacts(per_topic, out)
        _save_combined_confusion_matrices(
            cm, all_labels, overall["macro_f1"], per_topic, out,
        )

    return overall, per_topic


def main():
    parser = argparse.ArgumentParser(
        description="Run inference (and optional eval) with a saved model")
    parser.add_argument(
        "--model-dir", "-m", type=str, required=True,
        help="Saved model directory")
    parser.add_argument(
        "--data", "-d", type=str, required=True,
        help="Input CSV")
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory (default: model dir)")
    parser.add_argument(
        "--infer-only", action="store_true",
        help="Skip evaluation, just predict")
    parser.add_argument(
        "--text-col", type=str, default=None,
        help="Text column (auto-detected)")
    parser.add_argument(
        "--topic-col", type=str, default=None,
        help="Topic column (auto-detected)")
    parser.add_argument(
        "--label-col", type=str, default=None,
        help="Label column (auto-detected)")
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Max tokenization length")
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Skip Arabic text normalization")
    args = parser.parse_args()

    evaluate_model(
        model_dir=args.model_dir,
        data_path=args.data,
        output_dir=args.output_dir,
        text_col=args.text_col,
        topic_col=args.topic_col,
        label_col=args.label_col,
        max_length=args.max_length,
        normalize=not args.no_normalize,
        infer_only=args.infer_only,
    )


if __name__ == "__main__":
    main()
