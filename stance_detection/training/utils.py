"""Shared training helpers: CLI, data/model/TrainingArguments, metrics, eval/predict, single-phase train."""

import os
import zipfile
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)

from ..config import TrainConfig
from ..data_loader import load_subtask


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_common_args(parser: argparse.ArgumentParser):
    """Register CLI arguments shared by all training scripts."""
    parser.add_argument("-st", "--subtask", type=str, default="B", choices=["A", "B"])
    parser.add_argument("-m", "--model", type=str, default=None, help="Model name/path")
    parser.add_argument("-e", "--epochs", type=int, default=None)
    parser.add_argument("-bs", "--batch-size", type=int, default=None)
    parser.add_argument("-lr", "--lr", type=float, default=None)
    parser.add_argument("-ml", "--max-length", type=int, default=None)
    parser.add_argument("-od", "--output-dir", type=str, default=None)
    parser.add_argument("-nn", "--no-normalize", action="store_true",
                        help="Disable Arabic normalization")
    parser.add_argument("-nck", "--no-checkpoints", action="store_true",
                        help="Skip intermediate checkpoint saves")
    parser.add_argument("-ap", "--arabert-prep", type=str, default=None,
                        help="AraBERT model name for preprocessing")
    parser.add_argument("-tp", "--train-path", type=str, default=None,
                        help="Custom training CSV path")
    parser.add_argument("-vp", "--val-path", type=str, default=None,
                        help="Custom validation CSV path with labels")
    parser.add_argument("-cd", "--classifier-dropout", type=float, default=None,
                        help="Classification head dropout")
    parser.add_argument("-cv", "--cross-validate", action="store_true",
                        help="Run cross-validation")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("-saf", "--save-all-folds", action="store_true",
                        help="Save model from each fold")


def make_config(args, output_dir: str = None) -> TrainConfig:
    """Create a TrainConfig with CLI overrides applied.

    Uses getattr so it works regardless of which args the caller defined.
    """
    config = TrainConfig(subtask=getattr(args, "subtask", "B"))

    if getattr(args, "model", None):
        config.model_name = args.model
    if getattr(args, "epochs", None):
        config.num_epochs = args.epochs
    if getattr(args, "batch_size", None):
        config.batch_size = args.batch_size
    if getattr(args, "lr", None):
        config.learning_rate = args.lr
    if getattr(args, "max_length", None):
        config.max_length = args.max_length
    if getattr(args, "output_dir", None):
        config.output_dir = args.output_dir
    if getattr(args, "no_normalize", False):
        config.normalize_arabic = False
    if getattr(args, "no_checkpoints", False):
        config.save_checkpoints = False
    if getattr(args, "classifier_dropout", None) is not None:
        config.classifier_dropout = args.classifier_dropout
    if getattr(args, "seed", None):
        config.seed = args.seed

    if output_dir:
        config.output_dir = output_dir
    return config


def make_training_args(config: TrainConfig, eval_dataset=None,
                       **extra) -> TrainingArguments:
    """Build TrainingArguments from TrainConfig; extra kwargs forwarded to HF."""
    save_ckpt = config.save_checkpoints
    use_best = save_ckpt and bool(eval_dataset)
    kwargs = dict(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch" if save_ckpt else "no",
        save_total_limit=2 if save_ckpt else None,
        load_best_model_at_end=use_best,
        metric_for_best_model="macro_f1" if use_best else None,
        report_to="none",
        seed=config.seed,
        fp16=torch.cuda.is_available(),
    )
    kwargs.update(extra)
    return TrainingArguments(**kwargs)


def load_data(config: TrainConfig, arabert_model=None, train_path=None,
              val_path=None, load_val_labels=False, load_test=False):
    """Load tokenizer and datasets, print a summary.

    Returns (tokenizer, data_dict).
    """
    print(f"\nLoading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    print(f"Loading data for Subtask {config.subtask}...")
    data = load_subtask(
        subtask=config.subtask,
        tokenizer=tokenizer,
        max_length=config.max_length,
        normalize=config.normalize_arabic,
        arabert_model=arabert_model,
        train_path=train_path,
        val_path=val_path,
        load_val_labels=load_val_labels,
        load_test=load_test,
    )

    print(f"  Train:  {len(data['train_dataset'])} samples")
    val_status = "with labels" if load_val_labels else "unlabeled"
    print(f"  Val:    {len(data['val_dataset'])} samples ({val_status})")
    if load_test and "test_dataset" in data:
        print(f"  Test:   {len(data['test_dataset'])} samples (unlabeled)")

    return tokenizer, data


def build_model(config: TrainConfig, data: dict):
    """Initialise a fresh AutoModelForSequenceClassification."""
    kwargs = dict(
        num_labels=data["num_labels"],
        id2label=data["id2label"],
        label2id=data["label2id"],
        trust_remote_code=True,
    )
    if config.classifier_dropout is not None:
        kwargs["classifier_dropout"] = config.classifier_dropout
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_name, **kwargs,
    )


def get_eval_dataset(data: dict, load_val_labels: bool):
    """Return the val dataset if labels were loaded, else None."""
    if load_val_labels:
        return data["val_dataset"]
    return None


def compute_metrics(eval_pred):
    """Metric function for HuggingFace Trainer (handles tuple logits)."""
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }


def _extract_logits(predictions):
    """Get logits from Trainer predictions (handles hidden-state tuples)."""
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


def evaluate_model(trainer, data, config: TrainConfig,
                   dataset_name: str = "validation"):
    """Evaluate on a labelled set, print report, save artefacts."""
    print(f"\n{'=' * 50}")
    print(f"EVALUATING ON {dataset_name.upper()} SET")
    print(f"{'=' * 50}")

    predictions = trainer.predict(data["val_dataset"])
    logits = _extract_logits(predictions)

    pred_ids = np.argmax(logits, axis=1)
    pred_labels = [data["id2label"][p] for p in pred_ids]
    true_labels = predictions.label_ids
    true_label_names = [data["id2label"][t] for t in true_labels]

    accuracy = accuracy_score(true_labels, pred_ids)
    macro_f1 = f1_score(true_labels, pred_ids, average="macro")
    weighted_f1 = f1_score(true_labels, pred_ids, average="weighted")

    print(f"\n{dataset_name.capitalize()} Metrics:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    report_str = classification_report(true_label_names, pred_labels, digits=4)
    print(f"\nClassification Report:\n{report_str}")

    os.makedirs(config.output_dir, exist_ok=True)
    report_path = os.path.join(config.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report - {dataset_name.capitalize()} Set\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(report_str)
    print(f"Saved report to {report_path}")

    label_names = sorted(set(true_label_names))
    cm = confusion_matrix(true_label_names, pred_labels, labels=label_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={"label": "Count"})
    plt.title(f"Confusion Matrix - {dataset_name.capitalize()} Set\n"
              f"Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    cm_path = os.path.join(config.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predictions": pred_labels,
        "true_labels": true_label_names,
    }


def predict(trainer, data, config: TrainConfig,
            dataset_key: str = "test", output_suffix: str = "test"):
    """Generate and save predictions."""
    dataset_name = "test" if dataset_key == "test" else "validation"
    dataset = data[f"{dataset_key}_dataset"]
    df = data[f"{dataset_key}_df"]

    print(f"\n{'=' * 50}")
    print(f"PREDICTING ON {dataset_name.upper()} SET")
    print(f"{'=' * 50}")

    predictions = trainer.predict(dataset)
    logits = _extract_logits(predictions)

    pred_ids = np.argmax(logits, axis=1)
    pred_labels = [data["id2label"][p] for p in pred_ids]

    result_df = df.copy()
    result_df["prediction"] = pred_labels

    print(f"\nPrediction distribution:")
    print(pd.Series(pred_labels).value_counts())

    predictions_csv = config.predictions_csv.replace(".csv", f"_{output_suffix}.csv")
    predictions_zip = config.predictions_zip.replace(".zip", f"_{output_suffix}.zip")

    result_df.to_csv(predictions_csv, index=False)
    print(f"Saved predictions to {predictions_csv}")

    with zipfile.ZipFile(predictions_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(predictions_csv, os.path.basename(predictions_csv))
    print(f"Created ZIP: {predictions_zip}")

    return result_df


def run_single_phase(args, trainer_factory, method_label: str = ""):
    """Train on train, eval on val, save model, run val predictions."""
    config = make_config(args)

    label = f" ({method_label})" if method_label else ""
    print(f"\n{'=' * 70}")
    print(f"TRAINING{label}")
    print(f"{'=' * 70}")
    print("Train on train set  ->  Evaluate on val set")
    print(f"{'=' * 70}")

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    arabert = getattr(args, "arabert_prep", None)
    if arabert:
        print(f"AraBERT preprocessing: {arabert}")
    if config.classifier_dropout is not None:
        print(f"Classifier dropout: {config.classifier_dropout}")

    train_path = getattr(args, "train_path", None)
    val_path = getattr(args, "val_path", None)

    set_seed(config.seed)
    tokenizer, data = load_data(
        config, arabert_model=arabert,
        train_path=train_path, val_path=val_path,
        load_val_labels=True,
    )

    eval_ds = get_eval_dataset(data, load_val_labels=True)
    model = build_model(config, data)
    training_args = make_training_args(config, eval_dataset=eval_ds)
    trainer = trainer_factory(model, training_args, data["train_dataset"], eval_ds)

    print(f"\n{'=' * 50}\nTRAINING\n{'=' * 50}")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    eval_results = evaluate_model(trainer, data, config)

    predict(trainer, data, config,
            dataset_key="val", output_suffix="val")

    print(f"\n\n{'=' * 70}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Model:     {config.output_dir}")
    print(f"Macro F1:  {eval_results['macro_f1']:.4f}")
    print(f"{'=' * 70}")
