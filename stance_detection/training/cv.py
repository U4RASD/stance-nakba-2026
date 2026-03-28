"""Stratified K-fold CV helpers: FoldContext, metrics, CrossValidator, fold TrainingArguments."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report,
)
from transformers import AutoTokenizer, TrainingArguments

from ..config import TrainConfig
from ..data_loader import normalize_arabic, SUBTASK_CONFIG


@dataclass
class CVConfig:
    """Cross-validation configuration."""
    n_folds: int = 5
    save_all_folds: bool = False


@dataclass
class FoldContext:
    """Everything a fold runner needs to train and evaluate one fold."""
    fold_idx: int
    n_folds: int
    fold_train_df: pd.DataFrame
    fold_val_df: pd.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    tokenizer: Any
    text_col: str
    topic_col: Optional[str]
    label_col: str
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    num_labels: int
    train_config: TrainConfig
    output_dir: str
    save_fold_model: bool
    prepared_data: Any = None


# Type alias
FoldRunnerFn = Callable[[FoldContext], Dict[str, float]]


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cv_compute_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer (handles tuple outputs)."""
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'macro_f1': f1_score(labels, predictions, average='macro'),
        'weighted_f1': f1_score(labels, predictions, average='weighted'),
        'macro_precision': precision_score(labels, predictions, average='macro'),
        'macro_recall': recall_score(labels, predictions, average='macro'),
    }


def compute_per_topic_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    fold_val_df: pd.DataFrame,
    topic_col: str,
    id2label: Dict[int, str],
    num_labels: int,
) -> Dict[str, Dict[str, float]]:
    """Compute per-topic metrics for a fold.

    Returns a dict mapping topic name -> {accuracy, macro_f1, weighted_f1,
    macro_precision, macro_recall, support, classification_report}.
    """
    topics = fold_val_df[topic_col].values
    unique_topics = sorted(set(topics))
    per_topic: Dict[str, Dict[str, float]] = {}

    for topic in unique_topics:
        mask = topics == topic
        t_preds = preds[mask]
        t_labels = labels[mask]

        if len(t_labels) == 0:
            continue

        target_names = [id2label[i] for i in range(num_labels)]
        present_labels = sorted(set(t_labels) | set(t_preds))
        present_names = [id2label[i] for i in present_labels]

        per_topic[str(topic)] = {
            'accuracy': float(accuracy_score(t_labels, t_preds)),
            'macro_f1': float(f1_score(t_labels, t_preds, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(t_labels, t_preds, average='weighted', zero_division=0)),
            'macro_precision': float(precision_score(t_labels, t_preds, average='macro', zero_division=0)),
            'macro_recall': float(recall_score(t_labels, t_preds, average='macro', zero_division=0)),
            'support': int(mask.sum()),
            'classification_report': classification_report(
                t_labels, t_preds,
                labels=present_labels,
                target_names=present_names,
                zero_division=0,
            ),
        }

    return per_topic


def evaluate_fold(trainer, val_dataset, id2label, num_labels,
                  fold_val_df=None, topic_col=None):
    """Standard fold evaluation: evaluate + predict + classification report.

    Returns a metrics dict with: accuracy, macro_f1, weighted_f1,
    classification_report.  Works for any training method.

    If fold_val_df and topic_col are provided, per-topic metrics are
    included under the per_topic_metrics key.
    """
    eval_results = trainer.evaluate()

    predictions = trainer.predict(val_dataset)
    if isinstance(predictions.predictions, tuple):
        logits = predictions.predictions[0]
    else:
        logits = predictions.predictions
    preds = np.argmax(logits, axis=1)
    labels = predictions.label_ids

    target_names = [id2label[i] for i in range(num_labels)]
    report = classification_report(labels, preds, target_names=target_names)

    result = {
        'accuracy': eval_results['eval_accuracy'],
        'macro_f1': eval_results['eval_macro_f1'],
        'weighted_f1': eval_results['eval_weighted_f1'],
        'macro_precision': eval_results['eval_macro_precision'],
        'macro_recall': eval_results['eval_macro_recall'],
        'classification_report': report,
    }

    if fold_val_df is not None and topic_col and topic_col in fold_val_df.columns:
        result['per_topic_metrics'] = compute_per_topic_metrics(
            preds, labels, fold_val_df, topic_col, id2label, num_labels,
        )

    return result


def make_fold_training_args(ctx: FoldContext, **extra) -> TrainingArguments:
    """Create standard TrainingArguments for a CV fold.

    Pass extra keyword arguments to override or add fields
    (e.g. remove_unused_columns=False for multitask).
    """
    save_ckpt = ctx.train_config.save_checkpoints
    kwargs = dict(
        output_dir=ctx.output_dir,
        num_train_epochs=ctx.train_config.num_epochs,
        per_device_train_batch_size=ctx.train_config.batch_size,
        per_device_eval_batch_size=ctx.train_config.batch_size,
        warmup_ratio=ctx.train_config.warmup_ratio,
        weight_decay=ctx.train_config.weight_decay,
        learning_rate=ctx.train_config.learning_rate,
        logging_dir=f'./logs/fold_{ctx.fold_idx}',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch" if save_ckpt else "no",
        save_total_limit=1 if save_ckpt else None,
        load_best_model_at_end=save_ckpt,
        metric_for_best_model="macro_f1" if save_ckpt else None,
        report_to="none",
        seed=ctx.train_config.seed,
        fp16=torch.cuda.is_available(),
    )
    kwargs.update(extra)
    return TrainingArguments(**kwargs)


class CrossValidator:
    """Load train CSV, split folds, call fold_runner_fn(ctx), aggregate and save JSON/CSV."""

    def __init__(
        self,
        train_config: TrainConfig,
        cv_config: CVConfig,
        fold_runner_fn: FoldRunnerFn,
        method_name: str = "unknown",
        arabert_model: str = None,
        train_path: str = None,
        prepare_fn: Callable = None,
    ):
        self.train_config = train_config
        self.cv_config = cv_config
        self.fold_runner_fn = fold_runner_fn
        self.method_name = method_name
        self.arabert_model = arabert_model
        self.train_path = train_path
        self.prepare_fn = prepare_fn

        self.fold_results: List[Dict[str, float]] = []
        self.fold_reports: List[str] = []
        self.fold_topic_results: List[Dict[str, Dict[str, float]]] = []
        self.best_fold_idx: int = -1
        self.best_fold_score: float = -1.0

    def run(self) -> Dict[str, Any]:
        """Run full cross-validation and return aggregated results."""

        set_seed(self.train_config.seed)

        print("=" * 60)
        print(f"CROSS-VALIDATION: {self.cv_config.n_folds}-Fold")
        print(f"Method: {self.method_name}")
        print("=" * 60)

        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")

        print(f"\nLoading tokenizer: {self.train_config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.train_config.model_name, trust_remote_code=True
        )

        subtask_config = SUBTASK_CONFIG[self.train_config.subtask]
        train_csv = self.train_path if self.train_path else subtask_config["train"]
        train_df = pd.read_csv(train_csv)

        text_col = subtask_config["text_col"]
        topic_col = subtask_config["topic_col"]
        label_col = subtask_config["label_col"]

        print(f"Training data: {train_csv} ({len(train_df)} samples)")

        if self.train_config.normalize_arabic:
            train_df[text_col] = train_df[text_col].apply(normalize_arabic)
            if topic_col:
                train_df[topic_col] = train_df[topic_col].apply(normalize_arabic)

        unique_labels = sorted(train_df[label_col].dropna().unique())
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

        print(f"Labels: {label2id}")

        # Optional preparation step (e.g. pre-labeling for multitask)
        prepared_data = None
        if self.prepare_fn:
            prepared_data = self.prepare_fn(train_df, subtask_config)

        labels = train_df[label_col].values
        skf = StratifiedKFold(
            n_splits=self.cv_config.n_folds,
            shuffle=True,
            random_state=self.train_config.seed,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, labels)):
            print(f"\n{'=' * 60}")
            print(f"FOLD {fold_idx + 1}/{self.cv_config.n_folds}")
            print(f"{'=' * 60}")
            print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

            ctx = FoldContext(
                fold_idx=fold_idx,
                n_folds=self.cv_config.n_folds,
                fold_train_df=fold_train_df,
                fold_val_df=fold_val_df,
                train_idx=train_idx,
                val_idx=val_idx,
                tokenizer=tokenizer,
                text_col=text_col,
                topic_col=topic_col,
                label_col=label_col,
                label2id=label2id,
                id2label=id2label,
                num_labels=num_labels,
                train_config=self.train_config,
                output_dir=f"{self.train_config.output_dir}/fold_{fold_idx}",
                save_fold_model=self.cv_config.save_all_folds,
                prepared_data=prepared_data,
            )

            fold_metrics = self.fold_runner_fn(ctx)

            self.fold_results.append(fold_metrics)
            self.fold_reports.append(fold_metrics.get('classification_report', ''))

            per_topic = fold_metrics.pop('per_topic_metrics', None)
            self.fold_topic_results.append(per_topic if per_topic else {})

            if fold_metrics['macro_f1'] > self.best_fold_score:
                self.best_fold_score = fold_metrics['macro_f1']
                self.best_fold_idx = fold_idx

            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Accuracy:        {fold_metrics['accuracy']:.4f}")
            print(f"  Macro F1:        {fold_metrics['macro_f1']:.4f}")
            print(f"  Weighted F1:     {fold_metrics['weighted_f1']:.4f}")
            print(f"  Macro Precision: {fold_metrics['macro_precision']:.4f}")
            print(f"  Macro Recall:    {fold_metrics['macro_recall']:.4f}")

            if 'classification_report' in fold_metrics:
                print(f"\n  Classification Report (Fold {fold_idx + 1}):")
                print(fold_metrics['classification_report'])

            if per_topic:
                print(f"\n  Per-Topic Results (Fold {fold_idx + 1}):")
                for topic_name, topic_metrics in sorted(per_topic.items()):
                    print(f"    Topic: {topic_name} (n={topic_metrics['support']})")
                    print(f"      Accuracy: {topic_metrics['accuracy']:.4f}  "
                          f"Macro F1: {topic_metrics['macro_f1']:.4f}  "
                          f"Weighted F1: {topic_metrics['weighted_f1']:.4f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results = self._aggregate_results()
        self._save_results(results)

        return results

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all folds."""

        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 60)

        metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']

        fold_results_for_json = [
            {k: v for k, v in fold.items() if k != 'classification_report'}
            for fold in self.fold_results
        ]

        results: Dict[str, Any] = {
            'method': self.method_name,
            'n_folds': self.cv_config.n_folds,
            'fold_results': fold_results_for_json,
            'fold_reports': self.fold_reports,
            'best_fold_idx': self.best_fold_idx,
        }

        print(f"\nMethod: {self.method_name}")
        print(f"Folds: {self.cv_config.n_folds}")
        print()

        for metric in metrics:
            values = [fold[metric] for fold in self.fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)

            results[f'{metric}_mean'] = mean_val
            results[f'{metric}_std'] = std_val
            results[f'{metric}_values'] = values

            print(f"{metric.upper()}")
            print(f"  Mean:  {mean_val:.4f} +/- {std_val:.4f}")
            print(f"  Folds: {[f'{v:.4f}' for v in values]}")

        print(f"\nBest fold: {self.best_fold_idx + 1} "
              f"(Macro F1: {self.best_fold_score:.4f})")

        # --- Per-topic aggregation across folds ---
        has_topic_results = any(tr for tr in self.fold_topic_results)
        if has_topic_results:
            topic_agg = self._aggregate_topic_results()
            results['per_topic_results'] = topic_agg

        return results

    def _aggregate_topic_results(self) -> Dict[str, Any]:
        """Aggregate per-topic metrics across folds (mean +/- std)."""

        topic_metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']

        all_topics: set = set()
        for fold_topics in self.fold_topic_results:
            all_topics.update(fold_topics.keys())

        topic_agg: Dict[str, Any] = {}

        print("\n" + "=" * 60)
        print("PER-TOPIC CROSS-VALIDATION RESULTS")
        print("=" * 60)

        for topic in sorted(all_topics):
            fold_values: Dict[str, List[float]] = {m: [] for m in topic_metrics}
            fold_reports: List[str] = []
            supports: List[int] = []

            for fold_idx, fold_topics in enumerate(self.fold_topic_results):
                if topic in fold_topics:
                    for m in topic_metrics:
                        fold_values[m].append(fold_topics[topic][m])
                    supports.append(fold_topics[topic]['support'])
                    fold_reports.append(fold_topics[topic].get('classification_report', ''))

            agg: Dict[str, Any] = {
                'n_folds_present': len(supports),
                'total_support': sum(supports),
            }

            print(f"\nTopic: {topic}  "
                  f"(present in {agg['n_folds_present']}/{self.cv_config.n_folds} folds, "
                  f"total samples: {agg['total_support']})")

            for m in topic_metrics:
                vals = fold_values[m]
                if vals:
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals))
                    agg[f'{m}_mean'] = mean_val
                    agg[f'{m}_std'] = std_val
                    agg[f'{m}_values'] = vals
                    print(f"  {m.upper():20s} Mean: {mean_val:.4f} +/- {std_val:.4f}  "
                          f"Folds: {[f'{v:.4f}' for v in vals]}")

            agg['fold_reports'] = fold_reports
            topic_agg[topic] = agg

        return topic_agg

    def _save_results(self, results: Dict[str, Any]):
        """Save CV results to JSON and per-topic results to CSV."""

        output_dir = Path(self.train_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "cv_results.json"

        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [deep_convert(i) for i in obj]
            return convert_for_json(obj)

        results_json = deep_convert(results)

        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"\nResults saved to: {results_path}")

        # Save per-topic results as a CSV for easy analysis
        if 'per_topic_results' in results:
            topic_metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
            rows = []
            for topic, agg in results['per_topic_results'].items():
                row = {'topic': topic, 'n_folds_present': agg['n_folds_present'],
                       'total_support': agg['total_support']}
                for m in topic_metrics:
                    row[f'{m}_mean'] = agg.get(f'{m}_mean')
                    row[f'{m}_std'] = agg.get(f'{m}_std')
                rows.append(row)

            topic_df = pd.DataFrame(rows).sort_values('topic').reset_index(drop=True)
            topic_csv_path = output_dir / "cv_per_topic_results.csv"
            topic_df.to_csv(topic_csv_path, index=False)
            print(f"Per-topic results saved to: {topic_csv_path}")
