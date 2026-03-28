"""LLM predictions for stance (and related tasks): run batch inference, write JSON/CSV/ZIP, optional eval."""

import os
import json
import zipfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

import pandas as pd
from tqdm import tqdm

from .client import LLMClient, LLMConfig, LLMProvider, LLMError
from .prompting import (
    StanceDetectionResponse,
    AugmentationResponse,
    detect_stance,
    detect,
    SYSTEM_PROMPT,
    TASK_CONFIG,
)
from ..data_loader import SUBTASK_CONFIG, normalize_arabic


@dataclass
class PredictionResult:
    id: int
    topic: str
    text: str
    predicted_label: str
    reasoning: str
    confidence: float
    true_label: Optional[str] = None
    error: Optional[str] = None


def load_data(
    subtask: str,
    split: str = "val",
    sample: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    config = SUBTASK_CONFIG[subtask]
    
    data_path = config.get(split)
    if data_path is None:
        raise ValueError(f"Unknown split '{split}' for subtask {subtask}")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
        print(f"Sampled {sample} rows from {len(pd.read_csv(data_path))} total")
    
    print(f"Loaded {len(df)} samples from {data_path}")
    return df


def run_predictions(
    df: pd.DataFrame,
    client: LLMClient,
    subtask: str,
    output_dir: Path,
    resume: bool = False,
    normalize: bool = True,
    batch_save_every: int = 10,
    include_labels: bool = False,
    prompt_template: Optional[str] = None,
    response_model: Optional[Any] = None,
    fallback_label: str = "neutral",
) -> List[PredictionResult]:
    config = SUBTASK_CONFIG[subtask]
    text_col = config["text_col"]
    topic_col = config["topic_col"]
    label_col = config["label_col"]

    existing_results = {}
    
    if resume:
        for candidate in ("llm_results_partial.json", "llm_results.json"):
            candidate_path = output_dir / candidate
            if candidate_path.exists():
                with open(candidate_path) as f:
                    existing_data = json.load(f)
                    existing_results = {r["id"]: r for r in existing_data.get("results", [])}
                errors = sum(1 for r in existing_results.values() if r.get("error"))
                print(f"Resuming from {candidate}: {len(existing_results)} predictions ({errors} errors to retry)")
                break
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        sample_id = row["id"]

        if sample_id in existing_results and not existing_results[sample_id].get("error"):
            results.append(PredictionResult(**existing_results[sample_id]))
            continue
        
        text = str(row[text_col])
        topic = str(row[topic_col]) if topic_col else None

        true_label = None
        if include_labels and label_col in row and pd.notna(row[label_col]):
            true_label = str(row[label_col])

        if normalize:
            text_for_llm = normalize_arabic(text)
            topic_for_llm = normalize_arabic(topic) if topic else None
        else:
            text_for_llm = text
            topic_for_llm = topic
        
        try:
            extra_kwargs = {}
            if response_model is AugmentationResponse and true_label:
                extra_kwargs["stance"] = true_label

            if prompt_template is not None and response_model is not None:
                response = detect(
                    topic=topic_for_llm or "",
                    text=text_for_llm,
                    prompt_template=prompt_template,
                    response_model=response_model,
                    client=client,
                    **extra_kwargs,
                )
            else:
                response = detect_stance(
                    topic=topic_for_llm or "",
                    text=text_for_llm,
                    client=client,
                )
            
            if isinstance(response, AugmentationResponse):
                result = PredictionResult(
                    id=sample_id,
                    topic=topic or "",
                    text=text,
                    predicted_label=json.dumps(
                        {"aug1": response.aug1, "aug2": response.aug2, "aug3": response.aug3},
                        ensure_ascii=False,
                    ),
                    reasoning="augmentation",
                    confidence=1.0,
                    true_label=true_label,
                )
            else:
                result = PredictionResult(
                    id=sample_id,
                    topic=topic or "",
                    text=text,
                    predicted_label=str(response.label.value),
                    reasoning=response.reasoning,
                    confidence=response.confidence,
                    true_label=true_label,
                )
        except LLMError as e:
            print(f"\nError on sample {sample_id}: {e}")
            result = PredictionResult(
                id=sample_id,
                topic=topic or "",
                text=text,
                predicted_label=fallback_label,
                reasoning="",
                confidence=0.0,
                true_label=true_label,
                error=str(e),
            )
        
        results.append(result)

        if len(results) % batch_save_every == 0:
            save_partial_results(results, output_dir)
    
    return results


def save_partial_results(results: List[PredictionResult], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "llm_results_partial.json"
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "count": len(results),
        "results": [asdict(r) for r in results],
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_final_results(
    results: List[PredictionResult],
    output_dir: Path,
    model_name: str,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    full_results_path = output_dir / "llm_results.json"

    errors = [r for r in results if r.error]
    avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
    
    full_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "total_samples": len(results),
            "errors": len(errors),
            "average_confidence": avg_confidence,
        },
        "results": [asdict(r) for r in results],
    }
    
    with open(full_results_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    print(f"Saved full results to {full_results_path}")

    prediction_csv_path = output_dir / "prediction.csv"
    
    pred_df = pd.DataFrame([
        {"id": r.id, "label": r.predicted_label}
        for r in results
    ])
    pred_df.to_csv(prediction_csv_path, index=False)
    print(f"Saved prediction.csv to {prediction_csv_path}")

    zip_path = output_dir / "prediction.zip"
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_csv_path, "prediction.csv")
    print(f"Created {zip_path}")

    partial_path = output_dir / "llm_results_partial.json"
    if partial_path.exists():
        partial_path.unlink()
    
    return {
        "full_results": full_results_path,
        "prediction_csv": prediction_csv_path,
        "prediction_zip": zip_path,
    }


def evaluate_predictions(
    results: List[PredictionResult],
    output_dir: Path,
) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        classification_report,
        confusion_matrix,
    )

    labeled = [r for r in results if r.true_label is not None]
    
    if not labeled:
        print("No ground truth labels available for evaluation")
        return {}
    
    y_true = [r.true_label for r in labeled]
    y_pred = [r.predicted_label for r in labeled]

    labels = sorted(set(y_true) | set(y_pred))
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=labels),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=labels),
        "per_class_f1": {
            label: f1_score(y_true, y_pred, labels=[label], average="micro")
            for label in labels
        },
    }

    report = classification_report(y_true, y_pred, labels=labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nSamples evaluated: {len(labeled)}")
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"\nPer-class F1:")
    for label, f1 in metrics["per_class_f1"].items():
        print(f"  {label}: {f1:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix (rows=true, cols=pred):")
    print(f"Labels: {labels}")
    print(cm)
    print("=" * 60)

    eval_path = output_dir / "evaluation.json"
    eval_data = {
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "per_class_f1": metrics["per_class_f1"],
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "num_samples": len(labeled),
    }
    
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved evaluation to {eval_path}")

    errors_analysis = analyze_errors(results, output_dir)
    
    return metrics


def analyze_errors(
    results: List[PredictionResult],
    output_dir: Path,
) -> Dict[str, Any]:
    errors = [
        r for r in results
        if r.true_label is not None and r.predicted_label != r.true_label
    ]
    
    if not errors:
        print("No misclassifications to analyze!")
        return {}

    error_types = {}
    for r in errors:
        key = f"{r.true_label} -> {r.predicted_label}"
        if key not in error_types:
            error_types[key] = []
        error_types[key].append({
            "id": r.id,
            "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
            "topic": r.topic,
            "reasoning": r.reasoning,
            "confidence": r.confidence,
        })

    errors_path = output_dir / "error_analysis.json"
    error_data = {
        "total_errors": len(errors),
        "error_rate": len(errors) / len([r for r in results if r.true_label]),
        "by_type": {k: {"count": len(v), "samples": v[:5]} for k, v in error_types.items()},
    }
    
    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(error_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nError Analysis:")
    print(f"  Total misclassifications: {len(errors)}")
    print(f"  Error breakdown:")
    for error_type, samples in sorted(error_types.items(), key=lambda x: -len(x[1])):
        print(f"    {error_type}: {len(samples)}")
    print(f"\nSaved error analysis to {errors_path}")
    
    return error_data


def save_augmentation_csv(
    results: List[PredictionResult],
    output_dir: Path,
    model_name: str,
) -> Dict[str, Path]:
    """
    Expand augmentation results into a training-format CSV.

    Each successful result produces 3 rows (aug1/aug2/aug3) with columns
    matching the original training data: id, sentence, topic, label.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    for r in results:
        if r.error:
            skipped += 1
            continue
        try:
            augs = json.loads(r.predicted_label)
        except json.JSONDecodeError:
            skipped += 1
            continue
        for key in ("aug1", "aug2", "aug3"):
            if key in augs and augs[key]:
                rows.append({
                    "id": f"{r.id}_{key}",
                    "sentence": augs[key],
                    "topic": r.topic,
                    "label": r.true_label,
                })

    aug_csv_path = output_dir / "augmented.csv"
    df = pd.DataFrame(rows)
    df.to_csv(aug_csv_path, index=False)

    print(f"\nAugmentation: {len(rows)} new samples from {len(results)} originals "
          f"({skipped} skipped due to errors)")
    print(f"Saved augmented.csv to {aug_csv_path}")

    return {"augmented_csv": aug_csv_path}


def print_summary(results: List[PredictionResult]):
    total = len(results)
    errors = sum(1 for r in results if r.error)

    label_counts = {}
    for r in results:
        label_counts[r.predicted_label] = label_counts.get(r.predicted_label, 0) + 1

    confidences = [r.confidence for r in results if not r.error]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    min_conf = min(confidences) if confidences else 0
    max_conf = max(confidences) if confidences else 0
    
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"API Errors: {errors}")
    print(f"\nPredicted label distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / total * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    print(f"\nConfidence stats:")
    print(f"  Average: {avg_conf:.3f}")
    print(f"  Min: {min_conf:.3f}")
    print(f"  Max: {max_conf:.3f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM predictions on a subtask split"
    )
    parser.add_argument(
        "-st", "--subtask", type=str, default="B", choices=["A", "B"],
        help="Subtask (default: B)"
    )
    parser.add_argument(
        "-s", "--split", type=str, default="val", choices=["train", "val", "test"],
        help="Data split (default: val)"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sample N rows from data"
    )
    parser.add_argument(
        "-m", "--model", type=str,
        help="LLM model to use"
    )
    parser.add_argument(
        "-p", "--provider", type=str,
        choices=["openai", "openrouter", "local"],
        help="LLM provider (default: openrouter)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from partial results"
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Disable Arabic text normalization"
    )
    parser.add_argument(
        "-t", "--temperature", type=float,
        help="LLM temperature"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock LLM"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--task", type=str, default="stance",
        choices=list(TASK_CONFIG.keys()),
        help=f"Detection task: {', '.join(TASK_CONFIG.keys())} (default: stance)"
    )
    
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = f"_sample{args.sample}" if args.sample else ""
        output_dir = Path(__file__).parent / "outputs" / f"llm_subtask_{args.subtask}_{args.split}{suffix}"
    
    print(f"Output directory: {output_dir}")

    if args.mock:
        client = LLMClient.mock()
        model_name = "mock"
    else:
        config = LLMConfig.from_env()

        if args.provider:
            config.provider = LLMProvider(args.provider)
        if args.model:
            config.model = args.model
        if args.temperature:
            config.temperature = args.temperature
        
        client = LLMClient(config)
        model_name = config.model

    task_cfg = TASK_CONFIG[args.task]
    prompt_template = task_cfg["prompt"]
    response_model = task_cfg["response_model"]
    
    print(f"Using model: {model_name}")
    print(f"Task: {args.task}")
    print(f"Split: {args.split}")

    df = load_data(
        subtask=args.subtask,
        split=args.split,
        sample=args.sample,
        seed=args.seed,
    )

    include_labels = args.split in ("train", "val")
    results = run_predictions(
        df=df,
        client=client,
        subtask=args.subtask,
        output_dir=output_dir,
        resume=args.resume,
        normalize=not args.no_normalize,
        include_labels=include_labels,
        prompt_template=prompt_template,
        response_model=response_model,
    )

    paths = save_final_results(results, output_dir, model_name)
    
    if args.task == "augmentation":
        aug_paths = save_augmentation_csv(results, output_dir, model_name)
        paths.update(aug_paths)
    else:
        print_summary(results)

        if include_labels and args.task == "stance":
            evaluate_predictions(results, output_dir)
        elif include_labels:
            print(f"\nSkipping evaluation: ground truth labels are stance, but task is '{args.task}'")
    
    print(f"\nFiles created:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
