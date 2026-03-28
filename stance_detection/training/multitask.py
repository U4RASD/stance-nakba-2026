"""Multitask stance + sarcasm + sentiment: labeling, datasets, model, trainer, train/predict."""

import os
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    PreTrainedModel,
)

from ..config import TrainConfig
from ..data_loader import normalize_arabic, SUBTASK_CONFIG
from .utils import set_seed, make_training_args, compute_metrics

SARCASM_MODEL = "MohamedGalal/arabert-sarcasm-detector"
SENTIMENT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"

CACHE_DIR = Path(__file__).parent.parent / "cache"


def get_cache_path(subtask: str, train_path: str = None) -> Path:
    """Return the cache path for auxiliary labels."""
    CACHE_DIR.mkdir(exist_ok=True)
    if train_path:
        import hashlib
        h = hashlib.md5(train_path.encode()).hexdigest()[:8]
        return CACHE_DIR / f"multitask_labels_subtask_{subtask}_{h}.json"
    return CACHE_DIR / f"multitask_labels_subtask_{subtask}.json"


def label_with_model(texts: List[str], model_name: str, task_name: str,
                     batch_size: int = 32, device: str = None) -> List[int]:
    """Label texts using a pretrained HuggingFace classifier."""
    from transformers import AutoModelForSequenceClassification

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[Labeling {task_name}]")
    print(f"  Model: {model_name}  Device: {device}  Samples: {len(texts)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, trust_remote_code=True
    ).to(device)
    model.eval()

    labels: List[int] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Labeling {task_name}"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(device)
            preds = torch.argmax(model(**inputs).logits, dim=-1).cpu().tolist()
            labels.extend(preds)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return labels


def label_dataset(train_df: pd.DataFrame, text_col: str,
                  use_cache: bool = True, subtask: str = "B",
                  train_path: str = None) -> Dict[str, List[int]]:
    """Label training data with sarcasm & sentiment (with caching)."""
    cache_path = get_cache_path(subtask, train_path)

    if use_cache and cache_path.exists():
        print(f"\n[Loading cached labels from {cache_path}]")
        with open(cache_path, "r") as f:
            cached = json.load(f)
        if len(cached.get("sarcasm_labels", [])) == len(train_df):
            print(f"  Sarcasm: {len(cached['sarcasm_labels'])}  "
                  f"Sentiment: {len(cached['sentiment_labels'])}")
            return cached
        print("  Cache size mismatch, re-labeling...")

    texts = train_df[text_col].tolist()
    result = {
        "sarcasm_labels": label_with_model(texts, SARCASM_MODEL, "sarcasm"),
        "sentiment_labels": label_with_model(texts, SENTIMENT_MODEL, "sentiment"),
    }

    print(f"\n[Caching labels to {cache_path}]")
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result


def _column_to_int_labels(series: pd.Series):
    """Map arbitrary column values to contiguous integer labels."""
    unique_vals = sorted(series.dropna().unique(), key=str)
    val2id = {v: i for i, v in enumerate(unique_vals)}
    return [val2id[v] for v in series], val2id


class MultitaskDataset(Dataset):
    """Dataset returning stance, sarcasm, and sentiment labels."""

    def __init__(self, texts, topics, stance_labels, sarcasm_labels,
                 sentiment_labels, tokenizer, max_length=512):
        self.texts = texts
        self.topics = topics
        self.stance_labels = stance_labels
        self.sarcasm_labels = sarcasm_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        topic = str(self.topics[idx]) if self.topics else None

        args = (topic, text) if topic else (text,)
        encoding = self.tokenizer(
            *args,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "stance_labels": torch.tensor(self.stance_labels[idx], dtype=torch.long),
            "sarcasm_labels": torch.tensor(self.sarcasm_labels[idx], dtype=torch.long),
            "sentiment_labels": torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].flatten()
        return item


class MultitaskInferenceDataset(Dataset):
    """Dataset for inference (no labels)."""

    def __init__(self, texts, topics, tokenizer, max_length=512):
        self.texts = texts
        self.topics = topics
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        topic = str(self.topics[idx]) if self.topics else None

        args = (topic, text) if topic else (text,)
        encoding = self.tokenizer(
            *args,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].flatten()
        return item


class MultitaskOutput:
    """Output container for the multitask model."""

    def __init__(self, loss=None, stance_logits=None, sarcasm_logits=None,
                 sentiment_logits=None, hidden_states=None,
                 cls_embeddings=None):
        self.loss = loss
        self.stance_logits = stance_logits
        self.sarcasm_logits = sarcasm_logits
        self.sentiment_logits = sentiment_logits
        self.hidden_states = hidden_states
        self.cls_embeddings = cls_embeddings
        self.logits = stance_logits  # HuggingFace Trainer compatibility


class MultitaskModel(PreTrainedModel):
    """Shared encoder with per-task classification heads.

    Uses uncertainty weighting to automatically balance task losses.
    Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    (https://arxiv.org/abs/1705.07115)
    """

    def __init__(self, config, num_stance_labels: int,
                 num_sarcasm_labels: int, num_sentiment_labels: int,
                 classifier_dropout: float = None):
        super().__init__(config)
        self.num_stance_labels = num_stance_labels
        self.num_sarcasm_labels = num_sarcasm_labels
        self.num_sentiment_labels = num_sentiment_labels

        from transformers import AutoModel
        self.encoder = AutoModel.from_config(config)

        drop_p = classifier_dropout if classifier_dropout is not None else 0.1
        h = config.hidden_size
        self.stance_classifier = nn.Sequential(
            nn.Dropout(drop_p), nn.Linear(h, h), nn.GELU(),
            nn.Dropout(drop_p), nn.Linear(h, num_stance_labels),
        )
        self.sarcasm_classifier = nn.Sequential(
            nn.Dropout(drop_p), nn.Linear(h, num_sarcasm_labels),
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(drop_p), nn.Linear(h, num_sentiment_labels),
        )

        self.log_var_stance = nn.Parameter(torch.zeros(1))
        self.log_var_sarcasm = nn.Parameter(torch.zeros(1))
        self.log_var_sentiment = nn.Parameter(torch.zeros(1))

        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                stance_labels=None, sarcasm_labels=None,
                sentiment_labels=None, output_hidden_states=False,
                return_cls_embeddings=False, **kwargs):
        enc_kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            enc_kw["token_type_ids"] = token_type_ids

        outputs = self.encoder(**enc_kw, output_hidden_states=output_hidden_states)
        cls_emb = outputs.last_hidden_state[:, 0, :]

        stance_logits = self.stance_classifier(cls_emb)
        sarcasm_logits = self.sarcasm_classifier(cls_emb)
        sentiment_logits = self.sentiment_classifier(cls_emb)

        loss = None
        if stance_labels is not None:
            ce = nn.CrossEntropyLoss()
            loss = (
                torch.exp(-self.log_var_stance) * ce(stance_logits, stance_labels)
                + self.log_var_stance / 2
                + torch.exp(-self.log_var_sarcasm) * ce(sarcasm_logits, sarcasm_labels)
                + self.log_var_sarcasm / 2
                + torch.exp(-self.log_var_sentiment) * ce(sentiment_logits, sentiment_labels)
                + self.log_var_sentiment / 2
            )

        return MultitaskOutput(
            loss=loss,
            stance_logits=stance_logits,
            sarcasm_logits=sarcasm_logits,
            sentiment_logits=sentiment_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            cls_embeddings=cls_emb if return_cls_embeddings else None,
        )


def load_multitask_model(model_name: str, num_stance_labels: int,
                         num_sarcasm_labels: int = 2,
                         num_sentiment_labels: int = 3,
                         classifier_dropout: float = None) -> MultitaskModel:
    """Create a MultitaskModel with pretrained encoder weights."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = MultitaskModel(config, num_stance_labels,
                           num_sarcasm_labels, num_sentiment_labels,
                           classifier_dropout=classifier_dropout)

    from transformers import AutoModel
    pretrained = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.encoder.load_state_dict(pretrained.state_dict())
    del pretrained
    return model


class MultitaskTrainer(Trainer):
    """Trainer for multitask learning with uncertainty-weighted loss."""

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None):
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            outputs = model(**inputs)
            logits = outputs.stance_logits
            labels = inputs.get("stance_labels", None)
        return (None, logits, labels)

    def log(self, logs, start_time=None):
        if self.model is not None and hasattr(self.model, "log_var_stance"):
            logs["weight_stance"] = float(torch.exp(-self.model.log_var_stance).item())
            logs["weight_sarcasm"] = float(torch.exp(-self.model.log_var_sarcasm).item())
            logs["weight_sentiment"] = float(torch.exp(-self.model.log_var_sentiment).item())
        super().log(logs, start_time)


def train_multitask(
    config: TrainConfig,
    trainer_factory,
    use_cache: bool = True,
    arabert_model: str = None,
    train_path: str = None,
    val_path: str = None,
    method_label: str = "Multitask",
    extra_label_config: dict = None,
):
    """Load data, build auxiliary labels if needed, train multitask model, save artifacts."""
    set_seed(config.seed)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    print(f"\n[{method_label} Training]")
    print(f"  Primary: Stance   Auxiliary: Sarcasm, Sentiment")
    print(f"  Loss weighting: Uncertainty (automatic)")
    if arabert_model:
        print(f"  AraBERT preprocessing: {arabert_model}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name,
                                              trust_remote_code=True)

    subtask_cfg = SUBTASK_CONFIG[config.subtask]
    train_csv = train_path or subtask_cfg["train"]
    val_csv = val_path or subtask_cfg["val"]
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print(f"  Train: {train_csv} ({len(train_df)} samples)")
    print(f"  Val:   {val_csv} ({len(val_df)} samples)")

    text_col = subtask_cfg["text_col"]
    topic_col = subtask_cfg["topic_col"]
    label_col = subtask_cfg["label_col"]
    val_label_col = subtask_cfg.get("val_label_col", label_col)
    val_has_stance = (val_label_col in val_df.columns
                      and val_df[val_label_col].notna().all())

    if config.normalize_arabic:
        for df in (train_df, val_df):
            df[text_col] = df[text_col].apply(normalize_arabic)
            if topic_col:
                df[topic_col] = df[topic_col].apply(normalize_arabic)

    unique_labels = sorted(train_df[label_col].dropna().unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {v: k for k, v in label2id.items()}
    num_stance = len(label2id)
    print(f"  Stance labels: {label2id}")

    train_has_aux = {"sarcasm", "sentiment"} <= set(train_df.columns)
    val_has_aux = {"sarcasm", "sentiment"} <= set(val_df.columns)

    if train_has_aux:
        print("\n[Using enriched CSV columns for auxiliary labels]")
        train_sarcasm_all, sarcasm2id = _column_to_int_labels(train_df["sarcasm"])
        train_sentiment_all, sentiment2id = _column_to_int_labels(train_df["sentiment"])
        aux = {"sarcasm_labels": train_sarcasm_all,
               "sentiment_labels": train_sentiment_all}
        print(f"  Sarcasm: {sarcasm2id}  Sentiment: {sentiment2id}")
    else:
        print("\n[Labeling auxiliary tasks with pretrained models]")
        aux = label_dataset(train_df, text_col, use_cache,
                            config.subtask, train_path)

    num_sarcasm = len(set(aux["sarcasm_labels"]))
    num_sentiment = len(set(aux["sentiment_labels"]))
    print(f"  Sarcasm labels: {num_sarcasm}  Sentiment labels: {num_sentiment}")

    eval_dataset = None
    train_sarcasm = aux["sarcasm_labels"]
    train_sentiment = aux["sentiment_labels"]

    train_dataset = MultitaskDataset(
        train_df[text_col].tolist(),
        train_df[topic_col].tolist() if topic_col else None,
        [label2id[l] for l in train_df[label_col]],
        train_sarcasm, train_sentiment, tokenizer, config.max_length,
    )

    if val_has_stance and val_has_aux:
        val_sarcasm, _ = _column_to_int_labels(val_df["sarcasm"])
        val_sentiment, _ = _column_to_int_labels(val_df["sentiment"])
        val_dataset = MultitaskDataset(
            val_df[text_col].tolist(),
            val_df[topic_col].tolist() if topic_col else None,
            [label2id[l] for l in val_df[val_label_col]],
            val_sarcasm, val_sentiment, tokenizer, config.max_length,
        )
        eval_dataset = val_dataset
        print("Val has stance + aux labels -> using as eval set")
    else:
        val_dataset = MultitaskInferenceDataset(
            val_df[text_col].tolist(),
            val_df[topic_col].tolist() if topic_col else None,
            tokenizer, config.max_length,
        )

    print(f"\nTrain: {len(train_dataset)}  Val: {len(val_dataset)}"
          + (f"  Eval: {len(eval_dataset)} (labeled)" if eval_dataset else ""))

    model = load_multitask_model(config.model_name, num_stance,
                                 num_sarcasm, num_sentiment,
                                 classifier_dropout=config.classifier_dropout)
    training_args = make_training_args(config, eval_dataset,
                                       remove_unused_columns=False)
    trainer = trainer_factory(model, training_args, train_dataset, eval_dataset)

    if config.classifier_dropout is not None:
        print(f"  Classifier dropout: {config.classifier_dropout}")
    print(f"\n{'=' * 50}\nSTARTING {method_label.upper()} TRAINING\n{'=' * 50}")
    trainer.train()

    print("\n[Uncertainty Weights]")
    print(f"  Stance:    {torch.exp(-model.log_var_stance).item():.4f}")
    print(f"  Sarcasm:   {torch.exp(-model.log_var_sarcasm).item():.4f}")
    print(f"  Sentiment: {torch.exp(-model.log_var_sentiment).item():.4f}")

    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    label_cfg = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "num_stance_labels": num_stance,
        "num_sarcasm_labels": num_sarcasm,
        "num_sentiment_labels": num_sentiment,
    }
    if extra_label_config:
        label_cfg.update(extra_label_config)
    with open(os.path.join(config.output_dir, "label_config.json"), "w") as f:
        json.dump(label_cfg, f, indent=2)

    return trainer, {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "eval_dataset": eval_dataset,
        "val_df": val_df,
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": num_stance,
    }


def predict_multitask(trainer, data, config: TrainConfig,
                      dataset_key: str = "val"):
    """Generate stance predictions on a dataset."""
    dataset_name = "test" if dataset_key == "test" else "validation"
    print(f"\n{'=' * 50}\nPREDICTING ON {dataset_name.upper()} SET\n{'=' * 50}")

    predictions = trainer.predict(data[f"{dataset_key}_dataset"])
    pred_ids = np.argmax(predictions.predictions, axis=1)
    pred_labels = [data["id2label"][p] for p in pred_ids]

    df = data[f"{dataset_key}_df"].copy()
    df["prediction"] = pred_labels

    print(f"\nPrediction distribution:")
    print(pd.Series(pred_labels).value_counts())

    predictions_csv = config.predictions_csv
    predictions_zip = config.predictions_zip
    if dataset_key != "val":
        predictions_csv = predictions_csv.replace(".csv", f"_{dataset_key}.csv")
        predictions_zip = predictions_zip.replace(".zip", f"_{dataset_key}.zip")

    df.to_csv(predictions_csv, index=False)
    print(f"Saved predictions to {predictions_csv}")

    with zipfile.ZipFile(predictions_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(predictions_csv, os.path.basename(predictions_csv))
    print(f"Created ZIP: {predictions_zip}")

    return df
