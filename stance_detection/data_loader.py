"""
Data loader for StanceNakba 2026 shared task.
"""

import re
from pathlib import Path
from typing import Optional, Dict, List, Callable, Literal
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from arabert.preprocess import ArabertPreprocessor
    ARABERT_AVAILABLE = True
except ImportError:
    ARABERT_AVAILABLE = False


DATA_DIR = Path(__file__).parent / "data"

SUBTASK_CONFIG = {
    "A": {
        "train": DATA_DIR / "Subtask_A" / "Subtask_A_train.csv",
        "val": DATA_DIR / "Subtask_A" / "Subtask_A_val_labeled.csv",
        "test": DATA_DIR / "Subtask_A" / "Subtask_A_test_noLabel.csv",
        "text_col": "text",
        "topic_col": None,
        "label_col": "label",
        "val_label_col": "prediction",
    },
    "B": {
        "train": DATA_DIR / "Subtask_B" / "Subtask_B_train.csv",
        "val": DATA_DIR / "Subtask_B" / "Subtask_B_eval_labeled.csv",
        "test": DATA_DIR / "Subtask_B" / "Subtask_B_test_noLabel.csv",
        "text_col": "sentence",
        "topic_col": "topic",
        "label_col": "label",
        "val_label_col": "prediction",
    },
}


def normalize_arabic(text: str) -> str:
    """Basic Arabic text normalization."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Normalize alef variants
    text = re.sub(r'[إأآا]', 'ا', text)
    
    # Normalize alef maqsura to yaa
    text = re.sub(r'ى', 'ي', text)
    
    # Normalize taa marbuta to haa
    text = re.sub(r'ة', 'ه', text)
    
    # Remove tatweel
    text = re.sub(r'ـ', '', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs and mentions
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    
    return text.strip()


class StanceDataset(Dataset):
    """PyTorch Dataset for stance detection."""
    
    def __init__(
        self,
        texts: List[str],
        topics: Optional[List[str]],
        labels: Optional[List[int]],
        tokenizer,
        max_length: int = 128,
        tokenize_fn: Optional[Callable] = None,
    ):
        self.texts = texts
        self.topics = topics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenize_fn = tokenize_fn
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        topic = str(self.topics[idx]) if self.topics else None
        
        if self.tokenize_fn:
            encoding = self.tokenize_fn(text, topic, self.tokenizer, self.max_length)
        else:
            encoding = self._default_tokenize(text, topic)
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item
    
    def _default_tokenize(self, text: str, topic: Optional[str]):
        """[CLS] topic [SEP] text [SEP] for BERT-style models."""
        if topic:
            return self.tokenizer(
                topic, text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            return self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )


def load_subtask(
    subtask: Literal["A", "B"],
    tokenizer,
    max_length: int = 128,
    normalize: bool = True,
    arabert_model: Optional[str] = None,
    tokenize_fn: Optional[Callable] = None,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    load_val_labels: bool = False,
    load_test: bool = False,
) -> Dict[str, any]:
    """Load train/val (and optionally test) CSVs, preprocess text, build label maps and StanceDatasets."""
    config = SUBTASK_CONFIG[subtask]

    train_csv = train_path if train_path else config["train"]
    val_csv = val_path if val_path else config["val"]
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(config["test"]) if load_test else None
    
    print(f"[Data Loader] Training data: {train_csv} ({len(train_df)} samples)")
    print(f"[Data Loader] Validation data: {val_csv} ({len(val_df)} samples)")
    if load_test:
        print(f"[Data Loader] Test data: {config['test']} ({len(test_df)} samples)")

    unique_labels = sorted(train_df[config["label_col"]].dropna().unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {v: k for k, v in label2id.items()}

    text_col = config["text_col"]
    topic_col = config["topic_col"]

    sample_idx = 0
    sample_before = str(train_df[text_col].iloc[sample_idx])
    
    if arabert_model:
        if not ARABERT_AVAILABLE:
            raise ImportError(
                "arabert package not installed. Install with: pip install arabert"
            )
        arabert_prep = ArabertPreprocessor(model_name=arabert_model)
        
        def preprocess_text(text):
            if pd.isna(text) or not isinstance(text, str):
                return ""
            return arabert_prep.preprocess(text)
        
        train_df[text_col] = train_df[text_col].apply(preprocess_text)
        val_df[text_col] = val_df[text_col].apply(preprocess_text)
        if test_df is not None:
            test_df[text_col] = test_df[text_col].apply(preprocess_text)
        if topic_col:
            train_df[topic_col] = train_df[topic_col].apply(preprocess_text)
            val_df[topic_col] = val_df[topic_col].apply(preprocess_text)
            if test_df is not None:
                test_df[topic_col] = test_df[topic_col].apply(preprocess_text)
        
        preprocess_method = f"AraBERT ({arabert_model})"
    elif normalize:
        train_df[text_col] = train_df[text_col].apply(normalize_arabic)
        val_df[text_col] = val_df[text_col].apply(normalize_arabic)
        if test_df is not None:
            test_df[text_col] = test_df[text_col].apply(normalize_arabic)
        if topic_col:
            train_df[topic_col] = train_df[topic_col].apply(normalize_arabic)
            val_df[topic_col] = val_df[topic_col].apply(normalize_arabic)
            if test_df is not None:
                test_df[topic_col] = test_df[topic_col].apply(normalize_arabic)
        
        preprocess_method = "Basic Arabic normalization"
    else:
        preprocess_method = "None"

    sample_after = str(train_df[text_col].iloc[sample_idx])
    print(f"\n[Preprocessing Sample - {preprocess_method}]")
    print(f"  BEFORE: {sample_before[:100]}{'...' if len(sample_before) > 100 else ''}")
    print(f"  AFTER:  {sample_after[:100]}{'...' if len(sample_after) > 100 else ''}")

    train_texts = train_df[text_col].tolist()
    val_texts = val_df[text_col].tolist()
    
    train_topics = train_df[topic_col].tolist() if topic_col else None
    val_topics = val_df[topic_col].tolist() if topic_col else None
    
    train_labels = [label2id[l] for l in train_df[config["label_col"]]]

    val_labels = None
    if load_val_labels:
        val_label_col = config.get("val_label_col", config["label_col"])
        if val_label_col in val_df.columns:
            val_df_with_labels = val_df[val_df[val_label_col].notna() & (val_df[val_label_col] != '')]
            if len(val_df_with_labels) > 0:
                val_texts = val_df_with_labels[text_col].tolist()
                val_topics = val_df_with_labels[topic_col].tolist() if topic_col else None
                val_labels = [label2id[l] for l in val_df_with_labels[val_label_col]]
                val_df = val_df_with_labels
                print(f"[Data Loader] Loaded {len(val_labels)} validation labels")

    train_dataset = StanceDataset(
        texts=train_texts,
        topics=train_topics,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        tokenize_fn=tokenize_fn,
    )
    
    val_dataset = StanceDataset(
        texts=val_texts,
        topics=val_topics,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        tokenize_fn=tokenize_fn,
    )

    test_dataset = None
    if load_test and test_df is not None:
        test_texts = test_df[text_col].tolist()
        test_topics = test_df[topic_col].tolist() if topic_col else None
        test_dataset = StanceDataset(
            texts=test_texts,
            topics=test_topics,
            labels=None,
            tokenizer=tokenizer,
            max_length=max_length,
            tokenize_fn=tokenize_fn,
        )
    
    result = {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_df": train_df,
        "val_df": val_df,
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": len(label2id),
    }
    
    if load_test:
        result["test_dataset"] = test_dataset
        result["test_df"] = test_df
    
    return result
