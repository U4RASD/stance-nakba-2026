"""RankMe effective rank for two checkpoints on a subtask split.

Based on the reptrix library: https://github.com/BARL-SSL/reptrix
"""

import os
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA  # type: ignore
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)

from ..data_loader import StanceDataset, normalize_arabic, SUBTASK_CONFIG


def get_eigenspectrum(
    activations_np: np.ndarray, max_eigenvals: int = 2048
) -> np.ndarray:
    """Get eigenspectrum of activation covariance matrix.

    Args:
        activations_np (np.ndarray): Numpy arr of activations,
                                    shape (bsz,d1,d2...dn)
        max_eigenvals (int, optional): Maximum #eigenvalues to compute.
                                        Defaults to 2048.

    Returns:
        np.ndarray: Returns the eigenspectrum of the activation covariance matrix
    """
    feats = activations_np.reshape(activations_np.shape[0], -1)
    feats_center = feats - feats.mean(axis=0)
    pca = PCA(
        n_components=min(max_eigenvals, feats_center.shape[0], feats_center.shape[1]),
        svd_solver="full",
    )
    pca.fit(feats_center)
    eigenspectrum = pca.explained_variance_ratio_
    return eigenspectrum


def get_rank(eigen: np.ndarray) -> float:
    """Get effective rank of the representation covariance matrix

    Args:
        eigen (np.ndarray): Eigenspectrum of the representation covariance matrix

    Returns:
        float: Effective rank
    """
    l1 = np.sum(np.abs(eigen))
    eps = 1e-7
    eigen_norm = eigen / l1 + eps
    entropy = -np.sum(eigen_norm * np.log(eigen_norm))
    return float(np.exp(entropy))


def get_rankme(activations: torch.Tensor, max_eigenvals: int = 2048) -> float:
    """Get RankMe metric
    (https://proceedings.mlr.press/v202/garrido23a)

    Args:
        activations (np.ndarray): Activation tensor of shape (bsz,d1,d2...dn)
        max_eigenvals (int, optional): Maximum #eigenvalues to compute.
                                    Defaults to 2048.

    Returns:
        float: RankMe metric
    """
    if activations.requires_grad:
        activations_arr = activations.detach().cpu()
    else:
        activations_arr = activations.cpu()
    eigen = get_eigenspectrum(activations_arr.numpy(), max_eigenvals=max_eigenvals)
    return get_rank(eigen)


def load_model(model_dir: str):
    """Load a saved model directory and return (model, tokenizer)."""
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    label_config_path = model_dir / "label_config.json"
    if label_config_path.exists():
        with open(label_config_path) as f:
            label_cfg = json.load(f)
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        from ..training.multitask import MultitaskModel

        model = MultitaskModel(
            config,
            label_cfg["num_stance_labels"],
            label_cfg.get("num_sarcasm_labels", 2),
            label_cfg.get("num_sentiment_labels", 3),
        )
        safetensors_path = model_dir / "model.safetensors"
        bin_path = model_dir / "pytorch_model.bin"
        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
        elif bin_path.exists():
            state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No model weights found in {model_dir}")
        model.load_state_dict(state_dict)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), trust_remote_code=True,
        )

    return model, tokenizer


@torch.no_grad()
def extract_features(model, dataloader, device: str) -> torch.Tensor:
    """Run dataloader through model and return CLS-token hidden states."""
    model.eval()
    model.to(device)
    all_features: list[torch.Tensor] = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = model(**inputs, output_hidden_states=True)
        cls = outputs.hidden_states[-1][:, 0, :]
        all_features.append(cls.cpu())

    return torch.cat(all_features, dim=0)


def load_split_texts(subtask: str, split: str):
    """Load and normalise texts (and topics) for a given subtask/split."""
    cfg = SUBTASK_CONFIG[subtask]
    df = pd.read_csv(cfg[split])

    text_col = cfg["text_col"]
    topic_col = cfg["topic_col"]

    df[text_col] = df[text_col].apply(normalize_arabic)
    if topic_col:
        df[topic_col] = df[topic_col].apply(normalize_arabic)

    texts = df[text_col].tolist()
    topics = df[topic_col].tolist() if topic_col else None
    return texts, topics


def main():
    parser = ArgumentParser(description="Compare RankMe effective rank of two models")
    parser.add_argument("-st", "--subtask", type=str, default="B", choices=["A", "B"])
    parser.add_argument("-s", "--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("-m1", "--model1", type=str, required=True,
                        help="Path to first saved model directory")
    parser.add_argument("-m2", "--model2", type=str, required=True,
                        help="Path to second saved model directory")
    parser.add_argument("-me", "--max-eigenvals", type=int, default=2048)
    parser.add_argument("-bs", "--batch-size", type=int, default=32)
    parser.add_argument("-ml", "--max-length", type=int, default=512)
    parser.add_argument("-o", "--output-dir", type=str, default="outputs/effective_rank")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading Subtask {args.subtask} ({args.split} split) ...")
    texts, topics = load_split_texts(args.subtask, args.split)
    print(f"  Samples: {len(texts)}")

    print(f"\nLoading model 1: {args.model1}")
    model1, tokenizer1 = load_model(args.model1)
    print(f"Loading model 2: {args.model2}")
    model2, tokenizer2 = load_model(args.model2)

    ds1 = StanceDataset(texts, topics, labels=None,
                        tokenizer=tokenizer1, max_length=args.max_length)
    loader1 = DataLoader(ds1, batch_size=args.batch_size, shuffle=False)

    print("\nExtracting features from model 1 ...")
    features1 = extract_features(model1, loader1, device)
    print(f"  Shape: {tuple(features1.shape)}")

    del model1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ds2 = StanceDataset(texts, topics, labels=None,
                        tokenizer=tokenizer2, max_length=args.max_length)
    loader2 = DataLoader(ds2, batch_size=args.batch_size, shuffle=False)

    print("\nExtracting features from model 2 ...")
    features2 = extract_features(model2, loader2, device)
    print(f"  Shape: {tuple(features2.shape)}")

    del model2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nComputing RankMe for model 1 (max eigenvalues={args.max_eigenvals}) ...")
    rank1 = get_rankme(features1, args.max_eigenvals)

    print(f"\nComputing RankMe for model 2 (max eigenvalues={args.max_eigenvals}) ...")
    rank2 = get_rankme(features2, args.max_eigenvals)

    with open(os.path.join(args.output_dir, "rankme.json"), "w") as f:
        json.dump({"model1": rank1, "model2": rank2}, f, indent=4)

    lines = [
        "=" * 55,
        "  EFFECTIVE RANK COMPARISON",
        "=" * 55,
        f"  Subtask {args.subtask}  |  Split: {args.split}  |  Samples: {len(texts)}",
        "-" * 55,
        f"  Model 1 : {args.model1}",
        f"  RankMe  : {rank1:.4f}",
        "-" * 55,
        f"  Model 2 : {args.model2}",
        f"  RankMe  : {rank2:.4f}",
        "-" * 55,
    ]
    if rank1 > rank2:
        lines.append(f"  >>> Model 1 has higher effective rank "
                     f"({rank1:.4f} vs {rank2:.4f})")
    elif rank2 > rank1:
        lines.append(f"  >>> Model 2 has higher effective rank "
                     f"({rank2:.4f} vs {rank1:.4f})")
    else:
        lines.append(f"  >>> Both models have equal effective rank")
    lines.append("=" * 55)

    report = "\n".join(lines)
    print(f"\n{report}")

    report_path = os.path.join(args.output_dir, "rankme_report.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
