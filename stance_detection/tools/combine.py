"""Merge subtask CSVs with LLM sentiment/sarcasm columns from llm_results.json."""

import json
import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from ..data_loader import SUBTASK_CONFIG


def load_llm_labels(results_path: str, column_name: str) -> Dict[int, str]:
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    labels = {}
    for r in data["results"]:
        labels[r["id"]] = r["predicted_label"]

    print(f"  Loaded {len(labels)} {column_name} labels from {results_path}")
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Add LLM sentiment/sarcasm columns to a subtask CSV"
    )
    parser.add_argument(
        "-st", "--subtask", type=str, default="B", choices=["A", "B"],
        help="Subtask (default: B)",
    )
    parser.add_argument(
        "-s", "--split", type=str, default="train",
        choices=["train", "val", "test"],
        help="Data split to combine (default: train)",
    )
    parser.add_argument(
        "--sentiment", type=str, default=None,
        help="Path to sentiment llm_results.json",
    )
    parser.add_argument(
        "--sarcasm", type=str, default=None,
        help="Path to sarcasm llm_results.json",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output CSV path",
    )

    args = parser.parse_args()

    if not args.sentiment and not args.sarcasm:
        parser.error("Provide at least one of --sentiment or --sarcasm")

    config = SUBTASK_CONFIG[args.subtask]
    data_path = config[args.split]

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    for col_name, results_path in [("sentiment", args.sentiment),
                                    ("sarcasm", args.sarcasm)]:
        if results_path is None:
            continue

        labels = load_llm_labels(results_path, col_name)

        df[col_name] = df["id"].map(labels)

        matched = df[col_name].notna().sum()
        missing = df[col_name].isna().sum()
        if missing > 0:
            print(f"  WARNING: {missing} rows have no {col_name} prediction (matched {matched}/{len(df)})")
        else:
            print(f"  All {matched} rows matched for {col_name}")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(data_path).parent / f"{args.split}_enriched.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved enriched dataset to {out_path}  ({len(df)} rows, columns: {list(df.columns)})")


if __name__ == "__main__":
    main()
