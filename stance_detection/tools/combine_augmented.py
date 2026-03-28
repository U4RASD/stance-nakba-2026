"""Concatenate original split CSV with one or more augmented.csv files from LLM augmentation."""

import argparse
from pathlib import Path

import pandas as pd

from ..data_loader import SUBTASK_CONFIG

COLS = ["id", "sentence", "topic", "label"]


def combine(
    subtask: str,
    aug_paths: list[str],
    output_path: str | None = None,
    include_originals: bool = True,
    split: str = "train",
) -> pd.DataFrame:
    config = SUBTASK_CONFIG[subtask]
    frames = []

    if include_originals:
        orig_path = config[split]
        orig_df = pd.read_csv(orig_path)[COLS]
        frames.append(orig_df)
        print(f"Original: {len(orig_df)} samples  ({orig_path})")

    for path in aug_paths:
        aug_df = pd.read_csv(path)
        missing = [c for c in COLS if c not in aug_df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        aug_df = aug_df[COLS].dropna(subset=["sentence", "label"])
        frames.append(aug_df)
        print(f"Augmented: {len(aug_df)} samples  ({path})")

    combined = pd.concat(frames, ignore_index=True)

    # Re-assign sequential ids
    combined["id"] = range(1, len(combined) + 1)

    if output_path is None:
        orig_dir = Path(config[split]).parent
        output_path = str(orig_dir / f"{split}_augmented.csv")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"\nCombined: {len(combined)} samples")
    print(f"Label distribution:")
    for label, count in combined["label"].value_counts().items():
        print(f"  {label}: {count} ({count / len(combined) * 100:.1f}%)")
    print(f"Saved to: {output_path}")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Combine original + augmented CSVs into one training file"
    )
    parser.add_argument(
        "aug_files", nargs="+",
        help="Path(s) to augmented.csv file(s)",
    )
    parser.add_argument(
        "-st", "--subtask", type=str, default="B", choices=["A", "B"],
        help="Subtask (default: B)",
    )
    parser.add_argument(
        "-s", "--split", type=str, default="train",
        choices=["train", "val"],
        help="Base split to augment (default: train)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--no-originals", action="store_true",
        help="Skip originals, only augmented samples",
    )

    args = parser.parse_args()

    combine(
        subtask=args.subtask,
        aug_paths=args.aug_files,
        output_path=args.output,
        include_originals=not args.no_originals,
        split=args.split,
    )


if __name__ == "__main__":
    main()
