"""Train a clustering-based stance classifier from a folder of documents."""

import argparse
import sys
from pathlib import Path

from .classifier import StanceClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Train clustering stance classifier from folder structure")
    parser.add_argument("data_path", help="Path to data folder (topic/stance/*.txt)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output path for trained model")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3-Embedding-8B",
                        help="HuggingFace model for embeddings")
    parser.add_argument("-ml", "--max-length", type=int, default=8192,
                        help="Maximum token length")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine",
                        help="Distance metric")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output")
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        print(f"Error: Data path does not exist: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    classifier = StanceClassifier(
        model_name=args.model,
        max_length=args.max_length,
        metric=args.metric,
    )

    verbose = not args.quiet
    if verbose:
        print(f"Training classifier from: {args.data_path}")
        print(f"Model: {args.model}")
        print(f"Metric: {args.metric}")
        print()

    classifier.fit(args.data_path, verbose=verbose)
    classifier.save(args.output)

    if verbose:
        print(f"\nModel saved to: {args.output}")


if __name__ == "__main__":
    main()
