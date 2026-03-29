"""Predict stance for a text using a trained clustering model."""

import argparse
import json
import sys
from pathlib import Path

from .classifier import StanceClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Predict stance with a clustering model")
    parser.add_argument("model_path", help="Path to trained model (.pkl)")
    parser.add_argument("topic", help="Topic to classify against")
    parser.add_argument("text", help="Text to classify")
    parser.add_argument("--scores", action="store_true",
                        help="Show confidence scores")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    classifier = StanceClassifier.load(args.model_path)

    if args.topic not in classifier.get_topics():
        print(f"Error: Topic '{args.topic}' not found in model", file=sys.stderr)
        print(f"Available topics: {', '.join(classifier.get_topics())}", file=sys.stderr)
        sys.exit(1)

    if args.scores:
        scores = classifier.predict_with_scores(args.text, args.topic)
        prediction = max(scores, key=scores.get)
        if args.json:
            print(json.dumps({"prediction": prediction, "scores": scores}, indent=2))
        else:
            print(f"Prediction: {prediction}")
            print("Scores:")
            for stance, score in sorted(scores.items(), key=lambda x: -x[1]):
                print(f"  {stance}: {score:.4f}")
    else:
        prediction = classifier.predict(args.text, args.topic)
        if args.json:
            print(json.dumps({"prediction": prediction}))
        else:
            print(prediction)


if __name__ == "__main__":
    main()
