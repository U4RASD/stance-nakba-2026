"""Nearest-centroid stance classifier using document embeddings."""

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from .cluster import ClusterManager, DistanceMetric
from .data_loader import DataLoader
from .embedder import HuggingFaceEmbedder


class StanceClassifier:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        max_length: int = 8192,
        device: str = "cuda",
        cache_dir: str = None,
        metric: str = "cosine",
    ):
        self.model_name = model_name
        self.metric = DistanceMetric(metric)
        self.embedder = HuggingFaceEmbedder(model_name, max_length, device, cache_dir)
        self.cluster_manager = ClusterManager(self.metric)
        self._is_fitted = False

    def fit(self, data_path: str, verbose: bool = True) -> "StanceClassifier":
        """Train from a folder of topic/stance/*.txt documents."""
        loader = DataLoader(data_path)
        data = loader.load()
        if not data:
            raise ValueError(f"No data found in {data_path}")

        for topic, stances in data.items():
            if verbose:
                print(f"Processing topic: {topic}")
            for stance, documents in stances.items():
                if verbose:
                    print(f"  Embedding {len(documents)} documents for stance: {stance}")
                embeddings = self.embedder.embed(documents)
                self.cluster_manager.add_cluster(topic, stance, embeddings)

        self._is_fitted = True
        if verbose:
            print(f"Fitted classifier with {len(data)} topics")
        return self

    def predict(self, text: str, topic: str) -> str:
        """Predict the stance for a given text and topic."""
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        embedding = self.embedder.embed(text)[0]
        stance, _ = self.cluster_manager.find_nearest_stance(embedding, topic)
        return stance

    def predict_with_scores(self, text: str, topic: str) -> Dict[str, float]:
        """Predict stance with softmax-normalised confidence scores."""
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        embedding = self.embedder.embed(text)[0]
        distances = self.cluster_manager.get_all_distances(embedding, topic)

        # Convert distances to scores (higher is better)
        if self.metric == DistanceMetric.COSINE:
            scores = {s: -d for s, d in distances.items()}
        else:
            max_dist = max(distances.values()) + 1e-8
            scores = {s: 1 - (d / max_dist) for s, d in distances.items()}

        # Softmax normalisation
        total = sum(np.exp(v) for v in scores.values())
        return {s: np.exp(v) / total for s, v in scores.items()}

    def predict_batch(self, texts: List[str], topic: str) -> List[str]:
        """Predict stances for multiple texts."""
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        predictions = []
        for text in texts:
            embedding = self.embedder.embed(text)[0]
            stance, _ = self.cluster_manager.find_nearest_stance(embedding, topic)
            predictions.append(stance)
        return predictions

    def get_topics(self) -> List[str]:
        return self.cluster_manager.get_topics()

    def get_stances(self, topic: str) -> List[str]:
        return self.cluster_manager.get_stances(topic)

    def save(self, path: str):
        """Save the classifier to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "model_name": self.model_name,
            "metric": self.metric.value,
            "clusters": self.cluster_manager.clusters,
            "is_fitted": self._is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path: str) -> "StanceClassifier":
        """Load a classifier from disk."""
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        classifier = cls(
            model_name=save_data["model_name"],
            metric=save_data["metric"],
        )
        classifier.cluster_manager.clusters = save_data["clusters"]
        classifier._is_fitted = save_data["is_fitted"]
        return classifier

    def info(self) -> Dict:
        """Return model metadata and per-topic cluster info."""
        info = {
            "model_name": self.model_name,
            "metric": self.metric.value,
            "is_fitted": self._is_fitted,
            "topics": {},
        }
        for topic in self.get_topics():
            info["topics"][topic] = {}
            for stance in self.get_stances(topic):
                info["topics"][topic][stance] = self.cluster_manager.get_cluster_info(topic, stance)
        return info
