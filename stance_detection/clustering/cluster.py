"""Cluster management and distance metrics for centroid-based classification."""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch


class DistanceMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class ClusterManager:
    def __init__(self, metric: DistanceMetric = DistanceMetric.COSINE):
        self.metric = metric
        # {topic: {stance: {"embeddings": Tensor, "centroid": Tensor}}}
        self.clusters: Dict[str, Dict[str, Dict]] = {}

    def _to_tensor(self, embeddings) -> torch.Tensor:
        if isinstance(embeddings, torch.Tensor):
            return embeddings
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], torch.Tensor):
            return torch.cat(embeddings, dim=0)
        return torch.as_tensor(embeddings)

    def add_cluster(self, topic: str, stance: str, embeddings):
        if topic not in self.clusters:
            self.clusters[topic] = {}
        embeddings = self._to_tensor(embeddings)
        centroid = embeddings.mean(dim=0)
        self.clusters[topic][stance] = {
            "embeddings": embeddings,
            "centroid": centroid,
        }

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        norm_a = a.norm()
        norm_b = b.norm()
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return (a @ b / (norm_a * norm_b)).item()

    def _euclidean_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return (a - b).norm().item()

    def get_distance(self, embedding: torch.Tensor, centroid: torch.Tensor) -> float:
        if self.metric == DistanceMetric.COSINE:
            # Negative similarity so lower is better (consistent with distance semantics)
            return -self._cosine_similarity(embedding, centroid)
        return self._euclidean_distance(embedding, centroid)

    def find_nearest_stance(self, embedding, topic: str) -> Tuple[str, float]:
        if topic not in self.clusters:
            raise ValueError(f"Topic '{topic}' not found in clusters")
        embedding = self._to_tensor(embedding)
        best_stance = None
        best_distance = float("inf")
        for stance, cluster_data in self.clusters[topic].items():
            distance = self.get_distance(embedding, cluster_data["centroid"])
            if distance < best_distance:
                best_distance = distance
                best_stance = stance
        return best_stance, best_distance

    def get_all_distances(self, embedding, topic: str) -> Dict[str, float]:
        if topic not in self.clusters:
            raise ValueError(f"Topic '{topic}' not found in clusters")
        embedding = self._to_tensor(embedding)
        return {
            stance: self.get_distance(embedding, cd["centroid"])
            for stance, cd in self.clusters[topic].items()
        }

    def get_topics(self) -> List[str]:
        return list(self.clusters.keys())

    def get_stances(self, topic: str) -> List[str]:
        if topic not in self.clusters:
            return []
        return list(self.clusters[topic].keys())

    def get_cluster_info(self, topic: str, stance: str) -> Optional[Dict]:
        if topic not in self.clusters or stance not in self.clusters[topic]:
            return None
        cluster = self.clusters[topic][stance]
        return {
            "num_embeddings": len(cluster["embeddings"]),
            "centroid_shape": tuple(cluster["centroid"].shape),
        }
