"""Clustering-based stance classification using embedding centroids."""

from .classifier import StanceClassifier
from .cluster import ClusterManager, DistanceMetric
from .embedder import HuggingFaceEmbedder
