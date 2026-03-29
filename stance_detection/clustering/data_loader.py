"""Folder-based document loader for topic/stance/*.txt structure."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List


class DataLoader:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.topics = []
        self.stances = defaultdict(list)

    def load(self) -> Dict[str, Dict[str, List[str]]]:
        """Load documents from topic/stance folder structure.

        Returns:
            {topic: {stance: [document_texts]}}
        """
        data = {}
        if not self.base_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.base_path}")

        for topic_dir in self.base_path.iterdir():
            if not topic_dir.is_dir():
                continue
            topic_name = topic_dir.name
            data[topic_name] = {}
            self.topics.append(topic_name)

            for stance_dir in topic_dir.iterdir():
                if not stance_dir.is_dir():
                    continue
                stance_name = stance_dir.name
                self.stances[topic_name].append(stance_name)
                documents = []
                for doc_file in stance_dir.glob("*.txt"):
                    text = doc_file.read_text(encoding="utf-8").strip()
                    if text:
                        documents.append(text)
                if documents:
                    data[topic_name][stance_name] = documents

        return data

    def get_topics(self) -> List[str]:
        return self.topics

    def get_stances(self, topic: str) -> List[str]:
        return self.stances[topic]
