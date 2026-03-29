"""HuggingFace embedding wrapper with mean-pooling."""

from typing import List, Union

import torch
from transformers import AutoTokenizer, AutoModel


class HuggingFaceEmbedder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        max_length: int = 8192,
        device: str = "cuda",
        cache_dir: str = None,
    ):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir

    def _get_text_embedding(self, text: str):
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self.model(**encoded)
        return out.last_hidden_state, encoded["attention_mask"]

    def _get_mean_pooled_embedding(self, hidden, mask):
        mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        masked = hidden * mask
        return masked.sum(1) / mask.sum(1)

    def embed(self, texts: Union[str, List[str]]) -> List[torch.Tensor]:
        """Embed texts, returning one mean-pooled embedding per text."""
        if isinstance(texts, str):
            texts = [texts]
        texts_embeddings = [self._get_text_embedding(text) for text in texts]
        hidden_states = [e[0] for e in texts_embeddings]
        attention_masks = [e[1] for e in texts_embeddings]
        return [
            self._get_mean_pooled_embedding(h, m)
            for h, m in zip(hidden_states, attention_masks)
        ]
