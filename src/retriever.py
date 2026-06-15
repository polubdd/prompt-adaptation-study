"""kNN-ретривер демонстраций для retrieval-based in-context learning."""

import numpy as np
from sentence_transformers import SentenceTransformer


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DemoRetriever:
    def __init__(self, demo_pool: list, embed_model: str = EMBED_MODEL):
        self.pool = demo_pool
        self.embedder = SentenceTransformer(embed_model)
        texts = [d["question"] for d in demo_pool]
        self.embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def topk(self, query: str, k: int = 3) -> list:
        q = self.embedder.encode([query], convert_to_numpy=True,
                                 normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        idx = np.argsort(-sims)[:k]
        return [self.pool[i] for i in idx]
