"""Vector store backed by sentence-transformers + FAISS, with a NumPy fallback."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np


@dataclass
class Chunk:
    text: str
    source: str


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._index = None

    # ---- build -----------------------------------------------------------
    def add(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)

    def build(self) -> None:
        texts = [c.text for c in self.chunks]
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._embeddings = np.asarray(emb, dtype="float32")
        try:
            import faiss

            self._index = faiss.IndexFlatIP(self._embeddings.shape[1])
            self._index.add(self._embeddings)
        except ImportError:
            self._index = None  # fall back to brute-force cosine

    # ---- query -----------------------------------------------------------
    def search(self, query: str, k: int = 4) -> List[Tuple[Chunk, float]]:
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        if self._index is not None:
            scores, idx = self._index.search(q, k)
            return [(self.chunks[i], float(s)) for s, i in zip(scores[0], idx[0]) if i != -1]
        sims = (self._embeddings @ q[0])
        top = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in top]
