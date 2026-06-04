"""End-to-end RAG pipeline: ingest -> retrieve -> answer."""
from __future__ import annotations
from pathlib import Path
from typing import List
import os

from .chunking import load_text, chunk_text
from .store import VectorStore, Chunk


class RAGPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.store = VectorStore(model_name)
        self._built = False

    def ingest_dir(self, directory: str) -> int:
        added = 0
        for path in Path(directory).rglob("*"):
            if path.suffix.lower() in {".txt", ".md", ".pdf"}:
                for chunk in chunk_text(load_text(path)):
                    self.store.add([Chunk(text=chunk, source=path.name)])
                    added += 1
        self.store.build()
        self._built = True
        return added

    def answer(self, question: str, k: int = 4) -> dict:
        if not self._built:
            raise RuntimeError("Ingest documents before asking questions.")
        hits = self.store.search(question, k=k)
        context = "\n\n".join(f"[{c.source}] {c.text}" for c, _ in hits)
        answer = self._generate(question, context)
        return {
            "answer": answer,
            "sources": list({c.source for c, _ in hits}),
            "scores": [round(s, 3) for _, s in hits],
        }

    def _generate(self, question: str, context: str) -> str:
        """Use OpenAI if OPENAI_API_KEY is set, else return an extractive answer."""
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                client = OpenAI()
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "Answer ONLY from the provided context. If unknown, say so."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                    ],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:  # pragma: no cover
                return f"[LLM error, falling back to extractive] {e}\n\n{context[:600]}"
        # Extractive fallback — fully offline, no API key needed
        return ("(Extractive answer — set OPENAI_API_KEY for generative answers)\n\n"
                + context[:800])
