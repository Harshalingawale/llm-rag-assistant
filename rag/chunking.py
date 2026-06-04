"""Text loading and chunking utilities."""
from __future__ import annotations
from pathlib import Path
from typing import List
import re


def load_text(path: Path) -> str:
    """Load a .txt or .md file. (PDF support via `pypdf` is optional.)"""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install pypdf to read PDFs: pip install pypdf") from e
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Split text into overlapping word-windows so context isn't cut mid-idea."""
    words = re.split(r"\s+", text.strip())
    if not words or words == [""]:
        return []
    chunks, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += step
    return chunks
