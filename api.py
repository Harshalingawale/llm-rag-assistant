"""FastAPI service exposing the RAG pipeline.

Run:  uvicorn api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from pydantic import BaseModel

from rag.pipeline import RAGPipeline

app = FastAPI(title="LLM RAG Assistant", version="0.1.0")
pipeline = RAGPipeline()


class IngestRequest(BaseModel):
    directory: str = "data/docs"


class AskRequest(BaseModel):
    question: str
    k: int = 4


@app.post("/ingest")
def ingest(req: IngestRequest):
    n = pipeline.ingest_dir(req.directory)
    return {"chunks_indexed": n, "directory": req.directory}


@app.post("/ask")
def ask(req: AskRequest):
    return pipeline.answer(req.question, k=req.k)


@app.get("/health")
def health():
    return {"status": "ok"}
