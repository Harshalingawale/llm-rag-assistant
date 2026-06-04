#  LLM RAG Assistant

> A production-style **Retrieval-Augmented Generation** service that answers questions grounded in your own documents — runs fully offline by default, plugs into an LLM when you add a key.

<p align="left">
  <img src="https://github.com/Harshalingawale/llm-rag-assistant/actions/workflows/ci.yml/badge.svg" alt="CI" />
  <a href="https://github.com/Harshalingawale/llm-rag-assistant/pkgs/container/llm-rag-assistant"><img src="https://img.shields.io/badge/ghcr.io-image-2496ED?logo=docker&logoColor=white" alt="GHCR" /></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/sentence--transformers-Embeddings-FFB000" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-009688" />
  <img src="https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

##  Why this exists

LLMs hallucinate when asked about things outside their training data. **RAG** fixes this by retrieving the most relevant chunks from *your* knowledge base and forcing the model to answer from them. This repo implements the full loop — ingestion, chunking, embedding, vector search, and grounded generation — with clean, testable modules.

A nice property: it works **without any API key** (returns an extractive answer from the top chunks), and upgrades to generative answers the moment you set `OPENAI_API_KEY`.

##  How it works

```
docs/ ──▶ chunking.py ──▶ embeddings (all-MiniLM-L6-v2) ──▶ FAISS index
                                                              │
                          question ──▶ embed ──▶ similarity search (top-k)
                                                              │
                                          context + question ──▶ LLM ──▶ grounded answer
```

| Module | Responsibility |
|---|---|
| `rag/chunking.py` | Load `.txt/.md/.pdf`, split into overlapping windows |
| `rag/store.py` | Embed with sentence-transformers, index with FAISS (NumPy fallback) |
| `rag/pipeline.py` | Orchestrate ingest → retrieve → answer (OpenAI optional) |
| `api.py` | FastAPI service (`/ingest`, `/ask`, `/health`) |
| `cli.py` | One-shot command-line Q&A |

##  Quickstart

```bash
git clone https://github.com/harshalingawale/llm-rag-assistant.git
cd llm-rag-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Ask a question over the bundled sample docs (no API key needed)
python cli.py --ask "What is retrieval-augmented generation?"
```

### As an API

```bash
uvicorn api:app --reload
# then:
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d '{"directory":"data/docs"}'
curl -X POST localhost:8000/ask    -H "Content-Type: application/json" -d '{"question":"What does FAISS do?"}'
```

Interactive docs at **http://127.0.0.1:8000/docs**.

### Generative answers (optional)

```bash
export OPENAI_API_KEY=sk-...        # answers become LLM-generated & grounded
export OPENAI_MODEL=gpt-4o-mini     # optional
```

### Docker

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 rag-assistant
```

##  Tests

```bash
pytest -q
```

##  Tech Stack

**Python · FastAPI · sentence-transformers · FAISS · Pydantic · Uvicorn · Docker** (OpenAI optional)

##  Roadmap

- [ ] Streaming responses + chat history
- [ ] Re-ranking with a cross-encoder
- [ ] Pluggable vector DBs (Qdrant, pgvector)
- [ ] Evaluation harness (faithfulness / answer relevance)

##  License

MIT © [Harshal Ingawale](https://github.com/harshalingawale)
