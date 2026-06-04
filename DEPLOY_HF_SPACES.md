# Deploy the live RAG demo to Hugging Face Spaces (free)

This repo already contains `app.py` (a Gradio UI) — Spaces auto-runs it.

## One-time setup
1. Create a free account at https://huggingface.co and a write token at
   https://huggingface.co/settings/tokens
2. Create a new Space: https://huggingface.co/new-space
   - SDK: **Gradio**
   - Hardware: **CPU basic (free)**
3. Push this repo to the Space (replace USER):

```bash
git remote add space https://huggingface.co/spaces/USER/llm-rag-assistant
git push space main
# when prompted for a password, paste your Hugging Face write token
```

Add this YAML to the **top** of the Space's README (Spaces requires it):

```yaml
---
title: LLM RAG Assistant
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---
```

Your live demo will be at: `https://huggingface.co/spaces/USER/llm-rag-assistant`
