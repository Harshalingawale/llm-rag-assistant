"""Gradio UI for the RAG assistant — entry point for Hugging Face Spaces.

Locally:  python app.py    (or)   gradio app.py
On Spaces: this file is auto-detected (sdk: gradio).
"""
import gradio as gr
from rag.pipeline import RAGPipeline

rag = RAGPipeline()
_n = rag.ingest_dir("data/docs")


def answer(question: str):
    if not question.strip():
        return "Ask a question about the indexed documents."
    res = rag.answer(question, k=4)
    src = ", ".join(res["sources"])
    return f"{res['answer']}\n\n---\nSources: {src}\nScores: {res['scores']}"


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="Your question", placeholder="What is retrieval-augmented generation?"),
    outputs=gr.Textbox(label="Grounded answer", lines=10),
    title="🤖 LLM RAG Assistant",
    description=f"Retrieval-augmented Q&A over {_n} indexed document chunks. "
                "Runs offline (extractive); set OPENAI_API_KEY for generative answers.",
    examples=["What is retrieval-augmented generation?",
              "What does FAISS do?",
              "How do sentence embeddings work?"],
)

if __name__ == "__main__":
    demo.launch()
