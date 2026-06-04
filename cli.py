"""Command-line interface for the RAG assistant.

    python cli.py --docs data/docs --ask "What is PatchCore?"
"""
import argparse
from rag.pipeline import RAGPipeline


def main():
    p = argparse.ArgumentParser(description="LLM RAG Assistant")
    p.add_argument("--docs", default="data/docs", help="Folder of .txt/.md/.pdf docs")
    p.add_argument("--ask", required=True, help="Your question")
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args()

    rag = RAGPipeline()
    n = rag.ingest_dir(args.docs)
    print(f"Indexed {n} chunks from {args.docs}\n")
    result = rag.answer(args.ask, k=args.k)
    print("ANSWER:\n" + result["answer"])
    print("\nSOURCES:", ", ".join(result["sources"]))
    print("SCORES:", result["scores"])


if __name__ == "__main__":
    main()
