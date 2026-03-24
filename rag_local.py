from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import chromadb
import ollama
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log"}


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> Iterable[str]:
    """Split text into overlapping chunks for retrieval."""
    clean = text.strip()
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    n = len(clean)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(clean[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def read_documents(data_dir: Path) -> list[tuple[str, str]]:
    """Read text-like files recursively from data directory."""
    docs: list[tuple[str, str]] = []
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"[warn] Failed to read {path}: {exc}")
            continue
        if content.strip():
            docs.append((str(path), content))
    return docs


def get_embedding(text: str, embed_model: str) -> list[float]:
    res = ollama.embeddings(model=embed_model, prompt=text)
    return res["embedding"]


def ingest(data_dir: Path, db_dir: Path, collection_name: str, embed_model: str) -> None:
    docs = read_documents(data_dir)
    if not docs:
        print(f"[info] No supported files found under: {data_dir}")
        return

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    # Simple refresh approach for a small starter project.
    if collection.count() > 0:
        collection.delete(where={})

    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []

    for file_path, content in docs:
        for idx, chunk in enumerate(chunk_text(content)):
            ids.append(f"{file_path}::{idx}")
            texts.append(chunk)
            metas.append({"source": file_path, "chunk_index": idx})

    print(f"[info] Embedding {len(texts)} chunks from {len(docs)} files ...")
    embeddings = [get_embedding(text, embed_model) for text in tqdm(texts)]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=embeddings,
    )
    print(f"[ok] Ingestion complete. Stored in: {db_dir}")


def retrieve(
    query: str,
    db_dir: Path,
    collection_name: str,
    embed_model: str,
    top_k: int = 4,
) -> list[dict]:
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        return []

    query_embedding = get_embedding(query, embed_model)
    res = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    out: list[dict] = []
    for doc, meta, dist in zip(docs, metas, distances):
        out.append({"document": doc, "metadata": meta, "distance": dist})
    return out


def build_prompt(query: str, contexts: list[dict]) -> str:
    if not contexts:
        context_text = "No matching local context was found."
    else:
        context_lines = []
        for i, item in enumerate(contexts, start=1):
            source = item["metadata"].get("source", "unknown")
            context_lines.append(f"[{i}] Source: {source}\n{item['document']}")
        context_text = "\n\n".join(context_lines)

    return f"""You are a helpful assistant. Answer only from the provided context when possible.
If context is insufficient, say what is missing.

Context:
{context_text}

User question:
{query}
"""


def chat(db_dir: Path, collection_name: str, embed_model: str, llm_model: str) -> None:
    print("[info] Type 'exit' to quit.")
    while True:
        try:
            query = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[info] Exiting.")
            return

        if query.lower() in {"exit", "quit"}:
            print("[info] Exiting.")
            return
        if not query:
            continue

        contexts = retrieve(query, db_dir, collection_name, embed_model, top_k=4)
        prompt = build_prompt(query, contexts)
        reply = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        answer = reply["message"]["content"]

        print("\nAssistant>")
        print(answer)
        if contexts:
            print("\n[retrieved]")
            for i, c in enumerate(contexts, start=1):
                src = c["metadata"].get("source", "unknown")
                dist = c["distance"]
                print(f"{i}. {src} (distance={dist:.4f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple local RAG chat in Python.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory with source docs.")
    parser.add_argument("--db-dir", type=Path, default=Path("rag_db"), help="Local Chroma DB directory.")
    parser.add_argument("--collection", default="local_rag", help="Chroma collection name.")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="Local embedding model in Ollama.")
    parser.add_argument("--llm-model", default="llama3.1:8b", help="Local chat model in Ollama.")
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only ingest docs and exit.",
    )
    return parser.parse_args()


def ensure_ollama_models(embed_model: str, llm_model: str) -> None:
    # Fast check: if models are missing, Ollama will error with useful text.
    try:
        _ = ollama.list()
    except Exception as exc:
        print("[error] Ollama is not reachable. Start Ollama locally first.")
        print(f"Details: {exc}")
        sys.exit(1)

    for model in (embed_model, llm_model):
        try:
            ollama.show(model)
        except Exception:
            print(f"[warn] Model '{model}' not found locally.")
            print(f"       Pull it with: ollama pull {model}")


def main() -> None:
    args = parse_args()
    ensure_ollama_models(args.embed_model, args.llm_model)

    args.db_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    ingest(args.data_dir, args.db_dir, args.collection, args.embed_model)
    if args.ingest_only:
        return
    chat(args.db_dir, args.collection, args.embed_model, args.llm_model)


if __name__ == "__main__":
    main()
