from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import chromadb
import ollama
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log"}
_ST_MODEL = None
_ST_MODEL_NAME = None


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


def build_ollama_options(num_gpu: int | None, num_thread: int | None) -> dict:
    options: dict = {}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    if num_thread is not None:
        options["num_thread"] = num_thread
    return options


def get_embedding(
    text: str,
    embed_model: str,
    ollama_options: dict,
    embedding_backend: str,
    local_embed_model: str,
    retries: int = 3,
    backoff_seconds: float = 1.5,
) -> list[float]:
    if embedding_backend == "sentence-transformers":
        global _ST_MODEL, _ST_MODEL_NAME
        if _ST_MODEL is None or _ST_MODEL_NAME != local_embed_model:
            from sentence_transformers import SentenceTransformer

            print(f"[info] Loading local embedding model: {local_embed_model}")
            _ST_MODEL = SentenceTransformer(local_embed_model)
            _ST_MODEL_NAME = local_embed_model
        emb = _ST_MODEL.encode(text, normalize_embeddings=True)
        return emb.tolist()

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            res = ollama.embeddings(model=embed_model, prompt=text, options=ollama_options or None)
            return res["embedding"]
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            wait_s = backoff_seconds * attempt
            print(f"[warn] Embedding request failed (attempt {attempt}/{retries}): {exc}")
            print(f"[info] Retrying in {wait_s:.1f}s ...")
            time.sleep(wait_s)

    raise RuntimeError(
        "Embedding failed after retries. "
        "Ollama may have restarted/crashed (often VRAM pressure). "
        "Try lowering GPU usage with --num-gpu 0 or 1 and rerun."
    ) from last_exc


def ingest(
    data_dir: Path,
    db_dir: Path,
    collection_name: str,
    embed_model: str,
    ollama_options: dict,
    embedding_backend: str,
    local_embed_model: str,
) -> None:
    docs = read_documents(data_dir)
    if not docs:
        print(f"[info] No supported files found under: {data_dir}")
        return

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    # Simple refresh approach for a small starter project.
    # Newer Chroma versions reject delete(where={}), so recreate the collection.
    if collection.count() > 0:
        client.delete_collection(name=collection_name)
        collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []

    for file_path, content in docs:
        for idx, chunk in enumerate(chunk_text(content)):
            ids.append(f"{file_path}::{idx}")
            texts.append(chunk)
            metas.append({"source": file_path, "chunk_index": idx})

    print(f"[info] Embedding {len(texts)} chunks from {len(docs)} files ...")
    embeddings = [
        get_embedding(text, embed_model, ollama_options, embedding_backend, local_embed_model)
        for text in tqdm(texts)
    ]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=embeddings,
    )
    print(f"[ok] Ingestion complete. Stored in: {db_dir}")


def collection_has_data(db_dir: Path, collection_name: str) -> bool:
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    return collection.count() > 0


def retrieve(
    query: str,
    db_dir: Path,
    collection_name: str,
    embed_model: str,
    ollama_options: dict,
    embedding_backend: str,
    local_embed_model: str,
    top_k: int = 4,
) -> list[dict]:
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        return []

    query_embedding = get_embedding(
        query, embed_model, ollama_options, embedding_backend, local_embed_model
    )
    res = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    out: list[dict] = []
    for doc, meta, dist in zip(docs, metas, distances):
        out.append({"document": doc, "metadata": meta, "distance": dist})
    return out


def build_prompt(query: str, contexts: list[dict], max_context_chars: int) -> str:
    if not contexts:
        context_text = "No matching local context was found."
    else:
        context_lines = []
        for i, item in enumerate(contexts, start=1):
            source = item["metadata"].get("source", "unknown")
            snippet = item["document"][:max_context_chars]
            context_lines.append(f"[{i}] Source: {source}\n{snippet}")
        context_text = "\n\n".join(context_lines)

    return f"""You are a helpful assistant. Answer only from the provided context when possible.
If context is insufficient, say what is missing.

Context:
{context_text}

User question:
{query}
"""


def chat(
    db_dir: Path,
    collection_name: str,
    embed_model: str,
    llm_model: str,
    ollama_options: dict,
    embedding_backend: str,
    local_embed_model: str,
    top_k: int,
    max_context_chars: int,
    max_tokens: int,
    stream: bool,
    keep_alive: str,
) -> None:
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

        contexts = retrieve(
            query,
            db_dir,
            collection_name,
            embed_model,
            ollama_options,
            embedding_backend,
            local_embed_model,
            top_k=top_k,
        )
        prompt = build_prompt(query, contexts, max_context_chars=max_context_chars)
        chat_options = {"temperature": 0.1, "num_predict": max_tokens, **ollama_options}
        print("\nAssistant>")
        if stream:
            stream_resp = ollama.chat(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                options=chat_options,
                stream=True,
                keep_alive=keep_alive,
            )
            parts: list[str] = []
            for chunk in stream_resp:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    parts.append(token)
            print()
        else:
            reply = ollama.chat(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                options=chat_options,
                keep_alive=keep_alive,
            )
            answer = reply["message"]["content"]
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
        "--embedding-backend",
        choices=["ollama", "sentence-transformers"],
        default="ollama",
        help="Embedding backend. Use sentence-transformers to avoid Ollama embedding instability.",
    )
    parser.add_argument(
        "--local-embed-model",
        default="all-MiniLM-L6-v2",
        help="Local embedding model name for sentence-transformers backend.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="How many chunks to retrieve per question.")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=800,
        help="Max characters taken from each retrieved chunk for prompt context.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per answer (lower is faster).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Low-latency chat preset: top-k=1, max-context-chars=300, max-tokens=96.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they are generated for faster perceived response.",
    )
    parser.add_argument(
        "--keep-alive",
        default="15m",
        help="How long Ollama keeps the chat model loaded (e.g., 0, 5m, 1h).",
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=None,
        help="Ollama num_gpu option. Set >0 to force GPU layers, 0 for CPU-only.",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=None,
        help="Ollama num_thread option for CPU threads.",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only ingest docs and exit.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-ingestion by rebuilding embeddings from data directory.",
    )
    return parser.parse_args()


def ensure_ollama_models(embed_model: str, llm_model: str, embedding_backend: str) -> None:
    # Fast check: if models are missing, Ollama will error with useful text.
    try:
        _ = ollama.list()
    except Exception as exc:
        print("[error] Ollama is not reachable. Start Ollama locally first.")
        print(f"Details: {exc}")
        sys.exit(1)

    models_to_check = [llm_model]
    if embedding_backend == "ollama":
        models_to_check.append(embed_model)

    for model in models_to_check:
        try:
            ollama.show(model)
        except Exception:
            print(f"[warn] Model '{model}' not found locally.")
            print(f"       Pull it with: ollama pull {model}")


def main() -> None:
    args = parse_args()
    ensure_ollama_models(args.embed_model, args.llm_model, args.embedding_backend)
    ollama_options = build_ollama_options(args.num_gpu, args.num_thread)
    top_k = args.top_k
    max_context_chars = args.max_context_chars
    max_tokens = args.max_tokens

    if args.fast:
        top_k = 1
        max_context_chars = 300
        max_tokens = 96
        print("[info] Fast mode enabled (top-k=1, max-context-chars=300, max-tokens=96).")

    args.db_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    has_index = collection_has_data(args.db_dir, args.collection)
    should_ingest = args.rebuild or args.ingest_only or not has_index
    if should_ingest:
        if args.rebuild:
            print("[info] Rebuild requested. Re-ingesting all documents.")
        elif not has_index:
            print("[info] No existing vector index found. Running first ingestion.")
        ingest(
            args.data_dir,
            args.db_dir,
            args.collection,
            args.embed_model,
            ollama_options,
            args.embedding_backend,
            args.local_embed_model,
        )
    else:
        print("[info] Reusing existing vector index (skip ingestion). Use --rebuild to refresh.")

    if args.ingest_only:
        return
    chat(
        args.db_dir,
        args.collection,
        args.embed_model,
        args.llm_model,
        ollama_options,
        args.embedding_backend,
        args.local_embed_model,
        top_k=top_k,
        max_context_chars=max_context_chars,
        max_tokens=max_tokens,
        stream=args.stream,
        keep_alive=args.keep_alive,
    )


if __name__ == "__main__":
    main()
