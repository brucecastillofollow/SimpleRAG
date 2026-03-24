# Simple Local RAG Chat (Python)

This project is a minimal Retrieval-Augmented Generation (RAG) chat system where:

- Vector DB is local (`ChromaDB` persisted to disk)
- Embeddings are local (`Ollama` embedding model)
- Chat model is local (`Ollama` LLM model)

## 1) Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running on your machine

Pull local models (examples):

```powershell
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

## 2) Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Add your documents

Create a `data` folder and place your files there (`.txt`, `.md`, `.py`, `.json`, `.yaml`, `.yml`, `.csv`, `.log`).

```powershell
mkdir data
```

## 4) Run ingestion + chat

```powershell
python rag_local.py
```

This will:
- read files in `data`
- chunk and embed them locally
- store vectors in `rag_db`
- start an interactive chat

Type `exit` to quit.

## Useful options

Ingest only:

```powershell
python rag_local.py --ingest-only
```

Custom model names:

```powershell
python rag_local.py --embed-model nomic-embed-text --llm-model llama3.1:8b
```

Custom data/db paths:

```powershell
python rag_local.py --data-dir my_docs --db-dir my_rag_db
```

## Scrape public websites into `data/`

You can create local docs from public URLs for your RAG system:

```powershell
python scrape_to_data.py --url https://example.com
```

Multiple URLs:

```powershell
python scrape_to_data.py --url https://example.com --url https://example.org/docs
```

From a file (one URL per line):

```powershell
python scrape_to_data.py --url-file urls.txt
```

Custom output folder:

```powershell
python scrape_to_data.py --url-file urls.txt --output-dir data
```

After scraping, run ingestion/chat again:

```powershell
python rag_local.py --ingest-only
python rag_local.py
```
