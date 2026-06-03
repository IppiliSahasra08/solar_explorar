# Solar Explorar

Solar Explorar is an AI-powered solar knowledge assistant built around a retrieval-augmented generation (RAG) pipeline. It ingests solar and photovoltaic reference material, stores it in a local ChromaDB vector database, retrieves the most relevant chunks, and generates grounded answers with Gemini.

## What is in this repository

- `ingest.py` — builds the local vector index from the chunked document corpus in `data/chunks/chunks.json`
- `search.py` — runs a fast retrieval test against the vector store
- `rag_app.py` — demonstrates the end-to-end RAG flow with Gemini
- `evaluate.py` — evaluates answer quality with RAGAS and an Ollama-backed evaluator
- `src/` — core retrieval, embedding, and LLM generation logic
- `model_implementation/` — the legacy roof-segmentation / solar analytics prototype

## Current workflow

1. Prepare a corpus of solar documents and chunk them into `data/chunks/chunks.json`.
2. Ingest the chunks into the local ChromaDB collection named `solar_knowledge`.
3. Retrieve the best matching context using embeddings + reranking.
4. Generate a grounded answer with Gemini.
5. Optionally evaluate the RAG quality with the scripts in `evaluate.py`.

## Quick start

### 1. Prerequisites

- Python 3.10+
- A working Ollama installation if you want to run the evaluation script
- A Gemini API key exported as `GEMINI_API_KEY`

### 2. Create and activate an environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the packages required for retrieval and generation:

```bash
pip install google-genai sentence-transformers chromadb transformers torch ragas datasets openai
```

If you want the legacy segmentation prototype in `model_implementation/`, install its separate requirements from that folder as well:

```bash
pip install -r model_implementation/requirements.txt
```

### 4. Set your API key

```powershell
$env:GEMINI_API_KEY="your_gemini_api_key"
```

### 5. Build the vector index

```bash
python ingest.py
```

### 6. Test retrieval

```bash
python search.py
```

### 7. Run the RAG answer example

```bash
python rag_app.py
```

### 8. Run evaluation

```bash
python evaluate.py
```

## Project structure

```text
.
├── ingest.py              # Ingest chunks into ChromaDB
├── search.py              # Retrieval smoke test
├── rag_app.py             # End-to-end RAG example
├── evaluate.py            # RAGAS evaluation workflow
├── data/                  # ChromaDB, chunks, evaluation datasets, extracted documents
├── src/                   # Embedding, retrieval, and LLM logic
└── model_implementation/  # Legacy solar analytics / segmentation prototype
```

## Notes

- The current main project is the retrieval and question-answering pipeline, not the older image-segmentation demo.
- The vector store uses a local persistent ChromaDB database under `data/chroma_db/`.
- The evaluation script expects Ollama-compatible endpoints for the evaluator model and embeddings.

## Team

- Ippili Sahasra
- Sri Poojitha Sudalagunta
- Shreya Kailash
- Ananya Arumbakkam

