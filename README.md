# Solar Explorar

Solar Explorar is an AI-powered solar knowledge assistant built around a retrieval-augmented generation (RAG) pipeline. It ingests solar and photovoltaic reference material, stores it in a local ChromaDB vector database, retrieves the most relevant chunks, and generates grounded answers with Gemini.

## What this project does

This repository currently focuses on:

- ingesting solar-domain text into a local vector database
- retrieving the most relevant chunks for a user question
- generating grounded answers with Gemini
- evaluating the RAG pipeline with RAGAS/Ollama

## Main files

- `ingest.py` — builds the local ChromaDB index from `data/chunks/chunks.json`
- `search.py` — tests retrieval quality against the indexed chunks
- `rag_app.py` — runs the end-to-end RAG example with Gemini
- `evaluate.py` — evaluates answer quality with RAGAS
- `src/` — core embedding, retrieval, and LLM logic
- `model_implementation/` — older roof-segmentation / solar analytics prototype

## Quick start

### 1. Prerequisites

- Python 3.10+
- A working Ollama setup if you want to run evaluation
- A Gemini API key available as `GEMINI_API_KEY`

### 2. Create a virtual environment

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

```bash
pip install google-genai sentence-transformers chromadb transformers torch ragas datasets openai
```

If you also want the legacy prototype in `model_implementation/`, install its requirements:

```bash
pip install -r model_implementation/requirements.txt
```

### 4. Set your Gemini API key

Windows (PowerShell):

```powershell
$env:GEMINI_API_KEY="your_gemini_key_here"
```

### 5. Build the vector index

```bash
python ingest.py
```

### 6. Test retrieval

```bash
python search.py
```

### 7. Run the RAG example

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
├── ingest.py
├── search.py
├── rag_app.py
├── evaluate.py
├── data/
│   ├── chunks/
│   ├── chroma_db/
│   └── evaluation/
├── src/
└── model_implementation/
```

## Notes

- The main active project is the RAG and question-answering workflow.
- The vector store uses local persistent ChromaDB storage in `data/chroma_db/`.
- The evaluation script expects an Ollama-compatible evaluator setup.

## Team

- Ippili Sahasra
- Sri Poojitha Sudalagunta
- Shreya Kailash
- Ananya Arumbakkam
