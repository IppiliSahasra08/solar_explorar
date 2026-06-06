import os
import time
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.retrieval.vector_store import SolarVectorStore
from src.llm.generator import SolarRAGGenerator
from src.telemetry.logger import log_query, QUERY_LOG
import math
from fastapi.responses import JSONResponse

# ── App init FIRST ──
app = FastAPI(title="Solar Explorer RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ──
v_store = SolarVectorStore(collection_name="solar_knowledge")
generator = SolarRAGGenerator()

class AskRequest(BaseModel):
    question: str

# ── Routes ──
@app.get("/")
def root():
    return {"status": "Solar Explorer RAG API is running"}

@app.post("/api/ask")
def ask(req: AskRequest):
    start = time.time()
    chunks = v_store.retrieve_and_rerank(req.question, top_k_vector=20, final_top_n=5)
    answer = generator.generate_answer(req.question, chunks)
    latency_ms = (time.time() - start) * 1000
    log_query(req.question, answer, latency_ms, len(chunks))
    return {
        "answer": answer,
        "sources": [{"source": c["source"], "page": c["page"]} for c in chunks],
        "latency_ms": round(latency_ms, 2),
        "chunks_retrieved": len(chunks)
    }
import math
from fastapi.responses import JSONResponse

# 1. Serve it from FastAPI (add to src/api/main.py)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/dashboard")
def dashboard():
    return FileResponse("frontend/index.html")

@app.get("/api/telemetry")
def get_telemetry():
    if not QUERY_LOG.exists():
        return JSONResponse({"total_queries": 0, "avg_latency_ms": 0, "avg_chunks_retrieved": 0, "recent_queries": []})
    
    df = pd.read_csv(QUERY_LOG)
    
    def clean(val):
        """Convert NaN/inf to None for JSON safety."""
        if val is None:
            return None
        try:
            if math.isnan(val) or math.isinf(val):
                return None
        except (TypeError, ValueError):
            pass
        return val
    
    # Clean each row manually
    rows = []
    for _, row in df.tail(5).iterrows():
        rows.append({
            "timestamp": str(row.get("timestamp", "")),
            "question": str(row.get("question", "")),
            "answer_length": clean(row.get("answer_length")),
            "chunks_retrieved": clean(row.get("chunks_retrieved")),
            "latency_ms": clean(row.get("latency_ms")),
        })
    
    lat = df["latency_ms"].dropna()
    chunks = df["chunks_retrieved"].dropna()
    
    return JSONResponse({
        "total_queries": len(df),
        "avg_latency_ms": round(float(lat.mean()), 2) if len(lat) else 0,
        "avg_chunks_retrieved": round(float(chunks.mean()), 2) if len(chunks) else 0,
        "recent_queries": rows
    })