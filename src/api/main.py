import os
import time

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.retrieval.vector_store import SolarVectorStore
from src.llm.generator import SolarRAGGenerator
from src.telemetry.logger import log_query
#uvicorn src.api.main:app --reload
#Open your browser and go to http://localhost:8000/docs — you'll see an interactive Swagger UI. Click /api/ask → "Try it out" → paste a question → Execute.
# visit http://localhost:8000/api/telemetry in your browser to see your logged queries.
app = FastAPI(title="Solar Explorer RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
v_store = SolarVectorStore(collection_name="solar_knowledge")
generator = SolarRAGGenerator()

class AskRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "Solar Explorer RAG API is running"}

@app.get("/api/telemetry")
def get_telemetry():
    from src.telemetry.logger import QUERY_LOG

    if not QUERY_LOG.exists():
        return {"queries": [], "total": 0, "avg_latency_ms": 0}

    df = pd.read_csv(QUERY_LOG)
    return {
        "total_queries": len(df),
        "avg_latency_ms": round(df["latency_ms"].mean(), 2),
        "avg_chunks_retrieved": round(df["chunks_retrieved"].mean(), 2),
        "recent_queries": df.tail(5).to_dict(orient="records"),
    }

@app.post("/api/ask")
def ask(req: AskRequest):
    start = time.time()
    
    chunks = v_store.retrieve_and_rerank(
        req.question, 
        top_k_vector=20, 
        final_top_n=5
    )
    answer = generator.generate_answer(req.question, chunks)
    
    latency_ms = (time.time() - start) * 1000
    log_query(req.question, answer, latency_ms, len(chunks))
    
    return {
        "answer": answer,
        "sources": [{"source": c["source"], "page": c["page"]} for c in chunks],
        "latency_ms": round(latency_ms, 2),
        "chunks_retrieved": len(chunks)
    }