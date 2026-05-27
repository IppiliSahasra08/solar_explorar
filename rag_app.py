import os
import sys
from src.retrieval.vector_store import SolarVectorStore
from src.llm.generator import SolarRAGGenerator

def main():
    if "GEMINI_API_KEY" not in os.environ:
        print("❌ Error: GEMINI_API_KEY environment variable is missing.")
        sys.exit(1)

    question = "What is rapid shutdown in photovoltaic systems?"
    
    print(f"🚀 Initializing Solar Explorar Upgraded RAG Engine (With Reranking)...")
    
    # Connects to database and automatically loads both standard and reranker models
    v_store = SolarVectorStore(collection_name="solar_knowledge")
    generator = SolarRAGGenerator()
    
    print(f"\n🔎 Question: '{question}'")
    print(f"⏳ Executing Two-Stage Retrieval (Vector Search 20 ➔ Cross-Encoder Rerank 5)...")
    
    # Execute the two-stage lookup
    best_chunks = v_store.retrieve_and_rerank(
        question=question, 
        top_k_vector=20,  # Grab 20 candidates first
        final_top_n=5     # Narrow down to the best 5
    )
    
    print("\n🎯 Top 5 Reranked Sources Chosen for Gemini:")
    print("=" * 80)
    for idx, c in enumerate(best_chunks, start=1):
        print(f" Rank {idx} | Score: {c['rerank_score']:.4f} | Source: {c['source']} (Page {c['page']})")
        # Print a small snippet of the text to inspect alignment
        snippet = c['text'][:90].replace('\n', ' ')
        print(f"   Snippet: {snippet}...")
        print("-" * 80)

    print(f"\n🧠 Streaming optimized context payload to Gemini...")
    ai_response = generator.generate_answer(question=question, retrieved_chunks=best_chunks)
    
    print("\n" + "="*80)
    print("🤖 GROUNDED RAG RESPONSE (RERANKED):")
    print("="*80)
    print(ai_response)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()