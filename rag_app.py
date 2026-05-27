import os
import sys
from src.retrieval.vector_store import SolarVectorStore
from src.llm.generator import SolarRAGGenerator

def main():
    # Defensive check: Ensure API key is configured before spinning up models
    if "GEMINI_API_KEY" not in os.environ:
        print("❌ Error: GEMINI_API_KEY environment variable is missing.")
        print("Set it via terminal before running this script.")
        print("Example (Windows): set GEMINI_API_KEY=AIzaSy...")
        print("Example (Bash/Zsh): export GEMINI_API_KEY='AIzaSy...'")
        sys.exit(1)

    # 1. Define the user's technical question
    question = "How should conductors be sized?"
    
    print(f"🚀 Initializing Solar Explorar RAG Application Core...")
    
    # 2. Connect to the local Vector Database layer
    v_store = SolarVectorStore(collection_name="solar_knowledge")
    
    # 3. Instantiate the Gemini generation engine
    generator = SolarRAGGenerator()
    
    print(f"\n🔎 Question: '{question}'")
    print(f"⏳ Step 1: Querying ChromaDB for top relevant context records...")
    
    # 4. Pull top 5 relevant document chunks on demand
    top_chunks = v_store.retrieve_top_chunks(question=question, n_results=5)
    print(f"✅ Found {len(top_chunks)} chunks in vector space.")
    
    # Quick debug block to see what documents are being sent to Gemini
    print("\n📚 Context Sources Sent to Gemini:")
    for c in top_chunks:
        print(f"  ├─ Document: {c['source']} (Page {c['page']}) [Distance: {c['distance']:.4f}]")

    print(f"\n🧠 Step 2: Injecting context and streaming payload to Gemini ({generator.model_name})...")
    
    # 5. Run the complete generation pipeline
    ai_response = generator.generate_answer(question=question, retrieved_chunks=top_chunks)
    
    # 6. Output the final grounded response
    print("\n" + "="*80)
    print("🤖 GROUNDED RAG RESPONSE:")
    print("="*80)
    print(ai_response)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()