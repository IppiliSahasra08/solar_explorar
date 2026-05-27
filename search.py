from src.retrieval.vector_store import SolarVectorStore

def main():
    # 1. Connect to our localized vector database service layer
    v_store = SolarVectorStore(collection_name="solar_knowledge")
    
    # 2. Define your domain testing question
    user_question = "What is rapid shutdown in photovoltaic systems?"
    
    print(f"\n🔎 Incoming User Query: '{user_question}'")
    print("⏳ Translating query into a vector and scanning ChromaDB index arrays...\n")
    
    # 3. Fetch top 5 chunks dynamically on demand
    top_chunks = v_store.retrieve_top_chunks(question=user_question, n_results=5)
    
    # 4. Display the retrieved chunks
    if not top_chunks:
        print("⚠️ No matching context records found. Is your database populated?")
        return
        
    print(f"🎯 Successfully retrieved {len(top_chunks)} relevant context blocks:")
    print("=" * 70)
    
    for idx, chunk in enumerate(top_chunks, start=1):
        print(f" Rank {idx} | ID: {chunk['chunk_id']} | Source: {chunk['source']} (Page {chunk['page']})")
        print(f" Similarity Score (Cosine Distance): {chunk['distance']:.4f}")
        print("-" * 70)
        print(f"{chunk['text']}")
        print("=" * 70)

if __name__ == "__main__":
    main()