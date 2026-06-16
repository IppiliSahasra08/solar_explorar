import json
from pathlib import Path
from src.retrieval.vector_store import SolarVectorStore

def run_ingestion_pipeline():
    project_root = Path(__file__).resolve().parent
    chunks_file = project_root / "data" / "chunks" / "chunks.json"
    
    if not chunks_file.exists():
        print(f"❌ Target context source file missing at: {chunks_file}")
        return

    # 1. Load the chunk data schema
    print(f"📖 Reading structural chunk schema from {chunks_file.name}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        
    print(f"🔗 Total data objects found for conversion: {len(chunks_data)}")

    # 2. Instantiate our vector storage layer
    print("📦 Connecting to local persistent ChromaDB storage engine...")
    v_store = SolarVectorStore(collection_name="solar_knowledge_v2")

    # 3. Parse fields into flat lists
    print("\n⚡ Processing text layers and constructing metadata maps...")
    all_ids = []
    all_documents = []
    all_metadatas = []
    
    for chunk in chunks_data:
        all_ids.append(f"id_{chunk['chunk_id']}")
        all_documents.append(chunk["text"])
        all_metadatas.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"]
        })

    # 4. Commit data to cloud embeddings using 100-item chunks
    print("🧠 Uploading text chunks to Gemini Cloud Embeddings in batches...")
    
    batch_size = 100
    total_chunks = len(all_ids)
    
    for i in range(0, total_chunks, batch_size):
        # Slice lists into blocks of 100 items
        batch_ids = all_ids[i:i + batch_size]
        batch_docs = all_documents[i:i + batch_size]
        batch_meta = all_metadatas[i:i + batch_size]
        
        print(f"💾 Syncing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} (Items {i} to {min(i + batch_size, total_chunks)})...")
        
        v_store.upsert_chunks(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )

    print("\n--- INGESTION COMPLETE ---")
    print(f"✅ Successfully initialized at: data/chroma_db/")
    print(f"📊 Total Vector rows synced to SQLite index table: {v_store.get_count()}")

if __name__ == "__main__":
    run_ingestion_pipeline()