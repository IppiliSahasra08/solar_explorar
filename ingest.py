import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
# Import the custom interface abstraction from your retrieval layer
from src.retrieval.vector_store import SolarVectorStore

def run_ingestion_pipeline():
    project_root = Path(__file__).resolve().parent
    chunks_file = project_root / "data" / "chunks" / "chunks.json"
    
    if not chunks_file.exists():
        print(f"❌ Target context source file missing at: {chunks_file}")
        print("Please ensure you run your extraction and chunking layers first!")
        return

    # 1. Load the chunk data schema
    print(f"📖 Reading structural chunk schema from {chunks_file.name}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        
    print(f"🔗 Total data objects found for conversion: {len(chunks_data)}")

    # 2. Instantiate our vector storage layer
    print("📦 Connecting to local persistent ChromaDB storage engine...")
    v_store = SolarVectorStore(collection_name="solar_knowledge")

    # 3. Load the ML encoder weights
    print("🤖 Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4. Parse fields into flat lists for parallel array alignment
    print("\n⚡ Processing text layers and constructing metadata maps...")
    ids = []
    documents = []
    metadatas = []
    
    # Locate this section inside your ingest.py script:
    for chunk in chunks_data:
        ids.append(f"id_{chunk['chunk_id']}")
        documents.append(chunk["text"])
        
        # FIX: Replace the hardcoded "page": 1 with the real JSON attribute mapping
        metadatas.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"]  # <-- Dynamically reading from our updated chunker!
        })

    # 5. Extract vector matrices from the raw strings
    print("🧠 Generating high-dimensional vector space embeddings...")
    embeddings_matrix = model.encode(documents, show_progress_bar=True)
    embeddings_list = embeddings_matrix.tolist()

    # 6. Commit arrays into persistent disk tables via our manager class
    print("💾 Committing vectors and clean texts to disk storage tables...")
    v_store.upsert_chunks(
        ids=ids,
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas
    )

    print("\n--- INGESTION COMPLETE ---")
    print(f"✅ Successfully initialized at: data/chroma_db/")
    print(f"📊 Vector rows synced to SQLite index table: {v_store.get_count()}")

if __name__ == "__main__":
    run_ingestion_pipeline()