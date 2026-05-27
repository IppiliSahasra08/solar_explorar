import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

def main():
    # 1. Setup absolute paths relative to this script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    
    chunks_file = project_root / "data" / "chunks" / "chunks.json"
    chroma_db_dir = project_root / "data" / "chroma_db"
    
    # Check if the text chunk asset is missing
    if not chunks_file.exists():
        print(f"❌ Chunks database not found at {chunks_file}. Run chunker.py first!")
        return

    # 2. Load the text chunks
    print(f"📖 Loading parsed chunks from {chunks_file.name}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        
    if not chunks_data:
        print("❌ No text chunks found inside the JSON file.")
        return
        
    print(f"🔗 Loaded {len(chunks_data)} chunks to index.")

    # 3. Initialize Local Persistent ChromaDB Instance
    # This automatically builds the underlying SQLite & vector file structure on your disk
    print(f"📦 Initializing local persistent ChromaDB client at: data/chroma_db/")
    chroma_client = chromadb.PersistentClient(path=str(chroma_db_dir))
    
    # Define or fetch a specific project collection namespace
    collection_name = "solar_knowledge"
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # Use cosine similarity for solar domain text matching
    )

    # 4. Load the local embedding architecture model
    print("🤖 Loading embedding model 'all-MiniLM-L6-v2' (384 dimensions)...")
    # This downloads the small 90MB weights file locally on the first run, then caches it
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 5. Extract and format the content into parallel structural arrays for Chroma DB batch injection
    print("\n⚡ Generating dense vectors and upserting data into ChromaDB...")
    
    ids = []
    documents = []
    metadatas = []
    
    for chunk in chunks_data:
        # Chroma expects structural unique IDs as strings
        ids.append(f"id_{chunk['chunk_id']}")
        documents.append(chunk["text"])
        metadatas.append({"source": chunk["source"]})

    # 6. Execute Embedding Generation via Sentence Transformers
    # We pass the list of text documents to the model to get a matrix of dense numerical arrays
    raw_embeddings = model.encode(documents, show_progress_bar=True)
    
    # Convert numpy matrix outputs into standard native Python floats for JSON/Chroma serialization
    embeddings_list = raw_embeddings.tolist()

    # 7. Upsert the parallel payload arrays directly into your local database indices
    # We use upsert so that running this multiple times updates existing keys instead of throwing errors
    collection.upsert(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas,
        documents=documents
    )

    print("\n📊 Verification Statistics:")
    print(f"  └─ Target Path: {chroma_db_dir.relative_to(project_root)}/")
    print(f"  └─ Collection Name: '{collection.name}'")
    print(f"  └─ Total Indexed Rows: {collection.count()}")
    print("✨ Embedding generation and persistent database synchronization complete!")

if __name__ == "__main__":
    main()