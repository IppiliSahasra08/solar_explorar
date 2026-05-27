from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

class SolarVectorStore:
    def __init__(self, collection_name: str = "solar_knowledge"):
        """
        Initializes a local persistent instance of ChromaDB and caches the embedding model.
        Works for both data ingestion (writing) and on-demand retrieval (reading).
        """
        script_dir = Path(__file__).resolve().parent
        self.chroma_db_dir = script_dir.parents[1] / "data" / "chroma_db"
        
        # Connect to your persistent storage on disk (creates chroma.sqlite3 if missing)
        self.client = chromadb.PersistentClient(path=str(self.chroma_db_dir))
        
        # Pull your collection optimized with Cosine Similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Cache the encoder model locally inside the instance
        print("🤖 Loading embedding model 'all-MiniLM-L6-v2' (384 dimensions)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def upsert_chunks(self, ids: list, embeddings: list, documents: list, metadatas: list):
        """
        Saves parallel arrays of data directly down into local binary storage tables.
        Required by ingest.py.
        """
        if not ids:
            print("⚠️ No IDs provided to upsert.")
            return
            
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def retrieve_top_chunks(self, question: str, n_results: int = 5) -> list:
        """
        Processes a raw text question, converts it to a vector, 
        and extracts the Top N most semantically relevant chunks.
        Required by search.py and future FastAPI endpoints.
        """
        # 1. On-the-fly embedding generation for the incoming question
        question_embedding = self.model.encode(question).tolist()
        
        # 2. Query the persistent database vector space
        raw_results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results
        )
        
        # 3. Restructure Chroma's parallel arrays output into a clean list of dictionaries
        formatted_chunks = []
        
        if not raw_results or not raw_results['ids'] or len(raw_results['ids'][0]) == 0:
            return formatted_chunks
            
        for i in range(len(raw_results['ids'][0])):
            formatted_chunks.append({
                "chunk_id": raw_results['ids'][0][i],
                "text": raw_results['documents'][0][i],
                "source": raw_results['metadatas'][0][i].get("source", "Unknown"),
                "page": raw_results['metadatas'][0][i].get("page", "N/A"),
                "distance": raw_results['distances'][0][i] if 'distances' in raw_results else None
            })
            
        return formatted_chunks

    def get_count(self) -> int:
        """Returns total records inside the current storage index"""
        return self.collection.count()