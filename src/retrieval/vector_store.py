import os
from pathlib import Path
import chromadb
from google import genai
from chromadb import Documents, EmbeddingFunction, Embeddings

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        # Initializes the client using your system environment variable
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    def __call__(self, input: Documents) -> Embeddings:
        # Convert ChromaDB's Documents type to a clean list of pure Python strings
        texts = [str(doc) for doc in input]
        
        # Using the direct text-embedding-004 identifier string
        response = self.client.models.embed_content(
            model="text-embedding-004", 
            contents=texts
        )
        return [embedding.values for embedding in response.embeddings]

class SolarVectorStore:
    def __init__(self, collection_name: str = "solar_knowledge_v2"):
        """
        Initializes persistent ChromaDB client using cloud-backed 
        Gemini embeddings to maintain zero-RAM footprint on Render Free.
        """
        # 1. Connect our custom Gemini Embedding engine
        self.embedding_function = GeminiEmbeddingFunction()
        
        script_dir = Path(__file__).resolve().parent
        self.chroma_db_dir = script_dir.parents[1] / "data" / "chroma_db"
        
        # 2. Connect to persistent storage
        self.client = chromadb.PersistentClient(path=str(self.chroma_db_dir))
        
        # 3. Register the embedding function directly with ChromaDB
        # This tells ChromaDB to automatically use Gemini whenever we call .query()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        print("🚀 SolarVectorStore initialized using Gemini Cloud Embeddings!")

    def upsert_chunks(self, ids: list, documents: list, metadatas: list):
        """Saves text chunks directly. ChromaDB handles the embeddings automatically now."""
        if not ids: return
        # Notice we don't pass an embeddings array anymore; ChromaDB handles it via self.embedding_function
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def retrieve_and_rerank(self, question: str, final_top_n: int = 5) -> list:
        """
        Retrieves relevant solar data context using Gemini embeddings.
        Note: Local Cross-Encoder removed to preserve Render container memory.
        """
        # ChromaDB automatically embeds the question string using Gemini behind the scenes
        raw_results = self.collection.query(
            query_texts=[question],
            n_results=final_top_n
        )
        
        if not raw_results or not raw_results['ids'] or len(raw_results['ids'][0]) == 0:
            return []

        final_chunks = []
        for i in range(len(raw_results['ids'][0])):
            final_chunks.append({
                "chunk_id": raw_results['ids'][0][i],
                "text": raw_results['documents'][0][i],
                "source": raw_results['metadatas'][0][i].get("source", "Unknown"),
                "page": raw_results['metadatas'][0][i].get("page", "N/A"),
                "vector_distance": raw_results['distances'][0][i] if 'distances' in raw_results else None
            })

        return final_chunks

    def get_count(self) -> int:
        return self.collection.count()