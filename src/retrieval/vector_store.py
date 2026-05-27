from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SolarVectorStore:
    def __init__(self, collection_name: str = "solar_knowledge"):
        """
        Initializes persistent ChromaDB client, standard embedding model,
        and an advanced Cross-Encoder Reranker model.
        """
        script_dir = Path(__file__).resolve().parent
        self.chroma_db_dir = script_dir.parents[1] / "data" / "chroma_db"
        
        # Connect to persistent storage
        self.client = chromadb.PersistentClient(path=str(self.chroma_db_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Stage 1 Model (Bi-Encoder)
        print("🤖 Loading Stage 1 Embedding Model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Stage 2 Model (Cross-Encoder Reranker)
        print("🔥 Loading Stage 2 Reranker Model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
        self.reranker_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(self.reranker_name)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_name)
        self.rerank_model.eval() # Set model to evaluation mode

    def upsert_chunks(self, ids: list, embeddings: list, documents: list, metadatas: list):
        """Saves text and vector arrays to local database tables."""
        if not ids: return
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def retrieve_and_rerank(self, question: str, top_k_vector: int = 20, final_top_n: int = 5) -> list:
        """
        Two-Stage Retrieval:
        1. Fetches a wide window of candidates via Vector Search.
        2. Re-scores and re-ranks them using a Cross-Encoder to output the best N items.
        """
        # --- STAGE 1: WIDE VECTOR RETRIEVAL ---
        question_embedding = self.model.encode(question).tolist()
        raw_results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k_vector
        )
        
        if not raw_results or not raw_results['ids'] or len(raw_results['ids'][0]) == 0:
            return []

        initial_chunks = []
        for i in range(len(raw_results['ids'][0])):
            initial_chunks.append({
                "chunk_id": raw_results['ids'][0][i],
                "text": raw_results['documents'][0][i],
                "source": raw_results['metadatas'][0][i].get("source", "Unknown"),
                "page": raw_results['metadatas'][0][i].get("page", "N/A"),
                "vector_distance": raw_results['distances'][0][i] if 'distances' in raw_results else None
            })

        # --- STAGE 2: CROSS-ENCODER RERANKING ---
        # Prepare pairs for the Cross-Encoder model: [[Question, Text1], [Question, Text2], ...]
        pairs = [[question, chunk["text"]] for chunk in initial_chunks]
        
        with torch.no_grad():
            # Tokenize the pairs together
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
            # Generate logits (raw alignment scores)
            outputs = self.rerank_model(**inputs)
            # Extract scores from the single output logit channel
            scores = outputs.logits.view(-1).tolist()

        # Inject the new deep semantic score back into our chunk objects
        for idx, score in enumerate(scores):
            initial_chunks[idx]["rerank_score"] = score

        # Sort the chunks descending based on their fresh re-ranked score evaluations
        reranked_chunks = sorted(initial_chunks, key=lambda x: x["rerank_score"], reverse=True)

        # Truncate and return only the absolute highest-quality subset requested
        return reranked_chunks[:final_top_n]

    def get_count(self) -> int:
        return self.collection.count()