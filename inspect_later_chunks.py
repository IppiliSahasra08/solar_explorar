# inspect_later_chunks.py

from src.retrieval.vector_store import SolarVectorStore

v_store = SolarVectorStore(
    collection_name="solar_knowledge"
)

results = v_store.collection.get(
    offset=500,
    limit=20,
    include=["metadatas"]
)

for i, metadata in enumerate(results["metadatas"]):
    print(f"\nChunk {500 + i}")
    print(metadata)