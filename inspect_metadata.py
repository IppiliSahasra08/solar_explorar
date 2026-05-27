from src.retrieval.vector_store import SolarVectorStore

v_store = SolarVectorStore(
    collection_name="solar_knowledge"
)

results = v_store.collection.get(
    limit=10,
    include=["metadatas"]
)

for i, metadata in enumerate(results["metadatas"]):
    print(f"\nChunk {i}")
    print(metadata)