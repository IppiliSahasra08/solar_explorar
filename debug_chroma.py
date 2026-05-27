from src.retrieval.vector_store import SolarVectorStore


def main():
    print("\n🚀 Connecting to ChromaDB...\n")

    try:
        v_store = SolarVectorStore(
            collection_name="solar_knowledge"
        )

        print("✅ Connected successfully")

        # Total chunks
        try:
            total_chunks = v_store.collection.count()

            print("\n📊 Database Statistics")
            print("=" * 50)
            print(f"Total chunks stored: {total_chunks}")

        except AttributeError:
            print(
                "\n⚠️ Could not access "
                "'v_store.collection'"
            )

            print(
                "Your collection may be stored "
                "under a different variable name."
            )

            print(
                "\nAvailable attributes:"
            )

            for attr in dir(v_store):
                if not attr.startswith("_"):
                    print(f"  - {attr}")

            return

        print("=" * 50)

        # Test retrieval
        test_question = (
            "What is rapid shutdown "
            "in photovoltaic systems?"
        )

        print(
            f"\n🔎 Running retrieval test:\n"
            f"'{test_question}'\n"
        )

        results = v_store.retrieve_top_chunks(
            question=test_question,
            n_results=5
        )

        print(
            f"✅ Retrieved {len(results)} chunks\n"
        )

        for idx, chunk in enumerate(results, start=1):

            print("=" * 80)
            print(f"Rank {idx}")

            print(
                f"Source: "
                f"{chunk.get('source', 'Unknown')}"
            )

            print(
                f"Page: "
                f"{chunk.get('page', 'Unknown')}"
            )

            print(
                f"Chunk ID: "
                f"{chunk.get('chunk_id', 'Unknown')}"
            )

            print(
                f"Distance: "
                f"{chunk.get('distance', 'Unknown')}"
            )

            print("-" * 80)

            preview = chunk.get("text", "")

            if len(preview) > 300:
                preview = preview[:300] + "..."

            print(preview)

            print()

        print("\n🎉 Debug completed successfully")

    except Exception as e:
        print("\n❌ ERROR")
        print("=" * 50)
        print(type(e).__name__)
        print(e)


if __name__ == "__main__":
    main()