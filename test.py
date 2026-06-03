# test_fixed_final.py
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

print("=" * 60)
print("RAGAS 0.4.3 - Ollama Setup via OpenAI-Compatible API")
print("=" * 60)

# ============================================
# STEP 1: Create OpenAI client pointing to Ollama
# ============================================
print("\n[1/4] Creating OpenAI-compatible client for Ollama...")

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama's OpenAI API endpoint
    api_key="ollama"  # Ollama doesn't require a real API key
)

print(f"    ✓ Client created: {ollama_client.base_url}")

# ============================================
# STEP 2: Create LLM using llm_factory
# ============================================
print("\n[2/4] Creating evaluator LLM...")

MODEL_NAME = "llama3.1:8b"  # Change this to your actual model

evaluator_llm = llm_factory(
    model=MODEL_NAME,
    provider="openai",  # Use OpenAI provider with custom client
    client=ollama_client
)

print(f"    ✓ LLM type: {type(evaluator_llm)}")
print(f"    ✓ Model: {MODEL_NAME}")

# ============================================
# STEP 3: Create Embeddings using embedding_factory
# ============================================
print("\n[3/4] Creating embeddings...")

EMBED_MODEL = "nomic-embed-text"  # Change this to your embedding model

# Note: For embeddings, we may need a different approach
# Check if Ollama supports embeddings via OpenAI API
try:
    evaluator_embeddings = embedding_factory(
        model=EMBED_MODEL,
        provider="openai",
        client=ollama_client
    )
    print(f"    ✓ Embeddings type: {type(evaluator_embeddings)}")
    print(f"    ✓ Embedding model: {EMBED_MODEL}")
except Exception as e:
    print(f"    ⚠ Embeddings setup failed: {e}")
    print("    → Some metrics (AnswerRelevancy) may not work")
    evaluator_embeddings = None

# ============================================
# STEP 4: Test metric creation
# ============================================
print("\n[4/4] Testing metric creation...")

from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision

try:
    print("\n    Creating Faithfulness...")
    f = Faithfulness(llm=evaluator_llm)
    print(f"    ✓ Faithfulness: {f.name}")
except Exception as e:
    print(f"    ✗ Faithfulness failed: {e}")

try:
    print("\n    Creating ContextRecall...")
    cr = ContextRecall(llm=evaluator_llm)
    print(f"    ✓ ContextRecall: {cr.name}")
except Exception as e:
    print(f"    ✗ ContextRecall failed: {e}")

try:
    print("\n    Creating ContextPrecision...")
    cp = ContextPrecision(llm=evaluator_llm)
    print(f"    ✓ ContextPrecision: {cp.name}")
except Exception as e:
    print(f"    ✗ ContextPrecision failed: {e}")

if evaluator_embeddings:
    try:
        print("\n    Creating AnswerRelevancy...")
        ar = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        print(f"    ✓ AnswerRelevancy: {ar.name}")
    except Exception as e:
        print(f"    ✗ AnswerRelevancy failed: {e}")

print("\n" + "=" * 60)
print("Setup complete!")
print("=" * 60)

# ============================================
# USAGE: Your evaluate() call should now work
# ============================================
print("\nReady to use in evaluate():\n")

evaluate_code = '''result = evaluate(
    dataset,
    metrics=[
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        ) if evaluator_embeddings else None,
        ContextRecall(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm)
    ]
)'''

print(evaluate_code)