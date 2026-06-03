"""
RAG Evaluation Script using RAGAS with local LLM
Evaluates the Solar Explorer RAG system against ground truth questions.
Uses Ollama with Llama3 (or similar) for zero API costs.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Ensure the project root is on sys.path for local package imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# RAGAS and dataset imports
from ragas import evaluate
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_recall import context_recall
from ragas.metrics._context_precision import context_precision
from datasets import Dataset

from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

# RAG system imports
from src.retrieval.vector_store import SolarVectorStore
from src.llm.generator import SolarRAGGenerator


def load_questions(json_path: str) -> list:
    """Load evaluation questions from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def init_evaluator_llm():
    from openai import OpenAI
    from ragas.llms import llm_factory
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return llm_factory(model="gemma2-ctx8k", provider="openai", client=ollama_client)

def init_evaluator_embeddings():
    from openai import OpenAI
    from ragas.embeddings.base import embedding_factory
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return embedding_factory(model="nomic-embed-text", provider="openai", client=ollama_client)

def run_rag_system(questions: List[Dict], top_k_vector: int = 20, final_top_n: int = 5) -> Dict[str, List]:
    """
    Run the RAG system on all questions to generate answers and retrieve contexts.
    
    Args:
        questions: List of question dictionaries with ground_truth
        top_k_vector: Initial retrieval count
        final_top_n: Final reranked count
    
    Returns:
        Dictionary with user_input, response, retrieved_contexts, reference
    """
    print("\n🚀 Running RAG system on all questions...")
    
    # Initialize RAG components
    v_store = SolarVectorStore(collection_name="solar_knowledge")
    generator = SolarRAGGenerator()
    
    results = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": []
    }
    
    print(f"📊 Processing {len(questions)} questions...\n")
    print("-" * 80)
    
    for idx, q in enumerate(questions, 1):
        question_text = q['question']
        ground_truth = q.get('ground_truth', '')
        
        print(f"\n[{idx}/{len(questions)}] {question_text[:65]}...")
        
        try:
            # Two-stage retrieval
            retrieved_chunks = v_store.retrieve_and_rerank(
                question=question_text,
                top_k_vector=top_k_vector,
                final_top_n=final_top_n
            )
            
            # Generate answer
            answer = generator.generate_answer(
                question=question_text,
                retrieved_chunks=retrieved_chunks
            )
            
            # Build context strings (RAGAS format: list of strings)
            contexts = [c['text'] for c in retrieved_chunks]
            
            # Store results
            results["user_input"].append(question_text)
            results["response"].append(answer)
            results["retrieved_contexts"].append(contexts)
            results["reference"].append(ground_truth)
            
            print(f"   ✓ Answer: {len(answer)} chars | Contexts: {len(contexts)} chunks")
            
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            results["user_input"].append(question_text)
            results["response"].append(f"Error: {str(e)}")
            results["retrieved_contexts"].append([])
            results["reference"].append(ground_truth)
    
    print("\n" + "-" * 80)
    return results


def evaluate_with_ragas(
    eval_data: Dict[str, List],
    evaluator_llm,
    evaluator_embeddings
) -> Dict[str, Any]:
    import time
    import numpy as np
    from ragas.run_config import RunConfig

    print("\n🔬 Computing RAGAS metrics...")

    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings
    context_recall.llm = evaluator_llm
    context_precision.llm = evaluator_llm

    BATCH_SIZE = 4        # 4 questions × 4 metrics = 16 calls, just under 20 RPM
    SLEEP_BETWEEN = 65    # wait 65 seconds between batches (resets the per-minute quota)

    all_scores = {
        'faithfulness': [],
        'answer_relevancy': [],
        'context_recall': [],
        'context_precision': []
    }

    questions = eval_data['user_input']
    total_batches = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(questions))

        print(f"\n   Batch {batch_idx+1}/{total_batches} (questions {start+1}–{end})")

        batch_data = {k: v[start:end] for k, v in eval_data.items()}
        dataset = Dataset.from_dict(batch_data)

        run_config = RunConfig(timeout=180, max_retries=3, max_wait=60, max_workers=1)

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            raise_exceptions=False,
            run_config=run_config
        )

        for key in all_scores:
            val = result[key]
            if isinstance(val, list):
                valid = [v for v in val if v is not None and not np.isnan(v)]
                all_scores[key].append(float(np.mean(valid)) if valid else 0.0)
            elif val is not None and not np.isnan(float(val)):
                all_scores[key].append(float(val))
            else:
                all_scores[key].append(0.0)

        if batch_idx < total_batches - 1:
            print(f"   ⏳ Waiting 65s to respect free tier rate limit...")
            time.sleep(65)

    scores = {
        'faithfulness':      float(np.mean(all_scores['faithfulness'])),
        'answer_relevance':  float(np.mean(all_scores['answer_relevancy'])),
        'context_recall':    float(np.mean(all_scores['context_recall'])),
        'context_precision': float(np.mean(all_scores['context_precision'])),
    }

    print("   ✓ RAGAS evaluation complete")
    return scores

def save_results(scores: Dict[str, float], output_path: str):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {output_path}")


def main():
    """Main evaluation workflow."""
    print("=" * 80)
    print("🚀 Solar Explorer RAG Evaluation (RAGAS + Ollama)")
    print("=" * 80)
    
    # Check for required environment variables (for generator)
    if "GEMINI_API_KEY" not in os.environ:
        print("❌ Error: GEMINI_API_KEY environment variable is missing.")
        print("   The RAG generator needs GEMINI_API_KEY to generate answers.")
        sys.exit(1)
    
    # Configuration
    MODEL_NAME = os.environ.get("EVALUATION_MODEL", "llama3.1:8b")  # Ollama model
    EMBED_MODEL = os.environ.get("EVALUATION_EMBED", "nomic-embed-text")  # Ollama embed model
    TOP_K_VECTOR = 20
    FINAL_TOP_N = 5
    
    print(f"\n📋 Configuration:")
    print(f"   Evaluator LLM: {MODEL_NAME} (via Ollama OpenAI-compatible API)")
    print(f"   Evaluator Embeddings: {EMBED_MODEL}")
    print(f"   Retrieval: Vector Search ({TOP_K_VECTOR}) → Rerank ({FINAL_TOP_N})")
    
    # Paths
    questions_path = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
    results_path = PROJECT_ROOT / "data" / "evaluation" / "ragas_results.json"
    
    # Verify input file
    if not questions_path.exists():
        print(f"\n❌ Error: Questions file not found at {questions_path}")
        print("   Please ensure data/evaluation/questions.json exists")
        sys.exit(1)
    
    # Load questions
    print(f"\n📂 Loading questions from: {questions_path}")
    questions = load_questions(str(questions_path))
    # Optional: limit the sample size for a quick smoke test
    questions = questions[:20]
    print(f"   Found {len(questions)} questions to evaluate")
    
    # Step 1: Run RAG system on all questions
    print("\n" + "=" * 80)
    print("PHASE 1: RAG Response Generation")
    print("=" * 80)
    eval_data = run_rag_system(questions, TOP_K_VECTOR, FINAL_TOP_N)
    
    # Step 2: Initialize evaluator LLM and embeddings
    print("\n" + "=" * 80)
    print("PHASE 2: RAGAS Evaluation")
    print("=" * 80)
    
    evaluator_llm = init_evaluator_llm()
    evaluator_embeddings = init_evaluator_embeddings()
    
    # Step 3: Evaluate with RAGAS
    scores = evaluate_with_ragas(
        eval_data,
        evaluator_llm,
        evaluator_embeddings
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("📈 EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Faithfulness:        {scores['faithfulness']:.4f}")
    print(f"  Answer Relevance:    {scores['answer_relevance']:.4f}")
    print(f"  Context Recall:       {scores['context_recall']:.4f}")
    print(f"  Context Precision:    {scores['context_precision']:.4f}")
    print("=" * 80)
    
    # Save results
    save_results(scores, str(results_path))
    
    print("\n✨ Evaluation complete!")


if __name__ == "__main__":
    main()