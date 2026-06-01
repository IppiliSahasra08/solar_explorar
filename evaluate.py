"""
RAG Evaluation Script
Evaluates the Solar Explorer RAG system against ground truth questions.
Computes metrics without external RAGAS dependency.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

# Ensure the project root is on sys.path for local package imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.vector_store import SolarVectorStore
from src.llm.generator import SolarRAGGenerator


def load_questions(json_path: str) -> list:
    """Load evaluation questions from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_keyword_overlap(text: str, keywords: List[str]) -> float:
    """Compute the ratio of keywords found in text."""
    if not keywords:
        return 1.0
    
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute simple word-based similarity between two texts.
    Uses Jaccard similarity on word sets.
    """
    # Tokenize and normalize
    def tokenize(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return set(words)
    
    set1 = tokenize(text1)
    set2 = tokenize(text2)
    
    if not set1 or not set2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Compute faithfulness score: how well the answer is supported by contexts.
    Checks if expected keywords from ground truth appear in the context.
    """
    if not contexts:
        return 0.0
    
    combined_context = ' '.join(contexts).lower()
    answer_lower = answer.lower()
    
    # Extract key claims from answer (simplified approach)
    # Check if the context contains most of the answer's content
    answer_words = set(re.findall(r'\b\w{4,}\b', answer_lower))
    context_words = set(re.findall(r'\b\w{4,}\b', combined_context))
    
    if not answer_words:
        return 1.0
    
    # Ratio of answer words found in context
    overlap = len(answer_words & context_words) / len(answer_words)
    return min(overlap, 1.0)


def compute_answer_relevance(question: str, answer: str) -> float:
    """
    Compute answer relevance: how relevant the answer is to the question.
    Uses keyword overlap and semantic similarity.
    """
    # Extract key terms from question
    question_keywords = set(re.findall(r'\b\w{4,}\b', question.lower()))
    answer_keywords = set(re.findall(r'\b\w{4,}\b', answer.lower()))
    
    if not question_keywords:
        return 1.0
    
    # Check keyword overlap
    keyword_overlap = len(question_keywords & answer_keywords) / len(question_keywords)
    
    # Semantic similarity
    semantic_sim = compute_semantic_similarity(question, answer)
    
    # Combined score (weighted average)
    return 0.4 * keyword_overlap + 0.6 * semantic_sim


def compute_context_recall(retrieved_contexts: List[str], expected_contexts: List[str]) -> float:
    """
    Compute context recall: how well retrieved contexts cover expected contexts.
    Uses keyword matching against expected contexts.
    """
    if not expected_contexts:
        return 1.0
    
    combined_retrieved = ' '.join(retrieved_contexts).lower()
    combined_expected = ' '.join(expected_contexts).lower()
    
    expected_keywords = set(re.findall(r'\b\w{4,}\b', combined_expected))
    retrieved_keywords = set(re.findall(r'\b\w{4,}\b', combined_retrieved))
    
    if not expected_keywords:
        return 1.0
    
    # Jaccard similarity between expected and retrieved keywords
    overlap = len(expected_keywords & retrieved_keywords) / len(expected_keywords)
    return overlap


def compute_context_precision(retrieved_contexts: List[str], expected_keywords: List[str]) -> float:
    """
    Compute context precision: how precisely relevant content is ranked.
    Checks if expected keywords appear in top-ranked contexts.
    """
    if not expected_keywords or not retrieved_contexts:
        return 0.0
    
    # Weight each context by its position (earlier = more important)
    total_score = 0.0
    weight_sum = 0.0
    
    for i, context in enumerate(retrieved_contexts, 1):
        context_lower = context.lower()
        # Position weight: higher weight for earlier positions
        position_weight = 1.0 / i
        
        # Check keyword presence
        found = sum(1 for kw in expected_keywords if kw.lower() in context_lower)
        score = found / len(expected_keywords) if expected_keywords else 0.0
        
        total_score += score * position_weight
        weight_sum += position_weight
    
    return total_score / weight_sum if weight_sum > 0 else 0.0


def run_evaluation(questions: list, top_k_vector: int = 20, final_top_n: int = 5) -> Dict[str, float]:
    """
    Run complete evaluation on all questions.
    
    Args:
        questions: List of question dictionaries
        top_k_vector: Initial number of chunks to retrieve
        final_top_n: Number of chunks after reranking
    
    Returns:
        Dictionary with average metric scores
    """
    # Initialize RAG components
    v_store = SolarVectorStore(collection_name="solar_knowledge")
    generator = SolarRAGGenerator()
    
    # Accumulate scores
    faithfulness_scores = []
    answer_relevance_scores = []
    context_recall_scores = []
    context_precision_scores = []
    
    print(f"\n📊 Running evaluation on {len(questions)} questions...\n")
    print("-" * 80)
    
    for idx, q in enumerate(questions, 1):
        question_text = q['question']
        ground_truth = q.get('ground_truth', '')
        expected_contexts = q.get('contexts', [])
        expected_keywords = q.get('expected_keywords', [])
        
        print(f"\n[{idx}/{len(questions)}] {question_text[:70]}...")
        
        try:
            # Execute two-stage retrieval
            retrieved_chunks = v_store.retrieve_and_rerank(
                question=question_text,
                top_k_vector=top_k_vector,
                final_top_n=final_top_n
            )
            
            # Generate answer with retrieved context
            answer = generator.generate_answer(
                question=question_text,
                retrieved_chunks=retrieved_chunks
            )
            
            # Build context texts from retrieved chunks
            retrieved_contexts = [
                f"Source: {c['source']} (Page {c['page']})\n{c['text']}"
                for c in retrieved_chunks
            ]
            
            # Compute metrics
            faithfulness = compute_faithfulness(answer, retrieved_contexts)
            answer_relevance = compute_answer_relevance(question_text, answer)
            context_recall = compute_context_recall(retrieved_contexts, expected_contexts)
            context_precision = compute_context_precision(
                retrieved_contexts, 
                expected_keywords + q.get('expected_keywords', [])
            )
            
            faithfulness_scores.append(faithfulness)
            answer_relevance_scores.append(answer_relevance)
            context_recall_scores.append(context_recall)
            context_precision_scores.append(context_precision)
            
            print(f"   ✓ F: {faithfulness:.3f} | AR: {answer_relevance:.3f} | CR: {context_recall:.3f} | CP: {context_precision:.3f}")
            
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            # Use neutral scores for failed evaluations
            faithfulness_scores.append(0.0)
            answer_relevance_scores.append(0.0)
            context_recall_scores.append(0.0)
            context_precision_scores.append(0.0)
    
    print("\n" + "-" * 80)
    
    # Compute average scores
    n = len(questions)
    avg_scores = {
        'faithfulness': sum(faithfulness_scores) / n if n > 0 else 0.0,
        'answer_relevance': sum(answer_relevance_scores) / n if n > 0 else 0.0,
        'context_recall': sum(context_recall_scores) / n if n > 0 else 0.0,
        'context_precision': sum(context_precision_scores) / n if n > 0 else 0.0
    }
    
    return avg_scores


def save_results(scores: Dict[str, float], output_path: str):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {output_path}")


def main():
    """Main evaluation workflow."""
    print("=" * 80)
    print("🚀 Solar Explorer RAG Evaluation")
    print("=" * 80)
    
    # Check for required environment variables
    if "GEMINI_API_KEY" not in os.environ:
        print("❌ Error: GEMINI_API_KEY environment variable is missing.")
        sys.exit(1)
    
    # Paths
    questions_path = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
    results_path = PROJECT_ROOT / "data" / "evaluation" / "ragas_results.json"
    
    # Verify input file exists
    if not questions_path.exists():
        print(f"❌ Error: Questions file not found at {questions_path}")
        sys.exit(1)
    
    # Load questions
    print(f"\n📂 Loading questions from: {questions_path}")
    questions = load_questions(str(questions_path))
    print(f"   Found {len(questions)} questions to evaluate")
    
    # Run evaluation
    print(f"\n🔄 Running RAG system on all questions...")
    print("   Retrieval: Vector Search (20) → Cross-Encoder Rerank (5)")
    scores = run_evaluation(questions)
    
    # Print results
    print("\n" + "=" * 80)
    print("📈 EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Faithfulness:       {scores['faithfulness']:.4f}")
    print(f"  Answer Relevance:   {scores['answer_relevance']:.4f}")
    print(f"  Context Recall:     {scores['context_recall']:.4f}")
    print(f"  Context Precision:  {scores['context_precision']:.4f}")
    print("=" * 80)
    
    # Save results
    save_results(scores, str(results_path))
    
    print("\n✨ Evaluation complete!")


if __name__ == "__main__":
    main()