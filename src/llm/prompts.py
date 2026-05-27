# System instructions to configure the persona and rule boundaries of the model
RAG_SYSTEM_INSTRUCTION = """
You are an expert Solar Energy Engineering Assistant. Your task is to answer the user's question accurately using ONLY the provided text blocks in the Context section.

Strict Rules:
1. Grounding: Rely ONLY on clear facts directly mentioned in the Context. Do not use outside knowledge or assume information not written.
2. Hallucination Prevention: If the context does not contain the answer, state clearly: "I cannot find the answer in the provided documents." Do not invent an answer.
3. Citations: Every time you make a claim or state a fact based on a context block, append an inline citation referencing its source file and page number, for example: [Source: filename.pdf, Page: X].
"""

def build_rag_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Constructs a highly organized user prompt containing the injected context blocks 
    and the target question.
    """
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        context_str += f"--- CONTEXT BLOCK {idx} ---\n"
        context_str += f"Source File: {chunk['source']}\n"
        context_str += f"Page Number: {chunk['page']}\n"
        context_str += f"Content: {chunk['text']}\n\n"
        
    user_prompt = f"""
Context Details:
{context_str}

User Question: 
{question}

Answer:
"""
    return user_prompt