import os
from google import genai
from google.genai import types
from src.llm.prompts import RAG_SYSTEM_INSTRUCTION, build_rag_prompt

class SolarRAGGenerator:
    def __init__(self):
        """
        Initializes the official Google GenAI client.
        Expects GEMINI_API_KEY to be set in your environment variables.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ GEMINI_API_KEY not found in environment variables!\n"
                "Please set it using: export GEMINI_API_KEY='your_key' (Mac/Linux) "
                "or set GEMINI_API_KEY='your_key' (Windows CMD)"
            )
            
        # Initialize the modern SDK client
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"

    def generate_answer(self, question: str, retrieved_chunks: list) -> str:
        """
        Takes a question and retrieved database context, formats the payload prompt,
        and returns a complete, citation-grounded response from Gemini.
        """
        if not retrieved_chunks:
            return "⚠️ No relevant reference context was found in the database to answer this question."

        user_content = build_rag_prompt(question, retrieved_chunks)

        try:
            # Call the Gemini API using relaxed token constraints to avoid truncation
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=RAG_SYSTEM_INSTRUCTION,
                    temperature=0.2,  # Low temperature keeps it highly grounded
                    max_output_tokens=2000,  # Increased to give the model plenty of room to finish
                )
            )
            
            # Return text safely, ensuring we strip trailing whitespace
            if response.text:
                return response.text.strip()
            else:
                return "⚠️ Gemini returned an empty response. Check safety settings or context lengths."
            
        except Exception as e:
            return f"❌ Gemini API Execution Failure: {str(e)}"