# test_gemini.py

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print(f"Key loaded: {api_key is not None}")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain what a photovoltaic system is in one sentence."
)

print("\nResponse:")
print(response.text)