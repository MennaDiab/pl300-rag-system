"""
07_prompting.py
-----------------
Step 7: Build the RAG prompt from retrieved context and call the LLM
via OpenRouter (https://openrouter.ai) to answer the question.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

from importlib import import_module
retrieve_mod = import_module("06_retrieve_context")

load_dotenv()

OPENROUTER_API_KEY = st.secrets.get(
    "OPENROUTER_API_KEY",
    os.getenv("OPENROUTER_API_KEY")
)

OPENROUTER_MODEL = st.secrets.get(
    "OPENROUTER_MODEL",
    os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def build_prompt(question, context):
    prompt = f"""
You are an expert Microsoft PL-300 certification tutor.

Instructions:
1. Answer ONLY using the provided context.
2. If the answer is not in the context, say:
   "I don't have enough information in the knowledge base."
3. If the question is about an exam concept, explain it clearly.
4. If appropriate, give an example.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt


def ask_llm(prompt, model=OPENROUTER_MODEL):
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file "
            "(see .env.example) or export it as an environment variable."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def answer_question(question, k=5):
    chunks_result = retrieve_mod.retrieve(question, k=k)
    context = "\n\n".join(chunks_result)
    prompt = build_prompt(question, context)
    answer = ask_llm(prompt)
    return answer, chunks_result


if __name__ == "__main__":
    answer, chunks = answer_question("What is DAX?")
    print(answer)
