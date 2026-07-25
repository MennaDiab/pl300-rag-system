"""
07_prompting.py
-----------------
Step 7: Build the RAG prompt from retrieved context and call the local
Ollama LLM (qwen3:8b) to answer the question.
"""

import os
import ollama

from importlib import import_module
retrieve_mod = import_module("06_retrieve_context")

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")


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


def ask_llm(prompt, model=OLLAMA_MODEL):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def answer_question(question, k=5):
    chunks_result = retrieve_mod.retrieve(question, k=k)
    context = "\n\n".join(chunks_result)
    prompt = build_prompt(question, context)
    answer = ask_llm(prompt)
    return answer, chunks_result


if __name__ == "__main__":
    answer, chunks = answer_question("What is DAX?")
    print(answer)
