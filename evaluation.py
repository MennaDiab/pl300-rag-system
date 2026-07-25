"""
evaluation.py
--------------
Step 8: Evaluate the retriever with Precision@K, Recall@K, Hit Rate@K, and MRR
against a small hand-labeled ground-truth set.
"""

import os
import pandas as pd

from importlib import import_module
retrieve_mod = import_module("06_retrieve_context")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

GROUND_TRUTH = {
    "What is DAX?": [551, 708, 723],
    "How do I configure row-level security?": [105, 110, 111],
    "What is a calculated column?": [150, 622, 556],
    "What is the difference between measures and calculated columns?": [557, 181, 556],
    "What is filter context?": [560, 562, 28],
    "What are DAX variables?": [719, 694, 695],
}


def retrieve_top_k_semantic(query, k=5, base_dir=BASE_DIR):
    index, metadata, embedding_model = retrieve_mod.load_index_and_metadata(base_dir)
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding.astype("float32"), k)

    results = pd.DataFrame({
        "document_id": indices[0],
        "distance": distances[0],
        "text": [metadata[i]["text"] for i in indices[0]]
    })
    return results


def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_at_k = retrieved_ids[:k]
    hits = set(retrieved_at_k).intersection(set(relevant_ids))
    return len(hits) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_at_k = retrieved_ids[:k]
    hits = set(retrieved_at_k).intersection(set(relevant_ids))
    return len(hits) / len(relevant_ids)


def hit_rate_at_k(retrieved_ids, relevant_ids, k):
    retrieved_at_k = retrieved_ids[:k]
    hits = set(retrieved_at_k).intersection(set(relevant_ids))
    return 1 if len(hits) > 0 else 0


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank
    return 0


def evaluate_retriever(retriever_name, retrieval_function, ground_truth, k=5):
    rows = []
    for query, relevant_ids in ground_truth.items():
        if not relevant_ids:
            continue
        results = retrieval_function(query, k)
        retrieved_ids = results["document_id"].tolist()
        rows.append({
            "retriever": retriever_name,
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
            f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
            f"hit_rate@{k}": hit_rate_at_k(retrieved_ids, relevant_ids, k),
            "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
        })
    return pd.DataFrame(rows)


def run_evaluation(k=5, base_dir=BASE_DIR):
    eval_df = evaluate_retriever(
        retriever_name="Embeddings (FAISS)",
        retrieval_function=lambda query, kk: retrieve_top_k_semantic(query, kk, base_dir),
        ground_truth=GROUND_TRUTH,
        k=k
    )

    summary = eval_df[[f"precision@{k}", f"recall@{k}", f"hit_rate@{k}", "reciprocal_rank"]].mean()
    print(summary)

    os.makedirs(os.path.join(base_dir, "evaluation"), exist_ok=True)
    out_path = os.path.join(base_dir, "evaluation", "retrieval_evaluation.csv")
    eval_df.to_csv(out_path, index=False)
    print(f"Saved evaluation results to {out_path}")
    return eval_df, summary


if __name__ == "__main__":
    run_evaluation()
