"""
05_create_index_store.py
--------------------------
Step 5: Build the FAISS vector index from processed_data/chunks.csv
and persist it (plus metadata) to vector_db/.

This is the FAISS equivalent of a "create_chroma_store" step.
Run this once (and again any time chunks.csv changes).
"""

import os
import pickle
import pandas as pd
import faiss

from importlib import import_module
vector_repr_mod = import_module("04_vector_representation")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def build_index(base_dir=BASE_DIR):
    chunks_path = os.path.join(base_dir, "processed_data", "chunks.csv")
    chunks_df = pd.read_csv(chunks_path)

    model = vector_repr_mod.load_embedding_model()
    texts = chunks_df["text"].tolist()
    embeddings = vector_repr_mod.embed_texts(model, texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    os.makedirs(os.path.join(base_dir, "vector_db"), exist_ok=True)
    faiss.write_index(index, os.path.join(base_dir, "vector_db", "pl300.index"))

    metadata = chunks_df.to_dict(orient="records")
    with open(os.path.join(base_dir, "vector_db", "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index built with {index.ntotal} vectors and saved to vector_db/")
    return index, metadata


if __name__ == "__main__":
    build_index()
