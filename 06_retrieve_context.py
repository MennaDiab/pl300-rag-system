"""
06_retrieve_context.py
------------------------
Step 6: Load the persisted FAISS index + metadata, and expose a retrieve()
function used by both evaluation.py and streamlit_app.py.
"""

import os
import pickle
import faiss

from importlib import import_module
vector_repr_mod = import_module("04_vector_representation")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

_index = None
_metadata = None
_embedding_model = None


def load_index_and_metadata(base_dir=BASE_DIR):
    global _index, _metadata, _embedding_model
    if _index is None:
        _index = faiss.read_index(os.path.join(base_dir, "vector_db", "pl300.index"))
        with open(os.path.join(base_dir, "vector_db", "metadata.pkl"), "rb") as f:
            _metadata = pickle.load(f)
        _embedding_model = vector_repr_mod.load_embedding_model()
    return _index, _metadata, _embedding_model


def retrieve(query, k=5, base_dir=BASE_DIR):
    index, metadata, embedding_model = load_index_and_metadata(base_dir)
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding.astype("float32"), k)
    results = [metadata[idx]["text"] for idx in indices[0]]
    return results


if __name__ == "__main__":
    for chunk in retrieve("What is DAX?", k=3):
        print("=" * 80)
        print(chunk[:300])
