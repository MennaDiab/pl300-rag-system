"""
04_vector_representation.py
-----------------------------
Step 4: Turn text chunks into embedding vectors using SentenceTransformers.
"""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model, texts, show_progress_bar=True):
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
    return embeddings


if __name__ == "__main__":
    model = load_embedding_model()
    vectors = embed_texts(model, ["What is DAX?", "What is a calculated column?"])
    print(vectors.shape)
