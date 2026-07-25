"""
03_chunking.py
---------------
Step 3: Split cleaned documents into overlapping chunks and save them.
Running this file end-to-end performs steps 1 -> 2 -> 3.
"""

import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from importlib import import_module

documents_mod = import_module("01_documents")
preprocessing_mod = import_module("02_preprocessing")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def chunk_documents(df):
    chunks = []
    for _, row in df.iterrows():
        split_texts = text_splitter.split_text(row["clean_text"])
        for chunk in split_texts:
            chunks.append({
                "text": chunk,
                "source": row["source"],
                "file_name": row["file_name"]
            })
    chunks_df = pd.DataFrame(chunks)
    return chunks_df


def build_chunks_csv(base_dir=BASE_DIR):
    df = documents_mod.load_documents(base_dir)
    df = preprocessing_mod.clean_documents(df)
    chunks_df = chunk_documents(df)

    os.makedirs(os.path.join(base_dir, "processed_data"), exist_ok=True)
    out_path = os.path.join(base_dir, "processed_data", "chunks.csv")
    chunks_df.to_csv(out_path, index=False)
    print(f"Saved {len(chunks_df)} chunks to {out_path}")
    return chunks_df


if __name__ == "__main__":
    build_chunks_csv()
