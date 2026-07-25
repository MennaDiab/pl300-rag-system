"""
01_documents.py
----------------
Step 1: Load raw PDF documents (Microsoft Learn + exam dumps) into a DataFrame.
"""

import os
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def extract_pdf_text(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}")
        print(e)
    return text


def load_documents(base_dir=BASE_DIR):
    documents = []

    microsoft_folder = os.path.join(base_dir, "data", "microsoft_learn")
    for file in tqdm(os.listdir(microsoft_folder), desc="Microsoft Learn PDFs"):
        if file.endswith(".pdf"):
            path = os.path.join(microsoft_folder, file)
            text = extract_pdf_text(path)
            documents.append({"text": text, "source": "microsoft", "file_name": file})

    dump_folder = os.path.join(base_dir, "data", "dumps")
    for file in tqdm(os.listdir(dump_folder), desc="Exam dump PDFs"):
        if file.endswith(".pdf"):
            path = os.path.join(dump_folder, file)
            text = extract_pdf_text(path)
            documents.append({"text": text, "source": "dump", "file_name": file})

    df = pd.DataFrame(documents)
    print(f"Total documents loaded: {len(df)}")
    if not df.empty:
        print(df["source"].value_counts())
    return df


if __name__ == "__main__":
    df = load_documents()
    print(df.head())
