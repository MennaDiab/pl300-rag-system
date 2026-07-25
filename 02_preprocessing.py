"""
02_preprocessing.py
--------------------
Step 2: Clean the raw extracted text (lowercase, whitespace normalization).
"""

import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_documents(df):
    df["clean_text"] = df["text"].apply(clean_text)
    return df


if __name__ == "__main__":
    # quick manual test
    sample = "  This   is\n\na   MESSY   PDF   text.  "
    print(clean_text(sample))
