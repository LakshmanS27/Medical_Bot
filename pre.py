import pandas as pd
import spacy
import string
import os

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in string.punctuation and not token.is_stop]
    return " ".join(tokens)

def preprocess_dataset(input_csv="dataset.csv", output_csv="dataset_cleaned.csv"):
    df = pd.read_csv(input_csv)

    if "prompt" not in df.columns or "response" not in df.columns:
        raise ValueError("Missing 'prompt' or 'response' columns.")

    df["prompt"] = df["prompt"].astype(str).apply(preprocess_text)
    df["response"] = df["response"].astype(str).apply(preprocess_text)

    df.to_csv(output_csv, index=False)
    print(f"✅ Saved cleaned dataset to {output_csv}")

if __name__ == "__main__":
    preprocess_dataset()
