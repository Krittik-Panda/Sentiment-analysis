# ------------------------------------------
# CREATE TRAIN/TEST SPLIT FROM FULL DATASET
# ------------------------------------------
# Dataset : data/Twitter_Data.csv
# Columns : clean_text | category
# Classes : -1 → negative | 0 → neutral | 1 → positive

import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import LABEL_MAP
import os

os.makedirs("data", exist_ok=True)

def make_split():
    print("🔹 Loading Twitter_Data.csv ...")
    df = pd.read_csv("data/Twitter_Data.csv", encoding="utf-8")

    assert "clean_text" in df.columns and "category" in df.columns, \
        "CSV must have 'clean_text' and 'category' columns."

    # Drop missing rows
    before = len(df)
    df = df.dropna(subset=["clean_text", "category"])
    print(f"   Dropped {before - len(df)} rows with NaN values.")

    # Map numeric labels → string labels
    df["category"] = df["category"].astype(float).astype(int)
    df["sentiment"] = df["category"].map(LABEL_MAP)
    df = df.dropna(subset=["sentiment"])

    df = df[["clean_text", "sentiment"]].rename(columns={"clean_text": "text"})

    print("\n📊 Class distribution (full dataset):")
    print(df["sentiment"].value_counts().to_string())

    # Stratified 80/20 split
    print("\n🔹 Splitting data (80% TRAIN / 20% TEST) with stratification ...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    train_df.to_csv("data/train_split.csv", index=False)
    test_df.to_csv("data/test.csv",         index=False)

    print(f"\n✅ DONE!  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print("   ✔ data/train_split.csv")
    print("   ✔ data/test.csv")

if __name__ == "__main__":
    make_split()
