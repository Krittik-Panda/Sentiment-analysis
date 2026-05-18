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
    print("training_1600000_processed_noemoticon.csv")
    df = pd.read_csv("data/training_1600000_processed_noemoticon.csv", encoding="latin-1")

    assert "text of the tweet" in df.columns and "polarity of tweet" in df.columns, \
        "CSV must have 'text of the tweet' and 'polarity of tweet' columns."

    # Drop missing rows
    before = len(df)
    df = df.dropna(subset=["text of the tweet", "polarity of tweet"])
    print(f"   Dropped {before - len(df)} rows with NaN values.")



    # Map numeric labels → string labels
    df["polarity of tweet"] = df["polarity of tweet"].astype(float).astype(int)
    df["sentiment"] = df["polarity of tweet"].map(LABEL_MAP) # .map() is a  panda series function here this  df["polarity of tweet"] is a single column or series .
    df = df.dropna(subset=["sentiment"])



    df = df[["text of the tweet", "sentiment"]].rename(columns={"text of the tweet": "text"})  # keep only necessery column

    print("\n Class distribution (full dataset):")
    print(df["sentiment"].value_counts().to_string())

    # Stratified 80/20 split
    print("\n Splitting data (80% TRAIN / 20% TEST) with stratification ...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    train_df.to_csv("data/train_split.csv", index=False)
    test_df.to_csv("data/test.csv",         index=False)

    print(f"\n  DONE!  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print("    data/train_split.csv")
    print("    data/test.csv")

if __name__ == "__main__":
    make_split()
