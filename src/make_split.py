# ------------------------------------------
# CREATE TRAIN/TEST SPLIT FROM FULL DATASET
# ------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

def make_split():
    print("🔹 Loading original train.csv ...")
    df = pd.read_csv("data/train.csv", encoding="latin1", header=None)

    # Drop fake header row
    df = df.iloc[1:]

    # Set correct column names
    df.columns = ["sentiment", "id", "date", "query", "user", "text"]

    # Convert sentiment codes to words
    df["sentiment"] = df["sentiment"].astype(int).map({
        0: "negative",
        2: "neutral",
        4: "positive"
    })

    # Keep only useful columns
    df = df[["text", "sentiment"]]

    print("🔹 Splitting data (80% TRAIN / 20% TEST)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv("data/train_split.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("\n✅ DONE!")
    print("✔ data/train_split.csv created")
    print("✔ data/test.csv created")

if __name__ == "__main__":
    make_split()
