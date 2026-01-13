# ------------------------------------------
# CREATE TRAIN/TEST SPLIT FROM TRAIN.CSV
# ------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

def make_split():
    print("🔹 Loading original train.csv ...")
    df = pd.read_csv("data/train.csv", encoding="latin1")

    print("🔹 Splitting 80% train + 20% test ...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv("data/train_split.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("\n✅ Done!")
    print("Created: data/train_split.csv (for training)")
    print("Created: data/test.csv (for testing)\n")

if __name__ == "__main__":
    make_split()
