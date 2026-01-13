# -----------------------------------------------
# PREDICT & COMPARE WITH TRUE TEST LABELS
# -----------------------------------------------

import pandas as pd
from scipy import sparse
import pickle

def predict():
    print("🔹 Loading model & vectorizer...")
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    print("🔹 Loading test set...")
    X_test = sparse.load_npz("X_test_sparse.npz")
    y_test = pd.read_csv("y_test.csv")["sentiment"]

    test_df = pd.read_csv("data/test.csv")

    print("\n--- SAMPLE PREDICTIONS ---")
    preds = model.predict(X_test)

    for text, pred, true in zip(test_df["text"][:10], preds[:10], y_test[:10]):
        print(f"\n{text}")
        print(f"PREDICTED: {pred}")
        print(f"TRUE:      {true}")

    accuracy = (preds == y_test).mean()
    print(f"\n🎯 TEST ACCURACY: {accuracy:.4f}")

if __name__ == "__main__":
    predict()
