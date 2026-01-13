# -----------------------------------------------
# COMPARE PREDICTIONS WITH TRUE TEST LABELS
# -----------------------------------------------

import pandas as pd
from scipy import sparse
import pickle
from preprocess import clean_text, load_stopwords

def predict():
    print("🔹 Loading model and vectorizer...")
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    stopwords = load_stopwords("stop-words-list.txt")

    print("🔹 Loading test data...")
    df = pd.read_csv("data/test.csv", encoding="latin1")

    # Clean using SAME preprocessing
    df["clean"] = df["text"].apply(lambda t: clean_text(str(t), stopwords))

    print("🔹 Converting test tweets to TF-IDF features...")
    X_test = vectorizer.transform(df["clean"])
    y_test = df["sentiment"].values

    print("\n--- SAMPLE COMPARISON (Tweet | Prediction | True Label) ---")
    preds = model.predict(X_test)

    for text, pred, true in zip(df["text"][:10], preds[:10], y_test[:10]):
        print(f"\nTWEET: {text}")
        print(f"PREDICTED: {pred}")
        print(f"TRUE LABEL: {true}")

    # Calculate accuracy
    accuracy = (preds == y_test).mean()
    print("\n🎯 TEST ACCURACY:", accuracy)

if __name__ == "__main__":
    predict()
