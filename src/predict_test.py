# -----------------------------------------------
# PRINT TWEET + PREDICTED SENTIMENT + ACCURACY
# -----------------------------------------------

import pandas as pd
from scipy import sparse
import pickle
from preprocess import clean_text, load_stopwords

def predict():
    # Load model + vectorizer
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    stopwords = load_stopwords("stop-words-list.txt")

    # Load original tweets & labels
    df = pd.read_csv("data/train.csv", encoding="latin1")

    # Clean tweets again using SAME logic
    df["clean"] = df["text"].apply(lambda t: clean_text(str(t), stopwords))

    # Transform using saved vectorizer
    X = vectorizer.transform(df["clean"])
    y = df["sentiment"].values

    # Predict
    preds = model.predict(X)

    # Print some tweet + prediction pairs
    print("\n--- SAMPLE PREDICTIONS ---")
    for text, pred in zip(df["text"][:10], preds[:10]):
        print(f"{text}  --->  {pred}")

    # Accuracy
    accuracy = (preds == y).mean()
    print("\nAccuracy:", accuracy)

if __name__ == "__main__":
    predict()
