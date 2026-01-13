# ------------------------------------
# BUILD FEATURES MATRIX AND LABELS
# ------------------------------------
# Covers Assignment:
# 1.4 List unique words throughout document
# 1.5 Calculate TF-IDF
# 2 Use TF-IDF as feature
# 3 Build X for training/testing
# 4 Build Y (labels) file

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text, load_stopwords
from scipy import sparse
import pickle

def build_features():
    print("🔹 Loading stopwords...")
    stopwords = load_stopwords("stop-words-list.txt")

    print("🔹 Reading dataset...")
    df = pd.read_csv("data/train.csv", encoding="latin1")
    print(f"   Loaded {len(df)} tweets.")

    print("🔹 Preprocessing tweets (cleaning + removing stopwords)...")
    df["clean"] = df["text"].apply(lambda t: clean_text(str(t), stopwords))
    print("   ✔ Text cleaning done!")

    print("🔹 Building TF-IDF vocabulary (max 10,000 words)...")
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(df["clean"])
    print("   ✔ TF-IDF matrix created!")

    print("🔹 Saving X (sparse matrix) and labels Y...")
    sparse.save_npz("X_train_sparse.npz", X)
    df["sentiment"].to_csv("y_train.csv", index=False)

    print("🔹 Saving vectorizer...")
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print("\n\n\n DONE! All features and labels created!")
    print("   ➤ X_train_sparse.npz (features)")
    print("   ➤ y_train.csv (labels)")
    print("   ➤ vectorizer.pkl (vocab builder)\n")

if __name__ == "__main__":
    build_features()
