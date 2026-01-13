# ------------------------------------
# BUILD TF-IDF FEATURES AND LABEL FILES
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
    print("🔹 Reading split CSVs ...")
    train_df = pd.read_csv("data/train_split.csv")
    test_df = pd.read_csv("data/test.csv")

    stopwords = load_stopwords("stop-words-list.txt")

    print("🔹 Cleaning training tweets...")
    train_clean = train_df["text"].apply(lambda t: clean_text(str(t), stopwords))

    print("🔹 Cleaning testing tweets...")
    test_clean = test_df["text"].apply(lambda t: clean_text(str(t), stopwords))

    print("🔹 Creating TF-IDF with max 10,000 features...")
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(train_clean)
    X_test = vectorizer.transform(test_clean)

    print("🔹 Saving sparse TF-IDF matrices...")
    sparse.save_npz("X_train_sparse.npz", X_train)
    sparse.save_npz("X_test_sparse.npz", X_test)

    print("🔹 Saving label files...")
    train_df["sentiment"].to_csv("y_train.csv", index=False)
    test_df["sentiment"].to_csv("y_test.csv", index=False)

    print("🔹 Saving vectorizer...")
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print("\n🎉 DONE BUILDING FEATURES!")
    print(f"✔ X_train_sparse.npz shape: {X_train.shape}")
    print(f"✔ X_test_sparse.npz shape:  {X_test.shape}")

if __name__ == "__main__":
    build_features()
