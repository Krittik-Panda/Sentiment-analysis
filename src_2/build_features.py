# ------------------------------------
# BUILD TF-IDF FEATURES AND LABEL FILES
# ------------------------------------
# Covers Assignment:
# 1.4 List unique words throughout document
# 1.5 Calculate TF-IDF
# 2   Use TF-IDF as feature
# 3   Build X for training/testing
# 4   Build Y (labels) file
#
# All artefacts (matrices, labels, vectoriser) saved to:  features/

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text, load_stopwords
from scipy import sparse
import pickle

FEATURES_DIR = "features"
os.makedirs(FEATURES_DIR, exist_ok=True)

def build_features():
    print("🔹 Reading split CSVs ...")
    train_df = pd.read_csv("data/train_split.csv")
    test_df  = pd.read_csv("data/test.csv")

    stopwords = load_stopwords("stop-words-list.txt")

    print("🔹 Cleaning training tweets ...")
    train_clean = train_df["text"].apply(lambda t: clean_text(str(t), stopwords))

    print("🔹 Cleaning test tweets ...")
    test_clean  = test_df["text"].apply(lambda t: clean_text(str(t), stopwords))

    # 1.4 Unique vocabulary
    all_tokens = set(tok for doc in train_clean for tok in doc.split())
    print(f"\n📌 Unique words in training corpus : {len(all_tokens):,}")

    # 1.5 TF-IDF vectorisation
    print("\n🔹 Creating TF-IDF vectoriser (max 10,000 features, unigrams+bigrams) ...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True     # log(1+tf) — normalises long tweets
    )

    X_train = vectorizer.fit_transform(train_clean)
    X_test  = vectorizer.transform(test_clean)

    # ── Save all artefacts to  features/  ───────────────────────────────────
    print(f"\n🔹 Saving artefacts to  {FEATURES_DIR}/  ...")

    sparse.save_npz(os.path.join(FEATURES_DIR, "X_train_sparse.npz"), X_train)
    sparse.save_npz(os.path.join(FEATURES_DIR, "X_test_sparse.npz"),  X_test)

    train_df["sentiment"].to_csv(os.path.join(FEATURES_DIR, "y_train.csv"), index=False)
    test_df["sentiment"].to_csv( os.path.join(FEATURES_DIR, "y_test.csv"),  index=False)

    pickle.dump(vectorizer, open(os.path.join(FEATURES_DIR, "vectorizer.pkl"), "wb"))

    print("\n✅ DONE BUILDING FEATURES!")
    print(f"   ✔ X_train : {X_train.shape}")
    print(f"   ✔ X_test  : {X_test.shape}")
    print(f"   ✔ Vocab   : {len(vectorizer.vocabulary_):,} terms")
    print(f"\n📊 Training label distribution:")
    print(train_df["sentiment"].value_counts().to_string())
    print(f"\n📊 Test label distribution:")
    print(test_df["sentiment"].value_counts().to_string())

if __name__ == "__main__":
    build_features()
