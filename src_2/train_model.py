# -------------------------------------------
# TRAIN LOGISTIC REGRESSION  (3-CLASS / SOFTMAX)
# -------------------------------------------
# Covers Assignment:
# 5  Feed X and Y into ML model
# 6  Train the model
#
# FIX: `multi_class` was removed in scikit-learn 1.5+.
#      solver='lbfgs' on a multi-class problem automatically
#      uses softmax (multinomial cross-entropy) — no extra flag needed.

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import sparse
import pickle

FEATURES_DIR = "features"

def train():
    print("🔹 Loading training features ...")
    X_train = sparse.load_npz(os.path.join(FEATURES_DIR, "X_train_sparse.npz"))
    y_train = pd.read_csv(os.path.join(FEATURES_DIR, "y_train.csv"))["sentiment"]

    print(f"   X_train shape : {X_train.shape}")
    print(f"   Classes       : {sorted(y_train.unique())}")

    # lbfgs + 3 classes → softmax (multinomial) loss, sklearn 1.5+ compatible
    print("\n🔹 Training Logistic Regression (Softmax via lbfgs) ...")
    model = LogisticRegression(
        solver="lbfgs",    # softmax output for multi-class automatically
        max_iter=500,
        C=1.0,             # L2 regularisation
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # In-sample sanity check
    train_preds = model.predict(X_train)
    print("\n📊 Training Classification Report:")
    print(classification_report(y_train, train_preds,
                                target_names=["negative", "neutral", "positive"]))

    pickle.dump(model, open("model.pkl", "wb"))
    print("✅ MODEL SAVED → model.pkl")
    print(f"   Classes learnt : {model.classes_.tolist()}")

if __name__ == "__main__":
    train()
