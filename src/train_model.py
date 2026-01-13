# -------------------------
# TRAIN LOGISTIC REGRESSION
# -------------------------
# Covers Assignment:
# 5 Feed X and Y into ML model
# 6 Train the model

import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import pickle

def train():
    print("🔹 Loading training features...")
    X_train = sparse.load_npz("X_train_sparse.npz")
    y_train = pd.read_csv("y_train.csv")["sentiment"]

    print("🔹 Training Logistic Regression...")
    model = LogisticRegression(max_iter=300, n_jobs=-1)
    model.fit(X_train, y_train)

    pickle.dump(model, open("model.pkl", "wb"))
    print("\n🎉 MODEL TRAINED AND SAVED!")

if __name__ == "__main__":
    train()
