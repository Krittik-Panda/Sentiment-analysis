# -------------------------
# TRAIN ML MODEL
# -------------------------
# Covers Assignment:
# 5 Feed X and Y into ML model
# 6 Train the model

import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import pickle

def train():
    # Load features + labels
    X = sparse.load_npz("X_train_sparse.npz")
    y = pd.read_csv("y_train.csv")["sentiment"].values

    # Train Logistic Regression (Step 6)
    model = LogisticRegression(max_iter=300)
    model.fit(X, y)

    # Save trained model
    pickle.dump(model, open("model.pkl", "wb"))

    print("Model trained and saved!")

if __name__ == "__main__":
    train()
