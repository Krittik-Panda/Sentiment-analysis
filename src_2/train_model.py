

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import sparse
import pickle

FEATURES_DIR = "features"

def train():
    print(" Loading training features ...")
    X_train = sparse.load_npz(os.path.join(FEATURES_DIR, "X_train_sparse.npz"))
    y_train = pd.read_csv(os.path.join(FEATURES_DIR, "y_train.csv"))["sentiment"] # explicitely write sentiment to convert it into series ,,,coz read_csv return a df.

    print(f"   X_train shape : {X_train.shape}")
    print(f"   Classes       : {sorted(y_train.unique())}")

    
    print("\n Training Logistic Regression (Softmax via lbfgs) ...")
    model = LogisticRegression(
        solver="lbfgs",    
        max_iter=7000,
        C=1.0,             # L2 regularisation
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # In-sample sanity check
    train_preds = model.predict(X_train)
    print("\ Training Classification Report:")
    print(classification_report(y_train, train_preds,
                                target_names=["negative", "positive"]))

    pickle.dump(model, open("model.pkl", "wb"))
    print(" MODEL SAVED → model.pkl")
    print(f"   Classes learnt : {model.classes_.tolist()}")

if __name__ == "__main__":
    train()
