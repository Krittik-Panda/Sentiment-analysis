# -----------------------------------------------
# PREDICT & EVALUATE ON TEST SET (3-CLASS)
# -----------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)
import pickle

FEATURES_DIR = "features"
PLOTS_DIR    = "plots"
CLASSES      = ["negative", "neutral", "positive"]
os.makedirs(PLOTS_DIR, exist_ok=True)

def predict():
    print("🔹 Loading model & vectoriser ...")
    model      = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open(os.path.join(FEATURES_DIR, "vectorizer.pkl"), "rb"))

    print("🔹 Loading test set ...")
    X_test  = sparse.load_npz(os.path.join(FEATURES_DIR, "X_test_sparse.npz"))
    y_test  = pd.read_csv(os.path.join(FEATURES_DIR, "y_test.csv"))["sentiment"]
    test_df = pd.read_csv("data/test.csv")

    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)   # softmax probabilities (n, 3)

    # ── Sample predictions ───────────────────────────────────────────────────
    print("\n--- SAMPLE PREDICTIONS (first 10) ---")
    for i in range(10):
        text = str(test_df["text"].iloc[i])
        prob = probas[i]
        print(f"\n📝 {text[:90]}{'...' if len(text) > 90 else ''}")
        print(f"   Softmax  → neg:{prob[0]:.3f}  neu:{prob[1]:.3f}  pos:{prob[2]:.3f}")
        print(f"   PREDICTED: {preds[i]:<10}  TRUE: {y_test.iloc[i]}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc = accuracy_score(y_test, preds)
    print(f"\n{'='*55}")
    print(f"  TEST ACCURACY : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'='*55}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds, target_names=CLASSES))

    # ── Confusion Matrix → saved to plots/ ──────────────────────────────────
    cm   = confusion_matrix(y_test, preds, labels=CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix — 3-Class Sentiment", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"\n✅ Confusion matrix saved → {out}")

if __name__ == "__main__":
    predict()
