#!/usr/bin/env python
# ============================================================
# RUN_ALL.PY  —  Full Pipeline Runner
# ============================================================
# Runs every stage in the correct order with timing info.
#
# Usage (from project root, with .venv active):
#
#   python src/run_all.py
#
# Or run individual stages:
#
#   python src/make_split.py
#   python src/build_features.py
#   python src/train_model.py
#   python src/predict_test.py
#   python src/visualize.py
#
# Expected folder layout BEFORE running:
#
#   Sentiment-analysis/
#   ├── data/
#   │   └── Twitter_Data.csv          ← raw dataset
#   ├── stop-words-list.txt           ← one stopword per line
#   └── src/
#       ├── run_all.py                ← this file
#       ├── preprocess.py
#       ├── make_split.py
#       ├── build_features.py
#       ├── train_model.py
#       ├── predict_test.py
#       └── visualize.py
#
# Outputs created automatically:
#
#   data/
#   ├── train_split.csv
#   └── test.csv
#   features/
#   ├── X_train_sparse.npz
#   ├── X_test_sparse.npz
#   ├── y_train.csv
#   ├── y_test.csv
#   └── vectorizer.pkl
#   model.pkl
#   plots/
#   ├── 01_class_distribution_bar.png
#   ├── 02_class_distribution_pie.png
#   ├── 03_tweet_length_histogram.png
#   ├── 04_tweet_length_boxplot.png
#   ├── 05_top_tfidf_terms.png
#   ├── 06_tfidf_correlation_heatmap.png
#   └── confusion_matrix.png
# ============================================================

import sys, os, time, subprocess

# ── Make sure imports resolve when run from project root ─────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

STAGES = [
    ("make_split.py",    "1/5  Split dataset"),
    ("build_features.py","2/5  Build TF-IDF features"),
    ("train_model.py",   "3/5  Train model"),
    ("predict_test.py",  "4/5  Predict & evaluate"),
    ("visualize.py",     "5/5  Visualise dataset"),
]

SEP = "=" * 60

def run_stage(script: str, label: str):
    path = os.path.join(SRC_DIR, script)
    print(f"\n{SEP}")
    print(f"  ▶  {label}")
    print(SEP)
    t0  = time.time()
    ret = subprocess.run([sys.executable, path], check=False)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"\n❌  Stage FAILED: {script}  (exit code {ret.returncode})")
        sys.exit(ret.returncode)
    print(f"\n   ⏱  Done in {elapsed:.1f}s")


def main():
    total_start = time.time()
    print(SEP)
    print("  🚀  SENTIMENT ANALYSIS PIPELINE  —  3-CLASS")
    print(SEP)

    for script, label in STAGES:
        run_stage(script, label)

    total = time.time() - total_start
    print(f"\n{SEP}")
    print(f"  ✅  ALL STAGES COMPLETE  |  Total time: {total:.1f}s")
    print(SEP)
    print("\nOutputs:")
    print("  data/train_split.csv  &  data/test.csv")
    print("  features/  (TF-IDF matrices, labels, vectorizer)")
    print("  model.pkl")
    print("  plots/     (7 PNG files)")

if __name__ == "__main__":
    main()
