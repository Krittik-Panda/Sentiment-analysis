# ============================================================
# VISUALIZE.PY  —  Dataset & Model Visualisations
# ============================================================
# All plots saved to:  plots/
#
#  01_class_distribution_bar.png
#  02_class_distribution_pie.png
#  03_tweet_length_histogram.png
#  04_tweet_length_boxplot.png
#  05_top_tfidf_terms.png
#  06_tfidf_correlation_heatmap.png
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text, load_stopwords

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

PALETTE   = {"negative": "#E74C3C", "neutral": "#3498DB", "positive": "#2ECC71"}
CLASSES   = ["negative", "neutral", "positive"]
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def savefig(name: str):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✔ Saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    print("🔹 Loading data ...")
    train_df = pd.read_csv("data/train_split.csv")
    test_df  = pd.read_csv("data/test.csv")
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["text"] = df["text"].astype(str)
    df["tweet_length"] = df["text"].apply(lambda t: len(t.split()))
    return df


# ── Plot 1: Class Distribution Bar ───────────────────────────────────────────
def plot_class_bar(df):
    print("\n📊 Plot 1 : Class Distribution Bar ...")
    counts = df["sentiment"].value_counts().reindex(CLASSES)
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(CLASSES, counts.values,
                  color=[PALETTE[c] for c in CLASSES],
                  edgecolor="white", linewidth=1.2, width=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.004,
                f"{val:,}\n({val/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Sentiment Class Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Sentiment Class", fontsize=11)
    ax.set_ylabel("Number of Tweets", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.18)
    sns.despine()
    savefig("01_class_distribution_bar.png")


# ── Plot 2: Class Distribution Pie ───────────────────────────────────────────
def plot_class_pie(df):
    print("📊 Plot 2 : Class Distribution Pie ...")
    counts = df["sentiment"].value_counts().reindex(CLASSES)

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=CLASSES,
        colors=[PALETTE[c] for c in CLASSES],
        autopct="%1.1f%%",
        startangle=140,
        explode=[0.04] * 3,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title("Sentiment Class Distribution", fontsize=14, fontweight="bold", pad=12)
    savefig("02_class_distribution_pie.png")


# ── Plot 3: Tweet Length Histogram ───────────────────────────────────────────
def plot_length_histogram(df):
    print("📊 Plot 3 : Tweet Length Histogram ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        data = df[df["sentiment"] == cls]["tweet_length"]
        ax.hist(data, bins=40, color=PALETTE[cls], edgecolor="white", alpha=0.85)
        ax.axvline(data.median(), color="black", linestyle="--", linewidth=1.5,
                   label=f"Median = {data.median():.0f}")
        ax.set_title(cls.capitalize(), fontsize=12, fontweight="bold", color=PALETTE[cls])
        ax.set_xlabel("Words per Tweet")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Frequency")
    fig.suptitle("Tweet Length Distribution by Sentiment", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig("03_tweet_length_histogram.png")


# ── Plot 4: Tweet Length Box Plot ────────────────────────────────────────────
def plot_length_boxplot(df):
    print("📊 Plot 4 : Tweet Length Box Plot ...")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="sentiment", y="tweet_length", order=CLASSES,
                palette=PALETTE, width=0.45, linewidth=1.3,
                flierprops=dict(marker="o", markerfacecolor="grey",
                                markersize=3, alpha=0.4), ax=ax)
    ax.set_title("Tweet Length by Sentiment Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment Class", fontsize=11)
    ax.set_ylabel("Words per Tweet", fontsize=11)
    sns.despine()
    savefig("04_tweet_length_boxplot.png")


# ── Plot 5: Top-20 TF-IDF Terms per Class ────────────────────────────────────
def plot_top_tfidf_per_class(df, stopwords):
    print("📊 Plot 5 : Top-20 TF-IDF Terms per Class ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, cls in zip(axes, CLASSES):
        corpus = df[df["sentiment"] == cls]["text"].apply(
            lambda t: clean_text(t, stopwords)
        )
        vec        = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tmat       = vec.fit_transform(corpus)
        mean_score = np.asarray(tmat.mean(axis=0)).ravel()
        top_idx    = mean_score.argsort()[-20:][::-1]
        terms      = [vec.get_feature_names_out()[i] for i in top_idx]
        scores     = mean_score[top_idx]

        y_pos = np.arange(len(terms))
        ax.barh(y_pos, scores, color=PALETTE[cls], edgecolor="white", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"Top-20 Terms — {cls.capitalize()}",
                     fontsize=12, fontweight="bold", color=PALETTE[cls])
        ax.set_xlabel("Mean TF-IDF Score")
        sns.despine(ax=ax)

    fig.suptitle("Top-20 TF-IDF Terms per Sentiment Class",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    savefig("05_top_tfidf_terms.png")


# ── Plot 6: TF-IDF Correlation Heatmap ───────────────────────────────────────
def plot_tfidf_heatmap(df, stopwords):
    print("📊 Plot 6 : TF-IDF Correlation Heatmap (top-50 terms) ...")
    corpus  = df["text"].apply(lambda t: clean_text(t, stopwords))
    vec     = TfidfVectorizer(max_features=50)
    tmat    = vec.fit_transform(corpus).toarray()
    terms   = vec.get_feature_names_out()
    corr_df = pd.DataFrame(tmat, columns=terms).corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_df, ax=ax, cmap="coolwarm", center=0,
                linewidths=0.3, xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.7})
    ax.set_title("TF-IDF Feature Correlation (Top 50 Terms)",
                 fontsize=14, fontweight="bold", pad=12)
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7)
    plt.tight_layout()
    savefig("06_tfidf_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
def visualize():
    df        = load_data()
    stopwords = load_stopwords("stop-words-list.txt")

    print(f"\n   Total tweets : {len(df):,}")
    print(df["sentiment"].value_counts().to_string())

    plot_class_bar(df)
    plot_class_pie(df)
    plot_length_histogram(df)
    plot_length_boxplot(df)
    plot_top_tfidf_per_class(df, stopwords)
    plot_tfidf_heatmap(df, stopwords)

    print(f"\n✅ All 6 plots saved to → {PLOTS_DIR}/")

if __name__ == "__main__":
    visualize()
