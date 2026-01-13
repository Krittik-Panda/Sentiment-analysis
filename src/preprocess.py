# ---------------------
# PREPROCESSING MODULE
# ---------------------
# Covers Assignment Step 1:
# 1.1 Tokenization
# 1.2 Remove special characters & stop words
# 1.3 Return cleaned tweets

import re

def load_stopwords(path):
    with open(path, "r") as f:
        return set(word.strip() for word in f.readlines())

def clean_text(text, stopwords):
    # Lowercase
    text = text.lower()

    # Remove special characters & numbers (1.2)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenization (1.1)
    tokens = text.split()

    # Remove stopwords (1.2)
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]

    # Return cleaned tweet (1.3)
    return " ".join(tokens)
