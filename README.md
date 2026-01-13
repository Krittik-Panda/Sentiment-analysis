# 📌 Sentiment Analysis Project

**Logistic Regression + TF-IDF + Manual Preprocessing**

---

## 📖 Overview

This project performs **sentiment analysis** on tweets using:

* manual text preprocessing
* TF-IDF feature extraction
* Logistic Regression classification

The model predicts whether a tweet is:

```
positive
negative
neutral
```

The entire pipeline follows a step-by-step assignment requirement (tokenization → TF-IDF → feature matrix → model → predictions → evaluation).

---

## 📁 Project Structure

```
Sentiment-analysis/
│
├── data/
│   └── train.csv              # training dataset
│
├── stop-words-list.txt        # stopwords for preprocessing
│
├── src/
│   ├── preprocess.py          # clean and tokenize tweets
│   ├── build_features.py      # generate TF-IDF features + labels
│   ├── train_model.py         # train Logistic Regression
│   └── predict_test.py        # predict & evaluate
│
└── .venv/                     # virtual environment
```

---

## 🧠 How It Works (Simple Steps)

### **1. Preprocessing**

`preprocess.py`:

* Converts text to lowercase
* Removes punctuation and symbols
* Tokenizes text into words
* Removes stopwords in `stop-words-list.txt`

### **2. Feature Extraction**

`build_features.py`:

* Loads dataset
* Cleans all tweets
* Converts tweets into numerical features using **TF-IDF**
* Builds vocab up to 10,000 most important words
* Saves:

  * `X_train_sparse.npz` (features)
  * `y_train.csv` (labels)
  * `vectorizer.pkl` (vocab builder)

### **3. Training**

`train_model.py`:

* Loads features and labels
* Trains **Logistic Regression**
* Saves trained model as `model.pkl`

### **4. Prediction & Evaluation**

`predict_test.py`:

* Loads model, vectorizer, and original tweets
* Cleans tweets again (same rules)
* Converts cleaned tweets into feature vectors
* Predicts sentiment
* Prints first 10 tweet + prediction pairs
* Calculates overall accuracy

---

## 🛠️ Installation & Setup

### 1️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 2️⃣ Install required packages

```
pandas
numpy
scikit-learn
scipy
```

Run:

```bash
pip install pandas numpy scikit-learn scipy
```

(or create a requirements.txt if needed)

---

## ▶️ Running the Pipeline

### Step 1 — Build Features

```bash
python src/build_features.py
```

Creates:

* `X_train_sparse.npz`
* `y_train.csv`
* `vectorizer.pkl`

### Step 2 — Train Model

```bash
python src/train_model.py
```

Creates:

* `model.pkl`

### Step 3 — Predict and Evaluate

```bash
python src/predict_test.py
```

Prints examples like:

```
Tweet: "I love this!"
Predicted: positive

Accuracy: 0.78
```

---

## 📊 Output

You will see:

* cleaned tweets printed during feature build
* progress steps logged
* sample predictions printed
* overall accuracy

---

## ❗ Notes

* This project uses the **training dataset for testing** (training accuracy).
* For real evaluation, add a train/test split later.
* Sparse matrices prevent memory freezes when creating TF-IDF features.

---

## 🚀 Possible Extensions

* Split data into train + test
* Add a real test dataset
* Support live user input
* Deploy using Flask or FastAPI
* Add additional ML models (SVM, Naive Bayes, Random Forest)

---

## 🎉 Conclusion

This project demonstrates a **complete ML workflow**:
✔ data loading
✔ preprocessing
✔ feature engineering
✔ machine learning model
✔ prediction
✔ evaluation

All steps align with the textbook sentiment analysis pipeline and meet assignment requirements.

Enjoy experimenting! 😎

---

