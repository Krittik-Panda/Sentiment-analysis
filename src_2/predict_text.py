import pickle
import os
from preprocess import clean_text, load_stopwords


FEATURES_DIR = "features"
model      = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open(os.path.join(FEATURES_DIR, "vectorizer.pkl"), "rb"))
stopwords = load_stopwords("stop-words-list.txt")

def predict():
        input_text = str(input("Enter a text : "))
        clean_input_text = clean_text(input_text, stopwords) # clean 
  

        X = vectorizer.transform([clean_input_text]) # must pass a list
        #if X is not in vocabulary then all will be zero
        if X.nnz == 0:
            print("Text contains no known vocabulary.")
            return


        prediction = model.predict(X)
        probability = model.predict_proba(X)

        neg_prob = probability[0][0]
        pos_prob = probability[0][1]

        if 0.45 <= neg_prob <= 0.55 or 0.45 <= pos_prob <= 0.55:
            print("The text is neutral")
            print(f"Probability is {probability}") 
        elif 0.50 <= neg_prob < 0.60:
            print("The text is neutral to negative")
            print(f"Probability is {probability}")

        elif 0.50 <= pos_prob <= 0.60:
            print("The text is neutral to positive")
            print(f"Probability is {probability}")

        else:
            print(f"Text is {prediction[0]}")
            print(f"Probability is {probability}")


if __name__=="__main__":
    predict()
