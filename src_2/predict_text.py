import pickle
import os
from preprocess import clean_text, load_stopwords


FEATURES_DIR = "features"
model      = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open(os.path.join(FEATURES_DIR, "vectorizer.pkl"), "rb"))
stopwords = load_stopwords("stop-words-list.txt")

def predict():
        input_text = str(input("Enter a text : "))
        clean_input_text = clean_text(input_text, stopwords)
        # check if the text is empty after clean or not
        if clean_input_text.strip() == "":
            return "Text contains no known vocabulary."

        X = vectorizer.transform([clean_input_text]) # must pass a list
        prediction = model.predict(X)
        probability = model.predict_proba(X)

        print(f"Text is {prediction[0]}")
        print(f"probability is {probability}")



if __name__=="__main__":
    predict()
