from flask import Flask , render_template , url_for , redirect
import pickle
import os
from preprocess import clean_text, load_stopwords
#from predict_text import predict


app = Flask(__name__)


FEATURES_DIR = "features"
model      = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open(os.path.join(FEATURES_DIR, "vectorizer.pkl"), "rb"))
stopwords = load_stopwords("stop-words-list.txt")

@app.route("/", methods = ["GET"])
def welcome():
    return render_template("index.html")

@app.route("/result/<string:label>")  
def result():
    return render_template("result.html")  

@app.route("/predict/", methods = ["POST"])
def predict():
        text = str(request.index(["tweet"]))
        clean_input_text = clean_text(input_text, stopwords) # clean 
        X = vectorizer.transform([clean_input_text])
        if X.nnz == 0:
            print("Text contains no known vocabulary.")
            return


        prediction = model.predict(X)
        probability = model.predict_proba(X)

        neg_prob = probability[0][0]
        pos_prob = probability[0][1]

        if 0.45 <= neg_prob <= 0.55 or 0.45 <= pos_prob <= 0.55:
            label = "The text is neutral"
            return redirect(url_for('result' , label = label))
            #print(f"Probability is {probability}") 
        elif 0.50 <= neg_prob < 0.60:
            label = "The text is neutral to negative"
            return redirect(url_for('result' , label = label))
            #print(f"Probability is {probability}")

        elif 0.50 <= pos_prob <= 0.60:
            label = "The text is neutral to positive"
            return redirect(url_for('result' , label = label))
            #print(f"Probability is {probability}")

        else:
            label = f"Text is {prediction[0]}"
            return redirect(url_for('result' , label = label))
            #print(f"Probability is {probability}")







if __name__ == "__main__":
    app.run(debug= True)
