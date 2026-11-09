from flask import Flask, render_template,request
import joblib
app=Flask(__name__)
model=joblib.load("model/logistic_model.pkl")
tfidf=joblib.load("model/tfidf_vectorizer.pkl")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    job_description = request.form["job_description"].strip()

    if len(job_description) < 50:
        return render_template(
            "index.html",
            error="Please enter a job description with at least 50 characters."
        )

    transformed_text = tfidf.transform([job_description])
    prediction = int(model.predict(transformed_text)[0])
    confidence=float(model.predict_proba(transformed_text)[0][prediction]*100)
    [prediction]*100

    if prediction == 1:
        result = "Fraudulent Job Posting"
    else:
        result = "Genuine Job Posting"

    return render_template("index.html", prediction=result, confidence=round(confidence,2))
if __name__=="__main__":
    app.run(debug=True)
