from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model/sentiment_model.joblib")
vectorizer = joblib.load("model/tfidf_vectorizer.joblib")


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    confidence = None
    review = ""

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if review:
            # Convert review into TF-IDF features
            review_vector = vectorizer.transform([review])

            # Predict sentiment
            prediction = model.predict(review_vector)[0]

            # Get confidence
            probabilities = model.predict_proba(review_vector)[0]
            confidence = round(max(probabilities) * 100, 2)

            if prediction == 1:
                sentiment = "Positive"
            else:
                sentiment = "Negative"

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        review=review
    )


if __name__ == "__main__":
    app.run(debug=True)