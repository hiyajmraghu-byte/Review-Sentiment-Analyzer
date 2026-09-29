import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
data = pd.read_csv("data/reviews.csv")

# Display dataset information
print("Dataset loaded successfully!")
print("Total reviews:", len(data))
print("\nSentiment distribution:")
print(data["sentiment"].value_counts())

# Convert sentiment labels to numbers
data["sentiment"] = data["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data["review"],
    data["sentiment"],
    test_size=0.20,
    random_state=42,
    stratify=data["sentiment"]
)

print("\nTraining reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# Convert text into numerical TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
print("\nTraining model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("Training complete.")

# Test model
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model and vectorizer
joblib.dump(model, "model/sentiment_model.joblib")
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")

print("\nModel saved successfully as:")
print("model/sentiment_model.joblib")

print("Vectorizer saved successfully as:")
print("model/tfidf_vectorizer.joblib")