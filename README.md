# Review Sentiment Analyzer

A machine learning web application that analyzes movie reviews and predicts whether the sentiment is **positive** or **negative**.

## Features

- Sentiment classification using Machine Learning
- TF-IDF text feature extraction
- Logistic Regression classification
- Confidence score for predictions
- Simple Flask web interface
- Runs locally on your computer

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- Flask
- HTML and CSS

## Dataset

The model was trained using the **IMDb Dataset of 50K Movie Reviews**.

The dataset contains:

- 50,000 movie reviews
- 25,000 positive reviews
- 25,000 negative reviews

The dataset is not included in this GitHub repository because of its size.

To retrain the model, place the dataset in:

```text
data/reviews.csv