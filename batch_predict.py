import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Load vectorizer and model
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

# Load test tweets with true labels
data = pd.read_csv("Cleaned_Tweets.csv")  # make sure this is test set
tweets = data["text"]
true_labels = data["airline_sentiment"]

# Transform tweets
X_new = vectorizer.transform(tweets)

# Predict
predictions = model.predict(X_new)

# Compare with true labels
print("Accuracy:", accuracy_score(true_labels, predictions))
print("\nClassification Report:\n", classification_report(true_labels, predictions))

# Optional: save predictions
results = pd.DataFrame({
    "tweet": tweets,
    "true_sentiment": true_labels,
    "predicted_sentiment": predictions
})
# results.to_csv("Predicted_Tweets_with_labels.csv", index=False)
