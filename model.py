import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load the preprocessed dataset
data = pd.read_csv("Cleaned_Tweets.csv")

# 2. Features (X) = text, Labels (y) = sentiment
X = data['cleaned_text']
y = data['airline_sentiment']
# remove rows where cleaned_text is NaN
data = data.dropna(subset=['cleaned_text'])

# just to be safe, convert everything to string
X = data['cleaned_text'].astype(str)
y = data['airline_sentiment']


# 3. Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # use top 5000 words
X_tfidf = vectorizer.fit_transform(X)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# 5. Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# save the TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# save the trained model
joblib.dump(model, "sentiment_model.pkl")
