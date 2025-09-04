import joblib

# Load the vectorizer and model
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

print("🚀 Twitter Sentiment Analyzer Ready!")
print("Type 'exit' to quit.\n")

while True:
    tweet = input("Enter a tweet: ")

    if tweet.lower() == "exit":
        print("Goodbye! 👋")
        break

    X_new = vectorizer.transform([tweet])
    prediction = model.predict(X_new)[0]

    print(f"Predicted Sentiment: {prediction}")
    print("-" * 40)
