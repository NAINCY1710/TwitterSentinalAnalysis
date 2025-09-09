# twitter_api_predict.py
import tweepy
import joblib
import re
import streamlit as st
import os
from dotenv import load_dotenv

# ----------------------------
# 1. Load Twitter API credentials
# ----------------------------
load_dotenv()
bearer_token = os.getenv("BEARER_TOKEN")
client = tweepy.Client(bearer_token=bearer_token)

# ----------------------------
# 2. Load saved model & vectorizer
# ----------------------------
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

# ----------------------------
# 3. Tweet cleaning function
# ----------------------------
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"#", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    return text

# ----------------------------
# 4. Function to predict from hashtag
# ----------------------------
def predict_from_hashtag(hashtag, max_results=10):
    query = f"#{hashtag} -is:retweet lang:en"
    tweets = client.search_recent_tweets(query=query, max_results=max_results)

    if not tweets.data:
        st.warning(f"No tweets found for #{hashtag}")
        return

    # Clean tweets
    cleaned_tweets = [clean_tweet(tweet.text) for tweet in tweets.data]

    # Transform and predict
    X_new = vectorizer.transform(cleaned_tweets)
    predictions = model.predict(X_new)

    # Display results
    st.subheader(f"Sentiment Predictions for #{hashtag}")
    results = []
    for tweet, sentiment in zip(tweets.data, predictions):
        results.append({"tweet": tweet.text, "predicted_sentiment": sentiment})
    df = st.dataframe(results)

    # Optional: download
    import pandas as pd
    df_csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", df_csv, f"{hashtag}_predictions.csv", "text/csv")
