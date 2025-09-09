# twitter_api_predict.py
import tweepy
import joblib
import re

# ----------------------------
# 1. Twitter API credentials
# ----------------------------
import os
from dotenv import load_dotenv
load_dotenv()
bearer_token = os.getenv("BEARER_TOKEN")  # <-- replace with your token

# ----------------------------
# 2. Connect to Twitter API
# ----------------------------
client = tweepy.Client(bearer_token=bearer_token)

# ----------------------------
# 3. Define tweet cleaning function
# ----------------------------
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove links
    text = re.sub(r"@\w+", '', text)                     # remove mentions
    text = re.sub(r"#", '', text)                        # remove hashtags symbol
    text = re.sub(r"[^\w\s]", '', text)                 # remove punctuation
    return text

# ----------------------------
# 4. Load saved model & vectorizer
# ----------------------------
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

# ----------------------------
# 5. Fetch tweets from Twitter
# ----------------------------
hashtag = input("Enter hashtag to search: ")  # user input
query = f"#{hashtag} -is:retweet lang:en"

tweets = client.search_recent_tweets(query=query, max_results=10)

if not tweets.data:
    print("No tweets found for this hashtag.")
else:
    # ----------------------------
    # 6. Preprocess tweets
    # ----------------------------
    cleaned_tweets = [clean_tweet(tweet.text) for tweet in tweets.data]

    # ----------------------------
    # 7. Transform and predict
    # ----------------------------
    X_new = vectorizer.transform(cleaned_tweets)
    predictions = model.predict(X_new)

    # ----------------------------
    # 8. Display tweets with predictions
    # ----------------------------
    print("\nSentiment Predictions for live tweets:\n")
    for tweet, sentiment in zip(tweets.data, predictions):
        print(f"{sentiment.upper()}: {tweet.text}\n")
