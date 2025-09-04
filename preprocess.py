import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only needed first time)
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("Tweets.csv")

# Initialize tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Cleaning function
def clean_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)                      # remove mentions
    text = re.sub(r"#", "", text)                         # remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)               # remove numbers/punctuations
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Apply cleaning
df["cleaned_text"] = df["text"].apply(clean_tweet)

# Keep only useful columns
df = df[["text", "cleaned_text", "airline_sentiment"]]

# Show preview
print(df.head())

# Save cleaned dataset
df.to_csv("Cleaned_Tweets.csv", index=False)
print("\n✅ Cleaned dataset saved as Cleaned_Tweets.csv")
