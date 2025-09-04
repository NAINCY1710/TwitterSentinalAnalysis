import pandas as pd

# Load dataset
df = pd.read_csv("Tweets.csv")

# 1. Dataset size
print("Shape:", df.shape)

# 2. Column names
print("\nColumns:", df.columns)

# 3. First 5 rows
print("\nSample data:\n", df.head())

# 4. Null values
print("\nNull values:\n", df.isnull().sum())

# 5. Sentiment distribution (if sentiment column exists)
if "sentiment" in df.columns:
    print("\nSentiment distribution:\n", df["sentiment"].value_counts())
