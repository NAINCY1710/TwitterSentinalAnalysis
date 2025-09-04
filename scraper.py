import snscrape.modules.twitter as sntwitter
import pandas as pd

# Query: tweets about Python
query = "Python since:2023-01-01 until:2023-12-31 lang:en"

tweets = []
limit = 100   # number of tweets to fetch

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i > limit:
        break
    tweets.append([tweet.date, tweet.user.username, tweet.content])

# Convert to DataFrame
df = pd.DataFrame(tweets, columns=["Date", "User", "Content"])
print(df.head())
df.to_csv("tweets.csv", index=False)
