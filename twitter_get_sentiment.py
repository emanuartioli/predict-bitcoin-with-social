import numpy as np
import pandas as pd
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tweepy
import datetime

# import old tweets
tweets = pd.read_csv('C:/Users/emanu/Desktop/Thesis Project/raw_data/tweets_old.csv',
                     usecols=['user', 'date', 'time', 'text', 'replies', 'likes', 'retweets'], 
                     parse_dates={'datetime': ['date', 'time']}, infer_datetime_format=True)

# text will be translated into a sentiment score (in chunks written in a csv since it takes days and I fear crashes)
start = 0
step = 100000
for _ in range(0, int(np.ceil(len(tweets)/step))):
    nltk = [SentimentIntensityAnalyzer().polarity_scores(t)['compound'] for t in tweets.text[start:start+step]]
    textblob = [TextBlob(t).sentiment.polarity for t in tweets.text[start:start+step]]
    sentiment = pd.DataFrame({'nltk': nltk, 'textblob': textblob})
    sentiment.to_csv('C:/Users/emanu/Desktop/Thesis Project/elaboration_data/tweets_old_sentiment1.csv', 
                     header=False, index=False, mode='a')
    start += step

# add sentiment to tweets
sentiment = pd.read_csv('C:/Users/emanu/Desktop/Thesis Project/raw_data/tweets_old_sentiment.csv', header=None)
sentiment.columns = ['nltk', 'textblob']
tweets = tweets.merge(sentiment, left_index=True, right_index=True)
# TODO: sentiment score values are often 0. Maybe it's because they are technical tweets with no actual sentiment,
# but it could make sense to prune them.

# generate tweets with sentiment csv
tweets.to_csv('C:/Users/emanu/Desktop/Thesis Project/raw_data/tweets_old_with_sentiment.csv')

# new tweets are obtained via tweepy
auth = tweepy.OAuthHandler('bmsiVlL462bm9yFigRJmdPsxV', '4nZM83QAygsfpYFKLkQy2oS75zSwLaksXK8sz1dJvhZ9hvZfwj')
auth.set_access_token('1304145794149355521-fx46uLo6Z6532TOe1uQr4uLimj56jr', 'yjEb9unkcZ7JtCeihrbRHTGpDJJKVeJD49g4u2hGOHFL3')
auth = tweepy.AppAuthHandler('bmsiVlL462bm9yFigRJmdPsxV', '4nZM83QAygsfpYFKLkQy2oS75zSwLaksXK8sz1dJvhZ9hvZfwj')
api = tweepy.API(auth, wait_on_rate_limit=True)

tweets = pd.DataFrame(columns=('id', 'datetime', 'author', 'text', 'followers', 'coordinates', 'retweets', 'likes', 'language', 'replying_to'))
# avoid getting tweets we already have
stop_date = datetime.datetime(2019, 11, 1, tzinfo=datetime.timezone.utc)
# cursor retrieves tweets from the most recent, and doesn't accept time intervals
for i, tweet in enumerate(tweepy.Cursor(api.search_tweets, q='bitcoin|btc').items()):
    tweets = tweets.append({
        'id': tweet.id_str,
        'datetime': tweet.created_at,
        'author': tweet.user.id_str,
        'text': tweet.text,
        'followers': tweet.user.followers_count,
        'coordinates': tweet.coordinates,
        'retweets': tweet.retweet_count,
        'likes': tweet.favorite_count,
        'language': tweet.lang,
        'replying_to': tweet.in_reply_to_status_id_str,
        # sentiment scores are added at the same time
        'nltk': SentimentIntensityAnalyzer().polarity_scores(tweet.text)['compound'],
        'textblob': TextBlob(tweet.text).sentiment.polarity
    }, ignore_index=True)
    # every 10k tweets (to avoid slowing too much) if the stop date has been reached, stop.
    if (i % 10000 == 0):
        tweets.to_csv('C:/Users/emanu/Desktop/Thesis Project/raw_data/tweets_new_with_sentiment.csv', mode='a')
        if tweet.created_at < stop_date : break

# tweepy can't fetch more than a few thousand tweets, import recent tweets from dataset
tweets = pd.read_csv('C:/Users/emanu/Desktop/Thesis Project/raw_data/tweets_new.csv')
# columns of interest
tweets = tweets[['user_name', 'user_followers', 'user_verified', 'date', 'text', 'is_retweet']]
# not many NaNs, dropped
tweets = tweets.dropna()
# text will be translated into a sentiment score
nltk = [SentimentIntensityAnalyzer().polarity_scores(t)['compound'] for t in tweets.text]
textblob = [TextBlob(t).sentiment.polarity for t in tweets.text]
sentiment = pd.DataFrame({'nltk': nltk, 'textblob': textblob})
# add sentiment to tweets
tweets = tweets.reset_index().merge(sentiment, left_index=True, right_index=True)
# remove old index
tweets.pop('index')
# save as csv
tweets.to_csv('C:/Users/emanu/Desktop/Thesis Project/raw_data/tweets_new_with_sentiment.csv')

# unite all tweets
old = pd.read_csv('C:/Users/emanu/Desktop/Thesis Project/raw_data/tweets_old_with_sentiment.csv')
new = pd.read_csv('C:/Users/emanu/Desktop/Thesis Project/raw_data/tweets_new_with_sentiment.csv')

