import numpy as np
import pandas as pd
import datetime as dt
from pandas.core.arrays.string_ import StringDtype
from pandas.core.tools.datetimes import to_datetime
import praw
from psaw import PushshiftAPI
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# submissions are obtained with pushshift api

# reddit connection
reddit = praw.Reddit(
    client_id="Rq6K6sP3qJy9Eg",
    client_secret="wOCW30BdZdt1R5fGrOvTuBAXZ6M0Vw",
    user_agent="bitcoin-prediction-sentiment by u/emanuartioli",
    username="emanuartioli"
)

# get submissions
api = PushshiftAPI(reddit)
submissions = list(api.search_submissions(subreddit='Bitcoin'))

# reddit has a forest of nested comments, therefore a recursive function evaluates the sentiment for every comment
# and returns the average weighted on the score of the comment
def forest_explorer(submission):

    nltk_sentiment = list()
    textblob_sentiment = list()
    scores = list()

    # sentiment of the submission
    nltk_sentiment.append(SentimentIntensityAnalyzer().polarity_scores(submission.selftext)['compound'])
    textblob_sentiment.append(TextBlob(submission.selftext).sentiment.polarity)
    scores.append(submission.score)

    # recursion for sentiment of every comment
    def comment_explorer(comment):
        nltk_sentiment.append(SentimentIntensityAnalyzer().polarity_scores(comment.selftext)['compound'])
        textblob_sentiment.append(TextBlob(comment.selftext).sentiment.polarity)
        scores.append(comment.score)
        for reply in comment.replies:
            comment_explorer(reply)
        return

    # submission might not have comments, and comments might not have replies, both throw an AttributeError
    try:
        for comment in submission._comments:
            comment_explorer(comment)
    except AttributeError:
        pass

    # Once the forest has been evaluated, compute the weighted average
    nltk_score = np.ma.average(nltk_sentiment, weights=scores)
    textblob_score = np.ma.average(textblob_sentiment, weights=scores)

    # we need some more info from the submission
    # when the author of a post deletes its account, author is None
    try:
        author = submission.author.name
    except AttributeError:
        author = 'deleted'
    datetime = str(dt.datetime.fromtimestamp(submission.created))
    downvotes = submission.downs
    id = submission.id
    body = submission.selftext
    title = submission.title
    awards = submission.total_awards_received
    upvotes = submission.ups
    views = submission.view_count

    return [id, datetime, nltk_score, textblob_score, author, title, body, awards, upvotes, downvotes, views]

submissions = [forest_explorer(sub) for sub in submissions]

submissions = pd.DataFrame(submissions, columns=('id', 'datetime', 'nltk', 'textblob', 'author', 'title',
                                                 'body', 'awards', 'upvotes', 'downvotes', 'views'))

submissions.to_csv('C:/Users/emanu/Desktop/Thesis Project/raw_data/reddit_sentiment.csv')