from adapters.sentiment_analysis.adapter import SentimentAnalysis
from adapters.sentiment_analysis.analyzer_client import SentimentAnalyzerClient

sentiment_analyzer_client = SentimentAnalyzerClient(host='172.17.0.2', port=50005)

analyzer = SentimentAnalysis(sentiment_analyzer_client)

FILE_NAME = "tweets.txt"

f = open(FILE_NAME, "r")

for tweet in f:
    tweet = tweet.replace("\n", "")
    print(analyzer.compute_sentiment(tweet))
    print(tweet)
f.close()