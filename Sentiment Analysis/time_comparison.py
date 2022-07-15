from adapters.sentiment_analysis.adapter import SentimentAnalysis
from adapters.sentiment_analysis.analyzer_client import SentimentAnalyzerClient

import json
from pysentimiento import create_analyzer
import time


sentiment_analyzer_client = SentimentAnalyzerClient(host='172.17.0.2', port=50005)
freeling_analyzer = SentimentAnalysis(sentiment_analyzer_client)
pysentimiento_analyzer = create_analyzer(task="sentiment", lang="es")

with open('ada_colau.json') as f:
    data = json.load(f)
ada_colau_tweets =  []
for tweet in data:
    ada_colau_tweets.append(tweet["body"])

# freeling
scores_freeling = []
t0 = time.perf_counter()
for tweet in ada_colau_tweets:
    score = freeling_analyzer.compute_sentiment(tweet).sentiment_score
    scores_freeling.append(score)
dt = time.perf_counter() - t0
print(f'Analyzing {len(ada_colau_tweets)} tweets took {round(dt, 2)} seconds using freeling')
#print(scores_freeling)

# pysentimiento method 1
scores_pysentimiento_1 = []
t0 = time.perf_counter()
for tweet in ada_colau_tweets:
    output = pysentimiento_analyzer.predict(tweet)
    prob_pos, prob_neu, prob_neg = output.probas["POS"], output.probas["NEU"], output.probas["NEG"]
    score = 1*prob_pos + 0.5*prob_neu + 0*prob_neg
    scores_pysentimiento_1.append(score)
dt = time.perf_counter() - t0
print(f'Analyzing {len(ada_colau_tweets)} tweets took {round(dt, 2)} seconds using pysentimiento - method 1')
#print(scores_pysentimiento_1)

# pysentimiento method 2
scores_pysentimiento_2 = []
t0 = time.perf_counter()
outputs = pysentimiento_analyzer.predict(ada_colau_tweets)
for output in outputs:
    prob_pos, prob_neu, prob_neg = output.probas["POS"], output.probas["NEU"], output.probas["NEG"]
    score = 1*prob_pos + 0.5*prob_neu + 0*prob_neg
    scores_pysentimiento_2.append(score)
dt = time.perf_counter() - t0
print(f'Analyzing {len(ada_colau_tweets)} tweets took {round(dt, 2)} seconds using pysentimiento - method 2')
#print(scores_pysentimiento_2)