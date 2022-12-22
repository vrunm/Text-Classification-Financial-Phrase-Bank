**Financial Phrase Bank dataset**:
Polar sentiment dataset of sentences from financial news. The dataset consists of 4840 sentences from English language financial news categorised by sentiment. 

The dataset contains two columns <br>
**Sentiment**: The sentiment can be negative, neutral or positive.
<br>
**News Headline**: Headlines of the news articles.
Predicting the sentiment based on the news headlines.
<br>

Parameter settings:

    batch size 32
    warmup step 4000
    epoch 5


| Model | Epochs | Accuracy | F1 Score(Weighted) |
| --- | --- | --- | --- |
| FinBERT| 3 | 90.9% | 0.91|
| BERT | 3 | 86% |0.86|
| DistilBert | 3 | 76% |0.74|






















