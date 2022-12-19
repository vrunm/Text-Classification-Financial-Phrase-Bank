**Financial Phrase Bank dataset**:
Polar sentiment dataset of sentences from financial news. The dataset consists of 4840 sentences from English language financial news categorised by sentiment. 

The dataset contains two columns <br>
**Sentiment**: The sentiment can be negative, neutral or positive.
<br>
**News Headline**: Headlines of the news articles.
Predicting the sentiment based on the news headlines.

Parameter settings:

    batch size 32
    warmup step 4000
    epoch 5

| Model | Epochs | Accuracy | F1 Score(Weighted) |
| --- | --- | --- | --- |
| FinBERT| 3 | 90.9% | 0.91|
| BERT | 3 | 86% |0.86|
| DistilBert | 3 | 76% |0.74|



| Model | Epochs | Accuracy | F1 Score(Weighted) | Optimizer | Learning Rate $\gamma$| Momentum $\eta$ | Alpha $\alpha$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FinBERT| 3 | 90.9% | 0.91 | AdamW | 5e-5 |
| FinBERT| 3 | 90.9% | 0.91 | SGD   | 0.01 | 
| FinBERT| 3 | 90.9% | 0.91 | SGD(Momentum)| 5e-5 | 0.001|
| FinBERT| 3 | 90.9% | 0.91 | RMSprop | 0.01 | 0.01 | 0.99
| FinBERT| 3 | 91.3222% | 0.91 | Adagrad | 5e-5 |

Plotting the loss per epoch for AdamW:
<img src = "optim1_adam/1.loss_f1.png">

Plotting the log loss per epoch:
<img src = "optim1_adam/2.loss_epoch.png">

Plotting f1 score per epoch:
<img src = "loss_f1.png">


















