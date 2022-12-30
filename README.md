# Financial Phrase Bank Sentiment Analysis

- Built a sentiment analysis model to predict the sentiment score of a financial news article
- The data consists of 4845 english articles that were cateogorized by sentiment class and were annotated by 16 reasearchers with a financial background.
- A BERT model was used as a baseline. The **FinBERT and DistilBERT** models were fine-tuned to get the best results.
- The best results were obtained using the fine-tuned on the FINBERT model.It achieved an **Accuracy of 90.9%** and a **F1 Score** of 0.91. 

## Data:

The Financial PhraseBank dataset consists of 4840 sentences from English language financial news categorised by sentiment. 
These sentences then were annotated by 16 people with background in finance and business.
The dataset can be downloaded from [here](https://huggingface.co/datasets/financial_phrasebank).

The dataset contains two columns <br>

**Sentiment**: The sentiment can be negative, neutral or positive.
<br>
**News Headline**: Headlines of the news articles.
Predicting the sentiment based on the news headlines.
<br>

## Experiments:
#### **BERT:**

- A baseline was created using the BERT model. Training the model with an **Adam optimizer with learning rate of 5e-5** for **6 epochs** yielded an **Accuracy of 86% and an F1 Score of 0.86.**

#### **DistilBERT**

- The DistilBERT model was fine tuned on the data. Training the model with an **AdamW optimizer with learning rate of 5e-5, yielded an **Accuracy of 82% and an F1 Score of 0.81.**

#### **FINBERT**

- The FINBERT model was fine tuned on the data.Training the model with an **Adam optimizer** with learning rate of 5e-5,  for **6 epochs** yielded an **Accuracy of 90.91% and an F1 Score of 0.91.**
Parameter settings:

    batch size 32
    warmup step 4000
    epoch 5


| Model | Epochs | Accuracy | F1 Score(Weighted) |
| --- | --- | --- | --- |
| FinBERT| 3 | 90.9% | 0.91|
| BERT | 3 | 86% | 0.86|
| DistilBERT | 3 | 82% |0.81|






















