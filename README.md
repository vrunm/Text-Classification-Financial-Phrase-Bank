# Financial Phrase Bank Sentiment Analysis

- Built a sentiment analysis model to predict the sentiment of a financial news sentence.
- The data consists of English financial news sentences categorised by sentiment (negative / neutral / positive) and annotated by 16 researchers with a financial background.
- A BERT model was used as a baseline. The **FinBERT and DistilBERT** models were fine-tuned to get the best results.
- The best results were obtained using the fine-tuned on the FINBERT model. It achieved an **Accuracy of 90.9%** and a **F1 Score** of 0.91.

## Project Structure

```
.
├── src/
│   └── finbert_sentiment/
│       └── train.py    # Training / evaluation entry point
├── plots/              # Loss and metric figures
├── pyproject.toml      # Project metadata & dependencies
├── LICENSE
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The dataset is downloaded automatically from the [Hugging Face Hub](https://huggingface.co/datasets/takala/financial_phrasebank)
on the first run via the `datasets` library — no manual download required.

## Usage

```bash
finbert-train \
    --config sentences_allagree \
    --model ProsusAI/finbert \
    --epochs 3 \
    --batch-size 32
```

The `--config` flag selects one of the four annotator-agreement subsets of the
dataset (see below). Run `finbert-train --help` to see all options (learning
rate, max sequence length, validation split, output directory, etc.).
Checkpoints are written to `checkpoints/` after each epoch.

## Data:

The [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
dataset consists of sentences from English language financial news categorised
by sentiment. The sentences were annotated by 16 people with a background in
finance and business.

The dataset is published on the Hugging Face Hub as
`takala/financial_phrasebank` with two features:

- **sentence** (`string`): the financial news sentence.
- **label** (`ClassLabel`): the sentiment, one of `negative` (0),
  `neutral` (1) or `positive` (2).

It is split into four configurations by the proportion of annotators that
agreed on the label. Higher agreement means cleaner but fewer examples:

| Config | Annotator agreement | Sentences |
| ------ | ------------------- | --------- |
| `sentences_50agree`  | ≥ 50%  | 4,846 |
| `sentences_66agree`  | ≥ 66%  | 4,217 |
| `sentences_75agree`  | ≥ 75%  | 3,453 |
| `sentences_allagree` | 100%   | 2,264 |

The dataset ships a single `train` split, so the script carves out a
stratified validation set (controlled by `--val-size`, default 20%).

```python
from datasets import load_dataset

ds = load_dataset("takala/financial_phrasebank", "sentences_allagree", split="train")
print(ds[0])  # {'sentence': '...', 'label': 1}
```

## Experiments:
#### **BERT:**

- A baseline was created using the BERT model. Training the model with an **Adam optimizer with learning rate of 5e-5** for **3 epochs** yielded an **Accuracy of 86% and an F1 Score of 0.86.**

#### **DistilBERT**

- The DistilBERT model was fine tuned on the data. Training the model with an **AdamW optimizer with learning rate of 5e-5** yielded an **Accuracy of 82% and an F1 Score of 0.81.**

#### **FINBERT**

- The FINBERT model was fine tuned on the data. Training the model with an **Adam optimizer** with learning rate of 5e-5 for **3 epochs** yielded an **Accuracy of 90.91% and an F1 Score of 0.91.**


| Model | Epochs | Accuracy | F1 Score(Weighted) |
| --- | --- | --- | --- |
| FinBERT| 3 | 90.9% | 0.91|
| BERT | 3 | 86% | 0.86|
| DistilBERT | 3 | 82% |0.81|


- We have tuned a subset of the optimization hyperparameters by running a set of trials to maximize performance over the validation set.
- The inclusion relationships hold in all cases a more generalized optimizer never underperforms any of its specializations.
- The most general optimizers we considered were RMSprop, ADAM which do not include each other as special cases and whose relative performance is not predicted by inclusion relationships.



Optimizer    | Learning Rate $\gamma$ |   Momentum $\eta$ | Alpha $\alpha$ | Beta1 $\beta_1$ | Beta2 $\beta_2$ | Epsilon $\epsilon$ |
| ---        | ---                    | ---               | ---            | ---             | ---             | ---                |
AdamW        | 5e-5                   | 0.01              | 0.9            | 0.9             | 0.999           | 1e-5               |
RMSprop      | 0.01                   | 0.01              | 0.99           | -               | -               | 1e-5                 |
NAG          | 5e-5 |                 | -                 | -              | -               |-                | -                  |
SGD(Momentum)| 5e-5                   | 0.001                | -              |  -           |-                | -                  |
SGD          | 0.01 |                 |      -             |     -           |       -       |    -            |     -               |

- **Adam** converged the fastest as the training loss became constant after certain epochs.

- **RMSprop** optimizer gets stuck in a local minima and takes longer to converge.

- **SGD** and **SGD with momentum** and **NAG** take longer to converge.

## Results:

The best results were obtained using a fine-tuned FinBERT model. It was used for generating the final predictions.

The results from all the text classification models have been summarized below:

| Model | Epochs | Accuracy | F1 Score(Weighted) |
| ----- | ------ | -------- | ------------------ |
| FinBERT| 3 | 90.9% | 0.91|
| BERT | 3 | 86% |0.86|
| DistilBERT | 3 | 82% |0.81|

The values learning rate for the adam optimizer and batch size of the model were taken from:
[[1]](https://www.researchgate.net/publication/358284785_FinancialBERT_-_A_Pretrained_Language_Model_for_Financial_Text_Mining).
[[2]](https://arxiv.org/pdf/1908.10063.pdf).

Considering the FinBERT model a detailed analysis of the optimizers used for training has been done.
The table lists out the different optimizers and their parameters used in training.

Taking inspiration of the empirical comparison of optimizers in [[3]](https://arxiv.org/pdf/1910.05446.pdf) the FinBERT model has been fine tuned on different optimizers mentioned below:

Empirical Relations of optimizers has been used from:
[[4]](https://arxiv.org/pdf/1705.08292.pdf)
[[5]](https://arxiv.org/pdf/1705.07774.pdf)


**Comparing the Training loss of all optimizers** for the fine tuned FinBERT model
<br>
<img src = "plots/1.phrase_train_loss_all.png">


**Comparing the Validation loss of all optimizers** for the fine tuned FinBERT model
<br>
<img src = "plots/1.phrase_val_loss_all.png">


The rate of convergence of the Adam optimizer is the fastest.

We can conclude the order of convergence of the optimizers:
AdamW > RMSprop > NAG > SGD (Momentum) > SGD

References:

[1] [FinancialBERT - A Pretrained Language Model for Financial Text Mining](https://www.researchgate.net/publication/358284785_FinancialBERT_-_A_Pretrained_Language_Model_for_Financial_Text_Mining)

[2] [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/pdf/1908.10063.pdf)

[3] [On Empirical Comparisons of Optimizers for Deep Learning](https://arxiv.org/pdf/1910.05446.pdf)

[4] [The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/pdf/1705.08292.pdf)

[5] [Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients](https://arxiv.org/pdf/1705.07774.pdf)
