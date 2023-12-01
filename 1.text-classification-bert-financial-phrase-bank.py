import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import logging

logging.set_verbosity_error()
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

# Load the dataset

financial_data = pd.read_csv(
    "financial_phrase_bank.csv", encoding="latin-1", names=["sentiment", "NewsHeadline"]
)


# Label encode the sentiment values
# The unique values in sentiment column are returned as a NumPy array.
# Enumerate method adds counter to an iterable and returns it. The returned object is an enumerate object.
# Convert enumerate objects to list.
def encode_sentiments_values(df):
    possible_sentiments = df.sentiment.unique()
    sentiment_dict = {}

    for index, possible_sentiment in enumerate(possible_sentiments):
        sentiment_dict[possible_sentiment] = index

    # Encode all the sentiment values
    df["label"] = df.sentiment.replace(sentiment_dict)

    return df, sentiment_dict


# Encode the sentiment column
financial_data, sentiment_dict = encode_sentiments_values(financial_data)

# Create training and validation data
# Training set as 80% and test set as 20%

X_train, X_val, y_train, y_val = train_test_split(
    financial_data.index.values,
    financial_data.label.values,
    test_size=0.20,
    random_state=2022,
    stratify=financial_data.label.values,
)

# Get the BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

# Encode the Training and Validation Data
# encode_plus method performs the following tasks:
# split our news headlines into tokens,
# add the special [CLS] and [SEP] tokens
# convert these tokens into indexes of the tokenizer vocabulary,
# pad or truncate sentences to max length, then finally create an attention mask.

# return_tensors (str, optional, defaults to None) – Can be set to ‘tf’ or ‘pt’ to return respectively TensorFlow tf.constant or PyTorch torch.Tensor instead of a list of python integers.

# add_special_tokens (bool, optional, defaults to True) – If set to True, the sequences will be encoded with the special tokens relative to their model.

# return_attention_masks (bool, optional, defaults to none) –

# Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer’s default,

# pad_to_max_length (bool, optional, defaults to False) –
# If set to True, the returned sequences will be padded according to the model’s padding side and padding index, up to their max length.

# max_length (int, optional, defaults to None) – If set to a number, will limit the total sequence returned so that it has a maximum length
# 150 is used since it is the maximum length observed in the headlines

encoded_data_train = tokenizer.batch_encode_plus(
    X_train.NewsHeadline.values,
    return_tensors="pt",
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=150,
)

encoded_data_val = tokenizer.batch_encode_plus(
    X_val.NewsHeadline.values,
    return_tensors="pt",
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=150,
)


input_ids_train = encoded_data_train["input_ids"]
attention_masks_train = encoded_data_train["attention_mask"]
labels_train = torch.tensor(y_train.values)

input_ids_val = encoded_data_val["input_ids"]
attention_masks_val = encoded_data_val["attention_mask"]
sentiments_val = torch.tensor(y_val.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, sentiments_val)


###Torch DataLoader
# torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)
# Samples elements randomly. If without replacement, then sample from a shuffled dataset. If with replacement, then user can specify num_samples to draw.
# data_source (Dataset) – dataset to sample from
# replacement (bool) – samples are drawn on-demand with replacement if True, default=``False``
# num_samples (int) – number of samples to draw, default=`len(dataset)`

# torch.utils.data.SequentialSampler(data_source)
# Samples elements sequentially, always in the same order.
# data_source (Dataset) – dataset to sample from

batch_size = 32

dataloader_train = DataLoader(
    dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size
)

dataloader_validation = DataLoader(
    dataset_val, sampler=RandomSampler(dataset_val), batch_size=batch_size
)

model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert", num_labels=len(sentiment_dict)
)

# To construct an Optimizer you have to give it an iterable containing the parameters (all should be Variable s) to optimize. Then, you can specify optimizer-specific options such as the learning rate, weight decay, etc.

# torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, *, maximize=False, foreach=None, capturable=False)

# transformers.get_linear_schedule_with_warmup
# Parameters optimizer (~torch.optim.Optimizer) — The optimizer for which to schedule the learning rate.
# num_warmup_steps (int) — The number of steps for the warmup phase.
# num_training_steps (int) — The total number of training steps.
# last_epoch (int, optional, defaults to -1) — The index of the last epoch when resuming training.
epochs = 3
optimizer1 = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(
    optimizer1, num_warmup_steps=0, num_training_steps=len(dataloader_train) * epochs
)


seed_val = 2022
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def evaluate(dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs + 1)):
    model.train()

    loss_train_total = 0

    progress_bar = tqdm(
        dataloader_train, desc="Epoch {:1d}".format(epoch), leave=False, disable=False
    )
    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        # Gradient Clipping is done to restrict the values of the gradient(To prevent the model from exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer1.step()
        scheduler.step()

        progress_bar.set_postfix(
            {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
        )

    torch.save(model.state_dict(), f"finetuned_BERT_epoch_{epoch}.model")

    tqdm.write(f"\nEpoch {epoch}")

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f"Training loss: {loss_train_avg}")

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score(predictions, true_vals, average="weighted")
    tqdm.write(f"Validation loss: {val_loss}")
    tqdm.write(f"F1 Score (Weighted): {val_f1}")


# Load the best model & Make Predictions

model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert", num_labels=len(sentiment_dict)
)

model.to(device)

model.load_state_dict(
    torch.load("finetuned_BERT_epoch_1.model", map_location=torch.device("cpu"))
)

_, predictions, true_vals = evaluate(dataloader_validation)

print("Accuracy: ", accuracy_score(predictions, true_vals))
