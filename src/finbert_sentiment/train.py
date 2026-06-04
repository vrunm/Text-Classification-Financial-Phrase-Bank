"""Fine-tune a transformer model for sentiment classification on the
Financial PhraseBank dataset.

The dataset is streamed directly from the Hugging Face Hub
(``takala/financial_phrasebank``); no manual download is required.

Example
-------
    finbert-train --config sentences_allagree --model ProsusAI/finbert

or equivalently::

    python -m finbert_sentiment.train --config sentences_allagree
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    logging,
)

logging.set_verbosity_error()

DATASET_ID = "takala/financial_phrasebank"
# Agreement-level configurations exposed by the Hub dataset.
DATASET_CONFIGS = (
    "sentences_50agree",
    "sentences_66agree",
    "sentences_75agree",
    "sentences_allagree",
)


def set_seed(seed: int) -> None:
    """Seed all sources of randomness for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_phrasebank(
    config: str,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load a Financial PhraseBank configuration from the Hugging Face Hub.

    Returns the list of sentences, the integer-encoded labels, and the
    ordered list of class names (index ``i`` is the name of label ``i``).
    The ``label`` feature is already a :class:`~datasets.ClassLabel` with the
    canonical mapping ``0=negative, 1=neutral, 2=positive``.
    """
    dataset = load_dataset(DATASET_ID, config, split="train")
    label_names = dataset.features["label"].names
    sentences = dataset["sentence"]
    labels = np.array(dataset["label"], dtype=np.int64)
    return sentences, labels, label_names


def encode_texts(
    tokenizer: BertTokenizer, texts: list[str], max_length: int
) -> dict[str, torch.Tensor]:
    """Tokenize a list of texts into padded/truncated tensors."""
    return tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def build_dataset(
    tokenizer: BertTokenizer,
    texts: list[str],
    labels: np.ndarray,
    max_length: int,
) -> TensorDataset:
    encoded = encode_texts(tokenizer, texts, max_length)
    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(labels),
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return average loss, predicted class ids, and true labels."""
    model.eval()
    loss_total = 0.0
    predictions, true_vals = [], []

    for batch in dataloader:
        input_ids, attention_mask, labels = (b.to(device) for b in batch)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss_total += outputs.loss.item()
        predictions.append(outputs.logits.detach().cpu().numpy())
        true_vals.append(labels.cpu().numpy())

    loss_avg = loss_total / len(dataloader)
    predictions = np.concatenate(predictions, axis=0).argmax(axis=1)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_avg, predictions, true_vals


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sentences, labels, label_names = load_phrasebank(args.config)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        sentences,
        labels,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=labels,
    )

    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=True)
    dataset_train = build_dataset(
        tokenizer, train_texts, train_labels, args.max_length
    )
    dataset_val = build_dataset(
        tokenizer, val_texts, val_labels, args.max_length
    )

    dataloader_train = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=args.batch_size,
    )
    dataloader_val = DataLoader(
        dataset_val,
        sampler=SequentialSampler(dataset_val),
        batch_size=args.batch_size,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={name: idx for idx, name in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train) * args.epochs,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        model.train()
        loss_train_total = 0.0
        progress_bar = tqdm(
            dataloader_train, desc=f"Epoch {epoch}", leave=False
        )
        for batch in progress_bar:
            model.zero_grad()
            input_ids, attention_mask, batch_labels = (
                b.to(device) for b in batch
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels,
            )
            loss = outputs.loss
            loss_train_total += loss.item()
            loss.backward()
            # Clip gradients to guard against exploding gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix(training_loss=f"{loss.item():.3f}")

        torch.save(
            model.state_dict(),
            output_dir / f"finetuned_{args.model.split('/')[-1]}_epoch_{epoch}.pt",
        )

        loss_train_avg = loss_train_total / len(dataloader_train)
        val_loss, predictions, true_vals = evaluate(
            model, dataloader_val, device
        )
        val_f1 = f1_score(true_vals, predictions, average="weighted")
        val_acc = accuracy_score(true_vals, predictions)
        tqdm.write(
            f"Epoch {epoch} | train_loss={loss_train_avg:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f} "
            f"| val_f1={val_f1:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="sentences_allagree",
        choices=DATASET_CONFIGS,
        help=(
            "Financial PhraseBank agreement-level configuration to load from "
            "the Hugging Face Hub (takala/financial_phrasebank)."
        ),
    )
    parser.add_argument(
        "--model",
        default="ProsusAI/finbert",
        help="Hugging Face model identifier.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Directory to write model checkpoints to.",
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
