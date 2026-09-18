"""Data helpers shared by training and inference; no bundled private assets."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def load_sentiment_vocab(path, vocab_size):
    if not path:
        raise ValueError("--do_mlm requires --sentiment_vocab_file for the selected tokenizer.")
    values = np.load(Path(path).expanduser(), allow_pickle=False)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < vocab_size:
        raise ValueError(f"Sentiment vocabulary must have shape (3, >= {vocab_size}); got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Sentiment vocabulary contains non-finite values.")
    return values


@dataclass
class ContrastiveCollator:
    tokenizer: object

    def __call__(self, features):
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        count = len(features[0]["input_ids"])
        if any(len(feature["input_ids"]) != count for feature in features):
            raise ValueError("Each row must contain the same number of sentences.")
        keys = ("input_ids", "attention_mask", "token_type_ids")
        flat = [{key: feature[key][i] for key in keys if key in feature}
                for feature in features for i in range(count)]
        batch = self.tokenizer.pad(flat, padding=True, return_tensors="pt")
        length = batch["input_ids"].shape[-1]
        # Tokenizer.pad does not pad custom MLM columns. Pad them explicitly.
        for key, pad_value in (("mlm_input_ids", self.tokenizer.pad_token_id), ("mlm_labels", -100)):
            if key not in features[0]:
                continue
            rows = []
            for feature in features:
                for values in feature[key]:
                    values = list(values)
                    padding = [pad_value] * (length - len(values))
                    rows.append(padding + values if self.tokenizer.padding_side == "left" else values + padding)
            batch[key] = torch.tensor(rows, dtype=torch.long)
        return {key: value.reshape(len(features), count, -1) for key, value in batch.items()}
