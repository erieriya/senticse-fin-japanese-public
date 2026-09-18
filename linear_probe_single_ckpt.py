#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Single-checkpoint linear probing evaluator for EWS (retarfi/economy-watchers-survey).

Usage:
python linear_probe_single_ckpt.py \
  --checkpoint_dir /path/to/checkpoint-2500 \
  --tokenizer_dir rinna/japanese-roberta-base \
  --current_or_future current \
  --device cuda:0 \
  --pred_csv_dir ./pred_csv_single \
  --metrics_json_path ./metrics_single.json
"""

import argparse
from argparse import Namespace
import csv
import json
import os
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    cohen_kappa_score,
)
from transformers import AutoConfig, AutoTokenizer, T5Tokenizer

from senticse.models import BertForCL, RobertaForCL, ElectraModelForCL


# ==========================================
# EWS current/future のラベルマップ（3クラス）
#   2 = pos（◎/○）
#   1 = neu（□）
#   0 = neg（▲/×）
# ==========================================
LABEL_MAP: Dict[str, int] = {
    "◎": 2,
    "○": 2,
    "▲": 0,
    "×": 0,
    "□": 1,
}
LABEL_NAMES = ["neg", "neu", "pos"]


def _safe_get(ex: Dict[str, Any], key: str) -> str:
    v = ex.get(key, None)
    if v is None:
        return ""
    return str(v)


def load_ews(split: str, current_or_future: str) -> List[Dict[str, Any]]:
    """
    EWS dataset (retarfi/economy-watchers-survey) の {current_or_future}/{split}.jsonl を読み込み、
    予測詳細CSVに必要なメタ情報も含めた records を返す。

    共通で:
      label_char, y_true(0/1/2), text_for_embedding(埋め込みに使う本文)
    """
    assert split in {"train", "validation", "test"}
    assert current_or_future in {"current", "future"}

    filename = f"{current_or_future}/{split}.jsonl"

    local_path = hf_hub_download(
        repo_id="retarfi/economy-watchers-survey",
        filename=filename,
        repo_type="dataset",
        revision="2025.11.0",
    )

    records: List[Dict[str, Any]] = []
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)

            base = {
                "id": _safe_get(ex, "id"),
                "year-month": _safe_get(ex, "year-month"),
                "地域": _safe_get(ex, "地域"),
                "関連": _safe_get(ex, "関連"),
                "業種・職種": _safe_get(ex, "業種・職種"),
            }

            if current_or_future == "future":
                label_char = ex.get("景気の先行き判断", None)
                reason = ex.get("景気の先行きに対する判断理由", None)
                if label_char is None or reason is None:
                    continue

                y = LABEL_MAP.get(label_char, None)
                if y is None:
                    continue

                rec = dict(base)
                rec.update(
                    {
                        "景気の先行き判断": str(label_char),
                        "景気の先行きに対する判断理由": str(reason),
                        "label_char": str(label_char),
                        "y_true": int(y),
                        "text_for_embedding": str(reason),
                    }
                )
                records.append(rec)

            else:
                label_char = ex.get("景気の現状判断", None)
                reason = ex.get("判断の理由", None)
                detail = ex.get("追加説明及び具体的状況の説明", None)
                if label_char is None or detail is None:
                    continue

                y = LABEL_MAP.get(label_char, None)
                if y is None:
                    continue

                rec = dict(base)
                rec.update(
                    {
                        "景気の現状判断": str(label_char),
                        "判断の理由": "" if reason is None else str(reason),
                        "追加説明及び具体的状況の説明": str(detail),
                        "label_char": str(label_char),
                        "y_true": int(y),
                        "text_for_embedding": str(detail),
                    }
                )
                records.append(rec)

    print(f"[INFO] EWS {current_or_future}/{split} 有効サンプル数: {len(records)}")
    return records


class RecordDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {"text": r["text_for_embedding"]}


def extract_step_from_path(path: str) -> int:
    base = os.path.basename(path.rstrip("/"))
    if base.startswith("checkpoint-"):
        try:
            return int(base.split("-")[-1])
        except ValueError:
            return 0
    return 0


def _load_tokenizer(tokenizer_dir: str):
    """
    rinna/japanese-roberta-base 系は SentencePiece ベースの T5Tokenizer を試す
    """
    try:
        tok = T5Tokenizer.from_pretrained(tokenizer_dir)
        print("[INFO] Using T5Tokenizer (SentencePiece)")
        return tok
    except Exception as e:
        print(f"[WARN] T5Tokenizer load failed ({e}), falling back to AutoTokenizer")
        return AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)


def encode_records(
    model: RobertaForCL,
    tokenizer,
    records: List[Dict[str, Any]],
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 64,
    pooling: str = "cls_before_pooler",
) -> np.ndarray:
    dataset = RecordDataset(records)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    embs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding", leave=False):
            enc = tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            if pooling == "senticse":
                emb = model(**enc, return_dict=True, sent_emb=True).pooler_output
            else:
                outputs = model.base_model(**enc, return_dict=True)
                emb = outputs.last_hidden_state[:, 0, :]

            embs.append(emb.detach().cpu().numpy())

    return np.concatenate(embs, axis=0)


def summarize_predictions(y_true: List[int], y_pred: np.ndarray, label_names: List[str]) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    overall_acc = float((y_true_arr == y_pred_arr).mean())

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", labels=list(range(len(label_names)))))
    micro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="micro", labels=list(range(len(label_names)))))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", labels=list(range(len(label_names)))))

    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    qwk = float(
        cohen_kappa_score(
            y_true_arr,
            y_pred_arr,
            labels=list(range(len(label_names))),
            weights="quadratic",
        )
    )

    per_class = {}
    for i, name in enumerate(label_names):
        mask = (y_true_arr == i)
        total = int(mask.sum())
        correct = int((y_pred_arr[mask] == i).sum()) if total > 0 else 0
        acc_i = (correct / total) if total > 0 else 0.0
        per_class[name] = {"correct": correct, "total": total, "acc": float(acc_i)}

    cm = confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(label_names))),
    )

    mis = []
    k = len(label_names)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            n = int(cm[i, j])
            if n > 0:
                mis.append({"true": label_names[i], "pred": label_names[j], "count": n})

    return {
        "overall_acc": overall_acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "mse": mse,
        "qwk": qwk,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "misclassifications": mis,
    }


def write_split_pred_csv(
    out_csv_path: str,
    step: int,
    split: str,
    current_or_future: str,
    records: List[Dict[str, Any]],
    y_pred: np.ndarray,
):
    assert len(records) == len(y_pred)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    common_cols = ["step", "split", "id", "year-month", "地域", "関連", "業種・職種"]

    if current_or_future == "current":
        ews_cols = ["景気の現状判断", "判断の理由", "追加説明及び具体的状況の説明"]
    else:
        ews_cols = ["景気の先行き判断", "景気の先行きに対する判断理由"]

    extra_cols = ["text_for_embedding"]
    pred_cols = ["y_true", "y_true_name", "y_pred", "y_pred_name", "correct"]
    header = common_cols + ews_cols + extra_cols + pred_cols

    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for r, yp in zip(records, y_pred):
            yt = int(r["y_true"])
            yp = int(yp)
            correct = 1 if yt == yp else 0

            row = [
                step,
                split,
                r.get("id", ""),
                r.get("year-month", ""),
                r.get("地域", ""),
                r.get("関連", ""),
                r.get("業種・職種", ""),
            ]
            for c in ews_cols:
                row.append(r.get(c, ""))
            row.append(r.get("text_for_embedding", ""))

            row.extend([yt, LABEL_NAMES[yt], yp, LABEL_NAMES[yp], correct])
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="checkpoint-xxxx のディレクトリ")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="tokenizer のディレクトリ or HF model id")
    parser.add_argument(
        "--current_or_future",
        type=str,
        default="current",
        choices=["current", "future"],
        help="EWS の current 版 or future 版を使うか",
    )
    parser.add_argument("--device", type=str, default="cpu", help="使用するデバイス (例: cpu, cuda:0)")
    parser.add_argument("--max_length", type=int, default=128, help="max_seq_length")
    parser.add_argument("--batch_size", type=int, default=64, help="埋め込み計算のバッチサイズ")

    parser.add_argument("--pred_csv_dir", type=str, default="", help="(任意) 予測詳細CSVの出力先ディレクトリ")
    parser.add_argument("--metrics_json_path", type=str, default="", help="(任意) 指標JSONの出力先パス")

    parser.add_argument("--pooling", choices=["cls_before_pooler", "senticse"], default="cls_before_pooler", help="Legacy raw CLS or checkpoint pooling including the trained MLP")

    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        raise FileNotFoundError(f"--checkpoint_dir not found: {args.checkpoint_dir}")

    step = extract_step_from_path(args.checkpoint_dir)
    device = torch.device(args.device)
    print(f"[INFO] checkpoint_dir={args.checkpoint_dir} (step={step})")
    print(f"[INFO] device={device}")

    tokenizer = _load_tokenizer(args.tokenizer_dir)

    config = AutoConfig.from_pretrained(args.checkpoint_dir)
    model_class = {"bert": BertForCL, "roberta": RobertaForCL, "electra": ElectraModelForCL}.get(config.model_type)
    if model_class is None:
        raise ValueError(f"Unsupported backbone: {config.model_type}")
    model = model_class.from_pretrained(args.checkpoint_dir).to(device)

    # ---- data ----
    train_records = load_ews("train", args.current_or_future)
    val_records = load_ews("validation", args.current_or_future)
    test_records = load_ews("test", args.current_or_future)

    y_train = [int(r["y_true"]) for r in train_records]
    y_val = [int(r["y_true"]) for r in val_records]
    y_test = [int(r["y_true"]) for r in test_records]

    # ---- embeddings ----
    X_train = encode_records(model, tokenizer, train_records, device, args.max_length, args.batch_size, args.pooling)
    X_val = encode_records(model, tokenizer, val_records, device, args.max_length, args.batch_size, args.pooling)
    X_test = encode_records(model, tokenizer, test_records, device, args.max_length, args.batch_size, args.pooling)

    # ---- linear classifier ----
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_metrics = summarize_predictions(y_val, val_pred, LABEL_NAMES)
    test_metrics = summarize_predictions(y_test, test_pred, LABEL_NAMES)

    print(f"\n[RESULT] step={step}  (validation)")
    print(f"  overall_acc = {val_metrics['overall_acc']:.4f}")
    print(f"  macro_f1    = {val_metrics['macro_f1']:.4f}")
    print(f"  micro_f1    = {val_metrics['micro_f1']:.4f}")
    print(f"  weighted_f1 = {val_metrics['weighted_f1']:.4f}")
    print(f"  mse         = {val_metrics['mse']:.4f}")
    print(f"  qwk         = {val_metrics['qwk']:.4f}")
    print("  confusion matrix (rows=true, cols=pred)  order=[neg, neu, pos]")
    print(np.array(val_metrics["confusion_matrix"]))

    print(f"\n[RESULT] step={step}  (test)")
    print(f"  overall_acc = {test_metrics['overall_acc']:.4f}")
    print(f"  macro_f1    = {test_metrics['macro_f1']:.4f}")
    print(f"  micro_f1    = {test_metrics['micro_f1']:.4f}")
    print(f"  weighted_f1 = {test_metrics['weighted_f1']:.4f}")
    print(f"  mse         = {test_metrics['mse']:.4f}")
    print(f"  qwk         = {test_metrics['qwk']:.4f}")
    print("  confusion matrix (rows=true, cols=pred)  order=[neg, neu, pos]")
    print(np.array(test_metrics["confusion_matrix"]))

    # ---- optional outputs ----
    if args.pred_csv_dir:
        step_dir = os.path.join(args.pred_csv_dir, f"step{step}")
        write_split_pred_csv(
            out_csv_path=os.path.join(step_dir, "validation.csv"),
            step=step,
            split="validation",
            current_or_future=args.current_or_future,
            records=val_records,
            y_pred=val_pred,
        )
        write_split_pred_csv(
            out_csv_path=os.path.join(step_dir, "test.csv"),
            step=step,
            split="test",
            current_or_future=args.current_or_future,
            records=test_records,
            y_pred=test_pred,
        )
        print(f"[INFO] Saved prediction CSVs under: {step_dir}")

    if args.metrics_json_path:
        out_dir = os.path.dirname(args.metrics_json_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        payload = {
            "checkpoint_dir": args.checkpoint_dir,
            "step": step,
            "tokenizer_dir": args.tokenizer_dir,
            "current_or_future": args.current_or_future,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "validation": val_metrics,
            "test": test_metrics,
        }
        with open(args.metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved metrics JSON: {args.metrics_json_path}")

    # cleanup
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
