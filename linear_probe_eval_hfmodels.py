#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HF models linear probing on EWS (current/future) with robust local snapshot loading.

Fixes:
- Avoid MecabTokenizer (requires fugashi) by using BertJapaneseTokenizer for Japanese BERT models.
- Remove T5Tokenizer fallback (was causing TypeError and is unnecessary here).
- Uses snapshot_download -> local_dir and loads tokenizer/model from local_dir.

Examples:

python linear_probe_eval_hfmodels.py \
  --model_ids "izumi-lab/bert-base-japanese-fin-additional,rinna/japanese-roberta-base" \
  --current_or_future current \
  --pooling cls \
  --gpus "0,1" \
  --pred_csv_dir ./pred_csv \
  --metrics_json_dir ./metrics_per_model \
  --metrics_json_path ./metrics_summary.json
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import multiprocessing as mp

from huggingface_hub import hf_hub_download, snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    cohen_kappa_score,
)
from transformers import AutoTokenizer, AutoModel
from transformers import T5Tokenizer
from transformers import BertJapaneseTokenizer


# ==========================================
# EWS current/future label map (3-class)
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
    Load EWS dataset (retarfi/economy-watchers-survey) from HF dataset repo.
    Returns records with meta fields and:
      - label_char, y_true (0/1/2), text_for_embedding
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

    def __getitem__(self, idx: int):
        r = self.records[idx]
        return {"text": r["text_for_embedding"]}


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "__")


def _snapshot_model(model_id: str) -> str:
    """
    Snapshot model repo locally and return local directory path.
    """
    local_dir = snapshot_download(
        repo_id=model_id,
        # revision="main",
        # token=True,  # if needed
    )
    return local_dir


def _find_spm_file(local_dir: str) -> str:
    """
    SentencePiece model file を snapshot dir から探して返す。
    見つからなければ例外。
    """
    cand = [
        os.path.join(local_dir, "spiece.model"),
        os.path.join(local_dir, "sentencepiece.bpe.model"),
        os.path.join(local_dir, "sentencepiece.model"),
        os.path.join(local_dir, "tokenizer.model"),
    ]
    for p in cand:
        if os.path.isfile(p):
            return p

    # それでも無ければディレクトリを走査（保険）
    for root, _, files in os.walk(local_dir):
        for fn in files:
            if fn.endswith(".model") and "sentencepiece" in fn.lower():
                return os.path.join(root, fn)

    raise FileNotFoundError(f"SentencePiece model file not found under: {local_dir}")


def _load_tokenizer_for_model(model_id: str) -> Tuple[Any, str]:
    """
    Robust tokenizer loader:
      - snapshot_download -> local_dir
      - rinna/japanese-roberta-base は SentencePiece なので T5Tokenizer を優先
      - Japanese BERT系は BertJapaneseTokenizer（fugashi不要）
      - それ以外は AutoTokenizer
    """
    local_dir = _snapshot_model(model_id)
    lowered = model_id.lower()

    # ---- (1) rinna roberta: prefer T5Tokenizer ----
    if "rinna/japanese-roberta-base" in lowered:
        try:
            # まずは普通に（うまく行く環境もある）
            tok = T5Tokenizer.from_pretrained(local_dir)
            print(f"[INFO] Tokenizer=T5Tokenizer (local) for {model_id}")
            return tok, local_dir
        except Exception as e:
            print(f"[WARN] T5Tokenizer.from_pretrained(local) failed for {model_id}: {e}")
            # ローカルのSentencePieceファイルを明示して構築
            spm_path = _find_spm_file(local_dir)
            tok = T5Tokenizer(vocab_file=spm_path)
            print(f"[INFO] Tokenizer=T5Tokenizer(vocab_file=...) for {model_id}  spm={spm_path}")
            return tok, local_dir

    # ---- (2) Japanese BERT-like: avoid MecabTokenizer requiring fugashi ----
    if ("bert-base-japanese" in lowered) or ("cl-tohoku" in lowered) or ("izumi-lab/bert" in lowered):
        tok = BertJapaneseTokenizer.from_pretrained(local_dir)
        print(f"[INFO] Tokenizer=BertJapaneseTokenizer (local) for {model_id}")
        return tok, local_dir

    # ---- (3) Default: AutoTokenizer ----
    try:
        tok = AutoTokenizer.from_pretrained(local_dir, use_fast=False)
        print(f"[INFO] Tokenizer=AutoTokenizer (local) for {model_id}")
        return tok, local_dir
    except Exception as e:
        print(f"[WARN] AutoTokenizer(local) failed for {model_id}: {e}")

    # ---- (4) Last resort: BertJapaneseTokenizer ----
    tok = BertJapaneseTokenizer.from_pretrained(local_dir)
    print(f"[INFO] Tokenizer=BertJapaneseTokenizer (local, fallback) for {model_id}")
    return tok, local_dir



def _load_encoder_model(local_dir: str, device: torch.device):
    """
    Load encoder with AutoModel from local_dir.
    Works for BERT/RoBERTa/DeBERTa etc.
    """
    model = AutoModel.from_pretrained(local_dir).to(device)
    model.eval()
    return model


def encode_records(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 64,
    pooling: str = "cls",  # "cls" or "mean"
) -> np.ndarray:
    dataset = RecordDataset(records)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding", leave=False):
            inputs = tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, T, H]

            if pooling == "cls":
                emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            embs.append(emb.detach().cpu().numpy())

    return np.concatenate(embs, axis=0)


def summarize_predictions(y_true: List[int], y_pred: np.ndarray, label_names: List[str]) -> Dict[str, Any]:
    """
    Summarize metrics: acc, macro/micro/weighted F1, MSE, QWK(quadratic), per-class acc, confusion matrix, misclassifications.
    """
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
    model_id: str,
    pooling: str,
    split: str,
    current_or_future: str,
    records: List[Dict[str, Any]],
    y_pred: np.ndarray,
):
    """
    Write per-sample predictions with EWS meta to CSV.
    """
    assert len(records) == len(y_pred)
    out_dir = os.path.dirname(out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    common_cols = [
        "model_id",
        "pooling",
        "split",
        "id",
        "year-month",
        "地域",
        "関連",
        "業種・職種",
    ]

    if current_or_future == "current":
        ews_cols = [
            "景気の現状判断",
            "判断の理由",
            "追加説明及び具体的状況の説明",
        ]
    else:
        ews_cols = [
            "景気の先行き判断",
            "景気の先行きに対する判断理由",
        ]

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
                model_id,
                pooling,
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


def linear_probe_for_model(
    model_id: str,
    device: torch.device,
    current_or_future: str,
    max_length: int = 128,
    batch_size: int = 64,
    pooling: str = "cls",
    pred_csv_dir: str = "",
    metrics_json_dir: str = "",
) -> Tuple[str, float, float, Dict[str, Any]]:
    """
    For one HF model_id:
      - snapshot -> local load tokenizer/model
      - encode train/val/test
      - fit logistic regression on train
      - evaluate on val/test
      - optional: write CSV + per-model JSON
    """
    print(f"[INFO] Evaluating HF model: {model_id}")

    tokenizer, local_dir = _load_tokenizer_for_model(model_id)
    model = _load_encoder_model(local_dir, device)

    # Load dataset
    train_records = load_ews("train", current_or_future)
    val_records = load_ews("validation", current_or_future)
    test_records = load_ews("test", current_or_future)

    train_labels = [int(r["y_true"]) for r in train_records]
    val_labels = [int(r["y_true"]) for r in val_records]
    test_labels = [int(r["y_true"]) for r in test_records]

    # Encode
    train_embs = encode_records(model, tokenizer, train_records, device, max_length, batch_size, pooling=pooling)
    val_embs = encode_records(model, tokenizer, val_records, device, max_length, batch_size, pooling=pooling)
    test_embs = encode_records(model, tokenizer, test_records, device, max_length, batch_size, pooling=pooling)

    # Linear classifier
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(train_embs, train_labels)

    val_pred = clf.predict(val_embs)
    test_pred = clf.predict(test_embs)

    # CSV outputs
    if pred_csv_dir:
        model_key = _safe_name(model_id)
        model_dir = os.path.join(pred_csv_dir, f"{model_key}__pool-{pooling}")
        write_split_pred_csv(
            out_csv_path=os.path.join(model_dir, "validation.csv"),
            model_id=model_id,
            pooling=pooling,
            split="validation",
            current_or_future=current_or_future,
            records=val_records,
            y_pred=val_pred,
        )
        write_split_pred_csv(
            out_csv_path=os.path.join(model_dir, "test.csv"),
            model_id=model_id,
            pooling=pooling,
            split="test",
            current_or_future=current_or_future,
            records=test_records,
            y_pred=test_pred,
        )
        print(f"[INFO] Saved prediction CSVs under: {model_dir}")

    # Basic acc
    val_acc = float(accuracy_score(val_labels, val_pred))
    test_acc = float(accuracy_score(test_labels, test_pred))

    # Detailed metrics
    val_metrics = summarize_predictions(val_labels, val_pred, LABEL_NAMES)
    test_metrics = summarize_predictions(test_labels, test_pred, LABEL_NAMES)

    # Log
    print(f"\n[RESULT] model={model_id} pooling={pooling}  (validation)")
    print(f"  overall_acc = {val_metrics['overall_acc']:.4f}")
    print(f"  macro_f1    = {val_metrics['macro_f1']:.4f}")
    print(f"  micro_f1    = {val_metrics['micro_f1']:.4f}")
    print(f"  weighted_f1 = {val_metrics['weighted_f1']:.4f}")
    print(f"  mse         = {val_metrics['mse']:.4f}")
    print(f"  qwk         = {val_metrics['qwk']:.4f}")

    print(f"\n[RESULT] model={model_id} pooling={pooling}  (test)")
    print(f"  overall_acc = {test_metrics['overall_acc']:.4f}")
    print(f"  macro_f1    = {test_metrics['macro_f1']:.4f}")
    print(f"  micro_f1    = {test_metrics['micro_f1']:.4f}")
    print(f"  weighted_f1 = {test_metrics['weighted_f1']:.4f}")
    print(f"  mse         = {test_metrics['mse']:.4f}")
    print(f"  qwk         = {test_metrics['qwk']:.4f}")

    per_model_json = {
        "model_id": model_id,
        "pooling": pooling,
        "current_or_future": current_or_future,
        "max_length": max_length,
        "batch_size": batch_size,
        "validation": val_metrics,
        "test": test_metrics,
    }

    if metrics_json_dir:
        os.makedirs(metrics_json_dir, exist_ok=True)
        out_path = os.path.join(metrics_json_dir, f"{_safe_name(model_id)}__pool-{pooling}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(per_model_json, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved metrics JSON: {out_path}")

    # Cleanup
    del model, clf, train_embs, val_embs, test_embs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return model_id, val_acc, test_acc, per_model_json


def _chunk_round_robin(items: List[str], n: int) -> List[List[str]]:
    chunks = [[] for _ in range(n)]
    for i, x in enumerate(items):
        chunks[i % n].append(x)
    return chunks


def _worker_run_models(
    gpu_id: str,
    model_ids: List[str],
    current_or_future: str,
    max_length: int,
    batch_size: int,
    pooling: str,
    pred_csv_dir: str,
    metrics_json_dir: str,
) -> List[Tuple[str, float, float, Dict[str, Any]]]:
    """
    One GPU (= one process) evaluates assigned model_ids sequentially.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[WORKER gpu={gpu_id}] visible device={device}  num_models={len(model_ids)}")

    out: List[Tuple[str, float, float, Dict[str, Any]]] = []
    for mid in model_ids:
        out.append(
            linear_probe_for_model(
                mid,
                device,
                current_or_future=current_or_future,
                max_length=max_length,
                batch_size=batch_size,
                pooling=pooling,
                pred_csv_dir=pred_csv_dir,
                metrics_json_dir=metrics_json_dir,
            )
        )
    return out


def _select_best_on_validation_models(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Choose best model on validation per metric.
    - max: acc, F1, QWK, per-class acc
    - min: MSE
    """
    best: Dict[str, Any] = {}

    def update_best_max(key: str, model_id: str, value: float):
        cur = best.get(key, None)
        if (cur is None) or (value > cur["value"]):
            best[key] = {"model_id": model_id, "value": float(value)}

    def update_best_min(key: str, model_id: str, value: float):
        cur = best.get(key, None)
        if (cur is None) or (value < cur["value"]):
            best[key] = {"model_id": model_id, "value": float(value)}

    for m in all_metrics:
        model_id = m["model_id"]
        v = m["validation"]

        update_best_max("val_overall_acc", model_id, float(v["overall_acc"]))
        update_best_max("val_macro_f1", model_id, float(v["macro_f1"]))
        update_best_max("val_micro_f1", model_id, float(v["micro_f1"]))
        update_best_max("val_weighted_f1", model_id, float(v["weighted_f1"]))
        update_best_max("val_qwk", model_id, float(v["qwk"]))
        update_best_min("val_mse", model_id, float(v["mse"]))

        for cls in LABEL_NAMES:
            update_best_max(f"val_{cls}_acc", model_id, float(v["per_class"][cls]["acc"]))

    return best


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_ids",
        type=str,
        required=True,
        help='Comma-separated HF model_ids. e.g. "izumi-lab/bert...,rinna/japanese-roberta-base"',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="(when --gpus not set) device string (e.g., cpu, cuda:0)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="max_seq_length for tokenization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size for embedding encoding",
    )
    parser.add_argument(
        "--current_or_future",
        type=str,
        default="current",
        choices=["current", "future"],
        help="EWS split to use: current or future",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="Pooling for sentence embedding: cls or mean",
    )

    # GPU parallel
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help='Comma-separated GPU ids for multiprocessing. e.g. "0,1,2". Empty => single process',
    )

    # prediction CSV output dir
    parser.add_argument(
        "--pred_csv_dir",
        type=str,
        default="",
        help="(optional) output dir for per-sample prediction CSVs",
    )

    # per-model metrics JSON output dir
    parser.add_argument(
        "--metrics_json_dir",
        type=str,
        default="",
        help="(optional) output dir for per-model metrics JSON",
    )

    # summary JSON output path
    parser.add_argument(
        "--metrics_json_path",
        type=str,
        default="./metrics_summary.json",
        help="output path for summary JSON",
    )

    args = parser.parse_args()

    model_ids = [m.strip() for m in args.model_ids.split(",") if m.strip()]
    if not model_ids:
        raise ValueError("Invalid --model_ids")

    print("[INFO] Target HF models:")
    for m in model_ids:
        print(f"  {m}")

    # ---- GPU multiprocessing ----
    if args.gpus.strip():
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if len(gpu_list) == 0:
            raise ValueError("Invalid --gpus")

        model_chunks = _chunk_round_robin(model_ids, len(gpu_list))

        # Use spawn to avoid CUDA+fork issues
        ctx = mp.get_context("spawn")

        with ctx.Pool(processes=len(gpu_list)) as pool:
            jobs = []
            for gpu_id, chunk in zip(gpu_list, model_chunks):
                if not chunk:
                    continue
                jobs.append(
                    pool.apply_async(
                        _worker_run_models,
                        (
                            gpu_id,
                            chunk,
                            args.current_or_future,
                            args.max_length,
                            args.batch_size,
                            args.pooling,
                            args.pred_csv_dir,
                            args.metrics_json_dir,
                        ),
                    )
                )
            nested = [j.get() for j in jobs]

        raw_results = [x for sub in nested for x in sub]

    # ---- Single process (CPU / single GPU) ----
    else:
        device = torch.device(args.device)
        print(f"[INFO] Using device: {device}")

        raw_results: List[Tuple[str, float, float, Dict[str, Any]]] = []
        for mid in model_ids:
            raw_results.append(
                linear_probe_for_model(
                    mid,
                    device,
                    current_or_future=args.current_or_future,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    pooling=args.pooling,
                    pred_csv_dir=args.pred_csv_dir,
                    metrics_json_dir=args.metrics_json_dir,
                )
            )

    # Sort by model_id
    raw_results.sort(key=lambda x: x[0])

    print("\n=== Summary (model_id, val_acc, test_acc) ===")
    for model_id, val_acc, test_acc, _ in raw_results:
        print(f"{model_id}\t{val_acc:.4f}\t{test_acc:.4f}")

    all_metrics = [m for (_, _, _, m) in raw_results]
    best = _select_best_on_validation_models(all_metrics)

    summary = {
        "model_ids": model_ids,
        "current_or_future": args.current_or_future,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "models": all_metrics,
        "best_on_validation": best,
        "simple_table": [
            {
                "model_id": m["model_id"],
                "pooling": m["pooling"],
                "val_overall_acc": float(m["validation"]["overall_acc"]),
                "val_macro_f1": float(m["validation"]["macro_f1"]),
                "val_mse": float(m["validation"]["mse"]),
                "val_qwk": float(m["validation"]["qwk"]),
                "test_overall_acc": float(m["test"]["overall_acc"]),
                "test_macro_f1": float(m["test"]["macro_f1"]),
                "test_mse": float(m["test"]["mse"]),
                "test_qwk": float(m["test"]["qwk"]),
            }
            for m in all_metrics
        ],
    }

    out_dir = os.path.dirname(args.metrics_json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved metrics summary JSON: {args.metrics_json_path}")


if __name__ == "__main__":
    main()
