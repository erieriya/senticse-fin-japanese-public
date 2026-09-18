"""Offline regression tests: synthetic tokens only; no private data or downloads."""
import csv
import json
import re
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, RobertaConfig, ElectraConfig, TrainingArguments

from senticse.data import ContrastiveCollator, load_sentiment_vocab
from senticse.models import BertForCL, RobertaForCL, ElectraModelForCL
from senticse.trainers import CLTrainer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nneutral\nnews\n")
    return BertTokenizerFast(vocab_file=str(vocab))


def config(cls=BertConfig):
    return cls(vocab_size=9, hidden_size=16, num_hidden_layers=1,
               num_attention_heads=2, intermediate_size=24, pad_token_id=0,
               embedding_size=16)


@pytest.mark.parametrize("model_cls,config_cls", [(BertForCL, BertConfig), (RobertaForCL, RobertaConfig), (ElectraModelForCL, ElectraConfig)])
def test_forward_backward_and_reload_without_private_assets(tmp_path, monkeypatch, model_cls, config_cls):
    monkeypatch.chdir(tmp_path)
    model = model_cls(config(config_cls))
    ids = torch.randint(5, 9, (2, 6, 5))
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    model.save_pretrained(tmp_path / "checkpoint")
    reloaded = model_cls.from_pretrained(tmp_path / "checkpoint").eval()
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=ids[:, 0], attention_mask=torch.ones_like(ids[:, 0]), sent_emb=True).pooler_output
        actual = reloaded(input_ids=ids[:, 0], attention_mask=torch.ones_like(ids[:, 0]), sent_emb=True).pooler_output
    torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("side", ["left", "right"])
def test_dynamic_padding_mlm(tokenizer, side):
    tokenizer.padding_side = side
    features = [{"input_ids": [[2, 5, 3], [2, 6, 8, 3]],
                 "attention_mask": [[1, 1, 1], [1, 1, 1, 1]],
                 "mlm_input_ids": [[2, 4, 3], [2, 4, 8, 3]],
                 "mlm_labels": [[-100, 5, -100], [-100, 6, -100, -100]]}]
    batch = ContrastiveCollator(tokenizer)(features)
    assert batch["input_ids"].shape == (1, 2, 4)
    padding_position = 0 if side == "left" else -1
    assert batch["mlm_labels"][0, 0, padding_position] == -100
    assert batch["mlm_input_ids"][0, 0, padding_position] == 0


@pytest.mark.parametrize("model_cls,config_cls", [(BertForCL, BertConfig), (RobertaForCL, RobertaConfig)])
def test_mlm_cpu(tmp_path, model_cls, config_cls):
    vocab = np.zeros((3, 9))
    vocab[0, 5:] = 1
    vocab[1, 5:7] = 1
    vocab[2, 7:] = 1
    path = tmp_path / "sentiment.npy"
    np.save(path, vocab)
    model = model_cls(config(config_cls), model_args=SimpleNamespace(do_mlm=True, sentiment_vocab_file=str(path)))
    ids = torch.randint(5, 9, (2, 6, 5))
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), mlm_input_ids=ids, mlm_labels=ids)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert "sentiment_vocab_file" not in model.config.senticse_args
    model.save_pretrained(tmp_path / "mlm-checkpoint")
    reloaded = model_cls.from_pretrained(tmp_path / "mlm-checkpoint")
    assert reloaded.model_args.do_mlm
    assert reloaded.positive_score is None
    assert torch.isfinite(reloaded(input_ids=ids[:, 0], attention_mask=torch.ones_like(ids[:, 0]), sent_emb=True).pooler_output).all()


def test_vocab_validation(tmp_path):
    with pytest.raises(ValueError, match="requires"):
        load_sentiment_vocab(None, 9)
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="shape"):
        load_sentiment_vocab(path, 9)


@pytest.mark.parametrize("mlm", [False, True])
def test_cli_train_and_resume_without_network(tmp_path, tokenizer, mlm):
    base = tmp_path / "arbitrary-model-name"
    tokenizer.save_pretrained(base)
    BertModel(config()).save_pretrained(base)
    data = tmp_path / "synthetic.tsv"
    with data.open("w") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["pos1", "pos2", "neu1", "neg1", "neg2", "neu2"])
        writer.writerows([["good news", "good", "neutral", "bad news", "bad", "neutral news"]] * 4)
    plot_file = tmp_path / "plot.csv"
    plot_file.write_text("text,label\ngood news,2\nbad,0\nneutral news,1\n")
    output = tmp_path / "output"
    command = [sys.executable, str(ROOT / "senticse.py"), "--model_name_or_path", str(base),
               "--cache_dir", str(tmp_path / "cache"), "--train_file", str(data), "--output_dir", str(output), "--do_train",
               "--use_cpu", "--per_device_train_batch_size", "2", "--save_steps", "1",
               "--max_seq_length", "8", "--pad_to_max_length", "false", "--report_to", "none",
               "--evaluation_strategy", "no", "--disable_tqdm", "true",
               "--embedding_plot_file", str(plot_file), "--embedding_plot_steps", "1"]
    if mlm:
        vocab = np.zeros((3, 9)); vocab[0, 5:] = 1; vocab[1, 5:7] = 1; vocab[2, 7:] = 1
        path = tmp_path / "sentiment.npy"; np.save(path, vocab)
        command += ["--do_mlm", "--sentiment_vocab_file", str(path), "--sentimlm_probability", "1"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", MPLCONFIGDIR=str(tmp_path / "mpl-cache"))
    def run(extra):
        result = subprocess.run(command + extra, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    run(["--max_steps", "1"])
    assert (output / "checkpoint-1" / "trainer_state.json").is_file()
    run(["--max_steps", "2", "--resume_from_checkpoint", str(output / "checkpoint-1")])
    assert (output / "checkpoint-2" / "trainer_state.json").is_file()
    assert (output / "embeddings.html").is_file()
    assert (output / "embeddings.svg").is_file()
    assert (output / "checkpoint-2" / "embedding-plot.npz").is_file()
    html = (output / "embeddings.html").read_text()
    payload = json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert [frame["step"] for frame in payload["frames"]] == [0, 1, 2]


def test_senteval_requires_explicit_data(tmp_path):
    trainer = CLTrainer(model=BertForCL(config()), args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none"))
    with pytest.raises(ValueError, match="senteval_data_dir"):
        trainer.evaluate()


def test_embedding_api_preserves_custom_pooler(tmp_path, tokenizer):
    from senticse import SimCSE
    model = BertForCL(config()).eval()
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    encoder = SimCSE(str(tmp_path / "model"), device="cpu")
    inputs = tokenizer(["good news"], return_tensors="pt")
    with torch.no_grad():
        expected = model(**inputs, sent_emb=True).pooler_output
    actual = encoder.encode(["good news"], normalize_to_unit=False)
    torch.testing.assert_close(expected, actual)


def test_senteval_strategy_initialization(tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", eval_strategy="steps")
    trainer = CLTrainer(model=BertForCL(config()), args=args, senteval_data_dir=str(tmp_path))
    assert trainer.eval_dataset is None
