# 日本語金融テキスト向け SentiCSE

このリポジトリは、日本人工知能学会全国大会2026の発表「経済ドメインにおける3値センチメントに基づく埋め込みモデルの検討」で使った実験コードを改良したものです。景気ウォッチャー調査のコメントを対象に、肯定・中立・否定の3値センチメントを捉える文埋め込みを学習します。[SentiCSE](https://github.com/nayohan/SentiCSE) の対照学習を拡張し、同じラベルの文を近づけ、異なるラベルの文を離す手法です。

このリポジトリには、学習データ、感情語辞書、学習済みモデル、実験結果は含まれていません。データは利用者が用意し、ファイルのパスを指定してください。

## セットアップ

Python 3.11 と Transformers 4.57.6 で動作を確認しています。以下は CPU で試す手順です。GPU を使う場合は、環境に合った PyTorch を先にインストールしてください。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## 学習

ヘッダー付きの学習用 CSV を用意し、先頭 6 列に `pos1,pos2,neu1,neg1,neg2,neu2` の順で文を並べます。肯定・中立・否定から各 2 文を使います。

```bash
python senticse.py \
  --model_name_or_path rinna/japanese-roberta-base \
  --train_file /path/to/train_6tuple.csv \
  --output_dir outputs/experiment \
  --do_train --eval_strategy no \
  --per_device_train_batch_size 16 --max_seq_length 64 \
  --max_steps 1000 --save_steps 250 \
  --learning_rate 1e-5 \
  --pos_neu_weight 1.0 --pos_neg_weight 2.0 --neu_neg_weight 1.5
```

このコマンドは動作例であり、発表の実験条件を再現するものではありません。モデルは初回実行時に Hugging Face から取得します。学習を再開するときは、同じコマンドに `--resume_from_checkpoint outputs/experiment/checkpoint-250` を追加してください。

感情語を使った MLM は標準では無効です。有効にする場合は、使用するトークナイザーのトークン ID に対応した `.npy` 辞書を用意し、`--do_mlm --sentiment_vocab_file /path/to/sentiment_vocab.npy` を追加します。辞書は感情語フラグ、肯定スコア、否定スコアを行に持つ配列です。

## 学習中の埋め込みを見る

学習中に同じ文の埋め込みを記録し、2 次元の散布図で比較できます。`text,label` 列を持つ CSV に観察用の文を入れ、学習コマンドに次を追加してください。ラベルは `0` が否定、`1` が中立、`2` が肯定です。

```bash
--embedding_plot_file /path/to/plot_sentences.csv \
--embedding_plot_steps 250
```

学習を始めた時点の埋め込みで PCA を計算し、以後の記録も同じ軸へ投影します。`outputs/experiment/embeddings.html` をブラウザで開くと、スライダーで学習ステップを切り替えられます。同じ観察用 CSV を指定して学習を再開すると、図の軸と記録済みのステップを引き継ぎます。

学習せずに表示だけを試す場合は、人工データのデモを実行します。この図はモデルの学習結果ではありません。

```bash
python -m senticse.visualization --demo --output outputs/demo.html
```

## 評価と開発

学習したモデルの線形プローブ評価には `linear_probe_single_ckpt.py` を使います。実行時に Economy Watchers Survey のデータを取得します。引数は `python linear_probe_single_ckpt.py --help` で確認できます。

テストは人工データと小さなモデルで実行できます。

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

元の研究の説明と引用情報は [SentiCSE_README.md](SentiCSE_README.md) を参照してください。
