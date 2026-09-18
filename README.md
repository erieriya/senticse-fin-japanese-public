# SentiCSE for Japanese Financial Sentiment

日本語の金融・景況感テキストを対象にした、感情を考慮する文埋め込みの研究コードです。
肯定・中立・否定の **6文組による対照学習**、線形プローブ評価、インタラクティブな可視化を提供します。
元の研究と引用情報は [SentiCSE_README.md](SentiCSE_README.md) を参照してください。

この公開版には、実データ・感情辞書・学習済みモデル・実験ログ・非公開リポジトリの履歴を含めません。
デモとテストは人工データを実行時に生成します。研究上の性能を示すものではありません。

## セットアップ

Python **3.11** を推奨します。検証済みの API 基準は Transformers **4.57.6** です。
旧版の Python 3.8 / Transformers 4.2.1 / CUDA 11.3 固定環境は不要です。
Transformers 5 はこのリリースの対応範囲外です。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPUで試す場合。GPUの場合は環境に合うPyTorchを先にインストールしてください。
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

依存範囲は `requirements.txt`、今回のLinux / Python 3.11 / CPUの検証環境は
[docs/validation.md](docs/validation.md) に記載しています。
既存環境への上書きではなく、新しい仮想環境を使用してください。

## 学習中の埋め込み分布を見る

学習前・途中・終了時に、**同じ文**の埋め込みを2次元で確認できます。
最初のスナップショットでPCAの基準を作り、その後は同じ軸へ投影します。
ステップごとにPCAを再学習しないため、軸の回転による見かけの変化を避けられます。

```bash
# 人工データで表示だけ試す（モデルの学習結果ではありません）
python -m senticse.visualization --demo --output outputs/embedding-demo.html
```

HTMLをブラウザで開き、スライダーでステップを切り替えます。
色は否定・中立・肯定です。各ステップの散布図はSVGとして保存できます。
Webサーバー・外部通信は不要です。

実際の学習では、観察する文を `text,label` 列の外部CSVで用意します。
ラベルは `0=否定、1=中立、2=肯定`。学習に使っていない固定の確認用文を使うと、変化を追いやすくなります。
学習コマンドに次の引数を追加してください。

```bash
--embedding_plot_file /path/to/private/plot_sentences.csv \
--embedding_plot_steps 250 --embedding_plot_max_samples 300
```

`output_dir/embeddings.html` と、最後の図の `embeddings.svg` を更新します。
最初・指定ステップ間隔・終了時に、チェックポイントのpooling（MLPを含む設定）で埋め込みを計算します。
300文を超える場合は固定seedで同じ文を抽出します。
既定では最初と直近の計30スナップショットを保持し、`--embedding_plot_max_frames` で変更できます。
PCAの基準・固定文の識別情報・表示履歴をチェックポイントの `embedding-plot.npz` に保存します。
学習再開時も同じCSV・tokenizer・最大長を指定すると、同じ軸と履歴を復元します。
設定が変わっていればエラーにし、比較できない図が混ざるのを防ぎます。
可視化状態がない旧チェックポイントから始める場合は、再開時点を基準に
`embeddings-from-step-N.html` / `.svg` を別途作成します。

L2正規化後のPCAです。初期PCAの軸外で起きる変化は見えないため、図だけで性能を判断しないでください。
軸範囲はHTML内の全ステップで共通です。本文や元データのIDは含みませんが、実験由来の図はprivate側で管理します。
この機能は単一プロセスのCPU / GPU学習向けで、GPUは未実機検証です。分散学習・FSDP・DeepSpeedとの併用には対応していません。

## 学習

外部CSV/TSVの先頭6列を、次の順番に並べます。

| 列 | 内容 |
|---|---|
| 1・2 | 肯定の文 |
| 3 | 中立の文 |
| 4・5 | 否定の文 |
| 6 | 中立の文 |

ヘッダーが必要です。JSON/JSONLとHugging Faceの `--dataset_name` も同じ6列構成を使用します。
実データはリポジトリの外、またはGit管理外の `private/` に置いてください。

```bash
python senticse.py \
  --model_name_or_path rinna/japanese-roberta-base \
  --train_file /path/to/private/train_6tuple.csv \
  --output_dir outputs/experiment \
  --do_train --eval_strategy no --report_to none \
  --per_device_train_batch_size 16 --max_seq_length 64 \
  --max_steps 1000 --save_steps 250 --save_total_limit 3 \
  --learning_rate 1e-5 --pooler_type cls --temp 0.05 \
  --pos_neu_weight 1.0 --pos_neg_weight 2.0 --neu_neg_weight 1.5
```

CPUを明示する場合は `--use_cpu`、対応GPUで混合精度を使う場合は `--fp16` を追加します。
最初の実行ではモデル・tokenizerをHugging Faceから取得します。
学習ループ・デバイス管理・保存・再開は標準の `Trainer` / Accelerate に委ねています。
旧引数 `--evaluation_strategy` は互換用に受け付けます。未知の引数はエラーになります。

### Sentiment MLM

既定はMLMなしで、辞書ファイルは不要です。MLMありの場合だけ追加してください。

```bash
--do_mlm --mlm_weight 0.15 \
--sentiment_vocab_file /path/to/private/tokenizer_sentiment.npy
```

辞書は、**使用するtokenizerと同じtoken ID**に対応する `(3, vocab_size以上)` の数値配列です。
行は感情語フラグ（0/1）、肯定スコア、否定スコアの順です。公開版には同梱しません。
BERT / RoBERTaで対応し、ELECTRAではMLMなしを使用します。

### 再開・評価

同じ学習引数に `--resume_from_checkpoint outputs/experiment/checkpoint-250` を追加して再開します。
単にローカルのモデルを指定しただけでは、自動的に学習再開しません。
チェックポイントは `checkpoint-N` に標準形式で保存されます。
従来の「bestだけを出力ルートに保存」とは異なります。

```bash
python linear_probe_single_ckpt.py \
  --checkpoint_dir outputs/experiment/checkpoint-1000 \
  --tokenizer_dir outputs/experiment/checkpoint-1000 \
  --current_or_future current --device cpu
```

この評価は実行時にEWSデータセットを取得します。データ自体は公開版に含めません。
既存実験との比較のため既定は `--pooling cls_before_pooler`（encoderのCLS）です。
学習済みMLPを含むSentiCSEのpoolingを評価する場合は `--pooling senticse` を明示してください。
公開HFモデルの比較には `linear_probe_eval_hfmodels.py` を使えます。

旧SentEval評価は `--senteval_data_dir /path/to/tasks` で外部データを指定します。
SentEvalのタスク一式を使う評価・GPU転移分類・既存実験の性能再現は今回の検証対象外です。

## テスト

```bash
python -m pip install -r requirements-dev.txt
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q
```

小さなランダム初期化モデルと人工データを使い、学習・勾配・保存・再開・MLM・辞書なし推論・公開書き出しを確認します。
モデルや実データのダウンロードは不要です。

## private / public の運用

[docs/public-release.md](docs/public-release.md) を参照してください。
公開対象は `public-files.txt` の明示リストだけです。新しい実験ファイルが自動で公開されることはありません。

## 出典

このコードは [SentiCSE](https://github.com/nayohan/SentiCSE) を基にしています。
同梱のSentEvalには [SentEval/LICENSE](SentEval/LICENSE) が適用されます。
既存の著作権表示と引用情報は保持しています。
