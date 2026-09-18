# Validation

検証環境：Linux / Python 3.11.15 / CPU。GPU学習、分散学習、実データでの精度再現は未検証です。

| Package | Tested version |
|---|---|
| torch | 2.14.0+cpu |
| transformers | 4.57.6 |
| accelerate | 1.15.0 |
| datasets | 4.8.5 |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 |
| tokenizers | 0.22.2 |
| huggingface-hub | 0.36.2 |
| matplotlib | 3.11.2 |

データなしの公開候補で **27 tests passed** を確認しました。
Chromiumでステップ切替・SVG保存・モバイル幅・JavaScriptエラーなし・外部通信なしも検証済みです。

## 自動検証の範囲

- BERT / RoBERTa / ELECTRA：6文組loss、backward、辞書なしの保存と再読込
- BERT / RoBERTa：CPUでのsentiment MLM、safetensors保存、辞書なし推論
- 学習CLI：人工TSV・ローカルの小型BERT・ネットワーク無効で学習→保存→再開（MLMあり/なし）
- 長さの異なる入力とMLMラベルの左右padding
- 不正な辞書・未知の入力の検出、SentEvalの明示的なデータ指定
- 保存したSentiCSEのMLPを保持した埋め込みAPI
- 公開用書き出し：privateファイル・履歴の除外、既存出力先・パストラバーサル・symlinkの拒否
- 可視化：固定PCA基準・固定文抽出・ステップ別SVG/HTML・RNGとtrain/evalモードの保持・保持枚数上限・再開時のPCA基準/履歴復元・確認用文の変更検出
- 学習CLIに可視化callbackを組み込み、学習途中・終了時の図の生成を確認

CIにはPython 3.11 / 3.12の同じオフラインテストを設定しています。
ローカルで実行確認したのはPython 3.11です。CI結果はGitHub Actionsで確認してください。

## 互換性の変更

古いTrainerのコピーと独自デバイス初期化を削除し、標準Trainer/Accelerateへ移行しました。
学習再開は明示的な `--resume_from_checkpoint`、保存は標準の `checkpoint-N` です。
削除されたscikit-learnの `multi_class` 引数は公開対象の線形プローブから除去しました。
モデルの種類はパス名ではなくconfigから判定し、tokenizer引数の初期化漏れも修正しました。

MLMなしでは辞書の読み込みとマスキング処理をしません。MLMありでは外部辞書を指定します。
既存MLMのランダム置換が `int(0.1)` により無効だった箇所を修正したため、過去と乱数系列・学習結果は一致しません。
最適化・保存・デバイス処理の更新もあるため、以前の実験結果とのビット単位の再現は保証しません。
旧チェックポイントは推論用のモデルとして読み込めますが、旧optimizer状態からの再開は未検証です。

private側の年月日付き・コピー版の実験スクリプトは履歴資料として保持し、公開・移行対象にしていません。
公開対象は `public-files.txt` が基準です。

API参考：[Transformers Trainer](https://huggingface.co/docs/transformers/v4.57.6/en/main_classes/trainer)。

Claude Codeによる静的レビューも実施し、可視化状態をチェックポイントに保存・復元する処理を追加しました。
