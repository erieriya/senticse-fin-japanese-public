# private / public の分離

## 役割

- `senticse-fin-Japanese`：private。開発元、既存の実験資産・データ・履歴を保持する。
- `senticse-fin-japanese-public`：public。レビュー済みのコードとドキュメントだけを配布する。

privateのブランチをそのままpublicにpushしないでください。
Gitのブランチはデータのアクセス権を分離せず、過去コミットにデータが残るためです。
publicの初回公開は、許可リストから書き出したファイルだけで新しいGit履歴を作ります。

コード変更は原則private側で行い、公開対象の変更をpublicへ反映します。
publicで変更した場合は先に同じコード変更をprivateへ取り込んでから、次の書き出しを行います。
双方向の自動同期は行いません。

## 公開対象を作る

privateリポジトリで実行します。書き出し先は存在しないディレクトリを指定します。

```bash
python scripts/export_public.py ../release-candidate
python scripts/export_public.py ../release-candidate --check
```

`public-files.txt` はファイル単位の許可リストです。globやディレクトリ一括指定は使いません。
書き出しはGit履歴・未指定ファイルをコピーしません。シンボリックリンクやデータ拡張子も拒否します。
`--check` はファイル集合と内容を開発元と比較します。検査時の `.git/` は無視します。

公開前には候補ディレクトリ内でテストし、差分を確認してください。
追加したコードやMarkdownに秘密情報や実データを直接埋め込んだ場合までは、許可リストでは検出できません。
そのため公開対象のソース内容もレビューが必要です。

## 初回公開

レビュー・テストを済ませた候補だけで実行します。

```bash
cd ../release-candidate
git init -b main
git add .
git commit -m "Publish data-free Japanese financial SentiCSE code"
git remote add origin https://github.com/erieriya/senticse-fin-japanese-public.git
git push -u origin main
```

## 2回目以降

新しい候補を別の空ディレクトリに書き出します。publicリポジトリを別途cloneし、候補とのソース差分を確認して反映します。
許可リストから外したファイルの削除も反映し、private側からそのpublic作業ツリーを `--check` で照合します。
テストが作るキャッシュや出力は、照合前に作業ツリーの外へ移してください。
publicの既存履歴は保持し、通常のコミットで更新します。privateの履歴をmergeしません。

## データと認証情報

実データ、感情辞書、重み、tokenizerのコピー、ログ、グラフ、ノートブック、生成レポートは公開対象外です。
公開版にサンプルデータファイルは置かず、デモとテストの人工データを実行時に生成します。
入力は `--train_file` / `--sentiment_vocab_file` / `--senteval_data_dir` で指定します。

`.gitignore` は新規の混入を防ぎますが、private側で既に追跡済みのデータには作用しません。
既存データを削除・履歴改変する必要はありません。privateを維持してください。
PATなどの認証情報はGitのcredential helperやGitHub CLIで管理し、ファイルに含めません。
