# Live2D Semi-Auto Parts Splitter

1枚のキャラクター画像を、人が修正できるマスクでLive2D向けの素材へ分けるデスクトップツールです。
現在は手動編集からPNG出力までの初期実装です。完全自動モデル生成やリギングは行いません。

## 開発・起動

Python 3.12以上と [uv](https://docs.astral.sh/uv/) を使用します。

```bash
uv sync --locked
uv run --locked live2d-semi-auto
```

クラウド環境では、リポジトリに移動して次を実行できます。

```bash
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PROJECT_ENVIRONMENT=/workspace/.venvs/live2d-semi-auto
export XDG_CACHE_HOME=/workspace/.cache
uv sync --locked
QT_QPA_PLATFORM=offscreen uv run --locked pytest -q
```

`offscreen` はテスト用です。GUIを操作するには、ローカルのWindows・macOS・Linuxデスクトップなどの画面環境が必要です。クラウドからの画面配信機能は含みません。

## 操作

1. 「画像を読み込む」でPNG・JPEG・WebPを開き、「パーツを追加」を押します。
2. 名前・種類・表示状態を設定し、「属性を適用」を押します。
3. 「マスク編集・オーバーレイ」でブラシ／なげなわの追加・消去を選び、左ドラッグで編集します。半径は原画のピクセル単位です。
4. ホイールで拡大縮小、中ボタンドラッグで移動します。「全体表示」は `Ctrl+0` です。
5. 「選択パーツ」「再合成」「差分」「原画」で結果を確認します。リストの下ほど手前のレイヤーです。マスク濃度を0にするとオーバーレイが消えます。
6. `Ctrl+S` で `.l2split` に保存し、`Ctrl+O` で再開します。原画とマスクを含むため、元画像が移動しても再開できます。
7. 「PNG出力」で親フォルダーと**新しい**出力フォルダー名を指定します。透過PNG、`manifest.json`、`preview.png` が作られます。既存の出力には上書きしません。

Undoは `Ctrl+Z`、Redoは `Ctrl+Shift+Z` です。パーツ作成・削除・属性・順序変更・マスク編集を取り消せます。履歴はセッション内の直近50操作です。

「不透明領域のマスク候補」は非透明画素を選択するローカルの簡易バックエンドです。確認ダイアログで採用するまで編集状態は変わりません。背景を含む不透明画像では全画像が候補になります。意味的な髪・顔などの自動分割は未実装です。

空マスクは作業途中の保存には使用できますが、PNG出力ではエラーになります。名前の重複、キャンバスの不一致なども検出します。

## テスト・サンプル

```bash
uv run --locked pytest -q
```

GUIテストは画面のない環境でも実行します。小さな合成画像でコア処理を検証し、[samples/original_character.png](samples/original_character.png) でも保存・再合成・PNG出力を確認します。このピンク髪のサンプルはAIで生成したオリジナルキャラクターです。出典は [samples/README.md](samples/README.md) を参照してください。

## 範囲と設計

- 原画の上書きや外部への画像送信は行いません。GPU・AIモデルのダウンロードは不要です。
- PNG出力は原画と同じキャンバス寸法です。原画のICCプロファイルを保持します。PSD出力やCubismでの互換性検証は未実装です。
- UIとコア、保存処理、推論バックエンドを分離しています。[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) に座標系・保存形式・制限を記載しています。
- 製品仕様は [docs/SPEC.md](docs/SPEC.md)、実装ルールは [AGENTS.md](AGENTS.md) を参照してください。

## プロジェクトの目標

キャラクター画像からパーツ候補を作成し、マスクと前後関係を人間が修正し、隠れ領域の補完を確認して、PNG・将来的なCubism互換PSD素材として出力することを目指します。原画と再合成の比較、プロジェクトの保存・再開を重視します。

完全自動のLive2Dモデル生成、自動リギング、モーション生成、VTube Studio向け完成モデルや `.moc3` の直接生成は対象外です。AI推論はコアから分離し、バックエンドを交換できる構成です。
