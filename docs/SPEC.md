# Live2D Semi-Auto Parts Splitter — Product Specification

Version: 0.2
Status: Web character milestone (updated 2026-10-05)

## 1. Overview

本プロジェクトは、キャラクターイラストからパーツを半自動で作成・編集し、Webアプリへ組み込める動く2Dキャラクターを制作する。

生成・分割・位置合わせは人間が確認・修正できる候補とし、Web上で表情とモーションを再生するまでを現在のゴールとする。

AI・画像処理で素材分けを支援し、パーツの役割と動作を設定して、ブラウザーでまばたき、口、視線、顔向き、呼吸、髪の揺れ、表情・モーションを確認する。組み込み用の画像・設定・JavaScriptを出力する。Cubism SDKや専用形式への互換性は現在のゴールに含めない。

基本思想は以下。

> AIが完成品を決めるのではなく、AIが下処理を行い、人間が最後の判断をする。

---

# 2. Goals

## Primary goal

1枚のキャラクターイラストを入力すると、以下の作業を支援する。

1. キャラクター領域の認識
2. Live2D向けパーツ候補の推定
3. 各パーツのマスク生成
4. 前後関係の推定
5. 隠れている領域の補完
6. 人間によるマスク・分類・前後関係の修正
7. Live2D向けレイヤー素材への変換
8. Webアプリへ組み込めるキャラクターモデルと再生コードの出力

最終的に、Web上で動くキャラクターの制作と組み込みの手間を減らす。

---

# 3. Non-goals

MVPでは以下を対象外とする。

- Live2D Cubismモデルの完全自動生成
- Cubism専用のArtMesh・デフォーマ・パラメータ生成
- 任意のユーザー作成タイムラインや複雑な物理シミュレーション
- VTube Studio向け完成モデル生成
- `.moc3` の直接生成
- 学習用データセットの自動収集
- ユーザー画像を無断で学習に使用すること

Web版では、編集可能なパーツ役割、単純なメッシュ変形、既定の表情・モーション、髪の補助動作を実装する。専用形式の生成や汎用モーション制作ツールは別工程とする。

---

# 4. Target User

主なユーザーは以下。

- Live2Dモデラー
- VTuberモデル制作者
- イラストレーター
- 個人VTuber
- Live2D制作を内製したい小規模チーム

Live2D制作経験が浅いユーザーでも使えることを目指すが、MVPではプロ・準プロによる素材分け支援を優先する。

---

# 5. Core Workflow

## Step 1 — Import

ユーザーがキャラクター画像を読み込む。

MVP対応候補:

- PNG
- JPEG
- WebP

将来:

- PSD
- CLIP STUDIO形式

読み込み時に以下を取得する。

- canvas width
- canvas height
- alpha channel
- color profile
- file hash
- source filename

元画像は変更しない。

---

## Step 2 — Character Analysis

画像からキャラクター構造を解析する。

検出候補:

### Head

- face
- ear_left
- ear_right
- neck

### Hair

- hair_back
- hair_front
- bangs
- hair_side_left
- hair_side_right
- hair_strand_*

### Eyes

左右を分離する。

- eye_white_L
- eye_white_R
- iris_L
- iris_R
- pupil_L
- pupil_R
- eye_highlight_L
- eye_highlight_R
- eyelash_upper_L
- eyelash_upper_R
- eyelash_lower_L
- eyelash_lower_R

### Face features

- eyebrow_L
- eyebrow_R
- nose
- mouth
- blush_L
- blush_R

### Body

- torso
- shoulder_L
- shoulder_R
- arm_L
- arm_R
- hand_L
- hand_R

### Clothing

固定カテゴリだけでなく、可変個数のパーツを許可する。

例:

- clothes_base
- collar
- ribbon
- sleeve_L
- sleeve_R

### Accessories

可変個数。

- accessory_001
- accessory_002
- ...

カテゴリの追加・変更はユーザーが行えること。

---

# 6. Segmentation

AIまたは画像処理によってパーツ候補ごとのマスクを作成する。

ただしAI推論結果は常に「候補」として扱う。

ユーザーは以下を変更できる。

- mask
- part type
- part name
- layer order
- parent group
- visibility
- merge
- split

必須編集機能:

- brush add
- brush erase
- lasso add
- lasso erase
- undo
- redo
- zoom
- pan
- mask visibility toggle
- solo part
- overlay opacity

---

# 7. Occlusion Handling

Live2D素材分けでは、元画像で隠れている領域を復元する必要がある。

例:

- 前髪の裏にある額
- 髪の裏にある顔輪郭
- 腕の裏にある胴体
- 服の裏にある身体
- 前髪の裏にある眉
- まぶたに隠れている眼球
- アクセサリーの裏側

システムは各パーツについて、

- visible region
- estimated hidden region

を区別する。

隠れ領域の補完は別工程として扱う。

---

# 8. Hidden Region Repair / Inpainting

補完処理には以下のモードを用意する。

## Manual

ユーザーが直接描画・修正する。

## Assisted

周辺画素や形状から画像処理で補完する。

## AI assisted

画像生成・inpaintingモデルを使用する。

重要:

AI補完結果を自動採用しない。

必ずユーザーが確認できる状態にする。

Original / Generated の比較を可能にする。

再生成可能にする。

---

# 9. Layer Ordering

各パーツについて z-order を管理する。

例:

hair_back  
↓  
body  
↓  
face  
↓  
eyes  
↓  
bangs  
↓  
accessories

自動推定を行ってよいが、ユーザーがドラッグ操作等で変更できること。

内部データでは明示的な整数または安定した順序として保存する。

---

# 10. Recomposition Preview

分割後の各パーツを再合成し、元画像と比較できること。

表示モード:

- Original
- Reconstructed
- Difference
- Split overlay
- Individual layer

目的は、分割によって見た目が変化していないことを確認するため。

可視領域については元画像を最大限保持する。

AI生成を使用するのは、原則として元画像に存在しない隠れ領域のみとする。

---

# 11. Simple Separation Test

本格的なLive2Dリギングを行う前に、素材分けが十分か確認する簡易テストを提供する。

例:

- 目を左右へ数px移動
- 前髪を左右へ移動
- 顔を少し移動
- 腕を移動
- パーツを一時的に非表示

これによって、

- 穴が見える
- 背景が露出する
- 隠れ部分の塗りが足りない
- パーツ境界がおかしい

といった問題を早期発見する。

これはLive2Dリギングではなく素材検証機能である。

---

# 12. Project File

編集状態を独自プロジェクトとして保存できるようにする。

仮拡張子:

`.l2split`

初期実装では実体をJSON + assets directoryとしてもよい。

最低限保存する情報:

- source image
- source hash
- canvas size
- parts
- masks
- layer order
- hierarchy
- generation history
- manual edits
- export settings
- application version

データ形式にはバージョン番号を持たせる。

---

# 13. Output

## Required output

最低限以下を出力できること。

### PNG package

```text
output/
  parts/
    face.png
    hair_back.png
    hair_front.png
    eye_white_L.png
    eye_white_R.png
    ...
  manifest.json
  preview.png
```

PNGは透過背景を持つ。

すべて元キャンバスと同一サイズで出力できるモードを用意する。

---

## Live2D PSD

PSDは任意の素材交換用出力とする。Webモデルの完成判定には使用しない。

出力PSDでは以下を満たすこと。

- RGB
- 8bit/channel
- sRGB
- パーツごとに独立したレイヤー
- 同一レイヤー名を作らない
- 安定したレイヤー名
- 適切なレイヤー順
- 不要なレイヤーマスクに依存しない
- Cubismで扱えない特殊なレイヤー構造へ依存しない

PSD生成方法は実装技術に依存するため、Exporter interfaceの背後に隠蔽する。

直接PSD生成が不安定な場合でも、コアロジックをPSDライブラリへ密結合させない。

## Web用モデル

必須出力は、独立した透過パーツ画像、安定IDと役割・前後関係・基準座標を持つ `model.json`、ブラウザー用の再生JavaScript、動作例のHTMLを含むZIPとする。

エディターサーバーを起動しなくても、書き出した動作例で再生できること。別のWebページでは同じ再生コードへCanvasと設定を渡して組み込めること。パラメータで視線・顔向き・まばたき・口・呼吸を制御し、表情と既定モーションを切り替えられること。

素材が持たない形状や隠れ領域の完成は保証しない。動作中に欠けを検査し、マスク・補完・位置合わせ・パーツ役割を修正できること。

---

# 14. Naming Convention

内部IDと表示名を分離する。

例:

```json
{
  "id": "part_01J...",
  "type": "eye_white",
  "side": "left",
  "name": "eye_white_L"
}
```

IDは変更しない。

ユーザーがレイヤー名を変更しても内部参照が壊れない設計にする。

左右は内部では `left` / `right` を使用する。

---

# 15. Architecture

推奨初期構成:

```text
src/
  app/
  core/
    project/
    parts/
    masks/
    compositing/
    validation/
  inference/
    segmentation/
    inpainting/
  exporters/
    png/
    psd/
  ui/
  infrastructure/

tests/
docs/
scripts/
```

---

# 16. Preferred Initial Technology

MVPではPythonを第一候補とする。

推奨:

- Python 3.12+
- PySide6
- NumPy
- Pillow
- OpenCV
- Pydantic
- pytest

AI推論を導入する場合:

- PyTorch

ただしAIモデルをcoreへ直接依存させない。

以下のような抽象化を置く。

```python
class SegmentationBackend:
    def segment(self, image, request):
        ...
```

```python
class InpaintingBackend:
    def inpaint(self, image, mask, context):
        ...
```

これによって、

- CPU実装
- GPU実装
- ローカルAI
- 外部API
- 将来モデル

を交換可能にする。

---

# 17. GPU Policy

GPUは必須にしない。

最低限:

- UI
- project editing
- manual mask editing
- PNG export

はCPUだけで動作すること。

AI推論のみGPUを利用してよい。

GPUがない場合は機能を無効化またはCPUへフォールバックする。

---

# 18. AI Model Policy

特定の巨大モデルをプロジェクトの前提にしない。

初期段階では、

1. inference interface
2. test backend
3. manual workflow
4. lightweight baseline

を先に実装する。

その後モデルを比較する。

評価項目:

- anime illustration accuracy
- VRAM requirement
- inference speed
- license
- commercial use
- offline availability
- model size
- maintenance
- mask quality

モデルの自動ダウンロードはユーザーの明示操作なしに行わない。

---

# 19. Privacy

キャラクター画像はユーザー資産として扱う。

デフォルトではローカル処理を優先する。

外部APIへ画像を送信する場合は、

- 送信先
- 送信内容
- 使用目的

を明示し、ユーザー操作なしに送信しない。

入力画像を学習データとして保存しない。

---

# 20. Quality Requirements

## Non-destructive

元画像を上書きしない。

## Deterministic project state

保存・再読み込みで編集状態が変化しない。

## Recoverable

AI処理の失敗によって手動編集結果を失わない。

## Incremental

1パーツだけ再処理できる。

全画像を毎回再推論する必要がない設計を目指す。

## Reversible

AI結果、手動修正、生成結果を可能な限り取り消せる。

---

# 21. MVP

MVPの完成条件を以下とする。

### Import

- PNG/JPEGを読み込める

### Parts

- パーツを手動作成できる
- segmentation backendからマスク候補を受け取れる
- パーツ名を変更できる
- パーツ種類を変更できる
- layer orderを変更できる

### Mask editing

- brush add/erase
- undo/redo
- zoom/pan

### Preview

- individual part
- overlay
- reconstructed image
- source comparison

### Project

- save
- load

### Export

- transparent PNG layers
- manifest.json

### Validation

- duplicate names
- empty masks
- canvas mismatch
- invalid layer order
- missing project assets

を検出する。

## 現在のWebマイルストーン

- ブラウザーで画像／プロジェクトを読み込み、既存の手動編集・候補採用・補完・保存・出力を操作できる。
- 分割パーツをCanvas上のメッシュ変形と階層変換で動かす。
- まばたき、口開閉、視線、顔向き、呼吸、髪の補助動作を制御できる。
- 笑顔・怒り・泣き・驚き・照れを含む表情プリセットと、待機・うなずき・首振り・挨拶を含むモーションを再生できる。素材にない部位は無理に新規生成しない。
- パーツ役割を修正し、設定を保存・再読込できる。
- 組み込み用WebモデルZIPを書き出し、エディターから独立したページで再生できる。
- 実ブラウザーで編集、候補確認、表情・動作、書き出しと再生を検証する。
- 原画は上書きせず、外部AIへの送信には明示操作を必要とする。APIキーを素材・設定・ログへ保存しない。

---

# 22. Post-MVP

MVP後、以下を順次検討する。

## Phase 2

- automatic part detection
- anime-specific segmentation
- automatic hierarchy estimation
- automatic z-order estimation

## Phase 3

- hidden-region detection
- AI inpainting
- before/after comparison
- regeneration controls

## Phase 4

- Cubism-compatible PSD export
- PSD preflight validator

## Phase 5

- eye/mouth specific preprocessing
- simple separation/deformation test
- batch operations

## Future

- pose estimation
- Live2D parameter assistance
- ArtMesh assistance
- automatic deformer suggestions
- Cubism plugin/integration

---

# 23. Acceptance Criteria

MVPを完了とみなすためには以下を満たす。

1. 1枚のキャラクター画像を読み込める。
2. 複数の独立したLive2D候補パーツへ分けられる。
3. マスクを人間が修正できる。
4. レイヤー順を人間が変更できる。
5. 元画像と再合成画像を比較できる。
6. プロジェクトを保存して再開できる。
7. 全パーツを透過PNGとして出力できる。
8. 出力後に元画像を再構成できる。
9. AIが利用できなくても手動編集機能は使用できる。
10. AIの推論結果をユーザーが拒否・修正できる。

PSD exportはMVP直後の最優先マイルストーンとする。

---

# 24. Product Principles

実装判断に迷った場合は以下の順で優先する。

1. 原画を壊さない
2. 人が修正できる
3. Live2D制作工程で実際に使える
4. 処理をやり直せる
5. AIモデルを交換できる
6. ローカルで動作できる
7. 自動化率を高める

「完全自動だが修正不能」より、

「80%自動で、残り20%を素早く直せる」

方を正解とする。
