# Live2D Semi-Auto Parts Splitter

1枚のキャラクターイラストから、Live2D Cubism向けのパーツ分け素材を半自動で作成するためのツールです。

完全自動でLive2Dモデルを生成することではなく、

**AIや画像処理で下処理を行い、人間が確認・修正しながらLive2D用素材を効率よく作ること**

を目的としています。

## Project Status

🚧 Early development / MVP planning

現在は初期設計・MVP構築段階です。

## Goals

主な目標は以下です。

- キャラクター画像の読み込み
- Live2D向けパーツ候補の作成
- マスクの編集
- パーツの前後関係の調整
- 隠れた領域の補完支援
- 元画像と分割後素材の比較
- プロジェクトの保存・再開
- Live2D向け透過PNG素材の出力
- 将来的なCubism互換PSD出力

## Non-goals

MVPでは以下を対象外とします。

- Live2Dモデルの完全自動生成
- 自動リギング
- 自動モーション生成
- VTube Studio向け完成モデルの直接生成
- `.moc3` の直接生成

## Product Principle

このプロジェクトでは、

> 完全自動だが修正しにくい

よりも、

> 半自動で、人間が簡単に修正できる

ことを優先します。

元画像を壊さず、AIの結果を常に確認・修正できる設計を目指します。

## Planned Technology

初期候補:

- Python 3.12+
- PySide6
- NumPy
- Pillow
- OpenCV
- Pydantic
- pytest

AI推論については、コア機能と分離し、後からモデルを交換できる設計にします。

## Repository Structure

予定している構成:

```text
src/
  app/
  core/
  inference/
  exporters/
  ui/
  infrastructure/

tests/
docs/