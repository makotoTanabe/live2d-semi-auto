# Webキャラクターの制作と組み込み

Webエディターでパーツ・役割・動作を編集し、ブラウザーだけで再生できるZIPを書き出します。モデルは独立した透過画像とJSON設定をCanvas2Dで描画する形式です。アプリの再生側にPython、Node、外部SDK、CDN、APIキーは必要ありません。

## エディターを起動する

```bash
uv sync --locked --extra web
uv run --locked --extra web live2d-semi-auto-web
```

http://127.0.0.1:8080 を開きます。`--port 8081` のようにポートを変更できます。`--host` の初期値は `127.0.0.1` です。エディターはローカルまたは私有環境で使用するものとして提供しており、公開ホスティング・利用者認証・共同編集のサービスは含みません。静的な書き出しモデルは別のWebホストへ自由に配置できます。

`web` オプションはFastAPI、Uvicorn、python-multipartを追加します。ローカルLaMa補完も使う場合は `uv sync --locked --extra web --extra ai` で依存を入れ、[MODELS.md](MODELS.md) の明示的な重み取得を別途実行してください。起動も `uv run --locked --extra web --extra ai live2d-semi-auto-web` として両方のオプションを指定します。`uv run` で `ai` を省略すると、同期済みでもPyTorchが外れる場合があります。通常の編集・動作・出力にAIモデルやGPUは不要です。

## 制作の流れ

1. 「サンプルを動かす」で配置済みのキャラクターを開きます。自身のPNG・JPEG・WebPや `.l2split` も読み込めます。
2. 左側でパーツを選び、名前、種類、表示状態、前後関係を設定します。「動きの役割」は名前からの初期推定を上書きできます。
3. 中央のアニメーション表示で、右側の表情とモーションを切り替えます。顔の左右・上下・傾き、目の開き、口、呼吸を調整し、自動まばたき、自然な動き、マウス追従を試します。
4. 「マスク編集」表示でブラシ・投げ縄の追加／消去を行います。ホイールで拡大、中ボタンで移動できます。Undo/Redoはボタンと `Ctrl`／`⌘` + `Z`、`Shift` + `Z` で操作します。原画・選択パーツ・再合成・差分の表示も確認してください。
5. 必要なら色による分割、GPTのパーツ推定／シート位置合わせ、隠れ領域の補完を実行します。推論結果はプレビューし、「候補を採用」または破棄を選びます。失敗・拒否で既存の手動パーツは消えません。
6. 「保存」で編集可能な `.l2split` をダウンロードします。役割・支点・動作の強さ・メッシュ設定はパーツの安定IDを参照し、プロジェクトの操作履歴に `web-runtime-settings` として保持します。名前変更で参照は切れません。表情選択、再生状態、ライブの顔向き・目・口などのスライダーはプレビュー用で、すべてをプロジェクトへ保存する形式ではありません。
7. 「Web用モデルを書き出す」で `web-character.zip` を取得します。PNG素材一式とPSDも追加の出力として利用できます。

パーツの役割は `body`、`face`、`hair_back`、`hair_front`、`hair_side`、`eye`、`iris`、`eyebrow`、`mouth`、頭／胴体のアクセサリー、左右の腕、脚、固定を用意しています。パーツ名からの推定が違う場合は役割を変更してください。パーツ単位の独立した画像を動かすため、顔画像に焼き込まれた目・口だけでは、その部位を独立して制御できません。

表情は `neutral`（通常）、`smile`（笑顔）、`angry`（怒り）、`cry`（泣き）、`surprised`（驚き）、`shy`（照れ）です。モーションは `idle`（待機）、`nod`（うなずき）、`shake`（首振り）、`greeting`（挨拶）です。メッシュ変形と役割ごとの変換を組み合わせ、頬の赤みや涙は描画時の重ね描きで表現します。独自タイムラインや複雑な物理シミュレーションは現在の範囲に含みません。

固定状態や移動・回転だけの描画は画像を直接描き、変形が必要な場合にメッシュを使用します。透明な境界へ三角形の継ぎ目が現れないように、非線形のメッシュはレイヤーごとに透明度を考慮して合成します。

## 書き出したモデルを再生する

ZIPをすべて同じフォルダーへ展開してください。

```text
character/
  index.html
  model.json
  runtime.js
  README.md
  parts/
    <stable-id-derived-name>.png
    ...
```

`index.html` は実際のモデル設定を安全にエスケープしたJSONとして内包し、相対パスで画像と `runtime.js` を読み込みます。エディターサーバーを止めても、フォルダーを別の静的HTTPサーバーで配信して再生できます。ローカルファイルの読み込みを許可するブラウザーでは、このHTMLを直接開く方法も使えます。

現在の検証では、エディターから独立したHTTP配信でのモデル再生を確認しています。このクラウド環境のChromiumは管理ポリシーにより `file://` を `ERR_BLOCKED_BY_ADMINISTRATOR` で拒否するため、直接ファイルを開く任意の確認はスキップしました。ポリシーは回避せず、HTTP配信で確認してください。例えば展開先で `python -m http.server 8081 --bind 127.0.0.1` を実行し、http://127.0.0.1:8081 を開けます。

`model.json` は共通キャンバス寸法、sRGBの画像、各パーツのID・役割・順序・表示・元座標上の範囲、頭／胴体の支点、動作設定、表情プリセットを記録します。PNGは必要領域だけを切り出した透明画像です。画像を切り出しても配置座標は元のキャンバスに対応します。書き出しは元の画像・マスクを変更しません。

## 自分のWebアプリへ組み込む

書き出した一式を `/character/` に配置した場合、次のコードでCanvasへ表示できます。`runtime.js` は通常のscriptとして読み込み、`window.Live2DWeb.Character` を使用します。

```html
<canvas id="avatar" width="600" height="800"></canvas>
<script src="/character/runtime.js"></script>
<script>
(async () => {
  const response = await fetch('/character/model.json');
  if (!response.ok) throw new Error('キャラクター設定を取得できません。');
  const model = await response.json();
  const avatar = new window.Live2DWeb.Character(
    document.getElementById('avatar'), model, {baseUrl: '/character/'}
  );
  await avatar.load();
  avatar.setExpression('smile');
  avatar.setMotion('idle');
  avatar.setParameters({angleX: 0.2, mouthOpen: 0.4});
  avatar.start();
  window.avatar = avatar;
})().catch(error => console.error(error.message));
</script>
```

`baseUrl` は画像の置き場所で、末尾に `/` を付けます。同一オリジンに配置する場合、追加のCORS設定は不要です。別オリジンから設定や画像を配信する場合は、そのホストで必要なCORSを設定してください。

ローカルファイル自体を開けるブラウザーでもJSONの `fetch()` が制限される場合は、書き出された `index.html` の方式を使用します。そこにあるエスケープ済みの `<script type="application/json" id="model">…</script>` ブロックを保持し、次のように読み取ります。`file://` 自体が管理ポリシーで禁止されている場合は、HTTP配信を使用してください。

```js
const model = JSON.parse(document.getElementById('model').textContent);
const avatar = new window.Live2DWeb.Character(
  document.getElementById('avatar'), model, {baseUrl: 'character/'}
);
await avatar.load();
avatar.start();
```

この例の `await` はasync関数内で使用してください。モデルをHTMLへ内包する場合は、出力されたエスケープ済みブロックを利用します。独自に生成する場合、名前などに含まれる `<`、`>`、`&` を安全にエスケープしてscript要素の終端として解釈されないようにしてください。

Reactなどで表示部分を破棄するときは `avatar.destroy()` を呼び、再生フレームとポインターイベントを解放します。コンポーネントの停止中は `stop()`、再開時は `start()` を使用できます。

CanvasのCSS表示サイズを変える場合は、表示幅・高さに `devicePixelRatio` を掛けた値を `avatar.resize(width, height)` へ渡してください。描画領域と表示領域の縦横比を揃えると、キャラクター全体が引き伸ばされません。書き出し例のHTMLは表示サイズの変化に追従します。

## 再生API

| API | 用途 |
| --- | --- |
| `new Live2DWeb.Character(canvas, model, options)` | Canvasと設定から再生オブジェクトを作成 |
| `await load()` | 透明画像を取得し、寸法・レイヤーを検証 |
| `start()` / `stop()` / `destroy()` | 再生・一時停止・資源解放 |
| `resize(width, height)` | Canvasの描画サイズを表示領域へ合わせる |
| `setParameters(values)` | 指定したパラメータだけ更新 |
| `setExpression(name)` | 表情プリセットを変更 |
| `setMotion(name)` | モーションを変更 |
| `setAutoBlink(boolean)` | 自動まばたきを切り替え |
| `setIdle(boolean)` | 呼吸・髪などの自然な動きを切り替え |
| `setPointerTracking(boolean)` | ポインターへの顔・視線追従を切り替え |

`setParameters()` の `angleX`・`angleY`・`angleZ` は −1〜1、`eyeOpen`・`mouthOpen`・`breath` は0〜1です。顔の向き・傾きを度数で直接渡す形式ではありません。自動動作・表情とパラメータが組み合わさるため、値の変化だけを確認したい場合は自動まばたき・自然な動き・ポインター追従を無効にして通常表情で試してください。

`options.baseUrl` はパーツ画像の基準URLです。`options.assetUrls` に `{[partId]: imageUrl}` を渡すと、そのパーツだけ明示的なURLへ置き換えられます。アプリが保有するBlob URLなどを使用する場合に便利です。URLはレイヤー名ではなく安定IDで対応させ、Canvasを破棄した後にアプリ側でBlob URLも解放してください。

## データと外部AI

画像とプロジェクトのアップロード先は、初期設定では自身のローカルエディターです。セッションはメモリ上で分離され、有効期限と数の上限があります。現在は無操作で約1時間経過すると失効するため、編集結果は「保存」でダウンロードしてください。サーバー再起動後にメモリ上の作業は残りません。

1ファイルのアップロードは32MiB、画像の各辺は4096px、パーツ数は128までです。原画・シート・パーツ画像・可視／隠れ／生成／編集マスクなど、保存するすべての画像の縦横画素数を合計して `64 × 1024²` 画素までとします。マスクも画像1枚として計上するため、同じ原画から候補を増やすと上限へ達する場合があります。読み込み・追加・候補処理で上限を超えた場合は、既存の編集内容を保持して処理を拒否します。不要なパーツを減らす、候補数を減らす、元ファイルを残して低解像度の別画像を使う方法で再試行してください。

GPT機能は別の明示操作です。確認画面で送る画像と `api.openai.com`、用途、モデル、料金を示した後に外部リクエストを行います。キーは画面のパスワード欄、またはサーバーの `GPT_API_KEY`／`OPENAI_API_KEY` から渡します。ブラウザーストレージ、プロジェクト、モデル、書き出しZIP、ログには保存しません。候補の採用を取り消すことはできますが、送信済みのリクエスト・料金は取り消せません。[MODELS.md](MODELS.md) にモデルと取得手順をまとめています。

## 素材品質の確認

配置済みサンプルの [元シート](../samples/parts_atlas.png)、[比較画像](../samples/results/alignment/comparison.png)、[検証レポート](../samples/results/alignment/report.json) を参照できます。実際のGPTリクエスト、保存・再読込、PSD各レイヤーの画素保持は確認しています。

この素材には元絵との形状差、透明な輪郭の粗さ、欠け、独立した腕などの不足があります。大きな部品には約21〜43pxの基準点残差も残ります。動作が再生できることは、素材の仕上がりを保証しません。まばたきで目の周囲が欠ける、髪を動かすと隠れていた領域が露出する、挨拶で腕が動かないなどの場合は、必要なパーツ・隠れ領域を編集して再出力してください。

## 実ブラウザーでの確認

Pythonの検証は `QT_QPA_PLATFORM=offscreen uv run --locked --extra web pytest -q` で実行します。ブラウザー検証にはNodeと開発用Playwrightを使用します。アプリの描画にNodeやブラウザー自動化パッケージは不要です。

```bash
npm ci
CHROMIUM_PATH=/usr/bin/chromium node scripts/check_web.mjs \
  --url http://127.0.0.1:8080 \
  --output /path/to/new_browser_check
```

既存のChromiumがない場合は明示的に `npx playwright install chromium` を実行し、`CHROMIUM_PATH` を省略してPlaywrightのブラウザーを使用できます。新しい出力先を選んでください。確認用スクリプトは編集・Undo/Redo、候補の採用／拒否、表情・モーション、プロジェクトの再読込、ダウンロードしたWebモデルの独立再生を確認し、スクリーンショット、レポート、動画とダウンロードしたデータを保存します。実行結果は保存されたレポートで判断してください。

リポジトリ内の検証記録先は [samples/results/web](../samples/results/web) です。`report.json`、エディター／独立HTTP再生の画面、`character_motion.webm`、ダウンロードしたWebモデル、展開済み `standalone/index.html` などを合わせて確認できます。クラウドの管理ポリシーによる `file://` のスキップは、HTTP再生の成功と分けてレポートへ記録しています。
