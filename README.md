# PocketCV PDF

PocketCV PDF は、スマートフォン上で動作する画像処理アプリです。カメラ撮影または画像選択を行うと、ブラウザ内で画像処理パイプラインを実行し、透視補正・去陰影・文字強調を行ったスキャン画像と PDF を端末内で生成します。

画像はサーバーへアップロードされません。PDF 生成までブラウザ上で完結します。

## 公開デモ

[PocketCV PDF をブラウザで開く](https://johnshinsei.github.io/pocketcv-pdf/)

スマートフォンでは、このリンクを開いて画像を選択するか、カメラを起動して文稿を撮影できます。処理は端末内で行われます。

開発方針は、まず OpenCV ベースのスキャン品質を固め、その出力を OCR に渡し、最後に読み取り結果から文書レイアウトを復元する三段構成です。

PDF 生成後の `PDFを共有` は、対応ブラウザでは生成済み PDF ファイルそのものを Web Share API で共有します。ファイル共有に非対応のブラウザでは、ページ URL ではなく PDF を端末に保存する動作に戻します。

PWA の Service Worker は新しい処理ロジックを検出しやすいように、キャッシュをバージョン管理し、起動時に更新確認を行います。

## 主な機能

- スマートフォンのカメラまたは画像ピッカーから画像を入力
- 撮影・選択した画像を上部の四隅調整フレームに表示し、ユーザーが紙面範囲を確認してからスキャン画像を生成
- 撮影画像は高精細キャンバスで取り込み、ページカードに実際の処理解像度を表示
- ブラウザの Canvas API によるオンデバイス画像処理
- 文書らしい四角形領域の推定と透視補正
- 照明ムラ補正、コントラスト補正、シャープ化
- 局所背景推定による去陰影と、文字インク強度・局所コントラストに基づく背景ノイズ抑制
- 二値化後の孤立ノイズと大きな端部影を除去するスキャン後処理
- 高密度な黒白スキャンで文字が太りすぎる場合に、小さな文字部品を保護しながら筆画を細くする後処理
- 影ムラ・文字墨量・加太りリスクを診断し、白黒 / グレー出力を自動選択するおすすめモード
- 白黒候補よりグレー候補の品質が明確に高い場合は、文字欠けや黒つぶれを避けるため自動的にグレースキャンを選択
- 軽量 dewarp の補正量が大きい書籍・卷き紙では、曲面リスクとして品質診断に表示
- 固定帳票・申請書・請求書向けのテンプレート画像による照明補正
- YOLO / segmentation などのローカル文書検出モデルから四隅 JSON を受け取る外部 detector hook
- DocShadow / DocScanner / DocTr++ などのローカル推論を接続できる外部復元コマンド hook
- Tesseract.js によるブラウザ内 OCR、OCR 結果のコピー、TXT / DOCX 保存、PDF への非表示文字層埋め込み
- OCR の行座標から左右カラム、見出し、段落を推定する Markdown / DOCX 文書復元
- 二段組み文書では、中央タイトルや全幅見出しを本文カラムより先に保つ読み順復元
- グレースケール化、二値化、エッジマップ生成
- 透視補正信頼度、エッジ密度、コントラスト、二値化しきい値の算出
- ブラウザ版 / Python 版ともに OCR word box の傾きから文字行水平度を評価し、読取結果の信頼性診断に反映
- グレー出力では粗い文字前景を除外して背景照明面を推定し、文字密度が高い陰影文書でも紙面ムラを補正
- 処理後のスキャン画像を PNG と A4 PDF として保存し、OCR 実行後は searchable PDF として出力
- 必要な場合のみ、元画像・処理後画像・エッジマップ・評価指標を含む解析レポートを生成
- PDF 内の画像はページ種別に応じて JPEG / lossless Flate を自動選択
- 自動検出された四隅をドラッグ調整してから透視補正・去陰影・文字強調を実行
- 文字行投影と近水平 Hough 線分 fallback による傾き補正、列ごとの文字行オフセットによる軽量カーブ補正
- スキャン画像の近辺縁に残る黒帯・影・紙外ノイズを連通成分で除去
- ページの順序変更、削除、PDF ファイル名指定に対応
- ページごとの品質スコアと平均品質の表示
- PWA としてのオフライン起動、端末への追加導線、PDF ファイル共有ボタン
- Python/OpenCV による CLI 版の画像処理パイプラインとテスト
- Python CLI/API ではスマートフォン JPEG の EXIF Orientation を読み取り、縦横方向を補正してから四隅検出を実行
- Python CLI/API でも、処理後画像を RapidOCR / Tesseract / PaddleOCR に渡す任意 OCR ステージを実行可能
- Python CLI/API から画像 PDF と OCR 文字層付き searchable PDF を出力可能
- Python CLI/API で複数画像をまとめて処理し、OCR 付き多ページ PDF、Markdown、DOCX、batch report を生成可能
- OCR 信頼度、文字編集距離、CER、文字行水平度を用いた読み取り品質評価
- OCR 後の復元 Markdown から DOCX 文書を生成可能
- 文字行投影に基づく軽量 dewarp により、軽い紙面カーブを補正
- OCR バックエンドの導入状態を確認する診断コマンド
- 手動四隅や処理レポートから、文書領域検出モデル学習用の mask / corner dataset を生成

## デモの使い方

ローカルで起動する場合:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[api,dev]
python scripts/generate_sample.py
python -m unittest discover -s tests
uvicorn clearscan_cv.api:app --reload
```

ブラウザで `http://127.0.0.1:8000` を開きます。

スマートフォンで撮影ボタンを使う場合は HTTPS が推奨です。GitHub Pages などの静的ホスティングに置くと、端末のカメラ画面・写真ライブラリ・PDF ファイル共有に近い形で試せます。

## ローカル後端 / Android テスト

Web 版の処理品質に依存せず、Python/OpenCV 後端で処理したい場合はローカル後端を起動します。

PC 上でローカルアプリを起動する場合:

```bash
pip install -e .[api,ocr]
pocketcv-local
```

起動後、`http://127.0.0.1:8765/local` が開きます。この画面は画像を `/api/process` に送り、Python/OpenCV 側でスキャン画像、PDF、OCR、DOCX を生成します。

Android 実機から PC の後端へ接続する場合は、同じ Wi-Fi 上で後端を LAN 向けに起動します。

```bash
pocketcv-local --host 0.0.0.0 --port 8765
```

Android アプリ側の Backend URL:

- Android Emulator: `http://10.0.2.2:8765`
- 実機: `http://<PCのLAN IP>:8765` 例 `http://192.168.1.20:8765`

Android APK は GitHub Actions の `Android APK` workflow で `pocketcv-android-debug-apk` artifact として生成します。アプリは WebView ではなく原生 Android クライアントです。`カメラで撮影` は Android のカメラアプリを開き、MediaStore 経由で高解像度 JPEG を受け取ります。`画像を選択` では既存画像を読み込みます。撮影または選択した画像は上部の四隅調整フレームに表示され、自動検出された角をドラッグで直してから生成できます。`端末内OpenCVでスキャン` は APK に同梱した OpenCV Android で、調整後の四隅を使った Homography 透視補正、照明正規化、グレー / 白黒スキャン、PNG / PDF 保存を端末内で実行します。生成後は `PNG共有` / `PDF共有` から Android の共有シートへファイル本体を渡せます。`カラースキャン` は四隅の切り出しと透視補正だけを行い、文字強調や二値化は適用しません。

より重い Python/OpenCV パイプライン、OCR、DOCX、searchable PDF を試す場合は、`PC後端でスキャン生成` を使います。この場合も上部の四隅調整フレームの座標を `corners_space=input` としてローカル FastAPI 後端に送り、返ってきた PNG / PDF / DOCX を保存できます。アプリ起動後は先に `API確認` を押し、後端に接続できることを確認してから画像を選択します。

Android SDK / adb がある環境では、debug APK をインストールして起動できます。

```powershell
.\scripts\install_android_debug.ps1 -ApkPath android\app\build\outputs\apk\debug\app-debug.apk
```

Android Emulator で端末内 OpenCV とローカル後端の両方を確認する場合は、`PocketCV_API35` AVD を用意してから QA スクリプトを実行します。スクリプトは debug APK のビルド、FastAPI 後端の起動、APK インストール、`API確認`、サンプル画像選択、四隅自動検出、端末内 OpenCV 処理、`/api/process` 実行、スクリーンショット保存までを adb で検証します。

```powershell
.\scripts\android_emulator_qa.ps1
```

出力は `tmp\android-qa\` に保存されます。すでに APK をビルド済みの場合は `-SkipBuild` を付けると、Android 側の確認だけを再実行できます。

Windows のローカル後端 EXE を作る場合:

```powershell
.\scripts\build_windows_local_app.ps1
```

生成物は `dist\PocketCV-PDF-Local\PocketCV-PDF-Local.exe` です。

## モード

- `白黒スキャン`: 去陰影と文字強調を行った白黒スキャン画像を生成
- `グレースキャン`: 去陰影と文字強調を行ったグレースケール画像を生成
- `カラースキャン`: 四隅の切り出しと透視補正だけを行い、元の色と質感を残した画像を生成
- `解析レポート`: 元画像、処理後画像、エッジマップ、評価指標をまとめた PDF レポートを生成
- `Edges`: エッジマップを PDF 化

## 開発ロードマップ

1. OpenCV スキャン強化: 文書検出、四隅補正、透視変換、去陰影、文字強調、二値化ノイズ除去を安定させる。
2. OCR 接続: スキャン済み画像を入力として、ブラウザ内 OCR エンジンへ渡し、テキストと信頼度を抽出する。
3. レイアウト復元: OCR の文字位置を使い、見出し、段組み、段落を Markdown 文書として再構成する。

## CLI

Python/OpenCV 版の処理パイプラインも含まれています。

端から端までのデモ出力を一括生成する場合:

```bash
python scripts/run_demo_pipeline.py examples/generated/sample_document.jpg --out outputs/demo --mode auto
```

このコマンドは、処理後スキャン画像、処理前後の比較画像、画像 PDF、品質レポート、OCR バックエンド診断、読みやすさ指標をまとめて `demo_summary.json` に保存します。RapidOCR / Tesseract / PaddleOCR のいずれかが利用可能な環境では、OCR テキスト、復元 Markdown、DOCX、searchable PDF も同時に生成します。

OCR なしで画像処理デモだけを確認する場合:

```bash
python scripts/run_demo_pipeline.py C:\path\to\photo.jpg --out outputs/demo --mode auto --no-ocr
```

```bash
clearscan examples/generated/sample_document.jpg --out outputs --mode color --compare
```

画像 PDF も保存する場合:

```bash
clearscan examples/generated/sample_document.jpg --out outputs --mode auto --pdf
```

複数ページの PDF を作る場合:

```bash
clearscan page1.jpg page2.jpg page3.jpg --out outputs/batch --mode binary --pdf --readability
```

この場合、各ページの `*_clearscan.png` と `*_report.json` に加えて、`clearscan_batch_scan.pdf` と `clearscan_batch_report.json` が保存されます。

複数ページを OCR して searchable PDF と復元文書も作る場合:

```bash
clearscan page1.jpg page2.jpg page3.jpg --out outputs/batch --mode binary --ocr --layout --docx --searchable-pdf --ocr-lang jpn+eng
```

この場合、各ページの OCR テキストと layout Markdown に加えて、`clearscan_batch_searchable.pdf`、`clearscan_batch_ocr.txt`、`clearscan_batch_layout.md`、`clearscan_batch_layout.docx` が保存されます。

読み取り品質だけを評価する場合:

```bash
clearscan examples/generated/sample_document.jpg --out outputs --mode gray --readability
```

平面文書として処理したい場合は `--no-dewarp` で軽量 dewarp を無効化できます。

自動検出が外れた写真を、手動四隅で再生成する場合:

```bash
clearscan photo.jpg --out outputs/manual --mode binary --corners "223,414 1864,279 2207,2685 0,2943" --pdf
```

外部の文書検出モデルを接続する場合は、信頼できるローカルコマンドに `{input}` と `{output}` を渡します。コマンドは `{"corners":[[x,y],...],"confidence":0.88,"method":"your_model"}` のような JSON を出力します。失敗した場合は OpenCV の contour / Hough / connected-paper fallback に戻り、`external_detector` レポートに理由を保存します。

```bash
clearscan photo.jpg --out outputs/detect --mode auto --external-detector-command "python detect_page.py --input {input} --output {output}" --pdf
```

外部の文書復元モデルを接続する場合は、信頼できるローカルコマンドに `{input}` と `{output}` を渡します。コマンドが失敗した場合は従来の OpenCV パイプラインに自動で戻り、`external_restorer` レポートに理由を保存します。

```bash
clearscan photo.jpg --out outputs/deep --mode auto --external-restorer-command "python infer.py --input {input} --output {output}" --pdf
```

固定フォームや請求書など、理想テンプレート画像がある場合は `--template-image` を指定すると、透視補正後の低周波照明面をテンプレートに近づけてから通常の OpenCV 強調を行います。

```bash
clearscan invoice-photo.jpg --out outputs/forms --mode auto --template-image invoice-template.png --pdf --readability
```

`--corners` は入力画像上の座標で、左上・右上・右下・左下の順に指定します。順序が多少入れ替わっていても内部で並べ替えます。JSON 形式の `[[x,y], ...]` や `{"corners":[{"x":...,"y":...}, ...]}` も指定できます。

`*_report.json` に出力された `document_detection.corners` を再利用する場合は、処理後の縮小座標なので `--corners-space processed` を付けます。

```bash
clearscan photo.jpg --out outputs/manual --mode binary --corners "223,414 1864,279 2207,2685 0,2943" --corners-space processed --pdf
```

OCR も実行する場合:

```bash
pip install -e .[ocr]
clearscan examples/generated/sample_document.jpg --out outputs --mode binary --ocr --ocr-engine auto --ocr-lang jpn+eng --layout --docx --searchable-pdf --expected-text answer.txt
```

OCR 環境を確認する場合:

```bash
clearscan --ocr-status --ocr-lang jpn+eng
```

文書領域検出モデルを学習するためのデータセットを作る場合:

実写データが少ない段階では、スマートフォン撮影に近い透視、影、背景、ぼけ、ノイズを持つ合成データから始められます。

```bash
clearscan-synth --out datasets/docnet-synth --count 1000 --width 960 --height 1280 --seed 42
```

手動四隅調整済みの実写データがある場合:

```bash
clearscan-dataset --reports outputs/photo_report.json --out datasets/docnet
```

手動で作った JSONL アノテーションからも生成できます。

```jsonl
{"id":"page_001","image":"photos/page_001.jpg","corners":[[223,414],[1864,279],[2207,2685],[0,2943]]}
```

```bash
clearscan-dataset --annotations annotations.jsonl --image-root . --out datasets/docnet
```

出力される `manifest.jsonl` には、学習画像、二値 mask、ピクセル座標の四隅、0-1 正規化四隅、train / val split が含まれます。第一段階の学習対象は、スマートフォン写真から文書 mask と四隅を予測する軽量 detector です。予測結果は既存の OpenCV Homography、去陰影、PDF/OCR パイプラインに接続できます。

軽量な文書 mask detector を学習して、既存の透視補正パイプラインに接続する場合:

```bash
pip install -e .[train]
clearscan-docnet train --dataset datasets/docnet-synth --out models/docnet.pt --epochs 20 --image-size 256
clearscan-docnet predict --checkpoint models/docnet.pt --input photos/page_001.jpg --output outputs/page_001_corners.json
clearscan photos/page_001.jpg --out outputs/model_scan --mode gray --external-detector-command "clearscan-docnet predict --checkpoint models/docnet.pt --input {input} --output {output}"
```

このモデルは最初から画像全体を復元するのではなく、文書領域の mask と四隅を安定して推定する役割に限定しています。透視変換、影除去、文字強調、PDF/OCR 出力は従来の OpenCV パイプラインで行うため、失敗時は自動検出や手動四隅調整に戻せます。

出力:

- `*_clearscan.png`: 処理後画像
- `*_comparison.png`: 処理前後の比較画像
- `*_report.json`: 検出結果と品質指標
- `*_scan.pdf`: 処理後画像を埋め込んだ PDF
- `clearscan_batch_scan.pdf`: 複数入力から生成した多ページ PDF
- `clearscan_batch_searchable.pdf`: 複数入力から生成した OCR 文字層付き多ページ PDF
- `clearscan_batch_ocr.txt`: 複数ページ OCR テキスト
- `clearscan_batch_layout.md`: 複数ページの復元 Markdown
- `clearscan_batch_layout.docx`: 複数ページの復元 DOCX
- `clearscan_batch_report.json`: 複数ページ処理の集約レポート
- `*_searchable.pdf`: OCR 文字層付き PDF
- `*_ocr.txt`: OCR テキスト
- `*_layout.md`: OCR 行座標から復元した Markdown
- `*_layout.docx`: 復元文書の Word 互換 DOCX
- `readability`: report 内の OCR 信頼度、低信頼行比率、文字編集距離、CER、文字行水平度、OCR を含む品質診断
- `quality`: report 内の影ムラ残り、影スコア、文字墨量、加太りリスク、文字欠けリスク、エッジ密度、コントラスト、品質診断と推奨操作

Web 版の `おすすめ自動` は、白黒化で細い文字が欠ける場合に `文字欠けリスク` を検出し、可読性を優先してグレースキャンを選びます。白黒/エッジ出力のプレビュー画像は PNG として保持し、再圧縮による劣化を避けます。

Web 版の `おすすめ自動`、`グレースキャン`、`白黒スキャン`、`解析レポート` は、透視補正後に文字行投影から列ごとの上下オフセットを推定し、軽い紙面カーブだけを保守的に補正します。`カラースキャン` は、四隅の切り出しと透視補正だけを行い、dewarp、去影、シャープ化、二値化などの画像強調は適用しません。

Python 版の OCR エンジンは任意依存です。軽量に試す場合は `.[ocr]` で Tesseract ラッパー、ONNX ベースの端末側 OCR を試す場合は `.[rapidocr]`、PaddleOCR を使う場合は `.[paddleocr]` を追加します。Tesseract を使う場合は、別途 Tesseract 本体と言語データも必要です。OCR 依存を入れていない環境でも、画像処理パイプラインはそのまま動作します。

`--ocr-status` は、RapidOCR、Tesseract、PaddleOCR の Python パッケージ、Tesseract 実行ファイル、言語データの有無を JSON で表示します。

ローカル API の `/api/process` でも、同じ形式の `corners` フォーム値を送ると手動四隅で処理できます。`corners_space` は `input` または `processed` を指定できます。

例:

```bash
curl -X POST http://127.0.0.1:8000/api/process ^
  -F "file=@photo.jpg" ^
  -F "mode=binary" ^
  -F "corners=223,414 1864,279 2207,2685 0,2943" ^
  -F "corners_space=processed" ^
  -F "pdf=true" ^
  -F "readability=true"
```

複数ファイルをまとめて処理し、多ページ PDF を返す場合:

```bash
curl -X POST http://127.0.0.1:8000/api/process-batch ^
  -F "files=@page1.jpg" ^
  -F "files=@page2.jpg" ^
  -F "mode=binary" ^
  -F "searchable_pdf=true" ^
  -F "ocr=true" ^
  -F "layout=true" ^
  -F "docx=true" ^
  -F "readability=true"
```

## 技術構成

```text
src/clearscan_cv/
  api.py          ローカル開発用 FastAPI サーバー
  cli.py          コマンドライン実行用エントリポイント
  corners.py      手動四隅の解析、座標変換、検証
  dewarp.py       文字行投影に基づく軽量 dewarp
  geometry.py     OpenCV による輪郭検出と透視変換
  model_hooks.py  外部 detector / 文書復元モデルを接続する fallback-safe hook
  docnet.py       任意 PyTorch による軽量文書 mask detector の学習 / 推論 CLI
  synthetic_data.py 合成スマートフォン撮影データセットの生成
  training_data.py 手動四隅 / report から学習用 mask と corner manifest を生成
  pipeline.py     画像処理パイプライン
  quality.py      画像品質指標の計算
  evaluation.py   OCR と読み取り品質の評価
  ocr.py          任意 OCR エンジン接続と Markdown レイアウト復元
  export.py       画像 PDF、searchable PDF、DOCX の生成
  static/
    index.html    スマートフォン向けオンデバイス画像処理アプリ
    sw.js         PWA 用 Service Worker
tests/
  test_api.py
  test_demo_pipeline.py
  test_evaluation.py
  test_export.py
  test_ocr.py
  test_pipeline.py
scripts/
  generate_sample.py
  run_demo_pipeline.py
docs/
  resume.md
```

## 実装上のポイント

- フロントエンドでは Canvas API のピクセル操作を用いて画像処理を実装
- 文書画素をタイルごとに解析し、局所的な紙面背景を推定して影と紙面ムラを補正
- 文字候補は背景との差分、局所エッジ、正規化輝度を組み合わせて強調し、紙面テクスチャは白側へ抑制
- JavaScript のみで PDF バイナリを生成し、サーバーを使わずにダウンロード
- PDF 生成時に画像ピクセル数を制御し、写真ページは JPEG、二値化・エッジページは灰度 Flate 圧縮で埋め込み
- PDF タイトル、作成日時、生成元などのメタデータをローカルで付与
- Service Worker と Web Share API により、スマートフォン上でのオフライン利用と PDF 共有に対応
- 画像処理結果を数値指標として可視化し、単なる PDF 変換ではなく CV プロジェクトとして説明可能
- 市販のスキャンアプリに近いレビュー体験として、ページ管理、品質表示、保存名指定を実装
- 自動検出結果を初期値にした四隅ドラッグ編集で、難しい写真でもユーザーが紙面範囲を修正可能
- OCR 結果の行・単語座標を使い、左右カラム、見出し、段落を Markdown として復元
- Python 側では RapidOCR、Tesseract、PaddleOCR を可選択の OCR バックエンドとして扱い、出力を共通の line / word / bbox / confidence 形式に正規化
- Python 側でも画像 PDF と OCR 行 bbox を使った非表示文字層付き PDF を生成し、スキャン画像・TXT・Markdown・PDF まで一つの流れで出力
- Python CLI の複数入力では、各ページを同じ OpenCV パイプラインで処理し、A4 の多ページ PDF と batch report に集約
- 復元 Markdown を WordprocessingML に変換し、外部ライブラリなしで DOCX として保存
- OCR 後の評価として mean confidence、低信頼行比率、編集距離、CER、文字行水平度を report に保存し、画像処理パラメータ改善の根拠にできる
- 透視補正後、Python/OpenCV と Web/Canvas の両方で文字行投影から列ごとの上下オフセットを推定し、軽い曲面歪みを補正
- Python/OpenCV 側では輪郭検出、明るい紙領域のフォールバック検出、四点透視変換、文字行 deskew、背景推定型の照明正規化、保守的な二値化、品質評価を実装

## 研究ベースの改善

文書画像補正・文書解析の代表的な研究を読み、端末内で実行できる形に落とし込んでいます。今回の実装では、カメラ文書補正で使われる text baseline / textline 幾何制約に基づく文字行 deskew と、幾何補正後の光照補正・文字強調を追加しました。

参照論文と実装への反映は [研究メモ](docs/research-notes.md) にまとめています。

## 今後の改善案

- OpenCV.js または WebAssembly による高速化
- 曲がった紙面のデワープと OCR 用前処理の強化
- 表、注記、箇条書きのレイアウト復元精度向上
- 読みやすさスコアと OCR 信頼度のページ別レビュー

## ライセンス

MIT
