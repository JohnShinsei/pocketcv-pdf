# PocketCV PDF

PocketCV PDF は、スマートフォン上で動作する画像処理アプリです。カメラ撮影または画像選択を行うと、ブラウザ内で画像処理パイプラインを実行し、透視補正・去陰影・文字強調を行ったスキャン画像と PDF を端末内で生成します。

画像はサーバーへアップロードされません。PDF 生成までブラウザ上で完結します。

## 公開デモ

[PocketCV PDF をブラウザで開く](https://johnshinsei.github.io/pocketcv-pdf/)

スマートフォンでは、このリンクを開いて画像を選択するか、カメラを起動して文稿を撮影できます。処理は端末内で行われます。

開発方針は、まず OpenCV ベースのスキャン品質を固め、その出力を OCR に渡し、最後に読み取り結果から文書レイアウトを復元する三段構成です。

## 主な機能

- スマートフォンのカメラまたは画像ピッカーから画像を入力
- 撮影・選択した画像を上部の四隅調整フレームに表示し、ユーザーが紙面範囲を確認してからスキャン画像を生成
- 撮影画像は高精細キャンバスで取り込み、ページカードに実際の処理解像度を表示
- ブラウザの Canvas API によるオンデバイス画像処理
- 文書らしい四角形領域の推定と透視補正
- 照明ムラ補正、コントラスト補正、シャープ化
- 局所背景推定による去陰影と、文字インク強度・局所コントラストに基づく背景ノイズ抑制
- 二値化後の孤立ノイズと大きな端部影を除去するスキャン後処理
- Tesseract.js によるブラウザ内 OCR、OCR 結果のコピーと TXT 保存
- OCR の行座標から左右カラム、見出し、段落を推定する Markdown 文書復元
- グレースケール化、二値化、エッジマップ生成
- 透視補正信頼度、エッジ密度、コントラスト、二値化しきい値の算出
- 処理後のスキャン画像を PNG と A4 PDF として保存
- 必要な場合のみ、元画像・処理後画像・エッジマップ・評価指標を含む解析レポートを生成
- PDF 内の画像はページ種別に応じて JPEG / lossless Flate を自動選択
- 自動検出された四隅をドラッグ調整してから透視補正・去陰影・文字強調を実行
- ページの順序変更、削除、PDF ファイル名指定に対応
- ページごとの品質スコアと平均品質の表示
- PWA としてのオフライン起動、端末への追加導線、PDF ファイル共有ボタン
- Python/OpenCV による CLI 版の画像処理パイプラインとテスト
- Python CLI/API でも、処理後画像を RapidOCR / Tesseract / PaddleOCR に渡す任意 OCR ステージを実行可能
- Python CLI/API から画像 PDF と OCR 文字層付き searchable PDF を出力可能
- OCR 信頼度、文字編集距離、CER、文字行水平度を用いた読み取り品質評価
- OCR 後の復元 Markdown から DOCX 文書を生成可能
- 文字行投影に基づく軽量 dewarp により、軽い紙面カーブを補正
- OCR バックエンドの導入状態を確認する診断コマンド

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

## モード

- `白黒スキャン`: 去陰影と文字強調を行った白黒スキャン画像を生成
- `グレースキャン`: 去陰影と文字強調を行ったグレースケール画像を生成
- `カラースキャン`: 文字を強調しつつ色を少し残したスキャン画像を生成
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
python scripts/run_demo_pipeline.py examples/generated/sample_document.jpg --out outputs/demo --mode binary
```

このコマンドは、処理後スキャン画像、処理前後の比較画像、画像 PDF、品質レポート、OCR バックエンド診断、読みやすさ指標をまとめて `demo_summary.json` に保存します。RapidOCR / Tesseract / PaddleOCR のいずれかが利用可能な環境では、OCR テキスト、復元 Markdown、DOCX、searchable PDF も同時に生成します。

OCR なしで画像処理デモだけを確認する場合:

```bash
python scripts/run_demo_pipeline.py C:\path\to\photo.jpg --out outputs/demo --mode binary --no-ocr
```

```bash
clearscan examples/generated/sample_document.jpg --out outputs --mode color --compare
```

画像 PDF も保存する場合:

```bash
clearscan examples/generated/sample_document.jpg --out outputs --mode binary --pdf
```

読み取り品質だけを評価する場合:

```bash
clearscan examples/generated/sample_document.jpg --out outputs --mode gray --readability
```

平面文書として処理したい場合は `--no-dewarp` で軽量 dewarp を無効化できます。

自動検出が外れた写真を、手動四隅で再生成する場合:

```bash
clearscan photo.jpg --out outputs/manual --mode binary --corners "223,414 1864,279 2207,2685 0,2943" --pdf
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

出力:

- `*_clearscan.png`: 処理後画像
- `*_comparison.png`: 処理前後の比較画像
- `*_report.json`: 検出結果と品質指標
- `*_scan.pdf`: 処理後画像を埋め込んだ PDF
- `*_searchable.pdf`: OCR 文字層付き PDF
- `*_ocr.txt`: OCR テキスト
- `*_layout.md`: OCR 行座標から復元した Markdown
- `*_layout.docx`: 復元文書の Word 互換 DOCX
- `readability`: report 内の OCR 信頼度、低信頼行比率、文字編集距離、CER、文字行水平度

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

## 技術構成

```text
src/clearscan_cv/
  api.py          ローカル開発用 FastAPI サーバー
  cli.py          コマンドライン実行用エントリポイント
  corners.py      手動四隅の解析、座標変換、検証
  dewarp.py       文字行投影に基づく軽量 dewarp
  geometry.py     OpenCV による輪郭検出と透視変換
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
- 復元 Markdown を WordprocessingML に変換し、外部ライブラリなしで DOCX として保存
- OCR 後の評価として mean confidence、低信頼行比率、編集距離、CER、文字行水平度を report に保存し、画像処理パラメータ改善の根拠にできる
- 透視補正後、文字行投影から列ごとの上下オフセットを推定し、`cv2.remap` で軽い曲面歪みを補正
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
