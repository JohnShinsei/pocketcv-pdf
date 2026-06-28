# PocketCV PDF

PocketCV PDF は、スマートフォン上で動作する画像処理アプリです。カメラ撮影または画像選択を行うと、ブラウザ内で画像処理パイプラインを実行し、元画像・処理後画像・エッジマップ・評価指標を含む PDF レポートを端末内で生成します。

画像はサーバーへアップロードされません。PDF 生成までブラウザ上で完結します。

## 主な機能

- スマートフォンのカメラまたは画像ピッカーから画像を入力
- ブラウザの Canvas API によるオンデバイス画像処理
- 文書らしい四角形領域の推定と透視補正
- 照明ムラ補正、コントラスト補正、シャープ化
- グレースケール化、二値化、エッジマップ生成
- 透視補正信頼度、エッジ密度、コントラスト、二値化しきい値の算出
- 元画像・処理後画像・エッジマップ・評価指標を含む A4 PDF レポート生成
- PDF 内の画像はページ種別に応じて JPEG / lossless Flate を自動選択
- ページの順序変更、削除、PDF ファイル名指定に対応
- ページごとの品質スコアと平均品質の表示
- PWA としてのオフライン起動、端末への追加導線、PDF 共有ボタン
- Python/OpenCV による CLI 版の画像処理パイプラインとテスト

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

スマートフォンでカメラプレビューを使う場合は HTTPS が推奨です。GitHub Pages などの静的ホスティングに置くと、より実機に近い形で試せます。

## モード

- `CV report`: 元画像、処理後画像、エッジマップ、評価指標をまとめた PDF レポートを生成
- `Enhanced`: 照明・コントラスト補正後の画像を PDF 化
- `Gray`: グレースケール画像を PDF 化
- `Binary`: 二値化画像を PDF 化
- `Edges`: エッジマップを PDF 化

## CLI

Python/OpenCV 版の処理パイプラインも含まれています。

```bash
clearscan examples/generated/sample_document.jpg --out outputs --mode color --compare
```

出力:

- `*_clearscan.png`: 処理後画像
- `*_comparison.png`: 処理前後の比較画像
- `*_report.json`: 検出結果と品質指標

## 技術構成

```text
src/clearscan_cv/
  api.py          ローカル開発用 FastAPI サーバー
  cli.py          コマンドライン実行用エントリポイント
  geometry.py     OpenCV による輪郭検出と透視変換
  pipeline.py     画像処理パイプライン
  quality.py      画像品質指標の計算
  static/
    index.html    スマートフォン向けオンデバイス画像処理アプリ
    sw.js         PWA 用 Service Worker
tests/
  test_pipeline.py
scripts/
  generate_sample.py
docs/
  resume.md
```

## 実装上のポイント

- フロントエンドでは Canvas API のピクセル操作を用いて画像処理を実装
- JavaScript のみで PDF バイナリを生成し、サーバーを使わずにダウンロード
- PDF 生成時に画像ピクセル数を制御し、写真ページは JPEG、二値化・エッジページは灰度 Flate 圧縮で埋め込み
- PDF タイトル、作成日時、生成元などのメタデータをローカルで付与
- Service Worker と Web Share API により、スマートフォン上でのオフライン利用と PDF 共有に対応
- 画像処理結果を数値指標として可視化し、単なる PDF 変換ではなく CV プロジェクトとして説明可能
- 市販のスキャンアプリに近いレビュー体験として、ページ管理、品質表示、保存名指定を実装
- Python/OpenCV 側では輪郭検出、明るい紙領域のフォールバック検出、四点透視変換、照明正規化、二値化、品質評価を実装

## 今後の改善案

- 手動で四隅を調整できる UI
- OpenCV.js または WebAssembly による高速化
- OCR 読み取りや読みやすさスコアの追加

## ライセンス

MIT
