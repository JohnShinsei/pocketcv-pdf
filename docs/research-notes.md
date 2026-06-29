# 研究メモ: 文書画像補正と実装への反映

PocketCV PDF は、スマートフォンや PC ブラウザで完結する文書スキャンを目標にしています。ここでは、参照した代表的な論文と、現時点で実装へ反映した点を整理します。

## 今回実装した改善

### 文字行ベースの小角度 deskew

透視補正後の画像に対して、文字行・罫線の投影プロファイルを使って `-6°` から `6°` の範囲で支配的な水平角を推定し、微小な傾きを自動補正します。

この改善は、カメラ文書解析で文書境界だけでなくテキスト行や段落 alignment を幾何補正に使うという古典的な方向、および近年の文書 rectification で textline を明示的な幾何制約として扱う方向を、端末内で軽量に実装したものです。

実装箇所:

- `src/clearscan_cv/pipeline.py`: `estimate_textline_skew`, `deskew_by_text_lines`
- `src/clearscan_cv/static/index.html`: `estimateTextLineSkew`, `deskewCanvasByTextLines`
- UI のページ品質表示と解析レポートに `文字行傾き補正` を追加

### Hough 直線 fallback による四隅復元

紙面の輪郭が影や背景で途切れ、最大輪郭だけでは四角形にならない場合に備えて、長い水平・垂直寄りの直線を Hough 的に投票し、上下左右の境界線の交点から四隅を復元する fallback を追加しました。

この改善は、平面文書では四つの対応点から Homography が決まるという前提を保ちつつ、文書境界線が閉じた輪郭として検出できない写真にも対応するためのものです。Python/OpenCV 版は `HoughLinesP`、Web 版は Canvas のサンプリングエッジに対する軽量 accumulator で実装しています。

実装箇所:

- `src/clearscan_cv/geometry.py`: `detect_hough_document_region`, `detect_fallback_document_region`
- `src/clearscan_cv/static/index.html`: `detectHoughLineQuad`
- テスト: 断裂した紙面エッジだけを持つ合成ページで、`hough_lines` fallback が四隅を復元することを確認

参照した考え方:

- Jagannathan and Jawahar, "Perspective Correction Methods for Camera-Based Document Analysis", CBDAR 2005: 文書境界、文字行、レイアウト alignment を透視補正の手掛かりに使う。
- Yin et al., "A Multi-Stage Strategy to Perspective Rectification for Mobile Phone Camera-Based Document Images", ICDAR 2007: モバイル撮影文書で、境界線だけでなく text baselines や block alignment を段階的に使う。
- Feng et al., "Geometric Representation Learning for Document Image Rectification", ECCV 2022: textlines を局所的な幾何制約として扱う。
- Verhoeven et al., "UVDoc: Neural Grid-based Document Unwarping", SIGGRAPH Asia 2023: unwarping 評価で line straightness を重視する。

### 局所背景推定と stroke-aware 二値化

透視補正と deskew の後に、紙面の大域的な明るさではなく局所背景を推定して正規化し、文字候補は「背景との差分」「局所コントラスト」「エッジ勾配」「濃いインク」の複数条件で判定します。これにより、影や紙のしわ・テクスチャを黒いノイズとして拾いにくくします。

実装箇所:

- `src/clearscan_cv/pipeline.py`: `estimate_luminance_background`, `normalize_shadow_luminance`, `to_clean_binary`
- `src/clearscan_cv/static/index.html`: tile background 推定、`paperNoiseGuard`, `strokeContrast`
- テスト: 影付き・紙纹ノイズ付きの合成ページで、空白領域を白く保ちつつ文字領域を残すことを確認

参照した考え方:

- Anvari and Athitsos, "A Survey on Deep learning based Document Image Enhancement", 2021: 文書增强を二値化、去影、去ノイズ、文字可読性改善などの複合問題として整理。
- Li et al., "Document Rectification and Illumination Correction using a Patch-based CNN", 2019: 幾何補正と光照補正を分離し、局所領域で補正する設計。
- Feng et al., "DocTr", 2021: 幾何 unwarping と illumination correction を連結し、OCR 入力品質を上げる設計。

## 論文別メモ

| 領域 | 論文 | このプロジェクトでの扱い |
| --- | --- | --- |
| 総論 | Liang, Doermann, Li, "Camera-based analysis of text and documents: a survey", IJDAR 2005 | カメラ文書特有の透視歪み、低解像度、ぼけ、背景混入を前提問題として整理。 |
| 画像強調 | Anvari and Athitsos, "A Survey on Deep learning based Document Image Enhancement", arXiv 2021 | 二値化、去ぼけ、ノイズ除去、影除去などの分類を参考に、局所背景推定と stroke-aware 二値化を実装。 |
| 文書解析 | Zhang et al., "Document Parsing Unveiled", arXiv 2024 | OCR 後の Markdown 復元、段組み推定、将来の JSON/Word 出力のロードマップに反映。 |
| 透視補正 | Jagannathan and Jawahar, "Perspective Correction Methods for Camera-Based Document Analysis", CBDAR 2005 | Canny/contour に加え、文字行 deskew を追加。 |
| モバイル透視補正 | Yin et al., "A Multi-Stage Strategy to Perspective Rectification for Mobile Phone Camera-Based Document Images", ICDAR 2007 | 境界検出、text baseline、block alignment を段階的に使う思想を、端末内の軽量 deskew として採用。 |
| 深度 dewarp | Ma et al., "DocUNet", CVPR 2018 | 将来の曲面補正候補。現版ではモデルを同梱せず、四点透視補正と小角度 deskew に留める。 |
| 3D dewarp | Das et al., "DewarpNet", ICCV 2019 | 3D shape ベースの dewarp は将来の別モデル候補。端末内版では未実装。 |
| patch flow + 光照 | Li et al., "Document Rectification and Illumination Correction using a Patch-based CNN", SIGGRAPH Asia 2019 | 幾何補正と照明補正を分ける設計を参考に、透視/deskew の後に局所背景正規化を実行。 |
| Transformer | Feng et al., "DocTr", ACM MM 2021 | 幾何補正と光照補正を両方扱う構成を参考に、現版では軽量版として Canvas の去影・文字強調処理に分解。 |
| progressive | Feng et al., "DocScanner", arXiv 2021 / IJCV 2025 | 反復的に rectified image を改善する方向は、将来の複数段補正候補。 |
| textline geometry | Feng et al., "DocGeoNet", ECCV 2022 | textline を幾何制約として扱う考えを、今回の文字行 deskew に反映。 |
| grid unwarping | Verhoeven et al., "UVDoc", SIGGRAPH Asia 2023 | line straightness 評価を参考に、角度補正を品質指標として表示。 |
| unrestricted rectification | Feng et al., "DocTr++", arXiv 2023 | 文書全体が写っていないケースを扱う方向。現版では誤裁断防止と手動四隅補正で対応。 |

## 出典

- Liang, J., Doermann, D., Li, H. "Camera-based analysis of text and documents: a survey." IJDAR, 2005. https://doi.org/10.1007/s10032-004-0138-z
- Anvari, Z., Athitsos, V. "A Survey on Deep learning based Document Image Enhancement." arXiv:2112.02719, 2021. https://arxiv.org/abs/2112.02719
- Zhang, Q. et al. "Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Information Extraction." arXiv:2410.21169, 2024. https://arxiv.org/abs/2410.21169
- Jagannathan, L., Jawahar, C. V. "Perspective Correction Methods for Camera-Based Document Analysis." CBDAR, 2005. https://cvit.iiit.ac.in/images/ConferencePapers/2005/jagannathan05Perspective.pdf
- Yin, X.-C. et al. "A Multi-Stage Strategy to Perspective Rectification for Mobile Phone Camera-Based Document Images." ICDAR, 2007. https://ieeexplore.ieee.org/document/4376980/
- Ma, K. et al. "DocUNet: Document Image Unwarping via a Stacked U-Net." CVPR, 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Ma_DocUNet_Document_Image_CVPR_2018_paper.html
- Das, S. et al. "DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks." ICCV, 2019. https://openaccess.thecvf.com/content_ICCV_2019/papers/Das_DewarpNet_Single-Image_Document_Unwarping_With_Stacked_3D_and_2D_Regression_ICCV_2019_paper.pdf
- Li, X. et al. "Document Rectification and Illumination Correction using a Patch-based CNN." ACM TOG, 2019. https://arxiv.org/abs/1909.09470
- Feng, H. et al. "DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction." ACM MM, 2021. https://arxiv.org/abs/2110.12942
- Feng, H. et al. "DocScanner: Robust Document Image Rectification with Progressive Learning." arXiv:2110.14968, 2021; IJCV, 2025. https://arxiv.org/abs/2110.14968
- Feng, H. et al. "Geometric Representation Learning for Document Image Rectification." ECCV, 2022. https://arxiv.org/abs/2210.08161
- Verhoeven, F. et al. "UVDoc: Neural Grid-based Document Unwarping." SIGGRAPH Asia, 2023. https://arxiv.org/abs/2302.02887
- Feng, H. et al. "DocTr++: Deep Unrestricted Document Image Rectification." arXiv:2304.08796, 2023. https://arxiv.org/abs/2304.08796
