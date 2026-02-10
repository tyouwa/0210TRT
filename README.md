# トランクルーム東京 空室検索アプリ（社内向け）

## 概要
`trunkroomtokyo.jp` の各店舗ページを、検索ボタン押下のたびに都度取得して空室候補を一覧表示する Streamlit アプリです。

- サイト改修なし
- 追加データベースなし（結果の永続保存なし）
- 取得時刻（日本時間）を画面表示
- 画面の表示結果をそのまま CSV ダウンロード

## ファイル構成
- `app.py`: Streamlit 画面、検索条件、表示、CSV出力
- `scraper.py`: Playwright 取得、BeautifulSoup 解析、空室判定
- `stores.json`: 固定の対象店舗リスト
- `requirements.txt`: 依存パッケージ
- `packages.txt`: Streamlit Community Cloud 用の OS 依存パッケージ
- `preflight_check.py`: 依存・Chromium の事前チェック
- `README.md`: 起動手順

## 前提
- Python 3.10 以上

## セットアップ
1. 仮想環境を作成
```powershell
python -m venv .venv
```

2. 仮想環境を有効化
```powershell
.\.venv\Scripts\Activate.ps1
```

3. 依存パッケージをインストール
```powershell
python -m pip install -r requirements.txt
```

4. Playwright の Chromium をインストール
```powershell
python -m playwright install chromium
```

5. 事前チェック（推奨）
```powershell
python preflight_check.py
```

## 起動
```powershell
python -m streamlit run app.py
```

## ログイン
- 起動後、パスワード入力画面が表示されます。
- パスワードは社内管理者から共有されたものを入力してください。

## Streamlit Community Cloud 公開時の補足
- このリポジトリには `requirements.txt` に加えて `packages.txt` を含めています。
- `packages.txt` は Playwright (Chromium) 実行に必要な Linux 依存を導入するためのものです。
- Main file path を `vscode/app.py` にする構成では、`packages.txt` をリポジトリ直下と `vscode/` 配下の両方に置いてください。
- 初回起動時に Chromium バイナリが未導入の場合、アプリ側で `playwright install chromium` を自動実行して再試行します。

## 使い方
1. 対象店舗（複数可）を選択
2. 広さ検索モードを選択
3. `範囲検索` の場合は「最低広さ（帖）」「最高広さ（帖）」を入力（未入力可）
4. `完全一致` の場合は「完全一致広さ（帖）」を入力
5. 予算上限（円/月）を必要に応じて入力
6. 価格基準（割引後/通常料金）を選択
7. `検索（最新取得）` を押す
8. 一覧を確認し、必要なら `CSVダウンロード`

## 実装上のポイント
- 毎回全店舗を再取得（検索ボタン押下時）
- キャッシュ回避の工夫
  - URL に `?cb=<UNIXミリ秒>` を付与
  - HTTP ヘッダーに `Cache-Control: no-cache`, `Pragma: no-cache`
  - 検索ごとに新規ブラウザコンテキストを作成して終了時に破棄
- 取得待機
  - `page.goto(..., wait_until="networkidle")`
  - `table` 要素待機 + 見出しに `空き状況` を含む表を優先
  - 1店舗あたり 25 秒タイムアウト
- 負荷対策
  - 同時取得は最大 3 件
  - 店舗ごとに 0.3〜1.0 秒のランダム待機
- 空室判定
  - `お見積` を含む: `空室`
  - `お問合せ` / `お問い合わせ` を含む: `要確認`（内部判定のみ。画面表示対象外）
  - それ以外: 非表示（将来拡張しやすい構造）

## 補足
- 取得失敗した店舗は、店舗名と理由を画面に表示します。
- 解析はヘッダー文字列ベースで列マッピングするため、列順の変更にある程度追随できます。
