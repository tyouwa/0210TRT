from __future__ import annotations

import hmac
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from scraper import fetch_stores_latest

STORE_FILE = Path(__file__).with_name("stores.json")
APP_PASSWORD = "centric01"
MAX_LOGIN_ATTEMPTS = 5
LOCK_SECONDS = 60

PRICE_MODE_OPTIONS = {
    "割引後（表示が “A円 → B円” のときはBを使う）": "discounted",
    "通常料金（表示が “A円 → B円” のときはAを使う）": "normal",
}

DISPLAY_COLUMNS = [
    "店舗名",
    "店舗URL",
    "部屋番号",
    "広さ（帖）",
    "月額（通常料金）",
    "月額（割引後）",
    "幅・奥行・高さ",
    "取得時刻",
]


def load_stores() -> list[dict[str, str]]:
    if not STORE_FILE.exists():
        raise FileNotFoundError(f"店舗リストファイルが見つかりません: {STORE_FILE}")

    text = STORE_FILE.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("stores.json は配列形式である必要があります。")

    stores: list[dict[str, str]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        url = str(row.get("url", "")).strip()
        enabled = bool(row.get("enabled", True))
        if not enabled:
            continue
        if not name or not url:
            continue
        stores.append({"name": name, "url": url})

    return stores


def parse_size_or_none(value: str) -> float | None:
    raw = value.strip().replace(",", "")
    if not raw:
        return None
    size = float(raw)
    return max(size, 0.0)


def parse_budget(value: str) -> int | None:
    raw = value.strip().replace(",", "")
    if not raw:
        return None
    budget = int(raw)
    return max(budget, 0)


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "store_name",
                "store_url",
                "room_no",
                "size_jo",
                "size_text_raw",
                "price_normal",
                "price_discounted",
                "dims",
                "status_label",
                "fetched_at",
            ]
        )

    df = pd.DataFrame(rows)
    df["size_jo"] = pd.to_numeric(df.get("size_jo"), errors="coerce")
    if "size_text_raw" not in df.columns:
        df["size_text_raw"] = ""
    else:
        df["size_text_raw"] = df["size_text_raw"].fillna("").astype(str)
    df["price_normal"] = pd.to_numeric(df.get("price_normal"), errors="coerce").astype("Int64")
    df["price_discounted"] = pd.to_numeric(df.get("price_discounted"), errors="coerce").astype("Int64")
    return df


def filter_and_sort(
    df: pd.DataFrame,
    min_size: float,
    max_size: float | None,
    exact_size: float | None,
    budget: int | None,
    price_mode: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df[df["status_label"].isin(["空室"])].copy()
    filtered["size_jo"] = pd.to_numeric(filtered["size_jo"], errors="coerce")
    if "size_text_raw" not in filtered.columns:
        filtered["size_text_raw"] = ""
    else:
        filtered["size_text_raw"] = filtered["size_text_raw"].fillna("").astype(str)

    # 検索判定は実際の帖数値を使う。数値欠損時のみ生テキストから補完する。
    size_from_text = pd.to_numeric(
        filtered["size_text_raw"].str.extract(r"([0-9]+(?:\.[0-9]+)?)\s*帖", expand=False),
        errors="coerce",
    )
    filtered["match_size_jo"] = filtered["size_jo"].fillna(size_from_text)

    if exact_size is not None:
        filtered = filtered[
            filtered["match_size_jo"].notna()
            & ((filtered["match_size_jo"] - exact_size).abs() <= 1e-9)
        ]
    else:
        filtered = filtered[filtered["match_size_jo"].fillna(0) >= min_size]
        if max_size is not None:
            filtered = filtered[
                filtered["match_size_jo"].notna() & (filtered["match_size_jo"] <= max_size)
            ]

    sort_price_col = "price_discounted" if price_mode == "discounted" else "price_normal"
    filtered["sort_price"] = pd.to_numeric(filtered[sort_price_col], errors="coerce")
    filtered["sort_size"] = pd.to_numeric(filtered["match_size_jo"], errors="coerce")

    if budget is not None:
        filtered = filtered[(filtered["sort_price"].notna()) & (filtered["sort_price"] <= budget)]

    filtered = filtered.sort_values(
        by=["sort_price", "sort_size", "store_name", "room_no"],
        ascending=[True, False, True, True],
        na_position="last",
    )
    return filtered


def to_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    display_df = pd.DataFrame(
        {
            "店舗名": df["store_name"],
            "店舗URL": df["store_url"],
            "部屋番号": df["room_no"],
            "広さ（帖）": df["size_jo"],
            "月額（通常料金）": df["price_normal"],
            "月額（割引後）": df["price_discounted"],
            "幅・奥行・高さ": df["dims"],
            "取得時刻": df["fetched_at"],
        }
    )

    display_df["広さ（帖）"] = pd.to_numeric(display_df["広さ（帖）"], errors="coerce").round(2)
    if "size_text_raw" in df.columns:
        size_text = (
            df["size_text_raw"]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        has_size_text = size_text.str.contains("帖", na=False)
        display_df.loc[has_size_text, "広さ（帖）"] = size_text[has_size_text]

    display_df["月額（通常料金）"] = pd.to_numeric(display_df["月額（通常料金）"], errors="coerce").astype("Int64")
    display_df["月額（割引後）"] = pd.to_numeric(display_df["月額（割引後）"], errors="coerce").astype("Int64")
    return display_df


def sanitize_timestamp_for_filename(timestamp: str) -> str:
    return timestamp.replace(" ", "_").replace(":", "").replace("/", "-")


def build_fetch_error_message(exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    hints: list[str] = []

    lower_detail = detail.lower()
    if "chromium install failed" in lower_detail or "playwright" in lower_detail:
        hints.append("Playwright/Chromium の初期化に失敗しています。")
        hints.append("`python -m playwright install chromium` を実行して再試行してください。")
    if "no module named" in lower_detail:
        hints.append("依存パッケージが不足している可能性があります。")
        hints.append("`python -m pip install -r requirements.txt` を実行してください。")

    message = f"取得処理でエラーが発生しました: {detail}"
    if hints:
        message += "\n\n" + "\n".join(hints)
    return message


def get_app_password() -> str | None:
    return APP_PASSWORD


def init_session_state() -> None:
    defaults = {
        "display_df": None,
        "fetched_at": "",
        "errors": [],
        "is_authenticated": False,
        "login_attempts": 0,
        "lock_until": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_password_gate() -> None:
    configured_password = get_app_password()

    if st.session_state.get("is_authenticated", False):
        with st.sidebar:
            if st.button("ログアウト"):
                st.session_state["is_authenticated"] = False
                st.rerun()
        return

    lock_until = float(st.session_state.get("lock_until", 0.0))
    now = time.time()
    if lock_until > now:
        remaining = int(lock_until - now) + 1
        st.warning(f"ログイン試行回数が上限に達したため、{remaining}秒後に再試行してください。")
        st.stop()

    st.subheader("ログイン")
    st.info("このアプリを利用するにはパスワードの入力が必要です。")
    with st.form("password_form", clear_on_submit=True):
        password = st.text_input("パスワード", type="password")
        submitted = st.form_submit_button("ログイン")

    if submitted:
        if hmac.compare_digest(password, configured_password):
            st.session_state["is_authenticated"] = True
            st.session_state["login_attempts"] = 0
            st.session_state["lock_until"] = 0.0
            st.rerun()
        else:
            attempts = int(st.session_state.get("login_attempts", 0)) + 1
            st.session_state["login_attempts"] = attempts
            if attempts >= MAX_LOGIN_ATTEMPTS:
                st.session_state["login_attempts"] = 0
                st.session_state["lock_until"] = time.time() + LOCK_SECONDS
                st.error("パスワードが違います。しばらく待ってから再試行してください。")
                st.stop()

            remaining = MAX_LOGIN_ATTEMPTS - attempts
            st.error("パスワードが違います。")
            st.caption(f"あと {remaining} 回失敗すると一時ロックされます。")

    st.stop()


def main() -> None:
    st.set_page_config(page_title="トランクルーム　空室検索", layout="wide")
    st.title("トランクルーム　空室検索")
    st.caption("検索ボタン押下時に店舗ページを都度取得します。表示時刻は日本時間です。")

    init_session_state()
    render_password_gate()

    try:
        stores = load_stores()
    except Exception as exc:
        st.error(f"店舗リストの読み込みに失敗しました: {exc}")
        st.stop()

    if not stores:
        st.warning("有効な店舗が登録されていません。`stores.json` を確認してください。")
        st.stop()

    store_names = [store["name"] for store in stores]

    selected_store_names = st.multiselect(
        "対象店舗（複数選択）",
        options=store_names,
        default=store_names,
    )

    size_search_mode = st.radio(
        "広さ検索モード",
        options=["範囲検索（最低〜最高）", "完全一致（指定値）"],
        index=0,
        horizontal=True,
    )

    if size_search_mode == "範囲検索（最低〜最高）":
        col1, col2, col3 = st.columns(3)
        with col1:
            min_size_text = st.text_input("最低広さ（帖）", value="", placeholder="未入力なら 0")
        with col2:
            max_size_text = st.text_input("最高広さ（帖）", value="", placeholder="未入力なら上限なし")
        with col3:
            budget_text = st.text_input("予算上限（円/月）", value="", placeholder="未入力なら予算で絞り込まない")
        exact_size_text = ""
    else:
        col1, col2 = st.columns(2)
        with col1:
            exact_size_text = st.text_input("完全一致広さ（帖）", value="", placeholder="例: 4.0")
        with col2:
            budget_text = st.text_input("予算上限（円/月）", value="", placeholder="未入力なら予算で絞り込まない")
        min_size_text = ""
        max_size_text = ""

    price_mode_label = st.radio("価格の使い方", options=list(PRICE_MODE_OPTIONS.keys()), index=0)

    search_clicked = st.button("検索（最新取得）", type="primary")

    if search_clicked:
        if not selected_store_names:
            st.warning("対象店舗を1つ以上選択してください。")
        else:
            min_size = 0.0
            max_size = None
            exact_size = None

            if size_search_mode == "範囲検索（最低〜最高）":
                try:
                    parsed_min = parse_size_or_none(min_size_text)
                    parsed_max = parse_size_or_none(max_size_text)
                except ValueError:
                    st.error("最低広さ・最高広さ（帖）は数値で入力してください。")
                    st.stop()

                min_size = parsed_min if parsed_min is not None else 0.0
                max_size = parsed_max

                if max_size is not None and max_size < min_size:
                    st.error("最高広さ（帖）は最低広さ（帖）以上で入力してください。")
                    st.stop()
            else:
                try:
                    exact_size = parse_size_or_none(exact_size_text)
                except ValueError:
                    st.error("完全一致広さ（帖）は数値で入力してください。")
                    st.stop()

                if exact_size is None:
                    st.error("完全一致検索では広さ（帖）を入力してください。")
                    st.stop()

            try:
                budget = parse_budget(budget_text)
            except ValueError:
                st.error("予算上限（円/月）は整数で入力してください。")
                st.stop()

            selected_stores = [store for store in stores if store["name"] in selected_store_names]
            total = len(selected_stores)

            progress_text = st.empty()
            progress_bar = st.progress(0.0)

            def on_progress(done: int, all_count: int, store_name: str, ok: bool, message: str) -> None:
                state = "完了" if ok else f"失敗: {message}"
                progress_text.info(f"取得中 {done}/{all_count} - {store_name} ({state})")
                progress_bar.progress(done / all_count)

            try:
                rooms, errors, fetched_at = fetch_stores_latest(
                    stores=selected_stores,
                    progress_callback=on_progress,
                    timeout_sec=25,
                    max_concurrency=3,
                )
            except Exception as exc:
                st.session_state["display_df"] = None
                st.session_state["errors"] = []
                st.session_state["fetched_at"] = ""
                progress_text.empty()
                progress_bar.empty()
                st.error(build_fetch_error_message(exc))
                st.stop()

            progress_bar.progress(1.0)
            progress_text.success(f"取得完了: {total}店舗")

            raw_df = rows_to_dataframe(rooms)
            filtered_df = filter_and_sort(
                df=raw_df,
                min_size=min_size,
                max_size=max_size,
                exact_size=exact_size,
                budget=budget,
                price_mode=PRICE_MODE_OPTIONS[price_mode_label],
            )
            display_df = to_display_dataframe(filtered_df)

            st.session_state["display_df"] = display_df
            st.session_state["errors"] = errors
            st.session_state["fetched_at"] = fetched_at

    fetched_at = st.session_state.get("fetched_at", "")
    display_df = st.session_state.get("display_df")
    errors = st.session_state.get("errors", [])

    if fetched_at:
        st.markdown(f"**取得時刻（日本時間）:** {fetched_at}")

    if errors:
        st.warning(f"取得失敗: {len(errors)}店舗")
        error_df = pd.DataFrame(errors).rename(
            columns={
                "store_name": "店舗名",
                "store_url": "店舗URL",
                "error": "理由",
            }
        )
        st.dataframe(
            error_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "店舗URL": st.column_config.LinkColumn("店舗URL"),
            },
        )

    if display_df is not None:
        st.write(f"表示件数: {len(display_df)}件")
        if display_df.empty:
            st.info("条件に合う空室データはありませんでした。")
        else:
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "店舗URL": st.column_config.LinkColumn("店舗URL"),
                },
            )

            csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
            filename_ts = sanitize_timestamp_for_filename(fetched_at or "result")
            st.download_button(
                label="CSVダウンロード",
                data=csv_data,
                file_name=f"trunkroom_search_{filename_ts}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
