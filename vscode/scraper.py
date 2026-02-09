from __future__ import annotations

import asyncio
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import BrowserContext, async_playwright

JST = timezone(timedelta(hours=9))
SIZE_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*帖")
PRICE_PATTERN = re.compile(r"([0-9][0-9,]*)\s*円")

ProgressCallback = Callable[[int, int, str, bool, str], None]
BROWSER_INSTALL_LOCK = threading.Lock()
CHROMIUM_READY = False
BROWSER_LAUNCH_ARGS = ["--disable-dev-shm-usage", "--no-sandbox"]


@dataclass
class HeaderMap:
    status_idx: int | None = None
    room_idx: int | None = None
    size_idx: int | None = None
    price_idx: int | None = None
    dims_indices: list[int] = field(default_factory=list)


def current_jst_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_header(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("　", ""))


def _safe_get(values: list[str], idx: int | None) -> str:
    if idx is None:
        return ""
    if idx < 0 or idx >= len(values):
        return ""
    return values[idx]


def _parse_size_jo(text: str) -> float | None:
    match = SIZE_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_prices(text: str) -> tuple[int | None, int | None]:
    values: list[int] = []
    for raw in PRICE_PATTERN.findall(text):
        try:
            values.append(int(raw.replace(",", "")))
        except ValueError:
            continue

    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    return values[0], values[1]


def _classify_status(status_text: str) -> str | None:
    if "お見積" in status_text:
        return "空室"
    if "お問合せ" in status_text or "お問い合わせ" in status_text:
        return "要確認"
    return None


def _map_headers(headers: list[str]) -> HeaderMap:
    mapped = HeaderMap()
    for idx, header in enumerate(headers):
        h = _normalize_header(header)
        hl = h.lower()

        if mapped.status_idx is None and "空き状況" in h:
            mapped.status_idx = idx

        if mapped.room_idx is None and (
            "部屋番号" in h
            or ("部屋" in h and ("番号" in h or "no" in hl))
            or h == "部屋"
        ):
            mapped.room_idx = idx

        if mapped.size_idx is None and ("広さ" in h or "帖" in h):
            mapped.size_idx = idx

        if mapped.price_idx is None and any(key in h for key in ["月額使用料", "月額", "使用料", "料金"]):
            mapped.price_idx = idx

        if any(key in h for key in ["幅", "奥行", "高さ"]):
            mapped.dims_indices.append(idx)

    return mapped


def _extract_headers_and_rows(table: Any) -> tuple[list[str], list[Any]]:
    rows = table.find_all("tr")
    if not rows:
        return [], []

    header_index = 0
    for i, row in enumerate(rows):
        if row.find_all("th"):
            header_index = i
            break

    header_cells = rows[header_index].find_all(["th", "td"])
    headers = [_normalize_text(cell.get_text(" ", strip=True)) for cell in header_cells]
    data_rows = rows[header_index + 1 :]
    return headers, data_rows


def parse_store_page(html: str, store: dict[str, str], fetched_at: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rooms: list[dict[str, Any]] = []

    for table in soup.find_all("table"):
        headers, data_rows = _extract_headers_and_rows(table)
        if not headers:
            continue

        mapping = _map_headers(headers)
        if mapping.status_idx is None:
            continue

        for row in data_rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            cell_texts = [_normalize_text(cell.get_text(" ", strip=True)) for cell in cells]
            status_text = _safe_get(cell_texts, mapping.status_idx)
            status_label = _classify_status(status_text)
            if status_label is None:
                continue

            room_no = _safe_get(cell_texts, mapping.room_idx)
            size_text = _safe_get(cell_texts, mapping.size_idx)
            price_text = _safe_get(cell_texts, mapping.price_idx)

            size_jo = _parse_size_jo(size_text)
            price_normal, price_discounted = _parse_prices(price_text)
            if price_normal is None:
                price_normal, price_discounted = _parse_prices(" ".join(cell_texts))

            dims_parts = [
                _safe_get(cell_texts, idx)
                for idx in mapping.dims_indices
                if _safe_get(cell_texts, idx)
            ]
            dims = " / ".join(dict.fromkeys(dims_parts)) if dims_parts else ""

            rooms.append(
                {
                    "store_name": store["name"],
                    "store_url": store["url"],
                    "room_no": room_no,
                    "size_jo": size_jo,
                    "price_normal": price_normal,
                    "price_discounted": price_discounted,
                    "dims": dims,
                    "status_label": status_label,
                    "fetched_at": fetched_at,
                }
            )

    return rooms


def _append_cache_buster(url: str) -> str:
    ts = int(time.time() * 1000)
    delimiter = "&" if "?" in url else "?"
    return f"{url}{delimiter}cb={ts}"


def _short_error(exc: Exception, limit: int = 120) -> str:
    text = str(exc).replace("\n", " ").strip()
    if not text:
        text = exc.__class__.__name__
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def _is_missing_browser_error(exc: Exception) -> bool:
    text = str(exc)
    return "Executable doesn't exist" in text or "Please run the following command to download new browsers" in text


def _install_chromium_once() -> None:
    global CHROMIUM_READY
    if CHROMIUM_READY:
        return

    with BROWSER_INSTALL_LOCK:
        if CHROMIUM_READY:
            return

        env = dict(os.environ)
        env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / ".cache" / "ms-playwright"))

        cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            if len(detail) > 300:
                detail = detail[:300] + "..."
            raise RuntimeError(f"Chromium install failed: {detail}")

        CHROMIUM_READY = True


async def _fetch_store_html(context: BrowserContext, store_url: str, timeout_ms: int) -> str:
    page = await context.new_page()
    try:
        target_url = _append_cache_buster(store_url)
        try:
            await page.goto(target_url, wait_until="networkidle", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            # 一部店舗ページは解析用テーブルが表示されていても networkidle が成立しない場合がある。
            # その場合のみ domcontentloaded で再取得して解析を継続する。
            await page.goto(target_url, wait_until="domcontentloaded", timeout=timeout_ms)

        await page.wait_for_selector("table", timeout=timeout_ms)
        try:
            await page.wait_for_function(
                """
                () => Array.from(document.querySelectorAll('table')).some(
                    (table) => /空き状況/.test((table.innerText || ''))
                )
                """,
                timeout=min(7000, timeout_ms),
            )
        except PlaywrightTimeoutError:
            # テーブル見出しが見つからない店舗ページでも、取得済みHTMLで解析を試みる。
            pass

        return await page.content()
    finally:
        await page.close()


async def _fetch_single_store(
    context: BrowserContext,
    store: dict[str, str],
    fetched_at: str,
    timeout_ms: int,
    semaphore: asyncio.Semaphore,
) -> tuple[list[dict[str, Any]], str | None]:
    await asyncio.sleep(random.uniform(0.3, 1.0))
    async with semaphore:
        try:
            html = await _fetch_store_html(context, store["url"], timeout_ms)
            return parse_store_page(html, store, fetched_at), None
        except PlaywrightTimeoutError:
            return [], f"タイムアウト({timeout_ms // 1000}秒)"
        except Exception as exc:
            return [], _short_error(exc)


async def _fetch_all_async(
    stores: list[dict[str, str]],
    fetched_at: str,
    timeout_ms: int,
    max_concurrency: int,
    progress_callback: ProgressCallback | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    total = len(stores)
    all_rooms: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    completed = 0
    lock = asyncio.Lock()

    async with async_playwright() as playwright:
        global CHROMIUM_READY
        try:
            browser = await playwright.chromium.launch(headless=True, args=BROWSER_LAUNCH_ARGS)
            CHROMIUM_READY = True
        except Exception as exc:
            if not _is_missing_browser_error(exc):
                raise

            _install_chromium_once()
            browser = await playwright.chromium.launch(headless=True, args=BROWSER_LAUNCH_ARGS)
            CHROMIUM_READY = True
        context = await browser.new_context(
            extra_http_headers={
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
        )

        async def run_for_store(store: dict[str, str]) -> tuple[dict[str, str], list[dict[str, Any]], str | None]:
            nonlocal completed
            rooms, error = await _fetch_single_store(
                context=context,
                store=store,
                fetched_at=fetched_at,
                timeout_ms=timeout_ms,
                semaphore=semaphore,
            )
            async with lock:
                completed += 1
                if progress_callback:
                    progress_callback(
                        completed,
                        total,
                        store["name"],
                        error is None,
                        error or "",
                    )
            return store, rooms, error

        try:
            tasks = [asyncio.create_task(run_for_store(store)) for store in stores]
            for task in asyncio.as_completed(tasks):
                store, rooms, error = await task
                all_rooms.extend(rooms)
                if error:
                    errors.append(
                        {
                            "store_name": store["name"],
                            "store_url": store["url"],
                            "error": error,
                        }
                    )
        finally:
            await context.close()
            await browser.close()

    return all_rooms, errors


def _run_async_in_new_loop(coro_factory: Callable[[], Any]) -> Any:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro_factory())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _run_async_with_windows_proactor(coro_factory: Callable[[], Any]) -> Any:
    proactor_policy_cls = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
    if proactor_policy_cls is None:
        return _run_async_in_new_loop(coro_factory)

    original_policy = asyncio.get_event_loop_policy()
    try:
        asyncio.set_event_loop_policy(proactor_policy_cls())
        return _run_async_in_new_loop(coro_factory)
    finally:
        asyncio.set_event_loop_policy(original_policy)


def _run_async(coro_factory: Callable[[], Any]) -> Any:
    if sys.platform == "win32":
        selector_policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if selector_policy_cls is not None and isinstance(asyncio.get_event_loop_policy(), selector_policy_cls):
            return _run_async_with_windows_proactor(coro_factory)

    try:
        return asyncio.run(coro_factory())
    except RuntimeError as exc:
        if "asyncio.run()" not in str(exc):
            raise
        try:
            if sys.platform == "win32":
                return _run_async_with_windows_proactor(coro_factory)
            return _run_async_in_new_loop(coro_factory)
        except NotImplementedError:
            if sys.platform == "win32":
                return _run_async_with_windows_proactor(coro_factory)
            raise
    except NotImplementedError:
        if sys.platform == "win32":
            return _run_async_with_windows_proactor(coro_factory)
        raise


def fetch_stores_latest(
    stores: list[dict[str, str]],
    progress_callback: ProgressCallback | None = None,
    timeout_sec: int = 25,
    max_concurrency: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], str]:
    """
    店舗ページを都度取得し、空室候補の部屋一覧を返す。

    Returns:
        rooms: 取得できた部屋データ
        errors: 取得失敗した店舗情報
        fetched_at: 検索開始時刻（日本時間）
    """
    fetched_at = current_jst_str()
    if not stores:
        return [], [], fetched_at

    timeout_ms = max(5000, int(timeout_sec * 1000))

    rooms, errors = _run_async(
        lambda: _fetch_all_async(
            stores=stores,
            fetched_at=fetched_at,
            timeout_ms=timeout_ms,
            max_concurrency=max_concurrency,
            progress_callback=progress_callback,
        )
    )

    return rooms, errors, fetched_at
