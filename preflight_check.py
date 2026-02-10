from __future__ import annotations

import importlib
import sys
from pathlib import Path

REQUIRED_MODULES = ["streamlit", "pandas", "bs4", "playwright"]


def main() -> int:
    errors: list[str] = []
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[OK] import {module_name}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[NG] import {module_name}: {exc}")

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            chromium_path = Path(playwright.chromium.executable_path)
            if chromium_path.exists():
                print(f"[OK] Chromium executable: {chromium_path}")
            else:
                errors.append(
                    f"[NG] Chromium executable not found: {chromium_path}"
                )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[NG] Playwright runtime check failed: {exc}")

    if errors:
        print("\nPreflight check failed:")
        for error in errors:
            print(error)
        print("\nSuggested commands:")
        print("python -m pip install -r requirements.txt")
        print("python -m playwright install chromium")
        return 1

    print("\nPreflight check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
