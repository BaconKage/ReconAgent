"""Regenerate the README screenshots from the live app.

Committed images go stale silently: the UI changes, the picture in the README
keeps showing the old one, and nobody notices because a screenshot is not
something a test suite looks at. This script makes them reproducible, so a
screenshot is a build artifact rather than a souvenir.

    pip install playwright && python -m playwright install chromium
    python docs/capture_screenshots.py

It starts the Streamlit app on a spare port, drives it with a headless browser,
and writes docs/*.png. Nothing here is imported by the application.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
PORT = 8765

#: (tab label, element to wait for, output file). The wait target matters: the
#: app renders its shell before the data arrives, and screenshotting too early
#: captures an empty panel that looks like a bug.
SHOTS = [
    (None, "Auto-match rate", "overview.png"),
    ("Exceptions", "What the engine did", "exceptions.png"),
    ("Cash position", "In the account", "cash.png"),
]


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:\n"
              "    pip install playwright\n"
              "    python -m playwright install chromium")
        return 1

    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.port", str(PORT)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(18)  # Streamlit's first render includes a full reconciliation.
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000},
                                    device_scale_factor=2, color_scheme="dark")
            page.goto(f"http://localhost:{PORT}", wait_until="networkidle",
                      timeout=90_000)
            for tab, marker, filename in SHOTS:
                if tab:
                    page.get_by_role("tab", name=tab).click()
                page.wait_for_selector(f"text={marker}", timeout=60_000)
                time.sleep(3)  # let charts and tables finish painting
                page.screenshot(path=str(DOCS / filename))
                print(f"  wrote docs/{filename}")
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
