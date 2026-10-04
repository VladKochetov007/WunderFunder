from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    """Check offline rendering, filters, downloads and narrow-screen layout."""
    output = Path(__file__).resolve().parent
    history = json.loads((output / "leaderboard_history.json").read_text())
    errors: list[str] = []
    network: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1050})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                network.append(request.url) if request.url.startswith(("http:", "https:")) else None
            ),
        )
        page.goto((output / "leaderboard_climb.html").as_uri())
        assert page.locator("#ledger-body tr[id]").count() == len(history["rows"])
        assert "0.6660" in page.locator(".cards").inner_text()
        page.screenshot(path=str(output / "leaderboard_climb_desktop.png"), full_page=True)
        with page.expect_download() as downloaded:
            page.locator("#save-csv").click()
        downloaded.value.save_as(output / "leaderboard_climb.csv")
        with (output / "leaderboard_climb.csv").open(newline="") as source:
            csv_rows = list(csv.DictReader(source))
        assert len(csv_rows) == len(history["rows"])
        best_csv = next(row for row in csv_rows if row["id"] == "2e5j6bnt")
        assert float(best_csv["public"]) == 0.6660
        with page.expect_download() as downloaded:
            page.locator("#save-json").click()
        downloaded.value.save_as(output / "browser_download_history.json")
        assert json.loads((output / "browser_download_history.json").read_text()) == history
        page.locator("#public-only").click()
        assert not page.locator('[data-series="validation_full"]').is_checked()
        page.locator("#range").select_option("latest")
        page.locator("#spacing").select_option("time")
        page.screenshot(path=str(output / "leaderboard_climb_latest.png"), full_page=False)
        page.locator("#rows-filter").select_option("records")
        assert page.locator("#ledger-body tr[id]").count() == sum(
            bool(row.get("new_record")) for row in history["rows"]
        )
        page.locator("#search").fill("combo_s4")
        assert page.locator("#ledger-body tr[id]").count() == 1
        page.locator("#search").fill("")
        page.locator("#rows-filter").select_option("error")
        assert page.locator("#ledger-body tr[id]").count() == 1
        page.locator("#rows-filter").select_option("all")
        page.locator("#range").select_option("all")
        page.locator("#compare").click()
        assert page.locator('[data-series="validation_full"]').is_checked()
        page.set_viewport_size({"width": 420, "height": 900})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path=str(output / "leaderboard_climb_mobile.png"), full_page=False)
        assert not errors and not network
        browser.close()
    report = {
        "passed": True,
        "html_sha256": hashlib.sha256((output / "leaderboard_climb.html").read_bytes()).hexdigest(),
        "history_sha256": hashlib.sha256(
            (output / "leaderboard_history.json").read_bytes()
        ).hexdigest(),
        "rows": len(history["rows"]),
        "best_public": 0.6660,
        "javascript_errors": errors,
        "external_requests": network,
        "csv_json_downloads": "passed",
        "filter_search_and_view_controls": "passed",
        "mobile_horizontal_page_overflow": False,
    }
    (output / "browser_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
