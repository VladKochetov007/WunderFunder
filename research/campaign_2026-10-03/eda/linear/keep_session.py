# Holds a headless browser tab on the marimo server so a kernel session exists for pairing.
import time
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--disable-gpu"])
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    pg.goto("http://127.0.0.1:2732", wait_until="domcontentloaded")
    while True:
        time.sleep(30)
