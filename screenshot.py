import asyncio
from playwright.async_api import async_playwright

PAGES = [
    ("Overview",         6000),
    ("Strategy Signals", 18000),
    ("L/S Pairs",        25000),
    ("Backtest",         8000),
    ("Monte Carlo",      8000),
    ("Trade Log",        4000),
    ("Position Sizer",   5000),
    ("Settings",         4000),
]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await ctx.new_page()

        await page.goto("http://localhost:8501", timeout=30000)
        await page.wait_for_timeout(12000)

        for label, wait_ms in PAGES:
            safe = label.lower().replace("/", "").replace(" ", "_").strip("_")
            path = f"/home/user/Finance/screenshots/{safe}.png"
            try:
                await page.get_by_text(label, exact=True).first.click()
                await page.wait_for_timeout(wait_ms)
            except Exception:
                pass
            await page.screenshot(path=path, full_page=True)
            print(f"Saved {path}")

        await browser.close()

asyncio.run(main())
