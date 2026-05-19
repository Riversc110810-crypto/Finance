import asyncio
from playwright.async_api import async_playwright

PAGES = [
    ("Overview",         None),
    ("Strategy Signals", None),
    ("Backtest",         None),
    ("Monte Carlo",      None),
    ("Trade Log",        None),
    ("Position Sizer",   None),
    ("Settings",         None),
]

async def shot(page, label, path):
    # click sidebar radio for this page
    try:
        await page.get_by_text(label, exact=True).first.click()
        await page.wait_for_timeout(6000)
    except Exception:
        pass
    await page.screenshot(path=path, full_page=True)
    print(f"Saved {path}")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await ctx.new_page()

        await page.goto("http://localhost:8501", timeout=30000)
        await page.wait_for_timeout(10000)  # let streamlit fully load

        for label, _ in PAGES:
            safe = label.lower().replace(" ", "_")
            await shot(page, label, f"/home/user/Finance/screenshots/{safe}.png")

        await browser.close()

asyncio.run(main())
