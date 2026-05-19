import asyncio
from playwright.async_api import async_playwright

LOGIN_USER = "trader"
LOGIN_PASS = "trading123"

PAGES = [
    ("Overview",          6000),
    ("Company Search",    5000),
    ("Market Research",   8000),
    ("AI Assistant",      4000),
    ("Strategy Signals",  18000),
    ("L/S Pairs",         25000),
    ("Strategy Overview", 12000),
    ("Backtesting",       10000),
    ("Monte Carlo",       8000),
    ("Trade Log",         4000),
    ("Position Sizer",    5000),
    ("Settings",          4000),
]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await ctx.new_page()

        await page.goto("http://localhost:8501", timeout=30000)
        await page.wait_for_timeout(10000)

        # Screenshot login page before logging in
        await page.screenshot(path="/home/user/Finance/screenshots/login.png", full_page=True)
        print("Saved login.png")

        # Log in using label-based selectors
        try:
            await page.locator("input[aria-label='Username']").fill(LOGIN_USER, timeout=10000)
            await page.locator("input[aria-label='Password']").fill(LOGIN_PASS, timeout=10000)
            await page.locator("button:has-text('Sign In')").click(timeout=10000)
            await page.wait_for_timeout(5000)
            print("Logged in")
        except Exception as e:
            print(f"Login attempt 1 failed: {e}")
            # Fallback: try nth inputs
            try:
                inputs = page.locator("input")
                await inputs.nth(0).fill(LOGIN_USER)
                await inputs.nth(1).fill(LOGIN_PASS)
                await page.locator("button").filter(has_text="Sign").click()
                await page.wait_for_timeout(5000)
                print("Logged in (fallback)")
            except Exception as e2:
                print(f"Login fallback also failed: {e2}")

        for label, wait_ms in PAGES:
            safe = label.lower().replace("/", "").replace(" ", "_").strip("_")
            path = f"/home/user/Finance/screenshots/{safe}.png"
            try:
                await page.get_by_text(label, exact=True).first.click(timeout=10000)
                await page.wait_for_timeout(wait_ms)
            except Exception as e:
                print(f"  nav error for {label}: {e}")
            await page.screenshot(path=path, full_page=True)
            print(f"Saved {path}")

        await browser.close()

asyncio.run(main())
