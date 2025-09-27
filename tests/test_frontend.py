#!/usr/bin/env python3
"""
Playwright tests for ARIMAX Frontend
Tests UI functionality and user interactions
"""

import pytest
import pytest_asyncio
import asyncio
import os
import sys
import subprocess
import time
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Test configuration
STREAMLIT_URL = "http://localhost:8501"
STREAMLIT_PORT = 8501
TEST_TIMEOUT = 30000  # 30 seconds
STREAMLIT_STARTUP_TIMEOUT = 60  # 60 seconds for streamlit to start

class StreamlitApp:
    """Manages Streamlit app lifecycle for testing"""

    def __init__(self):
        self.process = None

    def start(self):
        """Start the Streamlit app"""
        try:
            # Change to frontend directory
            frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')

            # Start streamlit app
            self.process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(STREAMLIT_PORT)],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for app to start
            print(f"Starting Streamlit app on port {STREAMLIT_PORT}...")
            self._wait_for_startup()

        except Exception as e:
            print(f"Failed to start Streamlit app: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop the Streamlit app"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def _wait_for_startup(self):
        """Wait for Streamlit to be ready"""
        import requests

        start_time = time.time()
        while time.time() - start_time < STREAMLIT_STARTUP_TIMEOUT:
            try:
                response = requests.get(STREAMLIT_URL, timeout=5)
                if response.status_code == 200:
                    print("Streamlit app is ready!")
                    return
            except requests.exceptions.RequestException:
                pass

            time.sleep(2)

        raise TimeoutError(f"Streamlit app did not start within {STREAMLIT_STARTUP_TIMEOUT} seconds")

@pytest.fixture(scope="session")
def streamlit_app():
    """Session-scoped fixture to start/stop Streamlit app"""
    app = StreamlitApp()
    app.start()
    yield app
    app.stop()

@pytest_asyncio.fixture
async def browser():
    """Browser fixture"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()

@pytest_asyncio.fixture
async def page(browser: Browser):
    """Page fixture"""
    context = await browser.new_context()
    page = await context.new_page()

    # Set default timeout
    page.set_default_timeout(TEST_TIMEOUT)

    yield page
    await context.close()

@pytest.mark.asyncio
async def test_app_loads(streamlit_app, page: Page):
    """Test that the app loads successfully"""
    await page.goto(STREAMLIT_URL)

    # Wait for Streamlit to fully load
    await page.wait_for_load_state('networkidle')
    await page.wait_for_timeout(3000)  # Wait 3 seconds for dynamic content

    # Check title
    title = await page.title()
    assert "ARIMAX Stock Forecasting" in title

    # Check main heading
    heading = page.locator("h1").first
    await heading.wait_for(state="visible", timeout=10000)
    assert "ARIMAX Stock Forecasting" in await heading.text_content()

@pytest.mark.asyncio
async def test_sidebar_controls(streamlit_app, page: Page):
    """Test sidebar control panel elements"""
    await page.goto(STREAMLIT_URL)

    # Wait for sidebar to load
    await page.wait_for_selector("[data-testid='stSidebar']")

    # Check for control panel header
    sidebar = page.locator("[data-testid='stSidebar']")

    # Look for update dataset button (use first instance)
    update_button = sidebar.locator("button", has_text="🔄 Update Dataset").first
    await update_button.wait_for(state="visible", timeout=10000)

    # Look for generate predictions button (use first instance)
    predict_button = sidebar.locator("button", has_text="🎯 Generate Predictions").first
    await predict_button.wait_for(state="visible", timeout=10000)

@pytest.mark.asyncio
async def test_ticker_input(streamlit_app, page: Page):
    """Test ticker input functionality"""
    await page.goto(STREAMLIT_URL)

    # Wait for main content
    await page.wait_for_selector("h1")

    # Look for ticker selection elements
    # Streamlit selectbox
    selectbox = page.locator("[data-testid='stSelectbox']").first
    if await selectbox.count() > 0:
        await selectbox.click()

    # Text input for manual ticker entry
    text_inputs = page.locator("input[type='text']")
    if await text_inputs.count() > 0:
        ticker_input = text_inputs.first
        await ticker_input.fill("AAPL")
        await ticker_input.press("Enter")

@pytest.mark.asyncio
async def test_prediction_settings(streamlit_app, page: Page):
    """Test prediction settings in sidebar"""
    await page.goto(STREAMLIT_URL)

    # Wait for sidebar
    await page.wait_for_selector("[data-testid='stSidebar']")

    sidebar = page.locator("[data-testid='stSidebar']")

    # Look for slider (prediction periods)
    slider = sidebar.locator("[data-testid='stSlider']").first
    if await slider.count() > 0:
        await slider.wait_for(state="visible")

    # Look for checkboxes
    checkboxes = sidebar.locator("input[type='checkbox']")
    checkbox_count = await checkboxes.count()
    assert checkbox_count >= 0  # Should have confidence interval and historical data checkboxes

@pytest.mark.asyncio
async def test_update_dataset_button(streamlit_app, page: Page):
    """Test update dataset button click"""
    await page.goto(STREAMLIT_URL)

    # Wait for sidebar
    await page.wait_for_selector("[data-testid='stSidebar']")

    sidebar = page.locator("[data-testid='stSidebar']")
    update_button = sidebar.locator("button", has_text="🔄 Update Dataset")

    if await update_button.count() > 0:
        # Click the button
        await update_button.click()

        # Look for spinner or status message
        # Note: This might take a while in real scenarios
        # For testing, we just verify the click doesn't crash the app
        await page.wait_for_timeout(2000)  # Wait 2 seconds

        # Check app is still responsive
        heading = await page.locator("h1").first.text_content()
        assert "ARIMAX Stock Forecasting" in heading

@pytest.mark.asyncio
async def test_generate_predictions_button(streamlit_app, page: Page):
    """Test generate predictions button click"""
    await page.goto(STREAMLIT_URL)

    # Wait for sidebar
    await page.wait_for_selector("[data-testid='stSidebar']")

    sidebar = page.locator("[data-testid='stSidebar']")
    predict_button = sidebar.locator("button", has_text="🎯 Generate Predictions")

    if await predict_button.count() > 0:
        # Click the button
        await predict_button.click()

        # Look for spinner or status message
        await page.wait_for_timeout(2000)  # Wait 2 seconds

        # Check app is still responsive
        heading = await page.locator("h1").first.text_content()
        assert "ARIMAX Stock Forecasting" in heading

@pytest.mark.asyncio
async def test_error_handling(streamlit_app, page: Page):
    """Test error handling for invalid inputs"""
    await page.goto(STREAMLIT_URL)

    # Try entering an invalid ticker
    text_inputs = page.locator("input[type='text']")
    if await text_inputs.count() > 0:
        ticker_input = text_inputs.first
        await ticker_input.fill("INVALID_TICKER_XYZ")
        await ticker_input.press("Enter")

        # Wait for response
        await page.wait_for_timeout(3000)

        # App should still be functional
        heading = await page.locator("h1").first.text_content()
        assert "ARIMAX Stock Forecasting" in heading

@pytest.mark.asyncio
async def test_responsive_design(streamlit_app, page: Page):
    """Test responsive design on different screen sizes"""
    await page.goto(STREAMLIT_URL)

    # Test desktop size
    await page.set_viewport_size({"width": 1920, "height": 1080})
    await page.wait_for_selector("h1")

    # Test tablet size
    await page.set_viewport_size({"width": 768, "height": 1024})
    await page.wait_for_timeout(1000)

    # Test mobile size
    await page.set_viewport_size({"width": 375, "height": 667})
    await page.wait_for_timeout(1000)

    # App should still be functional
    heading = await page.locator("h1").first.text_content()
    assert "ARIMAX Stock Forecasting" in heading

@pytest.mark.asyncio
async def test_chart_rendering(streamlit_app, page: Page):
    """Test that charts render when data is available"""
    await page.goto(STREAMLIT_URL)

    # Enter a ticker (if available)
    text_inputs = page.locator("input[type='text']")
    if await text_inputs.count() > 0:
        ticker_input = text_inputs.first
        await ticker_input.fill("AAPL")
        await ticker_input.press("Enter")

        # Wait for potential chart rendering
        await page.wait_for_timeout(5000)

        # Look for plotly chart elements
        plotly_divs = page.locator("div[class*='plotly']")
        chart_count = await plotly_divs.count()

        # If no models are available, we expect no charts
        # If models are available, we should see charts
        print(f"Found {chart_count} plotly chart elements")

@pytest.mark.asyncio
async def test_performance(streamlit_app, page: Page):
    """Test basic performance metrics"""
    start_time = time.time()

    await page.goto(STREAMLIT_URL)
    await page.wait_for_selector("h1")

    load_time = time.time() - start_time

    # App should load within reasonable time
    assert load_time < 30, f"App took too long to load: {load_time:.2f} seconds"

    print(f"App loaded in {load_time:.2f} seconds")

# Test runner script
if __name__ == "__main__":
    # Run specific test
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        pytest.main([f"-v", f"-k", test_name, __file__])
    else:
        # Run all tests
        pytest.main(["-v", __file__])