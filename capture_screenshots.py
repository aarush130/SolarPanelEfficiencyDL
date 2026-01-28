"""
Capture screenshots of the Streamlit application for the project report.
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

def capture_screenshots():
    """Capture screenshots of different tabs of the application."""
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--start-maximized")
    # chrome_options.add_argument("--headless")  # Uncomment for headless mode
    
    # Setup driver
    print("Setting up Chrome driver...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    screenshot_dir = r"C:\Users\aarushs\SolarPanelEfficiencyDL\screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    
    try:
        # Navigate to app
        print("Navigating to Streamlit app...")
        driver.get("http://localhost:8501")
        time.sleep(5)  # Wait for app to load
        
        # Screenshot 1: India Solar Map (first tab - default)
        print("Capturing India Solar Map...")
        time.sleep(3)
        driver.save_screenshot(os.path.join(screenshot_dir, "01_india_solar_map.png"))
        print("  Saved: 01_india_solar_map.png")
        
        # Scroll down to see more content
        driver.execute_script("window.scrollTo(0, 500)")
        time.sleep(2)
        driver.save_screenshot(os.path.join(screenshot_dir, "02_india_map_rankings.png"))
        print("  Saved: 02_india_map_rankings.png")
        
        # Scroll more
        driver.execute_script("window.scrollTo(0, 1000)")
        time.sleep(2)
        driver.save_screenshot(os.path.join(screenshot_dir, "03_state_comparison.png"))
        print("  Saved: 03_state_comparison.png")
        
        # Click on Prediction tab
        print("Capturing Prediction interface...")
        tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
        if len(tabs) > 1:
            tabs[1].click()
            time.sleep(3)
            driver.execute_script("window.scrollTo(0, 0)")
            driver.save_screenshot(os.path.join(screenshot_dir, "04_prediction_interface.png"))
            print("  Saved: 04_prediction_interface.png")
        
        # Click on Data Analysis tab
        print("Capturing Data Analysis...")
        tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
        if len(tabs) > 2:
            tabs[2].click()
            time.sleep(3)
            driver.execute_script("window.scrollTo(0, 0)")
            driver.save_screenshot(os.path.join(screenshot_dir, "05_data_analysis.png"))
            print("  Saved: 05_data_analysis.png")
            
            # Scroll for more visualizations
            driver.execute_script("window.scrollTo(0, 600)")
            time.sleep(2)
            driver.save_screenshot(os.path.join(screenshot_dir, "06_data_visualizations.png"))
            print("  Saved: 06_data_visualizations.png")
        
        # Click on Model Performance tab
        print("Capturing Model Performance...")
        tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
        if len(tabs) > 3:
            tabs[3].click()
            time.sleep(3)
            driver.execute_script("window.scrollTo(0, 0)")
            driver.save_screenshot(os.path.join(screenshot_dir, "07_model_performance.png"))
            print("  Saved: 07_model_performance.png")
        
        # Click on About tab
        print("Capturing About section...")
        tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
        if len(tabs) > 4:
            tabs[4].click()
            time.sleep(3)
            driver.execute_script("window.scrollTo(0, 0)")
            driver.save_screenshot(os.path.join(screenshot_dir, "08_about.png"))
            print("  Saved: 08_about.png")
        
        print(f"\nAll screenshots saved to: {screenshot_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    capture_screenshots()
