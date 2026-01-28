"""
Capture screenshots from the live Streamlit app
"""

from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
import time
import os

def capture_screenshots():
    # Setup Edge options
    edge_options = Options()
    edge_options.add_argument('--window-size=1920,1080')
    edge_options.add_argument('--start-maximized')
    
    screenshot_dir = r'C:\Users\aarushs\SolarPanelEfficiencyDL\screenshots'
    os.makedirs(screenshot_dir, exist_ok=True)
    
    driver = None
    try:
        print('Starting Edge browser...')
        driver = webdriver.Edge(options=edge_options)
        
        print('Navigating to app...')
        driver.get('https://solar-panel-efficiency.streamlit.app/')
        time.sleep(12)  # Wait for app to load
        
        # Screenshot 1: India Solar Map
        print('Capturing India Solar Map...')
        driver.save_screenshot(os.path.join(screenshot_dir, '01_india_solar_map.png'))
        print('  Saved: 01_india_solar_map.png')
        
        # Scroll down for rankings
        driver.execute_script('window.scrollTo(0, 600)')
        time.sleep(2)
        driver.save_screenshot(os.path.join(screenshot_dir, '02_rankings_table.png'))
        print('  Saved: 02_rankings_table.png')
        
        # Scroll more for charts
        driver.execute_script('window.scrollTo(0, 1200)')
        time.sleep(2)
        driver.save_screenshot(os.path.join(screenshot_dir, '03_state_comparison.png'))
        print('  Saved: 03_state_comparison.png')
        
        # Click Prediction tab
        print('Capturing Prediction tab...')
        tabs = driver.find_elements(By.CSS_SELECTOR, 'button[data-baseweb="tab"]')
        if len(tabs) > 1:
            tabs[1].click()
            time.sleep(4)
            driver.execute_script('window.scrollTo(0, 0)')
            driver.save_screenshot(os.path.join(screenshot_dir, '04_prediction.png'))
            print('  Saved: 04_prediction.png')
        
        # Click Data Analysis tab
        print('Capturing Data Analysis tab...')
        tabs = driver.find_elements(By.CSS_SELECTOR, 'button[data-baseweb="tab"]')
        if len(tabs) > 2:
            tabs[2].click()
            time.sleep(4)
            driver.execute_script('window.scrollTo(0, 0)')
            driver.save_screenshot(os.path.join(screenshot_dir, '05_data_analysis.png'))
            print('  Saved: 05_data_analysis.png')
            
            # Scroll for more charts
            driver.execute_script('window.scrollTo(0, 800)')
            time.sleep(2)
            driver.save_screenshot(os.path.join(screenshot_dir, '05b_data_charts.png'))
            print('  Saved: 05b_data_charts.png')
        
        # Click Model Performance tab
        print('Capturing Model Performance tab...')
        tabs = driver.find_elements(By.CSS_SELECTOR, 'button[data-baseweb="tab"]')
        if len(tabs) > 3:
            tabs[3].click()
            time.sleep(4)
            driver.execute_script('window.scrollTo(0, 0)')
            driver.save_screenshot(os.path.join(screenshot_dir, '06_model_performance.png'))
            print('  Saved: 06_model_performance.png')
        
        print(f'\nAll screenshots saved to: {screenshot_dir}')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    capture_screenshots()
