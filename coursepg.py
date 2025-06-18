from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import json

driver = webdriver.Firefox()
driver.get("")#enter url here <<<
time.sleep(5)

# Expand all dropdowns
for toggle in driver.find_elements(By.CLASS_NAME, "toggle"):
    try:
        toggle.click()
        time.sleep(0.5)
    except:
        continue

time.sleep(2)

# Final data list
scraped_data = []

# Use index loop to re-fetch links every time
for i in range(100):
    try:
        # Re-fetch sidebar links to avoid stale reference
        sidebar_links = driver.find_elements(By.CSS_SELECTOR, ".sidebar-nav a")

        if i >= len(sidebar_links):
            break

        link = sidebar_links[i]
        title = link.text.strip()
        href = link.get_attribute("href")

        # Click via JS to avoid visibility/staleness issues
        driver.execute_script("arguments[0].click();", link)
        time.sleep(1.5)

        # Extract content
        content_div = driver.find_element(By.CLASS_NAME, "content")
        content_html = content_div.get_attribute("innerHTML")
        content_text = content_div.text.strip()

        scraped_data.append({
            "title": title,
            "url": href,
            "text": content_text,
            "html": content_html
        })

    except Exception as e:
        print(f"[{i}] Error: {e}")
        continue
with open("tds_course_material.json", "w", encoding="utf-8") as f:
    json.dump(scraped_data, f, ensure_ascii=False, indent=2)

driver.quit()
