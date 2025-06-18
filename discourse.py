from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import json
def entry(urlin):
    global drivr
    USERNAME = "" #input username
    PASSWORD = "" #input password for the website
    drivr = webdriver.Firefox()
    drivr.get(urlin)
    time.sleep(2)
    # change the content until the next comment line depending on the website's layout and text
    login_button = drivr.find_element(By.XPATH, '//button[normalize-space()="Log In"]')
    login_button.click()
    time.sleep(2)
    username_input = drivr.find_element(By.ID, "login-account-name")
    password_input = drivr.find_element(By.ID, "login-account-password")

    username_input.send_keys(USERNAME)
    password_input.send_keys(PASSWORD)

    login_submit_button = drivr.find_element(By.XPATH, '//button[normalize-space()="Log In"]')
    login_submit_button.click()
    time.sleep(5)
entry("https://exmaplewebsite.com") # enter what website to scrape
posts = drivr.find_elements(By.CSS_SELECTOR, 'div.contents.ember-view') # check with ctrl+shift+c for exact elements.
from datetime import datetime
import re
time.sleep(5)
import time

SCROLL_PAUSE_TIME = 2
MAX_SCROLLS = 6 # max number of scrolls before stopping
scrolls = 0

last_height = drivr.execute_script("return document.body.scrollHeight")

while scrolls < MAX_SCROLLS:
    drivr.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)
    new_height = drivr.execute_script("return document.body.scrollHeight")

    if new_height == last_height:
        print("Reached the bottom or no new content loaded.")
        break

    last_height = new_height
    scrolls += 1

if scrolls == MAX_SCROLLS:
    print("Stopped scrolling after max scroll limit.")

td_elements = drivr.find_elements(By.CSS_SELECTOR, 'td.topic-list-data.age')


posts = []

for td in td_elements:
    title_attr = td.get_attribute('title')
    if title_attr and "Created:" in title_attr:
        # Extract created date substring using regex
        match = re.search(r'Created:\s*(.+?)(?:\s+Latest:|$)', title_attr)

        if match:
            created_str = match.group(1)
            post_date = datetime.strptime(created_str, '%b %d, %Y %I:%M %p')
            
            parent_row = td.find_element(By.XPATH, './ancestor::tr')
            title_element = parent_row.find_element(By.CSS_SELECTOR, 'a.title')
            title = title_element.text
            url = title_element.get_attribute('href')
            
            posts.append({
                'title': title,
                'url': url,
                'date': post_date,
            })

start_date = datetime(2025, 1, 1) #change start time
end_date = datetime(2025, 4, 14) #change end time

filtered_posts = [p for p in posts if start_date <= p['date'] <= end_date]
#
##########################################################################################################################################################################################################
#
all_posts_data=[]
for post in filtered_posts:
    # print(post['date'], post['title'], post['url'])
    url=post['url']
    print("Opening:", url)
    drivr.get(url)
    time.sleep(3)
    cooked_divs = drivr.find_elements(By.CLASS_NAME, "cooked")

    question_text = cooked_divs[0].text if cooked_divs else "No content"
    replies = [div.text for div in cooked_divs[1:]] 
    print("Question:", question_text)
    print("Replies:", replies)
    post_data = {
        "title": post['title'],
        "url": url,
        "date": post['date'].isoformat(),
        "question": "",
        "replies": []
    }

    post_blocks = drivr.find_elements(By.CSS_SELECTOR, 'div.topic-post')
    for block in post_blocks:
        try:
            post_number = block.get_attribute("data-post-number")
            content_div = block.find_element(By.CSS_SELECTOR, 'div.cooked')
            content = content_div.text.strip()
            if post_number == "1":
                post_data["question"]=content
            else:
                post_data["replies"].append(content)

        except Exception as e:
            print(f"❌ Error processing post {post_number}: {e}")
        time.sleep(2)
    all_posts_data.append(post_data)
drivr.quit()
with open("forum_post_data.json", "w", encoding="utf-8") as f:
    json.dump(all_posts_data, f, indent=2, ensure_ascii=False)
