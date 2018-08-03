from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import sys
import base64
import os
from PIL import Image
import requests
from io import BytesIO
import time

MATERIAL_DIR = "material" + os.sep + sys.argv[1] + os.sep
browser = webdriver.Chrome()

def scroll_to_bottom(driver):
    SCROLL_PAUSE_TIME = 0.5

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    return driver

try:
    if not os.path.exists(MATERIAL_DIR):
        os.mkdir(MATERIAL_DIR)

    browser.get('https://www.google.com.tw/imghp?hl=zh-TW')


    wait = WebDriverWait(browser, 10)
    wait.until(EC.presence_of_element_located((By.ID, 'lst-ib')))
 


    input = browser.find_element_by_id('lst-ib')
    input.send_keys(sys.argv[1])
    input.send_keys(Keys.ENTER)

    browser = scroll_to_bottom(browser)

    images = browser.find_elements_by_class_name('rg_ic')


    
    count_image = 0

    for image in images:
        count_image += 1

        src = image.get_attribute('src')
        info = src.split(",")[0]
        if "png" in info:
            image_path = MATERIAL_DIR + str(count_image) + ".png"
        elif "jpeg" in info:
            image_path = MATERIAL_DIR + str(count_image) + ".jpg"
        else:
            image_path = MATERIAL_DIR + str(count_image) + ".png"

        if src[0:4] == "http":
            response = requests.get(src)
            img = Image.open(BytesIO(response.content))
            img.save(image_path)
        else:
            img = src.split(",")[1]
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(img))
        time.sleep(1)


    #result = browser.execute_script("return document.readyState")
    #print(result)
    #print(browser.current_url)
    #print(browser.get_cookies())
    #print(browser.page_source)
except:
    pass
