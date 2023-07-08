import PIL as Image
import requests
import urllib3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def testing_links():
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("start-maximized")
        options.add_argument('disable-infobars')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        driver.get('http://127.0.0.1:5000/genre')
        links = driver.find_elements(By.CSS_SELECTOR, 'a')
        total=len(links)

        print("Testing links...")
        start = time.time()
        working_links = 0
        bad_links = 0
        bad_links_list = []
        for link in links:
            try:
             if link.get_attribute('href') == None:
                 total -=1
             r = requests.head((link.get_attribute('href')))
            except:
             continue
            if r.status_code != 400:
                working_links += 1
            else:
                bad_links += 1
                bad_links_list.append((link.get_attribute('href'),
                                    r.status_code))
        context = {"working_links": working_links,
                "bad_links_list": bad_links_list, "bad_links": bad_links,
                "links_len": total, "time_links": round((time.time() - start), 3)}
        # print(context)
        return context
    except:
        print("bad link")


def testing_imgs():
    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    options.add_argument('disable-infobars')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get('http://127.0.0.1:5000/genre')
    links = driver.find_elements(By.CSS_SELECTOR, 'img')

    print("Testing Images...")
    start = time.time()

    working_links = 0
    bad_links = 0
    bad_links_list = []
    for link in links:
        r = requests.head(link.get_attribute('src'))
        if r.status_code != 400:
            working_links += 1
        else:
            bad_links += 1
            bad_links_list.append((link.get_attribute('href'),
                                   r.status_code))
    context = {"working_links": working_links,
               "bad_links_list": bad_links_list, "bad_links": bad_links,
               "links_len": len(links), "time_links": round((time.time() - start), 3)}
    return context


print(str(testing_links()))
print(str(testing_imgs()))