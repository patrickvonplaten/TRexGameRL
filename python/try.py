#!/usr/bin/env python

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('disable-infobars')
options.add_argument('--headless')
options.binary_location = "/usr/bin/google-chrome"
#options.binary_location = "/etc/chromium"
driver = webdriver.Chrome(chrome_options = options, executable_path=r'/u/platen/chromedriver')
driver.get('http://google.com/')
print("Chrome Browser Invoked")
driver.quit()
