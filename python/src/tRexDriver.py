from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import os
import base64
import numpy as np
import cv2
import ipdb

HEIGHT = 150
WIDTH = 600

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
CHROME_EXECUTABLEPATH = CUR_PATH + '/../../chromedriver'
CHROME_PATH = '/usr/bin/google-chrome'
TREX_HTML_PATH = 'file://{}/../../javascript/index.html'.format(CUR_PATH)


class ChromeDriver(object):
    def __init__(self, display):
        chrome_options = ['disable-infobars']
        chrome_options.append('--headless')
        self.driver = self.configure_driver(chrome_options)

    def configure_driver(self, chromeOptions):
        chrome_options = Options()
        for option in chromeOptions:
            chrome_options.add_argument(option)
        chrome_options.binary_location = CHROME_PATH

        driver = webdriver.Chrome(executable_path=CHROME_EXECUTABLEPATH, chrome_options=chrome_options)
        driver.get(TREX_HTML_PATH)
        # https://stackoverflow.com/questions/40632204/selenium-webdriver-screenshot-in-python-has-the-wrong-resolution
        dx, dy = driver.execute_script("var w=window; return [w.outerWidth - w.innerWidth, w.outerHeight - w.innerHeight];")
        driver.set_window_size(WIDTH + dx, HEIGHT + dy)
        return driver

    def _get_raw_image(self):
        screenshot = self.driver.get_screenshot_as_base64()
        b64screenshot = screenshot.encode()
        return base64.b64decode(b64screenshot)

    def get_image_as(self, dtype):
        return cv2.imdecode(np.frombuffer(self._get_raw_image(), dtype), 0)

    def press_up(self):
        return self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        return self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def execute_script(self, script):
        return self.driver.execute_script(script)
