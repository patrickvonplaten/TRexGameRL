from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import os
import base64
import numpy as np
import cv2
import ipdb  # noqa: F401

HEIGHT = 150
WIDTH = 600
CHROME_EXECUTABLEPATH = '/usr/bin/chromedriver'
CHROME_PATH = '/usr/bin/google-chrome'
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
TREX_HTML_PATH = 'file://{}/../../javascript/index.html'.format(CUR_PATH)


class ChromeDriver(object):
    def __init__(self, display):
        options = self.set_options(display)
        self.driver = self.configure_driver(options)

    def set_options(self, display):
        options = ['disable-infobars']
        if not display:
            options.append('--headless')
        return options

    def configure_driver(self, options):
        chrome_options = Options()
        for option in options:
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
