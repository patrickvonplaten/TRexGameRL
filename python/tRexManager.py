from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from matplotlib import pyplot 
import time
import base64
import io 
import numpy as np 
import cv2
import tensorflow as tf

class ChromeDriver(object): 

    def __init__(self, chromeExecutablePath='/usr/bin/chromedriver', tRexHtmlPath='file:///home/patrick/TRexGameRL/javascript/index.html'):
        self.chromeExecutablePath = chromeExecutablePath
        self.tRexHtmlPath = tRexHtmlPath
        self.chromeOptions = ['disable-infobars']
        self.driver = self.configureDriver(self.chromeOptions)

    def configureDriver(self, chromeOptions):
        options = Options()
        for option in chromeOptions: 
            options.add_argument(option)
        
        driver=webdriver.Chrome(executable_path=self.chromeExecutablePath, chrome_options=options)        
        driver.set_window_position(x=-10,y=0)
        driver.set_window_size(200, 300)
        driver.get(self.tRexHtmlPath)
        return driver


class Game(object):
    def __init__(self):
        self.driver = ChromeDriver().driver

    def showEnvironmentImage(self, envImage):
        pyplot.imshow(envImage)
        pyplot.show()

    def getEnvironmentState(self):
        return self.getScore(), self.getEnvironmentImage()

    def getEnvironmentImage(self):
        screenAsBase64 = self.driver.get_screenshot_as_base64().encode()
        screenAsBytes = base64.b64decode(screenAsBase64)
        screenAsNumpyArray = cv2.imdecode(np.frombuffer(screenAsBytes, np.uint8), 0)
        screenAsNumpyArrayShortened = (screenAsNumpyArray[30:141]/128).astype(int)
        return screenAsNumpyArrayShortened

    def restartGame(self):
        self.driver.execute_script("Runner.instance_.restart()")

    def pressUp(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def pressDown(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def isCrashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")

    def isRunning(self):
        return self.driver.execute_script("return Runner.instance_.playing")

    def getScore(self):
        scoreArray = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(scoreArray) 
        return int(score)

    def end(self):
        self.driver.close()


class Agent(object):

    def __init__(self, model=None):
        self.game = Game() 
        self.model = model
        self.actions = [self.jump, self.run, self.duck]

    def jump(self):
        print('jump')
        self.game.pressUp()

    def duck(self):
        print('duck')
        self.game.pressDown()

    def run(self):
        print('run')
        time.sleep(.01)

    def takeAction(self, code):
        return self.actions[code].__call__()

    def play(self):
        self.jump()
        print("Start game")
        while(self.game.isRunning()):
            self.processEnvironmentToAction(timeStep=0.25)

    def processEnvironmentToAction(self, timeStep):
        startTime = time.time()
        score, environment = self.game.getEnvironmentState()
        self.takeAction(self.model.getAction(environment))
        passedTime = time.time() - startTime
        if(passedTime < timeStep):
            time.sleep(timeStep - passedTime)

    def train(self):
        pass

class Model(object):

    def __init__(self):
        self.weights = None 

    def getAction(self, environmentState):
        from random import randint
        return randint(0, 2)

if __name__ == "__main__":
    agent = Agent(model = Model())
    agent.play()
