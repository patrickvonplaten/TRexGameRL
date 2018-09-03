#!/usr/bin/env python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from matplotlib import pyplot 
from tRexModel import TFRexModel
from imageio import imwrite
import os
import time
import base64
import io
import numpy as np
import sys
import cv2
import ipdb
import argparse

WINDOW_SIZE = '400,300'
CHROME_EXECUTABLEPATH = '/usr/bin/chromedriver' 
CHROME_PATH = '/usr/bin/google-chrome'
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
TREX_HTML_PATH = 'file://{}/../javascript/index.html'.format(CUR_PATH)
PATH_TO_IMAGE_FOLDER = os.path.join(CUR_PATH,'../imagesToCheck')

class ChromeDriver(object): 

    def __init__(self):
        chromeOptions = ['disable-infobars', '--window-size=%s' % WINDOW_SIZE]
#        chromeOptions.append('--headless')
        self.driver = self.configureDriver(chromeOptions)

    def configureDriver(self, chromeOptions):
        chrome_options = Options()
        for option in chromeOptions: 
            chrome_options.add_argument(option)
        chrome_options.binary_location = CHROME_PATH
        
        driver=webdriver.Chrome(executable_path=CHROME_EXECUTABLEPATH, chrome_options=chrome_options)        
        driver.get(TREX_HTML_PATH)
        return driver


class Action(object):
    def __init__(self, action_fn, reward, code=None):
        self.action = action_fn
        self.code = code
        self.reward = reward
    def __call__(self):
        if self.code:
            print(self.code)
        self.action()

class Game(object):
    def __init__(self, framesPerSample, frameSamplingRate):
        self.driver = ChromeDriver().driver
        self.screenshotShape = self.transfromBase64ToUint8(self.driver.get_screenshot_as_base64()).shape
        self.framesPerSample = framesPerSample
        self.frameSamplingRate = frameSamplingRate
        self.environmentDataShape = (self.framesPerSample,) + self.screenshotShape

    def showEnvironmentImage(self, envImage):
        pyplot.imshow(envImage)
        pyplot.show()

    def transfromBase64ToUint8(self, screenshot):
        screenAsBase64 = screenshot.encode()
        screenAsBytes = base64.b64decode(screenAsBase64)
        return cv2.imdecode(np.frombuffer(screenAsBytes, np.uint8), 0)

    def getEnvironmentState(self):
        environmentData = np.zeros(self.environmentDataShape, dtype=np.uint8)
        for i in range(self.framesPerSample):
            startTime = time.time()
            screenshot = self.driver.get_screenshot_as_base64()
            environmentData[i] = self.transfromBase64ToUint8(screenshot)
#            ipdb.set_trace()
            endTime = time.time() - startTime
            if(endTime < self.frameSamplingRate):
                time.sleep(self.frameSamplingRate - endTime)
        return environmentData

    def isCrashed(self):
        raise NotImplementedError()

    def isRunning(self):
        raise NotImplementedError()

    def getState(self, action):
        raise NotImplementedError()

    def restart(self):
        raise NotImplementedError()

    def end(self):
        raise NotImplementedError()


class TRexGame(Game):
    def __init__(self, framesPerSample=4, frameSamplingRate=0.1):
        super().__init__(framesPerSample, frameSamplingRate)
        jump = Action(self._pressUp, -5, "jump")
        duck = Action(self._pressDown, -3, "duck")
        run = Action(lambda: None, 1, "run")
        self.actions = [jump, duck, run]

    def _pressUp(self):
        return self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def _pressDown(self):
        return self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def restart(self):
        return self.driver.execute_script("Runner.instance_.restart()")

    def isCrashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")

    def isRunning(self):
        return self.driver.execute_script("return Runner.instance_.playing")

    def getScore(self):
        scoreArray = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(scoreArray) 
        print('Score' + str(score))
        return int(score)

    def end(self):
        self.driver.quit()

    def getReward(self, actionCode):
        if(self.isCrashed()):
            return -100
        else:
            action = self.actions[actionCode]
            return action.reward


class Agent(object):

    def __init__(self, game, model, mode, epochToCollectData):
        self.game = game
        self.model = model
        self.mode = mode 
        self.epochToCollectData = epochToCollectData
        self.actionCode = ['jump', 'run','duck']
        self.trainingData = None
        self.pathToImageFolder = PATH_TO_IMAGE_FOLDER
        if not os.path.isdir(self.pathToImageFolder):
            os.mkdir(self.pathToImageFolder)

    def takeAction(self, code):
        if(self.actionCode[code] == 'jump'):
            return self.game.actions[0]()
        elif(self.actionCode[code] == 'run'):
            return self.game.actions[1]()
        elif(self.actionCode[code] == 'duck'):
            return self.game.actions[2]()

    def execute(self): 
        if(self.mode == 'play'):
            self.play()
        if(self.mode == 'train'):
            self.train()

    def play(self):
        self.takeAction(0)
        while(self.game.isRunning()):
            self.processEnvironmentToAction()

    def processEnvironmentToAction(self):
        environment = self.game.getEnvironmentState()
        actionCode = self.model.getAction(environment)
        self.takeAction(actionCode)
#        self.game.getScore()
        return actionCode, environment

    def train(self):
        trainingData = self.collectTrainingData()

    def collectTrainingData(self):
        self.takeAction(0)
        self.trainingData = [[None, None, None]]
        oldEnvironment = None
        for i in range(self.epochToCollectData+2):
            print('iter start' + str(i), flush=True)
            actionCode, environment = self.processEnvironmentToAction()
            print('iter end' + str(i), flush=True)
            reward = self.game.getReward(actionCode)
            if(self.game.isCrashed()):
                self.game.restart()
            sample = []
            sample.append(environment)
            sample.append(actionCode)
            sample.append(reward)
            self.trainingData[i-1].append(environment)
            self.trainingData.append(sample)
        self.trainingData = self.trainingData[1:-1]
        return self.trainingData

    def end(self):
        return self.game.end()

    def saveEnvironmentScreenshots(self, numberOfEnvStatesToPrint=None):
        numberOfEnvStatesToPrint = numberOfEnvStatesToPrint if numberOfEnvStatesToPrint is not None else self.epochToCollectData
        for i in range(len(self.trainingData[:numberOfEnvStatesToPrint])):
            for j in range(4):
                imageToSave = self.trainingData[i][0][j]
#                ipdb.set_trace()
                image_name = 'env_{}_{}.jpg'.format(i, j+4)
                imwrite(os.path.join(self.pathToImageFolder, image_name), imageToSave)


if __name__ == "__main__":
    model = TFRexModel()
    game = TRexGame()
    agent = Agent(game = game, model = model,mode='train', epochToCollectData=20)
    agent.execute()
    agent.saveEnvironmentScreenshots()
    agent.end()
