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
import tensorflow as tf
import ipdb

WINDOW_SIZE = '400,300'
CHROME_EXECUTABLEPATH = '/usr/bin/chromedriver' 
CHROME_PATH = '/usr/bin/google-chrome'
TREX_HTML_PATH = 'file:///home/patrick/TRexGameRL/javascript/index.html'
PATH_TO_IMAGE_FOLDER = '/home/patrick/TRexGameRL/imagesToCheck'

class ChromeDriver(object): 

    def __init__(self):
        chromeOptions = ['disable-infobars', '--window-size=%s' % WINDOW_SIZE]
        chromeOptions.append('--headless')
        self.driver = self.configureDriver(chromeOptions)

    def configureDriver(self, chromeOptions):
        chrome_options = Options()
        for option in chromeOptions: 
            chrome_options.add_argument(option)
        chrome_options.binary_location = CHROME_PATH
        
        driver=webdriver.Chrome(executable_path=CHROME_EXECUTABLEPATH, chrome_options=chrome_options)        
        driver.get(TREX_HTML_PATH)
        return driver

class Game(object):
    def __init__(self):
        self.driver = ChromeDriver().driver
        self.screenshotHorizontalDim = self.transfromBase64ToUint8(self.driver.get_screenshot_as_base64()).shape[1]
        self.screenshotVerticalDimCoord = (30,141)
        self.screenshotVerticalDim = self.screenshotVerticalDimCoord[1] - self.screenshotVerticalDimCoord[0]
        self.discretizationLevel = 128
        self.framesPerSample = 4
        self.frameSamplingRate = 0.1
        self.tempEnvironmentData = np.zeros((self.framesPerSample, self.screenshotVerticalDim, self.screenshotHorizontalDim), dtype=np.int8)
        self.dataShape = (2*self.tempEnvironmentData.shape[0],) + self.tempEnvironmentData.shape[1:]

    def showEnvironmentImage(self, envImage):
        pyplot.imshow(envImage)
        pyplot.show()

    def transfromBase64ToUint8(self, screenshot):
        screenAsBase64 = screenshot.encode()
        screenAsBytes = base64.b64decode(screenAsBase64)
        return cv2.imdecode(np.frombuffer(screenAsBytes, np.uint8), 0).astype(np.uint8)
    
    def getEnvironmentState(self):
        environmentData = np.zeros(self.dataShape, dtype=np.int8)
        environmentData[:self.framesPerSample] = self.tempEnvironmentData
        for i in range(self.framesPerSample):
            startTime = time.time()
            screenshot = self.driver.get_screenshot_as_base64()
            screenshotAsUint8 = self.transfromBase64ToUint8(screenshot)
            environmentData[i+self.framesPerSample] = (screenshotAsUint8[self.screenshotVerticalDimCoord[0]:self.screenshotVerticalDimCoord[1]]/self.discretizationLevel).astype(int)
            endTime = time.time() - startTime
            if(endTime < self.frameSamplingRate):
                time.sleep(self.frameSamplingRate - endTime)
        self.tempEnvironmentData = environmentData[self.framesPerSample:]
        return environmentData

    def restartGame(self):
        return self.driver.execute_script("Runner.instance_.restart()")

    def pressUp(self):
        return self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def pressDown(self):
        return self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

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


class Agent(object):

    def __init__(self, model, mode, epochToCollectData):
        self.game = Game() 
        self.model = model
        self.mode = mode 
        self.epochToCollectData = epochToCollectData
        self.actionCode = ['jump', 'run','duck']
        self.trainingData = None
        self.pathToImageFolder = PATH_TO_IMAGE_FOLDER

    def jump(self):
        print('jump')
        self.game.pressUp()

    def duck(self):
        print('duck')
        self.game.pressDown()

    def run(self):
        print('run')
        pass

    def takeAction(self, code):
        if(self.actionCode[code] == 'jump'):
            return self.jump()
        elif(self.actionCode[code] == 'run'):
            return self.run()
        elif(self.actionCode[code] == 'duck'):
            return self.duck()

    def execute(self): 
        if(self.mode == 'play'):
            self.play()
        if(self.mode == 'train'):
            self.train()

    def play(self):
        self.jump()
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
        self.jump()
        self.trainingData = [[None, None, None]]
        oldEnvironment = None
        for i in range(self.epochToCollectData+2):
            print('iter start' + str(i), flush=True)
            actionCode, environment = self.processEnvironmentToAction()
            print('iter end' + str(i), flush=True)
            reward = self.getReward(actionCode)
            if(self.game.isCrashed()):
                self.game.restartGame()
            sample = []
            sample.append(environment)
            sample.append(actionCode)
            sample.append(reward)
            self.trainingData[i-1].append(environment)
            self.trainingData.append(sample)
        self.trainingData = self.trainingData[1:-1]
        return self.trainingData

    def saveEnvironmentScreenshots(self, numberOfEnvStatesToPrint=None):
        numberOfEnvStatesToPrint = numberOfEnvStatesToPrint if numberOfEnvStatesToPrint is not None else self.epochToCollectData
        for i in range(len(self.trainingData[:numberOfEnvStatesToPrint])):
            for j in range(4):
                imageToSave = self.trainingData[i][0][j+4]
#                ipdb.set_trace()
                imwrite(os.path.join(self.pathToImageFolder,'env_' + str(i) + '_' + str(j+4) + '.jpg'), imageToSave)

    def getReward(self, code):
        if(self.game.isCrashed()):
            return -100
        elif(self.actionCode[code] == 'jump'): 
            return -5
        elif(self.actionCode[code] == 'duck'): 
            return -3 
        elif(self.actionCode[code] == 'run'): 
            return 1

if __name__ == "__main__":
    agent = Agent(model = TFRexModel(),mode='train', epochToCollectData=20)
    agent.execute()
    agent.saveEnvironmentScreenshots()
    agent.game.end()
