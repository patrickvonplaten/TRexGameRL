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
import ipdb

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
        self.screenshotHorizontalDim = self.transfromBase64ToUint8(self.driver.get_screenshot_as_base64()).shape[1]
        self.screenshotVerticalDimCoord = (30,141)
        self.screenshotVerticalDim = self.screenshotVerticalDimCoord[1] - self.screenshotVerticalDimCoord[0]
        self.discretizationLevel = 128
        self.framesPerSample = 4
        self.frameSamplingRate = 0.25
        self.tempEnvironmentData = np.zeros((self.framesPerSample, self.screenshotVerticalDim, self.screenshotHorizontalDim), dtype=np.int8)
        self.dataShape = (2*self.tempEnvironmentData.shape[0],) + self.tempEnvironmentData.shape[1:]

    def showEnvironmentImage(self, envImage):
        pyplot.imshow(envImage)
        pyplot.show()

    def transfromBase64ToUint8(self, screenshot):
        screenAsBase64 = screenshot.encode()
        screenAsBytes = base64.b64decode(screenAsBase64)
        return cv2.imdecode(np.frombuffer(screenAsBytes, np.uint8), 0)
    
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
            print('EnvState' + str(i))
        print('Finish')
        self.tempEnvironmentData = environmentData[self.framesPerSample:]
        return environmentData

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

    def __init__(self, model, mode, epochToCollectData):
        self.game = Game() 
        self.model = model
        self.actions = [self.jump, self.run, self.duck]
        self.mode = mode 
        self.epochToCollectData = epochToCollectData
        self.timeStep = 0.25

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
        return self.actions[code].__call__()

    def run(self): 
        if(self.mode == 'play'):
            self.play()
        if(self.mode == 'train'):
            self.train()

    def play(self):
        self.jump()
        print("Start game")
        while(self.game.isRunning()):
            self.processEnvironmentToAction()

    def processEnvironmentToAction(self):
        environment = self.game.getEnvironmentState()
        actionCode = self.model.getAction(environment)
        self.takeAction(actionCode)
        print('Action')
        return actionCode, environment

    def train(self):
        trainingData = self.collectTrainingData()

    def collectTrainingData(self):
        self.jump()
        trainingData = [[None, None, None]]
        oldEnvironment = None
        for i in range(self.epochToCollectData+1):
            actionCode, environment = self.processEnvironmentToAction()
            reward = self.getReward(actionCode)
            if(self.game.isCrashed()):
                self.game.restartGame()
            sample = []
            sample.append(environment)
            sample.append(actionCode)
            sample.append(reward)
            sample.append(None)
            trainingData[i-1].append(environment)
            trainingData.append(sample)
            print(i)
            if(i==2):
                ipdb.set_trace()
        trainingData = trainingData[1:-1]
        return trainingData
            
    def getReward(self, actionCode):
        if(self.game.isCrashed()):
            return -100
        elif(self.actions[actionCode].__name__ == 'jump'):
            return -5
        elif(self.actions[actionCode].__name__ == 'duck'):
            return -3 
        else: 
            return 1

class Model(object):

    def __init__(self):
        self.weights = None 

    def getAction(self, environmentState):
        from random import randint
        return randint(0, 2)

if __name__ == "__main__":
    agent = Agent(model = Model(),mode='train', epochToCollectData=40)
    agent.run()
