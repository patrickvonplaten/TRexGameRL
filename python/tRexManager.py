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
from argparse import ArgumentParser

WINDOW_SIZE = '400,300'
CHROME_EXECUTABLEPATH = '/usr/bin/chromedriver' 
CHROME_PATH = '/usr/bin/google-chrome'
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
TREX_HTML_PATH = 'file://{}/../javascript/index.html'.format(CUR_PATH)
PATH_TO_IMAGE_FOLDER = os.path.join(CUR_PATH,'../imagesToCheck')


#TODO naming convention python!


class ChromeDriver(object): 

    def __init__(self, display):
        chrome_options = ['disable-infobars', '--window-size=%s' % WINDOW_SIZE]
        # TODO display changes data shape? -> let's write TODO in waffle https://waffle.io/patrickvonplaten/TRexGameRL
        if not display:
            chrome_options.append('--headless')
        self.driver = self.configure_driver(chrome_options)
        self.data_shape = self.get_image_as(np.uint8).shape
        print(self.data_shape)

    def configure_driver(self, chromeOptions):
        chrome_options = Options()
        for option in chromeOptions: 
            chrome_options.add_argument(option)
        chrome_options.binary_location = CHROME_PATH
        
        driver = webdriver.Chrome(executable_path=CHROME_EXECUTABLEPATH, chrome_options=chrome_options)
        driver.get(TREX_HTML_PATH)
        return driver

    def _get_raw_image(self):
        screenshot = self.driver.get_screenshot_as_base64()
        screenAsBase64 = screenshot.encode()
        screenAsBytes = base64.b64decode(screenAsBase64)
        return screenAsBytes

    def get_image_as(self, dtype):
        return cv2.imdecode(np.frombuffer(self._get_raw_image(), dtype), 0)

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
    def __init__(self, framesPerSample, frameSamplingRate, driver):
        self.driver = driver
        self.framesPerSample = framesPerSample
        self.frameSamplingRate = frameSamplingRate
        self.timestamp = 0

    def showEnvironmentImage(self, envImage):
        pyplot.imshow(envImage)
        pyplot.show()

    def getEnvironmentState(self):
        """Deprecate
        Frames per sample should also be preprocessing.
        """
        pass

    def isCrashed(self):
        raise NotImplementedError()

    def isRunning(self):
        raise NotImplementedError()

    def get_state(self, action):
        raise NotImplementedError()

    def restart(self):
        self.timestamp = 0
        return self._restart()

    def end(self):
        raise NotImplementedError()

    def do_action(self, action_code):
        self.timestamp += 1
        return self._do_action(action_code)

class TRexGame(Game):
    def __init__(self, framesPerSample=4, frameSamplingRate=0.1, display=False):
        chromeDriver = ChromeDriver(display)
        super().__init__(framesPerSample, frameSamplingRate, driver)
        jump = Action(self._pressUp, -5, "jump")
        duck = Action(self._pressDown, -3, "duck")
        run = Action(lambda: None, 1, "run")
        self.actions = [jump, duck, run]

    def _pressUp(self):
        return self.chromeDriver.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def _pressDown(self):
        return self.chromeDriver.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def _restart(self):
        return self.chromeDriver.driver.execute_script("Runner.instance_.restart()")

    def isCrashed(self):
        return self.chromeDriver.driver.execute_script("return Runner.instance_.crashed")

    def isRunning(self):
        return self.chromeDriver.driver.execute_script("return Runner.instance_.playing")

    def getScore(self):
        scoreArray = self.chromeDriver.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(scoreArray) 
        print('Score' + str(score))
        return int(score)

    def end(self):
        self.chromeDriver.driver.quit()

    def _do_action(self, action_code, time_to_execute_action):
        start_time = time.time()
        self.actions[action_code]()
        time_needed_to_execute_action = time.time() - start_time
        if(time_needed_to_execute_action > 0):
            time.sleep(time_to_execute_action - time_needed_to_execute_action)

    def get_state(self, actionCode):
        crashed = self.isCrashed()
        if crashed:
            reward = -100
        else:
            action = self.actions[actionCode]
            reward = action.reward

        image = self.driver.get_image_as(np.uint8)
        return image, reward, crashed, self.timestamp

class State(object):
    def __init__(self):
        self.image = image
        self.reward = reward
        self.crashed = crashed
        self.timestamp = timestap
        self.state_data_as_list = [self.image, self.reward, self.crashed]

    def get_state_data_as_list(self):
        return self.state_data_as_list

    def get_image(self):
        return self.image

    def is_crashed(self):
        return self.crashed

    def get_time_stamp(self):
        return self.timestamp

class Agent(object):
    def __init__(self, game, model, mode, epochToCollectData):
        self.game = game
        self.model = model
        self.mode = mode 
        self.epochToCollectData = epochToCollectData
        self.trainingData = None
        self.pathToImageFolder = PATH_TO_IMAGE_FOLDER
        if not os.path.isdir(self.pathToImageFolder):
            os.mkdir(self.pathToImageFolder)

    def execute(self): 
        if(self.mode == 'play'):
            self.play()
        if(self.mode == 'train'):
            self.train()

    def play(self):
        raise NotImplementedError

    def train(self):
        self.trainingData = []
        for i in range(self.epochToCollectData):
            self.game.do_action(0) # jump
            state = self.game.get_state(action_code)
            image = state.get_image()
            crashed = state.is_crashed()
            epoch_data = []
            while not crashed:
#                print('iter start' + str(i), flush=True)
                action_code = self.model.get_action(image)
                self.game.do_action(action_code)
                state = self.game.get_state(action_code)
                crashed = state.is_crashed()
        #        self.game.getScore()
#                print('iter end' + str(i), flush=True)
                epoch_data.append(state)
            print("Game {} ended!".format(i))
            self.game.restart()
            self.trainingData.append(epoch_data)
        self.model.train(self.trainingData)

    def end(self):
        return self.game.end()

    def saveEnvironmentScreenshots(self, save_every_x=1):
        for epoch, epoch_data in enumerate(self.trainingData):
            for state in epoch_data[::save_every_x]:
                image = state.get_image()
#                ipdb.set_trace()
                image_name = 'env_{}_{}.jpg'.format(epoch, state.get_time_stamp())
                imwrite(os.path.join(self.pathToImageFolder, image_name), image)

        print("Saved images to {}".format(self.pathToImageFolder))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--display', default=False, action='store_true')
    args = parser.parse_args()
    model = TFRexModel()
    game = TRexGame(display=args.display)
    agent = Agent(game=game, model=model,mode='train', epochToCollectData=2)
    agent.execute()
    agent.saveEnvironmentScreenshots()
    agent.end()
