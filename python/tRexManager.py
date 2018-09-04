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
        b64screenshot = screenshot.encode()
        return base64.b64decode(b64screenshot)

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
    def __init__(self):
        self.timestamp = 0

    def show_environment_image(self, env_image):
        pyplot.imshow(env_image)
        pyplot.show()

    def get_environmentState(self):
        """Deprecate
        Frames per sample should also be preprocessing.
        """
        pass

    def is_crashed(self):
        raise NotImplementedError()

    def is_running(self):
        raise NotImplementedError()

    def get_state(self, action):
        raise NotImplementedError()

    def restart(self):
        self.timestamp = 0
        return self._restart()

    def end(self):
        raise NotImplementedError()

    def do_action(self, action_code, time_to_execute_action):
        self.timestamp += 1
        return self._do_action(action_code, time_to_execute_action)

class TRexGame(Game):
    def __init__(self,display=False):
        super().__init__()
        self.chrome_driver = ChromeDriver(display)
        jump = Action(self._press_up, -5, "jump")
        duck = Action(self._press_down, -3, "duck")
        run = Action(lambda: None, 1, "run")
        self.actions = [jump, duck, run]

    def _press_up(self):
        return self.chrome_driver.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def _press_down(self):
        return self.chrome_driver.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def _restart(self):
        return self.chrome_driver.driver.execute_script("Runner.instance_.restart()")

    def is_crashed(self):
        return self.chrome_driver.driver.execute_script("return Runner.instance_.crashed")

    def is_running(self):
        return self.chrome_driver.driver.execute_script("return Runner.instance_.playing")

    def get_score(self):
        scoreArray = self.chrome_driver.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(scoreArray) 
        print('Score' + str(score))
        return int(score)

    def end(self):
        self.chrome_driver.driver.quit()

    def _do_action(self, action_code, time_to_execute_action):
        start_time = time.time()
        self.actions[action_code]()
        time_needed_to_execute_action = time.time() - start_time
        if(time_to_execute_action - time_needed_to_execute_action > 0):
            time.sleep(time_to_execute_action - time_needed_to_execute_action)

    def get_state(self, action_code):
        crashed = self.is_crashed()
        if crashed:
            reward = -100
        else:
            action = self.actions[action_code]
            reward = action.reward

        image = self.chrome_driver.get_image_as(np.uint8)
        return State(image, reward, crashed, self.timestamp)

class State(object):
    def __init__(self, image, reward, crashed, timestamp):
        self.image = image
        self.reward = reward
        self.crashed = crashed
        self.timestamp = timestamp
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
    def __init__(self, game, model, mode, epoch_to_collect_data):
        self.game = game
        self.model = model
        self.time_to_execute_action = model.get_time_to_execute_action()
        self.mode = mode 
        self.epoch_to_collect_data = epoch_to_collect_data
        self.training_data = None
        self.path_to_image_folder = PATH_TO_IMAGE_FOLDER
        if not os.path.isdir(self.path_to_image_folder):
            os.mkdir(self.path_to_image_folder)

    def execute(self): 
        if(self.mode == 'play'):
            self.play()
        if(self.mode == 'train'):
            self.train()

    def do_action(self, action_code):
        return self.game.do_action(action_code, self.time_to_execute_action)

    def play(self):
        raise NotImplementedError

    def train(self):
        self.training_data = []
        for i in range(self.epoch_to_collect_data):
            action_code = 0 # jump to start game
            self.do_action(action_code) 
            state = self.game.get_state(action_code)
            environment = state.get_image()
            crashed = state.is_crashed()
            epoch_data = []
            while not crashed:
#                print('iter start' + str(i), flush=True)
                action_code = self.model.get_action(environment)
                self.do_action(action_code)
                state = self.game.get_state(action_code)
                crashed = state.is_crashed()
        #        self.game.get_score()
#                print('iter end' + str(i), flush=True)
                epoch_data.append(state)
            print("Game {} ended!".format(i))
            self.game.restart()
            self.training_data.append(epoch_data)
        self.model.train(self.training_data)

    def end(self):
        return self.game.end()

    def save_environment_screenshots(self, save_every_x=1):
        for epoch, epoch_data in enumerate(self.training_data):
            for state in epoch_data[::save_every_x]:
                image = state.get_image()
#                ipdb.set_trace()
                image_name = 'env_{}_{}.jpg'.format(epoch, state.get_time_stamp())
                imwrite(os.path.join(self.path_to_image_folder, image_name), image)

        print("Saved images to {}".format(self.path_to_image_folder))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--display', default=False, action='store_true')
    args = parser.parse_args()
    model = TFRexModel()
    game = TRexGame(display=args.display)
    agent = Agent(game=game, model=model,mode='train', epoch_to_collect_data=2)
    agent.execute()
    agent.save_environment_screenshots()
    agent.end()
