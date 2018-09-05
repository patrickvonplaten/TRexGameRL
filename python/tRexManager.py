#!/usr/bin/env python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from matplotlib import pyplot
from tRexModel import TFRexModel
from imageio import imwrite
from collections import deque
import os
import time
import base64
import numpy as np
import cv2
import ipdb
from argparse import ArgumentParser

HEIGHT = 150
WIDTH = 600
CHROME_EXECUTABLEPATH = '/usr/bin/chromedriver'
CHROME_PATH = '/usr/bin/google-chrome'
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
TREX_HTML_PATH = 'file://{}/../javascript/index.html'.format(CUR_PATH)
PATH_TO_IMAGE_FOLDER = os.path.join(CUR_PATH, '../imagesToCheck')


class ChromeDriver(object):

    def __init__(self, display):
        chrome_options = ['disable-infobars']
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

    def is_crashed(self):
        raise NotImplementedError()

    def is_running(self):
        raise NotImplementedError()

    def _get_state(self, action):
        raise NotImplementedError()

    def restart(self):
        self.timestamp = 0
        return self._restart()

    def end(self):
        raise NotImplementedError()

    def process_action_to_state(self, action_code, time_to_execute_action):
        self.timestamp += 1
        return self._process_action_to_state(action_code, time_to_execute_action)


class TRexGame(Game):
    def __init__(self, display=False):
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

    def _process_action_to_state(self, action_code, time_to_execute_action):
        start_time = time.time()
        self.actions[action_code]()
        time_needed_to_execute_action = time.time() - start_time
        time_difference = time_to_execute_action - time_needed_to_execute_action
        if(time_difference > 0):
            time.sleep(time_difference)
        return self._get_state(action_code)

    def _get_state(self, action_code):
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

    def get_image(self):
        return self.image

    def is_crashed(self):
        return self.crashed

    def get_time_stamp(self):
        return self.timestamp

    def get_reward(self):
        return self.reward


class Agent(object):
    def __init__(self, game, model, mode, epoch_to_collect_data):
        self.game = game
        self.model = model
        self.time_to_execute_action = model.get_time_to_execute_action()
        self.mode = mode
        self.memory = Memory(10000)
        self.epoch_to_collect_data = epoch_to_collect_data
        self.training_data = None
        self.path_to_image_folder = PATH_TO_IMAGE_FOLDER
        self.preprocessor = Prepocessor(vertical_crop_intervall=(50,150), horizontal_crop_intervall=(0,400), buffer_size=4)

        if not os.path.isdir(self.path_to_image_folder):
            os.mkdir(self.path_to_image_folder)

    def execute(self):
        if(self.mode == 'play'):
            self.play()
        if(self.mode == 'train'):
            self.train()

    def process_action_to_state(self, action_code):
        return self.game.process_action_to_state(action_code, self.time_to_execute_action)

    def play(self):
        raise NotImplementedError

    def train(self):
        self.training_data = []
        for i in range(self.epoch_to_collect_data):
            action_code = 0  # jump to start game
            crashed = False
            image = self.process_action_to_state(action_code).get_image()
            environment_prev = self.preprocessor.process(image)

            while not crashed:
                action = self.model.get_action(environment_prev)
                state = self.process_action_to_state(action)

                reward = state.get_reward()
                crashed = state.is_crashed()
                image = state.get_image()
                environment_next = self.preprocessor.process(image)

                sample = Sample(environment_prev, action, reward, environment_next, crashed)
                self.memory.add(sample)

                environment_prev = environment_next
            print("Game {} ended!".format(i))
#            ipdb.set_trace()
            self.replay()
            self.game.restart()

    def replay(self):
        pass

    def end(self):
        return self.game.end()

    def save_environment_screenshots(self, save_every_x=1):
        for epoch, epoch_data in enumerate(self.training_data):
            for state in epoch_data[::save_every_x]:
                image = state.get_image()
                image_name = 'env_{}_{}.jpg'.format(epoch, state.get_time_stamp())
                imwrite(os.path.join(self.path_to_image_folder, image_name), image)

        print("Saved images to {}".format(self.path_to_image_folder))


class Memory(object):
    """
    Stores states for replay.
    """
    def __init__(self, size):
        """
        Args:
            size: storage size
            state_size: Number of elements one state exists of.
        """
        # This should be general.
        SAMPLE_SIZE = 5
        self.storage = np.ndarray((size, SAMPLE_SIZE), dtype=object)
        self.size = size
        self.pos = 0
        self.cur_size = 0

    def add(self, state):
        """Adds elements to the storage.

        Args (State): The state to store.
        """
        self.storage[self.pos] = state.to_list()
        self.pos = (self.pos + 1) % self.size
        self.cur_size = min(self.cur_size + 1, self.size)

    def sample(self, n):
        """Sample n images from memory."""
        if self.cur_size < n:
            return
        idxs = np.random.choice(self.cur_size, n, replace=False)
        return self.storage[idxs, :]



class Prepocessor(object):
    def __init__(self, vertical_crop_intervall, horizontal_crop_intervall, buffer_size):
        self.vertical_crop_start = vertical_crop_intervall[0]
        self.vertical_crop_end = vertical_crop_intervall[1]
        self.vertical_crop_length = self.vertical_crop_start - self.vertical_crop_end
        self.horizontal_crop_start = horizontal_crop_intervall[0]
        self.horizontal_crop_end = horizontal_crop_intervall[1]
        self.horizontal_crop_length = self.horizontal_crop_start - self.horizontal_crop_end
        self.buffer_size = buffer_size
        self.image_processed_buffer = deque([None, None, None, None], maxlen=self.buffer_size)

    def process(self, image):
        image_processed = self.crop_image(image)
        self.image_processed_buffer.pop()
        self.image_processed_buffer.appendleft(image_processed)

        # copy otherwise reference will be returned and data inside always overwritten.
        environment = self.image_processed_buffer.copy()
        return environment

    def crop_image(self, image):
        return image[self.vertical_crop_start:self.vertical_crop_end, self.horizontal_crop_start:self.horizontal_crop_end]


class Sample(object):

    def __init__(self, environment_prev, action, reward, environment_next, crashed):
        self.environment_prev = environment_prev
        self.action = action
        self.reward = reward
        self.environment_next = environment_next
        self.crashed = crashed

    def get_environment_prev(self):
        return self.environment_prev

    def get_environment_next(self):
        return self.environment_next

    def get_reward(self):
        return self.reward

    def get_action(self):
        return self.action

    def to_list(self):
        return [self.environment_prev, self.action, self.reward, self.environment_next, self.crashed]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--display', default=False, action='store_true')
    args = parser.parse_args()
    model = TFRexModel()
    game = TRexGame(display=args.display)
    agent = Agent(game=game, model=model, mode='train', epoch_to_collect_data=2)
    agent.execute()
    agent.save_environment_screenshots()
    agent.end()
