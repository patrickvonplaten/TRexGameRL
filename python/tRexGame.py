from tRexDriver import ChromeDriver
from matplotlib import pyplot
import time
import numpy as np


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
        self.actions = [jump, run, duck]

    def _press_up(self):
        return self.chrome_driver.press_up()

    def _press_down(self):
        return self.chrome_driver.press_down()

    def _restart(self):
        return self.chrome_driver.execute_script("Runner.instance_.restart()")

    def is_crashed(self):
        return self.chrome_driver.execute_script("return Runner.instance_.crashed")

    def is_running(self):
        return self.chrome_driver.execute_script("return Runner.instance_.playing")

    def get_score(self):
        scoreArray = self.chrome_driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(scoreArray)
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


class Action(object):
    def __init__(self, action_fn, reward, code=None):
        self.action = action_fn
        self.code = code
        self.reward = reward

    def __call__(self):
        self.action()


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