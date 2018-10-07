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

    def process_to_first_state(self):
        self.timestamp = 0
        return self._process_to_first_state()

    def end(self):
        raise NotImplementedError()

    def restart(self):
        return self._restart()

    def process_action_to_state(self, action_code, time_to_execute_action):
        self.timestamp += 1
        return self._process_action_to_state(action_code, time_to_execute_action)


class TRexGame(Game):
    def __init__(self, display=False, wait_after_restart=0):
        super().__init__()
        self.chrome_driver = ChromeDriver(display)
        self.wait_after_restart = wait_after_restart
        jump = Action(self._press_up, -1, "jump")
        duck = Action(self._press_down, -3, "duck")
        run = Action(lambda: None, 0, "run")
        self.actions = [jump, run, duck]

    def _press_up(self):
        return self.chrome_driver.press_up()

    def _press_down(self):
        return self.chrome_driver.press_down()

    def _process_to_first_state(self):
        self._restart()
        return self._process_action_to_state(1, self.wait_after_restart)

    def _restart(self):
        self.chrome_driver.execute_script("Runner.instance_.restart()")

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
