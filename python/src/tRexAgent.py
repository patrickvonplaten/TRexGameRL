import os
import time
import numpy as np
from imageio import imwrite
from tRexGame import TRexGame
from tRexMemory import Memory
from tRexPreprocessor import Prepocessor
from tRexLogger import Logger
import tRexUtils
import ipdb

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_IMAGE_FOLDER = os.path.join(CUR_PATH, '../../imagesToCheck')

class Agent(object):
    def __init__(self, model, mode, config):
        self.PATH_TO_WEIGHTS = config['PATH_TO_WEIGHTS']
        self.PATH_TO_LOG_FILE_TRAIN  = config['PATH_TO_LOG_FILE_TRAIN']
        self.model = model
        self.mode = mode
        self.game = TRexGame(display=config['display'], wait_after_restart=config['wait_after_restart'])
        self.time_to_execute_action = config['time_to_execute_action']
        self.memory = Memory(config['memory_size'])
        self.epoch_to_train = config['epoch_to_train']
        self.decay_fn = getattr(tRexUtils, config['decay_fn'])
        self.warmup_steps = config['warmup_steps']
        self.epsilon_final = config['epsilon_final']
        self.decay_period = config['decay_period']
        self.training_data = None
        self.path_to_image_folder = PATH_TO_IMAGE_FOLDER
        self.preprocessor = Prepocessor(vertical_crop_intervall=config['vertical_crop_intervall'],
                horizontal_crop_intervall=config['horizontal_crop_intervall'], buffer_size=config['buffer_size'], resize=config['resize_dim'])
        self.num_actions = config['num_actions']
        # Number of elements used for training. The model batch size will later determine how many updates this will lead to.
        self.batch_size = config['batch_size']
        self.logger = Logger(self.PATH_TO_LOG_FILE_TRAIN)
        if not os.path.isdir(self.path_to_image_folder):
            os.mkdir(self.path_to_image_folder)
        self.execute()

    def execute(self):
        if(self.mode == 'play'):
            self.model.load_weights(self.PATH_TO_WEIGHTS)
            self.play()
        if(self.mode == 'train'):
            self.train()

    def process_action_to_state(self, action_code):
        return self.game.process_action_to_state(action_code, self.time_to_execute_action)

    def play(self):
        state = self.game.process_to_first_state()
        while not state.is_crashed():
            image = state.get_image()
            environment = self.preprocessor.process(image)
            action = self.model.get_action(environment)
            state = self.process_action_to_state(action)
        print('Final score: {}'.format(self.game.get_score()))

    def get_epsilon(self, step):
        return self.decay_fn(step, self.decay_period, self.warmup_steps, self.epsilon_final)

    def train(self):
        self.training_data = []
        start_time = time.time()
#        self.game.start()

        for i in range(self.epoch_to_train):
            first_state = self.game.process_to_first_state()
            self.training_data.append(first_state)
            environment_prev = self.preprocessor.process(first_state.get_image())

            crashed = False
            epsilon = self.get_epsilon(i)

            while not crashed:
                random = False
                if np.random.random() < epsilon:
                    action = np.random.randint(0, self.num_actions)
                    random = True
                else:
                    action = self.model.get_action(environment_prev)

                state = self.process_action_to_state(action)
                self.training_data.append(state)

                reward = state.get_reward()
                crashed = state.is_crashed()
                image = state.get_image()
                environment_next = self.preprocessor.process(image)

                self.memory.add((environment_prev, action, reward, environment_next, crashed))
                environment_prev = environment_next

            loss = self.replay(i)
            self.logger.log_parameter(epoch=i+1, start_time=start_time, score=self.game.get_score(),
                    loss=loss, epsilon=epsilon, random=random, epoch_to_train=self.epoch_to_train)

        self.model.save_weights(self.PATH_TO_WEIGHTS)
        self.logger.close()

    def replay(self, epoch):
        if self.memory.cur_size < self.batch_size:
            return

        environment_prevs, actions, rewards, environment_nexts, crasheds = self.memory.sample(self.batch_size)
        return self.model.train(environment_prevs, actions, rewards, environment_nexts, crasheds)[0]

    def end(self):
        return self.game.end()

    def save_environment_screenshots(self, save_every_x=1):
        for state_idx, state in enumerate(self.training_data[::save_every_x]):
            image = self.preprocessor._process(state.get_image())
            image_name = 'env_{}_{}.jpg'.format(state_idx, state.get_time_stamp())
            imwrite(os.path.join(self.path_to_image_folder, image_name), image)

        print("Saved images to {}".format(self.path_to_image_folder))
