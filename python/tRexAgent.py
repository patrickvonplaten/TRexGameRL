import os
import time
import numpy as np
import datetime
from imageio import imwrite
from tRexGame import TRexGame
from tRexMemory import Memory
from tRexPreprocessor import Prepocessor
import tRexUtils

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_IMAGE_FOLDER = os.path.join(CUR_PATH, '../imagesToCheck')


class Agent(object):
    def __init__(self, model, mode, config):
        self.model = model
        self.mode = mode
        self.game = TRexGame(display=config['display'])
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
        if not os.path.isdir(self.path_to_image_folder):
            os.mkdir(self.path_to_image_folder)
        self.execute()

    def execute(self):
        if(self.mode == 'play'):
            self.play()
        if(self.mode == 'train'):
            self.train()

    def process_action_to_state(self, action_code):
        return self.game.process_action_to_state(action_code, self.time_to_execute_action)

    def play(self):
        raise NotImplementedError

    def get_epsilon(self, step):
        return self.decay_fn(step, self.decay_period, self.warmup_steps, self.epsilon_final)

    def train(self):
        self.training_data = []
        start_time = time.time()
        for i in range(self.epoch_to_train):
            action_code = 0  # jump to start game
            crashed = False
            epsilon = self.get_epsilon(i)
            image = self.process_action_to_state(action_code).get_image()

            environment_prev = self.preprocessor.process(image)

            while not crashed:
                if np.random.random() < epsilon:
                    action = np.random.randint(0, self.num_actions)
                else:
                    action = self.model.get_action(environment_prev)
                state = self.process_action_to_state(action)

                reward = state.get_reward()
                crashed = state.is_crashed()
                image = state.get_image()
                environment_next = self.preprocessor.process(image)

                self.memory.add((environment_prev, action, reward, environment_next, crashed))

                environment_prev = environment_next
            loss = self.replay(i)
            self.print_train_log(epoch=i+1, start_time=start_time, score=self.game.get_score(), loss=loss, epsilon=epsilon)

            self.game.restart()

    def print_train_log(self, epoch, start_time, score, loss, epsilon):
        time_elapsed = time.time() - start_time
        avg_time_per_epoch = time_elapsed/(epoch+1)
        time_elapsed_formatted = datetime.timedelta(seconds=int(time_elapsed))
        avg_time_per_epoch_formatted = datetime.timedelta(seconds=int(avg_time_per_epoch))
        loss_formatted = '{:.4f}'.format(loss) if not not loss else 'No train'
        log = "Epoch: {}/{} | ".format(epoch, self.epoch_to_train)
        log += "Score: {} | ".format(score)
        log += "Loss : {} | ".format(loss_formatted)
        log += "Epsilon: {0:.2f} | ".format(epsilon)
        log += "Time elapsed: {} | ".format(time_elapsed_formatted)
        log += "Avg Time Epoch: {}".format(avg_time_per_epoch_formatted)
        print(log)

    def replay(self, epoch):
        if self.memory.cur_size < self.batch_size:
            return

        environment_prevs, actions, rewards, environment_nexts, crasheds = self.memory.sample(self.batch_size)
        return self.model.train(environment_prevs, actions, rewards, environment_nexts, crasheds)[0]

    def end(self):
        return self.game.end()

    def save_environment_screenshots(self, save_every_x=1):
        for epoch, epoch_data in enumerate(self.training_data):
            for state in epoch_data[::save_every_x]:
                image = state.get_image()
                image_name = 'env_{}_{}.jpg'.format(epoch, state.get_time_stamp())
                imwrite(os.path.join(self.path_to_image_folder, image_name), image)

        print("Saved images to {}".format(self.path_to_image_folder))
