import os
import time
import numpy as np
from imageio import imwrite
import tRexUtils
import ipdb  # noqa: F401

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_IMAGE_FOLDER = os.path.join(CUR_PATH, '../../imagesToCheck')


class Agent(object):
    def __init__(self, model, logger, preprocessor, game, memory, config):
        self.path_to_image_folder = PATH_TO_IMAGE_FOLDER
        self.model = model
        self.logger = logger
        self.preprocessor = preprocessor
        self.game = game
        self.memory = memory
        self.epochs_to_train = config['epochs_to_train']
        self.decay_fn = getattr(tRexUtils, config['decay_fn'])
        self.warmup_steps = config['warmup_steps']
        self.epsilon_final = config['epsilon_final']
        self.epsilon_init = config['epsilon_init']
        self.decay_period = config['decay_period']
        self.mode = config['mode']
        self.num_control_environments = config['num_control_environments']
        self.copy_train_to_target_every_epoch = config['copy_train_to_target_every_epoch']
        self.current_collected_reward = 0
        self.training_data = None
        self.control_environments = np.zeros((self.num_control_environments, ) + self.preprocessor.environment_processed_shape)

    def run(self):
        if(self.mode == 'play'):
            self.play()
        elif(self.mode == 'train'):
            self.train()

    def process_action_to_state(self, action_code):
        return self.game.process_action_to_state(action_code)

    def play(self):
        while True:
            state = self.game.process_to_first_state()
            while not state.is_crashed():
                image = state.get_image()
                environment = self.preprocessor.process(image)
                action = self.model.get_action(environment)
                state = self.process_action_to_state(action)
            print('Score: {}'.format(self.game.get_score()))

    def get_epsilon(self, step):
        return self.decay_fn(step, self.epsilon_init, self.decay_period, self.warmup_steps, self.epsilon_final)

    def train(self):
        self.collect_control_environment_set(self.num_control_environments)
        start_time = time.time()
        for epoch in range(self.model.start_epoch, self.epochs_to_train):
            self.train_epoch(epoch, start_time)
        self.logger.close()

    def train_epoch(self, epoch, start_time):
        first_state = self.game.process_to_first_state()
        first_image = first_state.get_image()
        self.environment = self.preprocessor.process(first_image)
        epsilon = self.get_epsilon(epoch)
        self.current_collected_reward = 0
        is_crashed = 0
        while not is_crashed:
            is_crashed = self.fill_memory(epsilon)
        loss = self.replay(epoch)
        avg_control_q = self.get_sum_of_q_values_over_control_envs()
        reward = self.current_collected_reward
        self.logger.log_parameter(epoch=epoch, start_time=start_time, score=self.game.get_score(),
                loss=loss, epsilon=epsilon, epochs_to_train=self.epochs_to_train,
                reward=reward, avg_control_q=avg_control_q, start_epoch=self.model.start_epoch)  # noqa: E128

    def fill_memory(self, epsilon):
        # process to next state and add (s-1, a, r, s) to memory
        environment_prev = self.environment
        action = self.get_action(epsilon, environment_prev)
        state = self.process_action_to_state(action)
        is_crashed = state.is_crashed()
        image = state.get_image()
        reward = state.get_reward()
        environment_next = self.preprocessor.process(image)
        self.memory.add((environment_prev, action, reward, environment_next, is_crashed))
        self.current_collected_reward += reward
        self.environment = environment_next
        return is_crashed

    def get_action(self, epsilon, environment_prev):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.model.get_num_actions())
        return self.model.get_action(environment_prev)

    def replay(self, epoch):
        # train DQN
        if self.memory.cur_size < self.memory.get_batch_size():
            return
        if(epoch % self.copy_train_to_target_every_epoch is 0):
            self.model.copy_weights_to_target_model()
        batch, sample_weights = self.memory.sample(epoch)
        losses = self.model.train(batch, sample_weights)
        self.memory.update(losses)
        return np.mean(losses)

    def collect_control_environment_set(self, num_control_environments):
        # run once in beginning to collect samples for monitoring
        state = None
        for i in range(num_control_environments):
            state = self.get_random_state(state)
            self.control_environments[i] = self.preprocessor.process(state.get_image())
        self.preprocessor.reset()

    def get_sum_of_q_values_over_control_envs(self):
        # get the predicted q from a control set. Good for plotting progress (see Atari paper)
        if(self.control_environments.size is 0):
            return 0
        q_values = self.model.train_model.predict_on_batch(self.control_environments)
        return np.average(np.max(q_values, axis=1))

    def get_random_state(self, prev_state):
        if(prev_state is None or prev_state.is_crashed()):
            return self.game.process_to_first_state()
        random_action = self.get_action(1, None)
        return self.process_action_to_state(random_action)

    def end(self):
        return self.game.end()

    def save_screenshots(self, save_every_x=1):
        if not os.path.isdir(self.path_to_image_folder):
            os.mkdir(self.path_to_image_folder)
        if(self.preprocessor.screenshots_for_visual is None):
            print('No screenshots to be saved!')
            return
        for image_idx, image in enumerate(self.preprocessor.screenshots_for_visual[::save_every_x]):
            image_name = 'env_{}.jpg'.format(image_idx)
            path_to_image = os.path.join(self.path_to_image_folder, image_name)
            imwrite(path_to_image, image)
        print("Saved images to {}".format(self.path_to_image_folder))
