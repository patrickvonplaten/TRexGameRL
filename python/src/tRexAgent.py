import os
import time
import numpy as np
from imageio import imwrite
import tRexUtils
import ipdb  # noqa: F401


CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_IMAGE_FOLDER = os.path.join(CUR_PATH, '../../imagesToCheck')


class Agent(object):
    def __init__(self, model, logger, preprocessor, game, memory, mode, config):
        self.path_to_image_folder = PATH_TO_IMAGE_FOLDER
        self.game = game
        self.memory = memory
        self.epochs_to_train = config['epochs_to_train']
        self.decay_fn = getattr(tRexUtils, config['decay_fn'])
        self.warmup_steps = config['warmup_steps']
        self.epsilon_final = config['epsilon_final']
        self.epsilon_init = config['epsilon_init']
        self.decay_period = config['decay_period']
        self.training_data = None
        self.num_control_environments = config['num_control_environments']
        self.copy_train_to_target_every_epoch = config['copy_train_to_target_every_epoch']
        self.mode = mode
        self.model = model
        self.preprocessor = preprocessor
        self.logger = logger
        if not os.path.isdir(self.path_to_image_folder):
            os.mkdir(self.path_to_image_folder)
        self.control_environments = np.zeros((self.num_control_environments, ) + self.preprocessor.environment_processed_shape)
        self.execute()
        self.save_screenshots()
        self.end()

    def execute(self):
        if(self.mode == 'play'):
            self.play()
        if(self.mode == 'train'):
            self.train()

    def process_action_to_state(self, action_code):
        return self.game.process_action_to_state(action_code)

    def play(self):
        self.model.load_weights()
        state = self.game.process_to_first_state()
        while not state.is_crashed():
            image = state.get_image()
            environment = self.preprocessor.process(image)
            action = self.model.get_action(environment)
            state = self.process_action_to_state(action)
        print('Final score: {}'.format(self.game.get_score()))

    def get_epsilon(self, step):
        return self.decay_fn(step, self.epsilon_init, self.decay_period, self.warmup_steps, self.epsilon_final)

    def train(self):
        self.collect_control_environment_set(self.num_control_environments)
        start_time = time.time()

        for epoch in range(self.model.start_epoch, self.epochs_to_train):
            first_state = self.game.process_to_first_state()
            environment_prev = self.preprocessor.process(first_state.get_image())
            crashed = False
            reward_sum = 0
            epsilon = self.get_epsilon(epoch)

            while not crashed:
                action = self.get_action(epsilon, environment_prev)
                state = self.process_action_to_state(action)

                reward = state.get_reward()
                crashed = state.is_crashed()
                image = state.get_image()
                environment_next = self.preprocessor.process(image)

                self.memory.add((environment_prev, action, reward, environment_next, crashed))
                environment_prev = environment_next
                reward_sum += reward

            loss = self.replay(epoch)
            avg_control_q = self.get_sum_of_q_values_over_control_envs()
            self.logger.log_parameter(epoch=epoch, start_time=start_time, score=self.game.get_score(),
                    loss=loss, epsilon=epsilon, epochs_to_train=self.epochs_to_train,
                    reward=reward_sum, avg_control_q=avg_control_q, start_epoch=self.model.start_epoch)  # noqa: E128
            self.logger.save_model(epoch, self.model.train_model)
        self.logger.close()

    def get_action(self, epsilon, environment_prev):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.model.get_num_actions())
        return self.model.get_action(environment_prev)

    def replay(self, epoch):
        if self.memory.cur_size < self.memory.get_batch_size():
            return
        if(epoch % self.copy_train_to_target_every_epoch is 0):
            self.model.copy_weights_to_target_model()
        batch, sample_weights = self.memory.sample(epoch)
        losses = self.model.train(batch, sample_weights)
        self.memory.update(losses)
        return np.mean(losses)

    def collect_control_environment_set(self, num_control_environments):
        state = None
        for i in range(num_control_environments):
            state = self.get_random_state(state)
            self.control_environments[i] = self.preprocessor.process(state.get_image())
        self.preprocessor.reset()

    def get_sum_of_q_values_over_control_envs(self):
        # get the predicted q from a control set. Good for plotting progress (see Atari paper)
        if(self.control_environments.size is 0):
            return 0
        q_values = self.model.predict_on_batch(self.control_environments, self.model.train_model)
        return np.average(np.max(q_values, axis=1))

    def get_random_state(self, prev_state):
        if(prev_state is None or prev_state.is_crashed()):
            return self.game.process_to_first_state()
        random_action = self.get_action(1, None)
        return self.process_action_to_state(random_action)

    def end(self):
        return self.game.end()

    def save_screenshots(self, save_every_x=1):
        if(self.preprocessor.screenshots_for_visual is None):
            print('No screenshots to be saved!')
            return
        for image_idx, image in enumerate(self.preprocessor.screenshots_for_visual[::save_every_x]):
            image_name = 'env_{}.jpg'.format(image_idx)
            imwrite(os.path.join(self.path_to_image_folder, image_name), image)

        print("Saved images to {}".format(self.path_to_image_folder))
