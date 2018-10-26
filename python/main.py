#!/usr/bin/env python
import sys
import os
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_TREX_MODULES = CUR_PATH + '/src'
sys.path.insert(0, PATH_TO_TREX_MODULES)

PATH_TO_MODELS = os.path.join(CUR_PATH, './models')
PATH_TO_LOG = os.path.join(CUR_PATH, './log')

from tRexModel import TFRexModel  # noqa: E402
from tensorflow.python.keras.activations import relu  # noqa: E402
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Input, Add, Subtract, Lambda  # noqa: E402
from tensorflow.python.keras.models import Model  # noqa: E402
from tensorflow.python.keras.optimizers import RMSprop  # noqa: E402
from tRexLogger import Logger  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from tRexAgent import Agent  # noqa: E402
import ipdb  # noqa: E402, F401


def create_memory_config(is_priority_experience_replay):
    memory_config = {
        'memory_size': 50000,
        'warmup_steps': 20,
        'epsilon_init': 0.1,
        'epsilon_final': 0,
        'decay_fn': 'linearly_decaying_epsilon',
        'decay_period': 4000,
        'priority_epsilon': 0.01,
        'priority_alpha': 0.6,
        'priority_beta': 0.4,
        'priority_beta_decay_period': 10000,
        'clipped_max_priority_score': 1
    }
    if not is_priority_experience_replay:
        memory_config.update({
            'priority_epsilon': 0,
            'priority_alpha': 0,
            'priority_beta': 0,
            'priority_beta_decay_period': 1,
            'clipped_max_priority_score': 0
        })
    return memory_config


def create_log_config():
    return {
        'PATH_TO_LOG': PATH_TO_LOG,
        'PATH_TO_MODELS': PATH_TO_MODELS,
        'keep_models': 5,
        'save_model_every_epoch': 1,
        'running_avg': 200
    }


def create_model_config():
    return {
        'time_to_execute_action': 0.1,
        'batch_size': 32,
        'metrics': ['mse'],
        'loss': 'logcosh',
        'optimizer': RMSprop(lr=0.00025, rho=0.9, epsilon=None, decay=0),
        'discount_factor': 0.99
    }


def create_agent_config():
    return {
        'epochs_to_train': 2,
        'num_control_environments': 0,
        'copy_train_to_target_every_epoch': 20
    }


def create_game_config():
    return {
        'wait_after_restart': 1.5,
        'crash_reward': -100,
        'run_reward': 1,
        'jump_reward': -1,
        'duck_reward': 0
    }


def create_preprocessor_config():
    return {
        'vertical_crop_intervall': (50, 150),
        'horizontal_crop_intervall': (0, 400),
        'resize_dim': 80,
        'buffer_size': 4,
        'save_screenshots': False
    }


def create_debug_config():
    return {
        'epochs_to_train': 2,
        'num_control_environments': 0
    }


def create_config(is_priority_experience_replay=True, is_debug=False):
    config = {}
    config.update(create_memory_config(is_priority_experience_replay))
    config.update(create_log_config())
    config.update(create_model_config())
    config.update(create_agent_config())
    config.update(create_game_config())
    config.update(create_preprocessor_config())
    if(is_debug):
        config.update(create_debug_config())
    return config


def create_dqn(dqn='duel_dqn'):
    conv_initialization = 'glorot_normal'
    dense_initialization = 'glorot_normal'

    input_shape = Input(shape=(80, 80, 4))
    conv1 = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding='valid', activation=relu, kernel_initializer=conv_initialization)(input_shape)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation=relu, kernel_initializer=conv_initialization)(max_pool1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=relu, kernel_initializer=conv_initialization)(conv2)
    flatten = Flatten()(conv3)

    def standard_dqn():
        dense = Dense(512, activation=relu, kernel_initializer=dense_initialization)(flatten)
        out = Dense(config['num_actions'], kernel_initializer=dense_initialization)(dense)
        return out

    def duel_dqn():

        def average_tensor(x):
            from tensorflow.python.keras.backend import mean
            return mean(x, axis=1)

        dense_value = Dense(512, activation=relu, kernel_initializer=dense_initialization)(flatten)
        out_value = Dense(1, kernel_initializer=dense_initialization)(dense_value)

        dense_advantage = Dense(512, activation=relu, kernel_initializer=dense_initialization)(flatten)
        out_std_advantage = Dense(config['num_actions'], kernel_initializer=dense_initialization)(dense_advantage)
        out_avg_advantage = Lambda(average_tensor)(out_std_advantage)
        out_advantage = Subtract()([out_std_advantage, out_avg_advantage])
        out = Add()([out_value, out_advantage])
        return out

    create_dqn = {
        'standard_dqn': standard_dqn,
        'duel_dqn': duel_dqn
    }

    return Model(inputs=input_shape, outputs=create_dqn[dqn]())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    config = create_config(is_debug=args.debug)
    config['display'] = args.display

    mode = 'train'
    network = create_dqn()

    logger = Logger(config)
#    model = TFRexModel.restore_from_epoch(epoch=-1, config=config, logger=logger)
    model = TFRexModel(network=network, config=config, logger=logger)
    agent = Agent(model=model, logger=logger, mode=mode, config=config)
    agent.save_screenshots()
    agent.end()
