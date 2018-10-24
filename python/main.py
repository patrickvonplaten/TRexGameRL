#!/usr/bin/env python
import sys
import os
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_TREX_MODULES = CUR_PATH + '/src'
sys.path.insert(0, PATH_TO_TREX_MODULES)

PATH_TO_MODELS = os.path.join(CUR_PATH, './models')
PATH_TO_LOG = os.path.join(CUR_PATH, './log')

from tRexModel import TFRexModel
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Input, Add, Subtract, Lambda, MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import RMSprop
from tRexLogger import Logger
from argparse import ArgumentParser
from tRexAgent import Agent
import ipdb


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
            'priority_beta_decay_period': 0,
            'clipped_max_priority_score': 0
        })
    return memory_config


def create_config(is_priority_experience_replay=True):
    config = {
        'PATH_TO_MODELS': PATH_TO_MODELS,
        'PATH_TO_LOG': PATH_TO_LOG,
        'path_to_init_weights': None,
        'save_screenshots': False,
        'layer_to_init_with_weights': ['layer1, layer2, layer3'],
        'num_actions': 2,
        'time_to_execute_action': 0.05,
        'buffer_size': 4,
        'discount_factor': 0.95,
        'batch_size': 32,
        'metrics': ['mse'],
        'loss': 'logcosh',
        'epochs_to_train': 30000,
        'vertical_crop_intervall': (50, 150),
        'horizontal_crop_intervall': (0, 400),
        'resize_dim': 80,
        'buffer_size': 4,
        'wait_after_restart': 3,
        'num_control_environments': 500,
        'copy_train_to_target_every_epoch': 20,
        'keep_models': 5,
        'save_model_every_epoch': 10,
        'optimizer': RMSprop(lr=0.00025, rho=0.9, epsilon=None, decay=0),
        'run_reward': 0,
        'jump_reward': 0,
        'duck_reward': 0,
        'crash_reward': -100
    }
    config.update(create_memory_config(is_priority_experience_replay))

    return config


def create_dqn(dqn='duel_dqn'):
    conv_initialization = 'glorot_normal'
    dense_initialization = 'glorot_normal'

    input_shape = Input(shape=(80, 80, 4))
    conv1 = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding='valid', activation=relu, kernel_initializer=conv_initialization)(input_shape)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
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
    args = parser.parse_args()

    config = create_config()
    config['display'] = args.display

    mode = 'train'
    network = create_dqn()

    logger = Logger(config)
#    model = TFRexModel.restore_from_epoch(epoch=-1, config=config, logger=logger)
    model = TFRexModel(network=network, config=config)
    agent = Agent(model=model, logger=logger, mode=mode, config=config)
    agent.save_screenshots()
    agent.end()
