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
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Input, Add, Subtract, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import RMSprop
from tRexLogger import Logger
from argparse import ArgumentParser
from tRexAgent import Agent
import ipdb

mode = 'train'

config = {
    'PATH_TO_MODELS': PATH_TO_MODELS,
    'PATH_TO_LOG': PATH_TO_LOG,
    'path_to_init_weights': None,
    'layer_to_init_with_weights': ['layer1, layer2, layer3'], # not set up yet
    'num_actions': 2,
    'time_to_execute_action': 0.1,
    'buffer_size': 4,
    'discount_factor': 0.99,
    'batch_size': 32,
    'metrics': ['mse'],
    'loss': 'logcosh',
    'epochs_to_train': 20000,
    'vertical_crop_intervall': (50, 150),
    'horizontal_crop_intervall': (0, 400),
    'memory_size': 10000,
    'resize_dim': 80,
    'buffer_size': 4,
    'warmup_steps': 100,
    'epsilon_final': 0.01,
    'decay_fn': 'linearly_decaying_epsilon',
    'decay_period': 1500,
    'wait_after_restart': 1.5,
    'num_control_environments': 400,
    'copy_train_to_target_every_epoch': 1,
    'keep_models': 5,
    'save_model_every_epoch': 10,
    'restore_from_epoch': None
}

optimizer = RMSprop(lr=0.00025, rho=0.9, epsilon=None, decay=0)
conv_initialization = 'glorot_normal'
dense_initialization = 'glorot_normal'

input_shape = Input(shape=(80, 80, 4))
conv1 = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding='valid', activation=relu, kernel_initializer=conv_initialization)(input_shape)
conv2 = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation=relu, kernel_initializer=conv_initialization)(conv1)
conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=relu, kernel_initializer=conv_initialization)(conv2)
flatten = Flatten()(conv3)


def create_standard_dqn():
    dense = Dense(512, activation=relu, kernel_initializer=dense_initialization)(flatten)
    out = Dense(config['num_actions'], kernel_initializer=dense_initialization)(dense)
    return out


def average_tensor(x):
    from tensorflow.python.keras.backend import mean
    return mean(x, axis=1)


def create_duel_dqn():
    dense_value = Dense(512, activation=relu, kernel_initializer=dense_initialization)(flatten)
    out_value = Dense(1, kernel_initializer=dense_initialization)(dense_value)

    dense_advantage = Dense(512, activation=relu, kernel_initializer=dense_initialization)(flatten)
    out_std_advantage = Dense(config['num_actions'], kernel_initializer=dense_initialization)(dense_advantage)
    out_avg_advantage = Lambda(average_tensor)(out_std_advantage)
    out_advantage = Subtract()([out_std_advantage, out_avg_advantage])
    out = Add()([out_value, out_advantage])
    return out


network = Model(inputs=input_shape, outputs=create_duel_dqn())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--display', default=False, action='store_true')
    args = parser.parse_args()
    config['display'] = args.display
    model = TFRexModel(network=network, optimizer=optimizer, config=config)
    logger = Logger(config)
    agent = Agent(model=model, logger=logger, mode=mode, config=config)
    agent.end()
