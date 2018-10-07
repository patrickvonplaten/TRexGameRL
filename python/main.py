#!/usr/bin/env python3

import os
import sys

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_TREX_MODULES = CUR_PATH+'/src'
sys.path.insert(0, PATH_TO_TREX_MODULES)

PATH_TO_WEIGHTS = os.path.join(CUR_PATH, 'model.h5')
PATH_TO_LOG_FILE_TRAIN = os.path.join(CUR_PATH, 'train_log.txt')

from tRexModel import TFRexModel
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD, RMSprop
from argparse import ArgumentParser
from tRexAgent import Agent

mode = 'train'

config = {
    'PATH_TO_WEIGHTS': PATH_TO_WEIGHTS,
    'PATH_TO_LOG_FILE_TRAIN': PATH_TO_LOG_FILE_TRAIN,
    'num_actions': 2,
    'time_to_execute_action': 0.1,
    'buffer_size': 4,
    'discount_factor': 0.99,
    'batch_size': 32,
    'metrics': ['mse'],
    'loss': 'logcosh',
    'epoch_to_train': 1,
    'vertical_crop_intervall': (50, 150),
    'horizontal_crop_intervall': (0, 400),
    'memory_size': 10000,
    'resize_dim': 80,
    'buffer_size': 4,
    'warmup_steps': 0,
    'epsilon_final': 0.05,
    'decay_fn': 'linearly_decaying_epsilon',
    'decay_period': 20,
    'wait_after_restart': 1.5,
    'num_control_environments': 10,
}

optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=None, decay=0.0)

network = Sequential([
    Conv2D(input_shape=(80, 80, 4), filters=32, kernel_size=(8, 8), strides=(4, 4), padding='valid', activation=relu, kernel_initializer='random_uniform'),
    Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation=relu, kernel_initializer='random_uniform'),
    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=relu, kernel_initializer='random_uniform'),
    Flatten(),
    Dense(512, activation=relu, kernel_initializer='random_uniform'),
    Dense(config['num_actions'], kernel_initializer='random_uniform'),
])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--display', default=False, action='store_true')
    args = parser.parse_args()
    config['display'] = args.display

    model = TFRexModel(network=network, optimizer=optimizer, config=config)
    agent = Agent(model=model, mode=mode, config=config)
    agent.end()
