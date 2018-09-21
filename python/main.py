#!/usr/bin/env python3

from tRexModel import TFRexModel
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD, RMSprop
from argparse import ArgumentParser
from tRexAgent import Agent

mode = 'play'

config = {
    'num_actions': 2,
    'time_to_execute_action': 0.1,
    'buffer_size': 4,
    'discount_factor': 0.99,
    'batch_size': 32,
    'metrics': ['mse'],
    'loss': 'logcosh',
    'epoch_to_train': 2,
    'vertical_crop_intervall': (50, 150),
    'horizontal_crop_intervall': (0, 400),
    'memory_size': 10000,
    'resize_dim': 80,
    'buffer_size': 4,
    'warmup_steps': 0,
    'epsilon_final': 0.05,
    'decay_fn': 'linearly_decaying_epsilon',
    'decay_period': 20,
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
