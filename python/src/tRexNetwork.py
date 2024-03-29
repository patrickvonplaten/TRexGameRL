from tensorflow.python.keras.activations import relu  # noqa: E402
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Add, Subtract, Lambda  # noqa: E402
import tRexUtils


def base_network(input_shape, conv_initialization):
    conv1 = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding='valid', activation=relu, kernel_initializer=conv_initialization)(input_shape)
    conv2 = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation=relu, kernel_initializer=conv_initialization)(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=relu, kernel_initializer=conv_initialization)(conv2)
    flatten = Flatten()(conv3)
    return flatten


def standard_dqn(flatten, dense_initialization, num_actions):
    # original network
    dense = Dense(512, activation=relu, kernel_initializer=dense_initialization)(flatten)
    out = Dense(num_actions, kernel_initializer=dense_initialization)(dense)
    return out


def duel_dqn(flatten, dense_initialization, num_actions):
    # implementation of the duel network architecture (see http://proceedings.mlr.press/v48/wangf16.pdf)
    # value stream
    dense_value = Dense(512, activation=relu, kernel_initializer=dense_initialization)(flatten)
    out_value = Dense(1, kernel_initializer=dense_initialization)(dense_value)
    # advantage stream
    dense_advantage = Dense(512, activation=relu, kernel_initializer=dense_initialization)(flatten)
    out_std_advantage = Dense(num_actions, kernel_initializer=dense_initialization)(dense_advantage)
    average_tensor = getattr(tRexUtils, 'average_tensor')
    out_avg_advantage = Lambda(average_tensor)(out_std_advantage)
    out_advantage = Subtract()([out_std_advantage, out_avg_advantage])
    # combine
    out = Add()([out_value, out_advantage])
    return out
