import numpy as np
from tensorflow.python.keras.layers import Dense  # noqa: F401
from tensorflow.python.keras.activations import relu  # noqa: F401


def linearly_decaying_epsilon(step, epsilon_init, decay_period, warmup_steps, epsilon_final):
    assert epsilon_init > epsilon_final
    steps_left = decay_period + warmup_steps - step
    bonus = (epsilon_init - epsilon_final) * steps_left / decay_period
    bonus = np.clip(bonus, 0., epsilon_init - epsilon_final)
    return epsilon_final + bonus


def linearly_decaying_beta(step, decay_period, beta):
    steps_left = decay_period - step
    bonus = (1.0 - beta) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1.0 - beta)
    return beta + bonus


def average_tensor(x):
    from tensorflow.python.keras.backend import mean
    return mean(x, axis=1)


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def convert_config_to_correct_type(variable):
    variable_type = type(variable)
    if(variable_type is not dict and variable_type is not list):
        return convert_to_value(variable)
    elif(variable_type is dict):
        return convert_to_dict(variable)
    elif(variable_type is list):
        return convert_to_value_list(variable)
    return variable


def convert_to_dict(variable):
    dictionary = variable
    for key in dictionary.keys():
        dictionary[key] = convert_config_to_correct_type(dictionary[key])
    return dictionary


def convert_to_value_list(variable):
    value_list = variable
    for idx, value in enumerate(value_list):
        value_list[idx] = convert_config_to_correct_type(value)
    return value_list


def convert_to_value(variable):
    value = variable
    if(is_int(value)):
        return int(value)
    elif(is_float(value)):
        return float(value)
    return value
