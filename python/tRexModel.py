from tensorflow.python import keras as keras
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Reshape, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from random import randint

class TFRexModel(object):
    def __init__(self):
        self.weights = None 
        self.time_to_execute_action = 0.1
        self.buffer_size = 4
        self.width = 80
        self.height = 80
        self.num_actions = 3

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape=(self.buffer_size, self.height, self.width),
                         filters=32, kernel_size=(8,8), strides=(4,4),
                         padding='valid', data_format="channels_first", activation=relu))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(4,4), strides=(2,2),
                         padding='valid', data_format="channels_first", activation=relu))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
                         padding='valid', data_format="channels_first", activation=relu))
        flatten = Flatten()
        model.add(flatten)
        dense_shape = flatten.output_shape[1]
        model.add(Dense(dense_shape))
        model.add(Dense(self.num_actions))
        
    def get_action(self, environmentState):
        return randint(0, 2)

    def get_time_to_execute_action(self):
        return self.time_to_execute_action

    def train(self, state, target, action, ):
        pass
