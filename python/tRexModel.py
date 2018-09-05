import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D 
from tensorflow.python.keras.activations import relu
from random import randint

class  TFRexModel(object):
    def __init__(self):
        self.weights = None 
        self.time_to_execute_action = 0.1

    def build_model(self):
        model = tf.keras.Sequential([
            Conv2D(filter=32, kernel_size=(8,8), strides=(4,4), padding='valid', data_format="channels_first", activation=relu),
#            apply maxpooling (2,2)
            Conv2D(filter=64, kernel_size=(4,4), strides=(2,2), padding='valid', data_format="channels_first", activation=relu),
            Conv2D(filter=64, kernel_size=(3,3), strides=(1,1), padding='valid', data_format="channels_first", activation=relu),
        ])
        
    def get_action(self, environmentState):
        return randint(0, 2)

    def get_time_to_execute_action(self):
        return self.time_to_execute_action

    def train(self, training_data):
        pass
