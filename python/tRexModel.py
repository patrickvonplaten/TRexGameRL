from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD
import numpy as np


class TFRexModel(object):
    def __init__(self):
        self.weights = None
        self.time_to_execute_action = 0.1
        self.buffer_size = 4
        self.width = 80
        self.height = 80
        self.num_actions = 3
        self.discount_factor = 0.99
        self.model = self.build_model()
        self.batch_size
        self.training_configs = {
            'learning_rate': 1e-3,
            'momentum': 0.9,
            'metrics': ['accuracy'],
            'loss': 'mean_squared_error'
        }

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
        
    def get_action(self, environment):
        # model weights need to be loaded before
        return self.model.predict(environment, batch_size=1)

    def get_time_to_execute_action(self):
        return self.time_to_execute_action

    def _get_targets(self, environment_prevs, actions, rewards, environment_nexts):
        q_values = self.model.predict_on_batch(environment_prevs)
        max_q_value_next = np.amax(self.model.predict(environment_nexts), axis=1)
        q_values[np.arange(q_values.shape(0)),actions] = max_q_value_next
        return q_values
        
    def train_on_batch(self, environment_prevs, actions, rewards, environment_nexts):
        #TODO: exploring with decaying prob needs to be implemented!
        assert environment_nexts.size == actions.size == rewards.size == environment_nexts.size == self.batch_size, 'all types of data needed for training should have same length'

        optimizer = SGD(lr=self.training_configs['learning_rate'], momentum=self.training_configs['momentum'])
        self.model.compile(optimizer=optimizer, loss=self.training_configs['loss'], metrics=self.training_configs['metrics'])

        x = environment_prevs
        y = self._get_targets(self, environment_prevs, actions, rewards, environment_nexts)
        self.model.train_on_batch(x, y)

    def save_trained_weights(self):
        pass
