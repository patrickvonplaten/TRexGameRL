from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD
import numpy as np
import ipdb


class TFRexModel(object):
    def __init__(self):
        self.weights = None
        self.time_to_execute_action = 0.1
        self.buffer_size = 4
        self.width = 80
        self.height = 80
        self.input_shape = (self.height, self.width, self.buffer_size)
        self.num_actions = 3
        self.discount_factor = 0.99
        self.batch_size = 32
        self.training_configs = {
            'learning_rate': 1e-3,
            'momentum': 0.9,
            'metrics': ['accuracy'],
            'loss': 'mean_squared_error'
        }
        self.model = self.build_model()
        self.compile_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape=self.input_shape, filters=32, kernel_size=(8,8), strides=(4,4), padding='valid', activation=relu, kernel_initializer='random_uniform'))
        model.add(Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding='valid', activation=relu, kernel_initializer='random_uniform'))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation=relu, kernel_initializer='random_uniform'))
        model.add(Flatten())
        model.add(Dense(512, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dense(self.num_actions, kernel_initializer='random_uniform'))
        return model
    
    def get_action(self, environment):
        expanded_environment = np.expand_dims(environment, axis=0)
        result = self.model.predict(expanded_environment, batch_size=1)
        return np.argmax(result, axis=1)[0]
        
    def get_time_to_execute_action(self):
        return self.time_to_execute_action

    def _get_targets(self, environment_prevs, actions, rewards, environment_nexts, crasheds):
        q_values = self.model.predict_on_batch(environment_prevs)
        max_q_value_next = np.amax(self.model.predict_on_batch(environment_nexts), axis=1)
        max_q_value_next[crasheds] = 0
        q_values[np.arange(q_values.shape[0]), actions] = rewards + self.discount_factor * max_q_value_next
        return q_values

    def compile_model(self):
        # optimizer in construct, otherwise running stats will be reset!
        self.optimizer = SGD(lr=self.training_configs['learning_rate'], momentum=self.training_configs['momentum'])
        self.model.compile(optimizer=self.optimizer, loss=self.training_configs['loss'], metrics=self.training_configs['metrics'])

    def train(self, environment_prevs, actions, rewards, environment_nexts, crasheds):
        """
        Train model on given data.
        Data might not be able to fit into one batch.
        """
        assert environment_nexts.shape[0] == actions.shape[0] == rewards.shape[0] == environment_nexts.shape[0] == self.batch_size, 'all types of data needed for training should have same length'

        x = environment_prevs
        y = self._get_targets(environment_prevs, actions, rewards, environment_nexts, crasheds)
        self.model.train_on_batch(x, y)

    def save_trained_weights(self):
        pass
