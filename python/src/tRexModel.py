import numpy as np
import ipdb
from tensorflow.python.keras.models import clone_model


class TFRexModel(object):
    def __init__(self, config, network, optimizer):
        self.weights = None
        self.time_to_execute_action = config['time_to_execute_action']
        self.num_actions = config['num_actions']
        self.discount_factor = config['discount_factor']
        self.batch_size = config['batch_size']
        self.metrics = config['metrics']
        self.loss = config['loss']
        self.train_model = network
        self.target_model = clone_model(self.train_model)
        self.optimizer = optimizer
        self.compile_train_model()

    def get_action(self, environment):
        expanded_environment = np.expand_dims(environment, axis=0)
        result = self.train_model.predict(expanded_environment, batch_size=1)
        return np.argmax(result, axis=1)[0]

    def get_time_to_execute_action(self):
        return self.time_to_execute_action

    def save_weights(self, path_to_save_weights):
        self.train_model.save_weights(path_to_save_weights)
        print('Saved weights to {}'.format(path_to_save_weights))

    def load_weights(self, path_to_load_weights):
        self.train_model.load_weights(path_to_load_weights)

    def _get_targets(self, environment_prevs, actions, rewards, environment_nexts, crasheds):
        q_values = self.predict_on_batch(environment_prevs, self.train_model)
        max_q_value_next = np.amax(self.predict_on_batch(environment_nexts, self.target_model), axis=1)
        max_q_value_next[crasheds] = 0
        q_values[np.arange(q_values.shape[0]), actions] = rewards + self.discount_factor * max_q_value_next
        return q_values

    def predict_on_batch(self, environments, model):
        return model.predict_on_batch(environments)

    def compile_train_model(self):
        # optimizer in construct, otherwise running stats will be reset!
        self.train_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def copy_weights_to_target_model(self):
        self.target_model.set_weights(self.train_model.get_weights())

    def train(self, batch):
        environment_prevs, actions, rewards, environment_nexts, crasheds = self.split_batch_into_parts(batch)
        assert environment_nexts.shape[0] == actions.shape[0] == rewards.shape[0] == environment_nexts.shape[0] == self.batch_size, 'all types of data needed for training should have same length'

        x = environment_prevs
        y = self._get_targets(environment_prevs, actions, rewards, environment_nexts, crasheds)
        return self.train_model.train_on_batch(x, y)

    def split_batch_into_parts(self, batch):
        num_parts = batch.shape[1]
        return (np.stack(batch[:, i]) for i in range(num_parts))

    def save_trained_weights(self):
        pass
