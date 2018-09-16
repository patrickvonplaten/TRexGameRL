from tensorflow.python.keras.optimizers import SGD
import numpy as np
import ipdb


class TFRexModel(object):
    def __init__(self, config, network):
        self.weights = None
        self.time_to_execute_action = config['time_to_execute_action']
        self.num_actions = config['num_actions']
        self.discount_factor = config['discount_factor']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.momentum = config['momentum']
        self.metrics = config['metrics']
        self.loss = config['loss']
        self.model = network
        self.compile_model()
        self.train_epoch_counter = 1

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
        self.optimizer = SGD(lr=self.learning_rate, momentum=self.momentum)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def train(self, environment_prevs, actions, rewards, environment_nexts, crasheds):
        """
        Train model on given data.
        Data might not be able to fit into one batch.
        """
        assert environment_nexts.shape[0] == actions.shape[0] == rewards.shape[0] == environment_nexts.shape[0] == self.batch_size, 'all types of data needed for training should have same length'

        x = environment_prevs
        y = self._get_targets(environment_prevs, actions, rewards, environment_nexts, crasheds)
        log = self.model.train_on_batch(x, y)
        print("Train Epoch: {} | Loss: {} | Accuracy: {}".format(self.train_epoch_counter, log[0], log[1]))

    def save_trained_weights(self):
        pass
