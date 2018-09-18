import numpy as np
import ipdb


class TFRexModel(object):
    def __init__(self, config, network, optimizer):
        self.weights = None
        self.time_to_execute_action = config['time_to_execute_action']
        self.num_actions = config['num_actions']
        self.discount_factor = config['discount_factor']
        self.batch_size = config['batch_size']
        self.metrics = config['metrics']
        self.loss = config['loss']
        self.model = network
        self.optimizer = optimizer
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
#        ipdb.set_trace()
        max_q_value_next[crasheds] = 0
        q_values[np.arange(q_values.shape[0]), actions] = rewards + self.discount_factor * max_q_value_next
        return q_values

    def compile_model(self):
        # optimizer in construct, otherwise running stats will be reset!
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def train(self, environment_prevs, actions, rewards, environment_nexts, crasheds):
        """
        Train model on given data.
        Data might not be able to fit into one batch.
        """
        assert environment_nexts.shape[0] == actions.shape[0] == rewards.shape[0] == environment_nexts.shape[0] == self.batch_size, 'all types of data needed for training should have same length'

        x = environment_prevs
        # TODO: maybe it is better to do customized cost function
        y = self._get_targets(environment_prevs, actions, rewards, environment_nexts, crasheds)
        return self.model.train_on_batch(x, y)

    def save_trained_weights(self):
        pass
