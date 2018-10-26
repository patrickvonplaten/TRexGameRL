import numpy as np
import ipdb  # noqa: F401
from tensorflow.python.keras.models import clone_model, load_model


class TFRexModel(object):
    def __init__(self, config, network, logger, start_epoch=0):
        self.start_epoch = start_epoch
        self.weights = None
        self.discount_factor = config['discount_factor']
        self.batch_size = config['batch_size']
        self.metrics = config['metrics']
        self.loss = config['loss']
        self.optimizer = config['optimizer']
        self.train_model = network
        self.target_model = clone_model(self.train_model)
        self.compile_train_model()
        self.logger = logger
        self.logger.set_start_epoch(start_epoch)
        self.tensor_board = self.logger.get_tensor_board()
        self.tensor_board.set_model(self.train_model)

    @classmethod
    def restore_from_epoch(cls, epoch, config, logger):
        if epoch < 0:
            epoch = logger.get_epoch_of_last_saved_model()
        path_to_model = logger.get_file_path(epoch)
        print("Restoring from {}".format(path_to_model))
        start_epoch = epoch + 1
        return cls(config, network=load_model(path_to_model), logger=logger, start_epoch=start_epoch)

    def get_action(self, environment):
        expanded_environment = np.expand_dims(environment, axis=0)
        result = self.train_model.predict(expanded_environment, batch_size=1)
        return np.argmax(result, axis=1)[0]

    def restore_from_path(self, path_to_model):
        self.target_model = load_model(path_to_model)

    def _get_targets(self, environment_prevs, actions, rewards, environment_nexts, crasheds):
        q_values = self.predict_on_batch(environment_prevs, self.train_model)
        max_q_value_next = np.amax(self.predict_on_batch(environment_nexts, self.target_model), axis=1)
        max_q_value_next[crasheds] = 0
        q_values[np.arange(q_values.shape[0]), actions] = rewards + self.discount_factor * max_q_value_next
        return q_values

    def predict_on_batch(self, samples, model):
        return model.predict_on_batch(samples)

    def compile_train_model(self):
        # optimizer in construct, otherwise running stats will be reset!
        self.train_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def copy_weights_to_target_model(self):
        self.target_model.set_weights(self.train_model.get_weights())

    def train(self, batch, sample_weights):
        environment_prevs, actions, rewards, environment_nexts, crasheds = self.split_batch_into_parts(batch)
        assert environment_nexts.shape[0] == actions.shape[0] == rewards.shape[0] == environment_nexts.shape[0] == self.batch_size, 'all types of data needed for training should have same length'

        samples = environment_prevs
        targets = self._get_targets(environment_prevs, actions, rewards, environment_nexts, crasheds)
        losses_per_sample = self.get_abs_losses_per_sample(samples, targets)
        self.train_model.train_on_batch(samples, targets, sample_weight=sample_weights)
        return losses_per_sample

    def get_abs_losses_per_sample(self, samples, targets):
        predictions = self.predict_on_batch(samples, self.train_model)
        return np.abs(np.sum(predictions - targets, axis=1))

    def split_batch_into_parts(self, batch):
        num_parts = batch.shape[1]
        return (np.stack(batch[:, i]) for i in range(num_parts))
