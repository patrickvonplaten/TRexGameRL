import numpy as np
import ipdb
from tensorflow.python.keras.models import clone_model, load_model
from tensorflow.python.keras.callbacks import TensorBoard


class TFRexModel(object):
<<<<<<< HEAD
    def __init__(self, config, network, optimizer, restore=False):
=======
    def __init__(self, config, network, start_epoch=0):
        self.start_epoch = start_epoch
>>>>>>> 157159a7c20a731b1451a672d031ac24d13c0812
        self.weights = None
        self.time_to_execute_action = config['time_to_execute_action']
        self.num_actions = config['num_actions']
        self.discount_factor = config['discount_factor']
        self.batch_size = config['batch_size']
        self.metrics = config['metrics']
        self.loss = config['loss']
        self.optimizer = config['optimizer']
        self.train_model = network
        self.target_model = clone_model(self.train_model)
        self.compile_train_model()
        self.tensor_board = TensorBoard(log_dir=config['PATH_TO_LOG'], histogram_freq=0, write_graph=True, write_images=True)
        self.tensor_board.set_model(self.train_model)

    @classmethod
    def restore_from_epoch(cls, epoch, config, logger):
        if epoch < 0:
            epoch = logger.get_epoch_of_last_saved_model()
        path_to_model = logger.get_file_path(epoch)
        print("Restoring from {}".format(path_to_model))
        return cls(config, network=load_model(path_to_model), start_epoch=epoch+1)

    def get_action(self, environment):
        expanded_environment = np.expand_dims(environment, axis=0)
        result = self.train_model.predict(expanded_environment, batch_size=1)
        return np.argmax(result, axis=1)[0]

    def get_time_to_execute_action(self):
        return self.time_to_execute_action

    def restore_from_path(self, path_to_model):
        self.target_model = load_model(path_to_model)

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
