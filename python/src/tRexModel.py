import numpy as np
import ipdb  # noqa: F401
from tensorflow.python.keras.models import clone_model, load_model
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Model  # noqa: E402
from tensorflow.python.keras.layers import Input  # noqa: E402
import tRexNetwork


class TFRexModel(object):
    def __init__(self, config, network, logger, start_epoch=0):
        self.discount_factor = config['discount_factor']
        self.metrics = config['metrics']
        self.loss = config['loss']
        self.num_actions = config['num_actions']
        self.train_model = network
        self.logger = logger
        self.start_epoch = start_epoch
        self.weights = None
        self.optimizer = self.create_optimizer(config)
        self.target_model = clone_model(self.train_model)
        self.compile_train_model()
        self.logger.set_start_epoch(start_epoch)
        self.tensor_board = self.logger.get_tensor_board()
        self.tensor_board.set_model(self.train_model)

    @classmethod
    def restore_from_epoch(cls, epoch, config, logger):
        if epoch < 0:
            epoch = logger.get_epoch_of_last_saved_model()
        path_to_model = logger.get_network_path(epoch)
        print("Restoring from {}".format(path_to_model))
        start_epoch = epoch + 1
        return cls(config, network=load_model(path_to_model), logger=logger, start_epoch=start_epoch)

    @classmethod
    def create_network(cls, config, logger):
        conv_initialization = config['conv_init']
        dense_initialization = config['dense_init']
        network_type = config['network_type']
        num_actions = config['num_actions']
        resize_dim = config['resize_dim']
        num_input_images = config['buffer_size']
        lstm_dropout = config['lstm_dropout']
        input_shape = Input(shape=(resize_dim, resize_dim, num_input_images))
        base_network = getattr(tRexNetwork, 'base_network')(input_shape, conv_initialization, lstm_dropout)
        end_network = getattr(tRexNetwork, network_type)(base_network, dense_initialization, num_actions)
        network = Model(inputs=input_shape, outputs=end_network)
        path_to_weights_to_load = config['path_to_weights_to_load'] if 'path_to_weights_to_load' in config else None
        if(path_to_weights_to_load is not None):
            network.load_weights(path_to_weights_to_load)
            print("Loading weights from {}".format(path_to_weights_to_load))
        return cls(config, network=network, logger=logger)

    def create_optimizer(self, config):
        decay = config['learning_rate_decay'] if 'learning_rate_decay' in config else 0
        optimizer = getattr(optimizers, config['optimizer'])
        return optimizer(lr=config['learning_rate'], decay=decay)

    def get_action(self, environment):
        expanded_environment = np.expand_dims(environment, axis=0)
        result = self.train_model.predict(expanded_environment, batch_size=1)
        return np.argmax(result, axis=1)[0]

    def get_num_actions(self):
        return self.num_actions

    def _get_targets(self, environment_prevs, actions, rewards, environment_nexts, crasheds):
        # calculate q_target values according to deep double Q-learning ( http://proceedings.mlr.press/v48/wangf16.pdf )
        q_values = self.train_model.predict_on_batch(environment_prevs)
        max_actions_next = np.argmax(self.train_model.predict_on_batch(environment_nexts), axis=1)
        num_batch_aranged = np.arange(max_actions_next.shape[0])
        q_values_next = self.target_model.predict_on_batch(environment_nexts)[num_batch_aranged, max_actions_next]
        q_values_next[crasheds] = 0
        q_values[num_batch_aranged, actions] = rewards + self.discount_factor * q_values_next
        return q_values

    def compile_train_model(self):
        # optimizer in construct, otherwise running stats will be reset!
        self.train_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def copy_weights_to_target_model(self):
        self.target_model.set_weights(self.train_model.get_weights())

    def train(self, batch, sample_weights):
        environment_prevs, actions, rewards, environment_nexts, crasheds = self.split_batch_into_parts(batch)
        assert environment_nexts.shape[0] == actions.shape[0] == rewards.shape[0] == environment_nexts.shape[0], 'all types of data needed for training should have same length'
        samples = environment_prevs
        targets = self._get_targets(environment_prevs, actions, rewards, environment_nexts, crasheds)
        losses_per_sample = self.get_abs_losses_per_sample(samples, targets)
        self.train_model.train_on_batch(samples, targets, sample_weight=sample_weights)
        return losses_per_sample

    def get_abs_losses_per_sample(self, samples, targets):
        predictions = self.train_model.predict_on_batch(samples)
        return np.abs(np.sum(predictions - targets, axis=1))

    def split_batch_into_parts(self, batch):
        num_parts = batch.shape[1]
        return (np.stack(batch[:, i]) for i in range(num_parts))
