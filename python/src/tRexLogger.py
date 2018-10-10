import time
import datetime
import ipdb
import os
import glob


class Logger(object):

    def __init__(self, config):
        self.path_to_log = config['PATH_TO_LOG']
        if not os.path.isdir(self.path_to_log):
            os.mkdir(self.path_to_log)
        self.path_to_file = os.path.join(self.path_to_log, 'train_log.txt')
        self.path_to_weights = config['PATH_TO_WEIGHTS']
        if not os.path.isdir(self.path_to_weights):
            os.mkdir(self.path_to_weights)

        self.save_weights_every_epoch = config['save_weights_every_epoch']
        self.keep_weights = config['keep_weights']
        self.file = None
        self.file_name_template = 'network.epoch.{:07}.h5'
        self.saved_weights = []

    def create_log(self, parameters):
        log = ''
        for param in parameters.keys():
            log += str(param) + ': ' + str(parameters[param]) + ' | '
        return log

    def get_avg_time_per_epoch(self, time_elapsed, epoch):
        return time_elapsed/(epoch+1)

    def format_avg_time_per_epoch(self, start_time, epoch):
        time_elapsed = self.get_time_elapsed(start_time)
        avg_time_per_epoch = self.get_avg_time_per_epoch(time_elapsed, epoch)
        return datetime.timedelta(seconds=int(avg_time_per_epoch))

    def get_time_elapsed(self, start_time):
        return time.time() - start_time

    def format_time_elapsed(self, start_time):
        time_elapsed = self.get_time_elapsed(start_time)
        return datetime.timedelta(seconds=int(time_elapsed))

    def get_time_left(self, average_time_per_epoch, epoch, epochs_to_train):
        epoch_left = epochs_to_train - epoch
        return average_time_per_epoch*epoch_left

    def format_time_left(self, start_time, epoch, epochs_to_train):
        time_elapsed = self.get_time_elapsed(start_time)
        avg_time_per_epoch = self.get_avg_time_per_epoch(time_elapsed, epoch)
        time_left = self.get_time_left(avg_time_per_epoch, epoch, epochs_to_train)
        return datetime.timedelta(seconds=int(time_left))

    def format_loss(self, loss):
        return self.format_float(loss[0]) if not not loss else 'No train'

    def format_float(self, float_value):
        return '{:.4f}'.format(float_value)

    def format_epoch(self, epoch, epochs_to_train):
        return '{}/{}'.format(epoch, epochs_to_train)

    def open(self):
        self.file = open(self.path_to_file, 'w')

    def close(self):
        self.file.close()

    def save_weights(self, epoch, model):
        """
        Keeps the last 20 models of CURRENT run, older are deleted.
        Args:
            model (keras model): The model whose weights are saved.
        """
        if epoch % self.save_weights_every_epoch is 0:
            weight_file_path = self.get_file_path(epoch, self.path_to_weights)
            model.save_weights(weight_file_path)
            print('Saved weights to {}'.format(weight_file_path))
            self.saved_weights = [weight_file_path] + self.saved_weights
            if len(self.saved_weights) > self.keep_weights:
                oldest = self.saved_weights.pop()
                os.remove(oldest)
                print('Deleted {}'.format(oldest))

    def get_last_file_path(self, path_to_weights):
        return os.path.join(path_to_weights, max(glob.glob(path_to_weights)))

    def get_file_path(self, epoch, path_to_weights):
        return os.path.join(path_to_weights, self.file_name_template.format(int(epoch)))

    def log_parameter(self, epoch, epochs_to_train, start_time, score, loss, epsilon, reward_sum, avg_control_q):
        log = self.create_log({
            'epoch': self.format_epoch(epoch, epochs_to_train),
            'score': score,
            'loss': self.format_loss(loss),
            'reward': reward_sum,
            'avg_q': self.format_float(avg_control_q),
            'epsilon': round(epsilon, 2),
            'time elapsed': self.format_time_elapsed(start_time),
            'avg time per epoch': self.format_avg_time_per_epoch(start_time, epoch),
            'time left': self.format_time_left(start_time, epoch, epochs_to_train),
        })
        print(log)
        self.open()
        self.file.write(log + '\n')
        self.close()
