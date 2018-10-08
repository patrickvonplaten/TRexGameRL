import time
import datetime
import ipdb
import os


class Logger(object):

    def __init__(self, path_to_log):
        self.path_to_log = path_to_log
        self.path_to_file = os.path.join(self.path_to_log, 'train_log.txt')
        self.file = None

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
        self.file = open(self.path_to_file, 'a')

    def close(self):
        self.file.close()

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
