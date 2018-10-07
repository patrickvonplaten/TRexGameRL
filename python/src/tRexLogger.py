import time
import datetime
import ipdb


class Logger(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.file = None
        self.open()
        self.close()

    def create_log(self, parameters):
        log = ''
        for param in parameters.keys():
            log += str(param) + ': ' + str(parameters[param]) + ' | '
        return log

    def format_running_times(self, start_time, epoch):
        time_elapsed = time.time() - start_time
        avg_time_per_epoch = time_elapsed/(epoch+1)
        time_elapsed_formatted = datetime.timedelta(seconds=int(time_elapsed))
        avg_time_per_epoch_formatted = datetime.timedelta(seconds=int(avg_time_per_epoch))
        return time_elapsed_formatted, avg_time_per_epoch_formatted

    def format_loss(self, loss):
        return self.format_float(loss[0]) if not not loss else 'No train'

    def format_float(self, float_value):
        return '{:.4f}'.format(float_value)

    def format_epoch(self, epoch, epoch_to_train):
        return '{}/{}'.format(epoch, epoch_to_train)

    def close(self):
        self.file.close()

    def open(self):
        self.file = open(self.file_name, 'a')

    def log_parameter(self, epoch, epoch_to_train, start_time, score, loss, epsilon, reward_sum, avg_control_q):
        time_elapsed, avg_time_per_epoch = self.format_running_times(start_time, epoch)
        log = self.create_log({
            'epoch': self.format_epoch(epoch, epoch_to_train),
            'score': score,
            'loss': self.format_loss(loss),
            'reward': reward_sum,
            'avg_q': self.format_float(avg_control_q),
            'epsilon': round(epsilon, 2),
            'time elapsed': time_elapsed,
            'avg time per epoch': avg_time_per_epoch,
        })
        print(log)
        self.open()
        self.file.write(log + '\n')
        self.close()
