import time
import datetime


class Logger(object):

    def __init__(self, file_name):
        self.file = open(file_name, 'w')

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
        return '{:.4f}'.format(loss) if not not loss else 'No train'

    def format_epoch(self, epoch, epoch_to_train):
        return '{}/{}'.format(epoch, epoch_to_train)

    def close(self):
        self.file.close()

    def log_parameter(self, epoch, epoch_to_train, start_time, score, loss, epsilon, random):
        time_elapsed, avg_time_per_epoch = self.format_running_times(start_time, epoch)
        log = self.create_log({
            'epoch': self.format_epoch(epoch, epoch_to_train),
            'score': score,
            'loss': self.format_loss(loss),
            'epsilon': round(epsilon, 2),
            'time elapsed': time_elapsed,
            'avg time per epoch': avg_time_per_epoch,
            'random': random
        })
        print(log)
        self.file.write(log + '\n')
