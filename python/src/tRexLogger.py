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
        self.path_to_models = config['PATH_TO_MODELS']
        if not os.path.isdir(self.path_to_models):
            os.mkdir(self.path_to_models)
        self.save_model_every_epoch = config['save_model_every_epoch']
        self.keep_models = config['keep_models']
        self.file = None
        self.saved_models = []
        self.file_name_template = 'network.epoch.{:07}.h5'

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
        epochs_left = epochs_to_train - epoch
        return average_time_per_epoch*epochs_left

    def format_time_left(self, start_time, epoch, epochs_to_train, start_epoch):
        time_elapsed = self.get_time_elapsed(start_time)
        avg_time_per_epoch = self.get_avg_time_per_epoch(time_elapsed, epoch - start_epoch)
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

    def save_model(self, epoch, model):
        if epoch % self.save_model_every_epoch is 0:
            model_file_path = self.get_file_path(epoch)
            model.save(model_file_path)
            print('Saved model to {}'.format(model_file_path))
            self.saved_models = [model_file_path] + self.saved_models
            if len(self.saved_models) > self.keep_models:
                oldest = self.saved_models.pop()
                os.remove(oldest)
                print('Deleted {}'.format(oldest))

    def get_file_path(self, epoch):
        return os.path.join(self.path_to_models, self.file_name_template.format(int(epoch)))

    def get_epoch_of_last_saved_model(self):
        ipdb.set_trace()
        list_of_model_files = self.get_list_of_model_files()
        latest_model_path = max(list_of_model_files, key=os.path.getctime)
        return self.extract_epoch_from_model_path(latest_model_path)

    def get_list_of_model_files(self):
        return glob.glob(self.path_to_models + '/*')

    def extract_epoch_from_model_path(self, model_path):
        model_name = model_path.split('/')[-1]
        return int(model_name.split('.')[-2])

    def log_parameter(self, epoch, epochs_to_train, start_time, score, loss,
            epsilon, reward_sum, avg_control_q, start_epoch):
        log = self.create_log({
            'epoch': self.format_epoch(epoch, epochs_to_train),
            'score': score,
            'loss': self.format_loss(loss),
            'reward': reward_sum,
            'avg_q': self.format_float(avg_control_q),
            'epsilon': round(epsilon, 2),
            'time elapsed': self.format_time_elapsed(start_time),
            'avg time per epoch': self.format_avg_time_per_epoch(start_time, epoch-start_epoch),
            'time left': self.format_time_left(start_time, epoch, epochs_to_train, start_epoch),
        })
        print(log)
        self.open()
        self.file.write(log + '\n')
        self.close()
