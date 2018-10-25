import time
import datetime
import ipdb  # noqa: F401
import os
import glob
import collections
import statistics


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
        self.running_avg = config['running_avg']
        self.running_scores = collections.deque(maxlen=self.running_avg)
        self.epoch = None
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
        return self.format_float(loss) if not not loss else 'No train'

    def format_float(self, float_value):
        return '{:.4f}'.format(float_value)

    def format_epoch(self, epoch, epochs_to_train):
        return '{}/{}'.format(epoch+1, epochs_to_train)

    def format_score_avg(self):
        return '{}'.format(self.get_avg_score())

    def format_score_std_dev(self):
        return '{}'.format(self.get_std_dev_score())

    def format_time(self):
        return datetime.datetime.now().strftime('%H:%M:%S')

    def get_avg_score(self):
        return round(sum(self.running_scores)/float(len(self.running_scores)))

    def get_std_dev_score(self):
        if(len(self.running_scores) is 1):
            return 0
        return round(statistics.stdev(self.running_scores), 2)

    def set_running_scores(self, score, epoch):
        assert self.epoch is epoch, 'set_running_scores should be called only once per epoch'
        self.epoch += 1
        self.running_scores.appendleft(score)

    def set_start_epoch(self, epoch):
        if(self.epoch is None):
            self.epoch = epoch

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
        list_of_model_files = self.get_list_of_model_files()
        latest_model_path = max(list_of_model_files, key=os.path.getctime)
        return self.extract_epoch_from_model_path(latest_model_path)

    def get_list_of_model_files(self):
        return glob.glob(self.path_to_models + '/*')

    def extract_epoch_from_model_path(self, model_path):
        model_name = model_path.split('/')[-1]
        return int(model_name.split('.')[-2])

    def log_parameter(self, epoch, epochs_to_train, start_time, score, loss,
            epsilon, reward, avg_control_q, start_epoch):  # noqa: E128
        self.set_start_epoch(start_epoch)
        self.set_running_scores(score, epoch)
        log = self.create_log({
            'ep': self.format_epoch(epoch, epochs_to_train),
            'score': score,
            'score_avg': self.format_score_avg(),
            'score_dev': self.format_score_std_dev(),
            'loss': self.format_loss(loss),
            'reward': reward,
            'avg_q': self.format_float(avg_control_q),
            'epsilon': round(epsilon, 2),
            'time_elap': self.format_time_elapsed(start_time),
            'avg_ep_time': self.format_avg_time_per_epoch(start_time, epoch-start_epoch),
            'time_left': self.format_time_left(start_time, epoch, epochs_to_train, start_epoch),
            'time': self.format_time()
        })
        print(log)
        self.open()
        self.file.write(log + '\n')
        self.close()
