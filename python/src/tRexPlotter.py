import re
from matplotlib import pyplot as plt
import numpy as np
import os
from argparse import ArgumentParser


class Plotter(object):
    def __init__(self, train_log_file, parameters_to_plot, folder_to_save_plots, title):
        self.train_log_file = train_log_file
        self.parameters_to_plot = parameters_to_plot
        self.folder_to_save_plots = folder_to_save_plots
        self.title = title
        self.content = self.read_in_file()
        self.parameter_dict = self.set_up_parameter_dict(self.parameters_to_plot)

    def read_in_file(self):
        with open(self.train_log_file) as log_file:
            content = log_file.readlines()
        return [x.strip() for x in content]

    def set_up_parameter_dict(self, parameters_to_plot):
        parameter_dict = {}
        for parameter_to_plot in self.parameters_to_plot:
            parameter_dict[parameter_to_plot] = self.extract_parameter_values_from_content(parameter_to_plot)
        return parameter_dict

    def extract_parameter_values_from_content(self, parameter):
        parameter_values = []
        for line in self.content:
            parameter_values.append(self.extract_parameter_value_from_line(parameter, line))
        return [x for x in parameter_values if x is not None]

    def extract_parameter_value_from_line(self, parameter, line):
        value = re.search(parameter + ':(.*)', line).group(1).split('|')[0]
        if(not self.is_float(value)):
            return
        return float(value)

    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def plot_parameters(self):
        for parameter_to_plot in self.parameters_to_plot:
            parameter_values = self.parameter_dict[parameter_to_plot]
            self.plot(parameter_to_plot, parameter_values)

    def plot(self, parameter_name, parameter_values):
        values_array = np.asarray(parameter_values)
        epochs_array = np.arange(values_array.size)

        fig, ax = plt.subplots()
        ax.plot(epochs_array, values_array)
        ax.set(xlabel='epochs', ylabel=parameter_name, title=self.title)
        ax.grid()
        fig.savefig(os.path.join(self.folder_to_save_plots, parameter_name + '_plot_' + self.title + '.png'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--log', help='path to the log file to be plotted')
    parser.add_argument('--save', help='path to folder to save plots')
    parser.add_argument('--title', help='title for all plots', default='trial')
    parser.add_argument('--params', help='all parameters to be plotted', default='loss,score,avg_q')
    args = parser.parse_args()
    parameters_to_plot = args.params.split(',')
    train_log_file = args.log
    folder_to_save_plots = args.save
    title = args.title

    plotter = Plotter(train_log_file, parameters_to_plot, folder_to_save_plots, title)
    plotter.plot_parameters()
