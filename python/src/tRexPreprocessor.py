from collections import deque
import numpy as np
import cv2
import ipdb


class Prepocessor(object):
    def __init__(self, config):
        self.vertical_crop_start = config['vertical_crop_intervall'][0]
        self.vertical_crop_end = config['vertical_crop_intervall'][1]
        self.vertical_crop_length = self.vertical_crop_start - self.vertical_crop_end
        self.horizontal_crop_start = config['horizontal_crop_intervall'][0]
        self.horizontal_crop_end = config['horizontal_crop_intervall'][1]
        self.horizontal_crop_length = self.horizontal_crop_start - self.horizontal_crop_end
        self.buffer_size = config['buffer_size']
        self.resize = config['resize_dim']
        # use [np.zeros() for i in range(..)] to have independent arrays
        self.image_processed_buffer = deque([np.zeros((self.resize, self.resize), dtype=np.uint8) for i in range(self.buffer_size)], maxlen=self.buffer_size)
        self.environment_processed_shape = (self.resize, self.resize, self.buffer_size)
        self.screenshots_for_visual = [] if config['save_screenshots'] is True else None

    def _process(self, image):
        image_processed = self.crop_image(image)
        image_processed = cv2.resize(image_processed, (self.resize, self.resize))
        image_processed = image_processed / 128 - 1.0
        return image_processed

    def reset(self):
        self.image_processed_buffer = deque([np.zeros((self.resize, self.resize), dtype=np.uint8) for i in range(self.buffer_size)], maxlen=self.buffer_size)

    def process(self, image):
        image_processed = self._process(image)
        if(self.screenshots_for_visual is not None):
            self.screenshots_for_visual.append(image_processed)
        self.image_processed_buffer.pop()
        self.image_processed_buffer.appendleft(image_processed)

        # np.array performs copy
        environment = np.array(self.image_processed_buffer)
        return environment.reshape(self.environment_processed_shape)

    def crop_image(self, image):
        return image[self.vertical_crop_start:self.vertical_crop_end, self.horizontal_crop_start:self.horizontal_crop_end]
