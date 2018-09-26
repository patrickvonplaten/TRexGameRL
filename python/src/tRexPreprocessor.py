from collections import deque
import numpy as np
import cv2
import ipdb


class Prepocessor(object):
    def __init__(self, vertical_crop_intervall, horizontal_crop_intervall, buffer_size, resize):
        self.vertical_crop_start = vertical_crop_intervall[0]
        self.vertical_crop_end = vertical_crop_intervall[1]
        self.vertical_crop_length = self.vertical_crop_start - self.vertical_crop_end
        self.horizontal_crop_start = horizontal_crop_intervall[0]
        self.horizontal_crop_end = horizontal_crop_intervall[1]
        self.horizontal_crop_length = self.horizontal_crop_start - self.horizontal_crop_end
        self.buffer_size = buffer_size
        self.resize = resize
        # use [np.zeros() for i in range(..)] to have independent arrays
        self.image_processed_buffer = deque([np.zeros((self.resize, self.resize), dtype=np.uint8) for i in range(self.buffer_size)], maxlen=self.buffer_size)
        self.environment_processed_shape = (self.resize, self.resize, self.buffer_size)

    def _process(self, image):
        image_processed = self.crop_image(image)
        image_processed = cv2.resize(image_processed, (self.resize, self.resize))
        image_processed = image_processed / 128 - 1.0
        return image_processed

    def process(self, image):
        image_processed = self._process(image)
        self.image_processed_buffer.pop()
        self.image_processed_buffer.appendleft(image_processed)

        # np.asarray performs copy
        environment = np.array(self.image_processed_buffer)
        return environment.reshape(self.environment_processed_shape)

    def crop_image(self, image):
        return image[self.vertical_crop_start:self.vertical_crop_end, self.horizontal_crop_start:self.horizontal_crop_end]
