import numpy as np
import ipdb


class Memory(object):
    """
    Stores states for replay.
    """
    def __init__(self, size, sample_size=5):
        """
        Args:
            size: storage size
            sample_size: Number of elements one samlpe exists of.
        """
        # This should be general.
        self.sample_size = sample_size
        self.storage = np.ndarray((size, sample_size), dtype=object)
        self.size = size
        self.pos = 0
        self.cur_size = 0

    def add(self, training_sample):
        """Adds training sample to the storage.

        Args (training_sample): The training sample to store.
        """
        self.storage[self.pos] = training_sample
        self.pos = (self.pos + 1) % self.size
        self.cur_size = min(self.cur_size + 1, self.size)

    def sample(self, n):
        """Sample n images from memory."""
        num_samples = min(n, self.cur_size)
        idxs = np.random.choice(self.cur_size, num_samples, replace=False)
        return self.storage[idxs]
