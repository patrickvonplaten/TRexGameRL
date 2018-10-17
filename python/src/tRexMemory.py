import numpy as np
import ipdb


class Memory(object):

    def __init__(self, config):
        self.sample_size = 5
        self.size = config['memory_size']
        self.storage = np.ndarray((self.size, self.sample_size), dtype=object)
        self.pos = 0
        self.cur_size = 0
        self.sum_tree = PrioritySumTree(self.size, config)

    def add(self, training_sample):
        self.storage[self.pos] = training_sample
        self.sum_tree.add(self.pos)
        self.pos = (self.pos + 1) % self.size
        self.cur_size = min(self.cur_size + 1, self.size)

    def sample(self, n):
        num_samples = min(n, self.cur_size)
        idxs = np.random.choice(self.cur_size, num_samples, replace=False)
        return self.storage[idxs]


class PrioritySumTree(object):
    """
    This SumTree code is taken and modified from: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb
    """
    def __init__(self, capacity, config):
        self.num_leaf_nodes = capacity
        self.num_sum_nodes = self.num_leaf_nodes - 1
        self.tree = np.zeros(self.num_leaf_nodes + self.num_sum_nodes)
        self.priority_epsilon = config['priority_epsilon']
        self.max_clipped_priority_score = config['clipped_max_priority_score']
        self.max_priority_score = self.max_clipped_priority_score

    def add(self, storage_index, priority_score=None):
        priority_score = self.max_priority_score if priority_score is None else priority_score
        self.update(storage_index, priority_score)

    def get_leaf_index_from_storage_index(self, storage_index):
        return storage_index + self.num_sum_nodes 

    def get_storage_index_from_leaf_index(self, leaf_index):
        return leaf_index - self.num_sum_nodes

    def update(self, storage_index, loss_of_sample):
        priority_score = min(self.max_priority_score, loss_of_sample) + self.priority_epsilon
        leaf_index = self.get_leaf_index_from_storage_index(storage_index)
        change = priority_score - self.tree[leaf_index]
        self.tree[leaf_index] = priority_score
        parent_index = leaf_index

        while parent_index != 0:
            parent_index = (parent_index - 1) // 2
            self.tree[parent_index] += change

    def get_storage_index_from_value(self, value, parent_index=0):
        left_child_index = 2 * parent_index + 1
        right_child_index = left_child_index + 1

        if left_child_index >= len(self.tree):
            leaf_index = parent_index
            return self.get_storage_index_from_leaf_index(leaf_index)

        if value < self.tree[left_child_index]:
            return self.get_leaf_from_sampled_value(value, left_child_index)
        else:
            return self.get_leaf_from_value(value - self.tree[left_child_index], right_child_index)

    def get_priority_value_from_storage_index(self, storage_index):
        return self.tree[self.get_leaf_index_from_storage_index(storage_index)]

    def get_total_priority_sum(self):
        return self.tree[0]
