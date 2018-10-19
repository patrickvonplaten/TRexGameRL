import numpy as np
import ipdb
import tRexUtils


class Memory(object):

    def __init__(self, config):
        self.sample_size = 5
        self.size = config['memory_size']
        self.priority_beta = config['priority_beta']
        self.priority_beta_decay_period = config['priority_beta_decay_period']
        self.get_priority_beta = getattr(tRexUtils, 'linearly_decaying_beta')
        self.storage = np.ndarray((self.size, self.sample_size), dtype=object)
        self.pos = 0
        self.cur_size = 0
        self.sum_tree = PrioritySumTree(self.size, config)
        self.batch_size = config['batch_size']
        self.sample_weights = np.zeros(self.batch_size)
        self.storage_indexes = np.zeros(self.batch_size, dtype=np.int)

    def add(self, training_sample):
        self.storage[self.pos] = training_sample
        self.sum_tree.add(self.pos)
        self.pos = (self.pos + 1) % self.size
        self.cur_size = min(self.cur_size + 1, self.size)

    def sample(self, epoch):
        prob_interval = self.build_prob_interval(self.batch_size)

        for batch_idx in range(self.batch_size):
            self.storage_indexes[batch_idx] = self.sum_tree.get_storage_index_from_value(prob_interval[batch_idx])
            priority_prob = self.sum_tree.get_priority_prob_from_storage_index(self.storage_indexes[batch_idx])
            self.sample_weights[batch_idx] = self.get_weight_for_sample(priority_prob, epoch)

        return self.storage[self.storage_indexes], self.sample_weights/np.max(self.sample_weights)

    def get_weight_for_sample(self, priority_prob, epoch):
        return (self.batch_size * priority_prob)**(
                    -self.get_priority_beta(epoch, self.priority_beta_decay_period, self.priority_beta))

    def update(self, losses):
        [self.sum_tree.update(self.storage_indexes[x], losses[x]) for x in range(self.batch_size)]

    def build_prob_interval(self, num_samples):
        total_priority_sum = self.sum_tree.get_total_priority_sum()
        interval_length = total_priority_sum / float(num_samples)
        return [np.random.uniform(low=x*interval_length, high=(x+1)*interval_length) for x in list(range(num_samples))]


class PrioritySumTree(object):
    """
    This SumTree code is taken and modified from: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb
    """
    def __init__(self, capacity, config):
        self.num_leaf_nodes = capacity
        self.num_sum_nodes = self.num_leaf_nodes - 1
        self.tree = np.zeros(self.num_leaf_nodes + self.num_sum_nodes)
        self.max_clipped_priority_score = config['clipped_max_priority_score']
        self.priority_epsilon = config['priority_epsilon']
        self.priority_alpha = config['priority_alpha']
        self.max_priority_score = self.max_clipped_priority_score

    def add(self, storage_index, priority_score=None):
        priority_score = self.max_priority_score if priority_score is None else priority_score
        self.update(storage_index, priority_score)

    def get_leaf_index_from_storage_index(self, storage_index):
        return storage_index + self.num_sum_nodes

    def get_storage_index_from_leaf_index(self, leaf_index):
        return leaf_index - self.num_sum_nodes

    def update(self, storage_index, loss_of_sample):
        priority_score_final = self.get_priority_score_final(loss_of_sample)
        leaf_index = self.get_leaf_index_from_storage_index(storage_index)
        change = priority_score_final - self.tree[leaf_index]
        self.tree[leaf_index] = priority_score_final
        parent_index = leaf_index

        while parent_index != 0:
            parent_index = (parent_index - 1) // 2
            self.tree[parent_index] += change

    def get_priority_score_final(self, loss):
        return (min(self.max_priority_score, loss) + self.priority_epsilon)**self.priority_alpha

    def get_storage_index_from_value(self, value, parent_index=0):
        left_child_index = 2 * parent_index + 1
        right_child_index = left_child_index + 1

        if left_child_index >= len(self.tree):
            leaf_index = parent_index
            return self.get_storage_index_from_leaf_index(leaf_index)

        if value < self.tree[left_child_index]:
            return self.get_storage_index_from_value(value, left_child_index)
        else:
            return self.get_storage_index_from_value(value - self.tree[left_child_index], right_child_index)

    def get_priority_prob_from_storage_index(self, storage_index):
        return self.tree[self.get_leaf_index_from_storage_index(storage_index)]/float(self.get_total_priority_sum())

    def get_total_priority_sum(self):
        return self.tree[0]
