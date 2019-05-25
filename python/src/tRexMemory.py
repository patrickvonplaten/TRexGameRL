import numpy as np
import ipdb  # noqa F401
import tRexUtils


class Memory(object):

    def __init__(self, config):
        self.sample_size = 5
        self.pos = 0
        self.cur_size = 0
        self.batch_size = config['batch_size']
        self.size = config['memory_size']
        self.priority_beta = config['priority_beta']
        self.priority_beta_decay_period = config['priority_beta_decay_period']
        self.get_priority_beta = getattr(tRexUtils, 'linearly_decaying_beta')
        self.storage = np.ndarray((self.size, self.sample_size), dtype=object)
        self.sample_weights = np.zeros(self.batch_size)
        self.storage_indexes = np.zeros(self.batch_size, dtype=np.int)
        self.sum_tree = PrioritySumTree(self.size, config)

    def add(self, training_sample):
        self.storage[self.pos] = training_sample
        self.sum_tree.add(self.pos)
        self.pos = (self.pos + 1) % self.size
        self.cur_size = min(self.cur_size + 1, self.size)

    def get_batch_size(self):
        return self.batch_size

    def sample(self, epoch):
        # build a list of uniformly sampled values restricted to be in ascending order
        # and to be on average equally distanced to each other ranging from 0 to 1
        prob_interval = self.build_prob_interval(self.batch_size)
        for batch_index in range(self.batch_size):
            sampled_value = prob_interval[batch_index]
            # sample the index of an image sample according to the sampled value
            storage_index = self.sum_tree.sample_storage_index(sampled_value)
            self.storage_indexes[batch_index] = storage_index
            priority_prob = self.sum_tree.get_priority_prob(storage_index)
            self.sample_weights[batch_index] = self.get_weight(priority_prob, epoch)
            processed_sampled_weights = self.sample_weights/np.max(self.sample_weights)
        return self.storage[self.storage_indexes], processed_sampled_weights

    def get_weight(self, priority_prob, epoch):
        priority_beta_exponent = self.get_priority_beta(epoch, self.priority_beta_decay_period, self.priority_beta)
        weight = (self.batch_size * priority_prob)**(-priority_beta_exponent)
        return weight

    def update(self, losses):
        for batch_index in range(self.batch_size):
            self.sum_tree.update(self.storage_indexes[batch_index], losses[batch_index])

    def build_prob_interval(self, num_samples):
        total_priority_sum = self.sum_tree.get_total_priority_sum()
        interval_length = total_priority_sum / float(num_samples)
        return [np.random.uniform(low=x*interval_length, high=(x+1)*interval_length) for x in list(range(num_samples))]


class PrioritySumTree(object):

    def __init__(self, capacity, config):
        self.num_leaf_nodes = capacity
        self.num_sum_nodes = self.num_leaf_nodes - 1
        self.num_tree_nodes = self.num_leaf_nodes + self.num_sum_nodes
        self.tree = np.zeros(self.num_tree_nodes)
        self.max_clipped_priority_score = config['clipped_max_priority_score']
        self.priority_epsilon = config['priority_epsilon']
        self.priority_alpha = config['priority_alpha']
        self.max_priority_score = self.max_clipped_priority_score

    def add(self, storage_index, priority_score=None):
        priority_score = self.max_priority_score if priority_score is None else priority_score
        self.update(storage_index, priority_score)

    def get_leaf_index(self, storage_index):
        return storage_index + self.num_sum_nodes

    def get_storage_index(self, leaf_index):
        return leaf_index - self.num_sum_nodes

    def update(self, storage_index, loss_of_sample):
        priority_score_final = self.get_priority_score_final(loss_of_sample)
        leaf_index = self.get_leaf_index(storage_index)
        change = priority_score_final - self.tree[leaf_index]
        self.tree[leaf_index] = priority_score_final
        parent_index = leaf_index
        # priority score change trickles down the tree
        while parent_index != 0:
            parent_index = (parent_index - 1) // 2
            self.tree[parent_index] += change

    def get_priority_score_final(self, loss):
        # priority score is calculated according to the loss
        priority_score_final = (min(self.max_priority_score, loss) + self.priority_epsilon)**self.priority_alpha
        return priority_score_final

    def sample_storage_index(self, value, parent_index=0):
        left_child_index = 2 * parent_index + 1
        right_child_index = left_child_index + 1
        if left_child_index >= len(self.tree):
            leaf_index = parent_index
            return self.get_storage_index(leaf_index)
        value_left_child = self.tree[left_child_index]
        if value < value_left_child:
            return self.sample_storage_index(value, left_child_index)
        else:
            return self.sample_storage_index(value - value_left_child, right_child_index)

    def get_priority_prob(self, storage_index):
        leaf_index = self.get_leaf_index(storage_index)
        leaf_priority_prob = self.tree[leaf_index]/float(self.get_total_priority_sum())
        return leaf_priority_prob

    def get_total_priority_sum(self):
        return self.tree[0]
