#!/usr/bin/env python
import sys
import os

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_TREX_MODULES = CUR_PATH + '/../src'
sys.path.insert(0, PATH_TO_TREX_MODULES)

from tRexMemory import Memory  # noqa: E402
from tRexLogger import Logger  # noqa: E402
import numpy as np  # noqa: E402
import random  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402


CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_MODELS = os.path.join(CUR_PATH, './modelsTest')
PATH_TO_LOG = os.path.join(CUR_PATH, './logTest')


def test_tRex_memory():
    memory_config = {
        'memory_size': 20,
        'priority_beta': 0.4,
        'priority_beta_decay_period': 100,
        'batch_size': 32,
        'clipped_max_priority_score': 2,
        'priority_epsilon': 0.01,
        'priority_alpha': 0.6
    }
    memory = Memory(memory_config)

    assert memory.sum_tree.num_leaf_nodes is 20
    assert memory.sum_tree.num_sum_nodes is 19

    def build_training_samples(num_samples=20):
        environment_prev = np.random.rand(num_samples, 80, 80, 4)
        environment_next = np.random.rand(num_samples, 80, 80, 4)
        crashed = np.random.randint(2, size=num_samples)
        action = np.random.randint(2, size=num_samples)
        reward = np.random.rand(num_samples)
        return environment_prev, action, reward, environment_next, crashed

    def test_add():
        memory = Memory(memory_config)
        environment_prev, action, reward, environment_next, crashed = build_training_samples()

        for i in range(20):
            assert memory.pos is i
            memory.add((environment_prev[i], action[i], reward[i], environment_next[i], crashed[i]))
        assert memory.pos is 0
        np.testing.assert_almost_equal(memory.storage[3][0], environment_prev[3], 5)
        np.testing.assert_almost_equal(memory.storage[3][0], environment_prev[3], 5)
        np.testing.assert_almost_equal(memory.storage[3][4], crashed[3], 5)
        np.testing.assert_almost_equal(memory.storage[18][1], action[18], 5)
        np.testing.assert_almost_equal(memory.storage[18][2], reward[18], 5)
        np.testing.assert_almost_equal(memory.storage[10][3], environment_next[10], 5)
        for i in range(20):
            np.testing.assert_almost_equal(memory.sum_tree.tree[memory.sum_tree.get_leaf_index_from_storage_index(i)], (2 + 0.01)**0.6, 5)

    def test_get_index():
        memory = Memory(memory_config)

        assert memory.sum_tree.get_leaf_index_from_storage_index(10) is 29
        assert memory.sum_tree.get_storage_index_from_leaf_index(24) is 5
        assert memory.sum_tree.get_storage_index_from_leaf_index(memory.sum_tree.get_leaf_index_from_storage_index(10)) is 10

    def test_get_priority_score_final():
        memory = Memory(memory_config)

        np.testing.assert_almost_equal(memory.sum_tree.get_priority_score_final(2), (2 + 0.01)**0.6, 5)

    def test_sum_tree_total_priority_sum():
        memory = Memory(memory_config)
        values = [random.random() for x in range(20)]
        indexes = list(range(20))
        priority_scores_final = [memory.sum_tree.get_priority_score_final(x) for x in values]
        for i in range(20):
            memory.sum_tree.add(indexes[i], values[i])

        np.testing.assert_almost_equal(memory.sum_tree.get_total_priority_sum(), sum(priority_scores_final), 5)

    def test_build_prob_intervall():
        memory = Memory(memory_config)
        memory.sum_tree.tree[0] = 100
        size = 10
        prob_interval = memory.build_prob_interval(size)

        assert len(prob_interval) is size
        for i in range(size-1):
            assert prob_interval[i] < prob_interval[i+1]
        assert prob_interval[size - 1] < memory.sum_tree.get_total_priority_sum()

    def test_get_weight_for_sample():
        memory = Memory(memory_config)
        priority_prob = random.random()
        epoch = 80

        np.testing.assert_almost_equal((32 * priority_prob)**(
            -memory.get_priority_beta(epoch, 100, 0.4)), memory.get_weight_for_sample(priority_prob, epoch), 5)

    def test_get_leaf_index_from_value():
        memory = Memory(memory_config)
        value = 5
        memory.sum_tree.add(0, value)
        memory.sum_tree.add(6, value)
        memory.sum_tree.add(10, value)
        priority_score_final = memory.sum_tree.get_priority_score_final(value)
        margin = priority_score_final/float(2)

        assert memory.sum_tree.get_storage_index_from_value(margin) is 0
        assert memory.sum_tree.get_storage_index_from_value(priority_score_final + margin) is 6
        assert memory.sum_tree.get_storage_index_from_value(2*priority_score_final + margin) is 10

    def test_update():
        memory = Memory(memory_config)
        values = [random.random() for x in range(20)]
        indexes = list(range(20))
        for i in range(20):
            memory.sum_tree.add(indexes[i], values[i])
        total_priority_sum_prev = memory.sum_tree.get_total_priority_sum()
        priority_value_at_storage_5 = memory.sum_tree.tree[memory.sum_tree.get_leaf_index_from_storage_index(5)]
        loss = random.random()
        priority_score_final = memory.sum_tree.get_priority_score_final(loss)
        memory.sum_tree.update(5, loss)
        total_priority_sum_next = memory.sum_tree.get_total_priority_sum()
        priority_prob = priority_score_final/float(total_priority_sum_next)

        np.testing.assert_almost_equal(memory.sum_tree.get_priority_prob_from_storage_index(5), priority_prob, 5)
        np.testing.assert_almost_equal(total_priority_sum_next - total_priority_sum_prev, priority_score_final - priority_value_at_storage_5)

    test_add()
    test_get_index()
    test_get_priority_score_final()
    test_sum_tree_total_priority_sum()
    test_build_prob_intervall()
    test_get_weight_for_sample()
    test_get_leaf_index_from_value()
    test_update()


def test_tRex_logger():
    logger_config = {
        'PATH_TO_LOG': PATH_TO_LOG,
        'PATH_TO_MODELS': PATH_TO_MODELS,
        'keep_models': 5,
        'save_model_every_epoch': 1,
        'running_avg': 20,
    }

    logger = Logger(logger_config)
    assert logger.running_avg is 20

    def test_set_running_scores():
        scores = [random.randint(40, 200) for x in range(40)]
        running_sum = 0
        for idx, score in enumerate(scores):
            logger.set_start_epoch(idx)
            logger.set_running_scores(score, idx)
            size_of_running_scores = min(logger.running_avg, idx+1)
            assert len(logger.running_scores) is size_of_running_scores

            running_sum += score

            if(idx > logger.running_avg-1):
                running_sum -= scores[idx - logger.running_avg]
            avg = round(running_sum/size_of_running_scores)

            if(idx is 0):
                std_dev = 0
            else:
                idx_first_elem = max(0, idx-logger.running_avg+1)
                std_dev = round(statistics.stdev(scores[idx_first_elem:size_of_running_scores+idx_first_elem]), 2)

            assert logger.running_scores[0] is score
            assert logger.get_avg_score() == avg
            assert math.isclose(logger.get_std_dev_score(), std_dev, abs_tol=1e-2)

    test_set_running_scores()


if __name__ == "__main__":
    test_tRex_memory()
    test_tRex_logger()
