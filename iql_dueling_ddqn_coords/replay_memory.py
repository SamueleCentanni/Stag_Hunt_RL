# Prioritized Experience Replay (PER)
# Idea: by treating all samples the same, we are ignoring a simple intuition from the real world, that is, we can potentially learn more from the experiences 
# for which the outcomes differ more from our expectations. 
# To leverage this fact, prioritized experience replay (PER) samples experiences with probability p_t proportional to the absolute difference between the target
# and estimation values for this experience: this is known as TD error


import numpy as np
from sum_tree import SumTree
import random

class PERBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=200000, epsilon=1e-5):
        """
        :param capacity:     max number of transitions
        :param alpha:        how much prioritization (0=uniform, 1=full)
        :param beta_start:   initial Importance Sampling correction 
        :param beta_frames:  over how many steps beta anneals to 1.0
        :param epsilon:      small constant to avoid zero priority
        """
        self.tree        = SumTree(capacity)
        self.alpha       = alpha
        self.beta        = beta_start
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.epsilon     = epsilon
        self.frame       = 1

    @property
    def max_priority(self):
        """
        Returns current maximum leaf priority. New transitions get this value
        so they are sampled at least once before priority is refined in replay().
        """
        if len(self.tree) == 0:
            return 1.0
        return float(np.max(self.tree.tree[-self.tree.capacity:]))

    def _get_priority(self, error):
        """
        Converts a TD-error into a priority.

        :param error: scalar TD-error of a transition
        :return: priority value = (|error| + epsilon)^alpha
        """
        return (abs(error) + self.epsilon) ** self.alpha
    
    def add(self, priority, transition):
        """
        Adds a new transition to the buffer with the given priority.
        Typically called with max_priority so new transitions are sampled at least once.

        :param priority: priority value (already transformed, not a raw TD-error)
        :param transition: tuple (obs, actions, new_obs, rewards, terminated)
        """
        self.tree.add(priority, transition)

    def sample(self, n):
        """
        Samples n transitions using stratified sampling and computes IS weights.
        Beta is annealed toward 1.0 over beta_frames steps to correct sampling bias.

        :param n: number of transitions to sample
        :return: tuple (batch, indices, weights)
                 - batch:   list of transition tuples
                 - indices: list of tree indices (needed for update())
                 - weights: numpy array of IS weights, normalized to [0, 1]
        """
        batch, indices, weights = [], [], []
        segment = self.tree.total() / n

        # anneal beta toward 1.0
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        # max weight for normalization (corresponds to min priority)
        leaves = self.tree.tree[-self.tree.capacity:]
        # I only want the min leaves value s.t. the value is > 0
        min_prob = np.min(leaves[leaves > 0])
        
        min_prob /= self.tree.total()

        # w_i = (1 / N · 1/P(i))^β
        max_weight = (min_prob * len(self.tree)) ** (-self.beta) # to normalize the weights in [0, 1]

        for i in range(n):
            left = segment * i
            right = segment * (i + 1)
            s = random.uniform(left, right)

            idx, priority, data = self.tree.get(s)
            prob = priority / self.tree.total()
            weight = ((prob * len(self.tree)) ** (-self.beta)) / max_weight

            batch.append(data)
            indices.append(idx)
            weights.append(weight)

        return batch, indices, np.array(weights, dtype=np.float32)

    def update(self, indices, errors):
        """
        Updates the priorities of sampled transitions after computing new TD-errors.

        :param indices: list of tree indices returned by sample()
        :param errors:  list or tensor of new TD-errors for each sampled transition
        """
        for idx, error in zip(indices, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree)