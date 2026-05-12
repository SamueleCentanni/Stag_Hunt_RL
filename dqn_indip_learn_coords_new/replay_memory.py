# Prioritized Experience Replay (PER)
# Idea: by treating all samples the same, we are ignoring a simple intuition from the real world, that is, we can potentially learn more from the experiences for which the outcomes differ more from our expectations. 
# To leverage this fact, prioritized experience replat samples experiences with probability p_t proportional to the absolute difference between the target and estimation values for this experience which is known as TD error




"""
Cos'è la PER
Con il replay buffer uniforme (quello che hai già), ogni transizione ha la stessa probabilità di essere campionata. Ma alcune transizioni sono più "sorprendenti" di altre — quelle dove l'errore TD è alto, cioè dove la rete si sbagliava di più.
La Prioritized Experience Replay campiona le transizioni con probabilità proporzionale al loro TD-error:
P(i) = p_i^α / Σ p_k^α
Dove p_i è la priorità della transizione i (= TD-error), e α controlla quanto pesare le priorità (α=0 → uniforme, α=1 → completamente prioritizzato).

Il problema: bias
Campionare non uniformemente introduce un bias — le transizioni ad alta priorità vengono viste troppo spesso. Si corregge con Importance Sampling weights:
w_i = (1 / N · 1/P(i))^β
β parte basso (0.4) e cresce fino a 1 durante il training — all'inizio il bias non importa molto, verso la fine del training deve essere corretto completamente.
Beta annealing corrects the bias introduced by non-uniform sampling.

In order to speed up the PER buffer, we need to define a sum_tree to efficiently sample accordingly to transitions priorities, where a transition is a sample in the buffer (obs, actions, new_obs, rewards, terminated), and the priority is a number telling us how important is that transition for learning, and it is based on the TD-error = |r + γ * max Q(s', a') - Q(s, a)|
"""

import numpy as np
from sum_tree import SumTree
import random

class PERBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=300000, epsilon=1e-5):
        """
        :param capacity:     max number of transitions
        :param alpha:        how much prioritization (0=uniform, 1=full)
        :param beta_start:   initial Importance Sampling correction (grows to 1.0)
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
        min_prob = np.min(self.tree.tree[-self.tree.capacity:][self.tree.tree[-self.tree.capacity:] > 0])
        min_prob /= self.tree.total()

        # w_i = (1 / N · 1/P(i))^β
        max_weight = (min_prob * len(self.tree)) ** (-self.beta)

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