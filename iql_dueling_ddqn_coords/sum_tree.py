import numpy as np


class SumTree:
    """
    Binary tree data structure where each leaf stores the priority of a transition
    and each internal node stores the sum of its children.
    The root contains the total sum of all priorities, used for proportional sampling.
    
    Memory layout (flat array of size 2*capacity - 1):
        - indices [0, capacity-2]: internal nodes
        - indices [capacity-1, 2*capacity-2]: leaves (priorities)
    """

    def __init__(self, capacity):
        """
        :param capacity: maximum number of transitions to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)      # internal nodes + leaves
        self.data = np.zeros(capacity, dtype=object) # transitions
        self.write = 0                               # circular pointer
        self.n_entries = 0

    def _propagate(self, idx, change):
        """
        Propagates a priority change up the tree from a leaf to the root,
        updating all ancestor nodes.

        :param idx: index of the node that was updated
        :param change: difference between new and old priority
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        Recursively traverses the tree to find the leaf corresponding to value s.

        :param idx: current node index (start from 0, i.e. root)
        :param s: value to retrieve, in range [0, total()]
        :return: index of the selected leaf
        """
        left  = 2 * idx + 1
        right = 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """
        Returns the total sum of all priorities (value of the root node).
        Used to normalize probabilities when sampling.
        """
        return self.tree[0]

    def add(self, priority, data):
        """
        Adds a new transition with the given priority.
        If the buffer is full, overwrites the oldest entry (circular buffer).

        :param priority: priority of the transition (typically the TD-error)
        :param data: transition tuple (obs, actions, new_obs, rewards, terminated)
        """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        """
        Updates the priority of an existing leaf and propagates the change upward.
        Called after recomputing the TD-error for a sampled transition.

        :param idx: leaf index in the tree array
        :param priority: new priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """
        Samples a transition by traversing the tree with value s.
        Returns the leaf whose priority range contains s.

        :param s: random value in [0, total()]
        :return: tuple (tree_idx, priority, transition)
                 - tree_idx: index in the tree array (needed for update())
                 - priority: priority of the selected leaf
                 - transition: the stored (obs, actions, new_obs, rewards, terminated) tuple
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self):
        return self.n_entries