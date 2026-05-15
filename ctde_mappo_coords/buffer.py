import numpy as np
import torch


class RolloutBuffer:
    """
    On-policy rollout buffer for single-env CTDE MAPPO.

    Stores the team reward (sum of per-agent rewards) and per-agent
    log-probabilities together with a single joint-state value V(s) per
    timestep. Generalized Advantage Estimation is computed on the team
    return; advantages are broadcast across agents only inside the actor
    loss.

    :param size: Number of environment steps stored per rollout (T).
    :param obs_dim: Dimension of a single-agent observation vector.
    :param n_agents: Number of agents producing actions per step.
    :param device: Torch device used when yielding minibatches.
    :param gamma: Discount factor for return computation.
    :param gae_lambda: GAE trade-off parameter (1.0 = Monte Carlo, 0.0 = TD).
    """

    def __init__(self, size, obs_dim, n_agents, device, gamma=0.99, gae_lambda=0.95):
        self.size = size
        self.n_agents = n_agents
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = np.zeros((size, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, n_agents), dtype=np.int64)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.log_probs = np.zeros((size, n_agents), dtype=np.float32)
        self.values = np.zeros((size,), dtype=np.float32)
        self.ep_starts = np.zeros((size,), dtype=np.float32)
        self.advantages = np.zeros((size,), dtype=np.float32)
        self.returns = np.zeros((size,), dtype=np.float32)

        self.pos = 0

    def reset(self):
        """Rewind the write cursor so the next rollout overwrites the buffer."""
        self.pos = 0

    def add(self, obs, action, reward, ep_start, value, log_prob):
        """
        Append a single environment transition.

        :param obs: Per-agent observation, shape (n_agents, obs_dim).
        :param action: Per-agent action indices, shape (n_agents,).
        :param reward: Per-agent reward, shape (n_agents,). Stored as the
            scalar team reward (sum across agents) — see class docstring.
        :param ep_start: True iff this step is the first of a new episode.
        :param value: Centralized critic value V(s) for the joint state (scalar).
        :param log_prob: Per-agent log-prob of the sampled action, shape (n_agents,).
        """
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = float(np.asarray(reward).sum())
        self.ep_starts[i] = float(ep_start)
        self.values[i] = float(value)
        self.log_probs[i] = log_prob
        self.pos += 1

    def compute_returns_and_advantage(self, last_value, last_done):
        """
        Run the GAE(lambda) backward pass to fill `advantages` and `returns`.

        The critic predicts a single joint V(s) and regresses on the
        discounted team return. The TD error at step t is:
            delta_t = r_team_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        The resulting scalar advantage is broadcast across agents inside the
        actor loss.

        :param last_value: Bootstrap V(s) estimate for the state after the
            final stored transition.
        :param last_done: True iff the final stored transition was terminal.
        """
        gae = 0.0

        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = float(last_value)
            else:
                next_non_terminal = 1.0 - self.ep_starts[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size):
        """
        Iterate over random minibatches that together cover the full buffer.

        :param batch_size: Number of timesteps per minibatch.
        :yields: Dict with keys 'obs', 'actions', 'old_values', 'old_logp',
            'advantages', 'returns'.
        """
        idxs = np.random.permutation(self.size)
        for start in range(0, self.size, batch_size):
            b = idxs[start : start + batch_size]
            yield {
                "obs": torch.as_tensor(self.obs[b], device=self.device),
                "actions": torch.as_tensor(self.actions[b], device=self.device),
                "old_values": torch.as_tensor(self.values[b], device=self.device),
                "old_logp": torch.as_tensor(self.log_probs[b], device=self.device),
                "advantages": torch.as_tensor(self.advantages[b], device=self.device),
                "returns": torch.as_tensor(self.returns[b], device=self.device),
            }
