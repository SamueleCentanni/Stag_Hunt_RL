# Implementation of the Dualing Double DQN, in which proposed a simple trick to decompose the action selection from the action evaluation. 
# More specifically, in the double DQN, the online network is used to select an action and the target network is
# used to generate the target value for that action. 

import random
import numpy as np
import torch
import torch.nn as nn

from dqn import DQN
from replay_memory import PERBuffer

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01


class Agent:
    """
    Dueling DDQN Agent with Prioritized Experience Replay.
    """

    def __init__(self, state_size, action_size, arguments, device):
        """
        :param state_size:  dimension of the observation vector
        :param action_size: number of discrete actions
        :param arguments:   dict of hyperparameters
        :param device:      torch device (cpu or cuda)

        arguments keys:
            learning_rate:       Adam learning rate
            gamma:               discount factor
            batch_size:          number of transitions per training step
            memory_capacity:     max transitions in buffer
            target_frequency:    steps between target net syncs
            maximum_exploration: steps over which epsilon decays
            num_nodes:           hidden layer size
        """
        self.state_size   = state_size
        self.action_size  = action_size
        self.device       = device
        self.gamma        = arguments['gamma']
        self.batch_size   = arguments['batch_size']
        self.update_target_frequency  = arguments['target_frequency']
        self.max_exploration_step     = arguments['maximum_exploration']
        self.step = 0
        self.epsilon = MAX_EPSILON

        # independent policy and target networks
        self.policy_net = DQN(state_size, action_size, num_nodes=arguments['num_nodes']).to(device)
        self.target_net = DQN(state_size, action_size, num_nodes=arguments['num_nodes']).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # we don't train it directly, we just use it to predict the target Q values

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=arguments['learning_rate'])
        self.loss_fn   = nn.SmoothL1Loss(reduction='none')  # per-sample loss for IS weighting

        self.memory = PERBuffer(arguments['memory_capacity'])

    #  action selection 

    def act(self, state):
        """
        Epsilon-greedy action selection using the policy net.

        :param state: observation array for one agent
        :return: integer action
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q = self.policy_net(torch.tensor(state, dtype=torch.float32).to(self.device))
            return q.argmax().item()

    #  storing transitions 

    def observe(self, transition):
        """
        Stores a transition in the PER buffer with max priority so it is
        sampled at least once. The real TD-error is computed in replay()
        and used to refine the priority there.

        :param transition: tuple (obs, action, new_obs, reward, terminated)
                           note: single agent transition, not the joint one
        """
        self.memory.add(self.memory.max_priority, transition)

    # epsilon decay 

    def decay_epsilon(self):
        """
        Linearly decays epsilon from MAX_EPSILON to MIN_EPSILON
        over max_exploration_step steps.
        """
        self.step += 1
        if self.step < self.max_exploration_step:
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
                           (self.max_exploration_step - self.step) / self.max_exploration_step
        else:
            self.epsilon = MIN_EPSILON

    # training step 

    def replay(self):
        """
        Samples a batch from PER and performs one DDQN gradient update.
        Loss is weighted by Importance Sampling weights.
        Priorities are updated after the gradient step.
        """
        if len(self.memory) < self.batch_size:
            return

        batch, indices, weights = self.memory.sample(self.batch_size)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # unpack, single agent transitions (one host→device transfer per array)
        obs_np     = np.stack([t[0] for t in batch]).astype(np.float32)
        new_obs_np = np.stack([t[2] for t in batch]).astype(np.float32)
        acts_np    = np.fromiter((t[1] for t in batch), dtype=np.int64, count=len(batch))
        rews_np    = np.fromiter((t[3] for t in batch), dtype=np.float32, count=len(batch))
        dones_np   = np.fromiter((float(t[4]) for t in batch), dtype=np.float32, count=len(batch))

        obs     = torch.from_numpy(obs_np).to(self.device)
        new_obs = torch.from_numpy(new_obs_np).to(self.device)
        acts    = torch.from_numpy(acts_np).to(self.device)
        rews    = torch.from_numpy(rews_np).to(self.device)
        dones   = torch.from_numpy(dones_np).to(self.device)

        # current Q values
        current_q = self.policy_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)

        # DDQN target
        with torch.no_grad():
            best_actions = self.policy_net(new_obs).argmax(dim=1)
            max_next_q = self.target_net(new_obs).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rews + self.gamma * max_next_q * (1.0 - dones)


        # Train
        # IS-weighted loss
        td_errors = (current_q - target_q).abs()
        loss_per_sample = self.loss_fn(current_q, target_q)
        loss = (weights_tensor * loss_per_sample).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # update PER priorities
        self.memory.update(indices, td_errors.detach().cpu().numpy())

    # target network sync 
    def update_target_model(self):
        """
        Copies weights from policy net to target net every update_target_frequency steps.
        """
        if self.step % self.update_target_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())