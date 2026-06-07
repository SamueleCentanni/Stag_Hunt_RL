# Cooperative Stag Hunt: MAPPO vs Independent Dueling DDQN Approaches

## Problem
This project aims to solve the **Stag Hunt problem**, which is a canonical coordination dilemma in which agents must jointly choose between a high-payoff cooperative strategy and a safer, individually rational defection.

## Techniques
We study this problem in a multi-agent gridworld and compare two reinforcement learning paradigms: *Independent Q-Learning (IQL)* with a **Dueling Double DQN** and **Prioritized Experience Replay**, and **Multi-Agent PPO (MAPPO)** under *Centralized Training with Decentralized Execution (CTDE)*. 

## Results
Experiments are conducted on different grid sizes under two stag movement policies: stag pursues nearest agent and stag performs a random walk. 
IQL achieves near-perfect cooperation when the stag follows agents, but collapses to forage-harvesting under stochastic stag movement. MAPPO resolves this asymmetry: on a 5 × 5 grid with random stag movement *it achieves perfect cooperation*, demonstrating that centralized training is necessary to coordinate against a non-deterministic target.