# Cooperative Stag Hunt: MAPPO vs Independent Dueling DDQN

## Problem

This project studies the **Stag Hunt problem**, a canonical coordination dilemma in which agents must jointly choose between a high-payoff cooperative strategy and a safer, individually rational defection strategy.

## Techniques

We compare two reinforcement learning paradigms on a multi-agent gridworld:

- **Independent Q-Learning (IQL)** with a Dueling Double DQN and Prioritized Experience Replay
- **Multi-Agent PPO (MAPPO)** under Centralized Training with Decentralized Execution (CTDE)

## Setup

### 1. Install the custom environment

```bash
cd Gymnasium-Stag-Hunt
pip install -e .
cd ..
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download pretrained weights

Pretrained model weights are available on [SharePoint](https://liveunibo-my.sharepoint.com/:f:/g/personal/samuele_centanni_studio_unibo_it/IgDhjoqdFI5VSIS3gUG_wvTgAWmHm_n9S-4nDVV5q5aMMek?e=cPfiaq).

## How to Run

### CTDE MAPPO

```bash
cd ctde_mappo_coords
python trainer.py
```

> Update the model path in `trainer.py` (`__main__` block) with the path to your downloaded weights before running.

### IQL Dueling DDQN

```bash
cd dqn_indip_learn_coords_new
python agent_train.py
```

> Update the model path in `agent_train.py` (`__main__` block) with the path to your downloaded weights before running.

### Reward sensitivity analysis (IQL only)

```bash
cd dqn_indip_learn_coords_new
python reward_sensitivity.py
```

## Results

Experiments are conducted on 3×3 and 5×5 grids under two stag movement policies: **stag pursues the nearest agent** and **stag performs a random walk**.

IQL achieves near-perfect cooperation when the stag follows agents, but collapses to forage-harvesting equilibrium under stochastic stag movement. MAPPO resolves this asymmetry: on a 5×5 grid with random stag movement it achieves a **100% cooperation rate** across 100 test episodes, demonstrating that centralized training is necessary to coordinate against a non-deterministic target.

| Setting | IQL cooperation rate | MAPPO cooperation rate |
|---|---|---|
| 5×5, stag follows | **100%** | **100%** |
| 5×5, stag random | ~0% | **100%** |