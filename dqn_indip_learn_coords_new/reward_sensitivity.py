# In this file I will investigate how modifiying the rewards impacts cooperation
# The idea is to find the rewards values such that cooperation emerges.


# The standard rewards for the stag-hunt game are:
# mauling_punishment = -5
# stag_reward = 5
# forage_reward = 1

# What we observed for the 3x3 and 5x5 grid is:
# In both cases, agents learn to stay close one another, in particular on the same cell.
# When the stag_follows flag is set to True, the agents learn to stay still on the same cell, since they've learned that the stag will come to them
# When the stag_follows flag is set to False instead, the agents still learn to cooperate but in a suboptimal way, i.e. they stay together in the same cell but they hunt forage and run away from the stag
    # Each agent optimises its own Q-function independently, with no model of the other agent. With random stag, the probability that both agents occupy the stag's cell simultaneously is low → expected value of approaching the stag is negative (mauling risk dominates). 
    # Foraging yields guaranteed positive reward. Agents independently converge to the risk-dominant equilibrium: stay together on plants, avoid the stag


# Therefore, it make sense to investigate how, modifying the rewards / punishments values, we can have agents that learn to cooperate even when stag_follows = False

# The idea then is the following. We always keep forage_reward = 1, and then I investigate the effect of varying mauling_punishment and stag_reward values:
    # In particular, I will first fix mauling_punishment = -5, while increasing the stag_reward to 7, 10, 15. This will tell the agents that collaborating is much more important than not being mauled.
    # Then, I will instead keep fixed stag_reward, and decrease mauling_punishment to -3, -2, -1. This will tell each agent to try to hunt the stag, and hopefully will introduce an undirect cooperation
# Note: the reward sensitivity is performed for only stag_follows=False!!

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import csv
import torch
import numpy as np
import gymnasium as gym
import gymnasium_stag_hunt
import matplotlib.pyplot as plt
from tqdm import tqdm

from ddqn_agent import Agent

# ---------------------------------------------------------------------------
# Observations (stag_follows=False, 3x3 grid):
#   - agents learn to stay together on the same cell
#   - when stag_follows=True: agents wait still, stag comes to them -> stag hunt
#   - when stag_follows=False: agents cooperate but on forage, actively avoid stag
#     Each agent optimises its own Q-function independently, with no model of the
#     other agent. With random stag, P(both on stag simultaneously) is low ->
#     expected value of approaching stag is negative (mauling risk dominates).
#     Foraging yields guaranteed positive reward. Agents independently converge
#     to the risk-dominant equilibrium: stay together on plants, avoid the stag.
#
# Goal: find reward values such that cooperation on stag hunt emerges even with
#       stag_follows=False.
#
# Sweep strategy (forage_reward=1 fixed throughout):
#   Phase 1 — fix mauling_punishment=-5, increase stag_reward: 7, 10, 15
#             higher stag_reward makes cooperation more attractive
#   Phase 2 — fix stag_reward=5, decrease |mauling_punishment|: -3, -2, -1
#             lower risk may be enough to tip agents toward hunting
# ---------------------------------------------------------------------------

GRID_SIZE  = (5, 5)
NUM_TRAIN  = 5000
NUM_TEST   = 100

ARGUMENTS = {
    'learning_rate':    0.001,
    'gamma':            0.95,
    'batch_size':       32,
    'memory_capacity':  50000,
    'target_frequency': 1000,
    'maximum_exploration': 800000,
    'num_nodes':        64,
    'filling_steps':    1000,
    'replay_steps':     4,
}

CONFIGS = [
     # baseline 
    {"stag_reward": 5,  "mauling_punishment": -5,  "forage_reward": 1},
    # phase 1: increase stag_reward
    {"stag_reward": 7,  "mauling_punishment": -5,  "forage_reward": 1},
    {"stag_reward": 10, "mauling_punishment": -5,  "forage_reward": 1},
    {"stag_reward": 15, "mauling_punishment": -5,  "forage_reward": 1},
    # phase 2: reduce mauling punishment
    {"stag_reward": 5,  "mauling_punishment": -3,  "forage_reward": 1},
    {"stag_reward": 5,  "mauling_punishment": -2,  "forage_reward": 1},
    {"stag_reward": 5,  "mauling_punishment": -1,  "forage_reward": 1},
   
    # phase 3: extreme punishment reduction, increase haunting reward
    {"stag_reward": 15,  "mauling_punishment": -1,  "forage_reward": 0},
    {"stag_reward": 5,  "mauling_punishment": -1,  "forage_reward": 0},

]


def config_tag(cfg, grid_size):
    return f"{grid_size[0]}x{grid_size[1]}_sr{cfg['stag_reward']}_mp{abs(int(cfg['mauling_punishment']))}_fr{cfg['forage_reward']}"


def train_config(cfg, arguments, grid_size, num_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag    = config_tag(cfg, grid_size)

    env = gym.make(
        "StagHunt-Hunt-v0",
        grid_size=grid_size,
        obs_type="coords",
        max_timesteps=200,
        stag_follows=False,
        stag_reward=cfg["stag_reward"],
        mauling_punishment=cfg["mauling_punishment"],
        forage_reward=cfg["forage_reward"],
    )
    state_size  = env.observation_space.shape[1]
    action_size = env.action_space.n

    agent_0 = Agent(state_size, action_size, arguments, device)
    agent_1 = Agent(state_size, action_size, arguments, device)

    filling_steps = arguments['filling_steps']
    replay_steps  = arguments['replay_steps']

    total_step = 0
    stag_catches_list = np.zeros(num_episodes)

    print(f"\n=== Training config: {tag} ===")
    for i in tqdm(range(num_episodes)):
        obs = env.reset()[0]
        terminated, truncated = False, False
        stag_catches_episode  = 0
        ep_reward             = np.zeros(2)

        while not terminated and not truncated:
            action0 = agent_0.act(obs[0])
            action1 = agent_1.act(obs[1])

            new_obs, rewards, terminated, truncated, _ = env.step([action0, action1])

            agent_0.observe((obs[0], action0, new_obs[0], rewards[0], terminated))
            agent_1.observe((obs[1], action1, new_obs[1], rewards[1], terminated))

            if rewards[0] == cfg["stag_reward"] and rewards[1] == cfg["stag_reward"]:
                stag_catches_episode += 1

            ep_reward += rewards
            obs        = new_obs
            total_step += 1

            if total_step >= filling_steps:
                agent_0.decay_epsilon()
                agent_1.decay_epsilon()
                if total_step % replay_steps == 0:
                    agent_0.replay()
                    agent_1.replay()
                agent_0.update_target_model()
                agent_1.update_target_model()

        stag_catches_list[i] = stag_catches_episode

        if (i + 1) % 50 == 0:
            tqdm.write(
                f"  ep {i+1:4d} | R=({ep_reward[0]:+.1f},{ep_reward[1]:+.1f}) "
                f"stag={stag_catches_episode} eps={agent_0.epsilon:.3f}"
            )

    env.close()

    torch.save(agent_0.policy_net.state_dict(), f"sensitivity_agent0_{tag}.pt")
    torch.save(agent_1.policy_net.state_dict(), f"sensitivity_agent1_{tag}.pt")

    return stag_catches_list


def test_config(cfg, arguments, grid_size, num_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag    = config_tag(cfg, grid_size)

    env = gym.make(
        "StagHunt-Hunt-v0",
        grid_size=grid_size,
        obs_type="coords",
        max_timesteps=200,
        stag_follows=False,
        stag_reward=cfg["stag_reward"],
        mauling_punishment=cfg["mauling_punishment"],
        forage_reward=cfg["forage_reward"],
    )
    state_size  = env.observation_space.shape[1]
    action_size = env.action_space.n

    agent_0 = Agent(state_size, action_size, arguments, device)
    agent_1 = Agent(state_size, action_size, arguments, device)

    agent_0.policy_net.load_state_dict(torch.load(f"reward_sensitivity/sensitivity_agent0_{tag}.pt", map_location=device))
    agent_1.policy_net.load_state_dict(torch.load(f"reward_sensitivity/sensitivity_agent1_{tag}.pt", map_location=device))
    agent_0.policy_net.eval()
    agent_1.policy_net.eval()
    agent_0.epsilon = 0.0
    agent_1.epsilon = 0.0

    stag_catches  = 0
    total_rewards = np.zeros(2)

    for _ in range(num_episodes):
        obs = env.reset()[0]
        terminated, truncated    = False, False
        episode_rewards          = np.zeros(2)
        stag_catches_episode     = 0

        while not terminated and not truncated:
            action0 = agent_0.act(obs[0])
            action1 = agent_1.act(obs[1])
            obs, rewards, terminated, truncated, _ = env.step([action0, action1])
            episode_rewards += rewards
            if rewards[0] == cfg["stag_reward"] and rewards[1] == cfg["stag_reward"]:
                stag_catches_episode += 1

        total_rewards += episode_rewards
        if stag_catches_episode > 0:
            stag_catches += 1

    env.close()

    cooperation_rate = stag_catches / num_episodes
    avg_reward       = total_rewards / num_episodes

    print(
        f"  {tag}: cooperation_rate={cooperation_rate:.2f} "
        f"avg_reward=({avg_reward[0]:.1f},{avg_reward[1]:.1f})"
    )
    return {"tag": tag, "config": cfg, "cooperation_rate": cooperation_rate,
            "avg_reward_0": avg_reward[0], "avg_reward_1": avg_reward[1]}


def plot_results(results):
    tags   = [r["tag"] for r in results]
    rates  = [r["cooperation_rate"] for r in results]
    avgr   = [(r["avg_reward_0"] + r["avg_reward_1"]) / 2 for r in results]

    x = np.arange(len(tags))
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x, rates)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tags, rotation=30, ha="right")
    ax1.set_ylabel("Cooperation rate (episodes with ≥1 stag catch)")
    ax1.set_title("Reward sensitivity – cooperation rate")
    ax1.set_ylim(0, 1)

    ax2.bar(x, avgr)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tags, rotation=30, ha="right")
    ax2.set_ylabel("Avg combined reward per episode")
    ax2.set_title("Reward sensitivity – avg reward")

    plt.tight_layout()
    plt.savefig("reward_sensitivity_results.png")
    # plt.show()


def save_results_csv(results):
    path = "reward_sensitivity_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tag", "stag_reward", "mauling_punishment", "forage_reward",
            "cooperation_rate", "avg_reward_0", "avg_reward_1"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "tag":                r["tag"],
                "stag_reward":        r["config"]["stag_reward"],
                "mauling_punishment": r["config"]["mauling_punishment"],
                "forage_reward":      r["config"]["forage_reward"],
                "cooperation_rate":   r["cooperation_rate"],
                "avg_reward_0":       r["avg_reward_0"],
                "avg_reward_1":       r["avg_reward_1"],
            })
    print(f"Results saved to {path}")


if __name__ == "__main__":
    results = []

    for cfg in CONFIGS:
        # train_config(cfg, ARGUMENTS, GRID_SIZE, NUM_TRAIN)
        result = test_config(cfg, ARGUMENTS, GRID_SIZE, NUM_TEST)
        results.append(result)

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['tag']}: cooperation={r['cooperation_rate']:.2f}  "
              f"avg_reward=({r['avg_reward_0']:.1f},{r['avg_reward_1']:.1f})")

    save_results_csv(results)
    plot_results(results)
