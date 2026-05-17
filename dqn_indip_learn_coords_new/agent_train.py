import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import csv
import os
import torch
import numpy as np
import gymnasium as gym
import gymnasium_stag_hunt
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from ddqn_agent import Agent

np.set_printoptions(suppress=True, precision=2)


def train(num_episodes, arguments, grid_size=(3, 3), social_welfare=False, stag_follows=False, max_timesteps=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make("StagHunt-Hunt-v0", grid_size=grid_size, obs_type="coords",max_timesteps=max_timesteps, stag_follows=stag_follows)
    state_size  = env.observation_space.shape[1]
    action_size = env.action_space.n

    # two independent agents, each with their own network and PER buffer (IQL)
    agent_0 = Agent(state_size, action_size, arguments, device)
    agent_1 = Agent(state_size, action_size, arguments, device)

    filling_steps = arguments['filling_steps']  # steps before training starts, used to fill the replay buffer
    replay_steps  = arguments['replay_steps']   # steps between each replay

    
    base_env = env.unwrapped
    stag_reward = getattr(base_env, "stag_reward", 5)
    forage_reward = getattr(base_env, "forage_reward", 1)
    maul_punish = getattr(base_env, "mauling_punishment", -5)

    rewards_list = np.zeros((2, num_episodes))
    stag_catches_list = np.zeros(num_episodes)
    total_step = 0

    # CSV reward log
    log_path = os.path.join(os.path.dirname(__file__), f"reward_log_{grid_size[0]}x{grid_size[1]}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode", "total_steps",
        "ep_reward_0", "ep_reward_1",
        "stag_catches",
        "forage_count_0", "forage_count_1",
        "maul_count_0",   "maul_count_1",
        "zero_count_0",   "zero_count_1",
        "epsilon_0", "epsilon_1",
        "per_alpha_0", "per_beta_0",
        "per_alpha_1", "per_beta_1",
    ])

    for i in tqdm(range(num_episodes)):
        obs = env.reset()[0]
        terminated, truncated = False, False
        stag_catches_episode  = 0

        # 2 agents
        ep_reward = np.zeros(2)
        forage_count = np.zeros(2, dtype=int)
        maul_count = np.zeros(2, dtype=int)
        zero_count = np.zeros(2, dtype=int)

        while not terminated and not truncated:
            # each agent has its own observations, the other agent is part of the environment
            action0 = agent_0.act(obs[0])
            action1 = agent_1.act(obs[1])

            new_obs, rewards, terminated, truncated, _ = env.step([action0, action1])

            # store single-agent transitions in separate buffers
            # First approach: Individual Utility
            rewards = list(rewards)
            single_rewards = rewards[:]

            # Second approach: social welfare
            if social_welfare:
                social_reward = rewards[0] + rewards[1]
                rewards[0] = social_reward
                rewards[1] = social_reward

            # add transitions to PER buffer
            transition_0 = (obs[0], action0, new_obs[0], rewards[0], False)
            transition_1 = (obs[1], action1, new_obs[1], rewards[1], False)
            agent_0.observe(transition_0)
            agent_1.observe(transition_1)

            # Save stats
            if single_rewards[0] == stag_reward and single_rewards[1] == stag_reward:
                stag_catches_episode += 1
            for k in (0, 1):
                r = single_rewards[k]
                if r == forage_reward:
                    forage_count[k] += 1
                elif r == maul_punish:
                    maul_count[k] += 1
                elif r == 0:
                    zero_count[k] += 1
            ep_reward += single_rewards

            obs = new_obs
            total_step += 1

            # only start training after filling_steps random transitions
            if total_step >= filling_steps:
                agent_0.decay_epsilon()
                agent_1.decay_epsilon()

                # replay every replay_steps steps
                if total_step % replay_steps == 0:
                    agent_0.replay()
                    agent_1.replay()

                agent_0.update_target_model()
                agent_1.update_target_model()

        rewards_list[:, i] = ep_reward
        stag_catches_list[i] = stag_catches_episode

        log_writer.writerow([
            i, total_step,
            float(ep_reward[0]), float(ep_reward[1]),
            int(stag_catches_episode),
            int(forage_count[0]), int(forage_count[1]),
            int(maul_count[0]),   int(maul_count[1]),
            int(zero_count[0]),   int(zero_count[1]),
            float(agent_0.epsilon), float(agent_1.epsilon),
            float(agent_0.memory.alpha), float(agent_0.memory.beta),
            float(agent_1.memory.alpha), float(agent_1.memory.beta),
        ])

        if (i + 1) % 50 == 0:
            log_file.flush()
            tqdm.write(
                f"ep {i+1:5d} | R=({ep_reward[0]:+.1f},{ep_reward[1]:+.1f}) "
                f"stag={stag_catches_episode} "
                f"forage=({forage_count[0]},{forage_count[1]}) "
                f"maul=({maul_count[0]},{maul_count[1]}) "
                f"eps=({agent_0.epsilon:.3f},{agent_1.epsilon:.3f}) "
                f"PER a/b=({agent_0.memory.alpha:.2f}/{agent_0.memory.beta:.2f})"
            )

    log_file.close()
    env.close()

    # save models
    torch.save(agent_0.policy_net.state_dict(), f"agent0_dueling_ddqn_per_{grid_size[0]}x{grid_size[1]}.pt")
    torch.save(agent_1.policy_net.state_dict(), f"agent1_dueling_ddqn_per_{grid_size[0]}x{grid_size[1]}.pt")

    # plot
    episode_rewards = rewards_list[0] + rewards_list[1]
    sum_rewards = np.zeros(num_episodes)
    for x in range(num_episodes):
        sum_rewards[x] = np.mean(episode_rewards[max(0, x - 100):(x + 1)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(sum_rewards)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Avg combined reward (100-ep window)")
    ax1.set_title(f"Dueling DDQN PER – Hunt {grid_size[0]}x{grid_size[1]}")

    ax2.plot(stag_catches_list)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Stag catches per episode")
    ax2.set_title("Dueling DDQN PER – Cooperation rate")

    plt.tight_layout()
    plt.savefig(f"dueling_ddqn_per_{grid_size[0]}x{grid_size[1]}.png")
    plt.show()


def test(agent_0_path, agent_1_path, num_episodes, arguments, grid_size=(3, 3), load_renderer=False, max_timesteps=200, stag_follows=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("StagHunt-Hunt-v0", grid_size=grid_size, obs_type="coords", max_timesteps=max_timesteps, load_renderer=load_renderer, stag_follows=stag_follows)
    state_size  = env.observation_space.shape[1]
    action_size = env.action_space.n

    agent_0 = Agent(state_size, action_size, arguments, device)
    agent_1 = Agent(state_size, action_size, arguments, device)

    agent_0.policy_net.load_state_dict(torch.load(agent_0_path, map_location=device))
    agent_1.policy_net.load_state_dict(torch.load(agent_1_path, map_location=device))
    agent_0.policy_net.eval()
    agent_1.policy_net.eval()

    # no exploration during test
    agent_0.epsilon = 0.0
    agent_1.epsilon = 0.0

    stag_catches  = 0
    total_rewards = np.zeros(2)
    forage_total = np.zeros(2, dtype=int)
    maul_total = np.zeros(2, dtype=int)

    for i in range(num_episodes):
        obs = env.reset()[0]
        terminated, truncated = False, False
        episode_rewards = np.zeros(2)
        ep_forage = np.zeros(2, dtype=int)
        ep_maul = np.zeros(2, dtype=int)
        stag_catches_episode = 0

        while not terminated and not truncated:
            action0 = agent_0.act(obs[0])
            action1 = agent_1.act(obs[1])

            obs, rewards, terminated, truncated, _ = env.step([action0, action1])
            episode_rewards += rewards

            if load_renderer:
                env.unwrapped.render()
                time.sleep(1.0)

                
            if rewards[0] == 5 and rewards[1] == 5:
                stag_catches_episode += 1
            for k in (0, 1):
                if rewards[k] == 1:
                    ep_forage[k] += 1
                elif rewards[k] == -5:
                    ep_maul[k] += 1

            

        total_rewards += episode_rewards
        forage_total += ep_forage
        maul_total += ep_maul
        if stag_catches_episode > 0:
            stag_catches += 1

        print(f"Episode {i+1:3d}: rewards = {episode_rewards}, stag catches = {stag_catches_episode}")

    print(f"\nStag caught : {stag_catches}/{num_episodes}")
    print(f"Avg rewards : agent0={total_rewards[0]/num_episodes:.2f}  agent1={total_rewards[1]/num_episodes:.2f}")
    print(f"Avg team reard: {(total_rewards[0] + total_rewards[1]) / num_episodes}")
    print(f"Avg forage events: {forage_total/num_episodes}")
    print(f"Avg maul event: {maul_total/num_episodes}")
    env.close()


if __name__ == "__main__":
    arguments = {
        'learning_rate':   0.001,
        'gamma':           0.95,
        'batch_size':      32,
        'memory_capacity': 50000,
        'target_frequency': 1000,   
        'maximum_exploration': 3500000,  
        'num_nodes':       64,
        'filling_steps':   1000,
        'replay_steps':    4,
    }

    agent_0_path = "/home/samuelecentanni/Desktop/University/AAS/exam/dqn_indip_learn_coords_new/5x5_results_Stagfollows_False_socialwelf/agent0_dueling_ddqn_per_5x5.pt"
    agent_1_path = "/home/samuelecentanni/Desktop/University/AAS/exam/dqn_indip_learn_coords_new/5x5_results_Stagfollows_False_socialwelf/agent1_dueling_ddqn_per_5x5.pt"
    max_timesteps = 200
    stag_follows = False

    # train(num_episodes=5000, arguments=arguments, grid_size=(5, 5), social_welfare=False, stag_follows=stag_follows, max_timesteps=max_timesteps)
    test(agent_0_path=agent_0_path, agent_1_path=agent_1_path, num_episodes=100, arguments=arguments, grid_size=(5, 5), load_renderer=True, max_timesteps=max_timesteps, stag_follows=stag_follows)