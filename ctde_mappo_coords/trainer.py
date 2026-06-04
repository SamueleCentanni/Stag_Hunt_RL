import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import csv
import os
import time

import torch
import numpy as np
import gymnasium as gym
import gymnasium_stag_hunt
import matplotlib.pyplot as plt

from mappo import MAPPO

np.set_printoptions(suppress=True, precision=2)


def train(
    total_timesteps,
    hparams,
    grid_size=(5, 5),
    max_timesteps=200,
    stag_follows=True,
    save_dir=None,
    eval_interval=10,
    eval_episodes=10,
):
    """
    Train a CTDE MAPPO agent on StagHunt-Hunt-v0
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(
        "StagHunt-Hunt-v0",
        grid_size=grid_size,
        obs_type="coords",
        max_timesteps=max_timesteps,
        stag_follows=stag_follows,
    )

    mappo = MAPPO(env=env, device=device, **hparams)

    history = mappo.learn(
        total_timesteps=total_timesteps,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
    )

    # ----- CSV log -----
    log_path = os.path.join(save_dir, f"mappo_log_{grid_size[0]}x{grid_size[1]}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "num_steps",
            "actor_loss", "critic_loss", "entropy", "ratio",
            "eval_team_return", "eval_ep_length",
        ])
        for rec in history:
            ev = rec.get("eval")
            writer.writerow([
                rec["iteration"], rec["num_steps"],
                f"{rec['actor_loss']:.6f}",
                f"{rec['critic_loss']:.6f}",
                f"{rec['entropy']:.6f}",
                f"{rec['ratio']:.6f}",
                f"{ev['mean_team_return']:.4f}" if ev else "",
                f"{ev['mean_episode_length']:.2f}" if ev else "",
            ])
    print(f"CSV log: {log_path}")

    # ----- Save model -----
    model_path = os.path.join(save_dir, f"mappo_{grid_size[0]}x{grid_size[1]}.pt")
    mappo.save(model_path)
    print(f"Model: {model_path}")

    # ----- Plot -----
    iters = [r["iteration"] for r in history]
    actor_l = [r["actor_loss"] for r in history]
    critic_l = [r["critic_loss"] for r in history]
    entropy = [r["entropy"] for r in history]
    eval_pts = [(r["iteration"], r["eval"]["mean_team_return"])
                for r in history if "eval" in r]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(iters, actor_l, label="actor", color="tab:blue")
    axes[0].plot(iters, critic_l, label="critic", color="tab:orange")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"MAPPO losses – Hunt {grid_size[0]}x{grid_size[1]}")
    axes[0].legend()

    axes[1].plot(iters, entropy, color="tab:green")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Mean entropy")
    axes[1].set_title("Policy entropy")

    if eval_pts:
        xs, ys = zip(*eval_pts)
        axes[2].plot(xs, ys, marker="o", color="tab:red")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Mean team return")
        axes[2].set_title(f"Eval ({eval_episodes} episodes)")
    else:
        axes[2].set_title("Eval — disabled")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"mappo_{grid_size[0]}x{grid_size[1]}.png")
    plt.savefig(plot_path)
    print(f"Plot: {plot_path}")
    plt.show()

    env.close()
    return mappo, history


def test(
    model_path,
    hparams,
    grid_size=(5, 5),
    n_episodes=100,
    max_timesteps=200,
    stag_follows=True,
    load_renderer=False,
    deterministic=True,
):
    """
    Roll out a trained MAPPO policy and report aggregate statistics.

    If ``load_renderer`` is True, the env is rendered after each step
    (slows the rollout substantially).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(
        "StagHunt-Hunt-v0",
        grid_size=grid_size,
        obs_type="coords",
        max_timesteps=max_timesteps,
        load_renderer=load_renderer,
        stag_follows=stag_follows,
    )

    mappo = MAPPO(env=env, device=device, **hparams)
    mappo.load(model_path)
    mappo.actor_network.eval()
    mappo.critic_network.eval()

    # Reward constants for diagnostics
    base_env = env.unwrapped
    stag_reward = getattr(base_env, "stag_reward", 5)
    forage_reward = getattr(base_env, "forage_reward", 1)
    maul_punish = getattr(base_env, "mauling_punishment", -5)

    stag_catches = 0
    total_rewards = np.zeros(2)
    forage_total = np.zeros(2, dtype=int)
    maul_total = np.zeros(2, dtype=int)
    lengths = []

    for i in range(n_episodes):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        ep_reward = np.zeros(2)
        ep_forage = np.zeros(2, dtype=int)
        ep_maul = np.zeros(2, dtype=int)
        ep_stag = 0
        length = 0

        while True:
            actions, _, _ = mappo.predict(obs, deterministic=deterministic)
            next_obs, rewards, terminated, truncated, _ = env.step(
                actions.tolist()
            )
            rewards = np.asarray(rewards, dtype=np.float32)
            ep_reward += rewards
            length += 1

            if rewards[0] == stag_reward and rewards[1] == stag_reward:
                ep_stag += 1
            for k in (0, 1):
                if rewards[k] == forage_reward:
                    ep_forage[k] += 1
                elif rewards[k] == maul_punish:
                    ep_maul[k] += 1

            if load_renderer:
                env.unwrapped.render()
                time.sleep(0.7)

            if terminated or truncated:
                break
            obs = np.asarray(next_obs, dtype=np.float32)

        total_rewards += ep_reward
        forage_total += ep_forage
        maul_total += ep_maul
        if ep_stag > 0:
            stag_catches += 1
        lengths.append(length)

        print(
            f"Ep {i+1:3d}: R=({ep_reward[0]:+.1f},{ep_reward[1]:+.1f}) "
            f"stag={ep_stag} forage=({ep_forage[0]},{ep_forage[1]}) "
            f"maul=({ep_maul[0]},{ep_maul[1]}) len={length}"
        )

    print(f"\nStag-catching episodes: {stag_catches}/{n_episodes}")
    print(f"Avg per-agent reward:   "
          f"a0={total_rewards[0]/n_episodes:.2f}  "
          f"a1={total_rewards[1]/n_episodes:.2f}")
    print(f"Avg team reward:        {total_rewards.sum()/n_episodes:.2f}")
    print(f"Avg forage events:      "
          f"a0={forage_total[0]/n_episodes:.2f}  "
          f"a1={forage_total[1]/n_episodes:.2f}")
    print(f"Avg maul events:        "
          f"a0={maul_total[0]/n_episodes:.2f}  "
          f"a1={maul_total[1]/n_episodes:.2f}")
    print(f"Avg episode length:     {np.mean(lengths):.1f}")

    env.close()


if __name__ == "__main__":
    hparams = {
        "n_agents":       2,
        "rollout_steps":  2048,
        "n_epochs":       4,
        "batch_size":     64,
        "clip_ratio":     0.2,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "lr":             3e-4,
        "ent_coef":       0.01,
        "vf_coef":        0.5,
        "max_grad_norm":  0.5,
        "log_interval":   10,
    }

    # train(total_timesteps=2_000_000, hparams=hparams, grid_size=(7, 7), max_timesteps=200, stag_follows=False, eval_interval=10, eval_episodes=50)

    test(model_path="/home/samuelecentanni/Desktop/University/AAS/exam/ctde_mappo_coords/5x5_stagFollows_False/mappo_5x5.pt", hparams=hparams, grid_size=(5,5), n_episodes=100, load_renderer=True, stag_follows=False)
