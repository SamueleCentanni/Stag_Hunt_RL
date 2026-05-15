import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import gymnasium_stag_hunt
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from buffer import RolloutBuffer

# Centralized Training, Decentralized Execution (CTDE) MAPPO implementation.


class RunningMeanStd:
    """Welford-style running mean/std used for input and return normalization."""

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            x = x[None]
        # Flatten everything except the last (feature) axis when shape is non-scalar.
        if self.mean.shape == ():
            x_flat = x.reshape(-1)
        else:
            x_flat = x.reshape(-1, self.mean.shape[0])
        batch_mean = x_flat.mean(axis=0)
        batch_var = x_flat.var(axis=0)
        batch_count = x_flat.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var) + 1e-8


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, n_agents=2, hidden_dim=128):
        super().__init__()
        # joint state = cat of all agent obs (i.e. n_agents * obs_dim): (batch, n_agents * obs_dim)
        self.net = nn.Sequential(
            nn.Linear(n_agents * obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs):
        # joint_obs: (batch, n_agents, obs_dim)
        batch = joint_obs.shape[0]
        x = joint_obs.reshape(batch, -1)  # (batch, n_agents * obs_dim)
        return self.net(x)  # (batch, 1)


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        # obs_dim = 6 + 2*forage_quantity  (coords of agent0, agent1, stag, plants)
        # action_dim = 5  (up, down, left, right, stay)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),  # raw logits, no activation
        )

    def forward(self, obs):
        # obs: (batch, obs_dim)
        logits = self.net(obs.float())
        return Categorical(logits=logits) # because of integer actions (right, left, up, down, stay)

    def act(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, obs, actions):
        dist = self.forward(obs)
        return dist.log_prob(actions), dist.entropy()
    

class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization implementation.
    Handles on-policy rollout collection and agent training.
    """

    def __init__(
        self,
        env,
        device,
        n_agents=2,
        rollout_steps=128,
        n_epochs=4,
        batch_size=64,
        clip_ratio=0.2,
        gamma=0.9,
        gae_lambda=0.95,
        lr=5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        total_timesteps=1_000_000,
        log_interval=10,
    ):
        """
        Initialize the trainer and hyperparameters.

        :param env: Gymnasium-compatible environment.
        :param n_agents: Number of agents in the system.
        :param rollout_steps: Environment steps collected before training.
        :param n_epochs: Times to reuse the buffer per update.
        :param batch_size: Mini-batch size for SGD.
        :param clip_ratio: PPO epsilon for policy clipping.
        :param gamma: Discount factor.
        :param gae_lambda: GAE tradeoff (1.0=MC, 0.0=TD).
        :param lr: Learning rate.
        :param ent_coef: Entropy bonus weight (exploration).
        :param vf_coef: Value function loss weight.
        :param max_grad_norm: Gradient clipping threshold.
        :param total_timesteps: Total training steps.
        :param log_interval: Frequency of metric logging.
        """
        

        self.device = device
        self.env = env
        self.n_agents = n_agents
        self.rollout_steps = rollout_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.num_trains = 0

        self.create_actor_critic(self.env)
        # Separate optimizers (MAPPO paper: independent Adam updates on θ and φ)
        self.actor_optim = optim.Adam(self.actor_network.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic_network.parameters(), lr=self.lr)

        # Running statistics for input and return normalization
        obs_dim = self.env.observation_space.shape[-1]
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.ret_rms = RunningMeanStd(shape=())

        # Initial entropy coefficient kept for linear annealing in learn()
        self.ent_coef_init = self.ent_coef
        self.ent_coef_final = 1e-3


    def create_actor_critic(self, env):
        # observation_space.shape = (n_agents, obs_dim) — one local observation per agent
        obs_dim = env.observation_space.shape[-1]
        action_dim = env.action_space.n

        self.critic_network = CriticNetwork(
            obs_dim=obs_dim, n_agents=self.n_agents, hidden_dim=128
        ).to(self.device)
        self.actor_network = ActorNetwork(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64
        ).to(self.device)

    def _normalize_obs(self, obs_t):
        """Apply running per-feature obs normalization (obs_t can be (..., obs_dim))."""
        mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=self.device)
        std = torch.as_tensor(self.obs_rms.std, dtype=torch.float32, device=self.device)
        return (obs_t - mean) / std

    def predict(self, obs, deterministic=False):
        """
        Run actor on per-agent obs and critic on joint obs.

        :param obs: np.ndarray shape (n_agents, obs_dim) for a single env step.
        :param deterministic: argmax instead of sampling.
        :return: actions (n_agents,), log_probs (n_agents,), value (scalar).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = self._normalize_obs(obs_t)

        with torch.no_grad():
            batch_policy = self.actor_network(obs_t)
            if deterministic:
                actions = batch_policy.probs.argmax(dim=-1)
            else:
                actions = batch_policy.sample()
            log_probs = batch_policy.log_prob(actions)

            joint = obs_t.unsqueeze(0)  # critic sees the full joint observation
            value = self.critic_network(joint).squeeze()

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            value.cpu().item(),
        )
    
    def collect_rollouts(self, num_rollout_steps):
        """
        Run the environment for `num_rollout_steps` and fill the rollout buffer.

        At the end of the rollout, bootstrap the value of the final state and
        compute returns and GAE advantages.

        :param num_rollout_steps: Number of env steps to collect.
        """
        idx = 0
        done = False
        last_episode_start = True

        while idx < num_rollout_steps:
            obs, _ = self.env.reset()
            obs = np.asarray(obs, dtype=np.float32)  # (n_agents, obs_dim)

            while True:
                idx += 1
                self.num_steps += 1

                actions, log_probs, value = self.predict(obs)

                next_obs, rewards, terminated, truncated, _ = self.env.step(
                    actions.tolist()
                )
                next_obs = np.asarray(next_obs, dtype=np.float32)
                rewards = np.asarray(rewards, dtype=np.float32)  # (n_agents,)
                done = bool(terminated or truncated)

                # Buffer stores team reward (sum of per-agent rewards) for the
                # centralized critic; per-agent log-probs are kept for the actor.
                self.rollout_buffer.add(
                    obs=obs,
                    action=actions,
                    reward=rewards,
                    ep_start=last_episode_start,
                    value=value,
                    log_prob=log_probs,
                )

                obs = next_obs
                last_episode_start = done

                if idx >= num_rollout_steps or done:
                    break

        # Bootstrap the value of the last observed state
        with torch.no_grad():
            _, _, last_value = self.predict(obs)

        self.rollout_buffer.compute_returns_and_advantage(
            last_value=last_value, last_done=last_episode_start
        )

        # Update running statistics from this rollout (use raw, unnormalized data)
        self.obs_rms.update(self.rollout_buffer.obs)
        self.ret_rms.update(self.rollout_buffer.returns)

    def compute_loss(self, batch):
        """
        obs              (B, A, D)
        │
        │ reshape
        ▼
        flat_obs         (B*A, D)
        │
        │ actor MLP
        ▼
        batch_policy     Categorical(batch_shape=(B*A,), event_shape=())
        │
        │ log_prob(flat_actions)
        ▼
        flat_logp        (B*A,)
        │
        │ reshape back
        ▼
        new_logp         (B, A)   ← stessa shape di old_logp, advantages

        """
        """
        Compute the PPO loss on a single minibatch.

        The actor uses per-agent observations and produces per-agent
        log-probabilities and entropy. The centralized critic predicts a
        single joint V(s) per timestep, broadcast across agents to match
        the per-agent return target.

        :param batch: Dict from ``RolloutBuffer.get_minibatches`` with keys
            'obs', 'actions', 'old_logp', 'advantages', 'returns'.
        :return: Tuple (total_loss, metrics_dict).
        """
        obs = batch["obs"]                  # (B, n_agents, obs_dim)
        actions = batch["actions"]          # (B, n_agents)
        old_logp = batch["old_logp"]        # (B, n_agents)
        old_values = batch["old_values"]    # (B,)
        advantages = batch["advantages"]    # (B,)  team-reward advantages
        returns = batch["returns"]          # (B,)  team-reward returns

        B, A, D = obs.shape # Batch (size), (num) Agents, (obs) Dimension

        # Apply running per-feature obs normalization (same as during rollout)
        obs = self._normalize_obs(obs)

        # Normalize advantages across the minibatch for stable updates
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Actor: forward on (B*A, obs_dim), reshape outputs to (B, A) ---
        flat_obs = obs.reshape(B * A, D)
        batch_policy = self.actor_network(flat_obs)
        flat_actions = actions.reshape(B * A)
        new_logp = batch_policy.log_prob(flat_actions).reshape(B, A)
        entropy = batch_policy.entropy().reshape(B, A)

        # PPO clipped surrogate objective — same team advantage for both agents
        adv_exp = advantages.unsqueeze(-1).expand(B, A)
        ratio = torch.exp(new_logp - old_logp)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        actor_loss = -torch.mean(torch.min(ratio * adv_exp, clipped_ratio * adv_exp))

        # --- Critic: one V(s) regresses on team return.
        # Returns are rescaled by running std before the MSE so the loss
        # magnitude stays roughly constant as policy performance grows.
        ret_std = float(self.ret_rms.std)
        ret_mean = float(self.ret_rms.mean)
        values = self.critic_network(obs).squeeze(-1)  # (B,)
        values_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_ratio, self.clip_ratio
        )
        v_n = (values - ret_mean) / ret_std
        v_clip_n = (values_clipped - ret_mean) / ret_std
        ret_n = (returns - ret_mean) / ret_std
        unclipped_loss = (v_n - ret_n) ** 2
        clipped_loss = (v_clip_n - ret_n) ** 2
        critic_loss = torch.max(unclipped_loss, clipped_loss).mean()

        entropy_loss = -entropy.mean()

        # Separate losses: paper uses independent Adam updates on θ and φ.
        actor_total = actor_loss + self.ent_coef * entropy_loss
        critic_total = self.vf_coef * critic_loss

        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.mean().item(),
            "ratio": ratio.mean().item(),
        }
        return actor_total, critic_total, metrics

    def train(self):
        """
        Run the PPO update on the currently collected rollout buffer.

        Iterates over ``n_epochs`` passes; each pass shuffles the buffer into
        minibatches of size ``batch_size`` and performs one optimizer step per
        minibatch. Gradients are clipped to ``max_grad_norm`` for stability.

        :return: Dict of mean metrics aggregated across all minibatches.
        """
        agg = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "ratio": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get_minibatches(self.batch_size):
                actor_total, critic_total, metrics = self.compute_loss(batch)

                self.actor_optim.zero_grad()
                actor_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_network.parameters(), self.max_grad_norm
                )
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_network.parameters(), self.max_grad_norm
                )
                self.critic_optim.step()

                for k in agg:
                    agg[k] += metrics[k]
                n_updates += 1
                self.num_trains += 1

        # keys are "actor_loss", "critic_loss", "entropy", "ratio"
        return {k: v / max(n_updates, 1) for k, v in agg.items()}

    def display_training_metrics(self, metrics):
        """
        Shows total env steps, optimizer updates run so far, and the mean
        loss/entropy/ratio metrics from the latest ``train()`` call.

        :param metrics: Dict returned by :meth:`train`.
        """
        console = Console()

        training_info = Table(show_header=False, box=None, padding=(0, 1))
        training_info.add_row(
            "Steps:", f"[bold cyan]{self.num_steps:,}[/bold cyan]"
        )
        training_info.add_row(
            "Training Runs:", f"[bold magenta]{self.num_trains:,}[/bold magenta]"
        )

        metrics_table = Table(
            title="Training Metrics",
            show_header=True,
            header_style="bold blue",
        )
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta", justify="right")
        for k, v in metrics.items():
            metric_name = k.replace("_", " ").title()
            metrics_table.add_row(metric_name, f"{v:.4f}")

        training_panel = Panel(
            training_info,
            title="Training Progress",
            border_style="blue",
            width=32,
        )
        console.print()
        console.print(training_panel)
        console.print()
        console.print(metrics_table)
        console.print()

    def save(self, path):
        """Save actor and critic weights to ``path`` (PyTorch ``.pt`` file)."""
        torch.save(
            {
                "actor": self.actor_network.state_dict(),
                "critic": self.critic_network.state_dict(),
                "obs_rms_mean": self.obs_rms.mean,
                "obs_rms_var": self.obs_rms.var,
                "obs_rms_count": self.obs_rms.count,
                "ret_rms_mean": self.ret_rms.mean,
                "ret_rms_var": self.ret_rms.var,
                "ret_rms_count": self.ret_rms.count,
            },
            path,
        )

    def load(self, path):
        """Load actor and critic weights from ``path``."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_network.load_state_dict(ckpt["actor"])
        self.critic_network.load_state_dict(ckpt["critic"])
        if "obs_rms_mean" in ckpt:
            self.obs_rms.mean = ckpt["obs_rms_mean"]
            self.obs_rms.var = ckpt["obs_rms_var"]
            self.obs_rms.count = ckpt["obs_rms_count"]
            self.ret_rms.mean = ckpt["ret_rms_mean"]
            self.ret_rms.var = ckpt["ret_rms_var"]
            self.ret_rms.count = ckpt["ret_rms_count"]

    def evaluate(self, n_episodes=10, deterministic=True, render=False):
        """
        Roll the current policy out for ``n_episodes`` and return the average
        per-agent and team rewards.

        The actor is queried with ``deterministic=True`` by default so the
        evaluation reflects the greedy policy rather than the sampling
        behaviour used during training. Gradients are not tracked.

        :param n_episodes: Number of full episodes to play.
        :param deterministic: If True, take argmax actions; otherwise sample.
        :param render: If True, call ``env.render()`` after each step.
        :return: Dict with keys 'mean_team_return', 'mean_agent_returns',
            'mean_episode_length'.
        """
        team_returns = []
        agent_returns = []
        ep_lengths = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            ep_reward = np.zeros(self.n_agents, dtype=np.float32)
            length = 0

            while True:
                actions, _, _ = self.predict(obs, deterministic=deterministic)
                next_obs, rewards, terminated, truncated, _ = self.env.step(
                    actions.tolist()
                )
                ep_reward += np.asarray(rewards, dtype=np.float32)
                length += 1

                if render:
                    self.env.render()

                if terminated or truncated:
                    break
                obs = np.asarray(next_obs, dtype=np.float32)

            team_returns.append(float(ep_reward.sum()))
            agent_returns.append(ep_reward.copy())
            ep_lengths.append(length)

        return {
            "mean_team_return": float(np.mean(team_returns)),
            "mean_agent_returns": np.mean(np.stack(agent_returns), axis=0).tolist(),
            "mean_episode_length": float(np.mean(ep_lengths)),
        }

    def learn(self, total_timesteps=None, eval_interval=None, eval_episodes=10):
        """
        Main PPO training loop.

        Allocates the rollout buffer once, then repeats:
        collect rollout, train on it, reset buffer — until the total
        environment-step budget is exhausted.

        :param total_timesteps: Total env steps to train for. Defaults to the
            value passed to ``__init__``.
        :param eval_interval: If set, run :meth:`evaluate` every
            ``eval_interval`` iterations and store the result in the
            returned history.
        :param eval_episodes: Number of episodes per evaluation call.
        :return: List of dicts (one per iteration) with keys
            'iteration', 'num_steps', plus the keys returned by
            :meth:`train` and (optionally) 'eval'.
        """
        if total_timesteps is None:
            total_timesteps = self.total_timesteps

        self.num_steps = 0
        obs_dim = self.env.observation_space.shape[-1]

        self.rollout_buffer = RolloutBuffer(
            size=self.rollout_steps,
            obs_dim=obs_dim,
            n_agents=self.n_agents,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        history = []
        iteration = 0
        progress_bar = tqdm(
            total=total_timesteps, desc="Training", unit="step", dynamic_ncols=True
        )
        while self.num_steps < total_timesteps:
            prev_steps = self.num_steps
            self.collect_rollouts(self.rollout_steps)
            # Linear entropy-coefficient annealing: init -> final over the budget
            frac = min(1.0, self.num_steps / max(1, total_timesteps))
            self.ent_coef = self.ent_coef_final + (
                self.ent_coef_init - self.ent_coef_final
            ) * (1.0 - frac)
            metrics = self.train()
            self.rollout_buffer.reset()
            iteration += 1

            record = {"iteration": iteration, "num_steps": self.num_steps, **metrics}

            if eval_interval is not None and iteration % eval_interval == 0:
                record["eval"] = self.evaluate(n_episodes=eval_episodes)

            history.append(record)

            progress_bar.update(self.num_steps - prev_steps)
            postfix = {
                "iter": iteration,
                "actor": f"{metrics['actor_loss']:.3f}",
                "critic": f"{metrics['critic_loss']:.2f}",
                "H": f"{metrics['entropy']:.3f}",
            }
            if "eval" in record:
                postfix["eval"] = f"{record['eval']['mean_team_return']:.2f}"
            progress_bar.set_postfix(postfix)

            if iteration % self.log_interval == 0:
                self.display_training_metrics(metrics)

        progress_bar.close()
        return history
