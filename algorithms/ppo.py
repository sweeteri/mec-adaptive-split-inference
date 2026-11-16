"""
Algorithm 7: Proximal Policy Optimization (PPO) with Factorized Multi-Discrete Policy.
Separates action decision into factorized categorical heads for split point k, bitwidth b, and frequency f.
Implements Generalized Advantage Estimation (GAE), clipped surrogate objective, and value network.
"""

from typing import Dict, Any, Tuple, Optional, List
import random
import os
import json
import math

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.categorical import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from algorithms.base_policy import BasePolicy


class ActorCriticNetwork(nn.Module if HAS_TORCH else object):
    """
    Factorized Actor-Critic Network for MultiDiscrete([L+1, 4, N_freqs]).
    Computes shared feature representations, 3 factorized action heads, and 1 state-value head.
    """

    def __init__(
        self,
        state_dim: int,
        num_layers: int,
        num_bitwidths: int,
        num_freqs: int,
        hidden_dim: int = 128
    ):
        if HAS_TORCH:
            super().__init__()
            # Shared trunk
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            # Factorized policy heads
            self.head_k = nn.Linear(hidden_dim, num_layers + 1)
            self.head_b = nn.Linear(hidden_dim, num_bitwidths)
            self.head_f = nn.Linear(hidden_dim, num_freqs)
            # Critic value head
            self.head_value = nn.Linear(hidden_dim, 1)

    def forward(self, state: Any) -> Tuple[Any, Any, Any, Any]:
        if HAS_TORCH:
            features = self.trunk(state)
            logits_k = self.head_k(features)
            logits_b = self.head_b(features)
            logits_f = self.head_f(features)
            value = self.head_value(features)
            return logits_k, logits_b, logits_f, value
        return None, None, None, None


class PPOTrajectoryBuffer:
    """Stores rollout transitions for GAE computation and mini-batch PPO updates."""

    def __init__(self):
        self.states: List[List[float]] = []
        self.actions_k: List[int] = []
        self.actions_b: List[int] = []
        self.actions_f: List[int] = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def clear(self) -> None:
        self.states.clear()
        self.actions_k.clear()
        self.actions_b.clear()
        self.actions_f.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)


class PPOPolicy(BasePolicy):
    """
    Proximal Policy Optimization (PPO) agent with factorized categorical policy heads.
    """

    def __init__(
        self,
        total_layers: int = 6,
        num_bitwidths: int = 4,
        num_freq_levels: int = 5,
        state_dim: int = 8,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.20,
        c_value: float = 0.50,
        c_entropy: float = 0.01,
        ppo_epochs: int = 4,
        batch_size: int = 32,
        seed: Optional[int] = 42
    ):
        super().__init__(
            name="PPO",
            description="Proximal Policy Optimization with factorized multi-discrete policy"
        )
        self.total_layers = total_layers
        self.num_bitwidths = num_bitwidths
        self.num_freq_levels = num_freq_levels
        self.state_dim = state_dim
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.c_value = c_value
        self.c_entropy = c_entropy
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.is_trainable = True
        self.is_trained = False
        self.loss_history: List[float] = []
        self.episode_rewards: List[float] = []
        self.rng = random.Random(seed)

        self.device = "cpu"
        if HAS_TORCH:
            self.ac = ActorCriticNetwork(
                state_dim=state_dim,
                num_layers=total_layers,
                num_bitwidths=num_bitwidths,
                num_freqs=num_freq_levels
            ).to(self.device)
            self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        else:
            self.ac = None
            self.optimizer = None

        self.buffer = PPOTrajectoryBuffer()

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
        deterministic: bool = False
    ) -> Tuple[int, int, int]:
        """Samples action from factorized categorical policy."""
        if HAS_TORCH and self.ac is not None:
            with torch.no_grad():
                obs_t = torch.tensor(list(observation), dtype=torch.float32).unsqueeze(0).to(self.device)
                logits_k, logits_b, logits_f, value = self.ac(obs_t)

                if deterministic:
                    k = int(torch.argmax(logits_k, dim=-1).item())
                    b = int(torch.argmax(logits_b, dim=-1).item())
                    f = int(torch.argmax(logits_f, dim=-1).item())
                else:
                    dist_k = Categorical(logits=logits_k)
                    dist_b = Categorical(logits=logits_b)
                    dist_f = Categorical(logits=logits_f)

                    act_k = dist_k.sample()
                    act_b = dist_b.sample()
                    act_f = dist_f.sample()

                    k = int(act_k.item())
                    b = int(act_b.item())
                    f = int(act_f.item())

                    logprob = (
                        dist_k.log_prob(act_k) + dist_b.log_prob(act_b) + dist_f.log_prob(act_f)
                    ).item()
                    val = float(value.item())

                    # Track rollout transition for training
                    self.buffer.states.append(list(observation))
                    self.buffer.actions_k.append(k)
                    self.buffer.actions_b.append(b)
                    self.buffer.actions_f.append(f)
                    self.buffer.logprobs.append(logprob)
                    self.buffer.values.append(val)

                return (k, b, f)

        # Fallback random exploration
        k = self.rng.randint(0, self.total_layers)
        b = self.rng.randint(0, self.num_bitwidths - 1)
        f = self.rng.randint(0, self.num_freq_levels - 1)
        if not deterministic:
            self.buffer.states.append(list(observation))
            self.buffer.actions_k.append(k)
            self.buffer.actions_b.append(b)
            self.buffer.actions_f.append(f)
            self.buffer.logprobs.append(0.0)
            self.buffer.values.append(0.0)
        return (k, b, f)

    def record_step_result(self, reward: float, done: bool) -> None:
        """Stores the environment reward and termination signal corresponding to last action."""
        if len(self.buffer.states) > len(self.buffer.rewards):
            self.buffer.rewards.append(reward)
            self.buffer.dones.append(done)

    def train_step(self) -> Optional[float]:
        """Runs PPO update over collected trajectory buffer with GAE advantages."""
        if not HAS_TORCH or len(self.buffer) < self.batch_size:
            return None

        n = len(self.buffer.rewards)
        if n == 0:
            return None

        # Compute GAE and Returns
        advantages = [0.0] * n
        returns = [0.0] * n
        gae = 0.0
        last_val = 0.0

        for t in reversed(range(n)):
            next_non_terminal = 1.0 - float(self.buffer.dones[t])
            next_val = last_val if t == n - 1 else self.buffer.values[t + 1]
            delta = self.buffer.rewards[t] + self.gamma * next_val * next_non_terminal - self.buffer.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.buffer.values[t]

        # Convert to tensors
        states_t = torch.tensor(self.buffer.states[:n], dtype=torch.float32).to(self.device)
        actions_k_t = torch.tensor(self.buffer.actions_k[:n], dtype=torch.long).to(self.device)
        actions_b_t = torch.tensor(self.buffer.actions_b[:n], dtype=torch.long).to(self.device)
        actions_f_t = torch.tensor(self.buffer.actions_f[:n], dtype=torch.long).to(self.device)
        old_logprobs_t = torch.tensor(self.buffer.logprobs[:n], dtype=torch.float32).to(self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss_epoch = 0.0
        dataset_size = n

        # Multiple PPO optimization epochs
        for _ in range(self.ppo_epochs):
            indices = list(range(dataset_size))
            self.rng.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                b_idx = indices[start:end]

                b_states = states_t[b_idx]
                b_act_k = actions_k_t[b_idx]
                b_act_b = actions_b_t[b_idx]
                b_act_f = actions_f_t[b_idx]
                b_old_logprobs = old_logprobs_t[b_idx]
                b_adv = advantages_t[b_idx]
                b_returns = returns_t[b_idx]

                logits_k, logits_b, logits_f, values = self.ac(b_states)
                dist_k = Categorical(logits=logits_k)
                dist_b = Categorical(logits=logits_b)
                dist_f = Categorical(logits=logits_f)

                new_logprob = (
                    dist_k.log_prob(b_act_k) + dist_b.log_prob(b_act_b) + dist_f.log_prob(b_act_f)
                )
                entropy = (
                    dist_k.entropy() + dist_b.entropy() + dist_f.entropy()
                ).mean()

                ratio = torch.exp(new_logprob - b_old_logprobs)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * ((values.squeeze(-1) - b_returns) ** 2).mean()
                loss = policy_loss + self.c_value * value_loss - self.c_entropy * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss_epoch += float(loss.item())

        self.buffer.clear()
        self.is_trained = True
        mean_loss = total_loss_epoch / max(1, self.ppo_epochs * math.ceil(dataset_size / self.batch_size))
        self.loss_history.append(mean_loss)
        return mean_loss

    def save_checkpoint(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if HAS_TORCH and self.ac is not None:
            torch.save({
                "ac_state": self.ac.state_dict(),
                "loss_history": self.loss_history[-100:]
            }, filepath)

    def load_checkpoint(self, filepath: str) -> bool:
        if HAS_TORCH and os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.ac.load_state_dict(checkpoint["ac_state"])
            self.is_trained = True
            return True
        return False
