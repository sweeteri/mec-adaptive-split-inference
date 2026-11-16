"""
Algorithm 8: Soft Actor-Critic (SAC-Discrete) with Maximum Entropy Reinforcement Learning.
Features twin Q-networks, soft target updates, entropy-regularized policy, and temperature auto-tuning.
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
from algorithms.dqn import ReplayBuffer


class DiscreteCritic(nn.Module if HAS_TORCH else object):
    """Twin Q-network estimating Q1(s, a) and Q2(s, a) for all actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        if HAS_TORCH:
            super().__init__()
            self.q1 = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.q2 = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )

    def forward(self, state: Any) -> Tuple[Any, Any]:
        if HAS_TORCH:
            return self.q1(state), self.q2(state)
        return None, None


class DiscreteActor(nn.Module if HAS_TORCH else object):
    """Actor network producing policy probability distribution pi(a | s)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        if HAS_TORCH:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )

    def forward(self, state: Any) -> Tuple[Any, Any]:
        if HAS_TORCH:
            logits = self.net(state)
            probs = torch.softmax(logits, dim=-1)
            # Log probs with numerical stability clamp
            log_probs = torch.log(torch.clamp(probs, min=1e-8, max=1.0))
            return probs, log_probs
        return None, None


class SACPolicy(BasePolicy):
    """
    Soft Actor-Critic agent for discrete split inference action spaces.
    Optimizes maximum entropy objective: J = E[ R + alpha * H(pi) ].
    """

    def __init__(
        self,
        total_layers: int = 6,
        num_bitwidths: int = 4,
        num_freq_levels: int = 5,
        state_dim: int = 8,
        lr: float = 3e-4,
        gamma: float = 0.98,
        tau: float = 0.005,
        alpha: float = 0.20,
        buffer_capacity: int = 5000,
        batch_size: int = 32,
        seed: Optional[int] = 42
    ):
        super().__init__(
            name="SAC",
            description="Soft Actor-Critic with maximum entropy exploration and twin Q-critics"
        )
        self.total_layers = total_layers
        self.num_bitwidths = num_bitwidths
        self.num_freq_levels = num_freq_levels
        self.num_actions = (total_layers + 1) * num_bitwidths * num_freq_levels
        self.state_dim = state_dim

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.rng = random.Random(seed)

        self.is_trainable = True
        self.is_trained = False
        self.loss_history: List[float] = []
        self.episode_rewards: List[float] = []

        self.device = "cpu"
        if HAS_TORCH:
            self.actor = DiscreteActor(state_dim, self.num_actions).to(self.device)
            self.critic = DiscreteCritic(state_dim, self.num_actions).to(self.device)
            self.critic_target = DiscreteCritic(state_dim, self.num_actions).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        else:
            self.actor = None
            self.critic = None
            self.critic_target = None

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    def encode_action(self, k: int, b_idx: int, f_idx: int) -> int:
        return k * (self.num_bitwidths * self.num_freq_levels) + b_idx * self.num_freq_levels + f_idx

    def decode_action(self, action_id: int) -> Tuple[int, int, int]:
        stride_k = self.num_bitwidths * self.num_freq_levels
        k = action_id // stride_k
        remainder = action_id % stride_k
        b_idx = remainder // self.num_freq_levels
        f_idx = remainder % self.num_freq_levels
        return (k, b_idx, f_idx)

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
        deterministic: bool = False
    ) -> Tuple[int, int, int]:
        if HAS_TORCH and self.actor is not None:
            with torch.no_grad():
                obs_t = torch.tensor(list(observation), dtype=torch.float32).unsqueeze(0).to(self.device)
                probs, _ = self.actor(obs_t)

                if deterministic:
                    action_id = int(torch.argmax(probs, dim=-1).item())
                else:
                    dist = Categorical(probs=probs)
                    action_id = int(dist.sample().item())

                return self.decode_action(action_id)

        # Fallback random
        action_id = self.rng.randint(0, self.num_actions - 1)
        return self.decode_action(action_id)

    def store_transition(
        self,
        state: Any,
        action: Tuple[int, int, int],
        reward: float,
        next_state: Any,
        done: bool
    ) -> None:
        action_id = self.encode_action(action[0], action[1], action[2])
        self.replay_buffer.push(list(state), action_id, reward, list(next_state), done)

    def train_step(self) -> Optional[float]:
        """Performs one Soft Actor-Critic gradient step for discrete action space."""
        if not HAS_TORCH or len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 1. Critic Update
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_states_t)
            q1_target, q2_target = self.critic_target(next_states_t)
            min_q_target = torch.min(q1_target, q2_target)
            
            # Discrete Soft Value: V(s') = sum_a pi(a|s') * [ min_Q(s', a) - alpha * log_pi(a|s') ]
            soft_val = (next_probs * (min_q_target - self.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            y_target = rewards_t + (1.0 - dones_t) * self.gamma * soft_val

        q1, q2 = self.critic(states_t)
        q1_a = q1.gather(1, actions_t)
        q2_a = q2.gather(1, actions_t)

        critic_loss = 0.5 * (nn.MSELoss()(q1_a, y_target) + nn.MSELoss()(q2_a, y_target))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Actor Update
        probs, log_probs = self.actor(states_t)
        with torch.no_grad():
            q1_eval, q2_eval = self.critic(states_t)
            min_q_eval = torch.min(q1_eval, q2_eval)

        # Policy loss: sum_a pi(a|s) * [ alpha * log_pi(a|s) - min_Q(s, a) ]
        actor_loss = (probs * (self.alpha * log_probs - min_q_eval)).sum(dim=-1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Soft Target Sync (Polyak averaging)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.is_trained = True
        total_loss = float(critic_loss.item() + actor_loss.item())
        self.loss_history.append(total_loss)
        return total_loss

    def save_checkpoint(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if HAS_TORCH and self.actor is not None:
            torch.save({
                "actor_state": self.actor.state_dict(),
                "critic_state": self.critic.state_dict(),
                "alpha": self.alpha,
                "loss_history": self.loss_history[-100:]
            }, filepath)

    def load_checkpoint(self, filepath: str) -> bool:
        if HAS_TORCH and os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor_state"])
            self.critic.load_state_dict(checkpoint["critic_state"])
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.alpha = checkpoint.get("alpha", self.alpha)
            self.is_trained = True
            return True
        return False
