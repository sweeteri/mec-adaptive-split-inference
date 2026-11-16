"""
Algorithm 6: Deep Q-Network (DQN) for Adaptive Split Inference in MEC.
Uses a joint discrete action space mapping (k, b_idx, f_idx) <-> action_id.
Implements replay buffer, target Q-network, epsilon-greedy exploration, and model persistence.
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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from algorithms.base_policy import BasePolicy


class QNetwork(nn.Module if HAS_TORCH else object):
    """Deep Q-Network mapping state (dim 8) to Q-values for all joint actions."""

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
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x: Any) -> Any:
        if HAS_TORCH:
            return self.net(x)
        return None


class ReplayBuffer:
    """Experience replay buffer for off-policy DQN training."""

    def __init__(self, capacity: int = 10000, seed: Optional[int] = 42):
        self.capacity = capacity
        self.buffer: List[Tuple[Any, int, float, Any, bool]] = []
        self.position = 0
        self.rng = random.Random(seed)

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple[Any, int, float, Any, bool]]:
        return self.rng.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNPolicy(BasePolicy):
    """
    DQN Agent for Adaptive Split Inference:
    Learns Q(s, a) to dynamically choose split point k, bitwidth b, and frequency f.
    """

    def __init__(
        self,
        total_layers: int = 6,
        num_bitwidths: int = 4,
        num_freq_levels: int = 5,
        state_dim: int = 8,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.992,
        buffer_capacity: int = 5000,
        batch_size: int = 32,
        target_update_interval: int = 50,
        seed: Optional[int] = 42
    ):
        super().__init__(
            name="DQN",
            description="Deep Q-Network learning optimal discrete (k, b, f) choices"
        )
        self.total_layers = total_layers
        self.num_bitwidths = num_bitwidths
        self.num_freq_levels = num_freq_levels
        self.num_actions = (total_layers + 1) * num_bitwidths * num_freq_levels
        self.state_dim = state_dim
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.train_steps = 0
        self.rng = random.Random(seed)

        self.is_trainable = True
        self.is_trained = False
        self.loss_history: List[float] = []
        self.episode_rewards: List[float] = []

        # PyTorch Neural Networks
        self.device = "cpu"
        if HAS_TORCH:
            self.q_net = QNetwork(state_dim, self.num_actions).to(self.device)
            self.target_net = QNetwork(state_dim, self.num_actions).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        else:
            self.q_net = None
            self.target_net = None
            self.optimizer = None

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    def encode_action(self, k: int, b_idx: int, f_idx: int) -> int:
        """Bijective mapping from (k, b_idx, f_idx) -> action_id in [0, num_actions - 1]."""
        return k * (self.num_bitwidths * self.num_freq_levels) + b_idx * self.num_freq_levels + f_idx

    def decode_action(self, action_id: int) -> Tuple[int, int, int]:
        """Bijective inverse mapping from action_id -> (k, b_idx, f_idx)."""
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
        """Epsilon-greedy action selection."""
        # Exploration branch
        if not deterministic and self.rng.random() < self.epsilon:
            action_id = self.rng.randint(0, self.num_actions - 1)
            return self.decode_action(action_id)

        # Exploitation branch
        if HAS_TORCH and self.q_net is not None:
            with torch.no_grad():
                obs_tensor = torch.tensor(list(observation), dtype=torch.float32).unsqueeze(0).to(self.device)
                q_vals = self.q_net(obs_tensor)
                action_id = int(torch.argmax(q_vals, dim=-1).item())
                return self.decode_action(action_id)

        # Fallback if torch not loaded
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
        """Samples a batch from the replay buffer and performs one gradient step."""
        if not HAS_TORCH or len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Current Q-values
        curr_q = self.q_net(states_t).gather(1, actions_t)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        loss_val = float(loss.item())
        self.loss_history.append(loss_val)

        # Update target network periodically
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.is_trained = True

        return loss_val

    def save_checkpoint(self, filepath: str) -> None:
        """Saves weights and training state to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if HAS_TORCH and self.q_net is not None:
            torch.save({
                "q_net_state": self.q_net.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
                "loss_history": self.loss_history[-100:]
            }, filepath)
        else:
            meta = {
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
                "loss_history": self.loss_history[-100:]
            }
            with open(filepath + ".json", "w") as f:
                json.dump(meta, f)

    def load_checkpoint(self, filepath: str) -> bool:
        """Loads weights and training state from file."""
        if HAS_TORCH and os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_net.load_state_dict(checkpoint["q_net_state"])
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
            self.train_steps = checkpoint.get("train_steps", 0)
            self.is_trained = True
            return True
        return False
