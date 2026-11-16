"""
Reinforcement Learning Training Pipeline for MEC Split Inference.
Supports DQN, PPO, and SAC with real training progress, loss tracking, and checkpoint persistence.
"""

from typing import Dict, Any, List, Optional, Callable
import time
import os

from simulation.environment import MECSimulationEnv
from algorithms.base_policy import BasePolicy
from algorithms.dqn import DQNPolicy
from algorithms.ppo import PPOPolicy
from algorithms.sac import SACPolicy


def train_rl_agent(
    agent: BasePolicy,
    env: MECSimulationEnv,
    num_episodes: int = 50,
    eval_interval: int = 10,
    progress_callback: Optional[Callable[[int, float, float, float], None]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes actual training of RL agent on MECSimulationEnv.
    - Tracks episode reward, loss, and exploration metrics.
    - Invokes optional progress_callback(episode, reward, loss, progress_pct).
    """
    if not agent.is_trainable:
        raise ValueError(f"Agent {agent.name} is not trainable.")

    episode_rewards: List[float] = []
    episode_losses: List[float] = []
    training_start = time.time()

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep * 13 + 42)
        ep_reward = 0.0
        step_losses: List[float] = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(obs, info, deterministic=False)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            ep_reward += reward

            # Algorithm-specific transition handling
            if isinstance(agent, DQNPolicy):
                agent.store_transition(obs, action, reward, next_obs, terminated)
                loss = agent.train_step()
                if loss is not None:
                    step_losses.append(loss)

            elif isinstance(agent, PPOPolicy):
                agent.record_step_result(reward, terminated)

            elif isinstance(agent, SACPolicy):
                agent.store_transition(obs, action, reward, next_obs, terminated)
                loss = agent.train_step()
                if loss is not None:
                    step_losses.append(loss)

            obs = next_obs
            info = next_info

        # PPO updates at end of trajectory rollouts
        if isinstance(agent, PPOPolicy):
            ppo_loss = agent.train_step()
            if ppo_loss is not None:
                step_losses.append(ppo_loss)

        avg_loss = sum(step_losses) / max(1, len(step_losses)) if step_losses else 0.0
        episode_rewards.append(round(ep_reward, 4))
        episode_losses.append(round(avg_loss, 4))
        agent.episode_rewards.append(round(ep_reward, 4))

        if progress_callback:
            progress_pct = (ep + 1) / float(num_episodes)
            progress_callback(ep + 1, ep_reward, avg_loss, progress_pct)

    training_duration = time.time() - training_start
    agent.is_trained = True

    if save_path:
        if hasattr(agent, "save_checkpoint"):
            agent.save_checkpoint(save_path)

    return {
        "agent_name": agent.name,
        "episodes_trained": num_episodes,
        "final_reward": episode_rewards[-1] if episode_rewards else 0.0,
        "avg_reward_last10": sum(episode_rewards[-10:]) / min(10, len(episode_rewards)),
        "episode_rewards": episode_rewards,
        "loss_history": episode_losses,
        "training_duration_seconds": round(training_duration, 2),
        "is_trained": True
    }
