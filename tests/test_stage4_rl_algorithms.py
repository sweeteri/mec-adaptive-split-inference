"""
Unit tests and verification script for Stage 4: DQN, PPO, and SAC Reinforcement Learning algorithms.
"""

import sys
import os
import shutil

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import MECSimulationEnv
from algorithms.dqn import DQNPolicy
from algorithms.ppo import PPOPolicy
from algorithms.sac import SACPolicy
from training.train import train_rl_agent
from training.model_manager import ModelManager


def test_dqn_policy():
    print("Testing Algorithm 6: DQN Policy...")
    env = MECSimulationEnv(seed=42, max_steps_per_episode=10)
    dqn = DQNPolicy(total_layers=6, batch_size=8, seed=42)
    
    # Action selection test
    obs, info = env.reset(seed=42)
    action = dqn.select_action(obs, info, deterministic=False)
    assert len(action) == 3
    k, b_idx, f_idx = action
    assert 0 <= k <= 6
    assert 0 <= b_idx <= 3
    assert 0 <= f_idx <= 4
    
    # Step and transition storing
    next_obs, reward, term, trunc, next_info = env.step(action)
    dqn.store_transition(obs, action, reward, next_obs, term)
    assert len(dqn.replay_buffer) == 1
    
    # Short training run
    res = train_rl_agent(dqn, env, num_episodes=3)
    assert res["episodes_trained"] == 3
    assert dqn.is_trained
    
    # Checkpoint saving and loading
    tmp_path = "results/models/test_dqn.pth"
    dqn.save_checkpoint(tmp_path)
    loaded = dqn.load_checkpoint(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print(f"  DQN passed. 3 episodes trained, reward: {res['final_reward']:.3f}")


def test_ppo_policy():
    print("Testing Algorithm 7: PPO Policy with factorized multi-discrete categorical heads...")
    env = MECSimulationEnv(seed=42, max_steps_per_episode=10)
    ppo = PPOPolicy(total_layers=6, batch_size=8, seed=42)
    
    obs, info = env.reset(seed=42)
    action = ppo.select_action(obs, info, deterministic=False)
    assert len(action) == 3
    k, b_idx, f_idx = action
    assert 0 <= k <= 6
    assert 0 <= b_idx <= 3
    assert 0 <= f_idx <= 4

    next_obs, reward, term, trunc, next_info = env.step(action)
    ppo.record_step_result(reward, term)
    assert len(ppo.buffer) == 1

    # Short training run
    res = train_rl_agent(ppo, env, num_episodes=3)
    assert res["episodes_trained"] == 3
    assert ppo.is_trained
    print(f"  PPO passed. 3 episodes trained, reward: {res['final_reward']:.3f}")


def test_sac_policy():
    print("Testing Algorithm 8: SAC Policy with maximum entropy discrete actions...")
    env = MECSimulationEnv(seed=42, max_steps_per_episode=10)
    sac = SACPolicy(total_layers=6, batch_size=8, seed=42)
    
    obs, info = env.reset(seed=42)
    action = sac.select_action(obs, info, deterministic=False)
    assert len(action) == 3
    k, b_idx, f_idx = action
    assert 0 <= k <= 6
    assert 0 <= b_idx <= 3
    assert 0 <= f_idx <= 4

    next_obs, reward, term, trunc, next_info = env.step(action)
    sac.store_transition(obs, action, reward, next_obs, term)
    assert len(sac.replay_buffer) == 1

    # Short training run
    res = train_rl_agent(sac, env, num_episodes=3)
    assert res["episodes_trained"] == 3
    assert sac.is_trained
    print(f"  SAC passed. 3 episodes trained, reward: {res['final_reward']:.3f}")


if __name__ == "__main__":
    test_dqn_policy()
    test_ppo_policy()
    test_sac_policy()
    print("\nALL STAGE 4 RL ALGORITHMS (DQN, PPO, SAC) VERIFIED SUCCESSFULLY!")
