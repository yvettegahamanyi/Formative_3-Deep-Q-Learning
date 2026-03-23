"""
Shared utilities for DQN training experiments on Pong.
"""

import os
import shutil
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor


class PrintEpisodeRewardCallback(BaseCallback):
    """
    Callback to print episode rewards during training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Reset flags indicate episode boundaries in VecEnv
        for i, done in enumerate(self.model.env.buf_dones):
            if done:
                if hasattr(self.model.env, "buf_rews"):
                    ep_reward = self.model.env.buf_rews[i]
                    if hasattr(self.model.env, "ep_info_buffer") and len(self.model.env.ep_info_buffer) > 0:
                        ep_info = self.model.env.ep_info_buffer[-1]
                        if "r" in ep_info:
                            ep_reward = ep_info["r"]
                        if "l" in ep_info:
                            ep_len = ep_info["l"]
                            self.episode_lengths.append(ep_len)
                    self.episode_rewards.append(ep_reward)

        return True


def make_pong_env(
    seed: int = 0,
    monitor_csv_path: str | None = None,
    full_action_space: bool = False,
    n_envs: int = 1,
):
    """
    Create a Pong environment with standard wrappers.

    Args:
        seed: Random seed
        monitor_csv_path: Path to save monitor CSV (if provided)
        full_action_space: Use full action space (18) or reduced (6). Default False for consistency.
        n_envs: Number of parallel environments

    Returns:
        Vectorized and stacked Pong environment
    """
    # Create Pong environment with specified action space
    env = make_atari_env(
        "ALE/Pong-v5",
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"full_action_space": full_action_space},
    )
    env = VecFrameStack(env, n_stack=4)

    # Only wrap with VecMonitor if monitor_csv_path is provided
    # This avoids potential double-Monitor wrapping issues
    if monitor_csv_path is not None:
        os.makedirs(os.path.dirname(monitor_csv_path) or ".", exist_ok=True)
        env = VecMonitor(env, filename=monitor_csv_path)

    return env


def evaluate_dqn_vecenv(model, env, num_episodes: int = 5, print_episodes: bool = False):
    """
    Evaluate a trained DQN model on a vectorized environment.

    Args:
        model: Trained DQN model
        env: Vectorized environment
        num_episodes: Number of episodes to evaluate
        print_episodes: Whether to print episode info

    Returns:
        Tuple of (avg_reward, max_reward, avg_episode_length)
    """
    episode_rewards = []
    episode_lengths = []

    obs = env.reset()
    episode_reward = 0.0
    episode_length = 0

    # Run episodes and track via env's internal tracking
    for step in range(50000):  # Max steps for safety (prevents infinite loops)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        episode_reward += float(reward[0])
        episode_length += 1

        if done[0]:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if print_episodes:
                print(f"  Episode {len(episode_rewards)}: Reward={episode_reward:.1f}, Length={episode_length}")

            if len(episode_rewards) >= num_episodes:
                break

            episode_reward = 0.0
            episode_length = 0

    avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    max_reward = np.max(episode_rewards) if episode_rewards else 0.0
    avg_ep_len = np.mean(episode_lengths) if episode_lengths else 0.0

    return avg_reward, max_reward, avg_ep_len


def copy_best_model(src: str, dst: str):
    """
    Copy a model file from src to dst.

    Args:
        src: Source model path (without .zip)
        dst: Destination model path (without .zip)
    """
    src_zip = f"{src}.zip" if not src.endswith(".zip") else src
    dst_zip = f"{dst}.zip" if not dst.endswith(".zip") else dst

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dst_zip) or ".", exist_ok=True)

    # Copy the file
    shutil.copy2(src_zip, dst_zip)


# Backward compatibility - alias for renamed function
make_private_eye_env = make_pong_env
