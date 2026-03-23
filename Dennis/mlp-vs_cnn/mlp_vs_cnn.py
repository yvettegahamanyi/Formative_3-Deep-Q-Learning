"""
Dennis Mwai Kimiri - MLP vs CNN Policy Comparison (DQN - Pong)

This script:
- Trains DQN with MLPPolicy and CnnPolicy
- Compares performance
- Logs results
"""

import gymnasium as gym
import ale_py
import torch
import numpy as np
import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

print("CUDA Available:", torch.cuda.is_available())

gym.register_envs(ale_py)

# ENV FUNCTION (SAME FOR BOTH)
def make_env(n_envs=1):
    def make_single_env():
        env = gym.make("ALE/Pong-v5")
        env = AtariWrapper(env)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_single_env for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env



# TRAIN + EVALUATE FUNCTION
def train_and_evaluate(policy_name, total_timesteps=1000000):

    print(f"\n{'='*50}")
    print(f"Training with {policy_name}")
    print(f"{'='*50}")

    env = make_env(n_envs=4)
    eval_env = make_env(n_envs=1)

    model = DQN(
        policy=policy_name,
        env=env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=10000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        target_update_interval=1000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=f"./tensorboard_{policy_name}/",
        verbose=1
    )

    model.learn(total_timesteps=total_timesteps)

    # Evaluation
    rewards = []
    obs = eval_env.reset()

    for ep in range(5):
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _ = eval_env.step(action)
            ep_reward += reward[0]
            done = dones[0]

        rewards.append(ep_reward)
        obs = eval_env.reset()
        print(f"Episode {ep+1}: {ep_reward}")

    avg_reward = np.mean(rewards)

    print(f"\nAverage Reward ({policy_name}): {avg_reward:.2f}")

    model.save(f"dqn_{policy_name}_model")

    env.close()
    eval_env.close()

    return avg_reward


# RUN COMPARISON

results = []

for policy in ["MlpPolicy", "CnnPolicy"]:
    avg_reward = train_and_evaluate(policy)

    results.append({
        "Policy": policy,
        "Average Reward": avg_reward
    })

# SAVE + PRINT RESULTS

results_df = pd.DataFrame(results)
results_df.to_csv("mlp_vs_cnn_results.csv", index=False)

print("\nFINAL COMPARISON:")
print(results_df)

best_policy = results_df.loc[results_df["Average Reward"].idxmax()]

print("\nBEST POLICY:")
print(best_policy)