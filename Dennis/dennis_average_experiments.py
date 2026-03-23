"""
Dennis Mwai Kimiri - Advanced RL Experiment Pipeline (Pong)
Includes:
- Hyperparameter tuning
- Early stopping
- Best model selection
- Final 1M training
"""

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import pandas as pd
import numpy as np
import torch

print("CUDA Available:", torch.cuda.is_available())

gym.register_envs(ale_py)

# ENV CREATION (FIXED)
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

def make_env(n_envs, seed):
    
    def make_single_env():
        env = gym.make("ALE/Pong-v5")
        env = AtariWrapper(env)
        env = Monitor(env)  
        return env

    env = DummyVecEnv([make_single_env for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    return env
# BASE CONFIG
BASELINE_CONFIG = {
    "policy": "CnnPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 50000,
    "learning_starts": 10000,
    "batch_size": 32,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.01,
    "target_update_interval": 1000,
    "verbose": 0,
}

# EXPERIMENTS
EXPERIMENTS = [
    {"name": "Exp1", "learning_rate": 2e-4},
    {"name": "Exp2", "learning_rate": 7.5e-5},
    {"name": "Exp3", "learning_rate": 1.5e-4, "buffer_size": 75000},
    {"name": "Exp4", "gamma": 0.97},
    {"name": "Exp5", "gamma": 0.98},
    {"name": "Exp6", "gamma": 0.97, "learning_rate": 1.5e-4},
    {"name": "Exp7", "batch_size": 40},
    {"name": "Exp8", "batch_size": 48},
    {"name": "Exp9", "exploration_final_eps": 0.005},
    {"name": "Exp10", "learning_rate": 1.5e-4, "gamma": 0.97, "batch_size": 48,
     "exploration_final_eps": 0.005, "buffer_size": 75000},
]
#policykwargs
policy_kwargs = dict(net_arch=[256, 256])

# TRAIN FUNCTION WITH EARLY STOPPING
def train_and_evaluate(config, exp_name, exp_num, total_timesteps=200000):
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {exp_num}: {exp_name}")
    print(f"{'='*60}")

    # Use identical env pipeline
    env = make_env(n_envs=4, seed=0)
    eval_env = make_env(n_envs=1, seed=42)

    # Early stopping
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=18, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=10000,
        n_eval_episodes=3,
        best_model_save_path="./best_models/",
        verbose=0
    )

    model = DQN(
        policy=config["policy"],
        env=env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        tau=config["tau"],
        gamma=config["gamma"],
        train_freq=config["train_freq"],
        gradient_steps=config["gradient_steps"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        target_update_interval=config["target_update_interval"],
        policy_kwargs=policy_kwargs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tensorboard_logs/",
        verbose=0
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Evaluation
    rewards = []
    obs = eval_env.reset()

    for _ in range(3):
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _ = eval_env.step(action)
            ep_reward += reward[0]
            done = dones[0]
        
        rewards.append(ep_reward)
        obs = eval_env.reset()

    avg_reward = np.mean(rewards)
    print(f"Avg Reward: {avg_reward:.2f}")

    env.close()
    eval_env.close()

    return avg_reward

# RUN EXPERIMENTS
results = []

for i, exp in enumerate(EXPERIMENTS, 1):
    config = {**BASELINE_CONFIG, **{k: v for k, v in exp.items() if k != "name"}}
    avg_reward = train_and_evaluate(config, exp["name"], i)
    
    results.append({
        "Exp_Num": i,
        "Experiment": exp["name"],
        "Avg_Reward": avg_reward
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)

print("\nRESULTS:")
print(results_df)


# SELECT BEST CONFIG
best_idx = results_df["Avg_Reward"].idxmax()
best_exp = EXPERIMENTS[best_idx]

print("\nBEST EXPERIMENT:", best_exp)

best_config = {
    **BASELINE_CONFIG,
    **{k: v for k, v in best_exp.items() if k != "name"}
}

# FINAL TRAINING (1M STEPS)
print("\nSTARTING FINAL TRAINING (1M STEPS)...")

env = make_env(n_envs=4, seed=0)

model = DQN(
    policy=best_config["policy"],
    env=env,
    learning_rate=best_config["learning_rate"],
    buffer_size=best_config["buffer_size"],
    learning_starts=best_config["learning_starts"],
    batch_size=best_config["batch_size"],
    tau=best_config["tau"],
    gamma=best_config["gamma"],
    train_freq=best_config["train_freq"],
    gradient_steps=best_config["gradient_steps"],
    exploration_fraction=best_config["exploration_fraction"],
    exploration_final_eps=best_config["exploration_final_eps"],
    target_update_interval=best_config["target_update_interval"],
    device="cuda" if torch.cuda.is_available() else "cpu",
    tensorboard_log="./tensorboard_logs/",
    verbose=1
)

model.learn(total_timesteps=10_000_000)
model.save("final_pong_model")

env.close()

print("\nFINAL MODEL TRAINED AND SAVED")