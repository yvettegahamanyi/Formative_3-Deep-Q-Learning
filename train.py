import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

def train_private_eye(policy_type="CnnPolicy", total_timesteps=100000):
    # 1. Create and Wrap the Environment
    # We use make_atari_env which applies standard wrappers (NoopReset, MaxAndSkip, Resize, Grayscale)
    env = make_atari_env("ALE/PrivateEye-v5", n_envs=1, seed=0)
    
    # Frame stacking helps the agent perceive motion (crucial for navigating streets)
    env = VecFrameStack(env, n_stack=4)

    # 2. Define the DQN Agent
    # Default Hyperparameters for the baseline
    model = DQN(
        policy=policy_type,
        env=env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=10000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log="./dqn_private_eye_logs/"
    )

    # 3. Train the Agent
    print(f"Starting training with {policy_type}...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"dqn_{policy_type}")

    # 4. Save the Model
    model.save("dqn_model")
    print("Model saved as dqn_model.zip")

if __name__ == "__main__":
    # Experiment 1: Baseline with CNNPolicy
    train_private_eye(policy_type="CnnPolicy", total_timesteps=100000)