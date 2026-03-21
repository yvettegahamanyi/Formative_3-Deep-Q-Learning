import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

def train_private_eye(policy_type="CnnPolicy", total_timesteps=100000):
    """
    Train a DQN agent on PrivateEye environment.
    
    Args:
        policy_type (str): "CnnPolicy" for visual agent or "MlpPolicy" for linear agent
        total_timesteps (int): Total training timesteps
    
    Note: 
        - CnnPolicy: Recommended for visual Atari (learns spatial features)
        - MlpPolicy: For comparison (simpler but less effective for image-based games)
    """
    # Validate policy type
    if policy_type not in ["CnnPolicy", "MlpPolicy"]:
        raise ValueError("policy_type must be 'CnnPolicy' or 'MlpPolicy'")
    
    # 1. Create and Wrap the Environment
    # We use make_atari_env which applies standard wrappers (NoopReset, MaxAndSkip, Resize, Grayscale)
    env = make_atari_env("ALE/PrivateEye-v5", n_envs=1, seed=0)
    
    # Frame stacking helps the agent perceive motion (crucial for navigating environments)
    env = VecFrameStack(env, n_stack=4)

    # 2. Define the DQN Agent
    # Baseline Hyperparameters
    print(f"\n{'='*70}")
    print(f"TRAINING DQN AGENT - PrivateEye")
    print(f"{'='*70}")
    print(f"Policy Type: {policy_type}")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"\nHyperparameters:")
    print(f"  Learning Rate (lr):           1e-4")
    print(f"  Gamma (γ) - Discount:         0.99")
    print(f"  Batch Size:                   32")
    print(f"  Epsilon Final:                0.01")
    print(f"  Exploration Fraction:         0.1")
    print(f"  Buffer Size:                  50000")
    print(f"  Target Update Interval:       1000")
    print(f"{'='*70}\n")
    
    model = DQN(
        policy=policy_type,
        env=env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=10000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,  # Discount factor - balance immediate vs future rewards
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,  # epsilon_decay equivalent
        exploration_final_eps=0.01,  # epsilon_end - minimum exploration rate
        target_update_interval=1000,
        verbose=1,
        tensorboard_log="./dqn_private_eye_logs/"
    )

    # 3. Train the Agent
    print(f"Starting training with {policy_type}...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"dqn_{policy_type}")

    # 4. Save the Model
    model.save("dqn_model")
    print(f"\n✓ Model saved as dqn_model.zip")
    print(f"✓ TensorBoard logs saved to ./dqn_private_eye_logs/")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir=./dqn_private_eye_logs/")

if __name__ == "__main__":
    # TASK 1: Compare CNNPolicy vs MlpPolicy
    print("\n" + "="*70)
    print("DQN POLICY COMPARISON: CNN vs MLP")
    print("="*70)
    print("\nCNNPolicy: Recommended for visual environments (learns spatial features)")
    print("MlpPolicy: For comparison (simpler, less effective for image-based games)")
    print("\nRunning baseline with CnnPolicy...\n")
    
    # Experiment 1: Baseline with CNNPolicy (Recommended)
    train_private_eye(policy_type="CnnPolicy", total_timesteps=100000)
    
    # Optional: Uncomment to also test MlpPolicy for comparison
    # print("\n" + "="*70)
    # print("Now testing with MlpPolicy for comparison...")
    # print("="*70 + "\n")
    # train_private_eye(policy_type="MlpPolicy", total_timesteps=100000)