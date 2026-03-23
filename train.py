import ale_py
import gymnasium as gym
import gym
import argparse
import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecMonitor
from dqn_utils import PrintEpisodeRewardCallback

# Fix for gymnasium compatibility - patch BEFORE any imports happen
if not hasattr(gym, '__version__'):
    gym.__version__ = '0.26.0'

# Also patch the stable_baselines3 utils to handle missing gym version
import stable_baselines3.common.utils as sb3_utils
original_get_system_info = sb3_utils.get_system_info

def patched_get_system_info(print_info=True):
    try:
        return original_get_system_info(print_info)
    except AttributeError:
        # If gym.__version__ doesn't exist, return a minimal response
        import sys
        info_dict = {
            "OS": sys.platform,
            "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "gym": "0.21.0 (patched)",
            "numpy": "installed",
            "torch": "installed"
        }
        info_str = "\n".join([f"{k}: {v}" for k, v in info_dict.items()])
        return info_dict, info_str

sb3_utils.get_system_info = patched_get_system_info


ENV_ID = "ALE/Pong-v5"


def make_env(seed: int = 0, monitor_dir: str | None = None, full_action_space: bool = False):
    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=seed,
        env_kwargs={"full_action_space": full_action_space},
    )
    env = VecFrameStack(env, n_stack=4)
    if monitor_dir is not None:
        os.makedirs(monitor_dir, exist_ok=True)
        env = VecMonitor(env, filename=os.path.join(monitor_dir, "monitor.csv"))
    return env


def train_pong(policy_type="CnnPolicy", total_timesteps=700000, seed: int = 0):
    """
    Train a DQN agent on Pong environment.
    
    Args:
        policy_type (str): "CnnPolicy" for visual agent or "MlpPolicy" for linear agent
        total_timesteps (int): Total training timesteps
        seed (int): Random seed
    
    Note: 
        - CnnPolicy: Recommended for visual Atari (learns spatial features)
        - MlpPolicy: For comparison (simpler but less effective for image-based games)
    """
    # Validate policy type
    if policy_type not in ["CnnPolicy", "MlpPolicy"]:
        raise ValueError("policy_type must be 'CnnPolicy' or 'MlpPolicy'")
    
    # 1. Create and Wrap the Environment
    # We use make_atari_env which applies standard wrappers (NoopReset, MaxAndSkip, Resize, Grayscale)
    env = make_env(seed=seed, monitor_dir="./dqn_pong_logs", full_action_space=True)

    # 2. Define the DQN Agent
    # Baseline Hyperparameters
    print(f"\n{'='*70}")
    print(f"TRAINING DQN AGENT - Pong")
    print(f"{'='*70}")
    print(f"Policy Type: {policy_type}")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"\nHyperparameters:")
    print(f"  Learning Rate (lr):           1e-4")
    print(f"  Gamma (γ) - Discount:         0.99")
    print(f"  Batch Size:                   32")
    print(f"  Epsilon Final:                0.01")
    print(f"  Exploration Fraction:         0.1")
    print(f"  Buffer Size:                  30000")
    print(f"  Target Update Interval:       1000")
    print(f"{'='*70}\n")
    
    model = DQN(
        policy=policy_type,
        env=env,
        learning_rate=1e-4,
        buffer_size=30000,
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
        tensorboard_log="./dqn_pong_logs/"
    )

    # 3. Train the Agent
    print(f"Starting training with {policy_type}...")
    print_callback = PrintEpisodeRewardCallback()
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"dqn_{policy_type}", callback=print_callback)

    # 4. Save the Model
    model.save("dqn_model")
    print(f"\n✓ Model saved as dqn_model.zip")
    print(f"✓ TensorBoard logs saved to ./dqn_pong_logs/")
    print(f"✓ Episode reward/length logs saved to ./dqn_pong_logs/monitor.csv")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir=./dqn_pong_logs/")

def continue_training(model_path: str = "dqn_model", additional_timesteps: int = 700_000, seed: int = 0):
    env = None
    model = None

    # Try both action-space sizes to handle older checkpoints.
    for full_action_space in (True, False):
        trial_env = make_env(seed=seed, monitor_dir="./dqn_pong_logs", full_action_space=full_action_space)
        try:
            model = DQN.load(model_path, env=trial_env)
            env = trial_env
            print(f"\n✓ Loaded {model_path}.zip with full_action_space={full_action_space}")
            break
        except ValueError as e:
            trial_env.close()
            if "Action spaces do not match" in str(e):
                print(f"Action space mismatch for full_action_space={full_action_space}. Retrying...")
                continue
            raise
        except Exception:
            trial_env.close()
            raise

    if model is None or env is None:
        raise RuntimeError(f"Could not load model {model_path}.zip with either action space setting.")

    try:
        print(f"Continuing for +{additional_timesteps} timesteps...")
        print_callback = PrintEpisodeRewardCallback()
        model.learn(
            total_timesteps=additional_timesteps,
            tb_log_name=f"{model_path}_cont",
            callback=print_callback,
            reset_num_timesteps=False,
        )
        model.save(model_path)
        print(f"\n✓ Updated model saved to {model_path}.zip")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Pong (Atari) and save dqn_model.zip")
    parser.add_argument("--policy", choices=["CnnPolicy", "MlpPolicy"], default="CnnPolicy")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", action="store_true", help="Train both CnnPolicy and MlpPolicy (separately)")
    parser.add_argument("--continue-from", type=str, default="", help="If set, loads this model and continues training.")
    parser.add_argument("--additional-timesteps", type=int, default=500_000, help="Timesteps to add when using --continue-from.")
    args = parser.parse_args()

    if args.continue_from:
        continue_training(model_path=args.continue_from, additional_timesteps=args.additional_timesteps, seed=args.seed)
    elif args.compare:
        print("\n" + "="*70)
        print("DQN POLICY COMPARISON: CNN vs MLP")
        print("="*70)
        train_pong(policy_type="CnnPolicy", total_timesteps=args.timesteps, seed=args.seed)
        train_pong(policy_type="MlpPolicy", total_timesteps=args.timesteps, seed=args.seed)
    else:
        train_pong(policy_type=args.policy, total_timesteps=args.timesteps, seed=args.seed)