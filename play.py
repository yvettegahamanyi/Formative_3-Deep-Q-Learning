import argparse

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)


def play_private_eye(
    model_path: str = "yvette_best_model",
    num_episodes: int = 5,
    render: bool = True,
    full_action_space_mode: str = "auto",
):
    """
    Load a trained DQN model and play Pong with greedy actions.

    This script tries both action-space variants (full_action_space True/False),
    because older checkpoints may have been trained with different settings.
    """
    if full_action_space_mode not in ("auto", "true", "false"):
        raise ValueError("--full-action-space must be one of: auto, true, false")

    if full_action_space_mode == "auto":
        trial_spaces = (True, False)
    else:
        trial_spaces = (full_action_space_mode == "true",)

    model = None
    matched_full_action_space: bool | None = None

    # 1) Load the model using a non-rendering env, just to match the action space.
    for full_action_space in trial_spaces:
        trial_env = make_atari_env(
            "ALE/Pong-v5",
            n_envs=1,
            seed=0,
            env_kwargs={"full_action_space": full_action_space},
        )
        trial_env = VecFrameStack(trial_env, n_stack=4)

        try:
            model = DQN.load(model_path, env=trial_env)
            matched_full_action_space = full_action_space
            print(f"Model loaded successfully from {model_path}.zip (full_action_space={full_action_space})")
            break
        except ValueError as e:
            trial_env.close()
            if "Action spaces do not match" in str(e):
                continue
            raise
        except FileNotFoundError:
            trial_env.close()
            print(f"Error: Model file {model_path}.zip not found!")
            print("Please ensure you have trained the model first using train.py or experiments.py")
            return

    if model is None or matched_full_action_space is None:
        raise RuntimeError(f"Could not load model {model_path}.zip with either action space setting.")

    # 2) Create a render-ready env for gameplay (only after model load succeeds).
    play_env_kwargs = {"full_action_space": matched_full_action_space}
    if render:
        play_env_kwargs["render_mode"] = "human"

    env = make_atari_env(
        "ALE/Pong-v5",
        n_envs=1,
        seed=0,
        env_kwargs=play_env_kwargs,
    )
    env = VecFrameStack(env, n_stack=4)

    print(f"Playing {num_episodes} episodes with the trained agent...\n")

    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += float(reward[0])
            steps += 1
            done = bool(done[0])

            if render:
                # Prefer rendering the underlying single env (VecEnv wrapper).
                try:
                    if hasattr(env, "envs") and env.envs:
                        env.envs[0].render()
                    else:
                        env.render()
                except Exception:
                    pass

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {steps}")

    print(f"\n--- Episode Statistics ---")
    print(f"Average Reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Max Reward: {max(total_rewards):.2f}")
    print(f"Min Reward: {min(total_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a trained DQN model on Pong (Greedy).")
    parser.add_argument("--model", type=str, default="yvette_best_model", help="Model name without .zip (e.g. yvette_best_model, Mariam/best_mariam_model)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--no-render", action="store_true", help="Disable env.render() / GUI")
    parser.add_argument(
        "--full-action-space",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Force action space variant (auto tries both; false is 6 actions, true is 18).",
    )
    args = parser.parse_args()

    play_private_eye(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
        full_action_space_mode=args.full_action_space,
    )
