"""
Mariam Awini Issah - 10 hyperparameter tuning experiments (Lower/Stability-focused).

Outputs (inside this folder):
- models/dqn_mariam_exp{N}.zip
- results_mariam.csv
- best_mariam_model.zip
"""

import os
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

# Register ALE environments
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_utils import copy_best_model, evaluate_dqn_vecenv, make_pong_env, PrintEpisodeRewardCallback  # type: ignore


HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MONITOR_DIR = HERE / "monitor"
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = HERE / "results_mariam.csv"
BEST_MODEL_ZIP = HERE / "best_mariam_model.zip"

CSV_COLUMNS = [
    "Member",
    "Exp_Num",
    "Experiment",
    "lr",
    "gamma",
    "batch_size",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay",
    "buffer_size",
    "Avg_Reward",
    "Max_Reward",
    "Avg_Ep_Len",
    "Status",
]


BASELINE = {
    "policy": "CnnPolicy",
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "batch_size": 32,
    "buffer_size": 30_000,
    "learning_starts": 10_000,
    "tau": 1.0,
    "train_freq": 4,
    "gradient_steps": 1,
    # epsilon_start ~= 1.0 in SB3; decay governed by exploration_fraction
    "exploration_fraction": 0.10,     # epsilon_decay (fraction of training spent decaying)
    "exploration_final_eps": 0.01,    # epsilon_end
    "target_update_interval": 1000,
    "verbose": 0,
}


EXPERIMENTS = [
    # Group 1: lr + batch, gamma + buffer, then combine
    {"name": "Exp1: Lower LR + Smaller Batch", "learning_rate": 5e-5, "batch_size": 24},
    {"name": "Exp2: Lower Gamma + Smaller Buffer", "gamma": 0.95, "buffer_size": 30_000},
    {
        "name": "Exp3: Combine (Lower LR + Smaller Batch + Lower Gamma + Smaller Buffer)",
        "learning_rate": 5e-5,
        "gamma": 0.95,
        "batch_size": 24,
        "buffer_size": 30_000,
    },

    # Group 2: very low lr + epsilon, very low gamma + faster decay, then combine
    {"name": "Exp4: Very Low LR + Lower Epsilon End", "learning_rate": 1e-5, "exploration_final_eps": 0.005},
    {"name": "Exp5: Very Low Gamma + Faster Epsilon Decay", "gamma": 0.90, "exploration_fraction": 0.05},
    {
        "name": "Exp6: Combine (Very Low LR + Very Low Gamma + Epsilon Tweaks)",
        "learning_rate": 1e-5,
        "gamma": 0.90,
        "exploration_final_eps": 0.005,
        "exploration_fraction": 0.05,
    },

    # Group 3: batch + buffer, batch + epsilon, then combine
    {"name": "Exp7: Smaller Batch + Smaller Buffer", "batch_size": 16, "buffer_size": 30_000},
    {"name": "Exp8: Very Small Batch + Very Low Epsilon End", "batch_size": 8, "exploration_final_eps": 0.001},
    {
        "name": "Exp9: Combine (Very Small Batch + Smaller Buffer + Very Low Epsilon End)",
        "batch_size": 8,
        "buffer_size": 30_000,
        "exploration_final_eps": 0.001,
    },

    # Group 4: final combined “best lower” guess using patterns above
    {
        "name": "Exp10: Combined Lower (best guess)",
        "learning_rate": 5e-5,
        "gamma": 0.95,
        "batch_size": 16,
        "buffer_size": 30_000,
        "exploration_final_eps": 0.001,
        "exploration_fraction": 0.05,
    },
]


def _load_existing_results() -> dict[int, dict]:
    if not RESULTS_CSV.exists():
        return {}
    try:
        df = pd.read_csv(RESULTS_CSV)
        if "Exp_Num" not in df.columns:
            return {}
        out: dict[int, dict] = {}
        for row in df.to_dict(orient="records"):
            try:
                exp_num = int(row["Exp_Num"])
            except Exception:
                continue

            # Normalize older column names -> assignment-friendly names.
            normalized = dict(row)
            if "Learning_Rate" in normalized and "lr" not in normalized:
                normalized["lr"] = normalized["Learning_Rate"]
            if "Gamma" in normalized and "gamma" not in normalized:
                normalized["gamma"] = normalized["Gamma"]
            if "Batch_Size" in normalized and "batch_size" not in normalized:
                normalized["batch_size"] = normalized["Batch_Size"]
            if "Epsilon_Start" in normalized and "epsilon_start" not in normalized:
                normalized["epsilon_start"] = normalized["Epsilon_Start"]
            if "Epsilon_End" in normalized and "epsilon_end" not in normalized:
                normalized["epsilon_end"] = normalized["Epsilon_End"]
            if "Epsilon_Decay" in normalized and "epsilon_decay" not in normalized:
                normalized["epsilon_decay"] = normalized["Epsilon_Decay"]
            if "Buffer_Size" in normalized and "buffer_size" not in normalized:
                normalized["buffer_size"] = normalized["Buffer_Size"]
            # Keep Avg_Reward / Max_Reward / Avg_Ep_Len / Status / Experiment / Member / Exp_Num as-is.

            out[exp_num] = normalized
        return out
    except Exception:
        return {}


def run(
    total_timesteps: int = 50_000,
    num_eval_episodes: int = 2,
    seed: int = 0,
    extra_timesteps: int = 20_000,
    improve_existing: bool = False,
):
    existing = _load_existing_results()
    rows: list[dict] = []

    for exp_num, exp in enumerate(EXPERIMENTS, start=1):
        model_zip = MODELS_DIR / f"dqn_mariam_exp{exp_num}.zip"
        already_ok = (
            exp_num in existing
            and str(existing[exp_num].get("Status", "")).startswith("✓")
            and model_zip.exists()
        )

        if already_ok and not improve_existing:
            print(f"Skipping Exp{exp_num} (already completed): {existing[exp_num].get('Experiment')}")
            rows.append(existing[exp_num])
            continue

        config = dict(BASELINE)
        config.update({k: v for k, v in exp.items() if k != "name"})

        print("\n" + "=" * 70)
        print(f"EXPERIMENT {exp_num}: {exp['name']}")
        print("=" * 70)
        print(
            f"lr={config['learning_rate']}, gamma={config['gamma']}, batch={config['batch_size']}, "
            f"eps_end={config['exploration_final_eps']}, eps_decay={config['exploration_fraction']}, "
            f"buffer={config['buffer_size']}"
        )

        # Try both action space settings to match the saved model if loading
        env = None
        model = None
        for full_action_space_trial in (False, True):
            try:
                env = make_pong_env(seed=seed, monitor_csv_path=str(MONITOR_DIR / f"exp{exp_num}.monitor.csv"), full_action_space=full_action_space_trial)
                
                print_callback = PrintEpisodeRewardCallback()
                if already_ok and improve_existing:
                    print(f"Improving existing Exp{exp_num} for +{extra_timesteps} timesteps (action_space={full_action_space_trial})...")
                    try:
                        model = DQN.load(str(model_zip.with_suffix("").as_posix()), env=env)
                        model.learn(
                            total_timesteps=extra_timesteps,
                            callback=print_callback,
                            reset_num_timesteps=False,
                        )
                        break  # Success - exit the loop
                    except ValueError as e:
                        if "Action spaces do not match" in str(e):
                            env.close()
                            env = None
                            continue  # Try next action space setting
                        raise
                else:
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
                        verbose=config["verbose"],
                    )
                    model.learn(total_timesteps=total_timesteps, callback=print_callback)
                    break  # Success - exit the loop
                
            except ValueError as e:
                if env is not None:
                    env.close()
                    env = None
                if "Action spaces do not match" in str(e):
                    continue  # Try next action space setting
                raise
            except Exception as e:
                if env is not None:
                    env.close()
                    env = None
                raise
        
        if model is None or env is None:
            raise RuntimeError(f"Failed to create or load model for Exp{exp_num} with either action space setting")
        
        try:
            model.save(str(model_zip.with_suffix("").as_posix()))

            avg_reward, max_reward, avg_ep_len = evaluate_dqn_vecenv(
                model, env, num_episodes=num_eval_episodes, print_episodes=True
            )
            status = "✓ Success" if not (already_ok and improve_existing) else "✓ Success (improved)"
        except Exception as e:
            avg_reward, max_reward, avg_ep_len = 0.0, 0.0, 0
            status = f"✗ Failed: {e}"
        finally:
            env.close()

        rows.append(
            {
                "Member": "Mariam Awini Issah",
                "Exp_Num": exp_num,
                "Experiment": exp["name"],
                "lr": config["learning_rate"],
                "gamma": config["gamma"],
                "batch_size": config["batch_size"],
                "epsilon_start": 1.0,
                "epsilon_end": config["exploration_final_eps"],
                "epsilon_decay": config["exploration_fraction"],
                "buffer_size": config["buffer_size"],
                "Avg_Reward": float(np.round(avg_reward, 2)),
                "Max_Reward": float(np.round(max_reward, 2)),
                "Avg_Ep_Len": int(avg_ep_len),
                "Status": status,
            }
        )

        pd.DataFrame(rows).sort_values("Exp_Num")[CSV_COLUMNS].to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows).sort_values("Exp_Num")
    df = df[CSV_COLUMNS]
    ok_df = df[df["Status"].astype(str).str.startswith("✓")].copy()
    if len(ok_df) > 0:
        best_row = ok_df.loc[ok_df["Avg_Reward"].astype(float).idxmax()]
        best_exp = int(best_row["Exp_Num"])
        best_src = MODELS_DIR / f"dqn_mariam_exp{best_exp}.zip"
        copy_best_model(str(best_src), str(BEST_MODEL_ZIP))
        print(f"\n✓ Best Mariam model: Exp{best_exp} -> {BEST_MODEL_ZIP}")
    else:
        print("\nNo successful runs to select a best model yet.")

    print(f"\n✓ Results saved: {RESULTS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mariam: 10 hyperparameter experiments with optional improvement.")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 10K timesteps, 1 eval episode (for testing)")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training timesteps (default: 500K full, 10K quick)")
    parser.add_argument("--extra-timesteps", type=int, default=None, help="Extra timesteps for improvement (default: 200K full, 5K quick)")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes (default: 2 full, 1 quick)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--improve-existing", action="store_true", help="Continue training existing models for +extra-timesteps")
    args = parser.parse_args()

    # Auto-adjust defaults based on --quick flag
    if args.quick:
        total_timesteps = args.total_timesteps if args.total_timesteps is not None else 10_000
        extra_timesteps = args.extra_timesteps if args.extra_timesteps is not None else 5_000
        num_eval_episodes = args.eval_episodes if args.eval_episodes is not None else 1
        print("\n[QUICK MODE] 10K training, 1 eval episode (for testing)")
    else:
        total_timesteps = args.total_timesteps if args.total_timesteps is not None else 500_000
        extra_timesteps = args.extra_timesteps if args.extra_timesteps is not None else 200_000
        num_eval_episodes = args.eval_episodes if args.eval_episodes is not None else 2
        print("\n[FULL MODE] 500K training, 2 eval episodes")

    run(
        total_timesteps=total_timesteps,
        extra_timesteps=extra_timesteps,
        num_eval_episodes=num_eval_episodes,
        seed=args.seed,
        improve_existing=args.improve_existing,
    )

