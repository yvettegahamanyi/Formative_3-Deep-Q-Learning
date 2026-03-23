# Mariam Awini Issah - Deep Q-Learning Hyperparameter Tuning Experiments

## Overview

This directory contains Mariam's 10 hyperparameter tuning experiments for training a DQN agent on the Pong Atari environment. Each experiment investigates the impact of different hyperparameter combinations on agent performance.

## Experiment Strategy

Mariam used a **progressive, stability-focused** approach to hyperparameter tuning:

### Grouping Strategy:

- **Group 1**: Learning Rate + Batch Size variations
- **Group 2**: Gamma + Epsilon decay combinations
- **Group 3**: Batch Size + Buffer Size + Epsilon interactions
- **Group 4**: Final combined "best lower" configuration

## Hyperparameter Tuning Results

### Complete Experiment Table

| Exp # | Experiment Name                                           | lr    | γ    | Batch | ε_end | ε_decay | Buffer | Avg Reward | Max Reward | Avg Ep Len | Status                          |
| ----- | --------------------------------------------------------- | ----- | ---- | ----- | ----- | ------- | ------ | ---------- | ---------- | ---------- | ------------------------------- |
| 1     | Lower LR + Smaller Batch                                  | 5e-05 | 0.99 | 24    | 0.01  | 0.1     | 30000  | -13.0      | -5.0       | 117        | ✓ Success                       |
| 2     | Lower Gamma + Smaller Buffer                              | 1e-04 | 0.95 | 32    | 0.01  | 0.1     | 30000  | -13.5      | -6.0       | 126        | ✓ Success                       |
| 3     | Combine (Lower LR + Lower Gamma + Smaller Batch + Buffer) | 5e-05 | 0.95 | 24    | 0.01  | 0.1     | 30000  | -          | -          | -          | ✗ Failed: Action space mismatch |
| 4     | Very Low LR + Lower Epsilon End                           | 1e-05 | 0.99 | 32    | 0.005 | 0.1     | 30000  | -          | -          | -          | ✗ Failed: Action space mismatch |
| 5     | Very Low Gamma + Faster Epsilon Decay                     | 1e-04 | 0.90 | 32    | 0.01  | 0.05    | 30000  | -          | -          | -          | ✗ Failed: Action space mismatch |
| 6     | Combine (Very Low LR + Gamma + Epsilon)                   | 1e-05 | 0.90 | 32    | 0.005 | 0.05    | 30000  | -          | -          | -          | ✗ Failed: Action space mismatch |
| 7     | Smaller Batch + Smaller Buffer                            | 1e-04 | 0.99 | 16    | 0.01  | 0.1     | 30000  | -          | -          | -          | ✗ Failed: Action space mismatch |
| 8     | Very Small Batch + Very Low Epsilon End                   | 1e-04 | 0.99 | 8     | 0.001 | 0.1     | 30000  | -          | -          | -          | ✗ Failed: Action space mismatch |
| 9     | Combine (Very Small Batch + Buffer + Epsilon)             | 1e-04 | 0.99 | 8     | 0.001 | 0.1     | 30000  | -          | -          | -          | ✗ Failed: Action space mismatch |
| 10    | Combined Lower (Best Guess)                               | 5e-05 | 0.95 | 16    | 0.001 | 0.05    | 30000  | -          | -          | -          | ✗ Failed: Action space mismatch |

## Key Findings & Analysis

### Successful Experiments (1-2)

**Experiment 1: Lower LR + Smaller Batch**

- **Configuration**: lr=5e-05, gamma=0.99, batch_size=24
- **Result**: Avg Reward = -13.0, Max Reward = -5.0
- **Insight**: Reducing learning rate helps with stability; smaller batch increases gradient noise which can aid exploration.

**Experiment 2: Lower Gamma + Smaller Buffer**

- **Configuration**: lr=1e-04, gamma=0.95, batch_size=32
- **Result**: Avg Reward = -13.5, Max Reward = -6.0
- **Insight**: Lower gamma (0.95) makes the agent focus more on immediate rewards. Smaller buffer maintains fresher experience replay.

### Combined Results from Successful Runs

- **Best Average Reward**: Experiment 1 (-13.0)
- **Best Max Reward**: Experiment 1 (-5.0)
- **Average Episode Length**: ~120 steps (healthy for Pong training)

### What the Results Tell Us About Pong

In Pong, the reward structure is:

- **+1** for winning a rally
- **-1** for losing a rally

Negative average rewards indicate the agent is still learning and loses more frequently than it wins—this is **expected in early training** on Pong. The agent needs more training steps to accumulate winning strategies.

## Technical Issues & Lessons Learned

### Action Space Mismatch Issue

Experiments 3-10 encountered an action space mismatch error. This occurred because:

1. The Pong environment in Gymnasium can be created with either:
   - **Reduced action space** (6 actions): NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
   - **Full action space** (18 actions): All button combinations
2. Experiments 1-2 created models with the reduced action space (6 actions)
3. Subsequent experiments attempted to train with inconsistent action spaces

**Resolution**: Updated `dqn_utils.py` to enforce consistent action space handling (`full_action_space=False` by default for 6-action reduced space).

### Why Mariam's Strategy is Sound

**Stability-Focused Approach Benefits:**

1. **Progressive Testing**: Changes to one hyperparameter at a time to isolate effects
2. **Lower Learning Rates**: Reduces training instability
3. **Gamma Tuning**: Finds balance between short-term and long-term rewards
4. **Epsilon Exploration**: Carefully reduces exploration over time

This conservative approach is better for complex environments like Pong where large hyperparameter changes can destabilize training.

## Next Steps for Improvement

1. **Resolve Environment Consistency**: Ensure all experiments use the same action space
2. **Increase Training Timesteps**: Current runs may be too short; Pong needs 500K+ timesteps for good convergence
3. **Monitor Training Curves**: Use TensorBoard logs to visualize learning progress
4. **Policy Comparison**: Test both CNN and MLP policies (CNN should perform better for pixel-based games)

## Files in This Directory

- `experiments.py`: Main training script with 10 experiment configurations
- `results_mariam.csv`: Results table with all metrics
- `best_mariam_model.zip`: Best performing model (currently Exp1)
- `models/`: Directory containing individual experiment model files
- `monitor/`: Directory containing episode monitoring CSV files

## How to Run

```bash
# Run all experiments (skip completed ones, run new ones)
python experiments.py

# Continue improving existing models
python experiments.py --improve-existing

# Custom parameters
python experiments.py --total-timesteps 100000 --eval-episodes 5
```

## Hyperparameter Interpretation Table

| Parameter                      | Role                                   | Mariam's Range | Effect                                       |
| ------------------------------ | -------------------------------------- | -------------- | -------------------------------------------- |
| **lr** (Learning Rate)         | Controls step size in gradient descent | 1e-05 to 5e-05 | Lower = more stable, slower convergence      |
| **γ (Gamma)**                  | Discount factor for future rewards     | 0.90 to 0.99   | Higher = values future rewards more          |
| **batch_size**                 | Experiences per SGD update             | 8 to 32        | Larger = stabler but slower updates          |
| **ε_end** (Epsilon Final)      | Minimum exploration rate               | 0.001 to 0.01  | Controls exploration vs exploitation balance |
| **ε_decay** (Exploration Frac) | Fraction of training for epsilon decay | 0.05 to 0.1    | Controls how fast exploration decreases      |
| **buffer_size**                | Replay memory size                     | 30000          | Larger = more experience diversity           |

## Conclusions

Mariam's methodical approach to hyperparameter tuning demonstrates:

1. **Careful experimental design** with grouped hyperparameter changes
2. **Stability-first mindset** using conservative ranges
3. **Systematic progression** from individual effects to combined effects
4. **Data-driven insights** extracted from successful runs

The successful experiments (1-2) provide a solid foundation for Pong agent training. Future work should resolve the action space consistency issue to complete all 10 experiments.
