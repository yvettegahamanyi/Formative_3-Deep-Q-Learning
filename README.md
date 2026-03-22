# Formative 3: Deep Q-Learning (DQN) - Pong Atari

## Team Members:

- **Yvette Gahamanyi**
- **Mariam Awini Issah**
- **Dennis Mwai Kimiri**

## Project Overview

This project implements a Deep Q-Network (DQN) agent trained on the **Pong** Atari environment using Stable Baselines3 and Gymnasium. Each group member conducts 10 hyperparameter tuning experiments to optimize agent performance.

### Environment

- **Game**: Pong-v5 (ALE/Pong-v5)
- **Agent Type**: DQN with CNNPolicy
- **Objective**: Play Pong without losing (maximize score)

## Project Structure

```
├── train.py                  # Baseline training (saves dqn_model.zip)
├── play.py                   # Load a model and render gameplay
├── dqn_utils.py              # Shared env + evaluation helpers
├── Mariam/
│   ├── experiments.py        # 10 experiments + results + best model
│   └── models/               # saved experiment models
├── Yvette/
│   ├── experiments.py
│   └── models/
├── Dennis/
│   ├── experiments.py
│   └── models/
└── README.md                 # This file
```

## Scripts Description

### train.py

Trains a DQN agent on Pong environment with baseline hyperparameters:

```bash
python train.py
```

**Output**: `dqn_model.zip` (trained policy)

### play.py

Loads trained model and visualizes agent gameplay:

```bash
python play.py
```

**Output**: Real-time game rendering + performance statistics

### Member experiments (10 per member)

Each member runs **10 different hyperparameter combinations** and records results to a CSV inside their folder. Each script also copies their best experiment model to a single “best model” file for easy comparison.

- `Mariam/experiments.py` → `Mariam/results_mariam.csv`, `Mariam/best_mariam_model.zip`
- `Yvette/experiments.py` → `Yvette/results_yvette.csv`, `Yvette/best_yvette_model.zip`
- `Dennis/experiments.py` → `Dennis/results_dennis.csv`, `Dennis/best_dennis_model.zip`

## Hyperparameter Tuning Results

### Hyperparameters Tested (Per Member):

Each member conducted 10 experiments varying:

- **Learning Rate (lr)**: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
- **Gamma (γ)**: [0.90, 0.95, 0.99, 0.995]
- **Batch Size**: [16, 32, 64, 128]
- **Epsilon Final**: [0.001, 0.01, 0.05, 0.1]
- **Target Update Interval**: [500, 1000, 2000, 5000]
- **Buffer Size**: [30000, 50000, 100000]

### Results Table: All 30 Experiments

| Member                 | Exp# | lr | gamma | batch | epsilon_start | epsilon_end | epsilon_decay | Avg Reward | Max Reward | Avg Ep Len | Observed Behavior |
| ---------------------- | ---- | -- | ----- | ----- | ------------- | ----------- | ------------- | ---------- | ---------- | ---------- | ----------------- |
| **Yvette Gahamanyi**   | 1    |               |       |            |               |            |            |                   |
|                        | 2    |               |       |            |               |            |            |                   |
|                        | 3    |               |       |            |               |            |            |                   |
|                        | 4    |               |       |            |               |            |            |                   |
|                        | 5    |               |       |            |               |            |            |                   |
|                        | 6    |               |       |            |               |            |            |                   |
|                        | 7    |               |       |            |               |            |            |                   |
|                        | 8    |               |       |            |               |            |            |                   |
|                        | 9    |               |       |            |               |            |            |                   |
|                        | 10   |               |       |            |               |            |            |                   |
| **Mariam Awini Issah** | 1    |               |       |            |               |            |            |                   |
|                        | 2    |               |       |            |               |            |            |                   |
|                        | 3    |               |       |            |               |            |            |                   |
|                        | 4    |               |       |            |               |            |            |                   |
|                        | 5    |               |       |            |               |            |            |                   |
|                        | 6    |               |       |            |               |            |            |                   |
|                        | 7    |               |       |            |               |            |            |                   |
|                        | 8    |               |       |            |               |            |            |                   |
|                        | 9    |               |       |            |               |            |            |                   |
|                        | 10   |               |       |            |               |            |            |                   |
| **Dennis Mwai Kimiri** | 1    |               |       |            |               |            |            |                   |
|                        | 2    |               |       |            |               |            |            |                   |
|                        | 3    |               |       |            |               |            |            |                   |
|                        | 4    |               |       |            |               |            |            |                   |
|                        | 5    |               |       |            |               |            |            |                   |
|                        | 6    |               |       |            |               |            |            |                   |
|                        | 7    |               |       |            |               |            |            |                   |
|                        | 8    |               |       |            |               |            |            |                   |
|                        | 9    |               |       |            |               |            |            |                   |
|                        | 10   |               |       |            |               |            |            |                   |

### Key Findings

#### Most Impactful Hyperparameter Changes:

- **Learning Rate**: [To be filled]
- **Gamma**: [To be filled]
- **Batch Size**: [To be filled]
- **Exploration (Epsilon)**: [To be filled]

#### Best Configuration Per Member:

- **Yvette Gahamanyi**: Experiment #**_ (Avg Reward: _**)
- **Mariam Awini Issah**: Experiment #**_ (Avg Reward: _**)
- **Dennis Mwai Kimiri**: Experiment #**_ (Avg Reward: _**)

## Agent Performance

### Baseline Model Performance:

- Training Configuration: CNNPolicy, lr=1e-4, gamma=0.99, batch_size=32
- Final Average Reward: [To be filled]

### Best Performing Model:

- Configuration: [To be filled after experiments]
- Average Reward: [To be filled]
- Gameplay Video: [Link to video recording]

## Technical Stack

- **Framework**: Stable Baselines3 (RL library)
- **Environment**: Gymnasium with Atari (ALE)
- **Neural Network**: CNN for image-based agent
- **Language**: Python 3.x

## Dependencies

```bash
pip install stable-baselines3 gymnasium ale-py matplotlib pandas numpy
```

## How to run (commands)

### Baseline training (saves `dqn_model.zip`)

```bash
python train.py
```

Optional CNN vs MLP comparison:

```bash
python train.py --compare --timesteps 100000
```

### Member experiments (each runs 10 experiments)

```bash
python Mariam/experiments.py
python Yvette/experiments.py
python Dennis/experiments.py
```

### Play / record gameplay (loads `dqn_model.zip` by default)

```bash
python play.py
```

## Work Distribution

| Member             | Responsibilities              | Variation Strategy                      |
| ------------------ | ----------------------------- | --------------------------------------- |
| Yvette Gahamanyi   | 10 Hyperparameter Experiments | **Higher** parameters (faster learning) |
| Mariam Awini Issah | 10 Hyperparameter Experiments | **Lower** parameters (stability)        |
| Dennis Mwai Kimiri | 10 Hyperparameter Experiments | **Average/Balanced** parameters         |

**Collaboration Notes:**

- All members use shared `train.py` and `play.py`
- Independent experiment notebooks for each member
- Results consolidated in README table
- Group presentation (10 min total: 2 min/member + 4 min gameplay)

## DQN Hyperparameter Guide

| Parameter                  | Description          | Typical Range | Effect                               |
| -------------------------- | -------------------- | ------------- | ------------------------------------ |
| **lr**                     | Learning rate        | 1e-5 to 1e-3  | Controls speed of weight updates     |
| **gamma**                  | Discount factor      | 0.90 to 0.995 | Balances immediate vs future rewards |
| **batch_size**             | Samples per update   | 16 to 128     | Affects stability and speed          |
| **exploration_final_eps**  | Min exploration rate | 0.001 to 0.1  | Controls randomness in actions       |
| **target_update_interval** | Q-network sync freq  | 500 to 5000   | Stability of learning                |
| **buffer_size**            | Experience memory    | 30k to 100k   | Diversity of training samples        |

### Epsilon (ε) Parameter Mapping:

The assignment mentions **epsilon_start, epsilon_end, epsilon_decay**. In Stable Baselines3 DQN, these map to:

- **epsilon_start** (initial exploration) → `exploration_fraction=0.1`
  - Fraction of total timesteps where epsilon gradually decreases
- **epsilon_end** (final min exploration) → `exploration_final_eps=0.01`
  - Minimum exploration rate (probability of random action)
- **epsilon_decay** (decay rate) → Controlled by `exploration_fraction`
  - Higher fraction = slower decay, more exploration phase
  - Lower fraction = faster decay, earlier exploitation

**ε-greedy strategy**:

- With probability ε: take random action (explore)
- With probability 1-ε: take best known action (exploit)

## Policy Architecture Decision: MLP vs CNN

### CNNPolicy (Convolutional Neural Network) ✅ **RECOMMENDED**

- **Use for**: Image-based Atari games like Pong
- **Advantage**: Learns spatial features (edges, objects, patterns)
- **Why**: Visual agent needs to recognize game state from pixels
- **Performance**: Generally superior for visual environments

### MlpPolicy (Multilayer Perceptron)

- **Use for**: Flattened low-dimensional state spaces
- **Advantage**: Simpler, fewer parameters
- **Disadvantage**: Treats pixel input as flat vector - loses spatial information
- **Performance**: Typically underperforms on image-based games

**For Pong**: Use CNNPolicy for the agent to learn optimal paddle control!

## Lessons Learned

1. **Exploration vs Exploitation Trade-off**:
2. **Pong Challenge**:
3. **Hyperparameter Sensitivity**:
4. **CNN Advantage**:

## Presentation Details

- **Duration**: 10 minutes (2 min per member + 4 min gameplay)
- **Date**: [To be scheduled - Week 6]
- **Cameras**: MUST remain ON for entire presentation
- **Format**:
  - 2 minutes per member on experiments and insights
  - Group gameplay demo
  - Q&A session

### Recording Gameplay Video (for presentation)

Use `play.py` to record agent performance:

```bash
# Run play.py and record screen output
python play.py
```

**Recording Options:**

1. **OBS Studio** (Free)
   - Open Broadcaster Software - record screen while play.py runs
   - Output as MP4 for submission

2. **Screen Capture (Windows)**

   ```powershell
   # Built-in Windows recording
   # Press Win + G to open game bar
   # Press Win + Alt + R to start recording
   ```

3. **Python with cv2**
   - Modify play.py to save video frames directly
   - (Optional for advanced implementation)

**What to capture:**

- Agent playing Pong
- Multiple episodes showing learned behavior
- Reward accumulation per episode
- Duration: 30-60 seconds recommended

---

### Group Presentation Q&A Preparation

**Key Topics All Members Must Be Ready to Answer:**

#### 1. Understanding of DQN/RL Concepts

- **Q**: What is the exploration-exploration trade-off?
  - **A**: Balance between trying new actions (explore) vs using best known actions (exploit)
- **Q**: Why does gamma matter?
  - **A**: Higher gamma values future rewards more; lower gamma focuses on immediate rewards
- **Q**: What is the reward structure in Pong?
  - **A**: In Pong, the agent gets +1 for winning a point and -1 for losing a point; objective is to maximize score
- **Q**: How does the DQN agent learn?
  - **A**: Uses experience replay and target networks to learn Q-values from trial-and-error

#### 2. Hyperparameter Tuning Trade-offs

- **Q**: Why did you choose higher/lower/average parameters?
  - **A**: To test different learning speeds and stability levels
- **Q**: What trade-off did you observe?
  - **A**: Higher LR = faster learning but potential instability; Lower LR = stable but slow
- **Q**: Which parameter had the biggest impact?
  - **A**: [To be filled based on experiments]

#### 3. Model Behavior

- **Q**: Why does your final model behave the way it does?
  - **A**: [Describe learned navigation strategy based on experiments]
- **Q**: Did the agent learn to explore or exploit more?
  - **A**: [Based on epsilon_final and rewards collected]

#### 4. Policy Architecture

- **Q**: Why did you choose CNNPolicy over MlpPolicy?
  - **A**: CNN learns spatial features from pixels; essential for visual navigation task
- **Q**: What happens if you use MlpPolicy instead?
  - **A**: Agent treats flattened pixels as independent values - loses spatial information

---

## Assessment Rubric (30 points total)

| Criterion                                     | Points |
| --------------------------------------------- | ------ |
| Understanding of DQN/RL Concepts              | 10     |
| Hyperparameter Tuning & Documentation         | 5      |
| Evaluation & Agent Performance (play.py)      | 5      |
| Group Collaboration & Individual Contribution | 10     |
| **Total**                                     | **30** |

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [Atari Games](https://gymnasium.farama.org/environments/atari/)

## Submission Checklist

- [ ] Experiment 1-10 (Yvette) - Complete
- [ ] Experiment 1-10 (Mariam) - Complete
- [ ] Experiment 1-10 (Dennis) - Complete
- [ ] Results Table Filled - Empty cells populated
- [ ] Gameplay Video - Recorded from play.py
- [ ] README Updated - With all results
- [ ] GitHub Repository - All files pushed
- [ ] Coach Slot - Booked for Week 6
- [ ] zip file or URL - Ready for submission

---

**Last Updated**: March 16, 2026
**Status**: In Progress - Awaiting member experiments
