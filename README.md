# Task 11: Reinforcement Learning Controller

## Goal
Train a SAC (Soft Actor-Critic) RL agent using Stable Baselines 3 to move the OT-2 pipette tip to target positions within the robot's work envelope.

---

## Files
- `ot2_gym_wrapper.py` — Gymnasium environment wrapper defining the action space, observation space, reward function, and termination conditions
- `train_rl.py` — SAC training script with ClearML integration for remote execution on a GPU server
- `eval_rl.py` — Loads the saved model and evaluates final positioning error across multiple episodes
- `test_wrapper.py` — Runs the environment for 1000 steps with random actions to validate the API

---

## Environment Design

### Action Space
3 continuous actions in range [-1, 1], scaled to actual velocities:
- vx, vy scaled by `vel_limit_xy = 0.10 m/s`
- vz scaled by `vel_limit_z = 0.25 m/s`

### Observation Space
9 floats: current position (x, y, z), target position (x, y, z), error vector (ex, ey, ez)

### Reward
- Negative distance to target each step
- Small action penalty to reduce jitter (`action_penalty = 0.005`)
- Success bonus when within tolerance for `settle_steps` (`success_bonus = 2.0`)

### Termination
- Success: distance below `tol = 0.001 m` for 10 consecutive steps
- Truncation: after `max_steps = 400` steps

### Work Envelope
- X: -0.1870 → 0.2530
- Y: -0.1705 → 0.2195
- Z: 0.17 → 0.2795 (safe range)

---

## Hyperparameters
| Parameter | Value |
|---|---|
| Algorithm | SAC |
| Total timesteps | 150,000 |
| Learning rate | 1e-4 |
| Buffer size | 300,000 |
| Batch size | 512 |
| Tau | 0.01 |
| Gamma | 0.98 |
| Train frequency | 4 |
| Gradient steps | 4 |
| Learning starts | 8,000 |
| Network architecture | [512, 512] |

---

## How to Run

### Requirements
```bash
pip install stable-baselines3[extra] gymnasium clearml numpy pybullet
```

### File structure
All files should be in the same directory:
```text
/your_folder/
  sim_class.py
  ot2_gym_wrapper.py
  train_rl.py
  eval_rl.py
  test_wrapper.py
  textures/
```

### Submit training job via ClearML
```bash
python train_rl.py
```
The script will enqueue itself to the ClearML GPU queue and terminate locally. Monitor progress at https://app.clear.ml.

### Evaluate trained model
```bash
python eval_rl.py
```

---

## Results
> ⚠️ **Note: The training code has not been successfully tested yet due to GPU server access issues during development. The ClearML job was submitted but remained in a pending state as no agent was available to pick it up. Results will be filled in once training has completed successfully.**

| Metric | Value |
|---|---|
| Mean final error (m) | TBD |
| Worst final error (m) | TBD |
| Success rate | TBD |
| Mean points (client rubric) | TBD |

---

## Client Scoring Rubric
| Positioning Error | Points |
|---|---|
| < 0.001 m (1 mm) | 8 |
| 0.001 m ≤ error < 0.005 m | 6 |
| 0.005 m ≤ error < 0.01 m | 4 |
| ≥ 0.01 m | 0 |
