# G1 Running: BeyondMimic vs AMP Comparison

Training a Unitree G1 (29-DOF) humanoid robot to complete a **full running cycle** (stand -> accelerate -> sustained running -> decelerate -> stop) using two motion imitation architectures, based on [robot_lab](https://github.com/fan-ziqi/robot_lab).

## Two Architectures

### BeyondMimic (Trajectory Tracking)

Frame-by-frame trajectory imitation with anchor-relative tracking. The policy learns to precisely reproduce a reference motion clip.

- **RL framework**: rsl_rl (PPO)
- **Workflow**: Manager-Based (Isaac Lab)
- **Reward**: Exponential tracking error on 14 body positions, orientations, and velocities
- **Key innovation**: Anchor-relative coordinates tolerate XY drift; adaptive sampling focuses training on difficult segments
- **Task**: `RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0`

### AMP (Adversarial Motion Priors)

Style-based imitation via a learned discriminator + task-specific velocity reward. The discriminator learns "what running looks like" while the velocity reward pushes toward a target speed.

- **RL framework**: skrl (PPO + AMP)
- **Workflow**: Direct (Isaac Lab)
- **Reward**: `0.5 x task_reward + 0.5 x style_reward`
  - Task: forward velocity tracking (target: 4.0 m/s) + reduced imitation penalties
  - Style: discriminator trained on reference running data (3-frame observation history)
- **Key advantage**: Can potentially generalize beyond reference data speed
- **Task**: `RobotLab-Isaac-G1-AMP-Run-Direct-v0`

## Architecture Comparison

| | BeyondMimic | AMP-Run |
|--|-------------|---------|
| Imitation type | Frame-aligned trajectory | Style distribution |
| Speed control | Fixed (follows reference) | Controllable (velocity reward) |
| Generalization | None (reproduces reference only) | Can exceed reference speed |
| Networks | Actor + Critic (asymmetric) | Actor + Critic + Discriminator |
| RL library | rsl_rl (SGD) | skrl (Adam) |

## Motion Data

**Source**: [LAFAN1 Retargeted for G1](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (30 FPS)

**Selected clip**: `run2_subject1.csv` frames 1943-2564 (20.7s full cycle)
- Stand (0.8s) -> Accelerate -> Sustained running at 2-3.3 m/s (18s) -> Decelerate -> Stand (0.5s)
- Both pipelines use the same source data, converted to their respective NPZ formats

## Project Structure

```
source/robot_lab/robot_lab/tasks/
├── manager_based/beyondmimic/          # BeyondMimic pipeline
│   ├── tracking_env_cfg.py             # MDP definition (rewards, obs, terminations)
│   ├── mdp/
│   │   ├── commands.py                 # Motion loading + adaptive sampling
│   │   ├── rewards.py                  # Anchor-relative tracking rewards
│   │   ├── observations.py             # Asymmetric actor-critic observations
│   │   └── terminations.py             # Z-axis only termination (drift-tolerant)
│   └── config/g1/
│       ├── flat_env_cfg.py             # G1 running config
│       └── agents/rsl_rl_ppo_cfg.py    # PPO hyperparameters
│
└── direct/g1_amp/                      # AMP pipeline
    ├── g1_amp_env.py                   # Base AMP environment (dance)
    ├── g1_amp_run_env.py               # Running variant (+ velocity reward)
    ├── g1_amp_run_env_cfg.py           # Running config (target_vel=4.0)
    ├── agents/skrl_run_amp_cfg.yaml    # AMP training config
    └── motions/
        ├── motion_loader.py            # NPZ loader with interpolation
        └── csv2npz_run.py              # LAFAN1 CSV -> AMP NPZ converter
```

## Training

Requires RunPod with NGC Docker image `nvcr.io/nvidia/isaac-lab:2.3.2`.

```bash
# 1. Generate motion data
# BeyondMimic NPZ (requires Isaac Sim FK)
/isaac-sim/python.sh scripts/tools/beyondmimic/csv_to_npz.py \
  -f .../motion/run2_subject1.csv --input_fps 30 --frame_range 1943 2564 --headless

# AMP NPZ (requires Pinocchio FK)
/isaac-sim/python.sh source/robot_lab/robot_lab/tasks/direct/g1_amp/motions/csv2npz_run.py

# 2. Train BeyondMimic (~2-4 hours, 30k iterations)
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --headless --num_envs 4096

# 3. Train AMP-Run (~2-4 hours, 500k timesteps)
/isaac-sim/python.sh scripts/reinforcement_learning/skrl/train.py \
  --task=RobotLab-Isaac-G1-AMP-Run-Direct-v0 --algorithm AMP --headless --num_envs 4096
```

## Experimental Questions

1. **Can BeyondMimic reproduce the 2-3 m/s running cycle?** (trajectory tracking accuracy)
2. **Can AMP-Run match the running style from 2-3 m/s reference data?** (style transfer quality)
3. **Can AMP-Run generalize to 4 m/s using velocity reward?** (beyond-reference generalization)
4. **How do the two approaches compare on motion quality, robustness, and training efficiency?**

## Tech Stack

| Component | Choice |
|-----------|--------|
| Simulation | Isaac Lab 2.3.x + Isaac Sim 5.x (PhysX) |
| BeyondMimic RL | rsl_rl (PPO, SGD) |
| AMP RL | skrl (PPO + AMP, Adam) |
| Sim-to-Sim | MuJoCo + Menagerie G1 model |
| Motion Data | LAFAN1 Retargeted for G1 (30 FPS) |
| Robot | Unitree G1 29-DOF |

## References

- [robot_lab](https://github.com/fan-ziqi/robot_lab) - Base project
- [BeyondMimic (whole_body_tracking)](https://github.com/HybridRobotics/whole_body_tracking) - BeyondMimic reference
- [AMP (Peng et al. 2021)](https://arxiv.org/abs/2104.02180) - Adversarial Motion Priors
- [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) - Motion data
