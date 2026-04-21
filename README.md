# G1 Running: BeyondMimic vs AMP Comparison

Training a Unitree G1 (29-DOF) humanoid robot to complete a **full running cycle** (stand -> accelerate -> sustained running -> decelerate -> stop) using two motion imitation architectures, based on [robot_lab](https://github.com/fan-ziqi/robot_lab).

## Current Status (2026-04-17)

**AMP-Run Route B — ✅ milestone achieved.** After 7 tuning phases (see `docs/amp_run_training_log.md`), the Phase 7 policy (heading -10 cos reward) tracks **4 m/s cruise with 0.043 m/s error** and reproduces in MuJoCo sim-to-sim within 0.5 m/s.
- Latest checkpoint: `outputs/run7_phase7_latest_20k/checkpoints/agent_20000.pt`
- Sim-to-sim validation: [`docs/sim2sim_validation.md`](docs/sim2sim_validation.md)

### Demo — Phase 7 ramp command (0 → 4 → 0 m/s)

**Isaac Lab evaluation** (training environment)

https://github.com/Jake-Liu-Robot/Unitree_G1_High-speed-Running/raw/main/docs/media/isaac_lab_ramp.mp4

**MuJoCo sim-to-sim** (same exported policy, independent physics)

https://github.com/Jake-Liu-Robot/Unitree_G1_High-speed-Running/raw/main/docs/media/mujoco_sim2sim_ramp.mp4

Known issues (next-iteration targets): lateral drift during 8–12 s cruise, single-shock at deceleration step. See `docs/amp_run_training_log.md` §17.8.

**BeyondMimic Route A — planned.** Code is in the repo but not yet trained.

## Two Architectures

### BeyondMimic (Trajectory Tracking)

Frame-by-frame trajectory imitation with anchor-relative tracking. The policy learns to precisely reproduce a reference motion clip.

- **RL framework**: rsl_rl (PPO)
- **Workflow**: Manager-Based (Isaac Lab)
- **Reward**: Exponential tracking error on 14 body positions, orientations, and velocities
- **Key innovation**: Anchor-relative coordinates tolerate XY drift; adaptive sampling focuses training on difficult segments
- **Task**: `RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0`

### AMP (Adversarial Motion Priors)

Style-based imitation via a learned discriminator + velocity command tracking. The discriminator learns "what running looks like" while random velocity commands drive the policy to track any speed from 0 to 4 m/s.

- **RL framework**: skrl (PPO + AMP)
- **Workflow**: Direct (Isaac Lab)
- **Reward**: `0.5 x task_reward + 0.5 x style_reward` (rebalanced in Phase 4 from 0.7/0.3)
  - Task: velocity command tracking (random [0, 4] m/s) + regularization (upright, height, lateral, yaw, action rate)
  - Style: discriminator trained on reference running data (3-frame observation history)
  - No env-side imitation reward — discriminator handles all style enforcement
- **Key design**: Observation split — policy sees velocity command (109-dim), discriminator does NOT (105-dim), preventing speed-constrained style enforcement
- **Key advantage**: Generalizes beyond reference speed; can cruise at any speed for any duration
- **Task**: `RobotLab-Isaac-G1-AMP-Run-Direct-v0`

## Architecture Comparison

| | BeyondMimic | AMP-Run |
|--|-------------|---------|
| Imitation type | Frame-aligned trajectory | Style distribution (discriminator) |
| Speed control | Fixed (follows reference) | Controllable (random velocity commands) |
| Generalization | None (reproduces reference only) | Any speed [0, 4] m/s, any duration |
| Networks | Actor + Critic (asymmetric) | Actor + Critic + Discriminator |
| RL library | rsl_rl (SGD) | skrl (Adam) |

## Motion Data

**Source**: [LAFAN1 Retargeted for G1](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (30 FPS)

**Selected clip**: `run2_subject1.csv` frames 1943-2564 (20.7s full cycle)
- Stand (0.8s) -> Accelerate -> Sustained running at 2-3.3 m/s (18s) -> Decelerate -> Stand (0.5s)

**⚠️ Preprocessing pipeline** (raw LAFAN1 data is not used directly):

| Step | Purpose | Effect |
|------|---------|--------|
| **Trajectory straightening** | Reference clip had ~113° yaw drift over 20s; discriminator would learn to reward turning | Remove per-frame world yaw so forward direction is constant +X; joint-level motion preserved |
| **L/R mirroring** | LAFAN1 subject has asymmetric gait; without mirroring, policy favors one leg | Duplicate trajectory with left/right joints swapped and lateral direction flipped |
| **Synthetic standing frames** | Reference has almost no standing frames; policy collapses when commanded to stop | Insert ~27% standing frames computed from G1 default pose (Pinocchio FK, pelvis_z=0.777) |

Final AMP reference file: `g1_run_and_stand.npz` — **1697 frames / 56.5 s** (73% running / 27% standing, both mirrored).
Both pipelines (BeyondMimic and AMP) use this same preprocessed data, converted to their respective NPZ formats.
See [`docs/amp_run_training_log.md`](docs/amp_run_training_log.md) §13–14 for the data-pipeline rationale and code refs.

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
    ├── g1_amp_run_env.py               # Running env (random vel cmds, no imitation)
    ├── g1_amp_run_env_cfg.py           # Running config (cmd_vel[0,4], obs=109)
    ├── agents/skrl_run_amp_cfg.yaml    # AMP training config (task=0.7, style=0.3)
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
2. **Can AMP-Run learn running style from 2-3 m/s data via discriminator?** (style transfer quality)
3. **Can AMP-Run generalize to 4 m/s via velocity commands?** (beyond-reference generalization)
4. **Can AMP-Run follow arbitrary velocity sequences (0→4→4→...→0)?** (command tracking)
5. **How do the two approaches compare on motion quality, robustness, and training efficiency?**

## Tech Stack

| Component | Choice |
|-----------|--------|
| Simulation | Isaac Lab 2.3.x + Isaac Sim 5.x (PhysX) |
| BeyondMimic RL | rsl_rl (PPO, SGD) |
| AMP RL | skrl (PPO + AMP, Adam) |
| Sim-to-Sim | MuJoCo + Menagerie G1 model |
| Motion Data | LAFAN1 Retargeted for G1 (30 FPS) |
| Robot | Unitree G1 29-DOF |

## Documentation Map

- [`CLAUDE.md`](CLAUDE.md) — Project setup, RunPod config, full code structure
- [`docs/amp_run_training_log.md`](docs/amp_run_training_log.md) — Complete tuning narrative (Run 1–7 + Phase 1–7)
- [`docs/sim2sim_validation.md`](docs/sim2sim_validation.md) — MuJoCo sim-to-sim alignment notes
- [`docs/reward_tuning_guide.md`](docs/reward_tuning_guide.md) — Methodology for reward adjustments
- [`outputs/README.md`](outputs/README.md) — Per-run checkpoint + eval archive (local only)

## References

- [robot_lab](https://github.com/fan-ziqi/robot_lab) - Base project
- [BeyondMimic (whole_body_tracking)](https://github.com/HybridRobotics/whole_body_tracking) - BeyondMimic reference
- [AMP (Peng et al. 2021)](https://arxiv.org/abs/2104.02180) - Adversarial Motion Priors
- [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) - Motion data
