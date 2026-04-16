# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from isaaclab.utils import configclass

from .g1_amp_env_cfg import G1AmpDanceEnvCfg

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class G1AmpRunEnvCfg(G1AmpDanceEnvCfg):
    """G1 AMP environment config for running task.

    Reward design:
      - AMP discriminator → style reward (running gait from reference data)
      - Env → task reward (velocity command tracking + regularization)
      - No env-side imitation — discriminator handles style enforcement

    Velocity command:
      - Random velocity sampled from [command_vel_min, command_vel_max]
      - Held for a random duration [command_duration_min, command_duration_max]
      - Policy learns to track ANY commanded speed, including 0 (stop)
      - At deployment: send any velocity sequence (e.g. 0→4→4→...→0)

    Motion speed-up:
      - motion_speed > 1.0 scales reference data to faster speeds
      - E.g. 1.3x turns 3.3 m/s peak → 4.3 m/s, gait pattern preserved
    """

    # --- motion data ---
    motion_file = os.path.join(MOTIONS_DIR, "g1_run_and_stand.npz")
    motion_speed: float = 1.0  # 1.0 = natural speed; increase if discriminator blocks high-speed running

    # --- episode ---
    episode_length_s = 20.0

    # --- observation spaces ---
    # Policy obs: base AMP obs (105) + root_vel_body (3) + target_vel (1) = 109
    observation_space = 109
    # AMP obs remains 105 (inherited): discriminator sees motion features only,
    # no velocity command → won't penalize speed generalization

    # --- velocity command with curriculum ---
    # Three speed bands: low [0, 1), mid [1, 3), high [3, 4] m/s
    # Curriculum: start conservative, linearly ramp to final distribution
    command_vel_min: float = 0.0
    command_vel_max: float = 4.0
    command_vel_low_cutoff: float = 1.0   # boundary between low/mid bands
    command_vel_high_cutoff: float = 3.0  # boundary between mid/high bands
    # Start distribution (curriculum_level = 0)
    command_prob_high_start: float = 0.15   # 15% sprint
    command_prob_mid_start: float = 0.35    # 35% jog
    # P(low) = 50% standing/start
    # Final distribution (curriculum_level = 1)
    command_prob_high_final: float = 0.45   # 45% sprint
    command_prob_mid_final: float = 0.30    # 30% jog
    # P(low) = 25% standing/start
    # Adaptive curriculum (based on EMA of normalized velocity tracking reward)
    # metric = mean(exp(-4*(v_wx - cmd)²)), range [0, 1], 1.0 = perfect tracking
    curriculum_advance_threshold: float = 0.5   # advance when tracking > 50%
    curriculum_retreat_threshold: float = 0.3    # retreat when tracking < 30% (standing-only ≈ 0.23)
    curriculum_step_size: float = 0.05           # level change per trigger
    curriculum_cooldown_steps: int = 20000       # resample calls (~1200 sim steps ≈ 1 full episode)
    command_duration_min: float = 3.0  # seconds
    command_duration_max: float = 7.0  # seconds

    # --- velocity tracking reward ---
    rew_velocity_tracking: float = 1.5      # exp(-4·(v_wx - cmd)²), world X

    # --- shared rewards (always active) ---
    rew_upright: float = 1.0                # body +Z projected to world +Z
    rew_action_rate: float = -0.1           # smooth actions
    target_base_height: float = 0.75        # G1 pelvis standing height

    # --- running rewards (scaled by run_scale, active when speed > 1 m/s) ---
    rew_heading_run: float = -1.0           # face world +X, small-angle correction
    rew_lateral_vel_run: float = -0.5       # body-frame crab-walking
    rew_base_height_run: float = -3.0       # penalize squatting while running

    # --- standing rewards (scaled by stand_scale, active when speed < 1 m/s) ---
    rew_standing_height: float = -2.0       # penalize squatting (gentle start, was -10 causing collapse)
    rew_standing_still: float = -0.01       # no joint jitter (gentle)
    rew_yaw_rate_stand: float = -0.3        # no spinning in place (gentle)
    rew_heading_stand: float = -0.5         # face +X when stopped

    # --- disable env-side imitation (discriminator handles style) ---
    rew_imitation_pos = 0.0
    rew_imitation_rot = 0.0
    rew_imitation_joint_pos = 0.0
    rew_imitation_joint_vel = 0.0

    # --- domain randomization ---
    # Push perturbation
    push_enable: bool = True
    push_interval_min: float = 3.0    # seconds between pushes
    push_interval_max: float = 7.0
    push_force_min: float = 30.0      # Newtons (gentle)
    push_force_max: float = 100.0     # Newtons (moderate)
    # Observation noise
    obs_noise_enable: bool = True
    obs_noise_std: float = 0.02       # small Gaussian noise on observations
    # PD gain randomization (per-env, at reset)
    pd_gain_random_enable: bool = True
    pd_gain_random_range: float = 0.2  # ±20% variation
    # Initial joint position offset (at reset)
    joint_pos_offset_enable: bool = True
    joint_pos_offset_std: float = 0.05  # radians (~3°)
    # Added mass on torso (simulates payload uncertainty)
    added_mass_enable: bool = True
    added_mass_range: float = 2.0      # ±2 kg on torso

    # --- termination ---
    termination_height = 0.25  # lowered from 0.4 to give more learning time

    # --- reset ---
    reset_strategy = "default"  # start from standing pose; "random" was causing instant falls
