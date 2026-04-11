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
    motion_file = os.path.join(MOTIONS_DIR, "g1_run2_subject1_30.npz")
    motion_speed: float = 1.0  # 1.0 = natural speed; increase if discriminator blocks high-speed running

    # --- episode ---
    episode_length_s = 20.0

    # --- observation spaces ---
    # Policy obs: base AMP obs (105) + root_vel_body (3) + target_vel (1) = 109
    observation_space = 109
    # AMP obs remains 105 (inherited): discriminator sees motion features only,
    # no velocity command → won't penalize speed generalization

    # --- velocity command (conservative start) ---
    # Three speed bands: low [0, 1), mid [1, 3), high [3, 4] m/s
    # Start conservative — robot must learn to stand/walk before sprinting
    # Increase prob_high later once forward_vel shows tracking ability
    command_vel_min: float = 0.0
    command_vel_max: float = 4.0
    command_vel_low_cutoff: float = 1.0   # boundary between low/mid bands
    command_vel_high_cutoff: float = 3.0  # boundary between mid/high bands
    command_prob_high: float = 0.4        # P(high band [3, 4]) — increased, robot can run now
    command_prob_mid: float = 0.3         # P(mid band [1, 3]) — jogging
    # P(low band [0, 1]) = 1 - high - mid = 0.3 — standing/start
    command_duration_min: float = 3.0  # seconds
    command_duration_max: float = 7.0  # seconds

    # --- velocity tracking reward ---
    rew_velocity_tracking: float = 1.5

    # --- regularization rewards ---
    rew_upright: float = 0.5        # keep pelvis upright (z-up dot product), increased for better posture
    rew_base_height: float = -2.0   # penalize deviation from target height
    target_base_height: float = 0.75  # G1 pelvis height during running (~m)
    rew_lateral_vel: float = -0.5   # penalize sideways drift
    rew_yaw_rate: float = -0.5      # penalize spinning, increased to prevent direction drift
    rew_action_rate: float = -0.05  # penalize jerky actions

    # --- disable env-side imitation (discriminator handles style) ---
    rew_imitation_pos = 0.0
    rew_imitation_rot = 0.0
    rew_imitation_joint_pos = 0.0
    rew_imitation_joint_vel = 0.0

    # --- termination ---
    termination_height = 0.25  # lowered from 0.4 to give more learning time

    # --- reset ---
    reset_strategy = "default"  # start from standing pose; "random" was causing instant falls
