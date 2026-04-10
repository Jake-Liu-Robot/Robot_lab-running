# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from isaaclab.utils import configclass

from .g1_amp_env_cfg import G1AmpDanceEnvCfg

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class G1AmpRunEnvCfg(G1AmpDanceEnvCfg):
    """G1 AMP environment config for running task."""

    # --- motion data ---
    motion_file = os.path.join(MOTIONS_DIR, "g1_run2_subject1_30.npz")

    # --- episode ---
    episode_length_s = 20.0

    # --- velocity tracking reward ---
    target_velocity: float = 4.0
    rew_velocity_tracking: float = 1.5

    # --- reduce imitation weights (AMP discriminator handles style) ---
    rew_imitation_pos = 0.5
    rew_imitation_rot = 0.25
    rew_imitation_joint_pos = 1.0
    rew_imitation_joint_vel = 0.5

    # --- termination ---
    termination_height = 0.4

    # --- reset ---
    reset_strategy = "random"
