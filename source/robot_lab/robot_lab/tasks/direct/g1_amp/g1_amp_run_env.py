# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply, quat_rotate_inverse

from .g1_amp_env import G1AmpEnv, compute_obs, compute_rewards
from .g1_amp_run_env_cfg import G1AmpRunEnvCfg


class G1AmpRunEnv(G1AmpEnv):
    """G1 AMP environment for running task.

    Design:
      - AMP discriminator enforces running style from reference motion data
      - Velocity command (random, resampled every few seconds) drives forward speed
      - No env-side imitation rewards — discriminator handles all style learning

    Observation split (key for velocity generalization):
      - Policy obs (109-dim): AMP obs (105) + root_vel_body (3) + cmd_vel (1)
        → policy knows its current speed and the target
      - AMP obs (105-dim): joint state + body pose + progress
        → discriminator does NOT see cmd_vel, so it won't penalize faster running

    Velocity command:
      - Random speed sampled from [cmd_vel_min, cmd_vel_max] every 3-7s
      - Policy learns to track any speed including 0 (stand) and 4 (sprint)
      - At deployment: send any velocity sequence (e.g. 0→4→4→...→0)
    """

    cfg: G1AmpRunEnvCfg

    def __init__(self, cfg: G1AmpRunEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Buffers
        self.previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.velocity_commands = torch.zeros(self.num_envs, device=self.device)
        self.command_time_left = torch.zeros(self.num_envs, device=self.device)
        # Initial heading direction (set at reset, used for heading reward)
        self.initial_heading_vec = torch.zeros(self.num_envs, 2, device=self.device)
        self.initial_heading_vec[:, 0] = 1.0  # default: facing +x

        # Speed up reference motion so discriminator learns faster gait
        if self.cfg.motion_speed != 1.0:
            s = self.cfg.motion_speed
            self._motion_loader.dt /= s
            self._motion_loader.duration = self._motion_loader.dt * (self._motion_loader.num_frames - 1)
            self._motion_loader.dof_velocities *= s
            self._motion_loader.body_linear_velocities *= s
            self._motion_loader.body_angular_velocities *= s
            print(
                f"[G1AmpRunEnv] Motion sped up {s:.1f}x: "
                f"duration {self._motion_loader.duration:.1f}s, "
                f"dt {self._motion_loader.dt:.4f}s"
            )

    # ------------------------------------------------------------------ #
    #  Velocity commands
    # ------------------------------------------------------------------ #

    def _resample_commands(self, env_ids: torch.Tensor):
        """Sample new velocity commands from biased speed bands.

        Three bands with configurable probabilities:
          high [high_cutoff, vel_max]  — 50% (sprint practice)
          mid  [low_cutoff, high_cutoff] — 30% (jogging)
          low  [vel_min, low_cutoff]   — 20% (standing/start)
        """
        n = len(env_ids)
        # Band selection
        roll = torch.rand(n, device=self.device)
        high_mask = roll < self.cfg.command_prob_high
        mid_mask = (~high_mask) & (roll < self.cfg.command_prob_high + self.cfg.command_prob_mid)
        low_mask = ~(high_mask | mid_mask)

        vel = torch.zeros(n, device=self.device)
        # High band: [high_cutoff, vel_max]
        n_high = high_mask.sum()
        if n_high > 0:
            vel[high_mask] = (
                self.cfg.command_vel_high_cutoff
                + (self.cfg.command_vel_max - self.cfg.command_vel_high_cutoff) * torch.rand(n_high, device=self.device)
            )
        # Mid band: [low_cutoff, high_cutoff]
        n_mid = mid_mask.sum()
        if n_mid > 0:
            vel[mid_mask] = (
                self.cfg.command_vel_low_cutoff
                + (self.cfg.command_vel_high_cutoff - self.cfg.command_vel_low_cutoff) * torch.rand(n_mid, device=self.device)
            )
        # Low band: [vel_min, low_cutoff]
        n_low = low_mask.sum()
        if n_low > 0:
            vel[low_mask] = (
                self.cfg.command_vel_min
                + (self.cfg.command_vel_low_cutoff - self.cfg.command_vel_min) * torch.rand(n_low, device=self.device)
            )

        self.velocity_commands[env_ids] = vel
        self.command_time_left[env_ids] = (
            self.cfg.command_duration_min
            + (self.cfg.command_duration_max - self.cfg.command_duration_min) * torch.rand(n, device=self.device)
        )

    # ------------------------------------------------------------------ #
    #  Pre-physics: tick command timer
    # ------------------------------------------------------------------ #

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        # Tick command timer (one env step = physics_dt * decimation)
        self.command_time_left -= self.physics_dt * self.cfg.decimation
        expired = self.command_time_left <= 0
        if expired.any():
            self._resample_commands(expired.nonzero(as_tuple=False).squeeze(-1))

    # ------------------------------------------------------------------ #
    #  Observations
    # ------------------------------------------------------------------ #

    def _get_observations(self) -> dict:
        # --- AMP obs (105-dim, identical to base class) ---
        progress = (self.episode_length_buf.squeeze(-1).float() / (self.max_episode_length - 1)).unsqueeze(-1)
        root_pos_relative = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
        key_body_pos_relative = (
            self.robot.data.body_pos_w[:, self.key_body_indexes] - self.scene.env_origins.unsqueeze(1)
        )

        amp_obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            root_pos_relative,
            self.robot.data.body_quat_w[:, self.ref_body_index],
            key_body_pos_relative,
            progress,
        )

        # --- update AMP observation buffer (discriminator input, 105-dim) ---
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = amp_obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        # --- policy obs (109-dim) = amp_obs(105) + root_vel_body(3) + cmd_vel(1) ---
        root_quat_w = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_vel_w = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_vel_b = quat_rotate_inverse(root_quat_w, root_vel_w)

        cmd_vel = self.velocity_commands.unsqueeze(-1)  # (num_envs, 1)

        policy_obs = torch.cat([amp_obs, root_vel_b, cmd_vel], dim=-1)
        return {"policy": policy_obs}

    # ------------------------------------------------------------------ #
    #  Rewards
    # ------------------------------------------------------------------ #

    def _get_rewards(self) -> torch.Tensor:
        # ================= velocity tracking ==========================
        root_quat_w = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_vel_w = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_vel_b = quat_rotate_inverse(root_quat_w, root_vel_w)
        forward_vel = root_vel_b[:, 0]
        lateral_vel = root_vel_b[:, 1]

        cmd_vel = self.velocity_commands
        rew_velocity = self.cfg.rew_velocity_tracking * torch.exp(
            -4.0 * torch.square(forward_vel - cmd_vel)
        )

        # ================= upright reward =============================
        up_ref = torch.zeros(self.num_envs, 3, device=self.device)
        up_ref[:, 2] = 1.0
        up_vec = quat_apply(root_quat_w, up_ref)
        rew_upright = self.cfg.rew_upright * up_vec[:, 2]

        # ================= base height penalty ========================
        pelvis_height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
        rew_base_height = self.cfg.rew_base_height * torch.square(
            pelvis_height - self.cfg.target_base_height
        )

        # ================= lateral velocity penalty ===================
        rew_lateral_vel = self.cfg.rew_lateral_vel * torch.square(lateral_vel)

        # ================= yaw rate penalty ===========================
        root_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.ref_body_index]
        yaw_rate = root_ang_vel_w[:, 2]
        rew_yaw_rate = self.cfg.rew_yaw_rate * torch.square(yaw_rate)

        # ================= action rate penalty ========================
        rew_action_rate = self.cfg.rew_action_rate * torch.sum(
            torch.square(self.actions - self.previous_actions), dim=-1
        )
        self.previous_actions = self.actions.clone()

        # ================= heading penalty (Plan C) ===================
        # Penalize deviation from initial facing direction
        forward_ref = torch.zeros(self.num_envs, 3, device=self.device)
        forward_ref[:, 0] = 1.0
        heading_vec = quat_apply(root_quat_w, forward_ref)  # current facing direction
        heading_xy = heading_vec[:, :2]  # project to XY plane
        # Dot product with initial heading (1.0 = aligned, 0 = 90° off)
        heading_dot = (heading_xy * self.initial_heading_vec).sum(dim=-1)
        rew_heading = self.cfg.rew_heading * (1.0 - heading_dot)  # 0 when aligned, negative when deviated

        # ================= standing joint penalty (Plan D) ============
        # Penalize joint velocity when ACTUAL speed is low (smooth standing)
        # Use actual forward velocity, not cmd — avoids harsh penalty during deceleration
        actual_speed = torch.abs(forward_vel)
        low_speed_scale = torch.clamp(1.0 - actual_speed, 0.0, 1.0)  # 1.0 when stopped, 0.0 when v>=1
        joint_vel_sq = torch.sum(torch.square(self.robot.data.joint_vel), dim=-1)
        rew_standing_still = self.cfg.rew_standing_still * low_speed_scale * joint_vel_sq

        # ================= basic penalties ============================
        basic_reward, basic_reward_log = compute_rewards(
            self.cfg.rew_termination,
            self.cfg.rew_action_l2,
            self.cfg.rew_joint_pos_limits,
            self.cfg.rew_joint_acc_l2,
            self.cfg.rew_joint_vel_l2,
            self.reset_terminated,
            self.actions,
            self.robot.data.joint_pos,
            self.robot.data.soft_joint_pos_limits,
            self.robot.data.joint_acc,
            self.robot.data.joint_vel,
        )

        # ================= total env reward ===========================
        total_reward = (
            rew_velocity
            + rew_upright
            + rew_base_height
            + rew_lateral_vel
            + rew_yaw_rate
            + rew_action_rate
            + rew_heading
            + rew_standing_still
            + basic_reward
        )

        # ================= logging ====================================
        log_dict = {
            "rew_velocity": rew_velocity.mean().item(),
            "forward_vel": forward_vel.mean().item(),
            "cmd_vel": cmd_vel.mean().item(),
            "rew_upright": rew_upright.mean().item(),
            "rew_base_height": rew_base_height.mean().item(),
            "rew_lateral_vel": rew_lateral_vel.mean().item(),
            "rew_yaw_rate": rew_yaw_rate.mean().item(),
            "rew_action_rate": rew_action_rate.mean().item(),
            "rew_heading": rew_heading.mean().item(),
            "rew_standing_still": rew_standing_still.mean().item(),
            "heading_dot": heading_dot.mean().item(),
            "total_reward": total_reward.mean().item(),
        }
        for key, value in basic_reward_log.items():
            log_dict[key] = value.mean().item() if isinstance(value, torch.Tensor) else float(value)

        self.extras["log"] = log_dict

        if hasattr(self, "_skrl_agent") and getattr(self, "_skrl_agent", None) is not None:
            try:
                agent = self._skrl_agent
                for k, v in log_dict.items():
                    agent.track_data(f"Reward / {k}", v)
            except Exception:
                pass

        return total_reward

    # ------------------------------------------------------------------ #
    #  Reset
    # ------------------------------------------------------------------ #

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.previous_actions[env_ids] = 0.0
        self._resample_commands(env_ids)
        # Record initial heading for heading reward
        root_quat = self.robot.data.body_quat_w[env_ids, self.ref_body_index]
        forward_ref = torch.zeros(len(env_ids), 3, device=self.device)
        forward_ref[:, 0] = 1.0
        heading = quat_apply(root_quat, forward_ref)
        self.initial_heading_vec[env_ids] = heading[:, :2]
        # Normalize
        heading_norm = torch.norm(self.initial_heading_vec[env_ids], dim=-1, keepdim=True).clamp(min=1e-6)
        self.initial_heading_vec[env_ids] /= heading_norm
