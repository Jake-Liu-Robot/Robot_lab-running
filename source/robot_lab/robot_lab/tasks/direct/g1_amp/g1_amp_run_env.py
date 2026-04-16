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

        # Domain randomization buffers
        self.push_interval = torch.zeros(self.num_envs, device=self.device)
        self._randomize_push_timer(torch.arange(self.num_envs, device=self.device))
        # PD gain scale per env (1.0 = nominal, randomized at reset)
        self.pd_gain_scale = torch.ones(self.num_envs, self.cfg.action_space, device=self.device)

        # Foot body indices for gait phase reward
        self.left_foot_idx = self.robot.data.body_names.index("left_ankle_roll_link")
        self.right_foot_idx = self.robot.data.body_names.index("right_ankle_roll_link")

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
    #  Domain Randomization
    # ------------------------------------------------------------------ #

    def _randomize_push_timer(self, env_ids: torch.Tensor):
        """Randomize time until next push."""
        n = len(env_ids)
        self.push_interval[env_ids] = (
            self.cfg.push_interval_min
            + (self.cfg.push_interval_max - self.cfg.push_interval_min) * torch.rand(n, device=self.device)
        )

    def _apply_random_push(self, env_ids: torch.Tensor):
        """Apply random external force to pelvis."""
        n = len(env_ids)
        # Random force in XY plane
        force_mag = self.cfg.push_force_min + (
            self.cfg.push_force_max - self.cfg.push_force_min
        ) * torch.rand(n, device=self.device)
        angle = 2 * 3.14159 * torch.rand(n, device=self.device)
        forces = torch.zeros(n, 3, device=self.device)
        forces[:, 0] = force_mag * torch.cos(angle)
        forces[:, 1] = force_mag * torch.sin(angle)

        # Apply as velocity impulse (force * dt / mass ≈ velocity change)
        robot_mass = 50.0  # approximate G1 mass in kg
        vel_change = forces / robot_mass
        current_vel = self.robot.data.root_com_vel_w[env_ids, :3].clone()
        current_vel[:, :2] += vel_change[:, :2]
        self.robot.write_root_com_velocity_to_sim(
            torch.cat([current_vel, self.robot.data.root_com_vel_w[env_ids, 3:]], dim=-1),
            env_ids,
        )

    def _randomize_added_mass(self, env_ids: torch.Tensor):
        """Randomize torso mass at reset to simulate payload uncertainty."""
        n = len(env_ids)
        mass_range = self.cfg.added_mass_range
        # Random mass offset: uniform [-mass_range, +mass_range]
        mass_offset = 2 * mass_range * torch.rand(n, device=self.device) - mass_range

        # Get torso body index and modify mass
        torso_idx = self.robot.data.body_names.index("torso_link")
        # Read current masses and apply offset
        default_mass = self.robot.root_physx_view.get_body_masses()
        new_masses = default_mass[env_ids].clone()
        new_masses[:, torso_idx] += mass_offset
        new_masses[:, torso_idx] = torch.clamp(new_masses[:, torso_idx], min=1.0)  # prevent negative mass
        self.robot.root_physx_view.set_body_masses(new_masses, env_ids)

    # ------------------------------------------------------------------ #
    #  Velocity commands
    # ------------------------------------------------------------------ #

    def _resample_commands(self, env_ids: torch.Tensor):
        """Sample new velocity commands from fixed speed bands.

        Three bands: high [3,4] 30%, mid [1,3] 35%, low [0,1] 35%.
        """
        n = len(env_ids)
        prob_high = self.cfg.command_prob_high
        prob_mid = self.cfg.command_prob_mid
        # Band selection
        roll = torch.rand(n, device=self.device)
        high_mask = roll < prob_high
        mid_mask = (~high_mask) & (roll < prob_high + prob_mid)
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

        # 5% exact zero velocity for standing training (Round 1: minimal)
        zero_mask = torch.rand(n, device=self.device) < 0.05
        vel[zero_mask] = 0.0

        self.velocity_commands[env_ids] = vel
        self.command_time_left[env_ids] = (
            self.cfg.command_duration_min
            + (self.cfg.command_duration_max - self.cfg.command_duration_min) * torch.rand(n, device=self.device)
        )

    # ------------------------------------------------------------------ #
    #  Action application with PD gain randomization
    # ------------------------------------------------------------------ #

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        if self.cfg.pd_gain_random_enable:
            # Scale the action around offset to simulate PD gain variation
            target = self.action_offset + self.pd_gain_scale * (target - self.action_offset)
        self.robot.set_joint_position_target(target)

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

        # Domain randomization: random push
        if self.cfg.push_enable:
            dt = self.physics_dt * self.cfg.decimation
            self.push_interval -= dt
            push_envs = (self.push_interval <= 0).nonzero(as_tuple=False).squeeze(-1)
            if len(push_envs) > 0:
                self._apply_random_push(push_envs)
                self._randomize_push_timer(push_envs)

    # ------------------------------------------------------------------ #
    #  Observations
    # ------------------------------------------------------------------ #

    def _get_observations(self) -> dict:
        # --- AMP obs (105-dim, identical to base class) ---
        progress = (self.episode_length_buf.float() / (self.max_episode_length - 1)).view(-1, 1)
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

        # --- observation noise (domain randomization) ---
        if self.cfg.obs_noise_enable:
            noise = torch.randn_like(policy_obs) * self.cfg.obs_noise_std
            policy_obs = policy_obs + noise

        return {"policy": policy_obs}

    # ------------------------------------------------------------------ #
    #  Rewards
    # ------------------------------------------------------------------ #

    def _get_rewards(self) -> torch.Tensor:
        # ================= common quantities ==============================
        root_quat_w = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_vel_w = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.ref_body_index]
        root_vel_b = quat_rotate_inverse(root_quat_w, root_vel_w)
        pelvis_height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
        cmd_vel = self.velocity_commands

        forward_vel = root_vel_w[:, 0]       # world X velocity
        lateral_vel_body = root_vel_b[:, 1]   # body Y velocity (crab-walking)
        yaw_rate = root_ang_vel_w[:, 2]       # world Z angular velocity

        # heading: body +X projected to world +X = cos(yaw)
        forward_ref = torch.zeros(self.num_envs, 3, device=self.device)
        forward_ref[:, 0] = 1.0
        heading_vec = quat_apply(root_quat_w, forward_ref)
        heading_cos = heading_vec[:, 0]       # 1.0 = facing +X, 0.0 = 90° off

        # upright: body +Z projected to world +Z
        up_ref = torch.zeros(self.num_envs, 3, device=self.device)
        up_ref[:, 2] = 1.0
        up_vec = quat_apply(root_quat_w, up_ref)

        # speed-dependent scales
        speed = torch.abs(forward_vel)
        stand_scale = torch.clamp(1.0 - speed, 0.0, 1.0)   # 1 @ v=0, 0 @ v≥1
        run_scale = torch.clamp(speed - 1.0, 0.0, 1.0)      # 0 @ v≤1, 1 @ v≥2

        # ================= SHARED rewards (always active) =================

        # 1) velocity tracking (world X) — primary objective
        rew_velocity = self.cfg.rew_velocity_tracking * torch.exp(
            -4.0 * torch.square(forward_vel - cmd_vel)
        )

        # 2) upright — always keep pelvis vertical
        rew_upright = self.cfg.rew_upright * up_vec[:, 2]

        # 3) action rate — smooth actions (always, but running naturally has larger values)
        rew_action_rate = self.cfg.rew_action_rate * torch.sum(
            torch.square(self.actions - self.previous_actions), dim=-1
        )
        self.previous_actions = self.actions.clone()

        # 4) basic penalties — joint limits, action L2, etc (always)
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

        # ================= RUNNING rewards (scale with run_scale) =========

        # 5) heading — face world +X (small-angle correction signal)
        #    Provides gradient where velocity_tracking is flat (cos(θ) ≈ 1 for small θ)
        rew_heading = self.cfg.rew_heading_run * run_scale * (1.0 - heading_cos)

        # 6) lateral velocity (body frame) — prevent crab-walking
        rew_lateral_vel = self.cfg.rew_lateral_vel_run * run_scale * torch.square(lateral_vel_body)

        # 7) base height — penalize squatting at ALL speeds (removed run_scale gating
        #    to prevent policy from crouching during sub-1m/s acceleration phase)
        rew_base_height_run = self.cfg.rew_base_height_run * torch.square(
            pelvis_height - self.cfg.target_base_height
        )

        # ================= STANDING rewards (scale with stand_scale) ======

        # 8) standing height — must stand tall (h ≥ target)
        squat_error = torch.clamp(self.cfg.target_base_height - pelvis_height, min=0.0)
        rew_standing_height = self.cfg.rew_standing_height * stand_scale * torch.square(squat_error)

        # 9) standing still — no joint jitter
        joint_vel_sq = torch.sum(torch.square(self.robot.data.joint_vel), dim=-1)
        rew_standing_still = self.cfg.rew_standing_still * stand_scale * joint_vel_sq

        # 10) standing yaw rate — no spinning in place
        rew_yaw_rate_stand = self.cfg.rew_yaw_rate_stand * stand_scale * torch.square(yaw_rate)

        # 11) standing heading — face +X when stopped
        rew_heading_stand = self.cfg.rew_heading_stand * stand_scale * (1.0 - heading_cos)

        # ================= total env reward ===========================
        total_reward = (
            rew_velocity
            + rew_upright
            + rew_action_rate
            + basic_reward
            + rew_heading
            + rew_lateral_vel
            + rew_base_height_run
            + rew_standing_height
            + rew_standing_still
            + rew_yaw_rate_stand
            + rew_heading_stand
        )

        # ================= logging ====================================
        log_dict = {
            "rew_velocity": rew_velocity.mean().item(),
            "forward_vel": forward_vel.mean().item(),
            "cmd_vel": cmd_vel.mean().item(),
            "rew_upright": rew_upright.mean().item(),
            "rew_action_rate": rew_action_rate.mean().item(),
            "rew_heading_run": rew_heading.mean().item(),
            "rew_lateral_vel_run": rew_lateral_vel.mean().item(),
            "rew_base_height_run": rew_base_height_run.mean().item(),
            "rew_standing_height": rew_standing_height.mean().item(),
            "rew_standing_still": rew_standing_still.mean().item(),
            "rew_yaw_rate_stand": rew_yaw_rate_stand.mean().item(),
            "rew_heading_stand": rew_heading_stand.mean().item(),
            "heading_cos": heading_cos.mean().item(),
            "pelvis_height": pelvis_height.mean().item(),
            "total_reward": total_reward.mean().item(),
        }
        for key, value in basic_reward_log.items():
            log_dict[key] = value.mean().item() if isinstance(value, torch.Tensor) else float(value)

        self.extras["log"] = log_dict

        # Write per-term rewards to TensorBoard (requires train.py to attach _skrl_agent)
        if hasattr(self, "_skrl_agent") and getattr(self, "_skrl_agent", None) is not None:
            try:
                agent = getattr(self, "_skrl_agent")
                for k, v in log_dict.items():
                    agent.track_data(f"Reward / {k}", v)
            except Exception:
                pass

        # Print curriculum status every 1000 steps
        if self.common_step_counter % 1000 == 0:
            print(f"[STEP {self.common_step_counter}] "
                  f"ep={self.episode_length_buf.float().mean().item():.0f} "
                  f"fwd={forward_vel.mean().item():.2f} "
                  f"cmd={cmd_vel.mean().item():.2f} "
                  f"h={pelvis_height.mean().item():.3f} "
                  f"head={heading_cos.mean().item():.3f} "
                  f"rew={total_reward.mean().item():.2f}")

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
        # Reset push timer
        if self.cfg.push_enable:
            self._randomize_push_timer(env_ids)
        # Randomize PD gains per env
        if self.cfg.pd_gain_random_enable:
            r = self.cfg.pd_gain_random_range
            self.pd_gain_scale[env_ids] = 1.0 + (2 * r * torch.rand(len(env_ids), self.cfg.action_space, device=self.device) - r)
        # Randomize initial joint positions
        if self.cfg.joint_pos_offset_enable:
            offset = torch.randn(len(env_ids), self.cfg.action_space, device=self.device) * self.cfg.joint_pos_offset_std
            current_pos = self.robot.data.joint_pos[env_ids].clone()
            self.robot.write_joint_state_to_sim(current_pos + offset, self.robot.data.joint_vel[env_ids], None, env_ids)
        # Randomize torso mass
        if self.cfg.added_mass_enable:
            try:
                self._randomize_added_mass(env_ids)
            except Exception:
                pass  # PhysX API may not support per-env mass modification
