# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_rotate_inverse

from .g1_amp_env import G1AmpEnv, compute_rewards, exp_reward_with_floor
from .g1_amp_run_env_cfg import G1AmpRunEnvCfg


class G1AmpRunEnv(G1AmpEnv):
    """G1 AMP environment for running task.

    Adds forward velocity tracking reward on top of the base AMP environment.
    The AMP discriminator provides style reward (running gait from reference data).
    The velocity reward drives the robot toward a target forward speed.
    """

    cfg: G1AmpRunEnvCfg

    def _get_rewards(self) -> torch.Tensor:
        # ================= velocity tracking reward ==========================
        # Get root velocity in body frame
        root_quat_w = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_vel_w = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_vel_b = quat_rotate_inverse(root_quat_w, root_vel_w)
        forward_vel = root_vel_b[:, 0]  # x-axis = forward in body frame

        rew_velocity = self.cfg.rew_velocity_tracking * torch.exp(
            -4.0 * torch.square(forward_vel - self.cfg.target_velocity)
        )

        # ================= imitation reward (reduced weights) ================
        with torch.no_grad():
            current_times = (self.episode_length_buf * self.physics_dt).cpu().numpy()
            (
                ref_dof_positions,
                ref_dof_velocities,
                ref_body_positions,
                ref_body_rotations,
                _,
                _,
            ) = self._motion_loader.sample(num_samples=self.num_envs, times=current_times)

            ref_joint_pos = ref_dof_positions[:, self.motion_dof_indexes]
            ref_joint_vel = ref_dof_velocities[:, self.motion_dof_indexes]
            ref_root_pos = ref_body_positions[:, self.motion_ref_body_index]
            ref_root_quat = ref_body_rotations[:, self.motion_ref_body_index]

        # joint angle imitation
        joint_pos_error = torch.square(self.robot.data.joint_pos - ref_joint_pos).sum(dim=-1)
        rew_joint_pos = exp_reward_with_floor(
            joint_pos_error, self.cfg.rew_imitation_joint_pos, self.cfg.imitation_sigma_joint_pos, floor=4.0
        )
        rew_joint_pos = torch.clamp(rew_joint_pos, min=-1.0)

        # joint velocity imitation
        joint_vel_error = torch.square(self.robot.data.joint_vel - ref_joint_vel).sum(dim=-1)
        rew_joint_vel = exp_reward_with_floor(
            joint_vel_error, self.cfg.rew_imitation_joint_vel, self.cfg.imitation_sigma_joint_vel, floor=6.0
        )
        rew_joint_vel = torch.clamp(rew_joint_vel, min=-1.0)

        # root position imitation
        current_relative_pos = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
        pos_err = torch.square(current_relative_pos - ref_root_pos).sum(dim=-1)
        rew_pos = exp_reward_with_floor(pos_err, self.cfg.rew_imitation_pos, self.cfg.imitation_sigma_pos, floor=4.0)
        rew_pos = torch.clamp(rew_pos, min=-1.0)

        # root orientation imitation
        quat_dot = torch.abs(torch.sum(self.robot.data.body_quat_w[:, self.ref_body_index] * ref_root_quat, dim=-1))
        ang_err = 2 * torch.arccos(torch.clamp(quat_dot, -1.0, 1.0))
        rew_rot = self.cfg.rew_imitation_rot * torch.exp(-torch.square(ang_err) / (self.cfg.imitation_sigma_rot**2))

        imitation_reward = rew_joint_pos + rew_joint_vel + rew_pos + rew_rot

        # ================= basic reward (penalties) ==========================
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

        # ================= total reward ==========================
        total_reward = rew_velocity + imitation_reward + basic_reward

        # ============== log ================================
        log_dict = {
            "rew_velocity": rew_velocity.mean().item(),
            "forward_vel": forward_vel.mean().item(),
            "rew_imitation": imitation_reward.mean().item(),
            "rew_joint_pos": rew_joint_pos.mean().item(),
            "rew_joint_vel": rew_joint_vel.mean().item(),
            "rew_pos": rew_pos.mean().item(),
            "rew_rot": rew_rot.mean().item(),
            "total_reward": total_reward.mean().item(),
        }
        for key, value in basic_reward_log.items():
            log_dict[key] = value.mean().item() if isinstance(value, torch.Tensor) else float(value)

        self.extras["log"] = log_dict

        if hasattr(self, "_skrl_agent") and getattr(self, "_skrl_agent", None) is not None:
            try:
                agent = getattr(self, "_skrl_agent")
                for k, v in log_dict.items():
                    agent.track_data(f"Reward / {k}", v)
            except Exception:
                pass

        return total_reward
