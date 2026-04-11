# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate AMP-Run checkpoint with configurable velocity command.

USAGE (RunPod):
  # Fixed 4 m/s, 1 robot, full episode
  /isaac-sim/python.sh scripts/reinforcement_learning/skrl/eval_amp_run.py \
    --checkpoint <path_to_agent.pt> --cmd_vel 4.0

  # Ramp 0→4→0, 2 robots
  /isaac-sim/python.sh scripts/reinforcement_learning/skrl/eval_amp_run.py \
    --checkpoint <path_to_agent.pt> --cmd_vel ramp --num_envs 2

  # Random commands (same as training)
  /isaac-sim/python.sh scripts/reinforcement_learning/skrl/eval_amp_run.py \
    --checkpoint <path_to_agent.pt> --cmd_vel random
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate AMP-Run with configurable velocity.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent checkpoint (.pt)")
parser.add_argument("--cmd_vel", type=str, default="4.0",
                    help="Velocity command: float (fixed), 'ramp' (0→4→0), or 'random' (training distribution)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--video_length", type=int, default=1200, help="Video length in steps (1200=20s at 60Hz)")
parser.add_argument("--output", type=str, default=None, help="Output video path (default: auto)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import skrl
import torch
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}.")
    exit()

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

TASK = "RobotLab-Isaac-G1-AMP-Run-Direct-v0"
agent_cfg_entry_point = "skrl_amp_cfg_entry_point"


@hydra_task_config(TASK, agent_cfg_entry_point)
def main(env_cfg: DirectRLEnvCfg, experiment_cfg: dict):
    """Evaluate AMP-Run with configurable velocity."""

    # --- parse cmd_vel mode ---
    cmd_vel_mode = args_cli.cmd_vel
    fixed_vel = None
    if cmd_vel_mode not in ("ramp", "random"):
        try:
            fixed_vel = float(cmd_vel_mode)
            cmd_vel_mode = "fixed"
        except ValueError:
            print(f"[ERROR] Invalid --cmd_vel: {cmd_vel_mode}. Use a float, 'ramp', or 'random'.")
            return

    # --- env config ---
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # for fixed/ramp mode, override command range to prevent random resampling interfering
    if cmd_vel_mode == "fixed":
        env_cfg.command_vel_min = fixed_vel
        env_cfg.command_vel_max = fixed_vel
        print(f"[EVAL] Fixed velocity: {fixed_vel} m/s")
    elif cmd_vel_mode == "ramp":
        # we'll manually override commands each step, but set range wide
        env_cfg.command_vel_min = 0.0
        env_cfg.command_vel_max = 4.0
        print("[EVAL] Ramp velocity: 4 m/s for 15s → 0 m/s for 5s")
    else:
        print("[EVAL] Random velocity commands (training distribution)")

    # --- output path ---
    output_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    output_dir = os.path.join(os.path.dirname(output_dir), "videos", "eval")
    os.makedirs(output_dir, exist_ok=True)

    if args_cli.output:
        output_path = args_cli.output
    else:
        ckpt_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
        output_path = os.path.join(output_dir, f"eval_{ckpt_name}_{cmd_vel_mode}.mp4")

    # --- enlarge ground plane for long-distance running ---
    env_cfg.scene.env_spacing = 100.0  # default 4.0, need ~60m for 15s at 4m/s

    # --- create env ---
    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")

    video_kwargs = {
        "video_folder": output_dir,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
        "name_prefix": f"eval_{cmd_vel_mode}",
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # --- load agent ---
    env_wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env_wrapped, experiment_cfg)

    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    runner.agent.load(os.path.abspath(args_cli.checkpoint))
    runner.agent.set_running_mode("eval")

    # --- get raw env for command override ---
    raw_env = env.unwrapped

    # --- run evaluation ---
    obs, _ = env_wrapped.reset()
    total_steps = args_cli.video_length
    reward_sum = torch.zeros(args_cli.num_envs, device=raw_env.device)

    # --- CSV log file ---
    ckpt_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    csv_path = os.path.join(output_dir, f"eval_{ckpt_name}_{cmd_vel_mode}.csv")
    csv_file = open(csv_path, "w")
    csv_file.write("step,time_s,cmd_vel,fwd_vel,lateral_vel,pelvis_height,reward_step\n")

    from isaaclab.utils.math import quat_rotate_inverse

    print(f"[EVAL] Running {total_steps} steps ({total_steps / 60:.1f}s at 60Hz)...")
    print(f"[EVAL] Logging to: {csv_path}")

    for step in range(total_steps):
        # --- override velocity command for fixed/ramp modes ---
        if cmd_vel_mode == "fixed":
            raw_env.velocity_commands[:] = fixed_vel
            raw_env.command_time_left[:] = 999.0  # prevent resampling
        elif cmd_vel_mode == "ramp":
            t_sec = step / 60.0  # current time in seconds (60Hz)
            if t_sec < 15.0:      # 0-15s: command 4 m/s (robot accelerates itself)
                vel = 4.0
            else:                 # 15-20s: command 0 m/s (robot decelerates itself)
                vel = 0.0
            raw_env.velocity_commands[:] = vel
            raw_env.command_time_left[:] = 999.0

        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rewards, _, _, _ = env_wrapped.step(actions)
            reward_sum += rewards.squeeze()

        # --- camera tracking: behind-and-above, looking down at robot ---
        if hasattr(raw_env, 'robot'):
            robot_pos = raw_env.robot.data.body_pos_w[0, raw_env.ref_body_index].cpu().numpy()
            # Behind the robot, slightly above, looking down — ground always visible
            cam_eye = [robot_pos[0] - 4.0, robot_pos[1] - 1.0, 2.5]
            cam_target = [robot_pos[0] + 2.0, robot_pos[1], 0.0]
            raw_env.sim.set_camera_view(cam_eye, cam_target)

        # --- log every step to CSV ---
        fwd_vel = 0.0
        lat_vel = 0.0
        pelvis_h = 0.0
        if hasattr(raw_env, 'robot'):
            root_quat = raw_env.robot.data.body_quat_w[0, raw_env.ref_body_index]
            root_vel_w = raw_env.robot.data.body_lin_vel_w[0, raw_env.ref_body_index]
            vel_body = quat_rotate_inverse(root_quat.unsqueeze(0), root_vel_w.unsqueeze(0)).squeeze(0)
            fwd_vel = vel_body[0].item()
            lat_vel = vel_body[1].item()
            pelvis_h = raw_env.robot.data.body_pos_w[0, raw_env.ref_body_index, 2].item()

        cmd = raw_env.velocity_commands[0].item() if hasattr(raw_env, 'velocity_commands') else 0
        step_reward = rewards[0].item() if rewards.dim() > 0 else rewards.item()
        t_sec = step / 60.0

        csv_file.write(f"{step},{t_sec:.3f},{cmd:.3f},{fwd_vel:.3f},{lat_vel:.3f},{pelvis_h:.3f},{step_reward:.4f}\n")

        # --- print to terminal every 60 steps (1s) ---
        if (step + 1) % 60 == 0 or step == total_steps - 1:
            print(f"  t={t_sec:.1f}s | cmd={cmd:.1f} m/s | fwd={fwd_vel:.2f} m/s | lat={lat_vel:.2f} | h={pelvis_h:.2f}m | rew={reward_sum[0].item():.1f}")

    csv_file.close()

    # --- summary ---
    print(f"\n[RESULT] Total reward: {reward_sum.mean().item():.1f}")
    print(f"[RESULT] CSV saved to: {csv_path}")
    print(f"[RESULT] Video saved to: {output_dir}/")

    # --- find and rename video ---
    import glob
    videos = sorted(glob.glob(os.path.join(output_dir, "*.mp4")))
    if videos and args_cli.output:
        os.rename(videos[-1], args_cli.output)
        print(f"[RESULT] Renamed to: {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
