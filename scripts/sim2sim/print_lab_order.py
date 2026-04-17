"""Dump Isaac Lab's joint_names and body_names for RobotLab-Isaac-G1-AMP-Run-Direct-v0.

Run on RunPod:
    /isaac-sim/python.sh scripts/sim2sim/print_lab_order.py --headless
Paste the output back to sim2sim so JOINT_NAMES/KEY_BODY_NAMES can be aligned.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "RobotLab-Isaac-G1-AMP-Run-Direct-v0"
cfg = parse_env_cfg(TASK, num_envs=1)
env = gym.make(TASK, cfg=cfg)
env.reset()
u = env.unwrapped

print("=== JOINT_NAMES (Isaac Lab action / joint_pos / joint_vel order) ===")
for i, n in enumerate(u.robot.data.joint_names):
    print(f"{i}: {n}")
print()
print("=== BODY_NAMES (body_pos_w / body_quat_w index order) ===")
for i, n in enumerate(u.robot.data.body_names):
    print(f"{i}: {n}")
print()
print(f"=== REF_BODY_INDEX: {u.ref_body_index} ===")
print(f"=== KEY_BODY_INDEXES: {u.key_body_indexes} ===")
print("=== Key body names (in order used by obs) ===")
for k, idx in enumerate(u.key_body_indexes):
    print(f"key[{k}] -> body_names[{idx}] = {u.robot.data.body_names[idx]}")

app.close()
