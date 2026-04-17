"""Dump Isaac Lab's joint_names and body_names for RobotLab-Isaac-G1-AMP-Run-Direct-v0.

Run on RunPod:
    /isaac-sim/python.sh scripts/sim2sim/print_lab_order.py --headless
Paste the output back to sim2sim so JOINT_NAMES/KEY_BODY_NAMES can be aligned.
"""

import argparse
import sys

def log(msg):
    print(f"[print_lab_order] {msg}", flush=True)

log("Launching Isaac Sim app (cold start ~30-60s)...")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

log("AppLauncher ready. Importing modules...")

import gymnasium as gym  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "RobotLab-Isaac-G1-AMP-Run-Direct-v0"
log(f"Parsing env cfg for {TASK}...")
cfg = parse_env_cfg(TASK, num_envs=1)
log("Creating env...")
env = gym.make(TASK, cfg=cfg)
log("Resetting env (spawn robot)...")
env.reset()
u = env.unwrapped
log("Env ready. Dumping joint/body order:")
sys.stdout.flush()

print("=== JOINT_NAMES (Isaac Lab action / joint_pos / joint_vel order) ===", flush=True)
for i, n in enumerate(u.robot.data.joint_names):
    print(f"{i}: {n}", flush=True)
print(flush=True)
print("=== BODY_NAMES (body_pos_w / body_quat_w index order) ===", flush=True)
for i, n in enumerate(u.robot.data.body_names):
    print(f"{i}: {n}", flush=True)
print(flush=True)
print(f"=== REF_BODY_INDEX: {u.ref_body_index} ===", flush=True)
print(f"=== KEY_BODY_INDEXES: {u.key_body_indexes} ===", flush=True)
print("=== Key body names (in order used by obs) ===", flush=True)
for k, idx in enumerate(u.key_body_indexes):
    print(f"key[{k}] -> body_names[{idx}] = {u.robot.data.body_names[idx]}", flush=True)

log("Done. Closing app.")
app.close()
