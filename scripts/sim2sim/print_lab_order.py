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
# Match eval_amp_run.py: disable ALL domain randomization for a clean comparison.
cfg.push_enable = False
cfg.obs_noise_enable = False
cfg.pd_gain_random_enable = False
cfg.joint_pos_offset_enable = False
cfg.added_mass_enable = False
log("Domain randomization disabled (matches eval_amp_run.py).")
log("Creating env...")
env = gym.make(TASK, cfg=cfg)
log("Resetting env (spawn robot)...")
env.reset()
u = env.unwrapped
log("Env ready. Dumping joint/body order + action scaling:")
sys.stdout.flush()

# Action scaling as used by _apply_action: target = offset + scale * action
import torch  # noqa: E402
soft = u.robot.data.soft_joint_pos_limits[0]  # (num_joints, 2)
soft_lo = soft[:, 0].cpu().numpy()
soft_hi = soft[:, 1].cpu().numpy()
offset = (0.5 * (soft_hi + soft_lo))
scale = (soft_hi - soft_lo)
default_pos = u.robot.data.default_joint_pos[0].cpu().numpy()

print("=== ACTION_OFFSET_AND_SCALE (one line per joint) ===", flush=True)
print("# idx joint_name soft_lo soft_hi offset scale default_pos", flush=True)
for i, n in enumerate(u.robot.data.joint_names):
    print(f"{i}: {n} {soft_lo[i]:+.4f} {soft_hi[i]:+.4f} {offset[i]:+.4f} {scale[i]:.4f} {default_pos[i]:+.4f}", flush=True)
print("=== END_ACTION_SCALE ===", flush=True)


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

log("Dumping Isaac Lab's initial 109-dim policy observation (cmd_vel = 0.0)...")
# Force cmd_vel=0 so the comparison to sim2sim --cmd_vel 0.0 is apples-to-apples.
obs_dict, _ = env.reset()
u.velocity_commands[:] = 0.0
u.command_time_left[:] = 999.0
# Recompute obs with the overridden cmd (otherwise it still holds the pre-override random cmd)
obs_new = u._get_observations()
o = obs_new["policy"][0].detach().cpu().numpy()
h = u.robot.data.body_pos_w[0, u.ref_body_index, 2].item()
vw = u.robot.data.body_lin_vel_w[0, u.ref_body_index].detach().cpu().numpy()
qp = u.robot.data.joint_pos[0].detach().cpu().numpy()
qv = u.robot.data.joint_vel[0].detach().cpu().numpy()

print(f"=== INIT_STATE h={h:.4f} body_lin_vel_w=[{vw[0]:+.4f},{vw[1]:+.4f},{vw[2]:+.4f}] ===", flush=True)
print("=== INIT_JOINT_POS ===", flush=True)
for i, n in enumerate(u.robot.data.joint_names):
    print(f"{i}: {n} qpos={qp[i]:+.6f} qvel={qv[i]:+.6f}", flush=True)
print("=== INIT_OBS (109-dim, cmd_vel=0.0) ===", flush=True)
for i, x in enumerate(o):
    print(f"obs[{i}]={x:+.6f}", flush=True)
print("=== END_INIT_OBS ===", flush=True)

log("Done. Closing app.")
app.close()
