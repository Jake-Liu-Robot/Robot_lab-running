"""
MuJoCo sim-to-sim for AMP-Run G1 policy.

Loads exported policy (.pt from export_policy.py), runs in MuJoCo.
No dependency on Isaac Lab or skrl — pure PyTorch + MuJoCo.

USAGE (local machine):
    python scripts/sim2sim/sim2sim_mujoco.py \
        --policy policy_exported.pt \
        --xml_path <path_to_g1_29dof_rev_1_0.xml> \
        --cmd_vel 4.0

    # Ramp test
    python scripts/sim2sim/sim2sim_mujoco.py \
        --policy policy_exported.pt \
        --cmd_vel ramp

REQUIREMENTS:
    pip install mujoco torch numpy
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn


# ============================================================
#  Policy Network (matches skrl training architecture)
# ============================================================

class PolicyNetwork(nn.Module):
    """MLP policy [1024, 512] matching skrl GaussianMixin."""

    def __init__(self, obs_dim=109, act_dim=29, hidden_dims=(1024, 512)):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


class ObservationNormalizer:
    """Replicates skrl RunningStandardScaler for inference."""

    def __init__(self, mean, var, clip=5.0):
        self.mean = mean
        self.var = var
        self.clip = clip

    def normalize(self, obs):
        return torch.clamp((obs - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


# ============================================================
#  G1 Configuration (matches Isaac Lab exactly)
# ============================================================

# Joint names in Isaac Lab order (same as MuJoCo XML order)
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Key body names for observation (13 bodies)
KEY_BODY_NAMES = [
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "right_rubber_hand", "left_rubber_hand",
    "right_ankle_roll_link", "left_ankle_roll_link",
    "torso_link",
    "right_hip_yaw_link", "left_hip_yaw_link",
    "right_knee_link", "left_knee_link",
]

REF_BODY_NAME = "pelvis"

# PD gains per joint (from unitree.py)
# Format: {joint_name: (stiffness, damping)}
PD_GAINS = {}
# Motor type 7520-14: hip_pitch, hip_yaw, waist_yaw
for jn in ["left_hip_pitch_joint", "right_hip_pitch_joint",
           "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_yaw_joint"]:
    PD_GAINS[jn] = (40.18, 2.558)
# Motor type 7520-22: hip_roll, knee
for jn in ["left_hip_roll_joint", "right_hip_roll_joint",
           "left_knee_joint", "right_knee_joint"]:
    PD_GAINS[jn] = (99.10, 6.309)
# Motor type 5020 (2x): ankle, waist_roll, waist_pitch
for jn in ["left_ankle_pitch_joint", "right_ankle_pitch_joint",
           "left_ankle_roll_joint", "right_ankle_roll_joint",
           "waist_roll_joint", "waist_pitch_joint"]:
    PD_GAINS[jn] = (28.50, 1.814)
# Motor type 5020: shoulder, elbow, wrist_roll
for jn in ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
           "left_shoulder_roll_joint", "right_shoulder_roll_joint",
           "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
           "left_elbow_joint", "right_elbow_joint",
           "left_wrist_roll_joint", "right_wrist_roll_joint"]:
    PD_GAINS[jn] = (14.25, 0.907)
# Motor type 4010: wrist_pitch, wrist_yaw
for jn in ["left_wrist_pitch_joint", "right_wrist_pitch_joint",
           "left_wrist_yaw_joint", "right_wrist_yaw_joint"]:
    PD_GAINS[jn] = (16.78, 1.068)


# ============================================================
#  MuJoCo Model Setup
# ============================================================

def setup_mujoco_model(xml_path):
    """Load MuJoCo model and add PD actuators."""
    # Load XML and modify to add actuators
    with open(xml_path, "r") as f:
        xml_content = f.read()

    # Build actuator XML using <general> for proper PD control
    # force = kp * (ctrl - qpos) - kd * qvel
    actuator_xml = "  <actuator>\n"
    for jn in JOINT_NAMES:
        kp, kd = PD_GAINS[jn]
        actuator_xml += (
            f'    <general name="{jn}_actuator" joint="{jn}" '
            f'gainprm="{kp} 0 0" biasprm="0 -{kp} -{kd}" '
            f'ctrlrange="-3.14159 3.14159"/>\n'
        )
    actuator_xml += "  </actuator>"

    # Replace empty actuator section
    xml_content = xml_content.replace("  <actuator>\n  </actuator>", actuator_xml)

    # Write modified XML to temp file (same directory for mesh resolution)
    import os
    import tempfile
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    tmp_xml = os.path.join(xml_dir, "_sim2sim_temp.xml")
    with open(tmp_xml, "w") as f:
        f.write(xml_content)
    model = mujoco.MjModel.from_xml_path(tmp_xml)
    os.remove(tmp_xml)
    data = mujoco.MjData(model)

    # Set simulation timestep to match Isaac Lab (1/60 s)
    model.opt.timestep = 1.0 / 60.0

    return model, data, xml_content


def get_body_id(model, name):
    """Get MuJoCo body ID by name."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def get_joint_ids(model, joint_names):
    """Get MuJoCo joint qpos/qvel indices."""
    qpos_ids = []
    qvel_ids = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qpos_ids.append(model.jnt_qposadr[jid])
        qvel_ids.append(model.jnt_dofadr[jid])
    return qpos_ids, qvel_ids


# ============================================================
#  Observation Computation (matches Isaac Lab exactly)
# ============================================================

def quat_apply(quat_wxyz, vec):
    """Apply quaternion rotation to vector. Quat in [w,x,y,z] format."""
    w, x, y, z = quat_wxyz
    # Using quaternion rotation formula
    t = 2.0 * np.cross([x, y, z], vec)
    return vec + w * t + np.cross([x, y, z], t)


def quaternion_to_tangent_and_normal(quat_wxyz):
    """Convert quaternion to tangent+normal (6-dim), matching Isaac Lab."""
    ref_tangent = np.array([1.0, 0.0, 0.0])
    ref_normal = np.array([0.0, 0.0, 1.0])
    tangent = quat_apply(quat_wxyz, ref_tangent)
    normal = quat_apply(quat_wxyz, ref_normal)
    return np.concatenate([tangent, normal])  # 6-dim


def compute_observation(model, data, joint_qpos_ids, joint_qvel_ids,
                        ref_body_id, key_body_ids, cmd_vel, progress):
    """Compute 109-dim policy observation from MuJoCo state.

    Structure:
      joint_pos (29) + joint_vel (29) + root_height (1) +
      root_tangent_normal (6) + key_body_rel_pos (13×3=39) +
      progress (1) + root_vel_body (3) + cmd_vel (1) = 109
    """
    # Joint positions and velocities (29 each)
    joint_pos = np.array([data.qpos[i] for i in joint_qpos_ids], dtype=np.float32)
    joint_vel = np.array([data.qvel[i] for i in joint_qvel_ids], dtype=np.float32)

    # Root (pelvis) state
    # MuJoCo free joint: qpos[0:3]=pos, qpos[3:7]=quat(w,x,y,z)
    root_pos = data.xpos[ref_body_id].copy()
    root_quat_wxyz = data.xquat[ref_body_id].copy()  # MuJoCo uses [w,x,y,z]

    # Root height (1-dim)
    root_height = np.array([root_pos[2]], dtype=np.float32)

    # Root orientation → tangent + normal (6-dim)
    root_tangent_normal = quaternion_to_tangent_and_normal(root_quat_wxyz).astype(np.float32)

    # Key body positions relative to root (13×3 = 39-dim)
    key_body_rel = []
    for bid in key_body_ids:
        body_pos = data.xpos[bid].copy()
        rel_pos = body_pos - root_pos
        key_body_rel.append(rel_pos)
    key_body_rel = np.concatenate(key_body_rel).astype(np.float32)  # 39-dim

    # Progress (1-dim)
    progress_arr = np.array([progress], dtype=np.float32)

    # AMP obs (105-dim)
    amp_obs = np.concatenate([
        joint_pos, joint_vel, root_height,
        root_tangent_normal, key_body_rel, progress_arr
    ])

    # Root velocity in body frame (3-dim)
    # MuJoCo: cvel[ref_body] gives [angular(3), linear(3)] in world frame
    root_vel_world = data.cvel[ref_body_id][3:6].copy()  # linear velocity
    # Rotate world velocity to body frame using inverse quaternion
    # For [w,x,y,z]: inverse = [w,-x,-y,-z] (for unit quat)
    quat_inv = np.array([root_quat_wxyz[0], -root_quat_wxyz[1],
                         -root_quat_wxyz[2], -root_quat_wxyz[3]])
    root_vel_body = quat_apply(quat_inv, root_vel_world).astype(np.float32)

    # Command velocity (1-dim)
    cmd_vel_arr = np.array([cmd_vel], dtype=np.float32)

    # Full policy observation (109-dim)
    obs = np.concatenate([amp_obs, root_vel_body, cmd_vel_arr])
    return obs


# ============================================================
#  Action Scaling
# ============================================================

def compute_action_scaling(model, joint_qpos_ids):
    """Compute action offset and scale from joint limits (matches Isaac Lab)."""
    lower = np.array([model.jnt_range[model.jnt_qposadr == i, 0].item()
                      if (model.jnt_qposadr == i).any()
                      else -3.14 for i in joint_qpos_ids], dtype=np.float32)
    upper = np.array([model.jnt_range[model.jnt_qposadr == i, 1].item()
                      if (model.jnt_qposadr == i).any()
                      else 3.14 for i in joint_qpos_ids], dtype=np.float32)

    # Try reading directly from joint range
    jnt_ids = []
    for jn in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        jnt_ids.append(jid)

    lower = np.array([model.jnt_range[jid, 0] for jid in jnt_ids], dtype=np.float32)
    upper = np.array([model.jnt_range[jid, 1] for jid in jnt_ids], dtype=np.float32)

    action_offset = 0.5 * (upper + lower)
    action_scale = upper - lower
    return action_offset, action_scale


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MuJoCo sim-to-sim for AMP-Run G1")
    parser.add_argument("--policy", type=str, required=True, help="Exported policy .pt file")
    parser.add_argument("--xml_path", type=str,
                        default="/home/jake/Unitree_rl_gym/unitree_rl_gym/resources/robots/g1_description/g1_29dof_rev_1_0.xml",
                        help="Path to G1 MuJoCo XML")
    parser.add_argument("--cmd_vel", type=str, default="4.0",
                        help="Velocity command: float or 'ramp'")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration (seconds)")
    parser.add_argument("--no_render", action="store_true", help="Disable visualization")
    args = parser.parse_args()

    # --- Parse cmd_vel ---
    cmd_vel_mode = args.cmd_vel
    fixed_vel = 4.0
    if cmd_vel_mode != "ramp":
        fixed_vel = float(cmd_vel_mode)
        cmd_vel_mode = "fixed"

    # --- Load policy ---
    print(f"[INFO] Loading policy: {args.policy}")
    export = torch.load(args.policy, map_location="cpu")

    policy = PolicyNetwork(
        obs_dim=export["obs_dim"],
        act_dim=export["act_dim"],
        hidden_dims=export["hidden_dims"],
    )

    # Load weights (need to map skrl key names to our network)
    state_dict = {}
    policy_state = export["policy_state"]
    # skrl naming: "policy_net_0.weight" → our "net.0.weight"
    # net structure: Linear(109,1024), ReLU, Linear(1024,512), ReLU, Linear(512,29)
    # skrl indices:  net_0           ,     , net_2           ,     , net_4
    # our indices:   net.0           , net.1, net.2           , net.3, net.4
    key_mapping = {
        "policy_net_0.weight": "net.0.weight",
        "policy_net_0.bias": "net.0.bias",
        "policy_net_2.weight": "net.2.weight",
        "policy_net_2.bias": "net.2.bias",
        "policy_net_4.weight": "net.4.weight",
        "policy_net_4.bias": "net.4.bias",
    }
    for skrl_key, our_key in key_mapping.items():
        if skrl_key in policy_state:
            state_dict[our_key] = policy_state[skrl_key]
        else:
            print(f"  [WARN] Key {skrl_key} not found in checkpoint")

    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"  Policy loaded: {export['obs_dim']} → [{export['hidden_dims']}] → {export['act_dim']}")

    # Load observation normalizer
    normalizer = None
    prep = export.get("preprocessor_state", {})
    if prep:
        mean_key = [k for k in prep if "running_mean" in k]
        var_key = [k for k in prep if "running_variance" in k]
        if mean_key and var_key:
            mean = prep[mean_key[0]]
            var = prep[var_key[0]]
            normalizer = ObservationNormalizer(mean, var)
            print(f"  Normalizer loaded: mean shape={mean.shape}")

    # --- Setup MuJoCo ---
    print(f"[INFO] Loading MuJoCo model: {args.xml_path}")
    model, data, xml_str = setup_mujoco_model(args.xml_path)

    # Get IDs
    ref_body_id = get_body_id(model, REF_BODY_NAME)
    key_body_ids = [get_body_id(model, name) for name in KEY_BODY_NAMES]
    joint_qpos_ids, joint_qvel_ids = get_joint_ids(model, JOINT_NAMES)

    # Action scaling: prefer exported values from Isaac Lab, fallback to MuJoCo limits
    if export.get("action_offset") is not None:
        action_offset = export["action_offset"].numpy()
        action_scale = export["action_scale"].numpy()
        print("  Action scaling: from Isaac Lab export")
    else:
        action_offset, action_scale = compute_action_scaling(model, joint_qpos_ids)
        print("  Action scaling: from MuJoCo joint limits (fallback)")
    print(f"  Ref body: {REF_BODY_NAME} (id={ref_body_id})")
    print(f"  Key bodies: {len(key_body_ids)}")
    print(f"  Joints: {len(joint_qpos_ids)}")

    # --- Reset to standing ---
    mujoco.mj_resetData(model, data)
    # Free joint: qpos[0:3]=pos, qpos[3:7]=quat(w,x,y,z)
    data.qpos[2] = 0.75   # pelvis height
    data.qpos[3] = 1.0    # quat w (upright)
    # Joint angles: use keyframe if available, otherwise zeros (default URDF pose)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[2] = 0.75  # override height
    mujoco.mj_forward(model, data)
    print(f"  Initial pelvis height: {data.xpos[ref_body_id][2]:.3f}m")

    # --- Simulation loop ---
    dt = model.opt.timestep
    total_steps = int(args.duration / dt)
    print(f"\n[SIM] Running {total_steps} steps ({args.duration}s at {1/dt:.0f}Hz)")
    print(f"[SIM] Mode: {cmd_vel_mode}" + (f" = {fixed_vel} m/s" if cmd_vel_mode == "fixed" else ""))
    print(f"{'step':>6} | {'time':>5} | {'cmd':>5} | {'fwd_vel':>8} | {'height':>6}")
    print("-" * 50)

    # CSV log
    csv_path = args.policy.replace(".pt", "_sim2sim.csv")
    csv_file = open(csv_path, "w")
    csv_file.write("step,time_s,cmd_vel,fwd_vel,pelvis_height\n")

    # Viewer
    viewer = None
    if not args.no_render:
        viewer = mujoco.viewer.launch_passive(model, data)

    for step in range(total_steps):
        t = step * dt

        # --- Velocity command ---
        if cmd_vel_mode == "fixed":
            cmd_vel = fixed_vel
        else:  # ramp
            if t < 15.0:
                cmd_vel = 4.0
            else:
                cmd_vel = 0.0

        # --- Compute observation ---
        progress = t / args.duration
        obs_np = compute_observation(
            model, data, joint_qpos_ids, joint_qvel_ids,
            ref_body_id, key_body_ids, cmd_vel, progress
        )
        obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

        # --- Normalize observation ---
        if normalizer:
            obs = normalizer.normalize(obs)

        # --- Policy inference ---
        with torch.no_grad():
            action = policy(obs).squeeze(0).numpy()

        # --- Apply action (PD position control) ---
        target_pos = action_offset + action_scale * action
        for i, qpos_id in enumerate(joint_qpos_ids):
            # Find the actuator index for this joint
            data.ctrl[i] = target_pos[i]

        # --- Step physics ---
        mujoco.mj_step(model, data)

        # --- Get forward velocity ---
        root_quat = data.xquat[ref_body_id].copy()
        root_vel_w = data.cvel[ref_body_id][3:6].copy()
        quat_inv = np.array([root_quat[0], -root_quat[1], -root_quat[2], -root_quat[3]])
        root_vel_b = quat_apply(quat_inv, root_vel_w)
        fwd_vel = root_vel_b[0]
        pelvis_h = data.xpos[ref_body_id][2]

        # --- Log ---
        csv_file.write(f"{step},{t:.3f},{cmd_vel:.3f},{fwd_vel:.3f},{pelvis_h:.3f}\n")

        if (step + 1) % 60 == 0:
            print(f"{step+1:>6} | {t:>5.1f} | {cmd_vel:>5.1f} | {fwd_vel:>8.2f} | {pelvis_h:>6.3f}")

        # --- Update viewer ---
        if viewer and viewer.is_running():
            viewer.sync()
            if not args.no_render:
                time.sleep(max(0, dt - 0.001))  # rough real-time
        elif viewer and not viewer.is_running():
            break

    csv_file.close()
    print(f"\n[RESULT] CSV saved: {csv_path}")

    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
