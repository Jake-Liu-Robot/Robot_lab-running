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
import os
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
        normed = (obs - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(normed, -self.clip, self.clip).float()


# ============================================================
#  G1 Configuration (matches Isaac Lab exactly)
# ============================================================

# Joint names in Isaac Lab's ArticulationData.joint_names order.
# ⚠ Confirmed via scripts/sim2sim/print_lab_order.py on RunPod: Isaac Lab uses
#   breadth-first-like ordering from pelvis, interleaving L/R/waist at each
#   level — NOT the URDF / MuJoCo native order (which is L-leg, R-leg, waist,
#   L-arm, R-arm). The policy outputs actions and reads joint_pos/joint_vel
#   in THIS order; if sim2sim uses the URDF order the action vector gets
#   scrambled (leg targets applied to arms, etc.) and the robot collapses.
JOINT_NAMES = [
    "left_hip_pitch_joint",    "right_hip_pitch_joint",    "waist_yaw_joint",        # 0-2
    "left_hip_roll_joint",     "right_hip_roll_joint",     "waist_roll_joint",       # 3-5
    "left_hip_yaw_joint",      "right_hip_yaw_joint",      "waist_pitch_joint",      # 6-8
    "left_knee_joint",         "right_knee_joint",                                   # 9-10
    "left_shoulder_pitch_joint","right_shoulder_pitch_joint",                        # 11-12
    "left_ankle_pitch_joint",  "right_ankle_pitch_joint",                            # 13-14
    "left_shoulder_roll_joint","right_shoulder_roll_joint",                          # 15-16
    "left_ankle_roll_joint",   "right_ankle_roll_joint",                             # 17-18
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",                           # 19-20
    "left_elbow_joint",        "right_elbow_joint",                                  # 21-22
    "left_wrist_roll_joint",   "right_wrist_roll_joint",                             # 23-24
    "left_wrist_pitch_joint",  "right_wrist_pitch_joint",                            # 25-26
    "left_wrist_yaw_joint",    "right_wrist_yaw_joint",                              # 27-28
]

# Key body names for observation (13 bodies).
# ⚠ URDF-only bodies that MJCF merges via fixed joint must be remapped to the
#    MJCF parent + the URDF offset, else `mj_name2id` returns -1 and xpos[-1]
#    silently feeds garbage into the policy.
# rubber_hand (fixed to wrist_yaw_link) offset from g1_29dof_rev_1_0.urdf:
#   left_hand_palm_joint:  xyz="0.0415  0.003 0"
#   right_hand_palm_joint: xyz="0.0415 -0.003 0"
KEY_BODY_NAMES = [
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "right_rubber_hand", "left_rubber_hand",
    "right_ankle_roll_link", "left_ankle_roll_link",
    "torso_link",
    "right_hip_yaw_link", "left_hip_yaw_link",
    "right_knee_link", "left_knee_link",
]

# body_name → (MJCF parent body, local offset in parent frame) for URDF-only bodies
#   that MJCF merges via fixed joints.
KEY_BODY_REMAP = {
    "left_rubber_hand":  ("left_wrist_yaw_link",  np.array([0.0415,  0.003, 0.0])),
    "right_rubber_hand": ("right_wrist_yaw_link", np.array([0.0415, -0.003, 0.0])),
}

REF_BODY_NAME = "pelvis"

# Soft joint position limit factor (from unitree.py UNITREE_G1_29DOF_CFG).
# Isaac Lab's action_offset/action_scale uses soft limits = factor × hard range.
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9

# Initial state at reset (from unitree.py UNITREE_G1_29DOF_CFG.init_state).
# reset_strategy="default" in g1_amp_run_env_cfg.py, so lab eval starts here.
INIT_PELVIS_HEIGHT = 0.76
INIT_JOINT_POS = {
    "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669, "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6, "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2, "right_shoulder_pitch_joint": 0.2,
}

# Early-termination height from g1_amp_run_env_cfg.py (lab terminates below this).
TERMINATION_HEIGHT = 0.45

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

def setup_mujoco_model(xml_path, offwidth=1280, offheight=720,
                       physics_timestep=0.002, integrator="implicitfast"):
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

    # Inject/override <option> with implicit integrator + small timestep so the
    # stiff PD controllers don't blow up (PhysX in Isaac Lab substeps internally;
    # MuJoCo defaults to Euler + 2ms but we get 1/60 after the override below).
    option_block = (
        f'  <option timestep="{physics_timestep}" integrator="{integrator}"/>\n'
    )
    if "<option" in xml_content:
        import re
        xml_content = re.sub(r"<option[^/>]*/>", option_block.strip(), xml_content, count=1)
    else:
        xml_content = xml_content.replace(
            "<compiler", option_block + "  <compiler", 1
        )

    # Ensure offscreen framebuffer is large enough for HD video.
    # MuJoCo only allows one <global> element inside <visual>; if one already
    # exists (e.g. Unitree G1 XML has azimuth/elevation there), add the offwidth
    # /offheight attributes into it rather than inserting a second element.
    import re
    global_pat = re.compile(r"<global([^/>]*)/>")
    m = global_pat.search(xml_content)
    if m:
        attrs = m.group(1)
        if "offwidth" not in attrs:
            attrs = f'{attrs} offwidth="{offwidth}" offheight="{offheight}"'
            xml_content = xml_content[: m.start()] + f"<global{attrs}/>" + xml_content[m.end():]
    else:
        visual_block = (
            f'  <visual>\n    <global offwidth="{offwidth}" offheight="{offheight}"/>\n  </visual>\n'
        )
        xml_content = xml_content.replace("<worldbody>", visual_block + "  <worldbody>", 1)

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

    # timestep/integrator already set via XML <option>. Leave as-is.
    return model, data, xml_content


def get_body_id(model, name):
    """Get MuJoCo body ID by name. Raises if not found — mj_name2id returns -1
    silently, which indexes the last body and produces silent garbage."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"Body '{name}' not found in MuJoCo model")
    return bid


def resolve_key_body(model, name):
    """Return (body_id, local_offset) for a URDF-named key body.

    For bodies that MJCF merges via fixed joint (e.g. rubber_hand), look up the
    MJCF parent and include the URDF offset so the world-frame point matches
    Isaac Lab's body_pos_w[rubber_hand].
    """
    if name in KEY_BODY_REMAP:
        parent, offset = KEY_BODY_REMAP[name]
        return get_body_id(model, parent), offset.astype(np.float64)
    return get_body_id(model, name), np.zeros(3, dtype=np.float64)


def key_body_world_pos(data, body_id, local_offset):
    """World position of a point rigidly attached to body_id at local_offset."""
    if np.all(local_offset == 0.0):
        return data.xpos[body_id].copy()
    # data.xmat is row-major rotation matrix; shape (nbody, 9)
    R = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + R @ local_offset


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
                        ref_body_id, key_bodies, cmd_vel, progress):
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

    # Key body positions relative to root (13×3 = 39-dim).
    # key_bodies is list[(body_id, local_offset)] — offset supports rubber_hand
    # points that MJCF merged into wrist_yaw_link.
    key_body_rel = []
    for bid, offset in key_bodies:
        body_pos = key_body_world_pos(data, bid, offset)
        key_body_rel.append(body_pos - root_pos)
    key_body_rel = np.concatenate(key_body_rel).astype(np.float32)  # 39-dim

    # Progress (1-dim)
    progress_arr = np.array([progress], dtype=np.float32)

    # AMP obs (105-dim)
    amp_obs = np.concatenate([
        joint_pos, joint_vel, root_height,
        root_tangent_normal, key_body_rel, progress_arr
    ])

    # Root velocity in body frame (3-dim).
    # Match Isaac Lab's body_lin_vel_w[pelvis] (body link origin velocity in world
    # frame). For a free joint, MuJoCo's qvel[0:3] is exactly that — linear velocity
    # of the body origin in the world frame. data.cvel[body][3:6] would give COM
    # velocity instead, which differs by omega × r_com for dynamic motion.
    root_vel_world = data.qvel[0:3].copy()
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
    parser.add_argument("--no_render", action="store_true", help="Disable interactive viewer")
    parser.add_argument("--no_video", action="store_true", help="Disable offscreen video recording")
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for video+csv (default: <policy_dir>/sim2sim/)")
    parser.add_argument("--physics_dt", type=float, default=0.002,
                        help="MuJoCo physics timestep (s). Substeps per policy step = (1/60)/physics_dt.")
    parser.add_argument("--debug_steps", type=int, default=0,
                        help="Print obs/action stats for first N control steps (0 = disabled).")
    args = parser.parse_args()
    control_dt = 1.0 / 60.0
    physics_substeps = max(1, int(round(control_dt / args.physics_dt)))

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
    # skrl layouts seen:
    #   legacy: policy_net_0.weight / policy_net_0.bias / ... policy_net_4.*
    #   newer:  policy_net_container.0.weight / ... policy_net_container.4.*
    # our target: net.0.*, net.2.*, net.4.*
    def _find_key(policy_state, layer_idx, suffix):
        candidates = [
            f"policy_net_{layer_idx}.{suffix}",
            f"policy_net_container.{layer_idx}.{suffix}",
            f"policy.net.{layer_idx}.{suffix}",
            f"policy.net_container.{layer_idx}.{suffix}",
        ]
        for c in candidates:
            if c in policy_state:
                return c
        return None

    for layer_idx in (0, 2, 4):
        for suffix in ("weight", "bias"):
            src = _find_key(policy_state, layer_idx, suffix)
            if src is None:
                print(f"  [WARN] No match for layer {layer_idx} {suffix}")
                continue
            state_dict[f"net.{layer_idx}.{suffix}"] = policy_state[src]

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
    model, data, xml_str = setup_mujoco_model(
        args.xml_path, offwidth=args.video_width, offheight=args.video_height,
        physics_timestep=args.physics_dt,
    )
    print(f"  Physics: dt={model.opt.timestep:.4f}s integrator={model.opt.integrator} "
          f"substeps/control={physics_substeps}")

    # Get IDs. For merged URDF-only bodies (rubber_hand), also resolve local offset.
    ref_body_id = get_body_id(model, REF_BODY_NAME)
    key_bodies = [resolve_key_body(model, name) for name in KEY_BODY_NAMES]
    joint_qpos_ids, joint_qvel_ids = get_joint_ids(model, JOINT_NAMES)
    for name, (bid, offset) in zip(KEY_BODY_NAMES, key_bodies):
        mj_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if np.any(offset != 0.0):
            print(f"  key body {name:24s} → MJCF {mj_name} + offset {tuple(offset)}")

    # Action scaling: prefer exported values from Isaac Lab, fallback to MuJoCo limits
    # multiplied by soft_joint_pos_limit_factor (0.9 in unitree.py), which is what
    # Isaac Lab actually uses via `self.robot.data.soft_joint_pos_limits`.
    if export.get("action_offset") is not None:
        action_offset = export["action_offset"].numpy()
        action_scale = export["action_scale"].numpy()
        print("  Action scaling: from Isaac Lab export")
    else:
        action_offset, action_scale = compute_action_scaling(model, joint_qpos_ids)
        action_scale = action_scale * SOFT_JOINT_POS_LIMIT_FACTOR
        print(f"  Action scaling: from MuJoCo hard limits × soft factor "
              f"({SOFT_JOINT_POS_LIMIT_FACTOR})")
    print(f"  Ref body: {REF_BODY_NAME} (id={ref_body_id})")
    print(f"  Key bodies: {len(key_bodies)}")
    print(f"  Joints: {len(joint_qpos_ids)}")

    # --- Reset to Isaac Lab default state (reset_strategy="default") ---
    # unitree.py UNITREE_G1_29DOF_CFG.init_state: pos=(0,0,0.76) + specific joints.
    mujoco.mj_resetData(model, data)
    # Free joint: qpos[0:3]=pos, qpos[3:7]=quat(w,x,y,z)
    data.qpos[0:3] = [0.0, 0.0, INIT_PELVIS_HEIGHT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    # Apply specific joint angles (all others stay at 0).
    for jn, angle in INIT_JOINT_POS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = angle
    # Zero all velocities.
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"  Initial pelvis height: {data.xpos[ref_body_id][2]:.3f}m "
          f"(target {INIT_PELVIS_HEIGHT})")

    # --- Simulation loop ---
    dt = control_dt  # policy / log / video frequency
    total_steps = int(args.duration / dt)
    print(f"\n[SIM] Running {total_steps} control steps ({args.duration}s at {1/dt:.0f}Hz), "
          f"{physics_substeps} physics substeps/step")
    print(f"[SIM] Mode: {cmd_vel_mode}" + (f" = {fixed_vel} m/s" if cmd_vel_mode == "fixed" else ""))
    print(f"{'step':>6} | {'time':>5} | {'cmd':>5} | {'fwd_vel':>8} | {'lat':>6} | {'height':>6} | {'rew':>6}")
    print("-" * 68)

    # --- Output paths (match eval_amp_run.py convention) ---
    ckpt_name = os.path.splitext(os.path.basename(args.policy))[0]
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(args.policy)), "sim2sim")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"sim2sim_{ckpt_name}_{cmd_vel_mode}.csv")
    video_path = os.path.join(output_dir, f"sim2sim_{ckpt_name}_{cmd_vel_mode}.mp4")

    csv_file = open(csv_path, "w")
    csv_file.write("step,time_s,cmd_vel,fwd_vel,lateral_vel,pelvis_height,reward_step\n")

    # --- Interactive viewer (optional) ---
    viewer = None
    if not args.no_render:
        viewer = mujoco.viewer.launch_passive(model, data)

    # --- Offscreen video recorder ---
    video_writer = None
    renderer = None
    if not args.no_video:
        import imageio
        renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
        fps = int(round(1.0 / dt))
        video_writer = imageio.get_writer(video_path, fps=fps, codec="libx264",
                                          quality=7, macro_block_size=1)
        print(f"[VIDEO] Recording to {video_path} ({args.video_width}x{args.video_height} @ {fps}fps)")

        # Tracking camera (behind-and-above, matching eval_amp_run.py style)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 4.5
        cam.elevation = -20.0
        cam.azimuth = 140.0

    reward_sum = 0.0
    for step in range(total_steps):
        t = step * dt

        # --- Velocity command ---
        if cmd_vel_mode == "fixed":
            cmd_vel = fixed_vel
        else:  # ramp — matches eval_amp_run.py: 4 m/s for 15s, 0 for 5s
            cmd_vel = 4.0 if t < 15.0 else 0.0

        # --- Compute observation ---
        progress = t / args.duration
        obs_np = compute_observation(
            model, data, joint_qpos_ids, joint_qvel_ids,
            ref_body_id, key_bodies, cmd_vel, progress
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
        for i in range(len(joint_qpos_ids)):
            data.ctrl[i] = target_pos[i]

        # --- Debug: print obs/action stats for first N steps ---
        if step < args.debug_steps:
            cur_qpos = np.array([data.qpos[j] for j in joint_qpos_ids])
            print(f"\n=== SIM2SIM_STEP {step} h={data.xpos[ref_body_id][2]:.4f} cmd={cmd_vel:.3f} ===")
            print("=== SIM2SIM_OBS (109-dim, pre-norm) ===")
            for i, x in enumerate(obs_np):
                print(f"obs[{i}]={x:+.6f}")
            print("=== SIM2SIM_ACTION (29-dim, raw policy output) ===")
            for i, x in enumerate(action):
                print(f"act[{i}]={x:+.6f}")
            print("=== SIM2SIM_TARGET (29-dim, PD target = offset + scale*action) ===")
            for i, x in enumerate(target_pos):
                print(f"tgt[{i}]={x:+.6f}")
            print("=== END_SIM2SIM_STEP ===")

        # --- Step physics (substep to decouple physics dt from control dt) ---
        for _ in range(physics_substeps):
            mujoco.mj_step(model, data)

        # --- Extract state for logging (match lab: body_lin_vel_w world X as fwd_vel) ---
        root_pos = data.xpos[ref_body_id].copy()
        root_quat = data.xquat[ref_body_id].copy()
        root_vel_w = data.qvel[0:3].copy()  # world-frame linear velocity of pelvis
        quat_inv = np.array([root_quat[0], -root_quat[1], -root_quat[2], -root_quat[3]])
        root_vel_b = quat_apply(quat_inv, root_vel_w)
        fwd_vel = float(root_vel_w[0])    # lab logs world X velocity as fwd_vel
        lat_vel = float(root_vel_w[1])    # lab logs world Y as lateral_vel
        pelvis_h = float(root_pos[2])

        # velocity-tracking reward (matches g1_amp_run_env.py: 1.5 * exp(-4 * err^2))
        rew_vel = 1.5 * float(np.exp(-4.0 * (fwd_vel - cmd_vel) ** 2))
        reward_sum += rew_vel

        csv_file.write(f"{step},{t:.3f},{cmd_vel:.3f},{fwd_vel:.3f},"
                       f"{lat_vel:.3f},{pelvis_h:.3f},{rew_vel:.4f}\n")

        if (step + 1) % 60 == 0 or step == total_steps - 1:
            print(f"{step+1:>6} | {t:>5.1f} | {cmd_vel:>5.1f} | {fwd_vel:>8.2f} | "
                  f"{lat_vel:>6.2f} | {pelvis_h:>6.3f} | {rew_vel:>6.3f}")

        # --- Write video frame (track robot) ---
        if video_writer is not None:
            cam.lookat[0] = root_pos[0]
            cam.lookat[1] = root_pos[1]
            cam.lookat[2] = max(0.5, root_pos[2])
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            video_writer.append_data(frame)

        # --- Update interactive viewer ---
        if viewer and viewer.is_running():
            viewer.sync()
            time.sleep(max(0, dt - 0.001))
        elif viewer and not viewer.is_running():
            break

    csv_file.close()
    if video_writer is not None:
        video_writer.close()
    if renderer is not None:
        renderer.close()
    if viewer:
        viewer.close()

    print(f"\n[RESULT] Total reward (velocity tracking): {reward_sum:.1f}")
    print(f"[RESULT] CSV:   {csv_path}")
    if not args.no_video:
        print(f"[RESULT] Video: {video_path}")


if __name__ == "__main__":
    main()
