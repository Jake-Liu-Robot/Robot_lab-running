# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Convert LAFAN1 CSV running motion to AMP NPZ format for G1.

USAGE (RunPod):
    cd /workspace/robot_lab
    pip install pinocchio  # if not installed
    python source/robot_lab/robot_lab/tasks/direct/g1_amp/motions/csv2npz_run.py

Reads: lafan1_g1/g1/run2_subject1.csv  (frames 1943-2564, 30fps, ~20.7s)
Outputs: source/robot_lab/robot_lab/tasks/direct/g1_amp/motions/g1_run2_subject1_30.npz
"""

import argparse
import os

import numpy as np
import pandas as pd
import pinocchio as pin

try:
    from pinocchio.robot_wrapper import RobotWrapper
except ImportError:
    RobotWrapper = pin.RobotWrapper

# --- defaults ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "..", ".."))

DEFAULT_CSV = os.path.join(REPO_ROOT, "lafan1_g1", "g1", "run2_subject1.csv")
DEFAULT_URDF = os.path.join(REPO_ROOT, "lafan1_g1", "robot_description", "g1", "g1_29dof_rev_1_0.urdf")
DEFAULT_MESH = os.path.join(REPO_ROOT, "lafan1_g1", "robot_description", "g1")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "g1_run2_subject1_30.npz")
DEFAULT_START = 1943  # 1-indexed, inclusive
DEFAULT_END = 2564  # 1-indexed, inclusive


def quaternion_inverse(q):
    w, x, y, z = q
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq < 1e-8:
        norm_sq = 1e-8
    return np.array([w, -x, -y, -z], dtype=q.dtype) / norm_sq


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=q1.dtype)


def compute_angular_velocity(q_prev, q_next, dt, eps=1e-8):
    q_inv = quaternion_inverse(q_prev)
    q_rel = quaternion_multiply(q_inv, q_next)
    norm_q_rel = np.linalg.norm(q_rel)
    if norm_q_rel < eps:
        return np.zeros(3, dtype=np.float32)
    q_rel /= norm_q_rel
    w = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < eps:
        return np.zeros(3, dtype=np.float32)
    axis = q_rel[1:] / sin_half
    return (angle / dt) * axis


def main():
    parser = argparse.ArgumentParser(description="Convert LAFAN1 CSV to AMP NPZ for G1")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Input CSV file")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="G1 URDF file")
    parser.add_argument("--mesh_dir", default=DEFAULT_MESH, help="Mesh directory")
    parser.add_argument("--output", default=DEFAULT_OUT, help="Output NPZ file")
    parser.add_argument("--start", type=int, default=DEFAULT_START, help="Start frame (1-indexed, inclusive)")
    parser.add_argument("--end", type=int, default=DEFAULT_END, help="End frame (1-indexed, inclusive)")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    args = parser.parse_args()

    fps = args.fps
    dt = 1.0 / fps

    # 1. Read CSV
    df = pd.read_csv(args.csv, header=None)
    start_idx = args.start - 1  # convert to 0-indexed
    end_idx = args.end  # pandas iloc end is exclusive, so 1-indexed inclusive → 0-indexed exclusive
    data_orig = df.iloc[start_idx:end_idx].to_numpy(dtype=np.float32)
    N = data_orig.shape[0]
    print(f"Loaded: {args.csv}, frames [{args.start}-{args.end}], {N} frames, {(N-1)/fps:.1f}s")

    root_data = data_orig[:, :7]  # (N, 7) = pos3 + quat4(xyzw)
    joint_data = data_orig[:, 7:]  # (N, 29)

    # Joint names (same order as LAFAN1 retargeting for G1)
    joint_names = [
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
    dof_names = np.array(joint_names, dtype=np.str_)

    # Joint positions & velocities (central differences)
    dof_positions = joint_data.copy()
    dof_velocities = np.zeros_like(dof_positions)
    dof_velocities[1:-1] = (dof_positions[2:] - dof_positions[:-2]) / (2 * dt)
    dof_velocities[0] = (dof_positions[1] - dof_positions[0]) / dt
    dof_velocities[-1] = (dof_positions[-1] - dof_positions[-2]) / dt

    # Body names (14 key bodies for AMP)
    body_names = [
        "pelvis", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
        "left_elbow_link", "right_elbow_link", "right_rubber_hand", "left_rubber_hand",
        "right_ankle_roll_link", "left_ankle_roll_link", "torso_link",
        "right_hip_yaw_link", "left_hip_yaw_link", "right_knee_link", "left_knee_link",
    ]
    body_names_arr = np.array(body_names, dtype=np.str_)
    B = len(body_names)

    # 2. Pinocchio FK
    print(f"Running FK with Pinocchio ({N} frames, {B} bodies)...")
    robot = RobotWrapper.BuildFromURDF(args.urdf, args.mesh_dir, pin.JointModelFreeFlyer())
    model = robot.model
    data_pk = robot.data

    body_positions = np.zeros((N, B, 3), dtype=np.float32)
    body_rotations = np.zeros((N, B, 4), dtype=np.float32)

    q_pin = pin.neutral(model)
    for i in range(N):
        q_pin[0:3] = root_data[i, 0:3]
        q_pin[3:7] = root_data[i, 3:7]  # CSV quat is (x,y,z,w) → pinocchio expects (x,y,z,w)
        q_pin[7:7 + joint_data.shape[1]] = joint_data[i, :]
        pin.forwardKinematics(model, data_pk, q_pin)
        pin.updateFramePlacements(model, data_pk)
        for j, link_name in enumerate(body_names):
            fid = model.getFrameId(link_name)
            link_tf = data_pk.oMf[fid]
            body_positions[i, j, :] = link_tf.translation
            quat_xyzw = pin.Quaternion(link_tf.rotation)
            body_rotations[i, j, :] = np.array(
                [quat_xyzw.w, quat_xyzw.x, quat_xyzw.y, quat_xyzw.z], dtype=np.float32
            )

    # 3. Body velocities
    body_linear_velocities = np.zeros_like(body_positions)
    body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
    body_linear_velocities[0] = (body_positions[1] - body_positions[0]) / dt
    body_linear_velocities[-1] = (body_positions[-1] - body_positions[-2]) / dt

    body_angular_velocities = np.zeros((N, B, 3), dtype=np.float32)
    for j in range(B):
        quats = body_rotations[:, j, :]
        ang_vels = np.zeros((N, 3), dtype=np.float32)
        if N > 1:
            ang_vels[0] = compute_angular_velocity(quats[0], quats[1], dt)
            ang_vels[-1] = compute_angular_velocity(quats[-2], quats[-1], dt)
        for k in range(1, N - 1):
            av1 = compute_angular_velocity(quats[k - 1], quats[k], dt)
            av2 = compute_angular_velocity(quats[k], quats[k + 1], dt)
            ang_vels[k] = 0.5 * (av1 + av2)
        body_angular_velocities[:, j, :] = ang_vels

    # 4. Save
    data_dict = {
        "fps": fps,
        "dof_names": dof_names,
        "body_names": body_names_arr,
        "dof_positions": dof_positions,
        "dof_velocities": dof_velocities,
        "body_positions": body_positions,
        "body_rotations": body_rotations,
        "body_linear_velocities": body_linear_velocities,
        "body_angular_velocities": body_angular_velocities,
    }
    np.savez(args.output, **data_dict)
    print(f"\nSaved: {args.output}")
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
