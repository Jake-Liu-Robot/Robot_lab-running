"""
Generate clean standing data for AMP discriminator.

Two modes:
1. --from-npz: Extract the quietest frame from an existing NPZ and replicate it
   with zero velocities. Body positions are FK-correct (from actual motion data).
   This is the RECOMMENDED mode.

2. Default: Generate synthetic standing with all joints=0 and approximate body
   positions. Less accurate but doesn't need existing data.

USAGE (recommended):
    python generate_standing.py --from-npz g1_run2_subject1_30_straight_mirror.npz

    # Output: g1_standing_5s.npz (same directory)
"""

import argparse
import os

import numpy as np


def find_quietest_frame(data):
    """Find the frame with minimum joint velocity (most standing-like)."""
    dof_vel = data["dof_velocities"]  # (T, 29)
    # RMS of joint velocities per frame
    vel_rms = np.sqrt(np.mean(dof_vel ** 2, axis=1))
    best_idx = np.argmin(vel_rms)
    print(f"  Quietest frame: {best_idx} (vel_rms={vel_rms[best_idx]:.4f})")
    print(f"  Joint pos range: [{data['dof_positions'][best_idx].min():.3f}, {data['dof_positions'][best_idx].max():.3f}]")
    print(f"  Height: {data['body_positions'][best_idx, 0, 2]:.3f}m")
    return best_idx


def generate_from_npz(npz_path, output_path, duration_s=5.0, fps=30):
    """Extract quietest frame from existing NPZ and replicate with zero velocities."""
    data = dict(np.load(npz_path, allow_pickle=True))
    print(f"Source: {npz_path} ({data['dof_positions'].shape[0]} frames)")

    best_idx = find_quietest_frame(data)
    N = int(duration_s * fps) + 1

    result = {}
    # Replicate the single frame N times
    for key in ["dof_positions", "body_positions", "body_rotations"]:
        frame = data[key][best_idx:best_idx + 1]  # (1, ...)
        result[key] = np.tile(frame, (N,) + (1,) * (frame.ndim - 1)).astype(np.float32)

    # Set all velocities to zero (perfectly still standing)
    result["dof_velocities"] = np.zeros((N, data["dof_positions"].shape[1]), dtype=np.float32)
    result["body_linear_velocities"] = np.zeros((N,) + data["body_positions"].shape[1:], dtype=np.float32)
    result["body_angular_velocities"] = np.zeros((N,) + data["body_positions"].shape[1:], dtype=np.float32)

    result["dof_names"] = data["dof_names"]
    result["body_names"] = data["body_names"]
    result["fps"] = data["fps"]

    np.savez(output_path, **result)
    print(f"\nGenerated: {output_path}")
    print(f"  Frames: {N}, Duration: {duration_s}s")
    print(f"  Body positions: from FK (frame {best_idx})")
    print(f"  All velocities: zero")


def generate_synthetic(output_path, duration_s=5.0, fps=30):
    """Generate synthetic standing with approximate body positions."""
    N = int(duration_s * fps) + 1
    num_dofs = 29
    num_bodies = 14

    dof_names = [
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
    body_names = [
        "pelvis", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
        "left_elbow_link", "right_elbow_link", "right_rubber_hand", "left_rubber_hand",
        "right_ankle_roll_link", "left_ankle_roll_link", "torso_link",
        "right_hip_yaw_link", "left_hip_yaw_link", "right_knee_link", "left_knee_link",
    ]

    # Approximate body positions (rough estimates from G1 URDF)
    body_pos = np.zeros((num_bodies, 3), dtype=np.float32)
    body_pos[0] = [0.0, 0.0, 0.78]
    body_pos[1] = [0.0, 0.18, 1.05]
    body_pos[2] = [0.0, -0.18, 1.05]
    body_pos[3] = [0.0, 0.30, 0.85]
    body_pos[4] = [0.0, -0.30, 0.85]
    body_pos[5] = [0.0, -0.40, 0.75]
    body_pos[6] = [0.0, 0.40, 0.75]
    body_pos[7] = [0.0, -0.10, 0.04]
    body_pos[8] = [0.0, 0.10, 0.04]
    body_pos[9] = [0.0, 0.0, 0.95]
    body_pos[10] = [0.0, -0.10, 0.68]
    body_pos[11] = [0.0, 0.10, 0.68]
    body_pos[12] = [0.0, -0.10, 0.38]
    body_pos[13] = [0.0, 0.10, 0.38]

    result = {
        "dof_positions": np.zeros((N, num_dofs), dtype=np.float32),
        "dof_velocities": np.zeros((N, num_dofs), dtype=np.float32),
        "body_positions": np.tile(body_pos, (N, 1, 1)),
        "body_rotations": np.zeros((N, num_bodies, 4), dtype=np.float32),
        "body_linear_velocities": np.zeros((N, num_bodies, 3), dtype=np.float32),
        "body_angular_velocities": np.zeros((N, num_bodies, 3), dtype=np.float32),
        "dof_names": np.array(dof_names, dtype=np.str_),
        "body_names": np.array(body_names, dtype=np.str_),
        "fps": fps,
    }
    result["body_rotations"][:, :, 0] = 1.0  # identity quaternion

    np.savez(output_path, **result)
    print(f"Generated (synthetic): {output_path}")
    print(f"  Frames: {N}, Duration: {duration_s}s")
    print(f"  WARNING: Body positions are approximate, not FK-computed")


def main():
    parser = argparse.ArgumentParser(description="Generate standing data for AMP")
    parser.add_argument("--from-npz", type=str, default=None,
                        help="Extract quietest frame from existing NPZ (recommended)")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output = args.output or os.path.join(script_dir, "g1_standing_5s.npz")

    if args.from_npz:
        generate_from_npz(args.from_npz, output, args.duration)
    else:
        generate_synthetic(output, args.duration)


if __name__ == "__main__":
    main()
