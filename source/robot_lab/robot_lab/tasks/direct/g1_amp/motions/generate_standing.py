"""
Generate clean standing data for AMP discriminator.

Uses Pinocchio FK to compute exact body positions for G1 default pose
(all joints = 0). Replicates this single frame for 5 seconds with zero velocities.

MUST run on RunPod (requires Pinocchio + G1 URDF).

USAGE:
    /isaac-sim/python.sh generate_standing.py

Output: g1_standing_5s.npz (same directory)
"""

import os

import numpy as np

try:
    import pinocchio as pin
    from pinocchio.robot_wrapper import RobotWrapper
except ImportError:
    try:
        import pinocchio as pin
        RobotWrapper = pin.RobotWrapper
    except ImportError:
        print("ERROR: Pinocchio not found. Run: pip install pin")
        exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "..", ".."))

# G1 URDF (same as csv2npz_run.py uses)
DEFAULT_URDF = os.path.join(REPO_ROOT, "source", "robot_lab", "data", "Robots",
                            "unitree", "g1_description", "urdf", "g1_29dof_rev_1_0.urdf")
DEFAULT_MESH = os.path.join(REPO_ROOT, "source", "robot_lab", "data", "Robots", "unitree")

# 14 key bodies for AMP (same order as csv2npz_run.py)
BODY_NAMES = [
    "pelvis", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link", "right_rubber_hand", "left_rubber_hand",
    "right_ankle_roll_link", "left_ankle_roll_link", "torso_link",
    "right_hip_yaw_link", "left_hip_yaw_link", "right_knee_link", "left_knee_link",
]

DOF_NAMES = [
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


def pin_rotation_to_quat_wxyz(R):
    """Convert 3x3 rotation matrix to quaternion (wxyz convention)."""
    quat_xyzw = pin.Quaternion(R).coeffs()  # pinocchio returns xyzw
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate standing NPZ using Pinocchio FK")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="G1 URDF path")
    parser.add_argument("--mesh_dir", default=DEFAULT_MESH, help="Mesh directory")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--output", default=None, help="Output path")
    args = parser.parse_args()

    output_path = args.output or os.path.join(SCRIPT_DIR, "g1_standing_5s.npz")
    N = int(args.duration * args.fps) + 1
    num_bodies = len(BODY_NAMES)
    num_dofs = len(DOF_NAMES)

    # --- Load robot model ---
    print(f"Loading URDF: {args.urdf}")
    robot = RobotWrapper.BuildFromURDF(args.urdf, [args.mesh_dir])
    model = robot.model
    data = robot.data

    # --- Compute FK for default pose (all joints = 0) ---
    q = pin.neutral(model)  # default configuration
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    print(f"\nDefault pose FK results:")
    body_positions = np.zeros((num_bodies, 3), dtype=np.float32)
    body_rotations = np.zeros((num_bodies, 4), dtype=np.float32)

    for i, body_name in enumerate(BODY_NAMES):
        frame_id = model.getFrameId(body_name)
        if frame_id >= len(data.oMf):
            print(f"  WARNING: body '{body_name}' not found, using zero position")
            body_rotations[i, 0] = 1.0  # identity quat
            continue

        placement = data.oMf[frame_id]
        pos = placement.translation.astype(np.float32)
        quat = pin_rotation_to_quat_wxyz(placement.rotation)

        body_positions[i] = pos
        body_rotations[i] = quat
        print(f"  {body_name:30s}: pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")

    pelvis_h = body_positions[0, 2]
    print(f"\nPelvis height: {pelvis_h:.3f}m")

    # --- Build NPZ arrays ---
    dof_positions = np.zeros((N, num_dofs), dtype=np.float32)
    dof_velocities = np.zeros((N, num_dofs), dtype=np.float32)
    body_pos_all = np.tile(body_positions, (N, 1, 1))  # (N, 14, 3)
    body_rot_all = np.tile(body_rotations, (N, 1, 1))  # (N, 14, 4)
    body_lv = np.zeros((N, num_bodies, 3), dtype=np.float32)
    body_av = np.zeros((N, num_bodies, 3), dtype=np.float32)

    # --- Save ---
    np.savez(
        output_path,
        dof_positions=dof_positions,
        dof_velocities=dof_velocities,
        body_positions=body_pos_all,
        body_rotations=body_rot_all,
        body_linear_velocities=body_lv,
        body_angular_velocities=body_av,
        dof_names=np.array(DOF_NAMES, dtype=np.str_),
        body_names=np.array(BODY_NAMES, dtype=np.str_),
        fps=args.fps,
    )

    print(f"\nSaved: {output_path}")
    print(f"  Frames: {N}, Duration: {args.duration}s, FPS: {args.fps}")
    print(f"  All joints = 0 (URDF default pose)")
    print(f"  All velocities = 0 (perfectly still)")
    print(f"  Body positions = FK computed (exact)")


if __name__ == "__main__":
    main()
