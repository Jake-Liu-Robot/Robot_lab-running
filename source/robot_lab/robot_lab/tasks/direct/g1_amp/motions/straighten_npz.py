"""
Straighten and mirror AMP reference motion NPZ for G1.

Two operations:
1. STRAIGHTEN: Remove cumulative yaw rotation from each frame.
   The character runs in a constant heading direction (no turning).
   Joint angles are preserved — only global path curvature is removed.

2. MIRROR: Create left-right mirrored version by swapping left/right
   joint indices and negating y-components. Combined with original
   to eliminate directional bias in the discriminator.

USAGE (local or RunPod):
    python straighten_npz.py --input g1_run2_subject1_30.npz --output g1_run2_straight_mirror.npz

Output NPZ has doubled frame count (original + mirrored), both straightened.
"""

import argparse
import os

import numpy as np


# ------------------------------------------------------------------ #
#  Quaternion utilities (wxyz convention)
# ------------------------------------------------------------------ #

def quat_multiply(q1, q2):
    """Multiply quaternions q1 * q2 (wxyz convention)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_from_yaw(yaw):
    """Create quaternion from yaw angle (rotation around z-axis, wxyz)."""
    half = yaw * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def quat_extract_yaw(q):
    """Extract yaw angle from quaternion (wxyz convention)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw


def rotate_vector_by_yaw(yaw, v):
    """Rotate 3D vector(s) by yaw angle around z-axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    vx = c * v[..., 0] - s * v[..., 1]
    vy = s * v[..., 0] + c * v[..., 1]
    vz = v[..., 2]
    return np.stack([vx, vy, vz], axis=-1)


# ------------------------------------------------------------------ #
#  G1 29-DOF left-right joint mapping
# ------------------------------------------------------------------ #

# Joint index pairs: (left, right)
JOINT_MIRROR_PAIRS = [
    (0, 6),    # hip_pitch
    (1, 7),    # hip_roll
    (2, 8),    # hip_yaw
    (3, 9),    # knee
    (4, 10),   # ankle_pitch
    (5, 11),   # ankle_roll
    (15, 22),  # shoulder_pitch
    (16, 23),  # shoulder_roll
    (17, 24),  # shoulder_yaw
    (18, 25),  # elbow
    (19, 26),  # wrist_roll
    (20, 27),  # wrist_pitch
    (21, 28),  # wrist_yaw
]

# Joints that negate sign when mirrored (roll and yaw joints)
JOINT_NEGATE = {1, 2, 7, 8, 13, 16, 17, 23, 24}
# roll joints: hip_roll(1,7), waist_roll(13), shoulder_roll(16,23)
# yaw joints: hip_yaw(2,8), shoulder_yaw(17,24)

# Body index pairs: (left, right) in the 14-body list
BODY_MIRROR_PAIRS = [
    (1, 2),    # shoulder_yaw_link
    (3, 4),    # elbow_link
    (6, 5),    # rubber_hand (note: left=6, right=5 in body_names)
    (8, 7),    # ankle_roll_link (left=8, right=7)
    (11, 10),  # hip_yaw_link (left=11, right=10)
    (13, 12),  # knee_link (left=13, right=12)
]

# Non-mirrored bodies: 0 (pelvis), 9 (torso_link)


# ------------------------------------------------------------------ #
#  Straighten
# ------------------------------------------------------------------ #

def straighten(data):
    """Remove cumulative yaw from root rotation and adjust velocities."""
    body_rot = data["body_rotations"].copy()      # (T, B, 4) wxyz
    body_pos = data["body_positions"].copy()       # (T, B, 3)
    body_lv = data["body_linear_velocities"].copy()   # (T, B, 3)
    body_av = data["body_angular_velocities"].copy()   # (T, B, 3)

    T, B, _ = body_rot.shape

    for i in range(T):
        # Extract yaw from pelvis (body 0) rotation
        yaw = quat_extract_yaw(body_rot[i, 0])
        neg_yaw = -yaw

        # Remove yaw from all body rotations
        inv_yaw_quat = quat_from_yaw(neg_yaw)
        for b in range(B):
            body_rot[i, b] = quat_multiply(inv_yaw_quat, body_rot[i, b])

        # Rotate positions to remove yaw (relative to first frame origin)
        for b in range(B):
            body_pos[i, b] = rotate_vector_by_yaw(neg_yaw, body_pos[i, b])

        # Rotate velocities
        for b in range(B):
            body_lv[i, b] = rotate_vector_by_yaw(neg_yaw, body_lv[i, b])
            body_av[i, b] = rotate_vector_by_yaw(neg_yaw, body_av[i, b])

    result = dict(data)
    result["body_rotations"] = body_rot
    result["body_positions"] = body_pos
    result["body_linear_velocities"] = body_lv
    result["body_angular_velocities"] = body_av
    return result


# ------------------------------------------------------------------ #
#  Mirror
# ------------------------------------------------------------------ #

def mirror(data):
    """Create left-right mirrored version of motion data."""
    dof_pos = data["dof_positions"].copy()     # (T, 29)
    dof_vel = data["dof_velocities"].copy()    # (T, 29)
    body_pos = data["body_positions"].copy()   # (T, B, 3)
    body_rot = data["body_rotations"].copy()   # (T, B, 4) wxyz
    body_lv = data["body_linear_velocities"].copy()   # (T, B, 3)
    body_av = data["body_angular_velocities"].copy()   # (T, B, 3)

    # 1. Mirror joint positions and velocities: swap left/right indices
    m_dof_pos = dof_pos.copy()
    m_dof_vel = dof_vel.copy()
    for l, r in JOINT_MIRROR_PAIRS:
        m_dof_pos[:, l] = dof_pos[:, r]
        m_dof_pos[:, r] = dof_pos[:, l]
        m_dof_vel[:, l] = dof_vel[:, r]
        m_dof_vel[:, r] = dof_vel[:, l]

    # Negate roll/yaw joints (they flip sign in mirror)
    for j in JOINT_NEGATE:
        m_dof_pos[:, j] = -m_dof_pos[:, j]
        m_dof_vel[:, j] = -m_dof_vel[:, j]

    # Waist yaw (index 12) also negates
    m_dof_pos[:, 12] = -m_dof_pos[:, 12]
    m_dof_vel[:, 12] = -m_dof_vel[:, 12]

    # 2. Mirror body positions: swap left/right, negate y
    m_body_pos = body_pos.copy()
    for l, r in BODY_MIRROR_PAIRS:
        m_body_pos[:, l] = body_pos[:, r].copy()
        m_body_pos[:, r] = body_pos[:, l].copy()
    m_body_pos[:, :, 1] = -m_body_pos[:, :, 1]  # negate y for all bodies

    # 3. Mirror body rotations: swap left/right, negate y and z quaternion components
    m_body_rot = body_rot.copy()
    for l, r in BODY_MIRROR_PAIRS:
        m_body_rot[:, l] = body_rot[:, r].copy()
        m_body_rot[:, r] = body_rot[:, l].copy()
    # Mirror quaternion: reflect across XZ plane → negate y and z
    m_body_rot[:, :, 2] = -m_body_rot[:, :, 2]  # negate qy
    m_body_rot[:, :, 3] = -m_body_rot[:, :, 3]  # negate qz

    # 4. Mirror velocities: swap left/right, negate y
    m_body_lv = body_lv.copy()
    m_body_av = body_av.copy()
    for l, r in BODY_MIRROR_PAIRS:
        m_body_lv[:, l] = body_lv[:, r].copy()
        m_body_lv[:, r] = body_lv[:, l].copy()
        m_body_av[:, l] = body_av[:, r].copy()
        m_body_av[:, r] = body_av[:, l].copy()
    m_body_lv[:, :, 1] = -m_body_lv[:, :, 1]  # negate vy
    m_body_av[:, :, 0] = -m_body_av[:, :, 0]  # negate wx (mirror angular vel)
    m_body_av[:, :, 2] = -m_body_av[:, :, 2]  # negate wz

    result = dict(data)
    result["dof_positions"] = m_dof_pos
    result["dof_velocities"] = m_dof_vel
    result["body_positions"] = m_body_pos
    result["body_rotations"] = m_body_rot
    result["body_linear_velocities"] = m_body_lv
    result["body_angular_velocities"] = m_body_av
    return result


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Straighten and mirror AMP motion NPZ")
    parser.add_argument("--input", type=str, required=True, help="Input NPZ file")
    parser.add_argument("--output", type=str, default=None, help="Output NPZ file")
    parser.add_argument("--no-mirror", action="store_true", help="Only straighten, skip mirroring")
    parser.add_argument("--no-straighten", action="store_true", help="Only mirror, skip straightening")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        suffix = "_straight" if not args.no_straighten else ""
        suffix += "_mirror" if not args.no_mirror else ""
        args.output = f"{base}{suffix}.npz"

    # Load
    raw = dict(np.load(args.input, allow_pickle=True))
    T = raw["dof_positions"].shape[0]
    fps = float(raw["fps"])
    print(f"Loaded: {args.input}")
    print(f"  Frames: {T}, FPS: {fps}, Duration: {(T-1)/fps:.1f}s")
    print(f"  DOFs: {len(raw['dof_names'])}, Bodies: {len(raw['body_names'])}")

    # Verify yaw drift in original
    yaw_first = quat_extract_yaw(raw["body_rotations"][0, 0])
    yaw_last = quat_extract_yaw(raw["body_rotations"][-1, 0])
    print(f"  Original yaw drift: {np.degrees(yaw_first):.1f}° → {np.degrees(yaw_last):.1f}° "
          f"(total: {np.degrees(yaw_last - yaw_first):.1f}°)")

    data = raw

    # Straighten
    if not args.no_straighten:
        data = straighten(data)
        yaw_first = quat_extract_yaw(data["body_rotations"][0, 0])
        yaw_last = quat_extract_yaw(data["body_rotations"][-1, 0])
        print(f"  After straighten: yaw {np.degrees(yaw_first):.1f}° → {np.degrees(yaw_last):.1f}°")

    # Mirror and concatenate
    if not args.no_mirror:
        mirrored = mirror(data)
        # Concatenate: original + mirrored
        combined = {}
        for key in ["dof_positions", "dof_velocities", "body_positions",
                     "body_rotations", "body_linear_velocities", "body_angular_velocities"]:
            combined[key] = np.concatenate([data[key], mirrored[key]], axis=0).astype(np.float32)
        combined["dof_names"] = data["dof_names"]
        combined["body_names"] = data["body_names"]
        combined["fps"] = data["fps"]
        data = combined
        print(f"  After mirror: {data['dof_positions'].shape[0]} frames (original {T} + mirrored {T})")

    # Save
    np.savez(args.output, **{k: v for k, v in data.items()})
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
