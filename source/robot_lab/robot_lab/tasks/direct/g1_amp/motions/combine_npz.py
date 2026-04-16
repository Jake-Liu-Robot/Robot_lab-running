"""
Combine running and standing NPZ files for AMP training.

Concatenates running data (straightened+mirrored) with standing data,
adding a separator marker so MotionLoader samples from both segments.

The combined file uses a simple approach: concatenate frames sequentially.
The MotionLoader samples random times uniformly, so with:
  - 1244 running frames (20.7s × 2 for mirror)
  - 151 standing frames (5s)
  Standing sampling probability: 151/1395 = 10.8%

To increase standing probability, use --repeat-standing N to repeat
the standing segment N times.

USAGE:
    python combine_npz.py \
      --running g1_run2_subject1_30_straight_mirror.npz \
      --standing g1_standing_5s.npz \
      --output g1_run_and_stand.npz \
      --repeat-standing 3
"""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Combine running and standing NPZ for AMP")
    parser.add_argument("--running", type=str, required=True, help="Running NPZ (straightened+mirrored)")
    parser.add_argument("--standing", type=str, required=True, help="Standing NPZ")
    parser.add_argument("--output", type=str, default=None, help="Output NPZ")
    parser.add_argument("--repeat-standing", type=int, default=3, help="Repeat standing data N times (increase sampling ratio)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.running), "g1_run_and_stand.npz")

    # Load
    run_data = dict(np.load(args.running, allow_pickle=True))
    stand_data = dict(np.load(args.standing, allow_pickle=True))

    run_frames = run_data["dof_positions"].shape[0]
    stand_frames = stand_data["dof_positions"].shape[0]
    fps_run = float(run_data["fps"])
    fps_stand = float(stand_data["fps"])

    assert fps_run == fps_stand, f"FPS mismatch: running={fps_run}, standing={fps_stand}"

    print(f"Running: {run_frames} frames ({(run_frames-1)/fps_run:.1f}s)")
    print(f"Standing: {stand_frames} frames ({(stand_frames-1)/fps_stand:.1f}s)")
    print(f"Repeating standing {args.repeat_standing}x")

    # Repeat standing
    repeated_stand = {}
    for key in ["dof_positions", "dof_velocities", "body_positions",
                 "body_rotations", "body_linear_velocities", "body_angular_velocities"]:
        repeated_stand[key] = np.tile(stand_data[key], (args.repeat_standing,) + (1,) * (stand_data[key].ndim - 1))

    total_stand = repeated_stand["dof_positions"].shape[0]

    # Concatenate
    combined = {}
    for key in ["dof_positions", "dof_velocities", "body_positions",
                 "body_rotations", "body_linear_velocities", "body_angular_velocities"]:
        combined[key] = np.concatenate([run_data[key], repeated_stand[key]], axis=0).astype(np.float32)

    combined["dof_names"] = run_data["dof_names"]
    combined["body_names"] = run_data["body_names"]
    combined["fps"] = run_data["fps"]

    total_frames = combined["dof_positions"].shape[0]
    stand_ratio = total_stand / total_frames * 100

    # Save
    np.savez(args.output, **{k: v for k, v in combined.items()})

    print(f"\nCombined: {args.output}")
    print(f"  Total frames: {total_frames} (running {run_frames} + standing {total_stand})")
    print(f"  Standing ratio: {stand_ratio:.1f}%")
    print(f"  Duration: {(total_frames-1)/fps_run:.1f}s")


if __name__ == "__main__":
    main()
