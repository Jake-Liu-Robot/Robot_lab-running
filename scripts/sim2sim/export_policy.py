"""
Export skrl AMP checkpoint to standalone policy file for sim2sim.

Extracts policy network weights + observation normalizer from skrl checkpoint.
Output is a single .pt file that can be loaded without skrl/Isaac Lab.

USAGE (RunPod):
    /isaac-sim/python.sh scripts/sim2sim/export_policy.py \
        --checkpoint logs/skrl/g1_amp_run/.../checkpoints/agent_180000.pt \
        --output policy_exported.pt
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Export skrl AMP policy for sim2sim")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to skrl agent checkpoint")
    parser.add_argument("--output", type=str, default="policy_exported.pt", help="Output file path")
    args = parser.parse_args()

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Flatten nested state dicts so the rest of the script works on both
    # legacy-flat (policy_net_0.weight) and new-nested (policy -> OrderedDict) layouts.
    flat = {}
    for top_key, top_val in ckpt.items():
        if hasattr(top_val, "items") and not hasattr(top_val, "shape"):
            for sub_key, sub_val in top_val.items():
                flat[f"{top_key}.{sub_key}"] = sub_val
        else:
            flat[top_key] = top_val

    print(f"\n[INFO] Flattened checkpoint keys ({len(flat)}):")
    for key in sorted(flat.keys()):
        v = flat[key]
        shape = v.shape if hasattr(v, "shape") else type(v).__name__
        print(f"  {key}: {shape}")

    # --- Extract policy network weights ---
    # Match either "policy_net_N.weight" (flat) or "policy.net_N.weight" (nested).
    # Output uses the flat form expected by sim2sim_mujoco.py.
    policy_state = {}
    for key, value in flat.items():
        norm_key = key.replace("policy.", "policy_")
        if norm_key.startswith("policy_net_") or norm_key.startswith("policy_log_std"):
            policy_state[norm_key] = value
            print(f"  Policy: {key} → {norm_key} {tuple(value.shape)}")

    # --- Extract observation normalizer (RunningStandardScaler) ---
    preprocessor_state = {}
    for key, value in flat.items():
        if "state_preprocessor" in key and "amp" not in key and "value" not in key:
            preprocessor_state[key] = value
            shape = value.shape if hasattr(value, "shape") else value
            print(f"  Preprocessor: {key} → {shape}")

    # --- Extract action log_std (fixed) ---
    log_std = None
    for key, value in flat.items():
        if "log_std" in key:
            log_std = value
            print(f"  Log std: {key} → {value}")

    # --- Extract action scaling (if available) ---
    action_offset = None
    action_scale = None
    for key, value in flat.items():
        if "action_offset" in key:
            action_offset = value
            print(f"  Action offset: {key} → {value.shape}")
        if "action_scale" in key:
            action_scale = value
            print(f"  Action scale: {key} → {value.shape}")

    # --- Save standalone policy ---
    export = {
        "policy_state": policy_state,
        "preprocessor_state": preprocessor_state,
        "log_std": log_std,
        "action_offset": action_offset,
        "action_scale": action_scale,
        "obs_dim": 109,
        "act_dim": 29,
        "hidden_dims": [1024, 512],
    }

    torch.save(export, args.output)
    print(f"\n[SAVED] {args.output}")
    print("Use this file with sim2sim_mujoco.py on your local machine.")


if __name__ == "__main__":
    main()
