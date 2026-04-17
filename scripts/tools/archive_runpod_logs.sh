#!/usr/bin/env bash
# Archive AMP-Run training logs/checkpoints on RunPod.
#
# Subcommands:
#   pack      Slim keep-dirs → /tmp/amp_archive, then tar.gz (safe, doesn't touch originals)
#   send      runpodctl send the tarball (prints code)
#   cleanup   Delete zero-value dirs + slim archived dirs (DESTRUCTIVE — run only after local verify)
#   all       = pack (does NOT run send or cleanup)
#
# Typical flow:
#   1) On Pod: bash scripts/tools/archive_runpod_logs.sh pack
#   2) On Pod: bash scripts/tools/archive_runpod_logs.sh send
#   3) On local: runpodctl receive <code> && tar xzf amp_run_archive_*.tar.gz
#   4) Verify locally (ckpt loads, videos play, TB opens)
#   5) On Pod: bash scripts/tools/archive_runpod_logs.sh cleanup

set -euo pipefail

LOGS_ROOT="${LOGS_ROOT:-/workspace/robot_lab/logs/skrl/g1_amp_run}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/tmp/amp_archive}"
DATE_TAG="$(date +%Y%m%d)"
TARBALL="/tmp/amp_run_archive_${DATE_TAG}.tar.gz"

# ── Directory classification ──────────────────────────────────────────────────
# Slim: best_agent + last agent_<iter>.pt + TB events + params + videos
SLIM_DIRS=(
  "2026-04-11_16-21-16_amp_torch:run2b_4mps_baseline_180k"
  "2026-04-16_18-04-40_amp_torch:phase3_cmd06_heading-5_60k"
  "2026-04-16_20-43-29_amp_torch:phase4_main_cmdvel_gating_65k"
  "2026-04-16_22-19-55_amp_torch:phase4plus_abs_heading_35k"
  "2026-04-16_19-53-44_amp_torch:phase5_start_heading-10_35k"
  "2026-04-16_23-45-51_amp_torch:phase5_cont1_20k"
  "2026-04-17_00-31-08_amp_torch:phase5_cont2_35k"
  "2026-04-17_01-19-59_amp_torch:phase5_LATEST_20k_src_of_tb_data"
)

# TB-only (+ videos if present): history for plotting, no checkpoints
TB_ONLY_DIRS=(
  "2026-04-11_22-11-15_amp_torch:run3_heading_drift_60k"
  "2026-04-16_12-55-47_amp_torch:run7_p1_tune_rewards_20k"
  "2026-04-16_15-54-46_amp_torch:run7_p2_remove_runscale_30k"
  "2026-04-16_16-43-48_amp_torch:run7_p3_heading-3_40k"
)

# Zero-value: delete at cleanup
DELETE_DIRS=(
  "2026-04-11_13-49-28_amp_torch"   # Run 1 failed
  "2026-04-11_15-14-03_amp_torch"   # Run 2 early (superseded by 2b)
  "2026-04-11_21-58-32_amp_torch"   # crashed empty
  "2026-04-12_00-00-09_amp_torch"   # Run 3 cont
  "2026-04-12_02-29-52_amp_torch"   # Run 3b failed
  "2026-04-12_04-06-58_amp_torch"   # Run 4 early failed
  "2026-04-15_18-59-41_amp_torch"   # Run 5 crash
  "2026-04-15_22-23-53_amp_torch"   # Run 5b decline
  "2026-04-16_00-44-17_amp_torch"   # Run 5b cont
  "2026-04-16_02-21-02_amp_torch"   # Run 6 straight (local tar.gz exists)
  "2026-04-16_04-24-20_amp_torch"   # Run 6 stand (local tar.gz exists)
  "2026-04-16_17-51-40_amp_torch"   # crashed empty
  "2026-04-17_00-30-59_amp_torch"   # crashed empty
)

# Post-archive slim targets: keep best + last, drop intermediate agent_*.pt
POST_SLIM_DIRS=(
  "2026-04-11_16-21-16_amp_torch"   # 180k run2b, many intermediate ckpts
  "2026-04-16_18-04-40_amp_torch"   # 60k phase3
  "2026-04-16_20-43-29_amp_torch"   # 60k phase4
)

# ── Helpers ──────────────────────────────────────────────────────────────────
slim_copy() {
  local src="$LOGS_ROOT/$1"
  local tag="$2"
  local out="$ARCHIVE_DIR/$tag"

  if [ ! -d "$src" ]; then
    echo "  [skip] $tag: source not found ($src)"
    return
  fi

  mkdir -p "$out/checkpoints"
  cp "$src/checkpoints/best_agent.pt" "$out/checkpoints/" 2>/dev/null || true

  local last
  last=$(ls -t "$src/checkpoints/agent_"*.pt 2>/dev/null | head -1 || true)
  [ -n "$last" ] && cp "$last" "$out/checkpoints/"

  cp "$src"/events.out.tfevents.* "$out/" 2>/dev/null || true
  [ -d "$src/params" ] && cp -r "$src/params" "$out/" 2>/dev/null || true
  [ -d "$src/videos" ] && cp -r "$src/videos" "$out/" 2>/dev/null || true

  echo "  [slim] $tag → $(du -sh "$out" | cut -f1)"
}

tb_copy() {
  local src="$LOGS_ROOT/$1"
  local tag="$2"
  local out="$ARCHIVE_DIR/tb_only/$tag"

  if [ ! -d "$src" ]; then
    echo "  [skip] $tag: source not found ($src)"
    return
  fi

  mkdir -p "$out"
  cp "$src"/events.out.tfevents.* "$out/" 2>/dev/null || true
  [ -d "$src/videos" ] && cp -r "$src/videos" "$out/" 2>/dev/null || true

  echo "  [tb]   $tag → $(du -sh "$out" | cut -f1)"
}

write_readme() {
  cat > "$ARCHIVE_DIR/README.md" <<EOF
# AMP-Run Training Archive (${DATE_TAG})

Source: \`$LOGS_ROOT\` on RunPod.
Created by \`scripts/tools/archive_runpod_logs.sh\`.

## Slim archives (best + last ckpt + TB events + params + videos)

| Tag | Source dir | What it is |
|-----|------------|------------|
| run2b_4mps_baseline_180k | 2026-04-11_16-21-16_amp_torch | Run 2b, 4 m/s first success (pre-Phase era) |
| phase3_cmd06_heading-5_60k | 2026-04-16_18-04-40_amp_torch | Phase 3: cmd_high 0.6 + heading -5 (commit 78c3922) |
| phase4_main_cmdvel_gating_65k | 2026-04-16_20-43-29_amp_torch | Phase 4: cmd_vel gating fix for deceleration bug (commits 9adcc69, 8548271) |
| phase4plus_abs_heading_35k | 2026-04-16_22-19-55_amp_torch | Phase 4+: absolute-angle heading formula (commit 200e19d) |
| phase5_start_heading-10_35k | 2026-04-16_19-53-44_amp_torch | Phase 5 first launch: heading -5 → -10 (commit 5bd48b7) |
| phase5_cont1_20k | 2026-04-16_23-45-51_amp_torch | Phase 5 continuation 1 |
| phase5_cont2_35k | 2026-04-17_00-31-08_amp_torch | Phase 5 continuation 2 |
| **phase5_LATEST_20k_src_of_tb_data** | 2026-04-17_01-19-59_amp_torch | **Source of the TB data pasted in the tuning session — heading_cos 0.9963, fwd_vel 1.58** |

## TB-only (+ videos, history reference)

| Tag | Source dir | What it is |
|-----|------------|------------|
| run3_heading_drift_60k | 2026-04-11_22-11-15_amp_torch | Run 3 heading drift issue |
| run7_p1_tune_rewards_20k | 2026-04-16_12-55-47_amp_torch | Run 7 Phase 1: tune rewards (height×3.3, termination 0.45, AMP 0.5/0.5) |
| run7_p2_remove_runscale_30k | 2026-04-16_15-54-46_amp_torch | Run 7 Phase 2: remove run_scale gating from height penalty |
| run7_p3_heading-3_40k | 2026-04-16_16-43-48_amp_torch | Run 7 Phase 3: heading -1 → -3 |

## Usage

- **Resume training**: use \`agent_<iter>.pt\` (NOT \`best_agent.pt\`, whose eval-return selection can lag current state)
  \`\`\`
  --checkpoint <tag>/checkpoints/agent_<iter>.pt
  \`\`\`
- **Plot history**: \`tensorboard --logdir .\` from the archive root
- **Videos**: each dir's \`videos/\` contains eval mp4s

Full tuning narrative: see \`docs/amp_run_training_log.md\` section 17 in the repo.
EOF
}

# ── Commands ──────────────────────────────────────────────────────────────────
cmd_pack() {
  echo "=== PACK: archiving from $LOGS_ROOT → $ARCHIVE_DIR ==="
  rm -rf "$ARCHIVE_DIR"
  mkdir -p "$ARCHIVE_DIR"

  echo
  echo ">>> Slim (best + last ckpt + TB + videos):"
  for entry in "${SLIM_DIRS[@]}"; do
    slim_copy "${entry%%:*}" "${entry#*:}"
  done

  echo
  echo ">>> TB + videos only:"
  for entry in "${TB_ONLY_DIRS[@]}"; do
    tb_copy "${entry%%:*}" "${entry#*:}"
  done

  write_readme

  echo
  echo ">>> Archive contents:"
  du -sh "$ARCHIVE_DIR"/*
  echo
  echo "Total uncompressed: $(du -sh "$ARCHIVE_DIR" | cut -f1)"

  echo
  echo ">>> Compressing to $TARBALL ..."
  cd "$(dirname "$ARCHIVE_DIR")"
  tar czf "$TARBALL" "$(basename "$ARCHIVE_DIR")"
  echo "Tarball: $(ls -lh "$TARBALL" | awk '{print $5, $9}')"

  echo
  echo ">>> Done. Next: bash $0 send"
}

cmd_send() {
  if [ ! -f "$TARBALL" ]; then
    echo "ERROR: $TARBALL not found. Run 'pack' first."
    exit 1
  fi
  echo "=== SEND: runpodctl send $TARBALL ==="
  runpodctl send "$TARBALL"
}

cmd_cleanup() {
  echo "=== CLEANUP: deleting zero-value dirs + slimming archived dirs ==="
  echo "This is DESTRUCTIVE. Ctrl+C now if you haven't verified the local archive."
  sleep 5

  cd "$LOGS_ROOT"

  echo
  echo ">>> Deleting zero-value dirs:"
  for d in "${DELETE_DIRS[@]}"; do
    if [ -d "$d" ]; then
      local size
      size=$(du -sh "$d" | cut -f1)
      rm -rf "$d"
      echo "  [rm]   $d ($size)"
    fi
  done

  echo
  echo ">>> Slimming archived large dirs (keep best + last agent_*.pt):"
  for d in "${POST_SLIM_DIRS[@]}"; do
    if [ -d "$d" ]; then
      local last
      last=$(ls -t "$d/checkpoints/agent_"*.pt 2>/dev/null | head -1 || true)
      if [ -n "$last" ]; then
        find "$d/checkpoints/" -name "agent_*.pt" ! -wholename "$last" -delete
        echo "  [slim] $d → $(du -sh "$d" | cut -f1)"
      fi
    fi
  done

  echo
  echo ">>> After cleanup:"
  du -sh "$LOGS_ROOT"
}

# ── Entry ─────────────────────────────────────────────────────────────────────
case "${1:-pack}" in
  pack)    cmd_pack ;;
  send)    cmd_send ;;
  cleanup) cmd_cleanup ;;
  all)     cmd_pack ;;
  *)       echo "Usage: $0 {pack|send|cleanup}"; exit 2 ;;
esac
