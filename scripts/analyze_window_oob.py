#!/usr/bin/env python3
"""
Analyze window_oob.jsonl to find root cause of out-of-map trajectories
in use_segment_wise_sliding_window=True mode.

Usage: python scripts/analyze_window_oob.py [--log logs/window_oob.jsonl] [--thresh 40]
"""

import json
import sys
import argparse
import os
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="logs/window_oob.jsonl")
    p.add_argument("--thresh", type=float, default=40.0, help="OOB threshold (world coords)")
    return p.parse_args()

def is_oob(xy, thresh):
    return abs(xy[0]) > thresh or abs(xy[1]) > thresh

def analyze(log_path, thresh):
    if not os.path.exists(log_path):
        print(f"[ERROR] Log file not found: {log_path}")
        print("  Experiment may not have run yet, or no OOB events detected.")
        return

    episodes = []
    current_episode = None
    current_steps = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event") == "oob_episode":
                if current_episode is not None:
                    episodes.append((current_episode, current_steps))
                current_episode = rec
                current_steps = []
            else:
                current_steps.append(rec)

    if current_episode is not None:
        episodes.append((current_episode, current_steps))

    print(f"=== Window OOB Analysis ===")
    print(f"Log: {log_path}  |  Threshold: ±{thresh}")
    print(f"Total OOB episodes logged: {len(episodes)}\n")

    if not episodes:
        print("No OOB episodes found.")
        return

    # Per-episode analysis
    first_oob_token_pos = defaultdict(int)   # which token position (0=anchor, 1..seg=new)
    first_oob_at_step = []
    anchor_oob_count = 0
    seg_oob_count = 0

    for ep_idx, (hdr, steps) in enumerate(episodes):
        seg_size = hdr.get("seg_size", "?")
        fs = hdr.get("frame_stack", 1)
        n_steps = hdr.get("n_steps", len(steps))
        win_tokens = 1 + (seg_size if isinstance(seg_size, int) else 0)  # 1+seg

        # Find first OOB step and token
        first_oob_step = None
        first_oob_tok = None
        first_oob_xy = None
        anchor_ok_at_first_oob = None

        for step_rec in steps:
            step_idx = step_rec["step"]
            xy_all = step_rec["xy"]    # ((1+seg)*fs, b, 2)
            from_nl = step_rec["from_nl"]  # (1+seg, b)

            n_t_frames = len(xy_all)
            n_b = len(xy_all[0]) if xy_all else 0

            for b in range(n_b):
                # Check each token (group fs frames per token)
                for tok in range(win_tokens):
                    tok_frames = xy_all[tok * fs : (tok + 1) * fs]
                    for frame_xy in tok_frames:
                        if is_oob(frame_xy[b], thresh):
                            if first_oob_step is None:
                                first_oob_step = step_idx
                                first_oob_tok = tok
                                first_oob_xy = frame_xy[b]
                                # Check anchor's state at same step
                                anchor_frames = xy_all[0:fs]
                                anchor_oob = any(is_oob(anchor_frames[f][b], thresh) for f in range(min(fs, len(anchor_frames))))
                                anchor_ok_at_first_oob = not anchor_oob
                                anchor_nl = from_nl[0][b] if from_nl and from_nl[0] else None
                            break
                    if first_oob_step is not None:
                        break
                if first_oob_step is not None:
                    break
            if first_oob_step is not None:
                break

        if first_oob_step is not None:
            first_oob_at_step.append(first_oob_step)
            tok_label = "ANCHOR(t=0)" if first_oob_tok == 0 else f"SEG_TOKEN(t={first_oob_tok})"
            first_oob_token_pos[tok_label] += 1

            if first_oob_tok == 0:
                anchor_oob_count += 1
            else:
                seg_oob_count += 1

            print(f"[Episode {ep_idx+1}] seg_size={seg_size} fs={fs} total_steps={n_steps}")
            print(f"  First OOB at denoising step={first_oob_step}, token={tok_label}")
            print(f"  OOB xy={[round(v,2) for v in first_oob_xy]}")
            print(f"  Anchor valid at that step: {anchor_ok_at_first_oob}")
            print()

    print("=== Summary ===")
    print(f"Total episodes analyzed: {len(episodes)}")
    if first_oob_at_step:
        import statistics
        print(f"First OOB step: min={min(first_oob_at_step)} mean={statistics.mean(first_oob_at_step):.1f} max={max(first_oob_at_step)}")
    print(f"First OOB token distribution:")
    for label, cnt in sorted(first_oob_token_pos.items()):
        print(f"  {label}: {cnt} ({100*cnt/len(episodes):.0f}%)")
    print()
    print("=== Root Cause Hypothesis ===")
    if anchor_oob_count == 0 and seg_oob_count > 0:
        print("ANCHOR is always valid when OOB first appears → SEG TOKENS go OOB first")
        print("Likely cause: positional embedding mismatch (model sees position 1..seg")
        print("  but actual plan position is prefix_len+1..prefix_len+seg)")
        print("  → model produces off-map predictions for 'middle-of-plan' tokens")
    elif anchor_oob_count > 0:
        print("ANCHOR itself is OOB → problem propagates from parent node's plan")
        print("Likely cause: upstream denoising failure passed bad anchor to window")
    else:
        print("Inconclusive — check episode details above")

    # ── Pred_x0 before vs after guidance ────────────────────────────────────
    print()
    print("=== Pred_x0 Before vs After Guidance (last seg token, diverging batch items) ===")
    DIVERGE_THRESH2 = 80.0  # show guidance contribution when x0 exceeds this

    for ep_idx, (hdr, steps) in enumerate(episodes):
        if not steps:
            continue
        seg_size = hdr.get("seg_size", 0)
        fs = hdr.get("frame_stack", 1)
        win_tokens = 1 + (seg_size if isinstance(seg_size, int) else 0)

        # For each step, check if pred_x0_after for any batch exceeds threshold
        found_diverge = False
        for step_rec in steps:
            p0_after = step_rec.get("pred_x0_after")  # list[b][fs][2] or None
            p0_before = step_rec.get("pred_x0_before")
            if p0_after is None:
                continue
            for b_i, b_frames in enumerate(p0_after):
                max_x = max(abs(f[0]) for f in b_frames)
                if max_x > DIVERGE_THRESH2:
                    before_max = None
                    if p0_before and b_i < len(p0_before):
                        before_max = round(max(abs(f[0]) for f in p0_before[b_i]), 1)
                    if not found_diverge:
                        print(f"\n[Episode {ep_idx+1}]")
                        found_diverge = True
                    print(f"  step={step_rec['step']} batch={b_i}: "
                          f"x0_after_max={max_x:.1f}  x0_before_max={before_max}  "
                          f"guidance_delta={round(max_x - (before_max or max_x), 1)}")
                    break  # only show first diverging batch per step
            if found_diverge and step_rec['step'] > 15:
                break  # show only early steps

    # ── Divergence step analysis ─────────────────────────────────────────────
    print()
    print("=== Divergence Step Analysis (last seg token per batch) ===")
    DIVERGE_THRESH = 100.0  # world coords: well past any valid antmaze position

    for ep_idx, (hdr, steps) in enumerate(episodes):
        if not steps:
            continue
        seg_size = hdr.get("seg_size", 0)
        fs = hdr.get("frame_stack", 1)
        win_tokens = 1 + (seg_size if isinstance(seg_size, int) else 0)
        last_tok = win_tokens - 1  # tok index of last seg token

        # Collect max |x| for each batch item of last seg token, per step
        n_b = None
        step_max_x = {}  # {batch_i: [(step, max_x_in_tok_frames), ...]}

        for step_rec in steps:
            step_idx = step_rec["step"]
            xy_all = step_rec["xy"]
            if n_b is None:
                n_b = len(xy_all[0]) if xy_all else 0
            for b in range(n_b):
                tok_frames = xy_all[last_tok * fs : (last_tok + 1) * fs]
                max_x = max(abs(frame_xy[b][0]) for frame_xy in tok_frames)
                step_max_x.setdefault(b, []).append((step_idx, max_x))

        print(f"\n[Episode {ep_idx+1}] anchor=OOB:{hdr.get('event','')}")
        for b in range(n_b or 0):
            vals = step_max_x.get(b, [])
            # Find first diverge step
            first_div = next((s for s, v in vals if v > DIVERGE_THRESH), None)
            # Summarize progression in 10-step buckets
            buckets = {}
            for s, v in vals:
                bucket = (s // 10) * 10
                buckets[bucket] = max(buckets.get(bucket, 0), v)
            bucket_str = "  ".join(f"[{k}:{v:.0f}]" for k, v in sorted(buckets.items()))
            print(f"  batch={b}: first_div_step={first_div}  per-10step-max: {bucket_str}")

if __name__ == "__main__":
    args = parse_args()
    analyze(args.log, args.thresh)
