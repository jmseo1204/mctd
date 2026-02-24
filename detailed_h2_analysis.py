#!/usr/bin/env python3
"""
Detailed H2 Analysis: First Noisy Token Initialization
Focus on whether noisy_parts initialization is the root cause
"""

import json
import numpy as np
import re

def load_logs(log_file):
    records = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except:
                pass
    return records

def extract_stats(record):
    if 'data' not in record:
        return None
    data = record['data']
    if 'mean' in data and 'std' in data:
        return {
            'mean': float(data.get('mean', 0)),
            'std': float(data.get('std', 0)),
            'min': float(data.get('min', 0)),
            'max': float(data.get('max', 0))
        }
    return None

def main():
    log_file = "logs_memory_debug/2026-02-24_15-07-17.jsonl"
    records = load_logs(log_file)
    print(f"📚 Loaded {len(records)} records\n")

    print("="*80)
    print("🔬 DETAILED H2 ANALYSIS: First Noisy Token Initialization")
    print("="*80)

    # Collect all relevant data
    first_noisy_data = {}
    first_token_data = {}
    denoising_updates = {}
    noise_level_checks = {}

    for r in records:
        tag = r.get('tag', '')

        if tag == 'plan.segment_boundary.first_noisy':
            label = r.get('label', '')
            depth_match = re.search(r'depth_(\d+)', label)
            depth = int(depth_match.group(1)) if depth_match else 0
            stats = extract_stats(r)
            if stats:
                if depth not in first_noisy_data:
                    first_noisy_data[depth] = []
                first_noisy_data[depth].append(stats)

        elif tag == 'plan.segment_boundary.first_token':
            label = r.get('label', '')
            depth_match = re.search(r'depth_(\d+)', label)
            depth = int(depth_match.group(1)) if depth_match else 0
            stats = extract_stats(r)
            if stats:
                if depth not in first_token_data:
                    first_token_data[depth] = []
                first_token_data[depth].append(stats)

        elif tag == 'denoising.token_0.before_update':
            step = r.get('step', -1)
            stats = extract_stats(r)
            if stats:
                denoising_updates[step] = denoising_updates.get(step, {})
                denoising_updates[step]['before'] = stats

        elif tag == 'denoising.token_0.after_update':
            step = r.get('step', -1)
            stats = extract_stats(r)
            if stats:
                denoising_updates[step] = denoising_updates.get(step, {})
                denoising_updates[step]['after'] = stats

        elif tag == 'mcts.noise_level.check':
            data = r.get('data', {})
            noise_level_checks[len(noise_level_checks)] = data

    # ANALYSIS 1: First Noisy Token Statistics
    print("\n📊 ANALYSIS 1: First Noisy Token (Raw Initialization)")
    for depth in sorted(first_noisy_data.keys()):
        stats_list = first_noisy_data[depth]
        means = [s['mean'] for s in stats_list]
        stds = [s['std'] for s in stats_list]

        print(f"\n  Depth {depth}:")
        print(f"    Count: {len(stats_list)}")
        print(f"    Mean (avg of means): {np.mean(means):.6f}")
        print(f"    Std (avg of stds): {np.mean(stds):.6f}")
        print(f"    Std deviation of means: {np.std(means):.6f}")
        print(f"    Max std observed: {np.max(stds):.6f}")

        if np.mean(stds) > 0.5:
            print(f"    ⚠️  HIGH VARIANCE - first_noisy not properly initialized!")

    # ANALYSIS 2: First Token vs First Noisy
    print("\n📊 ANALYSIS 2: First Token (position 0) vs First Noisy (position 1)")
    for depth in sorted(set(first_token_data.keys()) | set(first_noisy_data.keys())):
        ft_stats = first_token_data.get(depth, [])
        fn_stats = first_noisy_data.get(depth, [])

        if ft_stats and fn_stats:
            ft_means = [s['mean'] for s in ft_stats]
            ft_stds = [s['std'] for s in ft_stats]
            fn_means = [s['mean'] for s in fn_stats]
            fn_stds = [s['std'] for s in fn_stats]

            print(f"\n  Depth {depth}:")
            print(f"    First Token (pos 0):")
            print(f"      mean={np.mean(ft_means):.6f}, std={np.mean(ft_stds):.6f}")
            print(f"    First Noisy (pos 1):")
            print(f"      mean={np.mean(fn_means):.6f}, std={np.mean(fn_stds):.6f}")

            # Check if they're different (indicating denoising)
            mean_ratio = abs(np.mean(fn_means)) / (abs(np.mean(ft_means)) + 1e-6)
            std_ratio = np.mean(fn_stds) / (np.mean(ft_stds) + 1e-6)

            print(f"    Mean magnitude ratio (fn/ft): {mean_ratio:.2f}x")
            print(f"    Std ratio (fn/ft): {std_ratio:.2f}x")

            if std_ratio > 1.2:
                print(f"    ⚠️  First noisy has HIGHER std - denoising not applied!")

    # ANALYSIS 3: Denoising Loop Evolution
    print("\n📊 ANALYSIS 3: Token 0 Evolution During Denoising Loop")
    if denoising_updates:
        before_stds = []
        after_stds = []

        for step in sorted(denoising_updates.keys())[:10]:  # First 10 steps
            update_data = denoising_updates[step]
            if 'before' in update_data and 'after' in update_data:
                before = update_data['before']
                after = update_data['after']
                before_stds.append(before['std'])
                after_stds.append(after['std'])

                if step < 5:  # Show first few steps
                    print(f"  Step {step}:")
                    print(f"    Before: mean={before['mean']:.6f}, std={before['std']:.6f}")
                    print(f"    After:  mean={after['mean']:.6f}, std={after['std']:.6f}")

        # Check if std is decreasing (denoising happening)
        if len(before_stds) > 1:
            std_changes = [after_stds[i] - before_stds[i] for i in range(len(before_stds))]
            avg_change = np.mean(std_changes)
            print(f"\n  Average std change per step: {avg_change:.6f}")
            if avg_change > -0.01:  # Std not decreasing much
                print(f"  ⚠️  STD NOT DECREASING - Token 0 not being denoised!")
            else:
                print(f"  ✓ Std is decreasing (denoising working)")

    # ANALYSIS 4: Noise Level Checks
    print("\n📊 ANALYSIS 4: Noise Level Checks (all_first_seg_zero)")
    if noise_level_checks:
        all_zero_count = sum(1 for d in noise_level_checks.values() if d.get('all_first_seg_zero', False))
        total_count = len(noise_level_checks)
        print(f"  Total checks: {total_count}")
        print(f"  First segment fully denoised (==0): {all_zero_count}/{total_count}")

        if all_zero_count == total_count:
            print(f"  ✓ All checks show first segment denoised to noise_level=0")
        else:
            print(f"  ⚠️  Not all checks show complete denoising")

    # FINAL DIAGNOSIS
    print("\n" + "="*80)
    print("🔍 DIAGNOSIS")
    print("="*80)

    # Check various indicators
    h2_indicators = []

    # Indicator 1: High first_noisy std
    if first_noisy_data:
        avg_fn_std = np.mean([s['std'] for stats_list in first_noisy_data.values() for s in stats_list])
        if avg_fn_std > 0.5:
            h2_indicators.append(f"✓ High first_noisy std ({avg_fn_std:.4f})")

    # Indicator 2: first_noisy std > first_token std
    if first_noisy_data and first_token_data:
        fn_stds_avg = np.mean([s['std'] for stats_list in first_noisy_data.values() for s in stats_list])
        ft_stds_avg = np.mean([s['std'] for stats_list in first_token_data.values() for s in stats_list])
        # Note: first_token includes obs_parent, so it might be different

    # Indicator 3: Std not decreasing in denoising loop
    if len(before_stds) > 1:
        avg_change = np.mean([after_stds[i] - before_stds[i] for i in range(len(before_stds))])
        if avg_change > -0.01:
            h2_indicators.append(f"✓ Token 0 std not decreasing ({avg_change:+.6f}/step)")

    print("\nH2 Evidence:")
    if h2_indicators:
        for indicator in h2_indicators:
            print(f"  {indicator}")
        print(f"\n✅ H2 LIKELY - First noisy token not properly denoised")
    else:
        print(f"  No strong H2 indicators")
        print(f"\n❌ H2 UNLIKELY")

if __name__ == '__main__':
    main()
