#!/usr/bin/env python3
"""
Analyze rollout logs and generate trajectory visualization images.
Reads JSONL log files from logs_memory_debug/ and creates visualization images.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime

def load_jsonl_logs(log_path: str) -> List[Dict[str, Any]]:
    """Load JSONL log file"""
    logs = []
    with open(log_path) as f:
        for line in f:
            logs.append(json.loads(line))
    return logs

def extract_rollout_logs(logs: List[Dict]) -> List[Dict[str, Any]]:
    """Extract rollout-related logs from all logs"""
    rollout_logs = []
    for log in logs:
        if 'rollout' in log.get('tag', ''):
            rollout_logs.append(log)
    return rollout_logs

def group_rollout_pairs(rollout_logs: List[Dict]) -> List[Dict[str, Any]]:
    """Group pre/post rollout logs into pairs"""
    pairs = []
    i = 0
    while i < len(rollout_logs):
        if rollout_logs[i]['tag'] == 'rollout.plan_slice_continuity':
            pre_log = rollout_logs[i]['data']
            
            if i + 1 < len(rollout_logs) and rollout_logs[i+1]['tag'] == 'rollout.final_state_continuity':
                post_log = rollout_logs[i+1]['data']
                
                pairs.append({
                    'pre': pre_log,
                    'post': post_log,
                    'pre_ts': rollout_logs[i]['ts'],
                    'post_ts': rollout_logs[i+1]['ts'],
                })
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return pairs

def create_trajectory_visualization(rollout_pair: Dict, save_path: Path, idx: int) -> str:
    """Create and save trajectory visualization image"""
    
    pre_data = rollout_pair['pre']
    post_data = rollout_pair['post']
    
    # Extract trajectory positions from post_data (where it's actually logged)
    trajectory_obs_positions = np.array(post_data.get('trajectory_obs_positions', []))
    plan_positions = np.array(post_data.get('plan_positions', []))
    
    if len(trajectory_obs_positions) == 0:
        print(f"  [Rollout {idx}] No trajectory positions found, skipping")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot executed trajectory
    if len(trajectory_obs_positions) > 0:
        ax.plot(trajectory_obs_positions[:, 0], trajectory_obs_positions[:, 1], 
                'b-', linewidth=2.5, label='Executed trajectory', alpha=0.8)
        ax.scatter(trajectory_obs_positions[0, 0], trajectory_obs_positions[0, 1], 
                  color='green', s=150, marker='o', label='Start (executed)', zorder=5, edgecolors='darkgreen', linewidth=2)
        ax.scatter(trajectory_obs_positions[-1, 0], trajectory_obs_positions[-1, 1], 
                  color='red', s=150, marker='s', label='End (executed)', zorder=5, edgecolors='darkred', linewidth=2)
    
    # Plot plan trajectory for comparison
    if len(plan_positions) > 0:
        ax.plot(plan_positions[:, 0], plan_positions[:, 1], 
                'r--', linewidth=1.5, alpha=0.6, label='Plan (reference)')
        ax.scatter(plan_positions[0, 0], plan_positions[0, 1], 
                  color='orange', s=100, marker='^', label='Plan start', zorder=4, alpha=0.7)
        ax.scatter(plan_positions[-1, 0], plan_positions[-1, 1], 
                  color='purple', s=100, marker='v', label='Plan end', zorder=4, alpha=0.7)
    
    # Add parent position annotation
    parent_qpos = pre_data.get('parent_qpos', [])
    if parent_qpos:
        ax.scatter(parent_qpos[0], parent_qpos[1], 
                  color='cyan', s=200, marker='*', label='Parent state', zorder=6, edgecolors='darkcyan', linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('X position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y position', fontsize=12, fontweight='bold')
    
    start_idx = pre_data.get('start_idx', '?')
    end_idx = pre_data.get('end_idx', '?')
    traj_len = len(trajectory_obs_positions)
    plan_len = len(plan_positions)
    
    ax.set_title(
        f'Rollout #{idx}: Trajectory Visualization\n'
        f'Plan slice [{start_idx}:{end_idx}] | '
        f'Executed length: {traj_len} | Plan length: {plan_len}',
        fontsize=13, fontweight='bold'
    )
    
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # Save figure
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return str(save_path)

def generate_summary_report(rollout_pairs: List[Dict], output_dir: Path) -> None:
    """Generate a summary report with all rollout metrics"""
    
    report_path = output_dir / 'rollout_summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("ROLLOUT TRAJECTORY ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        for idx, pair in enumerate(rollout_pairs, 1):
            pre = pair['pre']
            post = pair['post']
            
            f.write(f"Rollout #{idx}\n")
            f.write("-" * 100 + "\n")
            
            # Plan info
            f.write(f"  Plan slice range: [{pre.get('start_idx', '?')}:{pre.get('end_idx', '?')}]\n")
            f.write(f"  Plan shape: {pre.get('plan_slice_shape', 'unknown')}\n")
            f.write(f"  Trajectory shape: {pre.get('trajectory_shape', 'unknown')}\n\n")
            
            # Continuity info
            f.write("  State Continuity Analysis:\n")
            parent_qpos = pre.get('parent_qpos', [])
            plan_first = pre.get('plan_slice_first_qpos', [])
            first_diff = pre.get('first_frame_diff', 0)
            
            f.write(f"    Parent state:          {parent_qpos}\n")
            f.write(f"    Plan first position:   {plan_first}\n")
            f.write(f"    First frame diff:      {first_diff:.4f}\n\n")
            
            # Execution result
            plan_last = post.get('plan_slice_last_qpos', [])
            final_qpos = post.get('final_sim_state_qpos', [])
            last_diff = post.get('last_frame_diff', 0)
            
            f.write("  Execution Result Analysis:\n")
            f.write(f"    Plan final position:   {plan_last}\n")
            f.write(f"    Executed final state:  {final_qpos}\n")
            f.write(f"    Final frame diff:      {last_diff:.4f}\n\n")
            
            # Trajectory length
            traj_len = pre.get('trajectory_obs_positions_length', 0)
            f.write(f"  Executed trajectory length: {traj_len} frames\n\n")
            
        f.write("=" * 100 + "\n")
    
    print(f"✓ Summary report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze rollout logs and generate visualizations')
    parser.add_argument('--log-file', type=str, 
                       help='Path to JSONL log file (if not specified, uses most recent)')
    parser.add_argument('--output-dir', type=str, default='rollout_visualizations',
                       help='Output directory for visualization images')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip generating summary report')
    
    args = parser.parse_args()
    
    # Find log file
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        # Find most recent log file
        log_dir = Path('logs_memory_debug')
        log_files = sorted(log_dir.glob('*.jsonl'), key=lambda x: x.stat().st_mtime, reverse=True)
        if not log_files:
            print("❌ No JSONL log files found in logs_memory_debug/")
            return
        log_path = log_files[0]
    
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return
    
    print(f"📖 Loading log file: {log_path}")
    logs = load_jsonl_logs(str(log_path))
    print(f"   Loaded {len(logs)} log entries")
    
    # Extract rollout logs
    rollout_logs = extract_rollout_logs(logs)
    print(f"📊 Found {len(rollout_logs)} rollout-related log entries")
    
    # Group into pairs
    rollout_pairs = group_rollout_pairs(rollout_logs)
    print(f"✓ Grouped into {len(rollout_pairs)} rollout pairs")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n🎨 Generating visualization images...")
    
    # Generate visualizations
    image_paths = []
    for idx, pair in enumerate(rollout_pairs, 1):
        image_path = output_dir / f"rollout_{idx:03d}.png"
        try:
            result = create_trajectory_visualization(pair, image_path, idx)
            if result:
                image_paths.append(result)
                print(f"  ✓ Rollout {idx}: {image_path.name}")
        except Exception as e:
            print(f"  ❌ Rollout {idx}: Failed - {str(e)}")
    
    print(f"\n✓ Generated {len(image_paths)} visualization images")
    
    # Generate summary report
    if not args.no_summary:
        generate_summary_report(rollout_pairs, output_dir)
    
    print(f"\n📁 All outputs saved to: {output_dir.absolute()}")
    print(f"   HTML report: {(output_dir / 'rollout_summary_report.txt').name}")

if __name__ == '__main__':
    main()
