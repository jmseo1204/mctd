#!/usr/bin/env python3
"""
Loop 2 Visualization: trajectory_obs_positions (3605) vs plan_positions
Shows the critical anomaly where trajectory is stuck at [52.0, 0.0]
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LOG_FILE = "/mnt/c/Users/USER/Desktop/test_ogbench/mctd_repo/logs_memory_debug/run_20260225_030235.jsonl"
OUTPUT_DIR = Path("/mnt/c/Users/USER/Desktop/test_ogbench/mctd_repo/visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load logs
with open(LOG_FILE, 'r') as f:
    for idx, line in enumerate(f):
        try:
            entry = json.loads(line.strip())
            if entry.get("tag") == "rollout.final_state_continuity" and idx > 400 and idx < 500:
                # This is Loop 2
                loop2_data = entry["data"]
                break
        except:
            continue

# Extract data
trajectory_obs_positions = np.array(loop2_data["trajectory_obs_positions"])
plan_positions = np.array(loop2_data["plan_positions"])
parent_qpos = np.array([0.0, 36.0])  # From start log

print(f"Loop 2 Data:")
print(f"  trajectory_length (3603): {loop2_data['trajectory_length']}")
print(f"  trajectory_obs_positions (3605): {trajectory_obs_positions}")
print(f"  plan_positions length: {len(plan_positions)}")
print(f"  Parent position: {parent_qpos}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Loop 2 Anomaly: trajectory_obs_positions (3605) vs plan_positions", 
             fontsize=16, fontweight='bold', color='red')

# --- Plot 1: XY Comparison (Main visualization) ---
ax = axes[0, 0]
ax.plot(plan_positions[:, 0], plan_positions[:, 1], 'b-o', label='Plan Positions', 
        linewidth=2.5, markersize=6, alpha=0.7)
ax.plot(trajectory_obs_positions[:, 0], trajectory_obs_positions[:, 1], 'r*', 
        label='Trajectory Obs (1 point)', markersize=20, zorder=5)
ax.plot(parent_qpos[0], parent_qpos[1], 'g^', label='Parent (Goal)', markersize=12, zorder=4)

ax.set_xlabel('X Position', fontsize=11)
ax.set_ylabel('Y Position', fontsize=11)
ax.set_title('XY Space: Plan vs Trajectory Obs', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Add annotations
ax.annotate(f'Plan Start\n{plan_positions[0]}', 
           xy=plan_positions[0], xytext=(plan_positions[0, 0]-5, plan_positions[0, 1]+5),
           fontsize=9, ha='center',
           bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3),
           arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

ax.annotate(f'Trajectory Stuck\n{trajectory_obs_positions[0]}\n⚠️ ANOMALY', 
           xy=trajectory_obs_positions[0], xytext=(trajectory_obs_positions[0, 0]+3, trajectory_obs_positions[0, 1]-5),
           fontsize=10, ha='left', color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='red', alpha=0.2),
           arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax.annotate(f'Plan End\n{plan_positions[-1]}', 
           xy=plan_positions[-1], xytext=(plan_positions[-1, 0]-5, plan_positions[-1, 1]-3),
           fontsize=9, ha='center',
           bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3),
           arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

# --- Plot 2: Distances over plan frames ---
ax = axes[0, 1]
plan_frames = np.arange(len(plan_positions))

# Distance from trajectory point to each plan frame
distances = [np.linalg.norm(trajectory_obs_positions[0] - plan_positions[i]) 
             for i in range(len(plan_positions))]

ax.bar(plan_frames, distances, color='orange', alpha=0.7, edgecolor='red', linewidth=1.5)
ax.axhline(y=np.mean(distances), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.2f}')
ax.axhline(y=np.max(distances), color='r', linestyle='--', linewidth=2, label=f'Max: {np.max(distances):.2f}')
ax.set_xlabel('Plan Frame Index', fontsize=11)
ax.set_ylabel('Distance to Trajectory Point', fontsize=11)
ax.set_title('Distance: Trajectory Obs to Each Plan Frame', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# --- Plot 3: Plan trajectory path ---
ax = axes[1, 0]
ax.plot(plan_positions[:, 0], plan_positions[:, 1], 'b-o', linewidth=2.5, markersize=5)
ax.plot(plan_positions[0, 0], plan_positions[0, 1], 'go', markersize=12, label='Start', zorder=5)
ax.plot(plan_positions[-1, 0], plan_positions[-1, 1], 'r^', markersize=12, label='End', zorder=5)
ax.plot(parent_qpos[0], parent_qpos[1], 'k*', markersize=15, label='Parent/Goal', zorder=4)

ax.set_xlabel('X Position', fontsize=11)
ax.set_ylabel('Y Position', fontsize=11)
ax.set_title('Expected Plan Path (100 frames)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# --- Plot 4: Summary statistics ---
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
LOOP 2 CRITICAL ANOMALY SUMMARY
{'='*50}

Data from Log File:
  • trajectory_length [3603]: {loop2_data['trajectory_length']} frame
  • trajectory_obs_positions [3605]: {trajectory_obs_positions[0]}
  • plan_positions length: {len(plan_positions)} frames
  
Parent & Plan Info:
  • Parent position (goal): {parent_qpos}
  • Plan first frame: {plan_positions[0]}
  • Plan last frame: {plan_positions[-1]}
  
Key Metrics:
  • Parent → Trajectory: {np.linalg.norm(parent_qpos - trajectory_obs_positions[0]):.4f} units
  • Trajectory → Plan(0): {np.linalg.norm(trajectory_obs_positions[0] - plan_positions[0]):.4f} units
  • Trajectory → Plan(last): {np.linalg.norm(trajectory_obs_positions[0] - plan_positions[-1]):.4f} units
  • Mean distance to plan: {np.mean(distances):.4f} units
  
Status: 🔴 CRITICAL - Early Termination
  • Expected trajectory: ~200 frames
  • Actual trajectory: 1 frame
  • Position: Fixed at [52.0, 0.0] (parent/goal)
  
Root Cause:
  • done flag = True after first env.step()
  • Execution loop broke immediately
  • Only reset observation captured, no rollout
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loop2_visualization.png", dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {OUTPUT_DIR / 'loop2_visualization.png'}")

# Print detailed analysis
print("\n" + "="*80)
print("LOOP 2 DETAILED ANALYSIS")
print("="*80)
print(f"\ndf_planning.py:3603 - trajectory_length: {loop2_data['trajectory_length']}")
print(f"df_planning.py:3605 - trajectory_obs_positions:")
for i, pos in enumerate(trajectory_obs_positions):
    print(f"  Frame {i}: {pos}")

print(f"\nPlan has {len(plan_positions)} frames but trajectory only has {loop2_data['trajectory_length']} frame")
print(f"Trajectory is stuck at {trajectory_obs_positions[0]} (the parent/goal position)")
print(f"\nThis is a STRUCTURAL BUG in _execute_plan_in_env:")
print(f"  - done.any() returned True after first env.step()")
print(f"  - Loop broke at line 3377-3378 in df_planning.py")
print(f"  - Only 1 frame collected instead of ~200")

plt.show()

EOF
python3 /mnt/c/Users/USER/Desktop/test_ogbench/mctd_repo/loop2_visualization.py
