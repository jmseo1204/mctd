#!/usr/bin/env python3
"""
Comprehensive trajectory divergence analysis and visualization.
Extracts plan_positions vs trajectory_obs_positions from log file and analyzes divergence patterns.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Configuration
LOG_FILE = "/mnt/c/Users/USER/Desktop/test_ogbench/mctd_repo/logs_memory_debug/run_20260225_030235.jsonl"
OUTPUT_DIR = Path("/mnt/c/Users/USER/Desktop/test_ogbench/mctd_repo/trajectory_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

class TrajectoryAnalyzer:
    """Analyze trajectory divergence from bidirectional MCTS planning logs."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.continuity_logs: Dict[str, List[Dict]] = {
            "plan_slice_continuity": [],
            "final_state_continuity": []
        }
        self.pre_rollout_logs: List[Dict] = []
        self.post_rollout_logs: List[Dict] = []
        
    def load_logs(self):
        """Load and parse JSONL log file."""
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    tag = entry.get("tag")
                    
                    if tag == "rollout.plan_slice_continuity":
                        self.continuity_logs["plan_slice_continuity"].append(entry)
                    elif tag == "rollout.final_state_continuity":
                        self.continuity_logs["final_state_continuity"].append(entry)
                    elif tag == "bidir_mcts._rollout_leaf_plan.pre_rollout":
                        self.pre_rollout_logs.append(entry)
                    elif tag == "bidir_mcts._rollout_leaf_plan.post_rollout":
                        self.post_rollout_logs.append(entry)
                except json.JSONDecodeError:
                    continue
                    
    def analyze_loop(self, loop_idx: int) -> Dict[str, Any]:
        """Analyze a single loop (1-indexed)."""
        # Find corresponding continuity logs
        if loop_idx <= len(self.continuity_logs["plan_slice_continuity"]):
            start_log = self.continuity_logs["plan_slice_continuity"][loop_idx - 1]
            end_log = self.continuity_logs["final_state_continuity"][loop_idx - 1]
        else:
            return {}
            
        # Extract data
        plan_positions = np.array(end_log["data"]["plan_positions"])
        trajectory_positions = np.array(end_log["data"]["trajectory_obs_positions"])
        trajectory_length = end_log["data"]["trajectory_length"]
        
        # Calculate divergence metrics
        divergence_per_frame = []
        if len(trajectory_positions) > 0:
            for i, traj_pos in enumerate(trajectory_positions):
                if i < len(plan_positions):
                    # Euclidean distance from plan to trajectory
                    dist = np.linalg.norm(traj_pos - plan_positions[i])
                    divergence_per_frame.append(dist)
        
        # Calculate statistics
        analysis = {
            "loop": loop_idx,
            "parent_qpos": np.array(start_log["data"]["parent_qpos"]),
            "plan_slice_first_qpos": np.array(start_log["data"]["plan_slice_first_qpos"]),
            "plan_slice_last_qpos": np.array(end_log["data"]["plan_slice_last_qpos"]),
            "trajectory_first_qpos": np.array(end_log["data"]["trajectory_obs_first"]),
            "trajectory_last_qpos": np.array(end_log["data"]["trajectory_obs_last"]),
            "first_frame_diff": start_log["data"]["first_frame_diff"],
            "last_frame_diff": end_log["data"]["last_frame_diff"],
            "trajectory_length": trajectory_length,
            "plan_length": len(plan_positions),
            "plan_positions": plan_positions,
            "trajectory_positions": trajectory_positions,
            "divergence_per_frame": divergence_per_frame,
            "divergence_mean": np.mean(divergence_per_frame) if divergence_per_frame else 0,
            "divergence_max": np.max(divergence_per_frame) if divergence_per_frame else 0,
            "divergence_std": np.std(divergence_per_frame) if divergence_per_frame else 0,
        }
        
        return analysis
    
    def print_summary(self):
        """Print comprehensive summary of all loops."""
        print("\n" + "="*80)
        print("TRAJECTORY DIVERGENCE ANALYSIS SUMMARY")
        print("="*80 + "\n")
        
        for loop_idx in range(1, 5):
            analysis = self.analyze_loop(loop_idx)
            if not analysis:
                continue
                
            print(f"\n{'─'*80}")
            print(f"LOOP {loop_idx} ANALYSIS")
            print(f"{'─'*80}")
            
            print(f"Start Discontinuity:")
            print(f"  Parent position:       {analysis['parent_qpos']}")
            print(f"  Plan 1st frame:        {analysis['plan_slice_first_qpos']}")
            print(f"  Distance:              {analysis['first_frame_diff']:.4f}")
            
            print(f"\nEnd Discontinuity:")
            print(f"  Plan last frame:       {analysis['plan_slice_last_qpos']}")
            print(f"  Trajectory last frame: {analysis['trajectory_last_qpos']}")
            print(f"  Distance:              {analysis['last_frame_diff']:.4f}")
            
            print(f"\nFrame Counts:")
            print(f"  Plan frames:           {analysis['plan_length']}")
            print(f"  Trajectory frames:     {analysis['trajectory_length']}")
            
            print(f"\nFrame-by-Frame Divergence:")
            if analysis['divergence_per_frame']:
                print(f"  Mean:                  {analysis['divergence_mean']:.4f}")
                print(f"  Max:                   {analysis['divergence_max']:.4f}")
                print(f"  Std Dev:               {analysis['divergence_std']:.4f}")
                
                # Find frame with max divergence
                max_frame_idx = np.argmax(analysis['divergence_per_frame'])
                print(f"  Max divergence at frame {max_frame_idx}: {analysis['divergence_max']:.4f}")
            else:
                print(f"  (No trajectory data)")
    
    def plot_loop_trajectory(self, loop_idx: int, save_path: Path = None):
        """Plot trajectory vs plan for a single loop."""
        analysis = self.analyze_loop(loop_idx)
        if not analysis or len(analysis['plan_positions']) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Loop {loop_idx} - Trajectory Divergence Analysis", fontsize=14, fontweight='bold')
        
        plan_pos = analysis['plan_positions']
        traj_pos = analysis['trajectory_positions']
        
        # --- Plot 1: XY trajectory comparison ---
        ax = axes[0, 0]
        ax.plot(plan_pos[:, 0], plan_pos[:, 1], 'b-o', label='Plan Path', linewidth=2, markersize=4, alpha=0.7)
        ax.plot(traj_pos[:, 0], traj_pos[:, 1], 'r-s', label='Trajectory Path', linewidth=2, markersize=4, alpha=0.7)
        
        # Mark start and end
        ax.plot(plan_pos[0, 0], plan_pos[0, 1], 'bo', markersize=10, label='Plan Start')
        ax.plot(plan_pos[-1, 0], plan_pos[-1, 1], 'b^', markersize=10, label='Plan End')
        ax.plot(traj_pos[0, 0], traj_pos[0, 1], 'rs', markersize=10, label='Traj Start')
        ax.plot(traj_pos[-1, 0], traj_pos[-1, 1], 'r^', markersize=10, label='Traj End')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('XY Trajectory Comparison')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # --- Plot 2: X position over time ---
        ax = axes[0, 1]
        plan_frames = np.arange(len(plan_pos))
        traj_frames = np.arange(len(traj_pos))
        ax.plot(plan_frames, plan_pos[:, 0], 'b-o', label='Plan X', linewidth=2, markersize=4)
        ax.plot(traj_frames, traj_pos[:, 0], 'r-s', label='Trajectory X', linewidth=2, markersize=4)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('X Position')
        ax.set_title('X Position Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # --- Plot 3: Y position over time ---
        ax = axes[1, 0]
        ax.plot(plan_frames, plan_pos[:, 1], 'b-o', label='Plan Y', linewidth=2, markersize=4)
        ax.plot(traj_frames, traj_pos[:, 1], 'r-s', label='Trajectory Y', linewidth=2, markersize=4)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Y Position')
        ax.set_title('Y Position Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # --- Plot 4: Frame-by-frame divergence ---
        ax = axes[1, 1]
        if analysis['divergence_per_frame']:
            divergence = analysis['divergence_per_frame']
            frame_indices = np.arange(len(divergence))
            ax.bar(frame_indices, divergence, color='orange', alpha=0.7)
            ax.axhline(y=analysis['divergence_mean'], color='g', linestyle='--', label=f"Mean: {analysis['divergence_mean']:.4f}")
            ax.axhline(y=analysis['divergence_max'], color='r', linestyle='--', label=f"Max: {analysis['divergence_max']:.4f}")
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Euclidean Distance')
            ax.set_title('Frame-by-Frame Divergence')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
    
    def plot_all_loops_comparison(self, save_path: Path = None):
        """Plot all 4 loops for comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("All Loops - XY Trajectory Comparison", fontsize=14, fontweight='bold')
        axes = axes.flatten()
        
        for loop_idx in range(1, 5):
            analysis = self.analyze_loop(loop_idx)
            if not analysis or len(analysis['plan_positions']) == 0:
                continue
                
            ax = axes[loop_idx - 1]
            plan_pos = analysis['plan_positions']
            traj_pos = analysis['trajectory_positions']
            
            ax.plot(plan_pos[:, 0], plan_pos[:, 1], 'b-o', label='Plan', linewidth=2, markersize=4, alpha=0.7)
            ax.plot(traj_pos[:, 0], traj_pos[:, 1], 'r-s', label='Trajectory', linewidth=2, markersize=4, alpha=0.7)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f"Loop {loop_idx} (Start Δ: {analysis['first_frame_diff']:.2f}, End Δ: {analysis['last_frame_diff']:.2f})")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
    
    def plot_divergence_comparison(self, save_path: Path = None):
        """Plot divergence metrics across all loops."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Divergence Metrics Across All Loops", fontsize=14, fontweight='bold')
        
        loops = []
        start_disconts = []
        end_disconts = []
        max_divergences = []
        mean_divergences = []
        
        for loop_idx in range(1, 5):
            analysis = self.analyze_loop(loop_idx)
            if not analysis:
                continue
            loops.append(f"Loop {loop_idx}")
            start_disconts.append(analysis['first_frame_diff'])
            end_disconts.append(analysis['last_frame_diff'])
            max_divergences.append(analysis['divergence_max'])
            mean_divergences.append(analysis['divergence_mean'])
        
        # Plot 1: Start and End discontinuities
        ax = axes[0]
        x = np.arange(len(loops))
        width = 0.35
        ax.bar(x - width/2, start_disconts, width, label='Start Discontinuity', color='skyblue')
        ax.bar(x + width/2, end_disconts, width, label='End Discontinuity', color='lightcoral')
        ax.set_ylabel('Distance (units)')
        ax.set_title('Start vs End Discontinuities')
        ax.set_xticks(x)
        ax.set_xticklabels(loops)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Threshold lines
        ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Warning Threshold (5.0)')
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Critical Threshold (20.0)')
        ax.legend()
        
        # Plot 2: Max and Mean frame-by-frame divergence
        ax = axes[1]
        x = np.arange(len(loops))
        width = 0.35
        ax.bar(x - width/2, max_divergences, width, label='Max Divergence', color='orange')
        ax.bar(x + width/2, mean_divergences, width, label='Mean Divergence', color='green')
        ax.set_ylabel('Distance (units)')
        ax.set_title('Frame-by-Frame Divergence Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(loops)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

def main():
    """Main execution."""
    analyzer = TrajectoryAnalyzer(LOG_FILE)
    analyzer.load_logs()
    
    # Print comprehensive summary
    analyzer.print_summary()
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Individual loop plots
    for loop_idx in range(1, 5):
        output_file = OUTPUT_DIR / f"loop_{loop_idx}_trajectory_analysis.png"
        analyzer.plot_loop_trajectory(loop_idx, save_path=output_file)
    
    # Comparison plots
    output_file = OUTPUT_DIR / "all_loops_xy_comparison.png"
    analyzer.plot_all_loops_comparison(save_path=output_file)
    
    output_file = OUTPUT_DIR / "divergence_metrics_comparison.png"
    analyzer.plot_divergence_comparison(save_path=output_file)
    
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
