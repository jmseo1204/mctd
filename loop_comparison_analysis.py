#!/usr/bin/env python3
"""
Comprehensive Loop Analysis: df_planning.py:3539 vs 3596 for all 4 loops
Compares start vs end discontinuities and trajectory characteristics
"""

import json
import numpy as np
from pathlib import Path

# Log file path
LOG_FILE = "/mnt/c/Users/USER/Desktop/test_ogbench/mctd_repo/logs_memory_debug/run_20260225_030235.jsonl"

def load_logs():
    """Load and parse JSONL log file."""
    start_continuity = []  # From df_planning.py:3539
    end_continuity = []    # From df_planning.py:3596
    
    with open(LOG_FILE, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                tag = entry.get("tag")
                
                if tag == "rollout.plan_slice_continuity":
                    start_continuity.append(entry)
                elif tag == "rollout.final_state_continuity":
                    end_continuity.append(entry)
            except json.JSONDecodeError:
                continue
    
    return start_continuity, end_continuity

def analyze_loop(loop_idx, start_log, end_log):
    """Analyze a single loop comparing start and end states."""
    
    start_data = start_log["data"]
    end_data = end_log["data"]
    
    # Extract key values
    parent_qpos = np.array(start_data["parent_qpos"])
    plan_slice_first_qpos = np.array(start_data["plan_slice_first_qpos"])
    first_frame_diff = start_data["first_frame_diff"]
    
    plan_slice_last_qpos = np.array(end_data["plan_slice_last_qpos"])
    final_sim_state_qpos = np.array(end_data["final_sim_state_qpos"])
    last_frame_diff = end_data["last_frame_diff"]
    
    trajectory_length = end_data["trajectory_length"]
    plan_length = len(end_data["plan_positions"])
    trajectory_obs_positions = np.array(end_data["trajectory_obs_positions"])
    plan_positions = np.array(end_data["plan_positions"])
    
    # Calculate additional metrics
    trajectory_first_obs = trajectory_obs_positions[0] if len(trajectory_obs_positions) > 0 else None
    trajectory_last_obs = trajectory_obs_positions[-1] if len(trajectory_obs_positions) > 0 else None
    
    # Distance from parent to first trajectory obs
    parent_to_traj_first = np.linalg.norm(parent_qpos - trajectory_first_obs) if trajectory_first_obs is not None else None
    
    # Distance from parent to first plan frame
    parent_to_plan_first = np.linalg.norm(parent_qpos - plan_slice_first_qpos)
    
    # Distance between last plan and last trajectory
    plan_to_traj_last = np.linalg.norm(plan_slice_last_qpos - trajectory_last_obs) if trajectory_last_obs is not None else None
    
    # Frame-by-frame divergence analysis
    divergence_per_frame = []
    if len(trajectory_obs_positions) > 0 and len(plan_positions) > 0:
        min_len = min(len(trajectory_obs_positions), len(plan_positions))
        for i in range(min_len):
            dist = np.linalg.norm(trajectory_obs_positions[i] - plan_positions[i])
            divergence_per_frame.append(dist)
    
    return {
        "loop": loop_idx,
        "parent_qpos": parent_qpos,
        "plan_slice_first_qpos": plan_slice_first_qpos,
        "plan_slice_last_qpos": plan_slice_last_qpos,
        "trajectory_first_obs": trajectory_first_obs,
        "trajectory_last_obs": trajectory_last_obs,
        "final_sim_state_qpos": final_sim_state_qpos,
        
        # Start (3539)
        "first_frame_diff": first_frame_diff,
        "parent_to_plan_first": parent_to_plan_first,
        "parent_to_traj_first": parent_to_traj_first,
        
        # End (3596)
        "last_frame_diff": last_frame_diff,
        "plan_to_traj_last": plan_to_traj_last,
        
        # Trajectory characteristics
        "trajectory_length": trajectory_length,
        "plan_length": plan_length,
        "trajectory_total_distance": np.sum([np.linalg.norm(trajectory_obs_positions[i+1] - trajectory_obs_positions[i]) 
                                             for i in range(len(trajectory_obs_positions)-1)]) if len(trajectory_obs_positions) > 1 else 0,
        "plan_total_distance": np.sum([np.linalg.norm(plan_positions[i+1] - plan_positions[i]) 
                                       for i in range(len(plan_positions)-1)]) if len(plan_positions) > 1 else 0,
        
        # Divergence stats
        "divergence_per_frame": divergence_per_frame,
        "divergence_mean": np.mean(divergence_per_frame) if divergence_per_frame else 0,
        "divergence_max": np.max(divergence_per_frame) if divergence_per_frame else 0,
        "divergence_std": np.std(divergence_per_frame) if divergence_per_frame else 0,
    }

def severity_level(value):
    """Classify severity based on discontinuity value."""
    if value < 2.0:
        return "✓ PASS"
    elif value < 5.0:
        return "⚠ WARNING"
    elif value < 20.0:
        return "✗ ERROR"
    else:
        return "✗✗ CRITICAL"

def print_full_comparison():
    """Print comprehensive comparison of all loops."""
    
    start_logs, end_logs = load_logs()
    
    if len(start_logs) != 4 or len(end_logs) != 4:
        print(f"Error: Expected 4 loops, got {len(start_logs)} start and {len(end_logs)} end logs")
        return
    
    # Analyze all loops
    analyses = []
    for i in range(4):
        analysis = analyze_loop(i+1, start_logs[i], end_logs[i])
        analyses.append(analysis)
    
    # Print header
    print("\n" + "="*150)
    print("LOOP COMPARISON ANALYSIS: df_planning.py:3539 vs 3596")
    print("="*150 + "\n")
    
    # Print detailed comparison for each loop
    for analysis in analyses:
        loop_num = analysis["loop"]
        print(f"\n{'─'*150}")
        print(f"LOOP {loop_num} - DETAILED COMPARISON")
        print(f"{'─'*150}")
        
        # ========== START OF ROLLOUT (3539) ==========
        print(f"\n[df_planning.py:3539] START OF ROLLOUT (plan_slice_continuity)")
        print(f"{'─'*75}")
        print(f"  Parent Position:              {analysis['parent_qpos']}")
        print(f"  Plan First Frame:             {analysis['plan_slice_first_qpos']}")
        print(f"  First Frame Discontinuity:    {analysis['first_frame_diff']:.6f} {severity_level(analysis['first_frame_diff'])}")
        print(f"  Parent → Plan(1st):           {analysis['parent_to_plan_first']:.6f}")
        
        # ========== END OF ROLLOUT (3596) ==========
        print(f"\n[df_planning.py:3596] END OF ROLLOUT (final_state_continuity)")
        print(f"{'─'*75}")
        print(f"  Plan Last Frame:              {analysis['plan_slice_last_qpos']}")
        print(f"  Final Sim State:              {analysis['final_sim_state_qpos']}")
        print(f"  Last Frame Discontinuity:     {analysis['last_frame_diff']:.6f} {severity_level(analysis['last_frame_diff'])}")
        print(f"  Plan(last) → Sim State:       {analysis['plan_to_traj_last']:.6f}")
        
        # ========== TRAJECTORY INFORMATION ==========
        print(f"\n[TRAJECTORY CHARACTERISTICS]")
        print(f"{'─'*75}")
        print(f"  Trajectory Length:            {analysis['trajectory_length']} frames")
        print(f"  Plan Length:                  {analysis['plan_length']} frames")
        print(f"  Trajectory First Obs:         {analysis['trajectory_first_obs']}")
        print(f"  Trajectory Last Obs:          {analysis['trajectory_last_obs']}")
        
        if analysis['trajectory_length'] > 0:
            print(f"  Parent → Traj(1st):           {analysis['parent_to_traj_first']:.6f}")
        
        print(f"  Trajectory Total Distance:    {analysis['trajectory_total_distance']:.6f}")
        print(f"  Plan Total Distance:          {analysis['plan_total_distance']:.6f}")
        
        # ========== DIVERGENCE ANALYSIS ==========
        if analysis['divergence_per_frame']:
            print(f"\n[FRAME-BY-FRAME DIVERGENCE]")
            print(f"{'─'*75}")
            print(f"  Mean Divergence:              {analysis['divergence_mean']:.6f}")
            print(f"  Max Divergence:               {analysis['divergence_max']:.6f}")
            print(f"  Std Dev:                      {analysis['divergence_std']:.6f}")
            print(f"  Frames Analyzed:              {len(analysis['divergence_per_frame'])}")
    
    # ========== SUMMARY TABLE ==========
    print(f"\n\n{'='*150}")
    print("SUMMARY TABLE: All Loops at a Glance")
    print(f"{'='*150}\n")
    
    print(f"{'Loop':<8} {'Start Δ':<12} {'End Δ':<12} {'Traj Len':<10} {'Plan Len':<10} {'Div Mean':<12} {'Div Max':<12} {'Status':<20}")
    print(f"{'-'*150}")
    
    for analysis in analyses:
        loop = analysis["loop"]
        start_d = analysis["first_frame_diff"]
        end_d = analysis["last_frame_diff"]
        traj_len = analysis["trajectory_length"]
        plan_len = analysis["plan_length"]
        div_mean = analysis["divergence_mean"]
        div_max = analysis["divergence_max"]
        
        # Determine overall status
        if end_d > 30:
            status = "CRITICAL"
        elif end_d > 20:
            status = "ERROR"
        elif end_d > 5:
            status = "WARNING"
        elif traj_len == 1:
            status = "ANOMALY"
        else:
            status = "OK"
        
        print(f"Loop {loop:<3} {start_d:<12.4f} {end_d:<12.4f} {traj_len:<10} {plan_len:<10} {div_mean:<12.4f} {div_max:<12.4f} {status:<20}")
    
    # ========== KEY FINDINGS ==========
    print(f"\n{'='*150}")
    print("KEY FINDINGS")
    print(f"{'='*150}\n")
    
    # Find issues
    critical_loops = [a for a in analyses if a["last_frame_diff"] > 30]
    anomaly_loops = [a for a in analyses if a["trajectory_length"] == 1]
    
    if critical_loops:
        print(f"⚠️  CRITICAL DISCONTINUITIES (end_diff > 30):")
        for a in critical_loops:
            print(f"   - Loop {a['loop']}: end_diff = {a['last_frame_diff']:.4f}")
    
    if anomaly_loops:
        print(f"\n⚠️  TRAJECTORY ANOMALIES (length = 1):")
        for a in anomaly_loops:
            print(f"   - Loop {a['loop']}: trajectory_length = {a['trajectory_length']}")
            print(f"     → Stuck at position: {a['trajectory_last_obs']}")
    
    # Best and worst loops
    best_end = min(analyses, key=lambda a: a["last_frame_diff"])
    worst_end = max(analyses, key=lambda a: a["last_frame_diff"])
    
    print(f"\n📊 BEST END DISCONTINUITY: Loop {best_end['loop']} ({best_end['last_frame_diff']:.4f})")
    print(f"📊 WORST END DISCONTINUITY: Loop {worst_end['loop']} ({worst_end['last_frame_diff']:.4f})")
    
    # From/To analysis
    print(f"\n📍 FROM_START vs FROM_GOAL:")
    from_start = [a for a in analyses if a["loop"] in [1, 3]]
    from_goal = [a for a in analyses if a["loop"] in [2, 4]]
    
    print(f"   FROM_START (Loops 1, 3):")
    for a in from_start:
        print(f"   - Loop {a['loop']}: end_diff={a['last_frame_diff']:.4f}, traj_len={a['trajectory_length']}")
    
    print(f"   FROM_GOAL (Loops 2, 4):")
    for a in from_goal:
        print(f"   - Loop {a['loop']}: end_diff={a['last_frame_diff']:.4f}, traj_len={a['trajectory_length']}")
    
    # Depth analysis
    print(f"\n📊 DEPTH 1 vs DEPTH 2:")
    depth1 = [a for a in analyses if a["loop"] in [1, 2]]
    depth2 = [a for a in analyses if a["loop"] in [3, 4]]
    
    print(f"   DEPTH 1 (Loops 1, 2):")
    for a in depth1:
        print(f"   - Loop {a['loop']}: end_diff={a['last_frame_diff']:.4f}, traj_len={a['trajectory_length']}")
    
    print(f"   DEPTH 2 (Loops 3, 4):")
    for a in depth2:
        print(f"   - Loop {a['loop']}: end_diff={a['last_frame_diff']:.4f}, traj_len={a['trajectory_length']}")

if __name__ == "__main__":
    print_full_comparison()
