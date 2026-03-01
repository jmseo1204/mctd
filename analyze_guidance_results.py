#!/usr/bin/env python3
"""
Guidance Scale Analysis Script
Parses experiment logs and generates a summary table of guidance scale performance.
"""
import sys
import collections
import os
from typing import Dict, List, Tuple

LOG_DIR = "experiment_logs"

def parse_log_file(log_path: str) -> Tuple[Dict[float, List[float]], Dict[float, List[float]], Dict[float, int], Dict[float, List[float]], Dict[float, Dict[str, List[float]]]]:
    avg_dist_by_scale = collections.defaultdict(list)
    final_dist_by_scale = collections.defaultdict(list)
    clip_warnings_by_scale = collections.defaultdict(int)
    grad_norms_by_scale = collections.defaultdict(list)
    ratios_by_scale = collections.defaultdict(lambda: collections.defaultdict(list))
    
    current_avg = None
    current_final = None
    current_scale = None
    
    if not os.path.exists(log_path):
        return {}, {}, {}, {}, {}

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                # Track clipping warnings
                if '[CLIP WARNING]' in line:
                    if current_scale is not None:
                        clip_warnings_by_scale[current_scale] += 1
                
                # Track gradient norms
                if '[GUIDANCE STATS] Grad Norm:' in line:
                    grad_norm = float(line.split('Grad Norm:')[1].strip())
                    if current_scale is not None:
                        grad_norms_by_scale[current_scale].append(grad_norm)
                
                if 'Dist per batch:' in line:
                    content = line.split('batch:')[1].strip()
                    content = content.replace('[', '').replace(']', '')
                    content = content.replace('tensor(', '').replace(')', '')
                    content = content.replace(', device=\'cuda:0\'', '').replace(', device=\'cuda:1\'', '')
                    current_avg = [float(x.strip()) for x in content.split(',') if x.strip()]
                    
                elif 'Final token dist:' in line:
                    content = line.split('dist:')[1].strip()
                    content = content.replace('[', '').replace(']', '')
                    content = content.replace('tensor(', '').replace(')', '')
                    content = content.replace(', device=\'cuda:0\'', '').replace(', device=\'cuda:1\'', '')
                    current_final = [float(x.strip()) for x in content.split(',') if x.strip()]
                    
                elif 'Scales:' in line:
                    content = line.split('Scales:')[1].strip()
                    content = content.replace('[', '').replace(']', '')
                    content = content.replace('tensor(', '').replace(')', '')
                    content = content.replace(', device=\'cuda:0\'', '').replace(', device=\'cuda:1\'', '')
                    scales = [float(x.strip()) for x in content.split(',') if x.strip()]
                    
                    if current_avg and current_final and len(scales) == len(current_avg) == len(current_final):
                        for s, a, f in zip(scales, current_avg, current_final):
                            avg_dist_by_scale[s].append(a)
                            final_dist_by_scale[s].append(f)
                            current_scale = s  # Track current scale
                
                # Track guidance ratios
                if '[GUIDANCE RATIO]' in line:
                    if current_scale is not None:
                        # Format: [GUIDANCE RATIO] anchor: 10.00% | goal: 20.00% | rdf: 70.00%
                        parts = line.split('RATIO]')[1].strip().split('|')
                        for p in parts:
                            name, val = p.strip().split(':')
                            val_float = float(val.strip().replace('%', ''))
                            ratios_by_scale[current_scale][name.strip()].append(val_float)
            except Exception:
                continue
    
    return dict(avg_dist_by_scale), dict(final_dist_by_scale), dict(clip_warnings_by_scale), dict(grad_norms_by_scale), {k: dict(v) for k, v in ratios_by_scale.items()}

def generate_summary_table(avg_dist_by_scale: Dict[float, List[float]],
                          final_dist_by_scale: Dict[float, List[float]], 
                          clip_warnings: Dict[float, int],
                          grad_norms: Dict[float, List[float]],
                          ratios: Dict[float, Dict[str, List[float]]],
                          log_name: str) -> None:
    if not final_dist_by_scale:
        print(f"\nNo valid data found in {log_name}.")
        return
    
    baseline_f = None
    if 0.0 in final_dist_by_scale:
        baseline_f = sum(final_dist_by_scale[0.0]) / len(final_dist_by_scale[0.0])
    
    print("\n" + "="*160)
    print(f"Analysis for: {log_name}")
    print("="*160)
    print(f"{'Scale':<10} {'Avg Batch Dist':<18} {'Avg Final Dist':<18} {'Samples':<10} {'Improvement':<15} {'Clip':<8} {'Grad Norm':<12} {'Ratio (Anc/Goal/RDF)':<25}")
    print("-" * 160)
    
    for scale in sorted(final_dist_by_scale.keys()):
        # Average Batch Distance
        a_vals = avg_dist_by_scale.get(scale, [])
        a_avg = sum(a_vals) / len(a_vals) if a_vals else 0.0
        
        # Average Final Token Distance
        f_vals = final_dist_by_scale[scale]
        f_avg = sum(f_vals) / len(f_vals)
        
        n_samples = len(f_vals)
        
        improvement = "Baseline" if scale == 0.0 else (f"{((f_avg - baseline_f) / baseline_f) * 100:+.2f}%" if baseline_f else "-")
        
        clip_count = clip_warnings.get(scale, 0)
        
        grad_norm_vals = grad_norms.get(scale, [])
        avg_grad_norm = sum(grad_norm_vals) / len(grad_norm_vals) if grad_norm_vals else 0.0
        
        # Guidance Ratios
        scale_ratios = ratios.get(scale, {})
        def get_avg_ratio(name):
            vals = scale_ratios.get(name, [])
            return sum(vals) / len(vals) if vals else 0.0
        
        r_anc = get_avg_ratio("anchor")
        r_goal = get_avg_ratio("goal")
        r_rdf = get_avg_ratio("rdf")
        ratio_str = f"{r_anc:4.1f}% / {r_goal:4.1f}% / {r_rdf:4.1f}%"
        
        print(f"{scale:<10.1f} {a_avg:<18.6f} {f_avg:<18.6f} {n_samples:<10} {improvement:<15} {clip_count:<8} {avg_grad_norm:<12.4f} {ratio_str:<25}")
    
    print("="*160)
    print(f"Total samples: {sum(len(v) for v in final_dist_by_scale.values())}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_guidance_results.py <log_file_path>")
        return

    log_path = sys.argv[1]
    log_name = os.path.basename(log_path)
    
    if not os.path.exists(log_path):
        print(f"Error: File '{log_path}' not found.")
        return

    avg_dist, final_dist, clip_warnings, grad_norms, ratios = parse_log_file(log_path)
    generate_summary_table(avg_dist, final_dist, clip_warnings, grad_norms, ratios, log_name)

if __name__ == "__main__":
    main()
