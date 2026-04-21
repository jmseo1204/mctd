# Hamiltonian Path Evaluation: Sampling Strategy and Metric Rationale

## Context

The evaluation compares two planners on antmaze-giant:

- **Online Hamiltonian planner** (`route_mode=online`): re-solves the Hamiltonian path at runtime using the MCTD model.
- **Fixed temporal-distance baseline** (`route_mode=fixed_temporal`): pre-commits to the Hamiltonian ordering derived from HILP temporal distance before execution.

For each test case `{S, G, W1, W2, W3}`, the baseline uses temporal distance to rank waypoint visit orders and picks the best one. The online planner can correct this ordering during planning. The gap between the two is meaningful only when the temporal-distance ordering is *wrong* — i.e., when it diverges from the graph-optimal Hamiltonian path.

---

## Dataset Structure

From the full-combos analysis over 476,420 waypoint combinations:

| Statistic | Value |
|---|---|
| Total combos | 476,420 |
| Mismatch (temporal ≠ graph optimal order) | 74,878 (15.72 %) |
| Median `graph_second_best_gap_rel` across mismatch | 0.038 |

Of the 15.72 % mismatch cases, the ordering disagreement is driven primarily by **ordering instability** (small gap between 1st and 2nd best Hamiltonian orderings), not by temporal distance inaccuracy.

---

## Why Not Simple Random Sampling from All Mismatches

Random sampling from the 74,878 mismatch cases would produce a benchmark dominated by **near-tie cases** (Q1 of `graph_second_best_gap_rel`):

| `graph_second_best_gap_rel` quartile | Mismatch rate | avg `gap_abs` |
|---|---|---|
| Q1 (lowest, ≤ 25th pct) | 39.5 % | 1.33 |
| Q2 | 17.8 % | 3.60 |
| Q3 | 5.4 % | 7.70 |
| Q4 (highest, ≥ 75th pct) | 0.6 % | 16.79 |

The bottom quartile (Q1) of `graph_second_best_gap_rel` has a median `gap_abs = 0.11`. Even if the online planner perfectly identifies the optimal ordering, the *actual plan length difference* would be negligible. This would cause two problems:

1. **Success-rate metric**: near-tie cases are inherently noisy — any small perturbation flips which ordering is "correct", making success rate an unreliable discriminator.
2. **Plan length metric**: even a correct reordering yields near-zero improvement when `gap_abs ≈ 0`.

---

## Why Not Top-K by `gap_abs` Only

Selecting the top-K cases by largest `gap_abs` produces cases where the plan improvement is maximally visible. However, these are also the cases where the temporal distance model is most confidently wrong — and may not represent the distribution of errors encountered in real deployments. This introduces **selection bias** that overstates the benefit.

---

## Chosen Strategy: Stratified Sampling by `graph_second_best_gap_rel`

### Definition

`graph_second_best_gap_rel = (G_cost(2nd-best order) − G_cost(optimal order)) / G_cost(optimal order)`

This measures **how unambiguous the correct ordering is** under the ground-truth graph metric. It is computed entirely from the graph (ground truth), independently of the temporal distance model or the online planner — making it a **model-independent selection criterion**.

### Sampling procedure

For each task (5 tasks), mismatch groups are divided into bins by their `graph_second_best_gap_rel` value:

| Bin | Quantile range | Interpretation |
|---|---|---|
| **high** | top 25 % (≥ 75th pct) | Unambiguous ordering: correct order is clearly better |
| **mid** | 25–50 % (50th–75th pct) | Moderate ordering signal |
| **low** | 50–75 % (25th–50th pct) | Weak ordering signal, near-tie |

The bottom 25 % (`graph_second_best_gap_rel` < 25th pct) is excluded: these cases have median `gap_abs < 0.2`, where even a perfectly correct planner would show negligible plan length improvement.

**N groups per bin** (default: 7) are sampled uniformly at random using a fixed seed (42) for reproducibility. The same seed is stored in the task override YAML under `eval_sampling.seed`.

### Why this is defensible

1. **Selection criterion is model-independent.** `graph_second_best_gap_rel` is derived entirely from Dijkstra shortest paths on the sampled graph — not from any model output. There is no way for model quality to influence which cases are selected.

2. **Excludes trivially uninformative cases, not hard cases.** Excluding the bottom 25 % is equivalent to excluding cases where the optimal and second-best orderings are within X % of each other in plan cost. This is methodologically identical to standard navigation benchmarks that require a minimum distance between start and goal.

3. **Full difficulty spectrum is visible.** The three bins span the space from near-tie (low) to unambiguous (high). Reporting results broken down by bin allows readers to see that the online planner's advantage scales with ordering difficulty — a principled result that is *harder* to achieve by cherry-picking.

4. **Reproducible.** Fixed seed 42 is stored in the YAML. Any reviewer can re-run annotation and obtain identical group selection.

---

## Evaluation Metric: `postprocessed_plan_length_rel_error`

### Definition

```
rel_error = (plan_length − optimal_plan_length) / optimal_plan_length
```

where `optimal_plan_length` is the graph-optimal Hamiltonian path length through the selected waypoints.

- **0** = plan achieves the graph-optimal route length.
- **> 0** = plan is longer than optimal (positive error).

### Why not success rate

Success rate (did the planner pick the graph-optimal ordering?) is binary. For near-tie cases (small `graph_second_best_gap_rel`), the binary outcome is dominated by noise — two orderings may have nearly identical cost but one is arbitrarily labeled "correct". This inflates variance and makes significance testing unreliable.

`rel_error` is a continuous metric that:
- Naturally reflects the *magnitude* of ordering errors (large `gap_abs` → large rel_error if wrong ordering chosen)
- Produces near-zero values for near-tie cases rather than binary 0/1 noise
- Allows comparison across bins: high-bin cases should show larger rel_error reduction when the correct ordering is found

---

## Expected Outcome

If the online Hamiltonian planner correctly identifies the graph-optimal ordering:

| Bin | Expected `rel_error` reduction | Reasoning |
|---|---|---|
| high | Large | Large `gap_abs`; correct ordering substantially shortens the plan |
| mid | Moderate | Moderate `gap_abs` |
| low | Small | Small `gap_abs`; near-tie, marginal improvement even if correct |

A monotonically decreasing pattern of `rel_error` from high to low bins for the baseline, and smaller `rel_error` for the online planner — especially in the high bin — constitutes strong evidence for the online planner's effectiveness without overstating the result.
