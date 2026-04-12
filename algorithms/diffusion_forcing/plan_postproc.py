"""plan_postproc.py
Plan post-processing utilities for MCTD trajectory extraction and deduplication.

Provides PlanPostprocMixin — a mixin class whose methods handle:
  - Depth-based prefix length calculation (_get_prefix_len_frames_from_depth)
  - Plan obs extraction at a segment boundary (_extract_obs_at_boundary)
  - Endpoint-based plan deduplication (_deduplicate_by_endpoint)
  - Greedy proximity reordering of combined FWD+BWD plans (_reorder_plan_by_proximity)
  - FWD-BWD gap computation (_compute_plan_gap)
  - Final output plan construction (_extract_output_plan)

Intended to be inherited by DiffusionForcingPlanning alongside DiffusionForcingBase.
All methods reference `self.*` attributes/methods provided by the base class or other mixins.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


class PlanPostprocMixin:
    """Mixin providing plan post-processing methods."""

    # ------------------------------------------------------------------
    # Depth / prefix helpers
    # ------------------------------------------------------------------

    def _get_prefix_len_frames_from_depth(self, depth: int, seg_size: int) -> int:
        return depth * seg_size * self.frame_stack

    def _get_denoised_frame_prefix_len(
        self, depth: int, seg_size: int, total_frames: Optional[int] = None
    ) -> int:
        prefix_len = max(self._get_prefix_len_frames_from_depth(depth, seg_size), 0)
        if total_frames is not None:
            prefix_len = min(prefix_len, total_frames)
        return prefix_len

    def _get_plan_viz_valid_frame_bounds(
        self,
        depth: int,
        seg_size: int,
        total_frames: int,
        is_forward_tree: bool,
    ) -> Tuple[int, int]:
        prefix_len = self._get_denoised_frame_prefix_len(depth, seg_size, total_frames)
        if is_forward_tree:
            return 0, prefix_len
        return total_frames - prefix_len, total_frames

    def _mask_plan_obs_outside_valid_frames(
        self,
        plan_obs: np.ndarray,
        valid_frame_bounds: List[Tuple[int, int]],
    ) -> np.ndarray:
        total_frames = plan_obs.shape[0]
        for i, (valid_start, valid_end) in enumerate(valid_frame_bounds):
            valid_start = max(0, min(int(valid_start), total_frames))
            valid_end = max(valid_start, min(int(valid_end), total_frames))
            if valid_start > 0:
                plan_obs[:valid_start, i, :] = np.nan
            if valid_end < total_frames:
                plan_obs[valid_end:, i, :] = np.nan
        return plan_obs

    def _extract_obs_at_boundary(
        self,
        plan: torch.Tensor,  # (T, N, c) — last denoising step already selected by caller
        depth: int,
        seg_size: int,
    ) -> np.ndarray:
        """Extract unnormalized observations at frame index `depth * seg_size * fs - 1`.

        Unified replacement for the former _extract_plan_endpoint_obs /
        _extract_unc_tail_obs pair.  The caller controls which boundary to look at by
        choosing `depth`, and which samples to include by choosing N (1 for a single
        candidate plan, G*K for the uncertainty batch).

        Args:
            plan: (T, N, c) tensor — the relevant denoising step.
                  For a single-candidate plan: pass plan_tensor.unsqueeze(1) so N=1.
                  For the uncertainty batch:   pass unc_plan_hists[-1] so N=G*K.
            depth: Boundary index.  Frame extracted = depth * seg_size * fs - 1.
                   • For endpoint deduplication (current boundary):  pass child_depth.
                   • For uncertainty sigma (next undenoised boundary): pass child_depth + 1.
            seg_size: plan_tokens // sequence_dividing_factor.

        Returns:
            (N, obs_dim) numpy array of unnormalized world-coordinate observations.
            For N=1 callers that need a 1-D result, index with [0].
        """
        idx = min(
            self._get_prefix_len_frames_from_depth(depth, seg_size) - 1,
            plan.shape[0] - 1,
        )
        frame = plan[idx]  # (N, c)
        unnorm = self._unnormalize_x(frame.unsqueeze(0))  # (1, N, c)
        return unnorm[0, :, self.obs_bundle_indices].cpu().numpy()  # (N, obs_dim)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_by_endpoint(
        self,
        expanded_node_candidates: list,  # list of candidate dicts, len B
        candidate_obses: list,  # list of np.ndarray (observation_dim,), len B
        is_feasible: list,  # list of bool, len B
    ) -> list:
        """Among feasible plans from the same parent, kill duplicates whose endpoints
        are within diverge_threshold. Keep the plan whose endpoint is closer to the
        target node (measured by HILP value).

        Args:
            expanded_node_candidates: Candidate dicts (each has 'parent_node', 'target_node').
            candidate_obses: Predicted endpoint obs per candidate.
            is_feasible: Feasibility flags per candidate.

        Returns:
            is_kept: list of bool, len B. True if the plan should create a child node.
        """
        B = len(expanded_node_candidates)
        is_kept = list(is_feasible)  # start with feasibility mask

        # Group candidate indices by parent node name
        parent_groups: dict = {}  # parent_name -> list of candidate indices
        for i in range(B):
            parent_name = expanded_node_candidates[i]["parent_node"].name
            if parent_name not in parent_groups:
                parent_groups[parent_name] = []
            parent_groups[parent_name].append(i)

        # Within each parent group, compute HILP values and pairwise deduplicate
        for parent_name, indices in parent_groups.items():
            # All siblings in the same group share the same target_node (set by _select_dynamic_goal)
            target_node = expanded_node_candidates[indices[0]].get("target_node")
            if target_node is not None and target_node.obs is not None:
                # Stack endpoints for this group → compute V(endpoint_i, target) in one batch
                group_obs = np.stack([candidate_obses[i] for i in indices])  # (G, n_obs)
                target_tiled = np.tile(
                    target_node.obs, (len(indices), 1)
                )  # (G, n_obs) — obs already indexed by obs_dim_indices
                hilp_vals = self._compute_hilp_values(
                    group_obs,
                    target_tiled,
                    use_no_grad=True,
                ).cpu().numpy()  # (G,)
                group_values = {idx: float(hilp_vals[local_i]) for local_i, idx in enumerate(indices)}
            else:
                # Fallback: assign equal value (no preference between duplicates)
                group_values = {idx: 0.0 for idx in indices}

            feasible_indices = [i for i in indices if is_kept[i]]
            for a_idx in range(len(feasible_indices)):
                i = feasible_indices[a_idx]
                if not is_kept[i]:
                    continue
                for b_idx in range(a_idx + 1, len(feasible_indices)):
                    j = feasible_indices[b_idx]
                    if not is_kept[j]:
                        continue
                    _obs_i = candidate_obses[i].reshape(1, -1).astype(np.float32)
                    _obs_j = candidate_obses[j].reshape(1, -1).astype(np.float32)
                    dist = float(self._compute_distance(_obs_i, _obs_j)[0])
                    if dist < self.diverge_threshold:
                        # Duplicate: kill the one with lower HILP value (farther from target)
                        if group_values[i] >= group_values[j]:
                            is_kept[j] = False
                        else:
                            is_kept[i] = False
                            break  # i is killed, no more comparisons for i

        return is_kept

    # ------------------------------------------------------------------
    # Proximity reordering
    # ------------------------------------------------------------------

    def _reorder_plan_by_proximity(self, plan_unnormalized: torch.Tensor) -> torch.Tensor:
        """
        Postprocess a combined FWD+BWD plan by greedily reordering frames in euclidean space.

        Starting from frame idx=0, at each step look ahead at all remaining frames (idx > current):
          - If any fall within `meeting_delta` distance: pick the one with the highest idx.
          - Otherwise: pick the nearest frame regardless of distance.

        This resolves spatial discontinuities at the FWD-BWD junction without interpolation.
        Intermediate frames that are skipped are dropped; the output plan may be shorter than input.

        Args:
            plan_unnormalized: Tensor shape (T*fs, 1, c), unnormalized (maze coordinate scale).

        Returns:
            Reordered plan tensor of shape (K, 1, c), K <= T*fs.
        """
        n_frames = plan_unnormalized.shape[0]
        if n_frames <= 1:
            return plan_unnormalized

        threshold = self.meeting_delta
        obs_frames = plan_unnormalized[:, 0, self.obs_bundle_indices].detach().cpu().numpy()  # (N, n_obs)

        result_indices = [0]
        current_idx = 0

        while current_idx < n_frames - 1:
            remaining_start = current_idx + 1
            M = n_frames - remaining_start
            _cur_obs_rep = np.broadcast_to(obs_frames[current_idx:current_idx + 1], (M, obs_frames.shape[1])).copy()
            dists = self._compute_distance(obs_frames[remaining_start:], _cur_obs_rep)

            within = np.where(dists <= threshold)[0]            # local indices, ascending
            if len(within) > 0:
                # Convert to original indices (ascending)
                orig_within = [remaining_start + int(w) for w in within]
                # Check if all within-threshold states are consecutive from current_idx+1
                all_consecutive = all(
                    orig_within[j] == current_idx + 1 + j
                    for j in range(len(orig_within))
                )
                if all_consecutive:
                    # i→a→b→c all differ by 1: pick the nearest among them
                    next_local = int(within[int(np.argmin(dists[within]))])
                else:
                    # Default: highest original idx among those within threshold
                    next_local = int(within[-1])
            else:
                # Nearest frame among all remaining
                next_local = int(np.argmin(dists))

            current_idx = remaining_start + next_local
            result_indices.append(current_idx)

        idx_tensor = torch.tensor(result_indices, dtype=torch.long, device=plan_unnormalized.device)
        return plan_unnormalized[idx_tensor]

    # ------------------------------------------------------------------
    # Gap computation
    # ------------------------------------------------------------------

    def _compute_plan_gap(
        self,
        best_node,
        plan_tokens: int,
        is_tree1: bool,
    ) -> Optional[float]:
        """
        Compute the minimum pairwise distance between the FWD plan segment (best_node)
        and the BWD plan segment (best_node.target_node), in unnormalized space.

        Returns None if target_node.plan_history is empty (opposite tree not yet expanded).
        Uses TD metric or Euclidean distance depending on self.use_TD_metric_as_dist.
        """
        seg_size: int = plan_tokens // self.sequence_dividing_factor

        if len(best_node.plan_history) == 0 or len(best_node.plan_history[-1]) == 0:
            return None
        if best_node.target_node is None or len(best_node.target_node.plan_history) == 0:
            return None

        plan_a_full: torch.Tensor = best_node.plan_history[-1][-1]
        a_len: int = self._get_prefix_len_frames_from_depth(best_node.depth, seg_size)
        t1_segments: torch.Tensor = plan_a_full[:a_len]

        plan_b_full: torch.Tensor = best_node.target_node.plan_history[-1][-1]
        b_len: int = self._get_prefix_len_frames_from_depth(best_node.target_node.depth, seg_size)
        t2_flipped: torch.Tensor = torch.flip(plan_b_full[:b_len], [0])

        if a_len == 0 or b_len == 0:
            return None

        # Prepare observation data
        _obs_std_np = np.array(self.observation_std)
        _obs_mean_np = np.array(self.observation_mean)
        _seg_a_obs = t1_segments[:, self.obs_bundle_indices].detach().cpu().numpy() * _obs_std_np + _obs_mean_np
        _seg_b_obs = t2_flipped[:, self.obs_bundle_indices].detach().cpu().numpy() * _obs_std_np + _obs_mean_np

        # Compute pairwise distances
        A, B = _seg_a_obs.shape[0], _seg_b_obs.shape[0]
        dists = self._compute_distance(
            np.repeat(_seg_a_obs, B, axis=0),
            np.tile(_seg_b_obs, (A, 1)),
        )
        return float(dists.min())

    # ------------------------------------------------------------------
    # Output plan construction
    # ------------------------------------------------------------------

    def _extract_output_plan(
        self,
        best_node,
        plan_tokens: int,
        is_tree1: bool,
        goal_normalized: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct the final output plan from the best selected leaf TreeNode.

        In bidirectional mode (best_node.target_node is not None):
            - Takes plan_A from best_node (forward tree leaf) sliced by depth.
            - Takes plan_B from best_node.target_node (backward tree leaf) sliced by depth, then flipped.
            - Returns the concatenated plan: plan_A + flip(plan_B).

        In unidirectional mode (best_node.target_node is None):
            - Returns plan_A only (forward tree leaf sliced by depth).

        Args:
            best_node: The selected best leaf TreeNode (from _select_best_leaf).
            plan_tokens: Total number of plan tokens for the tree (determines seg_size).

        Returns:
            output_plan: Tensor of shape (T_combined*fs, 1, c), where T = combined path length.
        """
        seg_size: int = plan_tokens // self.sequence_dividing_factor

        # --- Plan A: forward tree leaf ---
        assert len(best_node.plan_history) > 0, \
            f"best_node.plan_history must be non-empty for expanded nodes, but got {best_node.plan_history}"
        assert len(best_node.plan_history[-1]) > 0, \
            f"best_node.plan_history[-1] must be non-empty, but got {best_node.plan_history[-1]}"

        plan_a_full: torch.Tensor = best_node.plan_history[-1][-1]  # (T_total*fs, c)
        a_len: int = self._get_prefix_len_frames_from_depth(best_node.depth, seg_size)
        t1_segments: torch.Tensor = plan_a_full[:a_len]  # (A_len, c)

        # --- Bidirectional search: target_node handling ---
        assert best_node.target_node is not None, \
            "target_node must be set in bidirectional MCTS (opposite tree leaf must be available)"
        if len(best_node.target_node.plan_history) == 0:
            # --- Early iteration or missing opposite tree: use plan_A only ---
            combined = t1_segments
        else:
            # --- Bidirectional: flip plan_B and concat ---
            plan_b_full: torch.Tensor = best_node.target_node.plan_history[-1][-1]  # (T_total*fs, c)
            b_len: int = self._get_prefix_len_frames_from_depth(best_node.target_node.depth, seg_size)
            t2_flipped: torch.Tensor = torch.flip(
                plan_b_full[:b_len], [0]
            )  # (B_len, c)

            # Always combine FWD+BWD: called only when meeting condition is satisfied
            combined = torch.cat([t1_segments, t2_flipped], dim=0)  # (A_len+B_len, c)

        if not is_tree1:
            combined = torch.flip(combined, [0])  # (A_len+B_len, c)

        # Pad goal state at the end: seg_size * frame_stack frames filled with goal obs
        if goal_normalized is not None:
            c = combined.shape[-1]
            n_goal_pad = 1
            goal_frame = torch.zeros(n_goal_pad, c, dtype=combined.dtype, device=combined.device)
            goal_obs = goal_normalized[0]  # (n_obs,) — goal_normalized is already obs-only
            goal_frame[:, self.obs_bundle_indices] = goal_obs.unsqueeze(0).expand(n_goal_pad, -1)
            combined = torch.cat([combined, goal_frame], dim=0)  # (A_len+B_len+n_goal_pad, c)

        output = combined.unsqueeze(1)  # (T, 1, c)

        # Validate output shape before returning
        assert output.ndim == 3, f"output.ndim={output.ndim}, expected 3"
        assert output.shape[1] == 1, f"output.shape[1]={output.shape[1]}, expected 1"

        return output
