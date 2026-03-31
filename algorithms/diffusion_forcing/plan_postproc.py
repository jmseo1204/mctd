"""plan_postproc.py
Plan post-processing utilities for MCTD trajectory extraction and deduplication.

Provides PlanPostprocMixin — a mixin class whose methods handle:
  - Depth-based prefix length calculation (_get_prefix_len_frames_from_depth)
  - Plan endpoint extraction (_extract_plan_endpoint_obs_pos)
  - Endpoint-based plan deduplication (_deduplicate_by_endpoint)
  - Greedy proximity reordering of combined FWD+BWD plans (_reorder_plan_by_proximity)
  - FWD-BWD gap computation (_compute_plan_gap)
  - Final output plan construction (_extract_output_plan)

Intended to be inherited by DiffusionForcingPlanning alongside DiffusionForcingBase.
All methods reference `self.*` attributes/methods provided by the base class or other mixins.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


class PlanPostprocMixin:
    """Mixin providing plan post-processing methods."""

    # ------------------------------------------------------------------
    # Depth / prefix helpers
    # ------------------------------------------------------------------

    def _get_prefix_len_frames_from_depth(self, depth: int, seg_size: int) -> int:
        return depth * seg_size * self.frame_stack

    def _extract_plan_endpoint_obs_pos(
        self,
        plan_hists_last: torch.Tensor,  # (t*fs, c) — last denoising step for one candidate
        child_depth: int,
        seg_size: int,
    ) -> np.ndarray:
        """Extract the obs_pos at the child node's denoised boundary from a plan tensor.

        Args:
            plan_hists_last: Plan tensor for one candidate, shape (plan_tokens*fs, c).
            child_depth: Depth of the child node (parent.depth + 1).
            seg_size: plan_tokens // sequence_dividing_factor.

        Returns:
            obs_pos: np.ndarray of shape (observation_dim,).
        """
        plan_unnormalized = self._unnormalize_x(
            plan_hists_last.unsqueeze(1)
        )  # (t*fs, 1, c)
        new_denoised_end: int = self._get_prefix_len_frames_from_depth(child_depth, seg_size)
        return plan_unnormalized[new_denoised_end - 1, 0, :self.observation_dim].cpu().numpy()

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_by_endpoint(
        self,
        expanded_node_candidates: list,  # list of candidate dicts, len B
        candidate_obs_poses: list,  # list of np.ndarray (observation_dim,), len B
        is_feasible: list,  # list of bool, len B
    ) -> list:
        """Among feasible plans from the same parent, kill duplicates whose endpoints
        are within diverge_threshold. Keep the plan whose endpoint is closer to the
        target node (measured by HILP value).

        Args:
            expanded_node_candidates: Candidate dicts (each has 'parent_node', 'target_node').
            candidate_obs_poses: Predicted endpoint obs_pos per candidate.
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
            if target_node is not None and target_node.obs_pos is not None:
                # Stack endpoints for this group → compute V(endpoint_i, target) in one batch
                group_obs = np.stack([candidate_obs_poses[i] for i in indices])  # (G, obs_dim)
                target_tiled = np.tile(
                    target_node.obs_pos[: self.observation_dim], (len(indices), 1)
                )  # (G, obs_dim)
                hilp_vals = self._compute_hilp_values(
                    group_obs[:, : self.observation_dim],
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
                    if self.use_TD_metric_as_dist:
                        _obs_i = candidate_obs_poses[i].reshape(1, -1).astype(np.float32)
                        _obs_j = candidate_obs_poses[j].reshape(1, -1).astype(np.float32)
                        dist = float(self._compute_state_temporal_dist_np(_obs_i, _obs_j)[0])
                    else:
                        dist = np.linalg.norm(
                            candidate_obs_poses[i][:self.pos_dim] - candidate_obs_poses[j][:self.pos_dim]
                        )
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
        if self.use_TD_metric_as_dist:
            obs_frames = plan_unnormalized[:, 0, :self.observation_dim].detach().cpu().numpy()  # (N, obs_dim)
        else:
            # (N, pos_dim) — positions in maze coordinates
            positions = plan_unnormalized[:, 0, :self.pos_dim].detach().cpu().numpy()

        result_indices = [0]
        current_idx = 0

        while current_idx < n_frames - 1:
            remaining_start = current_idx + 1
            if self.use_TD_metric_as_dist:
                M = n_frames - remaining_start
                _cur_rep = np.broadcast_to(obs_frames[current_idx:current_idx + 1], (M, obs_frames.shape[1])).copy()
                dists = self._compute_state_temporal_dist_np(obs_frames[remaining_start:], _cur_rep)
            else:
                remaining_pos = positions[remaining_start:]  # (M, pos_dim)
                dists = np.linalg.norm(remaining_pos - positions[current_idx], axis=1)  # (M,)

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

        if self.use_TD_metric_as_dist:
            _std_np = self.data_std[:self.observation_dim].cpu().numpy() if isinstance(self.data_std, torch.Tensor) else np.array(self.data_std[:self.observation_dim])
            _mean_np = self.data_mean[:self.observation_dim].cpu().numpy() if isinstance(self.data_mean, torch.Tensor) else np.array(self.data_mean[:self.observation_dim])
            _seg_a_unnorm = t1_segments[:, :self.observation_dim].detach().cpu().numpy() * _std_np + _mean_np
            _seg_b_unnorm = t2_flipped[:, :self.observation_dim].detach().cpu().numpy() * _std_np + _mean_np
            A, B = _seg_a_unnorm.shape[0], _seg_b_unnorm.shape[0]
            _obs_ab = np.repeat(_seg_a_unnorm, B, axis=0)
            _goal_ab = np.tile(_seg_b_unnorm, (A, 1))
            return float(self._compute_state_temporal_dist_np(_obs_ab, _goal_ab).min())
        else:
            _std_np = self.data_std[:self.pos_dim].cpu().numpy() if isinstance(self.data_std, torch.Tensor) else np.array(self.data_std[:self.pos_dim])
            _mean_np = self.data_mean[:self.pos_dim].cpu().numpy() if isinstance(self.data_mean, torch.Tensor) else np.array(self.data_mean[:self.pos_dim])
            _seg_a_unnorm = t1_segments[:, :self.pos_dim].detach().cpu().numpy() * _std_np + _mean_np
            _seg_b_unnorm = t2_flipped[:, :self.pos_dim].detach().cpu().numpy() * _std_np + _mean_np
            _diffs = _seg_a_unnorm[:, None, :] - _seg_b_unnorm[None, :, :]  # (A, B, pos_dim)
            return float(np.linalg.norm(_diffs, axis=2).min())

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
            goal_obs = goal_normalized[0, : self.observation_dim]  # (obs_dim,)
            goal_frame[:, : self.observation_dim] = goal_obs.unsqueeze(0).expand(n_goal_pad, -1)
            combined = torch.cat([combined, goal_frame], dim=0)  # (A_len+B_len+n_goal_pad, c)

        output = combined.unsqueeze(1)  # (T, 1, c)

        # Validate output shape before returning
        assert output.ndim == 3, f"output.ndim={output.ndim}, expected 3"
        assert output.shape[1] == 1, f"output.shape[1]={output.shape[1]}, expected 1"

        return output
