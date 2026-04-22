"""plan_postproc.py
Plan post-processing utilities for MCTD trajectory extraction and deduplication.

Provides PlanPostprocMixin — a mixin class whose methods handle:
  - Depth-based prefix length calculation (_get_prefix_len_frames_from_depth)
  - Plan obs extraction at a segment boundary (_extract_obs_at_boundary)
  - Plan obs slicing for new segments / meeting-gap checks
  - Endpoint-based plan deduplication (_deduplicate_by_endpoint)
  - Greedy proximity reordering of assembled meeting/connection plans (_reorder_plan_by_proximity)
  - Meeting-gap computation (_compute_plan_gap)
  - Final output plan construction (_extract_output_plan)

Intended to be inherited by DiffusionForcingPlanning alongside DiffusionForcingBase.
All methods reference `self.*` attributes/methods provided by the base class or other mixins.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class PlanPostprocMixin:
    """Mixin providing plan post-processing methods."""

    # ------------------------------------------------------------------
    # Depth / prefix helpers
    # ------------------------------------------------------------------

    def _get_prefix_len_frames_from_depth(self, depth: int, seg_size: int) -> int:
        return depth * seg_size * self.frame_stack

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
            parent_key = expanded_node_candidates[i].get("parent_key")
            if parent_key is None:
                parent_name = expanded_node_candidates[i]["parent_node"].name
                selected_tree = expanded_node_candidates[i].get("selected_tree")
                if selected_tree is not None:
                    parent_key = f"{selected_tree.tag}:{parent_name}"
                else:
                    parent_key = parent_name
            if parent_key not in parent_groups:
                parent_groups[parent_key] = []
            parent_groups[parent_key].append(i)

        # Within each parent group, compute HILP values and pairwise deduplicate
        for parent_name, indices in parent_groups.items():
            # All siblings in the same group share the same target_node (set by _select_dynamic_goal)
            target_node = expanded_node_candidates[indices[0]].get("target_node")
            group_source_flags = [
                self._require_source_is_left(
                    expanded_node_candidates[idx].get("source_is_left"),
                    context=f"endpoint_dedup:{parent_name}:{expanded_node_candidates[idx]['name']}",
                )
                for idx in indices
            ]
            source_is_left = bool(group_source_flags[0])
            if any(flag != source_is_left for flag in group_source_flags[1:]):
                self._log_directional_debug(
                    "endpoint_dedup.direction_mismatch",
                    parent=parent_name,
                    flags=group_source_flags,
                )
                raise ValueError(
                    f"Sibling candidates under '{parent_name}' disagree on source_is_left: {group_source_flags}"
                )
            if target_node is not None and target_node.obs is not None:
                # Stack endpoints for this group → compute V(endpoint_i, target) in one batch
                group_obs = np.stack([candidate_obses[i] for i in indices])  # (G, n_obs)
                target_tiled = np.tile(
                    target_node.obs, (len(indices), 1)
                )  # (G, n_obs) — obs already indexed by obs_dim_indices
                hilp_vals = self._compute_hilp_values(
                    group_obs if source_is_left else target_tiled,
                    target_tiled if source_is_left else group_obs,
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
                    dist = float(
                        self._compute_sequence_distance(
                            _obs_i,
                            _obs_j,
                            source_is_left=source_is_left,
                            sequence_mode="route_execution",
                            context=f"endpoint_dedup:{parent_name}:{i}->{j}",
                        )[0]
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
        Postprocess a combined multi-segment plan by greedily reordering frames in euclidean space.

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
            dists = self._compute_distance(
                _cur_obs_rep,
                obs_frames[remaining_start:],
                mode="execution_sequence",
            )

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

    def _extract_plan_obs_frames(self, plan_frames: torch.Tensor) -> np.ndarray:
        """Convert normalized bundle frames into unnormalized observation vectors."""
        _obs_std_np = np.array(self.observation_std)
        _obs_mean_np = np.array(self.observation_mean)
        plan_obs = (
            plan_frames[:, self.obs_bundle_indices].detach().cpu().numpy()
            * _obs_std_np
            + _obs_mean_np
        )
        return np.asarray(plan_obs, dtype=np.float32)

    def _extract_node_obs_slice(
        self,
        node,
        plan_tokens: int,
        start_depth: int,
        end_depth: int,
        reverse: bool = False,
    ) -> Optional[np.ndarray]:
        """Extract an unnormalized observation slice from a node's materialized plan."""
        if node is None:
            return None
        if len(node.plan_history) == 0 or len(node.plan_history[-1]) == 0:
            return None

        seg_size: int = plan_tokens // self.sequence_dividing_factor
        start_len = self._get_prefix_len_frames_from_depth(int(start_depth), seg_size)
        end_len = self._get_prefix_len_frames_from_depth(int(end_depth), seg_size)
        if end_len <= start_len:
            return None

        plan_full: torch.Tensor = node.plan_history[-1][-1]
        total_frames = int(plan_full.shape[0])
        start_len = min(max(start_len, 0), total_frames)
        end_len = min(max(end_len, start_len), total_frames)
        if end_len <= start_len:
            return None

        plan_slice = plan_full[start_len:end_len]
        if plan_slice.shape[0] == 0:
            return None
        if reverse:
            plan_slice = torch.flip(plan_slice, [0])
        return self._extract_plan_obs_frames(plan_slice)

    def _node_has_materialized_plan(self, node) -> bool:
        """Return whether `node` has a materialized plan prefix we can slice."""
        if node is None:
            return False
        plan_history = getattr(node, "plan_history", None)
        if plan_history is None or len(plan_history) == 0:
            return False
        latest_hist = plan_history[-1]
        return latest_hist is not None and len(latest_hist) > 0

    def _get_node_endpoint_mode(self, node) -> Optional[str]:
        """Classify a node endpoint as a materialized segment or an anchor point."""
        if node is None or getattr(node, "obs", None) is None:
            return None
        if self._node_has_materialized_plan(node) and int(getattr(node, "depth", 0)) > 0:
            return "segment"
        return "anchor"

    def _describe_connection_kind(self, source_node, target_node) -> Optional[str]:
        source_mode = self._get_node_endpoint_mode(source_node)
        target_mode = self._get_node_endpoint_mode(target_node)
        if source_mode is None or target_mode is None:
            return None
        return f"{source_mode}_to_{target_mode}"

    def _extract_node_endpoint_obs(
        self,
        node,
        plan_tokens: int,
        *,
        source_is_left: bool,
        role: str,
    ) -> Optional[np.ndarray]:
        """Return the execution-ordered endpoint observations for gap checks.

        Materialized nodes contribute their route-ordered prefix segment. Root-only
        or otherwise unmaterialized nodes fall back to a singleton anchor point.
        """
        endpoint_mode = self._get_node_endpoint_mode(node)
        if endpoint_mode is None:
            return None
        if endpoint_mode == "segment":
            return self._extract_route_ordered_segment_obs(
                node,
                plan_tokens=plan_tokens,
                start_depth=0,
                end_depth=int(node.depth),
                source_is_left=source_is_left,
                role=role,
            )
        return np.asarray(node.obs, dtype=np.float32).reshape(1, -1)

    def _build_normalized_obs_frame(self, obs_normalized: torch.Tensor) -> torch.Tensor:
        """Build a single observation-only normalized frame.

        plan_history stores frames in per-frame (unstacked) format: (T*fs, unstacked_dim).
        We must match that format here, not the stacked x_stacked_shape[0] dimension.
        """
        c = int(self.unstacked_dim)
        obs_frame = torch.zeros(
            1, c, dtype=obs_normalized.dtype, device=obs_normalized.device
        )
        obs_frame[:, self.obs_bundle_indices] = obs_normalized.unsqueeze(0)
        return obs_frame

    def _extract_connection_endpoint_frames(
        self,
        node,
        plan_tokens: int,
        *,
        role: str,
    ) -> Optional[torch.Tensor]:
        """Return normalized connection frames for a source/target endpoint.

        Source endpoints always contribute at least one frame. Target anchor points
        are omitted here because callers append the target root observation
        explicitly after assembling the connection.
        """
        endpoint_mode = self._get_node_endpoint_mode(node)
        if endpoint_mode is None:
            return None
        if endpoint_mode == "anchor":
            if role == "target":
                return None
            return self._build_normalized_obs_frame(
                self._normalize_obs_np(np.asarray(node.obs, dtype=np.float32))
            )

        seg_size: int = plan_tokens // self.sequence_dividing_factor
        plan_full: torch.Tensor = node.plan_history[-1][-1]
        prefix_len: int = self._get_prefix_len_frames_from_depth(node.depth, seg_size)
        prefix = plan_full[:prefix_len]
        if role == "target":
            return torch.flip(prefix, [0])
        return prefix

    def _extract_route_ordered_segment_obs(
        self,
        node,
        plan_tokens: int,
        start_depth: int,
        end_depth: int,
        *,
        source_is_left: bool,
        role: str,
    ) -> Optional[np.ndarray]:
        """Return a node segment in global route-execution order without changing storage."""
        reverse = not bool(source_is_left)
        if reverse:
            self._log_directional_debug(
                "segment.reverse_execution_view",
                role=role,
                node="?" if node is None else getattr(node, "name", "?"),
                start_depth=int(start_depth),
                end_depth=int(end_depth),
            )
        return self._extract_node_obs_slice(
            node,
            plan_tokens=plan_tokens,
            start_depth=start_depth,
            end_depth=end_depth,
            reverse=reverse,
        )

    def _extract_new_segment_obs(
        self,
        node,
        plan_tokens: int,
    ) -> Optional[np.ndarray]:
        """Extract the newly created segment observations for `node`.

        The returned segment contains only the frames added in the round that
        created `node`, expressed in unnormalized observation space and kept in
        stored local-canonical order. Order-sensitive callers must explicitly
        request a route-ordered view via `_extract_route_ordered_segment_obs()`.
        """
        if node is None or node.depth <= 0:
            return None

        parent_depth = max(int(node.depth) - 1, 0)
        # Multi-tree mode treats every anchor-rooted tree uniformly: the newly
        # expanded local segment is always the prefix slice [parent_depth:node.depth).
        return self._extract_node_obs_slice(
            node,
            plan_tokens=plan_tokens,
            start_depth=parent_depth,
            end_depth=int(node.depth),
        )

    def _compute_plan_gap_to_target(
        self,
        best_node,
        target_node,
        plan_tokens: int,
    ) -> Optional[float]:
        """
        Compute the minimum pairwise distance between two route-ordered endpoints.

        Materialized nodes use their prefix segments; root-only endpoints fall back
        to singleton anchor observations. Returns None only when either node lacks
        an observable endpoint.
        """
        if best_node is None or target_node is None:
            return None

        seg_a_obs = self._extract_node_endpoint_obs(
            best_node,
            plan_tokens=plan_tokens,
            source_is_left=True,
            role="meeting_left",
        )
        seg_b_obs = self._extract_node_endpoint_obs(
            target_node,
            plan_tokens=plan_tokens,
            source_is_left=False,
            role="meeting_right",
        )
        if seg_a_obs is None or seg_b_obs is None:
            return None

        A, B = seg_a_obs.shape[0], seg_b_obs.shape[0]
        dists = self._compute_distance(
            np.repeat(seg_a_obs, B, axis=0),
            np.tile(seg_b_obs, (A, 1)),
            mode="execution_sequence",
        )
        return float(dists.min())

    def _compute_plan_gap(
        self,
        best_node,
        plan_tokens: int,
    ) -> Optional[float]:
        """
        Compute the minimum pairwise distance between `best_node`'s prefix segment
        and the reversed prefix of its paired target node.

        Returns None if target_node.plan_history is empty (opposite tree not yet expanded).
        Uses TD metric or Euclidean distance depending on self.use_TD_metric_as_dist.
        """
        return self._compute_plan_gap_to_target(
            best_node=best_node,
            target_node=best_node.target_node,
            plan_tokens=plan_tokens,
        )

    # ------------------------------------------------------------------
    # Output plan construction
    # ------------------------------------------------------------------

    def _assemble_connection_frames(
        self,
        source_node,
        target_node,
        plan_tokens: int,
    ) -> torch.Tensor:
        """Build the canonical source-endpoint + reversed target-segment frame stack."""
        source_prefix = self._extract_connection_endpoint_frames(
            source_node,
            plan_tokens=plan_tokens,
            role="source",
        )
        if source_prefix is None:
            raise ValueError(
                "source_node must provide either a materialized prefix or an anchor observation"
            )

        target_prefix_reversed = self._extract_connection_endpoint_frames(
            target_node,
            plan_tokens=plan_tokens,
            role="target",
        )
        if target_prefix_reversed is None:
            return source_prefix
        if (
            source_prefix.dtype != target_prefix_reversed.dtype
            or source_prefix.device != target_prefix_reversed.device
        ):
            source_prefix = source_prefix.to(
                dtype=target_prefix_reversed.dtype,
                device=target_prefix_reversed.device,
            )

        return torch.cat([source_prefix, target_prefix_reversed], dim=0)

    def _append_normalized_obs_frame(
        self,
        plan_frames: torch.Tensor,
        obs_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Append a single observation-only frame onto a normalized plan tensor."""
        obs_frame = self._build_normalized_obs_frame(
            obs_normalized.to(dtype=plan_frames.dtype, device=plan_frames.device)
        )
        return torch.cat([plan_frames, obs_frame], dim=0)

    def _build_postprocessed_plan_bundle_from_output_plan(
        self,
        output_plan: torch.Tensor,
    ) -> dict:
        """Reuse the same output-plan -> unnormalize -> reorder materialization flow."""
        plan_unnormalized = self._unnormalize_x(output_plan.unsqueeze(0))[-1]
        postprocessed_plan = self._reorder_plan_by_proximity(plan_unnormalized)
        return {
            "output_plan": output_plan,
            "plan_unnormalized": plan_unnormalized,
            "postprocessed_plan": postprocessed_plan,
            "postprocessed_len": int(postprocessed_plan.shape[0]),
        }

    def _extract_output_plan(
        self,
        best_node,
        plan_tokens: int,
        reverse_output: bool,
        goal_normalized: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct the final assembled plan from the selected leaf and its paired node.

        The canonical assembly is:
            source-root prefix + reverse(target-root prefix)

        `reverse_output=True` is used when a selected tree stores its prefix in the
        opposite execution orientation and the assembled result must be flipped.

        Args:
            best_node: The selected best leaf TreeNode (from _select_best_leaf).
            plan_tokens: Total number of plan tokens for the tree (determines seg_size).
            reverse_output: Whether to reverse the assembled path before returning it.

        Returns:
            output_plan: Tensor of shape (T_combined*fs, 1, c), where T = combined path length.
        """
        # --- Primary prefix segment from the selected node ---
        assert len(best_node.plan_history) > 0, \
            f"best_node.plan_history must be non-empty for expanded nodes, but got {best_node.plan_history}"
        assert len(best_node.plan_history[-1]) > 0, \
            f"best_node.plan_history[-1] must be non-empty, but got {best_node.plan_history[-1]}"

        # --- Paired node prefix, reversed for connection ---
        assert best_node.target_node is not None, \
            "target_node must be set when assembling a paired-tree connection"
        combined = self._assemble_connection_frames(
            source_node=best_node,
            target_node=best_node.target_node,
            plan_tokens=plan_tokens,
        )

        if reverse_output:
            combined = torch.flip(combined, [0])

        # Pad goal state at the end: seg_size * frame_stack frames filled with goal obs
        if goal_normalized is not None:
            goal_obs = goal_normalized[0]  # (n_obs,) — goal_normalized is already obs-only
            combined = self._append_normalized_obs_frame(combined, goal_obs)

        output = combined.unsqueeze(1)  # (T, 1, c)

        # Validate output shape before returning
        assert output.ndim == 3, f"output.ndim={output.ndim}, expected 3"
        assert output.shape[1] == 1, f"output.shape[1]={output.shape[1]}, expected 1"

        return output

    def _extract_connection_plan_from_node_pair(
        self,
        source_node,
        target_node,
        plan_tokens: int,
        append_target_obs_normalized: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build a canonical source-root -> target-root connection plan."""
        combined = self._assemble_connection_frames(
            source_node=source_node,
            target_node=target_node,
            plan_tokens=plan_tokens,
        )

        if append_target_obs_normalized is not None:
            combined = self._append_normalized_obs_frame(
                combined,
                append_target_obs_normalized,
            )

        return combined.unsqueeze(1)

    def _build_connection_plan_bundle(
        self,
        source_node,
        target_node,
        plan_tokens: int,
        append_target_obs_normalized: Optional[torch.Tensor] = None,
    ) -> dict:
        """Build an edge-local execution bundle for a source/target node pair."""
        output_plan = self._extract_connection_plan_from_node_pair(
            source_node,
            target_node,
            plan_tokens=plan_tokens,
            append_target_obs_normalized=append_target_obs_normalized,
        )
        return self._build_postprocessed_plan_bundle_from_output_plan(output_plan)

    def _build_postprocessed_plan_from_node(
        self,
        best_node,
        plan_tokens: int,
        reverse_output: bool,
        goal_normalized: Optional[torch.Tensor] = None,
    ) -> dict:
        """Materialize the final output plan and its reordered execution variant.

        Returns a small bundle so callers can reuse the postprocessed plan without
        recomputing output extraction + unnormalization + greedy reordering.
        """
        output_plan = self._extract_output_plan(
            best_node,
            plan_tokens=plan_tokens,
            reverse_output=reverse_output,
            goal_normalized=goal_normalized,
        )  # (T_combined*fs+goal_pad, 1, c)
        return self._build_postprocessed_plan_bundle_from_output_plan(output_plan)
