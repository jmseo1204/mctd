"""env_executor.py
Plan execution and simulation state management utilities.

Provides PlanExecutorMixin — a mixin class whose methods handle:
  - Physical simulation state get/set (_get_sim_state, _set_sim_state)
  - Action computation from sim states (_compute_action_from_plan)
  - Per-step environment rollout (_execute_plan_in_env)
  - Leaf-node plan rollout for MCTS expansion (_rollout_leaf_plan)

Intended to be inherited by DiffusionForcingPlanning alongside DiffusionForcingBase.
All methods reference `self.*` attributes/methods that are expected to be provided by
the base class or by other mixins in the MRO.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

import numpy as np
import torch


class PlanExecutorMixin:
    """Mixin providing plan-execution and sim-state management methods."""

    # ------------------------------------------------------------------
    # Simulation state helpers
    # ------------------------------------------------------------------

    def _get_sim_state(self, envs: Any) -> Optional[dict]:
        """Extract current qpos/qvel from envs (DummyVecEnv)."""
        try:
            data = envs.get_attr("data")
            if data and len(data) > 0:
                return {"qpos": data[0].qpos.copy(), "qvel": data[0].qvel.copy()}
        except Exception:
            pass
        return None

    def _set_sim_state(self, envs: Any, sim_state: Optional[dict]) -> None:
        """Restore qpos/qvel to envs (DummyVecEnv)."""
        if sim_state is None:
            return
        try:
            envs.env_method("set_state", sim_state["qpos"], sim_state["qvel"])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Action computation
    # ------------------------------------------------------------------

    def _compute_action_from_plan(
        self,
        agent: Optional[Any] = None,
        sub_goal_sim_state: Optional[dict] = None,
        current_sim_state: Optional[dict] = None,
        prev_sim_state: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Compute action for a single timestep from sim_states.

        Args:
            agent: RL agent for antmaze (None for pointmaze)
            sub_goal_sim_state: sim_state of the sub-goal (qpos[:pos_dim] used as target position)
            current_sim_state: sim_state of the current agent state (qpos + qvel)
            prev_sim_state: sim_state from previous step (qpos[:pos_dim] used for plan velocity in PID)

        Returns:
            action: (1, action_dim) - clipped to [-1, 1]
        """
        if "antmaze" in self.env_id:
            assert agent is not None, "agent must be provided for antmaze"
            assert current_sim_state is not None, "current_sim_state must be provided for antmaze"
            assert sub_goal_sim_state is not None, "sub_goal_sim_state must be provided for antmaze"

            state_29d = np.concatenate([current_sim_state["qpos"], current_sim_state["qvel"]], axis=0)  # (29,)
            state_input = state_29d[np.newaxis, :]  # (1, 29)
            sub_goal_pos = sub_goal_sim_state["qpos"][:2]  # physical: qpos[:2] = world (x,y)

            action = agent.sample_action(state_input, sub_goal_pos)
            return torch.from_numpy(action).float().reshape(1, -1)
        else:
            # PointMaze: PID-like controller
            assert current_sim_state is not None, "current_sim_state must be provided for pointmaze"
            assert sub_goal_sim_state is not None, "sub_goal_sim_state must be provided for pointmaze"

            current_pos = torch.tensor(current_sim_state["qpos"][:2], dtype=torch.float32).unsqueeze(0)  # (1, 2): physical world (x,y)
            current_vel = torch.tensor(current_sim_state["qvel"][:2], dtype=torch.float32).unsqueeze(0)  # (1, 2): physical world (x,y)
            target_pos = torch.tensor(sub_goal_sim_state["qpos"][:2], dtype=torch.float32).unsqueeze(0)  # (1, 2): physical world (x,y)

            if prev_sim_state is None:
                plan_vel = target_pos - current_pos
            else:
                prev_pos = torch.tensor(prev_sim_state["qpos"][:2], dtype=torch.float32).unsqueeze(0)  # (1, 2): physical world (x,y)
                plan_vel = target_pos - prev_pos

            action = 12.5 * (target_pos - current_pos) + 1.2 * (plan_vel - current_vel)
            return torch.clip(action, -1, 1)

    # ------------------------------------------------------------------
    # Plan execution loop
    # ------------------------------------------------------------------

    def _execute_plan_in_env(
        self,
        plan_frame_format: torch.Tensor,  # (T*fs, 1, c) - frame format
        envs: Any,
        agent: Optional[Any] = None,
        use_diffused_action: bool = False,
        parent_sim_state: Optional[dict] = None,
        is_backward: bool = False,
    ) -> tuple:
        """
        Execute a plan segment in environment with unified action computation.

        Handles both antmaze and pointmaze with:
        - Sub-goal interval updates for antmaze
        - PID controller for pointmaze
        - Trajectory and reward collection
        - Optional parent state injection for state stitching

        Args:
            plan_frame_format: Plan in frame format, shape (T*fs, 1, c)
            envs: Vectorized environment
            agent: RL agent for antmaze (None for pointmaze)
            use_diffused_action: If True, use action directly from diffusion
            parent_sim_state: Optional sim state (qpos, qvel) to restore before execution

        Returns:
            (trajectory_list, reward_dict, rollout_viz)
            - trajectory_list: List of trajectory bundles
            - reward_dict: Dict with keys 'reached', 'episode_reward', etc.
            - rollout_viz: Dict with per-step agent/sub-goal positions for visualization
        """
        trajectory = []

        self._set_sim_state(envs, parent_sim_state)

        # Get the full observation from environment (qpos + qvel concatenation for antmaze)
        # For DQL agent compatibility, we need the full obs [qpos, qvel], not just qpos[:pos_dim]
        current_sim_state = self._get_sim_state(envs)
        qpos = current_sim_state["qpos"]  # shape: (15,)
        qvel = current_sim_state["qvel"]  # shape: (14,)
        # Select obs_dim_indices from [qpos, qvel] concatenation (generalizes contiguous and non-contiguous cases)
        obs_flat = np.concatenate([qpos, qvel], axis=0)[self.obs_dim_indices]
        obs_numpy = obs_flat[np.newaxis, :]



        batch_size = plan_frame_format.shape[1]
        reached = np.zeros(batch_size, dtype=bool)
        episode_reward = np.zeros(batch_size)
        episode_reward_if_stay = np.zeros(batch_size)
        first_reach = np.zeros(batch_size)

        # Initialize sub_goal for antmaze
        plan_slice_np = None
        sub_goal_pos = None  # 2D position
        sub_goal_sim_state = None  # Full 29D sim_state
        sub_goal_step = None
        if "antmaze" in self.env_id:
            plan_slice_np = plan_frame_format[:, 0, :].detach().cpu().numpy()  # (T*fs, c)
            # Find the plan frame nearest to the agent's current position, then start
            # sub_goal_interval ahead of it.  This handles the case where plan_frame_format
            # is an accumulated FWD+BWD plan that starts from the episode origin (52, 28),
            # while the agent has already advanced deep into the maze.
            if self.use_TD_metric_as_dist:
                _n_frames = plan_slice_np.shape[0]
                _cur_obs = obs_flat.reshape(1, -1)  # (1, obs_dim)
                _cur_rep = np.broadcast_to(_cur_obs, (_n_frames, _cur_obs.shape[1])).copy()
                _plan_obs = plan_slice_np[:, :self.observation_dim]  # (T*fs, obs_dim)
                dists_to_plan = self._compute_state_temporal_dist_np(_plan_obs, _cur_rep)
            else:
                current_pos = current_sim_state["qpos"][:2]  # physical (x,y) always at qpos[:2]
                plan_positions = plan_slice_np[:, self.pos_dim_indices]  # (T*fs, pos_dim)
                dists_to_plan = np.linalg.norm(plan_positions - current_pos, axis=1)
            nearest_frame = int(np.argmin(dists_to_plan))
            sub_goal_idx = min(nearest_frame + self.sub_goal_interval, plan_frame_format.shape[0] - 1)
            sub_goal_pos = plan_slice_np[sub_goal_idx, self.pos_dim_indices]
            sub_goal_step = sub_goal_idx
            # Initialize sub_goal_sim_state as current state (will be updated during rollout)
            sub_goal_sim_state = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in current_sim_state.items()} if current_sim_state is not None else None
            sub_goal_sim_state["qpos"][:2] = sub_goal_pos  # physical: qpos[:2] is always world (x,y)

        # Execute plan: iterate, ensuring at least open_loop_horizon steps
        prev_sim_state = None
        loop_cnt = 0
        _reached_last_subgoal = False
        rollout_agent_positions = []
        rollout_subgoal_positions = []
        mujoco_frames = []
        _mujoco_renderer = getattr(self, "_cached_mujoco_renderer", None)
        _overhead_camera = getattr(self, "_cached_overhead_camera", None)

        # [TIMING] execution breakdown
        _exec_t0 = time.time()
        _env_step_ms: list = []
        _action_ms: list = []
        _dist_ms: list = []       # subgoal distance computation (TD metric or L2)
        _get_state_ms: list = []  # _get_sim_state() after env.step()
        _overhead_ms: list = []   # everything else per step (reward tracking, trajectory append, etc.)

        while loop_cnt < self.open_loop_horizon:
            _loop_t0 = time.time()

            # Update sub_goal for antmaze (with interval logic)
            if "antmaze" in self.env_id:
                _dist_t0 = time.time()
                if self.use_TD_metric_as_dist:
                    _cur_obs = np.concatenate([current_sim_state["qpos"], current_sim_state["qvel"]])[self.obs_dim_indices].reshape(1, -1)
                    _sg_obs = np.concatenate([sub_goal_sim_state["qpos"], sub_goal_sim_state["qvel"]])[self.obs_dim_indices].reshape(1, -1)
                    dist_to_sg = float(self._compute_state_temporal_dist_np(_cur_obs, _sg_obs)[0])
                else:
                    dist_to_sg = np.linalg.norm(current_sim_state["qpos"][:2] - sub_goal_sim_state["qpos"][:2])  # physical
                _dist_ms.append((time.time() - _dist_t0) * 1000)

                if dist_to_sg < self.meeting_delta:
                    if sub_goal_step < plan_frame_format.shape[0] - self.sub_goal_interval:
                        sub_goal_step += self.sub_goal_interval
                        sub_goal_sim_state["qpos"][:2] = plan_slice_np[sub_goal_step, self.pos_dim_indices]
                    # else:
                        # Reached last sub_goal — take one final step to collect reward, then exit
                        # _reached_last_subgoal = True

            rollout_agent_positions.append(current_sim_state["qpos"][:2].copy())  # physical (x,y)
            if sub_goal_sim_state is not None:
                rollout_subgoal_positions.append(sub_goal_sim_state["qpos"][:2].copy())  # physical (x,y)
            else:
                rollout_subgoal_positions.append(np.full((len(self.pos_dim_indices),), np.nan, dtype=np.float32))

            # Compute action
            _act_t0 = time.time()
            if use_diffused_action:
                plan_frame = plan_frame_format[min(loop_cnt, plan_frame_format.shape[0] - 1)]
                _, action, _ = self.split_bundle(plan_frame)
            else:
                action = self._compute_action_from_plan(
                    agent=agent,
                    sub_goal_sim_state=sub_goal_sim_state,
                    current_sim_state=current_sim_state,
                    prev_sim_state=prev_sim_state,
                )
            _action_ms.append((time.time() - _act_t0) * 1000)

            # Execute action in environment
            action_np = action.detach().cpu().numpy()

            _step_t0 = time.time()
            obs_numpy, reward, done, _ = envs.step(np.nan_to_num(action_np))
            _env_step_ms.append((time.time() - _step_t0) * 1000)

            # Ensure obs_numpy is 2D: (batch_size, obs_dim)
            if obs_numpy.ndim == 1:
                obs_numpy = obs_numpy[None, :]  # Add batch dimension

            # [MUJOCO RENDER] Capture overhead frame after each step
            if _mujoco_renderer is not None and _overhead_camera is not None:
                try:
                    import mujoco as _mujoco
                    import numpy as _np
                    _mujoco_renderer.update_scene(envs.envs[0].data, camera=_overhead_camera)
                    _frame = _mujoco_renderer.render()
                    # Rotate 90° clockwise to correct overhead orientation
                    _frame = _np.rot90(_frame, k=3)
                    mujoco_frames.append(_frame.copy())
                except Exception:
                    pass

            _overhead_t0 = time.time()

            # Track rewards
            reached = np.logical_or(reached, reward >= 1.0)
            episode_reward += reward
            episode_reward_if_stay += np.where(~reached, reward, 1)
            first_reach += ~reached

            # Collect trajectory
            obs_torch = torch.from_numpy(obs_numpy[:, self.obs_dim_indices]).float()
            bundle = self.make_bundle(obs_torch, action, reward[..., None])
            trajectory.append(bundle)

            # Update sim states for next iteration
            prev_sim_state = current_sim_state
            _get_t0 = time.time()
            current_sim_state = self._get_sim_state(envs)
            _get_state_ms.append((time.time() - _get_t0) * 1000)

            _overhead_ms.append((time.time() - _overhead_t0) * 1000)

            # Increment counter
            loop_cnt += 1

            # Check for episode termination
            # For backward planning, we skip the done condition because the environment
            # signals done when reaching the goal, but in backward mode we're planning
            # from the goal towards the start. So done=True initially and we should ignore it.
            if not is_backward and done.any():
                break

            # Exit after collecting reward from the final sub_goal step
            # if _reached_last_subgoal:
            #     break

        # [TIMING] Execute plan timing summary
        _exec_total_ms = (time.time() - _exec_t0) * 1000
        _n_steps = max(loop_cnt, 1)
        _dist_total   = sum(_dist_ms)
        _action_total = sum(_action_ms)
        _step_total   = sum(_env_step_ms)
        _state_total  = sum(_get_state_ms)
        _overhead_total = sum(_overhead_ms)
        # overhead includes _get_state time; separate it for clarity
        _overhead_excl_state = _overhead_total - _state_total
        self._tlog("timing.execute_plan", {
            "loop_cnt": loop_cnt,
            "total_exec_ms": round(_exec_total_ms, 1),
            # --- per-component totals ---
            "dist_total_ms":    round(_dist_total, 1),
            "action_total_ms":  round(_action_total, 1),
            "env_step_total_ms": round(_step_total, 1),
            "get_state_total_ms": round(_state_total, 1),
            "overhead_total_ms": round(_overhead_excl_state, 1),
            # --- per-step means ---
            "dist_mean_ms":     round(_dist_total / _n_steps, 2),
            "action_mean_ms":   round(_action_total / _n_steps, 2),
            "env_step_mean_ms": round(_step_total / _n_steps, 2),
            "env_step_max_ms":  round(max(_env_step_ms), 2) if _env_step_ms else 0.0,
            "get_state_mean_ms": round(_state_total / _n_steps, 2),
            "overhead_mean_ms": round(_overhead_excl_state / _n_steps, 2),
            # --- percentage of total exec time ---
            "dist_pct":     round(_dist_total / (_exec_total_ms + 1e-6) * 100, 1),
            "action_pct":   round(_action_total / (_exec_total_ms + 1e-6) * 100, 1),
            "env_step_pct": round(_step_total / (_exec_total_ms + 1e-6) * 100, 1),
            "get_state_pct": round(_state_total / (_exec_total_ms + 1e-6) * 100, 1),
            "overhead_pct": round(_overhead_excl_state / (_exec_total_ms + 1e-6) * 100, 1),
        }, depth=0)


        # Return results
        reward_dict = {
            "reached": reached,
            "episode_reward": episode_reward,
            "episode_reward_if_stay": episode_reward_if_stay,
            "first_reach": first_reach,
        }

        rollout_viz = {
            "agent_positions": np.asarray(rollout_agent_positions, dtype=np.float32),
            "subgoal_positions": np.asarray(rollout_subgoal_positions, dtype=np.float32),
            "mujoco_frames": np.stack(mujoco_frames) if mujoco_frames else None,
        }

        return trajectory, reward_dict, rollout_viz

    # ------------------------------------------------------------------
    # Leaf plan rollout for MCTS expansion
    # ------------------------------------------------------------------

    def _rollout_leaf_plan(
        self,
        leaf_plan_unnormalized: torch.Tensor,
        new_denoised_start_idx: int,
        new_denoised_end_idx: int,
        agent: Any,
        envs: Any,
        parent_sim_state: Optional[dict] = None,
        is_backward: bool = False,
    ) -> Optional[dict]:
        """
        Execute a freshly denoised plan segment in the actual environment.
        Restores the parent's physical state before stepping to ensure consistency.

        Args:
            leaf_plan_unnormalized: Fully assembled plan tensor, shape (T*fs, 1, c) unnormalized, where T is tokens and fs is frame_stack.
            new_denoised_start_idx: Start frame index (T*fs) of the freshly denoised chunk.
            new_denoised_end_idx: End frame index (exclusive, T*fs) of the freshly denoised chunk.
            agent: RL agent (used for antmaze sub-goal following).
            envs: Vectorized environment.
            parent_sim_state: Physical state (qpos/qvel) of the parent node to restore.

        Returns:
            final_sim_state: dictionary containing reached qpos/qvel (None if extraction failed).
        """
        # Restore parent's physical state before simulation
        assert parent_sim_state is not None, (
            "Parent sim state must be provided for rollout"
        )

        self._set_sim_state(envs, parent_sim_state)

        plan_slice = leaf_plan_unnormalized[
            new_denoised_start_idx:new_denoised_end_idx
        ]  # (chunk_t*fs, 1, c) in frame format

        if plan_slice.shape[0] == 0:
            return self._get_sim_state(envs)

        # Verify that current sim state matches parent_sim_state after restoration
        current_sim_state = self._get_sim_state(envs)
        assert current_sim_state is not None, "Failed to get current sim state"

        parent_qpos = parent_sim_state["qpos"][:2]  # physical (x,y)
        current_qpos = current_sim_state["qpos"][:2]  # physical (x,y)

        qpos_diff = np.linalg.norm(current_qpos - parent_qpos)
        assert qpos_diff < 1e-5, (
            f"After _set_sim_state, current qpos {current_qpos} does not match "
            f"parent_sim_state qpos {parent_qpos}. Diff: {qpos_diff}"
        )

        # Execute plan with parent state injection for continuous state stitching
        # This restores parent's complete sim state before rolling out the new plan
        trajectory, _, _ = self._execute_plan_in_env(
            plan_frame_format=plan_slice,
            envs=envs,
            agent=agent if "antmaze" in self.env_id else None,
            use_diffused_action=False,
            parent_sim_state=parent_sim_state,  # Pass complete state for restoration
            is_backward=is_backward,
        )
        # Extract all obs from trajectory frames
        # trajectory is a list of bundles, not a tensor
        trajectory_bundle_list = []
        for trajectory_bundle in trajectory:
            obs_t, _, _ = self.split_bundle(trajectory_bundle)
            trajectory_bundle_list.append(obs_t)

        # Stack all obs: (T, batch, obs_dim) -> (T, obs_dim) when batch=1 and cat along dim=0
        trajectory_all_obs = torch.cat(trajectory_bundle_list, dim=0)  # (T, obs_dim)
        # Extract x,y positions: if batch dimension exists, extract it
        if trajectory_all_obs.dim() == 3:
            # Shape is (T, batch, obs_dim)
            trajectory_obs_positions = trajectory_all_obs[:, 0, self.pos_dim_indices].detach().cpu().numpy()  # (T, pos_dim)
        else:
            # Shape is (T, obs_dim) - batch was singleton and got removed
            trajectory_obs_positions = trajectory_all_obs[:, self.pos_dim_indices].detach().cpu().numpy()  # (T, pos_dim)

        # Also get final obs for state update
        obs, _, _ = self.split_bundle(trajectory[-1])

        final_sim_state = self._get_sim_state(envs)  # dummy sim_state
        final_sim_state["qpos"][:2] = obs[0, self.pos_dim_indices]  # physical: qpos[:2] = world (x,y)

        return final_sim_state
