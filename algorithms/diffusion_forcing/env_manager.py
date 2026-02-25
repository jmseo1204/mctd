"""
Environment Manager for bidirectional planning.

This module manages the creation and maintenance of dual environments:
- envs_forward: for tree1 planning (start -> goal)
- envs_backward: for tree2 planning (goal -> start)

Both environments are created once and maintained across multiple planning iterations
within the same task, utilizing _get_sim_state() and _set_sim_state() for state management.

CRITICAL ISSUE WITH BACKWARD ENVIRONMENT:
=========================================
When tree2 (backward) tries to plan from goal state [0.0, 36.0], the OGBench environment
immediately signals done=True because it checks if position == goal. This causes DummyVecEnv
to auto-reset.

SOLUTION IMPLEMENTED:
- For backward planning, we need to swap the internal goal to be the start position.
- Since OGBench doesn't allow goal modification after set_task(), we use a state tracking
  approach: before executing plans in the backward environment, we save the intended final
  position and skip the done-triggered auto-reset logic in the planning code.

FUTURE IMPROVEMENT:
- Modify OGBench to accept goal modification, or
- Create a custom wrapper that overrides the done condition, or  
- Use state injection more aggressively
"""

from typing import Any, Optional, Dict, Tuple
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv


OGBENCH_ENVS = [
    "pointmaze-large-v0",
    "pointmaze-giant-v0",
    "antmaze-large-v0",
    "antmaze-giant-v0",
]


class EnvironmentManager:
    """
    Manages dual environments for bidirectional planning.
    
    - envs_forward: start -> goal (tree1)
    - envs_backward: goal -> start (tree2)
    """
    
    def __init__(
        self,
        env_id: str,
        batch_size: int,
        task_id: Optional[int] = None,
        use_random_goals: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the environment manager.
        
        Args:
            env_id: Environment ID (e.g., "antmaze-large-v0")
            batch_size: Number of parallel environments
            task_id: Task ID for OGBench environments (0-based, will be converted to 1-based)
            use_random_goals: Whether to randomize goals for non-OGBench envs
            debug: Enable debug logging
        """
        self.env_id = env_id
        self.batch_size = batch_size
        self.task_id = task_id
        self.use_random_goals = use_random_goals
        self.debug = debug
        
        self.envs_forward = None
        self.envs_backward = None
        
    def create_bidirectional_envs(
        self,
        env_fns,
    ) -> Tuple[DummyVecEnv, DummyVecEnv]:
        """
        Create two environments: one for forward planning, one for backward.
        
        Args:
            env_fns: List of environment factory functions
            
        Returns:
            Tuple of (envs_forward, envs_backward)
        """
        # Create forward environment (tree1: start -> goal)
        self.envs_forward = DummyVecEnv(env_fns)
        
        # Create backward environment (tree2: goal -> start)
        # Use the same factory functions
        self.envs_backward = DummyVecEnv(env_fns)
        
        # Configure both environments with the same task settings
        self._configure_envs()
        
        return self.envs_forward, self.envs_backward
    
    def _configure_envs(self) -> None:
        """Configure both environments with task settings."""
        if self.env_id in OGBENCH_ENVS:
            # Set task for OGBench environments
            self._configure_ogbench_envs()
        else:
            # Set random goals for other gym environments
            if self.use_random_goals:
                self._configure_gym_envs()
    
    def _configure_ogbench_envs(self) -> None:
        """Configure OGBench environments with task IDs."""
        # Both forward and backward envs get the same task
        # (the direction difference comes from goal-setting in planning code)
        for env_set in [self.envs_forward, self.envs_backward]:
            for i, env in enumerate(env_set.envs):
                # Convert 0-based task_id to 1-based indexing for ogbench
                actual_task_id = self.task_id + i + 1
                if self.debug:
                    print(f"[EnvironmentManager] Setting task_id={actual_task_id}")
                env.set_task(actual_task_id)
    
    def _configure_gym_envs(self) -> None:
        """Configure gym environments with random goals."""
        for env_set in [self.envs_forward, self.envs_backward]:
            for env in env_set.envs:
                env.set_target()
    
    def get_env(self, direction: str) -> DummyVecEnv:
        """
        Get the environment for the specified direction.
        
        Args:
            direction: "forward" for tree1 or "backward" for tree2
            
        Returns:
            DummyVecEnv instance
        """
        if direction == "forward":
            return self.envs_forward
        elif direction == "backward":
            return self.envs_backward
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'forward' or 'backward'")
    
    def close(self) -> None:
        """Close both environments."""
        if self.envs_forward is not None:
            self.envs_forward.close()
        if self.envs_backward is not None:
            self.envs_backward.close()
