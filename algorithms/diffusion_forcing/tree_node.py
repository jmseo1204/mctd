import numpy as np
import math
import torch
import torch.nn
from typing import Optional

class TreeNode():
    def __init__(self, name: str, depth: int, parent_node: Optional['TreeNode'], 
                 children_node_guidance_scales: list, plan_history: list, 
                 guidance_scale: Optional[float] = None, terminal_depth: Optional[int] = None, 
                 value: Optional[float] = None, replanned_plan: Optional[torch.Tensor] = None,
                 virtual_visit_weight: float = 0.0, current_levels: Optional[np.ndarray] = None,
                 obs: Optional[np.ndarray] = None,
                 sim_state: Optional[dict] = None,
                 target_node: Optional['TreeNode'] = None,
                 selection_count: Optional[int] = None,
                 cluster_subplans: Optional[list] = None):
        self.name = name
        self.depth = depth
        self._parent_node = parent_node
        if cluster_subplans is not None and len(cluster_subplans) == 0:
            # No feasible uncertainty survivors remain for this node, so keep it
            # structurally unexpandable from creation time onward.
            children_node_guidance_scales = []
        self._children_node_guidance_scales = children_node_guidance_scales
        self.plan_history = plan_history
        self.guidance_scale = guidance_scale
        self.terminal_depth = terminal_depth
        self.virtual_visit_weight = virtual_visit_weight
        self.virtual_visit_count = 0
        # The maximum number of children nodes is same to the number of children node guidances
        self._children_nodes = [
            {'guidance_scale': self._children_node_guidance_scales[i], 'node': None, "virtually_visited": False, "permanently_dead": False}
            for i in range(len(self._children_node_guidance_scales))
        ]

        self.value = value
        self.replanned_plan = replanned_plan
        self.visit_count = 0
        
        # Store denoising state for incremental segment-wise denoising in MCTD
        self.current_levels = current_levels

        # Agent's actual last reached position after MPC rollout from this node's plan.
        # Root node: set to start (tree1) or goal (tree2) coords.
        # Child nodes: set after _rollout_leaf_plan() in the interact loop.
        self.obs: Optional[np.ndarray] = obs
        self.sim_state: Optional[dict] = sim_state
        # The opposite tree's leaf node that this node targets during bidirectional search.
        # Set by _select_dynamic_goal() in _run_mcts_search().
        self.target_node: Optional['TreeNode'] = target_node
        # Global selection-event counter bookkeeping.
        # selection_count: count at which this node itself was created.
        # last_selection_count: latest count at which this node was selected as a parent.
        self.selection_count: Optional[int] = selection_count
        self.last_selection_count: Optional[int] = selection_count

        # Per-cluster sub-plans derived from uncertainty sampling.
        # Populated after _compute_node_uncertainty when use_cluster_subplan_as_expansion=True.
        # Each entry is a dict with keys:
        #   "plan_hist":      (fast_steps+1, plan_tokens*fs, c) tensor — full denoising history
        #                     of the representative sample selected for this cluster.
        #   "guidance_scale": float — guidance scale used for that sample.
        #   "current_levels": (1, plan_tokens) np.ndarray — noise state for next expansion.
        #   "feasible_score": float — optional maximin KDE score used for representative selection.
        # len(cluster_subplans) == n_clusters for this node.
        self.cluster_subplans: Optional[list] = cluster_subplans

        # Expansion state flag: tracks whether this node can be expanded further.
        # For leaf nodes: initialized based on is_expandable() evaluation.
        # For non-leaf nodes: updated during backpropagation when all children become unexpandable.
        self.is_expandable_flag: bool = self.is_expandable()

        # Reset counter: tracks how many times this node's children have been wiped and
        # re-opened for expansion (triggered when all children are exhausted but non-terminal).
        self._reset_count: int = 0

    def __lt__(self, other):
        return self.name < other.name
    
    def __eq__(self, other):
        return self.name == other.name

    def is_root_node(self):
        return self._parent_node is None
    
    def is_leaf_node(self):
        if not self._children_nodes:
            return False
        if self._children_nodes[0]["node"] is None:
            return False
        return True
    
    def is_expandable(self, consider_virtually_visited=False):
        if self.depth == self.terminal_depth:
            return False
        for child_node in self._children_nodes:
            if child_node["node"] is None and not child_node["permanently_dead"] and (not consider_virtually_visited or not child_node["virtually_visited"]):
                return True
        return False

    def mark_slot_permanently_dead(self, slot_index: int):
        """Mark a child slot as permanently dead (e.g. dedup-killed).
        Dead slots are skipped by is_expandable() and get_expandable_candidate()
        and can never be revived, unlike the temporary virtually_visited flag."""
        self._children_nodes[slot_index]["permanently_dead"] = True
        self.is_expandable_flag = self.is_expandable()

    def reset_children_slots(self, n_slots: int, guidance_scales: list) -> None:
        """Replace _children_nodes with a fresh set of n_slots slots.

        Used by the cluster-reuse expansion path to resize the children array
        to match the number of clusters found during uncertainty sampling.
        All existing slot state (permanently_dead, virtually_visited, node
        references) is discarded and replaced with clean empty slots.

        Args:
            n_slots: Number of new child slots (= number of clusters).
            guidance_scales: Per-slot guidance scale values, length n_slots.
        """
        assert len(guidance_scales) == n_slots, (
            f"reset_children_slots: guidance_scales length {len(guidance_scales)} "
            f"!= n_slots {n_slots}"
        )
        self._children_node_guidance_scales = list(guidance_scales)
        self._children_nodes = [
            {
                "guidance_scale": guidance_scales[i],
                "node": None,
                "virtually_visited": False,
                "permanently_dead": False,
            }
            for i in range(n_slots)
        ]
        self.is_expandable_flag = self.is_expandable()

    def reset_dead_children(self) -> bool:
        """Wipe all child slots (both permanently_dead and alive-but-exhausted) so the
        node can be re-expanded from scratch.  Called by select() when every child's
        subtree is exhausted without any terminal node being reached.

        Returns True if the reset was applied, False if max_resets has been reached."""
        for slot in self._children_nodes:
            slot["node"] = None
            slot["permanently_dead"] = False
            slot["virtually_visited"] = False
        self._reset_count += 1
        self.is_expandable_flag = self.is_expandable()
        return True

    def is_selectable(self):
        if not self._children_nodes:
            return False
        for child_node in self._children_nodes:
            if child_node["node"] is None and not child_node["permanently_dead"]:
                return False
        return True

    def is_terminal(self):
        if self.depth == self.terminal_depth:
            return True
        return False
    
    def set_value(self, value):
        self.value = value

    def set_plan_history(self, plan_history):
        self.plan_history.append(plan_history)
    
    def set_replanned_plan(self, replanned_plan):
        self.replanned_plan = replanned_plan

    # Following UCT(Upper Confidence Boundary of Tree)
    def select(self, exp_weight=math.sqrt(2), leaf_parallelization=False, max_child_resets: int = 3):
        for child_node in self._children_nodes:
            if child_node["node"] is None and not child_node["permanently_dead"]:
                raise ValueError('Child node is None in select method')

        # Filter out unexpandable children - only consider expandable ones for selection
        # Skip permanently_dead slots (node is None, would cause AttributeError)
        expandable_children_indices = [
            i for i, child_node in enumerate(self._children_nodes)
            if not child_node["permanently_dead"] and child_node["node"] is not None and child_node["node"].is_expandable_flag
        ]

        if not expandable_children_indices:
            # All alive children have exhausted subtrees.
            # If none are terminal and we still have reset budget, wipe children and
            # return self so the outer selection loop re-expands from this node.
            alive_slots = [c for c in self._children_nodes
                           if not c["permanently_dead"] and c["node"] is not None]
            has_terminal = any(c["node"].is_terminal() for c in alive_slots)
            if not has_terminal and self._reset_count < max_child_resets:
                self.reset_dead_children()
                return self  # outer while-loop re-evaluates: now is_expandable=True → exits
            raise ValueError('No expandable children available for selection')

        # Calculate total visit count only for expandable children
        total_visit_count = sum([
            self._children_nodes[i]["node"].visit_count +
            self._children_nodes[i]["node"].virtual_visit_count * self.virtual_visit_weight
            for i in expandable_children_indices
        ])

        # Calculate UCT values only for expandable children
        uct_values = []
        for idx in expandable_children_indices:
            _node = self._children_nodes[idx]["node"]
            _value = _node.value
            _visit_count = _node.visit_count + _node.virtual_visit_count * self.virtual_visit_weight
            uct_values.append(_value + exp_weight * np.sqrt(np.log(1e-6 + total_visit_count) / (1e-6 + _visit_count)))

        # Select the best among expandable children
        best_idx_in_expandable = np.argmax(uct_values)
        selected_index = expandable_children_indices[best_idx_in_expandable]

        if leaf_parallelization:
            self._children_nodes[selected_index]["node"].virtual_visit_count += len(self._children_nodes)
        else:
            self._children_nodes[selected_index]["node"].virtual_visit_count += 1
        return self._children_nodes[selected_index]["node"]
    
    def expand(self, **kwargs):
        if len(kwargs) > 0:
            selected_index = int(kwargs['name'].split('-')[-1])
            assert self.name == kwargs['parent_node'].name, "Parent node is the same as the current node"
            # Create a new child node
            self._children_nodes[selected_index]['node'] = TreeNode(**kwargs)
            self._children_nodes[selected_index]['virtually_visited'] = False # virtually visit flag is False when the node is created
        return self._children_nodes[selected_index]['node']

    def get_expandable_candidate(self, index=None, consider_virtually_visited=False):
        if index is None:
            remaining_children_indices = [
                i for i in range(len(self._children_nodes))
                if self._children_nodes[i]['node'] is None
                and not self._children_nodes[i]['permanently_dead']
                and (not consider_virtually_visited or not self._children_nodes[i]['virtually_visited'])
            ]
            #selected_index = remaining_children_indices[0] # To check the best-of-n MCTD
            selected_index = np.random.choice(remaining_children_indices)
        else:
            selected_index = index
        candidate_info = {
            'name': self.name + f'-{selected_index}',
            'depth': self.depth + 1,
            'parent_node': self,
            'children_node_guidance_scales': self._children_node_guidance_scales,
            'plan_history': self.plan_history.copy(),
            'guidance_scale': self._children_nodes[selected_index]['guidance_scale'],
            'value': None,
            'replanned_plan': None,
            'virtual_visit_weight': self.virtual_visit_weight,
            'terminal_depth': self.terminal_depth,
            'current_levels': self.current_levels,  # Inherit parent's denoising state
            'obs': None,  # Will be set by _rollout_leaf_plan() after MPC rollout
            'sim_state': None,
            'target_node': None,  # Will be set by _select_dynamic_goal() in _run_mcts_search()
            'selection_count': self.last_selection_count,
        }
        # this function call means that the node is virtually visited in parallel search
        self._children_nodes[selected_index]['virtually_visited'] = True
        return candidate_info

    def get_expandable_node_names(self, consider_virtually_visited=True):
        expandable_nodes = []
        for child_idx, child_node in enumerate(self._children_nodes):
            if child_node["node"] is None and (not consider_virtually_visited or not child_node["virtually_visited"]):
                expandable_nodes.append(self.name + f'-{child_idx}')
            else:
                if child_node["node"] is not None:
                    expandable_nodes.extend(child_node["node"].get_expandable_node_names(consider_virtually_visited=consider_virtually_visited))
        return expandable_nodes

    def backpropagate(self, preserve_value: bool = False):
        self.visit_count += 1
        self.virtual_visit_count = 0 # reset the virtual visit count
        if not preserve_value:
            self.value = max([child_node["node"].value for child_node in self._children_nodes if child_node["node"] is not None])

        # [EXPANSION STATE] Update is_expandable_flag during backpropagation
        # For leaf nodes: evaluate using is_expandable() method
        # For non-leaf nodes: mark unexpandable if all children are unexpandable
        if not self._children_nodes or all(c["node"] is None for c in self._children_nodes):
            # Leaf node: evaluate current expansion capability
            self.is_expandable_flag = self.is_expandable()
        else:
            # Non-leaf node: check if all created children are unexpandable
            created_children = [c["node"] for c in self._children_nodes if c["node"] is not None]
            has_empty_slots = any(c["node"] is None and not c["permanently_dead"] for c in self._children_nodes)
            if not has_empty_slots and created_children and all(not c.is_expandable_flag for c in created_children):
                # All children are unexpandable and no empty slots remain → mark parent as unexpandable too
                self.is_expandable_flag = False

        if self._parent_node is not None:
            self._parent_node.backpropagate(preserve_value=preserve_value)

    def check_virtual_visit_count(self):
        for child_node in self._children_nodes:
            if child_node["node"] is not None:
                if child_node["node"].virtual_visit_count > 0:
                    raise ValueError(f"Virtual visit count of {child_node['node'].name} is {child_node['node'].virtual_visit_count}")
                child_node["node"].check_virtual_visit_count()
