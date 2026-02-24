import numpy as np
import math
import torch
import torch.nn
from typing import Optional

class TreeNode():
    def __init__(self, name: str, depth: int, parent_node: Optional['TreeNode'], 
                 children_node_guidance_scales: list, plan_history: list, 
                 guidance_scale: Optional[float] = None, terminal_depth: Optional[int] = None, 
                 value: Optional[float] = None, value_estimation_plan: Optional[torch.Tensor] = None, 
                 virtual_visit_weight: float = 0.0, current_levels: Optional[np.ndarray] = None,
                 obs_pos: Optional[np.ndarray] = None,
                 sim_state: Optional[dict] = None,
                 target_node: Optional['TreeNode'] = None):
        self.name = name
        self.depth = depth
        self._parent_node = parent_node
        self._children_node_guidance_scales = children_node_guidance_scales
        self.plan_history = plan_history
        self.guidance_scale = guidance_scale
        self.terminal_depth = terminal_depth
        self.virtual_visit_weight = virtual_visit_weight
        self.virtual_visit_count = 0
        # The maximum number of children nodes is same to the number of children node guidances
        self._children_nodes = [
            {'guidance_scale': self._children_node_guidance_scales[i], 'node': None, "virtually_visited": False}
            for i in range(len(self._children_node_guidance_scales))
        ]

        self.value = value
        self.value_estimation_plan = value_estimation_plan
        self.visit_count = 0
        
        # Store denoising state for incremental segment-wise denoising in MCTD
        self.current_levels = current_levels

        # Agent's actual last reached position after MPC rollout from this node's plan.
        # Root node: set to start (tree1) or goal (tree2) coords.
        # Child nodes: set after _rollout_leaf_plan() in the interact loop.
        self.obs_pos: Optional[np.ndarray] = obs_pos
        self.sim_state: Optional[dict] = sim_state
        # The opposite tree's leaf node that this node targets during bidirectional search.
        # Set by _select_dynamic_goal() in _run_mcts_search().
        self.target_node: Optional['TreeNode'] = target_node

        # Expansion state flag: tracks whether this node can be expanded further.
        # For leaf nodes: initialized based on is_expandable() evaluation.
        # For non-leaf nodes: updated during backpropagation when all children become unexpandable.
        self.is_expandable_flag: bool = self.is_expandable()

    def __lt__(self, other):
        return self.name < other.name
    
    def __eq__(self, other):
        return self.name == other.name

    def is_root_node(self):
        return self._parent_node is None
    
    def is_leaf_node(self):
        if self._children_nodes[0]["node"] is None:
            return False
        return True
    
    def is_expandable(self, consider_virtually_visited=False):
        if self.depth == self.terminal_depth:
            return False
        for child_node in self._children_nodes:
            if child_node["node"] is None and (not consider_virtually_visited or not child_node["virtually_visited"]):
                return True
        return False

    def is_selectable(self):
        for child_node in self._children_nodes:
            if child_node["node"] is None:
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
    
    def set_value_estimation_plan(self, value_estimation_plan):
        self.value_estimation_plan = value_estimation_plan

    # Following UCT(Upper Confidence Boundary of Tree)
    def select(self, exp_weight=math.sqrt(2), leaf_parallelization=False):
        for child_node in self._children_nodes:
            if child_node["node"] is None:
                raise ValueError('Child node is None in select method')

        # Filter out unexpandable children - only consider expandable ones for selection
        expandable_children_indices = [
            i for i, child_node in enumerate(self._children_nodes)
            if child_node["node"].is_expandable_flag
        ]

        if not expandable_children_indices:
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
                i for i in range(len(self._children_nodes)) if self._children_nodes[i]['node'] is None and (not consider_virtually_visited or not self._children_nodes[i]['virtually_visited'])
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
            'value_estimation_plan': None,
            'virtual_visit_weight': self.virtual_visit_weight,
            'terminal_depth': self.terminal_depth,
            'current_levels': self.current_levels,  # Inherit parent's denoising state
            'obs_pos': None,  # Will be set by _rollout_leaf_plan() after MPC rollout
            'sim_state': None,
            'target_node': None,  # Will be set by _select_dynamic_goal() in _run_mcts_search()
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

    def backpropagate(self):
        self.visit_count += 1
        self.virtual_visit_count = 0 # reset the virtual visit count
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
            if created_children and all(not c.is_expandable_flag for c in created_children):
                # All children are unexpandable → mark parent as unexpandable too
                self.is_expandable_flag = False

        if self._parent_node is not None:
            self._parent_node.backpropagate()

    def check_virtual_visit_count(self):
        for child_node in self._children_nodes:
            if child_node["node"] is not None:
                if child_node["node"].virtual_visit_count > 0:
                    raise ValueError(f"Virtual visit count of {child_node['node'].name} is {child_node['node'].virtual_visit_count}")
                child_node["node"].check_virtual_visit_count()