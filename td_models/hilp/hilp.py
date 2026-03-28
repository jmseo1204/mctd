import numpy as np
from typing import Sequence, Dict, Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: Optional[int] = None,
                 activations: Callable = F.gelu,
                 activate_final: bool = False,
                 layer_norm: bool = False):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        if output_dim is not None:
            dims.append(output_dim)

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2 or activate_final:
                # Add activation module instance
                if activations == F.gelu:
                    self.layers.append(nn.GELU())
                elif activations == F.relu:
                     self.layers.append(nn.ReLU())
                else:
                     try:
                         self.layers.append(activations())
                     except:
                         raise ValueError(f"Unsupported activation function: {activations}")

                if layer_norm:
                    self.layers.append(nn.LayerNorm(dims[i+1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class LayerNormRepresentation(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: int, # Final output dimension
                 activate_final: bool = True,
                 use_layer_norm: bool = True,
                 ensemble: bool = True):
        super().__init__()
        self.ensemble = ensemble

        mlp_params = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "activate_final": activate_final,
            "layer_norm": use_layer_norm,
            "activations": F.gelu # Keep consistent with JAX default
        }

        if self.ensemble:
            self.mlp1 = MLP(**mlp_params)
            self.mlp2 = MLP(**mlp_params)
        else:
            self.mlp = MLP(**mlp_params)

    def forward(self, observations: torch.Tensor):
        if self.ensemble:
            out1 = self.mlp1(observations)
            out2 = self.mlp2(observations)
            return out1, out2
        else:
            return self.mlp(observations)


class GoalConditionedPhiValue(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 hidden_dims: Sequence[int],
                 skill_dim: int,
                 use_layer_norm: bool = True,
                 ensemble: bool = True):
        super().__init__()
        self.ensemble = ensemble
        self.skill_dim = skill_dim

        self.phi_net = LayerNormRepresentation(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=skill_dim, # Phi output dim
            activate_final=False, # JAX version had activate_final=False for the phi output layer
            use_layer_norm=use_layer_norm,
            ensemble=ensemble
        )

    def get_phi(self, observations: torch.Tensor) -> torch.Tensor:
        """Returns the phi representation, taking the first element if ensembled."""
        phi_output = self.phi_net(observations)
        # Return first element of ensemble, consistent with JAX implementation's apparent use
        return phi_output[0] if self.ensemble else phi_output

    def forward(self,
                observations: torch.Tensor,
                goals: torch.Tensor):
        """Calculates value V(s,g) = -||phi(s) - phi(g)||."""
        phi_s = self.phi_net(observations)
        phi_g = self.phi_net(goals)

        if self.ensemble:
            dist_sq1 = ((phi_s[0] - phi_g[0])**2).sum(dim=-1)
            dist_sq2 = ((phi_s[1] - phi_g[1])**2).sum(dim=-1)
            v1 = -torch.sqrt(torch.clamp(dist_sq1, min=1e-6))
            v2 = -torch.sqrt(torch.clamp(dist_sq2, min=1e-6))
            return v1, v2
        else:
            dist_sq = ((phi_s - phi_g)**2).sum(dim=-1)
            v = -torch.sqrt(torch.clamp(dist_sq, min=1e-6))
            return v


class HILP(nn.Module):
    def __init__(
        self,
        obs_dim,
        skill_dim,
        device,
        value_hidden_dims=(512, 512, 512),
        use_layer_norm=True,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.value_hidden_dims = value_hidden_dims
        self.use_layer_norm = use_layer_norm
        self.device = device

        self.value = GoalConditionedPhiValue(
            obs_dim=obs_dim,
            hidden_dims=self.value_hidden_dims,
            skill_dim=self.skill_dim,
            use_layer_norm=self.use_layer_norm,
            ensemble=True # HILP uses ensemble V based on phi distance
        ).to(device)
        self.target_value = GoalConditionedPhiValue(
             obs_dim=obs_dim,
            hidden_dims=self.value_hidden_dims,
            skill_dim=self.skill_dim,
            use_layer_norm=self.use_layer_norm,
            ensemble=True
        ).to(device)
        self.target_value.load_state_dict(self.value.state_dict())
        for p in self.target_value.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_phi(self, observations: np.ndarray) -> np.ndarray:
        """Computes the phi representation (using the first ensemble member)."""
        self.eval() # Set to evaluation mode
        obs_tensor = torch.from_numpy(observations).to(self.device).float()

        is_batch = len(obs_tensor.shape) > 1
        if not is_batch:
             obs_tensor = obs_tensor.unsqueeze(0)

        phi_tensor = self.value.get_phi(obs_tensor)

        if not is_batch:
            phi_tensor = phi_tensor.squeeze(0)

        self.train() # Set back to training mode
        return phi_tensor.cpu().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, self.device))


# ---------------------------------------------------------------------------
# JAX/Flax DualHILP wrapper — uses ../HILP/hilp_gcrl/ source directly
# ---------------------------------------------------------------------------

def _find_hilp_gcrl_path(pkl_path: str) -> str:
    """Auto-detect hilp_gcrl directory relative to the mctd_repo root."""
    import os
    # pkl is at <workspace>/mctd_repo/td_models/*.pkl
    # hilp_gcrl is at <workspace>/HILP/hilp_gcrl/
    # td_models/hilp/hilp.py → go up 3 levels to reach workspace root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(pkl_path))))
    candidate = os.path.join(repo_root, "HILP", "hilp_gcrl")
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"hilp_gcrl not found at {candidate}. "
        "Ensure ../HILP/hilp_gcrl exists relative to the workspace root."
    )


def _read_pkl_meta(pkl_path: str) -> dict:
    """Read a sidecar JSON (same stem as pkl) for non-structural config.

    e.g. params_1000000_20260324_002241.pkl  →  params_1000000_20260324_002241.json
    Returns {} if the sidecar does not exist.
    """
    import os, json
    stem = os.path.splitext(pkl_path)[0]
    sidecar = stem + ".json"
    if os.path.isfile(sidecar):
        with open(sidecar) as f:
            return json.load(f)
    return {}


class HILPJax:
    """Wraps the HILP DualHILP JAX/Flax agent using hilp_gcrl source directly.

    All structural hyper-parameters (obs_dim, skill_dim, action_dim,
    share_encoder) are auto-detected from the pkl weights.

    The only parameter that cannot be inferred from weights alone is
    ``aggregator`` (inner_prod vs neg_l2).  HILPJax reads it from a
    sidecar JSON placed next to the pkl file:

        td_models/params_<step>_<ts>.json  →  {"aggregator": "inner_prod"}

    If the sidecar is absent, ``inner_prod`` is used as the default
    (matching the training-script default in train_dual_ogbench.py).

    Interface is compatible with the original HILP class:
      ``value(obs, goal) -> (v, v)``  (duplicate for ensemble compatibility)
    """

    def __init__(self, pkl_path: str, device, hilp_gcrl_path: str = None):
        import sys
        import os
        import pickle
        import numpy as np

        self.device = device

        # ---- 1. Locate hilp_gcrl source ----------------------------------
        gcrl_path = hilp_gcrl_path or _find_hilp_gcrl_path(pkl_path)
        if gcrl_path not in sys.path:
            sys.path.insert(0, gcrl_path)

        import flax
        from train_dual_ogbench import DualHILP

        # ---- 2. Read pkl and infer all structural params -----------------
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        net_params = raw["agent"]["network"]["params"]

        # obs_dim: first Dense input size in phi network
        phi_lnmlp = net_params["networks_value"]["phi"]["LayerNormMLP_0"]
        obs_dim = int(phi_lnmlp["Dense_0"]["kernel"].shape[0])

        # skill_dim: last Dense output size in phi network
        last_dense = sorted(k for k in phi_lnmlp if k.startswith("Dense_"))[-1]
        skill_dim = int(phi_lnmlp[last_dense]["kernel"].shape[1])

        # action_dim: q input = [s, g, a]  →  shape[1] - 2*obs_dim
        q_k = net_params["networks_q_func"]["critic_net"]["VmapLayerNormMLP_0"]["Dense_0"]["kernel"]
        action_dim = int(q_k.shape[1]) - 2 * obs_dim

        # share_encoder: separate psi key exists only when share_encoder=False
        share_encoder = "psi" not in net_params["networks_value"]

        # ---- 3. Read aggregator from sidecar JSON (default: inner_prod) --
        meta = _read_pkl_meta(pkl_path)
        aggregator = meta.get("aggregator", "inner_prod")

        # ---- 4. Build dummy agent and restore weights --------------------
        dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
        dummy_act = np.zeros((1, action_dim), dtype=np.float32)
        agent = DualHILP.create(
            seed=0,
            ex_observations=dummy_obs,
            ex_actions=dummy_act,
            skill_dim=skill_dim,
            aggregator=aggregator,
            share_encoder=share_encoder,
        )
        self._agent = flax.serialization.from_state_dict(agent, raw["agent"])
        self._aggregator = aggregator  # stored separately; config field not serialized

        print(f"[HILPJax] loaded  obs_dim={obs_dim}  skill_dim={skill_dim}"
              f"  action_dim={action_dim}  share_encoder={share_encoder}"
              f"  aggregator={aggregator}"
              + ("  (from sidecar)" if "aggregator" in meta else "  (default)"))

    # ------------------------------------------------------------------
    # PyTorch-compatible interface
    # ------------------------------------------------------------------

    def eval(self):
        """No-op: JAX models have no train/eval mode."""
        return self

    def parameters(self):
        """No-op: JAX model params are not PyTorch Parameters."""
        return iter([])

    def value(self, obs: torch.Tensor, goal: torch.Tensor):
        """Compute V(obs, goal); returns (v, v) for ensemble compatibility."""
        import numpy as np
        obs_np  = obs.detach().cpu().numpy().astype(np.float32)
        goal_np = goal.detach().cpu().numpy().astype(np.float32)

        psi_s = np.array(self._agent.get_psi(obs_np))    # (N, D)
        phi_g = np.array(self._agent.get_phi_goal(goal_np))  # (N, D)

        if self._aggregator == "neg_l2":
            dist_sq = ((psi_s - phi_g) ** 2).sum(axis=-1)
            v_np = -np.sqrt(np.maximum(dist_sq, 1e-6))
        else:  # inner_prod
            v_np = (psi_s * phi_g).sum(axis=-1)

        v = torch.from_numpy(v_np).float().to(self.device)
        return v, v

    def get_phi(self, obs: torch.Tensor) -> torch.Tensor:
        """Return psi(obs) as a torch tensor."""
        import numpy as np
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        psi = np.array(self._agent.get_psi(obs_np))
        return torch.from_numpy(psi).float().to(self.device)

    def compute_grads(self, obs_np: np.ndarray, goal_np: np.ndarray) -> np.ndarray:
        """Compute ∂V/∂obs using JAX grad (vmap+grad over the batch).

        Falls back to central finite differences if JAX grad fails.

        Args:
            obs_np:  (N, obs_dim) float32 — grid observations
            goal_np: (1, obs_dim) float32 — fixed target observation

        Returns:
            grads: (N, 2) float32 — gradient w.r.t. the first two (x, y) dims
        """
        try:
            return self._compute_grads_jax(obs_np, goal_np)
        except Exception:
            return self._compute_grads_fd(obs_np, goal_np)

    def _compute_grads_jax(self, obs_np: np.ndarray, goal_np: np.ndarray) -> np.ndarray:
        """JAX-native gradient via jax.vmap(jax.grad(...)).

        _grad_fn is compiled once per HILPJax instance and reused across all calls.
        phi_g is passed as an explicit argument (not captured in the closure) so
        the XLA-compiled program is goal-agnostic and cache hits across different goals.
        """
        import jax
        import jax.numpy as jnp

        # Build and cache grad_fn once per instance.
        # phi_g is an explicit arg → the compiled program does not embed the goal
        # value, so the same compilation is reused regardless of goal.
        if not hasattr(self, '_grad_fn'):
            aggregator = self._aggregator
            agent = self._agent

            def value_fn(obs_single, phi_g_single):  # (obs_dim,), (skill_dim,) → scalar
                psi = agent.get_psi(obs_single[None])[0]  # (skill_dim,)
                if aggregator == "neg_l2":
                    dist_sq = jnp.sum((psi - phi_g_single) ** 2)
                    return -jnp.sqrt(jnp.maximum(dist_sq, 1e-6))
                else:  # inner_prod
                    return jnp.sum(psi * phi_g_single)

            self._grad_fn = jax.jit(jax.vmap(jax.grad(value_fn, argnums=0)))

        # Compute phi(goal) — result cached per unique goal bytes to avoid
        # redundant Flax forward passes when the goal is constant across calls.
        goal_key = goal_np[:1].tobytes()
        if not hasattr(self, '_phi_g_cache') or self._phi_g_cache[0] != goal_key:
            phi_g = self._agent.get_phi_goal(goal_np[:1])
            if hasattr(phi_g, 'ndim') and phi_g.ndim > 1:
                phi_g = phi_g[0]
            self._phi_g_cache = (goal_key, jnp.array(phi_g))
        phi_g = self._phi_g_cache[1]  # (skill_dim,)

        N = obs_np.shape[0]
        phi_g_batch = jnp.broadcast_to(phi_g[None], (N, phi_g.shape[0]))
        grads_jnp = self._grad_fn(jnp.array(obs_np), phi_g_batch)  # (N, obs_dim)
        return np.array(grads_jnp)[:, :2]  # (N, 2)

    def compute_values_np(self, obs_np: np.ndarray, goal_np: np.ndarray) -> np.ndarray:
        """Compute V(obs, goal) for numpy inputs. Returns (N,) float32.

        Note: For production use, prefer planner._compute_hilp_values() which goes through
        the same pessimistic min(v1,v2) path and applies _hilp_ref_obs padding consistently.
        For HILPJax, v1==v2 (no real ensemble), so results are identical in practice.

        Args:
            obs_np:  (N, obs_dim) float32 — observations
            goal_np: (1, obs_dim) float32 — fixed target (broadcast to N)

        Returns:
            values: (N,) float32 — V(obs, goal) per observation
        """
        N = obs_np.shape[0]
        obs_np = obs_np.astype(np.float32)
        goal_rep = np.broadcast_to(goal_np[:1], (N, goal_np.shape[-1])).copy().astype(np.float32)
        psi_s = np.array(self._agent.get_psi(obs_np))        # (N, D)
        phi_g = np.array(self._agent.get_phi_goal(goal_rep))  # (N, D)
        if self._aggregator == "neg_l2":
            dist_sq = ((psi_s - phi_g) ** 2).sum(axis=-1)
            return -np.sqrt(np.maximum(dist_sq, 1e-6)).astype(np.float32)
        else:  # inner_prod
            return (psi_s * phi_g).sum(axis=-1).astype(np.float32)

    def _compute_grads_fd(self, obs_np: np.ndarray, goal_np: np.ndarray,
                          eps: float = 0.5) -> np.ndarray:
        """Central finite-difference fallback: 4 forward passes total."""
        N = obs_np.shape[0]
        goal_t = torch.from_numpy(
            np.broadcast_to(goal_np[:1], (N, goal_np.shape[-1])).copy()
        ).float().to(self.device)
        grads = np.zeros((N, 2), dtype=np.float32)
        for dim in range(2):
            obs_p = obs_np.copy(); obs_p[:, dim] += eps
            obs_m = obs_np.copy(); obs_m[:, dim] -= eps
            v_p, _ = self.value(torch.from_numpy(obs_p).float().to(self.device), goal_t)
            v_m, _ = self.value(torch.from_numpy(obs_m).float().to(self.device), goal_t)
            grads[:, dim] = (v_p.cpu().numpy() - v_m.cpu().numpy()) / (2 * eps)
        return grads
