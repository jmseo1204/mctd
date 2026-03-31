"""hilp_loader.py
HILP value function loading and grid-memoization utilities.

Public API
----------
HILPMemoizedWrapper
    Drop-in replacement for HILP / HILPJax using precomputed encoder grids and
    bilinear interpolation. Implements the same interface:
        .value(obs_t, goal_t)               → (v1, v2) torch.Tensor
        .compute_grads(obs_np, goal_np)     → (N, 2) np.float32
        .compute_values_np(obs_np, goal_np) → (N,)   np.float32
        .eval() / .parameters()             → no-ops

load_raw_hilp_model(checkpoint_path, device, obs_dim, skill_dim) → model
    Load raw HILP (.pt) or HILPJax (.pkl) from disk.

get_hilp_fn(...)
    Factory: returns either the raw model or a HILPMemoizedWrapper depending on
    use_memoization.  Handles cache hit/miss transparently.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# HILPMemoizedWrapper
# ---------------------------------------------------------------------------

class HILPMemoizedWrapper:
    """Encoder-grid memoized drop-in replacement for HILP / HILPJax.

    Precomputes φ(s) and φ_goal(s) on a G×G (x,y) grid and uses bilinear
    interpolation at query time instead of running the full encoder.

    Ensemble handling:
        - HILPJax  : psi_grids / phi_g_grids each have 1 element → returns (v, v)
        - HILP PyTorch ensemble : 2 elements → returns (v1, v2)
    """

    def __init__(
        self,
        psi_grids: List[np.ndarray],
        phi_g_grids: List[np.ndarray],
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        aggregator: str,
        device,
    ):
        # psi_grids    : list of (G, G, D) np.float32 — obs encoder per member
        # phi_g_grids  : list of (G, G, D) np.float32 — goal encoder per member
        self._psi_grids   = psi_grids
        self._phi_g_grids = phi_g_grids
        self._x_min, self._x_max = x_min, x_max
        self._y_min, self._y_max = y_min, y_max
        self._G          = psi_grids[0].shape[0]
        self._aggregator = aggregator
        self.device      = device

    # ------------------------------------------------------------------
    # No-op compatibility with HILP / HILPJax
    # ------------------------------------------------------------------
    def eval(self):        return self
    def parameters(self):  return iter([])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _xy_to_fidx(self, obs_np: np.ndarray):
        """Map world (x, y) → fractional grid indices (N,) each."""
        G = self._G
        xi = (obs_np[:, 0] - self._x_min) / (self._x_max - self._x_min) * (G - 1)
        yi = (obs_np[:, 1] - self._y_min) / (self._y_max - self._y_min) * (G - 1)
        return xi, yi

    @staticmethod
    def _bilinear(grid: np.ndarray, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
        """Bilinear interpolation.  grid (G,G,D), xi/yi (N,) → (N,D) float32."""
        G = grid.shape[0]
        x0 = np.clip(np.floor(xi).astype(np.int32), 0, G - 2)
        y0 = np.clip(np.floor(yi).astype(np.int32), 0, G - 2)
        x1, y1 = x0 + 1, y0 + 1
        dx = (xi - x0)[:, None]   # (N, 1)
        dy = (yi - y0)[:, None]
        return (grid[x0, y0] * (1 - dx) * (1 - dy) +
                grid[x1, y0] * dx       * (1 - dy) +
                grid[x0, y1] * (1 - dx) * dy       +
                grid[x1, y1] * dx       * dy).astype(np.float32)

    def _aggregate(self, psi: np.ndarray, phi_g: np.ndarray) -> np.ndarray:
        """Compute scalar value from embeddings → (N,) float32."""
        if self._aggregator == "neg_l2":
            dist_sq = ((psi - phi_g) ** 2).sum(axis=-1)
            return -np.sqrt(np.maximum(dist_sq, 1e-6)).astype(np.float32)
        else:  # inner_prod
            return (psi * phi_g).sum(axis=-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Public interface (matching HILP / HILPJax)
    # ------------------------------------------------------------------
    def value(self, obs_t, goal_t):
        """value(obs, goal) → (v1, v2) — matches HILP/HILPJax interface."""
        import torch
        obs_np  = obs_t.detach().cpu().numpy().astype(np.float32)
        goal_np = goal_t.detach().cpu().numpy().astype(np.float32)
        xi_s, yi_s = self._xy_to_fidx(obs_np)
        xi_g, yi_g = self._xy_to_fidx(goal_np)

        vs = []
        for psi_grid, phi_g_grid in zip(self._psi_grids, self._phi_g_grids):
            psi   = self._bilinear(psi_grid,   xi_s, yi_s)
            phi_g = self._bilinear(phi_g_grid, xi_g, yi_g)
            vs.append(torch.from_numpy(self._aggregate(psi, phi_g)).to(self.device))

        v0 = vs[0]
        v1 = vs[1] if len(vs) > 1 else vs[0]
        return v0, v1

    def compute_values_np(self, obs_np: np.ndarray, goal_np: np.ndarray) -> np.ndarray:
        """Compute V(obs, goal) → (N,) float32 numpy (ensemble member 0)."""
        N = obs_np.shape[0]
        obs_np   = obs_np.astype(np.float32)
        goal_rep = np.broadcast_to(goal_np[:1], (N, goal_np.shape[-1])).copy()
        xi_s, yi_s = self._xy_to_fidx(obs_np)
        xi_g, yi_g = self._xy_to_fidx(goal_rep)
        psi   = self._bilinear(self._psi_grids[0],   xi_s, yi_s)
        phi_g = self._bilinear(self._phi_g_grids[0], xi_g, yi_g)
        return self._aggregate(psi, phi_g)

    def compute_grads(
        self,
        obs_np: np.ndarray,
        goal_np: np.ndarray,
        eps: float = 0.5,
    ) -> np.ndarray:
        """∂V/∂(x,y) via finite differences on the memoized grid → (N, 2) float32."""
        N = obs_np.shape[0]
        obs_np = obs_np.astype(np.float32)
        # Compute phi_g once — goal is fixed across all N observations
        goal_rep = np.broadcast_to(goal_np[:1], (N, goal_np.shape[-1])).copy().astype(np.float32)
        xi_g, yi_g = self._xy_to_fidx(goal_rep)
        phi_g = self._bilinear(self._phi_g_grids[0], xi_g, yi_g)  # (N, D)

        grads = np.zeros((N, 2), dtype=np.float32)
        for dim in range(2):
            obs_p = obs_np.copy(); obs_p[:, dim] += eps
            obs_m = obs_np.copy(); obs_m[:, dim] -= eps
            psi_p = self._bilinear(self._psi_grids[0], *self._xy_to_fidx(obs_p))
            psi_m = self._bilinear(self._psi_grids[0], *self._xy_to_fidx(obs_m))
            grads[:, dim] = (self._aggregate(psi_p, phi_g) - self._aggregate(psi_m, phi_g)) / (2 * eps)
        return grads


# ---------------------------------------------------------------------------
# Raw model loader
# ---------------------------------------------------------------------------

def load_raw_hilp_model(
    checkpoint_path: str,
    device,
    hilp_obs_dim: int = 29,
    hilp_skill_dim: int = 256,
):
    """Load raw HILP (.pt) or HILPJax (.pkl) from checkpoint.

    Returns a model with interface:
        .value(obs_t, goal_t) → (v1, v2)
        .eval() / .parameters()
        .compute_grads(obs_np, goal_np) [HILPJax only]
    """
    import sys

    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if checkpoint_path.endswith(".pkl"):
        import jax
        _jax_cache = os.path.expanduser("~/.jax_cache")
        os.makedirs(_jax_cache, exist_ok=True)
        os.makedirs(os.path.join(_jax_cache, "xla_gpu_per_fusion_autotune_cache_dir"), exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", _jax_cache)
        from td_models.hilp import HILPJax
        model = HILPJax(checkpoint_path, device)
    else:
        from td_models.hilp import HILP
        import torch
        model = HILP(
            obs_dim=hilp_obs_dim,
            skill_dim=hilp_skill_dim,
            device=device,
            value_hidden_dims=(512, 512, 512),
            use_layer_norm=True,
        )
        model.load(checkpoint_path)

    model.eval()
    if hasattr(model, "parameters"):
        for param in model.parameters():
            param.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Grid builder + cache I/O
# ---------------------------------------------------------------------------

def _build_memo_grids(
    hilp_model,
    grid_size: int,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    hilp_obs_dim: int,
    ref_obs: Optional[np.ndarray],
    device,
) -> HILPMemoizedWrapper:
    """Compute encoder grids from a loaded hilp_model and return a wrapper."""
    import torch

    G = grid_size
    xs = np.linspace(x_min, x_max, G)
    ys = np.linspace(y_min, y_max, G)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")   # (G, G)  row=x, col=y
    xi_flat = XX.ravel().astype(np.float32)         # (G*G,)
    yi_flat = YY.ravel().astype(np.float32)

    N = G * G
    if ref_obs is not None:
        obs_flat = np.tile(ref_obs, (N, 1)).astype(np.float32)
    else:
        obs_flat = np.zeros((N, hilp_obs_dim), dtype=np.float32)
    obs_flat[:, 0] = xi_flat
    obs_flat[:, 1] = yi_flat

    is_jax = hasattr(hilp_model, "_agent")

    if is_jax:
        psi_flat   = np.array(hilp_model._agent.get_psi(obs_flat))       # (N, D)
        phi_g_flat = np.array(hilp_model._agent.get_phi_goal(obs_flat))   # (N, D)
        aggregator = hilp_model._aggregator
        psi_grids   = [psi_flat.reshape(G, G, -1).astype(np.float32)]
        phi_g_grids = [phi_g_flat.reshape(G, G, -1).astype(np.float32)]
    else:
        # PyTorch HILP: ensemble phi_net (mlp1, mlp2) — symmetric encoder
        with torch.no_grad():
            obs_t   = torch.from_numpy(obs_flat).float().to(device)
            phi_out = hilp_model.value.phi_net(obs_t)   # → (phi1, phi2)
        aggregator  = "neg_l2"
        psi_grids   = [phi_out[0].cpu().numpy().reshape(G, G, -1).astype(np.float32),
                       phi_out[1].cpu().numpy().reshape(G, G, -1).astype(np.float32)]
        phi_g_grids = psi_grids   # symmetric encoder — same grid for obs and goal

    return HILPMemoizedWrapper(
        psi_grids=psi_grids,
        phi_g_grids=phi_g_grids,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        aggregator=aggregator,
        device=device,
    )


def _load_memo_from_cache(cache_path: str, device) -> HILPMemoizedWrapper:
    """Load a previously saved grid cache and return a wrapper."""
    data = np.load(cache_path, allow_pickle=False)
    psi_grids   = [data["psi_grid_0"]]
    phi_g_grids = [data["phi_g_grid_0"]]
    if "psi_grid_1" in data:
        psi_grids.append(data["psi_grid_1"])
        phi_g_grids.append(data["phi_g_grid_1"])
    return HILPMemoizedWrapper(
        psi_grids=psi_grids,
        phi_g_grids=phi_g_grids,
        x_min=float(data["x_min"]),
        x_max=float(data["x_max"]),
        y_min=float(data["y_min"]),
        y_max=float(data["y_max"]),
        aggregator=str(data["aggregator"]),
        device=device,
    )


def _save_memo_cache(wrapper: HILPMemoizedWrapper, cache_path: str) -> None:
    """Persist grid arrays to an .npz file."""
    save_dict = {
        "psi_grid_0":   wrapper._psi_grids[0],
        "phi_g_grid_0": wrapper._phi_g_grids[0],
        "aggregator":   np.array(wrapper._aggregator),
        "x_min": np.float32(wrapper._x_min),
        "x_max": np.float32(wrapper._x_max),
        "y_min": np.float32(wrapper._y_min),
        "y_max": np.float32(wrapper._y_max),
    }
    if len(wrapper._psi_grids) > 1:
        save_dict["psi_grid_1"]   = wrapper._psi_grids[1]
        save_dict["phi_g_grid_1"] = wrapper._phi_g_grids[1]
    np.savez(cache_path, **save_dict)


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------

def get_hilp_fn(
    checkpoint_path: str,
    device,
    *,
    use_memoization: bool = False,
    # raw model params (ignored when cache hit)
    hilp_obs_dim: int = 29,
    hilp_skill_dim: int = 256,
    # memoization params
    grid_size: int = 100,
    x_min: float = 0.0, x_max: float = 1.0,
    y_min: float = 0.0, y_max: float = 1.0,
    ref_obs: Optional[np.ndarray] = None,
):
    """Return a HILP value function (raw or memoized wrapper).

    Args:
        checkpoint_path : Path to .pkl (HILPJax) or .pt (HILP PyTorch) file.
        device          : torch device.
        use_memoization : If True, return HILPMemoizedWrapper (fast bilinear lookup).
        hilp_obs_dim    : obs dim for legacy .pt checkpoints.
        hilp_skill_dim  : skill dim for legacy .pt checkpoints.
        grid_size       : G — grid resolution per axis (G×G total points).
        x_min/x_max     : World x bounds for the encoder grid.
        y_min/y_max     : World y bounds for the encoder grid.
        ref_obs         : (hilp_obs_dim,) reference obs for non-spatial dims.
                          If None, non-spatial dims are set to zero.

    Returns:
        Raw HILP / HILPJax model  OR  HILPMemoizedWrapper.
        Both expose .value(obs_t, goal_t) → (v1, v2) and .eval() / .parameters().
    """
    if not use_memoization:
        return load_raw_hilp_model(checkpoint_path, device, hilp_obs_dim, hilp_skill_dim)

    # Derive cache path: <ckpt_dir>/<ckpt_stem>_memo_G<G>.npz
    ckpt_dir  = os.path.dirname(os.path.abspath(checkpoint_path))
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    cache_path = os.path.join(ckpt_dir, f"{ckpt_stem}_memo_G{grid_size}.npz")

    if os.path.exists(cache_path):
        print(f"[HILP memo] Loading grid cache  G={grid_size}  {cache_path}", flush=True)
        return _load_memo_from_cache(cache_path, device)

    # Cache miss — build from loaded model
    print(f"[HILP memo] Cache not found — building G={grid_size} grid → {cache_path}", flush=True)
    hilp_model = load_raw_hilp_model(checkpoint_path, device, hilp_obs_dim, hilp_skill_dim)
    wrapper = _build_memo_grids(
        hilp_model,
        grid_size=grid_size,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        hilp_obs_dim=hilp_obs_dim,
        ref_obs=ref_obs,
        device=device,
    )
    _save_memo_cache(wrapper, cache_path)
    print(f"[HILP memo] Saved  G={grid_size}  D={wrapper._psi_grids[0].shape[-1]}"
          f"  ensemble={len(wrapper._psi_grids)}  aggregator={wrapper._aggregator}", flush=True)
    return wrapper
