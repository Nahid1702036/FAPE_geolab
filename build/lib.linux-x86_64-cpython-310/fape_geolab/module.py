"""
fape_geolab.module
nn.Module wrapper, autograd Function, and mesh utilities.
"""

from __future__ import annotations
import importlib
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional


# ── Lazy extension loader ─────────────────────────────────────────────────────

def _load_ext():
    try:
        return importlib.import_module("fape_geolab._C")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "fape_geolab._C extension not built.\n"
            "Run:  pip install <path-to-fape_geolab>"
        ) from exc


# ── Mesh utilities ────────────────────────────────────────────────────────────

def compute_vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """Area-weighted, L2-normalised per-vertex normals. (V,3) → (V,3)"""
    v0, v1, v2 = vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]
    fn = torch.cross(v1-v0, v2-v0, dim=1)
    n  = torch.zeros_like(vertices)
    for k in range(3):
        n.scatter_add_(0, faces[:,k].long().unsqueeze(1).expand_as(fn), fn)
    return n / n.norm(dim=1, keepdim=True).clamp(min=1e-8)


def build_edges(faces: Tensor) -> Tensor:
    """Unique undirected edges from (F,3) int32 face tensor → (E,2) int32."""
    import numpy as np
    e = torch.cat([
        torch.stack([faces[:,0],faces[:,1]],1),
        torch.stack([faces[:,1],faces[:,2]],1),
        torch.stack([faces[:,2],faces[:,0]],1),
    ], dim=0)
    e, _ = e.sort(dim=1)
    e_np  = np.unique(e.cpu().numpy(), axis=0)
    return torch.from_numpy(e_np).to(dtype=torch.int32, device=faces.device)


# ── autograd Function ─────────────────────────────────────────────────────────

class _SurgicalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, faces, normals, edges,
                all_cams, hero_cams, tau, sigma_s, sigma_n, epsilon):
        out_v, out_val = _load_ext().run_surgical_geometry_pass(
            vertices, faces, normals, edges,
            all_cams, hero_cams, tau, sigma_s, sigma_n, epsilon)
        ctx.save_for_backward(out_val)
        return out_v, out_val

    @staticmethod
    def backward(ctx, grad_v, grad_val):
        (out_val,) = ctx.saved_tensors
        # Straight-through estimator modulated by validity
        return grad_v * out_val.unsqueeze(-1), \
               None, None, None, None, None, None, None, None, None


# ── Public nn.Module ──────────────────────────────────────────────────────────

class FapeSurgicalPass(nn.Module):
    """
    FAPE Surgical Geometry Pass.

    Parameters
    ----------
    tau     : Gumbel-Softmax temperature          (default 0.1)
    sigma_s : Bilateral spatial variance          (default 0.05)
    sigma_n : Bilateral normal variance           (default 0.15)
    epsilon : Leaky gate floor                    (default 1e-4)
    """

    def __init__(self, tau=0.1, sigma_s=0.05, sigma_n=0.15, epsilon=1e-4):
        super().__init__()
        self.tau, self.sigma_s, self.sigma_n, self.epsilon = \
            tau, sigma_s, sigma_n, epsilon

    def forward(
        self,
        vertices:       Tensor,
        faces:          Tensor,
        all_cam_mats:   Tensor,
        hero_cam_mats:  Tensor,
        vertex_normals: Optional[Tensor] = None,
        edges:          Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        out_vertices : (V,3) float32 — smoothed positions
        out_validity : (V,)  float32 — per-vertex validity in (0,1]
        """
        if vertex_normals is None:
            with torch.no_grad():
                vertex_normals = compute_vertex_normals(vertices.detach(), faces)
        if edges is None:
            with torch.no_grad():
                edges = build_edges(faces)

        return _SurgicalFn.apply(
            vertices, faces, vertex_normals, edges,
            all_cam_mats, hero_cam_mats,
            self.tau, self.sigma_s, self.sigma_n, self.epsilon,
        )
