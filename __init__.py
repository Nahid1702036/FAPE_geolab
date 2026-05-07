"""
fape_geolab
===========
PyTorch CUDA extension for FAPE Surgical Geometry.

Quick start
-----------
>>> from fape_geolab import FapeSurgicalPass
>>> import torch
>>> passer = FapeSurgicalPass()
>>> out_v, validity = passer(vertices, faces, all_cams, hero_cams)
"""

from .module import FapeSurgicalPass, build_edges, compute_vertex_normals

__version__ = "1.0.0"
__all__ = ["FapeSurgicalPass", "build_edges", "compute_vertex_normals"]
