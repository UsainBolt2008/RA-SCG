from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


VIEW_NAMES = (
    "identity",
    "global_075",
    "global_125",
    "semantic_zoom_075",
    "semantic_zoom_050",
)


def _mean_abs_normalize(g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = g.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return g / denom


def _normalize_map01(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [B,H,W], got {tuple(x.shape)}")
    lo = x.flatten(1).min(dim=1).values[:, None, None]
    hi = x.flatten(1).max(dim=1).values[:, None, None]
    return (x - lo) / (hi - lo).clamp_min(1e-8)


def weighted_semantic_centers(fused_map_7x7: torch.Tensor) -> torch.Tensor:
    """Return Bx2 normalized (x,y) crop centers, clamped to [-1,1]."""
    if fused_map_7x7.ndim != 3 or tuple(fused_map_7x7.shape[-2:]) != (7, 7):
        raise ValueError(f"Expected [B,7,7], got {tuple(fused_map_7x7.shape)}")
    m = fused_map_7x7.float().clamp_min(0)
    mass = m.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-8)
    coords = torch.linspace(-1.0, 1.0, 7, device=m.device, dtype=m.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    cx = (m * xx).flatten(1).sum(dim=1, keepdim=True) / mass
    cy = (m * yy).flatten(1).sum(dim=1, keepdim=True) / mass
    return torch.cat([cx, cy], dim=1).clamp(-1.0, 1.0)


def semantic_zoom(x: torch.Tensor, centers_xy: torch.Tensor, scale: float) -> torch.Tensor:
    """Differentiable crop centered on semantic centroid, resized back to input HW."""
    if not (0.0 < scale <= 1.0):
        raise ValueError(scale)
    b = x.shape[0]
    if tuple(centers_xy.shape) != (b, 2):
        raise ValueError((tuple(centers_xy.shape), b))
    max_shift = 1.0 - float(scale)
    centers = centers_xy.clamp(-max_shift, max_shift)
    theta = torch.zeros((b, 2, 3), device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = float(scale)
    theta[:, 1, 1] = float(scale)
    theta[:, 0, 2] = centers[:, 0]
    theta[:, 1, 2] = centers[:, 1]
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def global_resize_roundtrip(x: torch.Tensor, scale: float) -> torch.Tensor:
    h, w = x.shape[-2:]
    h2 = max(8, int(round(h * float(scale))))
    w2 = max(8, int(round(w * float(scale))))
    y = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
    return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


def build_views(x: torch.Tensor, centers_xy: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "identity": x,
        "global_075": global_resize_roundtrip(x, 0.75),
        "global_125": global_resize_roundtrip(x, 1.25),
        "semantic_zoom_075": semantic_zoom(x, centers_xy, 0.75),
        "semantic_zoom_050": semantic_zoom(x, centers_xy, 0.50),
    }
