from __future__ import annotations

import torch


def _normalize_01(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    lo = x.min()
    hi = x.max()
    if float(hi - lo) < 1e-8:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def _normalized_stack(
    maps: list[torch.Tensor],
) -> torch.Tensor:
    if not maps:
        raise RuntimeError("No maps supplied")
    return torch.stack(
        [_normalize_01(x) for x in maps],
        dim=0,
    )


def consistency_from_maps(
    maps: list[torch.Tensor],
) -> torch.Tensor:
    """
    High score means the patch response is consistent across maps.
    """
    if len(maps) == 1:
        return torch.ones_like(maps[0], dtype=torch.float32)

    stack = _normalized_stack(maps)
    std = stack.std(dim=0, unbiased=False)

    # Low variance => high consistency.
    return 1.0 - _normalize_01(std)


def build_semantic_consistency(
    relevance_result: dict,
) -> dict:
    """
    Input is the validated Phase5A build_map() output.

    Returns:
      relevance: Phase5A fused relevance
      layer_consistency
      view_consistency
      consistency
      fused_preview
    """
    relevance = _normalize_01(
        torch.as_tensor(
            relevance_result["map"],
            dtype=torch.float32,
        )
    )

    layer_maps = [
        torch.as_tensor(v, dtype=torch.float32)
        for _, v in sorted(
            relevance_result["per_layer"].items(),
            key=lambda kv: int(kv[0]),
        )
    ]

    view_maps = [
        torch.as_tensor(v, dtype=torch.float32)
        for _, v in sorted(
            relevance_result["per_type"].items(),
            key=lambda kv: str(kv[0]),
        )
    ]

    layer_consistency = consistency_from_maps(
        layer_maps
    )

    if len(view_maps) >= 2:
        view_consistency = consistency_from_maps(
            view_maps
        )
    else:
        view_consistency = torch.ones_like(
            relevance
        )

    consistency = _normalize_01(
        0.5 * layer_consistency
        + 0.5 * view_consistency
    )

    # Parameter-free preview fusion:
    # geometric mean rewards patches that are both relevant and consistent.
    fused_preview = _normalize_01(
        torch.sqrt(
            torch.clamp(relevance, 0, 1)
            * torch.clamp(consistency, 0, 1)
            + 1e-12
        )
    )

    return {
        "relevance": relevance,
        "layer_consistency": _normalize_01(
            layer_consistency
        ),
        "view_consistency": _normalize_01(
            view_consistency
        ),
        "consistency": consistency,
        "fused_preview": fused_preview,
        "num_layer_maps": len(layer_maps),
        "num_view_maps": len(view_maps),
    }
