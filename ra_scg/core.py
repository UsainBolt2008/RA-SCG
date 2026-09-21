from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .views import VIEW_NAMES, build_views, weighted_semantic_centers

SELECTED_LAYERS = (2, 5, 8, 11)
GLOBAL_3V = ("identity", "global_075", "global_125")


def mean_abs_normalize(g, eps=1e-12):
    return g / g.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)


def normalize_map01_batch(x):
    flat = x.flatten(-2)
    lo = flat.min(-1).values[..., None, None]
    hi = flat.max(-1).values[..., None, None]
    return (x - lo) / (hi - lo).clamp_min(1e-8)


def l1_mass_normalize(x, eps=1e-12):
    x = x.clamp_min(0)
    return x / x.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)


def _first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    raise TypeError(type(output))


def _as_bld(x, expected_tokens):
    if x.ndim != 3:
        raise RuntimeError(f"Expected 3D token tensor, got {tuple(x.shape)}")
    if x.shape[1] == expected_tokens:
        return x
    if x.shape[0] == expected_tokens:
        return x.permute(1, 0, 2).contiguous()
    raise RuntimeError(f"Cannot identify token dimension: {tuple(x.shape)}")


def capture_image_and_patch_features(bridge, raw_view):
    visual = bridge.model.visual
    blocks = visual.transformer.resblocks
    expected_tokens = int(visual.positional_embedding.shape[0])
    if expected_tokens != 50:
        raise RuntimeError(f"Expected ViT-B/32 50 tokens, got {expected_tokens}")
    captured, handles = {}, []

    def mk(idx):
        def hook(_m, _i, out):
            captured[idx] = _first_tensor(out)
        return hook

    for idx in SELECTED_LAYERS:
        handles.append(blocks[idx].register_forward_hook(mk(idx)))
    try:
        image_feat = bridge.inference_image(bridge.normalization(raw_view))["image_feat"].float()
    finally:
        for h in handles:
            h.remove()

    projected = {}
    for idx in SELECTED_LAYERS:
        if idx not in captured:
            raise RuntimeError(f"Missing patch hook layer {idx}")
        tokens = _as_bld(captured[idx], expected_tokens)
        patches = visual.ln_post(tokens[:, 1:, :].float())
        if visual.proj is not None:
            patches = patches @ visual.proj
        projected[idx] = F.normalize(patches.float(), dim=-1)
    return F.normalize(image_feat.float(), dim=-1), projected


def _warp_maps_like_view(maps, centers_xy, view_name):
    if view_name in ("identity", "global_075", "global_125"):
        return maps
    from .views import semantic_zoom
    if view_name == "semantic_zoom_075":
        return semantic_zoom(maps, centers_xy, 0.75)
    if view_name == "semantic_zoom_050":
        return semantic_zoom(maps, centers_xy, 0.50)
    raise KeyError(view_name)


@dataclass
class RelationViewSupport:
    node_support: torch.Tensor
    confidence: torch.Tensor


def build_relation_view_supports(entity_maps, spatial_maps, relation_valid, fmap, centers_xy, active_views):
    if entity_maps.ndim != 4 or entity_maps.shape[1:] != (5, 7, 7):
        raise ValueError(tuple(entity_maps.shape))
    if spatial_maps.shape != entity_maps.shape or relation_valid.shape != entity_maps.shape[:2]:
        raise ValueError("relation shape mismatch")
    b = entity_maps.shape[0]
    out = {}
    for view_name in active_views:
        E = normalize_map01_batch(_warp_maps_like_view(entity_maps, centers_xy, view_name))
        S = normalize_map01_batch(_warp_maps_like_view(spatial_maps, centers_xy, view_name))
        Fv = normalize_map01_batch(_warp_maps_like_view(fmap.unsqueeze(1), centers_xy, view_name).squeeze(1))
        Ea = E * (1.0 + Fv[:, None])
        Sa = S * (1.0 + Fv[:, None])
        valid = relation_valid & (Ea.sum((-2, -1)) > 1e-8) & (Sa.sum((-2, -1)) > 1e-8)
        vf = valid.float()
        Eprob = l1_mass_normalize(Ea) * vf[:, :, None, None]
        Sprob = l1_mass_normalize(Sa) * vf[:, :, None, None]
        node = (Eprob + Sprob).reshape(b, 5, 49).sum(1)
        count = vf.sum(1)
        node = node / node.sum(-1, keepdim=True).clamp_min(1e-12)
        node = torch.where((count > 0)[:, None], node, torch.zeros_like(node))
        out[view_name] = RelationViewSupport(node_support=node, confidence=count / 5.0)
    return out


@dataclass
class GradientBundle:
    gradients: torch.Tensor
    pairwise_cosine: torch.Tensor
    consensus: torch.Tensor
    semantic_quality: torch.Tensor
    weights: torch.Tensor
    aggregate: torch.Tensor
    relation_scale_by_view: torch.Tensor


class RASCGAblationAttacker:
    """RA-SCG optimization core.

    Final RA-SCG:
      - five views
      - confidence-weighted per-view relation residual
      - directional consensus d_i
      - semantic-quality term q_i
      - weights w_i proportional to d_i*q_i

    Every ablation toggles exactly one paper-facing component while preserving the
    frozen outer protocol (epsilon, step size, 10 updates, batch size handled upstream).
    """

    def __init__(
        self,
        *,
        use_relation=True,
        use_relation_confidence=True,
        use_directional_consensus=True,
        use_semantic_quality=True,
        active_views=VIEW_NAMES,
        relation_lambda=0.5,
        epsilon=2/255,
        step_size=0.5/255,
        steps=10,
        alpha=3.0,
    ):
        self.use_relation = bool(use_relation)
        self.use_relation_confidence = bool(use_relation_confidence)
        self.use_directional_consensus = bool(use_directional_consensus)
        self.use_semantic_quality = bool(use_semantic_quality)
        self.active_views = tuple(active_views)
        if not self.active_views or any(v not in VIEW_NAMES for v in self.active_views):
            raise ValueError(self.active_views)
        self.relation_lambda = float(relation_lambda)
        self.epsilon = float(epsilon)
        self.step_size = float(step_size)
        self.steps = int(steps)
        self.alpha = float(alpha)
        if abs(self.relation_lambda - 0.5) > 1e-12:
            raise ValueError("relation lambda must remain frozen at 0.5")
        if abs(self.epsilon - 2/255) > 1e-12 or abs(self.step_size - 0.5/255) > 1e-12 or self.steps != 10 or abs(self.alpha - 3.0) > 1e-12:
            raise ValueError("frozen protocol mismatch")

    @torch.no_grad()
    def clean_view_cache(self, bridge, clean, centers):
        views = build_views(clean, centers)
        feats, patches = [], []
        for name in self.active_views:
            if self.use_relation:
                f, p = capture_image_and_patch_features(bridge, views[name])
                patches.append({k: v.detach() for k, v in p.items()})
            else:
                f = bridge.inference_image(bridge.normalization(views[name]))["image_feat"].float()
                f = F.normalize(f, dim=-1)
                patches.append(None)
            feats.append(f)
        return torch.stack(feats, 1).detach(), patches

    def semantic_quality(self, grads, R, C, fmap):
        b, v, _, h, w = grads.shape
        e = F.adaptive_avg_pool2d(
            grads.abs().mean(2).reshape(b*v, 1, h, w), (7, 7)
        ).reshape(b, v, 7, 7)
        e = l1_mass_normalize(e)
        rn = normalize_map01_batch(R)[:, None]
        cn = normalize_map01_batch(C)[:, None]
        fn = normalize_map01_batch(fmap)[:, None]
        return torch.pow(
            (e * rn).sum((2, 3)).clamp_min(1e-12)
            * (e * cn).sum((2, 3)).clamp_min(1e-12)
            * (e * fn).sum((2, 3)).clamp_min(1e-12),
            1/3,
        )

    def _node_loss_raw(self, adv_patches, clean_patches, support):
        vals = []
        n = support.node_support
        for layer in SELECTED_LAYERS:
            pa = (n[:, :, None] * adv_patches[layer]).sum(1)
            pc = (n[:, :, None] * clean_patches[layer].detach()).sum(1)
            vals.append(1.0 - (F.normalize(pa, dim=-1) * F.normalize(pc, dim=-1)).sum(-1))
        return torch.stack(vals, 1).mean(1)

    def gradient_bundle(self, bridge, adv, centers, clean_feat, clean_patches, text_feat, R, C, fmap, supports):
        raw, scales = [], []
        text_feat = F.normalize(text_feat.detach().float(), dim=-1)
        batch_size = adv.shape[0]

        for vi, name in enumerate(self.active_views):
            base = adv.detach().clone().requires_grad_(True)
            view = build_views(base, centers)[name]
            if self.use_relation:
                feat, patches = capture_image_and_patch_features(bridge, view)
            else:
                feat = bridge.inference_image(bridge.normalization(view))["image_feat"].float()
                feat = F.normalize(feat, dim=-1)
                patches = None
            image_div = 1.0 - (feat * clean_feat[:, vi]).sum(-1)
            text_sim = torch.einsum("bd,bkd->bk", feat, text_feat).mean(1)
            base_per = image_div + self.alpha * (1.0 - text_sim)

            if self.use_relation:
                rel_raw = self._node_loss_raw(patches, clean_patches[vi], supports[name])
                g0 = torch.autograd.grad(base_per.mean(), base, retain_graph=True)[0].detach()
                gr = torch.autograd.grad(rel_raw.mean(), base)[0].detach()
                g0n = mean_abs_normalize(g0)
                rel_ma = gr.abs().mean((1, 2, 3), keepdim=True)
                grn = torch.where(rel_ma > 1e-14, gr / rel_ma.clamp_min(1e-12), torch.zeros_like(gr))
                if self.use_relation_confidence:
                    confidence = supports[name].confidence
                else:
                    confidence = torch.ones((batch_size,), device=adv.device, dtype=g0n.dtype)
                scale = self.relation_lambda * confidence
                g = g0n + scale[:, None, None, None] * grn
            else:
                g0 = torch.autograd.grad(base_per.mean(), base)[0].detach()
                g = mean_abs_normalize(g0)
                scale = torch.zeros((batch_size,), device=adv.device, dtype=g.dtype)

            raw.append(g)
            scales.append(scale)

        raw = torch.stack(raw, 1)
        gn = torch.stack([mean_abs_normalize(raw[:, i]) for i in range(raw.shape[1])], 1)
        v = len(self.active_views)
        if self.use_directional_consensus:
            flat = F.normalize(gn.flatten(2), dim=-1)
            pair = torch.einsum("bvi,bwi->bvw", flat, flat).clamp(-1, 1)
            eye = torch.eye(v, device=pair.device, dtype=torch.bool)[None]
            if v > 1:
                cons = ((pair.masked_fill(eye, 0).sum(2) / (v - 1) + 1) * 0.5).clamp(0, 1)
            else:
                cons = torch.ones((gn.shape[0], 1), device=gn.device)
        else:
            pair = torch.zeros((gn.shape[0], v, v), device=gn.device, dtype=gn.dtype)
            cons = torch.ones((gn.shape[0], v), device=gn.device, dtype=gn.dtype)

        if self.use_semantic_quality:
            sq = self.semantic_quality(gn, R, C, fmap)
        else:
            sq = torch.ones((gn.shape[0], v), device=gn.device, dtype=gn.dtype)

        d_used = cons
        q_used = sq
        rw = d_used.clamp_min(1e-6) * q_used.clamp_min(1e-6)
        w = rw / rw.sum(1, keepdim=True).clamp_min(1e-12)
        agg = mean_abs_normalize((w[:, :, None, None, None] * gn).sum(1))
        return GradientBundle(gn, pair, cons, sq, w, agg, torch.stack(scales, 1))

    def run(self, bridge, clean_images, text_feat, R, C, fmap, entity_maps, spatial_maps, relation_valid):
        centers = weighted_semantic_centers(fmap)
        clean_feat, clean_patches = self.clean_view_cache(bridge, clean_images, centers)
        supports = (
            build_relation_view_supports(entity_maps, spatial_maps, relation_valid, fmap, centers, self.active_views)
            if self.use_relation else None
        )
        adv = (clean_images + torch.empty_like(clean_images).uniform_(-self.epsilon, self.epsilon)).clamp(0, 1).detach()
        momentum = torch.zeros_like(clean_images)
        hist = []

        for step in range(self.steps):
            b = self.gradient_bundle(bridge, adv, centers, clean_feat, clean_patches, text_feat, R, C, fmap, supports)
            momentum = momentum + b.aggregate
            adv = adv.detach() + self.step_size * mean_abs_normalize(momentum).sign()
            adv = (clean_images + (adv - clean_images).clamp(-self.epsilon, self.epsilon)).clamp(0, 1).detach()

            conf = (
                torch.stack([supports[n].confidence for n in self.active_views], 1)
                if supports is not None
                else torch.zeros((clean_images.shape[0], len(self.active_views)), device=clean_images.device)
            )
            hist.append({
                "step": step,
                "relation_confidence_mean": float(conf.mean()),
                "effective_relation_scale_mean": float(b.relation_scale_by_view.mean()),
                "directional_consensus_mean": float(b.consensus.mean()),
                "semantic_quality_mean": float(b.semantic_quality.mean()),
                "view_weight_entropy_mean": float((-(b.weights.clamp_min(1e-12) * b.weights.clamp_min(1e-12).log()).sum(1)).mean()),
                "view_weight_max_mean": float(b.weights.max(1).values.mean()),
                "extra_relation_backward_evals": len(self.active_views) if self.use_relation else 0,
                # Small per-sample/per-view traces are useful for Fig. 1 reconstruction.
                "relation_confidence_by_sample_view": conf.detach().cpu().tolist(),
                "effective_relation_scale_by_sample_view": b.relation_scale_by_view.detach().cpu().tolist(),
                "directional_consensus_by_sample_view": b.consensus.detach().cpu().tolist(),
                "semantic_quality_by_sample_view": b.semantic_quality.detach().cpu().tolist(),
                "view_weights_by_sample_view": b.weights.detach().cpu().tolist(),
            })
        return adv, hist, centers.detach()
