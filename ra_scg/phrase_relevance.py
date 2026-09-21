
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class Phrase:
    text: str
    kind: str
    caption_index: int

def _dedup(items: Iterable[Phrase]) -> list[Phrase]:
    seen, out = set(), []
    for p in items:
        key = (p.text.casefold(), p.kind, int(p.caption_index))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out

def extract_phrase_views(caption: str, caption_index: int) -> list[Phrase]:
    import re, nltk
    tokens = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?", str(caption))
    if not tokens:
        return [Phrase(str(caption).strip(), "fallback", int(caption_index))]
    tags = nltk.pos_tag(tokens)
    out = []

    i = 0
    while i < len(tags):
        start = i
        while i < len(tags) and tags[i][1].startswith("JJ"):
            i += 1
        noun_start = i
        while i < len(tags) and tags[i][1].startswith("NN"):
            i += 1
        if i > noun_start:
            text = " ".join(tokens[start:i])
            out.append(Phrase(text, "entity", int(caption_index)))
            if noun_start > start:
                out.append(Phrase(text, "attribute", int(caption_index)))
            continue
        i = start + 1

    for i, (_, tag) in enumerate(tags):
        if tag.startswith("VB"):
            lo, hi = max(0, i - 1), min(len(tokens), i + 4)
            for j in range(i + 1, hi):
                if tags[j][1].startswith("NN"):
                    hi = j + 1
                    break
            out.append(Phrase(" ".join(tokens[lo:hi]), "action", int(caption_index)))

    spatial = {
        "in","on","near","beside","between","around","along","through",
        "over","under","above","below","behind","inside","outside","across",
        "within","by","next"
    }
    for i, (token, tag) in enumerate(tags):
        if token.casefold() not in spatial:
            continue
        hi = min(len(tokens), i + 5)
        for j in range(i + 1, hi):
            if tags[j][1].startswith("NN"):
                hi = j + 1
                break
        if hi - i >= 2:
            out.append(Phrase(" ".join(tokens[i:hi]), "spatial", int(caption_index)))

    out = _dedup(out)
    return out or [Phrase(" ".join(tokens), "fallback", int(caption_index))]

def _first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    raise TypeError(type(output))

def _as_bld(x: torch.Tensor, expected_tokens: int) -> torch.Tensor:
    if x.ndim != 3:
        raise RuntimeError(f"Expected 3D token tensor, got {tuple(x.shape)}")
    if x.shape[1] == expected_tokens:
        return x
    if x.shape[0] == expected_tokens:
        return x.permute(1,0,2).contiguous()
    raise RuntimeError(
        f"Cannot identify token dimension: shape={tuple(x.shape)}, "
        f"expected_tokens={expected_tokens}"
    )

class RemoteCLIPPatchPhraseRelevance:
    def __init__(self, bridge, selected_layers=(2,5,8,11)):
        self.bridge = bridge
        self.visual = bridge.model.visual
        self.selected_layers = tuple(int(x) for x in selected_layers)
        blocks = self.visual.transformer.resblocks
        if len(blocks) != 12:
            raise RuntimeError(f"Expected 12 blocks, got {len(blocks)}")
        self.blocks = blocks
        self.expected_tokens = int(self.visual.positional_embedding.shape[0])
        if self.expected_tokens != 50:
            raise RuntimeError(f"Expected 50 tokens, got {self.expected_tokens}")
        self.patch_count = 49
        self.grid_size = 7

    @torch.no_grad()
    def encode_phrases(self, phrases):
        return F.normalize(
            self.bridge.encode_text_strings([p.text for p in phrases]).float(),
            dim=-1
        )

    def capture(self, raw_images):
        captured, handles = {}, []
        def mk(idx):
            def hook(_m,_i,o):
                captured[idx] = _first_tensor(o)
            return hook
        for idx in self.selected_layers:
            handles.append(self.blocks[idx].register_forward_hook(mk(idx)))
        try:
            with torch.no_grad():
                image_feat = self.bridge.inference_image(
                    self.bridge.normalization(raw_images)
                )["image_feat"]
        finally:
            for h in handles:
                h.remove()
        projected = {}
        for idx in self.selected_layers:
            tokens = _as_bld(captured[idx], self.expected_tokens)
            patches = self.visual.ln_post(tokens[:,1:,:].float())
            if self.visual.proj is not None:
                patches = patches @ self.visual.proj
            projected[idx] = F.normalize(patches.float(), dim=-1)
        return projected, F.normalize(image_feat.float(), dim=-1)

    @staticmethod
    def relative(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    @torch.no_grad()
    def build_map(self, raw_image, phrases):
        if raw_image.ndim == 3:
            raw_image = raw_image.unsqueeze(0)
        if raw_image.shape[0] != 1:
            raise RuntimeError("build_map expects B=1")
        layer_patch, image_feat = self.capture(raw_image)
        phrase_feat = self.encode_phrases(phrases)

        layer_maps = {}
        type_pool = defaultdict(list)
        for layer, patch in layer_patch.items():
            sim = patch[0] @ phrase_feat.T
            pmaps = []
            for j,p in enumerate(phrases):
                m = self.relative(sim[:,j])
                pmaps.append(m)
                type_pool[p.kind].append(m)
            layer_maps[layer] = torch.stack(pmaps).mean(0).reshape(7,7)

        type_maps = {
            kind: torch.stack(ms).mean(0).reshape(7,7)
            for kind, ms in type_pool.items()
        }
        type_balanced = torch.stack(list(type_maps.values())).mean(0)
        layer_consensus = torch.stack(list(layer_maps.values())).mean(0)
        fused = self.relative(
            (0.5 * type_balanced + 0.5 * layer_consensus).flatten()
        ).reshape(7,7)

        return {
            "map": fused.cpu(),
            "per_layer": {int(k):v.cpu() for k,v in layer_maps.items()},
            "per_type": {str(k):v.cpu() for k,v in type_maps.items()},
            "image_feature": image_feat[0].cpu(),
            "num_tokens": 50,
            "num_patches": 49,
            "grid_size": 7,
        }

def topk_patch_coordinates(m: torch.Tensor, k=5):
    values, indices = torch.topk(m.flatten(), k=min(k,m.numel()))
    out = []
    for v,idx in zip(values.tolist(), indices.tolist()):
        out.append({
            "row": int(idx // m.shape[1]),
            "col": int(idx % m.shape[1]),
            "score": float(v),
        })
    return out
