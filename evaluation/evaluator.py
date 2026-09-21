from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import torch

from .fixed_gallery import (
    joint_i2t_fixed,
    joint_t2i_fixed,
    joint_i2t_fixed_matrix,
    joint_t2i_fixed_matrix,
)
from .metrics import summarize_asr

EXPECTED_VIEWS = [
    "identity", "global_075", "global_125", "semantic_zoom_075", "semantic_zoom_050"
]


def _tensor_list(x) -> list[int]:
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.tolist()]
    return [int(v) for v in x]


def validate_cache(cache: dict[str, Any], n_images: int, n_captions: int):
    required = {
        "image_features", "text_features", "i2t_ranks", "t2i_ranks",
        "caption_to_image", "positive_caption_indices",
    }
    missing = sorted(required - set(cache))
    if missing:
        raise KeyError(f"clean cache missing keys: {missing}")
    if int(cache["image_features"].shape[0]) != n_images:
        raise RuntimeError("clean cache image count mismatch")
    if int(cache["text_features"].shape[0]) != n_captions:
        raise RuntimeError("clean cache caption count mismatch")
    if len(cache["i2t_ranks"]) != n_images or len(cache["t2i_ranks"]) != n_captions:
        raise RuntimeError("clean rank count mismatch")
    cti = _tensor_list(cache["caption_to_image"])
    positives = [[int(v) for v in row] for row in cache["positive_caption_indices"]]
    if len(cti) != n_captions or len(positives) != n_images:
        raise RuntimeError("clean cache mapping cardinality mismatch")
    if any(len(row) != 5 for row in positives):
        raise RuntimeError("expected exactly five official captions per image")
    for i, row in enumerate(positives):
        for c in row:
            if not 0 <= c < n_captions or cti[c] != i:
                raise RuntimeError(f"positive mapping inconsistency image={i} caption={c}")
    return cti, positives


def load_text_records(path: Path, expected_captions: int) -> tuple[list[str], list[str], dict]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("method") != "BA-SP":
        raise RuntimeError(f"expected BA-SP text artifact, got {obj.get('method')!r}")
    records = obj.get("records", [])
    by = {int(r["caption_index"]): r for r in records}
    expected = list(range(expected_captions))
    if len(records) != expected_captions or sorted(by) != expected:
        raise RuntimeError(
            f"BA-SP caption ordering mismatch: records={len(records)}, expected={expected_captions}"
        )
    clean, adv = [], []
    for i in expected:
        record = by[i]
        if "original_caption" not in record or "adversarial_caption" not in record:
            raise KeyError(f"BA-SP record {i} lacks original/adversarial caption")
        clean.append(str(record["original_caption"]))
        adv.append(str(record["adversarial_caption"]))
    return clean, adv, obj


def validate_attack_artifact(
    artifact: dict[str, Any], *, n_images: int, expected_dataset: str, expected_source: str
) -> torch.Tensor:
    if str(artifact.get("method")) != "RA-SCG":
        raise RuntimeError(f"expected RA-SCG image artifact, got {artifact.get('method')!r}")
    if str(artifact.get("dataset")).lower() != expected_dataset.lower():
        raise RuntimeError("attack dataset mismatch")
    if str(artifact.get("surrogate_key")) != expected_source:
        raise RuntimeError("attack source mismatch")
    required_protocol = {
        "target_queries": 0,
        "candidate_generation_during_attack": False,
        "steps": 10,
    }
    for key, expected in required_protocol.items():
        if artifact.get(key) != expected:
            raise RuntimeError(f"paper attack protocol mismatch: {key}={artifact.get(key)!r}")
    for key, expected in (("epsilon", 2/255), ("step_size", 0.5/255), ("relation_lambda", 0.5)):
        if abs(float(artifact.get(key)) - expected) > 1e-12:
            raise RuntimeError(f"paper attack protocol mismatch: {key}={artifact.get(key)!r}")
    if "active_views" in artifact and list(artifact["active_views"]) != EXPECTED_VIEWS:
        raise RuntimeError(f"active view contract mismatch: {artifact['active_views']!r}")
    for flag in ("use_relation", "use_relation_confidence", "use_directional_consensus", "use_semantic_quality"):
        if flag in artifact and artifact[flag] is not True:
            raise RuntimeError(f"Full artifact unexpectedly disables {flag}")
    if "variant" in artifact and str(artifact["variant"]).lower() != "full":
        raise RuntimeError(f"expected Full artifact, got variant={artifact['variant']!r}")
    idx = _tensor_list(artifact["image_indices"])
    if idx != list(range(n_images)):
        raise RuntimeError("full paper evaluation requires image indices 0..N-1 exactly")
    adv = artifact["adv_images"].float()
    if len(adv) != n_images or adv.ndim != 4 or tuple(adv.shape[1:]) != (3, 224, 224):
        raise RuntimeError(f"unexpected adversarial image shape: {tuple(adv.shape)}")
    if "clean_images" in artifact:
        clean = artifact["clean_images"].float()
        if clean.shape != adv.shape:
            raise RuntimeError("clean/adv image shape mismatch")
        linf = (adv - clean).abs().flatten(1).max(1).values
        if float(linf.max()) > 2/255 + 1e-6:
            raise RuntimeError(f"Linf budget violation: {float(linf.max())}")
    return adv


def summarize_joint(
    *, cache: dict[str, Any], adv_image_features: torch.Tensor, adv_text_features: torch.Tensor,
    source_key: str, victim_key: str, dataset: str,
    attack_artifact_sha256: str | None = None, text_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    n_images = int(cache["image_features"].shape[0])
    n_captions = int(cache["text_features"].shape[0])
    cti, positives = validate_cache(cache, n_images, n_captions)
    clean_img = cache["image_features"].float()
    clean_txt = cache["text_features"].float()
    if tuple(adv_image_features.shape) != tuple(clean_img.shape):
        raise RuntimeError("adversarial image feature shape mismatch")
    if tuple(adv_text_features.shape) != tuple(clean_txt.shape):
        raise RuntimeError("adversarial text feature shape mismatch")

    dataset_key = str(dataset).strip().upper()

    if dataset_key == "RSICD":
        # Historical RSICD numerical-replay contract:
        # construct the complete query x gallery matrices before ranking.
        adv_i2t = joint_i2t_fixed_matrix(
            adv_image_features,
            clean_txt,
            adv_text_features,
            positives,
        )
        adv_t2i = joint_t2i_fixed_matrix(
            clean_img,
            adv_image_features,
            adv_text_features,
            cti,
        )
        gallery_protocol = "record_level_fixed_gallery_rsicd_full_matrix"
        score_construction = "full_query_gallery_matrix"
        rank_definition = "torch.argsort_dim1_historical_rsicd"
    else:
        # Preserve the already-validated RSITMD / generic legacy path.
        iidx = list(range(n_images))
        cidx = list(range(n_captions))
        adv_i2t = joint_i2t_fixed(
            adv_image_features,
            clean_txt,
            adv_text_features,
            positives,
            iidx,
        )
        adv_t2i = joint_t2i_fixed(
            clean_img,
            adv_image_features,
            adv_text_features,
            cti,
            cidx,
        )
        gallery_protocol = "record_level_fixed_gallery"
        score_construction = "rowwise_query_gallery_scores"
        rank_definition = "torch.argsort_exact_legacy_semantics"

    clean_i2t = cache["i2t_ranks"].long().cpu()
    clean_t2i = cache["t2i_ranks"].long().cpu()

    result = {
        "schema": "ra-scg.fixed-gallery-evaluation.v1",
        "method": "RA-SCG",
        "dataset": dataset,
        "source_key": source_key,
        "victim_key": victim_key,
        "excluded_from_source_bb4": bool(victim_key == source_key),
        "gallery_protocol": gallery_protocol,
        "score_construction": score_construction,
        "rank_definition": rank_definition,
        "num_images": n_images,
        "num_captions": n_captions,
        "I2T": summarize_asr(clean_i2t, adv_i2t),
        "T2I": summarize_asr(clean_t2i, adv_t2i),
    }
    if attack_artifact_sha256:
        result["attack_artifact_sha256"] = attack_artifact_sha256
    if text_artifact_sha256:
        result["text_artifact_sha256"] = text_artifact_sha256
    return result
