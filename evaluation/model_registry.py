from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import importlib
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose
import yaml

MODEL_KEYS = (
    "remoteclip_vit_b32",
    "remoteclip_rn50",
    "georsclip_vit_b32_ret2",
    "openai_clip_vit_b32",
    "openai_clip_rn50",
)
SOURCE_KEYS = ("remoteclip_vit_b32", "georsclip_vit_b32_ret2")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class EvaluatorSpec:
    key: str
    backend: str
    architecture: str
    checkpoint: Path
    checkpoint_sha256: str | None
    capitalize_text: bool
    strict: bool = True
    openai_clip_root: Path | None = None
    open_clip_cache_dir: Path | None = None
    openai_clip_source_sha256: dict[str, str] | None = None


def _resolve(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def _verify_openai_source(root: Path, expected: dict[str, str] | None) -> None:
    if not expected:
        return
    for rel, digest in expected.items():
        p = (root / rel).resolve()
        if root not in p.parents:
            raise RuntimeError(f"OpenAI CLIP fingerprint path escapes source root: {rel}")
        if not p.is_file():
            raise FileNotFoundError(p)
        got = sha256_file(p)
        if got != str(digest):
            raise RuntimeError(
                f"OpenAI CLIP source fingerprint mismatch for {rel}: expected={digest}, got={got}"
            )


def load_evaluator_registry(path: Path) -> dict[str, EvaluatorSpec]:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), dict):
        raise TypeError("evaluator registry must contain a 'models' mapping")
    base = path.parent
    global_openai = _resolve(raw.get("openai_clip_root"), base)
    global_cache = _resolve(raw.get("open_clip_cache_dir"), base)
    global_source_hashes = raw.get("openai_clip_source_sha256") or None
    if global_source_hashes is not None and not isinstance(global_source_hashes, dict):
        raise TypeError("openai_clip_source_sha256 must be a mapping")

    out: dict[str, EvaluatorSpec] = {}
    for key, cfg in raw["models"].items():
        if key not in MODEL_KEYS:
            raise KeyError(f"unsupported evaluator key: {key}")
        if not isinstance(cfg, dict):
            raise TypeError(f"model config must be mapping: {key}")
        checkpoint = _resolve(cfg["checkpoint"], base)
        assert checkpoint is not None
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        expected_ckpt = cfg.get("checkpoint_sha256")
        if expected_ckpt:
            got = sha256_file(checkpoint)
            if got != str(expected_ckpt):
                raise RuntimeError(
                    f"checkpoint SHA256 mismatch for {key}: expected={expected_ckpt}, got={got}"
                )
        spec = EvaluatorSpec(
            key=key,
            backend=str(cfg["backend"]),
            architecture=str(cfg["architecture"]),
            checkpoint=checkpoint,
            checkpoint_sha256=str(expected_ckpt) if expected_ckpt else None,
            capitalize_text=bool(cfg.get("capitalize_text", False)),
            strict=bool(cfg.get("strict", True)),
            openai_clip_root=_resolve(cfg.get("openai_clip_root"), base) or global_openai,
            open_clip_cache_dir=_resolve(cfg.get("open_clip_cache_dir"), base) or global_cache,
            openai_clip_source_sha256=global_source_hashes,
        )
        if spec.backend == "openai_clip":
            if spec.openai_clip_root is None or not spec.openai_clip_root.is_dir():
                raise RuntimeError(f"{key} requires a valid openai_clip_root")
            _verify_openai_source(spec.openai_clip_root, spec.openai_clip_source_sha256)
        elif spec.backend != "open_clip":
            raise KeyError(f"unsupported backend {spec.backend!r} for {key}")
        out[key] = spec

    missing = [k for k in MODEL_KEYS if k not in out]
    if missing:
        raise RuntimeError(f"registry missing required evaluators: {missing}")
    return out


def _unwrap_checkpoint(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError(type(obj))
    return {str(k).removeprefix("module."): v for k, v in obj.items()}


def _load_local_openai_clip(root: Path):
    root = Path(root).resolve()
    prior = sys.modules.get("clip")
    if prior is not None:
        prior_file = Path(getattr(prior, "__file__", "")).resolve()
        if root not in prior_file.parents:
            raise RuntimeError(
                "A different 'clip' module is already imported: "
                f"{prior_file}; expected source under {root}"
            )
        return prior
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("clip")
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
    module_file = Path(module.__file__).resolve()
    if root not in module_file.parents:
        raise RuntimeError(
            f"OpenAI CLIP provenance mismatch: imported {module_file}, expected under {root}"
        )
    return module


class EvaluationBridge:
    def __init__(self, spec: EvaluatorSpec, device: torch.device):
        self.spec = spec
        self.device = torch.device(device)
        if spec.backend == "open_clip":
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                spec.architecture,
                pretrained="openai",  # paper/frozen evaluator construction path
                device="cpu",
                cache_dir=str(spec.open_clip_cache_dir) if spec.open_clip_cache_dir else None,
            )
            state = _unwrap_checkpoint(torch.load(spec.checkpoint, map_location="cpu"))
            message = model.load_state_dict(state, strict=spec.strict)
            print(f"Checkpoint load [{spec.key}]:", message)
            # Match the historical evaluator exactly rather than relying on a newer tokenizer factory.
            self._tokenize = open_clip.tokenize
        else:
            clip = _load_local_openai_clip(spec.openai_clip_root)
            model, preprocess = clip.load(str(spec.checkpoint.resolve()), device="cpu", jit=False)
            self._tokenize = clip.tokenize
            print(f"Checkpoint load [{spec.key}]: OpenAI CLIP local loader")

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model.float().to(self.device).eval()
        self.raw_preprocess = Compose(preprocess.transforms[:-1])
        self.normalization = preprocess.transforms[-1]

    def transform_texts(self, texts: list[str]) -> list[str]:
        out = [" ".join(str(x).strip().split()) for x in texts]
        if self.spec.capitalize_text:
            out = [x.capitalize() for x in out]
        return out

    def load_raw_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            tensor = self.raw_preprocess(image.convert("RGB"))
        if tensor.ndim != 3:
            raise RuntimeError(f"unexpected raw image shape: {tuple(tensor.shape)}")
        if float(tensor.min()) < -1e-7 or float(tensor.max()) > 1.0000001:
            raise RuntimeError("raw preprocessing must produce RGB tensors in [0,1]")
        return tensor

    @torch.inference_mode()
    def encode_images(self, images: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
        rows = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(self.device, non_blocking=True)
            feat = self.model.encode_image(self.normalization(batch))
            rows.append(F.normalize(feat.float(), dim=-1).cpu())
        out = torch.cat(rows, 0)
        if not torch.isfinite(out).all():
            raise RuntimeError("image embeddings contain NaN/Inf")
        return out

    @torch.inference_mode()
    def encode_image_paths(self, paths: list[Path], batch_size: int = 128) -> torch.Tensor:
        rows = []
        for start in range(0, len(paths), batch_size):
            raw = torch.stack([self.load_raw_image(p) for p in paths[start:start + batch_size]])
            rows.append(self.encode_images(raw, batch_size=batch_size))
        return torch.cat(rows, 0)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], batch_size: int = 512) -> torch.Tensor:
        texts = self.transform_texts(texts)
        rows = []
        for start in range(0, len(texts), batch_size):
            tokens = self._tokenize(texts[start:start + batch_size]).to(self.device)
            feat = self.model.encode_text(tokens)
            rows.append(F.normalize(feat.float(), dim=-1).cpu())
        out = torch.cat(rows, 0)
        if not torch.isfinite(out).all():
            raise RuntimeError("text embeddings contain NaN/Inf")
        return out
