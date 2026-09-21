from __future__ import annotations
from pathlib import Path
import torch
from .models.remoteclip import RemoteCLIPAttackBridge
from .models.georsclip import GeoRSCLIPAttackBridge


def resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(repo_root)/p).resolve()


def build_source_bridge(source_cfg: dict, *, device: torch.device, repo_root: Path):
    name=str(source_cfg['name'])
    checkpoint=resolve_repo_path(source_cfg['checkpoint'], repo_root)
    architecture=str(source_cfg.get('architecture','ViT-B-32'))
    capitalize=bool(source_cfg.get('capitalize_text', name=='remoteclip_vit_b32'))
    if name == 'remoteclip_vit_b32':
        return RemoteCLIPAttackBridge(
            checkpoint=checkpoint,
            device=device,
            architecture=architecture,
            bert_tokenizer=None,
            capitalize_text=capitalize,
        )
    if name == 'georsclip_vit_b32_ret2':
        return GeoRSCLIPAttackBridge(
            checkpoint=checkpoint,
            device=device,
            architecture=architecture,
            bert_tokenizer=None,
            capitalize_text=capitalize,
            strict=bool(source_cfg.get('strict',False)),
        )
    raise KeyError(f'Unsupported RA-SCG surrogate: {name}')
