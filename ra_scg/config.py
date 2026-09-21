from __future__ import annotations
from pathlib import Path
import yaml

def load_config(path: str | Path) -> tuple[Path, dict]:
    p=Path(path).expanduser().resolve()
    data=yaml.safe_load(p.read_text(encoding='utf-8'))
    if not isinstance(data,dict):
        raise TypeError('config root must be a mapping')
    return p,data

def repo_root_from_config(config_path: Path) -> Path:
    # Standard public configs live under <repo>/configs/*.yaml.
    return config_path.parent.parent.resolve()

def resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (repo_root/p).resolve()

def resolve_dataset_paths(config: dict, repo_root: Path):
    d=config['dataset']
    root=resolve_repo_path(d['root'], repo_root)
    j=Path(d['json']).expanduser(); j=j.resolve() if j.is_absolute() else (root/j).resolve()
    im=Path(d['images_dir']).expanduser(); im=im.resolve() if im.is_absolute() else (root/im).resolve()
    return root,j,im
