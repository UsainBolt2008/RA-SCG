from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def _norm_ws(s: str) -> str:
    return ' '.join(str(s).strip().split())

def load_frozen_ba_sp(path: Path, *, expected_captions: int, expected_bank_sha256: str, clean_captions: list[str] | None=None) -> tuple[dict[str,Any], list[str]]:
    path=Path(path)
    obj=json.loads(path.read_text(encoding='utf-8'))
    if obj.get('method') not in (None,'BA-SP'):
        raise RuntimeError(f"Expected BA-SP text file, got method={obj.get('method')!r}")
    observed=obj.get('candidate_bank_sha256')
    if observed != expected_bank_sha256:
        raise RuntimeError(f"Frozen SP-bank hash mismatch: expected={expected_bank_sha256}, observed={observed}")
    records=obj.get('records',[])
    by={int(r['caption_index']):r for r in records}
    expected_ids=list(range(expected_captions))
    if len(records)!=expected_captions or sorted(by)!=expected_ids:
        raise RuntimeError(f"Frozen BA-SP caption-index contract mismatch: records={len(records)}, expected={expected_captions}")
    adv=[]
    for i in expected_ids:
        r=by[i]
        if 'adversarial_caption' not in r:
            raise KeyError(f"record {i} missing adversarial_caption")
        if clean_captions is not None and r.get('original_caption') is not None:
            if _norm_ws(r['original_caption']).casefold() != _norm_ws(clean_captions[i]).casefold():
                raise RuntimeError(f"BA-SP original-caption content mismatch at caption {i}")
        adv.append(str(r['adversarial_caption']))
    return obj, adv
