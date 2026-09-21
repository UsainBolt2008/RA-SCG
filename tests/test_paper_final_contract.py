from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from ra_scg.text_input import load_frozen_ba_sp


def _assigned_names(target):
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out=set()
        for item in target.elts:
            out |= _assigned_names(item)
        return out
    return set()


def _ids(node):
    out=set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            out.add(item.id)
        elif isinstance(item, ast.Attribute):
            out.add(item.attr)
    return out


def test_confidence_weighted_relation_residual_is_in_public_core():
    root = Path(__file__).resolve().parents[1]
    text = (root / "ra_scg" / "core.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(text)

    confidence_scale = 0
    gated_scale = 0
    relation_update = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        names = set()
        for target in node.targets:
            names |= _assigned_names(target)

        ids = _ids(node.value)

        if "scale" in names:
            if (
                {"relation_lambda", "confidence"} <= ids
                and "gate" not in ids
            ):
                confidence_scale += 1

            if {
                "relation_lambda",
                "confidence",
                "gate",
            } <= ids:
                gated_scale += 1

        if "g" in names:
            if {"g0n", "scale", "grn"} <= ids:
                relation_update += 1

    assert confidence_scale == 1
    assert gated_scale == 0
    assert relation_update == 1



def _validate(original, clean, expect_pass):
    payload={
        'method':'BA-SP',
        'candidate_bank_sha256':'bank',
        'records':[{
            'caption_index':0,
            'original_caption':original,
            'adversarial_caption':'changed caption',
        }],
    }
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'x.json'
        p.write_text(json.dumps(payload),encoding='utf-8')
        passed=True
        try:
            load_frozen_ba_sp(
                p,
                expected_captions=1,
                expected_bank_sha256='bank',
                clean_captions=[clean],
            )
        except RuntimeError:
            passed=False
    assert passed is expect_pass


def test_frozen_text_provenance_allows_case_and_whitespace_only():
    _validate('The airport is very large .','the airport is very large .',True)
    _validate('A  road near water .','a road near water .',True)
    _validate('A road near water .','a road near forest .',False)
