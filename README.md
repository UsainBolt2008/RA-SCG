# RA-SCG

**Relation-Augmented Semantic Consensus Gradient for Bidirectional Black-Box
Transfer Attacks on Remote-Sensing Image-Text Retrieval**

This repository provides the paper-faithful public implementation of RA-SCG.

## What this repository contains

- final RA-SCG Full attack core;
- caption-guided semantic localization and five-view construction;
- relation augmentation and adaptive semantic consensus;
- RemoteCLIP ViT-B/32 and GeoRSCLIP ViT-B/32 RET-2 surrogate bridges;
- portable RSITMD/RSICD JSON/image discovery;
- clean source-feature cache construction;
- explicit frozen BA-SP input validation;
- end-to-end source-side RA-SCG attack CLI;
- exact fixed-gallery ranking and conditional-ASR primitives;
- contract tests for the paper-facing attack and evaluation pipeline.

## Installation

    conda env create -f environment.yml
    conda activate ra-scg
    pip install -e .
    python scripts/setup_nltk.py --download
    export NLTK_DATA="$PWD/.nltk_data"
    python scripts/check_install.py

`environment.yml` defines the standalone public reproduction environment
(`ra-scg`, Python 3.10). Dependencies are also declared in `pyproject.toml`;
a separate `requirements.txt` is not required.

The original experiments used a machine-local Conda activation hook for
NLTK. The public workflow does not depend on that hidden side effect.

## Configure a paper run

Edit one of:

- `configs/rsitmd_remoteclip.yaml`
- `configs/rsitmd_georsclip.yaml`
- `configs/rsicd_remoteclip.yaml`
- `configs/rsicd_georsclip.yaml`

Set:

- dataset root;
- surrogate checkpoint;
- source-specific frozen BA-SP caption JSON.

The frozen BA-SP JSON is checked against the paper-frozen SP-bank SHA256 before
optimization starts. No live text candidate generation occurs in RA-SCG.

## Prepare the source clean cache

    python scripts/prepare_clean.py \
      --config configs/rsitmd_remoteclip.yaml \
      --output-dir outputs/clean/rsitmd/remoteclip_vit_b32

This creates portable `query_metadata.json` paths relative to the dataset root,
plus `features_and_ranks.pt` and clean retrieval metrics.

## Validate frozen text

    python scripts/validate_frozen_text.py \
      --config configs/rsitmd_remoteclip.yaml \
      --metadata outputs/clean/rsitmd/remoteclip_vit_b32/query_metadata.json

## Run RA-SCG

    python scripts/attack.py \
      --config configs/rsitmd_remoteclip.yaml \
      --metadata outputs/clean/rsitmd/remoteclip_vit_b32/query_metadata.json \
      --output-dir outputs/attack/rsitmd/remoteclip_vit_b32

Smoke test:

    python scripts/attack.py \
      --config configs/rsitmd_remoteclip.yaml \
      --metadata outputs/clean/rsitmd/remoteclip_vit_b32/query_metadata.json \
      --output-dir outputs/smoke/rsitmd_remoteclip \
      --limit-images 2

## Frozen image protocol

- Linf epsilon = 2/255
- step size = 0.5/255
- 10 image updates
- outer batch size = 2
- seed = 0
- momentum = 1
- base-loss alpha = 3
- relation lambda = 0.5
- zero-based ViT layers = 2,5,8,11
- views = identity, global 0.75, global 1.25, semantic zoom 0.75,
  semantic zoom 0.50
- victim queries during generation = 0

## Evaluation

This release includes the five-evaluator fixed-gallery evaluation pipeline,
source-excluded BB4 aggregation, and full-precision paper reference checks.
See `docs/evaluation.md` and `docs/reproduction.md` for the evaluation workflow.

See `docs/protocol.md`, `docs/method_to_code.md`, `docs/sp_bank.md`,
`docs/nltk.md`, and `docs/reproduction.md`.


## Paper reference results

The frozen paper reference reports conditional source-excluded BB4 ASR (%).
For each active source, BB4 averages the four non-source victims and both
I2T/T2I directions. Full-precision values are stored in
`metadata/paper_reference_results.json`.

| Dataset | Source | ASR@1 | ASR@5 | ASR@10 |
|---|---|---:|---:|---:|
| RSITMD | RemoteCLIP ViT-B/32 | 68.1038 | 53.1373 | 47.2808 |
| RSITMD | GeoRSCLIP ViT-B/32 RET-2 | 73.7608 | 63.8786 | 57.0512 |
| RSICD | RemoteCLIP ViT-B/32 | 72.3616 | 61.8977 | 55.2675 |
| RSICD | GeoRSCLIP ViT-B/32 RET-2 | 79.0901 | 71.9807 | 65.9644 |
| **RSITMD aggregate** | **two-source macro** | **70.9323** | **58.5080** | **52.1660** |
| **RSICD aggregate** | **two-source macro** | **75.7259** | **66.9392** | **60.6159** |
| **Overall** | **dataset macro** | **73.3291** | **62.7236** | **56.3910** |

The full-precision overall values are:

- ASR@1: `73.329095613922`
- ASR@5: `62.7235671212366`
- ASR@10: `56.39095066466163`


## License

This project is released under the MIT License. See `LICENSE`.
