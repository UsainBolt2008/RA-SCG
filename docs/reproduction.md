# Reproduction workflow

## 1. Configure assets

Edit the relevant source config under `configs/` and set:

- dataset root and annotation JSON;
- surrogate checkpoint;
- source-specific frozen BA-SP caption JSON;
- optional NLTK data directory.

Edit `configs/evaluators.yaml` with the five evaluator checkpoint paths, local OpenAI CLIP source root, and OpenCLIP cache directory. Expected hashes are already included.

## 2. Prepare clean source metadata/cache

```bash
python scripts/prepare_clean.py \
  --config configs/rsitmd_remoteclip.yaml \
  --output-dir outputs/clean/rsitmd/remoteclip_vit_b32
```

Repeat for the second source and, for evaluation, prepare clean caches for all five evaluator encoders using `scripts/prepare_eval_cache.py`.

## 3. Validate frozen text

```bash
python scripts/validate_frozen_text.py \
  --config configs/rsitmd_remoteclip.yaml \
  --metadata outputs/clean/rsitmd/remoteclip_vit_b32/query_metadata.json
```

## 4. Generate adversarial images

```bash
python scripts/attack.py \
  --config configs/rsitmd_remoteclip.yaml \
  --metadata outputs/clean/rsitmd/remoteclip_vit_b32/query_metadata.json \
  --output-dir outputs/attack/rsitmd/remoteclip_vit_b32
```

The full paper run uses the complete test split and batch size 2. `--limit-images 2` is smoke-test only.

## 5. Evaluate each source/victim cell

```bash
python scripts/evaluate_attack.py \
  --dataset-config configs/rsitmd_remoteclip.yaml \
  --metadata outputs/clean/rsitmd/openai_clip_vit_b32/query_metadata.json \
  --evaluator-registry configs/evaluators.yaml \
  --source remoteclip_vit_b32 \
  --victim openai_clip_vit_b32 \
  --image-attack-file outputs/attack/rsitmd/remoteclip_vit_b32/adversarial_images.pt \
  --text-attack-file /path/to/rsitmd_remoteclip_ba_sp.json \
  --clean-cache outputs/clean/rsitmd/openai_clip_vit_b32/features_and_ranks.pt \
  --output-dir outputs/eval/rsitmd/remoteclip_vit_b32/openai_clip_vit_b32
```

Repeat for both sources and all five evaluators.

## 6. Aggregate BB4

```bash
python scripts/aggregate_bb4.py \
  --dataset RSITMD \
  --source-dir remoteclip_vit_b32=outputs/eval/rsitmd/remoteclip_vit_b32 \
  --source-dir georsclip_vit_b32_ret2=outputs/eval/rsitmd/georsclip_vit_b32_ret2 \
  --output outputs/eval/rsitmd_aggregate.json
```

Repeat for RSICD, then verify against the frozen full-precision reference:

```bash
python scripts/check_reference_results.py \
  --rsitmd outputs/eval/rsitmd_aggregate.json \
  --rsicd outputs/eval/rsicd_aggregate.json
```

Expected dataset-level cross-source BB4 values are stored in `metadata/paper_reference_results.json`.
