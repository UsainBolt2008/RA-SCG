from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ra_scg.config import (
    load_config,
    repo_root_from_config,
    resolve_dataset_paths,
)
from ra_scg.data import read_dataset
from ra_scg.factory import build_source_bridge
from evaluation.clean_cache import build_clean_cache


def main():
    ap=argparse.ArgumentParser(description='Build portable RA-SCG clean metadata/cache for a surrogate.')
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--device',default=None)
    ap.add_argument('--image-batch-size',type=int,default=128)
    ap.add_argument('--text-batch-size',type=int,default=512)
    a=ap.parse_args()

    cfg_path,cfg=load_config(a.config)
    repo_root=repo_root_from_config(cfg_path)
    dataset_root,dataset_json,images_dir=resolve_dataset_paths(cfg,repo_root)
    source=cfg['surrogate']
    device=torch.device(a.device or cfg.get('runtime',{}).get('device','cuda:0'))
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    meta,cti,positives=read_dataset(
        dataset_root,dataset_json,images_dir,
        split=str(cfg['dataset'].get('split','test')),
        capitalize_text=bool(source.get('capitalize_text',False)),
        expected_captions_per_image=5,
    )
    expected_images=int(cfg['dataset']['expected_images'])
    expected_captions=int(cfg['dataset']['expected_captions'])
    if len(meta['images'])!=expected_images or len(meta['captions'])!=expected_captions:
        raise RuntimeError(
            f"dataset cardinality mismatch: images={len(meta['images'])}/{expected_images}, "
            f"captions={len(meta['captions'])}/{expected_captions}"
        )

    bridge=build_source_bridge(source,device=device,repo_root=repo_root)
    cache,metrics=build_clean_cache(
        bridge,meta,cti,positives,dataset_root,
        image_batch_size=a.image_batch_size,
        text_batch_size=a.text_batch_size,
    )

    out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    (out/'query_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    torch.save(cache,out/'features_and_ranks.pt')
    metrics.update({
        'dataset':cfg['dataset']['name'],
        'surrogate':source['name'],
        'num_images':len(meta['images']),
        'num_captions':len(meta['captions']),
        'portable_resolved_paths':True,
    })
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))
    print('RA_SCG_PREPARE_CLEAN_STATUS=PASS')

if __name__=='__main__':
    main()
