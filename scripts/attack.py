from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ra_scg import RASCGAttack
from ra_scg.config import (
    load_config,
    repo_root_from_config,
    resolve_dataset_paths,
    resolve_repo_path,
)
from ra_scg.data import load_metadata
from ra_scg.factory import build_source_bridge
from ra_scg.text_input import load_frozen_ba_sp, sha256_file


def set_all_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap=argparse.ArgumentParser(description='Run paper-faithful RA-SCG image optimization.')
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--metadata',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--clean-cache',type=Path,default=None,help='Defaults to features_and_ranks.pt beside --metadata.')
    ap.add_argument('--device',default=None)
    ap.add_argument('--limit-images',type=int,default=None,help='Smoke-test only; default attacks the full split.')
    ap.add_argument('--save-history',action='store_true')
    a=ap.parse_args()

    cfg_path,cfg=load_config(a.config)
    repo_root=repo_root_from_config(cfg_path)
    dataset_root,_,_=resolve_dataset_paths(cfg,repo_root)
    source=cfg['surrogate']
    attack_cfg=cfg['attack']
    device=torch.device(a.device or cfg.get('runtime',{}).get('device','cuda:0'))
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    # Lock public CLI to the paper protocol.
    required={
        'epsilon':2/255,
        'step_size':0.5/255,
        'steps':10,
        'batch_size':2,
        'seed':0,
        'momentum':1.0,
        'base_loss_alpha':3.0,
        'relation_lambda':0.5,
    }
    for key,value in required.items():
        observed=float(attack_cfg[key]) if key not in ('steps','batch_size','seed') else int(attack_cfg[key])
        if abs(observed-value)>1e-12:
            raise RuntimeError(f'paper protocol mismatch: {key}={observed!r}, expected={value!r}')

    meta,cti,positives=load_metadata(a.metadata)
    cache_path=(a.clean_cache.resolve() if a.clean_cache is not None else (a.metadata.resolve().parent/'features_and_ranks.pt'))
    if not cache_path.is_file():
        raise FileNotFoundError(
            f'Clean source cache not found: {cache_path}. Run scripts/prepare_clean.py first.'
        )
    cache=torch.load(cache_path,map_location='cpu')
    cache_cti=[int(x) for x in cache['caption_to_image'].tolist()]
    cache_pos=[[int(x) for x in row] for row in cache['positive_caption_indices']]
    if cache_cti != cti or cache_pos != positives:
        raise RuntimeError('clean cache metadata mapping mismatch')
    expected_images=int(cfg['dataset']['expected_images'])
    expected_captions=int(cfg['dataset']['expected_captions'])
    if len(meta['images'])!=expected_images or len(meta['captions'])!=expected_captions:
        raise RuntimeError('metadata cardinality mismatch')

    clean_captions=[x['caption'] for x in meta['captions']]
    text_path=resolve_repo_path(cfg['text']['frozen_ba_sp_captions'],repo_root)
    expected_bank=str(cfg['text']['expected_candidate_bank_sha256'])
    text_obj,adv_captions=load_frozen_ba_sp(
        text_path,
        expected_captions=expected_captions,
        expected_bank_sha256=expected_bank,
        clean_captions=clean_captions,
    )

    bridge=build_source_bridge(source,device=device,repo_root=repo_root)
    nltk_cfg=cfg.get('runtime',{}).get('nltk_data')
    nltk_path=resolve_repo_path(nltk_cfg,repo_root) if nltk_cfg else None
    if nltk_path is not None and not nltk_path.exists():
        nltk_path=None
    attacker=RASCGAttack(bridge,nltk_data=nltk_path)

    image_indices=list(range(expected_images))
    if a.limit_images is not None:
        if a.limit_images <= 0:
            raise ValueError('--limit-images must be positive')
        image_indices=image_indices[:min(a.limit_images,len(image_indices))]

    set_all_seeds(0)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0=time.time()

    adv_chunks=[]
    history_rows=[]
    linf_max=0.0

    for start in range(0,len(image_indices),2):
        bidx=image_indices[start:start+2]
        clean=torch.stack([
            bridge.load_raw_image((dataset_root/meta['images'][i]['resolved_path']).resolve())
            for i in bidx
        ]).to(device)
        with torch.no_grad():
            cur=bridge.inference_image(bridge.normalization(clean))['image_feat'].float()
            cached=cache['image_features'][bidx].to(device).float()
            cos=(cur*cached).sum(-1)
        if float(cos.min()) < 0.9999:
            raise RuntimeError(f'clean preprocessing/cache mismatch: min cosine={float(cos.min())}')
        groups=[cache_pos[i] for i in bidx]
        if any(len(g)!=5 for g in groups):
            raise RuntimeError('not exactly five positives')
        clean_groups=[[clean_captions[c] for c in g] for g in groups]
        adv_groups=[[adv_captions[c] for c in g] for g in groups]

        # Exact frozen runner seed schedule: seed + global batch start.
        batch_seed=int(attack_cfg['seed']) + start
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)

        result=attacker.run_batch(clean,clean_groups,adv_groups)
        adv=result['adv_images'].detach()
        linf=(adv-clean).abs().flatten(1).max(1).values
        linf_max=max(linf_max,float(linf.max()))
        if float(linf.max()) > 2/255 + 1e-6:
            raise RuntimeError(f'Linf budget violation: {float(linf.max())}')
        adv_chunks.append(adv.cpu())
        if a.save_history:
            history_rows.append({'image_indices':bidx,'history':result['history']})
        print(f"RA_SCG_PROGRESS {min(start+2,len(image_indices))}/{len(image_indices)} linf={float(linf.max()):.9f}",flush=True)

    if device.type=='cuda':
        torch.cuda.synchronize(device)
        peak_bytes=int(torch.cuda.max_memory_allocated(device))
    else:
        peak_bytes=None
    elapsed=time.time()-t0
    adv_all=torch.cat(adv_chunks,0)

    payload={
        'method':'RA-SCG',
        'dataset':cfg['dataset']['name'],
        'surrogate_key':source['name'],
        'image_indices':torch.tensor(image_indices,dtype=torch.long),
        'adv_images':adv_all,
        'target_queries':0,
        'candidate_generation_during_attack':False,
        'epsilon':2/255,
        'step_size':0.5/255,
        'steps':10,
        'batch_size':2,
        'seed':0,
        'momentum':1.0,
        'active_views':['identity','global_075','global_125','semantic_zoom_075','semantic_zoom_050'],
        'relation_lambda':0.5,
        'candidate_bank_sha256':expected_bank,
        'text_attack_sha256':sha256_file(text_path),
    }
    if a.save_history:
        payload['histories']=history_rows

    out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    torch.save(payload,out/'adversarial_images.pt')

    selected_caption_ids=[c for i in image_indices for c in positives[i]]
    selected_records={int(r['caption_index']):r for r in text_obj['records']}
    text_out={
        'method':'BA-SP',
        'dataset':cfg['dataset']['name'],
        'source_key':source['name'],
        'candidate_bank_sha256':expected_bank,
        'records':[selected_records[c] for c in selected_caption_ids],
    }
    (out/'frozen_adversarial_captions.json').write_text(
        json.dumps(text_out,ensure_ascii=False,indent=2),encoding='utf-8'
    )
    summary={
        'status':'PASS',
        'method':'RA-SCG',
        'dataset':cfg['dataset']['name'],
        'surrogate_key':source['name'],
        'num_images':len(image_indices),
        'num_captions':len(selected_caption_ids),
        'linf_max':linf_max,
        'image_optimization_seconds':elapsed,
        'peak_cuda_memory_bytes_torch':peak_bytes,
        'target_queries':0,
        'candidate_generation_during_attack':False,
        'candidate_bank_sha256':expected_bank,
        'text_attack_sha256':sha256_file(text_path),
    }
    (out/'attack_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('RA_SCG_ATTACK_STATUS=PASS')

if __name__=='__main__':
    main()
