from __future__ import annotations
import argparse
from pathlib import Path
import torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--atol',type=float,default=0.0)
    a=ap.parse_args()
    c=torch.load(a.candidate,map_location='cpu')
    r=torch.load(a.reference,map_location='cpu')
    cidx=[int(x) for x in c['image_indices'].tolist()]
    ridx=[int(x) for x in r['image_indices'].tolist()]
    pos={v:i for i,v in enumerate(ridx)}
    if any(i not in pos for i in cidx):
        raise RuntimeError('candidate image indices are not contained in reference')
    ref=torch.stack([r['adv_images'][pos[i]].float() for i in cidx])
    cand=c['adv_images'].float()
    if cand.shape!=ref.shape:
        raise RuntimeError(f'shape mismatch: {tuple(cand.shape)} vs {tuple(ref.shape)}')
    diff=(cand-ref).abs()
    mx=float(diff.max())
    mean=float(diff.mean())
    exact=bool(torch.equal(cand,ref))
    print(f'candidate_images={len(cidx)}')
    print(f'max_abs_diff={mx:.12g}')
    print(f'mean_abs_diff={mean:.12g}')
    print(f'exact_equal={exact}')
    if mx > a.atol:
        raise SystemExit(f'PARITY_FAIL max_abs_diff={mx} > atol={a.atol}')
    print('RA_SCG_REFERENCE_PARITY_STATUS=PASS')

if __name__=='__main__':
    main()
