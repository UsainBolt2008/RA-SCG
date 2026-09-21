from __future__ import annotations

import json
from pathlib import Path

import torch


def compute_i2t_ranks(
    scores: torch.Tensor,
    positive_caption_indices: list[list[int]],
) -> torch.Tensor:
    order = torch.argsort(scores, dim=1, descending=True)
    inverse = torch.empty_like(order)
    rank_values = torch.arange(
        scores.shape[1], dtype=order.dtype
    ).unsqueeze(0).expand_as(order)
    inverse.scatter_(1, order, rank_values)
    ranks=[]
    for i,pos in enumerate(positive_caption_indices):
        p=torch.tensor(pos,dtype=torch.long)
        ranks.append(int(inverse[i,p].min().item())+1)
    return torch.tensor(ranks,dtype=torch.long)


def compute_t2i_ranks(
    scores: torch.Tensor,
    caption_to_image: list[int],
) -> torch.Tensor:
    order=torch.argsort(scores.T,dim=1,descending=True)
    positives=torch.tensor(caption_to_image,dtype=torch.long).unsqueeze(1)
    matches=order.eq(positives)
    if not matches.any(dim=1).all():
        raise RuntimeError('At least one T2I positive image was not found')
    return matches.to(torch.long).argmax(dim=1)+1


def summarize_ranks(ranks: torch.Tensor) -> dict[str,float]:
    rf=ranks.float()
    out={
        'R@1': float((ranks<=1).float().mean()*100),
        'R@5': float((ranks<=5).float().mean()*100),
        'R@10': float((ranks<=10).float().mean()*100),
        'MRR': float((1.0/rf).mean()),
        'mean_rank': float(rf.mean()),
        'median_rank': float(rf.median()),
    }
    return {k:round(v,6) for k,v in out.items()}


@torch.inference_mode()
def encode_images_with_bridge(
    bridge,
    metadata: dict,
    dataset_root: Path,
    *,
    batch_size: int=128,
) -> torch.Tensor:
    rows=[]
    images=metadata['images']
    for start in range(0,len(images),batch_size):
        batch=[]
        for rec in images[start:start+batch_size]:
            path=(Path(dataset_root)/rec['resolved_path']).resolve()
            batch.append(bridge.load_raw_image(path))
        raw=torch.stack(batch).to(bridge.device, non_blocking=True)
        feat=bridge.inference_image(bridge.normalization(raw))['image_feat'].float()
        rows.append(feat.cpu())
    out=torch.cat(rows,0)
    if not torch.isfinite(out).all():
        raise RuntimeError('Image embeddings contain NaN or Inf')
    return out


@torch.inference_mode()
def encode_texts_with_bridge(
    bridge,
    metadata: dict,
    *,
    batch_size: int=512,
) -> torch.Tensor:
    texts=[x['caption'] for x in metadata['captions']]
    rows=[]
    for start in range(0,len(texts),batch_size):
        feat=bridge.encode_text_strings(texts[start:start+batch_size]).float()
        rows.append(feat.cpu())
    out=torch.cat(rows,0)
    if not torch.isfinite(out).all():
        raise RuntimeError('Text embeddings contain NaN or Inf')
    return out


def build_clean_cache(
    bridge,
    metadata: dict,
    caption_to_image: list[int],
    positive_caption_indices: list[list[int]],
    dataset_root: Path,
    *,
    image_batch_size: int=128,
    text_batch_size: int=512,
):
    image_features=encode_images_with_bridge(
        bridge, metadata, dataset_root, batch_size=image_batch_size
    )
    text_features=encode_texts_with_bridge(
        bridge, metadata, batch_size=text_batch_size
    )
    scores=image_features @ text_features.T
    if not torch.isfinite(scores).all():
        raise RuntimeError('Similarity matrix contains NaN or Inf')
    i2t=compute_i2t_ranks(scores,positive_caption_indices)
    t2i=compute_t2i_ranks(scores,caption_to_image)
    metrics={
        'I2T':summarize_ranks(i2t),
        'T2I':summarize_ranks(t2i),
    }
    metrics['mean_recall']=round(sum([
        metrics['I2T']['R@1'],metrics['I2T']['R@5'],metrics['I2T']['R@10'],
        metrics['T2I']['R@1'],metrics['T2I']['R@5'],metrics['T2I']['R@10'],
    ])/6.0,6)
    cache={
        'image_features':image_features,
        'text_features':text_features,
        'i2t_ranks':i2t,
        't2i_ranks':t2i,
        'caption_to_image':torch.tensor(caption_to_image,dtype=torch.long),
        'positive_caption_indices':positive_caption_indices,
    }
    return cache,metrics
