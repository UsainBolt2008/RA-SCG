import torch

from evaluation.evaluator import summarize_joint
from evaluation.fixed_gallery import (
    ranks_from_score_matrix,
    joint_i2t_fixed_matrix,
    joint_t2i_fixed_matrix,
)


def _reference_ranks_matrix(scores, posmap):
    order=torch.argsort(scores,dim=1,descending=True)
    inverse=torch.empty_like(order)
    rv=torch.arange(scores.shape[1],dtype=order.dtype).unsqueeze(0).expand_as(order)
    inverse.scatter_(1,order,rv)
    out=[]
    for qi,pos in enumerate(posmap):
        out.append(int(inverse[qi,torch.as_tensor(pos,dtype=torch.long)].min().item())+1)
    return torch.tensor(out,dtype=torch.long)


def test_ranks_from_score_matrix_matches_historical_reference():
    scores=torch.tensor([
        [0.2,0.8,0.1,0.4],
        [0.7,0.3,0.6,0.5],
    ],dtype=torch.float32)
    positives=[[1,3],[0]]
    assert torch.equal(
        ranks_from_score_matrix(scores,positives),
        _reference_ranks_matrix(scores,positives),
    )


def test_rsicd_matrix_joint_matches_historical_reference_formula():
    torch.manual_seed(7)
    n=3
    c=15
    d=8
    clean_i=torch.nn.functional.normalize(torch.rand(n,d),dim=-1)
    adv_i=torch.nn.functional.normalize(torch.rand(n,d),dim=-1)
    clean_t=torch.nn.functional.normalize(torch.rand(c,d),dim=-1)
    adv_t=torch.nn.functional.normalize(torch.rand(c,d),dim=-1)
    positives=[list(range(i*5,(i+1)*5)) for i in range(n)]
    cti=torch.tensor([i for i in range(n) for _ in range(5)])

    si=adv_i@clean_t.T
    for i,p in enumerate(positives):
        pp=torch.tensor(p,dtype=torch.long)
        si[i,pp]=adv_i[i]@adv_t[pp].T
    expected_i=_reference_ranks_matrix(si,positives)

    st=adv_t@clean_i.T
    ar=torch.arange(c)
    st[ar,cti]=(adv_t*adv_i[cti]).sum(-1)
    expected_t=_reference_ranks_matrix(st,[[int(cti[x])] for x in range(c)])

    got_i=joint_i2t_fixed_matrix(adv_i,clean_t,adv_t,positives)
    got_t=joint_t2i_fixed_matrix(clean_i,adv_i,adv_t,cti)
    assert torch.equal(got_i,expected_i)
    assert torch.equal(got_t,expected_t)


def test_summarize_joint_selects_rsicd_historical_matrix_contract():
    torch.manual_seed(11)
    n=3
    c=15
    d=8
    clean_i=torch.nn.functional.normalize(torch.rand(n,d),dim=-1)
    clean_t=torch.nn.functional.normalize(torch.rand(c,d),dim=-1)
    adv_i=torch.nn.functional.normalize(torch.rand(n,d),dim=-1)
    adv_t=torch.nn.functional.normalize(torch.rand(c,d),dim=-1)
    positives=[list(range(i*5,(i+1)*5)) for i in range(n)]
    cti=torch.tensor([i for i in range(n) for _ in range(5)])

    clean_scores=clean_i@clean_t.T
    clean_i_r=_reference_ranks_matrix(clean_scores,positives)
    clean_t_r=_reference_ranks_matrix(clean_t@clean_i.T,[[int(cti[x])] for x in range(c)])
    cache={
        "image_features":clean_i,
        "text_features":clean_t,
        "i2t_ranks":clean_i_r,
        "t2i_ranks":clean_t_r,
        "caption_to_image":cti,
        "positive_caption_indices":positives,
    }

    out=summarize_joint(
        cache=cache,
        adv_image_features=adv_i,
        adv_text_features=adv_t,
        source_key="remoteclip_vit_b32",
        victim_key="remoteclip_vit_b32",
        dataset="RSICD",
    )
    assert out["score_construction"]=="full_query_gallery_matrix"
    assert out["rank_definition"]=="torch.argsort_dim1_historical_rsicd"
    assert out["gallery_protocol"]=="record_level_fixed_gallery_rsicd_full_matrix"
