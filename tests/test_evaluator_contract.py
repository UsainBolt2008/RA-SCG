import torch
from evaluation.evaluator import summarize_joint,validate_attack_artifact

def test_dataset_agnostic_joint_summary_shapes():
    torch.manual_seed(3); n=3; c=15; d=8
    img=torch.nn.functional.normalize(torch.rand(n,d),dim=-1); txt=torch.nn.functional.normalize(torch.rand(c,d),dim=-1)
    positives=[list(range(i*5,(i+1)*5)) for i in range(n)]; cti=torch.tensor([i for i in range(n) for _ in range(5)])
    scores=img@txt.T
    from evaluation.clean_cache import compute_i2t_ranks,compute_t2i_ranks
    cache={"image_features":img,"text_features":txt,"i2t_ranks":compute_i2t_ranks(scores,positives),"t2i_ranks":compute_t2i_ranks(scores,cti.tolist()),"caption_to_image":cti,"positive_caption_indices":positives}
    out=summarize_joint(cache=cache,adv_image_features=img,adv_text_features=txt,source_key="remoteclip_vit_b32",victim_key="openai_clip_vit_b32",dataset="SYNTH")
    assert out["num_images"]==3 and out["num_captions"]==15; assert out["I2T"]["ASR@1"]["attack_successes"]==0; assert out["T2I"]["ASR@10"]["attack_successes"]==0

def test_full_attack_protocol_guard():
    n=2; clean=torch.zeros(n,3,224,224); adv=clean.clone()
    a={"method":"RA-SCG","dataset":"RSITMD","surrogate_key":"remoteclip_vit_b32","target_queries":0,"candidate_generation_during_attack":False,"epsilon":2/255,"step_size":0.5/255,"steps":10,"relation_lambda":0.5,"active_views":["identity","global_075","global_125","semantic_zoom_075","semantic_zoom_050"],"image_indices":torch.arange(n),"clean_images":clean,"adv_images":adv,"variant":"Full"}
    out=validate_attack_artifact(a,n_images=n,expected_dataset="RSITMD",expected_source="remoteclip_vit_b32"); assert out.shape==adv.shape
