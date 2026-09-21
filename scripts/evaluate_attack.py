from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from ra_scg.config import load_config,repo_root_from_config,resolve_dataset_paths
from ra_scg.data import load_metadata
from evaluation.model_registry import load_evaluator_registry,EvaluationBridge,sha256_file
from evaluation.evaluator import load_text_records,validate_attack_artifact,validate_cache,summarize_joint


def _resolve_eval_batch_sizes(dataset_name: str, image_batch_size: int | None, text_batch_size: int | None):
    dataset_key=str(dataset_name).strip().upper()
    # Exact historical numerical replay for RSICD used 64/256.
    # Keep the already-validated RC3 RSITMD defaults unchanged.
    default_image=64 if dataset_key=="RSICD" else 128
    default_text=256 if dataset_key=="RSICD" else 512
    return (
        int(image_batch_size) if image_batch_size is not None else default_image,
        int(text_batch_size) if text_batch_size is not None else default_text,
    )


def main():
    p=argparse.ArgumentParser(description="Evaluate a full RA-SCG attack under the frozen fixed-gallery protocol.")
    p.add_argument("--dataset-config",type=Path,required=True); p.add_argument("--metadata",type=Path,required=True); p.add_argument("--evaluator-registry",type=Path,required=True)
    p.add_argument("--source",required=True); p.add_argument("--victim",required=True); p.add_argument("--image-attack-file",type=Path,required=True); p.add_argument("--text-attack-file",type=Path,required=True)
    p.add_argument("--clean-cache",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda:0")
    p.add_argument("--image-batch-size",type=int,default=None); p.add_argument("--text-batch-size",type=int,default=None); p.add_argument("--skip-clean-cache-check",action="store_true"); a=p.parse_args()
    cfg_path,cfg=load_config(a.dataset_config); repo=repo_root_from_config(cfg_path); dataset_root,_,_=resolve_dataset_paths(cfg,repo)
    dataset_name=str(cfg["dataset"]["name"])
    image_batch_size,text_batch_size=_resolve_eval_batch_sizes(dataset_name,a.image_batch_size,a.text_batch_size)
    meta,cti_meta,pos_meta=load_metadata(a.metadata); ni=int(cfg["dataset"]["expected_images"]); nc=int(cfg["dataset"]["expected_captions"])
    if len(meta["images"])!=ni or len(meta["captions"])!=nc: raise RuntimeError("metadata cardinality mismatch")
    cache=torch.load(a.clean_cache,map_location="cpu"); cti_cache,pos_cache=validate_cache(cache,ni,nc)
    if cti_cache!=cti_meta or pos_cache!=pos_meta: raise RuntimeError("clean cache/metadata mapping mismatch")
    attack=torch.load(a.image_attack_file,map_location="cpu"); adv_images=validate_attack_artifact(attack,n_images=ni,expected_dataset=dataset_name,expected_source=a.source)
    original_texts,adv_texts,text_obj=load_text_records(a.text_attack_file,nc); text_sha=sha256_file(a.text_attack_file)
    if attack.get("text_attack_sha256") and str(attack["text_attack_sha256"])!=text_sha:
        raise RuntimeError(f"text artifact SHA256 mismatch vs image artifact: expected={attack['text_attack_sha256']}, got={text_sha}")
    if attack.get("candidate_bank_sha256") and text_obj.get("candidate_bank_sha256")!=attack.get("candidate_bank_sha256"):
        raise RuntimeError("candidate-bank hash mismatch between image and text artifacts")
    expected_bank=cfg.get("text",{}).get("expected_candidate_bank_sha256")
    if expected_bank and text_obj.get("candidate_bank_sha256")!=expected_bank: raise RuntimeError("dataset candidate-bank hash mismatch")
    registry=load_evaluator_registry(a.evaluator_registry); bridge=EvaluationBridge(registry[a.victim],torch.device(a.device))
    clean_check={"performed":False}
    if not a.skip_clean_cache_check:
        paths=[(dataset_root/r["resolved_path"]).resolve() for r in meta["images"]]
        enc=bridge.encode_image_paths(paths,batch_size=image_batch_size); cached=cache["image_features"].float(); cos=(enc*cached).sum(-1)
        enc_txt=bridge.encode_texts(original_texts,batch_size=text_batch_size); txt_cos=(enc_txt*cache["text_features"].float()).sum(-1)
        clean_check={"performed":True,"image_cosine_min":float(cos.min()),"image_cosine_mean":float(cos.mean()),"text_cosine_min":float(txt_cos.min()),"text_cosine_mean":float(txt_cos.mean())}
        if float(cos.min())<0.9999: raise RuntimeError(f"clean image/cache preprocessing mismatch: {float(cos.min())}")
        if float(txt_cos.min())<0.9999: raise RuntimeError(f"clean text/cache preprocessing mismatch: {float(txt_cos.min())}")
    adv_img_feat=bridge.encode_images(adv_images,batch_size=image_batch_size); adv_txt_feat=bridge.encode_texts(adv_texts,batch_size=text_batch_size)
    metrics=summarize_joint(cache=cache,adv_image_features=adv_img_feat,adv_text_features=adv_txt_feat,source_key=a.source,victim_key=a.victim,dataset=dataset_name,attack_artifact_sha256=sha256_file(a.image_attack_file),text_artifact_sha256=text_sha)
    metrics["evaluation_batch_size"]={"images":image_batch_size,"texts":text_batch_size}
    metrics["clean_cache_check"]=clean_check; metrics["candidate_bank_sha256"]=text_obj.get("candidate_bank_sha256")
    out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2)); print("RA_SCG_FIXED_GALLERY_EVAL_STATUS=PASS")
if __name__=="__main__": main()
