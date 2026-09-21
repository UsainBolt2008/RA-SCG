from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from ra_scg.config import load_config,repo_root_from_config,resolve_dataset_paths
from ra_scg.data import read_dataset
from evaluation.clean_cache import compute_i2t_ranks,compute_t2i_ranks,summarize_ranks
from evaluation.model_registry import load_evaluator_registry,EvaluationBridge,sha256_file


def main():
    p=argparse.ArgumentParser(description="Build a clean fixed-gallery cache for one paper evaluator.")
    p.add_argument("--dataset-config",type=Path,required=True); p.add_argument("--evaluator-registry",type=Path,required=True)
    p.add_argument("--model",required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda:0")
    p.add_argument("--image-batch-size",type=int,default=128); p.add_argument("--text-batch-size",type=int,default=512); a=p.parse_args()
    cfg_path,cfg=load_config(a.dataset_config); repo=repo_root_from_config(cfg_path); dataset_root,dataset_json,images_dir=resolve_dataset_paths(cfg,repo)
    meta,cti,positives=read_dataset(dataset_root,dataset_json,images_dir,split=str(cfg["dataset"].get("split","test")),capitalize_text=False,expected_captions_per_image=5)
    ni=int(cfg["dataset"]["expected_images"]); nc=int(cfg["dataset"]["expected_captions"])
    if len(meta["images"])!=ni or len(meta["captions"])!=nc: raise RuntimeError("dataset cardinality mismatch")
    registry=load_evaluator_registry(a.evaluator_registry); spec=registry[a.model]; bridge=EvaluationBridge(spec,torch.device(a.device))
    paths=[(dataset_root/r["resolved_path"]).resolve() for r in meta["images"]]
    image_features=bridge.encode_image_paths(paths,batch_size=a.image_batch_size)
    texts=[x["caption"] for x in meta["captions"]]; text_features=bridge.encode_texts(texts,batch_size=a.text_batch_size)
    scores=image_features@text_features.T; i2t=compute_i2t_ranks(scores,positives); t2i=compute_t2i_ranks(scores,cti)
    cache={"image_features":image_features,"text_features":text_features,"i2t_ranks":i2t,"t2i_ranks":t2i,
           "caption_to_image":torch.tensor(cti,dtype=torch.long),"positive_caption_indices":positives}
    metrics={"schema":"ra-scg.clean-cache.v1","dataset":cfg["dataset"]["name"],"model":a.model,"num_images":ni,"num_captions":nc,
             "checkpoint_sha256":sha256_file(spec.checkpoint),"I2T":summarize_ranks(i2t),"T2I":summarize_ranks(t2i)}
    out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); torch.save(cache,out/"features_and_ranks.pt")
    (out/"query_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2)); print("RA_SCG_PREPARE_EVAL_CACHE_STATUS=PASS")
if __name__=="__main__": main()
