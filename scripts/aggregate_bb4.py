from __future__ import annotations
import argparse,json
from pathlib import Path
from evaluation.bb4 import aggregate_dataset_from_paths,aggregate_overall
from evaluation.model_registry import SOURCE_KEYS

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--source-dir",action="append",default=[]); p.add_argument("--output",type=Path,required=True); p.add_argument("--other-dataset-aggregate",action="append",default=[]); a=p.parse_args()
    mapping={}
    for item in a.source_dir:
        key,value=item.split("=",1); mapping[key]=Path(value)
    if set(mapping)!=set(SOURCE_KEYS): raise RuntimeError(f"need exactly source dirs for {SOURCE_KEYS}")
    dataset=aggregate_dataset_from_paths(a.dataset,mapping); payload={"dataset_aggregate":dataset}
    if a.other_dataset_aggregate:
        datasets={a.dataset:dataset}
        for item in a.other_dataset_aggregate:
            obj=json.loads(Path(item).read_text(encoding="utf-8")); other=obj.get("dataset_aggregate",obj); datasets[str(other["dataset"])]=other
        payload["overall"]=aggregate_overall(datasets)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2)); print("RA_SCG_BB4_AGGREGATION_STATUS=PASS")
if __name__=="__main__": main()
