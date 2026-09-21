from __future__ import annotations
from pathlib import Path
from typing import Any
import json
from .model_registry import MODEL_KEYS, SOURCE_KEYS

KS=(1,5,10); DIRECTIONS=("I2T","T2I")


def _asr(metrics: dict[str, Any], direction: str, k: int) -> float:
    node=metrics[direction][f"ASR@{k}"]
    value=node["ASR"] if isinstance(node,dict) else node
    if value is None: raise RuntimeError(f"undefined ASR for {direction}@{k}")
    return float(value)


def aggregate_source_bb4(metrics_by_victim: dict[str,dict],source_key:str,dataset:str|None=None)->dict:
    if source_key not in SOURCE_KEYS: raise KeyError(f"unsupported paper source: {source_key}")
    if set(metrics_by_victim)!=set(MODEL_KEYS):
        raise RuntimeError(f"victim set mismatch: got={sorted(metrics_by_victim)}, expected={list(MODEL_KEYS)}")
    included=[k for k in MODEL_KEYS if k!=source_key]
    if len(included)!=4: raise AssertionError("BB4 must contain four non-source victims")
    observed_datasets=set()
    for victim in MODEL_KEYS:
        m=metrics_by_victim[victim]
        if m.get("source_key")!=source_key or m.get("victim_key")!=victim:
            raise RuntimeError(f"source/victim identity mismatch for {victim}")
        if m.get("method")!="RA-SCG": raise RuntimeError(f"method mismatch for {victim}")
        observed_datasets.add(str(m.get("dataset")))
    if len(observed_datasets)!=1: raise RuntimeError(f"mixed datasets in source aggregate: {observed_datasets}")
    observed_dataset=next(iter(observed_datasets))
    if dataset is not None and observed_dataset.lower()!=dataset.lower():
        raise RuntimeError(f"dataset mismatch: metrics={observed_dataset}, requested={dataset}")
    out={"dataset":observed_dataset,"source_key":source_key,"excluded_source_victim":source_key,
         "included_victims":included,"directions":list(DIRECTIONS),"terms_per_k":8,"ASR":{},"terms":{}}
    for k in KS:
        terms=[]
        for victim in included:
            for direction in DIRECTIONS:
                terms.append({"victim_key":victim,"direction":direction,"K":k,
                              "ASR":_asr(metrics_by_victim[victim],direction,k)})
        if len(terms)!=8: raise AssertionError("BB4 source aggregate requires exactly eight terms")
        out["terms"][f"ASR@{k}"]=terms
        out["ASR"][f"ASR@{k}"]=sum(x["ASR"] for x in terms)/8.0
    return out


def aggregate_cross_source(source_aggregates:dict[str,dict])->dict:
    if set(source_aggregates)!=set(SOURCE_KEYS):
        raise RuntimeError(f"cross-source aggregate requires exactly {SOURCE_KEYS}")
    ds={str(x["dataset"]) for x in source_aggregates.values()}
    if len(ds)!=1: raise RuntimeError(f"mixed datasets across sources: {ds}")
    out={"dataset":next(iter(ds)),"sources":list(SOURCE_KEYS),
         "definition":"mean of the two source-specific BB4 macros","ASR":{}}
    for k in KS:
        vals=[float(source_aggregates[s]["ASR"][f"ASR@{k}"]) for s in SOURCE_KEYS]
        out["ASR"][f"ASR@{k}"]=sum(vals)/2.0
    return out


def aggregate_dataset_from_paths(dataset:str,source_dirs:dict[str,Path])->dict:
    if set(source_dirs)!=set(SOURCE_KEYS):
        raise RuntimeError(f"need exactly source dirs for {SOURCE_KEYS}")
    source_aggs={}
    for source in SOURCE_KEYS:
        directory=Path(source_dirs[source]); metrics={}
        for victim in MODEL_KEYS:
            path=directory/victim/"metrics.json"
            if not path.is_file(): raise FileNotFoundError(path)
            metrics[victim]=json.loads(path.read_text(encoding="utf-8"))
        source_aggs[source]=aggregate_source_bb4(metrics,source,dataset)
    return {"schema":"ra-scg.bb4.v1","dataset":dataset,"source_bb4":source_aggs,
            "cross_source_bb4":aggregate_cross_source(source_aggs)}


def aggregate_overall(dataset_aggregates:dict[str,dict])->dict:
    if not dataset_aggregates: raise RuntimeError("no dataset aggregates supplied")
    out={"schema":"ra-scg.overall-bb4.v1","datasets":sorted(dataset_aggregates),
         "definition":"macro mean of dataset-level cross-source BB4","ASR":{}}
    for key,obj in dataset_aggregates.items():
        if str(obj.get("dataset")).lower()!=str(key).lower():
            raise RuntimeError(f"dataset aggregate identity mismatch for {key}")
    for k in KS:
        vals=[float(obj["cross_source_bb4"]["ASR"][f"ASR@{k}"]) for obj in dataset_aggregates.values()]
        out["ASR"][f"ASR@{k}"]=sum(vals)/len(vals)
    return out
