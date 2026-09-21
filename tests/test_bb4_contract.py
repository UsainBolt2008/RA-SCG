from evaluation.bb4 import aggregate_source_bb4,aggregate_cross_source,aggregate_overall
from evaluation.model_registry import MODEL_KEYS,SOURCE_KEYS

def _metric(source,victim,base,dataset="SYNTH"):
    return {"method":"RA-SCG","dataset":dataset,"source_key":source,"victim_key":victim,
            "I2T":{f"ASR@{k}":{"ASR":base+k} for k in (1,5,10)},"T2I":{f"ASR@{k}":{"ASR":base+k+0.5} for k in (1,5,10)}}

def test_bb4_excludes_source_and_has_eight_terms():
    s=SOURCE_KEYS[0]; metrics={v:_metric(s,v,10+i) for i,v in enumerate(MODEL_KEYS)}; out=aggregate_source_bb4(metrics,s,"SYNTH")
    assert out["excluded_source_victim"]==s and s not in out["included_victims"] and len(out["included_victims"])==4
    for k in (1,5,10): assert len(out["terms"][f"ASR@{k}"])==8

def test_cross_source_and_overall_contract():
    aggs={}
    for si,s in enumerate(SOURCE_KEYS): aggs[s]=aggregate_source_bb4({v:_metric(s,v,20+si+i) for i,v in enumerate(MODEL_KEYS)},s,"SYNTH")
    cross=aggregate_cross_source(aggs); assert cross["sources"]==list(SOURCE_KEYS)
    ds={"SYNTH":{"dataset":"SYNTH","cross_source_bb4":cross},"OTHER":{"dataset":"OTHER","cross_source_bb4":dict(cross,dataset="OTHER")}}
    out=aggregate_overall(ds); assert set(out["datasets"])=={"SYNTH","OTHER"}
