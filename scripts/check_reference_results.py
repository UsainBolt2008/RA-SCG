from __future__ import annotations
import argparse,json
from pathlib import Path

KS=(1,5,10)


def _load_dataset(path: Path):
    obj=json.loads(path.read_text(encoding='utf-8'))
    return obj.get('dataset_aggregate',obj)


def main():
    p=argparse.ArgumentParser(description='Check dataset/overall BB4 against the frozen paper reference values.')
    p.add_argument('--rsitmd',type=Path,required=True)
    p.add_argument('--rsicd',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path(__file__).resolve().parents[1]/'metadata/paper_reference_results.json')
    p.add_argument('--atol',type=float,default=1e-12)
    a=p.parse_args()
    ref=json.loads(a.reference.read_text(encoding='utf-8'))
    got={'RSITMD':_load_dataset(a.rsitmd),'RSICD':_load_dataset(a.rsicd)}
    rows=[]
    for ds in ('RSITMD','RSICD'):
        vals=[float(got[ds]['cross_source_bb4']['ASR'][f'ASR@{k}']) for k in KS]
        exp=[float(x) for x in ref['dataset_cross_source_bb4'][ds]]
        diff=[abs(x-y) for x,y in zip(vals,exp)]
        rows.append({'dataset':ds,'observed':vals,'expected':exp,'abs_diff':diff})
        if max(diff)>a.atol:
            raise SystemExit(f'REFERENCE_RESULT_MISMATCH dataset={ds} max_abs_diff={max(diff)}')
    overall=[sum(float(got[d]['cross_source_bb4']['ASR'][f'ASR@{k}']) for d in ('RSITMD','RSICD'))/2 for k in KS]
    exp=[float(x) for x in ref['overall_macro']]
    diff=[abs(x-y) for x,y in zip(overall,exp)]
    print(json.dumps({'datasets':rows,'overall':{'observed':overall,'expected':exp,'abs_diff':diff}},indent=2))
    if max(diff)>a.atol:
        raise SystemExit(f'OVERALL_REFERENCE_RESULT_MISMATCH max_abs_diff={max(diff)}')
    print('RA_SCG_PAPER_REFERENCE_RESULTS_STATUS=PASS')

if __name__=='__main__':
    main()
