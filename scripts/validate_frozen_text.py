from __future__ import annotations
import argparse, json
from pathlib import Path
from ra_scg.config import load_config, repo_root_from_config, resolve_repo_path
from ra_scg.data import load_metadata
from ra_scg.text_input import load_frozen_ba_sp, sha256_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    a=p.parse_args()
    cfg_path,cfg=load_config(a.config)
    repo=repo_root_from_config(cfg_path)
    meta,_,_=load_metadata(a.metadata)
    clean=[x['caption'] for x in meta['captions']]
    text_path=resolve_repo_path(cfg['text']['frozen_ba_sp_captions'],repo)
    obj,adv=load_frozen_ba_sp(
        text_path,
        expected_captions=int(cfg['dataset']['expected_captions']),
        expected_bank_sha256=str(cfg['text']['expected_candidate_bank_sha256']),
        clean_captions=clean,
    )
    out={
        'status':'PASS',
        'method':obj.get('method'),
        'records':len(obj.get('records',[])),
        'candidate_bank_sha256':obj.get('candidate_bank_sha256'),
        'ba_sp_file_sha256':sha256_file(text_path),
    }
    print(json.dumps(out,indent=2))
    print('RA_SCG_FROZEN_TEXT_STATUS=PASS')

if __name__=='__main__':
    main()
