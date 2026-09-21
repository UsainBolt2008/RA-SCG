from __future__ import annotations
import json
import tempfile
from pathlib import Path
from ra_scg.data import read_dataset, validate_metadata


def test_portable_dataset_loader():
    with tempfile.TemporaryDirectory() as td:
        tmp_path=Path(td)
        root=tmp_path/'D'; images=root/'images'; images.mkdir(parents=True)
        for name in ('a.jpg','b.jpg'):
            (images/name).write_bytes(b'x')
        payload={'images':[]}
        for idx,name in enumerate(('a.jpg','b.jpg')):
            payload['images'].append({
                'split':'test','filename':name,'imgid':idx,
                'sentences':[{'raw':f'caption {idx}-{j}'} for j in range(5)]
            })
        j=root/'dataset.json'; j.write_text(json.dumps(payload),encoding='utf-8')
        meta,cti,pos=read_dataset(root,j,images,capitalize_text=True)
        assert len(meta['images'])==2 and len(meta['captions'])==10
        assert meta['images'][0]['resolved_path']=='images/a.jpg'
        assert meta['captions'][0]['caption']=='Caption 0-0'
        cti2,pos2=validate_metadata(meta)
        assert cti2==cti and pos2==pos
