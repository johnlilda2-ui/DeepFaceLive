"""Fit a private lip-expression map from manifest-verified prepared assets."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from avatar_lip_expression import LIP_CONTROLS, fit_lip_basis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('Choose a new output; existing map left intact')
    manifest=json.loads(a.manifest.read_text())
    def verified(name):
        rel='human_model_files/'+name
        matches=[x for x in manifest if x['path']==rel]
        path=a.prior/rel
        data=path.read_bytes()
        if len(matches)!=1 or hashlib.sha256(data).hexdigest()!=matches[0]['sha256']:
            raise ValueError('Prepared asset checksum mismatch: '+rel)
        return path, matches[0]['sha256']
    flame,flame_sha=verified('flame/FLAME_NEUTRAL.pkl')
    arkit,arkit_sha=verified('flame_assets/flame_arkit_bs.npy')
    order,order_sha=verified('flame_assets/pred_expression.json')
    # Trusted, manifest-verified model pickle only. Never unpickle user uploads.
    with flame.open('rb') as f:
        model=pickle.load(f,encoding='latin1')
    dirs=np.asarray(model['shapedirs'])[:,:,300:400]
    names=json.loads(order.read_text())['names']
    offsets=np.load(arkit,allow_pickle=False)
    selected=[names.index(n) for n in LIP_CONTROLS]
    projection,metadata=fit_lip_basis(dirs,offsets[selected],LIP_CONTROLS)
    metadata.update(flame_sha256=flame_sha,arkit_sha256=arkit_sha,order_sha256=order_sha,
                    visual_quality_verified=False,private_derived_asset=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:
        np.savez_compressed(f,projection=projection,metadata=json.dumps(metadata))
    print('LIP_BASIS_REPORT',json.dumps(metadata),flush=True)


if __name__=='__main__':
    main()
