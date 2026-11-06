"""Download only pinned official human-portrait weights; never start compute hosting."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

CODE_REVISION = '9b294b3d0536135442ea73cb01e6cb3ca7029dd3'
WEIGHTS_REVISION = '82a4fa6735ca58432b6ce39301b4b9ee066dea47'
WEIGHT_FILES = [
    'liveportrait/base_models/appearance_feature_extractor.pth',
    'liveportrait/base_models/motion_extractor.pth',
    'liveportrait/base_models/warping_module.pth',
    'liveportrait/base_models/spade_generator.pth',
    'liveportrait/retargeting_models/stitching_retargeting_module.pth',
]


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-dir',required=True)
    args = parser.parse_args()
    root = Path(args.runtime_dir).resolve()
    if not root.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',
                        'https://github.com/KlingAIResearch/LivePortrait',str(root)],check=True)
    if not (root/'.git').is_dir():
        raise ValueError('Portrait runtime must be its own official dependency checkout.')
    origin = subprocess.check_output(['git','-C',str(root),'remote','get-url','origin'],text=True).strip().removesuffix('.git').rstrip('/')
    if origin not in ('https://github.com/KlingAIResearch/LivePortrait',
                      'https://github.com/KlingTeam/LivePortrait','https://github.com/KwaiVGI/LivePortrait'):
        raise ValueError('Refusing to change a checkout that is not the official portrait runtime.')
    subprocess.run(['git','-C',str(root),'fetch','--depth','1','origin',CODE_REVISION],check=True)
    subprocess.run(['git','-C',str(root),'checkout','--detach',CODE_REVISION],check=True)
    with urllib.request.urlopen('https://huggingface.co/api/models/KlingTeam/LivePortrait/revision/'+WEIGHTS_REVISION+'?blobs=true',timeout=45) as response:
        info = json.load(response)
    if info['sha'] != WEIGHTS_REVISION:
        raise RuntimeError('Unexpected model revision.')
    siblings = {item['rfilename']:item for item in info['siblings']}
    manifest = {'code_revision':CODE_REVISION,'weights_revision':WEIGHTS_REVISION,'files':{}}
    for filename in WEIGHT_FILES:
        expected = siblings[filename]['lfs']['sha256']
        destination = root/'pretrained_weights'/filename
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.is_file() or digest(destination) != expected:
            partial = destination.with_suffix('.download')
            try:
                with urllib.request.urlopen('https://huggingface.co/KlingTeam/LivePortrait/resolve/'+WEIGHTS_REVISION+'/'+filename,timeout=120) as response, partial.open('wb') as handle:
                    for chunk in iter(lambda:response.read(1024*1024),b''):
                        handle.write(chunk)
                if digest(partial) != expected:
                    raise RuntimeError('Model SHA256 mismatch: '+filename)
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)
        manifest['files'][filename] = expected
        print('VERIFIED',filename,destination.stat().st_size,flush=True)
    (root/'portrait-runtime-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PORTRAIT_WEIGHTS_READY',flush=True)


if __name__ == '__main__':
    main()
