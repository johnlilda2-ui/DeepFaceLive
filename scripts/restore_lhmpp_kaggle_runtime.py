#!/usr/bin/env python3
"""Restore the previously verified experimental runtime after a Kaggle restart.

Requires the retained upstream checkout, installed Python 3.10, package freeze
and asset manifest from the earlier successful test. Leaves face/desktop envs alone.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def ensure_prior_link(prior_link, target):
    """Repair Kaggle's observed empty-file serialization without deleting entries."""
    target = target.resolve(strict=True)
    if not target.is_dir():
        raise RuntimeError('Prior target must be an existing directory')
    if os.path.lexists(prior_link):
        if prior_link.is_symlink() and prior_link.resolve() == target:
            return
        if (not prior_link.is_symlink() and prior_link.is_file()
                and prior_link.stat().st_size == 0):
            backup = prior_link.with_name(prior_link.name + '.kaggle-empty-backup')
            if os.path.lexists(backup):
                # Kaggle can serialize this symlink as an empty file again on
                # later restarts. Preserve every older backup, then reserve a
                # new name instead of repeatedly blocking the known repair.
                for suffix in range(1,1001):
                    candidate = prior_link.with_name(prior_link.name + f'.kaggle-empty-backup-{suffix}')
                    if not os.path.lexists(candidate):
                        backup = candidate
                        break
                else:
                    raise RuntimeError('Too many prior placeholder backups; left intact')
            prior_link.rename(backup)
            print('PRESERVED_KAGGLE_EMPTY_PRIOR_ENTRY', backup, flush=True)
        else:
            raise RuntimeError('Existing experimental prior path points elsewhere; left intact')
    prior_link.symlink_to(target, target_is_directory=True)


def main():
    import requests
    root = Path('/kaggle/working/dfl-fullbody-runtime')
    evidence = Path('/kaggle/working/dfl-fullbody-evidence')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if revision != '906b5d9fb967ab42efb92f6fa55bf22cac86b653':
        raise RuntimeError('Unexpected upstream revision')
    env_python = '/tmp/dfl-fullbody-env/bin/python'
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH='/usr/local/nvidia/lib64', TORCH_CUDA_ARCH_LIST='7.5',
               MAX_JOBS='2', UV_CACHE_DIR='/tmp/dfl-fullbody-package-cache',
               UV_PYTHON_INSTALL_DIR='/tmp/dfl-fullbody-python')
    def run(command, cwd=root, timeout=240):
        print('RUN', command[:5], flush=True)
        subprocess.run(command, cwd=cwd, env=env, check=True, timeout=timeout)
    if not Path(env_python).exists():
        # Restored /kaggle/working snapshots may lose executable permissions.
        # Restore only the experimental interpreter on the executable temp disk.
        run(['uv', 'python', 'install', '3.10'], timeout=90)
        python = next(Path('/tmp/dfl-fullbody-python').glob('cpython-3.10*/bin/python3.10'))
        run(['uv', 'venv', '--python', str(python), '/tmp/dfl-fullbody-env'], timeout=30)
    run(['uv', 'pip', 'install', '--python', env_python, 'pip==25.3', 'setuptools==74.0.0', 'wheel==0.43.0'])
    excluded = ('pointops', 'diff-gaussian-rasterization', 'simple-knn')
    lines = (evidence / 'experimental-requirements.txt').read_text().splitlines()
    filtered = [line for line in lines if not line.lower().replace('_', '-').startswith(excluded)]
    requirements = Path('/tmp/dfl-fullbody-restore-requirements.txt')
    requirements.write_text('\n'.join(filtered) + '\n')
    run(['uv', 'pip', 'install', '--python', env_python, '--no-build-isolation-package', 'chumpy',
         '-r', str(requirements), '--find-links', 'https://data.pyg.org/whl/torch-2.3.0+cu121.html',
         '--find-links', 'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html',
         '--find-links', 'https://docs.gsplat.studio/whl/gsplat/'])
    # Reuse the matching compiled pointops artifact retained in /kaggle/working.
    compiled = list((root / 'lib/pointops/build').glob('lib*/pointops_cuda*.so'))
    if compiled:
        site = subprocess.check_output([env_python, '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
        shutil.copy2(compiled[0], Path(site) / compiled[0].name)
        print('REUSED_POINTOPS_BINARY', compiled[0].name, flush=True)
    else:
        run([env_python, 'setup.py', 'install'], cwd=root / 'lib/pointops')
    wheels = Path('/kaggle/working/dfl-fullbody-wheels')
    wheels.mkdir(exist_ok=True)
    for name, repo, sha in [
        ('diff_gaussian_rasterization', 'ashawkey/diff-gaussian-rasterization', '8829d14f814fccdaf840b7b0f3021a616583c0a1'),
        ('simple_knn', 'camenduru/simple-knn', '60f461f4a56b7967e5d8045bf92f8c33f36976d0')]:
        candidates = list(wheels.glob(name + '-*.whl'))
        if not candidates:
            source = Path('/tmp/dfl-fullbody-sources') / name
            if not source.exists():
                run(['git', 'clone', '--depth', '1', 'https://github.com/' + repo + '.git', str(source)], timeout=90)
            run(['git', 'fetch', '--depth', '1', 'origin', sha], cwd=source, timeout=90)
            run(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=source)
            run(['git', 'submodule', 'update', '--init', '--recursive', '--depth', '1'], cwd=source, timeout=90)
            run([env_python, '-m', 'pip', 'wheel', '--no-build-isolation', '--no-deps',
                 '--wheel-dir', str(wheels), str(source)], cwd=source)
            candidates = list(wheels.glob(name + '-*.whl'))
        if len(candidates) != 1:
            raise RuntimeError('Expected exactly one pinned extension wheel for ' + name)
        run(['uv', 'pip', 'install', '--python', env_python, '--no-deps', str(candidates[0])])
    manifest = json.loads((evidence / 'asset-manifest.json').read_text())
    assets = Path('/tmp/dfl-fullbody-assets')
    revisions = {'3DAIGC/LHMPP-700M': 'fc9f73664b9bfcc457e96210ddc06fe7caf0d559',
                 '3DAIGC/LHMPP-Prior': 'b683c8f68bede4f318b0bb539730b8e6711d30a0'}
    def file_sha256(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    def download(item):
        if revisions.get(item['repo']) != item['revision']:
            raise RuntimeError('Unexpected asset repository/revision')
        folder = 'model' if item['repo'] == '3DAIGC/LHMPP-700M' else 'prior'
        target = assets / folder / item['path']
        if not target.resolve().is_relative_to((assets / folder).resolve()):
            raise RuntimeError('Asset path must stay inside experimental storage')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_sha256(target) == item['sha256']:
            return
        url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{item['path']}"
        part = target.with_suffix(target.suffix + '.part')
        digest = hashlib.sha256()
        with requests.get(url, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with part.open('wb') as output:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        digest.update(chunk)
        if part.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise RuntimeError('Asset integrity check failed: ' + item['path'])
        part.replace(target)
        print('RESTORED_ASSET', item['path'], flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, manifest))
    prior_link = root / 'pretrained_models'
    ensure_prior_link(prior_link, assets / 'prior')
    if not (prior_link / 'human_model_files/pose_estimate/multiHMR_896_L.pt').is_file():
        raise RuntimeError('Restored pose asset is not accessible from the runtime')
    motion = Path('/tmp/dfl-fullbody-motion')
    if not (motion / 'motion_video/BasketBall_I/smplx_params').exists():
        url = 'https://huggingface.co/3DAIGC/LHMPP-Assets/resolve/c50158bda6b0f2fafec0d81ecdfaaf0430c44d60/motion_video.tar'
        archive = Path('/tmp/dfl-fullbody-motion.tar')
        digest = hashlib.sha256()
        with requests.get(url, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with archive.open('wb') as output:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        digest.update(chunk)
        if digest.hexdigest() != 'e2612a18c0def13cf1834951155ac8d9c28529a9bd1ac6906fe1462160c4c8bc':
            raise RuntimeError('Motion checksum failed')
        motion.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(motion, filter='data')
    print('EXPERIMENTAL_RUNTIME_RESTORED', flush=True)


if __name__ == '__main__':
    main()

