"""Independent CUDA-12 portrait venv; never mutate Kaggle or the face-swap env."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

from prepare_headless_env import ensure_environment

PACKAGES = [
    'onnxruntime-gpu==1.22.0','onnx==1.18.0','numpy==2.2.6',
    'opencv-python-headless==4.11.0.86','wrapt==1.17.2','h5py==3.12.1',
    'numexpr==2.10.2','psutil==7.0.0','Pillow==11.3.0','PyYAML==6.0.2',
    'rich==13.9.4','aiohttp==3.13.3','aiortc==1.15.0','av==17.0.0','pyee==13.0.1',
    'scipy==1.15.3',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    interpreter = ensure_environment(directory)
    # Repaired interpreter copies stay intact; retain installed packages on restart.
    venv.EnvBuilder(system_site_packages=False,with_pip=False,symlinks=False).create(directory)
    subprocess.run([sys.executable,'-m','pip','--python',str(interpreter),'install','--no-cache-dir',
                    'torch==2.9.1','--index-url','https://download.pytorch.org/whl/cu128'],check=True)
    subprocess.run([sys.executable,'-m','pip','--python',str(interpreter),'install','--no-cache-dir',*PACKAGES],check=True)
    subprocess.run([str(interpreter),'-c',
                    "import torch; assert torch.__version__.startswith('2.9.1'); assert torch.version.cuda.startswith('12.'); assert torch.cuda.is_available(); x=torch.ones((64,64),device='cuda'); y=x@x; torch.cuda.synchronize(); print('PORTRAIT_TORCH_CUDA_READY',torch.__version__,y.device)"],check=True)
    print(interpreter)


if __name__ == '__main__':
    main()
