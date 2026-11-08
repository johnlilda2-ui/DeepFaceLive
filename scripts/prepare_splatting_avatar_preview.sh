#!/usr/bin/env bash
# Prepare the public sample and GPU renderer. Supply the licensed SMPL model separately.
set -euo pipefail
root="$(realpath -m "${1:?Usage: bash scripts/prepare_splatting_avatar_preview.sh NEW_RUNTIME_DIRECTORY}")"
mkdir "$root"
cd "$root"

checkout() {
    git init -q "$1"
    git -C "$1" fetch --depth 1 "$2" "$3"
    git -C "$1" checkout --quiet --detach FETCH_HEAD
}
checkout source https://github.com/initialneil/SplattingAvatar.git 6fe2554628429ae996744cff6fa4b98f4111b7c5
checkout raster https://github.com/graphdeco-inria/diff-gaussian-rasterization.git 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
git -C raster submodule update --init --depth 1
sed -i '1i#include <cstdint>' raster/cuda_rasterizer/rasterizer_impl.h

python -m venv --without-pip --system-site-packages env
env/bin/python -m pip install omegaconf plyfile trimesh libigl
env/bin/python -c 'import torch; assert torch.cuda.is_available(), "Start a CUDA GPU session first."'
TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=2 env/bin/python -m pip install --no-build-isolation ./raster

curl -fLsS --retry 2 'https://drive.usercontent.google.com/download?id=1efKzm1AyDNoC1CbYfZdytXlocasumdm0&export=download&confirm=t' -o people.zip
mkdir sample
subject=peoplesnapshot/female-3-casual
checkpoint="$subject/output-splatting/last_checkpoint/point_cloud/iteration_30000"
unzip -j people.zip "$checkpoint/embedding.json" "$checkpoint/embedding.obj" "$checkpoint/point_cloud.ply" "$subject/poses.npz" -d sample
rm people.zip

transforms=imports/pytorch3d/transforms
mkdir -p "$transforms"
touch imports/pytorch3d/__init__.py
printf 'from .rotation_conversions import *\n' > "$transforms/__init__.py"
curl -fLsS --retry 2 https://raw.githubusercontent.com/facebookresearch/pytorch3d/e4ad6e5687a360ccdc6ebd91348e8dbbac97bf31/pytorch3d/transforms/rotation_conversions.py -o rotation.py
printf '%s  %s\n' e411e6506b158ed0485d89c0726b6f71aad11556d666e4b82cac8a6105b47381 rotation.py | sha256sum -c -
sed 's/from \.\.common\.datatypes import Device/from typing import Union\nDevice = Union[str, torch.device]/' rotation.py > "$transforms/rotation_conversions.py"
rm rotation.py
