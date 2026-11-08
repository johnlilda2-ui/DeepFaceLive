"""Render the pinned SplattingAvatar PeopleSnapshot checkpoint in a GPU child."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


class SplattingAvatarBackend:
    def __init__(self, config):
        self.config = config
        self.stats = {}
        self.model = self.body = None

    def prepare(self, pose):
        import torch
        from omegaconf import OmegaConf
        import igl

        root = Path(self.config['runtime_root']).resolve()
        checkpoint = Path(self.config['checkpoint']).resolve()
        body_path = Path(self.config['smpl_model']).resolve()
        if not body_path.is_file():
            raise ValueError('Supply the official female SMPL model as smpl_model.')
        if not torch.cuda.is_available():
            raise RuntimeError('SplattingAvatar requires CUDA.')
        if not hasattr(igl, 'read_obj'):
            igl.read_obj = igl.readOBJ
        sys.path[:0] = [str(root / 'imports'), str(root / 'source')]
        from gaussian_renderer import render
        from model.smplx_utils.smplx import SMPL
        from scene.cameras import MiniCam
        from utils.graphics_utils import getProjectionMatrix2

        # Training-only dependencies are unnecessary when loading a checkpoint.
        source = root / 'source/model/splatting_avatar_model.py'
        code = source.read_text()
        for line in (
            'from simple_phongsurf import PhongSurfacePy3d',
            'from simple_knn._C import distCUDA2',
            "        self.phongsurf = PhongSurfacePy3d(cano_verts, cano_faces, cano_norms,\n"
            "                                          outer_loop=2, inner_loop=50, method='uvd').to(self.device)",
        ):
            if code.count(line) != 1:
                raise RuntimeError('Unexpected SplattingAvatar source version.')
            code = code.replace(line, '')
        spec = importlib.util.spec_from_file_location('model.splatting_avatar_preview', source)
        module = importlib.util.module_from_spec(spec)
        exec(compile(code, str(source), 'exec'), module.__dict__)

        self.torch, self.render_model = torch, render
        self.model = module.SplattingAvatarModel(OmegaConf.create({'sh_degree': 0}))
        self.model.load_ply(str(checkpoint / 'point_cloud.ply'))
        self.model.load_from_embedding(str(checkpoint / 'embedding.json'))
        with np.load(self.config['sample_poses'], allow_pickle=False) as sample:
            self.betas = self.tensor(sample['betas']).reshape(1, 10)
        try:
            self.body = SMPL(str(body_path), gender='female', num_betas=10, batch_size=1).to('cuda').eval()
        except ModuleNotFoundError as error:
            if error.name == 'chumpy':
                raise ValueError('Prepare a Chumpy-free copy of the official SMPL model.') from error
            raise
        if not np.array_equal(self.body.faces, self.model.cano_faces.cpu().numpy()):
            raise ValueError('SMPL face topology does not match the SplattingAvatar sample.')
        self.face_indices = self.model.cano_faces.flatten()
        width, height = map(int, pose['img_size_wh'])
        fx, fy = pose['focal']
        cx, cy = pose['princpt']
        projection = getProjectionMatrix2(width, height, fx, fy, cx, cy, .1, 20).cuda().T
        self.camera = MiniCam(width, height, 2 * np.arctan(height / (2 * fy)),
                              2 * np.arctan(width / (2 * fx)), .1, 20,
                              torch.eye(4, device='cuda'), projection)
        self.pipe = SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
        self.black = torch.zeros(3, device='cuda')
        self.white = torch.ones(3, device='cuda')
        self.output_shape = (height, width, 4)
        if np.count_nonzero(self.render(pose)[..., 3] > 32) < width * height * .01:
            raise ValueError('Initial pose places the SplattingAvatar sample outside the preview.')
        self.stats = {'gpu': torch.cuda.get_device_name(0), 'backend': 'splattingavatar'}

    def tensor(self, value):
        return self.torch.as_tensor(value, device='cuda', dtype=self.torch.float32)

    def render(self, pose):
        torch = self.torch
        with torch.no_grad():
            body_pose = torch.nn.functional.pad(self.tensor(pose['body_pose']).reshape(1, 63), (0, 6))
            result = self.body(betas=self.betas, body_pose=body_pose,
                               global_orient=self.tensor(pose['root_pose']).reshape(1, 3),
                               transl=torch.zeros((1, 3), device='cuda'))
            verts = result.vertices[0] - result.joints[0, 0] + self.tensor(pose['trans'])
            a, b, c = verts[self.model.cano_faces].unbind(1)
            face_normals = torch.linalg.cross(b - a, c - a)
            normals = torch.zeros_like(verts).index_add_(
                0, self.face_indices, face_normals.repeat_interleave(3, dim=0))
            self.model.update_to_posed_mesh({
                'mesh_verts': verts,
                'mesh_norms': torch.nn.functional.normalize(normals, dim=-1),
            })
            premultiplied = self.render_model(self.camera, self.model, self.pipe, self.black)['render']
            on_white = self.render_model(self.camera, self.model, self.pipe, self.white)['render']
            alpha = (1 - (on_white - premultiplied).mean(dim=0, keepdim=True)).clamp(0, 1)
            rgba = torch.cat((premultiplied, alpha)).clamp(0, 1)
            return rgba.mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()

    def close(self):
        self.model = self.body = None
