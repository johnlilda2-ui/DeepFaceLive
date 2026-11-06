"""Optional portrait animation worker; face-swap and desktop paths stay independent.

Uses pinned official LivePortrait code/weights and this project's detector/alignment.
Output is the uploaded portrait's canvas, never the driver's hair/body/background.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import headless_inference_smoke_test as smoke
from headless_cloud_worker import validate_frame

UPSTREAM_REVISION = '9b294b3d0536135442ea73cb01e6cb3ca7029dd3'


def paste_portrait(source, generated, transform):
    """Paste the generated head crop onto the SOURCE portrait, not the driver."""
    import cv2
    import numpy as np
    h, w = source.shape[:2]
    size = generated.shape[0]
    if generated.shape != (size, size, 3) or generated.dtype != np.uint8:
        raise ValueError('Invalid portrait decoder output.')
    mask = np.zeros((size, size), np.float32)
    margin = max(2, size // 24)
    mask[margin:-margin, margin:-margin] = 1
    sigma = max(1, size / 64)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    matrix = transform.invert().to_exact_mat(size, size, w, h)
    alpha = cv2.warpAffine(mask, matrix, (w, h))
    face = cv2.warpAffine(generated.astype(np.float32) * mask[..., None], matrix, (w, h))
    return np.clip(np.rint(source * (1-alpha[..., None]) + face), 0, 255).astype(np.uint8)


class HeadlessPortraitWorker:
    mode = 'portrait'

    def __init__(self, source_image, evidence_dir, runtime_dir):
        import subprocess
        import torch
        import cv2
        self.closed = False
        self.frame_count = 0
        self.threshold = .5
        self.runtime_dir = Path(runtime_dir).resolve()
        revision = subprocess.check_output(['git','-C',str(self.runtime_dir),'rev-parse','HEAD'],text=True).strip()
        if revision != UPSTREAM_REVISION:
            raise ValueError('Portrait runtime revision does not match the reviewed version.')
        subprocess.run(['git','-C',str(self.runtime_dir),'diff','--exit-code'],check=True,capture_output=True)
        from prepare_portrait_runtime import digest, WEIGHTS_REVISION, WEIGHT_FILES
        manifest = json.loads((self.runtime_dir/'portrait-runtime-manifest.json').read_text())
        if (manifest['weights_revision'] != WEIGHTS_REVISION or manifest['code_revision'] != revision or
                set(manifest['files']) != set(WEIGHT_FILES)):
            raise ValueError('Unexpected portrait weight manifest.')
        for filename, expected in manifest['files'].items():
            if digest(self.runtime_dir/'pretrained_weights'/filename) != expected:
                raise ValueError('Portrait weight checksum mismatch: '+filename)
        self.weight_manifest = manifest
        if not torch.cuda.is_available():
            raise RuntimeError('Portrait mode requires actual CUDA; CPU fallback is disabled.')
        sys.path.insert(0, str(self.runtime_dir))
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.camera import get_rotation_matrix
        self.rotation = get_rotation_matrix
        self.torch = torch
        Path(evidence_dir).resolve().mkdir(parents=True,exist_ok=True)
        self.profile_dir = Path(tempfile.mkdtemp(prefix='portrait-',dir=Path(evidence_dir).resolve()))
        device = smoke.choose_device(True)
        from modelhub.onnx import YoloV5Face, InsightFace2D106
        with smoke.profiled_sessions(self.profile_dir, True):
            self.detector, self.marker = YoloV5Face(device), InsightFace2D106(device)
        for model in (self.detector,self.marker):
            if 'CUDAExecutionProvider' not in model._sess.get_providers():
                raise RuntimeError('Portrait face alignment did not activate CUDA.')
        self.wrapper = LivePortraitWrapper(InferenceConfig(flag_use_half_precision=True))
        self.cuda_calls = {}
        self.hooks = []
        modules = {'appearance':self.wrapper.appearance_feature_extractor,
                   'motion':self.wrapper.motion_extractor,'warp':self.wrapper.warping_module,
                   'decoder':self.wrapper.spade_generator,
                   'stitching':self.wrapper.stitching_retargeting_module['stitching']}
        for label, module in modules.items():
            if not all(parameter.is_cuda for parameter in module.parameters()):
                raise RuntimeError('Portrait model was not placed on CUDA: '+label)
            def observe(module, inputs, output, label=label):
                def tensors(value):
                    if isinstance(value, torch.Tensor):
                        return [value]
                    if isinstance(value, dict):
                        return [t for v in value.values() for t in tensors(v)]
                    if isinstance(value, (tuple,list)):
                        return [t for v in value for t in tensors(v)]
                    return []
                values = tensors(output)
                if not values or not all(value.is_cuda for value in values):
                    raise RuntimeError('Portrait output was not on CUDA: '+label)
                self.cuda_calls[label] = self.cuda_calls.get(label,0)+1
            self.hooks.append(module.register_forward_hook(observe))
        source = cv2.imread(str(source_image))
        if source is None:
            raise ValueError('Cannot decode source portrait.')
        self.set_identity(source)

    def set_identity(self, image):
        if self.closed:
            raise RuntimeError('Worker is closed.')
        validate_frame(image)
        if len(self.detector.extract(image,threshold=self.threshold)[0]) != 1:
            raise ValueError('Choose a portrait with exactly one visible face.')
        crop, transform = smoke.detect_and_align_face(image,'portrait',self.detector,self.marker,
                                                     2.3,512,self.threshold,return_transform=True)
        with self.torch.inference_mode():
            prepared = self.wrapper.prepare_source(crop[..., ::-1])
            info = self.wrapper.get_kp_info(prepared)
            feature = self.wrapper.extract_feature_3d(prepared)
            keypoints = self.wrapper.transform_keypoint(info)
            rotation = self.rotation(info['pitch'],info['yaw'],info['roll'])
        # Commit only after all validation and CUDA inference succeeded.
        self.source = image.copy()
        self.transform = transform
        self.source_info, self.source_feature = info, feature
        self.source_keypoints, self.source_rotation = keypoints, rotation
        self.reset_motion()

    def reset_motion(self):
        self.driving_initial = None
        self.last_output = self.source.copy()

    def process_frame(self, frame, preserve_mouth=False):
        if self.closed:
            raise RuntimeError('Worker is closed.')
        if preserve_mouth:
            raise ValueError('Portrait mode animates portrait lips; real-mouth overlay is face-swap only.')
        validate_frame(frame)
        try:
            crop = smoke.detect_and_align_face(frame,'driver',self.detector,self.marker,
                                              2.3,256,self.threshold)
        except smoke.NoFaceDetected:
            return self.last_output.copy(), 'avatar_no_face'
        with self.torch.inference_mode():
            driving = self.wrapper.get_kp_info(self.wrapper.prepare_source(crop[..., ::-1]))
            rotation = self.rotation(driving['pitch'],driving['yaw'],driving['roll'])
            if self.driving_initial is None:
                self.driving_initial = driving
                self.driving_rotation = rotation
            initial, source = self.driving_initial, self.source_info
            relative_rotation = rotation @ self.driving_rotation.permute(0,2,1) @ self.source_rotation
            expression = source['exp'] + driving['exp'] - initial['exp']
            scale = source['scale'] * (driving['scale'] / initial['scale'])
            translation = (source['t'] + driving['t'] - initial['t']).clone()
            translation[...,2] = 0
            keypoints = scale[...,None] * (source['kp'] @ relative_rotation + expression) + translation[:,None,:]
            keypoints = self.wrapper.stitching(self.source_keypoints,keypoints)
            decoded = self.wrapper.warp_decode(self.source_feature,self.source_keypoints,keypoints)
            generated = self.wrapper.parse_output(decoded['out'])[0][..., ::-1].copy()
        output = paste_portrait(self.source,generated,self.transform)
        self.last_output = output
        self.frame_count += 1
        return output, 'avatar'

    def close(self):
        if self.closed:
            return
        self.closed = True
        evidence = {'upstream_revision':UPSTREAM_REVISION,'torch':self.torch.__version__,
                    'weight_manifest':self.weight_manifest,
                    'device':self.torch.cuda.get_device_name(0),'cuda_model_calls':self.cuda_calls,
                    'frames':self.frame_count,'models':{}}
        for label, model in (('detector',self.detector),('landmarks',self.marker)):
            profile = model._sess.end_profiling()
            counts = dict(smoke.profile_counts(profile))
            evidence['models'][label] = {'profile':profile,'executed_nodes':counts}
        evidence['cuda_verified'] = (all(self.cuda_calls.get(label,0)>0 for label in
                                    ('appearance','motion','warp','decoder','stitching')) and
                                    all(value['executed_nodes'].get('CUDAExecutionProvider',0)>0
                                        for value in evidence['models'].values()))
        (self.profile_dir/'portrait-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
        for hook in self.hooks:
            hook.remove()
        if not evidence['cuda_verified']:
            raise RuntimeError('Portrait GPU execution evidence was incomplete.')
