"""Pinned LHM++ backend for one dedicated GPU worker process.

Use fictional or consented references. This is an in-process renderer, not a
public service. Prepare assets first; this module never downloads them.
"""
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import weakref

UPSTREAM_COMMIT = '906b5d9fb967ab42efb92f6fa55bf22cac86b653'


class LHMAvatarBackend:
    def __init__(self, runtime_root, model_dir, reference, *, render_height=420,
                 framing_box=None, optimized_decoder=False):
        self.model = self.cache = self.reference_tensor = self.animation_betas = None
        self.output_rgb = self.mask = self.helper = None
        self.torch = None
        self.stats = {'model_loads': 0, 'avatar_reconstructions': 0, 'render_calls': 0,
                      'cache_present': False, 'closed': False,
                      'live_verified': False, 'optimized_decoder': optimized_decoder}
        self.height = render_height
        self.framing_box = tuple(framing_box) if framing_box is not None else None
        if type(render_height) is not int or render_height < 14 or render_height > 840 or render_height % 14:
            raise ValueError('Render height must be a multiple of 14 in 14..840')
        root, model_dir, reference = [Path(p).resolve(strict=True)
                                     for p in (runtime_root, model_dir, reference)]
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                          text=True, timeout=10).strip()
        if revision != UPSTREAM_COMMIT:
            raise RuntimeError('Unexpected upstream runtime revision')
        self.stats['upstream_commit'] = revision
        # Upstream assets use relative paths. Run this backend in its own process.
        os.chdir(root)
        sys.path.insert(0, str(root))
        try:
            import numpy as np
            import torch
            from accelerate import Accelerator
            from PIL import Image, ImageOps
            from safetensors.torch import load_file
            self.torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required; CPU fallback is forbidden')
            torch._dynamo.config.disable = True
            Accelerator()
            random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            self.stats.update(gpu=torch.cuda.get_device_name(0),
                              allocated_before_load=torch.cuda.memory_allocated())
            self.source = ImageOps.pad(Image.open(reference).convert('RGB'), (504, 840),
                                      color='white', method=Image.Resampling.LANCZOS)
            from engine.pose_estimation.pose_estimator import PoseEstimator
            estimator = PoseEstimator('./pretrained_models/human_model_files', device='cuda')
            with torch.inference_mode():
                shape = estimator(np.asarray(self.source))
            if not shape.is_full_body:
                raise RuntimeError('Reference rejected by full-body shape estimator')
            self.betas = np.asarray(shape.beta).copy()
            del estimator, shape
            gc.collect(); torch.cuda.empty_cache()
            from core.models.modeling_humana4o_lrm import ModelHumanA4OLRM
            from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
            config = json.loads((model_dir / 'config.json').read_text())
            # Published inference weights omit the training-only identity loss net.
            config['use_face_id'] = False
            # Constructor-frame retention was observed in GPU lifecycle tests.
            # Keep the only model reference on self so close() can clear it.
            self.model = ModelHumanA4OLRM(**config)
            weights = load_file(str(model_dir / 'model.safetensors'), device='cpu')
            prefix = 'neural_renderer.transformer_block.'
            if type(self.model.neural_renderer) is not PatchDPT4DecoderOnly or any(
                    k.startswith(prefix) for k in weights):
                raise RuntimeError('Unexpected inference checkpoint/decoder layout')
            # This exact decoder uses patchify/chunk/dense_predictor; the
            # inherited transformer branch is unused and absent from the weights.
            self.model.neural_renderer.transformer_block = torch.nn.ModuleList()
            self.model.load_state_dict(weights, strict=True)
            del weights
            gc.collect()
            self.model.eval()
            self.model.to('cuda')
            if not all(p.is_cuda for p in self.model.parameters()):
                raise RuntimeError('All model parameters must be on CUDA')
            from lhmpp_t4_attention import enable_t4_attention
            self.stats['attention_adapter'] = enable_t4_attention(self.model)
            if optimized_decoder:
                from lhmpp_decoder_precision import enable_decoder_fp16
                self.stats['decoder_precision'] = enable_decoder_fp16(self.model, torch, True)
            else:
                def check_decoder_output(_module, _inputs, values):
                    for value in values:
                        if value is not None and not torch.isfinite(value).all().item():
                            raise RuntimeError('Nonfinite decoder output before encoding')
                self.model.neural_renderer.register_forward_hook(check_decoder_output)
            self.stats.update(model_loads=1, strict_weights_loaded=True,
                              finite_decoder_outputs_checked=True)
            spec = importlib.util.spec_from_file_location(
                'avatar_worker_pose_helpers', root / 'scripts/inference/to_gs_ply.py')
            self.helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.helper)
            self.reference_tensor = torch.from_numpy(np.asarray(self.source).copy()).permute(
                2, 0, 1).float()[None].to('cuda') / 255
        except BaseException:
            self.close()
            raise

    def _motion(self, packet):
        import numpy as np
        from lhmpp_camera_framing import framing_transform, transform_intrinsics
        torch, helper = self.torch, self.helper
        raw = {k: torch.tensor(v, dtype=torch.float32) for k, v in packet.items()}
        raw['betas'] = torch.from_numpy(self.betas).float()
        c2w, intr = helper._load_pose_minimal(raw)
        params = helper._stack_smplx_params_list_minimal([raw], raw['betas'])
        motion = helper._to_motion_batch_dict_minimal(
            c2w[None], intr[None], torch.ones((1, 3)), params, [], False)
        w, h = packet['img_size_wh']
        framing = framing_transform(w, h, self.height, self.framing_box)
        width, height = framing['width'], framing['height']
        if width > 840:
            raise ValueError('Render canvas too wide; choose a bounded camera crop')
        if hasattr(self, 'output_shape') and self.output_shape != (height, width, 3):
            raise ValueError('Render canvas changed during session')
        self.output_shape = (height, width, 3)
        sp = motion['smplx_params']
        focal, principal = transform_intrinsics(
            [sp['focal'][..., 0], sp['focal'][..., 1]],
            [sp['princpt'][..., 0], sp['princpt'][..., 1]], framing)
        sp['focal'] = torch.stack(focal, -1)
        sp['princpt'] = torch.stack(principal, -1)
        motion['render_intrs'][..., 0, 0], motion['render_intrs'][..., 1, 1] = focal
        motion['render_intrs'][..., 0, 2], motion['render_intrs'][..., 1, 2] = principal
        sp['img_size_wh'][...] = torch.tensor([width, height])
        if self.mask is None:
            self.mask = np.full((height, width), 255, dtype=np.uint8)
            self.output_rgb = torch.ones((height, width, 3))
        self.stats['camera_framing'] = framing
        return motion

    def prepare(self, packet):
        if self.stats['closed'] or self.cache is not None:
            raise RuntimeError('Backend is closed or already has an avatar')
        torch = self.torch
        motion = self._motion(packet)
        params = {k: v.to('cuda') for k, v in motion['smplx_params'].items()}
        with torch.inference_mode():
            outputs = self.model.infer_single_view(
                self.reference_tensor[None], None, None,
                render_c2ws=motion['render_c2ws'].to('cuda'),
                render_intrs=motion['render_intrs'].to('cuda'),
                render_bg_colors=motion['render_bg_colors'].to('cuda'),
                smplx_params=params, ref_imgs_bool=torch.ones((1, 1), dtype=torch.bool, device='cuda'),
                return_pred_shape=getattr(self.model, 'use_pred_shape_for_render', False))
        if len(outputs) not in (7, 8):
            raise RuntimeError('Unexpected reconstruction output layout')
        self.cache = dict(zip(('gs_model_list', 'query_points', 'transform_mat_neutral_pose',
                               'gs_hidden_features', 'image_latents', 'motion_emb', 'pos_emb'), outputs[:7]))
        predicted_shape = outputs[7] if len(outputs) == 8 else None
        if predicted_shape is not None and getattr(self.model, 'use_pred_shape_for_render', False):
            params = type(self.model).smplx_params_with_pred_shape_betas(params, predicted_shape)
        self.animation_betas = params['betas']
        self.stats.update(avatar_reconstructions=1, cache_present=True)

    def render(self, packet, *, output_renderer='neural'):
        if output_renderer not in ('neural', 'gs'):
            raise ValueError('Output renderer must be neural or gs')
        if self.stats['closed'] or self.cache is None:
            raise RuntimeError('Avatar cache is unavailable')
        torch = self.torch
        motion = self._motion(packet)
        params = {k: v.to('cuda') for k, v in motion['smplx_params'].items()}
        params.update(betas=self.animation_betas,
                      transform_mat_neutral_pose=self.cache['transform_mat_neutral_pose'])
        kwargs = {k: v for k, v in self.cache.items() if k != 'transform_mat_neutral_pose'}
        if kwargs.get('pos_emb') is None:
            kwargs.pop('pos_emb')
        with torch.inference_mode():
            rgb, _ = self.model.animation_infer(
                **kwargs, smplx_params=params,
                render_c2ws=motion['render_c2ws'].to('cuda'),
                render_intrs=motion['render_intrs'].to('cuda'),
                render_bg_colors=motion['render_bg_colors'].to('cuda'),
                mask_seqs=[self.mask], offset_list=[[1.0, 1.0, 0, 0]],
                output_rgb=self.output_rgb, infer_output_renderer=output_renderer)
            if rgb.shape != (1, *self.output_shape) or not torch.isfinite(rgb).all().item():
                raise RuntimeError('Invalid floating-point rendered frame')
            frame = (rgb[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy().copy()
        self.stats['render_calls'] += 1
        return frame

    def close(self):
        if self.stats['closed']:
            return
        self.stats.update(closed=True, cache_present=False)
        model_ref = weakref.ref(self.model) if self.model is not None else None
        for name in ('model', 'cache', 'reference_tensor', 'animation_betas', 'mask', 'output_rgb', 'helper'):
            setattr(self, name, None)
        self.betas = self.source = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_initialized():
            self.stats['allocated_after_gc'] = self.torch.cuda.memory_allocated()
            self.stats['model_python_released'] = model_ref is None or model_ref() is None
            self.torch.cuda.synchronize()
            # Pinned torch 2.3/cu121 retains a per-handle cuBLAS workspace
            # independently of model tensors. This process owns that workspace.
            clear_workspace = getattr(self.torch._C, '_cuda_clearCublasWorkspaces', None)
            if clear_workspace is not None:
                clear_workspace()
                self.stats['cublas_workspace_cleared'] = True
            self.torch.cuda.empty_cache()
            self.stats.update(allocated_after_close=self.torch.cuda.memory_allocated(),
                              reserved_after_close=self.torch.cuda.memory_reserved())
