"""Headless adapter for DeepFaceLive's existing LIA portrait animator.

Uses the working ONNX CUDA environment. The entire returned canvas comes from
the uploaded portrait, not the driver's hair, clothes, or background.
"""
import json
from pathlib import Path
import tempfile

import headless_inference_smoke_test as smoke
from headless_cloud_worker import validate_frame


class HeadlessLiaPortraitWorker:
    mode = 'portrait'
    engine = 'lia'

    def __init__(self, source_image, evidence_dir, runtime_dir=None):
        import cv2
        self.closed = False
        self.frame_count = 0
        self.threshold = .5
        root = Path(evidence_dir).resolve()
        root.mkdir(parents=True,exist_ok=True)
        self.profile_dir = Path(tempfile.mkdtemp(prefix='lia-portrait-',dir=root))
        model_dir = smoke.REPO_ROOT/'modelhub/onnx/LIA'
        merged = model_dir/'generator.onnx'
        if not merged.is_file():
            for index,size in enumerate([50331648,50331648,50331648,41560533]):
                part = model_dir/f'generator.onnx.part{index}'
                if not part.is_file() or part.stat().st_size != size:
                    raise ValueError('Download the original LIA model parts from this project first.')
        from modelhub.onnx import LIA, YoloV5Face, InsightFace2D106
        device = smoke.choose_device(True)
        with smoke.profiled_sessions(self.profile_dir,True):
            self.detector, self.marker = YoloV5Face(device), InsightFace2D106(device)
            self.animator = LIA(device)
        if merged.stat().st_size != 192555477:
            raise ValueError('LIA model assembly size mismatch.')
        for _,session in self.sessions():
            if 'CUDAExecutionProvider' not in session.get_providers():
                raise RuntimeError('LIA portrait models require CUDA; fallback is disabled.')
        image = cv2.imread(str(source_image))
        if image is None:
            raise ValueError('Cannot decode source portrait.')
        self.set_identity(image)

    def sessions(self):
        return [('detector',self.detector._sess),('landmarks',self.marker._sess),
                ('portrait_generator',self.animator._generator)]

    def set_identity(self, image):
        if self.closed:
            raise RuntimeError('Worker is closed.')
        validate_frame(image)
        if len(self.detector.extract(image,threshold=self.threshold)[0]) != 1:
            raise ValueError('Choose a portrait with exactly one clearly visible face.')
        from xlib.image import ImageProcessor
        processor = ImageProcessor(image)
        processor.fit_in(TW=256,TH=256,pad_to_target=True,allow_upscale=True)
        prepared = processor.get_image('HWC').copy()
        self.source = prepared
        self.reset_motion()

    def reset_motion(self):
        self.driving_initial = None
        self.last_output = self.source.copy()

    def process_frame(self, frame, preserve_mouth=False):
        import numpy as np
        if self.closed:
            raise RuntimeError('Worker is closed.')
        if preserve_mouth:
            raise ValueError('Real-mouth overlay is face-swap only.')
        validate_frame(frame)
        try:
            driver = smoke.detect_and_align_face(frame,'driver',self.detector,self.marker,
                                                2.2,256,self.threshold)
        except smoke.NoFaceDetected:
            return self.last_output.copy(),'avatar_no_face'
        if self.driving_initial is None:
            initial = self.animator.extract_motion(driver)
            if initial.shape != (1,20) or not np.isfinite(initial).all():
                raise RuntimeError('Invalid portrait reference motion.')
            self.driving_initial = initial
        output = self.animator.generate(self.source,driver,self.driving_initial,power=1.0)
        if output.shape != self.source.shape or output.dtype != np.uint8 or not np.isfinite(output).all():
            raise RuntimeError('Invalid portrait generator output.')
        self.last_output = output.copy()
        self.frame_count += 1
        return output,'avatar'

    def close(self):
        if self.closed:
            return
        self.closed = True
        evidence = {'engine':'DeepFaceLive LIA','native_resolution':[256,256],
                    'frames':self.frame_count,'models':{}}
        for label,session in self.sessions():
            profile = session.end_profiling()
            counts = dict(smoke.profile_counts(profile))
            evidence['models'][label] = {'profile':profile,'executed_nodes':counts,
                                         'providers':session.get_providers()}
        evidence['cuda_verified'] = all(x['executed_nodes'].get('CUDAExecutionProvider',0)>0
                                        for x in evidence['models'].values())
        (self.profile_dir/'portrait-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
        if not evidence['cuda_verified']:
            raise RuntimeError('Actual CUDA execution was not proven for every portrait model.')
