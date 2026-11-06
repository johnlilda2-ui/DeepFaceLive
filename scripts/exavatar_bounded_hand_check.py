"""One bounded canonical-hand/camera check of the saved official ExAvatar model.

Run after exavatar_pretrained_probe.py has prepared the pinned runtime and assets.
Does not train or change the LHM++ baseline. Generates three diagnostic stills.
"""
from pathlib import Path
import gc
import json
import math
import sys
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path('/kaggle/working/dfl-exavatar-probe')
sys.path[:0] = [str(ROOT/'inference-overlay'), str(ROOT/'runtime/avatar/main'),
                str(ROOT/'runtime/avatar/common')]
from config import cfg
cfg.human_model_path = '/tmp/dfl-fullbody-assets/prior/human_model_files'
cfg.dataset = 'Custom'
from utils.smpl_x import smpl_x
from nets.module import HumanGaussian, GaussianRenderer


def main():
    assert subprocess.check_output(['git', '-C', str(ROOT/'runtime'), 'rev-parse', 'HEAD'],
                                   text=True).strip() == '518f5647c8fa75cb648315c1677fdc105101fe11'
    smpl_x.set_id_info(*[torch.tensor(json.loads((ROOT/name).read_text()), dtype=torch.float32)
        for name in ['shape_param.json', 'face_offset.json', 'joint_offset.json', 'locator_offset.json']])
    checkpoint = torch.load(ROOT/'snapshot_4.pth', map_location='cpu', mmap=True, weights_only=False)
    state = {k.removeprefix('human_gaussian.'): v for k, v in checkpoint['network'].items()
             if k.startswith('human_gaussian.')}
    assert state
    human = HumanGaussian().cuda()
    with torch.no_grad():
        human.init()
    human.load_state_dict(state, strict=True)
    human.eval()
    renderer = GaussianRenderer().cuda()
    del checkpoint, state
    gc.collect()
    records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    raw = records[24]['pose']
    base = {k: torch.tensor(raw[k], dtype=torch.float32, device='cuda').flatten()
            for k in ['body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose']}
    base.update(root_pose=torch.tensor([math.pi, 0., 0.], device='cuda'),
                trans=torch.tensor([0., 0., 4.], device='cuda'),
                expr=torch.zeros(smpl_x.expr_param_dim, device='cuda'))
    cases = []
    for label in ['Tracked hands', 'Zero hand rotations', 'Canonical body and hands']:
        pose = {k: v.clone() for k, v in base.items()}
        if label != 'Tracked hands':
            pose['lhand_pose'].zero_()
            pose['rhand_pose'].zero_()
        if label == 'Canonical body and hands':
            pose['body_pose'] = smpl_x.neutral_body_pose.flatten().cuda()
        cases.append((label, pose))

    def camera(focal=1., translation=(0., 0., 0.)):
        return dict(R=torch.eye(3, device='cuda'), t=torch.tensor(translation, device='cuda'),
                    focal=torch.full((2,), focal*1024, device='cuda'),
                    princpt=torch.full((2,), 512., device='cuda'))

    with torch.inference_mode():
        # Pose translations are world coordinates here. The upstream default
        # converts camera poses back to world and would cancel view translation.
        means = [human(pose, camera(), is_world_coord=True)[0]['mean_3d'] for _, pose in cases]
        xyz = torch.cat(means)
        center = (xyz.amin(0)+xyz.amax(0))/2
        translation = [-float(center[0]), -float(center[1]), 0.]
        shifted = xyz+torch.tensor(translation, device='cuda')
        assert shifted[:, 2].min() > 0
        uv = shifted[:, :2]/shifted[:, 2:]
        focal = .40/float(uv.abs().max())
        cam = camera(focal, translation)
        evidence = []
        board = Image.new('RGB', (1536, 550), 'white')
        draw = ImageDraw.Draw(board)
        for index, (label, pose) in enumerate(cases):
            assets = human(pose, cam, is_world_coord=True)[0]
            rgb = renderer(assets, (1024, 1024), cam)['img']
            torch.cuda.synchronize()
            assert torch.isfinite(rgb).all()
            pixels = rgb.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            foreground = (pixels.min(2) < 235)
            ys, xs = np.where(foreground)
            assert len(xs) > 1000
            margins = [int(xs.min()), int(ys.min()), int(1023-xs.max()), int(1023-ys.max())]
            Image.fromarray(pixels).save(ROOT/f'bounded-hands-{index}.png')
            board.paste(Image.fromarray(pixels).resize((512, 512)), (index*512, 30))
            draw.text((index*512+10, 8), label, fill='black')
            evidence.append(dict(label=label, foreground_margins_px=margins))
    board.save('/kaggle/working/exavatar-bounded-hands.png')
    report = dict(execution_passed=True, quality_verified=False, live_verified=False,
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  strict_human_weights_loaded=True, frames=len(cases),
                  camera=dict(normalized_focal=focal, translation=translation,
                              correction='World-coordinate poses and centered view translation; upstream projection ignores principal point offsets'),
                  cases=evidence, framing_passed=all(min(e['foreground_margins_px']) >= 40 for e in evidence),
                  limitations=['Official demo identity; not the generated target',
                               'Canonical rotations do not verify tracked finger movement',
                               'Visual quality requires inspection; no live performance claim'])
    (ROOT/'bounded-hands-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('BOUNDED_HAND_REPORT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
