#!/usr/bin/env python3
"""Create a synchronized, enlarged review video from an existing avatar render.

Presentation only: no new inference, interpolation, pose correction or compositing.
Requires ffmpeg/ffprobe on PATH, NumPy and Pillow.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,duration,nb_frames',
        '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)['streams']
    if len(streams) != 1:
        raise ValueError('Require one video stream')
    return streams[0]


def decode(path, start, frames, fps, width=480):
    # Normalize both timelines to the declared review FPS, retaining frame order.
    resize = [f'scale={width}:-2'] if width else []
    video_filter = ','.join([f'fps={fps}'] + resize)
    command = ['ffmpeg', '-v', 'error', '-ss', str(start), '-i', str(path),
        '-t', str(frames / fps), '-an', '-vf', video_filter,
        '-frames:v', str(frames), '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    data = subprocess.run(command, check=True, capture_output=True).stdout
    # Decode one PNG for its post-rotation dimensions, including portrait inputs.
    png = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(start), '-i', str(path),
        '-frames:v', '1'] + (['-vf', ','.join(resize)] if resize else []) +
        ['-f', 'image2pipe', '-vcodec', 'png', '-'],
        check=True, capture_output=True).stdout
    import io
    width, height = Image.open(io.BytesIO(png)).size
    expected = frames * width * height * 3
    if len(data) != expected:
        raise ValueError(f'Expected {frames} complete frames; decoded {len(data) // (width * height * 3)}')
    return np.frombuffer(data, np.uint8).reshape(frames, height, width, 3)


def union_crop(frames, threshold=230, margin=0.06):
    """One fixed crop for the whole clip; preserve all thresholded foreground."""
    mask = np.any(frames.min(axis=-1) < threshold, axis=0)
    y, x = np.where(mask)
    if not len(x):
        raise ValueError('Avatar video is blank')
    height, width = mask.shape
    pad = max(4, int(np.ceil((y.max() - y.min() + 1) * margin)))
    return (max(0, int(x.min()) - pad), max(0, int(y.min()) - pad),
            min(width, int(x.max()) + 1 + pad), min(height, int(y.max()) + 1 + pad))


def font(size):
    for candidate in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                      '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def paste_contained(canvas, frame, box):
    left, top, right, bottom = box
    tile = ImageOps.contain(frame, (right - left, bottom - top),
                            method=Image.Resampling.LANCZOS)
    x = left + (right - left - tile.width) // 2
    y = top + (bottom - top - tile.height) // 2
    canvas.paste(tile, (x, y))
    return [x, y, tile.width, tile.height]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver-video', type=Path, required=True)
    parser.add_argument('--avatar-video', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--driver-start', type=float, default=0)
    parser.add_argument('--frames', type=int, default=18)
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    if not np.isfinite(args.driver_start) or args.driver_start < 0:
        raise ValueError('Driver start must be finite and nonnegative')
    if not 2 <= args.frames <= 180 or not 1 <= args.fps <= 30 or not 1 <= args.repeat <= 5:
        raise ValueError('Review must be bounded to 2..180 frames, 1..30 FPS and 1..5 repeats')
    for path in [args.driver_video, args.avatar_video]:
        if not path.is_file():
            raise ValueError('Input video not found')
    output = args.output_dir.resolve()
    # Never overwrite existing input/output or evidence directories.
    output.mkdir(parents=True, exist_ok=False)
    source = decode(args.driver_video, args.driver_start, args.frames, args.fps)
    avatar = decode(args.avatar_video, 0, args.frames, args.fps, width=None)
    crop = union_crop(avatar)
    x0, y0, x1, y1 = crop
    preview_frames = []
    placement = None
    for index, (left, right) in enumerate(zip(source, avatar)):
        board = Image.new('RGB', (1080, 800), '#101a2b')
        draw = ImageDraw.Draw(board)
        draw.text((28, 18), 'BODY MOTION COMPARISON', font=font(26), fill='white')
        draw.text((28, 54), f'{args.frames / args.fps:g}-second excerpt | replayed {args.repeat} times | not live', font=font(17), fill='#b4c6df')
        for x, title in [(28, 'Your recorded movement'), (552, 'Test avatar')]:
            draw.rounded_rectangle((x, 94, x + 500, 710), radius=12, fill='#1d2b40')
            draw.text((x + 16, 107), title, font=font(23), fill='white')
        paste_contained(board, Image.fromarray(left), (42, 151, 514, 696))
        # Keep the true white renderer background; no inferred alpha/segmentation.
        placement = paste_contained(board, Image.fromarray(right[y0:y1, x0:x1]),
                                    (566, 151, 1038, 696))
        draw.text((28, 730), 'Same moment in both panels | head-to-feet view', font=font(20), fill='white')
        draw.text((28, 762), 'Head and hand detail unfinished. No finger tracking or lip sync.',
                  font=font(18), fill='#b4c6df')
        preview_frames.append(np.asarray(board))
    frames = np.stack(preview_frames)
    movie = output / 'body-motion-comparison.mp4'
    encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1080x800', '-r', str(args.fps), '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', '-n', str(movie)], stdin=subprocess.PIPE)
    try:
        for _ in range(args.repeat):
            encoder.stdin.write(frames.tobytes())
        encoder.stdin.close()
        if encoder.wait(timeout=60):
            raise RuntimeError('Review video encoding failed')
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    # Different poses, selected from the same review sequence.
    Image.fromarray(frames[args.frames // 2]).save(output / 'comparison-preview.jpg', quality=94)
    info = probe(movie)
    if int(info['nb_frames']) != args.frames * args.repeat or (info['width'], info['height']) != (1080, 800):
        raise RuntimeError('Encoded review geometry/frame count mismatch')
    report = dict(scope='presentation_only_existing_render', new_inference=False,
        pose_changed=False, interpolation=False, driver_start_seconds=args.driver_start,
        unique_frames=args.frames, fps=args.fps, repeat=args.repeat,
        output_frames=args.frames * args.repeat, output_seconds=args.frames * args.repeat / args.fps,
        fixed_avatar_crop=list(crop), decoded_avatar_size=list(avatar.shape[1:3][::-1]),
        avatar_placement=placement, output_size=[1080, 800],
        foreground_detection='nonwhite threshold, not semantic segmentation',
        limitations=['Upscaling does not add source detail', 'White renderer background retained',
                     'Original head/hand imperfections retained', 'Offline test; no lip sync'],
        sha256=hashlib.sha256(movie.read_bytes()).hexdigest(), passed=True)
    (output / 'comparison-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
