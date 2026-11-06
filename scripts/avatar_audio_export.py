"""Mux source audio onto a duration-matched offline avatar replay.

Keeps the source media timeline. Does not infer phonemes, synthesize a voice or
claim lip synchronization. Live audio transport is deliberately separate.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess


def probe(path):
    result=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format',
                           '-of','json',str(path)],check=True,capture_output=True,text=True,timeout=30)
    return json.loads(result.stdout)


def number(value):
    value=float(value)
    if not math.isfinite(value):raise ValueError('Invalid media timestamp')
    return value


def export_audio(rendered, source, output, audio_offset_ms=0.):
    rendered,source=Path(rendered).resolve(strict=True),Path(source).resolve(strict=True)
    output=Path(output).resolve()
    if output.exists() or output in (rendered,source):
        raise ValueError('Choose a new output; inputs remain intact')
    if not math.isfinite(audio_offset_ms) or abs(audio_offset_ms)>2000:
        raise ValueError('Audio adjustment must be within two seconds')
    rp,sp=probe(rendered),probe(source)
    rv=[s for s in rp['streams'] if s['codec_type']=='video']
    sv=[s for s in sp['streams'] if s['codec_type']=='video']
    sa=[s for s in sp['streams'] if s['codec_type']=='audio']
    if len(rv)!=1 or len(sv)!=1 or not sa:
        raise ValueError('Need one rendered/source video stream and source audio')
    duration=number(rv[0].get('duration',rp['format']['duration']))
    source_duration=number(sv[0].get('duration',sp['format']['duration']))
    if not 0<duration<=300 or abs(duration-source_duration)>.100:
        raise ValueError('Replay must preserve source duration within 100 ms; no speed correction is applied')
    if abs(number(rv[0].get('start_time',0)))>.001:
        raise ValueError('Rendered replay must begin at zero')
    delta=number(sa[0].get('start_time',0))-number(sv[0].get('start_time',0))+audio_offset_ms/1000
    trim=max(-delta,0.)
    delay=max(delta,0.)
    samples=round(delay*int(sa[0]['sample_rate']))
    audio=(f'[1:a:0]asetpts=PTS-STARTPTS,atrim=start={trim:.9f},asetpts=PTS-STARTPTS,'
           f'adelay=delays={samples}S:all=1,apad=whole_dur={duration:.9f}[aligned]')
    output.parent.mkdir(parents=True,exist_ok=True)
    command=['ffmpeg','-nostdin','-v','error','-n','-i',str(rendered),'-i',str(source),
             '-filter_complex',audio,'-map','0:v:0','-map','[aligned]',
             '-c:v','copy','-c:a','aac','-t',f'{duration:.9f}','-movflags','+faststart',str(output)]
    subprocess.run(command,check=True,capture_output=True,text=True,timeout=180)
    result=probe(output)
    if not any(s['codec_type']=='audio' for s in result['streams']):
        raise RuntimeError('Export produced no audio stream')
    return dict(duration_seconds=duration,source_audio_relative_start_seconds=delta,
                trimmed_audio_seconds=trim,delayed_audio_samples=samples,
                audio_offset_ms=audio_offset_ms,lip_sync_verified=False,live_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('rendered','source','output'):p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--audio-offset-ms',type=float,default=0.)
    a=p.parse_args()
    print(json.dumps(export_audio(a.rendered,a.source,a.output,a.audio_offset_ms),indent=2))
