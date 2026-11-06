import subprocess
import tempfile
from pathlib import Path
import unittest
import numpy as np
from avatar_audio_export import export_audio,probe


class AudioTimelineTests(unittest.TestCase):
    def test_real_ffmpeg_delayed_audio_retains_source_event_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.mp4';render=root/'render.mp4';out=root/'out.mp4'
            # Tone begins 250ms after video. A separate flash starts at 500ms.
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                "color=black:s=64x64:r=20:d=2,drawbox=x=0:y=0:w=64:h=64:color=white:t=fill:enable='between(t,0.5,0.6)'",
                '-itsoffset','0.25','-f','lavfi','-i','sine=frequency=1000:sample_rate=48000:duration=1.5',
                '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(source)],check=True,capture_output=True)
            subprocess.run(['ffmpeg','-v','error','-i',str(source),'-an','-c:v','copy',str(render)],check=True,capture_output=True)
            report=export_audio(render,source,out)
            raw=subprocess.run(['ffmpeg','-v','error','-i',str(out),'-vn','-ar','48000',
                                '-ac','1','-f','f32le','-'],check=True,capture_output=True).stdout
            samples=np.frombuffer(raw,dtype='<f4')
            # AAC priming is recorded in the source stream start, not guessed.
            first=np.flatnonzero(np.abs(samples)>.03)[0]/48000
            self.assertAlmostEqual(first,.25,delta=.03)
            video=subprocess.run(['ffmpeg','-v','error','-i',str(out),'-an','-pix_fmt','gray',
                                  '-f','rawvideo','-'],check=True,capture_output=True).stdout
            brightness=np.frombuffer(video,dtype=np.uint8).reshape(-1,64,64).mean((1,2))
            flash=np.flatnonzero(brightness>200)[0]/20
            self.assertAlmostEqual(flash,.5,delta=.001)
            self.assertAlmostEqual(flash-first,.25,delta=.03)
            self.assertFalse(report['lip_sync_verified'])
            self.assertAlmostEqual(float(probe(out)['format']['duration']),2.,delta=.05)
            with self.assertRaises(ValueError):export_audio(render,source,out)


if __name__=='__main__':unittest.main()
