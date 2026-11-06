import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from avatar_pipe_client import AvatarPipeClient


FAKE='''import sys,json
print(json.dumps(dict(type='ready',shape=[14,14,3],gpu='CPU protocol fixture')),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['type']=='close':break
 n=14*14*3
 if r['id']==99:n+=1
 print(json.dumps(dict(type='frame',id=r['id'],bytes=n)),flush=True)
 sys.stdout.buffer.write(bytes([r['id']%256])*n);sys.stdout.buffer.flush()
'''


class PipeTests(unittest.TestCase):
    def test_frame_boundaries_and_malformed_size_close_child(self):
        real_popen=subprocess.Popen
        def fake(*args,**kwargs):return real_popen([sys.executable,'-u','-c',FAKE],**kwargs)
        with tempfile.TemporaryDirectory() as tmp,patch('avatar_pipe_client.subprocess.Popen',side_effect=fake):
            client=AvatarPipeClient(sys.executable,'unused',Path(tmp)/'log')
            try:
                for index in (1,2):
                    frame=client.render(index,float(index),{})
                    self.assertEqual(frame.shape,(14,14,3));self.assertTrue((frame==index).all())
                with self.assertRaises(RuntimeError):client.render(99,99.,{})
                self.assertTrue(client.failed)
            finally:client.close()
            self.assertIsNotNone(client.process.returncode)


if __name__=='__main__':unittest.main()
