"""Actual GPU lip controls and two authenticated loopback WebRTC sessions.

Uses a prerecorded consented clip, not a laptop camera or a public TURN path.
"""
import argparse
import asyncio
import copy
from fractions import Fraction
import json
from pathlib import Path
import secrets
import time

import numpy as np


async def run(args):
    import cv2
    from PIL import Image,ImageDraw
    from aiortc import RTCConfiguration,RTCPeerConnection,RTCSessionDescription,MediaStreamTrack
    from av import VideoFrame
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer
    from avatar_lip_expression import LipExpressions
    from avatar_pipe_client import AvatarPipeClient
    from avatar_stream_tracker import MediaPipeAvatarTracker
    from headless_lhmpp_stream_worker import HeadlessLHMStreamWorker
    from headless_webrtc_server import create_app
    from avatar_hand_guard import HandObservationGuard

    config=json.loads(args.config.read_text())
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    renderer=worker=tracker=None
    report=dict(scope='actual_t4_lip_and_loopback_avatar_transport',passed=False,
                public_network_verified=False,windows_obs_verified=False,
                lip_sync_verified=False,hand_reconstruction_quality_verified=False)
    server=None
    try:
        renderer=AvatarPipeClient(args.gpu_python,args.config,output/'gpu.log')
        # Only facial expressions change in these controlled diagnostic stills.
        frames=[];trials=[('Neutral',{}),('Fitted pucker',{'mouthPucker':.8}),
                          ('Fitted funnel',{'mouthFunnel':.8}),
                          ('Fitted press',{'mouthPressLeft':.8,'mouthPressRight':.8})]
        for index,(label,scores) in enumerate(trials):
            packet=copy.deepcopy(config['initial_pose'])
            lips=LipExpressions.load(config['lip_map'])
            lips.apply(packet,scores,0.)
            packet=lips.apply(packet,scores,1000.)
            rgb=renderer.render(index,index*1000.,packet)
            frames.append(rgb)
        h,w=frames[0].shape[:2]
        board=Image.new('RGB',(w*len(frames),h+32),'white');draw=ImageDraw.Draw(board)
        for i,((label,_),frame) in enumerate(zip(trials,frames)):
            board.paste(Image.fromarray(frame),(i*w,32));draw.text((i*w+8,8),label,fill='black')
        board.save(output/'lip-controls.png')
        report['lip_changes']=[dict(label=label,changed_pixels=int((np.abs(frame.astype(float)-frames[0]).mean(2)>1).sum()))
                               for (label,_),frame in zip(trials[1:],frames[1:])]
        report['cuda_compute_events']=renderer.cuda_compute_events
        if not renderer.cuda_compute_events:raise RuntimeError('No actual CUDA compute evidence')

        capture=cv2.VideoCapture(str(args.driver_video));input_frames=[]
        try:
            for index in (0,24,48,90):
                capture.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok,image=capture.read()
                if not ok:raise RuntimeError('Cannot decode recorded driving fixture')
                input_frames.append(image)
        finally:capture.release()

        class RecordedSource(MediaStreamTrack):
            kind='video'
            def __init__(self):super().__init__();self.index=0;self.started=time.monotonic()
            async def recv(self):
                await asyncio.sleep(max(0,self.started+self.index/12-time.monotonic()))
                frame=VideoFrame.from_ndarray(input_frames[self.index%len(input_frames)],format='bgr24')
                frame.pts=self.index*7500;frame.time_base=Fraction(1,90000);self.index+=1
                return frame

        tracker=MediaPipeAvatarTracker(config)
        worker=HeadlessLHMStreamWorker(tracker,renderer)
        token=secrets.token_urlsafe(32)
        server=TestServer(create_app(worker,token,session_seconds=120,idle_seconds=40))
        await server.start_server()
        report['sessions']=[]
        async with ClientSession() as http:
            for attempt in range(2):
                pc=RTCPeerConnection(RTCConfiguration(iceServers=[]));source=RecordedSource()
                received=asyncio.get_running_loop().create_future()
                consumer=None
                @pc.on('track')
                def on_track(track):
                    nonlocal consumer
                    async def consume():
                        try:
                            decoded=[await track.recv() for _ in range(3)]
                            if not received.done():received.set_result(decoded)
                        except BaseException as error:
                            if not received.done():received.set_exception(error)
                    consumer=asyncio.create_task(consume())
                sid=secrets.token_urlsafe(18)
                headers={'Authorization':'Bearer '+token}
                try:
                    pc.addTrack(source)
                    await pc.setLocalDescription(await pc.createOffer())
                    response=await http.post(server.make_url('/offer'),headers=headers,
                        json=dict(type='offer',id=sid,sdp=pc.localDescription.sdp))
                    if response.status!=200:raise RuntimeError('Authenticated avatar offer failed: '+str(response.status))
                    answer=await response.json()
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'],type=answer['type']))
                    decoded=await asyncio.wait_for(received,40)
                    pts=[float(f.pts*f.time_base) for f in decoded]
                    if not all(b>a for a,b in zip(pts,pts[1:])):raise RuntimeError('Nonmonotonic returned video')
                    shapes=[list(f.to_ndarray(format='bgr24').shape) for f in decoded]
                    if any(s!=list(renderer.shape) for s in shapes):raise RuntimeError('Returned avatar dimensions changed')
                    status=await (await http.get(server.make_url('/session/'+sid),headers=headers)).json()
                    if not status['avatarFrames']:raise RuntimeError('No animated avatar frames in return')
                    report['sessions'].append(dict(attempt=attempt+1,received_frames=len(decoded),
                        return_timestamps_seconds=pts,shapes=shapes,server_status=status,
                        queue_bound=1,gpu_next_frame_id=worker.frame_id))
                finally:
                    await http.delete(server.make_url('/session/'+sid),headers=headers)
                    await pc.close();source.stop()
                    if consumer is not None:
                        consumer.cancel();await asyncio.gather(consumer,return_exceptions=True)
        report['worker_stats']=copy.deepcopy(worker.stats)
        if args.hand_diagnostics:
            guards={side:HandObservationGuard() for side in ('left','right')}
            counters={side:dict(accepted=0,rejected=0,missing=0) for side in guards}
            details=json.loads(args.hand_diagnostics.read_text())
            for item in details:
                for side,guard in guards.items():
                    hand=item['hands'][side]
                    if 'landmarks_world' not in hand:
                        counters[side]['missing']+=1;continue
                    points=np.asarray(hand['landmarks_world'])*[1,-1,-1]
                    try:accepted,_=guard.check(points,float(item['frame_id']*1000/12))
                    except ValueError:accepted=False
                    counters[side]['accepted' if accepted else 'rejected']+=1
            report['saved_clip_hand_guard']=counters
        report['passed']=True
    finally:
        if server is not None:await server.close()
        elif worker is not None:worker.close()
        else:
            if tracker is not None:tracker.close()
            if renderer is not None:renderer.close()
        if renderer is not None:report['gpu_child_exit_code']=renderer.process.returncode
        (output/'pipeline-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PIPELINE_STAGE_REPORT',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','driver-video','output-dir','hand-diagnostics'):
        p.add_argument('--'+name,required=name!='hand-diagnostics',type=Path)
    p.add_argument('--gpu-python',required=True)
    asyncio.run(run(p.parse_args()))
