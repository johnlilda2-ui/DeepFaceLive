# Private SplattingAvatar browser preview

This optional backend uses the original camera in the left panel and composites the authors’ PeopleSnapshot sample person into the room in the right panel. It uses the existing authenticated, single-session WebRTC connection. Start with the camera fixed and your whole body visible. The preview tracks body pose and position. Mouth motion, individual fingers, moving cameras, and other people in view are outside this first test.

## Prepare the sample

1. Start a Linux T4 environment with CUDA-enabled PyTorch, a matching CUDA toolkit, and a working `pip`. Run `bash scripts/prepare_splatting_avatar_preview.sh /tmp/dfl-splat-preview` from the repository root. This builds the GPU rasterizer and downloads the public sample.
2. Obtain the **female SMPL Python model** from the [official SMPL download site](https://smpl.is.tue.mpg.de/). The model is licensed separately. Keep it in the private runtime and use only a trusted copy. For an older pickle with Chumpy objects, prepare a Chumpy-free copy using the [upstream conversion instructions](https://github.com/vchoutas/smplx/tree/main/tools). The PeopleSnapshot sample uses SMPL topology.
3. Provide the local MediaPipe Pose Landmarker Lite `.task` model and an empty-room image taken with the same camera, position, orientation, and framing. The server center-crops both sources to the same portrait aspect ratio. Keep the plate and model files outside Git.

Use a private JSON configuration, with absolute paths:

```json
{
  "backend": "splattingavatar",
  "runtime_root": "/tmp/dfl-splat-preview",
  "checkpoint": "/tmp/dfl-splat-preview/sample",
  "sample_poses": "/tmp/dfl-splat-preview/sample/poses.npz",
  "smpl_model": "/private/SMPL_FEMALE.pkl",
  "pose_model": "/private/pose_landmarker_lite.task",
  "room_plate": "/private/empty-room.jpg",
  "initial_pose": {
    "root_pose": [3.14159, 0, 0],
    "body_pose": [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                  [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                  [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    "trans": [0, 0, 3],
    "focal": [500, 500],
    "princpt": [160, 288],
    "img_size_wh": [320, 576]
  }
}
```

Install `scripts/requirements_avatar_stream_cpu.txt` in the existing CPU signaling environment. Run:

```bash
python scripts/headless_webrtc_server.py --mode avatar \
  --avatar-config /private/splat-preview.json \
  --gpu-python /tmp/dfl-splat-preview/env/bin/python \
  --evidence-dir /private/splat-preview-logs
```

For a remote GPU, use the existing [private browser connection procedure](windows-cloud-connection.md). The server defaults to loopback. Enter the connection key, then select **Start camera**. The page reports returned frames, dropped frames, median server processing time, and network RTT when available. Server processing time and RTT are separate measures. To measure the visible camera-to-preview delay, record both panels together while making a sharp movement and count the time between that motion appearing in each panel.

The sample GPU benchmark used synthetic poses and one RGB render per frame. The preview adds body tracking, SMPL skinning, a second render to recover opacity, compositing, and video transport. Measure the complete preview before treating the benchmark frame rate as live performance.
