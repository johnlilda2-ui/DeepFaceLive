# Windows-to-cloud connection stage

Active branch: `experiment/cloud-headless-webrtc`.

## What is ready

The browser page is served by `scripts/headless_webrtc_server.py`. It provides:

- Start camera and Test demo, with local and processed previews.
- Private bearer-token signaling; tokens never go in URLs or browser storage.
- One GPU session at a time; malformed offers, audio/data tracks, unauthorized
  origins and oversized request bodies are rejected.
- Stop, reconnect, idle cleanup, and a ten-minute maximum session.
  A stopped connection ID cannot be resurrected by a delayed offer.
- Incoming frame counts, cloud processing counts, stale-frame drops, and selected
  ICE candidate types/round-trip time.
- Optional STUN/TURN configuration and browser relay-only mode.

The server binds only to `127.0.0.1` by default. No public tunnel or paid service
is started by this code or the verification notebook.

## Verified evidence

GitHub Actions run `42141597002` passed all 24 unit/regression tests, both CPU
WebRTC integration tests, and Chromium browser integration at commit
`f932443703a392fa27d7f5e72c732901eca2f1a9`.

The browser test used a demo video and a **synthetic camera**, each returning
13 decoded 640×480 frames. It captured zero audio tracks. Stop ended all camera
tracks and released the server slot, and reconnect succeeded. The 390-pixel
mobile layout had no horizontal overflow. CPU browser tests intentionally invert
colors to distinguish processed return video from an unchanged echo; they do
not claim GPU or real-webcam verification.

Screenshots and the browser report:
https://github.com/johnlilda2-ui/DeepFaceLive/actions/runs/42141597002/artifacts/13466858663

Kaggle T4 verification used real HTTP offers/answers and GPU face swapping:

| Check | Observed result |
| --- | --- |
| Returned full frames | 5 at 574×574 |
| Incoming video packets | 14 |
| Session/ICE state | connected/completed |
| Processed/stale-dropped frames | 6/14 at status read |
| Processing error | false |
| Stop released server slot | true |
| CUDA detector/landmarks/swap/identity compute events | 1368/549/1808/130 |
| Returned vs original pixel MAE | 11.46 |
| Returned vs processed-reference pixel MAE | 5.18 |

Reports, the returned frame, and raw CUDA profiles are under
`/kaggle/working/dfl-http-signaling-results/run-f98__orb/`, with the log at
`/kaggle/working/dfl-http-signaling-test.log`.

This test was loopback inside Kaggle. An external Windows connection, physical
webcam, TURN relay path, and OBS Virtual Camera are not yet verified.

## Prepare the private GPU server

Use the existing Kaggle GPU environment and model files. Notebook Step 7 tests
HTTP signaling and closes its server automatically. To run an interactive server
later, execute from `/kaggle/working/DeepFaceLive`:

```bash
/kaggle/working/dfl-cuda12-env/bin/python scripts/headless_webrtc_server.py \
  --source-image doc/celebs/Rob_Doe/Rob_Doe.png \
  --bind 127.0.0.1 --port 8080 \
  --token-file /tmp/dfl-signaling-token
```

The default token file is created with mode 0600, outside saved notebook outputs.
Alternatively, set `DFL_SIGNALING_TOKEN` privately to your own random token of at
least 32 characters. In Kaggle this can be a user-created notebook Secret; do not
print it in saved cells, commit it to GitHub, or send it in chat. Enter the same
token into the connection field on the Windows browser page.

For a CPU-only UI check, add `--diagnostic-cpu` and omit `--source-image`. That
mode inverts colors; it does not perform face swapping.

## External laptop test, after the endpoint is approved

1. Start a temporary HTTPS forwarding route to the loopback server. The user
   approved and tested a temporary route; a stopped preview needs a fresh route.
2. Configure the server with `--allowed-origin https://THE-EXACT-PREVIEW-HOST`.
   Proxy forwarding headers are not trusted to choose allowed origins.
3. Open that HTTPS URL in Chrome or Edge on Windows and enter the private token.
4. Try **Test demo** first, then **Start camera** and grant video permission.
5. Confirm that Cloud return displays video and Received frames increases.
   Stop should release both camera and server session.

An HTTPS forwarding route carries HTTP signaling, not WebRTC media. If media
cannot find a path through the networks, configure a TURN relay. Keep credentials
in a private JSON file passed with `--ice-config /tmp/dfl-ice.json`:

```json
[
  {"urls": "stun:YOUR-STUN-SERVER:3478"},
  {"urls": "turns:YOUR-TURN-SERVER:443?transport=tcp",
   "username": "YOUR-TEMPORARY-USERNAME", "credential": "YOUR-TEMPORARY-CREDENTIAL"}
]
```

`--relay-only` requires TURN and forces the browser to use a relay, useful for a
specific relay-path test. The token-protected `/config` endpoint supplies the ICE
configuration to the browser; no TURN credentials are embedded in the HTML or JS.

Do not assume remote connectivity from local test results. Do not rent RunPod or
another paid GPU without user approval. Stop the Kaggle session after testing to
release free GPU quota. OBS integration follows a successful physical laptop test.


## Final lifecycle audit

An additional regression covers Stop arriving before a delayed offer. The server
retains a bounded, expiring set of stopped request IDs and rejects those late
offers instead of creating orphan sessions. Reconnect uses a fresh request ID.
This adds an eighth signaling regression to the original 24-test suite.

## Preview duration and interruptions

The physical laptop demo returned visibly swapped frames over a direct srflx
media path. The user also reported successful live webcam face swapping.
OBS output and a TURN relay path are still unverified.

The temporary notebook preview deliberately shuts down after 20 minutes. This
can produce HTTP 530 once Cloudflare no longer has its tunnel connector. The
server separately ends each video session after 10 minutes; the client displays
that session countdown. Neither limit is a GPU performance failure.

Status requests have a six-second timeout. Transient network/HTTP server errors
are retried for up to 15 seconds while existing video remains open. Recovery
keeps the same session; a persistent outage releases media with an expiry/link
message. A fresh preview is necessary after its deliberate shutdown; client
retries do not restart cloud processes.

## Choosing the replacement face and preserving lip movement

The browser accepts a JPEG or PNG identity photo (up to 4 MB, one visible face).
Select the photo, confirm permission, and click **Use this face photo** while
video is stopped. The authenticated server checks dimensions before decoding,
limits images to 8 megapixels and 4096 pixels per side, normalizes EXIF rotation,
and reduces the decoded photo to 1024 pixels per side. Invalid or multi-face
photos leave the previous identity unchanged. Uploads are not written to disk;
the worker keeps the derived identity vector until replaced or the server stops.
The chosen photo preview stays in the local browser. Reloading the page does
not reset the cloud identity; its revision is reported by `/config`.

**Swap face** can be disabled before starting for an original-video comparison
through the same cloud transport. **Face result** distinguishes swapped, no-face
passthrough and bypass frames. The swap changes the face region; hair, body and
background remain from the webcam.

**Preserve my real mouth and lip movements** uses the existing 106 landmarks,
converted through the repository's 68-point mapping, to exclude a feathered mouth
region from the swap. This preserves the real lips and teeth, and may retain some
of the user's appearance. It is an optional mouth composite, not audio-driven
lip synthesis or measured audio/video synchronization. Audio capture remains off.


## Portrait avatar preview

Use the portrait-mode preview to retain the reference photo's hair and visible upper-body appearance. Enter the existing connection key, choose a clear front-facing photo of yourself or a consenting person, confirm permission, and upload it before starting the camera. Face forward with a neutral expression at Start; then speak or move your head gently. The model follows webcam motion, not microphone audio. Native LIA output is 256×256 and the shoulders mostly stay still; full-body replacement is not implemented. Uploaded portrait pixels stay in server memory until replaced or the server stops. Use Stop before replacing the portrait or recalibrating. Temporary Kaggle previews expire; the interface shows the session countdown.
