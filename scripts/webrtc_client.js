'use strict';
const $ = id => document.getElementById(id);
let peer, stream, controller, sessionId, connectionToken = '', epoch = 0;
let polling, rendering, connectionDeadline, pollingBusy = false;
const message = value => { $('status').textContent = value; };
function buttons(running) {
  $('camera').disabled = running; $('demo').disabled = running;
  $('stop').disabled = !running; $('token').disabled = running;
}
async function api(path, options = {}) {
  const response = await fetch(path, {cache:'no-store', ...options,
    headers:{Authorization:`Bearer ${connectionToken}`, 'Content-Type':'application/json', ...options.headers}});
  if (!response.ok) {
    const labels = {401:'Connection token rejected.',403:'This page origin is not allowed.',
      409:'The cloud worker is already in use. Stop the other session first.',
      404:'The cloud session ended.',504:'Cloud negotiation timed out. Check STUN/TURN settings.'};
    throw new Error(labels[response.status] || `Cloud request failed (${response.status}).`);
  }
  return response.status === 204 ? null : response.json();
}
async function stop(reason = 'Stopped. Camera and cloud session released.') {
  ++epoch;
  const oldId = sessionId, oldToken = connectionToken;
  sessionId = null; connectionToken = '';
  controller?.abort(); controller = null;
  clearInterval(polling); clearInterval(rendering); clearTimeout(connectionDeadline);
  const oldPeer = peer; peer = null; oldPeer?.close();
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  $('local').srcObject = null; $('output').srcObject = null;
  $('fullscreen').disabled = true;
  $('connection').textContent = 'Not connected';
  if (oldId && oldToken) {
    try {
      await fetch(`/session/${encodeURIComponent(oldId)}`, {method:'DELETE', cache:'no-store',
        headers:{Authorization:`Bearer ${oldToken}`}, signal:AbortSignal.timeout(6000)});
    } catch (_) { /* The server's idle timeout also releases abandoned sessions. */ }
  }
  buttons(false); message(reason);
}
function waitForIce(pc, signal) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve, reject) => {
    const finish = error => {clearTimeout(timer); pc.removeEventListener('icegatheringstatechange', check);
      signal.removeEventListener('abort', abort); error ? reject(error) : resolve();};
    const check = () => {if (pc.iceGatheringState === 'complete') finish();};
    const abort = () => finish(new DOMException('Stopped','AbortError'));
    const timer = setTimeout(() => finish(new Error('Could not gather connection candidates. Check relay settings.')), 18000);
    pc.addEventListener('icegatheringstatechange', check); signal.addEventListener('abort', abort, {once:true});
    if (signal.aborted) abort();
  });
}
async function demoStream(myEpoch) {
  const image = new Image(); image.src = '/demo.png'; await image.decode();
  if (myEpoch !== epoch) throw new DOMException('Stopped','AbortError');
  const canvas = $('demoCanvas'), context = canvas.getContext('2d');
  const draw = () => {context.fillStyle='#080c13'; context.fillRect(0,0,640,480);
    context.drawImage(image,80,0,480,480); context.fillStyle='#81e0c8';
    context.fillRect(0,465,(Date.now()/15)%640,6);};
  draw(); rendering = setInterval(draw, 66);
  return canvas.captureStream(15);
}
async function poll(myEpoch) {
  if (!peer || myEpoch !== epoch || pollingBusy) return;
  pollingBusy = true;
  try {
    const status = await api(`/session/${encodeURIComponent(sessionId)}`, {signal:controller?.signal});
    if (myEpoch !== epoch || !peer) return;
    $('processing').textContent = `${status.processed} frames · ${status.dropped} stale dropped`;
    if (status.error) throw new Error('Cloud video processing failed. Stop and check the cloud log.');
    const stats = await peer.getStats();
    let selected;
    for (const report of stats.values()) {
      if (report.type === 'inbound-rtp' && (report.kind === 'video' || report.mediaType === 'video'))
        $('frames').textContent = String(report.framesDecoded ?? 0);
      if (report.type === 'transport' && report.selectedCandidatePairId) selected = stats.get(report.selectedCandidatePairId);
      if (!selected && report.type === 'candidate-pair' && report.nominated && report.state === 'succeeded') selected = report;
    }
    if (selected) {
      const local = stats.get(selected.localCandidateId), remote = stats.get(selected.remoteCandidateId);
      const latency = selected.currentRoundTripTime == null ? '' : ` · ${Math.round(selected.currentRoundTripTime*1000)} ms RTT`;
      $('network').textContent = `${local?.candidateType || '?'} → ${remote?.candidateType || '?'}${latency}`;
    }
  } catch (error) {if (myEpoch === epoch) await stop(error.message);}
  finally {pollingBusy = false;}
}
async function start(mode) {
  if (peer || stream || sessionId) return;
  connectionToken = $('token').value.trim();
  if (connectionToken.length < 32) {message('Paste the private connection token first.'); return;}
  if (!window.isSecureContext || !window.RTCPeerConnection || !crypto.randomUUID) {
    connectionToken = ''; message('Open this page using HTTPS or localhost to use video.'); return;
  }
  const myEpoch = ++epoch;
  buttons(true); sessionId = crypto.randomUUID(); controller = new AbortController();
  $('frames').textContent = '0'; $('processing').textContent = '0 frames';
  try {
    message('Checking cloud connection…');
    const config = await api('/config', {signal:controller.signal});
    if (myEpoch !== epoch) return;
    if (mode === 'camera' && !navigator.mediaDevices)
      throw new Error('Open this page using HTTPS or localhost to use video.');
    message(mode === 'camera' ? 'Waiting for your camera permission…' : 'Starting the consented demo image…');
    const input = mode === 'camera' ? await navigator.mediaDevices.getUserMedia({audio:false,
      video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:15,max:15}}}) : await demoStream(myEpoch);
    if (myEpoch !== epoch) {input.getTracks().forEach(track=>track.stop()); return;}
    stream = input; $('local').srcObject = stream;
    peer = new RTCPeerConnection({iceServers:config.iceServers,
      iceTransportPolicy:config.relayOnly ? 'relay' : 'all'});
    const currentPeer = peer;
    peer.ontrack = event => {if (myEpoch !== epoch) return;
      $('output').srcObject = new MediaStream([event.track]); $('fullscreen').disabled = false;};
    peer.onconnectionstatechange = () => {
      if (myEpoch !== epoch) return;
      $('connection').textContent = currentPeer.connectionState;
      if (currentPeer.connectionState === 'connected') {clearTimeout(connectionDeadline);
        message('Video connected. The right preview is returning from the cloud.');}
      if (currentPeer.connectionState === 'failed') void stop('Media connection failed. Check STUN/TURN relay settings.');
    };
    stream.getVideoTracks().forEach(track => peer.addTrack(track, stream));
    message('Negotiating the video connection…');
    await peer.setLocalDescription(await peer.createOffer());
    await waitForIce(peer, controller.signal);
    if (myEpoch !== epoch) return;
    const answer = await api('/offer', {method:'POST', signal:controller.signal,
      body:JSON.stringify({type:peer.localDescription.type,sdp:peer.localDescription.sdp,id:sessionId})});
    if (myEpoch !== epoch) return;
    if (answer.id !== sessionId || answer.type !== 'answer') throw new Error('Invalid cloud answer.');
    await peer.setRemoteDescription({type:answer.type,sdp:answer.sdp});
    connectionDeadline = setTimeout(() => {if (myEpoch === epoch && peer?.connectionState !== 'connected')
      void stop('No media path was found. Configure a TURN relay for this network.');}, 25000);
    $('network').textContent = config.hasIceServers ? 'Selecting media path…' : 'No STUN/TURN configured';
    polling = setInterval(() => void poll(myEpoch), 1000);
    void poll(myEpoch);
  } catch (error) {
    if (myEpoch === epoch) await stop(error.name === 'NotAllowedError' ? 'Camera permission was not granted.' :
      error.name === 'NotFoundError' ? 'No camera was found. Try Test demo.' : error.message || 'Connection failed.');
  }
}
$('camera').addEventListener('click',()=>void start('camera'));
$('demo').addEventListener('click',()=>void start('demo'));
$('stop').addEventListener('click',()=>void stop());
$('fullscreen').addEventListener('click',()=>void $('output').requestFullscreen());
window.addEventListener('pagehide', () => {
  if (sessionId && connectionToken) void fetch(`/session/${encodeURIComponent(sessionId)}`,
    {method:'DELETE',headers:{Authorization:`Bearer ${connectionToken}`},keepalive:true}).catch(()=>{});
  controller?.abort(); peer?.close();
  stream?.getTracks().forEach(track => track.stop());});
