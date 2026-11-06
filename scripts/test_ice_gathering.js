'use strict';
// Exercise the actual client's ICE wait without a webcam, network or credentials.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(require('node:path').join(__dirname, 'webrtc_client.js'), 'utf8');
const start = source.indexOf('function waitForIce(');
const end = source.indexOf('async function demoStream', start);
const waitForIce = vm.runInNewContext(source.slice(start, end) + '\nwaitForIce;',
  {setTimeout, clearTimeout, DOMException});
function peer() {
  const pc = new EventTarget();
  pc.iceGatheringState = 'gathering';
  pc.localDescription = {sdp:''};
  return pc;
}
(async () => {
  let pc = peer();
  const waiting = waitForIce(pc, new AbortController().signal, true);
  pc.localDescription.sdp = 'a=candidate:1 1 udp 1 192.0.2.1 1234 typ relay\r\n';
  pc.dispatchEvent(new Event('icecandidate'));
  await waiting;
  assert.equal(pc.iceGatheringState, 'gathering', 'A usable relay must not wait for unrelated route checks');
  pc = peer();
  const failed = waitForIce(pc, new AbortController().signal, true);
  const error = new Event('icecandidateerror'); error.errorCode = 701;
  pc.dispatchEvent(error);
  pc.iceGatheringState = 'complete'; pc.dispatchEvent(new Event('icegatheringstatechange'));
  await assert.rejects(failed, /could not obtain a TURN relay route.*701/);
  pc = peer();
  const controller = new AbortController();
  const stopped = waitForIce(pc, controller.signal, true); controller.abort();
  await assert.rejects(stopped, {name:'AbortError'});
  pc = peer(); pc.iceGatheringState = 'complete';
  await waitForIce(pc, new AbortController().signal, false);
  console.log('ICE gathering tests passed: early relay, explicit allocation failure, Stop, direct completion.');
})().catch(error => {console.error(error); process.exitCode = 1;});
