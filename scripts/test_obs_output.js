'use strict';
// DOM contract test only. Actual Windows OBS/Virtual Camera remains a user test.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const nodes = new Map();
function node(id) {
  if (!nodes.has(id)) nodes.set(id, {srcObject:null, disabled:false, files:[], value:'',
    handlers:{},addEventListener(type,cb){this.handlers[type]=cb;},
    closest(){return {hidden:false};},removeAttribute(){}});
  return nodes.get(id);
}
const output = {srcObject:null,play:()=>Promise.resolve()};
const notice = {hidden:false};
const child = {closed:false,document:{getElementById:id=>id==='processedVideo'?output:notice}};
const handlers={};
const context=vm.createContext({document:{getElementById:node},
  window:{addEventListener:(type,cb)=>{handlers[type]=cb;},open:()=>child},
  location:{origin:'https://unit.example'},clearTimeout,clearInterval,setTimeout,setInterval,
  AbortSignal,console});
vm.runInContext(fs.readFileSync(__dirname+'/webrtc_client.js','utf8'),context);
(async()=>{
  const processed={id:'cloud-return'},local={id:'private-local-camera'};
  node('output').srcObject=processed;node('local').srcObject=local;
  node('obsOutput').handlers.click();
  assert.equal(output.srcObject,processed);
  assert.notEqual(output.srcObject,local);
  output.srcObject=null;
  handlers.message({origin:'https://untrusted.example',source:child,data:{type:'dfl-output-ready'}});
  assert.equal(output.srcObject,null);
  handlers.message({origin:'https://unit.example',source:child,data:{type:'dfl-output-ready'}});
  assert.equal(output.srcObject,processed);
  await vm.runInContext('stop()',context);
  assert.equal(output.srcObject,null);assert.equal(notice.hidden,false);
  console.log('OBS output DOM contract passed; Windows device test still required.');
})().catch(error=>{console.error(error);process.exitCode=1;});
