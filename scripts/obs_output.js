'use strict';
if (window.opener) window.opener.postMessage({type:'dfl-output-ready'},location.origin);
setInterval(() => {
  if (!window.opener || window.opener.closed) {
    document.getElementById('processedVideo').srcObject = null;
    document.getElementById('notice').hidden = false;
  }
}, 500);
