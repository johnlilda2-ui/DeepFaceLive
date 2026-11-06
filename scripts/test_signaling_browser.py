#!/usr/bin/env python3
"""Browser integration with synthetic media and CPU inversion, never real webcam.

Run in CI with Playwright Chromium installed. Tests the shipped browser UI,
actual HTTP signaling and actual WebRTC return video. Does not claim GPU or
external-network verification.
"""
import asyncio
import json
import io
from pathlib import Path

from aiohttp import web
from playwright.async_api import async_playwright

from headless_webrtc_server import create_app, MANAGER

TOKEN = 'ci-only-not-a-real-credential-' + 'x' * 32


class DiagnosticWorker:
    def set_identity(self, image):
        self.identity_shape = image.shape
    def process_frame(self, image, preserve_mouth=False):
        return 255 - image, 'browser_diagnostic'
    def close(self):
        pass


async def wait_for_return(page):
    # Chromium can defer painting offscreen video below the expanded upload controls.
    await page.locator('#output').scroll_into_view_if_needed()
    await page.wait_for_function("""() => document.querySelector('#output').videoWidth > 0
        && document.querySelector('#local').readyState >= 2
        && document.querySelector('#output').readyState >= 2
        && Number(document.querySelector('#frames').textContent) >= 3""", timeout=45000)
    await page.evaluate("""() => Promise.all(['local','output'].map(id => new Promise(resolve =>
        document.getElementById(id).requestVideoFrameCallback(() => resolve()))))""")
    return await page.evaluate("""() => {
      const input = document.querySelector('#local'), output = document.querySelector('#output');
      function pixels(video) { const c=document.createElement('canvas'); c.width=160; c.height=120;
        const x=c.getContext('2d'); x.drawImage(video,0,0,160,120); return x.getImageData(0,0,160,120).data; }
      const a=pixels(input), b=pixels(output); let error=0, unchanged=0, count=0;
      for(let i=0;i<a.length;i++) if(i%4!==3){error+=Math.abs(b[i]-(255-a[i])); unchanged+=Math.abs(b[i]-a[i]); count++;}
      return {frames:Number(document.querySelector('#frames').textContent),
        outputWidth:output.videoWidth,outputHeight:output.videoHeight,
        inversionMae:error/count,originalMae:unchanged/count,
        connection:document.querySelector('#connection').textContent};
    }""")


async def main():
    artifacts = Path('artifacts/signaling-browser')
    artifacts.mkdir(parents=True, exist_ok=True)
    app = create_app(DiagnosticWorker(), TOKEN, idle_seconds=60)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    report = {'scope':'Chromium synthetic camera and demo over local HTTP/WebRTC',
              'gpu_verified':False, 'real_webcam_tested':False, 'sessions':[]}
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(args=[
                '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'])
            context = await browser.new_context(viewport={'width':1280,'height':900}, permissions=['camera'])
            page = await context.new_page()
            page_errors = []
            page.on('pageerror', lambda error: page_errors.append(str(error)))
            await page.goto(url)
            await page.get_by_label('Connection token', exact=True).fill('wrong-token-that-is-long-enough-for-validation')
            await page.get_by_role('button', name='Test demo', exact=True).click()
            await page.get_by_role('status').filter(has_text='Connection token rejected.').wait_for()
            assert app[MANAGER].session is None, 'Unauthorized browser obtained a session'
            await page.get_by_label('Connection token', exact=True).fill(TOKEN)
            from PIL import Image
            photo = io.BytesIO()
            Image.new('RGB',(128,128),(180,90,40)).save(photo,format='PNG')
            await page.get_by_label('Face photo to use for the swap',exact=True).set_input_files({
                'name':'synthetic-test-photo.png','mimeType':'image/png','buffer':photo.getvalue()})
            await page.get_by_role('button',name='Use this face photo',exact=True).click()
            await page.get_by_text('Confirm that you have permission to use this photo.',exact=True).wait_for()
            await page.get_by_label('I have permission to use this face photo.',exact=True).check()
            await page.get_by_role('button',name='Use this face photo',exact=True).click()
            await page.get_by_text('Face photo ready (revision 1).',exact=False).wait_for()
            assert app[MANAGER].worker.identity_shape == (128,128,3)
            report['photo_upload_ui_verified'] = True
            for mode in ('Test demo', 'Start camera'):
                if mode == 'Start camera':
                    await page.get_by_label('Preserve my real mouth and lip movements',exact=False).check()
                await page.get_by_role('button', name=mode, exact=True).click()
                stats = await wait_for_return(page)
                assert app[MANAGER].session['track'].swap_enabled
                assert stats['connection'] == 'connected', stats
                assert stats['inversionMae'] < 20 and stats['inversionMae'] < stats['originalMae'], stats
                assert app[MANAGER].session['track'].preserve_mouth == (mode == 'Start camera')
                await page.evaluate("window.savedTestTracks = document.querySelector('#local').srcObject.getTracks()")
                audio_tracks = await page.evaluate("document.querySelector('#local').srcObject.getAudioTracks().length")
                assert audio_tracks == 0, 'Audio was captured'
                if mode == 'Start camera':
                    failures = {'enabled': True, 'count': 0}
                    async def interrupt_status(route):
                        if route.request.method == 'GET' and failures['enabled']:
                            failures['count'] += 1
                            await route.fulfill(status=530, body='Temporary tunnel unavailable')
                        else:
                            await route.continue_()
                    await page.route('**/session/*', interrupt_status)
                    await page.get_by_role('status').filter(has_text='Cloud status link interrupted.').wait_for()
                    before = await page.evaluate("document.querySelector('#output').getVideoPlaybackQuality().totalVideoFrames")
                    await page.wait_for_function("previous => document.querySelector('#output').getVideoPlaybackQuality().totalVideoFrames > previous+3", arg=before)
                    assert await page.evaluate("window.savedTestTracks.every(t=>t.readyState==='live')"), 'Transient HTTP error stopped camera'
                    failures['enabled'] = False
                    await page.get_by_role('status').filter(has_text='Cloud connection recovered.').wait_for()
                    await page.unroute('**/session/*', interrupt_status)
                    report['transient_530_preserved_video'] = True
                await page.screenshot(path=str(artifacts / ('demo.png' if mode == 'Test demo' else 'synthetic-camera.png')))
                await page.get_by_role('button', name='Stop', exact=True).click()
                await page.get_by_role('status').filter(has_text='Camera and cloud session released.').wait_for(timeout=10000)
                assert await page.evaluate("window.savedTestTracks.every(t=>t.readyState==='ended')"), 'Camera tracks leaked'
                assert app[MANAGER].session is None, 'Server slot leaked after Stop'
                report['sessions'].append({'mode':mode,**stats,'audio_tracks':audio_tracks,'stop_released_resources':True})
            await page.get_by_label('Swap face (turn off',exact=False).uncheck()
            await page.get_by_role('button',name='Test demo',exact=True).click()
            bypass = await wait_for_return(page)
            assert bypass['originalMae'] < 20 and bypass['originalMae'] < bypass['inversionMae'], bypass
            assert app[MANAGER].session['track'].last_status == 'bypass'
            report['bypass_comparison_verified'] = True
            await page.get_by_role('button',name='Stop',exact=True).click()
            await page.get_by_role('status').filter(has_text='Camera and cloud session released.').wait_for()
            await page.get_by_label('Swap face (turn off',exact=False).check()
            await page.get_by_label('Preserve my real mouth and lip movements',exact=False).uncheck()
            # A persistent outage must still release local media and the server slot.
            await page.get_by_role('button', name='Test demo', exact=True).click()
            await wait_for_return(page)
            await page.evaluate("window.savedTestTracks = document.querySelector('#local').srcObject.getTracks()")
            async def persistent_outage(route):
                if route.request.method == 'GET':
                    await route.fulfill(status=530, body='Tunnel expired')
                else:
                    await route.continue_()
            await page.route('**/session/*', persistent_outage)
            await page.get_by_role('status').filter(has_text='Cloud link unavailable or expired.').wait_for(timeout=25000)
            assert await page.evaluate("window.savedTestTracks.every(t=>t.readyState==='ended')"), 'Persistent outage leaked media'
            assert app[MANAGER].session is None, 'Persistent outage leaked server session'
            await page.unroute('**/session/*', persistent_outage)
            report['persistent_530_released_resources'] = True
            app[MANAGER].worker.mode = 'portrait'
            calibrations = []
            app[MANAGER].worker.reset_motion = lambda:calibrations.append(True)
            await page.get_by_label('Preserve my real mouth and lip movements',exact=False).check()
            await page.get_by_role('button',name='Test demo',exact=True).click()
            await wait_for_return(page)
            assert await page.get_by_label('Preserve my real mouth and lip movements',exact=False).is_disabled()
            assert not app[MANAGER].session['track'].preserve_mouth
            assert calibrations
            await page.get_by_text('Portrait avatar: the photo supplies hair',exact=False).wait_for()
            await page.get_by_role('button',name='Stop',exact=True).click()
            await page.get_by_role('status').filter(has_text='Camera and cloud session released.').wait_for()
            report['portrait_ui_options_verified'] = True  # CPU UI only, not avatar synthesis.
            await page.set_viewport_size({'width':390,'height':844})
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
            await page.screenshot(path=str(artifacts / 'mobile.png'), full_page=True)
            assert not page_errors, page_errors
            await browser.close()
        report['status'] = 'passed'
        (artifacts / 'browser-report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
