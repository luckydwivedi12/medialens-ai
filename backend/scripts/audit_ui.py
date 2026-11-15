"""Phase 13 browser audit. Use isolated headless Edge, real local API and validation assets."""
import os, base64, csv, hashlib, json, subprocess, tempfile, time, urllib.request
from pathlib import Path
from typing import Any
from websockets.sync.client import connect

root = Path(__file__).resolve().parents[2]
out = Path(os.environ.get('MEDIALENS_AUDIT_OUTPUT', root / 'docs' / 'phase13'))
out.mkdir(exist_ok=True)
meta = json.loads(Path(r'D:\MediaLensData\models\medialens_mobilenetv3\model_metadata.json').read_text())['dataset']
manifest = Path(meta['split_manifest'])
assert 'sha256:'+hashlib.sha256(manifest.read_bytes()).hexdigest() == meta['split_manifest_checksum']
with manifest.open(encoding='utf-8-sig', newline='') as f:
    row = next(r for r in csv.DictReader(f) if r['split']=='validation' and r['paired_subset_candidate']=='True' and r['status']=='crop_saved')
img = Path(row['crop_path'])
vid = Path(r'D:\MediaLensData\raw\faceforensics') / row['relative_path']
for path,key in [(img,'checksum'),(vid,'source_video_checksum')]:
    assert 'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()==row[key]
with tempfile.TemporaryDirectory() as profile:
    proc=subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe','--headless=new','--disable-gpu','--no-first-run','--disable-extensions','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(150):
            portfile=Path(profile)/'DevToolsActivePort'
            if portfile.exists(): break
            time.sleep(.1)
        port=portfile.read_text().splitlines()[0]
        url=next(t['webSocketDebuggerUrl'] for t in json.load(urllib.request.urlopen('http://localhost:'+port+'/json')) if t['type']=='page')
        with connect(url,max_size=32000000) as ws:
            serial=0
            errors=[]
            def call(method: str, **params: Any) -> Any:
                global serial
                serial+=1; ws.send(json.dumps(dict(id=serial,method=method,params=params)))
                while True:
                    m=json.loads(ws.recv(timeout=30))
                    if m.get('method')=='Runtime.exceptionThrown' or (m.get('method')=='Runtime.consoleAPICalled' and m['params']['type']=='error'): errors.append(m['method'])
                    if m.get('id')==serial:
                        assert 'error' not in m,m
                        return m['result']
            def js(expr: str) -> Any:
                r=call('Runtime.evaluate',expression=expr,returnByValue=True,awaitPromise=True)
                assert 'exceptionDetails' not in r,r
                return r['result'].get('value')
            def wait(expr: str) -> None:
                for _ in range(250):
                    if js("Boolean("+expr+")"): return
                    time.sleep(.1)
                raise AssertionError(expr)
            def shot(name: str) -> None:
                (out/(name+'.png')).write_bytes(base64.b64decode(call('Page.captureScreenshot',format='png')['data']))
            def width(w: int) -> None:
                call('Emulation.setDeviceMetricsOverride',width=w,height=1000 if w>400 else 844,deviceScaleFactor=1,mobile=False)
                assert js('document.documentElement.scrollWidth <= innerWidth')
            def select(path: Path) -> None:
                doc=call('DOM.getDocument')['root']['nodeId']
                node=call('DOM.querySelector',nodeId=doc,selector='.workspace:not([hidden]) input[type=file]')['nodeId']
                call('DOM.setFileInputFiles',nodeId=node,files=[str(path)])
            def click(label: str) -> None:
                js("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim().startsWith("+json.dumps(label)+") && b.offsetWidth).click()")
            call('Page.enable');call('Runtime.enable');call('Page.navigate',url='http://localhost:5175')
            wait("document.body?.innerText.includes('ready · CPU')")
            for w in [1440,1366,768,390]:
                width(w);shot('initial-'+str(w))
            width(1440)
            call('Input.dispatchKeyEvent',type='keyDown',key='Tab',windowsVirtualKeyCode=9);call('Input.dispatchKeyEvent',type='keyUp',key='Tab',windowsVirtualKeyCode=9)
            assert js("document.activeElement.matches(':focus-visible')")
            select(img);wait("document.querySelector('.image-preview')?.naturalWidth>0");shot('image-upload')
            js("document.addEventListener('change', e=>window.auditFile=e.target.files[0], {capture:true,once:true})")
            select(img)
            js("(()=>{let dt=new DataTransfer();dt.items.add(window.auditFile);document.querySelector('.dropzone').dispatchEvent(new DragEvent('drop',{bubbles:true,dataTransfer:dt}))})()")
            click('Analyze Image');wait("document.querySelector('.assessment')");shot('image-result')
            print('Image:',js("document.querySelector('.probabilities').innerText"))
            select(root/'README.md');wait("document.querySelector('[role=alert]')");shot('error')
            click('Video Analysis');select(vid);wait("document.querySelector('video') && !Array.from(document.querySelectorAll('button')).find(b=>b.textContent.startsWith('Analyze Video')).disabled");shot('video-upload')
            click('Analyze Video');wait("document.body.innerText.includes('PROCESSING')");shot('video-processing')
            wait("document.body.innerText.includes('Average model fake score')");shot('video-result')
            print('Video:',js("document.querySelector('.workspace:not([hidden]) dl').innerText"))
            js("document.querySelector('summary').click()")
            for w in [1440,1366,768,390]: width(w)
            js("document.querySelector('.workspace:not([hidden]) .results-panel').scrollIntoView()");shot('mobile-result')
            click('About');assert js("document.querySelector('#about').offsetWidth>0")
            print('JS exceptions and console.error:',errors);assert not errors
    finally:
        proc.terminate();proc.wait(timeout=15)

