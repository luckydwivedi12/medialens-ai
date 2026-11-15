import { useEffect, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { apiUrl, predict } from './api'
import type { ModelStatus, Prediction } from './api'
import VideoAnalysis from './VideoAnalysis'
import { validateImage } from './upload'

type Connection = 'checking' | 'connected' | 'unavailable'

export default function App() {
  const [mode, setMode] = useState<'image' | 'video' | 'about'>('image')
  const [connection, setConnection] = useState<Connection>('checking')
  const [model, setModel] = useState<ModelStatus | null>(null)
  const [attempt, setAttempt] = useState(0)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<Prediction | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const request = useRef<AbortController | null>(null)

  useEffect(() => {
    let active = true
    const controller = new AbortController()
    const timeout = window.setTimeout(() => controller.abort(), 35000)
    setConnection('checking'); setModel(null)
    void (async () => {
      try {
        const health = await fetch(apiUrl('/api/health'), { signal: controller.signal })
        const payload = await health.json()
        if (!health.ok || payload?.status !== 'ok' || payload?.service !== 'medialens-ai') throw new Error()
        if (active) setConnection('connected')
        const response = await fetch(apiUrl('/api/v1/model/status'), { signal: controller.signal })
        const status = await response.json()
        if (!response.ok || typeof status?.model_ready !== 'boolean') throw new Error()
        if (active) setModel(status as ModelStatus)
      } catch {
        if (active) { setModel(null); setConnection(current => current === 'connected' ? current : 'unavailable') }
      } finally { window.clearTimeout(timeout) }
    })()
    return () => { active = false; window.clearTimeout(timeout); controller.abort() }
  }, [attempt])
  useEffect(() => {
    if (!file) { setPreview(''); return }
    const url = URL.createObjectURL(file); setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])
  useEffect(() => () => request.current?.abort(), [])

  function selectFile(selected?: File) {
    if (!selected || busy) return
    setResult(null); setError('')
    const validationError = validateImage(selected)
    if (validationError) { setFile(null); setError(validationError); return }
    setFile(selected)
  }
  function drop(event: DragEvent<HTMLElement>) {
    event.preventDefault(); setDragging(false)
    if (event.dataTransfer.files.length !== 1) { setError('Choose one image at a time.'); return }
    selectFile(event.dataTransfer.files[0])
  }
  async function analyze() {
    if (!file || busy) return
    setBusy(true); setError(''); setResult(null)
    const controller = new AbortController(); request.current = controller
    const timeout = window.setTimeout(() => controller.abort(), 60000)
    try { setResult(await predict(file, controller.signal)) }
    catch (cause) { setError(controller.signal.aborted ? 'Analysis timed out. Check the backend and retry.' : cause instanceof Error ? cause.message : 'The backend could not be reached. Please retry.') }
    finally { window.clearTimeout(timeout); request.current = null; setBusy(false) }
  }

  return <div className="app-shell">
    <header className="site-header">
      <a className="wordmark" href="#analyze" onClick={() => setMode('image')}><span className="brand-icon" aria-hidden="true">◉</span> MediaLens <span>AI</span></a>
      <nav aria-label="Main navigation"><button aria-pressed={mode === 'image'} onClick={() => setMode('image')}>Image Analysis</button><button aria-pressed={mode === 'video'} onClick={() => setMode('video')}>Video Analysis</button><button aria-pressed={mode === 'about'} onClick={() => setMode('about')}>About</button></nav>
      <span className="research-label">RESEARCH PREVIEW</span>
    </header>
    <main>
      <section className="hero" id="analyze"><p className="eyebrow">Media forensics</p><h1>Review facial media<span>.</span></h1><p className="introduction">An experimental AI system for detecting manipulated facial imagery.<br className="desktop-break" /> A closer look, with the context to make your own assessment.</p></section>
      {mode === 'video' && <VideoAnalysis ready={connection === 'connected' && !!model?.video_ready} />}
      <div className="workspace" hidden={mode !== 'image'}>
        <section className="upload-panel" aria-labelledby="upload-heading">
          <div className="section-title"><h2 id="upload-heading"><span>01</span> Your image</h2><span className="small-label">IMAGE ANALYSIS</span></div>
          <div className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-image' : ''}`} onDragOver={event => { event.preventDefault(); if (!busy) setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={drop}>
            {file && preview ? <><img className="image-preview" src={preview} alt="Selected image for analysis" onError={() => { setFile(null); setError('This image could not be previewed. Choose a valid image.') }} /><div className="file-detail"><span title={file.name}>{file.name}</span><span>{(file.size / 1024 / 1024).toFixed(2)} MB</span></div></> : <div className="empty-upload">
              <svg width="42" height="42" viewBox="0 0 42 42" fill="none" aria-hidden="true"><rect x="5" y="5" width="32" height="32" rx="5" stroke="currentColor" strokeWidth="1.3"/><circle cx="15" cy="15" r="3" stroke="currentColor" strokeWidth="1.3"/><path d="m6 30 10-9 7 6 6-7 8 9" stroke="currentColor" strokeWidth="1.3"/></svg>
              <h3>Drop an image here</h3><p>or choose one from your device</p><button className="browse-button" type="button" onClick={() => input.current?.click()}>Browse image <span aria-hidden="true">↗</span></button><p className="format-note">JPG, PNG or WEBP · Up to 10 MB</p>
            </div>}
          </div>
          <input ref={input} type="file" tabIndex={-1} accept="image/jpeg,image/png,image/webp" onChange={event => { selectFile(event.target.files?.[0]); event.target.value = '' }} className="file-input" aria-label="Upload image" disabled={busy} />
          {file && <div className="file-actions"><button type="button" disabled={busy} onClick={() => input.current?.click()}>Replace image</button><button type="button" disabled={busy} onClick={() => { setFile(null); setResult(null); setError('') }}>Remove</button></div>}
          <p className="crop-note">For the most relevant assessment, upload a single facial crop. The model does not locate faces automatically.</p>
          <button className="analyze-button" disabled={!file || busy || !model?.model_ready || connection !== 'connected'} onClick={() => void analyze()}>{busy ? 'Analyzing image…' : error && file ? 'Retry analysis' : 'Analyze Image'}<span aria-hidden="true">{busy ? '···' : '→'}</span></button>
          <p className="privacy-note">Processed locally. Your image is not permanently retained.</p>{error && <div className="error-message" role="alert">{error}</div>}
        </section>
        <section className="results-panel" aria-labelledby="result-heading" aria-busy={busy}>
          <div className="section-title"><h2 id="result-heading"><span>02</span> Model assessment</h2><span className="small-label">{result ? 'COMPLETE' : busy ? 'PROCESSING' : 'AWAITING IMAGE'}</span></div>
          <div aria-live="polite" className="result-content">
            {busy ? <div className="result-empty"><span className="processing-indicator" /><h3>Analyzing facial crop</h3><p>The local model is processing your image.<br />This may take a moment on CPU.</p></div> : result ? <>
              <p className="eyebrow result-kicker">Model assessment</p><h3 className="assessment">{result.predicted_label === 'fake' ? 'Likely manipulated' : 'Likely unmanipulated'}</h3>
              <p className="result-description">{result.predicted_label === 'fake' ? 'The model found patterns associated with manipulated facial crops.' : 'The model found patterns associated with original facial crops.'} This assessment can be wrong.</p>
              <div className="probabilities">{(['real', 'fake'] as const).map(label => <div className={`probability ${label}`} key={label}><div><span>{label === 'real' ? 'Real' : 'Fake'}</span><strong>{(result.probabilities[label] * 100).toFixed(1)}%</strong></div><meter min={0} max={1} value={result.probabilities[label]} aria-label={`${label} model-estimated probability`} /></div>)}</div>
              <p className="probability-note">Model-estimated class probabilities, not guaranteed authenticity.</p>
              <dl className="model-details"><div><dt>Model</dt><dd>{result.model_name}</dd></div><div><dt>Decision threshold</dt><dd>{result.threshold.toFixed(2)} (fake)</dd></div><div><dt>Version</dt><dd>{result.model_version}</dd></div></dl><p className="warning">{result.warning}</p>
            </> : <div className="result-empty"><span className="result-symbol" aria-hidden="true">◎</span><h3>Awaiting image analysis</h3><p>Upload an image and run an analysis.<br />Your model assessment will appear here.</p><span className="empty-caption">Evidence for review. Never a final verdict.</span></div>}
          </div>
        </section>
      </div>
      <div className="service-bar" role="status"><div><span className={`status-dot ${connection === 'connected' ? 'ready' : ''}`} />Backend: {connection === 'connected' ? 'connected' : connection === 'checking' ? 'checking' : 'unavailable'}<span className="service-divider">/</span><span className={`status-dot ${model?.model_ready ? 'ready' : ''}`} />Model: {model?.model_ready ? 'ready · CPU' : connection === 'checking' ? 'checking' : 'unavailable'}</div><button disabled={connection === 'checking'} onClick={() => setAttempt(value => value + 1)}>Check connection ↻</button></div>
      {connection !== 'checking' && !model?.model_ready && <p className="status-help">{connection === 'unavailable' ? 'Backend is unreachable. If waking from cold start, wait a moment and click Check connection ↻.' : 'The backend is responding, but the model is unavailable. Check the checkpoint configuration and restart the backend.'}</p>}
      <section hidden={mode !== 'about'} className="about-section" id="about" aria-labelledby="about-heading"><div><p className="eyebrow">About MediaLens AI</p><h2 id="about-heading">Model scope and limitations</h2></div><div className="about-copy"><p>MediaLens AI uses a MobileNetV3-Large classifier trained on a small, matched subset of facial crops from FaceForensics++: original images and Deepfakes manipulations at c23 compression.</p><p>This experimental model can miss manipulations and flag original images. Its labels are provisional, its scores are uncalibrated, and performance on other sources, methods and full images is not established. Video analysis applies this same classifier to YuNet facial crops from sampled frames. Video-level performance has not been validated. Audio is not analyzed.</p><p>Use the assessment alongside source verification and independent review. An AI prediction is never proof of authenticity.</p></div></section>
    </main><footer><span>MediaLens AI <span className="footer-tagline">/ Media forensics</span></span><span>Local processing. Human judgment.</span></footer>
  </div>
}
