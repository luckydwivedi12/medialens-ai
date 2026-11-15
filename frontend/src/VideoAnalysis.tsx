import { useEffect, useRef, useState } from 'react'
import { predictVideo } from './api'
import type { VideoPrediction } from './api'
import { validateDuration, validateVideo } from './upload'

export function VideoResults({ result }: { result: VideoPrediction }) {
  return <><p className="eyebrow">Experimental video assessment</p>
    <h3 className="assessment">{result.status === 'insufficient_analysis' ? 'Insufficient analysis' : result.predicted_label === 'fake' ? 'Likely manipulated' : 'Likely unmanipulated'}</h3>
    <p>This assessment can be wrong. {result.status === 'insufficient_analysis' && 'At least three frames with valid faces are required for classification.'}</p>
    <dl className="model-details">
      <div><dt>Average model fake score</dt><dd>{result.average_fake_score === null ? 'Unavailable' : `${(result.average_fake_score * 100).toFixed(1)}%`}</dd></div>
      <div><dt>Analyzed frames</dt><dd>{result.successfully_analyzed_frames} / {result.sampled_frames}</dd></div>
      <div><dt>Faces analyzed</dt><dd>{result.faces_analyzed}</dd></div>
      <div><dt>Frames without faces</dt><dd>{result.frames_without_faces}</dd></div>
      <div><dt>Frames at or above threshold</dt><dd>{result.frames_above_threshold} {result.successfully_analyzed_frames > 0 ? `(${result.percentage_frames_above_threshold.toFixed(1)}% of analyzed frames)` : '(no analyzed frames)'}</dd></div>
      <div><dt>Decoding failures</dt><dd>{result.decoding_failures}</dd></div>
      <div><dt>Threshold / processing time</dt><dd>{result.threshold.toFixed(2)} / {result.processing_time_seconds.toFixed(2)} s</dd></div>
      <div><dt>Model</dt><dd>{result.model_name} · {result.model_version}</dd></div>
    </dl>
    <p className="warning">{result.warning} The average score is not a calibrated probability of the entire video.</p>
    <details><summary>Frame-level assessment details</summary>{result.frame_level_results.map(frame => <div className="frame-detail" key={frame.frame_index}>
      <strong>Frame {frame.frame_index} · {frame.timestamp_seconds.toFixed(2)} s</strong><p>{frame.status.replaceAll('_', ' ')} · Fake score: {frame.average_fake_score === null ? 'unavailable' : `${(frame.average_fake_score * 100).toFixed(1)}%`} · {frame.faces.length} detections</p>
      <small>{frame.timestamp_method.replaceAll('_', ' ')}</small>{frame.error && <p>{frame.error}</p>}
      {frame.faces.map((face, i) => <p key={i}>Face {i + 1}: {face.predicted_label ?? 'invalid crop'}{face.probabilities && ` · real ${(face.probabilities.real * 100).toFixed(1)}% / fake ${(face.probabilities.fake * 100).toFixed(1)}%`} · box [{face.bounding_box.map(n => n.toFixed(0)).join(', ')}]</p>)}
    </div>)}</details>
    <ul className="limitations">{result.limitations.map(text => <li key={text}>{text}</li>)}</ul>
  </>
}

export default function VideoAnalysis({ ready }: { ready: boolean }) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')
  const [previewNote, setPreviewNote] = useState('')
  const [durationError, setDurationError] = useState('')
  const [result, setResult] = useState<VideoPrediction | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const request = useRef<AbortController | null>(null)
  useEffect(() => {
    if (!file) { setPreview(''); return }
    const url = URL.createObjectURL(file); setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])
  useEffect(() => () => request.current?.abort(), [])
  function select(selected?: File) {
    if (!selected || busy) return
    setResult(null); setError(''); setDurationError(''); setPreviewNote('')
    const message = validateVideo(selected)
    setFile(message ? null : selected); setError(message ?? '')
  }
  async function analyze() {
    if (!file || busy || durationError) return
    setBusy(true); setError(''); setResult(null)
    const controller = new AbortController(); request.current = controller
    const timeout = setTimeout(() => controller.abort(), 180000)
    try { setResult(await predictVideo(file, controller.signal)) }
    catch (cause) {
      if (controller.signal.aborted) {
        setError('Video analysis timed out. The server may be busy or waking up; please retry.')
      } else {
        setError(cause instanceof Error ? cause.message : 'Video analysis failed.')
      }
    }
    finally { clearTimeout(timeout); request.current = null; setBusy(false) }
  }
  return <div className="workspace">
    <section className="upload-panel" aria-labelledby="video-heading">
      <div className="section-title"><h2 id="video-heading"><span>01</span> Your video</h2><span className="small-label">VIDEO ANALYSIS</span></div>
      <div className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-image' : ''}`} onDragOver={event => { event.preventDefault(); if (!busy) setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); if (busy) return; if (event.dataTransfer.files.length !== 1) { setError('Choose one video at a time.'); return } select(event.dataTransfer.files[0]) }}>
        {file && preview ? <><video key={preview} className="video-preview" src={preview} controls preload="metadata" onLoadedMetadata={event => { setDurationError(validateDuration(event.currentTarget.duration) ?? '') }} onError={() => setPreviewNote('Your browser cannot preview this codec. The server will validate format and duration before analysis.')} /><div className="file-detail"><span>{file.name}</span><span>{(file.size / 1024 / 1024).toFixed(2)} MB</span></div></> : <div className="empty-upload"><h3>Drop a video here</h3><p>or choose one from your device</p><button className="browse-button" onClick={() => input.current?.click()}>Browse video ↗</button><p className="format-note">MP4, MOV, AVI or MKV · Up to 50 MB · 120 seconds</p></div>}
      </div>
      <input className="file-input" ref={input} type="file" tabIndex={-1} accept=".mp4,.mov,.avi,.mkv" aria-label="Upload video" disabled={busy} onChange={event => { select(event.target.files?.[0]); event.target.value = '' }} />
      {file && <div className="file-actions"><button disabled={busy} onClick={() => input.current?.click()}>Replace video</button><button disabled={busy} onClick={() => { setFile(null); setResult(null); setError(''); setDurationError(''); setPreviewNote('') }}>Remove</button></div>}
      {previewNote && <p>{previewNote}</p>}
      <p className="crop-note">Twelve frames sampled across the video. YuNet locates faces; the existing image classifier assesses every valid facial crop.</p>
      <button className="analyze-button" disabled={!file || busy || !ready || !!durationError} onClick={() => void analyze()}>{busy ? 'Analyzing video…' : 'Analyze Video'}<span aria-hidden="true">→</span></button>
      <p className="privacy-note">Processed locally in temporary storage. Your video is not permanently retained.</p>
      {(error || durationError) && <div className="error-message" role="alert">{error || durationError}</div>}
      {!ready && <p className="status-help">Video analysis requires both the classifier and YuNet detector. Check backend configuration and restart if unavailable.</p>}
    </section>
    <section className="results-panel" aria-busy={busy} aria-labelledby="video-result-heading"><div className="section-title"><h2 id="video-result-heading"><span>02</span> Model assessment</h2><span className="small-label">{busy ? 'PROCESSING' : result ? 'COMPLETE' : 'AWAITING VIDEO'}</span></div>
      <div className="result-content" aria-live="polite">{busy ? <div className="result-empty"><span className="processing-indicator" /><h3>Analyzing sampled faces.</h3><p>Validating the video, locating faces and running CPU inference. Processing time depends on the video.</p></div> : result ? <VideoResults result={result} /> : <div className="result-empty"><h3>Awaiting video analysis</h3><p>Upload a video to review sampled facial assessments. Evidence for review. Never a final verdict.</p></div>}</div>
    </section>
  </div>
}
