export type ModelStatus = { video_ready?: boolean; model_ready: boolean; model_version: string | null; checkpoint_available: boolean; threshold: number | null }
export type Prediction = { success: true; predicted_label: 'real' | 'fake'; probabilities: { real: number; fake: number }; threshold: number; model_name: string; model_version: string; warning: string }

export const API_BASE = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL ? String(import.meta.env.VITE_API_BASE_URL) : '').replace(/\/+$/, '')

export function apiUrl(path: string): string {
  const p = path.startsWith('/') ? path : `/${path}`
  return API_BASE ? `${API_BASE}${p}` : p
}

export function isPrediction(value: unknown): value is Prediction {
  if (!value || typeof value !== 'object') return false
  const p = value as Prediction
  return p.success === true && ['real', 'fake'].includes(p.predicted_label)
    && typeof p.model_name === 'string' && typeof p.model_version === 'string' && typeof p.warning === 'string'
    && typeof p.threshold === 'number' && p.threshold > 0 && p.threshold < 1
    && typeof p.probabilities?.real === 'number' && typeof p.probabilities?.fake === 'number'
    && p.probabilities.real >= 0 && p.probabilities.real <= 1 && p.probabilities.fake >= 0 && p.probabilities.fake <= 1
    && Math.abs(p.probabilities.real + p.probabilities.fake - 1) < 0.00001
}

export async function predict(file: File, signal: AbortSignal): Promise<Prediction> {
  const form = new FormData(); form.append('file', file)
  const response = await fetch(apiUrl('/api/v1/detect/image'), { method: 'POST', body: form, signal })
  const payload: unknown = await response.json()
  if (!response.ok) {
    const message = (payload as { error?: { message?: string } })?.error?.message
    throw new Error(message || 'Analysis failed. Please try again.')
  }
  if (!isPrediction(payload)) throw new Error('Unexpected prediction response. Please retry.')
  return payload
}
export type VideoFace = { bounding_box: number[]; crop_box: number[] | null; detection_confidence: number; status: 'analyzed' | 'invalid_crop'; probabilities: {real: number; fake: number} | null; predicted_label: 'real' | 'fake' | null }
export type VideoFrame = { frame_index: number; timestamp_seconds: number; timestamp_method: string; status: 'analyzed' | 'no_faces' | 'invalid_crops' | 'decode_failed'; average_fake_score: number | null; faces: VideoFace[]; error: string | null }
export type VideoPrediction = { success: true; status: 'analyzed' | 'insufficient_analysis'; predicted_label: 'real' | 'fake' | null; average_fake_score: number | null; average_real_score: number | null; threshold: number; sampled_frames: number; successfully_analyzed_frames: number; frames_without_faces: number; faces_analyzed: number; frames_above_threshold: number; percentage_frames_above_threshold: number; decoding_failures: number; processing_time_seconds: number; duration_seconds: number; frame_level_results: VideoFrame[]; model_name: string; model_version: string; warning: string; limitations: string[] }

export function isVideoPrediction(value: unknown): value is VideoPrediction {
  if (!value || typeof value !== 'object') return false
  const p = value as VideoPrediction
  const score = (n: unknown) => typeof n === 'number' && Number.isFinite(n) && n >= 0 && n <= 1
  const count = (n: unknown) => Number.isInteger(n) && (n as number) >= 0
  return p.success === true && ['analyzed', 'insufficient_analysis'].includes(p.status)
    && (p.status === 'analyzed' ? ['real', 'fake'].includes(p.predicted_label!) && p.successfully_analyzed_frames >= 3 : p.predicted_label === null && p.successfully_analyzed_frames < 3)
    && (p.average_fake_score === null && p.average_real_score === null || score(p.average_fake_score) && score(p.average_real_score) && Math.abs(p.average_fake_score! + p.average_real_score! - 1) < .00001)
    && score(p.threshold) && p.threshold > 0 && p.threshold < 1
    && [p.sampled_frames, p.successfully_analyzed_frames, p.frames_without_faces, p.faces_analyzed, p.frames_above_threshold, p.decoding_failures].every(count)
    && p.sampled_frames <= 24 && p.successfully_analyzed_frames <= p.sampled_frames
    && Number.isFinite(p.processing_time_seconds) && p.processing_time_seconds >= 0
    && Number.isFinite(p.duration_seconds) && p.duration_seconds > 0 && p.duration_seconds <= 120
    && Number.isFinite(p.percentage_frames_above_threshold) && p.percentage_frames_above_threshold >= 0 && p.percentage_frames_above_threshold <= 100
    && typeof p.model_name === 'string' && typeof p.model_version === 'string' && typeof p.warning === 'string'
    && Array.isArray(p.limitations) && p.limitations.every(x => typeof x === 'string')
    && Array.isArray(p.frame_level_results) && p.frame_level_results.length === p.sampled_frames
    && p.frame_level_results.every(f => count(f.frame_index) && Number.isFinite(f.timestamp_seconds) && f.timestamp_seconds >= 0
      && typeof f.timestamp_method === 'string' && ['analyzed', 'no_faces', 'invalid_crops', 'decode_failed'].includes(f.status)
      && (f.average_fake_score === null || score(f.average_fake_score)) && Array.isArray(f.faces)
      && f.faces.every(face => ['analyzed', 'invalid_crop'].includes(face.status) && Array.isArray(face.bounding_box) && face.bounding_box.length === 4 && face.bounding_box.every(Number.isFinite)
        && score(face.detection_confidence) && (face.status === 'invalid_crop' || face.probabilities && score(face.probabilities.fake) && score(face.probabilities.real) && ['real', 'fake'].includes(face.predicted_label!))))
}

export async function predictVideo(file: File, signal: AbortSignal): Promise<VideoPrediction> {
  const form = new FormData(); form.append('file', file)
  const response = await fetch(apiUrl('/api/v1/detect/video'), { method: 'POST', body: form, signal })
  const payload: unknown = await response.json()
  if (!response.ok) throw new Error((payload as {error?: {message?: string}})?.error?.message || 'Video analysis failed. Please retry.')
  if (!isVideoPrediction(payload)) throw new Error('Unexpected video response. Please retry.')
  return payload
}
