# Phase 12 — experimental video deepfake detection

Implemented on 2026-11-15 (Asia/Calcutta). This workflow samples facial crops from
uploaded videos and applies the existing Phase 10 image classifier. It is not a
validated video detector or proof of authenticity. No training, threshold tuning,
dataset modification, locked-test access, dependency installation or GitHub push
was performed.

## Running and configuration

Use the backend/frontend commands in [application_setup.md](application_setup.md).
Restart the backend after updating from Phase 11 so it loads the video service.
One worker, CPU only, loopback binding are appropriate for this machine.

The classifier uses the same application-lifetime `ImageInferenceService` as the
image endpoint; no second classifier or model-loading implementation is created.
Default checkpoint: `D:\MediaLensData\models\medialens_mobilenetv3\best_model.pth`.
`MEDIALENS_CHECKPOINT` retains its existing meaning. Weights and trusted sibling
metadata stay outside the repository and are never changed.

`MEDIALENS_YUNET_DIRECTORY` defaults to `D:\MediaLensData\models\yunet`. It must
contain the existing `face_detection_yunet_2026may.onnx` and its pinned provenance
`compatibility_20261115T084812012784Z_d75d8ac01dcc47eba2991d993b209523.json`.
The existing detector verifies model SHA-256 and provenance. The application sets
`OPENCV_FORCE_DNN_ENGINE=2` before importing OpenCV; an incompatible pre-existing
engine setting makes video initialization fail instead of silently falling back.
A detector failure preserves image inference; `/api/v1/model/status` now includes
`video_ready`, requiring both the classifier and detector.

The consolidated Vite proxy on localhost:5175 forwards /api to 127.0.0.1:8000. See [application setup](application_setup.md) for the Windows launcher.

## Upload API and validation

`POST /api/v1/detect/video`, multipart field **file**:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/detect/video `
  -F "file=@C:\path\to\clip.mp4;type=video/mp4"
```

Supported containers and MIME types:

| Extension | Media type |
| --- | --- |
| MP4 | video/mp4 |
| MOV | video/quicktime |
| AVI | video/x-msvideo or video/avi |
| MKV | video/x-matroska |

`application/octet-stream` is also accepted with a supported extension for clients
without a video MIME mapping. Filename extensions and MIME alone are insufficient:
check RIFF/AVI magic, ISO BMFF/QuickTime brands (or legacy MOV atom), or EBML with
Matroska DocType. WebM is rejected. OpenCV must then open and decode the video;
codec support depends on the installed OpenCV/FFmpeg build.

Maximum file size is **50 MiB**, labeled 50 MB in the UI; maximum duration is
**120 seconds**. A separate 64 KiB multipart allowance bounds request bodies even
with absent/dishonest Content-Length. Requests spool after 1 MiB, and the upload
is copied in 1 MiB chunks to a randomly named temporary directory. Client filenames
never become server paths. All handles and temporary files close on success or
failure. No uploaded media, frames or crops are archived or returned.

The validator requires positive finite frame count/FPS/dimensions, at most 4K
(3840×2160 pixel area), and at most 120 FPS. It decodes sequentially, retaining
only the current frame, to catch truncation/corruption outside sampled positions.
Decoded count must match reported count; estimated duration and decoder timestamp
must stay within the duration limit. Metadata inconsistencies are rejected rather
than guessed. This may reject otherwise playable variable-frame-rate files.

Errors use the existing `{success:false,error:{code,message}}` envelope: 400 for
invalid/corrupt/over-duration/resource-limit media, 413 for oversized requests or
files, 415 for unsupported/mismatched declared formats, 422 for a missing field,
503 for classifier/detector unavailability, and 500 for processing/inference
failures. Public responses never include exception details, media bytes or server
paths. File size is checked server-side even if browser validation was bypassed.

## Sampling, faces and model inference

For N frames and target T=12, sample sorted unique indices
`min(N-1, floor(N*(i+0.5)/T))`, i=0..T-1. These are uniformly spaced temporal-bin
centers and avoid the fragile exact end frame. Short clips yield fewer unique
indices. The helper accepts at most 24 samples; the public endpoint always uses
12. The existing streaming frame extractor is extended with optional indices,
preserving earlier phases' five-frame defaults. Seek/read, decoded position and
frame dimensions are checked, and `VideoCapture` is released in `finally`, even
when inference interrupts iteration. Sample decoding failures are explicit rows.
Timestamp uses decoder POS_MSEC where available, otherwise index/FPS, with the
method recorded. Frame-index sampling approximates time for variable-rate media.

YuNet detects every face at original resolution using the existing confidence
0.9, NMS 0.3 and top_k 5000. The existing crop function uses margin 0.2 per side:
`ceil(max(width,height)*1.4)`, capped to the shorter image dimension and shifted
inside image boundaries. Crops are square, native resolution, without alignment
or tracking. Invalid crops retain their detection box with `invalid_crop` status.
Every valid face is converted BGR→RGB and passed to the shared classifier.

Shared model transform: RGB, direct antialiased bilinear resize to 224×224,
float [0,1], ImageNet mean [0.485,0.456,0.406] and std [0.229,0.224,0.225].
No augmentation, masks, presumed manipulated-target selection or input-video
labels enter inference. CPU model forwards use the existing inference lock and
`torch.inference_mode()`. YuNet validation and video processing are serialized
per process, and only one frame/crop is processed at a time.

## Scores and response

Each valid face records real/fake softmax values and its predicted class using
the existing checkpoint threshold **0.5**, real=0/fake=1. Average valid face fake
scores within each frame, then average the successfully analyzed frame scores.
Real scores use the same two-stage average. Frames have equal weight even when
some contain more faces. No-face, invalid-only and failed frames are excluded
from score means, never treated as real or assigned zero fake scores.

Count frames whose average fake score is **at or above** the saved threshold;
the reported percentage denominator is successfully analyzed frames. With zero
analyzed frames both averages are null and the percentage is zero (count 0/0;
no score interpretation). With one or two analyzed frames, available scores are
returned descriptively, but `predicted_label` is null and status is
`insufficient_analysis`. A successful request has `success=true` even when
insufficient. At least three analyzed frames permits status `analyzed` and a
real/fake experimental label from the aggregate and existing threshold.

The typed response includes success, status, predicted_label, average_fake_score,
average_real_score, threshold, sampled_frames, successfully_analyzed_frames,
frames_without_faces, faces_analyzed, frames_above_threshold,
percentage_frames_above_threshold, decoding_failures, frame_level_results,
duration_seconds, processing_time_seconds, model_name, model_version, warning
and limitations. Each frame has index, timestamp/method, status, average fake
score, face results and a sanitized optional decoding error. Face results include
original bounding box, confidence, crop box, status, scores and predicted class.

The UI has Image Analysis / Video Analysis / About controls. Video supports
browse and drag/drop, size and browser-readable duration validation, preview
where its codec is supported, replacement/removal, genuine pending/error states,
aggregate results and expandable frame/face details. If the browser cannot
preview a codec, the server still validates duration before inference. No fake
progress or sample predictions are shipped.

## Reproduction and actual verification

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\smoke_video.py
.\.venv\Scripts\python.exe scripts\smoke_application.py
cd ..\frontend
npm test
npm run build
```

The smoke scripts require both services running. Optional `--backend` and
`--frontend` arguments select verification ports. `smoke_video.py` verifies the
saved split-manifest checksum, filters for a paired validation record **before**
resolving any media path, checks the source-video checksum, uploads via backend
and frontend proxy, checks per-face/per-frame aggregation and repeated exact
frame results, and verifies unchanged source/checkpoint plus temporary-directory
cleanup. No ground-truth label or manipulation mask is sent to inference. It
prints results to stdout without saving uploaded media. No test path is opened.

Actual verification on validation video **004_982**, duration 12.36 seconds:
12 unique sampled frames, 12 successfully analyzed frames, 12 faces, zero no-face
frames and zero decoding failures. Average model fake score **0.734908347328504**;
12/12 analyzed frames met the existing threshold. The experimental prediction
was `fake`; this single smoke is not an accuracy or video-performance metric.
The live browser displayed 73.5%, 12/12 frames and all 12 face details, the pending
state, supported video preview and uncertainty text. Desktop visual inspection
identified and corrected preview/file-detail spacing. The browser session used
localhost:5175 with a separate verification backend on port 8001, preserving the
previous server. Full responsive/mobile and every native codec preview combination
were not exhaustively tested.

Final checks: **155 backend tests passed**, **13 frontend tests passed**, and the
production TypeScript/Vite build passed. Backend fixtures are tiny generated
AVI/MP4/MOV/MKV files and controlled face/classifier outputs; real checkpoint
inference is checked separately. Frontend tests exercise validation, multipart
requests, response validation, errors/aborts, component result rendering and real
mode/upload/pending/result/error handlers through a small hook host using existing
Node/React/TypeScript dependencies. These tests do not replace browser rendering;
the real video flow was also observed in the browser. The existing Starlette/httpx
deprecation warning remains. On this Windows environment the full backend tests
need execution outside the sandbox to use external temporary directories required
by earlier dataset-output guards; source code safety guards were preserved.

Image regression independently reproduced the existing Phase 10 probabilities:
fake **0.7683594822883606** for validation crop `004_982:f92:d0`, with matching API
and proxy responses and exact repeated responses. Checkpoint SHA-256 remained
`5e19cc61d7b0c74b1f7979fb0682bc510dbc781c97d06f77945c50a25ab58622`.

## Limitations and resource behavior

Scores are uncalibrated model outputs, including the average across frames, and
are never proof of authenticity. This classifier was trained on a small matched
FaceForensics++ Deepfakes/c23 facial-crop subset, not video-level supervision.
Other methods, acquisition artifacts, generated content and unseen sources may
be outside its training distribution. Detector mistakes and bystanders affect
the result; no tracking, target attribution, audio or temporal classifier exists.
Brief manipulations between samples may be missed. No locked-test evaluation,
calibration, new performance metric, retraining or weight modification occurred.

CPU time grows with full-video decode, resolution and faces per frame. Initial
validation decodes the whole bounded clip; inference only uses sampled frames.
There is no hard subprocess timeout for native decoder/detector/model calls, no
public deployment hardening, authentication or multi-user queue. Keep this local,
use one worker, and prefer short 720p/1080p clips on the stated i5/16 GB machine.
Strict metadata/container checks intentionally reject some playable unusual files;
OpenCV cannot prove all malformed containers are safe or universally supported.
Temporary storage is OS-managed and deleted after processing, not cryptographic
erasure; OS/process termination can interrupt normal cleanup. Stop after Phase 12.
